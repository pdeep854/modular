"""Host validation for the W8A8 integer arm of the HMX matmul (`hmx_int8.mojo`).

TWO GATES, and the split is the point.

  1. EXACT. The stripe's integer arm, run through the same
     `matmul_bt_stripe_hmx_step` the fp16 arm runs through, must be BIT-IDENTICAL
     to a scalar reference that performs the same quantized arithmetic row-major
     — quantize, int32-accumulate, requant through the record, dequantize. The
     reference never touches a crouton, a ring slot or a step, so this gate
     constrains the layout, the schedule, the record encoding and the `/512`
     convert with no tolerance anywhere.

  2. BANDED. That quantized chain must then land within a PRE-REGISTERED
     tolerance of an fp32 reference matmul. This is the accuracy claim and the
     only place a tolerance appears; the band comes from per-channel int8 theory
     (see `BAND_LSB` / `BAND_PEAK_FRAC`), not from what the run happened to
     produce.

Plus three falsifying controls that MUST fire, one per way the arithmetic can be
silently wrong: a per-channel scale landing on the wrong channel, the activation
zero point left unfolded, and the requant record programmed AFTER the tile's
stores instead of before (the order-118 mutation).

WHAT THIS CANNOT PROVE. No integer HMX instruction has executed anywhere in this
tree. `HMX_EMIT_MMA` is off here, so what runs is `_hmx_int8_mma_model` — a model
of the data flow and the convert, whose composition of the guide's two statements
is an inference. `:cm`, the chunk boundary, the store's `:after` half and the
hardware's rounding mode are device rungs 2-7 of
`.agentwork/flux-npu/HMX_INT8_SCOPE.md` §4.2, and rung 1 is what tells us the
`/512` rule holds on our silicon at all.

Run from this directory (imports the sibling core):
  mojo run test_hmx_int8.mojo
"""
from std.math import align_up, round, sqrt
from std.memory import AddressSpace, alloc, bitcast

from layout.coord import Coord
from layout.tile_layout import row_major

from hexagon_ondevice.hmx_crouton import (
    HMX_DTYPE,
    HMX_EMIT_MMA,
    HMX_RECORD_LANES,
    HMX_TILE,
    HMX_UNPACK_ARM_SCALAR_OFF,
    HMX_XFER_COPY,
    HmxRows2D,
    hmx_k_tiles,
    hmx_stripe_act_slots,
    hmx_stripe_arena_bytes,
    hmx_stripe_arena_in,
)
from hexagon_ondevice.hmx_int8 import (
    HMX_INT8_CONVERT_DIV,
    HMX_INT8_DTYPE,
    HMX_INT8_IDENTITY_SCALE,
    HMX_INT8_MAX,
    HmxInt8Qp,
    hmx_int8_artefact_fits,
    hmx_int8_artefact_record,
    hmx_int8_header_offset,
    hmx_int8_quantize_act,
    hmx_int8_quantize_weight,
    hmx_int8_read_qp,
    hmx_int8_record_bias,
    hmx_int8_record_scale,
    hmx_int8_record_word,
    hmx_int8_records_offset,
    hmx_int8_repack_weight_cpm,
    hmx_int8_requant,
    hmx_int8_weight_scale,
)
from hexagon_ondevice.matmul_kernel import (
    MMTile,
    hmx_stripe_plan,
    matmul_bt_hmx_accepts,
    matmul_bt_stripe_hmx_step,
)

comptime GEN = AddressSpace.GENERIC

comptime F32 = DType.float32

comptime BAND_LSB = Scalar[F32](1)
"""Output LSBs of slack the band allows, on top of the peak-fraction term.

The readout is uint8, so a single output quantum is the floor of what any
correct W8A8 chain can achieve; half of it is the ideal round-off and one is the
allowance for the model's round-to-nearest disagreeing with a hardware mode that
is unpublished (rung 2)."""

comptime BAND_PEAK_FRAC = Scalar[F32](0.015)
"""Fraction of the output's peak magnitude the band allows for INPUT error.

Per-channel int8 theory: a weight quantum is `max_k|W[c,k]|/127` and an
activation quantum is `range(A)/255`, so each contributes a relative RMS near
`1/(sqrt(12) * 127) = 0.23%` per term. Over a `K`-long sum whose magnitude grows
like `sqrt(K)` the RELATIVE error does not grow with `K`, so a 1.5% ceiling is
several times theory and still far tighter than any wrong-scale, wrong-channel
or unfolded-zero-point defect, all of which the controls below put orders of
magnitude outside it."""

comptime BAND_RMS_FRAC = Scalar[F32](0.02)
"""Relative RMS error the band allows. The aggregate half of the same theory.

RE-DERIVED, NOT RELAXED, when the convert was corrected to truncate. The
convert's own error is one term of this, and truncation is a strictly worse
term than round-to-nearest: rounding leaves a residual uniform on
`[-1/2, +1/2)` LSB -- zero mean, RMS `1/sqrt(12) = 0.289` -- while truncation
leaves one uniform on `[-1, 0)` -- mean `-1/2`, and RMS
`sqrt(1/4 + 1/12) = 0.577`. That is exactly **2x**, and the bias half of it
does not average out over a tile the way the spread does.

So the ceiling doubles with the mechanism, `0.01 -> 0.02`, and the measured
values move the way the derivation predicts rather than by an unexplained
amount: `M=64 N=96 K=128` reads 0.0117 and `M=32 N=64 K=192` reads 0.0144
against a previous 0.0078, i.e. 1.5-1.85x, inside the 2x bound with the
remainder being the input-quantization terms that did not change.

THIS IS A REAL ACCURACY COST OF THE INTEGER ARM, not bookkeeping: the
per-family rollout gates on Tier-B PSNR >= 30 dB, and a systematic half-LSB
output bias is exactly the kind of term that shows up there. A future convert
that rounds -- if the hardware turns out to offer one -- should take this
constant back to 0.01 rather than keep the slack."""

comptime MUT_NONE = 0
"""No mutation: the artefact as the repacker built it."""

comptime MUT_SCALE_CHANNEL = 1
"""A per-channel scale landing on the WRONG channel."""

comptime MUT_ZP_UNFOLDED = 2
"""The activation zero point left out of the effective bias."""


def fill_coprime(t: HmxRows2D, rows: Int, cols: Int, a: Int, b: Int, m: Int):
    """Fills `t` with a period-coprime pattern centred on zero.

    Periods coprime with each other and with the 32-wide tile, for the reason
    recorded in `test_hmx_crouton.mojo`: a period sharing a factor with the tile
    can fold a transposed gather back onto the right answer and hide the
    highest-probability bug. Centred on zero so both zero points are exercised.
    """
    for r in range(rows):
        for c in range(cols):
            t.store[1](
                Coord(r, c), Scalar[HMX_DTYPE]((a * r + b * c) % m - m // 2)
            )


def fill_weight(W: HmxRows2D, N: Int, K: Int):
    """Fills a weight whose per-channel RANGES differ, not just its values.

    THE DEGENERATE-FILL TRAP, in its per-channel form. A plain coprime fill gives
    every output channel the same `max_k |W[c, k]|`, so every baked scale is the
    same number and the wrong-channel control cannot fire -- the record's whole
    per-channel machinery would be untested by a suite that looked complete.
    The row multiplier gives five distinct channel scales inside every 32-wide
    column tile, which is what makes a scale landing one lane out of step
    visible. Same family as the fp16 packers' coprime-period fill (KB
    `kernels/hmx-crouton-tile-layout`).
    """
    for n in range(N):
        for k in range(K):
            W.store[1](
                Coord(n, k),
                Scalar[HMX_DTYPE](((3 * n + 11 * k) % 7 - 3) * (1 + n % 5)),
            )


def reference_f32(
    C: HmxRows2D, A: HmxRows2D, W: HmxRows2D, M: Int, N: Int, K: Int
):
    """`C = A @ W^T` in fp32: the unquantized answer the band is measured from.
    """
    for m in range(M):
        for n in range(N):
            var acc = Scalar[F32](0)
            for k in range(K):
                acc += (
                    A.load[1](Coord(m, k)).cast[F32]()
                    * W.load[1](Coord(n, k)).cast[F32]()
                )
            C.store[1](Coord(m, n), acc.cast[HMX_DTYPE]())


@fieldwise_init
struct Int8Calib(Copyable, ImplicitlyCopyable, Movable):
    """One tensor's per-tensor quantization: a quantum and a zero point."""

    var scale: Scalar[F32]
    """The quantum."""

    var zp: Int
    """The zero point, `0..255`."""


def calibrate(t: HmxRows2D, rows: Int, cols: Int) -> Int8Calib:
    """An asymmetric per-tensor `(scale, zero_point)` covering `t`'s range.

    THE v1 SIMPLIFICATION, stated where it is made: one scale and one zero point
    for the whole tensor, computed offline from the tensor itself. A real
    calibration takes the range from a corpus of activations rather than from
    the tensor at hand, and a DiT's outliers across the timestep schedule are
    exactly what a per-tensor range absorbs badly (scope §3.4). This function is
    the seam that a calibration set replaces; it is not a calibration.
    """
    var lo = Scalar[F32](0)
    var hi = Scalar[F32](0)
    for r in range(rows):
        for c in range(cols):
            var v = t.load[1](Coord(r, c)).cast[F32]()
            if v < lo:
                lo = v
            if v > hi:
                hi = v
    var scale = (hi - lo) / Scalar[F32](255)
    if scale == 0:
        scale = Scalar[F32](1)
    var zp = Int(round(-lo / scale))
    if zp < 0:
        zp = 0
    if zp > 255:
        zp = 255
    return Int8Calib(scale, zp)


def reference_quantized(
    C: HmxRows2D,
    A: HmxRows2D,
    W: HmxRows2D,
    M: Int,
    N: Int,
    K: Int,
    qp: HmxInt8Qp,
    mutate: Int,
):
    """The same quantized arithmetic, row-major, touching no crouton.

    THE EXACT GATE'S REFERENCE. It quantizes with the module's own element-wise
    functions -- so a quantizer bug is shared and is the band gate's problem,
    not this one's -- and then does everything else independently: its own
    int32 accumulation over `k`, its own record built from
    `hmx_int8_record_scale` / `hmx_int8_record_bias`, its own requant, its own
    dequantization. What it deliberately does NOT share with the stripe is the
    crouton layout, the column-panel-major artefact order, the ring, the work
    partition and the step schedule, which is precisely the set the exact gate
    constrains.

    `mutate` reproduces a control's defect here as well, so a control measures
    the stripe against a correct reference rather than two broken things
    agreeing.
    """
    for n in range(N):
        var wscale = hmx_int8_weight_scale(W, n, K)
        var col_sum = 0
        for k in range(K):
            col_sum += _signed(
                hmx_int8_quantize_weight(W.load[1](Coord(n, k)), wscale)
            )
        var scale = hmx_int8_record_scale(qp, wscale)
        if mutate == MUT_SCALE_CHANNEL and n % HMX_TILE < 2:
            # The neighbouring channel's scale, which is what a record built
            # one lane out of step delivers.
            var other = n + 1 if n % HMX_TILE == 0 else n - 1
            scale = hmx_int8_record_scale(
                qp, hmx_int8_weight_scale(W, other, K)
            )
        var bias = hmx_int8_record_bias(qp, wscale, col_sum)
        if mutate == MUT_ZP_UNFOLDED:
            bias = hmx_int8_record_bias(qp, wscale, 0)
        var word = hmx_int8_record_word(scale, Scalar[HMX_DTYPE](0))
        for m in range(M):
            var acc = Int32(0)
            for k in range(K):
                acc += Int32(
                    Int(hmx_int8_quantize_act(A.load[1](Coord(m, k)), qp))
                ) * Int32(
                    _signed(
                        hmx_int8_quantize_weight(W.load[1](Coord(n, k)), wscale)
                    )
                )
            var q = Int(hmx_int8_requant(acc, word, bias))
            C.store[1](
                Coord(m, n),
                ((Scalar[F32](q) - Scalar[F32](qp.out_zp)) * qp.out_scale).cast[
                    HMX_DTYPE
                ](),
            )


def _signed(b: Scalar[DType.uint8]) -> Int:
    """A weight byte read the way `weight.b` reads it."""
    var v = Int(b)
    return v - 256 if v > HMX_INT8_MAX else v


def _pairlow_index(k: Int) -> Int:
    """The device's fitted wrong activation index, as an executable oracle.

    `lane/run-184F` legs D/E: within each 64-deep chained crouton pair, the
    plain int8 stream's second activation MAC re-reads the FIRST crouton's
    bytes — `effective_act_index(k) = 64*(k//64) + (k%32)` — bit-exact against
    the device at K=64/128/256, and the same fault order 146 originally named
    (`hmx_int8.mojo`'s pack docstring cites it). The redirect is the identity
    for K <= 32, which is why every single-crouton probe passed.
    """
    return 64 * (k // 64) + (k % 32)


def reference_pairlow(
    C: HmxRows2D,
    A: HmxRows2D,
    W: HmxRows2D,
    M: Int,
    N: Int,
    K: Int,
    qp: HmxInt8Qp,
):
    """`reference_quantized`'s MUST-NOT-MATCH sibling: the pair-low defect.

    Identical arithmetic except the activation is read through
    `_pairlow_index`; the weight side and the record stay correct, because
    leg C isolated the defect to the activation operand and 184F proved no
    record field was ever wrong. This is what a fixed stream is scored
    AGAINST: an output matching this reference IS the addressing bug,
    whatever the exact gate beside it says.
    """
    for n in range(N):
        var wscale = hmx_int8_weight_scale(W, n, K)
        var col_sum = 0
        for k in range(K):
            col_sum += _signed(
                hmx_int8_quantize_weight(W.load[1](Coord(n, k)), wscale)
            )
        var word = hmx_int8_record_word(
            hmx_int8_record_scale(qp, wscale), Scalar[HMX_DTYPE](0)
        )
        var bias = hmx_int8_record_bias(qp, wscale, col_sum)
        for m in range(M):
            var acc = Int32(0)
            for k in range(K):
                acc += Int32(
                    Int(
                        hmx_int8_quantize_act(
                            A.load[1](Coord(m, _pairlow_index(k))), qp
                        )
                    )
                ) * Int32(
                    _signed(
                        hmx_int8_quantize_weight(W.load[1](Coord(n, k)), wscale)
                    )
                )
            var q = Int(hmx_int8_requant(acc, word, bias))
            C.store[1](
                Coord(m, n),
                ((Scalar[F32](q) - Scalar[F32](qp.out_zp)) * qp.out_scale).cast[
                    HMX_DTYPE
                ](),
            )


def mutate_artefact(
    art: HmxRows2D, W: HmxRows2D, N: Int, K: Int, qp: HmxInt8Qp, mutate: Int
):
    """Rewrites the baked records with a named defect. Nothing else moves.

    Both defects are invisible to every gate the loader has: the file length,
    the layout tag and the census all still agree, and the only symptom is wrong
    numbers. That is the whole reason they are controls.
    """
    if mutate == MUT_NONE:
        return
    for ct in range(N // HMX_TILE):
        var record = hmx_int8_artefact_record(art, ct)
        for lane in range(HMX_RECORD_LANES):
            var channel = ct * HMX_TILE + lane
            var wscale = hmx_int8_weight_scale(W, channel, K)
            if mutate == MUT_SCALE_CHANNEL:
                if lane > 1:
                    continue
                var other = channel + 1 if lane == 0 else channel - 1
                record.store[1](
                    Coord(0, lane),
                    hmx_int8_record_word(
                        hmx_int8_record_scale(
                            qp, hmx_int8_weight_scale(W, other, K)
                        ),
                        Scalar[HMX_DTYPE](0),
                    ),
                )
            else:
                record.store[1](
                    Coord(1, lane),
                    bitcast_u32(hmx_int8_record_bias(qp, wscale, 0)),
                )


def bitcast_u32(v: Int32) -> Scalar[DType.uint32]:
    """The record's int32 half, as the word the layout stores."""
    return bitcast[DType.uint32, 1](v)


@fieldwise_init
struct Int8Result(Copyable, ImplicitlyCopyable, Movable):
    """What one W8A8 stripe run is scored on."""

    var exact_wrong: Int
    """Elements differing from the quantized reference. MUST be 0."""

    var max_err: Scalar[F32]
    """Largest absolute deviation from the fp32 reference."""

    var rms_rel: Scalar[F32]
    """RMS deviation from the fp32 reference, relative to its RMS."""

    var peak: Scalar[F32]
    """Largest absolute value of the fp32 reference."""

    var out_scale: Scalar[F32]
    """The output quantum, which sets the floor of `max_err`."""


def int8_stripe_run[
    record_late: Bool = False
](
    M: Int, N: Int, K: Int, act_slots: Int, feeders: Int, mutate: Int
) -> Int8Result:
    """Runs the integer arm end to end and scores it against both references.

    The stripe driver is `test_hmx_crouton.mojo`'s, unchanged in shape: the
    array thread and the feeders are a loop over one `matmul_bt_stripe_hmx_step`
    per participant per step. That is deliberate -- if the integer arm needed a
    different driver it would not be the same body.
    """
    var k_tiles = hmx_k_tiles(K)
    var a_buf = alloc[Scalar[HMX_DTYPE]](M * K)
    var w_buf = alloc[Scalar[HMX_DTYPE]](N * K)
    var art_buf = alloc[Scalar[HMX_DTYPE]](N * K)
    var f32_buf = alloc[Scalar[HMX_DTYPE]](M * N)
    var qref_buf = alloc[Scalar[HMX_DTYPE]](M * N)
    var out_buf = alloc[Scalar[HMX_DTYPE]](M * N)

    var A = HmxRows2D(
        ptr=a_buf.as_unsafe_any_origin(), layout=row_major(Coord(M, K))
    )
    var W = HmxRows2D(
        ptr=w_buf.as_unsafe_any_origin(), layout=row_major(Coord(N, K))
    )
    var art = HmxRows2D(
        ptr=art_buf.as_unsafe_any_origin(), layout=row_major(Coord(N, K))
    )
    var Cf32 = HmxRows2D(
        ptr=f32_buf.as_unsafe_any_origin(), layout=row_major(Coord(M, N))
    )
    var Cq = HmxRows2D(
        ptr=qref_buf.as_unsafe_any_origin(), layout=row_major(Coord(M, N))
    )

    fill_coprime(A, M, K, 5, 7, 13)
    fill_weight(W, N, K)
    for i in range(N * K):
        art_buf[i] = Scalar[HMX_DTYPE](0)
    for i in range(M * N):
        out_buf[i] = Scalar[HMX_DTYPE](0)

    reference_f32(Cf32, A, W, M, N, K)
    var act = calibrate(A, M, K)
    var out = calibrate(Cf32, M, N)
    var qp = HmxInt8Qp(act.scale, act.zp, out.scale, out.zp)

    hmx_int8_repack_weight_cpm(art, W, N, K, qp)
    mutate_artefact(art, W, N, K, qp, mutate)
    reference_quantized(Cq, A, W, M, N, K, qp, mutate)

    var Amm = MMTile(
        ptr=a_buf.as_unsafe_any_origin(), layout=row_major(Coord(M, K))
    )
    var Bmm = MMTile(
        ptr=art_buf.as_unsafe_any_origin(), layout=row_major(Coord(N, K))
    )
    var Cmm = MMTile(
        ptr=out_buf.as_unsafe_any_origin(), layout=row_major(Coord(M, N))
    )

    var arena_bytes = hmx_stripe_arena_bytes[HMX_INT8_DTYPE](k_tiles, act_slots)
    var raw = alloc[UInt8](arena_bytes + 4096)
    var base = UnsafePointer[UInt8, MutUntrackedOrigin](
        unsafe_from_address=align_up(Int(raw), 4096)
    )
    var plan = hmx_stripe_plan[HMX_INT8_DTYPE](M, N, K, arena_bytes)
    var arena = hmx_stripe_arena_in[GEN, HMX_INT8_DTYPE](
        base, plan.k_tiles, plan.act_slots
    )

    # `HMX_XFER_COPY` because the host has no DMA engine; the transport's own
    # gates live in `test_hmx_crouton.mojo` and are dtype-generic, so they cover
    # this arm's staging too.
    comptime step_fn = matmul_bt_stripe_hmx_step[
        HMX_EMIT_MMA, True, HMX_XFER_COPY, True, HMX_INT8_DTYPE, record_late
    ]
    var unpack_arm = HMX_UNPACK_ARM_SCALAR_OFF
    for step in range(plan.steps()):
        for t in range(feeders):
            step_fn(Cmm, Amm, Bmm, arena, plan, step, t, feeders, unpack_arm)
        step_fn(Cmm, Amm, Bmm, arena, plan, step, feeders, feeders, unpack_arm)

    var exact_wrong = 0
    var max_err = Scalar[F32](0)
    var peak = Scalar[F32](0)
    var sq_err = Scalar[F32](0)
    var sq_ref = Scalar[F32](0)
    for m in range(M):
        for n in range(N):
            var got = Cmm.load[1](Coord(m, n))
            if got != Cq.load[1](Coord(m, n)):
                exact_wrong += 1
            var want = Cf32.load[1](Coord(m, n)).cast[F32]()
            var err = got.cast[F32]() - want
            if err < 0:
                err = -err
            if err > max_err:
                max_err = err
            var mag = want if want >= 0 else -want
            if mag > peak:
                peak = mag
            sq_err += err * err
            sq_ref += want * want

    var rms_rel = Scalar[F32](0)
    if sq_ref > 0:
        rms_rel = sqrt(sq_err / sq_ref)

    a_buf.free()
    w_buf.free()
    art_buf.free()
    f32_buf.free()
    qref_buf.free()
    out_buf.free()
    raw.free()
    return Int8Result(exact_wrong, max_err, rms_rel, peak, qp.out_scale)


def check_convert_identity() -> Bool:
    """Asserts the `/512` convert, and reproduces the all-zero autopsy directly.

    Two claims from scope §2 H1, both arithmetic and both checkable here:

    - the INTEGER identity is 512.0, and under it the readout is the raw
      accumulator (which is what rung 2 expects back from silicon);
    - the FP16 identity 1.0 with `acc <= 255` -- what every earlier probe
      programmed and sized for -- truncates to zero at every position, so that
      probe's PASS predicate was unsatisfiable before any hardware ran.
    """
    var integer = hmx_int8_record_word(
        HMX_INT8_IDENTITY_SCALE, Scalar[HMX_DTYPE](0)
    )
    var fp16 = hmx_int8_record_word(Scalar[HMX_DTYPE](1), Scalar[HMX_DTYPE](0))
    var bad = 0
    var zeroed = 0
    for acc in range(0, 256):
        if Int(hmx_int8_requant(Int32(acc), integer, 0)) != acc:
            bad += 1
        if Int(hmx_int8_requant(Int32(acc), fp16, 0)) == 0:
            zeroed += 1
    # Saturation is the store's, not the record's: past 255 the byte pins.
    if Int(hmx_int8_requant(Int32(4096), integer, 0)) != 255:
        bad += 1
    if bad != 0 or zeroed != 256:
        print(
            "  FAIL convert identity -",
            bad,
            "raw mismatches,",
            256 - zeroed,
            "fp16-identity positions were nonzero",
        )
        return False
    print(
        "  ok   convert identity: scale",
        HMX_INT8_CONVERT_DIV,
        ".0 is raw, scale 1.0 zeroes every acc <= 255",
    )
    return True


def check_artefact_layout(N: Int, K: Int) -> Bool:
    """Asserts the artefact's three regions, and the length invariant.

    The length is the property M4.3 bought and this carries forward: the file is
    the fp16 weight's byte count, so the manifest, the sha256 gate, the census
    and the graph's constant binding all still agree and only the layout tag
    says what the bytes mean.
    """
    var ok = True
    if not hmx_int8_artefact_fits(N, K):
        print("  FAIL artefact does not fit N=", N, "K=", K)
        return False
    if hmx_int8_records_offset(N, K) != N * K:
        print("  FAIL record region does not follow the croutons")
        ok = False
    if hmx_int8_header_offset(N, K) != N * K + (N // HMX_TILE) * 256:
        print("  FAIL header does not follow the records")
        ok = False

    var buf = alloc[Scalar[HMX_DTYPE]](N * K)
    var w_buf = alloc[Scalar[HMX_DTYPE]](N * K)
    var art = HmxRows2D(
        ptr=buf.as_unsafe_any_origin(), layout=row_major(Coord(N, K))
    )
    var W = HmxRows2D(
        ptr=w_buf.as_unsafe_any_origin(), layout=row_major(Coord(N, K))
    )
    for i in range(N * K):
        buf[i] = Scalar[HMX_DTYPE](0)
    fill_weight(W, N, K)
    var qp = HmxInt8Qp(Scalar[F32](0.05), 128, Scalar[F32](0.25), 128)
    hmx_int8_repack_weight_cpm(art, W, N, K, qp)

    var back = hmx_int8_read_qp(art)
    if (
        back.act_scale != qp.act_scale
        or back.act_zp != qp.act_zp
        or back.out_scale != qp.out_scale
        or back.out_zp != qp.out_zp
    ):
        print("  FAIL the per-tensor header did not round trip")
        ok = False

    # Every channel's baked scale must be the one its own weight range implies.
    var off = 0
    for ct in range(N // HMX_TILE):
        var record = hmx_int8_artefact_record(art, ct)
        for lane in range(HMX_RECORD_LANES):
            var want = hmx_int8_record_word(
                hmx_int8_record_scale(
                    qp, hmx_int8_weight_scale(W, ct * HMX_TILE + lane, K)
                ),
                Scalar[HMX_DTYPE](0),
            )
            if record.load[1](Coord(0, lane)) != want:
                off += 1
    if off != 0:
        print("  FAIL", off, "baked scales are not their channel's")
        ok = False

    buf.free()
    w_buf.free()
    if ok:
        print(
            "  ok   artefact N=",
            N,
            "K=",
            K,
            "- croutons, records, header inside",
            2 * N * K,
            "bytes",
        )
    return ok


def check_arm_geometry(k_tiles: Int, budget: Int) -> Bool:
    """Asserts the integer arm's residency is HALF the fp16 arm's.

    The bandwidth case for int8 in the one place this PR can check it: the same
    VTCM budget holds roughly twice the activation panels, which is what deeper
    A-blocking and fewer weight re-reads are bought with (scope §3.2). The array
    throughput claim is a device measurement and is NOT checked here.
    """
    var fp16 = hmx_stripe_arena_bytes[HMX_DTYPE](k_tiles, 4)
    var int8 = hmx_stripe_arena_bytes[HMX_INT8_DTYPE](k_tiles, 4)
    var fp16_slots = hmx_stripe_act_slots[HMX_DTYPE](k_tiles, budget, 4096)
    var int8_slots = hmx_stripe_act_slots[HMX_INT8_DTYPE](k_tiles, budget, 4096)
    # The record is a fixed 256 bytes in both, so the panels halve exactly and
    # the totals differ by it.
    if int8 * 2 - fp16 != 256:
        print("  FAIL arena bytes", fp16, "vs", int8, "are not a halving")
        return False
    if int8_slots < 2 * fp16_slots:
        print("  FAIL act slots", fp16_slots, "->", int8_slots)
        return False
    print(
        "  ok   arena k_tiles=",
        k_tiles,
        ":",
        fp16,
        "->",
        int8,
        "bytes, slots",
        fp16_slots,
        "->",
        int8_slots,
    )
    return True


def check_artefact_fits_is_implied() -> Bool:
    """Asserts the artefact tail fits for EVERY geometry the array accepts.

    Worth an assertion rather than a comment because it is what keeps the arms'
    shape predicate identical: the records and header ride the same file, so if
    the tail could overrun, `matmul_bt_hmx_accepts` would need an arm parameter
    and the kernel a second refusal. It cannot -- the tail is `8N + 16` bytes
    against `N * K` free ones and the array already bounds `K` at `MM_W` -- so
    `hmx_int8_artefact_fits` is the REPACKER's refusal, where `N` and `K` come
    from a manifest rather than from a predicate that already checked them.

    The degenerate case is asserted too: a `K` below the array's own bound is
    where it would fail, which is what makes the refusal a real function.
    """
    var ok = True
    for n_tiles in [1, 4, 32, 256]:
        var N = n_tiles * HMX_TILE
        for K in [64, 128, 3072]:
            if not matmul_bt_hmx_accepts(K, N, K, 0, N):
                continue
            if not hmx_int8_artefact_fits(N, K):
                print("  FAIL tail overruns at accepted N=", N, "K=", K)
                ok = False
    if hmx_int8_artefact_fits(8192, 8):
        print("  FAIL the repacker refusal admits K=8, which the tail overruns")
        ok = False
    if ok:
        print("  ok   artefact tail fits at every geometry the array accepts")
    return ok


def check_stripe(M: Int, N: Int, K: Int, act_slots: Int, feeders: Int) -> Bool:
    """THE PR'S GATE: exact against the quantized reference, banded against fp32.
    """
    var r = int8_stripe_run(M, N, K, act_slots, feeders, MUT_NONE)
    var band = BAND_LSB * r.out_scale + BAND_PEAK_FRAC * r.peak
    var ok = True
    if r.exact_wrong != 0:
        print(
            "  FAIL stripe M=",
            M,
            "N=",
            N,
            "K=",
            K,
            "-",
            r.exact_wrong,
            "elements differ from the quantized reference",
        )
        ok = False
    if r.max_err > band:
        print(
            "  FAIL band M=",
            M,
            "N=",
            N,
            "K=",
            K,
            "- max err",
            r.max_err,
            "over",
            band,
        )
        ok = False
    if r.rms_rel > BAND_RMS_FRAC:
        print("  FAIL rms M=", M, "N=", N, "K=", K, "-", r.rms_rel)
        ok = False
    if ok:
        print(
            "  ok   stripe M=",
            M,
            "N=",
            N,
            "K=",
            K,
            "slots=",
            act_slots,
            "exact; max err",
            r.max_err,
            "of",
            band,
            "band, rms",
            r.rms_rel,
        )
    return ok


def oracle_mutation(M: Int, N: Int, K: Int, mutate: Int, label: String) -> Bool:
    """MUST fire: a named defect in the baked record moves the answer.

    Scored against the fp32 reference rather than the quantized one, because the
    quantized reference carries the same defect on purpose -- so what this
    asserts is that the defect is VISIBLE at all, which is the property that
    makes the exact gate above meaningful.
    """
    var clean = int8_stripe_run(M, N, K, 2, 4, MUT_NONE)
    var broken = int8_stripe_run(M, N, K, 2, 4, mutate)
    var band = BAND_LSB * clean.out_scale + BAND_PEAK_FRAC * clean.peak
    if broken.max_err <= band:
        print("  FAIL oracle", label, "did NOT fire - max err", broken.max_err)
        return False
    print(
        "  ok   oracle",
        label,
        "fired - max err",
        broken.max_err,
        "against a band of",
        band,
    )
    return True


def oracle_record_late(M: Int, N: Int, K: Int) -> Bool:
    """MUST fire: the requant record programmed AFTER the tile's stores.

    THE ORDER-118 CONTROL. On silicon a store issued before the requant file is
    programmed reads back zeros from a MAC whose operands were streaming
    correctly -- observed for the fp16 arm, and the shape of the failure the
    int8 probe spent three device windows on. In the twin the same reordering is
    deterministic: the tile requants through whatever the PREVIOUS column tile
    left in the record, so the per-channel scales land a whole tile out of step.

    If this ever stops firing, `hmx_int8_load_record`'s position in the step has
    stopped being load-bearing and the rule is a comment again.
    """
    var wrong = int8_stripe_run[record_late=True](M, N, K, 2, 4, MUT_NONE)
    if wrong.exact_wrong == 0:
        print("  FAIL oracle record-late did NOT fire")
        return False
    print("  ok   oracle record-late fired -", wrong.exact_wrong, "differ")
    return True


def _pairlow_disagreement(M: Int, N: Int, K: Int, uniform_act: Bool) -> Int:
    """Elements on which the correct and pair-low references disagree.

    Corpus and calibration are `int8_stripe_run`'s own, so a disagreement
    count here speaks for the same fill the stripe gates run on. With
    `uniform_act` the activation is the record-ramp's constant fill instead
    — the one every pre-184F "identity is bit-exact" claim rested on.
    """
    var a_buf = alloc[Scalar[HMX_DTYPE]](M * K)
    var w_buf = alloc[Scalar[HMX_DTYPE]](N * K)
    var f32_buf = alloc[Scalar[HMX_DTYPE]](M * N)
    var q_buf = alloc[Scalar[HMX_DTYPE]](M * N)
    var p_buf = alloc[Scalar[HMX_DTYPE]](M * N)

    var A = HmxRows2D(
        ptr=a_buf.as_unsafe_any_origin(), layout=row_major(Coord(M, K))
    )
    var W = HmxRows2D(
        ptr=w_buf.as_unsafe_any_origin(), layout=row_major(Coord(N, K))
    )
    var Cf32 = HmxRows2D(
        ptr=f32_buf.as_unsafe_any_origin(), layout=row_major(Coord(M, N))
    )
    var Cq = HmxRows2D(
        ptr=q_buf.as_unsafe_any_origin(), layout=row_major(Coord(M, N))
    )
    var Cp = HmxRows2D(
        ptr=p_buf.as_unsafe_any_origin(), layout=row_major(Coord(M, N))
    )

    if uniform_act:
        for m in range(M):
            for k in range(K):
                A.store[1](Coord(m, k), Scalar[HMX_DTYPE](1))
    else:
        fill_coprime(A, M, K, 5, 7, 13)
    fill_weight(W, N, K)

    reference_f32(Cf32, A, W, M, N, K)
    var act = calibrate(A, M, K)
    var out = calibrate(Cf32, M, N)
    var qp = HmxInt8Qp(act.scale, act.zp, out.scale, out.zp)

    reference_quantized(Cq, A, W, M, N, K, qp, MUT_NONE)
    reference_pairlow(Cp, A, W, M, N, K, qp)

    var differ = 0
    for m in range(M):
        for n in range(N):
            if Cq.load[1](Coord(m, n)) != Cp.load[1](Coord(m, n)):
                differ += 1

    a_buf.free()
    w_buf.free()
    f32_buf.free()
    q_buf.free()
    p_buf.free()
    return differ


def check_pairlow_discriminates(M: Int, N: Int, K: Int) -> Bool:
    """The must-not-match oracle must be able to fire on this corpus.

    A must-not-match gate is vacuously green whenever its wrong-reference
    happens to equal the right one, so the oracle pair is itself gated: on
    the stripe corpus at a paired depth the two references must disagree on
    at least a quarter of the outputs (the device measured ~93% at K=64).
    """
    var differ = _pairlow_disagreement(M, N, K, False)
    if differ * 4 < M * N:
        print(
            "  FAIL pairlow oracle cannot fire at K=",
            K,
            "-",
            differ,
            "of",
            M * N,
            "differ",
        )
        return False
    print(
        "  ok   pairlow oracle discriminates K=",
        K,
        "-",
        differ,
        "of",
        M * N,
        "differ",
    )
    return True


def check_pairlow_needs_paired_depth(M: Int, N: Int) -> Bool:
    """MUST agree at K=32: below one crouton pair the redirect is the identity.

    The model's own structure, asserted so the oracle stays honest about its
    scope: a probe at K <= 32 can never see this defect, which is exactly how
    174C's small case passed while production K failed.
    """
    var differ = _pairlow_disagreement(M, N, 32, False)
    if differ != 0:
        print("  FAIL pairlow model is not the identity at K=32 -", differ)
        return False
    print("  ok   pairlow model is the identity at K=32")
    return True


def check_pairlow_blind_on_uniform(M: Int, N: Int, K: Int) -> Bool:
    """MUST agree: a uniform activation fill cannot see the pair-low defect.

    The record-ramp lesson (A=1 everywhere) as a standing assertion — a
    redirect among equal values changes nothing, so wrong-model == correct
    EXACTLY on this fill. Together with `check_pairlow_discriminates` this
    is the mutation-at-birth pair for the oracle itself: one leg proves the
    corpus can see the bug, the other proves the seeing comes from the
    corpus and not from an accident of the model code.
    """
    var differ = _pairlow_disagreement(M, N, K, True)
    if differ != 0:
        print("  FAIL pairlow oracle fired on a blind fill -", differ)
        return False
    print("  ok   pairlow oracle is blind on the uniform fill, as derived")
    return True


def check_stripe_refutes_pairlow(
    M: Int, N: Int, K: Int, act_slots: Int, feeders: Int
) -> Bool:
    """The twin must match the correct reference AND refuse the fitted defect.

    This is the host half of the fix-acceptance gate: exact against
    `reference_quantized` (already `check_stripe`'s job, re-asserted here on
    the same run) and disagreeing with `reference_pairlow` on at least a
    quarter of the outputs. On silicon the same oracle pair scores the
    device dump — a dump matching pairlow IS the order-146/184F addressing
    bug, bit for bit, whatever else passes beside it.
    """
    var k_tiles = hmx_k_tiles(K)
    var a_buf = alloc[Scalar[HMX_DTYPE]](M * K)
    var w_buf = alloc[Scalar[HMX_DTYPE]](N * K)
    var art_buf = alloc[Scalar[HMX_DTYPE]](N * K)
    var f32_buf = alloc[Scalar[HMX_DTYPE]](M * N)
    var q_buf = alloc[Scalar[HMX_DTYPE]](M * N)
    var p_buf = alloc[Scalar[HMX_DTYPE]](M * N)
    var out_buf = alloc[Scalar[HMX_DTYPE]](M * N)

    var A = HmxRows2D(
        ptr=a_buf.as_unsafe_any_origin(), layout=row_major(Coord(M, K))
    )
    var W = HmxRows2D(
        ptr=w_buf.as_unsafe_any_origin(), layout=row_major(Coord(N, K))
    )
    var art = HmxRows2D(
        ptr=art_buf.as_unsafe_any_origin(), layout=row_major(Coord(N, K))
    )
    var Cf32 = HmxRows2D(
        ptr=f32_buf.as_unsafe_any_origin(), layout=row_major(Coord(M, N))
    )
    var Cq = HmxRows2D(
        ptr=q_buf.as_unsafe_any_origin(), layout=row_major(Coord(M, N))
    )
    var Cp = HmxRows2D(
        ptr=p_buf.as_unsafe_any_origin(), layout=row_major(Coord(M, N))
    )

    fill_coprime(A, M, K, 5, 7, 13)
    fill_weight(W, N, K)
    for i in range(N * K):
        art_buf[i] = Scalar[HMX_DTYPE](0)
    for i in range(M * N):
        out_buf[i] = Scalar[HMX_DTYPE](0)

    reference_f32(Cf32, A, W, M, N, K)
    var act = calibrate(A, M, K)
    var out = calibrate(Cf32, M, N)
    var qp = HmxInt8Qp(act.scale, act.zp, out.scale, out.zp)

    hmx_int8_repack_weight_cpm(art, W, N, K, qp)
    reference_quantized(Cq, A, W, M, N, K, qp, MUT_NONE)
    reference_pairlow(Cp, A, W, M, N, K, qp)

    var Amm = MMTile(
        ptr=a_buf.as_unsafe_any_origin(), layout=row_major(Coord(M, K))
    )
    var Bmm = MMTile(
        ptr=art_buf.as_unsafe_any_origin(), layout=row_major(Coord(N, K))
    )
    var Cmm = MMTile(
        ptr=out_buf.as_unsafe_any_origin(), layout=row_major(Coord(M, N))
    )

    var arena_bytes = hmx_stripe_arena_bytes[HMX_INT8_DTYPE](k_tiles, act_slots)
    var raw = alloc[UInt8](arena_bytes + 4096)
    var base = UnsafePointer[UInt8, MutUntrackedOrigin](
        unsafe_from_address=align_up(Int(raw), 4096)
    )
    var plan = hmx_stripe_plan[HMX_INT8_DTYPE](M, N, K, arena_bytes)
    var arena = hmx_stripe_arena_in[GEN, HMX_INT8_DTYPE](
        base, plan.k_tiles, plan.act_slots
    )

    comptime step_fn = matmul_bt_stripe_hmx_step[
        HMX_EMIT_MMA, True, HMX_XFER_COPY, True, HMX_INT8_DTYPE, False
    ]
    var unpack_arm = HMX_UNPACK_ARM_SCALAR_OFF
    for step in range(plan.steps()):
        for t in range(feeders):
            step_fn(Cmm, Amm, Bmm, arena, plan, step, t, feeders, unpack_arm)
        step_fn(Cmm, Amm, Bmm, arena, plan, step, feeders, feeders, unpack_arm)

    var wrong = 0
    var refuted = 0
    for m in range(M):
        for n in range(N):
            var got = Cmm.load[1](Coord(m, n))
            if got != Cq.load[1](Coord(m, n)):
                wrong += 1
            if got != Cp.load[1](Coord(m, n)):
                refuted += 1

    a_buf.free()
    w_buf.free()
    art_buf.free()
    f32_buf.free()
    q_buf.free()
    p_buf.free()
    raw.free()

    if wrong != 0:
        print("  FAIL stripe K=", K, "-", wrong, "differ from the reference")
        return False
    if refuted * 4 < M * N:
        print(
            "  FAIL stripe K=",
            K,
            "MATCHES the pair-low defect on",
            M * N - refuted,
            "of",
            M * N,
        )
        return False
    print(
        "  ok   stripe K=",
        K,
        "exact, and refutes pair-low on",
        refuted,
        "of",
        M * N,
    )
    return True


def main() raises:
    print("==== HMX W8A8 integer arm (tile", HMX_TILE, "uint8 x int8) ====")
    var failures = 0

    if not check_convert_identity():
        failures += 1

    for k_tiles in [2, 4, 12]:
        if not check_arm_geometry(k_tiles, 1 << 20):
            failures += 1

    if not check_artefact_fits_is_implied():
        failures += 1

    # `K` a multiple of 64 because the microtile seam carries the depth as
    # 64-element vectors; `N` several column tiles so the per-channel records
    # differ from tile to tile, which a single-tile shape cannot show.
    var art_n = [96, 64, 128]
    var art_k = [128, 192, 256]
    for i in range(len(art_n)):
        if not check_artefact_layout(art_n[i], art_k[i]):
            failures += 1

    # `act_slots = 1` forces one A-block per row tile, which is the multi-block
    # path; `2` covers several row tiles in one block. Both must be exact.
    for slots in [1, 2]:
        if not check_stripe(64, 96, 128, slots, 4):
            failures += 1
    if not check_stripe(32, 64, 192, 1, 4):
        failures += 1
    # Fewer feeders is order 084's graceful-degradation path: the partition
    # changes, the arithmetic does not.
    if not check_stripe(64, 96, 128, 2, 2):
        failures += 1

    if not oracle_mutation(64, 96, 128, MUT_SCALE_CHANNEL, "scale-channel"):
        failures += 1
    if not oracle_mutation(64, 96, 128, MUT_ZP_UNFOLDED, "zp-unfolded"):
        failures += 1
    if not oracle_record_late(64, 96, 128):
        failures += 1

    # The pair-low addressing oracle (order 146 / lane/run-184F): the fitted
    # wrong-model at its three device-verified depths, its blindness controls,
    # and the twin scored against it. The device gate applies the same pair.
    for depth in [64, 128, 256]:
        if not check_pairlow_discriminates(64, 96, depth):
            failures += 1
    if not check_pairlow_needs_paired_depth(32, 64):
        failures += 1
    if not check_pairlow_blind_on_uniform(64, 96, 128):
        failures += 1
    if not check_stripe_refutes_pairlow(64, 96, 128, 2, 4):
        failures += 1
    if not check_stripe_refutes_pairlow(32, 64, 256, 1, 4):
        failures += 1

    if failures != 0:
        raise Error(String(failures), " check(s) failed")
    print("==== all W8A8 checks passed ====")
