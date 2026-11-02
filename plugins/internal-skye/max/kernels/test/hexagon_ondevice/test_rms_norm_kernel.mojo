"""Off-device algorithm test for the HVX RMS norm kernel.

Runs the kernel's own `_rms_norm` (which drives `RMSNorm2Pass`) on the host and
compares against an independent float64 reference. This proves the ALGORITHM —
the sum-of-squares reduction, the eps guard, the gamma scaling and its weight
offset, both sides of the `multiply_before_cast` split, the tail handling, the
row split across launch threads — is numerically correct. It says nothing about
whether the device emits vector instructions for it: the SIMD width resolves the
same on host and device, so a scalar host build would pass this too.
Vectorization is settled separately, by disassembling the v81 object (build.sh's
HVX-mnemonic count) and ultimately on silicon.

Until the megakernel extraction this kernel had no twin at all — it was authored
in `npu_kernel.mojo` and shipped from there, so nothing off-device scored it.
That is the gap this file closes, and it closes it over the shipped body: the
`hexagon_ondevice.rms_norm_kernel` imported here is the one the skel compiles.

The reference accumulates in float64 where the kernel accumulates in reassociated
float32, so the two disagree by ~1e-7 before the down-cast — far below the fp16 /
bf16 output resolution, which is why the tolerance is set at that resolution and
the reported max errors sit well inside it.
"""

from hexagon_ondevice.rms_norm_kernel import (
    RMS_ROPE_MAGIC,
    RMS_ROPE_MAX_HEAD,
    RMSNorm2Pass,
    RowMajor2D,
    _rms_norm,
    _rms_norm_rope,
    _rms_rope_header,
    rms_row,
)
from hexagon_ondevice.rope_kernel import (
    RowMajor1D as RopeRowMajor1D,
    RowMajor2D as RopeRowMajor2D,
    rope_striped,
)
from layout.coord import Coord
from layout.tile_layout import row_major

from std.math import isfinite, sqrt
from std.memory import alloc

comptime SENTINEL: Float32 = 30000.0
"""A finite value normalized output never takes, pre-written to every output
element so an unwritten (missing) one is detectable after the launch."""


def _fill_seeded[
    dtype: DType
](
    ptr: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin],
    n: Int,
    seed: UInt32,
    scale: Float32,
):
    """Deterministic pseudo-random fill in roughly [-scale, +scale]."""
    var state = seed | 1
    for i in range(n):
        state = state * 1664525 + 1013904223
        var u = Float32(Int(state >> 8) & 0xFFFF) / Float32(0xFFFF)
        ptr.store(i, ((u * 2.0 - 1.0) * scale).cast[dtype]())


def _fill_ramp[
    dtype: DType
](
    ptr: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin],
    num_rows: Int,
    length: Int,
):
    """A linear ramp across each row, offset per row — smoothly varying, so the
    row magnitudes differ and each row gets a different scale factor."""
    for r in range(num_rows):
        for i in range(length):
            var t = Float32(i) / Float32(length)
            ptr.store(
                r * length + i,
                ((t * 4.0 - 2.0) + Float32(r) * 0.5).cast[dtype](),
            )


def _fill_const[
    dtype: DType
](ptr: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin], n: Int, c: Float32):
    for i in range(n):
        ptr.store(i, c.cast[dtype]())


def _fill_sentinel[
    dtype: DType
](ptr: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin], n: Int):
    for i in range(n):
        ptr.store(i, SENTINEL.cast[dtype]())


def _ref_rms_norm[
    dtype: DType,
    multiply_before_cast: Bool,
](
    out_ref: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin],
    input: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin],
    gamma: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin],
    num_rows: Int,
    length: Int,
    eps: Float32,
    weight_offset: Float32,
):
    """Independent float64 reference: mean square, then scale by gamma.

    Reads the same stored dtype inputs the kernel reads. The reduction runs in
    float64 throughout; what it does mirror exactly is the ORDER of the cast in
    the `multiply_before_cast=False` arm, where the kernel narrows the normalized
    value to `dtype` before multiplying by gamma. That narrowing is a real part
    of the result, not an accumulation difference, so a reference that skipped it
    would be scoring a different kernel.
    """
    for r in range(num_rows):
        var base = r * length

        var s = Float64(0)
        for i in range(length):
            var x = Float64(input.load(base + i).cast[DType.float32]())
            s += x * x
        var inv_rms = Float64(1) / sqrt(s / Float64(length) + Float64(eps))

        for i in range(length):
            var x = Float64(input.load(base + i).cast[DType.float32]())
            var g = Float64(gamma.load(i).cast[DType.float32]())
            comptime if multiply_before_cast:
                out_ref.store(
                    base + i,
                    (x * inv_rms * (g + Float64(weight_offset))).cast[dtype](),
                )
            else:
                var normed = (x * inv_rms).cast[dtype]()
                var gsum = (Float64(g) + Float64(weight_offset)).cast[dtype]()
                out_ref.store(base + i, normed * gsum)


def _run_case[
    dtype: DType,
    multiply_before_cast: Bool,
](
    name: String,
    num_rows: Int,
    length: Int,
    input: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin],
    gamma: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin],
    eps: Float32,
    weight_offset: Float32,
    atol: Float32,
    rtol: Float32,
) -> Bool:
    """Drives the real striped `_rms_norm` over the four compute threads, then
    scores it against the float64 reference. Returns True on pass."""
    var count = num_rows * length
    var out = alloc[Scalar[dtype]](count).as_unsafe_any_origin()
    var reference = alloc[Scalar[dtype]](count).as_unsafe_any_origin()

    _fill_sentinel[dtype](out, count)
    _ref_rms_norm[dtype, multiply_before_cast](
        reference, input, gamma, num_rows, length, eps, weight_offset
    )

    # Stands in for the C-ABI seam: the kernel's entry points build these views
    # from the skel's pointers and extents, and `_rms_norm` takes them built.
    # `gamma` is one row because one weight row scales every data row.
    var extent = row_major(Coord(num_rows, length))
    var dst = RowMajor2D[dtype](ptr=out, layout=extent)
    var src = RowMajor2D[dtype](ptr=input, layout=extent)
    var gamma_row = rms_row[dtype](gamma, length)

    # Each compute thread writes the rows its launch index owns; the union covers
    # every row exactly once. Threads 4+ (the DMA thread) take none and return.
    for t in range(4):
        _rms_norm[multiply_before_cast=multiply_before_cast](
            dst,
            src,
            gamma_row,
            UInt32(t),
            eps,
            weight_offset.cast[dtype](),
        )

    var max_err = Float32(0)
    var max_abs = Float32(0)
    var missing = 0
    var nonfinite = 0
    for i in range(count):
        var o = out.load(i).cast[DType.float32]()
        var e = reference.load(i).cast[DType.float32]()
        if not isfinite(o):
            nonfinite += 1
            continue
        var s = SENTINEL
        if (o - s if o > s else s - o) < 1.0:
            missing += 1
        var abs_e = e if e >= 0 else -e
        if abs_e > max_abs:
            max_abs = abs_e
        var diff = o - e
        if diff < 0:
            diff = -diff
        if diff > max_err:
            max_err = diff

    var tol = atol + rtol * max_abs
    var ok = (max_err <= tol) and (missing == 0) and (nonfinite == 0)

    print(
        name,
        "rows",
        num_rows,
        "D",
        length,
        "mbc" if multiply_before_cast else "mac",
        "-> ",
        "PASS" if ok else "FAIL",
        "(max_err=",
        max_err,
        "tol=",
        tol,
        "max|out|=",
        max_abs,
        "missing=",
        missing,
        "nonfinite=",
        nonfinite,
        ")",
    )

    out.free()
    reference.free()
    return ok


def _cases[
    dtype: DType, multiply_before_cast: Bool
](label: String, atol: Float32, rtol: Float32) -> Bool:
    """Runs the full shape/edge matrix for one dtype and one cast order."""
    print("---", label, "---")
    var eps = Float32(1e-6)
    var all_ok = True

    # One gamma buffer wide enough for every case below; each case reads its
    # first `length` weights, which is what the kernel does.
    var gamma = alloc[Scalar[dtype]](6145).as_unsafe_any_origin()
    _fill_seeded[dtype](gamma, 6145, 0x6A33, 1.0)

    # A weight offset of 1.0 is the `gamma + 1` form the registered kernel uses
    # for the checkpoints that store the residual weight; 0.0 is the plain form.
    # Both arms run so neither the offset nor its absence is assumed.
    var offsets = [Float32(0.0), Float32(1.0)]

    for ref off in offsets:
        # Real FLUX MMDiT width, single and multi-row.
        var big = alloc[Scalar[dtype]](8 * 6144).as_unsafe_any_origin()
        _fill_seeded[dtype](big, 8 * 6144, 0xF10C, 3.0)
        all_ok = (
            _run_case[dtype, multiply_before_cast](
                "D6144 x1   ", 1, 6144, big, gamma, eps, off, atol, rtol
            )
            and all_ok
        )
        all_ok = (
            _run_case[dtype, multiply_before_cast](
                "D6144 x8   ", 8, 6144, big, gamma, eps, off, atol, rtol
            )
            and all_ok
        )
        big.free()

        # Small whole-vector width (64 lanes) and two vectors, multi-row so the
        # row split across four threads runs on counts that do and do not divide.
        var small = alloc[Scalar[dtype]](12 * 128).as_unsafe_any_origin()
        _fill_seeded[dtype](small, 12 * 128, 0x5EED, 3.0)
        all_ok = (
            _run_case[dtype, multiply_before_cast](
                "D64 x1     ", 1, 64, small, gamma, eps, off, atol, rtol
            )
            and all_ok
        )
        all_ok = (
            _run_case[dtype, multiply_before_cast](
                "D128 x12   ", 12, 128, small, gamma, eps, off, atol, rtol
            )
            and all_ok
        )
        all_ok = (
            _run_case[dtype, multiply_before_cast](
                "D128 x10   ", 10, 128, small, gamma, eps, off, atol, rtol
            )
            and all_ok
        )
        small.free()

        # Tails: widths that are not a multiple of 64, including one below a full
        # vector, so the masked partial-vector path runs in both passes. The
        # gamma tail is masked by the same predicate as the data tail.
        var tail = alloc[Scalar[dtype]](4 * 6145).as_unsafe_any_origin()
        _fill_seeded[dtype](tail, 4 * 6145, 0x7A11, 3.0)
        all_ok = (
            _run_case[dtype, multiply_before_cast](
                "tail D65   ", 1, 65, tail, gamma, eps, off, atol, rtol
            )
            and all_ok
        )
        all_ok = (
            _run_case[dtype, multiply_before_cast](
                "tail D100  ", 1, 100, tail, gamma, eps, off, atol, rtol
            )
            and all_ok
        )
        all_ok = (
            _run_case[dtype, multiply_before_cast](
                "tail D6145 ", 4, 6145, tail, gamma, eps, off, atol, rtol
            )
            and all_ok
        )
        tail.free()

        # Ramp: smoothly varying, large nonzero mean — a different regime from
        # the zero-mean random fills.
        var ramp = alloc[Scalar[dtype]](3 * 6144).as_unsafe_any_origin()
        _fill_ramp[dtype](ramp, 3, 6144)
        all_ok = (
            _run_case[dtype, multiply_before_cast](
                "ramp D6144 ", 3, 6144, ramp, gamma, eps, off, atol, rtol
            )
            and all_ok
        )
        ramp.free()

        # All-zero rows: mean square 0, so the eps guard is the only thing
        # between the reciprocal and a divide-by-zero. Output must be exactly 0,
        # finite, whatever gamma says.
        var zeros = alloc[Scalar[dtype]](2 * 6144).as_unsafe_any_origin()
        _fill_const[dtype](zeros, 2 * 6144, 0.0)
        all_ok = (
            _run_case[dtype, multiply_before_cast](
                "zero D6144 ", 2, 6144, zeros, gamma, eps, off, atol, rtol
            )
            and all_ok
        )
        zeros.free()

        # Large magnitude: a sum of squares that overflows fp16 (~6e4 max) but
        # not the fp32 accumulator, so a passing result is direct evidence the
        # accumulation is fp32. Output stays normalized regardless of scale.
        var large = alloc[Scalar[dtype]](2 * 6144).as_unsafe_any_origin()
        _fill_seeded[dtype](large, 2 * 6144, 0xB16, 2000.0)
        all_ok = (
            _run_case[dtype, multiply_before_cast](
                "large D6144", 2, 6144, large, gamma, eps, off, atol, rtol
            )
            and all_ok
        )
        large.free()

    gamma.free()
    return all_ok


def run_fused_rope[
    dtype: DType
](name: String, num_rows: Int, length: Int, max_pos: Int) -> Int:
    """The fused arm's bit-identity claim, proven on host.

    Sequential arm: the shipped `_rms_norm` (multiply_before_cast=True, the
    value every qk dispatch passes) into a temp buffer, then the shipped
    `rope_striped[interleaved=True]` over it — selector 5 then selector 9,
    exactly as the block stack dispatches them. Fused arm: `_rms_norm_rope` on
    the same inputs. Every output element must match EXACTLY: the fused path
    stores the normalized row to fp16 memory before rotating, which is the bit
    the sequential path stores and reloads through DDR. A tolerance here would
    hide the exact defect the device gate is built to refuse.

    `positions` is a non-identity permutation so the fused row lookup is proven
    to follow the table, not a row-index assumption.
    """
    comptime SENTINEL_V = Scalar[dtype](-1234.5)
    var eps = Float32(1e-6)
    var weight_offset = Float32(0)
    var n = num_rows * length
    var x = alloc[Scalar[dtype]](n).as_unsafe_any_origin()
    var gamma = alloc[Scalar[dtype]](length).as_unsafe_any_origin()
    var freqs = alloc[Scalar[DType.float32]](
        max_pos * length
    ).as_unsafe_any_origin()
    var positions = alloc[Scalar[DType.uint32]](num_rows).as_unsafe_any_origin()
    var norm_tmp = alloc[Scalar[dtype]](n).as_unsafe_any_origin()
    var out_seq = alloc[Scalar[dtype]](n).as_unsafe_any_origin()
    var out_fused = alloc[Scalar[dtype]](n).as_unsafe_any_origin()

    _fill_seeded[dtype](x, n, 17, 3.0)
    _fill_seeded[dtype](gamma, length, 29, 1.5)
    _fill_seeded[DType.float32](freqs, max_pos * length, 43, 1.0)
    for r in range(num_rows):
        positions.store(r, UInt32((r * 7 + 3) % max_pos))

    var extent = row_major(Coord(num_rows, length))
    var src = RowMajor2D[dtype](ptr=x, layout=extent)
    var gamma_row = rms_row[dtype](gamma, length)
    var freqs_extent = row_major(Coord(max_pos, length))
    var pos_extent = row_major(Coord(num_rows))

    # Sequential arm, both halves the shipped bodies over the shipped split.
    for t in range(4):
        _rms_norm[multiply_before_cast=True](
            RowMajor2D[dtype](ptr=norm_tmp, layout=extent),
            src,
            gamma_row,
            UInt32(t),
            eps,
            weight_offset.cast[dtype](),
        )
    for t in range(4):
        rope_striped[dtype, interleaved=True](
            RopeRowMajor2D[dtype](ptr=out_seq, layout=extent),
            RopeRowMajor2D[dtype](ptr=norm_tmp, layout=extent),
            RopeRowMajor2D[DType.float32](ptr=freqs, layout=freqs_extent),
            RopeRowMajor1D[DType.uint32](ptr=positions, layout=pos_extent),
            UInt32(t),
        )

    var failures = 0

    # Fused arm over the launch-striped split, sentinel-prefilled so a row an
    # index never wrote is counted, not just a row written wrongly. Thread ids
    # 4..5 must take the DMA thread's empty run.
    for i in range(n):
        out_fused.store(i, SENTINEL_V)
    for t in range(6):
        _rms_norm_rope[dtype](
            RowMajor2D[dtype](ptr=out_fused, layout=extent),
            src,
            gamma_row,
            RowMajor2D[DType.float32](ptr=freqs, layout=freqs_extent),
            RopeRowMajor1D[DType.uint32](ptr=positions, layout=pos_extent),
            UInt32(t),
            eps,
            weight_offset.cast[dtype](),
        )

    var mismatched = 0
    var uncovered = 0
    for i in range(n):
        if out_seq.load(i) != out_fused.load(i):
            mismatched += 1
        if out_fused.load(i) == SENTINEL_V:
            uncovered += 1
    if mismatched != 0 or uncovered != 0:
        print(
            "  FAIL",
            name,
            "fused norm+rope mismatch elems=",
            mismatched,
            "uncovered=",
            uncovered,
        )
        failures += 1
    else:
        print("  ok  ", name, "fused == rms_norm;rope_interleaved, bit-exact")

    x.free()
    gamma.free()
    freqs.free()
    positions.free()
    norm_tmp.free()
    out_seq.free()
    out_fused.free()
    return failures


def run_fused_header_checks() -> Int:
    """The fused contract's refusal table, one probe per guard.

    The header reader is the whole device-side validation — the entry above it
    only routes the verdict to the arm word — so proving each guard refuses
    here is what makes 'arm word 2, nothing written' trustworthy on silicon.
    """
    comptime WORDS = 4096
    var w = alloc[Scalar[DType.uint32]](WORDS).as_unsafe_any_origin()
    for i in range(WORDS):
        w.store(i, UInt32(0))

    # A valid contract at byte offset 512: positions at 1024, freqs at 2048.
    var ctl = 512
    var hdr = ctl // 4
    w.store(hdr, RMS_ROPE_MAGIC)
    w.store(hdr + 1, UInt32(1024))
    w.store(hdr + 2, UInt32(2048))
    w.store(hdr + 3, UInt32(128))
    w.store(hdr + 4, UInt32(64))

    var failures = 0

    def expect(name: String, got: Bool, want: Bool) {mut failures}:
        if got != want:
            print("  FAIL header check:", name)
            failures += 1

    var ok = _rms_rope_header(w, ctl, 128)
    expect("valid contract accepted", ok.ok, True)
    expect("positions delta echoed", ok.pos_delta == 1024, True)
    expect("freqs delta echoed", ok.freqs_delta == 2048, True)

    expect("length mismatch", _rms_rope_header(w, ctl, 100).ok, False)
    expect("ctl below floor", _rms_rope_header(w, 60, 128).ok, False)
    expect("ctl unaligned", _rms_rope_header(w, ctl + 2, 128).ok, False)
    expect("ctl negative", _rms_rope_header(w, -8, 128).ok, False)

    w.store(hdr, UInt32(0x52464E30))
    expect("magic mismatch", _rms_rope_header(w, ctl, 128).ok, False)
    w.store(hdr, RMS_ROPE_MAGIC)

    w.store(hdr + 3, UInt32(129))
    expect("odd row length", _rms_rope_header(w, ctl, 129).ok, False)
    w.store(hdr + 3, UInt32(RMS_ROPE_MAX_HEAD + 128))
    expect(
        "row length past the staging cap",
        _rms_rope_header(w, ctl, RMS_ROPE_MAX_HEAD + 128).ok,
        False,
    )
    w.store(hdr + 3, UInt32(128))

    w.store(hdr + 1, UInt32(1026))
    expect("positions delta unaligned", _rms_rope_header(w, ctl, 128).ok, False)
    w.store(hdr + 1, UInt32(32))
    expect(
        "positions delta below floor", _rms_rope_header(w, ctl, 128).ok, False
    )
    w.store(hdr + 1, UInt32(1024))

    w.store(hdr + 2, UInt32(2048 + 64))
    expect(
        "freqs delta off vector alignment",
        _rms_rope_header(w, ctl, 128).ok,
        False,
    )
    w.store(hdr + 2, UInt32(2048))

    expect("contract restored", _rms_rope_header(w, ctl, 128).ok, True)

    if failures == 0:
        print("  ok   fused-contract header: every guard refuses")
    w.free()
    return failures


def main() raises:
    var all_ok = True

    # The width the kernel compiled at, reported so a host run that resolved a
    # different lane count than it claims is visible in the log rather than
    # silently scoring a narrower kernel.
    print("compiled width fp16:", RMSNorm2Pass[DType.float16].width)
    print("compiled width bf16:", RMSNorm2Pass[DType.bfloat16].width)

    # fp16: ~11 mantissa bits, ulp ~1e-3 at |out|~4.
    all_ok = (
        _cases[DType.float16, True]("float16 multiply_before_cast", 2e-2, 1e-2)
        and all_ok
    )
    all_ok = (
        _cases[DType.float16, False]("float16 multiply_after_cast", 2e-2, 1e-2)
        and all_ok
    )

    # bf16: 8 mantissa bits, ulp ~2e-2 at |out|~4 — a coarser tolerance for a
    # coarser type, still far tighter than any algorithm bug would clear.
    all_ok = (
        _cases[DType.bfloat16, True](
            "bfloat16 multiply_before_cast", 8e-2, 2e-2
        )
        and all_ok
    )
    all_ok = (
        _cases[DType.bfloat16, False](
            "bfloat16 multiply_after_cast", 8e-2, 2e-2
        )
        and all_ok
    )

    # The fused norm+rope arm (SKYE_HMX_ROPE_FUSED_NORM): bit-exact against the
    # selector 5 -> selector 9 sequence, at the FLUX head shape, a scalar-tail
    # length, a mixed vector+tail length, and a sub-thread row count; then the
    # contract header's refusal table. fp16 only — the fused arm compiles for
    # the one production selector.
    print("")
    print("==== fused norm + rope (bit-exact vs sequential) ====")
    var fused_failures = 0
    fused_failures += run_fused_rope[DType.float16](
        "fp16 flux head rows", 37, 128, 96
    )
    fused_failures += run_fused_rope[DType.float16](
        "fp16 scalar tail", 9, 100, 16
    )
    fused_failures += run_fused_rope[DType.float16](
        "fp16 vector+tail", 12, 192, 32
    )
    fused_failures += run_fused_rope[DType.float16](
        "fp16 sub-thread", 3, 128, 8
    )
    fused_failures += run_fused_header_checks()
    all_ok = fused_failures == 0 and all_ok

    print("")
    if all_ok:
        print("PASS")
    else:
        print("FAIL")
        raise Error("rms_norm algorithm test failed")
