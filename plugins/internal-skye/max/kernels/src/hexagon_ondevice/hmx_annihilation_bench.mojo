"""Rung 1 of the crouton-crossing annihilation ladder: the two-arm microbench
body (`DESIGN-crouton-crossing-annihilation.md` section 9).

The chain under test is the double block's MLP half at `klein_mini` geometry
(`block_stack.mojo`): `FF_IMG = A @ W_ff_in` -> `swiglu` -> `FFO_IMG = MLP @
W_ff_out`, the one place in the whole denoise sequence where BOTH sides of
BOTH matmuls are elementwise — the full tunnel. Two arms over identical
inputs:

  - **baseline** — today's code: the FF matmul unpacks its output to
    row-major, `swiglu_2d` reads and writes row-major, the FFO matmul packs
    its activation from row-major. Two crossings paid.
  - **native** — the FF matmul writes croutons (`output_native`),
    `swiglu_croutons` reads and writes croutons, the FFO matmul reads
    croutons (`activation_native`). Zero crossings: the design's geometry
    fact made executable — the FF output's crouton order IS the FFO
    activation's, `(rt * w_tiles + wt)` both, so nothing between them
    translates anything.

Both arms share the weight arm (the packing gather, `weight_native=False`) so
`STAGE_WGT` cancels in the A/B, and both end row-major: the FFO matmul's
output stays `output_native=False` in each, so the two verdict buffers are
directly comparable. The pre-registered gate is BIT-IDENTITY — a pure layout
change carries no numerics, the same bar `weight_native` shipped under — and
the pre-registered device reading is the `HMX_PHASE_PROBE` table: `PACK_ACT`
on the native FFO dispatch and `UNPACK` on the native FF dispatch drop to the
linear copy, or the ladder stops here (design section 9, the falsification
clause).

WHY THIS LIVES IN KERNEL SOURCE: the host gate (`test_annihilation_bench.mojo`)
and the device probe must run the SAME arm configuration, fills and verdict,
not two transcriptions of one idea — the `matmul_bt_stripe_hmx_step` drift
rule one level up. The device entry is `mojo_hmx_annihilation_bench`
(`npu_kernel.mojo`, `NPUKernels.def` selector 43, re-sign window item 7): a
thin shim over this module's carve, fills, per-stage bodies and verdict, one
launch per stage-arm so the phase table attributes per dispatch, which is
what the design's verdict reads. The shim-protocol section below is shared
by both drivers so neither can drift from the other.

The flags this module instantiates are default-off everywhere else, so no
shipped GRAPH path changes by this file's existence; the shim entry itself
is the window item and lands only with the batched skel rebuild + re-sign.
"""
from layout.coord import Coord
from layout.tile_layout import row_major

from std.memory import AddressSpace, bitcast

from .hmx_crouton import (
    HMX_TILE,
    HMX_TILE_BYTES,
    CroutonPanel,
    CroutonPanelOf,
    HmxRows2D,
    HmxStripeArena,
    hmx_panel,
)
from .matmul_kernel import (
    MMTile,
    HmxStripePlan,
    hmx_work_share,
    matmul_bt_stripe_hmx_step,
)
from .silu_swiglu_kernel import swiglu_croutons, swiglu_striped

comptime ANNI_M = 128
"""Rows: `klein_mini`'s `image_rows` (`image_seq = 128`, batch folded)."""

comptime ANNI_DIM = 512
"""Model width: `klein_mini`'s `heads * head_dim = 4 * 128`."""

comptime ANNI_HIDDEN = 1536
"""SwiGLU half-width: `mlp_ratio * dim = 3 * 512`."""

comptime ANNI_FF_N = 2 * ANNI_HIDDEN
"""The FF matmul's output width: gate and up in one buffer (`ff_in_width`)."""

comptime ANNI_ACT_SCALE = Scalar[DType.float16](0.015625)
"""1/64, exact in fp16. Keeps the chain's magnitudes finite: an unscaled
coprime fill drives `silu(gate) * up` past fp16's 65504 at this K, and an
`inf * 0` NaN would break the bit-identity verdict on values rather than on
layout."""


def annihilation_fill_activation(a: HmxRows2D, rows: Int, cols: Int):
    """The modulated-input fill: coprime periods, both `silu` branches, scaled
    to keep every downstream value finite. fp16-exact by construction."""
    for r in range(rows):
        for c in range(cols):
            a.store[1, alignment=2](
                Coord(r, c),
                Scalar[DType.float16]((5 * r + 7 * c) % 13 - 6)
                * ANNI_ACT_SCALE,
            )


def annihilation_fill_weight(b: HmxRows2D, rows: Int, cols: Int, salt: Int):
    """A weight fill distinct per `salt`, so the two matmuls cannot swap
    weights and still agree. Small integers, exact in fp16."""
    for r in range(rows):
        for c in range(cols):
            b.store[1, alignment=2](
                Coord(r, c),
                Scalar[DType.float16]((3 * r + 11 * c + salt) % 7 - 3),
            )


@always_inline
def annihilation_matmul_step[
    space: AddressSpace, //, native_a: Bool, native_c: Bool
](
    C: MMTile,
    A: MMTile,
    B: MMTile,
    arena: HmxStripeArena[space],
    plan: HmxStripePlan,
    step: Int,
    thread_idx: Int,
    feeders: Int,
    mut unpack_arm: UInt32,
):
    """One thread's work in one step of a bench matmul: THE arm configuration.

    Stated once so the host gate and the device shim instantiate identical
    arms. `weight_native=False` in every arm — the packing gather, identical
    on both sides of the A/B, so the weight phase cancels — and the two
    native flags are the only thing that distinguishes a bench matmul from
    the production step function.
    """
    matmul_bt_stripe_hmx_step[
        weight_native=False,
        activation_native=native_a,
        output_native=native_c,
    ](C, A, B, arena, plan, step, thread_idx, feeders, unpack_arm)


@always_inline
def annihilation_swiglu_thread[
    native: Bool,
    in_space: AddressSpace,
    out_space: AddressSpace,
](
    ff_rows: HmxRows2D,
    mlp_rows: HmxRows2D,
    ff_croutons: CroutonPanelOf[DType.float16, in_space],
    mlp_croutons: CroutonPanelOf[DType.float16, out_space],
    rows: Int,
    hidden: Int,
    thread_idx: Int,
    feeders: Int,
):
    """One thread's share of the bench swiglu, either arm.

    The baseline arm is `swiglu_striped` — the production row split over the
    row-major views. The native arm stripes ROW TILES with the stripe's own
    `hmx_work_share`, so its four shares partition the row-panel range the
    way the matmul's feeder shares partition a panel; the unused pair of
    views in each arm is simply never read.
    """
    comptime if native:
        var rt0 = hmx_work_share(rows // HMX_TILE, thread_idx, feeders)
        var rt1 = hmx_work_share(rows // HMX_TILE, thread_idx + 1, feeders)
        if rt1 > rt0:
            swiglu_croutons(ff_croutons, mlp_croutons, hidden, rt0, rt1 - rt0)
    else:
        swiglu_striped[DType.float16](ff_rows, mlp_rows, UInt32(thread_idx))


def annihilation_verdict(
    c_baseline: HmxRows2D, c_native: HmxRows2D, rows: Int, cols: Int
) -> Int:
    """Mismatched elements between the two arms' final outputs, as bits.

    The pre-registered gate: zero, or the ladder stops. Bits rather than a
    tolerance because a pure layout change carries no numerics — any non-zero
    return is a layout defect, not a rounding argument.
    """
    var wrong = 0
    for r in range(rows):
        for c in range(cols):
            if c_baseline.load[1, alignment=2](Coord(r, c)) != c_native.load[
                1, alignment=2
            ](Coord(r, c)):
                wrong += 1
    return wrong


# ===------------------------------------------------------------------------===
# The shim protocol (`mojo_hmx_annihilation_bench`, selector 43).
#
# Everything the device entry and the host gate must agree on byte for byte —
# the scratch carve, the per-stage fills, the verdict path and the witness
# words — lives HERE, so the two drivers instantiate one statement of it. The
# entry in `npu_kernel.mojo` adds only what the host cannot run: the launch
# roles, the HMX grant, the rendezvous and the VTCM arena.
#
# One scratch buffer carries the whole chain. The host allocates
# `ANNI_SCRATCH_BYTES` of ordinary device memory once and passes it to every
# dispatch; the shim aligns the base and carves the same nine sections every
# time, so a buffer written by dispatch N is exactly where dispatch N+1 reads
# it. Every section is a whole number of croutons and every offset a multiple
# of `HMX_TILE_BYTES`, so the two crouton panels inherit the alignment the
# `:deep` load needs from the one `align_up` on the base.
# ===------------------------------------------------------------------------===

comptime ANNI_ARM_BASELINE = 0
"""Today's code: row-major intermediates, two crossings paid."""

comptime ANNI_ARM_NATIVE = 1
"""The rung-1 arm: crouton intermediates, zero crossings."""

comptime ANNI_STAGE_FF = 0
"""`FF = A @ W_ff_in`, the `[M, 2*hidden]` gate/up projection."""

comptime ANNI_STAGE_SWIGLU = 1
"""`MLP = silu(gate) * up`, `[M, hidden]`."""

comptime ANNI_STAGE_FFO = 2
"""`C = MLP @ W_ff_out`, `[M, dim]`. The native-arm dispatch of this stage
carries the pre-registered verdict."""

comptime ANNI_STAGE_FALSIFY = 3
"""The verdict's must-fire control: plants one flipped output bit, runs the
same verdict path, restores the bits. A zero verdict from THIS stage means
the gate cannot fail and no pass from it counts (design section 9's
falsification clause, applied to the instrument itself)."""

comptime ANNI_WIT_MARKER = 25
"""`ANNI_MARKER_DONE` when the stage protocol ran to its end on thread 0,
`ANNI_MARKER_REFUSED` when the shim declined the dispatch. Above the matmul
witness map (words 0-22, `MATMUL_WIT_*` and the words 8-14 lock-refusal
report), which the shim's matmul stages reuse unchanged."""

comptime ANNI_WIT_ARM = 26
"""The `arm` scalar echoed, so a readback names the dispatch it came from."""

comptime ANNI_WIT_STAGE = 27
"""The `stage` scalar echoed, beside the arm for the same reason."""

comptime ANNI_WIT_VERDICT = 28
"""`annihilation_verdict`'s mismatch count. Seeded `ANNI_VERDICT_UNSET` on
every dispatch; written only by the native-FFO stage (the real verdict) and
the falsify stage (the planted one), so any other stage reads back the
seed."""

comptime ANNI_WIT_WORDS = 32
"""Words the witness buffer must hold. The matmul map's top word is 22 and
the bench's is 28; 32 leaves headroom without a second capacity scalar."""

comptime ANNI_MARKER_DONE: UInt32 = 0x414E4931
"""'ANI1' — the stage completed."""

comptime ANNI_MARKER_REFUSED: UInt32 = 0x414E4930
"""'ANI0' — the shim refused the arguments rather than scribbling."""

comptime ANNI_VERDICT_UNSET: UInt32 = 0xFFFFFFFF
"""No verdict was computed by this dispatch. A count can never reach this
value: the output is `ANNI_M * ANNI_DIM` elements."""

comptime ANNI_SEED_INTERMEDIATE = Scalar[DType.float16](-4096.0)
"""Sentinel for the stage intermediates, either arm."""

comptime ANNI_SEED_C_BASELINE = Scalar[DType.float16](-4096.0)
"""Sentinel for the baseline verdict buffer."""

comptime ANNI_SEED_C_NATIVE = Scalar[DType.float16](4096.0)
"""Sentinel for the native verdict buffer — DIFFERENT from the baseline's,
so a region left unwritten by both arms cannot compare equal."""

comptime _ANNI_BYTES_A = 2 * ANNI_M * ANNI_DIM
comptime _ANNI_BYTES_W_FF = 2 * ANNI_FF_N * ANNI_DIM
comptime _ANNI_BYTES_W_FFO = 2 * ANNI_DIM * ANNI_HIDDEN
comptime _ANNI_BYTES_FF = 2 * ANNI_M * ANNI_FF_N
comptime _ANNI_BYTES_MLP = 2 * ANNI_M * ANNI_HIDDEN
comptime _ANNI_BYTES_C = 2 * ANNI_M * ANNI_DIM

comptime ANNI_OFF_A = 0
"""`a_mod [M, dim]` — the shared modulated input, both arms."""

comptime ANNI_OFF_W_FF = ANNI_OFF_A + _ANNI_BYTES_A
"""`w_ff [2*hidden, dim]` — the FF weight (salt 1), both arms."""

comptime ANNI_OFF_W_FFO = ANNI_OFF_W_FF + _ANNI_BYTES_W_FF
"""`w_ffo [dim, hidden]` — the FFO weight (salt 2), both arms."""

comptime ANNI_OFF_FF_BASE = ANNI_OFF_W_FFO + _ANNI_BYTES_W_FFO
"""Baseline FF output, row-major `[M, 2*hidden]`."""

comptime ANNI_OFF_MLP_BASE = ANNI_OFF_FF_BASE + _ANNI_BYTES_FF
"""Baseline swiglu output, row-major `[M, hidden]`."""

comptime ANNI_OFF_C_BASE = ANNI_OFF_MLP_BASE + _ANNI_BYTES_MLP
"""Baseline verdict buffer, row-major `[M, dim]`."""

comptime ANNI_OFF_FF_NAT = ANNI_OFF_C_BASE + _ANNI_BYTES_C
"""Native FF output = the swiglu input: ONE row-panel-major crouton panel,
handed through untouched (the rung's whole claim)."""

comptime ANNI_OFF_MLP_NAT = ANNI_OFF_FF_NAT + _ANNI_BYTES_FF
"""Native swiglu output = the FFO activation, the same statement again."""

comptime ANNI_OFF_C_NAT = ANNI_OFF_MLP_NAT + _ANNI_BYTES_MLP
"""Native verdict buffer, row-major `[M, dim]` — both arms end row-major so
the verdict compares directly."""

comptime ANNI_SCRATCH_END = ANNI_OFF_C_NAT + _ANNI_BYTES_C

comptime ANNI_SCRATCH_BYTES = ANNI_SCRATCH_END + HMX_TILE_BYTES
"""What the host must allocate: the carve plus one crouton of slack for the
base alignment. 7,473,152 bytes at `klein_mini`."""

comptime ANNI_FF_TILES = (ANNI_M // HMX_TILE) * (ANNI_FF_N // HMX_TILE)
comptime ANNI_MLP_TILES = (ANNI_M // HMX_TILE) * (ANNI_HIDDEN // HMX_TILE)


# A `TileTensor[.., MutUnsafeAnyOrigin]` cannot be a struct field, so the
# carve is a family of view functions rather than one returned struct: each
# section has exactly one function that names its offset and shape, and the
# arm-dependent sections take `arm` so the baseline/native selection is
# stated once, here, for both drivers.


@always_inline
def _anni_rows(base: Int, off: Int, rows: Int, cols: Int) -> HmxRows2D:
    # Every section a whole number of croutons: one aligned base aligns all.
    comptime assert ANNI_OFF_W_FF % HMX_TILE_BYTES == 0, "carve misaligned"
    comptime assert ANNI_OFF_W_FFO % HMX_TILE_BYTES == 0, "carve misaligned"
    comptime assert ANNI_OFF_FF_BASE % HMX_TILE_BYTES == 0, "carve misaligned"
    comptime assert ANNI_OFF_MLP_BASE % HMX_TILE_BYTES == 0, "carve misaligned"
    comptime assert ANNI_OFF_C_BASE % HMX_TILE_BYTES == 0, "carve misaligned"
    comptime assert ANNI_OFF_FF_NAT % HMX_TILE_BYTES == 0, "carve misaligned"
    comptime assert ANNI_OFF_MLP_NAT % HMX_TILE_BYTES == 0, "carve misaligned"
    comptime assert ANNI_OFF_C_NAT % HMX_TILE_BYTES == 0, "carve misaligned"
    comptime assert ANNI_SCRATCH_END % HMX_TILE_BYTES == 0, "carve misaligned"
    return HmxRows2D(
        ptr=UnsafePointer[Scalar[DType.float16], MutUnsafeAnyOrigin](
            unsafe_from_address=base + off
        ),
        layout=row_major(Coord(rows, cols)),
    )


@always_inline
def _anni_panel(
    base: Int, off: Int, tiles: Int
) -> CroutonPanel[AddressSpace.GENERIC]:
    return hmx_panel[AddressSpace.GENERIC](
        UnsafePointer[Scalar[DType.float16], MutUntrackedOrigin](
            unsafe_from_address=base + off
        ),
        tiles,
    )


@always_inline
def annihilation_a_mod(base: Int) -> HmxRows2D:
    """The shared modulated input, `[M, dim]`."""
    return _anni_rows(base, ANNI_OFF_A, ANNI_M, ANNI_DIM)


@always_inline
def annihilation_w_ff(base: Int) -> HmxRows2D:
    """The FF weight (salt 1), `[2*hidden, dim]`."""
    return _anni_rows(base, ANNI_OFF_W_FF, ANNI_FF_N, ANNI_DIM)


@always_inline
def annihilation_w_ffo(base: Int) -> HmxRows2D:
    """The FFO weight (salt 2), `[dim, hidden]`."""
    return _anni_rows(base, ANNI_OFF_W_FFO, ANNI_DIM, ANNI_HIDDEN)


@always_inline
def annihilation_ff_out(base: Int, arm: Int) -> HmxRows2D:
    """The arm's FF output as the matmul seam's `[M, 2*hidden]` view — the
    baseline's row-major section, or the native crouton section's bytes
    behind the same shape (`hmx_native_row_panel` re-derives the panel from
    it inside the step)."""
    if arm == ANNI_ARM_NATIVE:
        return _anni_rows(base, ANNI_OFF_FF_NAT, ANNI_M, ANNI_FF_N)
    return _anni_rows(base, ANNI_OFF_FF_BASE, ANNI_M, ANNI_FF_N)


@always_inline
def annihilation_mlp_out(base: Int, arm: Int) -> HmxRows2D:
    """The arm's swiglu output as `[M, hidden]`, selected as `ff_out` is."""
    if arm == ANNI_ARM_NATIVE:
        return _anni_rows(base, ANNI_OFF_MLP_NAT, ANNI_M, ANNI_HIDDEN)
    return _anni_rows(base, ANNI_OFF_MLP_BASE, ANNI_M, ANNI_HIDDEN)


@always_inline
def annihilation_c_out(base: Int, arm: Int) -> HmxRows2D:
    """The arm's verdict buffer, `[M, dim]`. Both arms end row-major, so the
    two returns of this function are what the verdict compares."""
    if arm == ANNI_ARM_NATIVE:
        return _anni_rows(base, ANNI_OFF_C_NAT, ANNI_M, ANNI_DIM)
    return _anni_rows(base, ANNI_OFF_C_BASE, ANNI_M, ANNI_DIM)


@always_inline
def annihilation_ff_panel(base: Int) -> CroutonPanel[AddressSpace.GENERIC]:
    """The native FF output = swiglu input crouton panel: `ff_out`'s native
    bytes under their real meaning."""
    return _anni_panel(base, ANNI_OFF_FF_NAT, ANNI_FF_TILES)


@always_inline
def annihilation_mlp_panel(base: Int) -> CroutonPanel[AddressSpace.GENERIC]:
    """The native swiglu output = FFO activation crouton panel."""
    return _anni_panel(base, ANNI_OFF_MLP_NAT, ANNI_MLP_TILES)


def annihilation_seed_rows(
    dst: HmxRows2D, rows: Int, cols: Int, value: Scalar[DType.float16]
):
    """Seeds a section with a sentinel through its row view. The native
    panels seed through their `_mm` twin — same bytes, cheaper index."""
    for r in range(rows):
        for c in range(cols):
            dst.store[1, alignment=2](Coord(r, c), value)


def annihilation_stage_prepare(base: Int, arm: Int, stage: Int):
    """Fills what the stage reads and seeds what it writes, on ONE thread,
    before the launch's compute rendezvous.

    Idempotent per (arm, stage), so host-driven repeats re-run a stage
    without accumulating state. A stage never re-fills what a PREVIOUS stage
    wrote — the swiglu's input and the FFO's activation arrive from the
    prior dispatch through the shared scratch, which is the handoff under
    test. The falsify stage prepares nothing: it reads the two verdict
    buffers exactly as the last protocol dispatch left them.
    """
    var seed = ANNI_SEED_C_BASELINE
    if arm == ANNI_ARM_NATIVE:
        seed = ANNI_SEED_C_NATIVE
    if stage == ANNI_STAGE_FF:
        annihilation_fill_activation(annihilation_a_mod(base), ANNI_M, ANNI_DIM)
        annihilation_fill_weight(
            annihilation_w_ff(base), ANNI_FF_N, ANNI_DIM, 1
        )
        annihilation_seed_rows(
            annihilation_ff_out(base, arm),
            ANNI_M,
            ANNI_FF_N,
            ANNI_SEED_INTERMEDIATE,
        )
    elif stage == ANNI_STAGE_SWIGLU:
        annihilation_seed_rows(
            annihilation_mlp_out(base, arm),
            ANNI_M,
            ANNI_HIDDEN,
            ANNI_SEED_INTERMEDIATE,
        )
    elif stage == ANNI_STAGE_FFO:
        annihilation_fill_weight(
            annihilation_w_ffo(base), ANNI_DIM, ANNI_HIDDEN, 2
        )
        annihilation_seed_rows(
            annihilation_c_out(base, arm), ANNI_M, ANNI_DIM, seed
        )


def annihilation_shim_verdict(
    base: Int,
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    plant: Bool,
):
    """THE SHIM'S VERDICT PATH: the pre-registered gate into the witness.

    Builds both arms' verdict views from the carve itself, so the device
    entry and the host gate exercise one path from scratch base to witness
    word. `plant` is the must-fire control — it flips the low bit of one
    output element (a guaranteed value change for the chain's finite fp16
    values), runs the SAME verdict, and restores the exact saved bits, so a
    follow-up verdict still reads clean. A reader that cannot see this word
    go non-zero under `plant` has no business believing its zero.
    """
    var c_base = annihilation_c_out(base, ANNI_ARM_BASELINE)
    var c_nat = annihilation_c_out(base, ANNI_ARM_NATIVE)
    var saved = c_nat.load[1, alignment=2](Coord(0, 0))
    if plant:
        c_nat.store[1, alignment=2](
            Coord(0, 0),
            bitcast[DType.float16, 1](bitcast[DType.uint16, 1](saved) ^ 1),
        )
    var wrong = annihilation_verdict(c_base, c_nat, ANNI_M, ANNI_DIM)
    if plant:
        c_nat.store[1, alignment=2](Coord(0, 0), saved)
    witness.store(ANNI_WIT_VERDICT, UInt32(wrong))
