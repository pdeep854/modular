"""The standalone HMX characterization bench: pure-array / movement / overlap,
fp16 + int8 (FLEET-V2; device legs ride order DRAFT-210).

THE QUESTION THIS BENCH ANSWERS, in three separable numbers per
(dtype, geometry):

  a. PURE ARRAY — operands pre-resident in VTCM, a tight chained-MMA loop,
     UPCYCLE + QTimer bracketed around compute only -> GMAC/s against the
     MEASURED fp16 port line (159R: 1280.04 GB/s marginal = ~10.2 TMAC/s
     at the instruction's 8 MAC/byte — a ceiling, per the 1466-vs-1280
     reconciliation draft), with the 1,279 GMAC/s in-tree rating
     (1024 MAC/cyc x 1249 MHz) named as the stale anchor the leg retires —
     the rating's digits collide with the measurement's GB/s and the two
     are different quantities eight-fold apart. cyc/packet reads against
     the `cdsp_peak` v81 decode (fp16 8.78, int8 6.80;
     `HMX_INT8_SCOPE.md` section 3.1) as context, never as a band.
  b. MOVEMENT ONLY — DDR->VTCM staging alone, per transport: HVX demand
     loads (with and without `l2fetch`) AND the user-DMA engine ->
     GB/s per path, against the measured 35.49 GB/s (HVX, 4 threads) and
     31.7 GB/s (uDMA) anchors.
  c. OVERLAPPED — the production pipelined stripe (stage column tile
     `j + 1` while the array multiplies tile `j`), phase-split by
     `HMX_PHASE_PROBE` -> achieved GMAC/s and overlap efficiency against
     the (a, b) composition floor.

WHAT ALREADY EXISTS SIGNED, AND IS REUSED RATHER THAN REBUILT. Three of the
four device legs run on selectors already in the catalogue, so they need no
re-sign at all:

  - (a) fp16 is selector 42 `mxmem_port_probe` (`npu_kernel.mojo`): iters
    complete fp16 MMAs over VTCM-resident crouton pairs, grant held across
    passes, two-clock bracket, sum-fold witness. Its `tiles` axis IS the K
    axis: K = 32 * tiles.
  - (b) is selector 33 `membw_probe`: modes LOADS / LOADS_PF / UDMA /
    UDMA_ONCE walk the same DDR slice at the same chunk grain, per-thread
    quarters, participant mask varying concurrency and nothing else.
    `npu_membw_dispatch` sweeps both, plus the operand-port table.
  - (c) is selector 24 `matmul` with the HMX arm, under an
    `HMX_PHASE_PROBE=1` + `MATMUL_PERF_TIMING=1` probe skel — the 190C
    instrument. The phase partition (PACK_ACT / STAGE_WGT / WGT_AWAIT /
    UNPACK / MMA / BARRIER / IDLE) sums to the dispatch by construction.

WHAT DOES NOT EXIST, WHICH THIS MODULE SUPPLIES THE BODY FOR: an
iters-amortized int8 pure-array arm under the SAME bracket and witness
contract as the fp16 one. Order 184's stripe-i8 harness could not express
the fp16 sibling ("the fp16-sibling premise was inexpressible",
HANDOFF.md:9850-9857), and the only int8 timing on record is selector 29's
single un-amortized chain (184B: one output tile, feeders in the bracket,
unusable as a ratio). The two pass bodies below are that missing pair,
stated once so the host gate and the device shim instantiate identical arms
— the `matmul_bt_stripe_hmx_step` drift rule, exactly as
`hmx_annihilation_bench.mojo` applies it.

THE DEVICE ENTRY IS `mojo_npu_hmx_bench` (`npu_kernel.mojo`, selector 44 —
43 is window item 7's `hmx_annihilation_bench`, which merges first —
window item 8's build): a thin shim over this module's pass bodies and
accounting, whose `NPUKernels.def` row is the signed boundary and therefore
LANDS ONLY inside the next re-sign window (`PERF_PLAN.md` section "The
re-sign window queue"). Buffers: the movement modes' DDR sink — its own
slot, because the device can only dereference a buffer passed as its own
pointer argument (the JIT per-arg buffer map), which is the one deviation
from this docstring's original single-buffer sketch — then `membw_probe`'s
witness (`MEMBW_WITNESS_STRIDE` words per thread, reader unchanged).
Scalars: `(mode, k_tiles, iters, depth)` — `depth` is window item 9's
ring-depth knob on the uDMA movement mode (DRAFT-092 leg A' runs on this
body), accepted only as 1 everywhere else; `bench_case_status` is the
composed refusal ladder the shim runs.
Modes: 0 = pure-array fp16 `:deep`, 1 = pure-array fp16 plain, 2 =
pure-array int8 plain (the `:deep` int8 form does not assemble in this
tree — `hmx_int8.mojo`'s form table; its MC support is PR #438, open), 3 =
VTCM->DDR uDMA writeback, 4 = VTCM->DDR HVX stores — the two directions
`membw_probe` does not cover. The shim body is the port probe's: carve
`hmx_scratch_in` from the VTCM grant, fill, program the record once, take
the grant, bracket `iters` passes with UPCYCLE + QTimer, fold the readout,
report. The uDMA arms are modeled on the reference backend's queue
(learn-before-invent: `~/work/reference/llama.cpp` @ `1692f9e50bb`,
`ggml/src/ggml-hexagon/htp/dma-queue.h` — per-thread `dmlink` ring,
per-descriptor `done` + `dmpoll` completion, `push_vtcm_to_ddr` as a
production-normal direction), with one NAMED DELTA the bench reports
rather than hides: our shim holds one outstanding transfer per thread and
refuses chaining, so every uDMA rate here is a depth-1 rate against the
reference's 256-deep ring (`HMX_BENCH_NOTES.md` section 2b). Until the
window lands the row, the shipped catalogue is unchanged: a signed skel
predating it refuses the selector by name, which the driver reports and
survives.

THE DMA ANSWER THE BENCH SITS UNDER, from source, so the order's brief does
not restate it wrong: in today's production matmul the WEIGHT panel
(STAGE_WGT) moves by user-DMA only on the weight-native arm
(`SKYE_HMX_WEIGHTS_NATIVE=1`, the 202-config denoise skel):
`_hmx_stage_weight` defaults `weight_xfer = HMX_XFER_DMA`
(`matmul_kernel.mojo:2020`) and submits one linear type-0 descriptor per
feeder through `skye_udma_submit_linear_lo32` (`hmx_crouton.mojo:2461`,
shim `qshim_qurt.c:2002-2121`), awaited by `qshimUDmaAwait` = `dmwait` +
the descriptor's own `dstate` poll. On the arm=0 skel (the 202 VAE and
encoder) the same phase is `hmx_pack_panel[True]` — a transposing gather
by HVX butterfly or scalar element loads, no DMA. The ACTIVATION side
(PACK_ACT) never uses DMA on any arm: it is a transform (interleave,
quantize, or NHWC gather), not a copy, and only rung 1's
`activation_native` linear-copy form could ever become one. 2D (type-1)
uDMA is refused by the engine from a user PD on this part — settled on
silicon by order 161 (`EVAL_161.md`, PR #423) — so every DMA statement
above is linear type-0, which the column-panel-major artefact exists to
make sufficient.

WHICH int8 NUMBERS ARE TRUSTWORTHY PRE-FIX: the chained plain-stream int8
arm carries the pair-low addressing defect (each 64-deep crouton pair's
second activation MAC re-reads the LOW crouton; fitted model
`effective_act_index(k) = 64*(k//64) + (k%32)`, oracle `reference_pairlow`
on PR #449's branch `mac/int8-addressing-fix`). TIMING is valid — the
defect changes WHICH bytes are read, not how many, at unchanged issue
count (order 184 measured timing under exactly this rule) — so mode-2
rates are quotable pre-fix with the defect named. NUMERICS at
k_tiles >= 2 are NOT (K=32's single crouton is the identity case that
never saw the bug — 184F): the correctness leg gates on the oracle (a
dump matching `reference_pairlow` IS the bug), and chained-value claims
wait for the fix.
"""
from std.memory import AddressSpace, bitcast

from layout.coord import Coord

from .hmx_crouton import (
    HMX_DTYPE,
    HMX_EMIT_MMA,
    HMX_TILE,
    CroutonPanelOf,
    HmxOperands,
    HmxScratch,
    HmxSplit,
    HmxTileBytes,
    hmx_mma_panel,
    hmx_port_probe_bytes,
    hmx_port_probe_chain,
    hmx_port_probe_min_tiles,
    hmx_program_identity_record,
)
from .hmx_int8 import (
    HMX_INT8_DTYPE,
    HMX_INT8_IDENTITY_SCALE,
    hmx_int8_mma_panel,
    hmx_int8_program_record,
)
from .matmul_kernel import hmx_work_share

comptime BENCH_QTIMER_HZ = 19_200_000
"""The fixed-rate QTimer the brackets report wall time through. No clock
change scales it, which is what makes a rate against it a rate."""

comptime BENCH_ARRAY_RATED_GMACS = 1279
"""The in-tree array rating: 1024 MAC/cyc x 1249 MHz
(`VAE_REFERENCE_MODEL.md` section 3.1). Carried as the anchor mode (a)
formally RETIRES, not as the expectation — see the constant below."""

comptime BENCH_ARRAY_PORT_MEASURED_GBS = 1280
"""159R's measured marginal port rate, in GB/s — 606.08 B/core-cycle at the
bracket's own 2112 MHz; ~10.2 TMAC/s at the paired packet's 8 MAC/byte, a
measured CEILING (`orders/DRAFT-1466-vs-1280-...`). NOT the same quantity
as `BENCH_ARRAY_RATED_GMACS` despite the colliding digits: GB/s here,
GMAC/s there, eight-fold apart. Both constants exist so a reader who
conflates them trips over this docstring instead of citing the wrong
one."""

comptime BENCH_K_SMALL = 64
comptime BENCH_K_MID = 1024
comptime BENCH_K_PROD = 12288
"""The 184-harness K ladder. `BENCH_K_SMALL` is 2 croutons and sits BELOW
both dtypes' pure-array refusal floors (`hmx_port_probe_min_tiles`: 29 fp16,
32 int8) — its pure-array row is a pre-registered REFUSAL, not a gap; the
movement and overlap modes serve it."""

comptime BENCH_PROD_M = 1536
comptime BENCH_PROD_N = 3072
"""With `BENCH_K_PROD`: `sgl out-proj`, order 184's primary shape — the
denoise's dominant matmul. Mode (c) dispatches it through selector 24; modes
(a)/(b) see only its K and its panel bytes, because a pure-array rate is
per-tile by construction (one 32x32 output tile per pass) and M, N multiply
pass counts, not the rate."""

comptime BENCH_FORM_DEEP = 0
"""Stream the activation with `:deep`: one packet per 32 fp16 croutons."""

comptime BENCH_FORM_PLAIN = 1
"""One packet per crouton — the only form the int8 arm can issue."""

comptime BENCH_ARM_FP16: UInt32 = 0x1600
comptime BENCH_ARM_W8A8: UInt32 = 0x0800
"""Which pass body ran, echoed by the pass so a build that silently routed
the wrong arm reads as a FAIL rather than as a plausible number — the
treatment-delivery fingerprint every A/B on this port quotes per leg."""

comptime BENCH_ARM_ISSUED: UInt32 = 0x1
"""OR'd into the arm code when the instructions were issued rather than
modelled. A rate must never be reported from a run without this bit: the
modelled arm is many orders too slow to time and reports the build, not the
part."""

comptime BENCH_MODE_FP16_DEEP = 0
"""Pure array, fp16, `:deep` addressing — selector 42's own arm, carried so
the int8 sibling and the fp16 line share one bracket and one witness
contract."""

comptime BENCH_MODE_FP16_PLAIN = 1
"""Pure array, fp16, plain addressing: the int8 ratio's denominator arm —
matched issue discipline, one packet per crouton on both sides."""

comptime BENCH_MODE_INT8_PLAIN = 2
"""Pure array, int8, plain addressing — the iters-amortized sibling order
184's harness could not express (selector 29's single chain brackets its
feeders and cannot be a ratio). TIMING is valid pre-fix; see the module
docstring's trust boundary."""

comptime BENCH_MODE_UDMA_OUT = 3
"""Movement, VTCM->DDR by linear uDMA with per-transfer completion: the
write direction `membw_probe` lacks, the reference backend's
`dma_queue_push_vtcm_to_ddr`. Depth-1 by the shim's one-outstanding cap —
report its rates as such."""

comptime BENCH_MODE_FP16_INSTRIPE = 5
"""In-stripe residency: every pass DMAs its own act+wgt slot DDR->VTCM
inside the bracket before the same MMA chain modes 0-1 time — the
production stripe's residency class on one thread, so the rate carries
per-chain staging where the pure-array modes amortize it away. `depth` is
the pass-slot ring: 1 = stage-then-chain fully serial (the staging-
exposure upper bound); 2 = submit pass `j+1`'s slot, chain pass `j`,
await — the deepest overlap one outstanding transfer per submitter
admits (order 096, the same bound `HMX_WGT_RING`'s ceiling states). The
three-row composition (mode 0, this at depth 1, this at depth 2) splits
compute vs staging vs overlap through `bench_compose_floor_ticks` /
`bench_overlap_efficiency_ppm` with no witness-layout change; read
against CHAINRATE_PLAYBOOK's branch table, it is the direct
feeder-quiesce-vs-array-active discrimination the parked split
instrument could not deliver: an overlapped row near mode 0 says the
in-dispatch gap lives in the cross-thread schedule (branch a); an
overlapped row near the production in-dispatch rate says per-chain
delivery itself is the gap (the d/e family)."""

comptime BENCH_MODE_STORES_OUT = 4
"""Movement, VTCM->DDR by HVX stores: the UNPACK linear-copy form as a
rate, and the store-side twin of `membw_probe`'s LOADS mode."""

comptime BENCH_STATUS_NOT_BUILT: UInt32 = 1 << 16
"""This build has no array arm (`SKYE_HMX_EMIT_MMA` off) — the mirror of
`MXMEM_STATUS_NOT_BUILT`, under the same rule: a rate must never be
reported beside it."""

comptime BENCH_STATUS_NO_VTCM: UInt32 = 1 << 17
"""No VTCM grant, or the geometry does not fit the one there is."""

comptime BENCH_STATUS_NO_GRANT: UInt32 = 1 << 18
"""The HMX unit was refused. The low 16 bits carry the lock's own status."""

comptime BENCH_STATUS_GEOMETRY: UInt32 = 1 << 19
"""The geometry cannot produce a rate the bench is willing to report —
`bench_geometry_status` names each branch's reason."""

comptime BENCH_STATUS_BAD_MODE: UInt32 = 1 << 20
"""The mode names no arm. Refused rather than defaulted: a defaulted mode
would run a plausible wrong arm and report it under the asked-for label."""

comptime BENCH_STATUS_WRONG_ARM: UInt32 = 1 << 21
"""The pass echoed an arm other than the mode's, or ran without the issued
bit. The low 16 bits carry the echo. This is the routing tripwire: a build
that silently swapped the arms reads as a FAIL, not as a plausible number."""

comptime BENCH_STATUS_DEPTH: UInt32 = 1 << 22
"""The depth names no runnable schedule: below 1, past `iters`, past
`BENCH_MOVE_DEPTH_MAX`, or any value but 1 on a mode with no depth axis.
Refused rather than clamped: a clamped depth would report a rate under a
label the run did not execute -- the mislabeled-arm class again."""


@always_inline
def bench_array_macs(k_tiles: Int, iters: Int) -> UInt64:
    """MACs `iters` passes over a `k_tiles`-crouton pair stream execute.

    One pass is one 32x32 output tile accumulated over `32 * k_tiles` of
    depth: `32 * 32 * 32 * k_tiles` MACs. Geometry-derived so the GMAC/s
    denominator has nothing measured in it — the discipline
    `hmx_port_probe_bytes` set. 64-bit because this part's `Int` is 32 bits
    and a timeable run can execute more MACs than that holds.
    """
    return (
        UInt64(HMX_TILE)
        * UInt64(HMX_TILE)
        * UInt64(HMX_TILE)
        * UInt64(k_tiles)
        * UInt64(iters)
    )


@always_inline
def bench_panel_bytes[dtype: DType = HMX_DTYPE](k_tiles: Int) -> Int:
    """Bytes one staged operand panel holds: the movement leg's unit.

    What one weight-ring slot receives per column tile, and therefore the
    transfer size the production stripe's uDMA path actually issues (before
    the feeder split below).
    """
    return k_tiles * HmxTileBytes[dtype]


@always_inline
def bench_stage_chunk_bytes[
    dtype: DType = HMX_DTYPE
](k_tiles: Int, feeders: Int) -> Int:
    """Feeder 0's share of one panel, in bytes: the production staging grain.

    The stripe splits a panel's croutons across feeders by `hmx_work_share`,
    so this is the chunk size a `membw_probe` sweep must include for its uDMA
    row to price the transfer the stripe actually submits — at
    (`BENCH_K_PROD`, 4 feeders) that is 96 croutons = 196,608 bytes.
    """
    return (
        hmx_work_share(k_tiles, 1, feeders)
        - hmx_work_share(k_tiles, 0, feeders)
    ) * HmxTileBytes[dtype]


comptime BENCH_MOVE_FEEDERS = 4
"""Feeders the movement grain divides a panel across: the production
stripe's compute-thread count. Fixed rather than read from the launch so
the chunk IS the production staging grain at every participant mask — the
mask varies concurrency and nothing else, `membw_probe`'s axis
discipline."""

comptime BENCH_MOVE_MAX_THREAD_BYTES = 256 * 1024 * 1024
"""Ceiling on one participant's moved bytes. This part's `Int` is 32 bits
and the per-thread DDR regions sit at `thread * chunk * iters`, so the
bound keeps every offset (and the four regions' total) far inside signed
32-bit addressing while holding an order more than a ~16 MiB measurement
row needs."""

comptime BENCH_MOVE_DEPTH_MAX = 255
"""Ceiling on the uDMA mode's ring depth (window item 9's knob): the
usable depth of the shim's default descriptor ring, capacity 256 per
thread with one slot open (`SKYE_UDMA_RING_CAPACITY`, `qshim_qurt.c`).
Past it the shim's pop-on-full backpressure would cap the requested depth
SILENTLY and the row would mislabel its arm, so the bench refuses instead.
An A/B build that lowers the ring capacity must sweep below its own
ceiling for the same reason; the pre-registered sweep tops out at 16
(`UDMA_RING_DESIGN.md` section 8.2.3), far under either."""


@always_inline
def bench_move_chunk_bytes(k_tiles: Int) -> Int:
    """One transfer of the movement modes: feeder 0's share of the panel.

    `bench_stage_chunk_bytes` at the production feeder count, so
    `k_tiles = 384` (K=12288) prices exactly the 196,608-byte transfer the
    stripe submits — the write-side row DRAFT-210's movement leg needs.
    Zero when the first share is empty, which the geometry gate refuses.
    """
    return bench_stage_chunk_bytes[HMX_DTYPE](k_tiles, BENCH_MOVE_FEEDERS)


comptime BENCH_INSTRIPE_DEPTH_MAX = 2
"""Ceiling on the in-stripe pass-slot ring. One outstanding transfer per
submitter (order 096) makes 2 the deepest schedule one thread can run —
submit-ahead one slot while the chain consumes the other. A deeper value
would serialize silently at the shim and report a rate under a label the
run did not execute, the `BENCH_MOVE_DEPTH_MAX` refusal class."""


@always_inline
def bench_instripe_slot_bytes(k_tiles: Int) -> Int:
    """Bytes one in-stripe pass slot stages: the act and wgt panels,
    adjacent, so ONE linear descriptor moves a whole pass's operands (the
    type-0 length field carries 24 bits — 16 MiB — so the production-shape
    slot of ~1.5 MiB is one transfer, not a chain)."""
    return 2 * bench_panel_bytes[HMX_DTYPE](k_tiles)


@always_inline
def bench_instripe_ddr_bytes(k_tiles: Int, iters: Int) -> UInt64:
    """Bytes the in-stripe walk reads from DDR: `iters` distinct slots.

    Pass `j` stages source slot `j` and the walk never revisits a line, so
    the rate cannot be an L2 read-hit rate re-labelled —
    `bench_move_bytes`' no-revisit discipline, read-side."""
    return UInt64(bench_instripe_slot_bytes(k_tiles)) * UInt64(iters)


@always_inline
def bench_move_bytes(k_tiles: Int, iters: Int) -> UInt64:
    """Bytes one participant moves: `iters` chunks, each line written once.

    The walk never wraps — thread `t`'s region is `t * chunk * iters` bytes
    into the sink — so a rate against this cannot be an L2 write-hit rate
    re-labelled: nothing is revisited.
    """
    return UInt64(bench_move_chunk_bytes(k_tiles)) * UInt64(iters)


@always_inline
def bench_array_program_record[
    space: AddressSpace, dtype: DType, //
](scratch: HmxScratch[space, dtype]):
    """Programs the identity record for either arm, ONCE, outside the bracket.

    The fp16 identity is 1.0; the integer identity is 512.0
    (`HMX_INT8_IDENTITY_SCALE`) because the integer convert divides by 512
    where the fp16 datapath does not — the `/512` lesson, pinned here so the
    bench cannot repeat the all-zero probes (`HMX_INT8_SCOPE.md` section 2
    H1). Outside the timed loop, mirroring `mxmem_port_probe`: the record
    LOAD instruction is inside every pass either way (`_hmx_mma_open` /
    `_hmx_int8_mma_open`), and that fixed per-pass cost is what the
    `hmx_port_probe_min_tiles` refusal bounds at 2%.
    """
    comptime if dtype == HMX_INT8_DTYPE:
        hmx_int8_program_record(
            scratch.record, HMX_INT8_IDENTITY_SCALE, Int32(0)
        )
    else:
        hmx_program_identity_record(scratch.record)


@always_inline
def bench_array_pass[
    space: AddressSpace,
    dtype: DType,
    //,
    form: Int = BENCH_FORM_DEEP,
    emit_hmx: Bool = HMX_EMIT_MMA,
](scratch: HmxScratch[space, dtype], k_tiles: Int) -> UInt32:
    """One pure-array pass: a complete open/stream/store MMA, either arm.

    THE arm configuration, stated once so the host gate and the device shim
    instantiate identical arms rather than two transcriptions of one idea.
    The fp16 arm is the port probe's own chain, unchanged; the integer arm is
    `hmx_int8_mma_panel` — the same open/stream/store shape over byte
    croutons, plain-form by toolchain bound.

    Returns:
        The arm echo: `BENCH_ARM_*` OR `BENCH_ARM_ISSUED` when the
        instructions were issued rather than modelled.
    """
    comptime if dtype == HMX_INT8_DTYPE:
        comptime assert form == BENCH_FORM_PLAIN, (
            "the int8 `:deep` form does not assemble in this tree"
            " (hmx_int8.mojo form table; MC support is PR #438) — a deep"
            " int8 arm would time the compiler's gap, not the port"
        )
        hmx_int8_mma_panel[emit_hmx=emit_hmx](scratch, k_tiles)
        comptime if emit_hmx:
            return BENCH_ARM_W8A8 | BENCH_ARM_ISSUED
        else:
            return BENCH_ARM_W8A8
    else:
        hmx_port_probe_chain[deep=form == BENCH_FORM_DEEP, emit_hmx=emit_hmx](
            scratch.out,
            scratch.record,
            HmxOperands[space, dtype](scratch.act, scratch.wgt, k_tiles),
        )
        comptime if emit_hmx:
            return BENCH_ARM_FP16 | BENCH_ARM_ISSUED
        else:
            return BENCH_ARM_FP16


@always_inline
def bench_expected_arm[dtype: DType]() -> UInt32:
    """The arm echo a healthy pass of this dtype returns, before the issued
    bit. What the device shim compares the live echo against."""
    comptime if dtype == HMX_INT8_DTYPE:
        return BENCH_ARM_W8A8
    else:
        return BENCH_ARM_FP16


comptime BENCH_FILL_SCALE = Scalar[DType.float16](0.015625)
"""1/64, exact in fp16: keeps a `BENCH_K_PROD`-deep fp16 accumulation finite
(worst-case coherent sum ~3.5e3 against fp16's 65,504)."""


def bench_fill_fp16_croutons[
    space: AddressSpace, dtype: DType, //
](panel: CroutonPanelOf[dtype, space], tiles: Int, salt: Int):
    """Fills fp16 croutons with a position-distinct, fp16-exact pattern.

    Distinct along EVERY axis (tile, slab, lane, sub), coprime with the tile
    side and the slab pairing, because a fill uniform along any axis is blind
    to a permutation along it — the 184F lesson that made every earlier
    "identity is bit-exact" claim vacuous. `salt` separates the two operands
    so they cannot be swapped and still agree.

    Generic over `dtype` in SIGNATURE only, so `bench_array_stage`'s arms
    type-check eagerly (the trap `kernels/hmx-w8a8-integer-arm` names); the
    values are the fp16 arm's, computed in fp16 and cast.
    """
    comptime SPLIT = HmxSplit[dtype]
    for t in range(tiles):
        for sp in range(HMX_TILE // SPLIT):
            for lane in range(HMX_TILE):
                for sub in range(SPLIT):
                    var mix = (3 * t + 5 * sp + 7 * lane + 11 * sub + salt) % 13
                    panel.store[1](
                        Coord(t, sp, lane, sub),
                        (Scalar[HMX_DTYPE](mix - 6) * BENCH_FILL_SCALE).cast[
                            dtype
                        ](),
                    )


def bench_fill_byte_croutons[
    space: AddressSpace, dtype: DType, //
](panel: CroutonPanelOf[dtype, space], tiles: Int, salt: Int, period: Int,):
    """Fills byte croutons with small non-negative values in `[0, period)`.

    Position-distinct for the same reason the fp16 fill is. Non-negative so
    one fill serves both operands (`activation.ub` reads unsigned, `weight.b`
    signed — the same bits) and so an identity-record readout never clamps at
    zero and hides distinctness; the caller sizes `period` so the accumulator
    stays inside the saturating byte convert at the gated depth. Generic over
    `dtype` for the reason the fp16 fill is.
    """
    comptime SPLIT = HmxSplit[dtype]
    for t in range(tiles):
        for sp in range(HMX_TILE // SPLIT):
            for lane in range(HMX_TILE):
                for sub in range(SPLIT):
                    var mix = (3 * t + 5 * sp + 7 * lane + 11 * sub + salt) % 13
                    panel.store[1](
                        Coord(t, sp, lane, sub),
                        Scalar[dtype](mix % period),
                    )


comptime BENCH_ACT_SALT = 0
comptime BENCH_WGT_SALT = 3
"""The two operands' fill salts: distinct so a swapped pair cannot agree."""

comptime BENCH_INT8_ACT_PERIOD = 3
comptime BENCH_INT8_WGT_PERIOD = 2
"""The byte fills' value periods, distinct per operand. At the host gate's
depth they keep every accumulator inside the saturating convert; at bench
depths the readout saturates, which a TIMING row never reads — the fold
stays a non-zero wrote-at-all witness either way."""


@always_inline
def bench_zero_readout[
    space: AddressSpace, dtype: DType, //
](panel: CroutonPanelOf[dtype, space]):
    """Zeroes the one-crouton readout tile, either arm.

    Scalar stores by design: the device stages before taking the HMX grant
    and a thread holding the array must execute no HVX. Zeroed BEFORE the
    run because the readout tile is VTCM the launch does not own between
    dispatches — a non-zero fold over a tile nobody zeroed would be last
    dispatch's (`mojo_npu_mxmem_port_probe`'s rule, stated where both arms
    share it).
    """
    comptime SPLIT = HmxSplit[dtype]
    for sp in range(HMX_TILE // SPLIT):
        for lane in range(HMX_TILE):
            for sub in range(SPLIT):
                panel.store[1](Coord(0, sp, lane, sub), Scalar[dtype](0))


@always_inline
def bench_array_stage[
    space: AddressSpace, dtype: DType, //
](scratch: HmxScratch[space, dtype], k_tiles: Int):
    """Stages one pure-array case: fills, a zeroed readout, the record.

    Everything the bracket must not contain, stated once so the device shim
    and the host gate stage identically rather than as two transcriptions.
    """
    comptime if dtype == HMX_INT8_DTYPE:
        bench_fill_byte_croutons(
            scratch.act, k_tiles, BENCH_ACT_SALT, BENCH_INT8_ACT_PERIOD
        )
        bench_fill_byte_croutons(
            scratch.wgt, k_tiles, BENCH_WGT_SALT, BENCH_INT8_WGT_PERIOD
        )
    else:
        bench_fill_fp16_croutons(scratch.act, k_tiles, BENCH_ACT_SALT)
        bench_fill_fp16_croutons(scratch.wgt, k_tiles, BENCH_WGT_SALT)
    bench_zero_readout(scratch.out)
    bench_array_program_record(scratch)


@always_inline
def bench_array_stream[
    space: AddressSpace,
    dtype: DType,
    //,
    form: Int,
    emit_hmx: Bool = HMX_EMIT_MMA,
](scratch: HmxScratch[space, dtype], k_tiles: Int, iters: Int) -> UInt32:
    """Issues `iters` complete passes: THE TIMED BODY of the pure-array modes.

    Scalar by construction — the loop is a counter and each pass is the
    inline chain — the rule a thread holding the unit lives under.

    Returns:
        The last pass's arm echo. The shim refuses a run whose echo is not
        the mode's expected arm with the issued bit.
    """
    var arm: UInt32 = 0
    for _ in range(iters):
        arm = bench_array_pass[form=form, emit_hmx=emit_hmx](scratch, k_tiles)
    return arm


@always_inline
def bench_readout_fold[
    space: AddressSpace, dtype: DType, //
](panel: CroutonPanelOf[dtype, space]) -> UInt32:
    """Folds one readout tile into a witness word, by SUM of bit patterns.

    The control that says the array wrote at all, for either arm —
    `hmx_port_probe_fold` generalized over the operand type. SUM and not XOR
    for the reason that function's docstring carries: an XOR fold over
    identical values is constantly zero, and a control that cannot pass is
    not a control. A checksum must not round, so it sums bit patterns, never
    values; 1024 elements at 0xFFFF cannot wrap the word.
    """
    comptime SPLIT = HmxSplit[dtype]
    var fold: UInt32 = 0
    for slab in range(HMX_TILE):
        for lane in range(HMX_TILE):
            var value = panel.load[1](
                Coord(0, slab // SPLIT, lane, slab % SPLIT)
            )
            comptime if dtype == HMX_INT8_DTYPE:
                fold += value.cast[DType.uint32]()
            else:
                fold += bitcast[DType.uint16, 1](value).cast[DType.uint32]()
    return fold


@always_inline
def bench_compose_floor_ticks(
    compute_ticks: UInt64, movement_ticks: UInt64
) -> UInt64:
    """The (a, b) composition floor: a perfectly overlapped pipeline's wall.

    A stripe that stages tile `j + 1` while multiplying tile `j` can go no
    faster than its slower stream, so the floor is the MAX of the two legs'
    QTimer spans for the same work — not their sum, which is the fully
    serialized wall `bench_compose_serial_ticks` reports for scale.
    """
    if compute_ticks > movement_ticks:
        return compute_ticks
    return movement_ticks


@always_inline
def bench_compose_serial_ticks(
    compute_ticks: UInt64, movement_ticks: UInt64
) -> UInt64:
    """The no-overlap wall: the two legs back to back."""
    return compute_ticks + movement_ticks


@always_inline
def bench_overlap_efficiency_ppm(
    floor_ticks: UInt64, measured_ticks: UInt64
) -> Int:
    """Overlap efficiency in parts per million: `floor / measured`.

    1,000,000 is a pipeline that hides its staging completely; the DRAFT-210
    pre-registration is >= 800,000 or the pipeline structure is the finding.
    Above 1,000,000 is reported as computed, not clamped — it means the
    composition legs ran slower than the pipeline achieves (a finding about
    the legs, and hiding it would be the placebo class). Zero-measured is
    answered with 0 rather than a division fault: a bracket that measured
    nothing has no efficiency.
    """
    if measured_ticks == 0:
        return 0
    return Int((floor_ticks * 1_000_000) // measured_ticks)


@always_inline
def bench_min_k_tiles[dtype: DType = HMX_DTYPE]() -> Int:
    """The pure-array refusal floor, per dtype: `hmx_port_probe_min_tiles`.

    29 croutons at fp16 and 32 at int8 — below these the record + readout
    share the geometry-derived denominator excludes exceeds 2%, and the
    kernel refuses rather than reporting a plausible understated rate (the
    false-contradiction trap `hmx_port_probe_min_tiles`'s docstring names).
    `BENCH_K_SMALL` (2 croutons) sits below both by design; its pure-array
    row is a refusal the order pre-registers.
    """
    return hmx_port_probe_min_tiles[dtype]()


def bench_geometry_status(mode: Int, k_tiles: Int, iters: Int) -> UInt32:
    """The selector's refusal ladder: 0, or the status bit that names why.

    Host-gateable on purpose — the device shim runs THIS function, so the
    refusal boundary the gate proves is the boundary silicon enforces. Every
    branch converts a plausible wrong number into a named refusal: an
    unknown mode would run a defaulted arm under the asked-for label; a
    pure-array geometry below `bench_min_k_tiles` understates its rate past
    the documented 2% and fires DRAFT-210's contradiction rows on an
    instrument artifact; a byte count past 2^32 reports modulo the witness
    word; an empty movement chunk times nothing; a movement walk past
    `BENCH_MOVE_MAX_THREAD_BYTES` overflows 32-bit region offsets.
    """
    if mode < BENCH_MODE_FP16_DEEP or mode > BENCH_MODE_FP16_INSTRIPE:
        return BENCH_STATUS_BAD_MODE
    if iters < 1:
        return BENCH_STATUS_GEOMETRY
    if mode == BENCH_MODE_UDMA_OUT or mode == BENCH_MODE_STORES_OUT:
        if bench_move_chunk_bytes(k_tiles) < 1:
            return BENCH_STATUS_GEOMETRY
        if bench_move_bytes(k_tiles, iters) > UInt64(
            BENCH_MOVE_MAX_THREAD_BYTES
        ):
            return BENCH_STATUS_GEOMETRY
        return 0
    if mode == BENCH_MODE_INT8_PLAIN:
        if k_tiles < bench_min_k_tiles[HMX_INT8_DTYPE]():
            return BENCH_STATUS_GEOMETRY
        if hmx_port_probe_bytes[HMX_INT8_DTYPE](k_tiles, iters) > UInt64(
            0xFFFFFFFF
        ):
            return BENCH_STATUS_GEOMETRY
        return 0
    if mode == BENCH_MODE_FP16_INSTRIPE:
        if k_tiles < bench_min_k_tiles[HMX_DTYPE]():
            return BENCH_STATUS_GEOMETRY
        if hmx_port_probe_bytes[HMX_DTYPE](k_tiles, iters) > UInt64(
            0xFFFFFFFF
        ):
            return BENCH_STATUS_GEOMETRY
        if bench_instripe_ddr_bytes(k_tiles, iters) > UInt64(
            BENCH_MOVE_MAX_THREAD_BYTES
        ):
            return BENCH_STATUS_GEOMETRY
        return 0
    if k_tiles < bench_min_k_tiles[HMX_DTYPE]():
        return BENCH_STATUS_GEOMETRY
    if hmx_port_probe_bytes[HMX_DTYPE](k_tiles, iters) > UInt64(0xFFFFFFFF):
        return BENCH_STATUS_GEOMETRY
    return 0


def bench_case_status(
    mode: Int, k_tiles: Int, iters: Int, depth: Int
) -> UInt32:
    """The selector's whole refusal ladder: geometry, then the depth
    contract.

    The one function the device shim calls (window item 9 widened the wire
    to four scalars), so the boundary the host gate proves is the boundary
    silicon enforces. `bench_geometry_status` keeps its pre-depth signature
    for its existing callers and is composed here, not duplicated.

    The depth contract: the ring depth axis exists only on the uDMA mode --
    `1 <= depth <= min(iters, BENCH_MOVE_DEPTH_MAX)`. A prime deeper than
    the walk never reaches steady state (and `iters == depth` is the
    pre-registered prologue-in-isolation row: prime plus drain, zero steady
    chunks). Every other mode has no depth axis and accepts only 1 -- a
    knob that did nothing under an asked-for label would be the defaulted
    arm the mode gate already refuses.
    """
    var refused = bench_geometry_status(mode, k_tiles, iters)
    if refused != 0:
        return refused
    if mode == BENCH_MODE_UDMA_OUT:
        if depth < 1 or depth > iters or depth > BENCH_MOVE_DEPTH_MAX:
            return BENCH_STATUS_DEPTH
        return 0
    if mode == BENCH_MODE_FP16_INSTRIPE:
        # The in-stripe ring: `iters == depth` never reaches steady state
        # here either, but unlike the uDMA walk a two-pass run at depth 2
        # is exactly the prologue-plus-drain row, so only the hard bounds
        # refuse.
        if depth < 1 or depth > iters or depth > BENCH_INSTRIPE_DEPTH_MAX:
            return BENCH_STATUS_DEPTH
        return 0
    if depth != 1:
        return BENCH_STATUS_DEPTH
    return 0
