from layout.coord import Coord
from layout.tile_layout import row_major
from layout.tile_tensor import TileTensor

from std.ffi import external_call
from std.math import min
from std.sys import align_of, get_defined_int, size_of

from hexagon_ondevice.hvx_permute import (
    HVX_SHUFF_INTERLEAVE,
    hvx_lanes,
    hvx_vshuff,
)

# ===--------------------------------------------------------------------------===
# Transpose + per-output-row bias — the FLUX.2-Klein VAE decode's layout round
# trip (`VAE_CODE_CRITIQUE.md` finding F4), moved on-device.
#
# The host driver (`npu_vae_decoder_dispatch.cpp`) ran two EXACTLY INVERSE
# transposes per conv, on the ARM host, at up to 512 KiB destination stride:
# `scatterConvRows` turns the fused conv kernel's raw `[M, C_out]` result into
# `[C_out, spatial]` (CHW, GroupNorm's channel-major ABI) while adding the
# per-channel bias the conv kernel has no buffer for; `nhwc_from_chw`
# (`conv_im2col.h:335`) turns the NEXT conv's CHW activation back into the
# `[H, W, C_in]` map the fused im2col gather reads. Both are the same operation
# — transpose `[R, N] -> [N, R]`, optionally adding a per-output-row constant —
# so both route through this one kernel instead of a host loop measured at
# 4.206 s + 2.682 s = 6.888 s of a 96 s decode.
#
# This does NOT touch `npu_kernel.mojo`'s `_matmul_microtile` / `_conv_stripe`:
# that nest is tuned to the instruction (`_conv_stripe`'s own comment: "even
# provably dead code added to it moves the schedule"), so folding a transposed
# store into its epilogue would put the ALREADY-SIGNED `conv_im2col` (selector
# 36) and `matmul` (selector 24) selectors back in a re-sign window too. A
# transpose is also a different memory-access shape entirely — the microtile's
# store is one scalar per accumulator, in-bounds and unconditional; this walks
# the WHOLE `[R, N]` extent with neither axis contiguous on both sides of the
# swap — so it earns its own kernel rather than a comptime branch grafted onto
# that one.
#
# Bias is unconditional, never a branch: the leg that wants a pure transpose
# (CHW -> NHWC, the activation pack) passes a zero-filled bias row rather than
# the kernel testing a flag, which keeps this body's instruction count
# independent of which caller reached it — the same reason
# `skye_conv::fused_conv_supported` is a caller-side predicate and not a
# kernel-side branch.
# ===--------------------------------------------------------------------------===

comptime HVX_VECTOR_SIZE = 128
"""Size of an HVX vector in bytes (`+hvx-length128b`). Unused by the scalar
walk below — stated for the same reason every sibling module in this
directory states it: this source also compiles for the host, where deriving
the width from `simd_width_of` would silently test a different target."""

comptime TB_DTYPE = DType.float16
"""The VAE decode's I/O dtype. No bf16 twin: nothing on this port produces a
bf16 activation, and an unused selector is surface a re-sign window has to
carry for no caller."""

comptime NUM_COMPUTE_THREADS: UInt32 = 4
"""Threads of a core that compute. The fifth is the part's DMA thread."""

comptime QSHIM_QUERY_THREAD_ID: UInt32 = 0
comptime QSHIM_QUERY_NUM_THREADS: UInt32 = 1

comptime TB_TILE = 32
"""Block extent over both axes — the same constant `nhwc_from_chw`
(`conv_im2col.h:335-349`) blocks `(c_in, w)` at, for the same reason: a
transpose is never contiguous on both sides of the swap, so blocking is what
keeps the DISFAVORED axis's access inside one cache line's neighborhood
instead of striding the whole extent once per element."""

comptime HMX_TB_VEC = get_defined_int["SKYE_HMX_TB_VEC", 0]()
"""Off (0): the scalar blocked walk below, byte-for-byte the shipped body.
On (1): full 64x64 halfword blocks transpose through the HVX perfect-shuffle
network instead of one `memuh` per element; ragged edges keep the scalar
path. Order 228 measured the scalar walk at 523.3 ms/dispatch — the walk,
not the dispatch, is where selector 41's refusal-by-economics (EVAL_175,
re-confirmed x95) lives, and this arm is the re-price of that walk."""

comptime _TB_VEC_LANES = hvx_lanes[TB_DTYPE]()
"""Halfwords in one HVX vector — the vector block's extent on both axes."""

comptime HMX_TB_STRIPE64 = get_defined_int["SKYE_HMX_TB_STRIPE64", 0]()
"""Off (0): `_thread_rows` splits `N` evenly across the compute threads —
the shipped split, byte-for-byte. On (1): slices are built from 64-row
units whenever at least two exist, so a slice can carry `_tb_vec_block64`;
pairs with `SKYE_HMX_TB_VEC=1`, which gates the vector body this split
feeds. The 242 device legs measured why the even split is wrong at the
decode's biggest tensors: at N=128 it hands every thread a 32-row slice,
the 64x64 block can never fire, and the seven up3 scatters run the scalar
walk at ~300 ms each — 2.13 s of selector 41's 3.274 s slow mode
(`W4_FUSED_EPILOGUE_DESIGN.md` §2, `FIXA_STRIPE64.md` §1)."""

comptime _TB_VEC_ROUNDS = 6
"""log2(64): riffle rounds that transpose a 64x64 halfword block. Composing
the six exchange stages of `vshuff` rotates the 12-bit (vector, lane) index
by one bit per round (`hvx_permute.mojo`'s own note on
`HVX_SHUFF_INTERLEAVE`); six rounds swap the two 6-bit halves outright,
which IS the transpose — no bit-reversal residue. The host twin executes
LLVM's transcribed lane models, so `test_transpose_bias_kernel.mojo` proves
the network byte-for-byte off-device."""


comptime RowMajor2D[dtype: DType] = TileTensor[
    dtype, type_of(row_major(Coord(0, 0))), origin=MutUnsafeAnyOrigin
]
"""The `(rows, cols)` view the C-ABI entry wraps its pointers in. Both extents
are dynamic; what the parameter makes compile-time is the rank, which is what
removes `i * cols` from every access below."""

comptime SCALAR_ALIGN[dtype: DType] = align_of[Scalar[dtype]]()
"""Stated on every scalar access for the reason every sibling module states
it: `TileTensor.load`/`.store` default to 1-byte alignment off a GPU
(`tile_tensor.mojo:798, 838`), and a 1-aligned fp16 access on Hexagon costs a
real instruction (two `memub` plus a shift instead of one `memuh`)."""


@always_inline
def _tb_vec_block64(
    output: RowMajor2D[TB_DTYPE],  # [N, R]
    input: RowMajor2D[TB_DTYPE],  # [R, N]
    bias: RowMajor2D[TB_DTYPE],  # [1, N]
    r0: Int,
    n0: Int,
):
    """One full `64x64` block of the transpose+bias, vectorized.

    Byte movement only differs from the scalar body: the block loads as 64
    row vectors, six riffle rounds of the perfect halfword interleave turn
    row vectors into column vectors (`_TB_VEC_ROUNDS`'s docstring carries
    the index argument), and each column vector then takes the SAME
    widen-add-narrow bias the scalar body applies — fp32 add per element,
    one rounding to fp16 — so agreement with the scalar path is BIT-EXACT
    by construction, and the twin checks it that way.
    """
    comptime assert (
        size_of[Scalar[TB_DTYPE]]() == 2
    ), "the riffle network below is halfword-strided"

    var v = Array[SIMD[TB_DTYPE, _TB_VEC_LANES], _TB_VEC_LANES](
        fill=SIMD[TB_DTYPE, _TB_VEC_LANES]()
    )
    var w = Array[SIMD[TB_DTYPE, _TB_VEC_LANES], _TB_VEC_LANES](
        fill=SIMD[TB_DTYPE, _TB_VEC_LANES]()
    )
    comptime for i in range(_TB_VEC_LANES):
        v[i] = input.load[_TB_VEC_LANES, alignment=SCALAR_ALIGN[TB_DTYPE]](
            Coord(r0 + i, n0)
        )
    comptime for _round in range(_TB_VEC_ROUNDS):
        # comptime index keeps v/w register-resident: a runtime i forces the
        # SIMD arrays to memory and emits ONE rolled vshuff per round (the 242
        # census read +13, not the ~+192 the pricing assumed).
        comptime for i in range(_TB_VEC_LANES // 2):
            var pair = hvx_vshuff[HVX_SHUFF_INTERLEAVE](
                rebind[SIMD[TB_DTYPE, _TB_VEC_LANES]](
                    v[i + _TB_VEC_LANES // 2]
                ),
                rebind[SIMD[TB_DTYPE, _TB_VEC_LANES]](v[i]),
            )
            w[2 * i] = pair.slice[_TB_VEC_LANES]()
            w[2 * i + 1] = pair.slice[
                _TB_VEC_LANES, offset=_TB_VEC_LANES
            ]()
        comptime for i in range(_TB_VEC_LANES):
            v[i] = w[i]
    comptime for j in range(_TB_VEC_LANES):
        var b = bias.load[1, alignment=SCALAR_ALIGN[TB_DTYPE]](
            Coord(0, n0 + j)
        ).cast[DType.float32]()
        output.store[_TB_VEC_LANES, alignment=SCALAR_ALIGN[TB_DTYPE]](
            Coord(n0 + j, r0),
            (v[j].cast[DType.float32]() + b).cast[TB_DTYPE](),
        )


@always_inline
def _transpose_bias_block[
    dtype: DType
](
    output: RowMajor2D[dtype],  # [N, R]
    input: RowMajor2D[dtype],  # [R, N]
    bias: RowMajor2D[dtype],  # [1, N]
    n_start: Int,
    n_end: Int,
):
    """Writes `output[n, r] = fp16(fp32(input[r, n]) + fp32(bias[0, n]))` for
    `n` in `[n_start, n_end)`, over every `r` in `[0, R)`.

    The widen-add-narrow is `addBias`'s own definition
    (`npu_vae_decoder_dispatch.cpp:618`, "shared by the reference and the
    device epilogue so the two cannot drift"), kept here for the same reason:
    an fp16-native add could round differently than the host's fp32 one, and
    the zero-bias (pure transpose) leg still reproduces its input bit-exactly
    through the round trip -- fp16 -> fp32 is exact, and fp32 -> fp16 of a
    value that came from fp16 plus exactly zero rounds to the same bits it
    started from.

    Blocked `TB_TILE x TB_TILE` over `(r, n)`: within one block, `r` is the
    contiguous axis on the read side and `n` is the contiguous axis on the
    write side, so no loop order vectorizes either access, and blocking is
    what keeps the OTHER axis's touch inside one cache line's neighborhood
    rather than striding the whole extent per element — `nhwc_from_chw`'s own
    reason for the same constant.
    """
    var num_r = Int(input.dim[0]())

    # Flag-off this folds to the shipped constant and the vector branch
    # below folds away entirely — the flag-off object is byte-identical,
    # which is what lets the arm certify against the signed body.
    comptime TB_STEP = _TB_VEC_LANES if HMX_TB_VEC != 0 else TB_TILE

    var n0 = n_start
    while n0 < n_end:
        var n1 = min(n0 + TB_STEP, n_end)
        var r0 = 0
        while r0 < num_r:
            var r1 = min(r0 + TB_STEP, num_r)
            comptime if HMX_TB_VEC != 0 and dtype == TB_DTYPE:
                if n1 - n0 == _TB_VEC_LANES and r1 - r0 == _TB_VEC_LANES:
                    _tb_vec_block64(
                        rebind[RowMajor2D[TB_DTYPE]](output),
                        rebind[RowMajor2D[TB_DTYPE]](input),
                        rebind[RowMajor2D[TB_DTYPE]](bias),
                        r0,
                        n0,
                    )
                    r0 = r1
                    continue
            for n in range(n0, n1):
                var b = bias.load[1, alignment=SCALAR_ALIGN[dtype]](
                    Coord(0, n)
                ).cast[DType.float32]()
                for r in range(r0, r1):
                    var x = input.load[1, alignment=SCALAR_ALIGN[dtype]](
                        Coord(r, n)
                    ).cast[DType.float32]()
                    output.store[1, alignment=SCALAR_ALIGN[dtype]](
                        Coord(n, r), (x + b).cast[dtype]()
                    )
            r0 = r1
        n0 = n1


@always_inline
def transpose_bias_2d[
    dtype: DType
](
    output: RowMajor2D[dtype],
    input: RowMajor2D[dtype],
    bias: RowMajor2D[dtype],
):
    """Transposes `input[R, N]` into `output[N, R]`, adding `bias[n]` to every
    element of output row `n`. The whole-tile caller; `transpose_bias_striped`
    below is the same body over one launch thread's `N` slice.
    """
    _transpose_bias_block[dtype](output, input, bias, 0, Int(output.dim[0]()))


# ===--------------------------------------------------------------------------===
# Device services + the launch-striped entry.
#
# The `external_call`s are the same `qshim` queries every sibling module in
# this directory spells out, repeated here because a standalone
# `mojo build --emit=object` for the bare-metal Hexagon triple cannot reach the
# overlay. They are undefined until the link binds them against
# `qshim_qurt.c`; a host build that never calls the entry leaves them
# unreferenced, which is what lets the algorithm above be tested off-device.
# ===--------------------------------------------------------------------------===


@always_inline
def _qshim_query(attribute: UInt32) -> Int64:
    """Reads one device attribute, as the overlay's `_qshim_query_u8` does."""
    var value: Int64 = 0
    _ = external_call["qshimQuery", UInt32](attribute, UnsafePointer(to=value))
    return value


@always_inline
def _npu_thread_idx() -> Int:
    """Index of this thread within the core, 0-based (narrowed to 8 bits)."""
    return Int(UInt8(_qshim_query(QSHIM_QUERY_THREAD_ID)))


@always_inline
def _qurt_thread_id() -> UInt32:
    """The RTOS thread this code is running on, read only to tell threads
    apart — a launch index alone proves nothing about how many threads back
    those indices."""
    return external_call["qurt_thread_get_id", UInt32]()


@always_inline
def _transpose_bias_witness[
    dtype: DType
](
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    thread_idx: Int,
):
    """Reports the participant count and the RTOS thread each launch index ran
    on — the softmax/silu/groupnorm witness layout. There is no SIMD width to
    report: the walk above is scalar by construction, not by a fallback that
    failed to vectorize."""
    if thread_idx == 0:
        witness.store(0, UInt32(0))
        witness.store(1, UInt32(0))
        witness.store(2, UInt32(_qshim_query(QSHIM_QUERY_NUM_THREADS)))
    witness.store(3 + thread_idx, _qurt_thread_id())


@always_inline
def _thread_rows(num_rows: Int, thread_id: UInt32) -> Tuple[Int, Int]:
    """The `[start, end)` run of `N` rows this launch index owns.

    Each of the four compute threads owns a contiguous run; the fifth (DMA)
    thread and any launch index past the row count get the empty run `(0,
    0)`. The split stays hand-written rather than `.distribute()`, which
    would give a cyclic partition (KB
    `kernels/distribute-is-cyclic-this-part-wants-blocked`) — the same choice
    `silu_swiglu_kernel.mojo` and `groupnorm_kernel.mojo` make.

    Under `SKYE_HMX_TB_STRIPE64`, runs are whole 64-row units when at least
    two exist (`num_rows >= 128`), ceil-split across the threads with the
    last unit-owning thread absorbing the ragged tail — coverage is exact
    for every `num_rows`, threads past the unit count idle. At N=128 that
    is two threads on one vector block each: two idle, but a 64-wide vector
    slice beats a 32-wide scalar one by ~25x per element on the same write
    stride (the up2 vector scatters price it — `FIXA_STRIPE64.md` §1.2).
    Every 64-multiple N >= 256 splits identically to the even split, so the
    flag's behavioral delta in the production census is the N=128 scatters
    alone. Below 128 the even split stands: no slicing can make a vector
    block where no full unit exists.
    """
    if thread_id >= NUM_COMPUTE_THREADS:
        return (0, 0)
    comptime if HMX_TB_STRIPE64 != 0:
        if num_rows >= 2 * _TB_VEC_LANES:
            var units = num_rows // _TB_VEC_LANES
            var per_thread_units = (
                units + Int(NUM_COMPUTE_THREADS) - 1
            ) // Int(NUM_COMPUTE_THREADS)
            var start_unit = Int(thread_id) * per_thread_units
            if start_unit >= units:
                return (0, 0)
            var end_unit = start_unit + per_thread_units
            if end_unit >= units:
                return (start_unit * _TB_VEC_LANES, num_rows)
            return (start_unit * _TB_VEC_LANES, end_unit * _TB_VEC_LANES)
    var per_thread = (num_rows + Int(NUM_COMPUTE_THREADS) - 1) // Int(
        NUM_COMPUTE_THREADS
    )
    var start_row = Int(thread_id) * per_thread
    if start_row >= num_rows:
        return (0, 0)
    var end_row = start_row + per_thread
    if end_row > num_rows:
        end_row = num_rows
    return (start_row, end_row)


@always_inline
def transpose_bias_striped[
    dtype: DType
](
    output: RowMajor2D[dtype],
    input: RowMajor2D[dtype],
    bias: RowMajor2D[dtype],
    thread_id: UInt32,
):
    """Runs `transpose_bias_2d`'s body over the `N` (output-row) slice
    `thread_id` owns. Partitioned on the WRITE axis, never the read axis: two
    threads writing the same output row would race, and every `(n, r)` pair
    this body touches has exactly one owning `n`. The slice geometry is
    `_thread_rows`'s contract: even element split, or 64-row units under
    `SKYE_HMX_TB_STRIPE64` — either way the runs are disjoint and cover
    `[0, N)`.
    """
    var rows = _thread_rows(Int(output.dim[0]()), thread_id)
    _transpose_bias_block[dtype](output, input, bias, rows[0], rows[1])


@always_inline
def _transpose_bias_entry[
    dtype: DType
](
    output: RowMajor2D[dtype],
    input: RowMajor2D[dtype],
    bias: RowMajor2D[dtype],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
):
    """Transpose+bias over the rows this launch index owns. The seam
    `npu_kernel.mojo`'s `mojo_transpose_bias_f16` C-ABI entry calls."""
    var thread_idx = _npu_thread_idx()
    _transpose_bias_witness[dtype](witness, thread_idx)
    transpose_bias_striped[dtype](output, input, bias, UInt32(thread_idx))
