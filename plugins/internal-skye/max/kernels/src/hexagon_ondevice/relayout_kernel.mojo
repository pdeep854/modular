from layout.coord import Coord
from layout.tile_layout import row_major
from layout.tile_tensor import TileTensor

from std.ffi import external_call
from std.math import min
from std.sys import align_of, size_of

from hexagon_ondevice.hvx_permute import (
    HVX_SHUFF_INTERLEAVE,
    hvx_lanes,
    hvx_vshuff,
)

# ===--------------------------------------------------------------------------===
# Batched 2-D relayout: `output[b, j, i] = input[b, i, j]` -- the serve graph's
# NHWC <-> NCHW materializations moved on-device.
#
# The host-walk census (`HOST_WALK_CENSUS.md`, branch `mac/host-walk-census`)
# prices 77 surviving `mogg.call.materialize` permutes at 14.36 s of a
# 17.824 s non-dispatch pool: the VAE decode is NCHW-authored with NHWC device
# convs, and every conv wrap's permuted view falls to `_copy_view_to_dense`'s
# per-element scalar odometer (fitted 11.9 ns/elem). Every one of those sites
# is, per batch index, one 2-D transpose: `[C, H*W] <-> [H*W, C]` at fp16 --
# which is selector 41's access pattern without the bias row and without the
# hand-driver's `[R, N]`-only framing. This kernel is that one primitive,
# batched, both directions through the same body (the direction IS the choice
# of `(I, J)`).
#
# A NEW row rather than routing to selector 41: 41's contract carries a
# mandatory bias row (a zero row would be one more host staging allocation per
# materialize, on the path this row exists to shorten), and its vector arm
# sits behind `SKYE_HMX_TB_VEC` because its scalar walk is the SIGNED body --
# flipping the define moves certified bytes (EVAL_175's economics were priced
# on that walk). A new row ships the vector arm as the only body, refused by
# name on older skels -- the port's standing degradation mode.
#
# The transpose core ADOPTS the in-tree idiom rather than inventing one: the
# 64x64 halfword riffle network of `transpose_bias_kernel._tb_vec_block64`
# (six `vshuff(-2)` exchange rounds -- `hvx_permute.mojo` carries the lane
# model and the LLVM provenance), the same instruction family the reference
# corpora move layouts with (ggml-hexagon `hmx_interleave_cols_to_tiles`, the
# qaic attention `transposeHMX` butterflies). Order 242's device legs priced
# the 64-wide vector slice at ~25x the scalar walk per element on the same
# write stride; order 228 priced the scalar walk itself at 523.3 ms on an
# up-block scatter.
# ===--------------------------------------------------------------------------===

comptime HVX_VECTOR_SIZE = 128
"""Size of an HVX vector in bytes (`+hvx-length128b`). Stated for the reason
every sibling module states it: this source also compiles for the host twin,
where deriving the width from `simd_width_of` would silently test a different
target."""

comptime RL_DTYPE = DType.float16
"""The serve graph's activation dtype. The census's 77 sites are all fp16;
a wider family waits for a caller, as `TB_DTYPE`'s rule says."""

comptime NUM_COMPUTE_THREADS: UInt32 = 4
"""Threads of a core that compute. The fifth is the part's DMA thread."""

comptime QSHIM_QUERY_THREAD_ID: UInt32 = 0
comptime QSHIM_QUERY_NUM_THREADS: UInt32 = 1

comptime RL_TILE = 32
"""Scalar-edge block extent over both axes -- `TB_TILE`'s constant and its
reason: a transpose is never contiguous on both sides of the swap, so blocking
keeps the disfavored axis's access inside one cache line's neighborhood."""

comptime _RL_VEC_LANES = hvx_lanes[RL_DTYPE]()
"""Halfwords in one HVX vector (64) -- the vector block's extent on both axes."""

comptime _RL_VEC_ROUNDS = 6
"""log2(64): riffle rounds that transpose a 64x64 halfword block. Composing
the six exchange stages of `vshuff` rotates the 12-bit (vector, lane) index by
one bit per round; six rounds swap the two 6-bit halves outright, which IS the
transpose (`transpose_bias_kernel._TB_VEC_ROUNDS` carries the full argument).
The host twin executes LLVM's transcribed lane models, so the harness proves
the network byte-for-byte off-device."""

comptime RELAYOUT_PATH_RAN: UInt32 = 1
"""Witness word 7: the walk ran over the declared extents."""

comptime RELAYOUT_PATH_REFUSED: UInt32 = 2
"""Witness word 7: an extent was non-positive; nothing was written."""


comptime RlRowMajor2D[dtype: DType] = TileTensor[
    dtype, type_of(row_major(Coord(0, 0))), origin=MutUnsafeAnyOrigin
]
"""The `(rows, cols)` view the C-ABI entry wraps its pointers in. Both extents
are dynamic; the parameter makes the rank compile-time, which removes
`i * cols` from every access below."""

comptime SCALAR_ALIGN[dtype: DType] = align_of[Scalar[dtype]]()
"""Stated on every scalar access: `TileTensor.load`/`.store` default to 1-byte
alignment off a GPU, and a 1-aligned fp16 access on Hexagon costs a real
instruction (two `memub` plus a shift instead of one `memuh`)."""


@always_inline
def _rl_vec_block64(
    output: RlRowMajor2D[RL_DTYPE],  # [J, I]
    input: RlRowMajor2D[RL_DTYPE],  # [I, J]
    i0: Int,
    j0: Int,
):
    """One full `64x64` block of the transpose, through the riffle network.

    `_tb_vec_block64`'s byte movement without the bias arm: the block loads
    as 64 row vectors, six riffle rounds of the perfect halfword interleave
    turn row vectors into column vectors, and each column vector stores as
    one output row segment. Pure movement -- agreement with the scalar body
    is bit-exact by construction, and the twin checks it that way.
    """
    comptime assert (
        size_of[Scalar[RL_DTYPE]]() == 2
    ), "the riffle network below is halfword-strided"

    var v = Array[SIMD[RL_DTYPE, _RL_VEC_LANES], _RL_VEC_LANES](
        fill=SIMD[RL_DTYPE, _RL_VEC_LANES]()
    )
    var w = Array[SIMD[RL_DTYPE, _RL_VEC_LANES], _RL_VEC_LANES](
        fill=SIMD[RL_DTYPE, _RL_VEC_LANES]()
    )
    comptime for i in range(_RL_VEC_LANES):
        v[i] = input.load[_RL_VEC_LANES, alignment=SCALAR_ALIGN[RL_DTYPE]](
            Coord(i0 + i, j0)
        )
    comptime for _round in range(_RL_VEC_ROUNDS):
        # comptime index keeps v/w register-resident: a runtime i forces the
        # SIMD arrays to memory and emits ONE rolled vshuff per round (the
        # 242 census read on selector 41's arm).
        comptime for i in range(_RL_VEC_LANES // 2):
            var pair = hvx_vshuff[HVX_SHUFF_INTERLEAVE](
                rebind[SIMD[RL_DTYPE, _RL_VEC_LANES]](
                    v[i + _RL_VEC_LANES // 2]
                ),
                rebind[SIMD[RL_DTYPE, _RL_VEC_LANES]](v[i]),
            )
            w[2 * i] = pair.slice[_RL_VEC_LANES]()
            w[2 * i + 1] = pair.slice[_RL_VEC_LANES, offset=_RL_VEC_LANES]()
        comptime for i in range(_RL_VEC_LANES):
            v[i] = w[i]
    comptime for j in range(_RL_VEC_LANES):
        output.store[_RL_VEC_LANES, alignment=SCALAR_ALIGN[RL_DTYPE]](
            Coord(j0 + j, i0), v[j]
        )


@always_inline
def _relayout_block[
    dtype: DType
](
    output: RlRowMajor2D[dtype],  # [J, I]
    input: RlRowMajor2D[dtype],  # [I, J]
    j_start: Int,
    j_end: Int,
):
    """Writes `output[j, i] = input[i, j]` for `j` in `[j_start, j_end)`,
    over every `i` in `[0, I)`.

    Full `64x64` blocks take the riffle network; ragged edges take the
    blocked scalar walk. Unlike selector 41 there is no build define between
    the two: this row was never signed with a scalar-only body, so the vector
    arm IS the body and the scalar walk is its edge case.
    """
    var num_i = Int(input.dim[0]())

    comptime RL_STEP = _RL_VEC_LANES if dtype == RL_DTYPE else RL_TILE

    var j0 = j_start
    while j0 < j_end:
        var j1 = min(j0 + RL_STEP, j_end)
        var i0 = 0
        while i0 < num_i:
            var i1 = min(i0 + RL_STEP, num_i)
            comptime if dtype == RL_DTYPE:
                if j1 - j0 == _RL_VEC_LANES and i1 - i0 == _RL_VEC_LANES:
                    _rl_vec_block64(
                        rebind[RlRowMajor2D[RL_DTYPE]](output),
                        rebind[RlRowMajor2D[RL_DTYPE]](input),
                        i0,
                        j0,
                    )
                    i0 = i1
                    continue
            for j in range(j0, j1):
                for i in range(i0, i1):
                    output.store[1, alignment=SCALAR_ALIGN[dtype]](
                        Coord(j, i),
                        input.load[1, alignment=SCALAR_ALIGN[dtype]](
                            Coord(i, j)
                        ),
                    )
            i0 = i1
        j0 = j1


@always_inline
def relayout_2d[
    dtype: DType
](output: RlRowMajor2D[dtype], input: RlRowMajor2D[dtype]):
    """Transposes `input[I, J]` into `output[J, I]`. The whole-tile caller;
    `relayout_striped` below is the same body over one launch thread's `J`
    slice."""
    _relayout_block[dtype](output, input, 0, Int(output.dim[0]()))


@always_inline
def _rl_thread_rows(num_rows: Int, thread_id: UInt32) -> Tuple[Int, Int]:
    """The `[start, end)` run of output rows this launch index owns.

    64-row units whenever at least two exist, else the even element split --
    selector 41's `SKYE_HMX_TB_STRIPE64` split as the ONLY split, because its
    lesson is baked in from the first signing: at `J = 128` an even split
    hands every thread a 32-row slice and the vector block can never fire
    (`FIXA_STRIPE64.md` section 1). The fifth (DMA) thread and any launch
    index past the row count get the empty run.
    """
    if thread_id >= NUM_COMPUTE_THREADS:
        return (0, 0)
    if num_rows >= 2 * _RL_VEC_LANES:
        var units = num_rows // _RL_VEC_LANES
        var per_thread_units = (units + Int(NUM_COMPUTE_THREADS) - 1) // Int(
            NUM_COMPUTE_THREADS
        )
        var start_unit = Int(thread_id) * per_thread_units
        if start_unit >= units:
            return (0, 0)
        var end_unit = start_unit + per_thread_units
        if end_unit >= units:
            return (start_unit * _RL_VEC_LANES, num_rows)
        return (start_unit * _RL_VEC_LANES, end_unit * _RL_VEC_LANES)
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
def relayout_striped[
    dtype: DType
](
    output: RlRowMajor2D[dtype],
    input: RlRowMajor2D[dtype],
    thread_id: UInt32,
):
    """Runs `relayout_2d`'s body over the output-row slice `thread_id` owns.

    Partitioned on the WRITE axis, never the read axis: two threads writing
    the same output row would race, and every `(j, i)` this body touches has
    exactly one owning `j`. The runs are disjoint and cover `[0, J)`
    (`_rl_thread_rows`'s contract).
    """
    var rows = _rl_thread_rows(Int(output.dim[0]()), thread_id)
    _relayout_block[dtype](output, input, rows[0], rows[1])


# ===--------------------------------------------------------------------------===
# Device services + the launch-striped entry. The `external_call`s are the
# same `qshim` queries every sibling module spells out, because a standalone
# `mojo build --emit=object` for the bare-metal Hexagon triple cannot reach
# the overlay; a host build that never calls the entry leaves them
# unreferenced, which is what lets the algorithm above be tested off-device.
# ===--------------------------------------------------------------------------===


@always_inline
def _qshim_query(attribute: UInt32) -> Int64:
    """Reads one device attribute, as the overlay's `_qshim_query_u8` does."""
    var value: Int64 = 0
    _ = external_call["qshimQuery", UInt32](attribute, Pointer(to=value))
    return value


@always_inline
def _npu_thread_idx() -> Int:
    """Index of this thread within the core, 0-based (narrowed to 8 bits)."""
    return Int(UInt8(_qshim_query(QSHIM_QUERY_THREAD_ID)))


@always_inline
def _qurt_thread_id() -> UInt32:
    """The RTOS thread this code is running on, read only to tell threads
    apart."""
    return external_call["qurt_thread_get_id", UInt32]()


@always_inline
def _relayout_witness(
    witness: Pointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    thread_idx: Int,
):
    """The participant count and per-launch-index RTOS ids -- the
    transpose_bias witness layout, bounded to the compute threads (the P5
    M2.4 rule the two conv entries ride this window for missing)."""
    if thread_idx == 0:
        witness.unsafe_store(0, UInt32(0))
        witness.unsafe_store(1, UInt32(0))
        witness.unsafe_store(2, UInt32(_qshim_query(QSHIM_QUERY_NUM_THREADS)))
    if thread_idx < Int(NUM_COMPUTE_THREADS):
        witness.unsafe_store(3 + thread_idx, _qurt_thread_id())


@always_inline
def _relayout_run[
    dtype: DType
](
    output: Pointer[Scalar[dtype], MutUnsafeAnyOrigin],
    input: Pointer[Scalar[dtype], MutUnsafeAnyOrigin],
    witness: Pointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    batch: Int,
    rows: Int,
    cols: Int,
    thread_idx: Int,
):
    """Batched transpose over the rows this launch index owns: for every
    `b`, `output[b][j][i] = input[b][i][j]` with `input` `[batch, rows,
    cols]` and `output` `[batch, cols, rows]`, both dense.

    Word 7 echoes what ran rather than leaving the host to assume it
    (`RELAYOUT_PATH_*`): a non-positive extent refuses by name and writes
    nothing -- the host stub owns shape agreement, and a refusal the guard
    can catch beats a zero-trip walk that reads as success.

    Split from `_relayout_entry` so the harness drives every launch index
    without linking the qshim overlay -- the transpose_bias module's own
    host-twin arrangement, with the thread index made an argument.
    """
    if batch <= 0 or rows <= 0 or cols <= 0:
        if thread_idx == 0:
            witness.unsafe_store(7, RELAYOUT_PATH_REFUSED)
        return
    if thread_idx == 0:
        witness.unsafe_store(7, RELAYOUT_PATH_RAN)

    var plane = rows * cols
    for b in range(batch):
        relayout_striped[dtype](
            RlRowMajor2D[dtype](
                ptr=output.unsafe_offset(b * plane),
                layout=row_major(Coord(cols, rows)),
            ),
            RlRowMajor2D[dtype](
                ptr=input.unsafe_offset(b * plane),
                layout=row_major(Coord(rows, cols)),
            ),
            UInt32(thread_idx),
        )


@always_inline
def _relayout_entry[
    dtype: DType
](
    output: Pointer[Scalar[dtype], MutUnsafeAnyOrigin],
    input: Pointer[Scalar[dtype], MutUnsafeAnyOrigin],
    witness: Pointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    batch: Int,
    rows: Int,
    cols: Int,
):
    """The seam `npu_kernel.mojo`'s `mojo_relayout_f16` C-ABI entry calls:
    reads this launch index off the device, stamps the witness, runs."""
    var thread_idx = _npu_thread_idx()
    _relayout_witness(witness, thread_idx)
    _relayout_run[dtype](
        output, input, witness, batch, rows, cols, thread_idx
    )
