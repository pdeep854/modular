"""HVX rotary-position-embedding (RoPE) kernel for the on-device Hexagon NPU.

Decode-shaped: applies RoPE to `num_rows` Q/K head vectors of `head_size`
elements each, one precomputed cos/sin row per position. This is the candidate
on-device counterpart of the Skye production RoPE
(`max/kernels/src/qaic/rope_split_store.mojo`); it reproduces that kernel's
rotary convention exactly:

- HF `rotate_half`: the real half `x[0:half]` is paired with the imaginary half
  `x[half:2*half]`, and `out[i] = a*cos - b*sin`, `out[i+half] = a*sin + b*cos`
  with `a = x[i]`, `b = x[i+half]` (this is the Gemma4 production path). An
  `interleaved` variant pairing neighbours `(2p, 2p+1)` is also provided, which
  is the convention FLUX.2 selects. Both vectorize; the pairing decides only how
  the two elements reach the lanes.
- `freqs` holds cos/sin interleaved per pair: row `pos` is
  `[cos_0, sin_0, cos_1, sin_1, ...]`, `head_size` entries (== `half` pairs).
- RoPE math runs in fp32; I/O is fp16 (bf16 works through the generic helper).

It follows the on-device HVX idioms of the verified `rms_norm`/`gemv`/`softmax`
example kernels: the HVX lane count is the 128-byte vector width over the element
size, NOT `simd_width_of[dtype]()` (whose host-target default reports the host
width and silently emits scalar code); fp16 loads widen to fp32 for the
arithmetic and narrow again on store; a scalar tail covers a half-dim that is not
a multiple of the vector width; and the `@export ... abi("C")` entry stripes rows
across the core's compute threads and reports the compiled vector width through
`witness`. The `witness` layout matches the gemv/softmax kernels so the on-device
dispatch check can prove both vectorization and the by-index thread split the
same way: `[0]` compiled vector width, `[1]` fp32 accumulator lane count, `[2]`
participant count the shim reports, `[3 + launch_index]` the RTOS thread each
launch index ran on.

Addressing is the tile model's: the extents the C-ABI boundary hands down become
layouts once, at the seam, and every access below that line is a coordinate into
one. No row stride is restated, `row * head_size` appears nowhere, and no
signature below the seam carries an extent -- a table or a row is handed over as
one value that already knows its own shape. The translation is KB
`kernels/tile-model-porting-a-pointer-kernel-to-tiletensor`;
that the machine code is unchanged by it is checked, not assumed, by
`plugins/internal-skye/utils/kernel_asm_compare.sh`. What the tile model does NOT
express here is the interleaved convention's pairing -- see `rope_apply_row`.

The pure math lives in `rope_apply_row`, which carries no device dependencies:
the off-device test drives it directly on the host, where the same source lowers
to host SIMD, so the test exercises the identical arithmetic the HVX build runs.

MAX's own Mojo/HVX only -- no HMX, no vendor archive.
"""

from layout.coord import All, Coord
from layout.tile_layout import row_major
from layout.tile_tensor import TileTensor

from std.ffi import external_call
from std.math import ceildiv
from std.sys import align_of, size_of

comptime HVX_VECTOR_SIZE = 128
"""HVX vector register width in bytes (`+hvx-length128b` in build.sh)."""

comptime NUM_COMPUTE_THREADS: UInt32 = 4
"""Compute threads a core exposes; the fifth is the part's DMA/HMX thread."""

comptime IO_DTYPE = DType.float16
"""The fp16 instantiation dtype; the I/O element type."""

comptime IO_DTYPE_BF = DType.bfloat16
"""The bf16 instantiation dtype. Same 64-lane width; the rotation is fp32 either
way, so what the bf16 entries confirm on silicon is v81's qfloat lowering of the
bf16 load, deinterleave and store."""

comptime FREQ_DTYPE = DType.float32
"""Precision of the cos/sin inputs; fp32 keeps the rotation exact (generic)."""

comptime IO_VW = HVX_VECTOR_SIZE // size_of[IO_DTYPE]()
"""HVX fp16 lanes per vector: 128 B / 2 B = 64. See the module docstring on why
this is not `simd_width_of`."""


@always_inline
def rope_vector_width[dtype: DType]() -> Int:
    """HVX lanes per vector at `dtype`: 64 16-bit elements in a 128-byte vector.

    The documented register width over the element size rather than
    `simd_width_of`, which resolves against the host and scalarizes silently.
    """
    return Int(HVX_VECTOR_SIZE) // size_of[dtype]()


# ===-----------------------------------------------------------------------===#
# The views the boundary's pointers are wrapped in.
# ===-----------------------------------------------------------------------===#


comptime RowMajor2D[dtype: DType] = TileTensor[
    dtype, type_of(row_major(Coord(0, 0))), origin=MutUnsafeAnyOrigin
]
"""A `(rows, head_size)` table: `x`, `dst`, and the `freqs` cos/sin table.

Both extents are dynamic, so the layout carries runtime values; what it makes
compile-time is the *rank* and the row-major relationship between them, which is
what removes `row * head_size` from every access.
"""


comptime RowMajor1D[dtype: DType] = TileTensor[
    dtype, type_of(row_major(Coord(0))), origin=MutUnsafeAnyOrigin
]
"""The `(num_rows,)` view for `positions`, which is a vector, not a table."""


comptime Row[dtype: DType] = type_of(
    RowMajor2D[dtype](
        ptr=UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin](
            unsafe_from_address=0
        ),
        layout=row_major(Coord(0, 0)),
    ).slice(0, All)
)
"""One row of a `RowMajor2D`, as `.slice(i, All)` returns it.

Spelled through `type_of` because `.slice()`'s result type is built by filtering
the parent's shape and stride type lists and has no shorter name.

`rope_apply_row` takes three of these rather than three tables and three row
indices: the multiply that selects a row is then resolved once, where the row is
chosen, which is the row-level reading of KB
`kernels/tiletensor-resolve-the-base-once-not-per-access`.
"""


comptime FREQ_ROWS_UNKNOWN = -1
"""Stands in for the `freqs` row count, which the boundary does not carry.

`freqs` is `[max_position, head_size]` and only `head_size` is on the wire:
`positions` selects the row and the caller owns the bound. A row-major layout
derives its strides right-to-left, so the leading extent never enters an address
-- `.slice(pos, All)` keeps only the row stride, which comes from `head_size` --
and this value is therefore inert. It is spelled the way the layout package
spells an unknown extent rather than invented as a plausible-looking count, so
that nothing reads it as a claim about how many rows `freqs` has.

`layout.int_tuple`'s `UNKNOWN_VALUE` is not imported for it: that module reaches
`layout.math` -> `max.algorithm.reduction`, which does not resolve under the
on-device import path (KB `kernels/tiletensor-operation-surface-on-hexagon`).
"""


# ===-----------------------------------------------------------------------===#
# Device services (spelled out for the bare-metal Hexagon triple).
# ===-----------------------------------------------------------------------===#
#
# The same `qshim` calls the stdlib overlay lowers to, respelled here because a
# standalone `mojo build --emit=object` for `hexagon-unknown-unknown-elf` cannot
# resolve the overlay (its target gate is exact against the musl triple). Only
# the ones the row-striping and the witness need are carried over.

comptime QSHIM_QUERY_THREAD_ID: UInt32 = 0
comptime QSHIM_QUERY_NUM_THREADS: UInt32 = 1


@always_inline
def _qshim_query(attribute: UInt32) -> Int64:
    """Reads one device attribute, as the overlay's `_qshim_query_u8` does."""
    var value: Int64 = 0
    _ = external_call["qshimQuery", UInt32](attribute, UnsafePointer(to=value))
    return value


@always_inline
def _npu_thread_idx() -> Int:
    """Index of this thread within the core, 0-based (narrowed to 8 bits like
    the overlay's `qshim_get_thread_id`)."""
    return Int(UInt8(_qshim_query(QSHIM_QUERY_THREAD_ID)))


@always_inline
def _qurt_thread_id() -> UInt32:
    """The RTOS thread this code runs on, read only to tell threads apart.

    A launch index is what the shim reports; an index per participant proves
    nothing on its own about how many threads sit behind those indices, so the
    witness records the distinct RTOS ids too.
    """
    return external_call["qurt_thread_get_id", UInt32]()


# ===-----------------------------------------------------------------------===#
# Pure HVX RoPE math (target-independent; the test drives this directly).
# ===-----------------------------------------------------------------------===#


@always_inline
def rope_apply_row[
    dtype: DType,
    freq_dtype: DType,
    //,
    interleaved: Bool,
    W: Int,
](x_row: Row[dtype], dst_row: Row[dtype], freq_row: Row[freq_dtype]):
    """Applies RoPE to one head vector, `x` -> `dst` (fp32 math).

    The three rows are views the caller has already selected: the head vector,
    where it goes, and the `freqs` row for its position. The head dim is not a
    separate argument -- a row view carries its own extent, and a caller passing
    `head_size / 2` alongside it would be restating that extent with nothing
    checking the two against each other. `dst_row` must not alias `x_row`: the
    `rotate_half` vector loop writes the imaginary half while the real half is
    still being read across iterations. `interleaved=False` is
    HF `rotate_half`; `interleaved=True` pairs neighbours `(2p, 2p+1)`. Both run
    on the vector units, and the pairing is what decides how the operands reach
    them: `rotate_half`'s two elements sit `half` apart, so they are two plain
    loads, while interleaved's sit adjacent, so the run splits into even and odd
    lanes on the way in and recombines on the way out.

    That split stays a LANE PERMUTATION and is deliberately not expressed as a
    stride-2 layout, which is the other thing the tile model would let one write
    here. `deinterleave` is one `vshuff`-class op on a vector already in a
    register; a strided view would put the pairing in the addressing and issue
    two half-rate loads for the run this reads once (KB
    `kernels/ondevice-rope-interleaved-vectorization`, which records the same
    permutation as what took this branch from 0 HVX ops to 61).
    """
    # The scalar tails state the element's own alignment. `TileTensor`'s default
    # is `align_of[...] if is_gpu() else 1` (`tile_tensor.mojo:799, :838`), and a
    # 1-byte claim on a scalar fp16 access makes the compiler assemble the
    # element from two `memub` plus a shift and an or: leaving it at the default
    # cost +52 and +48 instructions on the two entry points, all of it here. The
    # pointer form this replaced asserted natural alignment implicitly, and the
    # rows do satisfy it, so the claim is unchanged; only its spelling is.
    comptime io_align = align_of[Scalar[dtype]]()
    comptime freq_align = align_of[Scalar[freq_dtype]]()

    # Both conventions split the head vector into `head_dim / 2` pairs; which
    # two elements form a pair is what differs. The row is the only thing that
    # knows `head_dim`, so this is where it is read.
    var half = x_row.num_elements() // 2

    comptime if interleaved:
        # Neighbours are lanes of one contiguous run, so the pair splits with
        # the same `deinterleave` the `rotate_half` branch already applies to
        # `freqs`, and the rotated halves recombine through its inverse.
        var p0 = 0
        while p0 + W <= half:
            var fpair = freq_row.load[2 * W](Coord(2 * p0))
            var cs = fpair.deinterleave()
            var cosv = rebind[SIMD[freq_dtype, W]](cs[0]).cast[DType.float32]()
            var sinv = rebind[SIMD[freq_dtype, W]](cs[1]).cast[DType.float32]()
            var xpair = x_row.load[2 * W](Coord(2 * p0))
            var ri = xpair.deinterleave()
            var re = rebind[SIMD[dtype, W]](ri[0]).cast[DType.float32]()
            var im = rebind[SIMD[dtype, W]](ri[1]).cast[DType.float32]()
            var out_re = (re * cosv - im * sinv).cast[dtype]()
            var out_im = (re * sinv + im * cosv).cast[dtype]()
            dst_row.store[2 * W](Coord(2 * p0), out_re.interleave(out_im))
            p0 += W

        # Scalar tail for any remainder (half not a multiple of W).
        while p0 < half:
            var cos = Float32(
                freq_row.load[1, alignment=freq_align](Coord(2 * p0))
            )
            var sin = Float32(
                freq_row.load[1, alignment=freq_align](Coord(2 * p0 + 1))
            )
            var x_re = Float32(x_row.load[1, alignment=io_align](Coord(2 * p0)))
            var x_im = Float32(
                x_row.load[1, alignment=io_align](Coord(2 * p0 + 1))
            )
            dst_row.store[1, alignment=io_align](
                Coord(2 * p0), Scalar[dtype](x_re * cos - x_im * sin)
            )
            dst_row.store[1, alignment=io_align](
                Coord(2 * p0 + 1), Scalar[dtype](x_re * sin + x_im * cos)
            )
            p0 += 1
    else:
        # `freqs` stores cos/sin interleaved per pair, so a contiguous `2*W` load
        # deinterleaved into even/odd lanes yields the cos and sin vectors.
        var p0 = 0
        while p0 + W <= half:
            var fpair = freq_row.load[2 * W](Coord(2 * p0))
            var cs = fpair.deinterleave()
            # deinterleave reports size (2*W)/2; rebind to the equal width W.
            var cosv = rebind[SIMD[freq_dtype, W]](cs[0]).cast[DType.float32]()
            var sinv = rebind[SIMD[freq_dtype, W]](cs[1]).cast[DType.float32]()
            var a = x_row.load[W](Coord(p0)).cast[DType.float32]()
            var b = x_row.load[W](Coord(p0 + half)).cast[DType.float32]()
            dst_row.store[W](Coord(p0), (a * cosv - b * sinv).cast[dtype]())
            dst_row.store[W](
                Coord(p0 + half), (a * sinv + b * cosv).cast[dtype]()
            )
            p0 += W

        # Scalar tail for any remainder (half not a multiple of W).
        while p0 < half:
            var cos = Float32(
                freq_row.load[1, alignment=freq_align](Coord(2 * p0))
            )
            var sin = Float32(
                freq_row.load[1, alignment=freq_align](Coord(2 * p0 + 1))
            )
            var a = Float32(x_row.load[1, alignment=io_align](Coord(p0)))
            var b = Float32(x_row.load[1, alignment=io_align](Coord(p0 + half)))
            dst_row.store[1, alignment=io_align](
                Coord(p0), Scalar[dtype](a * cos - b * sin)
            )
            dst_row.store[1, alignment=io_align](
                Coord(p0 + half), Scalar[dtype](a * sin + b * cos)
            )
            p0 += 1


# ===-----------------------------------------------------------------------===#
# On-device entry points (`@export ... abi("C")`).
#
# Named `hvx_*` like every other authoring module's, not `mojo_*` like the
# manifest's: `NPUKernels.def:99,105` binds the four `mojo_rope_*` symbols that
# `npu_kernel.mojo` defines, and two definitions of one name cannot both be in
# the package. The dispatch path reaches kernels through selector strings
# (`rope_half`, `rope_interleaved`), never through a symbol, so nothing follows
# the rename.
# ===-----------------------------------------------------------------------===#


@always_inline
def _thread_rows(num_rows: Int, thread_id: UInt32) -> Tuple[Int, Int]:
    """The `[start, end)` run of rows this launch index owns.

    Each of the four compute threads owns a contiguous run; the fifth (DMA)
    thread and any launch index past the row count get the empty run `(0, 0)`,
    which the row loop then runs zero times. The split stays hand-written:
    `.distribute()` would give a *cyclic* partition and this part wants blocked
    (KB `kernels/distribute-is-cyclic-this-part-wants-blocked`).

    `thread_id` is an argument rather than a shim read so the partition is
    testable off-device; only `rope_entry` below reads the shim.
    """
    if thread_id >= NUM_COMPUTE_THREADS:
        return (0, 0)
    var rows_per_thread = ceildiv(num_rows, Int(NUM_COMPUTE_THREADS))
    var start_row = Int(thread_id) * rows_per_thread
    if start_row >= num_rows:
        return (0, 0)
    var end_row = start_row + rows_per_thread
    if end_row > num_rows:
        end_row = num_rows
    return (start_row, end_row)


@always_inline
def rope_striped[
    dtype: DType,
    interleaved: Bool,
](
    dst_tile: RowMajor2D[dtype],
    x_tile: RowMajor2D[dtype],
    freq_tile: RowMajor2D[FREQ_DTYPE],
    pos_tile: RowMajor1D[DType.uint32],
    thread_id: UInt32,
):
    """Ropes the rows `thread_id` owns; no cross-thread state, so no barrier.

    The tables arrive as tiles: the seam that turns the C-ABI boundary's pointers
    and extents into layouts is in `rope_entry` below, so `num_rows` and
    `head_size` are not on this signature and cannot disagree with the buffers
    they describe.
    """
    comptime W = rope_vector_width[dtype]()

    # `alignment` for the same reason the tails state it: at the 1-byte default
    # this `uint32` becomes four `memub` plus shifts and an `or`, worth 15 and 20
    # instructions on the two entry points.
    comptime pos_align = align_of[Scalar[DType.uint32]]()

    var rows = _thread_rows(Int(x_tile.dim[0]()), thread_id)
    for row in range(rows[0], rows[1]):
        var pos = Int(pos_tile.load[1, alignment=pos_align](Coord(row)))
        rope_apply_row[interleaved=interleaved, W=W](
            x_tile.slice(row, All),
            dst_tile.slice(row, All),
            freq_tile.slice(pos, All),
        )


@always_inline
def rope_entry[
    dtype: DType,
    interleaved: Bool,
](
    dst: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin],
    x: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin],
    freqs: UnsafePointer[Scalar[FREQ_DTYPE], MutUnsafeAnyOrigin],
    positions: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    num_rows: Int,
    head_size: Int,
):
    """RoPE over the rows this launch index owns.

    This is the seam. The C-ABI signatures that reach it are the frozen
    `NPUKernels.def` contract and are not ours to move, so the extents they
    carry become layouts here -- once -- and no signature under this line takes
    an extent again. Once is also what the addressing needs: re-resolving a tile
    per access instead of per construction costs 4x the vector work and forces
    spills (KB `kernels/tiletensor-resolve-the-base-once-not-per-access`).

    `witness` keeps its pointer where the tables do not -- it is a side channel
    of scalars at fixed slots, not a tensor -- and reports what the output
    cannot: the compiled vector width, the fp32 accumulator lane count, the
    participant count the shim reports, and the RTOS thread each launch index
    ran on. A build that resolved the width against the host would still rope
    correctly, at a lane count that says the vector units never ran.
    """
    comptime W = rope_vector_width[dtype]()

    var thread_idx = _npu_thread_idx()

    if thread_idx == 0:
        witness.store(0, UInt32(W))
        witness.store(1, UInt32(SIMD[DType.float32, W]().size))
        witness.store(2, UInt32(_qshim_query(QSHIM_QUERY_NUM_THREADS)))
    witness.store(3 + thread_idx, _qurt_thread_id())

    var heads = row_major(Coord(num_rows, head_size))
    rope_striped[dtype, interleaved=interleaved](
        RowMajor2D[dtype](ptr=dst, layout=heads),
        RowMajor2D[dtype](ptr=x, layout=heads),
        RowMajor2D[FREQ_DTYPE](
            ptr=freqs, layout=row_major(Coord(FREQ_ROWS_UNKNOWN, head_size))
        ),
        RowMajor1D[DType.uint32](
            ptr=positions, layout=row_major(Coord(num_rows))
        ),
        UInt32(thread_idx),
    )


@export
def hvx_rope_half_f16(
    dst: UnsafePointer[Scalar[IO_DTYPE], MutUnsafeAnyOrigin],
    x: UnsafePointer[Scalar[IO_DTYPE], MutUnsafeAnyOrigin],
    freqs: UnsafePointer[Scalar[FREQ_DTYPE], MutUnsafeAnyOrigin],
    positions: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    num_rows: Int,
    head_size: Int,
) abi("C"):
    """RoPE (HF `rotate_half`) over `[num_rows, head_size]` head vectors.

    `x`/`dst` are row-major fp16; `dst` must not alias `x`. `freqs` is
    `[max_position, head_size]` fp32 cos/sin interleaved per pair; `positions[row]`
    picks the freqs row (all heads of one decode token share a position).

    The extents this C-ABI signature carries become layouts in `rope_entry`.
    """
    rope_entry[IO_DTYPE, interleaved=False](
        dst, x, freqs, positions, witness, num_rows, head_size
    )


@export
def hvx_rope_interleaved_f16(
    dst: UnsafePointer[Scalar[IO_DTYPE], MutUnsafeAnyOrigin],
    x: UnsafePointer[Scalar[IO_DTYPE], MutUnsafeAnyOrigin],
    freqs: UnsafePointer[Scalar[FREQ_DTYPE], MutUnsafeAnyOrigin],
    positions: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    num_rows: Int,
    head_size: Int,
) abi("C"):
    """RoPE (interleaved neighbour pairs), the reference's second convention.

    Spelled as in `hvx_rope_half_f16`.
    """
    rope_entry[IO_DTYPE, interleaved=True](
        dst, x, freqs, positions, witness, num_rows, head_size
    )
