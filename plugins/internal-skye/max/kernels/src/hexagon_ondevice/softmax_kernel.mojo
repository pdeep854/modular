from layout.coord import All, Coord
from layout.tile_layout import row_major
from layout.tile_tensor import TileTensor
from std.ffi import external_call
from std.math import ceildiv
from std.memory import bitcast
from std.sys import align_of, bit_width_of, size_of
from std.utils.numerics import min_or_neg_inf

# All arithmetic runs in fp32 regardless of the fp16/bf16 I/O type. The row sum
# in particular needs it: a length-N all-equal row sums to N, and an fp16
# accumulator saturates past 2048 (fp16 represents integers exactly only up to
# that), silently truncating the sum for N >= 4096. fp32 keeps softmax stable
# across the shapes decode actually sees.
comptime ACCUM = DType.float32

comptime HVX_VECTOR_SIZE = 128
"""Size of an HVX vector in bytes (`+hvx-length128b`)."""


comptime RowMajor2D[dtype: DType] = TileTensor[
    dtype, type_of(row_major(Coord(0, 0))), origin=MutUnsafeAnyOrigin
]
"""The `(num_rows, row_length)` view the C-ABI entries' pointers are wrapped in.

Both extents are dynamic, so the layout carries runtime values; what it makes
compile-time is the *rank* and the row-major relationship between them, which is
what removes `r * row_length` from every access.

The C-ABI boundary is frozen and is not ours to move, so the wrap happens one
line inside it, in each `@export` below -- and only there. Re-resolving a tile
per row or per access instead costs 4x the vector work and forces spills (KB
`kernels/tiletensor-resolve-the-base-once-not-per-access`).
"""


@always_inline
def reduce_max[
    dtype: DType, width: Int
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Horizontal max across all lanes; every lane holds the result.

    A rotate-and-max tree, not `SIMD.reduce_max()`: the hand-rolled form lowers
    to `V6_*` HVX ops on Hexagon, where the builtin reduction scalarizes.
    """
    var reduced = x
    comptime for i in range(8, -1, -1):
        comptime if (1 << i) < width:
            reduced = max(reduced, reduced.rotate_right[1 << i]())
    return reduced


@always_inline
def reduce_add[
    dtype: DType, width: Int
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Horizontal sum across all lanes; every lane holds the result."""
    var reduced = x
    comptime for i in range(8, -1, -1):
        comptime if (1 << i) < width:
            reduced += reduced.rotate_right[1 << i]()
    return reduced


# ===--------------------------------------------------------------------------===
# Vector exponential -- `std.math.exp` SCALARIZES on `hexagon-unknown-unknown-elf`
# at every width: a single `exp(SIMD[f32, 64])` compiles to 64 `vinsert`/
# `vextract` pairs and 64 out-of-line `floorf` calls, which is what this kernel's
# two exp passes over `row_max`-shifted input were emitting (130 `floorf`, 256
# `vinsert`, measured off the shipped object -- `FACTS.md:454`).
#
# `_exp_hvx` is the attention kernel's vector exp (M4.0b, `attention_kernel.mojo`),
# duplicated here rather than imported: `npu_kernel.mojo`'s softmax section states
# the reason this file does not take a dependency on the attention module for it
# -- a numerics change belongs with its own device confirm, not with an
# extraction. Every operation below is a lane-wise fp32 multiply/add, a lane-wise
# int32 add/shift, or a reinterpretation, so no cross-lane traffic can introduce
# the interleaved-lane class KB `kernels/hvx-fold-lane-order-audit` gates.
#
# Domain here is exactly what `_exp_hvx` was built for: `v - row_max` and
# `x - row_max` are non-positive by construction (`row_max` is the row's max),
# so this port widens nothing -- unlike SiLU's, which needs the numerically
# stable `exp(-|x|)` split to stay in this same domain (`silu_swiglu_kernel.mojo`).
# ===--------------------------------------------------------------------------===

comptime EXP_LOG2_E = Scalar[ACCUM](1.44269504088896340736)
"""Base change log2(e), which turns the reduction into an exponent field."""

comptime EXP_LN2_HI = Scalar[ACCUM](0.693359375)
"""Leading part of ln2. Exactly representable in fp32 (0x3F318000), so
`n * EXP_LN2_HI` is exact and the subtraction below cancels without error."""

comptime EXP_LN2_LO = Scalar[ACCUM](-2.12194440e-4)
"""Tail of ln2, i.e. `ln2 - EXP_LN2_HI`. Carrying it separately is what keeps the
reduced argument accurate at the top of the range, where `n` is ~128."""

comptime EXP_RNE_MAGIC = Scalar[ACCUM](12582912.0)
"""1.5 * 2^23. Adding it forces round-to-nearest-even into the fp32 mantissa
and subtracting it back leaves the rounded integer, for |x| < 2^22."""

comptime EXP_RNE_MAGIC_BITS = Int32(0x4B400000)
"""Bit pattern of `EXP_RNE_MAGIC`. With the magic add the float's low bits ARE
the integer, so the integer `n` is one int subtract away and no fp-to-int
convert (which this target lowers out of line) is needed."""

comptime EXP_MIN_ARG = Scalar[ACCUM](-87.3)
"""Clamp below. `exp(-87.3)` is 1.4e-38, the smallest normal fp32: one ulp
lower and the exponent field would underflow to a denormal or wrap."""


@always_inline
def _exp_hvx[width: Int](x: SIMD[ACCUM, width]) -> SIMD[ACCUM, width]:
    """`exp(x)` for `x <= 0`, in lane-wise fp32 and int32 ops only.

    Verbatim twin of `attention_kernel.mojo`'s `_exp_hvx`; see there for the
    accuracy argument (degree-6 Horner, ~1.2e-7 relative truncation).
    """
    var xc = max(x, SIMD[ACCUM, width](EXP_MIN_ARG))
    var t = xc * SIMD[ACCUM, width](EXP_LOG2_E) + SIMD[ACCUM, width](
        EXP_RNE_MAGIC
    )
    var n = t - SIMD[ACCUM, width](EXP_RNE_MAGIC)
    var n_int = bitcast[DType.int32, width](t) - SIMD[DType.int32, width](
        EXP_RNE_MAGIC_BITS
    )
    var r = (xc - n * SIMD[ACCUM, width](EXP_LN2_HI)) - n * SIMD[ACCUM, width](
        EXP_LN2_LO
    )

    var p = SIMD[ACCUM, width](1.0 / 720.0)
    p = p * r + SIMD[ACCUM, width](1.0 / 120.0)
    p = p * r + SIMD[ACCUM, width](1.0 / 24.0)
    p = p * r + SIMD[ACCUM, width](1.0 / 6.0)
    p = p * r + SIMD[ACCUM, width](0.5)
    p = p * r + SIMD[ACCUM, width](1.0)
    p = p * r + SIMD[ACCUM, width](1.0)

    var pow2 = bitcast[ACCUM, width](
        (n_int + SIMD[DType.int32, width](127)) << 23
    )
    return p * pow2


@always_inline
def _softmax_rows[
    in_dtype: DType
](
    src: RowMajor2D[in_dtype],
    dst: RowMajor2D[in_dtype],
    start_row: Int,
    end_row: Int,
):
    """Softmaxes rows `[start_row, end_row)` of the tiles.

    The row range is the only thing that distinguishes the whole-tile caller from
    the launch-striped one, so both run this body rather than two transcriptions
    of it. See `softmax_2d` for the arithmetic.

    Each row of `src` is softmaxed independently into the matching row of `dst`
    (the decode case is a single row). I/O is `in_dtype` (fp16/bf16); the
    reduction and exponential run in fp32.

    Addressing is the tile model's: the extents arrive folded into the tiles'
    layout, and every access is a coordinate into it. No row stride is restated
    and `r * row_length` appears nowhere. Taking tiles rather than a pointer
    pair plus `num_rows`/`row_length` means there is no second copy of the
    geometry that could disagree with the one the accesses actually use.

    `src` and `dst` are required to carry the same extents; the seam builds both
    from one layout.
    """
    # One 128-byte HVX register's worth of I/O elements (64 for fp16/bf16),
    # stated from the register geometry rather than taken from `simd_width_of`.
    # KB `kernels/tiletensor-vector-width-state-it-explicitly` retired the
    # blanket form of that rule (the descriptor bug is fixed, #114/#146); what
    # survives is the reason that applies here -- `test_softmax.mojo` compiles
    # this same source for the HOST, where the query correctly returns 8, so a
    # derived width would test a different kernel than it ships.
    comptime LANES = HVX_VECTOR_SIZE // size_of[in_dtype]()
    comptime assert (
        LANES * bit_width_of[in_dtype]() == HVX_VECTOR_SIZE * 8
    ), "LANES must be exactly one HVX vector register"

    # Stated on the scalar tail accesses below. `TileTensor.load`/`.store`
    # default their alignment to `align_of[...] if is_gpu() else 1`
    # (`tile_tensor.mojo:798, :838`), where `UnsafePointer` defaults to the
    # element's own unconditionally. At the 1-byte default the compiler
    # assembles a scalar fp16 element from two `memub` plus a shift and an or,
    # and the matching store defeats the tail loop's post-increment addressing.
    # Rows of a `Scalar[in_dtype]` buffer are element-aligned by construction,
    # so this is a true statement rather than a hint (KB
    # `kernels/tiletensor-off-gpu-alignment-default-degrades-scalar-tails`).
    #
    # The vector accesses deliberately keep the default: HVX emits the unaligned
    # `vmemu` for a 128-byte access either way, so there is nothing to win, and
    # a row view is not 128-aligned in general.
    comptime SCALAR_ALIGN = align_of[Scalar[in_dtype]]()

    # Read back off the layout the seam folded them into, rather than carried
    # alongside it. `dim` answers in the layout's own index type -- int64 for a
    # dynamic layout in the generic address space (`tile_tensor.mojo:3956`) --
    # and the loop arithmetic below wants the `Int` the C ABI was handed.
    var row_length = Int(src.dim[1]())

    var vectorized_end = row_length - (row_length % LANES)

    for r in range(start_row, end_row):
        # A rank-1 row view, not the 2-D tile plus a row index: slicing resolves
        # `r * row_length` once, where the row is chosen.
        var in_row = src.slice(r, All)
        var out_row = dst.slice(r, All)

        # Pass 1: row max. Seed with -inf so the scalar tail alone is correct
        # when row_length < LANES (no full vector iterations run).
        var max_vec = SIMD[ACCUM, LANES](min_or_neg_inf[ACCUM]())
        for j in range(0, vectorized_end, LANES):
            max_vec = max(max_vec, in_row.load[LANES](Coord(j)).cast[ACCUM]())
        var row_max = reduce_max(max_vec)[0]
        for j in range(vectorized_end, row_length):
            row_max = max(
                row_max,
                in_row.load[1, alignment=SCALAR_ALIGN](Coord(j)).cast[ACCUM]()[
                    0
                ],
            )

        # Pass 2: sum of exp(x - max). `_exp_hvx` replaces the scalarizing
        # `std.math.exp` (measured off the shipped object, `FACTS.md:454`: 137
        # out-of-line calls in `mojo_softmax_f16`); the argument is non-positive
        # by construction, `_exp_hvx`'s native domain, so this is a straight
        # substitution, not a numerics widening.
        var max_bcast = SIMD[ACCUM, LANES](row_max)
        var sum_vec = SIMD[ACCUM, LANES](0)
        for j in range(0, vectorized_end, LANES):
            var v = in_row.load[LANES](Coord(j)).cast[ACCUM]()
            sum_vec += _exp_hvx(v - max_bcast)
        var row_sum = reduce_add(sum_vec)[0]
        for j in range(vectorized_end, row_length):
            var x = in_row.load[1, alignment=SCALAR_ALIGN](Coord(j)).cast[
                ACCUM
            ]()[0]
            row_sum += _exp_hvx[1](SIMD[ACCUM, 1](x - row_max))[0]

        # Pass 3: normalize. exp is recomputed rather than staged through a
        # scratch buffer so no fp16 rounding lands on the intermediate. The
        # normalizing divide stays scalar (one `row_sum` per row, broadcast) --
        # it is not the per-lane divide that scalarizes; only the reduction's
        # own `exp` calls do.
        var inv_sum = Scalar[ACCUM](1) / row_sum
        var inv_bcast = SIMD[ACCUM, LANES](inv_sum)
        for j in range(0, vectorized_end, LANES):
            var v = in_row.load[LANES](Coord(j)).cast[ACCUM]()
            var y = _exp_hvx(v - max_bcast) * inv_bcast
            out_row.store[LANES](Coord(j), y.cast[in_dtype]())
        for j in range(vectorized_end, row_length):
            var x = in_row.load[1, alignment=SCALAR_ALIGN](Coord(j)).cast[
                ACCUM
            ]()[0]
            var y = _exp_hvx[1](SIMD[ACCUM, 1](x - row_max))[0] * inv_sum
            out_row.store[1, alignment=SCALAR_ALIGN](
                Coord(j), SIMD[in_dtype, 1](y.cast[in_dtype]())
            )


@always_inline
def softmax_2d[
    in_dtype: DType
](src: RowMajor2D[in_dtype], dst: RowMajor2D[in_dtype]):
    """Numerically stable row-wise softmax: `y = exp(x - max) / sum(exp(x - max))`.

    Each row of `src` is softmaxed independently into the matching row of `dst`
    (the decode case is a single row). I/O is `in_dtype` (fp16/bf16); the
    reduction and exponential run in fp32.

    Addressing is the tile model's: the extents arrive folded into the tiles'
    layout, and every access is a coordinate into it. No row stride is restated
    and `r * row_length` appears nowhere. Taking tiles rather than a pointer
    pair plus `num_rows`/`row_length` means there is no second copy of the
    geometry that could disagree with the one the accesses actually use.

    `src` and `dst` are required to carry the same extents; the seam builds both
    from one layout.
    """
    _softmax_rows[in_dtype](src, dst, 0, Int(src.dim[0]()))


# ===--------------------------------------------------------------------------===
# Device services + the launch-striped entry.
#
# The `external_call`s are the same `qshim` queries `npu_kernel.mojo` spells out,
# repeated here because a standalone `mojo build --emit=object` for the bare-metal
# Hexagon triple cannot reach the overlay. They are undefined until the link binds
# them against `qshim_qurt.c`; a host build that never calls the entry leaves them
# unreferenced, which is what lets the algorithm be tested off-device.
#
# The entry below is what the FLUX dispatch path reaches (megakernel slice 4). It
# differs from the `hvx_*` exports at the bottom of the file in one way: the rows
# divide across the launch's four compute threads by launch index, so each
# participant runs the row body over the run it owns and the union covers every
# row exactly once.
# ===--------------------------------------------------------------------------===

comptime SOFTMAX_DTYPE = DType.float16
"""The dtype decode softmax is used at; the I/O element type."""

comptime NUM_COMPUTE_THREADS: UInt32 = 4
"""Threads of a core that compute. The fifth is the part's DMA thread."""

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
    """Index of this thread within the core, 0-based (narrowed to 8 bits)."""
    return Int(UInt8(_qshim_query(QSHIM_QUERY_THREAD_ID)))


@always_inline
def _qurt_thread_id() -> UInt32:
    """The RTOS thread this code is running on, read only to tell threads
    apart — a launch index alone proves nothing about how many threads back
    those indices."""
    return external_call["qurt_thread_get_id", UInt32]()


@always_inline
def _thread_rows(num_rows: Int, thread_id: UInt32) -> Tuple[Int, Int]:
    """The `[start, end)` run of rows this launch index owns.

    Each of the four compute threads owns a contiguous run; the fifth (DMA)
    thread and any launch index past the row count get the empty run `(0, 0)`,
    which the row body then loops zero times over. The split stays hand-written:
    `.distribute()` would give a *cyclic* partition, and this part wants blocked
    (KB `kernels/distribute-is-cyclic-this-part-wants-blocked`).

    `thread_id` is an argument rather than a shim read so the partition is
    testable off-device; only `_softmax_entry` below reads the shim.
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
def softmax_striped[
    in_dtype: DType
](src: RowMajor2D[in_dtype], dst: RowMajor2D[in_dtype], thread_id: UInt32,):
    """Softmaxes the rows `thread_id` owns."""
    var rows = _thread_rows(Int(src.dim[0]()), thread_id)
    _softmax_rows[in_dtype](src, dst, rows[0], rows[1])


@always_inline
def _softmax_entry[
    in_dtype: DType
](
    src: RowMajor2D[in_dtype],
    dst: RowMajor2D[in_dtype],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
):
    """Row-wise softmax over the rows this launch index owns.

    `witness` reports what the output cannot: the SIMD width the kernel compiled
    at (64 for the vector units, or the host width had it resolved there), the
    fp32 accumulator width, the participant count the shim reports, and the RTOS
    thread each launch index ran on. It stays a raw pointer where the data does
    not: it is a side channel of fixed slots indexed by launch thread.
    """
    comptime W = HVX_VECTOR_SIZE // size_of[in_dtype]()
    var thread_idx = _npu_thread_idx()

    if thread_idx == 0:
        witness.store(0, UInt32(W))
        witness.store(1, UInt32(SIMD[ACCUM, W]().size))
        witness.store(2, UInt32(_qshim_query(QSHIM_QUERY_NUM_THREADS)))
    witness.store(3 + thread_idx, _qurt_thread_id())

    softmax_striped[in_dtype](src, dst, UInt32(thread_idx))


@export
def hvx_softmax_f16(
    input: UnsafePointer[Scalar[DType.float16], MutUnsafeAnyOrigin],
    output: UnsafePointer[Scalar[DType.float16], MutUnsafeAnyOrigin],
    num_rows: Int,
    row_length: Int,
) abi("C"):
    """C-ABI fp16 entry point for the FastRPC skel, and the seam."""
    var extent = row_major(Coord(num_rows, row_length))
    softmax_2d[DType.float16](
        RowMajor2D[DType.float16](ptr=input, layout=extent),
        RowMajor2D[DType.float16](ptr=output, layout=extent),
    )


@export
def hvx_softmax_bf16(
    input: UnsafePointer[Scalar[DType.bfloat16], MutUnsafeAnyOrigin],
    output: UnsafePointer[Scalar[DType.bfloat16], MutUnsafeAnyOrigin],
    num_rows: Int,
    row_length: Int,
) abi("C"):
    """C-ABI bf16 entry point for the FastRPC skel, and the seam."""
    var extent = row_major(Coord(num_rows, row_length))
    softmax_2d[DType.bfloat16](
        RowMajor2D[DType.bfloat16](ptr=input, layout=extent),
        RowMajor2D[DType.bfloat16](ptr=output, layout=extent),
    )
