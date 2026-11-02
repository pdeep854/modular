from layout.coord import All, Coord
from layout.tile_layout import row_major
from layout.tile_tensor import TileTensor

from std.ffi import external_call
from std.math import ceildiv, iota, sqrt
from std.sys import align_of, bit_width_of, masked_load, masked_store, size_of
from std.utils import StaticTuple

# ===--------------------------------------------------------------------------===
# `mo.reduce.layer_norm` (no-affine) — HVX candidate for the FLUX MMDiT hot path
#
# The math is `LayerNorm3Pass` / `_layer_norm` from
# `max/kernels/src/qaic/layer_norm.mojo`, the registered implementation of
# `mo.reduce.layer_norm`, restricted to the no-affine case FLUX.2-Klein uses
# (gamma == 1, beta == 0, so the whole affine term drops out). The HVX idioms —
# the derived SIMD width, the fp32-scalar normalize, the thread-striped rows, the
# rotate-tree reduction — are `RMSNorm2Pass`'s from `npu_kernel.mojo`, the first
# vector kernel verified on v81 hardware.
#
# This lives in its own file rather than in `npu_kernel.mojo`: it is a candidate
# awaiting on-device confirmation, and the shared kernel + its `NPUKernels.def`
# manifest are serialized through the device lane. When it lands it moves there
# under the same obligation the rms and moe ports carry — a change to one is a
# change to both until the kernel can be shared.
#
# LayerNorm differs from rms_norm in the one way that matters here: it subtracts
# the row MEAN before normalizing, and divides by the standard deviation of the
# mean-centred row rather than by the RMS. So it is two reductions, not one —
# mean, then variance about that mean — and the subtraction is carried in fp32
# to keep `x - mean` from cancelling in half precision when x is near the mean.
#
# Addressing is the tile model's: the caller's extents become a layout once at
# the C-ABI entry, and every access after that is a coordinate into it. No row
# stride is restated, `i * length` appears nowhere, and no signature below that
# entry takes a base beside an extent. The translation is KB
# `kernels/tile-model-porting-a-pointer-kernel-to-tiletensor`; that the machine
# code is unchanged by it is checked, not assumed, by
# `plugins/internal-skye/utils/kernel_asm_compare.sh`.
# ===--------------------------------------------------------------------------===

comptime HVX_VECTOR_SIZE = 128
"""Size of an HVX vector in bytes (`+hvx-length128b`)."""

comptime LN_DTYPE = DType.float16
"""The fp16 instantiation dtype — widest HVX lane count and FLUX's I/O dtype."""

comptime LN_DTYPE_BF = DType.bfloat16
"""The bf16 instantiation dtype. Same 64-lane width; v81 qfloat lowering is the
part that still needs silicon confirmation (see HVX_KERNEL_STATUS.md)."""

comptime NUM_COMPUTE_THREADS: UInt32 = 4
"""Threads of a core that compute. The fifth is the part's DMA thread."""

comptime QSHIM_QUERY_THREAD_ID: UInt32 = 0
comptime QSHIM_QUERY_NUM_THREADS: UInt32 = 1


comptime RowMajor2D[dtype: DType] = TileTensor[
    dtype, type_of(row_major(Coord(0, 0))), origin=MutUnsafeAnyOrigin
]
"""The `(num_rows, length)` view the C-ABI entries' pointers are wrapped in.

Both extents are dynamic, so the layout carries runtime values; what it makes
compile-time is the *rank* and the row-major relationship between them, which is
what removes `i * length` from every access.
"""


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

Why a row VIEW and not the 2-D tile plus a row index: the two are equivalent to
read and are NOT equivalent to compile. Indexing `Coord(row, j)` inside a
`@no_inline` pass re-derives `row * length` on every entry -- measured at +9
scalar instructions per pass, with the vector work unchanged. Slicing resolves
that multiply once, at the point the row is chosen, which is the row-level
reading of KB `kernels/tiletensor-resolve-the-base-once-not-per-access`.
"""


@always_inline
def _ln_unroll_factor[dtype: DType]() -> Int:
    """Mirrors `get_unroll_factor`, whose value is the element's byte width."""
    return size_of[dtype]()


@always_inline
def _pred[width: Int](count: Int) -> SIMD[DType.bool, width]:
    """Lanes below `count`, for a masked load or store of a partial vector."""
    return iota[DType.int, width]().lt(SIMD[DType.int, width](count))


comptime _bin_op = def(lhs: SIMD, rhs: type_of(lhs)) thin -> type_of(lhs)


@always_inline
def _reduce[op: _bin_op](vec: SIMD) -> type_of(vec):
    """Folds a vector into every lane by rotate-and-combine.

    Lowers to HVX `vror` + the combining op, so a 64-lane sum is six packets
    rather than a scalar drain.
    """
    var reduced = vec

    comptime for i in range(8, -1, -1):
        comptime if (1 << i) < vec.size:
            reduced = op(reduced, reduced.rotate_right[1 << i]())
    return reduced


comptime _reduce_add = _reduce[SIMD.__add__]


@always_inline
def _reduce_tuple[
    dtype: DType,
    width: SIMDLength,
    //,
    op: _bin_op,
](tuple: StaticTuple[SIMD[dtype, width], _]) -> SIMD[dtype, width]:
    """Combines a tuple of accumulators pairwise down to one."""
    comptime assert (
        tuple.size & (tuple.size - 1)
    ) == 0, "size must be power of two"

    var result = tuple

    comptime for i in range(tuple.size):
        comptime step = tuple.size >> (i + 1)

        comptime if step > 0:
            comptime for j in range(step):
                result[j] = op(result[j], result[j + step])

    return result[0]


@align(128)
struct LayerNorm3Pass[
    dtype: DType,
    unroll: Int = _ln_unroll_factor[dtype](),
]:
    """Two-pass mean/variance layer normalization for a single row, no affine.

    Pass 1: mean    = sum(x) / N                     — float32 accumulator
    Pass 2: var     = sum((x - mean)^2) / N          — float32 accumulator
    Pass 3: out[i]  = (x[i] - mean) / sqrt(var + eps) — float32 normalize

    `mean` and `inv_std` stay float32 scalars broadcast into the vector ops, so
    the centring and scaling never round-trip through half precision. That is the
    `RMSNorm2Pass._apply` idiom rather than the registered kernel's, which casts
    both to the working dtype first; the two agree to well within the output's
    resolution, and the fp32 form is the one the verified vector port uses.
    """

    comptime width = Int(HVX_VECTOR_SIZE) // size_of[Self.dtype]()
    comptime unrolled_size = Self.width * Self.unroll

    comptime RowView = Row[Self.dtype]
    """One row of the caller's `(num_rows, length)` tile."""

    var hw_width_elems: Int
    var unrolled_elems: Int
    var has_tail: Bool
    var tail_elems: Int
    var mask: SIMD[DType.bool, Self.width]
    var length: Int
    var eps: Float32

    @always_inline
    def __init__(out self, src: RowMajor2D[Self.dtype], eps: Float32 = 1e-6):
        """Derives the per-row geometry from the tile the passes will index.

        The row length is `src`'s second extent, read out of the layout rather
        than passed beside it, so no caller can configure this for a length the
        tile does not have.
        """
        # One 128-byte HVX register's worth of lanes, and nothing else. Stated
        # from the register geometry rather than taken from `simd_width_of`, but
        # NOT for the reason KB
        # `kernels/tiletensor-vector-width-state-it-explicitly` gives -- that
        # descriptor bug is fixed (#114/#146) and measured fixed here: this
        # target reports 64 for fp16 and the derived spelling compiles to
        # identical code. It is stated because this same source is compiled for
        # the HOST by its twin test, where the query correctly returns 8. A
        # derived width would test a different kernel than it ships.
        comptime assert (
            Self.width * bit_width_of[Self.dtype]() == HVX_VECTOR_SIZE * 8
        ), "width must be exactly one HVX vector register"

        var length = Int(src.dim[1]())

        self.length = length
        self.eps = eps
        self.hw_width_elems = (length // Self.width) * Self.width
        self.unrolled_elems = (
            length // Self.unrolled_size
        ) * Self.unrolled_size
        self.tail_elems = length - self.hw_width_elems
        self.has_tail = self.tail_elems > 0
        self.mask = _pred[Self.width](self.tail_elems)

    # ===----------------------------------------------------------------------===
    # Pass 1: mean = sum(x) / N    (float32 accumulator)
    # ===----------------------------------------------------------------------===

    @always_inline
    def _tail(self, row: Self.RowView) -> SIMD[Self.dtype, Self.width]:
        """Loads a row's ragged last partial vector, invalid lanes zeroed.

        The address comes from the tile's own layout -- no stride is restated by
        the author. It is still a pointer because `masked_load` is the only
        predicated access this target has and TileTensor has no tile-level
        equivalent; that is a real gap in the tile model, not a preference. This
        and the matching store in pass 3 are the only two places in the kernel
        that resolve one, and both resolve it from the tile.

        TODO(tile-model): use a predicated TileTensor load once one exists.
        """
        return masked_load[
            dtype=Self.dtype, size=Self.width, alignment=align_of[Self.dtype]()
        ](
            addr=row.ptr_at_offset(Coord(self.hw_width_elems)),
            mask=self.mask,
            passthrough=SIMD[Self.dtype, Self.width](0),
        )

    @no_inline
    def pass1(self, row: Self.RowView) -> Float32:
        var unrolled_acc: StaticTuple[
            SIMD[DType.float32, Self.width], Self.unroll
        ] = {fill = SIMD[DType.float32, Self.width](0)}

        for i in range(0, self.unrolled_elems, Self.unrolled_size):
            comptime for k in range(Self.unroll):
                unrolled_acc[k] += row.load[Self.width](
                    Coord(i + k * Self.width)
                ).cast[DType.float32]()

        var running_acc = _reduce_tuple[SIMD.__add__](unrolled_acc)

        for i in range(self.unrolled_elems, self.hw_width_elems, Self.width):
            running_acc += row.load[Self.width](Coord(i)).cast[DType.float32]()

        # Tail: passthrough=0 so invalid lanes add 0 to the running sum.
        if self.has_tail:
            running_acc += self._tail(row).cast[DType.float32]()

        return _reduce_add(running_acc)[0] / Float32(self.length)

    # ===----------------------------------------------------------------------===
    # Pass 2: var = sum((x - mean)^2) / N    (float32 accumulator)
    # ===----------------------------------------------------------------------===

    @no_inline
    def pass2(self, row: Self.RowView, mean: Float32) -> Float32:
        var unrolled_acc: StaticTuple[
            SIMD[DType.float32, Self.width], Self.unroll
        ] = {fill = SIMD[DType.float32, Self.width](0)}

        for i in range(0, self.unrolled_elems, Self.unrolled_size):
            comptime for k in range(Self.unroll):
                var d = (
                    row.load[Self.width](Coord(i + k * Self.width)).cast[
                        DType.float32
                    ]()
                    - mean
                )
                unrolled_acc[k] += d * d

        var running_acc = _reduce_tuple[SIMD.__add__](unrolled_acc)

        for i in range(self.unrolled_elems, self.hw_width_elems, Self.width):
            var d = row.load[Self.width](Coord(i)).cast[DType.float32]() - mean
            running_acc += d * d

        # Tail: unlike the sum in pass 1, an invalid lane's (0 - mean)^2 is not
        # zero, so zero the difference on the mask before squaring rather than
        # leaning on the passthrough value.
        if self.has_tail:
            var d = self.mask.select(
                self._tail(row).cast[DType.float32]() - mean,
                SIMD[DType.float32, Self.width](0),
            )
            running_acc += d * d

        return _reduce_add(running_acc)[0] / Float32(self.length)

    # ===----------------------------------------------------------------------===
    # Pass 3: out[i] = (x[i] - mean) * inv_std    (float32 normalize, then cast)
    # ===----------------------------------------------------------------------===

    @always_inline
    def _apply(
        self,
        x: SIMD[Self.dtype, Self.width],
        mean: Float32,
        inv_std: Float32,
    ) -> SIMD[Self.dtype, Self.width]:
        return ((x.cast[DType.float32]() - mean) * inv_std).cast[Self.dtype]()

    @no_inline
    def pass3(
        self,
        dst_row: Self.RowView,
        src_row: Self.RowView,
        mean: Float32,
        inv_std: Float32,
    ):
        for i in range(0, self.unrolled_elems, Self.unrolled_size):
            comptime for k in range(Self.unroll):
                var x = src_row.load[Self.width](Coord(i + k * Self.width))
                dst_row.store[Self.width](
                    Coord(i + k * Self.width), self._apply(x, mean, inv_std)
                )

        for i in range(self.unrolled_elems, self.hw_width_elems, Self.width):
            var x = src_row.load[Self.width](Coord(i))
            dst_row.store[Self.width](Coord(i), self._apply(x, mean, inv_std))

        if self.has_tail:
            masked_store[size=Self.width, alignment=align_of[Self.dtype]()](
                value=self._apply(self._tail(src_row), mean, inv_std),
                addr=dst_row.ptr_at_offset(Coord(self.hw_width_elems)),
                mask=self.mask,
            )

    @always_inline
    def __call__(self, dst_row: Self.RowView, src_row: Self.RowView):
        var mean = self.pass1(src_row)
        var variance = self.pass2(src_row, mean)
        var inv_std = Float32(1) / sqrt(variance + self.eps)
        self.pass3(dst_row, src_row, mean, inv_std)


@always_inline
def _layer_norm[
    dtype: DType,
    //,
    unroll: Int = _ln_unroll_factor[dtype](),
](
    dst: RowMajor2D[dtype],
    src: RowMajor2D[dtype],
    thread_id: UInt32,
    eps: Float32 = 1e-6,
):
    """Normalizes rows on the launch's threads, dividing them by launch index.

    Each of the four compute threads owns a contiguous run of rows; the fifth
    (DMA) thread takes none. The split stays hand-written: `.distribute()` would
    give a *cyclic* partition, and this part wants blocked (KB
    `kernels/distribute-is-cyclic-this-part-wants-blocked`).

    HVX addresses DDR directly, so the compute runs on the caller's buffers —
    no VTCM staging, the same choice the gemv port makes. Staging is available
    (`mojo_rms_norm_f16_staged`, UDMA_STATUS.md); this port has simply not been
    carried onto it.

    The tiles arrive built: the seam is the C-ABI entry above, so nothing here
    or below it takes an extent beside a base. `thread_id` and the row bounds it
    derives stay scalars — a worker split is not geometry, and the rows this
    thread owns are not a shape anything indexes.
    """
    if thread_id >= UInt32(NUM_COMPUTE_THREADS):
        return

    var num_rows = Int(src.dim[0]())
    var rows_per_thread = ceildiv(num_rows, Int(NUM_COMPUTE_THREADS))
    var start_row = Int(thread_id) * rows_per_thread
    if start_row >= num_rows:
        return

    var end_row = start_row + rows_per_thread
    if end_row > num_rows:
        end_row = num_rows

    var kernel = LayerNorm3Pass[dtype, unroll](src, eps)
    for i in range(start_row, end_row):
        kernel(dst.slice(i, All), src.slice(i, All))


# ===--------------------------------------------------------------------------===
# Device services + C-ABI entry points.
#
# The `external_call`s are the same `qshim` queries `npu_kernel.mojo` spells out,
# repeated here because a standalone `mojo build --emit=object` for the bare-metal
# Hexagon triple cannot reach the overlay. They are undefined until the link binds
# them against `qshim_qurt.c`; a host build that never calls the exports leaves
# them unreferenced, which is what lets the algorithm be tested off-device.
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
def _layer_norm_entry[
    dtype: DType,
](
    dst: RowMajor2D[dtype],
    src: RowMajor2D[dtype],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    eps: Float32,
):
    """Shared body of the C-ABI entries: report the compiled width, then run.

    `witness` reports what the output cannot: the SIMD width the kernel compiled
    at, the fp32 accumulator lane count, how many threads the shim says
    participate, and the RTOS thread each launch index ran on. The width is the
    one worth reading — a build that resolved it against the host would still
    normalize correctly, at a lane count that says the vector units never ran.

    It stays a raw pointer where the data does not: it is a side channel of
    fixed slots indexed by launch thread, and a layout over it would state a
    geometry it does not have.
    """
    var thread_idx = _npu_thread_idx()

    if thread_idx == 0:
        witness.store(0, UInt32(LayerNorm3Pass[dtype].width))
        witness.store(
            1, UInt32(SIMD[DType.float32, LayerNorm3Pass[dtype].width]().size)
        )
        witness.store(2, UInt32(_qshim_query(QSHIM_QUERY_NUM_THREADS)))
    witness.store(3 + thread_idx, _qurt_thread_id())

    _layer_norm(dst, src, UInt32(thread_idx), eps)


@export
def hvx_layernorm_f16(
    output: UnsafePointer[Scalar[LN_DTYPE], MutUnsafeAnyOrigin],
    input: UnsafePointer[Scalar[LN_DTYPE], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    num_rows: Int,
    length: Int,
    eps: Float32,
) abi("C"):
    """No-affine layer norm over fp16 rows, spread across the launch's threads.

    This is the seam. The FastRPC skel's C ABI hands down raw pointers and
    extents and is not ours to move; the extents become a layout here, once, and
    every signature below this line takes the tile instead. Resolving it here
    rather than per access is the difference between the same vector work and
    4x it with spills (KB
    `kernels/tiletensor-resolve-the-base-once-not-per-access`).
    """
    var extent = row_major(Coord(num_rows, length))
    _layer_norm_entry[LN_DTYPE](
        RowMajor2D[LN_DTYPE](ptr=output, layout=extent),
        RowMajor2D[LN_DTYPE](ptr=input, layout=extent),
        witness,
        eps,
    )


@export
def hvx_layernorm_bf16(
    output: UnsafePointer[Scalar[LN_DTYPE_BF], MutUnsafeAnyOrigin],
    input: UnsafePointer[Scalar[LN_DTYPE_BF], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    num_rows: Int,
    length: Int,
    eps: Float32,
) abi("C"):
    """No-affine layer norm over bf16 rows, spread across the launch's threads.

    The seam, as `hvx_layernorm_f16` documents it.
    """
    var extent = row_major(Coord(num_rows, length))
    _layer_norm_entry[LN_DTYPE_BF](
        RowMajor2D[LN_DTYPE_BF](ptr=output, layout=extent),
        RowMajor2D[LN_DTYPE_BF](ptr=input, layout=extent),
        witness,
        eps,
    )
