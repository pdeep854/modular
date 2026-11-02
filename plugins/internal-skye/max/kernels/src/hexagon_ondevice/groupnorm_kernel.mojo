from layout.coord import All, Coord
from layout.tile_layout import row_major
from layout.tile_tensor import TileTensor

from std.ffi import external_call
from std.math import ceildiv, iota, sqrt
from std.sys import align_of, bit_width_of, masked_load, size_of
from std.utils import StaticTuple

# ===--------------------------------------------------------------------------===
# `mo.reduce.group_norm` (affine) — HVX candidate for the FLUX.2 VAE decoder
#
# The math is `group_norm_gpu_block` from `max/kernels/src/nn/normalization.mojo`,
# the registered implementation of `mo.reduce.group_norm` (`group_norm_gpu`),
# restricted to the affine case the VAE uses (`affine=True`, per-channel gamma /
# beta). The HVX idioms — the derived SIMD width, the fp32-scalar normalize, the
# thread-striped work units, the rotate-tree reduction — are `LayerNorm3Pass`'s
# from `layernorm_kernel.mojo`, the mean/variance sibling verified against a
# float64 reference on the same transport.
#
# This lives in its own file rather than in `npu_kernel.mojo`: it is a candidate
# awaiting on-device confirmation, and the shared kernel + its `NPUKernels.def`
# manifest are serialized through the device lane. When it lands it moves there
# under the same obligation the rms and layer-norm ports carry — a change to one
# is a change to both until the kernel can be shared.
#
# GroupNorm differs from LayerNorm in two ways that matter here:
#
# - **Grouping.** On the `[N, C, spatial]` layout the op normalizes each
#   `(n, group)` over the group's `(C / num_groups) * spatial` elements. Because
#   channels within a group are contiguous and spatial is contiguous within a
#   channel, that set is one contiguous block — the same reshape the GPU kernel
#   makes (`group_norm_reshape` -> `(N * num_groups, group_size)`). So the two
#   reductions (mean, then variance about it) are `LayerNorm3Pass`'s unchanged,
#   run over a block of length `group_size`.
# - **Affine.** After normalizing, a per-channel `gamma` / `beta` applies. Within
#   a group's block the first `spatial` elements are one channel, the next
#   `spatial` the next, so the affine is applied channel by channel: a scalar
#   `gamma[c]` / `beta[c]` broadcast over that channel's spatial run. The scale
#   and shift stay float32 scalars, so the normalize never round-trips through
#   half precision — the `LayerNorm3Pass._apply` idiom, matching the GPU kernel's
#   `(x - mean) * norm_factor * gamma + beta` element for element.
#
# Addressing is the tile model's: the caller's extents become layouts once at the
# seam, and every access after that is a coordinate into them. A group's block is
# described twice, because the two halves of the op read it differently — as one
# flat run for the reductions, and as the `(channels_per_group, spatial)` matrix
# the affine walks — and neither view restates a stride. `c * spatial` and
# `g * channels_per_group` appear nowhere. The translation is KB
# `kernels/tile-model-porting-a-pointer-kernel-to-tiletensor`; that the machine
# code is unchanged by it is checked, not assumed, by
# `plugins/internal-skye/utils/kernel_asm_compare.sh`.
# ===--------------------------------------------------------------------------===

comptime HVX_VECTOR_SIZE = 128
"""Size of an HVX vector in bytes (`+hvx-length128b`)."""

comptime GN_DTYPE = DType.float16
"""The fp16 instantiation dtype — widest HVX lane count and the VAE's I/O dtype."""

comptime GN_DTYPE_BF = DType.bfloat16
"""The bf16 instantiation dtype. Same 64-lane width; v81 qfloat lowering is the
part that still needs silicon confirmation (see HVX_KERNEL_STATUS.md)."""

comptime NUM_COMPUTE_THREADS: UInt32 = 4
"""Threads of a core that compute. The fifth is the part's DMA thread."""

comptime QSHIM_QUERY_THREAD_ID: UInt32 = 0
comptime QSHIM_QUERY_NUM_THREADS: UInt32 = 1


comptime RowMajor2D[dtype: DType] = TileTensor[
    dtype, type_of(row_major(Coord(0, 0))), origin=MutUnsafeAnyOrigin
]
"""A `(rows, length)` view of one of the C-ABI entries' pointers.

Both extents are dynamic, so the layout carries runtime values; what it makes
compile-time is the rank and the row-major relationship between them, which is
what removes the stride from every access.
"""


comptime RowMajor3D[dtype: DType] = TileTensor[
    dtype, type_of(row_major(Coord(0, 0, 0))), origin=MutUnsafeAnyOrigin
]
"""The activations as `(N * num_groups, channels_per_group, spatial)`.

This is the op's own shape with the two group axes already folded, and it is why
the affine needs no address arithmetic: the channel a `gamma` entry belongs to
is a dimension, not an offset the author computes.
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

Why a row VIEW and not the tile plus an index: the two are equivalent to read and
are NOT equivalent to compile. Re-indexing inside a `@no_inline` pass re-derives
the row base on every entry -- measured at +9 scalar instructions per pass on the
layernorm port, with the vector work unchanged. Slicing resolves that multiply
once, where the row is chosen, which is the row-level reading of KB
`kernels/tiletensor-resolve-the-base-once-not-per-access`.
"""


comptime ChannelBlock[dtype: DType] = type_of(
    RowMajor3D[dtype](
        ptr=UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin](
            unsafe_from_address=0
        ),
        layout=row_major(Coord(0, 0, 0)),
    ).slice(0, All, All)
)
"""One group's `(channels_per_group, spatial)` block, as `.slice(r, All, All)`
returns it. Slicing it again by channel yields a `Row` over that channel's
spatial run.
"""


@always_inline
def _gn_unroll_factor[dtype: DType]() -> Int:
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
struct GroupNorm3Pass[
    dtype: DType,
    unroll: Int = _gn_unroll_factor[dtype](),
]:
    """Two-pass mean/variance group normalization for a single group, affine.

    A group is one `(n, group)` block of `channels_per_group * spatial`
    contiguous elements.

    Pass 1: mean    = sum(x) / M                          — float32 accumulator
    Pass 2: var     = sum((x - mean)^2) / M               — float32 accumulator
    Pass 3: out[i]  = (x[i] - mean) / sqrt(var + eps) * gamma[c] + beta[c]

    where `M = channels_per_group * spatial` and `c` is the channel the element
    falls in. Passes 1 and 2 run over the whole contiguous block, the same fold
    `LayerNorm3Pass` uses. Pass 3 walks the group's channels, applying a scalar
    `gamma[c]` / `beta[c]` over each channel's `spatial` run; `mean` and
    `inv_std` stay float32 scalars broadcast into the vector ops, so the
    centring and scaling never round-trip through half precision.
    """

    comptime width = Int(HVX_VECTOR_SIZE) // size_of[Self.dtype]()
    comptime unrolled_size = Self.width * Self.unroll

    comptime RowView = Row[Self.dtype]
    """A group's block as one flat run — what the two reductions fold over."""

    comptime BlockView = ChannelBlock[Self.dtype]
    """The same block as `(channels_per_group, spatial)` — what the affine walks.
    """

    # Block tiling, for the two reductions over `group_size` elements.
    var group_size: Int
    var hw_width_elems: Int
    var unrolled_elems: Int
    var has_tail: Bool
    var tail_elems: Int
    var mask: SIMD[DType.bool, Self.width]

    # Spatial tiling, for the per-channel affine in pass 3. The channel count is
    # deliberately not held: it is the block view's own leading extent, and pass
    # 3 reads it there, so no stored copy can disagree with the tile it walks.
    var sp_hw_elems: Int
    var sp_tail_elems: Int

    var eps: Float32

    @always_inline
    def __init__(
        out self,
        blocks: RowMajor3D[Self.dtype],
        eps: Float32 = 1e-6,
    ):
        """Sizes both tilings from the activation tile's own layout.

        Args:
            blocks: The groups as `(N * num_groups, channels_per_group,
                spatial)`. Its channel and spatial extents are where the
                geometry lives; taking them as scalars instead would let the
                staging sizes disagree with the tile the passes are handed, and
                nothing would check it.
            eps: Added to the variance before the reciprocal square root.
        """
        # One 128-byte HVX register's worth of lanes, and nothing else. Stated
        # from the register geometry rather than taken from `simd_width_of`
        # because this same source is compiled for the HOST by its twin test,
        # where that query correctly returns 8 (KB
        # `kernels/tiletensor-vector-width-state-it-explicitly`). A derived width
        # would test a different kernel than it ships.
        comptime assert (
            Self.width * bit_width_of[Self.dtype]() == HVX_VECTOR_SIZE * 8
        ), "width must be exactly one HVX vector register"

        var channels_per_group = Int(blocks.dim[1]())
        var spatial = Int(blocks.dim[2]())

        self.eps = eps

        self.group_size = channels_per_group * spatial
        self.hw_width_elems = (self.group_size // Self.width) * Self.width
        self.unrolled_elems = (
            self.group_size // Self.unrolled_size
        ) * Self.unrolled_size
        self.tail_elems = self.group_size - self.hw_width_elems
        self.has_tail = self.tail_elems > 0
        self.mask = _pred[Self.width](self.tail_elems)

        self.sp_hw_elems = (spatial // Self.width) * Self.width
        self.sp_tail_elems = spatial - self.sp_hw_elems

    # ===----------------------------------------------------------------------===
    # Pass 1: mean = sum(x) / M    (float32 accumulator)
    # ===----------------------------------------------------------------------===

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

        # Tail: passthrough=0 so invalid lanes add 0 to the running sum. The
        # address comes from the tile's own layout; it is still a pointer because
        # `masked_load` is the only predicated access this target has and
        # TileTensor has no tile-level equivalent -- a real gap in the tile model,
        # not a preference.
        #
        # TODO(tile-model): use a predicated TileTensor load once one exists.
        if self.has_tail:
            var vec = masked_load[
                dtype=Self.dtype,
                size=Self.width,
                alignment=align_of[Self.dtype](),
            ](
                addr=row.ptr_at_offset(Coord(self.hw_width_elems)),
                mask=self.mask,
                passthrough=SIMD[Self.dtype, Self.width](0),
            )
            running_acc += vec.cast[DType.float32]()

        return _reduce_add(running_acc)[0] / Float32(self.group_size)

    # ===----------------------------------------------------------------------===
    # Pass 2: var = sum((x - mean)^2) / M    (float32 accumulator)
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
            var vec = masked_load[
                dtype=Self.dtype,
                size=Self.width,
                alignment=align_of[Self.dtype](),
            ](
                addr=row.ptr_at_offset(Coord(self.hw_width_elems)),
                mask=self.mask,
                passthrough=SIMD[Self.dtype, Self.width](0),
            )
            var d = self.mask.select(
                vec.cast[DType.float32]() - mean,
                SIMD[DType.float32, Self.width](0),
            )
            running_acc += d * d

        return _reduce_add(running_acc)[0] / Float32(self.group_size)

    # ===----------------------------------------------------------------------===
    # Pass 3: out[i] = (x[i] - mean) * inv_std * gamma[c] + beta[c]
    #                                       (float32 normalize + affine, per channel)
    # ===----------------------------------------------------------------------===

    @always_inline
    def _apply(
        self,
        x: SIMD[Self.dtype, Self.width],
        mean: Float32,
        inv_std: Float32,
        gamma_c: Float32,
        beta_c: Float32,
    ) -> SIMD[Self.dtype, Self.width]:
        return (
            (x.cast[DType.float32]() - mean) * inv_std * gamma_c + beta_c
        ).cast[Self.dtype]()

    @no_inline
    def pass3(
        self,
        output: Self.BlockView,
        input: Self.BlockView,
        gamma: Self.RowView,
        beta: Self.RowView,
        mean: Float32,
        inv_std: Float32,
    ):
        # One channel at a time: gamma / beta are constant across a channel's
        # spatial run, so they broadcast as float32 scalars rather than being
        # recomputed per lane the way a whole-block vector load would force. A
        # channel is a dimension of the block, so selecting one is a slice rather
        # than an offset the author computes -- and the number of them is that
        # dimension's extent, read off the tile rather than carried alongside it.
        for c in range(Int(input.dim[0]())):
            # `alignment` is stated on every scalar access below because
            # TileTensor's default is 1 off-GPU (`tile_tensor.mojo:799, 838`),
            # and a 1-aligned 16-bit access does not lower to `memuh` / `memh` on
            # this target -- it is synthesized from a pair of `memub` / `memb`
            # plus a shift, three extra instructions each, inside the spatial
            # tail loop. The element type's own alignment is what the pointer
            # form assumed, and the tile is built over the same buffer.
            var gamma_c = gamma.load[1, align_of[Self.dtype]()](Coord(c)).cast[
                DType.float32
            ]()
            var beta_c = beta.load[1, align_of[Self.dtype]()](Coord(c)).cast[
                DType.float32
            ]()
            var chan_in = input.slice(c, All)
            var chan_out = output.slice(c, All)

            for i in range(0, self.sp_hw_elems, Self.width):
                var x = chan_in.load[Self.width](Coord(i))
                chan_out.store[Self.width](
                    Coord(i), self._apply(x, mean, inv_std, gamma_c, beta_c)
                )

            # Per-channel spatial tail as scalar stores. A channel's tail does
            # not start on a vector boundary when `c * spatial` is not a multiple
            # of the width, so a masked vector store does not lower to HVX's
            # aligned predicated store the way the reduction-dimension tails do;
            # at the last channel of the last group it overruns the output buffer
            # and faults the NSP. Scalars touch only the remaining valid
            # elements — the softmax / silu tail idiom, verified on silicon.
            for t in range(self.sp_tail_elems):
                var x = chan_in.load[1, align_of[Self.dtype]()](
                    Coord(self.sp_hw_elems + t)
                ).cast[DType.float32]()
                var y = (x - mean) * inv_std * gamma_c + beta_c
                chan_out.store[1, align_of[Self.dtype]()](
                    Coord(self.sp_hw_elems + t), y.cast[Self.dtype]()
                )

    @always_inline
    def __call__(
        self,
        output: Self.BlockView,
        input: Self.BlockView,
        row: Self.RowView,
        gamma: Self.RowView,
        beta: Self.RowView,
    ):
        # `row` and `input` are the same elements under two descriptions: the
        # reductions fold the group as one contiguous run (one ragged tail), the
        # affine walks it as `(channels_per_group, spatial)`. Both come from the
        # seam, so neither derives the other's addressing.
        var mean = self.pass1(row)
        var variance = self.pass2(row, mean)
        var inv_std = Float32(1) / sqrt(variance + self.eps)
        self.pass3(output, input, gamma, beta, mean, inv_std)


@always_inline
def _group_norm[
    dtype: DType,
    //,
    unroll: Int = _gn_unroll_factor[dtype](),
](
    output: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin],
    input: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin],
    gamma: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin],
    beta: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin],
    N: Int,
    C: Int,
    spatial: Int,
    num_groups: Int,
    thread_id: UInt32,
    eps: Float32 = 1e-6,
):
    """Normalizes `(n, group)` blocks on the launch's threads, by launch index.

    The `N * num_groups` groups divide across the four compute threads; the
    fifth (DMA) thread takes none. The split stays hand-written: `.distribute()`
    would give a *cyclic* partition, and this part wants blocked (KB
    `kernels/distribute-is-cyclic-this-part-wants-blocked`).

    Each group is one contiguous block of `(C / num_groups) * spatial` elements
    — the reshape the GPU kernel makes. HVX addresses DDR directly, so the
    compute runs on the caller's buffers, the same choice the rms and layer-norm
    ports make (the multi-NSP uDMA driver faults this domain).

    This is the seam. The C-ABI boundary above hands down raw pointers and
    extents and is not ours to move; the extents become layouts here, once, and
    below this line nothing restates a stride.
    """
    if thread_id >= UInt32(NUM_COMPUTE_THREADS):
        return

    var channels_per_group = C // num_groups
    var group_size = channels_per_group * spatial
    var num_group_rows = N * num_groups

    var rows_per_thread = ceildiv(num_group_rows, Int(NUM_COMPUTE_THREADS))
    var start_row = Int(thread_id) * rows_per_thread
    if start_row >= num_group_rows:
        return

    var end_row = start_row + rows_per_thread
    if end_row > num_group_rows:
        end_row = num_group_rows

    # Built once, outside the row loop: resolving a tile per access instead of
    # per construction costs 4x the vector work and forces spills (KB
    # `kernels/tiletensor-resolve-the-base-once-not-per-access`).
    var flat_extent = row_major(Coord(num_group_rows, group_size))
    var block_extent = row_major(
        Coord(num_group_rows, channels_per_group, spatial)
    )
    # `gamma` and `beta` are per-channel over all of `C`, and a group owns a
    # contiguous run of channels, so `C` is `(num_groups, channels_per_group)`
    # and the group's run is a row of it.
    var affine_extent = row_major(Coord(num_groups, channels_per_group))

    var src_flat = RowMajor2D[dtype](ptr=input, layout=flat_extent)
    var src_block = RowMajor3D[dtype](ptr=input, layout=block_extent)
    var dst_block = RowMajor3D[dtype](ptr=output, layout=block_extent)
    var gamma_tile = RowMajor2D[dtype](ptr=gamma, layout=affine_extent)
    var beta_tile = RowMajor2D[dtype](ptr=beta, layout=affine_extent)

    var kernel = GroupNorm3Pass[dtype, unroll](src_block, eps)
    for r in range(start_row, end_row):
        var g = r % num_groups
        kernel(
            dst_block.slice(r, All, All),
            src_block.slice(r, All, All),
            src_flat.slice(r, All),
            gamma_tile.slice(g, All),
            beta_tile.slice(g, All),
        )


# ===--------------------------------------------------------------------------===
# Device services + C-ABI entry points.
#
# The `external_call`s are the same `qshim` queries `layernorm_kernel.mojo` spells
# out, repeated here because a standalone `mojo build --emit=object` for the
# bare-metal Hexagon triple cannot reach the overlay. They are undefined until the
# link binds them against `qshim_qurt.c`; a host build that never calls the
# exports leaves them unreferenced, which is what lets the algorithm be tested
# off-device.
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
def _group_norm_entry[
    dtype: DType,
](
    output: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin],
    input: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin],
    gamma: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin],
    beta: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    N: Int,
    C: Int,
    spatial: Int,
    num_groups: Int,
    eps: Float32,
):
    """Shared body of the C-ABI entries: report the compiled width, then run.

    `witness` reports what the output cannot: the SIMD width the kernel compiled
    at, the fp32 accumulator lane count, how many threads the shim says
    participate, and the RTOS thread each launch index ran on. The width is the
    one worth reading — a build that resolved it against the host would still
    normalize correctly, at a lane count that says the vector units never ran.
    """
    var thread_idx = _npu_thread_idx()

    if thread_idx == 0:
        witness.store(0, UInt32(GroupNorm3Pass[dtype].width))
        witness.store(
            1, UInt32(SIMD[DType.float32, GroupNorm3Pass[dtype].width]().size)
        )
        witness.store(2, UInt32(_qshim_query(QSHIM_QUERY_NUM_THREADS)))
    witness.store(3 + thread_idx, _qurt_thread_id())

    _group_norm(
        output,
        input,
        gamma,
        beta,
        N,
        C,
        spatial,
        num_groups,
        UInt32(thread_idx),
        eps,
    )


@export
def hvx_groupnorm_f16(
    output: UnsafePointer[Scalar[GN_DTYPE], MutUnsafeAnyOrigin],
    input: UnsafePointer[Scalar[GN_DTYPE], MutUnsafeAnyOrigin],
    gamma: UnsafePointer[Scalar[GN_DTYPE], MutUnsafeAnyOrigin],
    beta: UnsafePointer[Scalar[GN_DTYPE], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    N: Int,
    C: Int,
    spatial: Int,
    num_groups: Int,
    eps: Float32,
) abi("C"):
    """Affine group norm over fp16 `[N, C, spatial]`, spread across launch threads.
    """
    _group_norm_entry[GN_DTYPE](
        output, input, gamma, beta, witness, N, C, spatial, num_groups, eps
    )


@export
def hvx_groupnorm_bf16(
    output: UnsafePointer[Scalar[GN_DTYPE_BF], MutUnsafeAnyOrigin],
    input: UnsafePointer[Scalar[GN_DTYPE_BF], MutUnsafeAnyOrigin],
    gamma: UnsafePointer[Scalar[GN_DTYPE_BF], MutUnsafeAnyOrigin],
    beta: UnsafePointer[Scalar[GN_DTYPE_BF], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    N: Int,
    C: Int,
    spatial: Int,
    num_groups: Int,
    eps: Float32,
) abi("C"):
    """Affine group norm over bf16 `[N, C, spatial]`, spread across launch threads.
    """
    _group_norm_entry[GN_DTYPE_BF](
        output, input, gamma, beta, witness, N, C, spatial, num_groups, eps
    )
