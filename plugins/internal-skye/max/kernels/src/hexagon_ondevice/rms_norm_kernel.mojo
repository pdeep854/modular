from layout.coord import All, Coord
from layout.tile_layout import row_major
from layout.tile_tensor import TileTensor, stack_allocation

from std.ffi import external_call
from std.math import align_up, ceildiv, iota, sqrt
from std.sys import align_of, bit_width_of, masked_load, masked_store, size_of
from std.utils import StaticTuple

# The interleaved rotation's per-row passes, reused verbatim by the fused
# norm+rope arm below so its rotation is the shipped rope kernel's arithmetic,
# not a transcription of it (the C2 drift lesson, as the fused adaLN arm's
# LayerNorm3Pass import states it).
from hexagon_ondevice.rope_kernel import (
    FREQ_DTYPE as ROPE_FREQ_DTYPE,
    FREQ_ROWS_UNKNOWN,
    RowMajor1D as RopeRowMajor1D,
    rope_apply_row,
)

# ===--------------------------------------------------------------------------===
# `mo.reduce.rms_norm` — HVX root-mean-square normalization
#
# The math is `RMSNorm2Pass` / `_rms_norm` from
# `max/kernels/src/qaic/rms_norm.mojo`, the registered implementation of
# `mo.reduce.rms_norm`. This was the first vector kernel confirmed on v81
# silicon, and the HVX idioms every norm beside it uses — the width stated from
# the register geometry, the fp32 scalar normalize, the thread-striped rows, the
# rotate-tree reduction — are this kernel's, borrowed outward rather than in.
#
# What is *not* carried over from the registered kernel is `rms_norm_impl`, the
# multi-NSP driver above `_rms_norm`. The C-ABI entry runs the compute on the
# caller's DDR pointers, which HVX addresses directly; the staged variant beside
# it in `npu_kernel.mojo` moves each row through VTCM by uDMA instead, and drives
# the same `RMSNorm2Pass` over its scratch. Both reach the same answers, and the
# row split across the launch's threads is `_rms_norm`'s own either way.
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

comptime RMS_DTYPE = DType.float16
"""The dtype the registered kernel is used at, and the widest HVX lane count."""

comptime RMS_DTYPE_BF = DType.bfloat16
"""The bf16 instantiation dtype. Same 64-lane width; v81's qfloat lowering of
the bf16 square-accumulate is what dispatching the bf16 entry confirms on
silicon."""

comptime NUM_COMPUTE_THREADS: UInt32 = 4
"""Threads of a core that compute. The fifth is the part's DMA thread."""

comptime QSHIM_QUERY_THREAD_ID: UInt32 = 0
comptime QSHIM_QUERY_NUM_THREADS: UInt32 = 1


@always_inline
def staged_norm_slot_bytes(length: Int) -> Int:
    """VTCM one compute thread stages a `length`-wide row through.

    Three vector-aligned rows -- the thread's own copy of the weights, the row
    coming in and the row going out. This is `mojo_rms_norm_f16_staged`'s carve,
    stated here rather than only at that call site so the step arena's residue
    can be checked against it without restating it: a restatement is exactly the
    drift `hmx_stripe_act_slots`' docstring records, where a term added in one
    place and missed in the other does not fail loudly.
    """
    return 3 * align_up(length * size_of[RMS_DTYPE](), Int(HVX_VECTOR_SIZE))


@always_inline
def staged_norm_bytes(length: Int) -> Int:
    """VTCM one staged-norm launch needs for a `length`-wide row.

    The launch's requirement is the slot times the compute threads, not one slot
    plus one shared gamma: the weights are duplicated per thread because sharing
    one copy would need a rendezvous to publish it, and that rendezvous is the
    thing the staged path exists to avoid.
    """
    return staged_norm_slot_bytes(length) * Int(NUM_COMPUTE_THREADS)


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
`@no_inline` pass re-derives `row * length` on every entry, with the vector work
unchanged. Slicing resolves that multiply once, at the point the row is chosen,
which is the row-level reading of KB
`kernels/tiletensor-resolve-the-base-once-not-per-access`.

`gamma` is one row of a `(1, length)` tile rather than a rank-1 type of its own:
it is the same shape as the row it scales, and giving it the same type is what
lets the two be loaded by one spelling.
"""


@always_inline
def _rms_unroll_factor[dtype: DType]() -> Int:
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
struct RMSNorm2Pass[
    dtype: DType,
    multiply_before_cast: Bool = True,
    unroll: Int = _rms_unroll_factor[dtype](),
]:
    """Two-pass rms normalization for a single row.

    Pass 1: inv_rms = 1 / sqrt((sum(x^2) / N) + eps)   — float32 accumulator
    Pass 2: out[i]  = (gamma[i] + weight_offset) * x[i] * inv_rms

    `inv_rms` stays a float32 scalar broadcast into the vector ops, so the
    scaling never round-trips through half precision.

    `multiply_before_cast` picks which side of the cast the gamma multiply falls
    on. It is a parameter rather than an argument because the registered kernel
    binds it per architecture; the entry below compiles both and selects one, so
    the ragged tail goes through each branch.
    """

    comptime width = Int(HVX_VECTOR_SIZE) // size_of[Self.dtype]()
    comptime unrolled_size = Self.width * Self.unroll

    comptime RowView = Row[Self.dtype]
    """One row of the caller's `(num_rows, length)` tile, and of `gamma`."""

    var hw_width_elems: Int
    var unrolled_elems: Int
    var has_tail: Bool
    var tail_elems: Int
    var mask: SIMD[DType.bool, Self.width]
    var length: Int
    var eps: Float32
    var weight_offset: Scalar[dtype=Self.dtype]

    @always_inline
    def __init__(
        out self,
        src: RowMajor2D[Self.dtype],
        eps: Float32 = 1e-6,
        weight_offset: Scalar[dtype=Self.dtype] = 0.0,
    ):
        """Derives the per-row geometry from the tile the passes will index.

        The row length is `src`'s second extent, read out of the layout rather
        than passed beside it, so no caller can configure this for a length the
        tile does not have.
        """
        # One 128-byte HVX register's worth of lanes, and nothing else. Stated
        # from the register geometry rather than taken from `simd_width_of`
        # because this same source is compiled for the HOST by its twin test,
        # where the query correctly returns 8. A derived width would test a
        # different kernel than it ships. KB
        # `kernels/tiletensor-vector-width-state-it-explicitly`.
        comptime assert (
            Self.width * bit_width_of[Self.dtype]() == HVX_VECTOR_SIZE * 8
        ), "width must be exactly one HVX vector register"

        var length = Int(src.dim[1]())

        self.length = length
        self.eps = eps
        self.weight_offset = weight_offset
        self.hw_width_elems = (length // Self.width) * Self.width
        self.unrolled_elems = (
            length // Self.unrolled_size
        ) * Self.unrolled_size
        self.tail_elems = length - self.hw_width_elems
        self.has_tail = self.tail_elems > 0
        self.mask = _pred[Self.width](self.tail_elems)

    @always_inline
    def _tail(self, row: Self.RowView) -> SIMD[Self.dtype, Self.width]:
        """Loads a row's ragged last partial vector, invalid lanes zeroed.

        The address comes from the tile's own layout — no stride is restated by
        the author. It is still a pointer because `masked_load` is the only
        predicated access this target has and TileTensor has no tile-level
        equivalent; that is a real gap in the tile model, not a preference.

        TODO(tile-model): use a predicated TileTensor load once one exists.
        """
        return masked_load[
            dtype=Self.dtype,
            size=Self.width,
            alignment=align_of[Self.dtype](),
        ](
            addr=row.ptr_at_offset(Coord(self.hw_width_elems)),
            mask=self.mask,
            passthrough=SIMD[Self.dtype, Self.width](0),
        )

    # ===----------------------------------------------------------------------===
    # Pass 1: inv_rms = 1 / sqrt(sum(x^2) / N + eps)    (float32 accumulator)
    # ===----------------------------------------------------------------------===

    @no_inline
    def pass1(self, row: Self.RowView) -> Float32:
        var unrolled_acc: StaticTuple[
            SIMD[DType.float32, Self.width], Self.unroll
        ] = {fill = SIMD[DType.float32, Self.width](0)}

        for i in range(0, self.unrolled_elems, Self.unrolled_size):
            comptime for k in range(Self.unroll):
                var vec = row.load[Self.width](Coord(i + k * Self.width)).cast[
                    DType.float32
                ]()
                unrolled_acc[k] += vec * vec

        var running_acc = _reduce_tuple[SIMD.__add__](unrolled_acc)

        for i in range(self.unrolled_elems, self.hw_width_elems, Self.width):
            var vec = row.load[Self.width](Coord(i)).cast[DType.float32]()
            running_acc += vec * vec

        # Tail: passthrough=0 so invalid lanes contribute 0 to the sum.
        if self.has_tail:
            var vec_f32 = self._tail(row).cast[DType.float32]()
            running_acc += vec_f32 * vec_f32

        var ms_f32 = _reduce_add(running_acc)[0] / Float32(self.length)
        return Float32(1) / sqrt(ms_f32 + self.eps)

    # ===----------------------------------------------------------------------===
    # Pass 2: out[i] = (gamma[i] + offset) * x[i] * inv_rms
    # ===----------------------------------------------------------------------===

    @always_inline
    def _apply(
        self,
        x: SIMD[Self.dtype, Self.width],
        g: SIMD[Self.dtype, Self.width],
        inv_rms: Float32,
    ) -> SIMD[Self.dtype, Self.width]:
        comptime if Self.multiply_before_cast:
            var gamma_accum = (
                g.cast[DType.float32]()
                + self.weight_offset.cast[DType.float32]()
            )
            return (x.cast[DType.float32]() * inv_rms * gamma_accum).cast[
                Self.dtype
            ]()
        else:
            var normed = (x.cast[DType.float32]() * inv_rms).cast[Self.dtype]()
            return normed * (g + self.weight_offset.cast[Self.dtype]())

    @no_inline
    def pass2(
        self,
        dst_row: Self.RowView,
        src_row: Self.RowView,
        gamma: Self.RowView,
        inv_rms: Float32,
    ):
        for i in range(0, self.unrolled_elems, Self.unrolled_size):
            comptime for k in range(Self.unroll):
                var x = src_row.load[Self.width](Coord(i + k * Self.width))
                var g = gamma.load[Self.width](Coord(i + k * Self.width))
                dst_row.store[Self.width](
                    Coord(i + k * Self.width), self._apply(x, g, inv_rms)
                )

        for i in range(self.unrolled_elems, self.hw_width_elems, Self.width):
            var x = src_row.load[Self.width](Coord(i))
            var g = gamma.load[Self.width](Coord(i))
            dst_row.store[Self.width](Coord(i), self._apply(x, g, inv_rms))

        if self.has_tail:
            masked_store[size=Self.width, alignment=align_of[Self.dtype]()](
                value=self._apply(
                    self._tail(src_row), self._tail(gamma), inv_rms
                ),
                addr=dst_row.ptr_at_offset(Coord(self.hw_width_elems)),
                mask=self.mask,
            )

    @always_inline
    def __call__(
        self,
        dst_row: Self.RowView,
        src_row: Self.RowView,
        gamma: Self.RowView,
    ):
        self.pass2(dst_row, src_row, gamma, self.pass1(src_row))


@always_inline
def _rms_norm[
    dtype: DType,
    //,
    multiply_before_cast: Bool = True,
    unroll: Int = _rms_unroll_factor[dtype](),
](
    dst: RowMajor2D[dtype],
    src: RowMajor2D[dtype],
    gamma: Row[dtype],
    thread_id: UInt32,
    eps: Float32 = 1e-6,
    weight_offset: Scalar[dtype=dtype] = 0.0,
):
    """Normalizes rows on the launch's threads, dividing them by launch index.

    Each of the four compute threads owns a contiguous run of rows; the fifth
    (DMA) thread takes none. The split stays hand-written: `.distribute()` would
    give a *cyclic* partition, and this part wants blocked (KB
    `kernels/distribute-is-cyclic-this-part-wants-blocked`).

    HVX addresses DDR directly, so the compute runs on the caller's buffers —
    no VTCM staging. Staging is available (`mojo_rms_norm_f16_staged`,
    UDMA_STATUS.md) and drives this same `RMSNorm2Pass` over its scratch.

    The tiles arrive built: the seam is the C-ABI entry above, so nothing here
    or below it takes an extent beside a base. `gamma` arrives as a row rather
    than a tile because one weight row scales every data row — there is no
    second extent to carry. `thread_id` and the row bounds it derives stay
    scalars: a worker split is not geometry, and the rows this thread owns are
    not a shape anything indexes.
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

    var kernel = RMSNorm2Pass[
        dtype, multiply_before_cast=multiply_before_cast, unroll=unroll
    ](src, eps, weight_offset)
    for i in range(start_row, end_row):
        kernel(dst.slice(i, All), src.slice(i, All), gamma)


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
def _rms_norm_witness[
    dtype: DType
](
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    thread_idx: Int,
):
    """Reports the compiled width, the unroll factor, the participant count, and
    the RTOS thread each launch index ran on.

    The width is the one worth reading: a build that resolved it against the
    host would still normalize correctly, at a lane count that says the vector
    units never ran.

    `witness` stays a raw pointer where the data does not. It is a side channel
    of fixed slots indexed by launch thread, and a layout over it would state a
    geometry it does not have.
    """
    if thread_idx == 0:
        witness.store(0, UInt32(RMSNorm2Pass[dtype].width))
        witness.store(1, UInt32(RMSNorm2Pass[dtype].unroll))
        witness.store(2, UInt32(_qshim_query(QSHIM_QUERY_NUM_THREADS)))
    witness.store(3 + thread_idx, _qurt_thread_id())


@always_inline
def _rms_norm_entry[
    dtype: DType,
](
    dst: RowMajor2D[dtype],
    src: RowMajor2D[dtype],
    gamma: Row[dtype],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    eps: Float32,
    weight_offset: Float32,
    multiply_before_cast: Int,
):
    """Shared body of the C-ABI entries: report the compiled width, then run.

    `multiply_before_cast` is a parameter of the kernel rather than an argument,
    so both instantiations are compiled and one is selected here. That is what
    the registered op does too — it binds the parameter per architecture — and
    running both is what puts the tail through each branch.
    """
    var thread_idx = _npu_thread_idx()
    _rms_norm_witness[dtype](witness, thread_idx)

    var offset = weight_offset.cast[dtype]()
    if multiply_before_cast != 0:
        _rms_norm[multiply_before_cast=True](
            dst, src, gamma, UInt32(thread_idx), eps, offset
        )
    else:
        _rms_norm[multiply_before_cast=False](
            dst, src, gamma, UInt32(thread_idx), eps, offset
        )


@always_inline
def rms_row[
    dtype: DType
](ptr: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin], length: Int) -> Row[
    dtype
]:
    """Views one contiguous run of `length` elements as a row.

    The staged entry's scratch is what needs this: its VTCM slots hold exactly
    one row each, so there is no `(num_rows, length)` tile to slice them out of.
    A `(1, length)` layout says that, and says it once.
    """
    return RowMajor2D[dtype](ptr=ptr, layout=row_major(Coord(1, length))).slice(
        0, All
    )


# ===--------------------------------------------------------------------------===
# Fused RMSNorm + interleaved RoPE (the SKYE_HMX_ROPE_FUSED_NORM arm).
#
# Every qk-norm dispatch in the FLUX.2 block stack (selector 5) is followed by a
# rope_interleaved dispatch (selector 9) that re-reads and re-writes the SAME
# head rows as a separate full DDR pass (WASTE_AUDIT §6: 0.63 s/render of pure
# re-crossing). Fusing the rotation into the norm's write pass deletes that
# read+write crossing; the normalized intermediate never reaches DDR.
#
# The C ABI is frozen, so the rope tables cannot arrive as new buffer slots.
# They ride inside the allocation the `witness` argument names instead — the
# one buffer whose contents are ours rather than the model's — located by a
# control word the host passes in the `multiply_before_cast` scalar slot:
#
#   ctl == 0 or 1   the slot keeps its legacy meaning; plain norm runs (any
#                   host that never opted in is untouched by construction)
#   ctl otherwise   byte offset from the witness base to a header:
#                   word 0  magic 'RFN1' (0x52464E31)
#                   word 1  byte offset, witness base -> positions (u32/row)
#                   word 2  byte offset, witness base -> freqs table
#                           ([max_position, length] fp32, cos/sin per pair,
#                           128-byte aligned so the vector loads match the
#                           standalone kernel's)
#                   word 3  freqs row length; must equal the dispatch's
#                   word 4  freqs row count (host/probe bookkeeping; the row
#                           lookup trusts `positions` as selector 9 does)
#                   words 5-7  zero, reserved
#
# An offset is valid on both sides only within one mapped allocation (the
# transport resolves each buffer to its allocation and folds the offset back),
# which is why the header cannot carry pointers and why everything is
# witness-relative. Any malformed contract refuses loudly: arm word 2, no
# output written, so the host's re-seed makes the miss detectable — the fused
# path must never half-run.
#
# Bit-identity with the two-kernel sequence hangs on the staging row: pass 2
# stores the normalized row to fp16 MEMORY (a stack slot standing in for the
# DDR row selector 5 would have written) and `rope_apply_row` loads it back,
# exactly the round trip the sequential pair takes through the intermediate
# buffer. Both passes are the shipped kernels' own bodies; only the row's
# address changed. Skipping the round trip would be numerically better and is
# deliberately not done: the arming gate is bit-identity.
# ===--------------------------------------------------------------------------===

comptime RMS_WITNESS_ARM_WORD = 8
"""THE ARM IS ECHOED, NOT ASSUMED (the fused adaLN arm's word-8 rule, applied
to selector 5): the entry stores which arm ran — 0 = plain norm, 1 = fused
norm+rope, 2 = fused contract refused — so the host CONFIGURES ITSELF from a
one-row probe instead of trusting an env var to match a comptime define. Word
8 sits beyond the norm's witness prologue (words 0-2 and the per-thread ids at
3..7). Every thread stores the same constant, the same benign raciness as the
thread-id stores."""

comptime RMS_ROPE_MAGIC: UInt32 = 0x52464E31
"""'RFN1' — the fused-contract header's first word. A header that does not
open with it is not a header, whatever the control word claimed."""

comptime RMS_ROPE_MAX_HEAD = 512
"""Widest head row the staging slot holds (elements). Production is 128; the
cap is a comptime stack carve, so a wider dispatch refuses (arm word 2) rather
than overrunning a frame slot."""

comptime RMS_ROPE_HDR_WORDS = 8
"""Header length in u32 words."""

comptime _RMS_ROPE_MIN_DELTA = 64
"""Smallest witness-relative byte offset the contract accepts, for the header
and both tables: past the norm's own words (0..8) with margin, so a legacy
value or a garbage control word cannot alias the kernel's report slots."""


@fieldwise_init
struct _RopeFusedHeader(ImplicitlyCopyable, Movable):
    """The fused contract, read once from the witness allocation and judged.

    `ok` is the single verdict every guard folds into; the deltas are byte
    offsets from the witness base, already validated for range and alignment
    when `ok` is true.
    """

    var ok: Bool
    var pos_delta: Int
    var freqs_delta: Int


@always_inline
def _rms_rope_header(
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    ctl: Int,
    length: Int,
) -> _RopeFusedHeader:
    """Reads and validates the fused-contract header `ctl` names.

    Every check is a host-layout duty restated as a refusal: the kernel cannot
    abort on device, so an invalid contract returns `ok=False` and the entry
    echoes arm word 2 and writes nothing. The row-count word (4) is trusted
    bookkeeping, as selector 9 trusts `positions` — no per-row bound is
    checked, matching the standalone kernel's trust level.
    """
    var refused = _RopeFusedHeader(False, 0, 0)
    if ctl < _RMS_ROPE_MIN_DELTA or ctl % size_of[UInt32]() != 0:
        return refused

    var hdr = ctl // size_of[UInt32]()
    if UInt32(witness.load(hdr)) != RMS_ROPE_MAGIC:
        return refused
    var pos_delta = Int(witness.load(hdr + 1))
    var freqs_delta = Int(witness.load(hdr + 2))
    var freq_row_elems = Int(witness.load(hdr + 3))

    if freq_row_elems != length or length % 2 != 0:
        return refused
    if length > RMS_ROPE_MAX_HEAD:
        return refused
    if pos_delta < _RMS_ROPE_MIN_DELTA or pos_delta % size_of[UInt32]() != 0:
        return refused
    if (
        freqs_delta < _RMS_ROPE_MIN_DELTA
        or freqs_delta % Int(HVX_VECTOR_SIZE) != 0
    ):
        return refused
    return _RopeFusedHeader(True, pos_delta, freqs_delta)


def _rms_norm_rope[
    dtype: DType
](
    dst: RowMajor2D[dtype],
    src: RowMajor2D[dtype],
    gamma: Row[dtype],
    freqs: RowMajor2D[ROPE_FREQ_DTYPE],
    positions: RopeRowMajor1D[DType.uint32],
    thread_id: UInt32,
    eps: Float32,
    weight_offset: Scalar[dtype=dtype],
):
    """Normalizes then ropes the rows this launch index owns.

    The blocked row split, the two norm passes and the rotation are the
    shipped kernels' own (`_rms_norm`'s split, `RMSNorm2Pass`, and selector
    9's `rope_apply_row[interleaved=True]`); this body only re-addresses the
    intermediate row from DDR to the staging slot. `multiply_before_cast` is
    fixed True: the control word replaced that scalar, and True is the value
    every qk-norm dispatch passes — the contract's docstring in
    `NPUKernels.def` is where that narrowing is signed.
    """
    comptime W = Int(HVX_VECTOR_SIZE) // size_of[dtype]()
    comptime assert (
        W * bit_width_of[dtype]() == HVX_VECTOR_SIZE * 8
    ), "width must be exactly one HVX vector register"
    comptime pos_align = align_of[Scalar[DType.uint32]]()

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

    var length = Int(src.dim[1]())
    var kernel = RMSNorm2Pass[dtype, multiply_before_cast=True](
        src, eps, weight_offset
    )
    # The stand-in for the DDR row the sequential pair round-trips through:
    # same element type, same vector alignment, per-thread by construction
    # because it is a frame slot.
    var staging = stack_allocation[dtype=dtype, alignment=Int(HVX_VECTOR_SIZE)](
        row_major[RMS_ROPE_MAX_HEAD]()
    )
    var staged_row = rms_row[dtype](staging.ptr.as_unsafe_any_origin(), length)

    for i in range(start_row, end_row):
        var src_row = src.slice(i, All)
        kernel.pass2(staged_row, src_row, gamma, kernel.pass1(src_row))
        var pos = Int(positions.load[1, alignment=pos_align](Coord(i)))
        rope_apply_row[interleaved=True, W=W](
            staged_row, dst.slice(i, All), freqs.slice(pos, All)
        )


@always_inline
def _rms_norm_rope_entry[
    dtype: DType,
](
    dst: RowMajor2D[dtype],
    src: RowMajor2D[dtype],
    gamma: Row[dtype],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    eps: Float32,
    weight_offset: Float32,
    ctl: Int,
):
    """Selector 5's body when the fused arm is compiled in.

    `ctl` is the wire's `multiply_before_cast` slot. The legacy values keep
    their exact meaning so a host that never opted in dispatches the plain
    norm bit-for-bit; anything else names the fused contract, which either
    validates and runs fused or refuses loudly (arm word 2, output untouched,
    detectable against the host's re-seed).
    """
    var thread_idx = _npu_thread_idx()
    _rms_norm_witness[dtype](witness, thread_idx)

    if ctl == 0 or ctl == 1:
        witness.store(RMS_WITNESS_ARM_WORD, UInt32(0))
        var offset = weight_offset.cast[dtype]()
        if ctl != 0:
            _rms_norm[multiply_before_cast=True](
                dst, src, gamma, UInt32(thread_idx), eps, offset
            )
        else:
            _rms_norm[multiply_before_cast=False](
                dst, src, gamma, UInt32(thread_idx), eps, offset
            )
        return

    var length = Int(src.dim[1]())
    var header = _rms_rope_header(witness, ctl, length)
    if not header.ok:
        witness.store(RMS_WITNESS_ARM_WORD, UInt32(2))
        return
    witness.store(RMS_WITNESS_ARM_WORD, UInt32(1))

    # The seam's second half: the two tables the contract carries become tiles
    # here, once, from witness-relative offsets — the one place below the C ABI
    # where an address is computed rather than indexed.
    var base = witness.bitcast[UInt8]()
    var freqs = RowMajor2D[ROPE_FREQ_DTYPE](
        ptr=(base + header.freqs_delta).bitcast[Scalar[ROPE_FREQ_DTYPE]](),
        layout=row_major(Coord(FREQ_ROWS_UNKNOWN, length)),
    )
    var positions = RopeRowMajor1D[DType.uint32](
        ptr=(base + header.pos_delta).bitcast[Scalar[DType.uint32]](),
        layout=row_major(Coord(Int(src.dim[0]()))),
    )
    _rms_norm_rope[dtype](
        dst,
        src,
        gamma,
        freqs,
        positions,
        UInt32(thread_idx),
        eps,
        weight_offset.cast[dtype](),
    )
