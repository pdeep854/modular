from layout.coord import All, Coord, Idx
from layout.tile_layout import Layout, row_major
from layout.tile_tensor import TileTensor

from std.ffi import external_call
from std.math import ceildiv, sqrt
from std.sys import align_of, bit_width_of, size_of

# The no-affine layer norm's per-row passes, reused verbatim by the fused
# adaLN arm below so its normalization is the standalone kernel's arithmetic,
# not a transcription of it (the C2 drift lesson).
from hexagon_ondevice.layernorm_kernel import LayerNorm3Pass

# The two elementwise ops every FLUX.2 MMDiT block runs around its attention and
# feed-forward sub-blocks (`flux2_modulev3/transformer_blocks.py`): adaLN
# modulation on the normalized activation, `(1 + scale) * norm(x) + shift`, and
# the gated residual that closes each sub-block, `x + gate * sub_block(x)`. The
# arithmetic is a per-channel affine, but the intermediate is fp32 regardless of
# the fp16/bf16 I/O: a modulated activation can land a few thousand in magnitude,
# and folding `(1 + scale) * x + shift` in fp16 loses the low bits of the product
# before the add. fp32 in, one round out, matching how the reference folds it and
# the softmax/silu kernels beside it.
#
# Addressing is the tile model's: the extents the C-ABI hands down become a
# layout at the `@export` shells below and nowhere else, so every signature
# between there and the arithmetic names a view rather than a base pointer plus
# the scalars needed to interpret it. `r * row_length` appears nowhere, and the
# broadcast that made the params awkward to address by hand is a stride the
# view's own type carries. The translation is KB
# `kernels/tile-model-porting-a-pointer-kernel-to-tiletensor`; that the machine
# code is unchanged by it is checked, not assumed, by
# `plugins/internal-skye/utils/kernel_asm_compare.sh`.
comptime ACCUM = DType.float32

comptime HVX_VECTOR_SIZE = 128
"""Size of an HVX vector in bytes (`+hvx-length128b`)."""

comptime FP16_FINITE_MAX = 65504.0
"""The largest finite fp16 magnitude. The residual folds in fp32, so a sum past
this only becomes Inf at the store back to fp16 -- and an Inf entering the next
block's LayerNorm takes the whole row to NaN."""


comptime Rows2D[dtype: DType] = TileTensor[
    dtype, type_of(row_major(Coord(0, 0))), origin=MutUnsafeAnyOrigin
]
"""A `(num_rows, row_length)` view whose row stride is a runtime value.

Both extents are dynamic, so the layout carries runtime values; what it makes
compile-time is the rank and the unit stride along a row, which is what removes
`r * row_length` from every access. It is also the only argument type the two
kernels below take: an extent they would otherwise have received as a scalar is
readable from the view (`x.layout.shape[0]()`), and a caller therefore cannot
hand one of them a buffer alongside a length that does not describe it.

`_rows_tile` builds the contiguous instance (the activations); `_param_tile`
builds the ones whose rows may overlap.
"""


@always_inline
def _rows_tile[
    dtype: DType
](
    data: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin],
    num_rows: Int,
    row_length: Int,
) -> Rows2D[dtype]:
    """Views a contiguous `[num_rows, row_length]` activation."""
    return Rows2D[dtype](
        ptr=data, layout=row_major(Coord(num_rows, row_length))
    )


@always_inline
def _param_tile[
    dtype: DType
](
    params: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin],
    num_rows: Int,
    row_length: Int,
    row_stride: Int,
) -> Rows2D[dtype]:
    """Views a per-channel modulation vector as one `row_length` row per token.

    The row stride is the caller's rather than the shape's, which is the whole
    reason this is not a `row_major` layout: `row_stride == row_length` is a
    materialized `[num_rows, D]`, and `row_stride == 0` is the FLUX broadcast,
    which a strided layout states exactly -- a zero-stride row axis *is* "every
    row reads the same vector". The view then spans
    `(num_rows - 1) * row_stride + row_length` elements either way, which is the
    length the callers allocate.
    """
    return Rows2D[dtype](
        ptr=params,
        layout=Layout(Coord(num_rows, row_length), Coord(row_stride, Idx[1])),
    )


@always_inline
def _ada_modulate_rows[
    in_dtype: DType
](
    x: Rows2D[in_dtype],
    scale: Rows2D[in_dtype],
    shift: Rows2D[in_dtype],
    output: Rows2D[in_dtype],
    start_row: Int,
    end_row: Int,
):
    """Applies adaLN modulation to rows `[start_row, end_row)`.

    The row range is the only thing that distinguishes the whole-tile caller from
    the launch-striped one, so both run this body rather than two transcriptions
    of it. See `ada_modulate_2d` for what the arithmetic is and why the params'
    row stride is carried by their view.

    `x` is a normalized activation `[num_rows, row_length]` (`row_length` is the
    hidden dim `D`, 6144 for FLUX.2-Klein), and `output` matches it. `scale` and
    `shift` are the per-channel modulation vectors, and their row stride -- the
    one thing that differs between the two shapes the graph produces -- is
    carried by their view rather than passed alongside it:

    - a zero row stride is a single length-`D` vector shared by every row. This
      is the FLUX case -- modulation comes from a `[B, D]` time embedding and
      broadcasts across the token axis.
    - a row stride of `row_length` is distinct length-`D` params per row, i.e. a
      materialized `[num_rows, D]`.

    Both spellings are `_param_tile`'s job; from here they are one type, and the
    loop below reads a broadcast row and a materialized row identically.

    `scale == -1` collapses to `out = shift`; the anchor test exercises it.
    """
    # Full 128-byte HVX vector of I/O elements (64 for fp16/bf16), stated from
    # the register geometry with the tie asserted rather than taken from
    # `simd_width_of[in_dtype]()`. Not because that query is wrong -- it is
    # correct on this target since #114/#146 -- but because this same source is
    # compiled for the HOST by `test_modulation.mojo`, where it correctly
    # returns 8. A derived width would test a different kernel than it ships
    # (KB `kernels/tiletensor-vector-width-state-it-explicitly`).
    comptime LANES = HVX_VECTOR_SIZE // size_of[in_dtype]()
    comptime assert (
        LANES * bit_width_of[in_dtype]() == HVX_VECTOR_SIZE * 8
    ), "LANES must be exactly one HVX vector register"

    # The element alignment, stated at every access: `TileTensor.load`/`store`
    # default to 1 off-GPU where the pointer form defaulted to the element's
    # own, and on the scalar tail that difference is a halfword access lowered
    # to two byte loads and a shift. The caller's buffers are element-aligned
    # and nothing more, so that is what is stated.
    comptime ALIGN = align_of[in_dtype]()

    # The column bound is a property of the activation, so it is read off it
    # rather than passed beside it.
    var row_length = x.layout.shape[1]()

    var vectorized_end = row_length - (row_length % LANES)
    var one_vec = SIMD[ACCUM, LANES](1)

    for r in range(start_row, end_row):
        # Rank-1 row views rather than `Coord(r, j)` per access: the row offset
        # is resolved where the row is chosen, and the tiles themselves were
        # resolved once at the seam -- resolving either per access costs 4x the
        # vector work and forces spills (KB
        # `kernels/tiletensor-resolve-the-base-once-not-per-access`).
        var x_row = x.slice(r, All)
        var out_row = output.slice(r, All)
        var scale_row = scale.slice(r, All)
        var shift_row = shift.slice(r, All)

        for j in range(0, vectorized_end, LANES):
            var xv = x_row.load[LANES, ALIGN](Coord(j)).cast[ACCUM]()
            var sv = scale_row.load[LANES, ALIGN](Coord(j)).cast[ACCUM]()
            var shv = shift_row.load[LANES, ALIGN](Coord(j)).cast[ACCUM]()
            out_row.store[LANES, ALIGN](
                Coord(j), ((one_vec + sv) * xv + shv).cast[in_dtype]()
            )

        for j in range(vectorized_end, row_length):
            var xv = x_row.load[1, ALIGN](Coord(j)).cast[ACCUM]()
            var sv = scale_row.load[1, ALIGN](Coord(j)).cast[ACCUM]()
            var shv = shift_row.load[1, ALIGN](Coord(j)).cast[ACCUM]()
            out_row.store[1, ALIGN](
                Coord(j), ((Scalar[ACCUM](1) + sv) * xv + shv).cast[in_dtype]()
            )


@always_inline
def ada_modulate_2d[
    in_dtype: DType
](
    x: Rows2D[in_dtype],
    scale: Rows2D[in_dtype],
    shift: Rows2D[in_dtype],
    output: Rows2D[in_dtype],
):
    """Applies adaLN modulation: `out[r,i] = (1 + scale[r,i]) * x[r,i] + shift[r,i]`.

    `x` is a normalized activation `[num_rows, row_length]` (`row_length` is the
    hidden dim `D`, 6144 for FLUX.2-Klein), and `output` matches it. `scale` and
    `shift` are the per-channel modulation vectors, and their row stride -- the
    one thing that differs between the two shapes the graph produces -- is
    carried by their view rather than passed alongside it:

    - a zero row stride is a single length-`D` vector shared by every row. This
      is the FLUX case -- modulation comes from a `[B, D]` time embedding and
      broadcasts across the token axis.
    - a row stride of `row_length` is distinct length-`D` params per row, i.e. a
      materialized `[num_rows, D]`.

    Both spellings are `_param_tile`'s job; from here they are one type, and the
    loop below reads a broadcast row and a materialized row identically.

    `scale == -1` collapses to `out = shift`; the anchor test exercises it.
    """
    _ada_modulate_rows[in_dtype](
        x, scale, shift, output, 0, Int(x.layout.shape[0]())
    )


@always_inline
def _gate_residual_rows[
    in_dtype: DType, clamp_fp16: Bool = False
](
    x: Rows2D[in_dtype],
    y: Rows2D[in_dtype],
    gate: Rows2D[in_dtype],
    output: Rows2D[in_dtype],
    start_row: Int,
    end_row: Int,
):
    """Applies the gated residual to rows `[start_row, end_row)`.

    The row range is the only thing that distinguishes the whole-tile caller from
    the launch-striped one, as it is for `_ada_modulate_rows`. See
    `gate_residual_2d` for the arithmetic and the clamp.

    `x` is the residual stream and `y` the sub-block output (attention or FF),
    both `[num_rows, row_length]`, and `output` matches them. `gate` is a
    per-channel vector like the modulation params, and carries the same choice
    in its own row stride: zero shares one `[D]` vector across the token rows
    (the FLUX broadcast), `row_length` is a materialized `[num_rows, D]`.
    `gate == 0` collapses to `out = x`; the anchor test exercises it.

    `clamp_fp16` bounds the sum to fp16's finite range before the store. It is
    the guard `transformer_flux2.py` applies to the stream a block closes over,
    and a no-op for any value already representable.
    """
    comptime LANES = HVX_VECTOR_SIZE // size_of[in_dtype]()
    comptime assert (
        LANES * bit_width_of[in_dtype]() == HVX_VECTOR_SIZE * 8
    ), "LANES must be exactly one HVX vector register"
    comptime ALIGN = align_of[in_dtype]()

    var row_length = x.layout.shape[1]()

    var vectorized_end = row_length - (row_length % LANES)

    for r in range(start_row, end_row):
        var x_row = x.slice(r, All)
        var y_row = y.slice(r, All)
        var out_row = output.slice(r, All)
        var gate_row = gate.slice(r, All)

        for j in range(0, vectorized_end, LANES):
            var xv = x_row.load[LANES, ALIGN](Coord(j)).cast[ACCUM]()
            var yv = y_row.load[LANES, ALIGN](Coord(j)).cast[ACCUM]()
            var gv = gate_row.load[LANES, ALIGN](Coord(j)).cast[ACCUM]()
            var acc = xv + gv * yv
            comptime if clamp_fp16:
                acc = acc.clamp(-FP16_FINITE_MAX, FP16_FINITE_MAX)
            out_row.store[LANES, ALIGN](Coord(j), acc.cast[in_dtype]())

        for j in range(vectorized_end, row_length):
            var xv = x_row.load[1, ALIGN](Coord(j)).cast[ACCUM]()
            var yv = y_row.load[1, ALIGN](Coord(j)).cast[ACCUM]()
            var gv = gate_row.load[1, ALIGN](Coord(j)).cast[ACCUM]()
            var acc = xv + gv * yv
            comptime if clamp_fp16:
                acc = acc.clamp(-FP16_FINITE_MAX, FP16_FINITE_MAX)
            out_row.store[1, ALIGN](Coord(j), acc.cast[in_dtype]())


@always_inline
def gate_residual_2d[
    in_dtype: DType, clamp_fp16: Bool = False
](
    x: Rows2D[in_dtype],
    y: Rows2D[in_dtype],
    gate: Rows2D[in_dtype],
    output: Rows2D[in_dtype],
):
    """Gated residual: `out[r,i] = x[r,i] + gate[r,i] * y[r,i]`.

    `x` is the residual stream and `y` the sub-block output (attention or FF),
    both `[num_rows, row_length]`, and `output` matches them. `gate` is a
    per-channel vector like the modulation params, and carries the same choice
    in its own row stride: zero shares one `[D]` vector across the token rows
    (the FLUX broadcast), `row_length` is a materialized `[num_rows, D]`.
    `gate == 0` collapses to `out = x`; the anchor test exercises it.

    `clamp_fp16` bounds the sum to fp16's finite range before the store. It is
    the guard `transformer_flux2.py` applies to the stream a block closes over,
    and a no-op for any value already representable.
    """
    _gate_residual_rows[in_dtype, clamp_fp16](
        x, y, gate, output, 0, Int(x.layout.shape[0]())
    )


# ===-----------------------------------------------------------------------===#
# Device services + the launch-striped entries.
#
# The `external_call`s are the same `qshim` queries `npu_kernel.mojo` spells out,
# repeated here because a standalone `mojo build --emit=object` for the bare-metal
# Hexagon triple cannot reach the overlay. They are undefined until the link binds
# them against `qshim_qurt.c`; a host build that never calls the entries leaves
# them unreferenced, which is what lets the algorithm be tested off-device.
#
# The entries below are what the FLUX dispatch path reaches (megakernel slice 4);
# the `hvx_*` exports at the bottom of the file are the standalone ones. They
# differ in one way: the rows divide across the launch's four compute threads by
# launch index, so each participant runs the row body over the run it owns and the
# union covers every row exactly once.
# ===-----------------------------------------------------------------------===#

comptime MOD_DTYPE = DType.float16
"""The fp16 instantiation dtype — widest HVX lane count and FLUX's I/O dtype."""

comptime MOD_DTYPE_BF = DType.bfloat16
"""The bf16 instantiation dtype. Same 64-lane width; v81's qfloat lowering of
the bf16 multiply is what dispatching the bf16 entries confirms on silicon."""

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


comptime MOD_WITNESS_ARM_WORD = 8
"""THE ARM IS ECHOED, NOT ASSUMED (the conv selector's word-18 rule, applied
here after order 229's double-LN incident): the two adaLN entries store which
arm this skel actually computes — 0 = plain modulate, 1 = fused LN+modulate —
so the host CONFIGURES ITSELF from the kernel's own answer instead of trusting
an env var to match a comptime define. Word 8 sits beyond the shared witness
prologue (words 0-2) and the per-thread ids (words 3..3+threads-1, five
threads on this part = 3..7). Every thread stores the same constant, which is
the same benign raciness as the thread-id stores. `gate_residual` shares
`_modulation_witness` but NOT this word — its entry has no arm to echo."""


@always_inline
def _modulation_witness[
    dtype: DType
](
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    thread_idx: Int,
):
    """Reports the compiled width, the fp32 accumulator lane count, the
    participant count, and the RTOS thread each launch index ran on — the
    softmax/layernorm/silu witness layout. The width is the one worth reading: a
    build that resolved it against the host would still compute the right affine
    at a lane count that says the vector units never ran.

    `witness` stays a raw pointer where the data does not: it is a side channel
    of fixed slots indexed by launch thread, and a layout over it would state a
    geometry it does not have.
    """
    comptime W = HVX_VECTOR_SIZE // size_of[dtype]()
    if thread_idx == 0:
        witness.store(0, UInt32(W))
        witness.store(1, UInt32(SIMD[ACCUM, W]().size))
        witness.store(2, UInt32(_qshim_query(QSHIM_QUERY_NUM_THREADS)))
    witness.store(3 + thread_idx, _qurt_thread_id())


@always_inline
def _thread_rows(num_rows: Int, thread_id: UInt32) -> Tuple[Int, Int]:
    """The `[start, end)` run of rows this launch index owns.

    Each of the four compute threads owns a contiguous run; the fifth (DMA)
    thread and any launch index past the row count get the empty run `(0, 0)`,
    which the row body then loops zero times over. The split stays hand-written:
    `.distribute()` would give a *cyclic* partition, and this part wants blocked
    (KB `kernels/distribute-is-cyclic-this-part-wants-blocked`).

    `thread_id` is an argument rather than a shim read so the partition is
    testable off-device; only the two entries below read the shim.
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
def ada_modulate_striped[
    in_dtype: DType
](
    x: Rows2D[in_dtype],
    scale: Rows2D[in_dtype],
    shift: Rows2D[in_dtype],
    output: Rows2D[in_dtype],
    thread_id: UInt32,
):
    """Applies adaLN modulation to the rows `thread_id` owns."""
    var rows = _thread_rows(Int(x.layout.shape[0]()), thread_id)
    _ada_modulate_rows[in_dtype](x, scale, shift, output, rows[0], rows[1])


@always_inline
def gate_residual_striped[
    in_dtype: DType, clamp_fp16: Bool = False
](
    x: Rows2D[in_dtype],
    y: Rows2D[in_dtype],
    gate: Rows2D[in_dtype],
    output: Rows2D[in_dtype],
    thread_id: UInt32,
):
    """Applies the gated residual to the rows `thread_id` owns."""
    var rows = _thread_rows(Int(x.layout.shape[0]()), thread_id)
    _gate_residual_rows[in_dtype, clamp_fp16](
        x, y, gate, output, rows[0], rows[1]
    )


@always_inline
def _ada_modulate_entry[
    dtype: DType
](
    x: Rows2D[dtype],
    scale: Rows2D[dtype],
    shift: Rows2D[dtype],
    output: Rows2D[dtype],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
):
    """adaLN modulation over the rows this launch index owns."""
    var thread_idx = _npu_thread_idx()
    _modulation_witness[dtype](witness, thread_idx)
    witness.store(MOD_WITNESS_ARM_WORD, UInt32(0))
    ada_modulate_striped[dtype](x, scale, shift, output, UInt32(thread_idx))


# ===----------------------------------------------------------------------===
# Fused no-affine LayerNorm + adaLN modulation (the SKYE_ADA_FUSED_LN arm).
#
# Every adaLN site in the FLUX.2 block stack is a `layernorm` dispatch whose
# output is consumed by exactly one `ada_modulate` dispatch (dbl norm1/norm2,
# sgl norm, norm_out). Fusing deletes the normalized intermediate's DDR round
# trip and one dispatch per pair; `x` here is the PRE-norm activation.
#
# Bit-identity with the two-kernel sequence is load-bearing and hangs on one
# line: the normalized value is rounded to the I/O dtype (`_apply`'s trailing
# cast) BEFORE the modulate arithmetic reads it back up to fp32 — exactly the
# fp16 the standalone layernorm would have stored and `_ada_modulate_rows`
# would have loaded. Dropping that round would be numerically better and is
# deliberately not done: the arming gate is bit-identity.
# ===----------------------------------------------------------------------===


def _ada_modulate_ln_rows[
    in_dtype: DType
](
    x: Rows2D[in_dtype],
    scale: Rows2D[in_dtype],
    shift: Rows2D[in_dtype],
    output: Rows2D[in_dtype],
    eps: Float32,
    start_row: Int,
    end_row: Int,
):
    """Normalizes (no affine) then modulates rows `[start_row, end_row)`.

    The mean/variance passes and the normalize arithmetic are
    `LayerNorm3Pass`'s own; only its pass-3 store is replaced by the modulate
    body, which consumes the fp16-rounded normalized vector in place of a
    loaded one.
    """
    comptime LANES = HVX_VECTOR_SIZE // size_of[in_dtype]()
    comptime assert (
        LANES * bit_width_of[in_dtype]() == HVX_VECTOR_SIZE * 8
    ), "LANES must be exactly one HVX vector register"
    comptime ALIGN = align_of[in_dtype]()

    var row_length = x.layout.shape[1]()
    var vectorized_end = row_length - (row_length % LANES)
    var one_vec = SIMD[ACCUM, LANES](1)

    var ln = LayerNorm3Pass[in_dtype](x, eps)

    for r in range(start_row, end_row):
        var x_row = x.slice(r, All)
        var out_row = output.slice(r, All)
        var scale_row = scale.slice(r, All)
        var shift_row = shift.slice(r, All)

        var mean = ln.pass1(x_row)
        var variance = ln.pass2(x_row, mean)
        var inv_std = Float32(1) / sqrt(variance + eps)

        for j in range(0, vectorized_end, LANES):
            var n = ln._apply(x_row.load[LANES](Coord(j)), mean, inv_std)
            var xv = n.cast[ACCUM]()
            var sv = scale_row.load[LANES, ALIGN](Coord(j)).cast[ACCUM]()
            var shv = shift_row.load[LANES, ALIGN](Coord(j)).cast[ACCUM]()
            out_row.store[LANES, ALIGN](
                Coord(j), ((one_vec + sv) * xv + shv).cast[in_dtype]()
            )

        for j in range(vectorized_end, row_length):
            var n = (
                (x_row.load[1, ALIGN](Coord(j)).cast[DType.float32]() - mean)
                * inv_std
            ).cast[in_dtype]()
            var xv = n.cast[ACCUM]()
            var sv = scale_row.load[1, ALIGN](Coord(j)).cast[ACCUM]()
            var shv = shift_row.load[1, ALIGN](Coord(j)).cast[ACCUM]()
            out_row.store[1, ALIGN](
                Coord(j), ((Scalar[ACCUM](1) + sv) * xv + shv).cast[in_dtype]()
            )


@always_inline
def ada_modulate_ln_striped[
    in_dtype: DType
](
    x: Rows2D[in_dtype],
    scale: Rows2D[in_dtype],
    shift: Rows2D[in_dtype],
    output: Rows2D[in_dtype],
    eps: Float32,
    thread_id: UInt32,
):
    """Fused norm+modulate over the rows `thread_id` owns."""
    var rows = _thread_rows(Int(x.layout.shape[0]()), thread_id)
    _ada_modulate_ln_rows[in_dtype](
        x, scale, shift, output, eps, rows[0], rows[1]
    )


@always_inline
def _ada_modulate_ln_entry[
    dtype: DType
](
    x: Rows2D[dtype],
    scale: Rows2D[dtype],
    shift: Rows2D[dtype],
    output: Rows2D[dtype],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    eps: Float32,
):
    """Fused norm+modulate over the rows this launch index owns."""
    var thread_idx = _npu_thread_idx()
    _modulation_witness[dtype](witness, thread_idx)
    witness.store(MOD_WITNESS_ARM_WORD, UInt32(1))
    ada_modulate_ln_striped[dtype](
        x, scale, shift, output, eps, UInt32(thread_idx)
    )


@always_inline
def _gate_residual_entry[
    dtype: DType, clamp_fp16: Bool = False
](
    x: Rows2D[dtype],
    y: Rows2D[dtype],
    gate: Rows2D[dtype],
    output: Rows2D[dtype],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
):
    """Gated residual over the rows this launch index owns."""
    var thread_idx = _npu_thread_idx()
    _modulation_witness[dtype](witness, thread_idx)
    gate_residual_striped[dtype, clamp_fp16](
        x, y, gate, output, UInt32(thread_idx)
    )


# ===-----------------------------------------------------------------------===#
# C-ABI entry points for the FastRPC skel. `abi("C")` follows the argument list.
#
# These are the seam, and the last place in the file a raw pointer appears. The
# skel's argument list is the signed contract and is not ours to move, so each
# shell turns it into views once -- which is also where the stride that selects
# the broadcast stops being a value anyone can forget to pass on.
# ===-----------------------------------------------------------------------===#


@export
def hvx_ada_modulate_f16(
    x: UnsafePointer[Scalar[DType.float16], MutUnsafeAnyOrigin],
    scale: UnsafePointer[Scalar[DType.float16], MutUnsafeAnyOrigin],
    shift: UnsafePointer[Scalar[DType.float16], MutUnsafeAnyOrigin],
    output: UnsafePointer[Scalar[DType.float16], MutUnsafeAnyOrigin],
    num_rows: Int,
    row_length: Int,
    param_row_stride: Int,
) abi("C"):
    """C-ABI fp16 entry: `out = (1 + scale) * x + shift`."""
    comptime DT = DType.float16
    ada_modulate_2d[DT](
        _rows_tile[DT](x, num_rows, row_length),
        _param_tile[DT](scale, num_rows, row_length, param_row_stride),
        _param_tile[DT](shift, num_rows, row_length, param_row_stride),
        _rows_tile[DT](output, num_rows, row_length),
    )


@export
def hvx_ada_modulate_bf16(
    x: UnsafePointer[Scalar[DType.bfloat16], MutUnsafeAnyOrigin],
    scale: UnsafePointer[Scalar[DType.bfloat16], MutUnsafeAnyOrigin],
    shift: UnsafePointer[Scalar[DType.bfloat16], MutUnsafeAnyOrigin],
    output: UnsafePointer[Scalar[DType.bfloat16], MutUnsafeAnyOrigin],
    num_rows: Int,
    row_length: Int,
    param_row_stride: Int,
) abi("C"):
    """C-ABI bf16 entry: `out = (1 + scale) * x + shift`."""
    comptime DT = DType.bfloat16
    ada_modulate_2d[DT](
        _rows_tile[DT](x, num_rows, row_length),
        _param_tile[DT](scale, num_rows, row_length, param_row_stride),
        _param_tile[DT](shift, num_rows, row_length, param_row_stride),
        _rows_tile[DT](output, num_rows, row_length),
    )


@export
def hvx_gate_residual_f16(
    x: UnsafePointer[Scalar[DType.float16], MutUnsafeAnyOrigin],
    y: UnsafePointer[Scalar[DType.float16], MutUnsafeAnyOrigin],
    gate: UnsafePointer[Scalar[DType.float16], MutUnsafeAnyOrigin],
    output: UnsafePointer[Scalar[DType.float16], MutUnsafeAnyOrigin],
    num_rows: Int,
    row_length: Int,
    gate_row_stride: Int,
) abi("C"):
    """C-ABI fp16 entry: `out = x + gate * y`."""
    comptime DT = DType.float16
    gate_residual_2d[DT](
        _rows_tile[DT](x, num_rows, row_length),
        _rows_tile[DT](y, num_rows, row_length),
        _param_tile[DT](gate, num_rows, row_length, gate_row_stride),
        _rows_tile[DT](output, num_rows, row_length),
    )


@export
def hvx_gate_residual_bf16(
    x: UnsafePointer[Scalar[DType.bfloat16], MutUnsafeAnyOrigin],
    y: UnsafePointer[Scalar[DType.bfloat16], MutUnsafeAnyOrigin],
    gate: UnsafePointer[Scalar[DType.bfloat16], MutUnsafeAnyOrigin],
    output: UnsafePointer[Scalar[DType.bfloat16], MutUnsafeAnyOrigin],
    num_rows: Int,
    row_length: Int,
    gate_row_stride: Int,
) abi("C"):
    """C-ABI bf16 entry: `out = x + gate * y`."""
    comptime DT = DType.bfloat16
    gate_residual_2d[DT](
        _rows_tile[DT](x, num_rows, row_length),
        _rows_tile[DT](y, num_rows, row_length),
        _param_tile[DT](gate, num_rows, row_length, gate_row_stride),
        _rows_tile[DT](output, num_rows, row_length),
    )


@export
def hvx_gate_residual_clamp_f16(
    x: UnsafePointer[Scalar[DType.float16], MutUnsafeAnyOrigin],
    y: UnsafePointer[Scalar[DType.float16], MutUnsafeAnyOrigin],
    gate: UnsafePointer[Scalar[DType.float16], MutUnsafeAnyOrigin],
    output: UnsafePointer[Scalar[DType.float16], MutUnsafeAnyOrigin],
    num_rows: Int,
    row_length: Int,
    gate_row_stride: Int,
) abi("C"):
    """C-ABI fp16 entry clamped to fp16's finite range.

    There is no bf16 counterpart: bf16 carries fp32's exponent range, so the
    bound cannot be reached, and the reference guards on fp16 alone.
    """
    comptime DT = DType.float16
    gate_residual_2d[DT, clamp_fp16=True](
        _rows_tile[DT](x, num_rows, row_length),
        _rows_tile[DT](y, num_rows, row_length),
        _param_tile[DT](gate, num_rows, row_length, gate_row_stride),
        _rows_tile[DT](output, num_rows, row_length),
    )
