from layout.coord import All, Coord
from layout.tile_layout import row_major
from layout.tile_tensor import TileTensor

from std.ffi import external_call
from std.math import ceildiv
from std.memory import AddressSpace, bitcast
from std.sys import align_of, bit_width_of, size_of

from .hmx_crouton import HMX_TILE, CroutonPanelOf, HmxSplit

# ===--------------------------------------------------------------------------===
# HVX SiLU and fused SwiGLU — hexagonv81 (`+hvx,+hvx-length128b,+hvx-qfloat`).
#
# Addressing is the tile model's: the caller's extents become a layout once at
# the seam, and every access below that is a coordinate into it. `r * row_pitch`
# appears nowhere. The translation is KB
# `kernels/tile-model-porting-a-pointer-kernel-to-tiletensor`; that the machine
# code is unchanged by it is checked, not assumed, by
# `plugins/internal-skye/utils/kernel_asm_compare.sh`.
#
# The seam is the `@export abi("C")` line and nothing below it. The FastRPC skel
# hands over pointers and extents, the entry point turns that pair into a tile,
# and `silu_2d` / `swiglu_2d` take tiles — so a caller cannot hand them a length
# that disagrees with the buffer, and the row and tail bounds are read back off
# the layout rather than travelling beside it.
#
# SwiGLU's `row_pitch` is the one extent worth reading twice. It is the row
# stride of the *input*, which is what lets a packed gate/up pair be a column
# slice of a wider fused projection; the output is dense at `inner_dim`. Those
# are two different layouts over two different pointers, and typing the
# arguments is what puts that contract in the signature: `swiglu_2d`'s pitch is
# now a property of the input tile it is given, not a scalar it is trusted with.
# The C scalar list itself is frozen ABI (`NPUKernels.def` selector 14) and is
# untouched here.
# ===--------------------------------------------------------------------------===

comptime HVX_VECTOR_SIZE = 128
"""Size of an HVX vector in bytes (`+hvx-length128b`).

The kernels take their lane count from this register geometry rather than from
`simd_width_of`, and not because the target query is wrong: #114/#146 fixed it
and the derived spelling now compiles identically on v81. It is stated because
this same source is compiled for the HOST by `test_silu_swiglu.mojo`, where the
query correctly returns 8 — a derived width would test a different kernel than
it ships. KB `kernels/tiletensor-vector-width-state-it-explicitly`.
"""

# The sigmoid and the gated product run in fp32 regardless of the fp16/bf16 I/O.
# A saturating fp16 sigmoid rounds to 0 or 1 several values early, and SwiGLU's
# gated product `silu(gate) * up` then loses the small-magnitude tail it depends
# on; this mirrors the softmax and rms_norm kernels beside it, which reduce in
# fp32 for the same reason.
comptime ACCUM = DType.float32

comptime SILU_DTYPE = DType.float16
"""The fp16 instantiation dtype — widest HVX lane count and FLUX's I/O dtype."""

comptime SILU_DTYPE_BF = DType.bfloat16
"""The bf16 instantiation dtype. Same 64-lane width; v81's qfloat lowering of
the bf16 exp path is what dispatching the bf16 entries confirms on silicon."""


comptime RowMajor2D[dtype: DType] = TileTensor[
    dtype, type_of(row_major(Coord(0, 0))), origin=MutUnsafeAnyOrigin
]
"""The `(num_rows, pitch)` view the C-ABI entries' pointers are wrapped in, and
the argument type of everything below them.

Both extents are dynamic, so the layout carries runtime values; what it makes
compile-time is the *rank* and the row-major relationship between them, which is
what removes `r * pitch` from every access.
"""


comptime SCALAR_ALIGN[dtype: DType] = align_of[Scalar[dtype]]()
"""Natural alignment of one element, stated on the scalar tail accesses.

`TileTensor.load`/`.store` default their alignment to `align_of[...]() if
is_gpu() else 1` (`tile_tensor.mojo:798, 838`), so off a GPU every access is
assumed byte-aligned. On Hexagon that costs a real instruction: a 1-alignment
fp16 load is emitted as two `memub` plus a shift-or rather than one `memuh`,
and the byte-pair store defeats the tail loop's post-increment addressing. An
element of a `Scalar[dtype]` array is aligned by construction, so saying so is
a true statement rather than a hint. The vector accesses keep the default:
HVX's unaligned `vmemu` is what both spellings already emit, and the buffers
are not 128-aligned when the packed pair is a column slice.
"""


# ===--------------------------------------------------------------------------===
# Vector exponential and reciprocal -- `std.math.exp` and a bare `/` both
# SCALARIZE on `hexagon-unknown-unknown-elf`: Hexagon has no vector floating
# divide (`matmul_kernel.mojo:397` states the integer case; a 64-lane fp32
# `fdiv` scalarizes the same way), and `exp` compiles to 64 out-of-line
# `floorf` calls per vector (measured off the shipped object, `FACTS.md:454`:
# 71 out-of-line calls in `mojo_swiglu_f16`). Together they were 96% of SiLU's
# per-element cost (`VAE_DEVICE_CRITIQUE.md` section 3).
#
# `_exp_hvx` is the attention kernel's vector exp (M4.0b, `attention_kernel.mojo`),
# duplicated here for the same reason `softmax_kernel.mojo`'s copy is: a
# numerics change belongs with its own device confirm, not with a cross-module
# dependency on the attention family.
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
    accuracy argument (degree-6 Horner, ~1.2e-7 relative truncation). `silu`
    below only ever calls this at `-abs(x)`, so the domain matches exactly --
    no widening past what this was built and verified for.
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
def _recip_hvx[width: Int](x: SIMD[ACCUM, width]) -> SIMD[ACCUM, width]:
    """Lane-wise `1 / x` for `x` in `[1, 2]`, bit-trick seed plus two
    Newton-Raphson refinements -- lane-wise fp32 multiply/add/sub and one
    reinterpretation only, the same no-scalarizing-op discipline as `_exp_hvx`.

    The seed (`0x7EEEEEEE - bits(x)`, the standard fast-reciprocal magic
    constant) is only valid across a bounded exponent range -- verified
    numerically, not assumed: at this same seed, `x` near float32's largest
    representable value makes `bits(x) > 0x7EEEEEEE`, so the subtraction
    underflows to a NaN bit pattern. `silu` below only ever calls this at
    `1 + exp(-|x|)`, always in `[1, 2]`, which sits nowhere near that
    boundary -- restricting the domain is what makes this safe, not a
    wider-range fix to the seed. Two Newton iterations bring the worst-case
    relative error over `[1, 2]` to ~2e-5 (verified by sweep), three orders
    below fp16's ~1e-3 resolution.
    """
    var bits = bitcast[DType.int32, width](x)
    var y = bitcast[ACCUM, width](SIMD[DType.int32, width](0x7EEEEEEE) - bits)
    y = y * (SIMD[ACCUM, width](2.0) - x * y)
    y = y * (SIMD[ACCUM, width](2.0) - x * y)
    return y


@always_inline
def silu[width: Int](x: SIMD[ACCUM, width]) -> SIMD[ACCUM, width]:
    """Computes SiLU `x * sigmoid(x)` via the numerically stable split
    `sigmoid(x) = 1 / (1 + exp(-|x|))` for `x >= 0`, `exp(-|x|) / (1 + exp(-|x|))`
    for `x < 0` -- the standard branchless form (every ML framework's `sigmoid`
    uses it) that keeps the `exp` argument non-positive regardless of `x`'s
    sign or magnitude.

    That matters here specifically because it means `_exp_hvx` needs no domain
    widening: both branches exponentiate `-abs(x)`, `_exp_hvx`'s native `x <= 0`
    domain. A single-branch `exp(min(-x, 88))` form (the naive port) instead
    pushes the exp argument up to +88 for very negative `x`, and with it
    `1 + exp(arg)` up to ~1.65e38 -- verified numerically to break
    `_recip_hvx`'s bit-trick seed at that extreme (see there). Splitting on
    sign keeps `1 + exp(-|x|)` in `[1, 2]` always, where both the exp and the
    reciprocal are cheap and nowhere near a failure boundary. Every call site
    widens to fp32 first, so the arithmetic runs in fp32.

    Args:
        x: Inputs (fp32 at every call site).

    Returns:
        `x * sigmoid(x)`, with the width of `x`.
    """
    var neg_abs_x = -abs(x)
    var z = _exp_hvx(neg_abs_x)
    var r = _recip_hvx(SIMD[ACCUM, width](1.0) + z)
    # `.lt()` rather than the `<` operator: at `width == 1` (the scalar tail
    # loop) a bare comparison operator resolves to `Bool`, which has no
    # `.select()`; the explicitly-typed method call keeps this a
    # `SIMD[DType.bool, width]` at every width (KB pattern in
    # `groupnorm_kernel.mojo`'s `_pred`).
    var is_neg: SIMD[DType.bool, width] = x.lt(SIMD[ACCUM, width](0.0))
    var sigmoid = is_neg.select(z * r, r)
    return x * sigmoid


@always_inline
def _silu_rows[
    in_dtype: DType
](
    input: RowMajor2D[in_dtype],
    output: RowMajor2D[in_dtype],
    start_row: Int,
    end_row: Int,
):
    """Applies SiLU to rows `[start_row, end_row)` of the tiles.

    The row range is the only thing that distinguishes the whole-tile caller
    below from the launch-striped one: both run this body, so the kernel a host
    twin scores and the kernel a launch runs are the same instructions rather
    than two transcriptions of one idea.
    """
    comptime LANES = HVX_VECTOR_SIZE // size_of[in_dtype]()
    comptime assert (
        LANES * bit_width_of[in_dtype]() == HVX_VECTOR_SIZE * 8
    ), "LANES must be exactly one HVX vector register"

    var row_length = Int(input.dim[1]())

    var vectorized_end = row_length - (row_length % LANES)
    for r in range(start_row, end_row):
        # A rank-1 row view, not the 2-D tile plus a row index: slicing resolves
        # `r * row_length` once, where the row is chosen, rather than at every
        # access inside the two loops below.
        var in_row = input.slice(r, All)
        var out_row = output.slice(r, All)
        for j in range(0, vectorized_end, LANES):
            var x = in_row.load[LANES](Coord(j)).cast[ACCUM]()
            out_row.store[LANES](Coord(j), silu(x).cast[in_dtype]())
        for j in range(vectorized_end, row_length):
            var x = in_row.load[1, alignment=SCALAR_ALIGN[in_dtype]](
                Coord(j)
            ).cast[ACCUM]()
            # Explicit `[1]`: a width-1 `SIMD` argument here resolves to the
            # bare scalar type at the call site, and comptime-parameter
            # inference does not unify a bare scalar against `SIMD[.., width]`.
            out_row.store[1, alignment=SCALAR_ALIGN[in_dtype]](
                Coord(j), silu[1](x).cast[in_dtype]()
            )


@always_inline
def silu_2d[
    in_dtype: DType
](input: RowMajor2D[in_dtype], output: RowMajor2D[in_dtype]):
    """Applies SiLU elementwise over `[num_rows, row_length]`: `out = silu(in)`.

    The VAE ResNets use bare SiLU. I/O is `in_dtype` (fp16/bf16); every element
    is widened to fp32, passed through `silu`, and narrowed once on store.

    Both tiles are `[num_rows, row_length]`, and that is the whole argument
    list: the extents arrive inside the layout the caller built at the seam
    rather than as scalars beside it, so there is no length here that could
    disagree with the buffer it describes.
    """
    _silu_rows[in_dtype](input, output, 0, Int(input.dim[0]()))


@always_inline
def _swiglu_rows[
    in_dtype: DType
](
    input: RowMajor2D[in_dtype],
    output: RowMajor2D[in_dtype],
    start_row: Int,
    end_row: Int,
):
    """Applies the fused SwiGLU gate to rows `[start_row, end_row)`.

    The row range is the only thing that distinguishes the whole-tile caller
    below from the launch-striped one, as it is for `_silu_rows`.
    """
    comptime LANES = HVX_VECTOR_SIZE // size_of[in_dtype]()
    comptime assert (
        LANES * bit_width_of[in_dtype]() == HVX_VECTOR_SIZE * 8
    ), "LANES must be exactly one HVX vector register"

    # The column bound comes off the dense output: its row extent is
    # `inner_dim`, the half-width the gate and up columns are counted in. The
    # input's own row extent is the pitch and is deliberately never read — it is
    # an address property, and the layout is what applies it.
    var inner_dim = Int(output.dim[1]())

    var vectorized_end = inner_dim - (inner_dim % LANES)
    for r in range(start_row, end_row):
        var in_row = input.slice(r, All)
        var out_row = output.slice(r, All)
        # `inner_dim + j` is a column of the gate/up row, not an address: the
        # up half starts one `inner_dim` into the same row view.
        for j in range(0, vectorized_end, LANES):
            var g = in_row.load[LANES](Coord(j)).cast[ACCUM]()
            var u = in_row.load[LANES](Coord(inner_dim + j)).cast[ACCUM]()
            out_row.store[LANES](Coord(j), (silu(g) * u).cast[in_dtype]())
        for j in range(vectorized_end, inner_dim):
            var g = in_row.load[1, alignment=SCALAR_ALIGN[in_dtype]](
                Coord(j)
            ).cast[ACCUM]()
            var u = in_row.load[1, alignment=SCALAR_ALIGN[in_dtype]](
                Coord(inner_dim + j)
            ).cast[ACCUM]()
            out_row.store[1, alignment=SCALAR_ALIGN[in_dtype]](
                Coord(j), (silu[1](g) * u).cast[in_dtype]()
            )


@always_inline
def swiglu_2d[
    in_dtype: DType
](input: RowMajor2D[in_dtype], output: RowMajor2D[in_dtype]):
    """Applies the fused SwiGLU gate: `out = silu(gate) * up`, split-half.

    `input` is `[num_rows, row_pitch]` and `output` `[num_rows, inner_dim]`
    dense; the two shapes are what distinguishes the arguments, and they are
    the reason this takes two differently-shaped tiles rather than one shape
    twice. Each input row's first `inner_dim` elements are the gate and the
    next `inner_dim` the up projection, the two contiguous halves
    `chunk(x, 2, axis=-1)` produces in `Flux2SwiGLU`. The input's row stride is
    `2*inner_dim` when it is dense, and the projection's full width when the
    packed pair is a column slice of a wider fused projection — the layout
    FLUX.2's single-stream blocks produce, whose start rides in the tile's own
    base pointer. That pitch is now carried by the input tile's layout instead
    of by a scalar this function has to be trusted with. gate, up, and the
    product are handled in fp32; the result is narrowed once on store. This is
    the FFN gate on every FLUX.2-Klein MMDiT block and the Qwen3 encoder.

    The row count comes off the dense output, whose row extent is the half-width
    the gate and up columns are counted in.
    """
    _swiglu_rows[in_dtype](input, output, 0, Int(output.dim[0]()))


@always_inline
def swiglu_croutons[
    in_dtype: DType, in_space: AddressSpace, out_space: AddressSpace, //
](
    input: CroutonPanelOf[in_dtype, in_space],
    output: CroutonPanelOf[in_dtype, out_space],
    hidden: Int,
    row_tile0: Int,
    row_tiles: Int,
):
    """`swiglu_2d` over crouton-tiled I/O: annihilation rung 1's elementwise leg.

    Same gate, same numerics — `silu` widened to fp32 per lane, the product
    narrowed once on store — over buffers whose bytes are row-panel-major
    croutons (`hmx_crouton.mojo`'s activation rule) instead of row-major rows.
    The pairing needs ZERO cross-tile bookkeeping: with `hidden` a multiple of
    `HMX_TILE`, gate element `(r, j)` and up element `(r, hidden + j)` sit at
    the SAME in-tile coordinate of tiles `(rt, jt)` and `(rt, h_tiles + jt)`,
    and the output element lands at that coordinate of output tile
    `(rt, jt)` — so the in-tile permutation cancels and each output crouton is
    one lane-wise pass over two input croutons. The design doc's §2.2 states
    this as the elementwise contract; this is that contract as code.

    `input` is the producing matmul's `[rows, 2 * hidden]` output as croutons
    (its `output_native` artefact), `output` the consuming matmul's
    `[rows, hidden]` activation as croutons (its `activation_native`
    artefact). The tile-index rule is `hmx_native_row_panel`'s; restating it
    here per operand is what this function's own test pins.

    `row_tile0`/`row_tiles` are the 32-row blocks this call owns, so a launch
    can stripe row panels across threads the way `_thread_rows` stripes rows;
    whole-tensor callers pass `(0, rows // HMX_TILE)`.

    The row count and `hidden` must be whole numbers of crouton tiles. The
    caller owns that refusal by name — the same residual-tile discipline
    `hmx_repack_weight_cpm` states — because a partial tile has no crouton
    form for this function to fall back to.
    """
    comptime LANES = HVX_VECTOR_SIZE // size_of[in_dtype]()
    comptime assert (
        LANES == HMX_TILE * HmxSplit[in_dtype]
    ), "one gate vector must be exactly one 128-byte crouton slab pair"
    var h_tiles = hidden // HMX_TILE
    var in_row_tiles = 2 * h_tiles
    for rt in range(row_tile0, row_tile0 + row_tiles):
        for jt in range(h_tiles):
            var g_tile = rt * in_row_tiles + jt
            var u_tile = g_tile + h_tiles
            var o_tile = rt * h_tiles + jt
            for s in range(HMX_TILE // HmxSplit[in_dtype]):
                var g = input.load[LANES, alignment=HVX_VECTOR_SIZE](
                    Coord(g_tile, s, 0, 0)
                ).cast[ACCUM]()
                var u = input.load[LANES, alignment=HVX_VECTOR_SIZE](
                    Coord(u_tile, s, 0, 0)
                ).cast[ACCUM]()
                output.store[LANES, alignment=HVX_VECTOR_SIZE](
                    Coord(o_tile, s, 0, 0), (silu(g) * u).cast[in_dtype]()
                )


# ===--------------------------------------------------------------------------===
# Device services + the launch-striped entries.
#
# The `external_call`s are the same `qshim` queries `npu_kernel.mojo` spells out,
# repeated here because a standalone `mojo build --emit=object` for the bare-metal
# Hexagon triple cannot reach the overlay. They are undefined until the link binds
# them against `qshim_qurt.c`; a host build that never calls the entries leaves
# them unreferenced, which is what lets the algorithm be tested off-device.
#
# The entries below are what the FLUX dispatch path reaches (megakernel slice 4).
# They differ from the `hvx_*` exports at the bottom of the file in one way: the
# rows divide across the launch's four compute threads by launch index, so each
# participant runs `_silu_rows` / `_swiglu_rows` over the run it owns and the
# union covers every row exactly once.
# ===--------------------------------------------------------------------------===

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
def _silu_swiglu_witness[
    dtype: DType
](
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    thread_idx: Int,
):
    """Reports the compiled width, the fp32 accumulator lane count, the
    participant count, and the RTOS thread each launch index ran on — the
    softmax/layernorm witness layout. The width is the one worth reading: a
    build that resolved it against the host would still compute the right
    activations at a lane count that says the vector units never ran.

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
    which every caller then loops zero times over. The split stays hand-written:
    `.distribute()` would give a *cyclic* partition, and this part wants blocked
    (KB `kernels/distribute-is-cyclic-this-part-wants-blocked`).

    `thread_id` is an argument rather than a shim read so the partition is
    testable off-device: its twin drives all four indices and checks the union
    covers every row exactly once. Only `_silu_entry` / `_swiglu_entry` below
    read the shim.
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
def silu_striped[
    in_dtype: DType
](
    input: RowMajor2D[in_dtype],
    output: RowMajor2D[in_dtype],
    thread_id: UInt32,
):
    """Applies SiLU to the rows `thread_id` owns."""
    var rows = _thread_rows(Int(input.dim[0]()), thread_id)
    _silu_rows[in_dtype](input, output, rows[0], rows[1])


@always_inline
def swiglu_striped[
    in_dtype: DType
](
    input: RowMajor2D[in_dtype],
    output: RowMajor2D[in_dtype],
    thread_id: UInt32,
):
    """Applies the fused SwiGLU gate to the rows `thread_id` owns."""
    var rows = _thread_rows(Int(output.dim[0]()), thread_id)
    _swiglu_rows[in_dtype](input, output, rows[0], rows[1])


@always_inline
def _silu_entry[
    in_dtype: DType
](
    input: RowMajor2D[in_dtype],
    output: RowMajor2D[in_dtype],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
):
    """Plain SiLU over the rows this launch index owns."""
    var thread_idx = _npu_thread_idx()
    _silu_swiglu_witness[in_dtype](witness, thread_idx)
    silu_striped[in_dtype](input, output, UInt32(thread_idx))


@always_inline
def _swiglu_entry[
    in_dtype: DType
](
    input: RowMajor2D[in_dtype],
    output: RowMajor2D[in_dtype],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
):
    """Fused SwiGLU over the rows this launch index owns."""
    var thread_idx = _npu_thread_idx()
    _silu_swiglu_witness[in_dtype](witness, thread_idx)
    swiglu_striped[in_dtype](input, output, UInt32(thread_idx))


# The four entry points below are the seam, and the only place in this file
# where a pointer meets an extent. Each builds its tiles once, outside any loop:
# resolving a tile per access instead of per construction costs 4x the vector
# work and forces spills (KB
# `kernels/tiletensor-resolve-the-base-once-not-per-access`).


@export
def hvx_silu_f16(
    input: UnsafePointer[Scalar[DType.float16], MutUnsafeAnyOrigin],
    output: UnsafePointer[Scalar[DType.float16], MutUnsafeAnyOrigin],
    num_rows: Int,
    row_length: Int,
) abi("C"):
    """C-ABI fp16 entry point for the FastRPC skel."""
    var extent = row_major(Coord(num_rows, row_length))
    silu_2d[DType.float16](
        RowMajor2D[DType.float16](ptr=input, layout=extent),
        RowMajor2D[DType.float16](ptr=output, layout=extent),
    )


@export
def hvx_silu_bf16(
    input: UnsafePointer[Scalar[DType.bfloat16], MutUnsafeAnyOrigin],
    output: UnsafePointer[Scalar[DType.bfloat16], MutUnsafeAnyOrigin],
    num_rows: Int,
    row_length: Int,
) abi("C"):
    """C-ABI bf16 entry point for the FastRPC skel."""
    var extent = row_major(Coord(num_rows, row_length))
    silu_2d[DType.bfloat16](
        RowMajor2D[DType.bfloat16](ptr=input, layout=extent),
        RowMajor2D[DType.bfloat16](ptr=output, layout=extent),
    )


@export
def hvx_swiglu_f16(
    input: UnsafePointer[Scalar[DType.float16], MutUnsafeAnyOrigin],
    output: UnsafePointer[Scalar[DType.float16], MutUnsafeAnyOrigin],
    num_rows: Int,
    inner_dim: Int,
    row_pitch: Int,
) abi("C"):
    """C-ABI fp16 entry point for the FastRPC skel."""
    swiglu_2d[DType.float16](
        RowMajor2D[DType.float16](
            ptr=input, layout=row_major(Coord(num_rows, row_pitch))
        ),
        RowMajor2D[DType.float16](
            ptr=output, layout=row_major(Coord(num_rows, inner_dim))
        ),
    )


@export
def hvx_swiglu_bf16(
    input: UnsafePointer[Scalar[DType.bfloat16], MutUnsafeAnyOrigin],
    output: UnsafePointer[Scalar[DType.bfloat16], MutUnsafeAnyOrigin],
    num_rows: Int,
    inner_dim: Int,
    row_pitch: Int,
) abi("C"):
    """C-ABI bf16 entry point for the FastRPC skel."""
    swiglu_2d[DType.bfloat16](
        RowMajor2D[DType.bfloat16](
            ptr=input, layout=row_major(Coord(num_rows, row_pitch))
        ),
        RowMajor2D[DType.bfloat16](
            ptr=output, layout=row_major(Coord(num_rows, inner_dim))
        ),
    )
