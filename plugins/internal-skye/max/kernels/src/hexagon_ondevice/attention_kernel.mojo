"""HVX-only flash attention (`mo.mha.no_cache`) for the on-device Hexagon NPU.

Track 1 of the FLUX.2-Klein prefill attention: a naive online-softmax
FlashAttention that runs entirely on the HVX vector units, with no HMX and no
vendor archive. It is the strict stepping stone the HMX flash (Track 2) shares an
algorithm and thread structure with -- correct, not yet fast -- and it unblocks
MMDiT block assembly without waiting on the HMX byte-order confirm or the
uDMA-into-VTCM fix.

Shape (FLUX.2-Klein): `head_dim = 128`, `num_heads = 48`, dense non-causal
(`NULL_MASK`), no KV cache, fp16 activations, `scale = 1/sqrt(head_dim)`. RoPE
(interleaved, theta=2000) is applied to Q/K by the verified RoPE kernel as a
prologue -- it is not in this kernel's ABI, which consumes already-roped Q/K.

Layout: Q, K, V, O are row-major fp16 `[seq_len, num_heads, head_dim]` (BSHD with
batch folded into `seq_len`, the FLUX production layout, so integration is a
graph-level concat/slice rather than a transpose). Element `(s, h, d)` sits at
`(s * num_heads + h) * head_dim + d`; a token's `head_dim` vector is contiguous,
which is what the HVX dot and axpy want, and consecutive tokens of one head are
`num_heads * head_dim` apart.

This is the package module the shipped kernel runs. `npu_kernel.mojo` used to
carry a second copy of this body "under the same obligation: a change to one is
a change to both"; what it keeps now are the two `mojo_mha_*` exports the
manifest binds, as seams. The dense seam still owns what a host cannot have --
the launch's witness, the HMX arms' grant and lock, and the cycle bracket -- and
calls the striped entry below for the vector arm, so the two arms cannot drift
on numerics.

That layout is what the caller's extents are turned into, once, at the seams at
the bottom of this file, the only place in it that still holds a pointer: a
rank-3 `row_major(Coord(seq_len, num_heads, head_dim))` tile.
A (token, head) vector is then one `slice(s, h, All)` row view, so below the seam
no address is computed by hand and neither `num_heads * head_dim` nor
`head_idx * head_dim` is restated anywhere. Every internal signature takes the
tile or the row view, never a pointer plus the extents to interpret it with, so
an extent cannot disagree with the buffer it describes. The translation is KB
`kernels/tile-model-porting-a-pointer-kernel-to-tiletensor`; that the machine
code does not regress under it is checked, not assumed, by
`plugins/internal-skye/utils/kernel_asm_compare.sh`, which measures this
spelling 36 instructions and 9 packets BELOW the hand-addressed original at
identical HVX-op and vector-spill counts.

Online-softmax recurrence, per (head, query TILE of `BM` rows), keys walked in
tiles of `BN` (dense -- every tile, no causal skip). The softmax state is per
*tile*, which is what makes one pass over a key tile serve all `BM` rows:

    m[BM] = -inf; l[BM] = 0; O[BM][D] = 0 (fp32)
    for each key tile:                         # OUTSIDE the query-row loop
      for a in tile rows:
        s[0:BN] = scale * dot(Q_a, K_b)        # QK^T, HVX fp16->fp32 MAC
        m_new   = max(m[a], vector_max s)      # one rotate-tree over the keys
        corr    = exp(m[a] - m_new)            # 0 on the first tile (m = -inf)
        p[0:BN] = exp(s - m_new)               # ONE vector exp over BN keys
        l[a]    = l[a] * corr + vector_sum p
        O_a     = corr * O_a + sum_b p[b]*V_b  # O_a in registers for the tile
        m[a]    = m_new
    O_row = O[a] / l[a]                        # fp32 accumulate, fp16 store

The key loop sits OUTSIDE the query-row loop, not inside it. That is the whole
point of `BM`: a K/V row is pulled from DDR once per `BM` query rows instead of
once per row, and the `BM - 1` rows that follow hit it in cache. `M4.0a` in
`.agentwork/flux-npu/m4-attention/ANALYSIS.md` measured the inverted nesting at
a 768x re-read of K and V against a ~0.5 MiB effective L2, which is where ~76%
of this kernel's cycles were going; `FA_BM` carries the working-set arithmetic.

All reductions and the exponential run in fp32 regardless of the fp16 I/O -- an
fp16 softmax sum saturates past 2048 -- exactly as the verified softmax kernel
does. The exp is taken over a *vector* of `BN` keys rather than once per (query,
key) pair: the scalar spelling put `floorf`, `__truncsfhf2` and `__extendhfsf2`
in the innermost loop as out-of-line calls, and the calls in turn forced the O
accumulator to spill and refill on every key (`M4.0b`). It is `_exp_hvx`, not
`std.math.exp`, because `std.math.exp` scalarizes on this target whatever width
it is handed -- see that function's own note. P stays fp32: the fp16 narrowing
of `p` was only ever there to feed a fp16 MAC that the lane-order fix removed.

The QK^T dot runs on the vector units through `_fmac_sf_hf`, the fp16 x fp16 ->
fp32 `V6_vmpy_sf_hf_acc` MAC the verified GEMV uses, folded to a scalar by an
order-independent reduction that absorbs the instruction's interleaved output
lanes. The P@V accumulate stores its O column as a vector, so it runs in plain
fp32 lanes (widen V, fp32 multiply-add) -- the lane order the store assumes; the
interleaving MAC cannot feed a vector store. Both properties survive the
restructure, and KB `kernels/hvx-fold-lane-order-audit` is the audit they answer
to: every widening multiply here is a plain `vmpy(v.hf, v.hf)` cast, every
vector-stored accumulate is `vmpy(v.sf, v.sf)` against a `vsplat` broadcast.

Host-testability. The HVX MAC is a Hexagon intrinsic that neither runs nor
lowers on the host, so the compute is parameterized on `use_hvx`: the device
entry instantiates the intrinsic MAC, the host test instantiates a portable
fp32 MAC that computes the identical `c + a*b` (fp16 widened to fp32, fp32
accumulate). `comptime if use_hvx` dead-codes the untaken branch, so the host
build never instantiates the intrinsic. Matching the fp32 reference on the host
proves the *algorithm* (tiling, the online-softmax bookkeeping, dense masking,
multi-head, the RoPE prologue composition); the actual `V6_vmpy_sf_hf_acc`
numerics and the `+hvx-qfloat` fp16<->fp32 path still need an on-device confirm,
which the device lane serializes behind the HMX byte-order work.

MAX's own Mojo/HVX only -- no HMX, no vendor archive (Track 1 is HVX-only by
design).
"""

from layout.coord import All, Coord
from layout.tile_layout import row_major
from layout.tile_tensor import TileTensor, stack_allocation

from hexagon_ondevice.hmx_crouton import HMX_ACT_PREFETCH

from std.ffi import external_call
from std.math import ceildiv
from std.memory import bitcast
from std.sys import align_of, bit_width_of, size_of
from std.sys.intrinsics import llvm_intrinsic
from std.utils import StaticTuple
from std.utils.numerics import min_or_neg_inf

comptime HVX_VECTOR_SIZE = 128
"""HVX vector register width in bytes (`+hvx-length128b` in build.sh)."""

comptime NUM_COMPUTE_THREADS: UInt32 = 4
"""Compute threads a core exposes; the fifth is the part's DMA/HMX thread."""

comptime IO_DTYPE = DType.float16
"""I/O element type of the FLUX activations."""

comptime ACCUM = DType.float32
"""Reduction, exp, softmax-state, and O-accumulator precision. An fp16 sum
saturates past 2048; the QK^T dot over `head_dim` and the running softmax sum
both need fp32."""

comptime ATT_NK = 4
"""Key rows whose QK^T dots are computed together.

The QK^T half is the cheapest large lever in this kernel and it is structural,
not latency. Two costs it carries per (query, key) pair: the Q row is reloaded
for every key although it is invariant across all of them, and each dot ends in
its own `_reduce_add` -- a 9-step rotate-and-add chain amortized over just
`D / W = 2` vector MACs, where the matmul amortizes the identical fold over 48
to 192.

Blocking the key loop fixes both. Q is hoisted to one load per query row, and
`ATT_NK` dots are accumulated and reduced together, so the reduction chains
interleave instead of serializing one at a time on rotate latency.

Register budget per query row: 2 Q vectors + `ATT_NK` K vectors +
`2 * ATT_NK` accumulator registers (an fp32 64-lane vector is a PAIR) +
`2 * DV` for the live `o_acc` -- 18 of 32 at NK=4, 30 at NK=8. Four leaves
headroom for the pass-B state.

One compiles back to the unblocked loop, which is the A/B control. The blocking
cannot change a score: each dot still accumulates the same k-vectors in the
same order into its own accumulator, and is folded by the same reduction."""

comptime MMA_W = HVX_VECTOR_SIZE // size_of[IO_DTYPE]()
"""HVX fp16 lanes per vector: 128 B / 2 B = 64.

Derived from the byte width, not from `simd_width_of[IO_DTYPE]()`. Not because
that query is broken -- KB `kernels/tiletensor-vector-width-state-it-explicitly`
records the descriptor bug as fixed (#114/#146) and this target now reports
64 -- but because this same source is compiled for the HOST by its twins, where
the query correctly returns 8. A derived width would test an 8-lane kernel and
ship a 64-lane one. It is also the width `_fmac_sf_hf` requires."""

# FLUX.2-Klein prefill attention parameters, baked into the device entry.
comptime FA_BM = 16
"""Query rows one pass over a key tile serves -- the reuse factor on K and V.

Sized against the two budgets that actually bind on this part, at the profiled
FLUX shape (S=1536 joint sequence, H=24 heads, D=128, fp16):

*DDR traffic.* A (query, key) pair touches one 256 B K row and one 256 B V row.
Ungrouped that is `H * S * S * 512 B` = **29.0 GB** per launch against 18.9 MB
compulsory -- the 768x re-read. Grouping `BM` query rows divides it by `BM`:

| BM | address stream | at the measured 35.49 GB/s (4 threads) | vs the 0.46 s compute floor |
|---:|---:|---:|---:|
|  1 | 29.0 GB | 817 ms | 178% |
|  8 |  3.6 GB | 102 ms |  22% |
| 16 |  1.8 GB |  51 ms |  11% |
| 32 |  0.9 GB |  26 ms |   6% |

*Stack.* The per-tile O accumulator is `BM * D * 4 B` and it cannot live in
registers -- one fp32 64-lane vector is a register PAIR, so one query row's O is
4 of the 32 HVX registers and eight rows would be the whole file. It is the
frame, and the frame is what caps `BM`. Measured from the v81 prologue of
`mojo_mha_no_cache_f16`:

| BM | frame | fits participant 0's ~16 KiB? |
|---:|---:|---|
|  8 |  5,888 B | yes |
| 16 |  9,984 B | yes, 6.4 KiB spare |
| 32 | 18,304 B | **no** |

Participants 1..3 get 32 KiB (`NPU_LAUNCH_STACK_BYTES`, mojo_npu_imp.c:375),
but participant 0 runs the kernel on the FastRPC invocation thread's own stack
and nothing in `QuRTDeviceContext` raises that from the SDK default, so ~16 KiB
is the budget that binds and a 32-row tile overruns it. QuRT faults such a
thread rather than corrupting a peer, which makes this the failure the device
lane is worst placed to absorb.

So 16: the largest tile that fits the smallest stack, with the DDR term already
down to a ninth of the compute floor. 32 is worth a further ~1.15x on the
modelled cycle budget and needs either a raised FastRPC stack or the accumulator
moved off the frame -- which is why this is a `comptime` knob and not a literal
in the loop."""
comptime FA_BN = 64
"""Key-tile width (`_mha_per_tile.mojo` BN).

Also the exp vector length: `BN` scores are exponentiated in one
`SIMD[f32, 64]` call, so `BN` a multiple of the 64-lane width costs no padding.
A `BN` that is not (the `BN=1` tile-invariance control) pads the score buffer up
to `NS * W` with `-inf`, whose exp is 0 and which therefore cannot win the row
max or add to the row sum."""
comptime FA_HEAD_DIM = 128
"""FLUX.2 `attention_head_dim`."""


# ===-----------------------------------------------------------------------===#
# The views the compute runs on.
# ===-----------------------------------------------------------------------===#


comptime BSHD[dtype: DType] = TileTensor[
    dtype, type_of(row_major(Coord(0, 0, 0))), origin=MutUnsafeAnyOrigin
]
"""The `(seq_len, num_heads, head_dim)` view an activation pointer is wrapped
in.

All three extents are dynamic, so the layout carries runtime values; what it
makes compile-time is the *rank* and the row-major relationship between them,
which is what removes `(s * num_heads + h) * head_dim` from every access.
"""


comptime Row[dtype: DType] = type_of(
    BSHD[dtype](
        ptr=UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin](
            unsafe_from_address=0
        ),
        layout=row_major(Coord(0, 0, 0)),
    ).slice(0, 0, All)
)
"""One (token, head)'s contiguous `head_dim` vector, as `.slice(s, h, All)`
returns it.

Spelled through `type_of` because `.slice()`'s result type is built by filtering
the parent's shape and stride type lists and has no shorter name; spelled as the
same single slice the runtime path takes so the two cannot drift.

Why a row VIEW and not the tile plus two indices at each access: the two are
equivalent to read and are NOT equivalent to compile. Re-indexing
`Coord(s, h, j)` per access re-derives `s * num_heads * head_dim`; slicing
resolves it once, where the row is chosen (KB
`kernels/tiletensor-resolve-the-base-once-not-per-access`).

Why ONE slice and not a head plane then a row: measured. Folding the head first
(`.slice(All, h, All)` per tensor, then `.slice(s, All)`) reads better and costs
**+35 instructions and +9 packets** on this kernel -- the chained view's nested
offset type defeats common-subexpression elimination of the row-stride multiply
across Q/K/V/O, emitting four `mpyi` where the single slice emits one. Resolve
the base once means once; a head fold in front of the row fold resolves it twice
and the compiler does not recover the difference.
"""


comptime RowMajor2D[dtype: DType] = TileTensor[
    dtype, type_of(row_major(Coord(0, 0))), origin=MutUnsafeAnyOrigin
]
"""The `(seq_len, seq_len)` view the masked variant's additive bias is wrapped
in. Broadcast across heads, so it has no head extent."""


comptime BiasRow[dtype: DType] = type_of(
    RowMajor2D[dtype](
        ptr=UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin](
            unsafe_from_address=0
        ),
        layout=row_major(Coord(0, 0)),
    ).slice(0, All)
)
"""One query row of the bias: every key's additive mask for that query."""


# ===-----------------------------------------------------------------------===#
# Device services (spelled out for the bare-metal Hexagon triple).
# ===-----------------------------------------------------------------------===#
#
# The same `qshim` calls the stdlib overlay lowers to, respelled here because a
# standalone `mojo build --emit=object` for `hexagon-unknown-unknown-elf` cannot
# resolve the overlay (its target gate is exact against the musl triple). Only
# the ones the head-striping and the witness need are carried over; they mirror
# the verified GEMV / softmax / RoPE example kernels exactly.

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


@always_inline
def _l2fetch_head_rows[
    D: Int
](t: BSHD[IO_DTYPE], row0: Int, head_idx: Int, rows: Int):
    """Hints `rows` of one head's `D` contiguous elements into L2, from `row0`.

    `l2fetch(Rs32,Rtt32)` takes its 2D region from the control word: height in
    `Rtt[15:0]`, width in `Rtt[31:16]`, stride in `Rtt[47:32]`. All three are
    16-bit fields, and a region that does not fit them is DROPPED rather than
    truncated -- a truncated control word names a different region, which cannot
    change a value but would spend bandwidth on lines nothing reads, so the hint
    would stop being the thing under measurement.

    Both the address and the row stride come from the layout, which is what
    makes this describe the head's slice of a `[seq, heads, D]` tensor rather
    than restate `num_heads * D` by hand.

    A prefetch moves cache lines, never data, so it cannot change any value the
    kernel computes.
    """
    var stride_bytes = Int(t.layout.stride[0]()) * size_of[IO_DTYPE]()
    comptime row_bytes = D * size_of[IO_DTYPE]()
    if rows <= 0 or rows > 0xFFFF or stride_bytes > 0xFFFF:
        return
    comptime assert row_bytes <= 0xFFFF, "l2fetch width is a 16-bit field"
    var control = (
        (UInt64(stride_bytes) << 32) | (UInt64(row_bytes) << 16) | UInt64(rows)
    )
    llvm_intrinsic["llvm.hexagon.Y5.l2fetch", NoneType](
        t.ptr_at_offset(Coord(row0, head_idx, 0)).bitcast[NoneType](), control
    )


# ===-----------------------------------------------------------------------===#
# HVX fp16 x fp16 -> fp32 multiply-accumulate (the QK^T and P@V primitive).
# ===-----------------------------------------------------------------------===#


@always_inline
def _fmac_sf_hf[
    width: Int
](
    c: SIMD[ACCUM, width],
    a: SIMD[IO_DTYPE, width],
    b: SIMD[IO_DTYPE, width],
) -> SIMD[ACCUM, width]:
    """Returns `c + a*b` through the HVX `V6_vmpy_sf_hf_acc` instruction.

    Respelled from `_fmac_sf_hf` in the verified GEMV. A single fused
    fp16 x fp16 + fp32 -> fp32 multiply-accumulate: plain
    `c + a.cast[f32]() * b.cast[f32]()` lowers to ~14 packets, so the intrinsic
    is issued directly. Result lanes land in the instruction's interleaved
    order, so its only consumer here is the QK^T dot, whose order-independent
    horizontal reduction absorbs the permutation. The P@V accumulate must not
    use it: that column is stored as a vector, and the interleave would permute
    O within each 64-lane block. The host portable MAC does not interleave, so
    the host proof never exercised this.
    """
    comptime assert (
        width == 64
    ), "V6_vmpy_sf_hf_acc operates on 64 fp16 lanes (1024-bit HVX, 128B mode)"

    var acc_i = bitcast[DType.int32, width](c)
    var a_i = bitcast[DType.int32, width // 2](a)
    var b_i = bitcast[DType.int32, width // 2](b)

    var res_i = llvm_intrinsic[
        "llvm.hexagon.V6.vmpy.sf.hf.acc.128B",
        SIMD[DType.int32, width],
        has_side_effect=False,
    ](acc_i, a_i, b_i)

    return bitcast[ACCUM, width](res_i)


@always_inline
def _mac[
    use_hvx: Bool, width: Int
](
    c: SIMD[ACCUM, width],
    a: SIMD[IO_DTYPE, width],
    b: SIMD[IO_DTYPE, width],
) -> SIMD[ACCUM, width]:
    """`c + a*b`, on the HVX MAC on device or a portable fp32 MAC on the host.

    `use_hvx=True` issues `V6_vmpy_sf_hf_acc`; `use_hvx=False` widens to fp32 and
    accumulates. Both compute the same fp16 x fp16 + fp32 -> fp32 value. The
    `comptime if` dead-codes the untaken branch, so the host build never
    instantiates the Hexagon intrinsic.
    """
    comptime if use_hvx:
        return _fmac_sf_hf(c, a, b)
    else:
        return c + a.cast[ACCUM]() * b.cast[ACCUM]()


@always_inline
def _reduce_add[
    dtype: DType, width: Int
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Horizontal sum across all lanes; every lane holds the result.

    The rotate-and-combine tree the verified kernels use, not
    `SIMD.reduce_add()`: the hand-rolled form lowers to `V6_*` HVX ops where the
    builtin scalarizes.
    """
    var reduced = x
    comptime for i in range(8, -1, -1):
        comptime if (1 << i) < width:
            reduced += reduced.rotate_right[1 << i]()
    return reduced


@always_inline
def _reduce_max[
    dtype: DType, width: Int
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Horizontal max across all lanes; every lane holds the result.

    The same rotate-and-combine tree as `_reduce_add`, spelled for `max`, and
    the same reason: the hand-rolled form lowers to `V6_*` HVX ops where
    `SIMD.reduce_max()` scalarizes. Respelled from the verified softmax kernel's
    `_reduce_max`, which is what the row max of a `BN`-key tile wants now that
    the scores arrive as a vector rather than one at a time.
    """
    var reduced = x
    comptime for i in range(8, -1, -1):
        comptime if (1 << i) < width:
            reduced = max(reduced, reduced.rotate_right[1 << i]())
    return reduced


# ===-----------------------------------------------------------------------===#
# Vector exponential (the softmax primitive).
# ===-----------------------------------------------------------------------===#
#
# `std.math.exp` is not usable in this loop, and the reason is only visible in
# the disassembly: on `hexagon-unknown-unknown-elf` it SCALARIZES. A single
# `exp(SIMD[f32, 64])` compiles to 64 `vinsert`/`vextract` pairs and 64 out-of-
# line `floorf` calls -- the shipped softmax kernel emits 130 `floorf` and 256
# `vinsert` for its two exp passes, so "the vector exp already ships" is true of
# the source and false of the machine code. Those calls are what M4.0b set out
# to delete; calling the other kernel's exp would have moved them, not removed
# them.
#
# So the range reduction is spelled out. It is `_sincos_exp_f32`'s construction
# at fp32 vector width: round-to-nearest by the magic add (`_sincos_rne_integer`
# -- "two exact-width IEEE adds and no `round` for a bare-metal target to
# miss"), a two-part ln2 so the reduced argument keeps its low bits, a Horner
# polynomial, and the power of two built by placing the exponent field. Every
# operation is a lane-wise fp32 multiply/add, a lane-wise int32 add/shift, or a
# reinterpretation -- no cross-lane traffic at all, so the interleaved-lane
# class KB `kernels/hvx-fold-lane-order-audit` gates cannot arise here.

comptime EXP_LOG2_E = Scalar[ACCUM](1.44269504088896340736)
"""Base change log2(e), which turns the reduction into an exponent field."""

comptime EXP_LN2_HI = Scalar[ACCUM](0.693359375)
"""Leading part of ln2. Exactly representable in fp32 (0x3F318000), so
`n * EXP_LN2_HI` is exact and the subtraction below cancels without error."""

comptime EXP_LN2_LO = Scalar[ACCUM](-2.12194440e-4)
"""Tail of ln2, i.e. `ln2 - EXP_LN2_HI`. Carrying it separately is what keeps the reduced
argument accurate at the top of the range, where `n` is ~128."""

comptime EXP_RNE_MAGIC = Scalar[ACCUM](12582912.0)
"""1.5 * 2^23. Adding it forces round-to-nearest-even into the fp32 mantissa
and subtracting it back leaves the rounded integer, for |x| < 2^22."""

comptime EXP_RNE_MAGIC_BITS = Int32(0x4B400000)
"""Bit pattern of `EXP_RNE_MAGIC`. With the magic add the float's low bits ARE
the integer, so the integer `n` is one int subtract away and no fp-to-int
convert (which this target lowers out of line) is needed."""

comptime EXP_MIN_ARG = Scalar[ACCUM](-87.3)
"""Clamp below. `exp(-87.3)` is 1.4e-38, the smallest normal fp32: one ulp
lower and the exponent field would underflow to a denormal or wrap. The online
softmax reaches it through `-inf` -- the initial running max, and the padding
this kernel writes into a short key tile."""


@always_inline
def _exp_hvx[width: Int](x: SIMD[ACCUM, width]) -> SIMD[ACCUM, width]:
    """`exp(x)` for `x <= 0`, in lane-wise fp32 and int32 ops only.

    Domain: the online softmax only ever exponentiates `s - m_new` and
    `m - m_new`, both non-positive by construction because `m_new` is the
    running max. Inputs above ~88 would overflow the exponent field, and this
    deliberately does not clamp them -- a positive argument here would mean the
    recurrence had broken, and hiding that behind a saturate would turn a
    wrong answer into a plausible one.

    Accuracy: the degree-6 Horner is the Taylor series of `e^r` on
    `|r| <= ln2/2`, whose truncation is ~1.2e-7 relative -- three decimal
    orders below fp16's 4.9e-4 resolution and below the fp32 accumulation
    slack the online recurrence already carries.
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


# ===-----------------------------------------------------------------------===#
# Pure flash-attention compute (target-independent; the host test drives this).
# ===-----------------------------------------------------------------------===#


@always_inline
def _dot[
    use_hvx: Bool, W: Int
](a: Row[IO_DTYPE], b: Row[IO_DTYPE]) -> Scalar[ACCUM]:
    """fp32 dot product of two contiguous fp16 rows of equal length.

    The length is the row's own -- `a.dim[0]()`, the `head_dim` extent the
    `slice(s, h, All)` kept -- not a separate argument. A row view that had to be
    told how long it is would be stating the same extent twice, and only one of
    the two statements would be checked against the buffer.

    The QK^T primitive: `W` lanes at a time through the HVX MAC, folded to a
    scalar, with a scalar tail for a length that is not a multiple of `W`
    (FLUX's `head_dim = 128` has none). The tail is a width-1 tile load rather
    than a predicated one, so this kernel needs no `masked_load` escape hatch.

    The vector loads take the default alignment of 1 -- these buffers carry no
    alignment guarantee and the HVX unaligned load `vmemu` is what both the
    pointer original and this compile to. The scalar tail states its element
    alignment: `TileTensor.load`'s off-GPU default is 1
    (`tile_tensor.mojo:799`), and a sub-word alignment on a scalar load makes
    the backend synthesize the word from `memub` bytes rather than issue one.
    """
    comptime ALIGN = align_of[Scalar[IO_DTYPE]]()
    var length = Int(a.dim[0]())
    var acc = SIMD[ACCUM, W](0)
    var kv = 0
    while kv + W <= length:
        acc = _mac[use_hvx](acc, a.load[W](Coord(kv)), b.load[W](Coord(kv)))
        kv += W
    var total = _reduce_add(acc)[0]
    while kv < length:
        total += (
            a.load[1, alignment=ALIGN](Coord(kv)).cast[ACCUM]()[0]
            * b.load[1, alignment=ALIGN](Coord(kv)).cast[ACCUM]()[0]
        )
        kv += 1
    return total


@always_inline
def flash_attention_head[
    use_hvx: Bool, BM: Int, BN: Int, D: Int, W: Int
](
    Q: BSHD[IO_DTYPE],
    K: BSHD[IO_DTYPE],
    V: BSHD[IO_DTYPE],
    O: BSHD[IO_DTYPE],
    head_idx: Int,
    scale: Scalar[ACCUM],
):
    """Online-softmax flash attention for one head, dense (`NULL_MASK`).

    Q/K/V/O are the caller's `[seq_len, num_heads, D]` fp16 activations as a
    rank-3 tile and `head_idx` selects the head; both `seq_len` and `num_heads`
    are the layout's business and neither is an argument. The query rows are
    walked in tiles of `BM` and the keys in tiles of `BN`; the softmax state
    (`m`, `l`, `O`) is per query row but held for a whole tile of them at once,
    so the key loop runs once per `BM` rows and the result is still independent
    of `BM`/`BN` -- the property the tile-invariance test checks. `O` must not
    alias `Q`/`K`/`V`, and all four share the one layout.
    """
    comptime assert (
        D % W == 0
    ), "head_dim must be a multiple of the HVX fp16 width for the vector path"
    comptime DV = D // W
    comptime NS = ceildiv(BN, W)
    comptime SN = NS * W
    comptime ALIGN_ACCUM = align_of[Scalar[ACCUM]]()

    # The per-tile softmax state. `o_acc` is the one local too big for
    # registers (`BM * D` fp32; one row alone is 4 of the 32 HVX registers), so
    # it is a scratch tile rather than a `StaticTuple` of vectors -- allocated
    # once for the whole head, outside every loop, because an `alloca` is a
    # frame slot and not a loop body's business.
    var o_acc = stack_allocation[dtype=ACCUM, alignment=HVX_VECTOR_SIZE](
        row_major[BM, D]()
    )
    # Scores in, probabilities out: Pass A writes `BN` scalars, the softmax
    # reads them as `NS` vectors, and Pass B reads the probabilities back as
    # scalars to broadcast. The round trip is what the axpy shape costs -- P is
    # needed per key as a splat, and only a memory round trip turns a fold's
    # scalar into a lane. It is a `SN`-element frame slot, hot in L1.
    var p_buf = stack_allocation[dtype=ACCUM, alignment=HVX_VECTOR_SIZE](
        row_major[SN]()
    )
    var m: StaticTuple[Scalar[ACCUM], BM] = {fill = min_or_neg_inf[ACCUM]()}
    var l: StaticTuple[Scalar[ACCUM], BM] = {fill = Scalar[ACCUM](0)}

    var seq_len = Int(Q.dim[0]())
    var qi = 0
    while qi < seq_len:
        var qrows = BM
        if qi + BM > seq_len:
            qrows = seq_len - qi

        for a in range(qrows):
            m[a] = min_or_neg_inf[ACCUM]()
            l[a] = Scalar[ACCUM](0)
            var o_row = o_acc.slice(a, All)
            comptime for dv in range(DV):
                o_row.store[W](Coord(dv * W), SIMD[ACCUM, W](0))

        var kj = 0
        while kj < seq_len:
            var krows = BN
            if kj + BN > seq_len:
                krows = seq_len - kj

            # One key tile of lead on the K/V stream, and no more. A K or V row
            # of this head is `D` contiguous fp16 -- 256 B -- every row stride,
            # 6144 B at the render's 24 heads, so the stream is the strided
            # class order 096 leg A measured at 26x the contiguous one, and a
            # width-equals-stride descriptor cannot name it. The tile below is
            # read once per query row before `kj` advances, which is the lead;
            # fetching further ahead or wider per issue is what the
            # `MATMUL_PF_B_ROWS` sweep found gives the gain back. The masked
            # twin is deliberately left un-hinted: it is the text encoder's,
            # which M4.8 measured host-bound.
            comptime if use_hvx and HMX_ACT_PREFETCH:
                var pf_kj = kj + BN
                if pf_kj < seq_len:
                    var pf_rows = BN
                    if pf_kj + BN > seq_len:
                        pf_rows = seq_len - pf_kj
                    _l2fetch_head_rows[D](K, pf_kj, head_idx, pf_rows)
                    _l2fetch_head_rows[D](V, pf_kj, head_idx, pf_rows)

            for a in range(qrows):
                var q_row = Q.slice(qi + a, head_idx, All)

                # Invariant across every key of the tile, so it is loaded once
                # per (query row, key tile) rather than once per key.
                var qv: StaticTuple[SIMD[IO_DTYPE, W], DV] = {
                    fill = SIMD[IO_DTYPE, W](0)
                }
                comptime for dv in range(DV):
                    qv[dv] = q_row.load[W](Coord(dv * W))

                # Pass A: scores for the tile. Unchanged from the per-row
                # spelling -- what moved is which loop it sits inside.
                var b = 0
                while b + ATT_NK <= krows:
                    var dacc: StaticTuple[SIMD[ACCUM, W], ATT_NK] = {
                        fill = SIMD[ACCUM, W](0)
                    }
                    comptime for nk in range(ATT_NK):
                        var k_row = K.slice(kj + b + nk, head_idx, All)
                        comptime for dv in range(DV):
                            dacc[nk] = _mac[use_hvx](
                                dacc[nk], qv[dv], k_row.load[W](Coord(dv * W))
                            )
                    comptime for nk in range(ATT_NK):
                        p_buf.store[1, alignment=ALIGN_ACCUM](
                            Coord(b + nk),
                            SIMD[ACCUM, 1](scale * _reduce_add(dacc[nk])[0]),
                        )
                    b += ATT_NK
                while b < krows:
                    var k_row = K.slice(kj + b, head_idx, All)
                    p_buf.store[1, alignment=ALIGN_ACCUM](
                        Coord(b),
                        SIMD[ACCUM, 1](scale * _dot[use_hvx, W](q_row, k_row)),
                    )
                    b += 1

                # A short tile (the sequence tail, or `BN` below the vector
                # width) leaves lanes the previous row's probabilities wrote.
                # `-inf` is the identity for both consumers below: it cannot win
                # the max, and its exp is 0 so it cannot join the sum.
                if krows < SN:
                    for pad in range(krows, SN):
                        p_buf.store[1, alignment=ALIGN_ACCUM](
                            Coord(pad),
                            SIMD[ACCUM, 1](min_or_neg_inf[ACCUM]()),
                        )

                # The tile's softmax, across keys rather than one at a time.
                # Per key it was three out-of-line library calls (`floorf`,
                # `__truncsfhf2`, `__extendhfsf2`), and they in turn forced
                # `o_acc` to spill and refill on every key.
                var s_vec: StaticTuple[SIMD[ACCUM, W], NS] = {
                    fill = SIMD[ACCUM, W](0)
                }
                var mx = SIMD[ACCUM, W](min_or_neg_inf[ACCUM]())
                comptime for sv in range(NS):
                    s_vec[sv] = p_buf.load[W](Coord(sv * W))
                    mx = max(mx, s_vec[sv])

                var m_new = _reduce_max(mx)[0]
                if m[a] > m_new:
                    m_new = m[a]
                # exp(-inf - finite) = 0, so the first tile rescales 0 by 0.
                var corr = _exp_hvx[1](SIMD[ACCUM, 1](m[a] - m_new))[0]
                var m_bcast = SIMD[ACCUM, W](m_new)
                var sum_v = SIMD[ACCUM, W](0)
                comptime for sv in range(NS):
                    var pv = _exp_hvx(s_vec[sv] - m_bcast)
                    sum_v += pv
                    p_buf.store[W](Coord(sv * W), pv)

                l[a] = l[a] * corr + _reduce_add(sum_v)[0]
                m[a] = m_new

                # Pass B: the P@V accumulate, with this row's O in registers
                # for the whole tile -- one load and one store per (row, tile)
                # where the scalar spelling spilled and refilled per key.
                # Unlike the QK^T dot, this axpy's result is stored as an
                # O-column vector, so it runs in plain fp32 lane order:
                # `_fmac_sf_hf` leaves the lanes in `V6_vmpy_sf_hf_acc`'s
                # interleaved order, which would permute O within each 64-lane
                # block. P stays fp32 -- the fp16 narrowing was there to feed
                # that MAC, which the lane-order fix already removed.
                var o_row = o_acc.slice(a, All)
                var ov: StaticTuple[SIMD[ACCUM, W], DV] = {
                    fill = SIMD[ACCUM, W](0)
                }
                comptime for dv in range(DV):
                    ov[dv] = o_row.load[W](Coord(dv * W)) * SIMD[ACCUM, W](corr)
                for kb in range(krows):
                    var p = SIMD[ACCUM, W](
                        p_buf.load[1, alignment=ALIGN_ACCUM](Coord(kb))[0]
                    )
                    var v_row = V.slice(kj + kb, head_idx, All)
                    comptime for dv in range(DV):
                        var vv = v_row.load[W](Coord(dv * W))
                        ov[dv] = ov[dv] + p * vv.cast[ACCUM]()
                comptime for dv in range(DV):
                    o_row.store[W](Coord(dv * W), ov[dv])

            kj += BN

        for a in range(qrows):
            var inv = Scalar[ACCUM](1) / l[a]
            var o_row = o_acc.slice(a, All)
            var out_row = O.slice(qi + a, head_idx, All)
            comptime for dv in range(DV):
                var ov = o_row.load[W](Coord(dv * W)) * SIMD[ACCUM, W](inv)
                out_row.store[W](Coord(dv * W), ov.cast[IO_DTYPE]())

        qi += BM


@always_inline
def flash_attention_heads[
    use_hvx: Bool, BM: Int, BN: Int, D: Int, W: Int
](
    Q: BSHD[IO_DTYPE],
    K: BSHD[IO_DTYPE],
    V: BSHD[IO_DTYPE],
    O: BSHD[IO_DTYPE],
    head_begin: Int,
    head_end: Int,
    scale: Scalar[ACCUM],
):
    """Runs `flash_attention_head` over the head range `[head_begin, head_end)`.

    The host test drives this directly (all heads, `use_hvx=False`); the device
    entry calls it with the head slice a thread owns and `use_hvx=True`.

    Below the seam, so what arrives is the `[seq_len, num_heads, D]` tile itself
    and not a pointer plus the extents to read it with. `head_begin`/`head_end`
    stay plain integers: they are a work split across the core's threads, not
    geometry, and the range a thread owns is not a property of the tensor. The
    split stays a hand-written blocked one rather than `.distribute()`, which
    partitions *cyclically* (KB
    `kernels/distribute-is-cyclic-this-part-wants-blocked`).
    """
    for h in range(head_begin, head_end):
        flash_attention_head[use_hvx, BM, BN, D, W](Q, K, V, O, h, scale)


@always_inline
def attention_thread_heads(num_heads: Int, thread_id: Int) -> Tuple[Int, Int]:
    """The `[begin, end)` run of heads this launch index owns.

    Each of the four compute threads owns a contiguous run of heads with no
    cross-thread state, so no barrier; the fifth (DMA/HMX) thread and any launch
    index past the head count get the empty run, which the head loop then runs
    zero times. On the dense entry the fifth thread getting no heads is exactly
    what leaves it free to hold the HMX grant.

    `thread_id` is an argument rather than a shim read so the partition is
    testable off-device; only the entries below read the shim. The split stays
    hand-written and blocked -- `.distribute()` partitions cyclically, and a
    cyclic head split would interleave the streams every thread walks.
    """
    if UInt32(thread_id) >= NUM_COMPUTE_THREADS:
        return (num_heads, num_heads)
    var heads_per_thread = ceildiv(num_heads, Int(NUM_COMPUTE_THREADS))
    var head_begin = thread_id * heads_per_thread
    if head_begin >= num_heads:
        return (num_heads, num_heads)
    var head_end = head_begin + heads_per_thread
    if head_end > num_heads:
        head_end = num_heads
    return (head_begin, head_end)


@always_inline
def flash_attention_striped[
    use_hvx: Bool, BM: Int, BN: Int, D: Int, W: Int
](
    Q: BSHD[IO_DTYPE],
    K: BSHD[IO_DTYPE],
    V: BSHD[IO_DTYPE],
    O: BSHD[IO_DTYPE],
    thread_id: Int,
    scale: Scalar[ACCUM],
):
    """Attends the heads `thread_id` owns; the launch-index split, in one place.
    """
    var heads = attention_thread_heads(Int(Q.dim[1]()), thread_id)
    flash_attention_heads[use_hvx, BM, BN, D, W](
        Q, K, V, O, heads[0], heads[1], scale
    )


# ===-----------------------------------------------------------------------===#
# Causal + padding masked variant (Qwen3 text encoder, P4).
# ===-----------------------------------------------------------------------===#
#
# The one genuinely new encoder kernel: the dense flash above with an additive
# score mask. The Qwen3 encoder attention is causal + padding, delivered as an
# explicit additive bias the host precomputes (0 where visible, a large negative
# where masked -- the causal triangle and the padding folded into one `[S, S]`
# tensor, broadcast across heads, exactly as MAX's `masked_flash_attention_gpu`
# consumes it). The only delta from `NULL_MASK` is one add: `bias[query, key]`
# into each score before the online softmax's running max, so a masked key
# contributes `exp(-inf) = 0` and no per-tile causal skip is needed. Everything
# else -- the online-softmax recurrence, the P@V accumulate through the linear
# `.cast[f32]()` widen, the head split -- is identical to `flash_attention_head`.
# GQA (Qwen3 is 32:8) is resolved on the host by materializing K/V to 32 heads,
# so this kernel is dense MHA with no GQA math (`QWEN3_ENCODER.md` §5).
#
# The bias is fp32 so a large-negative sentinel keeps its magnitude (fp16 would
# saturate a real `-inf` and risk `NaN` on a fully-masked row); a finite sentinel
# leaves a fully-masked row a finite average of V rather than a `NaN`.


@always_inline
def flash_attention_head_masked[
    use_hvx: Bool, BM: Int, BN: Int, D: Int, W: Int
](
    Q: BSHD[IO_DTYPE],
    K: BSHD[IO_DTYPE],
    V: BSHD[IO_DTYPE],
    O: BSHD[IO_DTYPE],
    bias: RowMajor2D[ACCUM],
    head_idx: Int,
    scale: Scalar[ACCUM],
):
    """Online-softmax flash attention for one head with an additive score mask.

    Identical to `flash_attention_head` -- same tiling, same per-tile softmax
    state, same vector exp, same fp32 P -- except each score gains
    `bias[query, key]` (fp32 `[seq_len, seq_len]`, broadcast across heads) before
    the running max. `head_idx` selects the head of Q/K/V/O; `bias` has no head
    extent because it is shared by every head, and no `seq_len` argument because
    both tiles already carry it. The result is independent of `BM`/`BN`. `O` must
    not alias `Q`/`K`/`V`/`bias`.

    The two bodies move together on purpose: `test_attention_masked_kernel.mojo`
    gates a zero bias against the dense kernel at *bit equality*, which is what
    makes "the mask path is the dense path plus one add" a checked claim rather
    than a comment. This kernel is the Qwen3 text encoder's, run once per image
    against the dense kernel's hundreds, so it inherits M4.0a/M4.0b for the
    equality rather than for its own cost.
    """
    comptime assert (
        D % W == 0
    ), "head_dim must be a multiple of the HVX fp16 width for the vector path"
    comptime DV = D // W
    comptime NS = ceildiv(BN, W)
    comptime SN = NS * W
    comptime ALIGN_ACCUM = align_of[Scalar[ACCUM]]()

    var o_acc = stack_allocation[dtype=ACCUM, alignment=HVX_VECTOR_SIZE](
        row_major[BM, D]()
    )
    var p_buf = stack_allocation[dtype=ACCUM, alignment=HVX_VECTOR_SIZE](
        row_major[SN]()
    )
    var m: StaticTuple[Scalar[ACCUM], BM] = {fill = min_or_neg_inf[ACCUM]()}
    var l: StaticTuple[Scalar[ACCUM], BM] = {fill = Scalar[ACCUM](0)}

    var seq_len = Int(Q.dim[0]())
    var qi = 0
    while qi < seq_len:
        var qrows = BM
        if qi + BM > seq_len:
            qrows = seq_len - qi

        for a in range(qrows):
            m[a] = min_or_neg_inf[ACCUM]()
            l[a] = Scalar[ACCUM](0)
            var o_row = o_acc.slice(a, All)
            comptime for dv in range(DV):
                o_row.store[W](Coord(dv * W), SIMD[ACCUM, W](0))

        var kj = 0
        while kj < seq_len:
            var krows = BN
            if kj + BN > seq_len:
                krows = seq_len - kj

            for a in range(qrows):
                var q_row = Q.slice(qi + a, head_idx, All)
                var bias_row = bias.slice(qi + a, All)

                # Pass A: masked scores for the tile. The `bias` add is the only
                # delta from the dense kernel; the scalar fp32 load is bounded to
                # the tile's valid keys (`krows`), so a sequence tail never reads
                # past the row. Its alignment is stated because
                # `TileTensor.load` defaults to 1 off-GPU, and a 4-byte value at
                # alignment 1 is assembled from four `memub` bytes -- measured,
                # in this innermost loop.
                for b in range(krows):
                    var k_row = K.slice(kj + b, head_idx, All)
                    var s = (
                        scale * _dot[use_hvx, W](q_row, k_row)
                        + bias_row.load[1, alignment=ALIGN_ACCUM](
                            Coord(kj + b)
                        )[0]
                    )
                    p_buf.store[1, alignment=ALIGN_ACCUM](
                        Coord(b), SIMD[ACCUM, 1](s)
                    )

                if krows < SN:
                    for pad in range(krows, SN):
                        p_buf.store[1, alignment=ALIGN_ACCUM](
                            Coord(pad),
                            SIMD[ACCUM, 1](min_or_neg_inf[ACCUM]()),
                        )

                var s_vec: StaticTuple[SIMD[ACCUM, W], NS] = {
                    fill = SIMD[ACCUM, W](0)
                }
                var mx = SIMD[ACCUM, W](min_or_neg_inf[ACCUM]())
                comptime for sv in range(NS):
                    s_vec[sv] = p_buf.load[W](Coord(sv * W))
                    mx = max(mx, s_vec[sv])

                var m_new = _reduce_max(mx)[0]
                if m[a] > m_new:
                    m_new = m[a]
                # exp(-inf - finite) = 0, so the first tile rescales 0 by 0.
                var corr = _exp_hvx[1](SIMD[ACCUM, 1](m[a] - m_new))[0]
                var m_bcast = SIMD[ACCUM, W](m_new)
                var sum_v = SIMD[ACCUM, W](0)
                comptime for sv in range(NS):
                    var pv = _exp_hvx(s_vec[sv] - m_bcast)
                    sum_v += pv
                    p_buf.store[W](Coord(sv * W), pv)

                l[a] = l[a] * corr + _reduce_add(sum_v)[0]
                m[a] = m_new

                # Pass B: the P@V accumulate, in plain fp32 lane order (the
                # interleaved-lane MAC would permute the stored O -- the fix the
                # dense kernel's device confirm forced).
                var o_row = o_acc.slice(a, All)
                var ov: StaticTuple[SIMD[ACCUM, W], DV] = {
                    fill = SIMD[ACCUM, W](0)
                }
                comptime for dv in range(DV):
                    ov[dv] = o_row.load[W](Coord(dv * W)) * SIMD[ACCUM, W](corr)
                for kb in range(krows):
                    var p = SIMD[ACCUM, W](
                        p_buf.load[1, alignment=ALIGN_ACCUM](Coord(kb))[0]
                    )
                    var v_row = V.slice(kj + kb, head_idx, All)
                    comptime for dv in range(DV):
                        var vv = v_row.load[W](Coord(dv * W))
                        ov[dv] = ov[dv] + p * vv.cast[ACCUM]()
                comptime for dv in range(DV):
                    o_row.store[W](Coord(dv * W), ov[dv])

            kj += BN

        for a in range(qrows):
            var inv = Scalar[ACCUM](1) / l[a]
            var o_row = o_acc.slice(a, All)
            var out_row = O.slice(qi + a, head_idx, All)
            comptime for dv in range(DV):
                var ov = o_row.load[W](Coord(dv * W)) * SIMD[ACCUM, W](inv)
                out_row.store[W](Coord(dv * W), ov.cast[IO_DTYPE]())

        qi += BM


@always_inline
def flash_attention_heads_masked[
    use_hvx: Bool, BM: Int, BN: Int, D: Int, W: Int
](
    Q: BSHD[IO_DTYPE],
    K: BSHD[IO_DTYPE],
    V: BSHD[IO_DTYPE],
    O: BSHD[IO_DTYPE],
    bias: RowMajor2D[ACCUM],
    head_begin: Int,
    head_end: Int,
    scale: Scalar[ACCUM],
):
    """Runs `flash_attention_head_masked` over the head range.

    The host test drives this directly (all heads, `use_hvx=False`); the device
    entry calls it with the head slice a thread owns and `use_hvx=True`. Below
    the seam like the dense one, with the bias arriving as its own
    `[seq_len, seq_len]` tile -- the one place the two layouts differ, and now
    the only place that difference is written down.
    """
    for h in range(head_begin, head_end):
        flash_attention_head_masked[use_hvx, BM, BN, D, W](
            Q, K, V, O, bias, h, scale
        )


@always_inline
def flash_attention_masked_striped[
    use_hvx: Bool, BM: Int, BN: Int, D: Int, W: Int
](
    Q: BSHD[IO_DTYPE],
    K: BSHD[IO_DTYPE],
    V: BSHD[IO_DTYPE],
    O: BSHD[IO_DTYPE],
    bias: RowMajor2D[ACCUM],
    thread_id: Int,
    scale: Scalar[ACCUM],
):
    """Attends the masked heads `thread_id` owns, on the dense split."""
    var heads = attention_thread_heads(Int(Q.dim[1]()), thread_id)
    flash_attention_heads_masked[use_hvx, BM, BN, D, W](
        Q, K, V, O, bias, heads[0], heads[1], scale
    )


@always_inline
def mha_masked_entry(
    O: UnsafePointer[Scalar[IO_DTYPE], MutUnsafeAnyOrigin],
    Q: UnsafePointer[Scalar[IO_DTYPE], MutUnsafeAnyOrigin],
    K: UnsafePointer[Scalar[IO_DTYPE], MutUnsafeAnyOrigin],
    V: UnsafePointer[Scalar[IO_DTYPE], MutUnsafeAnyOrigin],
    bias: UnsafePointer[Scalar[ACCUM], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    seq_len: Int,
    num_heads: Int,
    scale: Scalar[ACCUM],
):
    """Masked attention over the heads this launch index owns.

    This is the seam. The C-ABI signature that reaches it is the frozen
    `NPUKernels.def` contract and is not ours to move, so the extents it carries
    become layouts here -- once -- and no signature under this line takes an
    extent again.

    `witness` keeps its pointer where the activations do not: it is a side
    channel of scalars at fixed slots, not a tensor. It reports what the output
    cannot -- the compiled vector width, the fp32 accumulator lane count, the
    participant count the shim reports, and the RTOS thread each launch index
    ran on -- laid out as the GEMV / softmax / RoPE kernels lay it out.
    """
    var thread_idx = _npu_thread_idx()

    if thread_idx == 0:
        witness.store(0, UInt32(MMA_W))
        witness.store(1, UInt32(SIMD[ACCUM, MMA_W]().size))
        witness.store(2, UInt32(_qshim_query(QSHIM_QUERY_NUM_THREADS)))
    witness.store(3 + thread_idx, _qurt_thread_id())

    var extent = row_major(Coord(seq_len, num_heads, FA_HEAD_DIM))
    flash_attention_masked_striped[
        use_hvx=True, BM=FA_BM, BN=FA_BN, D=FA_HEAD_DIM, W=MMA_W
    ](
        BSHD[IO_DTYPE](ptr=Q, layout=extent),
        BSHD[IO_DTYPE](ptr=K, layout=extent),
        BSHD[IO_DTYPE](ptr=V, layout=extent),
        BSHD[IO_DTYPE](ptr=O, layout=extent),
        RowMajor2D[ACCUM](ptr=bias, layout=row_major(Coord(seq_len, seq_len))),
        thread_idx,
        scale,
    )


# ===-----------------------------------------------------------------------===#
# On-device entry point (`@export ... abi("C")`).
#
# Named `hvx_*` like every other authoring module's, not `mojo_*` like the
# manifest's: `NPUKernels.def:179` binds the `mojo_mha_no_cache_f16` symbol that
# `npu_kernel.mojo` defines, and two definitions of one name cannot both be in
# the `hexagon_ondevice` package. The dispatch path reaches kernels through
# selector strings (`mha_no_cache`), never through a symbol, so nothing follows
# the rename. The rope module's exports were renamed the same way and for the
# same reason (#357).
# ===-----------------------------------------------------------------------===#


@export
def hvx_mha_no_cache_f16(
    O: UnsafePointer[Scalar[IO_DTYPE], MutUnsafeAnyOrigin],
    Q: UnsafePointer[Scalar[IO_DTYPE], MutUnsafeAnyOrigin],
    K: UnsafePointer[Scalar[IO_DTYPE], MutUnsafeAnyOrigin],
    V: UnsafePointer[Scalar[IO_DTYPE], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    seq_len: Int,
    num_heads: Int,
    scale: Float32,
    rt: Int,
) abi("C"):
    """Dense fp16 flash attention over `[seq_len, num_heads, head_dim=128]`.

    The heads divide across the core's compute threads by launch index (Track-1
    parallelism: one head slice per thread, no cross-thread state, so no
    barrier); finer within-head q-tile parallelism is a device-lane perf
    refinement. `witness` reports what the output cannot -- the compiled vector
    width, the fp32 accumulator lane count, the participant count the shim
    reports, and the RTOS thread each launch index ran on -- laid out as the
    GEMV / softmax / RoPE kernels lay it out, so the on-device dispatch check
    proves both vectorization and the by-index thread split the same way. It
    stays a raw pointer: it is a side-channel of scalars, not a tensor.

    `rt` is reserved for the Track-2 HMX crouton row-tile and is unused on the
    HVX path.

    This is the seam, and the last pointer in the file: the frozen C ABI hands
    down raw addresses and extents, they become one layout here, and everything
    below takes the tile.
    """
    # `MMA_W` is 64 fp16 lanes *because* that is one 128-byte HVX register --
    # asserted as the invariant rather than against whatever width the target
    # reports, so the check survives a descriptor change (KB
    # `kernels/tiletensor-vector-width-state-it-explicitly`).
    comptime assert (
        MMA_W * bit_width_of[IO_DTYPE]() == HVX_VECTOR_SIZE * 8
    ), "MMA_W must be exactly one HVX vector register"

    var thread_idx = _npu_thread_idx()

    if thread_idx == 0:
        witness.store(0, UInt32(MMA_W))
        witness.store(1, UInt32(SIMD[ACCUM, MMA_W]().size))
        witness.store(2, UInt32(_qshim_query(QSHIM_QUERY_NUM_THREADS)))
    witness.store(3 + thread_idx, _qurt_thread_id())

    var extent = row_major(Coord(seq_len, num_heads, FA_HEAD_DIM))
    flash_attention_striped[
        use_hvx=True, BM=FA_BM, BN=FA_BN, D=FA_HEAD_DIM, W=MMA_W
    ](
        BSHD[IO_DTYPE](ptr=Q, layout=extent),
        BSHD[IO_DTYPE](ptr=K, layout=extent),
        BSHD[IO_DTYPE](ptr=V, layout=extent),
        BSHD[IO_DTYPE](ptr=O, layout=extent),
        thread_idx,
        scale,
    )

    _ = rt
