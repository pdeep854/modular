"""HVX permute intrinsics, bound as lane semantics rather than as raw builtins.

Mojo binds no HVX permute before this file: a sweep of the device sources
returns exactly one `llvm.hexagon.V6.*` intrinsic, the multiply-accumulate in
`npu_kernel.mojo`'s `_fmac_sf_hf`. That absence is why `hmx_crouton.mojo` packs
croutons a scalar element at a time, and the pack is what order 096 leg B
measured at 65% of a matmul dispatch. This file exists to remove that, and every
instruction in it has a caller — a permute nobody calls is a claim about the
hardware that nothing checks.

The instructions are `V6_vshuffvdd`, `Vdd = vshuff(Vu, Vv, Rt)`, and its inverse
`V6_vdealvdd`. Their lane rule is NOT "interleave", despite that being how both
public references call it; the interleave is one value of `Rt` out of eight.
`Rt` is a BITMASK over byte strides, and the instruction runs one exchange stage
per set bit — ascending for the shuffle, DESCENDING for the deal, which is the
whole of the difference between them:

    Vdd[0:128] = Vv;  Vdd[128:256] = Vu
    for stride in 1, 2, 4, ... 64:
        if Rt & stride:
            for i in 0..127 where (i & stride) == 0:
                swap(Vdd[128 + i], Vdd[i + stride])

That is `shuffles::vshuffvdd` in LLVM's own Hexagon backend
(`llvm/lib/Target/Hexagon/HexagonISelDAGToDAGHVX.cpp:810`, with `vdealvdd` at
`:830` differing only in the loop's direction), which is the model the compiler
uses to SELECT these instructions for an arbitrary `shufflevector` — so it is the
semantics every LLVM-generated Hexagon shuffle already depends on, not a reading
of the manual. `_vshuffvdd_model` and `_vdealvdd_model` below are those loops
transcribed, and they are what the arm64 twin executes, exactly as
`_hmx_mma_model` stands in for the MAC. What neither can claim is silicon: the
model is a compiler's, and the device A/B is what closes it.

Because each exchange stage is its own inverse and the two orders are reverses of
each other, `vdeal(vshuff(a, b))` is the IDENTITY at every `Rt` — which is a
property the twin asserts rather than a remark, and the only cheap check that the
descending order was transcribed the right way round.

Two `Rt` values are used, and each is corroborated by a working reference:

- `Rt = -2` (every stride set) composes to the perfect halfword interleave:
  `lo[2i] = Vv[i]`, `lo[2i+1] = Vu[i]` over halfwords. This is the activation
  crouton rule in one instruction — `ggml-hexagon`'s `hmx_interleave_cols_to_tiles`
  (`htp/hmx-utils.h:180`) and this tree's own
  `_rm_i8_to_crouton_mxblock` (`qaic/attention/_mha_per_tile_split_barrier.mojo:439`)
  both build a crouton with this exact call.
- `Rt = 4 * stride`, one bit, is a single exchange stage — the butterfly a
  vector transpose is built from. The vendor's `transposeHMX`
  (`_mha_per_tile_split_barrier.mojo:504-515`) is three of these; the crouton
  weight rule needs the full five.
- The deal at `Rt = -2` splits a crouton vector pair back into its two source
  rows, which is the S/P softmax seam: `ggml-hexagon` reads scores out of the
  array with `Q6_W_vdeal_VVR(s1, s0, -2)` (`htp/flash-attn-ops.c:1125`) and
  writes probabilities back with the matching `Q6_W_vshuff_VVR` (`:1331`).

A third instruction is bound below, and it is not a permute:
`V6_vS32b_qpred_ai`, the byte-masked vector store. It is here because it is the
deal's CONSUMER and shares its problem — a crouton readout row is 64 bytes,
which is half a vector, and half a vector is the one thing an HVX store cannot
write without being told which half. See `hvx_store_masked`.

A fourth binding is the VTCM word scatter, `V6_vscattermw`, with its
`scatter_release` ordering store: 32 word stores at register-held offsets in
one issue, which is the reference backend's whole K-staging transpose
(`ggml-hexagon` `htp/hmx-utils.h`, `hmx_interleave_rows_to_tiles`) and the
form this port once REJECTED for the offline repack — the destination must be
VTCM, which `GENERIC` pre-packing cannot promise. The in-kernel attention K
pack's destination IS the VTCM arena, which is what re-opens it there
(`hmx_pack_panel_vsc`). Unlike the permutes, no compiler ever selects this
instruction from portable IR, so the model below is a transcription of the
manual rather than of LLVM's own selector — the same standing as the masked
store's, and the reason its device A/B opens with a bit-identity leg. See
`hvx_vscatter_words`.

Target: Hexagon v81, HVX in 128-byte mode. `--target-features=+hvx-length128b`
is what makes `V6_*_128B` the right suffix; the 64-byte forms are a different
instruction and this file does not bind them.
"""
from std.math import iota
from std.memory import AddressSpace, bitcast
from std.sys import inlined_assembly, size_of
from std.sys.info import is_triple
from std.sys.intrinsics import llvm_intrinsic

comptime HVX_BYTES = 128
"""Bytes in one HVX vector, in the 128-byte mode this target builds for."""

comptime HVX_HALF_BYTES = HVX_BYTES // 2
"""Bytes in half an HVX vector, which is one crouton readout row (32 fp16).

Named here, next to the mask that divides a vector by it, because the crouton
side and the store side must not drift about what a half is."""

comptime HVX_WORDS = HVX_BYTES // 4
"""Word lanes in one HVX vector (32): the scatter's offset and value count.

The word scatter reads its offsets and data as 32-bit lanes whatever the
element type of the region being written, so the count is a property of the
vector, named beside `HVX_BYTES` rather than derived at every call site."""

comptime HVX_SHUFF_INTERLEAVE = -2
"""The `Rt` that makes `hvx_vshuff` a perfect halfword interleave.

Every stride bit is set, and composing all six exchange stages is a rotation of
the element index — which is what an interleave is. Written as a named constant
because `-2` at a call site reads like an offset."""

comptime HVX_DEAL_SEPARATE = -2
"""The `Rt` that makes `hvx_vdeal` a perfect halfword DE-interleave.

Numerically the same mask as `HVX_SHUFF_INTERLEAVE` and deliberately a separate
name: the two instructions run the same stages in opposite order, so what the
mask means depends on which one reads it, and a call site that says "interleave"
while calling the deal is the kind of comment that survives being wrong."""


comptime _IS_HEXAGON = is_triple["hexagon-unknown-unknown-elf"]()
"""Whether this compile targets the device.

The intrinsic has no host encoding, and this file is compiled for the arm64 twin
from the same source. Only the ISSUE is gated; the lane semantics above it are
the twin's to execute, which is what lets a byte-identity gate run off the
device at all."""


@always_inline
def hvx_lanes[dtype: DType]() -> Int:
    """Elements of `dtype` in one HVX vector: 64 for fp16, 128 for a byte type.
    """
    return HVX_BYTES // size_of[Scalar[dtype]]()


@always_inline
def _vshuffvdd_model[
    dtype: DType, lanes: Int, //, rt: Int
](vu: SIMD[dtype, lanes], vv: SIMD[dtype, lanes]) -> SIMD[dtype, 2 * lanes]:
    """`V6_vshuffvdd`'s permutation in portable arithmetic, byte for byte.

    LLVM's `shuffles::vshuffvdd` (`HexagonISelDAGToDAGHVX.cpp:810`) transcribed,
    including its ascending stride order — the stages do not commute, so the
    order is part of the semantics rather than an implementation detail.

    Element type is irrelevant to a byte permute, so the body works on bytes and
    the caller's `dtype` only decides how the result is read back.
    """
    var d = bitcast[DType.uint8, HVX_BYTES](vv).join(
        bitcast[DType.uint8, HVX_BYTES](vu)
    )
    comptime for shift in range(7):
        comptime stride = 1 << shift
        comptime if (rt & stride) != 0:
            for i in range(HVX_BYTES):
                if (i & stride) == 0:
                    var hold = d[HVX_BYTES + i]
                    d[HVX_BYTES + i] = d[i + stride]
                    d[i + stride] = hold
    return bitcast[dtype, 2 * lanes](d)


@always_inline
def _vdealvdd_model[
    dtype: DType, lanes: Int, //, rt: Int
](vu: SIMD[dtype, lanes], vv: SIMD[dtype, lanes]) -> SIMD[dtype, 2 * lanes]:
    """`V6_vdealvdd`'s permutation in portable arithmetic, byte for byte.

    LLVM's `shuffles::vdealvdd` (`HexagonISelDAGToDAGHVX.cpp:830`): the same
    exchange network `_vshuffvdd_model` runs, walked from the widest stride down
    instead of up. That single difference is what makes it the shuffle's inverse,
    so the descending range below is the semantics and not a transcription
    convenience.
    """
    var d = bitcast[DType.uint8, HVX_BYTES](vv).join(
        bitcast[DType.uint8, HVX_BYTES](vu)
    )
    comptime for shift in range(6, -1, -1):
        comptime stride = 1 << shift
        comptime if (rt & stride) != 0:
            for i in range(HVX_BYTES):
                if (i & stride) == 0:
                    var hold = d[HVX_BYTES + i]
                    d[HVX_BYTES + i] = d[i + stride]
                    d[i + stride] = hold
    return bitcast[dtype, 2 * lanes](d)


@always_inline
def hvx_vdeal[
    dtype: DType, lanes: Int, //, rt: Int
](vu: SIMD[dtype, lanes], vv: SIMD[dtype, lanes]) -> SIMD[dtype, 2 * lanes]:
    """`Vdd = vdeal(Vu, Vv, Rt)`: `hvx_vshuff`'s exact inverse at the same `Rt`.

    Same pair convention as `hvx_vshuff` — low vector first — so
    `hvx_vdeal[rt](pair.slice[lanes, offset=lanes](), pair.slice[lanes]())`
    returns `(vv, vu)` for any pair the shuffle produced.

    At `HVX_DEAL_SEPARATE` this is the softmax seam's read: a crouton vector pair
    holds two source rows interleaved halfword-wise, and the deal returns them as
    one whole row each, which is the layout the row-wise max and `exp` want.

    Parameters:
        dtype: Element type of the operands. A byte permute does not read it;
            it decides only how the result is interpreted.
        lanes: Elements per vector. Must fill one HVX vector exactly.
        rt: The stride bitmask. `HVX_DEAL_SEPARATE` for the halfword
            de-interleave.

    Args:
        vu: The operand supplying the pair's HIGH vector.
        vv: The operand supplying the pair's LOW vector.

    Returns:
        The vector pair, low vector in the first `lanes` elements.
    """
    comptime assert (
        lanes * size_of[Scalar[dtype]]() == HVX_BYTES
    ), "an HVX operand is exactly one 128-byte vector"

    comptime if _IS_HEXAGON:
        comptime WORDS = HVX_BYTES // size_of[Scalar[DType.int32]]()
        return bitcast[dtype, 2 * lanes](
            llvm_intrinsic[
                "llvm.hexagon.V6.vdealvdd.128B",
                SIMD[DType.int32, 2 * WORDS],
                has_side_effect=False,
            ](
                bitcast[DType.int32, WORDS](vu),
                bitcast[DType.int32, WORDS](vv),
                Int32(rt),
            )
        )
    else:
        return _vdealvdd_model[rt](vu, vv)


@always_inline
def hvx_run_mask(off: Int, run: Int) -> SIMD[DType.bool, HVX_BYTES]:
    """The `hvx_store_masked` mask covering `run` bytes at window offset `off`.

    A vector predicate is per BYTE, so a contiguous run inside the window is a
    comparison on the byte index and nothing more. Unsigned wraparound is what
    makes one comparison serve every offset: a lane below `off` wraps to the top
    of the byte range and fails the bound, so no second compare and no branch —
    and the branch is the point, because `off` comes from an address bit that
    varies per destination row and a branch on it inside a readout loop costs
    the loop its zero-overhead hardware form, the trade P5 M4.6-L2 measured at
    53 packets against 115.

    `off + run` must not exceed `HVX_BYTES`: a run that would straddle two
    windows needs two stores, and it is the caller's alignment gate — not this
    mask — that rules the straddle out (`hmx_unpack_vec_admits`).

    Args:
        off: Byte offset of the run within the 128-byte window.
        run: Bytes in the run.

    Returns:
        The byte mask, `HVX_BYTES` lanes wide.
    """
    return (iota[DType.uint8, HVX_BYTES]() - UInt8(off)).lt(UInt8(run))


@always_inline
def _store_masked_model[
    dtype: DType,
    lanes: Int,
    ptr_dtype: DType,
    origin: MutOrigin,
    address_space: AddressSpace,
    //,
](
    mask: SIMD[DType.bool, HVX_BYTES],
    dst: MutUnsafePointer[
        Scalar[ptr_dtype], origin, address_space=address_space
    ],
    v: SIMD[dtype, lanes],
):
    """`V6_vS32b_qpred_ai`'s store in portable code, truncation and all.

    The window the instruction writes is `dst` with its low seven address bits
    CLEARED, and this is the one place that arithmetic is spelled: a masked
    store is the only operation in this file whose semantics are an address
    rather than a lane order, and modelling it any other way — as a 64-byte copy
    to `dst` — would leave the twin unable to see a wrong mask, which is the
    mistake the arm is most likely to make.
    """
    var window = dst.bitcast[Byte]() - (Int(dst) & (HVX_BYTES - 1))
    var bytes = bitcast[DType.uint8, HVX_BYTES](v)
    for i in range(HVX_BYTES):
        if mask[i]:
            window[i] = bytes[i]


@always_inline
def hvx_store_masked[
    dtype: DType,
    lanes: Int,
    ptr_dtype: DType,
    origin: MutOrigin,
    address_space: AddressSpace,
    //,
](
    mask: SIMD[DType.bool, HVX_BYTES],
    dst: MutUnsafePointer[
        Scalar[ptr_dtype], origin, address_space=address_space
    ],
    v: SIMD[dtype, lanes],
):
    """`if (Qv) vmem(Rt+#0) = Vs`: store the bytes of `v` that `mask` enables.

    ADDRESS TRUNCATION IS THE SEMANTICS rather than a caller's obligation: the
    instruction writes the 128-byte window `dst` falls in, and `mask` chooses
    which of those bytes it writes. That is what lets a 64-byte crouton row be
    stored on its own, and the two alternatives are both worse. A
    read-modify-write of the surrounding vector doubles the traffic of a phase
    that is ~100% byte movement already (`FEED_EVIDENCE_STUDY.md` §3). A plain
    unaligned store of a whole vector writes 64 bytes of the NEXT column tile,
    which is a clobber that happens to be repaired at the next dispatch step
    everywhere except the panel's last tile — correct by an argument about time,
    which is not a property to build a readout on.

    Mojo lowers a sub-vector store to this instruction unaided, so the binding
    earns its lines only through the geometry the caller can prove and the
    compiler cannot: through a dynamic row stride LLVM cannot see the run's
    64-byte alignment, so it emits two masked stores and two `vlalign` rotations
    per run — 5 packets where the caller's own alignment gate admits 1.

    Parameters:
        dtype: Element type of `v`. A byte-masked store does not read it.
        lanes: Elements in `v`. Must fill one HVX vector exactly.
        ptr_dtype: Element type `dst` addresses, which the instruction reads
            even less than `dtype`: the two are INDEPENDENT because a window of
            bytes has no element type, and tying them would force a crouton
            readout to bitcast a destination it addresses through a layout.
        origin: Origin of the destination pointer.
        address_space: Residency of the destination.

    Args:
        mask: Bytes of the window to write, one lane per byte.
        dst: An address in the window to write. Its low seven bits are IGNORED
            by the instruction, so the caller passes the run's own address and
            says which half of the window that is through `mask`.
        v: The vector whose enabled bytes are stored, at their own byte
            positions within the window.
    """
    comptime assert (
        lanes * size_of[Scalar[dtype]]() == HVX_BYTES
    ), "an HVX store operand is exactly one 128-byte vector"

    comptime if _IS_HEXAGON:
        comptime WORDS = HVX_BYTES // size_of[Scalar[DType.int32]]()
        llvm_intrinsic[
            "llvm.hexagon.V6.vS32b.qpred.ai.128B",
            NoneType,
            has_side_effect=True,
        ](mask, dst, bitcast[DType.int32, WORDS](v))
    else:
        _store_masked_model(mask, dst, v)


@always_inline
def hvx_vshuff[
    dtype: DType, lanes: Int, //, rt: Int
](vu: SIMD[dtype, lanes], vv: SIMD[dtype, lanes]) -> SIMD[dtype, 2 * lanes]:
    """`Vdd = vshuff(Vu, Vv, Rt)`: one exchange stage per set stride bit of `rt`.

    Returns the 256-byte vector PAIR the instruction writes, low vector first,
    which is how the pair is addressed at a call site: `slice[lanes, offset=0]`
    is `Vdd.v[0]` and `slice[lanes, offset=lanes]` is `Vdd.v[1]`. Returning the
    pair rather than a tuple is the shape the intrinsic itself has.

    `rt` is a parameter, not an argument: every call site knows it, and a
    constant is what lets the twin fold the permutation and the device fold the
    register load.

    Parameters:
        dtype: Element type of the operands. A byte permute does not read it;
            it decides only how the result is interpreted.
        lanes: Elements per vector. Must fill one HVX vector exactly.
        rt: The stride bitmask. `HVX_SHUFF_INTERLEAVE` for the halfword
            interleave; `4 * s` for the single word-granularity exchange stage a
            transpose butterfly issues at stride `s`.

    Args:
        vu: The operand that lands in the ODD positions of an interleave, and in
            `Vdd.v[1]`'s half of an exchange.
        vv: The operand that lands in the EVEN positions.

    Returns:
        The vector pair, low vector in the first `lanes` elements.
    """
    comptime assert (
        lanes * size_of[Scalar[dtype]]() == HVX_BYTES
    ), "an HVX operand is exactly one 128-byte vector"

    comptime if _IS_HEXAGON:
        comptime WORDS = HVX_BYTES // size_of[Scalar[DType.int32]]()
        return bitcast[dtype, 2 * lanes](
            llvm_intrinsic[
                "llvm.hexagon.V6.vshuffvdd.128B",
                SIMD[DType.int32, 2 * WORDS],
                has_side_effect=False,
            ](
                bitcast[DType.int32, WORDS](vu),
                bitcast[DType.int32, WORDS](vv),
                Int32(rt),
            )
        )
    else:
        return _vshuffvdd_model[rt](vu, vv)


@always_inline
def _vscatter_words_model[
    ptr_dtype: DType,
    origin: MutOrigin,
    address_space: AddressSpace,
    //,
    region: Int,
](
    base: MutUnsafePointer[
        Scalar[ptr_dtype], origin, address_space=address_space
    ],
    offsets: SIMD[DType.int32, HVX_WORDS],
    v: SIMD[DType.int32, HVX_WORDS],
):
    """`V6_vscattermw`'s stores in portable code, mask and truncation included.

    The manual's address rule transcribed: each word lane stores at
    `base + (offset & region)`, with the effective address truncated to a word
    boundary. Both clauses are the semantics rather than caller obligations,
    and both are modelled for the reason `_store_masked_model` spells the
    window truncation: an offset past the region WRAPS instead of landing
    outside it, and a model that added offsets unmasked could not see a wrong
    region argument — which is one of the two mistakes a scatter caller can
    make (the twin's region falsifier fires on exactly this line).

    What no model can claim is silicon's reading of the same sentence: unlike
    the permutes, LLVM never selects this instruction from portable IR, so
    there is no compiler-tested reference loop to transcribe. That residue is
    the pre-registered bit-identity leg of the device A/B that flips
    `HMX_KGATHER_VSC` on.
    """
    var window = base.bitcast[Byte]()
    for i in range(HVX_WORDS):
        var at = (Int(offsets[i]) & region) & ~3
        (window + at).bitcast[Scalar[DType.int32]]()[0] = v[i]


@always_inline
def hvx_vscatter_words[
    ptr_dtype: DType,
    origin: MutOrigin,
    address_space: AddressSpace,
    //,
    region: Int,
](
    base: MutUnsafePointer[
        Scalar[ptr_dtype], origin, address_space=address_space
    ],
    offsets: SIMD[DType.int32, HVX_WORDS],
    v: SIMD[DType.int32, HVX_WORDS],
):
    """`vscatter(Rt, Mu, Vv.w) = Vw`: 32 word stores at register-held offsets.

    The destination REGION is a hardware contract, not a hint: the instruction
    writes VTCM only, which on this port means an `AddressSpace.SHARED`
    panel — a device compile against any other space is refused below rather
    than left to fault on the part. The host model serves every space, which
    is what lets the twin run the same packer against ordinary allocations.

    Scatters are POSTED: the part does not order them against later reads by
    other engines or threads, so a packer that issues any must close with
    `hvx_scatter_release` before its destination is read across the barrier —
    the pairing the reference's K path carries implicitly in its phase
    structure and this port states explicitly.

    Parameters:
        ptr_dtype: Element type `base` addresses. The instruction reads it
            not at all — a scatter writes word lanes into a byte region — so
            it is inferred rather than tied to the data's type.
        origin: Origin of the destination pointer.
        address_space: Residency of the destination. Must be
            `AddressSpace.SHARED` on a device compile.
        region: The scatter region as a size-minus-one mask (`Mu`). A
            power-of-two extent, comptime because every call site knows its
            geometry and a constant is what lets the device fold the modifier
            register load.

    Args:
        base: The region's base address; on device it must sit in VTCM and be
            word-aligned.
        offsets: Byte offsets of the 32 word stores, each `& region` per the
            manual.
        v: The 32 words to store.
    """
    comptime assert region > 0 and (region & (region + 1)) == 0, (
        "the scatter region is a size-minus-one mask, so it must be a"
        " power of two minus one"
    )
    comptime if _IS_HEXAGON:
        comptime assert address_space == AddressSpace.SHARED, (
            "vscatter writes VTCM only; a device-side destination must be an"
            " AddressSpace.SHARED panel (KB"
            " kernels/tiletensor-on-vtcm-is-addressspace-shared)"
        )
        llvm_intrinsic[
            "llvm.hexagon.V6.vscattermw.128B", NoneType, has_side_effect=True
        ](Int32(Int(base)), Int32(region), offsets, v)
    else:
        _vscatter_words_model[region](base, offsets, v)


@always_inline
def hvx_scatter_release[
    ptr_dtype: DType,
    origin: MutOrigin,
    address_space: AddressSpace,
    //,
](
    addr: MutUnsafePointer[
        Scalar[ptr_dtype], origin, address_space=address_space
    ],
):
    """`vmem(Rt+#0):scatter_release`: drains prior scatters before later reads.

    Scatter stores retire before their data is observable — the scatter unit
    completes them out of order behind the VTCM port — and nothing else in the
    memory model closes that window: a thread can arrive at a barrier with its
    scatters still in flight, and the array's `mxmem` read on the other side
    is a different engine with no ordering of its own. This release is the
    ISA's one closing token, issued once per packed panel rather than per
    scatter because ordering is owed at the CONSUMER boundary, not between
    stores that never alias.

    On the host there is nothing to order — the model's stores are ordinary
    program-order writes — so the twin arm is empty rather than pretending a
    fence has semantics there.

    Args:
        addr: An address inside the scattered VTCM region.
    """
    comptime if _IS_HEXAGON:
        comptime assert (
            address_space == AddressSpace.SHARED
        ), "the release orders VTCM scatters, so it takes a VTCM address"
        inlined_assembly[
            "vmem($0+#0):scatter_release",
            NoneType,
            constraints="r,~{memory}",
            has_side_effect=True,
        ](UInt32(Int(addr)))
