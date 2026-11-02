"""Crouton tile packing for the HMX matrix array, as a layout rather than a rule.

HMX multiplies 32x32 fp16 tiles held in a permuted byte order — a *crouton*. The
permutation is one rule with THREE axis assignments (KB
`kernels/hmx-crouton-tile-layout`), and assuming one layout serves every operand
yields silently wrong numbers rather than a crash:

| Operand    | Slab `s`      | Lane `l`  | Source                     |
|------------|---------------|-----------|----------------------------|
| Weight     | `k` (depth)   | `n`       | `B[N, K]`, a transposing gather |
| Activation | `m` (out row) | `k`       | `A[M, K]`, not transposing |
| Output     | `m` (out row) | `n`       | `C[M, N]`, as an activation |

The byte rule is `offset(s, l) = (s / 2) * 128 + l * 4 + (s % 2) * 2`, which over
fp16 elements is `(s / 2) * 64 + l * 2 + (s % 2)`. Written that way it looks like
address arithmetic; it is not. Split the slab into `(s / 2, s % 2)` and the rule
is a plain strided layout of shape `(16, 32, 2)` and stride `(64, 2, 1)` — so a
crouton is expressible as a `TileTensor` whose *type* carries the permutation,
and packing becomes a copy between two coordinate spaces with no offset
expression anywhere. That is the whole reason this file has no multiplications by
a stride in it: the geometry is static and belongs to the layout, and only the
K-depth is a runtime value.

The split is not an fp16 constant. What the tile actually holds is a 4-BYTE LANE
WORD carrying `HMX_SPLIT` consecutive slab elements, so the same layout
expression describes the int8 tile the v75 replica records —
`(k / 4) * 128 + n * 4 + (k % 4)`, shape `(8, 32, 4)` stride `(128, 4, 1)` — with
nothing changed but `HMX_DTYPE`. That generality is deliberate and cheap; see
"int8" under `hmx_requant_word` for what an int8 path would still need beyond the
geometry, and note that it is not implemented here.

The three axis assignments collapse to ONE parameter here, `slab_is_depth`, and
nothing else differs between the weight and activation packers. That is the
KB's "activation and weight are transposes of each other in axis role" stated as
a comptime `Bool` instead of as two hand-written gathers that can drift apart.
The failing control in the twin is what keeps the parameter meaningful.

This file also issues the multiply, which it could not when it was written: the
hexagonv81 backend modelled no HMX at all, so both the intrinsic route and the
inline-assembly route were closed (changelog #232) and the MAC lived in a
`hexagon-clang -mhmx` shim. The `+hmx` MC-layer support this branch stacks on
opens the assembly route, and `hmx_mma_panel` takes it. Nothing about the split
of responsibilities changes — the closed vendor archive's only real content is
the packing convention (KB `kernels/hmx-v81-public-intrinsics-vs-archive`), so
owning the packer in Mojo is what made the kernel ours, and owning the five
instruction words is what removes the last C from the path.

Target: Hexagon v81 (HMX 32x32 fp16). No other revision carries the unit, and
there is no bf16 HMX datapath at any revision.
"""
from layout.coord import Coord, Idx
from layout.tile_layout import Layout, row_major
from layout.tile_tensor import TileTensor

from std.ffi import external_call
from std.math import ceildiv, iota
from std.memory import AddressSpace, bitcast
from std.sys import (
    align_of,
    get_defined_bool,
    get_defined_int,
    inlined_assembly,
    size_of,
)
from std.sys.info import is_triple
from std.sys.intrinsics import llvm_intrinsic

from .hvx_permute import (
    HVX_DEAL_SEPARATE,
    HVX_BYTES,
    HVX_HALF_BYTES,
    HVX_SHUFF_INTERLEAVE,
    HVX_WORDS,
    hvx_lanes,
    hvx_run_mask,
    hvx_scatter_release,
    hvx_store_masked,
    hvx_vdeal,
    hvx_vscatter_words,
    hvx_vshuff,
)

comptime HMX_DTYPE = DType.float16
"""The HMX operand type this path uses.

`.hf` is one of exactly two accumulator modes the unit has (the other is
fixed-point), and there is no `.bf` and no fp32 accumulate at any revision, so a
bf16 model reaches this datapath by converting, not by a wider mode."""

comptime HMX_TILE = 32
"""Elements on a side of one crouton tile: the array is 32x32."""

comptime HMX_TILE_ELEMS = HMX_TILE * HMX_TILE
"""Elements in one crouton tile (1024)."""

comptime HMX_TILE_BYTES = HMX_TILE_ELEMS * size_of[HMX_DTYPE]()
"""Bytes in one crouton tile (2048), and also its required alignment."""

comptime HMX_LANE_BYTES = 4
"""Bytes of one lane word inside a slab pair: the tile's real quantum.

A crouton is 32 lanes of 4 bytes per 128-byte slab pair, whatever the element
type. It is the element width that decides how many depth elements share a lane
word, which is why `HMX_SPLIT` is derived rather than written."""

comptime HMX_SPLIT = HMX_LANE_BYTES // size_of[HMX_DTYPE]()
"""Slab elements packed into one lane word: 2 for fp16, 4 for int8."""


comptime HmxSplit[dtype: DType] = HMX_LANE_BYTES // size_of[dtype]()
"""`HMX_SPLIT` for an arbitrary operand type: 2 for fp16, 4 for a byte type.

The lane word is 4 bytes whatever the element is, so this is the whole of what
an operand type changes about the tile GEOMETRY (`HMX_INT8_SCOPE.md` §2 H4). The
fp16 constant above stays because most of this file is the fp16 arm and reads
better for it; the two cannot drift because one is the other at `HMX_DTYPE`."""

comptime HmxTileBytes[dtype: DType] = HMX_TILE_ELEMS * size_of[dtype]()
"""Bytes in one crouton of `dtype`: 2048 for fp16, 1024 for a byte type.

Also its required alignment, and also the `:deep` load's unit — which is why an
int8 panel is not merely half the elements of an fp16 one but a differently
sized region, and why the arena carve below is parameterised rather than scaled
by a caller."""

comptime HMX_MAX_DEEP_TILES = 32
"""K-tiles one `:deep` load can stream.

The load takes an INCLUSIVE byte extent, `HMX_TILE_BYTES * tiles - 1`, in a
16-bit register. 32 tiles is exactly 65535, so it is a hard cap rather than a
tuning choice, and K past 1024 is several loads into one uncleared
accumulator."""


comptime HmxRows2D = TileTensor[
    HMX_DTYPE, type_of(row_major(Coord(0, 0))), origin=MutUnsafeAnyOrigin
]
"""A row-major `(rows, cols)` fp16 source or destination.

Structurally identical to `matmul_kernel.MMTile`, deliberately: the packers take
the caller's own operand views unchanged, so nothing between the matmul seam and
here restates an extent. Declared locally because the dependency runs
matmul -> crouton and must not run back."""


comptime CroutonPanelOf[dtype: DType, space: AddressSpace] = TileTensor[
    dtype,
    type_of(
        Layout(
            Coord(
                0,
                Idx[HMX_TILE // HmxSplit[dtype]],
                Idx[HMX_TILE],
                Idx[HmxSplit[dtype]],
            ),
            Coord(
                Idx[HMX_TILE_ELEMS],
                Idx[HMX_TILE * HmxSplit[dtype]],
                Idx[HmxSplit[dtype]],
                Idx[1],
            ),
        )
    ),
    origin=MutUntrackedOrigin,
    address_space=space,
]
"""`CroutonPanel` for an arbitrary operand type. See it for the whole note.

The int8 arm's operands are BYTE panels — uint8 activations and int8 weights are
the same bytes at the same offsets and differ only in which instruction reads
them (`activation.ub` vs `weight.b`), so `DType.uint8` describes both and
signedness lives in the packer and the model, never in the layout."""


comptime CroutonPanel[space: AddressSpace] = CroutonPanelOf[HMX_DTYPE, space]
"""`tiles` croutons, addressed as `(tile, slab / SPLIT, lane, slab % SPLIT)`.

The permutation IS this type. Every extent but the tile count is static, so the
compiler folds the whole index; the tile count is dynamic because K is. The
tiles are contiguous because that is what a `:deep` load streams — one register
range covering `tiles` of them.

`space` selects residency: `AddressSpace.GENERIC` for a host-side pre-pack of
weights, `AddressSpace.SHARED` for the VTCM staging a launch uses (KB
`kernels/tiletensor-on-vtcm-is-addressspace-shared`; there is no
`AddressSpace.VTCM`)."""


@always_inline
def hmx_panel[
    space: AddressSpace, dtype: DType = HMX_DTYPE
](
    base: UnsafePointer[Scalar[dtype], MutUntrackedOrigin, address_space=space],
    tiles: Int,
) -> CroutonPanelOf[dtype, space]:
    """Views `tiles * HMX_TILE_ELEMS` contiguous elements as a crouton panel.

    The single place the permutation is spelled. Every other line in this file
    and its callers names a coordinate; if the geometry were ever wrong, it would
    be wrong here and the twin's byte-order check would say so.

    `base` must be `HmxTileBytes[dtype]`-aligned -- the `:deep` load's
    requirement, not this view's.

    Parameters:
        space: Where the region lives.
        dtype: The operand type. `HMX_DTYPE` is the fp16 arm and every existing
            caller; `DType.uint8` is the W8A8 arm's byte panel.
    """
    return CroutonPanelOf[dtype, space](
        ptr=base,
        layout=Layout(
            Coord(
                tiles,
                Idx[HMX_TILE // HmxSplit[dtype]],
                Idx[HMX_TILE],
                Idx[HmxSplit[dtype]],
            ),
            Coord(
                Idx[HMX_TILE_ELEMS],
                Idx[HMX_TILE * HmxSplit[dtype]],
                Idx[HmxSplit[dtype]],
                Idx[1],
            ),
        ),
    )


@always_inline
def hmx_k_tiles(K: Int) -> Int:
    """Crouton tiles spanning `K`. `K` must be a multiple of `HMX_TILE`."""
    return K // HMX_TILE


@always_inline
def hmx_deep_range[dtype: DType = HMX_DTYPE](tiles: Int) -> Int:
    """The `:deep` load's inclusive byte extent for `tiles` croutons.

    A property of the tile geometry, never a tunable: passing 0 streams one byte
    into the array and reads back a correctly-shaped page of zeros.

    An int8 crouton is half the bytes, so the 16-bit field reaches TWICE the
    depth: `HMX_MAX_DEEP_TILES` is an fp16 constant and the byte arm's cap is
    `hmx_max_deep_tiles[dtype]`.
    """
    return HmxTileBytes[dtype] * tiles - 1


comptime HmxMaxDeepTiles[dtype: DType] = 65536 // HmxTileBytes[dtype]
"""K-tiles one `:deep` load of `dtype` croutons can stream.

`HMX_MAX_DEEP_TILES` at fp16 (32), and 64 for a byte type — the same 16-bit
inclusive extent over a crouton that is half the size. Derived rather than
written for the reason `HmxSplit` is."""


comptime HMX_PACK_VEC = get_defined_bool["SKYE_HMX_PACK_VEC", False]()
"""Whether `hmx_pack_panel` gathers with HVX permutes instead of element loads.

OFF by default, and off compiles the vector packer away entirely, so the shipped
object is the scalar build — the A/B control, for the reason `SKYE_HMX_EMIT_MMA`
and `SKYE_HMX_ACT_PREFETCH` are. A `-D SKYE_HMX_PACK_VEC=true` build define
(`build.sh HMX_PACK_VEC=1`) selects the other arm from the same reviewed source.

The default is OFF for a correctness reason, not a performance one. The two arms
are byte-identical on the twin, but the twin executes LLVM's MODEL of
`V6_vshuffvdd` (`hvx_permute.mojo`), and no silicon has yet agreed with it. A
crouton packed by a permute that behaves differently on the part is a wrong
ANSWER with nothing to signal it — the same failure mode the axis rule has. The
device A/B that flips this default therefore starts as a bit-identity check of
the matmul family's output and only then becomes a timing measurement.

Order 112 leg 1 then order 149 arm E gave silicon's vote for the ACTIVATION
rule: 6/6 byte-identical, `PH_PACK_ACT` 29.56-29.91x at production shapes. The
WEIGHT rule has never executed on silicon, and instantiating it is what the
order-149 refusal was — see `HMX_PACK_VEC_WGT` below, which now gates it."""


comptime HMX_PACK_VEC_WGT = get_defined_bool["SKYE_HMX_PACK_VEC_WGT", False]()
"""Whether the WEIGHT rule (`slab_is_depth=True`) may take the vector arm too.

OFF by default, and off keeps the 32x32 transpose butterfly out of the object
even when `HMX_PACK_VEC` is on. A separate gate because the two rules differ in
what they cost the STACK, not in what they compute: the activation rule holds
one vector pair live, while the butterfly holds all `HMX_TILE` rows — the whole
HVX register file — plus ~34 spill slots, and its `rows` array is a 4 KiB frame
allocation per inlined instance. Instantiated inside `mojo_mha_no_cache_f16`
(the attention K pack is the weight rule), that grew the kernel's frame from
11,784 to 16,520 bytes — past the 16 KiB stack of the thread that runs
participant 0, which is the FastRPC invoke thread on its SDK-default stack
(`mojo_npu_imp.c`: participant 0 runs "on the stack it already has"; nothing in
this repo raises it). The overflow is the order-149 refusal: `mha_no_cache`
`0x8000040d` at the first attention dispatch of every `HMX_PACK_VEC=1
HMX_ATTN=1` build, while the same tree at `HMX_ATTN=0` renders — its object
never instantiates the butterfly. Root cause and the frame census:
`.agentwork/flux-npu/takeover-2026-08-14/PACK_VEC_ROOTCAUSE.md`.

Turning this on is therefore a frame-budget question before it is a numerics
one: every kernel that instantiates the weight rule must be re-read for frame
growth against the smallest stack its participants can land on, and the
reference backend's own answer is to not have the problem — its transpose is a
`vscatter` holding three registers (`ggml-hexagon` `htp/hmx-utils.h`,
`hmx_interleave_rows_to_tiles`), the shape a rework here should take."""


comptime HMX_PACK_VEC_MHA = get_defined_bool["SKYE_HMX_PACK_VEC_MHA", False]()
"""Whether the attention V pack takes the split-pair vector arm.

Order 209 measured the rule split reaching matmul and NOT attention: the
matmul family's `PACK_ACT` fell 31.90x under `HMX_PACK_VEC` while
`mha_no_cache`'s fell 1.23x and stayed 35.4% of its own feeder time. The miss
is geometric, not a rule question. The TEAM arm's V panel is DV-tile major (PV
streams keys for a fixed output column, so a `:deep` load needs key tiles
contiguous), which makes every `hmx_attn_pack_v_share` call ONE crouton wide —
and `hmx_pack_panel_vec`'s step is always TWO adjacent croutons of a row, so
`tiles=1` falls through to the scalar tail on every call. The Q pack (whole
depth runs) vectorizes under `HMX_PACK_VEC` alone; that 1.23x is it.

The fix is not a new permute: both reference lineages ship the SAME
interleave with its two output croutons steered apart. The vendor V pack
stores "lo -> n-group, hi -> n-group+1", the groups a full panel stride apart
(`qaic/attention/_mha_per_tile_split_barrier.mojo`, `_rm_i8_to_crouton_mxblock`),
and the reference backend's `hmx_interleave_cols_to_tiles` writes `lo` to dim
tile `c` and `hi` to dim tile `c+1` at `tile_stride_elms` apart, slicing work
by ROW range (`ggml-hexagon` `htp/hmx-utils.h`, used at `flash-attn-ops.c:676`).
`hmx_pack_tile_pair_vec` below is that shape on the tile model, and the
attention step re-shares the V pack over KEY tiles so a feeder owns whole
row-pair steps (`hmx_attention.mojo`, `hmx_attn_pack_v_share_keyed`).

Frame class: the arm holds one vector pair live — the activation rule's O(1)
register shape, not the butterfly's — and the census at production defines is
pre-registered in `.agentwork/flux-npu/takeover-2026-08-14/ATTENTION_ATTACK.md`
before any device leg, per the order-149 lesson. Requires `HMX_PACK_VEC`
(silicon's vote for the interleave class) via `hmx_pack_vec_mha_selects`; this
flag alone changes nothing, and both off is the shipped scalar object."""


comptime HMX_PACK_VEC_CONV = get_defined_bool["SKYE_HMX_PACK_VEC_CONV", False]()
"""Whether the convolution's activation gather takes the vector arm on
interior panels.

The conv arm is the THIRD miss of `HMX_PACK_VEC`'s reach, after the V pack's:
`hmx_conv_pack_act` is its own loader body and never routes through
`hmx_pack_panel`, so the flag that recentred the matmul family's `PACK_ACT`
31.90x (order 209) leaves the conv gather byte-for-byte unchanged — measured
at the artifact: `mojo_conv_im2col_hmx_f16` compiles IDENTICAL under
`SKYE_HMX_PACK_VEC` either way, while `PACK_ACT` is 57.7% of the VAE conv's
device time (order 206 M3). The miss is a routing fact, not a geometry one.

What makes the vector arm REACH here is the interior panel: `panel_inside`
proves all `HMX_TILE` pixels sit on one output row with every tap in-bounds,
so pixel `slab`'s lane run is one contiguous read of the input map at row `h`,
column `(w + slab) * C_in + c` — the activation rule's own shape with the row
stride `C_in` instead of the matrix's `K`. The permute is the SAME
`vshuff(-2)` interleave silicon voted for (orders 112/149/209), issued over
two same-tap croutons and steered to the two adjacent destination croutons,
exactly as `hmx_pack_panel_vec[False]` steers it. A pair whose 64-element
depth run would straddle a tap boundary, and the odd head or tail of a share,
fall to the scalar body per pair — wrong bytes are not a degraded mode this
loader has.

Border panels stay scalar under this flag: their pixels wrap output rows and
read padding, which is the halo body's business (`hmx_conv_act.mojo`; the
64-wide mid/up0 stages are all-border by geometry and keep today's cost).

Frame class: one vector pair live, the activation rule's O(1) shape — not the
butterfly's. Requires `HMX_PACK_VEC` (silicon's vote for the interleave class)
via `hmx_pack_vec_conv_selects`; this flag alone changes no object, and both
off is the shipped scalar gather. Pricing and the census:
`.agentwork/flux-npu/takeover-2026-08-14/VAE_PACKVEC_PRICING.md`."""


comptime HMX_PACK_VEC_CONV_EDGE = get_defined_bool[
    "SKYE_HMX_PACK_VEC_CONV_EDGE", False
]()
"""Whether the conv gather's BORDER panels take the slab-pair vector arm.

Order 220 leg 5 measured the interior interleave at its full class ratio
(28.7x/32.7x on the 100%-reach 1x1 rows) wherever `panel_inside` admitted it,
and FLAT where it never fires — so the aggregate collapse is Amdahl on the
admission, not a property of the transform (the 87%-interior 3x3 rows read
3.8-6.9x for exactly this reason). A pad-1 border panel is >= 30 of 32 clean
pixels paying the full scalar walk for the one or two that touch padding or
wrap a row. This flag re-tests the interior proof per SLAB PAIR inside a
border panel: a pair whose two pixels each read their whole 64-element depth
run in bounds takes the same `vshuff(-2)` interleave with per-pixel addresses
(a row wrap just changes one load's row); any other pair takes the scalar walk
for exactly its two slabs. Bytes are `_pack_body[halo=True]`'s, which is the
twin's gate.

Requires the interior arm's flags via `hmx_pack_vec_conv_edge_selects`; alone
it changes no object. Off is #470's shipped split (interior vector, border
scalar)."""

comptime HMX_KGATHER_VSC = get_defined_bool["SKYE_HMX_KGATHER_VSC", False]()
"""Whether the attention K pack scatters with `vscatter` instead of gathering.

The 212 phase split prices the target: `mha_no_cache`'s `STAGE_WGT` is 23.95%
of feeder time — ~49.1 ms of the 205.0 ms dispatch — and unlike the matmul
family's, it is COMPUTE: the scalar transposing gather
(`hmx_pack_panel[True]`), one element load and store per K value, re-staged
once per q-block. The vector arms cannot reach it. The activation interleave
is the wrong rule, and the weight rule's butterfly is the order-149 refusal:
32 live rows plus a 4 KiB frame array on participant 0's 16 KiB stack
(`HMX_PACK_VEC_WGT` above, banned).

The reference backend never runs a butterfly here. Its K staging is a word
SCATTER — `hmx_interleave_rows_to_tiles` (`ggml-hexagon` `htp/hmx-utils.h`,
used by the K phase at `flash-attn-ops.c:628` @ `1692f9e50bb`): per source
row, one 64-element load per two destination croutons and one
`Q6_vscatter_RMVwV` whose offset vector IS the crouton byte rule — word `w`
at `w * 128 + lane * 4`, the layout's `(s / 2) * 128 + l * 4` with the depth
pair riding inside the word. Three vectors live, flat frame; the frame budget
passes by construction rather than by trimming. `hmx_pack_panel_vsc` below is
that shape on the tile model.

This port REJECTED the scatter once, for a stated reason that was true where
it was said: `vscatter` writes VTCM only and owes a `scatter_release`, and the
offline weight repack runs against `GENERIC` memory. At THIS site the
objection is void — the in-kernel K pack's destination is the VTCM arena
(`npu_kernel.mojo` `MHA_HMX_SPACE = AddressSpace.SHARED`) — so the rejection
is re-opened here and ONLY here: the butterfly stays the `GENERIC`-capable
arm, still gated by `HMX_PACK_VEC_WGT`, still banned for the frame.

Its own flag, deliberately, and independent of the `HMX_PACK_VEC` family: the
binding is a NEW instruction with no silicon vote and no compiler-selected
model to inherit — the same wrong-bits-with-nothing-to-signal-it class the
UNPACK_VEC mask carried — so the device A/B that flips it must move ONE
variable and open with a bit-identity leg. Composition map, by site: this
flag owns the attention K site; `HMX_PACK_VEC` owns the Q site (and matmul's
activations); `HMX_PACK_VEC` + `HMX_PACK_VEC_MHA` own the V site;
`HMX_PACK_VEC` + `HMX_PACK_VEC_WGT` own `hmx_pack_panel[True]`'s in-panel
routing, which the K site bypasses under this flag — the four touch disjoint
code and compose freely. Routing is `hmx_kgather_vsc_selects`, twin-pinned;
the census at production defines is pre-registered in
`.agentwork/flux-npu/takeover-2026-08-14/ATTENTION_SLICE2_VSCATTER.md` before
any device leg."""


@always_inline
def hmx_pack_panel[
    slab_is_depth: Bool, space: AddressSpace, dtype: DType = HMX_DTYPE
](
    dst: CroutonPanelOf[dtype, space],
    src: HmxRows2D,
    row0: Int,
    col0: Int,
    tiles: Int,
):
    """Packs `tiles` croutons out of `src` starting at `(row0, col0)`.

    `slab_is_depth` is the axis assignment and the only difference between the
    weight and activation packers:

    - `True` (weight): slab is the depth axis, lane is the source row, so this is
      a transposing gather out of the `[N, K]` pre-baked layout.
    - `False` (activation, and the output's inverse): slab is the source row,
      lane is the depth axis, so rows stay rows.

    The panel walks `tiles * HMX_TILE` columns of `src` from `col0`; the caller
    owns the bound, as it does at the matmul seam.

    Which packer runs is `hmx_pack_vec_selects` over (`HMX_PACK_VEC`,
    `HMX_PACK_VEC_WGT`) — the vector arm serves the activation rule under the
    first flag alone, and the weight rule only under both. The two must agree
    byte for byte: the scalar one below is the definition of the layout and the
    vector one is an optimisation of it, never a second opinion about it.
    """
    comptime if hmx_pack_vec_selects[
        slab_is_depth, HMX_PACK_VEC, HMX_PACK_VEC_WGT
    ]():
        hmx_pack_panel_vec[slab_is_depth, space, dtype](
            dst, src, row0, col0, tiles
        )
    else:
        hmx_pack_panel_scalar[slab_is_depth, space, dtype](
            dst, src, row0, col0, tiles
        )


@always_inline
def hmx_pack_vec_selects[
    slab_is_depth: Bool, pack_vec: Bool, pack_vec_wgt: Bool
]() -> Bool:
    """Whether `hmx_pack_panel` takes the vector arm for this rule and flags.

    A named function rather than an inline condition so the routing itself is a
    tested contract: the twin pins its whole truth table, and the dispatch above
    consumes this exact predicate, so the two cannot drift apart. The asymmetry
    it encodes — the weight rule needs BOTH flags — is `HMX_PACK_VEC_WGT`'s
    docstring, in one sentence: the butterfly's frame is what refused on device,
    and the activation interleave's is not.
    """
    return pack_vec and ((not slab_is_depth) or pack_vec_wgt)


@always_inline
def hmx_pack_vec_mha_selects[pack_vec: Bool, pack_vec_mha: Bool]() -> Bool:
    """Whether the attention step takes the split-pair V pack for these flags.

    A named function for the reason `hmx_pack_vec_selects` is one: the twin
    pins the truth table and `_attn_step_impl` consumes this exact predicate,
    so the routing cannot drift from the test. The asymmetry it encodes is
    `HMX_PACK_VEC_MHA`'s docstring in one sentence: the pair arm is the proven
    interleave class steered to strided destinations, so it never enters an
    object that did not also name the class itself.
    """
    return pack_vec and pack_vec_mha


@always_inline
def hmx_pack_vec_conv_selects[pack_vec: Bool, pack_vec_conv: Bool]() -> Bool:
    """Whether the conv loader's interior body takes the vector arm.

    A named function for the reason its two siblings are: the twin pins the
    truth table and `HmxConvActLoader.pack` consumes this exact predicate, so
    the routing cannot drift from the test. The asymmetry it encodes is
    `HMX_PACK_VEC_CONV`'s docstring in one sentence: the conv arm is the
    proven interleave class fed by the interior panel's contiguity proof, so
    it never enters an object that did not also name the class itself.
    """
    return pack_vec and pack_vec_conv


@always_inline
def hmx_pack_vec_conv_edge_selects[
    pack_vec: Bool, pack_vec_conv: Bool, edge: Bool
]() -> Bool:
    """Whether the conv loader's BORDER body takes the slab-pair vector arm.

    Strictly narrower than `hmx_pack_vec_conv_selects`: the edge arm is the
    same interleave class applied under a weaker proof (per slab pair instead
    of per panel), so it never enters an object whose interior arm is not
    already on. The twin pins the eight-row truth table.
    """
    return hmx_pack_vec_conv_selects[pack_vec, pack_vec_conv]() and edge


def hmx_kgather_vsc_selects[kgather_vsc: Bool]() -> Bool:
    """Whether the attention K pack takes the scatter arm for this flag.

    One flag and no conjunction, and the shape of the predicate is itself the
    contract: the scatter arm is a NEW instruction binding, not a steering of
    the proven interleave class, so tying it to `HMX_PACK_VEC` would let a
    pack-vec A/B move two mechanisms under one variable — the exact
    two-variable catch the 209 lane wrote down. A named function all the same,
    for the reason the other selects are: the twin pins both rows and
    `hmx_attn_pack_k_share` consumes this exact predicate, so a build naming
    the flag routes the K site and a build not naming it is byte-identical,
    with nothing in between. The site bypass it grants wins over
    `hmx_pack_panel[True]`'s own in-panel routing, which keeps the
    `HMX_PACK_VEC_WGT` bisection control reachable by naming THAT pair without
    this flag.
    """
    return kgather_vsc


@always_inline
def hmx_pack_panel_scalar[
    slab_is_depth: Bool, space: AddressSpace, dtype: DType = HMX_DTYPE
](
    dst: CroutonPanelOf[dtype, space],
    src: HmxRows2D,
    row0: Int,
    col0: Int,
    tiles: Int,
):
    """`hmx_pack_panel` one element at a time: the layout's reference.

    Kept, and kept as the thing the vector packer is gated against, for three
    reasons. It is the definition of the crouton rule that the byte-order test
    pins directly. It is the only arm that runs at every shape, so it is the
    tail of an odd tile count. And it is what the twin compares against, which is
    the whole of the evidence that a permute network computed the right
    permutation.
    """
    for t in range(tiles):
        for slab in range(HMX_TILE):
            for lane in range(HMX_TILE):
                var depth = (
                    col0 + t * HMX_TILE + (slab if slab_is_depth else lane)
                )
                var row = row0 + (lane if slab_is_depth else slab)
                # Alignment is stated because `TileTensor` defaults it to 1
                # off-GPU, and an align-1 fp16 store lowers on Hexagon to two
                # `memb` plus a shift instead of one `memh`. Both buffers are
                # `Scalar[HMX_DTYPE]` allocations; a crouton's own base is
                # `HMX_TILE_BYTES`-aligned, which is stricter still.
                dst.store[1, alignment=align_of[Scalar[dtype]]()](
                    Coord(
                        t,
                        slab // HmxSplit[dtype],
                        lane,
                        slab % HmxSplit[dtype],
                    ),
                    src.load[1, alignment=align_of[Scalar[HMX_DTYPE]]()](
                        Coord(row, depth)
                    ).cast[dtype](),
                )


comptime _HMX_XPOSE_PASSES = 5
"""Exchange stages the weight rule's transpose takes: `log2(HMX_TILE)`.

Written as a number because `comptime for` walks a range; the assert in
`hmx_pack_panel_vec` ties it back to `HMX_TILE` so the two cannot drift."""


@always_inline
def hmx_pack_panel_vec[
    slab_is_depth: Bool, space: AddressSpace, dtype: DType = HMX_DTYPE
](
    dst: CroutonPanelOf[dtype, space],
    src: HmxRows2D,
    row0: Int,
    col0: Int,
    tiles: Int,
):
    """`hmx_pack_panel`'s permutation issued as HVX permutes, not element loads.

    Same arguments and the same bytes as `hmx_pack_panel_scalar`, which is the
    only claim this function makes and the only one the twin's byte-identity gate
    can check.

    TWO croutons per step, because the geometry says so rather than as an
    unrolling choice: one HVX vector is 64 fp16, a crouton row is 32 columns, so
    the load that feeds a permute always spans two adjacent croutons and the pair
    the permute returns is exactly those two. An odd `tiles` therefore has a
    one-crouton tail, and the tail is the scalar packer — a partial vector step
    would have to read 32 columns past the panel, which is a source the caller
    never promised exists.

    The two rules use the same instruction and nothing else in common:

    - `False` (activation): crouton vector `p` of a tile is source rows
      `row0 + 2p` and `row0 + 2p + 1` interleaved halfword-wise, which is one
      `vshuff(-2)` per 128 elements and no transpose at all. The reference builds
      its V operand this way (`ggml-hexagon` `htp/hmx-utils.h:180`).
    - `True` (weight): crouton vector `p` gathers word `p` of all 32 source rows,
      so the step is a 32x32 word transpose — five exchange stages over the 32
      loaded rows, the vendor's `transposeHMX` butterfly at full width
      (`qaic/attention/_mha_per_tile_split_barrier.mojo:504`). The reference uses
      a `vscatter` here instead; that form is VTCM-only and owes a scatter
      release, and this packer also runs against `GENERIC` in the offline repack,
      so the butterfly is what serves both address spaces with one instruction
      bound.

    Element alignment, not vector alignment, is stated on both sides even though
    a crouton panel's base is `HMX_TILE_BYTES`-aligned by `hmx_panel`'s contract:
    the twin allocates its panels with `alloc`, whose default alignment is the
    element's, so claiming 128 here would be a promise the gate's own buffers
    break.
    """
    comptime assert HmxSplit[dtype] == 2, (
        "the interleave packs one lane word from two slabs; a wider split"
        " (int8) needs a different permute, not a different constant"
    )
    comptime assert (
        1 << _HMX_XPOSE_PASSES
    ) == HMX_TILE, "the transpose butterfly must span the tile"

    comptime LANES = hvx_lanes[dtype]()
    comptime STEP_TILES = LANES // HMX_TILE
    comptime SLAB_PAIRS = HMX_TILE // HmxSplit[dtype]
    comptime ALIGN = align_of[Scalar[dtype]]()

    var steps = tiles // STEP_TILES
    for s in range(steps):
        var t0 = s * STEP_TILES
        var col = col0 + t0 * HMX_TILE

        comptime if slab_is_depth:
            var rows = Array[SIMD[dtype, LANES], HMX_TILE](uninitialized=True)
            comptime for lane in range(HMX_TILE):
                rows[lane] = src.load[LANES, alignment=ALIGN](
                    Coord(row0 + lane, col)
                ).cast[dtype]()
            # Each stage exchanges word blocks between rows `stride` apart, so
            # after `log2(HMX_TILE)` of them row `j` holds word `j` of every
            # source row -- which is the crouton vector for slab pair `j`.
            comptime for shift in range(_HMX_XPOSE_PASSES):
                comptime stride = 1 << shift
                comptime for j in range(0, HMX_TILE, 2 * stride):
                    comptime for k in range(stride):
                        var pair = hvx_vshuff[HMX_LANE_BYTES * stride](
                            rows[j + k + stride], rows[j + k]
                        )
                        rows[j + k] = pair.slice[LANES, offset=0]()
                        rows[j + k + stride] = pair.slice[LANES, offset=LANES]()
            comptime for p in range(SLAB_PAIRS):
                dst.store[LANES, alignment=ALIGN](Coord(t0, p, 0, 0), rows[p])
                dst.store[LANES, alignment=ALIGN](
                    Coord(t0 + 1, p, 0, 0), rows[SLAB_PAIRS + p]
                )
        else:
            comptime for p in range(SLAB_PAIRS):
                # `vv` carries the even slab and `vu` the odd one, which is the
                # order the interleave writes them in.
                var pair = hvx_vshuff[HVX_SHUFF_INTERLEAVE](
                    src.load[LANES, alignment=ALIGN](
                        Coord(row0 + HmxSplit[dtype] * p + 1, col)
                    ).cast[dtype](),
                    src.load[LANES, alignment=ALIGN](
                        Coord(row0 + HmxSplit[dtype] * p, col)
                    ).cast[dtype](),
                )
                dst.store[LANES, alignment=ALIGN](
                    Coord(t0, p, 0, 0), pair.slice[LANES, offset=0]()
                )
                dst.store[LANES, alignment=ALIGN](
                    Coord(t0 + 1, p, 0, 0), pair.slice[LANES, offset=LANES]()
                )

    var packed = steps * STEP_TILES
    if packed < tiles:
        hmx_pack_panel_scalar[slab_is_depth, space, dtype](
            hmx_subpanel(dst, packed, tiles - packed),
            src,
            row0,
            col0 + packed * HMX_TILE,
            tiles - packed,
        )


@always_inline
def hmx_pack_tile_pair_vec[
    space: AddressSpace, dtype: DType = HMX_DTYPE
](
    dst_lo: CroutonPanelOf[dtype, space],
    dst_hi: CroutonPanelOf[dtype, space],
    src: HmxRows2D,
    row0: Int,
    col0: Int,
):
    """Packs one activation-rule vector step with its two croutons split apart.

    `hmx_pack_panel_vec`'s activation arm assumes the pair a `vshuff` returns
    lands in two ADJACENT croutons of one panel — true for a depth run, wrong
    for the attention V panel, whose two source-adjacent tiles sit a column
    stride apart (DV-tile major). This is the same interleave with the two
    stores steered to two one-crouton subpanels the CALLER names, which is how
    both reference lineages spell it: the vendor V pack's "lo -> n-group,
    hi -> n-group+1" and the reference backend's `hmx_interleave_cols_to_tiles`
    (`lo` to dim tile `c`, `hi` to dim tile `c+1`, `tile_stride_elms` apart).

    Byte identity is against `hmx_pack_panel_scalar[False]` run once per
    destination tile, which is the claim the twin gates. The register shape is
    the activation rule's — one vector pair live, no `rows` array — so the
    frame it rides into an attention kernel is the class order 149 CLEARED,
    not the butterfly it refused; the census that proves that is owed before
    any device leg.

    `src` must have `2 * HMX_TILE` readable columns at `col0`: the load that
    feeds the permute spans both tiles, exactly as the in-panel packer's does.

    Args:
        dst_lo: The crouton taking the pair's low vectors (source columns
            `[col0, col0 + HMX_TILE)`).
        dst_hi: The crouton taking the high vectors (source columns
            `[col0 + HMX_TILE, col0 + 2 * HMX_TILE)`).
        src: The row-major source.
        row0: First source row; the step covers `HMX_TILE` rows.
        col0: First source column of the two-tile span.
    """
    comptime assert HmxSplit[dtype] == 2, (
        "the interleave packs one lane word from two slabs; a wider split"
        " (int8) needs a different permute, not a different constant"
    )
    comptime LANES = hvx_lanes[dtype]()
    comptime SLAB_PAIRS = HMX_TILE // HmxSplit[dtype]
    comptime ALIGN = align_of[Scalar[dtype]]()

    comptime for p in range(SLAB_PAIRS):
        var pair = hvx_vshuff[HVX_SHUFF_INTERLEAVE](
            src.load[LANES, alignment=ALIGN](
                Coord(row0 + HmxSplit[dtype] * p + 1, col0)
            ).cast[dtype](),
            src.load[LANES, alignment=ALIGN](
                Coord(row0 + HmxSplit[dtype] * p, col0)
            ).cast[dtype](),
        )
        dst_lo.store[LANES, alignment=ALIGN](
            Coord(0, p, 0, 0), pair.slice[LANES, offset=0]()
        )
        dst_hi.store[LANES, alignment=ALIGN](
            Coord(0, p, 0, 0), pair.slice[LANES, offset=LANES]()
        )


@always_inline
def hmx_pack_panel_vsc[
    space: AddressSpace, dtype: DType = HMX_DTYPE
](
    dst: CroutonPanelOf[dtype, space],
    src: HmxRows2D,
    row0: Int,
    col0: Int,
    tiles: Int,
):
    """The WEIGHT rule of `hmx_pack_panel`, issued as word scatters.

    Same arguments and the same bytes as `hmx_pack_panel_scalar[True]`, which
    is the only claim this function makes and the only one the twin's
    byte-identity gate can check. The mechanism is the reference backend's K
    staging (`hmx_interleave_rows_to_tiles`, `ggml-hexagon` `htp/hmx-utils.h`,
    used at `flash-attn-ops.c:628` @ `1692f9e50bb`), adopted onto the tile
    model with the one delta every ported kernel carries (subpanels for raw
    pointers) and one stated addition: the reference orders its scatters
    against the array's reads with a work-queue phase boundary, and this port
    closes the same window explicitly with `hvx_scatter_release`.

    The geometry is the file-header rule read word-wise. A 64-element load
    from source row `lane` carries depth pairs `(2w, 2w + 1)` in word `w` —
    already the crouton's 4-byte lane word — so the whole transpose is WHERE
    each word lands: byte `w * 128 + lane * 4` of a TWO-crouton region, the
    layout's `(s / 2) * 128 + l * 4` with `w * 128` folding both tiles because
    16 slab pairs are exactly one tile. One offset table serves every call;
    the running values are two offset vectors and the loaded pair — three
    vector registers of standing state against the butterfly's thirty-two,
    which is the whole frame argument.

    TWO croutons per scatter for the reason `hmx_pack_panel_vec` steps two:
    one HVX load spans 64 depth elements. An odd `tiles` tail is the scalar
    packer over the last crouton — a partial vector step would read 32 columns
    the caller never promised. One stated deviation from the reference's loop
    shape: it walks rows in PAIRS because its row count can be odd and the
    missing row shuffles against `vzero`; a crouton lane axis is always
    exactly `HMX_TILE` rows, so this walks single rows and keeps three vectors
    live instead of five — the difference between riding inside the kernel's
    frame budget and spilling against it (the census pins this).

    VTCM-ONLY on device: the scatter writes VTCM and nothing else, so a device
    compile of this function demands an `AddressSpace.SHARED` destination
    (`hvx_vscatter_words` refuses others at compile time). That constraint is
    why the in-panel weight rule keeps the butterfly for `GENERIC` repacking
    and why THIS arm lives beside it instead of replacing it.
    """
    comptime assert HmxSplit[dtype] == 2, (
        "the scatter word carries one depth pair; a wider split (int8) needs"
        " a different word rule, not a different constant"
    )
    comptime LANES = hvx_lanes[dtype]()
    comptime STEP_TILES = LANES // HMX_TILE
    comptime ALIGN = align_of[Scalar[dtype]]()
    comptime PAIR_BYTES = HMX_TILE * HMX_LANE_BYTES
    comptime PAIR_REGION = STEP_TILES * HmxTileBytes[dtype] - 1
    comptime assert (
        HVX_WORDS * PAIR_BYTES == PAIR_REGION + 1
    ), "one scatter's 32 words must span exactly the two-crouton region"

    var steps = tiles // STEP_TILES
    if steps > 0:
        # One row per iteration, NOT the reference's row pairing: its pair
        # loop exists to shuffle a missing odd row against vzero, and a
        # crouton lane axis is always exactly HMX_TILE rows, so the pairing
        # would buy nothing here and holds five vectors live where this
        # shape holds three -- on a kernel whose participant-0 frame is
        # budgeted in hundreds of bytes, the narrower live set is the point.
        for lane in range(HMX_TILE):
            var offs = iota[DType.int32, HVX_WORDS]() * Int32(
                PAIR_BYTES
            ) + Int32(lane * HMX_LANE_BYTES)
            for s in range(steps):
                var v = src.load[LANES, alignment=ALIGN](
                    Coord(row0 + lane, col0 + s * LANES)
                ).cast[dtype]()
                hvx_vscatter_words[PAIR_REGION](
                    dst.ptr_at_offset(Coord(s * STEP_TILES, 0, 0, 0)),
                    offs,
                    bitcast[DType.int32, HVX_WORDS](v),
                )
        hvx_scatter_release(dst.ptr_at_offset(Coord(0, 0, 0, 0)))

    var packed = steps * STEP_TILES
    if packed < tiles:
        hmx_pack_panel_scalar[True, space, dtype](
            hmx_subpanel(dst, packed, tiles - packed),
            src,
            row0,
            col0 + packed * HMX_TILE,
            tiles - packed,
        )


comptime HMX_UNPACK_VEC = get_defined_bool["SKYE_HMX_UNPACK_VEC", False]()
"""Whether `hmx_unpack_panel` scatters with HVX permutes instead of element
stores.

OFF by default for the reason `HMX_PACK_VEC` is, and with one more: this arm
issues `V6_vS32b_qpred_ai`, whose whole content is which BYTES of a window a
predicate writes, and the twin checks that against a transcription of the
instruction rather than against silicon. A wrong mask is a wrong ANSWER in
half a row with nothing to signal it. A `-D SKYE_HMX_UNPACK_VEC=true` build
define (`build.sh HMX_UNPACK_VEC=1`) selects it from the same reviewed source,
and the device A/B opens with a bit-identity leg for that reason.

Separate from `HMX_PACK_VEC` deliberately, even though both are #332's one idea
in its two directions: order 145 prices the unpack at 76.16% of feeder time and
the pack at a fraction of that, so an A/B that moved both at once could not say
which one paid."""


comptime HMX_UNPACK_ARM_SCALAR_OFF: UInt32 = 1
"""`HMX_UNPACK_VEC` is off (or the call is not the readout rule): scalar ran."""

comptime HMX_UNPACK_ARM_VECTOR: UInt32 = 2
"""The flag is on AND `hmx_unpack_vec_admits` admitted: the vector arm ran."""

comptime HMX_UNPACK_ARM_SCALAR_REFUSED: UInt32 = 3
"""The flag is ON and admission REFUSED, so the scalar arm ran anyway.

THREE codes rather than two, and this is the third. `SCALAR_OFF` and this one
are both "the scalar arm ran" and folding them would make a build whose whole
point is the vector arm indistinguishable from one where the flag never took
effect -- #275's failure class, and here it is the exact silent no-op the flag's
admission gate can produce. A probe-off A/B that reports this code measured
nothing, and has to say so rather than reading as a clean null."""


comptime HMX_UNPACK_ARM_PAIRED: UInt32 = 4
"""Slice 2's paired readout ran: one store covered two adjacent column tiles.

Named BEFORE its producer exists, and that ordering is the point.
`_require_unpack_arm_known` refuses a code it cannot name, on the graph path a
render actually runs, so the first render of a build whose unpack reports a
fourth code raises at its first matmul -- with the arm working correctly. The
writer and the reader therefore have to cross the flag boundary in opposite
order: the reader accepts the code first, the writer emits it second. #393 is
this failure class arriving after the work was built; this is it arriving before,
at the cost of one constant.

The BEHAVIOURAL four-code test belongs to slice 2 rather than here: reaching this
code the way `SCALAR_REFUSED` is reached -- through a destination the gate
declines -- needs slice 2's paired arm to exist. What is testable today is its
DISTINCTNESS from the other three, which is the property a witness reader scores
on and the one a renumbering would silently break."""


comptime HMX_UNPACK_ARM_NATIVE: UInt32 = 5
"""The `output_native` readout ran: no unpack at all, a linear crouton copy.

Under `output_native` (the annihilation design's output-side flag) the readout
tile is written to `C` in crouton order, so neither unpack arm runs and
reporting either scalar code would misreport the arm — the exact class
`SCALAR_OFF`/`SCALAR_REFUSED` exist to keep apart. The reader
(`_require_unpack_arm_known`) accepts this code in the same change that adds
the writer, per `HMX_UNPACK_ARM_PAIRED`'s ordering rule: the reader crosses
first. No shipped configuration emits it — `output_native` defaults off
everywhere and only an explicit instantiation (the rung-1 microbench) sets
it."""


@always_inline
def hmx_unpack_panel[
    slab_is_depth: Bool,
    space: AddressSpace,
    dtype: DType = HMX_DTYPE,
    armed: Bool = HMX_UNPACK_VEC,
](
    dst: HmxRows2D,
    src: CroutonPanelOf[dtype, space],
    row0: Int,
    col0: Int,
    tiles: Int,
) -> UInt32:
    """Scatters `tiles` croutons back to row-major `dst` at `(row0, col0)`.

    The exact inverse of `hmx_pack_panel` under the same `slab_is_depth`, which
    is what makes a round trip a bit-exact identity rather than a tolerance.

    This is also the output path: an accumulator tile reads out as slab `m`,
    lane `n` — the activation's axis assignment over `C[M, N]` — so an output
    unpack is this function at `slab_is_depth=False` with one tile, not a third
    rule.

    Which arm runs is `HMX_UNPACK_VEC` AND the destination's own store geometry,
    and the second half of that is why the vector arm is admitted at runtime
    rather than selected at compile time: the readout's masked store needs a
    64-byte-aligned run, which is a fact about a caller's buffer. The scalar
    unpack below is the definition of the layout in either case.

    The vector arm serves the READOUT rule only (`slab_is_depth=False`, fp16).
    The weight rule's inverse would be the transpose butterfly a second time and
    has no caller — every unpack in the tree is a readout — so it stays scalar
    rather than becoming an arm no shape gate exercises.

    Parameters:
        slab_is_depth: Which layout rule the croutons follow.
        space: The source panel's address space.
        dtype: The crouton operand type.
        armed: Whether the vector arm is compiled in at all. Defaults to
            `HMX_UNPACK_VEC`, so the shipped selection is the flag and a
            flag-off object is byte-identical to its base. It is a parameter
            rather than a direct read of the flag for the reason
            `hmx_phase_mark`'s `enabled` is: a host gate has to drive BOTH arms
            from one build to show the three codes are distinguishable, and a
            gate that could only ever observe the shipped one would be
            asserting the default rather than the seam.

    Returns:
        Which arm actually ran, as an `HMX_UNPACK_ARM_*` code. Reported through
        the return rather than a witness store because this module is
        host-linkable and holds no witness pointer — the same rule
        `matmul_bt_stripe_hmx`'s refusal follows, so the skel's witness carries
        it. Callers with nothing to report it to discard it.
    """
    comptime if armed and not slab_is_depth and dtype == HMX_DTYPE:
        if hmx_unpack_vec_admits(dst, row0, col0):
            hmx_unpack_panel_vec[space, dtype](dst, src, row0, col0, tiles)
            return HMX_UNPACK_ARM_VECTOR

        hmx_unpack_panel_scalar[slab_is_depth, space, dtype](
            dst, src, row0, col0, tiles
        )
        return HMX_UNPACK_ARM_SCALAR_REFUSED

    hmx_unpack_panel_scalar[slab_is_depth, space, dtype](
        dst, src, row0, col0, tiles
    )
    return HMX_UNPACK_ARM_SCALAR_OFF


@always_inline
def hmx_unpack_panel_scalar[
    slab_is_depth: Bool, space: AddressSpace, dtype: DType = HMX_DTYPE
](
    dst: HmxRows2D,
    src: CroutonPanelOf[dtype, space],
    row0: Int,
    col0: Int,
    tiles: Int,
):
    """`hmx_unpack_panel` one element at a time: the layout's reference.

    Kept as `hmx_pack_panel_scalar` is kept, and for the same three reasons: it
    is the definition the byte-order test pins, it is the arm that runs at every
    destination geometry, and it is what the twin compares the vector arm
    against.
    """
    for t in range(tiles):
        for slab in range(HMX_TILE):
            for lane in range(HMX_TILE):
                var depth = (
                    col0 + t * HMX_TILE + (slab if slab_is_depth else lane)
                )
                var row = row0 + (lane if slab_is_depth else slab)
                dst.store[1, alignment=align_of[Scalar[HMX_DTYPE]]()](
                    Coord(row, depth),
                    src.load[1, alignment=align_of[Scalar[dtype]]()](
                        Coord(
                            t,
                            slab // HmxSplit[dtype],
                            lane,
                            slab % HmxSplit[dtype],
                        )
                    ).cast[HMX_DTYPE](),
                )


@always_inline
def _hmx_run_offset(dst: HmxRows2D, row: Int, col: Int) -> Int:
    """Where the readout run at `(row, col)` starts inside its 128-byte window.

    The low address bits `hvx_store_masked` ignores are exactly what
    `hvx_run_mask` needs back. Under `hmx_unpack_vec_admits` this is 0 or
    `HVX_HALF_BYTES` and never anything else, which is the whole content of that
    gate.
    """
    return Int(dst.ptr_at_offset(Coord(row, col))) & (HVX_BYTES - 1)


@always_inline
def hmx_unpack_vec_admits(dst: HmxRows2D, row0: Int, col0: Int) -> Bool:
    """Whether `hmx_unpack_panel_vec`'s store geometry holds for `dst`.

    Two conditions, and together they say that every 64-byte run the arm writes
    is one HALF of one 128-byte window rather than a straddle of two:

    - the first run's address is 64-byte aligned. `col0` is a whole number of
      tiles at every caller, so this is a fact about `dst`'s base;
    - the row stride is a whole number of runs, so every other row of the tile
      inherits that alignment, and the window half a run lands in depends only
      on the row's PARITY — which is what lets the arm hoist its two masks out
      of the crouton loop instead of branching per row.

    Runtime rather than a shape refusal because it is a property of a buffer: a
    `TileTensor` over an `alloc` whose alignment is its element's is the
    ordinary host-side destination and is NOT admitted, so a host gate that
    means to exercise the arm has to align its own output and assert this
    returned True.
    """
    var stride_bytes = (
        Int(dst.layout.stride[0]()) * size_of[Scalar[HMX_DTYPE]]()
    )
    if stride_bytes % HVX_HALF_BYTES != 0:
        return False
    return Int(dst.ptr_at_offset(Coord(row0, col0))) % HVX_HALF_BYTES == 0


@always_inline
def hmx_unpack_panel_vec[
    space: AddressSpace, dtype: DType = HMX_DTYPE
](
    dst: HmxRows2D,
    src: CroutonPanelOf[dtype, space],
    row0: Int,
    col0: Int,
    tiles: Int,
):
    """`hmx_unpack_panel_scalar`'s readout issued as HVX permutes and stores.

    Same bytes as the scalar unpack, which is the only claim it makes and the
    only one the twin's byte-identity gate can check.

    ONE crouton per step, where the vector PACKER takes two, and the asymmetry
    is the destination's rather than a choice. A pack reads 64 source columns
    and writes two whole crouton vectors; an unpack writes 32 columns of ONE row
    per half vector, and consecutive rows are a stride apart. Nor is there a
    second tile to pair with: every caller unpacks a single tile
    (`matmul_kernel.mojo:1417`, `:1805`, `hmx_attention.mojo:859`) because the
    array produces one accumulator tile at a time and the feeder walks m-panels
    at a FIXED column tile (`matmul_kernel.mojo:2098`). So the step is a 64-byte
    run, and the instruction that writes 64 bytes without disturbing the other
    64 is the masked store.

    DEALING A VECTOR AGAINST ITSELF is what makes one mask serve either half:
    `hvx_vdeal(v, v)` at `HVX_DEAL_SEPARATE` de-interleaves the crouton's slab
    pair and puts each source row in BOTH halves of its result vector, so the
    mask alone decides where the run lands and the loop needs no rotate, no
    read-modify-write and no branch. That is LLVM's model of `V6_vdealvdd`
    (`hvx_permute.mojo`) taken at its word; the twin executes the same model,
    and only the device leg says the part agrees with it.

    fp16 only, and the comptime assert says so rather than the caller: an int8
    readout is `HmxSplit == 4`, a lane word of FOUR slabs, so its de-interleave
    is a different permute and its run is a different width. The byte arm
    requants through `hmx_int8_unpack_out` and never reaches here.
    """
    comptime assert dtype == HMX_DTYPE, (
        "the readout deal separates a two-slab lane word; a byte operand's"
        " four-slab word needs a different permute, not a different constant"
    )

    comptime LANES = hvx_lanes[dtype]()
    comptime SLAB_PAIRS = HMX_TILE // HmxSplit[dtype]
    comptime ALIGN = align_of[Scalar[dtype]]()
    comptime RUN_BYTES = HMX_TILE * size_of[Scalar[HMX_DTYPE]]()
    comptime assert RUN_BYTES == HVX_HALF_BYTES, (
        "a readout run is one tile row, and the mask geometry below assumes it"
        " is half a vector"
    )

    for t in range(tiles):
        var col = col0 + t * HMX_TILE
        # Hoisted per tile, not per row: the window half a run lands in is
        # `(base + row * stride + col) / 64 & 1`, and with both terms whole
        # numbers of runs the row contributes only its parity. It DOES change
        # with the tile, hence per `t` rather than once.
        var q_even = hvx_run_mask(_hmx_run_offset(dst, row0, col), RUN_BYTES)
        var q_odd = hvx_run_mask(_hmx_run_offset(dst, row0 + 1, col), RUN_BYTES)

        comptime for p in range(SLAB_PAIRS):
            comptime slab = HmxSplit[dtype] * p
            # ONE load, named, because both operands of the deal are the same
            # vector -- that is the trick, not a repeated read.
            var v = src.load[LANES, alignment=ALIGN](Coord(t, p, 0, 0))
            var rows = hvx_vdeal[HVX_DEAL_SEPARATE](v, v)
            hvx_store_masked(
                q_even,
                dst.ptr_at_offset(Coord(row0 + slab, col)),
                rows.slice[LANES, offset=0](),
            )
            hvx_store_masked(
                q_odd,
                dst.ptr_at_offset(Coord(row0 + slab + 1, col)),
                rows.slice[LANES, offset=LANES](),
            )


comptime HMX_ACT_PREFETCH = get_defined_bool["SKYE_HMX_ACT_PREFETCH", False]()
"""Whether the ACTIVATION-side operands are hinted to L2 ahead of their reader.

OFF by default, and off compiles every hint and every address computed for one
away, so the object is the un-hinted build — the A/B control. A
`-D SKYE_HMX_ACT_PREFETCH=true` build define (`build.sh HMX_ACT_PREFETCH=1`) for
the reason `SKYE_HMX_EMIT_MMA` is one: both arms come from the same reviewed
source with no edit in between.

ONE flag for BOTH activation-side hints — this file's panel hint and the
attention kernel's K/V hint (`npu_kernel.mojo`) — because they are one device
A/B with one variable. The WEIGHT side is already hinted unconditionally
(`MATMUL_PF_B_ROWS`); what has never been hinted is the operand the consumer
walks, and order 096 leg A measured the strided access class it is read through
at 26x the contiguous one.

A prefetch moves cache lines and never data, so neither arm may move a byte of
any output. That is a gate rather than a hope: the host twin runs the stripe
differential with the flag ON and with it OFF, and both must be bit-identical to
the packing reference."""


comptime HMX_ACT_XBLOCK_PF = get_defined_bool[
    "SKYE_HMX_ACT_XBLOCK_PF", False
]()
"""Whether the stripe's drain phase hints the NEXT block's first panel.

OFF by default. The in-block hint above covers panel `p + 1` while packing
`p`, which leaves exactly one panel per block walking cold lines: panel 0,
packed at the top of the next block's phase 0 with nothing ahead of it. This
flag adds ONE hint call per feeder per block, from the drain phase (the step
with no weight panel left to stage), naming the next block's panel-0 rows for
the feeder's own share — the same one-panel-per-issue distance the order-097
sweep priced, one block earlier in time.

Additive to `SKYE_HMX_ACT_PREFETCH`, never a substitute: `hmx_prefetch_rows`
self-gates off that master flag, so this flag with the master off compiles
every added site away and the object is the un-hinted build. The block-level
act ping-pong this slice descends from is priced DEAD at rev 7 —
`ACT_PINGPONG.md` carries the arithmetic (phase-0 is ~2% of feeder time
post-pack_vec, and halving `act_slots` for two-block residency doubles the
weight-panel restream) — so a hint is the whole of what the mechanism still
affords."""


comptime _IS_HEXAGON = is_triple["hexagon-unknown-unknown-elf"]()
"""Whether this compile targets the device.

`l2fetch` is a Hexagon instruction no host assembler accepts, and this file is
compiled for the arm64 twin from the same source. Gating only the ISSUE — not
the geometry above it — is what lets the twin run BOTH arms of the flag and so
makes bit-identity between them a tested claim; it is the split
`hmx_mma_panel` makes between `_hmx_mma_issue` and `_hmx_mma_model`."""


@always_inline
def hmx_prefetch_rows(
    src: HmxRows2D, row0: Int, col0: Int, rows: Int, tiles: Int
):
    """Asks L2 for the region a later `hmx_pack_panel` will read.

    Target: Hexagon v81. The region is `rows` rows of `tiles` croutons' worth of
    columns from `(row0, col0)` — the exact argument shape `hmx_pack_panel`
    takes, so a hint cannot describe a region the pack does not read.

    `l2fetch(Rs32,Rtt32)` reads its 2D region out of the control word: height in
    `Rtt[15:0]`, width in `Rtt[31:16]`, stride in `Rtt[47:32]`, direction in
    `Rtt[63:48]`. This is the STRIDED form — an activation panel is 32 rows of a
    `[M, K]` matrix, so its width is a crouton share and its stride is the whole
    row. Width, height and stride are 16-bit fields; a region that does not fit
    them is DROPPED rather than truncated, because a truncated control word
    names a different region — harmless to values, but it would spend bandwidth
    on lines nothing reads and quietly stop being the hint under measurement.

    Distance is the caller's, and the caller's rule is exactly one panel ahead:
    order 096 leg A found the lead length is not what the gain is sensitive to,
    while the `MATMUL_PF_B_ROWS` sweep recorded in `npu_kernel.mojo` found that
    fetching MORE per issue walks the gain monotonically back to the un-hinted
    time, on an L2 the M-tile budget already fills to its knee.

    Args:
        src: The row-major activation the pack gathers from.
        row0: First row of the region.
        col0: First column of the region.
        rows: Rows to hint.
        tiles: Crouton tiles of columns to hint, from `col0`.
    """
    comptime if not HMX_ACT_PREFETCH:
        return

    if rows <= 0 or tiles <= 0:
        return
    # `HmxRows2D` is row-major by construction, so the row stride is the column
    # count; the pointer comes from the layout rather than from address
    # arithmetic, as every other operand address in this file does.
    var stride_bytes = src.layout.shape[1]() * size_of[HMX_DTYPE]()
    var row_bytes = tiles * HMX_TILE * size_of[HMX_DTYPE]()
    if rows > 0xFFFF or row_bytes > 0xFFFF or stride_bytes > 0xFFFF:
        return
    var addr = src.ptr_at_offset(Coord(row0, col0))
    var control = (
        (UInt64(stride_bytes) << 32) | (UInt64(row_bytes) << 16) | UInt64(rows)
    )
    comptime if _IS_HEXAGON:
        llvm_intrinsic["llvm.hexagon.Y5.l2fetch", NoneType](
            addr.bitcast[NoneType](), control
        )
    else:
        _ = addr
        _ = control


comptime HMX_RECORD_VECTORS = 2
"""Vectors the `bias = mxmem2` load consumes: the requant record is a pair."""

comptime HMX_RECORD_LANES = 32
"""Lane words per record vector, one per output column."""

comptime HMX_RECORD_BYTES = (
    HMX_RECORD_VECTORS * HMX_RECORD_LANES * size_of[DType.uint32]()
)
"""Bytes the requant record occupies (256)."""


comptime HmxRequantRecord[space: AddressSpace] = TileTensor[
    DType.uint32,
    type_of(row_major(Coord(0, 0))),
    origin=MutUntrackedOrigin,
    address_space=space,
]
"""The `[2, 32]` word image the `:after` store requants through.

A layout, like the crouton, so its contents are written in Mojo and checkable off
the device; `bias = mxmem2` only loads them. Row 0 carries the per-column
records, row 1 is zero."""


@always_inline
def hmx_requant_word(
    scale: Scalar[HMX_DTYPE], bias: Scalar[HMX_DTYPE]
) -> Scalar[DType.uint32]:
    """Packs one column's requant record: LOW halfword scale, HIGH halfword bias.

    This encoding was UNRESOLVED in the KB entry, which recorded two readings
    that its own reference's call sites could not separate, and warned against
    picking one from that evidence. It is now settled by the llama.cpp/MNN
    cross-check (`htp-ops-lib`'s `hmx_fp16_scale_bias_word`, validated at cosine
    similarity 0.99998) recorded in changelog #230, and the shipping shim's
    identity constant `0x00003c00` agrees with it read that way: low halfword
    `0x3c00` is fp16 1.0 (the scale) and the high halfword is fp16 0.0 (the
    bias). Under the rejected reading that constant would be a scale of zero.

    int8, NOT IMPLEMENTED HERE: the integer datapath is live on v81 (the standing
    all-zero negative was refuted in #230; the zeros were a record programmed
    with fp16 1.0). Its identity is `out = acc * scale / 512` with `scale =
    512.0`, and the bias is a per-channel int32 rather than an fp16 halfword, so
    an int8 path needs a different record ENCODING even though `CroutonPanel`
    already describes its tile. Device order 027 settles the remaining
    `weight.b` control-word question; nothing here should be given an fp16-only
    shape before it does.
    """
    var s = bitcast[DType.uint16, 1](scale).cast[DType.uint32]()
    var b = bitcast[DType.uint16, 1](bias).cast[DType.uint32]()
    return s | (b << 16)


@always_inline
def hmx_program_record[
    space: AddressSpace
](
    record: HmxRequantRecord[space],
    scale: Scalar[HMX_DTYPE],
    bias: Scalar[HMX_DTYPE],
):
    """Writes one uniform `(scale, bias)` across every output column.

    The record is per-COLUMN and this writes all 32 the same, which is what both
    consumers want: a matmul wants the identity, and attention wants `1/sqrt(D)`
    folded into the QK^T readout so the scale costs no pass of its own
    (`ggml-hexagon` `htp/flash-attn-ops.c:1908-1909` does exactly this, and the
    reason it is free is that the store requants anyway). A per-column record is
    what the layout can express; nothing here needs one yet.
    """
    # `alignment` is stated for the reason TILE_MODEL idiom 7 gives: TileTensor
    # defaults it to 1 off-GPU, and this loop measurably lowered to four `memb`
    # per word before the annotation. The record is a `Scalar[uint32]` region.
    comptime WORD_ALIGN = align_of[Scalar[DType.uint32]]()
    var word = hmx_requant_word(scale, bias)
    for lane in range(HMX_RECORD_LANES):
        record.store[1, alignment=WORD_ALIGN](Coord(0, lane), word)
        record.store[1, alignment=WORD_ALIGN](
            Coord(1, lane), Scalar[DType.uint32](0)
        )


@always_inline
def hmx_program_identity_record[
    space: AddressSpace
](record: HmxRequantRecord[space]):
    """Writes the pass-through record: unit scale, zero bias, zero second vector.

    Re-issued per output tile rather than once per lock. Whether a store or a
    `mxclracc` disturbs the register file is unverified, and a store issued
    before the file is programmed reads back zeros from a MAC whose operands were
    streaming correctly — observed on silicon. One instruction per tile is not
    worth carrying that doubt.
    """
    hmx_program_record(record, Scalar[HMX_DTYPE](1), Scalar[HMX_DTYPE](0))


@fieldwise_init
struct HmxScratch[
    space: AddressSpace = AddressSpace.SHARED, dtype: DType = HMX_DTYPE
](Copyable, Movable):
    """The VTCM residency one HMX output tile needs, as views rather than bases.

    Four allocations the launch owns and the microtile borrows: the activation
    and weight panels the `:deep` loads stream, the 32x32 accumulator readout,
    and the 256-byte scale/bias block the `:after` store requants through.

    Held as a struct rather than four arguments because their lifetime is the
    STRIPE, not the tile: the HMX unit must be locked once and released once
    around every tile a launch computes, since the release clears the
    accumulators (KB `known-limitations/ondevice-hmx-lock-protocol`). A body
    that acquired per tile would lose its accumulation silently, as wrong
    numbers rather than an error.

    Parameters:
        space: Where the four regions live. `AddressSpace.SHARED` is how VTCM
            residency is spelled on this part, and is the default because it is
            what a launch does. The host twin says `GENERIC` instead of casting
            an ordinary heap allocation into a residency the host does not have.
        dtype: The operand type of the three panels. `HMX_DTYPE` is the fp16
            arm; `DType.uint8` is the W8A8 arm, whose panels are half the bytes
            and whose readout is the requanted uint8 tile rather than fp16.
    """

    var act: CroutonPanelOf[Self.dtype, Self.space]
    """`k_tiles` activation croutons, slab `m` / lane `k`."""

    var wgt: CroutonPanelOf[Self.dtype, Self.space]
    """`k_tiles` weight croutons, slab `k` / lane `n`."""

    var out: CroutonPanelOf[Self.dtype, Self.space]
    """One accumulator readout tile, slab `m` / lane `n`."""

    var record: HmxRequantRecord[Self.space]
    """The `[2, 32]` requant record the `:after` store reads out through."""


comptime HMX_STORE_REGION_BYTES = 2048
"""Bytes the accumulator readout store OWNS forward of its destination,
whatever the convert width.

Two tiles at int8 and exactly the tile at fp16, and MEASURED rather than
derived: the vendor's silicon-captured `sat.ub` reference spans 2048 bytes
per output tile (`INT8_DATAFLOW_CONTRACT.md` section 2.2-2.3), and order 132
watched the integer store reach a record placed one int8 tile after `out`
(`HMX_STRIPE_GUARD_CROUTONS`, the stripe arena's statement of this same
hazard). The fp16 `:after.hf` store staying inside its 2048-byte tile is
proven by every green selector-42 row — its record has sat at `out + 2048`
from the start.

Anything the readout's requant depends on — the record above all — must sit
at or past this bound, or pass `N`'s store feeds pass `N + 1`'s requant
garbage. The integer arm reads the record fresh every pass and never
rewrites it, so a clobber there is not a race: it is a deterministic all-zero
readout from the second pass on (order 210 leg 3's `sink=0`)."""


@always_inline
def hmx_scratch_out_bytes[dtype: DType = HMX_DTYPE]() -> Int:
    """The readout slot: one tile, padded to the store's owned region.

    At fp16 the tile IS the region and nothing changes; at int8 the pad puts
    the record outside the bytes the store may write. Slack rather than a
    poisoned guard because nothing here reads a poison back — the stripe
    arena's guard is poisoned exactly because its array thread scores it.
    """
    comptime if HmxTileBytes[dtype] < HMX_STORE_REGION_BYTES:
        return HMX_STORE_REGION_BYTES
    else:
        return HmxTileBytes[dtype]


@always_inline
def hmx_scratch_bytes[dtype: DType = HMX_DTYPE](k_tiles: Int) -> Int:
    """Bytes one stripe's residency needs: two panels, a readout, a record.

    Every region starts on a `HmxTileBytes[dtype]` boundary, which the `:deep`
    load requires of the panels and which the record inherits by following a
    whole number of tiles (the readout slot is a whole number of tiles by
    construction). The caller aligns only the BASE; this size then covers all
    four.
    """
    return (
        2 * k_tiles * HmxTileBytes[dtype]
        + hmx_scratch_out_bytes[dtype]()
        + HMX_RECORD_BYTES
    )


@always_inline
def hmx_scratch_in[
    space: AddressSpace, dtype: DType = HMX_DTYPE
](
    base: UnsafePointer[UInt8, MutUntrackedOrigin, address_space=space],
    k_tiles: Int,
) -> HmxScratch[space, dtype]:
    """Carves `hmx_scratch_bytes(k_tiles)` at `base` into the four views.

    The one place the stripe's residency is laid out, so a caller supplies a
    `HMX_TILE_BYTES`-aligned base with room for `hmx_scratch_bytes` and names no
    offset itself. This is a seam construction — the sole reason a byte address
    is added to in this file — and everything it hands back is a coordinate
    space.

    The record follows the readout at `hmx_scratch_out_bytes`, NOT at the
    tile: the store owns `HMX_STORE_REGION_BYTES` forward of `out`, and a
    record inside that region is the deterministic all-zero readout order 210
    leg 3 measured.
    """
    var panel_bytes = k_tiles * HmxTileBytes[dtype]
    var act = base.bitcast[Scalar[dtype]]()
    var wgt = (base + panel_bytes).bitcast[Scalar[dtype]]()
    var out = (base + 2 * panel_bytes).bitcast[Scalar[dtype]]()
    var rec = (base + 2 * panel_bytes + hmx_scratch_out_bytes[dtype]()).bitcast[
        Scalar[DType.uint32]
    ]()
    return HmxScratch[space, dtype](
        act=hmx_panel[space, dtype](act, k_tiles),
        wgt=hmx_panel[space, dtype](wgt, k_tiles),
        out=hmx_panel[space, dtype](out, 1),
        record=HmxRequantRecord[space](
            ptr=rec,
            layout=row_major(Coord(HMX_RECORD_VECTORS, HMX_RECORD_LANES)),
        ),
    )


@always_inline
def hmx_subpanel[
    dtype: DType, space: AddressSpace, //
](
    panel: CroutonPanelOf[dtype, space], tile0: Int, tiles: Int
) -> CroutonPanelOf[dtype, space]:
    """A view of `tiles` croutons of `panel`, starting at crouton `tile0`.

    The one way a slot inside a multi-slot arena is named. The address comes
    from the panel's own layout rather than from a byte offset the caller
    computes, so a slot's base is a coordinate here exactly as a chunk base is
    in `_hmx_mma_issue`.

    Croutons are `HmxTileBytes[dtype]` apart and the panel's own base is
    `HmxTileBytes[dtype]`-aligned, so every subpanel inherits the `:deep` load's
    alignment requirement rather than needing it re-checked.
    """
    return hmx_panel[space, dtype](
        panel.ptr_at_offset(Coord(tile0, 0, 0, 0)), tiles
    )


comptime HMX_WGT_RING = get_defined_int["SKYE_HMX_WGT_RING", 2]()
"""Weight panels the arena holds in flight: the ring depth (P5 M4.3).

TWO, and two is the CEILING here, not a starting point. The reference solves
this at op-build against the VTCM budget (`ggml-hexagon.cpp:2350-2368`) over a
2-16-deep ring, and order 096 leg A' measured that our shim cannot host one:
it carries **one outstanding transfer per submitter** — one descriptor slot per
launch index, no chaining (KB
`known-limitations/hexagon-user-mode-dma-instructions`) — so a panel must be
awaited before the next is issued and 2 is the degenerate ring the double
buffer already was: one slot in flight, one being read.

That costs nothing the design wanted. Depth exists to hide transfer latency
behind compute, and the transfer for column tile `j + 1` is already issued in
the phase where the array multiplies column tile `j`; the stream itself has
50-75x margin (18.9 GB/step at the measured 31.7 GB/s linear rate, against the
30.3 s of scalar packing it replaces). A deeper ring is therefore a SHIM
extension, priced only if an A/B ever shows the stream near the critical path,
and not a tunable to sweep.

The geometry is parameterised anyway, because that is where a shim extension
would land and because the trade is worth having written down: at `sgl
out-proj`'s `k_tiles = 384` under an 8 MiB grant the closed form gives 8
activation panels / 6 blocks at depth 2, 7 / 7 at depth 3 and 6 / 8 at depth 4.
`CACHE_ROOFLINE_AUDIT.md` §7b calls that trade free now that a weight re-read
costs DMA seconds rather than scalar tens of seconds."""


comptime HMX_WGT_DEFER_AWAIT = get_defined_bool[
    "SKYE_HMX_WGT_DEFER_AWAIT", False
]()
"""Whether the steady-state weight await moves AFTER the step's unpack work.

OFF is the shipping schedule: `_hmx_stage_weight` submits and awaits in one
call, so a feeder eats the whole delivery latency at the await point
(`HMX_PHASE_WGT_AWAIT`, 72.3 G pcycles/render at rev 6 — order 222 leg 1)
while the step's UNPACK phase — its largest at 138 G, ~2x the await pool —
sits AFTER the stall as independent work. ON reorders one feeder step:
submit the ring transfer, unpack the previous tile's readout, then await.
That is the reference's own ordering (`matmul-ops.c:344-405`: push the
prefetch, run the quant task, pop as consumed).

What it does NOT change, each a named constraint: one outstanding transfer
per submitter (the await still precedes the next submit — order 096's shim
limit); the await still runs on the issuing thread (`dmwait`/`dmpoll` act on
that thread's engine); the await still precedes the rendezvous (order 100's
correction — the barrier orders arrival, not delivery); drain-before-exit is
structurally intact (every submit is awaited within its own step,
`UDMA_RING_DESIGN.md` §6). Depth stays `HMX_WGT_RING` and the arena does not
grow — this is the schedule half order 222 leg 1's null routed to after
depth 4 measured WORSE (+1.06 s/render). The prologue (phase 0) keeps its
synchronous await: once per block, deliberately out of scope so the arm is
one variable."""


comptime HMX_FOLD_TRAILING = get_defined_bool[
    "SKYE_HMX_FOLD_TRAILING", False
]()
"""Whether the stripe's trailing readout step folds into the next block's stage.

OFF is the shipping schedule: every A-block ends with a `n_tiles + 1` drain
step whose only work is the last column's unpack — feeders write out, the
array idles, and a full five-way rendezvous is paid for a step with one role
active (RENDEZVOUS_DESIGN.md §S.1). ON merges that unpack into the NEXT
block's phase 0, ahead of its activation pack and ring prologue, so
`HmxStripePlan.steps()` becomes `blocks * (n_tiles + 1) + 1` — one rendezvous
fewer per block boundary. The LAST block keeps its drain step, and at
`blocks == 1` the formula degenerates to `n_tiles + 2`: today's schedule
exactly.

Both orderings the deleted barrier carried are already carried by its
neighbours: the readout RAW (array wrote the last column at phase `n_tiles`;
the rendezvous after that phase orders the merged unpack behind it — the same
one-barrier distance the drain step had) and the pack/prologue WAR (the
array's last read of the act panels and its last weight-slot read are both in
phase `n_tiles`, ordered ahead of the merged step's writes by that same
rendezvous). The readout WAR is ordered by the rendezvous closing the merged
step: the array next writes set `(n_tiles - 1) & 1` no earlier than the
following phase. `steps()` stays a pure plan function, so the order-084
property — every participant computes the same arrival count — holds by
construction."""


comptime HMX_OUT_SETS = 2
"""Accumulator readout sets the arena holds: the readout ring's depth.

TWO today, which is the double buffer P5 M4.2 landed — the array writes set
`col_tile % HMX_OUT_SETS` while the feeders read the other. It is named rather
than spelled `2` in four places because it is ONE quantity: the `out` region's
extent, the per-slot cost in `hmx_stripe_act_slots`' closed form, the carve in
`hmx_stripe_arena_in`, and the modulus `out_tile` indexes with. A depth that
moved in three of those and not the fourth is an arena that overlaps a live
readout, which is a wrong ANSWER with nothing to signal it.

The depth is also what a PAIRED readout needs: one vector store covers two
adjacent column tiles, so the feeders must hold the pair while the array writes
ahead of it, and the aliasing argument wants `HMX_OUT_SETS >= batch + 1`.
Raising it is a separate change with its own arena table; this constant exists
so that change is one line rather than four."""


@fieldwise_init
struct HmxStripeArena[
    space: AddressSpace = AddressSpace.SHARED, dtype: DType = HMX_DTYPE
](Copyable, Movable):
    """Residency for a PIPELINED stripe: an A-block, a double-buffered weight
    panel, and a double-buffered set of accumulator readouts (P5 M4.2).

    `HmxScratch` sizes one output tile because the nest that uses it packs both
    operands immediately before each MMA on one thread. That is the arrangement
    P5 M4.2 replaces: it re-packs the SAME activation panel once per output
    column tile, so the scalar pack — not the array — sets the dispatch's wall
    time. This arena holds `act_slots` activation panels resident across a whole
    N sweep so each is packed once, and doubles the weight and output slots so
    the threads that pack and unpack run concurrently with the thread that holds
    the HMX grant.

    Slot geometry, all in croutons:

    - `act`: `act_slots * k_tiles`, panel `p` at crouton `p * k_tiles`.
    - `wgt`: `HMX_WGT_RING * k_tiles`, slot `b` at crouton `b * k_tiles`.
    - `out`: `HMX_OUT_SETS * act_slots`, PANEL-MAJOR — panel `p` set `b` at
      `p * HMX_OUT_SETS + b`, so one panel's consecutive column tiles are
      adjacent croutons (`out_tile`).

    Parameters:
        space: Where the regions live; `AddressSpace.SHARED` is VTCM on this
            part, and `GENERIC` is what the host twin says.
        dtype: The operand type of the three panel regions. `HMX_DTYPE` is the
            fp16 arm; `DType.uint8` is the W8A8 arm, which needs HALF the VTCM
            for the same `(k_tiles, act_slots)` and therefore admits more
            activation slots at the same budget.
    """

    var act: CroutonPanelOf[Self.dtype, Self.space]
    """`act_slots` activation panels, each `k_tiles` croutons, slab m / lane k.
    """

    var wgt: CroutonPanelOf[Self.dtype, Self.space]
    """`HMX_WGT_RING` weight panels, each `k_tiles` croutons, slab k / lane n."""

    var out: CroutonPanelOf[Self.dtype, Self.space]
    """Two sets of `act_slots` accumulator readouts, slab m / lane n."""

    var guard: CroutonPanelOf[Self.dtype, Self.space]
    """Poisoned slack absorbing a store that writes past the last readout tile.

    Nothing reads it for its contents; the array thread poisons it and then
    counts what survived, so an overrun is reported rather than silently
    landing on the record (`HMX_STRIPE_GUARD_CROUTONS`)."""

    var record: HmxRequantRecord[Self.space]
    """The `[2, 32]` requant record; only the array thread writes it."""

    var act_slots: Int
    """Activation panels resident, which is also the `out` set's stride."""

    var k_tiles: Int
    """Croutons spanning the reduction. One panel's extent."""

    @always_inline
    def act_panel(self, panel: Int) -> CroutonPanelOf[Self.dtype, Self.space]:
        """The activation panel for A-block row-tile `panel`."""
        return hmx_subpanel(self.act, panel * self.k_tiles, self.k_tiles)

    @always_inline
    def wgt_panel(
        self, col_tile: Int
    ) -> CroutonPanelOf[Self.dtype, Self.space]:
        """The ring slot output column tile `col_tile`'s weight panel lives in.

        The caller names the COLUMN TILE, never a slot, so the ring depth is
        spelled once here: a schedule that staged into `j & 1` by hand would
        have to be re-derived at every call site the day the depth moves.
        """
        return hmx_subpanel(
            self.wgt, (col_tile % HMX_WGT_RING) * self.k_tiles, self.k_tiles
        )

    @always_inline
    def out_tile(
        self, buf: Int, panel: Int
    ) -> CroutonPanelOf[Self.dtype, Self.space]:
        """The accumulator readout for row-tile `panel` in slot `buf`.

        PANEL-MAJOR, and the order is the point rather than a preference. Buffer
        -major (`buf * act_slots + panel`, what this was) puts one panel's
        consecutive column tiles `act_slots` croutons apart, so a two-tile view
        of them reads the NEXT PANEL instead of the next column tile — which is
        the shape a vectorised readout wants, since two adjacent column tiles
        are 64 contiguous halfwords of one `C` row and one whole HVX store.
        Panel-major makes those two croutons adjacent, so the same view is the
        pair.

        The two orders are bijections over the same `HMX_OUT_SETS * act_slots`
        croutons, so this costs no bytes and moves no slot: every readout still
        has exactly one home, and which crouton the array writes is the only
        thing that changes. The disjointness the schedule rests on is unmoved
        too, because it is a statement about the SET index and both orders keep
        distinct sets distinct.
        """
        return hmx_subpanel(self.out, panel * HMX_OUT_SETS + buf, 1)

    @always_inline
    def tile_scratch(
        self, col_tile: Int, panel: Int
    ) -> HmxScratch[Self.space, Self.dtype]:
        """The four views one `hmx_mma_panel` call needs, out of this arena.

        Composing `HmxScratch` per tile rather than giving the MMA a second
        entry point is what keeps ONE description of an output tile's operands:
        the pipelined nest and the single-tile nest hand `hmx_mma_panel` the
        same struct, so the instruction sequence, the chunking and therefore
        the accumulation order cannot differ between them.

        Takes the output COLUMN TILE, not a buffer index: the weight ring and
        the readout double buffer have different depths, and only this struct
        knows both. At `HMX_WGT_RING == 2` the two moduli coincide, which is
        why P5 M4.2's `j & 1` call sites read the same either way.
        """
        return HmxScratch[Self.space, Self.dtype](
            act=self.act_panel(panel),
            wgt=self.wgt_panel(col_tile),
            out=self.out_tile(col_tile & 1, panel),
            record=self.record,
        )


comptime HMX_STRIPE_GUARD_CROUTONS = 1
"""Croutons of poisoned padding between the readout region and the record.

THE RECORD USED TO SIT DIRECTLY AFTER `out`, and an accumulator store that
writes past its tile lands on it. Order 132 measured exactly that: the integer
readout came back as a CONSTANT `q = 0` in every slot on 5 of 9 runs, which is
what a requant file of zeros produces for any accumulator, and the images
varied run to run because the clobber races the array thread's once-per-step
record load.

One crouton, because the vendor's `:cm` output region is twice a tile
(`INT8_DATAFLOW_CONTRACT.md` §2.3) and one tile of slack absorbs that case
exactly. It is POISONED rather than merely reserved so the overrun is
MEASURED: `matmul_bt_stripe_hmx_step`'s array thread checks it after the sweep
and reports the damage, which turns a silent corruption into a number.

Why the fp16 arm never saw this: `hmx_mma_panel` rewrites the identity record
on EVERY panel (`:1330`), so it repairs a clobber before the next store reads
it. `hmx_int8_mma_panel` deliberately does not -- the integer record is
per-channel and is loaded once per step -- so the same corruption persists for
the rest of the step. That asymmetry is why order 132 saw fp16 exact and int8
nondeterministic on the SAME skel, in the same window."""


comptime HMX_GUARD_POISON: UInt8 = 0xA5
"""Byte the guard is filled with. Not 0 and not 0xFF, so a store that writes
either a cleared or a saturated tile is still visible as damage."""


@always_inline
def hmx_guard_poison[
    space: AddressSpace, dtype: DType, //
](guard: CroutonPanelOf[dtype, space]):
    """Fills the guard so an overrun can be counted rather than inferred."""
    var p = guard.ptr_at_offset(Coord(0, 0, 0, 0)).bitcast[UInt8]()
    for i in range(HMX_STRIPE_GUARD_CROUTONS * HmxTileBytes[dtype]):
        p[i] = HMX_GUARD_POISON


@always_inline
def hmx_guard_damage[
    space: AddressSpace, dtype: DType, //
](guard: CroutonPanelOf[dtype, space]) -> Int:
    """Bytes of the guard something overwrote. 0 is the healthy answer.

    Nonzero says an accumulator store wrote past its readout tile, which is
    order 132's corruption mechanism; the COUNT is the store's overrun in
    bytes, so it also measures the form's true extent.
    """
    var p = guard.ptr_at_offset(Coord(0, 0, 0, 0)).bitcast[UInt8]()
    var bad = 0
    for i in range(HMX_STRIPE_GUARD_CROUTONS * HmxTileBytes[dtype]):
        if p[i] != HMX_GUARD_POISON:
            bad += 1
    return bad


comptime HMX_ARENA_STAGGER = get_defined_bool["SKYE_HMX_ARENA_STAGGER", False]()
"""Whether the arena pads each region base by one crouton (the stagger arm).

OFF, and off is byte-identical by comptime fold. The audit behind it
(CHAINRATE_PLAYBOOK tiebreaker 2): every region base and stride is a multiple
of 2048 B and panel-to-panel offsets are multiples of `k_tiles * 2048`, so the
concurrent `:deep` streams advance at a relative offset that is congruent to 0
mod every power of two up to 2^15..2^18 at production shapes — the same bank
phase on every access, for any power-of-two banking. One crouton of pad per
region boundary moves the relative offsets off that residue class while
keeping every base crouton-aligned (the `:deep` requirement). Identity-tier:
layout only, no instruction change; the census gate is deltas confined to the
arena arithmetic."""

comptime HMX_ARENA_STAGGER_CROUTONS = 3 if HMX_ARENA_STAGGER else 0
"""One pad crouton at each of the three region boundaries (act|wgt, wgt|out,
out|guard). Appears in BOTH arena-size statements below — the twin-statement
hazard their docstrings name — and in the carve."""


@always_inline
def hmx_stripe_arena_bytes[
    dtype: DType = HMX_DTYPE
](k_tiles: Int, act_slots: Int) -> Int:
    """Bytes a pipelined stripe's residency needs.

    Every region is a whole number of croutons from an aligned base, which is
    the `:deep` load's requirement and what lets `hmx_subpanel` hand out slots
    without re-aligning.
    """
    return (
        act_slots * k_tiles
        + HMX_WGT_RING * k_tiles
        + HMX_OUT_SETS * act_slots
        + HMX_STRIPE_GUARD_CROUTONS
        + HMX_ARENA_STAGGER_CROUTONS
    ) * HmxTileBytes[dtype] + HMX_RECORD_BYTES


@always_inline
def hmx_stripe_act_slots[
    dtype: DType = HMX_DTYPE
](k_tiles: Int, budget: Int, want: Int) -> Int:
    """Activation panels `budget` bytes hold, capped at `want`. 0 means refuse.

    The inverse of `hmx_stripe_arena_bytes` in `act_slots`, closed-form because
    the output slots scale with the panel count too: the fixed cost is the
    weight ring, the guard crouton and the record, and each further activation
    panel costs `k_tiles + HMX_OUT_SETS` croutons — its own `k_tiles`, plus one
    readout in each output set.

    THE FIXED TERM HERE HAS TO TRACK `hmx_stripe_arena_bytes` EXACTLY. It is a
    second statement of the same arithmetic, so a region added there and missed
    here does not fail loudly: this over-reports the slots that fit, the caller
    carves an arena larger than its budget, and the overrun is silent. The
    round-trip assertion in `test_hmx_crouton.mojo` — one byte under a whole
    arena must yield fewer slots — is what holds the two together, and it is
    what caught the guard crouton being added to one and not the other.
    `HMX_OUT_SETS` is the per-slot half of the same hazard: it appears in both
    statements, which is why it is a name rather than a literal.

    `want` is the caller's whole A extent in row tiles, so a shape smaller than
    the budget allocates what it uses rather than the budget.
    """
    var fixed = (
        HMX_WGT_RING * k_tiles
        + HMX_STRIPE_GUARD_CROUTONS
        + HMX_ARENA_STAGGER_CROUTONS
    ) * HmxTileBytes[dtype] + HMX_RECORD_BYTES
    if budget <= fixed:
        return 0
    var slots = (budget - fixed) // (
        (k_tiles + HMX_OUT_SETS) * HmxTileBytes[dtype]
    )
    if slots > want:
        slots = want
    return slots


@always_inline
def hmx_stripe_arena_in[
    space: AddressSpace, dtype: DType = HMX_DTYPE
](
    base: UnsafePointer[UInt8, MutUntrackedOrigin, address_space=space],
    k_tiles: Int,
    act_slots: Int,
) -> HmxStripeArena[space, dtype]:
    """Carves `hmx_stripe_arena_bytes(k_tiles, act_slots)` at `base`.

    The pipelined sibling of `hmx_scratch_in`, and the only other place a byte
    address is added to in this file. The record follows a whole number of
    croutons, so it inherits the alignment the panels were given.
    """
    var act_bytes = act_slots * k_tiles * HmxTileBytes[dtype]
    var wgt_bytes = HMX_WGT_RING * k_tiles * HmxTileBytes[dtype]
    var out_bytes = HMX_OUT_SETS * act_slots * HmxTileBytes[dtype]
    var guard_bytes = HMX_STRIPE_GUARD_CROUTONS * HmxTileBytes[dtype]
    # One crouton of stagger at each region boundary (see HMX_ARENA_STAGGER);
    # zero when the flag is off, so the offsets below fold to the shipping
    # arithmetic byte for byte.
    var pad = HmxTileBytes[dtype] if HMX_ARENA_STAGGER else 0
    var act = base.bitcast[Scalar[dtype]]()
    var wgt = (base + act_bytes + pad).bitcast[Scalar[dtype]]()
    var out = (base + act_bytes + wgt_bytes + 2 * pad).bitcast[Scalar[dtype]]()
    # Between `out` and the record, never after it: an overrunning store runs
    # FORWARD from the last readout tile, so the slack has to be in its path.
    var guard = (base + act_bytes + wgt_bytes + out_bytes + 3 * pad).bitcast[
        Scalar[dtype]
    ]()
    var rec = (
        base + act_bytes + wgt_bytes + out_bytes + 3 * pad + guard_bytes
    ).bitcast[Scalar[DType.uint32]]()
    return HmxStripeArena[space, dtype](
        act=hmx_panel[space, dtype](act, act_slots * k_tiles),
        wgt=hmx_panel[space, dtype](wgt, HMX_WGT_RING * k_tiles),
        out=hmx_panel[space, dtype](out, HMX_OUT_SETS * act_slots),
        guard=hmx_panel[space, dtype](guard, HMX_STRIPE_GUARD_CROUTONS),
        record=HmxRequantRecord[space](
            ptr=rec,
            layout=row_major(Coord(HMX_RECORD_VECTORS, HMX_RECORD_LANES)),
        ),
        act_slots=act_slots,
        k_tiles=k_tiles,
    )


comptime HMX_EMIT_MMA = get_defined_bool["SKYE_HMX_EMIT_MMA", False]()
"""Whether `hmx_mma_panel` issues HMX instructions or models them.

OFF, and the default is the whole point. With it off this file assembles under
the kernels' own feature string, which does NOT carry `+hmx`, so an HMX word
cannot reach the device binary by accident and the next re-sign is provably a
no-op for the matmul. It is also what lets the host twin compile and run the
same source on arm64.

Turning it on is a device-lane decision and needs `+hmx` in the target features.
It is a `-D SKYE_HMX_EMIT_MMA=true` build define rather than an edit here so the
lane's A/B builds both arms from the SAME reviewed source; `build.sh` sets it
and `+hmx` together, since either without the other is a build that cannot say
what it produced. The array has never executed a matmul — only a single
proven-exact 32x32 tile — so the modelled path is not a fallback anyone should
ship: `matmul_bt_stripe_hmx` refuses to route stripes to a modelled MMA at
compile time."""


@always_inline
def _hmx_addr[
    space: AddressSpace, //, dtype: DType
](
    ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin, address_space=space]
) -> UInt32:
    """The 32-bit address an HMX operand register holds.

    Hexagon is a 32-bit part and VTCM is ordinary address space on it, so a
    `SHARED` pointer and a `GENERIC` one are the same number in a register --
    the address space is a Mojo-side residency statement, not an addressing
    mode. Every caller obtains `ptr` from a tile via `ptr_at_offset`, so this
    converts an address the layout computed; it never computes one.
    """
    return UInt32(Int(ptr))


@fieldwise_init
struct HmxOperands[
    space: AddressSpace = AddressSpace.SHARED, dtype: DType = HMX_DTYPE
](Copyable, Movable):
    """One `(activation, weight)` crouton stream of `tiles` croutons each.

    A `:deep` load streams croutons that are CONTIGUOUS, so a stream is the unit
    an accumulator chain can be built out of: everything the array multiplies in
    one go is either one of these or several, and which it is falls out of where
    the operands sit rather than out of any choice. `tiles = 0` is the empty
    stream, so a chain that wants only one segment says so by value.
    """

    var act: CroutonPanelOf[Self.dtype, Self.space]
    """The activation croutons, slab `m` / lane `k`."""

    var wgt: CroutonPanelOf[Self.dtype, Self.space]
    """The weight croutons, slab `k` / lane `n`."""

    var tiles: Int
    """Croutons this stream contributes to the reduction. May be 0."""


@always_inline
def _hmx_mma_open[space: AddressSpace, //](record: HmxRequantRecord[space]):
    """Programs the requant file and clears the accumulators, in that order.

    `bias = mxmem2` comes FIRST because a store issued before the file is
    programmed reads out zeros from a MAC whose operands were streaming
    correctly -- observed on silicon, not deduced. `mxclracc.hf` then clears
    once, OUTSIDE any K loop: every `:deep` load after it accumulates into what
    it cleared, which is what lets a chain span operands from different buffers.
    """
    inlined_assembly[
        "bias = mxmem2($0)",
        NoneType,
        constraints="r,~{memory}",
        has_side_effect=True,
    ](_hmx_addr(record.ptr_at_offset(Coord(0, 0))))

    inlined_assembly[
        "mxclracc.hf",
        NoneType,
        constraints="~{memory}",
        has_side_effect=True,
    ]()


@always_inline
def _hmx_mma_stream[
    space: AddressSpace, dtype: DType, //
](operands: HmxOperands[space, dtype]):
    """Streams one `(activation, weight)` pair into the LIVE accumulators.

    The two loads go in ONE packet. That pairing is the multiply: the array
    multiplies as the operands stream, so there is no separate MAC instruction to
    look for. The braces are load-bearing -- the two must co-issue, and the
    assembler's packet rules allow exactly this pair (see the `+hmx` patch's
    packet legality note).

    The chunk loop is here, not in the caller, because `:deep`'s byte extent is
    16-bit: `HMX_MAX_DEEP_TILES` is a hard encoding cap and a K past it is
    several issues into ONE live accumulator, which only this function can see is
    safe. Chunk bases come from the panels' own tile axis, so a chunk boundary is
    a coordinate rather than an address the loop advances by hand.

    Constraints are `r` rather than pinned registers: pinning is a device of the
    round-trip test, where every expected word must be a constant. Production
    code hands the allocator the choice. `~{memory}` and `has_side_effect`
    because the array reads memory the compiler cannot see and the instructions
    have a required order.
    """
    comptime MAX_TILES = HmxMaxDeepTiles[dtype]
    var t0 = 0
    while t0 < operands.tiles:
        var chunk = operands.tiles - t0
        if chunk > MAX_TILES:
            chunk = MAX_TILES
        inlined_assembly[
            (
                "{\n"
                "    activation.hf = mxmem($0, $2):deep\n"
                "    weight.hf = mxmem($1, $2)\n"
                "}"
            ),
            NoneType,
            constraints="r,r,r,~{memory}",
            has_side_effect=True,
        ](
            _hmx_addr(operands.act.ptr_at_offset(Coord(t0, 0, 0, 0))),
            _hmx_addr(operands.wgt.ptr_at_offset(Coord(t0, 0, 0, 0))),
            UInt32(hmx_deep_range[dtype](chunk)),
        )
        t0 += chunk


@always_inline
def _hmx_mma_store[
    space: AddressSpace, dtype: DType, //
](dst: CroutonPanelOf[dtype, space]):
    """Reads the accumulators back through the requant file into one tile.

    The second operand is `0` at every call site of the reference and in the
    probe that read exact on silicon; its semantics are unpinned, so it is left
    where the evidence puts it rather than derived.
    """
    inlined_assembly[
        "mxmem($0, $1):after.hf = acc",
        NoneType,
        constraints="r,r,~{memory}",
        has_side_effect=True,
    ](_hmx_addr(dst.ptr_at_offset(Coord(0, 0, 0, 0))), UInt32(0))


@always_inline
def _hmx_model_segment[
    space: AddressSpace, dtype: DType, //
](operands: HmxOperands[space, dtype], m: Int, n: Int) -> Scalar[DType.float32]:
    """One output element's share of one stream, in portable arithmetic.

    Which staged element reaches which sum, in the array's own slab and lane
    coordinates rather than in the packers' terms -- so a packer whose axis role
    is transposed fails here.
    """
    comptime SPLIT = HmxSplit[dtype]
    var acc = Scalar[DType.float32](0)
    for t in range(operands.tiles):
        for d in range(HMX_TILE):
            var a = operands.act.load[1](Coord(t, m // SPLIT, d, m % SPLIT))
            var b = operands.wgt.load[1](Coord(t, d // SPLIT, n, d % SPLIT))
            acc += a.cast[DType.float32]() * b.cast[DType.float32]()
    return acc


@always_inline
def _hmx_mma_issue[
    space: AddressSpace, dtype: DType, //
](scratch: HmxScratch[space, dtype], k_tiles: Int):
    """The fp16 MMA, in the five instruction words the array executes.

    Target: Hexagon v81 with `+hmx`. The sequence, and why it is exactly this:

    1. `bias = mxmem2(record)` loads the scale/bias register file the `:after`
       store requants through. It comes FIRST because a store issued before the
       file is programmed reads out zeros from a MAC whose operands were
       streaming correctly -- observed on silicon, not deduced.
    2. `mxclracc.hf` clears the accumulators once, outside the K loop. Every
       `:deep` load below accumulates into what this cleared.
    3. Per chunk, the activation and weight loads in ONE packet. That pairing is
       the multiply: the array multiplies as the operands stream, so there is no
       separate MAC instruction to look for. The braces are load-bearing -- the
       two must co-issue, and the assembler's packet rules allow exactly this
       pair (see the `+hmx` patch's packet legality note).
    4. `mxmem(out, 0):after.hf = acc` reads the accumulators back through the
       requant file. The second operand is `0` at every call site of the
       reference and in the probe that read exact on silicon; its semantics are
       unpinned, so it is left where the evidence puts it rather than derived.

    The chunk loop is here, not in the caller, because `:deep`'s byte extent is
    16-bit: `HMX_MAX_DEEP_TILES` is a hard encoding cap and a K past it is
    several issues into ONE live accumulator, which only this function can see is
    safe. Chunk bases come from the panels' own tile axis, so a chunk boundary is
    a coordinate rather than an address the loop advances by hand.

    Constraints are `r` rather than pinned registers: pinning is a device of the
    round-trip test, where every expected word must be a constant. Production
    code hands the allocator the choice. `~{memory}` on each, and
    `has_side_effect`, because the array reads and writes memory the compiler
    cannot see and the four instructions have a required order.

    Each word is spelled ONCE, in the three helpers above, because the flash loop
    chains streams from different buffers into one accumulator and a second
    spelling of `mxclracc` is exactly the drift that would let the two paths
    disagree about an instruction order silicon has already ruled on.
    """
    _hmx_mma_open(scratch.record)
    _hmx_mma_stream(
        HmxOperands[space, dtype](scratch.act, scratch.wgt, k_tiles)
    )
    _hmx_mma_store(scratch.out)


@always_inline
def _hmx_mma_model[
    space: AddressSpace, dtype: DType, //
](scratch: HmxScratch[space, dtype], k_tiles: Int):
    """What `_hmx_mma_issue` is believed to compute, in portable arithmetic.

    The host cannot execute HMX, so a twin that wants to score the microtile's
    arithmetic needs the multiply expressed in something arm64 runs. This is the
    `conv_im2col.h` split -- one core, the MAC swapped for a model in the twin --
    written as a Mojo comptime branch instead of a `#ifdef`.

    It is a model of the DATA FLOW, and that is the whole of what a twin built on
    it can claim: which staged element reaches which sum, in the array's own slab
    and lane coordinates rather than in the packers' terms, so a packer whose
    axis role is transposed fails here. What it cannot model is the instruction
    stream, the accumulator's internal width, or chunk-boundary behaviour.

    Two deliberate assumptions, both stated because they are unverified: the
    accumulation is fp32 (the array's internal width is not published, and the
    `.hf` mode's only alternative is fixed-point), and the requant is
    `out = acc * scale + bias` per output column, read from the same record the
    `bias = mxmem2` load consumes. `hmx_attn_acc_probe`
    (`hmx_attention.mojo`) is the device measurement that settles the first.
    """
    hmx_mma_chain[emit_hmx=False](
        scratch.out,
        scratch.record,
        HmxOperands[space, dtype](scratch.act, scratch.wgt, k_tiles),
        HmxOperands[space, dtype](scratch.act, scratch.wgt, 0),
    )


@always_inline
def hmx_mma_chain[
    space: AddressSpace, dtype: DType, //, emit_hmx: Bool = HMX_EMIT_MMA
](
    dst: CroutonPanelOf[dtype, space],
    record: HmxRequantRecord[space],
    head: HmxOperands[space, dtype],
    tail: HmxOperands[space, dtype],
):
    """Multiplies TWO streams into one accumulator and reads the tile out once.

    `out = requant(sum over head + sum over tail)`. The second stream is what
    makes the flash loop's rescale free: `ggml-hexagon`'s `hmx_fa_o_update_tile`
    (`htp/hmx-fa-kernels.h:119-192`) issues the pair `(diag(corr), O_old)` at the
    HEAD of the same chain the `(P, V)` pairs run in, so
    `O_new = diag(corr) * O_old + sum_j P_j V_j` costs one extra crouton pair
    and the vector engines never read or write `O`. A matmul passes an empty
    tail and gets the sequence it had.

    The record's CONTENTS are the caller's: `hmx_mma_panel` writes the identity,
    and attention writes `1/sqrt(D)` into the QK^T one. Programming it is not
    done here because a chain may reuse a record several tiles in a row.

    Parameters:
        emit_hmx: Whether to issue the instructions or model them.

    Args:
        dst: The accumulator readout tile.
        record: The `[2, 32]` requant record the `:after` store reads through.
        head: The first stream. Runs first, and the array's summation order is
            the only reason that is worth saying.
        tail: The second stream, or `tiles = 0` for none.
    """
    comptime if emit_hmx:
        _hmx_mma_open(record)
        _hmx_mma_stream(head)
        _hmx_mma_stream(tail)
        _hmx_mma_store(dst)
    else:
        for m in range(HMX_TILE):
            for n in range(HMX_TILE):
                var acc = _hmx_model_segment(head, m, n) + _hmx_model_segment(
                    tail, m, n
                )
                var word = record.load[1](Coord(0, n))
                var scale = bitcast[HMX_DTYPE, 1](word.cast[DType.uint16]())
                var bias = bitcast[HMX_DTYPE, 1](
                    (word >> 16).cast[DType.uint16]()
                )
                dst.store[1](
                    Coord(
                        0,
                        m // HmxSplit[dtype],
                        n,
                        m % HmxSplit[dtype],
                    ),
                    (acc.cast[HMX_DTYPE]() * scale + bias).cast[dtype](),
                )


@always_inline
def _hmx_mma_stream_plain[
    space: AddressSpace, dtype: DType, //
](operands: HmxOperands[space, dtype]):
    """Streams the same pair WITHOUT `:deep`: one crouton pair per packet.

    Target: Hexagon v81 with `+hmx`. Same two operands, same accumulator, same
    bytes -- only the ISSUE COUNT differs, which is the whole reason this exists
    (order 159R leg 2). `_hmx_mma_stream` covers `tiles` croutons in
    `ceildiv(tiles, HmxMaxDeepTiles)` packets; this one takes `tiles` of them,
    each carrying the inclusive byte extent of a single crouton.

    The extent is `hmx_deep_range(1)` rather than a constant of its own: a
    plain load's range is the same 16-bit inclusive byte field the `:deep` load
    takes, so one tile's worth is the tile geometry evaluated at one tile. The
    int8 arm reaches the identical value by the identical route
    (`HMX_INT8_LOAD_RANGE`), and successive load-mpy packets accumulate into the
    LIVE accumulator there (`hmx_int8.mojo`'s stream), so the two forms are
    expected to differ in issue count and not in arithmetic. Expected, not
    proved: the probe reports the readout fold from both arms so a device run
    can say whether they agree.

    THE ACTIVATION WORD HERE IS NEW TO THE fp16 PATH. `activation.hf =
    mxmem(Rs, Rt)` assembles to `92......e4` where the `:deep` form is
    `92......e2`, so `build.sh`'s fp16 word gate carries a member for it -- a
    build that emitted this and nothing else would otherwise satisfy a gate
    that cannot see it (`word_gate_test.py`).
    """
    for t in range(operands.tiles):
        inlined_assembly[
            (
                "{\n"
                "    activation.hf = mxmem($0, $2)\n"
                "    weight.hf = mxmem($1, $2)\n"
                "}"
            ),
            NoneType,
            constraints="r,r,r,~{memory}",
            has_side_effect=True,
        ](
            _hmx_addr(operands.act.ptr_at_offset(Coord(t, 0, 0, 0))),
            _hmx_addr(operands.wgt.ptr_at_offset(Coord(t, 0, 0, 0))),
            UInt32(hmx_deep_range[dtype](1)),
        )


@always_inline
def hmx_port_probe_issues[
    dtype: DType = HMX_DTYPE, //, deep: Bool = True
](tiles: Int) -> Int:
    """Operand-load packets `tiles` croutons take, per form.

    The `:deep` stream is capped by the 16-bit inclusive extent, so it needs one
    packet per `HmxMaxDeepTiles` croutons; the plain stream needs one each. Both
    move the SAME bytes, so this is the only term that separates them and it is
    what a rate difference between the two arms would have to be explained by.

    Parameters:
        dtype: The operand type, which sets the crouton size and so the cap.
        deep: Whether the activation load carries `:deep`.
    """
    comptime if deep:
        return ceildiv(tiles, HmxMaxDeepTiles[dtype])
    else:
        return tiles


@always_inline
def hmx_port_probe_bytes[
    dtype: DType = HMX_DTYPE
](tiles: Int, iters: Int) -> UInt64:
    """Operand bytes `iters` passes over a `tiles`-crouton pair stream move.

    TWO panels per pass -- the array reads an activation crouton and a weight
    crouton for every crouton of depth -- so the denominator is a property of
    the geometry alone and needs no measurement to state. It deliberately
    EXCLUDES the requant record (`HMX_RECORD_BYTES`) and the accumulator
    readout (`HmxTileBytes`) that bracket each pass: they are a fixed cost per
    pass rather than operand traffic, so a rate reported against this is a
    FLOOR for the port, understated by the excluded share.

    THAT SHARE IS `hmx_port_probe_excluded_ppm` AND IT GROWS AS `1/tiles`, which
    is why the caller must not choose `tiles` freely -- see
    `hmx_port_probe_min_tiles`, which is the bound the kernel refuses below.

    64-bit because this part is 32-bit: `Int` here is the device's pointer
    width, and a probe long enough to time can name more bytes than that holds.
    The caller is what decides whether the number still fits the witness word.

    Parameters:
        dtype: The operand type, which sets the crouton size.
    """
    return (
        UInt64(2) * UInt64(tiles) * UInt64(HmxTileBytes[dtype]) * UInt64(iters)
    )


comptime HMX_PORT_PROBE_BUDGET_PPM = 20000
"""Parts per million of the operand stream the denominator may understate: 2%.

The number `hmx_port_probe_bytes` documents itself against, in one place so the
docstring's promise and the kernel's refusal cannot drift. Parts per million
rather than a percentage because the bound is evaluated in integers on a part
with no fp64."""


@always_inline
def hmx_port_probe_excluded_ppm[dtype: DType = HMX_DTYPE](tiles: Int) -> Int:
    """Parts per million by which `hmx_port_probe_bytes` understates a pass.

    `(record + readout) / (2 * tiles * crouton)`, which at fp16 is
    `2304 / (4096 * tiles)` -- 0.5625/tiles, so 7.03% at 8 croutons and 1.94% at
    29. It is a per-PASS ratio and `iters` cancels out of it.

    Parameters:
        dtype: The operand type, which sets the crouton size.
    """
    var excluded = HMX_RECORD_BYTES + HmxTileBytes[dtype]
    var operand = 2 * tiles * HmxTileBytes[dtype]
    return (excluded * 1000000) // operand


@always_inline
def hmx_port_probe_min_tiles[dtype: DType = HMX_DTYPE]() -> Int:
    """Fewest croutons whose excluded share stays inside the documented budget.

    **29 at fp16**, and this is a REFUSAL rather than a caveat because of what
    the understatement does to order 159R sec 7. That table reads a port rate
    below 984 GB/s as a CONTRADICTION -- "either the MAC/byte geometry is wrong
    or the probe is not measuring the operand path", report and stop. At 8
    croutons the denominator understates by 7.03%, which reports a true
    ~1050 GB/s as ~977 and fires that row on an instrument artifact: precisely
    the false negative the row exists to catch, manufactured by the instrument.

    So the kernel refuses the geometry instead of the host correcting the rate.
    A host-side correction would leave the trap live for anyone reading witness
    word 4 directly, and there is no spare word to carry the excluded term in.

    Derived, not written: `tiles >= 100 * excluded / (budget * operand_per
    tile)`, in integers, so a change to the record, the crouton or the budget
    moves the bound and the host test's boundary rows move with it.

    Parameters:
        dtype: The operand type, which sets the crouton size.
    """
    var excluded = HMX_RECORD_BYTES + HmxTileBytes[dtype]
    var operand_per_tile = 2 * HmxTileBytes[dtype]
    return ceildiv(
        excluded * 1000000, HMX_PORT_PROBE_BUDGET_PPM * operand_per_tile
    )


@always_inline
def hmx_port_probe_fold[
    space: AddressSpace, //
](panel: CroutonPanelOf[HMX_DTYPE, space]) -> UInt32:
    """Folds one accumulator readout into the probe's witness word, by SUM.

    The control that says the array wrote at all: elapsed time cannot tell a
    streamed operand from a store that never landed, and this can.

    SUM, NOT XOR, AND THE DIFFERENCE IS THE WHOLE CONTROL. The probe's operands
    vary only along DEPTH, so every element of the 32x32 readout is the same
    sum by construction -- and an XOR fold over an even number of identical
    values is exactly zero. The shipped default was XOR, which means the control
    returned its own FAILURE VALUE on every successful run of order 159R: not a
    control that failed, a control that could never have passed. The lane
    diagnosed it in-window and re-skeled with this fold; all twelve rows read
    non-zero and the rates moved by <= 0.004%, so the measurement stands and only
    the witness was wrong.

    It sums the fp16 BIT PATTERNS rather than the values: this is a checksum,
    not arithmetic, and a checksum must not round. 1024 elements at 0xFFFF is
    67,107,840, so the accumulation cannot wrap the witness word either.

    WHAT ZERO MEANS, now that it means something: every element of the readout
    is `+0.0`. For a probe whose operands are non-zero by construction that is
    the `:after` store not landing, or landing a cleared accumulator -- which is
    exactly what the requant-file ordering bug produced on silicon.

    A control has to be shown capable of failing, and the demonstration belongs
    with the control: `check_port_probe_fold` in the host twin folds an
    unwritten tile to zero, folds a written one to a value it names, and holds
    the rejected XOR fold at zero over that same written tile -- so the defect
    above stays a test rather than a memory.
    """
    comptime ALIGN = align_of[Scalar[HMX_DTYPE]]()
    var fold: UInt32 = 0
    for slab in range(HMX_TILE):
        for lane in range(HMX_TILE):
            var value = panel.load[1, alignment=ALIGN](
                Coord(0, slab // HMX_SPLIT, lane, slab % HMX_SPLIT)
            )
            fold += bitcast[DType.uint16, 1](value).cast[DType.uint32]()
    return fold


@always_inline
def hmx_port_probe_chain[
    space: AddressSpace,
    dtype: DType,
    //,
    deep: Bool = True,
    emit_hmx: Bool = HMX_EMIT_MMA,
](
    dst: CroutonPanelOf[dtype, space],
    record: HmxRequantRecord[space],
    operands: HmxOperands[space, dtype],
):
    """One complete MMA over `operands`, in whichever addressing form is asked.

    The port probe's inner pass, and the ONLY caller of the plain stream. It is
    a separate entry rather than a parameter on `hmx_mma_chain` so that every
    production path -- the matmul stripe, attention -- keeps the function it
    has, unchanged and unre-verified: the two arms share `_hmx_mma_open`,
    `_hmx_mma_store` and (for `deep`) `_hmx_mma_stream` themselves, so each
    instruction word is still spelled exactly once.

    A full open/stream/store rather than a bare stream: an accumulator streamed
    into and never read back is not a shape this silicon has been observed to
    execute, and the readout is what lets the host see the array ran at all.

    Parameters:
        deep: Whether to stream with `:deep` (one packet per 32 croutons) or
            plain (one per crouton).
        emit_hmx: Whether to issue the instructions or model them. The modelled
            arm exists so this file still compiles without `+hmx`; it is far
            too slow to time and a probe must report the build it got rather
            than a number from it.

    Args:
        dst: The accumulator readout tile.
        record: The `[2, 32]` requant record the `:after` store reads through.
        operands: The `(activation, weight)` stream.
    """
    comptime if emit_hmx:
        _hmx_mma_open(record)
        comptime if deep:
            _hmx_mma_stream(operands)
        else:
            _hmx_mma_stream_plain(operands)
        _hmx_mma_store(dst)
    else:
        hmx_mma_chain[emit_hmx=False](
            dst,
            record,
            operands,
            HmxOperands[space, dtype](operands.act, operands.wgt, 0),
        )


@always_inline
def hmx_mma_panel[
    space: AddressSpace, dtype: DType, //, emit_hmx: Bool = HMX_EMIT_MMA
](scratch: HmxScratch[space, dtype], k_tiles: Int):
    """Multiplies the staged panels into `scratch.out` on the matrix array.

    Programs the requant record, then either issues the MMA or models it. The
    record's CONTENTS are written here rather than by the instruction that loads
    them, which is what keeps the encoding (`hmx_requant_word`) a layout Mojo
    owns and testable off-device.

    Parameters:
        emit_hmx: Whether to issue the instructions. Defaults to `HMX_EMIT_MMA`.

    Args:
        scratch: The staged panels, accumulator readout and requant record.
        k_tiles: Crouton tiles spanning the reduction.
    """
    hmx_program_identity_record(scratch.record)
    comptime if emit_hmx:
        _hmx_mma_issue(scratch, k_tiles)
    else:
        _hmx_mma_model(scratch, k_tiles)


# ===--------------------------------------------------------------------------===
# ARRAY-NATIVE WEIGHTS: repacked once on the host, streamed by user-DMA (P5 M4.3)
#
# `hmx_pack_panel[True]` is a transposing SCALAR gather, and the stripe runs it
# once per (A-block, column tile) — 30.3 s of the ~92 s matmul family per step
# (`CACHE_ROOFLINE_AUDIT.md` §7b). The reference backend reorders no weight byte
# at run time at all: the tiles are built on the host at model load and the bulk
# stream moves DDR->VTCM on the DMA engine (`HEXAGON_KERNEL_COMPARE.md` §1.2,
# §1.4). This section is our version of that, and it is deliberately small
# because the transform is the one we already own:
#
#   - `hmx_repack_weight_cpm` IS `hmx_pack_panel[True]`, called once per output
#     column tile into its own contiguous slot. Not a new algorithm, so the
#     packers' existing byte-order / surjectivity / coprime-fill oracles are its
#     oracles too.
#   - The tile ORDER is column-panel-major, the reference's
#     `(ct * n_k_tiles + kt) * tile_size` (`ggml-hexagon.cpp:523`). That is the
#     load-bearing choice: it makes one weight panel ONE CONTIGUOUS LINEAR
#     transfer, so this ships on the user-DMA primitive our tree has proven on
#     silicon (31.7 GB/s, KB
#     `kernels/ondevice-ddr-read-rate-and-the-matmul-latency-stall`) and needs
#     no 2D descriptor, which nothing in this tree has ever written
#     (`new-primitives/npu-tile-pipeline-designed-not-built`).
#   - Nothing about ACCUMULATION moves. The host produced the panel with the
#     same transform the device used to, so the bytes the array reads are
#     identical and the A/B gate is bit-identity rather than a tolerance.
#
# Weights are immutable for the whole render, so the DMA's cache bypass costs no
# flush discipline here. Do not extend this to activations or outputs without
# bringing dirty-range tracking with it (KB
# `known-limitations/ondevice-batched-launch-loses-the-implicit-cache-flush`).
# ===--------------------------------------------------------------------------===

comptime HMX_PHASE_PROBE = get_defined_bool["SKYE_HMX_PHASE_PROBE", False]()
"""Whether the stripe reports where each participant's cycles went, per phase.

OFF, and off is the shipping configuration: with it off no call below is
emitted, so the device object is byte-identical to a build of this branch's
base and the inertness gate is `cmp` rather than a claim.

The instrument answers what no existing one can. `NPU_LAUNCH_TIME` brackets a
whole dispatch; order 096 leg B's brackets sum a phase over a dispatch and yield
shares; order 112's per-leaf brackets taxed the scalar arm +32% to +163% and
were a lane-local transcription of `matmul_bt_stripe_hmx_step` rather than that
function. What the diagnosis needs is WITHIN a dispatch and ON the shipped
schedule: wait-at-barrier vs MMA vs the steps the schedule gives the array no
work in. `npu_phase_probe.h` owns the record and the partition property; this
side owns only where the boundaries are.

Paired with the shim's `SKYE_NPU_PHASE_PROBE` by `build.sh HMX_PHASE_PROBE=1`.
A mismatch is inert rather than a link error: the shim always defines the entry
points and their bodies are what the flag removes."""

comptime HMX_WEIGHTS_NATIVE = get_defined_bool[
    "SKYE_HMX_WEIGHTS_NATIVE", False
]()
"""Whether the stripe reads a REPACKED weight artefact instead of packing one.

OFF, beside the current path rather than replacing it, so the A/B is a single
flag over a single tuple. It is a `-D SKYE_HMX_WEIGHTS_NATIVE=true` build define
(`build.sh HMX_WEIGHTS_NATIVE=1`) because both arms must come from one reviewed
source, exactly as `SKYE_HMX_EMIT_MMA` does.

The define does NOT make the artefact safe on its own: a repacked weight and a
row-major one are the same bytes at the same length and differ only in meaning,
so the mismatch is silently wrong numbers. The loader carries the other half —
`MODULAR_NPU_WEIGHT_NATIVE` and the registry's per-weight layout tag, which
refuse an untagged artefact with the flag on and a tagged one with it off, by
name (`npu_registry_load.h`)."""


comptime HMX_PHASE_OTHER: UInt32 = 0
"""Anything unmarked. A large `OTHER` is a finding, not a rounding error."""

comptime HMX_PHASE_OPEN: UInt32 = 1
"""Grant, VTCM resolve, plan and arena, ahead of the first rendezvous."""

comptime HMX_PHASE_BARRIER: UInt32 = 2
"""Inside `qshimLaunchSyncWait`."""

comptime HMX_PHASE_MMA: UInt32 = 3
"""Array thread: `hmx_mma_panel` over the step's panels."""

comptime HMX_PHASE_IDLE: UInt32 = 4
"""Array thread: a step the schedule gives it no work in.

Deliberately not folded into `BARRIER`. One is what the rendezvous costs and the
other is the schedule declining to feed the array, and the whole diagnosis turns
on telling them apart: `phase == 0` stages a whole A-block on the feeders while
the array returns immediately, and at out-proj that is 6 of 588 steps carrying
two thirds of the feeders' work."""

comptime HMX_PHASE_PACK_ACT: UInt32 = 5
"""Feeder: activation crouton packing."""

comptime HMX_PHASE_STAGE_WGT: UInt32 = 6
"""Feeder: the transposing gather, or the user-DMA submit when native."""

comptime HMX_PHASE_WGT_AWAIT: UInt32 = 7
"""Feeder: waiting for its own staged transfer.

Bracketed nowhere before this, and the leading candidate for the ~70% of a
vector-arm feeder's cycles order 112's tables do not account for."""

comptime HMX_PHASE_UNPACK: UInt32 = 8
"""Feeder: reading accumulator tiles back out to `C`."""

comptime HMX_PHASE_SOFTMAX: UInt32 = 9
"""Feeder: attention's online softmax, seam permutes and diagonal build.

The one phase attention adds that the stripe's buckets cannot express. `PACK_ACT`
and `STAGE_WGT` take the Q/V and K packs because those ARE the two axis rules
under their own names, `UNPACK` takes the `O` readout, and `MMA` takes all three
array products — so this is the only new code, and it is new because folding the
softmax into `OTHER` would put the largest feeder phase in the bucket whose whole
job is to report what nobody bracketed (design §7.3 prices it at ~4.1 ms/launch,
against a ~14 ms array floor)."""


comptime HMX_SM_SPLIT = get_defined_bool["SKYE_HMX_SM_SPLIT", False]()
"""Whether the softmax bracket is sub-divided for the phase probe.

OFF, and off is byte-identical by comptime fold. Meaningful only with
`SKYE_HMX_PHASE_PROBE` — the sub-marks are emitted under `PROBE and SPLIT`,
so a split without the probe folds away instead of calling a shim nobody
opened.

The split adds NO phase ids: `NPU_PHASE_COUNT` is 10 and the shim's mark
guard silently drops ids past the table, so it REUSES ids that are
structurally zero on the feeder threads — `HMX_PHASE_MMA` (3) and
`HMX_PHASE_IDLE` (4) are array-thread labels — the same table-preserving
trick `SKYE_HMX_MMA_SPLIT` plays in the other direction. Under the flag,
on feeder threads only:

  4 (IDLE)    = the seam permutes (`scores_to_rows_at`, the mask pass,
                `probs_from_rows_at`)
  3 (MMA)     = the exp sweep (`SM.exp_nonpos` over the score row and the
                probability store)
  9 (SOFTMAX) = the remainder: max sweep, fold, corr, `(m, l)` stats, diag

The partition property — feeder 4 + 3 + 9 under the split ≈ the unsplit
feeder 9 within ±2% — is the instrument's own device gate; a broken
partition means the instrument is wrong and its numbers are not to be
interpreted."""


@always_inline
def hmx_phase_mark[
    enabled: Bool = HMX_PHASE_PROBE
](thread_idx: Int, phase: UInt32):
    """Ends the participant's open phase and begins `phase`.

    The caller never states a duration, only when one phase ended, so the shim's
    buckets partition the dispatch by construction — the property order 112's
    tables lacked, and the reason its shares and its own absolute cycles
    disagree by 3.3x on the vector arm.

    Parameters:
        enabled: Whether to emit the call at all.

    Args:
        thread_idx: This thread's launch index.
        phase: One of the `HMX_PHASE_*` codes.
    """
    comptime if enabled:
        external_call["qshimPhaseMark", NoneType](UInt32(thread_idx), phase)


@always_inline
def hmx_phase_open[enabled: Bool = HMX_PHASE_PROBE](thread_idx: Int):
    """Opens this participant's first span, once it knows its launch index.

    Args:
        thread_idx: This thread's launch index.
    """
    comptime if enabled:
        external_call["qshimPhaseOpen", NoneType](UInt32(thread_idx))


comptime HMX_MMA_SPLIT = get_defined_bool["SKYE_HMX_MMA_SPLIT", False]()
"""Whether the array thread's MMA bracket is split into its three spans.

OFF, and off is byte-identical by comptime fold. The split is the instrument
CHAINRATE_PLAYBOOK's tiebreaker 3 names as the missing datum: the aggregates
prove ~2-3x of the in-dispatch gap lives INSIDE the MMA bracket, and only a
sub-bracket read separates contention (branch a), an aliasing amplifier
(branch d) and pure attribution (branch e).

The report table is hard-sized to `NPU_PHASE_COUNT = 10` and the shim DROPS
marks with ids past it (`npu_phase_probe.h`, the mark guard), so the split
adds NO new ids: it reuses the feeder-only buckets, which are structurally
zero on the array thread — the same sanctioned reuse the header documents for
attention. On the ARRAY THREAD, under this flag:

  - `STAGE_WGT` (6) = requant-record programming + the `mxmem2` file load,
  - `MMA` (3)       = the `:deep` operand stream (the chain issue proper),
  - `UNPACK` (8)    = the `:after` accumulator readout store.

The report layout, the shim, and the production runner's fetch are all
untouched; only the READER's labels change for the array thread when this
flag was on, and the leg's order says so."""


@always_inline
def hmx_mma_panel_split[
    space: AddressSpace, dtype: DType, //, emit_hmx: Bool = HMX_EMIT_MMA
](scratch: HmxScratch[space, dtype], k_tiles: Int, thread_idx: Int):
    """`hmx_mma_panel` with the array bracket split three ways (HMX_MMA_SPLIT).

    Identical instruction sequence — the three helpers are the same ones,
    spelled once — with a phase mark at each span boundary. The caller's next
    mark closes the readout span, exactly as it closed the whole MMA span
    before.

    Parameters:
        emit_hmx: Whether to issue the instructions. Defaults to `HMX_EMIT_MMA`.

    Args:
        scratch: The staged panels, accumulator readout and requant record.
        k_tiles: Crouton tiles spanning the reduction.
        thread_idx: The array thread's launch index, for the marks.
    """
    hmx_phase_mark(thread_idx, HMX_PHASE_STAGE_WGT)
    hmx_program_identity_record(scratch.record)
    comptime if emit_hmx:
        _hmx_mma_open(scratch.record)
        hmx_phase_mark(thread_idx, HMX_PHASE_MMA)
        _hmx_mma_stream(
            HmxOperands[space, dtype](scratch.act, scratch.wgt, k_tiles)
        )
        hmx_phase_mark(thread_idx, HMX_PHASE_UNPACK)
        _hmx_mma_store(scratch.out)
    else:
        hmx_phase_mark(thread_idx, HMX_PHASE_MMA)
        _hmx_mma_model(scratch, k_tiles)


@always_inline
def hmx_native_weight_croutons(N: Int, K: Int) -> Int:
    """Croutons a column-panel-major artefact holds for a `[N, K]` weight.

    Exactly `N * K` elements, which is what makes the repack an in-place change
    of MEANING rather than of size: the staged file's byte count, the registry
    manifest and the kernel's own `[N, K]` view all still agree, and only the
    layout tag says which order the bytes are in.
    """
    return (N // HMX_TILE) * hmx_k_tiles(K)


@always_inline
def hmx_repack_weight_cpm[
    space: AddressSpace
](dst: CroutonPanel[space], src: HmxRows2D, N: Int, K: Int):
    """Repacks a whole `[N, K]` weight into column-panel-major croutons.

    THE OFFLINE REPACKER. Column panel `ct` occupies croutons
    `[ct * k_tiles, (ct + 1) * k_tiles)` — contiguous, which is the property the
    device side spends.

    `N` and `K` must be whole numbers of crouton tiles; the caller checks that,
    because a weight that is not is one the array would refuse anyway and
    silently repacking it would be the mismatch this whole path is guarded
    against.

    Args:
        dst: `hmx_native_weight_croutons(N, K)` croutons to fill.
        src: The `[N, K]` row-major weight, as the registry staged it.
        N: Output columns of the matmul this weight serves.
        K: The reduction extent.
    """
    var k_tiles = hmx_k_tiles(K)
    for ct in range(N // HMX_TILE):
        hmx_pack_panel[True, space](
            hmx_subpanel(dst, ct * k_tiles, k_tiles),
            src,
            ct * HMX_TILE,
            0,
            k_tiles,
        )


@always_inline
def hmx_native_weight_panel[
    dtype: DType = HMX_DTYPE
](B: HmxRows2D, col_tile: Int, tile0: Int, tiles: Int) -> CroutonPanelOf[
    dtype, AddressSpace.GENERIC
]:
    """`tiles` croutons of column panel `col_tile`, from crouton `tile0` in it.

    The read side of `hmx_repack_weight_cpm`, and the only place the artefact's
    tile order is named on the device. `B` arrives as the kernel's own `[N, K]`
    view because the buffer, its length and its binding are unchanged by the
    repack; the extents are read back off that view so the two orders cannot be
    given different geometries by accident.

    The pointer is re-originated rather than borrowed: `CroutonPanel` is an
    untracked view over a region the caller owns for the whole dispatch, the
    same construction `_matmul_stripe_hmx` makes over the VTCM base. It is one
    seam, and everything downstream of it is a coordinate.
    """
    var k_tiles = hmx_k_tiles(B.layout.shape[1]())
    return hmx_subpanel(
        hmx_panel[AddressSpace.GENERIC, dtype](
            UnsafePointer[Scalar[dtype], MutUntrackedOrigin](
                unsafe_from_address=Int(B.ptr_at_offset(Coord(0, 0)))
            ),
            hmx_native_weight_croutons(
                B.layout.shape[0](), B.layout.shape[1]()
            ),
        ),
        col_tile * k_tiles + tile0,
        tiles,
    )


# ===--------------------------------------------------------------------------===
# ARRAY-NATIVE ACTIVATIONS AND OUTPUTS: row-panel-major croutons (annihilation
# rung 1, `DESIGN-crouton-crossing-annihilation.md`)
#
# The weight artefact above removes the transposing gather; these remove the
# other two transforms every matmul dispatch pays — `PACK_ACT` on its
# activation input and `UNPACK` on its output, 72.08% of feeder time on the
# 190C phase table. The mechanism is the weight one an axis over: a buffer
# whose bytes are already croutons, behind the kernel's own row-major view.
#
#   - The tile order is ROW-panel-major, `(rt * w_tiles + wt)`: all of one
#     32-row block's width tiles, contiguous, then the next row block. That is
#     the activation packer's own sweep order (`_hmx_arm_pack_act` fills panel
#     `p`'s croutons `[t0, t1)` k-tile-contiguous), so one feeder's share is
#     ONE linear range on both sides.
#   - A matmul's output crouton and the next matmul's activation crouton use
#     the SAME axis rule (slab=m, lane=depth, `hmx_unpack_panel`'s
#     `slab_is_depth=False`), and under this tile order the same INDEX rule
#     too: output crouton `(rt * n_tiles + nt)` of `C[M, N]` is exactly
#     activation crouton `(rt * k_tiles + kt)` of the next matmul's
#     `A[M, K = N]`. An elementwise op between two matmuls preserves tile
#     shape and touches nothing else.
#   - The transport is a VECTOR COPY, deliberately not the weight path's DMA:
#     the DMA's cache bypass is safe there only because weights are immutable
#     for the whole render, and an activation was written by the previous
#     dispatch (the section note above; KB
#     `known-limitations/ondevice-batched-launch-loses-the-implicit-cache-flush`).
#     A vector copy reads through the same hierarchy as every other kernel
#     load, so it brings no new coherence class with it.
#   - Nothing about ACCUMULATION moves, so the A/B gate is bit-identity, as
#     the weight arm's is.
#
# Alignment is per row-block rather than per row: a row count that is not a
# multiple of `HMX_TILE` has no crouton form here, and the caller refuses the
# native path by name rather than this file padding on the fly — the same
# discipline `hmx_repack_weight_cpm` states for its extents.
# ===--------------------------------------------------------------------------===


@always_inline
def hmx_native_row_croutons(rows: Int, width: Int) -> Int:
    """Croutons a row-panel-major artefact holds for a `[rows, width]` tensor.

    Exactly `rows * width` elements, the same in-place change of MEANING the
    weight artefact makes: byte count, binding and the kernel's own 2-D view
    all still agree, and only the caller's staging says which order the bytes
    are in.
    """
    return (rows // HMX_TILE) * (width // HMX_TILE)


@always_inline
def hmx_repack_rows_rpm[
    space: AddressSpace
](dst: CroutonPanel[space], src: HmxRows2D, rows: Int, width: Int):
    """Repacks a whole `[rows, width]` tensor into row-panel-major croutons.

    THE OFFLINE REPACKER for the activation/output side, and the reference the
    native arms are scored against: it IS `hmx_pack_panel[False]` — the
    activation axis rule, rows stay rows — called once per 32-row block into
    that block's contiguous crouton range. Not a new transform, so the
    packers' byte-order / surjectivity oracles are its oracles too.

    `rows` and `width` must be whole numbers of crouton tiles; the caller
    checks that, for the reason `hmx_repack_weight_cpm` states.

    Args:
        dst: `hmx_native_row_croutons(rows, width)` croutons to fill.
        src: The `[rows, width]` row-major tensor.
        rows: Row count.
        width: Row length.
    """
    var w_tiles = width // HMX_TILE
    for rt in range(rows // HMX_TILE):
        hmx_pack_panel[False, space](
            hmx_subpanel(dst, rt * w_tiles, w_tiles),
            src,
            rt * HMX_TILE,
            0,
            w_tiles,
        )


@always_inline
def hmx_native_row_panel[
    dtype: DType = HMX_DTYPE
](X: HmxRows2D, row0: Int, col0: Int, tiles: Int) -> CroutonPanelOf[
    dtype, AddressSpace.GENERIC
]:
    """`tiles` croutons of `X`'s row panel `row0 / HMX_TILE`, from `col0`.

    The read (and write) side of `hmx_repack_rows_rpm`, and the only place the
    row-panel-major tile order is named on the device. `X` arrives as the
    kernel's own row-major `(rows, width)` view because the buffer, its length
    and its binding are unchanged by the repack; the width is read back off
    that view so the two orders cannot be given different geometries by
    accident. `row0`/`col0` are ELEMENT coordinates — the packers' own
    argument convention — so a native arm and a packing arm take identical
    call sites.

    The pointer is re-originated rather than borrowed, the
    `hmx_native_weight_panel` seam construction unchanged.
    """
    var w_tiles = X.layout.shape[1]() // HMX_TILE
    return hmx_subpanel(
        hmx_panel[AddressSpace.GENERIC, dtype](
            UnsafePointer[Scalar[dtype], MutUntrackedOrigin](
                unsafe_from_address=Int(X.ptr_at_offset(Coord(0, 0)))
            ),
            hmx_native_row_croutons(X.layout.shape[0](), X.layout.shape[1]()),
        ),
        (row0 // HMX_TILE) * w_tiles + (col0 // HMX_TILE),
        tiles,
    )


@always_inline
def hmx_copy_croutons[
    dtype: DType, dst_space: AddressSpace, src_space: AddressSpace, //
](
    dst: CroutonPanelOf[dtype, dst_space],
    src: CroutonPanelOf[dtype, src_space],
    tiles: Int,
):
    """Moves `tiles` croutons between two panels, order-preserving and linear.

    The native activation/output transport: what `PACK_ACT` and `UNPACK`
    become when the buffer on the far side is already crouton-tiled — a
    straight copy with no gather and no scatter, one aligned vector per
    128-byte slab pair. Written in the panel's own coordinates rather than as
    a byte range so a geometry that disagreed with `hmx_native_row_panel`
    would fail the byte-order oracle here rather than copy the wrong bytes
    correctly.

    Both panels are `HmxTileBytes`-aligned by their own construction rules, so
    the vector accesses state the alignment they in fact have rather than the
    1-byte off-GPU default (KB
    `kernels/tiletensor-off-gpu-alignment-default-degrades-scalar-tails`).
    """
    comptime lanes = HVX_BYTES // size_of[dtype]()
    comptime assert (
        lanes == HMX_TILE * HmxSplit[dtype]
    ), "one copy vector must be exactly one 128-byte slab pair"
    for t in range(tiles):
        for s in range(HMX_TILE // HmxSplit[dtype]):
            dst.store[lanes, alignment=HVX_BYTES](
                Coord(t, s, 0, 0),
                src.load[lanes, alignment=HVX_BYTES](Coord(t, s, 0, 0)),
            )


comptime HMX_XFER_DMA = 0
"""The device transport: the user-DMA engine, submitted then awaited."""

comptime HMX_XFER_COPY = 1
"""The host twin's transport, delivering at submit. Models an engine whose
transfer has already landed when the submit returns."""

comptime HMX_XFER_DEFERRED = 2
"""The host twin's transport, delivering at COMPLETION instead of at submit.

The one that matters. `HMX_XFER_COPY` cannot express the defect order 100
found, because a transport that delivers synchronously is correct whether or
not anything waits for it; this one delivers nothing at submit, poisons the
destination the way an engine mid-write would, and hands over the real bytes
only when the completion is awaited. A stripe that skips the completion
therefore reads the poison and the differential fails — which is what makes the
completion a TESTED requirement rather than a described one."""

comptime HMX_XFER_STALE = 3
"""The host twin's transport modelling a STALE completion read (order 102).

`HMX_XFER_DEFERRED` models an await that reads the truth; this one models an
await that reads a lie. It poisons at submit exactly as the deferred transport
does, and then its completion DELIVERS NOTHING -- which is what an await does
when the `dstate` it polls is a cache line a neighbouring submitter resurrected,
because it exits on the first read believing a transfer that never ran had
finished. Fast, wrong, and indistinguishable from success without telemetry:
the shim's `first=` counter is the bit that separates them, and this is the
transport that fixes what that counter MEANS.

So the pair is the A/B: `HMX_XFER_STALE` is the defect, `HMX_XFER_DEFERRED` is
the same schedule once the poll reads the truth."""


comptime HMX_XFER_POSTED = 4
"""The host twin's transport modelling a POSTED PAYLOAD WRITE (order 103).

The defect orders 100/102/103 kept measuring, once the descriptor side was
exhausted. `HMX_XFER_STALE` models a completion flag read from a stale line;
this models the flag being HONEST and EARLY -- the engine publishes `dstate`
before its last payload writes have landed in VTCM through the fabric, so a
consumer that correctly waits for completion still reads a panel whose tail is
missing.

Modelled as delivering every crouton but the last, because that is the shape a
posted write leaves and because it keeps the control deterministic: on silicon
which bytes are late depends on cache pressure and varies run to run, which is
precisely why the device sees unstable-and-wrong rather than stable-and-wrong.

The fix is not a longer wait -- the flag was not lying. It is
`SKYE_UDMA_VTCM_BYPASS`, which stops routing a VTCM destination through the L2
at all, so completion and residency become the same event."""


comptime HmxXferPoison[dtype: DType] = Scalar[dtype](
    -32768 if dtype.is_floating_point() else 0xA5
)
"""`HMX_XFER_POISON` for an arbitrary operand type.

A byte panel cannot hold -32768, so the byte arm poisons with `0xA5` — outside
every fill the int8 twin uses and, at a per-channel scale, worth tens of output
units, so a single poisoned crouton moves an element well past the tolerance
band a quantized differential is judged on."""

comptime HMX_XFER_POISON = HmxXferPoison[HMX_DTYPE]
"""What an unfinished transfer leaves behind in the twin.

Chosen outside every fill the tests use and large enough that a single poisoned
crouton moves an output element well past a rounding difference: a control that
fires only sometimes is not a control."""


@always_inline
def _hmx_weight_submit[
    dtype: DType, space: AddressSpace, //
](
    dst: CroutonPanelOf[dtype, space],
    src: CroutonPanelOf[dtype, AddressSpace.GENERIC],
    tiles: Int,
    thread_idx: Int,
):
    """Starts one panel's DDR-to-VTCM transfer on this thread's descriptor.

    Target: Hexagon v81. The transfer is LINEAR — a type-0 descriptor, the only
    kind the shim implements and the only kind this part's user-mode subset can
    drive without chaining (KB
    `known-limitations/hexagon-user-mode-dma-instructions`) — which is available
    only because the artefact is column-panel-major.

    Starts and does not wait. The wait is `_hmx_weight_await`, and the two are
    separate functions rather than one because the completion, not the submit,
    is the part order 100 proved load-bearing.

    The shim entry is `npu_kernel.mojo`'s `_qshim_linear_udma_submit` under its
    own name; it is bound again here because the dependency runs
    `npu_kernel -> matmul_kernel -> crouton` and must not run back. The C
    signature is the shim's, and a drift between the two bindings is a link
    failure rather than a silent one.
    """
    var status: UInt32 = 0
    var no_attrs = Optional[UnsafePointer[UInt8, MutAnyOrigin]](None)
    _ = external_call["skye_udma_submit_linear_lo32", UInt32](
        UInt32(thread_idx),
        _hmx_addr(src.ptr_at_offset(Coord(0, 0, 0, 0))),
        UInt32(tiles * HmxTileBytes[dtype]),
        _hmx_addr(dst.ptr_at_offset(Coord(0, 0, 0, 0))),
        no_attrs,
        Int32(1),
        UnsafePointer(to=status).as_unsafe_any_origin(),
    )


@always_inline
def _hmx_weight_await(thread_idx: Int):
    """Waits until THIS submitter's transfer has actually landed.

    `qshimUDmaAwait` is `dmwait` FOLLOWED BY a poll of the descriptor's own
    `dstate`, and the second half is the fix order 100 bought. `dmwait` drains a
    HARDWARE thread; a launch participant is a SOFTWARE thread "QuRT places
    where it likes" (`qshim_qurt.c`'s own launch-index comment), so a submit and
    a wait that straddle a reschedule drain a different engine and return with
    the transfer still in flight. `dstate` lives in memory, is written by the
    engine when the bytes have landed, and therefore survives a migration.

    `UDMA_STATUS.md`'s "Not verified" list named both halves of the gap before
    this path existed: "No case has one thread transferring while another
    computes", and "`dmwait` returns the engine's status word and the shim
    discards it, so a transfer that failed in the engine would look like one
    that ran". The five-thread round trip validates a participant's `dmwait`
    for its OWN later read — strictly weaker than what a stripe needs, where the
    reader is the array thread and the writer is a feeder.

    The return is deliberately dropped. The bound inside the shim is a
    wedged-engine escape rather than a tolerance, and there is no channel to
    report it through that this milestone is allowed to touch — the witness
    protocol is not M4.3's to extend. What detects a failure is the device
    order's Tier A, run more than once.
    """
    _ = external_call["qshimUDmaAwait", UInt32](UInt32(thread_idx))


@always_inline
def _hmx_weight_copy[
    dtype: DType, space: AddressSpace, //
](
    dst: CroutonPanelOf[dtype, space],
    src: CroutonPanelOf[dtype, AddressSpace.GENERIC],
    tiles: Int,
):
    """What the DMA moves, in something a host can execute.

    The twin's transport, and the reason the differential is a real gate: the
    host cannot drive the DMA engine, so it moves the same croutons by hand and
    the test then asserts the stripe's output is bit-identical to the arm that
    packs. Written in the panel's own coordinates rather than as a byte range so
    a geometry that disagreed with `hmx_native_weight_panel` would fail here
    rather than copy the wrong bytes correctly.
    """
    for t in range(tiles):
        for slab in range(HMX_TILE // HmxSplit[dtype]):
            for lane in range(HMX_TILE):
                for sub in range(HmxSplit[dtype]):
                    dst.store[1, alignment=align_of[Scalar[dtype]]()](
                        Coord(t, slab, lane, sub),
                        src.load[1, alignment=align_of[Scalar[dtype]]()](
                            Coord(t, slab, lane, sub)
                        ),
                    )


@always_inline
def _hmx_weight_poison[
    dtype: DType, space: AddressSpace, //
](dst: CroutonPanelOf[dtype, space], tiles: Int):
    """Leaves a slot in the state an unfinished transfer would leave it.

    Only the twin's deferred transport calls this. It models BOTH edges the
    device has: a slot whose new bytes have not arrived yet, and a slot an
    engine has begun writing while something else still reads it. A schedule
    that staged into the slot the array is reading would read poison here.
    """
    for t in range(tiles):
        for slab in range(HMX_TILE // HmxSplit[dtype]):
            for lane in range(HMX_TILE):
                for sub in range(HmxSplit[dtype]):
                    dst.store[1, alignment=align_of[Scalar[dtype]]()](
                        Coord(t, slab, lane, sub), HmxXferPoison[dtype]
                    )


@always_inline
def hmx_stream_weight_submit[
    dtype: DType, space: AddressSpace, //, xfer: Int
](
    dst: CroutonPanelOf[dtype, space],
    src: CroutonPanelOf[dtype, AddressSpace.GENERIC],
    tiles: Int,
    thread_idx: Int,
):
    """Starts the staging of `tiles` croutons into the arena's ring slot.

    Parameters:
        xfer: `HMX_XFER_DMA` on device; the twin's `HMX_XFER_COPY` delivers
            here, and `HMX_XFER_DEFERRED` deliberately does not.

    Args:
        dst: The ring slot's croutons.
        src: The artefact's croutons for the same column panel.
        tiles: Croutons to move.
        thread_idx: The launch index, which selects the DMA descriptor slot.
    """
    comptime if xfer == HMX_XFER_DMA:
        _hmx_weight_submit(dst, src, tiles, thread_idx)
    elif xfer == HMX_XFER_COPY:
        _hmx_weight_copy(dst, src, tiles)
    else:
        _hmx_weight_poison(dst, tiles)


@always_inline
def hmx_stream_weight_await[
    dtype: DType, space: AddressSpace, //, xfer: Int
](
    dst: CroutonPanelOf[dtype, space],
    src: CroutonPanelOf[dtype, AddressSpace.GENERIC],
    tiles: Int,
    thread_idx: Int,
):
    """Blocks until the staging started above has landed in the slot.

    THE FIX ORDER 100 BOUGHT, and the reason it is a separate call: a
    rendezvous orders thread ARRIVAL, and arrival is not delivery. This is what
    orders the delivery, and it runs on the producer, ahead of the rendezvous
    that releases the array — so by the time any thread returns from that
    barrier, every feeder's bytes are in VTCM rather than merely started.

    Producer-side rather than consumer-side, which is not the obvious choice
    and is forced: the descriptor is per SUBMITTER and is reused by that feeder
    for its next panel, so an array thread polling feeder `t`'s `dstate` at the
    moment of consumption cannot tell the transfer it needs from the one that
    feeder has already begun for a later column tile. A per-transfer record the
    consumer could name would be a shim ring the shim does not have. Given the
    rendezvous, producer-completes-then-arrives is exactly equivalent and needs
    nothing new.
    """
    comptime if xfer == HMX_XFER_DMA:
        _hmx_weight_await(thread_idx)
    elif xfer == HMX_XFER_DEFERRED:
        _hmx_weight_copy(dst, src, tiles)
    elif xfer == HMX_XFER_STALE:
        # Deliberately nothing. The await believed a `dstate` it had no right
        # to trust and returned; the slot still holds what the submit left.
        pass
    elif xfer == HMX_XFER_POSTED:
        # Everything but the tail: the completion was honest, the last writes
        # are still in the fabric.
        if tiles > 1:
            _hmx_weight_copy(dst, src, tiles - 1)
