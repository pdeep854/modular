"""The convolution's activation loader for the HMX stripe: a policy, not a kernel.

`matmul_bt_stripe_hmx_step` stages its activation panels through
`hmx_pack_panel[slab_is_depth=False]`, whose scalar body reads

```
row   = row0 + slab      # the output pixel m
depth = col0 + t * HMX_TILE + lane   # the contraction index k
src.load(Coord(row, depth))
```

-- that is, `A[m][k]` with `k` running along the lane. A stride-1 convolution in
the fused k order `k = (r * S + s) * C_in + c` IS `A[m][k]`, gathered from the
NHWC input map instead of from a materialised `[M, K]`. So the array's
convolution arm is the same loader swap M4.6-L1 proved on the HVX microtile,
applied one level up: this file changes where a crouton's elements come from and
nothing about the schedule, the ring, the roles or the readout.

**`C_in % HMX_TILE == 0` is what makes it a copy rather than a gather.** A
crouton's lane run is `HMX_TILE` consecutive depth indices; with the channel
count a multiple of that, a run never straddles a tap boundary, so each
`(slab, tile)` pair is ONE contiguous read from the input map at one halo test.
The fused HVX arm already contracts for `C_in % 64 == 0`
(`mm_conv_fused_supported`), which implies this and also makes `K = R * S * C_in`
a multiple of 64 -- so the conv arm needs none of the K padding the matmul
selector carries and satisfies `matmul_bt_hmx_accepts`'s K bound by
construction.

**The interior/border split is carried from the first line, not discovered.**
M4.6-L2 measured a per-vector halo branch costing the k-loop its zero-overhead
hardware loop (53 -> 115 packets on identical vector work). Here the test is
resolved ONCE per panel (`HmxConvGeom.panel_inside`) and the two bodies are
separate comptime instantiations, so an interior panel carries no branch at all
inside the copy.

How OFTEN that arm runs is decided by the output width, not by the loader, and
it is worth stating because it is not what one would assume. A panel is 32
consecutive output pixels aligned to 32, so with `pad > 0` it is interior only
when it neither wraps an output row nor touches a padded column, which needs
`W_out >= 2 * HMX_TILE + S - 1`. At the decoder's 512-wide stages that is ~87%
of panels; at 128-wide, half; and at the 64-wide mid and up0 stages NO panel
qualifies and the border body runs throughout. That is correct rather than a
defect -- every such panel really does read a padded column -- and
`test_hmx_conv_act.mojo` pins it so it cannot be discovered as a surprise on
silicon.

This file deliberately does NOT import `matmul_kernel`: the dependency runs
matmul -> conv-act -> crouton and must not run back, which is the same rule
`HmxRows2D` is declared under. The one thing that costs is the window
predicate, which `mm_conv_window_inside` also states; `test_hmx_conv_act.mojo`
asserts the two agree over a sweep rather than leaving them to drift.

Target: Hexagon v81 (HMX 32x32 fp16). The activation crouton rule is the same at
any element width, but this arm is fp16 only -- an int8 conv would additionally
have to quantize in the pack, which is `hmx_int8_pack_act`'s business.
"""
from layout.coord import Coord
from layout.tile_layout import row_major
from layout.tile_tensor import TileTensor

from std.memory import AddressSpace
from std.sys import align_of

from hexagon_ondevice.hmx_crouton import (
    HMX_DTYPE,
    HMX_PACK_VEC,
    HMX_PACK_VEC_CONV,
    HMX_PACK_VEC_CONV_EDGE,
    HMX_TILE,
    CroutonPanelOf,
    HmxRows2D,
    HmxSplit,
    hmx_pack_vec_conv_edge_selects,
    hmx_pack_vec_conv_selects,
    hmx_subpanel,
)
from hexagon_ondevice.hvx_permute import (
    HVX_SHUFF_INTERLEAVE,
    hvx_lanes,
    hvx_vshuff,
)


comptime HmxConvMap[o: Origin[mut=True]] = TileTensor[
    HMX_DTYPE, type_of(row_major(Coord(0, 0))), origin=o
]
"""An NHWC input map for one batch index, as `(H, W * C_in)`.

Structurally `HmxRows2D` with the origin left open, for the one reason
`MMImage` is: a struct field may not name `AnyOrigin` directly, so the loader
that holds a map is parameterized on the origin instead of pinning it and
asking every caller to cast. The seam below still takes `HmxRows2D`, which is
what the stripe step already holds."""


struct HmxConvGeom(TrivialRegisterPassable):
    """A stride-1 convolution's geometry, as the activation loader needs it.

    Wider than `MMConv` by the three extents the HVX arm reads off its rank-3
    input map: the stripe's `A` reaches this loader as a rank-2
    `(H, W * C_in)` tile, because that is the operand type the schedule already
    has and adding a second one at the seam would be a new ABI rather than a
    loader policy. The extents therefore travel beside it.

    Register-passable so the stripe step can take one as a defaulted argument
    without a memory operand on the dense path.
    """

    var r_extent: Int
    """Filter height."""

    var s_extent: Int
    """Filter width."""

    var pad_h: Int
    """Rows of zero padding above the input map."""

    var pad_w: Int
    """Columns of zero padding left of the input map."""

    var w_out: Int
    """Output width; `m == h_out * W_out + w_out`."""

    var h_lim: Int
    """Input map height."""

    var w_lim: Int
    """Input map width."""

    var c_extent: Int
    """Input channels; a multiple of `HMX_TILE` on every accepted shape."""

    @always_inline
    def __init__(out self):
        """The inert descriptor a dense dispatch carries.

        `w_out` of zero is what `usable` refuses on, so a conv arm reached with
        a defaulted geometry cannot divide by it.
        """
        self.r_extent = 0
        self.s_extent = 0
        self.pad_h = 0
        self.pad_w = 0
        self.w_out = 0
        self.h_lim = 0
        self.w_lim = 0
        self.c_extent = 0

    @always_inline
    def __init__(
        out self,
        R: Int,
        S: Int,
        pad_h: Int,
        pad_w: Int,
        W_out: Int,
        H: Int,
        W: Int,
        C_in: Int,
    ):
        """Records the window, the padding, and the map's extents.

        Args:
            R: Filter height.
            S: Filter width.
            pad_h: Rows of zero padding above the input map.
            pad_w: Columns of zero padding left of the input map.
            W_out: Output width.
            H: Input map height.
            W: Input map width.
            C_in: Input channels.
        """
        self.r_extent = R
        self.s_extent = S
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.w_out = W_out
        self.h_lim = H
        self.w_lim = W
        self.c_extent = C_in

    @always_inline
    def usable(self) -> Bool:
        """Whether this loader may serve the shape at all.

        The channel bound is the contract that makes a lane run contiguous; the
        rest are the divisions and extents the walk below performs. A caller
        that gets False must keep the arm it already had -- there is no
        degraded conv pack, because a straddling lane run would be wrong
        numbers rather than slow ones.
        """
        return (
            self.c_extent > 0
            and self.c_extent % HMX_TILE == 0
            and self.w_out > 0
            and self.r_extent > 0
            and self.s_extent > 0
            and self.h_lim > 0
            and self.w_lim > 0
        )

    @always_inline
    def k_extent(self) -> Int:
        """`K = R * S * C_in`, the reduction this geometry implies."""
        return self.r_extent * self.s_extent * self.c_extent

    @always_inline
    def panel_inside(self, row0: Int) -> Bool:
        """Whether all `HMX_TILE` pixels from `row0` read fully in bounds.

        The predicate that lets a panel drop the halo test, evaluated once per
        panel instead of once per element. It holds when the first pixel's
        top-left tap is inside the map and the last one's bottom-right tap
        still is, which for a stride-1 window is `0 <= row <= H - R`,
        `0 <= col`, and `col + HMX_TILE - 1 <= W - S`. The column bound also
        rules out a wrap onto the next output row.

        The same statement as `matmul_kernel.mm_conv_window_inside` at
        `rows = HMX_TILE`, restated here because the dependency may not run
        back; `test_hmx_conv_act.mojo` gates the two against each other.

        Args:
            row0: First output pixel of the panel.

        Returns:
            True when no tap of any of the panel's pixels leaves the map.
        """
        var h_out = row0 // self.w_out
        var row = h_out - self.pad_h
        var col = row0 - h_out * self.w_out - self.pad_w
        return (
            0 <= row <= self.h_lim - self.r_extent
            and 0 <= col
            and col + HMX_TILE - 1 <= self.w_lim - self.s_extent
        )


struct HmxConvActLoader[o: Origin[mut=True]](Copyable, Movable):
    """The conv arm's A source: an NHWC map addressed as `A[m][k]`.

    An expert object rather than a free function with eight arguments, for the
    reason `MMIm2colReader` is one: the address decomposition is the whole of
    the arm, it is stated once here, and every caller below the seam names a
    coordinate. Nothing in this file forms an offset by hand -- a tap is
    `(h, w * C_in + c)` on the caller's own tile, so the map's row stride is
    never restated.

    Parameters:
        o: Origin of the input map, carried from whatever the caller built. A
            struct field may not name `AnyOrigin` directly; the seam below
            still takes the `HmxRows2D` the stripe step holds.
    """

    var img: HmxConvMap[Self.o]
    """The input map as `(H, W * C_in)`: a rank-2 view of NHWC, no copy."""

    var geom: HmxConvGeom
    """The window, the padding and the extents."""

    @always_inline
    def __init__(out self, img: HmxConvMap[Self.o], geom: HmxConvGeom):
        """Binds the map and its geometry.

        Args:
            img: The input map as `(H, W * C_in)`.
            geom: The window, padding and extents.
        """
        self.img = img
        self.geom = geom

    @always_inline
    def pack[
        space: AddressSpace, dtype: DType = HMX_DTYPE
    ](
        self,
        dst: CroutonPanelOf[dtype, space],
        row0: Int,
        col0: Int,
        tiles: Int,
    ):
        """Packs `tiles` activation croutons of the convolution's A operand.

        The signature `hmx_pack_panel[slab_is_depth=False]` has, minus the
        source: `row0` is the first output pixel, `col0` the first contraction
        index, and the panel covers `HMX_TILE` pixels by `tiles * HMX_TILE`
        indices. That correspondence is the point -- the stripe step swaps one
        call for the other and learns nothing about convolutions.

        THE SPLIT IS HERE, one test per panel: a panel whose pixels all read in
        bounds runs the branchless body, and only an edge panel pays the halo
        test (M4.6-L2). Under `hmx_pack_vec_conv_selects` the interior body is
        the vector interleave instead — same bytes, the twin gates it — and the
        border body stays scalar under every flag, because its pixels wrap
        output rows and read padding, which the interleave's contiguity proof
        does not cover.

        Parameters:
            space: Where the destination panel lives.
            dtype: The panel's operand type. This arm is fp16 and the default
                says so; the parameter exists because Mojo type-checks a
                generic body eagerly, so the `comptime if` in the stripe's
                `_hmx_arm_pack_act` -- whose `dtype` also spans the byte arm --
                needs a callee that ACCEPTS the other arm's panel type even
                where it can never be instantiated with it. The refusal lives
                at that seam, where the type is concrete.

        Args:
            dst: The panel's croutons, `tiles` of them.
            row0: First output pixel of the panel.
            col0: First contraction index; a multiple of `HMX_TILE`.
            tiles: Croutons to write.
        """
        if self.geom.panel_inside(row0):
            comptime if hmx_pack_vec_conv_selects[
                HMX_PACK_VEC, HMX_PACK_VEC_CONV
            ]():
                self._pack_body_vec(dst, row0, col0, tiles)
            else:
                self._pack_body[halo=False](dst, row0, col0, tiles)
        else:
            comptime if hmx_pack_vec_conv_edge_selects[
                HMX_PACK_VEC, HMX_PACK_VEC_CONV, HMX_PACK_VEC_CONV_EDGE
            ]():
                self._pack_body_vec_edge(dst, row0, col0, tiles)
            else:
                self._pack_body[halo=True](dst, row0, col0, tiles)

    @always_inline
    def _pack_body[
        space: AddressSpace, dtype: DType, //, halo: Bool
    ](
        self,
        dst: CroutonPanelOf[dtype, space],
        row0: Int,
        col0: Int,
        tiles: Int,
    ):
        """`pack` with the halo test resolved, which is the whole difference.

        The walk mirrors `hmx_pack_panel_scalar`'s exactly -- tile, then slab,
        then lane, storing at `(t, slab / SPLIT, lane, slab % SPLIT)` -- so the
        two produce the same bytes for the same logical `A`, which is what the
        twin's byte-identity gate asserts. What differs is only the source
        address:

        - The tap `(r, s)` and channel base `c` come from `col0 + t * HMX_TILE`
          by two divisions per CROUTON, not per element: `C_in % HMX_TILE == 0`
          makes the whole lane run share them.
        - The pixel `(h_out, w_out)` comes from `row0` by ONE division per
          panel, stepped by add-and-carry across the slabs -- the pixel-anchor
          prebake KB `kernels/apple-conv2d-im2col` measured at 1.4x, and the
          reason `MMRowAnchor` exists on the HVX arm.

        A tap outside the map contributes zero, which is the convolution's
        padding and not a guard: the crouton position must still be written, or
        the array would multiply whatever the last dispatch left in VTCM.

        Parameters:
            space: Where the destination panel lives.
            halo: Whether a tap is tested against the map's edges. False is
                only instantiated where `panel_inside` proved it need not be.
        """
        var sc = self.geom.s_extent * self.geom.c_extent
        var h_out = row0 // self.geom.w_out
        var w_pix = row0 - h_out * self.geom.w_out

        for t in range(tiles):
            var depth0 = col0 + t * HMX_TILE
            var r = depth0 // sc
            var rem = depth0 - r * sc
            var s = rem // self.geom.c_extent
            var c0 = rem - s * self.geom.c_extent

            var pix_h = h_out
            var pix_w = w_pix
            for slab in range(HMX_TILE):
                var h = pix_h - self.geom.pad_h + r
                var w = pix_w - self.geom.pad_w + s
                var inside = True
                comptime if halo:
                    inside = (
                        0 <= h < self.geom.h_lim and 0 <= w < self.geom.w_lim
                    )
                var col = w * self.geom.c_extent + c0
                for lane in range(HMX_TILE):
                    var value = Scalar[HMX_DTYPE](0)
                    if inside:
                        value = self.img.load[
                            1, alignment=align_of[Scalar[HMX_DTYPE]]()
                        ](Coord(h, col + lane))
                    # Alignment is stated for the reason `hmx_pack_panel_scalar`
                    # states it: `TileTensor` defaults it to 1 off-GPU, and an
                    # align-1 fp16 store lowers on Hexagon to two `memb` plus a
                    # shift instead of one `memh`. The cast is that packer's too
                    # and is the identity on this arm; see `pack` for why the
                    # panel type is a parameter at all.
                    dst.store[1, alignment=align_of[Scalar[dtype]]()](
                        Coord(
                            t,
                            slab // HmxSplit[dtype],
                            lane,
                            slab % HmxSplit[dtype],
                        ),
                        value.cast[dtype](),
                    )

                pix_w += 1
                if pix_w == self.geom.w_out:
                    pix_w = 0
                    pix_h += 1

    @always_inline
    def _pack_body_vec[
        space: AddressSpace, dtype: DType, //
    ](
        self,
        dst: CroutonPanelOf[dtype, space],
        row0: Int,
        col0: Int,
        tiles: Int,
    ):
        """The interior body issued as the activation-rule interleave.

        Same bytes as `_pack_body[halo=False]`, which is the only claim this
        function makes and the only one the twin's byte-identity gate can
        check. Reachable only where `panel_inside` held, and that predicate is
        what pays for the whole shape: every pixel of the panel sits on ONE
        output row with every tap in bounds, so pixel `slab`'s lane run is one
        contiguous read of the map at row `h` and the pixel axis is a plain
        `C_in`-element stride — `hmx_pack_panel_vec[False]`'s geometry with the
        map's own strides, and the same `vshuff(-2)` silicon voted for
        (orders 112/149/209).

        TWO croutons per step, because one HVX vector is 64 fp16 and a crouton
        row is 32 — the in-panel packer's own reasoning — and here the step has
        one more admission: the pair's 64-element depth run must sit inside one
        tap (`c0 + 64 <= C_in`), or its source is two reads, not one. A pair
        that straddles, and the odd head or tail of a share, take the scalar
        interior body one crouton at a time; with `C_in % 64 == 0` — every VAE
        conv but `conv_in` — a straddle never happens and the tail is at most
        one crouton.

        The register shape is one live vector pair, the activation rule's O(1)
        class, not the butterfly's; the frame census beside the flag's PR is
        the receipt.

        Parameters:
            space: Where the destination panel lives.
            dtype: The panel's operand type; fp16 on this arm. See `pack` for
                why it is a parameter at all.
        """
        comptime assert HmxSplit[dtype] == 2, (
            "the interleave packs one lane word from two slabs; a wider split"
            " (int8) needs a different permute, not a different constant"
        )
        comptime LANES = hvx_lanes[dtype]()
        comptime SLAB_PAIRS = HMX_TILE // HmxSplit[dtype]
        comptime ALIGN = align_of[Scalar[dtype]]()

        var sc = self.geom.s_extent * self.geom.c_extent
        var h_out = row0 // self.geom.w_out
        var w_pix = row0 - h_out * self.geom.w_out

        var t = 0
        while t < tiles:
            var depth0 = col0 + t * HMX_TILE
            var r = depth0 // sc
            var rem = depth0 - r * sc
            var s = rem // self.geom.c_extent
            var c0 = rem - s * self.geom.c_extent
            if t + 1 < tiles and c0 + 2 * HMX_TILE <= self.geom.c_extent:
                var h = h_out - self.geom.pad_h + r
                # Pixel `slab`'s run starts one `C_in` stride after pixel
                # `slab - 1`'s: the interior proof is what makes this an
                # affine address rather than the scalar body's add-and-carry.
                var col_base = (
                    w_pix - self.geom.pad_w + s
                ) * self.geom.c_extent + c0
                comptime for p in range(SLAB_PAIRS):
                    # `vv` carries the even slab and `vu` the odd one, which
                    # is the order the interleave writes them in.
                    var pair = hvx_vshuff[HVX_SHUFF_INTERLEAVE](
                        self.img.load[LANES, alignment=ALIGN](
                            Coord(
                                h,
                                col_base
                                + (HmxSplit[dtype] * p + 1)
                                * self.geom.c_extent,
                            )
                        ).cast[dtype](),
                        self.img.load[LANES, alignment=ALIGN](
                            Coord(
                                h,
                                col_base
                                + HmxSplit[dtype] * p * self.geom.c_extent,
                            )
                        ).cast[dtype](),
                    )
                    dst.store[LANES, alignment=ALIGN](
                        Coord(t, p, 0, 0), pair.slice[LANES, offset=0]()
                    )
                    dst.store[LANES, alignment=ALIGN](
                        Coord(t + 1, p, 0, 0),
                        pair.slice[LANES, offset=LANES](),
                    )
                t += 2
            else:
                self._pack_body[halo=False](
                    hmx_subpanel(dst, t, 1), row0, depth0, 1
                )
                t += 1

    @always_inline
    def _pack_body_vec_edge[
        space: AddressSpace, dtype: DType, //
    ](
        self,
        dst: CroutonPanelOf[dtype, space],
        row0: Int,
        col0: Int,
        tiles: Int,
    ):
        """The border body issued slab-pair-wise: interleave where the pair is
        clean, the scalar walk for exactly the slabs that are not.

        Same bytes as `_pack_body[halo=True]`, which is the only claim this
        function makes and the only one the twin's byte-identity gate checks.
        Order 220 leg 5 is the reason it exists: the interior interleave
        delivers its full class ratio wherever `panel_inside` admits it, so
        the aggregate is Amdahl-capped by border panels — and a pad-1 border
        panel is >= 30 of 32 clean pixels. The interior proof is re-tested
        per SLAB PAIR: both pixels' taps in bounds and the pair's 64-element
        depth run inside one tap. A wrap onto the next output row is not a
        disqualifier — each pixel's lane run is contiguous in its OWN row, so
        the pair's two loads simply carry different rows, which is the one
        address-shape difference from `_pack_body_vec`'s single-row affine
        base. Pixels advance by the scalar body's own add-and-carry.

        A tap outside the map contributes written zeros through the scalar
        walk, never a skipped position (`check_surjective`'s clause).

        Parameters:
            space: Where the destination panel lives.
            dtype: The panel's operand type; fp16 on this arm. See `pack`.
        """
        comptime assert HmxSplit[dtype] == 2, (
            "the interleave packs one lane word from two slabs; a wider split"
            " (int8) needs a different permute, not a different constant"
        )
        comptime LANES = hvx_lanes[dtype]()
        comptime SLAB_PAIRS = HMX_TILE // HmxSplit[dtype]
        comptime ALIGN = align_of[Scalar[dtype]]()

        var sc = self.geom.s_extent * self.geom.c_extent
        var h_out0 = row0 // self.geom.w_out
        var w_pix0 = row0 - h_out0 * self.geom.w_out

        var t = 0
        while t < tiles:
            var depth0 = col0 + t * HMX_TILE
            var r = depth0 // sc
            var rem = depth0 - r * sc
            var s = rem // self.geom.c_extent
            var c0 = rem - s * self.geom.c_extent
            if t + 1 < tiles and c0 + 2 * HMX_TILE <= self.geom.c_extent:
                var pix_h = h_out0
                var pix_w = w_pix0
                comptime for p in range(SLAB_PAIRS):
                    var h_e = pix_h - self.geom.pad_h + r
                    var w_e = pix_w - self.geom.pad_w + s
                    var pw1 = pix_w + 1
                    var ph1 = pix_h
                    if pw1 == self.geom.w_out:
                        pw1 = 0
                        ph1 += 1
                    var h_o = ph1 - self.geom.pad_h + r
                    var w_o = pw1 - self.geom.pad_w + s
                    var even_in = (
                        0 <= h_e < self.geom.h_lim
                        and 0 <= w_e < self.geom.w_lim
                    )
                    var odd_in = (
                        0 <= h_o < self.geom.h_lim
                        and 0 <= w_o < self.geom.w_lim
                    )
                    if even_in and odd_in:
                        # `vv` even, `vu` odd — `_pack_body_vec`'s own order.
                        var pair = hvx_vshuff[HVX_SHUFF_INTERLEAVE](
                            self.img.load[LANES, alignment=ALIGN](
                                Coord(
                                    h_o, w_o * self.geom.c_extent + c0
                                )
                            ).cast[dtype](),
                            self.img.load[LANES, alignment=ALIGN](
                                Coord(
                                    h_e, w_e * self.geom.c_extent + c0
                                )
                            ).cast[dtype](),
                        )
                        dst.store[LANES, alignment=ALIGN](
                            Coord(t, p, 0, 0), pair.slice[LANES, offset=0]()
                        )
                        dst.store[LANES, alignment=ALIGN](
                            Coord(t + 1, p, 0, 0),
                            pair.slice[LANES, offset=LANES](),
                        )
                    else:
                        for du in range(HmxSplit[dtype]):
                            var hh = h_e if du == 0 else h_o
                            var ww = w_e if du == 0 else w_o
                            var inside = even_in if du == 0 else odd_in
                            var slab = HmxSplit[dtype] * p + du
                            for tt in range(2):
                                var colb = (
                                    ww * self.geom.c_extent
                                    + c0
                                    + tt * HMX_TILE
                                )
                                for lane in range(HMX_TILE):
                                    var value = Scalar[HMX_DTYPE](0)
                                    if inside:
                                        value = self.img.load[
                                            1,
                                            alignment=align_of[
                                                Scalar[HMX_DTYPE]
                                            ](),
                                        ](Coord(hh, colb + lane))
                                    dst.store[
                                        1,
                                        alignment=align_of[Scalar[dtype]](),
                                    ](
                                        Coord(
                                            t + tt,
                                            slab // HmxSplit[dtype],
                                            lane,
                                            slab % HmxSplit[dtype],
                                        ),
                                        value.cast[dtype](),
                                    )
                    pix_w = pw1 + 1
                    pix_h = ph1
                    if pix_w == self.geom.w_out:
                        pix_w = 0
                        pix_h += 1
                t += 2
            else:
                self._pack_body[halo=True](
                    hmx_subpanel(dst, t, 1), row0, depth0, 1
                )
                t += 1


@always_inline
def hmx_conv_pack_act[
    space: AddressSpace, dtype: DType = HMX_DTYPE
](
    dst: CroutonPanelOf[dtype, space],
    src: HmxRows2D,
    geom: HmxConvGeom,
    row0: Int,
    col0: Int,
    tiles: Int,
):
    """`hmx_pack_panel[False]` for a convolution: the stripe step's one seam.

    Takes the step's own operand and panel coordinates so the call it replaces
    and this one differ in two arguments and nothing else.

    Parameters:
        space: Where the destination panel lives.
        dtype: The panel's operand type; fp16 on this arm. See `pack`.

    Args:
        dst: The panel's croutons.
        src: The input map as `(H, W * C_in)`.
        geom: The window, padding and extents.
        row0: First output pixel of the panel.
        col0: First contraction index.
        tiles: Croutons to write.
    """
    HmxConvActLoader(src, geom).pack[space, dtype](dst, row0, col0, tiles)
