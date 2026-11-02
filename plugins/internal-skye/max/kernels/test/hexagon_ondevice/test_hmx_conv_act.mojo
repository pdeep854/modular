"""Host validation for the HMX conv activation loader (`hmx_conv_act.mojo`).

The loader's whole claim is an EQUALITY: the croutons it packs out of an NHWC
input map are the croutons the stock activation packer would have produced from
the materialised `[M, K]` im2col of that map. So the gate is that equality, at
the only bar a data movement admits -- bit-exact -- and the reference side is
built here by a direct statement of the im2col

    k = (r * S + s) * C_in + c,   A[m][k] = in[h_out - pad + r][w_out - pad + s][c]

that carries none of the loader's carried state, none of its add-and-carry, and
none of its tile decomposition. Two authors of the same order would still be
one author; what keeps this from being a tautology is that the reference walks
`(m, r, s, c)` and materialises, while the loader walks `(t, slab, lane)` and
never does.

Five levels, plus two controls that must fire:

  1. Byte identity against the stock packer, over interior panels, every edge a
     border panel can sit on, both filter windows the VAE uses, and panel
     origins that are not zero in either axis.
  2. Surjectivity. Every crouton position is written even when the whole panel
     is outside the map, because a position the pack skipped would multiply
     whatever the previous dispatch left in VTCM.
  3. Window predicate agreement. `HmxConvGeom.panel_inside` and
     `matmul_kernel.mm_conv_window_inside` state the same thing in two files
     because the dependency may not run back; the sweep here is what stops them
     drifting.
  4. Arm agreement. The interior and border bodies must be byte-identical
     wherever the window fits -- the split is an optimisation of one rule, not
     a second opinion about it.
  5. THE ARM, end to end. The pipelined stripe run with `act_conv` against a
     nested-loop fp32 conv2d that never says "im2col": exact under an integer
     fill, which is the structural gate on the indexing, and inside a stated
     band under a fill spanning a decade of exponents, which is the numerical
     one. Both weight arms run, since the conv arm changes the ACTIVATION
     loader and composing it with the M4.3 artefact reader is the interaction
     that could still be wrong.

The controls: the interior body run on a BORDER panel must differ from the
border body (or the split is vacuous and level 4 asserts nothing), and a
channel count the contract refuses must be refused (`usable`), since a
straddling lane run is wrong numbers rather than slow ones.

THE FILL IS PART OF THE CONTROL, for the reason `test_hmx_crouton.mojo` records:
periods sharing a factor with the 32-wide tile can fold a mis-assigned axis back
onto the right answer. The periods here are 5, 7, 11 and 13.

Run from this directory (imports the sibling core):
  mojo run test_hmx_conv_act.mojo
"""
from std.math import align_up
from std.memory import AddressSpace, alloc

from layout.coord import Coord
from layout.tile_layout import row_major
from layout.tile_tensor import TileTensor

from hexagon_ondevice.hmx_conv_act import (
    HmxConvActLoader,
    HmxConvGeom,
    hmx_conv_pack_act,
)
from hexagon_ondevice.hmx_crouton import (
    HMX_DTYPE,
    HMX_EMIT_MMA,
    HMX_TILE,
    HMX_UNPACK_ARM_SCALAR_OFF,
    HMX_TILE_BYTES,
    HMX_TILE_ELEMS,
    HMX_XFER_COPY,
    HmxRows2D,
    hmx_k_tiles,
    hmx_native_weight_croutons,
    hmx_pack_panel,
    hmx_pack_vec_conv_edge_selects,
    hmx_pack_vec_conv_selects,
    hmx_panel,
    hmx_repack_weight_cpm,
    hmx_stripe_arena_bytes,
    hmx_stripe_arena_in,
)
from hexagon_ondevice.matmul_kernel import (
    MMConv,
    MMRowAnchor,
    MMTile,
    hmx_stripe_plan,
    matmul_bt_hmx_accepts,
    matmul_bt_stripe_hmx_step,
    mm_conv_window_inside,
)

comptime GEN = AddressSpace.GENERIC

comptime SENTINEL = Scalar[HMX_DTYPE](-1024.0)
"""A value no fill below produces, so an unwritten crouton position shows.

Outside the fill's range rather than merely unlikely: the fill spans -6 to 6 and
the padding writes 0, so any sentinel inside that band reports written positions
as holes -- which is how this check first failed."""


@fieldwise_init
struct ConvCase(Copyable, Movable):
    """One convolution shape, as the sweep names it."""

    var h: Int
    var w: Int
    var c_in: Int
    var r: Int
    var s: Int
    var pad: Int

    def geom(self) -> HmxConvGeom:
        """The loader's descriptor for this spec."""
        return HmxConvGeom(
            self.r,
            self.s,
            self.pad,
            self.pad,
            self.w_out(),
            self.h,
            self.w,
            self.c_in,
        )

    def w_out(self) -> Int:
        """Output width for a stride-1, dilation-1 window."""
        return self.w + 2 * self.pad - self.s + 1

    def h_out(self) -> Int:
        """Output height for a stride-1, dilation-1 window."""
        return self.h + 2 * self.pad - self.r + 1

    def m(self) -> Int:
        """Output pixels, the matmul's `M`."""
        return self.h_out() * self.w_out()

    def k(self) -> Int:
        """The reduction extent, `R * S * C_in`."""
        return self.r * self.s * self.c_in

    def label(self) -> String:
        """A one-line name for a failure report."""
        return (
            "H="
            + String(self.h)
            + " W="
            + String(self.w)
            + " C_in="
            + String(self.c_in)
            + " R="
            + String(self.r)
            + " S="
            + String(self.s)
            + " pad="
            + String(self.pad)
        )


def fill_map(t: HmxRows2D, h: Int, w: Int, c_in: Int):
    """Fills the NHWC map with `(5*ih + 7*iw + 11*c) % 13 - 6`.

    Periods coprime with each other and with `HMX_TILE`, values small integers
    fp16 holds exactly, and both signs present so a dropped tap is visible.
    """
    for ih in range(h):
        for iw in range(w):
            for c in range(c_in):
                t.store[1](
                    Coord(ih, iw * c_in + c),
                    Scalar[HMX_DTYPE]((5 * ih + 7 * iw + 11 * c) % 13 - 6),
                )


def materialize_im2col(dst: HmxRows2D, src: HmxRows2D, spec: ConvCase):
    """The `[M, K]` im2col the loader must agree with, stated directly.

    Walks `(m, r, s, c)` and writes `dst[m][(r*S + s)*C_in + c]`, taking zero
    where the tap leaves the map. No tiles, no croutons, no carried state: this
    is the definition the fused arm is an optimisation of.
    """
    for m in range(spec.m()):
        var h_out = m // spec.w_out()
        var w_out = m - h_out * spec.w_out()
        for r in range(spec.r):
            for s in range(spec.s):
                var ih = h_out - spec.pad + r
                var iw = w_out - spec.pad + s
                var inside = 0 <= ih < spec.h and 0 <= iw < spec.w
                for c in range(spec.c_in):
                    var value = Scalar[HMX_DTYPE](0)
                    if inside:
                        value = src.load[1](Coord(ih, iw * spec.c_in + c))
                    dst.store[1](
                        Coord(m, (r * spec.s + s) * spec.c_in + c), value
                    )


def panel_mismatches(spec: ConvCase, row0: Int, col0: Int, tiles: Int) -> Int:
    """Crouton positions where the fused pack differs from the reference pack.

    The reference is the stock activation packer reading the materialised
    im2col; the subject is the loader reading the map. Both write `tiles`
    croutons, and the comparison is over every element of both panels through a
    flat view, so a mis-sized write is a mismatch rather than a silent pass.
    """
    var map_buf = alloc[Scalar[HMX_DTYPE]](spec.h * spec.w * spec.c_in)
    var im2col_buf = alloc[Scalar[HMX_DTYPE]](spec.m() * spec.k())
    var ref_buf = alloc[Scalar[HMX_DTYPE]](tiles * HMX_TILE_ELEMS)
    var new_buf = alloc[Scalar[HMX_DTYPE]](tiles * HMX_TILE_ELEMS)

    var map2d = HmxRows2D(
        ptr=map_buf.as_unsafe_any_origin(),
        layout=row_major(Coord(spec.h, spec.w * spec.c_in)),
    )
    var im2col = HmxRows2D(
        ptr=im2col_buf.as_unsafe_any_origin(),
        layout=row_major(Coord(spec.m(), spec.k())),
    )
    var ref_flat = HmxRows2D(
        ptr=ref_buf.as_unsafe_any_origin(),
        layout=row_major(Coord(1, tiles * HMX_TILE_ELEMS)),
    )
    var new_flat = HmxRows2D(
        ptr=new_buf.as_unsafe_any_origin(),
        layout=row_major(Coord(1, tiles * HMX_TILE_ELEMS)),
    )
    fill_map(map2d, spec.h, spec.w, spec.c_in)
    materialize_im2col(im2col, map2d, spec)
    for i in range(tiles * HMX_TILE_ELEMS):
        ref_flat.store[1](Coord(0, i), SENTINEL)
        new_flat.store[1](Coord(0, i), SENTINEL)

    hmx_pack_panel[False, GEN](
        hmx_panel[GEN](ref_buf, tiles), im2col, row0, col0, tiles
    )
    hmx_conv_pack_act[GEN](
        hmx_panel[GEN](new_buf, tiles), map2d, spec.geom(), row0, col0, tiles
    )

    var wrong = 0
    for i in range(tiles * HMX_TILE_ELEMS):
        if ref_flat.load[1](Coord(0, i)) != new_flat.load[1](Coord(0, i)):
            wrong += 1

    map_buf.free()
    im2col_buf.free()
    ref_buf.free()
    new_buf.free()
    return wrong


def check_panel(spec: ConvCase, row0: Int, col0: Int, tiles: Int) -> Bool:
    """Asserts one panel of one spec is byte-identical to the reference."""
    var wrong = panel_mismatches(spec, row0, col0, tiles)
    if wrong != 0:
        print(
            "  FAIL pack",
            spec.label(),
            "row0=",
            row0,
            "col0=",
            col0,
            "tiles=",
            tiles,
            "-",
            wrong,
            "of",
            tiles * HMX_TILE_ELEMS,
        )
        return False
    return True


def check_case(spec: ConvCase) -> Bool:
    """Sweeps the panels of one spec: interior, first row, and last row.

    The first output row is where `pad > 0` puts a whole tap row outside the
    map; the last panel of the last row is where the bottom and right edges
    meet, and it is also the one whose pixels wrap onto a following output row,
    which is the spec `panel_inside`'s column bound exists to refuse.
    """
    var ok = True
    var m_tiles = spec.m() // HMX_TILE
    var k_tiles = spec.k() // HMX_TILE
    if m_tiles == 0 or k_tiles == 0:
        print("  FAIL", spec.label(), "- shape has no whole panel")
        return False

    # Panel origins: the first, one in the middle of the map, and the last.
    var rows = [0, (m_tiles // 2) * HMX_TILE, (m_tiles - 1) * HMX_TILE]
    # Column origins: the first crouton, the second tap row where one exists,
    # and the last crouton -- so a k origin that is not zero is exercised in
    # both the `r` and the `s` axis.
    var cols = [0, (k_tiles // 2) * HMX_TILE, (k_tiles - 1) * HMX_TILE]
    for row0 in rows:
        for col0 in cols:
            var tiles = 1
            if col0 + 2 * HMX_TILE <= spec.k():
                tiles = 2
            if not check_panel(spec, row0, col0, tiles):
                ok = False
    # The whole K in one panel, which is what the stripe actually asks for when
    # one feeder owns every crouton.
    if not check_panel(spec, 0, 0, k_tiles):
        ok = False
    if ok:
        print("  ok   pack", spec.label())
    return ok


def check_surjective(spec: ConvCase, row0: Int) -> Bool:
    """Asserts every crouton position is written, even fully outside the map.

    A padding tap is a written zero, not a skipped position: the array reads
    the whole panel whatever the geometry, so a hole is the previous dispatch's
    data multiplied into this one's output.
    """
    var tiles = spec.k() // HMX_TILE
    var map_buf = alloc[Scalar[HMX_DTYPE]](spec.h * spec.w * spec.c_in)
    var dst_buf = alloc[Scalar[HMX_DTYPE]](tiles * HMX_TILE_ELEMS)
    var map2d = HmxRows2D(
        ptr=map_buf.as_unsafe_any_origin(),
        layout=row_major(Coord(spec.h, spec.w * spec.c_in)),
    )
    var flat = HmxRows2D(
        ptr=dst_buf.as_unsafe_any_origin(),
        layout=row_major(Coord(1, tiles * HMX_TILE_ELEMS)),
    )
    fill_map(map2d, spec.h, spec.w, spec.c_in)
    for i in range(tiles * HMX_TILE_ELEMS):
        flat.store[1](Coord(0, i), SENTINEL)

    hmx_conv_pack_act[GEN](
        hmx_panel[GEN](dst_buf, tiles), map2d, spec.geom(), row0, 0, tiles
    )

    var holes = 0
    for i in range(tiles * HMX_TILE_ELEMS):
        if flat.load[1](Coord(0, i)) == SENTINEL:
            holes += 1

    map_buf.free()
    dst_buf.free()
    if holes != 0:
        print(
            "  FAIL surjective",
            spec.label(),
            "row0=",
            row0,
            "-",
            holes,
            "unwritten",
        )
        return False
    print("  ok   surjective", spec.label(), "row0=", row0)
    return True


def check_predicate(spec: ConvCase) -> Bool:
    """Asserts the two statements of the window predicate agree everywhere.

    `HmxConvGeom.panel_inside` restates `mm_conv_window_inside` at
    `rows = HMX_TILE` because the module dependency may not run back to
    `matmul_kernel`. Restatement is the risk this closes: the anchor is built
    here the way `MMIm2colReader.anchor` builds it, and every panel origin of
    the spec is compared.
    """
    var geom = spec.geom()
    var conv = MMConv(spec.r, spec.s, spec.pad, spec.pad, spec.w_out())
    var disagree = 0
    for m_tile in range(spec.m() // HMX_TILE):
        var row0 = m_tile * HMX_TILE
        var h_out = row0 // spec.w_out()
        var anchor = MMRowAnchor(
            row=h_out - spec.pad,
            col=row0 - h_out * spec.w_out() - spec.pad,
            base=0,
        )
        var want = mm_conv_window_inside(anchor, HMX_TILE, conv, spec.h, spec.w)
        if geom.panel_inside(row0) != want:
            disagree += 1
    if disagree != 0:
        print("  FAIL predicate", spec.label(), "-", disagree, "panels differ")
        return False
    print("  ok   predicate", spec.label())
    return True


def arm_mismatches(spec: ConvCase, row0: Int, tiles: Int) -> Int:
    """Positions where the interior body differs from the border body.

    The bare body is deliberately run on a border panel, where its taps walk
    off the map -- that disagreement is what the caller asserts. Without a
    guard band those taps read outside the allocation: mapped-by-luck garbage
    on the ARM hosts, an unmapped page (SIGSEGV) on linux-x86. The
    sentinel-filled band keeps every stray tap inside the allocation and
    reading a value the halo arm never produces, so the arms disagree
    deterministically instead of by allocator accident.
    """
    var geom = spec.geom()
    var row_elems = spec.w * spec.c_in
    var guard = (geom.pad_h + geom.r_extent + 1) * row_elems
    var total = spec.h * row_elems
    var map_buf = alloc[Scalar[HMX_DTYPE]](total + 2 * guard)
    for i in range(total + 2 * guard):
        map_buf[i] = Scalar[HMX_DTYPE](-3)
    var halo_buf = alloc[Scalar[HMX_DTYPE]](tiles * HMX_TILE_ELEMS)
    var bare_buf = alloc[Scalar[HMX_DTYPE]](tiles * HMX_TILE_ELEMS)
    var map2d = HmxRows2D(
        ptr=(map_buf + guard).as_unsafe_any_origin(),
        layout=row_major(Coord(spec.h, row_elems)),
    )
    var halo_flat = HmxRows2D(
        ptr=halo_buf.as_unsafe_any_origin(),
        layout=row_major(Coord(1, tiles * HMX_TILE_ELEMS)),
    )
    var bare_flat = HmxRows2D(
        ptr=bare_buf.as_unsafe_any_origin(),
        layout=row_major(Coord(1, tiles * HMX_TILE_ELEMS)),
    )
    fill_map(map2d, spec.h, spec.w, spec.c_in)

    var loader = HmxConvActLoader(map2d, geom)
    loader._pack_body[halo=True](
        hmx_panel[GEN](halo_buf, tiles), row0, 0, tiles
    )
    loader._pack_body[halo=False](
        hmx_panel[GEN](bare_buf, tiles), row0, 0, tiles
    )

    var wrong = 0
    for i in range(tiles * HMX_TILE_ELEMS):
        if halo_flat.load[1](Coord(0, i)) != bare_flat.load[1](Coord(0, i)):
            wrong += 1

    map_buf.free()
    halo_buf.free()
    bare_buf.free()
    return wrong


def check_arms(spec: ConvCase, expect_interior: Bool) -> Bool:
    """Asserts arm agreement on an interior panel and the control on a border.

    Level 4 and its control in one place, because they are the same measurement
    read at two panel origins: the bodies must agree exactly where the
    predicate says the window fits, and must DISAGREE where it says it does
    not. A split that agreed everywhere would mean the halo test never fires,
    and level 1 would be asserting the padding rule against nothing.

    `expect_interior` pins a fact the panel geometry decides rather than the
    loader: a 32-pixel panel is aligned to 32, so with `pad > 0` it is interior
    only when it does not wrap an output row AND clears both padded columns,
    which needs `W >= 32 + HMX_TILE + S - 1`. Below that -- the decoder's
    64-wide mid and up0 stages included -- EVERY panel is a border panel and
    the branchless body never runs. Asserting the absence keeps that a recorded
    property instead of a silent one.

    Args:
        spec: The convolution shape.
        expect_interior: Whether this geometry admits an interior panel at all.

    Returns:
        Whether every assertion held.
    """
    var geom = spec.geom()
    var tiles = spec.k() // HMX_TILE
    var interior = -1
    var border = -1
    var inside_panels = 0
    var panels = spec.m() // HMX_TILE
    for m_tile in range(panels):
        var row0 = m_tile * HMX_TILE
        if geom.panel_inside(row0):
            inside_panels += 1
            if interior < 0:
                interior = row0
        elif border < 0:
            border = row0

    var ok = True
    if (interior >= 0) != expect_interior:
        print(
            "  FAIL arms",
            spec.label(),
            "- interior panels expected:",
            expect_interior,
            "found:",
            inside_panels,
        )
        return False
    if border < 0:
        print("  FAIL arms", spec.label(), "- no border panel to control on")
        return False

    if interior >= 0:
        var same = arm_mismatches(spec, interior, tiles)
        if same != 0:
            print(
                "  FAIL arms", spec.label(), "- interior panel differs:", same
            )
            ok = False
    var differ = arm_mismatches(spec, border, tiles)
    if differ == 0:
        print(
            "  FAIL arms control",
            spec.label(),
            "- the border panel agreed, so the halo test never fires",
        )
        ok = False
    if ok:
        print(
            "  ok   arms",
            spec.label(),
            "- interior",
            inside_panels,
            "of",
            panels,
            "panels, control differs at",
            differ,
        )
    return ok


def vec_mismatches(spec: ConvCase, row0: Int, col0: Int, tiles: Int) -> Int:
    """Positions where the vector interior body differs from the scalar one.

    Both buffers start at the sentinel, so a crouton position the vector body
    skipped shows as a mismatch against the scalar body's written value rather
    than as an agreement of two holes.
    """
    var map_buf = alloc[Scalar[HMX_DTYPE]](spec.h * spec.w * spec.c_in)
    var vec_buf = alloc[Scalar[HMX_DTYPE]](tiles * HMX_TILE_ELEMS)
    var ref_buf = alloc[Scalar[HMX_DTYPE]](tiles * HMX_TILE_ELEMS)
    var map2d = HmxRows2D(
        ptr=map_buf.as_unsafe_any_origin(),
        layout=row_major(Coord(spec.h, spec.w * spec.c_in)),
    )
    var vec_flat = HmxRows2D(
        ptr=vec_buf.as_unsafe_any_origin(),
        layout=row_major(Coord(1, tiles * HMX_TILE_ELEMS)),
    )
    var ref_flat = HmxRows2D(
        ptr=ref_buf.as_unsafe_any_origin(),
        layout=row_major(Coord(1, tiles * HMX_TILE_ELEMS)),
    )
    fill_map(map2d, spec.h, spec.w, spec.c_in)
    for i in range(tiles * HMX_TILE_ELEMS):
        vec_flat.store[1](Coord(0, i), SENTINEL)
        ref_flat.store[1](Coord(0, i), SENTINEL)

    var loader = HmxConvActLoader(map2d, spec.geom())
    loader._pack_body_vec(hmx_panel[GEN](vec_buf, tiles), row0, col0, tiles)
    loader._pack_body[halo=False](
        hmx_panel[GEN](ref_buf, tiles), row0, col0, tiles
    )

    var wrong = 0
    for i in range(tiles * HMX_TILE_ELEMS):
        if vec_flat.load[1](Coord(0, i)) != ref_flat.load[1](Coord(0, i)):
            wrong += 1

    map_buf.free()
    vec_buf.free()
    ref_buf.free()
    return wrong


def check_vec(spec: ConvCase) -> Bool:
    """Asserts the vector interior body is byte-identical to the scalar one.

    Swept over interior panel origins (the first, a middle one, the last) and
    k ranges shaped like the shares the stripe actually issues: the whole K in
    one call, a two-crouton pair at a non-zero k origin, a one-crouton call
    (which must take the scalar tail whole), and an odd three-crouton share
    (one pair plus a tail). A geometry with no interior panel asserts that
    absence instead — the border body owns those panels under every flag, so
    there is nothing for the vector arm to check.
    """
    var geom = spec.geom()
    var k_tiles = spec.k() // HMX_TILE
    var panels = spec.m() // HMX_TILE
    var interior_rows = [-1, -1, -1]
    var found = 0
    for m_tile in range(panels):
        var row0 = m_tile * HMX_TILE
        if geom.panel_inside(row0):
            if found == 0:
                interior_rows[0] = row0
            interior_rows[1] = row0
            found += 1
    if found == 0:
        print("  ok   vec", spec.label(), "- no interior panel, nothing owed")
        return True
    # A middle interior panel, found by walking again to the count's midpoint.
    var seen = 0
    for m_tile in range(panels):
        var row0 = m_tile * HMX_TILE
        if geom.panel_inside(row0):
            if seen == found // 2:
                interior_rows[2] = row0
                break
            seen += 1

    var ok = True
    for i in range(3):
        var row0 = interior_rows[i]
        if row0 < 0:
            continue
        var origins = [0, (k_tiles // 2) * HMX_TILE, (k_tiles - 1) * HMX_TILE]
        for col0 in origins:
            var span = k_tiles - col0 // HMX_TILE
            var shares = [span, 1, 2, 3]
            for tiles in shares:
                if tiles <= 0 or tiles > span:
                    continue
                var wrong = vec_mismatches(spec, row0, col0, tiles)
                if wrong != 0:
                    print(
                        "  FAIL vec",
                        spec.label(),
                        "row0=",
                        row0,
                        "col0=",
                        col0,
                        "tiles=",
                        tiles,
                        "-",
                        wrong,
                        "of",
                        tiles * HMX_TILE_ELEMS,
                    )
                    ok = False
    if ok:
        print(
            "  ok   vec",
            spec.label(),
            "-",
            found,
            "interior panels, byte-identical",
        )
    return ok


def edge_mismatches(spec: ConvCase, row0: Int, col0: Int, tiles: Int) -> Int:
    """Positions where the edge vector body differs from the scalar halo body.

    Same sentinel discipline as `vec_mismatches`: a skipped position shows as
    a mismatch, never as an agreement of two holes.
    """
    var map_buf = alloc[Scalar[HMX_DTYPE]](spec.h * spec.w * spec.c_in)
    var vec_buf = alloc[Scalar[HMX_DTYPE]](tiles * HMX_TILE_ELEMS)
    var ref_buf = alloc[Scalar[HMX_DTYPE]](tiles * HMX_TILE_ELEMS)
    var map2d = HmxRows2D(
        ptr=map_buf.as_unsafe_any_origin(),
        layout=row_major(Coord(spec.h, spec.w * spec.c_in)),
    )
    var vec_flat = HmxRows2D(
        ptr=vec_buf.as_unsafe_any_origin(),
        layout=row_major(Coord(1, tiles * HMX_TILE_ELEMS)),
    )
    var ref_flat = HmxRows2D(
        ptr=ref_buf.as_unsafe_any_origin(),
        layout=row_major(Coord(1, tiles * HMX_TILE_ELEMS)),
    )
    fill_map(map2d, spec.h, spec.w, spec.c_in)
    for i in range(tiles * HMX_TILE_ELEMS):
        vec_flat.store[1](Coord(0, i), SENTINEL)
        ref_flat.store[1](Coord(0, i), SENTINEL)

    var loader = HmxConvActLoader(map2d, spec.geom())
    loader._pack_body_vec_edge(
        hmx_panel[GEN](vec_buf, tiles), row0, col0, tiles
    )
    loader._pack_body[halo=True](
        hmx_panel[GEN](ref_buf, tiles), row0, col0, tiles
    )

    var wrong = 0
    for i in range(tiles * HMX_TILE_ELEMS):
        if vec_flat.load[1](Coord(0, i)) != ref_flat.load[1](Coord(0, i)):
            wrong += 1

    map_buf.free()
    vec_buf.free()
    ref_buf.free()
    return wrong


def check_vec_edge(spec: ConvCase) -> Bool:
    """Asserts the edge vector body is byte-identical to the scalar halo body.

    Swept over EVERY border panel origin of the spec (a border panel is any
    panel `panel_inside` refuses: the halo rows, the left/right edge panels,
    and the row-wrap panels), with the same k origins and share shapes
    `check_vec` uses. A spec with no border panel asserts that absence.
    """
    var geom = spec.geom()
    var k_tiles = spec.k() // HMX_TILE
    var panels = spec.m() // HMX_TILE
    var ok = True
    var found = 0
    for m_tile in range(panels):
        var row0 = m_tile * HMX_TILE
        if geom.panel_inside(row0):
            continue
        found += 1
        var origins = [0, (k_tiles // 2) * HMX_TILE, (k_tiles - 1) * HMX_TILE]
        for col0 in origins:
            var span = k_tiles - col0 // HMX_TILE
            var shares = [span, 1, 2, 3]
            for tiles in shares:
                if tiles <= 0 or tiles > span:
                    continue
                var wrong = edge_mismatches(spec, row0, col0, tiles)
                if wrong != 0:
                    print(
                        "  FAIL vec-edge",
                        spec.label(),
                        "row0=",
                        row0,
                        "col0=",
                        col0,
                        "tiles=",
                        tiles,
                        "-",
                        wrong,
                        "of",
                        tiles * HMX_TILE_ELEMS,
                    )
                    ok = False
    if found == 0:
        print(
            "  ok   vec-edge", spec.label(), "- no border panel, nothing owed"
        )
        return True
    if ok:
        print(
            "  ok   vec-edge",
            spec.label(),
            "-",
            found,
            "border panels, byte-identical",
        )
    return ok


def check_vec_edge_routing() -> Bool:
    """Pins `hmx_pack_vec_conv_edge_selects`'s whole truth table.

    The edge arm must be strictly narrower than the interior arm: on only
    when all three flags are, so it cannot enter an object whose interior
    interleave is off.
    """
    var bad = 0
    comptime for pv in range(2):
        comptime for pc in range(2):
            comptime for ed in range(2):
                comptime want = pv == 1 and pc == 1 and ed == 1
                comptime if hmx_pack_vec_conv_edge_selects[
                    pv == 1, pc == 1, ed == 1
                ]() != want:
                    bad += 1
    if bad != 0:
        print("  FAIL conv edge routing -", bad, "of 8 rows wrong")
        return False
    print("  ok   conv edge routing: 8/8 rows; the arm needs all three flags")
    return True


def check_vec_routing() -> Bool:
    """Pins `hmx_pack_vec_conv_selects`'s whole truth table.

    The predicate takes the flags as parameters and `HmxConvActLoader.pack`
    consumes this exact predicate, so passing here means the conv arm cannot
    enter an object that did not name both `SKYE_HMX_PACK_VEC` (silicon's vote
    for the interleave class) and `SKYE_HMX_PACK_VEC_CONV`.
    """
    var bad = 0
    comptime if hmx_pack_vec_conv_selects[False, False]():
        bad += 1
    comptime if hmx_pack_vec_conv_selects[False, True]():
        bad += 1
    comptime if hmx_pack_vec_conv_selects[True, False]():
        bad += 1
    comptime if not hmx_pack_vec_conv_selects[True, True]():
        bad += 1
    if bad != 0:
        print("  FAIL conv pack-vec routing -", bad, "of 4 rows wrong")
        return False
    print("  ok   conv pack-vec routing: 4/4 rows; the arm needs both flags")
    return True


comptime Rows2DF32 = TileTensor[
    DType.float32, type_of(row_major(Coord(0, 0))), origin=MutUnsafeAnyOrigin
]
"""The reference's `(M, N)` output. fp32 because the array accumulates there."""


def fill_filter(B: HmxRows2D, n: Int, k: Int, wide: Bool):
    """Fills the `[C_out, K]` b-transposed filter.

    Narrow: small integers, which fp16 and fp32 both hold exactly, so a stripe
    that computes the right sum in the wrong ORDER still lands on the reference
    bit for bit -- which is what makes the narrow arm a structural gate on the
    indexing rather than a numerical one.

    Wide: the same signs over a decade of exponents, so summands of very
    different magnitudes meet in one accumulator and the fold ORDER starts to
    matter. That is the band this arm actually owes, and it is deliberately
    harsher than a decoder's real filters.
    """
    for r in range(n):
        for c in range(k):
            var code = (3 * r + 11 * c) % 7 - 3
            var value = Scalar[HMX_DTYPE](code)
            if wide:
                var shift = (5 * r + 7 * c) % 11
                value = Scalar[HMX_DTYPE](code) * Scalar[HMX_DTYPE](
                    Float32(1 << shift) / 1024.0
                )
            B.store[1](Coord(r, c), value)


def fill_map_wide(t: HmxRows2D, h: Int, w: Int, c_in: Int):
    """`fill_map` spread over a decade of exponents. See `fill_filter`."""
    for ih in range(h):
        for iw in range(w):
            for c in range(c_in):
                var code = (5 * ih + 7 * iw + 11 * c) % 13 - 6
                var shift = (3 * ih + 5 * iw + 7 * c) % 11
                t.store[1](
                    Coord(ih, iw * c_in + c),
                    Scalar[HMX_DTYPE](code)
                    * Scalar[HMX_DTYPE](Float32(1 << shift) / 1024.0),
                )


def conv_reference(
    dst: Rows2DF32, src: HmxRows2D, filt: HmxRows2D, spec: ConvCase, n: Int
):
    """A nested-loop conv2d in fp32: the oracle that never says "im2col".

    Walks output pixel, output channel, then the window, and reads the input
    map at `(h_out - pad + r, w_out - pad + s, c)` directly. It shares no index
    arithmetic with the loader, no k order with the crouton pack, and no fold
    order with the array -- so agreement is evidence about the arm rather than
    about a decomposition both sides assumed.
    """
    for m in range(spec.m()):
        var h_out = m // spec.w_out()
        var w_out = m - h_out * spec.w_out()
        for out_c in range(n):
            var acc = Float32(0)
            for r in range(spec.r):
                for s in range(spec.s):
                    var ih = h_out - spec.pad + r
                    var iw = w_out - spec.pad + s
                    if not (0 <= ih < spec.h and 0 <= iw < spec.w):
                        continue
                    for c in range(spec.c_in):
                        var a = src.load[1](Coord(ih, iw * spec.c_in + c))
                        var b = filt.load[1](
                            Coord(out_c, (r * spec.s + s) * spec.c_in + c)
                        )
                        acc += Float32(a) * Float32(b)
            dst.store[1](Coord(m, out_c), acc)


def stripe_errors[
    weight_native: Bool = False
](spec: ConvCase, n: Int, wide: Bool, act_slots: Int, feeders: Int) -> Float32:
    """Runs the conv arm over the whole output and returns its worst rel error.

    The stripe is driven exactly as `test_hmx_crouton.mojo` drives the dense
    one -- one loop over `plan.steps()`, the array thread and every feeder
    called for each step -- because the point of `act_conv` is that the
    schedule does not know it happened. `emit_hmx` is off, so the host models
    the MMA; what this measures is the arm's arithmetic and its indexing, not
    the silicon's.

    Returns a NEGATIVE value when the run could not be made at all, so a
    refused plan cannot read as a perfect score.
    """
    var k = spec.k()
    var m = spec.m()
    var k_tiles = hmx_k_tiles(k)
    if not matmul_bt_hmx_accepts(m, n, k, 0, n):
        return -1.0

    var map_buf = alloc[Scalar[HMX_DTYPE]](spec.h * spec.w * spec.c_in)
    var b_buf = alloc[Scalar[HMX_DTYPE]](n * k)
    var nat_buf = alloc[Scalar[HMX_DTYPE]](n * k)
    var out_buf = alloc[Scalar[HMX_DTYPE]](m * n)
    var ref_buf = alloc[Float32](m * n)

    var map2d = HmxRows2D(
        ptr=map_buf.as_unsafe_any_origin(),
        layout=row_major(Coord(spec.h, spec.w * spec.c_in)),
    )
    var filt = HmxRows2D(
        ptr=b_buf.as_unsafe_any_origin(), layout=row_major(Coord(n, k))
    )
    var reference = Rows2DF32(
        ptr=ref_buf.as_unsafe_any_origin(), layout=row_major(Coord(m, n))
    )
    if wide:
        fill_map_wide(map2d, spec.h, spec.w, spec.c_in)
    else:
        fill_map(map2d, spec.h, spec.w, spec.c_in)
    fill_filter(filt, n, k, wide)
    conv_reference(reference, map2d, filt, spec, n)
    for i in range(m * n):
        out_buf[i] = SENTINEL

    # The artefact the offline repacker would have produced, built from the
    # same bytes by the same transform the device reads.
    hmx_repack_weight_cpm[GEN](
        hmx_panel[GEN](nat_buf, hmx_native_weight_croutons(n, k)), filt, n, k
    )

    var Cmm = MMTile(
        ptr=out_buf.as_unsafe_any_origin(), layout=row_major(Coord(m, n))
    )
    var Amm = MMTile(
        ptr=map_buf.as_unsafe_any_origin(),
        layout=row_major(Coord(spec.h, spec.w * spec.c_in)),
    )
    var Bmm = MMTile(
        ptr=b_buf.as_unsafe_any_origin(), layout=row_major(Coord(n, k))
    )
    var Bnat = MMTile(
        ptr=nat_buf.as_unsafe_any_origin(), layout=row_major(Coord(n, k))
    )

    var arena_bytes = hmx_stripe_arena_bytes(k_tiles, act_slots)
    var raw = alloc[UInt8](arena_bytes + HMX_TILE_BYTES)
    var base = UnsafePointer[UInt8, MutUntrackedOrigin](
        unsafe_from_address=align_up(Int(raw), HMX_TILE_BYTES)
    )
    var plan = hmx_stripe_plan(m, n, k, arena_bytes)
    var arena = hmx_stripe_arena_in[GEN](base, plan.k_tiles, plan.act_slots)

    comptime step_fn = matmul_bt_stripe_hmx_step[
        HMX_EMIT_MMA, weight_native, HMX_XFER_COPY, True, act_conv=True
    ]
    var Bstep = Bnat if weight_native else Bmm
    var geom = spec.geom()
    # Seeded and threaded exactly as the device sweep does, so the twin
    # exercises the witness accumulator the arm now rides beside.
    var unpack_arm = HMX_UNPACK_ARM_SCALAR_OFF
    for step in range(plan.steps()):
        for t in range(feeders + 1):
            step_fn(
                Cmm, Amm, Bstep, arena, plan, step, t, feeders, unpack_arm, geom
            )

    var worst = Float32(0)
    if plan.act_slots == 0:
        worst = -2.0
    else:
        for row in range(m):
            for col in range(n):
                var got = Float32(Cmm.load[1](Coord(row, col)))
                var want = reference.load[1](Coord(row, col))
                var scale = abs(want)
                if scale < 1.0:
                    scale = 1.0
                var rel = abs(got - want) / scale
                if rel > worst:
                    worst = rel

    map_buf.free()
    b_buf.free()
    nat_buf.free()
    out_buf.free()
    ref_buf.free()
    raw.free()
    return worst


comptime CONV_ARM_BAND = Float32(1.0e-3)
"""The conv arm's stated tolerance against a fp32 nested-loop conv2d.

Measured, not chosen: the array folds over croutons where the reference folds
over `(r, s, c)`, and the readout is fp16, so equality is a band and never a
`memcmp` -- the same thing the fused HVX arm owes against the channel-slowest
order. This bound is against the WIDE fill, whose summands span a decade of
exponents on purpose; the narrow fill is asserted at exactly zero beside it,
which is what says the band is arithmetic rather than a lost or duplicated tap.

The measurement, frozen at first sight: 4.85e-4 worst relative error over every
case here, both weight arms. That is one fp16 ulp (2^-11 = 4.88e-4), which says
the band is the READOUT rather than the reassociation -- the fold order costs
less than the store does. The bound is set at twice it so a genuine ordering
regression fails while fp16's own rounding does not. A production band belongs
to the render rung and is a Tier B order of its own, not this twin's to
state."""


def check_stripe[
    weight_native: Bool = False
](spec: ConvCase, n: Int, label: String) -> Bool:
    """Asserts the conv arm is exact on integers and inside the band on a wide
    fill.

    Two fills and one arm: the narrow one is a structural gate (any dropped,
    duplicated or mis-taken tap moves an integer sum, and integers are exact
    through both folds), the wide one is the numerical one.
    """
    var ok = True
    var exact = stripe_errors[weight_native](spec, n, False, 2, 4)
    if exact != 0.0:
        print(
            "  FAIL stripe",
            label,
            spec.label(),
            "N=",
            n,
            "- integer fill:",
            exact,
        )
        ok = False
    var band = stripe_errors[weight_native](spec, n, True, 2, 4)
    if band < 0.0 or band > CONV_ARM_BAND:
        print(
            "  FAIL stripe", label, spec.label(), "N=", n, "- wide fill:", band
        )
        ok = False
    if ok:
        print(
            "  ok   stripe",
            label,
            spec.label(),
            "N=",
            n,
            "- exact on integers, wide rel err",
            band,
        )
    return ok


def check_usable() -> Bool:
    """Asserts the contract refuses what a lane run cannot span.

    `C_in` not a multiple of the tile is the one geometry that would make a
    lane run straddle a tap, and there is no degraded pack for it -- the caller
    keeps whatever arm it had. A zeroed descriptor, which is what a dense
    dispatch carries, must be refused too or a defaulted geometry could divide
    by a zero output width.
    """
    var ok = True
    if not HmxConvGeom(3, 3, 1, 1, 64, 64, 64, 64).usable():
        print("  FAIL usable - a 64-channel window was refused")
        ok = False
    if HmxConvGeom(3, 3, 1, 1, 64, 64, 64, 32 + 16).usable():
        print("  FAIL usable - a channel count off the tile was accepted")
        ok = False
    if HmxConvGeom().usable():
        print("  FAIL usable - the inert descriptor was accepted")
        ok = False
    if ok:
        print("  ok   usable")
    return ok


def main() raises:
    """Runs every level over the VAE's own window shapes."""
    var failures = 0

    # The decoder's two windows at map sizes small enough to materialise the
    # im2col reference: 3x3 with pad 1 (every resnet and resize conv) and 1x1
    # with no pad (the projections). Channel counts are the two the decode
    # dispatches that satisfy the contract. `W = 96` is the narrowest map that
    # admits an interior panel at all (see `check_arms`), so it is what proves
    # the branchless body; the narrow maps beside it are where a panel wraps an
    # output row, which is the add-and-carry the loader would otherwise never
    # exercise.
    var cases = [
        ConvCase(8, 96, 64, 3, 3, 1),
        ConvCase(16, 16, 64, 3, 3, 1),
        ConvCase(16, 16, 128, 3, 3, 1),
        ConvCase(12, 20, 64, 3, 3, 1),
        ConvCase(16, 16, 64, 1, 1, 0),
        ConvCase(16, 16, 128, 1, 1, 0),
        # A window wider than it is tall, so a tap-row jump that assumed a
        # square filter cannot pass.
        ConvCase(16, 16, 64, 1, 3, 1),
        ConvCase(16, 16, 64, 3, 1, 1),
    ]

    print("conv activation loader:")
    for spec in cases:
        if not check_case(spec):
            failures += 1
        if not check_predicate(spec):
            failures += 1

    print()
    print("padding and surjectivity:")
    for spec in cases:
        if not check_surjective(spec, 0):
            failures += 1

    print()
    print("interior/border split:")
    for spec in cases:
        # A 1x1 window with no padding has no border panel at all, which is
        # correct rather than a gap: the control needs one.
        if spec.pad == 0:
            continue
        if not check_arms(spec, expect_interior=spec.w >= 32 + HMX_TILE):
            failures += 1

    print()
    print("interior vector body:")
    # The vec sweep adds two shapes the base cases do not carry: C_in = 96 is
    # a multiple of the tile but not of two tiles, so the per-pair tap-straddle
    # fallback fires inside an otherwise vectorized share; and the 1x1 window
    # at W = 64 is the narrowest projection map with an interior panel, the
    # shape the decode's shortcut convs take at 256 and 512 wide.
    var vec_cases = [
        ConvCase(8, 96, 64, 3, 3, 1),
        ConvCase(8, 96, 96, 3, 3, 1),
        ConvCase(8, 96, 128, 3, 3, 1),
        ConvCase(4, 64, 64, 1, 1, 0),
        ConvCase(16, 16, 64, 3, 3, 1),
    ]
    for spec in vec_cases:
        if not check_vec(spec):
            failures += 1

    print()
    print("border vector body (slab-pair admission):")
    # The edge sweep wants every border class the decode dispatches: the halo
    # rows and edge panels of the wide maps, the all-border 64-wide class
    # (16x16 here), and the row-wrap add-and-carry map. C_in = 96 exercises
    # the tap-straddle scalar fallback inside a border share.
    var edge_cases = [
        ConvCase(8, 96, 64, 3, 3, 1),
        ConvCase(8, 96, 96, 3, 3, 1),
        ConvCase(8, 96, 128, 3, 3, 1),
        ConvCase(16, 16, 64, 3, 3, 1),
        ConvCase(16, 16, 128, 3, 3, 1),
        ConvCase(12, 20, 64, 3, 3, 1),
        ConvCase(16, 16, 64, 1, 3, 1),
        ConvCase(16, 16, 64, 3, 1, 1),
    ]
    for spec in edge_cases:
        if not check_vec_edge(spec):
            failures += 1
    if not check_vec_edge_routing():
        failures += 1

    print()
    print("the arm, against a nested-loop conv2d:")
    # `M`, `N` and `K` all reach the array's predicate, and both weight arms
    # run: the conv arm changes the ACTIVATION loader, so composing it with the
    # M4.3 artefact reader is the one interaction that could still be wrong.
    var arm_cases = [
        ConvCase(8, 96, 64, 3, 3, 1),
        ConvCase(16, 16, 64, 3, 3, 1),
        ConvCase(16, 16, 128, 1, 1, 0),
    ]
    for spec in arm_cases:
        if not check_stripe(spec, 32, "gathered weight"):
            failures += 1
    if not check_stripe[weight_native=True](arm_cases[0], 32, "crouton weight"):
        failures += 1

    print()
    print("contract:")
    if not check_usable():
        failures += 1
    if not check_vec_routing():
        failures += 1

    print()
    if failures == 0:
        print("ALL HMX CONV ACT TESTS PASSED")
    else:
        raise Error(String(failures) + " conv-act check(s) FAILED")
