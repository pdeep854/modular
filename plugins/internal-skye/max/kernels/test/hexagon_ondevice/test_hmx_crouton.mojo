"""Host validation for the HMX crouton packers (`hmx_crouton.mojo`).

The packing is the part of an HMX matmul that no numerical test of the whole
kernel can localize: a mis-assigned axis produces a wrong ANSWER, never a crash,
and the vendor archive's only real content is exactly this convention. So it is
pinned here directly, device-free, at three levels:

  1. Byte order, per axis role. A tile is filled with unique ids, packed, and
     read back through a FLAT view of the same allocation, so the check asserts
     which id landed at which crouton offset — the KB rule
     `(s / 2) * 64 + l * 2 + (s % 2)` in elements — rather than asserting that
     some self-consistent permutation happened. This is the check that catches a
     transposed axis assignment; a round trip alone cannot, because a round trip
     is clean under ANY bijection.
  2. Surjectivity. Every crouton position must be written, so a stride typo that
     leaves holes cannot hide behind a round trip that only reads back what it
     wrote.
  3. Round trip, bit-exact, multi-tile and at a non-zero origin. fp16 in, fp16
     out, no arithmetic between: equality is the only acceptable bar.

Plus a falsifiable control that MUST fire: packing with the weight rule and
unpacking with the activation rule has to be reported as a mismatch, or checks
1-3 are asserting a permutation nobody constrained.

THE FILL IS PART OF THE CONTROL, and this is recorded from a real miss (KB
`kernels/hmx-crouton-tile-layout`): the obvious period-2 / period-4 fill makes a
weight packed with the ACTIVATION rule fold to a BYTE-IDENTICAL tile, because the
two orders share a parity over a 32-wide tile. Periods here are 5 and 7 — coprime
with each other and with 32 — so the control can see the transpose it exists to
catch.

What this cannot prove: that the hardware reads this rule. The layout is
reimplemented from a public MIT-licensed backend's own packers, not from the
closed archive, and it is confirmed only by a silicon run.

Run from this directory (imports the sibling core):
  mojo run test_hmx_crouton.mojo
"""
from std.math import align_up, exp, iota, sqrt
from std.memory import AddressSpace, alloc, bitcast
from std.sys import align_of

from layout.coord import Coord
from layout.tile_layout import row_major

from hexagon_ondevice.hmx_attention import (
    ATTN_BC,
    ATTN_BR,
    ATTN_STAT_L,
    ATTN_STAT_M,
    ATTN_THREADS,
    AttnSoftmax,
    AttnStats,
    HmxAttnArena,
    HmxAttnPlan,
    HmxAttnSched,
    hmx_attn_accepts,
    hmx_attn_arrivals,
    hmx_attn_acc_probe,
    hmx_attn_probe_fits,
    hmx_attn_arena_bytes,
    hmx_attn_arena_in,
    hmx_attn_normalise,
    hmx_attn_pack_k,
    hmx_attn_pack_k_share,
    hmx_attn_pack_k_share_vsc,
    hmx_attn_pack_q,
    hmx_attn_pack_v,
    hmx_attn_pack_v_share,
    hmx_attn_pack_v_share_keyed,
    hmx_attn_plan,
    hmx_attn_probs_from_rows,
    hmx_attn_program_records,
    hmx_attn_pv_block,
    hmx_attn_qk_block,
    hmx_attn_reset_diag,
    hmx_attn_sched,
    hmx_attn_scores_to_rows,
    hmx_attn_set_diag,
    hmx_attn_step,
    hmx_attn_step_masked,
    hmx_attn_store_o,
    hmx_attn_zero_panel,
)
from hexagon_ondevice.hmx_crouton import (
    HMX_OUT_SETS,
    HMX_DTYPE,
    HMX_EMIT_MMA,
    HMX_FOLD_TRAILING,
    HMX_LANE_BYTES,
    HMX_RECORD_LANES,
    HMX_SPLIT,
    HMX_TILE,
    HMX_TILE_BYTES,
    HMX_TILE_ELEMS,
    HMX_WGT_RING,
    HMX_XFER_COPY,
    HMX_XFER_DEFERRED,
    HMX_XFER_POSTED,
    HMX_XFER_STALE,
    CroutonPanel,
    HmxOperands,
    HmxRequantRecord,
    HmxRows2D,
    HmxScratch,
    hmx_deep_range,
    hmx_mma_chain,
    hmx_k_tiles,
    hmx_kgather_vsc_selects,
    hmx_native_row_croutons,
    hmx_native_weight_croutons,
    hmx_pack_panel,
    hmx_pack_panel_scalar,
    hmx_pack_panel_vec,
    hmx_pack_panel_vsc,
    hmx_pack_tile_pair_vec,
    hmx_pack_vec_mha_selects,
    hmx_pack_vec_selects,
    hmx_panel,
    HMX_PORT_PROBE_BUDGET_PPM,
    hmx_port_probe_bytes,
    hmx_port_probe_excluded_ppm,
    hmx_port_probe_fold,
    hmx_port_probe_issues,
    hmx_port_probe_min_tiles,
    hmx_repack_rows_rpm,
    hmx_repack_weight_cpm,
    hmx_scratch_bytes,
    hmx_scratch_in,
    hmx_stripe_act_slots,
    HMX_ARENA_STAGGER_CROUTONS,
    HMX_STRIPE_GUARD_CROUTONS,
    hmx_stripe_arena_bytes,
    hmx_stripe_arena_in,
    hmx_subpanel,
    hmx_unpack_panel,
    hmx_unpack_panel_scalar,
    hmx_unpack_panel_vec,
    hmx_unpack_vec_admits,
    HMX_UNPACK_ARM_NATIVE,
    HMX_UNPACK_ARM_SCALAR_OFF,
    HMX_UNPACK_ARM_SCALAR_REFUSED,
    HMX_UNPACK_ARM_VECTOR,
)
from hexagon_ondevice.hvx_permute import (
    HVX_BYTES,
    HVX_DEAL_SEPARATE,
    HVX_HALF_BYTES,
    HVX_SHUFF_INTERLEAVE,
    HVX_WORDS,
    hvx_lanes,
    hvx_run_mask,
    hvx_store_masked,
    hvx_vdeal,
    hvx_vscatter_words,
    hvx_vshuff,
)
from hexagon_ondevice.matmul_kernel import (
    MM_W,
    MMTile,
    hmx_stripe_arrivals,
    hmx_stripe_plan,
    hmx_stripe_roles,
    hmx_work_share,
    matmul_bt_hmx_accepts,
    matmul_bt_stripe_hmx,
    matmul_bt_stripe_hmx_step,
)

comptime GEN = AddressSpace.GENERIC

comptime SENTINEL = Scalar[HMX_DTYPE](-1.0)
"""A value the fills below never produce, so an unwritten crouton position is
distinguishable from a written one."""


def fill_coprime(t: HmxRows2D, rows: Int, cols: Int):
    """Fills `t[r, c]` with `5*r + 7*c`, distinct under exchanging the axes.

    Periods coprime with each other and with `HMX_TILE`. Values stay small
    integers, which fp16 holds exactly, so every comparison below is equality.
    """
    for r in range(rows):
        for c in range(cols):
            t.store[1](Coord(r, c), Scalar[HMX_DTYPE](5 * r + 7 * c))


def fill_bits(t: HmxRows2D, rows: Int, cols: Int):
    """Fills `t[r, c]` with a position-hashed BIT pattern, exact at any extent.

    `fill_coprime`'s values pass fp16's 2048 integer ceiling at production
    extents, where the format's granularity makes NEIGHBOURS equal and a
    misplaced word can alias its way past an fp comparison. This fill never
    does fp arithmetic on the value at all — the pattern is bits under a
    Knuth multiplicative hash of the position — so consumers must compare as
    uint16 bits too: the patterns include NaN payloads, which fp equality
    would miscount in both directions.
    """
    for r in range(rows):
        for c in range(cols):
            var word = UInt32(r * cols + c) * UInt32(2654435761)
            var bits = SIMD[DType.uint16, 1](UInt16((word >> 13) & 0xFFFF))
            t.store[1](Coord(r, c), bitcast[HMX_DTYPE, 1](bits)[0])


def fill_ids(t: HmxRows2D, rows: Int, cols: Int):
    """Fills `t[r, c]` with `r * cols + c`.

    Every position gets a distinct id. fp16 is exact to 2048 and one tile holds
    1024 elements, so an id names its source position without ambiguity.
    """
    for r in range(rows):
        for c in range(cols):
            t.store[1](Coord(r, c), Scalar[HMX_DTYPE](r * cols + c))


def check_byte_order[slab_is_depth: Bool](label: String) -> Bool:
    """Asserts WHICH source element lands at each crouton offset.

    One tile of unique ids is packed, then read through a flat `[1, 1024]` view
    of the same buffer and compared against the KB rule evaluated independently
    here. Under `slab_is_depth` the slab is the depth axis and the lane is the
    source row (the weight's transposing gather); otherwise the slab is the
    source row (the activation).
    """
    var src_buf = alloc[Scalar[HMX_DTYPE]](HMX_TILE_ELEMS)
    var dst_buf = alloc[Scalar[HMX_DTYPE]](HMX_TILE_ELEMS)
    var src = HmxRows2D(
        ptr=src_buf.as_unsafe_any_origin(),
        layout=row_major(Coord(HMX_TILE, HMX_TILE)),
    )
    var flat = HmxRows2D(
        ptr=dst_buf.as_unsafe_any_origin(),
        layout=row_major(Coord(1, HMX_TILE_ELEMS)),
    )
    fill_ids(src, HMX_TILE, HMX_TILE)

    hmx_pack_panel[slab_is_depth, GEN](hmx_panel[GEN](dst_buf, 1), src, 0, 0, 1)

    var bad = 0
    for slab in range(HMX_TILE):
        for lane in range(HMX_TILE):
            var offset = (slab // 2) * (HMX_TILE * 2) + lane * 2 + (slab % 2)
            var row = lane if slab_is_depth else slab
            var col = slab if slab_is_depth else lane
            var want = Scalar[HMX_DTYPE](row * HMX_TILE + col)
            if flat.load[1](Coord(0, offset)) != want:
                bad += 1

    src_buf.free()
    dst_buf.free()
    if bad != 0:
        print("  FAIL byte-order", label, "-", bad, "of", HMX_TILE_ELEMS)
        return False
    print("  ok   byte-order", label)
    return True


def check_surjective[slab_is_depth: Bool](tiles: Int, label: String) -> Bool:
    """Asserts the pack writes EVERY crouton position of every tile."""
    var cols = tiles * HMX_TILE
    var src_buf = alloc[Scalar[HMX_DTYPE]](HMX_TILE * cols)
    var dst_buf = alloc[Scalar[HMX_DTYPE]](tiles * HMX_TILE_ELEMS)
    var src = HmxRows2D(
        ptr=src_buf.as_unsafe_any_origin(),
        layout=row_major(Coord(HMX_TILE, cols)),
    )
    var flat = HmxRows2D(
        ptr=dst_buf.as_unsafe_any_origin(),
        layout=row_major(Coord(1, tiles * HMX_TILE_ELEMS)),
    )
    fill_coprime(src, HMX_TILE, cols)
    for i in range(tiles * HMX_TILE_ELEMS):
        flat.store[1](Coord(0, i), SENTINEL)

    hmx_pack_panel[slab_is_depth, GEN](
        hmx_panel[GEN](dst_buf, tiles), src, 0, 0, tiles
    )

    var holes = 0
    for i in range(tiles * HMX_TILE_ELEMS):
        if flat.load[1](Coord(0, i)) == SENTINEL:
            holes += 1

    src_buf.free()
    dst_buf.free()
    if holes != 0:
        print("  FAIL surjective", label, "-", holes, "unwritten")
        return False
    print("  ok   surjective", label, "tiles=", tiles)
    return True


def roundtrip_mismatches[
    pack_depth: Bool, unpack_depth: Bool
](tiles: Int, row0: Int, col0: Int) -> Int:
    """Packs a region, unpacks it into a cleared buffer, counts differences.

    Both rules are parameters so the same body serves the equality checks
    (`pack_depth == unpack_depth`) and the falsifiable control, which must report
    a nonzero count.
    """
    var rows = row0 + HMX_TILE
    var cols = col0 + tiles * HMX_TILE
    var src_buf = alloc[Scalar[HMX_DTYPE]](rows * cols)
    var back_buf = alloc[Scalar[HMX_DTYPE]](rows * cols)
    var pack_buf = alloc[Scalar[HMX_DTYPE]](tiles * HMX_TILE_ELEMS)
    var src = HmxRows2D(
        ptr=src_buf.as_unsafe_any_origin(), layout=row_major(Coord(rows, cols))
    )
    var back = HmxRows2D(
        ptr=back_buf.as_unsafe_any_origin(), layout=row_major(Coord(rows, cols))
    )
    fill_coprime(src, rows, cols)
    for r in range(rows):
        for c in range(cols):
            back.store[1](Coord(r, c), SENTINEL)

    hmx_pack_panel[pack_depth, GEN](
        hmx_panel[GEN](pack_buf, tiles), src, row0, col0, tiles
    )
    hmx_unpack_panel[unpack_depth, GEN](
        back, hmx_panel[GEN](pack_buf, tiles), row0, col0, tiles
    )

    var bad = 0
    for r in range(row0, rows):
        for c in range(col0, cols):
            if back.load[1](Coord(r, c)) != src.load[1](Coord(r, c)):
                bad += 1

    src_buf.free()
    back_buf.free()
    pack_buf.free()
    return bad


def check_roundtrip[
    slab_is_depth: Bool
](tiles: Int, row0: Int, col0: Int, label: String) -> Bool:
    """Asserts pack-then-unpack is the bit-exact identity over the region."""
    var bad = roundtrip_mismatches[slab_is_depth, slab_is_depth](
        tiles, row0, col0
    )
    if bad != 0:
        print("  FAIL roundtrip", label, "-", bad, "elements differ")
        return False
    print(
        "  ok   roundtrip",
        label,
        "tiles=",
        tiles,
        "origin=(",
        row0,
        ",",
        col0,
        ")",
    )
    return True


def oracle_wrong_axis(tiles: Int) -> Bool:
    """MUST fire: the weight rule packed and the activation rule unpacked.

    The two axis assignments are transposes of each other, so composing them is a
    transpose of every tile and the round-trip check has to report it. If this
    ever returns True quietly, the fill has gone degenerate again and every check
    above is asserting nothing about the axis roles.
    """
    var bad = roundtrip_mismatches[True, False](tiles, 0, 0)
    if bad == 0:
        print("  FAIL oracle wrong-axis did NOT fire (tiles=", tiles, ")")
        return False
    print("  ok   oracle wrong-axis fired -", bad, "elements differ")
    return True


def check_vshuff_interleave() -> Bool:
    """Pins `hvx_vshuff`'s lane rule at `HVX_SHUFF_INTERLEAVE`, on its own.

    The vector packer is only as right as the permute under it, and a permute
    that is wrong by a stride bit still produces a full, plausible-looking panel.
    So the rule is asserted here directly, in the same spirit as
    `check_byte_order`: which INPUT lane lands at which OUTPUT position, not that
    some bijection happened.

    What this cannot prove is the part: the twin runs LLVM's model of
    `V6_vshuffvdd`, so this pins the model and the packer against each other, and
    a device run is what pins the model against silicon.
    """
    comptime LANES = hvx_lanes[HMX_DTYPE]()
    comptime HALF = LANES // 2
    var vv = SIMD[HMX_DTYPE, LANES]()
    var vu = SIMD[HMX_DTYPE, LANES]()
    for i in range(LANES):
        # Distinct, and small enough that fp16 holds every one of them exactly.
        vv[i] = Scalar[HMX_DTYPE](i)
        vu[i] = Scalar[HMX_DTYPE](LANES + i)

    var pair = hvx_vshuff[HVX_SHUFF_INTERLEAVE](vu, vv)

    var bad = 0
    for i in range(HALF):
        if pair[2 * i] != vv[i]:
            bad += 1
        if pair[2 * i + 1] != vu[i]:
            bad += 1
        if pair[LANES + 2 * i] != vv[HALF + i]:
            bad += 1
        if pair[LANES + 2 * i + 1] != vu[HALF + i]:
            bad += 1

    if bad != 0:
        print("  FAIL vshuff interleave -", bad, "of", 2 * LANES, "positions")
        return False
    print("  ok   vshuff interleave lane rule")
    return True


def pack_vec_mismatches[
    slab_is_depth: Bool
](tiles: Int, row0: Int, col0: Int, ids: Bool) -> Int:
    """Packs one region with both packers and counts differing crouton elements.

    The scalar packer is the definition, so this is not a comparison of two
    opinions: it asks whether the permute network reproduced the layout exactly.
    Both destinations start at `SENTINEL`, so a position the vector packer failed
    to write differs rather than passing as agreement.
    """
    var rows = row0 + HMX_TILE
    var cols = col0 + tiles * HMX_TILE
    var src_buf = alloc[Scalar[HMX_DTYPE]](rows * cols)
    var ref_buf = alloc[Scalar[HMX_DTYPE]](tiles * HMX_TILE_ELEMS)
    var vec_buf = alloc[Scalar[HMX_DTYPE]](tiles * HMX_TILE_ELEMS)
    var src = HmxRows2D(
        ptr=src_buf.as_unsafe_any_origin(), layout=row_major(Coord(rows, cols))
    )
    var ref_flat = HmxRows2D(
        ptr=ref_buf.as_unsafe_any_origin(),
        layout=row_major(Coord(1, tiles * HMX_TILE_ELEMS)),
    )
    var vec_flat = HmxRows2D(
        ptr=vec_buf.as_unsafe_any_origin(),
        layout=row_major(Coord(1, tiles * HMX_TILE_ELEMS)),
    )
    if ids:
        fill_ids(src, rows, cols)
    else:
        fill_coprime(src, rows, cols)
    for i in range(tiles * HMX_TILE_ELEMS):
        ref_flat.store[1](Coord(0, i), SENTINEL)
        vec_flat.store[1](Coord(0, i), SENTINEL)

    hmx_pack_panel_scalar[slab_is_depth, GEN](
        hmx_panel[GEN](ref_buf, tiles), src, row0, col0, tiles
    )
    hmx_pack_panel_vec[slab_is_depth, GEN](
        hmx_panel[GEN](vec_buf, tiles), src, row0, col0, tiles
    )

    var bad = 0
    for i in range(tiles * HMX_TILE_ELEMS):
        if vec_flat.load[1](Coord(0, i)) != ref_flat.load[1](Coord(0, i)):
            bad += 1

    src_buf.free()
    ref_buf.free()
    vec_buf.free()
    return bad


def check_pack_vec[
    slab_is_depth: Bool
](tiles: Int, row0: Int, col0: Int, ids: Bool, label: String) -> Bool:
    """Asserts the vector packer is byte-identical to the scalar one."""
    var bad = pack_vec_mismatches[slab_is_depth](tiles, row0, col0, ids)
    if bad != 0:
        print("  FAIL pack-vec", label, "-", bad, "elements differ")
        return False
    print(
        "  ok   pack-vec",
        label,
        "tiles=",
        tiles,
        "origin=(",
        row0,
        ",",
        col0,
        ") fill=",
        "ids" if ids else "coprime",
    )
    return True


def pack_vec_perm_mismatches[rt: Int](tiles: Int) -> Int:
    """Packs the activation rule with `rt` in place of `HVX_SHUFF_INTERLEAVE`.

    The mutation lives in the PERMUTE rather than in the comparison, and the same
    body serves both directions: at `rt = HVX_SHUFF_INTERLEAVE` it is
    `hmx_pack_panel_vec`'s activation arm and must agree with the scalar packer,
    and at a mutated `rt` the network computes a different permutation and must
    not. Running the control and its own self-check through one body is what
    stops a broken harness from passing as a fired oracle.

    `tiles` must be even; the tail this body does not have is the scalar
    packer's, and it is not what the control is about.
    """
    comptime LANES = hvx_lanes[HMX_DTYPE]()
    comptime STEP_TILES = LANES // HMX_TILE
    comptime SLAB_PAIRS = HMX_TILE // HMX_SPLIT
    var cols = tiles * HMX_TILE
    var src_buf = alloc[Scalar[HMX_DTYPE]](HMX_TILE * cols)
    var ref_buf = alloc[Scalar[HMX_DTYPE]](tiles * HMX_TILE_ELEMS)
    var vec_buf = alloc[Scalar[HMX_DTYPE]](tiles * HMX_TILE_ELEMS)
    var src = HmxRows2D(
        ptr=src_buf.as_unsafe_any_origin(),
        layout=row_major(Coord(HMX_TILE, cols)),
    )
    var ref_flat = HmxRows2D(
        ptr=ref_buf.as_unsafe_any_origin(),
        layout=row_major(Coord(1, tiles * HMX_TILE_ELEMS)),
    )
    var vec_flat = HmxRows2D(
        ptr=vec_buf.as_unsafe_any_origin(),
        layout=row_major(Coord(1, tiles * HMX_TILE_ELEMS)),
    )
    fill_coprime(src, HMX_TILE, cols)

    hmx_pack_panel_scalar[False, GEN](
        hmx_panel[GEN](ref_buf, tiles), src, 0, 0, tiles
    )

    var dst = hmx_panel[GEN](vec_buf, tiles)
    for s in range(tiles // STEP_TILES):
        var t0 = s * STEP_TILES
        var col = t0 * HMX_TILE
        for p in range(SLAB_PAIRS):
            var pair = hvx_vshuff[rt](
                src.load[LANES](Coord(HMX_SPLIT * p + 1, col)),
                src.load[LANES](Coord(HMX_SPLIT * p, col)),
            )
            dst.store[LANES](Coord(t0, p, 0, 0), pair.slice[LANES, offset=0]())
            dst.store[LANES](
                Coord(t0 + 1, p, 0, 0), pair.slice[LANES, offset=LANES]()
            )

    var bad = 0
    for i in range(tiles * HMX_TILE_ELEMS):
        if vec_flat.load[1](Coord(0, i)) != ref_flat.load[1](Coord(0, i)):
            bad += 1

    src_buf.free()
    ref_buf.free()
    vec_buf.free()
    return bad


def check_pack_vec_perm(tiles: Int) -> Bool:
    """Asserts the control harness reproduces the packer at the true `rt`."""
    var bad = pack_vec_perm_mismatches[HVX_SHUFF_INTERLEAVE](tiles)
    if bad != 0:
        print("  FAIL pack-vec control harness -", bad, "elements differ")
        return False
    print("  ok   pack-vec control harness agrees at tiles=", tiles)
    return True


def oracle_pack_vec_mutant(tiles: Int) -> Bool:
    """MUST fire: one exchange stage dropped from the interleave network.

    `-4` is `HVX_SHUFF_INTERLEAVE` without its 2-byte stage, so the permute still
    moves every lane and still fills every crouton position -- it just pairs the
    wrong ones. That is the shape a real permute bug takes, and if the byte
    identity above cannot see it, the fill has gone degenerate and the gate is
    asserting that two packers agree about nothing.
    """
    var bad = pack_vec_perm_mismatches[-4](tiles)
    if bad == 0:
        print("  FAIL oracle pack-vec-mutant did NOT fire (tiles=", tiles, ")")
        return False
    print("  ok   oracle pack-vec-mutant fired -", bad, "elements differ")
    return True


def _unpack_dst(
    buf: UnsafePointer[Scalar[HMX_DTYPE], MutUntrackedOrigin],
    rows: Int,
    cols: Int,
) -> HmxRows2D:
    """A row-major readout destination over `buf`, cleared to `SENTINEL`.

    Cleared over its WHOLE extent, not just the panel's columns, because the
    masked store is the arm's risk: a run written to the wrong half of its
    window lands outside the panel, and a position nobody wrote has to be
    distinguishable from one written correctly for the comparison to see that.
    """
    var t = HmxRows2D(
        ptr=buf.as_unsafe_any_origin(), layout=row_major(Coord(rows, cols))
    )
    for r in range(rows):
        for c in range(cols):
            t.store[1](Coord(r, c), SENTINEL)
    return t


def unpack_vec_mismatches(
    tiles: Int, row0: Int, col0: Int, cols: Int, ids: Bool
) -> Int:
    """Unpacks a panel with both arms, counting elements that differ.

    The scalar unpack is the definition, so this asks whether the permute and
    the masked store between them reproduced it — over the ENTIRE destination,
    panel columns and all others, which is what makes a run stored to the wrong
    window half a mismatch rather than an invisible clobber of a neighbour
    nobody checked.

    The source panel is PACKED from a filled region rather than filled directly:
    a crouton holds what a readout would hold, and the values stay small
    integers fp16 is exact in, so every comparison is equality.
    """
    var rows = row0 + HMX_TILE
    var seed_buf = alloc[Scalar[HMX_DTYPE]](rows * cols)
    var src_buf = alloc[Scalar[HMX_DTYPE]](
        tiles * HMX_TILE_ELEMS, alignment=HMX_TILE_BYTES
    )
    var ref_buf = alloc[Scalar[HMX_DTYPE]](rows * cols, alignment=HVX_BYTES)
    var vec_buf = alloc[Scalar[HMX_DTYPE]](rows * cols, alignment=HVX_BYTES)

    var seed = HmxRows2D(
        ptr=seed_buf.as_unsafe_any_origin(), layout=row_major(Coord(rows, cols))
    )
    if ids:
        fill_ids(seed, rows, cols)
    else:
        fill_coprime(seed, rows, cols)
    var panel = hmx_panel[GEN](src_buf, tiles)
    hmx_pack_panel_scalar[False, GEN](panel, seed, row0, col0, tiles)

    var reference = _unpack_dst(ref_buf, rows, cols)
    var vec = _unpack_dst(vec_buf, rows, cols)
    hmx_unpack_panel_scalar[False, GEN](reference, panel, row0, col0, tiles)
    hmx_unpack_panel_vec[GEN](vec, panel, row0, col0, tiles)

    var bad = 0
    for r in range(rows):
        for c in range(cols):
            if vec.load[1](Coord(r, c)) != reference.load[1](Coord(r, c)):
                bad += 1

    seed_buf.free()
    src_buf.free()
    ref_buf.free()
    vec_buf.free()
    return bad


def check_unpack_vec(
    tiles: Int, row0: Int, col0: Int, cols: Int, ids: Bool
) -> Bool:
    """Asserts the vector readout is byte-identical to the scalar one."""
    var bad = unpack_vec_mismatches(tiles, row0, col0, cols, ids)
    if bad != 0:
        print(
            "  FAIL unpack-vec - ",
            bad,
            "elements differ (tiles=",
            tiles,
            "origin=(",
            row0,
            ",",
            col0,
            ") cols=",
            cols,
            ")",
        )
        return False
    print(
        "  ok   unpack-vec tiles=",
        tiles,
        "origin=(",
        row0,
        ",",
        col0,
        ") cols=",
        cols,
        " runs/row=",
        cols // HMX_TILE,
        "fill=",
        "ids" if ids else "coprime",
    )
    return True


def check_pack_vec_routing() -> Bool:
    """Pins `hmx_pack_vec_selects`'s whole truth table.

    The routing is the fix for the order-149 refusal — the weight rule's vector
    arm (the 32-register butterfly) must not instantiate under `HMX_PACK_VEC`
    alone — and this is the one place all eight flag combinations are decidable
    in a single compilation: the predicate takes the flags as parameters, and
    `hmx_pack_panel` consumes this exact predicate, so passing here means the
    dispatch cannot route the butterfly into an object that did not name
    `SKYE_HMX_PACK_VEC_WGT`.
    """
    var bad = 0
    # The flag off is scalar for both rules regardless of the weight gate.
    comptime if hmx_pack_vec_selects[False, False, False]():
        bad += 1
    comptime if hmx_pack_vec_selects[False, False, True]():
        bad += 1
    comptime if hmx_pack_vec_selects[True, False, False]():
        bad += 1
    comptime if hmx_pack_vec_selects[True, False, True]():
        bad += 1
    # The flag on vectorises the activation rule with or without the gate...
    comptime if not hmx_pack_vec_selects[False, True, False]():
        bad += 1
    comptime if not hmx_pack_vec_selects[False, True, True]():
        bad += 1
    # ...and the weight rule only when the gate names it too.
    comptime if hmx_pack_vec_selects[True, True, False]():
        bad += 1
    comptime if not hmx_pack_vec_selects[True, True, True]():
        bad += 1
    if bad != 0:
        print("  FAIL pack-vec routing -", bad, "of 8 rows wrong")
        return False
    print("  ok   pack-vec routing: 8/8 rows; weight rule needs both flags")
    return True


def check_pack_vec_mha_routing() -> Bool:
    """Pins `hmx_pack_vec_mha_selects`'s whole truth table.

    The attention step consumes this exact predicate, so passing here means the
    split-pair V pack cannot enter an object that did not name BOTH
    `SKYE_HMX_PACK_VEC` (the proven interleave class) and
    `SKYE_HMX_PACK_VEC_MHA` — and a build naming only the mha flag is inert
    rather than a hybrid nothing measured.
    """
    var bad = 0
    comptime if hmx_pack_vec_mha_selects[False, False]():
        bad += 1
    comptime if hmx_pack_vec_mha_selects[False, True]():
        bad += 1
    comptime if hmx_pack_vec_mha_selects[True, False]():
        bad += 1
    comptime if not hmx_pack_vec_mha_selects[True, True]():
        bad += 1
    if bad != 0:
        print("  FAIL pack-vec mha routing -", bad, "of 4 rows wrong")
        return False
    print("  ok   pack-vec mha routing: 4/4 rows; pair arm needs both flags")
    return True


def pack_tile_pair_mismatches(row0: Int, col0: Int, swapped: Bool) -> Int:
    """Split-pair packer vs two scalar single-tile packs, in one body.

    `swapped` steers the pair's destinations backwards, which is the harness's
    own falsifier: the same comparison that must count zero at the true wiring
    must count nonzero at the swapped one, or the zero proves nothing.
    """
    var rows = row0 + HMX_TILE
    var cols = col0 + 3 * HMX_TILE  # slack past the pair: a strided source
    var src_buf = alloc[Scalar[HMX_DTYPE]](rows * cols)
    var ref_buf = alloc[Scalar[HMX_DTYPE]](2 * HMX_TILE_ELEMS)
    var got_buf = alloc[Scalar[HMX_DTYPE]](2 * HMX_TILE_ELEMS)
    var src = HmxRows2D(
        ptr=src_buf.as_unsafe_any_origin(),
        layout=row_major(Coord(rows, cols)),
    )
    fill_coprime(src, rows, cols)

    # The reference is the layout's definition, one destination tile at a time
    # — exactly what the scalar share does for the V panel.
    var want = hmx_panel[GEN](ref_buf, 2)
    hmx_pack_panel_scalar[False, GEN](
        hmx_subpanel(want, 0, 1), src, row0, col0, 1
    )
    hmx_pack_panel_scalar[False, GEN](
        hmx_subpanel(want, 1, 1), src, row0, col0 + HMX_TILE, 1
    )

    var got = hmx_panel[GEN](got_buf, 2)
    var lo = hmx_subpanel(got, 0, 1)
    var hi = hmx_subpanel(got, 1, 1)
    if swapped:
        hmx_pack_tile_pair_vec(hi, lo, src, row0, col0)
    else:
        hmx_pack_tile_pair_vec(lo, hi, src, row0, col0)

    var bad = 0
    for t in range(2):
        for sp in range(HMX_TILE // HMX_SPLIT):
            for lane in range(HMX_TILE):
                for s in range(HMX_SPLIT):
                    var at = Coord(t, sp, lane, s)
                    if got.load[1](at) != want.load[1](at):
                        bad += 1
    src_buf.free()
    ref_buf.free()
    got_buf.free()
    return bad


def check_pack_tile_pair_vec(row0: Int, col0: Int) -> Bool:
    """Asserts the split-pair packer is byte-identical to the scalar packs."""
    var bad = pack_tile_pair_mismatches(row0, col0, False)
    if bad != 0:
        print("  FAIL pack-tile-pair -", bad, "elements differ")
        return False
    print("  ok   pack-tile-pair origin=(", row0, ",", col0, ")")
    return True


def oracle_pack_tile_pair_swapped() -> Bool:
    """MUST fire: swapped destinations have to be caught by the same count."""
    var bad = pack_tile_pair_mismatches(0, 0, True)
    if bad == 0:
        print("  FAIL oracle pack-tile-pair swap did NOT fire")
        return False
    print("  ok   oracle pack-tile-pair swap fired -", bad, "elements differ")
    return True


def attn_pack_v_keyed_mismatches(
    br: Int, bc: Int, d: Int, half: Int, k0: Int, flipped: Bool
) -> Int:
    """The keyed vector V share vs the scalar DV share, whole panel, one body.

    Both arenas' V panels are zeroed first and compared over BOTH halves, so a
    write landing outside the staged half counts as a mismatch instead of
    hiding in uninitialized bytes. `flipped` swaps the keyed share's staging
    half — the falsifier the whole-panel comparison exists to catch.
    """
    var plan = hmx_attn_plan(br, bc, d, ATTN_THREADS)
    if not plan.usable():
        return -1
    var seq = k0 + bc
    var cols = d + HMX_TILE  # slack past the head: the strided [seq, H*D] shape
    var v_buf = alloc[Scalar[HMX_DTYPE]](seq * cols)
    var V = HmxRows2D(
        ptr=v_buf.as_unsafe_any_origin(),
        layout=row_major(Coord(seq, cols)),
    )
    fill_coprime(V, seq, cols)

    var ref_raw = alloc[UInt8](hmx_attn_arena_bytes(plan) + HMX_TILE_BYTES)
    var got_raw = alloc[UInt8](hmx_attn_arena_bytes(plan) + HMX_TILE_BYTES)
    var ref_arena = hmx_attn_arena_in[GEN](
        UnsafePointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=align_up(Int(ref_raw), HMX_TILE_BYTES)
        ),
        plan,
    )
    var got_arena = hmx_attn_arena_in[GEN](
        UnsafePointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=align_up(Int(got_raw), HMX_TILE_BYTES)
        ),
        plan,
    )
    var v_tiles = 2 * plan.d_tiles * plan.bc_tiles
    hmx_attn_zero_panel(hmx_subpanel(ref_arena.v, 0, v_tiles), v_tiles)
    hmx_attn_zero_panel(hmx_subpanel(got_arena.v, 0, v_tiles), v_tiles)

    hmx_attn_pack_v_share(ref_arena, plan, V, k0, 0, half, 0, plan.d_tiles)
    var got_half = (1 - half) if flipped else half
    hmx_attn_pack_v_share_keyed(
        got_arena, plan, V, k0, 0, got_half, 0, plan.bc_tiles
    )

    var bad = 0
    for t in range(v_tiles):
        for sp in range(HMX_TILE // HMX_SPLIT):
            for lane in range(HMX_TILE):
                for s in range(HMX_SPLIT):
                    var at = Coord(t, sp, lane, s)
                    if got_arena.v.load[1](at) != ref_arena.v.load[1](at):
                        bad += 1
    v_buf.free()
    ref_raw.free()
    got_raw.free()
    return bad


def check_attn_pack_v_keyed(
    br: Int, bc: Int, d: Int, half: Int, k0: Int
) -> Bool:
    """Asserts the keyed V share stages the exact bytes the scalar share does.
    """
    var bad = attn_pack_v_keyed_mismatches(br, bc, d, half, k0, False)
    if bad != 0:
        print(
            "  FAIL attn V keyed share d=",
            d,
            "half=",
            half,
            "k0=",
            k0,
            "-",
            bad,
            "elements differ",
        )
        return False
    print(
        "  ok   attn V keyed share d=",
        d,
        "(",
        plan_d_label(d),
        ") half=",
        half,
        "k0=",
        k0,
    )
    return True


def plan_d_label(d: Int) -> String:
    """Names which arm the DV extent exercises, for the log line."""
    return "pairs+tail" if ((d // HMX_TILE) % 2) != 0 else "whole pairs"


def oracle_attn_pack_v_keyed_flipped() -> Bool:
    """MUST fire: the keyed share staged into the wrong half has to count."""
    var bad = attn_pack_v_keyed_mismatches(512, 256, 128, 0, 0, True)
    if bad <= 0:
        print("  FAIL oracle attn V keyed flip did NOT fire")
        return False
    print("  ok   oracle attn V keyed flip fired -", bad, "elements differ")
    return True


def check_kgather_vsc_routing() -> Bool:
    """Pins `hmx_kgather_vsc_selects`'s whole truth table.

    Two rows, and both are load-bearing: `hmx_attn_pack_k_share` consumes this
    exact predicate, so passing here means a build not naming
    `SKYE_HMX_KGATHER_VSC` keeps the scalar K gather byte for byte, and a
    build naming it routes the site — with no third state and no coupling to
    the `HMX_PACK_VEC` family, which is the one-variable contract the device
    A/B leans on.
    """
    var bad = 0
    comptime if hmx_kgather_vsc_selects[False]():
        bad += 1
    comptime if not hmx_kgather_vsc_selects[True]():
        bad += 1
    if bad != 0:
        print("  FAIL kgather vsc routing -", bad, "of 2 rows wrong")
        return False
    print("  ok   kgather vsc routing: 2/2 rows; its own flag, no conjunction")
    return True


def check_vscatter_words_binding() -> Bool:
    """Pins the scatter model's two semantic clauses, collision-free.

    Placement: word ids scattered at the crouton offset table must land at
    `w * 128 + lane * 4` of a two-tile region, and every byte the offsets do
    not name must keep its sentinel. Region: the SAME table displaced past the
    region must land at the MASKED addresses — the manual's `& Mu` — so a
    model that added offsets unmasked writes 4 KiB high and fails both counts.
    The displaced words map to distinct in-region addresses, so neither clause
    depends on colliding-lane order, which the ISA does not promise.
    """
    comptime WORDS_IN_REGION = 2 * HMX_TILE_ELEMS * HMX_SPLIT // HMX_LANE_BYTES
    comptime LANE = 5
    comptime REGION = 2 * HMX_TILE_BYTES - 1
    # Twice the region, so even an unmasked (wrong) model stays in bounds and
    # is SEEN rather than corrupting the heap.
    var buf = alloc[Scalar[DType.int32]](2 * WORDS_IN_REGION)

    var offs = iota[DType.int32, HVX_WORDS]() * Int32(
        HMX_TILE * HMX_LANE_BYTES
    ) + Int32(LANE * HMX_LANE_BYTES)
    var ids = iota[DType.int32, HVX_WORDS]() + Int32(1)

    var bad = 0
    for clause in range(2):
        for i in range(2 * WORDS_IN_REGION):
            buf[i] = Int32(-7)
        var shifted = offs + Int32(clause * (REGION + 1))
        hvx_vscatter_words[REGION](buf, shifted, ids)
        for i in range(2 * WORDS_IN_REGION):
            var byte_at = i * HMX_LANE_BYTES
            var expect = Int32(-7)
            if byte_at % (HMX_TILE * HMX_LANE_BYTES) == LANE * HMX_LANE_BYTES:
                var w = byte_at // (HMX_TILE * HMX_LANE_BYTES)
                if w < HVX_WORDS:
                    expect = Int32(w + 1)
            if buf[i] != expect:
                bad += 1
    buf.free()
    if bad != 0:
        print("  FAIL vscatter binding -", bad, "words off the model")
        return False
    print("  ok   vscatter binding: placement + region mask, both clauses")
    return True


def pack_panel_vsc_mismatches(
    rows: Int,
    cols: Int,
    row0: Int,
    col0: Int,
    tiles: Int,
    oracle_shift: Int,
) -> Int:
    """Scatter packer vs the scalar weight rule, whole panel, one body.

    The two destination panels are pre-filled with DIFFERENT sentinels, so a
    byte either arm fails to write counts as a mismatch instead of matching by
    luck, and the comparison is on uint16 BITS because `fill_bits` patterns
    include NaN payloads that fp equality would miscount. `oracle_shift`
    displaces the scalar reference's origin — the harness's own falsifier: the
    count that must be zero at the true origin must be nonzero at a shifted
    one, or the zero proves nothing.
    """
    var src_buf = alloc[Scalar[HMX_DTYPE]](rows * cols)
    var src = HmxRows2D(
        ptr=src_buf.as_unsafe_any_origin(),
        layout=row_major(Coord(rows, cols)),
    )
    fill_bits(src, rows, cols)

    var want_buf = alloc[Scalar[HMX_DTYPE]](tiles * HMX_TILE_ELEMS)
    var got_buf = alloc[Scalar[HMX_DTYPE]](tiles * HMX_TILE_ELEMS)
    for i in range(tiles * HMX_TILE_ELEMS):
        want_buf[i] = bitcast[HMX_DTYPE, 1](SIMD[DType.uint16, 1](0x5555))[0]
        got_buf[i] = bitcast[HMX_DTYPE, 1](SIMD[DType.uint16, 1](0xAAAA))[0]
    var want = hmx_panel[GEN](want_buf, tiles)
    var got = hmx_panel[GEN](got_buf, tiles)

    hmx_pack_panel_scalar[True, GEN](
        want, src, row0 + oracle_shift, col0, tiles
    )
    hmx_pack_panel_vsc(got, src, row0, col0, tiles)

    var bad = 0
    for t in range(tiles):
        for sp in range(HMX_TILE // HMX_SPLIT):
            for lane in range(HMX_TILE):
                for s in range(HMX_SPLIT):
                    var at = Coord(t, sp, lane, s)
                    var g = bitcast[DType.uint16, 1](got.load[1](at))
                    var w = bitcast[DType.uint16, 1](want.load[1](at))
                    if g != w:
                        bad += 1
    src_buf.free()
    want_buf.free()
    got_buf.free()
    return bad


def check_pack_panel_vsc(
    rows: Int, cols: Int, row0: Int, col0: Int, tiles: Int, label: StaticString
) -> Bool:
    """Asserts the scatter packer is byte-identical to the scalar weight rule.
    """
    var bad = pack_panel_vsc_mismatches(rows, cols, row0, col0, tiles, 0)
    if bad != 0:
        print(
            "  FAIL pack-panel vsc",
            label,
            "origin=(",
            row0,
            ",",
            col0,
            ") tiles=",
            tiles,
            "-",
            bad,
            "elements differ",
        )
        return False
    print(
        "  ok   pack-panel vsc",
        label,
        "origin=(",
        row0,
        ",",
        col0,
        ") tiles=",
        tiles,
    )
    return True


def oracle_pack_panel_vsc_shifted() -> Bool:
    """MUST fire: a displaced scalar reference has to be caught by the count."""
    var bad = pack_panel_vsc_mismatches(96, 256, 0, 0, 4, HMX_TILE)
    if bad <= 0:
        print("  FAIL oracle pack-panel vsc shift did NOT fire")
        return False
    print("  ok   oracle pack-panel vsc shift fired -", bad, "elements differ")
    return True


def attn_pack_k_vsc_mismatches(
    br: Int, bc: Int, d: Int, half: Int, k0: Int, head_off: Int, flipped: Bool
) -> Int:
    """The vsc K share vs the share the flag-off build runs, whole panel.

    Both arenas' K panels are zeroed first and compared over BOTH halves, so a
    write landing outside the staged half counts as a mismatch instead of
    hiding in uninitialized bytes. The reference arm is
    `hmx_attn_pack_k_share` itself — this suite compiles without the flag, so
    that call IS the scalar gather the device ships today. `flipped` swaps the
    vsc share's staging half — the falsifier the whole-panel comparison exists
    to catch.
    """
    var plan = hmx_attn_plan(br, bc, d, ATTN_THREADS)
    if not plan.usable():
        return -1
    var seq = k0 + bc
    var cols = head_off + d + HMX_TILE
    var k_buf = alloc[Scalar[HMX_DTYPE]](seq * cols)
    var K = HmxRows2D(
        ptr=k_buf.as_unsafe_any_origin(),
        layout=row_major(Coord(seq, cols)),
    )
    fill_bits(K, seq, cols)

    var ref_raw = alloc[UInt8](hmx_attn_arena_bytes(plan) + HMX_TILE_BYTES)
    var got_raw = alloc[UInt8](hmx_attn_arena_bytes(plan) + HMX_TILE_BYTES)
    var ref_arena = hmx_attn_arena_in[GEN](
        UnsafePointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=align_up(Int(ref_raw), HMX_TILE_BYTES)
        ),
        plan,
    )
    var got_arena = hmx_attn_arena_in[GEN](
        UnsafePointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=align_up(Int(got_raw), HMX_TILE_BYTES)
        ),
        plan,
    )
    var k_tiles_n = 2 * plan.bc_tiles * plan.d_tiles
    hmx_attn_zero_panel(hmx_subpanel(ref_arena.k, 0, k_tiles_n), k_tiles_n)
    hmx_attn_zero_panel(hmx_subpanel(got_arena.k, 0, k_tiles_n), k_tiles_n)

    hmx_attn_pack_k_share(
        ref_arena, plan, K, k0, head_off, half, 0, plan.bc_tiles
    )
    var got_half = (1 - half) if flipped else half
    hmx_attn_pack_k_share_vsc(
        got_arena, plan, K, k0, head_off, got_half, 0, plan.bc_tiles
    )

    var bad = 0
    for t in range(k_tiles_n):
        for sp in range(HMX_TILE // HMX_SPLIT):
            for lane in range(HMX_TILE):
                for s in range(HMX_SPLIT):
                    var at = Coord(t, sp, lane, s)
                    var g = bitcast[DType.uint16, 1](got_arena.k.load[1](at))
                    var w = bitcast[DType.uint16, 1](ref_arena.k.load[1](at))
                    if g != w:
                        bad += 1
    k_buf.free()
    ref_raw.free()
    got_raw.free()
    return bad


def check_attn_pack_k_vsc(
    br: Int, bc: Int, d: Int, half: Int, k0: Int, head_off: Int
) -> Bool:
    """Asserts the vsc K share stages the exact bytes the scalar share does."""
    var bad = attn_pack_k_vsc_mismatches(br, bc, d, half, k0, head_off, False)
    if bad != 0:
        print(
            "  FAIL attn K vsc share d=",
            d,
            "half=",
            half,
            "k0=",
            k0,
            "head_off=",
            head_off,
            "-",
            bad,
            "elements differ",
        )
        return False
    print(
        "  ok   attn K vsc share d=",
        d,
        "(",
        plan_d_label(d),
        ") half=",
        half,
        "k0=",
        k0,
        "head_off=",
        head_off,
    )
    return True


def oracle_attn_pack_k_vsc_flipped() -> Bool:
    """MUST fire: the vsc share staged into the wrong half has to count."""
    var bad = attn_pack_k_vsc_mismatches(512, 256, 128, 0, 0, 0, True)
    if bad <= 0:
        print("  FAIL oracle attn K vsc flip did NOT fire")
        return False
    print("  ok   oracle attn K vsc flip fired -", bad, "elements differ")
    return True


def check_unpack_vec_admits() -> Bool:
    """Asserts the arm's admission gate answers on the geometry it is about.

    Both directions, because the gate is the only thing standing between a
    64-byte run and a straddled window, and a gate that always said True would
    make the byte identity above a statement about one buffer.
    """
    var ok = True
    var buf = alloc[Scalar[HMX_DTYPE]](HMX_TILE * 128, alignment=HVX_BYTES)

    var aligned = HmxRows2D(
        ptr=buf.as_unsafe_any_origin(), layout=row_major(Coord(HMX_TILE, 64))
    )
    if not hmx_unpack_vec_admits(aligned, 0, 0):
        print("  FAIL unpack-vec gate refused an aligned destination")
        ok = False

    # One element in is 2 bytes in: a run that starts 2 bytes into its window
    # straddles the next one, and this is the ordinary alignment of a host
    # buffer.
    var skewed = HmxRows2D(
        ptr=(buf + 1).as_unsafe_any_origin(),
        layout=row_major(Coord(HMX_TILE, 64)),
    )
    if hmx_unpack_vec_admits(skewed, 0, 0):
        print("  FAIL unpack-vec gate admitted a 2-byte-aligned destination")
        ok = False

    # A row stride that is not a whole number of runs: row 1 starts mid-window
    # however well row 0 is aligned.
    var ragged = HmxRows2D(
        ptr=buf.as_unsafe_any_origin(), layout=row_major(Coord(HMX_TILE, 48))
    )
    if hmx_unpack_vec_admits(ragged, 0, 0):
        print("  FAIL unpack-vec gate admitted a 48-column row stride")
        ok = False

    buf.free()
    if ok:
        print("  ok   unpack-vec gate admits alignment and refuses skew")
    return ok


def check_unpack_arm_codes() -> Bool:
    """Asserts the seam reports THREE distinguishable arms, not two.

    The word this exercises exists because `HMX_UNPACK_VEC`'s failure mode is a
    silent admission early-out: the arm is refused, the scalar path runs, the
    bytes are identical and a probe-off A/B measures the arm it did not select.
    So the case that matters most here is REFUSED, and it is built the way
    `check_unpack_vec_admits` builds its negative — a destination the gate
    declines — rather than by disabling anything.

    Driven at `armed=True` and `armed=False` from ONE build, which is what the
    seam's `armed` parameter is for: the shipped object selects the arm with a
    flag, and a gate that could only see the flag's default would be asserting
    the default instead of the seam.
    """
    var ok = True
    comptime cols = 64
    var src_buf = alloc[Scalar[HMX_DTYPE]](
        HMX_TILE_ELEMS, alignment=HMX_TILE_BYTES
    )
    var seed_buf = alloc[Scalar[HMX_DTYPE]](HMX_TILE * cols)
    var dst_buf = alloc[Scalar[HMX_DTYPE]](
        HMX_TILE * cols + 1, alignment=HVX_BYTES
    )
    var seed = HmxRows2D(
        ptr=seed_buf.as_unsafe_any_origin(),
        layout=row_major(Coord(HMX_TILE, cols)),
    )
    fill_ids(seed, HMX_TILE, cols)
    var panel = hmx_panel[GEN](src_buf, 1)
    hmx_pack_panel_scalar[False, GEN](panel, seed, 0, 0, 1)

    # The aligned destination `check_unpack_vec_admits` admits.
    var aligned = _unpack_dst(dst_buf, HMX_TILE, cols)
    var off = hmx_unpack_panel[False, GEN, HMX_DTYPE, armed=False](
        aligned, panel, 0, 0, 1
    )
    if off != HMX_UNPACK_ARM_SCALAR_OFF:
        print("  FAIL flag-off unpack reported", off, "not SCALAR_OFF")
        ok = False

    var vec = hmx_unpack_panel[False, GEN, HMX_DTYPE, armed=True](
        aligned, panel, 0, 0, 1
    )
    if vec != HMX_UNPACK_ARM_VECTOR:
        print("  FAIL admitted unpack reported", vec, "not VECTOR")
        ok = False

    # A 2-byte-skewed base: every run straddles a window, so the gate refuses
    # and the scalar arm runs under a flag that asked for the vector one. This
    # is the code whose absence would make a null A/B unreadable.
    var skewed = _unpack_dst(dst_buf + 1, HMX_TILE, cols)
    var refused = hmx_unpack_panel[False, GEN, HMX_DTYPE, armed=True](
        skewed, panel, 0, 0, 1
    )
    if refused != HMX_UNPACK_ARM_SCALAR_REFUSED:
        print("  FAIL refused unpack reported", refused, "not SCALAR_REFUSED")
        ok = False

    # The whole point of three codes: the two scalar outcomes must not collide.
    if off == refused:
        print("  FAIL SCALAR_OFF and SCALAR_REFUSED are the same code")
        ok = False

    src_buf.free()
    seed_buf.free()
    dst_buf.free()
    if ok:
        print(
            "  ok   unpack arm codes distinguish off/vector/refused:",
            off,
            vec,
            refused,
        )
    return ok


def unpack_vec_mutant_mismatches[
    rt: Int, swap_masks: Bool
](tiles: Int, cols: Int) -> Int:
    """Runs the readout arm's body with `rt`, and optionally its masks swapped.

    The mutation lives in the ARM rather than in the comparison, and one body
    serves both directions: at `HVX_DEAL_SEPARATE` with masks in place it is
    `hmx_unpack_panel_vec` and must agree with the scalar unpack, and under
    either mutation it must not. Running the control and its self-check through
    one body is what stops a broken harness from passing as a fired oracle.

    The two mutations are the arm's two halves. `rt` breaks the PERMUTE, so rows
    come out paired wrong. `swap_masks` breaks the STORE, so each row is written
    to the half of the window the other row wanted — and that is observable only
    where a row stride is an ODD number of runs, because an even one gives both
    rows the same mask and exchanging equals is nothing. `cols` is the
    caller's to choose for that reason.
    """
    comptime LANES = hvx_lanes[HMX_DTYPE]()
    comptime SLAB_PAIRS = HMX_TILE // HMX_SPLIT
    comptime RUN_BYTES = HVX_HALF_BYTES
    var rows = HMX_TILE
    var seed_buf = alloc[Scalar[HMX_DTYPE]](rows * cols)
    var src_buf = alloc[Scalar[HMX_DTYPE]](
        tiles * HMX_TILE_ELEMS, alignment=HMX_TILE_BYTES
    )
    var ref_buf = alloc[Scalar[HMX_DTYPE]](rows * cols, alignment=HVX_BYTES)
    var vec_buf = alloc[Scalar[HMX_DTYPE]](rows * cols, alignment=HVX_BYTES)

    var seed = HmxRows2D(
        ptr=seed_buf.as_unsafe_any_origin(), layout=row_major(Coord(rows, cols))
    )
    fill_coprime(seed, rows, cols)
    var panel = hmx_panel[GEN](src_buf, tiles)
    hmx_pack_panel_scalar[False, GEN](panel, seed, 0, 0, tiles)

    var reference = _unpack_dst(ref_buf, rows, cols)
    var vec = _unpack_dst(vec_buf, rows, cols)
    hmx_unpack_panel_scalar[False, GEN](reference, panel, 0, 0, tiles)

    for t in range(tiles):
        var col = t * HMX_TILE
        var off_even = Int(vec.ptr_at_offset(Coord(0, col))) & (HVX_BYTES - 1)
        var off_odd = Int(vec.ptr_at_offset(Coord(1, col))) & (HVX_BYTES - 1)
        var q_even = hvx_run_mask(off_even, RUN_BYTES)
        var q_odd = hvx_run_mask(off_odd, RUN_BYTES)
        for p in range(SLAB_PAIRS):
            var v = panel.load[LANES](Coord(t, p, 0, 0))
            var pair = hvx_vdeal[rt](v, v)
            comptime if swap_masks:
                hvx_store_masked(
                    q_odd,
                    vec.ptr_at_offset(Coord(HMX_SPLIT * p, col)),
                    pair.slice[LANES, offset=0](),
                )
                hvx_store_masked(
                    q_even,
                    vec.ptr_at_offset(Coord(HMX_SPLIT * p + 1, col)),
                    pair.slice[LANES, offset=LANES](),
                )
            else:
                hvx_store_masked(
                    q_even,
                    vec.ptr_at_offset(Coord(HMX_SPLIT * p, col)),
                    pair.slice[LANES, offset=0](),
                )
                hvx_store_masked(
                    q_odd,
                    vec.ptr_at_offset(Coord(HMX_SPLIT * p + 1, col)),
                    pair.slice[LANES, offset=LANES](),
                )

    var bad = 0
    for r in range(rows):
        for c in range(cols):
            if vec.load[1](Coord(r, c)) != reference.load[1](Coord(r, c)):
                bad += 1

    seed_buf.free()
    src_buf.free()
    ref_buf.free()
    vec_buf.free()
    return bad


def check_unpack_vec_control(tiles: Int, cols: Int) -> Bool:
    """Asserts the control harness reproduces the arm unmutated."""
    var bad = unpack_vec_mutant_mismatches[HVX_DEAL_SEPARATE, False](
        tiles, cols
    )
    if bad != 0:
        print("  FAIL unpack-vec control harness -", bad, "elements differ")
        return False
    print("  ok   unpack-vec control harness agrees at cols=", cols)
    return True


def oracle_unpack_vec_deal_mutant(tiles: Int, cols: Int) -> Bool:
    """MUST fire: one exchange stage dropped from the de-interleave network.

    `-4` is `HVX_DEAL_SEPARATE` without its 2-byte stage, so the permute still
    moves every lane and still fills every position — it just pairs the wrong
    ones, which is the shape a real permute bug takes.
    """
    var bad = unpack_vec_mutant_mismatches[-4, False](tiles, cols)
    if bad == 0:
        print("  FAIL oracle unpack-vec deal-mutant did NOT fire")
        return False
    print("  ok   oracle unpack-vec deal-mutant fired -", bad, "differ")
    return True


def oracle_unpack_vec_mask_swap(tiles: Int, cols: Int) -> Bool:
    """MUST fire: the two rows' window masks exchanged.

    The store half of the arm, and the failure the new instruction introduces: a
    mask names WHICH bytes of a window a store writes, and a readout that names
    the other half writes each row over its neighbour's columns. `cols` must be
    an odd number of runs or the two masks are equal and there is nothing to
    swap — the caller's own precondition, asserted here so a passing oracle
    cannot be a degenerate one.
    """
    if (cols // HMX_TILE) % 2 == 0:
        print("  FAIL oracle unpack-vec mask-swap needs an odd runs/row")
        return False
    var bad = unpack_vec_mutant_mismatches[HVX_DEAL_SEPARATE, True](tiles, cols)
    if bad == 0:
        print("  FAIL oracle unpack-vec mask-swap did NOT fire")
        return False
    print("  ok   oracle unpack-vec mask-swap fired -", bad, "differ")
    return True


def microtile_mismatches(k_tiles: Int, perturb: Bool) -> Int:
    """Runs one HMX microtile with the MMA modelled, scores it against `A @ B^T`.

    `M = N = HMX_TILE`, so the stripe below is exactly one microtile: pack the
    activation, pack the weight, multiply, unpack the accumulator. The reference
    is a plain row-major dot product over the ORIGINAL operands, so nothing it
    computes knows the crouton exists.

    Equality is exact, not toleranced, and that is available rather than lucky:
    both sides accumulate fp32 over ascending `k` and round once on the store,
    so a difference is a wiring difference and never a rounding one.

    `perturb` corrupts one weight element after the reference is taken. It must
    make the count nonzero, or the comparison is reading something other than
    what the microtile wrote.
    """
    var K = k_tiles * HMX_TILE
    var a_buf = alloc[Scalar[HMX_DTYPE]](HMX_TILE * K)
    var b_buf = alloc[Scalar[HMX_DTYPE]](HMX_TILE * K)
    var c_buf = alloc[Scalar[HMX_DTYPE]](HMX_TILE * HMX_TILE)
    var act_buf = alloc[Scalar[HMX_DTYPE]](k_tiles * HMX_TILE_ELEMS)
    var wgt_buf = alloc[Scalar[HMX_DTYPE]](k_tiles * HMX_TILE_ELEMS)
    var out_buf = alloc[Scalar[HMX_DTYPE]](HMX_TILE_ELEMS)
    var rec_buf = alloc[Scalar[DType.uint32]](2 * HMX_RECORD_LANES)

    var A = HmxRows2D(
        ptr=a_buf.as_unsafe_any_origin(), layout=row_major(Coord(HMX_TILE, K))
    )
    var B = HmxRows2D(
        ptr=b_buf.as_unsafe_any_origin(), layout=row_major(Coord(HMX_TILE, K))
    )
    var C = HmxRows2D(
        ptr=c_buf.as_unsafe_any_origin(),
        layout=row_major(Coord(HMX_TILE, HMX_TILE)),
    )

    # Small signed integers, periods coprime with each other and with the tile,
    # so a transposed axis cannot fold to the same sum (the trap this file's
    # header records) and every product stays exact in fp16.
    for r in range(HMX_TILE):
        for c in range(K):
            A.store[1](Coord(r, c), Scalar[HMX_DTYPE]((5 * r + 7 * c) % 13 - 6))
            B.store[1](Coord(r, c), Scalar[HMX_DTYPE]((3 * r + 11 * c) % 7 - 3))
        for c in range(HMX_TILE):
            C.store[1](Coord(r, c), SENTINEL)

    var want = alloc[Scalar[HMX_DTYPE]](HMX_TILE * HMX_TILE)
    var W = HmxRows2D(
        ptr=want.as_unsafe_any_origin(),
        layout=row_major(Coord(HMX_TILE, HMX_TILE)),
    )
    for m in range(HMX_TILE):
        for n in range(HMX_TILE):
            var acc = Scalar[DType.float32](0)
            for k in range(K):
                acc += (
                    A.load[1](Coord(m, k)).cast[DType.float32]()
                    * B.load[1](Coord(n, k)).cast[DType.float32]()
                )
            W.store[1](Coord(m, n), acc.cast[HMX_DTYPE]())

    if perturb:
        B.store[1](Coord(0, 0), B.load[1](Coord(0, 0)) + 1)

    var scratch = HmxScratch[GEN](
        act=hmx_panel[GEN](act_buf, k_tiles),
        wgt=hmx_panel[GEN](wgt_buf, k_tiles),
        out=hmx_panel[GEN](out_buf, 1),
        record=HmxRequantRecord[GEN](
            ptr=rec_buf, layout=row_major(Coord(2, HMX_RECORD_LANES))
        ),
    )
    var ran = matmul_bt_stripe_hmx(
        MMTile(
            ptr=c_buf.as_unsafe_any_origin(),
            layout=row_major(Coord(HMX_TILE, HMX_TILE)),
        ),
        MMTile(
            ptr=a_buf.as_unsafe_any_origin(),
            layout=row_major(Coord(HMX_TILE, K)),
        ),
        MMTile(
            ptr=b_buf.as_unsafe_any_origin(),
            layout=row_major(Coord(HMX_TILE, K)),
        ),
        0,
        HMX_TILE,
        scratch,
    )

    var bad = 0
    if not ran:
        bad = HMX_TILE * HMX_TILE
    else:
        for m in range(HMX_TILE):
            for n in range(HMX_TILE):
                if C.load[1](Coord(m, n)) != W.load[1](Coord(m, n)):
                    bad += 1

    a_buf.free()
    b_buf.free()
    c_buf.free()
    act_buf.free()
    wgt_buf.free()
    out_buf.free()
    rec_buf.free()
    want.free()
    return bad


comptime VTCM_GRANT_BYTES = 8 * 1024 * 1024
"""The grant this part hands out: `VTCM_PAGE` 8388608, `VTCM_COUNT` 1.

Order 157's measurement, restated here because it is what bounds the operand
probe's working set -- `mxmem` requires VTCM, so a geometry past this is not a
slower run, it is a run that cannot happen."""


comptime PORT_FOLD_VALUE = Scalar[HMX_DTYPE](1.5)
"""What a readout element holds in the fold check. `1.5` is `0x3E00` in fp16.

Any non-zero value works; what matters is that EVERY element holds the SAME one,
because that is what the probe's own readout looks like -- its operands vary
only along depth, so every output element is the same sum."""

comptime PORT_FOLD_WORD = 0x3E00
"""`PORT_FOLD_VALUE`'s bit pattern, stated so the expected sum is arithmetic the
test does rather than a number it copies from a run."""


def check_port_probe_fold() -> Bool:
    """Asserts the sink control can pass, can fail, and is not the XOR.

    THE CONTROL IS SHOWN CAPABLE OF FAILING, and that demonstration is part of
    the control rather than a note beside it: an unwritten tile folds to the
    failure value, a written one does not, and the fold this probe SHIPPED WITH
    folds the written tile to the failure value too.

    That last row is the defect order 159R ran with. The probe's readout is
    identical in every element by construction, so the XOR fold over 1024 of
    them was exactly zero -- the control reported "the array never wrote" on
    twelve successful rows, and could never have reported anything else. The
    rates were unharmed (the lane re-skeled with the sum fold and they moved by
    <= 0.004%), but the witness was decorative for the whole window.
    """
    var ok = True
    var buf = alloc[Scalar[HMX_DTYPE]](HMX_TILE_ELEMS)
    var tile = hmx_panel[GEN](buf, 1)

    # The failure input: nothing wrote here. This is what the sink is FOR.
    hmx_attn_zero_panel(tile, 1)
    if hmx_port_probe_fold(tile) != 0:
        print("  FAIL an unwritten readout does not fold to the failure value")
        ok = False

    # The success input, in the shape the probe actually produces.
    comptime ALIGN = align_of[Scalar[HMX_DTYPE]]()
    for slab in range(HMX_TILE):
        for lane in range(HMX_TILE):
            tile.store[1, alignment=ALIGN](
                Coord(0, slab // HMX_SPLIT, lane, slab % HMX_SPLIT),
                PORT_FOLD_VALUE,
            )
    var want = UInt32(HMX_TILE_ELEMS * PORT_FOLD_WORD)
    var got = hmx_port_probe_fold(tile)
    if got != want:
        print("  FAIL written readout folded to", got, "want", want)
        ok = False
    if got == 0:
        print("  FAIL the control reports failure on a written readout")
        ok = False

    # The rejected fold, over that same written tile. Locked so the defect
    # cannot come back as a one-character edit.
    var xor_fold: UInt32 = 0
    for slab in range(HMX_TILE):
        for lane in range(HMX_TILE):
            xor_fold ^= bitcast[DType.uint16, 1](
                tile.load[1, alignment=ALIGN](
                    Coord(0, slab // HMX_SPLIT, lane, slab % HMX_SPLIT)
                )
            ).cast[DType.uint32]()
    if xor_fold != 0:
        print("  FAIL the XOR fold is no longer degenerate here; recheck why")
        ok = False

    buf.free()
    if ok:
        print("  ok   operand-port sink control (passes, fails, not the XOR)")
    return ok


def check_port_probe_geometry() -> Bool:
    """Asserts the operand probe's denominator and its two forms' issue counts.

    The probe reports a rate, and a rate is only as good as the number under it.
    That number is never measured: it is `2 * tiles * HmxTileBytes` per pass,
    closed here so a wrong denominator is a failing host test rather than a
    plausible GB/s on a device.

    THE TWO FORMS MOVE THE SAME BYTES, and this is what makes leg 2 an
    isolation: `:deep` and plain differ in the packet count and in nothing else,
    so any rate difference between them is an issue-count effect. Asserted
    rather than assumed, because the whole comparison rests on it.
    """
    var ok = True
    comptime PAIR = 2 * HMX_TILE_BYTES

    for tiles in [1, 8, 32, 33, 128, 1920]:
        var want = UInt64(tiles) * UInt64(PAIR) * UInt64(7)
        if hmx_port_probe_bytes(tiles, 7) != want:
            print("  FAIL port bytes tiles=", tiles, "want", want)
            ok = False

    # 64-bit, and the reason is the device rather than taste: `Int` there is a
    # 32-bit pointer width, so a probe long enough to time can name more bytes
    # than it holds. The kernel refuses such a geometry; it can only do that if
    # this arithmetic did not wrap first.
    if hmx_port_probe_bytes(1920, 250000) <= UInt64(0xFFFFFFFF):
        print("  FAIL port bytes wrapped at a geometry the kernel must refuse")
        ok = False

    # One packet per 32 croutons against one per crouton. `33` is the boundary:
    # the deep arm must take a second packet there, which is where an off-by-one
    # in the cap would show as a denominator that still looked right.
    for tiles in [1, 32, 33, 64, 1920]:
        var deep = hmx_port_probe_issues[deep=True](tiles)
        var plain = hmx_port_probe_issues[deep=False](tiles)
        var want_deep = (tiles + 31) // 32
        if deep != want_deep or plain != tiles:
            print("  FAIL port issues tiles=", tiles, deep, plain)
            ok = False
        if tiles > 1 and deep >= plain:
            print("  FAIL port issues do not separate the forms at", tiles)
            ok = False

    # The largest geometry the dispatch sweeps, and the next one it does not:
    # the table stops at 1920 croutons because 2048 does not fit the grant
    # beside the readout tile and the record.
    if hmx_scratch_bytes(1920) > VTCM_GRANT_BYTES:
        print("  FAIL 1920 croutons no longer fit the grant")
        ok = False
    if hmx_scratch_bytes(2048) <= VTCM_GRANT_BYTES:
        print("  FAIL 2048 croutons now fit; the sweep's ceiling is stale")
        ok = False

    # THE FLOOR, ON BOTH SIDES. The denominator excludes the record and the
    # readout, so every rate is a floor understated by `0.5625/tiles`; below the
    # bound that understatement fires order 159R sec 7's below-984 CONTRADICTION
    # row on an instrument artifact, which is why the kernel refuses there.
    # Asserted at the bound AND one below it: a bound that only passes above
    # itself would be satisfied by a refusal that never admits anything.
    var floor_tiles = hmx_port_probe_min_tiles[HMX_DTYPE]()
    if floor_tiles != 29:
        print("  FAIL port floor moved to", floor_tiles, "- expected 29")
        ok = False
    if hmx_port_probe_excluded_ppm(floor_tiles) > HMX_PORT_PROBE_BUDGET_PPM:
        print("  FAIL the floor itself exceeds the documented budget")
        ok = False
    if (
        hmx_port_probe_excluded_ppm(floor_tiles - 1)
        <= HMX_PORT_PROBE_BUDGET_PPM
    ):
        print("  FAIL one below the floor is inside budget; the bound is loose")
        ok = False
    # The ratio the finding is stated in, at the geometry that produced it.
    if hmx_port_probe_excluded_ppm(8) != 70312:
        print("  FAIL 8 croutons no longer understate by 7.03%")
        ok = False
    # Every geometry the dispatch sweeps must clear the floor, or the table
    # prints a FAIL row the reader would have to interpret.
    for tiles in [29, 32, 64, 128, 512, 1920]:
        if tiles < floor_tiles:
            print("  FAIL swept geometry", tiles, "is below the floor")
            ok = False

    if ok:
        print("  ok   operand-port geometry (bytes, issues, residency)")
    return ok


def check_microtile(k_tiles: Int) -> Bool:
    """Asserts the modelled microtile reproduces `A @ B^T` exactly."""
    var bad = microtile_mismatches(k_tiles, False)
    if bad != 0:
        print("  FAIL microtile k_tiles=", k_tiles, "-", bad, "of 1024 differ")
        return False
    print("  ok   microtile k_tiles=", k_tiles, "(1024 outputs exact)")
    return True


def oracle_microtile_perturbed(k_tiles: Int) -> Bool:
    """MUST fire: a corrupted weight has to change what the microtile writes."""
    var bad = microtile_mismatches(k_tiles, True)
    if bad == 0:
        print("  FAIL oracle microtile-perturb did NOT fire")
        return False
    print("  ok   oracle microtile-perturb fired -", bad, "outputs differ")
    return True


def check_microtile_refusal() -> Bool:
    """Asserts a `K` the `k_vecs` round trip cannot carry is refused, not run.

    `K = HMX_TILE` is a whole number of croutons and still illegal, because the
    microtile recovers the depth from `k_vecs * MM_W` and 32 is half a vector.
    Refusing is the point: the alternative is an output tile of pure bias.
    """
    var K = HMX_TILE
    var buf = alloc[Scalar[HMX_DTYPE]](HMX_TILE * K)
    var c_buf = alloc[Scalar[HMX_DTYPE]](HMX_TILE * HMX_TILE)
    var act_buf = alloc[Scalar[HMX_DTYPE]](HMX_TILE_ELEMS)
    var wgt_buf = alloc[Scalar[HMX_DTYPE]](HMX_TILE_ELEMS)
    var out_buf = alloc[Scalar[HMX_DTYPE]](HMX_TILE_ELEMS)
    var rec_buf = alloc[Scalar[DType.uint32]](2 * HMX_RECORD_LANES)

    var C = HmxRows2D(
        ptr=c_buf.as_unsafe_any_origin(),
        layout=row_major(Coord(HMX_TILE, HMX_TILE)),
    )
    for r in range(HMX_TILE):
        for c in range(HMX_TILE):
            C.store[1](Coord(r, c), SENTINEL)

    var scratch = HmxScratch[GEN](
        act=hmx_panel[GEN](act_buf, 1),
        wgt=hmx_panel[GEN](wgt_buf, 1),
        out=hmx_panel[GEN](out_buf, 1),
        record=HmxRequantRecord[GEN](
            ptr=rec_buf, layout=row_major(Coord(2, HMX_RECORD_LANES))
        ),
    )
    var ran = matmul_bt_stripe_hmx(
        MMTile(
            ptr=c_buf.as_unsafe_any_origin(),
            layout=row_major(Coord(HMX_TILE, HMX_TILE)),
        ),
        MMTile(
            ptr=buf.as_unsafe_any_origin(), layout=row_major(Coord(HMX_TILE, K))
        ),
        MMTile(
            ptr=buf.as_unsafe_any_origin(), layout=row_major(Coord(HMX_TILE, K))
        ),
        0,
        HMX_TILE,
        scratch,
    )

    var touched = 0
    for r in range(HMX_TILE):
        for c in range(HMX_TILE):
            if C.load[1](Coord(r, c)) != SENTINEL:
                touched += 1

    buf.free()
    c_buf.free()
    act_buf.free()
    wgt_buf.free()
    out_buf.free()
    rec_buf.free()

    if ran:
        print("  FAIL refusal K=", K, "ran instead of refusing")
        return False
    if touched != 0:
        print("  FAIL refusal K=", K, "wrote", touched, "outputs")
        return False
    print("  ok   refusal K=", K, "refused, nothing written")
    return True


def stripe_through_carved_scratch(k_tiles: Int) -> Int:
    """Runs one stripe through `hmx_scratch_in`, scoring it against `A @ B^T`.

    The residency the skel supplies, exercised as the skel builds it: ONE block
    of `hmx_scratch_bytes` bytes with the base aligned up, carved into four
    views rather than allocated four times. A differential test rather than a
    bounds check, because the failure this arrangement can actually have is an
    overlap — an activation panel aliasing its weight panel is inside every
    bound and produces a wrong answer, which is what this counts.

    Returns the number of output elements that disagree with the reference.
    """
    var K = k_tiles * HMX_TILE
    var a_buf = alloc[Scalar[HMX_DTYPE]](HMX_TILE * K)
    var b_buf = alloc[Scalar[HMX_DTYPE]](HMX_TILE * K)
    var c_buf = alloc[Scalar[HMX_DTYPE]](HMX_TILE * HMX_TILE)

    # Over-allocated by one tile so the base can be aligned up inside it, which
    # is what `_matmul_stripe_hmx` does to the VTCM grant.
    var raw = alloc[UInt8](hmx_scratch_bytes(k_tiles) + HMX_TILE_BYTES)
    var base = UnsafePointer[UInt8, MutUntrackedOrigin](
        unsafe_from_address=align_up(Int(raw), HMX_TILE_BYTES)
    )

    var A = HmxRows2D(
        ptr=a_buf.as_unsafe_any_origin(), layout=row_major(Coord(HMX_TILE, K))
    )
    var B = HmxRows2D(
        ptr=b_buf.as_unsafe_any_origin(), layout=row_major(Coord(HMX_TILE, K))
    )
    var C = HmxRows2D(
        ptr=c_buf.as_unsafe_any_origin(),
        layout=row_major(Coord(HMX_TILE, HMX_TILE)),
    )
    for r in range(HMX_TILE):
        for c in range(K):
            A.store[1](Coord(r, c), Scalar[HMX_DTYPE]((5 * r + 7 * c) % 13 - 6))
            B.store[1](Coord(r, c), Scalar[HMX_DTYPE]((3 * r + 11 * c) % 7 - 3))
        for c in range(HMX_TILE):
            C.store[1](Coord(r, c), SENTINEL)

    var ran = matmul_bt_stripe_hmx(
        MMTile(
            ptr=c_buf.as_unsafe_any_origin(),
            layout=row_major(Coord(HMX_TILE, HMX_TILE)),
        ),
        MMTile(
            ptr=a_buf.as_unsafe_any_origin(),
            layout=row_major(Coord(HMX_TILE, K)),
        ),
        MMTile(
            ptr=b_buf.as_unsafe_any_origin(),
            layout=row_major(Coord(HMX_TILE, K)),
        ),
        0,
        HMX_TILE,
        hmx_scratch_in[GEN](base, k_tiles),
    )

    var wrong = 0 if ran else HMX_TILE * HMX_TILE
    if ran:
        for m in range(HMX_TILE):
            for n in range(HMX_TILE):
                var acc = Scalar[DType.float32](0)
                for k in range(K):
                    var a = A.load[1](Coord(m, k)).cast[DType.float32]()
                    var b = B.load[1](Coord(n, k)).cast[DType.float32]()
                    acc += a * b
                if C.load[1](Coord(m, n)) != acc.cast[HMX_DTYPE]():
                    wrong += 1

    a_buf.free()
    b_buf.free()
    c_buf.free()
    raw.free()
    return wrong


def check_scratch_carve(k_tiles: Int) -> Bool:
    """Asserts a carved scratch computes the stripe, and its size covers it."""
    var want_bytes = (2 * k_tiles + 1) * HMX_TILE_BYTES + 256
    if hmx_scratch_bytes(k_tiles) != want_bytes:
        print(
            "  FAIL scratch_bytes(",
            k_tiles,
            ") =",
            hmx_scratch_bytes(k_tiles),
            "want",
            want_bytes,
        )
        return False
    var wrong = stripe_through_carved_scratch(k_tiles)
    if wrong != 0:
        print("  FAIL carved scratch k_tiles=", k_tiles, "wrong:", wrong)
        return False
    print("  ok   carved scratch k_tiles=", k_tiles)
    return True


def check_accepts() -> Bool:
    """Asserts the acceptance predicate and the stripe's own refusal agree.

    Two callers act on this predicate and cannot see each other: the stripe, and
    the kernel entry that must know the answer BEFORE it takes the HMX unit.
    Drift between them is a grant held for a stripe that then refuses, so the
    agreement is asserted rather than assumed.
    """
    var ok = True
    if not matmul_bt_hmx_accepts(64, 64, 2 * MM_W, 0, 64):
        print("  FAIL accepts rejected a conforming shape")
        ok = False
    # `K` a whole number of croutons but half a vector: the lossy round trip the
    # stripe refuses on, and the reason the bound is `MM_W` and not `HMX_TILE`.
    if matmul_bt_hmx_accepts(HMX_TILE, HMX_TILE, HMX_TILE, 0, HMX_TILE):
        print("  FAIL accepts allowed K =", HMX_TILE)
        ok = False
    if matmul_bt_hmx_accepts(HMX_TILE + 1, HMX_TILE, MM_W, 0, HMX_TILE):
        print("  FAIL accepts allowed a ragged M")
        ok = False
    if matmul_bt_hmx_accepts(HMX_TILE, HMX_TILE, MM_W, 0, HMX_TILE - 1):
        print("  FAIL accepts allowed a ragged stripe")
        ok = False
    if ok:
        print("  ok   acceptance predicate")
    return ok


def check_work_share(total: Int, parts: Int) -> Bool:
    """Asserts the feeder split is an exact partition (P5 M4.2).

    Every crouton a step stages, and every readout it writes back, is claimed by
    exactly one thread through this function. A share that overlaps its
    neighbour makes two threads write one crouton and a share that gaps leaves
    one unwritten, and BOTH show up on device as a wrong output rather than as
    an error, so the property is asserted here directly rather than only through
    the differential below.
    """
    var ok = True
    if hmx_work_share(total, 0, parts) != 0:
        print("  FAIL share start", total, parts)
        ok = False
    if hmx_work_share(total, parts, parts) != total:
        print("  FAIL share end", total, parts)
        ok = False
    var covered = 0
    for i in range(parts):
        var lo = hmx_work_share(total, i, parts)
        var hi = hmx_work_share(total, i + 1, parts)
        if hi < lo:
            print("  FAIL share not monotone at", i, total, parts)
            ok = False
        covered += hi - lo
    if covered != total:
        print("  FAIL share covers", covered, "of", total)
        ok = False
    if ok:
        print("  ok   work share total=", total, "parts=", parts)
    return ok


def fill_panel(panel: CroutonPanel[GEN], tiles: Int, value: Int):
    """Writes `value` into every element of `tiles` croutons of `panel`."""
    for t in range(tiles):
        for slab in range(HMX_TILE // HMX_SPLIT):
            for lane in range(HMX_TILE):
                for sub in range(HMX_SPLIT):
                    panel.store[1](
                        Coord(t, slab, lane, sub), Scalar[HMX_DTYPE](value)
                    )


def panel_holds(panel: CroutonPanel[GEN], tiles: Int, value: Int) -> Bool:
    """Whether every element of `tiles` croutons still reads `value`."""
    for t in range(tiles):
        for slab in range(HMX_TILE // HMX_SPLIT):
            for lane in range(HMX_TILE):
                for sub in range(HMX_SPLIT):
                    if panel.load[1](Coord(t, slab, lane, sub)) != Scalar[
                        HMX_DTYPE
                    ](value):
                        return False
    return True


def check_stripe_arena(k_tiles: Int, act_slots: Int) -> Bool:
    """Asserts the pipelined arena's sizing and that no two slots alias.

    The arena hands out `act_slots + HMX_WGT_RING + HMX_OUT_SETS * act_slots`
    independently addressed regions from one base, plus the guard crouton and
    the record. An overlap is inside every bound, produces a wrong answer rather
    than a fault, and is exactly what the double buffering exists to avoid — so
    every slot is stamped with its own value and every slot is then re-read.
    """
    var ok = True
    # The guard crouton is part of the arena's fixed cost: it sits between the
    # readout region and the record so a store that writes past its tile lands
    # on poison instead of on the requant file (order 132).
    var want = (
        act_slots * k_tiles
        + HMX_WGT_RING * k_tiles
        + HMX_OUT_SETS * act_slots
        + HMX_STRIPE_GUARD_CROUTONS
        + HMX_ARENA_STAGGER_CROUTONS
    ) * HMX_TILE_BYTES + 256
    var bytes = hmx_stripe_arena_bytes(k_tiles, act_slots)
    if bytes != want:
        print("  FAIL arena_bytes", bytes, "want", want)
        ok = False
    # The slot count is the closed-form inverse of the size, so a budget of
    # exactly one arena must yield exactly that arena and one byte less must not.
    if hmx_stripe_act_slots(k_tiles, bytes, act_slots + 8) != act_slots:
        print("  FAIL act_slots did not invert arena_bytes")
        ok = False
    if hmx_stripe_act_slots(k_tiles, bytes - 1, act_slots + 8) >= act_slots:
        print("  FAIL act_slots ignored a short budget")
        ok = False

    var raw = alloc[UInt8](bytes + HMX_TILE_BYTES)
    var base = UnsafePointer[UInt8, MutUntrackedOrigin](
        unsafe_from_address=align_up(Int(raw), HMX_TILE_BYTES)
    )
    var arena = hmx_stripe_arena_in[GEN](base, k_tiles, act_slots)
    for p in range(act_slots):
        fill_panel(arena.act_panel(p), k_tiles, p + 1)
    for b in range(HMX_WGT_RING):
        fill_panel(arena.wgt_panel(b), k_tiles, 100 + b)
    for b in range(HMX_OUT_SETS):
        for p in range(act_slots):
            fill_panel(arena.out_tile(b, p), 1, 200 + b * 16 + p)
    for p in range(act_slots):
        if not panel_holds(arena.act_panel(p), k_tiles, p + 1):
            print("  FAIL arena act panel", p, "was overwritten")
            ok = False
    for b in range(HMX_WGT_RING):
        if not panel_holds(arena.wgt_panel(b), k_tiles, 100 + b):
            print("  FAIL arena wgt slot", b, "was overwritten")
            ok = False
    for b in range(HMX_OUT_SETS):
        for p in range(act_slots):
            if not panel_holds(arena.out_tile(b, p), 1, 200 + b * 16 + p):
                print("  FAIL arena out slot", b, p, "was overwritten")
                ok = False

    # PANEL-MAJOR ADJACENCY, which the non-aliasing stamps above cannot see: a
    # buffer-major order passes every check so far and still puts one panel's
    # consecutive column tiles `act_slots` croutons apart, so a two-tile view of
    # them reads the next PANEL. That is a wrong answer with nothing to signal
    # it, and it is the property a paired readout store depends on.
    for p in range(act_slots):
        for b in range(HMX_OUT_SETS - 1):
            var here = Int(
                arena.out_tile(b, p).ptr_at_offset(Coord(0, 0, 0, 0))
            )
            var next = Int(
                arena.out_tile(b + 1, p).ptr_at_offset(Coord(0, 0, 0, 0))
            )
            if next - here != HMX_TILE_BYTES:
                print(
                    "  FAIL out sets not adjacent for panel",
                    p,
                    "set",
                    b,
                    "delta",
                    next - here,
                )
                ok = False
    raw.free()
    if ok:
        print("  ok   arena k_tiles=", k_tiles, "act_slots=", act_slots)
    return ok


def pipelined_mismatches[
    weight_native: Bool = False,
    feed_repacked: Bool = True,
    weight_xfer: Int = HMX_XFER_COPY,
    weight_await: Bool = True,
    fold_trailing: Bool = HMX_FOLD_TRAILING,
    fold_early: Bool = False,
](
    M: Int,
    N: Int,
    K: Int,
    act_slots: Int,
    array_first: Bool,
    drop_drain: Bool,
    feeders: Int,
) -> Int:
    """Scores the pipelined nest against the single-threaded one, bit for bit.

    THE PR'S OWN GATE, run on the host with the MMA modelled: P5 M4.2 changes
    which thread stages an operand and how many times, and nothing about how an
    output tile accumulates, so the two nests must agree to the last bit rather
    than to a tolerance.

    `array_first` is the hazard probe. The steps are separated by a rendezvous
    on device, so any order WITHIN a step is a legal interleaving; running the
    array thread before the feeders catches a feeder writing the slot the array
    is reading, and running it after catches the reverse. Both orders must
    produce the same bits as the single-threaded nest.

    `drop_drain` is the failing control: it runs one step short, which leaves the
    last column tile's readout in VTCM and never written out. A comparison that
    cannot fail proves nothing about the one that passes.

    `weight_native` runs the P5 M4.3 arm: the weight the stripe reads is the
    host-repacked column-panel-major artefact, staged by the twin's transport
    (a copy; the device's is the DMA engine) instead of gathered per column
    tile. The reference is unchanged, so the bar stays bit-identity — the host
    produced those croutons with the same transform the device used to.

    `feed_repacked` is that arm's failing control: hand the native arm the
    ROW-MAJOR weight, which is byte-for-byte what an untagged artefact would be,
    and the answer must be wrong. That is the failure the loader's provenance
    refusal exists to make impossible, so it has to be demonstrably reachable.

    `fold_trailing` runs the R4a schedule: the per-block trailing unpack rides
    the next block's stage step and only the last block drains. The fold moves
    WHEN the last unpack runs and no arithmetic, so the bar stays bit identity,
    both step orders. Under it `drop_drain` truncates the KEPT final drain —
    the last block's readout never lands — and must still fail. `fold_early`
    is the fold's own failing control (the unpack one step early, into the
    step whose array write it races); it must fail in the feeders-first order,
    where the stale set is read before the array fills it.
    """
    var k_tiles = hmx_k_tiles(K)
    var a_buf = alloc[Scalar[HMX_DTYPE]](M * K)
    var b_buf = alloc[Scalar[HMX_DTYPE]](N * K)
    var nat_buf = alloc[Scalar[HMX_DTYPE]](N * K)
    var ref_buf = alloc[Scalar[HMX_DTYPE]](M * N)
    var new_buf = alloc[Scalar[HMX_DTYPE]](M * N)

    var A = HmxRows2D(
        ptr=a_buf.as_unsafe_any_origin(), layout=row_major(Coord(M, K))
    )
    var B = HmxRows2D(
        ptr=b_buf.as_unsafe_any_origin(), layout=row_major(Coord(N, K))
    )
    # Periods coprime with each other and with the 32-wide tile: a period that
    # shares a factor with the tile can fold a transposed gather back onto the
    # right answer and hide the highest-probability bug (KB
    # `kernels/hmx-crouton-tile-layout`).
    for r in range(M):
        for c in range(K):
            A.store[1](Coord(r, c), Scalar[HMX_DTYPE]((5 * r + 7 * c) % 13 - 6))
    for r in range(N):
        for c in range(K):
            B.store[1](Coord(r, c), Scalar[HMX_DTYPE]((3 * r + 11 * c) % 7 - 3))
    for i in range(M * N):
        ref_buf[i] = SENTINEL
        new_buf[i] = SENTINEL

    # The artefact the offline repacker would have produced, built here from the
    # same bytes: same buffer length, same `[N, K]` binding, different order.
    hmx_repack_weight_cpm[GEN](
        hmx_panel[GEN](nat_buf, hmx_native_weight_croutons(N, K)), B, N, K
    )

    var Amm = MMTile(
        ptr=a_buf.as_unsafe_any_origin(), layout=row_major(Coord(M, K))
    )
    var Bmm = MMTile(
        ptr=b_buf.as_unsafe_any_origin(), layout=row_major(Coord(N, K))
    )
    var Bnat = MMTile(
        ptr=nat_buf.as_unsafe_any_origin(), layout=row_major(Coord(N, K))
    )
    var Cref = MMTile(
        ptr=ref_buf.as_unsafe_any_origin(), layout=row_major(Coord(M, N))
    )
    var Cnew = MMTile(
        ptr=new_buf.as_unsafe_any_origin(), layout=row_major(Coord(M, N))
    )

    var ref_raw = alloc[UInt8](hmx_scratch_bytes(k_tiles) + HMX_TILE_BYTES)
    var ref_base = UnsafePointer[UInt8, MutUntrackedOrigin](
        unsafe_from_address=align_up(Int(ref_raw), HMX_TILE_BYTES)
    )
    var ran = matmul_bt_stripe_hmx(
        Cref, Amm, Bmm, 0, N, hmx_scratch_in[GEN](ref_base, k_tiles)
    )

    var arena_bytes = hmx_stripe_arena_bytes(k_tiles, act_slots)
    var new_raw = alloc[UInt8](arena_bytes + HMX_TILE_BYTES)
    var new_base = UnsafePointer[UInt8, MutUntrackedOrigin](
        unsafe_from_address=align_up(Int(new_raw), HMX_TILE_BYTES)
    )
    # The device derives the plan from the VTCM grant; the test derives it from
    # the arena it just sized, so the same closed form is exercised either way.
    var plan = hmx_stripe_plan(M, N, K, arena_bytes)
    var arena = hmx_stripe_arena_in[GEN](new_base, plan.k_tiles, plan.act_slots)

    # `weight_dma` is off because the host has no DMA engine; that split is
    # exactly what this differential exists to bridge.
    comptime step_fn = matmul_bt_stripe_hmx_step[
        HMX_EMIT_MMA,
        weight_native,
        weight_xfer,
        weight_await,
        fold_trailing=fold_trailing,
        fold_early=fold_early,
    ]
    var Bstep = Bnat if (weight_native and feed_repacked) else Bmm

    var steps = plan.steps[fold_trailing]() - (1 if drop_drain else 0)
    var unpack_arm = HMX_UNPACK_ARM_SCALAR_OFF
    for step in range(steps):
        if array_first:
            step_fn(
                Cnew,
                Amm,
                Bstep,
                arena,
                plan,
                step,
                feeders,
                feeders,
                unpack_arm,
            )
        for t in range(feeders):
            step_fn(Cnew, Amm, Bstep, arena, plan, step, t, feeders, unpack_arm)
        if not array_first:
            step_fn(
                Cnew,
                Amm,
                Bstep,
                arena,
                plan,
                step,
                feeders,
                feeders,
                unpack_arm,
            )

    var wrong = 0
    if not ran or plan.act_slots != act_slots:
        wrong = M * N
    else:
        for m in range(M):
            for n in range(N):
                if Cnew.load[1](Coord(m, n)) != Cref.load[1](Coord(m, n)):
                    wrong += 1

    a_buf.free()
    b_buf.free()
    nat_buf.free()
    ref_buf.free()
    new_buf.free()
    ref_raw.free()
    new_raw.free()
    return wrong


def check_pipelined(
    M: Int, N: Int, K: Int, act_slots: Int, feeders: Int = 4
) -> Bool:
    """Asserts the pipelined nest is bit-identical in both step orders.

    `feeders` is the launch's participant count minus its array thread, so
    `feeders = 4` is the five the render and 082's harness both spawn. The
    smaller counts are P5 M4.2 order 084's graceful-degradation path: the work
    partition changes, the accumulation order per output tile does not, so the
    bar stays bit-identity.
    """
    var ok = True
    for first in [True, False]:
        var wrong = pipelined_mismatches(
            M, N, K, act_slots, first, False, feeders
        )
        if wrong != 0:
            print(
                "  FAIL pipelined M=",
                M,
                "N=",
                N,
                "K=",
                K,
                "slots=",
                act_slots,
                "array_first=",
                first,
                "wrong:",
                wrong,
            )
            ok = False
    if ok:
        print(
            "  ok   pipelined M=",
            M,
            "N=",
            N,
            "K=",
            K,
            "slots=",
            act_slots,
            "bit-identical both orders",
        )
    return ok


def oracle_pipelined_short(M: Int, N: Int, K: Int, act_slots: Int) -> Bool:
    """Asserts the differential FIRES when the drain step is dropped."""
    var wrong = pipelined_mismatches(M, N, K, act_slots, True, True, 4)
    if wrong == 0:
        print("  FAIL oracle: a stripe one step short still matched")
        return False
    print("  ok   oracle pipelined-short fired -", wrong, "elements differ")
    return True


def check_pipelined_folded(
    M: Int, N: Int, K: Int, act_slots: Int, feeders: Int = 4
) -> Bool:
    """Asserts the R4a folded schedule is bit-identical in both step orders.

    The fold moves WHEN each block's last readout is unpacked and nothing
    about how an output tile accumulates, so the bar is `check_pipelined`'s
    unchanged: bit identity against the single-threaded nest, under both
    interleavings — the feeders-first order is the one that would read a
    readout the fold had ordered wrongly.
    """
    var ok = True
    for first in [True, False]:
        var wrong = pipelined_mismatches[fold_trailing=True](
            M, N, K, act_slots, first, False, feeders
        )
        if wrong != 0:
            print(
                "  FAIL folded M=",
                M,
                "N=",
                N,
                "K=",
                K,
                "slots=",
                act_slots,
                "feeders=",
                feeders,
                "array_first=",
                first,
                "wrong:",
                wrong,
            )
            ok = False
    if ok:
        print(
            "  ok   folded M=",
            M,
            "N=",
            N,
            "K=",
            K,
            "slots=",
            act_slots,
            "feeders=",
            feeders,
            "bit-identical both orders",
        )
    return ok


def oracle_fold_short(M: Int, N: Int, K: Int, act_slots: Int) -> Bool:
    """Asserts the folded differential FIRES minus its kept final drain.

    The fold keeps exactly one trailing step — the last block's. Dropping it
    must leave that block's last column in VTCM, exactly as `drop_drain` does
    to the unfolded schedule; a fold whose truncation still matched would
    mean the kept step unpacks nothing and the schedule lost a readout.
    """
    var wrong = pipelined_mismatches[fold_trailing=True](
        M, N, K, act_slots, True, True, 4
    )
    if wrong == 0:
        print("  FAIL oracle: a folded stripe one step short still matched")
        return False
    print("  ok   oracle fold-short fired -", wrong, "elements differ")
    return True


def oracle_fold_early(M: Int, N: Int, K: Int, act_slots: Int) -> Bool:
    """Asserts the differential FIRES on the wrong fold, one step early.

    `fold_early` unpacks each block's last column in the very step the array
    writes it — the naive merge §S.8 item 6 parks. Feeders-first is the sweep
    that must catch it: sequenced ahead of the array, the feeder reads the
    same-parity readout set while it still holds the column from two steps
    back. Array-first would sequence the write ahead and pass by luck, which
    is why the failing order is pinned here rather than swept.
    """
    var wrong = pipelined_mismatches[fold_trailing=True, fold_early=True](
        M, N, K, act_slots, False, False, 4
    )
    if wrong == 0:
        print("  FAIL oracle: the early fold matched under feeders-first")
        return False
    print("  ok   oracle fold-early fired -", wrong, "elements differ")
    return True


def check_fold_census() -> Bool:
    """Pins the fold's rendezvous delta at the production shapes, off-device.

    The R4a window order's pre-registered step-0 numbers (RENDEZVOUS_DESIGN.md
    §S.2 item 4): both arms' closed forms asserted against the plan, and the
    per-launch crossing delta — exactly `blocks - 1` — printed per shape so
    zero device time is spent deriving it. Shapes are `check_render_plan`'s
    six render matmuls plus `check_encoder_plan`'s six encoder projections at
    the same 8 MiB grant.
    """
    comptime GRANT = 8 * 1024 * 1024
    var Ms = [1536, 1536, 1536, 1024, 1024, 1024, 512, 512, 512, 512, 512, 512]
    var Ks = [
        12288,
        3072,
        3072,
        9216,
        3072,
        3072,
        2560,
        2560,
        2560,
        4096,
        2560,
        9728,
    ]
    var Ns = [
        3072,
        18432,
        3072,
        3072,
        18432,
        3072,
        4096,
        1024,
        1024,
        2560,
        19456,
        2560,
    ]
    var ok = True
    for i in range(len(Ms)):
        var plan = hmx_stripe_plan(Ms[i], Ns[i], Ks[i], GRANT)
        var off = plan.steps[False]()
        var on = plan.steps[True]()
        if off != plan.blocks * (plan.n_tiles + 2):
            print("  FAIL census: unfolded closed form broke at shape", i)
            ok = False
        if on != plan.blocks * (plan.n_tiles + 1) + 1:
            print("  FAIL census: folded closed form broke at shape", i)
            ok = False
        if off - on != plan.blocks - 1:
            print(
                "  FAIL census M=",
                Ms[i],
                "N=",
                Ns[i],
                "K=",
                Ks[i],
                "delta",
                off - on,
                "want",
                plan.blocks - 1,
            )
            ok = False
        print(
            "  census M=",
            Ms[i],
            "N=",
            Ns[i],
            "K=",
            Ks[i],
            "blocks=",
            plan.blocks,
            "steps_off=",
            off,
            "steps_on=",
            on,
            "delta=",
            off - on,
        )
    if ok:
        print("  ok   fold census: delta is blocks-1 at all twelve shapes")
    return ok


def check_repack_cpm(N: Int, K: Int) -> Bool:
    """Asserts the artefact IS the device's weight packer, laid out column-major.

    Two claims, and the second is the one P5 M4.3 spends. Every column panel
    holds exactly the croutons `hmx_pack_panel[True]` produces for it — so the
    byte-order, surjectivity and coprime-fill oracles above are this repacker's
    oracles unchanged, which is the whole argument for relocating the packer
    rather than writing one. And panel `ct` occupies the CONTIGUOUS crouton
    range `[ct * k_tiles, (ct + 1) * k_tiles)`, which is what makes one weight
    panel one linear DMA and not a chained 2D descriptor.

    Read through a FLAT view of the artefact, so this pins where a byte landed
    rather than asserting that some self-consistent order happened.
    """
    var k_tiles = hmx_k_tiles(K)
    var n_tiles = N // HMX_TILE
    var panel_elems = k_tiles * HMX_TILE_ELEMS
    var src_buf = alloc[Scalar[HMX_DTYPE]](N * K)
    var art_buf = alloc[Scalar[HMX_DTYPE]](N * K)
    var one_buf = alloc[Scalar[HMX_DTYPE]](panel_elems)
    var src = HmxRows2D(
        ptr=src_buf.as_unsafe_any_origin(), layout=row_major(Coord(N, K))
    )
    var art = HmxRows2D(
        ptr=art_buf.as_unsafe_any_origin(), layout=row_major(Coord(1, N * K))
    )
    var one = HmxRows2D(
        ptr=one_buf.as_unsafe_any_origin(),
        layout=row_major(Coord(1, panel_elems)),
    )
    fill_coprime(src, N, K)
    for i in range(N * K):
        art.store[1](Coord(0, i), SENTINEL)

    hmx_repack_weight_cpm[GEN](
        hmx_panel[GEN](art_buf, hmx_native_weight_croutons(N, K)), src, N, K
    )

    var holes = 0
    for i in range(N * K):
        if art.load[1](Coord(0, i)) == SENTINEL:
            holes += 1

    var bad = 0
    for ct in range(n_tiles):
        hmx_pack_panel[True, GEN](
            hmx_panel[GEN](one_buf, k_tiles), src, ct * HMX_TILE, 0, k_tiles
        )
        for i in range(panel_elems):
            if art.load[1](Coord(0, ct * panel_elems + i)) != one.load[1](
                Coord(0, i)
            ):
                bad += 1

    src_buf.free()
    art_buf.free()
    one_buf.free()
    if holes != 0 or bad != 0:
        print(
            "  FAIL repack N=", N, "K=", K, "-", holes, "unwritten,", bad, "off"
        )
        return False
    print("  ok   repack N=", N, "K=", K, "column-panel-major, contiguous")
    return True


def oracle_repack_order(N: Int, K: Int) -> Bool:
    """MUST fire: the artefact read in the OTHER tile order.

    `(kt * n_tiles + ct)` instead of `(ct * k_tiles + kt)` — row-tile-major,
    the layout a reader would get by transposing the two tile axes. It has the
    same length and the same croutons, so nothing but this check distinguishes
    it, and a panel would then be `n_tiles` strided fetches rather than one
    linear transfer. If it ever stops firing, `check_repack_cpm` is asserting a
    permutation nobody constrained.
    """
    var k_tiles = hmx_k_tiles(K)
    var n_tiles = N // HMX_TILE
    var panel_elems = k_tiles * HMX_TILE_ELEMS
    var src_buf = alloc[Scalar[HMX_DTYPE]](N * K)
    var art_buf = alloc[Scalar[HMX_DTYPE]](N * K)
    var one_buf = alloc[Scalar[HMX_DTYPE]](panel_elems)
    var src = HmxRows2D(
        ptr=src_buf.as_unsafe_any_origin(), layout=row_major(Coord(N, K))
    )
    var art = HmxRows2D(
        ptr=art_buf.as_unsafe_any_origin(), layout=row_major(Coord(1, N * K))
    )
    var one = HmxRows2D(
        ptr=one_buf.as_unsafe_any_origin(),
        layout=row_major(Coord(1, panel_elems)),
    )
    fill_coprime(src, N, K)
    hmx_repack_weight_cpm[GEN](
        hmx_panel[GEN](art_buf, hmx_native_weight_croutons(N, K)), src, N, K
    )

    var bad = 0
    for ct in range(n_tiles):
        hmx_pack_panel[True, GEN](
            hmx_panel[GEN](one_buf, k_tiles), src, ct * HMX_TILE, 0, k_tiles
        )
        for kt in range(k_tiles):
            for i in range(HMX_TILE_ELEMS):
                var flat = (kt * n_tiles + ct) * HMX_TILE_ELEMS + i
                if art.load[1](Coord(0, flat)) != one.load[1](
                    Coord(0, kt * HMX_TILE_ELEMS + i)
                ):
                    bad += 1

    src_buf.free()
    art_buf.free()
    one_buf.free()
    if bad == 0:
        print("  FAIL oracle repack-order did NOT fire (N=", N, "K=", K, ")")
        return False
    print("  ok   oracle repack-order fired -", bad, "elements differ")
    return True


def check_native_stripe(
    M: Int, N: Int, K: Int, act_slots: Int, feeders: Int = 4
) -> Bool:
    """Asserts the weight-native arm is bit-identical to the packing arm.

    THE PR'S GATE (P5 M4.3), the same bar P5 M4.2's was: the host produced the
    weight croutons with the same transform the device used to, and every output
    tile still reaches `hmx_mma_panel` as one `HmxScratch` with the same
    `k_tiles` and chunk loop, so nothing may move but which code put the bytes
    in the ring slot. Both step orders, because a step's internal order is a
    legal interleaving either way.

    Run over BOTH twin transports (P5 M4.3 order 100). `HMX_XFER_COPY` delivers
    at submit and so is correct whether or not anything waits;
    `HMX_XFER_DEFERRED` delivers only at the completion the stripe now performs,
    and poisons the slot at submit the way an engine mid-write would — so it
    also exercises the overwrite edge, since a schedule that staged into the
    slot the array is reading would read poison rather than weights.

    What this still cannot prove is that the ENGINE publishes on the same
    boundary the shim's completion poll reads. That is the device order's first
    leg, and it is why `oracle_native_no_await` exists: it fixes the meaning of
    a missing completion so the silicon result can be read against it.
    """
    var ok = True
    for first in [True, False]:
        var wrong = pipelined_mismatches[True, True, HMX_XFER_COPY, True](
            M, N, K, act_slots, first, False, feeders
        )
        var deferred = pipelined_mismatches[
            True, True, HMX_XFER_DEFERRED, True
        ](M, N, K, act_slots, first, False, feeders)
        if deferred != 0:
            print(
                "  FAIL native-deferred M=",
                M,
                "N=",
                N,
                "K=",
                K,
                "array_first=",
                first,
                "wrong:",
                deferred,
            )
            ok = False
        if wrong != 0:
            print(
                "  FAIL native M=",
                M,
                "N=",
                N,
                "K=",
                K,
                "slots=",
                act_slots,
                "array_first=",
                first,
                "wrong:",
                wrong,
            )
            ok = False
    if ok:
        print(
            "  ok   native M=",
            M,
            "N=",
            N,
            "K=",
            K,
            "slots=",
            act_slots,
            "feeders=",
            feeders,
            "bit-identical to the packing arm",
        )
    return ok


def oracle_native_unrepacked(M: Int, N: Int, K: Int, act_slots: Int) -> Bool:
    """MUST fire: the native arm fed the ROW-MAJOR weight.

    Byte for byte what an untagged artefact is, and the same length, so nothing
    downstream can notice — the answer is simply wrong. This is the failure the
    loader's layout tag refuses by name, and a refusal whose failure mode cannot
    be demonstrated is a refusal nobody can size.
    """
    var wrong = pipelined_mismatches[True, False](
        M, N, K, act_slots, True, False, 4
    )
    if wrong == 0:
        print("  FAIL oracle: an unrepacked weight still matched")
        return False
    print("  ok   oracle native-unrepacked fired -", wrong, "elements differ")
    return True


def oracle_native_no_await(M: Int, N: Int, K: Int, act_slots: Int) -> Bool:
    """MUST fire: the native arm that starts its transfers and never waits.

    THE CONTROL FOR ORDER 100'S FIX. The deferred transport delivers at the
    completion call, so withholding that call is exactly "the array read the
    slot before the transfer into it landed" — the silicon signature the
    device leg found (four runs, four outputs, against a control stable over
    seven). If this ever stops firing, the completion has stopped being
    load-bearing in the twin and the differential above no longer constrains
    the thing that was actually broken.

    On silicon the same withheld wait is nondeterministic; here it is
    deterministic, because the twin's threads are a loop rather than five
    hardware contexts. That is the point of modelling it: a race that only
    reproduces sometimes cannot be a gate, and this one always fires.
    """
    var wrong = pipelined_mismatches[True, True, HMX_XFER_DEFERRED, False](
        M, N, K, act_slots, True, False, 4
    )
    if wrong == 0:
        print("  FAIL oracle: a stripe that never waits for its DMA matched")
        return False
    print("  ok   oracle native-no-await fired -", wrong, "elements differ")
    return True


def oracle_native_stale_dstate(M: Int, N: Int, K: Int, act_slots: Int) -> Bool:
    """MUST fire: an await that exits on a completion it read from a stale line.

    ORDER 102'S THIRD WORLD, made concrete. #313's await is unobservable by
    construction — it drops its return and the shim's bound is silent — so
    "the await waited and the race is downstream" and "the await never waited"
    are the same observation on the box. Here they are not: this transport
    exits the completion without delivering, exactly as a poll of a cache line
    a neighbouring submitter resurrected would, and the differential must catch
    it.

    It is the SAME schedule as `HMX_XFER_DEFERRED`, which passes. Only the
    completion's honesty differs, which is what makes this pair the twin's
    model of the fix rather than a second bug.
    """
    var wrong = pipelined_mismatches[True, True, HMX_XFER_STALE, True](
        M, N, K, act_slots, True, False, 4
    )
    if wrong == 0:
        print("  FAIL oracle: a stale completion read still matched")
        return False
    print("  ok   oracle native-stale-dstate fired -", wrong, "elements differ")
    return True


def oracle_native_posted_write(M: Int, N: Int, K: Int, act_slots: Int) -> Bool:
    """MUST fire: an honest completion whose payload tail has not landed.

    ORDER 103'S MECHANISM. #313 ordered the completion flag and #315 unshared
    the line it lives on; both act on the descriptor. This models the half
    neither touched -- the engine publishing `dstate` before its last payload
    writes reach VTCM through the L2 -- and it is the reading the lane's own
    discriminator points at, because a deterministic visibility bug would be
    stable-and-wrong while the box is unstable-and-wrong.

    The fix this oracle stands for is not a longer wait: the flag was not
    lying. It is the reference's per-side bypass rule
    (`htp/dma-queue.h:169-170`), which stops routing a VTCM destination through
    the cache so that completion and residency are the same event.
    """
    var wrong = pipelined_mismatches[True, True, HMX_XFER_POSTED, True](
        M, N, K, act_slots, True, False, 4
    )
    if wrong == 0:
        print("  FAIL oracle: a posted payload write still matched")
        return False
    print("  ok   oracle native-posted-write fired -", wrong, "elements differ")
    return True


def native_io_mismatches[
    activation_native: Bool,
    output_native: Bool,
    feed_repacked_a: Bool = True,
    compare_c_native: Bool = output_native,
](M: Int, N: Int, K: Int, act_slots: Int, array_first: Bool) -> Tuple[
    Int, UInt32
]:
    """Scores the activation/output-native arms against the reference nest.

    Annihilation rung 1's stripe-level gate, `pipelined_mismatches`' shape with
    the I/O sides swapped instead of the weight side. The reference is
    `matmul_bt_stripe_hmx` over row-major everything; the treatment reads `A`
    from a row-panel-major crouton artefact (`activation_native`) and/or
    writes `C` as one (`output_native`). Nothing about accumulation moves —
    the native arms change WHERE bytes come from and go to, never which bytes
    reach `hmx_mma_panel` — so the bar is bit-identity, as the weight arm's is.

    Under `output_native` the treatment's `C` croutons are compared against
    the REFERENCE output repacked by `hmx_repack_rows_rpm` (the definition of
    the layout), so the arm under test appears on one side only.

    `feed_repacked_a=False` is the activation side's failing control: hand the
    native arm the ROW-MAJOR activation bytes — what an unstaged buffer is —
    and the answer must be wrong. `compare_c_native=False` under
    `output_native` is the output side's: read the crouton `C` as if it were
    row-major, the consumer-side mispairing, and the comparison must fire.

    Returns:
        The mismatch count and the last `unpack_arm` code the treatment
        reported, so the caller can pin `HMX_UNPACK_ARM_NATIVE` on exactly the
        arm that claims it.
    """
    var k_tiles = hmx_k_tiles(K)
    var a_buf = alloc[Scalar[HMX_DTYPE]](M * K)
    var a_nat_buf = alloc[Scalar[HMX_DTYPE]](M * K)
    var b_buf = alloc[Scalar[HMX_DTYPE]](N * K)
    var ref_buf = alloc[Scalar[HMX_DTYPE]](M * N)
    var ref_nat_buf = alloc[Scalar[HMX_DTYPE]](M * N)
    var new_buf = alloc[Scalar[HMX_DTYPE]](M * N)

    var A = HmxRows2D(
        ptr=a_buf.as_unsafe_any_origin(), layout=row_major(Coord(M, K))
    )
    var B = HmxRows2D(
        ptr=b_buf.as_unsafe_any_origin(), layout=row_major(Coord(N, K))
    )
    for r in range(M):
        for c in range(K):
            A.store[1](Coord(r, c), Scalar[HMX_DTYPE]((5 * r + 7 * c) % 13 - 6))
    for r in range(N):
        for c in range(K):
            B.store[1](Coord(r, c), Scalar[HMX_DTYPE]((3 * r + 11 * c) % 7 - 3))
    for i in range(M * N):
        ref_buf[i] = SENTINEL
        new_buf[i] = SENTINEL

    # The artefact the producing matmul's `output_native` (or a host repack)
    # would have left: same buffer length, same `[M, K]` binding, row-panel-
    # major order.
    hmx_repack_rows_rpm[GEN](
        hmx_panel[GEN](a_nat_buf, hmx_native_row_croutons(M, K)), A, M, K
    )

    var Amm = MMTile(
        ptr=a_buf.as_unsafe_any_origin(), layout=row_major(Coord(M, K))
    )
    var Anat = MMTile(
        ptr=a_nat_buf.as_unsafe_any_origin(), layout=row_major(Coord(M, K))
    )
    var Bmm = MMTile(
        ptr=b_buf.as_unsafe_any_origin(), layout=row_major(Coord(N, K))
    )
    var Cref = MMTile(
        ptr=ref_buf.as_unsafe_any_origin(), layout=row_major(Coord(M, N))
    )
    var Cnew = MMTile(
        ptr=new_buf.as_unsafe_any_origin(), layout=row_major(Coord(M, N))
    )

    var ref_raw = alloc[UInt8](hmx_scratch_bytes(k_tiles) + HMX_TILE_BYTES)
    var ref_base = UnsafePointer[UInt8, MutUntrackedOrigin](
        unsafe_from_address=align_up(Int(ref_raw), HMX_TILE_BYTES)
    )
    var ran = matmul_bt_stripe_hmx(
        Cref, Amm, Bmm, 0, N, hmx_scratch_in[GEN](ref_base, k_tiles)
    )
    var ref_nat = hmx_panel[GEN](ref_nat_buf, hmx_native_row_croutons(M, N))
    hmx_repack_rows_rpm[GEN](ref_nat, Cref, M, N)

    var arena_bytes = hmx_stripe_arena_bytes(k_tiles, act_slots)
    var new_raw = alloc[UInt8](arena_bytes + HMX_TILE_BYTES)
    var new_base = UnsafePointer[UInt8, MutUntrackedOrigin](
        unsafe_from_address=align_up(Int(new_raw), HMX_TILE_BYTES)
    )
    var plan = hmx_stripe_plan(M, N, K, arena_bytes)
    var arena = hmx_stripe_arena_in[GEN](new_base, plan.k_tiles, plan.act_slots)

    comptime step_fn = matmul_bt_stripe_hmx_step[
        weight_native=False,
        weight_xfer=HMX_XFER_COPY,
        activation_native=activation_native,
        output_native=output_native,
    ]
    var Astep = Anat if (activation_native and feed_repacked_a) else Amm
    var feeders = 4
    var unpack_arm = HMX_UNPACK_ARM_SCALAR_OFF
    for step in range(plan.steps()):
        if array_first:
            step_fn(
                Cnew,
                Astep,
                Bmm,
                arena,
                plan,
                step,
                feeders,
                feeders,
                unpack_arm,
            )
        for t in range(feeders):
            step_fn(Cnew, Astep, Bmm, arena, plan, step, t, feeders, unpack_arm)
        if not array_first:
            step_fn(
                Cnew,
                Astep,
                Bmm,
                arena,
                plan,
                step,
                feeders,
                feeders,
                unpack_arm,
            )

    var wrong = 0
    if not ran or plan.act_slots != act_slots:
        wrong = M * N
    else:
        comptime if compare_c_native:
            var got = hmx_panel[GEN](new_buf, hmx_native_row_croutons(M, N))
            for t in range(hmx_native_row_croutons(M, N)):
                for s in range(HMX_TILE // HMX_SPLIT):
                    for lane in range(HMX_TILE):
                        for sub in range(HMX_SPLIT):
                            var g = got.load[1, alignment=2](
                                Coord(t, s, lane, sub)
                            )
                            var w = ref_nat.load[1, alignment=2](
                                Coord(t, s, lane, sub)
                            )
                            if g != w:
                                wrong += 1
        else:
            for m in range(M):
                for n in range(N):
                    if Cnew.load[1](Coord(m, n)) != Cref.load[1](Coord(m, n)):
                        wrong += 1

    a_buf.free()
    a_nat_buf.free()
    b_buf.free()
    ref_buf.free()
    ref_nat_buf.free()
    new_buf.free()
    ref_raw.free()
    new_raw.free()
    return (wrong, unpack_arm)


def check_native_io[
    activation_native: Bool, output_native: Bool
](M: Int, N: Int, K: Int, act_slots: Int, label: String) -> Bool:
    """Asserts a native-I/O arm is bit-identical to the reference nest, both
    step orders, and that `output_native` reports its own arm code."""
    var ok = True
    for first in [True, False]:
        var got = native_io_mismatches[activation_native, output_native](
            M, N, K, act_slots, first
        )
        if got[0] != 0:
            print(
                "  FAIL",
                label,
                "M=",
                M,
                "N=",
                N,
                "K=",
                K,
                "array_first=",
                first,
                "wrong:",
                got[0],
            )
            ok = False
        comptime if output_native:
            if got[1] != HMX_UNPACK_ARM_NATIVE:
                print(
                    "  FAIL",
                    label,
                    "unpack_arm =",
                    got[1],
                    "want NATIVE =",
                    HMX_UNPACK_ARM_NATIVE,
                )
                ok = False
    if ok:
        print(
            "  ok  ",
            label,
            "M=",
            M,
            "N=",
            N,
            "K=",
            K,
            "slots=",
            act_slots,
            "bit-identical both orders",
        )
    return ok


def oracle_native_act_unrepacked(
    M: Int, N: Int, K: Int, act_slots: Int
) -> Bool:
    """MUST fire: the activation-native arm fed the ROW-MAJOR activation.

    Byte for byte what an unstaged activation buffer is, at the same length —
    the weights-native mismatch class on its fourth family member, and the
    reason rung 2's production wiring owes a staging tooth before any graph
    path may set the flag.
    """
    var got = native_io_mismatches[True, False, False](M, N, K, act_slots, True)
    if got[0] == 0:
        print("  FAIL oracle: an unrepacked activation still matched")
        return False
    print(
        "  ok   oracle native-act-unrepacked fired -",
        got[0],
        "elements differ",
    )
    return True


def oracle_native_out_misread(M: Int, N: Int, K: Int, act_slots: Int) -> Bool:
    """MUST fire: a crouton `C` read as if it were row-major.

    The consumer-side half of the same class: `output_native`'s buffer handed
    to a reader that never learned the layout. Same bytes, same length, wrong
    numbers — demonstrably, or the bit-identity passes above constrain
    nothing.
    """
    var got = native_io_mismatches[False, True, True, False](
        M, N, K, act_slots, True
    )
    if got[0] == 0:
        print("  FAIL oracle: a crouton C read as row-major still matched")
        return False
    print("  ok   oracle native-out-misread fired -", got[0], "elements differ")
    return True


def check_wgt_ring(k_tiles: Int, act_slots: Int) -> Bool:
    """Asserts the weight ring's slots and the schedule's disjointness.

    The ring costs `HMX_WGT_RING * k_tiles` croutons of the arena's fixed half,
    its slots must not alias, and the schedule — stage column tile
    `j + HMX_WGT_RING - 1` while the array reads `j` — must never write the slot
    the array is reading. That last one is false at a depth of one and is the
    property any deeper ring has to keep.
    """
    var ok = True
    var fixed = hmx_stripe_arena_bytes(k_tiles, 0)
    var want_fixed = (
        HMX_WGT_RING * k_tiles
        + HMX_STRIPE_GUARD_CROUTONS
        + HMX_ARENA_STAGGER_CROUTONS
    ) * HMX_TILE_BYTES + 256
    if fixed != want_fixed:
        print("  FAIL ring fixed cost", fixed, "want", want_fixed)
        ok = False

    var bytes = hmx_stripe_arena_bytes(k_tiles, act_slots)
    var raw = alloc[UInt8](bytes + HMX_TILE_BYTES)
    var base = UnsafePointer[UInt8, MutUntrackedOrigin](
        unsafe_from_address=align_up(Int(raw), HMX_TILE_BYTES)
    )
    var arena = hmx_stripe_arena_in[GEN](base, k_tiles, act_slots)
    for b in range(HMX_WGT_RING):
        fill_panel(arena.wgt_panel(b), k_tiles, 100 + b)
    for b in range(HMX_WGT_RING):
        if not panel_holds(arena.wgt_panel(b), k_tiles, 100 + b):
            print("  FAIL ring slot", b, "aliases another")
            ok = False
    # Several column tiles, past the wrap in both the reader's and the stager's
    # index, so a modulo that is right only at zero cannot pass.
    for j in range(2 * HMX_WGT_RING + 1):
        fill_panel(arena.wgt_panel(j), k_tiles, 7)
        fill_panel(arena.wgt_panel(j + HMX_WGT_RING - 1), k_tiles, 9)
        if not panel_holds(arena.wgt_panel(j), k_tiles, 7):
            print(
                "  FAIL ring: staging",
                j + HMX_WGT_RING - 1,
                "clobbered the slot the array reads at",
                j,
            )
            ok = False
    raw.free()
    if ok:
        print(
            "  ok   weight ring depth=",
            HMX_WGT_RING,
            "k_tiles=",
            k_tiles,
            "slots distinct, schedule disjoint",
        )
    return ok


def check_render_plan() -> Bool:
    """Asserts the plan's closed form at the six render matmul shapes.

    Cross-checks the kernel's own arithmetic against `CACHE_ROOFLINE_AUDIT.md`
    §1.2, which is where the weight traffic this milestone attacks was counted:
    `blocks` is the B re-read factor, so a plan that disagrees with the table
    means the 18.9 GB/step the design is sized on was never this kernel's.

    `act_slots` is asserted only where the budget binds. Two of the six want
    fewer row tiles than the budget holds, and the table records the uncapped
    closed form for them.
    """
    comptime GRANT = 8 * 1024 * 1024
    var Ms = [1536, 1536, 1536, 1024, 1024, 1024]
    var Ks = [12288, 3072, 3072, 9216, 3072, 3072]
    var Ns = [3072, 18432, 3072, 3072, 18432, 3072]
    var want_blocks = [6, 2, 2, 3, 1, 1]
    var want_slots = [8, 39, 39, 12, 0, 0]
    var ok = True
    for i in range(len(Ms)):
        var plan = hmx_stripe_plan(Ms[i], Ns[i], Ks[i], GRANT)
        if plan.blocks != want_blocks[i]:
            print(
                "  FAIL plan M=",
                Ms[i],
                "K=",
                Ks[i],
                "N=",
                Ns[i],
                "blocks",
                plan.blocks,
                "want",
                want_blocks[i],
            )
            ok = False
        if want_slots[i] != 0 and plan.act_slots != want_slots[i]:
            print(
                "  FAIL plan M=",
                Ms[i],
                "K=",
                Ks[i],
                "N=",
                Ns[i],
                "act_slots",
                plan.act_slots,
                "want",
                want_slots[i],
            )
            ok = False
    if ok:
        print(
            "  ok   render plan at depth",
            HMX_WGT_RING,
            "matches CACHE_ROOFLINE_AUDIT 1.2 at all six shapes",
        )
    return ok


def check_encoder_plan() -> Bool:
    """Asserts the gate admits every Qwen3 encoder projection, and prices them.

    The encoder stage is 96% matmul by MAC (`m4-encoder/ANALYSIS.md` §4), so
    routing it to the array is a shape question before it is anything else, and
    order 118 leg 2 is why it is asked here rather than on device: the
    attention arm's block gate refused every harness shape and the refusal was
    only found in a device window. `mojo_matmul_f16` drops a refused shape to
    the vector nest silently by design, so a gate that says no costs a whole
    leg and reports as "no speedup".

    Geometry from `npu_qwen3_encoder_stack_dispatch.cpp`'s `klein` preset
    (hidden 2560, inter 9728, GQA 32:8 x 128, seq 512), one dispatch per
    projection, so `M` is the whole padded sequence.

    The rendezvous total is the second reason this is pinned. It is the
    kernel's own `hmx_stripe_arrivals` — the grant broadcast plus `steps()` —
    summed over six projections and 27 layers, and order 114 measured a barrier
    costing 0.51-0.77 of `NPU_SYNC_NAP_MICROS`. At the compiled 1020 us default
    that total is tens of seconds against a stage that is 63 s whole, which is
    what makes the nap knob part of this arm's recipe rather than a tuning
    afterthought.
    """
    comptime GRANT = 8 * 1024 * 1024
    comptime LAYERS = 27
    comptime PARTICIPANTS = 5
    var Ms = [512, 512, 512, 512, 512, 512]
    var Ks = [2560, 2560, 2560, 4096, 2560, 9728]
    var Ns = [4096, 1024, 1024, 2560, 19456, 2560]
    var want_blocks = [1, 1, 1, 1, 1, 2]
    var want_slots = [16, 16, 16, 16, 16, 11]
    var want_arrivals = 28620
    var ok = True
    var arrivals = 0
    for i in range(len(Ms)):
        if not matmul_bt_hmx_accepts(Ms[i], Ns[i], Ks[i], 0, Ns[i]):
            print(
                "  FAIL encoder gate refused M=",
                Ms[i],
                "K=",
                Ks[i],
                "N=",
                Ns[i],
            )
            ok = False
            continue
        var plan = hmx_stripe_plan(Ms[i], Ns[i], Ks[i], GRANT)
        if plan.blocks != want_blocks[i] or plan.act_slots != want_slots[i]:
            print(
                "  FAIL encoder plan M=",
                Ms[i],
                "K=",
                Ks[i],
                "N=",
                Ns[i],
                "blocks",
                plan.blocks,
                "slots",
                plan.act_slots,
                "want",
                want_blocks[i],
                want_slots[i],
            )
            ok = False
        arrivals += LAYERS * hmx_stripe_arrivals(
            PARTICIPANTS, True, True, plan.steps()
        )
    if ok and arrivals != want_arrivals:
        print("  FAIL encoder arrivals", arrivals, "want", want_arrivals)
        ok = False
    if ok:
        print(
            "  ok   encoder plan admits all six projections,",
            arrivals,
            "rendezvous per encode",
        )
    return ok


def check_stripe_roles() -> Bool:
    """Asserts the role plan is total over every count a launch can produce.

    P5 M4.2 order 084. `QuRTDeviceContext` admits any gapless mask up to
    `kNPUMaxLaunchThreads`, so the stripe must have an answer for each of those
    counts, and exactly one of them may be the array thread. A count with no
    array thread is a grant nobody holds; a count with two is two threads
    locking one unit.
    """
    var ok = True
    for participants in range(0, 9):
        var roles = hmx_stripe_roles(participants)
        if participants < 2:
            if roles.usable():
                print("  FAIL roles admitted", participants, "participants")
                ok = False
            continue
        if not roles.usable():
            print("  FAIL roles refused", participants, "participants")
            ok = False
            continue
        # Exactly one array thread, and every other index is a feeder.
        var arrays = 0
        var feeders = 0
        for t in range(participants):
            if t >= roles.feeders:
                arrays += 1
            else:
                feeders += 1
        if arrays != 1:
            print("  FAIL", participants, "participants ->", arrays, "arrays")
            ok = False
        if feeders != roles.feeders or feeders < 1:
            print("  FAIL", participants, "participants ->", feeders, "feeders")
            ok = False
        if roles.array_thread != participants - 1:
            print("  FAIL array thread is not the last participant")
            ok = False
    # The production count, spelled out: this is the split order 082 measured,
    # and the reason its numbers carry over unchanged.
    var five = hmx_stripe_roles(5)
    if five.feeders != 4 or five.array_thread != 4:
        print("  FAIL five participants is not 4 feeders + thread 4")
        ok = False
    if ok:
        print("  ok   role plan total over 0..8 participants")
    return ok


def stripe_arrivals_by_thread(
    participants: Int,
    accepts: Bool,
    granted: Bool,
    steps: Int,
    thread_idx: Int,
    diverge_at: Int,
) -> Int:
    """Rendezvous `thread_idx` makes, modelling `_matmul_stripe_hmx`'s flow.

    A MODEL of the kernel's control flow, not the kernel — the driver lives in
    `npu_kernel.mojo`, which a host test cannot import. What it reproduces is the
    only property that wedges: where each thread returns relative to the
    rendezvous.

    `diverge_at` is the mutation: a thread index that takes a refusal AFTER the
    others have started arriving, which is exactly what order 084's cold-VTCM
    race did (`qshimGetBaseVtcmAddr` acquires lazily with no once-guard, so on a
    process's first dispatch the losers of the race got NULL and returned while
    the winner waited). Negative disables it.
    """
    var roles = hmx_stripe_roles(participants)
    if not roles.usable() or not accepts:
        return 0
    if thread_idx == diverge_at:
        return 0
    if not granted:
        return 1
    return 1 + steps


def check_arrival_counts() -> Bool:
    """Asserts every participant makes the SAME number of rendezvous.

    This is the wedge condition itself, stated as a test: a rendezvous completes
    only when every participant arrives, so two threads disagreeing about how
    many times they meet is a device that stops. Order 084 spent 24 minutes and a
    reboot on one such disagreement.
    """
    var ok = True
    var counts = [1, 2, 3, 4, 5, 8]
    var steps = 7
    for i in range(len(counts)):
        var p = counts[i]
        # Four configurations, covering every exit the kernel has.
        for exit_kind in range(4):
            var accepts = exit_kind != 1
            var granted = exit_kind != 2
            var want = hmx_stripe_arrivals(p, accepts, granted, steps)
            for t in range(p):
                var got = stripe_arrivals_by_thread(
                    p, accepts, granted, steps, t, -1
                )
                if got != want:
                    print(
                        "  FAIL arrivals p=",
                        p,
                        "exit=",
                        exit_kind,
                        "thread=",
                        t,
                        "got",
                        got,
                        "want",
                        want,
                    )
                    ok = False
    if ok:
        print("  ok   arrivals equal on every participant, counts 1..8")
    return ok


def oracle_arrival_divergence() -> Bool:
    """Asserts the arrival test FIRES on order 084's own shape.

    One thread taking a refusal the others do not is the bug. If this control
    passes, `check_arrival_counts` is not testing anything.
    """
    var p = 5
    var steps = 7
    var want = hmx_stripe_arrivals(p, True, True, steps)
    var caught = False
    for t in range(p):
        if stripe_arrivals_by_thread(p, True, True, steps, t, 2) != want:
            caught = True
    if not caught:
        print("  FAIL oracle: a diverging thread went unnoticed")
        return False
    print("  ok   oracle arrival-divergence fired (thread 2 returned early)")
    return True


def check_depth_range() -> Bool:
    """Asserts the `:deep` extent is inclusive and caps one register at 32 tiles.

    A byte extent of `2048 * tiles` rather than `2048 * tiles - 1` is the shape
    of bug that streams one tile too many; the 16-bit ceiling is what forces K
    past 1024 to chunk at all, so both are asserted rather than commented.
    """
    var ok = True
    if hmx_deep_range(1) != 2047:
        print("  FAIL deep_range(1) =", hmx_deep_range(1), "want 2047")
        ok = False
    if hmx_deep_range(32) != 65535:
        print("  FAIL deep_range(32) =", hmx_deep_range(32), "want 65535")
        ok = False
    if hmx_k_tiles(3072) != 96:
        print("  FAIL k_tiles(3072) =", hmx_k_tiles(3072), "want 96")
        ok = False
    if ok:
        print("  ok   deep-range and k-tile arithmetic")
    return ok


# ===--------------------------------------------------------------------------===
# Attention on the array (P5 slices 2 and 3).
#
# The device arm splits in two: `hmx_attention.mojo` owns the residency, the two
# matmuls and the S/P seam, and `npu_kernel.mojo` owns the softmax arithmetic it
# shares with the HVX arm. Only the first half is host-linkable, so the driver
# below is the second half — the twin composes the shipped layout functions into
# the whole flash loop and puts its own scalar softmax between them.
#
# That boundary is exactly the claim this section can make: the crouton axis
# roles, the accumulator chain, the vdeal/vshuff seam and the fused diagonal
# rescale compute flash attention. It does NOT claim the device's `_exp_hvx`, the
# array's arithmetic, or the permute's behaviour on silicon — the first is
# already verified, and the other two are what the device orders are for.
# ===--------------------------------------------------------------------------===


comptime ATTN_NEG = Scalar[DType.float32](-3.0e38)
"""The running max before any key has been seen.

Finite rather than `-inf` so `m_old - m_new` underflows to zero through the same
subtraction the device does, instead of producing a NaN the twin would have to
special-case and the device would not."""


@fieldwise_init
struct AttnScore(ImplicitlyCopyable, Movable):
    """How far a flash run landed from the fp32 reference."""

    var cosine: Float64
    """Cosine similarity over the whole of `O`."""

    var max_rel: Float64
    """Largest per-element error, relative to that row's largest reference
    magnitude — the shape the design's tolerance leg uses, so a near-zero output
    element cannot manufacture a large relative error."""


def attn_fill(
    t: HmxRows2D, rows: Int, cols: Int, a: Int, b: Int, div: Int, gain: Int = 1
):
    """Fills with a bounded, coprime-period pattern.

    Periods coprime with each other and with `HMX_TILE`, so a transposed axis
    cannot fold to the same sum — the trap this file's header records — and the
    division keeps every score inside fp16's exact-integer range once `D`
    products have been summed.
    """
    for r in range(rows):
        for c in range(cols):
            t.store[1](
                Coord(r, c),
                Scalar[HMX_DTYPE](
                    Float32((a * r + b * c) % 13 - 6)
                    * Float32(gain)
                    / Float32(div)
                ),
            )


def attn_reference(
    Q: HmxRows2D,
    K: HmxRows2D,
    V: HmxRows2D,
    want: HmxRows2D,
    seq: Int,
    d: Int,
    head_off: Int,
    scale: Float32,
):
    """Dense attention for one head in fp32: the thing the chain must reproduce.

    Deliberately the textbook three-line form over the ORIGINAL operands — no
    blocking, no online recurrence, no crouton — so nothing it computes shares a
    mistake with the implementation under test.
    """
    for m in range(seq):
        var top = ATTN_NEG
        var scores = List[Scalar[DType.float32]]()
        for n in range(seq):
            var acc = Scalar[DType.float32](0)
            for k in range(d):
                acc += (
                    Q.load[1](Coord(m, head_off + k)).cast[DType.float32]()
                    * K.load[1](Coord(n, head_off + k)).cast[DType.float32]()
                )
            var s = acc * Scalar[DType.float32](scale)
            scores.append(s)
            if s > top:
                top = s
        var total = Scalar[DType.float32](0)
        for n in range(seq):
            scores[n] = exp(scores[n] - top)
            total += scores[n]
        for c in range(d):
            var acc = Scalar[DType.float32](0)
            for n in range(seq):
                acc += (
                    scores[n]
                    * V.load[1](Coord(n, head_off + c)).cast[DType.float32]()
                )
            want.store[1](
                Coord(m, head_off + c),
                (acc / total).cast[HMX_DTYPE](),
            )


def attn_compare(
    got: HmxRows2D,
    want: HmxRows2D,
    seq: Int,
    d: Int,
    head_off: Int,
) -> AttnScore:
    """Scores `got` against `want`: cosine over all of `O`, worst row-relative
    error.
    """
    var dot = Float64(0)
    var ng = Float64(0)
    var nr = Float64(0)
    var worst = Float64(0)
    for m in range(seq):
        var row_max = Float64(0)
        for c in range(d):
            var r = Float64(
                want.load[1](Coord(m, head_off + c)).cast[DType.float32]()
            )
            if abs(r) > row_max:
                row_max = abs(r)
        for c in range(d):
            var g = Float64(
                got.load[1](Coord(m, head_off + c)).cast[DType.float32]()
            )
            var r = Float64(
                want.load[1](Coord(m, head_off + c)).cast[DType.float32]()
            )
            dot += g * r
            ng += g * g
            nr += r * r
            if row_max > 0:
                var rel = abs(g - r) / row_max
                if rel > worst:
                    worst = rel
    var denom = sqrt(ng * nr)
    return AttnScore(cosine=(dot / denom) if denom > 0 else 0.0, max_rel=worst)


def attn_flash_twin(
    plan: HmxAttnPlan,
    arena: HmxAttnArena[GEN],
    Q: HmxRows2D,
    K: HmxRows2D,
    V: HmxRows2D,
    O: HmxRows2D,
    seq: Int,
    head_off: Int,
    scale: Scalar[HMX_DTYPE],
    reverse: Bool,
    mutation: Int,
) raises:
    """Drives one head's whole flash loop through the shipped array-arm pieces.

    The device's `_mha_flash_head_hmx` is this function with `_exp_hvx` in place
    of `exp` and the MMA issued rather than modelled; keeping the twin's copy
    here rather than importing the kernel's is what the module split costs, and
    what it buys is a driver that links on arm64 at all.

    `reverse` walks the key blocks from the end. The online recurrence must reach
    the same answer either way, which is the property the diagonal rescale exists
    to provide — so it is a differential over the ORDER, not a second shape.

    `mutation` is 0 for the real thing, and otherwise names a defect the controls
    below require to be visible: 1 drops the rescale from the accumulator chain,
    2 swaps the K and V crouton rules, 3 programs the QK^T requant with unit
    scale instead of `1/sqrt(D)`.
    """
    comptime SLAB_PAIRS = HMX_TILE // HMX_SPLIT
    var br = plan.br_tiles * HMX_TILE
    var bc = plan.bc_tiles * HMX_TILE
    var blocks = seq // bc

    hmx_attn_program_records(
        arena, Scalar[HMX_DTYPE](1) if mutation == 3 else scale
    )
    hmx_attn_reset_diag(arena, plan)

    var q0 = 0
    while q0 < seq:
        hmx_attn_pack_q(arena, plan, Q, q0, head_off)
        hmx_attn_zero_panel(arena.o, 2 * plan.br_tiles * plan.d_tiles)
        for i in range(br):
            arena.stats.store[1](Coord(ATTN_STAT_M, i), ATTN_NEG)
            arena.stats.store[1](
                Coord(ATTN_STAT_L, i), Scalar[DType.float32](0)
            )

        var cur = 0
        for step in range(blocks):
            var k0 = (blocks - 1 - step) * bc if reverse else step * bc

            if mutation == 2:
                for j in range(plan.bc_tiles):
                    hmx_pack_panel[False](
                        arena.k_run(plan, j),
                        K,
                        k0 + j * HMX_TILE,
                        head_off,
                        plan.d_tiles,
                    )
                for c in range(plan.d_tiles):
                    for j in range(plan.bc_tiles):
                        hmx_pack_panel[True](
                            arena.v_tile(plan, c, j),
                            V,
                            k0 + j * HMX_TILE,
                            head_off + c * HMX_TILE,
                            1,
                        )
            else:
                hmx_attn_pack_k(arena, plan, K, k0, head_off)
                hmx_attn_pack_v(arena, plan, V, k0, head_off)

            hmx_attn_qk_block[emit_hmx=False](arena, plan)

            for r in range(plan.br_tiles):
                for p in range(SLAB_PAIRS):
                    hmx_attn_scores_to_rows(arena, plan, r, p, 0)
                    for phase in range(HMX_SPLIT):
                        var local = HMX_SPLIT * p + phase
                        var row = r * HMX_TILE + local
                        var m_old = arena.stats.load[1](Coord(ATTN_STAT_M, row))
                        var l_old = arena.stats.load[1](Coord(ATTN_STAT_L, row))
                        var m_new = m_old
                        for i in range(bc):
                            var s = arena.rows.load[1](Coord(phase, i))
                            if s > m_new:
                                m_new = s
                        var corr = exp(m_old - m_new)
                        var total = Scalar[DType.float32](0)
                        for i in range(bc):
                            var pv = exp(
                                arena.rows.load[1](Coord(phase, i)) - m_new
                            )
                            arena.rows.store[1](Coord(phase, i), pv)
                            total += pv
                        arena.stats.store[1](Coord(ATTN_STAT_M, row), m_new)
                        arena.stats.store[1](
                            Coord(ATTN_STAT_L, row), l_old * corr + total
                        )
                        hmx_attn_set_diag(
                            arena, r, local, corr.cast[HMX_DTYPE]()
                        )
                    hmx_attn_probs_from_rows(arena, plan, r, p, 0)

            if mutation == 1:
                for r in range(plan.br_tiles):
                    for c in range(plan.d_tiles):
                        hmx_mma_chain[emit_hmx=False](
                            arena.o_tile(plan, 1 - cur, r, c),
                            arena.rec_id,
                            HmxOperands[GEN](
                                arena.p_run(plan, r),
                                arena.v_run(plan, c),
                                plan.bc_tiles,
                            ),
                            HmxOperands[GEN](
                                arena.p_run(plan, r), arena.v_run(plan, c), 0
                            ),
                        )
            else:
                hmx_attn_pv_block[emit_hmx=False](arena, plan, cur, 1 - cur)
            cur = 1 - cur

        for r in range(plan.br_tiles):
            for local in range(HMX_TILE):
                var l = arena.stats.load[1](
                    Coord(ATTN_STAT_L, r * HMX_TILE + local)
                )
                hmx_attn_set_diag(
                    arena,
                    r,
                    local,
                    (Scalar[DType.float32](1) / l).cast[HMX_DTYPE](),
                )
        hmx_attn_normalise[emit_hmx=False](arena, plan, cur, 1 - cur)
        cur = 1 - cur
        hmx_attn_store_o(arena, plan, O, q0, head_off, cur)
        q0 += br


def attn_run(
    br: Int, bc: Int, d: Int, seq: Int, reverse: Bool, mutation: Int
) raises -> AttnScore:
    """Builds a two-head tensor, runs head 1 through the chain, scores it.

    Head 1 rather than head 0 on purpose: `[seq, heads, D]` puts a head's `D`
    columns at a nonzero offset inside a strided row, and the packers address it
    through their own `col0`. Running head 0 would leave that offset at zero and
    the addressing untested.
    """
    comptime HEADS = 2
    var plan = hmx_attn_plan(br, bc, d, 1)
    var cols = HEADS * d
    var head_off = d

    var q_buf = alloc[Scalar[HMX_DTYPE]](seq * cols)
    var k_buf = alloc[Scalar[HMX_DTYPE]](seq * cols)
    var v_buf = alloc[Scalar[HMX_DTYPE]](seq * cols)
    var o_buf = alloc[Scalar[HMX_DTYPE]](seq * cols)
    var r_buf = alloc[Scalar[HMX_DTYPE]](seq * cols)
    var raw = alloc[UInt8](hmx_attn_arena_bytes(plan) + HMX_TILE_BYTES)

    var shape = row_major(Coord(seq, cols))
    var Q = HmxRows2D(ptr=q_buf.as_unsafe_any_origin(), layout=shape)
    var K = HmxRows2D(ptr=k_buf.as_unsafe_any_origin(), layout=shape)
    var V = HmxRows2D(ptr=v_buf.as_unsafe_any_origin(), layout=shape)
    var O = HmxRows2D(ptr=o_buf.as_unsafe_any_origin(), layout=shape)
    var R = HmxRows2D(ptr=r_buf.as_unsafe_any_origin(), layout=shape)

    attn_fill(Q, seq, cols, 5, 7, 8)
    attn_fill(K, seq, cols, 3, 11, 8)
    attn_fill(V, seq, cols, 2, 5, 4)
    for i in range(seq * cols):
        o_buf.store(i, SENTINEL)

    # The scale the kernel is handed, and the fp16 the requant record can hold.
    # The twin's reference uses the fp16 one so the comparison is about the
    # chain, not about a rounding the design already names as expected.
    var scale = Scalar[HMX_DTYPE](Float32(1.0) / sqrt(Float32(d)))
    attn_reference(Q, K, V, R, seq, d, head_off, scale.cast[DType.float32]()[0])

    var base = UnsafePointer[UInt8, MutUntrackedOrigin](
        unsafe_from_address=align_up(Int(raw), HMX_TILE_BYTES)
    )
    attn_flash_twin(
        plan,
        hmx_attn_arena_in[GEN](base, plan),
        Q,
        K,
        V,
        O,
        seq,
        head_off,
        scale,
        reverse,
        mutation,
    )

    var score = attn_compare(O, R, seq, d, head_off)

    q_buf.free()
    k_buf.free()
    v_buf.free()
    o_buf.free()
    r_buf.free()
    raw.free()
    return score


comptime ATTN_MIN_COSINE = 0.9995
"""Cosine floor for the twin's flash chain against its fp32 reference.

`S` and every key block's `O` partial round to fp16, so this is a tolerance and
not an equality — the design says so and says why (§5). The number is the twin's
alone: the DEVICE thresholds are set by the first slice-3 measurement on silicon
and frozen there, because a number written before that measurement either blesses
a defect or rejects a correct kernel."""

comptime ATTN_MAX_REL = 0.02
"""Row-relative error ceiling for the same comparison."""


def check_attn_flash(
    br: Int, bc: Int, d: Int, seq: Int, reverse: Bool
) raises -> Bool:
    """Asserts the composed flash chain reproduces dense attention."""
    var score = attn_run(br, bc, d, seq, reverse, 0)
    var order = "reverse" if reverse else "forward"
    if score.cosine < ATTN_MIN_COSINE or score.max_rel > ATTN_MAX_REL:
        print(
            "  FAIL attn flash br=",
            br,
            "bc=",
            bc,
            "d=",
            d,
            "seq=",
            seq,
            order,
            "- cos",
            score.cosine,
            "max_rel",
            score.max_rel,
        )
        return False
    print(
        "  ok   attn flash br=",
        br,
        "bc=",
        bc,
        "d=",
        d,
        "seq=",
        seq,
        order,
        "- cos",
        score.cosine,
        "max_rel",
        score.max_rel,
    )
    return True


def oracle_attn_mutation(
    br: Int, bc: Int, d: Int, seq: Int, mutation: Int, label: String
) raises -> Bool:
    """MUST fire: each named defect has to move the answer out of tolerance.

    Without this the tolerance above proves only that some arrangement of these
    functions produces plausible numbers. The rescale control is the load-bearing
    one — an online softmax that never corrects its running `O` still returns
    finite, well-scaled, entirely wrong values.
    """
    var score = attn_run(br, bc, d, seq, False, mutation)
    if score.cosine >= ATTN_MIN_COSINE and score.max_rel <= ATTN_MAX_REL:
        print(
            "  FAIL oracle attn",
            label,
            "did NOT fire - cos",
            score.cosine,
            "max_rel",
            score.max_rel,
        )
        return False
    print(
        "  ok   oracle attn",
        label,
        "fired - cos",
        score.cosine,
        "max_rel",
        score.max_rel,
    )
    return True


# ===--------------------------------------------------------------------------===
# Slice 4: the feeder/array split, driven as the device drives it.
#
# The device loop is `for step in range(sched.steps()): step(...); barrier()`.
# A barrier means every participant's work in a step happens before any
# participant's work in the next, and says NOTHING about the order within one —
# so a sequential sweep over thread indices is a legal interleaving, and running
# it in both directions brackets the two orders a real launch can produce.
# Array-first catches a feeder writing the slot the array read; array-last
# catches the reverse.
#
# What this cannot claim is the hardware: `emit_hmx=False` models the MAC. What
# it CAN claim, and what the schedule's new risk class needs, is that the
# double-buffer indices keep the array's operands away from the feeders' stores.
# `oracle_attn_reuse_hazard` is that control, and it is the one test here that a
# bit-identity check of a single step could never fail.
# ===--------------------------------------------------------------------------===


struct HostExp(AttnSoftmax):
    """The twin's exponential.

    `std.math.exp` where the device injects `_exp_hvx`, which is the substitution
    slice 3's twin already makes and the frozen device tolerance was measured
    across. The schedule itself is the SHIPPED function either way, which is the
    property this slice needed and slice 3 did not have.
    """

    @staticmethod
    def exp_nonpos[
        width: Int
    ](x: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
        return exp(x)


comptime ATTN_TEAM_W = 4
"""Vector width the twin's softmax reduces at. Divides every `bc` tested here."""


def attn_team_run(
    br: Int,
    bc: Int,
    d: Int,
    seq: Int,
    heads: Int,
    feeders: Int,
    array_first: Bool,
    reuse_hazard: Bool,
) raises -> AttnScore:
    """Runs EVERY head through the shipped step schedule and scores the worst.

    The worst head rather than the mean: a schedule that corrupts one head's
    ping-pong and leaves the rest exact is precisely the defect this exists to
    catch, and an average over 2-4 heads would dilute it below the tolerance.
    """
    var plan = hmx_attn_plan(br, bc, d, feeders)
    var sched = hmx_attn_sched(plan, seq, heads)
    var cols = heads * d

    var q_buf = alloc[Scalar[HMX_DTYPE]](seq * cols)
    var k_buf = alloc[Scalar[HMX_DTYPE]](seq * cols)
    var v_buf = alloc[Scalar[HMX_DTYPE]](seq * cols)
    var o_buf = alloc[Scalar[HMX_DTYPE]](seq * cols)
    var r_buf = alloc[Scalar[HMX_DTYPE]](seq * cols)
    var raw = alloc[UInt8](hmx_attn_arena_bytes(plan) + HMX_TILE_BYTES)

    var shape = row_major(Coord(seq, cols))
    var Q = HmxRows2D(ptr=q_buf.as_unsafe_any_origin(), layout=shape)
    var K = HmxRows2D(ptr=k_buf.as_unsafe_any_origin(), layout=shape)
    var V = HmxRows2D(ptr=v_buf.as_unsafe_any_origin(), layout=shape)
    var O = HmxRows2D(ptr=o_buf.as_unsafe_any_origin(), layout=shape)
    var R = HmxRows2D(ptr=r_buf.as_unsafe_any_origin(), layout=shape)

    # Per-head fills that are not translates of each other, so a head reading
    # another head's staged block is a wrong ANSWER rather than a coincidence.
    attn_fill(Q, seq, cols, 5, 7, 8)
    attn_fill(K, seq, cols, 3, 11, 8)
    attn_fill(V, seq, cols, 2, 5, 4)
    for i in range(seq * cols):
        o_buf.store(i, SENTINEL)

    var scale = Scalar[HMX_DTYPE](Float32(1.0) / sqrt(Float32(d)))
    for h in range(heads):
        attn_reference(
            Q, K, V, R, seq, d, h * d, scale.cast[DType.float32]()[0]
        )

    var base = UnsafePointer[UInt8, MutUntrackedOrigin](
        unsafe_from_address=align_up(Int(raw), HMX_TILE_BYTES)
    )
    var arena = hmx_attn_arena_in[GEN](base, plan)
    hmx_attn_program_records(arena, scale)
    hmx_attn_reset_diag(arena, plan)

    for step in range(sched.steps()):
        for slot in range(feeders + 1):
            var t = slot
            if array_first:
                t = feeders if slot == 0 else slot - 1
            hmx_attn_step[HostExp, ATTN_TEAM_W, emit_hmx=False](
                Q,
                K,
                V,
                O,
                arena,
                plan,
                sched,
                step,
                t,
                feeders,
                reuse_hazard,
            )

    var worst = AttnScore(cosine=1.0, max_rel=0.0)
    for h in range(heads):
        var score = attn_compare(O, R, seq, d, h * d)
        if score.cosine < worst.cosine:
            worst.cosine = score.cosine
        if score.max_rel > worst.max_rel:
            worst.max_rel = score.max_rel

    q_buf.free()
    k_buf.free()
    v_buf.free()
    o_buf.free()
    r_buf.free()
    raw.free()
    return worst


def check_attn_team(
    br: Int, bc: Int, d: Int, seq: Int, heads: Int, feeders: Int
) raises -> Bool:
    """Every head through the split schedule, in BOTH intra-step orders."""
    var okay = True
    for order in range(2):
        var array_first = order == 0
        var score = attn_team_run(
            br, bc, d, seq, heads, feeders, array_first, False
        )
        var label = "array-first" if array_first else "array-last"
        if score.cosine < ATTN_MIN_COSINE or score.max_rel > ATTN_MAX_REL:
            print(
                "  FAIL attn team br=",
                br,
                "bc=",
                bc,
                "d=",
                d,
                "seq=",
                seq,
                "heads=",
                heads,
                "feeders=",
                feeders,
                label,
                "- cos",
                score.cosine,
                "max_rel",
                score.max_rel,
            )
            okay = False
            continue
        print(
            "  ok   attn team br=",
            br,
            "bc=",
            bc,
            "d=",
            d,
            "seq=",
            seq,
            "heads=",
            heads,
            "feeders=",
            feeders,
            label,
            "- worst head cos",
            score.cosine,
            "max_rel",
            score.max_rel,
        )
    return okay


def attn_team_o_bits[
    epi_xpose: Bool
](
    br: Int, bc: Int, d: Int, seq: Int, heads: Int, feeders: Int
) raises -> List[UInt16]:
    """One full schedule run's O buffer, as raw fp16 bits.

    The harness is `attn_team_run`'s (same deterministic fills, same step
    loop), minus the reference scoring: the epilogue-transpose claim is
    BIT-identity against the shipped path, so the comparison wants bits,
    not a tolerance.
    """
    var plan = hmx_attn_plan(br, bc, d, feeders)
    var sched = hmx_attn_sched(plan, seq, heads)
    var cols = heads * d

    var q_buf = alloc[Scalar[HMX_DTYPE]](seq * cols)
    var k_buf = alloc[Scalar[HMX_DTYPE]](seq * cols)
    var v_buf = alloc[Scalar[HMX_DTYPE]](seq * cols)
    var o_buf = alloc[Scalar[HMX_DTYPE]](seq * cols)
    var raw = alloc[UInt8](hmx_attn_arena_bytes(plan) + HMX_TILE_BYTES)

    var shape = row_major(Coord(seq, cols))
    var Q = HmxRows2D(ptr=q_buf.as_unsafe_any_origin(), layout=shape)
    var K = HmxRows2D(ptr=k_buf.as_unsafe_any_origin(), layout=shape)
    var V = HmxRows2D(ptr=v_buf.as_unsafe_any_origin(), layout=shape)
    var O = HmxRows2D(ptr=o_buf.as_unsafe_any_origin(), layout=shape)

    attn_fill(Q, seq, cols, 5, 7, 8)
    attn_fill(K, seq, cols, 3, 11, 8)
    attn_fill(V, seq, cols, 2, 5, 4)
    for i in range(seq * cols):
        o_buf.store(i, SENTINEL)

    var scale = Scalar[HMX_DTYPE](Float32(1.0) / sqrt(Float32(d)))
    var base = UnsafePointer[UInt8, MutUntrackedOrigin](
        unsafe_from_address=align_up(Int(raw), HMX_TILE_BYTES)
    )
    var arena = hmx_attn_arena_in[GEN](base, plan)
    hmx_attn_program_records(arena, scale)
    hmx_attn_reset_diag(arena, plan)

    for step in range(sched.steps()):
        for slot in range(feeders + 1):
            hmx_attn_step[
                HostExp, ATTN_TEAM_W, emit_hmx=False, epi_xpose=epi_xpose
            ](Q, K, V, O, arena, plan, sched, step, slot, feeders, False)

    var bits = List[UInt16]()
    bits.reserve(seq * cols)
    var o16 = o_buf.bitcast[UInt16]()
    for i in range(seq * cols):
        bits.append(o16.load(i))

    q_buf.free()
    k_buf.free()
    v_buf.free()
    o_buf.free()
    raw.free()
    return bits^


def check_attn_epi_xpose_bit_identity(
    br: Int, bc: Int, d: Int, seq: Int, heads: Int, feeders: Int
) raises -> Bool:
    """The epilogue transpose returns the SHIPPED path's bits, exactly.

    `_exp_hvx`/`exp` carry no width branch and the batched `l` update is the
    impl's own two ops lane-wise, so lane `w` of the tile epilogue must
    compute bit-for-bit what row `w`'s width-1 chain computed. Any mismatch
    is a real defect (a reordered reduction, a width-dependent exp, a stat
    row misaddressed) — there is no tolerance to hide in.
    """
    var base_bits = attn_team_o_bits[False](br, bc, d, seq, heads, feeders)
    var xpose_bits = attn_team_o_bits[True](br, bc, d, seq, heads, feeders)
    var bad = 0
    for i in range(len(base_bits)):
        if base_bits[i] != xpose_bits[i]:
            bad += 1
    if bad != 0:
        print(
            "  FAIL attn epi-xpose bit identity br=", br, "bc=", bc, "d=", d,
            "seq=", seq, "heads=", heads, "feeders=", feeders,
            "- mismatched fp16 words:", bad, "of", len(base_bits),
        )
        return False
    print(
        "  ok   attn epi-xpose bit identity br=", br, "bc=", bc, "d=", d,
        "seq=", seq, "heads=", heads, "feeders=", feeders,
        "-", len(base_bits), "fp16 words equal",
    )
    return True


def oracle_attn_reuse_hazard(
    br: Int, bc: Int, d: Int, seq: Int, heads: Int, feeders: Int
) raises -> Bool:
    """MUST fire: staging the next key block into the half the array is reading.

    THE new risk class of this slice. The schedule's whole claim is that a
    feeder's stores and the array's reads never name the same crouton in the same
    step, and that claim rests on one `& 1`. Off by one, the array multiplies key
    block `j + 1`'s croutons into block `j`'s scores — which is still finite,
    still well-scaled, and wrong. Nothing that checks a single step against a
    reference can see it, because every step in isolation is correct.
    """
    var score = attn_team_run(br, bc, d, seq, heads, feeders, False, True)
    if score.cosine >= ATTN_MIN_COSINE and score.max_rel <= ATTN_MAX_REL:
        print(
            "  FAIL oracle attn reuse-hazard did NOT fire - cos",
            score.cosine,
            "max_rel",
            score.max_rel,
        )
        return False
    print(
        "  ok   oracle attn reuse-hazard fired - cos",
        score.cosine,
        "max_rel",
        score.max_rel,
    )
    return True


comptime ATTN_MASK_NEG = Scalar[DType.float32](-1.0e9)
"""The host's masked-out fill.

`npu_qwen3_encoder_dispatch.cpp`'s `buildBias` writes a large negative rather
than `-inf`, and the reason matters to the arm under test: `-inf` reaches the
exponential as `exp(-inf - finite)`, which is a clean zero, but it also reaches
the running max, and a row whose max is `-inf` divides by a zero denominator.
A large negative underflows to the same zero probability and keeps every
intermediate finite."""


def attn_mask_fill(mask: AttnStats[GEN], seq: Int, valid: Int, live: Bool):
    """The encoder's causal + padding bias, or the zero bias that must be inert.

    `keep = k <= q and k < valid` is `buildBias`'s predicate. Every query row
    keeps at least key 0, so no row is fully masked and the online recurrence's
    running max is always set from a visible score — the precondition a padding
    mask can violate and the one this fill deliberately satisfies.
    """
    for q in range(seq):
        for k in range(seq):
            var keep = k <= q and k < valid
            var b = Scalar[DType.float32](0)
            if live and not keep:
                b = ATTN_MASK_NEG
            mask.store[1](Coord(q, k), SIMD[DType.float32, 1](b))


def attn_fill_roped(
    t: HmxRows2D, rows: Int, cols: Int, a: Int, b: Int, div: Int, dc: Int
):
    """A fill shaped like a normed-then-rotated attention row.

    `attn_fill`'s coprime-periodic pattern is the right fixture for a LAYOUT
    bug and the wrong one for a RANGE bug, and order 134 is why that distinction
    now has a test. Scaling that pattern makes the scores large AND widely
    separated, so the softmax degenerates into a hard argmax that both arms pick
    correctly -- the array arm scores `max_rel` EXACTLY 0.0 at x16 and x64,
    which reads as "better" and is really "the question stopped being asked".

    What an RMSNorm'd, RoPE'd row looks like instead is every element sharing a
    large common magnitude with a small structured variation on top. Every score
    in a row then carries the same big term, and the part that decides the
    softmax is the small one. `dc` is that common magnitude and the pattern is
    the variation, so the ratio the arm has to resolve is the production one.
    """
    for r in range(rows):
        for c in range(cols):
            t.store[1](
                Coord(r, c),
                Scalar[HMX_DTYPE](
                    Float32(dc)
                    + Float32((a * r + b * c) % 13 - 6) / Float32(div)
                ),
            )


def attn_reference_masked(
    Q: HmxRows2D,
    K: HmxRows2D,
    V: HmxRows2D,
    want: HmxRows2D,
    mask: AttnStats[GEN],
    seq: Int,
    d: Int,
    head_off: Int,
    scale: Float32,
):
    """`attn_reference` with `+ bias[q, k]` before the max.

    Kept as its own textbook three-line form over the ORIGINAL operands rather
    than as a parameter on the dense reference, for the reason the dense one
    states: an oracle earns its place by sharing no blocking, no online
    recurrence and no crouton with the thing it judges, and threading a flag
    through it would couple the two answers at the one point the masked claim
    rests on.

    The add sits after the scale and before the running max, which is where
    `_mha_flash_head_masked` puts it and therefore what the array arm has to
    reproduce.
    """
    for m in range(seq):
        var top = ATTN_NEG
        var scores = List[Scalar[DType.float32]]()
        for n in range(seq):
            var acc = Scalar[DType.float32](0)
            for k in range(d):
                acc += (
                    Q.load[1](Coord(m, head_off + k)).cast[DType.float32]()
                    * K.load[1](Coord(n, head_off + k)).cast[DType.float32]()
                )
            var s = (
                acc * Scalar[DType.float32](scale)
                + mask.load[1](Coord(m, n))[0]
            )
            scores.append(s)
            if s > top:
                top = s
        var total = Scalar[DType.float32](0)
        for n in range(seq):
            scores[n] = exp(scores[n] - top)
            total += scores[n]
        for c in range(d):
            var acc = Scalar[DType.float32](0)
            for n in range(seq):
                acc += (
                    scores[n]
                    * V.load[1](Coord(n, head_off + c)).cast[DType.float32]()
                )
            want.store[1](
                Coord(m, head_off + c),
                (acc / total).cast[HMX_DTYPE](),
            )


def attn_team_run_masked[
    masked: Bool, center: Bool = True
](
    br: Int,
    bc: Int,
    d: Int,
    seq: Int,
    heads: Int,
    feeders: Int,
    valid: Int,
    live: Bool,
    array_first: Bool,
    O: HmxRows2D,
    gain: Int = 1,
    roped: Bool = False,
) raises -> AttnScore:
    """Every head through the shipped step schedule, scored against the masked
    reference.

    `masked` picks the ARM and `live` picks the BIAS, and they are independent
    on purpose — the two off-diagonal combinations are the controls:

    - `masked=False, live=True` runs the dense arm against a masked reference.
      It MUST fail, which is what makes a passing masked run evidence that the
      mask was applied rather than that the mask did not matter.
    - `masked=True, live=False` runs the masked arm on a zero bias. It must
      reproduce the dense arm exactly, which is the inertness claim.

    Worst head rather than mean, for `attn_team_run`'s reason and order 120's:
    an envelope averaged over heads dilutes a schedule that corrupts one of
    them below the tolerance.
    """
    var plan = hmx_attn_plan(br, bc, d, feeders)
    var sched = hmx_attn_sched(plan, seq, heads)
    var cols = heads * d

    var q_buf = alloc[Scalar[HMX_DTYPE]](seq * cols)
    var k_buf = alloc[Scalar[HMX_DTYPE]](seq * cols)
    var v_buf = alloc[Scalar[HMX_DTYPE]](seq * cols)
    var r_buf = alloc[Scalar[HMX_DTYPE]](seq * cols)
    var m_buf = alloc[Scalar[DType.float32]](seq * seq)
    var raw = alloc[UInt8](hmx_attn_arena_bytes(plan) + HMX_TILE_BYTES)

    var shape = row_major(Coord(seq, cols))
    var Q = HmxRows2D(ptr=q_buf.as_unsafe_any_origin(), layout=shape)
    var K = HmxRows2D(ptr=k_buf.as_unsafe_any_origin(), layout=shape)
    var V = HmxRows2D(ptr=v_buf.as_unsafe_any_origin(), layout=shape)
    var R = HmxRows2D(ptr=r_buf.as_unsafe_any_origin(), layout=shape)
    var mask = AttnStats[GEN](ptr=m_buf, layout=row_major(Coord(seq, seq)))

    # The dose rides Q and K only. `V` stays put because the failure being
    # reproduced is monotone in `|scale * QK^T|`, and scaling `V` would move the
    # reference's magnitude with it and hide exactly that.
    if roped:
        attn_fill_roped(Q, seq, cols, 5, 7, 8, gain)
        attn_fill_roped(K, seq, cols, 3, 11, 8, gain)
    else:
        attn_fill(Q, seq, cols, 5, 7, 8, gain)
        attn_fill(K, seq, cols, 3, 11, 8, gain)
    attn_fill(V, seq, cols, 2, 5, 4)
    attn_mask_fill(mask, seq, valid, live)
    for r in range(seq):
        for c in range(cols):
            O.store[1](Coord(r, c), SENTINEL)

    var scale = Scalar[HMX_DTYPE](Float32(1.0) / sqrt(Float32(d)))
    for h in range(heads):
        attn_reference_masked(
            Q, K, V, R, mask, seq, d, h * d, scale.cast[DType.float32]()[0]
        )

    var base = UnsafePointer[UInt8, MutUntrackedOrigin](
        unsafe_from_address=align_up(Int(raw), HMX_TILE_BYTES)
    )
    var arena = hmx_attn_arena_in[GEN](base, plan)
    hmx_attn_program_records(arena, scale)
    hmx_attn_reset_diag(arena, plan)

    for step in range(sched.steps()):
        for slot in range(feeders + 1):
            var t = slot
            if array_first:
                t = feeders if slot == 0 else slot - 1
            comptime if masked:
                hmx_attn_step_masked[
                    HostExp, ATTN_TEAM_W, emit_hmx=False, center=center
                ](
                    Q,
                    K,
                    V,
                    O,
                    mask,
                    arena,
                    plan,
                    sched,
                    step,
                    t,
                    feeders,
                    False,
                )
            else:
                hmx_attn_step[HostExp, ATTN_TEAM_W, emit_hmx=False](
                    Q, K, V, O, arena, plan, sched, step, t, feeders, False
                )

    var worst = AttnScore(cosine=1.0, max_rel=0.0)
    for h in range(heads):
        var score = attn_compare(O, R, seq, d, h * d)
        if score.cosine < worst.cosine:
            worst.cosine = score.cosine
        if score.max_rel > worst.max_rel:
            worst.max_rel = score.max_rel

    q_buf.free()
    k_buf.free()
    v_buf.free()
    r_buf.free()
    m_buf.free()
    raw.free()
    return worst


def check_attn_team_masked(
    br: Int, bc: Int, d: Int, seq: Int, heads: Int, feeders: Int, valid: Int
) raises -> Bool:
    """The masked arm against the masked reference, in BOTH intra-step orders.
    """
    var okay = True
    for order in range(2):
        var array_first = order == 0
        var o_buf = alloc[Scalar[HMX_DTYPE]](seq * heads * d)
        var O = HmxRows2D(
            ptr=o_buf.as_unsafe_any_origin(),
            layout=row_major(Coord(seq, heads * d)),
        )
        var score = attn_team_run_masked[True](
            br, bc, d, seq, heads, feeders, valid, True, array_first, O
        )
        o_buf.free()
        var label = "array-first" if array_first else "array-last"
        if score.cosine < ATTN_MIN_COSINE or score.max_rel > ATTN_MAX_REL:
            print(
                "  FAIL attn masked br=",
                br,
                "bc=",
                bc,
                "d=",
                d,
                "seq=",
                seq,
                "heads=",
                heads,
                "feeders=",
                feeders,
                "valid=",
                valid,
                label,
                "- cos",
                score.cosine,
                "max_rel",
                score.max_rel,
            )
            okay = False
            continue
        print(
            "  ok   attn masked br=",
            br,
            "bc=",
            bc,
            "d=",
            d,
            "seq=",
            seq,
            "heads=",
            heads,
            "feeders=",
            feeders,
            "valid=",
            valid,
            label,
            "- worst head cos",
            score.cosine,
            "max_rel",
            score.max_rel,
        )
    return okay


def oracle_attn_mask_off(
    br: Int, bc: Int, d: Int, seq: Int, heads: Int, feeders: Int, valid: Int
) raises -> Bool:
    """MUST fire: the DENSE arm judged against the masked reference.

    Without this a passing masked run is ambiguous between "the mask was
    applied correctly" and "this shape's mask barely changes the answer". The
    causal triangle removes about half of every row's keys, so a dense arm
    cannot survive the masked envelope — and if it ever does, the fill has
    stopped exercising the mask rather than the mask having become free.
    """
    var o_buf = alloc[Scalar[HMX_DTYPE]](seq * heads * d)
    var O = HmxRows2D(
        ptr=o_buf.as_unsafe_any_origin(),
        layout=row_major(Coord(seq, heads * d)),
    )
    var score = attn_team_run_masked[False](
        br, bc, d, seq, heads, feeders, valid, True, False, O
    )
    o_buf.free()
    if score.cosine >= ATTN_MIN_COSINE and score.max_rel <= ATTN_MAX_REL:
        print(
            "  FAIL oracle attn mask-off did NOT fire - cos",
            score.cosine,
            "max_rel",
            score.max_rel,
        )
        return False
    print(
        "  ok   oracle attn mask-off fired - cos",
        score.cosine,
        "max_rel",
        score.max_rel,
    )
    return True


def check_attn_mask_inert(
    br: Int, bc: Int, d: Int, seq: Int, heads: Int, feeders: Int
) raises -> Bool:
    """A zero bias must leave the masked arm EQUAL to the dense arm, element for
    element.

    Not "within tolerance": the masked arm is the dense arm plus one fp32 add of
    zero, and adding zero to a finite score is exact, so anything short of
    equality means the seam moved something else — the deal-in, the row-buffer
    phase, or the point in the recurrence the scores are read at. This is the
    control that keeps the render's dense arm safe from a change made for the
    encoder.
    """
    var cols = heads * d
    var dense_buf = alloc[Scalar[HMX_DTYPE]](seq * cols)
    var masked_buf = alloc[Scalar[HMX_DTYPE]](seq * cols)
    var shape = row_major(Coord(seq, cols))
    var dense = HmxRows2D(ptr=dense_buf.as_unsafe_any_origin(), layout=shape)
    var got = HmxRows2D(ptr=masked_buf.as_unsafe_any_origin(), layout=shape)

    _ = attn_team_run_masked[False](
        br, bc, d, seq, heads, feeders, seq, False, False, dense
    )
    _ = attn_team_run_masked[True, center=False](
        br, bc, d, seq, heads, feeders, seq, False, False, got
    )

    var differ = 0
    for r in range(seq):
        for c in range(cols):
            if got.load[1](Coord(r, c)) != dense.load[1](Coord(r, c)):
                differ += 1
    dense_buf.free()
    masked_buf.free()

    if differ != 0:
        print(
            "  FAIL attn zero-mask inertness -",
            differ,
            "of",
            seq * cols,
            "elements differ from the dense arm",
        )
        return False
    print(
        "  ok   attn zero-mask inertness - all",
        seq * cols,
        "elements equal the dense arm",
    )
    return True


def check_attn_mask_dose(
    br: Int,
    bc: Int,
    d: Int,
    seq: Int,
    heads: Int,
    feeders: Int,
    valid: Int,
    roped: Bool = False,
) raises -> Bool:
    """Operand magnitude must not move the answer. Order 134's STOP, on the host.

    The device leg measured the array arm right at probe magnitudes and WRONG
    at the encoder's: cosine 0.0108 against the vector arm, 5.7M of 7.9M
    elements different, and the same RMS -- a wrong answer of the right size.
    Its dose-response is what turned that into a mechanism: scaling Q and K by
    1, 4, 16 and 64 walked the error 0.001522 -> 0.004190 -> 0.056 -> 0.650
    while the vector arm stayed flat, and the production mask passed at x1, so
    the mask's shape and sentinel were excluded.

    A rope'd encoder Q/K row is in the hundreds, so production sits at the far
    end of that curve. These four doses are permanent cases: the arm may not be
    correct only where the fill is small.

    MECHANISM, and it is in the model rather than in this file:
    `hmx_crouton.mojo:1308` reads

        (acc.cast[HMX_DTYPE]() * scale + bias).cast[dtype]()

    so the fp32 accumulator is narrowed to fp16 BEFORE the requant multiply and
    add. Three consequences, and every one of the fingerprint's parts follows
    from them:

    - The value that must survive fp16 is the RAW dot over `D`, not the score:
      `1/sqrt(D)` is applied after the narrowing.
    - fp16 is a FLOATING point, so its absolute error is proportional to
      magnitude. The softmax exponentiates a DIFFERENCE, so what it needs is
      absolute precision. Error in the exponent therefore grows linearly with
      `|acc|` -- the dose-response, and why the vector arm (fp32 scores end to
      end, `_mha_flash_head_masked`) is flat.
    - Neither the record's `scale` nor its `bias` can defend against it, because
      both are applied AFTER the narrowing, and `bias` is indexed per output
      COLUMN anyway. The only defence is to make `acc` itself small.

    The `roped` fixture is the one that reproduces it and the `scaled` one is
    not, which is the trap worth keeping: scaling a coprime-periodic pattern
    grows the scores AND their separation together, so the softmax turns into a
    hard argmax both arms agree on and the case reports EXACTLY 0.0 error while
    the arm is at its most broken.
    """
    var okay = True
    print(
        "  dose-response (",
        "roped" if roped else "scaled",
        "), worst head, masked arm vs masked reference:",
    )
    for i in range(4):
        var gain = 1 << (2 * i)
        var o_buf = alloc[Scalar[HMX_DTYPE]](seq * heads * d)
        var O = HmxRows2D(
            ptr=o_buf.as_unsafe_any_origin(),
            layout=row_major(Coord(seq, heads * d)),
        )
        var score = attn_team_run_masked[True](
            br, bc, d, seq, heads, feeders, valid, True, False, O, gain, roped
        )
        o_buf.free()
        # Cosine FIRST and always: a degenerate output scores `max_rel` 0.0,
        # so a verdict on the error alone calls the worst case the best one.
        var bad = score.cosine < ATTN_MIN_COSINE or score.max_rel > ATTN_MAX_REL
        print(
            "    x",
            gain,
            "- cos",
            score.cosine,
            "max_rel",
            score.max_rel,
            "FAIL" if bad else "ok",
        )
        if bad:
            okay = False
    return okay


def check_attn_dense_envelope(
    br: Int, bc: Int, d: Int, seq: Int, heads: Int, feeders: Int
) raises -> Bool:
    """Where the DENSE arm's own curve starts, at the render's geometry.

    The narrowing at `hmx_crouton.mojo:1308` is under the dense arm too -- the
    one orders 118-123 put in the render, uncentred to this day. Tier B at 53.4
    dB says it is empirically fine at the model and inputs the render actually
    runs, and that is a statement about a working point, not about the kernel.
    So this measures the working point's distance from the cliff instead of
    asserting there is no cliff.

    REPORTS, DOES NOT GATE, on purpose. The dense arm is unchanged by this
    branch, and turning its bytes over means a MEF re-anchor and a Tier B re-run
    (the 123 discipline). A failing case here would block the encoder's fix on
    the render's rollout, which is the coupling the house pattern exists to
    avoid: document the envelope, gate the arm that changed.
    """
    print("  dense-arm envelope (roped), worst head vs dense reference:")
    for i in range(4):
        var gain = 1 << (2 * i)
        var o_buf = alloc[Scalar[HMX_DTYPE]](seq * heads * d)
        var O = HmxRows2D(
            ptr=o_buf.as_unsafe_any_origin(),
            layout=row_major(Coord(seq, heads * d)),
        )
        var score = attn_team_run_masked[False](
            br, bc, d, seq, heads, feeders, seq, False, False, O, gain, True
        )
        o_buf.free()
        print(
            "    x",
            gain,
            "- cos",
            score.cosine,
            "max_rel",
            score.max_rel,
            "(inside envelope)" if (
                score.max_rel <= ATTN_MAX_REL
                and score.cosine >= ATTN_MIN_COSINE
            ) else "(PAST the envelope)",
        )
    return True


def attn_arrivals_by_thread(
    participants: Int, feeders: Int, accepts: Bool, granted: Bool, steps: Int
) -> Int:
    """`hmx_attn_arrivals` has no thread argument, which is the property."""
    return hmx_attn_arrivals(participants, feeders, accepts, granted, steps)


def check_attn_team_arrivals() raises -> Bool:
    """Arrival counts must be equal across every thread index, every exit.

    Order 084's wedge was a refusal some threads took and others did not, and a
    bit-identity test cannot see that class at all — every thread that RETURNED
    wrote correct memory. The function under test takes no thread index, so the
    check is that its inputs are all launch-uniform and that each exit yields one
    number: 0 when the roles or the shape refuse, 1 when the grant did, and
    `1 + steps` when it ran.
    """
    var okay = True
    for participants in range(0, 9):
        var feeders = participants - 1
        for steps in range(0, 4):
            var want_refused = attn_arrivals_by_thread(
                participants, feeders, False, True, steps
            )
            if want_refused != 0:
                print(
                    "  FAIL attn arrivals: refusal at participants=",
                    participants,
                    "gave",
                    want_refused,
                )
                okay = False
            var want_ungranted = attn_arrivals_by_thread(
                participants, feeders, True, False, steps
            )
            var expect_ungranted = 1 if participants >= 2 else 0
            if want_ungranted != expect_ungranted:
                print(
                    "  FAIL attn arrivals: ungranted at participants=",
                    participants,
                    "gave",
                    want_ungranted,
                )
                okay = False
            var want_ran = attn_arrivals_by_thread(
                participants, feeders, True, True, steps
            )
            var expect_ran = (1 + steps) if participants >= 2 else 0
            if want_ran != expect_ran:
                print(
                    "  FAIL attn arrivals: granted at participants=",
                    participants,
                    "gave",
                    want_ran,
                )
                okay = False
    if okay:
        print("  ok   attn arrivals equal on every participant, counts 0..8")
    return okay


def check_attn_sched_shape() raises -> Bool:
    """The step decode must cover every step exactly once per query block.

    A schedule whose phases overlap or leave a gap is the defect the numeric
    comparison would report as a tolerance failure with no idea where — so the
    arithmetic is asserted directly: `steps_per_q` is the prologue, the `2(n+1)`
    pipelined pair-steps and the three that drain them, and nothing else.
    """
    var okay = True
    for n in range(1, 7):
        var sched = HmxAttnSched(2, 3, n)
        if sched.steps_per_q() != 2 * n + 6:
            print("  FAIL attn sched steps_per_q at n=", n)
            okay = False
        if sched.steps() != 2 * 3 * (2 * n + 6):
            print("  FAIL attn sched steps at n=", n)
            okay = False
    var refused = hmx_attn_sched(hmx_attn_plan(64, 64, 64, 4), 96, 2)
    if refused.usable() or refused.steps() != 0:
        print("  FAIL attn sched accepted a seq the blocks do not divide")
        okay = False
    if okay:
        print("  ok   attn sched step decode covers 2n+6 per query block")
    return okay


def check_attn_arena(br: Int, bc: Int, d: Int, threads: Int) -> Bool:
    """Asserts the arena's size is the design's table, buffer by buffer.

    Recomputed here from the table's own expressions rather than from the
    function under test, so a carve that grew a region silently fails here
    instead of only showing up as a VTCM refusal on the part.
    """
    var plan = hmx_attn_plan(br, bc, d, threads)
    if not plan.usable():
        print("  FAIL attn arena plan br=", br, "bc=", bc, "d=", d)
        return False
    var croutons = (
        br * d  # Q
        + 2 * br * d  # O ping-pong
        + 2 * bc * d  # K ping-pong
        + 2 * bc * d  # V ping-pong
        + 2 * br * bc  # S ping-pong
        + 2 * br * bc  # P ping-pong
        + br * HMX_TILE  # diag(corr), reused for diag(1/l)
    ) // HMX_TILE_ELEMS
    var want = (
        croutons * HMX_TILE_BYTES + 2 * 256 + 4 * br * 4 + 2 * threads * bc * 4
    )
    var got = hmx_attn_arena_bytes(plan)
    if got != want:
        print("  FAIL attn arena bytes", got, "want", want)
        return False
    print(
        "  ok   attn arena br=",
        br,
        "bc=",
        bc,
        "d=",
        d,
        "-",
        got,
        "bytes (",
        got * 100 // (8 * 1024 * 1024),
        "% of an 8 MiB grant )",
    )
    return True


def check_attn_acc_probe(variant: Int) -> Bool:
    """Asserts the accumulator probe can tell the three widths apart.

    Run against the MODEL, which accumulates in fp32, so the readout must equal
    the wide prediction. What is being checked here is not the array — it is that
    the staged panel SEPARATES the predictions at all: a probe whose three
    answers coincide would report agreement on device no matter what the
    accumulator does, which is the failure mode the design's open question 1
    cannot afford.
    """
    var plan = hmx_attn_plan(4 * HMX_TILE, 2 * HMX_TILE, 4 * HMX_TILE, 1)
    var raw = alloc[UInt8](hmx_attn_arena_bytes(plan) + HMX_TILE_BYTES)
    var base = UnsafePointer[UInt8, MutUntrackedOrigin](
        unsafe_from_address=align_up(Int(raw), HMX_TILE_BYTES)
    )
    var arena = hmx_attn_arena_in[GEN](base, plan)
    hmx_attn_program_records(arena, Scalar[HMX_DTYPE](1))
    var probe = hmx_attn_acc_probe[emit_hmx=False](arena, plan, variant)
    raw.free()

    var ok = True
    if probe.readout != probe.wide:
        print(
            "  FAIL attn acc probe",
            variant,
            "- modelled readout",
            probe.readout,
            "!= wide",
            probe.wide,
        )
        ok = False
    if probe.wide == probe.per_product:
        print(
            "  FAIL attn acc probe",
            variant,
            "- wide and per-product agree at",
            probe.wide,
            "; the probe cannot discriminate",
        )
        ok = False
    # Variant 0 is chosen so a per-PASS accumulator tracks the wide one, and
    # variant 1 so it tracks the narrow one. Together they separate all three;
    # separately neither does, which is why both must hold.
    var pass_is_wide = probe.per_pass == probe.wide
    if variant == 0 and not pass_is_wide:
        print("  FAIL attn acc probe 0 - per-pass should track wide")
        ok = False
    if variant == 1 and pass_is_wide:
        print("  FAIL attn acc probe 1 - per-pass should track per-product")
        ok = False
    if ok:
        print(
            "  ok   attn acc probe",
            variant,
            "- wide",
            probe.wide,
            "per-product",
            probe.per_product,
            "per-pass",
            probe.per_pass,
        )
    return ok


def check_vdeal_inverse() -> Bool:
    """Asserts `hvx_vdeal` undoes `hvx_vshuff` at every `Rt` the packers use.

    The deal's model is the shuffle's loop run backwards, so the ONE thing that
    can be wrong about the transcription is the direction — and the inverse
    property is exactly what a reversed direction breaks. Checked at the full
    mask and at a single stride bit, because a network that is self-inverse at
    `-2` by accident need not be at one stage.
    """
    comptime LANES = hvx_lanes[HMX_DTYPE]()
    var vv = SIMD[HMX_DTYPE, LANES]()
    var vu = SIMD[HMX_DTYPE, LANES]()
    for i in range(LANES):
        vv[i] = Scalar[HMX_DTYPE](i)
        vu[i] = Scalar[HMX_DTYPE](LANES + i)

    var bad = 0

    var shuffled = hvx_vshuff[HVX_SHUFF_INTERLEAVE](vu, vv)
    var back = hvx_vdeal[HVX_DEAL_SEPARATE](
        shuffled.slice[LANES, offset=LANES](), shuffled.slice[LANES, offset=0]()
    )
    for i in range(LANES):
        if back[i] != vv[i]:
            bad += 1
        if back[LANES + i] != vu[i]:
            bad += 1

    comptime ONE_STAGE = 4 * 2
    var staged = hvx_vshuff[ONE_STAGE](vu, vv)
    var undone = hvx_vdeal[ONE_STAGE](
        staged.slice[LANES, offset=LANES](), staged.slice[LANES, offset=0]()
    )
    for i in range(LANES):
        if undone[i] != vv[i]:
            bad += 1
        if undone[LANES + i] != vu[i]:
            bad += 1

    if bad != 0:
        print("  FAIL vdeal inverse -", bad, "of", 4 * LANES, "positions")
        return False
    print("  ok   vdeal inverts vshuff (full mask and one stage)")
    return True


def check_vdeal_separates() -> Bool:
    """Pins `hvx_vdeal`'s lane rule at `HVX_DEAL_SEPARATE`, on its own.

    The seam reads a crouton vector pair as two whole query rows, and that
    reading is only true if the deal collects the pair's EVEN halfwords into its
    low vector and the odd ones into its high vector, in order. Asserted directly
    rather than inferred from the inverse property, in the spirit of
    `check_byte_order`.
    """
    comptime LANES = hvx_lanes[HMX_DTYPE]()
    comptime HALF = LANES // 2
    var vv = SIMD[HMX_DTYPE, LANES]()
    var vu = SIMD[HMX_DTYPE, LANES]()
    for i in range(LANES):
        vv[i] = Scalar[HMX_DTYPE](i)
        vu[i] = Scalar[HMX_DTYPE](LANES + i)

    var pair = hvx_vdeal[HVX_DEAL_SEPARATE](vu, vv)

    var bad = 0
    for i in range(HALF):
        if pair[i] != vv[2 * i]:
            bad += 1
        if pair[HALF + i] != vu[2 * i]:
            bad += 1
        if pair[LANES + i] != vv[2 * i + 1]:
            bad += 1
        if pair[LANES + HALF + i] != vu[2 * i + 1]:
            bad += 1

    if bad != 0:
        print("  FAIL vdeal separate -", bad, "of", 2 * LANES, "positions")
        return False
    print("  ok   vdeal separate lane rule")
    return True


def main() raises:
    print("==== HMX crouton packers (tile", HMX_TILE, "fp16) ====")
    var failures = 0

    if not check_depth_range():
        failures += 1

    # Byte order, both axis roles. The output's role is the activation's over
    # C[M, N], so it is the same assertion and not a third case.
    if not check_byte_order[True]("weight  (slab=k, lane=n)"):
        failures += 1
    if not check_byte_order[False]("act/out (slab=m, lane=k|n)"):
        failures += 1

    for tiles in [1, 2, 3, 32]:
        if not check_surjective[True](tiles, "weight"):
            failures += 1
        if not check_surjective[False](tiles, "act/out"):
            failures += 1

    # Multi-tile exercises the 1024-element tile stride; the non-zero origins
    # exercise the caller's offsets, which is where an HMX nest indexes an
    # output tile out of the middle of a panel.
    var ts = [1, 2, 3, 4, 32]
    var r0 = [0, 0, 32, 96, 0]
    var c0 = [0, 32, 0, 64, 0]
    for i in range(len(ts)):
        if not check_roundtrip[True](ts[i], r0[i], c0[i], "weight"):
            failures += 1
        if not check_roundtrip[False](ts[i], r0[i], c0[i], "act/out"):
            failures += 1

    for tiles in [1, 3]:
        if not oracle_wrong_axis(tiles):
            failures += 1

    # The vector packer (P5 slice 1). The permute's own lane rule first, then
    # byte identity against the scalar packer at every shape class the two arms
    # can differ over: `tiles = 1` is the pure scalar tail, `2` and `4` are whole
    # vector steps, `3` and `5` are a step plus that tail, and the non-zero
    # origins are what a stripe's subpanel share looks like.
    if not check_vshuff_interleave():
        failures += 1
    var vts = [1, 2, 3, 4, 5, 8]
    var vr0 = [0, 0, 32, 96, 0, 0]
    var vc0 = [0, 32, 0, 64, 32, 0]
    for i in range(len(vts)):
        for ids in [False, True]:
            if not check_pack_vec[True](vts[i], vr0[i], vc0[i], ids, "weight "):
                failures += 1
            if not check_pack_vec[False](
                vts[i], vr0[i], vc0[i], ids, "act/out"
            ):
                failures += 1
    for tiles in [2, 4]:
        if not check_pack_vec_perm(tiles):
            failures += 1
        if not oracle_pack_vec_mutant(tiles):
            failures += 1
    if not check_pack_vec_routing():
        failures += 1

    # The attention split-pair pack (attention slice 1). The pair packer's byte
    # identity at the origins a keyed share produces, the routing that keeps it
    # out of single-flag objects, then the whole-panel share identity at the
    # render plan (whole pairs) and at an odd DV count (the scalar tail), each
    # with a falsifier that must fire.
    if not check_pack_vec_mha_routing():
        failures += 1
    var tr0 = [0, 32, 0]
    var tc0 = [0, 0, 64]
    for i in range(len(tr0)):
        if not check_pack_tile_pair_vec(tr0[i], tc0[i]):
            failures += 1
    if not oracle_pack_tile_pair_swapped():
        failures += 1
    if not check_attn_pack_v_keyed(512, 256, 128, 0, 0):
        failures += 1
    if not check_attn_pack_v_keyed(512, 256, 96, 1, 256):
        failures += 1
    if not oracle_attn_pack_v_keyed_flipped():
        failures += 1

    # The attention K scatter pack (attention slice 2). The binding's two
    # semantic clauses, the routing that keys the site to its own flag, the
    # packer's byte identity against the scalar weight rule — the render's
    # per-call geometry (d_tiles=4) at three origins, the odd tail, the
    # all-scalar edge, and the 184 census's deepest production run
    # (out-proj B[N, K] at K=12288, 384 croutons per call) — then the
    # share-level whole-panel identity at the render plan, with the
    # falsifiers that must fire.
    if not check_kgather_vsc_routing():
        failures += 1
    if not check_vscatter_words_binding():
        failures += 1
    var kv_r0 = [0, 32, 0]
    var kv_c0 = [0, 0, 64]
    for i in range(len(kv_r0)):
        if not check_pack_panel_vsc(
            96, 256, kv_r0[i], kv_c0[i], 4, "render-call"
        ):
            failures += 1
    if not check_pack_panel_vsc(96, 256, 0, 0, 3, "pairs+tail"):
        failures += 1
    if not check_pack_panel_vsc(96, 256, 0, 0, 1, "all-scalar"):
        failures += 1
    if not check_pack_panel_vsc(1536, 12352, 0, 0, 384, "184 out-proj"):
        failures += 1
    if not check_pack_panel_vsc(1536, 12352, 1504, 64, 384, "184 out-proj"):
        failures += 1
    if not oracle_pack_panel_vsc_shifted():
        failures += 1
    if not check_attn_pack_k_vsc(512, 256, 128, 0, 0, 0):
        failures += 1
    if not check_attn_pack_k_vsc(512, 256, 128, 1, 256, 0):
        failures += 1
    if not check_attn_pack_k_vsc(512, 256, 96, 1, 0, 32):
        failures += 1
    if not oracle_attn_pack_k_vsc_flipped():
        failures += 1

    # The vector readout (order 146). Byte identity against the scalar unpack
    # over the geometry classes the masked store can differ across: `cols` is
    # the row stride in 64-byte runs, and its PARITY decides whether the two
    # rows of a crouton share a window half, so 32/96 (odd) and 64 (even) are
    # different codepaths through one body. Non-zero origins are a stripe's
    # m-panel share and its column tile.
    if not check_unpack_vec_admits():
        failures += 1
    if not check_unpack_arm_codes():
        failures += 1
    var uts = [1, 1, 2, 2, 3, 1]
    var ur0 = [0, 32, 0, 32, 0, 0]
    var uc0 = [0, 0, 0, 32, 0, 32]
    var ucols = [32, 96, 64, 96, 96, 64]
    for i in range(len(uts)):
        for ids in [False, True]:
            if not check_unpack_vec(uts[i], ur0[i], uc0[i], ucols[i], ids):
                failures += 1
    for cols in [64, 96]:
        if not check_unpack_vec_control(2, cols):
            failures += 1
        if not oracle_unpack_vec_deal_mutant(2, cols):
            failures += 1
    if not oracle_unpack_vec_mask_swap(2, 96):
        failures += 1

    # The microtile with its MMA modelled. `k_tiles` even, because the seam
    # carries the depth as 64-element vectors; the odd case is the refusal.
    for tiles in [2, 4, 8]:
        if not check_microtile(tiles):
            failures += 1
    if not oracle_microtile_perturbed(2):
        failures += 1
    if not check_microtile_refusal():
        failures += 1

    # The residency and the predicate the kernel entry supplies (P5 M2.3).
    if not check_accepts():
        failures += 1
    for tiles in [2, 4]:
        if not check_scratch_carve(tiles):
            failures += 1

    # The operand-port probe's arithmetic (order 159R). Its device leg is a
    # rate; what is checkable here is the number under it.
    if not check_port_probe_geometry():
        failures += 1
    if not check_port_probe_fold():
        failures += 1

    # The pipelined stripe (P5 M4.2): the split, the arena, and the differential
    # against the single-threaded nest the array's exactness was banked on.
    var share_totals = [0, 1, 3, 4, 7, 96, 384]
    for i in range(len(share_totals)):
        if not check_work_share(share_totals[i], 4):
            failures += 1
    for slots in [1, 2, 3]:
        if not check_stripe_arena(4, slots):
            failures += 1

    # `act_slots = 1` forces one A-block per row tile, which is the multi-block
    # path; `2` covers the whole M in one block. Both must agree with the nest
    # that packs an activation panel per output tile.
    if not check_pipelined(64, 96, 128, 1):
        failures += 1
    if not check_pipelined(64, 96, 128, 2):
        failures += 1
    # More column tiles than double-buffer slots, so the weight slots alternate
    # several times and the readouts trail two steps behind.
    if not check_pipelined(64, 160, 128, 2):
        failures += 1
    if not oracle_pipelined_short(64, 96, 128, 2):
        failures += 1

    # R4a (RENDEZVOUS_DESIGN.md §S.1): the folded schedule over every plan
    # class the fold distinguishes — multi-block at odd and even `n_tiles`
    # (both parities of the folded readout set), a three-block chain, the
    # `blocks == 1` degeneracy (today's schedule exactly), the single-column
    # and uneven-share VAE classes, and the degraded feeder counts. Then the
    # two controls: the kept drain dropped, and the fold one step early.
    if not check_pipelined_folded(64, 96, 128, 1):
        failures += 1
    if not check_pipelined_folded(64, 128, 128, 1):
        failures += 1
    if not check_pipelined_folded(96, 96, 128, 1):
        failures += 1
    if not check_pipelined_folded(64, 96, 128, 2):
        failures += 1
    if not check_pipelined_folded(64, 160, 128, 2):
        failures += 1
    if not check_pipelined_folded(64, 32, 320, 1):
        failures += 1
    if not check_pipelined_folded(256, 32, 576, 2):
        failures += 1
    for fold_feeders in [1, 2, 3]:
        if not check_pipelined_folded(64, 96, 128, 1, fold_feeders):
            failures += 1
    if not oracle_fold_short(64, 96, 128, 1):
        failures += 1
    if not oracle_fold_early(64, 96, 128, 1):
        failures += 1
    if not check_fold_census():
        failures += 1

    # The weight-native arm (P5 M4.3): the offline repack's tile order, the
    # ring geometry it feeds, and the differential that keeps the device A/B a
    # bit-identity gate rather than a tolerance.
    for i in range(len(ts)):
        if not check_repack_cpm(64, 64 * ts[i]):
            failures += 1
    if not check_repack_cpm(96, 128):
        failures += 1
    if not oracle_repack_order(96, 128):
        failures += 1
    for k_tiles in [2, 4]:
        for slots in [1, 3]:
            if not check_wgt_ring(k_tiles, slots):
                failures += 1
    if not check_render_plan():
        failures += 1
    if not check_encoder_plan():
        failures += 1
    if not check_native_stripe(64, 96, 128, 1):
        failures += 1
    if not check_native_stripe(64, 96, 128, 2):
        failures += 1
    if not check_native_stripe(64, 160, 128, 2):
        failures += 1
    for feeders in [1, 2, 3]:
        if not check_native_stripe(64, 96, 128, 2, feeders):
            failures += 1
    if not oracle_native_unrepacked(64, 96, 128, 2):
        failures += 1

    # The VAE decode's own accepted shapes. Order 109 read the array-native
    # skel's VAE failure as a NARROW-CHANNEL defect ("channels narrow to 32"),
    # which would be a shape bug in these nests; the shapes it named are here
    # and the nests agree bit for bit, so the wrongness is the OPERAND, not the
    # geometry (`VAE_NARROW_CHANNEL_DIAGNOSIS.md`). They also cover three
    # classes the 128-deep shapes above cannot: `N = 32` is a single column
    # tile, so the weight ring never alternates; `K = 320` and `K = 576` are
    # 64-aligned but not 128-aligned, so the crouton share splits unevenly over
    # four feeders; and `K = 1152` is 36 croutons, past the 32-tile `:deep`
    # chunk cap, so one output tile takes several chunks into one accumulator.
    if not check_pipelined(64, 32, 320, 1):
        failures += 1
    if not check_pipelined(64, 32, 320, 2):
        failures += 1
    if not check_pipelined(256, 32, 576, 2):
        failures += 1
    if not check_native_stripe(64, 32, 320, 2):
        failures += 1
    if not check_native_stripe(224, 128, 1152, 1):
        failures += 1
    # The defect itself, at a shape the VAE really dispatches: an array-native
    # skel handed the row-major weight every hand driver stages.
    if not oracle_native_unrepacked(64, 32, 320, 2):
        failures += 1
    if not oracle_native_no_await(64, 96, 128, 2):
        failures += 1
    if not oracle_native_no_await(64, 160, 128, 1):
        failures += 1
    if not oracle_native_stale_dstate(64, 96, 128, 2):
        failures += 1
    if not oracle_native_stale_dstate(64, 160, 128, 1):
        failures += 1
    if not oracle_native_posted_write(64, 96, 128, 2):
        failures += 1
    if not oracle_native_posted_write(64, 160, 128, 1):
        failures += 1

    # Annihilation rung 1: the activation/output-native arms against the
    # reference nest, each side alone and both together, at a ring-alternating
    # shape and at the uneven-share `K = 320`. Then the two failing controls —
    # the fourth and fifth members of the weights-native mismatch family need
    # their own demonstrations, side by side with the third's.
    if not check_native_io[True, False](64, 96, 128, 2, "act-native"):
        failures += 1
    if not check_native_io[True, False](64, 32, 320, 2, "act-native"):
        failures += 1
    if not check_native_io[False, True](64, 96, 128, 2, "out-native"):
        failures += 1
    if not check_native_io[False, True](64, 160, 128, 1, "out-native"):
        failures += 1
    if not check_native_io[True, True](64, 96, 128, 2, "act+out-native"):
        failures += 1
    if not oracle_native_act_unrepacked(64, 96, 128, 2):
        failures += 1
    if not oracle_native_out_misread(64, 96, 128, 2):
        failures += 1

    # P5 M4.2 order 084: the roles, the arrival counts, and the degraded
    # participant counts the launcher may produce but never has.
    if not check_stripe_roles():
        failures += 1
    if not check_arrival_counts():
        failures += 1
    if not oracle_arrival_divergence():
        failures += 1
    for feeders in [1, 2, 3]:
        if not check_pipelined(64, 96, 128, 2, feeders):
            failures += 1

    # Attention on the array (P5 slices 2 and 3). The permute the S/P seam is
    # built on first, then the residency, then the whole chain against dense
    # attention -- and the controls that make that tolerance mean something.
    if not check_vdeal_inverse():
        failures += 1
    if not check_vdeal_separates():
        failures += 1

    # The production block geometry, and a small one the flash checks below run.
    if not check_attn_arena(ATTN_BR, ATTN_BC, 128, ATTN_THREADS):
        failures += 1
    if not check_attn_arena(64, 64, 64, 1):
        failures += 1
    if not hmx_attn_accepts(hmx_attn_plan(ATTN_BR, ATTN_BC, 128, 4), 1536):
        print("  FAIL attn plan refuses the render's own shape")
        failures += 1
    # An odd column-tile count has no partner for the seam's permute, and a
    # sequence that does not divide has no clean tail; both must be refused
    # rather than truncated.
    if hmx_attn_plan(64, 32, 64, 1).usable():
        print("  FAIL attn plan accepted an odd bc_tiles")
        failures += 1
    if hmx_attn_accepts(hmx_attn_plan(64, 64, 64, 1), 96):
        print("  FAIL attn accepts a sequence its blocks do not divide")
        failures += 1

    # The probe borrows panels rather than carving its own, so the plan it is
    # handed has to be big enough -- a `False` here is what stops an overrun
    # from reporting an accumulator width.
    if hmx_attn_probe_fits(hmx_attn_plan(HMX_TILE, 2 * HMX_TILE, HMX_TILE, 1)):
        print("  FAIL attn probe accepted a plan too small to hold it")
        failures += 1
    if not hmx_attn_probe_fits(hmx_attn_plan(ATTN_BR, ATTN_BC, 128, 4)):
        print("  FAIL attn probe refuses the render's own plan")
        failures += 1
    for variant in [0, 1]:
        if not check_attn_acc_probe(variant):
            failures += 1

    # `br == seq` is one query block; `br < seq` walks several and re-reads the
    # whole key stream per block, which is the shape the render runs. Four column
    # tiles exercise more than one seam step per row tile.
    for reverse in [False, True]:
        if not check_attn_flash(64, 64, 64, 128, reverse):
            failures += 1
        if not check_attn_flash(128, 64, 64, 128, reverse):
            failures += 1
        if not check_attn_flash(32, 128, 32, 128, reverse):
            failures += 1

    if not oracle_attn_mutation(64, 64, 64, 128, 1, "rescale-dropped"):
        failures += 1
    if not oracle_attn_mutation(64, 64, 64, 128, 2, "kv-rules-swapped"):
        failures += 1
    if not oracle_attn_mutation(64, 64, 64, 128, 3, "qk-scale-dropped"):
        failures += 1

    if not check_attn_sched_shape():
        failures += 1
    if not check_attn_team_arrivals():
        failures += 1
    # Feeder counts either side of the production four, and a `br_tiles` that
    # does not divide them, so an empty share is exercised rather than assumed.
    if not check_attn_team(64, 64, 64, 128, 2, 4):
        failures += 1
    if not check_attn_team(64, 64, 64, 128, 2, 1):
        failures += 1
    if not check_attn_team(64, 64, 64, 128, 3, 3):
        failures += 1
    if not check_attn_team(128, 64, 64, 128, 2, 4):
        failures += 1
    if not oracle_attn_reuse_hazard(64, 64, 64, 128, 2, 4):
        failures += 1
    # The epilogue transpose must return the shipped path's bits exactly —
    # multi-tile, multi-head, multi-feeder, and the production-geometry block
    # (Br 512, Bc 256, D 128) where a row tile's 32 lanes span two slab-pair
    # parities per feeder share.
    if not check_attn_epi_xpose_bit_identity(64, 64, 64, 128, 2, 4):
        failures += 1
    if not check_attn_epi_xpose_bit_identity(64, 64, 64, 128, 3, 3):
        failures += 1
    if not check_attn_epi_xpose_bit_identity(512, 256, 128, 512, 1, 4):
        failures += 1
    # The masked arm. `(128, 64, ..., seq=128)` is the encoder's own block
    # structure — one query block over two key blocks, which is what `seq` 512
    # against `Br` 512 and `Bc` 256 is — and `(64, 64, ...)` puts a second query
    # block behind it so a mask indexed from the block rather than from the
    # sequence is a wrong answer instead of a coincidence at `q0 == 0`.
    if not check_attn_team_masked(128, 64, 64, 128, 2, 4, 124):
        failures += 1
    if not check_attn_team_masked(64, 64, 64, 128, 2, 4, 124):
        failures += 1
    if not check_attn_team_masked(64, 64, 64, 128, 3, 3, 128):
        failures += 1
    if not oracle_attn_mask_off(64, 64, 64, 128, 2, 4, 124):
        failures += 1
    if not check_attn_mask_inert(64, 64, 64, 128, 2, 4):
        failures += 1
    if not check_attn_mask_dose(64, 64, 64, 128, 2, 4, 124):
        failures += 1
    if not check_attn_mask_dose(64, 64, 64, 128, 2, 4, 124, True):
        failures += 1
    # The ENCODER's real shape, which is also the render's: seq 512 over
    # Br 512 / Bc 256 / D 128. The small shapes above exercise the schedule;
    # this one is the shape the fix has to hold at.
    if not check_attn_mask_dose(512, 256, 128, 512, 1, 4, 508, True):
        failures += 1
    # The render's own geometry (Br 512 / Bc 256 / D 128, orders 118-123), one
    # head, so the envelope is measured where the dense arm actually runs.
    if not check_attn_dense_envelope(512, 256, 128, 512, 1, 4):
        failures += 1

    print()
    if failures == 0:
        print("ALL HMX CROUTON TESTS PASSED")
    else:
        raise Error(String(failures) + " crouton check(s) FAILED")
