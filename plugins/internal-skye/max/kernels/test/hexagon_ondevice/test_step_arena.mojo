"""The step arena's region table, against the carve it describes (slice 6).

`step_arena.mojo` states where every region of the HMX stripe sits, and
`hmx_stripe_arena_in` puts them there. Two statements of one layout is exactly
the drift `hmx_stripe_act_slots`' own docstring records -- "a region added there
and missed here does not fail loudly" -- so the description is only worth having
if something binds it to the carve. That is gate 3 below, and it is why the
arena is a DESCRIPTION plus a view rather than a second carve.

Seven gates, none of which is the module restated:

1. **The id space.** The region ids sit above the block's `Slot` ids and inside
   a 64-bit mask. An alias here turns a real hazard into a mask that reads
   clean, which is the one failure of the C3 scheme that is silent at run time.

2. **The table tiles the stripe.** Regions in bump order, no gap, no overlap,
   and a total equal to `hmx_stripe_arena_bytes` to the byte -- across the six
   production shapes the audit tabulates and both arms' croutons.

3. **Description equals carve.** Every region pointer the arena computes must be
   the pointer `hmx_stripe_arena_in` hands the stripe for the same geometry.
   This is the differential: the description cannot drift from the arithmetic
   without failing here, and it holds at the byte rather than at the region
   count.

4. **C1 purity.** Resolution takes a comptime id, a geometry and a base, and
   nothing else: five workers resolving independently, in different orders,
   agree exactly. A dispatcher-supplied value would show up here as a
   disagreement, which is the whole reason the contract's clause exists.

5. **`at()` is today's arithmetic.** The two call sites in `npu_kernel.mojo`
   each aligned the grant and spent the alignment out of the budget, separately
   and identically. The hoist is only safe if the arena reproduces that to the
   byte, including the refused-grant case.

6. **The two consumers are disjoint.** `scratch_base` must place the non-HMX
   region past every shape the step's maximum covers, and must REFUSE rather
   than return an address when the maximum does not fit -- the order-084 race
   class, closed by construction instead of by both consumers hoping.

7. **The staged norm's scratch clears the step's maximum, and the two ends meet
   nowhere.** Disjoint is not the same as fits: where the step's other VTCM
   consumer BEGINS has to be at or past where the arena's step-scoped maximum
   ENDS. That is a property of the compiled geometry, so it is a `comptime
   assert` and a geometry that breaks it fails the BUILD, naming itself. It is
   asserted as that address relation and not as `residue >= want`, which is
   looser by up to a crouton of alignment; `gate_loose_window` carries a case
   inside the gap as a standing control. The run-time half binds the two
   accessors: `scratch_top`, which is where the staged norm now carves, must
   land at or after `scratch_base`, which is where the maximum ends.

Mutation controls, each applied by hand and reverted (recorded in the PR):
drop the guard crouton from `arena_region` (gate 2), drop the readout ring's
term from `arena_region` (gate 2), swap the weight and readout regions' order
(gates 2 and 3), revert `out_tile` to buffer-major (gate 3), align the grant
DOWN in `at()` (gate 5), and set `ARENA_REGION_BASE` to 0 (gate 1). The
original list mutated `STEP_OUT_SETS` to 1; that constant is gone -- the depth
is `HMX_OUT_SETS`, named once in the carve and imported here -- so a depth that
moves in the description and not the carve is no longer representable, and the
control moved to the term drop above.

For gate 7, both halves of the refusal: fold `layer_norm`'s width into
`step_staged_norm_width` at `heads = 40` (the step then stages a dim-wide row,
122,880 B against a 100,096 B residue -- the build must refuse), and pass the
arena's own maximum as the budget to `step_arena_residue` (residue 0 -- the
build must refuse). The first control is the one that matters: it fires on a
geometry nobody would flag by reading the numbers. Gate 7's third control does
not need a hand: `gate_loose_window` runs the refused case every time.
"""

from std.math import align_up
from std.memory import AddressSpace

from layout.coord import Coord

from hexagon_ondevice.block_stack import NUM_SLOTS, klein_512, klein_mini
from hexagon_ondevice.rms_norm_kernel import staged_norm_bytes
from hexagon_ondevice.hmx_crouton import (
    HMX_DTYPE,
    HMX_OUT_SETS,
    HMX_RECORD_BYTES,
    HMX_STRIPE_GUARD_CROUTONS,
    HMX_TILE_BYTES,
    HMX_WGT_RING,
    HmxTileBytes,
    hmx_stripe_arena_bytes,
    hmx_stripe_arena_in,
)
from hexagon_ondevice.matmul_kernel import HmxStripePlan, hmx_stripe_plan
from hexagon_ondevice.step_arena import (
    ARENA_HMX_MASK,
    ARENA_REGION_BASE,
    NUM_ARENA_REGIONS,
    R_ACT,
    R_GUARD,
    R_OUT,
    R_REC,
    R_SCRATCH,
    R_WGT,
    StepArena,
    arena_region,
    arena_region_bit,
    arena_stripe_bytes,
    assert_region_ids_fit,
    assert_regions_tile,
    scratch_top_offset,
    worker_share,
)
from hexagon_ondevice.step_weights import (
    step_arena_bytes,
    step_arena_residue,
    step_staged_norm_width,
)

comptime HMX_INT8_CROUTON = DType.uint8
"""The W8A8 arm's crouton type: half the bytes for the same geometry, which is
why every region extent has to be parametric rather than measured once."""

comptime TWIN_SPACE = AddressSpace.GENERIC
"""What the twin says where the device says VTCM."""


@fieldwise_init
struct Shape(Copyable, ImplicitlyCopyable, Movable):
    """One production geometry, from `CACHE_ROOFLINE_AUDIT.md` sec 3.1's plan
    table: the six shapes a Klein step's matmuls actually run."""

    var name: StaticString
    var k_tiles: Int
    var act_slots: Int


def production_shapes() -> List[Shape]:
    """The audit's own table, at the 8 MiB grant it was computed against.

    Every one of these lands at 91-98% of the budget, which is the fact that
    makes the arena a per-stage re-carve of a step-scoped maximum rather than a
    per-shape residency. A table that is right only for the small shapes would
    pass a made-up geometry and fail here.
    """
    var shapes = List[Shape]()
    shapes.append(Shape("sgl out-proj", 384, 8))
    shapes.append(Shape("sgl fused mlp", 96, 39))
    shapes.append(Shape("sgl fused q/k/v", 96, 39))
    shapes.append(Shape("dbl ff_out img", 288, 12))
    shapes.append(Shape("dbl ff_in img", 96, 39))
    shapes.append(Shape("dbl qkv img", 96, 39))
    # Degenerate ends: one slot, and a k extent of one crouton. The closed forms
    # are linear so these are where an off-by-one in the fixed term shows.
    shapes.append(Shape("minimum", 1, 1))
    shapes.append(Shape("thin k", 1, 64))
    return shapes^


def gate_id_space() -> Bool:
    """Gate 1: the region ids fit, and do not alias the block's slots."""
    assert_region_ids_fit()

    # The fit itself is `assert_region_ids_fit`'s, and it is a COMPTIME assert:
    # a restatement here would be a check the compiler has already folded away.
    # What is left to check at run time is what the folding cannot see -- that
    # the two id spaces really are disjoint bit by bit.
    var ok = True

    # A slot's bit and a region's bit must never be the same bit.
    var slot_mask = UInt64(0)
    for slot in range(NUM_SLOTS):
        slot_mask |= UInt64(1) << UInt64(slot)
    var region_mask = UInt64(0)
    for region in range(NUM_ARENA_REGIONS):
        region_mask |= arena_region_bit(region)
    if slot_mask & region_mask != 0:
        print("  id space: a region aliases an activation slot")
        ok = False

    # The mask a matmul op ORs into `writes()` covers every HMX region and NOT
    # the scratch: claiming scratch would manufacture a hazard against the norm
    # and add a barrier nothing needs.
    var want = UInt64(0)
    for region in range(R_ACT, R_REC + 1):
        want |= arena_region_bit(region)
    if ARENA_HMX_MASK != want:
        print("  id space: ARENA_HMX_MASK is not R_ACT..R_REC")
        ok = False
    if ARENA_HMX_MASK & arena_region_bit(R_SCRATCH) != 0:
        print("  id space: ARENA_HMX_MASK claims the scratch region")
        ok = False
    return ok


def check_tiles[dtype: DType](shape: Shape) -> Bool:
    """Gate 2 for one shape and one arm."""
    var ok = True
    if not assert_regions_tile[dtype](shape.k_tiles, shape.act_slots):
        print("  tiling:", shape.name, "regions do not tile the stripe")
        ok = False

    var total = arena_stripe_bytes[dtype](shape.k_tiles, shape.act_slots)
    var want = hmx_stripe_arena_bytes[dtype](shape.k_tiles, shape.act_slots)
    if total != want:
        print("  tiling:", shape.name, "total", total, "want", want)
        ok = False

    # Pairwise disjointness, stated separately from the bump-order walk: the
    # walk proves contiguity, this proves no two ids name overlapping bytes even
    # if the order were rewritten.
    for a in range(R_ACT, R_REC + 1):
        var ra = arena_region[dtype](a, shape.k_tiles, shape.act_slots)
        for b in range(a + 1, R_REC + 1):
            var rb = arena_region[dtype](b, shape.k_tiles, shape.act_slots)
            if ra.overlaps(rb):
                print(
                    "  tiling:",
                    shape.name,
                    "regions",
                    a,
                    "and",
                    b,
                    "overlap",
                )
                ok = False

    # Every panel region is a whole number of croutons from the base, which is
    # the `:deep` load's requirement and the reason only the BASE is aligned.
    var tile = HmxTileBytes[dtype]
    for region in range(R_ACT, R_GUARD + 1):
        var extent = arena_region[dtype](region, shape.k_tiles, shape.act_slots)
        if extent.offset % tile != 0 or extent.bytes % tile != 0:
            print(
                "  tiling:",
                shape.name,
                "region",
                region,
                "is not crouton-aligned",
            )
            ok = False
    var rec = arena_region[dtype](R_REC, shape.k_tiles, shape.act_slots)
    if rec.offset % tile != 0:
        print("  tiling:", shape.name, "the record is not crouton-aligned")
        ok = False
    if rec.bytes != HMX_RECORD_BYTES:
        print("  tiling:", shape.name, "the record is", rec.bytes, "bytes")
        ok = False
    return ok


def _same(
    shape: Shape, name: StaticString, described: Int, carved: Int
) -> Bool:
    if described == carved:
        return True
    print(
        "  carve:", shape.name, name, "described", described, "carved", carved
    )
    return False


def check_description_equals_carve[dtype: DType](shape: Shape) -> Bool:
    """Gate 3: the described addresses ARE the carved addresses.

    The stripe view comes from `hmx_stripe_arena_in` unchanged, so this compares
    the region table against the only arithmetic that has ever placed a panel.
    A pointer, not an extent: an extent table can be self-consistently wrong.

    No backing allocation, and that is not a shortcut: both sides are pointer
    arithmetic on a base and neither dereferences, so a real buffer would only
    bound which shapes the twin could ask about -- and the widest shapes are
    exactly the ones an offset bug hides in.
    """
    var arena = StepArena[TWIN_SPACE](base=0x4000_0000, budget=1 << 24)
    var plan = HmxStripePlan(
        shape.act_slots, 1, shape.k_tiles, shape.act_slots, 1
    )
    var stripe = arena.stripe[dtype](plan)

    var ok = True
    if not _same(
        shape,
        "act",
        arena.region_addr[dtype](R_ACT, plan),
        Int(stripe.act.ptr_at_offset(Coord(0, 0, 0, 0))),
    ):
        ok = False
    for b in range(HMX_WGT_RING):
        # `wgt_panel` takes the output COLUMN TILE, and column tile `b` lands in
        # ring slot `b % HMX_WGT_RING`, so the low column tiles name the slots
        # in order.
        if not _same(
            shape,
            "wgt",
            arena.region_addr[dtype](R_WGT + b, plan),
            Int(stripe.wgt_panel(b).ptr_at_offset(Coord(0, 0, 0, 0))),
        ):
            ok = False
    # The readout ring is ONE region, panel-major inside: panel `p` set `b` at
    # crouton `p * HMX_OUT_SETS + b`. The description names the ring; where a
    # (set, panel) sits WITHIN it is `out_tile`'s statement, so both are bound
    # here -- the region's base to the first tile, and the interleaving to the
    # carve, which is what the two per-set contiguous ids this replaced could
    # not describe.
    var ring = arena.region_addr[dtype](R_OUT, plan)
    var tile_bytes = HmxTileBytes[dtype]
    for b in range(HMX_OUT_SETS):
        for p in range(shape.act_slots):
            if not _same(
                shape,
                "out",
                ring + (p * HMX_OUT_SETS + b) * tile_bytes,
                Int(stripe.out_tile(b, p).ptr_at_offset(Coord(0, 0, 0, 0))),
            ):
                ok = False
    if not _same(
        shape,
        "guard",
        arena.region_addr[dtype](R_GUARD, plan),
        Int(stripe.guard.ptr_at_offset(Coord(0, 0, 0, 0))),
    ):
        ok = False
    if not _same(
        shape,
        "record",
        arena.region_addr[dtype](R_REC, plan),
        Int(stripe.record.ptr_at_offset(Coord(0, 0))),
    ):
        ok = False
    return ok


def gate_purity() -> Bool:
    """Gate 4: five workers resolve the same layout, independently.

    The contract's C1 is a property of the SIGNATURE -- no dispatcher argument
    -- and this is what that property looks like from outside: whatever order
    the workers ask in, and whichever worker asks, the answer is the same
    function of the id, the geometry and the base.
    """
    var arena = StepArena[TWIN_SPACE](base=1 << 20, budget=1 << 23)
    var plan = hmx_stripe_plan(1536, 3072, 3072, 1 << 23)
    var ok = True

    var forward = List[Int]()
    for region in range(R_ACT, R_REC + 1):
        forward.append(arena.region_addr(region, plan))

    # Backwards, and once per worker: a resolution that consulted anything but
    # its arguments would have to agree with itself here by luck.
    for thread_id in range(5):
        var seen = List[Int]()
        for i in range(R_REC + 1):
            var region = R_REC - i
            seen.append(arena.region_addr(region, plan))
        for i in range(len(forward)):
            if seen[len(forward) - 1 - i] != forward[i]:
                print("  purity: worker", thread_id, "disagrees at region", i)
                ok = False

    # The one entry that takes a thread id partitions exactly: no item twice,
    # none left out, for any split -- including one narrower than the parts.
    for parts in range(1, 6):
        for total in range(0, 9):
            var cursor = 0
            for idx in range(parts):
                var first = worker_share(total, idx, parts)
                if first != cursor:
                    print(
                        "  purity: share", idx, "of", parts, "is not contiguous"
                    )
                    ok = False
                cursor = worker_share(total, idx + 1, parts)
            if cursor != total:
                print(
                    "  purity: shares of", total, "over", parts, "do not close"
                )
                ok = False
    return ok


def gate_grant_arithmetic() -> Bool:
    """Gate 5: `at()` is what both call sites did by hand."""
    var ok = True
    var sizes = List[Int]()
    sizes.append(8 << 20)
    sizes.append(4 << 20)
    sizes.append(HMX_TILE_BYTES)
    var bases = List[Int]()
    bases.append(0x1000_0000)
    bases.append(0x1000_0001)
    bases.append(0x1000_0000 + HMX_TILE_BYTES - 1)
    bases.append(0x1000_0000 + HMX_TILE_BYTES)

    for i in range(len(bases)):
        for j in range(len(sizes)):
            var vtcm = bases[i]
            var size = sizes[j]
            var arena = StepArena[TWIN_SPACE].at(vtcm, size)
            var want_base = align_up(vtcm, HMX_TILE_BYTES)
            if arena.base != want_base:
                print("  grant: base", arena.base, "want", want_base)
                ok = False
            if arena.budget != size - (want_base - vtcm):
                print(
                    "  grant: budget",
                    arena.budget,
                    "want",
                    size - (want_base - vtcm),
                )
                ok = False
            if arena.base % HMX_TILE_BYTES != 0:
                print("  grant: base is not crouton-aligned")
                ok = False

    # A refused grant is a refused arena, not an arena at address zero: the
    # `vtcm == 0` branch both call sites carry, kept.
    var refused = StepArena[TWIN_SPACE].at(0, 8 << 20)
    if refused.usable() or refused.base != 0 or refused.budget != 0:
        print("  grant: a refused grant produced a usable arena")
        ok = False
    var plan = hmx_stripe_plan(1536, 3072, 3072, 8 << 20)
    if refused.region_addr(R_ACT, plan) != 0:
        print("  grant: a refused arena handed out an address")
        ok = False
    return ok


def gate_consumers_disjoint() -> Bool:
    """Gate 6: the norm's region begins past every shape the maximum covers."""
    var budget = 8 << 20
    var arena = StepArena[TWIN_SPACE].at(0x2000_0000, budget)
    var ok = True

    # The step's maximum, taken over the audit's shapes exactly as slice 8 will
    # take it at comptime.
    var step_max = 0
    var shapes = production_shapes()
    for i in range(len(shapes)):
        var bytes = hmx_stripe_arena_bytes(
            shapes[i].k_tiles, shapes[i].act_slots
        )
        if bytes > step_max:
            step_max = bytes

    var scratch = arena.scratch_base(step_max)
    if scratch == 0:
        print(
            "  disjoint: the step maximum",
            step_max,
            "does not fit in",
            arena.budget,
        )
        return False
    for i in range(len(shapes)):
        var plan = HmxStripePlan(
            shapes[i].act_slots, 1, shapes[i].k_tiles, shapes[i].act_slots, 1
        )
        var last = arena.region(R_REC, plan)
        if arena.base + last.end() > scratch:
            print("  disjoint:", shapes[i].name, "runs into the scratch region")
            ok = False

    # And it refuses rather than returning an address it cannot back.
    if arena.scratch_base(arena.budget) != 0:
        print(
            "  disjoint: a maximum filling the grant still yielded a scratch"
            " base"
        )
        ok = False
    if arena.scratch_base(arena.budget + 1) != 0:
        print("  disjoint: an overlarge maximum still yielded a scratch base")
        ok = False
    return ok


def gate_scratch_clears_arena() -> Bool:
    """Gate 7: the staged norm begins past the step's maximum, refused at BUILD.

    Gate 6 proves the two consumers do not overlap. It cannot prove the second
    one fits, because fitting is a property of the geometry the kernel is
    compiled for and not of an arena opened at run time -- so this is a
    `comptime assert`, modelled on `test_step_weights.mojo`'s prefetch refusal:
    a geometry that breaks the disjointness has to fail the build, naming
    itself, rather than reach a device and carve into the stripe.

    It is the ADDRESS relation and not `residue >= want`, which is the #407
    review's finding. The two are not the same check: `scratch_top` aligns its
    base DOWN to a crouton while the arena's last region ends 256 B past one
    (the requant record), so the byte comparison passes for a 1,792-byte window
    of geometries in which the norm's slots still begin INSIDE the stripe. The
    relation the device runs is "where the consumer starts" against "where the
    maximum ends", so that is what is asserted, through the same
    `scratch_top_offset` the kernel calls rather than a second alignment written
    here. `gate_loose_window` below is the standing control that the window is
    real and that this form closes it.

    The margin is real but it is a REMAINDER, not a headroom. At an 8 MiB grant
    the arena's maximum takes 99.05% at Klein and the residue is what floor
    division happened to leave, so it is not monotone in `dim`: 50,944 B at 16
    heads, 79,616 B at 24, 141,056 B at 32, 100,096 B at 40. The staged norm's
    requirement, by contrast, is linear in the row width. Today the step stages
    only head vectors (128 wide, 3,072 B across four threads) and every geometry
    clears it; a dim-wide row would be refused at 24 heads with `mlp_ratio = 2`
    (73,728 B needed, 50,944 left) and at 40 and 48 heads for every ratio.
    Nothing about that is visible by inspection, which is the whole argument for
    checking it here.
    """
    comptime BUDGET = 8 << 20

    comptime RESIDUE_512 = step_arena_residue(klein_512(), BUDGET)
    comptime NORM_512 = staged_norm_bytes(step_staged_norm_width(klein_512()))
    comptime assert scratch_top_offset(BUDGET, NORM_512) >= step_arena_bytes(
        klein_512(), BUDGET
    ), (
        "the staged norm's top-anchored slots begin inside the step's VTCM"
        " arena at `klein_512`: the arena's maximum over BOTH block kinds'"
        " matmul shapes takes all but a floor-division remainder of the grant,"
        " and what is left does not clear the norm's three vector-aligned rows"
        " per compute thread once the consumer's base is aligned down to a"
        " crouton. Widening the staged path past the head vector, or changing"
        " `heads`/`mlp_ratio`, re-prices both sides -- neither is monotone, so"
        " re-read the pair rather than reasoning about the direction"
    )

    comptime RESIDUE_MINI = step_arena_residue(klein_mini(), BUDGET)
    comptime NORM_MINI = staged_norm_bytes(step_staged_norm_width(klein_mini()))
    comptime assert scratch_top_offset(BUDGET, NORM_MINI) >= step_arena_bytes(
        klein_mini(), BUDGET
    ), (
        "the staged norm's top-anchored slots begin inside the step's VTCM"
        " arena at `klein_mini`, the geometry the driver iterates on -- see the"
        " `klein_512` message above for what re-prices it"
    )

    print("  klein_512: arena", step_arena_bytes(klein_512(), BUDGET))
    print(
        "  klein_512: residue",
        RESIDUE_512,
        "staged norm",
        NORM_512,
        "over width",
        step_staged_norm_width(klein_512()),
    )
    print("  klein_mini: residue", RESIDUE_MINI, "staged norm", NORM_MINI)

    var ok = True

    # The residue is the scratch region's capacity, so the arena has to agree
    # with it: `scratch_base` past the maximum, plus the residue, is the end of
    # the budget. A disagreement here means the guard above priced a region the
    # carve does not actually leave.
    var arena = StepArena[TWIN_SPACE].at(0x2000_0000, BUDGET)
    var scratch = arena.scratch_base(step_arena_bytes(klein_512(), BUDGET))
    if scratch == 0:
        print("  residue: klein_512's maximum left no scratch base at all")
        return False
    if scratch + RESIDUE_512 != arena.base + arena.budget:
        print(
            "  residue:",
            RESIDUE_512,
            "does not reach the end of the budget from",
            scratch,
        )
        ok = False

    # And it refuses rather than under-reporting when nothing is left.
    if step_arena_residue(klein_512(), 1024) != 0:
        print("  residue: a budget no shape fits still reported a remainder")
        ok = False

    # The two ends of the arena, bound together. This is the invariant the
    # staged norm's reroute rests on: the consumer placed from the top begins
    # at or after where the step's maximum ends, so the guard above is not a
    # byte comparison that happens to hold but the address relation itself.
    var top = arena.scratch_top(NORM_512)
    if top == 0:
        print("  residue: the staged norm found no scratch at the top")
        return False
    if top < scratch:
        print("  residue: the top-anchored scratch", top, "runs into", scratch)
        ok = False
    if top % HMX_TILE_BYTES != 0:
        print("  residue: the top-anchored scratch is not crouton-aligned")
        ok = False
    if top + NORM_512 > arena.base + arena.budget:
        print("  residue: the top-anchored scratch runs past the budget")
        ok = False

    # `scratch_top` is the offset plus the base and nothing else, which is what
    # lets the comptime guard above use the offset alone.
    if top != arena.base + scratch_top_offset(BUDGET, NORM_512):
        print("  residue: scratch_top is not scratch_top_offset plus the base")
        ok = False

    # A consumer larger than the whole grant is a refusal, not an address that
    # would place its slots below the base.
    if arena.scratch_top(BUDGET + 1) != 0:
        print("  residue: an overlarge consumer still got a scratch base")
        ok = False
    if arena.scratch_top(0) != 0:
        print("  residue: a zero-sized consumer still got a scratch base")
        ok = False
    return ok


def gate_loose_window() -> Bool:
    """Gate 7's control: the byte comparison the address relation replaced.

    A guard that never fires is indistinguishable from one that cannot, so the
    window the #407 review found is carried here as a case rather than as an
    argument. `klein_512` at a 8,426,752 B grant is inside it: the step's
    maximum ends 8,423,680 B in, the residue is 3,072 B, and the staged norm
    wants 3,072 B -- so `residue >= want` holds EXACTLY, while the consumer's
    base, aligned down to a crouton, lands at 8,423,424 and its first 256 bytes
    sit on the stripe's requant record.

    The window is 1,792 bytes wide and not 2,047, because the arena's end is
    congruent to `HMX_RECORD_BYTES` modulo the crouton: every geometry with a
    residue-minus-requirement in `[0, 1792)` is one the old form passed and this
    one refuses. Both halves are asserted -- that the old form passed, and that
    the new one refuses -- since a control that only checked the second would
    also be satisfied by a case the old form had already caught.
    """
    comptime LOOSE_BUDGET = 8426752
    comptime NORM = staged_norm_bytes(step_staged_norm_width(klein_512()))
    var arena = step_arena_bytes(klein_512(), LOOSE_BUDGET)
    var residue = step_arena_residue(klein_512(), LOOSE_BUDGET)
    var top = scratch_top_offset(LOOSE_BUDGET, NORM)
    print(
        "  loose window: arena",
        arena,
        "residue",
        residue,
        "norm",
        NORM,
        "scratch top",
        top,
    )

    var ok = True
    if residue < NORM:
        print("  loose window: the byte comparison already refused this case")
        ok = False
    if residue - NORM >= HMX_TILE_BYTES - HMX_RECORD_BYTES:
        print(
            "  loose window: margin",
            residue - NORM,
            "is outside the window this case exists to sit in",
        )
        ok = False
    if top >= arena:
        print("  loose window: the address relation did not refuse the case")
        ok = False
    if arena - top != HMX_RECORD_BYTES:
        print(
            "  loose window: the overlap is",
            arena - top,
            "bytes, not the record's",
            HMX_RECORD_BYTES,
        )
        ok = False
    return ok


def main() raises:
    var failures = 0

    print("gate 1: the region id space")
    if not gate_id_space():
        failures += 1

    print("gate 2: the region table tiles the stripe")
    var shapes = production_shapes()
    for i in range(len(shapes)):
        if not check_tiles[HMX_DTYPE](shapes[i]):
            failures += 1
        if not check_tiles[HMX_INT8_CROUTON](shapes[i]):
            failures += 1

    print("gate 3: the description is the carve")
    for i in range(len(shapes)):
        if not check_description_equals_carve[HMX_DTYPE](shapes[i]):
            failures += 1
        if not check_description_equals_carve[HMX_INT8_CROUTON](shapes[i]):
            failures += 1

    print("gate 4: resolution is pure")
    if not gate_purity():
        failures += 1

    print("gate 5: the grant arithmetic is today's")
    if not gate_grant_arithmetic():
        failures += 1

    print("gate 6: the two VTCM consumers are disjoint")
    if not gate_consumers_disjoint():
        failures += 1

    print("gate 7: the staged norm's scratch clears the step's maximum")
    if not gate_scratch_clears_arena():
        failures += 1
    if not gate_loose_window():
        failures += 1

    print()
    if failures == 0:
        print("ALL STEP ARENA TESTS PASSED")
    else:
        raise Error(String(failures) + " step arena gate(s) FAILED")
