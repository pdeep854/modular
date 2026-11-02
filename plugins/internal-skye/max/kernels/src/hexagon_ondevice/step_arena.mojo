"""The step-scoped VTCM arena, carved once and named by region (slice 6).

Today every VTCM consumer resolves the grant for itself. The matmul entry does
it twice per dispatch -- once on the array thread inside `_matmul_hmx_open` to
decide whether the shape fits, once on every thread after the rendezvous to
carve the stripe -- and the staged norm does it a third time from a different
kernel, with neither consumer aware the other exists. Order 084 is what that
costs: `qshimGetBaseVtcmAddr` acquires the grant lazily with no once-guard, so
"resolve it on one thread and let the rendezvous broadcast the answer" is a
correctness rule enforced by convention at each call site rather than by a type.

`StepArena` is that base, resolved once and handed around. It carries the
aligned base and the budget that survived the alignment, and every region inside
it has a **comptime id** so a write into arena scratch is nameable by the same
mask the block program's hazard proof already uses for activation slots.

## What this module discharges, against the four-constraint contract

`FEED_ARCHITECTURE.md` sec 5.1a and `COMPOSED_LAUNCH_DESIGN.md` state the same
four clauses; both must be amended in one change. This module's part:

- **C1 (per-worker view resolution is a pure function).** Every address here is
  a closed form over `(comptime region id, geometry, base)`. Nothing takes a
  value a dispatcher would have to supply between ops, and `worker_share` is the
  one entry that takes `thread_id` -- delegating to `hmx_work_share`, whose own
  docstring records the property C1 needs ("every thread computes this
  independently and reaches the same answer"). `gate_purity` in
  the twin is the check that this stays true.
- **C2 (the row partition stays blocked and row-local).** Discharged by NOT
  participating: the arena is purely spatial. It hands out byte ranges, never
  row ranges, so neither `_thread_rows` nor `hmx_work_share` is restated,
  reinterpreted or wrapped here. The one partition this module touches is the
  weight-staging k-block, which is on the contraction axis.
- **C3 (every write gets a region id).** `ARENA_REGION_BASE` places the region
  ids directly above the block's `Slot` ids in one 64-bit mask space, so an op
  that stages a weight panel into ring slot `b` sets bit
  `ARENA_REGION_BASE + R_WGT + b` in its `writes()` and the existing hazard
  predicate covers it with no new machinery.
- **C4 (`BARRIER` lowers to the pool's rendezvous).** Discharged by not adding a
  sync form: this module contains no rendezvous, no counter, no poll and no
  atomic. A grep for `external_call` in this file returns nothing.

## What it deliberately does not do

It does not size itself. A step-scoped arena is the **max over the step's
matmul shapes** (`MEGAKERNEL_IMPLEMENTATION_PLAN.md` sec 3.1: every shape lands
at 91-98% of an 8 MiB grant, so two shapes' residency cannot coexist and the
arena is re-carved per stage), and that max is only a comptime constant once
each op's `(M, N, K)` is known at comptime -- which is slice 8. Until then
`scratch_base` refuses rather than guessing, so a caller cannot silently place
the non-HMX consumer on top of an arena sized for one shape.

The staged norm now comes through `scratch_top` rather than carving from the
same base the stripe carves from, which is where the two consumers become
disjoint by construction. It is `scratch_top` and not `scratch_base` because
the norm is entered through the manifest with its own extents and no step
geometry: it can price itself, it cannot price the step. What keeps the two
ends apart is `test_step_arena.mojo` gate 7, which fails the BUILD for a
geometry where they would meet -- and it states that as the ADDRESS relation
(`scratch_top_offset` at or above the step's maximum) rather than as a byte
comparison of residue against requirement, because the two differ by up to one
crouton of alignment and the byte comparison is the loose one.
"""

from std.memory import AddressSpace, UnsafePointer
from std.math import align_down, align_up

from .block_stack import NUM_SLOTS
from .hmx_crouton import (
    HMX_DTYPE,
    HMX_OUT_SETS,
    HMX_RECORD_BYTES,
    HMX_STRIPE_GUARD_CROUTONS,
    HMX_TILE_BYTES,
    HMX_WGT_RING,
    HmxStripeArena,
    HmxTileBytes,
    hmx_stripe_arena_bytes,
    hmx_stripe_arena_in,
)
from .matmul_kernel import HmxStripePlan, hmx_work_share

# ===-----------------------------------------------------------------------===#
# Region ids.
#
# One mask space with the block's activation slots, which is what makes C3 free:
# `_hazard` in `block_program.mojo` tests `writes(a) & reads(b)` over a `UInt64`
# of `Slot` ids, and a region id is just a slot id the arena owns. Weight ring
# slots are step-scoped SCRATCH -- the exclusion the barrier proof relies on is
# "no op writes the weight ARTEFACT", which staging into a ring slot does not
# violate, but staging IS a write and has to be visible to the proof.
# ===-----------------------------------------------------------------------===#

comptime ARENA_REGION_BASE = NUM_SLOTS
"""First bit the arena's regions occupy in the hazard mask.

Directly above the block's slots so the two id spaces cannot alias. It is
`NUM_SLOTS` rather than a round number on purpose: a slot added to
`block_stack.mojo` moves this, and `assert_region_ids_fit` fails the build
before an aliased bit can turn a hazard into a silent non-hazard."""

comptime R_ACT = 0
"""The resident activation panels: `act_slots * k_tiles` croutons.

Written by the feeders (row-blocked), read by the array thread."""

comptime R_WGT = 1
"""First of `HMX_WGT_RING` weight ring slots; slot `b` is `R_WGT + b`.

Written by whoever stages the panel -- the feeders today, the engine under the
feed design's slice 3a -- and read by the array thread. This is the region C3
exists for."""

comptime R_OUT = R_WGT + HMX_WGT_RING
"""The accumulator readout ring: `HMX_OUT_SETS * act_slots` croutons, ONE region.

Written by the array thread, read by the feeders (row-blocked). One region
rather than one per set, and the reason is the ring's panel-major order: panel
`p` set `b` lives at crouton `p * HMX_OUT_SETS + b` (`out_tile`), so a SET is a
residue class of croutons, not a byte range, and a table whose regions must
tile the stripe contiguously cannot name one. Nothing is lost: the double
buffer's disjointness is a statement about the SET INDEX, which the order keeps
distinct, and it is the schedule's proof -- not a byte-extent mask -- that
carries it. A hazard bit at set granularity would also be unsound at this
table's level, since any byte range containing set `b` contains its sibling.

This region used to be `STEP_OUT_SETS` ids over two contiguous halves; the
depth is `HMX_OUT_SETS` now, imported, so the drift its old docstring warned
about -- a depth moved in the carve and not here -- is unrepresentable."""

comptime R_GUARD = R_OUT + 1
"""The poisoned crouton between the readouts and the record.

Nothing reads it for its contents. It carries an id anyway because an overrun
that lands here is a write, and order 132's corruption is what a region with no
id looks like from the outside."""

comptime R_REC = R_GUARD + 1
"""The `[2, 32]` requant record. The array thread alone writes it."""

comptime R_SCRATCH = R_REC + 1
"""The non-HMX consumer's region: the staged norm's slots.

Anchored at the TOP of the grant (`scratch_top`) and sized by the consumer,
because a kernel entered through the manifest has its own extents and no step
geometry to compute the arena's maximum from. Disjoint from every region above
by construction rather than by hope: the two meet exactly when the step's
residue stops covering the consumer, and that is a BUILD refusal
(`test_step_arena.mojo` gate 7), not a run-time race."""

comptime NUM_ARENA_REGIONS = R_SCRATCH + 1
"""Regions the arena names."""


@always_inline
def arena_region_bit(region: Int) -> UInt64:
    """The hazard-mask bit region `region` occupies."""
    return UInt64(1) << UInt64(ARENA_REGION_BASE + region)


comptime ARENA_HMX_MASK = (
    (UInt64(1) << UInt64(NUM_ARENA_REGIONS - 1)) - 1
) << UInt64(ARENA_REGION_BASE)
"""Every HMX stripe region, as one mask: `R_ACT` through `R_REC`.

What a matmul op ORs into its `writes()`, since the stripe writes all of them
across its participants. `R_SCRATCH` is excluded -- a matmul does not touch it,
and a mask that claimed otherwise would manufacture a hazard against the norm
and add a barrier nothing needs."""


@always_inline
def assert_region_ids_fit():
    """Fails the build if the region ids do not fit beside the block's slots.

    A `UInt64` mask holds 64 ids. Overflowing it wraps a region onto a slot,
    which turns a real hazard into a mask that reads clean -- the one failure
    mode of this whole scheme that is silent at run time.
    """
    comptime assert ARENA_REGION_BASE + NUM_ARENA_REGIONS <= 64, (
        "the arena's region ids do not fit above the block's slot ids in a"
        " 64-bit hazard mask; a region would alias a slot and a hazard would"
        " read clean"
    )


# ===-----------------------------------------------------------------------===#
# Region extents.
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct ArenaExtent(Copyable, ImplicitlyCopyable, Movable):
    """One region's placement inside the arena, in bytes from the base."""

    var offset: Int
    """Bytes from the arena base."""

    var bytes: Int
    """Bytes the region spans."""

    @always_inline
    def end(self) -> Int:
        return self.offset + self.bytes

    @always_inline
    def overlaps(self, other: Self) -> Bool:
        return self.offset < other.end() and other.offset < self.end()


@always_inline
def arena_region[
    dtype: DType = HMX_DTYPE
](region: Int, k_tiles: Int, act_slots: Int) -> ArenaExtent:
    """Where region `region` sits in a stripe carved for `(k_tiles, act_slots)`.

    The bump order is `hmx_stripe_arena_in`'s, region for region, because this
    is a DESCRIPTION of that carve and not a second carve: `stripe` below still
    calls the original, and `assert_regions_tile` checks this description covers
    it exactly. Splitting the description from the pointer arithmetic is what
    lets a region be named in a hazard mask without a second address computation
    that could disagree with the first.

    Parameters:
        dtype: The arm's crouton operand type. A W8A8 crouton is half the bytes,
            so every panel region halves with it.

    Args:
        region: One of the `R_*` ids.
        k_tiles: Croutons spanning the reduction.
        act_slots: Activation panels resident.

    Returns:
        The region's offset and extent in bytes. An unknown id gives an empty
        extent at the arena's end, which `assert_regions_tile` rejects.
    """
    var tile = HmxTileBytes[dtype]
    var act_bytes = act_slots * k_tiles * tile
    var wgt_slot_bytes = k_tiles * tile
    var out_ring_bytes = HMX_OUT_SETS * act_slots * tile
    var wgt0 = act_bytes
    var out0 = wgt0 + HMX_WGT_RING * wgt_slot_bytes
    var guard0 = out0 + out_ring_bytes
    var rec0 = guard0 + HMX_STRIPE_GUARD_CROUTONS * tile

    if region == R_ACT:
        return ArenaExtent(0, act_bytes)
    if region >= R_WGT and region < R_WGT + HMX_WGT_RING:
        return ArenaExtent(
            wgt0 + (region - R_WGT) * wgt_slot_bytes, wgt_slot_bytes
        )
    if region == R_OUT:
        return ArenaExtent(out0, out_ring_bytes)
    if region == R_GUARD:
        return ArenaExtent(guard0, HMX_STRIPE_GUARD_CROUTONS * tile)
    if region == R_REC:
        return ArenaExtent(rec0, HMX_RECORD_BYTES)
    return ArenaExtent(rec0 + HMX_RECORD_BYTES, 0)


@always_inline
def arena_stripe_bytes[
    dtype: DType = HMX_DTYPE
](k_tiles: Int, act_slots: Int) -> Int:
    """Bytes the described regions span, end to end.

    Read off the region table rather than recomputed, so a region added above
    and missed here is impossible by construction; the drift this CAN have is
    against `hmx_stripe_arena_bytes`, which is what `assert_regions_tile`
    catches.
    """
    var last = arena_region[dtype](R_REC, k_tiles, act_slots)
    return last.end()


def assert_regions_tile[
    dtype: DType = HMX_DTYPE
](k_tiles: Int, act_slots: Int) -> Bool:
    """Whether the region table tiles the stripe exactly: no gap, no overlap.

    Three properties, all of which a hand-maintained region table loses quietly:
    the regions are pairwise disjoint, they leave no hole, and their total is
    `hmx_stripe_arena_bytes` to the byte. The third is the one that matters
    most, because it is the assertion that binds this description to the carve
    `hmx_stripe_arena_in` actually performs -- the same role
    `test_hmx_crouton.mojo`'s round-trip plays between `hmx_stripe_arena_bytes`
    and `hmx_stripe_act_slots`, and it is what caught the guard crouton being
    added to one and not the other.
    """
    var cursor = 0
    for region in range(R_ACT, R_REC + 1):
        var extent = arena_region[dtype](region, k_tiles, act_slots)
        if extent.offset != cursor:
            return False
        cursor = extent.end()
    return cursor == hmx_stripe_arena_bytes[dtype](k_tiles, act_slots)


# ===-----------------------------------------------------------------------===#
# The arena.
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct StepArena[space: AddressSpace = AddressSpace.SHARED](Copyable, Movable):
    """The step's VTCM grant: resolved once, aligned once, named by region.

    Parameters:
        space: Where it lives. `AddressSpace.SHARED` is VTCM on this part;
            `GENERIC` is what the host twin says.
    """

    var base: Int
    """The aligned base. 0 means the grant was refused, and every accessor
    below inherits that: a refused arena hands out no addresses."""

    var budget: Int
    """Bytes usable from `base`, after the alignment spent its own.

    Never the raw grant size: the alignment comes out of the budget, so a plan
    sized against this cannot carve itself past the end -- which is the bug the
    two call sites in `npu_kernel.mojo` each avoided separately and identically
    before this existed."""

    @staticmethod
    @always_inline
    def at(vtcm: Int, vtcm_bytes: Int) -> Self:
        """Opens the arena over a grant the caller has already resolved.

        The grant read stays at the call site rather than moving in here, and
        deliberately: `_qshim_get_base_vtcm_addr` acquires lazily on first use
        with no once-guard (order 084), so WHICH thread reads it and when is a
        property of the launch's rendezvous discipline, not of the arena. What
        moves here is the arithmetic that was written twice.

        Args:
            vtcm: The raw grant base, or 0 if there is none.
            vtcm_bytes: The raw grant size.

        Returns:
            The arena. `base == 0` when the grant was refused.
        """
        if vtcm == 0:
            return Self(base=0, budget=0)
        var base = align_up(vtcm, HMX_TILE_BYTES)
        return Self(base=base, budget=vtcm_bytes - (base - vtcm))

    @always_inline
    def usable(self) -> Bool:
        """Whether there is a grant to carve at all."""
        return self.base != 0

    @always_inline
    def region[
        dtype: DType = HMX_DTYPE
    ](self, region_id: Int, plan: HmxStripePlan) -> ArenaExtent:
        """Region `region_id`'s placement for a stage running `plan`.

        The C1 resolution, in its narrow spatial form: a closed form over a
        comptime id and a geometry, with no dispatcher input and no thread
        input. `plan` is a pure function of `(M, N, K, budget)` today and a
        comptime constant once slice 8's table carries each op's `(M, N, K)`,
        at which point every offset here folds.
        """
        return arena_region[dtype](region_id, plan.k_tiles, plan.act_slots)

    @always_inline
    def region_addr[
        dtype: DType = HMX_DTYPE
    ](self, region_id: Int, plan: HmxStripePlan) -> Int:
        """Region `region_id`'s address. 0 on a refused arena.

        An address rather than a pointer, which is what the grant itself is on
        this path (`Int(_qshim_get_base_vtcm_addr())`) and what lets a refusal
        be a value: `UnsafePointer` is non-nullable, so a null-returning
        accessor is not expressible and a refusal would have to be a second
        return channel. The typed views come from `stripe` below, which is where
        the refusal has already been checked.
        """
        if not self.usable():
            return 0
        return self.base + self.region[dtype](region_id, plan).offset

    @always_inline
    def stripe[
        dtype: DType = HMX_DTYPE
    ](self, plan: HmxStripePlan) -> HmxStripeArena[Self.space, dtype]:
        """The stripe's typed view over this arena's HMX regions.

        `hmx_stripe_arena_in` unchanged, at this arena's base: the arithmetic
        that decides where a panel lives stays in one place and this is a view
        over it, not a reimplementation. That is the whole reason `arena_region`
        above is a description rather than a carve.
        """
        return hmx_stripe_arena_in[Self.space, dtype](
            UnsafePointer[UInt8, MutUntrackedOrigin, address_space=Self.space](
                unsafe_from_address=self.base
            ),
            plan.k_tiles,
            plan.act_slots,
        )

    @always_inline
    def scratch_base[
        dtype: DType = HMX_DTYPE
    ](self, step_hmx_bytes: Int) -> Int:
        """Where `R_SCRATCH` begins, given the step's HMX maximum.

        `step_hmx_bytes` is the MAX over the step's matmul shapes, which is
        slice 8's to supply and is why this takes it rather than deriving it:
        derived from one shape it would place the norm's slots inside a larger
        shape's activation panels, and the two consumers would overlap exactly
        where the plan says they must not (sec 3.1: every shape lands at 91-98%
        of the grant, so the arena is sized as the max and re-carved per stage).

        Returns:
            The scratch base, or 0 when the maximum leaves no room -- which a
            caller must treat as a refusal and not as an address.
        """
        if not self.usable() or step_hmx_bytes < 0:
            return 0
        var start = self.base + step_hmx_bytes
        if step_hmx_bytes >= self.budget:
            return 0
        return start

    @always_inline
    def scratch_top(self, want_bytes: Int) -> Int:
        """Where `R_SCRATCH` begins for a consumer needing `want_bytes`.

        `scratch_base` from the other end, and the one a C-ABI kernel can
        actually call: an entry reached through the manifest carries its own
        extents and no step geometry, so it cannot compute the arena's maximum,
        and baking one geometry's maximum into it would be wrong for every
        other. Anchoring at the top needs nothing but the consumer's own size.

        What makes that safe is not this function. The two ends meet exactly
        when the step's residue stops covering the consumer, and that pair is
        checked at BUILD time (`test_step_arena.mojo` gate 7) for every geometry
        the kernel is compiled for -- the same standing as
        `test_step_weights.mojo`'s prefetch refusal. A run-time check here
        instead would be one this kernel has no geometry to write.

        Aligned DOWN to `HMX_TILE_BYTES`, which is a multiple of the HVX vector,
        so a consumer inherits both alignments without restating either. The
        base is already `HMX_TILE_BYTES`-aligned, so the result never precedes
        it.

        Returns:
            The scratch base, or 0 when the arena cannot back `want_bytes` --
            which a caller must treat as a refusal and not as an address.
        """
        if not self.usable():
            return 0
        var offset = scratch_top_offset(self.budget, want_bytes)
        if offset == 0:
            return 0
        return self.base + offset


@always_inline
def scratch_top_offset(budget: Int, want_bytes: Int) -> Int:
    """`scratch_top`'s address as an OFFSET, for a caller with no arena.

    The build-time half of the disjointness check needs this number and has no
    arena to open: `test_step_arena.mojo` gate 7 compares where the top-anchored
    consumer BEGINS against where the step's maximum ENDS, and both have to be
    the same arithmetic the device will run or the refusal guards a different
    layout than the one that ships. So the alignment lives here, once, and
    `scratch_top` is this plus the base.

    An offset of 0 is the refusal, not the base -- a consumer beginning at the
    arena base overlaps the activation panels and every region after them, so
    there is no case where 0 is an answer.
    """
    if want_bytes <= 0 or want_bytes > budget:
        return 0
    return align_down(budget - want_bytes, HMX_TILE_BYTES)


@always_inline
def worker_share(total: Int, thread_id: Int, parts: Int) -> Int:
    """The first item worker `thread_id` owns of `total`, split `parts` ways.

    C1's `thread_id` argument, and the only one in this module. It forwards to
    `hmx_work_share` rather than restating it: the stripe's k-blocked staging
    split and the arena's view of it must be the SAME function, or a feeder
    stages a panel the array thread does not read. Forwarding is also what keeps
    C2 true here -- this is the contraction axis, and no row partition passes
    through this file.
    """
    return hmx_work_share(total, thread_id, parts)
