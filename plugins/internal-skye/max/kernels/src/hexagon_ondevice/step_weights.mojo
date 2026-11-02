"""The step's weight table: every op's `(M, N, K)` at comptime (slice 8).

The arena (slice 6) can be carved once per step only if its size is a comptime
constant, and its size is the maximum over the step's matmul shapes. Every op's
geometry therefore has to be known before the step runs, not handed to a worker
when its turn comes. That requirement arrives from three directions and they are
the same requirement:

1. **The arena max folds.** `hmx_stripe_arena_bytes` over each op, maximised at
   comptime, is what `StepArena.scratch_base` needs so the non-HMX consumer can
   be placed past every shape (`MEGAKERNEL_IMPLEMENTATION_PLAN.md` sec 3.1).
2. **C1 stays true.** `resolve(region, geometry, base)` is pure only while the
   geometry is not a dispatcher argument. If `(M, N, K)` arrives per op from a
   dispatcher, that dispatcher runs between ops, the join comes back at every
   op boundary, and the barrier reduction the composition exists for is gone
   (`FEED_ARCHITECTURE.md` sec 5.1a).
3. **Residency needs it.** "Stop re-staging the same weights every step" means
   hoisting a weight stage out of the per-dispatch path, and a stage cannot be
   hoisted above the code that computes its extent (sec 11.3).

## Where the table comes from, and why it is not generated

`gen_manifest_arity.py` generates a Mojo table because its source of truth is an
X-macro `.def` that only C can expand. This table's source of truth is already
Mojo -- `double_block` and `single_block`, the transliterations of
`runDoubleBlock` and `runSingleBlock` that the gates in `test_block_stack.mojo`
hold against the driver -- so generating a second statement of the shapes would
ADD the drift the generator exists to prevent. The table is DERIVED instead: the
sequences are run through a sink that records nothing but geometry, at comptime.
A shape that changes in a block changes here with it, and no `--check` gate is
needed because there is no cache.

## The layout contract, and what it is keyed on

Rows are keyed on the weight's REAL `(N, K)` -- a property of the artefact --
and never on a schedule's `(act_slots, k_tiles, blocks)`, which depend on the
VTCM budget and on `M`. That is the shape a resident weight table has to have,
and it is the same answer the MXFP6 path reaches from the other side: its B
tensor is a fixed, M-independent grid written offline by the checkpoint
converter and "keyed on the tensor's REAL `(N_real, K_real)`, never on this
matmul's own `(kc, nc)` tile choice", with a thin per-call map from whatever
tile shape the scheduler picked onto fixed coordinates
(`WEIGHT_STUDY_B_SKYE.md` sec 2b). Ours already has that map and only that map:
`hmx_native_weight_panel` is "the only place the artefact's tile order is named
on the device" (`hmx_crouton.mojo:1618-1636`) and it reads the extents off the
artefact's own `[N, K]` view rather than off the plan.

The price of that decoupling is an alignment constraint, and the study's own
prior art is explicit that the equivalent constraint holds in practice without
being structurally guaranteed. Ours: every op's `K` must be a whole number of
croutons, because `hmx_k_tiles` divides rather than rounds. `weight_table_ok`
checks it for the whole step, so a geometry that broke it would fail a gate
instead of silently addressing a short panel.

## The prefetch fields are expressed and disabled

`is_prefetch` and `loop_distance` are the vocabulary cross-op weight prefetch
needs ("stage op `n+1`'s weights during op `n`",
`MEGAKERNEL_VM_DISPATCH_NOTE.md` sec 4). They are carried here so the table's
shape is settled, and `WEIGHT_PREFETCH_ENABLED` is a comptime `False` because
the shim carries ONE outstanding transfer per submitter and the feed design's
slice 3a probe has not yet said what the engine will take. A comptime constant
rather than a runtime flag: with it False the fields fold away entirely, so the
disabled path costs no instruction and cannot be flipped by an environment.
"""

from .block_stack import (
    BlockOps,
    KleinConfig,
    NUM_DOUBLE_WEIGHTS,
    NUM_WEIGHTS,
    SINGLE_WEIGHT_BASE,
    Slot,
    Weight,
    double_block,
    single_block,
)
from .hmx_crouton import (
    HMX_DTYPE,
    HMX_TILE,
    hmx_k_tiles,
    hmx_stripe_act_slots,
    hmx_stripe_arena_bytes,
)

comptime NO_WEIGHT = Weight(-1)
"""A weight id no block names, for a fold that is not asking about one."""


comptime WEIGHT_PREFETCH_ENABLED = False
"""Whether a row's `is_prefetch` / `loop_distance` are honoured. FALSE.

Gated on the feed design's **slice 3a** device probe: the shim carries one
outstanding transfer per submitter (order 096 leg A', KB
`known-limitations/hexagon-user-mode-dma-instructions`), so staging op `n+1`'s
panel while op `n` runs collides with the transfer op `n` is already awaiting.
3a prices what the engine will actually take -- 2D descriptors, `dmlink`
chaining, or neither -- and until it reports, a prefetch distance is a number
nothing may act on.

Comptime rather than an environment flag on purpose. A runtime flag would leave
the disabled arm's arithmetic in the object and make "is it off?" a question
about a device's environment; at comptime the fields fold and the answer is a
property of the build."""


@fieldwise_init
struct StepMatmul(Copyable, ImplicitlyCopyable, Movable):
    """One matmul op of the step, as the arena and the stager need it.

    `(m, n, k)` are the op's own extents; the weight's extent is `(n, k)` and is
    a property of the ARTEFACT, which is why two ops reading the same weight at
    different `m` name one weight row and not two.
    """

    var weight: Weight
    """Which weight this op reads."""

    var m: Int
    """Output rows: the activation extent, and the only one that varies between
    two ops sharing a weight."""

    var n: Int
    """Output columns, and the weight artefact's row count."""

    var k: Int
    """The reduction extent, and the weight artefact's column count."""

    var is_prefetch: Bool
    """Whether this op's weight may be staged ahead of its own stage.

    Always False while `WEIGHT_PREFETCH_ENABLED` is False; carried so the row
    shape is settled before slice 3a reports."""

    var loop_distance: Int
    """How many ops ahead the stage may run. 0 while prefetch is disabled."""

    @always_inline
    def m_tiles(self) -> Int:
        return self.m // HMX_TILE

    @always_inline
    def k_tiles(self) -> Int:
        return hmx_k_tiles(self.k)

    @always_inline
    def arena_bytes[dtype: DType = HMX_DTYPE](self, budget: Int) -> Int:
        """Bytes this op's stripe residency needs out of `budget`.

        The same closed form the stripe itself plans with, so the maximum below
        is the maximum of what will actually be carved rather than of a model of
        it.
        """
        var kt = self.k_tiles()
        var slots = hmx_stripe_act_slots[dtype](kt, budget, self.m_tiles())
        if slots == 0:
            return 0
        return hmx_stripe_arena_bytes[dtype](kt, slots)


@fieldwise_init
struct WeightRow(Copyable, ImplicitlyCopyable, Movable):
    """One weight's extent, keyed on the artefact and not on any schedule."""

    var id: Weight
    """The weight this row describes. Its index is the row's position."""

    var rows: Int
    """`N` for a projection; 1 for a modulation vector or a norm gain."""

    var cols: Int
    """`K` for a projection; the vector's width otherwise."""

    @always_inline
    def elems(self) -> Int:
        return self.rows * self.cols

    @always_inline
    def named(self) -> Bool:
        """Whether the block's sequence reads this weight at all."""
        return self.rows > 0 and self.cols > 0


# ===-----------------------------------------------------------------------===#
# The derivation.
#
# Two sinks, both recording geometry and nothing else. Every method the trait
# declares is here because the trait declares it; the ones that name no weight
# record nothing, which is why they read as empty rather than as forgotten.
# ===-----------------------------------------------------------------------===#


struct _GeometrySink(BlockOps):
    """Runs `double_block` for its shapes.

    Not a `Recorder`: the block's op ORDER is `test_block_stack.mojo`'s subject
    and this is only its geometry, so recording the sequence here would be a
    second, weaker statement of something already gated.
    """

    var matmuls: List[StepMatmul]
    var rows: List[Int]
    var cols: List[Int]

    def __init__(out self):
        self.matmuls = List[StepMatmul]()
        self.rows = List[Int]()
        self.cols = List[Int]()
        for i in range(NUM_WEIGHTS):
            _ = i
            self.rows.append(0)
            self.cols.append(0)

    @always_inline
    def _note(mut self, w: Weight, rows: Int, cols: Int):
        self.rows[w.index] = rows
        self.cols[w.index] = cols

    def layer_norm(mut self, dst: Slot, x: Slot, rows: Int, width: Int):
        pass

    def ada_modulate(
        mut self,
        dst: Slot,
        x: Slot,
        scale: Weight,
        shift: Weight,
        rows: Int,
        width: Int,
    ):
        self._note(scale, 1, width)
        self._note(shift, 1, width)

    def matmul(
        mut self, dst: Slot, x: Slot, w: Weight, rows: Int, k: Int, n: Int
    ):
        self.matmuls.append(
            StepMatmul(
                weight=w,
                m=rows,
                n=n,
                k=k,
                is_prefetch=False,
                loop_distance=0,
            )
        )
        self._note(w, n, k)

    def rms_norm(
        mut self, dst: Slot, x: Slot, gamma: Weight, head_rows: Int, width: Int
    ):
        self._note(gamma, 1, width)

    def join_rows(
        mut self,
        dst: Slot,
        top: Slot,
        top_rows: Int,
        bottom: Slot,
        bottom_rows: Int,
        width: Int,
    ):
        pass

    def join_cols(
        mut self,
        dst: Slot,
        left: Slot,
        left_width: Int,
        right: Slot,
        right_width: Int,
        rows: Int,
    ):
        pass

    def rope(mut self, dst: Slot, x: Slot, head_rows: Int, head_dim: Int):
        pass

    def mha(
        mut self, dst: Slot, q: Slot, k: Slot, v: Slot, seq: Int, heads: Int
    ):
        pass

    def take_rows(
        mut self, dst: Slot, src: Slot, first_row: Int, rows: Int, width: Int
    ):
        pass

    def gate_residual(
        mut self,
        dst: Slot,
        x: Slot,
        y: Slot,
        gate: Weight,
        rows: Int,
        width: Int,
        clamp: Bool,
    ):
        self._note(gate, 1, width)

    def swiglu(mut self, dst: Slot, x: Slot, rows: Int, hidden: Int):
        pass


def double_block_matmuls(cfg: KleinConfig) -> List[StepMatmul]:
    """Every matmul one double block runs, in program order.

    Comptime-evaluable, which is the whole point: `step_arena_bytes` below folds
    to a constant and the arena stops being re-carved per op.
    """
    var sink = _GeometrySink()
    double_block(sink, cfg, 0)
    return sink.matmuls.copy()


def double_block_weights(cfg: KleinConfig) -> List[WeightRow]:
    """The block's weight table, one row per `Weight`, in id order.

    Row `i` describes `Weight(i)`. That is asserted rather than assumed --
    `weight_table_ok` -- because a table whose row order drifts from the id
    space is the failure `NPUKernels.def`'s declared-selector-is-position
    discipline exists to make loud, applied here to a layout instead of a
    manifest.

    The table spans the whole id space, so the rows the double block does not
    read are present and empty. That is what makes "this kind names exactly its
    own range" checkable at all.
    """
    var sink = _GeometrySink()
    double_block(sink, cfg, 0)
    var table = List[WeightRow]()
    for i in range(NUM_WEIGHTS):
        table.append(WeightRow(Weight(i), sink.rows[i], sink.cols[i]))
    return table^


def single_block_matmuls(cfg: KleinConfig) -> List[StepMatmul]:
    """Every matmul one single block runs, in program order.

    Five, against the double block's twelve, and one of them -- the
    out-projection over `dim + mlp_hidden` -- is the deepest reduction in the
    step.
    """
    var sink = _GeometrySink()
    single_block(sink, cfg, 0)
    return sink.matmuls.copy()


def single_block_weights(cfg: KleinConfig) -> List[WeightRow]:
    """The single block's weight table, in the same whole-id-space form."""
    var sink = _GeometrySink()
    single_block(sink, cfg, 0)
    var table = List[WeightRow]()
    for i in range(NUM_WEIGHTS):
        table.append(WeightRow(Weight(i), sink.rows[i], sink.cols[i]))
    return table^


# ===-----------------------------------------------------------------------===#
# What the table is for.
# ===-----------------------------------------------------------------------===#


struct _ComptimeFoldSink[dtype: DType](BlockOps):
    """Runs `double_block` for the two facts that must survive comptime.

    Deliberately heap-free, and that is the whole reason it exists beside
    `_GeometrySink`: these are the derivations the contract needs folded, and
    an accumulator folds in a way a `List` of ops need not. The two sinks are
    held together by the twin's gate 4, which folds the list at run time and
    asserts the same numbers -- so the cheap derivation is checked against the
    descriptive one rather than trusted.

    It folds three things because three have to survive: the arena maximum (so
    slice 6's carve is a constant), one named weight's extent (so a weight stage
    can be hoisted above the code that computes it, which is what "stop
    re-staging every step" means mechanically), and the widest row the step
    routes through the staged norm (so the arena's residue can be checked
    against its other consumer at build time rather than by inspection).
    """

    var budget: Int
    var most: Int
    var want: Weight
    var want_rows: Int
    var want_cols: Int
    var widest_norm: Int

    def __init__(out self, budget: Int, want: Weight):
        self.budget = budget
        self.most = 0
        self.want = want
        self.want_rows = 0
        self.want_cols = 0
        self.widest_norm = 0

    def layer_norm(mut self, dst: Slot, x: Slot, rows: Int, width: Int):
        pass

    def ada_modulate(
        mut self,
        dst: Slot,
        x: Slot,
        scale: Weight,
        shift: Weight,
        rows: Int,
        width: Int,
    ):
        pass

    def matmul(
        mut self, dst: Slot, x: Slot, w: Weight, rows: Int, k: Int, n: Int
    ):
        var op = StepMatmul(w, rows, n, k, False, 0)
        var bytes = op.arena_bytes[Self.dtype](self.budget)
        if bytes > self.most:
            self.most = bytes
        if w == self.want:
            self.want_rows = n
            self.want_cols = k

    def rms_norm(
        mut self, dst: Slot, x: Slot, gamma: Weight, head_rows: Int, width: Int
    ):
        # `rms_norm` is the one family the step runs out of staged VTCM
        # (`mojo_rms_norm_f16_staged`); `layer_norm` computes in place through
        # the cache and carves nothing. Folding the width here rather than
        # naming `head_dim` at the check is what makes the residue guard track
        # the sequence: route a wider row through the staged path and the guard
        # re-evaluates itself instead of going quietly stale.
        if width > self.widest_norm:
            self.widest_norm = width

    def join_rows(
        mut self,
        dst: Slot,
        top: Slot,
        top_rows: Int,
        bottom: Slot,
        bottom_rows: Int,
        width: Int,
    ):
        pass

    def join_cols(
        mut self,
        dst: Slot,
        left: Slot,
        left_width: Int,
        right: Slot,
        right_width: Int,
        rows: Int,
    ):
        pass

    def rope(mut self, dst: Slot, x: Slot, head_rows: Int, head_dim: Int):
        pass

    def mha(
        mut self, dst: Slot, q: Slot, k: Slot, v: Slot, seq: Int, heads: Int
    ):
        pass

    def take_rows(
        mut self, dst: Slot, src: Slot, first_row: Int, rows: Int, width: Int
    ):
        pass

    def gate_residual(
        mut self,
        dst: Slot,
        x: Slot,
        y: Slot,
        gate: Weight,
        rows: Int,
        width: Int,
        clamp: Bool,
    ):
        pass

    def swiglu(mut self, dst: Slot, x: Slot, rows: Int, hidden: Int):
        pass


def step_arena_bytes[
    dtype: DType = HMX_DTYPE
](cfg: KleinConfig, budget: Int) -> Int:
    """The step-scoped arena's size: the max over the block's matmul shapes.

    Every shape lands at 91-98% of an 8 MiB grant (sec 3.1), so this is not an
    optimisation of a sum -- two shapes' residency cannot coexist, and the arena
    is one region re-carved per stage. What the maximum buys is that the carve
    arithmetic stops being repeated and the non-HMX consumer gets an address
    that is past EVERY shape rather than past the current one.

    Both kinds, one fold, and the maximum has to be read that way rather than
    attributed. At `klein_512` and an 8 MiB grant the double block's `ff_out`
    asks 8,308,992 B and the single block's fused projections ask 8,222,976 B,
    so the double block still sets it -- by 86,016 B, which is 1.0% of the
    grant and less than one activation panel of the shape that loses. At
    `klein_mini` the order is already the other way (the single block's
    out-projection asks 1,075,456 B against 608,512 B). "The double block
    dominates" is therefore a MEASUREMENT at one geometry, not a structural
    fact, and nothing here may be simplified as though it were.

    The deepest reduction is not the largest arena, which is the other thing
    worth reading off this fold: `sgl out-proj` spans 384 croutons and lands
    below the 96-crouton fused projections, because a k that deep leaves the
    budget room for 8 activation panels against 39.

    Args:
        cfg: The step geometry.
        budget: Bytes of VTCM the grant leaves after alignment.

    Returns:
        Bytes, or 0 if no shape fits -- which a caller must treat as a refusal.
    """
    var sink = _ComptimeFoldSink[dtype](budget, NO_WEIGHT)
    double_block(sink, cfg, 0)
    single_block(sink, cfg, 0)
    return sink.most


def step_arena_residue[
    dtype: DType = HMX_DTYPE
](cfg: KleinConfig, budget: Int) -> Int:
    """Bytes the step's arena leaves behind for the non-HMX consumer.

    Exactly what `StepArena.scratch_base(step_arena_bytes(...))` hands out, as a
    size rather than an address, so the one consumer that carves there can be
    priced without opening an arena. It is a floor-division remainder and NOT
    monotone in the geometry: at an 8 MiB grant it is 79,616 B at Klein's 24
    heads, 50,944 B at 16, and 141,056 B at 32, because `hmx_stripe_act_slots`
    truncates. That is the reason the check that consumes this is a build
    refusal and not a comment.

    Returns:
        Bytes past the arena's maximum, or 0 when nothing is left -- which a
        caller must treat as a refusal and not as a size.
    """
    var arena = step_arena_bytes[dtype](cfg, budget)
    if arena == 0 or arena >= budget:
        return 0
    return budget - arena


def step_staged_norm_width(cfg: KleinConfig) -> Int:
    """The widest row the step routes through the staged norm.

    `head_dim` today: QK-norm is the step's only staged-VTCM consumer, and the
    other carves from the same base (`mojo_npu_udma_roundtrip`,
    `mojo_moe_create_indices`, `mojo_npu_membw_probe`) are probes that no
    denoise step dispatches. Derived rather than written down so that widening
    the staged path re-prices its own guard.

    Over BOTH kinds, for the same reason `step_arena_bytes` is: the residue this
    number is checked against is the step's, so a row staged by either kind
    belongs in it. The two agree today at `head_dim`, which is why folding one
    of them would have looked right for as long as that held.
    """
    var sink = _ComptimeFoldSink[HMX_DTYPE](0, NO_WEIGHT)
    double_block(sink, cfg, 0)
    single_block(sink, cfg, 0)
    return sink.widest_norm


def matmul_weight_extent(cfg: KleinConfig, w: Weight) -> WeightRow:
    """One matmul weight's artefact extent, without touching the heap.

    The accessor a weight stager needs, and the reason it is separate from
    `double_block_weights`: a stage hoisted out of the per-dispatch path has to
    know its extent before the dispatch exists, so the extent has to fold. It
    spans both kinds because a stager holds a `Weight`, not a block kind. An
    empty row means no matmul in the step reads `w`.
    """
    var sink = _ComptimeFoldSink[HMX_DTYPE](0, w)
    double_block(sink, cfg, 0)
    single_block(sink, cfg, 0)
    return WeightRow(w, sink.want_rows, sink.want_cols)


def step_weight_elems(cfg: KleinConfig) -> Int:
    """Elements one double block's weights occupy.

    The residency figure the steer is about: this many elements are re-staged
    from DDR on every denoise step today, once per block per step.
    """
    var table = double_block_weights(cfg)
    var total = 0
    for i in range(len(table)):
        total += table[i].elems()
    return total


@always_inline
def prefetch_distance(op: StepMatmul) -> Int:
    """How far ahead `op`'s weight stage may run. 0 while prefetch is disabled.

    The one reader of `loop_distance`, so the gate is spelled once: a stager
    that consulted the field directly would honour a distance the shim cannot
    support the day someone set one.
    """
    comptime if not WEIGHT_PREFETCH_ENABLED:
        return 0
    return op.loop_distance


# ===-----------------------------------------------------------------------===#
# The build-time checks.
# ===-----------------------------------------------------------------------===#


def _weight_table_ok(
    table: List[WeightRow], ops: List[StepMatmul], first: Int, past: Int
) -> Bool:
    """Whether one block kind's rows of the table are well-formed.

    Four properties, each of which a hand-written table loses quietly:

    1. **Position is identity.** Row `i` describes `Weight(i)`.
    2. **The kind names exactly its own range.** Every id in `[first, past)` is
       named and no id outside it is. Both halves matter: an unnamed id inside
       means the id space and the sequence have drifted, and a named id outside
       means one kind reached into the other's artefacts, which would make a row
       describe two tensors. Before the single block existed this was "every
       weight is named", which cannot be asked of a two-kind id space at all.
    3. **Every matmul's extents are whole croutons.** `hmx_k_tiles` divides and
       `hmx_stripe_plan` truncates `M` and `N` the same way, so an extent that
       is not a multiple of `HMX_TILE` silently drops its tail rather than
       failing. On `K` this is also the alignment constraint the
       schedule-independent layout is bought with -- the stripe's k-tile starts
       have to land on the artefact's own crouton boundaries -- and it is stated
       here rather than assumed, which is the one thing the prior art
       (`WEIGHT_STUDY_B_SKYE.md` sec 2b) says its own equivalent is not.
    4. **A weight's extent agrees with every op that reads it.** Two ops sharing
       a weight at different `M` must agree on `(N, K)`, which is what makes the
       row a property of the artefact rather than of a schedule.
    """
    if len(table) != NUM_WEIGHTS:
        return False
    for i in range(len(table)):
        if table[i].id.index != i:
            return False
        if table[i].named() != (i >= first and i < past):
            return False

    for i in range(len(ops)):
        var op = ops[i]
        if op.k % HMX_TILE != 0:
            return False
        if op.m % HMX_TILE != 0:
            return False
        if op.n % HMX_TILE != 0:
            return False
        var row = table[op.weight.index]
        if row.rows != op.n or row.cols != op.k:
            return False
    return True


def weight_table_ok(cfg: KleinConfig) -> Bool:
    """Whether the double block's rows are well-formed for `cfg`."""
    return _weight_table_ok(
        double_block_weights(cfg),
        double_block_matmuls(cfg),
        0,
        NUM_DOUBLE_WEIGHTS,
    )


def single_weight_table_ok(cfg: KleinConfig) -> Bool:
    """Whether the single block's rows are well-formed for `cfg`."""
    return _weight_table_ok(
        single_block_weights(cfg),
        single_block_matmuls(cfg),
        SINGLE_WEIGHT_BASE,
        NUM_WEIGHTS,
    )
