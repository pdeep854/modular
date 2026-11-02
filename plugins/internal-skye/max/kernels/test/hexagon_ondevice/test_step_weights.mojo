"""The step's weight table, against the block sequence it is derived from.

`step_weights.mojo` claims three things a reader cannot check: that the table is
comptime-evaluable, that it is keyed on the artefact rather than on a schedule,
and that the arena maximum it hands slice 6 is the maximum of what will actually
be carved. Seven gates, none of which restates the module:

1. **Comptime.** The constants below are computed at COMPILE time from the block
   sequences. If the derivation were not comptime-evaluable this file would not
   build, which is the point: the whole contract rests on the geometry being
   known before the step runs, and a runtime table would satisfy every other
   gate here.

2. **The table is well-formed.** Position is identity, each kind names exactly
   its own range of the id space, every `K` is a whole number of croutons, and
   two ops sharing a weight agree on its extent.

3. **The census.** Twelve matmuls for the double block and five for the single,
   in order, with the weight each reads and the extents the geometry implies --
   derived from `KleinConfig` independently of the sequence, so a shape confused
   for another (`mlp_hidden` where `ff_in_width` belongs) fails here rather than
   reaching an arena.

4. **The maximum is the maximum.** The comptime constant equals a runtime fold
   over the same ops, over BOTH kinds. A fold that dropped one kind still
   agrees with itself, which is why the runtime side enumerates the two
   sequences separately rather than calling the same folding sink.

5. **The join with slice 6.** `StepArena.scratch_base` fed this maximum must
   place the non-HMX region past EVERY op's stripe, not past the current one.
   That is the whole reason slice 8 precedes the composition.

6. **Prefetch is disabled.** A row carrying a nonzero `loop_distance` still
   yields distance 0 while `WEIGHT_PREFETCH_ENABLED` is False. The fields are
   expressed; nothing acts on them until slice 3a reports.

7. **The rows are M-independent.** `WQ` and `WQ_ADD` are read at different row
   counts and describe the same artefact extent, which is the property that
   makes a row a weight's and not an op's.

Mutation controls, each applied by hand and reverted (recorded in the PR):
widen one matmul's `n` in `block_stack.mojo` (gate 3), build the table in
reverse id order (gate 2), set `WEIGHT_PREFETCH_ENABLED` to True (gate 6), take
the min instead of the max in `step_arena_bytes` (gates 4 and 5), and drop the
`_note` from `matmul` (gate 2). For the single block: pass `dim` where
`single_out_k()` belongs (gate 3), and fold only `double_block` in
`step_arena_bytes` (gate 4 at `klein_mini`, where the single block sets the
maximum -- and NOT at `klein_512`, which is why both geometries are run).
"""

from std.memory import AddressSpace

from hexagon_ondevice.block_stack import (
    KleinConfig,
    NUM_DOUBLE_WEIGHTS,
    NUM_WEIGHTS,
    SGL_WK,
    SGL_WO,
    SGL_WQ,
    SGL_WV,
    SGL_W_MLP,
    SINGLE_WEIGHT_BASE,
    WK,
    WK_ADD,
    WO,
    WO_ADD,
    WQ,
    WQ_ADD,
    WV,
    WV_ADD,
    W_FF_IN,
    W_FF_IN_CTX,
    W_FF_OUT,
    W_FF_OUT_CTX,
    Weight,
    klein_512,
    klein_mini,
)
from hexagon_ondevice.hmx_crouton import HMX_TILE, hmx_stripe_arena_bytes
from hexagon_ondevice.matmul_kernel import HmxStripePlan
from hexagon_ondevice.step_arena import R_REC, StepArena
from hexagon_ondevice.step_weights import (
    StepMatmul,
    WEIGHT_PREFETCH_ENABLED,
    double_block_matmuls,
    double_block_weights,
    matmul_weight_extent,
    prefetch_distance,
    single_block_matmuls,
    single_block_weights,
    single_weight_table_ok,
    step_arena_bytes,
    step_weight_elems,
    weight_table_ok,
)

comptime TWIN_SPACE = AddressSpace.GENERIC

comptime GRANT = 8 << 20
"""The VTCM budget the audit's plan table was computed against."""

comptime ARENA_512 = step_arena_bytes(klein_512(), GRANT)
"""GATE 1. Computed at COMPILE time from both block sequences.

If the derivation could not be evaluated at comptime this line would not build,
and the arena would be back to a per-op runtime carve -- which is exactly the
residency the composition exists to buy."""

comptime ARENA_MINI = step_arena_bytes(klein_mini(), GRANT)
"""GATE 1, at the reduced geometry the driver iterates on."""

comptime WQ_EXTENT = matmul_weight_extent(klein_512(), WQ)
"""GATE 1. A weight's artefact extent, folded at COMPILE time.

The residency precondition in its narrowest form: a stage hoisted out of the
per-dispatch path has to know its extent before the dispatch exists."""


def expected_matmuls(cfg: KleinConfig) -> List[StepMatmul]:
    """The twelve matmuls, derived from the geometry rather than the sequence.

    Written out so a shape confused for another fails: `linear_in` widens to
    `2 * mlp_hidden`, `linear_out` contracts from `mlp_hidden`, and the six
    stream projections are square in `dim`. This is `test_block_stack.mojo`'s
    gate 3 applied to the table instead of to the recording.
    """
    var dim = cfg.dim()
    var ir = cfg.image_rows()
    var tr = cfg.text_rows()
    var ffin = cfg.ff_in_width()
    var hidden = cfg.mlp_hidden()
    var want = List[StepMatmul]()
    want.append(StepMatmul(WQ, ir, dim, dim, False, 0))
    want.append(StepMatmul(WK, ir, dim, dim, False, 0))
    want.append(StepMatmul(WV, ir, dim, dim, False, 0))
    want.append(StepMatmul(WQ_ADD, tr, dim, dim, False, 0))
    want.append(StepMatmul(WK_ADD, tr, dim, dim, False, 0))
    want.append(StepMatmul(WV_ADD, tr, dim, dim, False, 0))
    want.append(StepMatmul(WO, ir, dim, dim, False, 0))
    want.append(StepMatmul(WO_ADD, tr, dim, dim, False, 0))
    want.append(StepMatmul(W_FF_IN, ir, ffin, dim, False, 0))
    want.append(StepMatmul(W_FF_IN_CTX, tr, ffin, dim, False, 0))
    want.append(StepMatmul(W_FF_OUT, ir, dim, hidden, False, 0))
    want.append(StepMatmul(W_FF_OUT_CTX, tr, dim, hidden, False, 0))
    return want^


def expected_single_matmuls(cfg: KleinConfig) -> List[StepMatmul]:
    """The five, likewise derived from the geometry rather than the sequence.

    Every row count is the JOINT one -- the streams arrive concatenated -- and
    the out-projection reduces over `dim + mlp_hidden` rather than `dim`, which
    is the extent the double block has nowhere and the one an eye reading the
    two bodies side by side is most likely to normalise away.
    """
    var dim = cfg.dim()
    var jr = cfg.joint_rows()
    var want = List[StepMatmul]()
    want.append(StepMatmul(SGL_WQ, jr, dim, dim, False, 0))
    want.append(StepMatmul(SGL_WK, jr, dim, dim, False, 0))
    want.append(StepMatmul(SGL_WV, jr, dim, dim, False, 0))
    want.append(StepMatmul(SGL_W_MLP, jr, cfg.ff_in_width(), dim, False, 0))
    want.append(StepMatmul(SGL_WO, jr, dim, dim + cfg.mlp_hidden(), False, 0))
    return want^


def gate_well_formed(name: StaticString, cfg: KleinConfig) -> Bool:
    """Gate 2, both kinds: each names exactly its own range of the id space."""
    var ok = True
    if not weight_table_ok(cfg):
        print("  table:", name, "double rows are not well-formed")
        ok = False
    if not single_weight_table_ok(cfg):
        print("  table:", name, "single rows are not well-formed")
        ok = False
    return ok


def _check_census(
    name: StaticString,
    kind: StaticString,
    got: List[StepMatmul],
    want: List[StepMatmul],
) -> Bool:
    var ok = True
    if len(got) != len(want):
        print("  census:", name, kind, len(got), "matmuls, want", len(want))
        return False
    for i in range(len(want)):
        if got[i].weight != want[i].weight:
            print("  census:", name, kind, "op", i, "reads the wrong weight")
            ok = False
        if got[i].m != want[i].m or got[i].n != want[i].n:
            print(
                "  census:",
                name,
                kind,
                "op",
                i,
                "is",
                got[i].m,
                "x",
                got[i].n,
                "want",
                want[i].m,
                "x",
                want[i].n,
            )
            ok = False
        if got[i].k != want[i].k:
            print(
                "  census:",
                name,
                kind,
                "op",
                i,
                "k is",
                got[i].k,
                "want",
                want[i].k,
            )
            ok = False
    return ok


def gate_census(name: StaticString, cfg: KleinConfig) -> Bool:
    """Gate 3, both kinds."""
    var ok = _check_census(
        name, "dbl", double_block_matmuls(cfg), expected_matmuls(cfg)
    )
    if not _check_census(
        name, "sgl", single_block_matmuls(cfg), expected_single_matmuls(cfg)
    ):
        ok = False
    return ok


def gate_maximum(name: StaticString, cfg: KleinConfig, folded: Int) -> Bool:
    """Gate 4: the folded constant is the fold over BOTH kinds' ops.

    The per-kind maxima are printed, not just the step's, because which kind
    sets it is geometry-dependent and reading it off is the only way to know:
    the double block's `ff_out` wins at `klein_512` by 1.0% of the grant, and
    the single block's out-projection wins outright at `klein_mini`.
    """
    var most = 0
    var per_kind = List[Int]()
    for kind in range(2):
        var ops = double_block_matmuls(
            cfg
        ) if kind == 0 else single_block_matmuls(cfg)
        var kind_most = 0
        for i in range(len(ops)):
            var bytes = ops[i].arena_bytes(GRANT)
            if bytes > kind_most:
                kind_most = bytes
        per_kind.append(kind_most)
        if kind_most > most:
            most = kind_most
    print(
        "  maximum:",
        name,
        "dbl",
        per_kind[0],
        "sgl",
        per_kind[1],
        "step",
        most,
    )

    var ok = True
    if folded != most:
        print("  maximum:", name, "folded", folded, "runtime", most)
        ok = False
    if most == 0:
        print("  maximum:", name, "no shape fits the grant")
        ok = False
    if most > GRANT:
        print("  maximum:", name, "exceeds the grant it was sized against")
        ok = False
    # A fold that dropped a kind would still equal itself, so the two per-kind
    # numbers are asserted to be numbers rather than assumed: a sequence that
    # stopped producing matmuls reads as 0 here instead of as agreement.
    for kind in range(2):
        if per_kind[kind] == 0:
            print("  maximum:", name, "kind", kind, "contributed no shape")
            ok = False
    return ok


def gate_join(name: StaticString, cfg: KleinConfig, folded: Int) -> Bool:
    """Gate 5: the norm's region begins past every op's stripe."""
    var arena = StepArena[TWIN_SPACE].at(0x2000_0000, GRANT)
    var scratch = arena.scratch_base(folded)
    if scratch == 0:
        print("  join:", name, "the step maximum", folded, "leaves no scratch")
        return False

    # Reported rather than inferred: what the step-scoped arena leaves the
    # non-HMX consumer is a number the composition has to design against, and
    # the audit's 91-98% band says in advance that it will be small.
    print(
        "  join:",
        name,
        "arena",
        folded,
        "of",
        arena.budget,
        "leaves",
        arena.budget - folded,
        "bytes of scratch",
    )

    var ok = True
    var ops = double_block_matmuls(cfg)
    ops.extend(single_block_matmuls(cfg))
    for i in range(len(ops)):
        var op = ops[i]
        var bytes = op.arena_bytes(GRANT)
        if bytes == 0:
            print("  join:", name, "op", i, "does not fit the grant")
            ok = False
            continue
        # The op's own carve, described through the arena's region table: its
        # last region must end at or before the scratch base.
        var plan = HmxStripePlan(
            op.m_tiles(),
            op.n // HMX_TILE,
            op.k_tiles(),
            _act_slots_of(op),
            1,
        )
        var last = arena.region(R_REC, plan)
        if arena.base + last.end() > scratch:
            print("  join:", name, "op", i, "runs into the scratch region")
            ok = False
    return ok


def _act_slots_of(op: StepMatmul) -> Int:
    """The slot count `arena_bytes` sized this op with.

    Recovered from the byte total rather than recomputed, so the join gate
    cannot pass by agreeing with a second copy of the sizing arithmetic.
    """
    var want = op.arena_bytes(GRANT)
    for slots in range(1, op.m_tiles() + 1):
        if hmx_stripe_arena_bytes(op.k_tiles(), slots) == want:
            return slots
    return 0


def gate_prefetch_disabled(cfg: KleinConfig) -> Bool:
    """Gate 6."""
    # A comptime assert rather than a runtime check: with the constant False
    # the compiler folds any `if` on it away, so build time is the only place
    # enforcement survives. Enabling prefetch therefore has to retire this line
    # consciously, which is exactly the ceremony the unpriced transfer limit
    # deserves.
    comptime assert not WEIGHT_PREFETCH_ENABLED, (
        "cross-op weight prefetch is enabled while the shim's"
        " one-outstanding-transfer limit is unpriced; the feed design's slice"
        " 3a probe is its gate"
    )

    var ok = True

    var ops = double_block_matmuls(cfg)
    for i in range(len(ops)):
        if ops[i].is_prefetch or ops[i].loop_distance != 0:
            print("  prefetch: op", i, "carries a distance the table set")
            ok = False
        if prefetch_distance(ops[i]) != 0:
            print("  prefetch: op", i, "yielded a nonzero distance")
            ok = False

    # A row that DOES carry a distance must still yield zero: the gate belongs
    # to the reader, not to whoever filled the field in.
    var armed = StepMatmul(WQ, 1024, 3072, 3072, True, 3)
    if prefetch_distance(armed) != 0:
        print("  prefetch: an armed row yielded", prefetch_distance(armed))
        ok = False
    return ok


def gate_rows_are_m_independent(cfg: KleinConfig) -> Bool:
    """Gate 7: a row describes the artefact, not the op that reads it."""
    var table = double_block_weights(cfg)
    var ops = double_block_matmuls(cfg)
    var ok = True

    # `WQ` and `WQ_ADD` are the same projection on the two streams, read at
    # different row counts. Their rows must be identical.
    var q = table[WQ.index]
    var q_add = table[WQ_ADD.index]
    if q.rows != q_add.rows or q.cols != q_add.cols:
        print(
            "  rows: the image and text Q projections describe different"
            " artefacts"
        )
        ok = False

    var m_img = 0
    var m_txt = 0
    for i in range(len(ops)):
        if ops[i].weight == WQ:
            m_img = ops[i].m
        if ops[i].weight == WQ_ADD:
            m_txt = ops[i].m
    if m_img == m_txt:
        print(
            "  rows: the two Q projections were read at the same M, so this"
            " gate proves nothing"
        )
        ok = False

    # And every row the kind owns is nonempty, which is what makes
    # `step_weight_elems` a residency figure rather than a partial sum. Each
    # kind is asked only about its own range: the double block's table leaves
    # the single block's ids empty BY CONSTRUCTION, and asking it about all 38
    # would fail for the right reason and the wrong one indistinguishably.
    for i in range(NUM_DOUBLE_WEIGHTS):
        if not table[i].named():
            print("  rows: weight", i, "is unnamed by the double sequence")
            ok = False
    var single = single_block_weights(cfg)
    for i in range(SINGLE_WEIGHT_BASE, NUM_WEIGHTS):
        if not single[i].named():
            print("  rows: weight", i, "is unnamed by the single sequence")
            ok = False

    # The single block's fused projections share an M and describe different
    # artefacts, which is the same property from the other side: `SGL_WQ` and
    # `SGL_W_MLP` are read at one row count and their extents still differ.
    if single[SGL_WQ.index].rows == single[SGL_W_MLP.index].rows:
        print("  rows: the fused Q and MLP projections have the same N")
        ok = False
    return ok


def main() raises:
    var failures = 0

    print("gate 1: the arena maximum folds at comptime")
    print("  klein_512 arena", ARENA_512, "bytes")
    print("  klein_mini arena", ARENA_MINI, "bytes")
    print("  klein_512 block weights", step_weight_elems(klein_512()), "elems")
    print("  klein_512 WQ artefact", WQ_EXTENT.rows, "x", WQ_EXTENT.cols)
    if ARENA_512 == 0 or ARENA_MINI == 0:
        print("  comptime: a folded constant is zero")
        failures += 1
    # The folded extent must be the table's, or the cheap accessor and the
    # descriptive one have drifted.
    var wq_row = double_block_weights(klein_512())[WQ.index]
    if WQ_EXTENT.rows != wq_row.rows or WQ_EXTENT.cols != wq_row.cols:
        print("  comptime: the folded WQ extent is not the table's")
        failures += 1

    print("gate 2: the table is well-formed")
    if not gate_well_formed("klein_512", klein_512()):
        failures += 1
    if not gate_well_formed("klein_mini", klein_mini()):
        failures += 1

    print("gate 3: the census")
    if not gate_census("klein_512", klein_512()):
        failures += 1
    if not gate_census("klein_mini", klein_mini()):
        failures += 1

    print("gate 4: the maximum is the maximum")
    if not gate_maximum("klein_512", klein_512(), ARENA_512):
        failures += 1
    if not gate_maximum("klein_mini", klein_mini(), ARENA_MINI):
        failures += 1

    print("gate 5: the join with the arena")
    if not gate_join("klein_512", klein_512(), ARENA_512):
        failures += 1
    if not gate_join("klein_mini", klein_mini(), ARENA_MINI):
        failures += 1

    print("gate 6: prefetch is expressed and disabled")
    if not gate_prefetch_disabled(klein_512()):
        failures += 1

    print("gate 7: the rows are M-independent")
    if not gate_rows_are_m_independent(klein_512()):
        failures += 1

    print()
    if failures == 0:
        print("ALL STEP WEIGHT TABLE TESTS PASSED")
    else:
        raise Error(String(failures) + " step weight gate(s) FAILED")
