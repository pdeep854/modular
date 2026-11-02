"""The blocks' Mojo sequences, against what the driver actually launches.

`double_block` and `single_block` (`block_stack.mojo`) are transliterations of
`runDoubleBlock` and `runSingleBlock`
(`npu_block_stack_dispatch.cpp:2823-2961` and `:2972-3046`), and a
transliteration is exactly the kind of change that compiles, reads correctly and
is wrong. Four independent gates, each run against both kinds, none of which is
the body restated:

1. **The census.** The per-family launch counts each block makes were derived
   from `buildPlan` and reconciled against a MEASURED render -- 450 launches on
   the block-stack driver, and three of the graph path's buckets exactly
   (`MEGAKERNEL_IMPLEMENTATION_PLAN.md` sec 1.2-1.4). This asserts the sequences
   reproduce that table family by family: 33 kernel launches plus 5 layout ops
   for the double block, 14 plus 1 for the single. An op dropped, doubled or
   spelled as the wrong family fails here, and the reference is a measurement.

2. **Dataflow.** Every buffer an op reads was written earlier in the block or is
   a block input, and the block's outputs are the two streams. This is what a
   reordering breaks, and it is the property a derived schedule would later have
   to preserve (`MEGAKERNEL_VM_DISPATCH_NOTE.md` sec 7 gate 3).

3. **Shapes.** The extents are checked against the geometry independently: the
   six stream projections are square in `dim`, `linear_in` widens to
   `2 * mlp_hidden`, SwiGLU halves it, `linear_out` contracts back, the QK-norm
   and RoPE run on head rows rather than token rows, and attention sees the
   joint sequence. A config whose `dim` and `mlp_hidden` were confused passes
   gates 1 and 2 and fails here.

4. **The walker.** The sequence is encoded as a RUN-mode `'NMGS'` block and
   walked by `megastep_walk` -- the same decoder the megastep dispatch path uses
   -- with a recording seam. What comes out must be the sequence that went in,
   selector for selector and operand for operand. This is the `#330` differential
   in the other direction: there, fixtures were walked against a golden; here,
   the block body produces the wire and the walker is what reads it back, so an
   arity the manifest would refuse is refused here instead of on a device.

The single block adds one property the double block does not have: the
ping-pong. `runSingleBlock` ends by swapping its two stream allocations, so an
even block reads `JOINT` and an odd one reads `JOINT_ALT`, and both parities are
run through gates 1-4. It also SHARES six joint buffers with the double block,
because the driver's `DevScratch` does; that sharing is asserted here, since
private copies would read as a tidy-up while telling a later hazard proof the
two kinds are independent.

Mutation controls, each of which must turn a gate red (run by hand against this
file, recorded in the PR): drop one `matmul` (gate 1), swap the `rope` and `mha`
calls (gate 2), pass `mlp_hidden` where `ff_in_width` belongs (gate 3), and skew
a recorded arity (gate 4) -- the last of which is live, run on every config for
both kinds. For the single block additionally: spell `join_cols` as `join_rows`
(gate 3, on the widths-sum-to-K relation) and drop the parity from
`single_stream_in` (gate 2, on the odd block).
"""

from std.memory import UnsafePointer
from std.testing import assert_equal, assert_true

from hexagon_ondevice.block_stack import (
    ATTN_IMG,
    ATTN_JOINT,
    ATTN_TXT,
    BlockOps,
    COMBINED,
    FFO_IMG,
    FFO_TXT,
    FF_IMG,
    FF_TXT,
    IMG,
    IMG_MID,
    JOINT,
    JOINT_ALT,
    KleinConfig,
    K_IMG,
    K_JOINT,
    K_ROPE,
    K_TXT,
    KN_IMG,
    KN_JOINT,
    KN_TXT,
    MLP_IMG,
    MLP_IN_JOINT,
    MLP_JOINT,
    MLP_TXT,
    MOD_IMG,
    MOD_JOINT,
    MOD_TXT,
    NORM_IMG,
    NORM_JOINT,
    NORM_TXT,
    NUM_SLOTS,
    NUM_WEIGHTS,
    OUT_IMG,
    OUT_JOINT,
    OUT_TXT,
    Q_IMG,
    Q_JOINT,
    Q_ROPE,
    Q_TXT,
    QN_IMG,
    QN_JOINT,
    QN_TXT,
    Slot,
    TXT,
    TXT_MID,
    V_IMG,
    V_JOINT,
    V_TXT,
    Weight,
    double_block,
    klein_512,
    klein_mini,
    single_block,
    single_stream_in,
    single_stream_out,
)
from hexagon_ondevice.megastep_walker import (
    MEGASTEP_BUFFER_SLOTS,
    MEGASTEP_HEADER_BYTES,
    MEGASTEP_MAGIC,
    MEGASTEP_MODE_RUN,
    MEGASTEP_RECORD_BYTES,
    MEGASTEP_REF_TABLE,
    MEGASTEP_TABLE_ENTRIES,
    MegastepSeam,
    SlotBases,
    SlotLengths,
    TableBases,
    TableBytes,
    WireBytes,
    megastep_walk,
)

comptime OP_LAYER_NORM = 0
comptime OP_ADA_MODULATE = 1
comptime OP_MATMUL = 2
comptime OP_RMS_NORM = 3
comptime OP_ROPE = 4
comptime OP_MHA = 5
comptime OP_GATE_RESIDUAL = 6
comptime OP_GATE_RESIDUAL_CLAMP = 7
comptime OP_SWIGLU = 8
comptime OP_JOIN_ROWS = 9
comptime OP_TAKE_ROWS = 10
comptime OP_JOIN_COLS = 11
comptime NUM_OPS = 12


@always_inline
def _op_name(kind: Int) -> StaticString:
    """The selector string the census counts this family by.

    `join_rows`, `take_rows` and `join_cols` have no kernel and no selector: they
    are the driver's host copies, named here so a failure reads as the op it is.
    """
    if kind == OP_LAYER_NORM:
        return "layernorm"
    if kind == OP_ADA_MODULATE:
        return "ada_modulate"
    if kind == OP_MATMUL:
        return "matmul"
    if kind == OP_RMS_NORM:
        return "rms_norm"
    if kind == OP_ROPE:
        return "rope_interleaved"
    if kind == OP_MHA:
        return "mha_no_cache"
    if kind == OP_GATE_RESIDUAL:
        return "gate_residual"
    if kind == OP_GATE_RESIDUAL_CLAMP:
        return "gate_residual_clamp"
    if kind == OP_SWIGLU:
        return "swiglu"
    if kind == OP_JOIN_ROWS:
        return "join_rows"
    if kind == OP_TAKE_ROWS:
        return "take_rows"
    return "join_cols"


comptime MAX_BUFFERS = 4
comptime MAX_SCALARS = 3

comptime WEIGHT_TABLE_BASE = NUM_SLOTS
"""Weights follow the activation slots in the session table, so one index space
covers both and an operand reference names exactly one buffer."""

comptime TWIN_BASE = 0x100000
comptime TWIN_STRIDE = 0x1000
comptime TWIN_BYTES = 0x1000
"""Synthetic bases: the walk never dereferences an operand, it only bounds an
offset against a length, so a table of made-up addresses is what it can be
checked against."""


@fieldwise_init
struct RecordedOp(Copyable, Movable):
    """One launch as the sink saw it: family, operands, extents.

    Operands are held as session-table indices rather than as `Slot`/`Weight`
    because this is the form the wire carries; the typed form is what the
    sequence is WRITTEN in, and the conversion happens once, here, at the
    recorder.
    """

    var kind: Int
    var buffers: List[Int]
    var scalars: List[Int]


@always_inline
def _slot_ref(s: Slot) -> Int:
    return s.index


@always_inline
def _weight_ref(w: Weight) -> Int:
    return WEIGHT_TABLE_BASE + w.index


struct Recorder(BlockOps):
    """Records the sequence instead of running it."""

    var ops: List[RecordedOp]

    def __init__(out self):
        self.ops = List[RecordedOp]()

    def _push(
        mut self, kind: Int, var buffers: List[Int], var scalars: List[Int]
    ):
        self.ops.append(RecordedOp(kind, buffers^, scalars^))

    def layer_norm(mut self, dst: Slot, x: Slot, rows: Int, width: Int):
        self._push(OP_LAYER_NORM, [_slot_ref(dst), _slot_ref(x)], [rows, width])

    def ada_modulate(
        mut self,
        dst: Slot,
        x: Slot,
        scale: Weight,
        shift: Weight,
        rows: Int,
        width: Int,
    ):
        self._push(
            OP_ADA_MODULATE,
            [
                _slot_ref(dst),
                _slot_ref(x),
                _weight_ref(scale),
                _weight_ref(shift),
            ],
            [rows, width],
        )

    def matmul(
        mut self, dst: Slot, x: Slot, w: Weight, rows: Int, k: Int, n: Int
    ):
        self._push(
            OP_MATMUL,
            [_slot_ref(dst), _slot_ref(x), _weight_ref(w)],
            [rows, k, n],
        )

    def rms_norm(
        mut self, dst: Slot, x: Slot, gamma: Weight, head_rows: Int, width: Int
    ):
        self._push(
            OP_RMS_NORM,
            [_slot_ref(dst), _slot_ref(x), _weight_ref(gamma)],
            [head_rows, width],
        )

    def join_rows(
        mut self,
        dst: Slot,
        top: Slot,
        top_rows: Int,
        bottom: Slot,
        bottom_rows: Int,
        width: Int,
    ):
        self._push(
            OP_JOIN_ROWS,
            [_slot_ref(dst), _slot_ref(top), _slot_ref(bottom)],
            [top_rows, bottom_rows, width],
        )

    def join_cols(
        mut self,
        dst: Slot,
        left: Slot,
        left_width: Int,
        right: Slot,
        right_width: Int,
        rows: Int,
    ):
        self._push(
            OP_JOIN_COLS,
            [_slot_ref(dst), _slot_ref(left), _slot_ref(right)],
            [left_width, right_width, rows],
        )

    def rope(mut self, dst: Slot, x: Slot, head_rows: Int, head_dim: Int):
        self._push(
            OP_ROPE, [_slot_ref(dst), _slot_ref(x)], [head_rows, head_dim]
        )

    def mha(
        mut self, dst: Slot, q: Slot, k: Slot, v: Slot, seq: Int, heads: Int
    ):
        self._push(
            OP_MHA,
            [_slot_ref(dst), _slot_ref(q), _slot_ref(k), _slot_ref(v)],
            [seq, heads],
        )

    def take_rows(
        mut self, dst: Slot, src: Slot, first_row: Int, rows: Int, width: Int
    ):
        self._push(
            OP_TAKE_ROWS,
            [_slot_ref(dst), _slot_ref(src)],
            [first_row, rows, width],
        )

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
        self._push(
            OP_GATE_RESIDUAL_CLAMP if clamp else OP_GATE_RESIDUAL,
            [_slot_ref(dst), _slot_ref(x), _slot_ref(y), _weight_ref(gate)],
            [rows, width],
        )

    def swiglu(mut self, dst: Slot, x: Slot, rows: Int, hidden: Int):
        self._push(OP_SWIGLU, [_slot_ref(dst), _slot_ref(x)], [rows, hidden])


def _record(cfg: KleinConfig, block: Int) raises -> List[RecordedOp]:
    var rec = Recorder()
    double_block(rec, cfg, block)
    return rec.ops.copy()


def _record_single(cfg: KleinConfig, block: Int) raises -> List[RecordedOp]:
    var rec = Recorder()
    single_block(rec, cfg, block)
    return rec.ops.copy()


# ===-----------------------------------------------------------------------===#
# Gate 1: the measured per-block census.
# ===-----------------------------------------------------------------------===#


def _check_census(ops: List[RecordedOp]) raises:
    """Per-family counts against the plan's measured per-block table.

    `MEGAKERNEL_IMPLEMENTATION_PLAN.md` sec 1.2, the `per-block dbl` column,
    which reproduces the block-stack driver's measured 450 launches exactly at
    `nd = 5, ns = 20` and three of the graph path's buckets on the nose. The two
    layout rows are the driver's host copies (3 `joinRows` + 2 `takeRows`), which
    the same source enumerates.
    """
    var want = List[Int]()
    for _ in range(NUM_OPS):
        want.append(0)
    want[OP_MATMUL] = 12
    want[OP_LAYER_NORM] = 4
    want[OP_ADA_MODULATE] = 4
    want[OP_RMS_NORM] = 4
    want[OP_ROPE] = 2
    want[OP_MHA] = 1
    want[OP_GATE_RESIDUAL] = 3
    want[OP_GATE_RESIDUAL_CLAMP] = 1
    want[OP_SWIGLU] = 2
    want[OP_JOIN_ROWS] = 3
    want[OP_TAKE_ROWS] = 2

    _assert_census(ops, want, 33, 38, "double")


def _assert_census(
    ops: List[RecordedOp],
    want: List[Int],
    kernel_launches: Int,
    total: Int,
    kind_name: StaticString,
) raises:
    """The per-family counts, and the two totals they have to add up to.

    Split out so the single block's census is the same assertion and not a
    similar one: the layout families are excluded from the kernel count in ONE
    place, which is the number the transport's rendezvous census is read
    against.
    """
    var got = List[Int]()
    for _ in range(NUM_OPS):
        got.append(0)
    for i in range(len(ops)):
        got[ops[i].kind] += 1

    var kernels = 0
    for kind in range(NUM_OPS):
        assert_equal(
            got[kind],
            want[kind],
            String("census: ", kind_name, " ", _op_name(kind), " count"),
        )
        if (
            kind != OP_JOIN_ROWS
            and kind != OP_TAKE_ROWS
            and kind != OP_JOIN_COLS
        ):
            kernels += got[kind]
    assert_equal(
        kernels,
        kernel_launches,
        String("kernel launches per ", kind_name, " block"),
    )
    assert_equal(
        len(ops),
        total,
        String("launches plus layout ops per ", kind_name, " block"),
    )


def _check_single_census(ops: List[RecordedOp]) raises:
    """The single block's census: 14 kernel launches and 1 layout op.

    The same source as the double block's (`MEGAKERNEL_IMPLEMENTATION_PLAN.md`
    sec 1.2, the `per-block sgl` column) and the same reconciliation against the
    measured 450 launches. The one layout row is `joinCols`, which the driver
    does on the host; unlike the double block's `joinRows`, it interleaves at
    row granularity, so a sink that reasons about partitions has to tag it
    LAYOUT -- compatible with nothing -- rather than inherit the row blocking
    every other op here keeps.
    """
    var want = List[Int]()
    for _ in range(NUM_OPS):
        want.append(0)
    want[OP_MATMUL] = 5
    want[OP_LAYER_NORM] = 1
    want[OP_ADA_MODULATE] = 1
    want[OP_RMS_NORM] = 2
    want[OP_ROPE] = 2
    want[OP_MHA] = 1
    want[OP_GATE_RESIDUAL_CLAMP] = 1
    want[OP_SWIGLU] = 1
    want[OP_JOIN_COLS] = 1

    _assert_census(ops, want, 14, 15, "single")


# ===-----------------------------------------------------------------------===#
# Gate 2: dataflow.
# ===-----------------------------------------------------------------------===#


def _check_dataflow(
    ops: List[RecordedOp], inputs: List[Int], outputs: List[Int]
) raises:
    """No op reads a buffer nothing has written, and the streams come back out.

    Weights and the block's stream inputs are live on entry; every other slot is
    scratch and must be written before it is read. A reordering that moves a
    consumer ahead of its producer fails here, which is the property a derived
    schedule would have to preserve.

    `inputs` and `outputs` are the block's, not a constant, because the single
    block's are the ping-pong's two halves and which is which is the block
    index's parity. Passing them in is also what makes the odd-block case a real
    check rather than a re-run: an odd block that still read `JOINT` would read
    a buffer the previous block did not write.
    """
    var live = List[Bool]()
    for i in range(NUM_SLOTS + NUM_WEIGHTS):
        live.append(i >= WEIGHT_TABLE_BASE)
    for i in range(len(inputs)):
        live[inputs[i]] = True

    var written = List[Bool]()
    for _ in range(NUM_SLOTS + NUM_WEIGHTS):
        written.append(False)

    for idx in range(len(ops)):
        for j in range(1, len(ops[idx].buffers)):
            assert_true(
                live[ops[idx].buffers[j]],
                String(
                    "dataflow: op ",
                    idx,
                    " (",
                    _op_name(ops[idx].kind),
                    ") reads slot ",
                    ops[idx].buffers[j],
                    " before anything wrote it",
                ),
            )
        var dst = ops[idx].buffers[0]
        assert_true(
            dst < WEIGHT_TABLE_BASE,
            String("dataflow: op ", idx, " writes a weight"),
        )
        live[dst] = True
        written[dst] = True

    for i in range(len(outputs)):
        assert_true(
            written[outputs[i]],
            String("the block must write slot ", outputs[i], " back"),
        )


# ===-----------------------------------------------------------------------===#
# Gate 3: shapes, derived independently from the geometry.
# ===-----------------------------------------------------------------------===#


def _check_shapes(cfg: KleinConfig, ops: List[RecordedOp]) raises:
    """Every extent against the config, computed here rather than read back."""
    var dim = cfg.heads * 128
    var ir = cfg.image_seq
    var tr = cfg.text_seq
    var hidden = cfg.mlp_ratio * dim

    var matmuls = List[Int]()
    for i in range(len(ops)):
        if ops[i].kind == OP_MATMUL:
            matmuls.append(i)
    assert_equal(len(matmuls), 12, "matmuls per block")

    # The six stream projections are square in `dim`: three image, three text.
    for p in range(6):
        var at = matmuls[p]
        var rows = ir if p < 3 else tr
        assert_equal(ops[at].scalars[0], rows, "projection rows")
        assert_equal(ops[at].scalars[1], dim, "projection K")
        assert_equal(ops[at].scalars[2], dim, "projection N")

    # The two out-projections, then `linear_in` widening to gate||up and
    # `linear_out` contracting the SwiGLU half back to `dim`.
    for p in range(6, 8):
        var at = matmuls[p]
        assert_equal(ops[at].scalars[1], dim, "out-projection K")
        assert_equal(ops[at].scalars[2], dim, "out-projection N")
    for p in range(8, 10):
        var at = matmuls[p]
        assert_equal(ops[at].scalars[1], dim, "linear_in K")
        assert_equal(ops[at].scalars[2], 2 * hidden, "linear_in N is gate||up")
    for p in range(10, 12):
        var at = matmuls[p]
        assert_equal(
            ops[at].scalars[1], hidden, "linear_out K is the SwiGLU half"
        )
        assert_equal(ops[at].scalars[2], dim, "linear_out N")

    for i in range(len(ops)):
        var op = ops[i].copy()
        if op.kind == OP_SWIGLU:
            assert_equal(op.scalars[1], hidden, "SwiGLU hidden")
        elif op.kind == OP_RMS_NORM or op.kind == OP_ROPE:
            # Head vectors, not token rows: the one place the row count is not
            # the stream's, and the one the driver gets wrong most easily.
            var rows = op.scalars[0]
            assert_true(
                rows == cfg.heads * ir
                or rows == cfg.heads * tr
                or rows == cfg.heads * (ir + tr),
                String(_op_name(op.kind), " must run on head rows"),
            )
            assert_equal(op.scalars[1], 128, "head vectors are head_dim wide")
        elif op.kind == OP_MHA:
            assert_equal(op.scalars[0], ir + tr, "attention sees the joint seq")
            assert_equal(op.scalars[1], cfg.heads, "attention head count")
        elif op.kind == OP_LAYER_NORM or op.kind == OP_ADA_MODULATE:
            assert_equal(op.scalars[1], dim, "norm/modulate width")


def _check_single_shapes(cfg: KleinConfig, ops: List[RecordedOp]) raises:
    """The single block's extents, likewise computed here rather than read back.

    Three things the double block cannot check, because it does not have them:
    every row count is the JOINT one, the fused MLP projection widens to
    `2 * mlp_hidden` off the SAME modulated buffer the q/k/v projections read,
    and the out-projection reduces over `dim + mlp_hidden`. That last extent is
    the one a transliteration normalises to `dim` without noticing -- it is the
    only `K` in the step that is not a buffer's own width.
    """
    var dim = cfg.heads * 128
    var jr = cfg.text_seq + cfg.image_seq
    var hidden = cfg.mlp_ratio * dim

    var matmuls = List[Int]()
    for i in range(len(ops)):
        if ops[i].kind == OP_MATMUL:
            matmuls.append(i)
    assert_equal(len(matmuls), 5, "matmuls per single block")

    for p in range(3):
        var at = matmuls[p]
        assert_equal(ops[at].scalars[0], jr, "fused projection rows")
        assert_equal(ops[at].scalars[1], dim, "fused projection K")
        assert_equal(ops[at].scalars[2], dim, "fused projection N")

    var mlp_in = matmuls[3]
    assert_equal(ops[mlp_in].scalars[0], jr, "fused MLP rows")
    assert_equal(ops[mlp_in].scalars[1], dim, "fused MLP K")
    assert_equal(ops[mlp_in].scalars[2], 2 * hidden, "fused MLP N is gate||up")

    var out = matmuls[4]
    assert_equal(ops[out].scalars[0], jr, "out-projection rows")
    assert_equal(
        ops[out].scalars[1], dim + hidden, "out-projection K is attn||mlp"
    )
    assert_equal(ops[out].scalars[2], dim, "out-projection N")

    for i in range(len(ops)):
        var op = ops[i].copy()
        if op.kind == OP_SWIGLU:
            assert_equal(op.scalars[1], hidden, "SwiGLU hidden")
        elif op.kind == OP_RMS_NORM or op.kind == OP_ROPE:
            assert_equal(op.scalars[0], cfg.heads * jr, "head rows are joint")
            assert_equal(op.scalars[1], 128, "head vectors are head_dim wide")
        elif op.kind == OP_MHA:
            assert_equal(op.scalars[0], jr, "attention sees the joint seq")
            assert_equal(op.scalars[1], cfg.heads, "attention head count")
        elif op.kind == OP_LAYER_NORM or op.kind == OP_ADA_MODULATE:
            assert_equal(op.scalars[0], jr, "norm/modulate rows")
            assert_equal(op.scalars[1], dim, "norm/modulate width")
        elif op.kind == OP_JOIN_COLS:
            # By column: the two widths add to the out-projection's K. Checked
            # as an ADDITION rather than as two literals, because the failure
            # this is here for is a `join_rows` in disguise -- correctly shaped,
            # wrong operand -- and only the relation to `K` catches it.
            assert_equal(op.scalars[0], dim, "join_cols left width")
            assert_equal(op.scalars[1], hidden, "join_cols right width")
            assert_equal(op.scalars[2], jr, "join_cols rows")
            assert_equal(
                op.scalars[0] + op.scalars[1],
                ops[out].scalars[1],
                "the joined widths must be the out-projection's K",
            )


# ===-----------------------------------------------------------------------===#
# Gate 4: the sequence through the megastep walker.
# ===-----------------------------------------------------------------------===#


struct SequenceSeam(MegastepSeam):
    """The walk's seam, recording the launches it is presented with.

    `buffer_count`/`scalar_count` are this sequence's manifest: the arity each
    family's kernel takes. The walk refuses a block whose counted layout
    disagrees, which is exactly the check being exercised.
    """

    var buffers_per_op: List[Int]
    var scalars_per_op: List[Int]
    var seen_kind: List[Int]
    var seen_buffers: List[Int]

    def __init__(
        out self, var buffers_per_op: List[Int], var scalars_per_op: List[Int]
    ):
        self.buffers_per_op = buffers_per_op^
        self.scalars_per_op = scalars_per_op^
        self.seen_kind = List[Int]()
        self.seen_buffers = List[Int]()

    def buffer_count(self, selector: UInt32) -> Int:
        return self.buffers_per_op[Int(selector)]

    def scalar_count(self, selector: UInt32) -> Int:
        return self.scalars_per_op[Int(selector)]

    def launch(
        mut self,
        selector: UInt32,
        thread_mask: UInt32,
        args: WireBytes,
        args_bytes: Int,
        buffers: SlotBases,
        lengths: SlotLengths,
    ) -> Int32:
        self.seen_kind.append(Int(selector))
        for j in range(self.buffers_per_op[Int(selector)]):
            self.seen_buffers.append(Int(buffers[j]))
        _ = thread_mask
        _ = args
        _ = args_bytes
        _ = lengths
        return 0


@always_inline
def _put_u32(mut block: List[UInt8], at: Int, value: UInt32):
    for i in range(4):
        block[at + i] = UInt8((value >> UInt32(8 * i)) & 0xFF)


@always_inline
def _put_u16(mut block: List[UInt8], at: Int, value: UInt32):
    block[at] = UInt8(value & 0xFF)
    block[at + 1] = UInt8((value >> 8) & 0xFF)


@always_inline
def _put_u64(mut block: List[UInt8], at: Int, value: UInt64):
    for i in range(8):
        block[at + i] = UInt8((value >> UInt64(8 * i)) & 0xFF)


def _encode(ops: List[RecordedOp], arity_skew: Int) raises -> List[UInt8]:
    """The recorded sequence as a RUN-mode `'NMGS'` block.

    Every operand is a TABLE reference: the session table names all of the
    block's activation slots and weights, which is the shape the megastep path
    registers a step's allocations in. `arity_skew` is the mutation control --
    the op index whose declared buffer count is off by one, or -1 for none.
    """
    var count = len(ops)
    var args_base = MEGASTEP_HEADER_BYTES + count * MEGASTEP_RECORD_BYTES
    var args_bytes = 0
    for i in range(len(ops)):
        args_bytes += 8 * (len(ops[i].buffers) + len(ops[i].scalars))

    var block = List[UInt8]()
    for _ in range(args_base + args_bytes):
        block.append(0)

    _put_u32(block, 0, MEGASTEP_MAGIC)
    _put_u32(block, 4, UInt32(count))
    _put_u32(block, 8, 0)
    _put_u32(block, 12, MEGASTEP_MODE_RUN)

    var args_at = args_base
    for i in range(count):
        var record = MEGASTEP_HEADER_BYTES + i * MEGASTEP_RECORD_BYTES
        var nbuf = len(ops[i].buffers)
        var nscalar = len(ops[i].scalars)
        _put_u32(block, record, UInt32(ops[i].kind))
        _put_u32(block, record + 4, 0x1F)
        _put_u32(block, record + 8, UInt32(args_at))
        var declared = nbuf + 1 if i == arity_skew else nbuf
        _put_u32(block, record + 12, UInt32(declared) | (UInt32(nscalar) << 16))
        for j in range(nbuf):
            _put_u16(
                block,
                record + 16 + 2 * j,
                MEGASTEP_REF_TABLE | UInt32(ops[i].buffers[j]),
            )
        for s in range(nscalar):
            _put_u64(block, args_at + 8 * s, UInt64(ops[i].scalars[s]))
        for j in range(nbuf):
            _put_u64(block, args_at + 8 * (nscalar + j), 0)
        args_at += 8 * (nbuf + nscalar)

    return block^


def _check_walker(ops: List[RecordedOp], arity_skew: Int) raises -> Int32:
    """Walks the encoded sequence and asserts what comes back is what went in.
    """
    var buffers_per_op = List[Int]()
    var scalars_per_op = List[Int]()
    for _ in range(NUM_OPS):
        buffers_per_op.append(0)
        scalars_per_op.append(0)
    for i in range(len(ops)):
        buffers_per_op[ops[i].kind] = len(ops[i].buffers)
        scalars_per_op[ops[i].kind] = len(ops[i].scalars)

    var table_base_list = List[UInt]()
    var table_bytes_list = List[UInt32]()
    for i in range(Int(MEGASTEP_TABLE_ENTRIES)):
        if i < NUM_SLOTS + NUM_WEIGHTS:
            table_base_list.append(UInt(TWIN_BASE + i * TWIN_STRIDE))
            table_bytes_list.append(UInt32(TWIN_BYTES))
        else:
            table_base_list.append(0)
            table_bytes_list.append(0)

    var block = _encode(ops, arity_skew)
    var slot_bases = List[UInt]()
    var slot_lengths_list = List[Int32]()
    for _ in range(MEGASTEP_BUFFER_SLOTS):
        slot_bases.append(UInt(TWIN_BASE))
        slot_lengths_list.append(Int32(TWIN_BYTES))

    var seam = SequenceSeam(buffers_per_op^, scalars_per_op^)
    var status = megastep_walk(
        seam,
        TableBases(unsafe_from_address=Int(table_base_list.unsafe_ptr())),
        TableBytes(unsafe_from_address=Int(table_bytes_list.unsafe_ptr())),
        WireBytes(unsafe_from_address=Int(block.unsafe_ptr())),
        len(block),
        UInt32(len(ops)),
        SlotBases(unsafe_from_address=Int(slot_bases.unsafe_ptr())),
        SlotLengths(unsafe_from_address=Int(slot_lengths_list.unsafe_ptr())),
        False,
    )
    if status != 0:
        return status

    assert_equal(
        len(seam.seen_kind), len(ops), "the walk presented every launch"
    )
    var at = 0
    for i in range(len(ops)):
        assert_equal(seam.seen_kind[i], ops[i].kind, "walked selector")
        for j in range(len(ops[i].buffers)):
            assert_equal(
                seam.seen_buffers[at],
                TWIN_BASE + ops[i].buffers[j] * TWIN_STRIDE,
                String("walked operand ", j, " of launch ", i),
            )
            at += 1
    return 0


def main() raises:
    var configs = List[KleinConfig]()
    configs.append(klein_512())
    configs.append(klein_mini())
    for c in range(len(configs)):
        var cfg = configs[c].copy()
        var ops = _record(cfg, 0)
        _check_census(ops)
        _check_dataflow(ops, [IMG.index, TXT.index], [IMG.index, TXT.index])
        _check_shapes(cfg, ops)
        assert_equal(_check_walker(ops, -1), Int32(0), "walk status")

        # The block index changes which weights the sink resolves, never the
        # sequence: 25 structurally identical blocks is what makes the step's
        # loop a loop.
        var later = _record(cfg, cfg.num_double - 1)
        assert_equal(len(later), len(ops), "block index changed the length")
        for i in range(len(ops)):
            assert_equal(later[i].kind, ops[i].kind, "block index changed a op")

        # The live control on gate 4: a declared arity the manifest does not
        # have must be refused rather than decoded.
        assert_true(
            _check_walker(ops, 0) != 0,
            "an arity skew must refuse the block",
        )

        # The same four gates on the single block, and a fifth property it
        # alone has: the ping-pong. An even block reads `JOINT` and writes
        # `JOINT_ALT`, an odd block the reverse, and both directions are walked
        # -- a body that ignored the parity would pass the even case and read an
        # unwritten buffer in the odd one.
        for block in range(2):
            var sgl = _record_single(cfg, block)
            _check_single_census(sgl)
            _check_dataflow(
                sgl,
                [single_stream_in(block).index],
                [single_stream_out(block).index],
            )
            _check_single_shapes(cfg, sgl)
            assert_equal(_check_walker(sgl, -1), Int32(0), "single walk status")
            assert_true(
                _check_walker(sgl, 0) != 0,
                "an arity skew must refuse the single block",
            )

        assert_equal(
            single_stream_in(1).index,
            single_stream_out(0).index,
            "the ping-pong must hand the odd block what the even one wrote",
        )
        assert_true(
            single_stream_in(0) != single_stream_in(1),
            "the ping-pong must alternate its two allocations",
        )

        # The two kinds share the six joint buffers the driver's `DevScratch`
        # aliases, and that sharing is what keeps a hazard proof conservative.
        # Asserted rather than left to the id table: private copies would read
        # as a tidy-up and would silently tell a proof the kinds are
        # independent.
        var shared = [
            Q_JOINT.index,
            K_JOINT.index,
            V_JOINT.index,
            Q_ROPE.index,
            K_ROPE.index,
            ATTN_JOINT.index,
        ]
        for i in range(len(shared)):
            var in_dbl = False
            var in_sgl = False
            for j in range(len(ops)):
                for b in range(len(ops[j].buffers)):
                    if ops[j].buffers[b] == shared[i]:
                        in_dbl = True
            var sgl0 = _record_single(cfg, 0)
            for j in range(len(sgl0)):
                for b in range(len(sgl0[j].buffers)):
                    if sgl0[j].buffers[b] == shared[i]:
                        in_sgl = True
            assert_true(
                in_dbl and in_sgl,
                String("slot ", shared[i], " must be touched by both kinds"),
            )

    print("ALL BLOCK SEQUENCE TESTS PASSED")
