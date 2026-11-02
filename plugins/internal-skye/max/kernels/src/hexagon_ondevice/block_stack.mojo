"""The MMDiT blocks' op sequences, as straight-line Mojo (slice 9).

`npu_block_stack_dispatch.cpp:2823-2961` and `:2972-3046` are the only places
that know what one FLUX.2-Klein transformer block IS: which kernels run, in what
order, over which buffers, at which extents. They know it in C++, written out by
hand, and nothing else in the tree can be read against them. The megakernel
plan's slice 9 is to say the same thing in Mojo, once, and to gate the
transliteration BEFORE anything is composed out of it
(`MEGAKERNEL_IMPLEMENTATION_PLAN.md` sec 7.1, "the transliteration is validated
before it is composed").

This is the straight-line form and it is deliberately the straight-line form.
`MEGAKERNEL_VM_DISPATCH_NOTE.md` sec 7 makes it the differential baseline a
VM-shaped composition would later have to reproduce: a schedule derived by a
framework is only checkable against a sequence somebody wrote down and gated.
So there is no scheduler here, no cost model, no dependency solver -- 38 and 15
statements in the order the driver runs them.

What varies and what does not. Within each kind, across the step's 25 blocks,
there is no structural variation at all: the only intra-loop differences are the
single block's ping-pong (which the block index's parity resolves) and the weight
index. Both bodies therefore take the block index as an argument and the geometry
as one `KleinConfig`, and every extent they pass is derived from that config
rather than restated -- an extent that disagrees with the buffer it describes is
the failure mode this whole effort is trying to remove from the argument lists.

Where the kernels are. They are NOT called here. Both bodies are parameterized
on a `BlockOps` sink, the same discipline `megastep_walk` follows with
`MegastepSeam`: one body, two instantiations, so a host twin can drive the exact
sequence a device instantiation would run. The twin binds a recorder, which is
what makes the sequence checkable with no device and no arena; slice 10 binds
the kernels themselves, at which point the arena (slice 6) and the weight layout
(slice 8) are what the sink resolves a `Slot` and a `Weight` through. Neither
exists yet, and neither needs to for the sequence to be right.

Operands are `Slot` and `Weight`, not integers. Both wrap an index and neither
converts to the other, so a weight cannot be passed where an activation belongs
-- the class of mistake a 53-statement transliteration of somebody else's
argument lists is otherwise wide open to, and the reason this file has no
function taking a bare `Int` operand.
"""

from hexagon_ondevice.attention_kernel import FA_HEAD_DIM


@fieldwise_init
struct Slot(Copyable, Equatable, ImplicitlyCopyable, Movable):
    """One activation buffer of the block: a stream, or a piece of scratch.

    An index into whatever the sink resolves buffers through -- the twin's
    recorder resolves it to a name, slice 10's sink will resolve it to a view
    carved from the VTCM arena. It is a type rather than an `Int` so that the
    sequence below cannot pass a weight where an activation goes.
    """

    var index: Int

    def __eq__(self, other: Self) -> Bool:
        return self.index == other.index

    def __ne__(self, other: Self) -> Bool:
        return self.index != other.index


@fieldwise_init
struct Weight(Copyable, Equatable, ImplicitlyCopyable, Movable):
    """One weight or modulation parameter, by index within the block's set.

    Per-block weights (`WQ`, `W_FF_IN`, ...) are named once here and paired with
    the block index by the sink, because which block's copy is the caller's
    business and the sequence is identical for every block.
    """

    var index: Int

    def __eq__(self, other: Self) -> Bool:
        return self.index == other.index

    def __ne__(self, other: Self) -> Bool:
        return self.index != other.index


# ===-----------------------------------------------------------------------===#
# The block's activation buffers, in the order `DevScratch` declares them
# (`npu_block_stack_dispatch.cpp`). The two streams come in and go out; every
# other slot is scratch the block owns.
# ===-----------------------------------------------------------------------===#

comptime IMG = Slot(0)
"""The image stream: the block's first input and its last output."""
comptime TXT = Slot(1)
"""The text stream: the block's second input and its last output."""
comptime NORM_IMG = Slot(2)
comptime NORM_TXT = Slot(3)
comptime MOD_IMG = Slot(4)
comptime MOD_TXT = Slot(5)
comptime Q_IMG = Slot(6)
comptime K_IMG = Slot(7)
comptime V_IMG = Slot(8)
comptime Q_TXT = Slot(9)
comptime K_TXT = Slot(10)
comptime V_TXT = Slot(11)
comptime QN_IMG = Slot(12)
comptime KN_IMG = Slot(13)
comptime QN_TXT = Slot(14)
comptime KN_TXT = Slot(15)
comptime Q_JOINT = Slot(16)
comptime K_JOINT = Slot(17)
comptime V_JOINT = Slot(18)
comptime Q_ROPE = Slot(19)
comptime K_ROPE = Slot(20)
comptime ATTN_JOINT = Slot(21)
comptime ATTN_IMG = Slot(22)
comptime ATTN_TXT = Slot(23)
comptime OUT_IMG = Slot(24)
comptime OUT_TXT = Slot(25)
comptime IMG_MID = Slot(26)
comptime TXT_MID = Slot(27)
comptime FF_IMG = Slot(28)
comptime FF_TXT = Slot(29)
comptime MLP_IMG = Slot(30)
comptime MLP_TXT = Slot(31)
comptime FFO_IMG = Slot(32)
comptime FFO_TXT = Slot(33)

# ===-----------------------------------------------------------------------===#
# The single block's private buffers. It is single-STREAM, not single-kernel:
# the two streams have already been joined, so where the double block carries an
# image and a text copy of everything this one carries a `*Joint` of each.
#
# Six of its buffers are NOT here, and that is the point. `Q_JOINT`, `K_JOINT`,
# `V_JOINT`, `Q_ROPE`, `K_ROPE` and `ATTN_JOINT` above are the SAME `DevScratch`
# members the driver hands both block kinds (`npu_block_stack_dispatch.cpp`
# `runDoubleBlock` writes `s.qJoint` through `joinRows`, `runSingleBlock` writes
# it through a projection), so they are shared here too. Giving the single block
# private copies would read as tidier and would be a lie to the hazard proof: it
# would say the two kinds touch disjoint buffers when the driver's allocations
# alias, and a proof that believes two ops are independent DROPS the barrier
# between them. Sharing can only add barriers, which is the direction a
# conservative proof is allowed to be wrong in.
# ===-----------------------------------------------------------------------===#

comptime JOINT = Slot(34)
"""The joint stream: the single block's input and, via the ping-pong, output."""
comptime NORM_JOINT = Slot(35)
comptime MOD_JOINT = Slot(36)
comptime MLP_IN_JOINT = Slot(37)
comptime QN_JOINT = Slot(38)
comptime KN_JOINT = Slot(39)
comptime MLP_JOINT = Slot(40)
comptime COMBINED = Slot(41)
"""`attn || mlp` by column: the out-projection's `dim + mlp_hidden` operand."""
comptime OUT_JOINT = Slot(42)
comptime JOINT_ALT = Slot(43)
"""The ping-pong's other half: the gated residual cannot write its own `x`."""

comptime NUM_SLOTS = 44
"""Activation buffers the step names, over both block kinds."""


# ===-----------------------------------------------------------------------===#
# The block's weights and modulation parameters (`DevDoubleBlock`, `DevWeights`).
# ===-----------------------------------------------------------------------===#

comptime WQ = Weight(0)
comptime WK = Weight(1)
comptime WV = Weight(2)
comptime WQ_ADD = Weight(3)
comptime WK_ADD = Weight(4)
comptime WV_ADD = Weight(5)
comptime RMS_Q = Weight(6)
comptime RMS_K = Weight(7)
comptime RMS_Q_ADD = Weight(8)
comptime RMS_K_ADD = Weight(9)
comptime WO = Weight(10)
comptime WO_ADD = Weight(11)
comptime W_FF_IN = Weight(12)
comptime W_FF_IN_CTX = Weight(13)
comptime W_FF_OUT = Weight(14)
comptime W_FF_OUT_CTX = Weight(15)

comptime IMG_ATTN_SCALE = Weight(16)
comptime IMG_ATTN_SHIFT = Weight(17)
comptime IMG_ATTN_GATE = Weight(18)
comptime TXT_ATTN_SCALE = Weight(19)
comptime TXT_ATTN_SHIFT = Weight(20)
comptime TXT_ATTN_GATE = Weight(21)
comptime IMG_MLP_SCALE = Weight(22)
comptime IMG_MLP_SHIFT = Weight(23)
comptime IMG_MLP_GATE = Weight(24)
comptime TXT_MLP_SCALE = Weight(25)
comptime TXT_MLP_SHIFT = Weight(26)
comptime TXT_MLP_GATE = Weight(27)

comptime NUM_DOUBLE_WEIGHTS = 28
"""Weights and modulation vectors one double block reads."""


# ===-----------------------------------------------------------------------===#
# The single block's weights (`DevSingleBlock`, `w.single`). Its projections are
# fused where the double block's are split: `to_qkv_mlp_proj` is one artefact in
# the checkpoint, staged as four allocations split along N, and its modulation
# is one triple rather than the double block's four.
#
# These are its OWN ids, unlike the activation slots above. A weight id resolves
# to an artefact and the two kinds read different artefacts, so sharing here
# would name one row for two tensors -- the opposite direction from the slots,
# where the driver's buffers really are the same memory.
# ===-----------------------------------------------------------------------===#

comptime SGL_WQ = Weight(28)
comptime SGL_WK = Weight(29)
comptime SGL_WV = Weight(30)
comptime SGL_W_MLP = Weight(31)
comptime SGL_RMS_Q = Weight(32)
comptime SGL_RMS_K = Weight(33)
comptime SGL_WO = Weight(34)
comptime SGL_SCALE = Weight(35)
comptime SGL_SHIFT = Weight(36)
comptime SGL_GATE = Weight(37)

comptime SINGLE_WEIGHT_BASE = NUM_DOUBLE_WEIGHTS
"""First id the single block's weights occupy. The two kinds' ids are disjoint
ranges of one space, so a table row is still its own id's row."""

comptime NUM_SINGLE_WEIGHTS = 10
"""Weights and modulation vectors one single block reads."""

comptime NUM_WEIGHTS = NUM_DOUBLE_WEIGHTS + NUM_SINGLE_WEIGHTS
"""Weights and modulation vectors the step reads, over both block kinds."""


@fieldwise_init
struct KleinConfig(Copyable, ImplicitlyCopyable, Movable):
    """The step geometry, from `StackConfig` (`npu_block_stack_dispatch.cpp`).

    Every extent the sequence passes is derived here rather than written at the
    call, so a shape can be wrong in one place instead of thirty-six.
    """

    var heads: Int
    """Attention heads. `dim = heads * head_dim`."""

    var text_seq: Int
    """Text tokens the encoder hands over."""

    var image_seq: Int
    """Latent tokens at the render resolution."""

    var mlp_ratio: Int
    """Feed-forward expansion; `mlp_hidden = mlp_ratio * dim`."""

    var num_double: Int
    """Double blocks in the stack."""

    var num_single: Int
    """Single blocks in the stack."""

    @always_inline
    def dim(self) -> Int:
        return self.heads * FA_HEAD_DIM

    @always_inline
    def head_dim(self) -> Int:
        """`attention_head_dim`, which the attention kernel bakes."""
        return FA_HEAD_DIM

    @always_inline
    def mlp_hidden(self) -> Int:
        return self.mlp_ratio * self.dim()

    @always_inline
    def ff_in_width(self) -> Int:
        """`linear_in`'s output width: SwiGLU consumes gate||up in one buffer.
        """
        return 2 * self.mlp_hidden()

    @always_inline
    def single_out_k(self) -> Int:
        """The single block's out-projection reduction: `attn || mlp` by column.

        12,288 at Klein, so 384 croutons -- the deepest reduction in the step,
        and the shape the audit calls `sgl out-proj`. It does NOT set the step's
        arena maximum: a k that deep leaves room for few activation panels, and
        the shallower fused projections resident at 39 panels ask for more
        (`step_arena_bytes`).
        """
        return self.dim() + self.mlp_hidden()

    @always_inline
    def text_rows(self) -> Int:
        return self.text_seq

    @always_inline
    def image_rows(self) -> Int:
        return self.image_seq

    @always_inline
    def joint_rows(self) -> Int:
        return self.text_rows() + self.image_rows()

    @always_inline
    def joint_seq(self) -> Int:
        """The sequence one attention dispatch sees. Batch is folded into rows
        everywhere else; attention is per-sequence."""
        return self.text_seq + self.image_seq

    @always_inline
    def head_rows(self, rows: Int) -> Int:
        """Rows a head-vector kernel (`rms_norm`, RoPE) sees over `rows` tokens.
        """
        return rows * self.heads


@always_inline
def klein_512() -> KleinConfig:
    """FLUX.2-Klein-4B at 512^2 (`kleinFull`), one block of each kind.

    `num_double`/`num_single` are the stack's, and the driver's default fixture
    runs one of each; the real stack is 5 and 20 (`config.py`).
    """
    return KleinConfig(
        heads=24,
        text_seq=512,
        image_seq=1024,
        mlp_ratio=3,
        num_double=5,
        num_single=20,
    )


@always_inline
def klein_mini() -> KleinConfig:
    """The reduced geometry the driver iterates on (`kleinMini`).

    `head_dim` stays 128 because the attention kernel bakes it; everything else
    shrinks, keeping every K a multiple of 64.
    """
    return KleinConfig(
        heads=4,
        text_seq=64,
        image_seq=128,
        mlp_ratio=3,
        num_double=1,
        num_single=1,
    )


trait BlockOps:
    """What the block sequence needs from whatever it is sequencing for.

    One method per kernel family plus the two pure layout ops, each taking its
    operands as `Slot`/`Weight` and its extents as the counts the manifest's
    scalars carry. A host twin binds a recorder, which is what makes the
    sequence comparable against the driver's without a device; slice 10 binds
    the kernels.

    Every method returns nothing and cannot fail: a sequence is not where a
    failure is handled. The megakernel's stage status (design sec 6.2) belongs to
    the sink, which knows what a stage IS.
    """

    def layer_norm(mut self, dst: Slot, x: Slot, rows: Int, width: Int):
        """Row-wise layer norm over `rows` rows of `width`."""
        ...

    def ada_modulate(
        mut self,
        dst: Slot,
        x: Slot,
        scale: Weight,
        shift: Weight,
        rows: Int,
        width: Int,
    ):
        """`(1 + scale) * x + shift`, the parameters broadcast across rows."""
        ...

    def matmul(
        mut self, dst: Slot, x: Slot, w: Weight, rows: Int, k: Int, n: Int
    ):
        """`dst[rows, n] = x[rows, k] @ w[k, n]`."""
        ...

    def rms_norm(
        mut self, dst: Slot, x: Slot, gamma: Weight, head_rows: Int, width: Int
    ):
        """QK-norm: RMS norm over head vectors, so its rows are head rows."""
        ...

    def join_rows(
        mut self,
        dst: Slot,
        top: Slot,
        top_rows: Int,
        bottom: Slot,
        bottom_rows: Int,
        width: Int,
    ):
        """Concatenates two streams by rows. A layout op with no kernel."""
        ...

    def join_cols(
        mut self,
        dst: Slot,
        left: Slot,
        left_width: Int,
        right: Slot,
        right_width: Int,
        rows: Int,
    ):
        """Concatenates two streams by columns. A layout op with no kernel.

        The single block's attention and MLP results become one `dim +
        mlp_hidden` operand for the out-projection, which is where the step's
        deepest reduction comes from. Unlike `join_rows` this interleaves at row
        granularity, so an implementor that treats it as a row concatenation
        produces a correctly-shaped and wrong buffer.

        For a sink that reasons about hazards, this is a LAYOUT partition -- the
        row axis stays blocked and row-local (contract C2), but the column split
        is compatible with nothing, so tag it as such. That direction can only
        add barriers, never drop one.
        """
        ...

    def rope(mut self, dst: Slot, x: Slot, head_rows: Int, head_dim: Int):
        """Interleaved RoPE over head vectors."""
        ...

    def mha(
        mut self,
        dst: Slot,
        q: Slot,
        k: Slot,
        v: Slot,
        seq: Int,
        heads: Int,
    ):
        """Dense flash attention over the joint sequence."""
        ...

    def take_rows(
        mut self, dst: Slot, src: Slot, first_row: Int, rows: Int, width: Int
    ):
        """Slices a row range out of a stream. A layout op with no kernel."""
        ...

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
        """`x + gate * y`, closing a sub-block. `clamp` is the fp16-range clamp
        the text stream's MLP residual alone applies."""
        ...

    def swiglu(mut self, dst: Slot, x: Slot, rows: Int, hidden: Int):
        """`silu(gate) * up` over a `gate||up` buffer of `2 * hidden`."""
        ...


@always_inline
def double_block[Ops: BlockOps](mut ops: Ops, cfg: KleinConfig, block: Int):
    """One MMDiT double block: 33 kernel launches and 5 layout ops, in order.

    A transliteration of `runDoubleBlock` (`npu_block_stack_dispatch.cpp:2823`)
    with its reference checks and its re-seeding removed -- those are the
    driver's differential scaffolding, not the block. What remains is the
    sequence and the operands, which is the thing nothing else in the tree
    states.

    The two streams arrive in `IMG`/`TXT` and leave in them; every other slot is
    scratch. `NORM_*`/`MOD_*` are deliberately reused by the MLP half rather
    than doubled, exactly as the driver reuses them: a second pair of full-width
    buffers per stream buys nothing.

    `block` selects which block's weights the sink resolves `Weight` against and
    is otherwise unused: there is no per-block structural variation in the
    stack, which is the property that lets 5 double blocks be one loop.
    """
    var dim = cfg.dim()
    var ir = cfg.image_rows()
    var tr = cfg.text_rows()
    var jr = cfg.joint_rows()

    # The attention half.
    ops.layer_norm(NORM_IMG, IMG, ir, dim)
    ops.layer_norm(NORM_TXT, TXT, tr, dim)
    ops.ada_modulate(MOD_IMG, NORM_IMG, IMG_ATTN_SCALE, IMG_ATTN_SHIFT, ir, dim)
    ops.ada_modulate(MOD_TXT, NORM_TXT, TXT_ATTN_SCALE, TXT_ATTN_SHIFT, tr, dim)

    # The six stream projections. At Klein's row count this is where a per-row
    # GEMV would have cost 4608 round trips per block.
    ops.matmul(Q_IMG, MOD_IMG, WQ, ir, dim, dim)
    ops.matmul(K_IMG, MOD_IMG, WK, ir, dim, dim)
    ops.matmul(V_IMG, MOD_IMG, WV, ir, dim, dim)
    ops.matmul(Q_TXT, MOD_TXT, WQ_ADD, tr, dim, dim)
    ops.matmul(K_TXT, MOD_TXT, WK_ADD, tr, dim, dim)
    ops.matmul(V_TXT, MOD_TXT, WV_ADD, tr, dim, dim)

    # QK-norm, per head vector rather than per token row.
    ops.rms_norm(QN_IMG, Q_IMG, RMS_Q, cfg.head_rows(ir), cfg.head_dim())
    ops.rms_norm(KN_IMG, K_IMG, RMS_K, cfg.head_rows(ir), cfg.head_dim())
    ops.rms_norm(QN_TXT, Q_TXT, RMS_Q_ADD, cfg.head_rows(tr), cfg.head_dim())
    ops.rms_norm(KN_TXT, K_TXT, RMS_K_ADD, cfg.head_rows(tr), cfg.head_dim())

    # The joint sequence: text leads, then image, matching the encoder-first
    # concat the reference model does before RoPE and attention.
    ops.join_rows(Q_JOINT, QN_TXT, tr, QN_IMG, ir, dim)
    ops.join_rows(K_JOINT, KN_TXT, tr, KN_IMG, ir, dim)
    ops.join_rows(V_JOINT, V_TXT, tr, V_IMG, ir, dim)

    ops.rope(Q_ROPE, Q_JOINT, cfg.head_rows(jr), cfg.head_dim())
    ops.rope(K_ROPE, K_JOINT, cfg.head_rows(jr), cfg.head_dim())

    ops.mha(ATTN_JOINT, Q_ROPE, K_ROPE, V_JOINT, cfg.joint_seq(), cfg.heads)

    ops.take_rows(ATTN_TXT, ATTN_JOINT, 0, tr, dim)
    ops.take_rows(ATTN_IMG, ATTN_JOINT, tr, ir, dim)

    ops.matmul(OUT_IMG, ATTN_IMG, WO, ir, dim, dim)
    ops.matmul(OUT_TXT, ATTN_TXT, WO_ADD, tr, dim, dim)

    ops.gate_residual(
        IMG_MID, IMG, OUT_IMG, IMG_ATTN_GATE, ir, dim, clamp=False
    )
    ops.gate_residual(
        TXT_MID, TXT, OUT_TXT, TXT_ATTN_GATE, tr, dim, clamp=False
    )

    # The MLP half, on the mid-block state.
    ops.layer_norm(NORM_IMG, IMG_MID, ir, dim)
    ops.layer_norm(NORM_TXT, TXT_MID, tr, dim)
    ops.ada_modulate(MOD_IMG, NORM_IMG, IMG_MLP_SCALE, IMG_MLP_SHIFT, ir, dim)
    ops.ada_modulate(MOD_TXT, NORM_TXT, TXT_MLP_SCALE, TXT_MLP_SHIFT, tr, dim)

    ops.matmul(FF_IMG, MOD_IMG, W_FF_IN, ir, dim, cfg.ff_in_width())
    ops.matmul(FF_TXT, MOD_TXT, W_FF_IN_CTX, tr, dim, cfg.ff_in_width())

    ops.swiglu(MLP_IMG, FF_IMG, ir, cfg.mlp_hidden())
    ops.swiglu(MLP_TXT, FF_TXT, tr, cfg.mlp_hidden())

    ops.matmul(FFO_IMG, MLP_IMG, W_FF_OUT, ir, cfg.mlp_hidden(), dim)
    ops.matmul(FFO_TXT, MLP_TXT, W_FF_OUT_CTX, tr, cfg.mlp_hidden(), dim)

    # The stream buffers are pure outputs here -- `*_MID` is the block's live
    # state -- so writing back into them is safe and the ping-pong stays two
    # deep. The text stream alone is clamped, mirroring `refDoubleBlock`.
    ops.gate_residual(IMG, IMG_MID, FFO_IMG, IMG_MLP_GATE, ir, dim, clamp=False)
    ops.gate_residual(TXT, TXT_MID, FFO_TXT, TXT_MLP_GATE, tr, dim, clamp=True)

    _ = block


@always_inline
def single_stream_in(block: Int) -> Slot:
    """The stream slot single block `block` reads.

    `runSingleBlock` ends in `std::swap(s.joint, s.jointAlt)`, so the two
    allocations trade roles every block and which one holds the live stream is
    the block index's parity. Stated once, here, because a twin that restated it
    would agree with this body by transcription rather than by construction --
    the drift `hmx_stripe_act_slots`' docstring records, in the small.
    """
    return JOINT if block % 2 == 0 else JOINT_ALT


@always_inline
def single_stream_out(block: Int) -> Slot:
    """The stream slot single block `block` writes: the other half of the swap.
    """
    return JOINT_ALT if block % 2 == 0 else JOINT


@always_inline
def single_block[Ops: BlockOps](mut ops: Ops, cfg: KleinConfig, block: Int):
    """One MMDiT single block: 14 kernel launches and 1 layout op, in order.

    A transliteration of `runSingleBlock`
    (`npu_block_stack_dispatch.cpp:2972-3046`), on the same terms as
    `double_block`: the driver's re-seeding and reference checks are its
    differential scaffolding and are not the block.

    The shape of the difference from the double block is worth stating, because
    it is not "half of one". The streams arrive already joined, so every buffer
    is a `*Joint` and every extent is `joint_rows`; the four projections come off
    one fused `to_qkv_mlp_proj` artefact staged as four allocations; the
    attention and MLP halves run in PARALLEL off one modulation rather than in
    series off two; and they rejoin by COLUMN into one `dim + mlp_hidden`
    operand, which is the step's deepest reduction.

    `block` is the ping-pong's parity and nothing else. Unlike the double
    block's, this argument is load-bearing: `runSingleBlock` swaps its two
    stream allocations at the end, so an even block reads `JOINT` and writes
    `JOINT_ALT` and an odd block does the reverse. The SEQUENCE is still
    identical for every block, which is what keeps 20 single blocks one loop.
    """
    var dim = cfg.dim()
    var jr = cfg.joint_rows()
    var stream = single_stream_in(block)
    var stream_out = single_stream_out(block)

    ops.layer_norm(NORM_JOINT, stream, jr, dim)
    ops.ada_modulate(MOD_JOINT, NORM_JOINT, SGL_SCALE, SGL_SHIFT, jr, dim)

    # `to_qkv_mlp_proj` as four matmuls off four base allocations. The weight was
    # split along N at stage time, which is bit-exact, so this is the one
    # projection without ever handing a kernel an offset pointer.
    ops.matmul(Q_JOINT, MOD_JOINT, SGL_WQ, jr, dim, dim)
    ops.matmul(K_JOINT, MOD_JOINT, SGL_WK, jr, dim, dim)
    ops.matmul(V_JOINT, MOD_JOINT, SGL_WV, jr, dim, dim)
    ops.matmul(MLP_IN_JOINT, MOD_JOINT, SGL_W_MLP, jr, dim, cfg.ff_in_width())

    ops.rms_norm(
        QN_JOINT, Q_JOINT, SGL_RMS_Q, cfg.head_rows(jr), cfg.head_dim()
    )
    ops.rms_norm(
        KN_JOINT, K_JOINT, SGL_RMS_K, cfg.head_rows(jr), cfg.head_dim()
    )

    ops.rope(Q_ROPE, QN_JOINT, cfg.head_rows(jr), cfg.head_dim())
    ops.rope(K_ROPE, KN_JOINT, cfg.head_rows(jr), cfg.head_dim())

    ops.mha(ATTN_JOINT, Q_ROPE, K_ROPE, V_JOINT, cfg.joint_seq(), cfg.heads)

    ops.swiglu(MLP_JOINT, MLP_IN_JOINT, jr, cfg.mlp_hidden())

    # By column, not by row: the out-projection reduces over `attn` and `mlp`
    # together, so the two land side by side in one row rather than one after
    # the other. An implementor that treated this as `join_rows` would produce a
    # correctly-sized and wrong operand.
    ops.join_cols(COMBINED, ATTN_JOINT, dim, MLP_JOINT, cfg.mlp_hidden(), jr)
    ops.matmul(OUT_JOINT, COMBINED, SGL_WO, jr, cfg.single_out_k(), dim)

    # Clamped, and always: unlike the double block's four gates, where only the
    # text MLP residual clamps, the single block's one residual is the fp16-range
    # clamp site (`refSingleBlock`).
    ops.gate_residual(
        stream_out, stream, OUT_JOINT, SGL_GATE, jr, dim, clamp=True
    )
