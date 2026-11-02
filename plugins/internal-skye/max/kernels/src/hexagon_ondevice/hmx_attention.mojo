"""Flash attention on the HMX matrix array: residency, the two matmuls, the seam.

Target: Hexagon v81 with `+hmx`. Everything here is the part of attention that is
LAYOUT rather than arithmetic — where the croutons live, which axis each operand
contracts over, and how scores and probabilities cross between the array and the
vector units. The softmax arithmetic itself (the max, `_exp_hvx`, the `(m, l)`
recurrence) stays in `npu_kernel.mojo` beside the HVX arm it is shared with, so
the two arms cannot drift on the numerics that are already silicon-verified.

The split is what makes any of this checkable off the device: nothing in this
file calls an HVX intrinsic that has no host model, so `test_hmx_crouton.mojo`
composes these functions into the whole flash loop and scores it against a scalar
attention reference on arm64. What a host twin can never claim is the hardware —
`hmx_mma_chain` models the MAC and `hvx_vdeal` models a permute, both from a
compiler's semantics rather than the part's.

The recipe is `ggml-hexagon`'s, read at pinned SHA `687e77892`
(`ggml/src/ggml-hexagon/htp/`), which runs this on this silicon:

  - QK^T on the array (`hmx-fa-kernels.h:51`), with `1/sqrt(D)` folded into the
    output requant record so the scale costs no pass (`flash-attn-ops.c:1908`).
  - PV on the array with the online-softmax rescale issued as ONE extra
    `(diag(corr), O_old)` pair at the head of the same accumulator chain
    (`hmx-fa-kernels.h:119`), so the vector engines never read or write `O`.
  - The final `diag(1/l) * O` normalise on the array too (`:194`).
  - Scores and probabilities cross in VTCM by one `vdeal` in
    (`flash-attn-ops.c:1125`) and one `vshuff` out (`:1331`). No DDR round trip,
    and no extra pass — the permute rides the load and store the softmax was
    doing anyway.

THE AXIS RULE IS CHOSEN BY THE CONTRACTION, NEVER BY OPERAND ROLE, and this is
the one thing here that produces a plausible wrong answer with nothing to signal
it. `K` is the QK^T weight and contracts over its COLUMN axis, so it needs the
transposing gather (`slab_is_depth=True`). `V` is the PV weight and contracts
over its ROW axis, so it takes the ACTIVATION rule (`False`) although it is a
weight. `P` and `O_old` need no repack at all: the array's `:after` output layout
(slab `m`, lane `n`) IS the next multiply's activation and weight layout, so the
loop closes on itself in VTCM. The reference corroborates each of these — its K
path is a vscatter transpose (`htp/hmx-utils.h:43`) and its V path a row-pair
vshuff (`:149`).

`1536 % 32 == 0` and `128 % 32 == 0` at the render's shape, so every crouton tile
is fully occupied at every axis and head-batching is unnecessary: the reference
only batches GQA heads because a DECODE `Q` cannot fill 32 rows.
"""
from layout.coord import Coord
from layout.tile_layout import row_major
from layout.tile_tensor import TileTensor, stack_allocation
from std.memory import AddressSpace
from std.sys import align_of, get_defined_bool, size_of

from .hmx_crouton import (
    HMX_DTYPE,
    HMX_EMIT_MMA,
    HMX_KGATHER_VSC,
    HMX_PACK_VEC,
    HMX_PACK_VEC_MHA,
    HMX_PHASE_IDLE,
    HMX_PHASE_MMA,
    HMX_PHASE_OTHER,
    HMX_PHASE_PACK_ACT,
    HMX_PHASE_PROBE,
    HMX_PHASE_SOFTMAX,
    HMX_PHASE_STAGE_WGT,
    HMX_PHASE_UNPACK,
    HMX_RECORD_BYTES,
    HMX_RECORD_LANES,
    HMX_RECORD_VECTORS,
    HMX_SM_SPLIT,
    HMX_SPLIT,
    HMX_TILE,
    HMX_TILE_BYTES,
    CroutonPanel,
    HmxOperands,
    HmxRequantRecord,
    HmxRows2D,
    hmx_kgather_vsc_selects,
    hmx_mma_chain,
    hmx_pack_panel,
    hmx_pack_panel_vsc,
    hmx_pack_tile_pair_vec,
    hmx_pack_vec_mha_selects,
    hmx_panel,
    hmx_phase_mark,
    hmx_program_record,
    hmx_subpanel,
    hmx_unpack_panel,
)
from .matmul_kernel import hmx_work_share
from .hvx_permute import (
    HVX_DEAL_SEPARATE,
    HVX_SHUFF_INTERLEAVE,
    hvx_lanes,
    hvx_vdeal,
    hvx_vshuff,
)

comptime HMX_ATTN = get_defined_bool["SKYE_HMX_ATTN", False]()
"""Whether attention has an array arm at all.

OFF by default, and off compiles every call below away, so the shipped object is
the HVX build and the inertness gate is `cmp` rather than a claim — the shape
`SKYE_HMX_EMIT_MMA`, `SKYE_HMX_PACK_VEC` and `SKYE_HMX_PHASE_PROBE` already have.
A `-D SKYE_HMX_ATTN=true` build define (`build.sh HMX_ATTN=1`) selects the other
arm from the same reviewed source, and needs `+hmx` and `SKYE_HMX_EMIT_MMA` with
it because a modelled MMA on the device would be a kernel that computes the right
answer through no array at all.

Even with the flag ON the render is untouched unless the caller asks for it: the
arm is selected by the kernel's `rt` scalar, which the graph passes as 0 and
which was reserved for exactly this. That is deliberately stronger than
"off by default" — a diagnostic build stays a build that can still render."""


comptime HMX_EPI_XPOSE = get_defined_bool["SKYE_HMX_EPI_XPOSE", False]()
"""Batches the online softmax's per-row epilogue across a row tile.

The 233 pricing put ~59% of the softmax bracket in the per-row epilogue —
serialization, not instruction count: per row, a width-1 `exp_nonpos`
(a ~20-op Horner chain that pipelines with nothing) plus scalar stat
traffic, ~3,072 rows per launch. This flag defers each row's `corr`
exponential, `l` update and diagonal scatter to one tile-level pass:
one pipelined `exp_nonpos[HMX_TILE]` where 32 dependent chains ran, the
same arithmetic per element (`_exp_hvx` carries no width branch), so the
render is bit-identical by construction. Sweeps, folds and the masked arm
are untouched; the flag-off object stays byte-identical because the taken
branch is the shipped body, unchanged.

A `-D SKYE_HMX_EPI_XPOSE=true` build define (`build.sh HMX_EPI_XPOSE=1`)
selects it; default off."""


comptime ATTN_BR = 512
"""Query rows one array-arm block carries (16 row tiles).

From the design's VTCM table. Larger `Br` re-reads the K/V stream fewer times
(`S/Br` per head) and costs `S` and `P` linearly; 512 is where the arena lands at
21% of the 8 MiB grant."""

comptime ATTN_BC = 256
"""Keys one array-arm block carries (8 col tiles).

Also the diagonal rescale's overhead denominator: the rescale is one extra
crouton pair per `Bc/32` PV pairs, so 12.5% here. Doubling it halves that and
doubles `S` and `P`; which is right is a measurement, not a preference."""

comptime ATTN_THREADS = 4
"""Row-buffer slots the arena carries: the launch's compute threads.

Slices 2 and 3 run one thread and use slot 0. The other three are carved anyway
so the feeder split (slice 4) inherits an arena it does not have to re-plan."""


comptime AttnStats[space: AddressSpace] = TileTensor[
    DType.float32,
    type_of(row_major(Coord(0, 0))),
    origin=MutUntrackedOrigin,
    address_space=space,
]
"""An fp32 `(rows, cols)` scratch: the `(m, l)` recurrence and the row buffers.

fp32 for the reason the HVX arm's accumulator is: a length-`Bc` row of equal
probabilities sums to `Bc`, and fp16 is exact only to 2048."""


comptime ATTN_STAT_M = 0
"""Row of `stats` holding the running max per query row."""

comptime ATTN_STAT_L = 1
"""Row of `stats` holding the running denominator per query row."""

comptime ATTN_STAT_ROWS = 4
"""Rows carved for `stats`. Two are live; the design's table budgets four."""


@fieldwise_init
struct HmxAttnPlan(Copyable, Movable):
    """The block geometry one head's array arm runs at, in crouton tiles.

    Held as tile counts rather than element extents because every loop below
    walks tiles: a plan that carried `Br` would have to divide by 32 at each
    use, and the one place a shape is checked against the array's 32x32 geometry
    would be spread over the file.
    """

    var br_tiles: Int
    """Row tiles per query block: `Br / 32`."""

    var bc_tiles: Int
    """Column tiles per key block: `Bc / 32`. Must be EVEN — the softmax seam's
    permute consumes crouton vectors in pairs."""

    var d_tiles: Int
    """Head-dimension tiles: `D / 32`. Both the QK^T depth and PV's output
    width, which is why one number serves both."""

    var threads: Int
    """Row-buffer slots. Feeders index by launch index; slices 2-3 use 0."""

    @always_inline
    def usable(self) -> Bool:
        """Whether this plan can be run at all.

        The odd-`bc_tiles` refusal is the seam's, not the array's: a lone crouton
        vector has no partner to deal against, and a partial step would read 32
        columns past the block. The reference refuses the analogous shape by
        returning `HTP_STATUS_NO_SUPPORT` (`flash-attn-ops.c:1797`).
        """
        return (
            self.br_tiles > 0
            and self.bc_tiles > 0
            and (self.bc_tiles & 1) == 0
            and self.d_tiles > 0
            and self.threads > 0
        )


@always_inline
def hmx_attn_plan(br: Int, bc: Int, d: Int, threads: Int) -> HmxAttnPlan:
    """The plan for a `(Br, Bc, D)` block, in tiles. Element extents in, tiles
    out.

    A shape that is not a whole number of tiles yields a plan `usable()` refuses,
    rather than a truncated one: a partial tile is the class of bug that reads
    plausible.
    """
    var br_tiles = br // HMX_TILE
    var bc_tiles = bc // HMX_TILE
    var d_tiles = d // HMX_TILE
    if (
        br_tiles * HMX_TILE != br
        or bc_tiles * HMX_TILE != bc
        or d_tiles * HMX_TILE != d
    ):
        return HmxAttnPlan(0, 0, 0, 0)
    return HmxAttnPlan(br_tiles, bc_tiles, d_tiles, threads)


@always_inline
def hmx_attn_accepts(plan: HmxAttnPlan, seq_len: Int) -> Bool:
    """Whether one head of `seq_len` tokens divides into this plan's blocks.

    Exact division only. A ragged tail is expressible — mask the scores — but it
    is not expressible for FREE, and slices 2-3 exist to settle the numerics of
    the clean case first. The render's `1536` divides by both 512 and 256.
    """
    if not plan.usable():
        return False
    var br = plan.br_tiles * HMX_TILE
    var bc = plan.bc_tiles * HMX_TILE
    return seq_len > 0 and seq_len % br == 0 and seq_len % bc == 0


@fieldwise_init
struct HmxAttnArena[space: AddressSpace = AddressSpace.SHARED](
    Copyable, Movable
):
    """One head's VTCM residency, as views rather than bases.

    Sized from the design's table at `Br = 512, Bc = 256, D = 128`: 1696 KiB of
    croutons plus 16.5 KiB of fp32 scratch and records, 1.672 MiB against an
    8 MiB grant. Every buffer the table names is here and at the size it names,
    including the ping-pong halves slices 2-3 do not use — a slice that re-plans
    the arena is a slice whose VTCM arithmetic has to be re-argued.

    The grant itself is NOT this struct's business. It is process-standing
    (`_qshim_get_base_vtcm_addr`) and each dispatch re-carves from the base, so
    attention's arena and the matmul stripe's are never live at the same time;
    dispatches serialise on the array.

    Parameters:
        space: Where the regions live. `AddressSpace.SHARED` is how VTCM
            residency is spelled on this part; the host twin says `GENERIC`
            rather than claiming a residency the host does not have.
    """

    var q: CroutonPanel[Self.space]
    """`br_tiles * d_tiles` croutons, row-tile major. Activation rule."""

    var o: CroutonPanel[Self.space]
    """`2 * br_tiles * d_tiles` croutons: the accumulator ping-pong. A chain
    reads `O_old` as its weight and writes `O_new`, so the two cannot be one
    buffer."""

    var k: CroutonPanel[Self.space]
    """`2 * bc_tiles * d_tiles` croutons, key-tile major so QK^T's depth run is
    contiguous. Weight rule (the transposing gather)."""

    var v: CroutonPanel[Self.space]
    """`2 * d_tiles * bc_tiles` croutons, DV-tile major so PV's key run is
    contiguous. Activation rule, although V is the weight operand."""

    var s: CroutonPanel[Self.space]
    """`2 * br_tiles * bc_tiles` score croutons, row-tile major."""

    var p: CroutonPanel[Self.space]
    """`2 * br_tiles * bc_tiles` probability croutons, laid out as `s` is."""

    var diag: CroutonPanel[Self.space]
    """`br_tiles` diagonal croutons: `corr` during the loop, `1/l` after it.

    One set, not two. The reference allocates a full block-diagonal
    `[g_br, g_br]` and reads only its diagonal blocks
    (`flash-attn-ops.c:1618`); these are contiguous, which is the same reads and
    16x less VTCM. Only the diagonal positions are ever written, so the zeros
    around them are written once at carve time and never again."""

    var stats: AttnStats[Self.space]
    """`[ATTN_STAT_ROWS, Br]` fp32: the `(m, l)` recurrence per query row."""

    var rows: AttnStats[Self.space]
    """`[2 * threads, Bc]` fp32: the softmax's row buffers, two per thread for
    the even/odd query-row phase the crouton interleave produces."""

    var rec_id: HmxRequantRecord[Self.space]
    """Pass-through requant: PV and the final normalise."""

    var rec_qk: HmxRequantRecord[Self.space]
    """`1/sqrt(D)` requant: QK^T. The scale costs no pass of its own."""

    @always_inline
    def q_run(self, plan: HmxAttnPlan, r: Int) -> CroutonPanel[Self.space]:
        """Row tile `r`'s activation croutons: QK^T's whole depth run."""
        return hmx_subpanel(self.q, r * plan.d_tiles, plan.d_tiles)

    @always_inline
    def k_run(self, plan: HmxAttnPlan, j: Int) -> CroutonPanel[Self.space]:
        """Key tile `j`'s weight croutons: QK^T's whole depth run."""
        return self.k_run_at(plan, 0, j)

    @always_inline
    def k_run_at(
        self, plan: HmxAttnPlan, half: Int, j: Int
    ) -> CroutonPanel[Self.space]:
        """Key tile `j`'s weight croutons in staging half `half`.

        The half is what makes a feeder's pack and the array's read disjoint
        while both run in the same step: slice 4 stages key block `j + 1` into
        `(j + 1) & 1` while the array reads block `j` out of `j & 1`. Slices 2
        and 3 are single-threaded and take half 0 through the wrapper above,
        which is why the two spellings are one implementation.
        """
        return hmx_subpanel(
            self.k, (half * plan.bc_tiles + j) * plan.d_tiles, plan.d_tiles
        )

    @always_inline
    def v_run(self, plan: HmxAttnPlan, c: Int) -> CroutonPanel[Self.space]:
        """DV tile `c`'s weight croutons over every key tile: PV's key run."""
        return self.v_run_at(plan, 0, c)

    @always_inline
    def v_run_at(
        self, plan: HmxAttnPlan, half: Int, c: Int
    ) -> CroutonPanel[Self.space]:
        """DV tile `c`'s weight croutons over every key tile, in half `half`."""
        return hmx_subpanel(
            self.v,
            (half * plan.d_tiles + c) * plan.bc_tiles,
            plan.bc_tiles,
        )

    @always_inline
    def v_tile(
        self, plan: HmxAttnPlan, c: Int, j: Int
    ) -> CroutonPanel[Self.space]:
        """One `(dv tile, key tile)` V crouton."""
        return self.v_tile_at(plan, 0, c, j)

    @always_inline
    def v_tile_at(
        self, plan: HmxAttnPlan, half: Int, c: Int, j: Int
    ) -> CroutonPanel[Self.space]:
        """One `(dv tile, key tile)` V crouton of half `half`."""
        return hmx_subpanel(
            self.v, (half * plan.d_tiles + c) * plan.bc_tiles + j, 1
        )

    @always_inline
    def s_run(self, plan: HmxAttnPlan, r: Int) -> CroutonPanel[Self.space]:
        """Row tile `r`'s score croutons across every key tile."""
        return hmx_subpanel(self.s, r * plan.bc_tiles, plan.bc_tiles)

    @always_inline
    def s_tile(
        self, plan: HmxAttnPlan, r: Int, j: Int
    ) -> CroutonPanel[Self.space]:
        """One `(row tile, key tile)` score crouton: a QK^T output tile."""
        return self.s_tile_at(plan, 0, r, j)

    @always_inline
    def s_tile_at(
        self, plan: HmxAttnPlan, half: Int, r: Int, j: Int
    ) -> CroutonPanel[Self.space]:
        """One `(row tile, key tile)` score crouton of half `half`."""
        return hmx_subpanel(
            self.s, (half * plan.br_tiles + r) * plan.bc_tiles + j, 1
        )

    @always_inline
    def p_run(self, plan: HmxAttnPlan, r: Int) -> CroutonPanel[Self.space]:
        """Row tile `r`'s probability croutons: PV's activation run."""
        return self.p_run_at(plan, 0, r)

    @always_inline
    def p_run_at(
        self, plan: HmxAttnPlan, half: Int, r: Int
    ) -> CroutonPanel[Self.space]:
        """Row tile `r`'s probability croutons of half `half`."""
        return hmx_subpanel(
            self.p, (half * plan.br_tiles + r) * plan.bc_tiles, plan.bc_tiles
        )

    @always_inline
    def o_tile(
        self, plan: HmxAttnPlan, buf: Int, r: Int, c: Int
    ) -> CroutonPanel[Self.space]:
        """One `(row tile, dv tile)` accumulator tile of ping-pong half `buf`.
        """
        return hmx_subpanel(
            self.o,
            buf * plan.br_tiles * plan.d_tiles + r * plan.d_tiles + c,
            1,
        )

    @always_inline
    def diag_tile(self, r: Int) -> CroutonPanel[Self.space]:
        """Row tile `r`'s diagonal crouton."""
        return hmx_subpanel(self.diag, r, 1)


@always_inline
def _attn_crouton_tiles(plan: HmxAttnPlan) -> Int:
    """Croutons the arena holds, in the order `hmx_attn_arena_in` carves them.
    """
    var br_d = plan.br_tiles * plan.d_tiles
    var bc_d = plan.bc_tiles * plan.d_tiles
    var br_bc = plan.br_tiles * plan.bc_tiles
    # Q, O x2, K x2, V x2, S x2, P x2, and one set of diagonals -- the design's
    # own table, line for line, so a region that grew has one place to fail.
    return (
        br_d
        + 2 * br_d
        + 2 * bc_d
        + 2 * bc_d
        + 2 * br_bc
        + 2 * br_bc
        + plan.br_tiles
    )


@always_inline
def hmx_attn_arena_bytes(plan: HmxAttnPlan) -> Int:
    """Bytes one head's residency needs, records and fp32 scratch included.

    The crouton region comes first so every panel inherits the
    `HMX_TILE_BYTES` base alignment the `:deep` load requires; the fp32 tail
    needs only its element's, which a whole number of tiles already gives it.
    """
    if not plan.usable():
        return 0
    var stats_bytes = (
        ATTN_STAT_ROWS * plan.br_tiles * HMX_TILE * size_of[DType.float32]()
    )
    var rows_bytes = (
        2 * plan.threads * plan.bc_tiles * HMX_TILE * size_of[DType.float32]()
    )
    return (
        _attn_crouton_tiles(plan) * HMX_TILE_BYTES
        + 2 * HMX_RECORD_BYTES
        + stats_bytes
        + rows_bytes
    )


@always_inline
def hmx_attn_arena_in[
    space: AddressSpace
](
    base: UnsafePointer[UInt8, MutUntrackedOrigin, address_space=space],
    plan: HmxAttnPlan,
) -> HmxAttnArena[space]:
    """Carves `hmx_attn_arena_bytes(plan)` at `base` into the views above.

    The one place attention's residency is laid out, so a caller supplies a
    `HMX_TILE_BYTES`-aligned base with room for `hmx_attn_arena_bytes` and names
    no offset itself. A seam construction — the only reason a byte address is
    added to in this file — and everything it hands back is a coordinate space.

    The diagonal croutons are NOT zeroed here. `hmx_attn_reset_diag` does it,
    because the arena is carved once and the diagonals are reset per head.
    """
    var br_d = plan.br_tiles * plan.d_tiles
    var bc_d = plan.bc_tiles * plan.d_tiles
    var br_bc = plan.br_tiles * plan.bc_tiles

    var at = 0
    var q = base.bitcast[Scalar[HMX_DTYPE]]()
    at += br_d * HMX_TILE_BYTES
    var o = (base + at).bitcast[Scalar[HMX_DTYPE]]()
    at += 2 * br_d * HMX_TILE_BYTES
    var k = (base + at).bitcast[Scalar[HMX_DTYPE]]()
    at += 2 * bc_d * HMX_TILE_BYTES
    var v = (base + at).bitcast[Scalar[HMX_DTYPE]]()
    at += 2 * bc_d * HMX_TILE_BYTES
    var s = (base + at).bitcast[Scalar[HMX_DTYPE]]()
    at += 2 * br_bc * HMX_TILE_BYTES
    var p = (base + at).bitcast[Scalar[HMX_DTYPE]]()
    at += 2 * br_bc * HMX_TILE_BYTES
    var diag = (base + at).bitcast[Scalar[HMX_DTYPE]]()
    at += plan.br_tiles * HMX_TILE_BYTES

    var rec_id = (base + at).bitcast[Scalar[DType.uint32]]()
    at += HMX_RECORD_BYTES
    var rec_qk = (base + at).bitcast[Scalar[DType.uint32]]()
    at += HMX_RECORD_BYTES

    var stats = (base + at).bitcast[Scalar[DType.float32]]()
    at += ATTN_STAT_ROWS * plan.br_tiles * HMX_TILE * size_of[DType.float32]()
    var rows = (base + at).bitcast[Scalar[DType.float32]]()

    comptime REC_SHAPE = Coord(HMX_RECORD_VECTORS, HMX_RECORD_LANES)
    return HmxAttnArena[space](
        q=hmx_panel[space](q, br_d),
        o=hmx_panel[space](o, 2 * br_d),
        k=hmx_panel[space](k, 2 * bc_d),
        v=hmx_panel[space](v, 2 * bc_d),
        s=hmx_panel[space](s, 2 * br_bc),
        p=hmx_panel[space](p, 2 * br_bc),
        diag=hmx_panel[space](diag, plan.br_tiles),
        stats=AttnStats[space](
            ptr=stats,
            layout=row_major(Coord(ATTN_STAT_ROWS, plan.br_tiles * HMX_TILE)),
        ),
        rows=AttnStats[space](
            ptr=rows,
            layout=row_major(Coord(2 * plan.threads, plan.bc_tiles * HMX_TILE)),
        ),
        rec_id=HmxRequantRecord[space](ptr=rec_id, layout=row_major(REC_SHAPE)),
        rec_qk=HmxRequantRecord[space](ptr=rec_qk, layout=row_major(REC_SHAPE)),
    )


# ===--------------------------------------------------------------------------===
# Staging: the packers, with the axis rule each operand's CONTRACTION picks.
# ===--------------------------------------------------------------------------===


@always_inline
def hmx_attn_zero_panel[
    space: AddressSpace, //
](panel: CroutonPanel[space], tiles: Int):
    """Writes zero over `tiles` croutons, a vector at a time.

    A crouton is a flat 1024-element region under this view, so the zeroing walks
    slab pairs rather than `(slab, lane)`: the shape does not matter to a
    constant, and the vector store is what the alternative scalar nest is not.
    """
    comptime LANES = hvx_lanes[HMX_DTYPE]()
    comptime SLAB_PAIRS = HMX_TILE // HMX_SPLIT
    comptime ALIGN = align_of[Scalar[HMX_DTYPE]]()
    for t in range(tiles):
        for pair in range(SLAB_PAIRS):
            panel.store[LANES, alignment=ALIGN](
                Coord(t, pair, 0, 0), SIMD[HMX_DTYPE, LANES]()
            )


@always_inline
def hmx_attn_reset_diag[
    space: AddressSpace, //
](arena: HmxAttnArena[space], plan: HmxAttnPlan):
    """Zeroes the diagonal croutons.

    Once per head, not per key block: the only positions the loop ever writes are
    the diagonal ones, so everything around them stays the zero written here. A
    rescale that re-zeroed a whole tile per key block would spend 1024 stores to
    change 32 of them.
    """
    hmx_attn_zero_panel(arena.diag, plan.br_tiles)


@always_inline
def hmx_attn_set_diag[
    space: AddressSpace, //
](arena: HmxAttnArena[space], r: Int, m_local: Int, value: Scalar[HMX_DTYPE],):
    """Puts `value` at `(m_local, m_local)` of row tile `r`'s diagonal crouton.

    The rescale is a matmul against this tile, so a value off the diagonal would
    mix query rows — which is why nothing else here ever writes one.
    """
    arena.diag.store[1, alignment=align_of[Scalar[HMX_DTYPE]]()](
        Coord(r, m_local // HMX_SPLIT, m_local, m_local % HMX_SPLIT),
        value,
    )


@always_inline
def hmx_attn_pack_q[
    space: AddressSpace, //
](
    arena: HmxAttnArena[space],
    plan: HmxAttnPlan,
    Q: HmxRows2D,
    q0: Int,
    head_off: Int,
):
    """Stages `Br` query rows of one head as activation croutons.

    `Q` is the whole `[seq_len, num_heads * D]` tensor and `head_off` selects the
    head's columns, so the strided `[seq, heads, D]` layout is addressed by the
    packer's own `col0` rather than by a second tile type. The head's `D` columns
    are contiguous within a row, which is what makes that legal.
    """
    hmx_attn_pack_q_share(arena, plan, Q, q0, head_off, 0, plan.br_tiles)


@always_inline
def hmx_attn_pack_q_share[
    space: AddressSpace, //
](
    arena: HmxAttnArena[space],
    plan: HmxAttnPlan,
    Q: HmxRows2D,
    q0: Int,
    head_off: Int,
    r0: Int,
    r1: Int,
):
    """Row tiles `[r0, r1)` of a query block. `Q` carries no ping-pong.

    The query block stays live for the whole of its own key sweep and is replaced
    only at the next one, which the schedule separates by the store step — so
    unlike K and V it needs no second half.
    """
    for r in range(r0, r1):
        hmx_pack_panel[False](
            arena.q_run(plan, r),
            Q,
            q0 + r * HMX_TILE,
            head_off,
            plan.d_tiles,
        )


@always_inline
def hmx_attn_pack_k[
    space: AddressSpace, //
](
    arena: HmxAttnArena[space],
    plan: HmxAttnPlan,
    K: HmxRows2D,
    k0: Int,
    head_off: Int,
):
    """Stages `Bc` key rows as QK^T weight croutons: the TRANSPOSING gather.

    `slab_is_depth=True` because K contracts over its column axis (`D`), which is
    the whole of why this operand differs from V below.
    """
    hmx_attn_pack_k_share(arena, plan, K, k0, head_off, 0, 0, plan.bc_tiles)


@always_inline
def hmx_attn_pack_k_share[
    space: AddressSpace, //
](
    arena: HmxAttnArena[space],
    plan: HmxAttnPlan,
    K: HmxRows2D,
    k0: Int,
    head_off: Int,
    half: Int,
    j0: Int,
    j1: Int,
):
    """Key tiles `[j0, j1)` of a block, into staging half `half`.

    The share is a KEY TILE range because that is the axis whose destinations are
    disjoint: two feeders packing different key tiles write different croutons of
    the same panel and never the same one. Sharing over `d_tiles` instead would
    have them interleave within a crouton.

    Which packer serves the range is `hmx_kgather_vsc_selects` — the scatter
    arm under `HMX_KGATHER_VSC`, else the scalar transposing gather (the 212
    phase split's `STAGE_WGT`, 23.95% of `mha_no_cache` feeder time). The
    branch is here rather than at the step sites so every K staging of the
    dense arm — t=0 and the per-step re-pack — routes as one policy; the
    centred pack below keeps the in-panel packer, and with it the masked
    arm's object, untouched by this flag.
    """
    comptime if hmx_kgather_vsc_selects[HMX_KGATHER_VSC]():
        hmx_attn_pack_k_share_vsc(arena, plan, K, k0, head_off, half, j0, j1)
    else:
        for j in range(j0, j1):
            hmx_pack_panel[True](
                arena.k_run_at(plan, half, j),
                K,
                k0 + j * HMX_TILE,
                head_off,
                plan.d_tiles,
            )


@always_inline
def hmx_attn_pack_k_share_vsc[
    space: AddressSpace, //
](
    arena: HmxAttnArena[space],
    plan: HmxAttnPlan,
    K: HmxRows2D,
    k0: Int,
    head_off: Int,
    half: Int,
    j0: Int,
    j1: Int,
):
    """`hmx_attn_pack_k_share`'s range, staged by the scatter packer.

    Same key-tile share, same destinations, same bytes — the twin gates the
    whole-panel identity against the scalar share, both halves compared so a
    stray write counts. The scatter's `hvx_scatter_release` rides inside
    `hmx_pack_panel_vsc`, so a feeder's K croutons are drained before it
    arrives at the step barrier the array's `:deep` read sits behind — the
    ordering the reference's K phase gets from its work-queue boundary
    (`flash-attn-ops.c:628`'s caller), owed here per packed run.

    A separate function rather than a branch in the loop so the twin can run
    this arm directly on a host arena: the site's own routing is comptime on
    the build flag, which a host test cannot flip.
    """
    for j in range(j0, j1):
        hmx_pack_panel_vsc(
            arena.k_run_at(plan, half, j),
            K,
            k0 + j * HMX_TILE,
            head_off,
            plan.d_tiles,
        )


comptime ATTN_CENTER_D = 128
"""Largest head dimension the centred K pack stages on the stack.

The render's and the encoder's head dim both, so the bound costs nothing today;
it exists because the staging tile is a stack allocation and a stack allocation
needs a comptime extent. `hmx_attn_center_fits` refuses a plan past it rather
than truncating."""


@always_inline
def hmx_attn_center_fits(plan: HmxAttnPlan) -> Bool:
    """Whether the centred pack's staging tile can hold this plan's head dim."""
    return plan.usable() and plan.d_tiles * HMX_TILE <= ATTN_CENTER_D


@always_inline
def hmx_attn_pack_k_share_centered[
    space: AddressSpace, //
](
    arena: HmxAttnArena[space],
    plan: HmxAttnPlan,
    K: HmxRows2D,
    k0: Int,
    head_off: Int,
    half: Int,
    j0: Int,
    j1: Int,
    kbar_row0: Int,
):
    """`hmx_attn_pack_k_share` with a per-head reference key subtracted first.

    THE RANGE FIX (order 134). `hmx_crouton.mojo:1308` narrows the fp32
    accumulator to fp16 BEFORE the requant, so what has to survive fp16 is the
    raw dot over `D` and its ABSOLUTE error grows with magnitude -- while the
    softmax exponentiates a DIFFERENCE and needs absolute precision. An
    RMSNorm'd, RoPE'd row gives every score in a row the same large common term,
    so the small part that decides the softmax is exactly what the narrowing
    destroys.

    Subtracting a fixed vector `kbar` from every key removes that common term
    where it can still be removed -- inside the accumulator, before the
    narrowing -- and costs the answer NOTHING, because

        q_i . (k_j - kbar) = q_i . k_j - q_i . kbar

    differs from the true score by `q_i . kbar`, which depends on the query row
    and not on the key. A per-row constant is exactly what the softmax is
    invariant to: it cancels in `exp(s - max(s))`. This is not an approximation
    of the answer, only of how much headroom is recovered.

    `kbar` MUST BE ONE VECTOR PER HEAD, not one per key block. The online
    recurrence carries `(m, l)` across blocks, so two blocks shifted by different
    per-row constants would be combined as if they were not -- the one way this
    transform stops being invariant. `kbar_row0` is therefore the head's FIRST
    key row for every block, and the mean is taken over the same `HMX_TILE` keys
    every time.

    The mean over 32 keys rather than over all of them, and over the first tile
    rather than a sample: any fixed vector is equally invariant, so this choice
    only has to be a good estimate of the common term, and 32 keys of a normed
    row already are. It costs `32 * D` reads per call against a block's
    `Br * Bc * D` MACs.

    The centred keys are staged through a stack tile so that `hmx_pack_panel`
    packs them unchanged. The packers are the matmul stripe's too and the
    crouton rules are being edited in another lane; borrowing them is the whole
    point of the tile model, and forking them for one subtraction would not be.
    """
    var d = plan.d_tiles * HMX_TILE
    var kbar = stack_allocation[dtype=DType.float32](row_major[ATTN_CENTER_D]())
    for c in range(d):
        var acc = Scalar[DType.float32](0)
        for r in range(HMX_TILE):
            acc += K.load[1](Coord(kbar_row0 + r, head_off + c)).cast[
                DType.float32
            ]()
        kbar.store[1](Coord(c), acc / Scalar[DType.float32](Float32(HMX_TILE)))

    var staged = stack_allocation[dtype=HMX_DTYPE](
        row_major[HMX_TILE, ATTN_CENTER_D]()
    )
    var view = HmxRows2D(
        ptr=staged.ptr.as_unsafe_any_origin(),
        layout=row_major(Coord(HMX_TILE, ATTN_CENTER_D)),
    )
    for j in range(j0, j1):
        var row0 = k0 + j * HMX_TILE
        for lane in range(HMX_TILE):
            for c in range(d):
                var centred = (
                    K.load[1](Coord(row0 + lane, head_off + c)).cast[
                        DType.float32
                    ]()
                    - kbar.load[1](Coord(c))[0]
                )
                staged.store[1](Coord(lane, c), centred.cast[HMX_DTYPE]())
        hmx_pack_panel[True](
            arena.k_run_at(plan, half, j), view, 0, 0, plan.d_tiles
        )


@always_inline
def hmx_attn_pack_v[
    space: AddressSpace, //
](
    arena: HmxAttnArena[space],
    plan: HmxAttnPlan,
    V: HmxRows2D,
    k0: Int,
    head_off: Int,
):
    """Stages `Bc` value rows as PV weight croutons, under the ACTIVATION rule.

    V contracts over its ROW axis (keys), so rows stay rows and no transpose is
    owed. The tiles are written DV-major because PV streams over keys for a fixed
    output column, and a `:deep` load streams contiguous croutons — so the layout
    is decided by the reduction, not by the source.

    One tile per call is the cost of that order: the in-panel vector packer
    covers two adjacent croutons of a ROW, which is the wrong pair here, so
    this share is scalar even under `HMX_PACK_VEC` — order 209 measured that
    miss at 1.23x against the matmul family's 31.90x. The feeders did measure
    this binding; the answer is `hmx_attn_pack_v_share_keyed` below (the pair
    split across the column stride, `HMX_PACK_VEC_MHA`), not a second layout.
    """
    hmx_attn_pack_v_share(arena, plan, V, k0, head_off, 0, 0, plan.d_tiles)


@always_inline
def hmx_attn_pack_v_share[
    space: AddressSpace, //
](
    arena: HmxAttnArena[space],
    plan: HmxAttnPlan,
    V: HmxRows2D,
    k0: Int,
    head_off: Int,
    half: Int,
    c0: Int,
    c1: Int,
):
    """DV tiles `[c0, c1)` of a value block, into staging half `half`.

    Shared over DV tiles rather than key tiles because V is stored DV-major: a
    DV-tile range is a contiguous crouton range, which is what makes two feeders'
    stores disjoint without either learning that the panel was split.
    """
    for c in range(c0, c1):
        for j in range(plan.bc_tiles):
            hmx_pack_panel[False](
                arena.v_tile_at(plan, half, c, j),
                V,
                k0 + j * HMX_TILE,
                head_off + c * HMX_TILE,
                1,
            )


@always_inline
def hmx_attn_pack_v_share_keyed[
    space: AddressSpace, //
](
    arena: HmxAttnArena[space],
    plan: HmxAttnPlan,
    V: HmxRows2D,
    k0: Int,
    head_off: Int,
    half: Int,
    j0: Int,
    j1: Int,
):
    """Key tiles `[j0, j1)` of a value block, packed as split vector pairs.

    The vector arm's share is a KEY TILE range, not a DV range, because the
    pair step reads TWO adjacent DV tiles' columns of one key tile's rows —
    under the DV share a feeder owns one DV tile and no legal pair exists
    (`plan.d_tiles` splits 4 ways at the render's shape). Keying the share
    keeps destinations disjoint exactly as the K pack's does: two feeders
    packing different key tiles write different croutons of every DV column.
    This is the reference's own slicing — `hmx_interleave_cols_to_tiles`
    takes `[start_row, end_row)` (`ggml-hexagon` `htp/hmx-utils.h`).

    The DV loop steps by pairs; an odd `plan.d_tiles` leaves a one-column
    tail to the scalar packer, the same contract `hmx_pack_panel_vec` keeps.
    The render's `d_tiles` is 4, so the tail costs nothing today and exists
    so the share never reads a column the plan does not have.
    """
    for j in range(j0, j1):
        var row0 = k0 + j * HMX_TILE
        var c = 0
        while c + 1 < plan.d_tiles:
            hmx_pack_tile_pair_vec(
                arena.v_tile_at(plan, half, c, j),
                arena.v_tile_at(plan, half, c + 1, j),
                V,
                row0,
                head_off + c * HMX_TILE,
            )
            c += 2
        if c < plan.d_tiles:
            hmx_pack_panel[False](
                arena.v_tile_at(plan, half, c, j),
                V,
                row0,
                head_off + c * HMX_TILE,
                1,
            )


@always_inline
def hmx_attn_store_o[
    space: AddressSpace, //
](
    arena: HmxAttnArena[space],
    plan: HmxAttnPlan,
    O: HmxRows2D,
    q0: Int,
    head_off: Int,
    buf: Int,
):
    """Reads the accumulator tiles of half `buf` back out to `O`.

    An accumulator reads out as slab `m`, lane `dv` — the activation's axis
    assignment over `O[Br, D]` — so this is `hmx_unpack_panel[False]`, not a
    third rule.
    """
    hmx_attn_store_o_share(arena, plan, O, q0, head_off, buf, 0, plan.br_tiles)


@always_inline
def hmx_attn_store_o_share[
    space: AddressSpace, //
](
    arena: HmxAttnArena[space],
    plan: HmxAttnPlan,
    O: HmxRows2D,
    q0: Int,
    head_off: Int,
    buf: Int,
    r0: Int,
    r1: Int,
):
    """Row tiles `[r0, r1)` of accumulator half `buf`, read back out to `O`."""
    for r in range(r0, r1):
        for c in range(plan.d_tiles):
            # Attention's readout reports no arm: its witness is full (the probe
            # bracket owns 8..31 here) and slice 2's A/B is the stripe's.
            _ = hmx_unpack_panel[False](
                O,
                arena.o_tile(plan, buf, r, c),
                q0 + r * HMX_TILE,
                head_off + c * HMX_TILE,
                1,
            )


# ===--------------------------------------------------------------------------===
# The array's three products, and the seam between them and the vector units.
# ===--------------------------------------------------------------------------===


@always_inline
def hmx_attn_program_records[
    space: AddressSpace, //
](arena: HmxAttnArena[space], scale: Scalar[HMX_DTYPE]):
    """Writes both requant records: identity for PV, `scale` for QK^T.

    `scale` is the kernel's `1/sqrt(D)` and lands in the QK^T readout, so the
    softmax reads scores that are already scaled and no pass multiplies them.
    fp16 is what the record encodes, so the scale itself rounds once here —
    a difference from the HVX arm's fp32 multiply, and one the tolerance leg
    has to see rather than be protected from.
    """
    hmx_program_record(arena.rec_id, Scalar[HMX_DTYPE](1), Scalar[HMX_DTYPE](0))
    hmx_program_record(arena.rec_qk, scale, Scalar[HMX_DTYPE](0))


@always_inline
def hmx_attn_qk_block[
    space: AddressSpace, //, emit_hmx: Bool = HMX_EMIT_MMA
](arena: HmxAttnArena[space], plan: HmxAttnPlan):
    """`S = scale * Q K^T` for the staged block, one accumulator chain per tile.

    `d_tiles` crouton pairs per output tile and one `:after` store — the whole of
    QK^T. The scale rides the store.
    """
    hmx_attn_qk_block_at[emit_hmx=emit_hmx](arena, plan, 0)


@always_inline
def hmx_attn_qk_block_at[
    space: AddressSpace, //, emit_hmx: Bool = HMX_EMIT_MMA
](arena: HmxAttnArena[space], plan: HmxAttnPlan, half: Int):
    """`S = scale * Q K^T` out of staging half `half`, into score half `half`.

    One half index for both operand and result because the schedule never needs
    them apart: a key block is staged, multiplied and consumed under the same
    parity, and giving the two their own indices would make an off-by-one
    expressible in a second place without making anything expressible that the
    schedule wants.
    """
    for r in range(plan.br_tiles):
        for j in range(plan.bc_tiles):
            hmx_mma_chain[emit_hmx=emit_hmx](
                arena.s_tile_at(plan, half, r, j),
                arena.rec_qk,
                HmxOperands[space](
                    arena.q_run(plan, r),
                    arena.k_run_at(plan, half, j),
                    plan.d_tiles,
                ),
                HmxOperands[space](
                    arena.q_run(plan, r), arena.k_run_at(plan, half, j), 0
                ),
            )


@always_inline
def hmx_attn_pv_block[
    space: AddressSpace, //, emit_hmx: Bool = HMX_EMIT_MMA
](arena: HmxAttnArena[space], plan: HmxAttnPlan, src: Int, dst: Int):
    """`O_new = diag(corr) * O_old + sum_j P_j V_j`, in ONE accumulator chain.

    The rescale is the chain's HEAD pair and the `Bc/32` `(P, V)` pairs are its
    tail, so the online softmax's correction costs one crouton pair per output
    tile and the vector engines never touch `O`. That fusion is the reason this
    lever is worth the array at all: the alternative reads and writes the whole
    of `O` on HVX once per key block.
    """
    hmx_attn_pv_block_at[emit_hmx=emit_hmx](arena, plan, src, dst, 0)


@always_inline
def hmx_attn_pv_block_at[
    space: AddressSpace, //, emit_hmx: Bool = HMX_EMIT_MMA
](
    arena: HmxAttnArena[space],
    plan: HmxAttnPlan,
    src: Int,
    dst: Int,
    half: Int,
):
    """`O_new = diag(corr) * O_old + sum_j P_j V_j` out of staging half `half`.

    `src`/`dst` index the ACCUMULATOR ping-pong and `half` the key block's, and
    they advance on different clocks — `src` flips once per key block while
    `half` is the block's own parity — so they are separate arguments rather
    than one derived from the other.
    """
    for r in range(plan.br_tiles):
        for c in range(plan.d_tiles):
            hmx_mma_chain[emit_hmx=emit_hmx](
                arena.o_tile(plan, dst, r, c),
                arena.rec_id,
                HmxOperands[space](
                    arena.diag_tile(r), arena.o_tile(plan, src, r, c), 1
                ),
                HmxOperands[space](
                    arena.p_run_at(plan, half, r),
                    arena.v_run_at(plan, half, c),
                    plan.bc_tiles,
                ),
            )


@always_inline
def hmx_attn_normalise[
    space: AddressSpace, //, emit_hmx: Bool = HMX_EMIT_MMA
](arena: HmxAttnArena[space], plan: HmxAttnPlan, src: Int, dst: Int):
    """`O = diag(1/l) * O`, also on the array.

    The same shape as the rescale with an empty tail, which is why it is the same
    function and not a special case: the caller has already written `1/l` into
    the diagonal croutons.
    """
    for r in range(plan.br_tiles):
        for c in range(plan.d_tiles):
            hmx_mma_chain[emit_hmx=emit_hmx](
                arena.o_tile(plan, dst, r, c),
                arena.rec_id,
                HmxOperands[space](
                    arena.diag_tile(r), arena.o_tile(plan, src, r, c), 1
                ),
                HmxOperands[space](
                    arena.diag_tile(r), arena.o_tile(plan, src, r, c), 0
                ),
            )


@always_inline
def hmx_attn_scores_to_rows[
    space: AddressSpace, //
](
    arena: HmxAttnArena[space],
    plan: HmxAttnPlan,
    r: Int,
    slab_pair: Int,
    thread: Int,
):
    """Deals row tile `r`'s scores for query rows `2p` and `2p+1` into fp32 rows.

    A crouton vector holds two query rows interleaved halfword-wise, so ONE
    `vdeal` over two adjacent score croutons returns 64 whole-row scores each —
    the layout the row-wise max and `exp` want, with no gather and no DDR. The
    even query row lands in row buffer `2 * thread`, the odd one in `2*thread+1`;
    that phase is the reference's own (`flash-attn-ops.c:1147-1161`).
    """
    hmx_attn_scores_to_rows_at(arena, plan, 0, r, slab_pair, thread)


@always_inline
def hmx_attn_scores_to_rows_at[
    space: AddressSpace, //
](
    arena: HmxAttnArena[space],
    plan: HmxAttnPlan,
    half: Int,
    r: Int,
    slab_pair: Int,
    thread: Int,
):
    """`hmx_attn_scores_to_rows` out of score half `half`."""
    comptime LANES = hvx_lanes[HMX_DTYPE]()
    comptime ALIGN = align_of[Scalar[HMX_DTYPE]]()
    comptime F32_ALIGN = align_of[Scalar[DType.float32]]()
    var base = (half * plan.br_tiles + r) * plan.bc_tiles
    var j = 0
    while j + 2 <= plan.bc_tiles:
        var pair = hvx_vdeal[HVX_DEAL_SEPARATE](
            arena.s.load[LANES, alignment=ALIGN](
                Coord(base + j + 1, slab_pair, 0, 0)
            ),
            arena.s.load[LANES, alignment=ALIGN](
                Coord(base + j, slab_pair, 0, 0)
            ),
        )
        arena.rows.store[LANES, alignment=F32_ALIGN](
            Coord(2 * thread, j * HMX_TILE),
            pair.slice[LANES, offset=0]().cast[DType.float32](),
        )
        arena.rows.store[LANES, alignment=F32_ALIGN](
            Coord(2 * thread + 1, j * HMX_TILE),
            pair.slice[LANES, offset=LANES]().cast[DType.float32](),
        )
        j += 2


@always_inline
def hmx_attn_probs_from_rows[
    space: AddressSpace, //
](
    arena: HmxAttnArena[space],
    plan: HmxAttnPlan,
    r: Int,
    slab_pair: Int,
    thread: Int,
):
    """Shuffles the row buffers back into probability croutons: the seam's other
    half.

    Exactly the inverse permute of `hmx_attn_scores_to_rows`, which is what makes
    the round trip an identity on the layout and leaves only the fp32-to-fp16
    narrowing as a value change. `P` is then already in PV's activation layout,
    so nothing repacks it.
    """
    hmx_attn_probs_from_rows_at(arena, plan, 0, r, slab_pair, thread)


@always_inline
def hmx_attn_probs_from_rows_at[
    space: AddressSpace, //
](
    arena: HmxAttnArena[space],
    plan: HmxAttnPlan,
    half: Int,
    r: Int,
    slab_pair: Int,
    thread: Int,
):
    """`hmx_attn_probs_from_rows` into probability half `half`."""
    comptime LANES = hvx_lanes[HMX_DTYPE]()
    comptime ALIGN = align_of[Scalar[HMX_DTYPE]]()
    comptime F32_ALIGN = align_of[Scalar[DType.float32]]()
    var base = (half * plan.br_tiles + r) * plan.bc_tiles
    var j = 0
    while j + 2 <= plan.bc_tiles:
        var pair = hvx_vshuff[HVX_SHUFF_INTERLEAVE](
            arena.rows.load[LANES, alignment=F32_ALIGN](
                Coord(2 * thread + 1, j * HMX_TILE)
            ).cast[HMX_DTYPE](),
            arena.rows.load[LANES, alignment=F32_ALIGN](
                Coord(2 * thread, j * HMX_TILE)
            ).cast[HMX_DTYPE](),
        )
        arena.p.store[LANES, alignment=ALIGN](
            Coord(base + j, slab_pair, 0, 0),
            pair.slice[LANES, offset=0](),
        )
        arena.p.store[LANES, alignment=ALIGN](
            Coord(base + j + 1, slab_pair, 0, 0),
            pair.slice[LANES, offset=LANES](),
        )
        j += 2


# ===--------------------------------------------------------------------------===
# The feeder/array split: one step function, one barrier per step (design §4.7).
#
# Slice 3 ran a head on one thread, so the array sat idle through every pack and
# every softmax and the launch's other four threads sat idle through the whole
# head. This is the schedule that fixes both, and it is the matmul stripe's
# topology with attention's dependency chain substituted: the LAST participant
# holds the grant and does nothing but issue MMA chains, and the rest pack, run
# the softmax and read `O` back out.
#
# WHY THE SPLIT IS FORCED RATHER THAN PREFERRED: a thread with HVX enabled
# cannot have HMX enabled (`qurt_hmx.h:184`). The softmax is HVX and the packers
# are HVX since #332, so every one of them is illegal on the granted thread. The
# array's arm here therefore touches nothing but `hmx_mma_chain`.
#
# THE SCHEDULE, per query block, with `j` the key block and `&1` the ping-pong
# halves `HmxAttnArena` has carried since #341 and nothing has used until now:
#
#   t=0        PROLOGUE  feeders: pack Q, zero both O halves, reset (m, l),
#                        pack K(0) into half 0.            array: idle
#   t=1+2j     A(j)      array:   QK^T(j)  K[j&1] -> S[j&1]
#                        feeders: softmax(j-1) on S[(j-1)&1] -> P[(j-1)&1] and
#                                 diag(corr); pack V(j) into V[j&1]
#   t=2+2j     B(j)      array:   PV(j-1)  P,V[(j-1)&1] -> O
#                        feeders: pack K(j+1) into K[(j+1)&1]
#   t=3+2n     NORMPREP  feeders: write diag(1/l).         array: idle
#   t=4+2n     NORM      array:   O = diag(1/l) O.         feeders: idle
#   t=5+2n     STORE     feeders: unpack O.                array: idle
#
# EVERY HAZARD THE SKEW HAS TO CLEAR, since a step's two roles run concurrently
# and only the rendezvous after it orders them:
#
#   A(j): the array writes S[j&1] while the feeders read S[(j-1)&1] -- different
#         halves. The feeders write V[j&1], which the array does not read until
#         B(j+1), and K is untouched by both.
#   B(j): the array reads V[(j-1)&1] and P[(j-1)&1] while the feeders write
#         K[(j+1)&1]. Those are the SAME parity as the array's reads, which is
#         exactly why only K is staged here: staging V(j+1) in this step would
#         overwrite the block the array is multiplying. That is the reuse hazard
#         this slice introduces, it is invisible to a bit-identity test of a
#         single step, and `oracle_attn_reuse_hazard` in the twin is the control
#         that replays it.
#   diag: written by softmax(j-1) at A(j), read by PV(j-1) at B(j), overwritten
#         by softmax(j) at A(j+1) -- strictly sequential, so one set suffices
#         and the design's single `br_tiles` allocation stands.
#
# A row tile belongs to the same feeder in every step, which is what lets the
# online recurrence's `(m, l)` stay per-thread state without a lock: the shares
# come from `hmx_work_share` on a step-independent axis.
# ===--------------------------------------------------------------------------===


trait AttnSoftmax:
    """The exponential the online softmax uses, injected rather than imported.

    `_exp_hvx` lives beside the HVX arm in `npu_kernel.mojo` and must stay there:
    it is the arithmetic silicon has already ruled on, and moving it would put
    the shipped vector kernel's emission at risk for a refactor's convenience.
    This module cannot import it (the dependency runs the other way), and copying
    it would be the drift the file's whole split exists to prevent.

    So the schedule takes it as a parameter. The device injects `_exp_hvx`; the
    host twin injects `std.math.exp`, which is the same substitution slice 3's
    twin already makes and the frozen tolerance was measured against.
    """

    @staticmethod
    def exp_nonpos[
        width: Int
    ](x: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
        """`exp(x)` for `x <= 0`, which is the only domain the recurrence uses.
        """
        ...


@always_inline
def _attn_fold_max[
    width: Int
](x: SIMD[DType.float32, width]) -> Scalar[DType.float32]:
    """Horizontal max, as the rotate-and-combine fold `_reduce` spells it.

    `SIMD.reduce_max()` scalarizes on this triple, which is the finding
    `_reduce_max` records; the shape is copied rather than imported for the same
    dependency reason `AttnSoftmax` exists.
    """
    var reduced = x
    comptime for i in range(8, -1, -1):
        comptime if (1 << i) < width:
            reduced = max(reduced, reduced.rotate_right[1 << i]())
    return reduced[0]


@always_inline
def _attn_fold_add[
    width: Int
](x: SIMD[DType.float32, width]) -> Scalar[DType.float32]:
    """Horizontal sum by the same fold, so the summation ORDER matches the HVX
    arm's."""
    var reduced = x
    comptime for i in range(8, -1, -1):
        comptime if (1 << i) < width:
            reduced = reduced + reduced.rotate_right[1 << i]()
    return reduced[0]


@always_inline
def _attn_softmax_rows_impl[
    space: AddressSpace,
    mask_space: AddressSpace,
    //,
    SM: AttnSoftmax,
    W: Int,
    masked: Bool,
](
    arena: HmxAttnArena[space],
    plan: HmxAttnPlan,
    mask: AttnStats[mask_space],
    half: Int,
    r: Int,
    slab_pair: Int,
    slot: Int,
    q0: Int,
    k0: Int,
):
    """The online softmax over one crouton slab pair's two query rows.

    Reads the scores through the seam's `vdeal`, updates `(m, l)`, writes the
    probabilities back through the `vshuff`, and scatters `corr` into the
    diagonal crouton the array's next PV chain multiplies by. One row tile's
    state is one feeder's, so nothing here is shared across threads.

    ONE BODY WITH A COMPTIME BRANCH, NOT TWO FUNCTIONS, and that shape is
    measured rather than preferred. Lifting the recurrence into its own
    `@always_inline` callee — the obvious way to let the masked arm reuse it —
    left the dense arm's Hexagon object DIFFERENT from the base at the same
    instruction count: the same work, rescheduled, with different registers and
    `.new` forms. That is harmless arithmetic and still the wrong trade, because
    the render already ships this arm with `SKYE_HMX_ATTN=1` and a byte
    comparison against the previous object is how a regression is caught here.
    Adding a comptime-false branch to the body instead leaves the flag-on object
    byte-identical, which is the property worth the extra parameters.
    """
    comptime NEG = Scalar[DType.float32](-3.0e38)
    # SOFTMAX_PROBE sub-brackets. `slot` is the launch thread index (the
    # schedule passes `thread_idx` through), so the marks land in this
    # feeder's own buckets. Ids 4/3 are array-only labels, structurally zero
    # here — see `HMX_SM_SPLIT` in `hmx_crouton.mojo` for the reuse contract.
    comptime SM_MARKS = HMX_PHASE_PROBE and HMX_SM_SPLIT
    comptime SM_PHASE_SEAM = HMX_PHASE_IDLE
    comptime SM_PHASE_EXP = HMX_PHASE_MMA
    var bc = plan.bc_tiles * HMX_TILE
    hmx_phase_mark[SM_MARKS](slot, SM_PHASE_SEAM)
    hmx_attn_scores_to_rows_at(arena, plan, half, r, slab_pair, slot)
    comptime if masked:
        hmx_attn_mask_rows[W](arena, plan, mask, r, slab_pair, slot, q0, k0)
    hmx_phase_mark[SM_MARKS](slot, HMX_PHASE_SOFTMAX)
    for phase in range(HMX_SPLIT):
        var local = HMX_SPLIT * slab_pair + phase
        var row = r * HMX_TILE + local
        var buf = 2 * slot + phase

        var m_old = arena.stats.load[1](Coord(ATTN_STAT_M, row))[0]
        var l_old = arena.stats.load[1](Coord(ATTN_STAT_L, row))[0]

        var mx = SIMD[DType.float32, W](NEG)
        var i = 0
        while i < bc:
            mx = max(mx, arena.rows.load[W](Coord(buf, i)))
            i += W
        var m_new = _attn_fold_max(mx)
        if m_old > m_new:
            m_new = m_old

        # exp(-inf - finite) = 0 after the clamp, so the first key block rescales
        # a zeroed O by zero -- the HVX arm's own first-tile behaviour.
        var corr = SM.exp_nonpos[1](SIMD[DType.float32, 1](m_old - m_new))[0]
        var m_bcast = SIMD[DType.float32, W](m_new)
        var sum_v = SIMD[DType.float32, W](0)
        hmx_phase_mark[SM_MARKS](slot, SM_PHASE_EXP)
        i = 0
        while i < bc:
            var pv = SM.exp_nonpos[W](
                arena.rows.load[W](Coord(buf, i)) - m_bcast
            )
            sum_v += pv
            arena.rows.store[W](Coord(buf, i), pv)
            i += W
        hmx_phase_mark[SM_MARKS](slot, HMX_PHASE_SOFTMAX)

        arena.stats.store[1](
            Coord(ATTN_STAT_M, row), SIMD[DType.float32, 1](m_new)
        )
        arena.stats.store[1](
            Coord(ATTN_STAT_L, row),
            SIMD[DType.float32, 1](l_old * corr + _attn_fold_add(sum_v)),
        )
        # The rescale is a matmul against this diagonal, so `corr` reaches `O`
        # through the array rather than through a vector pass over `O`.
        hmx_attn_set_diag(arena, r, local, corr.cast[HMX_DTYPE]())
    hmx_phase_mark[SM_MARKS](slot, SM_PHASE_SEAM)
    hmx_attn_probs_from_rows_at(arena, plan, half, r, slab_pair, slot)
    hmx_phase_mark[SM_MARKS](slot, HMX_PHASE_SOFTMAX)


@always_inline
def _attn_softmax_rows_sweep[
    space: AddressSpace,
    //,
    SM: AttnSoftmax,
    W: Int,
](
    arena: HmxAttnArena[space],
    plan: HmxAttnPlan,
    half: Int,
    r: Int,
    slab_pair: Int,
    slot: Int,
    mut delta_v: SIMD[DType.float32, HMX_TILE],
    mut sum_v: SIMD[DType.float32, HMX_TILE],
):
    """`_attn_softmax_rows_impl`'s sweeps with the per-row epilogue DEFERRED.

    Everything through the probability write-back is the impl's arithmetic,
    unchanged: seam deal, running max, exp sweep, seam shuffle. What the impl
    finishes per row — the `corr` exponential, the `l` update, the diagonal
    scatter — this stashes as one lane per row (`delta_v`, `sum_v`) for
    `_attn_epi_xpose` to finish 32 rows at a time. `m` is stored here exactly
    as the impl stores it, because the NEXT key block's max sweep reads it;
    `l` is read by nothing until the epilogue, which is what makes the
    deferral legal. Dense arm only; the masked arm keeps the impl.

    Kept textually beside the impl on purpose: the two bodies must not drift,
    and the twin's bit-equality case (`attn_epi_xpose_bit_identity`) is the
    guard that they have not.
    """
    comptime NEG = Scalar[DType.float32](-3.0e38)
    var bc = plan.bc_tiles * HMX_TILE
    hmx_attn_scores_to_rows_at(arena, plan, half, r, slab_pair, slot)
    for phase in range(HMX_SPLIT):
        var local = HMX_SPLIT * slab_pair + phase
        var row = r * HMX_TILE + local
        var buf = 2 * slot + phase

        var m_old = arena.stats.load[1](Coord(ATTN_STAT_M, row))[0]

        var mx = SIMD[DType.float32, W](NEG)
        var i = 0
        while i < bc:
            mx = max(mx, arena.rows.load[W](Coord(buf, i)))
            i += W
        var m_new = _attn_fold_max(mx)
        if m_old > m_new:
            m_new = m_old

        var m_bcast = SIMD[DType.float32, W](m_new)
        var s = SIMD[DType.float32, W](0)
        i = 0
        while i < bc:
            var pv = SM.exp_nonpos[W](
                arena.rows.load[W](Coord(buf, i)) - m_bcast
            )
            s += pv
            arena.rows.store[W](Coord(buf, i), pv)
            i += W

        arena.stats.store[1](
            Coord(ATTN_STAT_M, row), SIMD[DType.float32, 1](m_new)
        )
        delta_v[local] = m_old - m_new
        sum_v[local] = _attn_fold_add(s)
    hmx_attn_probs_from_rows_at(arena, plan, half, r, slab_pair, slot)


@always_inline
def _attn_epi_xpose[
    space: AddressSpace, //, SM: AttnSoftmax
](
    arena: HmxAttnArena[space],
    r: Int,
    delta_v: SIMD[DType.float32, HMX_TILE],
    sum_v: SIMD[DType.float32, HMX_TILE],
):
    """One row tile's 32 per-row epilogues as lane-wise vector arithmetic.

    `corr = exp(m_old - m_new)` is the same polynomial `exp_nonpos[1]` runs,
    evaluated lane-wise at `HMX_TILE` width — `_exp_hvx` carries no width
    branch, so lane `w` computes bit-for-bit what row `w`'s width-1 chain
    computed, and 32 dependent Horner chains become one pipelined evaluation.
    `l_new = l * corr + sum` is the impl's own two ops, lane-wise over the
    stats row (`AttnStats` is row-major, so 32 consecutive query rows are one
    contiguous fp32 vector). The diagonal scatter keeps its per-row store —
    the destination is a crouton diagonal; only the VALUE computation is
    batched — and moves from inside each row's pass to after the tile's:
    the diagonal is read at B(j), after the rendezvous, so the array cannot
    observe the deferral.
    """
    var row0 = r * HMX_TILE
    var corr_v = SM.exp_nonpos[HMX_TILE](delta_v)
    var l_old = arena.stats.load[HMX_TILE](Coord(ATTN_STAT_L, row0))
    arena.stats.store[HMX_TILE](
        Coord(ATTN_STAT_L, row0), l_old * corr_v + sum_v
    )
    for local in range(HMX_TILE):
        hmx_attn_set_diag(arena, r, local, corr_v[local].cast[HMX_DTYPE]())


@always_inline
def hmx_attn_softmax_rows[
    space: AddressSpace, //, SM: AttnSoftmax, W: Int
](
    arena: HmxAttnArena[space],
    plan: HmxAttnPlan,
    half: Int,
    r: Int,
    slab_pair: Int,
    slot: Int,
):
    """The dense arm's seam: `mojo_mha_no_cache_f16`'s, unchanged.

    Both the device driver and the host twin call THIS function, which is what
    makes the schedule below a single description rather than two that agree
    today. `arena.stats` stands in for the mask because `masked` is False and
    every read of it is compiled away — a live view the arm never reads, rather
    than a null one that would fault if the comptime branch were ever wrong.
    """
    _attn_softmax_rows_impl[SM, W, masked=False](
        arena, plan, arena.stats, half, r, slab_pair, slot, 0, 0
    )


@always_inline
def hmx_attn_mask_rows[
    space: AddressSpace,
    mask_space: AddressSpace,
    //,
    W: Int,
](
    arena: HmxAttnArena[space],
    plan: HmxAttnPlan,
    mask: AttnStats[mask_space],
    r: Int,
    slab_pair: Int,
    slot: Int,
    q0: Int,
    k0: Int,
):
    """Adds the additive score mask into the dealt row buffers, in fp32.

    THE MASK RIDES THE SEAM RATHER THAN THE QK^T EPILOGUE, and the arithmetic
    that picks the placement is worth stating because both are expressible.

    On the array it would be an extra `(mask, I)` crouton pair at the head of
    each QK^T chain. At `Br = 512, Bc = 256, D = 128` a block is 512 QK^T pairs
    and 576 PV pairs, so 128 more pairs is +25% on QK^T and +12% on the whole
    block — levied on the ARRAY, which is the resource this lever exists to
    keep busy. It also needs the mask staged as croutons (`Br * Bc * 2` more
    bytes of arena per block), narrows it to fp16 where the host's
    large-negative fill is out of range, and — because `1/sqrt(D)` is folded
    into the QK^T requant record — would have the record scale the mask too, so
    the host would have to pre-divide it.

    Here it is one fp32 load, add and store per `W` scores: about 3 packets per
    32 scores against the 10-odd the same 32 scores already cost the
    exponential beside it. That is a real fraction of the SEAM — call it a
    quarter more vector work — but the seam runs on the feeders, concurrently
    with the array's chains, which is what the split schedule is for. Trading
    array time for feeder time is the right direction as long as the array is
    the bottleneck, and if it ever stops being one the answer is to stage the
    bias into VTCM, not to move the add onto the chains.

    The bias is `[seq_len, seq_len]` and broadcast across heads, exactly as
    `mojo_mha_masked_f16` reads it, and the key axis is its contiguous one — so
    a row of it is a straight vector load and the whole buffer is read once per
    query row per key block by every head, which is what keeps it cache-warm.

    Args:
        arena: One head's residency; `rows` already holds this pair's scores.
        plan: The block geometry.
        mask: The `[seq_len, seq_len]` fp32 additive bias.
        r: Row tile within the query block.
        slab_pair: Crouton slab pair — the even/odd query-row phase.
        slot: This feeder's row-buffer slot.
        q0: First query row of the query block, in the sequence.
        k0: First key of the key block, in the sequence.
    """
    var bc = plan.bc_tiles * HMX_TILE
    for phase in range(HMX_SPLIT):
        var row = q0 + r * HMX_TILE + HMX_SPLIT * slab_pair + phase
        var buf = 2 * slot + phase
        var i = 0
        while i < bc:
            arena.rows.store[W](
                Coord(buf, i),
                arena.rows.load[W](Coord(buf, i))
                + mask.load[W](Coord(row, k0 + i)),
            )
            i += W


@always_inline
def hmx_attn_softmax_rows_masked[
    space: AddressSpace,
    mask_space: AddressSpace,
    //,
    SM: AttnSoftmax,
    W: Int,
](
    arena: HmxAttnArena[space],
    plan: HmxAttnPlan,
    mask: AttnStats[mask_space],
    half: Int,
    r: Int,
    slab_pair: Int,
    slot: Int,
    q0: Int,
    k0: Int,
):
    """`hmx_attn_softmax_rows` with an additive score mask before the max.

    The masked arm is the dense arm plus one add, in that order: the scale is
    already in the scores (it rode the QK^T requant record), the bias goes in
    fp32, and the max, the correction and the `(m, l)` recurrence are then the
    SAME function the dense arm runs. That is what makes "the encoder's kernel
    is the render's kernel plus a mask" a checked claim — a zero mask must
    reproduce the dense arm bit for bit, which is the twin's first control and
    the obligation `_mha_flash_head_masked` already carries against its own
    dense twin.
    """
    _attn_softmax_rows_impl[SM, W, masked=True](
        arena, plan, mask, half, r, slab_pair, slot, q0, k0
    )


@fieldwise_init
struct HmxAttnSched(Copyable, Movable):
    """How many steps a launch runs, and how a step index decodes.

    Carried as counts rather than extents for the reason `HmxAttnPlan` is: every
    consumer walks blocks, and the one place a sequence length is divided into
    them is `hmx_attn_sched`.
    """

    var heads: Int
    """Attention heads the launch computes. The outer loop."""

    var q_blocks: Int
    """Query blocks per head: `seq_len / Br`."""

    var kv_blocks: Int
    """Key blocks per query block: `seq_len / Bc`."""

    @always_inline
    def usable(self) -> Bool:
        """Whether the launch has any work at all."""
        return self.heads > 0 and self.q_blocks > 0 and self.kv_blocks > 0

    @always_inline
    def steps_per_q(self) -> Int:
        """Steps one query block costs: prologue, `2(n+1)` pipelined, and the
        three that drain it."""
        return 2 * self.kv_blocks + 6

    @always_inline
    def steps(self) -> Int:
        """Steps the whole launch runs. A pure function of the shape, so every
        participant computes the same number without consulting another."""
        if not self.usable():
            return 0
        return self.heads * self.q_blocks * self.steps_per_q()


@always_inline
def hmx_attn_sched(
    plan: HmxAttnPlan, seq_len: Int, num_heads: Int
) -> HmxAttnSched:
    """The schedule for a launch, or an unusable one if the shape does not
    divide.

    Refuses through `usable()` rather than truncating, which is
    `hmx_attn_plan`'s rule and for the same reason: a partial block reads
    plausible.
    """
    if not hmx_attn_accepts(plan, seq_len) or num_heads <= 0:
        return HmxAttnSched(0, 0, 0)
    return HmxAttnSched(
        num_heads,
        seq_len // (plan.br_tiles * HMX_TILE),
        seq_len // (plan.bc_tiles * HMX_TILE),
    )


@always_inline
def hmx_attn_arrivals(
    participants: Int, feeders: Int, accepts: Bool, granted: Bool, steps: Int
) -> Int:
    """Rendezvous one participant makes in a whole dispatch.

    Has no thread-index parameter, and that is the property rather than an
    omission: order 084 wedged because a refusal some threads took and others did
    not left the takers waiting on arrivals that were never coming. Every input
    is a value every participant reads identically — the count is one
    release-stored word, `accepts` is arithmetic on the kernel's scalars, and
    `granted` is broadcast by the rendezvous that follows the lock.
    """
    if participants < 2 or feeders < 1 or not accepts:
        return 0
    if not granted:
        return 1
    return 1 + steps


@always_inline
def _attn_step_impl[
    space: AddressSpace,
    mask_space: AddressSpace,
    //,
    SM: AttnSoftmax,
    W: Int,
    emit_hmx: Bool = HMX_EMIT_MMA,
    masked: Bool = False,
    center: Bool = False,
    epi_xpose: Bool = HMX_EPI_XPOSE,
](
    Q: HmxRows2D,
    K: HmxRows2D,
    V: HmxRows2D,
    O: HmxRows2D,
    mask: AttnStats[mask_space],
    arena: HmxAttnArena[space],
    plan: HmxAttnPlan,
    sched: HmxAttnSched,
    step: Int,
    thread_idx: Int,
    feeders: Int,
    reuse_hazard: Bool = False,
):
    """One thread's work in one step of the whole launch.

    THE WHOLE SCHEDULE IS HERE, which is the point: the device driver is a loop
    that calls this `sched.steps()` times with a barrier after each, and the host
    twin is the same loop calling it for every `thread_idx` before advancing.
    Neither driver knows what a step contains, so the two cannot drift, and a
    sequential per-thread sweep IS a legal interleaving — which is what makes the
    twin a real test of the schedule rather than of one ordering of it.

    Parameters:
        SM: The exponential the softmax uses.
        W: The vector width the softmax reduces at. `Bc % W == 0` is the
            caller's to check.
        emit_hmx: Whether the chains issue HMX or model it.
        masked: Whether the softmax adds `mask` before its running max. The
            dense arm instantiates this False and `mask` is then dead at
            compile time, which is why the two arms are one schedule: the
            masked encoder and the dense render cannot drift on the ping-pong,
            the staging skew or the drain, because there is only one of each.
        center: Whether the K pack subtracts the head's reference key first --
            the order 134 range fix (`hmx_attn_pack_k_share_centered`). Separate
            from `masked` because the two are independent: the range fix is
            about what the fp16 narrowing is handed and the mask is about which
            scores exist, and the dense arm needs the first without the second.
        epi_xpose: Whether the dense arm batches the per-row softmax epilogue
            across the row tile -- one `exp_nonpos[HMX_TILE]` vector where 32
            serial width-1 Horner chains ran, one contiguous load/store pair
            for the `l` recurrence, the diagonal scattered with batched values.
            The masked arm keeps the per-row epilogue regardless. The deferral
            is legal because `m` is stored per-row exactly as before, nothing
            reads `l` until the epilogue, and the diagonal is read at B(j)
            after the rendezvous -- the array cannot observe it.

    Args:
        Q: The `[seq_len, num_heads * D]` queries.
        K: The keys, same layout.
        V: The values, same layout.
        O: The output, same layout.
        mask: The `[seq_len, seq_len]` fp32 additive bias, read only when
            `masked`. The dense arm passes `arena.stats` — a live view it never
            reads — rather than a null one, so no arm carries a pointer that
            would fault if the comptime branch were ever wrong.
        arena: One head's residency. Reused by every head in turn.
        plan: The block geometry every thread agreed on.
        sched: The step count and its decode.
        step: Which of `sched.steps()` steps this is.
        thread_idx: This thread's launch index.
        feeders: Threads that pack and run the softmax. `thread_idx >= feeders`
            is the one thread holding the HMX grant, and it only multiplies.
        reuse_hazard: Stages the NEXT key block into the half the array is
            reading, which is the off-by-one this schedule's double buffer
            exists to prevent. Only the twin's falsifying control sets it.
    """
    comptime SLAB_PAIRS = HMX_TILE // HMX_SPLIT
    var d = plan.d_tiles * HMX_TILE
    var br = plan.br_tiles * HMX_TILE
    var bc = plan.bc_tiles * HMX_TILE
    var n = sched.kv_blocks

    var per_q = sched.steps_per_q()
    var q_index = step // per_q
    var t = step - q_index * per_q
    var head = q_index // sched.q_blocks
    var q_block = q_index - head * sched.q_blocks
    var head_off = head * d
    var q0 = q_block * br

    # Which of the six kinds of step this is. `j` is the key block the A/B pair
    # belongs to and is meaningless outside it.
    var j = (t - 1) // 2
    var is_a = t >= 1 and t <= 2 * n + 2 and (t - 1) % 2 == 0
    var is_b = t >= 1 and t <= 2 * n + 2 and (t - 1) % 2 == 1

    if thread_idx >= feeders:
        # The array's two outcomes are marked apart on purpose: `IDLE` is the
        # schedule giving it nothing to do and `MMA` is it working, and the
        # stall diagnosis is exactly the question of which of those a share is.
        if is_a and j < n:
            hmx_phase_mark(thread_idx, HMX_PHASE_MMA)
            hmx_attn_qk_block_at[emit_hmx=emit_hmx](arena, plan, j & 1)
            hmx_phase_mark(thread_idx, HMX_PHASE_OTHER)
            return
        if is_b and j >= 1:
            var src = (j - 1) & 1
            hmx_phase_mark(thread_idx, HMX_PHASE_MMA)
            hmx_attn_pv_block_at[emit_hmx=emit_hmx](
                arena, plan, src, 1 - src, (j - 1) & 1
            )
            hmx_phase_mark(thread_idx, HMX_PHASE_OTHER)
            return
        if t == 2 * n + 4:
            var src = n & 1
            hmx_phase_mark(thread_idx, HMX_PHASE_MMA)
            hmx_attn_normalise[emit_hmx=emit_hmx](arena, plan, src, 1 - src)
            hmx_phase_mark(thread_idx, HMX_PHASE_OTHER)
            return
        hmx_phase_mark(thread_idx, HMX_PHASE_IDLE)
        return

    # A feeder's shares. Row tiles are the softmax's axis and the store's, key
    # tiles the K pack's, DV tiles the V pack's — each chosen so two feeders'
    # DESTINATIONS are disjoint croutons rather than disjoint sources.
    var r0 = hmx_work_share(plan.br_tiles, thread_idx, feeders)
    var r1 = hmx_work_share(plan.br_tiles, thread_idx + 1, feeders)

    if t == 0:
        hmx_phase_mark(thread_idx, HMX_PHASE_PACK_ACT)
        hmx_attn_pack_q_share(arena, plan, Q, q0, head_off, r0, r1)
        # Both halves, because the first key block's rescale reads the one it is
        # not writing and an uninitialised operand would be a NaN times zero.
        for r in range(r0, r1):
            for c in range(plan.d_tiles):
                hmx_attn_zero_panel(arena.o_tile(plan, 0, r, c), 1)
                hmx_attn_zero_panel(arena.o_tile(plan, 1, r, c), 1)
        for i in range(r0 * HMX_TILE, r1 * HMX_TILE):
            arena.stats.store[1](
                Coord(ATTN_STAT_M, i), SIMD[DType.float32, 1](-3.0e38)
            )
            arena.stats.store[1](
                Coord(ATTN_STAT_L, i), SIMD[DType.float32, 1](0)
            )
        hmx_phase_mark(thread_idx, HMX_PHASE_STAGE_WGT)
        var j0 = hmx_work_share(plan.bc_tiles, thread_idx, feeders)
        var j1 = hmx_work_share(plan.bc_tiles, thread_idx + 1, feeders)
        comptime if center:
            hmx_attn_pack_k_share_centered(
                arena, plan, K, 0, head_off, 0, j0, j1, 0
            )
        else:
            hmx_attn_pack_k_share(arena, plan, K, 0, head_off, 0, j0, j1)
        hmx_phase_mark(thread_idx, HMX_PHASE_OTHER)
        return

    if is_a:
        if j >= 1:
            hmx_phase_mark(thread_idx, HMX_PHASE_SOFTMAX)
            for r in range(r0, r1):
                comptime if epi_xpose and not masked:
                    var delta_v = SIMD[DType.float32, HMX_TILE](0)
                    var sum_v = SIMD[DType.float32, HMX_TILE](0)
                    for p in range(SLAB_PAIRS):
                        _attn_softmax_rows_sweep[SM, W](
                            arena,
                            plan,
                            (j - 1) & 1,
                            r,
                            p,
                            thread_idx,
                            delta_v,
                            sum_v,
                        )
                    _attn_epi_xpose[SM](arena, r, delta_v, sum_v)
                else:
                    for p in range(SLAB_PAIRS):
                        comptime if masked:
                            hmx_attn_softmax_rows_masked[SM, W](
                                arena,
                                plan,
                                mask,
                                (j - 1) & 1,
                                r,
                                p,
                                thread_idx,
                                q0,
                                (j - 1) * bc,
                            )
                        else:
                            hmx_attn_softmax_rows[SM, W](
                                arena, plan, (j - 1) & 1, r, p, thread_idx
                            )
            hmx_phase_mark(thread_idx, HMX_PHASE_OTHER)
        if j < n:
            hmx_phase_mark(thread_idx, HMX_PHASE_PACK_ACT)
            # The two arms share by DIFFERENT axes because the pair step needs
            # two adjacent DV tiles in one feeder's hands; the predicate is the
            # tested contract (`hmx_pack_vec_mha_selects`), so the routing
            # cannot drift from the twin's truth table.
            comptime if hmx_pack_vec_mha_selects[
                HMX_PACK_VEC, HMX_PACK_VEC_MHA
            ]():
                var vj0 = hmx_work_share(plan.bc_tiles, thread_idx, feeders)
                var vj1 = hmx_work_share(plan.bc_tiles, thread_idx + 1, feeders)
                hmx_attn_pack_v_share_keyed(
                    arena, plan, V, j * bc, head_off, j & 1, vj0, vj1
                )
            else:
                var c0 = hmx_work_share(plan.d_tiles, thread_idx, feeders)
                var c1 = hmx_work_share(plan.d_tiles, thread_idx + 1, feeders)
                hmx_attn_pack_v_share(
                    arena, plan, V, j * bc, head_off, j & 1, c0, c1
                )
            hmx_phase_mark(thread_idx, HMX_PHASE_OTHER)
        return

    if is_b:
        if j + 1 < n:
            hmx_phase_mark(thread_idx, HMX_PHASE_STAGE_WGT)
            var j0 = hmx_work_share(plan.bc_tiles, thread_idx, feeders)
            var j1 = hmx_work_share(plan.bc_tiles, thread_idx + 1, feeders)
            # `(j + 1) & 1` is the half the array is NOT reading this step. The
            # control stages into `j & 1` instead, which is the block the array
            # is multiplying right now.
            var half = (j & 1) if reuse_hazard else ((j + 1) & 1)
            # `kbar_row0` is 0, not `(j + 1) * bc`: the reference key is the
            # HEAD's first, so every block of a head is shifted by the same
            # per-row constant and the online recurrence stays invariant.
            comptime if center:
                hmx_attn_pack_k_share_centered(
                    arena, plan, K, (j + 1) * bc, head_off, half, j0, j1, 0
                )
            else:
                hmx_attn_pack_k_share(
                    arena, plan, K, (j + 1) * bc, head_off, half, j0, j1
                )
            hmx_phase_mark(thread_idx, HMX_PHASE_OTHER)
        return

    if t == 2 * n + 3:
        hmx_phase_mark(thread_idx, HMX_PHASE_SOFTMAX)
        for r in range(r0, r1):
            for local in range(HMX_TILE):
                var l = arena.stats.load[1](
                    Coord(ATTN_STAT_L, r * HMX_TILE + local)
                )[0]
                hmx_attn_set_diag(
                    arena,
                    r,
                    local,
                    (Scalar[DType.float32](1) / l).cast[HMX_DTYPE](),
                )
        hmx_phase_mark(thread_idx, HMX_PHASE_OTHER)
        return

    if t == 2 * n + 5:
        hmx_phase_mark(thread_idx, HMX_PHASE_UNPACK)
        hmx_attn_store_o_share(
            arena, plan, O, q0, head_off, 1 - (n & 1), r0, r1
        )
        hmx_phase_mark(thread_idx, HMX_PHASE_OTHER)
        return

    hmx_phase_mark(thread_idx, HMX_PHASE_IDLE)


@always_inline
def hmx_attn_step[
    space: AddressSpace,
    //,
    SM: AttnSoftmax,
    W: Int,
    emit_hmx: Bool = HMX_EMIT_MMA,
    epi_xpose: Bool = HMX_EPI_XPOSE,
](
    Q: HmxRows2D,
    K: HmxRows2D,
    V: HmxRows2D,
    O: HmxRows2D,
    arena: HmxAttnArena[space],
    plan: HmxAttnPlan,
    sched: HmxAttnSched,
    step: Int,
    thread_idx: Int,
    feeders: Int,
    reuse_hazard: Bool = False,
):
    """The dense arm's step: `mojo_mha_no_cache_f16`'s, unchanged.

    `arena.stats` stands in for the mask because `masked` is False and every
    read of it is compiled away — the signature the render's driver already
    calls is preserved exactly, which is what keeps this slice's flag-off object
    byte-identical.
    """
    _attn_step_impl[
        SM, W, emit_hmx=emit_hmx, masked=False, center=False,
        epi_xpose=epi_xpose,
    ](
        Q,
        K,
        V,
        O,
        arena.stats,
        arena,
        plan,
        sched,
        step,
        thread_idx,
        feeders,
        reuse_hazard,
    )


@always_inline
def hmx_attn_step_masked[
    space: AddressSpace,
    mask_space: AddressSpace,
    //,
    SM: AttnSoftmax,
    W: Int,
    emit_hmx: Bool = HMX_EMIT_MMA,
    center: Bool = True,
](
    Q: HmxRows2D,
    K: HmxRows2D,
    V: HmxRows2D,
    O: HmxRows2D,
    mask: AttnStats[mask_space],
    arena: HmxAttnArena[space],
    plan: HmxAttnPlan,
    sched: HmxAttnSched,
    step: Int,
    thread_idx: Int,
    feeders: Int,
    reuse_hazard: Bool = False,
):
    """The masked arm's step: the Qwen3 text encoder's attention on the array.

    A SIBLING RATHER THAN A PARAMETER ON `hmx_attn_step` because the dense arm's
    only caller is the shipped device driver, and widening its signature would
    edit `npu_kernel.mojo` — which the HVX mha family is being extracted out of
    concurrently. The two share `_attn_step_impl`, so this costs a wrapper
    rather than a second schedule.

    The encoder's shape is accepted by the same gate the render's is: `seq` 512
    divides `Br` 512 and `Bc` 256, so `hmx_attn_accepts` admits it with one
    query block and two key blocks and no ragged tail.

    Parameters:
        SM: The exponential the softmax uses.
        W: The vector width the softmax reduces at.
        emit_hmx: Whether the chains issue HMX or model it.
        center: Whether the K pack subtracts the head's reference key first --
            the order 134 range fix, described at
            `hmx_attn_pack_k_share_centered`. ON here and OFF on the dense arm,
            which is not a judgement that the dense arm does not need it: it
            measurably does (`check_attn_dense_envelope`). It is that turning
            the render's bytes over costs a MEF re-anchor and a Tier B re-run,
            so that is its own gated step rather than a passenger on this one.
    """
    _attn_step_impl[SM, W, emit_hmx=emit_hmx, masked=True, center=center](
        Q,
        K,
        V,
        O,
        mask,
        arena,
        plan,
        sched,
        step,
        thread_idx,
        feeders,
        reuse_hazard,
    )


# ===--------------------------------------------------------------------------===
# The accumulator-width probe (design §5, open question 1).
#
# `_hmx_mma_model` assumes the array accumulates in fp32 and its own docstring
# says the width is UNPINNED. It is the largest numerics unknown in the flash
# loop, because if the accumulator is fp16 then `Bc` becomes a numerics knob and
# every tolerance below has to be set against that instead of against two
# readout roundings per key block.
#
# Two staged panels answer it, and neither is a guess about the hardware: each
# has a wide sum that differs from a narrow one by far more than an fp16 ULP, and
# they differ from each other in WHERE the narrowing would have to happen.
#
#   A: one product of 2048, then 127 products of 1.
#      wide -> 2175, and fp16 rounds that to 2176.
#      fp16 after every product -> 2048 forever (2049 is not representable).
#      fp16 after every 32-product pass -> 2176, same as wide.
#   B: one product of 2048, then 127 products of 3/128.
#      wide -> 2050.98, which fp16 rounds to 2050.
#      fp16 after every product -> 2048.
#      fp16 after every pass -> 2048, because a pass adds under one ULP.
#
# So the PAIR separates three behaviours a single probe cannot: (2176, 2050) is a
# wide accumulator, (2176, 2048) is per-pass, (2048, 2048) is per-product. The
# device reports the readouts and the three predictions beside them, computed
# here rather than on the host, so the verdict does not depend on a host and a
# device agreeing about fp16 rounding as well.
# ===--------------------------------------------------------------------------===


comptime ATTN_PROBE_TILES = 4
"""Crouton tiles the probe's reduction spans: `K = 128`, the render's head dim.
"""

comptime ATTN_PROBE_BIG = Scalar[HMX_DTYPE](64)
"""Activation of the probe's first product. `64 * 32 = 2048` is where fp16's ULP
becomes 2, which is what makes a `+1` invisible to a narrow accumulator."""

comptime ATTN_PROBE_BIG_W = Scalar[HMX_DTYPE](32)
"""Weight of the probe's first product."""


@fieldwise_init
struct HmxAttnProbe(Copyable, Movable):
    """One accumulator probe's readout beside the three predictions.

    All four are fp16 because the readout is: comparing a device fp16 against a
    host fp32 would fold the question being asked into the comparison.
    """

    var readout: Scalar[HMX_DTYPE]
    """What the array (or the model) actually wrote."""

    var wide: Scalar[HMX_DTYPE]
    """Prediction if the accumulator is fp32 and only the store rounds."""

    var per_product: Scalar[HMX_DTYPE]
    """Prediction if every MAC rounds to fp16."""

    var per_pass: Scalar[HMX_DTYPE]
    """Prediction if each 32-product crouton pass rounds to fp16."""


@always_inline
def _attn_probe_small(variant: Int) -> Scalar[HMX_DTYPE]:
    """Returns the activation of the probe's 127 small products.

    Variant 0 makes a pass contribute far more than an ULP (so per-pass tracks
    wide); variant 1 makes it contribute less than one (so per-pass tracks
    per-product). `3/128` is exact in fp16 and its product with `1/8` is too, so
    neither prediction carries a rounding this probe did not intend.
    """
    if variant == 0:
        return Scalar[HMX_DTYPE](1)
    return Scalar[HMX_DTYPE](0.1875)


@always_inline
def _attn_probe_small_w(variant: Int) -> Scalar[HMX_DTYPE]:
    """The weight of the probe's 127 small products."""
    if variant == 0:
        return Scalar[HMX_DTYPE](1)
    return Scalar[HMX_DTYPE](0.125)


@always_inline
def hmx_attn_probe_fits(plan: HmxAttnPlan) -> Bool:
    """Whether this plan's arena has room for the probe's borrowed panels.

    The probe does not carve VTCM of its own — it runs before any attention work,
    so the `q`, `k` and `o` regions are free — but "free" is not "large enough",
    and a `ATTN_PROBE_TILES`-tile subpanel of a one-tile region reads and writes
    the region after it. That overrun is silent and its readout looks like a
    number, so it is a PRECONDITION with a name rather than a comment.
    """
    if not plan.usable():
        return False
    return (
        plan.br_tiles * plan.d_tiles >= ATTN_PROBE_TILES
        and 2 * plan.bc_tiles * plan.d_tiles >= ATTN_PROBE_TILES
    )


@always_inline
def hmx_attn_acc_probe[
    space: AddressSpace, //, emit_hmx: Bool = HMX_EMIT_MMA
](arena: HmxAttnArena[space], plan: HmxAttnPlan, variant: Int) -> HmxAttnProbe:
    """Stages one probe panel, runs it through the array, predicts three answers.

    Borrows the arena's `q`, `k` and `o` regions, which requires
    `hmx_attn_probe_fits(plan)` — a plan too small for the probe returns an
    all-zero result rather than reading past a region, so the caller sees
    "never ran" instead of a plausible number beside three real predictions.

    Reads back output element `(0, 0)` only: every other element of the tile is a
    product of zeros, which is itself a check that the staging wrote where it
    meant to.
    """
    if not hmx_attn_probe_fits(plan):
        return HmxAttnProbe(
            readout=Scalar[HMX_DTYPE](0),
            wide=Scalar[HMX_DTYPE](0),
            per_product=Scalar[HMX_DTYPE](0),
            per_pass=Scalar[HMX_DTYPE](0),
        )
    comptime ALIGN = align_of[Scalar[HMX_DTYPE]]()
    var act = hmx_subpanel(arena.q, 0, ATTN_PROBE_TILES)
    var wgt = hmx_subpanel(arena.k, 0, ATTN_PROBE_TILES)
    var dst = hmx_subpanel(arena.o, 0, 1)

    hmx_attn_zero_panel(act, ATTN_PROBE_TILES)
    hmx_attn_zero_panel(wgt, ATTN_PROBE_TILES)

    var small_a = _attn_probe_small(variant)
    var small_b = _attn_probe_small_w(variant)
    for d in range(ATTN_PROBE_TILES * HMX_TILE):
        var t = d // HMX_TILE
        var slab = d % HMX_TILE
        var a = small_a
        var b = small_b
        if d == 0:
            a = ATTN_PROBE_BIG
            b = ATTN_PROBE_BIG_W
        # Activation slab 0, lane `d`: output row 0's operand.
        act.store[1, alignment=ALIGN](Coord(t, 0, slab, 0), a)
        # Weight slab `d`, lane 0: output column 0's operand.
        wgt.store[1, alignment=ALIGN](
            Coord(t, slab // HMX_SPLIT, 0, slab % HMX_SPLIT), b
        )

    hmx_mma_chain[emit_hmx=emit_hmx](
        dst,
        arena.rec_id,
        HmxOperands[space](act, wgt, ATTN_PROBE_TILES),
        HmxOperands[space](act, wgt, 0),
    )

    var wide = Scalar[DType.float32](0)
    var per_product = Scalar[HMX_DTYPE](0)
    var per_pass = Scalar[HMX_DTYPE](0)
    var pass_acc = Scalar[DType.float32](0)
    for d in range(ATTN_PROBE_TILES * HMX_TILE):
        var a = small_a
        var b = small_b
        if d == 0:
            a = ATTN_PROBE_BIG
            b = ATTN_PROBE_BIG_W
        var product = a.cast[DType.float32]() * b.cast[DType.float32]()
        wide += product
        per_product = (per_product.cast[DType.float32]() + product).cast[
            HMX_DTYPE
        ]()
        pass_acc += product
        if (d % HMX_TILE) == HMX_TILE - 1:
            per_pass = (per_pass.cast[DType.float32]() + pass_acc).cast[
                HMX_DTYPE
            ]()
            pass_acc = Scalar[DType.float32](0)

    return HmxAttnProbe(
        readout=dst.load[1, alignment=ALIGN](Coord(0, 0, 0, 0)),
        wide=wide.cast[HMX_DTYPE](),
        per_product=per_product,
        per_pass=per_pass,
    )
