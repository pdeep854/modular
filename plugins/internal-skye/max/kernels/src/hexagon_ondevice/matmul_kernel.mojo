"""Host-testable register-blocked HVX matmul core: C[M, N] = A[M, K] @ B[N, K]^T.

It also serves stride-1 CONVOLUTION, and by design not by coincidence. The nest
below takes its A operand from an `MMAReader` rather than from a tile, so the
plain matmul (`matmul_bt`, `MMDenseReader`) and the fused online im2col
(`conv_bt`, `MMIm2colReader`) are the same kernel differing in one thing: where
an A vector comes from. That is the loader-swap seam the M5 conv2d and the MI355
`TileLoaderLDS` / `TileLoaderLDSIm2col` pair use (KB
`kernels/apple-conv2d-im2col`, `new-primitives/amd-tile-io-expert-objects`), and
the reason it is worth having here is that the fused arm never materialises the
`[M, K]` im2col matrix — where both existing conv paths on this port build it,
one on the host and one into a device scratch that at 1024x1024 exceeds the
transport's 2 GiB single-allocation cap. The dense arm's machine code is
unchanged by the seam, byte for byte
(`.agentwork/flux-npu/m46-fused-im2col/gate.sh`).

The batched/tiled extension of the one-output-row `gemv` in `npu_kernel.mojo`.
`gemv` computes one output row per dispatch (`C[N] = A[K] @ B[N, K]^T`), so a
DiT denoise loop at full-Klein M needs O(M) dispatches per projection — hundreds
of thousands over the loop, swamping the compute roofline on dispatch overhead.
This computes MANY output rows per dispatch: one launch produces the whole
`[M, N]` output, dropping the dispatch count from O(M) to O(1) per projection.

Register blocking (`MM_MR` A-rows x `MM_NR` B-rows) is the arithmetic-intensity
half: each loaded activation/weight HVX vector is reused across the tile, so
`MM_MR + MM_NR` vector loads feed `MM_MR * MM_NR` multiply-accumulates. A B row
(a weight row) loaded once is applied to every A row in the tile — the weight
read, which dominates a decode-shaped matmul, is amortized over `MM_MR` rows.
Cache blocking (`MM_MC_BYTES` of resident activation, an M-tile outside the N
sweep) is the other half: it bounds the activation rows one N sweep touches, so
a row crosses DRAM once per M-tile rather than once per N-pair.

`B` is stored N x K (b_transposed), the same contract `gemv` and the HMX matmul
path route on; `K` is a multiple of 64 (the HVX fp16 vector width). Each output
`C[m, n]` is the fp32 dot product of A row `m` with B row `n`, accumulated 64
lanes at a time and folded to one scalar — so the store to `C` is SCALAR, never
a masked vector store, and the N%64 tail-store fault class cannot arise here.

This file is the target-independent core the host test drives (`mm_fmac_host` is
the portable widen-multiply-add). The skel `npu_kernel.mojo` carries the same
loop nest with the device MAC `_fmac_sf_hf` (`V6_vmpy_sf_hf_acc`) and the
thread-split/witness wrapper `mojo_matmul_f16`. The two are kept in step by hand:
a change to the nest here is a change there until the kernel can be shared, and
that obligation is on the arithmetic rather than the spelling — the skel still
takes pointers and extents, and adopts the typed signature below at the next
re-sign window, where `mojo_matmul_f16` becomes the seam that builds the three
tiles. Only the MAC line differs — the device instruction lands its 64 fp32
lanes interleaved, which the lane-order-independent `mm_reduce_add` folds to the
same dot product (a reduction tolerates it; a non-reduced widen-store would
not).

Addressing is the tile model's, and so is the interface: EVERY function here
takes `MMTile` operands, so `M`, `N` and `K` are not parameters at all — they
are extents of the three layouts, read back where a bound is needed. The
transposed-weight contract is in `B`'s own extents rather than in a comment, a
caller cannot hand a `K` that disagrees with the stride its own operand was
built with, and `(m0 + mr) * K` / `(m0 + mr) * N` appear nowhere. The layouts
are built one line inside the C-ABI boundary ABOVE this file (`mojo_matmul_f16`
on device, the host test's own seam here), which is where raw pointers and
extents stop. The translation is KB
`kernels/tile-model-porting-a-pointer-kernel-to-tiletensor`.

What it costs is measured, not assumed, by
`plugins/internal-skye/utils/kernel_asm_compare.sh`: entered from an identical C
boundary, the whole nest is instruction-for-instruction the same as the
pointer-signature arm (2120 insns / 1393 HVX ops / 6 vector spills / 1256 VLIW
packets), so moving the seam outward is free — a layout the caller builds and
one the callee builds compile to the same address arithmetic. Establishing that
needs a probe, because this file declares no `@export` and every function is
`@always_inline`, so it emits no symbols of its own — see the PR for the C-ABI
shell the comparison runs through.
"""
from layout.coord import Coord
from layout.tile_layout import row_major
from layout.tile_tensor import TileTensor

from std.memory import AddressSpace
from std.sys import align_of, get_defined_bool, size_of
from std.utils import StaticTuple

from .hmx_conv_act import (
    HmxConvGeom,
    hmx_conv_pack_act,
)
from .hmx_crouton import (
    HMX_DTYPE,
    HMX_EMIT_MMA,
    HMX_ACT_XBLOCK_PF,
    HMX_FOLD_TRAILING,
    HMX_PHASE_IDLE,
    HMX_PHASE_MMA,
    HMX_PHASE_OTHER,
    HMX_PHASE_PACK_ACT,
    HMX_PHASE_STAGE_WGT,
    HMX_PHASE_UNPACK,
    HMX_PHASE_WGT_AWAIT,
    HMX_MMA_SPLIT,
    HMX_TILE,
    HMX_UNPACK_ARM_NATIVE,
    HMX_UNPACK_ARM_SCALAR_OFF,
    HMX_WEIGHTS_NATIVE,
    HMX_WGT_DEFER_AWAIT,
    HMX_WGT_RING,
    HMX_XFER_DMA,
    CroutonPanelOf,
    HmxScratch,
    HmxStripeArena,
    hmx_copy_croutons,
    hmx_k_tiles,
    hmx_mma_panel,
    hmx_mma_panel_split,
    hmx_native_row_panel,
    hmx_native_weight_panel,
    hmx_pack_panel,
    hmx_phase_mark,
    hmx_prefetch_rows,
    hmx_stream_weight_await,
    hmx_stream_weight_submit,
    hmx_stripe_act_slots,
    hmx_subpanel,
    hmx_unpack_panel,
)
from .hmx_int8 import (
    HMX_INT8,
    HMX_INT8_DTYPE,
    HMX_RECORD_LATE,
    hmx_int8_load_record,
    hmx_int8_mma_panel,
    hmx_int8_pack_act,
    hmx_int8_read_qp,
    hmx_int8_unpack_out,
    hmx_int8_weight_panel,
)

comptime MM_DTYPE = DType.float16
"""The matmul I/O element type; the whole path is fp16, as `gemv` is."""

comptime MM_ACCUM = DType.float32
"""The dot-product accumulator; an fp16 accumulator loses a long-K reduction."""

comptime MM_W = 128 // size_of[MM_DTYPE]()
"""HVX lanes per vector op: 64 fp16 elements in a 128-byte vector.

Derived from the register geometry deliberately, never from `simd_width_of`.
That query is now correct on this target (#114/#146 fixed the descriptor), but
this same source is compiled for the HOST by `test_matmul_kernel.mojo`, where it
correctly returns 8 — so a derived width would test an 8-lane kernel and ship a
64-lane one, with every numerical check still passing. KB
`kernels/tiletensor-vector-width-state-it-explicitly`. `matmul_bt_stripe` asserts
the tie."""

comptime MM_MR = 4
"""Register-block extent over A-rows (M). Each of `MM_MR * MM_NR` fp32
accumulators is a 64-lane vector pair (2 HVX registers); with `MM_MR` A-vectors
and `MM_NR` B-vectors also live, the k-loop working set is
`2*MM_MR*MM_NR + MM_MR + MM_NR` vectors, kept under the 32 HVX vector registers.
`MM_MR > MM_NR` favors weight (B) reuse — each B row feeds `MM_MR` A rows."""

comptime MM_NR = 2
"""Register-block extent over B-rows (N). See `MM_MR`."""

comptime MM_KU = 2
"""K-steps whose operand loads are issued before any of them is consumed.

The device k-loop costs ~90 cycles per k-step against a 33-to-39 packet static
body, so 51-57 cycles of it are stall. An HVX vector is exactly one cache line,
so a k-step has no spatial reuse -- each of its `MR + NR` loads is a separate
line -- and the only lever on that stall is how many lines are in flight.
Unrolling puts `MM_KU` steps' worth of misses in flight for one latency.

The k-loop working set is `2*MR*NR + (MR + NR) * MM_KU` vectors: 28 of the 32
HVX vector registers at KU=2, 34 at KU=3, which would spill. Two is the largest
that fits. One compiles back to the un-unrolled nest.

Kept in step with `MATMUL_KU` in npu_kernel.mojo, as the rest of this nest is:
the unroll cannot change a result, and this twin is what proves that."""

comptime MM_MC_BYTES = 512 * 1024
"""Activation bytes held resident across one N sweep — the M-tile budget.

Register blocking above reuses a loaded vector inside a tile; this is the cache
half. Without it the N-outer/M-inner nest re-sweeps the whole `M x K` activation
panel once per N-pair, so every MAC costs a byte of activation traffic. An M-tile
outside the N sweep drops that to one pass per tile.

The one empirical knob: too small and the weight stripe is re-read once per tiny
tile, too large and the panel is evicted before the N sweep reuses it. 512 KiB is
the measured knee of an on-device sweep of the full DiT step (534 s at 256 KiB,
499 s here, 704 s at 1 MiB, 1203 s untiled), so it is the effective L2 working
set for four compute threads plus a weight and an output stream — not the L2
capacity, which v81 does not publish. See BATCHED_MATMUL_STATUS.md."""

comptime MM_PATH_HMX = get_defined_bool["SKYE_HMX_MATMUL", False]()
"""Whether the launch routes its stripes to the matrix array instead of HVX.

OFF, and the default is the whole point: every line below `matmul_bt_stripe` is
untouched by the HMX work, so the current device binary's machine code is
unchanged and the next re-sign is provably a no-op for this kernel.

This is a knob the SKEL reads, not one this file acts on, and that placement is
forced rather than stylistic — see `matmul_bt_stripe_hmx`. Turning it on is a
device-lane decision (`build.sh HMX_MATMUL=1`, which also sets
`SKYE_HMX_EMIT_MMA` and `+hmx`): the array has never executed a matmul, only a
single proven-exact 32x32 tile."""


comptime MMTile = TileTensor[
    MM_DTYPE, type_of(row_major(Coord(0, 0))), origin=MutUnsafeAnyOrigin
]
"""The operand type of every function below: a 2-D row-major fp16 view.

The caller builds three of these at the C-ABI boundary and hands them down —
`C` as `(M, N)`, `A` as `(M, K)`, `B` as `(N, K)`:

```mojo
var Ct = MMTile(ptr=C, layout=row_major(Coord(M, N)))
var At = MMTile(ptr=A, layout=row_major(Coord(M, K)))
var Bt = MMTile(ptr=B, layout=row_major(Coord(N, K)))
matmul_bt_stripe[MR, NR](Ct, At, Bt, n_start, n_end, mc_bytes)
```

Building them there rather than per call is the resolve-once rule (KB
`kernels/tiletensor-resolve-the-base-once-not-per-access`, where re-resolving
costs 4x the vector work and forces spills), now applied one level further out:
one construction per launch serves every stripe and every microtile under it.

All three extents are dynamic, so the layout carries runtime values; what it
makes compile-time is the *rank* and the row-major relationship, which is what
lets a row be named by its index instead of by `index * stride`. The three
tensors do not share a stride — `A` and `B` step by `K`, `C` by `N` — and that
is precisely the arithmetic this type stops the author from restating. Because
one type serves all three roles, which extent is which is a statement the
caller's `row_major` makes and this file's signatures document; what the type
system does enforce is that a row stride is never re-derived below the seam.

This is the seam type for an HMX-backed body too: a crouton-packed path
(KB `kernels/hmx-crouton-tile-layout`) reads the same `(row, k)` and `(row, n)`
coordinates and differs only in how a block of them is fetched and multiplied.
"""


comptime MMImage[o: Origin[mut=True]] = TileTensor[
    MM_DTYPE, type_of(row_major(Coord(0, 0, 0))), origin=o
]
"""An NHWC input map for one batch index: `(H, W, C_in)`, channel fastest.

The operand `MMIm2colReader` gathers from. Rank 3 rather than rank 4 because
the batch loop is the caller's — one reader per batch index — which keeps the
gather's address arithmetic to the three axes it actually walks.

Parameters:
    o: Origin of the input map, carried from whatever the caller built. A
        struct field may not name `AnyOrigin` directly, so the reader that
        holds one of these is parameterized on the origin instead of pinning
        it to `UntrackedOrigin` and asking every caller to cast.
"""

comptime MMFlatImage[o: Origin[mut=True]] = TileTensor[
    MM_DTYPE, type_of(row_major(Coord(0))), origin=o
]
"""The same input map as one contiguous run of `H * W * C_in` elements.

A rank-1 `reshape` of `MMImage`, which is a view and moves nothing. The gather
walks the window by a carried linear offset rather than by `(h, w, c)`, because
a rank-3 access re-derives `h * W * C_in + w * C_in` at every k-vector — two
`mpyi` per A vector that the k-loop pays `MM_MR` times per step, and that
`micro_cost.sh` prices at 16 of the fused arm's 63 excess packets. Reshaping is
how a tile says "this axis triple is one address", rather than the loader
reaching past the tile for a pointer.

Parameters:
    o: Origin of the input map, carried from the rank-3 view it reshapes.
"""


struct MMConv(TrivialRegisterPassable):
    """Stride-1 2-D convolution geometry, minus what the operands already state.

    `H`, `W` and `C_in` are extents of the input-map tile and `C_out` is an
    extent of the filter, so none of them appear here — the same reason
    `matmul_bt_stripe` takes no `M`, `N` or `K`. What is left is the filter
    window, the padding, and the output width, which is the only way to turn a
    matmul row index back into an output pixel.
    """

    var R: Int
    """Filter height."""

    var S: Int
    """Filter width."""

    var pad_h: Int
    """Rows of zero padding above the input map."""

    var pad_w: Int
    """Columns of zero padding left of the input map."""

    var W_out: Int
    """Output width; `m == h_out * W_out + w_out`."""

    @always_inline
    def __init__(out self, R: Int, S: Int, pad_h: Int, pad_w: Int, W_out: Int):
        """Records the window, the padding and the output width.

        Args:
            R: Filter height.
            S: Filter width.
            pad_h: Rows of zero padding above the input map.
            pad_w: Columns of zero padding left of the input map.
            W_out: Output width.
        """
        self.R = R
        self.S = S
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.W_out = W_out


struct MMRowAnchor(TrivialRegisterPassable):
    """The K-invariant part of one A row's address, prebaked once per microtile.

    Dense: `row` is the activation row and `col` is unused. Fused conv: they are
    the input-map coordinates of the output pixel's TOP-LEFT tap,
    `h_out - pad_h` and `w_out - pad_w`, which the gather offsets by the tap
    `(r, s)`. Splitting the address this way is the pixel-anchor prebake KB
    `kernels/apple-conv2d-im2col` measured at 1.4x on the M5: the decomposition
    `m -> (h_out, w_out)` does not depend on `k`, but it sits under the gather's
    control flow where LICM will not hoist it, so the k-loop would otherwise pay
    a division per k-vector per row.

    `base` is the same anchor in LINEAR form: the element offset of that tap
    inside the input map, `(row * W + col) * C_in`. Carrying it is what lets the
    gather's address be one add of two registers. Deriving it inside `load`
    instead costs two `mpyi` per A vector, which order 061's deficit was measured
    to be made of (`micro_cost.sh`).

    Register-passable and three words wide so `StaticTuple` can hold one per A
    row of the microtile without touching memory.
    """

    var row: Int
    """Dense: the activation row. Conv: `h_out - pad_h`."""

    var col: Int
    """Dense: unused. Conv: `w_out - pad_w`."""

    var base: Int
    """Dense: unused. Conv: element offset of the top-left tap in the map."""

    @always_inline
    def __init__(out self, row: Int, col: Int, base: Int = 0):
        """Records the components of a row's K-invariant address.

        Args:
            row: Dense: the activation row. Conv: `h_out - pad_h`.
            col: Dense: unused. Conv: `w_out - pad_w`.
            base: Dense: unused. Conv: linear element offset of that tap.
        """
        self.row = row
        self.col = col
        self.base = base


trait MMAReader(Copyable, ImplicitlyDeletable):
    """Produces the A operand of the shared matmul body, one k-vector at a time.

    This is the loader-swap seam: `_mm_microtile` computes the same fp32 dot
    products, folds them the same way and stores them the same way whichever
    reader it is given — only where the A vector comes from changes. The dense
    reader takes it from an `(M, K)` activation tile; the fused-conv reader
    gathers it from an NHWC input map so the `[M, K]` im2col matrix is never
    materialised. It is the shape KB `new-primitives/amd-tile-io-expert-objects`
    describes (an owner per data movement rather than a family of free functions
    each restating the addressing) and the split the M5 conv2d and the MI355
    `TileLoaderLDS` / `TileLoaderLDSIm2col` pair already use
    (KB `kernels/apple-conv2d-im2col`).

    `anchors_rows` says whether this reader's `load` reads the anchor at all. It
    is not a hint: with it False the body does not BUILD the anchors, and the
    difference is measurable. Prebaking a dead `StaticTuple` of anchors on the
    dense arm left the instruction, HVX-op, spill and packet counts exactly equal
    but reordered three scalar row-index materializations in the store epilogue —
    dead code that survives DCE far enough to move the schedule. Skipping it at
    compile time restores byte-identity with the pre-seam nest, which is the
    property that says the plain matmul owes no re-sign.

    The k-vectors of one microtile are visited in strictly increasing order
    starting at zero, which is what lets `advance` be an add-and-carry instead of
    a division: a reader is constructed seeded at `k = 0`, the body takes it BY
    VALUE so each microtile starts from that seed, and `advance` runs exactly
    once per k-vector. `advance` is empty on the dense reader, which carries no
    k-state, so a dense build is unchanged.
    """

    comptime anchors_rows: Bool
    """Whether `load` reads the anchor; False lets the body skip the prebake."""

    @always_inline
    def anchor(self, m: Int) -> MMRowAnchor:
        """Prebakes the K-invariant address state of A row `m`."""
        ...

    @always_inline
    def anchor_next(self, prev: MMRowAnchor, m: Int) -> MMRowAnchor:
        """The anchor of row `m`, which is the row after `prev`.

        Both spellings of the same row are offered because the two readers want
        different ones. The fused reader steps from `prev` by add-and-carry,
        which is the difference between one software divide per microtile and
        `MR_c` of them — Hexagon has no divide instruction, so each is a
        `__hexagon_divsi3` call. The dense reader ignores `prev` and takes `m`,
        which keeps its `MR_c` row indices independent of each other and its
        machine code identical to the pre-seam nest's.
        """
        ...

    @always_inline
    def advance(mut self):
        """Steps the carried k-state on by one `MM_W`-lane k-vector."""
        ...

    @always_inline
    def load(
        self, anchor: MMRowAnchor, m: Int, k_off: Int
    ) -> SIMD[MM_DTYPE, MM_W]:
        """Reads the `MM_W` lanes of A at this row, k offset `k_off`.

        The row arrives spelled both ways, as it does in `anchor_next` and for
        the same reason: the dense reader wants the index `m` and the fused
        reader wants the prebaked anchor. Taking both means the anchor prebake is
        dead code on the dense arm and disappears entirely, which is what keeps
        that arm's machine code identical to the pre-seam nest's.
        """
        ...


struct MMDenseReader[o: Origin[mut=True]](MMAReader):
    """Reads A straight out of an `(M, K)` row-major activation tile.

    The plain-matmul arm of the seam, and deliberately nothing more than the
    expression it replaces: `A.load[MM_W](Coord(m, k_off))`. `advance` is empty
    because a dense row needs no k-state, and `load` takes the row index rather
    than the anchor, so the prebake is dead code on this arm — which is what
    keeps the dense build's machine code the one it had before the seam existed.

    Parameters:
        o: Origin of the activation tile, carried from the caller.
    """

    comptime anchors_rows = False
    """A dense `load` takes the row index, so the body builds no anchors."""

    var A: TileTensor[MM_DTYPE, type_of(row_major(Coord(0, 0))), origin=Self.o]
    """The `(M, K)` activation."""

    @always_inline
    def __init__(
        out self,
        A: TileTensor[MM_DTYPE, type_of(row_major(Coord(0, 0))), origin=Self.o],
    ):
        """Wraps an activation tile as the dense A operand.

        Args:
            A: The `(M, K)` activation tile.
        """
        self.A = A

    @always_inline
    def anchor(self, m: Int) -> MMRowAnchor:
        """Names row `m`; a dense row has no second address component.

        Args:
            m: The activation row.

        Returns:
            An anchor naming row `m`.
        """
        return MMRowAnchor(row=m, col=0)

    @always_inline
    def anchor_next(self, prev: MMRowAnchor, m: Int) -> MMRowAnchor:
        """Names row `m` outright; a dense row index needs no stepping.

        Args:
            prev: The previous row's anchor; unused.
            m: The activation row.

        Returns:
            An anchor naming row `m`.
        """
        return MMRowAnchor(row=m, col=0)

    @always_inline
    def advance(mut self):
        """Nothing to step: a dense row carries no k-state."""
        pass

    @always_inline
    def load(
        self, anchor: MMRowAnchor, m: Int, k_off: Int
    ) -> SIMD[MM_DTYPE, MM_W]:
        """One `MM_W`-lane contiguous A vector.

        Args:
            anchor: The row's anchor; unused, `m` says the same thing.
            m: The activation row.
            k_off: First k index of the vector.

        Returns:
            `A[m, k_off : k_off + MM_W]`.
        """
        return self.A.load[MM_W](Coord(m, k_off))


struct MMIm2colReader[o: Origin[mut=True], halo: Bool = True](MMAReader):
    """Gathers A from an NHWC input map: the fused online im2col.

    The `[M, K]` im2col matrix is never materialised. Row `m` of A is output
    pixel `(h_out, w_out) = divmod(m, W_out)`; k index
    `k = (r * S + s) * C_in + c` is that pixel's tap `(r, s)` on channel `c`,
    which lives at `input[h_out - pad_h + r, w_out - pad_w + s, c]` or is zero
    where the tap falls outside the input map.

    **The K ordering is the whole design.** Channel-FASTEST — MAX's own GPU
    im2col order (`nn/conv/gpu/im2col_matmul_2d.mojo`) — makes all `MM_W` lanes
    of one k-vector share a single `(r, s)`, so the gather is ONE contiguous
    128-byte NHWC load at the same address arithmetic the dense reader uses, and
    the halo test is one branch per VECTOR rather than one per lane. Both of this
    port's existing im2col paths order K channel-slowest, where the same vector
    is ~21 scattered runs of at most three halves. The order is a free choice for
    the dot product (it only permutes the summands) but it is NOT free for the
    fp32 fold: it permutes which k lands in which lane, so `mm_reduce_add`
    reassociates and a fused result is equal to a materialised one only within a
    tolerance band, never bitwise. See KB
    `kernels/ondevice-conv2d-im2col-gemv` and the caveat in
    `.agentwork/flux-npu/m4-vae/ANALYSIS.md`.

    **`halo` is why this arm can be as cheap as the dense one.** With it True the
    gather tests every vector against the map's edges; with it False the caller
    has promised the whole `R x S` window of every row in the microtile lands
    inside the map, and the test — and everything that exists only to feed it —
    is gone at compile time. Order 061 measured the True arm on silicon at 1.55x
    the materialising path's device time on the C=128 H=W=128 resnet conv, and
    `micro_cost.sh` prices the k-loop it was measuring at 1.47x the dense nest's
    packets on IDENTICAL vector work (12 `vmemu`, 44 `vmpy` on both arms). The
    whole deficit is scalar: 34 branches against 1, 22 compares against 1, 16
    `mpyi` and 18 `addasl` against none. `conv_bt_stripe` runs the False arm
    wherever the window provably fits, which on that layer is 92.3% of the
    microtiles.

    The address is carried linearly rather than rebuilt per access: `tap_off` is
    the current k-vector's element offset from the row's top-left tap, stepped by
    add-and-carry in `advance`, so a gather is `anchor.base + tap_off` — one add
    of two registers, which is where the `mpyi` pairs went. This works because
    the window's k order walks memory almost contiguously: within one tap ROW the
    `S * C_in` halves are adjacent, so all but every `S * C_in / MM_W`-th step is
    a bare `+= MM_W`, and the exception adds `(W - S) * C_in`.

    The contract that buys the contiguous load is `C_in % MM_W == 0`, which
    `mm_conv_fused_supported` states and `conv_bt_stripe` asserts: it makes every
    k-vector sit entirely inside one channel run, so no vector straddles a tap
    boundary. It also makes `K = R * S * C_in` a multiple of `MM_W` on its own,
    so the fused arm needs none of the K padding the materialising paths carry.
    Every VAE decoder convolution but `conv_in` (C_in = 32, 0.05% of the decode's
    MACs) satisfies it.

    `H`, `W` and `C_in` are read off the input map once here rather than at every
    access — TILE_MODEL idiom 9, and the halo test wants them in registers.

    Parameters:
        o: Origin of the input map, carried from the caller.
        halo: Whether `load` tests the tap against the map's edges. False is only
            valid where the caller has proved the window fits; see
            `mm_conv_window_inside`.
    """

    comptime anchors_rows = True
    """The gather reads the prebaked pixel anchor rather than re-deriving it."""

    var img: MMFlatImage[Self.o]
    """The input map as one contiguous run, which is how the gather walks it."""

    var w_out: Int
    """Output width, from `MMConv`; turns a row index into a pixel."""

    var pad_h: Int
    """Rows of zero padding above the input map."""

    var pad_w: Int
    """Columns of zero padding left of the input map."""

    var h_lim: Int
    """Input map height, read once from the operand's own extents."""

    var w_lim: Int
    """Input map width, read once from the operand's own extents."""

    var s_extent: Int
    """Filter width; the modulus the carried tap column wraps on."""

    var c_extent: Int
    """Input channels; the modulus the carried channel base wraps on."""

    var tap_row_jump: Int
    """`(W - S) * C_in`: what `tap_off` adds when the tap row steps."""

    var pixel_wrap: Int
    """`(W - W_out + 1) * C_in`: what `base` adds when a row of pixels wraps."""

    var r: Int
    """Carried k-state: the current tap row. Only the halo test reads it."""

    var s: Int
    """Carried k-state: the current tap column."""

    var c0: Int
    """Carried k-state: the current channel base, a multiple of `MM_W`."""

    var tap_off: Int
    """Carried k-state: element offset of this k-vector from the row's tap."""

    @always_inline
    def __init__(out self, img: MMImage[Self.o], conv: MMConv):
        """Binds the input map and seeds the carried k-state at `k = 0`.

        Args:
            img: The `(H, W, C_in)` input map for one batch index.
            conv: The window, padding and output width.
        """
        var h_lim = img.layout.shape[0]()
        var w_lim = img.layout.shape[1]()
        var c_extent = img.layout.shape[2]()
        self.img = img.reshape(Coord(h_lim * w_lim * c_extent))
        self.w_out = conv.W_out
        self.pad_h = conv.pad_h
        self.pad_w = conv.pad_w
        self.h_lim = h_lim
        self.w_lim = w_lim
        self.s_extent = conv.S
        self.c_extent = c_extent
        self.tap_row_jump = (w_lim - conv.S) * c_extent
        self.pixel_wrap = (w_lim - conv.W_out + 1) * c_extent
        self.r = 0
        self.s = 0
        self.c0 = 0
        self.tap_off = 0

    @always_inline
    def anchor(self, m: Int) -> MMRowAnchor:
        """Decomposes row `m` into its output pixel's top-left input tap.

        The one division the gather pays, and it is paid once per A row per
        microtile rather than once per k-vector.

        Args:
            m: The matmul row, a linearized output pixel.

        Returns:
            `(h_out - pad_h, w_out - pad_w)` and its linear form.
        """
        var h_out = m // self.w_out
        var row = h_out - self.pad_h
        var col = m - h_out * self.w_out - self.pad_w
        return MMRowAnchor(
            row=row, col=col, base=(row * self.w_lim + col) * self.c_extent
        )

    @always_inline
    def anchor_next(self, prev: MMRowAnchor, m: Int) -> MMRowAnchor:
        """The next output pixel, by add and carry rather than a second divide.

        Args:
            prev: The previous pixel's anchor.
            m: The matmul row; unused, because deriving it would be the divide
                this method exists to avoid.

        Returns:
            The anchor of output pixel `m`.
        """
        var col = prev.col + 1
        if col == self.w_out - self.pad_w:
            return MMRowAnchor(
                row=prev.row + 1,
                col=-self.pad_w,
                base=prev.base + self.pixel_wrap,
            )
        return MMRowAnchor(
            row=prev.row, col=col, base=prev.base + self.c_extent
        )

    @always_inline
    def advance(mut self):
        """Steps `(r, s, c0)` and the linear `tap_off` on by one k-vector.

        `C_in % MM_W == 0` is what makes this exact: the channel base walks a
        whole number of vectors before it wraps, so a k-vector never straddles a
        tap boundary and the carry is a comparison rather than a division. The
        bare `+= MM_W` is also the whole address step for every k-vector that
        does not end a tap row, because a tap row's `S * C_in` halves are
        contiguous in an NHWC map.
        """
        self.c0 += MM_W
        self.tap_off += MM_W
        if self.c0 == self.c_extent:
            self.c0 = 0
            self.s += 1
            if self.s == self.s_extent:
                self.s = 0
                self.tap_off += self.tap_row_jump
                comptime if Self.halo:
                    self.r += 1

    @always_inline
    def load(
        self, anchor: MMRowAnchor, m: Int, k_off: Int
    ) -> SIMD[MM_DTYPE, MM_W]:
        """One `MM_W`-lane A vector, gathered from the input map or zero.

        All `MM_W` lanes share the tap `(r, s)` the carried state holds, so with
        `halo` the test is one branch for the whole vector; without it there is
        no test at all and the gather is the dense reader's single contiguous
        load at the dense reader's cost.

        Args:
            anchor: The pixel's top-left input tap, from `anchor()`.
            m: The matmul row; unused, `anchor` is its prebaked form.
            k_off: First k index of the vector; unused, because the carried
                state already names the same position and reaching for `k_off`
                would put the division back in the k-loop.

        Returns:
            The tap's `MM_W` channels, or zero where the tap is outside the
            input map.
        """
        comptime if Self.halo:
            var h = anchor.row + self.r
            var w = anchor.col + self.s
            if not (0 <= h < self.h_lim and 0 <= w < self.w_lim):
                return SIMD[MM_DTYPE, MM_W](0)
        return self.img.load[MM_W](Coord(anchor.base + self.tap_off))


@always_inline
def mm_conv_window_inside(
    anchor: MMRowAnchor, rows: Int, conv: MMConv, H: Int, W: Int
) -> Bool:
    """Whether `rows` consecutive output pixels from `anchor` all read in bounds.

    The predicate that lets a microtile drop the halo test. It holds when the
    first pixel's top-left tap is inside the map and the last one's bottom-right
    tap still is, which for a stride-1 window is
    `0 <= row <= H - R` and `0 <= col` and `col + rows - 1 <= W - S`.

    The column bound also rules out a wrap onto the next output row, so no
    separate test is needed: `W_out - 1 - pad_w` is `W + pad_w - S`, which is at
    least `W - S`, so a `col` that satisfies the bound cannot be the last pixel
    of its row unless `rows` is 1.

    Args:
        anchor: The first pixel's top-left tap, `(h_out - pad_h, w_out - pad_w)`.
        rows: Consecutive output pixels the microtile covers.
        conv: The window, padding and output width.
        H: Input map height.
        W: Input map width.

    Returns:
        True when every tap of every one of those pixels is inside the map.
    """
    return (
        0 <= anchor.row <= H - conv.R
        and 0 <= anchor.col
        and anchor.col + rows - 1 <= W - conv.S
    )


@always_inline
def mm_conv_fused_supported(C_in: Int) -> Bool:
    """Whether the fused loader can serve a convolution with `C_in` channels.

    A whole number of `MM_W`-lane vectors per channel run is what makes the
    gather one contiguous load and the halo test per-vector; below it the loader
    would have to straddle tap boundaries lane by lane. A caller that gets
    `False` keeps whatever materialising path it already had.

    Args:
        C_in: The convolution's input channel count.

    Returns:
        True when the fused arm may be used.
    """
    return C_in > 0 and C_in % MM_W == 0


@always_inline
def mm_reduce_add[width: Int](x: SIMD[MM_ACCUM, width]) -> Scalar[MM_ACCUM]:
    """Horizontal sum across all lanes by rotate-and-add.

    The hand-rolled fold, not `SIMD.reduce_add()`: it lowers to `V6_*` HVX ops
    on Hexagon where the builtin scalarizes. Lane-order-independent, so it folds
    the device MAC's interleaved lanes to the same total.
    """
    var reduced = x
    comptime for i in range(8, -1, -1):
        comptime if (1 << i) < width:
            reduced += reduced.rotate_right[1 << i]()
    return reduced[0]


@always_inline
def mm_fmac_host[
    width: Int
](
    c: SIMD[MM_ACCUM, width],
    a: SIMD[MM_DTYPE, width],
    b: SIMD[MM_DTYPE, width],
) -> SIMD[MM_ACCUM, width]:
    """Portable `c + a*b`: the host stand-in for the skel's `_fmac_sf_hf`.

    Lane `i` is `c[i] + a[i]*b[i]` in natural order. The device
    `V6_vmpy_sf_hf_acc` computes the same 64 products but lands them interleaved;
    both fold to the identical dot product under `mm_reduce_add`, so this proves
    the algorithm and the on-device run confirms the instruction's numerics.
    """
    return c + a.cast[MM_ACCUM]() * b.cast[MM_ACCUM]()


@always_inline
def _mm_microtile[
    MR_c: Int, NR_c: Int, R: MMAReader
](C: MMTile, var reader: R, B: MMTile, m0: Int, n0: Int, k_vecs: Int,):
    """An `MR_c` x `NR_c` block of independent K-contiguous fp32 dot products.

    Output `(m0+mr, n0+nr)` is `dot(A row m0+mr, B row n0+nr)`. The `MR_c`
    A-vectors and `NR_c` B-vectors are loaded once per k-vector and reused across
    the tile. Callers pass only fully in-bounds tiles (`m0+MR_c <= M`,
    `n0+NR_c <= N`); the store is scalar, so no masked or over-wide vector store
    reaches `C`.

    `N` and `K` are gone from the signature because they were only ever row
    strides, and a stride is now a property of the tile. What is left — `m0`,
    `n0`, `k_vecs` — is the block this call owns, which is the argument list an
    HMX-backed body would want unchanged.

    A is not a tile here but an `MMAReader`: the plain matmul hands over an
    activation tile and the fused convolution hands over an online im2col gather,
    and NOTHING else about this body changes between them — same k-loop, same
    unroll, same fp32 accumulators, same fold, same scalar store. That is the
    loader-swap seam (KB `new-primitives/amd-tile-io-expert-objects`,
    `kernels/apple-conv2d-im2col`). The reader's row anchors are prebaked here,
    once per microtile, because the conv reader's `m -> (h_out, w_out)`
    decomposition is k-invariant but sits where LICM will not hoist it.

    Parameters:
        MR_c: A-rows in this block.
        NR_c: B-rows in this block.
        R: The A-operand reader; `MMDenseReader` or `MMIm2colReader`.
    """
    var anchor: StaticTuple[MMRowAnchor, MR_c] = {
        fill = MMRowAnchor(row=0, col=0)
    }
    comptime if R.anchors_rows:
        anchor[0] = reader.anchor(m0)
        comptime for mr in range(1, MR_c):
            anchor[mr] = reader.anchor_next(anchor[mr - 1], m0 + mr)

    var acc: StaticTuple[SIMD[MM_ACCUM, MM_W], MR_c * NR_c] = {
        fill = SIMD[MM_ACCUM, MM_W](0)
    }
    var kv = 0
    while kv + MM_KU <= k_vecs:
        var av: StaticTuple[SIMD[MM_DTYPE, MM_W], MR_c * MM_KU] = {
            fill = SIMD[MM_DTYPE, MM_W](0)
        }
        var bv: StaticTuple[SIMD[MM_DTYPE, MM_W], NR_c * MM_KU] = {
            fill = SIMD[MM_DTYPE, MM_W](0)
        }
        comptime for ku in range(MM_KU):
            var k_off = (kv + ku) * MM_W
            comptime for mr in range(MR_c):
                av[ku * MR_c + mr] = reader.load(anchor[mr], m0 + mr, k_off)
            comptime for nr in range(NR_c):
                bv[ku * NR_c + nr] = B.load[MM_W](Coord(n0 + nr, k_off))
            reader.advance()
        comptime for ku in range(MM_KU):
            comptime for mr in range(MR_c):
                comptime for nr in range(NR_c):
                    acc[mr * NR_c + nr] = mm_fmac_host(
                        acc[mr * NR_c + nr],
                        av[ku * MR_c + mr],
                        bv[ku * NR_c + nr],
                    )
        kv += MM_KU

    while kv < k_vecs:
        var k_off = kv * MM_W
        var av1: StaticTuple[SIMD[MM_DTYPE, MM_W], MR_c] = {
            fill = SIMD[MM_DTYPE, MM_W](0)
        }
        var bv1: StaticTuple[SIMD[MM_DTYPE, MM_W], NR_c] = {
            fill = SIMD[MM_DTYPE, MM_W](0)
        }
        comptime for mr in range(MR_c):
            av1[mr] = reader.load(anchor[mr], m0 + mr, k_off)
        comptime for nr in range(NR_c):
            bv1[nr] = B.load[MM_W](Coord(n0 + nr, k_off))
        reader.advance()
        comptime for mr in range(MR_c):
            comptime for nr in range(NR_c):
                acc[mr * NR_c + nr] = mm_fmac_host(
                    acc[mr * NR_c + nr], av1[mr], bv1[nr]
                )
        kv += 1

    # `alignment` is stated because `TileTensor.store` defaults it to 1 off-GPU
    # (`tile_tensor.mojo:836-838`), and an align-1 fp16 store lowers on Hexagon
    # to two `memb` plus a shift instead of one `memh`. The buffer is a
    # `Scalar[MM_DTYPE]` allocation, so its natural alignment is the same one the
    # raw-pointer store this replaces already assumed.
    comptime for mr in range(MR_c):
        comptime for nr in range(NR_c):
            C.store[1, alignment=align_of[Scalar[MM_DTYPE]]()](
                Coord(m0 + mr, n0 + nr),
                mm_reduce_add(acc[mr * NR_c + nr]).cast[MM_DTYPE](),
            )


@always_inline
def mm_tile_rows[MR: Int](K: Int, mc_bytes: Int) -> Int:
    """A-rows a `mc_bytes` activation panel holds: a positive multiple of `MR`.

    Rounding down to `MR` is load-bearing, not cosmetic. Every tile then starts
    on a multiple of `MR`, so the last tile's full-tile bound
    `m0 + ((M - m0) // MR) * MR` equals the untiled `(M // MR) * MR`, and the
    rows that fall to the single-row remainder path are the same rows as before
    the tiling — which is what makes the retile bit-exact rather than merely
    correct. A budget too small for even one `MR`-row tile rounds up to `MR`
    rather than stalling.

    This is the one function here that keeps a bare extent. It reads no memory,
    so there is no tile whose type could carry `K`, and the host twin checks the
    multiple-of-`MR` property directly over a `K` x budget grid — taking a tile
    would force that property test to fabricate operands it never dereferences,
    trading a live check for a uniform signature.
    """
    var rows = (mc_bytes // (K * size_of[MM_DTYPE]()) // MR) * MR
    if rows < MR:
        rows = MR
    return rows


@always_inline
def mm_conv_tile_rows[
    MR: Int
](conv: MMConv, W: Int, C_in: Int, mc_bytes: Int) -> Int:
    """Output rows a `mc_bytes` INPUT-MAP panel holds: a positive multiple of `MR`.

    `mm_tile_rows` sizes the M-tile in im2col bytes, `K * 2` per row. A fused
    loader never materialises those bytes; what stays resident across one N sweep
    is the slice of the INPUT MAP the tile's pixels read, which is
    `(rows_out + R - 1)` input rows of `W * C_in` halves each — the halo overlap
    between adjacent output rows is read once, not `R * S` times. At the same
    budget that buys several times more output rows, and the weight stripe is
    re-read `ceil(M / mc)` times, so the correction cuts the B re-read by the
    same factor it raises the tile. The 4.5x worked example is in
    `.agentwork/flux-npu/m4-vae/ANALYSIS.md`; treat the factor as shape-dependent
    and the formula as the thing that is true.

    Rounded down to `MR` for the same reason `mm_tile_rows` is: every tile then
    starts on a multiple of `MR`, so the rows that fall to the single-row
    remainder path are the rows that would have fallen there untiled, which is
    what makes a retile bit-exact rather than merely correct.

    `mc_bytes // in_row_bytes - (conv.R - 1)` UNDERFLOWS the instant one input
    row is at or below `mc_bytes / conv.R` — the subtraction can go negative,
    not just to zero. Left alone, that negative row count survives the
    `* conv.W_out` and the floor-to-`MR` rounding as a negative pixel count
    and is caught only by the final `< MR` clamp, at the smallest possible
    tile. Clamping `in_rows` to at least one output row before it is scaled by
    `conv.W_out` makes that state impossible instead: a tile is never sized
    finer than one full output row, however far short the budget falls of
    holding even `conv.R` input rows resident. Twin fix to `_conv_tile_rows`
    in `npu_kernel.mojo` — see there for the shapes that hit it.

    Parameters:
        MR: Register-block extent over A-rows.

    Args:
        conv: The window, padding and output width.
        W: Input map width.
        C_in: Input channels.
        mc_bytes: Input-map bytes held resident across one N sweep.

    Returns:
        Output rows per M-tile: a positive multiple of `MR`.
    """
    var in_row_bytes = W * C_in * size_of[MM_DTYPE]()
    var in_rows = mc_bytes // in_row_bytes - (conv.R - 1)
    if in_rows < 1:
        in_rows = 1
    var rows = (in_rows * conv.W_out // MR) * MR
    if rows < MR:
        rows = MR
    return rows


@always_inline
def mm_stripe[
    MR: Int, NR: Int, R: MMAReader
](
    C: MMTile,
    reader: R,
    B: MMTile,
    n_start: Int,
    n_end: Int,
    k_vecs: Int,
    mc: Int,
):
    """The shared nest: `C[:, n_start:n_end] += dot(A rows, B rows)`, tiled.

    Both arms of the loader-swap seam run THIS body — `matmul_bt_stripe` with a
    dense reader, `conv_bt_stripe` with the fused im2col gather. Everything the
    matmul was measured and gated on lives here and is written once: the
    register-blocked full tiles, the `M % MR` row edge and `stripe % NR` column
    edge falling to single-row / single-column microtiles (the scalar-remainder
    idiom, so a non-64-aligned N tail needs no masked vector store), and the
    M-tile sitting OUTSIDE the N sweep so an activation row crosses DRAM once per
    tile instead of once per N-pair.

    Reordering which independent output is computed when cannot change any
    output: each `C[m, n]` is still produced by exactly one microtile invocation
    accumulating the same k-vectors in the same order into the same fp32 register
    pair, and the `MR`-aligned tile start keeps the remainder-path row set
    identical.

    Parameters:
        MR: Register-block extent over A-rows.
        NR: Register-block extent over B-rows.
        R: The A-operand reader.

    Args:
        C: The `(M, N)` output tile.
        reader: The A-operand reader.
        B: The `(N, K)` weight tile — the transposed contract, stated in the
            operand's own extents.
        n_start: First output column this call owns.
        n_end: One past the last output column this call owns.
        k_vecs: `K / MM_W`, the reduction length in whole HVX vectors.
        mc: A-rows held resident across one N sweep.
    """
    var M = C.layout.shape[0]()
    var n_full = n_start + ((n_end - n_start) // NR) * NR

    var m0 = 0
    while m0 < M:
        var m_stop = m0 + mc
        if m_stop > M:
            m_stop = M
        var m_full = m0 + ((m_stop - m0) // MR) * MR

        var n = n_start
        while n < n_full:
            var m = m0
            while m < m_full:
                _mm_microtile[MR, NR](C, reader.copy(), B, m, n, k_vecs)
                m += MR
            while m < m_stop:
                _mm_microtile[1, NR](C, reader.copy(), B, m, n, k_vecs)
                m += 1
            n += NR
        while n < n_end:
            var m = m0
            while m < m_full:
                _mm_microtile[MR, 1](C, reader.copy(), B, m, n, k_vecs)
                m += MR
            while m < m_stop:
                _mm_microtile[1, 1](C, reader.copy(), B, m, n, k_vecs)
                m += 1
            n += 1
        m0 += mc


@always_inline
def _mm_conv_walk[
    MR: Int, NR_c: Int, RB: MMAReader, RI: MMAReader
](
    C: MMTile,
    border: RB,
    interior: RI,
    B: MMTile,
    conv: MMConv,
    H: Int,
    W: Int,
    m0: Int,
    m_full: Int,
    m_stop: Int,
    n: Int,
    k_vecs: Int,
):
    """One M-tile's rows for output channels `n ..< n + NR_c`, halo-split.

    The two readers compute the same A vector wherever both are valid — the
    interior one IS the border one with a provably-true test deleted — so which
    microtile gets which cannot move a result. That is what makes the split a
    pure schedule change and lets `test_conv_im2col_kernel.mojo` keep gating the
    fused arm as BIT-equal to a materialised im2col of the same k order.

    The pixel is carried across the walk by `anchor_next` rather than recovered
    from `m`, so the predicate costs compares and no division. The microtile
    prebakes its own anchors from `m0` regardless; this cursor exists only to
    decide which arm runs, and its cost is `MR` add-and-carries against a body
    of `k_vecs` k-steps.

    Only the full `MR`-row tiles are split. The `M % MR` remainder rows keep the
    halo reader unconditionally: they are at most `MR - 1` rows of a tile, the
    shapes that reach them are the small ones, and giving them their own pair of
    instantiations would double this kernel's code for no measurable work.

    Parameters:
        MR: Register-block extent over A-rows.
        NR_c: B-rows in these microtiles.
        RB: The halo-testing reader.
        RI: The reader with the halo test compiled out.

    Args:
        C: The `(H_out * W_out, C_out)` output tile.
        border: Reader for microtiles whose window can leave the map.
        interior: Reader for microtiles whose window provably cannot.
        B: The `(C_out, R * S * C_in)` filter.
        conv: The window, padding and output width.
        H: Input map height.
        W: Input map width.
        m0: First row of this M-tile.
        m_full: One past the last row covered by a full `MR`-row tile.
        m_stop: One past the last row of this M-tile.
        n: First output channel these microtiles own.
        k_vecs: `K / MM_W`, the reduction length in whole HVX vectors.
    """
    var cur = border.anchor(m0)
    var m = m0
    while m < m_full:
        if mm_conv_window_inside(cur, MR, conv, H, W):
            _mm_microtile[MR, NR_c](C, interior.copy(), B, m, n, k_vecs)
        else:
            _mm_microtile[MR, NR_c](C, border.copy(), B, m, n, k_vecs)
        comptime for _ in range(MR):
            cur = border.anchor_next(cur, 0)
        m += MR
    while m < m_stop:
        _mm_microtile[1, NR_c](C, border.copy(), B, m, n, k_vecs)
        m += 1


@always_inline
def conv_bt_stripe[
    MR: Int, NR: Int
](
    C: MMTile,
    img: MMImage[MutUnsafeAnyOrigin],
    B: MMTile,
    conv: MMConv,
    n_start: Int,
    n_end: Int,
    mc_bytes: Int,
):
    """One stride-1 convolution as a matmul over a FUSED online im2col.

    `C[m, n]` is output pixel `m` of output channel `n`: the dot product of that
    pixel's `R * S * C_in` taps with filter row `n`. The im2col matrix is never
    built — `MMIm2colReader` gathers each A vector out of the NHWC input map at
    load time, with a zero vector where a tap falls in the halo. Below the A load
    it is `_mm_microtile`, unchanged and shared with the dense matmul, which is
    the point: the two arms differ only in where an A vector comes from.

    The tile walk is spelled here rather than taken from `mm_stripe` because this
    arm chooses a reader per microtile (`_mm_conv_walk`) and the dense arm must
    not grow a comptime flag to carry a choice it never makes — `gate.sh` gates
    that nest as byte-identical to the build that had no seam, and the M4.6-L1
    record is that even provably dead code in it moves the schedule. The two
    walks are the same shape: an M-tile outside the N sweep, full `MR x NR`
    tiles, and the row and column edges on single-row / single-column microtiles
    so a non-`MM_W`-aligned N tail needs no masked vector store.

    The filter is the b-transposed `(C_out, K)` operand the matmul already takes,
    with `k = (r * S + s) * C_in + c` — channel FASTEST. A filter stored `FCRS`
    (channel-slowest) must be permuted to that order once at weight load; doing
    it per dispatch would put back the host-side cost this deletes.

    `C_in % MM_W == 0` is required (`mm_conv_fused_supported`). It is what makes
    each A vector one contiguous load rather than a lane-by-lane straddle, and it
    also makes `K = R * S * C_in` a multiple of `MM_W` on its own, so no K
    padding is needed on this arm.

    Numerics: the k order differs from the two materialising paths in-tree, so
    the fp32 lane a given `k` lands in differs and `mm_reduce_add` reassociates.
    Equality with a materialised im2col of the SAME k order is exact; equality
    with the FCRS order is a tolerance band. `test_conv_im2col_kernel.mojo` gates
    both statements.

    Parameters:
        MR: Register-block extent over A-rows.
        NR: Register-block extent over B-rows.

    Args:
        C: The `(H_out * W_out, C_out)` output tile for this batch index.
        img: The `(H, W, C_in)` NHWC input map for this batch index.
        B: The `(C_out, R * S * C_in)` filter, channel-fastest within a tap.
        conv: The window, padding and output width.
        n_start: First output channel this call owns.
        n_end: One past the last output channel this call owns.
        mc_bytes: INPUT-MAP bytes held resident across one N sweep — not im2col
            bytes; see `mm_conv_tile_rows`.
    """
    comptime assert (
        MM_W * size_of[MM_DTYPE]() * 8 == 128 * 8
    ), "MM_W must be exactly one HVX vector register"

    var H = img.layout.shape[0]()
    var W = img.layout.shape[1]()
    var C_in = img.layout.shape[2]()
    var k_vecs = conv.R * conv.S * C_in // MM_W
    var border = MMIm2colReader(img, conv)
    var interior = MMIm2colReader[halo=False](img, conv)
    var mc = mm_conv_tile_rows[MR](conv, W, C_in, mc_bytes)

    var M = C.layout.shape[0]()
    var n_full = n_start + ((n_end - n_start) // NR) * NR

    var m0 = 0
    while m0 < M:
        var m_stop = m0 + mc
        if m_stop > M:
            m_stop = M
        var m_full = m0 + ((m_stop - m0) // MR) * MR

        var n = n_start
        while n < n_full:
            _mm_conv_walk[MR, NR](
                C,
                border,
                interior,
                B,
                conv,
                H,
                W,
                m0,
                m_full,
                m_stop,
                n,
                k_vecs,
            )
            n += NR
        while n < n_end:
            _mm_conv_walk[MR, 1](
                C,
                border,
                interior,
                B,
                conv,
                H,
                W,
                m0,
                m_full,
                m_stop,
                n,
                k_vecs,
            )
            n += 1
        m0 += mc


@always_inline
def conv_bt[
    MR: Int, NR: Int
](
    C: MMTile,
    img: MMImage[MutUnsafeAnyOrigin],
    B: MMTile,
    conv: MMConv,
    mc_bytes: Int,
):
    """Whole convolution, one thread computing every output channel.

    Parameters:
        MR: Register-block extent over A-rows.
        NR: Register-block extent over B-rows.

    Args:
        C: The `(H_out * W_out, C_out)` output tile.
        img: The `(H, W, C_in)` NHWC input map.
        B: The `(C_out, R * S * C_in)` filter.
        conv: The window, padding and output width.
        mc_bytes: Input-map bytes held resident across one N sweep.
    """
    conv_bt_stripe[MR, NR](C, img, B, conv, 0, C.layout.shape[1](), mc_bytes)


@always_inline
def matmul_bt_stripe[
    MR: Int, NR: Int
](C: MMTile, A: MMTile, B: MMTile, n_start: Int, n_end: Int, mc_bytes: Int):
    """`C[:, n_start:n_end] = A @ B[n_start:n_end]^T`, register-blocked, fp16.

    `[n_start, n_end)` is the N stripe one compute thread owns on-device; the
    host test passes the whole `[0, N)`. Full `MR` x `NR` tiles take the
    register-blocked path; the `M % MR` row edge and the `stripe % NR` column
    edge fall to single-row / single-column microtiles — the scalar-remainder
    idiom — each still an exact dot product, so no masked vector store is needed
    for a non-64-aligned N tail. `K` must be a multiple of 64.

    An M-tile of `mm_tile_rows(K, mc_bytes)` rows sits OUTSIDE the N sweep, so
    the activation rows a sweep touches stay within a bounded budget and each
    row crosses DRAM once per M-tile instead of once per N-pair. Reordering
    which independent output is computed when cannot change any output: each
    `C[m, n]` is still produced by exactly one microtile invocation accumulating
    the same k-vectors in the same order into the same fp32 register pair, and
    the `MR`-aligned tile start keeps the remainder-path row set identical.

    `M`, `N` and `K` are not arguments. `M` and `K` are read back from the
    operands that own them, and `N` is not needed at all — it was only ever
    `C`'s row stride, which the layout now holds, plus a sweep bound the caller
    already states as `n_end`. The seam that turns pointers and extents into
    these three layouts sits one line inside the frozen C ABI above this file,
    which is where the untyped form belongs. The N stripe itself stays
    hand-written rather than a `.distribute()` view: that partition is cyclic
    and this one is blocked (KB
    `kernels/distribute-is-cyclic-this-part-wants-blocked`).

    Parameters:
        MR: Register-block extent over A-rows.
        NR: Register-block extent over B-rows.

    Args:
        C: The `(M, N)` output tile.
        A: The `(M, K)` activation tile.
        B: The `(N, K)` weight tile — the transposed contract, stated in the
            operand's own extents rather than carried alongside them as a flag
            a caller could contradict.
        n_start: First output column this call owns.
        n_end: One past the last output column this call owns.
        mc_bytes: Activation bytes held resident across one N sweep.
    """
    # One 128-byte HVX register's worth of fp16 lanes, and nothing else. Derived
    # from the register geometry, never from `simd_width_of` — this same source
    # is compiled for the HOST by `test_matmul_kernel.mojo`, where that query
    # correctly returns 8 and would silently test an 8-lane kernel (KB
    # `kernels/tiletensor-vector-width-state-it-explicitly`).
    comptime assert (
        MM_W * size_of[MM_DTYPE]() * 8 == 128 * 8
    ), "MM_W must be exactly one HVX vector register"

    # Each extent is read once, from the operand whose shape defines it: the row
    # count from the output being written, the reduction length from the
    # activation being read. Reading the same extent off a second operand would
    # hand the compiler two values it cannot prove equal. `layout.shape[i]()`
    # rather than `dim[i]()` — the former is already an `Int`, where the latter
    # returns `Scalar[linear_idx_type]`, int64 on this 32-bit part.
    var K = A.layout.shape[1]()
    var reader = MMDenseReader(A)
    mm_stripe[MR, NR](
        C, reader, B, n_start, n_end, K // MM_W, mm_tile_rows[MR](K, mc_bytes)
    )


@always_inline
def matmul_bt[MR: Int, NR: Int](C: MMTile, A: MMTile, B: MMTile, mc_bytes: Int):
    """Whole-matrix `C[M, N] = A[M, K] @ B[N, K]^T`, one thread computing all N.

    Parameters:
        MR: Register-block extent over A-rows.
        NR: Register-block extent over B-rows.

    Args:
        C: The `(M, N)` output tile.
        A: The `(M, K)` activation tile.
        B: The `(N, K)` weight tile.
        mc_bytes: Activation bytes held resident across one N sweep.
    """
    matmul_bt_stripe[MR, NR](C, A, B, 0, C.layout.shape[1](), mc_bytes)


@always_inline
def _mm_microtile_hmx[
    space: AddressSpace,
    //,
    MR_c: Int,
    NR_c: Int,
    emit_hmx: Bool = HMX_EMIT_MMA,
](
    C: MMTile,
    A: MMTile,
    B: MMTile,
    m0: Int,
    n0: Int,
    k_vecs: Int,
    scratch: HmxScratch[space],
):
    """One `MR_c` x `NR_c` output block computed on the matrix array.

    `_mm_microtile`'s contract, extended by exactly one argument. The seam args
    are unchanged and mean the same things: the block at `(m0, n0)` over a
    `k_vecs`-long depth, addressed as coordinates into the caller's own views.
    That the HVX and HMX bodies agree on this signature is what makes the two a
    substitution rather than two kernels.

    The block is one crouton tile, which is not a coincidence: `MR_c == NR_c ==
    HMX_TILE` is the array's own 32x32 shape, so the register-blocking parameters
    that tune the HVX body become a hardware constant here.

    `k_vecs` counts HVX fp16 vectors (64 elements); a crouton is 32 deep, so the
    array sees `2 * k_vecs` tiles. Converting here rather than changing the
    argument keeps a caller from having to know which unit it is feeding.

    WHY `scratch` HAD TO BE ADDED, rather than the body reading the operands
    directly: HMX multiplies packed tiles resident in VTCM, and neither the
    packing nor the residency is per-tile work. `scratch` is where the stripe
    staged them. A body that allocated or acquired per tile would also have to
    release per tile, and a release clears the accumulators (KB
    `known-limitations/ondevice-hmx-lock-protocol`).

    Parameters:
        MR_c: Register-block extent over A-rows; must be `HMX_TILE`.
        NR_c: Register-block extent over B-rows; must be `HMX_TILE`.
        emit_hmx: Whether `hmx_mma_panel` issues the instructions or models
            them. Defaults to `HMX_EMIT_MMA`, which is OFF, so a host twin gets
            the model without asking and a device build must ask.

    Args:
        C: The `(M, N)` output tile.
        A: The `(M, K)` activation tile.
        B: The `(N, K)` weight tile, already staged into `scratch.wgt`.
        m0: First output row of the block.
        n0: First output column of the block.
        k_vecs: HVX fp16 vectors spanning the reduction.
        scratch: The stripe's VTCM residency.
    """
    comptime assert (
        MR_c == HMX_TILE and NR_c == HMX_TILE
    ), "the matrix array computes exactly one 32x32 tile per pass"

    var k_tiles = hmx_k_tiles(k_vecs * MM_W)

    hmx_pack_panel[False, space](scratch.act, A, m0, 0, k_tiles)
    hmx_mma_panel[emit_hmx](scratch, k_tiles)
    # The reference nest is no longer dispatched, so its arm code has no witness
    # to reach; the pipelined step below is what the skel reports.
    _ = hmx_unpack_panel[False, space](C, scratch.out, m0, n0, 1)
    _ = B


@always_inline
def matmul_bt_hmx_accepts(
    M: Int, N: Int, K: Int, n_start: Int, n_end: Int
) -> Bool:
    """Whether the array can compute this stripe. Shape only; no residency.

    Named because two callers must agree on it and cannot see each other: the
    stripe refuses on it, and the entry that acquires the unit has to know the
    answer BEFORE it locks — a grant taken for a stripe that then refuses is a
    held unit doing nothing, and a release clears accumulators the next launch
    would have wanted.

    `K` is refused on `MM_W`, not on `HMX_TILE`, and the difference is a real
    one: `_mm_microtile_hmx` takes `k_vecs` so its signature can stay
    `_mm_microtile`'s, and recovers the depth as `k_vecs * MM_W`. That round
    trip is lossy for a `K` that is a multiple of 32 but not 64 — `K = 32`
    yields `k_vecs = 0`, a zero-tile panel, and an output tile of pure bias with
    nothing to say it never multiplied. `MM_W` is twice `HMX_TILE`, so this
    bound implies the tile bound rather than relaxing it.

    BOTH ARMS SHARE THIS PREDICATE. The W8A8 artefact carries its requant
    records and per-tensor header in the tail of the same file length, and that
    tail is `8N + 16` bytes against `N * K` free ones — implied by the `K % MM_W`
    bound above for every `N`, so the integer arm needs no extra refusal here.
    `hmx_int8_artefact_fits` states it where it is NOT implied: in the offline
    repacker, whose extents come from a manifest.

    Args:
        M: Output rows.
        N: Output columns.
        K: The reduction extent.
        n_start: First output column the stripe owns.
        n_end: One past the last output column the stripe owns.

    Returns:
        True when every extent is a whole number of crouton tiles.
    """
    if M % HMX_TILE != 0 or N % HMX_TILE != 0 or K % MM_W != 0:
        return False
    return (n_end - n_start) % HMX_TILE == 0


@always_inline
def matmul_bt_stripe_hmx[
    space: AddressSpace, //, emit_hmx: Bool = HMX_EMIT_MMA
](
    C: MMTile,
    A: MMTile,
    B: MMTile,
    n_start: Int,
    n_end: Int,
    scratch: HmxScratch[space],
) -> Bool:
    """`matmul_bt_stripe`'s contract on the matrix array. Returns whether it ran.

    THE DESIGN NOTE, since this is the piece that cannot be finished device-free.

    Why this is a sibling of `matmul_bt_stripe` and not a `comptime if` inside
    it: the two nests do not differ in their inner body, they differ in what
    lives outside their loops. This one holds the HMX unit across the whole
    stripe (acquiring per tile would clear the accumulators), it needs VTCM
    panels whose lifetime is the stripe, and it packs a weight panel once per
    N-tile rather than reading `B` in the inner loop. A selector at
    `_mm_microtile` could not express any of that; a selector at the SKEL, which
    already owns the VTCM grant and the lock, expresses all of it — hence
    `MM_PATH_HMX` living where it does.

    NO LONGER THE DISPATCHED NEST, and kept deliberately (P5 M4.2). The kernel
    routes stripes to `matmul_bt_stripe_hmx_step` below, which stages the same
    tiles on the launch's other four threads; this one is now the REFERENCE it
    is scored against. Its value is that it is the arrangement order 080 proved
    exact on silicon, so a host differential against it is a differential
    against banked evidence rather than against a second new thing. Do not
    delete it to remove a duplicate nest.

    Its `mc_bytes` note, kept because M4.2 half-refuted it: the reasoning was
    that this nest's activations are already resident in VTCM by the time the
    array reads them, so bounding a DRAM stream would tune something that no
    longer exists. True of the DRAFT stream and false of the cost that mattered
    — the activation panel is PACKED once per N-tile here, and the pack is
    scalar, so the N sweep repeats work rather than traffic. `act_slots` in
    `HmxStripePlan` is the same blocking knob at that cost's granularity.

    Refusal, not fallback. Every extent must be a multiple of `HMX_TILE`, and a
    stripe that is not is REFUSED with nothing written, reported through the
    return so the skel's witness can carry it. There is deliberately no quiet
    drop to `matmul_bt_stripe`: a silent fallback would report HVX numbers as
    HMX ones, which is the single failure this path must not have.

    WHAT REMAINS is a device A/B against selector 24 at equal shapes. Everything
    above the instruction words is host-checkable and checked; the instruction
    stream is not. What silicon has proven is ONE 32x32 tile at `range = 2047`,
    so multi-tile `:deep` accumulation, chunk boundaries, several output tiles
    under one lock and the accumulator's internal width are all still
    inferences.

    Parameters:
        emit_hmx: Whether the microtiles issue HMX or model it.

    Args:
        C: The `(M, N)` output tile.
        A: The `(M, K)` activation tile.
        B: The `(N, K)` weight tile.
        n_start: First output column this call owns.
        n_end: One past the last output column this call owns.
        scratch: The stripe's VTCM residency.
    """
    # Routing stripes to the array while the MMA is modelled would report a
    # scalar reference's numbers as the matrix unit's — the one failure this
    # path must not have, and the reason the two flags are coupled here rather
    # than in a comment.
    comptime assert (
        not MM_PATH_HMX or emit_hmx
    ), "MM_PATH_HMX needs HMX_EMIT_MMA: the modelled MMA is the host twin's"

    var M = C.layout.shape[0]()
    var K = A.layout.shape[1]()
    var N = B.layout.shape[0]()

    if not matmul_bt_hmx_accepts(M, N, K, n_start, n_end):
        return False

    var k_vecs = K // MM_W
    var k_tiles = hmx_k_tiles(K)

    var n = n_start
    while n < n_end:
        # Once per N-tile, outside the M sweep: a weight panel is the expensive
        # pack (a transposing gather) and every output row reuses it, which is
        # the HVX nest's `MM_MR > MM_NR` reasoning at the array's own shape.
        hmx_pack_panel[True, space](scratch.wgt, B, n, 0, k_tiles)
        var m = 0
        while m < M:
            _mm_microtile_hmx[HMX_TILE, HMX_TILE, emit_hmx](
                C, A, B, m, n, k_vecs, scratch
            )
            m += HMX_TILE
        n += HMX_TILE
    return True


# ===--------------------------------------------------------------------------===
# The PIPELINED stripe (P5 M4.2)
#
# `matmul_bt_stripe_hmx` above is one thread doing everything, and order 080
# recorded it 8.6-23x slower than the four-thread vector nest at all six render
# shapes. The mechanism is in its own loop nest rather than in the array: the
# activation panel for row-tile `m` is re-packed inside the `n` sweep, so it is
# packed `N / HMX_TILE` times, and `hmx_pack_panel` is a scalar element-at-a-time
# gather. Counting its iterations against 080's recorded wall times: activation
# packing alone accounts for 24-43 s of the 24-43 s recorded at the two largest
# shapes, at any per-element cost near a scalar load/store pair. The array is not
# what is slow; the staging around it is, and one thread does all of it.
#
# So this nest changes two things and nothing else:
#
#   1. An A-BLOCK of `act_slots` activation panels is packed ONCE and stays
#      resident across the whole N sweep. Panels per block are what VTCM holds,
#      so the weight panel is re-packed once per block instead of the activation
#      panel once per column tile — the same cache-blocking trade the HVX nest
#      makes with `mc_bytes`, at the array's own granularity.
#   2. The four compute threads pack and unpack while the fifth, which holds the
#      HMX grant, only multiplies. That is ggml-hexagon's topology (one
#      persistent HVX-free thread services the array; worker threads feed it) and
#      it is the ONLY split in which the packers may later be vectorised at all:
#      "a thread with HVX enabled cannot have HMX enabled" (`qurt_hmx.h:184`), so
#      an HVX pack on the granted thread would put the grant 080 just banked at
#      risk. Dropped from their design: the lock-free queue. The array's work
#      here is a small fraction of the feeders', so a queue would schedule the
#      cheap side; a step index every thread computes from `(M, N, K)` is enough.
#
# WHAT IS NOT CHANGED, deliberately, because it is the A/B gate: every output
# tile still reaches `hmx_mma_panel` as one `HmxScratch` holding one activation
# panel, one weight panel and one readout, with the same `k_tiles`, the same
# `mxclracc.hf` before it and the same chunk loop inside it. Accumulation order
# per output tile is therefore bit-identical to `matmul_bt_stripe_hmx`; only
# which thread stages an operand, and how many times, moves.
# ===--------------------------------------------------------------------------===


@fieldwise_init
struct HmxStripePlan(Copyable, Movable):
    """The pipelined stripe's geometry: pure arithmetic on `(M, N, K)`.

    Every thread computes this independently and reaches the same answer, which
    is what makes the schedule safe without a broadcast: `steps()` is the number
    of rendezvous each thread will make, so no thread can arrive at a barrier
    another one never reaches.
    """

    var m_tiles: Int
    """Output row tiles, `M // HMX_TILE`."""

    var n_tiles: Int
    """Output column tiles, `N // HMX_TILE`."""

    var k_tiles: Int
    """Croutons spanning the reduction, `K // HMX_TILE`."""

    var act_slots: Int
    """Activation panels resident per A-block. 0 means the budget refuses."""

    var blocks: Int
    """A-blocks the row tiles divide into."""

    @always_inline
    def steps[fold_trailing: Bool = HMX_FOLD_TRAILING](self) -> Int:
        """Rendezvous the whole stripe takes.

        `n_tiles + 2` per block: one to stage the block, then one per column
        tile, then one more so the last readout is unpacked after the array
        wrote it.

        Under `fold_trailing` every block's trailing unpack except the last
        rides the NEXT block's stage step instead of a step of its own —
        `blocks * (n_tiles + 1) + 1`, which at `blocks == 1` is the unfolded
        count exactly. Still pure arithmetic on the plan, so the order-084
        agreement property is unchanged: no participant's count depends on
        anything another participant cannot compute.

        Parameters:
            fold_trailing: Whether the folded schedule's count is returned.
                Defaulted to the build knob so the kernel and every caller
                agree without a second source of truth; the twin instantiates
                both arms explicitly.
        """
        comptime if fold_trailing:
            return self.blocks * (self.n_tiles + 1) + 1
        else:
            return self.blocks * (self.n_tiles + 2)


@always_inline
def hmx_stripe_plan[
    dtype: DType = HMX_DTYPE
](M: Int, N: Int, K: Int, budget: Int) -> HmxStripePlan:
    """Plans a pipelined stripe against a VTCM budget in bytes.

    Parameters:
        dtype: The arm's operand type. A W8A8 panel is HALF the bytes, so the
            same budget holds roughly twice the activation slots and the step
            count differs -- which is why the plan is per arm and not a shape
            property. Every participant computes it from the same inputs, so
            it stays the broadcast-free agreement it was.

    Args:
        M: Output rows.
        N: Output columns.
        K: The reduction extent.
        budget: Bytes of scratch the caller can carve.

    Returns:
        The plan; `act_slots == 0` when the budget cannot hold even one panel.
    """
    var m_tiles = M // HMX_TILE
    var n_tiles = N // HMX_TILE
    var k_tiles = hmx_k_tiles(K)
    var act_slots = hmx_stripe_act_slots[dtype](k_tiles, budget, m_tiles)
    var blocks = 0
    if act_slots > 0:
        blocks = (m_tiles + act_slots - 1) // act_slots
    return HmxStripePlan(m_tiles, n_tiles, k_tiles, act_slots, blocks)


@fieldwise_init
struct HmxStripeRoles(Copyable, Movable):
    """Which thread does what, derived from the launch's participant count.

    P5 M4.2 order 084: the first version assumed five participants because every
    `enqueue_function` in `registrations.mojo` asks for `NPU_THREAD_MASK`
    (`0x1F`). Assuming it is what makes a launch configured any other way a
    rendezvous nobody completes rather than a refusal, so the count is read and
    the roles follow from it.

    The array thread is the LAST participant, which at five is thread 4 — the
    one the HVX nest gives no output columns to, and the split order 082
    measured. That identity is why the work partition, and therefore every
    output tile's operands, are unchanged at the production count.
    """

    var feeders: Int
    """Threads that pack and unpack. 0 when the launch cannot host the roles."""

    var array_thread: Int
    """The one participant that holds the grant, and the feeder count's index.
    Negative when the launch cannot host the roles."""

    @always_inline
    def usable(self) -> Bool:
        """Whether the launch can host the pipeline at all."""
        return self.feeders >= 1


@always_inline
def hmx_stripe_roles(participants: Int) -> HmxStripeRoles:
    """The role plan for a launch of `participants` threads.

    Two participants is the floor: one feeder and one array thread. A lone
    participant would be the array thread with nobody to stage its operands, and
    is refused rather than served by making the array thread feed itself — which
    would put HVX-capable work on the granted thread, the one thing
    `qurt_hmx.h:184` forbids.

    Args:
        participants: Threads the launch actually spawned.

    Returns:
        The plan; `usable()` is False when the count cannot host it.
    """
    if participants < 2:
        return HmxStripeRoles(0, -1)
    return HmxStripeRoles(participants - 1, participants - 1)


@always_inline
def hmx_stripe_arrivals(
    participants: Int, accepts: Bool, granted: Bool, steps: Int
) -> Int:
    """Rendezvous one participant makes in a whole dispatch.

    Has no thread-index parameter, and that is the property rather than an
    omission: order 084 wedged because a refusal that some threads took and
    others did not left the takers waiting on arrivals that were never coming.
    Every input here is a value every participant reads identically — the count
    is one release-stored word, `accepts` is arithmetic on the kernel's scalar
    arguments, and `granted` is broadcast through witness slot 7 by the
    rendezvous itself.

    Args:
        participants: Threads the launch spawned.
        accepts: Whether the shape predicate admitted this dispatch.
        granted: Whether the array thread's resolution reported the unit taken.
        steps: `HmxStripePlan.steps()` for the dispatch.

    Returns:
        Arrivals, equal on every participant.
    """
    if not hmx_stripe_roles(participants).usable() or not accepts:
        return 0
    if not granted:
        return 1
    return 1 + steps


@always_inline
def hmx_work_share(total: Int, idx: Int, parts: Int) -> Int:
    """First item of share `idx` when `total` items divide `parts` ways.

    Shares are `[hmx_work_share(t, i, p), hmx_work_share(t, i + 1, p))`, which partitions
    exactly — no item is packed twice and none is left unpacked — for any
    `total`, including one smaller than `parts`.
    """
    return (total * idx) // parts


@always_inline
def _hmx_arm_pack_act[
    dtype: DType,
    space: AddressSpace,
    //,
    act_conv: Bool = False,
    activation_native: Bool = False,
](
    dst: CroutonPanelOf[dtype, space],
    A: MMTile,
    B: MMTile,
    row0: Int,
    col0: Int,
    tiles: Int,
    conv: HmxConvGeom,
):
    """Stages one feeder's share of an activation panel, for any arm.

    The fp16 arm copies; the integer arm quantizes as it copies, with the
    per-tensor scale and zero point read off the artefact `B` rather than passed
    down as a kernel argument. `B` travels with its own quantization because a
    registry may legitimately mix int8 and fp16 families and the frozen selector
    ABI has no scalar to spare for a per-dispatch pair.

    THE CONVOLUTION IS A THIRD SOURCE FOR THE SAME PANEL, and it is the whole of
    the conv arm: `act_conv` says `A` is an NHWC input map as `(H, W * C_in)`
    rather than a materialised `[M, K]`, and the loader gathers `A[m][k]` out of
    it in the fused k order. The destination, the crouton range and the bytes
    the array later reads are the same either way, which is what keeps the conv
    arm a loader policy rather than a second kernel (VAE_CONV_HMX_SCOPE.md).

    THE FOURTH SOURCE IS NO TRANSFORM AT ALL: `activation_native` says `A`'s
    bytes are already row-panel-major croutons behind the same `[M, K]` view —
    the weight-native statement one operand over (P5 M4.3; annihilation rung
    1) — so the share is one linear crouton range and the pack becomes
    `hmx_copy_croutons`. Same destination, same crouton range, same bytes the
    array reads; the A/B gate is bit-identity, as the weight arm's is. The
    flag is a comptime parameter with no build define behind it on purpose: a
    skel-wide define is the wrong scope for a per-buffer property (KB
    `known-limitations/weights-native-is-skel-wide-but-repackedness-is-per-weight`),
    and no shipped instantiation sets it.

    Parameters:
        act_conv: Whether `A` is a convolution's input map instead of an
            activation matrix.
        activation_native: Whether `A` is already a row-panel-major crouton
            artefact, so this stage is a linear copy rather than a pack.
    """
    comptime assert not act_conv or dtype != HMX_INT8_DTYPE, (
        "the conv arm is fp16: an int8 conv would have to quantize inside the"
        " gather, which is a pack policy this seam does not carry"
    )
    comptime assert not (activation_native and act_conv), (
        "a convolution's input map is not croutons: the two source policies"
        " name different byte meanings for the same `A` and cannot both hold"
    )
    comptime assert not activation_native or dtype != HMX_INT8_DTYPE, (
        "the int8 arm quantizes inside the pack; a pre-quantized crouton"
        " activation artefact is a policy this seam does not carry yet"
    )
    comptime if activation_native:
        hmx_copy_croutons(
            dst, hmx_native_row_panel[dtype](A, row0, col0, tiles), tiles
        )
    elif act_conv:
        hmx_conv_pack_act[space](dst, A, conv, row0, col0, tiles)
    elif dtype == HMX_INT8_DTYPE:
        hmx_int8_pack_act(dst, A, row0, col0, tiles, hmx_int8_read_qp(B))
    else:
        hmx_pack_panel[False, space](dst, A, row0, col0, tiles)


@always_inline
def _hmx_arm_unpack[
    dtype: DType, space: AddressSpace, //, output_native: Bool = False
](
    C: MMTile,
    B: MMTile,
    src: CroutonPanelOf[dtype, space],
    row0: Int,
    col0: Int,
) -> UInt32:
    """Writes one accumulator readout tile back to `C`, for any arm.

    The integer arm's readout is already requanted to uint8 by the store, so
    this is the dequantization and NOT a second requant: `(out - out_zp) *
    out_scale`. It is the exact inverse of the pack's model, which is what makes
    the twin's fp32 comparison a statement about quantization error alone.

    `output_native` is `activation_native`'s mirror (annihilation rung 1): `C`'s
    bytes are row-panel-major croutons behind the same `[M, N]` view, and the
    accumulator tile is already in exactly that order — the readout rule IS the
    activation rule over `C[M, N]` (`hmx_unpack_panel`'s own docstring) — so the
    scatter becomes a one-crouton linear copy and no unpack arm runs at all.
    A matmul whose `C` is native and whose consumer's `A` is native hands the
    tile through with zero transforms, which is the rung's whole claim. Like
    `activation_native`, deliberately a bare comptime parameter: no build
    define, no shipped instantiation.

    Parameters:
        output_native: Whether `C` is already a row-panel-major crouton
            artefact, so this readout is a linear copy rather than a scatter.

    Returns:
        The `HMX_UNPACK_ARM_*` code the readout took. The integer arm has no
        vector readout, so it reports `SCALAR_OFF` rather than a fourth code:
        the word answers "did the fp16 vector unpack run", and on this arm the
        answer is no for a reason that is not an admission refusal. The native
        arm reports `NATIVE`: neither scalar code is true of it, and
        misreporting the arm is the class the codes exist to prevent.
    """
    comptime assert not output_native or dtype != HMX_INT8_DTYPE, (
        "the int8 arm dequantizes inside the readout; a uint8 crouton output"
        " artefact is a policy this seam does not carry yet"
    )
    comptime if output_native:
        hmx_copy_croutons(hmx_native_row_panel[dtype](C, row0, col0, 1), src, 1)
        return HMX_UNPACK_ARM_NATIVE
    elif dtype == HMX_INT8_DTYPE:
        hmx_int8_unpack_out(C, src, row0, col0, hmx_int8_read_qp(B))
        return HMX_UNPACK_ARM_SCALAR_OFF

    return hmx_unpack_panel[False, space](C, src, row0, col0, 1)


@always_inline
def _hmx_weight_source[
    dtype: DType
](B: MMTile, col_tile: Int, tile0: Int, tiles: Int) -> CroutonPanelOf[
    dtype, AddressSpace.GENERIC
]:
    """The artefact croutons one arm's weight panel is staged from.

    The two artefact readers differ in the crouton SIZE and in nothing else --
    both name column panel `col_tile` at crouton `tile0` in the same
    column-panel-major order -- so this is where the arm is chosen and no other
    line downstream mentions a dtype.
    """
    comptime if dtype == HMX_INT8_DTYPE:
        return hmx_int8_weight_panel[dtype](B, col_tile, tile0, tiles)
    else:
        return hmx_native_weight_panel[dtype](B, col_tile, tile0, tiles)


@always_inline
def _hmx_stage_weight[
    space: AddressSpace,
    //,
    dtype: DType,
    weight_native: Bool,
    weight_xfer: Int,
    await_xfer: Bool,
](
    arena: HmxStripeArena[space, dtype],
    B: MMTile,
    col_tile: Int,
    t0: Int,
    t1: Int,
    thread_idx: Int,
):
    """Puts output column tile `col_tile`'s weight panel in its ring slot.

    The one place the two weight arms differ, and they differ only in HOW the
    croutons arrive: `weight_native` starts a transfer out of the artefact and
    then WAITS FOR IT, the other arm gathers them out of the `[N, K]` original.
    The destination, the crouton share `[t0, t1)` and therefore the bytes the
    array later reads are the same either way, which is what keeps the A/B a
    bit-identity gate (P5 M4.3).

    The wait is the correction order 100 bought. A packing feeder is finished
    when its stores retire; a transferring one is not finished when `dmstart`
    returns, and the rendezvous that follows orders thread ARRIVAL rather than
    delivery. `await_xfer` is a parameter only so a test can withhold it: with it
    off and the twin's deferred transport, the differential must fail, which is
    what makes the wait a tested requirement instead of a described one.

    A feeder's share can be empty when `k_tiles < feeders`; an empty share
    returns rather than submitting a zero-length transfer.

    THE W8A8 ARM HAS NO PACKING SIBLING, and that is a property rather than a
    gap: an int8 weight does not exist anywhere but in the artefact, because the
    quantization is what the repack DID. So the integer arm implies
    `weight_native`, the comptime assert below says so, and there is no arm in
    which the feeders could quantize per dispatch and report it as the offline
    one's numbers.

    Parameters:
        dtype: The arm's operand type, which changes only WHICH artefact view
            the croutons come from -- the destination, the share and the
            transport are the same code.
        weight_native: Whether `B` is a repacked column-panel-major artefact.
        weight_xfer: Which transport moves it (`HMX_XFER_*`). Only meaningful
            under `weight_native`.
        await_xfer: Whether the transfer is waited for before returning. TRUE
            everywhere but the falsifying control and the
            `HMX_WGT_DEFER_AWAIT` arm, whose caller awaits via
            `_hmx_await_weight` after the step's unpack work instead.

    Args:
        arena: The stripe's residency.
        B: The `(N, K)` weight, row-major or repacked per `weight_native`.
        col_tile: The output column tile whose panel this is.
        t0: First crouton of this thread's share.
        t1: One past the last crouton of this thread's share.
        thread_idx: The launch index, which selects the DMA descriptor slot.
    """
    comptime assert dtype != HMX_INT8_DTYPE or weight_native, (
        "the W8A8 arm reads a repacked int8 artefact; there is no row-major"
        " int8 weight for the feeders to gather"
    )

    if t1 <= t0:
        return
    var dst = hmx_subpanel(arena.wgt_panel(col_tile), t0, t1 - t0)
    comptime if weight_native:
        var src = _hmx_weight_source[dtype](B, col_tile, t0, t1 - t0)
        hmx_stream_weight_submit[weight_xfer](dst, src, t1 - t0, thread_idx)
        comptime if await_xfer:
            # Split out from the submit because the await is the one phase no
            # instrument has ever bracketed, and it is the leading candidate for
            # the ~70% of a vector-arm feeder's cycles order 112's tables leave
            # unaccounted. The mark returns the caller to `STAGE_WGT`, which its
            # own exit mark then closes.
            hmx_phase_mark(thread_idx, HMX_PHASE_WGT_AWAIT)
            hmx_stream_weight_await[weight_xfer](dst, src, t1 - t0, thread_idx)
            hmx_phase_mark(thread_idx, HMX_PHASE_STAGE_WGT)
    else:
        hmx_pack_panel[True, space](
            dst, B, col_tile * HMX_TILE, t0 * HMX_TILE, t1 - t0
        )


@always_inline
def _hmx_await_weight[
    space: AddressSpace,
    //,
    dtype: DType,
    weight_xfer: Int,
](
    arena: HmxStripeArena[space, dtype],
    B: MMTile,
    col_tile: Int,
    t0: Int,
    t1: Int,
    thread_idx: Int,
):
    """Awaits the transfer `_hmx_stage_weight` submitted for `col_tile`.

    The deferred half of the stage under `HMX_WGT_DEFER_AWAIT`: the submit ran
    before the step's unpack work, this runs after it, and the pair brackets
    the same descriptor because both recompute the same subpanel from the same
    share — `wgt_panel(col_tile)` is deterministic and `thread_idx` selects
    the same descriptor slot. Only the transferring arm has anything to await;
    the caller's `comptime` guard keeps the gather arm away from it.
    """
    if t1 <= t0:
        return
    var dst = hmx_subpanel(arena.wgt_panel(col_tile), t0, t1 - t0)
    var src = _hmx_weight_source[dtype](B, col_tile, t0, t1 - t0)
    hmx_stream_weight_await[weight_xfer](dst, src, t1 - t0, thread_idx)


@always_inline
def matmul_bt_stripe_hmx_step[
    space: AddressSpace,
    //,
    emit_hmx: Bool = HMX_EMIT_MMA,
    weight_native: Bool = HMX_WEIGHTS_NATIVE,
    weight_xfer: Int = HMX_XFER_DMA,
    weight_await: Bool = True,
    dtype: DType = HMX_DTYPE,
    record_late: Bool = False,
    act_conv: Bool = False,
    activation_native: Bool = False,
    output_native: Bool = False,
    fold_trailing: Bool = HMX_FOLD_TRAILING,
    fold_early: Bool = False,
](
    C: MMTile,
    A: MMTile,
    B: MMTile,
    arena: HmxStripeArena[space, dtype],
    plan: HmxStripePlan,
    step: Int,
    thread_idx: Int,
    feeders: Int,
    mut unpack_arm: UInt32,
    conv: HmxConvGeom = HmxConvGeom(),
):
    """One thread's work in one step of the pipelined stripe.

    THE WHOLE SCHEDULE IS HERE, which is the point: the caller is a loop that
    calls this `plan.steps()` times with a barrier after each, and the host twin
    is the same loop calling it for every `thread_idx` before advancing. Neither
    driver knows what a step contains, so the two cannot drift.

    Within a block, step `phase = 0` stages the block's activation panels and the
    weight ring's prologue; step `phase = j + 1` runs column tile `j` on the
    array while the feeders stage column tile `j + HMX_WGT_RING - 1` and write
    out column tile `j - 1`. The buffers make those disjoint: at step `j` the
    array reads weight slot `j % HMX_WGT_RING` and writes readout set `j & 1`,
    while the feeders write weight slot `(j - 1) % HMX_WGT_RING` and read
    readout set `(j - 1) & 1`. Neither slot the array is using is touched.

    Under `fold_trailing` the drain step (`phase = n_tiles + 1`) survives only
    for the LAST block; every other block's last-column unpack runs at the top
    of the NEXT block's phase 0 instead, before that block's pack and
    prologue. Both merged halves are feeder-only, and the rendezvous on either
    side of the merged step carry every ordering the deleted one did
    (`HMX_FOLD_TRAILING`'s docstring holds the trace). The readout, act and
    weight regions are distinct arena carves and the two blocks' C and A rows
    are distinct DDR, so the merged step stays destination-disjoint per feeder
    with the same shares.

    The weight side is where P5 M4.3 lands: `weight_native` swaps the feeders'
    transposing scalar gather for a linear transfer out of a host-repacked
    artefact, and nothing else about the step moves. Every output tile still
    reaches `hmx_mma_panel` as one `HmxScratch` with the same `k_tiles` and the
    same chunk loop, so the arms are bit-identical by construction.

    W8A8 IS A LOADER AND RECORD POLICY, NOT A SECOND KERNEL. `dtype` changes
    the operand type of every panel and therefore which four calls it makes --
    a quantizing activation pack, an int8 artefact read, the integer MMA, and a
    dequantizing readout -- and changes NOTHING about the schedule: the step
    partition, the ring modulus, the work shares, the readout double buffer and
    the rendezvous count are the same lines for both arms. The one thing the
    integer arm adds is `hmx_int8_load_record` at the top of the array's phase,
    which is the order-118 rule made structural: the per-channel requant record
    for column tile `j` is in the file before the first store of step `j` reads
    through it.

    A CONVOLUTION IS THE SAME STATEMENT ONE AXIS OVER: `act_conv` changes only
    where the activation panels come from -- an NHWC input map gathered in the
    fused k order rather than a materialised `[M, K]` -- and, like the integer
    arm, changes nothing about the schedule. The `[M, K]` im2col is never built,
    so the arm's cost is the input map rather than its `R * S`-fold expansion,
    and `hmx_prefetch_rows` is the one line that has to know: it hints
    activation ROWS, which under `act_conv` are input-map rows the pack does not
    walk in that order.

    Parameters:
        emit_hmx: Whether the microtiles issue HMX or model it.
        weight_native: Whether `B` is a repacked column-panel-major artefact
            instead of the row-major `[N, K]` original.
        weight_xfer: Which transport stages a native weight panel
            (`HMX_XFER_*`); the twin's arms exist for the differential.
        weight_await: Whether a staged transfer is waited for before the
            rendezvous. Only a falsifying control ever sets this False.
        dtype: The arm's crouton operand type, and the ONLY parameter that
            distinguishes the two arms. `HMX_DTYPE` is the fp16 arm;
            `DType.uint8` is W8A8, under which `B` is a quantized artefact and
            `weight_native` is implied. It is a plain parameter rather than a
            conditional alias so both arms instantiate from one build, which is
            what lets the host twin score them against each other.
        record_late: Whether the requant record is loaded AFTER the tile's MMAs
            instead of before. Only a falsifying control sets this True; it is
            the order-118 mutation, and the differential must fail under it.
        act_conv: Whether `A` is a convolution's NHWC input map instead of an
            activation matrix. fp16 only.
        activation_native: Whether `A` is already a row-panel-major crouton
            artefact (annihilation rung 1), making `PACK_ACT` a linear copy.
            fp16 only; excludes `act_conv`. Deliberately no build define:
            the property is per buffer, not per skel, and no shipped
            instantiation sets it.
        output_native: Whether `C` is a row-panel-major crouton artefact,
            making `UNPACK` a one-crouton linear copy that reports
            `HMX_UNPACK_ARM_NATIVE`. Same scope rules as `activation_native`.
        fold_trailing: Whether the per-block trailing unpack folds into the
            next block's stage step (R4a). The caller's loop bound must be
            `plan.steps[fold_trailing]()`, which the default keeps true for
            every caller that reads both from the build knob.
        fold_early: Wrong-fold mutation: the folded unpack runs one step
            EARLY, inside the last column step, reading the readout set the
            array is writing that same step. Only a falsifying control sets
            this True; the differential must fail under it in the
            feeders-first order.

    Args:
        C: The `(M, N)` output.
        A: The `(M, K)` activations, or under `act_conv` the `(H, W * C_in)`
            input map.
        B: The `(N, K)` transposed weights.
        arena: The stripe's residency, sized for `plan.act_slots` panels.
        plan: The geometry every thread agreed on.
        step: Which of `plan.steps()` steps this is.
        thread_idx: This thread's launch index.
        feeders: Threads that pack and unpack; `thread_idx >= feeders` is the
            one thread holding the HMX grant, and it only multiplies.
        unpack_arm: Set to the `HMX_UNPACK_ARM_*` code of this step's readout,
            for the skel to carry into its witness. A `mut` argument rather than
            a return because this function's five early exits are steps that
            unpack NOTHING, and a return would need a fourth code for them --
            leaving it untouched says the same thing without one. The caller
            seeds it and every writing step agrees, so the last write across a
            dispatch is the arm that ran.
        conv: The convolution's window, padding and extents. Read only under
            `act_conv`; the default is the inert descriptor.
    """
    comptime assert (
        not MM_PATH_HMX or emit_hmx
    ), "MM_PATH_HMX needs HMX_EMIT_MMA: the modelled MMA is the host twin's"
    comptime assert not fold_early or fold_trailing, (
        "fold_early mutates the folded schedule; it has no meaning without"
        " fold_trailing"
    )

    var per_block = plan.n_tiles + (1 if fold_trailing else 2)
    var block = step // per_block
    var phase = step - block * per_block
    comptime if fold_trailing:
        # The one drain step the fold keeps: the last block's, decoded from
        # the `+ 1` past the folded blocks so the arithmetic stays a pure
        # function of the plan.
        if block >= plan.blocks:
            block = plan.blocks - 1
            phase = plan.n_tiles + 1
    var p0 = block * plan.act_slots
    var panels = plan.m_tiles - p0
    if panels > plan.act_slots:
        panels = plan.act_slots
    if panels <= 0:
        return

    if thread_idx >= feeders:
        # The array's two outcomes are marked apart on purpose: `IDLE` is the
        # schedule giving it nothing to do, `MMA` is it working, and the stall
        # diagnosis is exactly the question of which of those the 95-99.6% is
        # (`.agentwork/flux-npu/ARRAY_STALL_DIAGNOSIS.md` §1a).
        if phase == 0:
            hmx_phase_mark(thread_idx, HMX_PHASE_IDLE)
            return
        var j = phase - 1
        if j >= plan.n_tiles:
            hmx_phase_mark(thread_idx, HMX_PHASE_IDLE)
            return
        hmx_phase_mark(thread_idx, HMX_PHASE_MMA)
        comptime if dtype == HMX_INT8_DTYPE:
            # The order-118 rule, made structural: column tile `j`'s
            # per-channel record is in the requant file before the first store
            # of step `j` reads through it. `record_late` is the mutation, and
            # the twin's control asserts the differential fails under it.
            comptime if not record_late:
                hmx_int8_load_record(arena.record, B, j)
            for p in range(panels):
                hmx_int8_mma_panel[emit_hmx](
                    arena.tile_scratch(j, p), plan.k_tiles
                )
            comptime if record_late:
                hmx_int8_load_record(arena.record, B, j)
        else:
            comptime if HMX_MMA_SPLIT:
                # The split reuses feeder-only bucket ids on this thread; the
                # MMA mark above still opens the span, and OTHER below still
                # closes the last one (the readout).
                for p in range(panels):
                    hmx_mma_panel_split[emit_hmx](
                        arena.tile_scratch(j, p), plan.k_tiles, thread_idx
                    )
            else:
                for p in range(panels):
                    hmx_mma_panel[emit_hmx](
                        arena.tile_scratch(j, p), plan.k_tiles
                    )
        hmx_phase_mark(thread_idx, HMX_PHASE_OTHER)
        return

    # A feeder's share of a panel is a crouton range, so its destination is a
    # subpanel and its source column follows from the same range — the packers
    # already take both, and neither learns that a panel was split.
    var t0 = hmx_work_share(plan.k_tiles, thread_idx, feeders)
    var t1 = hmx_work_share(plan.k_tiles, thread_idx + 1, feeders)

    if phase == 0:
        comptime if fold_trailing and not fold_early:
            if block >= 1:
                # The folded drain: the PREVIOUS block's last column, whose
                # readout the array finished one rendezvous ago, is written
                # out here ahead of this block's pack — prev-block `p0` and
                # `panels` recomputed from the plan, so the arithmetic stays
                # thread-uniform. `_hmx_arm_unpack` and the shares are the
                # drain step's verbatim.
                comptime if HMX_ACT_XBLOCK_PF:
                    comptime if not act_conv and not activation_native:
                        # The drain step this unpack rode in was also the
                        # cross-block hint's home; transposed with it, and
                        # issued first so the unpack below is the hint's
                        # lead distance.
                        hmx_prefetch_rows(
                            A, p0 * HMX_TILE, t0 * HMX_TILE, HMX_TILE, t1 - t0
                        )
                var fp0 = (block - 1) * plan.act_slots
                var fpanels = plan.m_tiles - fp0
                if fpanels > plan.act_slots:
                    fpanels = plan.act_slots
                var q0 = hmx_work_share(fpanels, thread_idx, feeders)
                var q1 = hmx_work_share(fpanels, thread_idx + 1, feeders)
                hmx_phase_mark(thread_idx, HMX_PHASE_UNPACK)
                for p in range(q0, q1):
                    unpack_arm = _hmx_arm_unpack[output_native=output_native](
                        C,
                        B,
                        arena.out_tile((plan.n_tiles - 1) & 1, p),
                        (fp0 + p) * HMX_TILE,
                        (plan.n_tiles - 1) * HMX_TILE,
                    )
        hmx_phase_mark(thread_idx, HMX_PHASE_PACK_ACT)
        for p in range(panels):
            # The next panel's rows, and only the next panel's: this feeder is
            # about to walk `p` a scalar element at a time, which is the whole
            # of the lead the hint needs, and the `MATMUL_PF_B_ROWS` sweep
            # recorded in `npu_kernel.mojo` gives the gain back when an issue
            # covers more than its consumer's next read. The share `[t0, t1)` is
            # this thread's own, so four feeders hint four disjoint regions.
            #
            # Withheld under `act_conv` rather than translated: the hint names
            # a rectangle of `A`, and the conv arm's `A` is an input map whose
            # rows are not the panel's rows. Hinting the wrong rectangle is
            # worse than hinting none -- it is bandwidth spent on lines the
            # pack will not read (order 097's `MATMUL_PF_B_ROWS` sweep is what
            # priced that). The map-shaped hint is a follow-up, not a silent
            # reinterpretation of this one. Withheld under `activation_native`
            # for the same reason: the native `A`'s rows are crouton runs, not
            # the row-major rectangle the hint would name.
            if p + 1 < panels:
                comptime if not act_conv and not activation_native:
                    hmx_prefetch_rows(
                        A,
                        (p0 + p + 1) * HMX_TILE,
                        t0 * HMX_TILE,
                        HMX_TILE,
                        t1 - t0,
                    )
            _hmx_arm_pack_act[
                act_conv=act_conv, activation_native=activation_native
            ](
                hmx_subpanel(arena.act_panel(p), t0, t1 - t0),
                A,
                B,
                (p0 + p) * HMX_TILE,
                t0 * HMX_TILE,
                t1 - t0,
                conv,
            )
        # The ring's prologue: every slot but the one the array will free at the
        # end of the first compute phase. At depth 2 -- which order 096 leg A'
        # makes the ceiling, not the starting point -- that is column tile 0,
        # and the schedule below is P5 M4.2's double buffer unchanged.
        hmx_phase_mark(thread_idx, HMX_PHASE_STAGE_WGT)
        for ct in range(HMX_WGT_RING - 1):
            if ct < plan.n_tiles:
                _hmx_stage_weight[
                    dtype, weight_native, weight_xfer, weight_await
                ](arena, B, ct, t0, t1, thread_idx)
        hmx_phase_mark(thread_idx, HMX_PHASE_OTHER)
        return

    var j = phase - 1
    # Steady state: stage the panel `HMX_WGT_RING - 1` column tiles ahead, which
    # lands in the slot the array last read at the PREVIOUS phase and is done
    # with. The slot the array is reading right now is `j % HMX_WGT_RING`, and
    # `(j + HMX_WGT_RING - 1) % HMX_WGT_RING` is never it for a depth of two or
    # more — that disjointness is what the rendezvous makes safe rather than
    # lucky.
    #
    # Under `HMX_WGT_DEFER_AWAIT` the stage SUBMITS here and the await moves
    # below the unpack loop, so delivery flies behind the step's largest phase
    # instead of stalling the feeder at the submit site. The deferral applies
    # only where a transfer exists (`weight_native`) and never under the
    # falsifying control (`weight_await=False` still means no await anywhere).
    comptime defer_await = (
        HMX_WGT_DEFER_AWAIT and weight_native and weight_await
    )
    var ahead = j + HMX_WGT_RING - 1
    var staged = ahead < plan.n_tiles
    if staged:
        hmx_phase_mark(thread_idx, HMX_PHASE_STAGE_WGT)
        _hmx_stage_weight[
            dtype, weight_native, weight_xfer, weight_await and not defer_await
        ](arena, B, ahead, t0, t1, thread_idx)
        hmx_phase_mark(thread_idx, HMX_PHASE_OTHER)
    if j >= 1:
        var q0 = hmx_work_share(panels, thread_idx, feeders)
        var q1 = hmx_work_share(panels, thread_idx + 1, feeders)
        hmx_phase_mark(thread_idx, HMX_PHASE_UNPACK)
        for p in range(q0, q1):
            unpack_arm = _hmx_arm_unpack[output_native=output_native](
                C,
                B,
                arena.out_tile((j - 1) & 1, p),
                (p0 + p) * HMX_TILE,
                (j - 1) * HMX_TILE,
            )
        hmx_phase_mark(thread_idx, HMX_PHASE_OTHER)
    comptime if fold_trailing and fold_early:
        # The wrong fold, for the twin only: this column's OWN readout, one
        # step before the fold's legal spot — the set the array is writing in
        # this very step, which the feeders-first sweep must catch as stale.
        if j == plan.n_tiles - 1:
            var m0 = hmx_work_share(panels, thread_idx, feeders)
            var m1 = hmx_work_share(panels, thread_idx + 1, feeders)
            for p in range(m0, m1):
                unpack_arm = _hmx_arm_unpack[output_native=output_native](
                    C,
                    B,
                    arena.out_tile(j & 1, p),
                    (p0 + p) * HMX_TILE,
                    j * HMX_TILE,
                )

    # The one panel the in-block hint cannot reach is the NEXT block's panel
    # 0 — phase 0 hints `p + 1` while packing `p`, so every block's first
    # panel walks cold lines. The drain phase (`j == plan.n_tiles`) has no
    # weight panel left to stage, so the feeder's spare tail hints that
    # panel for its own crouton share: one hint per feeder per block, the
    # order-097 one-panel-per-issue distance, a block earlier in time. A
    # hint moves cache lines and never data (the `HMX_ACT_PREFETCH`
    # contract), and `hmx_prefetch_rows` self-gates off that master flag.
    # Withheld under `act_conv` / `activation_native` for the in-block
    # hint's own reasons: the rectangle would name rows the pack does not
    # walk.
    comptime if HMX_ACT_XBLOCK_PF:
        comptime if not act_conv and not activation_native:
            if j == plan.n_tiles:
                var np0 = (block + 1) * plan.act_slots
                if np0 < plan.m_tiles:
                    hmx_prefetch_rows(
                        A,
                        np0 * HMX_TILE,
                        t0 * HMX_TILE,
                        HMX_TILE,
                        t1 - t0,
                    )
    # The deferred await: the transfer has had the whole UNPACK phase to land.
    # It still precedes the rendezvous (order 100: the barrier orders arrival,
    # not delivery), still precedes the next submit (order 096: one
    # outstanding transfer per submitter), and still runs on the thread that
    # issued it (`dmwait` is per-engine). `HMX_PHASE_WGT_AWAIT` brackets
    # exactly the exposed wait, so the phase tables stay comparable across
    # arms.
    comptime if defer_await:
        if staged:
            hmx_phase_mark(thread_idx, HMX_PHASE_WGT_AWAIT)
            _hmx_await_weight[dtype, weight_xfer](
                arena, B, ahead, t0, t1, thread_idx
            )
            hmx_phase_mark(thread_idx, HMX_PHASE_OTHER)
