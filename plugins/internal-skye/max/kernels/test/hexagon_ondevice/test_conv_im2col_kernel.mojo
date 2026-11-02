"""Host validation for the FUSED online im2col conv arm (`matmul_kernel.mojo`).

Runs the actual core (`conv_bt`) on the host CPU. The claim under test is the
one the loader-swap seam rests on: that gathering A out of the NHWC input map at
load time computes the same convolution as building the `[M, K]` im2col matrix
first and calling the matmul on it — and that it does so with the SAME index map,
so the two are bit-identical rather than merely close.

Checks per shape:
  1. Bit-exact equivalence to `matmul_bt` over a MATERIALISED im2col matrix built
     in the same `k = (r*S + s)*C_in + c` order. This is the M5's parity gate
     (KB `kernels/apple-conv2d-im2col`) and it is the strongest statement
     available: same k order, same fold, same accumulation, so anything but zero
     difference is a gather bug. It is the check that would fail if the halo
     zero-fill, the pixel anchor, or the carried `(r, s, c0)` state were wrong.
  2. Bit-exact equivalence across the `mc_bytes` sweep. The fused M-tile is sized
     in INPUT-MAP bytes (`mm_conv_tile_rows`), a different formula from the
     matmul's, so the retile is re-proven here rather than inherited: cache
     blocking only reorders independent outputs.
  3. A scalar fp32 convolution oracle written directly from the definition —
     four nested loops over `(r, s, c)` with an explicit halo test, sharing no
     code with the kernel — within tolerance.
  4. Agreement with the CHANNEL-SLOWEST k order both materialising paths in-tree
     use (`conv_im2col.h`, `registrations.mojo`), with the filter permuted to
     match, to a TOLERANCE and not bitwise. This check exists to record the
     numerics caveat rather than to hide it: the reorder permutes which k lands
     in which fp32 lane, so `mm_reduce_add` reassociates. Reported as a measured
     band, so a future change that widens it is visible.
  5. A bounds-guard canary after `C`, as the matmul twin keeps: the fused arm
     must not have introduced a wide store.

And every shape runs twice, under two fills:
  - Pseudo-random fp16, where rounding makes the accumulation ORDER observable,
    so checks 1 and 2 have teeth and check 4's band is real.
  - Small exact integers, where every product and partial sum is exact in fp32,
    so the fp32 oracle must match to ZERO error and a misindexed tap cannot hide
    under a tolerance.

Plus two falsifiable oracles that MUST fire: a shifted input map (which a gather
ignoring the halo would absorb) and a scaled filter.

Run from this directory (imports the sibling core):
  mojo run test_conv_im2col_kernel.mojo
"""
from std.math import max
from std.memory import alloc

from layout.coord import Coord
from layout.tile_layout import row_major

from hexagon_ondevice.matmul_kernel import (
    MM_ACCUM,
    MM_DTYPE,
    MM_MC_BYTES,
    MM_MR,
    MM_NR,
    MM_W,
    MMConv,
    MMImage,
    MMRowAnchor,
    MMTile,
    conv_bt,
    matmul_bt,
    mm_conv_fused_supported,
    mm_conv_tile_rows,
    mm_conv_window_inside,
)

comptime F32 = DType.float32
comptime Ptr = UnsafePointer[Scalar[MM_DTYPE], MutUnsafeAnyOrigin]

comptime TOL = Scalar[F32](0.02)
"""Slack for the fp32 oracle under the random fill; zero under the integer fill.
"""

comptime TOL_REASSOC = Scalar[F32](0.05)
"""Slack for check 4, where the two k orders fold in different lane orders.

Wider than `TOL` on purpose: this is not the kernel disagreeing with an oracle,
it is two legitimate summation orders of the same `R * S * C_in` products. The
check prints the measured band so the number is evidence, not an assumption.
"""

comptime MC_UNTILED = 64 * 1024 * 1024
"""A budget no test input map can fill, so the core runs as one M-tile."""


def fill_random(ptr: Ptr, count: Int, seed: UInt32, scale: Float32):
    """Deterministic xorshift fill in `[-scale, scale)`, rounded to `MM_DTYPE`.
    """
    var rng = seed
    for i in range(count):
        rng ^= rng << 13
        rng ^= rng >> 17
        rng ^= rng << 5
        var u = Float32(rng & 0xFFFF) / 65536.0 * 2.0 - 1.0
        ptr[i] = Scalar[MM_DTYPE](u * scale)


def fill_integers(ptr: Ptr, count: Int, seed: UInt32, bound: Int):
    """Deterministic xorshift fill over the integers in `[-bound, bound]`.

    Small enough that every product and every partial sum of an `R*S*C_in`-long
    dot product is exact in fp32, so the scalar oracle must agree to zero error.
    """
    var rng = seed
    var span = UInt32(2 * bound + 1)
    for i in range(count):
        rng ^= rng << 13
        rng ^= rng >> 17
        rng ^= rng << 5
        ptr[i] = Scalar[MM_DTYPE](Int((rng >> 8) % span) - bound)


def run_fused(
    C: Ptr,
    img: Ptr,
    B: Ptr,
    H: Int,
    W: Int,
    C_in: Int,
    C_out: Int,
    conv: MMConv,
    H_out: Int,
    mc_bytes: Int,
):
    """Drives the fused arm through the seam a C-ABI caller has to build.

    The one place in this file where extents become layouts, mirroring
    `test_matmul_kernel.mojo`'s `run_core`: a transposed extent is a single-site
    bug rather than one repeated per call.
    """
    conv_bt[MM_MR, MM_NR](
        MMTile(ptr=C, layout=row_major(Coord(H_out * conv.W_out, C_out))),
        MMImage[MutUnsafeAnyOrigin](
            ptr=img, layout=row_major(Coord(H, W, C_in))
        ),
        MMTile(ptr=B, layout=row_major(Coord(C_out, conv.R * conv.S * C_in))),
        conv,
        mc_bytes,
    )


def build_im2col(
    A: Ptr,
    img: Ptr,
    H: Int,
    W: Int,
    C_in: Int,
    conv: MMConv,
    H_out: Int,
    channel_fastest: Bool,
):
    """Materialises the `[M, K]` im2col matrix, in either k order.

    `channel_fastest` picks `k = (r*S + s)*C_in + c` (what the fused loader
    walks, and MAX's own GPU im2col) over `k = (c*R + r)*S + s` (what both
    materialising paths in-tree use today). Reference code, written from the
    definition rather than shared with the kernel.
    """
    var K = conv.R * conv.S * C_in
    for h_out in range(H_out):
        for w_out in range(conv.W_out):
            var row = A + (h_out * conv.W_out + w_out) * K
            for r in range(conv.R):
                var h_in = h_out - conv.pad_h + r
                for s in range(conv.S):
                    var w_in = w_out - conv.pad_w + s
                    var inside = 0 <= h_in < H and 0 <= w_in < W
                    for c in range(C_in):
                        var tap = Scalar[MM_DTYPE](0)
                        if inside:
                            tap = img[(h_in * W + w_in) * C_in + c]
                        var k = (r * conv.S + s) * C_in + c
                        if not channel_fastest:
                            k = (c * conv.R + r) * conv.S + s
                        row[k] = tap


def permute_filter(dst: Ptr, src: Ptr, C_in: Int, C_out: Int, conv: MMConv):
    """Rewrites a channel-fastest filter into channel-slowest k order.

    The permutation a caller would apply once at weight load to feed the fused
    arm from an `FCRS` checkpoint; here it runs the other way so check 4 can
    compare the two orders over the same convolution.
    """
    var K = conv.R * conv.S * C_in
    for n in range(C_out):
        for r in range(conv.R):
            for s in range(conv.S):
                for c in range(C_in):
                    dst[n * K + (c * conv.R + r) * conv.S + s] = src[
                        n * K + (r * conv.S + s) * C_in + c
                    ]


def conv_ref_fp32(
    C: Ptr,
    img: Ptr,
    B: Ptr,
    H: Int,
    W: Int,
    C_in: Int,
    C_out: Int,
    conv: MMConv,
    H_out: Int,
):
    """Independent scalar fp32 convolution oracle, straight from the definition.

    Shares no index arithmetic with the kernel or with `build_im2col`: it walks
    `(r, s, c)` directly and tests the halo per tap, so a k-order bug common to
    the kernel and the materialised reference cannot pass unnoticed.
    """
    var K = conv.R * conv.S * C_in
    for h_out in range(H_out):
        for w_out in range(conv.W_out):
            for n in range(C_out):
                var acc = Scalar[F32](0)
                for r in range(conv.R):
                    var h_in = h_out - conv.pad_h + r
                    if h_in < 0 or h_in >= H:
                        continue
                    for s in range(conv.S):
                        var w_in = w_out - conv.pad_w + s
                        if w_in < 0 or w_in >= W:
                            continue
                        for c in range(C_in):
                            acc += (
                                img[(h_in * W + w_in) * C_in + c].cast[F32]()
                                * B[n * K + (r * conv.S + s) * C_in + c].cast[
                                    F32
                                ]()
                            )
                C[(h_out * conv.W_out + w_out) * C_out + n] = acc.cast[
                    MM_DTYPE
                ]()


def run_case(
    H: Int,
    W: Int,
    C_in: Int,
    C_out: Int,
    R: Int,
    S: Int,
    pad_h: Int,
    pad_w: Int,
    name: String,
    integer_inputs: Bool,
) -> Bool:
    """Runs every check at every `mc_bytes` in the sweep for one convolution."""
    comptime PAD = MM_W  # wide enough to catch a 64-lane over-store past C
    var H_out = H + 2 * pad_h - (R - 1)
    var W_out = W + 2 * pad_w - (S - 1)
    var conv = MMConv(R=R, S=S, pad_h=pad_h, pad_w=pad_w, W_out=W_out)
    var M = H_out * W_out
    var K = R * S * C_in

    var img_n = H * W * C_in
    var b_n = C_out * K
    var c_n = M * C_out

    var img = alloc[Scalar[MM_DTYPE]](img_n).as_unsafe_any_origin()
    var B = alloc[Scalar[MM_DTYPE]](b_n).as_unsafe_any_origin()
    var B_slow = alloc[Scalar[MM_DTYPE]](b_n).as_unsafe_any_origin()
    var A = alloc[Scalar[MM_DTYPE]](M * K).as_unsafe_any_origin()
    var C_fused = alloc[Scalar[MM_DTYPE]](c_n + PAD).as_unsafe_any_origin()
    var C_mat = alloc[Scalar[MM_DTYPE]](c_n).as_unsafe_any_origin()
    var C_slow = alloc[Scalar[MM_DTYPE]](c_n).as_unsafe_any_origin()
    var C_ref = alloc[Scalar[MM_DTYPE]](c_n).as_unsafe_any_origin()

    if integer_inputs:
        fill_integers(img, img_n, 0x12345678, 3)
        fill_integers(B, b_n, 0x0BADF00D, 1)
    else:
        fill_random(img, img_n, 0x12345678, 1.0)
        fill_random(B, b_n, 0x0BADF00D, 0.05)

    # Check 1's reference: the same k order the fused loader walks, materialised.
    build_im2col(A, img, H, W, C_in, conv, H_out, True)
    matmul_bt[MM_MR, MM_NR](
        MMTile(ptr=C_mat, layout=row_major(Coord(M, C_out))),
        MMTile(ptr=A, layout=row_major(Coord(M, K))),
        MMTile(ptr=B, layout=row_major(Coord(C_out, K))),
        MC_UNTILED,
    )
    # Check 4's reference: the channel-slowest order both in-tree paths use.
    build_im2col(A, img, H, W, C_in, conv, H_out, False)
    permute_filter(B_slow, B, C_in, C_out, conv)
    matmul_bt[MM_MR, MM_NR](
        MMTile(ptr=C_slow, layout=row_major(Coord(M, C_out))),
        MMTile(ptr=A, layout=row_major(Coord(M, K))),
        MMTile(ptr=B_slow, layout=row_major(Coord(C_out, K))),
        MC_UNTILED,
    )
    conv_ref_fp32(C_ref, img, B, H, W, C_in, C_out, conv, H_out)

    # Budgets chosen so the derived tile height reaches the adversarial cases:
    # one microtile row, a height that does not divide M, and a panel larger
    # than the whole input map.
    var budgets = [
        W * C_in * 2 * R,
        W * C_in * 2 * (R + 1),
        W * C_in * 2 * (R + 3),
        MM_MC_BYTES,
        MC_UNTILED,
    ]

    var bit_vs_materialised = 0
    var max_err = Scalar[F32](0)
    var max_reassoc = Scalar[F32](0)
    var saw_nan = False
    var canary_ok = True
    var tiles_lo = M
    var tiles_hi = 1

    for bi in range(len(budgets)):
        var sentinel = Scalar[MM_DTYPE](-3.5)
        for i in range(c_n + PAD):
            C_fused[i] = sentinel

        run_fused(C_fused, img, B, H, W, C_in, C_out, conv, H_out, budgets[bi])

        var mc = mm_conv_tile_rows[MM_MR](conv, W, C_in, budgets[bi])
        var tiles = (M + mc - 1) // mc
        if tiles < tiles_lo:
            tiles_lo = tiles
        if tiles > tiles_hi:
            tiles_hi = tiles

        for i in range(c_n):
            if C_fused[i] != C_mat[i]:
                bit_vs_materialised += 1
            var v = C_fused[i].cast[F32]()
            if v != v:
                saw_nan = True
            max_err = max(max_err, abs(v - C_ref[i].cast[F32]()))
            max_reassoc = max(max_reassoc, abs(v - C_slow[i].cast[F32]()))
        for i in range(c_n, c_n + PAD):
            if C_fused[i] != sentinel:
                canary_ok = False

    # Which arm of the halo split each full microtile takes. Printed rather
    # than asserted because it is a property of the geometry: a shape whose
    # window never fits legitimately reports 0 interior. What it gates is that
    # the list below reaches BOTH arms — a bit-exact result proves nothing
    # about a path no shape runs.
    var split = count_split(H, W, R, S, pad_h, pad_w, H_out, W_out)

    var err_bound = Scalar[F32](0) if integer_inputs else TOL
    # The integer fill makes every partial sum exact, so the two k orders agree
    # bitwise there too and the reassociation band collapses with it.
    var reassoc_bound = Scalar[F32](0) if integer_inputs else TOL_REASSOC

    var ok = (
        (bit_vs_materialised == 0)
        and (not saw_nan)
        and (max_err <= err_bound)
        and (max_reassoc <= reassoc_bound)
        and canary_ok
    )

    print(
        "PASS" if ok else "FAIL",
        "|",
        name,
        "[int]" if integer_inputs else "[rand]",
        "| HxW=",
        H,
        "x",
        W,
        "C_in=",
        C_in,
        "C_out=",
        C_out,
        "RxS=",
        R,
        "x",
        S,
        "| M=",
        M,
        "K=",
        K,
        "| tiles=",
        tiles_lo,
        "..",
        tiles_hi,
        "| split=",
        split[0],
        "int/",
        split[1],
        "bord | vs_materialised=",
        bit_vs_materialised,
        "| max_err=",
        max_err,
        "<=",
        err_bound,
        "| reassoc=",
        max_reassoc,
        "<=",
        reassoc_bound,
        "| nan=",
        saw_nan,
        "| canary_ok=",
        canary_ok,
    )

    img.free()
    B.free()
    B_slow.free()
    A.free()
    C_fused.free()
    C_mat.free()
    C_slow.free()
    C_ref.free()
    return ok


def check_conv_tile_rows() -> Bool:
    """`mm_conv_tile_rows` must return a positive multiple of `MM_MR`, always.

    Check 2's bit-exactness argument rests on every M-tile starting on a multiple
    of `MM_MR`, exactly as the matmul's does. The sweep includes budgets far too
    small to hold even the filter's `R` input rows, which is where a formula that
    subtracts `R - 1` before clamping would go negative.
    """
    var widths = [16, 64, 129, 512]
    var channels = [64, 128, 512]
    var budgets = [1, 63, 127, 512, 1000, 40960, MM_MC_BYTES, MC_UNTILED]
    var Rs = [1, 3, 5]
    var bad = 0
    for wi in range(len(widths)):
        for ci in range(len(channels)):
            for ri in range(len(Rs)):
                for bj in range(len(budgets)):
                    var conv = MMConv(
                        R=Rs[ri],
                        S=Rs[ri],
                        pad_h=Rs[ri] // 2,
                        pad_w=Rs[ri] // 2,
                        W_out=widths[wi],
                    )
                    var rows = mm_conv_tile_rows[MM_MR](
                        conv, widths[wi], channels[ci], budgets[bj]
                    )
                    if rows < MM_MR or rows % MM_MR != 0:
                        bad += 1
    print(
        "PASS" if bad == 0 else "FAIL",
        "| fused tile height is always a positive multiple of MR=",
        MM_MR,
        "| violations=",
        bad,
    )
    return bad == 0


def check_conv_tile_rows_no_underflow() -> Bool:
    """`mm_conv_tile_rows` must not clamp to the `MM_MR` floor on the four
    Klein convolutions whose input row is exactly `MM_MC_BYTES / 2`.

    `VAE_DEVICE_CRITIQUE.md` section 4 and `tools/vae_roofline.py` name these
    by `(W, C_in)`: `up1.resize_conv` / `up2.resnet0.conv1` at
    `(W=256, C_in=512)` and `up2.resize_conv` / `up3.resnet0.conv1` at
    `(W=512, C_in=256)`, all `R=S=3`. Before the fix,
    `MM_MC_BYTES // in_row_bytes - (R - 1)` evaluates to `2 - 2 = 0`, and the
    tile silently clamps to `MM_MR` (4 output pixels) instead of the one
    output row (`W_out`) the residency budget actually supports. `check_conv_tile_rows`
    above does not catch this: 4 is itself a positive multiple of `MM_MR`, so
    the floor is a valid-looking return, not a violation of that property.
    This check pins the VALUE, not just its shape, at the production budget.
    """
    var floor_shapes = [(256, 512), (512, 256)]
    var bad = 0
    for i in range(len(floor_shapes)):
        var W = floor_shapes[i][0]
        var C_in = floor_shapes[i][1]
        var in_row_bytes = W * C_in * 2  # MM_DTYPE is fp16, 2 bytes.
        if in_row_bytes != MM_MC_BYTES // 2:
            bad += 1  # The premise moved; this shape no longer sits on the underflow.
            continue
        var conv = MMConv(R=3, S=3, pad_h=1, pad_w=1, W_out=W)
        var rows = mm_conv_tile_rows[MM_MR](conv, W, C_in, MM_MC_BYTES)
        if rows != W:
            bad += 1
    print(
        "PASS" if bad == 0 else "FAIL",
        "| fused tile height is one full output row (not the MR=",
        MM_MR,
        "floor) on the four census floor shapes | violations=",
        bad,
    )
    return bad == 0


def check_window_inside() -> Bool:
    """`mm_conv_window_inside` must never claim a window fits when it does not.

    This is the gate on the one new way the fused arm can be WRONG rather than
    slow. Where the predicate says True, `conv_bt_stripe` runs a reader with the
    halo test compiled out, which reads the map unconditionally — so a single
    False positive is an out-of-bounds read and a wrong output, not a tolerance
    excursion. The property is therefore soundness, not agreement: `True` must
    imply every tap is inside.

    `False` where the window does fit is legal and merely costs speed, so the
    conservatism is printed rather than failed. It is not zero, and the shape it
    takes is known: a span that wraps onto the next output row is refused even
    when its taps are all in bounds, which happens when a filter has no column
    halo at all (`pad_w = 0`, `S = 1`). The refusal is bounded by the last
    `rows - 1` pixels of an output row, so on a width that is a multiple of
    `MM_MR` — every VAE decoder layer — it costs nothing at all.

    Brute force is the whole `rows x R x S` tap set of the microtile against the
    map's extents, over geometries that include padded and unpadded windows, an
    even filter (so `pad != R // 2`), and a map narrower than one microtile.
    """
    var extents = [3, 4, 5, 8, 13]
    var taps = [1, 2, 3, 5]
    var spans = [1, 2, MM_MR, MM_MR + 1]
    var unsound = 0
    var conservative = 0
    var claimed = 0
    for hi in range(len(extents)):
        for wi in range(len(extents)):
            var H = extents[hi]
            var W = extents[wi]
            for ti in range(len(taps)):
                var R = taps[ti]
                var S = taps[ti]
                if R > H or S > W:
                    continue
                for pad in range(R):
                    var W_out = W + 2 * pad - (S - 1)
                    var H_out = H + 2 * pad - (R - 1)
                    if W_out < 1 or H_out < 1:
                        continue
                    var conv = MMConv(
                        R=R, S=S, pad_h=pad, pad_w=pad, W_out=W_out
                    )
                    for si in range(len(spans)):
                        var span = spans[si]
                        for m in range(H_out * W_out - span + 1):
                            var h_out = m // W_out
                            var anchor = MMRowAnchor(
                                row=h_out - pad,
                                col=m - h_out * W_out - pad,
                            )
                            var truth = True
                            for i in range(span):
                                var mi = m + i
                                var hi_out = mi // W_out
                                var wi_out = mi - hi_out * W_out
                                for r in range(R):
                                    for s in range(S):
                                        var y = hi_out - pad + r
                                        var x = wi_out - pad + s
                                        if not (0 <= y < H and 0 <= x < W):
                                            truth = False
                            var got = mm_conv_window_inside(
                                anchor, span, conv, H, W
                            )
                            if got:
                                claimed += 1
                                if not truth:
                                    unsound += 1
                            elif truth:
                                conservative += 1
    print(
        "PASS" if unsound == 0 else "FAIL",
        (
            "| halo-free predicate never claims an out-of-bounds window |"
            " false positives="
        ),
        unsound,
        "| claimed=",
        claimed,
        "| conservative refusals=",
        conservative,
    )
    return unsound == 0


def count_split(
    H: Int,
    W: Int,
    R: Int,
    S: Int,
    pad_h: Int,
    pad_w: Int,
    H_out: Int,
    W_out: Int,
) -> Tuple[Int, Int]:
    """Full `MM_MR`-row microtiles this geometry sends to each arm.

    `run_case` asserts both are non-zero somewhere in its shape list, because a
    bit-exact result proves nothing about a path no shape reaches. The split is
    a property of the geometry alone — the walk visits every `MM_MR`-aligned
    `m` — so it can be recomputed here without instrumenting the kernel.
    """
    var conv = MMConv(R=R, S=S, pad_h=pad_h, pad_w=pad_w, W_out=W_out)
    var interior = 0
    var border = 0
    var m = 0
    while m + MM_MR <= H_out * W_out:
        var h_out = m // W_out
        var anchor = MMRowAnchor(
            row=h_out - pad_h, col=m - h_out * W_out - pad_w
        )
        if mm_conv_window_inside(anchor, MM_MR, conv, H, W):
            interior += 1
        else:
            border += 1
        m += MM_MR
    return (interior, border)


def check_support_gate() -> Bool:
    """The fused arm must refuse exactly the channel counts it cannot serve.

    `C_in % MM_W == 0` is what makes a k-vector sit inside one channel run. The
    decoder's `conv_in` (C_in = 32) is the in-tree case that must be refused.
    """
    var accepted = [MM_W, 2 * MM_W, 3 * MM_W, 4 * MM_W, 8 * MM_W]
    var refused = [0, 1, 3, 32, 63, 65, 96, 160]
    var bad = 0
    for i in range(len(accepted)):
        if not mm_conv_fused_supported(accepted[i]):
            bad += 1
    for i in range(len(refused)):
        if mm_conv_fused_supported(refused[i]):
            bad += 1
    print(
        "PASS" if bad == 0 else "FAIL",
        "| fused-arm channel gate accepts multiples of",
        MM_W,
        "and nothing else | violations=",
        bad,
    )
    return bad == 0


def oracle_shifted_input(H: Int, W: Int, C_in: Int, C_out: Int) -> Bool:
    """Falsifiable control: the halo must matter.

    A gather that clamped instead of zero-filling, or that dropped the padding
    offset, would still produce a plausible convolution. Rotating the input map
    by one row changes the answer at every pixel, so the bit-exact check must
    report a difference — if it does not, check 1 is a placebo.

    Returns True when the control fires.
    """
    var conv = MMConv(R=3, S=3, pad_h=1, pad_w=1, W_out=W)
    var H_out = H
    var M = H_out * W
    var K = 9 * C_in
    var img_n = H * W * C_in
    var c_n = M * C_out

    var img = alloc[Scalar[MM_DTYPE]](img_n).as_unsafe_any_origin()
    var img2 = alloc[Scalar[MM_DTYPE]](img_n).as_unsafe_any_origin()
    var B = alloc[Scalar[MM_DTYPE]](C_out * K).as_unsafe_any_origin()
    var C_a = alloc[Scalar[MM_DTYPE]](c_n).as_unsafe_any_origin()
    var C_b = alloc[Scalar[MM_DTYPE]](c_n).as_unsafe_any_origin()

    fill_random(img, img_n, 0x2468ACE0, 1.0)
    fill_random(B, C_out * K, 0x13579BDF, 0.05)
    var row = W * C_in
    for i in range(img_n):
        img2[i] = img[(i + row) % img_n]

    run_fused(C_a, img, B, H, W, C_in, C_out, conv, H_out, MM_MC_BYTES)
    run_fused(C_b, img2, B, H, W, C_in, C_out, conv, H_out, MM_MC_BYTES)

    var differing = 0
    for i in range(c_n):
        if C_a[i] != C_b[i]:
            differing += 1

    var fired = differing > c_n // 2
    print(
        "PASS" if fired else "FAIL",
        "| control: a one-row input shift must move most outputs | differing=",
        differing,
        "of",
        c_n,
    )

    img.free()
    img2.free()
    B.free()
    C_a.free()
    C_b.free()
    return fired


def oracle_wrong_weight(H: Int, W: Int, C_in: Int, C_out: Int) -> Bool:
    """Falsifiable control: a scaled filter must make the fp32 check fire.

    Returns True when the control fires.
    """
    var conv = MMConv(R=3, S=3, pad_h=1, pad_w=1, W_out=W)
    var H_out = H
    var M = H_out * W
    var K = 9 * C_in
    var img_n = H * W * C_in
    var b_n = C_out * K
    var c_n = M * C_out

    var img = alloc[Scalar[MM_DTYPE]](img_n).as_unsafe_any_origin()
    var B = alloc[Scalar[MM_DTYPE]](b_n).as_unsafe_any_origin()
    var B_bad = alloc[Scalar[MM_DTYPE]](b_n).as_unsafe_any_origin()
    var C_fused = alloc[Scalar[MM_DTYPE]](c_n).as_unsafe_any_origin()
    var C_ref = alloc[Scalar[MM_DTYPE]](c_n).as_unsafe_any_origin()

    fill_random(img, img_n, 0x2468ACE0, 1.0)
    fill_random(B, b_n, 0x13579BDF, 0.05)
    for i in range(b_n):
        B_bad[i] = B[i] * Scalar[MM_DTYPE](1.5)

    run_fused(C_fused, img, B, H, W, C_in, C_out, conv, H_out, MM_MC_BYTES)
    conv_ref_fp32(C_ref, img, B_bad, H, W, C_in, C_out, conv, H_out)

    var worst = Scalar[F32](0)
    for i in range(c_n):
        worst = max(worst, abs(C_fused[i].cast[F32]() - C_ref[i].cast[F32]()))

    var fired = worst > TOL
    print(
        "PASS" if fired else "FAIL",
        "| control: a scaled filter must break the fp32 check | worst=",
        worst,
        ">",
        TOL,
    )

    img.free()
    B.free()
    B_bad.free()
    C_fused.free()
    C_ref.free()
    return fired


def main() raises:
    var ok = True
    ok = check_support_gate() and ok
    ok = check_conv_tile_rows() and ok
    ok = check_conv_tile_rows_no_underflow() and ok
    ok = check_window_inside() and ok

    for pass_i in range(2):
        var integers = pass_i == 0
        # A 3x3 stride-1 same-padding stack: the VAE decoder's whole shape
        # family, at the smallest channel count the fused arm accepts.
        ok = run_case(8, 8, 64, 64, 3, 3, 1, 1, "3x3 same", integers) and ok
        # Non-square, and an output width that does not divide the M-tile.
        ok = run_case(5, 11, 64, 66, 3, 3, 1, 1, "3x3 ragged", integers) and ok
        # Two channel runs per tap: the carried `(r, s, c0)` state has to wrap
        # the channel base before it steps the tap.
        ok = run_case(6, 6, 128, 64, 3, 3, 1, 1, "3x3 C128", integers) and ok
        # No padding: every tap is in bounds, so the halo branch is never taken.
        ok = run_case(7, 7, 64, 64, 3, 3, 0, 0, "3x3 valid", integers) and ok
        # 1x1: K is one channel run, and the tap loop never advances.
        ok = run_case(6, 6, 128, 128, 1, 1, 0, 0, "1x1", integers) and ok
        # 5x5 with a two-deep halo, and C_out below MM_NR so the N tail runs
        # alone.
        ok = run_case(6, 6, 64, 1, 5, 5, 2, 2, "5x5 N=1", integers) and ok
        # An output row count smaller than one microtile: M % MM_MR != 0 with
        # only the remainder path live.
        ok = run_case(3, 3, 64, 3, 3, 3, 1, 1, "tiny", integers) and ok
        # Wide enough that most microtiles' windows fit: the shape that puts
        # real work through the halo-free arm, which the shapes above barely
        # reach (their `split=` counts show it). The VAE layer this kernel is
        # for runs 92% of its microtiles there.
        ok = (
            run_case(20, 20, 64, 64, 3, 3, 1, 1, "3x3 interior", integers)
            and ok
        )
        # The two (W, C_in) pairs `VAE_DEVICE_CRITIQUE.md` section 4 names as
        # sitting exactly on `MM_MC_BYTES / 2` (`up1.resize_conv` /
        # `up2.resnet0.conv1` and `up2.resize_conv` / `up3.resnet0.conv1`,
        # scaled down in H for a fast host run — the underflow depends only
        # on W, C_in and R). Pre-fix, `mm_conv_tile_rows` clamped these to the
        # MM_MR floor; `check_conv_tile_rows_no_underflow` pins the value,
        # this proves the retile stays bit-exact at it.
        ok = (
            run_case(8, 256, 512, 64, 3, 3, 1, 1, "floor W=256 C=512", integers)
            and ok
        )
        ok = (
            run_case(8, 512, 256, 64, 3, 3, 1, 1, "floor W=512 C=256", integers)
            and ok
        )

    ok = oracle_shifted_input(6, 6, 64, 8) and ok
    ok = oracle_wrong_weight(6, 6, 64, 8) and ok

    print("ALL PASS" if ok else "SOME FAILED")
    if not ok:
        raise Error("fused im2col conv host validation failed")
