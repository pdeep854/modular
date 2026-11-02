"""Host validation for the transpose+bias kernel
(`transpose_bias_kernel.mojo`), the FLUX.2-Klein VAE decode's F4 fix
(`.agentwork/flux-npu/VAE_CODE_CRITIQUE.md`).

Runs the actual kernel (`transpose_bias_2d` / `transpose_bias_striped`) on the
host CPU against a reference computed directly from the formula
`output[n, r] = fp16(fp32(input[r, n]) + fp32(bias[n]))`, across shapes that
are, are not, and straddle a multiple of the kernel's `TB_TILE` block extent.

Because this is a pure data-movement op (a transpose plus one scalar add, no
reduction and no reordering), agreement is checked BIT-EXACT rather than to a
tolerance -- any mismatch is an addressing bug, not a rounding difference.

Two shapes are checked against the exact call-site formulas the VAE driver
will replace, so the kernel is proven against what it is actually wired to
replace and not only against a restatement of its own definition:

  - `scatterConvRows` (`npu_vae_decoder_dispatch.cpp:1248`): the fused conv's
    raw `[M, C_out]` result reshaped to `[C_out, spatial]` with the real
    per-channel bias riding along -- `output[co, pixel] = fp16(fp32(mn[pixel,
    co]) + fp32(bias[co]))`.
  - `nhwc_from_chw` (`conv_im2col.h:335`): a CHW activation packed to the
    `[H, W, C_in]` map the fused im2col gather reads, with NO bias -- treating
    `(H, W)` as one flat `spatial` axis, which is exactly what the flat `[R,
    N]` transpose computes when `N = spatial`.

Plus the launch-striped partition (four compute threads, each a contiguous
`N`-row slice) reproducing the whole-tile kernel exactly, and a falsifiable
oracle: perturbing the bias must change the output wherever bias is nonzero.

Run from this directory (imports the sibling core):
  mojo run test_transpose_bias_kernel.mojo
"""
from std.memory import alloc

from layout.coord import Coord
from layout.tile_layout import row_major

from hexagon_ondevice.transpose_bias_kernel import (
    RowMajor2D,
    _tb_vec_block64,
    transpose_bias_2d,
    transpose_bias_striped,
)

comptime DT = DType.float16
comptime F32 = DType.float32
comptime Ptr = UnsafePointer[Scalar[DT], MutUnsafeAnyOrigin]


def tile2d(ptr: Ptr, rows: Int, cols: Int) -> RowMajor2D[DT]:
    """Wraps a host allocation as the `[rows, cols]` tile the kernel takes,
    which is the construction `mojo_transpose_bias_f16` performs at the C-ABI
    seam."""
    return RowMajor2D[DT](ptr=ptr, layout=row_major(Coord(rows, cols)))


def fill_value(i: Int, salt: Int) -> Float64:
    """A deterministic value in a range small enough that every fp32
    intermediate is exact, so a mismatch can only be an addressing bug."""
    return Float64(Int((i * 2654435761 + salt) % 4001)) * 0.01 - 20.0


def fill(ptr: Ptr, n: Int, salt: Int):
    for i in range(n):
        ptr[i] = Scalar[DT](Float32(fill_value(i, salt)))


def ref_value(x: Scalar[DT], b: Scalar[DT]) -> Scalar[DT]:
    """`addBias`'s own definition (`npu_vae_decoder_dispatch.cpp:618`):
    widen both operands to fp32, add, narrow once."""
    return Scalar[DT](x.cast[F32]() + b.cast[F32]())


def check_exact(got: Ptr, want: Ptr, n: Int, name: String) -> Bool:
    var mismatches = 0
    var first_bad = -1
    for i in range(n):
        if got[i] != want[i]:
            mismatches += 1
            if first_bad < 0:
                first_bad = i
    var ok = mismatches == 0
    print(
        "PASS" if ok else "FAIL",
        "|",
        name,
        "| n=",
        n,
        "| mismatches=",
        mismatches,
        "| first_bad_index=",
        first_bad,
    )
    return ok


def run_case(R: Int, N: Int, use_bias: Bool, label: String) -> Bool:
    """Transposes a deterministic `[R, N]` input into `[N, R]`, with a real
    (nonzero, varied) per-column bias when `use_bias`, else the all-zero bias
    the pure-transpose leg passes. Checks bit-exact against the direct
    formula, computed with no code shared with the kernel."""
    var inp = alloc[Scalar[DT]](R * N).as_unsafe_any_origin()
    var bias = alloc[Scalar[DT]](N).as_unsafe_any_origin()
    var got = alloc[Scalar[DT]](N * R).as_unsafe_any_origin()
    var want = alloc[Scalar[DT]](N * R).as_unsafe_any_origin()

    fill(inp, R * N, 17)
    if use_bias:
        fill(bias, N, 977)
    else:
        for j in range(N):
            bias[j] = Scalar[DT](0.0)

    transpose_bias_2d[DT](
        tile2d(got, N, R), tile2d(inp, R, N), tile2d(bias, 1, N)
    )

    for r in range(R):
        for n in range(N):
            want[n * R + r] = ref_value(inp[r * N + n], bias[n])

    var ok = check_exact(got, want, N * R, label)

    inp.free()
    bias.free()
    got.free()
    want.free()
    return ok


def run_scatter_conv_rows_shape(m: Int, n_channels: Int) -> Bool:
    """Checks the kernel against `scatterConvRows`'s own formula
    (`npu_vae_decoder_dispatch.cpp:1248-1259`) at the shapes it is actually
    called at: the fused conv's `[M, C_out]` result, `M` pixels by `C_out`
    channels, becomes `[C_out, M]` with the real per-channel bias riding
    along. `M` stands in for `out_h * out_w`, already flattened -- the same
    flattening `scatterConvRows` itself does via `(co * out_h + i) * out_w +
    j == co * M + pixel`."""
    var mn = alloc[Scalar[DT]](m * n_channels).as_unsafe_any_origin()
    var bias = alloc[Scalar[DT]](n_channels).as_unsafe_any_origin()
    var got = alloc[Scalar[DT]](n_channels * m).as_unsafe_any_origin()
    var want = alloc[Scalar[DT]](n_channels * m).as_unsafe_any_origin()

    fill(mn, m * n_channels, 41)
    fill(bias, n_channels, 13)

    transpose_bias_2d[DT](
        tile2d(got, n_channels, m),
        tile2d(mn, m, n_channels),
        tile2d(bias, 1, n_channels),
    )

    for t in range(m):
        for co in range(n_channels):
            want[co * m + t] = ref_value(mn[t * n_channels + co], bias[co])

    var ok = check_exact(
        got,
        want,
        n_channels * m,
        "scatterConvRows-shape M=" + String(m) + " C_out=" + String(n_channels),
    )

    mn.free()
    bias.free()
    got.free()
    want.free()
    return ok


def run_nhwc_from_chw_shape(c_in: Int, h: Int, w: Int) -> Bool:
    """Checks the kernel against `nhwc_from_chw`'s own formula
    (`conv_im2col.h:335-349`) at the shapes it is actually called at: a CHW
    activation packed to the `[H, W, C_in]` map the fused im2col gather
    reads, with no bias. `(H, W)` is one flat `spatial` axis on both sides of
    the transpose -- `(ci * h + i) * w + j == ci * spatial + pixel` on the
    read, `(i * w + j) * c_in + ci == pixel * c_in + ci` on the write -- so
    the flat `[C_in, spatial] -> [spatial, C_in]` transpose this kernel
    computes IS `nhwc_from_chw`, not merely analogous to it."""
    var spatial = h * w
    var chw = alloc[Scalar[DT]](c_in * spatial).as_unsafe_any_origin()
    var zero_bias = alloc[Scalar[DT]](spatial).as_unsafe_any_origin()
    var got = alloc[Scalar[DT]](spatial * c_in).as_unsafe_any_origin()
    var want = alloc[Scalar[DT]](spatial * c_in).as_unsafe_any_origin()

    fill(chw, c_in * spatial, 191)
    for j in range(spatial):
        zero_bias[j] = Scalar[DT](0.0)

    transpose_bias_2d[DT](
        tile2d(got, spatial, c_in),
        tile2d(chw, c_in, spatial),
        tile2d(zero_bias, 1, spatial),
    )

    for i in range(h):
        for j in range(w):
            var pixel = i * w + j
            for ci in range(c_in):
                want[pixel * c_in + ci] = chw[ci * spatial + pixel]

    var ok = check_exact(
        got,
        want,
        spatial * c_in,
        "nhwc_from_chw-shape C_in="
        + String(c_in)
        + " H="
        + String(h)
        + " W="
        + String(w),
    )

    chw.free()
    zero_bias.free()
    got.free()
    want.free()
    return ok


def run_striped_case(R: Int, N: Int) -> Bool:
    """The four compute threads, each a contiguous `N`-row (write-axis) slice,
    must reproduce the whole-tile kernel exactly -- the same partition check
    `test_silu_swiglu.mojo` runs for its own launch-striped entry. A sentinel
    pre-fill makes an uncovered output row a failure rather than a silent
    zero, since the pre-fill would otherwise pass a bitwise `!=` check by
    accident."""
    comptime SENTINEL = Scalar[DT](-31000.0)

    var inp = alloc[Scalar[DT]](R * N).as_unsafe_any_origin()
    var bias = alloc[Scalar[DT]](N).as_unsafe_any_origin()
    var whole = alloc[Scalar[DT]](N * R).as_unsafe_any_origin()
    var striped_out = alloc[Scalar[DT]](N * R).as_unsafe_any_origin()

    fill(inp, R * N, 71)
    fill(bias, N, 503)
    for i in range(N * R):
        whole[i] = SENTINEL
        striped_out[i] = SENTINEL

    transpose_bias_2d[DT](
        tile2d(whole, N, R), tile2d(inp, R, N), tile2d(bias, 1, N)
    )
    for t in range(5):
        transpose_bias_striped[DT](
            tile2d(striped_out, N, R),
            tile2d(inp, R, N),
            tile2d(bias, 1, N),
            UInt32(t),
        )

    var mismatches = 0
    var uncovered = 0
    for i in range(N * R):
        if whole[i] != striped_out[i]:
            mismatches += 1
        if striped_out[i] == SENTINEL:
            uncovered += 1
    var ok = mismatches == 0 and uncovered == 0
    print(
        "PASS" if ok else "FAIL",
        "| striped == whole-tile | R=",
        R,
        "N=",
        N,
        "| mismatches=",
        mismatches,
        "| uncovered=",
        uncovered,
    )

    inp.free()
    bias.free()
    whole.free()
    striped_out.free()
    return ok


def run_bias_perturbation_oracle() -> Bool:
    """Falsifiable: perturbing every bias entry must change every output
    element wherever the corresponding bias is nonzero. A kernel that
    silently ignored the bias buffer (reading the wrong pointer, or the
    zero-bias branch a caller never asked for) would pass every check above
    that also uses a nonzero bias, since `check_exact` only compares against
    a reference computed the SAME way -- this is the control that would catch
    it."""
    var R = 9
    var N = 40
    var inp = alloc[Scalar[DT]](R * N).as_unsafe_any_origin()
    var bias_a = alloc[Scalar[DT]](N).as_unsafe_any_origin()
    var bias_b = alloc[Scalar[DT]](N).as_unsafe_any_origin()
    var out_a = alloc[Scalar[DT]](N * R).as_unsafe_any_origin()
    var out_b = alloc[Scalar[DT]](N * R).as_unsafe_any_origin()

    fill(inp, R * N, 3)
    fill(bias_a, N, 5)
    for j in range(N):
        bias_b[j] = Scalar[DT](bias_a[j].cast[F32]() + Float32(10.0))

    transpose_bias_2d[DT](
        tile2d(out_a, N, R), tile2d(inp, R, N), tile2d(bias_a, 1, N)
    )
    transpose_bias_2d[DT](
        tile2d(out_b, N, R), tile2d(inp, R, N), tile2d(bias_b, 1, N)
    )

    var unchanged = 0
    for i in range(N * R):
        if out_a[i] == out_b[i]:
            unchanged += 1
    var ok = unchanged == 0
    print(
        "PASS" if ok else "FAIL",
        "| bias-perturbation oracle fires | unchanged=",
        unchanged,
        "of",
        N * R,
    )

    inp.free()
    bias_a.free()
    bias_b.free()
    out_a.free()
    out_b.free()
    return ok


def run_vec_blocks_case(R: Int, N: Int, use_bias: Bool, label: String) -> Bool:
    """Drives `_tb_vec_block64` — the `SKYE_HMX_TB_VEC` arm's 64x64 riffle
    block — directly over every full block of an `[R, N]` extent, checking
    BIT-EXACT against the same no-shared-code formula the scalar cases use.
    Direct-called so the network is proven regardless of how the test binary
    was flagged; on the host the riffles execute `hvx_permute`'s
    LLVM-transcribed lane models, so this is a byte proof of the network
    itself, not of an intrinsic's availability. Extents must be multiples
    of 64 — ragged edges belong to the scalar path by construction."""
    var inp = alloc[Scalar[DT]](R * N).as_unsafe_any_origin()
    var bias = alloc[Scalar[DT]](N).as_unsafe_any_origin()
    var got = alloc[Scalar[DT]](N * R).as_unsafe_any_origin()
    var want = alloc[Scalar[DT]](N * R).as_unsafe_any_origin()

    fill(inp, R * N, 4242)
    if use_bias:
        fill(bias, N, 131)
    else:
        for j in range(N):
            bias[j] = Scalar[DT](0.0)
    # Poison the destination so an unwritten lane cannot read as a pass.
    for i in range(N * R):
        got[i] = Scalar[DT](-333.0)

    var out_t = tile2d(got, N, R)
    var in_t = tile2d(inp, R, N)
    var bias_t = tile2d(bias, 1, N)
    var n0 = 0
    while n0 < N:
        var r0 = 0
        while r0 < R:
            _tb_vec_block64(out_t, in_t, bias_t, r0, n0)
            r0 += 64
        n0 += 64

    for r in range(R):
        for n in range(N):
            want[n * R + r] = ref_value(inp[r * N + n], bias[n])

    var ok = check_exact(got, want, N * R, label)
    inp.free()
    bias.free()
    got.free()
    want.free()
    return ok


def main() raises:
    var failures = 0

    # The vector arm's riffle network, byte-proven against the formula.
    var vec_r = [64, 128, 192, 64]
    var vec_n = [64, 192, 64, 256]
    for vi in range(len(vec_r)):
        for use_bias in [True, False]:
            var vlabel = (
                "VEC R="
                + String(vec_r[vi])
                + " N="
                + String(vec_n[vi])
                + " bias="
                + String(use_bias)
            )
            if not run_vec_blocks_case(vec_r[vi], vec_n[vi], use_bias, vlabel):
                failures += 1

    # Tiny, exact-tile, tail, and straddling-a-tile-boundary shapes on both
    # axes, each with and without a real bias.
    var shape_r = [1, 1, 3, 31, 32, 33, 64, 65, 100, 257]
    var shape_n = [1, 5, 5, 33, 32, 31, 64, 127, 200, 61]
    for si in range(len(shape_r)):
        for use_bias in [True, False]:
            var label = (
                "R="
                + String(shape_r[si])
                + " N="
                + String(shape_n[si])
                + " bias="
                + String(use_bias)
            )
            if not run_case(shape_r[si], shape_n[si], use_bias, label):
                failures += 1

    # The two exact call-site shapes the VAE driver wires this kernel to.
    if not run_scatter_conv_rows_shape(64 * 64, 128):
        failures += 1
    if not run_scatter_conv_rows_shape(17, 5):
        failures += 1
    if not run_nhwc_from_chw_shape(128, 16, 16):
        failures += 1
    if not run_nhwc_from_chw_shape(64, 9, 7):
        failures += 1

    # The launch-striped partition the shipped entry uses.
    for si in range(len(shape_r)):
        if not run_striped_case(shape_r[si], shape_n[si]):
            failures += 1

    # The up3 scatter geometry (242's slow mode: N=C_out=128, R=spatial at
    # 512x512) against the independent formula, and striped-vs-whole at the
    # 64-unit boundary shapes the SKYE_HMX_TB_STRIPE64 split introduces. The
    # striping changes WHICH thread writes an output row, never the value:
    # bit-identity is the gate, and the sentinel makes an ownership gap a
    # failure rather than a silent zero.
    if not run_case(512 * 512, 128, True, "up3-scatter R=262144 N=128"):
        failures += 1
    var stripe_r = [512 * 512, 96, 96, 96, 96, 70, 96]
    var stripe_n = [128, 128, 129, 136, 192, 320, 127]
    for si in range(len(stripe_r)):
        if not run_striped_case(stripe_r[si], stripe_n[si]):
            failures += 1

    if not run_bias_perturbation_oracle():
        failures += 1

    print()
    if failures == 0:
        print("ALL TRANSPOSE_BIAS TESTS PASSED")
    else:
        raise Error(String(failures) + " transpose_bias test case(s) FAILED")
