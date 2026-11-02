"""Off-device algorithm test for the HVX affine GroupNorm kernel.

Runs the kernel's own `_group_norm` (which drives `GroupNorm3Pass`) on the host
and compares against an independent float64 reference. This proves the ALGORITHM
— the per-group two-pass mean/variance over the contiguous `(n, group)` block,
the mean subtraction, the eps guard, the per-channel affine, the block and
spatial tail handling, the group split across launch threads — is numerically
correct. It says nothing about whether the device emits vector instructions for
it: the SIMD width resolves the same on host and device, so a scalar host build
would pass this too. Vectorization is settled separately, by disassembling the
v81 object (build.sh's HVX-mnemonic count) and ultimately on silicon.

The reference accumulates in float64 where the kernel accumulates in reassociated
float32, so the two disagree by a small amount before the down-cast — far below
the fp16 / bf16 output resolution, which is why the tolerance is set at that
resolution and the reported max errors sit well inside it.
"""

from hexagon_ondevice.groupnorm_kernel import GroupNorm3Pass, _group_norm

from std.math import isfinite, sqrt
from std.memory import alloc

comptime SENTINEL: Float32 = 30000.0
"""A finite value normalized output never takes, pre-written to every output
element so an unwritten (missing) one is detectable after the launch."""


def _fill_seeded[
    dtype: DType
](
    ptr: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin],
    n: Int,
    seed: UInt32,
    scale: Float32,
):
    """Deterministic pseudo-random fill in roughly [-scale, +scale]."""
    var state = seed | 1
    for i in range(n):
        state = state * 1664525 + 1013904223
        var u = Float32(Int(state >> 8) & 0xFFFF) / Float32(0xFFFF)
        ptr.store(i, ((u * 2.0 - 1.0) * scale).cast[dtype]())


def _fill_gamma[
    dtype: DType
](ptr: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin], n: Int, seed: UInt32):
    """Affine scale near 1.0 (roughly [0.5, 1.5]) — a realistic GroupNorm
    weight, and one whose sign never flips so the test is not accidentally
    normalizing a degenerate case."""
    var state = seed | 1
    for i in range(n):
        state = state * 1664525 + 1013904223
        var u = Float32(Int(state >> 8) & 0xFFFF) / Float32(0xFFFF)
        ptr.store(i, (1.0 + (u - 0.5)).cast[dtype]())


def _fill_beta[
    dtype: DType
](ptr: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin], n: Int, seed: UInt32):
    """Affine bias in roughly [-0.5, 0.5]."""
    var state = seed | 1
    for i in range(n):
        state = state * 1664525 + 1013904223
        var u = Float32(Int(state >> 8) & 0xFFFF) / Float32(0xFFFF)
        ptr.store(i, ((u - 0.5)).cast[dtype]())


def _fill_ramp[
    dtype: DType
](
    ptr: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin],
    num_blocks: Int,
    group_size: Int,
):
    """A linear ramp across each group block, offset per block — smoothly
    varying, so the mean is far from zero and the variance well conditioned."""
    for r in range(num_blocks):
        for i in range(group_size):
            var t = Float32(i) / Float32(group_size)
            ptr.store(
                r * group_size + i,
                ((t * 4.0 - 2.0) + Float32(r) * 0.5).cast[dtype](),
            )


def _fill_const[
    dtype: DType
](ptr: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin], n: Int, c: Float32):
    for i in range(n):
        ptr.store(i, c.cast[dtype]())


def _fill_sentinel[
    dtype: DType
](ptr: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin], n: Int):
    for i in range(n):
        ptr.store(i, SENTINEL.cast[dtype]())


def _ref_group_norm[
    dtype: DType
](
    out_ref: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin],
    input: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin],
    gamma: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin],
    beta: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin],
    N: Int,
    C: Int,
    spatial: Int,
    num_groups: Int,
    eps: Float32,
):
    """Independent float64 reference. For each `(n, group)`: mean over the group
    block, variance about it, then the per-channel affine normalize. Reads the
    same stored dtype inputs and gamma / beta the kernel reads."""
    var channels_per_group = C // num_groups
    var group_size = channels_per_group * spatial

    for n in range(N):
        for g in range(num_groups):
            var c_base = g * channels_per_group
            var block_base = (n * num_groups + g) * group_size

            var s = Float64(0)
            for j in range(group_size):
                s += Float64(input.load(block_base + j).cast[DType.float32]())
            var mean = s / Float64(group_size)

            var vs = Float64(0)
            for j in range(group_size):
                var d = (
                    Float64(input.load(block_base + j).cast[DType.float32]())
                    - mean
                )
                vs += d * d
            var variance = vs / Float64(group_size)

            var inv_std = Float64(1) / sqrt(variance + Float64(eps))

            for cl in range(channels_per_group):
                var c = c_base + cl
                var gamma_c = Float64(gamma.load(c).cast[DType.float32]())
                var beta_c = Float64(beta.load(c).cast[DType.float32]())
                for sp in range(spatial):
                    var idx = block_base + cl * spatial + sp
                    var o = (
                        Float64(input.load(idx).cast[DType.float32]()) - mean
                    ) * inv_std * gamma_c + beta_c
                    out_ref.store(idx, o.cast[dtype]())


def _run_case[
    dtype: DType
](
    name: String,
    N: Int,
    C: Int,
    spatial: Int,
    num_groups: Int,
    input: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin],
    gamma: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin],
    beta: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin],
    eps: Float32,
    atol: Float32,
    rtol: Float32,
) -> Bool:
    """Drives the real striped `_group_norm` over the four compute threads, then
    scores it against the float64 reference. Returns True on pass."""
    var count = N * C * spatial
    var out = alloc[Scalar[dtype]](count).as_unsafe_any_origin()
    var reference = alloc[Scalar[dtype]](count).as_unsafe_any_origin()

    _fill_sentinel[dtype](out, count)
    _ref_group_norm[dtype](
        reference, input, gamma, beta, N, C, spatial, num_groups, eps
    )

    # Each compute thread writes the groups its launch index owns; the union
    # covers every group exactly once. Threads 4+ (the DMA thread) take none.
    for t in range(4):
        _group_norm(
            out, input, gamma, beta, N, C, spatial, num_groups, UInt32(t), eps
        )

    var max_err = Float32(0)
    var max_abs = Float32(0)
    var missing = 0
    var nonfinite = 0
    for i in range(count):
        var o = out.load(i).cast[DType.float32]()
        var e = reference.load(i).cast[DType.float32]()
        if not isfinite(o):
            nonfinite += 1
            continue
        var s = SENTINEL
        if (o - s if o > s else s - o) < 1.0:
            missing += 1
        var abs_e = e if e >= 0 else -e
        if abs_e > max_abs:
            max_abs = abs_e
        var diff = o - e
        if diff < 0:
            diff = -diff
        if diff > max_err:
            max_err = diff

    var tol = atol + rtol * max_abs
    var ok = (max_err <= tol) and (missing == 0) and (nonfinite == 0)

    print(
        name,
        "N",
        N,
        "C",
        C,
        "sp",
        spatial,
        "G",
        num_groups,
        "-> ",
        "PASS" if ok else "FAIL",
        "(max_err=",
        max_err,
        "tol=",
        tol,
        "max|out|=",
        max_abs,
        "missing=",
        missing,
        "nonfinite=",
        nonfinite,
        ")",
    )

    out.free()
    reference.free()
    return ok


def _run_guard_case[
    dtype: DType
](
    name: String,
    N: Int,
    C: Int,
    spatial: Int,
    num_groups: Int,
    seed: UInt32,
    in_scale: Float32,
    eps: Float32,
) -> Bool:
    """Asserts pass 3 never writes past the valid `N * C * spatial` extent.

    The per-channel affine store is the path that faulted the NSP on silicon: an
    over-wide vector store on a sub-vector spatial tail overruns the output
    buffer at the last channel of the last group. The host cannot reproduce the
    device fault (a masked or in-bounds store resolves correctly on x86), so
    this is a proxy, not a repro — it appends a sentinel guard region after the
    output and checks it stays untouched, holding the invariant the fix restores
    (every store lands inside the extent) so a regression to an over-wide tail
    store trips here. One HVX vector (64 elements at 2-byte dtype) bounds the
    largest possible overrun, so that is the guard width."""
    comptime GUARD = 64
    var count = N * C * spatial
    var out = alloc[Scalar[dtype]](count + GUARD).as_unsafe_any_origin()
    var input = alloc[Scalar[dtype]](count).as_unsafe_any_origin()
    var gamma = alloc[Scalar[dtype]](C).as_unsafe_any_origin()
    var beta = alloc[Scalar[dtype]](C).as_unsafe_any_origin()

    _fill_seeded[dtype](input, count, seed, in_scale)
    _fill_gamma[dtype](gamma, C, seed ^ 0xA5A5)
    _fill_beta[dtype](beta, C, seed ^ 0x5A5A)
    for i in range(count, count + GUARD):
        out.store(i, SENTINEL.cast[dtype]())

    for t in range(4):
        _group_norm(
            out, input, gamma, beta, N, C, spatial, num_groups, UInt32(t), eps
        )

    # Compare against the sentinel as it round-trips through `dtype`: 30000 is
    # exact in fp16 but not bf16, so the f32 constant would flag an untouched
    # bf16 guard element as an overrun.
    var sentinel_d = SENTINEL.cast[dtype]().cast[DType.float32]()
    var overruns = 0
    for i in range(count, count + GUARD):
        var o = out.load(i).cast[DType.float32]()
        var d = o - sentinel_d if o > sentinel_d else sentinel_d - o
        if d >= 1.0:
            overruns += 1

    var ok = overruns == 0
    print(
        name,
        "N",
        N,
        "C",
        C,
        "sp",
        spatial,
        "G",
        num_groups,
        "-> ",
        "PASS" if ok else "FAIL",
        "(guard overruns=",
        overruns,
        "of",
        GUARD,
        ")",
    )

    out.free()
    input.free()
    gamma.free()
    beta.free()
    return ok


def _case[
    dtype: DType
](
    name: String,
    N: Int,
    C: Int,
    spatial: Int,
    num_groups: Int,
    seed: UInt32,
    in_scale: Float32,
    eps: Float32,
    atol: Float32,
    rtol: Float32,
) -> Bool:
    """Allocates input + affine params for one shape, seeded random, and runs it.
    """
    var count = N * C * spatial
    var input = alloc[Scalar[dtype]](count).as_unsafe_any_origin()
    var gamma = alloc[Scalar[dtype]](C).as_unsafe_any_origin()
    var beta = alloc[Scalar[dtype]](C).as_unsafe_any_origin()

    _fill_seeded[dtype](input, count, seed, in_scale)
    _fill_gamma[dtype](gamma, C, seed ^ 0xA5A5)
    _fill_beta[dtype](beta, C, seed ^ 0x5A5A)

    var ok = _run_case[dtype](
        name, N, C, spatial, num_groups, input, gamma, beta, eps, atol, rtol
    )

    input.free()
    gamma.free()
    beta.free()
    return ok


def _cases[dtype: DType](label: String, atol: Float32, rtol: Float32) -> Bool:
    """Runs the full shape/edge matrix for one dtype."""
    print("---", label, "---")
    var eps = Float32(1e-6)
    var all_ok = True

    # VAE-real channel counts at num_groups=32 (channels_per_group 4 / 8 / 16),
    # spatial a whole number of 64-lane vectors — the decoder's common regime.
    all_ok = (
        _case[dtype]("vae C128 ", 1, 128, 256, 32, 0xF10C, 3.0, eps, atol, rtol)
        and all_ok
    )
    all_ok = (
        _case[dtype]("vae C256 ", 1, 256, 256, 32, 0x1234, 3.0, eps, atol, rtol)
        and all_ok
    )
    all_ok = (
        _case[dtype]("vae C512 ", 1, 512, 128, 32, 0x2468, 3.0, eps, atol, rtol)
        and all_ok
    )

    # Batch > 1: the N * num_groups group split runs across the four threads on
    # a count that divides (64) and one that does not (2 * 32 = 64 divides; use
    # C=256 spatial small to keep it cheap).
    all_ok = (
        _case[dtype]("batch N2 ", 2, 128, 128, 32, 0x3690, 3.0, eps, atol, rtol)
        and all_ok
    )

    # Spatial tail: spatial not a multiple of 64, so the per-channel affine runs
    # its masked partial-vector path (channels_per_group=4, spatial=96 -> group
    # block 384 is a whole number of vectors, but each 96-element channel run has
    # a 32-element tail).
    all_ok = (
        _case[dtype]("sp-tail  ", 1, 128, 96, 32, 0x7A11, 3.0, eps, atol, rtol)
        and all_ok
    )

    # Both tails: spatial=257 -> each channel run has a 1-element tail, and the
    # group block (4 * 257 = 1028) has a 4-element reduction tail.
    all_ok = (
        _case[dtype]("both-tail", 1, 128, 257, 32, 0x9E3F, 3.0, eps, atol, rtol)
        and all_ok
    )

    # Group count that does not divide by four: N=3, num_groups=2 -> 6 groups
    # over four threads (2, 2, 1, 1), the arrangement that takes the range clamp.
    all_ok = (
        _case[dtype]("split3x2 ", 3, 8, 64, 2, 0xBEEF, 3.0, eps, atol, rtol)
        and all_ok
    )

    # Small spatial: spatial=1 (1x1 feature map), so each channel is a single
    # element and the whole group block is a reduction tail.
    all_ok = (
        _case[dtype]("sp1      ", 1, 128, 1, 32, 0xC0DE, 3.0, eps, atol, rtol)
        and all_ok
    )

    # Bounds guard: the pass-3 per-channel tail store must stay inside the output
    # buffer on every sub-vector spatial tail — spatial=96 is the exact shape that
    # faulted the NSP, 257 (1-element tail) and 1 (whole channel a tail) bracket it.
    all_ok = (
        _run_guard_case[dtype]("guard96  ", 1, 128, 96, 32, 0x7A11, 3.0, eps)
        and all_ok
    )
    all_ok = (
        _run_guard_case[dtype]("guard257 ", 1, 128, 257, 32, 0x9E3F, 3.0, eps)
        and all_ok
    )
    all_ok = (
        _run_guard_case[dtype]("guard1   ", 1, 128, 1, 32, 0xC0DE, 3.0, eps)
        and all_ok
    )

    # Ramp: smoothly varying, large nonzero per-group mean — a different regime
    # from the zero-mean random fills.
    var ramp_count = 1 * 128 * 128
    var ramp = alloc[Scalar[dtype]](ramp_count).as_unsafe_any_origin()
    var ramp_g = alloc[Scalar[dtype]](128).as_unsafe_any_origin()
    var ramp_b = alloc[Scalar[dtype]](128).as_unsafe_any_origin()
    # 32 groups x (4 channels x 128 spatial) = 512-element blocks.
    _fill_ramp[dtype](ramp, 32, 512)
    _fill_gamma[dtype](ramp_g, 128, 0x11)
    _fill_beta[dtype](ramp_b, 128, 0x22)
    all_ok = (
        _run_case[dtype](
            "ramp     ", 1, 128, 128, 32, ramp, ramp_g, ramp_b, eps, atol, rtol
        )
        and all_ok
    )
    ramp.free()
    ramp_g.free()
    ramp_b.free()

    # Constant group: variance 0, so the eps guard is the only thing between the
    # reciprocal and a divide-by-zero. Normalized part is exactly 0, so output
    # must equal beta per channel — a direct check that the affine is applied.
    var cst_count = 1 * 128 * 128
    var cst = alloc[Scalar[dtype]](cst_count).as_unsafe_any_origin()
    var cst_g = alloc[Scalar[dtype]](128).as_unsafe_any_origin()
    var cst_b = alloc[Scalar[dtype]](128).as_unsafe_any_origin()
    _fill_const[dtype](cst, cst_count, 1.5)
    _fill_gamma[dtype](cst_g, 128, 0x33)
    _fill_beta[dtype](cst_b, 128, 0x44)
    all_ok = (
        _run_case[dtype](
            "const    ", 1, 128, 128, 32, cst, cst_g, cst_b, eps, atol, rtol
        )
        and all_ok
    )
    cst.free()
    cst_g.free()
    cst_b.free()

    # Large magnitude: a sum of squares that overflows fp16 (~6e4 max) but not the
    # fp32 accumulator, so a passing result is direct evidence the accumulation is
    # fp32. Output stays normalized regardless of input scale.
    all_ok = (
        _case[dtype](
            "large    ", 1, 256, 128, 32, 0xB16, 2000.0, eps, atol, rtol
        )
        and all_ok
    )

    return all_ok


def main() raises:
    var all_ok = True

    # fp16: ~11 mantissa bits, ulp ~1e-3 at |out|~4.
    all_ok = _cases[DType.float16]("float16", 2e-2, 1e-2) and all_ok

    # bf16: 8 mantissa bits, ulp ~2e-2 at |out|~4 — a coarser tolerance for a
    # coarser type, still far tighter than any algorithm bug would clear.
    all_ok = _cases[DType.bfloat16]("bfloat16", 8e-2, 2e-2) and all_ok

    print("")
    if all_ok:
        print("PASS")
    else:
        print("FAIL")
        raise Error("groupnorm algorithm test failed")
