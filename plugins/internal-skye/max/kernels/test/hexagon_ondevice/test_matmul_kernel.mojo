"""Host validation for the register-blocked HVX matmul core (`matmul_kernel.mojo`).

Runs the actual core (`matmul_bt`) on the host CPU against three references,
across the shapes the batched matmul must survive: full `MM_MR` x `MM_NR` tiles,
the M and N tails (including N not a multiple of 64 — the tail-store fault
shape), tiny M (the gemv M=1 case), tiny N (the conv_out N=3 case), K at several
multiples of 64, and the two worst K in the DiT plan.

Checks per shape:
  1. Exact equivalence to the one-row gemv computed row by row. The batched core
     must be BIT-IDENTICAL to `M` separate single-row dot products over the same
     inputs — register blocking shares the loop nest but must not move the
     per-output fp32 accumulation order. This is the kernel's actual claim.
  2. A CPU fp32 reference within tolerance, so a bug common to the batched and
     row-by-row host paths cannot pass unnoticed.
  3. A bounds-guard canary — a sentinel region after C that an over-wide store
     past the output would corrupt. The numerics checks are blind to it, and a
     masked vector store at a non-64-aligned N tail is exactly what would trip
     it; the scalar store must leave it pristine.

Every shape runs the whole `MM_MC_BYTES` sweep, adding the M-tile check:
  4. Exact equivalence between the tiled nest and the untiled one. Cache
     blocking only reorders independent outputs, so it must be BIT-IDENTICAL to
     the same core given a budget large enough for a single tile — not merely
     within tolerance. The sweep is chosen to reach the adversarial tile heights:
     one microtile row (`MC == MM_MR`), a height that does not divide `M`, and a
     budget larger than the whole panel.
  5. `mm_tile_rows` returns a positive multiple of `MM_MR` for every budget. The
     bit-exactness argument rests on it, so it is asserted rather than assumed.

And every shape runs twice, under two fills:
  - Pseudo-random fp16, where rounding makes the accumulation ORDER observable,
    so a K-split or a changed fold shows up in check 1 or 4.
  - Small exact integers, where every product and partial sum is exact in fp32,
    so the fp32 reference must match to ZERO error and a skipped, duplicated, or
    misindexed tile cannot hide under a tolerance.

Plus a falsifiable wrong-weight oracle that MUST fire: with a scaled weight
matrix the fp32-reference check has to report a mismatch, or the check has no
teeth.

The core is target-independent Mojo; only its lowering differs between host and
Hexagon. Matching here proves the algorithm (tiling, tails, equivalence). The
device MAC's interleaved-lane numerics still need on-device confirmation.

Run from this directory (imports the sibling core):
  mojo run test_matmul_kernel.mojo
"""
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
    MMTile,
    matmul_bt,
    mm_fmac_host,
    mm_reduce_add,
    mm_tile_rows,
)

comptime F32 = DType.float32
comptime Ptr = UnsafePointer[Scalar[MM_DTYPE], MutUnsafeAnyOrigin]

# fp16 carries ~3 decimal digits; the batched core and the fp32 oracle both round
# to fp16, so agreement is tight and the slack only absorbs the fp32 reduction
# order. Kept well below any true output magnitude so the wrong-weight oracle has
# room to fire.
comptime TOL = Scalar[F32](0.02)

comptime MC_UNTILED = 64 * 1024 * 1024
"""A budget no test panel can fill, so the core runs as one M-tile — the nest
this file's shapes ran before cache blocking, and the reference check 4 compares
every tiled run against."""


def run_core(C: Ptr, A: Ptr, B: Ptr, M: Int, N: Int, K: Int, mc_bytes: Int):
    """Drives the core through the seam a C-ABI caller has to build.

    The kernel's functions take tiles, so the extents become three layouts once,
    here, and this file's own boundary — allocations and loop bounds in plain
    `Int`s — stops exactly where the device wrapper `mojo_matmul_f16`'s does.
    Every call below goes through this one seam, so a transposed extent would be
    a single-site bug rather than one repeated per call site.
    """
    matmul_bt[MM_MR, MM_NR](
        MMTile(ptr=C, layout=row_major(Coord(M, N))),
        MMTile(ptr=A, layout=row_major(Coord(M, K))),
        MMTile(ptr=B, layout=row_major(Coord(N, K))),
        mc_bytes,
    )


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

    Small enough that every product and every partial sum of a K-long dot
    product is exact in fp32 (the widest here is 3 * 12288, far under 2^24), so
    the vector-lane fold and the scalar fp32 reference must agree to zero error
    rather than to a tolerance.
    """
    var rng = seed
    var span = UInt32(2 * bound + 1)
    for i in range(count):
        rng ^= rng << 13
        rng ^= rng >> 17
        rng ^= rng << 5
        ptr[i] = Scalar[MM_DTYPE](Int((rng >> 8) % span) - bound)


def matmul_ref_fp32(C: Ptr, A: Ptr, B: Ptr, M: Int, N: Int, K: Int):
    """Independent scalar fp32 oracle: `C[m,n] = sum_k A[m,k]*B[n,k]`."""
    for m in range(M):
        for n in range(N):
            var acc = Scalar[F32](0)
            for k in range(K):
                acc += A[m * K + k].cast[F32]() * B[n * K + k].cast[F32]()
            C[m * N + n] = acc.cast[MM_DTYPE]()


def gemv_row_by_row(C: Ptr, A: Ptr, B: Ptr, M: Int, N: Int, K: Int):
    """The one-row gemv inner loop (host MAC + reduce), run once per A-row.

    A transcription of `mojo_gemv_f16`'s body with the portable MAC: each output
    `(m, n)` is the 64-lane fp32 dot product of A-row `m` with B-row `n`, folded
    to a scalar. The batched core must equal this bit for bit.
    """
    var k_vecs = K // MM_W
    for m in range(M):
        var a_row = A + m * K
        for n in range(N):
            var b_row = B + n * K
            var acc = SIMD[MM_ACCUM, MM_W](0)
            for kv in range(k_vecs):
                var k_off = kv * MM_W
                acc = mm_fmac_host(
                    acc, a_row.load[MM_W](k_off), b_row.load[MM_W](k_off)
                )
            C[m * N + n] = mm_reduce_add(acc).cast[MM_DTYPE]()


def run_case(
    M: Int, N: Int, K: Int, name: String, integer_inputs: Bool
) -> Bool:
    """Runs every check at every `MM_MC_BYTES` in the sweep for one shape.

    The three references do not depend on the tile budget, so they are computed
    once and the sweep re-runs only the core.
    """
    comptime PAD = MM_W  # wide enough to catch a 64-lane over-store past C
    var a_n = M * K
    var b_n = N * K
    var c_n = M * N

    var A = alloc[Scalar[MM_DTYPE]](a_n).as_unsafe_any_origin()
    var B = alloc[Scalar[MM_DTYPE]](b_n).as_unsafe_any_origin()
    var C_bat = alloc[Scalar[MM_DTYPE]](c_n + PAD).as_unsafe_any_origin()
    var C_flat = alloc[Scalar[MM_DTYPE]](c_n).as_unsafe_any_origin()
    var C_row = alloc[Scalar[MM_DTYPE]](c_n).as_unsafe_any_origin()
    var C_ref = alloc[Scalar[MM_DTYPE]](c_n).as_unsafe_any_origin()

    if integer_inputs:
        fill_integers(A, a_n, 0x12345678, 3)
        fill_integers(B, b_n, 0x0BADF00D, 1)
    else:
        fill_random(A, a_n, 0x12345678, 1.0)
        fill_random(B, b_n, 0x0BADF00D, 0.05)

    gemv_row_by_row(C_row, A, B, M, N, K)
    matmul_ref_fp32(C_ref, A, B, M, N, K)
    run_core(C_flat, A, B, M, N, K, MC_UNTILED)

    # Budgets chosen so the derived tile height reaches the adversarial cases at
    # every K in the suite: one microtile row, a height that does not divide M,
    # and a panel larger than the whole of M.
    var budgets = [512, 1536, 4096, 40960, MM_MC_BYTES]

    var bit_vs_gemv = 0
    var bit_vs_flat = 0
    var max_err = Scalar[F32](0)
    var saw_nan = False
    var canary_ok = True
    var tiles_lo = M
    var tiles_hi = 1

    for bi in range(len(budgets)):
        # Seed the whole output (including the trailing PAD canary) with a
        # sentinel no correct output writes, so an over-wide store surfaces as a
        # changed PAD and a skipped tile surfaces as an unwritten output.
        var sentinel = Scalar[MM_DTYPE](-3.5)
        for i in range(c_n + PAD):
            C_bat[i] = sentinel

        run_core(C_bat, A, B, M, N, K, budgets[bi])

        var mc = mm_tile_rows[MM_MR](K, budgets[bi])
        var tiles = (M + mc - 1) // mc
        if tiles < tiles_lo:
            tiles_lo = tiles
        if tiles > tiles_hi:
            tiles_hi = tiles

        for i in range(c_n):
            if C_bat[i] != C_row[i]:
                bit_vs_gemv += 1
            if C_bat[i] != C_flat[i]:
                bit_vs_flat += 1
            var v = C_bat[i].cast[F32]()
            if v != v:
                saw_nan = True
            max_err = max(max_err, abs(v - C_ref[i].cast[F32]()))
        for i in range(c_n, c_n + PAD):
            if C_bat[i] != sentinel:
                canary_ok = False

    # Exact integer inputs leave no rounding for the fp32 reference to disagree
    # under, so the tolerance collapses to zero and the check gets its teeth.
    var err_bound = Scalar[F32](0) if integer_inputs else TOL

    var ok = (
        (bit_vs_gemv == 0)
        and (bit_vs_flat == 0)
        and (not saw_nan)
        and (max_err <= err_bound)
        and canary_ok
    )

    print(
        "PASS" if ok else "FAIL",
        "|",
        name,
        "[int]" if integer_inputs else "[rand]",
        "| M=",
        M,
        "N=",
        N,
        "K=",
        K,
        "| tiles=",
        tiles_lo,
        "..",
        tiles_hi,
        "| vs_gemv=",
        bit_vs_gemv,
        "| vs_untiled=",
        bit_vs_flat,
        "| max_err=",
        max_err,
        "<=",
        err_bound,
        "| nan=",
        saw_nan,
        "| canary_ok=",
        canary_ok,
    )

    A.free()
    B.free()
    C_bat.free()
    C_flat.free()
    C_row.free()
    C_ref.free()
    return ok


def check_tile_rows() -> Bool:
    """`mm_tile_rows` must return a positive multiple of `MM_MR`, always.

    The bit-exactness argument rests on every M-tile starting on a multiple of
    `MM_MR`: only then does the last tile's full-tile bound equal the untiled
    `(M // MM_MR) * MM_MR`, and only then do the same rows fall to the
    single-row remainder path as before the tiling. A budget that produced any
    other tile height would still compute correct values, but the equality in
    check 4 would no longer be guaranteed.
    """
    var Ks = [64, 128, 192, 256, 3072, 7680, 9216, 12288]
    var budgets = [
        1,
        63,
        127,
        512,
        1000,
        1536,
        4096,
        40960,
        MM_MC_BYTES,
        MC_UNTILED,
    ]
    var bad = 0
    for i in range(len(Ks)):
        for j in range(len(budgets)):
            var rows = mm_tile_rows[MM_MR](Ks[i], budgets[j])
            if rows < MM_MR or rows % MM_MR != 0:
                bad += 1
    print(
        "PASS" if bad == 0 else "FAIL",
        "| tile height is always a positive multiple of MR=",
        MM_MR,
        "| violations=",
        bad,
    )
    return bad == 0


def oracle_wrong_weight(M: Int, N: Int, K: Int, mc_bytes: Int) -> Bool:
    """Falsifiable control: a scaled weight must make the fp32 check fire.

    Returns True when the control fires (the batched result over the true
    weights diverges from the fp32 reference over the scaled weights by more than
    `TOL`). A False means the numeric check could not tell right weights from
    wrong — the check would be a placebo. Run at a budget that forces several
    M-tiles, so a tiling bug cannot turn the control into a no-op.
    """
    var a_n = M * K
    var b_n = N * K
    var c_n = M * N
    var A = alloc[Scalar[MM_DTYPE]](a_n).as_unsafe_any_origin()
    var B = alloc[Scalar[MM_DTYPE]](b_n).as_unsafe_any_origin()
    var B_bad = alloc[Scalar[MM_DTYPE]](b_n).as_unsafe_any_origin()
    var C_bat = alloc[Scalar[MM_DTYPE]](c_n).as_unsafe_any_origin()
    var C_ref_bad = alloc[Scalar[MM_DTYPE]](c_n).as_unsafe_any_origin()

    fill_random(A, a_n, 0x12345678, 1.0)
    fill_random(B, b_n, 0x0BADF00D, 0.05)
    for i in range(b_n):
        B_bad[i] = Scalar[MM_DTYPE](B[i].cast[F32]() * 1.5)

    run_core(C_bat, A, B, M, N, K, mc_bytes)
    matmul_ref_fp32(C_ref_bad, A, B_bad, M, N, K)

    var max_err = Scalar[F32](0)
    for i in range(c_n):
        max_err = max(
            max_err, abs(C_bat[i].cast[F32]() - C_ref_bad[i].cast[F32]())
        )

    var fired = max_err > TOL
    print(
        "ORACLE",
        "PASS" if fired else "FAIL",
        "| wrong-weight must diverge | M=",
        M,
        "N=",
        N,
        "K=",
        K,
        "| mc_bytes=",
        mc_bytes,
        "| max_err=",
        max_err,
        "> TOL=",
        TOL,
        "?",
        fired,
    )

    A.free()
    B.free()
    B_bad.free()
    C_bat.free()
    C_ref_bad.free()
    return fired


def main() raises:
    print(
        "==== HVX batched matmul host suite (MR=",
        MM_MR,
        "NR=",
        MM_NR,
        "MC_BYTES=",
        MM_MC_BYTES,
        ") ====",
    )
    var failures = 0

    if not check_tile_rows():
        failures += 1

    # Shapes cover: full MR x NR tiles, M tail (M%MR), N tail (N%NR), N not a
    # multiple of 64 (the tail-store fault shape), tiny M (gemv M=1), tiny N
    # (conv_out N=3), the MINI-block shape, and several K multiples of 64. The
    # last two carry the plan's two worst K — `sgl out-proj` and `sgl fused mlp`
    # — at a reduced M and N that keeps the K and the multi-tile count, which is
    # what the cache blocking is about, without the host runtime of the real M.
    var Ms = [8, 6, 8, 3, 5, 1, 4, 2, 64, 16, 5, 7, 86, 174]
    var Ns = [4, 4, 5, 130, 65, 64, 1, 3, 96, 130, 7, 128, 7, 34]
    var Ks = [64, 64, 128, 64, 64, 128, 64, 64, 64, 192, 256, 64, 12288, 3072]
    var names = [
        String("full-tiles"),
        String("M-tail"),
        String("N-tail-odd"),
        String("N-not-mult64"),
        String("both-tails-n65"),
        String("tiny-M1-gemv"),
        String("tiny-N1"),
        String("tiny-N3-convout"),
        String("mini-block-64x96"),
        String("M16-N130-K192"),
        String("K256"),
        String("N128-K64"),
        String("plan-sgl-outproj-K12288"),
        String("plan-sgl-mlp-K3072"),
    ]

    for i in range(len(Ms)):
        if not run_case(Ms[i], Ns[i], Ks[i], names[i], False):
            failures += 1
    for i in range(len(Ms)):
        if not run_case(Ms[i], Ns[i], Ks[i], names[i], True):
            failures += 1

    # Falsifiable controls: each must fire, or a numeric check is a placebo.
    var oracles_ok = 0
    var oracles_total = 0
    var om = [8, 3, 1, 86]
    var on = [4, 130, 64, 7]
    var ok_ = [64, 64, 128, 12288]
    var omc = [512, 512, MM_MC_BYTES, MM_MC_BYTES]
    for i in range(len(om)):
        oracles_total += 1
        if oracle_wrong_weight(om[i], on[i], ok_[i], omc[i]):
            oracles_ok += 1
    if oracles_ok != oracles_total:
        failures += oracles_total - oracles_ok

    print()
    if failures == 0:
        print("ALL MATMUL TESTS PASSED")
    else:
        raise Error(String(failures) + " matmul check(s) FAILED")
