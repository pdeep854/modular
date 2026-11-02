"""Off-device validation for the HVX flash-attention kernel (`npu_attention_kernel.mojo`).

Drives the actual kernel (`flash_attention_heads`) on the host -- where its
source lowers to host SIMD instead of HVX `V6_*` and its portable-MAC branch
stands in for the `V6_vmpy_sf_hf_acc` intrinsic -- against three independent
references, so agreement is evidence about the algorithm, not a tautology:

- `ref_naive_fp32`: full (non-tiled) softmax(scale * QK^T) @ V in fp32, P in
  fp32. The algorithm oracle.
- `ref_faithful_f16p`: the same, non-tiled, but modelling the kernel's exact
  numerics -- the softmax sum `l` accumulates fp32 probabilities while the P@V
  accumulator uses `fp16(p)`, then divides. The kernel's online recurrence is
  mathematically exact, so it must match this to fp32-accumulation-order slack.
  The tight correctness gate.
- `ref_flash_fp32`: an independent tiled online-softmax flash in fp32 (P fp32,
  fp32 O accumulator). Matching `ref_naive_fp32` proves the online recurrence
  equals the full softmax, independent of the kernel.

Coverage: dense `NULL_MASK` (every key), multi-head up to the FLUX 48, query and
key tiles that are exact multiples of `BM`/`BN` and tiles with tails, `head_dim`
64 and 128, sequence lengths from 1 up to a deep 18x9 tiling toward the M~4608
prefill, tile invariance (the canonical BM=32/BN=64 kernel vs an extreme
BM=1/BN=1 tiling must agree), and the RoPE prologue (interleaved, theta-agnostic
here: both paths read the same precomputed freqs) composed with attention.

The full 48-head x 4608-token prefill and the actual HVX instruction numerics
(the `V6_vmpy_sf_hf_acc` MAC and the `+hvx-qfloat` fp16<->fp32 path) are the
device lane's on-silicon confirm; the tiling here is sequence-length agnostic and
the deep-tiling cases exercise the same online accumulation the full prefill
does.
"""

from std.math import cos, exp, sin, sqrt
from std.memory import alloc
from std.sys import size_of
from std.testing import assert_true
from std.utils.numerics import min_or_neg_inf

from layout.coord import All, Coord
from layout.tile_layout import row_major

from hexagon_ondevice.attention_kernel import (
    BSHD,
    MMA_W,
    RowMajor2D,
    attention_thread_heads,
    flash_attention_heads,
    flash_attention_striped,
)
from hexagon_ondevice.rope_kernel import rope_apply_row

comptime F16 = DType.float16
comptime F32 = DType.float32
comptime F32P = Scalar[F32]

comptime Ptr16 = UnsafePointer[Scalar[F16], MutUnsafeAnyOrigin]
comptime Ptr32 = UnsafePointer[Scalar[F32], MutUnsafeAnyOrigin]


# ===-----------------------------------------------------------------------===#
# Inputs and comparison.
# ===-----------------------------------------------------------------------===#


def _fill_rand(ptr: Ptr16, n: Int, seed: UInt64):
    """Bounded deterministic fp16 fill in [-1, 1] via a 64-bit LCG.

    Bounded so the attention output (a convex combination of V rows) stays in
    [-1, 1], where fp16 resolves ~2^-11 and a real convention bug shows as an
    O(0.1-1) divergence, far above the gate.
    """
    var state = seed
    for i in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32((state >> 33) & 0xFFFF) / 65535.0
        ptr[i] = Scalar[F16](2.0 * u - 1.0)


def _maxdiff(a: Ptr16, b: Ptr16, n: Int) -> Float32:
    var worst = Float32(0)
    for i in range(n):
        var d = Float32(a[i]) - Float32(b[i])
        var ad = -d if d < 0 else d
        if ad > worst:
            worst = ad
    return worst


# ===-----------------------------------------------------------------------===#
# Independent references.
# ===-----------------------------------------------------------------------===#


def _ref_naive_fp32[
    D: Int
](
    Q: Ptr16,
    K: Ptr16,
    V: Ptr16,
    dst: Ptr16,
    seq_len: Int,
    num_heads: Int,
    scale: F32P,
):
    """Full softmax(scale * QK^T) @ V in fp32, P in fp32 -- the algorithm oracle.
    """
    var scores = alloc[Scalar[F32]](seq_len).as_unsafe_any_origin()
    var probs = alloc[Scalar[F32]](seq_len).as_unsafe_any_origin()
    var stride = num_heads * D
    for h in range(num_heads):
        for s in range(seq_len):
            var q = Q + s * stride + h * D
            var m = min_or_neg_inf[F32]()
            for t in range(seq_len):
                var k = K + t * stride + h * D
                var dot = F32P(0)
                for d in range(D):
                    dot += Float32(q[d]) * Float32(k[d])
                var sc = scale * dot
                scores[t] = sc
                if sc > m:
                    m = sc
            var denom = F32P(0)
            for t in range(seq_len):
                probs[t] = exp(scores[t] - m)
                denom += probs[t]
            var inv = F32P(1) / denom
            var o = dst + s * stride + h * D
            for d in range(D):
                var acc = F32P(0)
                for t in range(seq_len):
                    acc += probs[t] * inv * Float32(V[t * stride + h * D + d])
                o[d] = Scalar[F16](acc)
    scores.free()
    probs.free()


def _ref_faithful[
    D: Int
](
    Q: Ptr16,
    K: Ptr16,
    V: Ptr16,
    dst: Ptr16,
    seq_len: Int,
    num_heads: Int,
    scale: F32P,
):
    """Full softmax modelling the kernel's exact numerics (non-tiled).

    `l` sums fp32 probabilities; the P@V accumulator multiplies fp32 `p` by the
    widened V; the result is `(sum p * V) / l` rounded to fp16 -- what the online
    kernel computes once its (exact) running max reaches the global max.

    P was fp16 here until M4.0b, because the kernel narrowed `p` to feed a fp16
    MAC. That MAC is gone (the P@V lane-order fix replaced it with a fp32
    multiply against a widened V), so the narrowing was pure loss and the kernel
    no longer does it. Modelling it here would make this reference disagree with
    the kernel on purpose, which is the one thing a tight reference must not do.
    """
    var scores = alloc[Scalar[F32]](seq_len).as_unsafe_any_origin()
    var stride = num_heads * D
    for h in range(num_heads):
        for s in range(seq_len):
            var q = Q + s * stride + h * D
            var m = min_or_neg_inf[F32]()
            for t in range(seq_len):
                var k = K + t * stride + h * D
                var dot = F32P(0)
                for d in range(D):
                    dot += Float32(q[d]) * Float32(k[d])
                var sc = scale * dot
                scores[t] = sc
                if sc > m:
                    m = sc
            var l = F32P(0)
            for t in range(seq_len):
                l += exp(scores[t] - m)
            var inv = F32P(1) / l
            var o = dst + s * stride + h * D
            for d in range(D):
                var acc = F32P(0)
                for t in range(seq_len):
                    acc += exp(scores[t] - m) * Float32(
                        V[t * stride + h * D + d]
                    )
                o[d] = Scalar[F16](acc * inv)
    scores.free()


def _ref_flash_fp32[
    D: Int, BN: Int
](
    Q: Ptr16,
    K: Ptr16,
    V: Ptr16,
    dst: Ptr16,
    seq_len: Int,
    num_heads: Int,
    scale: F32P,
):
    """Independent tiled online-softmax flash in fp32 (P fp32, fp32 O accum).

    Structurally different from the kernel (fp32 O array, no fp16 P); matching
    `_ref_naive_fp32` proves the online recurrence equals the full softmax.
    """
    var stile = alloc[Scalar[F32]](BN).as_unsafe_any_origin()
    var o = alloc[Scalar[F32]](D).as_unsafe_any_origin()
    var stride = num_heads * D
    for h in range(num_heads):
        for s in range(seq_len):
            var q = Q + s * stride + h * D
            var m = min_or_neg_inf[F32]()
            var l = F32P(0)
            for d in range(D):
                o[d] = F32P(0)

            var kj = 0
            while kj < seq_len:
                var krows = BN
                if kj + BN > seq_len:
                    krows = seq_len - kj
                var tmax = min_or_neg_inf[F32]()
                for b in range(krows):
                    var k = K + (kj + b) * stride + h * D
                    var dot = F32P(0)
                    for d in range(D):
                        dot += Float32(q[d]) * Float32(k[d])
                    var sc = scale * dot
                    stile[b] = sc
                    if sc > tmax:
                        tmax = sc
                var m_new = m
                if tmax > m_new:
                    m_new = tmax
                var corr = exp(m - m_new)
                l *= corr
                for d in range(D):
                    o[d] *= corr
                for b in range(krows):
                    var p = exp(stile[b] - m_new)
                    l += p
                    for d in range(D):
                        o[d] += p * Float32(V[(kj + b) * stride + h * D + d])
                m = m_new
                kj += BN

            var inv = F32P(1) / l
            var op = dst + s * stride + h * D
            for d in range(D):
                op[d] = Scalar[F16](o[d] * inv)
    stile.free()
    o.free()


# ===-----------------------------------------------------------------------===#
# Cases.
# ===-----------------------------------------------------------------------===#


def _acts[D: Int](p: Ptr16, seq_len: Int, num_heads: Int) -> BSHD[F16]:
    """The `[seq_len, num_heads, D]` view of a test buffer.

    The kernel takes the tile, not a pointer plus the extents to read it with,
    so the test builds the seam its allocation implies -- the same one the
    device entry builds from the FastRPC skel's arguments.
    """
    return BSHD[F16](ptr=p, layout=row_major(Coord(seq_len, num_heads, D)))


def _run_case[
    D: Int, invariance: Bool
](seq_len: Int, num_heads: Int, seed: UInt64, name: String) raises:
    """Runs the kernel and the three references on one shape and gates them."""
    var n = seq_len * num_heads * D
    var scale = F32P(1.0) / sqrt(Float32(D))

    var q = alloc[Scalar[F16]](n).as_unsafe_any_origin()
    var k = alloc[Scalar[F16]](n).as_unsafe_any_origin()
    var v = alloc[Scalar[F16]](n).as_unsafe_any_origin()
    var got = alloc[Scalar[F16]](n).as_unsafe_any_origin()
    var got_ext = alloc[Scalar[F16]](n).as_unsafe_any_origin()
    var naive = alloc[Scalar[F16]](n).as_unsafe_any_origin()
    var faithful = alloc[Scalar[F16]](n).as_unsafe_any_origin()
    var flash = alloc[Scalar[F16]](n).as_unsafe_any_origin()

    _fill_rand(q, n, seed)
    _fill_rand(k, n, seed ^ 0x9E3779B97F4A7C15)
    _fill_rand(v, n, seed ^ 0xD1B54A32D192ED03)

    var q_t = _acts[D](q, seq_len, num_heads)
    var k_t = _acts[D](k, seq_len, num_heads)
    var v_t = _acts[D](v, seq_len, num_heads)

    # Canonical FLUX tiling BM=32/BN=64 (the device entry's tile).
    flash_attention_heads[use_hvx=False, BM=32, BN=64, D=D, W=MMA_W](
        q_t, k_t, v_t, _acts[D](got, seq_len, num_heads), 0, num_heads, scale
    )
    _ref_naive_fp32[D](q, k, v, naive, seq_len, num_heads, scale)
    _ref_faithful[D](q, k, v, faithful, seq_len, num_heads, scale)
    _ref_flash_fp32[D, BN=64](q, k, v, flash, seq_len, num_heads, scale)

    var d_tiling = _maxdiff(flash, naive, n)  # online == full softmax (fp32)
    var d_tight = _maxdiff(got, faithful, n)  # kernel bookkeeping exact
    var d_algo = _maxdiff(got, naive, n)  # kernel really computes attention

    var d_inv = Float32(0)
    comptime if invariance:
        # Extreme tiling: one query row and one key per tile. A different
        # accumulation path that must land on the same answer.
        flash_attention_heads[use_hvx=False, BM=1, BN=1, D=D, W=MMA_W](
            q_t,
            k_t,
            v_t,
            _acts[D](got_ext, seq_len, num_heads),
            0,
            num_heads,
            scale,
        )
        d_inv = _maxdiff(got, got_ext, n)

    print(
        "PASS?",
        name,
        "| H=",
        num_heads,
        "seq=",
        seq_len,
        "D=",
        D,
        "| tiling=",
        d_tiling,
        "tight=",
        d_tight,
        "algo=",
        d_algo,
        "inv=",
        d_inv,
    )

    # Independent-reference cross-check: online softmax == full softmax in fp32.
    assert_true(d_tiling <= 1.0e-3, String("tiling gate: ") + name)
    # Kernel matches its faithful model to fp32-accumulation-order slack.
    assert_true(d_tight <= 4.0e-3, String("tight gate: ") + name)
    # Kernel computes genuine attention (fp16 store rounding absorbed).
    assert_true(d_algo <= 1.5e-2, String("algo gate: ") + name)
    comptime if invariance:
        assert_true(d_inv <= 4.0e-3, String("tile-invariance gate: ") + name)

    q.free()
    k.free()
    v.free()
    got.free()
    got_ext.free()
    naive.free()
    faithful.free()
    flash.free()


def _check_thread_heads() raises:
    """Every head belongs to exactly one compute thread, and the DMA thread owns
    none.

    Scored by covering count so a split bug localizes here rather than being read
    off a numeric mismatch, and checked over head counts that do and do not
    divide by the thread count. The fifth thread owning no heads is what leaves
    it free to hold the HMX grant, so it is asserted rather than assumed.
    """
    for num_heads in range(0, 40):
        var covered = alloc[Scalar[DType.int32]](num_heads + 1)
        for i in range(num_heads):
            covered[i] = 0
        for t in range(5):
            var run = attention_thread_heads(num_heads, t)
            assert_true(run[0] <= run[1], "thread run is inverted")
            if t >= 4:
                assert_true(run[0] == run[1], "the DMA thread owns heads")
            for h in range(run[0], run[1]):
                assert_true(h < num_heads, "a run reaches past the last head")
                covered[h] += 1
        for h in range(num_heads):
            assert_true(
                covered[h] == 1,
                String(
                    "heads=", num_heads, " head ", h, " is not covered once"
                ),
            )
        covered.free()


def _run_striped[D: Int](seq_len: Int, num_heads: Int, seed: UInt64) raises:
    """The five launch indices, unioned, reproduce the whole-head result.

    Both arms run the same head body, so the gate is EQUALITY rather than a
    tolerance: any difference is a partition bug, not accumulation slack. The
    striped output is pre-filled with a sentinel no attention output can be (its
    inputs are bounded to [-1, 1] and the output is a convex combination of V
    rows), so a head no thread claimed fails loudly instead of reading as a
    plausible zero.
    """
    var n = seq_len * num_heads * D
    var scale = F32P(1.0) / sqrt(Float32(D))

    var q = alloc[Scalar[F16]](n).as_unsafe_any_origin()
    var k = alloc[Scalar[F16]](n).as_unsafe_any_origin()
    var v = alloc[Scalar[F16]](n).as_unsafe_any_origin()
    var whole = alloc[Scalar[F16]](n).as_unsafe_any_origin()
    var striped = alloc[Scalar[F16]](n).as_unsafe_any_origin()

    _fill_rand(q, n, seed)
    _fill_rand(k, n, seed ^ 0x9E3779B97F4A7C15)
    _fill_rand(v, n, seed ^ 0xD1B54A32D192ED03)
    for i in range(n):
        striped.store(i, Scalar[F16](-31000.0))

    var q_t = _acts[D](q, seq_len, num_heads)
    var k_t = _acts[D](k, seq_len, num_heads)
    var v_t = _acts[D](v, seq_len, num_heads)

    flash_attention_heads[use_hvx=False, BM=32, BN=64, D=D, W=MMA_W](
        q_t, k_t, v_t, _acts[D](whole, seq_len, num_heads), 0, num_heads, scale
    )
    for t in range(5):
        flash_attention_striped[use_hvx=False, BM=32, BN=64, D=D, W=MMA_W](
            q_t,
            k_t,
            v_t,
            _acts[D](striped, seq_len, num_heads),
            t,
            scale,
        )

    for i in range(n):
        assert_true(
            whole.load(i) == striped.load(i),
            String("striped union differs at ", i, ", heads=", num_heads),
        )

    print("PASS? striped | H=", num_heads, "seq=", seq_len, "D=", D)

    q.free()
    k.free()
    v.free()
    whole.free()
    striped.free()


def _run_rope_prologue[
    D: Int
](seq_len: Int, num_heads: Int, seed: UInt64) raises:
    """RoPE (interleaved) applied to Q/K, then attention, vs the same prologue
    into the naive reference.

    The kernel does not rope -- RoPE is the verified separate prologue kernel --
    so this proves the two compose: `attention(rope(Q), rope(K), V)` matches the
    oracle fed the same roped Q/K. The freqs schedule is arbitrary here (both
    paths read the same buffer); the theta=2000 interleaved layout is a host-side
    freqs precompute, not a kernel concern.
    """
    var n = seq_len * num_heads * D
    var scale = F32P(1.0) / sqrt(Float32(D))
    var half = D // 2

    var q = alloc[Scalar[F16]](n).as_unsafe_any_origin()
    var k = alloc[Scalar[F16]](n).as_unsafe_any_origin()
    var v = alloc[Scalar[F16]](n).as_unsafe_any_origin()
    var qr = alloc[Scalar[F16]](n).as_unsafe_any_origin()
    var kr = alloc[Scalar[F16]](n).as_unsafe_any_origin()
    var got = alloc[Scalar[F16]](n).as_unsafe_any_origin()
    var naive = alloc[Scalar[F16]](n).as_unsafe_any_origin()
    var freqs = alloc[Scalar[F32]](seq_len * D).as_unsafe_any_origin()

    _fill_rand(q, n, seed)
    _fill_rand(k, n, seed ^ 0x9E3779B97F4A7C15)
    _fill_rand(v, n, seed ^ 0xD1B54A32D192ED03)

    # cos/sin interleaved per pair, one row per position.
    for pos in range(seq_len):
        for p in range(half):
            var angle = Float32(pos) * 0.5 / (1.0 + Float32(p))
            freqs[pos * D + 2 * p] = Scalar[F32](cos(angle))
            freqs[pos * D + 2 * p + 1] = Scalar[F32](sin(angle))

    # Rope every (token, head) vector against its position's freqs row.
    #
    # The prologue is driven through the tile model both kernels ship as: the
    # activations are the attention kernel's own rank-3
    # `[seq_len, num_heads, head_dim]` tile and `freqs` is a `[seq_len, D]`
    # table, so a (token, head) vector and its position's freqs row are both one
    # `.slice(...)`. That the rank-3 slice and the rank-2 slice are the same
    # rank-1 row type -- which is what lets a rope kernel written against a
    # 2-D table consume a 3-D activation's row unconverted -- is a property of
    # this composition, and compiling this prelude is what checks it.
    var acts = row_major(Coord(seq_len, num_heads, D))
    var q_tile = BSHD[F16](ptr=q, layout=acts)
    var k_tile = BSHD[F16](ptr=k, layout=acts)
    var qr_tile = BSHD[F16](ptr=qr, layout=acts)
    var kr_tile = BSHD[F16](ptr=kr, layout=acts)
    var v_tile = BSHD[F16](ptr=v, layout=acts)
    var got_tile = BSHD[F16](ptr=got, layout=acts)
    var freq_tile = RowMajor2D[F32](
        ptr=freqs, layout=row_major(Coord(seq_len, D))
    )

    for s in range(seq_len):
        var freq_row = freq_tile.slice(s, All)
        for h in range(num_heads):
            rope_apply_row[interleaved=True, W=MMA_W](
                q_tile.slice(s, h, All), qr_tile.slice(s, h, All), freq_row
            )
            rope_apply_row[interleaved=True, W=MMA_W](
                k_tile.slice(s, h, All), kr_tile.slice(s, h, All), freq_row
            )

    # The roped tiles go straight in: the prologue's output view and the
    # attention kernel's input argument are now the same typed object, so the
    # composition needs no re-wrap and no restated extent between the two.
    flash_attention_heads[use_hvx=False, BM=32, BN=64, D=D, W=MMA_W](
        qr_tile, kr_tile, v_tile, got_tile, 0, num_heads, scale
    )
    _ref_naive_fp32[D](qr, kr, v, naive, seq_len, num_heads, scale)

    var d_rope = _maxdiff(got, naive, n)
    print(
        "PASS? rope-prologue | H=",
        num_heads,
        "seq=",
        seq_len,
        "D=",
        D,
        "| algo=",
        d_rope,
    )
    assert_true(d_rope <= 1.5e-2, "rope-prologue gate")

    q.free()
    k.free()
    v.free()
    qr.free()
    kr.free()
    got.free()
    naive.free()
    freqs.free()


def main() raises:
    # Small first: degenerate single token, sub-tile shapes, D=64 and D=128.
    _run_case[D=128, invariance=True](1, 1, 101, "single-token")
    _run_case[D=64, invariance=True](8, 1, 102, "sub-tile-d64")
    _run_case[D=128, invariance=True](8, 2, 103, "sub-tile-multihead")

    # Tails: q-tile tail (33 = 32 + 1), both-axis tails (100 = 3*32+4 q, 64+36 k).
    _run_case[D=128, invariance=True](33, 2, 104, "qtile-tail")
    _run_case[D=128, invariance=True](100, 3, 105, "both-tails")

    # Clean multiples and full FLUX head count.
    _run_case[D=128, invariance=True](128, 4, 106, "clean-4tiles")
    _run_case[D=128, invariance=False](64, 48, 107, "flux-48heads-seq64")
    _run_case[D=128, invariance=False](128, 48, 108, "flux-48heads-seq128")

    # Toward the M~4608 prefill: deep 18x9 tiling of the online accumulation.
    _run_case[D=128, invariance=False](576, 2, 109, "deep-tiling-576")

    # The launch-index head split, which the device entries now share: the
    # partition itself, then the union of the five launch indices against the
    # whole-head run at head counts that divide, do not divide, and fall below
    # the compute-thread count.
    _check_thread_heads()
    _run_striped[D=128](24, 1, 301)
    _run_striped[D=128](24, 3, 302)
    _run_striped[D=128](24, 4, 303)
    _run_striped[D=128](24, 7, 304)
    _run_striped[D=64](40, 8, 305)

    # RoPE prologue composed with attention.
    _run_rope_prologue[D=128](48, 2, 201)
    _run_rope_prologue[D=128](96, 3, 202)

    print("ALL FLASH-ATTENTION ALGORITHM TESTS PASSED")
