"""Off-device validation for the causal + padding masked HVX flash attention.

Drives the actual masked kernel (`flash_attention_heads_masked` in
`npu_attention_kernel.mojo`) on the host -- where its portable-MAC branch
(`use_hvx=False`) stands in for the `V6_vmpy_sf_hf` intrinsic -- against
independent fp32 references, and proves three things a dense attention test
cannot:

- **The mask is applied.** Masked flash vs a full (non-tiled)
  `softmax(scale * QK^T + bias) @ V` in fp32, over a causal mask, a padding
  mask, a combined causal+padding mask, and a sequence tail that straddles the
  key tiles. `ref_masked_naive_fp32` (P fp32) is the algorithm oracle;
  `ref_masked_faithful` (fp32 `l`, fp32-P accumulator) is the tight gate.
- **The base algorithm did not regress.** With `bias = 0` the masked kernel must
  reproduce the verified dense kernel `flash_attention_heads` bit-for-bit (fp32
  `x + 0 == x`), and match the dense fp32 reference -- so the added mask path
  costs the `NULL_MASK` numerics nothing.
- **A masked position does not leak.** Under a causal mask, perturbing K and V at
  a future key position must leave every earlier query row's output unchanged
  (bit-identical). The same perturbation under the dense kernel *does* change
  those rows -- so the control is live, not vacuous: a leak would fire it.

The bias is fp32 `[seq_len, seq_len]`, 0 where visible and a large finite
negative (`NEG`) where masked, broadcast across heads -- the additive mask MAX's
`masked_flash_attention_gpu` consumes, with the causal triangle and padding
folded together by the host. GQA is a host concern (K/V materialized to full
head count), so this kernel is dense MHA; the tests use equal head counts.

The actual HVX `V6_vmpy_sf_hf` numerics are the device lane's on-silicon confirm
(the device is currently human-gated); the masking logic here is exact and
sequence-length agnostic.
"""

from std.math import exp, sqrt
from std.memory import alloc
from std.testing import assert_true
from std.utils.numerics import min_or_neg_inf

from layout.coord import Coord
from layout.tile_layout import row_major

from hexagon_ondevice.attention_kernel import (
    BSHD,
    MMA_W,
    RowMajor2D,
    flash_attention_heads,
    flash_attention_heads_masked,
    flash_attention_masked_striped,
)

comptime F16 = DType.float16
comptime F32 = DType.float32
comptime F32P = Scalar[F32]

comptime Ptr16 = UnsafePointer[Scalar[F16], MutUnsafeAnyOrigin]
comptime Ptr32 = UnsafePointer[Scalar[F32], MutUnsafeAnyOrigin]

comptime NEG = F32P(-1.0e9)
"""The masked-position sentinel: large and finite, so a masked key contributes
`exp(-inf) = 0` while a fully-masked row stays a finite average rather than a
`NaN` (the host builds the bias the same way MAX does, not with a raw `-inf`)."""


# ===-----------------------------------------------------------------------===#
# Inputs and comparison.
# ===-----------------------------------------------------------------------===#


def _fill_rand(ptr: Ptr16, n: Int, seed: UInt64):
    """Bounded deterministic fp16 fill in [-1, 1] via a 64-bit LCG."""
    var state = seed
    for i in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32((state >> 33) & 0xFFFF) / 65535.0
        ptr[i] = Scalar[F16](2.0 * u - 1.0)


def _maxdiff(a: Ptr16, b: Ptr16, n: Int) -> Float32:
    """Largest absolute fp16 elementwise difference over the first `n` elements.
    """
    var worst = Float32(0)
    for i in range(n):
        var d = Float32(a[i]) - Float32(b[i])
        var ad = -d if d < 0 else d
        if ad > worst:
            worst = ad
    return worst


# ===-----------------------------------------------------------------------===#
# Additive masks (host-precomputed, `[seq_len, seq_len]` fp32).
# ===-----------------------------------------------------------------------===#


def _bias_zero(bias: Ptr32, seq_len: Int):
    """The `NULL_MASK` bias: every position visible."""
    for i in range(seq_len * seq_len):
        bias[i] = F32P(0)


def _bias_causal(bias: Ptr32, seq_len: Int):
    """Lower-triangular: query `i` sees keys `j <= i`."""
    for i in range(seq_len):
        for j in range(seq_len):
            bias[i * seq_len + j] = F32P(0) if j <= i else NEG


def _bias_padding(bias: Ptr32, seq_len: Int, valid_keys: Int):
    """Key columns `>= valid_keys` masked for every query (right padding)."""
    for i in range(seq_len):
        for j in range(seq_len):
            bias[i * seq_len + j] = F32P(0) if j < valid_keys else NEG


def _bias_causal_padding(bias: Ptr32, seq_len: Int, valid_keys: Int):
    """Causal and right-padding folded into one bias, as the encoder host does.
    """
    for i in range(seq_len):
        for j in range(seq_len):
            var keep = j <= i and j < valid_keys
            bias[i * seq_len + j] = F32P(0) if keep else NEG


# ===-----------------------------------------------------------------------===#
# Independent masked references.
# ===-----------------------------------------------------------------------===#


def _ref_masked_naive_fp32[
    D: Int
](
    Q: Ptr16,
    K: Ptr16,
    V: Ptr16,
    bias: Ptr32,
    dst: Ptr16,
    seq_len: Int,
    num_heads: Int,
    scale: F32P,
):
    """Full `softmax(scale * QK^T + bias) @ V` in fp32, P in fp32 -- the oracle.
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
                var sc = scale * dot + bias[s * seq_len + t]
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


def _ref_masked_faithful[
    D: Int
](
    Q: Ptr16,
    K: Ptr16,
    V: Ptr16,
    bias: Ptr32,
    dst: Ptr16,
    seq_len: Int,
    num_heads: Int,
    scale: F32P,
):
    """Full masked softmax modelling the kernel's exact numerics (non-tiled).

    `l` sums fp32 probabilities; the P@V accumulator multiplies fp32 `p` by the
    widened V (P was fp16 here until M4.0b removed the narrowing) -- what the
    online masked kernel computes once its running max reaches the global max.
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
                var sc = scale * dot + bias[s * seq_len + t]
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


# ===-----------------------------------------------------------------------===#
# Cases.
# ===-----------------------------------------------------------------------===#


def _alloc16(n: Int) -> Ptr16:
    return alloc[Scalar[F16]](n).as_unsafe_any_origin()


def _acts[D: Int](p: Ptr16, seq_len: Int, num_heads: Int) -> BSHD[F16]:
    """The `[seq_len, num_heads, D]` view of a test buffer.

    The kernel takes the tile, not a pointer plus the extents to read it with,
    so the test builds the seam its allocation implies -- the same one the
    device entry builds from the FastRPC skel's arguments.
    """
    return BSHD[F16](ptr=p, layout=row_major(Coord(seq_len, num_heads, D)))


def _mask(p: Ptr32, seq_len: Int) -> RowMajor2D[F32]:
    """The `[seq_len, seq_len]` additive-bias view, broadcast across heads."""
    return RowMajor2D[F32](ptr=p, layout=row_major(Coord(seq_len, seq_len)))


def _run_masked_striped[
    D: Int
](seq_len: Int, num_heads: Int, bias: Ptr32, seed: UInt64) raises:
    """The five launch indices, unioned, reproduce the whole-head masked result.

    Both arms run the same head body, so the gate is EQUALITY: any difference is
    a partition bug, not accumulation slack. The striped output starts at a
    sentinel no attention output can be, so a head no thread claimed fails
    loudly rather than reading as a plausible zero. The masked entry the
    manifest binds is the striped call with a witness in front of it, so this is
    the split that entry runs.
    """
    var n = seq_len * num_heads * D
    var scale = F32P(1.0) / sqrt(Float32(D))

    var q = _alloc16(n)
    var k = _alloc16(n)
    var v = _alloc16(n)
    var whole = _alloc16(n)
    var striped = _alloc16(n)

    _fill_rand(q, n, seed)
    _fill_rand(k, n, seed ^ 0x9E3779B97F4A7C15)
    _fill_rand(v, n, seed ^ 0xD1B54A32D192ED03)
    for i in range(n):
        striped.store(i, Scalar[F16](-31000.0))

    flash_attention_heads_masked[use_hvx=False, BM=32, BN=64, D=D, W=MMA_W](
        _acts[D](q, seq_len, num_heads),
        _acts[D](k, seq_len, num_heads),
        _acts[D](v, seq_len, num_heads),
        _acts[D](whole, seq_len, num_heads),
        _mask(bias, seq_len),
        0,
        num_heads,
        scale,
    )
    for t in range(5):
        flash_attention_masked_striped[
            use_hvx=False, BM=32, BN=64, D=D, W=MMA_W
        ](
            _acts[D](q, seq_len, num_heads),
            _acts[D](k, seq_len, num_heads),
            _acts[D](v, seq_len, num_heads),
            _acts[D](striped, seq_len, num_heads),
            _mask(bias, seq_len),
            t,
            scale,
        )

    for i in range(n):
        assert_true(
            whole.load(i) == striped.load(i),
            String("masked striped union differs at ", i),
        )
    print("PASS? masked-striped | H=", num_heads, "seq=", seq_len)

    q.free()
    k.free()
    v.free()
    whole.free()
    striped.free()


def _run_masked[
    D: Int
](
    seq_len: Int,
    num_heads: Int,
    bias: Ptr32,
    seed: UInt64,
    name: String,
) raises:
    """Runs the masked kernel against the two masked references on one shape."""
    var n = seq_len * num_heads * D
    var scale = F32P(1.0) / sqrt(Float32(D))

    var q = _alloc16(n)
    var k = _alloc16(n)
    var v = _alloc16(n)
    var got = _alloc16(n)
    var naive = _alloc16(n)
    var faithful = _alloc16(n)

    _fill_rand(q, n, seed)
    _fill_rand(k, n, seed ^ 0x9E3779B97F4A7C15)
    _fill_rand(v, n, seed ^ 0xD1B54A32D192ED03)

    flash_attention_heads_masked[use_hvx=False, BM=32, BN=64, D=D, W=MMA_W](
        _acts[D](q, seq_len, num_heads),
        _acts[D](k, seq_len, num_heads),
        _acts[D](v, seq_len, num_heads),
        _acts[D](got, seq_len, num_heads),
        _mask(bias, seq_len),
        0,
        num_heads,
        scale,
    )
    _ref_masked_naive_fp32[D](q, k, v, bias, naive, seq_len, num_heads, scale)
    _ref_masked_faithful[D](q, k, v, bias, faithful, seq_len, num_heads, scale)

    var d_algo = _maxdiff(got, naive, n)
    var d_tight = _maxdiff(got, faithful, n)

    print(
        "PASS?",
        name,
        "| H=",
        num_heads,
        "seq=",
        seq_len,
        "D=",
        D,
        "| algo=",
        d_algo,
        "tight=",
        d_tight,
    )
    assert_true(d_algo <= 1.5e-2, String("masked algo gate: ") + name)
    assert_true(d_tight <= 4.0e-3, String("masked tight gate: ") + name)

    q.free()
    k.free()
    v.free()
    got.free()
    naive.free()
    faithful.free()


def _run_nullmask_regression[
    D: Int
](seq_len: Int, num_heads: Int, seed: UInt64, name: String) raises:
    """`bias = 0` must reproduce the verified dense kernel bit-for-bit.

    The only delta from `flash_attention_heads` is `+ bias[query, key]`; with the
    bias zero, fp32 `x + 0 == x`, so the masked kernel and the verified dense
    kernel must agree exactly (`d_vs_dense == 0`), and both must match the dense
    fp32 reference. This is the no-regression proof for the base algorithm.
    """
    var n = seq_len * num_heads * D
    var scale = F32P(1.0) / sqrt(Float32(D))

    var q = _alloc16(n)
    var k = _alloc16(n)
    var v = _alloc16(n)
    var bias = alloc[Scalar[F32]](seq_len * seq_len).as_unsafe_any_origin()
    var got_masked = _alloc16(n)
    var got_dense = _alloc16(n)
    var naive = _alloc16(n)

    _fill_rand(q, n, seed)
    _fill_rand(k, n, seed ^ 0x9E3779B97F4A7C15)
    _fill_rand(v, n, seed ^ 0xD1B54A32D192ED03)
    _bias_zero(bias, seq_len)

    var q_t = _acts[D](q, seq_len, num_heads)
    var k_t = _acts[D](k, seq_len, num_heads)
    var v_t = _acts[D](v, seq_len, num_heads)

    flash_attention_heads_masked[use_hvx=False, BM=32, BN=64, D=D, W=MMA_W](
        q_t,
        k_t,
        v_t,
        _acts[D](got_masked, seq_len, num_heads),
        _mask(bias, seq_len),
        0,
        num_heads,
        scale,
    )
    flash_attention_heads[use_hvx=False, BM=32, BN=64, D=D, W=MMA_W](
        q_t,
        k_t,
        v_t,
        _acts[D](got_dense, seq_len, num_heads),
        0,
        num_heads,
        scale,
    )
    _ref_masked_naive_fp32[D](q, k, v, bias, naive, seq_len, num_heads, scale)

    var d_vs_dense = _maxdiff(got_masked, got_dense, n)
    var d_algo = _maxdiff(got_masked, naive, n)
    print(
        "PASS?",
        name,
        "| H=",
        num_heads,
        "seq=",
        seq_len,
        "| masked-vs-dense=",
        d_vs_dense,
        "algo=",
        d_algo,
    )
    # Byte-identical to the verified dense kernel: the mask path is free at bias=0.
    assert_true(d_vs_dense == 0.0, String("null-mask == dense: ") + name)
    assert_true(d_algo <= 1.5e-2, String("null-mask algo gate: ") + name)

    q.free()
    k.free()
    v.free()
    bias.free()
    got_masked.free()
    got_dense.free()
    naive.free()


def _run_causal_leak_oracle[
    D: Int
](seq_len: Int, num_heads: Int, seed: UInt64) raises:
    """A masked (future) key/value must not leak into an earlier query row.

    Under a causal mask, query rows `< t` never attend to key `t`. Perturbing K
    and V at `t = seq_len - 1` must leave rows `0..seq_len-2` bit-identical
    (`leak == 0`). The same perturbation on the dense kernel changes those rows
    (`dense_delta > 0`), so the oracle is live: a kernel that ignored the mask
    would fail the `leak == 0` assertion.
    """
    var n = seq_len * num_heads * D
    var stride = num_heads * D
    var scale = F32P(1.0) / sqrt(Float32(D))
    var t_pert = seq_len - 1
    var early = t_pert * stride  # elements of rows 0..seq_len-2

    var q = _alloc16(n)
    var k = _alloc16(n)
    var v = _alloc16(n)
    var k2 = _alloc16(n)
    var v2 = _alloc16(n)
    var bias = alloc[Scalar[F32]](seq_len * seq_len).as_unsafe_any_origin()

    _fill_rand(q, n, seed)
    _fill_rand(k, n, seed ^ 0x9E3779B97F4A7C15)
    _fill_rand(v, n, seed ^ 0xD1B54A32D192ED03)
    _bias_causal(bias, seq_len)

    # Perturb K and V only at the last key position, out of the [-1, 1] range so a
    # leak is unmistakable.
    for i in range(n):
        k2[i] = k[i]
        v2[i] = v[i]
    for h in range(num_heads):
        for d in range(D):
            k2[t_pert * stride + h * D + d] = Scalar[F16](3.0)
            v2[t_pert * stride + h * D + d] = Scalar[F16](3.0)

    var q_t = _acts[D](q, seq_len, num_heads)
    var k_t = _acts[D](k, seq_len, num_heads)
    var v_t = _acts[D](v, seq_len, num_heads)
    var k2_t = _acts[D](k2, seq_len, num_heads)
    var v2_t = _acts[D](v2, seq_len, num_heads)
    var mask = _mask(bias, seq_len)

    var o1 = _alloc16(n)
    var o2 = _alloc16(n)
    flash_attention_heads_masked[use_hvx=False, BM=32, BN=64, D=D, W=MMA_W](
        q_t,
        k_t,
        v_t,
        _acts[D](o1, seq_len, num_heads),
        mask,
        0,
        num_heads,
        scale,
    )
    flash_attention_heads_masked[use_hvx=False, BM=32, BN=64, D=D, W=MMA_W](
        q_t,
        k2_t,
        v2_t,
        _acts[D](o2, seq_len, num_heads),
        mask,
        0,
        num_heads,
        scale,
    )
    var leak = _maxdiff(o1, o2, early)

    # Live-control: the same perturbation must move the dense (unmasked) output.
    var d1 = _alloc16(n)
    var d2 = _alloc16(n)
    flash_attention_heads[use_hvx=False, BM=32, BN=64, D=D, W=MMA_W](
        q_t, k_t, v_t, _acts[D](d1, seq_len, num_heads), 0, num_heads, scale
    )
    flash_attention_heads[use_hvx=False, BM=32, BN=64, D=D, W=MMA_W](
        q_t, k2_t, v2_t, _acts[D](d2, seq_len, num_heads), 0, num_heads, scale
    )
    var dense_delta = _maxdiff(d1, d2, early)

    print(
        "PASS? causal-leak-oracle | H=",
        num_heads,
        "seq=",
        seq_len,
        "| leak(masked)=",
        leak,
        "dense_delta(control)=",
        dense_delta,
    )
    assert_true(
        leak == 0.0,
        "causal leak: a future key/value changed an earlier masked row",
    )
    assert_true(
        dense_delta > 0.0,
        "leak oracle is inert: the perturbation did not move the dense output",
    )

    q.free()
    k.free()
    v.free()
    k2.free()
    v2.free()
    bias.free()
    o1.free()
    o2.free()
    d1.free()
    d2.free()


def main() raises:
    # Causal: sub-tile, a q-tile tail (100 = 3*32 + 4), and a clean multi-tile.
    var causal_s = [8, 33, 100, 128]
    var causal_h = [1, 2, 3, 4]
    for idx in range(len(causal_s)):
        var s = causal_s[idx]
        var bias = alloc[Scalar[F32]](s * s).as_unsafe_any_origin()
        _bias_causal(bias, s)
        _run_masked[D=128](
            s,
            causal_h[idx],
            bias,
            300 + UInt64(idx),
            String("causal-") + String(s),
        )
        bias.free()

    # Padding: mask the right tail of keys; a key-tile tail straddles BN=64.
    var pad = alloc[Scalar[F32]](100 * 100).as_unsafe_any_origin()
    _bias_padding(pad, 100, 72)  # 72 valid keys, 28 padded, straddles BN
    _run_masked[D=128](100, 3, pad, 401, "padding-72of100")
    pad.free()

    # Combined causal + padding, with a sequence tail.
    var comb = alloc[Scalar[F32]](100 * 100).as_unsafe_any_origin()
    _bias_causal_padding(comb, 100, 80)
    _run_masked[D=128](100, 2, comb, 402, "causal+pad-80of100")
    comb.free()

    # Base-algorithm regression: bias = 0 reproduces the verified dense kernel.
    _run_nullmask_regression[D=128](64, 2, 501, "nullmask-clean")
    _run_nullmask_regression[D=128](100, 3, 502, "nullmask-tails")

    # A masked position must not leak into an earlier row.
    _run_causal_leak_oracle[D=128](64, 2, 601)
    _run_causal_leak_oracle[D=128](100, 3, 602)

    # The launch-index head split the manifest's masked entry runs, at head
    # counts that divide, do not divide, and fall below the thread count.
    var split = alloc[Scalar[F32]](64 * 64).as_unsafe_any_origin()
    _bias_causal(split, 64)
    _run_masked_striped[D=128](64, 3, split, 701)
    _run_masked_striped[D=128](64, 4, split, 702)
    _run_masked_striped[D=128](64, 7, split, 703)
    split.free()

    print("ALL MASKED FLASH-ATTENTION TESTS PASSED")
