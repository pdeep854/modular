# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Host validation for the `sinusoid_embed` / `rope_table` table kernels.

Runs the actual kernels (`sincos_tables.mojo`, the source the device library
duplicates) against the host libm in float64, across the FLUX.2 geometries:
H=128 timestep bands, the [4608, 4] packed id tensor read by column, and the
per-axis 16-frequency rope rows.

Three kinds of check, because they fail differently:

- Closed-form anchors, exact with no tolerance: theta 0 is (sin 0, cos 1) and
  exp(0) is 1 by construction of the reduction and cores, a zero timestep row
  is all-ones cos and all-zeros sin after fp16 rounding, a zero position row
  is (1, 0) pairs, and negating every position negates exactly the sin bits
  and leaves the cos bits -- the cores are exactly odd/even in the reduced
  argument, so sign symmetry is a bit invariant, not an approximate one.
- Wiring, exact with no tolerance: a column read out of the packed id tensor
  through the stride must be bit-identical to the same column densely copied,
  and flipping the half order must swap the halves bit for bit. The fills are
  non-uniform so a permutation cannot cancel.
- The independent reference: libm's f64 sin/cos/exp over the same quantized
  f32 inputs, rounded once to the output dtype. One ulp of slack, because the
  platform libm is itself only faithfully rounded; the kernel measured 0 ulp
  against glibc over these domains, and the device check needs no slack at
  all -- it compares against a bit-replica of this same op sequence.
"""

from std.math import cos, exp, sin
from std.memory import alloc, bitcast
from std.testing import assert_equal, assert_raises, assert_true, TestSuite

from hexagon_ondevice.registrations import _require_sincos_domain
from hexagon_ondevice.sincos_tables import (
    _exp_f32,
    _sincos_f32,
    _thread_rows,
    rope_table_2d,
    rope_table_striped,
    sinusoid_embed_2d,
    sinusoid_embed_striped,
)

comptime F16 = DType.float16
comptime F32 = DType.float32
comptime F64 = DType.float64
comptime I64 = DType.int64

comptime NEG_LN_10000 = Float32(-9.210340371976184)
"""-ln(10000) rounded once to fp32: the graph's folded band constant."""


def _ulp_f32(a: Float32, b: Float32) -> Int:
    """Distance in fp32 representation order, signed zeros collapsed."""
    var ia = Int(bitcast[DType.int32](a))
    var ib = Int(bitcast[DType.int32](b))
    if ia < 0:
        ia = -2147483648 - ia
    if ib < 0:
        ib = -2147483648 - ib
    var d = ia - ib
    return d if d >= 0 else -d


def _ulp_f16(a: Scalar[F16], b: Scalar[F16]) -> Int:
    var ia = Int(bitcast[DType.int16](a))
    var ib = Int(bitcast[DType.int16](b))
    if ia < 0:
        ia = -32768 - ia
    if ib < 0:
        ib = -32768 - ib
    var d = ia - ib
    return d if d >= 0 else -d


def test_sincos_anchors() raises:
    var sc = _sincos_f32(0.0)
    assert_equal(sc[0], Float32(0.0))
    assert_equal(sc[1], Float32(1.0))
    assert_equal(_exp_f32(0.0), Float32(1.0))


def test_sincos_sign_symmetry_is_bit_exact() raises:
    for p in range(1, 4700, 3):
        var theta = Float32(p) * Float32(0.000731)
        var sc = _sincos_f32(theta)
        var scn = _sincos_f32(-theta)
        assert_equal(
            bitcast[DType.int32](scn[0]),
            bitcast[DType.int32](Float32(-sc[0])),
        )
        assert_equal(bitcast[DType.int32](scn[1]), bitcast[DType.int32](sc[1]))


def test_sincos_matches_libm_over_the_flux_domains() raises:
    var worst = 0
    for p in range(0, 4700, 7):
        for kf in range(1, 17):
            var freq = Float32(
                exp(Float64(NEG_LN_10000) * Float64(2 * kf) / 32.0)
            )
            var theta = Float32(p) * freq
            var sc = _sincos_f32(theta)
            var ds = _ulp_f32(sc[0], Float32(sin(Float64(theta))))
            var dc = _ulp_f32(sc[1], Float32(cos(Float64(theta))))
            if ds > worst:
                worst = ds
            if dc > worst:
                worst = dc
    assert_true(worst <= 1, String("sincos ulp vs libm: ", worst))


def test_exp_matches_libm_over_the_band_domain() raises:
    var worst = 0
    for i in range(0, 923):
        var arg = Float32(-Float64(i) * 0.01)
        var d = _ulp_f32(_exp_f32(arg), Float32(exp(Float64(arg))))
        if d > worst:
            worst = d
    assert_true(worst <= 1, String("exp ulp vs libm: ", worst))


def test_sinusoid_embed_flux_geometry() raises:
    comptime H = 128
    var rows = 5
    var out = alloc[Scalar[F16]](rows * 2 * H).as_unsafe_any_origin()
    var flipped = alloc[Scalar[F16]](rows * 2 * H).as_unsafe_any_origin()
    var ts = alloc[Scalar[F32]](rows).as_unsafe_any_origin()
    ts[0] = 0.0
    ts[1] = 1.0
    ts[2] = 0.25
    ts[3] = 0.6180339887
    ts[4] = 0.999
    sinusoid_embed_2d(out, ts, rows, H, 1000.0, NEG_LN_10000, 1)
    sinusoid_embed_2d(flipped, ts, rows, H, 1000.0, NEG_LN_10000, 0)

    # t == 0: every angle is 0, so the cos half is exactly one and the sin
    # half exactly zero, independent of any trig accuracy.
    for i in range(H):
        assert_equal(out[i], Scalar[F16](1.0))
        assert_equal(out[H + i], Scalar[F16](0.0))

    # The flip swaps the halves bit for bit and changes nothing else.
    for b in range(rows):
        for i in range(H):
            assert_equal(
                bitcast[DType.int16](out[b * 2 * H + i]),
                bitcast[DType.int16](flipped[b * 2 * H + H + i]),
            )
            assert_equal(
                bitcast[DType.int16](out[b * 2 * H + H + i]),
                bitcast[DType.int16](flipped[b * 2 * H + i]),
            )

    # The independent reference: the same f32 quantization steps with libm
    # f64 trig, rounded once to fp16 through the same `.cast` path (the
    # `Scalar[F16](f32)` constructor rounds ties differently).
    var worst = 0
    for b in range(rows):
        var scaled = ts[b] * Float32(1000.0)
        var arg = scaled.cast[F16]().cast[F32]()
        for i in range(H):
            var e = (NEG_LN_10000 * Float32(i)) / Float32(H)
            var band = Float32(exp(Float64(e)))
            var theta = arg * band
            var want_cos = Float32(cos(Float64(theta))).cast[F16]()
            var want_sin = Float32(sin(Float64(theta))).cast[F16]()
            var dc = _ulp_f16(out[b * 2 * H + i], want_cos)
            var dsn = _ulp_f16(out[b * 2 * H + H + i], want_sin)
            if dc > worst:
                worst = dc
            if dsn > worst:
                worst = dsn
    assert_true(worst <= 1, String("sinusoid fp16 ulp vs libm: ", worst))

    out.free()
    flipped.free()
    ts.free()


def test_rope_table_flux_geometry() raises:
    comptime HD = 16
    comptime AXES = 4
    var rows = 4608
    var ids = alloc[Scalar[I64]](rows * AXES).as_unsafe_any_origin()
    for s in range(rows):
        for c in range(AXES):
            # Non-uniform, so a permuted or misaligned read cannot cancel.
            ids[s * AXES + c] = Int64((s * 37 + c * 911) % rows)
    ids[2 * AXES + 2] = 0  # a zero position somewhere in column 2
    var freqs = alloc[Scalar[F32]](HD).as_unsafe_any_origin()
    for k in range(HD):
        freqs[k] = Float32(exp(Float64(NEG_LN_10000) * Float64(2 * k) / 32.0))
    var out = alloc[Scalar[F32]](rows * 2 * HD).as_unsafe_any_origin()
    rope_table_2d(out, ids + 2, freqs, rows, HD, AXES)

    # Wiring: the strided column read must be bit-identical to the same
    # column copied densely and read at stride one.
    var dense = alloc[Scalar[I64]](rows).as_unsafe_any_origin()
    for s in range(rows):
        dense[s] = ids[s * AXES + 2]
    var out_dense = alloc[Scalar[F32]](rows * 2 * HD).as_unsafe_any_origin()
    rope_table_2d(out_dense, dense, freqs, rows, HD, 1)
    for i in range(rows * 2 * HD):
        assert_equal(
            bitcast[DType.int32](out[i]), bitcast[DType.int32](out_dense[i])
        )

    # Anchor: a zero position's row is (1, 0) pairs exactly.
    for k in range(HD):
        assert_equal(out[2 * 2 * HD + 2 * k], Float32(1.0))
        assert_equal(out[2 * 2 * HD + 2 * k + 1], Float32(0.0))

    # Sign symmetry: negating every position leaves every cos bit and negates
    # every nonzero sin bit.
    var neg = alloc[Scalar[I64]](rows).as_unsafe_any_origin()
    for s in range(rows):
        neg[s] = -dense[s]
    var out_neg = alloc[Scalar[F32]](rows * 2 * HD).as_unsafe_any_origin()
    rope_table_2d(out_neg, neg, freqs, rows, HD, 1)
    for s in range(rows):
        for k in range(HD):
            var idx = s * 2 * HD + 2 * k
            assert_equal(
                bitcast[DType.int32](out_neg[idx]),
                bitcast[DType.int32](out[idx]),
            )
            if out[idx + 1] != 0.0:
                assert_equal(
                    bitcast[DType.int32](out_neg[idx + 1]),
                    bitcast[DType.int32](Float32(-out[idx + 1])),
                )

    # The independent reference, on the same quantized f32 angles.
    var worst = 0
    for s in range(rows):
        var pos = dense[s].cast[F32]()
        for k in range(HD):
            var theta = Float64(pos * freqs[k])
            var dc = _ulp_f32(out[s * 2 * HD + 2 * k], Float32(cos(theta)))
            var dsn = _ulp_f32(out[s * 2 * HD + 2 * k + 1], Float32(sin(theta)))
            if dc > worst:
                worst = dc
            if dsn > worst:
                worst = dsn
    assert_true(worst <= 1, String("rope table f32 ulp vs libm: ", worst))

    ids.free()
    freqs.free()
    out.free()
    dense.free()
    out_dense.free()
    neg.free()
    out_neg.free()


def test_striped_union_reproduces_the_whole_table() raises:
    """The four compute threads' runs must together rebuild both tables exactly.

    Both arms run the same row body, so any difference is a partition bug and
    the comparison is bit-exact rather than a tolerance. The sentinel pre-fill
    makes an uncovered row fail instead of reading as a plausible zero, and the
    launch index runs to 4 -- the DMA thread, which must write nothing. Row
    counts straddle the thread count: below it, dividing it, and not.
    """
    comptime H = 8
    comptime HD = 6
    comptime AXES = 3
    comptime SENTINEL_F16 = Scalar[F16](-31000.0)
    comptime SENTINEL_F32 = Float32(-31000.0)
    var counts = [1, 3, 4, 7, 8, 13]

    for ci in range(len(counts)):
        var rows = counts[ci]

        var ts = alloc[Scalar[F32]](rows).as_unsafe_any_origin()
        for b in range(rows):
            ts[b] = Float32(b) * 0.37 + 0.11
        var whole = alloc[Scalar[F16]](rows * 2 * H).as_unsafe_any_origin()
        var striped = alloc[Scalar[F16]](rows * 2 * H).as_unsafe_any_origin()
        for i in range(rows * 2 * H):
            whole[i] = SENTINEL_F16
            striped[i] = SENTINEL_F16
        sinusoid_embed_2d(whole, ts, rows, H, 1000.0, NEG_LN_10000, 1)
        for t in range(5):
            sinusoid_embed_striped(
                striped, ts, rows, H, 1000.0, NEG_LN_10000, 1, UInt32(t)
            )
        for i in range(rows * 2 * H):
            assert_equal(
                bitcast[DType.int16](striped[i]),
                bitcast[DType.int16](whole[i]),
                String("sinusoid rows=", rows, " element ", i),
            )
            assert_true(
                bitcast[DType.int16](striped[i])
                != bitcast[DType.int16](SENTINEL_F16),
                String(
                    "sinusoid rows=", rows, " left element ", i, " unwritten"
                ),
            )

        var ids = alloc[Scalar[I64]](rows * AXES).as_unsafe_any_origin()
        for s in range(rows):
            for c in range(AXES):
                ids[s * AXES + c] = Int64((s * 37 + c * 911) % 4096)
        var freqs = alloc[Scalar[F32]](HD).as_unsafe_any_origin()
        for k in range(HD):
            freqs[k] = Float32(exp(Float64(NEG_LN_10000) * Float64(k) / 12.0))
        var whole_r = alloc[Scalar[F32]](rows * 2 * HD).as_unsafe_any_origin()
        var striped_r = alloc[Scalar[F32]](rows * 2 * HD).as_unsafe_any_origin()
        for i in range(rows * 2 * HD):
            whole_r[i] = SENTINEL_F32
            striped_r[i] = SENTINEL_F32
        rope_table_2d(whole_r, ids + 1, freqs, rows, HD, AXES)
        for t in range(5):
            rope_table_striped(
                striped_r, ids + 1, freqs, rows, HD, AXES, UInt32(t)
            )
        for i in range(rows * 2 * HD):
            assert_equal(
                bitcast[DType.int32](striped_r[i]),
                bitcast[DType.int32](whole_r[i]),
                String("rope table rows=", rows, " element ", i),
            )
            assert_true(
                bitcast[DType.int32](striped_r[i])
                != bitcast[DType.int32](SENTINEL_F32),
                String("rope rows=", rows, " left element ", i, " unwritten"),
            )

        ts.free()
        whole.free()
        striped.free()
        ids.free()
        freqs.free()
        whole_r.free()
        striped_r.free()


def test_thread_rows_partitions_without_gap_or_overlap() raises:
    """Every row belongs to exactly one compute thread, and the DMA thread
    owns none. Scored by covering count rather than by the tables, so a split
    bug is localized here rather than read off a numeric mismatch."""
    for rows in range(0, 40):
        var covered = alloc[Scalar[DType.int32]](
            rows + 1
        ).as_unsafe_any_origin()
        for i in range(rows):
            covered[i] = 0
        for t in range(5):
            var run = _thread_rows(rows, UInt32(t))
            assert_true(
                run[0] <= run[1],
                String("rows=", rows, " thread=", t, " inverted"),
            )
            if t >= 4:
                assert_equal(run[0], run[1], String("DMA thread owns rows"))
            for r in range(run[0], run[1]):
                assert_true(r < rows, String("rows=", rows, " past the end"))
                covered[r] += 1
        for r in range(rows):
            assert_equal(
                Int(covered[r]),
                1,
                String("rows=", rows, " row ", r, " coverage"),
            )
        covered.free()


def test_domain_guard_refuses_past_the_reduction_bound() raises:
    _require_sincos_domain(823549.0, "test")
    with assert_raises(contains="correctly-rounded reduction bound"):
        _require_sincos_domain(823550.0, "test")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
