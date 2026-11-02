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
"""Tests the per-batch parameter contract of the composite registrations.

The symbolic-batch matchers admit a per-channel parameter whose outermost
extent is the result's own -- one row per batch -- and the registration serves
it by walking the outermost axis, dispatching the signed kernel once per batch
slice. Two halves, tested where each is enforced:

- `_per_channel_rows` is the dispatch-time guard: it reads the concrete
  extents and returns how many rows the dispatch must step, 1 or the output's
  outermost. Shapes whose rows come from anywhere else must refuse loudly,
  because the per-batch offsets read past the allocation on them.

- The per-batch dispatch is offset arithmetic over the signed kernels: slice
  `b` of the activation at `b * rows_per_outer * row_length`, parameter row
  `b` at `b * row_length`. Those offsets are applied here to the actual kernel
  source the device library carries (`modulation_kernel.mojo`), on the host,
  at several batch extents, against an fp64 reference that indexes
  `[batch][token][channel]` explicitly. Values are small integers, so every
  exact answer is representable in fp16 and the comparison is equality.

Each exactness case carries its own failing control, run on every invocation:
the same per-batch inputs pushed through one whole-tensor dispatch -- which is
what the dispatch did before it learned to step rows, reading parameter row 0
for every batch -- must NOT match the reference (and must match it exactly at
batch 1, where the loop degenerates to that single dispatch). A test that
would pass under both dispatches proves nothing about the slicing.
"""

from extensibility import InputTensor
from extensibility.managed_tensor_slice import _get_unknown_tensor_spec
from std.memory import alloc
from std.testing import assert_equal, assert_raises, assert_true, TestSuite
from std.utils import IndexList

from hexagon_ondevice.modulation_kernel import (
    _param_tile,
    _rows_tile,
    ada_modulate_2d,
    gate_residual_2d,
)
from hexagon_ondevice.registrations import _per_channel_rows

comptime DTYPE = DType.float16
comptime F64 = DType.float64

comptime C = 64
"""Channels per row: a full HVX vector of fp16, so the kernels' vector path
runs, not just their tail."""

comptime T = 3
"""Token rows per batch. Deliberately not 1: a wrong `rows_per_outer` that
conflates tokens with batches is visible only when they differ."""

comptime SPEC1 = _get_unknown_tensor_spec[DTYPE, 1]()
comptime SPEC2 = _get_unknown_tensor_spec[DTYPE, 2]()
comptime SPEC3 = _get_unknown_tensor_spec[DTYPE, 3]()
comptime Param1 = InputTensor[static_spec=SPEC1]
comptime Param2 = InputTensor[static_spec=SPEC2]
comptime Param3 = InputTensor[static_spec=SPEC3]

comptime OP = "mo.composite.test"
comptime NAME = "the parameter"


def test_shared_row_dispatches_once() raises:
    var storage = alloc[Scalar[DTYPE]](C).as_unsafe_any_origin()
    var flat = Param1(storage, IndexList[1](C))
    assert_equal(_per_channel_rows(flat, C, 4, OP, NAME), 1)
    var spread = Param3(storage, IndexList[3](1, 1, C))
    assert_equal(_per_channel_rows(spread, C, 4, OP, NAME), 1)
    storage.free()


def test_per_batch_rows_step_the_outermost() raises:
    comptime B = 4
    var storage = alloc[Scalar[DTYPE]](B * C).as_unsafe_any_origin()
    var batched = Param3(storage, IndexList[3](B, 1, C))
    assert_equal(_per_channel_rows(batched, C, B, OP, NAME), B)
    var two_d = Param2(storage, IndexList[2](B, C))
    assert_equal(_per_channel_rows(two_d, C, B, OP, NAME), B)
    storage.free()


def test_wrong_channel_extent_refused() raises:
    var storage = alloc[Scalar[DTYPE]](C).as_unsafe_any_origin()
    var param = Param3(storage, IndexList[3](1, 1, C))
    with assert_raises(contains="must carry one element per channel"):
        _ = _per_channel_rows(param, 2 * C, 4, OP, NAME)
    storage.free()


def test_rows_off_the_outermost_axis_refused() raises:
    """`[1, B, C]` has B rows, but the per-batch offsets step axis 0."""
    comptime B = 4
    var storage = alloc[Scalar[DTYPE]](B * C).as_unsafe_any_origin()
    var param = Param3(storage, IndexList[3](1, B, C))
    with assert_raises(contains="one row, or one per outermost index"):
        _ = _per_channel_rows(param, C, B, OP, NAME)
    storage.free()


def test_rows_disagreeing_with_the_outer_extent_refused() raises:
    var storage = alloc[Scalar[DTYPE]](3 * C).as_unsafe_any_origin()
    var param = Param3(storage, IndexList[3](3, 1, C))
    with assert_raises(contains="one row, or one per outermost index"):
        _ = _per_channel_rows(param, C, 4, OP, NAME)
    storage.free()


def _fill_activation(
    ptr: UnsafePointer[Scalar[DTYPE], MutUnsafeAnyOrigin], count: Int
):
    for i in range(count):
        ptr[i] = Scalar[DTYPE](i % 7 - 3)


def _fill_branch(
    ptr: UnsafePointer[Scalar[DTYPE], MutUnsafeAnyOrigin], count: Int
):
    for i in range(count):
        ptr[i] = Scalar[DTYPE](i % 5 - 2)


def _fill_param(
    ptr: UnsafePointer[Scalar[DTYPE], MutUnsafeAnyOrigin], count: Int
):
    """Writes rows that all differ from row 0: `(7j mod 11) - 5` shifts by
    `8b mod 11` per row of C = 64, which is nonzero for every `b` below 11."""
    for j in range(count):
        ptr[j] = Scalar[DTYPE]((j * 7) % 11 - 5)


def _mismatches(
    got: UnsafePointer[Scalar[DTYPE], MutUnsafeAnyOrigin],
    want: UnsafePointer[Scalar[DTYPE], MutUnsafeAnyOrigin],
    count: Int,
) -> Int:
    var diffs = 0
    for i in range(count):
        if got[i] != want[i]:
            diffs += 1
    return diffs


def _check_gate_residual_at(batch: Int) raises:
    var n = batch * T * C
    var x = alloc[Scalar[DTYPE]](n).as_unsafe_any_origin()
    var y = alloc[Scalar[DTYPE]](n).as_unsafe_any_origin()
    var gate = alloc[Scalar[DTYPE]](batch * C).as_unsafe_any_origin()
    var got = alloc[Scalar[DTYPE]](n).as_unsafe_any_origin()
    var naive = alloc[Scalar[DTYPE]](n).as_unsafe_any_origin()
    var want = alloc[Scalar[DTYPE]](n).as_unsafe_any_origin()

    _fill_activation(x, n)
    _fill_branch(y, n)
    _fill_param(gate, batch * C)

    for i in range(n):
        var b = i // (T * C)
        var c = i % C
        var acc = Scalar[F64](x[i]) + Scalar[F64](gate[b * C + c]) * Scalar[
            F64
        ](y[i])
        want[i] = Scalar[DTYPE](acc)

    # The registration's per-batch walk: slice b at b*T*C, gate row b at b*C.
    for b in range(batch):
        gate_residual_2d[DTYPE](
            _rows_tile[DTYPE](x + b * T * C, T, C),
            _rows_tile[DTYPE](y + b * T * C, T, C),
            _param_tile[DTYPE](gate + b * C, T, C, 0),
            _rows_tile[DTYPE](got + b * T * C, T, C),
        )
    assert_equal(_mismatches(got, want, n), 0)

    # The failing control: one whole-tensor dispatch reads gate row 0 for
    # every batch, which is what the dispatch did before it stepped rows.
    gate_residual_2d[DTYPE](
        _rows_tile[DTYPE](x, batch * T, C),
        _rows_tile[DTYPE](y, batch * T, C),
        _param_tile[DTYPE](gate, batch * T, C, 0),
        _rows_tile[DTYPE](naive, batch * T, C),
    )
    if batch == 1:
        assert_equal(_mismatches(naive, want, n), 0)
    else:
        assert_true(
            _mismatches(naive, want, n) > 0,
            (
                "the one-dispatch control matched a per-batch gate; this test"
                " can no longer fail"
            ),
        )

    x.free()
    y.free()
    gate.free()
    got.free()
    naive.free()
    want.free()


def test_gate_residual_per_batch_slices_match_fp64() raises:
    _check_gate_residual_at(1)
    _check_gate_residual_at(2)
    _check_gate_residual_at(3)
    _check_gate_residual_at(5)


def _fill_wide_activation(
    ptr: UnsafePointer[Scalar[DTYPE], MutUnsafeAnyOrigin], count: Int
):
    """Multiples of 8192 in [-57344, 57344]: exact in fp16, and wide enough
    that the gated sum crosses float16's finite range."""
    for i in range(count):
        ptr[i] = Scalar[DTYPE](8192 * (i % 15 - 7))


def _fill_wide_param(
    ptr: UnsafePointer[Scalar[DTYPE], MutUnsafeAnyOrigin], count: Int
):
    """Multiples of 4096 in [-20480, 20480], rows rotated as `_fill_param`'s
    are, so no batch row equals row 0 for batches below 11."""
    for j in range(count):
        ptr[j] = Scalar[DTYPE](4096 * ((j * 7) % 11 - 5))


def _check_gate_residual_clamp_at(batch: Int) raises:
    """The clamped kernel's per-batch walk, exact against fp64.

    Every sum is a multiple of 4096 with magnitude at most 98304: inside the
    bound it is exactly representable in fp16, beyond it the reference clamps
    to +-65504 exactly, so the comparison is equality with no dead zone. Two
    failing controls run on every invocation. The UNCLAMPED kernel must differ
    at exactly the saturated sites -- inputs that never saturate would pass
    under both kernels and prove nothing about the clamp -- and the
    one-dispatch control must fail at batch > 1, as in `_check_gate_residual_at`.
    """
    var n = batch * T * C
    var x = alloc[Scalar[DTYPE]](n).as_unsafe_any_origin()
    var y = alloc[Scalar[DTYPE]](n).as_unsafe_any_origin()
    var gate = alloc[Scalar[DTYPE]](batch * C).as_unsafe_any_origin()
    var got = alloc[Scalar[DTYPE]](n).as_unsafe_any_origin()
    var plain = alloc[Scalar[DTYPE]](n).as_unsafe_any_origin()
    var naive = alloc[Scalar[DTYPE]](n).as_unsafe_any_origin()
    var want = alloc[Scalar[DTYPE]](n).as_unsafe_any_origin()

    _fill_wide_activation(x, n)
    _fill_branch(y, n)
    _fill_wide_param(gate, batch * C)

    var saturated = 0
    for i in range(n):
        var b = i // (T * C)
        var c = i % C
        var acc = Scalar[F64](x[i]) + Scalar[F64](gate[b * C + c]) * Scalar[
            F64
        ](y[i])
        if acc > 65504.0:
            acc = Scalar[F64](65504.0)
            saturated += 1
        elif acc < -65504.0:
            acc = Scalar[F64](-65504.0)
            saturated += 1
        want[i] = Scalar[DTYPE](acc)
    assert_true(
        saturated > 0,
        (
            "no site saturates; this test cannot tell the clamped kernel from"
            " the plain one"
        ),
    )

    # The registration's per-batch walk: slice b at b*T*C, gate row b at b*C.
    for b in range(batch):
        gate_residual_2d[DTYPE, clamp_fp16=True](
            _rows_tile[DTYPE](x + b * T * C, T, C),
            _rows_tile[DTYPE](y + b * T * C, T, C),
            _param_tile[DTYPE](gate + b * C, T, C, 0),
            _rows_tile[DTYPE](got + b * T * C, T, C),
        )
    assert_equal(_mismatches(got, want, n), 0)

    # The clamp control: the unclamped kernel overflows fp16 to +-inf at every
    # saturated site and only there, so its mismatch count is exactly the
    # saturation count.
    for b in range(batch):
        gate_residual_2d[DTYPE](
            _rows_tile[DTYPE](x + b * T * C, T, C),
            _rows_tile[DTYPE](y + b * T * C, T, C),
            _param_tile[DTYPE](gate + b * C, T, C, 0),
            _rows_tile[DTYPE](plain + b * T * C, T, C),
        )
    assert_equal(_mismatches(plain, want, n), saturated)

    # The slicing control: one whole-tensor dispatch reads gate row 0 for
    # every batch, which is what the dispatch did before it stepped rows.
    gate_residual_2d[DTYPE, clamp_fp16=True](
        _rows_tile[DTYPE](x, batch * T, C),
        _rows_tile[DTYPE](y, batch * T, C),
        _param_tile[DTYPE](gate, batch * T, C, 0),
        _rows_tile[DTYPE](naive, batch * T, C),
    )
    if batch == 1:
        assert_equal(_mismatches(naive, want, n), 0)
    else:
        assert_true(
            _mismatches(naive, want, n) > 0,
            (
                "the one-dispatch control matched a per-batch gate; this test"
                " can no longer fail"
            ),
        )

    x.free()
    y.free()
    gate.free()
    got.free()
    plain.free()
    naive.free()
    want.free()


def test_gate_residual_clamp_per_batch_slices_match_fp64() raises:
    _check_gate_residual_clamp_at(1)
    _check_gate_residual_clamp_at(2)
    _check_gate_residual_clamp_at(3)
    _check_gate_residual_clamp_at(5)


def _check_ada_modulate_at(batch: Int) raises:
    """Scale per batch, shift shared: the two counts step independently, the
    way the registration's `scale_step`/`shift_step` do."""
    var n = batch * T * C
    var x = alloc[Scalar[DTYPE]](n).as_unsafe_any_origin()
    var scale = alloc[Scalar[DTYPE]](batch * C).as_unsafe_any_origin()
    var shift = alloc[Scalar[DTYPE]](C).as_unsafe_any_origin()
    var got = alloc[Scalar[DTYPE]](n).as_unsafe_any_origin()
    var naive = alloc[Scalar[DTYPE]](n).as_unsafe_any_origin()
    var want = alloc[Scalar[DTYPE]](n).as_unsafe_any_origin()

    _fill_activation(x, n)
    _fill_param(scale, batch * C)
    _fill_branch(shift, C)

    for i in range(n):
        var b = i // (T * C)
        var c = i % C
        var acc = (Scalar[F64](1) + Scalar[F64](scale[b * C + c])) * Scalar[
            F64
        ](x[i]) + Scalar[F64](shift[c])
        want[i] = Scalar[DTYPE](acc)

    for b in range(batch):
        ada_modulate_2d[DTYPE](
            _rows_tile[DTYPE](x + b * T * C, T, C),
            _param_tile[DTYPE](scale + b * C, T, C, 0),
            _param_tile[DTYPE](shift, T, C, 0),
            _rows_tile[DTYPE](got + b * T * C, T, C),
        )
    assert_equal(_mismatches(got, want, n), 0)

    ada_modulate_2d[DTYPE](
        _rows_tile[DTYPE](x, batch * T, C),
        _param_tile[DTYPE](scale, batch * T, C, 0),
        _param_tile[DTYPE](shift, batch * T, C, 0),
        _rows_tile[DTYPE](naive, batch * T, C),
    )
    if batch == 1:
        assert_equal(_mismatches(naive, want, n), 0)
    else:
        assert_true(
            _mismatches(naive, want, n) > 0,
            (
                "the one-dispatch control matched a per-batch scale; this test"
                " can no longer fail"
            ),
        )

    x.free()
    scale.free()
    shift.free()
    got.free()
    naive.free()
    want.free()


def test_ada_modulate_per_batch_slices_match_fp64() raises:
    _check_ada_modulate_at(1)
    _check_ada_modulate_at(2)
    _check_ada_modulate_at(4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
