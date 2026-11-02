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
"""Tests the pitched-rows acceptance behind `mo.composite.swiglu`.

`_require_pitched_rows` is what lets the swiglu stub take a column slice of a
wider fused projection — trailing stride 1, row stride the projection's width,
start already in the pointer — and hand the selector its pitch, instead of
refusing the view into the host materializer. The check is a pure function of
shape and strides, so every acceptance and every refusal is exercised here
without a device: dense and pitched layouts return the right pitch, and the
layouts one pitch cannot address (a transpose, a stride-0 broadcast row, an
overlapping pitch, a batch axis with its own pitch) refuse the way
`_require_dense` would have.

The refusal cases are the failing controls: each is a layout the old dense
check also refused, so a regression that starts accepting one is caught here
red, not on the device.

A red result here means the acceptance set moved; the entry to update is
`lowering/execute-kernel-stride-support-does-not-stop-materialization` in
the plugin knowledge base, which this suite is the falsifier for.
"""

from std.testing import assert_equal, assert_raises, TestSuite
from std.utils import IndexList

from hexagon_ondevice.registrations import _require_pitched_rows

comptime OP: StaticString = "test.swiglu"
comptime NAME: StaticString = "the input"


def test_dense_returns_row_extent() raises:
    var pitch = _require_pitched_rows(
        IndexList[2](3, 8), IndexList[2](8, 1), OP, NAME
    )
    assert_equal(pitch, 8)


def test_pitched_slice_returns_pitch() raises:
    var pitch = _require_pitched_rows(
        IndexList[2](3, 8), IndexList[2](20, 1), OP, NAME
    )
    assert_equal(pitch, 20)


def test_batch_folds_onto_pitched_rows() raises:
    var pitch = _require_pitched_rows(
        IndexList[3](2, 3, 8), IndexList[3](60, 20, 1), OP, NAME
    )
    assert_equal(pitch, 20)


def test_rank1_is_one_dense_row() raises:
    var pitch = _require_pitched_rows(
        IndexList[1](8), IndexList[1](1), OP, NAME
    )
    assert_equal(pitch, 8)


def test_single_row_stride_unobservable() raises:
    var pitch = _require_pitched_rows(
        IndexList[2](1, 8), IndexList[2](999, 1), OP, NAME
    )
    assert_equal(pitch, 8)


def test_unit_trailing_axis_any_stride() raises:
    var pitch = _require_pitched_rows(
        IndexList[2](3, 1), IndexList[2](5, 7), OP, NAME
    )
    assert_equal(pitch, 5)


def test_empty_reports_dense_pitch() raises:
    var pitch = _require_pitched_rows(
        IndexList[2](0, 8), IndexList[2](20, 1), OP, NAME
    )
    assert_equal(pitch, 8)


def test_transposed_trailing_refused() raises:
    with assert_raises():
        _ = _require_pitched_rows(
            IndexList[2](3, 8), IndexList[2](1, 3), OP, NAME
        )


def test_broadcast_rows_refused() raises:
    with assert_raises():
        _ = _require_pitched_rows(
            IndexList[2](3, 8), IndexList[2](0, 1), OP, NAME
        )


def test_overlapping_pitch_refused() raises:
    with assert_raises():
        _ = _require_pitched_rows(
            IndexList[2](3, 8), IndexList[2](4, 1), OP, NAME
        )


def test_pitched_batch_refused() raises:
    with assert_raises():
        _ = _require_pitched_rows(
            IndexList[3](2, 3, 8), IndexList[3](100, 20, 1), OP, NAME
        )


def test_rank1_strided_refused() raises:
    with assert_raises():
        _ = _require_pitched_rows(IndexList[1](8), IndexList[1](2), OP, NAME)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
