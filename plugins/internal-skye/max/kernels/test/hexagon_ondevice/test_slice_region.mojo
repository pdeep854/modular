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
"""Tests the strided walk that materializes a view on this device.

The walk is a host copy -- device memory on this transport is host memory -- so
it is exactly the part of the materializer that is checkable without the part.

Cases are written as a slice, since that is the view that dominates a denoise
step, but they are handed to the walk the way the compiler hands it one: a view
carries its start in its pointer and its step in its strides, so `_run` folds
both in rather than passing them. That is what the walk really receives, and it
is why the stride-0 and wide-stride cases below are not edge cases -- a
`broadcast_to` and a `transpose` reach it as exactly those.

Every buffer is filled with its own flat index in `int32`, so each expected
element is an exact integer and the comparison needs no tolerance. That also
discriminates the walk from a plain contiguous copy, which a tolerance-free
comparison of constant data would not.

Each case keeps its storage alive past the call it lends a pointer to, the way
the registrations keep their witness buffers alive.
"""

from extensibility import InputTensor, OutputTensor
from extensibility.managed_tensor_slice import _get_unknown_tensor_spec
from std.testing import assert_equal, assert_raises, TestSuite
from std.utils import IndexList

from hexagon_ondevice.registrations import _copy_view_to_dense

comptime DTYPE = DType.int32
comptime CAP = 256

comptime Storage = Array[Scalar[DTYPE], CAP]


def _ramp(mut storage: Storage):
    """Fills with the flat index, so an element names where it came from."""
    for i in range(CAP):
        storage[i] = Scalar[DTYPE](i)


def _dense_strides[rank: Int](shape: IndexList[rank]) -> IndexList[rank]:
    var strides = IndexList[rank]()
    var acc = 1
    for i in range(rank - 1, -1, -1):
        strides[i] = acc
        acc *= shape[i]
    return strides


def _run[
    rank: Int
](
    in_strides: IndexList[rank],
    out_shape: IndexList[rank],
    starts: IndexList[rank],
    steps: IndexList[rank],
    expected: List[Int],
) raises:
    """Materializes `out_shape` out of the ramp, viewed at `in_strides`."""
    var src = Storage(uninitialized=True)
    _ramp(src)
    var dst = Storage(uninitialized=True)
    for i in range(CAP):
        dst[i] = Scalar[DTYPE](-1)

    # Fold the case's starts and steps into the view, which is where they live
    # by the time the walk sees one.
    var offset = 0
    var view_strides = IndexList[rank]()
    for a in range(rank):
        offset += starts[a] * in_strides[a]
        view_strides[a] = in_strides[a] * steps[a]

    comptime spec = _get_unknown_tensor_spec[DTYPE, rank]()
    var input = InputTensor[static_spec=spec](
        UnsafePointer(to=src[offset]).as_unsafe_any_origin(),
        out_shape,
        view_strides,
    )
    var output = OutputTensor[static_spec=spec](
        dst.unsafe_ptr(), out_shape, _dense_strides(out_shape)
    )

    _copy_view_to_dense(output, input, "a materializing view")

    for i in range(len(expected)):
        assert_equal(Int(dst[i]), expected[i])
    # The element past the region must be untouched, so a walk that overran its
    # extents cannot pass by writing the right prefix.
    assert_equal(Int(dst[len(expected)]), -1)

    _ = src^
    _ = dst^


def test_last_axis_chunk() raises:
    """The shape a fused projection is chunked into: 122 of FLUX's 141."""
    _run(
        _dense_strides(IndexList[3](2, 3, 4)),
        IndexList[3](2, 3, 2),
        IndexList[3](0, 0, 1),
        IndexList[3](1, 1, 1),
        [1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22],
    )


def test_interior_axis() raises:
    _run(
        _dense_strides(IndexList[3](2, 4, 3)),
        IndexList[3](2, 2, 3),
        IndexList[3](0, 1, 0),
        IndexList[3](1, 1, 1),
        [3, 4, 5, 6, 7, 8, 15, 16, 17, 18, 19, 20],
    )


def test_two_axes() raises:
    _run(
        _dense_strides(IndexList[3](3, 3, 3)),
        IndexList[3](1, 2, 3),
        IndexList[3](1, 1, 0),
        IndexList[3](1, 1, 1),
        [12, 13, 14, 15, 16, 17],
    )


def test_non_unit_step() raises:
    """Steps > 1 leave the run path: each element is loaded on its own."""
    _run(
        _dense_strides(IndexList[2](2, 6)),
        IndexList[2](2, 3),
        IndexList[2](0, 0),
        IndexList[2](1, 2),
        [0, 2, 4, 6, 8, 10],
    )


def test_strided_input() raises:
    """The input arrives as an alias of a wider buffer, not as its own.

    This is the case a slice creates for whatever comes next, and the case that
    a walk reading the input as dense would get silently wrong: the extents fit,
    the byte count is plausible, and every element is the wrong one.
    """
    _run(
        IndexList[2](8, 1),
        IndexList[2](2, 2),
        IndexList[2](1, 0),
        IndexList[2](1, 1),
        [8, 9, 16, 17],
    )


def test_broadcast_input() raises:
    """A stride of 0 spans an axis by repeating one element."""
    _run(
        IndexList[2](0, 1),
        IndexList[2](3, 2),
        IndexList[2](0, 2),
        IndexList[2](1, 1),
        [2, 3, 2, 3, 2, 3],
    )


def test_whole_tensor() raises:
    _run(
        IndexList[1](1),
        IndexList[1](5),
        IndexList[1](0),
        IndexList[1](1),
        [0, 1, 2, 3, 4],
    )


def test_strided_output_refused() raises:
    """The walk fills the output linearly, so a strided one would be wrong."""
    var src = Storage(uninitialized=True)
    _ramp(src)
    var dst = Storage(uninitialized=True)

    comptime spec = _get_unknown_tensor_spec[DTYPE, 2]()
    var input = InputTensor[static_spec=spec](
        src.unsafe_ptr(), IndexList[2](2, 2), IndexList[2](2, 1)
    )
    var output = OutputTensor[static_spec=spec](
        dst.unsafe_ptr(), IndexList[2](2, 2), IndexList[2](4, 1)
    )

    with assert_raises(contains="the output is not densely row-major"):
        _copy_view_to_dense(output, input, "a materializing view")

    _ = src^
    _ = dst^


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
