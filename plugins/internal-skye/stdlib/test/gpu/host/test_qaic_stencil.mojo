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
"""QAIC stencil test: proves 1D and 2D stencil patterns work on NPU.

Tests two scenarios via compile_function + enqueue_function:
1. 1D 3-point averaging stencil (128 elements)
2. 2D 3x3 box filter stencil (8x16 = 128 elements)

Kernels call the library stencil_1d_npu and stencil_2d_npu functions
to exercise the public algorithm API rather than inlining the stencil logic.

To run (requires QAIC hardware):
    mojo test_qaic_stencil.mojo
"""

from std.memory import UnsafePointer, alloc
from std.testing import TestSuite, assert_true, assert_almost_equal

from max.gpu.host import DeviceContext

from std._plugin.qaic.algorithm_backend import (
    _stencil_1d_impl_npu as stencil_1d_npu,
    _stencil_2d_impl_npu as stencil_2d_npu,
)

from qaic_test_utils import (
    alloc_sequential,
    compile_and_run_unary,
)


# ===-----------------------------------------------------------------------===#
# Kernels
# ===-----------------------------------------------------------------------===#


def stencil_1d_avg_kernel_128(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    """1D 3-point averaging stencil (128 elements).

    Calls the library stencil_1d_npu function with a 3-point averaging
    compute function. Boundaries are copied and interior elements are
    the average of their left, center, and right neighbors.
    """

    def avg3(l: Float32, c: Float32, r: Float32) capturing -> Float32:
        return (l + c + r) / 3.0

    stencil_1d_npu[avg3](input, output, 128)


def stencil_2d_box_kernel_8x16(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    """2D 3x3 box filter stencil (8 rows x 16 cols = 128 elements).

    Calls the library stencil_2d_npu function with a 3x3 box filter
    compute function. Border rows/columns are copied and interior
    elements are the average of their 3x3 neighborhood.
    """
    comptime rows = 8
    comptime cols = 16

    def box3x3(
        data: UnsafePointer[Float32, MutUnsafeAnyOrigin],
        r: Int,
        c: Int,
        num_cols: Int,
    ) capturing -> Float32:
        var total: Float32 = 0.0
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                total += data[(r + dr) * num_cols + (c + dc)]
        return total / 9.0

    stencil_2d_npu[box3x3](input, output, rows, cols)


# ===-----------------------------------------------------------------------===#
# Test Runners
# ===-----------------------------------------------------------------------===#


def run_stencil_1d_test() raises:
    """Test 1D 3-point averaging stencil on QAIC (128 elements)."""
    var n = 128
    var ctx = DeviceContext(device_id=0, api="qaic")

    var input_host = alloc_sequential[DType.float32](n)
    var output_host = compile_and_run_unary[stencil_1d_avg_kernel_128](
        ctx, input_host, n
    )

    # Verify boundaries: copied directly.
    assert_almost_equal(output_host[0], input_host[0], atol=0.001)
    assert_almost_equal(output_host[127], input_host[127], atol=0.001)

    # Verify interior: average of (left, center, right).
    for i in range(1, n - 1):
        var expected = (
            input_host[i - 1] + input_host[i] + input_host[i + 1]
        ) / 3.0
        assert_almost_equal(output_host[i], expected, atol=0.001)

    input_host.unsafe_free()
    output_host.unsafe_free()


def run_stencil_2d_test() raises:
    """Test 2D 3x3 box filter stencil on QAIC (8x16 = 128 elements)."""
    comptime rows = 8
    comptime cols = 16
    var n = rows * cols
    var ctx = DeviceContext(device_id=0, api="qaic")

    var input_host = alloc_sequential[DType.float32](n)
    var output_host = compile_and_run_unary[stencil_2d_box_kernel_8x16](
        ctx, input_host, n
    )

    # Verify border rows.
    for c in range(cols):
        assert_almost_equal(output_host[c], input_host[c], atol=0.001)
        assert_almost_equal(
            output_host[(rows - 1) * cols + c],
            input_host[(rows - 1) * cols + c],
            atol=0.001,
        )

    # Verify border columns.
    for r in range(1, rows - 1):
        assert_almost_equal(
            output_host[r * cols], input_host[r * cols], atol=0.001
        )
        assert_almost_equal(
            output_host[r * cols + cols - 1],
            input_host[r * cols + cols - 1],
            atol=0.001,
        )

    # Verify interior: 3x3 box filter average.
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            var expected: Float32 = 0.0
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    expected += input_host[(r + dr) * cols + (c + dc)]
            expected = expected / 9.0
            assert_almost_equal(output_host[r * cols + c], expected, atol=0.001)

    input_host.unsafe_free()
    output_host.unsafe_free()


# ===-----------------------------------------------------------------------===#
# Main
# ===-----------------------------------------------------------------------===#


def test_qaic_stencil() raises:
    var all_passed = True

    # Test 1: 1D 3-point averaging stencil.
    print("Test 1: 1D 3-point averaging stencil (128 elements)...")
    try:
        run_stencil_1d_test()
        print("  PASSED")
    except e:
        print("  FAILED:", e)
        all_passed = False

    # Test 2: 2D 3x3 box filter stencil.
    print("Test 2: 2D 3x3 box filter stencil (8x16 = 128 elements)...")
    try:
        run_stencil_2d_test()
        print("  PASSED")
    except e:
        print("  FAILED:", e)
        all_passed = False

    # Summary.
    print()
    if all_passed:
        print("STENCIL TEST PASSED: all tests correct")
    else:
        print("STENCIL TEST FAILED")
    assert_true(all_passed)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
