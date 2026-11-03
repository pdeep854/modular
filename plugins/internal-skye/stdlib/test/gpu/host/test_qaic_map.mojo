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
"""QAIC map test: proves map works correctly inside NPU kernels.

Tests two scenarios via compile_function + enqueue_function:
1. Map with computed values: output[i] = input[i] * 2.0 (128 elements)
2. Map with conditional logic: if input > 50 then *3 else *(-1) (100 elements)

map is a target-agnostic index loop (fn(Int) over [0, size)) with no
target-specific branches. These tests prove the control flow compiles
and executes correctly on Hexagon via QAIC JIT.

To run (requires QAIC hardware):
    mojo test_qaic_map.mojo
"""

from std.algorithm.functional import map
from std.memory import UnsafePointer, alloc
from std.testing import TestSuite, assert_true, assert_almost_equal

from max.gpu.host import DeviceContext
from std._plugin.qaic.host.target import npu_work_range

from qaic_test_utils import (
    alloc_sequential,
    compile_and_run_unary,
)


# ===-----------------------------------------------------------------------===#
# Kernels
# ===-----------------------------------------------------------------------===#


def map_mul2_kernel_128(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    """Map kernel: output[i] = input[i] * 2.0 using map (128 elements)."""

    def work(i: Int) {var input, var output}:
        output[i] = input[i] * 2.0

    map(128, work)


def map_conditional_kernel_100(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    """Map kernel with branching: if input > 50 then *3 else *(-1) (100 elements).
    """

    def work(i: Int) {var input, var output}:
        if input[i] > 50.0:
            output[i] = input[i] * 3.0
        else:
            output[i] = input[i] * -1.0

    map(100, work)


# ===-----------------------------------------------------------------------===#
# Test Runners
# ===-----------------------------------------------------------------------===#


def run_map_mul2_test() raises:
    """Test map with computed values on QAIC (128 elements)."""
    var n = 128
    var ctx = DeviceContext(device_id=0, api="qaic")

    var input_host = alloc_sequential[DType.float32](n)
    var output_host = compile_and_run_unary[map_mul2_kernel_128](
        ctx, input_host, n
    )

    # Verify: output[i] == (i + 1) * 2.0
    for i in range(n):
        var expected = Float32((i + 1) * 2)
        assert_almost_equal(output_host[i], expected, atol=0.001)

    input_host.unsafe_free()
    output_host.unsafe_free()


def run_map_conditional_test() raises:
    """Test map with conditional logic on QAIC (100 elements)."""
    var n = 100
    var ctx = DeviceContext(device_id=0, api="qaic")

    var input_host = alloc_sequential[DType.float32](n)
    var output_host = compile_and_run_unary[map_conditional_kernel_100](
        ctx, input_host, n
    )

    # Verify: if input > 50 (values 51..100, indices 50..99) then *3, else *(-1)
    for i in range(n):
        var value = Float32(i + 1)
        var expected: Float32
        if value > 50.0:
            expected = value * 3.0
        else:
            expected = value * -1.0
        assert_almost_equal(output_host[i], expected, atol=0.001)

    input_host.unsafe_free()
    output_host.unsafe_free()


# ===-----------------------------------------------------------------------===#
# Main
# ===-----------------------------------------------------------------------===#


def test_qaic_map() raises:
    var all_passed = True

    # Test 1: Map with computed values (index * 2)
    print("Test 1: Map multiply-by-2 (128 elements)...")
    try:
        run_map_mul2_test()
        print("  PASSED")
    except e:
        print("  FAILED:", e)
        all_passed = False

    # Test 2: Map with conditional logic
    print("Test 2: Map with conditional logic (100 elements)...")
    try:
        run_map_conditional_test()
        print("  PASSED")
    except e:
        print("  FAILED:", e)
        all_passed = False

    # Summary
    print()
    if all_passed:
        print("MAP TEST PASSED: all tests correct")
    else:
        print("MAP TEST FAILED")
    assert_true(all_passed)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
