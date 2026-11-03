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
"""QAIC elementwise test: proves elementwise-style operations work on NPU.

Tests two operation types via compile_function + enqueue_function:
1. Unary operation (negate): output[i] = -input[i] (128 elements, aligned)
2. Binary operation (add): output[i] = a[i] + b[i] (100 elements, unaligned)

Kernels use stdlib ``map`` for element-wise iteration instead of raw
for-loops. ``elementwise`` is a host-side orchestrator requiring
DeviceContext and cannot be called inside compile_function kernels,
so ``map`` is the correct device-side abstraction.

To run (requires QAIC hardware):
    mojo test_qaic_elementwise.mojo
"""

from std.testing import TestSuite
from std.algorithm.functional import map
from std.memory import UnsafePointer

from max.gpu.host import DeviceContext

from qaic_test_utils import (
    alloc_sequential,
    compile_and_run_unary,
    compile_and_run_binary,
    verify_buffer_almost_equal,
)


# ===-----------------------------------------------------------------------===#
# Kernels
# ===-----------------------------------------------------------------------===#


def negate_kernel(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    """Unary negate using stdlib map: output[i] = -input[i] (128 elements)."""

    def work(i: Int) {var input, var output}:
        output[i] = -input[i]

    map(128, work)


def add_kernel(
    a: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    b: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    """Binary add using stdlib map: output[i] = a[i] + b[i] (100 elements)."""

    def work(i: Int) {var a, var b, var output}:
        output[i] = a[i] + b[i]

    map(100, work)


# ===-----------------------------------------------------------------------===#
# Test Runners
# ===-----------------------------------------------------------------------===#


def run_negate_test() raises:
    """Tests unary negate operation on QAIC (128 elements, aligned)."""
    var n = 128
    var ctx = DeviceContext(device_id=0, api="qaic")

    var input_host = alloc_sequential[DType.float32](n, start=1)
    var output_host = compile_and_run_unary[negate_kernel](ctx, input_host, n)

    # Build expected: -(i + 1)
    var expected = alloc_sequential[DType.float32](n, start=1)
    for i in range(n):
        expected[i] = Float32(-(i + 1))

    verify_buffer_almost_equal(output_host, expected, n)

    input_host.unsafe_free()
    output_host.unsafe_free()
    expected.unsafe_free()


def run_add_test() raises:
    """Tests binary add operation on QAIC (100 elements, unaligned)."""
    var n = 100
    var ctx = DeviceContext(device_id=0, api="qaic")

    # a[i] = i, b[i] = 3*i
    var a_host = alloc_sequential[DType.float32](n, start=0)
    var b_host = alloc_sequential[DType.float32](n, start=0)
    for i in range(n):
        b_host[i] = Float32(3 * i)

    var output_host = compile_and_run_binary[add_kernel](ctx, a_host, b_host, n)

    # Build expected: i + 3*i = 4*i
    var expected = alloc_sequential[DType.float32](n, start=0)
    for i in range(n):
        expected[i] = Float32(4 * i)

    verify_buffer_almost_equal(output_host, expected, n)

    a_host.unsafe_free()
    b_host.unsafe_free()
    output_host.unsafe_free()
    expected.unsafe_free()


# ===-----------------------------------------------------------------------===#
# Main
# ===-----------------------------------------------------------------------===#


def test_qaic_elementwise() raises:
    print("Test 1: Unary negate (128 elements, aligned)...")
    run_negate_test()
    print("  PASSED")

    print("Test 2: Binary add (100 elements, unaligned)...")
    run_add_test()
    print("  PASSED")

    print()
    print("ELEMENTWISE TEST PASSED: all operations correct")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
