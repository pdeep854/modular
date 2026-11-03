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
"""QAIC vectorize test: proves vectorize works on NPU with HVX SIMD width.

Tests two scenarios:
1. Aligned buffer (128 elements = 4x HVX float32 width of 32)
2. Unaligned buffer (100 elements, exercises remainder handling)

Both tests compile a kernel that doubles each element using stdlib
``vectorize`` with SIMD-width closures, then verify results via
``assert_almost_equal``.

To run (requires QAIC hardware):
    mojo test_qaic_vectorize.mojo
"""

from std.testing import TestSuite
from std.algorithm.functional import vectorize
from std.memory import UnsafePointer

from max.gpu.host import DeviceContext

from qaic_test_utils import (
    alloc_sequential,
    compile_and_run_unary,
    verify_buffer_almost_equal,
)


# ===-----------------------------------------------------------------------===#
# Kernels
# ===-----------------------------------------------------------------------===#


def vectorize_mul2_128(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    """Doubles each element using stdlib vectorize with HVX SIMD width (128 elements, aligned).
    """

    @always_inline
    def work[width: Int](i: Int) {var input, var output}:
        output.store[width=width](i, input.load[width=width](i) * 2.0)

    vectorize[32](128, work)


def vectorize_mul2_100(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    """Doubles each element using stdlib vectorize with HVX SIMD width (100 elements, unaligned).
    """

    @always_inline
    def work[width: Int](i: Int) {var input, var output}:
        output.store[width=width](i, input.load[width=width](i) * 2.0)

    vectorize[32](100, work)


# ===-----------------------------------------------------------------------===#
# Test Runners
# ===-----------------------------------------------------------------------===#


def run_test_aligned() raises:
    """Tests vectorize with aligned buffer (128 elements = 4x HVX float32 width).
    """
    var n = 128
    var ctx = DeviceContext(device_id=0, api="qaic")

    var input_host = alloc_sequential[DType.float32](n, start=1)
    var output_host = compile_and_run_unary[vectorize_mul2_128](
        ctx, input_host, n
    )

    # Build expected: (i + 1) * 2.0
    var expected = alloc_sequential[DType.float32](n, start=1)
    for i in range(n):
        expected[i] = Float32((i + 1) * 2)

    verify_buffer_almost_equal(output_host, expected, n)

    input_host.unsafe_free()
    output_host.unsafe_free()
    expected.unsafe_free()


def run_test_unaligned() raises:
    """Tests vectorize with unaligned buffer (100 elements, not multiple of 32).
    """
    var n = 100
    var ctx = DeviceContext(device_id=0, api="qaic")

    var input_host = alloc_sequential[DType.float32](n, start=1)
    var output_host = compile_and_run_unary[vectorize_mul2_100](
        ctx, input_host, n
    )

    # Build expected: (i + 1) * 2.0
    var expected = alloc_sequential[DType.float32](n, start=1)
    for i in range(n):
        expected[i] = Float32((i + 1) * 2)

    verify_buffer_almost_equal(output_host, expected, n)

    input_host.unsafe_free()
    output_host.unsafe_free()
    expected.unsafe_free()


# ===-----------------------------------------------------------------------===#
# Main
# ===-----------------------------------------------------------------------===#


def test_qaic_vectorize() raises:
    print(
        "Test 1: Aligned buffer (128 elements, multiple of HVX float32"
        " width)..."
    )
    run_test_aligned()
    print("  PASSED")

    print("Test 2: Unaligned buffer (100 elements, remainder handling)...")
    run_test_unaligned()
    print("  PASSED")

    print()
    print("VECTORIZE TEST PASSED: all tests correct")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
