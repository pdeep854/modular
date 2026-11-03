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
"""QAIC sum/mean/product test: proves sum, mean, and product reductions work on NPU.

Tests three reduction operations distributed across all NPU workers:
1. Sum of 128 elements (aligned) = 8256.0
2. Mean of 100 elements (unaligned) = 50.5
3. Product of 8 elements (small) = 40320.0

Each worker reduces its chunk (via npu_work_range) into a partial result
written to output[npu_global_idx()].  The host combines all partials.

To run (requires QAIC hardware):
    mojo test_qaic_sum_mean.mojo
"""

from std.memory import UnsafePointer
from std.testing import TestSuite, assert_almost_equal

from max.gpu.host import DeviceContext
from std._plugin.qaic.host.target import npu_work_range, npu_global_idx

from qaic_test_utils import (
    alloc_sequential,
    compile_and_run_parallel_reduction,
    MAX_NPU_WORKERS,
)


# ===-----------------------------------------------------------------------===#
# Kernels
# ===-----------------------------------------------------------------------===#


def sum_kernel[
    size: Int
](
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    """Parallel sum: each worker sums its chunk of input[0..size]."""
    var (start, end) = npu_work_range(size)
    var partial = Float32(0.0)
    for i in range(start, end):
        partial += input[i]
    output[Int(npu_global_idx())] = partial


def product_kernel[
    size: Int
](
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    """Parallel product: each worker multiplies its chunk of input[0..size]."""
    var (start, end) = npu_work_range(size)
    var partial = Float32(1.0)
    for i in range(start, end):
        partial *= input[i]
    output[Int(npu_global_idx())] = partial


# ===-----------------------------------------------------------------------===#
# Test Runners
# ===-----------------------------------------------------------------------===#


def run_sum_test() raises:
    """Tests sum on QAIC (128 elements, aligned)."""
    var n = 128
    var ctx = DeviceContext(device_id=0, api="qaic")
    var input_host = alloc_sequential[DType.float32](n, start=1)
    var partials = compile_and_run_parallel_reduction[sum_kernel[128]](
        ctx, input_host, n, identity=Float32(0.0)
    )

    var result = Float32(0.0)
    for i in range(MAX_NPU_WORKERS):
        result += partials[i]

    # sum(1..128) = 128 * 129 / 2 = 8256.0
    assert_almost_equal(result, Float32(8256.0))
    input_host.unsafe_free()
    partials.unsafe_free()


def run_mean_test() raises:
    """Tests mean on QAIC (100 elements, unaligned)."""
    var n = 100
    var ctx = DeviceContext(device_id=0, api="qaic")
    var input_host = alloc_sequential[DType.float32](n, start=1)
    var partials = compile_and_run_parallel_reduction[sum_kernel[100]](
        ctx, input_host, n, identity=Float32(0.0)
    )

    var result = Float32(0.0)
    for i in range(MAX_NPU_WORKERS):
        result += partials[i]
    result = result / 100.0

    # mean(1..100) = 5050 / 100 = 50.5
    assert_almost_equal(result, Float32(50.5))
    input_host.unsafe_free()
    partials.unsafe_free()


def run_product_test() raises:
    """Tests product on QAIC (8 elements, small to avoid overflow)."""
    var n = 8
    var ctx = DeviceContext(device_id=0, api="qaic")
    var input_host = alloc_sequential[DType.float32](n, start=1)
    var partials = compile_and_run_parallel_reduction[product_kernel[8]](
        ctx, input_host, n, identity=Float32(1.0)
    )

    var result = Float32(1.0)
    for i in range(MAX_NPU_WORKERS):
        result *= partials[i]

    # 8! = 40320.0
    assert_almost_equal(result, Float32(40320.0), atol=0.01)
    input_host.unsafe_free()
    partials.unsafe_free()


# ===-----------------------------------------------------------------------===#
# Main
# ===-----------------------------------------------------------------------===#


def test_qaic_sum_mean() raises:
    print("Test 1: Sum (128 elements, aligned, expected=8256.0)...")
    run_sum_test()
    print("  PASSED")

    print("Test 2: Mean (100 elements, unaligned, expected=50.5)...")
    run_mean_test()
    print("  PASSED")

    print("Test 3: Product (8 elements, expected=40320.0)...")
    run_product_test()
    print("  PASSED")

    print()
    print("SUM/MEAN/PRODUCT TEST PASSED: all tests correct")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
