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
"""QAIC reduce test: proves reduce and reduce_boolean work on NPU.

Tests four reduction operations distributed across all NPU workers:
1. Reduce (add-based sum, 128 elements, aligned) = 8256.0
2. Reduce (mul-based product, 8 elements) = 40320.0
3. Reduce_boolean "any" (100 elements, unaligned) = 1.0
4. Reduce_boolean "all" (100 elements, unaligned) = 1.0

Each worker reduces its chunk (via npu_work_range) into a partial result
written to output[npu_global_idx()].  The host combines all partials.

To run (requires QAIC hardware):
    mojo test_qaic_reduce.mojo
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


def reduce_sum_kernel[
    size: Int
](
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    """Parallel reduce-sum: each worker sums its chunk of input[0..size]."""
    var (start, end) = npu_work_range(size)
    var partial = Float32(0.0)
    for i in range(start, end):
        partial += input[i]
    output[Int(npu_global_idx())] = partial


def reduce_product_kernel[
    size: Int
](
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    """Parallel reduce-product: each worker multiplies its chunk of input[0..size].
    """
    var (start, end) = npu_work_range(size)
    var partial = Float32(1.0)
    for i in range(start, end):
        partial *= input[i]
    output[Int(npu_global_idx())] = partial


def reduce_any_kernel[
    size: Int
](
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    """Parallel reduce-boolean any: each worker checks its chunk for input[i] > 50.
    """
    var (start, end) = npu_work_range(size)
    var found = Float32(0.0)
    for i in range(start, end):
        if input[i] > 50.0:
            found = 1.0
    output[Int(npu_global_idx())] = found


def reduce_all_kernel[
    size: Int
](
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    """Parallel reduce-boolean all: each worker checks its chunk for input[i] >= 0.
    """
    var (start, end) = npu_work_range(size)
    var all_ok = Float32(1.0)
    for i in range(start, end):
        if input[i] < 0.0:
            all_ok = 0.0
    output[Int(npu_global_idx())] = all_ok


# ===-----------------------------------------------------------------------===#
# Test Runners
# ===-----------------------------------------------------------------------===#


def run_reduce_sum_test() raises:
    """Tests reduce-sum on QAIC (128 elements, aligned)."""
    var n = 128
    var ctx = DeviceContext(device_id=0, api="qaic")
    var input_host = alloc_sequential[DType.float32](n, start=1)
    var partials = compile_and_run_parallel_reduction[reduce_sum_kernel[128]](
        ctx, input_host, n, identity=Float32(0.0)
    )

    var result = Float32(0.0)
    for i in range(MAX_NPU_WORKERS):
        result += partials[i]

    # sum(1..128) = 128 * 129 / 2 = 8256.0
    assert_almost_equal(result, Float32(8256.0))
    input_host.unsafe_free()
    partials.unsafe_free()


def run_reduce_product_test() raises:
    """Tests reduce-product on QAIC (8 elements, 1..8, product=8!=40320)."""
    var n = 8
    var ctx = DeviceContext(device_id=0, api="qaic")
    var input_host = alloc_sequential[DType.float32](n, start=1)
    var partials = compile_and_run_parallel_reduction[reduce_product_kernel[8]](
        ctx, input_host, n, identity=Float32(1.0)
    )

    var result = Float32(1.0)
    for i in range(MAX_NPU_WORKERS):
        result *= partials[i]

    # 8! = 40320.0
    assert_almost_equal(result, Float32(40320.0), atol=0.01)
    input_host.unsafe_free()
    partials.unsafe_free()


def run_reduce_any_test() raises:
    """Tests reduce-boolean any on QAIC (100 elements, unaligned)."""
    var n = 100
    var ctx = DeviceContext(device_id=0, api="qaic")
    var input_host = alloc_sequential[DType.float32](n, start=0)
    var partials = compile_and_run_parallel_reduction[reduce_any_kernel[100]](
        ctx, input_host, n, identity=Float32(0.0)
    )

    # any worker found > 50 => max of partials is 1.0
    var result = Float32(0.0)
    for i in range(MAX_NPU_WORKERS):
        if partials[i] > result:
            result = partials[i]

    assert_almost_equal(result, Float32(1.0))
    input_host.unsafe_free()
    partials.unsafe_free()


def run_reduce_all_test() raises:
    """Tests reduce-boolean all on QAIC (100 elements, unaligned)."""
    var n = 100
    var ctx = DeviceContext(device_id=0, api="qaic")
    var input_host = alloc_sequential[DType.float32](n, start=0)
    var partials = compile_and_run_parallel_reduction[reduce_all_kernel[100]](
        ctx, input_host, n, identity=Float32(1.0)
    )

    # all workers found >= 0 => min of partials is 1.0
    var result = Float32(1.0)
    for i in range(MAX_NPU_WORKERS):
        if partials[i] < result:
            result = partials[i]

    assert_almost_equal(result, Float32(1.0))
    input_host.unsafe_free()
    partials.unsafe_free()


# ===-----------------------------------------------------------------------===#
# Main
# ===-----------------------------------------------------------------------===#


def test_qaic_reduce() raises:
    print("Test 1: Reduce-sum (128 elements, aligned, expected=8256.0)...")
    run_reduce_sum_test()
    print("  PASSED")

    print("Test 2: Reduce-product (8 elements, expected=40320.0)...")
    run_reduce_product_test()
    print("  PASSED")

    print("Test 3: Reduce-boolean any (100 elements, expected=1.0)...")
    run_reduce_any_test()
    print("  PASSED")

    print("Test 4: Reduce-boolean all (100 elements, expected=1.0)...")
    run_reduce_all_test()
    print("  PASSED")

    print()
    print("REDUCE TEST PASSED: all tests correct")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
