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
"""QAIC min/max test: proves min and max reductions work on NPU.

Tests four scenarios distributed across all NPU workers:
1. Min of 128 elements (aligned, positive values) = 10.0
2. Min of 100 elements (unaligned, with negatives) = -49.0
3. Max of 128 elements (aligned, positive values) = 137.0
4. Max of 100 elements (unaligned, with negatives) = 50.0

Each worker finds the min/max of its chunk (via npu_work_range) and writes
the partial result to output[npu_global_idx()].  The host combines all
partials with the corresponding operation.

To run (requires QAIC hardware):
    mojo test_qaic_min_max.mojo
"""

from std.memory import UnsafePointer, alloc
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


def min_kernel[
    size: Int
](
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    """Parallel min: each worker finds the min of its chunk of input[0..size].
    """
    var (start, end) = npu_work_range(size)
    if start < end:
        var partial = input[start]
        for i in range(start + 1, end):
            if input[i] < partial:
                partial = input[i]
        output[Int(npu_global_idx())] = partial


def max_kernel[
    size: Int
](
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    """Parallel max: each worker finds the max of its chunk of input[0..size].
    """
    var (start, end) = npu_work_range(size)
    if start < end:
        var partial = input[start]
        for i in range(start + 1, end):
            if input[i] > partial:
                partial = input[i]
        output[Int(npu_global_idx())] = partial


# ===-----------------------------------------------------------------------===#
# Test Runners
# ===-----------------------------------------------------------------------===#


def run_min_128_test() raises:
    """Tests min on QAIC (128 elements, aligned, positive values)."""
    var n = 128
    var ctx = DeviceContext(device_id=0, api="qaic")

    # input[i] = i + 10 (10.0, 11.0, ..., 137.0)
    var input_host = alloc_sequential[DType.float32](n, start=10)

    var partials = compile_and_run_parallel_reduction[min_kernel[128]](
        ctx, input_host, n, identity=Float32(1e38)
    )

    var result = Float32(1e38)
    for i in range(MAX_NPU_WORKERS):
        if partials[i] < result:
            result = partials[i]

    # min(10.0, 11.0, ..., 137.0) = 10.0
    assert_almost_equal(result, Float32(10.0))
    input_host.unsafe_free()
    partials.unsafe_free()


def run_min_100_test() raises:
    """Tests min on QAIC (100 elements, unaligned, with negatives)."""
    var n = 100
    var ctx = DeviceContext(device_id=0, api="qaic")

    # input[i] = 50.0 - i (50.0, 49.0, ..., -49.0)
    var input_host = alloc[Float32](n)
    for i in range(n):
        input_host[i] = Float32(50.0 - Float64(i))

    var partials = compile_and_run_parallel_reduction[min_kernel[100]](
        ctx, input_host, n, identity=Float32(1e38)
    )

    var result = Float32(1e38)
    for i in range(MAX_NPU_WORKERS):
        if partials[i] < result:
            result = partials[i]

    # min(50.0, 49.0, ..., -49.0) = -49.0
    assert_almost_equal(result, Float32(-49.0))
    input_host.unsafe_free()
    partials.unsafe_free()


def run_max_128_test() raises:
    """Tests max on QAIC (128 elements, aligned, positive values)."""
    var n = 128
    var ctx = DeviceContext(device_id=0, api="qaic")

    # input[i] = i + 10 (10.0, 11.0, ..., 137.0)
    var input_host = alloc_sequential[DType.float32](n, start=10)

    var partials = compile_and_run_parallel_reduction[max_kernel[128]](
        ctx, input_host, n, identity=Float32(-1e38)
    )

    var result = Float32(-1e38)
    for i in range(MAX_NPU_WORKERS):
        if partials[i] > result:
            result = partials[i]

    # max(10.0, 11.0, ..., 137.0) = 137.0
    assert_almost_equal(result, Float32(137.0))
    input_host.unsafe_free()
    partials.unsafe_free()


def run_max_100_test() raises:
    """Tests max on QAIC (100 elements, unaligned, with negatives)."""
    var n = 100
    var ctx = DeviceContext(device_id=0, api="qaic")

    # input[i] = 50.0 - i (50.0, 49.0, ..., -49.0)
    var input_host = alloc[Float32](n)
    for i in range(n):
        input_host[i] = Float32(50.0 - Float64(i))

    var partials = compile_and_run_parallel_reduction[max_kernel[100]](
        ctx, input_host, n, identity=Float32(-1e38)
    )

    var result = Float32(-1e38)
    for i in range(MAX_NPU_WORKERS):
        if partials[i] > result:
            result = partials[i]

    # max(50.0, 49.0, ..., -49.0) = 50.0
    assert_almost_equal(result, Float32(50.0))
    input_host.unsafe_free()
    partials.unsafe_free()


# ===-----------------------------------------------------------------------===#
# Main
# ===-----------------------------------------------------------------------===#


def test_qaic_min_max() raises:
    print("Test 1: Min (128 elements, aligned, expected=10.0)...")
    run_min_128_test()
    print("  PASSED")

    print("Test 2: Min (100 elements, unaligned, expected=-49.0)...")
    run_min_100_test()
    print("  PASSED")

    print("Test 3: Max (128 elements, aligned, expected=137.0)...")
    run_max_128_test()
    print("  PASSED")

    print("Test 4: Max (100 elements, unaligned, expected=50.0)...")
    run_max_100_test()
    print("  PASSED")

    print()
    print("MIN/MAX TEST PASSED: all tests correct")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
