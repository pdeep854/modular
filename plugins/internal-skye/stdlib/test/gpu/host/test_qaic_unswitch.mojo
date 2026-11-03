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
"""QAIC unswitch test: proves unswitch works correctly inside NPU kernels.

Tests three unswitch scenarios via compile_function + enqueue_function:
1. 1-predicate unswitch, True path: output[i] = input[i] + 10 (64 elements)
2. 1-predicate unswitch, False path: output[i] = input[i] - 10 (64 elements)
3. 2-predicate unswitch, all 4 Bool combinations (TT=1, TF=2, FT=3, FF=4)

unswitch is a comptime Bool dispatch (if/else on static parameter) with no
target-specific branches. These tests prove the parameterized function dispatch
compiles and executes correctly on Hexagon via QAIC JIT.

Note: The raises variant of unswitch (SwitchedFunction = fn[Bool]() capturing
-> None raises) cannot be used inside QAIC kernels because kernel functions
cannot raise. The non-raises overload tested here has identical dispatch logic,
proving unswitch works on NPU by proxy.

To run (requires QAIC hardware):
    mojo test_qaic_unswitch.mojo
"""

from std.algorithm.functional import unswitch
from std.memory import UnsafePointer, alloc
from std.testing import TestSuite, assert_true, assert_almost_equal

from max.gpu.host import DeviceContext
from std._plugin.qaic.host.target import npu_work_range

from qaic_test_utils import (
    alloc_sequential,
    alloc_constant,
    compile_and_run_unary,
    compile_and_run_reduction,
    verify_scalar_almost_equal,
)


# ===-----------------------------------------------------------------------===#
# Kernels: 1-predicate unswitch
# ===-----------------------------------------------------------------------===#


def unswitch_true_kernel_64(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    """Unswitch with dynamic_switch=True: should take +10 path."""

    def switched[flag: Bool]() {imm}:
        for i in range(64):
            if flag:
                output[i] = input[i] + 10.0
            else:
                output[i] = input[i] - 10.0

    unswitch(True, switched)


def unswitch_false_kernel_64(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    """Unswitch with dynamic_switch=False: should take -10 path."""

    def switched[flag: Bool]() {imm}:
        for i in range(64):
            if flag:
                output[i] = input[i] + 10.0
            else:
                output[i] = input[i] - 10.0

    unswitch(False, switched)


# ===-----------------------------------------------------------------------===#
# Kernels: 2-predicate unswitch
# ===-----------------------------------------------------------------------===#


def unswitch2_tt_kernel(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    """2-predicate unswitch: True, True -> output[0] = 1.0."""

    def switched[a: Bool, b: Bool]() {imm}:
        if a and b:
            output[0] = 1.0
        elif a and not b:
            output[0] = 2.0
        elif not a and b:
            output[0] = 3.0
        else:
            output[0] = 4.0

    unswitch(True, True, switched)


def unswitch2_tf_kernel(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    """2-predicate unswitch: True, False -> output[0] = 2.0."""

    def switched[a: Bool, b: Bool]() {imm}:
        if a and b:
            output[0] = 1.0
        elif a and not b:
            output[0] = 2.0
        elif not a and b:
            output[0] = 3.0
        else:
            output[0] = 4.0

    unswitch(True, False, switched)


def unswitch2_ft_kernel(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    """2-predicate unswitch: False, True -> output[0] = 3.0."""

    def switched[a: Bool, b: Bool]() {imm}:
        if a and b:
            output[0] = 1.0
        elif a and not b:
            output[0] = 2.0
        elif not a and b:
            output[0] = 3.0
        else:
            output[0] = 4.0

    unswitch(False, True, switched)


def unswitch2_ff_kernel(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    """2-predicate unswitch: False, False -> output[0] = 4.0."""

    def switched[a: Bool, b: Bool]() {imm}:
        if a and b:
            output[0] = 1.0
        elif a and not b:
            output[0] = 2.0
        elif not a and b:
            output[0] = 3.0
        else:
            output[0] = 4.0

    unswitch(False, False, switched)


# ===-----------------------------------------------------------------------===#
# Test Runners
# ===-----------------------------------------------------------------------===#


def run_unswitch_true_test() raises:
    """Test 1-predicate unswitch True path on QAIC (64 elements)."""
    var n = 64
    var ctx = DeviceContext(device_id=0, api="qaic")

    var input_host = alloc_sequential[DType.float32](n)
    var output_host = compile_and_run_unary[unswitch_true_kernel_64](
        ctx, input_host, n
    )

    # Verify: output[i] == (i + 1) + 10.0
    for i in range(n):
        var expected = Float32(i + 11)
        assert_almost_equal(output_host[i], expected, atol=0.001)

    input_host.unsafe_free()
    output_host.unsafe_free()


def run_unswitch_false_test() raises:
    """Test 1-predicate unswitch False path on QAIC (64 elements)."""
    var n = 64
    var ctx = DeviceContext(device_id=0, api="qaic")

    var input_host = alloc_sequential[DType.float32](n)
    var output_host = compile_and_run_unary[unswitch_false_kernel_64](
        ctx, input_host, n
    )

    # Verify: output[i] == (i + 1) - 10.0
    for i in range(n):
        var expected = Float32(i - 9)
        assert_almost_equal(output_host[i], expected, atol=0.001)

    input_host.unsafe_free()
    output_host.unsafe_free()


def _run_unswitch2_sub[
    kernel_fn: def(
        UnsafePointer[Float32, MutUnsafeAnyOrigin],
        UnsafePointer[Float32, MutUnsafeAnyOrigin],
    ) thin -> None
](expected: Float32, label: String) raises:
    """Run a single 2-predicate unswitch sub-test."""
    var ctx = DeviceContext(device_id=0, api="qaic")

    var input_host = alloc_constant[DType.float32](1, Float32(0.0))
    var output_host = compile_and_run_reduction[kernel_fn](ctx, input_host, 1)

    verify_scalar_almost_equal(output_host, expected)

    input_host.unsafe_free()
    output_host.unsafe_free()


def run_unswitch2_test() raises:
    """Test 2-predicate unswitch on QAIC, all 4 Bool combinations."""
    _run_unswitch2_sub[unswitch2_tt_kernel](1.0, "TT")
    _run_unswitch2_sub[unswitch2_tf_kernel](2.0, "TF")
    _run_unswitch2_sub[unswitch2_ft_kernel](3.0, "FT")
    _run_unswitch2_sub[unswitch2_ff_kernel](4.0, "FF")


# ===-----------------------------------------------------------------------===#
# Main
# ===-----------------------------------------------------------------------===#


def test_qaic_unswitch() raises:
    var all_passed = True

    # Test 1: 1-predicate unswitch, True path
    print("Test 1: 1-predicate unswitch, True path (64 elements, +10)...")
    try:
        run_unswitch_true_test()
        print("  PASSED")
    except e:
        print("  FAILED:", e)
        all_passed = False

    # Test 2: 1-predicate unswitch, False path
    print("Test 2: 1-predicate unswitch, False path (64 elements, -10)...")
    try:
        run_unswitch_false_test()
        print("  PASSED")
    except e:
        print("  FAILED:", e)
        all_passed = False

    # Test 3: 2-predicate unswitch, all 4 combinations
    print("Test 3: 2-predicate unswitch, all 4 combinations (TT/TF/FT/FF)...")
    try:
        run_unswitch2_test()
        print("  PASSED")
    except e:
        print("  FAILED:", e)
        all_passed = False

    # Summary
    print()
    if all_passed:
        print("UNSWITCH TEST PASSED: all tests correct")
    else:
        print("UNSWITCH TEST FAILED")
    assert_true(all_passed)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
