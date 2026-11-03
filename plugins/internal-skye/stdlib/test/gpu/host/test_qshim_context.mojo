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
"""Test that qshim execution context functions work correctly.

This test verifies that the qshim runtime functions (qshim_get_thread_id,
qshim_get_core_id, etc.) return the same values as the entry_config struct
passed to QAIC kernels.

This is a prerequisite test for implementing parameterless npu_global_idx()
and similar functions that will use qshim internally instead of requiring
entry_config to be passed explicitly.

To run (requires QAIC hardware):
    mojo test_qshim_context.mojo
"""

from std.testing import TestSuite, assert_true
from common import print_separator, print_test_header

from std.memory import UnsafePointer, alloc

from max.gpu.host import DeviceContext, DeviceFunction, Dim
from std._plugin.qaic.host.target import (
    AicJitEntryPointConfig,
    AicJitPointerArray,
    npu_get_buffer,
)
from std._plugin.qaic.host.qshim import (
    qshim_get_thread_id,
    qshim_get_core_id,
    qshim_get_num_threads,
    qshim_get_num_cores,
)


# ===-----------------------------------------------------------------------===#
# Kernel: Compare entry_config with qshim functions
# ===-----------------------------------------------------------------------===#

# Result buffer layout:
# [0-3]: Match flags (0 = match, 1 = mismatch)
#   0: core_id match
#   1: num_cores match
#   2: thread_id match
#   3: num_threads match
# [4-7]: entry_config values
#   4: cfg_core_id
#   5: cfg_num_cores
#   6: cfg_thread_id
#   7: cfg_num_threads
# [8-11]: qshim values
#   8: qshim_core_id
#   9: qshim_num_cores
#   10: qshim_thread_id
#   11: qshim_num_threads


def qshim_comparison_kernel(
    entry_config: UnsafePointer[AicJitEntryPointConfig, MutUnsafeAnyOrigin],
    pointer_array: UnsafePointer[AicJitPointerArray, MutUnsafeAnyOrigin],
) -> UInt32:
    """Kernel that compares entry_config values with qshim function results.

    This kernel is designed to run with a single worker (1 core, 1 thread)
    to avoid race conditions when writing to the result buffer.
    """
    var out = npu_get_buffer[UInt32](pointer_array, 0)

    # Get values from entry_config (the known-working source)
    var cfg_core_id = entry_config[].core_id
    var cfg_num_cores = entry_config[].num_cores
    var cfg_thread_id = entry_config[].thread_id
    var cfg_num_threads = entry_config[].num_threads

    # Get values from qshim functions (what we want to verify)
    var qshim_core = UInt32(qshim_get_core_id())
    var qshim_num_cores = UInt32(qshim_get_num_cores())
    var qshim_thread = UInt32(qshim_get_thread_id())
    var qshim_num_threads = UInt32(qshim_get_num_threads())

    # Store match flags (0 = match, 1 = mismatch)
    out[0] = UInt32(1 if cfg_core_id != qshim_core else 0)
    out[1] = UInt32(1 if cfg_num_cores != qshim_num_cores else 0)
    out[2] = UInt32(1 if cfg_thread_id != qshim_thread else 0)
    out[3] = UInt32(1 if cfg_num_threads != qshim_num_threads else 0)

    # Store entry_config values for debugging
    out[4] = cfg_core_id
    out[5] = cfg_num_cores
    out[6] = cfg_thread_id
    out[7] = cfg_num_threads

    # Store qshim values for debugging
    out[8] = qshim_core
    out[9] = qshim_num_cores
    out[10] = qshim_thread
    out[11] = qshim_num_threads

    return 0


# ===-----------------------------------------------------------------------===#
# Test Functions
# ===-----------------------------------------------------------------------===#


def run_comparison_test(
    ctx: DeviceContext,
    kernel_func: DeviceFunction,
    num_cores: Int,
    thread_mask: Int,
    test_name: String,
) raises -> Bool:
    """Run the comparison kernel and verify results."""
    print("\n--- " + test_name + " ---")
    print(
        "Config: num_cores="
        + String(num_cores)
        + ", thread_mask="
        + hex(thread_mask)
    )

    # Allocate buffers
    var result_dev = ctx.enqueue_create_buffer[DType.uint32](16)
    var result_host = alloc[UInt32](16)

    # Initialize host buffer to sentinel values
    for i in range(16):
        result_host[i] = 0xDEADBEEF

    # Copy sentinel to device (to detect if kernel doesn't write)
    ctx.enqueue_copy(result_dev, result_host)
    ctx.synchronize()

    # Run pre-compiled kernel
    ctx.enqueue_function(
        kernel_func,
        result_dev,
        grid_dim=Dim(num_cores),
        block_dim=Dim(thread_mask),
    )
    ctx.synchronize()

    # Copy results back
    ctx.enqueue_copy(result_host, result_dev)
    ctx.synchronize()

    # Check if kernel wrote anything
    if result_host[0] == 0xDEADBEEF:
        print("ERROR: Kernel did not write to result buffer!")
        result_host.unsafe_free()
        return False

    # Display results
    print("\nentry_config values:")
    print("  core_id:", result_host[4])
    print("  num_cores:", result_host[5])
    print("  thread_id:", result_host[6])
    print("  num_threads:", result_host[7])

    print("\nqshim function values:")
    print("  qshim_get_core_id():", result_host[8])
    print("  qshim_get_num_cores():", result_host[9])
    print("  qshim_get_thread_id():", result_host[10])
    print("  qshim_get_num_threads():", result_host[11])

    # Check matches
    var all_match = True
    print("\nComparison results:")

    if result_host[0] == 0:
        print("  core_id: MATCH")
    else:
        print("  core_id: MISMATCH!")
        all_match = False

    if result_host[1] == 0:
        print("  num_cores: MATCH")
    else:
        print("  num_cores: MISMATCH!")
        all_match = False

    if result_host[2] == 0:
        print("  thread_id: MATCH")
    else:
        print("  thread_id: MISMATCH!")
        all_match = False

    if result_host[3] == 0:
        print("  num_threads: MATCH")
    else:
        print("  num_threads: MISMATCH!")
        all_match = False

    print("\nTest result:", "PASS" if all_match else "FAIL")

    result_host.unsafe_free()
    return all_match


def test_qshim_context() raises:
    print_test_header("QSHIM Context Functions Verification Test")

    try:
        var ctx = DeviceContext(device_id=0, api="qaic")

        # Pre-compile the kernel once
        print("\nCompiling kernel...")
        var kernel_func = ctx.compile_function[qshim_comparison_kernel]()
        print("Kernel compiled successfully.")

        var all_tests_passed = True

        # Test 1: Single core, single thread (simplest case)
        if not run_comparison_test(
            ctx,
            kernel_func,
            num_cores=1,
            thread_mask=0x01,  # 1 thread
            test_name="Single core, single thread",
        ):
            all_tests_passed = False

        # Test 2: Single core, multiple threads
        if not run_comparison_test(
            ctx,
            kernel_func,
            num_cores=1,
            thread_mask=0x03,  # 2 threads (bits 0-1)
            test_name="Single core, 2 threads",
        ):
            all_tests_passed = False

        # Test 3: Multiple cores, single thread each
        if not run_comparison_test(
            ctx,
            kernel_func,
            num_cores=4,
            thread_mask=0x01,  # 1 thread per core
            test_name="4 cores, 1 thread each",
        ):
            all_tests_passed = False

        # Test 4: Multiple cores, multiple threads
        if not run_comparison_test(
            ctx,
            kernel_func,
            num_cores=4,
            thread_mask=0x03,  # 2 threads per core
            test_name="4 cores, 2 threads each",
        ):
            all_tests_passed = False

        # Summary
        print_separator()
        if all_tests_passed:
            print("ALL TESTS PASSED!")
            print("\nConclusion: qshim functions return the same values as")
            print(
                "entry_config. Safe to use for parameterless npu_global_idx()."
            )
        else:
            print("SOME TESTS FAILED!")
            print("\nInvestigate the mismatches before proceeding.")
        print_separator()

        assert_true(all_tests_passed)
    except e:
        print("Error running tests:", e)
        assert_true(False, String(e))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
