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
"""Example: Using qshim_log for device-side logging on QAIC.

Demonstrates how to call `qshim_log` from a Hexagon kernel to emit log
messages through the QAIC device runtime.  The kernel writes progress
markers to an output buffer so the host can verify each logging call
completed without crashing.

Important Hexagon limitation:
    Only ONE arity of `qshimLog` may appear in a single compilation unit.
    This example uses the 0-format-arg overload exclusively. This is a known
    limitation and is being worked on.

To run (requires QAIC hardware):
    mojo test_qshim_log_example.mojo
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
    LOG_MASK_ERROR,
    LOG_MASK_WARN,
    LOG_MASK_INFO,
    LOG_MASK_DEBUG,
    qshim_log,
)


# ===-----------------------------------------------------------------------===#
# Kernel
# ===-----------------------------------------------------------------------===#

# Result buffer layout (UInt32 slots):
#   [0]  progress marker -- advances after each qshim_log call
#   Final value 0 means all calls completed successfully.

comptime _SENTINEL: UInt32 = 0xDEADBEEF


def log_example_kernel(
    entry_config: UnsafePointer[AicJitEntryPointConfig, MutUnsafeAnyOrigin],
    pointer_array: UnsafePointer[AicJitPointerArray, MutUnsafeAnyOrigin],
) -> UInt32:
    """Kernel that exercises qshim_log at every log level.

    Each qshim_log call emits a simple message (no format arguments) and
    advances a progress counter in the output buffer.  If the kernel
    crashes inside qshimLog, the host can read the counter to determine
    which call failed.

    Buffer layout:
        [0]: result buffer (single UInt32)
    """
    var out = npu_get_buffer[UInt32](pointer_array, 0)

    # Progress 1 -- FATAL (commented out: QAIC runtime treats FATAL as an
    # actual fatal error and aborts the command buffer, killing the kernel)
    # out[0] = 1
    # comptime fatal_msg = "qshim_log example: FATAL level"
    # qshim_log(LOG_MASK_FATAL, fatal_msg.unsafe_ptr())

    # Progress 2 -- ERROR
    out[0] = 2
    comptime error_msg = "qshim_log example: ERROR level"
    qshim_log(LOG_MASK_ERROR, error_msg.unsafe_ptr())

    # Progress 3 -- WARN
    out[0] = 3
    comptime warn_msg = "qshim_log example: WARN level"
    qshim_log(LOG_MASK_WARN, warn_msg.unsafe_ptr())

    # Progress 4 -- INFO
    out[0] = 4
    comptime info_msg = "qshim_log example: INFO level"
    qshim_log(LOG_MASK_INFO, info_msg.unsafe_ptr())

    # Progress 5 -- DEBUG
    out[0] = 5
    comptime debug_msg = "qshim_log example: DEBUG level"
    qshim_log(LOG_MASK_DEBUG, debug_msg.unsafe_ptr())

    # All calls completed
    out[0] = 0
    return 0


# ===-----------------------------------------------------------------------===#
# Host driver
# ===-----------------------------------------------------------------------===#


def run_log_example(
    ctx: DeviceContext,
    kernel_func: DeviceFunction,
) raises -> Bool:
    """Compile, execute, and verify the logging kernel."""
    # Allocate a 1-element UInt32 result buffer on the device.
    var result_dev = ctx.enqueue_create_buffer[DType.uint32](1)
    var result_host = alloc[UInt32](1)

    # Write sentinel so we can detect if the kernel never ran.
    result_host[0] = _SENTINEL
    ctx.enqueue_copy(result_dev, result_host)
    ctx.synchronize()

    # Execute with a single core / single thread.
    ctx.enqueue_function(
        kernel_func,
        result_dev,
        grid_dim=Dim(1),
        block_dim=Dim(0x01),
    )
    ctx.synchronize()

    # Read back results.
    ctx.enqueue_copy(result_host, result_dev)
    ctx.synchronize()

    var code = result_host[0]
    result_host.unsafe_free()

    if code == _SENTINEL:
        print("  ERROR: kernel did not execute (sentinel unchanged)")
        return False

    if code == 0:
        print("  All 4 qshim_log calls completed successfully.")
        return True

    print("  Kernel stopped at progress marker:", code)
    print("  (a non-zero value indicates a crash inside qshimLog)")
    return False


def test_qshim_log_example() raises:
    print_test_header("qshim_log Example")

    try:
        var ctx = DeviceContext(device_id=0, api="qaic")

        print("Compiling log_example_kernel...")
        var kernel_func = ctx.compile_function[log_example_kernel]()
        print("  Compiled.")

        print("Running kernel...")
        var passed = run_log_example(ctx, kernel_func)

        print()
        print_separator()
        if passed:
            print("PASSED")
        else:
            print("FAILED")
        print_separator()

        assert_true(passed)
    except e:
        print("Error:", e)
        assert_true(False, String(e))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
