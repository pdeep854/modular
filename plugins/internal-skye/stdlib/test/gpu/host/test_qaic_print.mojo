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
"""Test: print() from a QAIC kernel.

Verifies that the standard `print()` function works from device code on
Qualcomm AI Cloud (QAIC) accelerators.  Under the hood `print()` formats
values into a heap buffer and emits them via `qshimLog` at INFO level.

The kernel writes progress markers to an output buffer so the host can
confirm each print call completed without crashing.  The actual log
output is visible in the QAIC device log.

To run (requires QAIC hardware):
    mojo test_qaic_print.mojo
"""

from std.testing import TestSuite, assert_equal
from std.memory import UnsafePointer, alloc
from std.sys import stderr

from max.gpu.host import DeviceContext, Dim


# ===-----------------------------------------------------------------------===#
# Kernel
# ===-----------------------------------------------------------------------===#

# Result buffer layout (UInt32 slots):
#   [0]  progress marker -- advances after each print() call
#   Final value 0 means all calls completed successfully.

comptime _SENTINEL: UInt32 = 0xDEADBEEF


def print_kernel(
    out_val: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    """Kernel that exercises print() with various argument types."""

    # 1 -- Simple string
    out_val[0] = 1
    print("Hello from QAIC kernel!")

    # 2 -- Integer
    out_val[0] = 2
    print("The answer is", 42)

    # 3 -- Multiple values with separator
    out_val[0] = 3
    print(1, 2, 3, sep=", ")

    # 4 -- Pre-composed line (one qshim_log record per print() on QAIC)
    out_val[0] = 4
    print(String("no newline") + " continuation")

    # 5 -- file=stderr should log at ERROR level on QAIC
    out_val[0] = 5
    print("This is an error message", file=stderr)

    # All calls completed
    out_val[0] = 0


# ===-----------------------------------------------------------------------===#
# Host driver
# ===-----------------------------------------------------------------------===#


def test_qaic_print() raises:
    """Compile, execute, and verify the print kernel."""
    var ctx = DeviceContext(device_id=0, api="qaic")

    var result_dev = ctx.enqueue_create_buffer[DType.uint32](1)
    var result_host = alloc[UInt32](1)

    # Write sentinel so we can detect if the kernel never ran.
    result_host[0] = _SENTINEL
    ctx.enqueue_copy(result_dev, result_host)
    ctx.synchronize()

    var func = ctx.compile_function[print_kernel]()
    ctx.enqueue_function(func, result_dev, grid_dim=Dim(1), block_dim=Dim(0x01))
    ctx.synchronize()

    # Read back results.
    ctx.enqueue_copy(result_host, result_dev)
    ctx.synchronize()

    var code = result_host[0]
    result_host.unsafe_free()

    assert_equal(
        Int(code != _SENTINEL), 1, "kernel did not execute (sentinel unchanged)"
    )
    assert_equal(
        Int(code), 0, "kernel stopped at progress marker (crash inside print)"
    )
    print("All print() calls completed successfully.")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
