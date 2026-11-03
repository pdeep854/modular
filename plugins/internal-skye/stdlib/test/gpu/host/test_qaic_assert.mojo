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
"""Test: assert inside a QAIC kernel.

Verifies that the `assert` statement works from device code on
Qualcomm AI Cloud (QAIC) accelerators.  On QAIC, `assert` desugars to
`debug_assert`, which on a failing condition emits a single
`qshim_log` record at `LOG_MASK_FATAL` (with file:line:col + message),
then calls `abort()`.  `abort()` does not trap — it `longjmp`s back to
the entry-point wrapper's `setjmp`, which returns `JIT_DEV_ERROR_FATAL`
to the QAIC runtime.  Because `enqueue_function` is synchronous on
QAIC, the host sees the kernel error immediately rather than waiting
for the device watchdog.

The pass kernel uses progress markers so the host can confirm it ran
end-to-end; the fail kernel just expects the host enqueue to raise.

To run (requires QAIC hardware):
    mojo test_qaic_assert.mojo
"""

from std.testing import TestSuite, assert_equal
from std.memory import UnsafePointer, alloc

from max.gpu.host import DeviceContext, Dim


# ===-----------------------------------------------------------------------===#
# Kernel
# ===-----------------------------------------------------------------------===#

comptime _SENTINEL: UInt32 = 0xDEADBEEF


def assert_pass_kernel(
    out_val: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    """Kernel where all asserts pass — should run to completion."""

    out_val[0] = 1
    assert 1 == 1, "basic equality"

    out_val[0] = 2
    assert 42 > 0, "positive check"

    out_val[0] = 3
    assert True, "literal true"

    # All asserts passed.
    out_val[0] = 0


def assert_fail_kernel(
    out_val: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    """Kernel where an assert deliberately fails."""

    out_val[0] = 1
    assert 1 == 1, "this should pass"

    out_val[0] = 2
    # This assert should fire and halt the kernel.
    assert 1 == 2, "deliberate failure"

    # Should never reach here.
    out_val[0] = 99


# ===-----------------------------------------------------------------------===#
# Host driver
# ===-----------------------------------------------------------------------===#


def test_qaic_assert_pass() raises:
    """All asserts pass — kernel should run to completion."""
    var ctx = DeviceContext(device_id=0, api="qaic")

    var result_dev = ctx.enqueue_create_buffer[DType.uint32](1)
    var result_host = alloc[UInt32](1)

    result_host[0] = _SENTINEL
    ctx.enqueue_copy(result_dev, result_host)
    ctx.synchronize()

    var kernel = ctx.compile_function[assert_pass_kernel]()
    ctx.enqueue_function(
        kernel, result_dev, grid_dim=Dim(1), block_dim=Dim(0x01)
    )
    ctx.synchronize()

    ctx.enqueue_copy(result_host, result_dev)
    ctx.synchronize()

    var code = result_host[0]
    result_host.unsafe_free()

    assert_equal(
        Int(code != _SENTINEL), 1, "kernel did not execute (sentinel unchanged)"
    )
    assert_equal(
        Int(code), 0, "kernel stopped at progress marker " + String(code)
    )
    print("assert_pass: all asserts completed successfully.")


def test_qaic_assert_fail() raises:
    """A failing assert longjmps out of the kernel; the wrapper translates
    that into JIT_DEV_ERROR_FATAL. On QAIC, enqueue_function is
    synchronous, so it raises directly when the kernel reports a fatal
    error."""
    var ctx = DeviceContext(device_id=0, api="qaic")

    var result_dev = ctx.enqueue_create_buffer[DType.uint32](1)
    var result_host = alloc[UInt32](1)

    result_host[0] = _SENTINEL
    ctx.enqueue_copy(result_dev, result_host)
    ctx.synchronize()

    var kernel = ctx.compile_function[assert_fail_kernel]()
    var raised = False
    try:
        # Async enqueue: returns after recording + submitting. The kernel
        # runs later on the device; its assert failure surfaces at the
        # next blocking call (synchronize() here).
        ctx.enqueue_function(
            kernel, result_dev, grid_dim=Dim(1), block_dim=Dim(0x01)
        )
        ctx.synchronize()
    except:
        raised = True

    result_host.unsafe_free()

    assert_equal(
        Int(raised),
        1,
        (
            "expected enqueue_function or synchronize to raise after failing"
            " assert"
        ),
    )
    print("assert_fail: kernel correctly raised on failing assert.")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
