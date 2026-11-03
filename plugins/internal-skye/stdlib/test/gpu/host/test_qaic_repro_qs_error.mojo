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
"""Reproducer for the async-enqueue CI failure that reverted #276.

With asynchronous `enqueue_function`, a failing-assert kernel is submitted
and the call returns before the kernel runs. The kernel later faults
(longjmp -> JIT_DEV_ERROR_FATAL), leaving the QAIC device/stream in an
error state. Because work kept being submitted without waiting, subsequent
operations hit that error state and surface a QS_ERROR (status 500) from
the QAIC driver, raised out of the new async device-context code added by
the reverted commit (submitRecorded / waitForSubmitNumber). The CI crash
quoted it as:

    failed calling 'make_npu_wrapper' on device qaic:0 with error
    'QAICDeviceContext::submitRecorded: failed to submit command buffers:
    QS_ERROR'

The synchronous backend never hit this: each enqueue waited for the kernel,
so a fatal kernel raised immediately and nothing else was submitted on the
errored device. This file makes the failure deterministic: warm up, submit
a faulting kernel, keep submitting, then synchronize. The test passes when
the drain raises (the async contract: failures surface at the next blocking
call) and fails if synchronize completes silently.

Deliberately poisons the device (unusable for O(seconds)) — manual only,
like the assert tests.
"""

from max.gpu.host import DeviceContext, Dim
from std.memory import UnsafePointer
from std.sys import exit


def ok_kernel(out_val: UnsafePointer[UInt32, MutUnsafeAnyOrigin]):
    out_val[0] = 7


def fail_kernel(out_val: UnsafePointer[UInt32, MutUnsafeAnyOrigin]):
    out_val[0] = 2
    # Deliberate failure: longjmps to JIT_DEV_ERROR_FATAL, poisoning the
    # device.
    assert 1 == 2, "deliberate failure"


def main() raises:
    var ctx = DeviceContext(device_id=0, api="qaic")
    var out = ctx.enqueue_create_buffer[DType.uint32](1)

    var ok = ctx.compile_function[ok_kernel]()
    var bad = ctx.compile_function[fail_kernel]()

    # Healthy submit: the JIT stream / device are confirmed good.
    ctx.enqueue_function(ok, out, grid_dim=Dim(1), block_dim=Dim(1))
    ctx.synchronize()
    print("warmup OK; device healthy")

    # Submit a faulting kernel, then keep submitting more work without
    # syncing in between (the async contract). The faulting kernel poisons
    # the device while later submissions are still in flight.
    ctx.enqueue_function(bad, out, grid_dim=Dim(1), block_dim=Dim(1))
    for _ in range(8):
        ctx.enqueue_function(ok, out, grid_dim=Dim(1), block_dim=Dim(1))

    # Draining the in-flight submissions observes the poisoned device and
    # raises QS_ERROR from the async device-context path — the same error
    # the CI crash quoted. That raise is the expected outcome.
    print("=== synchronize on the poisoned device ===")
    var raised = False
    try:
        ctx.synchronize()
    except e:
        raised = True
        print("synchronize raised as expected:", e)

    if not raised:
        print("FAILED: synchronize completed without error (not reproduced)")
        exit(1)
