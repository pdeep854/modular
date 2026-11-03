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
"""Test: assert from one of several concurrent QAIC kernel threads.

Exercises the thread safety of the QAIC kernel-side unwind path. Four threads
run the same kernel; exactly one of them (id == _ASSERT_THREAD) hits a
failing assert and calls `abort()`. The implementation must:

  - Route that thread's longjmp back to *its own* setjmp frame (not some
    other thread's), via per-thread slots in qaic_unwind.c.
  - Let the other threads continue executing normally.
  - Surface a kernel error to the host so enqueue_function raises.

Without per-thread slots in qaic_unwind.c, the asserting thread's longjmp
would resume on a different thread's stack — UB that would corrupt both
threads, almost certainly producing a watchdog timeout instead of a fast
clean error.

To run (requires QAIC hardware):
    mojo test_qaic_assert_multithread.mojo
"""

from std.testing import TestSuite, assert_equal
from std.memory import UnsafePointer, alloc

from max.gpu.host import DeviceContext, Dim
from std._plugin.qaic.host.qshim import qshim_get_thread_id


# ===-----------------------------------------------------------------------===#
# Kernel
# ===-----------------------------------------------------------------------===#

comptime _NUM_THREADS: Int = 4
comptime _ASSERT_THREAD: Int = 1
comptime _THREAD_MASK: Int = 0x0F  # threads 0..3
comptime _SENTINEL: UInt32 = 0xDEADBEEF


def multithread_assert_kernel(
    out_val: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    """Each thread writes its progress marker; thread _ASSERT_THREAD asserts.

    Per-thread output slot transitions:
        SENTINEL → 1  (entered kernel)
        1 → 2         (completed; asserting thread never reaches this)
    """
    var tid = Int(qshim_get_thread_id())

    # Phase 1: announce arrival.
    out_val[tid] = 1

    # Phase 2: only the targeted thread asserts.
    if tid == _ASSERT_THREAD:
        assert 1 == 2, "deliberate failure"

    # Phase 3: signal normal completion.
    out_val[tid] = 2


# ===-----------------------------------------------------------------------===#
# Host driver
# ===-----------------------------------------------------------------------===#


def test_qaic_assert_multithread() raises:
    """One thread asserts; expect synchronous kernel-error from enqueue and
    verify per-thread state survived the unwind."""
    var ctx = DeviceContext(device_id=0, api="qaic")

    var result_dev = ctx.enqueue_create_buffer[DType.uint32](_NUM_THREADS)
    var result_host = alloc[UInt32](_NUM_THREADS)

    for i in range(_NUM_THREADS):
        result_host[i] = _SENTINEL
    ctx.enqueue_copy(result_dev, result_host)
    ctx.synchronize()

    var kernel = ctx.compile_function[multithread_assert_kernel]()
    var raised = False
    try:
        # Async enqueue: returns after submitting. The kernel's assert
        # failure surfaces at the next blocking call.
        ctx.enqueue_function(
            kernel, result_dev, grid_dim=Dim(1), block_dim=Dim(_THREAD_MASK)
        )
        ctx.synchronize()
    except:
        raised = True

    assert_equal(
        Int(raised),
        1,
        (
            "expected enqueue_function or synchronize to raise when one thread"
            " asserts"
        ),
    )

    # Read back per-thread markers to verify isolation: each non-asserting
    # thread should have completed (marker == 2), while the asserting thread
    # should have stopped at marker 1 (between the two writes).  If
    # cross-thread longjmp had corrupted other threads' stacks we'd see
    # asserter-side state in their slots, or sentinels left untouched.
    ctx.enqueue_copy(result_host, result_dev)
    ctx.synchronize()

    for i in range(_NUM_THREADS):
        var slot = result_host[i]
        if i == _ASSERT_THREAD:
            assert_equal(
                Int(slot),
                1,
                "asserting thread should have stopped at marker 1",
            )
        else:
            assert_equal(
                Int(slot),
                2,
                "non-asserting thread should have completed (marker 2)",
            )

    result_host.unsafe_free()

    print(
        "multithread_assert: per-thread unwind correctly surfaced as kernel"
        " error; other threads completed cleanly."
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
