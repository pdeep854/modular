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
"""Async enqueue with multi-core batches.

``test_qaic_async_queue_depth.mojo`` exercises deep single-core
queueing. Each multi-core launch instead submits a batch of ``grid_dim``
command buffers (one per NSP, single CU bit each) that complete
together; the pending-list entry holds the whole batch.
This test checks that batch-level recording, submission, completion
counting, and release all hold up across many in-flight multi-core
batches.

The kernel is the chained vector_add: workers partition the buffer via
``npu_work_range``, so the result is exact regardless of core/thread
count and any worker that misses its slice shows up as errors.

To run (requires QAIC hardware):

    mojo test_qaic_async_multicore.mojo
"""

from common import print_separator, print_test_header

from std.memory import UnsafePointer, alloc
from std.sys import exit

from max.gpu.host import DeviceContext, Dim
from std._plugin.qaic.host.target import npu_work_range


# ===-----------------------------------------------------------------------===#
# Kernel
# ===-----------------------------------------------------------------------===#


def add_kernel(
    a: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    b: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    c: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    n: UInt32,
):
    """Worker-partitioned vector add: ``c[i] = a[i] + b[i]``."""
    var range_pair = npu_work_range(Int(n))
    var start = range_pair[0]
    var end = range_pair[1]
    for i in range(start, end):
        c[i] = a[i] + b[i]


# ===-----------------------------------------------------------------------===#
# Configuration
# ===-----------------------------------------------------------------------===#

# Enough multi-CB batches queued back-to-back that the per-enqueue
# release has to keep the pending list under the QAIC stream's ceiling.
comptime N_LAUNCHES: Int = 80

comptime BUF_LEN: Int = 64 * 1024

# Multi-core hardware launch: 4 NSPs x 5 HVX threads = 20 workers.
comptime N_CORES: Int = 4
comptime THREAD_MASK: Int = 0x1F


# ===-----------------------------------------------------------------------===#
# Test
# ===-----------------------------------------------------------------------===#


def main() raises:
    print_test_header("QAIC Async Multi-Core Test")

    var ctx = DeviceContext(device_id=0, api="qaic")

    var buf_x = ctx.enqueue_create_buffer[DType.float32](BUF_LEN)
    var buf_y = ctx.enqueue_create_buffer[DType.float32](BUF_LEN)
    var buf_a = ctx.enqueue_create_buffer[DType.float32](BUF_LEN)

    var a_host = alloc[Float32](BUF_LEN)
    for i in range(BUF_LEN):
        a_host[i] = Float32(i % 7 + 1)

    ctx.enqueue_copy(buf_x, a_host)
    ctx.enqueue_copy(buf_a, a_host)
    ctx.synchronize()

    var func = ctx.compile_function[add_kernel]()
    var n_arg = UInt32(BUF_LEN)

    print(
        "  Enqueuing",
        N_LAUNCHES,
        "launches of",
        N_CORES,
        "command buffers each, no intervening synchronize()...",
    )
    for i in range(N_LAUNCHES):
        if i % 2 == 0:
            ctx.enqueue_function(
                func,
                buf_x,
                buf_a,
                buf_y,
                n_arg,
                grid_dim=Dim(N_CORES),
                block_dim=Dim(THREAD_MASK),
            )
        else:
            ctx.enqueue_function(
                func,
                buf_y,
                buf_a,
                buf_x,
                n_arg,
                grid_dim=Dim(N_CORES),
                block_dim=Dim(THREAD_MASK),
            )

    print("  All enqueues returned. Waiting for completion...")
    ctx.synchronize()

    # Final result is in buf_x because N_LAUNCHES is even, with each
    # element equal to (N_LAUNCHES + 1) * a_host[i].
    var c_host = alloc[Float32](BUF_LEN)
    ctx.enqueue_copy(c_host, buf_x)
    ctx.synchronize()

    var errors = 0
    var expected_factor = Float32(N_LAUNCHES + 1)
    for i in range(BUF_LEN):
        var expected = a_host[i] * expected_factor
        if abs(c_host[i] - expected) > 0.001 * abs(expected):
            errors += 1

    a_host.unsafe_free()
    c_host.unsafe_free()

    print()
    if errors == 0:
        print_separator()
        print(
            "MULTI-CORE ASYNC TEST PASSED:",
            N_LAUNCHES,
            "x",
            N_CORES,
            "command buffers all completed",
        )
        print_separator()
    else:
        print("MULTI-CORE ASYNC TEST FAILED:", errors, "errors out of", BUF_LEN)
        exit(1)
