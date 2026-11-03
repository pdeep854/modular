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
"""Stress test for async enqueue under deep queueing.

Submits ``N_LAUNCHES`` async kernels back-to-back without an intervening
``synchronize()``. The C++ side does not cap the in-flight depth; it only
runs a quick non-blocking check at the start of each enqueue that releases
any submissions whose kernels have already completed. This test verifies
that release alone keeps the pending list under the QAIC stream's own
ceiling for this workload — if it didn't, the stream would reject a
submission with status 11 (UNSUPPORTED) and the run would fail. It's a
pure correctness probe, indifferent to overlap timing.

The kernel is the chained vector_add pattern from
``test_qaic_async_enqueue.mojo``: each call reads the buffer the
previous wrote, with the result building up across iterations. After
``N_LAUNCHES`` calls and one trailing ``synchronize()``, the final
buffer must hold ``(N_LAUNCHES + 1) * a[i]``.

To run (requires QAIC hardware):

    mojo test_qaic_async_queue_depth.mojo
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
    """Single-thread vector add: ``c[i] = a[i] + b[i]`` for ``i`` in [0, n)."""
    var range_pair = npu_work_range(Int(n))
    var start = range_pair[0]
    var end = range_pair[1]
    for i in range(start, end):
        c[i] = a[i] + b[i]


# ===-----------------------------------------------------------------------===#
# Configuration
# ===-----------------------------------------------------------------------===#

# Enough back-to-back submissions that the pending list would blow past
# the QAIC stream's ceiling if the non-blocking release didn't keep up.
comptime N_LAUNCHES: Int = 80

# Smaller buffer than the overlap-timing test — we don't need long-running
# kernels here, just enough work that pending submissions actually
# accumulate.
comptime BUF_LEN: Int = 64 * 1024


# ===-----------------------------------------------------------------------===#
# Test
# ===-----------------------------------------------------------------------===#


def main() raises:
    print_test_header("QAIC Async Queue Depth Test")

    var ctx = DeviceContext(device_id=0, api="qaic")

    # Two ping-pong buffers and one fixed addend, just like
    # test_qaic_async_enqueue: out = in + a, alternating which of
    # buf_x/buf_y is the input vs output. Starting state buf_x = a_host;
    # after N_LAUNCHES rounds the result is in buf_x when N_LAUNCHES is
    # even.
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

    # Enqueue N_LAUNCHES back-to-back without any intervening sync. The
    # per-enqueue non-blocking release must drain completed submissions
    # fast enough to stay under the QAIC stream's ceiling; otherwise
    # qaicJitCommandBuffersSubmit rejects one with status 11.
    print(
        "  Enqueuing",
        N_LAUNCHES,
        "kernels without intervening synchronize()...",
    )
    for i in range(N_LAUNCHES):
        if i % 2 == 0:
            ctx.enqueue_function(
                func,
                buf_x,
                buf_a,
                buf_y,
                n_arg,
                grid_dim=Dim(1),
                block_dim=Dim(0x01),
            )
        else:
            ctx.enqueue_function(
                func,
                buf_y,
                buf_a,
                buf_x,
                n_arg,
                grid_dim=Dim(1),
                block_dim=Dim(0x01),
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
        print("QUEUE DEPTH TEST PASSED:", N_LAUNCHES, "launches all completed")
        print_separator()
    else:
        print("QUEUE DEPTH TEST FAILED:", errors, "errors out of", BUF_LEN)
        exit(1)
