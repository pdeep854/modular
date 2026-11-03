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
"""Test asynchronous QAIC kernel execution via host/device overlap timing.

Verifies that ``ctx.enqueue_function()`` returns before the kernel completes,
so host-side work can overlap with in-flight device work. The non-blocking
signal is wall-clock based: the time of (enqueue chain + host work +
synchronize) must be measurably less than the sum of doing them serially.

The chain runs ``N_LAUNCHES`` dependent vector_add launches (each kernel
reads the buffer the previous kernel wrote), additionally exercising:

* FIFO ordering across submissions on the single QAIC stream.
* Lifetime of staged scalar-arg device buffers across many in-flight
  submissions (the highest-risk path under async — see
  ``device_context.mojo`` ``meta_storage`` logic).
* ``ctx.synchronize()`` waiting for all queued work, not just the last
  submit.

Expected behaviour:

* Synchronous backend: ``t_combined ~= t_kernel_alone + t_host_alone`` (no
  overlap) -> overlap assertion FAILS.
* Asynchronous backend: ``t_combined ~= max(t_kernel_alone, t_host_alone)``
  -> overlap assertion PASSES.

To run (requires QAIC hardware):

    mojo test_qaic_async_enqueue.mojo
"""

from common import print_separator, print_test_header

from std.memory import UnsafePointer, alloc
from std.sys import exit
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext, Dim
from std._plugin.qaic.host.target import npu_work_range


# ===-----------------------------------------------------------------------===#
# Kernel
# ===-----------------------------------------------------------------------===#


# `n` is declared as a by-value scalar so the call site stages it through
# the meta_storage scalar-arg path in DeviceContext._call_with_npu — the
# lifetime path most likely to break under async execution.
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

# Total launches in the chain. Stay well under the documented ~50-deep
# stream queue limit; 20 leaves comfortable headroom and amortizes per-call
# overhead enough to dominate host calibration noise.
comptime N_LAUNCHES: Int = 20

# Single-thread, single-core kernels over BUF_LEN elements give a workload
# big enough to be measurable in wall clock without writing a custom slow
# kernel.
comptime BUF_LEN: Int = 256 * 1024

# Maximum acceptable t_combined as a fraction of (t_kernel_alone +
# t_host_alone). 0.8 means we require at least 20% wall-clock savings from
# host/device overlap.
comptime OVERLAP_THRESHOLD: Float64 = 0.8

# If either baseline is below this floor, skip the timing assertion: the
# signal-to-noise on overlap is too poor to be reliable.
comptime SANITY_FLOOR_NS: Int = 5_000_000  # 5 ms


# ===-----------------------------------------------------------------------===#
# Helpers
# ===-----------------------------------------------------------------------===#


@always_inline
def ms_from_ns(ns: Int) -> Float64:
    return Float64(ns) / 1_000_000.0


# Pure-CPU spin. Returns the accumulator so the caller can keep the
# computation live; `iterations` controls wall-clock duration.
@always_inline
def busy_work(iterations: Int) -> Float64:
    var x: Float64 = 1.0
    var acc: Float64 = 0.0
    for _ in range(iterations):
        x = x * 1.0000001 + 1e-9
        acc = acc + x
    return acc


# ===-----------------------------------------------------------------------===#
# Test
# ===-----------------------------------------------------------------------===#


def main() raises:
    print_test_header("QAIC Async Enqueue Test")

    var ctx = DeviceContext(device_id=0, api="qaic")

    # Two ping-pong buffers (buf_x, buf_y) and one fixed addend (buf_a).
    # Chain pattern: out = in + a, alternating which of buf_x/buf_y is the
    # input vs output. Starting state buf_x = a_host; after N_LAUNCHES
    # rounds the result is in buf_x when N_LAUNCHES is even.
    var buf_x = ctx.enqueue_create_buffer[DType.float32](BUF_LEN)
    var buf_y = ctx.enqueue_create_buffer[DType.float32](BUF_LEN)
    var buf_a = ctx.enqueue_create_buffer[DType.float32](BUF_LEN)

    var a_host = alloc[Float32](BUF_LEN)
    # Small integer values keep the running sum exactly representable in
    # float32 across N_LAUNCHES + 1 = 21 accumulations.
    for i in range(BUF_LEN):
        a_host[i] = Float32(i % 7 + 1)

    ctx.enqueue_copy(buf_x, a_host)
    ctx.enqueue_copy(buf_a, a_host)
    ctx.synchronize()

    var func = ctx.compile_function[add_kernel]()
    var n_arg = UInt32(BUF_LEN)

    # ---- Warmup (untimed): pays one-time JIT/load costs and primes
    # ---- whatever the runtime caches between calls.
    ctx.enqueue_copy(buf_x, a_host)
    ctx.synchronize()
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
    ctx.synchronize()

    # ---- t_kernel_alone: chain + sync, no host work.
    ctx.enqueue_copy(buf_x, a_host)
    ctx.synchronize()

    var tk0 = perf_counter_ns()
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
    ctx.synchronize()
    var tk1 = perf_counter_ns()
    var t_kernel_ns = Int(tk1 - tk0)

    # ---- Calibrate host work: pick an iteration count whose wall-clock
    # ---- time matches t_kernel_ns.
    var unit_iters = 1_000_000
    var tu0 = perf_counter_ns()
    var probe_unit = busy_work(unit_iters)
    var tu1 = perf_counter_ns()
    var t_unit_ns = Int(tu1 - tu0)
    if t_unit_ns < 1:
        t_unit_ns = 1
    var target_iters = Int(
        Float64(unit_iters) * Float64(t_kernel_ns) / Float64(t_unit_ns)
    )
    if target_iters < 1:
        target_iters = 1

    # ---- t_host_alone: just the calibrated host spin.
    var th0 = perf_counter_ns()
    var probe_host = busy_work(target_iters)
    var th1 = perf_counter_ns()
    var t_host_ns = Int(th1 - th0)

    # ---- t_combined: enqueue chain, run host work in parallel, then sync.
    ctx.enqueue_copy(buf_x, a_host)
    ctx.synchronize()

    var tc0 = perf_counter_ns()
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
    var probe_combined = busy_work(target_iters)
    ctx.synchronize()
    var tc1 = perf_counter_ns()
    var t_combined_ns = Int(tc1 - tc0)

    # ---- Correctness: final result is in buf_x because N_LAUNCHES is even.
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

    # Keep the spin probes live so the optimizer can't elide busy_work.
    print()
    print(
        "Probe sums (kept live; values not meaningful):",
        probe_unit,
        probe_host,
        probe_combined,
    )

    var sum_ns = t_kernel_ns + t_host_ns
    var savings_ns = sum_ns - t_combined_ns
    var savings_pct = 100.0 * Float64(savings_ns) / Float64(sum_ns)

    print()
    print("Measurements:")
    print("  N_LAUNCHES      =", N_LAUNCHES)
    print("  BUF_LEN         =", BUF_LEN)
    print("  target_iters    =", target_iters)
    print("  t_kernel_alone  =", ms_from_ns(t_kernel_ns), "ms")
    print("  t_host_alone    =", ms_from_ns(t_host_ns), "ms")
    print("  t_combined      =", ms_from_ns(t_combined_ns), "ms")
    print("  serial baseline =", ms_from_ns(sum_ns), "ms")
    print(
        "  savings         =", ms_from_ns(savings_ns), "ms (", savings_pct, "%)"
    )

    var threshold_ns = Int(OVERLAP_THRESHOLD * Float64(sum_ns))
    var timing_ok = t_combined_ns < threshold_ns

    if t_kernel_ns < SANITY_FLOOR_NS or t_host_ns < SANITY_FLOOR_NS:
        print()
        print(
            "WARNING: baseline times too small for reliable timing"
            " assertion; skipping overlap check."
        )
        timing_ok = True

    print()
    if errors == 0 and timing_ok:
        print_separator()
        print("ASYNC TEST PASSED")
        print_separator()
    else:
        if errors != 0:
            print("ASYNC TEST FAILED: correctness errors =", errors)
        if not timing_ok:
            print(
                "ASYNC TEST FAILED: t_combined",
                ms_from_ns(t_combined_ns),
                "ms is not <",
                OVERLAP_THRESHOLD,
                "* serial baseline",
                ms_from_ns(sum_ns),
                "ms (threshold",
                ms_from_ns(threshold_ns),
                "ms)",
            )
        exit(1)
