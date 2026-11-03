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
"""Test asynchronous QAIC kernel execution for the closure-based path.

Mirrors ``test_qaic_async_enqueue.mojo`` but exercises
``_call_with_closure_npu`` instead of ``_call_with_pack_npu``. The
closure path bakes the closure blob and grid config into the command
buffer as local storage at record time; this test checks that the
record happens synchronously inside the async enqueue, before the host
frees the blob bytes.

The kernel does in-place accumulation: each call computes
``out[i] = out[i] + inc[i]``. After ``N_LAUNCHES`` calls, ``out`` should
hold ``N_LAUNCHES * inc[i]``. If FIFO ordering is broken or any kernel
silently fails to read the right ``out`` after the previous call, the
final value won't match. The launch pins one worker (``num_cores=1``,
``thread_mask=0x01``) because zero-arg closure bodies run once per
hardware worker — at default max parallelism the accumulation would be
duplicated ~80x.

Non-blocking semantics are checked by the same wall-clock pattern used
in the pack-path test: enqueue + host work + synchronize must be
measurably faster than running them serially.

To run (requires QAIC hardware):

    mojo test_qaic_async_closure.mojo
"""

from common import print_separator, print_test_header

from std.memory import UnsafePointer, alloc
from std.sys import exit
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext, Dim


# ===-----------------------------------------------------------------------===#
# Configuration
# ===-----------------------------------------------------------------------===#

comptime N_LAUNCHES: Int = 20
comptime BUF_LEN: Int = 256 * 1024
comptime OVERLAP_THRESHOLD: Float64 = 0.8
comptime SANITY_FLOOR_NS: Int = 5_000_000  # 5 ms


# ===-----------------------------------------------------------------------===#
# Helpers
# ===-----------------------------------------------------------------------===#


@always_inline
def ms_from_ns(ns: Int) -> Float64:
    return Float64(ns) / 1_000_000.0


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
    print_test_header("QAIC Async Closure Test")

    var ctx = DeviceContext(device_id=0, api="qaic")

    var inc_dev = ctx.enqueue_create_buffer[DType.float32](BUF_LEN)
    var out_dev = ctx.enqueue_create_buffer[DType.float32](BUF_LEN)

    var inc_host = alloc[Float32](BUF_LEN)
    var out_host = alloc[Float32](BUF_LEN)
    # Small integer values keep the running sum exactly representable in
    # float32 across N_LAUNCHES = 20 accumulations.
    for i in range(BUF_LEN):
        inc_host[i] = Float32(i % 7 + 1)
        out_host[i] = 0.0

    ctx.enqueue_copy(inc_dev, inc_host)
    ctx.enqueue_copy(out_dev, out_host)
    ctx.synchronize()

    # `DeviceBuffer.unsafe_ptr()` now ties the pointer origin to the buffer
    # borrow; erase it so the captured pointers don't embed two tracked
    # mutable origins that the closure enqueue's aliasing check rejects.
    var inc_ptr = inc_dev.unsafe_ptr().as_unsafe_any_origin()
    var out_ptr = out_dev.unsafe_ptr().as_unsafe_any_origin()
    var size: Int32 = Int32(BUF_LEN)

    # In-place accumulation: out[i] += inc[i]. The closure captures the
    # device pointers and buffer length by value; the kernel reads the
    # *current contents* of out[] from device memory each call, so each
    # iteration depends on the previous having completed.
    def accum_kernel() {var inc_ptr, var out_ptr, var size}:
        for i in range(Int(size)):
            out_ptr[i] = out_ptr[i] + inc_ptr[i]

    var func = ctx.compile_function[type_of(accum_kernel)]()

    # ---- Warmup (untimed): pays one-time JIT/load costs.
    ctx.enqueue_copy(out_dev, out_host)
    ctx.synchronize()
    for _ in range(N_LAUNCHES):
        ctx.enqueue_function(
            func,
            accum_kernel,
            grid_dim=Dim(1),
            block_dim=Dim(0x01),
            num_cores=1,
            thread_mask=0x01,
        )
    ctx.synchronize()

    # ---- t_kernel_alone: chain + sync, no host work.
    ctx.enqueue_copy(out_dev, out_host)
    ctx.synchronize()
    var tk0 = perf_counter_ns()
    for _ in range(N_LAUNCHES):
        ctx.enqueue_function(
            func,
            accum_kernel,
            grid_dim=Dim(1),
            block_dim=Dim(0x01),
            num_cores=1,
            thread_mask=0x01,
        )
    ctx.synchronize()
    var tk1 = perf_counter_ns()
    var t_kernel_ns = Int(tk1 - tk0)

    # ---- Calibrate host work to roughly match t_kernel_ns.
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
    ctx.enqueue_copy(out_dev, out_host)
    ctx.synchronize()
    var tc0 = perf_counter_ns()
    for _ in range(N_LAUNCHES):
        ctx.enqueue_function(
            func,
            accum_kernel,
            grid_dim=Dim(1),
            block_dim=Dim(0x01),
            num_cores=1,
            thread_mask=0x01,
        )
    var probe_combined = busy_work(target_iters)
    ctx.synchronize()
    var tc1 = perf_counter_ns()
    var t_combined_ns = Int(tc1 - tc0)

    # ---- Correctness: out_dev should hold N_LAUNCHES * inc_host[i].
    var c_host = alloc[Float32](BUF_LEN)
    ctx.enqueue_copy(c_host, out_dev)
    ctx.synchronize()

    _ = inc_dev

    var errors = 0
    var expected_factor = Float32(N_LAUNCHES)
    for i in range(BUF_LEN):
        var expected = inc_host[i] * expected_factor
        if abs(c_host[i] - expected) > 0.001 * abs(expected):
            errors += 1

    inc_host.unsafe_free()
    out_host.unsafe_free()
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
        print("ASYNC CLOSURE TEST PASSED")
        print_separator()
    else:
        if errors != 0:
            print("ASYNC CLOSURE TEST FAILED: correctness errors =", errors)
        if not timing_ok:
            print(
                "ASYNC CLOSURE TEST FAILED: t_combined",
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
