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
"""QAIC device-graph micro-benchmarks: creation vs dispatch vs execution.

All three benchmarks run the *same* device function (`bench_step_kernel`, an
idempotent per-element affine recurrence) on the same input, `REPEATS` times
total, single NSP core / single HVX thread, so the total on-device work is
identical. They differ only in how the work is submitted:

0. `test_bench0_enqueue_function_per_call`: baseline that submits each kernel
   via the eager `DeviceContext.enqueue_function` path (no device graph), one
   at a time, `REPEATS` times. Same per-call strategy as Test 1, differing only
   in the submission mechanism.

1. `test_bench1_graph_per_call`: build a one-node graph and replay it, in a
   loop `REPEATS` times (a fresh graph each iteration). Pays graph creation
   (command-buffer record) + replay `REPEATS` times.

2. `test_bench2_one_graph_many_nodes`: build ONE graph with `REPEATS` nodes,
   then replay it once. Command-buffer creation happens once (timed
   separately); the replay submits all `REPEATS` recorded batches.

3. `test_bench3_fused_node`: build a one-node graph whose single node dispatches
   a fused kernel that performs the per-element work `REPEATS` times internally,
   then replay it. Creation + dispatch happen once; only the on-device
   execution is done `REPEATS` times.

Reading the results (same total device work in all three):

- test1_total - test2_replay  ~=  REPEATS x (graph create + teardown per node)
- test2_replay - test3_replay  ~=  REPEATS x (per-node submit/dispatch cost)
- test3_replay                 ~=  the on-device execution floor

To run (requires QAIC hardware, manual — perf/print-heavy, ~minutes):
    mojo test_qaic_device_graph_bench.mojo
"""

from std.memory import UnsafePointer, alloc
from std.time import perf_counter_ns

from max.gpu.host import DeviceGraph, DeviceGraphBuilder
from max.gpu.host import DeviceBuffer, DeviceContext
from std.testing import TestSuite


# ===-----------------------------------------------------------------------===#
# Configuration
# ===-----------------------------------------------------------------------===#

comptime N_ELEMS = 256
comptime INNER_ITERS = 32  # affine-recurrence steps per invocation of the fn
comptime REPEATS = 1000  # times the device function is executed per benchmark

# The affine map v -> v*0.5 + 1 has fixed point 2.0; INNER_ITERS steps from
# input 1.0 converge to ~2.0, so every benchmark's output is ~2.0.
comptime FIXED_POINT = Float32(2.0)


# ===-----------------------------------------------------------------------===#
# Device functions (single-thread loop over the whole buffer; idempotent)
# ===-----------------------------------------------------------------------===#


@no_inline
def bench_step_kernel(
    input: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    n: UInt32,
):
    """One invocation of the chosen device function: INNER_ITERS affine steps.
    """
    for i in range(Int(n)):
        var v = input[i]
        for _ in range(INNER_ITERS):
            v = v * 0.5 + 1.0
        output[i] = v


def bench_fused_kernel(
    input: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    n: UInt32,
):
    """Fused kernel: performs the per-element work REPEATS times in sequence."""
    comptime for _ in range(REPEATS):
        bench_step_kernel(input, output, n)


# ===-----------------------------------------------------------------------===#
# Helpers
# ===-----------------------------------------------------------------------===#


def _elapsed_ms(start_ns: Int, end_ns: Int) -> Float64:
    return Float64(end_ns - start_ns) / 1.0e6


def _verify_converged(
    ctx: DeviceContext,
    out_dev: DeviceBuffer[DType.float32],
    label: String,
) raises:
    var host = alloc[Float32](N_ELEMS)
    ctx.enqueue_copy(host, out_dev)
    ctx.synchronize()
    var mismatches = 0
    for i in range(N_ELEMS):
        if abs(host[i] - FIXED_POINT) > 0.01:
            mismatches += 1
    host.unsafe_free()
    if mismatches != 0:
        raise Error(
            String(label) + ": " + String(mismatches) + " outputs not ~2.0"
        )


# ===-----------------------------------------------------------------------===#
# Benchmarks
# ===-----------------------------------------------------------------------===#


def test_bench0_enqueue_function_per_call() raises:
    print(
        "Test 0: eager enqueue_function (no device graph),",
        REPEATS,
        "submissions",
    )
    var ctx = DeviceContext(device_id=0, api="qaic")
    var in_dev = ctx.enqueue_create_buffer[DType.float32](N_ELEMS)
    var out_dev = ctx.enqueue_create_buffer[DType.float32](N_ELEMS)

    var in_host = alloc[Float32](N_ELEMS)
    for i in range(N_ELEMS):
        in_host[i] = 1.0
    ctx.enqueue_copy(in_dev, in_host)
    ctx.synchronize()

    var func = ctx.compile_function[bench_step_kernel]()
    var n_arg = UInt32(N_ELEMS)

    # Same per-call strategy as Test 1 (submit one kernel, wait, repeat), but
    # via the eager DeviceContext.enqueue_function path instead of building and
    # replaying a device graph. The per-iteration synchronize matches Test 1's
    # synchronous replay so the two differ only in the submission mechanism.
    ctx.enqueue_function(
        func, in_dev, out_dev, n_arg, grid_dim=1, block_dim=1
    )  # warm-up
    ctx.synchronize()

    var t0 = Int(perf_counter_ns())
    for _ in range(REPEATS):
        ctx.enqueue_function(
            func, in_dev, out_dev, n_arg, grid_dim=1, block_dim=1
        )
        ctx.synchronize()
    var t1 = Int(perf_counter_ns())

    _verify_converged(ctx, out_dev, "test0")
    var total_ms = _elapsed_ms(t0, t1)
    print("  total (enqueue_function+sync x", REPEATS, ") =", total_ms, "ms")
    print(
        "  per iteration                       =",
        total_ms * 1000.0 / Float64(REPEATS),
        "us",
    )
    in_host.unsafe_free()
    print("  TEST 0 DONE")


def test_bench1_graph_per_call() raises:
    print("Test 1: one-node graph created + replayed,", REPEATS, "times")
    var ctx = DeviceContext(device_id=0, api="qaic")
    var in_dev = ctx.enqueue_create_buffer[DType.float32](N_ELEMS)
    var out_dev = ctx.enqueue_create_buffer[DType.float32](N_ELEMS)

    var in_host = alloc[Float32](N_ELEMS)
    for i in range(N_ELEMS):
        in_host[i] = 1.0
    ctx.enqueue_copy(in_dev, in_host)
    ctx.synchronize()

    var func = ctx.compile_function[bench_step_kernel]()
    var n_arg = UInt32(N_ELEMS)

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        _ = builder.add_function(
            func, in_dev, out_dev, n_arg, grid_dim=1, block_dim=1
        )

    # Warm-up (excluded from timing).
    var warm = DeviceGraph.create(ctx, build)
    warm.replay()

    var t0 = Int(perf_counter_ns())
    for _ in range(REPEATS):
        var g = DeviceGraph.create(ctx, build)
        g.replay()
    var t1 = Int(perf_counter_ns())

    _verify_converged(ctx, out_dev, "test1")
    var total_ms = _elapsed_ms(t0, t1)
    print("  total (create+replay x", REPEATS, ") =", total_ms, "ms")
    print(
        "  per iteration                =",
        total_ms * 1000.0 / Float64(REPEATS),
        "us",
    )
    in_host.unsafe_free()
    print("  TEST 1 DONE")


def test_bench2_one_graph_many_nodes() raises:
    print("Test 2: one graph with", REPEATS, "nodes, replayed once")
    var ctx = DeviceContext(device_id=0, api="qaic")
    var in_dev = ctx.enqueue_create_buffer[DType.float32](N_ELEMS)
    var out_dev = ctx.enqueue_create_buffer[DType.float32](N_ELEMS)

    var in_host = alloc[Float32](N_ELEMS)
    for i in range(N_ELEMS):
        in_host[i] = 1.0
    ctx.enqueue_copy(in_dev, in_host)
    ctx.synchronize()

    var func = ctx.compile_function[bench_step_kernel]()
    var n_arg = UInt32(N_ELEMS)

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        for _ in range(REPEATS):
            _ = builder.add_function(
                func, in_dev, out_dev, n_arg, grid_dim=1, block_dim=1
            )

    var t_create0 = Int(perf_counter_ns())
    var g = DeviceGraph.create(ctx, build)
    var t_create1 = Int(perf_counter_ns())

    var t_replay0 = Int(perf_counter_ns())
    g.replay()
    var t_replay1 = Int(perf_counter_ns())

    _verify_converged(ctx, out_dev, "test2")
    var create_ms = _elapsed_ms(t_create0, t_create1)
    var replay_ms = _elapsed_ms(t_replay0, t_replay1)
    print("  graph creation (", REPEATS, "nodes) =", create_ms, "ms")
    print("  replay (execute all nodes)     =", replay_ms, "ms")
    print(
        "  per-node create                =",
        create_ms * 1000.0 / Float64(REPEATS),
        "us",
    )
    print(
        "  per-node replay                =",
        replay_ms * 1000.0 / Float64(REPEATS),
        "us",
    )
    in_host.unsafe_free()
    print("  TEST 2 DONE")


def test_bench3_fused_node() raises:
    print("Test 3: one-node graph, fused kernel doing", REPEATS, "x work")
    var ctx = DeviceContext(device_id=0, api="qaic")
    var in_dev = ctx.enqueue_create_buffer[DType.float32](N_ELEMS)
    var out_dev = ctx.enqueue_create_buffer[DType.float32](N_ELEMS)

    var in_host = alloc[Float32](N_ELEMS)
    for i in range(N_ELEMS):
        in_host[i] = 1.0
    ctx.enqueue_copy(in_dev, in_host)
    ctx.synchronize()

    var func = ctx.compile_function[bench_fused_kernel]()
    var n_arg = UInt32(N_ELEMS)

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        _ = builder.add_function(
            func, in_dev, out_dev, n_arg, grid_dim=1, block_dim=1
        )

    var t_create0 = Int(perf_counter_ns())
    var g = DeviceGraph.create(ctx, build)
    var t_create1 = Int(perf_counter_ns())

    var t_replay0 = Int(perf_counter_ns())
    g.replay()
    var t_replay1 = Int(perf_counter_ns())

    _verify_converged(ctx, out_dev, "test3")
    print(
        "  graph creation (1 fused node)  =",
        _elapsed_ms(t_create0, t_create1),
        "ms",
    )
    print(
        "  replay (on-device execution)   =",
        _elapsed_ms(t_replay0, t_replay1),
        "ms",
    )
    in_host.unsafe_free()
    print("  TEST 3 DONE")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
