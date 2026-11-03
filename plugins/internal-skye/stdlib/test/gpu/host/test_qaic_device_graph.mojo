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
"""QAIC device-graph test: record a two-kernel chain once and replay it.

Exercises the first-pass QAIC `DeviceGraphBuilder` / `DeviceGraph`
(kernel-dispatch nodes only, with a barrier between every operation):

1. `add_one`   : mid[i] = in[i] + 1
2. `times_two` : out[i] = mid[i] * 2

`times_two` reads what `add_one` wrote, so a correct result proves the
barrier serialized the two dispatches. Inputs are staged with an eager
`enqueue_copy` before `replay()` and outputs read back with an eager
`enqueue_copy` after — the graph itself only holds kernel nodes. The graph
is replayed twice to confirm the command-buffer batch is reusable and the
barrier semaphores re-initialize across replays.

To run (requires QAIC hardware):
    mojo test_qaic_device_graph.mojo
"""

from std.algorithm.functional import map
from std.memory import UnsafePointer, alloc
from std.time import perf_counter_ns

from max.gpu.host import DeviceGraph, DeviceGraphBuilder
from max.gpu.host import DeviceContext
from std.testing import TestSuite

from std._plugin.qaic.host.target import npu_work_range

from qaic_test_utils import (
    alloc_constant,
    alloc_sequential,
    verify_buffer_almost_equal,
)


# ===-----------------------------------------------------------------------===#
# Timing helper
# ===-----------------------------------------------------------------------===#


def _elapsed_us(start_ns: Int, end_ns: Int) -> Float64:
    """Returns the elapsed microseconds between two `perf_counter_ns` reads."""
    return Float64(end_ns - start_ns) / 1000.0


# ===-----------------------------------------------------------------------===#
# Kernels (legacy QAIC style: raw pointers + stdlib `map`)
# ===-----------------------------------------------------------------------===#

comptime N = 64


def add_one_kernel(
    inp: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
):
    """Computes output[i] = inp[i] + 1."""

    def work(i: Int) {var inp, var output}:
        output[i] = inp[i] + 1.0

    map(N, work)


def times_two_kernel(
    inp: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
):
    """Computes output[i] = inp[i] * 2."""

    def work(i: Int) {var inp, var output}:
        output[i] = inp[i] * 2.0

    map(N, work)


# Multi-core variants: `npu_work_range(n)` partitions [0, n) across the launch's
# workers (participating NSP cores x HVX threads), so each worker computes only
# its slice. The result is exact regardless of how many cores/threads run.


def add_one_mc_kernel(
    inp: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    n: UInt32,
):
    """Computes output[i] = inp[i] + 1 over this worker's slice."""
    var work_range = npu_work_range(Int(n))
    for i in range(work_range[0], work_range[1]):
        output[i] = inp[i] + 1.0


def times_two_mc_kernel(
    inp: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    n: UInt32,
):
    """Computes output[i] = inp[i] * 2 over this worker's slice."""
    var work_range = npu_work_range(Int(n))
    for i in range(work_range[0], work_range[1]):
        output[i] = inp[i] * 2.0


# ===-----------------------------------------------------------------------===#
# Test
# ===-----------------------------------------------------------------------===#


def test_qaic_device_graph_chain() raises:
    print("Test: record + replay a two-kernel QAIC device graph.")
    with DeviceContext(device_id=0, api="qaic") as ctx:
        var in_dev = ctx.enqueue_create_buffer[DType.float32](N)
        var mid_dev = ctx.enqueue_create_buffer[DType.float32](N)
        var out_dev = ctx.enqueue_create_buffer[DType.float32](N)

        # Stage inputs eagerly (in[i] = i) before building/replaying the graph.
        var in_host = alloc_sequential[DType.float32](N, start=0)
        ctx.enqueue_copy(in_dev, in_host)
        ctx.synchronize()

        var add_one = ctx.compile_function[add_one_kernel]()
        var times_two = ctx.compile_function[times_two_kernel]()

        # Single core / single thread keeps the first-pass graph minimal: each op
        # is one command buffer (NSP0), serialized by the functional-stream FIFO.
        def build(mut builder: DeviceGraphBuilder) raises {imm}:
            _ = builder.add_function(
                add_one, in_dev, mid_dev, grid_dim=1, block_dim=1
            )
            _ = builder.add_function(
                times_two, mid_dev, out_dev, grid_dim=1, block_dim=1
            )

        # Time graph creation (records + ends the per-op command-buffer batches).
        var t_create0 = Int(perf_counter_ns())
        var graph = DeviceGraph.create(ctx, build)
        var t_create1 = Int(perf_counter_ns())
        var create_us = _elapsed_us(t_create0, t_create1)

        # Expected: (i + 1) * 2.
        var expected = alloc_sequential[DType.float32](N, start=0)
        for i in range(N):
            expected[i] = Float32((i + 1) * 2)

        var out_host = alloc[Float32](N)

        # First replay. `replay()` submits every batch and blocks until the device
        # finishes, so this wall-clock is the end-to-end replay time.
        var t_replay0 = Int(perf_counter_ns())
        graph.replay()
        var t_replay1 = Int(perf_counter_ns())
        var replay_us = _elapsed_us(t_replay0, t_replay1)
        ctx.enqueue_copy(out_host, out_dev)
        ctx.synchronize()
        verify_buffer_almost_equal(out_host, expected, N)
        print("  first replay PASSED")
        print("  timing: graph creation =", create_us, "us")
        print("  timing: replay         =", replay_us, "us")
        print("  timing: end-to-end     =", create_us + replay_us, "us")

        # Overwrite the intermediate and output buffers with a sentinel before the
        # second replay, so passing verification proves the replay actually
        # recomputed both kernels rather than observing values left over from the
        # first run.
        var sentinel = alloc_constant[DType.float32](N, Float32(-1.0))
        ctx.enqueue_copy(mid_dev, sentinel)
        ctx.enqueue_copy(out_dev, sentinel)
        ctx.synchronize()

        # Confirm the sentinel actually landed (guards against the copy silently
        # not taking effect, which would make the check below vacuous).
        ctx.enqueue_copy(out_host, out_dev)
        ctx.synchronize()
        for i in range(N):
            if out_host[i] != Float32(-1.0):
                raise Error("sentinel clear did not take effect before replay")

        # Second replay: proves the batch is reusable and recomputes from scratch
        # (times_two sees add_one's fresh output, not the cleared sentinel).
        graph.replay()
        ctx.synchronize()
        ctx.enqueue_copy(out_host, out_dev)
        ctx.synchronize()
        verify_buffer_almost_equal(out_host, expected, N)
        print("  second replay PASSED")

        in_host.unsafe_free()
        expected.unsafe_free()
        out_host.unsafe_free()
        sentinel.unsafe_free()

        print("DEVICE GRAPH TEST PASSED")


def test_qaic_device_graph_multicore() raises:
    print("Test: record + replay a multi-core (multi-NSP) QAIC device graph.")
    comptime MC_N = 4096
    comptime MC_CORES = 4  # dispatch across 4 NSPs (clamped to nspCount)
    comptime MC_THREAD_MASK = 0x1F  # all 5 HVX threads per core
    with DeviceContext(device_id=0, api="qaic") as ctx:
        var in_dev = ctx.enqueue_create_buffer[DType.float32](MC_N)
        var mid_dev = ctx.enqueue_create_buffer[DType.float32](MC_N)
        var out_dev = ctx.enqueue_create_buffer[DType.float32](MC_N)

        var in_host = alloc_sequential[DType.float32](MC_N, start=0)
        ctx.enqueue_copy(in_dev, in_host)
        ctx.synchronize()

        var add_one = ctx.compile_function[add_one_mc_kernel]()
        var times_two = ctx.compile_function[times_two_mc_kernel]()

        var n_arg = UInt32(MC_N)

        # Each operation records a batch of MC_CORES command buffers (one per NSP);
        # the workers partition the buffer via npu_work_range, so the two-kernel
        # chain is exact only if every worker ran and the ops stayed ordered.
        def build(mut builder: DeviceGraphBuilder) raises {imm}:
            _ = builder.add_function(
                add_one,
                in_dev,
                mid_dev,
                n_arg,
                grid_dim=MC_CORES,
                block_dim=MC_THREAD_MASK,
            )
            _ = builder.add_function(
                times_two,
                mid_dev,
                out_dev,
                n_arg,
                grid_dim=MC_CORES,
                block_dim=MC_THREAD_MASK,
            )

        # Time graph creation (records + ends MC_CORES command buffers per op).
        var t_create0 = Int(perf_counter_ns())
        var graph = DeviceGraph.create(ctx, build)
        var t_create1 = Int(perf_counter_ns())
        var create_us = _elapsed_us(t_create0, t_create1)

        var expected = alloc_sequential[DType.float32](MC_N, start=0)
        for i in range(MC_N):
            expected[i] = Float32((i + 1) * 2)

        var out_host = alloc[Float32](MC_N)

        # Time end-to-end replay (submit every batch, block until the device is
        # done).
        var t_replay0 = Int(perf_counter_ns())
        graph.replay()
        var t_replay1 = Int(perf_counter_ns())
        var replay_us = _elapsed_us(t_replay0, t_replay1)
        ctx.enqueue_copy(out_host, out_dev)
        ctx.synchronize()
        verify_buffer_almost_equal(out_host, expected, MC_N)
        print("  multi-core replay PASSED")
        print("  timing: graph creation =", create_us, "us")
        print("  timing: replay         =", replay_us, "us")
        print("  timing: end-to-end     =", create_us + replay_us, "us")

        in_host.unsafe_free()
        expected.unsafe_free()
        out_host.unsafe_free()

        print("MULTI-CORE DEVICE GRAPH TEST PASSED")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
