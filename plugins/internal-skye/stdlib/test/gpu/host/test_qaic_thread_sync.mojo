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
"""Test thread synchronization via qshim_create_thread_sync_point / qshim_sync_threads.

Demonstrates a two-phase parallel normalization kernel where the barrier is
**required** for correctness:

  Phase 1 — Each thread computes the sum of its chunk and writes it into a
            shared `partial_sums[thread_id]` slot in DDR.

  *** BARRIER (qshim_sync_threads) ***

  Phase 2 — Every thread reads ALL partial sums, computes the global sum,
            and divides each element in its chunk by that global sum.

Without the barrier a fast thread can reach Phase 2 before slower threads
finish Phase 1, reading zero (uninitialised) for their partial sums.  The
global sum is then too small and the normalised values are too large, making
the output sum != 1.0.

To maximise the race window the work is **deliberately skewed**: thread 0
gets only a tiny slice (64 elements) while threads 1-3 split the remaining
65472 elements equally (~21824 each).  Thread 0 finishes Phase 1 ~340x
faster and almost certainly reaches Phase 2 before the others write their
partial sums.

Buffer layout passed to the kernel via pointer_array:
    [0] input   — Float32[N]            (read-only)
    [1] output  — Float32[N]            (write-only)
    [2] meta    — UInt32[2 + NUM_THREADS + 2]
                  [0]: N (vector length)
                  [1]: NUM_THREADS
                  [2 .. 2+NUM_THREADS-1]: partial_sums (Float32, written by kernel)
                  [2+NUM_THREADS .. 2+NUM_THREADS+1]: QShimThreadSyncStruct

To run (requires QAIC hardware):
    mojo test_qaic_thread_sync.mojo
"""

from std.testing import TestSuite, assert_equal, assert_not_equal
from common import print_separator, print_test_header

from std.memory import UnsafePointer, alloc

from max.gpu.host import DeviceContext
from std._plugin.qaic.host.target import compile_info_npu
from max._plugin.qaic.host.qaic import (
    qaic_get_jit_device,
    qaic_allocate_jit_buffer,
    qaic_release_jit_buffer,
    qaic_load_jit_function,
    qaic_execute_jit_function,
    qaic_release_jit_function,
    qaic_copy_jit_buffer,
    QAicBuf,
    QAIC_BUFFER_TYPE_DEV_DDR,
    QAIC_DEV_BUF_IO_RW,
    QAIC_COPY_TO_DEVICE,
    QAIC_COPY_FROM_DEVICE,
)

from std._plugin.qaic.host.qshim import (
    qshim_create_thread_sync_point,
    qshim_get_thread_id,
    qshim_sync_threads,
)


# ===-----------------------------------------------------------------------===#
# Configuration
# ===-----------------------------------------------------------------------===#

comptime NUM_THREADS: Int = 4

# Thread 0 gets only this many elements; the rest is split among threads 1-3.
# Keeping thread 0's chunk tiny maximises the window in which it can race
# ahead of the other threads and read their not-yet-written partial sums.
comptime THREAD0_CHUNK: Int = 64


# ===-----------------------------------------------------------------------===#
# Skewed work distribution helper
# ===-----------------------------------------------------------------------===#


@always_inline
def _skewed_range(thread_id: Int, n: Int) -> Tuple[Int, Int]:
    """Return (start, end) for a deliberately skewed work split.

    Thread 0 gets THREAD0_CHUNK elements; the remaining (n - THREAD0_CHUNK)
    elements are divided equally among threads 1 .. NUM_THREADS-1.
    This makes thread 0 finish Phase 1 much faster than the others,
    widening the race window when the barrier is absent.
    """
    if thread_id == 0:
        return (0, THREAD0_CHUNK)

    var rest = n - THREAD0_CHUNK
    var per_thread = rest // (NUM_THREADS - 1)
    var start = THREAD0_CHUNK + (thread_id - 1) * per_thread
    var end: Int
    if thread_id == NUM_THREADS - 1:
        end = n  # last thread picks up any remainder
    else:
        end = start + per_thread
    return (start, end)


# ===-----------------------------------------------------------------------===#
# Kernel: Parallel Normalize (WITH barrier)
# ===-----------------------------------------------------------------------===#


def normalize_with_sync_kernel(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    meta: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    """Two-phase parallel normalization with thread synchronization.

    Uses qshim_get_thread_id() to identify the current thread and
    qshim_create_thread_sync_point() / qshim_sync_threads() to barrier
    all threads between the two phases.

    Phase 1: Each thread computes partial sum of its chunk.
    BARRIER: qshim_sync_threads ensures all partial sums are visible.
    Phase 2: Each thread computes global sum and normalizes its chunk.

    Without the barrier, threads may read stale partial_sums from other
    threads, producing incorrect output.

    Args:
        input: Input vector (float32).
        output: Output vector (float32).
        meta: Metadata array (uint32): [N, num_threads, partial_sums..., sync].
    """
    var thread_id = Int(qshim_get_thread_id())

    var n = Int(meta[0])
    var num_threads = Int(meta[1])

    # partial_sums lives at meta[2..2+num_threads-1] (reinterpreted as Float32)
    var partial_sums = (meta + 2).bitcast[Float32]()

    # Allocate the sync struct in kernel global memory using
    # qshim_create_thread_sync_point (tests pop.global_alloc path).
    # thread_mask 0x0F = threads 0-3.
    var sync_ptr = qshim_create_thread_sync_point[
        thread_mask=Int32((1 << NUM_THREADS) - 1)
    ]()

    # ── Skewed work distribution — thread 0 gets a tiny chunk ──
    var start: Int
    var end: Int
    start, end = _skewed_range(thread_id, n)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: Each thread computes partial sum of its chunk
    # ══════════════════════════════════════════════════════════════════
    var my_sum: Float32 = 0.0
    for i in range(start, end):
        my_sum += input[i]

    # Write partial sum to shared slot — other threads will need this.
    partial_sums[thread_id] = my_sum

    # ══════════════════════════════════════════════════════════════════
    # BARRIER: Wait for ALL threads to finish writing their partial sums.
    #
    # WITHOUT THIS BARRIER a fast thread can reach Phase 2 while slower
    # threads are still in Phase 1.  It would read 0.0 for those slots,
    # giving a global_sum that is too small.  The normalised values in
    # its chunk would then be too large → output sum != 1.0.
    # ══════════════════════════════════════════════════════════════════
    _ = qshim_sync_threads(UInt32(thread_id), sync_ptr.as_unsafe_any_origin())

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: Every thread computes the global sum from ALL partial sums
    # ══════════════════════════════════════════════════════════════════
    var global_sum: Float32 = 0.0
    for t in range(num_threads):
        global_sum += partial_sums[t]

    # Normalise this thread's chunk: output[i] = input[i] / global_sum
    for i in range(start, end):
        output[i] = input[i] / global_sum


# ===-----------------------------------------------------------------------===#
# Kernel: Parallel Normalize (WITHOUT barrier — demonstrates the bug)
# ===-----------------------------------------------------------------------===#


def normalize_no_sync_kernel(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    meta: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    """Same algorithm but WITHOUT the barrier.

    Threads may read uninitialised (zero) partial sums written by other
    threads, leading to an incorrect global_sum and wrong normalisation.

    Args:
        input: Input vector (float32).
        output: Output vector (float32).
        meta: Metadata array (uint32): [N, num_threads, partial_sums..., sync].
    """
    var thread_id = Int(qshim_get_thread_id())

    var n = Int(meta[0])
    var num_threads = Int(meta[1])
    var partial_sums = (meta + 2).bitcast[Float32]()

    # No sync struct needed — we intentionally skip the barrier.

    # ── Skewed work distribution — thread 0 gets a tiny chunk ──
    var start: Int
    var end: Int
    start, end = _skewed_range(thread_id, n)

    # Phase 1: compute partial sum
    var my_sum: Float32 = 0.0
    for i in range(start, end):
        my_sum += input[i]

    partial_sums[thread_id] = my_sum

    # *** NO BARRIER HERE ***
    # A fast thread may read other threads' partial_sums before they are
    # written, seeing 0.0 instead of the real value.

    # Phase 2: compute global sum and normalise
    var global_sum: Float32 = 0.0
    for t in range(num_threads):
        global_sum += partial_sums[t]

    for i in range(start, end):
        output[i] = input[i] / global_sum


# ===-----------------------------------------------------------------------===#
# Run one kernel variant and return the error count
# ===-----------------------------------------------------------------------===#


def run_kernel[
    kernel_fn: def(
        UnsafePointer[Float32, MutUnsafeAnyOrigin],
        UnsafePointer[Float32, MutUnsafeAnyOrigin],
        UnsafePointer[UInt32, MutUnsafeAnyOrigin],
    ) thin -> None,
    label: StaticString,
](vector_size: Int) raises -> Int:
    """Compile, execute, and verify a normalisation kernel.

    Returns the number of verification errors (0 = pass).
    """
    print()
    print_separator()
    print(label)
    print_separator()

    var data_size = vector_size * 4  # Float32 = 4 bytes

    # meta layout: [N:4][num_threads:4][partial_sums:NUM_THREADS*4][sync:8]
    var meta_uint32_count = (
        2 + NUM_THREADS + 2
    )  # N, num_threads, partials, sync
    var meta_size = meta_uint32_count * 4

    # ── Compile ──
    print("  Compiling...")
    var elf = compile_info_npu[kernel_fn]()
    print("  ELF size:", elf.asm.byte_length(), "bytes")

    # ── Device setup ──
    var ctx = DeviceContext(device_id=0, api="qaic")
    var jit_device = qaic_get_jit_device(ctx)
    var jit_func = qaic_load_jit_function(
        ctx,
        elf.function_name,
        elf.asm.unsafe_ptr().bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(elf.asm.byte_length()),
    )

    # ── Allocate device buffers ──
    var alignment: UInt32 = 128
    var cu_bitmap: UInt64 = 0xFFFFFFFFFFFFFFFF

    var input_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        UInt64(data_size),
        alignment,
        cu_bitmap,
    )
    var output_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        UInt64(data_size),
        alignment,
        cu_bitmap,
    )
    var meta_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        UInt64(meta_size),
        alignment,
        cu_bitmap,
    )

    # ── Prepare host data ──
    # Use distinct, non-uniform values so a wrong global_sum is obvious.
    # input[i] = (i + 1) so no zeros; expected global_sum = N*(N+1)/2.
    var input_host = alloc[Float32](vector_size)
    var expected_global_sum: Float32 = 0.0
    for i in range(vector_size):
        input_host[i] = Float32(i + 1)
        expected_global_sum += Float32(i + 1)

    var meta_host = alloc[UInt32](meta_uint32_count)
    meta_host[0] = UInt32(vector_size)
    meta_host[1] = UInt32(NUM_THREADS)
    # Zero-initialise partial_sums — this is what an unsynchronised thread
    # would read if it races ahead of a slower writer.
    for i in range(NUM_THREADS):
        meta_host[2 + i] = UInt32(0)
    # Initialise the QShimThreadSyncStruct (thread_mask, data)
    var thread_mask = UInt32((1 << NUM_THREADS) - 1)
    meta_host[2 + NUM_THREADS] = thread_mask  # sync.thread_mask
    meta_host[2 + NUM_THREADS + 1] = UInt32(0)  # sync.data

    print("  Vector size:", vector_size)
    print("  Threads:", NUM_THREADS, " thread_mask:", thread_mask)
    print("  Expected global sum:", expected_global_sum)

    # ── Copy to device ──
    qaic_copy_jit_buffer(
        ctx,
        input_buf,
        input_host.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(data_size),
        QAIC_COPY_TO_DEVICE,
    )
    qaic_copy_jit_buffer(
        ctx,
        meta_buf,
        meta_host.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(meta_size),
        QAIC_COPY_TO_DEVICE,
    )

    # ── Execute ──
    var buffers = alloc[QAicBuf](3)
    buffers[0] = input_buf
    buffers[1] = output_buf
    buffers[2] = meta_buf

    print("  Executing kernel...")
    qaic_execute_jit_function(
        ctx, jit_func, buffers.as_unsafe_any_origin(), 3, 1, thread_mask
    )
    print("  Done.")

    # ── Copy results back ──
    var output_host = alloc[Float32](vector_size)
    qaic_copy_jit_buffer(
        ctx,
        output_buf,
        output_host.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(data_size),
        QAIC_COPY_FROM_DEVICE,
    )

    # ── Verify ──
    # Correct normalisation: output[i] = input[i] / global_sum.
    # Therefore sum(output) should equal 1.0.
    var errors = 0
    var output_sum: Float32 = 0.0
    for i in range(vector_size):
        var expected = Float32(i + 1) / expected_global_sum
        var actual = output_host[i]
        output_sum += actual
        if abs(actual - expected) > 1e-4:
            errors += 1
            if errors <= 8:
                # Determine owning thread from skewed distribution
                var owning_thread: Int = 0
                if i >= THREAD0_CHUNK:
                    var per_thread = (vector_size - THREAD0_CHUNK) // (
                        NUM_THREADS - 1
                    )
                    owning_thread = 1 + (i - THREAD0_CHUNK) // per_thread
                    if owning_thread >= NUM_THREADS:
                        owning_thread = NUM_THREADS - 1
                print(
                    "    MISMATCH idx",
                    i,
                    "(thread",
                    owning_thread,
                    "): expected",
                    expected,
                    "got",
                    actual,
                )

    print("  Sum of all output elements:", output_sum, "(expected 1.0)")

    if errors == 0:
        print("  PASS — all", vector_size, "elements correct.")
    else:
        print("  FAIL —", errors, "mismatches out of", vector_size, "elements.")

    # ── Cleanup ──
    input_host.unsafe_free()
    meta_host.unsafe_free()
    output_host.unsafe_free()
    buffers.unsafe_free()
    qaic_release_jit_buffer(ctx, input_buf)
    qaic_release_jit_buffer(ctx, output_buf)
    qaic_release_jit_buffer(ctx, meta_buf)
    qaic_release_jit_function(jit_func)

    return errors


# ===-----------------------------------------------------------------------===#
# Main
# ===-----------------------------------------------------------------------===#


def test_qaic_thread_sync() raises:
    print_test_header(
        "QAIC Thread Synchronization Test\nTests qshim_create_thread_sync_point"
        " / qshim_sync_threads"
    )

    # 65536 elements with skewed distribution:
    #   Thread 0: 64 elements   (finishes Phase 1 almost instantly)
    #   Threads 1-3: ~21824 each (still computing when thread 0 enters Phase 2)
    var vector_size: Int = 65536

    # ── Run the CORRECT kernel (with barrier) ──
    var errors_sync: Int
    try:
        errors_sync = run_kernel[
            normalize_with_sync_kernel, "Kernel WITH synchronization"
        ](vector_size)
    except e:
        print("ERROR running synced kernel:", e)
        errors_sync = -1

    # ── Run the BUGGY kernel (without barrier) ──
    var errors_nosync: Int
    try:
        errors_nosync = run_kernel[
            normalize_no_sync_kernel, "Kernel WITHOUT synchronization (racy)"
        ](vector_size)
    except e:
        print("ERROR running unsynced kernel:", e)
        errors_nosync = -1

    # ── Summary ──
    print()
    print_separator()
    print("SUMMARY")
    print_separator()
    print(
        "  With sync   :",
        "PASS" if errors_sync == 0 else "FAIL",
        "(" + String(errors_sync) + " errors)",
    )
    print(
        "  Without sync:",
        "PASS" if errors_nosync == 0 else "FAIL",
        "(" + String(errors_nosync) + " errors)",
    )
    print()
    if errors_sync == 0 and errors_nosync > 0:
        print("Expected outcome: synced kernel passes, unsynced kernel fails.")
        print(
            "This confirms that qshim_sync_threads is required for"
            " cross-thread data visibility."
        )
    elif errors_sync == 0 and errors_nosync == 0:
        print("Note: The unsynced kernel happened to pass this run.  Thread")
        print("scheduling can mask the race.  Run repeatedly or with larger")
        print("vectors to observe failures.")
    else:
        print("Unexpected result — investigate.")
    print_separator()

    # Return error code if we have unexpected results
    assert_equal(errors_sync, 0)
    assert_not_equal(errors_nosync, 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
