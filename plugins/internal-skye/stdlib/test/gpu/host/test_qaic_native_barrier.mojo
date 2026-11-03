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
"""Tests native intra- and cross-NSP barriers on real hardware.

Each test compiles a synced kernel and an unsynced kernel of an otherwise
identical two phase parallel and normalize pattern; the synced kernel must pass
(sum(output) == 1.0) and the unsynced kernel must race and fail. Together
they pin the barrier as the cause of the correct run.

Work is deliberately skewed (a "fast" thread finishes Phase 1 while its peers
are still summing) to widen the race window that a missing barrier fails
through.

To run (requires QAIC hardware):
    mojo test_qaic_native_barrier.mojo
"""

from common import print_separator, print_test_header

from std.bit import pop_count
from std.memory import UnsafePointer, alloc
from std.testing import TestSuite, assert_equal, assert_not_equal

from max._plugin.qaic.host.qaic import (
    QAIC_BUFFER_TYPE_DEV_DDR,
    QAIC_COPY_FROM_DEVICE,
    QAIC_COPY_TO_DEVICE,
    QAIC_DEV_BUF_IO_RW,
    QAicBuf,
    qaic_allocate_jit_buffer,
    qaic_copy_jit_buffer,
    qaic_execute_jit_function,
    qaic_get_jit_device,
    qaic_load_jit_function,
    qaic_release_jit_buffer,
    qaic_release_jit_function,
)
from max.gpu.host import DeviceContext

from std._plugin.qaic.host.sync import (
    sync_hmx_and_dma_threads,
    sync_hmx_and_dma_threads_across_nsps,
    sync_hvx_and_dma_threads,
    sync_hvx_and_dma_threads_across_nsps,
    sync_hvx_and_hmx_threads,
    sync_hvx_and_hmx_threads_across_nsps,
    sync_hvx_threads,
    sync_hvx_threads_across_nsps,
    sync_threads,
    sync_threads_across_nsps,
)
from std._plugin.qaic.host.target import (
    compile_info_npu,
    hmx_and_dma_mask,
    hvx_and_dma_mask,
    hvx_and_hmx_mask,
    hvx_mask,
    is_aic100,
    npu_core_idx,
    npu_thread_idx,
    num_threads,
    thread_mask,
)


# ===-----------------------------------------------------------------------===#
# Configuration
# ===-----------------------------------------------------------------------===#

# Arch-parametric thread count. aic100: 5; aic200: 10. Meta buffers are sized
# so any launched thread's slot fits; non-participants leave slots at 0.
comptime NUM_THREADS_MAX: Int = num_threads()

# Fast thread's chunk is tiny so it enters Phase 2 well before its peers.
comptime THREAD0_CHUNK: Int = 64

# Subset test: threads 1 and 2 participate.
comptime SUBSET_MASK: Int = 0b0110

# Cross-NSP test: launch on this many NSPs.
comptime NUM_NSPS_TEST: Int = 4

# NSP 0's slice is tiny so it races into Phase 2 while its peers are still
# summing — same skew idea as THREAD0_CHUNK, one level up.
comptime NSP0_CHUNK: Int = 4096

# Barrier tags. The kernel template dispatches to a specific barrier (or none)
# via `comptime if tag == ...`.
comptime TAG_NONE = "none"
comptime TAG_HVX = "hvx"
comptime TAG_ALL = "all"
comptime TAG_SUBSET = "subset"
comptime TAG_HVX_HMX = "hvx_hmx"
comptime TAG_HVX_DMA = "hvx_dma"
comptime TAG_HMX_DMA = "hmx_dma"


# ===-----------------------------------------------------------------------===#
# Work-split helpers
# ===-----------------------------------------------------------------------===#


@always_inline
def _skewed_range(idx: Int, n: Int, count: Int) -> Tuple[Int, Int]:
    """Skewed split: slot 0 gets THREAD0_CHUNK, the rest share `n - THREAD0_CHUNK`.

    Args:
        idx: Zero-based slot in [0, count).
        n: Total number of elements to divide.
        count: Number of slots.

    Returns:
        `(start, end)` for this slot; `count == 1` returns the whole range.
    """
    if count == 1:
        return (0, n)
    if idx == 0:
        return (0, THREAD0_CHUNK)

    var rest = n - THREAD0_CHUNK
    var per = rest // (count - 1)
    var start = THREAD0_CHUNK + (idx - 1) * per
    var end: Int
    if idx == count - 1:
        end = n
    else:
        end = start + per
    return (start, end)


@always_inline
def _skewed_nsp_range(nsp_id: Int, n: Int, num_nsps: Int) -> Tuple[Int, Int]:
    """NSP-level skewed split. NSP 0 gets NSP0_CHUNK; the rest share the tail.
    """
    if num_nsps == 1:
        return (0, n)
    if nsp_id == 0:
        return (0, NSP0_CHUNK)

    var rest = n - NSP0_CHUNK
    var per = rest // (num_nsps - 1)
    var start = NSP0_CHUNK + (nsp_id - 1) * per
    var end: Int
    if nsp_id == num_nsps - 1:
        end = n
    else:
        end = start + per
    return (start, end)


@always_inline
def _local_participant_index(thread_id: Int, mask: Int) -> Int:
    """Zero-based index of `thread_id` among the set bits of `mask`.

    Handles non-contiguous participant sets (e.g. HMX+DMA on aic200 where the
    participants are threads 8 and 9). Undefined if `thread_id` is not a
    participant.
    """
    var idx = 0
    for t in range(thread_id):
        if (mask & (1 << t)) != 0:
            idx += 1
    return idx


# ===-----------------------------------------------------------------------===#
# Barrier dispatchers (comptime-selected)
# ===-----------------------------------------------------------------------===#


@always_inline
def _call_intra_barrier[tag: StaticString](tid: Int):
    """Emit the intra-NSP barrier named by `tag`, or nothing for `TAG_NONE`."""

    comptime if tag == TAG_NONE:
        pass  # baseline: no barrier, race
    elif tag == TAG_HVX:
        sync_hvx_threads(UInt32(tid))
    elif tag == TAG_ALL:
        sync_threads(UInt32(tid))
    elif tag == TAG_SUBSET:
        sync_threads[SUBSET_MASK, name="_test_subset_1_2"](UInt32(tid))
    elif tag == TAG_HVX_HMX:
        sync_hvx_and_hmx_threads(UInt32(tid))
    elif tag == TAG_HVX_DMA:
        sync_hvx_and_dma_threads(UInt32(tid))
    elif tag == TAG_HMX_DMA:
        sync_hmx_and_dma_threads(UInt32(tid))
    else:
        comptime assert False, "unknown intra-NSP barrier tag"


@always_inline
def _call_cross_nsp_barrier[tag: StaticString](tid: Int, num_nsps: Int):
    """Emit the cross-NSP barrier named by `tag`, or nothing for `TAG_NONE`."""

    comptime if tag == TAG_NONE:
        pass
    elif tag == TAG_HVX:
        sync_hvx_threads_across_nsps(UInt32(tid), num_nsps)
    elif tag == TAG_ALL:
        sync_threads_across_nsps(UInt32(tid), num_nsps)
    elif tag == TAG_HVX_HMX:
        sync_hvx_and_hmx_threads_across_nsps(UInt32(tid), num_nsps)
    elif tag == TAG_HVX_DMA:
        sync_hvx_and_dma_threads_across_nsps(UInt32(tid), num_nsps)
    elif tag == TAG_HMX_DMA:
        sync_hmx_and_dma_threads_across_nsps(UInt32(tid), num_nsps)
    else:
        comptime assert False, "unknown cross-NSP barrier tag"


# ===-----------------------------------------------------------------------===#
# Kernel templates
# ===-----------------------------------------------------------------------===#
#
# Intra-NSP meta:
#   [0]  N
#   [1]  participant_mask
#   [2]  num_participants (== pop_count(participant_mask))
#   [3 .. 3 + NUM_THREADS_MAX)  partial_sums (Float32)
#
# Cross-NSP meta:
#   [0]  N
#   [1]  num_nsps
#   [2]  num_participants_per_nsp
#   [3]  participant_mask (per-NSP thread bitmap)
#   [4 .. 4 + NUM_NSPS_TEST * NUM_THREADS_MAX)  partial_sums (Float32)
#
# partial_sums is indexed by actual thread_id (and, for cross-NSP,
# nsp_id * NUM_THREADS_MAX + thread_id); non-participants leave their slots at
# 0 so `sum(all_slots)` is exactly the sum over participants.


@always_inline
def _intra_barrier_body[
    tag: StaticString
](
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    meta: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    """Common intra-NSP parallel-normalize; barrier selected by `tag`."""
    var thread_id = Int(npu_thread_idx())
    var n = Int(meta[0])
    var participant_mask = Int(meta[1])
    var num_participants = Int(meta[2])
    var partial_sums = (meta + 3).bitcast[Float32]()

    var local_idx = _local_participant_index(thread_id, participant_mask)
    var start: Int
    var end: Int
    start, end = _skewed_range(local_idx, n, num_participants)

    var my_sum: Float32 = 0.0
    for i in range(start, end):
        my_sum += input[i]
    partial_sums[thread_id] = my_sum

    _call_intra_barrier[tag](thread_id)

    var global_sum: Float32 = 0.0
    for t in range(NUM_THREADS_MAX):
        global_sum += partial_sums[t]

    for i in range(start, end):
        output[i] = input[i] / global_sum


@always_inline
def _cross_nsp_barrier_body[
    tag: StaticString
](
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    meta: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    """Common cross-NSP parallel-normalize; barrier selected by `tag`."""
    var thread_id = Int(npu_thread_idx())
    var nsp_id = Int(npu_core_idx())
    var n = Int(meta[0])
    var num_nsps = Int(meta[1])
    var num_participants_per_nsp = Int(meta[2])
    var participant_mask = Int(meta[3])
    var partial_sums = (meta + 4).bitcast[Float32]()

    var nsp_start: Int
    var nsp_end: Int
    nsp_start, nsp_end = _skewed_nsp_range(nsp_id, n, num_nsps)
    var nsp_local_n = nsp_end - nsp_start

    var local_idx = _local_participant_index(thread_id, participant_mask)
    var local_start: Int
    var local_end: Int
    local_start, local_end = _skewed_range(
        local_idx, nsp_local_n, num_participants_per_nsp
    )
    var start = nsp_start + local_start
    var end = nsp_start + local_end

    var my_sum: Float32 = 0.0
    for i in range(start, end):
        my_sum += input[i]
    partial_sums[nsp_id * NUM_THREADS_MAX + thread_id] = my_sum

    _call_cross_nsp_barrier[tag](thread_id, num_nsps)

    var global_sum: Float32 = 0.0
    for i in range(num_nsps * NUM_THREADS_MAX):
        global_sum += partial_sums[i]

    for i in range(start, end):
        output[i] = input[i] / global_sum


# ===-----------------------------------------------------------------------===#
# Concrete kernel entry points
# ===-----------------------------------------------------------------------===#
#
# `compile_info_npu[fn]` needs a concrete function reference, so each variant
# gets a trivial wrapper that pins the barrier tag.


def intra_no_sync_kernel(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    meta: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    _intra_barrier_body[TAG_NONE](input, output, meta)


def intra_hvx_sync_kernel(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    meta: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    _intra_barrier_body[TAG_HVX](input, output, meta)


def intra_all_sync_kernel(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    meta: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    _intra_barrier_body[TAG_ALL](input, output, meta)


def intra_subset_sync_kernel(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    meta: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    _intra_barrier_body[TAG_SUBSET](input, output, meta)


def intra_hvx_hmx_sync_kernel(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    meta: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    _intra_barrier_body[TAG_HVX_HMX](input, output, meta)


def intra_hvx_dma_sync_kernel(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    meta: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    _intra_barrier_body[TAG_HVX_DMA](input, output, meta)


def intra_hmx_dma_sync_kernel(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    meta: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    _intra_barrier_body[TAG_HMX_DMA](input, output, meta)


def cross_nsp_no_sync_kernel(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    meta: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    _cross_nsp_barrier_body[TAG_NONE](input, output, meta)


def cross_nsp_hvx_sync_kernel(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    meta: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    _cross_nsp_barrier_body[TAG_HVX](input, output, meta)


def cross_nsp_all_sync_kernel(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    meta: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    _cross_nsp_barrier_body[TAG_ALL](input, output, meta)


def cross_nsp_hvx_hmx_sync_kernel(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    meta: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    _cross_nsp_barrier_body[TAG_HVX_HMX](input, output, meta)


def cross_nsp_hvx_dma_sync_kernel(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    meta: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    _cross_nsp_barrier_body[TAG_HVX_DMA](input, output, meta)


def cross_nsp_hmx_dma_sync_kernel(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    meta: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    _cross_nsp_barrier_body[TAG_HMX_DMA](input, output, meta)


# ===-----------------------------------------------------------------------===#
# Run helpers
# ===-----------------------------------------------------------------------===#


def run_intra_kernel[
    kernel_fn: def(
        UnsafePointer[Float32, MutUnsafeAnyOrigin],
        UnsafePointer[Float32, MutUnsafeAnyOrigin],
        UnsafePointer[UInt32, MutUnsafeAnyOrigin],
    ) thin -> None,
    label: StaticString,
](vector_size: Int, participant_mask: Int, launch_mask: UInt32) raises -> Int:
    """Compile, launch, and verify an intra-NSP kernel.

    Args:
        vector_size: Number of Float32 elements.
        participant_mask: Bits set for each participating thread (goes to
            meta[1]); num_participants is written to meta[2] as its popcount.
        launch_mask: Thread bitmap for `qaic_execute_jit_function`.

    Returns:
        Verification error count (0 = pass).
    """
    print()
    print_separator()
    print(label)
    print_separator()

    var data_size = vector_size * 4
    var num_participants = pop_count(participant_mask)

    var meta_uint32_count = 3 + NUM_THREADS_MAX
    var meta_size = meta_uint32_count * 4

    print("  Compiling...")
    var elf = compile_info_npu[kernel_fn]()
    print("  ELF size:", elf.asm.byte_length(), "bytes")

    var ctx = DeviceContext(device_id=0, api="qaic")
    var jit_device = qaic_get_jit_device(ctx)
    var jit_func = qaic_load_jit_function(
        ctx,
        elf.function_name,
        elf.asm.unsafe_ptr().bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(elf.asm.byte_length()),
    )

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

    var input_host = alloc[Float32](vector_size)
    var expected_global_sum: Float32 = 0.0
    for i in range(vector_size):
        input_host[i] = Float32(i + 1)
        expected_global_sum += Float32(i + 1)

    var meta_host = alloc[UInt32](meta_uint32_count)
    meta_host[0] = UInt32(vector_size)
    meta_host[1] = UInt32(participant_mask)
    meta_host[2] = UInt32(num_participants)
    for i in range(NUM_THREADS_MAX):
        meta_host[3 + i] = UInt32(0)

    print("  Vector size:", vector_size)
    print(
        "  participant_mask:",
        participant_mask,
        " participants:",
        num_participants,
        " launch_mask:",
        launch_mask,
    )
    print("  Expected global sum:", expected_global_sum)

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

    var buffers = alloc[QAicBuf](3)
    buffers[0] = input_buf
    buffers[1] = output_buf
    buffers[2] = meta_buf

    print("  Executing kernel...")
    qaic_execute_jit_function(
        ctx, jit_func, buffers.as_unsafe_any_origin(), 3, 1, launch_mask
    )
    print("  Done.")

    var output_host = alloc[Float32](vector_size)
    qaic_copy_jit_buffer(
        ctx,
        output_buf,
        output_host.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(data_size),
        QAIC_COPY_FROM_DEVICE,
    )

    var errors = 0
    var output_sum: Float32 = 0.0
    for i in range(vector_size):
        var expected = Float32(i + 1) / expected_global_sum
        var actual = output_host[i]
        output_sum += actual
        if abs(actual - expected) > 1e-4:
            errors += 1

    print("  Sum of all output elements:", output_sum, "(expected 1.0)")
    if errors == 0:
        print("  PASS - all", vector_size, "elements correct.")
    else:
        print("  FAIL -", errors, "mismatches out of", vector_size, "elements.")

    input_host.unsafe_free()
    meta_host.unsafe_free()
    output_host.unsafe_free()
    buffers.unsafe_free()
    qaic_release_jit_buffer(ctx, input_buf)
    qaic_release_jit_buffer(ctx, output_buf)
    qaic_release_jit_buffer(ctx, meta_buf)
    qaic_release_jit_function(jit_func)

    return errors


def run_cross_nsp_kernel[
    kernel_fn: def(
        UnsafePointer[Float32, MutUnsafeAnyOrigin],
        UnsafePointer[Float32, MutUnsafeAnyOrigin],
        UnsafePointer[UInt32, MutUnsafeAnyOrigin],
    ) thin -> None,
    label: StaticString,
](
    vector_size: Int,
    num_nsps: Int,
    participant_mask: Int,
    launch_mask: UInt32,
) raises -> Int:
    """Cross-NSP variant of `run_intra_kernel`.

    Args:
        vector_size: Number of Float32 elements.
        num_nsps: Participating NSP cores.
        participant_mask: Per-NSP thread bitmap.
        launch_mask: Thread bitmap for `qaic_execute_jit_function`; usually
            equal to `participant_mask`.

    Returns:
        Verification error count (0 = pass).
    """
    print()
    print_separator()
    print(label)
    print_separator()

    var data_size = vector_size * 4
    var num_participants_per_nsp = pop_count(participant_mask)

    var partials_count = NUM_NSPS_TEST * NUM_THREADS_MAX
    var meta_uint32_count = 4 + partials_count
    var meta_size = meta_uint32_count * 4

    print("  Compiling...")
    var elf = compile_info_npu[kernel_fn]()
    print("  ELF size:", elf.asm.byte_length(), "bytes")

    var ctx = DeviceContext(device_id=0, api="qaic")
    var jit_device = qaic_get_jit_device(ctx)
    var jit_func = qaic_load_jit_function(
        ctx,
        elf.function_name,
        elf.asm.unsafe_ptr().bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(elf.asm.byte_length()),
    )

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

    var input_host = alloc[Float32](vector_size)
    var expected_global_sum: Float32 = 0.0
    for i in range(vector_size):
        input_host[i] = Float32(i + 1)
        expected_global_sum += Float32(i + 1)

    var meta_host = alloc[UInt32](meta_uint32_count)
    meta_host[0] = UInt32(vector_size)
    meta_host[1] = UInt32(num_nsps)
    meta_host[2] = UInt32(num_participants_per_nsp)
    meta_host[3] = UInt32(participant_mask)
    for i in range(partials_count):
        meta_host[4 + i] = UInt32(0)

    print("  Vector size:", vector_size)
    print(
        "  NSPs:",
        num_nsps,
        " participant_mask:",
        participant_mask,
        " participants/nsp:",
        num_participants_per_nsp,
        " launch_mask:",
        launch_mask,
    )
    print("  Expected global sum:", expected_global_sum)

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

    var buffers = alloc[QAicBuf](3)
    buffers[0] = input_buf
    buffers[1] = output_buf
    buffers[2] = meta_buf

    print("  Executing kernel...")
    qaic_execute_jit_function(
        ctx,
        jit_func,
        buffers.as_unsafe_any_origin(),
        3,
        UInt32(num_nsps),
        launch_mask,
    )
    print("  Done.")

    var output_host = alloc[Float32](vector_size)
    qaic_copy_jit_buffer(
        ctx,
        output_buf,
        output_host.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(data_size),
        QAIC_COPY_FROM_DEVICE,
    )

    var errors = 0
    var output_sum: Float32 = 0.0
    for i in range(vector_size):
        var expected = Float32(i + 1) / expected_global_sum
        var actual = output_host[i]
        output_sum += actual
        if abs(actual - expected) > 1e-4:
            errors += 1

    print("  Sum of all output elements:", output_sum, "(expected 1.0)")
    if errors == 0:
        print("  PASS - all", vector_size, "elements correct.")
    else:
        print("  FAIL -", errors, "mismatches out of", vector_size, "elements.")

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
# Tests
# ===-----------------------------------------------------------------------===#


def test_sync_hvx_threads() raises:
    print_test_header("sync_hvx_threads (HVX threads only)")
    var vsz: Int = 65536
    var mask = hvx_mask()
    var errs_sync = run_intra_kernel[
        intra_hvx_sync_kernel, "sync_hvx_threads WITH barrier"
    ](vsz, mask, UInt32(mask))
    var errs_nosync = run_intra_kernel[
        intra_no_sync_kernel, "sync_hvx_threads WITHOUT barrier (racy)"
    ](vsz, mask, UInt32(mask))
    assert_equal(errs_sync, 0)
    assert_not_equal(errs_nosync, 0)


def test_sync_threads() raises:
    print_test_header("sync_threads (all-thread launch)")
    var vsz: Int = 65536
    var mask = thread_mask()
    var errs_sync = run_intra_kernel[
        intra_all_sync_kernel, "sync_threads WITH barrier"
    ](vsz, mask, UInt32(mask))
    var errs_nosync = run_intra_kernel[
        intra_no_sync_kernel, "sync_threads WITHOUT barrier (racy)"
    ](vsz, mask, UInt32(mask))
    assert_equal(errs_sync, 0)
    assert_not_equal(errs_nosync, 0)


def test_sync_threads_in_mask() raises:
    print_test_header("sync_threads_in_mask (subset: threads 1 and 2)")
    var vsz: Int = 65536
    var errs_sync = run_intra_kernel[
        intra_subset_sync_kernel, "sync_threads_in_mask WITH barrier"
    ](vsz, SUBSET_MASK, UInt32(SUBSET_MASK))
    var errs_nosync = run_intra_kernel[
        intra_no_sync_kernel, "sync_threads_in_mask WITHOUT barrier (racy)"
    ](vsz, SUBSET_MASK, UInt32(SUBSET_MASK))
    assert_equal(errs_sync, 0)
    assert_not_equal(errs_nosync, 0)


def test_sync_hvx_and_hmx_threads() raises:
    """AIC200-only: on aic100 comptimees `sync_threads`."""
    comptime if is_aic100():
        print(
            "[skipped] sync_hvx_and_hmx_threads: comptimees sync_threads on"
            " this target"
        )
        return
    print_test_header("sync_hvx_and_hmx_threads (HVX + HMX, aic200 only)")
    var vsz: Int = 65536
    var mask = hvx_and_hmx_mask()
    var errs_sync = run_intra_kernel[
        intra_hvx_hmx_sync_kernel, "sync_hvx_and_hmx_threads WITH barrier"
    ](vsz, mask, UInt32(mask))
    var errs_nosync = run_intra_kernel[
        intra_no_sync_kernel, "sync_hvx_and_hmx_threads WITHOUT barrier (racy)"
    ](vsz, mask, UInt32(mask))
    assert_equal(errs_sync, 0)
    assert_not_equal(errs_nosync, 0)


def test_sync_hvx_and_dma_threads() raises:
    """AIC200-only: on aic100 comptimees `sync_threads`."""
    comptime if is_aic100():
        print(
            "[skipped] sync_hvx_and_dma_threads: comptimees sync_threads on"
            " this target"
        )
        return
    print_test_header("sync_hvx_and_dma_threads (HVX + DMA, aic200 only)")
    var vsz: Int = 65536
    var mask = hvx_and_dma_mask()
    var errs_sync = run_intra_kernel[
        intra_hvx_dma_sync_kernel, "sync_hvx_and_dma_threads WITH barrier"
    ](vsz, mask, UInt32(mask))
    var errs_nosync = run_intra_kernel[
        intra_no_sync_kernel, "sync_hvx_and_dma_threads WITHOUT barrier (racy)"
    ](vsz, mask, UInt32(mask))
    assert_equal(errs_sync, 0)
    assert_not_equal(errs_nosync, 0)


def test_sync_hmx_and_dma_threads() raises:
    """AIC200-only: on aic100 this is a single-participant `syncht`."""
    comptime if is_aic100():
        print(
            "[skipped] sync_hmx_and_dma_threads: single-participant on this"
            " target"
        )
        return
    print_test_header("sync_hmx_and_dma_threads (HMX + DMA, aic200 only)")
    var vsz: Int = 65536
    var mask = hmx_and_dma_mask()
    var errs_sync = run_intra_kernel[
        intra_hmx_dma_sync_kernel, "sync_hmx_and_dma_threads WITH barrier"
    ](vsz, mask, UInt32(mask))
    var errs_nosync = run_intra_kernel[
        intra_no_sync_kernel, "sync_hmx_and_dma_threads WITHOUT barrier (racy)"
    ](vsz, mask, UInt32(mask))
    assert_equal(errs_sync, 0)
    assert_not_equal(errs_nosync, 0)


def test_sync_hvx_threads_across_nsps() raises:
    print_test_header("sync_hvx_threads_across_nsps (cross-NSP HVX barrier)")
    var vsz: Int = 65536
    var mask = hvx_mask()
    var errs_sync = run_cross_nsp_kernel[
        cross_nsp_hvx_sync_kernel, "sync_hvx_threads_across_nsps WITH barrier"
    ](vsz, NUM_NSPS_TEST, mask, UInt32(mask))
    var errs_nosync = run_cross_nsp_kernel[
        cross_nsp_no_sync_kernel,
        "sync_hvx_threads_across_nsps WITHOUT barrier (racy)",
    ](vsz, NUM_NSPS_TEST, mask, UInt32(mask))
    assert_equal(errs_sync, 0)
    assert_not_equal(errs_nosync, 0)


def test_sync_threads_across_nsps() raises:
    """AIC200-only gate for parity with the other cross-NSP variants."""
    comptime if is_aic100():
        print(
            "[skipped] sync_threads_across_nsps: gated to aic200 for parity"
            " with the AIC200-only cross-NSP variants"
        )
        return
    print_test_header(
        "sync_threads_across_nsps (all-thread cross-NSP, aic200 only)"
    )
    var vsz: Int = 65536
    var mask = thread_mask()
    var errs_sync = run_cross_nsp_kernel[
        cross_nsp_all_sync_kernel, "sync_threads_across_nsps WITH barrier"
    ](vsz, NUM_NSPS_TEST, mask, UInt32(mask))
    var errs_nosync = run_cross_nsp_kernel[
        cross_nsp_no_sync_kernel,
        "sync_threads_across_nsps WITHOUT barrier (racy)",
    ](vsz, NUM_NSPS_TEST, mask, UInt32(mask))
    assert_equal(errs_sync, 0)
    assert_not_equal(errs_nosync, 0)


def test_sync_hvx_and_hmx_threads_across_nsps() raises:
    """AIC200-only: on aic100 comptimees `sync_threads_across_nsps`."""
    comptime if is_aic100():
        print(
            "[skipped] sync_hvx_and_hmx_threads_across_nsps: comptimees"
            " sync_threads_across_nsps on this target"
        )
        return
    print_test_header(
        "sync_hvx_and_hmx_threads_across_nsps (cross-NSP HVX+HMX, aic200 only)"
    )
    var vsz: Int = 65536
    var mask = hvx_and_hmx_mask()
    var errs_sync = run_cross_nsp_kernel[
        cross_nsp_hvx_hmx_sync_kernel,
        "sync_hvx_and_hmx_threads_across_nsps WITH barrier",
    ](vsz, NUM_NSPS_TEST, mask, UInt32(mask))
    var errs_nosync = run_cross_nsp_kernel[
        cross_nsp_no_sync_kernel,
        "sync_hvx_and_hmx_threads_across_nsps WITHOUT barrier (racy)",
    ](vsz, NUM_NSPS_TEST, mask, UInt32(mask))
    assert_equal(errs_sync, 0)
    assert_not_equal(errs_nosync, 0)


def test_sync_hvx_and_dma_threads_across_nsps() raises:
    """AIC200-only: on aic100 comptimees `sync_threads_across_nsps`."""
    comptime if is_aic100():
        print(
            "[skipped] sync_hvx_and_dma_threads_across_nsps: comptimees"
            " sync_threads_across_nsps on this target"
        )
        return
    print_test_header(
        "sync_hvx_and_dma_threads_across_nsps (cross-NSP HVX+DMA, aic200 only)"
    )
    var vsz: Int = 65536
    var mask = hvx_and_dma_mask()
    var errs_sync = run_cross_nsp_kernel[
        cross_nsp_hvx_dma_sync_kernel,
        "sync_hvx_and_dma_threads_across_nsps WITH barrier",
    ](vsz, NUM_NSPS_TEST, mask, UInt32(mask))
    var errs_nosync = run_cross_nsp_kernel[
        cross_nsp_no_sync_kernel,
        "sync_hvx_and_dma_threads_across_nsps WITHOUT barrier (racy)",
    ](vsz, NUM_NSPS_TEST, mask, UInt32(mask))
    assert_equal(errs_sync, 0)
    assert_not_equal(errs_nosync, 0)


def test_sync_hmx_and_dma_threads_across_nsps() raises:
    """AIC200-only: on aic100 the per-NSP participant set is single-thread."""
    comptime if is_aic100():
        print(
            "[skipped] sync_hmx_and_dma_threads_across_nsps:"
            " single-participant per NSP on this target"
        )
        return
    print_test_header(
        "sync_hmx_and_dma_threads_across_nsps (cross-NSP HMX+DMA, aic200 only)"
    )
    var vsz: Int = 65536
    var mask = hmx_and_dma_mask()
    var errs_sync = run_cross_nsp_kernel[
        cross_nsp_hmx_dma_sync_kernel,
        "sync_hmx_and_dma_threads_across_nsps WITH barrier",
    ](vsz, NUM_NSPS_TEST, mask, UInt32(mask))
    var errs_nosync = run_cross_nsp_kernel[
        cross_nsp_no_sync_kernel,
        "sync_hmx_and_dma_threads_across_nsps WITHOUT barrier (racy)",
    ](vsz, NUM_NSPS_TEST, mask, UInt32(mask))
    assert_equal(errs_sync, 0)
    assert_not_equal(errs_nosync, 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
