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
"""Test multi-threaded QAIC vector addition kernel.

This test demonstrates how to parallelize work across HVX threads on QAIC
hardware using npu_work_range(), which automatically distributes work across
threads via qshim internally.

Thread distribution example (4 threads, 256 elements):
    Thread 0: elements [0, 64)
    Thread 1: elements [64, 128)
    Thread 2: elements [128, 192)
    Thread 3: elements [192, 256)

The kernel uses npu_work_range() to query thread_id and num_threads via
qshim, eliminating the need to access AicJitEntryPointConfig directly.

To run (requires QAIC hardware):
    mojo test_qaic_vector_add_multithread.mojo
"""

from std.testing import TestSuite, assert_equal
from common import print_separator, print_test_header

from std.memory import UnsafePointer, alloc

from max.gpu.host import DeviceContext, DeviceAttribute
from std._plugin.qaic.host.target import (
    compile_info_npu,
    npu_work_range,
)
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


# ===-----------------------------------------------------------------------===#
# Configuration
# ===-----------------------------------------------------------------------===#

# Number of HVX threads - must match thread_mask in execute call
comptime NUM_THREADS: Int = 4


# ===-----------------------------------------------------------------------===#
# Kernel: Multi-threaded Vector Add
# ===-----------------------------------------------------------------------===#


def vector_add_multithread_kernel(
    input_a: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    input_b: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    size_ptr: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    """Multi-threaded vector addition using npu_work_range.

    Each thread automatically gets its share of the work via npu_work_range(),
    which uses qshim internally to query the thread_id and num_threads.

    Args:
        input_a: First input vector (float32).
        input_b: Second input vector (float32).
        output: Output vector (float32).
        size_ptr: Pointer to vector size (uint32).
    """
    var size = Int(size_ptr[])
    var (start, end) = npu_work_range(size)

    for i in range(start, end):
        output[i] = input_a[i] + input_b[i]


# ===-----------------------------------------------------------------------===#
# Test
# ===-----------------------------------------------------------------------===#


def test_qaic_vector_add_multithread() raises:
    print_test_header("QAIC Multi-threaded Vector Add Test")
    print()

    # Use a size that divides evenly by NUM_THREADS (4)
    # 256 / 4 = 64 elements per thread
    var vector_size: Int = 256
    var data_size = vector_size * 4  # float32 = 4 bytes
    var size_buffer_size: Int = 4

    print("Configuration:")
    print("  Vector size:", vector_size, "elements")
    print("  Thread count:", NUM_THREADS)
    print("  Elements per thread:", vector_size // NUM_THREADS)
    print()

    # Compile the multi-threaded kernel
    print("Step 1: Compiling multi-threaded kernel...")
    var elf = compile_info_npu[vector_add_multithread_kernel]()
    print("  ELF size:", elf.asm.byte_length(), "bytes")
    print()

    # Create context and get JIT device
    print("Step 2: Creating QAIC DeviceContext...")
    var ctx = DeviceContext(device_id=0, api="qaic")
    var jit_device = qaic_get_jit_device(ctx)
    print("  Context and JIT device ready")
    print()
    var max_threads = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    print("Number of NSP available: ", max_threads)

    # Load kernel
    print("Step 3: Loading JIT function...")
    var jit_func = qaic_load_jit_function(
        ctx,
        elf.function_name,
        elf.asm.unsafe_ptr().bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(elf.asm.byte_length()),
    )
    print("  Loaded successfully")
    print()

    # Allocate buffers
    print("Step 4: Allocating buffers...")
    var alignment: UInt32 = 128
    var cu_bitmap: UInt64 = 0xFFFFFFFFFFFFFFFF

    var input_a_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        UInt64(data_size),
        alignment,
        cu_bitmap,
    )
    var input_b_buf = qaic_allocate_jit_buffer(
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
    var size_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        UInt64(size_buffer_size),
        alignment,
        cu_bitmap,
    )
    print("  Allocated input_a, input_b, output, size buffers")
    print()

    # Prepare input data
    print("Step 5: Preparing input data...")
    var input_a_host = alloc[Float32](vector_size)
    var input_b_host = alloc[Float32](vector_size)
    var size_host = alloc[UInt32](1)

    # a[i] = i, b[i] = 100 + i  -> output[i] = 100 + 2*i
    for i in range(vector_size):
        input_a_host[i] = Float32(i)
        input_b_host[i] = Float32(100 + i)

    size_host[0] = UInt32(vector_size)
    print("  Input: a[i] = i, b[i] = 100 + i")
    print("  Expected: output[i] = 100 + 2*i")
    print()

    # Copy to device
    print("Step 6: Copying data to device...")
    qaic_copy_jit_buffer(
        ctx,
        input_a_buf,
        input_a_host.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(data_size),
        QAIC_COPY_TO_DEVICE,
    )
    qaic_copy_jit_buffer(
        ctx,
        input_b_buf,
        input_b_host.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(data_size),
        QAIC_COPY_TO_DEVICE,
    )
    qaic_copy_jit_buffer(
        ctx,
        size_buf,
        size_host.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(size_buffer_size),
        QAIC_COPY_TO_DEVICE,
    )
    print("  Copied to device")
    print()

    # Execute kernel with multiple threads
    print("Step 7: Executing with", NUM_THREADS, "threads...")
    var buffers = alloc[QAicBuf](4)
    buffers[0] = input_a_buf
    buffers[1] = input_b_buf
    buffers[2] = output_buf
    buffers[3] = size_buf

    # thread_mask = 0x0F = binary 1111 = threads 0,1,2,3 enabled
    var thread_mask: UInt32 = UInt32(
        (1 << NUM_THREADS) - 1
    )  # 0x0F for 4 threads
    print("  Thread mask: 0x" + String(Int(thread_mask)))

    qaic_execute_jit_function(
        ctx, jit_func, buffers.as_unsafe_any_origin(), 4, 1, thread_mask
    )
    print("  Execution completed")
    print()

    # Copy results back
    print("Step 8: Copying results from device...")
    var output_host = alloc[Float32](vector_size)
    qaic_copy_jit_buffer(
        ctx,
        output_buf,
        output_host.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(data_size),
        QAIC_COPY_FROM_DEVICE,
    )
    print("  Copied output from device")
    print()

    # Verify results
    print("Step 9: Verifying results...")
    var errors = 0
    var chunk_size = (vector_size + NUM_THREADS - 1) // NUM_THREADS

    for i in range(vector_size):
        var expected = Float32(100 + 2 * i)
        var actual = output_host[i]
        if abs(actual - expected) > 0.0001:
            errors += 1
            if errors <= 5:
                var thread = i // chunk_size
                print(
                    "  ERROR at index",
                    i,
                    "(thread",
                    thread,
                    "): expected",
                    expected,
                    "got",
                    actual,
                )

    if errors == 0:
        print("  SUCCESS: All", vector_size, "elements verified!")
        print()
        print("  Sample outputs by thread chunk:")
        for t in range(NUM_THREADS):
            var start = t * chunk_size
            var mid = start + chunk_size // 2
            print(
                "    Thread",
                t,
                "[",
                start,
                "-",
                start + chunk_size - 1,
                "]: output[",
                mid,
                "] =",
                output_host[mid],
            )
    else:
        print("  FAILED:", errors, "errors")

    # Cleanup
    print()
    print("Step 10: Cleaning up...")
    input_a_host.unsafe_free()
    input_b_host.unsafe_free()
    size_host.unsafe_free()
    output_host.unsafe_free()
    buffers.unsafe_free()
    qaic_release_jit_buffer(ctx, input_a_buf)
    qaic_release_jit_buffer(ctx, input_b_buf)
    qaic_release_jit_buffer(ctx, output_buf)
    qaic_release_jit_buffer(ctx, size_buf)
    qaic_release_jit_function(jit_func)
    print("  Done")

    print()
    print_separator()
    if errors == 0:
        print("TEST PASSED!")
    else:
        print("TEST FAILED!")
    print_separator()

    assert_equal(errors, 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
