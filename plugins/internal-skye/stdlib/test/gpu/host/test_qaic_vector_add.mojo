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
"""Test single-threaded QAIC vector addition kernel.

This test demonstrates the basic workflow for QAIC kernel execution:
1. Compiling a Mojo kernel to Hexagon ELF
2. Loading it onto a QAIC device
3. Allocating device buffers
4. Copying input data to device
5. Executing the kernel
6. Copying results back and verifying

The vector_add_kernel computes: output[i] = a[i] + b[i]

This is the simplest QAIC kernel example, using a single HVX thread to
avoid division operations (which require __hexagon_divsi3, undefined in
JIT context).

To run (requires QAIC hardware):
    mojo test_qaic_vector_add.mojo
"""

from std.testing import TestSuite, assert_equal
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


# ===-----------------------------------------------------------------------===#
# Kernel: Vector Add (Single-threaded)
# ===-----------------------------------------------------------------------===#


def vector_add_kernel(
    input_a: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    input_b: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    size_ptr: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    """Simple vector addition kernel: output[i] = a[i] + b[i].

    This kernel runs on a single HVX thread. Buffer pointers are
    automatically extracted from the QAIC pointer_array by make_npu_wrapper.

    Args:
        input_a: First input vector (float32).
        input_b: Second input vector (float32).
        output: Output vector (float32).
        size_ptr: Pointer to vector size (uint32).
    """
    var size = Int(size_ptr[])
    for i in range(size):
        output[i] = input_a[i] + input_b[i]


# ===-----------------------------------------------------------------------===#
# Test
# ===-----------------------------------------------------------------------===#


def test_qaic_vector_add() raises:
    print_test_header("QAIC Vector Add Test (Single-threaded)")
    print()

    # Compile the kernel
    print("Step 1: Compiling vector_add kernel to Hexagon ELF...")
    var elf = compile_info_npu[vector_add_kernel]()

    print("  ELF size:", elf.asm.byte_length(), "bytes")
    print("  Function name:", elf.function_name)
    print()

    # Create DeviceContext
    print("Step 2: Creating QAIC DeviceContext...")
    var ctx = DeviceContext(device_id=0, api="qaic")
    print("  DeviceContext created")
    print("  API:", ctx.api())
    print("  Device ID:", ctx.id())
    print()

    # Get JIT device
    print("Step 3: Getting internal JIT device...")
    var jit_device = qaic_get_jit_device(ctx)
    print("  JIT device obtained")
    print()

    # Load kernel
    print("Step 4: Loading JIT function...")
    var jit_func = qaic_load_jit_function(
        ctx,
        elf.function_name,
        elf.asm.unsafe_ptr().bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(elf.asm.byte_length()),
    )
    print("  JIT function loaded successfully!")
    print()

    # Test parameters
    var vector_size: Int = 256
    var data_size = vector_size * 4  # float32 = 4 bytes
    var size_buffer_size: Int = 4  # uint32 = 4 bytes

    print("Step 5: Allocating device buffers...")
    print("  Vector size:", vector_size, "elements")
    print("  Data size:", data_size, "bytes")

    # Allocate device buffers (128-byte alignment for HVX)
    var alignment: UInt32 = 128
    var cu_bitmap: UInt64 = 0xFFFFFFFFFFFFFFFF  # All compute units

    var input_a_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        UInt64(data_size),
        alignment,
        cu_bitmap,
    )
    print("  Allocated input_a buffer")

    var input_b_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        UInt64(data_size),
        alignment,
        cu_bitmap,
    )
    print("  Allocated input_b buffer")

    var output_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        UInt64(data_size),
        alignment,
        cu_bitmap,
    )
    print("  Allocated output buffer")

    var size_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        UInt64(size_buffer_size),
        alignment,
        cu_bitmap,
    )
    print("  Allocated size buffer")
    print()

    # Prepare input data
    print("Step 6: Preparing and copying input data...")
    var input_a_host = alloc[Float32](vector_size)
    var input_b_host = alloc[Float32](vector_size)
    var size_host = alloc[UInt32](1)

    # Initialize: a[i] = i, b[i] = 2*i  -> output[i] = 3*i
    for i in range(vector_size):
        input_a_host[i] = Float32(i)
        input_b_host[i] = Float32(2 * i)

    size_host[0] = UInt32(vector_size)
    print("  Input: a[i] = i, b[i] = 2*i")
    print("  Expected: output[i] = 3*i")

    # Copy to device
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
    print("  Copied input data to device")

    # Free host input buffers
    input_a_host.unsafe_free()
    input_b_host.unsafe_free()
    size_host.unsafe_free()
    print()

    # Execute kernel
    print("Step 7: Executing kernel...")
    var buffers = alloc[QAicBuf](4)
    buffers[0] = input_a_buf
    buffers[1] = input_b_buf
    buffers[2] = output_buf
    buffers[3] = size_buf

    # Execute with 1 core and 1 HVX thread (0x01 = thread 0 only)
    qaic_execute_jit_function(
        ctx, jit_func, buffers.as_unsafe_any_origin(), 4, 1, 0x01
    )
    print("  Kernel execution completed!")
    print()

    # Copy results back
    print("Step 8: Copying results from device...")
    var output_ptr = alloc[Float32](vector_size)
    qaic_copy_jit_buffer(
        ctx,
        output_buf,
        output_ptr.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(data_size),
        QAIC_COPY_FROM_DEVICE,
    )
    print("  Copied output data from device")

    # Verify results
    print("Step 9: Verifying results...")
    var errors = 0
    var max_errors_to_print = 5

    for i in range(vector_size):
        var expected = Float32(3 * i)  # a[i] + b[i] = i + 2*i = 3*i
        var actual = output_ptr[i]

        if abs(actual - expected) > 0.0001:
            errors += 1
            if errors <= max_errors_to_print:
                print(
                    "  ERROR at index", i, ": expected", expected, "got", actual
                )

    if errors == 0:
        print("  SUCCESS: All", vector_size, "elements verified correctly!")
        print()
        print("  Sample outputs:")
        print("    output[0] =", output_ptr[0], "(expected 0)")
        print("    output[1] =", output_ptr[1], "(expected 3)")
        print("    output[10] =", output_ptr[10], "(expected 30)")
        print("    output[255] =", output_ptr[255], "(expected 765)")
    else:
        print("  FAILED:", errors, "errors out of", vector_size, "elements")
        if errors > max_errors_to_print:
            print("  (only first", max_errors_to_print, "errors shown)")

    print()

    # Cleanup
    print("Step 10: Cleaning up resources...")
    output_ptr.unsafe_free()
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
