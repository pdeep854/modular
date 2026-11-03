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
"""Test QAIC vector scaling kernel.

This test demonstrates a simple element-wise scaling operation:
    output[i] = input[i] * scale

The kernel shows how to pass a scalar parameter (the scale factor) as a
separate buffer rather than as part of the kernel signature. This pattern
is common in QAIC JIT kernels where all inputs/outputs are passed via
the pointer array.

Buffer layout:
    [0]: input  - Input vector (float32)
    [1]: output - Output vector (float32)
    [2]: scale  - Scale factor (float32, single value)
    [3]: size   - Vector size (uint32)

To run (requires QAIC hardware):
    mojo test_qaic_vector_scale.mojo
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
# Kernel: Vector Scale
# ===-----------------------------------------------------------------------===#


def vector_scale_kernel(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    scale_ptr: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    size_ptr: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    """Scale a vector by a constant: output[i] = input[i] * scale.

    Scalar parameters are passed as 1-element buffers.

    Args:
        input: Input vector (float32).
        output: Output vector (float32).
        scale_ptr: Scale factor (float32, single value).
        size_ptr: Vector size (uint32).
    """
    var scale = scale_ptr[]
    var size = Int(size_ptr[])

    for i in range(size):
        output[i] = input[i] * scale


# ===-----------------------------------------------------------------------===#
# Test
# ===-----------------------------------------------------------------------===#


def test_qaic_vector_scale() raises:
    print_test_header("QAIC Vector Scale Test")
    print()

    # Test parameters
    var vector_size: Int = 128
    var data_size = vector_size * 4  # float32 = 4 bytes
    var scale_size: Int = 4  # single float32
    var size_buffer_size: Int = 4  # uint32

    var scale_value: Float32 = 2.5

    print("Configuration:")
    print("  Vector size:", vector_size, "elements")
    print("  Scale factor:", scale_value)
    print()

    # Compile the kernel
    print("Step 1: Compiling vector_scale kernel...")
    var elf = compile_info_npu[vector_scale_kernel]()
    print("  ELF size:", elf.asm.byte_length(), "bytes")
    print()

    # Create context and get JIT device
    print("Step 2: Creating QAIC DeviceContext...")
    var ctx = DeviceContext(device_id=0, api="qaic")
    var jit_device = qaic_get_jit_device(ctx)
    print("  Context and JIT device ready")
    print()

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
    var scale_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        UInt64(scale_size),
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
    print("  Allocated input, output, scale, size buffers")
    print()

    # Prepare input data
    print("Step 5: Preparing input data...")
    var input_host = alloc[Float32](vector_size)
    var scale_host = alloc[Float32](1)
    var size_host = alloc[UInt32](1)

    # input[i] = i + 1  (1, 2, 3, ...)
    for i in range(vector_size):
        input_host[i] = Float32(i + 1)

    scale_host[0] = scale_value
    size_host[0] = UInt32(vector_size)

    print("  Input: input[i] = i + 1")
    print("  Scale:", scale_value)
    print("  Expected: output[i] = (i + 1) *", scale_value)
    print()

    # Copy to device
    print("Step 6: Copying data to device...")
    qaic_copy_jit_buffer(
        ctx,
        input_buf,
        input_host.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(data_size),
        QAIC_COPY_TO_DEVICE,
    )
    qaic_copy_jit_buffer(
        ctx,
        scale_buf,
        scale_host.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(scale_size),
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

    # Execute kernel
    print("Step 7: Executing kernel...")
    var buffers = alloc[QAicBuf](4)
    buffers[0] = input_buf
    buffers[1] = output_buf
    buffers[2] = scale_buf
    buffers[3] = size_buf

    qaic_execute_jit_function(
        ctx, jit_func, buffers.as_unsafe_any_origin(), 4, 1, 0x01
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

    for i in range(vector_size):
        var expected = Float32(i + 1) * scale_value
        var actual = output_host[i]
        if abs(actual - expected) > 0.0001:
            errors += 1
            if errors <= 5:
                print(
                    "  ERROR at index", i, ": expected", expected, "got", actual
                )

    if errors == 0:
        print("  SUCCESS: All", vector_size, "elements verified!")
        print()
        print("  Sample outputs:")
        print(
            "    output[0] =", output_host[0], "(expected", 1 * scale_value, ")"
        )
        print(
            "    output[9] =",
            output_host[9],
            "(expected",
            10 * scale_value,
            ")",
        )
        print(
            "    output[99] =",
            output_host[99],
            "(expected",
            100 * scale_value,
            ")",
        )
    else:
        print("  FAILED:", errors, "errors")

    # Cleanup
    print()
    print("Step 10: Cleaning up...")
    input_host.unsafe_free()
    scale_host.unsafe_free()
    size_host.unsafe_free()
    output_host.unsafe_free()
    buffers.unsafe_free()
    qaic_release_jit_buffer(ctx, input_buf)
    qaic_release_jit_buffer(ctx, output_buf)
    qaic_release_jit_buffer(ctx, scale_buf)
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
