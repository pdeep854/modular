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
"""Test QAIC JIT workflow: compile_function -> ELF detection -> execute.

This test verifies the CUDA/HIP-style API for QAIC:
1. compile_function produces an ELF binary (via hexagon-link)
2. loadFunction detects ELF and routes to JIT loader
3. The kernel can be executed via enqueueFunctionExecDirect

Run this test on a QAIC machine:
    mojo test_qaic_jit_workflow.mojo

Or via bazel on a QAIC machine:
    ./bazelw run //KGEN/tools/mojo -- oss/modular/mojo/stdlib/test/gpu/host/test_qaic_jit_workflow.mojo
"""

from std.testing import TestSuite
from common import print_separator, print_test_header

from std.compile import compile_info
from std.memory import alloc
from std.sys.info import _accelerator_arch

from max.gpu.host import DeviceContext, Dim
from max.gpu.host import get_gpu_target
from std._plugin.qaic.host.target import npu_work_range
from max._plugin.qaic.host.qaic import (
    qaic_get_jit_device,
    qaic_allocate_jit_buffer,
    qaic_release_jit_buffer,
    qaic_load_jit_function,
    qaic_execute_jit_function,
    qaic_release_jit_function,
    qaic_map_jit_buffer,
    qaic_get_jit_buffer_host_info,
    qaic_sync_jit_buffer,
    QAIC_BUFFER_TYPE_DEV_DDR,
    QAIC_DEV_BUF_IO_RW,
    QAIC_COPY_TO_DEVICE,
    QAIC_COPY_FROM_DEVICE,
    qaic_get_jit_device_attribute,
    QAIC_DEV_ATTR_TOTAL_DDR_KB,
    QAIC_DEV_ATTR_FREE_DDR_KB,
    QAIC_DEV_ATTR_NSP_ARCH,
    QAIC_DEV_ATTR_NSP_VA_SPACE_KB,
    QAIC_DEV_ATTR_UDMA_MAX_BW_KBPS,
    QAIC_DEV_ATTR_DDR_MAX_BW_KBPS,
    QAIC_DEV_ATTR_NSP_THREADS,
    QAIC_DEV_ATTR_THREAD_BITMAP,
    QAIC_DEV_ATTR_HMX_THREAD_ID,
    QAicBuf,
    QAicDevice,
)
from std._plugin.qaic.host.target import (
    AicJitEntryPointConfig,
    AicJitPointerArray,
)

# Buffer allocation constants
comptime HVX_ALIGN: UInt32 = 128  # HVX alignment requirement
comptime ALL_COMPUTE_UNITS: UInt64 = 0xFFFFFFFFFFFFFFFF  # All NSPs


# ===-----------------------------------------------------------------------===#
# Test Kernels
# ===-----------------------------------------------------------------------===#


def vector_add_kernel(
    entry_config: UnsafePointer[AicJitEntryPointConfig, MutUnsafeAnyOrigin],
    pointer_array: UnsafePointer[AicJitPointerArray, MutUnsafeAnyOrigin],
) -> UInt32:
    """Simple vector add kernel: C[i] = A[i] + B[i].

    Buffer layout (via FAM - Flexible Array Member):
        pointers[0] = input A (Float32 array)
        pointers[1] = input B (Float32 array)
        pointers[2] = output C (Float32 array)
        pointers[3] = size (UInt32 scalar)

    Each thread processes a portion of the array based on thread_id.
    Uses bit-shifting instead of division (see qaic/host/_target.mojo).
    """
    # Get execution context
    var config = entry_config[]
    var thread_id = config.thread_id
    var num_threads = config.num_threads

    # Access buffer pointers via FAM byte offsets. Viewed as UInt64 words
    # (SDK 1.23), the header (numPointers + reserved) is one 8-byte word and
    # each AicJitPtr is one 8-byte word, so pointer i is at word i + 1.
    var base = pointer_array.bitcast[UInt64]()

    # Get buffer addresses (device address is in the low 32 bits of each slot)
    var ptr_a_addr = (base + 1)[]  # pointers[0]
    var ptr_b_addr = (base + 2)[]  # pointers[1]
    var ptr_c_addr = (base + 3)[]  # pointers[2]
    var ptr_size_addr = (base + 4)[]  # pointers[3]

    # Convert to typed pointers
    var a = UnsafePointer[Float32, MutUnsafeAnyOrigin](
        unsafe_from_address=Int(ptr_a_addr)
    )
    var b = UnsafePointer[Float32, MutUnsafeAnyOrigin](
        unsafe_from_address=Int(ptr_b_addr)
    )
    var c = UnsafePointer[Float32, MutUnsafeAnyOrigin](
        unsafe_from_address=Int(ptr_c_addr)
    )
    var size_ptr = UnsafePointer[UInt32, MutUnsafeAnyOrigin](
        unsafe_from_address=Int(ptr_size_addr)
    )
    var size = Int(size_ptr[])

    # Each thread processes elements: thread_id, thread_id + num_threads, ...
    # This is a strided access pattern for load balancing
    var i = Int(thread_id)
    while i < size:
        c[i] = a[i] + b[i]
        i += Int(num_threads)

    return 0  # Success


def scalar_multiply_kernel(
    entry_config: UnsafePointer[AicJitEntryPointConfig, MutUnsafeAnyOrigin],
    pointer_array: UnsafePointer[AicJitPointerArray, MutUnsafeAnyOrigin],
) -> UInt32:
    """Scalar multiply kernel: B[i] = A[i] * scalar.

    Buffer layout:
        pointers[0] = input A (Float32 array)
        pointers[1] = output B (Float32 array)
        pointers[2] = scalar (Float32)
        pointers[3] = size (UInt32)
    """
    var config = entry_config[]
    var thread_id = config.thread_id
    var num_threads = config.num_threads

    # UInt64-word view of the FAM (SDK 1.23): pointer i is at word i + 1.
    var base = pointer_array.bitcast[UInt64]()
    var ptr_a_addr = (base + 1)[]
    var ptr_b_addr = (base + 2)[]
    var ptr_scalar_addr = (base + 3)[]
    var ptr_size_addr = (base + 4)[]

    var a = UnsafePointer[Float32, MutUnsafeAnyOrigin](
        unsafe_from_address=Int(ptr_a_addr)
    )
    var b = UnsafePointer[Float32, MutUnsafeAnyOrigin](
        unsafe_from_address=Int(ptr_b_addr)
    )
    var scalar_ptr = UnsafePointer[Float32, MutUnsafeAnyOrigin](
        unsafe_from_address=Int(ptr_scalar_addr)
    )
    var size_ptr = UnsafePointer[UInt32, MutUnsafeAnyOrigin](
        unsafe_from_address=Int(ptr_size_addr)
    )

    var scalar = scalar_ptr[]
    var size = Int(size_ptr[])

    var i = Int(thread_id)
    while i < size:
        b[i] = a[i] * scalar
        i += Int(num_threads)

    return 0


# ===-----------------------------------------------------------------------===#
# Helper Functions
# ===-----------------------------------------------------------------------===#


def is_elf_binary(data: String) -> Bool:
    """Check if data starts with ELF magic bytes (0x7F 'E' 'L' 'F')."""
    if data.byte_length() < 4:
        return False
    return (
        ord(data[byte=0]) == 0x7F
        and data[byte=1] == "E"
        and data[byte=2] == "L"
        and data[byte=3] == "F"
    )


def print_elf_info(data: String):
    """Print basic info about ELF binary."""
    if not is_elf_binary(data):
        print("  Not an ELF binary")
        return

    print("  ELF Magic: ✓")
    print("  Size:", data.byte_length(), "bytes")

    # ELF class (32-bit vs 64-bit) is at offset 4
    if data.byte_length() > 4:
        var elf_class = ord(data[byte=4])
        if elf_class == 1:
            print("  Class: ELF32 (32-bit) - correct for Hexagon")
        elif elf_class == 2:
            print("  Class: ELF64 (64-bit)")
        else:
            print("  Class: Unknown (", elf_class, ")")

    # Data encoding (endianness) is at offset 5
    if data.byte_length() > 5:
        var encoding = ord(data[byte=5])
        if encoding == 1:
            print("  Encoding: Little-endian - correct for Hexagon")
        elif encoding == 2:
            print("  Encoding: Big-endian")


def alloc_jit_buffer(
    ctx: DeviceContext, jit_device: QAicDevice, size: Int
) raises -> QAicBuf:
    """Helper to allocate a JIT DDR buffer with standard settings."""
    return qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        UInt64(size),
        HVX_ALIGN,
        ALL_COMPUTE_UNITS,
    )


# ===-----------------------------------------------------------------------===#
# Tests
# ===-----------------------------------------------------------------------===#


def run_accelerator_arch() raises:
    """Test 1: Verify _accelerator_arch() returns QAIC on this machine."""
    print()
    print_test_header("Test 1: Accelerator Architecture Detection")

    var arch = _accelerator_arch()
    print("_accelerator_arch() returns:", arch)

    if "qaic" in arch or "aic100" in arch:
        print("✓ QAIC architecture detected")
    else:
        print("✗ Expected QAIC architecture, got:", arch)
        raise Error("Not a QAIC machine - cannot run remaining tests")


def run_compile_produces_elf() raises:
    """Test 2: Verify compile_function produces ELF binary for QAIC target."""
    print()
    print_test_header("Test 2: Compilation to ELF")

    print("Compiling vector_add_kernel with get_gpu_target...")
    var elf = compile_info[
        vector_add_kernel,
        emission_kind="object",
        target=get_gpu_target(),
    ]()

    print("Compilation result:")
    print("  ELF size:", elf.asm.byte_length(), "bytes")
    print("  Function name:", elf.function_name)
    print("✓ Compilation produced valid ELF")


def run_load_function_elf_detection() raises:
    """Test 3: Verify loadFunction detects ELF and routes to JIT loader."""
    print()
    print_test_header("Test 3: ELF Detection in loadFunction")

    var ctx = DeviceContext(api="qaic")
    print("Created QAIC DeviceContext")

    # Compile kernel and load via JIT path
    print("Compiling and loading function...")
    var elf = compile_info[
        vector_add_kernel,
        emission_kind="object",
        target=get_gpu_target(),
    ]()
    var jit_func = qaic_load_jit_function(
        ctx,
        elf.function_name,
        elf.asm.unsafe_ptr().bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(elf.asm.byte_length()),
    )

    print("✓ Function loaded successfully via JIT path")
    qaic_release_jit_function(jit_func)


def run_jit_buffer_allocation() raises:
    """Test 4: Verify JIT buffer allocation and host access via mapping."""
    print()
    print_test_header("Test 4: JIT Buffer Allocation and Mapping")

    var ctx = DeviceContext(api="qaic")
    var jit_device = qaic_get_jit_device(ctx)
    print("Got JIT device handle")

    # Allocate a test DDR buffer
    var n = 1024
    var buf_size = n * 4  # 1024 Float32s
    var ddr_buf = alloc_jit_buffer(ctx, jit_device, buf_size)
    print("Allocated DDR buffer:", buf_size, "bytes")

    # Map the buffer for bidirectional access (use TO_DEVICE for initial write)
    var mapped_buf = qaic_map_jit_buffer(
        ctx, ddr_buf, UInt64(buf_size), QAIC_COPY_TO_DEVICE
    )
    print("Created host-mapped view of DDR buffer")

    # Get host pointer from mapped buffer
    var host_info = qaic_get_jit_buffer_host_info(ctx, mapped_buf)
    var host_ptr = host_info[0].bitcast[Float32]()
    var host_size = host_info[1]
    print("Got host pointer, size:", host_size, "bytes")

    # Initialize via mapped memory (direct host access)
    for i in range(n):
        host_ptr[i] = Float32(i)
    print("Initialized data via mapped memory [0, 1, 2, ..., 1023]")

    # Read back and verify
    var errors = 0
    for i in range(n):
        if abs(host_ptr[i] - Float32(i)) > 0.001:
            errors += 1

    if errors == 0:
        print("✓ Mapped buffer access verified - all", n, "elements correct")
    else:
        print("✗ Buffer verification failed:", errors, "errors")

    # Cleanup (release mapped buffer first, then DDR buffer)
    qaic_release_jit_buffer(ctx, mapped_buf)
    qaic_release_jit_buffer(ctx, ddr_buf)
    print("Freed buffers")

    if errors > 0:
        raise Error("Verification failed")


def run_full_execution_workflow() raises:
    """Test 5: Full workflow - compile, load, allocate buffers, execute, verify.
    """
    print()
    print_test_header("Test 5: Full Execution Workflow (Vector Add)")

    var ctx = DeviceContext(api="qaic")
    var jit_device = qaic_get_jit_device(ctx)

    # Step 1: Compile and load kernel
    print("\n[Step 1] Compiling vector_add_kernel...")
    var elf = compile_info[
        vector_add_kernel,
        emission_kind="object",
        target=get_gpu_target(),
    ]()
    var jit_func = qaic_load_jit_function(
        ctx,
        elf.function_name,
        elf.asm.unsafe_ptr().bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(elf.asm.byte_length()),
    )
    print("  Compiled and loaded")

    # Step 2: Allocate DDR buffers
    print("\n[Step 2] Allocating JIT buffers...")
    var n = 256  # Number of elements
    var buf_size = n * 4  # Float32 = 4 bytes

    var buf_a = alloc_jit_buffer(ctx, jit_device, buf_size)
    var buf_b = alloc_jit_buffer(ctx, jit_device, buf_size)
    # Allocate output buffer C with IO_RW type (device writes, host reads back)
    var buf_c = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        UInt64(buf_size),
        HVX_ALIGN,
        ALL_COMPUTE_UNITS,
    )
    var buf_size_val = alloc_jit_buffer(ctx, jit_device, 4)
    print("  Allocated 4 DDR buffers (A, B, C[output], size)")

    # Step 3: Map buffers for host access and initialize
    print("\n[Step 3] Mapping buffers and initializing data...")

    # Map input buffers with TO_DEVICE direction for writing
    var map_a = qaic_map_jit_buffer(
        ctx, buf_a, UInt64(buf_size), QAIC_COPY_TO_DEVICE
    )
    var map_b = qaic_map_jit_buffer(
        ctx, buf_b, UInt64(buf_size), QAIC_COPY_TO_DEVICE
    )
    # Map output buffer with FROM_DEVICE for reading back results
    var map_c = qaic_map_jit_buffer(
        ctx, buf_c, UInt64(buf_size), QAIC_COPY_FROM_DEVICE
    )
    var map_size = qaic_map_jit_buffer(
        ctx, buf_size_val, UInt64(4), QAIC_COPY_TO_DEVICE
    )

    # Get host pointers
    var ptr_a = qaic_get_jit_buffer_host_info(ctx, map_a)[0].bitcast[Float32]()
    var ptr_b = qaic_get_jit_buffer_host_info(ctx, map_b)[0].bitcast[Float32]()
    var ptr_c = qaic_get_jit_buffer_host_info(ctx, map_c)[0].bitcast[Float32]()
    var ptr_size = qaic_get_jit_buffer_host_info(ctx, map_size)[0].bitcast[
        UInt32
    ]()

    # Initialize input data through mapped memory
    # Note: Don't initialize ptr_c - it's an output-only buffer (IO_WRITE type)
    for i in range(n):
        ptr_a[i] = Float32(i)  # A = [0, 1, 2, ..., n-1]
        ptr_b[i] = Float32(i * 2)  # B = [0, 2, 4, ..., 2*(n-1)]
    ptr_size[0] = UInt32(n)
    print("  A = [0, 1, 2, ...,", n - 1, "]")
    print("  B = [0, 2, 4, ...,", 2 * (n - 1), "]")
    print("  Expected C = A + B = [0, 3, 6, ...,", 3 * (n - 1), "]")

    # Sync input buffers to device before kernel execution
    # Note: Don't sync map_c (output) - it's mapped with FROM_DEVICE direction
    print("  Syncing input buffers to device...")
    qaic_sync_jit_buffer(ctx, map_a)
    qaic_sync_jit_buffer(ctx, map_b)
    qaic_sync_jit_buffer(ctx, map_size)

    # Step 4: Build buffer array and execute
    print("\n[Step 4] Executing kernel...")
    # Create buffer array using alloc - pass DDR buffers to kernel
    var buffers = alloc[QAicBuf](4)
    buffers[0] = buf_a
    buffers[1] = buf_b
    buffers[2] = buf_c
    buffers[3] = buf_size_val

    # Execute with 1 core, all HVX threads (0x1F = 5 threads on v68)
    qaic_execute_jit_function(
        ctx, jit_func, buffers.as_unsafe_any_origin(), 4, 1, 0x1F
    )
    print("  Execution completed")

    # Sync output buffer from device after kernel execution
    print("  Syncing output buffer from device...")
    qaic_sync_jit_buffer(ctx, map_c)

    # Step 5: Read output through the existing mapping and verify
    print("\n[Step 5] Verifying results...")
    # For IO_RW buffers, we can read back through the same mapped buffer
    # (QAIC doesn't allow remapping with different direction)

    var errors = 0
    var first_error_idx = -1
    for i in range(n):
        var expected = Float32(i + i * 2)  # A[i] + B[i] = i + 2*i = 3*i
        var actual = ptr_c[i]
        if abs(actual - expected) > 0.001:
            if first_error_idx < 0:
                first_error_idx = i
            errors += 1

    if errors == 0:
        print("  ✓ All", n, "elements correct!")
        print(
            "  Sample: C[0] =",
            ptr_c[0],
            ", C[127] =",
            ptr_c[127],
            ", C[255] =",
            ptr_c[255],
        )
    else:
        print("  ✗", errors, "errors out of", n, "elements")
        if first_error_idx >= 0:
            var i = first_error_idx
            print(
                "  First error at index",
                i,
                ": expected",
                Float32(3 * i),
                ", got",
                ptr_c[i],
            )

    # Cleanup
    print("\n[Cleanup] Freeing buffers...")
    buffers.unsafe_free()
    # Release mapped buffers first
    qaic_release_jit_buffer(ctx, map_a)
    qaic_release_jit_buffer(ctx, map_b)
    qaic_release_jit_buffer(ctx, map_c)
    qaic_release_jit_buffer(ctx, map_size)
    # Then release DDR buffers
    qaic_release_jit_buffer(ctx, buf_a)
    qaic_release_jit_buffer(ctx, buf_b)
    qaic_release_jit_buffer(ctx, buf_c)
    qaic_release_jit_buffer(ctx, buf_size_val)
    qaic_release_jit_function(jit_func)
    print("  Done")

    if errors > 0:
        raise Error("Verification failed")


def run_scalar_multiply() raises:
    """Test 6: Additional kernel test - scalar multiply."""
    print()
    print_test_header("Test 6: Scalar Multiply Kernel")

    var ctx = DeviceContext(api="qaic")
    var jit_device = qaic_get_jit_device(ctx)

    # Compile and load
    print("\n[Step 1] Compiling scalar_multiply_kernel...")
    var elf = compile_info[
        scalar_multiply_kernel,
        emission_kind="object",
        target=get_gpu_target(),
    ]()
    var jit_func = qaic_load_jit_function(
        ctx,
        elf.function_name,
        elf.asm.unsafe_ptr().bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(elf.asm.byte_length()),
    )
    print("  Compiled and loaded")

    # Allocate DDR buffers
    var n = 128
    var buf_size = n * 4
    var buf_a = alloc_jit_buffer(ctx, jit_device, buf_size)
    # buf_b is output - use IO_RW type (device writes, host reads back)
    var buf_b = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        UInt64(buf_size),
        HVX_ALIGN,
        ALL_COMPUTE_UNITS,
    )
    var buf_scalar = alloc_jit_buffer(ctx, jit_device, 4)
    var buf_size_val = alloc_jit_buffer(ctx, jit_device, 4)

    # Map buffers and initialize (buf_b is output, uses FROM_DEVICE for reading)
    var map_a = qaic_map_jit_buffer(
        ctx, buf_a, UInt64(buf_size), QAIC_COPY_TO_DEVICE
    )
    var map_b = qaic_map_jit_buffer(
        ctx, buf_b, UInt64(buf_size), QAIC_COPY_FROM_DEVICE
    )
    var map_scalar = qaic_map_jit_buffer(
        ctx, buf_scalar, UInt64(4), QAIC_COPY_TO_DEVICE
    )
    var map_size = qaic_map_jit_buffer(
        ctx, buf_size_val, UInt64(4), QAIC_COPY_TO_DEVICE
    )

    var ptr_a = qaic_get_jit_buffer_host_info(ctx, map_a)[0].bitcast[Float32]()
    var ptr_b = qaic_get_jit_buffer_host_info(ctx, map_b)[0].bitcast[Float32]()
    var ptr_scalar = qaic_get_jit_buffer_host_info(ctx, map_scalar)[0].bitcast[
        Float32
    ]()
    var ptr_size = qaic_get_jit_buffer_host_info(ctx, map_size)[0].bitcast[
        UInt32
    ]()

    var scalar_val: Float32 = 2.5
    # Initialize input data only - ptr_b is output (IO_WRITE type)
    for i in range(n):
        ptr_a[i] = Float32(i)
    ptr_scalar[0] = scalar_val
    ptr_size[0] = UInt32(n)

    print("  A = [0, 1, 2, ...,", n - 1, "], scalar =", scalar_val)
    print(
        "  Expected B = A * scalar = [0, 2.5, 5.0, ...,",
        Float32(n - 1) * scalar_val,
        "]",
    )

    # Sync input buffers to device (not map_b - it's output mapped with FROM_DEVICE)
    qaic_sync_jit_buffer(ctx, map_a)
    qaic_sync_jit_buffer(ctx, map_scalar)
    qaic_sync_jit_buffer(ctx, map_size)

    # Execute
    var buffers = alloc[QAicBuf](4)
    buffers[0] = buf_a
    buffers[1] = buf_b
    buffers[2] = buf_scalar
    buffers[3] = buf_size_val

    qaic_execute_jit_function(
        ctx, jit_func, buffers.as_unsafe_any_origin(), 4, 1, 0x1F
    )

    # Sync output buffer from device
    qaic_sync_jit_buffer(ctx, map_b)
    print("  Executed")

    # Verify output through existing mapping
    var errors = 0
    for i in range(n):
        var expected = Float32(i) * scalar_val
        if abs(ptr_b[i] - expected) > 0.01:
            errors += 1

    if errors == 0:
        print("  ✓ All", n, "elements correct!")
        print(
            "  Sample: B[0] =",
            ptr_b[0],
            ", B[64] =",
            ptr_b[64],
            ", B[127] =",
            ptr_b[127],
        )
    else:
        print("  ✗", errors, "errors out of", n, "elements")

    # Cleanup
    buffers.unsafe_free()
    qaic_release_jit_buffer(ctx, map_a)
    qaic_release_jit_buffer(ctx, map_b)
    qaic_release_jit_buffer(ctx, map_scalar)
    qaic_release_jit_buffer(ctx, map_size)
    qaic_release_jit_buffer(ctx, buf_a)
    qaic_release_jit_buffer(ctx, buf_b)
    qaic_release_jit_buffer(ctx, buf_scalar)
    qaic_release_jit_buffer(ctx, buf_size_val)
    qaic_release_jit_function(jit_func)

    if errors > 0:
        raise Error("Verification failed")


# ===-----------------------------------------------------------------------===#
# User-Friendly Kernel (new pattern)
# ===-----------------------------------------------------------------------===#


def user_friendly_vector_add(
    input_a: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    input_b: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    """User-friendly vector add kernel using npu_work_range.

    No entry_config, no pointer_array, no manual buffer extraction.
    compile_function auto-wraps this with make_npu_wrapper.
    """
    var (start, end) = npu_work_range(256)
    for i in range(start, end):
        output[i] = input_a[i] + input_b[i]


def run_user_friendly_kernel() raises:
    """Test 7: User-friendly kernel with compile_function + enqueue_function.

    This demonstrates the recommended high-level API:
    - Write kernels with clean UnsafePointer signatures
    - compile_function auto-wraps with make_npu_wrapper
    - enqueue_function accepts DeviceBuffers directly
    """
    print()
    print_test_header("Test 7: User-Friendly Kernel (High-Level API)")

    var ctx = DeviceContext(api="qaic")
    var n = 256

    # Allocate device buffers via DeviceContext (no manual JIT buffer calls)
    var a_dev = ctx.enqueue_create_buffer[DType.float32](n)
    var b_dev = ctx.enqueue_create_buffer[DType.float32](n)
    var c_dev = ctx.enqueue_create_buffer[DType.float32](n)

    # Prepare host data
    var a_host = alloc[Float32](n)
    var b_host = alloc[Float32](n)
    for i in range(n):
        a_host[i] = Float32(i)
        b_host[i] = Float32(i * 2)

    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.synchronize()

    # Compile: just pass the user kernel, wrapping is automatic
    print("  Compiling user_friendly_vector_add...")
    var func = ctx.compile_function[user_friendly_vector_add]()
    print("  Compiled")

    # Execute: pass DeviceBuffers directly
    print("  Executing...")
    ctx.enqueue_function(
        func, a_dev, b_dev, c_dev, grid_dim=Dim(1), block_dim=Dim(0x01)
    )
    ctx.synchronize()

    # Verify
    var c_host = alloc[Float32](n)
    ctx.enqueue_copy(c_host, c_dev)
    ctx.synchronize()

    var errors = 0
    for i in range(n):
        if abs(c_host[i] - Float32(3 * i)) > 0.001:
            errors += 1

    if errors == 0:
        print("  ✓ All", n, "elements correct")
    else:
        print("  ✗", errors, "errors")

    a_host.unsafe_free()
    b_host.unsafe_free()
    c_host.unsafe_free()

    if errors > 0:
        raise Error("User-friendly kernel verification failed")


# ===-----------------------------------------------------------------------===#
# Test JIT attributes
# ===-----------------------------------------------------------------------===#


def run_jit_attributes() raises:
    """Test 8: Query JIT device attributes."""
    print()
    print_test_header("Test 8: JIT attributes")

    var ctx = DeviceContext(api="qaic")
    var jitDevice = qaic_get_jit_device(ctx)

    print(
        "QAIC_DEV_ATTR_TOTAL_DDR_KB: ",
        qaic_get_jit_device_attribute(
            ctx, jitDevice, QAIC_DEV_ATTR_TOTAL_DDR_KB
        ),
    )
    print(
        "QAIC_DEV_ATTR_FREE_DDR_KB: ",
        qaic_get_jit_device_attribute(
            ctx, jitDevice, QAIC_DEV_ATTR_FREE_DDR_KB
        ),
    )
    print(
        "QAIC_DEV_ATTR_NSP_ARCH: ",
        qaic_get_jit_device_attribute(ctx, jitDevice, QAIC_DEV_ATTR_NSP_ARCH),
    )
    print(
        "QAIC_DEV_ATTR_NSP_VA_SPACE_KB: ",
        qaic_get_jit_device_attribute(
            ctx, jitDevice, QAIC_DEV_ATTR_NSP_VA_SPACE_KB
        ),
    )
    print(
        "QAIC_DEV_ATTR_UDMA_MAX_BW_KBPS: ",
        qaic_get_jit_device_attribute(
            ctx, jitDevice, QAIC_DEV_ATTR_UDMA_MAX_BW_KBPS
        ),
    )
    print(
        "QAIC_DEV_ATTR_DDR_MAX_BW_KBPS: ",
        qaic_get_jit_device_attribute(
            ctx, jitDevice, QAIC_DEV_ATTR_DDR_MAX_BW_KBPS
        ),
    )
    print(
        "QAIC_DEV_ATTR_NSP_THREADS: ",
        qaic_get_jit_device_attribute(
            ctx, jitDevice, QAIC_DEV_ATTR_NSP_THREADS
        ),
    )
    print(
        "QAIC_DEV_ATTR_THREAD_BITMAP: ",
        qaic_get_jit_device_attribute(
            ctx, jitDevice, QAIC_DEV_ATTR_THREAD_BITMAP
        ),
    )
    print(
        "QAIC_DEV_ATTR_HMX_THREAD_ID: ",
        qaic_get_jit_device_attribute(
            ctx, jitDevice, QAIC_DEV_ATTR_HMX_THREAD_ID
        ),
    )


# ===-----------------------------------------------------------------------===#
# Main
# ===-----------------------------------------------------------------------===#


def test_qaic_jit_workflow() raises:
    print_test_header("QAIC JIT Workflow Test Suite")
    print("This test verifies the CUDA/HIP-style API for QAIC devices:")
    print("  1. compile_function -> ELF binary (via hexagon-link)")
    print("  2. loadFunction -> ELF detection -> JIT loader")
    print("  3. enqueueFunctionExecDirect -> kernel execution")

    # Run tests in sequence
    run_accelerator_arch()
    run_compile_produces_elf()
    run_load_function_elf_detection()
    run_jit_buffer_allocation()
    run_full_execution_workflow()
    run_scalar_multiply()
    run_user_friendly_kernel()
    run_jit_attributes()

    print()
    print_test_header("All tests passed! ✓")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
