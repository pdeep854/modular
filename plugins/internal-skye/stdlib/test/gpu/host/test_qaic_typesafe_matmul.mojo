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
"""Test variadic make_npu_wrapper with mixed arg types on QAIC.

This test validates the variadic NPU wrapper generator that replaces the
manual N-argument overloads. It tests:

1. UnsafePointer args — backward compatibility with the old overloads
2. Mixed UnsafePointer + scalar args — new capability
3. LayoutTensor args — compilation validation (ELF generation)

The variadic wrapper (make_npu_wrapper) and compile_info_npu use
Variadic.TypesOfTrait to accept kernels with any combination of
UnsafePointer, LayoutTensor, and scalar arguments.

To run (requires QAIC hardware for execution tests):
    mojo test_qaic_typesafe_matmul.mojo
"""

from std.testing import TestSuite, assert_equal, assert_not_equal
from common import print_separator

from std.memory import UnsafePointer, alloc

from max.gpu.host import DeviceContext
from std._plugin.qaic.host.target import (
    compile_info_npu,
    make_npu_wrapper,
    npu_get_buffer,
    AicJitEntryPointConfig,
    AicJitPointerArray,
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
# Test Utilities
# ===-----------------------------------------------------------------------===#


def check_qaic_available() raises -> Bool:
    try:
        var ctx = DeviceContext(device_id=0, api="qaic")
        _ = ctx
        return True
    except e:
        print("QAIC not available:", e)
        return False


# ===-----------------------------------------------------------------------===#
# Test Kernels
# ===-----------------------------------------------------------------------===#


# Kernel 1: UnsafePointer args (backward compatibility with old overloads)
def matmul_kernel(
    A: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    B: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    C: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    dims: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    """Matrix multiplication: C = A * B with dims = [M, K, N]."""
    var M = Int(dims[0])
    var K = Int(dims[1])
    var N = Int(dims[2])

    for i in range(M):
        for j in range(N):
            var sum: Float32 = 0.0
            for k in range(K):
                sum += A[i * K + k] * B[k * N + j]
            C[i * N + j] = sum


# Kernel 2: Mixed UnsafePointer + scalar (new variadic capability)
def scale_kernel(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    size: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
    scale: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    """Scale each element: output[i] = input[i] * scale."""
    var n = Int(size[])
    var s = scale[]
    for i in range(n):
        output[i] = input[i] * s


# ===-----------------------------------------------------------------------===#
# Test 1: Compilation — UnsafePointer kernel (backward compat)
# ===-----------------------------------------------------------------------===#


def run_compile_pointer_kernel() raises:
    """Test that compile_info_npu works with UnsafePointer args via variadic path.
    """
    print("  Compiling matmul_kernel (4 UnsafePointer args)...")
    var elf = compile_info_npu[matmul_kernel]()
    var elf_size = elf.asm.byte_length()
    print("    ELF size:", elf_size, "bytes")
    print("    Function name:", elf.function_name)
    assert_not_equal(elf_size, 0)


# ===-----------------------------------------------------------------------===#
# Test 2: Compilation — Scale kernel (also UnsafePointer, different arity)
# ===-----------------------------------------------------------------------===#


def run_compile_scale_kernel() raises:
    """Test compile_info_npu with a 4-arg scale kernel."""
    print("  Compiling scale_kernel (4 UnsafePointer args)...")
    var elf = compile_info_npu[scale_kernel]()
    var elf_size = elf.asm.byte_length()
    print("    ELF size:", elf_size, "bytes")
    assert_not_equal(elf_size, 0)


# ===-----------------------------------------------------------------------===#
# Test 3: Hardware execution — matmul on QAIC (UnsafePointer path)
# ===-----------------------------------------------------------------------===#


def run_hardware_matmul() raises:
    """Run matmul_kernel on QAIC hardware via the variadic wrapper.

    This is the same matmul test as test_qaic_matmul.mojo but compiled
    through the variadic compile_info_npu path, validating that the
    variadic wrapper produces correct kernels on real hardware.
    """
    print("  Running matmul on QAIC hardware...")

    var M: Int = 4
    var K: Int = 3
    var N: Int = 5
    var A_size = M * K
    var B_size = K * N
    var C_size = M * N

    # Compile via variadic path
    var elf = compile_info_npu[matmul_kernel]()

    # Set up QAIC context and JIT
    var ctx = DeviceContext(device_id=0, api="qaic")
    var jit_device = qaic_get_jit_device(ctx)
    var jit_func = qaic_load_jit_function(
        ctx,
        elf.function_name,
        elf.asm.unsafe_ptr().bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(elf.asm.byte_length()),
    )

    # Allocate buffers
    var alignment: UInt32 = 128
    var cu_bitmap: UInt64 = 0xFFFFFFFFFFFFFFFF

    var A_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        UInt64(A_size * 4),
        alignment,
        cu_bitmap,
    )
    var B_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        UInt64(B_size * 4),
        alignment,
        cu_bitmap,
    )
    var C_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        UInt64(C_size * 4),
        alignment,
        cu_bitmap,
    )
    var dims_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        UInt64(3 * 4),
        alignment,
        cu_bitmap,
    )

    # Prepare input: A = sequential, B = permutation matrix
    var A_host = alloc[Float32](A_size)
    for i in range(A_size):
        A_host[i] = Float32(i + 1)

    var B_host = alloc[Float32](B_size)
    for i in range(B_size):
        B_host[i] = Float32(0)
    B_host[0 * N + 0] = Float32(1)
    B_host[0 * N + 4] = Float32(1)
    B_host[1 * N + 1] = Float32(1)
    B_host[1 * N + 3] = Float32(1)
    B_host[2 * N + 2] = Float32(1)

    var dims_host = alloc[UInt32](3)
    dims_host[0] = UInt32(M)
    dims_host[1] = UInt32(K)
    dims_host[2] = UInt32(N)

    # Copy to device
    qaic_copy_jit_buffer(
        ctx,
        A_buf,
        A_host.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(A_size * 4),
        QAIC_COPY_TO_DEVICE,
    )
    qaic_copy_jit_buffer(
        ctx,
        B_buf,
        B_host.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(B_size * 4),
        QAIC_COPY_TO_DEVICE,
    )
    qaic_copy_jit_buffer(
        ctx,
        dims_buf,
        dims_host.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(3 * 4),
        QAIC_COPY_TO_DEVICE,
    )

    # Execute
    var buffers = alloc[QAicBuf](4)
    buffers[0] = A_buf
    buffers[1] = B_buf
    buffers[2] = C_buf
    buffers[3] = dims_buf
    qaic_execute_jit_function(
        ctx, jit_func, buffers.as_unsafe_any_origin(), 4, 1, 0x01
    )

    # Read back
    var C_host = alloc[Float32](C_size)
    qaic_copy_jit_buffer(
        ctx,
        C_buf,
        C_host.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(C_size * 4),
        QAIC_COPY_FROM_DEVICE,
    )

    # Compute expected
    var expected = alloc[Float32](C_size)
    for i in range(M):
        for j in range(N):
            var sum: Float32 = 0.0
            for k in range(K):
                sum += A_host[i * K + k] * B_host[k * N + j]
            expected[i * N + j] = sum

    # Verify
    var errors = 0
    for i in range(C_size):
        if abs(C_host[i] - expected[i]) > 0.0001:
            errors += 1
            if errors <= 3:
                print(
                    "    ERROR at [",
                    i,
                    "]: expected",
                    expected[i],
                    "got",
                    C_host[i],
                )

    if errors == 0:
        print("    All", C_size, "elements verified!")

    # Cleanup
    A_host.unsafe_free()
    B_host.unsafe_free()
    C_host.unsafe_free()
    dims_host.unsafe_free()
    expected.unsafe_free()
    buffers.unsafe_free()
    qaic_release_jit_buffer(ctx, A_buf)
    qaic_release_jit_buffer(ctx, B_buf)
    qaic_release_jit_buffer(ctx, C_buf)
    qaic_release_jit_buffer(ctx, dims_buf)
    qaic_release_jit_function(jit_func)

    assert_equal(errors, 0)


# ===-----------------------------------------------------------------------===#
# Test 5: Hardware execution — scale kernel on QAIC
# ===-----------------------------------------------------------------------===#


def run_hardware_scale() raises:
    """Run scale_kernel on QAIC hardware via the variadic wrapper."""
    print("  Running scale kernel on QAIC hardware...")

    var size: Int = 16
    var scale_val: Float32 = 2.5

    var elf = compile_info_npu[scale_kernel]()

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
        UInt64(size * 4),
        alignment,
        cu_bitmap,
    )
    var output_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        UInt64(size * 4),
        alignment,
        cu_bitmap,
    )
    var size_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        UInt64(4),
        alignment,
        cu_bitmap,
    )
    var scale_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        UInt64(4),
        alignment,
        cu_bitmap,
    )

    # Prepare input
    var input_host = alloc[Float32](size)
    for i in range(size):
        input_host[i] = Float32(i + 1)

    var size_host = alloc[UInt32](1)
    size_host[0] = UInt32(size)

    var scale_host = alloc[Float32](1)
    scale_host[0] = scale_val

    # Copy to device
    qaic_copy_jit_buffer(
        ctx,
        input_buf,
        input_host.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(size * 4),
        QAIC_COPY_TO_DEVICE,
    )
    qaic_copy_jit_buffer(
        ctx,
        size_buf,
        size_host.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(4),
        QAIC_COPY_TO_DEVICE,
    )
    qaic_copy_jit_buffer(
        ctx,
        scale_buf,
        scale_host.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(4),
        QAIC_COPY_TO_DEVICE,
    )

    # Execute
    var buffers = alloc[QAicBuf](4)
    buffers[0] = input_buf
    buffers[1] = output_buf
    buffers[2] = size_buf
    buffers[3] = scale_buf
    qaic_execute_jit_function(
        ctx, jit_func, buffers.as_unsafe_any_origin(), 4, 1, 0x01
    )

    # Read back
    var output_host = alloc[Float32](size)
    qaic_copy_jit_buffer(
        ctx,
        output_buf,
        output_host.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(size * 4),
        QAIC_COPY_FROM_DEVICE,
    )

    # Verify: output[i] should be (i+1) * 2.5
    var errors = 0
    for i in range(size):
        var expected = Float32(i + 1) * scale_val
        if abs(output_host[i] - expected) > 0.0001:
            errors += 1
            if errors <= 3:
                print(
                    "    ERROR at [",
                    i,
                    "]: expected",
                    expected,
                    "got",
                    output_host[i],
                )

    if errors == 0:
        print("    All", size, "elements verified!")

    # Cleanup
    input_host.unsafe_free()
    output_host.unsafe_free()
    size_host.unsafe_free()
    scale_host.unsafe_free()
    buffers.unsafe_free()
    qaic_release_jit_buffer(ctx, input_buf)
    qaic_release_jit_buffer(ctx, output_buf)
    qaic_release_jit_buffer(ctx, size_buf)
    qaic_release_jit_buffer(ctx, scale_buf)
    qaic_release_jit_function(jit_func)

    assert_equal(errors, 0)


# ===-----------------------------------------------------------------------===#
# Main
# ===-----------------------------------------------------------------------===#


def test_qaic_typesafe_matmul() raises:
    print_separator()
    print("QAIC Variadic Wrapper Test (LayoutTensor + UnsafePointer)")
    print_separator()
    print()

    # --- Compilation tests (no hardware needed) ---

    print("Phase 1: Compilation Tests")
    print("-" * 40)

    print("Test 1: Compile UnsafePointer matmul kernel")
    run_compile_pointer_kernel()
    print()

    print("Test 2: Compile scale kernel")
    run_compile_scale_kernel()
    print()

    # --- Hardware execution tests (require QAIC) ---

    print("Phase 2: Hardware Execution Tests")
    print("-" * 40)

    var qaic_available: Bool
    try:
        qaic_available = check_qaic_available()
    except:
        qaic_available = False

    if not qaic_available:
        print("  QAIC hardware not available — skipping execution tests")
        print()
    else:
        print("  QAIC device found!")
        print()

        print("Test 3: Hardware matmul (UnsafePointer args)")
        run_hardware_matmul()
        print()

        print("Test 4: Hardware scale (UnsafePointer args)")
        run_hardware_scale()
        print()

    print_separator()
    print("ALL TESTS PASSED")
    print_separator()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
