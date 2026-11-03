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
"""Test QAIC naive matrix multiplication kernel.

This test demonstrates matrix multiplication on QAIC:
    C = A * B
    C[i,j] = sum(A[i,k] * B[k,j]) for k in 0..K

Matrices are stored in row-major order:
    A[i,k] is at index i*K + k
    B[k,j] is at index k*N + j
    C[i,j] is at index i*N + j

This naive implementation uses the standard i-j-k loop order for clarity.
For optimized versions, see test_qaic_matmul_optimized.mojo.

Buffer layout:
    [0]: A      - Input matrix A (float32), shape [M, K]
    [1]: B      - Input matrix B (float32), shape [K, N]
    [2]: C      - Output matrix C (float32), shape [M, N]
    [3]: dims   - Dimensions array (uint32): [M, K, N]

To run (requires QAIC hardware):
    mojo test_qaic_matmul.mojo
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
# Kernel: Matrix Multiplication (Naive i-j-k order)
# ===-----------------------------------------------------------------------===#


def matmul_kernel(
    A: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    B: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    C: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    dims: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    """Matrix multiplication: C = A * B.

    Computes C[i,j] = sum(A[i,k] * B[k,j]) for k in 0..K.
    Matrices are stored in row-major order.

    Args:
        A: Input matrix A (float32), shape [M, K].
        B: Input matrix B (float32), shape [K, N].
        C: Output matrix C (float32), shape [M, N].
        dims: Dimensions array (uint32): [M, K, N].
    """
    var M = Int(dims[0])
    var K = Int(dims[1])
    var N = Int(dims[2])

    for i in range(M):
        for j in range(N):
            var sum: Float32 = 0.0
            for k in range(K):
                sum += A[i * K + k] * B[k * N + j]
            C[i * N + j] = sum


# ===-----------------------------------------------------------------------===#
# Test
# ===-----------------------------------------------------------------------===#


def test_qaic_matmul() raises:
    print_test_header("QAIC Matrix Multiplication Test (Naive)")
    print()

    # Test dimensions: small matrices for quick verification
    var M: Int = 4  # Rows of A and C
    var K: Int = 3  # Cols of A, Rows of B
    var N: Int = 5  # Cols of B and C

    var A_size = M * K  # 12 elements
    var B_size = K * N  # 15 elements
    var C_size = M * N  # 20 elements
    var dims_size: Int = 3  # M, K, N

    var A_bytes = A_size * 4
    var B_bytes = B_size * 4
    var C_bytes = C_size * 4
    var dims_bytes = dims_size * 4

    print("Matrix dimensions:")
    print("  A:", M, "x", K, "(", A_size, "elements )")
    print("  B:", K, "x", N, "(", B_size, "elements )")
    print("  C:", M, "x", N, "(", C_size, "elements )")
    print()

    # Compile the matmul kernel
    print("Step 1: Compiling matmul kernel...")
    var elf = compile_info_npu[matmul_kernel]()
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

    var A_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        UInt64(A_bytes),
        alignment,
        cu_bitmap,
    )
    var B_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        UInt64(B_bytes),
        alignment,
        cu_bitmap,
    )
    var C_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        UInt64(C_bytes),
        alignment,
        cu_bitmap,
    )
    var dims_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        UInt64(dims_bytes),
        alignment,
        cu_bitmap,
    )
    print("  Allocated A, B, C, dims buffers")
    print()

    # Prepare input data
    print("Step 5: Preparing input data...")

    # A = [[1,2,3], [4,5,6], [7,8,9], [10,11,12]] (4x3)
    var A_host = alloc[Float32](A_size)
    for i in range(M):
        for k in range(K):
            A_host[i * K + k] = Float32(i * K + k + 1)

    # B = sparse matrix with 1s at specific positions (3x5)
    # This creates a pattern for easy verification
    var B_host = alloc[Float32](B_size)
    for i in range(B_size):
        B_host[i] = Float32(0)
    # Set specific elements
    B_host[0 * N + 0] = Float32(1)  # B[0,0] = 1
    B_host[0 * N + 4] = Float32(1)  # B[0,4] = 1
    B_host[1 * N + 1] = Float32(1)  # B[1,1] = 1
    B_host[1 * N + 3] = Float32(1)  # B[1,3] = 1
    B_host[2 * N + 2] = Float32(1)  # B[2,2] = 1

    var dims_host = alloc[UInt32](dims_size)
    dims_host[0] = UInt32(M)
    dims_host[1] = UInt32(K)
    dims_host[2] = UInt32(N)

    print("  A (4x3):")
    for i in range(M):
        var row = String("    [")
        for k in range(K):
            row += String(Int(A_host[i * K + k]))
            if k < K - 1:
                row += ", "
        row += "]"
        print(row)

    print("  B (3x5): sparse matrix with 1s at specific positions")
    print()

    # Copy to device
    print("Step 6: Copying data to device...")
    qaic_copy_jit_buffer(
        ctx,
        A_buf,
        A_host.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(A_bytes),
        QAIC_COPY_TO_DEVICE,
    )
    qaic_copy_jit_buffer(
        ctx,
        B_buf,
        B_host.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(B_bytes),
        QAIC_COPY_TO_DEVICE,
    )
    qaic_copy_jit_buffer(
        ctx,
        dims_buf,
        dims_host.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(dims_bytes),
        QAIC_COPY_TO_DEVICE,
    )
    print("  Copied A, B, dims to device")
    print()

    # Execute kernel
    print("Step 7: Executing matmul kernel...")
    var buffers = alloc[QAicBuf](4)
    buffers[0] = A_buf
    buffers[1] = B_buf
    buffers[2] = C_buf
    buffers[3] = dims_buf

    qaic_execute_jit_function(
        ctx, jit_func, buffers.as_unsafe_any_origin(), 4, 1, 0x01
    )
    print("  Execution completed")
    print()

    # Copy results back
    print("Step 8: Copying results from device...")
    var C_host = alloc[Float32](C_size)
    qaic_copy_jit_buffer(
        ctx,
        C_buf,
        C_host.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(C_bytes),
        QAIC_COPY_FROM_DEVICE,
    )
    print("  Copied C from device")
    print()

    # Compute expected results on host
    var expected = alloc[Float32](C_size)
    for i in range(M):
        for j in range(N):
            var sum: Float32 = 0.0
            for k in range(K):
                sum += A_host[i * K + k] * B_host[k * N + j]
            expected[i * N + j] = sum

    # Verify results
    print("Step 9: Verifying results...")
    var errors = 0
    for i in range(M):
        for j in range(N):
            var idx = i * N + j
            var exp = expected[idx]
            var act = C_host[idx]
            if abs(act - exp) > 0.0001:
                errors += 1
                if errors <= 5:
                    print(
                        "  ERROR at C[",
                        i,
                        ",",
                        j,
                        "]: expected",
                        exp,
                        "got",
                        act,
                    )

    if errors == 0:
        print("  SUCCESS: All", C_size, "elements verified!")
        print()
        print("  Result matrix C (4x5):")
        for i in range(M):
            var row = String("    [")
            for j in range(N):
                row += String(Int(C_host[i * N + j]))
                if j < N - 1:
                    row += ", "
            row += "]"
            print(row)
    else:
        print("  FAILED:", errors, "errors")

    # Cleanup
    print()
    print("Step 10: Cleaning up...")
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
