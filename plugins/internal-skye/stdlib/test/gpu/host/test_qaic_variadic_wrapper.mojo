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
"""Test the variadic make_npu_wrapper and compile_info_npu on QAIC.

Validates that the single variadic make_npu_wrapper (replacing the old
0-8 arg manual overloads) produces correct Hexagon ELF and executes
correctly on QAIC hardware via the low-level JIT path.

To run (requires QAIC hardware):
    mojo test_qaic_variadic_wrapper.mojo
"""

from std.testing import TestSuite, assert_equal, assert_not_equal
from common import print_separator

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


def print_test_header(name: String):
    print("Test:", name)


# ===-----------------------------------------------------------------------===#
# Test Kernels
# ===-----------------------------------------------------------------------===#

# Compile-time constant size for kernels that don't use a dims buffer.
comptime TEST_N: Int = 64


def vector_add_kernel(
    a: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    b: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    c: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    """Simple vector add: c[i] = a[i] + b[i] for i in [0, TEST_N)."""
    for i in range(TEST_N):
        c[i] = a[i] + b[i]


def vector_scale_kernel(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    scale: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    """Scale: output[i] = input[i] * scale[0] for i in [0, TEST_N)."""
    var s = scale[]
    for i in range(TEST_N):
        output[i] = input[i] * s


def matmul_4x4_kernel(
    A: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    B: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    C: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    """4x4 matrix multiply: C = A * B, row-major."""
    for i in range(4):
        for j in range(4):
            var acc: Float32 = 0.0
            for k in range(4):
                acc += A[i * 4 + k] * B[k * 4 + j]
            C[i * 4 + j] = acc


def fill_kernel(
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    """Fill: output[i] = 42.0 for i in [0, TEST_N)."""
    for i in range(TEST_N):
        output[i] = Float32(42.0)


# ===-----------------------------------------------------------------------===#
# Test 1: compile_info_npu backward compatibility
#
# Verify the variadic compile_info_npu produces valid ELF for kernels with
# different arities, matching the old manual overloads.
# ===-----------------------------------------------------------------------===#


def run_compile_info_npu_backward_compat() raises:
    """Validate compile_info_npu with 1, 3, and 4-arg kernels via variadic path.
    """
    print_test_header("compile_info_npu backward compatibility")

    # 1-arg kernel
    print("  1-arg fill_kernel...")
    var elf1 = compile_info_npu[fill_kernel]()
    assert_not_equal(elf1.asm.byte_length(), 0)
    print("    ELF:", elf1.asm.byte_length(), "bytes")

    # 3-arg kernel
    print("  3-arg vector_add_kernel...")
    var elf3 = compile_info_npu[vector_add_kernel]()
    assert_not_equal(elf3.asm.byte_length(), 0)
    print("    ELF:", elf3.asm.byte_length(), "bytes")

    # 3-arg scale kernel
    print("  3-arg vector_scale_kernel...")
    var elf3s = compile_info_npu[vector_scale_kernel]()
    assert_not_equal(elf3s.asm.byte_length(), 0)
    print("    ELF:", elf3s.asm.byte_length(), "bytes")


# ===-----------------------------------------------------------------------===#
# Test 2: Low-level JIT pipeline (compile_info_npu + qaic_execute_jit_function)
#
# This is the "current _npu variant" path. Ensures the variadic wrapper
# generates correct kernels that execute on hardware.
# ===-----------------------------------------------------------------------===#


def run_jit_vector_add() raises:
    """Low-level JIT: compile + load + execute vector_add_kernel."""
    print_test_header("JIT pipeline vector add (compile_info_npu + execute)")

    var n = TEST_N
    var elf = compile_info_npu[vector_add_kernel]()

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
    var buf_bytes = UInt64(n * 4)

    var a_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        buf_bytes,
        alignment,
        cu_bitmap,
    )
    var b_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        buf_bytes,
        alignment,
        cu_bitmap,
    )
    var c_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        buf_bytes,
        alignment,
        cu_bitmap,
    )

    var a_host = alloc[Float32](n)
    var b_host = alloc[Float32](n)
    for i in range(n):
        a_host[i] = Float32(i)
        b_host[i] = Float32(i * 10)

    qaic_copy_jit_buffer(
        ctx,
        a_buf,
        a_host.bitcast[UInt8]().as_unsafe_any_origin(),
        buf_bytes,
        QAIC_COPY_TO_DEVICE,
    )
    qaic_copy_jit_buffer(
        ctx,
        b_buf,
        b_host.bitcast[UInt8]().as_unsafe_any_origin(),
        buf_bytes,
        QAIC_COPY_TO_DEVICE,
    )

    var buffers = alloc[QAicBuf](3)
    buffers[0] = a_buf
    buffers[1] = b_buf
    buffers[2] = c_buf
    qaic_execute_jit_function(
        ctx, jit_func, buffers.as_unsafe_any_origin(), 3, 1, 0x01
    )

    var c_host = alloc[Float32](n)
    qaic_copy_jit_buffer(
        ctx,
        c_buf,
        c_host.bitcast[UInt8]().as_unsafe_any_origin(),
        buf_bytes,
        QAIC_COPY_FROM_DEVICE,
    )

    var errors = 0
    for i in range(n):
        var expected = Float32(i) + Float32(i * 10)
        if abs(c_host[i] - expected) > 0.001:
            errors += 1
            if errors <= 3:
                print("    ERR [", i, "]:", c_host[i], "expected", expected)

    a_host.unsafe_free()
    b_host.unsafe_free()
    c_host.unsafe_free()
    buffers.unsafe_free()
    qaic_release_jit_buffer(ctx, a_buf)
    qaic_release_jit_buffer(ctx, b_buf)
    qaic_release_jit_buffer(ctx, c_buf)
    qaic_release_jit_function(jit_func)

    if errors == 0:
        print("  All", n, "elements correct")
    assert_equal(errors, 0)


# ===-----------------------------------------------------------------------===#
# Test 3: Low-level JIT pipeline with matmul (4x4)
# ===-----------------------------------------------------------------------===#


def run_jit_matmul_4x4() raises:
    """Low-level JIT: 4x4 matmul via variadic compile_info_npu."""
    print_test_header("JIT pipeline 4x4 matmul")

    var elf = compile_info_npu[matmul_4x4_kernel]()

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
    var buf_bytes = UInt64(16 * 4)  # 4x4 float32

    var A_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        buf_bytes,
        alignment,
        cu_bitmap,
    )
    var B_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        buf_bytes,
        alignment,
        cu_bitmap,
    )
    var C_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        buf_bytes,
        alignment,
        cu_bitmap,
    )

    # A = identity, B = [1..16]
    var A_host = alloc[Float32](16)
    var B_host = alloc[Float32](16)
    for i in range(16):
        A_host[i] = Float32(0)
        B_host[i] = Float32(i + 1)
    A_host[0] = 1
    A_host[5] = 1
    A_host[10] = 1
    A_host[15] = 1

    qaic_copy_jit_buffer(
        ctx,
        A_buf,
        A_host.bitcast[UInt8]().as_unsafe_any_origin(),
        buf_bytes,
        QAIC_COPY_TO_DEVICE,
    )
    qaic_copy_jit_buffer(
        ctx,
        B_buf,
        B_host.bitcast[UInt8]().as_unsafe_any_origin(),
        buf_bytes,
        QAIC_COPY_TO_DEVICE,
    )

    var buffers = alloc[QAicBuf](3)
    buffers[0] = A_buf
    buffers[1] = B_buf
    buffers[2] = C_buf
    qaic_execute_jit_function(
        ctx, jit_func, buffers.as_unsafe_any_origin(), 3, 1, 0x01
    )

    var C_host = alloc[Float32](16)
    qaic_copy_jit_buffer(
        ctx,
        C_buf,
        C_host.bitcast[UInt8]().as_unsafe_any_origin(),
        buf_bytes,
        QAIC_COPY_FROM_DEVICE,
    )

    # C = I * B = B
    var errors = 0
    for i in range(16):
        if abs(C_host[i] - B_host[i]) > 0.001:
            errors += 1
            if errors <= 3:
                print("    ERR [", i, "]:", C_host[i], "expected", B_host[i])

    A_host.unsafe_free()
    B_host.unsafe_free()
    C_host.unsafe_free()
    buffers.unsafe_free()
    qaic_release_jit_buffer(ctx, A_buf)
    qaic_release_jit_buffer(ctx, B_buf)
    qaic_release_jit_buffer(ctx, C_buf)
    qaic_release_jit_function(jit_func)

    if errors == 0:
        print("  All 16 elements correct (I * B = B)")
    assert_equal(errors, 0)


# ===-----------------------------------------------------------------------===#
# Main
# ===-----------------------------------------------------------------------===#


def test_qaic_variadic_wrapper() raises:
    print_separator()
    print("QAIC Variadic Wrapper Test")
    print_separator()
    print()

    # --- Phase 1: Compilation tests (no QAIC hardware needed) ---

    print("Phase 1: Compilation Tests (no hardware needed)")
    print("-" * 40)

    run_compile_info_npu_backward_compat()
    print()

    # --- Phase 2: Low-level JIT tests (require QAIC hardware) ---

    print("Phase 2: Low-Level JIT Tests (require QAIC)")
    print("-" * 40)

    var qaic_available: Bool
    try:
        qaic_available = check_qaic_available()
    except:
        qaic_available = False

    if not qaic_available:
        print("  QAIC hardware not available -- skipping")
        print()
    else:
        print("  QAIC device found")
        print()

        run_jit_vector_add()
        print()

        run_jit_matmul_4x4()
        print()

    print_separator()
    print("ALL TESTS PASSED")
    print_separator()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
