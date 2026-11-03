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
"""Tests for closure-based NPU kernel dispatch via enqueue_function.

Exercises the four capture categories handled by _call_with_closure_npu and
make_npu_wrapper_closure:

- test_pointer_scalar_closure:       UnsafePointer + scalar captures
- test_tile_tensor_closure:          mixed static + dynamic TileTensor captures

To run (requires QAIC hardware):
    mojo test_qaic_closure_enqueue.mojo
"""

from std.testing import TestSuite, assert_equal

from std.memory import AddressSpace, UnsafePointer, alloc, stack_allocation

from layout import TileTensor
from layout.tile_layout import RowMajorLayout, row_major
from layout.coord import ComptimeInt

from max.gpu.host import DeviceContext, Dim


comptime _TT = TileTensor[
    DType.float32,
    RowMajorLayout[ComptimeInt[1], ComptimeInt[16]],
    MutUnsafeAnyOrigin,
]

comptime N: Int = 16


@fieldwise_init
struct ScaleConfig(TrivialRegisterPassable):
    """Captured struct with a Float32 and an Int (64-bit host, 32-bit device).
    """

    var scale: Float32
    var offset: Int


def test_pointer_scalar_closure() raises:
    """Closure with pointer + scalar captures: output[i] = input[i] * scale."""
    var ctx = DeviceContext(api="qaic")

    var input_dev = ctx.enqueue_create_buffer[DType.float32](N)
    var output_dev = ctx.enqueue_create_buffer[DType.float32](N)

    var input_host = alloc[Float32](N)
    for i in range(N):
        input_host[i] = Float32(i + 1)
    ctx.enqueue_copy(input_dev, input_host)
    ctx.synchronize()
    input_host.unsafe_free()

    var scale: Float32 = 2.5
    var size: Int32 = Int32(N)
    # `unsafe_ptr()` now ties the pointer origin to the buffer borrow; erase
    # it so the captured pointers don't embed tracked mutable origins that the
    # closure enqueue's aliasing check rejects.
    var inp = input_dev.unsafe_ptr().as_unsafe_any_origin()
    var out = output_dev.unsafe_ptr().as_unsafe_any_origin()

    def scale_kernel() {var inp, var out, var scale, var size}:
        for i in range(Int(size)):
            out[i] = inp[i] * scale

    var func = ctx.compile_function[type_of(scale_kernel)]()
    ctx.enqueue_function(func, scale_kernel, grid_dim=Dim(1), block_dim=Dim(1))
    ctx.synchronize()

    _ = input_dev

    var result = alloc[Float32](N)
    ctx.enqueue_copy(result, output_dev)
    ctx.synchronize()

    for i in range(N):
        assert_equal(result[i], Float32(i + 1) * scale)

    result.unsafe_free()


def test_tile_tensor_closure() raises:
    """Closure with mixed static and dynamic TileTensor captures.

    a_tt and b_tt are static (ComptimeInt dims, pointer-only on device).
    c_tt is dynamic (Scalar dims, ptr + inline layout bytes in blob).
    """
    var ctx = DeviceContext(api="qaic")

    var a_dev = ctx.enqueue_create_buffer[DType.float32](N)
    var b_dev = ctx.enqueue_create_buffer[DType.float32](N)
    var c_dev = ctx.enqueue_create_buffer[DType.float32](N)

    var a_host = alloc[Float32](N)
    var b_host = alloc[Float32](N)
    for i in range(N):
        a_host[i] = Float32(i + 1)  # 1 .. 16
        b_host[i] = Float32(N - i)  # 16 .. 1

    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.synchronize()
    a_host.unsafe_free()
    b_host.unsafe_free()

    var layout = row_major[1, 16]()
    var a_tt = _TT(a_dev, layout)  # static
    var b_tt = _TT(b_dev, layout)  # static
    var c_tt = _TT(c_dev, layout).make_dynamic[DType.int32]()  # dynamic

    def tt_add_closure() {var a_tt, var b_tt, var c_tt}:
        for i in range(N):
            c_tt[0, i] = a_tt[0, i] + b_tt[0, i]

    var func = ctx.compile_function[type_of(tt_add_closure)]()
    ctx.enqueue_function(
        func, tt_add_closure, grid_dim=Dim(1), block_dim=Dim(0x01)
    )
    ctx.synchronize()
    _ = a_dev^
    _ = b_dev^

    var result = alloc[Float32](N)
    ctx.enqueue_copy(result, c_dev)
    ctx.synchronize()

    # a[i] + b[i] == (i+1) + (N-i) == N+1 for all i
    for i in range(N):
        assert_equal(result[i], Float32(N + 1))

    result.unsafe_free()


def test_vtcm_stack_allocation_closure() raises:
    """Closure that stages data through a VTCM scratch buffer.

    stack_allocation with AddressSpace.SHARED lowers to VTCM on Hexagon.
    The kernel copies input into VTCM scratch, then scales into output.
    """
    var ctx = DeviceContext(api="qaic")

    var input_dev = ctx.enqueue_create_buffer[DType.float32](N)
    var output_dev = ctx.enqueue_create_buffer[DType.float32](N)

    var input_host = alloc[Float32](N)
    for i in range(N):
        input_host[i] = Float32(i + 1)
    ctx.enqueue_copy(input_dev, input_host)
    ctx.synchronize()
    input_host.unsafe_free()

    var scale: Float32 = 2.5
    var inp = input_dev.unsafe_ptr().as_unsafe_any_origin()
    var out = output_dev.unsafe_ptr().as_unsafe_any_origin()

    def vtcm_scale_kernel() {var inp, var out, var scale}:
        var scratch = stack_allocation[
            N, Float32, address_space=AddressSpace.SHARED
        ]()
        for i in range(N):
            scratch[i] = inp[i]
        for i in range(N):
            out[i] = scratch[i] * scale

    var func = ctx.compile_function[type_of(vtcm_scale_kernel)]()
    ctx.enqueue_function(
        func, vtcm_scale_kernel, grid_dim=Dim(1), block_dim=Dim(1)
    )
    ctx.synchronize()

    _ = input_dev

    var result = alloc[Float32](N)
    ctx.enqueue_copy(result, output_dev)
    ctx.synchronize()

    for i in range(N):
        assert_equal(result[i], Float32(i + 1) * scale)

    result.unsafe_free()


def test_int_capture_closure() raises:
    """Closure with a bare Int capture — exercises Int→Int32 truncation in _serialize_npu_struct.
    """
    var ctx = DeviceContext(api="qaic")

    var output_dev = ctx.enqueue_create_buffer[DType.float32](N)
    var out = output_dev.unsafe_ptr().as_unsafe_any_origin()
    # Int on host (64-bit), truncated to Int32 (32-bit) on device
    var n: Int = N

    def int_kernel() {var out, var n}:
        for i in range(n):
            out[i] = Float32(i) * 3.0

    var func = ctx.compile_function[type_of(int_kernel)]()
    ctx.enqueue_function(func, int_kernel, grid_dim=Dim(1), block_dim=Dim(1))
    ctx.synchronize()

    var result = alloc[Float32](N)
    ctx.enqueue_copy(result, output_dev)
    ctx.synchronize()

    for i in range(N):
        assert_equal(result[i], Float32(i) * 3.0)

    result.unsafe_free()


def test_struct_capture_closure() raises:
    """Closure capturing a struct containing an Int — exercises recursive _serialize_npu_struct.
    """
    var ctx = DeviceContext(api="qaic")

    var output_dev = ctx.enqueue_create_buffer[DType.float32](N)
    var out = output_dev.unsafe_ptr().as_unsafe_any_origin()
    var config = ScaleConfig(scale=2.0, offset=5)

    def struct_kernel() {var out, var config}:
        for i in range(N):
            out[i] = Float32(i + config.offset) * config.scale

    var func = ctx.compile_function[type_of(struct_kernel)]()
    ctx.enqueue_function(func, struct_kernel, grid_dim=Dim(1), block_dim=Dim(1))
    ctx.synchronize()

    var result = alloc[Float32](N)
    ctx.enqueue_copy(result, output_dev)
    ctx.synchronize()

    for i in range(N):
        assert_equal(result[i], Float32(i + config.offset) * config.scale)

    result.unsafe_free()


def test_assert_in_closure() raises:
    """Closure kernel assert: passing case writes output, failing case raises.

    Pass: kernel asserts a positive captured value then writes it to the
    output buffer; the host confirms the write happened.

    Fail: kernel asserts a deliberately false condition; enqueue_function
    must raise rather than silently succeed.
    """
    var ctx = DeviceContext(api="qaic")

    var result_dev = ctx.enqueue_create_buffer[DType.uint32](1)
    var result_host = alloc[UInt32](1)
    result_host[0] = 0
    ctx.enqueue_copy(result_dev, result_host)
    ctx.synchronize()

    # --- passing assert ---
    var expected: UInt32 = 42
    var out = result_dev.unsafe_ptr().as_unsafe_any_origin()

    def assert_pass_kernel() {var out, var expected}:
        # Use VTCM variable here to ensure the assert handling code doesn't
        # break wrapper detection.
        var scratch = stack_allocation[
            1, UInt32, address_space=AddressSpace.SHARED
        ]()
        assert expected > 0, "captured bound must be positive"
        scratch[0] = expected
        out[0] = scratch[0]

    var pass_func = ctx.compile_function[type_of(assert_pass_kernel)]()
    ctx.enqueue_function(
        pass_func, assert_pass_kernel, grid_dim=Dim(1), block_dim=Dim(1)
    )
    ctx.synchronize()

    ctx.enqueue_copy(result_host, result_dev)
    ctx.synchronize()

    assert_equal(result_host[0], expected)
    # Note: Testing a failing assert causes hangs in following kernels


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
