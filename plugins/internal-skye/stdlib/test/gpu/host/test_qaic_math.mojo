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
"""Test various basic math support.

To run (requires QAIC hardware):
    mojo test_qaic_math.mojo
"""

from std.math import exp, pow, tanh
from std.memory import UnsafePointer, alloc
from std.testing import TestSuite, assert_almost_equal, assert_equal

from max.gpu.host import DeviceContext, Dim
from std._plugin.qaic.host.target import npu_work_range


comptime N: Int = 256


# ===-----------------------------------------------------------------------===#
# pow
# ===-----------------------------------------------------------------------===#


def pow_kernel(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    """Vector pow(x, 3)."""
    var (start, end) = npu_work_range(N)
    for i in range(start, end):
        output[i] = pow(input[i], 3)


def test_pow() raises:
    """Test pow(x, 3) over inputs in [-1, 1)."""
    var ctx = DeviceContext(device_id=0, api="qaic")
    var in_dev = ctx.enqueue_create_buffer[DType.float32](N)
    var out_dev = ctx.enqueue_create_buffer[DType.float32](N)

    var in_host = alloc[Float32](N)
    for i in range(N):
        in_host[i] = (Float32(i) - Float32(N) / 2.0) / Float32(N) * 2.0
    ctx.enqueue_copy(in_dev, in_host)
    ctx.synchronize()

    var func = ctx.compile_function[pow_kernel]()
    ctx.enqueue_function(
        func, in_dev, out_dev, grid_dim=Dim(1), block_dim=Dim(0x01)
    )
    ctx.synchronize()

    var out_host = alloc[Float32](N)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    for i in range(N):
        assert_almost_equal(out_host[i], pow(in_host[i], 3), atol=1e-4)

    in_host.unsafe_free()
    out_host.unsafe_free()


# ===-----------------------------------------------------------------------===#
# exp
# ===-----------------------------------------------------------------------===#


def exp_kernel(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    """Vector exp using SIMD width-32 loads/stores for HVX vectorization."""
    var (start, end) = npu_work_range(N)
    for i in range(start, end, 32):
        output.store[width=32](i, exp(input.load[width=32](i)))


def test_exp() raises:
    """SIMD-vectorized exp with HVX width-32 loads/stores."""
    var ctx = DeviceContext(device_id=0, api="qaic")
    var in_dev = ctx.enqueue_create_buffer[DType.float32](N)
    var out_dev = ctx.enqueue_create_buffer[DType.float32](N)

    var in_host = alloc[Float32](N)
    for i in range(N):
        in_host[i] = Float32(i) / Float32(N)
    ctx.enqueue_copy(in_dev, in_host)
    ctx.synchronize()

    var func = ctx.compile_function[exp_kernel]()
    ctx.enqueue_function(
        func, in_dev, out_dev, grid_dim=Dim(1), block_dim=Dim(0x01)
    )
    ctx.synchronize()

    var out_host = alloc[Float32](N)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    for i in range(N):
        assert_almost_equal(out_host[i], exp(in_host[i]), atol=1e-4)

    in_host.unsafe_free()
    out_host.unsafe_free()


def exp_f16_hvx_kernel(
    input: UnsafePointer[Float16, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float16, MutUnsafeAnyOrigin],
):
    """HVX-vectorized exp using SIMD[float16, 64]: hits _hexagon_runtime_exp__vhf.
    """
    var (start, end) = npu_work_range(N)
    for i in range(start, end, 64):
        output.store[width=64](i, exp(input.load[width=64](i)))


def test_exp_f16_hvx() raises:
    """SIMD[float16, 64] exp: exercises the hexagon_runtime_exp__vhf HVX path.
    """
    var ctx = DeviceContext(device_id=0, api="qaic")
    var in_dev = ctx.enqueue_create_buffer[DType.float16](N)
    var out_dev = ctx.enqueue_create_buffer[DType.float16](N)

    var in_host = alloc[Float16](N)
    for i in range(N):
        # Spread inputs over [-4, 4); exp stays in [~0.018, ~54.6], well within
        # the float16 range, and this exercises the non-trivial part of the
        # curve on both sides of zero.
        in_host[i] = Float16((Float32(i) - Float32(N) / 2.0) / Float32(N) * 8.0)
    ctx.enqueue_copy(in_dev, in_host)
    ctx.synchronize()

    var func = ctx.compile_function[exp_f16_hvx_kernel]()
    ctx.enqueue_function(
        func, in_dev, out_dev, grid_dim=Dim(1), block_dim=Dim(0x01)
    )
    ctx.synchronize()

    var out_host = alloc[Float16](N)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    for i in range(N):
        # Upcast to float32 for comparison. The HVX fp16 exp is a hardware
        # qhmath approximation, so use a relative tolerance: observed error is
        # ~1.25% near the bottom of the range, and rtol scales the bound across
        # the [~0.018, ~54.6] output range. This still catches the
        # bitcast-misinterpretation regression, which was off by orders of
        # magnitude.
        var expected = exp(Float32(in_host[i]))
        assert_almost_equal(Float32(out_host[i]), expected, rtol=3e-2)

    in_host.unsafe_free()
    out_host.unsafe_free()


# ===-----------------------------------------------------------------------===#
# tanh
# ===-----------------------------------------------------------------------===#


def tanh_scalar_kernel(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    """Scalar tanh (width=1)."""
    var (start, end) = npu_work_range(N)
    for i in range(start, end):
        output[i] = tanh(input[i])


def test_tanh() raises:
    """Scalar tanh (width=1)."""
    var ctx = DeviceContext(device_id=0, api="qaic")
    var in_dev = ctx.enqueue_create_buffer[DType.float32](N)
    var out_dev = ctx.enqueue_create_buffer[DType.float32](N)

    var in_host = alloc[Float32](N)
    for i in range(N):
        in_host[i] = (Float32(i) - Float32(N) / 2.0) / Float32(N)
    ctx.enqueue_copy(in_dev, in_host)
    ctx.synchronize()

    var func = ctx.compile_function[tanh_scalar_kernel]()
    ctx.enqueue_function(
        func, in_dev, out_dev, grid_dim=Dim(1), block_dim=Dim(0x01)
    )
    ctx.synchronize()

    var out_host = alloc[Float32](N)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    for i in range(N):
        assert_almost_equal(out_host[i], tanh(in_host[i]), atol=1e-4)

    in_host.unsafe_free()
    out_host.unsafe_free()


def tanh_f32_hvx_kernel(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    """HVX-vectorized tanh using SIMD[float32, 32]: hits _hexagon_runtime_tanh__vf.
    """
    var (start, end) = npu_work_range(N)
    for i in range(start, end, 32):
        output.store[width=32](i, tanh(input.load[width=32](i)))


def test_tanh_f32_hvx() raises:
    """SIMD[float32, 32] tanh: exercises the hexagon_runtime_tanh__vf HVX path.
    """
    var ctx = DeviceContext(device_id=0, api="qaic")
    var in_dev = ctx.enqueue_create_buffer[DType.float32](N)
    var out_dev = ctx.enqueue_create_buffer[DType.float32](N)

    var in_host = alloc[Float32](N)
    for i in range(N):
        # Spread inputs over [-2, 2) to cover the interesting part of tanh.
        in_host[i] = (Float32(i) - Float32(N) / 2.0) / Float32(N) * 4.0
    ctx.enqueue_copy(in_dev, in_host)
    ctx.synchronize()

    var func = ctx.compile_function[tanh_f32_hvx_kernel]()
    ctx.enqueue_function(
        func, in_dev, out_dev, grid_dim=Dim(1), block_dim=Dim(0x01)
    )
    ctx.synchronize()

    var out_host = alloc[Float32](N)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    for i in range(N):
        assert_almost_equal(out_host[i], tanh(in_host[i]), atol=1e-4)

    in_host.unsafe_free()
    out_host.unsafe_free()


def tanh_f16_hvx_kernel(
    input: UnsafePointer[Float16, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float16, MutUnsafeAnyOrigin],
):
    """HVX-vectorized tanh using SIMD[float16, 64]: hits _hexagon_runtime_tanh__vhf.
    """
    var (start, end) = npu_work_range(N)
    for i in range(start, end, 64):
        output.store[width=64](i, tanh(input.load[width=64](i)))


def test_tanh_f16_hvx() raises:
    """SIMD[float16, 64] tanh: exercises the hexagon_runtime_tanh__vhf HVX path.
    """
    var ctx = DeviceContext(device_id=0, api="qaic")
    var in_dev = ctx.enqueue_create_buffer[DType.float16](N)
    var out_dev = ctx.enqueue_create_buffer[DType.float16](N)

    var in_host = alloc[Float16](N)
    for i in range(N):
        # Spread inputs over [-2, 2); tanh is bounded in [-1, 1] so no
        # float16 overflow risk, and this range exercises the non-trivial
        # part of the curve.
        in_host[i] = Float16((Float32(i) - Float32(N) / 2.0) / Float32(N) * 4.0)
    ctx.enqueue_copy(in_dev, in_host)
    ctx.synchronize()

    var func = ctx.compile_function[tanh_f16_hvx_kernel]()
    ctx.enqueue_function(
        func, in_dev, out_dev, grid_dim=Dim(1), block_dim=Dim(0x01)
    )
    ctx.synchronize()

    var out_host = alloc[Float16](N)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    for i in range(N):
        # Upcast to float32 for comparison; float16 has ~3 decimal digits of
        # precision so 1e-2 is the appropriate tolerance.
        var expected = tanh(Float32(in_host[i]))
        assert_almost_equal(Float32(out_host[i]), expected, atol=1e-2)

    in_host.unsafe_free()
    out_host.unsafe_free()


# ===-----------------------------------------------------------------------===#
# UInt64 vector addition
# ===-----------------------------------------------------------------------===#


def uint64_vec_add_kernel(
    a: UnsafePointer[UInt64, MutUnsafeAnyOrigin],
    b: UnsafePointer[UInt64, MutUnsafeAnyOrigin],
    c: UnsafePointer[UInt64, MutUnsafeAnyOrigin],
):
    """Vector addition c[i] = a[i] + b[i] over UInt64 elements."""
    var (start, end) = npu_work_range(N)
    var i: UInt64 = UInt64(start)
    while i < UInt64(end):
        c[i] = a[i] + b[i]
        i += 1


def test_uint64_vec_add() raises:
    """Vector addition over UInt64 elements with UInt64 loop counter and pointer subscript.
    """
    var ctx = DeviceContext(device_id=0, api="qaic")
    var a_dev = ctx.enqueue_create_buffer[DType.uint64](N)
    var b_dev = ctx.enqueue_create_buffer[DType.uint64](N)
    var c_dev = ctx.enqueue_create_buffer[DType.uint64](N)

    var a_host = alloc[UInt64](N)
    var b_host = alloc[UInt64](N)
    for i in range(N):
        a_host[i] = UInt64(i)
        b_host[i] = UInt64(2 * i)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.synchronize()

    var func = ctx.compile_function[uint64_vec_add_kernel]()
    ctx.enqueue_function(
        func, a_dev, b_dev, c_dev, grid_dim=Dim(1), block_dim=Dim(0x01)
    )
    ctx.synchronize()

    var c_host = alloc[UInt64](N)
    ctx.enqueue_copy(c_host, c_dev)
    ctx.synchronize()

    for i in range(N):
        assert_equal(c_host[i], UInt64(3 * i))

    a_host.unsafe_free()
    b_host.unsafe_free()
    c_host.unsafe_free()


# ===-----------------------------------------------------------------------===#
# Int64 vector addition
# ===-----------------------------------------------------------------------===#


def int64_vec_add_kernel(
    a: UnsafePointer[Int64, MutUnsafeAnyOrigin],
    b: UnsafePointer[Int64, MutUnsafeAnyOrigin],
    c: UnsafePointer[Int64, MutUnsafeAnyOrigin],
):
    """Vector addition c[i] = a[i] + b[i] over Int64 elements."""
    var (start, end) = npu_work_range(N)
    var i: Int64 = Int64(start)
    while i < Int64(end):
        c[i] = a[i] + b[i]
        i += 1


def test_int64_vec_add() raises:
    """Vector addition over Int64 elements with Int64 loop counter and pointer subscript.
    """
    var ctx = DeviceContext(device_id=0, api="qaic")
    var a_dev = ctx.enqueue_create_buffer[DType.int64](N)
    var b_dev = ctx.enqueue_create_buffer[DType.int64](N)
    var c_dev = ctx.enqueue_create_buffer[DType.int64](N)

    var a_host = alloc[Int64](N)
    var b_host = alloc[Int64](N)
    for i in range(N):
        a_host[i] = Int64(i)
        b_host[i] = Int64(2 * i)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.synchronize()

    var func = ctx.compile_function[int64_vec_add_kernel]()
    ctx.enqueue_function(
        func, a_dev, b_dev, c_dev, grid_dim=Dim(1), block_dim=Dim(0x01)
    )
    ctx.synchronize()

    var c_host = alloc[Int64](N)
    ctx.enqueue_copy(c_host, c_dev)
    ctx.synchronize()

    for i in range(N):
        assert_equal(c_host[i], Int64(3 * i))

    a_host.unsafe_free()
    b_host.unsafe_free()
    c_host.unsafe_free()


# ===-----------------------------------------------------------------------===#
# GELU (from elemwise.mojo)
# ===-----------------------------------------------------------------------===#

comptime GELU_SQRT_2_OVER_PI: Float32 = 0.7978845608028654
comptime GELU_COEFF: Float32 = 0.044715


def gelu_kernel(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    """GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))."""
    var (start, end) = npu_work_range(N)
    for i in range(start, end):
        var x = input[i]
        output[i] = (
            0.5
            * x
            * (1.0 + tanh(GELU_SQRT_2_OVER_PI * (x + GELU_COEFF * pow(x, 3))))
        )


def test_gelu() raises:
    """GELU activation matching the gelu_op formula from elemwise.mojo."""
    var ctx = DeviceContext(device_id=0, api="qaic")
    var in_dev = ctx.enqueue_create_buffer[DType.float32](N)
    var out_dev = ctx.enqueue_create_buffer[DType.float32](N)

    var in_host = alloc[Float32](N)
    for i in range(N):
        in_host[i] = (Float32(i) - Float32(N) / 2.0) / Float32(N) * 4.0
    ctx.enqueue_copy(in_dev, in_host)
    ctx.synchronize()

    var func = ctx.compile_function[gelu_kernel]()
    ctx.enqueue_function(
        func, in_dev, out_dev, grid_dim=Dim(1), block_dim=Dim(0x01)
    )
    ctx.synchronize()

    var out_host = alloc[Float32](N)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    for i in range(N):
        var x = in_host[i]
        var expected = (
            0.5
            * x
            * (1.0 + tanh(GELU_SQRT_2_OVER_PI * (x + GELU_COEFF * pow(x, 3))))
        )
        assert_almost_equal(out_host[i], expected, atol=1e-4)

    in_host.unsafe_free()
    out_host.unsafe_free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
