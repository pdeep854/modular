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
"""Tests for all DeviceContext.enqueue_function overloads on QAIC.

To run (requires QAIC hardware):
    mojo test_qaic_enqueue_function_overloads.mojo
"""

from std.testing import TestSuite, assert_equal
from std.memory import UnsafePointer, alloc
from max.gpu.host import DeviceContext, DeviceBuffer, Dim
from std._plugin.qaic.host.target import npu_work_range
from max._plugin.qaic.host.qaic import qaic_split

comptime N: Int = 16
comptime NUM_CORES: Int = 4
comptime CHUNK: Int = N // NUM_CORES


def vec_add_kernel(
    a: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    b: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    c: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    var (start, end) = npu_work_range(N)
    for i in range(start, end):
        c[i] = a[i] + b[i]


def fill_chunk_kernel(buf: UnsafePointer[Float32, MutUnsafeAnyOrigin]):
    for i in range(CHUNK):
        buf[i] = Float32(42)


def double_kernel(
    a: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    b: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    var (start, end) = npu_work_range(N)
    for i in range(start, end):
        b[i] = a[i] * Float32(2)


def _make_sequential_dev_buf(
    ctx: DeviceContext, n: Int
) raises -> DeviceBuffer[DType.float32]:
    """Allocate a device buffer initialised to [1, 2, ..., n]."""
    var host = alloc[Float32](n)
    for i in range(n):
        host[i] = Float32(i + 1)
    var dev = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_copy(dev, host)
    ctx.synchronize()
    host.unsafe_free()
    return dev^


def _read_back(
    ctx: DeviceContext, dev: DeviceBuffer[DType.float32], n: Int
) raises -> UnsafePointer[Float32, MutUnsafeAnyOrigin]:
    """Copy device buffer to a host allocation; caller must free."""
    var host = alloc[Float32](n)
    ctx.enqueue_copy(host, dev)
    ctx.synchronize()
    return host.as_unsafe_any_origin()


def test_precompiled_checked_args() raises:
    """Overload: enqueue_function(DeviceFunction, *DevicePassable, ...) with DeviceBuffer args.
    """
    var ctx = DeviceContext(api="qaic")
    var a_dev = _make_sequential_dev_buf(ctx, N)
    var b_dev = ctx.enqueue_create_buffer[DType.float32](N)
    var c_dev = ctx.enqueue_create_buffer[DType.float32](N)

    var b_host = alloc[Float32](N)
    for i in range(N):
        b_host[i] = Float32(10)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.synchronize()
    b_host.unsafe_free()

    var func = ctx.compile_function[vec_add_kernel]()
    ctx.enqueue_function(
        func, a_dev, b_dev, c_dev, grid_dim=Dim(1), block_dim=Dim(0x01)
    )
    ctx.synchronize()

    var result = _read_back(ctx, c_dev, N)
    for i in range(N):
        assert_equal(result[i], Float32(i + 11))
    result.unsafe_free()


def test_npu_any_type_args() raises:
    """Overload: enqueue_function(DeviceFunction[npu], *AnyType, ...) bypassing DevicePassable.
    """
    var ctx = DeviceContext(api="qaic")
    var buf = ctx.enqueue_create_buffer[DType.float32](N)

    var func = ctx.compile_function[fill_chunk_kernel]()
    # qaic_split returns List[QAicBufferDescriptor], which is not DevicePassable,
    # so this selects the NPU-specific any-type overload.
    ctx.enqueue_function(
        func,
        qaic_split(buf, NUM_CORES),
        grid_dim=Dim(NUM_CORES),
        block_dim=Dim(0x01),
    )
    ctx.synchronize()

    var result = _read_back(ctx, buf, N)
    for i in range(N):
        assert_equal(result[i], Float32(42))
    result.unsafe_free()


def test_npu_precompiled_closure() raises:
    """Overload: enqueue_function(DeviceFunction[make_npu_wrapper_closure], closure, ...) NPU closure.
    """
    var ctx = DeviceContext(api="qaic")
    var a_dev = _make_sequential_dev_buf(ctx, N)
    var b_dev = ctx.enqueue_create_buffer[DType.float32](N)

    var scale: Float32 = 3.0
    # `unsafe_ptr()` now ties the pointer origin to the buffer borrow; erase
    # it so the captured pointers don't embed tracked mutable origins that the
    # closure enqueue's aliasing check rejects.
    var a = a_dev.unsafe_ptr().as_unsafe_any_origin()
    var b = b_dev.unsafe_ptr().as_unsafe_any_origin()

    def scale_closure() {var a, var b, var scale}:
        for i in range(N):
            b[i] = a[i] * scale

    var func = ctx.compile_function[type_of(scale_closure)]()
    ctx.enqueue_function(
        func, scale_closure, grid_dim=Dim(1), block_dim=Dim(0x01)
    )
    ctx.synchronize()

    var result = _read_back(ctx, b_dev, N)
    for i in range(N):
        assert_equal(result[i], Float32(i + 1) * scale)
    result.unsafe_free()


def test_inline_thin() raises:
    """Overload: enqueue_function[thin_func](*args, ...) compile-and-enqueue for module-level kernel.
    """
    var ctx = DeviceContext(api="qaic")
    var a_dev = _make_sequential_dev_buf(ctx, N)
    var b_dev = ctx.enqueue_create_buffer[DType.float32](N)

    ctx.enqueue_function[double_kernel](
        a_dev, b_dev, grid_dim=Dim(1), block_dim=Dim(0x01)
    )
    ctx.synchronize()

    var result = _read_back(ctx, b_dev, N)
    for i in range(N):
        assert_equal(result[i], Float32(i + 1) * Float32(2))
    result.unsafe_free()


# TODO: test_inline_capturing — disabled pending capturing-kernel support.
# Exercises: enqueue_function[capturing_func](*args, ...) with a locally-defined kernel.


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
