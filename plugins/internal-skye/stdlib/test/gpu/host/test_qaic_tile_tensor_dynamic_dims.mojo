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
"""Regression test: dynamic `TileTensor` with bare-`Int` runtime dims through
`enqueue_function`.

Previously the host-side serializer used a blind byte-copy of the
`TileTensor` struct into the device metadata buffer, which silently
truncated 8-byte host `Int` shape dims into the 4-byte Hexagon slot and
shifted subsequent fields. With `row_major(Int(M), Int(N))`, the device
would read `shape[1]` (the high 4 bytes of host `shape[0]`) as `0`, the
matmul kernel would run zero MMA iterations, and the output buffer would
come back as garbage. This test exercises that exact case and asserts
the kernel sees the dims the host sent.
"""

from std.memory import UnsafePointer, alloc
from max.gpu.host import DeviceContext, DeviceBuffer, Dim
from std.testing import TestSuite, assert_equal
from layout import TileTensor, TensorLayout, row_major


def probe_dims[
    c_layout: TensorLayout,
](
    c_tensor: TileTensor[DType.float16, c_layout, MutUnsafeAnyOrigin],
    out_buf: UnsafePointer[Int32, MutUnsafeAnyOrigin],
):
    """Writes the kernel-visible shape/stride of `c_tensor` into `out_buf`."""
    out_buf[0] = Int32(c_tensor.dim(0))
    out_buf[1] = Int32(c_tensor.dim(1))
    out_buf[2] = Int32(c_tensor.layout.stride[0]().value())
    out_buf[3] = Int32(c_tensor.layout.stride[1]().value())


def test_qaic_tile_tensor_dynamic_dims() raises:
    with DeviceContext(device_id=0, api="qaic") as ctx:
        var M: Int32 = 1024
        var N: Int32 = 512

        var data_buf = ctx.enqueue_create_buffer[DType.float16](Int(M) * Int(N))
        var out_dev = ctx.enqueue_create_buffer[DType.int32](4)
        var out_host = alloc[Int32](4)
        for i in range(4):
            out_host[i] = Int32(-1)
        ctx.enqueue_copy(out_dev, out_host)
        ctx.synchronize()

        # Bare `Int` runtime dims — the exact pattern the bug corrupted.
        var c_tensor = TileTensor(data_buf, row_major(Int(M), Int(N)))

        comptime func = probe_dims[c_tensor.LayoutType]
        var kernel = ctx.compile_function[func]()
        ctx.enqueue_function(
            kernel,
            c_tensor,
            out_dev,
            grid_dim=Dim(1),
            block_dim=Dim(0x01),
        )
        ctx.synchronize()
        ctx.enqueue_copy(out_host, out_dev)
        ctx.synchronize()

        var d0 = out_host[0]
        var d1 = out_host[1]
        var s0 = out_host[2]
        var s1 = out_host[3]
        out_host.unsafe_free()
        _ = data_buf
        _ = out_dev

        assert_equal(Int(d0), Int(M))
        assert_equal(Int(d1), Int(N))
        assert_equal(Int(s0), Int(N))
        assert_equal(Int(s1), 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
