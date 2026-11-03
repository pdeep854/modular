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
"""End-to-end correctness tests for `elementwise[..., target="npu"]`.

Routes through `max._plugin.qaic.QAICMaxPlugin.elementwise_fn`, which delegates
to `_elementwise_qaic`. Bodies are unified closures with explicit `{var ...}`
captures; the upstream `_elementwise_impl` passes them straight to the plugin
hook, which dispatches via `compile_function` / `enqueue_function` and the
indexed-closure FAM-slot wrapper (`_make_npu_index_closure_wrapper`).
"""

from max.algorithm import elementwise
from max.gpu.host import DeviceContext
from std.memory import alloc
from std.testing import TestSuite, assert_almost_equal
from std.utils.index import IndexList
from std.utils.coord import Coord as StdCoord, coord_to_index_list

from layout import Coord, Idx, TileTensor, row_major
from std.builtin.dtype import DType
from layout.tile_layout import RowMajorLayout
from layout.coord import ComptimeInt


def test_rank_1() raises:
    """Rank-1 dispatch with pointer + scalar captures.

    Exercises the bulk path of the FAM-slot wrapper (with
    `simd_width=1`, the tail loop is unreachable).
    """
    var ctx = DeviceContext(api="qaic")
    comptime N: Int = 64
    var multiplier: Float32 = 2.0
    var bias: Float32 = 1.0

    var inp_dev = ctx.enqueue_create_buffer[DType.float32](N)
    var out_dev = ctx.enqueue_create_buffer[DType.float32](N)

    var inp_host = alloc[Float32](N)
    for i in range(N):
        inp_host[i] = Float32(i + 1)
    ctx.enqueue_copy(inp_dev, inp_host)
    ctx.synchronize()

    var inp_ptr = inp_dev._device_ptr
    var out_ptr = out_dev._device_ptr

    def body[
        width: Int, alignment: Int = 1
    ](indices: StdCoord) {var inp_ptr, var out_ptr, var multiplier, var bias}:
        var idx = rebind[IndexList[1]](coord_to_index_list(indices))
        var i = idx[0]
        out_ptr[i] = inp_ptr[i] * multiplier + bias

    elementwise[simd_width=1, target="npu"](
        body, StdCoord(IndexList[1](N)), ctx
    )
    ctx.synchronize()

    _ = inp_dev

    var out_host = alloc[Float32](N)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    for i in range(N):
        var expected = Float32(i + 1) * multiplier + bias
        assert_almost_equal(out_host[i], expected, atol=1e-4)

    inp_host.unsafe_free()
    out_host.unsafe_free()


def test_rank_2() raises:
    """Rank-2 dispatch.

    Allocates an H×W row-major buffer, captures `row_width` so the body
    can flatten the rank-2 index, and checks every output element. The
    point of the rank-2 case is the device-side de-linearization in
    `_make_npu_index_closure_wrapper` — if it produces wrong (row, col)
    for any linear worker index, the manually-flattened write lands in
    the wrong slot and the assert fires.
    """
    var ctx = DeviceContext(api="qaic")
    comptime H: Int = 8
    comptime W: Int = 16
    comptime N: Int = H * W
    var row_width: Int32 = Int32(W)
    var multiplier: Float32 = 2.0
    var bias: Float32 = 1.0

    var inp_dev = ctx.enqueue_create_buffer[DType.float32](N)
    var out_dev = ctx.enqueue_create_buffer[DType.float32](N)

    var inp_host = alloc[Float32](N)
    for i in range(N):
        inp_host[i] = Float32(i + 1)
    ctx.enqueue_copy(inp_dev, inp_host)
    ctx.synchronize()

    var inp_ptr = inp_dev._device_ptr
    var out_ptr = out_dev._device_ptr

    def body[
        width: Int, alignment: Int = 1
    ](indices: StdCoord) {
        var inp_ptr, var out_ptr, var row_width, var multiplier, var bias
    }:
        # Buffer is row-major contiguous; flatten the rank-2 index.
        var idx = rebind[IndexList[2]](coord_to_index_list(indices))
        var i = idx[0] * Int(row_width) + idx[1]
        out_ptr[i] = inp_ptr[i] * multiplier + bias

    elementwise[simd_width=1, target="npu"](
        body, StdCoord(IndexList[2](H, W)), ctx
    )
    ctx.synchronize()

    _ = inp_dev

    var out_host = alloc[Float32](N)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    for i in range(N):
        var expected = Float32(i + 1) * multiplier + bias
        assert_almost_equal(out_host[i], expected, atol=1e-4)

    inp_host.unsafe_free()
    out_host.unsafe_free()


def test_tile_tensor_cast() raises:
    """Static `TileTensor` captures: NPU port of `_cast_bf16_to_fp8`.

    The body captures two static-layout `TileTensor`s. Each tile is
    pointer-sized on Hexagon (the layout is comptime), so the host-side
    serializer in `_call_with_index_closure_npu` emits one FAM slot per
    tile, and the device wrapper patches each tile's pointer field from
    its FAM slot before invocation.
    """
    var ctx = DeviceContext(api="qaic")
    comptime H: Int = 4
    comptime W: Int = 8
    comptime N: Int = H * W

    comptime _InTT = TileTensor[
        DType.float32,
        RowMajorLayout[ComptimeInt[H], ComptimeInt[W]],
        MutUnsafeAnyOrigin,
    ]
    comptime _OutTT = TileTensor[
        DType.float16,
        RowMajorLayout[ComptimeInt[H], ComptimeInt[W]],
        MutUnsafeAnyOrigin,
    ]

    var in_dev = ctx.enqueue_create_buffer[DType.float32](N)
    var out_dev = ctx.enqueue_create_buffer[DType.float16](N)

    var in_host = alloc[Float32](N)
    for i in range(N):
        in_host[i] = Float32(i + 1) * 0.5
    ctx.enqueue_copy(in_dev, in_host)
    ctx.synchronize()
    in_host.unsafe_free()

    var in_tile = _InTT(in_dev, row_major[H, W]())
    var out_tile = _OutTT(out_dev, row_major[H, W]())

    # Drop the origin parameter so the unified closure can capture by
    # value — same trick the upstream `_cast_bf16_to_fp8` uses.
    var out_tt = out_tile.as_unsafe_any_origin()
    var in_tt = in_tile.as_unsafe_any_origin()

    def cast_fn[
        width: Int, alignment: Int = 1
    ](indices: StdCoord) {var out_tt, var in_tt}:
        var idx = rebind[IndexList[2]](coord_to_index_list(indices))
        var coord = Coord(Int(idx[0]), Int(idx[1]))
        out_tt.store[width=1](
            coord,
            in_tt.load[width=1](coord).cast[out_tt.dtype](),
        )

    elementwise[simd_width=1, target="npu"](
        cast_fn, StdCoord(IndexList[2](H, W)), ctx
    )
    ctx.synchronize()
    _ = in_dev^

    var out_host = alloc[Scalar[DType.float16]](N)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    for i in range(N):
        var expected = Float32(i + 1) * 0.5
        assert_almost_equal(Float32(out_host[i]), expected, atol=0.01)

    out_host.unsafe_free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
