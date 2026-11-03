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
"""Gating test for nested `` closure capture on NPU.

The refactored `foreach` builds a `RegisterPassable` wrapper that captures
the user body (itself a `RegisterPassable` closure) plus the output
pointer and strides, then forwards to `elementwise[target="npu"]`. This
test isolates that pattern: an inner pure-compute closure captured by an
outer store closure, both `RegisterPassable`. Pass criterion is that
`compile_function` / `enqueue_function` accept the outer closure without
tripping MOCO-3654.
"""

from max.algorithm import elementwise
from max.gpu.host import DeviceContext
from std.memory import alloc
from std.testing import TestSuite, assert_almost_equal
from std.utils.index import IndexList
from std.utils.coord import Coord, coord_to_index_list

from std.builtin.dtype import DType


def test_nested_capture_rank_1() raises:
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

    # Inner: pure compute, returns a SIMD value. Mirrors the user-body
    # shape that the refactored `foreach` accepts.
    def inner[
        width: Int, alignment: Int = 1
    ](indices: IndexList[1]) {var inp_ptr, var multiplier, var bias} -> SIMD[
        DType.float32, width
    ]:
        return inp_ptr[indices[0]] * multiplier + bias

    # Outer: capture-all `{var}`. The migration doc's _stencil_impl_gpu
    # precedent uses this form for kernels that reference RegisterPassable
    # closures from the enclosing scope.
    def outer[width: Int, alignment: Int = 1](indices: Coord) {var}:
        var idx = rebind[IndexList[1]](coord_to_index_list(indices))
        var v = inner[width, alignment](idx)
        out_ptr[idx[0]] = v[0]

    elementwise[simd_width=1, target="npu"](outer, Coord(IndexList[1](N)), ctx)
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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
