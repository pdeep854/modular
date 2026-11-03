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
"""NPU capture serialization for named register-passable adapters.

A compiler-synthesized closure (`def {var ...}`) nests its captures in a
field-0 sub-struct; a named `@fieldwise_init` RegisterPassable adapter that
conforms to the elementwise body trait stores its captures as its OWN fields
(no sub-struct). The NPU host->device capture walkers
(serialize / count / fill / patch) must select the matching "captures" view for
each — otherwise an adapter's pointer fields past field 0 (here `out_ptr`) go
unpatched on device and the store faults (QS_ERROR).

`_npu_is_closure_with_captures_substruct` is that discriminator. These tests run
the same elementwise computation two ways — through a closure and through a
named adapter holding equivalent captures — and assert identical, correct
output, covering both sides of the discriminator end-to-end on real NPU.
"""

from max.algorithm import elementwise
from max.gpu.host import DeviceContext
from std.memory import alloc, UnsafePointer
from std.testing import TestSuite, assert_almost_equal
from std.utils.coord import Coord, coord_to_index_list
from std.utils.index import IndexList
from std.builtin.dtype import DType


# Named elementwise adapter whose captures are its OWN fields (no field-0
# captures sub-struct), mirroring the graph compiler's `_ElementwiseFusionAdapter`
# topology. `out_ptr` is past field 0, so a field-0-only capture walk would leave
# it unpatched on device; the whole-struct view selected by
# `_npu_is_closure_with_captures_substruct` patches both pointers.
@fieldwise_init
struct _ScaleAdapter(
    ImplicitlyCopyable,
    RegisterPassable,
    def[width: Int, alignment: Int = 1](Coord) -> None,
):
    @__allow_legacy_any_origin_fields
    var in_ptr: UnsafePointer[Scalar[DType.float32], MutUnsafeAnyOrigin]
    var multiplier: Float32

    @__allow_legacy_any_origin_fields
    var out_ptr: UnsafePointer[Scalar[DType.float32], MutUnsafeAnyOrigin]

    @always_inline
    def __call__[width: Int, alignment: Int = 1](self, indices: Coord):
        var idx = rebind[IndexList[1]](coord_to_index_list(indices))
        self.out_ptr[idx[0]] = self.in_ptr[idx[0]] * self.multiplier


def test_named_adapter_captures_npu() raises:
    """Named adapter: captures are the struct's own fields (whole-struct view).

    `out_ptr` is field 2; treating only field 0 as the captures leaves it
    unpatched and the device store hits a null pointer (QS_ERROR). The
    discriminator routes the whole adapter struct through serialize / count /
    fill / patch, so both pointer fields arrive on device.
    """
    var ctx = DeviceContext(api="qaic")
    comptime N: Int = 64
    var multiplier: Float32 = 3.0

    var inp_dev = ctx.enqueue_create_buffer[DType.float32](N)
    var out_dev = ctx.enqueue_create_buffer[DType.float32](N)

    var inp_host = alloc[Float32](N)
    for i in range(N):
        inp_host[i] = Float32(i + 1)
    ctx.enqueue_copy(inp_dev, inp_host)
    ctx.synchronize()

    var adapter = _ScaleAdapter(
        inp_dev._device_ptr.as_unsafe_any_origin(),
        multiplier,
        out_dev._device_ptr.as_unsafe_any_origin(),
    )
    elementwise[simd_width=1, target="npu"](
        adapter, Coord(IndexList[1](N)), ctx
    )
    ctx.synchronize()

    _ = inp_dev

    var out_host = alloc[Float32](N)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()
    for i in range(N):
        assert_almost_equal(out_host[i], Float32(i + 1) * multiplier, atol=1e-4)

    inp_host.unsafe_free()
    out_host.unsafe_free()


def test_closure_captures_npu() raises:
    """Closure path (field-0 captures sub-struct): same computation, unchanged
    by the discriminator (`_npu_is_closure_with_captures_substruct` is True).
    Guards that adding the adapter branch does not regress the closure walk.
    """
    var ctx = DeviceContext(api="qaic")
    comptime N: Int = 64
    var multiplier: Float32 = 3.0

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
    ](indices: Coord) {var inp_ptr, var out_ptr, var multiplier}:
        var idx = rebind[IndexList[1]](coord_to_index_list(indices))
        out_ptr[idx[0]] = inp_ptr[idx[0]] * multiplier

    elementwise[simd_width=1, target="npu"](body, Coord(IndexList[1](N)), ctx)
    ctx.synchronize()

    _ = inp_dev

    var out_host = alloc[Float32](N)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()
    for i in range(N):
        assert_almost_equal(out_host[i], Float32(i + 1) * multiplier, atol=1e-4)

    inp_host.unsafe_free()
    out_host.unsafe_free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
