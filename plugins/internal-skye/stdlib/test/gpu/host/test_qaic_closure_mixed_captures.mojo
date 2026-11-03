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
"""Test: closure capturing a heterogeneous mix of types.

Captures: `{UnsafePointer, Float32, static-TileTensor, Int32}` —
interleaves the three FAM-slot kinds (pointer / scalar / static-tile)
plus a second scalar to validate per-field FAM-slot ordering and that
the host-vs-device offset maps stay in sync across capture types.

Launches single-threaded; in-place writes are non-idempotent under
multiple threads.
"""

from common import print_separator, print_test_header

from std.memory import UnsafePointer, alloc
from std.testing import TestSuite, assert_equal

from layout import TileTensor
from layout.tile_layout import RowMajorLayout, row_major
from layout.coord import ComptimeInt

from max.gpu.host import DeviceContext, Dim


comptime N: Int = 8
comptime _TT = TileTensor[
    DType.float32,
    RowMajorLayout[ComptimeInt[1], ComptimeInt[N]],
    MutUnsafeAnyOrigin,
]


def test_qaic_closure_mixed_captures() raises:
    print_test_header("QAIC closure with mixed-type captures")
    print()

    var ctx = DeviceContext(api="qaic")
    var ptr_dev = ctx.enqueue_create_buffer[DType.float32](N)
    var tile_dev = ctx.enqueue_create_buffer[DType.float32](N)

    var host = alloc[Float32](N)
    for i in range(N):
        host[i] = Float32(i + 1)
    ctx.enqueue_copy(ptr_dev, host)
    ctx.enqueue_copy(tile_dev, host)
    ctx.synchronize()
    host.unsafe_free()

    var ptr = ptr_dev._device_ptr
    var add_offset: Float32 = 100.0
    var tile = _TT(tile_dev, row_major[1, N]())
    var multiplier: Int32 = 7

    def kernel() {var ptr, var add_offset, var tile, var multiplier}:
        for i in range(N):
            ptr[i] = ptr[i] + add_offset
            tile[0, i] = tile[0, i] * Float32(multiplier)

    var func = ctx.compile_function[type_of(kernel)]()
    ctx.enqueue_function(
        func,
        kernel,
        grid_dim=Dim(1),
        block_dim=Dim(1),
        num_cores=1,
        thread_mask=1,
    )
    ctx.synchronize()

    var out_ptr = alloc[Float32](N)
    var out_tile = alloc[Float32](N)
    ctx.enqueue_copy(out_ptr, ptr_dev)
    ctx.enqueue_copy(out_tile, tile_dev)
    ctx.synchronize()

    var errors = 0
    for i in range(N):
        var expected_ptr = Float32(i + 1) + add_offset
        var expected_tile = Float32(i + 1) * Float32(multiplier)
        if abs(out_ptr[i] - expected_ptr) > 0.001:
            errors += 1
            if errors <= 5:
                print(
                    "FAIL ptr[",
                    i,
                    "]: expected",
                    expected_ptr,
                    "got",
                    out_ptr[i],
                )
        if abs(out_tile[i] - expected_tile) > 0.001:
            errors += 1
            if errors <= 5:
                print(
                    "FAIL tile[",
                    i,
                    "]: expected",
                    expected_tile,
                    "got",
                    out_tile[i],
                )
    out_ptr.unsafe_free()
    out_tile.unsafe_free()

    print_separator()
    if errors == 0:
        print("TEST PASSED")
    else:
        print("TEST FAILED:", errors, "errors")
    print_separator()

    assert_equal(errors, 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
