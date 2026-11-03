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
"""Test: closure capturing a static TileTensor.

Static TileTensors have all compile-time layout dimensions, so the struct
is exactly one pointer in size on each target (8 B host, 4 B Hexagon).
The host serializer detects this via `looks_like_static_tile_tensor` and
emits a single FAM slot for the pointer; the device wrapper patches the
field with the device address.

Launches single-threaded (`num_cores=1, thread_mask=1`) so the
non-idempotent in-place scaling produces a deterministic result without
races.
"""

from common import print_separator, print_test_header

from std.memory import alloc
from std.testing import TestSuite, assert_equal

from layout import TileTensor
from layout.tile_layout import RowMajorLayout, row_major
from layout.coord import ComptimeInt

from max.gpu.host import DeviceContext, Dim


comptime N: Int = 16
comptime _TT = TileTensor[
    DType.float32,
    RowMajorLayout[ComptimeInt[1], ComptimeInt[N]],
    MutUnsafeAnyOrigin,
]


def test_qaic_closure_static_tile() raises:
    print_test_header("QAIC closure capturing a static TileTensor")
    print()

    var ctx = DeviceContext(api="qaic")
    var dev = ctx.enqueue_create_buffer[DType.float32](N)

    var host = alloc[Float32](N)
    for i in range(N):
        host[i] = Float32(i + 1)
    ctx.enqueue_copy(dev, host)
    ctx.synchronize()
    host.unsafe_free()

    var tile = _TT(dev, row_major[1, N]())
    var scale: Float32 = 3.0

    def kernel() {var tile, var scale}:
        for i in range(N):
            tile[0, i] = tile[0, i] * scale

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

    var out = alloc[Float32](N)
    ctx.enqueue_copy(out, dev)
    ctx.synchronize()

    var errors = 0
    for i in range(N):
        var expected = Float32(i + 1) * scale
        if abs(out[i] - expected) > 0.001:
            errors += 1
            if errors <= 5:
                print("FAIL at", i, ": expected", expected, "got", out[i])
    out.unsafe_free()

    print_separator()
    if errors == 0:
        print("TEST PASSED")
    else:
        print("TEST FAILED:", errors, "errors")
    print_separator()

    assert_equal(errors, 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
