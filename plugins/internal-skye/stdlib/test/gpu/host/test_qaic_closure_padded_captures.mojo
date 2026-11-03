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
"""Regression test: closure captures with diverging host/device layout.

The captures `{UnsafePointer, UInt8, UnsafePointer}` lay out differently on
host (8-byte pointer alignment) vs Hexagon (4-byte pointer alignment):

    host:    ptr1@0   marker@8   ptr2@16   (24 bytes incl. trailing pad)
    device:  ptr1@0   marker@4   ptr2@8    (12 bytes)

Before the layout fix, the host serializer packed the blob contiguously
(ptr1@0, marker@4, ptr2@5 = 9 bytes) while the device wrapper read at the
target's natural offsets — so ptr2 was read from out-of-bounds, dereference
went to a bogus address, and writes through it never reached buf2.

This test passes only when the host writes at the device target's offsets.
"""

from common import print_separator, print_test_header

from std.memory import UnsafePointer, alloc
from std.testing import TestSuite, assert_equal

from max.gpu.host import DeviceContext, Dim


def test_qaic_closure_padded_captures() raises:
    print_test_header("QAIC closure with padded captures")
    print()

    var ctx = DeviceContext(api="qaic")

    var buf1 = ctx.enqueue_create_buffer[DType.float32](2)
    var buf2 = ctx.enqueue_create_buffer[DType.float32](2)

    # Pre-fill with sentinels so missed writes are detectable as -999.
    var sentinel = alloc[Float32](2)
    sentinel[0] = -999.0
    sentinel[1] = -999.0
    ctx.enqueue_copy(buf1, sentinel)
    ctx.enqueue_copy(buf2, sentinel)
    ctx.synchronize()
    sentinel.unsafe_free()

    var ptr1 = buf1._device_ptr
    var marker: UInt8 = UInt8(0xAB)
    var ptr2 = buf2._device_ptr

    # Captures order matters: pointer, byte, pointer surfaces alignment
    # divergence between host (8B) and device (4B) pointer alignment.
    def kernel() {var ptr1, var marker, var ptr2}:
        var v = Float32(Int32(marker))
        ptr1[0] = v
        ptr1[1] = v + 1.0
        ptr2[0] = v * 2.0
        ptr2[1] = v * 3.0

    print("Compiling...")
    var func = ctx.compile_function[type_of(kernel)]()

    print("Launching...")
    ctx.enqueue_function(func, kernel, grid_dim=Dim(1), block_dim=Dim(1))
    ctx.synchronize()

    var host1 = alloc[Float32](2)
    var host2 = alloc[Float32](2)
    ctx.enqueue_copy(host1, buf1)
    ctx.enqueue_copy(host2, buf2)
    ctx.synchronize()

    var expected = Float32(0xAB)  # 171.0
    var errors = 0

    if abs(host1[0] - expected) > 0.001:
        errors += 1
        print("FAIL buf1[0]: expected", expected, "got", host1[0])
    if abs(host1[1] - (expected + 1.0)) > 0.001:
        errors += 1
        print("FAIL buf1[1]: expected", expected + 1.0, "got", host1[1])
    if abs(host2[0] - (expected * 2.0)) > 0.001:
        errors += 1
        print("FAIL buf2[0]: expected", expected * 2.0, "got", host2[0])
    if abs(host2[1] - (expected * 3.0)) > 0.001:
        errors += 1
        print("FAIL buf2[1]: expected", expected * 3.0, "got", host2[1])

    host1.unsafe_free()
    host2.unsafe_free()

    print_separator()
    if errors == 0:
        print("TEST PASSED")
    else:
        print("TEST FAILED:", errors, "errors")
    print_separator()

    assert_equal(errors, 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
