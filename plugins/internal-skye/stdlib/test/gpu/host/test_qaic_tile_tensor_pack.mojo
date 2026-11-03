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
"""Test host-side TileTensor packing for QAIC NPU.

Validates:
1. TileTensor metadata size computation for static and dynamic layouts
2. Compile-time type detection and FAM slot counting

Static TileTensors (all ComptimeInt dims) carry zero runtime layout bytes —
the layout is entirely encoded in the type.  Their host size is 8 bytes (ptr
only), so metadata_size == 0 and device_meta_size == 4.

Dynamic TileTensors (Scalar dims) carry runtime shape/stride data.  Their
host size is > 8 bytes, so metadata_size > 0 and device_meta_size > 4.

To run:
    bazel test --config=plugin @plugin//runtime/test/mojo/host:test_qaic_tile_tensor_pack.mojo.test
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true
from common import print_separator, print_test_header

from std.memory import UnsafePointer
from std.sys import size_of

from layout import TileTensor
from layout.tile_layout import RowMajorLayout
from layout.coord import ComptimeInt

from std._plugin.qaic.host.target import (
    npu_slot_count,
    looks_like_static_tile_tensor,
    looks_like_tile_tensor,
)


def run_static_tile_tensor_metadata_size() raises:
    """Verify static TileTensor metadata size.

    A fully-static TileTensor (all ComptimeInt dims) has zero runtime layout
    bytes.  On the host it is just a pointer (8 bytes).  The device metadata
    buffer is 4 bytes (the ptr_placeholder alone).
    """
    print("  Testing static TileTensor metadata size...")

    comptime TT = TileTensor[
        DType.float32,
        RowMajorLayout[ComptimeInt[4], ComptimeInt[8]],
        MutUnsafeAnyOrigin,
    ]
    var tt_size = size_of[TT]()
    var host_ptr_size = 8
    var metadata_size = tt_size - host_ptr_size
    var device_meta_size = 4 + metadata_size

    print("    TileTensor size:", tt_size, "bytes")
    print("    Metadata (no ptr):", metadata_size, "bytes")
    print("    Device metadata (with placeholder):", device_meta_size, "bytes")

    # Static layout carries no runtime data — layout is baked into the type.
    assert_equal(tt_size, 8)
    assert_equal(metadata_size, 0)
    assert_equal(device_meta_size, 4)


def run_dynamic_tile_tensor_metadata_size() raises:
    """Verify dynamic TileTensor metadata size.

    A TileTensor with Scalar dims carries runtime shape/stride data.
    Its host size is > 8 bytes, so metadata_size > 0.
    """
    print("  Testing dynamic TileTensor metadata size...")

    comptime TT = TileTensor[
        DType.float32,
        RowMajorLayout[Scalar[DType.int32], Scalar[DType.int32]],
        MutUnsafeAnyOrigin,
    ]
    var tt_size = size_of[TT]()
    var host_ptr_size = 8
    var metadata_size = tt_size - host_ptr_size
    var device_meta_size = 4 + metadata_size

    print("    TileTensor size:", tt_size, "bytes")
    print("    Metadata (no ptr):", metadata_size, "bytes")
    print("    Device metadata (with placeholder):", device_meta_size, "bytes")

    assert_true(metadata_size > 0)
    assert_true(tt_size > 8)


def run_type_detection() raises:
    """Verify type detection helpers and FAM slot counts.

    Static TileTensors use 1 FAM slot (no metadata buffer needed).
    Dynamic TileTensors use 2 FAM slots (data + metadata buffer).
    UnsafePointer uses 1 FAM slot.
    """
    print("  Testing type detection and slot counts...")

    comptime TT_STATIC = TileTensor[
        DType.float32,
        RowMajorLayout[ComptimeInt[4], ComptimeInt[8]],
        MutUnsafeAnyOrigin,
    ]
    comptime TT_DYNAMIC = TileTensor[
        DType.float32,
        RowMajorLayout[Scalar[DType.int32], Scalar[DType.int32]],
        MutUnsafeAnyOrigin,
    ]
    comptime PTR = UnsafePointer[Float32, MutUnsafeAnyOrigin]

    assert_true(looks_like_tile_tensor[TT_STATIC]())
    assert_true(looks_like_static_tile_tensor[TT_STATIC]())
    assert_false(looks_like_static_tile_tensor[TT_DYNAMIC]())
    assert_false(looks_like_tile_tensor[PTR]())
    assert_false(looks_like_static_tile_tensor[PTR]())
    assert_equal(npu_slot_count[TT_STATIC](), 1)
    assert_equal(npu_slot_count[TT_DYNAMIC](), 2)
    assert_equal(npu_slot_count[PTR](), 1)


def test_qaic_tile_tensor_pack() raises:
    print_test_header("QAIC TileTensor Packing Tests")
    print()

    print("Test 1: Static TileTensor metadata size")
    run_static_tile_tensor_metadata_size()
    print()

    print("Test 2: Dynamic TileTensor metadata size")
    run_dynamic_tile_tensor_metadata_size()
    print()

    print("Test 3: Type detection and slot counts")
    run_type_detection()
    print()

    print_separator()
    print("ALL TESTS PASSED")
    print_separator()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
