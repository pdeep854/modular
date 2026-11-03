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
"""QAIC tile test: proves tile variants work correctly inside NPU kernels.

Tests four tile algorithm variants via compile_function + enqueue_function:
1. Static tile (List[Int] tile sizes): 100 elements with [32, 8, 1] tiles
2. Dynamic tile (*args tile sizes): 100 elements with 32, 8, 1 runtime tiles
3. tile_and_unswitch (static): 100 elements with [32, 8] tiles, inner vs residue
4. tile_middle_unswitch_boundaries: 24 elements, left/middle/right partitioning

All tile algorithms are target-agnostic control flow (comptime loops + while
loops) with no target-specific branches. These tests prove the tiling logic
compiles and partitions work correctly on Hexagon via QAIC JIT.

To run (requires QAIC hardware):
    mojo test_qaic_tile.mojo
"""

from std.algorithm.functional import (
    tile,
    tile_and_unswitch,
    tile_middle_unswitch_boundaries,
)
from std.memory import UnsafePointer, alloc
from std.testing import TestSuite, assert_true, assert_almost_equal

from max.gpu.host import DeviceContext
from std._plugin.qaic.host.target import npu_work_range

from qaic_test_utils import (
    alloc_sequential,
    compile_and_run_unary,
)


# ===-----------------------------------------------------------------------===#
# Kernels
# ===-----------------------------------------------------------------------===#


def static_tile_kernel_100(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    """Static tile: process input in tiles of [32, 8, 1], output[i] = input[i] * 3.
    """

    def work[tile_size: Int](offset: Int) {imm}:
        for i in range(tile_size):
            output[offset + i] = input[offset + i] * 3.0

    tile[[32, 8, 1]](0, 100, work)


def dynamic_tile_kernel_100(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    """Dynamic tile: process input in tiles of 32, 8, 1 (runtime), output[i] = input[i] * 5.
    """

    @always_inline
    def work(offset: Int, tile_size: Int) {imm}:
        for i in range(tile_size):
            output[offset + i] = input[offset + i] * 5.0

    tile(0, 100, 32, 8, 1, workgroup_function=work)


def tile_unswitch_kernel_100(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    """Tile and unswitch: inner tiles multiply by 2, residue tiles multiply by 10.

    With tile sizes [32, 8] on 100 elements:
    - 32: handles 96 elements (indices 0..95), is_inner=True
    - 8: cannot fit remaining 4 elements (4 < 8), skips
    - Residue: 4 elements (indices 96..99), is_inner=False
    """

    @always_inline
    def work[
        tile_size: Int, is_inner: Bool
    ](offset: Int, upperbound: Int) {imm}:
        for i in range(tile_size):
            if is_inner:
                output[offset + i] = input[offset + i] * 2.0
            else:
                # Residue -- only process up to upperbound
                if offset + i < upperbound:
                    output[offset + i] = input[offset + i] * 10.0

    tile_and_unswitch[[32, 8]](0, 100, work)


def tile_middle_kernel_24(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    """Tile middle unswitch boundaries: left/right get +100, middle gets *2."""

    @always_inline
    def work[tile_size: Int, is_boundary: Bool](offset: Int) {imm}:
        for i in range(tile_size):
            if is_boundary:
                output[offset + i] = input[offset + i] + 100.0
            else:
                output[offset + i] = input[offset + i] * 2.0

    tile_middle_unswitch_boundaries[[8, 1]](0, 4, 20, 24, work)


# ===-----------------------------------------------------------------------===#
# Test Runners
# ===-----------------------------------------------------------------------===#


def run_static_tile_test() raises:
    """Test static tile on QAIC (100 elements, tiles [32, 8, 1])."""
    var n = 100
    var ctx = DeviceContext(device_id=0, api="qaic")

    var input_host = alloc_sequential[DType.float32](n)
    var output_host = compile_and_run_unary[static_tile_kernel_100](
        ctx, input_host, n
    )

    # Verify: output[i] == (i + 1) * 3.0
    for i in range(n):
        var expected = Float32((i + 1) * 3)
        assert_almost_equal(output_host[i], expected, atol=0.001)

    input_host.unsafe_free()
    output_host.unsafe_free()


def run_dynamic_tile_test() raises:
    """Test dynamic tile on QAIC (100 elements, tiles 32, 8, 1)."""
    var n = 100
    var ctx = DeviceContext(device_id=0, api="qaic")

    var input_host = alloc_sequential[DType.float32](n)
    var output_host = compile_and_run_unary[dynamic_tile_kernel_100](
        ctx, input_host, n
    )

    # Verify: output[i] == (i + 1) * 5.0
    for i in range(n):
        var expected = Float32((i + 1) * 5)
        assert_almost_equal(output_host[i], expected, atol=0.001)

    input_host.unsafe_free()
    output_host.unsafe_free()


def run_tile_unswitch_test() raises:
    """Test tile_and_unswitch on QAIC (100 elements, tiles [32, 8])."""
    var n = 100
    var ctx = DeviceContext(device_id=0, api="qaic")

    var input_host = alloc_sequential[DType.float32](n)
    var output_host = compile_and_run_unary[tile_unswitch_kernel_100](
        ctx, input_host, n
    )

    # Verify: inner tiles (0..95) get *2, residue (96..99) get *10
    for i in range(n):
        var value = Float32(i + 1)
        var expected: Float32
        if i < 96:
            expected = value * 2.0
        else:
            expected = value * 10.0
        assert_almost_equal(output_host[i], expected, atol=0.001)

    input_host.unsafe_free()
    output_host.unsafe_free()


def run_tile_middle_test() raises:
    """Test tile_middle_unswitch_boundaries on QAIC (24 elements)."""
    var n = 24
    var ctx = DeviceContext(device_id=0, api="qaic")

    var input_host = alloc_sequential[DType.float32](n)
    var output_host = compile_and_run_unary[tile_middle_kernel_24](
        ctx, input_host, n
    )

    # Verify: left [0,4) and right [20,24) get +100, middle [4,20) gets *2
    for i in range(n):
        var value = Float32(i + 1)
        var expected: Float32
        if i < 4:
            expected = value + 100.0
        elif i < 20:
            expected = value * 2.0
        else:
            expected = value + 100.0
        assert_almost_equal(output_host[i], expected, atol=0.001)

    input_host.unsafe_free()
    output_host.unsafe_free()


# ===-----------------------------------------------------------------------===#
# Main
# ===-----------------------------------------------------------------------===#


def test_qaic_tile() raises:
    var all_passed = True

    # Test 1: Static tile
    print("Test 1: Static tile (100 elements, tiles [32, 8, 1])...")
    try:
        run_static_tile_test()
        print("  PASSED")
    except e:
        print("  FAILED:", e)
        all_passed = False

    # Test 2: Dynamic tile
    print("Test 2: Dynamic tile (100 elements, tiles 32, 8, 1)...")
    try:
        run_dynamic_tile_test()
        print("  PASSED")
    except e:
        print("  FAILED:", e)
        all_passed = False

    # Test 3: tile_and_unswitch
    print(
        "Test 3: tile_and_unswitch (100 elements, tiles [32,8], inner *2 /"
        " residue *10)..."
    )
    try:
        run_tile_unswitch_test()
        print("  PASSED")
    except e:
        print("  FAILED:", e)
        all_passed = False

    # Test 4: tile_middle_unswitch_boundaries
    print(
        "Test 4: tile_middle_unswitch_boundaries (24 elements,"
        " left/middle/right)..."
    )
    try:
        run_tile_middle_test()
        print("  PASSED")
    except e:
        print("  FAILED:", e)
        all_passed = False

    # Summary
    print()
    if all_passed:
        print("TILE TEST PASSED: all tests correct")
    else:
        print("TILE TEST FAILED")
    assert_true(all_passed)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
