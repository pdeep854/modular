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
"""End-to-end TileTensor NPU execution tests.

Validates the full round-trip for TileTensor kernels on the QAIC NPU:
  pack metadata -> dispatch -> verify output

Tests:
1. TileTensor fill: scalar fill value -> TileTensor (1 TileTensor, 1 ptr)
2. TileTensor scale-in-place: TileTensor * scalar -> same TileTensor
3. TileTensor vector-add: A + B -> C  (3 static TileTensors)

Static TileTensors (all ComptimeInt layout dims) use 1 FAM slot each — no
metadata buffer is needed because the Hexagon struct is pointer-only.

The QAIC driver embeds the kernel symbol name in the FlatBuffer CommandSet.
Mojo mangles generic function names into very long strings (3000+ chars) that
can push the CommandSet over the 8192-byte limit.  The plugin hashes long
symbol names to a 25-character identifier before loading the ELF, keeping all
CommandSets within the limit regardless of argument count or complexity.

To run:
    bazel test --config=plugin \\
        @plugin//runtime/test/mojo/host:test_qaic_tt_e2e.mojo.test
"""

from std.testing import TestSuite, assert_equal
from common import print_separator, print_test_header

from std.memory import alloc
from std.utils.coord import ComptimeInt

from layout import TileTensor
from layout.tile_layout import RowMajorLayout, row_major

from max.gpu.host import DeviceContext, Dim


# ===-----------------------------------------------------------------------===#
# Compile-time TileTensor type: 1x16, row-major, float32
# ===-----------------------------------------------------------------------===#

comptime _TT = TileTensor[
    DType.float32,
    RowMajorLayout[ComptimeInt[1], ComptimeInt[16]],
    MutUnsafeAnyOrigin,
]

comptime N: Int = 16


# ===-----------------------------------------------------------------------===#
# Kernel 1: fill TileTensor with a constant (1 TileTensor + 1 UnsafePointer)
# ===-----------------------------------------------------------------------===#


def tt_fill(
    output: _TT,
    value: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    """Fill every element of output with value[]."""
    var v = value[]
    for i in range(N):
        output[0, i] = v


# ===-----------------------------------------------------------------------===#
# Kernel 2: scale TileTensor in-place (1 TileTensor + 1 UnsafePointer)
# ===-----------------------------------------------------------------------===#


def tt_scale_inplace(
    buf: _TT,
    scale: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    """Multiply every element of buf by scale[] in-place."""
    var s = scale[]
    for i in range(N):
        buf[0, i] = buf[0, i] * s


# ===-----------------------------------------------------------------------===#
# Kernel 3: vector add (3 TileTensors)
# ===-----------------------------------------------------------------------===#


def tt_vector_add(
    a: _TT,
    b: _TT,
    result: _TT,
):
    """Compute result[i] = a[i] + b[i] for all i."""
    for i in range(N):
        result[0, i] = a[0, i] + b[0, i]


# ===-----------------------------------------------------------------------===#
# Tests
# ===-----------------------------------------------------------------------===#


def run_tt_fill() raises -> Bool:
    """Test 1: fill TileTensor with constant (1 TileTensor + 1 ptr arg)."""
    print("  Setting up buffers...")

    var ctx = DeviceContext(api="qaic")

    # Allocate device buffers
    var output_dev = ctx.enqueue_create_buffer[DType.float32](N)
    var value_dev = ctx.enqueue_create_buffer[DType.float32](1)

    # Initialise fill value and copy to device
    var value_host = alloc[Float32](1)
    value_host[0] = Float32(42.0)

    ctx.enqueue_copy(value_dev, value_host)
    ctx.synchronize()

    value_host.unsafe_free()

    # Construct TileTensor from DeviceBuffer
    var layout = row_major[1, 16]()
    var output_tt = _TT(output_dev, layout)

    # Dispatch
    print("  Dispatching tt_fill...")
    var func = ctx.compile_function[tt_fill]()
    ctx.enqueue_function(
        func, output_tt, value_dev, grid_dim=Dim(1), block_dim=Dim(0x01)
    )
    ctx.synchronize()

    # Read back and verify
    var result_host = alloc[Float32](N)
    ctx.enqueue_copy(result_host, output_dev)
    ctx.synchronize()

    var errors = 0
    for i in range(N):
        if abs(result_host[i] - Float32(42.0)) > Float32(0.001):
            print(
                "    MISMATCH at",
                i,
                ": got",
                result_host[i],
                "expected 42.0",
            )
            errors += 1

    result_host.unsafe_free()

    if errors == 0:
        print("  PASS: all", N, "elements == 42.0")
        return True
    else:
        print("  FAIL:", errors, "mismatches")
        return False


def run_tt_scale_inplace() raises -> Bool:
    """Test 2: scale TileTensor in-place (1 TileTensor + 1 ptr, read+write)."""
    print("  Setting up buffers...")

    var ctx = DeviceContext(api="qaic")

    # Allocate device buffers
    var buf_dev = ctx.enqueue_create_buffer[DType.float32](N)
    var scale_dev = ctx.enqueue_create_buffer[DType.float32](1)

    # Initialise input data and copy to device
    var buf_host = alloc[Float32](N)
    var scale_host = alloc[Float32](1)

    for i in range(N):
        buf_host[i] = Float32(i + 1)  # 1.0 .. 16.0
    scale_host[0] = Float32(2.0)

    ctx.enqueue_copy(buf_dev, buf_host)
    ctx.enqueue_copy(scale_dev, scale_host)
    ctx.synchronize()

    buf_host.unsafe_free()
    scale_host.unsafe_free()

    # Construct TileTensor from DeviceBuffer
    var layout = row_major[1, 16]()
    var buf_tt = _TT(buf_dev, layout)

    # Dispatch
    print("  Dispatching tt_scale_inplace...")
    var func = ctx.compile_function[tt_scale_inplace]()
    ctx.enqueue_function(
        func, buf_tt, scale_dev, grid_dim=Dim(1), block_dim=Dim(0x01)
    )
    ctx.synchronize()

    # Read back and verify  (expected: 2, 4, 6, ... 32)
    var result_host = alloc[Float32](N)
    ctx.enqueue_copy(result_host, buf_dev)
    ctx.synchronize()

    var errors = 0
    for i in range(N):
        var expected = Float32(i + 1) * Float32(2.0)
        if abs(result_host[i] - expected) > Float32(0.001):
            print(
                "    MISMATCH at",
                i,
                ": got",
                result_host[i],
                "expected",
                expected,
            )
            errors += 1

    result_host.unsafe_free()

    if errors == 0:
        print("  PASS: all", N, "elements correct")
        return True
    else:
        print("  FAIL:", errors, "mismatches")
        return False


def run_tt_vector_add() raises -> Bool:
    """Test 3: vector-add across 3 TileTensors."""
    print("  Setting up buffers...")

    var ctx = DeviceContext(api="qaic")

    var a_dev = ctx.enqueue_create_buffer[DType.float32](N)
    var b_dev = ctx.enqueue_create_buffer[DType.float32](N)
    var result_dev = ctx.enqueue_create_buffer[DType.float32](N)

    var a_host = alloc[Float32](N)
    var b_host = alloc[Float32](N)
    for i in range(N):
        a_host[i] = Float32(i + 1)  # 1 .. 16
        b_host[i] = Float32((i + 1) * 2)  # 2 .. 32

    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.synchronize()

    a_host.unsafe_free()
    b_host.unsafe_free()

    var layout = row_major[1, 16]()
    var a_tt = _TT(a_dev, layout)
    var b_tt = _TT(b_dev, layout)
    var result_tt = _TT(result_dev, layout)

    print("  Dispatching tt_vector_add (3 static TileTensors)...")
    var func = ctx.compile_function[tt_vector_add]()
    ctx.enqueue_function(
        func,
        a_tt,
        b_tt,
        result_tt,
        grid_dim=Dim(1),
        block_dim=Dim(0x01),
    )
    ctx.synchronize()

    var result_host = alloc[Float32](N)
    ctx.enqueue_copy(result_host, result_dev)
    ctx.synchronize()

    var errors = 0
    for i in range(N):
        var expected = Float32(i + 1) + Float32((i + 1) * 2)
        if abs(result_host[i] - expected) > Float32(0.001):
            print(
                "    MISMATCH at",
                i,
                ": got",
                result_host[i],
                "expected",
                expected,
            )
            errors += 1

    result_host.unsafe_free()

    if errors == 0:
        print("  PASS: all", N, "elements correct")
        return True
    else:
        print("  FAIL:", errors, "mismatches")
        return False


# ===-----------------------------------------------------------------------===#
# Main
# ===-----------------------------------------------------------------------===#


def test_qaic_tt_e2e() raises:
    print_test_header("TileTensor E2E NPU Execution Tests")
    print()

    var total = 0
    var passed = 0

    total += 1
    print("Test 1: tt_fill (1 TileTensor output + scalar fill value)")
    if run_tt_fill():
        passed += 1
    print()

    total += 1
    print("Test 2: tt_scale_inplace (1 TileTensor read+write + scalar scale)")
    if run_tt_scale_inplace():
        passed += 1
    print()

    total += 1
    print("Test 3: tt_vector_add (3 static TileTensors)")
    if run_tt_vector_add():
        passed += 1
    print()

    print_separator()
    print("Results:", passed, "/", total, "tests passed")
    if passed == total:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print_separator()

    assert_equal(passed, total)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
