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
"""End-to-end test: 3-static-TileTensor NPU kernel dispatch.

Validates that a kernel with 3 static TileTensor arguments dispatches
successfully and produces correct output.

The QAIC driver embeds the kernel symbol name in the FlatBuffer CommandSet.
Mojo-mangled symbol names for generic functions with complex type parameters
can exceed 3000 characters, pushing the CommandSet over the 8192-byte limit.
The plugin addresses this by hashing long symbol names to a 25-char identifier
before loading the ELF, keeping all CommandSets within the limit regardless of
the number or complexity of TileTensor args.

Kernel: element-wise vector add C[i] = A[i] + B[i]

To run:
    bazel test --config=plugin \\
        @plugin//runtime/test/mojo/host:test_qaic_tt_3arg.mojo.test
"""

from std.testing import TestSuite, assert_equal
from common import print_separator, print_test_header

from std.memory import alloc

from layout import TileTensor
from layout.tile_layout import RowMajorLayout, row_major
from std.utils.coord import ComptimeInt

from max.gpu.host import DeviceContext, Dim


# ===-----------------------------------------------------------------------===#
# Compile-time TileTensor type: 1x16, row-major, float32 (static layout)
# ===-----------------------------------------------------------------------===#

comptime _TT = TileTensor[
    DType.float32,
    RowMajorLayout[ComptimeInt[1], ComptimeInt[16]],
    MutUnsafeAnyOrigin,
]

comptime N: Int = 16


# ===-----------------------------------------------------------------------===#
# Kernel: element-wise addition using 3 TileTensor args
# ===-----------------------------------------------------------------------===#


def tt_vector_add(a: _TT, b: _TT, c: _TT):
    """Compute c[i] = a[i] + b[i] for all i."""
    for i in range(N):
        c[0, i] = a[0, i] + b[0, i]


# ===-----------------------------------------------------------------------===#
# Test
# ===-----------------------------------------------------------------------===#


def run_tt_3arg_vector_add() raises -> Bool:
    """Dispatch a 3-static-TileTensor kernel and verify results."""
    print("  Setting up buffers...")

    var ctx = DeviceContext(api="qaic")

    # Allocate device buffers
    var a_dev = ctx.enqueue_create_buffer[DType.float32](N)
    var b_dev = ctx.enqueue_create_buffer[DType.float32](N)
    var c_dev = ctx.enqueue_create_buffer[DType.float32](N)

    # Initialise inputs on host
    var a_host = alloc[Float32](N)
    var b_host = alloc[Float32](N)
    for i in range(N):
        a_host[i] = Float32(i + 1)  # 1, 2, ..., 16
        b_host[i] = Float32(N - i)  # 16, 15, ..., 1

    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.synchronize()

    a_host.unsafe_free()
    b_host.unsafe_free()

    # Construct TileTensors
    var layout = row_major[1, 16]()
    var a_tt = _TT(a_dev, layout)
    var b_tt = _TT(b_dev, layout)
    var c_tt = _TT(c_dev, layout)

    # Dispatch
    print("  Dispatching tt_vector_add (3 static TileTensors)...")
    var func = ctx.compile_function[tt_vector_add]()
    ctx.enqueue_function(
        func, a_tt, b_tt, c_tt, grid_dim=Dim(1), block_dim=Dim(0x01)
    )
    ctx.synchronize()

    # Read back and verify: expected c[i] == (i+1) + (N-i) == N+1 for all i
    var result_host = alloc[Float32](N)
    ctx.enqueue_copy(result_host, c_dev)
    ctx.synchronize()

    var expected = Float32(N + 1)
    var errors = 0
    for i in range(N):
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
        print("  PASS: all", N, "elements ==", expected)
        return True
    else:
        print("  FAIL:", errors, "mismatches")
        return False


# ===-----------------------------------------------------------------------===#
# Main
# ===-----------------------------------------------------------------------===#


def test_qaic_tt_3arg() raises:
    print_test_header("3-Arg Static TileTensor NPU Dispatch Test")
    print()

    var total = 0
    var passed = 0

    total += 1
    print("Test 1: tt_vector_add (3 static TileTensors, a+b->c)")
    if run_tt_3arg_vector_add():
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
