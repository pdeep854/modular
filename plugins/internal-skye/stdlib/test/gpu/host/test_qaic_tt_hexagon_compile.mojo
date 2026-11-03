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
"""Tests for cross-compiling TileTensor kernels to Hexagon.

These tests verify that compile_info_npu can produce valid Hexagon ELF
when the kernel signature includes TileTensor parameters.

Note: These tests may be blocked by the same LLVM BumpPtrAllocator assertion
crash that affects LayoutTensor cross-compilation (see
test_qaic_lt_hexagon_compile.mojo). The crash reproduces when
cross-compiling tensor-parameterized functions for the Hexagon target.

To run:
    bazel test --config=plugin \
        @plugin//runtime/test/mojo/host:test_qaic_tt_hexagon_compile.mojo.test
"""

from std.testing import assert_not_equal
from common import print_separator, print_test_header

from std.memory import UnsafePointer

from layout import TileTensor
from layout.tile_layout import RowMajorLayout
from std.utils.coord import ComptimeInt

from std.compile import compile_info
from max.gpu.host import get_gpu_target
from std._plugin.qaic.host.target import make_npu_wrapper


# ===-----------------------------------------------------------------------===#
# Concrete TileTensor type used across all kernels below.
# ===-----------------------------------------------------------------------===#

comptime _TT = TileTensor[
    DType.float32,
    RowMajorLayout[ComptimeInt[1], ComptimeInt[16]],
    MutUnsafeAnyOrigin,
]


# ===-----------------------------------------------------------------------===#
# Minimal TileTensor kernels — simplest possible test cases.
# ===-----------------------------------------------------------------------===#


def tt_noop(a: _TT):
    """Single TileTensor arg, does nothing."""
    pass


def tt_read(a: _TT):
    """Single TileTensor arg, reads one element."""
    _ = a[0, 0]


def tt_write(a: _TT):
    """Single TileTensor arg, writes one element."""
    a[0, 0] = Float32(1.0)


def tt_vector_add(a: _TT, b: _TT, result: _TT):
    """Three TileTensor args, simple vector add."""
    for i in range(16):
        result[0, i] = a[0, i] + b[0, i]


def tt_mixed(
    a: _TT,
    scale: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    result: _TT,
):
    """Mixed TileTensor + UnsafePointer args."""
    var s = scale[]
    for i in range(16):
        result[0, i] = a[0, i] * s


# ===-----------------------------------------------------------------------===#
# Tests
# ===-----------------------------------------------------------------------===#


def run_tt_hexagon_compile[
    declared_arg_types: TypeList[Trait=AnyType, ...],
    //,
    kernel: def(* args: * declared_arg_types) thin -> None,
](name: String) raises:
    print("  Compiling", name, "...")
    var elf = compile_info[make_npu_wrapper[kernel], target=get_gpu_target()]()
    var elf_size = elf.asm.byte_length()
    print("    ELF size:", elf_size, "bytes")
    print("    Function:", elf.function_name)
    assert_not_equal(elf_size, 0)


def test_qaic_tt_hexagon_compile() raises:
    print_test_header("TileTensor Hexagon Compilation Tests")
    print()

    print("Test 1: tt_noop (1 TileTensor, no body)")
    run_tt_hexagon_compile[kernel=tt_noop]("tt_noop")
    print()

    print("Test 2: tt_read (1 TileTensor, read)")
    run_tt_hexagon_compile[kernel=tt_read]("tt_read")
    print()

    print("Test 3: tt_write (1 TileTensor, write)")
    run_tt_hexagon_compile[kernel=tt_write]("tt_write")
    print()

    print("Test 4: tt_vector_add (3 TileTensors)")
    run_tt_hexagon_compile[kernel=tt_vector_add]("tt_vector_add")
    print()

    print("Test 5: tt_mixed (TileTensor + UnsafePointer)")
    run_tt_hexagon_compile[kernel=tt_mixed]("tt_mixed")
    print()

    print_separator()
    print("ALL TESTS PASSED")
    print_separator()


def main() raises:
    # Manual dispatch: `discover_tests[__functions_in_module()]` cannot
    # introspect the parameterized `run_tt_hexagon_compile` helper above
    # (variadic `TypeList[Trait=AnyType, ...]` parameter), so call the
    # single test directly.
    test_qaic_tt_hexagon_compile()
