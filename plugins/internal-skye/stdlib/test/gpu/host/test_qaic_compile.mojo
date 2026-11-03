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
"""Test QAIC kernel compilation.

This test demonstrates compiling a Mojo function to a Hexagon ELF shared object
that can be loaded and executed on Qualcomm AI accelerators via the JIT API.

To run:
    mojo test_qaic_compile.mojo

This will:
1. Define a simple vector addition kernel
2. Compile it to Hexagon assembly and object code
3. Write the compiled ELF to a file for inspection
"""

from std.testing import TestSuite
from std.pathlib import Path
from std.sys import argv

from std._plugin.qaic.host.target import (
    compile_info_npu,
    FLOAT32_PER_HVX,
    HVX_VECTOR_SIZE,
)


# ===-----------------------------------------------------------------------===#
# QAIC Kernel: Vector Addition
# ===-----------------------------------------------------------------------===#


def vector_add_kernel(
    input_a: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    input_b: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    size_ptr: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    """Simple vector addition kernel for QAIC.

    Args:
        input_a: First input vector (float32).
        input_b: Second input vector (float32).
        output: Output vector (float32).
        size_ptr: Pointer to size (uint32).
    """
    var size = Int(size_ptr[])
    for i in range(size):
        output[i] = input_a[i] + input_b[i]


# ===-----------------------------------------------------------------------===#
# QAIC Kernel: Vector Scale
# ===-----------------------------------------------------------------------===#


def vector_scale_kernel(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    scale_ptr: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    size_ptr: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    """Scale a vector by a constant.

    Args:
        input: Input vector (float32).
        output: Output vector (float32).
        scale_ptr: Scale factor (float32, single value).
        size_ptr: Vector size (uint32).
    """
    var scale = scale_ptr[]
    var size = Int(size_ptr[])

    for i in range(size):
        output[i] = input[i] * scale


# ===-----------------------------------------------------------------------===#
# QAIC Kernel: ReLU Activation
# ===-----------------------------------------------------------------------===#


def relu_kernel(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    size_ptr: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    """ReLU activation: output = max(0, input).

    Args:
        input: Input vector (float32).
        output: Output vector (float32).
        size_ptr: Vector size (uint32).
    """
    var size = Int(size_ptr[])
    for i in range(size):
        var val = input[i]
        output[i] = val if val > 0 else 0


# ===-----------------------------------------------------------------------===#
# Main: Compile and output kernels
# ===-----------------------------------------------------------------------===#


def test_qaic_compile() raises:
    print("=== QAIC Kernel Compilation Test ===")
    print()

    # Parse output directory from command line
    var output_dir = Path(".")
    var args = argv()
    for i in range(len(args)):
        if args[i] == "-o" and i + 1 < len(args):
            output_dir = Path(args[i + 1])
            break

    # Compile vector_add kernel
    print("Compiling vector_add_kernel...")

    # Get LLVM IR (for debugging)
    var llvm_ir = compile_info_npu[vector_add_kernel, emission_kind="llvm"]()
    print("  LLVM IR size:", llvm_ir.asm.byte_length(), "bytes")

    # Get assembly (for inspection)
    var asm = compile_info_npu[vector_add_kernel, emission_kind="asm"]()
    print("  Assembly size:", asm.asm.byte_length(), "bytes")
    print("  Function name:", asm.function_name)

    # Get object code (ELF shared library)
    var obj = compile_info_npu[vector_add_kernel]()
    print("  Object size:", obj.asm.byte_length(), "bytes")

    # Write outputs
    var asm_path = output_dir / "vector_add.s"
    var obj_path = output_dir / "vector_add.o"
    var llvm_path = output_dir / "vector_add.ll"

    asm_path.write_text(String(asm.asm))
    print("  Wrote assembly to:", asm_path)

    # Write object file
    with open(obj_path, "w") as f:
        f.write(obj.asm)
    print("  Wrote object to:", obj_path)

    # Write LLVM IR
    llvm_path.write_text(String(llvm_ir.asm))
    print("  Wrote LLVM IR to:", llvm_path)

    print()

    # Compile relu kernel
    print("Compiling relu_kernel...")
    var relu_asm = compile_info_npu[relu_kernel, emission_kind="asm"]()
    print("  Assembly size:", relu_asm.asm.byte_length(), "bytes")

    var relu_obj = compile_info_npu[relu_kernel]()
    print("  Object size:", relu_obj.asm.byte_length(), "bytes")

    var relu_asm_path = output_dir / "relu.s"
    var relu_obj_path = output_dir / "relu.o"
    relu_asm_path.write_text(String(relu_asm.asm))
    with open(relu_obj_path, "w") as f:
        f.write(relu_obj.asm)
    print("  Wrote assembly to:", relu_asm_path)
    print("  Wrote object to:", relu_obj_path)

    print()
    print("=== Compilation Complete ===")
    print()
    print("To inspect the generated code:")
    print("  hexagon-llvm-objdump -d", obj_path)
    print("  cat", asm_path)
    print()
    print("To load and execute on QAIC hardware, run:")
    print("  mojo test_qaic_execute.mojo")
    print()
    print("The ELF binary can be loaded with qaic_load_jit_function()")
    print("and executed with qaic_execute_jit_function().")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
