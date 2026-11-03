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
"""Test passing a scalar value directly to enqueue_function.

Demonstrates that kernel parameters of scalar type (e.g. Float32) can be
passed by value on the host side. The unified `enqueue_function` detects
the scalar at compile time and auto-uploads it to a temporary device
buffer so that the QAIC JIT buffer map validation passes.
`make_npu_wrapper` then dereferences the FAM slot to recover the value
in device memory.

Kernel: output[i] = input[i] * scale

To run (requires QAIC hardware):
    mojo test_qaic_scalar_arg.mojo
"""

from std.memory import UnsafePointer, alloc
from std.testing import TestSuite, assert_equal

from max.gpu.host import DeviceContext, Dim
from std._plugin.qaic.host.target import npu_work_range

comptime N: Int = 256


# ===-----------------------------------------------------------------------===#
# Kernel
# ===-----------------------------------------------------------------------===#


def scale_kernel(
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    scale: Float32,
):
    """Multiply each element by a scalar: output[i] = input[i] * scale."""
    var (start, end) = npu_work_range(N)
    for i in range(start, end):
        output[i] = input[i] * scale


# ===-----------------------------------------------------------------------===#
# Test
# ===-----------------------------------------------------------------------===#


def test_qaic_scalar_arg() raises:
    var ctx = DeviceContext(device_id=0, api="qaic")

    var input_host = alloc[Float32](N)
    for i in range(N):
        input_host[i] = Float32(i + 1)  # 1, 2, ..., N
    var input_dev = ctx.enqueue_create_buffer[DType.float32](N)
    var output_dev = ctx.enqueue_create_buffer[DType.float32](N)
    ctx.enqueue_copy(input_dev, input_host)
    ctx.synchronize()

    # Pass the scale factor directly as a Float32 — no device buffer needed.
    var scale: Float32 = 3.0
    var kernel = ctx.compile_function[scale_kernel]()
    ctx.enqueue_function(
        kernel,
        input_dev,
        output_dev,
        scale,
        grid_dim=Dim(1),
        block_dim=Dim(0x01),
    )
    ctx.synchronize()

    var output_host = alloc[Float32](N)
    ctx.enqueue_copy(output_host, output_dev)
    ctx.synchronize()

    var errors = 0
    for i in range(N):
        var expected = Float32(i + 1) * scale
        if abs(output_host[i] - expected) > Float32(0.001):
            errors += 1

    input_host.unsafe_free()
    output_host.unsafe_free()

    if errors == 0:
        print("SCALAR ARG TEST PASSED: all", N, "elements correct")
    else:
        print("SCALAR ARG TEST FAILED:", errors, "errors out of", N)
    assert_equal(errors, 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
