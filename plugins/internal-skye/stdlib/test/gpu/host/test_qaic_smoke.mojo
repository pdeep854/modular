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
"""QAIC smoke test: minimal vector-add proving the full QAIC stack works.

Demonstrates the recommended high-level pattern for QAIC tests:
1. Check hardware availability (skip gracefully if absent)
2. Define a user-friendly kernel (UnsafePointer args, no entry_config)
3. Use DeviceContext high-level API (enqueue_create_buffer, enqueue_copy,
   compile_function, enqueue_function)
4. Verify results

The test-specific code (kernel + verification) is under 20 lines.
All boilerplate (device init, buffer management) uses the high-level API.

To run (requires QAIC hardware):
    mojo test_qaic_smoke.mojo
"""

from std.testing import TestSuite, assert_equal
from std.memory import UnsafePointer, alloc

from max.gpu.host import DeviceContext, Dim
from std._plugin.qaic.host.target import npu_work_range


# ===-----------------------------------------------------------------------===#
# Kernel
# ===-----------------------------------------------------------------------===#


def add_kernel(
    a: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    b: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    c: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    """Vector addition: c[i] = a[i] + b[i]."""
    var (start, end) = npu_work_range(256)
    for i in range(start, end):
        c[i] = a[i] + b[i]


# ===-----------------------------------------------------------------------===#
# Test
# ===-----------------------------------------------------------------------===#


def test_qaic_smoke() raises:
    # Use a single DeviceContext for both availability check and test execution.
    # Creating a separate probe context corrupts the shared Device's
    # MemoryManager state for subsequent contexts (GEX-XXXX).
    var n = 256
    var ctx = DeviceContext(device_id=0, api="qaic")

    # Allocate device buffers
    var a_dev = ctx.enqueue_create_buffer[DType.float32](n)
    var b_dev = ctx.enqueue_create_buffer[DType.float32](n)
    var c_dev = ctx.enqueue_create_buffer[DType.float32](n)

    # Prepare host data: a[i] = i, b[i] = 2*i
    var a_host = alloc[Float32](n)
    var b_host = alloc[Float32](n)
    for i in range(n):
        a_host[i] = Float32(i)
        b_host[i] = Float32(2 * i)

    # Copy to device, compile, execute
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.synchronize()

    var func = ctx.compile_function[add_kernel]()
    ctx.enqueue_function(
        func, a_dev, b_dev, c_dev, grid_dim=Dim(1), block_dim=Dim(0x01)
    )
    ctx.synchronize()

    # Copy results back and verify: c[i] == 3*i
    var c_host = alloc[Float32](n)
    ctx.enqueue_copy(c_host, c_dev)
    ctx.synchronize()

    var errors = 0
    for i in range(n):
        if abs(c_host[i] - Float32(3 * i)) > 0.001:
            errors += 1

    # Clean up
    a_host.unsafe_free()
    b_host.unsafe_free()
    c_host.unsafe_free()

    if errors == 0:
        print("SMOKE TEST PASSED: all", n, "elements correct")
    else:
        print("SMOKE TEST FAILED:", errors, "errors out of", n)
    assert_equal(errors, 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
