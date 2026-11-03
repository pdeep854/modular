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
"""Test: closure launched across multiple HVX threads.

Validates that `num_cores` / `thread_mask` actually drive parallel
hardware: the closure body calls `npu_work_range(N)` to partition the
work, and each thread writes a contiguous chunk of the output. With
`thread_mask=0x0F` (4 threads enabled) on `num_cores=1`, four workers
each handle N/4 elements.

If the dispatch were single-threaded, only the first thread's chunk
would be filled — the rest would remain at the pre-fill sentinel.
"""

from common import print_separator, print_test_header

from std.memory import UnsafePointer, alloc
from std.testing import TestSuite, assert_equal

from max.gpu.host import DeviceContext, Dim
from std._plugin.qaic.host.target import npu_work_range


comptime N: Int = 256
comptime NUM_THREADS: Int = 4


def test_qaic_closure_multithread() raises:
    print_test_header("QAIC closure multi-thread launch")
    print()

    var ctx = DeviceContext(api="qaic")
    var input_dev = ctx.enqueue_create_buffer[DType.float32](N)
    var output_dev = ctx.enqueue_create_buffer[DType.float32](N)

    var input_host = alloc[Float32](N)
    for i in range(N):
        input_host[i] = Float32(i + 1)
    ctx.enqueue_copy(input_dev, input_host)

    # Pre-fill output with a sentinel so unwritten slots are visible.
    var sentinel = alloc[Float32](N)
    for i in range(N):
        sentinel[i] = -999.0
    ctx.enqueue_copy(output_dev, sentinel)
    ctx.synchronize()
    input_host.unsafe_free()
    sentinel.unsafe_free()

    var inp = input_dev._device_ptr
    var out = output_dev._device_ptr
    var size: Int32 = Int32(N)

    def kernel() {var inp, var out, var size}:
        var (start, end) = npu_work_range(Int(size))
        for i in range(start, end):
            out[i] = inp[i] * 2.0

    var func = ctx.compile_function[type_of(kernel)]()
    var thread_mask: Int = (1 << NUM_THREADS) - 1  # 0x0F
    ctx.enqueue_function(
        func,
        kernel,
        grid_dim=Dim(1),
        block_dim=Dim(1),
        num_cores=1,
        thread_mask=thread_mask,
    )
    ctx.synchronize()

    _ = input_dev

    var output_host = alloc[Float32](N)
    ctx.enqueue_copy(output_host, output_dev)
    ctx.synchronize()

    var errors = 0
    for i in range(N):
        var expected = Float32(i + 1) * 2.0
        if abs(output_host[i] - expected) > 0.001:
            errors += 1
            if errors <= 5:
                print(
                    "FAIL at",
                    i,
                    ": expected",
                    expected,
                    "got",
                    output_host[i],
                )
    output_host.unsafe_free()

    print_separator()
    if errors == 0:
        print(
            "TEST PASSED — all",
            N,
            "elements written across",
            NUM_THREADS,
            "threads",
        )
    else:
        print("TEST FAILED:", errors, "errors")
    print_separator()

    assert_equal(errors, 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
