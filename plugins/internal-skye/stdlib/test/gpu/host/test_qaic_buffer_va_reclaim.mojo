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
"""Empirical proof that the high-level QAIC dispatch path reclaims per-NSP
virtual address (VA) space per command buffer.

`enqueueFunctionExecDirect` builds its buffer descriptors with
`QAIC_DEV_BUF_RESIDENT_CMD_BUF`, so a buffer's NSP-VA mapping is released when
its command buffer completes — unlike `RESIDENT_KEEP`, which holds the mapping
until the buffer is freed.

This test keeps `_NUM_BUFFERS` large buffers allocated at once and dispatches a
kernel on each. The cumulative mapping footprint (`_NUM_BUFFERS *
_BYTES_PER_BUF` = 5120 MiB) far exceeds the ~3,584 MiB per-NSP VA window, and a
single NSP (`grid_dim=1`) takes all the pressure. Under `RESIDENT_KEEP` the
mappings would coexist and overflow VA, putting the device in an unrecoverable
MMAP-ERROR state around the 7th buffer; under `RESIDENT_CMD_BUF` only the
in-flight command buffer's mapping is resident, so the run completes. The
second pass re-dispatches on each buffer and reads its sentinels back, proving
the contents survive the unmap/remap (the DDR allocation is never freed — only
the VA mapping is dropped and re-established by AUTO_MAP).

So this doubles as a regression guard: it passes under `RESIDENT_CMD_BUF` and
bricks under `RESIDENT_KEEP`.

To run (requires QAIC hardware):
    mojo test_qaic_buffer_va_reclaim.mojo
"""

from std.testing import TestSuite, assert_equal
from std.memory import UnsafePointer, alloc

from max.gpu.host import DeviceBuffer, DeviceContext, Dim


# 512 MiB per buffer, expressed as uint32 elements.
comptime _BYTES_PER_BUF: Int = 512 * 1024 * 1024
comptime _ELEMS_PER_BUF: Int = _BYTES_PER_BUF // 4
comptime _LAST: Int = _ELEMS_PER_BUF - 1

# 10 live buffers => 5120 MiB of mappings if they all had to be resident at
# once, well past the ~3,584 MiB per-NSP VA window and under the ~9.7 GiB
# allocatable DDR ceiling.
comptime _NUM_BUFFERS: Int = 10
comptime _CUMULATIVE_MIB: Int = _NUM_BUFFERS * (_BYTES_PER_BUF // (1024 * 1024))

comptime _SENTINEL: UInt32 = 7


def fill_ends_kernel(buf: UnsafePointer[UInt32, MutUnsafeAnyOrigin]):
    """Touch both ends of the full buffer mapping with a sentinel."""
    buf[0] = _SENTINEL
    buf[_LAST] = _SENTINEL


def sum_ends_kernel(
    buf: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
    dst: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    """Re-read both ends (re-mapped for this dispatch) and write their sum."""
    dst[0] = buf[0] + buf[_LAST]


def test_qaic_buffer_va_reclaim() raises:
    var ctx = DeviceContext(device_id=0, api="qaic")

    print(
        "Holding",
        _NUM_BUFFERS,
        "live buffers x",
        _BYTES_PER_BUF // (1024 * 1024),
        "MiB =",
        _CUMULATIVE_MIB,
        "MiB of mappings vs ~3584 MiB per-NSP VA window.",
    )

    # Allocate every buffer up front and keep them all live for the whole test.
    # Under RESIDENT_KEEP these mappings would coexist and overflow VA.
    var bufs = List[DeviceBuffer[DType.uint32]]()
    for _ in range(_NUM_BUFFERS):
        bufs.append(ctx.enqueue_create_buffer[DType.uint32](_ELEMS_PER_BUF))

    var fill = ctx.compile_function[fill_ends_kernel]()
    var sum_ends = ctx.compile_function[sum_ends_kernel]()

    # Fill pass: one dispatch per buffer while every buffer stays allocated.
    # This is where cumulative VA would cross the window under RESIDENT_KEEP.
    for i in range(_NUM_BUFFERS):
        ctx.enqueue_function(
            fill, bufs[i], grid_dim=Dim(1), block_dim=Dim(0x01)
        )
        ctx.synchronize()

    # Verify pass: re-dispatch on each buffer (forcing a remap under
    # RESIDENT_CMD_BUF) and confirm the sentinels survived the unmap/remap.
    var result_dev = ctx.enqueue_create_buffer[DType.uint32](1)
    var result_host = alloc[UInt32](1)
    for i in range(_NUM_BUFFERS):
        ctx.enqueue_function(
            sum_ends, bufs[i], result_dev, grid_dim=Dim(1), block_dim=Dim(0x01)
        )
        ctx.synchronize()
        ctx.enqueue_copy(result_host, result_dev)
        ctx.synchronize()
        assert_equal(
            Int(result_host[0]),
            Int(2 * _SENTINEL),
            "buffer " + String(i) + " lost its contents across unmap/remap",
        )

    result_host.unsafe_free()
    print(
        "PASSED: dispatched across",
        _CUMULATIVE_MIB,
        (
            "MiB of live buffers without exceeding the per-NSP VA window;"
            " all contents intact across unmap/remap."
        ),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
