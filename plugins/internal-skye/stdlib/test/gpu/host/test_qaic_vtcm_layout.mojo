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
"""Test VTCM layout: vector add followed by two add-1 kernels.

To run (requires QAIC hardware):
    mojo test_qaic_vtcm_layout.mojo
"""

from std.testing import TestSuite, assert_equal
from common import print_test_header

from std.memory import AddressSpace, UnsafePointer, alloc, stack_allocation

from layout import Layout, LayoutTensor

from max.gpu.host import DeviceContext, Dim
from std._plugin.qaic.host.target import (
    compile_info_npu,
    npu_barrier,
    npu_thread_idx,
    npu_work_range,
)
from std._plugin.qaic.host.qshim import (
    QShimUdmaDescCommonAttrs,
    qshim_create_thread_sync_point,
    qshim_linear_udma_submit,
    qshim_udma_wait,
    qshim_get_vtcm_size,
    qshim_get_base_vtcm_addr,
)


# ===-----------------------------------------------------------------------===#
# Kernel configuration
# ===-----------------------------------------------------------------------===#

comptime NUM_CORES: Int = 1
comptime THREAD_MASK: Int = 0x0F  # 4 HVX threads

# Add two tiles of Float32 that fill up the 8 MiB VTCM
comptime SIMD_WIDTH = 32
comptime ROWS = (8 * 1024 * 1024) // (SIMD_WIDTH * 4 * 2)
comptime layout = Layout.row_major(ROWS, SIMD_WIDTH)

comptime TOTAL_SIZE = ROWS * SIMD_WIDTH


def vtcm_vec_add(
    lhs: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    rhs: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    result: UnsafePointer[Float32, MutUnsafeAnyOrigin],
):
    """Loads lhs and rhs into separate VTCM LayoutTensors, then adds them."""
    var lhs_tile = LayoutTensor[
        DType.float32,
        layout,
        MutUnsafeAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var rhs_tile = LayoutTensor[
        DType.float32,
        layout,
        MutUnsafeAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    # The mojo compiler will lower the SHARED allocations above to VTCM. This
    # can also be done manually, and what the mojo compiler does is essentially
    # the same as the following commented code:
    #
    # var vtcm_base = qshim_get_base_vtcm_addr()
    # var lhs_tile = LayoutTensor[
    #     DType.float32, layout, MutUnsafeAnyOrigin,
    #     address_space = AddressSpace.GENERIC,
    # ](vtcm_base.bitcast[Float32]())
    # var rhs_tile = LayoutTensor[
    #     DType.float32, layout, MutUnsafeAnyOrigin,
    #     address_space = AddressSpace.GENERIC,
    # ]((vtcm_base + lhs_tile.storage_size).bitcast[Float32]())

    # Sync point shared across all 4 threads.
    var sync = qshim_create_thread_sync_point[Int32(THREAD_MASK)]()

    var tid = npu_thread_idx()
    if tid == 0:
        var status: UInt32 = 0
        var attrs = QShimUdmaDescCommonAttrs(order_and_reserved=1)
        var lhs_dst = UnsafePointer[UInt8, MutUnsafeAnyOrigin](
            unsafe_from_address=Int(lhs_tile.ptr.bitcast[UInt8]())
        )
        var rhs_dst = UnsafePointer[UInt8, MutUnsafeAnyOrigin](
            unsafe_from_address=Int(rhs_tile.ptr.bitcast[UInt8]())
        )
        var h0 = qshim_linear_udma_submit(
            UInt32(tid),
            lhs.bitcast[UInt8]().as_unsafe_any_origin(),
            UInt32(lhs_tile.storage_size),
            lhs_dst,
            UnsafePointer(to=attrs).as_unsafe_any_origin(),
            True,
            UnsafePointer(to=status).as_unsafe_any_origin(),
        )
        var h1 = qshim_linear_udma_submit(
            UInt32(tid),
            rhs.bitcast[UInt8]().as_unsafe_any_origin(),
            UInt32(rhs_tile.storage_size),
            rhs_dst,
            UnsafePointer(to=attrs).as_unsafe_any_origin(),
            True,
            UnsafePointer(to=status).as_unsafe_any_origin(),
        )
        _ = qshim_udma_wait(h0)
        _ = qshim_udma_wait(h1)

    # # All 4 threads wait until thread 0 has finished filling the tiles.
    _ = npu_barrier(sync.as_unsafe_any_origin())

    # Each thread processes its slice of rows via vectorized SIMD addition.
    var lhs_vec = lhs_tile.flatten().vectorize[SIMD_WIDTH]()
    var rhs_vec = rhs_tile.flatten().vectorize[SIMD_WIDTH]()
    var (start_row, end_row) = npu_work_range(ROWS)
    for row in range(start_row, end_row):
        var sum = lhs_vec[row] + rhs_vec[row]
        result.store(row * SIMD_WIDTH, sum)


def dma_buf_to_vtcm(
    buf: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    vtcm: UnsafePointer[
        Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED
    ],
):
    """Thread 0 DMAs buf from DDR into vtcm; caller must barrier before reading vtcm.
    """
    if npu_thread_idx() == 0:
        var status: UInt32 = 0
        var attrs = QShimUdmaDescCommonAttrs(order_and_reserved=1)
        var vtcm_dst = UnsafePointer[UInt8, MutUnsafeAnyOrigin](
            unsafe_from_address=Int(vtcm.bitcast[UInt8]())
        )
        var h = qshim_linear_udma_submit(
            UInt32(0),
            buf.bitcast[UInt8]().as_unsafe_any_origin(),
            UInt32(TOTAL_SIZE * 4),
            vtcm_dst,
            UnsafePointer(to=attrs).as_unsafe_any_origin(),
            True,
            UnsafePointer(to=status).as_unsafe_any_origin(),
        )
        _ = qshim_udma_wait(h)


@always_inline
def vtcm_plus1_inline(buf: UnsafePointer[Float32, MutUnsafeAnyOrigin]):
    var vtcm = stack_allocation[
        TOTAL_SIZE,
        DType.float32,
        alignment=SIMD_WIDTH * 4,
        address_space=AddressSpace.SHARED,
    ]()
    var sync = qshim_create_thread_sync_point[Int32(THREAD_MASK)]()
    dma_buf_to_vtcm(buf, vtcm)
    _ = npu_barrier(sync.as_unsafe_any_origin())

    var (start, end) = npu_work_range(ROWS)
    for row in range(start, end):
        buf.store(
            row * SIMD_WIDTH,
            vtcm.load[width=SIMD_WIDTH](row * SIMD_WIDTH) + 1.0,
        )


@no_inline
def vtcm_plus1_noinline(buf: UnsafePointer[Float32, MutUnsafeAnyOrigin]):
    var vtcm = stack_allocation[
        TOTAL_SIZE,
        DType.float32,
        alignment=SIMD_WIDTH * 4,
        address_space=AddressSpace.SHARED,
    ]()
    var sync = qshim_create_thread_sync_point[Int32(THREAD_MASK)]()
    dma_buf_to_vtcm(buf, vtcm)
    _ = npu_barrier(sync.as_unsafe_any_origin())

    var (start, end) = npu_work_range(ROWS)
    for row in range(start, end):
        buf.store(
            row * SIMD_WIDTH,
            vtcm.load[width=SIMD_WIDTH](row * SIMD_WIDTH) + 1.0,
        )


# ===-----------------------------------------------------------------------===#
# Main
# ===-----------------------------------------------------------------------===#


def test_qaic_vtcm_layout() raises:
    print_test_header("QAIC VTCM LayoutTensor vector add")

    var ctx = DeviceContext(device_id=0, api="qaic")

    var lhs_buf = ctx.enqueue_create_buffer[DType.float32](TOTAL_SIZE)
    var rhs_buf = ctx.enqueue_create_buffer[DType.float32](TOTAL_SIZE)
    var res_buf = ctx.enqueue_create_buffer[DType.float32](TOTAL_SIZE)

    var lhs_host = alloc[Float32](TOTAL_SIZE)
    var rhs_host = alloc[Float32](TOTAL_SIZE)
    for i in range(TOTAL_SIZE):
        lhs_host[i] = Float32(i)
        rhs_host[i] = Float32(100)

    ctx.enqueue_copy(lhs_buf, lhs_host)
    ctx.enqueue_copy(rhs_buf, rhs_host)

    var k = ctx.compile_function[vtcm_vec_add]()
    ctx.enqueue_function(
        k,
        lhs_buf,
        rhs_buf,
        res_buf,
        grid_dim=Dim(NUM_CORES),
        block_dim=Dim(THREAD_MASK),
    )

    # Second kernel using VTCM inlined in the wrapper
    var k2 = ctx.compile_function[vtcm_plus1_inline]()
    ctx.enqueue_function(
        k2,
        res_buf,
        grid_dim=Dim(NUM_CORES),
        block_dim=Dim(THREAD_MASK),
    )

    # Third kernel using VTCM not inlined in the wrapper
    var k3 = ctx.compile_function[vtcm_plus1_noinline]()
    ctx.enqueue_function(
        k3,
        res_buf,
        grid_dim=Dim(NUM_CORES),
        block_dim=Dim(THREAD_MASK),
    )

    var res_host = alloc[Float32](TOTAL_SIZE)
    ctx.enqueue_copy(res_host, res_buf)
    ctx.synchronize()

    var errors = 0
    for i in range(TOTAL_SIZE):
        var expected = Float32(i) + Float32(102)
        if res_host[i] != expected:
            if errors < 5:
                print(
                    "  error at", i, ": expected", expected, "got", res_host[i]
                )
            errors += 1
    lhs_host.unsafe_free()
    rhs_host.unsafe_free()
    res_host.unsafe_free()

    if errors == 0:
        print("  PASS")
    else:
        print("  FAIL:", errors, "errors")
    assert_equal(errors, 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
