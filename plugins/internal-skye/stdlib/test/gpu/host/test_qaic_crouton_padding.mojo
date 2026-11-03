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
"""Characterize crouton padding and sub-tile behavior at the DMA/HMX boundary.

Crouton dims round M and K up to a multiple of 32, so any shape that is not a
whole number of 32x32 tiles produces a crouton buffer with *pad* lanes that have
no row-major source element. Two questions matter for anyone building a kernel
on top of the conversions, and neither is answered by the existing tests (which
only use exact tile multiples):

1. Does `row-major -> crouton -> row-major` still reconstruct the valid [M, K]
   region exactly when M or K is sub-tile / unaligned? (The converters are told
   the real extent via `flat_dims`, so they *should*.)

2. What does the pack routine write into the pad lanes — does it zero-fill them,
   or leave whatever was already in the crouton buffer? This decides whether a
   caller must pre-zero crouton scratch before packing an unaligned tensor (the
   production tiled kernel pre-zeros its staging buffers; the bare library
   conversion may or may not).

To answer (2) the crouton VTCM scratch is pre-filled with a 0xAB sentinel before
packing, then the whole crouton buffer is DMA'd back to DDR and inspected: bytes
still equal to 0xAB were never touched by the pack routine.

This is an investigation, not a regression guard — it prints a characterization
table. It is registered `manual` because it deliberately drives sub-tile shapes
through paths the production kernels never use, and asserts only the property it
can stand behind (exact reconstruction of the valid region).
"""

from common import print_separator, print_test_header
from qaic_test_utils.rng import Xorshift32
from qaic_test_utils.hmx import vtcm_align, npu_is_hmx_thread

from std.memory import UnsafePointer, alloc
from std.testing import TestSuite, assert_equal

from max.gpu.host import DeviceContext
from std._plugin.qaic.host.target import compile_info_npu, npu_thread_idx
from std._plugin.qaic.host.qshim import (
    qshim_get_base_vtcm_addr,
    qshim_linear_udma_submit,
    qshim_udma_wait,
    QShimUdmaDescCommonAttrs,
)
from std._plugin.qaic.host.hmx import (
    hmx_flat_dims_yd,
    hmx_crouton_dims_from_flat,
    hmx_crouton_size_in_bytes,
    hmx_matmul_bias_init_16b,
    hmx_rm_matmul_lhs_to_crouton_16b,
    hmx_matmul_crouton_to_rm_16b,
    hmx_sync_hvx_and_hmx_threads,
    hmx_sync_hvx_and_dma_threads,
    HMX_ALIGNMENT,
    HMX_INPUT_ALIGNMENT,
)
from max._plugin.qaic.host.qaic import (
    qaic_get_jit_device,
    qaic_allocate_jit_buffer,
    qaic_release_jit_buffer,
    qaic_load_jit_function,
    qaic_execute_jit_function,
    qaic_release_jit_function,
    qaic_copy_jit_buffer,
    QAicBuf,
    QAicDevice,
    QAIC_BUFFER_TYPE_DEV_DDR,
    QAIC_DEV_BUF_IO_RW,
    QAIC_COPY_TO_DEVICE,
    QAIC_COPY_FROM_DEVICE,
)


comptime PAD_SENTINEL: UInt8 = 0xAB


def crouton_pad_probe_kernel(
    out_rm_ptr: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    out_crouton_ptr: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    inp_ptr: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    m_ptr: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
    k_ptr: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    """Pack [M,K] into a sentinel-filled crouton buffer and export both the
    reconstructed row-major result and the raw crouton bytes.
    """
    var thread_id = UInt32(npu_thread_idx())
    var is_hmx = npu_is_hmx_thread()

    var m = Int(m_ptr[0])
    var k = Int(k_ptr[0])
    var data_bytes = m * k * 2

    var flat = hmx_flat_dims_yd(m, k)
    var crouton = hmx_crouton_dims_from_flat(flat)
    var crouton_bytes = hmx_crouton_size_in_bytes(crouton)

    if is_hmx:
        _ = hmx_matmul_bias_init_16b()
    hmx_sync_hvx_and_hmx_threads(thread_id)

    var vtcm = qshim_get_base_vtcm_addr()
    var off = 0
    off = vtcm_align(off, HMX_INPUT_ALIGNMENT)
    var inp_vtcm = vtcm + off
    off += data_bytes
    off = vtcm_align(off, HMX_ALIGNMENT)
    var crouton_vtcm = vtcm + off
    off += crouton_bytes
    off = vtcm_align(off, HMX_INPUT_ALIGNMENT)
    var out_vtcm = vtcm + off

    # Phase 1: DMA input DDR -> VTCM and pre-fill the crouton scratch with a
    # sentinel so untouched pad lanes are recognizable after packing.
    if is_hmx:
        var st: UInt32 = 0
        var at = QShimUdmaDescCommonAttrs(order_and_reserved=0)
        var h = qshim_linear_udma_submit(
            thread_id,
            inp_ptr,
            UInt32(data_bytes),
            inp_vtcm,
            UnsafePointer[QShimUdmaDescCommonAttrs, MutUnsafeAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=at))
            ),
            True,
            UnsafePointer[UInt32, MutUnsafeAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=st))
            ),
        )
        _ = qshim_udma_wait(h)
        for i in range(crouton_bytes):
            crouton_vtcm[i] = PAD_SENTINEL
    hmx_sync_hvx_and_dma_threads(thread_id)

    # Phase 2: pack row-major -> crouton.
    if not is_hmx:
        _ = hmx_rm_matmul_lhs_to_crouton_16b(
            crouton_vtcm,
            inp_vtcm,
            UnsafePointer[UInt8, MutUnsafeAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=crouton))
            ),
            UnsafePointer[UInt8, MutUnsafeAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=flat))
            ),
            Int32(thread_id),
        )
    hmx_sync_hvx_and_hmx_threads(thread_id)

    # Phase 3: unpack crouton -> row-major (valid [M,K] extent only).
    if not is_hmx:
        _ = hmx_matmul_crouton_to_rm_16b(
            out_vtcm,
            crouton_vtcm,
            UnsafePointer[UInt8, MutUnsafeAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=flat))
            ),
            UnsafePointer[UInt8, MutUnsafeAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=crouton))
            ),
            Int32(thread_id),
        )
    hmx_sync_hvx_and_hmx_threads(thread_id)

    # Phase 4: export reconstructed row-major and raw crouton bytes.
    if is_hmx:
        var st: UInt32 = 0
        var at = QShimUdmaDescCommonAttrs(order_and_reserved=0)
        var h_rm = qshim_linear_udma_submit(
            thread_id,
            out_vtcm,
            UInt32(data_bytes),
            out_rm_ptr,
            UnsafePointer[QShimUdmaDescCommonAttrs, MutUnsafeAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=at))
            ),
            True,
            UnsafePointer[UInt32, MutUnsafeAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=st))
            ),
        )
        _ = qshim_udma_wait(h_rm)
        var h_cr = qshim_linear_udma_submit(
            thread_id,
            crouton_vtcm,
            UInt32(crouton_bytes),
            out_crouton_ptr,
            UnsafePointer[QShimUdmaDescCommonAttrs, MutUnsafeAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=at))
            ),
            True,
            UnsafePointer[UInt32, MutUnsafeAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=st))
            ),
        )
        _ = qshim_udma_wait(h_cr)
    hmx_sync_hvx_and_dma_threads(thread_id)


def probe_shape(M: Int, K: Int, ctx: DeviceContext) raises:
    var flat = hmx_flat_dims_yd(M, K)
    var crouton = hmx_crouton_dims_from_flat(flat)
    var crouton_bytes = hmx_crouton_size_in_bytes(crouton)
    var y_tiles = Int(crouton.y_dim)
    var d_tiles = Int(crouton.d_dim)
    var valid_elems = M * K
    var tile_elems = y_tiles * 32 * d_tiles * 32

    print(
        "  shape [",
        M,
        ",",
        K,
        "] -> crouton",
        y_tiles,
        "x",
        d_tiles,
        "tiles;",
        valid_elems,
        "valid /",
        tile_elems,
        "padded elements",
    )

    var size = M * K
    var data_bytes = size * 2
    var jit_device = qaic_get_jit_device(ctx)

    var inp_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        UInt64(data_bytes),
        128,
        0xFFFFFFFFFFFFFFFF,
    )
    var out_rm_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        UInt64(data_bytes),
        128,
        0xFFFFFFFFFFFFFFFF,
    )
    var out_crouton_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        UInt64(crouton_bytes),
        128,
        0xFFFFFFFFFFFFFFFF,
    )
    var m_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        UInt64(4),
        128,
        0xFFFFFFFFFFFFFFFF,
    )
    var k_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        UInt64(4),
        128,
        0xFFFFFFFFFFFFFFFF,
    )

    var inp_host = alloc[Float16](size)
    var out_rm_host = alloc[Float16](size)
    var out_crouton_host = alloc[UInt8](crouton_bytes)
    var rng = Xorshift32.from_time()
    # Avoid 0.0 so a zero in the reconstructed buffer is unambiguous.
    rng.fill_fp16(inp_host, size, scale=1.0)
    for i in range(size):
        if inp_host[i] == Float16(0):
            inp_host[i] = Float16(0.5)

    var m_host = alloc[UInt32](1)
    var k_host = alloc[UInt32](1)
    m_host[0] = UInt32(M)
    k_host[0] = UInt32(K)

    qaic_copy_jit_buffer(
        ctx,
        inp_buf,
        inp_host.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(data_bytes),
        QAIC_COPY_TO_DEVICE,
    )
    qaic_copy_jit_buffer(
        ctx,
        m_buf,
        m_host.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(4),
        QAIC_COPY_TO_DEVICE,
    )
    qaic_copy_jit_buffer(
        ctx,
        k_buf,
        k_host.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(4),
        QAIC_COPY_TO_DEVICE,
    )

    var elf = compile_info_npu[crouton_pad_probe_kernel]()
    var jit_func = qaic_load_jit_function(
        ctx,
        elf.function_name,
        elf.asm.unsafe_ptr().bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(elf.asm.byte_length()),
    )

    var buffers = alloc[QAicBuf](5)
    buffers[0] = out_rm_buf
    buffers[1] = out_crouton_buf
    buffers[2] = inp_buf
    buffers[3] = m_buf
    buffers[4] = k_buf
    qaic_execute_jit_function(
        ctx,
        jit_func,
        buffers.as_unsafe_any_origin(),
        5,
        num_cores=1,
        thread_mask=UInt32(0x1F),
    )

    qaic_copy_jit_buffer(
        ctx,
        out_rm_buf,
        out_rm_host.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(data_bytes),
        QAIC_COPY_FROM_DEVICE,
    )
    qaic_copy_jit_buffer(
        ctx,
        out_crouton_buf,
        out_crouton_host.as_unsafe_any_origin(),
        UInt64(crouton_bytes),
        QAIC_COPY_FROM_DEVICE,
    )

    # (1) Reconstruction of the valid region.
    var mismatches = 0
    for i in range(size):
        if out_rm_host[i] != inp_host[i]:
            mismatches += 1
    print(
        "    valid-region round-trip:",
        size - mismatches,
        "/",
        size,
        "exact (",
        mismatches,
        "mismatches )",
    )

    # (2) Pad-lane characterization of the packed crouton bytes.
    var sentinel = 0
    var zero = 0
    var other = 0
    for i in range(crouton_bytes):
        var b = out_crouton_host[i]
        if b == PAD_SENTINEL:
            sentinel += 1
        elif b == UInt8(0):
            zero += 1
        else:
            other += 1
    print(
        "    crouton bytes:",
        crouton_bytes,
        "total |",
        sentinel,
        "untouched(0xAB) |",
        zero,
        "zero |",
        other,
        "other",
    )
    if sentinel == 0:
        print("    -> pack writes every crouton byte (pad lanes overwritten)")
    else:
        print(
            "    -> pack leaves",
            sentinel,
            "pad bytes untouched; caller must pre-zero crouton scratch",
        )

    inp_host.unsafe_free()
    out_rm_host.unsafe_free()
    out_crouton_host.unsafe_free()
    m_host.unsafe_free()
    k_host.unsafe_free()
    buffers.unsafe_free()
    qaic_release_jit_buffer(ctx, inp_buf)
    qaic_release_jit_buffer(ctx, out_rm_buf)
    qaic_release_jit_buffer(ctx, out_crouton_buf)
    qaic_release_jit_buffer(ctx, m_buf)
    qaic_release_jit_buffer(ctx, k_buf)
    qaic_release_jit_function(jit_func)

    # The valid region must reconstruct exactly regardless of padding.
    assert_equal(mismatches, 0)


def test_crouton_padding_subtile() raises:
    print_test_header("crouton padding: sub-tile shapes (< 32)")
    var ctx = DeviceContext(device_id=0, api="qaic")
    probe_shape(8, 8, ctx)
    probe_shape(16, 16, ctx)
    probe_shape(31, 31, ctx)


def test_crouton_padding_unaligned() raises:
    print_test_header("crouton padding: unaligned shapes")
    var ctx = DeviceContext(device_id=0, api="qaic")
    probe_shape(33, 33, ctx)
    probe_shape(40, 24, ctx)
    probe_shape(33, 17, ctx)
    probe_shape(17, 65, ctx)
    print_separator()
    print("TEST PASSED!")
    print_separator()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
