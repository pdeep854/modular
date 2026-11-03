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
"""Shared single-NSP HMX FP16 matmul kernel and host launcher.

The kernel computes `C = A * B` (row-major FP16, FP32 accumulation) on one
NSP using the HMX unit. Dimensions are passed via device buffers so a single
loaded JIT function can be reused across matmuls of different shapes — the
tensor-parallel tests rely on this to run two back-to-back matmuls per rank.

The kernel was promoted out of `test_qaic_hmx_matmul.mojo` so the matmul and
the QCCL collective tests share one definition. Constraints inherited from the
single-tile design: `A`, `B`, `C` plus their crouton-format copies must all fit
in VTCM at once, so it targets small shapes (the tiled `hmx_gemm` variant
handles large ones).
"""

from std.memory import UnsafePointer, alloc

from max.gpu.host import DeviceContext
from std._plugin.qaic.host.target import compile_info_npu, npu_thread_idx
from std._plugin.qaic.host.qshim import (
    qshim_get_base_vtcm_addr,
    qshim_linear_udma_submit,
    qshim_udma_wait,
    QShimUdmaDescCommonAttrs,
)
from std._plugin.qaic.host.hmx import (
    QAicHMXDims,
    hmx_flat_dims_yd,
    hmx_crouton_dims_from_flat,
    hmx_crouton_size_in_bytes,
    hmx_matmul_bias_init_16b,
    hmx_rm_matmul_lhs_to_crouton_16b,
    hmx_rm_matmul_rhs_to_crouton_16b,
    hmx_matmul_with_crouton_hf,
    hmx_matmul_crouton_to_rm_16b,
    hmx_sync_hvx_and_hmx_threads,
    hmx_sync_hvx_and_dma_threads,
    HMX_ALIGNMENT,
    HMX_INPUT_ALIGNMENT,
    HMX_WEIGHT_ALIGNMENT,
)
from max._plugin.qaic.host.qaic import (
    qaic_allocate_jit_buffer,
    qaic_copy_jit_buffer,
    qaic_execute_jit_function,
    qaic_load_jit_function,
    qaic_release_jit_buffer,
    QAicBuf,
    QAicDevice,
    QAicJitFunction,
    QAIC_BUFFER_TYPE_DEV_DDR,
    QAIC_COPY_TO_DEVICE,
    QAIC_DEV_BUF_IO_RW,
)

from .hmx import vtcm_align, npu_is_hmx_thread


# ===-----------------------------------------------------------------------===#
# HMX Matmul Kernel (runs on device)
# ===-----------------------------------------------------------------------===#


def hmx_matmul_kernel(
    c_ptr: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    a_ptr: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    b_ptr: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    m_ptr: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
    k_ptr: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
    n_ptr: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    """HMX-accelerated FP16 matrix multiplication kernel.

    Computes C = A * B using HMX hardware on a single NSP.
    Dispatched with thread_mask=0x1F (threads 0-4).

    Thread roles (AI100 V68):
    - Threads 0-3 (HVX): crouton format conversions
    - Thread 4 (HMX+DMA): bias init, matmul, DMA transfers
    """
    var thread_id = UInt32(npu_thread_idx())
    var is_hmx = npu_is_hmx_thread()

    var m = Int(m_ptr[0])
    var k = Int(k_ptr[0])
    var n = Int(n_ptr[0])

    var lhs_bytes = m * k * 2
    var rhs_bytes = k * n * 2
    var out_bytes = m * n * 2

    # ---- Phase 0: HMX bias init (thread 4 only) ----
    if is_hmx:
        _ = hmx_matmul_bias_init_16b()
    hmx_sync_hvx_and_hmx_threads(thread_id)

    # ---- VTCM allocation ----
    var vtcm_base = qshim_get_base_vtcm_addr()
    var off = 0

    off = vtcm_align(off, HMX_INPUT_ALIGNMENT)
    var lhs_vtcm = vtcm_base + off
    off = off + lhs_bytes
    off = vtcm_align(off, HMX_INPUT_ALIGNMENT)
    var rhs_vtcm = vtcm_base + off
    off = off + rhs_bytes

    var lhs_flat = hmx_flat_dims_yd(m, k)
    var lhs_crouton = hmx_crouton_dims_from_flat(lhs_flat)
    var lhs_crouton_bytes = hmx_crouton_size_in_bytes(lhs_crouton)

    off = vtcm_align(off, HMX_ALIGNMENT)
    var lhs_crouton_vtcm = vtcm_base + off
    off = off + lhs_crouton_bytes

    var k_aligned = (k + 31) & ~31
    var n_aligned = (n + 31) & ~31
    var rhs_weight_bytes = k_aligned * n_aligned * 2

    off = vtcm_align(off, HMX_WEIGHT_ALIGNMENT)
    var rhs_crouton_vtcm = vtcm_base + off
    off = off + rhs_weight_bytes

    var out_flat = hmx_flat_dims_yd(m, n)
    var out_crouton = hmx_crouton_dims_from_flat(out_flat)
    var out_crouton_bytes = hmx_crouton_size_in_bytes(out_crouton)

    off = vtcm_align(off, HMX_ALIGNMENT)
    var out_crouton_vtcm = vtcm_base + off
    off = off + out_crouton_bytes

    off = vtcm_align(off, 128)
    var out_vtcm = vtcm_base + off
    off = off + out_bytes

    # ---- Phase 1: DMA input DDR -> VTCM (thread 4) ----
    if is_hmx:
        var st1: UInt32 = 0
        var st2: UInt32 = 0
        var at1 = QShimUdmaDescCommonAttrs(order_and_reserved=0)
        var at2 = QShimUdmaDescCommonAttrs(order_and_reserved=0)
        var h1 = qshim_linear_udma_submit(
            thread_id,
            a_ptr,
            UInt32(lhs_bytes),
            lhs_vtcm,
            UnsafePointer[QShimUdmaDescCommonAttrs, MutUnsafeAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=at1))
            ),
            True,
            UnsafePointer[UInt32, MutUnsafeAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=st1))
            ),
        )
        var h2 = qshim_linear_udma_submit(
            thread_id,
            b_ptr,
            UInt32(rhs_bytes),
            rhs_vtcm,
            UnsafePointer[QShimUdmaDescCommonAttrs, MutUnsafeAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=at2))
            ),
            True,
            UnsafePointer[UInt32, MutUnsafeAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=st2))
            ),
        )
        _ = qshim_udma_wait(h1)
        _ = qshim_udma_wait(h2)
    hmx_sync_hvx_and_dma_threads(thread_id)

    # ---- Stack-allocated dims ----
    var rhs_dims3 = QAicHMXDims(
        n_dim=Int32(1),
        x_dim=Int32(k),
        y_dim=Int32(n),
        d_dim=Int32(0),
        elem_size=Int32(0),
    )
    var wt_dims3 = QAicHMXDims(
        n_dim=Int32(1),
        x_dim=Int32(k),
        y_dim=Int32(n),
        d_dim=Int32(0),
        elem_size=Int32(0),
    )

    # ---- Phase 2: Convert to crouton (HVX threads 0-3) ----
    if not is_hmx:
        _ = hmx_rm_matmul_rhs_to_crouton_16b(
            rhs_crouton_vtcm,
            rhs_vtcm,
            UnsafePointer[UInt8, MutUnsafeAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=rhs_dims3))
            ),
            Int32(thread_id),
        )
        _ = hmx_rm_matmul_lhs_to_crouton_16b(
            lhs_crouton_vtcm,
            lhs_vtcm,
            UnsafePointer[UInt8, MutUnsafeAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=lhs_crouton))
            ),
            UnsafePointer[UInt8, MutUnsafeAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=lhs_flat))
            ),
            Int32(thread_id),
        )
    hmx_sync_hvx_and_hmx_threads(thread_id)

    # ---- Phase 3: HMX matmul (thread 4) ----
    if is_hmx:
        var lhs_input = lhs_crouton_vtcm
        _ = hmx_matmul_with_crouton_hf(
            out_crouton_vtcm,
            UnsafePointer[
                UnsafePointer[UInt8, MutUnsafeAnyOrigin], MutUnsafeAnyOrigin
            ](unsafe_from_address=Int(UnsafePointer(to=lhs_input))),
            rhs_crouton_vtcm,
            UnsafePointer[UInt8, MutUnsafeAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=out_flat))
            ),
            UnsafePointer[UInt8, MutUnsafeAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=out_crouton))
            ),
            UnsafePointer[UInt8, MutUnsafeAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=lhs_crouton))
            ),
            UnsafePointer[UInt8, MutUnsafeAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=wt_dims3))
            ),
            Int32(1),
            Int32(1),
        )
    hmx_sync_hvx_and_hmx_threads(thread_id)

    # ---- Phase 4: Convert output crouton -> row-major (HVX threads) ----
    if not is_hmx:
        _ = hmx_matmul_crouton_to_rm_16b(
            out_vtcm,
            out_crouton_vtcm,
            UnsafePointer[UInt8, MutUnsafeAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=out_flat))
            ),
            UnsafePointer[UInt8, MutUnsafeAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=out_crouton))
            ),
            Int32(thread_id),
        )
    hmx_sync_hvx_and_hmx_threads(thread_id)

    # ---- Phase 5: DMA output VTCM -> DDR (thread 4) ----
    if is_hmx:
        var st_out: UInt32 = 0
        var at_out = QShimUdmaDescCommonAttrs(order_and_reserved=0)
        var h_out = qshim_linear_udma_submit(
            thread_id,
            out_vtcm,
            UInt32(out_bytes),
            c_ptr,
            UnsafePointer[QShimUdmaDescCommonAttrs, MutUnsafeAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=at_out))
            ),
            True,
            UnsafePointer[UInt32, MutUnsafeAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=st_out))
            ),
        )
        _ = qshim_udma_wait(h_out)
    hmx_sync_hvx_and_dma_threads(thread_id)


# ===-----------------------------------------------------------------------===#
# Host-side launch helpers
# ===-----------------------------------------------------------------------===#


def alloc_dev_fp16(
    ctx: DeviceContext, jit_device: QAicDevice, n_elems: Int
) raises -> QAicBuf:
    """Allocate a read-write device DDR buffer holding `n_elems` FP16 values.

    Args:
        ctx: The QAIC DeviceContext.
        jit_device: The JIT device handle.
        n_elems: Number of FP16 elements (buffer is `2 * n_elems` bytes).

    Returns:
        A device buffer handle; release with `qaic_release_jit_buffer`.
    """
    return qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        UInt64(n_elems * 2),
        128,
        0xFFFFFFFFFFFFFFFF,
    )


def compile_and_load_hmx_matmul(ctx: DeviceContext) raises -> QAicJitFunction:
    """Compile `hmx_matmul_kernel` and load it onto `ctx`'s JIT device.

    Args:
        ctx: The QAIC DeviceContext to load the kernel onto.

    Returns:
        A loaded JIT function handle, reusable across matmuls of any shape.
    """
    var elf = compile_info_npu[hmx_matmul_kernel]()
    return qaic_load_jit_function(
        ctx,
        elf.function_name,
        elf.asm.unsafe_ptr().bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(elf.asm.byte_length()),
    )


def run_hmx_matmul(
    ctx: DeviceContext,
    jit_device: QAicDevice,
    jit_func: QAicJitFunction,
    c_buf: QAicBuf,
    a_buf: QAicBuf,
    b_buf: QAicBuf,
    m: Int,
    k: Int,
    n: Int,
) raises:
    """Run `C = A * B` on device, blocking until the kernel completes.

    `a_buf`/`b_buf` must already hold the row-major FP16 inputs (`[m, k]` and
    `[k, n]`); `c_buf` receives the row-major FP16 output (`[m, n]`). Dimensions
    are uploaded into scratch device buffers, so the same `jit_func` works for
    any shape. Execution is synchronous: on return, `c_buf` holds the result.

    Args:
        ctx: The QAIC DeviceContext.
        jit_device: The JIT device handle (for dim-buffer allocation).
        jit_func: A function handle from `compile_and_load_hmx_matmul`.
        c_buf: Output buffer, `[m, n]` FP16.
        a_buf: LHS buffer, `[m, k]` FP16.
        b_buf: RHS buffer, `[k, n]` FP16.
        m: Output rows.
        k: Contraction dimension.
        n: Output columns.
    """
    comptime ALIGNMENT: UInt32 = 128
    comptime CU_BITMAP: UInt64 = 0xFFFFFFFFFFFFFFFF
    comptime DIM_BYTES: UInt64 = 4

    var m_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        DIM_BYTES,
        ALIGNMENT,
        CU_BITMAP,
    )
    var k_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        DIM_BYTES,
        ALIGNMENT,
        CU_BITMAP,
    )
    var n_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        DIM_BYTES,
        ALIGNMENT,
        CU_BITMAP,
    )

    var m_host = alloc[UInt32](1)
    var k_host = alloc[UInt32](1)
    var n_host = alloc[UInt32](1)
    m_host[0] = UInt32(m)
    k_host[0] = UInt32(k)
    n_host[0] = UInt32(n)
    qaic_copy_jit_buffer(
        ctx,
        m_buf,
        m_host.bitcast[UInt8]().as_unsafe_any_origin(),
        DIM_BYTES,
        QAIC_COPY_TO_DEVICE,
    )
    qaic_copy_jit_buffer(
        ctx,
        k_buf,
        k_host.bitcast[UInt8]().as_unsafe_any_origin(),
        DIM_BYTES,
        QAIC_COPY_TO_DEVICE,
    )
    qaic_copy_jit_buffer(
        ctx,
        n_buf,
        n_host.bitcast[UInt8]().as_unsafe_any_origin(),
        DIM_BYTES,
        QAIC_COPY_TO_DEVICE,
    )

    var buffers = alloc[QAicBuf](6)
    buffers[0] = c_buf
    buffers[1] = a_buf
    buffers[2] = b_buf
    buffers[3] = m_buf
    buffers[4] = k_buf
    buffers[5] = n_buf

    # num_cores=1, thread_mask=0x1F (threads 0-4). Blocks until complete.
    qaic_execute_jit_function(
        ctx, jit_func, buffers.as_unsafe_any_origin(), 6, 1, 0x1F
    )

    m_host.unsafe_free()
    k_host.unsafe_free()
    n_host.unsafe_free()
    buffers.unsafe_free()
    qaic_release_jit_buffer(ctx, m_buf)
    qaic_release_jit_buffer(ctx, k_buf)
    qaic_release_jit_buffer(ctx, n_buf)
