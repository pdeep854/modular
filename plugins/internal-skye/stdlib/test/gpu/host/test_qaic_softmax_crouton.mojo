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
"""Test QAIC HMX softmax on crouton-format tensors.

Tests qaic_softmax_with_crouton_hf by:
1. DMA'ing a FP16 [M, K] row-major tensor from DDR into VTCM.
2. Converting the VTCM input to crouton format (HVX threads 0-3).
3. Running softmax along the K (D) dimension (HVX threads 0-3 only).
4. Converting the crouton result back to row-major in VTCM (HVX threads 0-3).
5. DMA'ing the result back to DDR.
6. Comparing against a CPU reference softmax.

Thread model: 5 threads (thread_mask=0x1F).
- Thread 4 (DMA/HMX): hmx_matmul_bias_init_16b + DMA in/out only.
- Threads 0-3 (HVX): crouton conversion and softmax.

DEV_ATTR_QSHIM_NUM_HVX_UNITS = 4 on AI100 (threads 0-3 are HVX-capable).
The softmax library uses a hardcoded count=4 for its internal
qaicSyncHVXThread barriers. All 4 HVX threads must participate, so the
input shape must have at least 4 Y-tiles (M >= 128) to ensure every HVX
thread receives at least one tile and participates in the barriers.
"""

from common import print_separator, print_test_header
from qaic_test_utils.rng import Xorshift32
from qaic_test_utils.hmx import vtcm_align, npu_is_hmx_thread

from std.memory import UnsafePointer, alloc
from std.testing import TestSuite, assert_equal
from std.math import exp

from max.gpu.host import DeviceContext
from std._plugin.qaic.host.target import (
    compile_info_npu,
    npu_thread_idx,
)
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
    hmx_softmax_with_crouton_hf,
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
    QAIC_BUFFER_TYPE_DEV_DDR,
    QAIC_DEV_BUF_IO_RW,
    QAIC_COPY_TO_DEVICE,
    QAIC_COPY_FROM_DEVICE,
)


# HVX vector size on Hexagon V68 in bytes.
comptime HVX_VECTOR_SIZE: Int = 128

# Scratch = 2 * numHVXUnits * HVX_VectorSize (from QAicHexagonHMX.h).
# DEV_ATTR_QSHIM_NUM_HVX_UNITS = 4 on AI100 (threads 0-3 are HVX-capable).
comptime NUM_HVX_UNITS: Int = 4
comptime SOFTMAX_SCRATCH_BYTES: Int = 2 * NUM_HVX_UNITS * HVX_VECTOR_SIZE


# ===-----------------------------------------------------------------------===#
# Device Kernel
# ===-----------------------------------------------------------------------===#


def hmx_softmax_kernel(
    out_ptr: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    inp_ptr: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    m_ptr: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
    k_ptr: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    """Softmax kernel: row-major FP16 [M, K] in DDR → row-major FP16 [M, K] out.

    5-thread model (thread_mask=0x1F):
    - Phase 0: thread 4 calls hmx_matmul_bias_init_16b; all sync (hvx+hmx).
    - Phase 1: thread 4 DMAs input DDR→VTCM; all sync (hvx+dma).
    - Phase 2: HVX threads (0-3) convert row-major VTCM → crouton VTCM;
               all sync (hvx+hmx).
    - Phase 3: HVX threads (0-3) call hmx_softmax_with_crouton_hf.
               The library distributes Y-tiles across the 4 HVX threads;
               each thread must receive ≥1 tile (requires M ≥ 128).
               All sync (hvx+hmx) after softmax completes.
    - Phase 4: HVX threads (0-3) convert crouton VTCM → row-major VTCM;
               all sync (hvx+hmx).
    - Phase 5: thread 4 DMAs output VTCM→DDR; all sync (hvx+dma).
    """
    var thread_id = UInt32(npu_thread_idx())
    var is_hmx = npu_is_hmx_thread()

    var m = Int(m_ptr[0])
    var k = Int(k_ptr[0])
    var data_bytes = m * k * 2

    var flat_dims = hmx_flat_dims_yd(m, k)
    var crouton_dims = hmx_crouton_dims_from_flat(flat_dims)
    var crouton_bytes = hmx_crouton_size_in_bytes(crouton_dims)

    # ---- Phase 0: HMX bias init (thread 4 only) ----
    if is_hmx:
        _ = hmx_matmul_bias_init_16b()
    hmx_sync_hvx_and_hmx_threads(thread_id)

    # ---- VTCM layout ----
    var vtcm_base = qshim_get_base_vtcm_addr()
    var off = 0

    off = vtcm_align(off, HMX_INPUT_ALIGNMENT)
    var inp_vtcm = vtcm_base + off
    off += data_bytes

    off = vtcm_align(off, HMX_ALIGNMENT)
    var crouton_in_vtcm = vtcm_base + off
    off += crouton_bytes

    off = vtcm_align(off, HMX_ALIGNMENT)
    var crouton_out_vtcm = vtcm_base + off
    off += crouton_bytes

    off = vtcm_align(off, HMX_INPUT_ALIGNMENT)
    var out_vtcm = vtcm_base + off
    off += data_bytes

    # Scratch: 2 * numHVXUnits * HVX_VECTOR_SIZE = 2 * 4 * 128 = 1024 bytes.
    # Only HVX threads (0-3) write to scratch[0..3]; thread 4 does not touch it.
    off = vtcm_align(off, HVX_VECTOR_SIZE)
    var scratch_vtcm = vtcm_base + off

    # ---- Phase 1: DMA input DDR → VTCM (thread 4) ----
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
    hmx_sync_hvx_and_dma_threads(thread_id)

    # ---- Phase 2: VTCM row-major → crouton (HVX threads 0-3) ----
    if not is_hmx:
        _ = hmx_rm_matmul_lhs_to_crouton_16b(
            crouton_in_vtcm,
            inp_vtcm,
            UnsafePointer[UInt8, MutUnsafeAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=crouton_dims))
            ),
            UnsafePointer[UInt8, MutUnsafeAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=flat_dims))
            ),
            Int32(thread_id),
        )
    hmx_sync_hvx_and_hmx_threads(thread_id)

    # ---- Phase 3: Softmax (HVX threads 0-3, crouton_in → crouton_out) ----
    if not is_hmx:
        _ = hmx_softmax_with_crouton_hf(
            crouton_in_vtcm,
            crouton_out_vtcm,
            scratch_vtcm,
            UnsafePointer[UInt8, MutUnsafeAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=flat_dims))
            ),
            Int32(thread_id),
        )
    hmx_sync_hvx_and_hmx_threads(thread_id)

    # ---- Phase 4: crouton → VTCM row-major (HVX threads 0-3) ----
    if not is_hmx:
        _ = hmx_matmul_crouton_to_rm_16b(
            out_vtcm,
            crouton_out_vtcm,
            UnsafePointer[UInt8, MutUnsafeAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=flat_dims))
            ),
            UnsafePointer[UInt8, MutUnsafeAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=crouton_dims))
            ),
            Int32(thread_id),
        )
    hmx_sync_hvx_and_hmx_threads(thread_id)

    # ---- Phase 5: DMA output VTCM → DDR (thread 4) ----
    if is_hmx:
        var st_out: UInt32 = 0
        var at_out = QShimUdmaDescCommonAttrs(order_and_reserved=0)
        var h_out = qshim_linear_udma_submit(
            thread_id,
            out_vtcm,
            UInt32(data_bytes),
            out_ptr,
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
# CPU Reference
# ===-----------------------------------------------------------------------===#


def cpu_softmax_f16(
    inp: UnsafePointer[Float16, MutUntrackedOrigin],
    output: UnsafePointer[Float16, MutUntrackedOrigin],
    m: Int,
    k: Int,
):
    """Compute row-wise softmax in FP32 precision, store as FP16."""
    for i in range(m):
        var row = inp + i * k

        var max_val = Float32(row[0])
        for j in range(1, k):
            var v = Float32(row[j])
            if v > max_val:
                max_val = v

        var sum_exp = Float32(0)
        for j in range(k):
            sum_exp += exp(Float32(row[j]) - max_val)

        var inv_sum = Float32(1) / sum_exp
        for j in range(k):
            output[i * k + j] = Float16(
                exp(Float32(row[j]) - max_val) * inv_sum
            )


# ===-----------------------------------------------------------------------===#
# Verification
# ===-----------------------------------------------------------------------===#


def verify_softmax_f16(
    actual: UnsafePointer[Float16, MutUntrackedOrigin],
    expected: UnsafePointer[Float16, MutUntrackedOrigin],
    size: Int,
    atol: Float32 = 0.02,
    max_errors_to_print: Int = 5,
) -> Int:
    var errors = 0
    var max_err = Float32(0)
    for idx in range(size):
        var act = Float32(actual[idx])
        var exp_val = Float32(expected[idx])
        var err = abs(act - exp_val)
        if err > max_err:
            max_err = err
        if err > atol:
            errors += 1
            if errors <= max_errors_to_print:
                print(
                    "  ERROR at idx",
                    idx,
                    ": expected",
                    exp_val,
                    "got",
                    act,
                    "err",
                    err,
                )
    print("  Max absolute error:", max_err)
    return errors


# ===-----------------------------------------------------------------------===#
# Test runner
# ===-----------------------------------------------------------------------===#


def run_hmx_softmax(M: Int, K: Int, ctx: DeviceContext) raises:
    print("  Shape:", M, "x", K)

    var size = M * K
    var data_bytes = size * 2
    var dim_bytes: Int = 4
    var alignment: UInt32 = 128
    var cu_bitmap: UInt64 = 0xFFFFFFFFFFFFFFFF

    var jit_device = qaic_get_jit_device(ctx)

    var inp_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        UInt64(data_bytes),
        alignment,
        cu_bitmap,
    )
    var out_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        UInt64(data_bytes),
        alignment,
        cu_bitmap,
    )
    var m_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        UInt64(dim_bytes),
        alignment,
        cu_bitmap,
    )
    var k_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        UInt64(dim_bytes),
        alignment,
        cu_bitmap,
    )

    var inp_host = alloc[Float16](size)
    var out_host = alloc[Float16](size)
    var ref_host = alloc[Float16](size)

    var rng = Xorshift32.from_time()
    rng.fill_fp16(inp_host, size)

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
        UInt64(dim_bytes),
        QAIC_COPY_TO_DEVICE,
    )
    qaic_copy_jit_buffer(
        ctx,
        k_buf,
        k_host.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(dim_bytes),
        QAIC_COPY_TO_DEVICE,
    )

    var elf = compile_info_npu[hmx_softmax_kernel]()
    var jit_func = qaic_load_jit_function(
        ctx,
        elf.function_name,
        elf.asm.unsafe_ptr().bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(elf.asm.byte_length()),
    )

    var buffers = alloc[QAicBuf](4)
    buffers[0] = out_buf
    buffers[1] = inp_buf
    buffers[2] = m_buf
    buffers[3] = k_buf
    qaic_execute_jit_function(
        ctx,
        jit_func,
        buffers.as_unsafe_any_origin(),
        4,
        num_cores=1,
        thread_mask=UInt32(0x1F),
    )

    qaic_copy_jit_buffer(
        ctx,
        out_buf,
        out_host.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(data_bytes),
        QAIC_COPY_FROM_DEVICE,
    )

    cpu_softmax_f16(inp_host, ref_host, M, K)

    print("  First 8 values — HMX vs CPU reference:")
    for i in range(min(8, size)):
        var act = Float32(out_host[i])
        var exp_val = Float32(ref_host[i])
        print(
            "    ["
            + String(i)
            + "] inp="
            + String(Float32(inp_host[i]))
            + " hmx="
            + String(act)
            + " ref="
            + String(exp_val)
            + " err="
            + String(abs(act - exp_val))
        )

    var errors = verify_softmax_f16(out_host, ref_host, size)
    if errors == 0:
        print("  PASSED: all", size, "elements within tolerance")
    else:
        print("  FAILED:", errors, "errors out of", size)

    inp_host.unsafe_free()
    out_host.unsafe_free()
    ref_host.unsafe_free()
    m_host.unsafe_free()
    k_host.unsafe_free()
    buffers.unsafe_free()
    qaic_release_jit_buffer(ctx, inp_buf)
    qaic_release_jit_buffer(ctx, out_buf)
    qaic_release_jit_buffer(ctx, m_buf)
    qaic_release_jit_buffer(ctx, k_buf)
    qaic_release_jit_function(jit_func)

    assert_equal(errors, 0)


def test_qaic_softmax_crouton() raises:
    print_test_header("QAIC HMX Softmax (crouton)")
    print()

    var ctx = DeviceContext(device_id=0, api="qaic")

    # Shapes must be multiples of 32 (crouton tile size) and M >= 128
    # (so each of the 4 HVX threads receives at least one Y-tile).
    run_hmx_softmax(32, 32, ctx)
    run_hmx_softmax(64, 64, ctx)
    run_hmx_softmax(128, 256, ctx)

    print()
    print_separator()
    print("TEST PASSED!")
    print_separator()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
