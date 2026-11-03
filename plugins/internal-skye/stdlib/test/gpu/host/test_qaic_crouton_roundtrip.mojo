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
"""Empirical investigation of DMA + crouton conversion, isolated from HMX.

Every existing HMX test couples crouton conversion to a compute step (matmul or
softmax) and then checks the *numeric* result against a CPU reference. That
masks conversion/DMA bugs whose error is smaller than the matmul tolerance, and
it never proves the conversion is the exact permutation it claims to be.

This test removes the HMX compute entirely and exercises just the two corners
of the intersection that move bytes around: the uDMA engine (DDR <-> VTCM) and
the HVX crouton <-> row-major conversions. Because no arithmetic happens, the
round trip `row-major -> crouton -> row-major` is a pure permutation of FP16
values, so the result must be **bit-identical** to the input. We assert exact
equality, not a tolerance.

Two pipelines are characterized:

1. `crouton_roundtrip_kernel` — DMA in, pack to crouton, unpack to row-major,
   DMA out. Isolates the HVX conversion pair.

2. `crouton_via_ddr_kernel` — additionally bounces the *crouton-laid-out*
   buffer out to DDR and back through the uDMA engine between pack and unpack.
   This proves the DMA engine carries crouton-format bytes (2048-byte tiles,
   not row-major) losslessly — the "DMA fidelity for crouton buffers" case that
   nothing else covers.

Thread model: 5 threads (thread_mask=0x1F).
- Thread 4 (HMX/DMA): bias init + all DMA.
- Threads 0-3 (HVX): crouton pack/unpack.

Shapes are restricted to multiples of 32 (whole crouton tiles); sub-tile and
unaligned shapes are characterized separately in test_qaic_crouton_padding.mojo.
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


# ===-----------------------------------------------------------------------===#
# Device kernels
# ===-----------------------------------------------------------------------===#


def crouton_roundtrip_kernel(
    out_ptr: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    inp_ptr: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    m_ptr: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
    k_ptr: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    """Row-major [M,K] -> crouton -> row-major, all in VTCM.

    Phases (thread_mask=0x1F):
    - 0: thread 4 HMX bias init; sync.
    - 1: thread 4 DMA input DDR->VTCM; sync.
    - 2: HVX threads pack VTCM row-major -> crouton; sync.
    - 3: HVX threads unpack crouton -> VTCM row-major; sync.
    - 4: thread 4 DMA output VTCM->DDR; sync.
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

    # Phase 1: DMA input DDR -> VTCM.
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

    # Phase 3: unpack crouton -> row-major.
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

    # Phase 4: DMA output VTCM -> DDR.
    if is_hmx:
        var st: UInt32 = 0
        var at = QShimUdmaDescCommonAttrs(order_and_reserved=0)
        var h = qshim_linear_udma_submit(
            thread_id,
            out_vtcm,
            UInt32(data_bytes),
            out_ptr,
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


def crouton_via_ddr_kernel(
    out_ptr: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    inp_ptr: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    crouton_ddr_ptr: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    m_ptr: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
    k_ptr: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    """As above, but DMA the crouton buffer out to DDR and back between
    pack and unpack — exercising uDMA on crouton-laid-out (not row-major) bytes.
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
    off = vtcm_align(off, HMX_ALIGNMENT)
    var crouton2_vtcm = vtcm + off
    off += crouton_bytes
    off = vtcm_align(off, HMX_INPUT_ALIGNMENT)
    var out_vtcm = vtcm + off

    # Phase 1: DMA input DDR -> VTCM.
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

    # Phase 2: pack row-major -> crouton (VTCM).
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

    # Phase 3: bounce the crouton bytes VTCM -> DDR -> VTCM.
    if is_hmx:
        var st: UInt32 = 0
        var at = QShimUdmaDescCommonAttrs(order_and_reserved=0)
        var h_out = qshim_linear_udma_submit(
            thread_id,
            crouton_vtcm,
            UInt32(crouton_bytes),
            crouton_ddr_ptr,
            UnsafePointer[QShimUdmaDescCommonAttrs, MutUnsafeAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=at))
            ),
            True,
            UnsafePointer[UInt32, MutUnsafeAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=st))
            ),
        )
        _ = qshim_udma_wait(h_out)
        var h_in = qshim_linear_udma_submit(
            thread_id,
            crouton_ddr_ptr,
            UInt32(crouton_bytes),
            crouton2_vtcm,
            UnsafePointer[QShimUdmaDescCommonAttrs, MutUnsafeAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=at))
            ),
            True,
            UnsafePointer[UInt32, MutUnsafeAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=st))
            ),
        )
        _ = qshim_udma_wait(h_in)
    hmx_sync_hvx_and_dma_threads(thread_id)

    # Phase 4: unpack the round-tripped crouton -> row-major.
    if not is_hmx:
        _ = hmx_matmul_crouton_to_rm_16b(
            out_vtcm,
            crouton2_vtcm,
            UnsafePointer[UInt8, MutUnsafeAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=flat))
            ),
            UnsafePointer[UInt8, MutUnsafeAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=crouton))
            ),
            Int32(thread_id),
        )
    hmx_sync_hvx_and_hmx_threads(thread_id)

    # Phase 5: DMA output VTCM -> DDR.
    if is_hmx:
        var st: UInt32 = 0
        var at = QShimUdmaDescCommonAttrs(order_and_reserved=0)
        var h = qshim_linear_udma_submit(
            thread_id,
            out_vtcm,
            UInt32(data_bytes),
            out_ptr,
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


# ===-----------------------------------------------------------------------===#
# Host helpers
# ===-----------------------------------------------------------------------===#


def count_mismatches(
    actual: UnsafePointer[Float16, MutUntrackedOrigin],
    expected: UnsafePointer[Float16, MutUntrackedOrigin],
    size: Int,
    max_to_print: Int = 5,
) -> Int:
    """Count elements that differ at all (the round trip must be exact)."""
    var mismatches = 0
    for i in range(size):
        if actual[i] != expected[i]:
            mismatches += 1
            if mismatches <= max_to_print:
                print(
                    "    MISMATCH at",
                    i,
                    ": in=",
                    Float32(expected[i]),
                    "out=",
                    Float32(actual[i]),
                )
    return mismatches


def _alloc_io(
    ctx: DeviceContext, jit_device: QAicDevice, nbytes: Int
) raises -> QAicBuf:
    return qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        UInt64(nbytes),
        128,
        0xFFFFFFFFFFFFFFFF,
    )


def run_vtcm_roundtrip(M: Int, K: Int, ctx: DeviceContext) raises:
    print("  [VTCM round-trip] shape", M, "x", K)
    var size = M * K
    var data_bytes = size * 2
    var jit_device = qaic_get_jit_device(ctx)

    var inp_buf = _alloc_io(ctx, jit_device, data_bytes)
    var out_buf = _alloc_io(ctx, jit_device, data_bytes)
    var m_buf = _alloc_io(ctx, jit_device, 4)
    var k_buf = _alloc_io(ctx, jit_device, 4)

    var inp_host = alloc[Float16](size)
    var out_host = alloc[Float16](size)
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

    var elf = compile_info_npu[crouton_roundtrip_kernel]()
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

    var mismatches = count_mismatches(out_host, inp_host, size)
    if mismatches == 0:
        print("    PASSED: bit-exact over all", size, "elements")
    else:
        print("    FAILED:", mismatches, "of", size, "elements differ")

    inp_host.unsafe_free()
    out_host.unsafe_free()
    m_host.unsafe_free()
    k_host.unsafe_free()
    buffers.unsafe_free()
    qaic_release_jit_buffer(ctx, inp_buf)
    qaic_release_jit_buffer(ctx, out_buf)
    qaic_release_jit_buffer(ctx, m_buf)
    qaic_release_jit_buffer(ctx, k_buf)
    qaic_release_jit_function(jit_func)

    assert_equal(mismatches, 0)


def run_ddr_roundtrip(M: Int, K: Int, ctx: DeviceContext) raises:
    print("  [crouton-via-DDR] shape", M, "x", K)
    var size = M * K
    var data_bytes = size * 2

    var flat = hmx_flat_dims_yd(M, K)
    var crouton = hmx_crouton_dims_from_flat(flat)
    var crouton_bytes = hmx_crouton_size_in_bytes(crouton)

    var jit_device = qaic_get_jit_device(ctx)
    var inp_buf = _alloc_io(ctx, jit_device, data_bytes)
    var out_buf = _alloc_io(ctx, jit_device, data_bytes)
    var crouton_buf = _alloc_io(ctx, jit_device, crouton_bytes)
    var m_buf = _alloc_io(ctx, jit_device, 4)
    var k_buf = _alloc_io(ctx, jit_device, 4)

    var inp_host = alloc[Float16](size)
    var out_host = alloc[Float16](size)
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

    var elf = compile_info_npu[crouton_via_ddr_kernel]()
    var jit_func = qaic_load_jit_function(
        ctx,
        elf.function_name,
        elf.asm.unsafe_ptr().bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(elf.asm.byte_length()),
    )

    var buffers = alloc[QAicBuf](5)
    buffers[0] = out_buf
    buffers[1] = inp_buf
    buffers[2] = crouton_buf
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
        out_buf,
        out_host.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(data_bytes),
        QAIC_COPY_FROM_DEVICE,
    )

    var mismatches = count_mismatches(out_host, inp_host, size)
    if mismatches == 0:
        print("    PASSED: bit-exact after crouton DMA round-trip")
    else:
        print("    FAILED:", mismatches, "of", size, "elements differ")

    inp_host.unsafe_free()
    out_host.unsafe_free()
    m_host.unsafe_free()
    k_host.unsafe_free()
    buffers.unsafe_free()
    qaic_release_jit_buffer(ctx, inp_buf)
    qaic_release_jit_buffer(ctx, out_buf)
    qaic_release_jit_buffer(ctx, crouton_buf)
    qaic_release_jit_buffer(ctx, m_buf)
    qaic_release_jit_buffer(ctx, k_buf)
    qaic_release_jit_function(jit_func)

    assert_equal(mismatches, 0)


# ===-----------------------------------------------------------------------===#
# Tests
# ===-----------------------------------------------------------------------===#


def test_crouton_roundtrip_vtcm() raises:
    print_test_header("crouton round-trip (VTCM only)")
    var ctx = DeviceContext(device_id=0, api="qaic")
    run_vtcm_roundtrip(32, 32, ctx)
    run_vtcm_roundtrip(64, 64, ctx)
    run_vtcm_roundtrip(128, 128, ctx)
    run_vtcm_roundtrip(32, 128, ctx)
    run_vtcm_roundtrip(128, 32, ctx)
    run_vtcm_roundtrip(256, 64, ctx)
    run_vtcm_roundtrip(64, 256, ctx)
    print_separator()
    print("TEST PASSED!")
    print_separator()


def test_crouton_roundtrip_via_ddr() raises:
    print_test_header("crouton round-trip (bounced through DDR)")
    var ctx = DeviceContext(device_id=0, api="qaic")
    run_ddr_roundtrip(32, 32, ctx)
    run_ddr_roundtrip(64, 64, ctx)
    run_ddr_roundtrip(128, 128, ctx)
    run_ddr_roundtrip(128, 256, ctx)
    print_separator()
    print("TEST PASSED!")
    print_separator()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
