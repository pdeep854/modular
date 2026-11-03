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
"""Characterize the concurrency available to the DMA/HMX/crouton pipeline.

The crouton round-trip and HMX matmul tests run the pipeline in its least
concurrent form (single NSP, fully barrier-serialized phases). This file
measures the parallelism the hardware actually offers:

1. `test_npu_topology` — dispatch to many NSPs and have every (core, thread)
   worker report its identity and the runtime's view of the machine. Settles how
   many NSP cores and HVX threads/core are really available (source docs
   disagree: 4 vs 16 cores), which bounds every other form of concurrency.

2. `test_dma_compute_overlap` — the uDMA engine (thread 4) and the HVX units
   (threads 0-3) are distinct hardware. Run a DMA-only kernel, a
   crouton-pack-only kernel, and one that does both with no barrier between, then
   compare wall times. `t_both ≈ max(t_dma, t_compute)` means DMA fully overlaps
   compute (so software pipelining pays off); `t_both ≈ t_dma + t_compute` means
   it does not.

3. `test_nsp_scaling` — run the real crouton round-trip pipeline with its rows
   partitioned across N NSPs (each NSP independently round-trips its slice in its
   own VTCM) for N in {1,2,4,8,16}, and report the speedup curve. Sub-linear
   scaling localizes the shared bottleneck (almost certainly DDR bandwidth, since
   the per-NSP work is otherwise independent).

4. `test_dma_any_thread` — have all five threads each DMA their own slice and
   check every slice round-trips, answering whether uDMA is restricted to the
   HMX thread (thread 4) or available to any thread.

5. `test_hmx_dma_overlap` — run HMX matmuls on thread 4 while thread 0 (an HVX
   thread, per finding 4) drives uDMA, and measure how much they overlap. On the
   standard pipeline HMX and DMA share thread 4 and cannot overlap; moving DMA to
   another thread is the only way to run them concurrently.

Timing is host wall-clock around the synchronous `qaic_execute_jit_function`,
min-of-N to reject scheduler noise. Registered `manual`: it is a print-heavy
characterization (and a perf measurement, which should not gate `//...`).
"""

from common import print_separator, print_test_header
from qaic_test_utils.rng import Xorshift32
from qaic_test_utils.hmx import vtcm_align, npu_is_hmx_thread

from std.memory import UnsafePointer, alloc
from std.testing import TestSuite, assert_equal, assert_true
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext
from std._plugin.qaic.host.target import (
    compile_info_npu,
    npu_core_idx,
    npu_thread_idx,
    npu_num_cores,
    npu_num_threads,
    npu_global_idx,
)
from std._plugin.qaic.host.qshim import (
    qshim_get_base_vtcm_addr,
    qshim_get_num_hvx_units,
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
    qaic_get_jit_device,
    qaic_allocate_jit_buffer,
    qaic_release_jit_buffer,
    qaic_load_jit_function,
    qaic_execute_jit_function,
    qaic_release_jit_function,
    qaic_copy_jit_buffer,
    QAicBuf,
    QAicDevice,
    QAicJitFunction,
    QAIC_BUFFER_TYPE_DEV_DDR,
    QAIC_DEV_BUF_IO_RW,
    QAIC_COPY_TO_DEVICE,
    QAIC_COPY_FROM_DEVICE,
)


# ===-----------------------------------------------------------------------===#
# Shared host helpers
# ===-----------------------------------------------------------------------===#


def _alloc_ddr(
    ctx: DeviceContext, dev: QAicDevice, nbytes: Int
) raises -> QAicBuf:
    return qaic_allocate_jit_buffer(
        ctx,
        dev,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        UInt64(nbytes),
        128,
        0xFFFFFFFFFFFFFFFF,
    )


def _u32_buf(
    ctx: DeviceContext, dev: QAicDevice, value: UInt32
) raises -> QAicBuf:
    """Allocate a 1-element UInt32 device buffer holding `value`."""
    var buf = _alloc_ddr(ctx, dev, 4)
    var host = alloc[UInt32](1)
    host[0] = value
    qaic_copy_jit_buffer(
        ctx, buf, host.bitcast[UInt8](), UInt64(4), QAIC_COPY_TO_DEVICE
    )
    host.unsafe_free()
    return buf


# ===-----------------------------------------------------------------------===#
# (1) Topology probe
# ===-----------------------------------------------------------------------===#

comptime TOPO_MAX_CORES: Int = 16
comptime TOPO_MAX_THREADS: Int = 8
comptime TOPO_FIELDS: Int = 6
comptime TOPO_CAP: Int = TOPO_MAX_CORES * TOPO_MAX_THREADS * TOPO_FIELDS


def topology_kernel(out_ptr: UnsafePointer[UInt32, MutUnsafeAnyOrigin]):
    """Each worker writes its identity + the runtime's machine view to its slot.

    Slot layout (6 UInt32): core_id, thread_id, num_cores, num_threads,
    num_hvx_units, global_idx. Slot index is core_id*TOPO_MAX_THREADS+thread_id
    so a fixed host-side stride works regardless of the real thread count.
    """
    var core = Int(npu_core_idx())
    var thread = Int(npu_thread_idx())
    var gid = core * TOPO_MAX_THREADS + thread
    var base = gid * TOPO_FIELDS
    if base + TOPO_FIELDS <= TOPO_CAP:
        out_ptr[base + 0] = UInt32(core)
        out_ptr[base + 1] = UInt32(thread)
        out_ptr[base + 2] = UInt32(npu_num_cores())
        out_ptr[base + 3] = UInt32(npu_num_threads())
        out_ptr[base + 4] = UInt32(qshim_get_num_hvx_units())
        out_ptr[base + 5] = UInt32(npu_global_idx())


def test_npu_topology() raises:
    print_test_header("NPU topology: how many NSP cores / HVX threads?")
    var ctx = DeviceContext(device_id=0, api="qaic")
    var dev = qaic_get_jit_device(ctx)

    var n = TOPO_CAP
    var out_buf = _alloc_ddr(ctx, dev, n * 4)
    var host = alloc[UInt32](n)
    for i in range(n):
        host[i] = 0xFFFFFFFF
    qaic_copy_jit_buffer(
        ctx, out_buf, host.bitcast[UInt8](), UInt64(n * 4), QAIC_COPY_TO_DEVICE
    )

    var elf = compile_info_npu[topology_kernel]()
    var func = qaic_load_jit_function(
        ctx,
        elf.function_name,
        elf.asm.unsafe_ptr().bitcast[UInt8](),
        UInt64(elf.asm.byte_length()),
    )

    var buffers = alloc[QAicBuf](1)
    buffers[0] = out_buf
    # Request the production matmul's default width; report what actually runs.
    qaic_execute_jit_function(
        ctx,
        func,
        buffers,
        1,
        num_cores=UInt32(TOPO_MAX_CORES),
        thread_mask=UInt32(0x1F),
    )

    qaic_copy_jit_buffer(
        ctx,
        out_buf,
        host.bitcast[UInt8](),
        UInt64(n * 4),
        QAIC_COPY_FROM_DEVICE,
    )

    var ran = 0
    var max_core = -1
    var max_thread = -1
    var rep_cores = 0
    var rep_threads = 0
    var rep_hvx = 0
    for c in range(TOPO_MAX_CORES):
        for t in range(TOPO_MAX_THREADS):
            var base = (c * TOPO_MAX_THREADS + t) * TOPO_FIELDS
            if host[base] != 0xFFFFFFFF:
                ran += 1
                var core_id = Int(host[base + 0])
                var thr_id = Int(host[base + 1])
                if core_id > max_core:
                    max_core = core_id
                if thr_id > max_thread:
                    max_thread = thr_id
                rep_cores = Int(host[base + 2])
                rep_threads = Int(host[base + 3])
                rep_hvx = Int(host[base + 4])

    print("  workers that executed:", ran)
    print("  distinct NSP cores observed:", max_core + 1)
    print(
        "  max HVX thread id observed:",
        max_thread,
        "( threads/core:",
        max_thread + 1,
        ")",
    )
    print("  runtime-reported num_cores :", rep_cores)
    print("  runtime-reported num_threads:", rep_threads)
    print("  runtime-reported num_hvx   :", rep_hvx)
    print(
        "  => available concurrency: ",
        max_core + 1,
        "NSP x",
        max_thread + 1,
        "threads =",
        (max_core + 1) * (max_thread + 1),
        "workers",
    )

    host.unsafe_free()
    buffers.unsafe_free()
    qaic_release_jit_buffer(ctx, out_buf)
    qaic_release_jit_function(func)

    assert_true(ran >= 1)
    assert_true(max_core >= 0)


# ===-----------------------------------------------------------------------===#
# (2) DMA / compute overlap
# ===-----------------------------------------------------------------------===#

comptime OVL_M: Int = 128
comptime OVL_K: Int = 128
comptime OVL_BLOCK_BYTES: Int = OVL_M * OVL_K * 2

comptime OVL_MODE_DMA: Int = 0
comptime OVL_MODE_COMPUTE: Int = 1
comptime OVL_MODE_BOTH: Int = 2


def overlap_kernel(
    scratch_ddr: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    mode_ptr: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
    d_ptr: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
    c_ptr: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    """Single-NSP DMA-engine vs HVX overlap probe.

    Thread 4 (DMA) runs `d` linear DDR->VTCM transfers; threads 0-3 (HVX) run
    `c` crouton packs. `mode` selects DMA-only / compute-only / both. The two
    workloads touch disjoint VTCM so `both` is a real concurrency test, not a
    race. A single barrier at the end joins them.
    """
    var thread_id = UInt32(npu_thread_idx())
    var is_hmx = npu_is_hmx_thread()

    var mode = Int(mode_ptr[0])
    var d_iters = Int(d_ptr[0])
    var c_iters = Int(c_ptr[0])

    var flat = hmx_flat_dims_yd(OVL_M, OVL_K)
    var crouton = hmx_crouton_dims_from_flat(flat)
    var crouton_bytes = hmx_crouton_size_in_bytes(crouton)

    if is_hmx:
        _ = hmx_matmul_bias_init_16b()
    hmx_sync_hvx_and_hmx_threads(thread_id)

    var vtcm = qshim_get_base_vtcm_addr()
    var off = 0
    off = vtcm_align(off, HMX_INPUT_ALIGNMENT)
    var dma_dst_vtcm = vtcm + off
    off += OVL_BLOCK_BYTES
    off = vtcm_align(off, HMX_INPUT_ALIGNMENT)
    var pack_src_vtcm = vtcm + off
    off += OVL_BLOCK_BYTES
    off = vtcm_align(off, HMX_ALIGNMENT)
    var crouton_vtcm = vtcm + off

    # Thread 4: DMA loop.
    if is_hmx and (mode == OVL_MODE_DMA or mode == OVL_MODE_BOTH):
        for _ in range(d_iters):
            var st: UInt32 = 0
            var at = QShimUdmaDescCommonAttrs(order_and_reserved=0)
            var h = qshim_linear_udma_submit(
                thread_id,
                scratch_ddr,
                UInt32(OVL_BLOCK_BYTES),
                dma_dst_vtcm,
                UnsafePointer[QShimUdmaDescCommonAttrs, MutUnsafeAnyOrigin](
                    unsafe_from_address=Int(UnsafePointer(to=at))
                ),
                True,
                UnsafePointer[UInt32, MutUnsafeAnyOrigin](
                    unsafe_from_address=Int(UnsafePointer(to=st))
                ),
            )
            _ = qshim_udma_wait(h)

    # HVX threads: crouton pack loop.
    if (not is_hmx) and (mode == OVL_MODE_COMPUTE or mode == OVL_MODE_BOTH):
        for _ in range(c_iters):
            _ = hmx_rm_matmul_lhs_to_crouton_16b(
                crouton_vtcm,
                pack_src_vtcm,
                UnsafePointer[UInt8, MutUnsafeAnyOrigin](
                    unsafe_from_address=Int(UnsafePointer(to=crouton))
                ),
                UnsafePointer[UInt8, MutUnsafeAnyOrigin](
                    unsafe_from_address=Int(UnsafePointer(to=flat))
                ),
                Int32(thread_id),
            )

    hmx_sync_hvx_and_dma_threads(thread_id)


def _time_overlap(
    ctx: DeviceContext,
    func: QAicJitFunction,
    buffers: UnsafePointer[QAicBuf, MutUnsafeAnyOrigin],
    reps: Int,
) raises -> Int:
    var best = -1
    for _ in range(reps):
        var t0 = Int(perf_counter_ns())
        qaic_execute_jit_function(
            ctx, func, buffers, 4, num_cores=UInt32(1), thread_mask=UInt32(0x1F)
        )
        var dt = Int(perf_counter_ns()) - t0
        if best < 0 or dt < best:
            best = dt
    return best


def test_dma_compute_overlap() raises:
    print_test_header("DMA engine vs HVX overlap (single NSP)")
    var ctx = DeviceContext(device_id=0, api="qaic")
    var dev = qaic_get_jit_device(ctx)

    comptime D_ITERS = 400
    comptime C_ITERS = 400
    comptime REPS = 5

    var scratch = _alloc_ddr(ctx, dev, OVL_BLOCK_BYTES)
    var d_buf = _u32_buf(ctx, dev, UInt32(D_ITERS))
    var c_buf = _u32_buf(ctx, dev, UInt32(C_ITERS))

    var elf = compile_info_npu[overlap_kernel]()
    var func = qaic_load_jit_function(
        ctx,
        elf.function_name,
        elf.asm.unsafe_ptr().bitcast[UInt8](),
        UInt64(elf.asm.byte_length()),
    )

    var buffers = alloc[QAicBuf](4)
    buffers[0] = scratch
    buffers[2] = d_buf
    buffers[3] = c_buf

    var mode_dma = _u32_buf(ctx, dev, UInt32(OVL_MODE_DMA))
    buffers[1] = mode_dma
    var t_dma = _time_overlap(ctx, func, buffers, REPS)
    qaic_release_jit_buffer(ctx, mode_dma)

    var mode_cmp = _u32_buf(ctx, dev, UInt32(OVL_MODE_COMPUTE))
    buffers[1] = mode_cmp
    var t_cmp = _time_overlap(ctx, func, buffers, REPS)
    qaic_release_jit_buffer(ctx, mode_cmp)

    var mode_both = _u32_buf(ctx, dev, UInt32(OVL_MODE_BOTH))
    buffers[1] = mode_both
    var t_both = _time_overlap(ctx, func, buffers, REPS)
    qaic_release_jit_buffer(ctx, mode_both)

    print("  DMA-only   (", D_ITERS, "x", OVL_BLOCK_BYTES, "B ):", t_dma, "ns")
    print("  pack-only  (", C_ITERS, "x 128x128 ):", t_cmp, "ns")
    print("  both       (concurrent ):", t_both, "ns")
    var serial = t_dma + t_cmp
    var slowest = t_dma if t_dma > t_cmp else t_cmp
    print("  serial sum (no overlap) :", serial, "ns")
    print("  slowest leg (full overlap):", slowest, "ns")
    if t_both > 0:
        # Fraction of the slower leg that the faster leg hid behind it.
        var hidden = serial - t_both
        print(
            "  overlap: both is",
            (serial * 100) // t_both,
            "% of serial-sum;",
            (hidden * 100) // serial,
            "% of work was hidden",
        )

    buffers.unsafe_free()
    qaic_release_jit_buffer(ctx, scratch)
    qaic_release_jit_buffer(ctx, d_buf)
    qaic_release_jit_buffer(ctx, c_buf)
    qaic_release_jit_function(func)

    # Concurrency sanity: overlapping can't be slower than doing both serially
    # (allow 15% scheduler noise), and shouldn't beat the ideal full-overlap.
    assert_true(t_both <= (serial * 115) // 100)


# ===-----------------------------------------------------------------------===#
# (3) Multi-NSP scaling of the crouton round-trip
# ===-----------------------------------------------------------------------===#

comptime SCALE_M: Int = 512
comptime SCALE_K: Int = 256


def scaling_kernel(
    out_ptr: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    inp_ptr: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    m_total_ptr: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
    k_ptr: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    """Each NSP independently crouton-round-trips its row slice of [M_total, K].

    Rows are split evenly across `npu_num_cores()` NSPs; within an NSP the usual
    5-thread DMA->pack->unpack->DMA pipeline runs in that NSP's own VTCM. No
    cross-NSP communication, so the only shared resource is DDR bandwidth.
    """
    var core = Int(npu_core_idx())
    var ncores = Int(npu_num_cores())
    var thread_id = UInt32(npu_thread_idx())
    var is_hmx = npu_is_hmx_thread()

    var m_total = Int(m_total_ptr[0])
    var k = Int(k_ptr[0])
    var rows = m_total // ncores
    var data_bytes = rows * k * 2
    var byte_off = core * data_bytes
    var my_inp = inp_ptr + byte_off
    var my_out = out_ptr + byte_off

    var flat = hmx_flat_dims_yd(rows, k)
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

    if is_hmx:
        var st: UInt32 = 0
        var at = QShimUdmaDescCommonAttrs(order_and_reserved=0)
        var h = qshim_linear_udma_submit(
            thread_id,
            my_inp,
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

    if is_hmx:
        var st: UInt32 = 0
        var at = QShimUdmaDescCommonAttrs(order_and_reserved=0)
        var h = qshim_linear_udma_submit(
            thread_id,
            out_vtcm,
            UInt32(data_bytes),
            my_out,
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


def _time_ncores(
    ctx: DeviceContext,
    func: QAicJitFunction,
    buffers: UnsafePointer[QAicBuf, MutUnsafeAnyOrigin],
    nc: Int,
    reps: Int,
) raises -> Int:
    """Min wall-clock ns of the scaling kernel dispatched across `nc` NSPs."""
    var best = -1
    for _ in range(reps):
        var t0 = Int(perf_counter_ns())
        qaic_execute_jit_function(
            ctx,
            func,
            buffers,
            4,
            num_cores=UInt32(nc),
            thread_mask=UInt32(0x1F),
        )
        var dt = Int(perf_counter_ns()) - t0
        if best < 0 or dt < best:
            best = dt
    return best


def test_nsp_scaling() raises:
    print_test_header("Multi-NSP scaling of the crouton round-trip")
    var ctx = DeviceContext(device_id=0, api="qaic")
    var dev = qaic_get_jit_device(ctx)

    var size = SCALE_M * SCALE_K
    var data_bytes = size * 2

    var inp_buf = _alloc_ddr(ctx, dev, data_bytes)
    var out_buf = _alloc_ddr(ctx, dev, data_bytes)
    var m_buf = _u32_buf(ctx, dev, UInt32(SCALE_M))
    var k_buf = _u32_buf(ctx, dev, UInt32(SCALE_K))

    var inp_host = alloc[Float16](size)
    var out_host = alloc[Float16](size)
    var rng = Xorshift32.from_time()
    rng.fill_fp16(inp_host, size)
    qaic_copy_jit_buffer(
        ctx,
        inp_buf,
        inp_host.bitcast[UInt8](),
        UInt64(data_bytes),
        QAIC_COPY_TO_DEVICE,
    )

    var elf = compile_info_npu[scaling_kernel]()
    var func = qaic_load_jit_function(
        ctx,
        elf.function_name,
        elf.asm.unsafe_ptr().bitcast[UInt8](),
        UInt64(elf.asm.byte_length()),
    )

    var buffers = alloc[QAicBuf](4)
    buffers[0] = out_buf
    buffers[1] = inp_buf
    buffers[2] = m_buf
    buffers[3] = k_buf

    comptime REPS = 5

    print(
        "  M_total =",
        SCALE_M,
        " K =",
        SCALE_K,
        " (",
        data_bytes,
        "B round-tripped )",
    )

    # 512 is divisible by every count below, so rows/NSP stays a 32-multiple.
    var ns1 = _time_ncores(ctx, func, buffers, 1, REPS)
    var ns2 = _time_ncores(ctx, func, buffers, 2, REPS)
    var ns4 = _time_ncores(ctx, func, buffers, 4, REPS)
    var ns8 = _time_ncores(ctx, func, buffers, 8, REPS)
    var ns16 = _time_ncores(ctx, func, buffers, 16, REPS)

    # out_buf currently holds the 16-NSP result — verify the most-partitioned
    # config reconstructed the whole tensor exactly.
    qaic_copy_jit_buffer(
        ctx,
        out_buf,
        out_host.bitcast[UInt8](),
        UInt64(data_bytes),
        QAIC_COPY_FROM_DEVICE,
    )
    var mismatches = 0
    for i in range(size):
        if out_host[i] != inp_host[i]:
            mismatches += 1

    print("  cores= 1 | rows/NSP=", SCALE_M, "|", ns1, "ns | speedup 100/100x")
    print(
        "  cores= 2 | rows/NSP=",
        SCALE_M // 2,
        "|",
        ns2,
        "ns | speedup",
        ns1 * 100 // ns2,
        "/100x",
    )
    print(
        "  cores= 4 | rows/NSP=",
        SCALE_M // 4,
        "|",
        ns4,
        "ns | speedup",
        ns1 * 100 // ns4,
        "/100x",
    )
    print(
        "  cores= 8 | rows/NSP=",
        SCALE_M // 8,
        "|",
        ns8,
        "ns | speedup",
        ns1 * 100 // ns8,
        "/100x",
    )
    print(
        "  cores=16 | rows/NSP=",
        SCALE_M // 16,
        "|",
        ns16,
        "ns | speedup",
        ns1 * 100 // ns16,
        "/100x",
    )
    print("  16-NSP round-trip mismatches:", mismatches, "of", size)
    assert_equal(mismatches, 0)

    inp_host.unsafe_free()
    out_host.unsafe_free()
    buffers.unsafe_free()
    qaic_release_jit_buffer(ctx, inp_buf)
    qaic_release_jit_buffer(ctx, out_buf)
    qaic_release_jit_buffer(ctx, m_buf)
    qaic_release_jit_buffer(ctx, k_buf)
    qaic_release_jit_function(func)

    print_separator()
    print("TEST PASSED!")
    print_separator()


# ===-----------------------------------------------------------------------===#
# (4) Can DMA run on any thread?
# ===-----------------------------------------------------------------------===#

comptime DMA_ANY_BLOCK_ELEMS: Int = 4096  # FP16 elems per thread
comptime DMA_ANY_THREADS: Int = 5  # thread_mask=0x1F


def dma_any_thread_kernel(
    out_ptr: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    inp_ptr: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
):
    """Every dispatched thread DMAs its own slice DDR -> VTCM -> DDR.

    Slice `t` sits at byte offset `t * block_bytes` in input and output and in a
    per-thread VTCM region, so the threads never share memory. If slice `t`
    round-trips, thread `t` was able to drive the uDMA engine.
    """
    var thread_id = UInt32(npu_thread_idx())
    var t = Int(thread_id)
    var block_bytes = DMA_ANY_BLOCK_ELEMS * 2

    var vtcm = qshim_get_base_vtcm_addr()
    var stride = vtcm_align(block_bytes, HMX_INPUT_ALIGNMENT)
    var my_vtcm = vtcm + t * stride
    var my_inp = inp_ptr + t * block_bytes
    var my_out = out_ptr + t * block_bytes

    var st1: UInt32 = 0
    var at1 = QShimUdmaDescCommonAttrs(order_and_reserved=0)
    var h1 = qshim_linear_udma_submit(
        thread_id,
        my_inp,
        UInt32(block_bytes),
        my_vtcm,
        UnsafePointer[QShimUdmaDescCommonAttrs, MutUnsafeAnyOrigin](
            unsafe_from_address=Int(UnsafePointer(to=at1))
        ),
        True,
        UnsafePointer[UInt32, MutUnsafeAnyOrigin](
            unsafe_from_address=Int(UnsafePointer(to=st1))
        ),
    )
    _ = qshim_udma_wait(h1)

    var st2: UInt32 = 0
    var at2 = QShimUdmaDescCommonAttrs(order_and_reserved=0)
    var h2 = qshim_linear_udma_submit(
        thread_id,
        my_vtcm,
        UInt32(block_bytes),
        my_out,
        UnsafePointer[QShimUdmaDescCommonAttrs, MutUnsafeAnyOrigin](
            unsafe_from_address=Int(UnsafePointer(to=at2))
        ),
        True,
        UnsafePointer[UInt32, MutUnsafeAnyOrigin](
            unsafe_from_address=Int(UnsafePointer(to=st2))
        ),
    )
    _ = qshim_udma_wait(h2)

    hmx_sync_hvx_and_dma_threads(thread_id)


def test_dma_any_thread() raises:
    print_test_header("Can DMA run on any thread?")
    var ctx = DeviceContext(device_id=0, api="qaic")
    var dev = qaic_get_jit_device(ctx)

    var block = DMA_ANY_BLOCK_ELEMS
    var size = block * DMA_ANY_THREADS
    var data_bytes = size * 2

    var inp_buf = _alloc_ddr(ctx, dev, data_bytes)
    var out_buf = _alloc_ddr(ctx, dev, data_bytes)

    var inp_host = alloc[Float16](size)
    var out_host = alloc[Float16](size)
    var rng = Xorshift32.from_time()
    rng.fill_fp16(inp_host, size)
    for i in range(size):
        out_host[i] = Float16(0)  # sentinel: a thread that never DMAs leaves 0

    qaic_copy_jit_buffer(
        ctx,
        inp_buf,
        inp_host.bitcast[UInt8](),
        UInt64(data_bytes),
        QAIC_COPY_TO_DEVICE,
    )
    qaic_copy_jit_buffer(
        ctx,
        out_buf,
        out_host.bitcast[UInt8](),
        UInt64(data_bytes),
        QAIC_COPY_TO_DEVICE,
    )

    var elf = compile_info_npu[dma_any_thread_kernel]()
    var func = qaic_load_jit_function(
        ctx,
        elf.function_name,
        elf.asm.unsafe_ptr().bitcast[UInt8](),
        UInt64(elf.asm.byte_length()),
    )

    var buffers = alloc[QAicBuf](2)
    buffers[0] = out_buf
    buffers[1] = inp_buf
    qaic_execute_jit_function(
        ctx, func, buffers, 2, num_cores=UInt32(1), thread_mask=UInt32(0x1F)
    )

    qaic_copy_jit_buffer(
        ctx,
        out_buf,
        out_host.bitcast[UInt8](),
        UInt64(data_bytes),
        QAIC_COPY_FROM_DEVICE,
    )

    var thread0_ok = True
    var thread4_ok = True
    var threads_ok = 0
    for t in range(DMA_ANY_THREADS):
        var bad = 0
        for j in range(block):
            if out_host[t * block + j] != inp_host[t * block + j]:
                bad += 1
        var ok = bad == 0
        if ok:
            threads_ok += 1
        if t == 0:
            thread0_ok = ok
        if t == 4:
            thread4_ok = ok
        print(
            "  thread",
            t,
            "DMA slice:",
            "OK" if ok else "FAILED",
            "(",
            bad,
            "mismatched elems )",
        )
    print("  threads that drove uDMA:", threads_ok, "/", DMA_ANY_THREADS)
    if thread0_ok:
        print(
            "  => DMA is NOT restricted to the HMX thread (HVX thread 0 works)"
        )

    inp_host.unsafe_free()
    out_host.unsafe_free()
    buffers.unsafe_free()
    qaic_release_jit_buffer(ctx, inp_buf)
    qaic_release_jit_buffer(ctx, out_buf)
    qaic_release_jit_function(func)

    # The decisive claim: a pure HVX thread (0) and the HMX/DMA thread (4) both
    # complete their transfers.
    assert_true(thread0_ok)
    assert_true(thread4_ok)


# ===-----------------------------------------------------------------------===#
# (5) Can HMX and DMA overlap (DMA on a non-HMX thread)?
# ===-----------------------------------------------------------------------===#
#
# On AIC100 thread 4 is both the HMX thread and the conventional DMA thread, so
# in the standard pipeline HMX and DMA are serialized on one thread. They can
# only overlap if DMA runs on a *different* thread — test (4) shows that is
# allowed, so here thread 4 runs HMX matmuls while thread 0 drives uDMA.

comptime OVL2_DIM: Int = 128  # M = K = N
comptime OVL2_DMA_THREAD: Int = 0
comptime OVL2_MODE_HMX: Int = 0
comptime OVL2_MODE_DMA: Int = 1
comptime OVL2_MODE_BOTH: Int = 2
comptime OVL2_MODE_SETUP: Int = 3


def hmx_dma_overlap_kernel(
    scratch_ddr: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    a_ptr: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    b_ptr: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    mode_ptr: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
    h_ptr: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
    d_ptr: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    """Setup once, then in the timed region thread 4 runs `h` HMX matmuls while
    thread 0 runs `d` uDMA transfers. `mode` selects HMX-only / DMA-only / both /
    setup-only (the last measures the fixed setup cost so it can be subtracted).
    """
    var thread_id = UInt32(npu_thread_idx())
    var is_hmx = npu_is_hmx_thread()
    var mode = Int(mode_ptr[0])
    var h_iters = Int(h_ptr[0])
    var d_iters = Int(d_ptr[0])

    var m = OVL2_DIM
    var k = OVL2_DIM
    var n = OVL2_DIM
    var lhs_bytes = m * k * 2
    var rhs_bytes = k * n * 2
    var block_bytes = OVL2_DIM * OVL2_DIM * 2

    var lhs_flat = hmx_flat_dims_yd(m, k)
    var lhs_crouton = hmx_crouton_dims_from_flat(lhs_flat)
    var lhs_crouton_bytes = hmx_crouton_size_in_bytes(lhs_crouton)
    var out_flat = hmx_flat_dims_yd(m, n)
    var out_crouton = hmx_crouton_dims_from_flat(out_flat)
    var out_crouton_bytes = hmx_crouton_size_in_bytes(out_crouton)
    var k_aligned = (k + 31) & ~31
    var n_aligned = (n + 31) & ~31
    var rhs_weight_bytes = k_aligned * n_aligned * 2

    if is_hmx:
        _ = hmx_matmul_bias_init_16b()
    hmx_sync_hvx_and_hmx_threads(thread_id)

    var vtcm = qshim_get_base_vtcm_addr()
    var off = 0
    off = vtcm_align(off, HMX_INPUT_ALIGNMENT)
    var lhs_vtcm = vtcm + off
    off += lhs_bytes
    off = vtcm_align(off, HMX_INPUT_ALIGNMENT)
    var rhs_vtcm = vtcm + off
    off += rhs_bytes
    off = vtcm_align(off, HMX_ALIGNMENT)
    var lhs_crouton_vtcm = vtcm + off
    off += lhs_crouton_bytes
    off = vtcm_align(off, HMX_WEIGHT_ALIGNMENT)
    var rhs_crouton_vtcm = vtcm + off
    off += rhs_weight_bytes
    off = vtcm_align(off, HMX_ALIGNMENT)
    var out_crouton_vtcm = vtcm + off
    off += out_crouton_bytes
    off = vtcm_align(off, HMX_INPUT_ALIGNMENT)
    var dma_dst_vtcm = vtcm + off

    # ---- Setup: DMA operands in (thread 4), pack to crouton (HVX) ----
    if is_hmx:
        var st1: UInt32 = 0
        var at1 = QShimUdmaDescCommonAttrs(order_and_reserved=0)
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
        _ = qshim_udma_wait(h1)
        var st2: UInt32 = 0
        var at2 = QShimUdmaDescCommonAttrs(order_and_reserved=0)
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
        _ = qshim_udma_wait(h2)
    hmx_sync_hvx_and_dma_threads(thread_id)

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

    # ---- Timed region (no barrier between the HMX and DMA legs) ----
    if is_hmx and (mode == OVL2_MODE_HMX or mode == OVL2_MODE_BOTH):
        for _ in range(h_iters):
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

    if (Int(thread_id) == OVL2_DMA_THREAD) and (
        mode == OVL2_MODE_DMA or mode == OVL2_MODE_BOTH
    ):
        for _ in range(d_iters):
            var st: UInt32 = 0
            var at = QShimUdmaDescCommonAttrs(order_and_reserved=0)
            var h = qshim_linear_udma_submit(
                thread_id,
                scratch_ddr,
                UInt32(block_bytes),
                dma_dst_vtcm,
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


def _time_overlap6(
    ctx: DeviceContext,
    func: QAicJitFunction,
    buffers: UnsafePointer[QAicBuf, MutUnsafeAnyOrigin],
    reps: Int,
) raises -> Int:
    var best = -1
    for _ in range(reps):
        var t0 = Int(perf_counter_ns())
        qaic_execute_jit_function(
            ctx, func, buffers, 6, num_cores=UInt32(1), thread_mask=UInt32(0x1F)
        )
        var dt = Int(perf_counter_ns()) - t0
        if best < 0 or dt < best:
            best = dt
    return best


def test_hmx_dma_overlap() raises:
    print_test_header(
        "Can HMX and DMA overlap? (HMX on thread 4, DMA on thread 0)"
    )
    var ctx = DeviceContext(device_id=0, api="qaic")
    var dev = qaic_get_jit_device(ctx)

    comptime H_ITERS = 400
    comptime D_ITERS = 400
    comptime REPS = 5
    var dim_bytes = OVL2_DIM * OVL2_DIM * 2

    var a_buf = _alloc_ddr(ctx, dev, dim_bytes)
    var b_buf = _alloc_ddr(ctx, dev, dim_bytes)
    var scratch = _alloc_ddr(ctx, dev, dim_bytes)
    var h_buf = _u32_buf(ctx, dev, UInt32(H_ITERS))
    var d_buf = _u32_buf(ctx, dev, UInt32(D_ITERS))

    var ab_host = alloc[Float16](OVL2_DIM * OVL2_DIM)
    var rng = Xorshift32.from_time()
    rng.fill_fp16(ab_host, OVL2_DIM * OVL2_DIM)
    qaic_copy_jit_buffer(
        ctx,
        a_buf,
        ab_host.bitcast[UInt8](),
        UInt64(dim_bytes),
        QAIC_COPY_TO_DEVICE,
    )
    qaic_copy_jit_buffer(
        ctx,
        b_buf,
        ab_host.bitcast[UInt8](),
        UInt64(dim_bytes),
        QAIC_COPY_TO_DEVICE,
    )

    var elf = compile_info_npu[hmx_dma_overlap_kernel]()
    var func = qaic_load_jit_function(
        ctx,
        elf.function_name,
        elf.asm.unsafe_ptr().bitcast[UInt8](),
        UInt64(elf.asm.byte_length()),
    )

    var buffers = alloc[QAicBuf](6)
    buffers[0] = scratch
    buffers[1] = a_buf
    buffers[2] = b_buf
    buffers[4] = h_buf
    buffers[5] = d_buf

    var mode_setup = _u32_buf(ctx, dev, UInt32(OVL2_MODE_SETUP))
    buffers[3] = mode_setup
    var t_setup = _time_overlap6(ctx, func, buffers, REPS)
    qaic_release_jit_buffer(ctx, mode_setup)

    var mode_hmx = _u32_buf(ctx, dev, UInt32(OVL2_MODE_HMX))
    buffers[3] = mode_hmx
    var t_hmx = _time_overlap6(ctx, func, buffers, REPS)
    qaic_release_jit_buffer(ctx, mode_hmx)

    var mode_dma = _u32_buf(ctx, dev, UInt32(OVL2_MODE_DMA))
    buffers[3] = mode_dma
    var t_dma = _time_overlap6(ctx, func, buffers, REPS)
    qaic_release_jit_buffer(ctx, mode_dma)

    var mode_both = _u32_buf(ctx, dev, UInt32(OVL2_MODE_BOTH))
    buffers[3] = mode_both
    var t_both = _time_overlap6(ctx, func, buffers, REPS)
    qaic_release_jit_buffer(ctx, mode_both)

    # Subtract the fixed setup (DMA-in + pack) so only the timed loops remain.
    var hmx_work = t_hmx - t_setup
    var dma_work = t_dma - t_setup
    var both_work = t_both - t_setup
    var serial = hmx_work + dma_work
    var slowest = hmx_work if hmx_work > dma_work else dma_work

    print("  setup-only          :", t_setup, "ns")
    print("  HMX-only  (", H_ITERS, "matmuls, thread 4 ):", hmx_work, "ns")
    print("  DMA-only  (", D_ITERS, "xfers, thread 0 ):", dma_work, "ns")
    print("  both concurrent     :", both_work, "ns")
    print("  serial sum (no overlap) :", serial, "ns")
    print("  slowest leg (full overlap):", slowest, "ns")
    if both_work > 0 and serial > 0:
        var hidden = serial - both_work
        print(
            "  overlap: both is",
            serial * 100 // both_work,
            "% of serial-sum;",
            hidden * 100 // serial,
            "% of work hidden",
        )

    ab_host.unsafe_free()
    buffers.unsafe_free()
    qaic_release_jit_buffer(ctx, a_buf)
    qaic_release_jit_buffer(ctx, b_buf)
    qaic_release_jit_buffer(ctx, scratch)
    qaic_release_jit_buffer(ctx, h_buf)
    qaic_release_jit_buffer(ctx, d_buf)
    qaic_release_jit_function(func)

    print_separator()
    print("TEST PASSED!")
    print_separator()

    # Overlapping must not be slower than running the legs serially (+15% noise).
    assert_true(both_work <= (serial * 115) // 100)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
