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
"""Confirm that VTCM contents persist across separate kernel launches.

A `QAIC_BUFFER_TYPE_DEV_TCM` buffer lives in the NSP's on-chip VTCM. When its
command-buffer descriptor uses `QAIC_DEV_BUF_RESIDENT_KEEP`, the VTCM mapping is
held until the buffer is freed rather than torn down per command buffer, so the
contents written by one launch are still there for the next launch — no
re-initialization (no zeroing) needed in between. `test_qaic_vtcm_persistence_perf.mojo`
measures the wall-clock that buys back.

Two allocation constraints, both enforced by the JIT runtime (`allocTCMDevBuffer`)
and verified here by construction:
  - the allocation must be the full VTCM size (8 MiB on AI100), and
  - the compute-unit bitmap must select exactly one NSP (VTCM is per-NSP).

To run (requires QAIC hardware):
    mojo test_qaic_vtcm_persistence.mojo
"""

from common import print_test_header

from std.memory import UnsafePointer, alloc
from std.testing import TestSuite, assert_equal

from max.gpu.host import DeviceContext
from std._plugin.qaic.host.target import compile_info_npu, npu_thread_idx
from max._plugin.qaic.host.qaic import (
    QAicBuf,
    QAicBufferDescriptor,
    QAicDevice,
    QAicJitFunction,
    QAIC_BUFFER_TYPE_DEV_DDR,
    QAIC_BUFFER_TYPE_DEV_TCM,
    QAIC_COPY_FROM_DEVICE,
    QAIC_DEV_BUF_ADDR_MODE_HSV32,
    QAIC_DEV_BUF_AUTO_MAP,
    QAIC_DEV_BUF_IO_RW,
    QAIC_DEV_BUF_RESIDENT_KEEP,
    qaic_allocate_jit_buffer,
    qaic_copy_jit_buffer,
    qaic_execute_jit_function,
    qaic_get_jit_device,
    qaic_load_jit_function,
    qaic_release_jit_buffer,
    qaic_release_jit_function,
)


comptime N: Int = 1024  # uint32 elements touched by the kernels (4 KiB)
comptime NBYTES: Int = N * 4
comptime VTCM_BYTES: Int = 8 * 1024 * 1024  # full VTCM; required for a TCM alloc
comptime ALIGN: UInt32 = 128
comptime NSP0_CU: UInt64 = 0x1  # exactly one NSP — required for a TCM alloc
comptime ONE_THREAD: UInt32 = 0x01
comptime PAT: UInt32 = 0xC0DE0000


# ===-----------------------------------------------------------------------===#
# Kernels (single thread on one NSP; arg 0 is the VTCM buffer)
# ===-----------------------------------------------------------------------===#


def write_ramp(tcm: UnsafePointer[UInt32, MutUnsafeAnyOrigin]):
    """Write a known ramp into VTCM."""
    if npu_thread_idx() == 0:
        for i in range(N):
            tcm[i] = PAT + UInt32(i)


def copy_out(
    tcm: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
    dst: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    """Copy VTCM (arg 0) into a DDR buffer (arg 1). Does not touch VTCM first.
    """
    if npu_thread_idx() == 0:
        for i in range(N):
            dst[i] = tcm[i]


def init_index(tcm: UnsafePointer[UInt32, MutUnsafeAnyOrigin]):
    """Seed VTCM[i] = i."""
    if npu_thread_idx() == 0:
        for i in range(N):
            tcm[i] = UInt32(i)


def increment(tcm: UnsafePointer[UInt32, MutUnsafeAnyOrigin]):
    """Read-modify-write VTCM in place: VTCM[i] += 1. Relies on the prior
    launch's contents still being resident."""
    if npu_thread_idx() == 0:
        for i in range(N):
            tcm[i] = tcm[i] + 1


# ===-----------------------------------------------------------------------===#
# Host helpers
# ===-----------------------------------------------------------------------===#


def load_write_ramp(ctx: DeviceContext) raises -> QAicJitFunction:
    var elf = compile_info_npu[write_ramp]()
    return qaic_load_jit_function(
        ctx,
        elf.function_name,
        elf.asm.unsafe_ptr().bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(elf.asm.byte_length()),
    )


def load_copy_out(ctx: DeviceContext) raises -> QAicJitFunction:
    var elf = compile_info_npu[copy_out]()
    return qaic_load_jit_function(
        ctx,
        elf.function_name,
        elf.asm.unsafe_ptr().bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(elf.asm.byte_length()),
    )


def load_init_index(ctx: DeviceContext) raises -> QAicJitFunction:
    var elf = compile_info_npu[init_index]()
    return qaic_load_jit_function(
        ctx,
        elf.function_name,
        elf.asm.unsafe_ptr().bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(elf.asm.byte_length()),
    )


def load_increment(ctx: DeviceContext) raises -> QAicJitFunction:
    var elf = compile_info_npu[increment]()
    return qaic_load_jit_function(
        ctx,
        elf.function_name,
        elf.asm.unsafe_ptr().bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(elf.asm.byte_length()),
    )


def alloc_tcm(ctx: DeviceContext, dev: QAicDevice) raises -> QAicBuf:
    """Allocate a full-VTCM, single-NSP DEV_TCM buffer."""
    return qaic_allocate_jit_buffer(
        ctx,
        dev,
        QAIC_BUFFER_TYPE_DEV_TCM,
        QAIC_DEV_BUF_IO_RW,
        UInt64(VTCM_BYTES),
        ALIGN,
        NSP0_CU,
    )


def alloc_ddr(
    ctx: DeviceContext, dev: QAicDevice, nbytes: Int
) raises -> QAicBuf:
    return qaic_allocate_jit_buffer(
        ctx,
        dev,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        UInt64(nbytes),
        ALIGN,
        0xFFFFFFFFFFFFFFFF,
    )


def keep_desc(buf: QAicBuf, nbytes: Int) -> QAicBufferDescriptor:
    """A RESIDENT_KEEP descriptor — the mapping survives until the buffer is
    freed, which is what makes the VTCM contents persist across launches."""
    return QAicBufferDescriptor(
        buf,
        0,
        UInt64(nbytes),
        QAIC_DEV_BUF_RESIDENT_KEEP,
        QAIC_DEV_BUF_AUTO_MAP,
        QAIC_DEV_BUF_ADDR_MODE_HSV32,
    )


def launch1(
    ctx: DeviceContext, func: QAicJitFunction, b0: QAicBuf, n0: Int
) raises:
    var d = alloc[QAicBufferDescriptor](1)
    d[0] = keep_desc(b0, n0)
    qaic_execute_jit_function(
        ctx, func, d.as_unsafe_any_origin(), 1, 1, ONE_THREAD
    )
    d.unsafe_free()


def launch2(
    ctx: DeviceContext,
    func: QAicJitFunction,
    b0: QAicBuf,
    n0: Int,
    b1: QAicBuf,
    n1: Int,
) raises:
    var d = alloc[QAicBufferDescriptor](2)
    d[0] = keep_desc(b0, n0)
    d[1] = keep_desc(b1, n1)
    qaic_execute_jit_function(
        ctx, func, d.as_unsafe_any_origin(), 2, 1, ONE_THREAD
    )
    d.unsafe_free()


def download(
    ctx: DeviceContext, buf: QAicBuf, n: Int
) raises -> UnsafePointer[UInt32, MutUntrackedOrigin]:
    var host = alloc[UInt32](n)
    qaic_copy_jit_buffer(
        ctx,
        buf,
        host.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(n * 4),
        QAIC_COPY_FROM_DEVICE,
        0,
    )
    return host


# ===-----------------------------------------------------------------------===#
# Tests
# ===-----------------------------------------------------------------------===#


def test_dev_tcm_persists_across_launches() raises:
    print_test_header("QAIC DEV_TCM + RESIDENT_KEEP persists across launches")

    var ctx = DeviceContext(device_id=0, api="qaic")
    var dev = qaic_get_jit_device(ctx)
    var tcm = alloc_tcm(ctx, dev)
    var dst = alloc_ddr(ctx, dev, NBYTES)

    var w = load_write_ramp(ctx)
    var c = load_copy_out(ctx)

    # Launch 1 writes the ramp into VTCM; launch 2 copies VTCM out without
    # re-writing it. If the mapping were not kept resident, launch 2 would see
    # cleared / re-mapped scratch instead of launch 1's ramp.
    launch1(ctx, w, tcm, VTCM_BYTES)
    launch2(ctx, c, tcm, VTCM_BYTES, dst, NBYTES)

    var host = download(ctx, dst, N)
    var errors = 0
    for i in range(N):
        if host[i] != PAT + UInt32(i):
            if errors < 5:
                print("  mismatch at", i, ": got", hex(host[i]))
            errors += 1
    host.unsafe_free()

    qaic_release_jit_buffer(ctx, tcm)
    qaic_release_jit_buffer(ctx, dst)
    qaic_release_jit_function(w)
    qaic_release_jit_function(c)

    print("  ", "PASS" if errors == 0 else String("FAIL: ") + String(errors))
    assert_equal(errors, 0, "VTCM contents did not survive the second launch")


def test_dev_tcm_accumulates_across_launches() raises:
    print_test_header("QAIC DEV_TCM read-modify-write carries across launches")

    comptime ROUNDS: Int = 8

    var ctx = DeviceContext(device_id=0, api="qaic")
    var dev = qaic_get_jit_device(ctx)
    var tcm = alloc_tcm(ctx, dev)
    var dst = alloc_ddr(ctx, dev, NBYTES)

    var seed = load_init_index(ctx)
    var inc = load_increment(ctx)
    var c = load_copy_out(ctx)

    # Seed once, then increment in ROUNDS separate launches. Each launch reads
    # what the previous one left in VTCM — no re-seed in between.
    launch1(ctx, seed, tcm, VTCM_BYTES)
    for _ in range(ROUNDS):
        launch1(ctx, inc, tcm, VTCM_BYTES)
    launch2(ctx, c, tcm, VTCM_BYTES, dst, NBYTES)

    var host = download(ctx, dst, N)
    var errors = 0
    for i in range(N):
        if host[i] != UInt32(i) + UInt32(ROUNDS):
            if errors < 5:
                print(
                    "  mismatch at",
                    i,
                    ": expected",
                    i + ROUNDS,
                    "got",
                    host[i],
                )
            errors += 1
    host.unsafe_free()

    qaic_release_jit_buffer(ctx, tcm)
    qaic_release_jit_buffer(ctx, dst)
    qaic_release_jit_function(seed)
    qaic_release_jit_function(inc)
    qaic_release_jit_function(c)

    print("  ", "PASS" if errors == 0 else String("FAIL: ") + String(errors))
    assert_equal(errors, 0, "VTCM accumulation across launches lost values")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
