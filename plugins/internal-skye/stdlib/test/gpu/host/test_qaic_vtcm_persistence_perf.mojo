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
"""Measure the time saved by *not* re-zeroing VTCM between kernel launches.

`test_qaic_vtcm_persistence.mojo` proves a DEV_TCM + RESIDENT_KEEP buffer keeps
its contents across launches. The payoff: scratch initialized once stays
initialized, so a kernel can skip the per-launch zeroing. This benchmark
quantifies that.

Two kernels share the same VTCM buffer and do the same useful work
(`tcm[0] += 1`); they differ only in whether they first zero the full VTCM:

  - `zero_then_bump` — memset the whole 8 MiB VTCM, then bump. This is the cost
    a kernel pays every launch if it cannot trust the scratch to persist.
  - `bump_only`      — bump only, trusting the persisted scratch.

We time each kernel best-of-N (the min cancels the fixed per-dispatch overhead
that both share), so `min(zero_then_bump) - min(bump_only)` is the zeroing cost
that persistence buys back per launch. `bump_only`'s accumulation across N
launches (`tcm[0]` ends at `N + ...`) doubles as proof the persisted scratch is
real and the work was not skipped.

Timing-only / print-heavy — run by hand, not under `//...`:
    mojo test_qaic_vtcm_persistence_perf.mojo
"""

from common import print_test_header

from std.memory import UnsafePointer, alloc
from std.testing import TestSuite, assert_equal, assert_true
from std.time import perf_counter_ns

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


comptime VTCM_BYTES: Int = 8 * 1024 * 1024  # full VTCM; required for a TCM alloc
comptime ZERO_ELEMS: Int = VTCM_BYTES // 4  # uint32s spanning the whole VTCM
comptime ZERO_W: Int = 32  # uint32s per HVX vector store (128 B)
comptime ALIGN: UInt32 = 128
comptime NSP0_CU: UInt64 = 0x1  # exactly one NSP — required for a TCM alloc
comptime ONE_THREAD: UInt32 = 0x01

comptime REPS: Int = 50  # timed launches per kernel
comptime PROJECT_LAUNCHES: Int = 100  # horizon for the projected-savings figure


# ===-----------------------------------------------------------------------===#
# Kernels (single thread on one NSP; arg 0 is the VTCM buffer)
# ===-----------------------------------------------------------------------===#


def zero_then_bump(tcm: UnsafePointer[UInt32, MutUnsafeAnyOrigin]):
    """Zero the full VTCM (vectorized), then do the useful work."""
    if npu_thread_idx() == 0:
        var z = SIMD[DType.uint32, ZERO_W](0)
        for i in range(0, ZERO_ELEMS, ZERO_W):
            tcm.store(i, z)
        tcm[0] = tcm[0] + 1


def bump_only(tcm: UnsafePointer[UInt32, MutUnsafeAnyOrigin]):
    """Do the useful work only, trusting the persisted scratch."""
    if npu_thread_idx() == 0:
        tcm[0] = tcm[0] + 1


def copy_first(
    tcm: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
    dst: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    if npu_thread_idx() == 0:
        dst[0] = tcm[0]


# ===-----------------------------------------------------------------------===#
# Host helpers
# ===-----------------------------------------------------------------------===#


def load_zero_then_bump(ctx: DeviceContext) raises -> QAicJitFunction:
    var elf = compile_info_npu[zero_then_bump]()
    return qaic_load_jit_function(
        ctx,
        elf.function_name,
        elf.asm.unsafe_ptr().bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(elf.asm.byte_length()),
    )


def load_bump_only(ctx: DeviceContext) raises -> QAicJitFunction:
    var elf = compile_info_npu[bump_only]()
    return qaic_load_jit_function(
        ctx,
        elf.function_name,
        elf.asm.unsafe_ptr().bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(elf.asm.byte_length()),
    )


def load_copy_first(ctx: DeviceContext) raises -> QAicJitFunction:
    var elf = compile_info_npu[copy_first]()
    return qaic_load_jit_function(
        ctx,
        elf.function_name,
        elf.asm.unsafe_ptr().bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(elf.asm.byte_length()),
    )


def alloc_tcm(ctx: DeviceContext, dev: QAicDevice) raises -> QAicBuf:
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
    return QAicBufferDescriptor(
        buf,
        0,
        UInt64(nbytes),
        QAIC_DEV_BUF_RESIDENT_KEEP,
        QAIC_DEV_BUF_AUTO_MAP,
        QAIC_DEV_BUF_ADDR_MODE_HSV32,
    )


def time_kernel(
    ctx: DeviceContext,
    func: QAicJitFunction,
    descs: UnsafePointer[QAicBufferDescriptor, MutUnsafeAnyOrigin],
    reps: Int,
) raises -> Tuple[Int, Int]:
    """Best-of-`reps` and total launch time (ns) for a 1-buffer kernel.

    One warmup launch first absorbs JIT/first-dispatch overhead. Returns
    (best_ns, total_ns).
    """
    qaic_execute_jit_function(ctx, func, descs, 1, 1, ONE_THREAD)
    var best = -1
    var total = 0
    for _ in range(reps):
        var t0 = Int(perf_counter_ns())
        qaic_execute_jit_function(ctx, func, descs, 1, 1, ONE_THREAD)
        var dt = Int(perf_counter_ns()) - t0
        total += dt
        if best < 0 or dt < best:
            best = dt
    return Tuple(best, total)


def ms(ns: Int) -> Float64:
    return Float64(ns) / 1_000_000.0


# ===-----------------------------------------------------------------------===#
# Benchmark
# ===-----------------------------------------------------------------------===#


def test_vtcm_zeroing_cost_saved_by_persistence() raises:
    print_test_header("VTCM persistence: time saved by skipping re-zeroing")

    var ctx = DeviceContext(device_id=0, api="qaic")
    var dev = qaic_get_jit_device(ctx)
    var tcm = alloc_tcm(ctx, dev)
    var dst = alloc_ddr(ctx, dev, 4)

    var zero_bump = load_zero_then_bump(ctx)
    var bump = load_bump_only(ctx)
    var cp = load_copy_first(ctx)

    var descs = alloc[QAicBufferDescriptor](1)
    descs[0] = keep_desc(tcm, VTCM_BYTES)
    var descs_p = descs.as_unsafe_any_origin()

    # Cost a kernel pays if it re-zeros VTCM every launch.
    var best_zero: Int
    var total_zero: Int
    best_zero, total_zero = time_kernel(ctx, zero_bump, descs_p, REPS)

    # bump_only relies on the scratch persisting. Seed tcm[0] with a single
    # zeroing launch so the post-run value is exactly the bump count, then time
    # REPS bump-only launches.
    qaic_execute_jit_function(ctx, zero_bump, descs_p, 1, 1, ONE_THREAD)
    var best_bump: Int
    var total_bump: Int
    best_bump, total_bump = time_kernel(ctx, bump, descs_p, REPS)
    descs.unsafe_free()

    # Confirm the work was real and the scratch persisted: tcm[0] should be
    # 1 (seed) + 1 (time_kernel warmup) + REPS (timed bumps).
    var rd = alloc[QAicBufferDescriptor](2)
    rd[0] = keep_desc(tcm, VTCM_BYTES)
    rd[1] = keep_desc(dst, 4)
    qaic_execute_jit_function(
        ctx, cp, rd.as_unsafe_any_origin(), 2, 1, ONE_THREAD
    )
    rd.unsafe_free()
    var host = alloc[UInt32](1)
    qaic_copy_jit_buffer(
        ctx,
        dst,
        host.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(4),
        QAIC_COPY_FROM_DEVICE,
        0,
    )
    var bumped = Int(host[0])
    host.unsafe_free()

    var mean_zero = total_zero // REPS
    var mean_bump = total_bump // REPS
    var save_best = best_zero - best_bump
    var save_mean = mean_zero - mean_bump
    var projected = save_best * (PROJECT_LAUNCHES - 1)

    print(
        "  zeroing footprint per launch :", VTCM_BYTES // (1024 * 1024), "MiB"
    )
    print("  timed launches per kernel    :", REPS)
    print(
        "  zero_then_bump  best/mean    :",
        ms(best_zero),
        "/",
        ms(mean_zero),
        "ms",
    )
    print(
        "  bump_only       best/mean    :",
        ms(best_bump),
        "/",
        ms(mean_bump),
        "ms",
    )
    print(
        "  saved per launch (best/mean) :",
        ms(save_best),
        "/",
        ms(save_mean),
        "ms",
    )
    print(
        "  projected save over",
        PROJECT_LAUNCHES,
        "launches  :",
        ms(projected),
        "ms",
    )
    print(
        "  bump_only accumulated tcm[0] :",
        bumped,
        "(expected",
        1 + 1 + REPS,
        ")",
    )

    qaic_release_jit_buffer(ctx, tcm)
    qaic_release_jit_buffer(ctx, dst)
    qaic_release_jit_function(zero_bump)
    qaic_release_jit_function(bump)
    qaic_release_jit_function(cp)

    # Correctness: the persisted scratch carried the running count across every
    # launch — the bump-only path did real work, it didn't just skip zeroing.
    assert_equal(
        bumped,
        1 + 1 + REPS,
        "persisted VTCM lost the running count across launches",
    )
    # Skipping the zeroing is never slower than doing it (best-of-N cancels the
    # shared per-dispatch overhead, so this isolates the zeroing cost).
    assert_true(
        best_bump <= best_zero,
        "bump_only should not be slower than zero_then_bump",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
