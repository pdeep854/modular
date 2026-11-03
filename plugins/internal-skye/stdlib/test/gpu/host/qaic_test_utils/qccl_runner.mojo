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
"""Shared 2-rank pthread harness for QCCL Mojo tests.

`run_2_ranks` factors the rank-worker scaffolding (pthread spawn/join,
serialized JIT device creation, host barrier, comm bring-up, JIT semaphore,
buffer alloc/copy + readback) out of individual tests so that each test only
writes the host-side init pattern, the QCCL call, and the verifier.
"""

from std.atomic import Atomic
from std.ffi import c_int, external_call
from std.memory import UnsafePointer, alloc

from max.gpu.host import DeviceContext
from max._plugin.qaic.host.qaic import (
    QAicBuf,
    QAicDevice,
    QAicStream,
    qaic_allocate_jit_buffer,
    qaic_copy_jit_buffer,
    qaic_create_jit_stream,
    qaic_get_jit_device,
    qaic_release_jit_buffer,
    qaic_release_jit_stream,
    QAIC_BUFFER_TYPE_DEV_DDR,
    QAIC_COPY_FROM_DEVICE,
    QAIC_COPY_TO_DEVICE,
    QAIC_DEV_BUF_IO_RW,
)
from max._plugin.qaic.host.qccl import (
    QAicQcclComm,
    QAicQcclUniqueId,
    QAicSemaphore,
    QAicSemaphoreSignal,
    QAicSemaphoreSignalList,
    QAIC_SEM_CHK_GE,
    QAIC_SEM_OP_INC,
    qaic_jit_semaphore_create,
    qaic_jit_semaphore_free,
    qaic_jit_semaphore_wait,
    qccl_comm_destroy,
    qccl_comm_init_rank,
    qccl_get_unique_id,
)


# ===-----------------------------------------------------------------------===#
# Constants
# ===-----------------------------------------------------------------------===#

comptime ALIGNMENT: UInt32 = 128
comptime CU_BITMAP: UInt64 = 0xFFFFFFFFFFFFFFFF

# Lower 31 bits = arrival count; high bit = abort signalled by some rank.
comptime BARRIER_ABORT: UInt32 = 0x80000000


# ===-----------------------------------------------------------------------===#
# Op-fn argument bundle
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct RankOpArgs(Copyable, Movable):
    """Arguments passed to the per-rank op closure invoked by `run_2_ranks`.

    The runner has already initialized the device, stream, communicator,
    semaphore + signal_list, and copied this rank's host src into `src_buf`
    by the time the op is called. The op only needs to issue the QCCL call
    using `signal_list_ptr`. The runner waits on `done_sem` after the op
    returns and copies `dst_buf` back into `dst_host` for the verifier.
    """

    var ctx: DeviceContext
    var jit_device: QAicDevice
    var stream: QAicStream
    var comm: QAicQcclComm
    var src_buf: QAicBuf
    var dst_buf: QAicBuf

    @__allow_legacy_any_origin_fields
    var signal_list_ptr: UnsafePointer[
        QAicSemaphoreSignalList, ImmUnsafeAnyOrigin
    ]
    var done_sem: QAicSemaphore
    var rank: UInt32
    var n_ranks: UInt32


# ===-----------------------------------------------------------------------===#
# Internal: barrier + rank context
# ===-----------------------------------------------------------------------===#


def _barrier_arrive_and_wait(
    barrier: UnsafePointer[Atomic[UInt32], MutUnsafeAnyOrigin],
    n_ranks: UInt32,
) raises:
    """Increment barrier, spin until all ranks arrive (or someone aborts)."""
    _ = barrier[].fetch_add(UInt32(1))
    while True:
        var v = barrier[].load()
        if (v & BARRIER_ABORT) != 0:
            raise Error("barrier aborted by peer rank")
        if (v & ~BARRIER_ABORT) >= n_ranks:
            return
        _ = external_call["sched_yield", c_int]()


def _barrier_abort(
    barrier: UnsafePointer[Atomic[UInt32], MutUnsafeAnyOrigin],
):
    """Mark the barrier as aborted so peers exit their wait."""
    _ = barrier[].fetch_add(BARRIER_ABORT)


@fieldwise_init
struct _RankCtx:
    """Per-rank state shared between the main thread and the worker thread."""

    var rank: UInt32
    var n_ranks: UInt32
    var src_bytes: UInt64
    var dst_bytes: UInt64
    var unique_id: QAicQcclUniqueId
    var ok: Bool
    var errors: Int

    @__allow_legacy_any_origin_fields
    var barrier: UnsafePointer[Atomic[UInt32], MutUnsafeAnyOrigin]

    @__allow_legacy_any_origin_fields
    var dev_token: UnsafePointer[Atomic[UInt32], MutUnsafeAnyOrigin]


def _round_up(x: UInt64, a: UInt64) -> UInt64:
    return (x + a - 1) & ~(a - 1)


# Per-instantiation thread entry. The user's init_host / op_fn / verify
# callbacks are comptime-substituted, so each instantiation is a concrete
# function pointer that pthread_create can accept.
def _rank_main[
    init_host: def(
        rank: UInt32,
        src_host: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
        src_bytes: UInt64,
    ) raises thin -> None,
    op_fn: def(args: RankOpArgs) raises thin -> None,
    verify: def(
        rank: UInt32,
        dst_host: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
        dst_bytes: UInt64,
    ) raises thin -> Int,
](rank_ctx_ptr: UnsafePointer[_RankCtx, MutUnsafeAnyOrigin]) raises:
    var rank = rank_ctx_ptr[].rank
    var n_ranks = rank_ctx_ptr[].n_ranks
    var src_bytes = rank_ctx_ptr[].src_bytes
    var dst_bytes = rank_ctx_ptr[].dst_bytes
    var unique_id_local = rank_ctx_ptr[].unique_id.copy()

    # Serialize device creation across ranks.
    while True:
        var token = rank_ctx_ptr[].dev_token[].load()
        if (token & BARRIER_ABORT) != 0:
            raise Error("dev_token aborted by peer rank")
        if token == rank:
            break
        _ = external_call["sched_yield", c_int]()

    print("[rank", rank, "] creating DeviceContext")
    var ctx = DeviceContext(device_id=Int(rank), api="qaic")
    var jit_device = qaic_get_jit_device(ctx)
    print("[rank", rank, "] JIT device ready, releasing creation token")
    _ = rank_ctx_ptr[].dev_token[].fetch_add(UInt32(1))

    var stream = qaic_create_jit_stream(ctx, jit_device)

    var src_alloc_bytes = _round_up(src_bytes, UInt64(ALIGNMENT))
    var dst_alloc_bytes = _round_up(dst_bytes, UInt64(ALIGNMENT))

    var src_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        src_alloc_bytes,
        ALIGNMENT,
        CU_BITMAP,
    )
    var dst_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        dst_alloc_bytes,
        ALIGNMENT,
        CU_BITMAP,
    )

    var src_host = alloc[UInt8](Int(src_bytes))
    var dst_host = alloc[UInt8](Int(dst_bytes))
    for i in range(Int(src_bytes)):
        src_host[i] = UInt8(0)
    for i in range(Int(dst_bytes)):
        dst_host[i] = UInt8(0)

    init_host(rank, src_host.as_unsafe_any_origin(), src_bytes)

    qaic_copy_jit_buffer(
        ctx,
        src_buf,
        src_host.as_unsafe_any_origin(),
        src_bytes,
        QAIC_COPY_TO_DEVICE,
    )
    qaic_copy_jit_buffer(
        ctx,
        dst_buf,
        dst_host.as_unsafe_any_origin(),
        dst_bytes,
        QAIC_COPY_TO_DEVICE,
    )

    var done_sem = qaic_jit_semaphore_create(ctx, 0)
    var signal = QAicSemaphoreSignal(done_sem, QAIC_SEM_OP_INC, 1)
    var signal_list = QAicSemaphoreSignalList(
        1, UnsafePointer(to=signal).as_unsafe_any_origin()
    )

    print("[rank", rank, "] reached pre-comm barrier")
    _barrier_arrive_and_wait(rank_ctx_ptr[].barrier, n_ranks)

    print("[rank", rank, "] qccl_comm_init_rank")
    var comm = qccl_comm_init_rank(
        ctx, jit_device, unique_id_local, rank, n_ranks
    )
    print("[rank", rank, "] comm initialized")

    var args = RankOpArgs(
        ctx,
        jit_device,
        stream,
        comm,
        src_buf,
        dst_buf,
        UnsafePointer(to=signal_list).as_unsafe_any_origin(),
        done_sem,
        rank,
        n_ranks,
    )
    op_fn(args)

    print("[rank", rank, "] waiting on done semaphore")
    qaic_jit_semaphore_wait(ctx, done_sem, QAIC_SEM_CHK_GE, 1)
    print("[rank", rank, "] op complete")

    qaic_copy_jit_buffer(
        ctx,
        dst_buf,
        dst_host.as_unsafe_any_origin(),
        dst_bytes,
        QAIC_COPY_FROM_DEVICE,
    )

    var errors = verify(rank, dst_host.as_unsafe_any_origin(), dst_bytes)
    rank_ctx_ptr[].errors = errors
    rank_ctx_ptr[].ok = errors == 0
    print("[rank", rank, "] verified: errors =", errors)

    src_host.unsafe_free()
    dst_host.unsafe_free()
    qaic_jit_semaphore_free(ctx, done_sem)
    qccl_comm_destroy(ctx, comm)
    qaic_release_jit_buffer(ctx, src_buf)
    qaic_release_jit_buffer(ctx, dst_buf)
    qaic_release_jit_stream(ctx, stream)


def _rank_thread_entry[
    init_host: def(
        rank: UInt32,
        src_host: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
        src_bytes: UInt64,
    ) raises thin -> None,
    op_fn: def(args: RankOpArgs) raises thin -> None,
    verify: def(
        rank: UInt32,
        dst_host: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
        dst_bytes: UInt64,
    ) raises thin -> Int,
](
    arg: UnsafePointer[NoneType, MutUnsafeAnyOrigin],
) -> UnsafePointer[
    NoneType, MutUnsafeAnyOrigin
]:
    """Pthread start routine. Each parameter set yields a concrete fn pointer.
    """
    var ctx_ptr = arg.bitcast[_RankCtx]()
    var rank = ctx_ptr[].rank
    try:
        _rank_main[init_host, op_fn, verify](ctx_ptr)
    except e:
        print("[rank", rank, "] error:", e)
        ctx_ptr[].ok = False
        # Release any peer ranks waiting on barrier or dev_token so the test
        # fails fast instead of hanging.
        _barrier_abort(ctx_ptr[].barrier)
        _barrier_abort(ctx_ptr[].dev_token)
    return UnsafePointer[NoneType, MutUnsafeAnyOrigin].unsafe_dangling()


def run_2_ranks[
    init_host: def(
        rank: UInt32,
        src_host: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
        src_bytes: UInt64,
    ) raises thin -> None,
    op_fn: def(args: RankOpArgs) raises thin -> None,
    verify: def(
        rank: UInt32,
        dst_host: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
        dst_bytes: UInt64,
    ) raises thin -> Int,
](description: String, src_bytes: UInt64, dst_bytes: UInt64,) raises -> Bool:
    """Run a 2-rank QCCL test using the standard pthread harness.

    Skips (returns True) if a second QAIC device isn't visible.

    Each rank thread creates its DeviceContext serially (the QAIC SDK's
    device-registration table races under concurrent
    `qaicJitDeviceCreate`), allocates `src_buf` (`src_bytes`) and `dst_buf`
    (`dst_bytes`) plus matching host shadows, calls `init_host` to populate
    the per-rank source pattern, copies host -> device, brings up the
    communicator at the host barrier, and then invokes `op_fn` to issue the
    QCCL call. After `op_fn` returns, the runner waits on the JIT semaphore,
    copies `dst_buf -> dst_host`, and calls `verify` to validate the result.
    The rank is OK iff `verify` returns zero.

    Parameters:
        init_host: Callback that fills the per-rank src host buffer.
        op_fn: Callback that issues the QCCL op via `args.signal_list_ptr`.
        verify: Callback that validates the dst host buffer; returns errors.

    Args:
        description: Human-readable test name (printed in the header).
        src_bytes: Size in bytes of the per-rank source buffer.
        dst_bytes: Size in bytes of the per-rank destination buffer.

    Returns:
        True on success or skip; False on any rank failure.
    """
    print("Running 2-rank QCCL test:", description)

    # Probe device 1's existence — if missing, skip.
    try:
        var probe = DeviceContext(device_id=1, api="qaic")
        _ = probe^
    except e:
        print("  SKIP — need 2 visible QAIC devices:", e)
        return True

    var unique_id = qccl_get_unique_id()

    var barrier = Atomic[UInt32](0)
    var dev_token = Atomic[UInt32](0)
    var barrier_ptr = UnsafePointer(to=barrier)
    var dev_token_ptr = UnsafePointer(to=dev_token)

    var rank0 = _RankCtx(
        0,
        UInt32(2),
        src_bytes,
        dst_bytes,
        unique_id.copy(),
        False,
        -1,
        barrier_ptr.as_unsafe_any_origin(),
        dev_token_ptr.as_unsafe_any_origin(),
    )
    var rank1 = _RankCtx(
        1,
        UInt32(2),
        src_bytes,
        dst_bytes,
        unique_id.copy(),
        False,
        -1,
        barrier_ptr.as_unsafe_any_origin(),
        dev_token_ptr.as_unsafe_any_origin(),
    )

    var tid0: UInt64 = 0
    var tid1: UInt64 = 0
    var nullattr = Optional[UnsafePointer[NoneType, MutUnsafeAnyOrigin]]()

    var rc0 = external_call["pthread_create", c_int](
        UnsafePointer(to=tid0),
        nullattr,
        _rank_thread_entry[init_host, op_fn, verify],
        UnsafePointer(to=rank0).bitcast[NoneType](),
    )
    var rc1 = external_call["pthread_create", c_int](
        UnsafePointer(to=tid1),
        nullattr,
        _rank_thread_entry[init_host, op_fn, verify],
        UnsafePointer(to=rank1).bitcast[NoneType](),
    )
    if rc0 != 0 or rc1 != 0:
        print("  FAIL — pthread_create failed:", rc0, rc1)
        return False

    var retval = UnsafePointer[NoneType, MutUnsafeAnyOrigin].unsafe_dangling()
    var jrc0 = external_call["pthread_join", c_int](
        tid0, UnsafePointer(to=retval)
    )
    var jrc1 = external_call["pthread_join", c_int](
        tid1, UnsafePointer(to=retval)
    )
    if jrc0 != 0 or jrc1 != 0:
        print("  FAIL — pthread_join failed:", jrc0, jrc1)
        return False

    if rank0.ok and rank1.ok:
        print("  PASS")
        return True
    print(
        "  FAIL — rank0.errors =",
        rank0.errors,
        "rank1.errors =",
        rank1.errors,
    )
    return False
