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
"""2-rank harness for tensor-parallel matmul + QCCL tests.

`run_2_ranks_raw` factors out the rank scaffolding (pthread spawn/join,
serialized JIT device creation, host barrier, comm bring-up, JIT semaphore)
but — unlike `qccl_runner.run_2_ranks` — does NOT allocate or plumb src/dst
buffers. Tensor-parallel ops need to allocate their own weight/activation/
output buffers, run a matmul, then issue a collective, so the rank closure
owns all of that. The harness hands the closure a ready communicator,
stream, and signal list, and treats the closure's return value as the error
count for the rank.

Also provides `fill_fp16_seeded` (deterministic, so both ranks materialize
identical full matrices) and `extract_col_block` (carve a contiguous column
shard out of a row-major matrix).
"""

from std.atomic import Atomic
from std.ffi import c_int, external_call
from std.memory import UnsafePointer, alloc

from max.gpu.host import DeviceContext
from max._plugin.qaic.host.qaic import (
    QAicDevice,
    QAicStream,
    qaic_create_jit_stream,
    qaic_get_jit_device,
    qaic_release_jit_stream,
)
from max._plugin.qaic.host.qccl import (
    QAicQcclComm,
    QAicQcclUniqueId,
    QAicSemaphore,
    QAicSemaphoreSignal,
    QAicSemaphoreSignalList,
    QAIC_SEM_OP_INC,
    qaic_jit_semaphore_create,
    qaic_jit_semaphore_free,
    qccl_comm_destroy,
    qccl_comm_init_rank,
    qccl_get_unique_id,
)

from .rng import Xorshift32


# ===-----------------------------------------------------------------------===#
# Constants
# ===-----------------------------------------------------------------------===#

# Lower 31 bits = arrival count; high bit = abort signalled by some rank.
comptime BARRIER_ABORT: UInt32 = 0x80000000


# ===-----------------------------------------------------------------------===#
# Host helpers for deterministic inputs
# ===-----------------------------------------------------------------------===#


def fill_fp16_seeded(
    ptr: UnsafePointer[Float16, MutUntrackedOrigin],
    count: Int,
    seed: UInt32,
    scale: Float64 = 1.0,
):
    """Fill `ptr[0:count]` with seed-deterministic FP16 values in `[0, scale)`.

    Both ranks call this with the same seed to materialize bit-identical full
    matrices, so each can compute the same CPU reference independently.

    Args:
        ptr: Destination FP16 buffer.
        count: Number of elements to fill.
        seed: PRNG seed (must be nonzero).
        scale: Upper bound for the generated values (default 1.0).
    """
    var rng = Xorshift32(state=seed)
    rng.fill_fp16(ptr, count, scale)


def extract_col_block(
    src: UnsafePointer[Float16, MutUntrackedOrigin],
    dst: UnsafePointer[Float16, MutUntrackedOrigin],
    rows: Int,
    full_cols: Int,
    c0: Int,
    block_cols: Int,
):
    """Copy the column band `src[:, c0:c0+block_cols]` into contiguous `dst`.

    `src` is row-major `[rows, full_cols]`; `dst` becomes a contiguous
    row-major `[rows, block_cols]` matrix. Used to carve a per-rank shard out
    of a replicated full matrix (the matmul kernel needs contiguous inputs).

    Args:
        src: Source row-major matrix `[rows, full_cols]`.
        dst: Destination row-major matrix `[rows, block_cols]`.
        rows: Number of rows.
        full_cols: Column count of `src`.
        c0: First source column of the band.
        block_cols: Number of columns in the band.
    """
    for r in range(rows):
        for c in range(block_cols):
            dst[r * block_cols + c] = src[r * full_cols + c0 + c]


# ===-----------------------------------------------------------------------===#
# Per-rank argument bundle
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct TPRankArgs(Copyable, Movable):
    """Arguments passed to the per-rank closure invoked by `run_2_ranks_raw`.

    By the time the closure runs, the harness has created the device, stream,
    communicator, and a JIT semaphore (`done_sem`) with its signal list. The
    closure allocates its own buffers, runs the matmul(s), issues the
    collective using `signal_list_ptr`, waits on `done_sem`, reads back, and
    returns the verification error count.
    """

    var ctx: DeviceContext
    var jit_device: QAicDevice
    var stream: QAicStream
    var comm: QAicQcclComm

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
struct _TPRankCtx(Copyable, Movable):
    """Per-rank state shared between the main thread and the worker thread.

    `unique_id` points at one shared id on the spawning thread's stack (all
    ranks copy from it), keeping this struct trivially storable in a heap array.
    """

    var rank: UInt32
    var n_ranks: UInt32

    @__allow_legacy_any_origin_fields
    var unique_id: UnsafePointer[QAicQcclUniqueId, ImmUnsafeAnyOrigin]
    var ok: Bool
    var errors: Int

    @__allow_legacy_any_origin_fields
    var barrier: UnsafePointer[Atomic[UInt32], MutUnsafeAnyOrigin]

    @__allow_legacy_any_origin_fields
    var dev_token: UnsafePointer[Atomic[UInt32], MutUnsafeAnyOrigin]


# Per-instantiation thread entry. The user's rank_fn is comptime-substituted,
# so each instantiation is a concrete function pointer pthread_create accepts.
def _tp_rank_main[
    rank_fn: def(args: TPRankArgs) raises thin -> Int,
](rank_ctx_ptr: UnsafePointer[_TPRankCtx, MutUnsafeAnyOrigin]) raises:
    var rank = rank_ctx_ptr[].rank
    var n_ranks = rank_ctx_ptr[].n_ranks
    var unique_id_local = rank_ctx_ptr[].unique_id[].copy()

    # Serialize device creation across ranks (the QAIC SDK's device-
    # registration table races under concurrent qaicJitDeviceCreate).
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

    var args = TPRankArgs(
        ctx,
        jit_device,
        stream,
        comm,
        UnsafePointer(to=signal_list).as_unsafe_any_origin(),
        done_sem,
        rank,
        n_ranks,
    )
    var errors = rank_fn(args)
    rank_ctx_ptr[].errors = errors
    rank_ctx_ptr[].ok = errors == 0
    print("[rank", rank, "] verified: errors =", errors)

    qaic_jit_semaphore_free(ctx, done_sem)
    qccl_comm_destroy(ctx, comm)
    qaic_release_jit_stream(ctx, stream)


def _tp_rank_thread_entry[
    rank_fn: def(args: TPRankArgs) raises thin -> Int,
](
    arg: UnsafePointer[NoneType, MutUnsafeAnyOrigin],
) -> UnsafePointer[
    NoneType, MutUnsafeAnyOrigin
]:
    """Pthread start routine. Each parameter set yields a concrete fn pointer.
    """
    var ctx_ptr = arg.bitcast[_TPRankCtx]()
    var rank = ctx_ptr[].rank
    try:
        _tp_rank_main[rank_fn](ctx_ptr)
    except e:
        print("[rank", rank, "] error:", e)
        ctx_ptr[].ok = False
        # Release any peer ranks waiting on barrier or dev_token so the test
        # fails fast instead of hanging.
        _barrier_abort(ctx_ptr[].barrier)
        _barrier_abort(ctx_ptr[].dev_token)
    return UnsafePointer[NoneType, MutUnsafeAnyOrigin].unsafe_dangling()


def run_n_ranks_raw[
    rank_fn: def(args: TPRankArgs) raises thin -> Int,
](description: String, n_ranks: Int) raises -> Bool:
    """Run an `n_ranks`-rank tensor-parallel test using the pthread harness.

    Skips (returns True) if fewer than `n_ranks` QAIC devices are visible.

    Each rank thread creates its DeviceContext serially (in rank order, to
    dodge the SDK's device-registration race), brings up the communicator at
    the host barrier, then invokes `rank_fn`, which allocates its own buffers,
    runs the matmul(s), issues the QCCL collective (signalling `args.done_sem`
    via `args.signal_list_ptr`), waits on the semaphore, reads back, and returns
    the verification error count. A rank is OK iff `rank_fn` returns zero.

    Parameters:
        rank_fn: Per-rank closure; returns the verification error count.

    Args:
        description: Human-readable test name (printed in the header).
        n_ranks: Number of ranks/devices to rendezvous (ranks map to devices
            0..n_ranks-1).

    Returns:
        True on success or skip; False on any rank failure.
    """
    print("Running", n_ranks, "-rank tensor-parallel test:", description)

    # Probe the highest-numbered device — if it's missing, so is the quorum.
    try:
        var probe = DeviceContext(device_id=n_ranks - 1, api="qaic")
        _ = probe^
    except e:
        print("  SKIP — need", n_ranks, "visible QAIC devices:", e)
        return True

    var unique_id = qccl_get_unique_id()
    var unique_id_ptr = UnsafePointer(to=unique_id)

    var barrier = Atomic[UInt32](0)
    var dev_token = Atomic[UInt32](0)
    var barrier_ptr = UnsafePointer(to=barrier)
    var dev_token_ptr = UnsafePointer(to=dev_token)

    # All ranks share one unique_id (by pointer) and the barrier/dev_token
    # atomics; each gets its own _TPRankCtx slot in this heap array.
    var ctxs = alloc[_TPRankCtx](n_ranks)
    for r in range(n_ranks):
        ctxs[r] = _TPRankCtx(
            UInt32(r),
            UInt32(n_ranks),
            unique_id_ptr.as_unsafe_any_origin(),
            False,
            -1,
            barrier_ptr.as_unsafe_any_origin(),
            dev_token_ptr.as_unsafe_any_origin(),
        )

    var tids = alloc[UInt64](n_ranks)
    var nullattr = Optional[UnsafePointer[NoneType, MutUnsafeAnyOrigin]]()

    var created = 0
    var create_ok = True
    for r in range(n_ranks):
        var rc = external_call["pthread_create", c_int](
            tids + r,
            nullattr,
            _tp_rank_thread_entry[rank_fn],
            (ctxs + r).bitcast[NoneType](),
        )
        if rc != 0:
            print("  FAIL — pthread_create failed for rank", r, ":", rc)
            create_ok = False
            break
        created += 1

    # If a create failed, abort so the already-running ranks don't hang on the
    # barrier, then join only the threads we actually created.
    if not create_ok:
        _barrier_abort(barrier_ptr.as_unsafe_any_origin())
        _barrier_abort(dev_token_ptr.as_unsafe_any_origin())

    var retval = UnsafePointer[NoneType, MutUnsafeAnyOrigin].unsafe_dangling()
    var join_ok = True
    for r in range(created):
        var jrc = external_call["pthread_join", c_int](
            tids[r], UnsafePointer(to=retval)
        )
        if jrc != 0:
            print("  FAIL — pthread_join failed for rank", r, ":", jrc)
            join_ok = False

    var all_ok = create_ok and join_ok
    if all_ok:
        for r in range(n_ranks):
            if not ctxs[r].ok:
                all_ok = False
                print("  FAIL — rank", r, "errors =", ctxs[r].errors)

    ctxs.unsafe_free()
    tids.unsafe_free()

    if all_ok:
        print("  PASS")
    return all_ok


def run_2_ranks_raw[
    rank_fn: def(args: TPRankArgs) raises thin -> Int,
](description: String) raises -> Bool:
    """Convenience wrapper for the common 2-rank case. See `run_n_ranks_raw`."""
    return run_n_ranks_raw[rank_fn](description, 2)
