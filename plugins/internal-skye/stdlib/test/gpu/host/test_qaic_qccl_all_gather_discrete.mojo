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
"""Exercises qccl_all_gather_discrete (in-place) on 2 ranks, FLOAT, 256
elements/rank.

Each rank allocates `n_ranks` distinct dst buffers (one per peer rank).
For in-place semantics — the only mode this build supports — `src_buf`
must alias `dst_bufs[rank]` (rank's "self" slot) so the gather happens
in place. The rank's contribution is preloaded into that slot; peer
slots start zeroed and receive peers' contributions from the SDK.

After the call, `dst_bufs[i]` holds rank `i`'s contribution on every
rank.

Like the in-place contiguous AllGather test, this does its own pthread +
comm setup rather than using `run_2_ranks` because the per-peer buffer
array doesn't fit the shared runner's single-dst model.
"""

from common import print_separator, print_test_header

from std.atomic import Atomic
from std.ffi import c_int, external_call
from std.memory import UnsafePointer, alloc
from std.testing import TestSuite, assert_true

from max.gpu.host import DeviceContext
from max._plugin.qaic.host.qaic import (
    QAicBuf,
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
    QAicQcclUniqueId,
    QAicSemaphoreSignal,
    QAicSemaphoreSignalList,
    QAicSemaphoreWaitList,
    QCCL_DTYPE_FLOAT,
    QAIC_SEM_CHK_GE,
    QAIC_SEM_OP_INC,
    qaic_jit_semaphore_create,
    qaic_jit_semaphore_free,
    qaic_jit_semaphore_wait,
    qccl_all_gather_discrete,
    qccl_comm_destroy,
    qccl_comm_init_rank,
    qccl_get_unique_id,
)

comptime N_RANKS: UInt32 = 2
comptime ELEMS_PER_RANK: Int = 256
comptime CHUNK_BYTES: UInt64 = UInt64(ELEMS_PER_RANK * 4)
comptime ALIGNMENT: UInt32 = 128
comptime CU_BITMAP: UInt64 = 0xFFFFFFFFFFFFFFFF
comptime BARRIER_ABORT: UInt32 = 0x80000000


def _rank_offset(rank: UInt32) -> Float32:
    return Float32(100 + 100 * Int(rank))


@fieldwise_init
struct RankCtx:
    var rank: UInt32
    var n_ranks: UInt32
    var unique_id: QAicQcclUniqueId
    var ok: Bool
    var errors: Int

    @__allow_legacy_any_origin_fields
    var barrier: UnsafePointer[Atomic[UInt32], MutUnsafeAnyOrigin]

    @__allow_legacy_any_origin_fields
    var dev_token: UnsafePointer[Atomic[UInt32], MutUnsafeAnyOrigin]


def barrier_arrive_and_wait(
    barrier: UnsafePointer[Atomic[UInt32], MutUnsafeAnyOrigin],
    n_ranks: UInt32,
) raises:
    _ = barrier[].fetch_add(UInt32(1))
    while True:
        var v = barrier[].load()
        if (v & BARRIER_ABORT) != 0:
            raise Error("barrier aborted by peer rank")
        if (v & ~BARRIER_ABORT) >= n_ranks:
            return
        _ = external_call["sched_yield", c_int]()


def barrier_abort(
    barrier: UnsafePointer[Atomic[UInt32], MutUnsafeAnyOrigin],
):
    _ = barrier[].fetch_add(BARRIER_ABORT)


def rank_main(rank_ctx_ptr: UnsafePointer[RankCtx, MutUnsafeAnyOrigin]) raises:
    var rank = rank_ctx_ptr[].rank
    var n_ranks = rank_ctx_ptr[].n_ranks
    var unique_id = rank_ctx_ptr[].unique_id.copy()

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

    # Allocate one dst buffer per peer rank; src_buf aliases the rank's
    # own slot (in-place form, the only one this SDK build supports).
    var dst_buf_0 = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        CHUNK_BYTES,
        ALIGNMENT,
        CU_BITMAP,
    )
    var dst_buf_1 = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        CHUNK_BYTES,
        ALIGNMENT,
        CU_BITMAP,
    )
    var src_buf: QAicBuf
    if rank == UInt32(0):
        src_buf = dst_buf_0
    else:
        src_buf = dst_buf_1

    var contribution = alloc[Float32](ELEMS_PER_RANK)
    var base = _rank_offset(rank)
    for i in range(ELEMS_PER_RANK):
        contribution[i] = base + Float32(i)

    var zeros = alloc[Float32](ELEMS_PER_RANK)
    for i in range(ELEMS_PER_RANK):
        zeros[i] = 0.0

    # Preload only this rank's slot; peer slots start zeroed.
    qaic_copy_jit_buffer(
        ctx,
        src_buf,
        contribution.bitcast[UInt8]().as_unsafe_any_origin(),
        CHUNK_BYTES,
        QAIC_COPY_TO_DEVICE,
    )
    if rank == UInt32(0):
        qaic_copy_jit_buffer(
            ctx,
            dst_buf_1,
            zeros.bitcast[UInt8]().as_unsafe_any_origin(),
            CHUNK_BYTES,
            QAIC_COPY_TO_DEVICE,
        )
    else:
        qaic_copy_jit_buffer(
            ctx,
            dst_buf_0,
            zeros.bitcast[UInt8]().as_unsafe_any_origin(),
            CHUNK_BYTES,
            QAIC_COPY_TO_DEVICE,
        )

    var done_sem = qaic_jit_semaphore_create(ctx, 0)
    var signal = QAicSemaphoreSignal(done_sem, QAIC_SEM_OP_INC, 1)
    var signal_list = QAicSemaphoreSignalList(
        1, UnsafePointer(to=signal).as_unsafe_any_origin()
    )
    var signal_list_ptr: UnsafePointer[
        QAicSemaphoreSignalList, ImmUnsafeAnyOrigin
    ] = UnsafePointer(to=signal_list).as_unsafe_any_origin()

    print("[rank", rank, "] reached pre-comm barrier")
    barrier_arrive_and_wait(rank_ctx_ptr[].barrier, n_ranks)

    print("[rank", rank, "] qccl_comm_init_rank")
    var comm = qccl_comm_init_rank(ctx, jit_device, unique_id, rank, n_ranks)
    print("[rank", rank, "] comm initialized")

    var dst_bufs_arr = alloc[QAicBuf](Int(N_RANKS))
    dst_bufs_arr[0] = dst_buf_0
    dst_bufs_arr[1] = dst_buf_1
    var dst_offsets_arr = alloc[UInt64](Int(N_RANKS))
    dst_offsets_arr[0] = 0
    dst_offsets_arr[1] = 0

    print(
        "[rank",
        rank,
        "] submitting AllGatherDiscrete (",
        ELEMS_PER_RANK,
        "elems/rank)",
    )
    qccl_all_gather_discrete(
        ctx,
        stream,
        src_buf,
        0,
        dst_bufs_arr.as_unsafe_any_origin(),
        dst_offsets_arr.as_unsafe_any_origin(),
        QCCL_DTYPE_FLOAT,
        UInt64(ELEMS_PER_RANK),
        comm,
        Optional[UnsafePointer[QAicSemaphoreWaitList, ImmUnsafeAnyOrigin]](),
        signal_list_ptr,
    )

    print("[rank", rank, "] waiting on done semaphore")
    qaic_jit_semaphore_wait(ctx, done_sem, QAIC_SEM_CHK_GE, 1)
    print("[rank", rank, "] AllGatherDiscrete complete")

    var dst_host_0 = alloc[Float32](ELEMS_PER_RANK)
    var dst_host_1 = alloc[Float32](ELEMS_PER_RANK)
    qaic_copy_jit_buffer(
        ctx,
        dst_buf_0,
        dst_host_0.bitcast[UInt8]().as_unsafe_any_origin(),
        CHUNK_BYTES,
        QAIC_COPY_FROM_DEVICE,
    )
    qaic_copy_jit_buffer(
        ctx,
        dst_buf_1,
        dst_host_1.bitcast[UInt8]().as_unsafe_any_origin(),
        CHUNK_BYTES,
        QAIC_COPY_FROM_DEVICE,
    )

    var errors = 0
    var b0 = _rank_offset(0)
    var b1 = _rank_offset(1)
    for i in range(ELEMS_PER_RANK):
        var e0 = b0 + Float32(i)
        var e1 = b1 + Float32(i)
        if abs(dst_host_0[i] - e0) > 0.01:
            if errors < 5:
                print(
                    "[rank",
                    rank,
                    "] dst[0] MISMATCH [",
                    i,
                    "] got",
                    dst_host_0[i],
                    "expected",
                    e0,
                )
            errors += 1
        if abs(dst_host_1[i] - e1) > 0.01:
            if errors < 5:
                print(
                    "[rank",
                    rank,
                    "] dst[1] MISMATCH [",
                    i,
                    "] got",
                    dst_host_1[i],
                    "expected",
                    e1,
                )
            errors += 1
    print("[rank", rank, "] verified: errors =", errors)

    rank_ctx_ptr[].errors = errors
    rank_ctx_ptr[].ok = errors == 0

    contribution.unsafe_free()
    zeros.unsafe_free()
    dst_host_0.unsafe_free()
    dst_host_1.unsafe_free()
    dst_bufs_arr.unsafe_free()
    dst_offsets_arr.unsafe_free()
    qaic_jit_semaphore_free(ctx, done_sem)
    qccl_comm_destroy(ctx, comm)
    # src_buf aliases dst_buf_{rank}, so don't release it twice.
    qaic_release_jit_buffer(ctx, dst_buf_0)
    qaic_release_jit_buffer(ctx, dst_buf_1)
    qaic_release_jit_stream(ctx, stream)


def rank_worker(
    arg: UnsafePointer[NoneType, MutUnsafeAnyOrigin],
) -> UnsafePointer[NoneType, MutUnsafeAnyOrigin]:
    var ctx_ptr = arg.bitcast[RankCtx]()
    var rank = ctx_ptr[].rank
    try:
        rank_main(ctx_ptr)
    except e:
        print("[rank", rank, "] error:", e)
        ctx_ptr[].ok = False
        barrier_abort(ctx_ptr[].barrier)
        barrier_abort(ctx_ptr[].dev_token)
    return UnsafePointer[NoneType, MutUnsafeAnyOrigin].unsafe_dangling()


def run() raises -> Bool:
    print("Running 2-rank AllGatherDiscrete (FLOAT, 256 elems/rank)")

    try:
        var probe = DeviceContext(device_id=1, api="qaic")
        _ = probe^
    except e:
        print("  SKIP — need 2 visible QAIC devices:", e)
        return True

    var unique_id = qccl_get_unique_id()

    var barrier = Atomic[UInt32](0)
    var dev_token = Atomic[UInt32](0)
    var barrier_ptr = UnsafePointer(to=barrier).as_unsafe_any_origin()
    var dev_token_ptr = UnsafePointer(to=dev_token).as_unsafe_any_origin()

    var rank0 = RankCtx(
        0, N_RANKS, unique_id.copy(), False, -1, barrier_ptr, dev_token_ptr
    )
    var rank1 = RankCtx(
        1, N_RANKS, unique_id.copy(), False, -1, barrier_ptr, dev_token_ptr
    )

    var tid0: UInt64 = 0
    var tid1: UInt64 = 0
    var nullattr = Optional[UnsafePointer[NoneType, MutUnsafeAnyOrigin]]()

    var rc0 = external_call["pthread_create", c_int](
        UnsafePointer(to=tid0),
        nullattr,
        rank_worker,
        UnsafePointer(to=rank0).bitcast[NoneType](),
    )
    var rc1 = external_call["pthread_create", c_int](
        UnsafePointer(to=tid1),
        nullattr,
        rank_worker,
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


def test_qaic_qccl_all_gather_discrete() raises:
    print_test_header("QCCL AllGatherDiscrete (FLOAT, 256 elems/rank, 2 ranks)")
    print()
    var ok = run()
    print()
    print_separator()
    if ok:
        print("TEST PASSED!")
    else:
        print("TEST FAILED!")
    assert_true(ok)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
