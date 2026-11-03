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
"""Column-parallel tensor-parallel linear layer: matmul + all-gather, 2 ranks.

This is the Megatron-LM `ColumnParallelLinear` pattern (QKV projection, MLP
up/gate projection). The input `X [M, K]` is replicated, and the weight
`W [K, N]` is sharded along its columns (the output dim `N`):

    rank r holds  W_r = W[:, r*N/P : (r+1)*N/P]   ([K, N/P])

Each rank computes its output shard `Y_r = X @ W_r` ([M, N/P]) locally with no
communication during the matmul. An `all_gather` then distributes every
shard to every rank.

We use the discrete (per-peer-buffer) all-gather, the form this SDK build
supports: each rank's matmul writes `Y_r` into its own slot `dst_bufs[rank]`
(`src_buf` aliases that slot), peer slots start zeroed, and after the call every
rank holds `dst_bufs[i] = Y_i` for all `i`. The gathered output is therefore in
`[P, M, N/P]` block layout — a rank-major concatenation of column shards, NOT a
row-major `[M, N]` matrix (reconstructing that needs an interleave). We verify
each gathered block `r` against the CPU column-slice matmul `X @ W[:, r-slice]`.
"""

from common import print_separator, print_test_header
from qaic_test_utils import (
    TPRankArgs,
    run_2_ranks_raw,
    fill_fp16_seeded,
    extract_col_block,
    alloc_dev_fp16,
    compile_and_load_hmx_matmul,
    run_hmx_matmul,
    cpu_matmul_f16,
    verify_matmul_f16,
    print_verify_result,
)

from std.memory import UnsafePointer, alloc
from std.testing import TestSuite, assert_true

from max._plugin.qaic.host.qaic import (
    QAicBuf,
    qaic_copy_jit_buffer,
    qaic_release_jit_buffer,
    qaic_release_jit_function,
    QAIC_COPY_FROM_DEVICE,
    QAIC_COPY_TO_DEVICE,
)
from max._plugin.qaic.host.qccl import (
    QAicSemaphoreWaitList,
    QCCL_DTYPE_FLOAT16,
    QAIC_SEM_CHK_GE,
    qaic_jit_semaphore_wait,
    qccl_all_gather_discrete,
)

# LLM-tile shapes: M tokens x K hidden -> N output cols, sharded across P ranks.
comptime N_RANKS: Int = 2
comptime M: Int = 32  # tokens (sequence length)
comptime K: Int = 128  # input hidden dim (replicated)
comptime N: Int = 256  # output dim (sharded across ranks)
comptime N_SHARD: Int = N // N_RANKS

comptime SEED_X: UInt32 = 0x0BAD_CAFE
comptime SEED_W: UInt32 = 0x5EED_1234
comptime X_SCALE: Float64 = 1.0
comptime W_SCALE: Float64 = 1.0

# Pure data movement after a single matmul (no reduction) — only the matmul's
# FP16-output rounding contributes error.
comptime REL_TOL: Float32 = 0.05


def rank_fn(args: TPRankArgs) raises -> Int:
    var ctx = args.ctx
    var rank = Int(args.rank)

    # Full inputs, materialized identically on both ranks via fixed seeds.
    var x_full = alloc[Float16](M * K)
    var w_full = alloc[Float16](K * N)
    fill_fp16_seeded(x_full, M * K, SEED_X, X_SCALE)
    fill_fp16_seeded(w_full, K * N, SEED_W, W_SCALE)

    # This rank's output-column weight shard, carved contiguous.
    var w_shard = alloc[Float16](K * N_SHARD)
    extract_col_block(w_full, w_shard, K, N, rank * N_SHARD, N_SHARD)

    var jit_func = compile_and_load_hmx_matmul(ctx)
    var x_buf = alloc_dev_fp16(ctx, args.jit_device, M * K)
    var w_buf = alloc_dev_fp16(ctx, args.jit_device, K * N_SHARD)
    # One gather slot per rank; the matmul writes this rank's slot in place,
    # the SDK fills the peer slots.
    var dst0 = alloc_dev_fp16(ctx, args.jit_device, M * N_SHARD)
    var dst1 = alloc_dev_fp16(ctx, args.jit_device, M * N_SHARD)

    qaic_copy_jit_buffer(
        ctx,
        x_buf,
        x_full.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(M * K * 2),
        QAIC_COPY_TO_DEVICE,
    )
    qaic_copy_jit_buffer(
        ctx,
        w_buf,
        w_shard.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(K * N_SHARD * 2),
        QAIC_COPY_TO_DEVICE,
    )

    # In-place discrete all-gather: src aliases this rank's own slot; zero the
    # peer slot so it cleanly receives the peer's contribution.
    var zeros = alloc[Float16](M * N_SHARD)
    for i in range(M * N_SHARD):
        zeros[i] = Float16(0)
    var src_buf: QAicBuf
    if rank == 0:
        src_buf = dst0
        qaic_copy_jit_buffer(
            ctx,
            dst1,
            zeros.bitcast[UInt8]().as_unsafe_any_origin(),
            UInt64(M * N_SHARD * 2),
            QAIC_COPY_TO_DEVICE,
        )
    else:
        src_buf = dst1
        qaic_copy_jit_buffer(
            ctx,
            dst0,
            zeros.bitcast[UInt8]().as_unsafe_any_origin(),
            UInt64(M * N_SHARD * 2),
            QAIC_COPY_TO_DEVICE,
        )

    # Local output shard: Y_r = X @ W_r  ([M, N/P]) into this rank's slot.
    run_hmx_matmul(
        ctx, args.jit_device, jit_func, src_buf, x_buf, w_buf, M, K, N_SHARD
    )

    var dst_bufs_arr = alloc[QAicBuf](N_RANKS)
    dst_bufs_arr[0] = dst0
    dst_bufs_arr[1] = dst1
    var dst_offsets_arr = alloc[UInt64](N_RANKS)
    dst_offsets_arr[0] = 0
    dst_offsets_arr[1] = 0

    var null_wait = Optional[
        UnsafePointer[QAicSemaphoreWaitList, ImmUnsafeAnyOrigin]
    ]()
    qccl_all_gather_discrete(
        ctx,
        args.stream,
        src_buf,
        0,
        dst_bufs_arr.as_unsafe_any_origin(),
        dst_offsets_arr.as_unsafe_any_origin(),
        QCCL_DTYPE_FLOAT16,
        UInt64(M * N_SHARD),
        args.comm,
        null_wait,
        args.signal_list_ptr,
    )
    qaic_jit_semaphore_wait(ctx, args.done_sem, QAIC_SEM_CHK_GE, 1)

    # Verify each gathered block r against the CPU column-slice X @ W[:, r-slice].
    var errors = 0
    var gathered = alloc[Float16](M * N_SHARD)
    var expected = alloc[Float16](M * N_SHARD)
    var w_r = alloc[Float16](K * N_SHARD)
    for r in range(N_RANKS):
        qaic_copy_jit_buffer(
            ctx,
            dst_bufs_arr[r],
            gathered.bitcast[UInt8]().as_unsafe_any_origin(),
            UInt64(M * N_SHARD * 2),
            QAIC_COPY_FROM_DEVICE,
        )
        extract_col_block(w_full, w_r, K, N, r * N_SHARD, N_SHARD)
        cpu_matmul_f16(x_full, w_r, expected, M, K, N_SHARD)
        var res = verify_matmul_f16(
            gathered, expected, M * N_SHARD, rel_tol=REL_TOL
        )
        print("[rank", args.rank, "] gathered block", r, "vs X@W shard", r, ":")
        print_verify_result(res, M * N_SHARD)
        errors += res.errors

    x_full.unsafe_free()
    w_full.unsafe_free()
    w_shard.unsafe_free()
    zeros.unsafe_free()
    gathered.unsafe_free()
    expected.unsafe_free()
    w_r.unsafe_free()
    dst_bufs_arr.unsafe_free()
    dst_offsets_arr.unsafe_free()
    qaic_release_jit_buffer(ctx, x_buf)
    qaic_release_jit_buffer(ctx, w_buf)
    qaic_release_jit_buffer(ctx, dst0)
    qaic_release_jit_buffer(ctx, dst1)
    qaic_release_jit_function(jit_func)
    return errors


def test_qaic_tp_column_parallel_all_gather() raises:
    print_test_header(
        "TP Column-Parallel Linear + AllGather (FP16, M=32 K=128 N=256, 2"
        " ranks)"
    )
    print()
    var ok = run_2_ranks_raw[rank_fn](
        "column-parallel linear: X@W_shard + all_gather_discrete"
    )
    print()
    print_separator()
    if ok:
        print("TEST PASSED!")
    else:
        print("TEST FAILED!")
    assert_true(ok)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
