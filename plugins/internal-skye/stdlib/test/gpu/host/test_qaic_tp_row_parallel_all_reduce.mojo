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
"""Row-parallel tensor-parallel linear layer: matmul + all-reduce on 2 ranks.

This is the Megatron-LM `RowParallelLinear` pattern (attention output
projection, MLP down-projection). The weight `W [K, N]` is sharded along its
rows (the contraction dim `K`), and the input `X [M, K]` is sharded along `K`
to match:

    rank r holds  X_r = X[:, r*K/P : (r+1)*K/P]   ([M, K/P])
                  W_r = W[r*K/P : (r+1)*K/P, :]    ([K/P, N])

Each rank computes a partial product `P_r = X_r @ W_r` ([M, N]) locally, then a
single `all_reduce(SUM)` sums the partials across ranks:

    Y = sum_r P_r = X @ W   ([M, N])   (identical on every rank)

The all-reduce is the only communication in a row-parallel forward. We verify
the reduced result against a full-`K` CPU matmul `X @ W`.
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
    qaic_copy_jit_buffer,
    qaic_release_jit_buffer,
    qaic_release_jit_function,
    QAIC_COPY_FROM_DEVICE,
    QAIC_COPY_TO_DEVICE,
)
from max._plugin.qaic.host.qccl import (
    QAicSemaphoreWaitList,
    QCCL_DTYPE_FLOAT16,
    QCCL_REDUCE_OP_SUM,
    QAIC_SEM_CHK_GE,
    qaic_jit_semaphore_wait,
    qccl_all_reduce,
)

# LLM-tile shapes: M tokens x K input-hidden -> N output-hidden, P=2.
comptime N_RANKS: Int = 2
comptime M: Int = 32  # tokens (sequence length)
comptime K: Int = 128  # input hidden dim (sharded across ranks)
comptime N: Int = 128  # output hidden dim
comptime K_SHARD: Int = K // N_RANKS

comptime SEED_X: UInt32 = 0x1234_5678
comptime SEED_W: UInt32 = 0x2BAD_F00D
comptime X_SCALE: Float64 = 1.0
comptime W_SCALE: Float64 = 1.0

# Two FP16 partials summed (effective K=128) — leave headroom over the matmul's
# own FP16-output rounding.
comptime REL_TOL: Float32 = 0.06


def rank_fn(args: TPRankArgs) raises -> Int:
    var ctx = args.ctx
    var rank = Int(args.rank)

    # Full inputs, materialized identically on both ranks via fixed seeds.
    var x_full = alloc[Float16](M * K)
    var w_full = alloc[Float16](K * N)
    fill_fp16_seeded(x_full, M * K, SEED_X, X_SCALE)
    fill_fp16_seeded(w_full, K * N, SEED_W, W_SCALE)

    # This rank's contraction-dim shard (both inputs sharded along K).
    var x_shard = alloc[Float16](M * K_SHARD)
    extract_col_block(x_full, x_shard, M, K, rank * K_SHARD, K_SHARD)
    # W row band is already contiguous in row-major layout.
    var w_shard = alloc[Float16](K_SHARD * N)
    for i in range(K_SHARD * N):
        w_shard[i] = w_full[rank * K_SHARD * N + i]

    var jit_func = compile_and_load_hmx_matmul(ctx)
    var x_buf = alloc_dev_fp16(ctx, args.jit_device, M * K_SHARD)
    var w_buf = alloc_dev_fp16(ctx, args.jit_device, K_SHARD * N)
    var partial_buf = alloc_dev_fp16(ctx, args.jit_device, M * N)
    var out_buf = alloc_dev_fp16(ctx, args.jit_device, M * N)

    qaic_copy_jit_buffer(
        ctx,
        x_buf,
        x_shard.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(M * K_SHARD * 2),
        QAIC_COPY_TO_DEVICE,
    )
    qaic_copy_jit_buffer(
        ctx,
        w_buf,
        w_shard.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(K_SHARD * N * 2),
        QAIC_COPY_TO_DEVICE,
    )

    # Local partial: P_r = X_r @ W_r  ([M, N]).
    run_hmx_matmul(
        ctx, args.jit_device, jit_func, partial_buf, x_buf, w_buf, M, K_SHARD, N
    )

    # All-reduce(SUM): out = sum_r P_r = X @ W on every rank.
    var null_wait = Optional[
        UnsafePointer[QAicSemaphoreWaitList, ImmUnsafeAnyOrigin]
    ]()
    qccl_all_reduce(
        ctx,
        args.stream,
        partial_buf,
        0,
        out_buf,
        0,
        QCCL_DTYPE_FLOAT16,
        UInt64(M * N),
        args.comm,
        QCCL_REDUCE_OP_SUM,
        null_wait,
        args.signal_list_ptr,
    )
    qaic_jit_semaphore_wait(ctx, args.done_sem, QAIC_SEM_CHK_GE, 1)

    var y_host = alloc[Float16](M * N)
    qaic_copy_jit_buffer(
        ctx,
        out_buf,
        y_host.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(M * N * 2),
        QAIC_COPY_FROM_DEVICE,
    )

    # Reference: full X @ W (FP32 accumulation, FP16 output).
    var expected = alloc[Float16](M * N)
    cpu_matmul_f16(x_full, w_full, expected, M, K, N)

    var result = verify_matmul_f16(y_host, expected, M * N, rel_tol=REL_TOL)
    print("[rank", args.rank, "] row-parallel all-reduce vs full X@W:")
    print_verify_result(result, M * N)

    x_full.unsafe_free()
    w_full.unsafe_free()
    x_shard.unsafe_free()
    w_shard.unsafe_free()
    y_host.unsafe_free()
    expected.unsafe_free()
    qaic_release_jit_buffer(ctx, x_buf)
    qaic_release_jit_buffer(ctx, w_buf)
    qaic_release_jit_buffer(ctx, partial_buf)
    qaic_release_jit_buffer(ctx, out_buf)
    qaic_release_jit_function(jit_func)
    return result.errors


def test_qaic_tp_row_parallel_all_reduce() raises:
    print_test_header(
        "TP Row-Parallel Linear + AllReduce (FP16, M=32 K=128 N=128, 2 ranks)"
    )
    print()
    var ok = run_2_ranks_raw[rank_fn](
        "row-parallel linear: X@W via K-shard partials + all_reduce(SUM)"
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
