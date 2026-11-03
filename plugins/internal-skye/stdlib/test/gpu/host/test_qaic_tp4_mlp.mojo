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
"""Fused Megatron-LM MLP block across 4 ranks (TP=4): col-parallel then row.

Same fused MLP as the 2-rank test, scaled to 4 devices — still exactly one
collective for the whole block:

    up   : A [H, I] sharded by columns -> A_r [H, I/4]
    down : B [I, H] sharded by rows    -> B_r [I/4, H]

    rank r:  h_r = X   @ A_r   ([M, I/4])   no communication
             Y_r = h_r @ B_r   ([M, H])     partial sum
    all-reduce(SUM):  Y = sum_r Y_r = (X @ A) @ B   ([M, H])

The nonlinearity between the two matmuls is omitted by design (see the 2-rank
test); the sharded two-matmul + all-reduce execution path is unchanged. We
verify against the CPU two-matmul `(X @ A) @ B` over the full weights.
"""

from common import print_separator, print_test_header
from qaic_test_utils import (
    TPRankArgs,
    run_n_ranks_raw,
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

# LLM-tile MLP: M tokens, model dim H, intermediate dim I (sharded), P=4.
comptime N_RANKS: Int = 4
comptime M: Int = 32  # tokens (sequence length)
comptime H: Int = 128  # model / hidden dim
comptime I: Int = 256  # intermediate dim (sharded across ranks)
comptime I_SHARD: Int = I // N_RANKS  # 64 (>= 32, crouton-aligned)

comptime SEED_X: UInt32 = 0x00C0_FFEE
comptime SEED_A: UInt32 = 0xA5A5_1111
comptime SEED_B: UInt32 = 0xB6B6_2222
# Keep magnitudes moderate through the two-matmul chain.
comptime X_SCALE: Float64 = 1.0
comptime A_SCALE: Float64 = 0.25
comptime B_SCALE: Float64 = 0.25

# Two chained FP16 matmuls plus a 4-way FP16 all-reduce.
comptime REL_TOL: Float32 = 0.08


def rank_fn(args: TPRankArgs) raises -> Int:
    var ctx = args.ctx
    var rank = Int(args.rank)

    # Full weights/inputs, materialized identically on every rank.
    var x_full = alloc[Float16](M * H)
    var a_full = alloc[Float16](H * I)
    var b_full = alloc[Float16](I * H)
    fill_fp16_seeded(x_full, M * H, SEED_X, X_SCALE)
    fill_fp16_seeded(a_full, H * I, SEED_A, A_SCALE)
    fill_fp16_seeded(b_full, I * H, SEED_B, B_SCALE)

    # Up-proj weight is column-sharded (carve contiguous); down-proj weight is
    # row-sharded (already contiguous in row-major).
    var a_shard = alloc[Float16](H * I_SHARD)
    extract_col_block(a_full, a_shard, H, I, rank * I_SHARD, I_SHARD)
    var b_shard = alloc[Float16](I_SHARD * H)
    for i in range(I_SHARD * H):
        b_shard[i] = b_full[rank * I_SHARD * H + i]

    var jit_func = compile_and_load_hmx_matmul(ctx)
    var x_buf = alloc_dev_fp16(ctx, args.jit_device, M * H)
    var a_buf = alloc_dev_fp16(ctx, args.jit_device, H * I_SHARD)
    var b_buf = alloc_dev_fp16(ctx, args.jit_device, I_SHARD * H)
    var h_buf = alloc_dev_fp16(ctx, args.jit_device, M * I_SHARD)
    var partial_buf = alloc_dev_fp16(ctx, args.jit_device, M * H)
    var out_buf = alloc_dev_fp16(ctx, args.jit_device, M * H)

    qaic_copy_jit_buffer(
        ctx,
        x_buf,
        x_full.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(M * H * 2),
        QAIC_COPY_TO_DEVICE,
    )
    qaic_copy_jit_buffer(
        ctx,
        a_buf,
        a_shard.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(H * I_SHARD * 2),
        QAIC_COPY_TO_DEVICE,
    )
    qaic_copy_jit_buffer(
        ctx,
        b_buf,
        b_shard.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(I_SHARD * H * 2),
        QAIC_COPY_TO_DEVICE,
    )

    # Up-proj (column-parallel, no comm): h_r = X @ A_r  ([M, I/P]).
    run_hmx_matmul(
        ctx, args.jit_device, jit_func, h_buf, x_buf, a_buf, M, H, I_SHARD
    )
    # Down-proj (row-parallel, partial): Y_r = h_r @ B_r  ([M, H]).
    run_hmx_matmul(
        ctx, args.jit_device, jit_func, partial_buf, h_buf, b_buf, M, I_SHARD, H
    )

    # All-reduce(SUM): out = sum_r Y_r = (X @ A) @ B on every rank.
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
        UInt64(M * H),
        args.comm,
        QCCL_REDUCE_OP_SUM,
        null_wait,
        args.signal_list_ptr,
    )
    qaic_jit_semaphore_wait(ctx, args.done_sem, QAIC_SEM_CHK_GE, 1)

    var y_host = alloc[Float16](M * H)
    qaic_copy_jit_buffer(
        ctx,
        out_buf,
        y_host.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(M * H * 2),
        QAIC_COPY_FROM_DEVICE,
    )

    # Reference: full two-matmul (X @ A) @ B with FP32 accumulation.
    var h_full = alloc[Float16](M * I)
    cpu_matmul_f16(x_full, a_full, h_full, M, H, I)
    var expected = alloc[Float16](M * H)
    cpu_matmul_f16(h_full, b_full, expected, M, I, H)

    var result = verify_matmul_f16(y_host, expected, M * H, rel_tol=REL_TOL)
    print("[rank", args.rank, "] TP=4 fused MLP all-reduce vs (X@A)@B:")
    print_verify_result(result, M * H)

    x_full.unsafe_free()
    a_full.unsafe_free()
    b_full.unsafe_free()
    a_shard.unsafe_free()
    b_shard.unsafe_free()
    h_full.unsafe_free()
    y_host.unsafe_free()
    expected.unsafe_free()
    qaic_release_jit_buffer(ctx, x_buf)
    qaic_release_jit_buffer(ctx, a_buf)
    qaic_release_jit_buffer(ctx, b_buf)
    qaic_release_jit_buffer(ctx, h_buf)
    qaic_release_jit_buffer(ctx, partial_buf)
    qaic_release_jit_buffer(ctx, out_buf)
    qaic_release_jit_function(jit_func)
    return result.errors


def test_qaic_tp4_mlp() raises:
    print_test_header(
        "TP=4 Fused MLP: col-parallel up + row-parallel down + AllReduce (FP16,"
        " M=32 H=128 I=256, 4 ranks)"
    )
    print()
    var ok = run_n_ranks_raw[rank_fn](
        "fused MLP: X@A_shard (col) -> h@B_shard (row) -> all_reduce(SUM)",
        N_RANKS,
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
