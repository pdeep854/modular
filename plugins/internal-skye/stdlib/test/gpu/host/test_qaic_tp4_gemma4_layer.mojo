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
"""Gemma4-31B TP=4 transformer-layer communication (QCCL collectives).

Captures the Gemma 4 31B-it dimensions (`max/tests/qaic/_gemma4_common.py`
`_REAL_*`: hidden_size=5376, intermediate_size=21504, num_attention_heads=32,
sliding head_dim=256, global head_dim=512, vocab_size=262144) and exercises the
QCCL collectives a Megatron-style tensor-parallel model issues at TP=4:

- Row-parallel reduction (attention out-proj / MLP down-proj): each rank holds a
  partial `[M, H]` output, closed by `all_reduce(SUM)` -> `[M, H]`. This is the
  one collective a decoder layer actually issues, twice per layer.
- Vocab gather (`lm_head`): each rank holds its `[M, V/4]` logit shard, closed
  by `all_gather` -> `[M, V]`.
- Column-parallel gather at the MLP shard width `[M, I/4]`. A primitive-level
  check at that width and rank count only: Megatron TP feeds column-parallel
  up/gate straight into row-parallel down and reduces only the down output, so
  the model itself never emits this gather.

These are the QCCL primitives underneath the collectives. The `mo.distributed.*`
graph ops the model actually lowers to are covered at these same rank counts by
`max/tests/qaic/test_collectives_tp4.py`.

To isolate and verify the *communication* at these shapes, the per-rank tensors
are filled with known per-rank sentinels (rank r -> r+1) standing in for the
projection outputs, so the reduced/gathered result is exactly predictable in
FP16 (small-integer sums are exact). The collective calls, dtype, buffer shapes,
and TP=4 rendezvous match a real layer; only the matmul that would produce the
operands is elided.

To run (requires 4 QAIC devices):
    mojo test_qaic_tp4_gemma4_layer.mojo
"""

from common import print_separator, print_test_header
from qaic_test_utils import TPRankArgs, run_n_ranks_raw, alloc_dev_fp16

from std.memory import UnsafePointer, alloc
from std.testing import TestSuite, assert_true

from max._plugin.qaic.host.qaic import (
    QAicBuf,
    qaic_copy_jit_buffer,
    qaic_release_jit_buffer,
    QAIC_COPY_FROM_DEVICE,
    QAIC_COPY_TO_DEVICE,
)
from max._plugin.qaic.host.qccl import (
    QAicSemaphoreWaitList,
    QCCL_DTYPE_FLOAT16,
    QCCL_REDUCE_OP_SUM,
    QAIC_SEM_CHK_GE,
    qaic_jit_semaphore_wait,
    qccl_all_gather_discrete,
    qccl_all_reduce,
)

# --- Gemma 4 31B-it shapes (max/tests/qaic/_gemma4_common.py `_REAL_*`) ---
comptime N_RANKS: Int = 4
comptime M: Int = 32  # token tile (sequence length)
comptime H: Int = 5376  # hidden_size
comptime I: Int = 21504  # intermediate_size
comptime I_SHARD: Int = I // N_RANKS  # 5376, column-parallel MLP shard
comptime VOCAB: Int = 262144  # vocab_size
comptime VOCAB_SHARD: Int = VOCAB // N_RANKS  # 65536, lm_head logit shard

# Rank r contributes the constant (r+1); SUM over the 4 ranks = 10, which is
# exact in FP16, so the reduced/gathered values compare with ==.
comptime RANK_VALUE_SUM: Int = N_RANKS * (N_RANKS + 1) // 2  # 10


def _fill(
    ptr: UnsafePointer[Float16, MutUntrackedOrigin], count: Int, value: Float16
):
    for i in range(count):
        ptr[i] = value


def _all_reduce_known(args: TPRankArgs, count: Int) raises -> Int:
    """Row-parallel reduction: fill `[count]` with (rank+1), `all_reduce(SUM)`,
    verify every element equals sum_r (r+1)."""
    var ctx = args.ctx
    var rank = Int(args.rank)

    var src_buf = alloc_dev_fp16(ctx, args.jit_device, count)
    var out_buf = alloc_dev_fp16(ctx, args.jit_device, count)

    var host = alloc[Float16](count)
    _fill(host, count, Float16(rank + 1))
    qaic_copy_jit_buffer(
        ctx,
        src_buf,
        host.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(count * 2),
        QAIC_COPY_TO_DEVICE,
    )

    var null_wait = Optional[
        UnsafePointer[QAicSemaphoreWaitList, ImmUnsafeAnyOrigin]
    ]()
    qccl_all_reduce(
        ctx,
        args.stream,
        src_buf,
        0,
        out_buf,
        0,
        QCCL_DTYPE_FLOAT16,
        UInt64(count),
        args.comm,
        QCCL_REDUCE_OP_SUM,
        null_wait,
        args.signal_list_ptr,
    )
    qaic_jit_semaphore_wait(ctx, args.done_sem, QAIC_SEM_CHK_GE, 1)

    qaic_copy_jit_buffer(
        ctx,
        out_buf,
        host.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(count * 2),
        QAIC_COPY_FROM_DEVICE,
    )
    var expected = Float16(RANK_VALUE_SUM)
    var errors = 0
    for i in range(count):
        if host[i] != expected:
            errors += 1
    print(
        "[rank",
        args.rank,
        "] all_reduce(SUM) of",
        count,
        "elems: errors =",
        errors,
    )

    host.unsafe_free()
    qaic_release_jit_buffer(ctx, src_buf)
    qaic_release_jit_buffer(ctx, out_buf)
    return errors


def _all_gather_known(args: TPRankArgs, shard: Int) raises -> Int:
    """Column-parallel gather: each rank fills its own slot with (rank+1),
    `all_gather_discrete`, verify gathered block r equals (r+1) on every rank.
    """
    var ctx = args.ctx
    var rank = Int(args.rank)

    # One gather slot per rank; this rank writes its own slot in place, the SDK
    # fills the peer slots (discrete all-gather is the form this build supports).
    var dst_bufs = alloc[QAicBuf](N_RANKS)
    for r in range(N_RANKS):
        dst_bufs[r] = alloc_dev_fp16(ctx, args.jit_device, shard)

    var host = alloc[Float16](shard)
    # Zero every peer slot so it cleanly receives the peer's contribution.
    _fill(host, shard, Float16(0))
    for r in range(N_RANKS):
        if r != rank:
            qaic_copy_jit_buffer(
                ctx,
                dst_bufs[r],
                host.bitcast[UInt8]().as_unsafe_any_origin(),
                UInt64(shard * 2),
                QAIC_COPY_TO_DEVICE,
            )
    # This rank's contribution.
    _fill(host, shard, Float16(rank + 1))
    qaic_copy_jit_buffer(
        ctx,
        dst_bufs[rank],
        host.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(shard * 2),
        QAIC_COPY_TO_DEVICE,
    )

    var src_buf = dst_bufs[rank]
    var dst_offsets = alloc[UInt64](N_RANKS)
    for r in range(N_RANKS):
        dst_offsets[r] = 0

    var null_wait = Optional[
        UnsafePointer[QAicSemaphoreWaitList, ImmUnsafeAnyOrigin]
    ]()
    qccl_all_gather_discrete(
        ctx,
        args.stream,
        src_buf,
        0,
        dst_bufs.as_unsafe_any_origin(),
        dst_offsets.as_unsafe_any_origin(),
        QCCL_DTYPE_FLOAT16,
        UInt64(shard),
        args.comm,
        null_wait,
        args.signal_list_ptr,
    )
    qaic_jit_semaphore_wait(ctx, args.done_sem, QAIC_SEM_CHK_GE, 1)

    var errors = 0
    for r in range(N_RANKS):
        qaic_copy_jit_buffer(
            ctx,
            dst_bufs[r],
            host.bitcast[UInt8]().as_unsafe_any_origin(),
            UInt64(shard * 2),
            QAIC_COPY_FROM_DEVICE,
        )
        var expected = Float16(r + 1)
        for i in range(shard):
            if host[i] != expected:
                errors += 1
    print(
        "[rank",
        args.rank,
        "] all_gather of",
        N_RANKS,
        "x",
        shard,
        "elems: errors =",
        errors,
    )

    host.unsafe_free()
    for r in range(N_RANKS):
        qaic_release_jit_buffer(ctx, dst_bufs[r])
    dst_bufs.unsafe_free()
    dst_offsets.unsafe_free()
    return errors


# Thin rank_fn wrappers so each instantiation is a concrete fn pointer the
# harness can hand to pthread_create.
def _row_parallel_all_reduce(args: TPRankArgs) raises -> Int:
    return _all_reduce_known(args, M * H)


def _mlp_shard_all_gather(args: TPRankArgs) raises -> Int:
    return _all_gather_known(args, M * I_SHARD)


def _lm_head_all_gather(args: TPRankArgs) raises -> Int:
    return _all_gather_known(args, M * VOCAB_SHARD)


def test_qaic_tp4_gemma4_row_parallel_all_reduce() raises:
    print_test_header(
        "TP=4 Gemma4 row-parallel AllReduce(SUM) over [M=32, H=5376], 4 ranks"
    )
    print()
    var ok = run_n_ranks_raw[_row_parallel_all_reduce](
        (
            "Gemma4 attn out-proj / MLP down-proj: partial [M,H] ->"
            " all_reduce(SUM)"
        ),
        N_RANKS,
    )
    print()
    print_separator()
    if ok:
        print("TEST PASSED!")
    else:
        print("TEST FAILED!")
    assert_true(ok)


def test_qaic_tp4_gemma4_mlp_shard_all_gather() raises:
    print_test_header(
        "TP=4 Gemma4 AllGather at MLP shard width [M=32, I/4=5376] -> [M,"
        " I=21504], 4 ranks"
    )
    print()
    var ok = run_n_ranks_raw[_mlp_shard_all_gather](
        "Gather at the column-parallel MLP shard width: [M,I/4] -> [M,I]",
        N_RANKS,
    )
    print()
    print_separator()
    if ok:
        print("TEST PASSED!")
    else:
        print("TEST FAILED!")
    assert_true(ok)


def test_qaic_tp4_gemma4_lm_head_all_gather() raises:
    print_test_header(
        "TP=4 Gemma4 lm_head AllGather [M=32, V/4=65536] -> [M, V=262144], 4"
        " ranks"
    )
    print()
    var ok = run_n_ranks_raw[_lm_head_all_gather](
        "Gemma4 lm_head column-parallel: [M,V/4] -> all_gather -> [M,V]",
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
