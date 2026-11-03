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
"""Exercises qccl_all_reduce(SUM) with FLOAT16 on 2 ranks, 256 elements.

Each rank contributes Float16(rank + 1). For n_ranks=2 the expected sum
is Float16(3.0), which is exactly representable.
"""

from common import print_separator, print_test_header
from qaic_test_utils import RankOpArgs, run_2_ranks
from std.memory import UnsafePointer
from std.testing import TestSuite, assert_true

from max._plugin.qaic.host.qccl import (
    QAicSemaphoreWaitList,
    QCCL_DTYPE_FLOAT16,
    QCCL_REDUCE_OP_SUM,
    qccl_all_reduce,
)

comptime N_RANKS: Int = 2
comptime NUM_ELEMENTS: Int = 256
comptime DATA_BYTES: UInt64 = UInt64(NUM_ELEMENTS * 2)


def init_host(
    rank: UInt32,
    src_host: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    src_bytes: UInt64,
) raises:
    var f = src_host.bitcast[Float16]()
    var contribution = Float16(Int(rank) + 1)
    for i in range(NUM_ELEMENTS):
        f[i] = contribution


def op_fn(args: RankOpArgs) raises:
    var null_wait = Optional[
        UnsafePointer[QAicSemaphoreWaitList, ImmUnsafeAnyOrigin]
    ]()
    qccl_all_reduce(
        args.ctx,
        args.stream,
        args.src_buf,
        0,
        args.dst_buf,
        0,
        QCCL_DTYPE_FLOAT16,
        UInt64(NUM_ELEMENTS),
        args.comm,
        QCCL_REDUCE_OP_SUM,
        null_wait,
        args.signal_list_ptr,
    )


def verify(
    rank: UInt32,
    dst_host: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    dst_bytes: UInt64,
) raises -> Int:
    var f = dst_host.bitcast[Float16]()
    var expected = Float16(N_RANKS * (N_RANKS + 1) // 2)  # 1 + 2 = 3
    var errors = 0
    for i in range(NUM_ELEMENTS):
        if abs(f[i] - expected) > Float16(0.01):
            if errors < 5:
                print(
                    "[rank",
                    rank,
                    "] MISMATCH [",
                    i,
                    "] got",
                    f[i],
                    "expected",
                    expected,
                )
            errors += 1
    return errors


def test_qaic_qccl_all_reduce_fp16() raises:
    print_test_header("QCCL AllReduce SUM (FLOAT16, 256 elems, 2 ranks)")
    print()
    var ok = run_2_ranks[init_host, op_fn, verify](
        "qccl_all_reduce SUM (FLOAT16, 256 elems)",
        DATA_BYTES,
        DATA_BYTES,
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
