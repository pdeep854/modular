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
"""Exercises qccl_broadcast (root=0) on 2 ranks, FLOAT, 256 elements.

Rank 0 fills its src with the broadcast pattern. Rank 1's src is left
zeroed (the SDK ignores non-root sources but still requires the param).
After the call both ranks expect to see the rank-0 pattern in `dst`.
"""

from common import print_separator, print_test_header
from qaic_test_utils import RankOpArgs, run_2_ranks
from std.memory import UnsafePointer
from std.testing import TestSuite, assert_true

from max._plugin.qaic.host.qccl import (
    QAicSemaphoreWaitList,
    QCCL_DTYPE_FLOAT,
    qccl_broadcast,
)

comptime NUM_ELEMENTS: Int = 256
comptime DATA_BYTES: UInt64 = UInt64(NUM_ELEMENTS * 4)
comptime ROOT: UInt32 = 0


def init_host(
    rank: UInt32,
    src_host: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    src_bytes: UInt64,
) raises:
    if rank != ROOT:
        return
    var f = src_host.bitcast[Float32]()
    for i in range(NUM_ELEMENTS):
        f[i] = Float32(i + 1) * 3.0


def op_fn(args: RankOpArgs) raises:
    var null_wait = Optional[
        UnsafePointer[QAicSemaphoreWaitList, ImmUnsafeAnyOrigin]
    ]()
    qccl_broadcast(
        args.ctx,
        args.stream,
        args.src_buf,
        0,
        args.dst_buf,
        0,
        QCCL_DTYPE_FLOAT,
        UInt64(NUM_ELEMENTS),
        args.comm,
        ROOT,
        null_wait,
        args.signal_list_ptr,
    )


def verify(
    rank: UInt32,
    dst_host: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    dst_bytes: UInt64,
) raises -> Int:
    var f = dst_host.bitcast[Float32]()
    var errors = 0
    for i in range(NUM_ELEMENTS):
        var expected = Float32(i + 1) * 3.0
        if abs(f[i] - expected) > 0.01:
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


def test_qaic_qccl_broadcast() raises:
    print_test_header("QCCL Broadcast (FLOAT, 256 elems, root=0)")
    print()
    var ok = run_2_ranks[init_host, op_fn, verify](
        "qccl_broadcast (FLOAT, 256 elems, root=0)",
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
