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
"""Exercises the QCCL Mojo bindings.

Two checks:

* unique_id binding smoke (always runs): generate a QCCL unique id and
  verify the SDK populated the 128-byte payload. Validates symbol
  resolution and the `QAicQcclUniqueId` binding layout.

* 2-rank AllReduce (skipped if fewer than 2 QAIC devices are visible):
  `qccl_all_reduce(SUM)` on 256 Float32 elements via the shared
  `run_2_ranks` harness. Each rank contributes `Float32(rank + 1)`; for
  n_ranks=2 the expected sum is 3.0.

The AllReduce check is currently known-broken on the AIC100 SDK:
`qccl_all_reduce` with `QCCL_DTYPE_FLOAT` produces an asymmetric NSP
allocation across ranks (rank 0: nsp 16, rank 1: nsp 1) inside the QCCL
request planner, deadlocking the collective and bricking the device
until reset. The FP16 variant (`test_qaic_qccl_all_reduce_fp16.mojo`)
does not hit this path and passes. Keep this test as a regression
target for the SDK fix; this file is excluded from the `qccl_tests`
smoke suite for now.

Run with:
    mojo plugins/internal-skye/stdlib/test/gpu/host/test_qaic_qccl.mojo
"""

from common import print_separator, print_test_header
from qaic_test_utils import RankOpArgs, run_2_ranks
from std.memory import UnsafePointer
from std.testing import TestSuite, assert_true

from max._plugin.qaic.host.qccl import (
    QAicSemaphoreWaitList,
    QCCL_DTYPE_FLOAT,
    QCCL_REDUCE_OP_SUM,
    QCCL_UNIQUE_ID_BYTES,
    qccl_all_reduce,
    qccl_get_unique_id,
)


def _hex_digit(d: Int) -> String:
    if d < 10:
        return String(d)
    return String(chr(ord("a") + d - 10))


def hex_byte(b: Int) -> String:
    return _hex_digit(b >> 4) + _hex_digit(b & 0xF)


# ===-----------------------------------------------------------------------===#
# unique_id binding smoke test
# ===-----------------------------------------------------------------------===#


def unique_id_smoke() raises -> Bool:
    """Returns True on success."""
    print("unique_id smoke: qccl_get_unique_id()")
    var uid = qccl_get_unique_id()

    # The SDK populates the buffer; expect at least one non-zero byte.
    var any_nonzero = False
    for i in range(QCCL_UNIQUE_ID_BYTES):
        if uid.data[i] != UInt8(0):
            any_nonzero = True
            break

    # Print the first 8 bytes hex for diagnostics.
    var prefix = String("  unique_id[0..8] =")
    for i in range(8):
        prefix += " "
        prefix += hex_byte(Int(uid.data[i]))
    print(prefix)

    if any_nonzero:
        print("  unique_id smoke PASS")
        return True
    print("  unique_id smoke FAIL: unique id is all zero")
    return False


# ===-----------------------------------------------------------------------===#
# 2-rank AllReduce (FLOAT32) collective via run_2_ranks harness
# ===-----------------------------------------------------------------------===#

comptime N_RANKS: Int = 2
comptime NUM_ELEMENTS: Int = 256
comptime DATA_BYTES: UInt64 = UInt64(NUM_ELEMENTS * 4)


def init_host(
    rank: UInt32,
    src_host: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    src_bytes: UInt64,
) raises:
    var f = src_host.bitcast[Float32]()
    var contribution = Float32(Int(rank) + 1)
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
        QCCL_DTYPE_FLOAT,
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
    var f = dst_host.bitcast[Float32]()
    var expected = Float32(N_RANKS * (N_RANKS + 1) // 2)
    var errors = 0
    for i in range(NUM_ELEMENTS):
        if abs(f[i] - expected) > Float32(0.01):
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


# ===-----------------------------------------------------------------------===#
# Test entry
# ===-----------------------------------------------------------------------===#


def test_qaic_qccl() raises:
    print_test_header("QCCL Mojo Bindings Test")
    print()

    var smoke_ok = unique_id_smoke()
    print()
    var collective_ok = run_2_ranks[init_host, op_fn, verify](
        "qccl_all_reduce SUM (FLOAT32, 256 elems)",
        DATA_BYTES,
        DATA_BYTES,
    )
    print()

    print_separator()
    if smoke_ok and collective_ok:
        print("TEST PASSED!")
    else:
        print("TEST FAILED!")
    assert_true(smoke_ok and collective_ok)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
