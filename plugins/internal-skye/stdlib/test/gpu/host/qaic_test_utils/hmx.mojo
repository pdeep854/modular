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
"""Shared HMX matmul test utilities.

Provides kernel-side helpers (VTCM alignment, HMX thread detection) and
host-side helpers (CPU reference matmul, verification) used across all
HMX matmul test variants.
"""

from std.memory import UnsafePointer
from std.sys import exit
from std.utils.index import IndexList

from layout import Coord, Idx, TileTensor, row_major
from std.builtin.dtype import DType
from linalg.matmul import matmul
from linalg.utils import elementwise_epilogue_type

from std._plugin.qaic.host.target import npu_thread_idx
from std._plugin.qaic.host.qshim import (
    qshim_query,
    QSHIM_QUERY_HMX_THREAD_ID,
)


# ===-----------------------------------------------------------------------===#
# Kernel-side helpers
# ===-----------------------------------------------------------------------===#


@always_inline
def vtcm_align(off: Int, alignment: Int) -> Int:
    """Round `off` up to the next multiple of `alignment`.

    Args:
        off: The offset to align.
        alignment: The required alignment in bytes (must be a power of 2).

    Returns:
        The smallest offset >= `off` that is a multiple of `alignment`.
    """
    return (off + alignment - 1) & ~(alignment - 1)


@always_inline
def npu_is_hmx_thread() -> Bool:
    """Return True if the calling thread is the HMX/DMA thread.

    Queries the HMX thread ID via qshim and compares to the caller's
    thread index.

    Returns:
        True if the caller is the HMX thread, False otherwise.

    """
    var hmx_tid: Int64 = -1
    var status = qshim_query(
        QSHIM_QUERY_HMX_THREAD_ID,
        UnsafePointer(to=hmx_tid).as_unsafe_any_origin(),
    )
    debug_assert(
        status == 0 and hmx_tid >= 0, "qshim HMX thread-id query failed"
    )
    return UInt32(npu_thread_idx()) == UInt32(hmx_tid)


# ===-----------------------------------------------------------------------===#
# Host-side helpers
# ===-----------------------------------------------------------------------===#


def cpu_matmul_f16(
    A: UnsafePointer[Float16, MutUntrackedOrigin],
    B: UnsafePointer[Float16, MutUntrackedOrigin],
    C: UnsafePointer[Float16, MutUntrackedOrigin],
    M: Int,
    K: Int,
    N: Int,
    naive: Bool = False,
) raises:
    """Compute C = A * B (f16 inputs, fp32 accumulation, f16 output).

    Args:
        A: LHS matrix [M, K] in row-major order.
        B: RHS matrix [K, N] in row-major order.
        C: Output matrix [M, N] in row-major order.
        M: Number of output rows.
        K: Common (contraction) dimension.
        N: Number of output columns.
        naive: If True, use a triple-loop implementation. If False (default),
            dispatch to `linalg.matmul` with a Float32 scratch C buffer and an
            elementwise epilogue that casts each tile to Float16 while the
            Float32 accumulator values are still hot in cache. The epilogue
            path is needed because the CPU matmul's inner kernel accumulates
            in `c.dtype`, so a Float32 C is required to get Float32
            accumulation.
    """
    if naive:
        # i-k-j loop order for cache-friendly B access (stride-1 in inner loop).
        # Accumulate in a Float32 buffer then cast to Float16 once at the end.
        var accum = alloc[Float32](M * N)
        for idx in range(M * N):
            accum[idx] = Float32(0)
        for i in range(M):
            for k_idx in range(K):
                var a_val = Float32(A[i * K + k_idx])
                for j in range(N):
                    accum[i * N + j] += a_val * Float32(B[k_idx * N + j])
        for idx in range(M * N):
            C[idx] = Float16(accum[idx])
        accum.unsafe_free()
        return

    var C_f32 = alloc[Float32](M * N)

    @__parameter
    @always_inline
    def cast_to_f16_epilogue[
        dtype: DType, width: SIMDLength, *, alignment: Int = 1
    ](coords: IndexList[2], val: SIMD[dtype, width]):
        (C + coords[0] * N + coords[1]).store[width=width, alignment=alignment](
            val.cast[DType.float16]()
        )

    matmul[
        elementwise_lambda_fn=Optional[elementwise_epilogue_type](
            cast_to_f16_epilogue
        ),
    ](
        TileTensor(C_f32, row_major(Coord(Int(M), Int(N)))),
        TileTensor(A, row_major(Coord(Int(M), Int(K)))),
        TileTensor(B, row_major(Coord(Int(K), Int(N)))),
        None,
    )
    C_f32.unsafe_free()


@fieldwise_init
struct MatmulVerifyResult:
    """Aggregated verification statistics from a matmul comparison."""

    var errors: Int
    """Number of elements that failed verification."""

    var max_err: Float32
    """Maximum absolute error across all elements."""

    var mean_err: Float32
    """Mean absolute error across all elements."""

    def passed(self) -> Bool:
        """Return True if no errors were found.

        Returns:
            True if `errors` is 0, False otherwise.
        """
        return self.errors == 0


def verify_matmul_f16(
    actual: UnsafePointer[Float16, MutUntrackedOrigin],
    expected: UnsafePointer[Float16, MutUntrackedOrigin],
    size: Int,
    rel_tol: Float32 = 0.05,
    max_errors_to_print: Int = 5,
) -> MatmulVerifyResult:
    """Compare actual vs expected Float16 buffers element-wise.

    Uses relative tolerance: an element fails if
    ``abs(actual - expected) / max(abs(expected), 1.0) > rel_tol``.

    Args:
        actual: Pointer to actual (device) output.
        expected: Pointer to CPU-reference output.
        size: Number of elements to compare.
        rel_tol: Relative error threshold (default 5%).
        max_errors_to_print: Cap on individual error messages.

    Returns:
        A `MatmulVerifyResult` with error count, max, and mean absolute error.
    """
    var errors = 0
    var max_err: Float32 = 0.0
    var sum_err: Float32 = 0.0
    for idx in range(size):
        var exp = Float32(expected[idx])
        var act = Float32(actual[idx])
        var err = abs(act - exp)
        sum_err += err
        var ref_val = max(abs(exp), Float32(1.0))
        if err / ref_val > rel_tol:
            errors += 1
            if errors <= max_errors_to_print:
                print(
                    "  ERROR at idx",
                    idx,
                    ": expected",
                    exp,
                    "got",
                    act,
                    "err",
                    err,
                )
        if err > max_err:
            max_err = err

    return MatmulVerifyResult(
        errors=errors,
        max_err=max_err,
        mean_err=sum_err / Float32(size),
    )


def print_verify_result(result: MatmulVerifyResult, size: Int):
    """Print verification summary and exit status.

    Args:
        result: The verification result from `verify_matmul_f16`.
        size: Total number of elements compared.
    """
    print("  Max absolute error:", result.max_err)
    print("  Mean absolute error:", result.mean_err)

    if result.passed():
        print("  SUCCESS: All", size, "elements verified!")
    else:
        print("  FAILED:", result.errors, "errors out of", size)
