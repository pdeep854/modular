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
"""NPU implementation of reduction operations."""

from std.math import align_down
from std.sys.info import simd_width_of

from std.utils.index import IndexList, StaticTuple

# ===-----------------------------------------------------------------------===#
# NPU reduce implementation
# ===-----------------------------------------------------------------------===#


@always_inline
def _reduce_generator_npu[
    num_reductions: Int,
    init_type: DType,
    input_0_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank]
    ) capturing[_] -> SIMD[dtype, width],
    output_0_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank], StaticTuple[SIMD[dtype, width], num_reductions]
    ) capturing[_] -> None,
    reduce_function: def[ty: DType, width: Int, reduction_idx: Int](
        SIMD[ty, width], SIMD[ty, width]
    ) capturing[_] -> SIMD[ty, width],
](
    shape: IndexList[_, element_type=DType.int64],
    init: StaticTuple[Scalar[init_type], num_reductions],
    reduce_dim: Int,
):
    """Reduce on NPU using sequential SIMD accumulate (no warp shuffles).

    NPU reduction strategy: Load data in HVX-width SIMD chunks, accumulate
    into SIMD registers, reduce the final SIMD vector to scalar, then handle
    scalar remainder for unaligned tails. Single-core, rank-1 only.

    Parameters:
        num_reductions: The number of fused reductions to perform.
        init_type: The initial accumulator value type.
        input_0_fn: The lambda to access the input tensor.
        output_0_fn: The lambda to store to the output tensor.
        reduce_function: The lambda implementing the reduction.

    Args:
        shape: The shape of the tensor we are reducing.
        init: The value to start the reduction from.
        reduce_dim: The dimension we are reducing.
    """
    comptime rank = shape.size
    comptime assert rank == 1, "NPU reduction currently supports rank 1 only"

    var reduce_size = Int(shape[reduce_dim])

    # SIMD accumulate loop
    comptime simd_width = simd_width_of[init_type]()
    var simd_compatible_size = align_down(reduce_size, simd_width)

    # Initialize SIMD accumulators
    var accum = StaticTuple[SIMD[init_type, simd_width], num_reductions]()
    comptime for i in range(num_reductions):
        accum[i] = init[i]

    # Main SIMD accumulate loop
    for offset in range(0, simd_compatible_size, simd_width):
        var idx = IndexList[rank](offset)
        var val = input_0_fn[init_type, simd_width, rank](idx)
        comptime for i in range(num_reductions):
            accum[i] = reduce_function[init_type, simd_width, i](accum[i], val)

    # Reduce SIMD vectors to scalars
    var scalar_accum = StaticTuple[Scalar[init_type], num_reductions]()
    comptime for i in range(num_reductions):
        scalar_accum[i] = accum[i].reduce[
            reduce_function[init_type, reduction_idx=i, ...], 1
        ]()

    # Scalar remainder loop for unaligned tail
    for offset in range(simd_compatible_size, reduce_size):
        var idx = IndexList[rank](offset)
        var val = input_0_fn[init_type, 1, rank](idx)
        comptime for i in range(num_reductions):
            scalar_accum[i] = reduce_function[init_type, 1, i](
                scalar_accum[i], val
            )

    # Write output
    var output_val = StaticTuple[SIMD[init_type, 1], num_reductions]()
    comptime for i in range(num_reductions):
        output_val[i] = scalar_accum[i]
    output_0_fn[init_type, 1, rank](IndexList[rank](0), output_val)
