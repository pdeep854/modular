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
"""Wrapper functions for functions that are implemented in
   external bitcode from IntrinsicsHVX.c in the Hexagon SDK.
"""

from std.memory import bitcast


@extern("_hexagon_runtime_exp__vf")
def _hexagon_runtime_exp__vf(
    vin: SIMD[DType.int32, 32]
) abi("C") -> SIMD[DType.int32, 32]:
    ...


@extern("_hexagon_runtime_exp__vhf")
def _hexagon_runtime_exp__vhf(
    vin: SIMD[DType.int32, 32]
) abi("C") -> SIMD[DType.int32, 32]:
    ...


@extern("_hexagon_runtime_tanh__vf")
def _hexagon_runtime_tanh__vf(
    vin: SIMD[DType.int32, 32]
) abi("C") -> SIMD[DType.int32, 32]:
    ...


@extern("_hexagon_runtime_tanh__vhf")
def _hexagon_runtime_tanh__vhf(
    vin: SIMD[DType.int32, 32]
) abi("C") -> SIMD[DType.int32, 32]:
    ...


@always_inline
def hexagon_runtime_exp__vf[
    dtype: DType,
    width: Int,
](vin: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Calculates elementwise exponential of the input vector.

    Using function from IntrinsicsHVX.c in the Hexagon SDK.

    Given an input vector $X$ and an output vector $Y$, sets $Y_i = e^{X_i}$ for
    each position $i$ in the input vector (where $e$ is the mathematical constant
    $e$).

    Constraints:
        The input must be a floating-point type.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        vin: The input SIMD vector.

    Returns:
        A SIMD vector containing $e$ raised to the power $X_i$ where $X_i$ is an
        element in the input SIMD vector.

    """

    comptime assert (dtype == DType.float32 and width == 32) or (
        dtype == DType.float16 and width == 64
    ), (
        "hexagon_runtime_exp__vf only supports SIMD vectors of DType.float32"
        " width 32 or DType.float16 width 64"
    )

    comptime if dtype == DType.float32:
        return bitcast[
            src_dtype=DType.int32, src_width=32, dtype=dtype, width=width
        ](
            _hexagon_runtime_exp__vf(
                bitcast[
                    src_dtype=dtype,
                    src_width=width,
                    dtype=DType.int32,
                    width=32,
                ](vin)
            )
        )
    return bitcast[
        src_dtype=DType.int32, src_width=32, dtype=dtype, width=width
    ](
        _hexagon_runtime_exp__vhf(
            bitcast[
                src_dtype=dtype, src_width=width, dtype=DType.int32, width=32
            ](vin)
        )
    )


@always_inline
def hexagon_runtime_tanh__vf[
    dtype: DType,
    width: Int,
](vin: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Calculates elementwise hyperbolic tangent of the input vector.

    Using functions from IntrinsicsHVX.c in the Hexagon SDK.

    Given an input vector $X$ and an output vector $Y$, sets $Y_i = tanh(X_i)$
    for each position $i$ in the input vector.

    Constraints:
        The input must be float32 with width 32 or float16 with width 64.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        vin: The input SIMD vector.

    Returns:
        A SIMD vector containing `tanh(X_i)` where `X_i` is an element in the
        input SIMD vector.
    """

    comptime assert (dtype == DType.float32 and width == 32) or (
        dtype == DType.float16 and width == 64
    ), (
        "hexagon_runtime_tanh__vf only supports SIMD vectors of DType.float32"
        " width 32 or DType.float16 width 64"
    )

    comptime if dtype == DType.float32:
        return bitcast[
            src_dtype=DType.int32, src_width=32, dtype=dtype, width=width
        ](
            _hexagon_runtime_tanh__vf(
                bitcast[
                    src_dtype=dtype,
                    src_width=width,
                    dtype=DType.int32,
                    width=32,
                ](vin)
            )
        )
    return bitcast[
        src_dtype=DType.int32, src_width=32, dtype=dtype, width=width
    ](
        _hexagon_runtime_tanh__vhf(
            bitcast[
                src_dtype=dtype, src_width=width, dtype=DType.int32, width=32
            ](vin)
        )
    )
