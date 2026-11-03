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
"""NPU implementation of stencil computation."""

# ===-----------------------------------------------------------------------===#
# stencil
# ===-----------------------------------------------------------------------===#


def _stencil_1d_impl_npu[
    compute_fn: def(Float32, Float32, Float32) capturing[_] -> Float32
](
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    size: Int,
):
    """Computes a 1D stencil on NPU using a 3-point window.

    Applies ``compute_fn`` to each interior element using its left, center,
    and right neighbors.  Boundary elements (first and last) are copied
    directly from ``input`` to ``output``.

    Parameters:
        compute_fn: A function that takes left, center, right neighbors
            and returns the stencil result.

    Args:
        input: Pointer to the input buffer (``size`` elements).
        output: Pointer to the output buffer (``size`` elements).
        size: Number of elements.
    """
    # Copy boundaries.
    output[0] = input[0]
    output[size - 1] = input[size - 1]

    # Interior: apply 3-point stencil.
    for i in range(1, size - 1):
        output[i] = compute_fn(input[i - 1], input[i], input[i + 1])


def _stencil_2d_impl_npu[
    compute_fn: def(
        UnsafePointer[Float32, MutUnsafeAnyOrigin], Int, Int, Int
    ) capturing[_] -> Float32
](
    input: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    rows: Int,
    cols: Int,
):
    """Computes a 2D stencil on NPU over a row-major grid.

    Applies ``compute_fn`` to each interior element, passing the full input
    buffer along with the current row, column, and column stride.  Border
    elements are copied directly from ``input`` to ``output``.

    Parameters:
        compute_fn: A function that takes ``(input_ptr, row, col, cols)``
            and returns the stencil result.

    Args:
        input: Pointer to the input buffer (``rows * cols`` elements,
            row-major).
        output: Pointer to the output buffer (``rows * cols`` elements,
            row-major).
        rows: Number of rows.
        cols: Number of columns.
    """
    # Copy border rows.
    for c in range(cols):
        output[c] = input[c]
        output[(rows - 1) * cols + c] = input[(rows - 1) * cols + c]

    # Copy border columns.
    for r in range(1, rows - 1):
        output[r * cols] = input[r * cols]
        output[r * cols + cols - 1] = input[r * cols + cols - 1]

    # Interior: apply stencil via compute_fn.
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            output[r * cols + c] = compute_fn(input, r, c, cols)
