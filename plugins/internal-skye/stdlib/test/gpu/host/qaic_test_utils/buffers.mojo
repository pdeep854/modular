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
"""Host buffer allocation helpers for QAIC tests."""

from std.memory import UnsafePointer, alloc


def alloc_sequential[
    dtype: DType
](n: Int, start: Int = 1) -> UnsafePointer[Scalar[dtype], MutUntrackedOrigin]:
    """Allocates a host buffer filled with sequential values.

    Creates a buffer of `n` elements where element ``i`` has value
    ``start + i``.  The caller is responsible for freeing the returned
    pointer.

    Parameters:
        dtype: The element data type.

    Args:
        n: Number of elements to allocate.
        start: First value in the sequence (default 1).

    Returns:
        A pointer to the allocated buffer.
    """
    var ptr = alloc[Scalar[dtype]](n)
    for i in range(n):
        ptr[i] = Scalar[dtype](start + i)
    return ptr


def alloc_constant[
    dtype: DType
](n: Int, value: Scalar[dtype]) -> UnsafePointer[
    Scalar[dtype], MutUntrackedOrigin
]:
    """Allocates a host buffer filled with a constant value.

    The caller is responsible for freeing the returned pointer.

    Parameters:
        dtype: The element data type.

    Args:
        n: Number of elements to allocate.
        value: The constant value for every element.

    Returns:
        A pointer to the allocated buffer.
    """
    var ptr = alloc[Scalar[dtype]](n)
    for i in range(n):
        ptr[i] = value
    return ptr


def alloc_zeros[
    dtype: DType
](n: Int) -> UnsafePointer[Scalar[dtype], MutUntrackedOrigin]:
    """Allocates a host buffer filled with zeros.

    The caller is responsible for freeing the returned pointer.

    Parameters:
        dtype: The element data type.

    Args:
        n: Number of elements to allocate.

    Returns:
        A pointer to the zero-filled buffer.
    """
    var ptr = alloc[Scalar[dtype]](n)
    for i in range(n):
        ptr[i] = Scalar[dtype](0)
    return ptr
