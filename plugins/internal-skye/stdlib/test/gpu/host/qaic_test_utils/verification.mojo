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
"""Verification helpers for QAIC test output buffers."""

from std.memory import UnsafePointer
from std.testing import assert_almost_equal


def verify_buffer_almost_equal(
    output: UnsafePointer[Float32, MutUntrackedOrigin],
    expected: UnsafePointer[Float32, MutUntrackedOrigin],
    n: Int,
    atol: Float64 = 0.001,
) raises:
    """Verifies that every element in `output` matches `expected` within tolerance.

    Uses `assert_almost_equal` per element, which fails fast on the first
    mismatch.

    Args:
        output: Pointer to actual output buffer.
        expected: Pointer to expected values buffer.
        n: Number of elements to compare.
        atol: Absolute tolerance for comparison (default 0.001).

    Raises:
        If any element pair differs by more than `atol`.
    """
    for i in range(n):
        assert_almost_equal(output[i], expected[i], atol=atol)


def verify_scalar_almost_equal(
    output: UnsafePointer[Float32, MutUntrackedOrigin],
    expected: Float32,
    atol: Float64 = 0.001,
) raises:
    """Verifies that ``output[0]`` matches `expected` within tolerance.

    Convenience wrapper for reduction kernels that produce a single
    scalar result.

    Args:
        output: Pointer to a 1-element output buffer.
        expected: The expected scalar value.
        atol: Absolute tolerance for comparison (default 0.001).

    Raises:
        If ``output[0]`` differs from `expected` by more than `atol`.
    """
    assert_almost_equal(output[0], expected, atol=atol)
