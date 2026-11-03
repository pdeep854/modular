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
"""Simple xorshift32 PRNG for generating random test inputs.

Seeded from perf_counter_ns so each run produces different values,
but the seed is printed for reproducibility.
"""

from std.time import perf_counter_ns
from std.memory import UnsafePointer


@fieldwise_init
struct Xorshift32:
    """Xorshift32 PRNG with [0, 1) float output."""

    var state: UInt32
    """Current state of the PRNG. Must be nonzero."""

    @staticmethod
    def from_time() -> Xorshift32:
        """Create a PRNG seeded from the current time.

        Returns:
            A Xorshift32 instance seeded with the current time.
        """

        var s = UInt32(perf_counter_ns() & 0xFFFFFFFF)
        if s == 0:
            s = 1
        return Xorshift32(state=s)

    def seed(self) -> UInt32:
        """Return the current seed (for printing/reproducibility).

        Returns:
            The current state of the PRNG, which serves as the seed.
        """
        return self.state

    def next_u32(mut self) -> UInt32:
        """Generate the next 32-bit pseudorandom value.

        Returns:
            The next pseudorandom UInt32 value.
        """
        self.state ^= self.state << 13
        self.state ^= self.state >> 17
        self.state ^= self.state << 5
        return self.state

    def next_float(mut self) -> Float64:
        """Generate a pseudorandom Float64 in [0.0, 1.0).

        Returns:
            A pseudorandom Float64 value in the range [0.0, 1.0).
        """
        return Float64(self.next_u32() & 0xFFFF) / 65536.0

    def fill_fp16(
        mut self,
        ptr: UnsafePointer[Float16, MutUntrackedOrigin],
        count: Int,
        scale: Float64 = 1.0,
    ):
        """Fill a Float16 buffer with random values in [0, scale).

        Args:
            ptr: Pointer to the Float16 buffer to fill.
            count: Number of elements to fill.
            scale: Maximum value for the random numbers (default 1.0).
        """
        for i in range(count):
            ptr[i] = Float16(self.next_float() * scale)
