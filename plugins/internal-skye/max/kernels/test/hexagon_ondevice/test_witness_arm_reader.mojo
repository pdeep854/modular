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
"""Tests the graph stub's witness readers against what each ARM leaves behind.

A witness word is only a fact on the arm that writes it, and the two arms of
`mojo_matmul_f16` disagree about word 22: the HMX arm echoes the crouton unpack
seam there, and the HVX nest -- which every M=1 matmul takes, because
`matmul_bt_hmx_accepts` refuses M=1 -- writes its per-thread cycle bracket over
words 8..31 and so lands a utimer tick in the same word. The reader's guard is
therefore not "is this code known" but "is this word mine to read", and only
slot 7 answers that.

The tick values below are not invented. They are the readings order 149 leg B
took off word 22 on the 24 M=1 matmul dispatches of a production render, in the
flag-ON and the flag-OFF build alike -- so a reader that validates them would
kill every render the moment a regenerated MEF carried it, on the shipping
default build and not merely under an experimental flag.
"""

from std.testing import (
    assert_equal,
    assert_not_equal,
    assert_raises,
    TestSuite,
)

from hexagon_ondevice.dispatch import (
    NPU_MATMUL_PATH_HMX_RAN,
    NPU_MATMUL_PATH_UNSET,
    NPU_UNPACK_ARM_NATIVE,
    NPU_UNPACK_ARM_PAIRED,
    NPU_UNPACK_ARM_SCALAR_OFF,
    NPU_UNPACK_ARM_SCALAR_REFUSED,
    NPU_UNPACK_ARM_UNSET,
    NPU_UNPACK_ARM_VECTOR,
)
from hexagon_ondevice.hmx_crouton import (
    HMX_UNPACK_ARM_NATIVE,
    HMX_UNPACK_ARM_PAIRED,
    HMX_UNPACK_ARM_SCALAR_OFF,
    HMX_UNPACK_ARM_SCALAR_REFUSED,
    HMX_UNPACK_ARM_VECTOR,
)
from hexagon_ondevice.registrations import _require_unpack_arm_known


def _nest_ticks() -> List[UInt32]:
    """Word 22 as the HVX nest leaves it: thread 2's utimer bracket, low half.

    Read on device across both arms of order 149 leg B — 1965..77873 with the
    vector arm on, 2058..77904 with it off — plus one neighbouring thread's tick
    from the same nest, so the range shows this is a clock and not a code space.
    """
    return [
        UInt32(1865),
        UInt32(1968),
        UInt32(2058),
        UInt32(11419),
        UInt32(22753),
        UInt32(34301),
        UInt32(68546),
        UInt32(77904),
    ]


def test_writer_and_reader_spell_the_same_code_space() raises:
    """The two statements of one code space agree, code for code.

    `HMX_UNPACK_ARM_*` (`hmx_crouton.mojo`, the device writer) and
    `NPU_UNPACK_ARM_*` (`dispatch.mojo`, the host reader) are independent
    spellings of the same numbers, and nothing bound them until this. That is the
    `hmx_stripe_arena_bytes` / `hmx_stripe_act_slots` drift class exactly: a code
    renumbered on one side does not fail loudly -- the reader ACCEPTS the wrong
    arm and every A/B afterwards scores the arm it did not select, which is the
    failure #393 spent a window on. Adding a fourth code doubles the surface, so
    the binding lands with it rather than after it.
    """
    assert_equal(HMX_UNPACK_ARM_SCALAR_OFF, NPU_UNPACK_ARM_SCALAR_OFF)
    assert_equal(HMX_UNPACK_ARM_VECTOR, NPU_UNPACK_ARM_VECTOR)
    assert_equal(HMX_UNPACK_ARM_SCALAR_REFUSED, NPU_UNPACK_ARM_SCALAR_REFUSED)
    assert_equal(HMX_UNPACK_ARM_PAIRED, NPU_UNPACK_ARM_PAIRED)
    assert_equal(HMX_UNPACK_ARM_NATIVE, NPU_UNPACK_ARM_NATIVE)


def test_the_four_arm_codes_are_distinct() raises:
    """No two arms share a code, which is the only property PAIRED has today.

    `SCALAR_OFF` vs `SCALAR_REFUSED` is the pair that matters most -- both mean
    the scalar arm ran and only the second means a build that asked for the
    vector arm measured nothing -- but a renumbering collides any pair silently,
    so all six are checked rather than the one with a story.
    """
    var codes = [
        NPU_UNPACK_ARM_SCALAR_OFF,
        NPU_UNPACK_ARM_VECTOR,
        NPU_UNPACK_ARM_SCALAR_REFUSED,
        NPU_UNPACK_ARM_PAIRED,
        NPU_UNPACK_ARM_NATIVE,
    ]
    for i in range(len(codes)):
        assert_not_equal(codes[i], NPU_UNPACK_ARM_UNSET)
        for j in range(i + 1, len(codes)):
            assert_not_equal(codes[i], codes[j])


def test_hmx_arm_accepts_every_code_the_seam_reports() raises:
    for arm in [
        NPU_UNPACK_ARM_UNSET,
        NPU_UNPACK_ARM_SCALAR_OFF,
        NPU_UNPACK_ARM_VECTOR,
        NPU_UNPACK_ARM_SCALAR_REFUSED,
        NPU_UNPACK_ARM_PAIRED,
        NPU_UNPACK_ARM_NATIVE,
    ]:
        _require_unpack_arm_known(NPU_MATMUL_PATH_HMX_RAN, arm, "mo.matmul")


def test_hmx_arm_refuses_a_code_from_a_newer_skel() raises:
    """The loudness #393 built, kept: an unnamed code on the arm that writes it.

    This is what stops the vector-nest case below from passing because the
    reader stopped reading.
    """
    with assert_raises(contains="an arm code this host does not know"):
        _require_unpack_arm_known(
            NPU_MATMUL_PATH_HMX_RAN, UInt32(6), "mo.matmul"
        )


def test_vector_nest_ticks_are_not_arm_codes() raises:
    """THE REGRESSION. M=1 goes to the nest, and the nest owns word 22.

    Slot 7 stays at the stub's seed on that arm -- the kernel writes it only on
    the HMX arm -- so the seed is the discriminator, and the word is simply not
    the seam's to answer for here.
    """
    for tick in _nest_ticks():
        _require_unpack_arm_known(NPU_MATMUL_PATH_UNSET, tick, "mo.matmul")


def test_vector_nest_seed_survives_nothing_and_needs_to_survive_nothing() raises:
    """The seed cannot carry the answer: the nest overwrites word 22 too.

    Stated as its own case because it is the reason the fix is a slot-7 gate and
    not a better sentinel -- there is no value the stub can write into 22 that a
    vector-nest dispatch will not clobber.
    """
    _require_unpack_arm_known(
        NPU_MATMUL_PATH_UNSET, NPU_UNPACK_ARM_UNSET, "mo.batch_matmul"
    )
    _require_unpack_arm_known(
        NPU_MATMUL_PATH_UNSET, NPU_UNPACK_ARM_VECTOR, "mo.batch_matmul"
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
