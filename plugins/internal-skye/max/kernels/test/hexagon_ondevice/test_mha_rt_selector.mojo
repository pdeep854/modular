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
"""Tests the `rt` arm selectors the two attention dispatches send.

Order 121 rendered with `HMX_ATTN=1` and measured the vector path anyway: the
array arm was correct over all 24 heads and 3.9x faster on the standalone
driver, and the graph could not reach it because this dispatch passed a literal
`Int32(0)` while `MODULAR_NPU_MHA_RT` reached only the driver. The third
print-vs-dispatch instance of the arc, after #349 dropped the driver's own read
and #350 restored it.

The first two tests are the ones that matter for every render that does not ask
for an arm. Since the default wave an unset variable selects the PRODUCTION arm
-- `MHA_RT_TEAM`, order 123's -- rather than the vector path, because the arm an
operator had to remember is the arm an operator forgets. Its partner test is
just as load-bearing: an explicit `0` must still reach the vector path, or the
control arm every A/B compares against would have quietly stopped existing. The
rest pin the arms the kernel defines and the refusal of everything else -- a
value the kernel has no arm for must fail the dispatch rather than fall back to
the vector path, which would report as an arm that ran and be exactly the class
this change closes.

`mo.composite.masked_flash_attention_gpu` is the encoder's, and it carried the
same literal until its array arm existed. Its selector is STRICTER rather than a
copy: the masked entry defines only the vector arm and `MHA_RT_TEAM`, so the
dense entry's `MHA_RT_DARK` and `MHA_RT_FLASH` are refused here instead of
reaching a kernel with no branch for them. It also reads its own variable, so
moving the render's arm cannot move the encoder's.
"""

from std.testing import assert_equal, assert_raises, assert_true, TestSuite

from hexagon_ondevice.npu_kernel import MHA_RT_DARK, MHA_RT_FLASH, MHA_RT_TEAM
from hexagon_ondevice.registrations import (
    MHA_MASKED_RT_ENV,
    MHA_MASKED_RT_TEAM,
    MHA_RT_ARM_MAX,
    MHA_RT_DEFAULT,
    MHA_RT_ENV,
    _mha_masked_rt_selector,
    _mha_rt_selector,
)


def test_unset_selects_the_production_arm() raises:
    """The default-wave gate: no variable, and the launch carries the arm the
    render has actually run since order 123 instead of the arm an operator had
    to remember to ask for."""
    assert_equal(_mha_rt_selector(""), Int32(MHA_RT_DEFAULT))
    assert_equal(Int32(MHA_RT_DEFAULT), Int32(MHA_RT_TEAM))


def test_explicit_zero_still_selects_the_vector_arm() raises:
    """The other half of the default flip, and the half that is easy to lose:
    every A/B in this campaign compares against the vector arm, so `=0` has to
    keep reaching it."""
    assert_equal(_mha_rt_selector("0"), Int32(0))


def test_each_arm_reaches_the_launch() raises:
    assert_equal(_mha_rt_selector("0"), Int32(0))
    assert_equal(_mha_rt_selector("1"), Int32(1))
    assert_equal(_mha_rt_selector("2"), Int32(2))
    assert_equal(_mha_rt_selector("3"), Int32(3))


def test_arm_above_the_last_refused() raises:
    with assert_raises():
        _ = _mha_rt_selector("4")


def test_negative_arm_refused() raises:
    with assert_raises():
        _ = _mha_rt_selector("-1")


def test_unparsable_spelling_refused() raises:
    """The driver's `strtol` yields 0 here; a typo must not render the vector
    path while the operator believes the array ran."""
    with assert_raises():
        _ = _mha_rt_selector("team")


def test_bound_tracks_the_kernel() raises:
    """The host layer spells its own bound because `npu_kernel.mojo` is the
    device compile's entry, not a package member the graph layer may import.
    This test is the only place the two copies meet."""
    assert_equal(MHA_RT_ARM_MAX, MHA_RT_TEAM)


def test_env_name_is_the_drivers() raises:
    """One binary and one variable: `npu_attention_dispatch.cpp` reads this
    exact name, so a rename here would silently split the two callers."""
    assert_equal(String(MHA_RT_ENV), String("MODULAR_NPU_MHA_RT"))


def test_masked_unset_selects_the_production_arm() raises:
    """The default-wave gate for the encoder's dispatch: order 140 collected
    the masked array arm in the render (pair bitwise identical, Tier B 52.778
    dB, 606.8 -> 75.0 ms per masked dispatch), and an unset variable now
    selects it.

    Its own variable still, so this moves independently of the dense entry's.
    """
    assert_equal(_mha_masked_rt_selector(""), Int32(MHA_MASKED_RT_TEAM))


def test_masked_explicit_zero_still_selects_the_vector_arm() raises:
    """The encoder's control arm, kept reachable by name."""
    assert_equal(_mha_masked_rt_selector("0"), Int32(0))


def test_masked_arms_reach_the_launch() raises:
    assert_equal(_mha_masked_rt_selector("0"), Int32(0))
    assert_equal(_mha_masked_rt_selector("3"), Int32(MHA_MASKED_RT_TEAM))


def test_masked_refuses_the_dense_only_arms() raises:
    """THE control this selector exists for.

    `MHA_RT_DARK` and `MHA_RT_FLASH` are arms of the dense entry. The masked
    entry has no branch for either, so passing one through would run the vector
    arm under a selector the operator believes engaged the array -- the
    print-vs-dispatch class, one level down from the literal that used to sit
    here. Both must fail the dispatch instead.
    """
    with assert_raises():
        _ = _mha_masked_rt_selector(String(MHA_RT_DARK))
    with assert_raises():
        _ = _mha_masked_rt_selector(String(MHA_RT_FLASH))


def test_masked_out_of_range_and_unparsable_refused() raises:
    with assert_raises():
        _ = _mha_masked_rt_selector("4")
    with assert_raises():
        _ = _mha_masked_rt_selector("-1")
    with assert_raises():
        _ = _mha_masked_rt_selector("team")


def test_masked_bound_tracks_the_kernel() raises:
    """The host layer's copy of the masked entry's array arm, against the
    kernel's own constant. The only place the two meet."""
    assert_equal(MHA_MASKED_RT_TEAM, MHA_RT_TEAM)


def test_masked_env_name_is_its_own() raises:
    """Separate from the dense entry's, so a dense diagnostic cannot refuse or
    silently re-arm the encoder's dispatch."""
    assert_equal(String(MHA_MASKED_RT_ENV), String("MODULAR_NPU_MHA_MASKED_RT"))
    assert_true(String(MHA_MASKED_RT_ENV) != String(MHA_RT_ENV))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
