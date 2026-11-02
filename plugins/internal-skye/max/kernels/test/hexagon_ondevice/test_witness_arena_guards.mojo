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
"""Pins the guard a stub ARMS to the guard this file's readers DESCRIBE.

The evaluation moved to the transport when the guards were deferred to the
flush, and the transport is generic: it asks "is this word one of these
values", knowing none of the codes. That is what keeps `MATMUL_PATH_*` and
`HMX_UNPACK_ARM_*` on this side, where they belong -- and it is also a new way
to be wrong, because the accept list a stub writes and the predicate
`_require_matmul_ran` states are now two expressions of one contract.

So the readers are not dead code: they are this test's oracle. Every probe
below must be accepted by the armed list exactly when the reader does not
raise on it. A code added to one and not the other fails here rather than on
device.
"""

from std.memory.alloc import alloc
from std.os import getenv
from std.testing import assert_equal, assert_raises, TestSuite

from hexagon_ondevice.dispatch import (
    NPU_ARENA_SLOT_BYTES,
    NPU_GUARD_ACCEPTS,
    NPU_GUARD_ACCEPT_COUNT,
    NPU_GUARD_CLASS,
    NPU_GUARD_CLASS_ATTRIBUTION,
    NPU_GUARD_CLASS_CORRECTNESS,
    NPU_GUARD_COND_NONE,
    NPU_GUARD_COND_VALUE,
    NPU_GUARD_COND_WORD,
    NPU_GUARD_HDR_COUNT,
    NPU_GUARD_HDR_MAGIC,
    NPU_GUARD_HDR_WORDS,
    NPU_GUARD_MAGIC,
    NPU_GUARD_WORD,
    NPU_GUARD_WORDS,
    NPU_MATMUL_PATH_HMX_RAN,
    NPU_MATMUL_PATH_UNSET,
    NPU_UNPACK_ARM_PAIRED,
    NPU_UNPACK_ARM_SCALAR_OFF,
    NPU_UNPACK_ARM_SCALAR_REFUSED,
    NPU_UNPACK_ARM_UNSET,
    NPU_UNPACK_ARM_VECTOR,
    NPU_WITNESS_BYTES,
    NPU_WITNESS_MATMUL_PATH,
    NPU_WITNESS_UNPACK_ARM,
)
from hexagon_ondevice.registrations import (
    _arm_matmul_ran,
    _arm_unpack_arm_known,
    _refuse_batched_graph_path,
    _refuse_megastep_graph_path,
    _require_matmul_ran,
    _require_unpack_arm_known,
)

comptime _SlotPtr = UnsafePointer[UInt32, MutUnsafeAnyOrigin]


def _slot() -> _SlotPtr:
    """One zeroed arena slot in plain host memory: no device, no context."""
    var words = NPU_ARENA_SLOT_BYTES // 4
    var raw = alloc[UInt32](words)
    var slot = _SlotPtr(unsafe_from_address=Int(raw))
    for i in range(words):
        slot[i] = 0
    return slot


def _entry(slot: _SlotPtr, index: Int) -> _SlotPtr:
    """Guard `index`'s record, decoded the way the transport decodes it."""
    var record = slot + NPU_WITNESS_BYTES // 4
    return record + (NPU_GUARD_HDR_WORDS + index * NPU_GUARD_WORDS)


def _accepts(entry: _SlotPtr, value: UInt32) -> Bool:
    var count = Int(entry[NPU_GUARD_ACCEPT_COUNT])
    for i in range(count):
        if entry[NPU_GUARD_ACCEPTS + i] == value:
            return True
    return False


def _probes() -> List[UInt32]:
    """Codes worth asking about: the seed, the pass, every refusal the kernel
    can write, a code from a newer skel, and two of the utimer ticks the HVX
    nest leaves in word 22 (order 149 leg B)."""
    return [
        UInt32(0),
        NPU_MATMUL_PATH_HMX_RAN,
        UInt32(1),
        UInt32(2),
        UInt32(3),
        UInt32(4),
        UInt32(5),
        UInt32(6),
        UInt32(7),
        UInt32(99),
        UInt32(1965),
        UInt32(77873),
        UInt32(0xFFFFFFFF),
    ]


def test_armed_record_is_well_formed() raises:
    var slot = _slot()
    _arm_matmul_ran(slot, "mo.matmul")
    _arm_unpack_arm_known(slot, "mo.matmul")

    var record = slot + NPU_WITNESS_BYTES // 4
    assert_equal(record[NPU_GUARD_HDR_MAGIC], NPU_GUARD_MAGIC)
    assert_equal(Int(record[NPU_GUARD_HDR_COUNT]), 2)

    var ran = _entry(slot, 0)
    assert_equal(ran[NPU_GUARD_CLASS], NPU_GUARD_CLASS_CORRECTNESS)
    assert_equal(Int(ran[NPU_GUARD_WORD]), NPU_WITNESS_MATMUL_PATH)
    assert_equal(Int(ran[NPU_GUARD_COND_WORD]), NPU_GUARD_COND_NONE)

    var arm = _entry(slot, 1)
    assert_equal(arm[NPU_GUARD_CLASS], NPU_GUARD_CLASS_ATTRIBUTION)
    assert_equal(Int(arm[NPU_GUARD_WORD]), NPU_WITNESS_UNPACK_ARM)
    # SLOT 22 IS THE SEAM'S ONLY ON THE HMX ARM. Off it the HVX nest owns words
    # 8..31 and leaves a tick here, so an unconditional guard would refuse every
    # M=1 matmul of the shipping default build.
    assert_equal(Int(arm[NPU_GUARD_COND_WORD]), NPU_WITNESS_MATMUL_PATH)
    assert_equal(arm[NPU_GUARD_COND_VALUE], NPU_MATMUL_PATH_HMX_RAN)


def test_matmul_accept_list_is_the_reader() raises:
    var slot = _slot()
    _arm_matmul_ran(slot, "mo.matmul")
    var entry = _entry(slot, 0)
    for probe in _probes():
        var accepted = _accepts(entry, probe)
        var raised = False
        try:
            _require_matmul_ran(probe, "mo.matmul")
        except:
            raised = True
        assert_equal(
            accepted,
            not raised,
            String(
                "the armed accept list and `_require_matmul_ran` disagree"
                " about code "
            )
            + String(probe),
        )


def test_unpack_arm_accept_list_is_the_reader() raises:
    var slot = _slot()
    _arm_unpack_arm_known(slot, "mo.matmul")
    var entry = _entry(slot, 0)
    for probe in _probes():
        var accepted = _accepts(entry, probe)
        var raised = False
        try:
            _require_unpack_arm_known(NPU_MATMUL_PATH_HMX_RAN, probe, "mo.x")
        except:
            raised = True
        assert_equal(
            accepted,
            not raised,
            String(
                "the armed accept list and `_require_unpack_arm_known`"
                " disagree about arm code "
            )
            + String(probe),
        )


def test_the_seed_is_the_clear() raises:
    """Both `UNSET` codes are zero, so clearing a reused slot seeds it.

    `_witness_slot` relies on this instead of writing the seeds a second time,
    and a nonzero seed would make a stale slot read as a refusal.
    """
    assert_equal(Int(NPU_MATMUL_PATH_UNSET), 0)
    assert_equal(Int(NPU_UNPACK_ARM_UNSET), 0)


def test_every_arm_code_is_accepted() raises:
    """The seam's whole code space passes the armed list.

    The reader's own test states why this matters more than it looks: a writer
    that ships ahead of its reader takes down the first matmul of the first
    render, so the accept list has to cross first.
    """
    var slot = _slot()
    _arm_unpack_arm_known(slot, "mo.matmul")
    var entry = _entry(slot, 0)
    for code in [
        NPU_UNPACK_ARM_UNSET,
        NPU_UNPACK_ARM_SCALAR_OFF,
        NPU_UNPACK_ARM_VECTOR,
        NPU_UNPACK_ARM_SCALAR_REFUSED,
        NPU_UNPACK_ARM_PAIRED,
    ]:
        assert_equal(_accepts(entry, code), True)


def test_batching_is_refused_on_the_graph_path() raises:
    """A0: the corrupting mode is refused where the graph path knows itself.

    `MODULAR_NPU_BATCH=1` reorders nothing and breaks nothing for a launch
    sequence with no host reader in it -- which is where it was developed and
    measured. This file is not that sequence, and a known-corrupting mode
    reachable by env var is the one that silently becomes a baseline.
    """
    with assert_raises(contains="MODULAR_NPU_BATCH=1 is refused"):
        _refuse_batched_graph_path(True)


def test_the_refusal_names_the_mechanism_not_just_the_flag() raises:
    """An error a reader cannot act on gets worked around, not fixed.

    The message has to carry why, or the next person sets the flag again from a
    different shell. These are the load-bearing halves: the class of op that
    reads early, the fact that nothing reports a failure, and where the
    evidence is.
    """
    for fragment in [
        String("host"),
        String("mo.concat"),
        String("witness guard"),
        String("186A"),
        String("#447"),
    ]:
        with assert_raises(contains=fragment):
            _refuse_batched_graph_path(True)


def test_the_refusal_is_silent_when_the_flag_is_not_set() raises:
    """The control. A refusal that fires unconditionally is not a refusal.

    Read the way the transport reads it (`npuBatchRequested`): the value is
    exactly "1", so an unset flag, "0", and "10" all leave the graph alone.
    """
    _refuse_batched_graph_path(False)
    _refuse_batched_graph_path(
        getenv("MODULAR_NPU_BATCH_UNSET_IN_TESTS") == "1"
    )


def test_megastep_is_refused_on_the_graph_path() raises:
    """The second holding transport refuses where the first one does.

    `MODULAR_NPU_MEGASTEP=1` holds a whole step's launches, across memAlloc
    and memFree deliberately, so every host reader the batch refusal names
    reads pre-launch bytes for longer. Same class (186A), same chokepoint,
    same loudness.
    """
    with assert_raises(contains="MODULAR_NPU_MEGASTEP=1 is refused"):
        _refuse_megastep_graph_path(True)


def test_the_megastep_refusal_names_the_mechanism_not_just_the_flag() raises:
    """The load-bearing halves, megastep edition.

    The op class that reads early, the silence, the evidence, the longer hold,
    and the guard whose landing lifts the refusal -- so nobody re-enables it
    from a different shell on the theory that the megastep is not the batch.
    """
    for fragment in [
        String("host"),
        String("mo.concat"),
        String("witness guard"),
        String("186A"),
        String("memAlloc"),
        String("constant-ranges"),
    ]:
        with assert_raises(contains=fragment):
            _refuse_megastep_graph_path(True)


def test_the_megastep_refusal_is_silent_when_the_flag_is_not_set() raises:
    """The control, read the way `npuMegastepRequested` reads it.

    The value is exactly "1", so an unset flag, "0", and "10" all leave the
    graph alone.
    """
    _refuse_megastep_graph_path(False)
    _refuse_megastep_graph_path(
        getenv("MODULAR_NPU_MEGASTEP_UNSET_IN_TESTS") == "1"
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
