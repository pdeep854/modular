"""The Mojo megastep walker against the C walker's recorded behaviour.

`npu_megastep_walk_hosttest.c` drives the C walk over a fixture set with a
recording sink and asserts the launch sequence against a checked-in golden.
This drives the SAME fixtures through `megastep_walk` and asserts the SAME
golden. Neither twin is the other's reference -- the golden is -- so a walker
that disagrees with the other disagrees with a file both are read against, and
either twin failing names which one moved.

What is compared, per fixture, is everything the walk decides: the launches it
presents in order, their selector, thread mask, argument bytes and every one of
the eight resolved operand bases and lengths, the session-table state a
registration leaves behind, and the terminal status. The walk never
dereferences an operand base, so the fixtures name allocations by synthetic
addresses and the tuples are comparable across two languages.

Run by bazel with both file paths in the environment; see the BUILD rule.
"""

from std.os import getenv
from std.memory import UnsafePointer
from std.testing import assert_equal, assert_true, TestSuite

from hexagon_ondevice.megastep_walker import (
    MEGASTEP_BUFFER_SLOTS,
    MEGASTEP_HEADER_BYTES,
    MEGASTEP_MODE_REGISTER,
    MEGASTEP_TABLE_ENTRIES,
    MegastepSeam,
    SlotBases,
    SlotLengths,
    TableBases,
    TableBytes,
    WireBytes,
    megastep_register,
    megastep_walk,
)

comptime TWIN_SLOT_BASE = 0x10000
comptime TWIN_BASE_STRIDE = 0x1000
"""Synthetic bases, matching `npu_megastep_walk_hosttest.c`."""


@always_inline
def _hex(value: Int, width: Int) -> String:
    """Lowercase hex, zero-padded to `width`, as C's `%0*x` writes it."""
    comptime DIGITS = "0123456789abcdef"
    var out = String("")
    var v = value
    while True:
        out = String(DIGITS[byte = v & 0xF]) + out
        v >>= 4
        if v == 0:
            break
    while out.byte_length() < width:
        out = "0" + out
    return out^


@always_inline
def _dec(value: Int) -> String:
    return String(value)


@always_inline
def _nibble(byte: UInt8) raises -> Int:
    """One lowercase hex digit's value."""
    var value = Int(byte)
    if value >= ord("0") and value <= ord("9"):
        return value - ord("0")
    if value >= ord("a") and value <= ord("f"):
        return value - ord("a") + 10
    raise Error("the fixture file carries a non-hex digit in a block")


struct RecordingSeam(MegastepSeam):
    """The walk's host seam, recording instead of launching.

    `buffer_count` and `scalar_count` restate the C twin's stand-in arity
    exactly -- buffers = selector % 4, scalars = selector % 3 -- rather than
    reading `NPUKernels.def`, because a fixture that exercises an arity SKEW
    needs an arity the real manifest does not have.
    """

    var fixture: String
    var launches: Int
    var log: String

    def __init__(out self, var fixture: String):
        self.fixture = fixture^
        self.launches = 0
        self.log = String("")

    def buffer_count(self, selector: UInt32) -> Int:
        return Int(selector % 4)

    def scalar_count(self, selector: UInt32) -> Int:
        return Int(selector % 3)

    def launch(
        mut self,
        selector: UInt32,
        thread_mask: UInt32,
        args: WireBytes,
        args_bytes: Int,
        buffers: SlotBases,
        lengths: SlotLengths,
    ) -> Int32:
        self.log += (
            self.fixture
            + " launch="
            + _dec(self.launches)
            + " sel="
            + _dec(Int(selector))
            + " mask=0x"
            + _hex(Int(thread_mask), 8)
            + " argsbytes="
            + _dec(args_bytes)
            + " args="
        )
        for i in range(args_bytes):
            self.log += _hex(Int(args[i]), 2)
        for i in range(MEGASTEP_BUFFER_SLOTS):
            self.log += (
                " b"
                + _dec(i)
                + "="
                + _hex(Int(buffers[i]), 1)
                + "/"
                + _dec(Int(lengths[i]))
            )
        self.log += "\n"
        self.launches += 1
        return 0


@fieldwise_init
struct Fixture(Copyable, Movable):
    var name: String
    var block: List[UInt8]
    var slot_lengths: List[Int32]
    """-1 marks a slot the call did not present."""


def _parse_fixtures(text: String) raises -> List[Fixture]:
    var out = List[Fixture]()
    for line in text.splitlines():
        var trimmed = String(line).strip()
        if trimmed.byte_length() == 0:
            continue
        var fields = trimmed.split(" ")
        # name, block_len, eight slot lengths, hex block.
        var name = String(fields[0])
        var block_len = Int(String(fields[1]))
        var slot_lengths = List[Int32]()
        for i in range(MEGASTEP_BUFFER_SLOTS):
            slot_lengths.append(Int32(Int(String(fields[2 + i]))))
        var hexed = fields[2 + MEGASTEP_BUFFER_SLOTS].as_bytes()
        var block = List[UInt8]()
        for i in range(block_len):
            block.append(
                UInt8(
                    _nibble(hexed[2 * i]) * 16 + _nibble(hexed[2 * i + 1])
                )
            )
        out.append(Fixture(name^, block^, slot_lengths^))
    return out^


def _walk_fixtures(fixtures: List[Fixture]) raises -> String:
    """Runs every fixture in order against one session table, as the C twin
    does, and returns the recorded log."""
    # The session table, as the two parallel arrays the C side owns on a
    # device. Lists rather than `Array` because 512 entries twice is more stack
    # than a test frame should carry.
    var table_base_list = List[UInt]()
    var table_bytes_list = List[UInt32]()
    for _ in range(Int(MEGASTEP_TABLE_ENTRIES)):
        table_base_list.append(0)
        table_bytes_list.append(0)
    var table_bases = TableBases(
        unsafe_from_address=Int(table_base_list.unsafe_ptr())
    )
    var table_bytes = TableBytes(
        unsafe_from_address=Int(table_bytes_list.unsafe_ptr())
    )
    var log = String("")

    for ref fixture in fixtures:
        var bases = Array[UInt, MEGASTEP_BUFFER_SLOTS](fill=0)
        var lengths = Array[Int32, MEGASTEP_BUFFER_SLOTS](fill=0)
        for i in range(MEGASTEP_BUFFER_SLOTS):
            if fixture.slot_lengths[i] >= 0:
                bases[i] = UInt(TWIN_SLOT_BASE + i * TWIN_BASE_STRIDE)
                lengths[i] = fixture.slot_lengths[i]

        var block = WireBytes(unsafe_from_address=Int(fixture.block.unsafe_ptr()))
        var slots = SlotBases(unsafe_from_address=Int(UnsafePointer(to=bases)))
        var slot_lengths = SlotLengths(
            unsafe_from_address=Int(UnsafePointer(to=lengths))
        )
        var block_len = len(fixture.block)

        var count = (
            UInt32(fixture.block[4])
            | (UInt32(fixture.block[5]) << 8)
            | (UInt32(fixture.block[6]) << 16)
            | (UInt32(fixture.block[7]) << 24)
        )
        var mode = (
            UInt32(fixture.block[12])
            | (UInt32(fixture.block[13]) << 8)
            | (UInt32(fixture.block[14]) << 16)
            | (UInt32(fixture.block[15]) << 24)
        )

        var status: Int32
        if mode == MEGASTEP_MODE_REGISTER:
            status = megastep_register(
                table_bases,
                table_bytes,
                block,
                block_len,
                count,
                slots,
                slot_lengths,
            )
            for i in range(Int(MEGASTEP_TABLE_ENTRIES)):
                if table_bases[i] != 0:
                    log += (
                        fixture.name
                        + " entry="
                        + _dec(i)
                        + " base="
                        + _hex(Int(table_bases[i]), 1)
                        + " bytes="
                        + _dec(Int(table_bytes[i]))
                        + "\n"
                    )
        else:
            var seam = RecordingSeam(fixture.name)
            status = megastep_walk(
                seam,
                table_bases,
                table_bytes,
                block,
                block_len,
                count,
                slots,
                slot_lengths,
                False,
            )
            log += seam.log
        log += (
            fixture.name
            + " status=0x"
            + _hex(Int(UInt32(Int(status) & 0xFFFFFFFF)), 8)
            + "\n"
        )
    return log^


def test_mojo_walker_matches_the_c_walkers_golden() raises:
    var fixtures_path = getenv("MEGASTEP_WALK_FIXTURES")
    var golden_path = getenv("MEGASTEP_WALK_GOLDEN")
    assert_true(
        fixtures_path.byte_length() > 0 and golden_path.byte_length() > 0,
        "MEGASTEP_WALK_FIXTURES and MEGASTEP_WALK_GOLDEN must be set",
    )

    var fixtures = _parse_fixtures(open(fixtures_path, "r").read())
    assert_true(len(fixtures) > 0, "the fixture file names at least one block")

    var produced = _walk_fixtures(fixtures)
    var golden = open(golden_path, "r").read()

    var produced_lines = produced.splitlines()
    var golden_lines = golden.splitlines()
    for i in range(min(len(produced_lines), len(golden_lines))):
        assert_equal(
            String(produced_lines[i]),
            String(golden_lines[i]),
            "line " + _dec(i + 1) + " of the walk differs from the golden",
        )
    assert_equal(
        len(produced_lines),
        len(golden_lines),
        "the Mojo walk emitted a different number of records",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
