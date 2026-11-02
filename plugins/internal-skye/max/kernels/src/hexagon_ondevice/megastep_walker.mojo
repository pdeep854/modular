"""The megastep walker, in Mojo (P5 megakernel slice 1, design section 3).

`npu_run_megastep` decodes a `'NMGS'` block -- a 16-byte header, 32-byte
records, eight 16-bit operand references per record -- resolving each reference
either to one of the call's eight interface slots or to an entry in a
session-persistent table, and presents every launch in order, stopping at the
first failure. That decode is the last piece of hand-written C dispatch logic in
the port, and it is pure arithmetic over a byte buffer: nothing in it needs
QuRT, the FastRPC entry, or the kernel manifest's preprocessor.

So it moves here. The records consumed, the refusals issued and the launches
presented are what they are in `npu_megastep_block.h`, byte for byte; this is a
transliteration and its gate is bit-identity.

What stays in C, and why (design section 3.2):

  * the skel entry, whose signature `mojo_npu.idl` fixes and whose interface md5
    is part of the signed tuple;
  * the manifest's arity switch, an X-macro expansion of `NPUKernels.def` whose
    `_Static_assert` is what keeps the host's numbering and the skel's
    identical -- Mojo has no preprocessor and no equivalent build-time
    reconciliation;
  * the launch body itself, which creates QuRT threads on statically reserved
    stacks.

The walk reaches all three through `MegastepSeam`, a parameter rather than a
hard-wired `external_call`. That is what lets a host twin drive the identical
body with a recorder in place of the device -- one source, two instantiations,
which is the same discipline `_hmx_stage_weight`'s `await_xfer` parameter
already follows.
"""

from std.memory import UnsafePointer


comptime MEGASTEP_MAGIC: UInt32 = 0x4E4D4753
"""'NMGS'. A block that does not open with this is not a megastep."""

comptime MEGASTEP_HEADER_BYTES = 16
"""Header: magic, count, report slot, mode."""

comptime MEGASTEP_RECORD_BYTES = 32
"""Record: selector, thread mask, argument offset, arity, then eight 16-bit
operand references."""

comptime MEGASTEP_ENTRY_BYTES = 8
"""Register entry: table index, slot of this call."""

comptime MEGASTEP_MODE_REGISTER: UInt32 = 0
comptime MEGASTEP_MODE_RUN: UInt32 = 1

comptime MEGASTEP_MAX_LAUNCHES: UInt32 = 1024
"""Bounds the loop against a corrupt count; a denoise step is ~403 launches."""

comptime MEGASTEP_TABLE_ENTRIES: UInt32 = 512
"""Session-registered allocations a table may name."""

comptime MEGASTEP_REF_TABLE: UInt32 = 0x8000
"""Set in an operand reference to name the session table; clear to name one of
this call's slots."""

comptime MEGASTEP_BUFFER_SLOTS = 8
comptime MEGASTEP_SCALAR_SLOT_BYTES = 8
comptime MEGASTEP_SKIP_COMPUTE: UInt32 = 0x00020000

comptime MEGASTEP_EBADPARM: Int32 = -0x7FFFFBF2
"""`AEE_EBADPARM`, 0x8000040E, as the signed status the C caller reads."""

comptime WireBytes = UnsafePointer[UInt8, ImmUnsafeAnyOrigin]
"""A buffer off the wire: no alignment guarantee, so words are composed from
byte loads rather than loaded whole."""

comptime SlotBases = UnsafePointer[UInt, ImmUnsafeAnyOrigin]
"""The call's eight operand bases, as ADDRESSES rather than pointers.

The walk never dereferences an operand: it compares a base against zero, bounds
an offset against the allocation's length and passes both on. Reading the C
side's `uint8 *const *` as pointer-sized words says exactly that, and it is
also the only spelling available -- a Mojo `UnsafePointer` is non-nullable by
construction, and a slot the call did not present IS null."""

comptime SlotLengths = UnsafePointer[Int32, ImmUnsafeAnyOrigin]

comptime TableBases = UnsafePointer[UInt, MutUnsafeAnyOrigin]
comptime TableBytes = UnsafePointer[UInt32, MutUnsafeAnyOrigin]
"""The session table, as two parallel arrays the C side owns.

Two arrays rather than an array of pairs because `{pointer, uint32}` pads to 8
bytes on the 32-bit device and 16 on a 64-bit host, so a shared struct layout
would agree on one target and not the other. Parallel arrays have no padding to
disagree about. The table lives in C because it is session state and Mojo has
no module-scope mutable global."""


trait MegastepSeam:
    """What the walk needs from whatever it is walking for.

    A device build binds these to the trampoline's `npu_buffer_count`,
    `npu_scalar_count` and `npu_run_launch`; a host twin binds them to a
    recorder, which is what makes the two walkers' launch sequences comparable
    without a device.
    """

    def buffer_count(self, selector: UInt32) -> Int:
        """Buffers this selector's kernel takes, per the manifest."""
        ...

    def scalar_count(self, selector: UInt32) -> Int:
        """Scalars this selector's kernel takes, per the manifest."""
        ...

    def launch(
        mut self,
        selector: UInt32,
        thread_mask: UInt32,
        args: WireBytes,
        args_bytes: Int,
        buffers: SlotBases,
        lengths: SlotLengths,
    ) -> Int32:
        """Presents one resolved launch. Non-zero stops the walk."""
        ...


@always_inline
def _read_u32(block: WireBytes, at: Int) -> UInt32:
    """Composes a little-endian word from byte loads.

    The block is unaligned by construction, and Hexagon and every host this twin
    runs on are little-endian, so this is the same value the C walker's
    `__builtin_memcpy` into a `uint32` produces.
    """
    return (
        UInt32(block[at])
        | (UInt32(block[at + 1]) << 8)
        | (UInt32(block[at + 2]) << 16)
        | (UInt32(block[at + 3]) << 24)
    )


@always_inline
def _read_u16(block: WireBytes, at: Int) -> UInt32:
    return UInt32(block[at]) | (UInt32(block[at + 1]) << 8)


@always_inline
def _read_u64(block: WireBytes, at: Int) -> UInt64:
    return UInt64(_read_u32(block, at)) | (
        UInt64(_read_u32(block, at + 4)) << 32
    )


def megastep_register(
    table_bases: TableBases,
    table_bytes: TableBytes,
    block: WireBytes,
    block_len: Int,
    count: UInt32,
    slots: SlotBases,
    slot_lengths: SlotLengths,
) -> Int32:
    """Fills the session table from a REGISTER-mode block.

    Entries name this call's slots, so a register call carries at most one entry
    per slot worth naming.
    """
    if count == 0 or count > UInt32(MEGASTEP_BUFFER_SLOTS):
        return MEGASTEP_EBADPARM
    if UInt32(block_len) < UInt32(MEGASTEP_HEADER_BYTES) + count * UInt32(
        MEGASTEP_ENTRY_BYTES
    ):
        return MEGASTEP_EBADPARM
    for i in range(Int(count)):
        var entry = MEGASTEP_HEADER_BYTES + i * MEGASTEP_ENTRY_BYTES
        var index = _read_u32(block, entry)
        var slot = _read_u32(block, entry + 4)
        if (
            index >= MEGASTEP_TABLE_ENTRIES
            or slot >= UInt32(MEGASTEP_BUFFER_SLOTS)
            or slots[Int(slot)] == 0
            or slot_lengths[Int(slot)] <= 0
        ):
            return MEGASTEP_EBADPARM
        table_bases[Int(index)] = slots[Int(slot)]
        table_bytes[Int(index)] = UInt32(slot_lengths[Int(slot)])
    return 0


def megastep_walk[
    Seam: MegastepSeam
](
    mut seam: Seam,
    table_bases: TableBases,
    table_bytes: TableBytes,
    block: WireBytes,
    block_len: Int,
    count: UInt32,
    slots: SlotBases,
    slot_lengths: SlotLengths,
    skip_compute: Bool,
) -> Int32:
    """Every launch a RUN-mode block names, in order, stopping at the first
    failure.

    The walk only does what the launch body cannot, because only it knows the
    block and the session table: it resolves each launch's operand references
    and refuses a block whose counted layout disagrees with the manifest, before
    the launch body is asked to trust it.
    """
    var buffers = Array[UInt, MEGASTEP_BUFFER_SLOTS](fill=0)
    var lengths = Array[Int32, MEGASTEP_BUFFER_SLOTS](fill=0)

    for i in range(Int(count)):
        var record = MEGASTEP_HEADER_BYTES + i * MEGASTEP_RECORD_BYTES
        var selector = _read_u32(block, record)
        var thread_mask = _read_u32(block, record + 4)
        var args_offset = _read_u32(block, record + 8)

        var buffer_count = seam.buffer_count(selector)
        var scalar_count = seam.scalar_count(selector)
        if buffer_count > MEGASTEP_BUFFER_SLOTS:
            return MEGASTEP_EBADPARM

        # The arity travels and is checked for the same reason the batch's is: a
        # launch's arguments are found by counting from an offset, so a host
        # built against a newer manifest is refused rather than decoded.
        var arity = _read_u32(block, record + 12)
        if (arity & 0xFFFF) != UInt32(buffer_count) or (arity >> 16) != UInt32(
            scalar_count
        ):
            return MEGASTEP_EBADPARM

        var args_bytes = (scalar_count + buffer_count) * MEGASTEP_SCALAR_SLOT_BYTES
        if args_offset > UInt32(block_len) or UInt32(block_len) - args_offset < UInt32(
            args_bytes
        ):
            return MEGASTEP_EBADPARM
        var args = block + Int(args_offset)

        for j in range(MEGASTEP_BUFFER_SLOTS):
            buffers[j] = 0
            lengths[j] = 0

        for j in range(buffer_count):
            var ref_word = _read_u16(block, record + 16 + 2 * j)
            var base: UInt
            var bytes: UInt32
            if (ref_word & MEGASTEP_REF_TABLE) != 0:
                var at = ref_word & (MEGASTEP_REF_TABLE - 1)
                # An entry nothing registered is refused, not read: a zero base
                # is the state every entry starts in and returns to nothing.
                if at >= MEGASTEP_TABLE_ENTRIES or table_bases[Int(at)] == 0:
                    return MEGASTEP_EBADPARM
                base = table_bases[Int(at)]
                bytes = table_bytes[Int(at)]
            else:
                if (
                    ref_word >= UInt32(MEGASTEP_BUFFER_SLOTS)
                    or slots[Int(ref_word)] == 0
                    or slot_lengths[Int(ref_word)] <= 0
                ):
                    return MEGASTEP_EBADPARM
                base = slots[Int(ref_word)]
                bytes = UInt32(slot_lengths[Int(ref_word)])
            var offset = _read_u64(
                args, (scalar_count + j) * MEGASTEP_SCALAR_SLOT_BYTES
            )
            if offset >= UInt64(bytes):
                return MEGASTEP_EBADPARM
            buffers[j] = base
            lengths[j] = Int32(bytes)

        var mask = thread_mask | (
            MEGASTEP_SKIP_COMPUTE if skip_compute else UInt32(0)
        )
        var status = seam.launch(
            selector,
            mask,
            args,
            args_bytes,
            SlotBases(unsafe_from_address=Int(UnsafePointer(to=buffers))),
            SlotLengths(unsafe_from_address=Int(UnsafePointer(to=lengths))),
        )
        if status != 0:
            return status
    return 0
