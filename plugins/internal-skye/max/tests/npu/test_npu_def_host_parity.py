"""Device-free parity gate for the HOST half of the dispatch ABI: every
`NPUKernels.def` row must satisfy the contract `QuRTDeviceContext.cpp` —
the host packer TU that expands the same file into the by-name selector
table the engine exe ships — can only enforce at runtime, on device.

Order 325 is why this exists: the shipped windows-arm64 `engine-cli.exe`
refused MEF dispatches naming rows 45/46/47 because its compiled-in
manifest predated them. The manifest is not hand-maintained — the host TU
`#include`s the `.def`, and its `static_assert` pins selector==position at
compile time — but two host-table properties have no compile-time witness
anywhere:

- `findKernel` resolves BY NAME, first match wins: a duplicated row name
  compiles green on every config and silently shadows the later row.
- `bufferCount` is checked against the FastRPC entry point's fixed slot
  count (`kNPUMaxBufferSlots`) only when a launch reaches `loadFunction`'s
  kernel at dispatch time — a row past the ceiling certifies green on the
  Mac ladder and fails at leg 1 of a ceremony, the
  gate-ladder-certifies-only-the-configs-it-compiles class.

This is the sibling of `test_npu_def_c_parity` (the device trampoline TU's
gate): same file pair discipline, other half of the expansion. What it
deliberately does NOT gate: called-entry declarations (the C parity gate's
job) and the shipped exe's manifest bytes (an artifact receipt, taken by
the relink lane's strings census, not a source property).
"""

import re
import unittest
from pathlib import Path

_PLUGIN_ROOT = Path(__file__).resolve().parents[3]
_DEF = _PLUGIN_ROOT / "runtime" / "lib" / "qurt" / "NPUKernels.def"
_HOST = _PLUGIN_ROOT / "runtime" / "lib" / "qurt" / "QuRTDeviceContext.cpp"

# The re-sign window rows this gate was bought by, with their frozen wire
# tuples (selector, bufferCount, scalarCount). A renumber or arity drift is
# loud here rather than in an on-device MEF refusal.
_WINDOW_ROWS = {
    "conv_im2col_hmx_ep": (45, 6, 12),
    "groupnorm_stats": (46, 5, 5),
    "relayout": (47, 3, 3),
}


def _strip_comments(text: str) -> str:
    """Drops // and /* */ comments: the .def's prose names real selectors."""
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", " ", text)


def _kernel_blocks(def_text: str) -> list[str]:
    """Returns each NPU_KERNEL(...) invocation's full argument text."""
    blocks = []
    for match in re.finditer(r"\bNPU_KERNEL\s*\(", def_text):
        depth = 1
        i = match.end()
        while depth and i < len(def_text):
            if def_text[i] == "(":
                depth += 1
            elif def_text[i] == ")":
                depth -= 1
            i += 1
        blocks.append(def_text[match.end() : i - 1])
    return blocks


def def_rows(def_text: str) -> list[tuple[str, int, int, int]]:
    """Each row's host-table fields: (name, selector, bufferCount,
    scalarCount) — exactly what the host `NPU_KERNEL` macro keeps; the call
    expression it drops is the C parity gate's territory.
    """
    rows = []
    for block in _kernel_blocks(_strip_comments(def_text)):
        match = re.match(
            r"\s*(\w+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,", block
        )
        assert match, f"unparseable NPU_KERNEL row: {block[:80]!r}"
        name, selector, buffers, scalars = match.groups()
        rows.append((name, int(selector), int(buffers), int(scalars)))
    return rows


def host_max_buffer_slots(host_text: str) -> int:
    """`kNPUMaxBufferSlots`, parsed from the host TU rather than restated:
    the ceiling is the FastRPC interface's, and this gate must move when it
    does."""
    match = re.search(
        r"constexpr\s+uint32_t\s+kNPUMaxBufferSlots\s*=\s*(\d+)\s*;",
        host_text,
    )
    assert match, "kNPUMaxBufferSlots not found in QuRTDeviceContext.cpp"
    return int(match.group(1))


class DefHostParityTest(unittest.TestCase):
    def setUp(self):
        self.rows = def_rows(_DEF.read_text())
        self.assertGreater(
            len(self.rows), 40, "the .def parse found too few rows"
        )

    def test_selectors_are_their_positions(self):
        """The host TU's static_assert, mirrored device-free with the row
        named: a selector is a signed library's ABI, so rows append only."""
        for position, (name, selector, _, _) in enumerate(self.rows):
            self.assertEqual(
                selector,
                position,
                f"row {name!r} declares selector {selector} at position"
                f" {position}: a row was inserted, reordered, or removed",
            )

    def test_names_are_unique(self):
        """`findKernel` is first-match by name; a duplicate compiles green
        everywhere and silently shadows the later row."""
        names = [name for name, _, _, _ in self.rows]
        dupes = {name for name in names if names.count(name) > 1}
        self.assertEqual(dupes, set(), f"duplicated kernel names: {dupes}")

    def test_buffer_counts_fit_host_slots(self):
        """The generic entry point declares a fixed slot count; the host
        refuses an over-wide row only at dispatch time, on device. Gate it
        here so the refusal is a Mac test failure, not a burned leg."""
        slots = host_max_buffer_slots(_HOST.read_text())
        over = [
            (name, buffers)
            for name, _, buffers, _ in self.rows
            if buffers > slots
        ]
        self.assertEqual(
            over,
            [],
            f"rows exceed the entry point's {slots} buffer slots: {over}",
        )

    def test_window_rows_wire_tuples(self):
        """The three order-325 rows, pinned with their full wire tuples so
        the relinked exe's manifest census has a stable source claim."""
        by_name = {name: tuple(rest) for name, *rest in self.rows}
        for name, want in _WINDOW_ROWS.items():
            self.assertIn(name, by_name, f"{name} lost its .def row")
            self.assertEqual(
                by_name[name],
                want,
                f"{name} wire tuple moved from {want}: that is a signed-ABI"
                " change, not an edit",
            )

    def test_gate_can_fail(self):
        """The falsifiers: a synthesized over-slot row and a synthesized
        duplicate must be flagged — a gate that cannot fail proves nothing."""
        bogus = (
            "NPU_KERNEL(vadd, /*selector=*/99, /*bufferCount=*/9,\n"
            "           /*scalarCount=*/0, mojo_bogus(NPU_BUFFER(0, float)))\n"
        )
        rows = def_rows(_DEF.read_text() + bogus)
        names = [name for name, _, _, _ in rows]
        self.assertGreater(names.count("vadd"), 1, "duplicate not parsed")
        slots = host_max_buffer_slots(_HOST.read_text())
        self.assertTrue(
            any(buffers > slots for _, _, buffers, _ in rows),
            "over-slot row not parsed",
        )


if __name__ == "__main__":
    unittest.main()
