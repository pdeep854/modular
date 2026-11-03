"""Device-free parity gate: every entry a `NPUKernels.def` row CALLS must be
DECLARED in `mojo_npu_imp.c`, the trampoline TU that includes the `.def`.

Order 322 is why this exists: rows 45/46/47 landed with their Mojo entries
real and proven (`FUNC GLOBAL` in `npu_kernel.o`, signatures matching) and
the window still could not compile — `mojo_npu_imp.c` carried 42 extern
declarations and none for the three new callees, so `hexagon-clang` failed
the trampoline with implicit-declaration errors and no skel was produced.
The Mac's 163-target `--full` gate never compiles that TU (the skel C
closure compiles only on the box's SDK leg), so the gate certified a tree
whose window build was broken: the gate-ladder-certifies-only-the-configs-
it-compiles class, one more instance. This test is the device-free half the
ladder was missing: it parses the two files the box compiles together and
fails on any called-but-undeclared entry, so the class dies at the Mac gate
instead of at leg 1 of a ceremony.

What it deliberately does NOT gate: declared-but-uncalled names (the TU
legitimately declares helpers), and signature AGREEMENT — the C compiler
type-checks the declarations against the `.def` call sites in the very TU
this test guards, so a wrong signature still fails loudly at the one
compile that can judge it. Parity of NAMES is the piece nothing checked.
"""

import re
import unittest
from pathlib import Path

_PLUGIN_ROOT = Path(__file__).resolve().parents[3]
_DEF = _PLUGIN_ROOT / "runtime" / "lib" / "qurt" / "NPUKernels.def"
_IMP = _PLUGIN_ROOT / "examples" / "hexagon-ondevice" / "mojo_npu_imp.c"

_MACRO_NAMES = {"NPU_KERNEL", "NPU_BUFFER", "NPU_SCALAR"}


def _strip_comments(text: str) -> str:
    """Drops // and /* */ comments: the .def's prose names real selectors."""
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
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


def def_callees(def_text: str) -> dict[str, str]:
    """Maps each row's kernel name to the entry its call expression names.

    A row with no call expression — the megastep transport row spells
    `(void)0` — contributes nothing, which is correct: nothing is linked
    for it, so nothing needs declaring.
    """
    callees = {}
    for block in _kernel_blocks(_strip_comments(def_text)):
        row_name = block.split(",", 1)[0].strip()
        for ident in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", block):
            name = ident.group(1)
            if name not in _MACRO_NAMES and name != "void":
                callees[row_name] = name
                break
    return callees


def imp_declared(imp_text: str) -> set[str]:
    """Every function name visible to the `.def` expansion in
    `mojo_npu_imp.c`: the extern declarations AND the TU's own definitions
    — `mojo_npu_power_vote` is a C function defined in this file above the
    include (the selector-37 trampoline half), and a definition declares.
    """
    return set(
        re.findall(
            r"\bvoid\s+([A-Za-z_]\w*)\s*\(",
            _strip_comments(imp_text),
        )
    )


class DefCParityTest(unittest.TestCase):
    def test_every_called_entry_is_declared(self):
        callees = def_callees(_DEF.read_text())
        declared = imp_declared(_IMP.read_text())
        self.assertGreater(len(callees), 40, "the .def parse found too few rows")
        self.assertGreater(len(declared), 40, "the imp.c parse found too few externs")
        missing = {
            row: fn for row, fn in sorted(callees.items()) if fn not in declared
        }
        self.assertEqual(
            missing,
            {},
            "NPUKernels.def rows call entries mojo_npu_imp.c never declares;"
            " hexagon-clang will fail the trampoline TU on the box (order"
            f" 322's class). Add the extern declarations: {missing}",
        )

    def test_new_window_rows_present_both_sides(self):
        """The three order-322 names, pinned by name so a rename or a row
        renumber that drops one is loud here rather than at the window."""
        callees = def_callees(_DEF.read_text())
        declared = imp_declared(_IMP.read_text())
        for fn in (
            "mojo_conv_im2col_hmx_ep_f16",
            "mojo_groupnorm_stats_f16",
            "mojo_relayout_f16",
        ):
            self.assertIn(fn, set(callees.values()), f"{fn} lost its .def row")
            self.assertIn(fn, declared, f"{fn} lost its extern declaration")

    def test_gate_can_fail(self):
        """The falsifier: a synthesized row calling an undeclared entry must
        be flagged — a parity check that cannot fail proves nothing."""
        bogus = (
            "NPU_KERNEL(bogus, /*selector=*/99, /*bufferCount=*/1,\n"
            "           /*scalarCount=*/0,\n"
            "           mojo_never_declared_f16(NPU_BUFFER(0, __fp16)))\n"
        )
        callees = def_callees(_DEF.read_text() + bogus)
        declared = imp_declared(_IMP.read_text())
        self.assertEqual(callees.get("bogus"), "mojo_never_declared_f16")
        self.assertNotIn("mojo_never_declared_f16", declared)

    def test_void_row_contributes_no_callee(self):
        """The megastep transport row is `(void)0`: no callee, no gate."""
        callees = def_callees(_DEF.read_text())
        self.assertNotIn("megastep", callees)


if __name__ == "__main__":
    unittest.main()
