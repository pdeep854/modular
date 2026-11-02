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
"""Rung 0 of the crouton-crossing annihilation ladder: `swiglu_croutons`
against `swiglu_2d`, bit for bit (`DESIGN-crouton-crossing-annihilation.md`
section 2.2).

The claim under test is the design's elementwise contract: a crouton tile's
in-tile permutation CANCELS for an op whose every output element depends only
on same-position inputs, so SwiGLU over row-panel-major croutons must produce
the exact bits `swiglu_2d` produces over the same values in row-major order —
a tolerance would mean the claim is false. The reference output is REPACKED
with `hmx_repack_rows_rpm` (which is `hmx_pack_panel[False]`, the definition
of the layout, byte-order-oracled in `test_hmx_crouton.mojo`) rather than the
crouton output unpacked, so the arm under test appears on exactly one side of
the comparison.

The must-fire control is the weights-native lesson
(`known-limitations/weights-native-is-skel-wide-but-repackedness-is-per-weight`)
applied to this artefact: the ROW-MAJOR bytes viewed as a crouton panel are
the same bytes at the same length, and the kernel must produce different bits
from them — demonstrably, or the passes above prove only that some
self-consistent permutation happened.

Geometry includes the rung-1 microbench's own shape — `klein_mini`'s
`swiglu(MLP_IMG, FF_IMG, 128, 1536)` (`block_stack.mojo`) — so the shape the
device order will run is the shape this gate passed.

Crouton allocations here follow `test_hmx_crouton.mojo`'s manual
alloc/align/free-at-the-end pattern rather than an owning struct: a panel is
an UNTRACKED view, so an owner destroyed at its last field read frees the
bytes while a callee still reads them — the allocator's free-metadata scribble
then corrupts one or two elements, which this file's first draft spent a
debugging pass on.
"""
from std.math import align_up
from std.memory import AddressSpace, alloc
from std.testing import assert_equal, assert_true, TestSuite

from layout.coord import Coord
from layout.tile_layout import row_major

from hexagon_ondevice.hmx_crouton import (
    HMX_DTYPE,
    HMX_TILE,
    HMX_TILE_BYTES,
    CroutonPanel,
    HmxRows2D,
    hmx_native_row_croutons,
    hmx_panel,
    hmx_repack_rows_rpm,
)
from hexagon_ondevice.silu_swiglu_kernel import swiglu_2d, swiglu_croutons

comptime GEN = AddressSpace.GENERIC

comptime SENTINEL = Scalar[HMX_DTYPE](-4096.0)
"""Outside every value the fills or the gate can produce, so an unwritten
crouton is distinguishable from a written one."""

comptime RawPtr = UnsafePointer[Scalar[HMX_DTYPE], MutUntrackedOrigin]


def _panel_alloc(tiles: Int) -> Tuple[RawPtr, CroutonPanel[GEN]]:
    """One crouton panel over a fresh allocation, seeded with `SENTINEL`.

    `hmx_panel` claims the `:deep` load's 2048-byte alignment and the crouton
    kernels claim 128-byte vector alignment, neither of which a bare `alloc`
    promises — so the raw allocation is over-sized and the view starts at the
    aligned base. The RAW pointer is returned beside the view and freed by the
    caller at the end of its function, never earlier: the view's origin is
    untracked, so nothing else keeps the bytes alive.
    """
    var elems = tiles * HMX_TILE * HMX_TILE
    var raw = alloc[Scalar[HMX_DTYPE]](elems + HMX_TILE_BYTES)
    var base = RawPtr(unsafe_from_address=align_up(Int(raw), HMX_TILE_BYTES))
    for i in range(elems):
        base[i] = SENTINEL
    return (raw, hmx_panel[GEN](base, tiles))


def _fill_signed(t: HmxRows2D, rows: Int, cols: Int):
    """`0.25 * ((5 r + 7 c) mod 23 - 11)`: both `silu` branches, exact fp16.

    Periods coprime with each other and with the 32-wide tile, the
    `test_hmx_crouton.mojo` rule: a fill whose period divides the tile can
    fold a wrong tile index back onto the right values.
    """
    for r in range(rows):
        for c in range(cols):
            t.store[1](
                Coord(r, c),
                Scalar[HMX_DTYPE]((5 * r + 7 * c) % 23 - 11) * 0.25,
            )


def _croutons_equal(
    got: CroutonPanel[GEN], want: CroutonPanel[GEN], tiles: Int
) -> Int:
    """Elements of `got` that differ from `want`, compared as bits over the
    panel's own coordinates."""
    var wrong = 0
    for t in range(tiles):
        for s in range(HMX_TILE // 2):
            for lane in range(HMX_TILE):
                for sub in range(2):
                    var g = got.load[1, alignment=2](Coord(t, s, lane, sub))
                    var w = want.load[1, alignment=2](Coord(t, s, lane, sub))
                    if g != w:
                        wrong += 1
    return wrong


def _swiglu_case(rows: Int, hidden: Int, feed_repacked: Bool) -> Int:
    """Runs one geometry and returns the crouton-output mismatch count.

    `feed_repacked=False` is the control arm: the kernel reads the row-major
    bytes through the panel view — the exact bytes an unstaged artefact would
    hand it — and the return must be non-zero.
    """
    var in_elems = rows * 2 * hidden
    var in_rm = alloc[Scalar[HMX_DTYPE]](in_elems)
    var ref_rm = alloc[Scalar[HMX_DTYPE]](rows * hidden)

    var input = HmxRows2D(
        ptr=in_rm.as_unsafe_any_origin(),
        layout=row_major(Coord(rows, 2 * hidden)),
    )
    _fill_signed(input, rows, 2 * hidden)
    var ref_out = HmxRows2D(
        ptr=ref_rm.as_unsafe_any_origin(), layout=row_major(Coord(rows, hidden))
    )
    swiglu_2d[HMX_DTYPE](input, ref_out)

    var in_tiles = hmx_native_row_croutons(rows, 2 * hidden)
    var out_tiles = hmx_native_row_croutons(rows, hidden)
    var in_cr = _panel_alloc(in_tiles)
    var out_cr = _panel_alloc(out_tiles)
    var ref_cr = _panel_alloc(out_tiles)

    hmx_repack_rows_rpm[GEN](in_cr[1], input, rows, 2 * hidden)
    hmx_repack_rows_rpm[GEN](ref_cr[1], ref_out, rows, hidden)

    # The control's source: the SAME row-major bytes the repacked artefact was
    # built from, viewed as croutons they are not. Copied into an aligned
    # allocation so the two arms differ in byte MEANING only, never in
    # alignment.
    var rm_cr = _panel_alloc(in_tiles)
    if feed_repacked:
        swiglu_croutons(in_cr[1], out_cr[1], hidden, 0, rows // HMX_TILE)
    else:
        var rm_base = rm_cr[1].ptr_at_offset(Coord(0, 0, 0, 0))
        for i in range(in_elems):
            rm_base[i] = in_rm[i]
        swiglu_croutons(rm_cr[1], out_cr[1], hidden, 0, rows // HMX_TILE)
    var wrong = _croutons_equal(out_cr[1], ref_cr[1], out_tiles)

    in_rm.free()
    ref_rm.free()
    in_cr[0].free()
    out_cr[0].free()
    ref_cr[0].free()
    rm_cr[0].free()
    return wrong


def test_bit_exact_small() raises:
    """One tile each way, and the multi-tile row-panel arithmetic at 3 x 2."""
    assert_equal(_swiglu_case(32, 32, True), 0)
    assert_equal(_swiglu_case(96, 64, True), 0)


def test_bit_exact_klein_mini_shape() raises:
    """The rung-1 microbench's own swiglu: `[128, 3072] -> [128, 1536]`."""
    assert_equal(_swiglu_case(128, 1536, True), 0)


def test_unrepacked_control_fires() raises:
    """Row-major bytes through the crouton view must be WRONG, demonstrably.

    Same bytes, same length, different meaning — the mismatch class no size or
    identity gate can see, so the gate above is only as good as this control.
    """
    assert_true(_swiglu_case(96, 64, False) != 0)
    assert_true(_swiglu_case(128, 1536, False) != 0)


def test_striping_partition_agrees() raises:
    """Half-and-half row-tile stripes produce the whole-tensor bits.

    The launch stripes row panels across threads exactly this way, so the
    partition arithmetic is scored here rather than trusted there.
    """
    var rows = 128
    var hidden = 96
    var in_rm = alloc[Scalar[HMX_DTYPE]](rows * 2 * hidden)
    var input = HmxRows2D(
        ptr=in_rm.as_unsafe_any_origin(),
        layout=row_major(Coord(rows, 2 * hidden)),
    )
    _fill_signed(input, rows, 2 * hidden)

    var in_tiles = hmx_native_row_croutons(rows, 2 * hidden)
    var out_tiles = hmx_native_row_croutons(rows, hidden)
    var in_cr = _panel_alloc(in_tiles)
    var whole = _panel_alloc(out_tiles)
    var striped = _panel_alloc(out_tiles)
    hmx_repack_rows_rpm[GEN](in_cr[1], input, rows, 2 * hidden)

    var row_tiles = rows // HMX_TILE
    swiglu_croutons(in_cr[1], whole[1], hidden, 0, row_tiles)
    swiglu_croutons(in_cr[1], striped[1], hidden, 0, row_tiles // 2)
    swiglu_croutons(
        in_cr[1],
        striped[1],
        hidden,
        row_tiles // 2,
        row_tiles - row_tiles // 2,
    )

    assert_equal(_croutons_equal(striped[1], whole[1], out_tiles), 0)
    in_rm.free()
    in_cr[0].free()
    whole[0].free()
    striped[0].free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
