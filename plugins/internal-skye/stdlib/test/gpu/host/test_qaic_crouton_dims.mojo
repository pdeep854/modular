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
"""Host-side characterization of the HMX crouton dimension math.

The crouton-conversion / HMX matmul pipeline allocates its VTCM buffers from
the dims returned by `hmx_flat_dims_yd` and `hmx_crouton_dims_from_flat`, and
sizes them with `hmx_crouton_size_in_bytes`. Those three pure functions are the
foundation the on-device DMA/HMX/crouton intersection rests on: get them wrong
and every crouton buffer is mis-sized regardless of what the hardware does.

This test exercises them on host (no device) across sub-tile, exact-tile, and
unaligned shapes and pins down the ground truth:

- flat dims encode [M, K] as YD: n=1, x=1, y=M, d=K, elem_size=2 (FP16).
- crouton dims round Y and D up to a multiple of 32 (the 32x32 FP16 tile) and
  report the count of tiles along each axis, not the element extent.
- the crouton byte size is `y_tiles * d_tiles * 32 * 32 * 2`, i.e. a whole
  number of 2048-byte tiles — the `HMX_ALIGNMENT` granularity.

It is a pure-math unit test, so it asserts exact equality against an
independent reference rather than a tolerance.
"""

from common import print_separator, print_test_header

from std.testing import TestSuite, assert_equal, assert_true

from std._plugin.qaic.host.hmx import (
    QAicHMXDims,
    hmx_flat_dims_yd,
    hmx_crouton_dims_from_flat,
    hmx_crouton_size_in_bytes,
    HMX_ALIGNMENT,
    HMX_CROUTON_TILE_Y,
    HMX_CROUTON_TILE_D,
)


# Independent reference for "round v up to a multiple of 32" — deliberately a
# different expression than the production `_align_to` bit-trick so a regression
# in either implementation is caught.
def _ref_round_up_32(v: Int) -> Int:
    return ((v + 31) // 32) * 32


def check_shape(M: Int, K: Int) raises:
    print("  shape [", M, ",", K, "]")

    var flat = hmx_flat_dims_yd(M, K)

    # Flat (row-major) YD encoding of [M, K].
    assert_equal(Int(flat.n_dim), 1)
    assert_equal(Int(flat.x_dim), 1)
    assert_equal(Int(flat.y_dim), M)
    assert_equal(Int(flat.d_dim), K)
    assert_equal(Int(flat.elem_size), 2)

    var crouton = hmx_crouton_dims_from_flat(flat)

    var y_tiles_ref = _ref_round_up_32(M) // HMX_CROUTON_TILE_Y
    var d_tiles_ref = _ref_round_up_32(K) // HMX_CROUTON_TILE_D

    # Crouton dims report tile counts, with Y/D padded up to the 32-tile grid.
    assert_equal(Int(crouton.n_dim), 1)
    assert_equal(Int(crouton.x_dim), 1)
    assert_equal(Int(crouton.y_dim), y_tiles_ref)
    assert_equal(Int(crouton.d_dim), d_tiles_ref)
    assert_equal(Int(crouton.elem_size), 2)

    # Every shape occupies at least one tile along each axis.
    assert_true(Int(crouton.y_dim) >= 1)
    assert_true(Int(crouton.d_dim) >= 1)

    var bytes = hmx_crouton_size_in_bytes(crouton)
    var bytes_ref = (
        y_tiles_ref * d_tiles_ref * HMX_CROUTON_TILE_Y * HMX_CROUTON_TILE_D * 2
    )
    assert_equal(bytes, bytes_ref)

    # The size is always a whole number of 2048-byte crouton tiles, so a buffer
    # placed at an HMX_ALIGNMENT offset stays aligned for the next one.
    assert_equal(bytes, y_tiles_ref * d_tiles_ref * HMX_ALIGNMENT)
    assert_equal(bytes % HMX_ALIGNMENT, 0)

    print(
        "    -> crouton tiles [",
        Int(crouton.y_dim),
        "x",
        Int(crouton.d_dim),
        "],",
        bytes,
        "bytes (",
        bytes // HMX_ALIGNMENT,
        "tiles )",
    )


def test_crouton_dims_subtile() raises:
    print_test_header("crouton dims: sub-tile shapes (< 32)")
    # Anything in 1..32 along an axis must collapse to exactly one 32-tile.
    check_shape(1, 1)
    check_shape(1, 32)
    check_shape(32, 1)
    check_shape(5, 7)
    check_shape(31, 31)
    check_shape(32, 32)


def test_crouton_dims_exact_tile() raises:
    print_test_header("crouton dims: exact tile multiples")
    check_shape(32, 64)
    check_shape(64, 64)
    check_shape(128, 128)
    check_shape(128, 256)
    check_shape(256, 64)


def test_crouton_dims_unaligned() raises:
    print_test_header("crouton dims: unaligned shapes")
    # Just over a tile boundary must round up to the next tile on that axis.
    check_shape(33, 17)
    check_shape(33, 33)
    check_shape(65, 97)
    check_shape(100, 200)
    check_shape(127, 129)


def test_crouton_dims_large() raises:
    print_test_header("crouton dims: large LLM-scale shapes")
    check_shape(1024, 4096)
    check_shape(4096, 4096)


def test_crouton_dims_boundary_step() raises:
    # Crossing each 32-boundary adds exactly one tile and 2048 * other_tiles
    # bytes — verify the step is monotone and quantized.
    print_test_header("crouton dims: tile-count is monotone in M")
    var prev_tiles = 0
    for m in range(1, 130):
        var crouton = hmx_crouton_dims_from_flat(hmx_flat_dims_yd(m, 32))
        var tiles = Int(crouton.y_dim)
        assert_true(tiles >= prev_tiles)
        assert_equal(tiles, _ref_round_up_32(m) // 32)
        prev_tiles = tiles
    print("  M in 1..129 -> y-tiles monotone, max", prev_tiles)
    print_separator()
    print("TEST PASSED!")
    print_separator()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
