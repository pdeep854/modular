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
"""The standalone HMX bench's host gate (`hmx_standalone_bench.mojo`,
device legs on order DRAFT-210).

What this gates is the BENCH's own claims — the accounting a device rate
divides by, the refusal boundaries, the fills' permutation-sensitivity, the
witness fold's controls, the arm echoes, and the integer identity record —
run through the same pass bodies the device shim will call, with the MMA
modelled (`emit_hmx=False`; the host cannot execute HMX). What it deliberately
does NOT gate: the MMA model's arithmetic against a reference, which
`test_hmx_crouton.mojo` and `test_hmx_int8.mojo` own — a third transcription
of the crouton layout here would be the drift this module pattern exists to
prevent.

Every control is shown able to fail: the fold has an unwritten-tile zero and
a one-element flip; the fills have a swapped-operand disagreement; the
integer record has the fp16-habit mutant (scale 1.0), which must produce the
all-zero readout that burned every pre-`/512` probe.
"""
from std.math import align_up
from std.memory import AddressSpace, alloc
from std.testing import assert_equal, assert_true, TestSuite

from layout.coord import Coord

from hexagon_ondevice.hmx_crouton import (
    HMX_DTYPE,
    HMX_TILE,
    HmxScratch,
    HmxSplit,
    HmxTileBytes,
    hmx_scratch_bytes,
    hmx_scratch_in,
)
from hexagon_ondevice.hmx_int8 import (
    HMX_INT8_DTYPE,
    hmx_int8_program_record,
)
from hexagon_ondevice.hmx_standalone_bench import (
    BENCH_ARM_FP16,
    BENCH_ARM_ISSUED,
    BENCH_ARM_W8A8,
    BENCH_FORM_DEEP,
    BENCH_FORM_PLAIN,
    BENCH_K_MID,
    BENCH_K_PROD,
    BENCH_K_SMALL,
    BENCH_PROD_N,
    bench_array_macs,
    bench_array_pass,
    bench_array_program_record,
    bench_compose_floor_ticks,
    bench_compose_serial_ticks,
    bench_fill_byte_croutons,
    bench_fill_fp16_croutons,
    bench_min_k_tiles,
    bench_overlap_efficiency_ppm,
    bench_panel_bytes,
    bench_readout_fold,
    bench_stage_chunk_bytes,
)
from hexagon_ondevice.matmul_kernel import hmx_work_share

comptime GEN = AddressSpace.GENERIC

comptime GATE_K_TILES = 2
"""The gated chain depth: 64 of K. Small enough that the byte fills below
keep every integer accumulator inside the saturating convert (<= 128 against
the 255 ceiling), so a clamped element cannot hide a distinctness failure."""

comptime ACT_SALT = 0
comptime WGT_SALT = 3
comptime ACT_PERIOD = 3
comptime WGT_PERIOD = 2


struct _Scratch[dtype: DType](ImplicitlyDeletable):
    """One aligned host scratch: the device shim's `hmx_scratch_in`, on heap."""

    var raw: UnsafePointer[UInt8, MutUntrackedOrigin]
    var views: HmxScratch[GEN, Self.dtype]

    def __init__(out self, k_tiles: Int):
        var bytes = hmx_scratch_bytes[Self.dtype](k_tiles)
        self.raw = alloc[UInt8](bytes + 4096)
        var base = UnsafePointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=align_up(Int(self.raw), 4096)
        )
        for i in range(bytes):
            base.store(i, 0)
        self.views = hmx_scratch_in[GEN, Self.dtype](base, k_tiles)

    def __del__(deinit self):
        self.raw.free()


def test_mac_accounting() raises:
    """The GMAC/s denominator equals a counted loop, and moves with `k_tiles`.
    """
    comptime K_TILES = 3
    comptime ITERS = 2
    var counted: UInt64 = 0
    for _ in range(ITERS):
        for _ in range(K_TILES):
            for _ in range(HMX_TILE):
                counted += UInt64(HMX_TILE * HMX_TILE)
    assert_equal(bench_array_macs(K_TILES, ITERS), counted)
    # The off-by-one mutation: a denominator blind to depth would report a
    # plausible rate for the wrong work.
    assert_true(bench_array_macs(K_TILES + 1, ITERS) != counted)
    # The production point, stated once so the order's arithmetic has a pin:
    # one pass at K=12288 is 384 croutons = 12.58 MMAC.
    assert_equal(bench_array_macs(BENCH_K_PROD // HMX_TILE, 1), 12_582_912)


def test_stage_bytes_partition() raises:
    """Feeder chunk shares partition a panel exactly, at both dtypes' sizes."""
    comptime K_TILES = BENCH_K_PROD // HMX_TILE
    comptime FEEDERS = 4
    assert_equal(bench_panel_bytes[HMX_DTYPE](K_TILES), K_TILES * 2048)
    assert_equal(bench_panel_bytes[HMX_INT8_DTYPE](K_TILES), K_TILES * 1024)
    # The production staging grain the membw sweep must include.
    assert_equal(bench_stage_chunk_bytes[HMX_DTYPE](K_TILES, FEEDERS), 196608)
    # Shares summed over feeders are the whole panel: no crouton staged twice,
    # none dropped — the property `hmx_work_share` promises, priced in bytes.
    var total = 0
    for t in range(FEEDERS):
        total += (
            hmx_work_share(K_TILES, t + 1, FEEDERS)
            - hmx_work_share(K_TILES, t, FEEDERS)
        ) * HmxTileBytes[HMX_DTYPE]
    assert_equal(total, bench_panel_bytes[HMX_DTYPE](K_TILES))


def test_min_k_tiles_and_ladder() raises:
    """The refusal floors, and where the 184 K ladder sits against them."""
    assert_equal(bench_min_k_tiles[HMX_DTYPE](), 29)
    assert_equal(bench_min_k_tiles[HMX_INT8_DTYPE](), 32)
    # K=64 is 2 croutons: BELOW both floors, so its pure-array row is the
    # pre-registered refusal DRAFT-210 carries, not a measurable rate.
    assert_true(BENCH_K_SMALL // HMX_TILE < bench_min_k_tiles[HMX_DTYPE]())
    # K=1024 admits fp16 and sits exactly ON the int8 floor; K=12288 clears
    # both with headroom.
    assert_true(BENCH_K_MID // HMX_TILE >= bench_min_k_tiles[HMX_DTYPE]())
    assert_equal(BENCH_K_MID // HMX_TILE, bench_min_k_tiles[HMX_INT8_DTYPE]())
    assert_true(BENCH_K_PROD // HMX_TILE >= bench_min_k_tiles[HMX_INT8_DTYPE]())
    # The production N divides into whole column tiles, so the overlap leg's
    # plan has no ragged panel to special-case.
    assert_equal(BENCH_PROD_N % HMX_TILE, 0)


def test_arm_codes() raises:
    """The arm echoes are distinct, and the issued bit clobbers neither."""
    assert_true(BENCH_ARM_FP16 != BENCH_ARM_W8A8)
    assert_equal(BENCH_ARM_FP16 & BENCH_ARM_ISSUED, 0)
    assert_equal(BENCH_ARM_W8A8 & BENCH_ARM_ISSUED, 0)


def test_fp16_pass_modelled() raises:
    """The fp16 pass body: arm echo, fold controls, fill sensitivity."""
    var scratch = _Scratch[HMX_DTYPE](GATE_K_TILES)

    # An unwritten (zeroed) readout folds to zero: the control's zero side.
    assert_equal(bench_readout_fold(scratch.views.out), 0)

    bench_fill_fp16_croutons(scratch.views.act, GATE_K_TILES, ACT_SALT)
    bench_fill_fp16_croutons(scratch.views.wgt, GATE_K_TILES, WGT_SALT)
    bench_array_program_record(scratch.views)

    var arm = bench_array_pass[form=BENCH_FORM_DEEP, emit_hmx=False](
        scratch.views, GATE_K_TILES
    )
    assert_equal(arm, BENCH_ARM_FP16)
    var fold = bench_readout_fold(scratch.views.out)
    assert_true(fold != 0)

    # The blindness control: a fill uniform along any axis folds a uniform
    # readout, and a permutation could hide in it (the 184F lesson). Two
    # distinct elements are the minimum evidence of sensitivity.
    var first = scratch.views.out.load[1](Coord(0, 0, 0, 0))
    var distinct = False
    for slab in range(HMX_TILE):
        for lane in range(HMX_TILE):
            comptime SPLIT = HmxSplit[HMX_DTYPE]
            if (
                scratch.views.out.load[1](
                    Coord(0, slab // SPLIT, lane, slab % SPLIT)
                )
                != first
            ):
                distinct = True
    assert_true(distinct)

    # The plain form models identically (the forms differ only in issue
    # count, which a model has none of).
    var plain = _Scratch[HMX_DTYPE](GATE_K_TILES)
    bench_fill_fp16_croutons(plain.views.act, GATE_K_TILES, ACT_SALT)
    bench_fill_fp16_croutons(plain.views.wgt, GATE_K_TILES, WGT_SALT)
    bench_array_program_record(plain.views)
    var plain_arm = bench_array_pass[form=BENCH_FORM_PLAIN, emit_hmx=False](
        plain.views, GATE_K_TILES
    )
    assert_equal(plain_arm, BENCH_ARM_FP16)
    assert_equal(bench_readout_fold(plain.views.out), fold)

    # The wrong-operand control: swapping the two fills must move the fold —
    # a bench whose readout cannot tell A from B measures nothing about
    # either.
    var swapped = _Scratch[HMX_DTYPE](GATE_K_TILES)
    bench_fill_fp16_croutons(swapped.views.act, GATE_K_TILES, WGT_SALT)
    bench_fill_fp16_croutons(swapped.views.wgt, GATE_K_TILES, ACT_SALT)
    bench_array_program_record(swapped.views)
    _ = bench_array_pass[form=BENCH_FORM_DEEP, emit_hmx=False](
        swapped.views, GATE_K_TILES
    )
    assert_true(bench_readout_fold(swapped.views.out) != fold)

    # The flip control: one moved bit pattern must move the fold.
    scratch.views.out.store[1](
        Coord(0, 0, 0, 0),
        scratch.views.out.load[1](Coord(0, 0, 0, 0)) + Scalar[HMX_DTYPE](0.25),
    )
    assert_true(bench_readout_fold(scratch.views.out) != fold)


def test_int8_pass_modelled() raises:
    """The integer pass body: arm echo, the `/512` record, the all-zero mutant.
    """
    var scratch = _Scratch[HMX_INT8_DTYPE](GATE_K_TILES)
    bench_fill_byte_croutons(
        scratch.views.act, GATE_K_TILES, ACT_SALT, ACT_PERIOD
    )
    bench_fill_byte_croutons(
        scratch.views.wgt, GATE_K_TILES, WGT_SALT, WGT_PERIOD
    )
    bench_array_program_record(scratch.views)

    # The integer identity is 512.0 (0x6000), not fp16's 1.0 (0x3c00): the
    # constant whose absence made every pre-`/512` probe unfalsifiable. The
    # record word carries it in the low halfword.
    assert_equal(
        scratch.views.record.load[1](Coord(0, 0)) & 0xFFFF,
        Scalar[DType.uint32](0x6000),
    )

    var arm = bench_array_pass[form=BENCH_FORM_PLAIN, emit_hmx=False](
        scratch.views, GATE_K_TILES
    )
    assert_equal(arm, BENCH_ARM_W8A8)
    var fold = bench_readout_fold(scratch.views.out)
    assert_true(fold != 0)

    # Distinctness, as the fp16 leg has it.
    var first = scratch.views.out.load[1](Coord(0, 0, 0, 0))
    var distinct = False
    for slab in range(HMX_TILE):
        for lane in range(HMX_TILE):
            comptime SPLIT = HmxSplit[HMX_INT8_DTYPE]
            if (
                scratch.views.out.load[1](
                    Coord(0, slab // SPLIT, lane, slab % SPLIT)
                )
                != first
            ):
                distinct = True
    assert_true(distinct)

    # THE MUTANT THE HISTORY DEMANDS: program the fp16 habit (scale 1.0) and
    # the convert's `/512` floors every accumulator this fill can reach to
    # zero — the all-zero autopsy (`HMX_INT8_SCOPE.md` section 2 H1) as a
    # control that must fire.
    var mutant = _Scratch[HMX_INT8_DTYPE](GATE_K_TILES)
    bench_fill_byte_croutons(
        mutant.views.act, GATE_K_TILES, ACT_SALT, ACT_PERIOD
    )
    bench_fill_byte_croutons(
        mutant.views.wgt, GATE_K_TILES, WGT_SALT, WGT_PERIOD
    )
    hmx_int8_program_record(
        mutant.views.record, Scalar[DType.float16](1), Int32(0)
    )
    _ = bench_array_pass[form=BENCH_FORM_PLAIN, emit_hmx=False](
        mutant.views, GATE_K_TILES
    )
    assert_equal(bench_readout_fold(mutant.views.out), 0)

    # The flip control.
    scratch.views.out.store[1](
        Coord(0, 0, 0, 0),
        scratch.views.out.load[1](Coord(0, 0, 0, 0))
        + Scalar[HMX_INT8_DTYPE](1),
    )
    assert_true(bench_readout_fold(scratch.views.out) != fold)


def test_overlap_arithmetic() raises:
    """The composition floor, the serial wall, and the efficiency's edges."""
    assert_equal(bench_compose_floor_ticks(500, 1000), 1000)
    assert_equal(bench_compose_floor_ticks(1000, 500), 1000)
    assert_equal(bench_compose_serial_ticks(500, 1000), 1500)
    # A pipeline exactly at the floor is 100%; the DRAFT-210 gate sits at 80%.
    assert_equal(bench_overlap_efficiency_ppm(1000, 1000), 1_000_000)
    assert_equal(bench_overlap_efficiency_ppm(1000, 1250), 800_000)
    # Faster than the floor is reported as computed — it indicts the legs,
    # and clamping it would be the placebo class.
    assert_equal(bench_overlap_efficiency_ppm(1000, 800), 1_250_000)
    # A bracket that measured nothing has no efficiency, not a fault.
    assert_equal(bench_overlap_efficiency_ppm(1000, 0), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
