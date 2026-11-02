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
"""The `hmx_bench` selector's host gate (re-sign window queue item 8).

`test_hmx_standalone_bench.mojo` gates the bench module's pass bodies and
accounting; this file gates what the SELECTOR adds on top of them — the mode
table, the refusal ladder the device runs verbatim (`bench_geometry_status`),
the movement grain, and the staged stream the device brackets
(`bench_array_stage` + `bench_array_stream`).

The stream tests are deliberately not unit-shaped: they import and CALL the
real pass bodies through the same two functions the device shim calls, so a
runner that dead-stripped the pass would fail here rather than ship a shim
around nothing — the #454 lesson (main()-only self-tests were dead-stripped
from the shipped runner) made executable. The `/512` all-zero mutant rides
along at this level too: staging programs the integer record, so the fp16
habit must again produce the all-zero readout it burned every pre-`/512`
probe with.
"""
from std.math import align_up
from std.memory import AddressSpace, alloc
from std.testing import assert_equal, assert_true, TestSuite

from layout.coord import Coord

from hexagon_ondevice.hmx_crouton import (
    HMX_DTYPE,
    HMX_RECORD_BYTES,
    HMX_STORE_REGION_BYTES,
    HMX_TILE,
    HmxScratch,
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
    BENCH_ARM_W8A8,
    BENCH_FORM_DEEP,
    BENCH_FORM_PLAIN,
    BENCH_K_PROD,
    BENCH_K_SMALL,
    BENCH_INSTRIPE_DEPTH_MAX,
    BENCH_MODE_FP16_DEEP,
    BENCH_MODE_FP16_INSTRIPE,
    BENCH_MODE_FP16_PLAIN,
    BENCH_MODE_INT8_PLAIN,
    BENCH_MODE_STORES_OUT,
    BENCH_MODE_UDMA_OUT,
    bench_instripe_ddr_bytes,
    bench_instripe_slot_bytes,
    BENCH_MOVE_DEPTH_MAX,
    BENCH_MOVE_MAX_THREAD_BYTES,
    BENCH_STATUS_BAD_MODE,
    BENCH_STATUS_DEPTH,
    BENCH_STATUS_GEOMETRY,
    bench_array_stage,
    bench_array_stream,
    bench_case_status,
    bench_expected_arm,
    bench_geometry_status,
    bench_min_k_tiles,
    bench_move_bytes,
    bench_move_chunk_bytes,
    bench_readout_fold,
    bench_stage_chunk_bytes,
)

comptime GEN = AddressSpace.GENERIC

comptime GATE_K_TILES = 2
"""The gated chain depth, `test_hmx_standalone_bench`'s own: small enough
that the staged byte fills keep every integer accumulator inside the
saturating convert."""


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


def test_mode_table_and_arms() raises:
    """The five modes are the docstring's numbering; the arms are theirs."""
    assert_equal(BENCH_MODE_FP16_DEEP, 0)
    assert_equal(BENCH_MODE_FP16_PLAIN, 1)
    assert_equal(BENCH_MODE_INT8_PLAIN, 2)
    assert_equal(BENCH_MODE_UDMA_OUT, 3)
    assert_equal(BENCH_MODE_STORES_OUT, 4)
    assert_equal(BENCH_MODE_FP16_INSTRIPE, 5)
    assert_equal(bench_expected_arm[HMX_DTYPE](), BENCH_ARM_FP16)
    assert_equal(bench_expected_arm[HMX_INT8_DTYPE](), BENCH_ARM_W8A8)


def test_geometry_refusals() raises:
    """Every refusal branch fires, and every boundary it guards admits."""
    # An unknown mode is refused, not defaulted to a plausible arm.
    assert_equal(bench_geometry_status(-1, 384, 100), BENCH_STATUS_BAD_MODE)
    assert_equal(bench_geometry_status(6, 384, 100), BENCH_STATUS_BAD_MODE)

    # Nothing timed is nothing reported, for every mode class.
    assert_equal(
        bench_geometry_status(BENCH_MODE_FP16_PLAIN, 384, 0),
        BENCH_STATUS_GEOMETRY,
    )
    assert_equal(
        bench_geometry_status(BENCH_MODE_UDMA_OUT, 384, 0),
        BENCH_STATUS_GEOMETRY,
    )

    # The pure-array floors, at their exact boundaries (29 fp16 / 32 int8):
    # below them the geometry-derived denominator understates past 2% and
    # DRAFT-210's contradiction rows would fire on an instrument artifact.
    var fp16_floor = bench_min_k_tiles[HMX_DTYPE]()
    var int8_floor = bench_min_k_tiles[HMX_INT8_DTYPE]()
    assert_equal(
        bench_geometry_status(BENCH_MODE_FP16_PLAIN, fp16_floor - 1, 1),
        BENCH_STATUS_GEOMETRY,
    )
    assert_equal(bench_geometry_status(BENCH_MODE_FP16_PLAIN, fp16_floor, 1), 0)
    assert_equal(bench_geometry_status(BENCH_MODE_FP16_DEEP, fp16_floor, 1), 0)
    assert_equal(
        bench_geometry_status(BENCH_MODE_INT8_PLAIN, int8_floor - 1, 1),
        BENCH_STATUS_GEOMETRY,
    )
    assert_equal(bench_geometry_status(BENCH_MODE_INT8_PLAIN, int8_floor, 1), 0)

    # K=64 (2 croutons) is DRAFT-210's pre-registered refusal row, on both
    # pure-array arms.
    assert_equal(BENCH_K_SMALL // HMX_TILE, 2)
    assert_equal(
        bench_geometry_status(BENCH_MODE_FP16_PLAIN, 2, 1000),
        BENCH_STATUS_GEOMETRY,
    )
    assert_equal(
        bench_geometry_status(BENCH_MODE_INT8_PLAIN, 2, 1000),
        BENCH_STATUS_GEOMETRY,
    )

    # The witness byte word is 32 bits: the boundary where a pure-array row
    # would wrap it, both sides, both dtypes (fp16 passes are twice int8's
    # bytes, so the admitted iters differ by exactly 2x at equal tiles).
    assert_equal(bench_geometry_status(BENCH_MODE_FP16_PLAIN, 384, 2730), 0)
    assert_equal(
        bench_geometry_status(BENCH_MODE_FP16_PLAIN, 384, 2731),
        BENCH_STATUS_GEOMETRY,
    )
    assert_equal(bench_geometry_status(BENCH_MODE_INT8_PLAIN, 384, 5461), 0)
    assert_equal(
        bench_geometry_status(BENCH_MODE_INT8_PLAIN, 384, 5462),
        BENCH_STATUS_GEOMETRY,
    )

    # A movement geometry whose first feeder share is empty times nothing.
    assert_equal(
        bench_geometry_status(BENCH_MODE_UDMA_OUT, 3, 100),
        BENCH_STATUS_GEOMETRY,
    )
    assert_equal(bench_geometry_status(BENCH_MODE_UDMA_OUT, 4, 100), 0)

    # The per-participant walk cap, at its exact boundary: past it the
    # region offsets leave 32-bit addressing on the part.
    var chunk = bench_move_chunk_bytes(384)
    var cap_iters = BENCH_MOVE_MAX_THREAD_BYTES // chunk
    assert_equal(
        bench_geometry_status(BENCH_MODE_STORES_OUT, 384, cap_iters), 0
    )
    assert_equal(
        bench_geometry_status(BENCH_MODE_STORES_OUT, 384, cap_iters + 1),
        BENCH_STATUS_GEOMETRY,
    )

    # The in-stripe mode: the pure-array floor applies (its denominator is
    # the same chain), and its DDR source walk carries the movement cap at
    # the slot grain — two panels per pass, never revisited.
    assert_equal(
        bench_geometry_status(BENCH_MODE_FP16_INSTRIPE, fp16_floor - 1, 1),
        BENCH_STATUS_GEOMETRY,
    )
    assert_equal(
        bench_geometry_status(BENCH_MODE_FP16_INSTRIPE, fp16_floor, 1), 0
    )
    var slot = bench_instripe_slot_bytes(384)
    var slot_cap_iters = BENCH_MOVE_MAX_THREAD_BYTES // slot
    assert_equal(
        bench_geometry_status(BENCH_MODE_FP16_INSTRIPE, 384, slot_cap_iters),
        0,
    )
    assert_equal(
        bench_geometry_status(
            BENCH_MODE_FP16_INSTRIPE, 384, slot_cap_iters + 1
        ),
        BENCH_STATUS_GEOMETRY,
    )
    assert_equal(
        bench_instripe_ddr_bytes(384, slot_cap_iters),
        UInt64(slot) * UInt64(slot_cap_iters),
    )


def test_depth_contract() raises:
    """Window item 9's depth scalar: the composed ladder the shim runs.

    `bench_case_status` is what the device calls; every branch fires at its
    exact boundary, and geometry outranks depth so a doubly-wrong case
    refuses under the name it always did.
    """
    # Geometry still answers first, unchanged status bits.
    assert_equal(bench_case_status(-1, 384, 100, 1), BENCH_STATUS_BAD_MODE)
    assert_equal(
        bench_case_status(BENCH_MODE_UDMA_OUT, 384, 0, 1),
        BENCH_STATUS_GEOMETRY,
    )

    # The uDMA depth axis, at its boundaries: never 0; never past the walk
    # (a prime deeper than `iters` has no steady state); never past the
    # shim ring's usable depth, where pop-on-full backpressure would cap it
    # SILENTLY and the row would mislabel its arm.
    assert_equal(
        bench_case_status(BENCH_MODE_UDMA_OUT, 384, 100, 0),
        BENCH_STATUS_DEPTH,
    )
    assert_equal(bench_case_status(BENCH_MODE_UDMA_OUT, 384, 100, 1), 0)
    assert_equal(bench_case_status(BENCH_MODE_UDMA_OUT, 384, 100, 100), 0)
    assert_equal(
        bench_case_status(BENCH_MODE_UDMA_OUT, 384, 100, 101),
        BENCH_STATUS_DEPTH,
    )
    assert_equal(
        bench_case_status(BENCH_MODE_UDMA_OUT, 384, 300, BENCH_MOVE_DEPTH_MAX),
        0,
    )
    assert_equal(
        bench_case_status(
            BENCH_MODE_UDMA_OUT, 384, 300, BENCH_MOVE_DEPTH_MAX + 1
        ),
        BENCH_STATUS_DEPTH,
    )

    # The pre-registered A' sweep points all admit, and `iters == depth` --
    # the prologue-in-isolation row -- is legal by construction.
    var sweep = [2, 4, 8, 16]
    for depth in sweep:
        assert_equal(bench_case_status(BENCH_MODE_UDMA_OUT, 384, 85, depth), 0)
        assert_equal(
            bench_case_status(BENCH_MODE_UDMA_OUT, 384, depth, depth), 0
        )

    # The in-stripe pass-slot ring: 1 and 2 admit (2 = the one-outstanding
    # ceiling, order 096); 0, past-the-walk, and past the ceiling refuse.
    # `iters == depth` is the prologue-plus-drain row, legal here too.
    assert_equal(
        bench_case_status(BENCH_MODE_FP16_INSTRIPE, 384, 64, 0),
        BENCH_STATUS_DEPTH,
    )
    assert_equal(bench_case_status(BENCH_MODE_FP16_INSTRIPE, 384, 64, 1), 0)
    assert_equal(bench_case_status(BENCH_MODE_FP16_INSTRIPE, 384, 64, 2), 0)
    assert_equal(
        bench_case_status(BENCH_MODE_FP16_INSTRIPE, 384, 64, 3),
        BENCH_STATUS_DEPTH,
    )
    assert_equal(
        bench_case_status(
            BENCH_MODE_FP16_INSTRIPE, 384, 64, BENCH_INSTRIPE_DEPTH_MAX + 1
        ),
        BENCH_STATUS_DEPTH,
    )
    assert_equal(
        bench_case_status(BENCH_MODE_FP16_INSTRIPE, 384, 1, 2),
        BENCH_STATUS_DEPTH,
    )
    assert_equal(bench_case_status(BENCH_MODE_FP16_INSTRIPE, 384, 2, 2), 0)

    # No other mode has a depth axis: an ignored knob under an asked-for
    # label is the defaulted-arm class, so anything but 1 refuses.
    assert_equal(bench_case_status(BENCH_MODE_STORES_OUT, 384, 100, 1), 0)
    assert_equal(
        bench_case_status(BENCH_MODE_STORES_OUT, 384, 100, 2),
        BENCH_STATUS_DEPTH,
    )
    var fp16_floor = bench_min_k_tiles[HMX_DTYPE]()
    assert_equal(bench_case_status(BENCH_MODE_FP16_PLAIN, fp16_floor, 1, 1), 0)
    assert_equal(
        bench_case_status(BENCH_MODE_FP16_PLAIN, fp16_floor, 1, 2),
        BENCH_STATUS_DEPTH,
    )
    assert_equal(
        bench_case_status(BENCH_MODE_INT8_PLAIN, 384, 10, 0),
        BENCH_STATUS_DEPTH,
    )


def test_move_accounting() raises:
    """The movement grain is the production staging grain, and bytes multiply.
    """
    # Feeder 0's share of the K=12288 panel: the host-gate-pinned 196,608 B
    # transfer the production stripe submits — the write-side row's grain.
    assert_equal(bench_move_chunk_bytes(BENCH_K_PROD // HMX_TILE), 196608)
    assert_equal(
        bench_move_chunk_bytes(BENCH_K_PROD // HMX_TILE),
        bench_stage_chunk_bytes[HMX_DTYPE](BENCH_K_PROD // HMX_TILE, 4),
    )
    assert_equal(bench_move_chunk_bytes(32), 16384)
    assert_equal(bench_move_bytes(384, 85), 196608 * 85)
    # The off-by-one mutation: a grain blind to the panel would price the
    # wrong transfer.
    assert_true(bench_move_chunk_bytes(380) != bench_move_chunk_bytes(384))


def test_fp16_stream_vacuity() raises:
    """The staged stream drives the REAL fp16 pass body, both forms.

    The #454 check: this call chain — `bench_array_stage` then
    `bench_array_stream` — is the device shim's, so a build in which the
    pass body was dead-stripped or the routing swapped fails here on the
    fold and the echo, not on silicon.
    """
    var scratch = _Scratch[HMX_DTYPE](GATE_K_TILES)
    bench_array_stage(scratch.views, GATE_K_TILES)

    # An un-run stream echoes no arm: a refusal upstream cannot fabricate a
    # delivered pass.
    assert_equal(
        bench_array_stream[form=BENCH_FORM_DEEP, emit_hmx=False](
            scratch.views, GATE_K_TILES, 0
        ),
        0,
    )
    assert_equal(bench_readout_fold(scratch.views.out), 0)

    var arm = bench_array_stream[form=BENCH_FORM_DEEP, emit_hmx=False](
        scratch.views, GATE_K_TILES, 2
    )
    assert_equal(arm, BENCH_ARM_FP16)
    var fold = bench_readout_fold(scratch.views.out)
    assert_true(fold != 0)

    # The plain form models identically (the forms differ only in issue
    # count, which a model has none of), through the same staged state.
    var plain = _Scratch[HMX_DTYPE](GATE_K_TILES)
    bench_array_stage(plain.views, GATE_K_TILES)
    var plain_arm = bench_array_stream[form=BENCH_FORM_PLAIN, emit_hmx=False](
        plain.views, GATE_K_TILES, 1
    )
    assert_equal(plain_arm, BENCH_ARM_FP16)
    assert_equal(bench_readout_fold(plain.views.out), fold)

    # The flip control: one moved bit pattern must move the fold.
    scratch.views.out.store[1](
        Coord(0, 0, 0, 0),
        scratch.views.out.load[1](Coord(0, 0, 0, 0)) + Scalar[HMX_DTYPE](0.25),
    )
    assert_true(bench_readout_fold(scratch.views.out) != fold)


def test_int8_stream_vacuity_and_mutant() raises:
    """The staged stream drives the REAL integer pass body, `/512` included."""
    var scratch = _Scratch[HMX_INT8_DTYPE](GATE_K_TILES)
    bench_array_stage(scratch.views, GATE_K_TILES)

    # Staging programmed the INTEGER identity (512.0 = 0x6000), not fp16's
    # 1.0: the constant whose absence made every pre-`/512` probe
    # unfalsifiable.
    assert_equal(
        scratch.views.record.load[1](Coord(0, 0)) & 0xFFFF,
        Scalar[DType.uint32](0x6000),
    )

    var arm = bench_array_stream[form=BENCH_FORM_PLAIN, emit_hmx=False](
        scratch.views, GATE_K_TILES, 2
    )
    assert_equal(arm, BENCH_ARM_W8A8)
    assert_true(bench_readout_fold(scratch.views.out) != 0)

    # THE MUTANT THE HISTORY DEMANDS, at the selector's own level: re-program
    # the fp16 habit over the staged record and the convert's `/512` floors
    # every accumulator this fill can reach to zero — the all-zero autopsy as
    # a control that must fire through the same call chain the device runs.
    var mutant = _Scratch[HMX_INT8_DTYPE](GATE_K_TILES)
    bench_array_stage(mutant.views, GATE_K_TILES)
    hmx_int8_program_record(
        mutant.views.record, Scalar[DType.float16](1), Int32(0)
    )
    _ = bench_array_stream[form=BENCH_FORM_PLAIN, emit_hmx=False](
        mutant.views, GATE_K_TILES, 1
    )
    assert_equal(bench_readout_fold(mutant.views.out), 0)


def test_int8_store_region_cannot_reach_the_record() raises:
    """Pass `N`'s store must not feed pass `N + 1`'s requant: order 210 leg 3.

    The device evidence this encodes: the emitted int8 store's destination,
    extent and scale all traced correct, the store measurably executed
    (91 pcycles/pass, flat in `k_tiles`), and the fold was still zero. The
    mechanism was the CARVE, not the pointer — the readout store owns
    `HMX_STORE_REGION_BYTES` forward of `out` (the vendor's captured `sat.ub`
    reference spans 2048 bytes; order 132 measured the reach on the stripe),
    the old carve put the record one int8 tile in, and the integer arm
    re-reads the record every pass without rewriting it. So pass 1's store
    zeroed pass 2's scale and every later readout requanted to zero:
    `sink=0x00000000`, deterministic, at every depth.

    Stamping only the first `HMX_RECORD_BYTES` past the logical tile keeps
    the write inside the carve under BOTH layouts: on the defective carve
    those bytes ARE the record (the failure fires through the real pass
    body), on the fixed one they are the store's own slack.
    """
    var scratch = _Scratch[HMX_INT8_DTYPE](GATE_K_TILES)
    bench_array_stage(scratch.views, GATE_K_TILES)
    _ = bench_array_stream[form=BENCH_FORM_PLAIN, emit_hmx=False](
        scratch.views, GATE_K_TILES, 1
    )
    var fold = bench_readout_fold(scratch.views.out)
    assert_true(fold != 0)

    # What the silicon store writes past the tile, converged: zeros (a
    # requant file of zeros produces q = 0 for any accumulator).
    var out_base = Int(scratch.views.out.ptr_at_offset(Coord(0, 0, 0, 0)))
    var region = UnsafePointer[UInt8, MutUntrackedOrigin](
        unsafe_from_address=out_base
    )
    comptime TILE = HmxTileBytes[HMX_INT8_DTYPE]
    for i in range(TILE, TILE + HMX_RECORD_BYTES):
        region.store(i, 0)

    # The same staged stream again, through the shim's own call chain: a
    # record inside the store's region reads back as scale 0 and folds 0
    # here exactly as it did on silicon.
    _ = bench_array_stream[form=BENCH_FORM_PLAIN, emit_hmx=False](
        scratch.views, GATE_K_TILES, 1
    )
    assert_equal(bench_readout_fold(scratch.views.out), fold)

    # The invariant itself, on the real carve.
    var rec_base = Int(scratch.views.record.ptr_at_offset(Coord(0, 0)))
    assert_true(rec_base - out_base >= HMX_STORE_REGION_BYTES)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
