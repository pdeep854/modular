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
"""The rung-1 microbench, run on the host with the modelled MMA
(`hmx_annihilation_bench.mojo`; design section 9).

Drives all six stage-arm bodies the device order will dispatch — baseline
FF -> swiglu -> FFO and native FF' -> swiglu' -> FFO' — through the same
step-driver functions the device shim will call, and asserts the
pre-registered gate: the two arms' final outputs are BIT-IDENTICAL. What the
host cannot read is the phase table (`HMX_PHASE_PROBE` is a device build
define, off here, so every mark compiles away); what it proves is everything
else the device run will lean on — the arm configuration, the fills, the
buffer geometry, the crouton handoff between stages, and the verdict.

Structural guards, so a pass constrains what it claims to:

  - The two arms' output buffers are seeded with DIFFERENT sentinels, so a
    region left unwritten by both arms cannot compare equal.
  - The native arm must REPORT itself: the FF' dispatch's unpack seam echoes
    `HMX_UNPACK_ARM_NATIVE` and the baseline's echoes `SCALAR_OFF` — the
    treatment-delivery fingerprint, so a flag that silently failed to
    instantiate reads as a FAIL rather than as a clean null.
  - Per-stage handoff compares localize a failure to the stage that made it
    before the end-to-end verdict pools everything.
  - The verdict must be shown ABLE to fail: one flipped element must flip it.
"""
from std.math import align_up
from std.memory import AddressSpace, alloc
from std.testing import assert_equal, assert_true, TestSuite

from layout.coord import Coord
from layout.tile_layout import row_major

from hexagon_ondevice.hmx_annihilation_bench import (
    ANNI_ARM_BASELINE,
    ANNI_ARM_NATIVE,
    ANNI_DIM,
    ANNI_M,
    ANNI_MARKER_DONE,
    ANNI_SCRATCH_BYTES,
    ANNI_SEED_C_BASELINE,
    ANNI_SEED_C_NATIVE,
    ANNI_SEED_INTERMEDIATE,
    ANNI_STAGE_FF,
    ANNI_STAGE_FFO,
    ANNI_STAGE_SWIGLU,
    ANNI_VERDICT_UNSET,
    ANNI_WIT_ARM,
    ANNI_WIT_MARKER,
    ANNI_WIT_STAGE,
    ANNI_WIT_VERDICT,
    ANNI_WIT_WORDS,
    annihilation_a_mod,
    annihilation_c_out,
    annihilation_ff_out,
    annihilation_fill_activation,
    annihilation_fill_weight,
    annihilation_matmul_step,
    annihilation_mlp_out,
    annihilation_shim_verdict,
    annihilation_stage_prepare,
    annihilation_swiglu_thread,
    annihilation_verdict,
    annihilation_w_ff,
    annihilation_w_ffo,
)
from hexagon_ondevice.hmx_crouton import (
    HMX_DTYPE,
    HMX_SPLIT,
    HMX_TILE,
    HMX_TILE_BYTES,
    HMX_UNPACK_ARM_NATIVE,
    HMX_UNPACK_ARM_SCALAR_OFF,
    CroutonPanel,
    HmxRows2D,
    hmx_native_row_croutons,
    hmx_panel,
    hmx_repack_rows_rpm,
    hmx_stripe_arena_bytes,
    hmx_stripe_arena_in,
)
from hexagon_ondevice.matmul_kernel import MMTile, hmx_stripe_plan

comptime GEN = AddressSpace.GENERIC

comptime FEEDERS = 4
"""The launch's compute threads; the fifth participant is the array."""

comptime ACT_SLOTS = 2
"""A-block panels resident at once, shared by every bench matmul."""

comptime RawPtr = UnsafePointer[Scalar[HMX_DTYPE], MutUntrackedOrigin]


def _alloc_fp16(elems: Int, seed: Scalar[HMX_DTYPE]) -> RawPtr:
    var buf = alloc[Scalar[HMX_DTYPE]](elems)
    for i in range(elems):
        buf[i] = seed
    return buf


def _panel_alloc(tiles: Int) -> Tuple[RawPtr, CroutonPanel[GEN]]:
    """A crouton panel over an aligned allocation; caller frees the raw
    pointer at the end of its function (`test_swiglu_crouton.mojo`'s rule)."""
    var elems = tiles * HMX_TILE * HMX_TILE
    var raw = alloc[Scalar[HMX_DTYPE]](elems + HMX_TILE_BYTES)
    var base = RawPtr(unsafe_from_address=align_up(Int(raw), HMX_TILE_BYTES))
    for i in range(elems):
        base[i] = Scalar[HMX_DTYPE](-4096.0)
    return (raw, hmx_panel[GEN](base, tiles))


def _rows(ptr: RawPtr, rows: Int, cols: Int) -> HmxRows2D:
    return HmxRows2D(
        ptr=ptr.as_unsafe_any_origin(), layout=row_major(Coord(rows, cols))
    )


def _mm(ptr: RawPtr, rows: Int, cols: Int) -> MMTile:
    return MMTile(
        ptr=ptr.as_unsafe_any_origin(), layout=row_major(Coord(rows, cols))
    )


def _run_matmul[
    native_a: Bool, native_c: Bool
](C: MMTile, A: MMTile, B: MMTile, M: Int, N: Int, K: Int) -> UInt32:
    """One bench matmul, every step and every thread, host order.

    The device shim's loop with the rendezvous replaced by sequential thread
    order — legal because the twin's threads are a loop, the
    `pipelined_mismatches` precedent.
    """
    var arena_bytes = hmx_stripe_arena_bytes(K // HMX_TILE, ACT_SLOTS)
    var raw = alloc[UInt8](arena_bytes + HMX_TILE_BYTES)
    var base = UnsafePointer[UInt8, MutUntrackedOrigin](
        unsafe_from_address=align_up(Int(raw), HMX_TILE_BYTES)
    )
    var plan = hmx_stripe_plan(M, N, K, arena_bytes)
    var arena = hmx_stripe_arena_in[GEN](base, plan.k_tiles, plan.act_slots)
    var unpack_arm = HMX_UNPACK_ARM_SCALAR_OFF
    for step in range(plan.steps()):
        for t in range(FEEDERS + 1):
            annihilation_matmul_step[native_a, native_c](
                C, A, B, arena, plan, step, t, FEEDERS, unpack_arm
            )
    raw.free()
    return unpack_arm


def _croutons_equal(
    got: CroutonPanel[GEN], want: CroutonPanel[GEN], tiles: Int
) -> Int:
    var wrong = 0
    for t in range(tiles):
        for s in range(HMX_TILE // HMX_SPLIT):
            for lane in range(HMX_TILE):
                for sub in range(HMX_SPLIT):
                    if got.load[1, alignment=2](
                        Coord(t, s, lane, sub)
                    ) != want.load[1, alignment=2](Coord(t, s, lane, sub)):
                        wrong += 1
    return wrong


def _chain_case(M: Int, dim: Int, hidden: Int) raises:
    """Both arms of the chain at one geometry, every gate asserted.

    Failure here is the ladder's stop condition, caught for free on the host
    before any device order is written.
    """
    var ff_n = 2 * hidden

    # Shared inputs: one modulated activation, two salted weights.
    var a_mod = _alloc_fp16(M * dim, 0)
    annihilation_fill_activation(_rows(a_mod, M, dim), M, dim)
    var w_ff = _alloc_fp16(ff_n * dim, 0)
    annihilation_fill_weight(_rows(w_ff, ff_n, dim), ff_n, dim, 1)
    var w_ffo = _alloc_fp16(dim * hidden, 0)
    annihilation_fill_weight(_rows(w_ffo, dim, hidden), dim, hidden, 2)

    # Baseline arm: row-major intermediates, different C sentinel from the
    # native arm's so a mutually-unwritten region cannot compare equal.
    var ff_base = _alloc_fp16(M * ff_n, -4096.0)
    var mlp_base = _alloc_fp16(M * hidden, -4096.0)
    var c_base = _alloc_fp16(M * dim, -4096.0)

    var base_ff_arm = _run_matmul[False, False](
        _mm(ff_base, M, ff_n),
        _mm(a_mod, M, dim),
        _mm(w_ff, ff_n, dim),
        M,
        ff_n,
        dim,
    )
    var dead_in = _panel_alloc(1)
    var dead_out = _panel_alloc(1)
    for t in range(FEEDERS):
        annihilation_swiglu_thread[False](
            _rows(ff_base, M, ff_n),
            _rows(mlp_base, M, hidden),
            dead_in[1],
            dead_out[1],
            M,
            hidden,
            t,
            FEEDERS,
        )
    var base_ffo_arm = _run_matmul[False, False](
        _mm(c_base, M, dim),
        _mm(mlp_base, M, hidden),
        _mm(w_ffo, dim, hidden),
        M,
        dim,
        hidden,
    )

    # Native arm: crouton intermediates. The FF' output panel IS the swiglu
    # input panel IS nothing else — one buffer, zero transforms.
    var ff_tiles = hmx_native_row_croutons(M, ff_n)
    var mlp_tiles = hmx_native_row_croutons(M, hidden)
    var ff_nat = _panel_alloc(ff_tiles)
    var mlp_nat = _panel_alloc(mlp_tiles)
    var c_nat = _alloc_fp16(M * dim, 4096.0)

    var ff_nat_base = ff_nat[1].ptr_at_offset(Coord(0, 0, 0, 0))
    var mlp_nat_base = mlp_nat[1].ptr_at_offset(Coord(0, 0, 0, 0))
    var nat_ff_arm = _run_matmul[False, True](
        _mm(ff_nat_base, M, ff_n),
        _mm(a_mod, M, dim),
        _mm(w_ff, ff_n, dim),
        M,
        ff_n,
        dim,
    )
    var dead_rows = _rows(c_nat, M, dim)
    for t in range(FEEDERS):
        annihilation_swiglu_thread[True](
            dead_rows,
            dead_rows,
            ff_nat[1],
            mlp_nat[1],
            M,
            hidden,
            t,
            FEEDERS,
        )
    var nat_ffo_arm = _run_matmul[True, False](
        _mm(c_nat, M, dim),
        _mm(mlp_nat_base, M, hidden),
        _mm(w_ffo, dim, hidden),
        M,
        dim,
        hidden,
    )

    # The treatment-delivery fingerprint: each arm names itself through the
    # unpack seam, so a flag that failed to instantiate is a FAIL, not a null.
    assert_equal(base_ff_arm, HMX_UNPACK_ARM_SCALAR_OFF)
    assert_equal(base_ffo_arm, HMX_UNPACK_ARM_SCALAR_OFF)
    assert_equal(nat_ff_arm, HMX_UNPACK_ARM_NATIVE)
    assert_equal(nat_ffo_arm, HMX_UNPACK_ARM_SCALAR_OFF)

    # Per-stage handoffs, so a failure names its stage: the native FF' output
    # must be the baseline FF output repacked, and the native swiglu output
    # the baseline swiglu output repacked.
    var ff_ref = _panel_alloc(ff_tiles)
    hmx_repack_rows_rpm[GEN](ff_ref[1], _rows(ff_base, M, ff_n), M, ff_n)
    assert_equal(_croutons_equal(ff_nat[1], ff_ref[1], ff_tiles), 0)
    var mlp_ref = _panel_alloc(mlp_tiles)
    hmx_repack_rows_rpm[GEN](mlp_ref[1], _rows(mlp_base, M, hidden), M, hidden)
    assert_equal(_croutons_equal(mlp_nat[1], mlp_ref[1], mlp_tiles), 0)

    # THE VERDICT, and its own falsifier: it must be zero, and it must be
    # shown able to be non-zero.
    assert_equal(
        annihilation_verdict(
            _rows(c_base, M, dim), _rows(c_nat, M, dim), M, dim
        ),
        0,
    )
    c_nat[0] = c_nat[0] + Scalar[HMX_DTYPE](1.0)
    assert_true(
        annihilation_verdict(
            _rows(c_base, M, dim), _rows(c_nat, M, dim), M, dim
        )
        != 0
    )

    a_mod.free()
    w_ff.free()
    w_ffo.free()
    ff_base.free()
    mlp_base.free()
    c_base.free()
    ff_nat[0].free()
    mlp_nat[0].free()
    c_nat.free()
    ff_ref[0].free()
    mlp_ref[0].free()
    dead_in[0].free()
    dead_out[0].free()


def test_shim_carve_and_prepare() raises:
    """The device shim's own carve and per-stage fills, on the host.

    Drives `annihilation_carve` and `annihilation_stage_prepare` — the exact
    functions `mojo_hmx_annihilation_bench` calls — over a heap stand-in for
    the device scratch, and spot-checks that each stage's fills and sentinels
    land in their own sections: the shared inputs get the module's fills, the
    two verdict buffers get DIFFERENT sentinels, and preparing one arm's
    section leaves the other arm's alone.
    """
    var raw = alloc[UInt8](ANNI_SCRATCH_BYTES)
    var base = align_up(Int(raw), HMX_TILE_BYTES)

    annihilation_stage_prepare(base, ANNI_ARM_BASELINE, ANNI_STAGE_FF)
    # (5*0 + 7*1) % 13 - 6 = 1, scaled by 1/64.
    assert_equal(
        annihilation_a_mod(base).load[1, alignment=2](Coord(0, 1)),
        Scalar[HMX_DTYPE](0.015625),
    )
    # (3*1 + 11*0 + 1) % 7 - 3 = 1: salt 1 in the FF weight.
    assert_equal(
        annihilation_w_ff(base).load[1, alignment=2](Coord(1, 0)),
        Scalar[HMX_DTYPE](1.0),
    )
    assert_equal(
        annihilation_ff_out(base, ANNI_ARM_BASELINE).load[1, alignment=2](
            Coord(0, 0)
        ),
        ANNI_SEED_INTERMEDIATE,
    )

    annihilation_stage_prepare(base, ANNI_ARM_NATIVE, ANNI_STAGE_SWIGLU)
    assert_equal(
        annihilation_mlp_out(base, ANNI_ARM_NATIVE).load[1, alignment=2](
            Coord(0, 0)
        ),
        ANNI_SEED_INTERMEDIATE,
    )

    annihilation_stage_prepare(base, ANNI_ARM_BASELINE, ANNI_STAGE_FFO)
    annihilation_stage_prepare(base, ANNI_ARM_NATIVE, ANNI_STAGE_FFO)
    # (3*0 + 11*0 + 2) % 7 - 3 = -1: salt 2 in the FFO weight.
    assert_equal(
        annihilation_w_ffo(base).load[1, alignment=2](Coord(0, 0)),
        Scalar[HMX_DTYPE](-1.0),
    )
    assert_equal(
        annihilation_c_out(base, ANNI_ARM_BASELINE).load[1, alignment=2](
            Coord(0, 0)
        ),
        ANNI_SEED_C_BASELINE,
    )
    assert_equal(
        annihilation_c_out(base, ANNI_ARM_NATIVE).load[1, alignment=2](
            Coord(0, 0)
        ),
        ANNI_SEED_C_NATIVE,
    )
    # The FF section was not disturbed by any later stage's prepare.
    assert_equal(
        annihilation_ff_out(base, ANNI_ARM_BASELINE).load[1, alignment=2](
            Coord(0, 0)
        ),
        ANNI_SEED_INTERMEDIATE,
    )

    raw.free()


def test_shim_verdict_path_must_fire() raises:
    """The shim's verdict path, mutation-tested: the planted mismatch MUST
    reach the witness word, and the plant must restore.

    `annihilation_shim_verdict` is the one function the device entry's
    verdict and falsify stages call; a zero it writes is only evidence
    because this test shows the same path writing a one under a planted
    single-bit flip — and writing zero again after the restore.
    """
    var raw = alloc[UInt8](ANNI_SCRATCH_BYTES)
    var base = align_up(Int(raw), HMX_TILE_BYTES)
    var wit = alloc[Scalar[DType.uint32]](ANNI_WIT_WORDS)
    for i in range(ANNI_WIT_WORDS):
        wit[i] = 0xDEADDEAD
    var witness = wit.as_unsafe_any_origin()

    # Different sentinels: a mutually-unwritten region reads as ALL wrong,
    # never as a clean pass.
    annihilation_stage_prepare(base, ANNI_ARM_BASELINE, ANNI_STAGE_FFO)
    annihilation_stage_prepare(base, ANNI_ARM_NATIVE, ANNI_STAGE_FFO)
    annihilation_shim_verdict(base, witness, False)
    assert_equal(Int(wit[ANNI_WIT_VERDICT]), ANNI_M * ANNI_DIM)

    # Identical contents: the pre-registered gate's zero.
    annihilation_fill_activation(
        annihilation_c_out(base, ANNI_ARM_BASELINE), ANNI_M, ANNI_DIM
    )
    annihilation_fill_activation(
        annihilation_c_out(base, ANNI_ARM_NATIVE), ANNI_M, ANNI_DIM
    )
    annihilation_shim_verdict(base, witness, False)
    assert_equal(Int(wit[ANNI_WIT_VERDICT]), 0)

    # THE MUST-FIRE CONTROL: one planted bit flips the verdict to exactly 1.
    annihilation_shim_verdict(base, witness, True)
    assert_equal(Int(wit[ANNI_WIT_VERDICT]), 1)

    # And the plant restored the exact bits.
    annihilation_shim_verdict(base, witness, False)
    assert_equal(Int(wit[ANNI_WIT_VERDICT]), 0)

    wit.free()
    raw.free()


def test_shim_protocol_constants_pin_the_harness() raises:
    """Tripwire for the deliberate restatement in the host driver.

    `npu_hmx_matmul_dispatch.cpp` mode `anni` restates these values rather
    than sharing them (the selector-refusal test's rule); whoever moves one
    here is told, by this failure, to move the harness's copy too.
    """
    assert_equal(ANNI_SCRATCH_BYTES, 7473152)
    assert_equal(ANNI_WIT_MARKER, 25)
    assert_equal(ANNI_WIT_ARM, 26)
    assert_equal(ANNI_WIT_STAGE, 27)
    assert_equal(ANNI_WIT_VERDICT, 28)
    assert_equal(Int(ANNI_MARKER_DONE), 0x414E4931)
    assert_equal(Int(ANNI_VERDICT_UNSET), 0xFFFFFFFF)
    assert_equal(Int(HMX_UNPACK_ARM_NATIVE), 5)


def test_rung1_chain_bit_identical() raises:
    """The chain at the harness geometry: every structural feature of the
    klein_mini run — two row blocks over `ACT_SLOTS`, ring-alternating
    column tiles on both matmuls, an uneven k-share on the FFO side, and a
    multi-tile swiglu — at a size the modelled MMA finishes in seconds.

    The device bench runs `ANNI_M x ANNI_DIM x ANNI_HIDDEN` (klein_mini
    itself, the module's constants): the real array makes that geometry
    cheap on silicon where the host's modelled MMA makes it 8 minutes here.
    The klein_mini swiglu shape is pinned host-side by
    `test_swiglu_crouton.mojo`, and the native matmul arms are
    shape-independent (`test_hmx_crouton.mojo` covers the uneven-share and
    deep-chunk shapes), so nothing klein_mini-specific goes unproven by the
    reduction.
    """
    _chain_case(64, 128, 192)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
