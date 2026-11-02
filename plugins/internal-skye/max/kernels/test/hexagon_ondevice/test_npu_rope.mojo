"""Off-device validation for the HVX RoPE kernel (`hexagon_ondevice.rope_kernel`).

Drives the kernel's pure math helper `rope_apply_row` on the host -- where the
same source lowers to host SIMD instead of HVX `V6_*` -- and compares it against
an independent, fully scalar reference of the identical rotary transform. The
two implementations differ structurally (vectorized `deinterleave` load/store
vs. an element-by-element scalar loop), so agreement is evidence the vectorized
algorithm is correct, not a tautology.

Covered: both rotary conventions (HF `rotate_half` and interleaved), fp16 and
bf16 I/O, fp16 and fp32 cos/sin, and head dims that exercise the multi-vector
body, the single-vector case, and the scalar tail (half not a multiple of the
64-lane fp16 width). RoPE math runs in fp32 in both, so the fp16/bf16 outputs
are expected bit-identical; the gate tolerance is far tighter than any real
convention/pairing bug would produce.

On-device (physical NPU / Hexagon simulator) confirmation of the actual HVX
instruction stream is still required; see the module docstring of the kernel.
"""

from layout.coord import All, Coord
from layout.tile_layout import row_major

from std.math import cos, sin
from std.memory import UnsafePointer, alloc
from std.sys import size_of
from std.testing import assert_true

from hexagon_ondevice.rope_kernel import (
    RowMajor1D,
    RowMajor2D,
    _thread_rows,
    rope_apply_row,
    rope_striped,
)


@always_inline
def _rope_reference_row[
    dtype: DType,
    freq_dtype: DType,
    //,
    interleaved: Bool,
](
    x: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin],
    x_off: Int,
    dst: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin],
    dst_off: Int,
    freqs: UnsafePointer[Scalar[freq_dtype], MutUnsafeAnyOrigin],
    freq_row: Int,
    half: Int,
):
    """Scalar fp32 reference for one head vector -- the same math, no vectors.
    """
    comptime if interleaved:
        for p in range(half):
            var c = Float32(freqs[freq_row + 2 * p])
            var s = Float32(freqs[freq_row + 2 * p + 1])
            var re = Float32(x[x_off + 2 * p])
            var im = Float32(x[x_off + 2 * p + 1])
            dst[dst_off + 2 * p] = Scalar[dtype](re * c - im * s)
            dst[dst_off + 2 * p + 1] = Scalar[dtype](re * s + im * c)
    else:
        for i in range(half):
            var c = Float32(freqs[freq_row + 2 * i])
            var s = Float32(freqs[freq_row + 2 * i + 1])
            var a = Float32(x[x_off + i])
            var b = Float32(x[x_off + i + half])
            dst[dst_off + i] = Scalar[dtype](a * c - b * s)
            dst[dst_off + i + half] = Scalar[dtype](a * s + b * c)


def _run_case[
    dtype: DType,
    freq_dtype: DType,
    interleaved: Bool,
](num_rows: Int, head_size: Int, seed: UInt64) raises:
    """Ropes `num_rows` head vectors both ways and asserts they agree."""
    comptime W = 128 // size_of[dtype]()  # matches the kernel's IO_VW
    var half = head_size // 2
    var n = num_rows * head_size

    var x = alloc[Scalar[dtype]](n)
    var got = alloc[Scalar[dtype]](n)
    var expect = alloc[Scalar[dtype]](n)
    var freqs = alloc[Scalar[freq_dtype]](num_rows * head_size)
    var positions = alloc[Scalar[DType.uint32]](num_rows)

    # Bounded deterministic inputs in [-1, 1] via a 64-bit LCG.
    var state = seed
    for i in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32((state >> 33) & 0xFFFF) / 65535.0
        x[i] = Scalar[dtype](2.0 * u - 1.0)

    for r in range(num_rows):
        positions[r] = UInt32(r)

    # cos/sin interleaved per pair. The angle schedule is arbitrary (both paths
    # read the same buffer); only bounded magnitudes matter here.
    for pos in range(num_rows):
        for p in range(half):
            var angle = Float32(pos) * 0.5 / (1.0 + Float32(p))
            freqs[pos * head_size + 2 * p] = Scalar[freq_dtype](cos(angle))
            freqs[pos * head_size + 2 * p + 1] = Scalar[freq_dtype](sin(angle))

    var xp = x.as_unsafe_any_origin()
    var gp = got.as_unsafe_any_origin()
    var ep = expect.as_unsafe_any_origin()
    var fp = freqs.as_unsafe_any_origin()

    # The kernel is driven through the tile model it ships as; the reference
    # stays on raw pointers with its own hand arithmetic, which is what keeps
    # the two implementations structurally independent.
    var heads = row_major(Coord(num_rows, head_size))
    var x_tile = RowMajor2D[dtype](ptr=xp, layout=heads)
    var got_tile = RowMajor2D[dtype](ptr=gp, layout=heads)
    var freq_tile = RowMajor2D[freq_dtype](ptr=fp, layout=heads)

    for r in range(num_rows):
        var fr = r * head_size  # positions[r] == r
        rope_apply_row[interleaved=interleaved, W=W](
            x_tile.slice(r, All),
            got_tile.slice(r, All),
            freq_tile.slice(r, All),
        )
        _rope_reference_row[interleaved=interleaved](
            xp, r * head_size, ep, r * head_size, fp, fr, half
        )

    var max_err = Float32(0)
    for i in range(n):
        var d = Float32(got[i]) - Float32(expect[i])
        var ad = -d if d < 0 else d
        if ad > max_err:
            max_err = ad

    print(
        "case rows=",
        num_rows,
        " head_size=",
        head_size,
        " W=",
        W,
        " interleaved=",
        interleaved,
        " io=",
        dtype,
        " freq=",
        freq_dtype,
        " -> max_err=",
        max_err,
        sep="",
    )
    assert_true(
        max_err <= 1e-3,
        "RoPE kernel diverged from the scalar reference",
    )

    x.free()
    got.free()
    expect.free()
    freqs.free()
    positions.free()


def _run_striped_case[
    dtype: DType,
    interleaved: Bool,
](num_rows: Int, head_size: Int, seed: UInt64) raises:
    """The four compute threads' runs must rebuild the whole table exactly.

    Both arms run `rope_apply_row` over the same inputs, so any difference is a
    partition or an indirection bug and the comparison is BIT-exact rather than
    the tolerance `_run_case` needs against its independent reference. The
    sentinel pre-fill makes an uncovered row fail instead of reading as a
    plausible zero, and the launch index runs to 4 -- the DMA thread, which must
    write nothing. `positions` is a non-identity permutation so a striped arm
    that reached the wrong freqs row could not agree by accident.
    """
    comptime W = 128 // size_of[dtype]()
    comptime SENTINEL = Scalar[dtype](-3.0)
    var n = num_rows * head_size
    var half = head_size // 2

    var x = alloc[Scalar[dtype]](n)
    var whole = alloc[Scalar[dtype]](n)
    var striped = alloc[Scalar[dtype]](n)
    var freqs = alloc[Scalar[DType.float32]](num_rows * head_size)
    var positions = alloc[Scalar[DType.uint32]](num_rows)

    var state = seed
    for i in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32((state >> 33) & 0xFFFF) / 65535.0
        x[i] = Scalar[dtype](2.0 * u - 1.0)
        whole[i] = SENTINEL
        striped[i] = SENTINEL

    for r in range(num_rows):
        positions[r] = UInt32((r * 37 + 5) % num_rows)

    for pos in range(num_rows):
        for p in range(half):
            var angle = Float32(pos) * 0.5 / (1.0 + Float32(p))
            freqs[pos * head_size + 2 * p] = Float32(cos(angle))
            freqs[pos * head_size + 2 * p + 1] = Float32(sin(angle))

    var heads = row_major(Coord(num_rows, head_size))
    var x_tile = RowMajor2D[dtype](ptr=x.as_unsafe_any_origin(), layout=heads)
    var whole_tile = RowMajor2D[dtype](
        ptr=whole.as_unsafe_any_origin(), layout=heads
    )
    var striped_tile = RowMajor2D[dtype](
        ptr=striped.as_unsafe_any_origin(), layout=heads
    )
    var freq_tile = RowMajor2D[DType.float32](
        ptr=freqs.as_unsafe_any_origin(), layout=heads
    )
    var pos_tile = RowMajor1D[DType.uint32](
        ptr=positions.as_unsafe_any_origin(),
        layout=row_major(Coord(num_rows)),
    )

    for r in range(num_rows):
        rope_apply_row[interleaved=interleaved, W=W](
            x_tile.slice(r, All),
            whole_tile.slice(r, All),
            freq_tile.slice(Int(positions[r]), All),
        )

    for t in range(5):
        rope_striped[dtype, interleaved=interleaved](
            striped_tile, x_tile, freq_tile, pos_tile, UInt32(t)
        )

    for i in range(n):
        assert_true(
            striped[i] == whole[i],
            String(
                "striped != whole at ",
                i,
                " rows=",
                num_rows,
                " head_size=",
                head_size,
            ),
        )
        assert_true(
            striped[i] != SENTINEL,
            String("row element ", i, " left unwritten, rows=", num_rows),
        )

    x.free()
    whole.free()
    striped.free()
    freqs.free()
    positions.free()


def _check_thread_rows() raises:
    """Every row belongs to exactly one compute thread, and the DMA thread owns
    none. Scored by covering count, so a split bug localizes here rather than
    being read off a numeric mismatch."""
    for rows in range(0, 40):
        var covered = alloc[Scalar[DType.int32]](rows + 1)
        for i in range(rows):
            covered[i] = 0
        for t in range(5):
            var run = _thread_rows(rows, UInt32(t))
            assert_true(run[0] <= run[1], "thread run is inverted")
            if t >= 4:
                assert_true(run[0] == run[1], "the DMA thread owns rows")
            for r in range(run[0], run[1]):
                assert_true(r < rows, "a run reaches past the last row")
                covered[r] += 1
        for r in range(rows):
            assert_true(
                covered[r] == 1,
                String("rows=", rows, " row ", r, " is not covered once"),
            )
        covered.free()


def main() raises:
    # HF rotate_half (Gemma4 production path), fp16 I/O + fp32 cos/sin:
    #   128 -> half 64  (one full vector, no tail)
    #   256 -> half 128 (two vectors, no tail)
    #    64 -> half 32  (< W: pure scalar tail)
    #    96 -> half 48  (< W: pure scalar tail)
    #   160 -> half 80  (one vector + 16-element scalar tail)
    _run_case[DType.float16, DType.float32, interleaved=False](8, 128, 1)
    _run_case[DType.float16, DType.float32, interleaved=False](4, 256, 2)
    _run_case[DType.float16, DType.float32, interleaved=False](8, 64, 3)
    _run_case[DType.float16, DType.float32, interleaved=False](8, 96, 4)
    _run_case[DType.float16, DType.float32, interleaved=False](8, 160, 5)

    # fp16 cos/sin.
    _run_case[DType.float16, DType.float16, interleaved=False](8, 160, 6)

    # bf16 I/O.
    _run_case[DType.bfloat16, DType.float32, interleaved=False](8, 160, 7)

    # Interleaved convention.
    _run_case[DType.float16, DType.float32, interleaved=True](8, 128, 8)
    _run_case[DType.float16, DType.float32, interleaved=True](8, 160, 9)

    # The launch split, at both dtypes and both conventions. Row counts straddle
    # the thread count: below it, dividing it, and not. The bf16 arms are what
    # score the dtype the entries are now parameterized on -- `_run_case` above
    # reaches `rope_apply_row` directly and never runs the split.
    _check_thread_rows()
    for rows in [1, 3, 4, 7, 8, 13]:
        _run_striped_case[DType.float16, interleaved=False](rows, 160, 20)
        _run_striped_case[DType.float16, interleaved=True](rows, 128, 21)
        _run_striped_case[DType.bfloat16, interleaved=False](rows, 160, 22)
        _run_striped_case[DType.bfloat16, interleaved=True](rows, 128, 23)

    print("ALL ROPE ALGORITHM TESTS PASSED")
