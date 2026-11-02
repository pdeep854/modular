"""Host validation for the HVX softmax kernel.

Runs the actual kernel (`softmax_2d` from `softmax_kernel.mojo`) on the host CPU
against an independent scalar fp32 reference, across the shapes and input
patterns decode softmax has to survive: all-equal rows (stresses the sum),
large-magnitude values (stresses the max-subtract), negatives, near one-hot,
and row lengths that are not a multiple of the vector width (stresses the tail).

The kernel source is target-independent Mojo; only its lowering differs between
host and Hexagon. Matching the reference here proves the *algorithm*. On-device
numerics (chiefly the exp primitive and the fp16<->fp32 conversion path under
`+hvx-qfloat`) still need hardware confirmation.
"""

from layout.coord import Coord
from layout.tile_layout import row_major
from std.math import exp
from std.memory import alloc
from std.utils.numerics import min_or_neg_inf

from hexagon_ondevice.softmax_kernel import (
    RowMajor2D,
    softmax_2d,
    softmax_striped,
)

comptime F32 = DType.float32


def fill_pattern[
    in_dtype: DType
](
    ptr: UnsafePointer[Scalar[in_dtype], MutUnsafeAnyOrigin],
    num_rows: Int,
    row_length: Int,
    pattern: Int,
):
    """Writes a deterministic input row pattern in fp32, rounded to `in_dtype`.
    """
    for r in range(num_rows):
        for j in range(row_length):
            var x: Float32
            if pattern == 0:  # all-equal: softmax -> uniform 1/N
                x = 3.0
            elif (
                pattern == 1
            ):  # large positive magnitudes: exp overflows w/o max-subtract
                x = 60.0 + Float32(j % 41)
            elif pattern == 2:  # large negatives
                x = -80.0 - Float32(j % 37)
            elif pattern == 3:  # ascending ramp (mixed sign)
                x = -5.0 + Float32(j) * 0.01
            elif pattern == 4:  # near one-hot: single spike per row
                x = 20.0 if (j == r % row_length) else 0.0
            else:  # deterministic pseudo-random in [-6, 6)
                x = (
                    Float32(Int((j * 2654435761 + r * 40503) % 1200)) * 0.01
                    - 6.0
                )
            ptr[r * row_length + j] = Scalar[in_dtype](x)


def reference_softmax[
    in_dtype: DType
](
    input: UnsafePointer[Scalar[in_dtype], MutUnsafeAnyOrigin],
    output: UnsafePointer[Scalar[in_dtype], MutUnsafeAnyOrigin],
    num_rows: Int,
    row_length: Int,
):
    """Independent scalar oracle: stable softmax in fp32, output rounded to `in_dtype`.
    """
    for r in range(num_rows):
        var base = r * row_length
        var m = min_or_neg_inf[F32]()
        for j in range(row_length):
            m = max(m, input[base + j].cast[F32]())
        var s = Scalar[F32](0)
        for j in range(row_length):
            s += exp(input[base + j].cast[F32]() - m)
        var inv = Scalar[F32](1) / s
        for j in range(row_length):
            var y = exp(input[base + j].cast[F32]() - m) * inv
            output[base + j] = y.cast[in_dtype]()


def run_case[
    in_dtype: DType
](num_rows: Int, row_length: Int, pattern: Int, name: String) -> Bool:
    var n = num_rows * row_length
    var inp = alloc[Scalar[in_dtype]](n).as_unsafe_any_origin()
    var got = alloc[Scalar[in_dtype]](n).as_unsafe_any_origin()
    var want = alloc[Scalar[in_dtype]](n).as_unsafe_any_origin()

    fill_pattern[in_dtype](inp, num_rows, row_length, pattern)

    # The same seam the C-ABI entries build: extents in, tiles out.
    var extent = row_major(Coord(num_rows, row_length))
    softmax_2d[in_dtype](
        RowMajor2D[in_dtype](ptr=inp, layout=extent),
        RowMajor2D[in_dtype](ptr=got, layout=extent),
    )

    reference_softmax[in_dtype](inp, want, num_rows, row_length)

    # Elementwise agreement with the oracle (both fp32 math, rounded to in_dtype).
    var max_diff = Scalar[F32](0)
    for k in range(n):
        var d = abs(got[k].cast[F32]() - want[k].cast[F32]())
        max_diff = max(max_diff, d)

    # Every row must be a valid probability distribution: finite and sums to 1.
    var worst_sum_err = Scalar[F32](0)
    var saw_nan = False
    for r in range(num_rows):
        var row_sum = Scalar[F32](0)
        for j in range(row_length):
            var v = got[r * row_length + j].cast[F32]()
            if v != v:
                saw_nan = True
            row_sum += v
        worst_sum_err = max(worst_sum_err, abs(row_sum - 1.0))

    # fp16 carries ~3 decimal digits; bf16 fewer. Both kernel and oracle round
    # identically, so agreement is tight and the slack absorbs summation order.
    var diff_tol = Scalar[F32](2.0e-3)
    var sum_tol = Scalar[F32](3.0e-2)
    var ok = (
        (not saw_nan) and (max_diff <= diff_tol) and (worst_sum_err <= sum_tol)
    )

    var status = "PASS" if ok else "FAIL"
    print(
        status,
        "|",
        name,
        "| rows=",
        num_rows,
        "cols=",
        row_length,
        "| max_diff=",
        max_diff,
        "| worst_row_sum_err=",
        worst_sum_err,
        "| nan=",
        saw_nan,
    )

    inp.free()
    got.free()
    want.free()
    return ok


def run_suite[in_dtype: DType](label: String) -> Int:
    print("==== softmax suite:", label, "====")
    var failures = 0

    # Shapes cover exact-vector, tail, tiny (< vector), multi-row, and large rows
    # that stress fp32 accumulation. row_lengths chosen for tail coverage against
    # the 64-lane fp16 vector.
    var shape_rows = [1, 1, 1, 1, 4, 2, 1, 1]
    var shape_cols = [16, 64, 127, 1000, 128, 65, 4096, 4097]
    var pattern_ids = [0, 1, 2, 3, 4, 5]
    var pattern_names = [
        String("all-equal"),
        String("large-positive"),
        String("large-negative"),
        String("ramp-mixed"),
        String("near-one-hot"),
        String("pseudo-random"),
    ]

    for si in range(len(shape_rows)):
        for pi in range(len(pattern_ids)):
            if not run_case[in_dtype](
                shape_rows[si],
                shape_cols[si],
                pattern_ids[pi],
                pattern_names[pi],
            ):
                failures += 1

    return failures


def run_striped[
    in_dtype: DType
](name: String, num_rows: Int, row_length: Int) -> Int:
    """Checks the launch-striped split against the whole-tile kernel.

    The four compute threads each run `softmax_striped` over the rows their
    launch index owns; the union must reproduce, element for element, what
    `softmax_2d` produces over the whole tile. Exact equality rather than a
    tolerance: both arms run the same row body, so a difference is a partition
    bug -- a row computed twice, or none at all -- not arithmetic. A sentinel
    pre-fill makes an uncovered row a failure rather than a plausible-looking
    zero.

    This is the shipped path: `mojo_softmax_f16` in `npu_kernel.mojo` reads the
    launch index off the shim and calls exactly this.
    """
    var failures = 0
    comptime SENTINEL = Scalar[in_dtype](-31000.0)

    var count = num_rows * row_length
    var src = alloc[Scalar[in_dtype]](count).as_unsafe_any_origin()
    for i in range(count):
        src.store(i, Scalar[in_dtype](Float64(i % 83) / 83.0 * 6.0 - 3.0))

    var whole = alloc[Scalar[in_dtype]](count).as_unsafe_any_origin()
    var striped_out = alloc[Scalar[in_dtype]](count).as_unsafe_any_origin()
    for i in range(count):
        whole.store(i, SENTINEL)
        striped_out.store(i, SENTINEL)

    var extent = row_major(Coord(num_rows, row_length))
    var src_t = RowMajor2D[in_dtype](ptr=src, layout=extent)

    softmax_2d[in_dtype](src_t, RowMajor2D[in_dtype](ptr=whole, layout=extent))
    for t in range(5):
        softmax_striped[in_dtype](
            src_t,
            RowMajor2D[in_dtype](ptr=striped_out, layout=extent),
            UInt32(t),
        )

    var bad = 0
    var uncovered = 0
    for i in range(count):
        if whole.load(i) != striped_out.load(i):
            bad += 1
        if striped_out.load(i) == SENTINEL:
            uncovered += 1
    if bad != 0 or uncovered != 0:
        print(
            "  FAIL",
            name,
            "striped mismatch elems=",
            bad,
            "uncovered=",
            uncovered,
        )
        failures += 1
    else:
        print("  ok  ", name, "striped == whole-tile, rows", num_rows)

    src.free()
    whole.free()
    striped_out.free()
    return failures


def main() raises:
    var failures = 0
    failures += run_suite[DType.float16]("fp16")
    failures += run_suite[DType.bfloat16]("bf16")

    # The launch-striped partition the shipped entry uses. Row counts that do
    # and do not divide by the four compute threads, and one below the thread
    # count so some indices get an empty run.
    print("==== launch-striped partition ====")
    for ref rows in [1, 3, 4, 7, 8, 13]:
        failures += run_striped[DType.float16](
            "fp16 rows=" + String(rows), rows, 128
        )
        failures += run_striped[DType.bfloat16](
            "bf16 rows=" + String(rows), rows, 96
        )

    print()
    if failures == 0:
        print("ALL SOFTMAX TESTS PASSED")
    else:
        raise Error(String(failures) + " softmax test case(s) FAILED")
