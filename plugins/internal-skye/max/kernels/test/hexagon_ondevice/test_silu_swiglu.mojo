"""Host validation for the HVX SiLU and SwiGLU kernels.

Runs the actual kernels (`silu_2d` / `swiglu_2d` from `silu_swiglu_kernel.mojo`)
on the host CPU against an independent fp64 reference, across the shapes and
input patterns the FLUX.2-Klein activations have to survive: zeros, saturating
large magnitudes of either sign, a ramp through the non-monotonic region, the
neighborhood of the SiLU minimum, row lengths that are not a multiple of the
vector width (the masked-remainder tail), multiple rows, the D=18432 SwiGLU
half a fused MLP projection produces, and row-pitched packed pairs — a column
slice of a wider fused projection, walked in place through the kernel's
`row_pitch` — each with a wrong-pitch control that must disagree.

The reference is deliberately *not* the kernel's math: it evaluates SiLU as
`x * 0.5 * (1 + tanh(x / 2))` in fp64, where the kernel uses an fp32
`exp(-|x|)` sigmoid. Agreement therefore cross-checks the algebra through a
different primitive (tanh vs exp) and at higher precision, not merely the
kernel against a copy of itself. SwiGLU is fused split-half: the input row is
`[gate, up]` and `chunk(x, 2, axis=-1)` splits it in half, exactly as
`Flux2SwiGLU` does.

The kernel source is target-independent Mojo; only its lowering differs between
host and Hexagon. Matching here proves the *algorithm*. On-device numerics
(chiefly the exp primitive and the fp16<->fp32 conversion path under
`+hvx-qfloat`) still need hardware confirmation.
"""

from layout.coord import Coord
from layout.tile_layout import row_major

from std.math import tanh
from std.memory import alloc

from hexagon_ondevice.silu_swiglu_kernel import (
    RowMajor2D,
    silu_2d,
    silu_striped,
    swiglu_2d,
    swiglu_striped,
)

comptime F64 = DType.float64

# SiLU's global minimum is ~-0.27846 at x ~= -1.2784; no output may fall below
# it (modulo fp16/bf16 rounding of that bound). A sign or argument mistake in
# the sigmoid violates this regardless of the reference.
comptime SILU_MIN = -0.2785


def tile2d[
    in_dtype: DType
](
    ptr: UnsafePointer[Scalar[in_dtype], MutUnsafeAnyOrigin],
    num_rows: Int,
    row_length: Int,
) -> RowMajor2D[in_dtype]:
    """Wraps a host allocation as the `[num_rows, row_length]` tile the kernels
    take, which is the construction their C-ABI entry points perform.

    The test states its own geometries rather than sharing the kernel's, because
    one of them has to be wrong on purpose: `run_swiglu_pitched`'s control walks
    a pitched buffer at the dense pitch, and must disagree.
    """
    return RowMajor2D[in_dtype](
        ptr=ptr, layout=row_major(Coord(num_rows, row_length))
    )


def ref_silu_f64(x: Float64) -> Float64:
    """Evaluates SiLU in fp64 via the tanh identity, independent of the kernel.
    """
    return x * 0.5 * (1.0 + tanh(0.5 * x))


def gate_value(j: Int, pattern: Int) -> Float64:
    """Returns the deterministic gate input for column `j` under `pattern`."""
    if pattern == 0:  # zeros: silu(0) == 0 exactly
        return 0.0
    elif pattern == 1:  # large positive: sigmoid -> 1, silu -> x
        return 20.0 + Float64(j % 21)
    elif pattern == 2:  # large negative incl. extreme: no exp overflow
        return -20.0 - Float64((j * 7) % 61)
    elif pattern == 3:  # ramp across sign, through the non-monotonic region
        return -8.0 + Float64(j % 1600) * 0.01
    elif pattern == 4:  # neighborhood of the SiLU minimum (~ -1.2784)
        return -2.0 + Float64(j % 30) * 0.05
    else:  # pseudo-random in [-6, 6)
        return Float64(Int((j * 2654435761 + 12345) % 1200)) * 0.01 - 6.0


def up_value(j: Int) -> Float64:
    """The SwiGLU `up` multiplicand: signed, non-trivial, and distinct from
    every gate pattern so an argument swap cannot pass."""
    return 0.75 + Float64((j * 3) % 17) * 0.1 - Float64((j * 5) % 11) * 0.15


def fill_silu[
    in_dtype: DType
](
    ptr: UnsafePointer[Scalar[in_dtype], MutUnsafeAnyOrigin],
    num_rows: Int,
    row_length: Int,
    pattern: Int,
):
    """Writes a deterministic SiLU input in fp64, rounded to `in_dtype`."""
    for r in range(num_rows):
        for j in range(row_length):
            ptr[r * row_length + j] = Scalar[in_dtype](
                Float32(gate_value(j, pattern))
            )


def fill_swiglu[
    in_dtype: DType
](
    ptr: UnsafePointer[Scalar[in_dtype], MutUnsafeAnyOrigin],
    num_rows: Int,
    inner_dim: Int,
    pattern: Int,
):
    """Writes a fused `[num_rows, 2*inner_dim]` input: gate in each row's first
    half, up in the second, the split-half layout `chunk(x, 2)` consumes."""
    for r in range(num_rows):
        var base = r * 2 * inner_dim
        for j in range(inner_dim):
            ptr[base + j] = Scalar[in_dtype](Float32(gate_value(j, pattern)))
            ptr[base + inner_dim + j] = Scalar[in_dtype](Float32(up_value(j)))


def check[
    in_dtype: DType
](
    got: UnsafePointer[Scalar[in_dtype], MutUnsafeAnyOrigin],
    want: UnsafePointer[Scalar[in_dtype], MutUnsafeAnyOrigin],
    n: Int,
    atol: Float64,
    rtol: Float64,
    name: String,
    check_silu_min: Bool,
) -> Bool:
    """Compares `got` against the fp64 reference `want`, with a NaN guard and,
    for plain SiLU, the analytic lower-bound anchor."""
    var max_abs_diff = Float64(0)
    var worst_over_tol = Float64(0)
    var min_out = Float64(0)
    var saw_nan = False
    for k in range(n):
        var g = got[k].cast[F64]()
        var w = want[k].cast[F64]()
        if g != g:
            saw_nan = True
        var d = abs(g - w)
        max_abs_diff = max(max_abs_diff, d)
        worst_over_tol = max(worst_over_tol, d - (atol + rtol * abs(w)))
        min_out = min(min_out, g)

    var bound_ok = (not check_silu_min) or (min_out >= SILU_MIN + (-atol))
    var ok = (not saw_nan) and (worst_over_tol <= 0.0) and bound_ok

    var status = "PASS" if ok else "FAIL"
    print(
        status,
        "|",
        name,
        "| n=",
        n,
        "| max_abs_diff=",
        max_abs_diff,
        "| worst_over_tol=",
        worst_over_tol,
        "| min_out=",
        min_out,
        "| nan=",
        saw_nan,
    )
    return ok


def run_silu_case[
    in_dtype: DType
](
    num_rows: Int,
    row_length: Int,
    pattern: Int,
    atol: Float64,
    rtol: Float64,
    name: String,
) -> Bool:
    var n = num_rows * row_length
    var inp = alloc[Scalar[in_dtype]](n).as_unsafe_any_origin()
    var got = alloc[Scalar[in_dtype]](n).as_unsafe_any_origin()
    var want = alloc[Scalar[in_dtype]](n).as_unsafe_any_origin()

    fill_silu[in_dtype](inp, num_rows, row_length, pattern)
    silu_2d[in_dtype](
        tile2d[in_dtype](inp, num_rows, row_length),
        tile2d[in_dtype](got, num_rows, row_length),
    )
    for k in range(n):
        want[k] = ref_silu_f64(inp[k].cast[F64]()).cast[in_dtype]()

    var ok = check[in_dtype](got, want, n, atol, rtol, name, True)
    inp.free()
    got.free()
    want.free()
    return ok


def run_swiglu_case[
    in_dtype: DType
](
    num_rows: Int,
    inner_dim: Int,
    pattern: Int,
    atol: Float64,
    rtol: Float64,
    name: String,
) -> Bool:
    var inp = alloc[Scalar[in_dtype]](
        num_rows * 2 * inner_dim
    ).as_unsafe_any_origin()
    var got = alloc[Scalar[in_dtype]](
        num_rows * inner_dim
    ).as_unsafe_any_origin()
    var want = alloc[Scalar[in_dtype]](
        num_rows * inner_dim
    ).as_unsafe_any_origin()

    fill_swiglu[in_dtype](inp, num_rows, inner_dim, pattern)
    swiglu_2d[in_dtype](
        tile2d[in_dtype](inp, num_rows, 2 * inner_dim),
        tile2d[in_dtype](got, num_rows, inner_dim),
    )
    for r in range(num_rows):
        var in_base = r * 2 * inner_dim
        for i in range(inner_dim):
            var s = ref_silu_f64(inp[in_base + i].cast[F64]())
            var u = inp[in_base + inner_dim + i].cast[F64]()
            want[r * inner_dim + i] = (s * u).cast[in_dtype]()

    var ok = check[in_dtype](
        got, want, num_rows * inner_dim, atol, rtol, name, False
    )
    inp.free()
    got.free()
    want.free()
    return ok


def run_swiglu_anchors[in_dtype: DType](label: String) -> Int:
    """Checks the two properties that pin the gate wiring, independent of the
    fp64 reference: `up == 1` reduces SwiGLU to SiLU, and `gate == 0` zeros the
    output whatever `up` is (`silu(0) == 0`). Both exercise the split-half
    layout: an interleaved read would fail the first."""
    var failures = 0
    var num_rows = 3
    var inner_dim = 320  # a 64-lane tail: 320 = 5*64, plus below uses 200 too

    var inp = alloc[Scalar[in_dtype]](
        num_rows * 2 * inner_dim
    ).as_unsafe_any_origin()
    var swig = alloc[Scalar[in_dtype]](
        num_rows * inner_dim
    ).as_unsafe_any_origin()
    var silu = alloc[Scalar[in_dtype]](
        num_rows * inner_dim
    ).as_unsafe_any_origin()

    # up == 1  =>  swiglu(gate, 1) == silu(gate).
    for r in range(num_rows):
        var base = r * 2 * inner_dim
        for j in range(inner_dim):
            inp[base + j] = Scalar[in_dtype](Float32(gate_value(j, 3)))
            inp[base + inner_dim + j] = Scalar[in_dtype](1.0)
    swiglu_2d[in_dtype](
        tile2d[in_dtype](inp, num_rows, 2 * inner_dim),
        tile2d[in_dtype](swig, num_rows, inner_dim),
    )
    # silu over the gate halves alone, as a contiguous [num_rows, inner_dim].
    var gate_only = alloc[Scalar[in_dtype]](
        num_rows * inner_dim
    ).as_unsafe_any_origin()
    for r in range(num_rows):
        for j in range(inner_dim):
            gate_only[r * inner_dim + j] = Scalar[in_dtype](
                Float32(gate_value(j, 3))
            )
    silu_2d[in_dtype](
        tile2d[in_dtype](gate_only, num_rows, inner_dim),
        tile2d[in_dtype](silu, num_rows, inner_dim),
    )
    var mismatch = 0
    for k in range(num_rows * inner_dim):
        if swig[k] != silu[k]:
            mismatch += 1
    var a1 = mismatch == 0
    print(
        "PASS" if a1 else "FAIL",
        "| anchor up==1 -> swiglu==silu |",
        label,
        "| mismatches=",
        mismatch,
    )
    if not a1:
        failures += 1

    # gate == 0  =>  output all zero, whatever up holds.
    for r in range(num_rows):
        var base = r * 2 * inner_dim
        for j in range(inner_dim):
            inp[base + j] = Scalar[in_dtype](0.0)
            inp[base + inner_dim + j] = Scalar[in_dtype](Float32(up_value(j)))
    swiglu_2d[in_dtype](
        tile2d[in_dtype](inp, num_rows, 2 * inner_dim),
        tile2d[in_dtype](swig, num_rows, inner_dim),
    )
    var worst = Float64(0)
    for k in range(num_rows * inner_dim):
        worst = max(worst, abs(swig[k].cast[F64]()))
    var a2 = worst == 0.0
    print(
        "PASS" if a2 else "FAIL",
        "| anchor gate==0 -> out==0 |",
        label,
        "| max|out|=",
        worst,
    )
    if not a2:
        failures += 1

    inp.free()
    swig.free()
    silu.free()
    gate_only.free()
    return failures


def run_swiglu_pitched[
    in_dtype: DType
](
    num_rows: Int,
    inner_dim: Int,
    row_pitch: Int,
    col_offset: Int,
    pattern: Int,
    atol: Float64,
    rtol: Float64,
    label: String,
) -> Int:
    """Runs the kernel over a packed pair living as a column slice of a wider
    `[num_rows, row_pitch]` buffer — the layout a fused projection hands the
    graph, its start folded into the pointer. Everything outside the pair is
    poisoned, so three checks discriminate the addressing: agreement with the
    fp64 reference, bitwise equality with the dense kernel over gathered rows
    (same math, different addressing only), and a deliberately dense-pitched
    walk over the same storage that MUST disagree — the control that proves
    the comparison would catch a kernel ignoring its pitch."""
    var failures = 0
    var buf = alloc[Scalar[in_dtype]](
        num_rows * row_pitch
    ).as_unsafe_any_origin()
    var got = alloc[Scalar[in_dtype]](
        num_rows * inner_dim
    ).as_unsafe_any_origin()
    var want = alloc[Scalar[in_dtype]](
        num_rows * inner_dim
    ).as_unsafe_any_origin()
    var dense = alloc[Scalar[in_dtype]](
        num_rows * 2 * inner_dim
    ).as_unsafe_any_origin()
    var got_dense = alloc[Scalar[in_dtype]](
        num_rows * inner_dim
    ).as_unsafe_any_origin()

    for k in range(num_rows * row_pitch):
        buf[k] = Scalar[in_dtype](999.0)
    for r in range(num_rows):
        var base = r * row_pitch + col_offset
        for j in range(inner_dim):
            buf[base + j] = Scalar[in_dtype](Float32(gate_value(j, pattern)))
            buf[base + inner_dim + j] = Scalar[in_dtype](Float32(up_value(j)))

    var view = buf + col_offset
    swiglu_2d[in_dtype](
        tile2d[in_dtype](view, num_rows, row_pitch),
        tile2d[in_dtype](got, num_rows, inner_dim),
    )

    for r in range(num_rows):
        var base = r * row_pitch + col_offset
        for i in range(inner_dim):
            var s = ref_silu_f64(buf[base + i].cast[F64]())
            var u = buf[base + inner_dim + i].cast[F64]()
            want[r * inner_dim + i] = (s * u).cast[in_dtype]()
    if not check[in_dtype](
        got, want, num_rows * inner_dim, atol, rtol, label, False
    ):
        failures += 1

    for r in range(num_rows):
        var src = r * row_pitch + col_offset
        var dst = r * 2 * inner_dim
        for j in range(2 * inner_dim):
            dense[dst + j] = buf[src + j]
    swiglu_2d[in_dtype](
        tile2d[in_dtype](dense, num_rows, 2 * inner_dim),
        tile2d[in_dtype](got_dense, num_rows, inner_dim),
    )
    var mismatch = 0
    for k in range(num_rows * inner_dim):
        if got[k] != got_dense[k]:
            mismatch += 1
    var bitwise_ok = mismatch == 0
    print(
        "PASS" if bitwise_ok else "FAIL",
        "| pitched == dense bitwise |",
        label,
        "| mismatches=",
        mismatch,
    )
    if not bitwise_ok:
        failures += 1

    # The failing control: the same storage walked at the dense pitch reads
    # poison from the second row on, so it must NOT match. A run where it
    # does means the comparison could not have caught a pitch mistake. The
    # pitch now rides in the input tile's layout, so stating it wrongly means
    # handing the kernel a differently-shaped tile.
    swiglu_2d[in_dtype](
        tile2d[in_dtype](view, num_rows, 2 * inner_dim),
        tile2d[in_dtype](got, num_rows, inner_dim),
    )
    var control = 0
    for k in range(num_rows * inner_dim):
        if got[k] != got_dense[k]:
            control += 1
    var control_ok = control > 0
    print(
        "PASS" if control_ok else "FAIL",
        "| wrong-pitch control disagrees |",
        label,
        "| mismatches=",
        control,
    )
    if not control_ok:
        failures += 1

    buf.free()
    got.free()
    want.free()
    dense.free()
    got_dense.free()
    return failures


def run_striped[
    in_dtype: DType
](name: String, num_rows: Int, inner_dim: Int) -> Int:
    """Checks the launch-striped split against the whole-tile kernel.

    The four compute threads each run `silu_striped` / `swiglu_striped` over the
    rows their launch index owns; the union must reproduce, element for element,
    what `silu_2d` / `swiglu_2d` produce over the whole tile. That is an exact
    equality rather than a tolerance: both arms run the same row body, so any
    difference is a partition bug -- a row computed twice, or none at all -- not
    arithmetic. A sentinel pre-fill makes an uncovered row a failure rather than
    a silent zero.

    This is the shipped path. `mojo_silu_*` / `mojo_swiglu_*` in
    `npu_kernel.mojo` read the launch index off the shim and call exactly these,
    so a partition that drops a row drops it in FLUX.
    """
    var failures = 0
    comptime SENTINEL = Scalar[in_dtype](-31000.0)

    var pitch = 2 * inner_dim
    var src = alloc[Scalar[in_dtype]](num_rows * pitch).as_unsafe_any_origin()
    for i in range(num_rows * pitch):
        var t = Float64(i % 251) / 251.0
        src.store(i, Scalar[in_dtype]((t * 8.0 - 4.0)))

    var whole = alloc[Scalar[in_dtype]](num_rows * pitch).as_unsafe_any_origin()
    var striped_out = alloc[Scalar[in_dtype]](
        num_rows * pitch
    ).as_unsafe_any_origin()

    # SiLU: both tiles are the full `[num_rows, pitch]`.
    for i in range(num_rows * pitch):
        whole.store(i, SENTINEL)
        striped_out.store(i, SENTINEL)
    silu_2d[in_dtype](
        tile2d[in_dtype](src, num_rows, pitch),
        tile2d[in_dtype](whole, num_rows, pitch),
    )
    for t in range(5):
        silu_striped[in_dtype](
            tile2d[in_dtype](src, num_rows, pitch),
            tile2d[in_dtype](striped_out, num_rows, pitch),
            UInt32(t),
        )
    var silu_bad = 0
    for i in range(num_rows * pitch):
        if whole.load(i) != striped_out.load(i):
            silu_bad += 1
    if silu_bad != 0:
        print("  FAIL", name, "silu striped mismatch elems=", silu_bad)
        failures += 1

    # SwiGLU: input keeps the pitch, output is dense at `inner_dim`.
    for i in range(num_rows * pitch):
        whole.store(i, SENTINEL)
        striped_out.store(i, SENTINEL)
    swiglu_2d[in_dtype](
        tile2d[in_dtype](src, num_rows, pitch),
        tile2d[in_dtype](whole, num_rows, inner_dim),
    )
    for t in range(5):
        swiglu_striped[in_dtype](
            tile2d[in_dtype](src, num_rows, pitch),
            tile2d[in_dtype](striped_out, num_rows, inner_dim),
            UInt32(t),
        )
    var swiglu_bad = 0
    for i in range(num_rows * inner_dim):
        if whole.load(i) != striped_out.load(i):
            swiglu_bad += 1
    var uncovered = 0
    for i in range(num_rows * inner_dim):
        if striped_out.load(i) == SENTINEL:
            uncovered += 1
    if swiglu_bad != 0 or uncovered != 0:
        print(
            "  FAIL",
            name,
            "swiglu striped mismatch elems=",
            swiglu_bad,
            "uncovered=",
            uncovered,
        )
        failures += 1

    if failures == 0:
        print("  ok  ", name, "striped == whole-tile, rows", num_rows)

    src.free()
    whole.free()
    striped_out.free()
    return failures


def run_suite[
    in_dtype: DType
](label: String, atol: Float64, rtol: Float64) -> Int:
    print("==== silu/swiglu suite:", label, "====")
    var failures = 0

    # tiny (< vector), tails, exact vectors, large tail, the D=18432 SwiGLU
    # half, and a large exact multiple. LANES is 64 for both fp16 and bf16.
    var shape_rows = [1, 1, 1, 2, 1, 1, 1, 4]
    var shape_cols = [16, 63, 64, 127, 128, 4097, 18432, 200]
    var patterns = [0, 1, 2, 3, 4, 5]
    var pattern_names = [
        String("zeros"),
        String("large-positive"),
        String("large-negative"),
        String("ramp-mixed"),
        String("near-min"),
        String("pseudo-random"),
    ]

    for si in range(len(shape_rows)):
        for pi in range(len(patterns)):
            if not run_silu_case[in_dtype](
                shape_rows[si],
                shape_cols[si],
                patterns[pi],
                atol,
                rtol,
                "silu/" + pattern_names[pi],
            ):
                failures += 1
            if not run_swiglu_case[in_dtype](
                shape_rows[si],
                shape_cols[si],
                patterns[pi],
                atol,
                rtol,
                "swiglu/" + pattern_names[pi],
            ):
                failures += 1

    failures += run_swiglu_anchors[in_dtype](label)

    # Pitched packed pairs: a column slice of a wider projection, the layout
    # FLUX.2's single-stream blocks produce. Geometries cover an interior
    # slice at model-like width, tails with an odd (vector-unaligned) start,
    # an exact-lane slice at an aligned start, and a left-aligned slice whose
    # pitch is pure right-padding.
    var p_rows = [3, 2, 2, 4]
    var p_inner = [320, 63, 64, 200]
    var p_pitch = [1024, 200, 192, 448]
    var p_off = [384, 37, 64, 0]
    for gi in range(len(p_rows)):
        for pat in [3, 5]:
            failures += run_swiglu_pitched[in_dtype](
                p_rows[gi],
                p_inner[gi],
                p_pitch[gi],
                p_off[gi],
                pat,
                atol,
                rtol,
                "swiglu-pitched/" + label,
            )
    return failures


def main() raises:
    var failures = 0
    # fp16: ~11-bit mantissa. bf16: ~8-bit, so a wider band absorbs the rare
    # one-ulp straddle where the fp32 kernel and fp64 reference round apart.
    failures += run_suite[DType.float16]("fp16", 3.0e-3, 3.0e-3)
    failures += run_suite[DType.bfloat16]("bf16", 1.5e-2, 1.5e-2)

    # The launch-striped partition the shipped entries use. Row counts that do
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
        print("ALL SILU/SWIGLU TESTS PASSED")
    else:
        raise Error(String(failures) + " silu/swiglu test case(s) FAILED")
