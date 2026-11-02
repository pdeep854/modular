"""Host validation for the HVX adaLN-modulation and gated-residual kernels.

Runs the actual kernels (`ada_modulate_2d` / `gate_residual_2d` from
`modulation_kernel.mojo`) on the host CPU against an independent scalar fp32
reference, across the shapes and value patterns each op sees in an MMDiT block:
the FLUX hidden dim `D = 6144`, small and tail row lengths, multi-row batches,
both parameter broadcast layouts (a shared `[D]` vector vs a materialized
`[rows, D]`), and the edge cases that matter numerically -- `scale == -1`
(modulation collapses to `shift`), `gate == 0` (residual collapses to `x`), and
large magnitudes that need the fp32 intermediate.

The kernel source is target-independent Mojo; only its lowering differs between
host and Hexagon. Matching the reference here proves the *algorithm*. On-device
numerics (the fp16<->fp32 conversion path under `+hvx-qfloat`, chiefly the
`vmpy(v.hf,v.hf)` product) still need hardware confirmation.
"""

from std.memory import alloc

from hexagon_ondevice.layernorm_kernel import _layer_norm
from hexagon_ondevice.modulation_kernel import (
    _ada_modulate_ln_rows,
    _param_tile,
    _rows_tile,
    ada_modulate_2d,
    ada_modulate_ln_striped,
    ada_modulate_striped,
    gate_residual_2d,
    gate_residual_striped,
)

# The reference runs in fp64, a wider type than the kernel's fp32 accumulator, so
# agreement is not the kernel checked against a copy of its own arithmetic: a
# real loss in the fp32 fold would round to a different fp16/bf16 value than the
# fp64 fold and be counted. For a plain multiply-add there is no independent
# *primitive* to cross-check the way `tanh` cross-checks `exp` in silu, so the
# independent higher precision is what the comparison rests on.
comptime F64 = DType.float64


def fill_x[
    in_dtype: DType
](
    ptr: UnsafePointer[Scalar[in_dtype], MutUnsafeAnyOrigin],
    count: Int,
    pattern: Int,
):
    """Writes a deterministic activation pattern in fp32, rounded to `in_dtype`.
    """
    for i in range(count):
        var x: Float32
        if pattern == 0:  # moderate ramp, mixed sign
            x = -3.0 + Float32(i % 600) * 0.01
        elif pattern == 1:  # large positive: needs the fp32 intermediate
            x = 40.0 + Float32(i % 25)
        elif pattern == 2:  # large negative
            x = -50.0 - Float32(i % 19)
        else:  # deterministic pseudo-random in [-5, 5)
            x = Float32(Int((i * 40503 + 7) % 1000)) * 0.01 - 5.0
        ptr[i] = Scalar[in_dtype](x)


def fill_param[
    in_dtype: DType
](
    ptr: UnsafePointer[Scalar[in_dtype], MutUnsafeAnyOrigin],
    count: Int,
    pattern: Int,
):
    """Writes a deterministic scale/shift/gate pattern in fp32, rounded to
    `in_dtype`."""
    for i in range(count):
        var v: Float32
        if pattern == 0:  # exactly -1: collapses (1 + scale) to 0
            v = -1.0
        elif pattern == 1:  # exactly 0: collapses gate*y to 0
            v = 0.0
        elif pattern == 2:  # large magnitude, both signs
            v = Float32(i % 21) - 10.0
        elif pattern == 3:  # pseudo-random small in [-2, 2)
            v = Float32(Int((i * 2654435761 + 12345) % 400)) * 0.01 - 2.0
        else:  # ramp through -1, exercising the collapse point
            v = -1.0 + Float32(i % 100) * 0.02
        ptr[i] = Scalar[in_dtype](v)


def reference_modulate[
    in_dtype: DType
](
    x: UnsafePointer[Scalar[in_dtype], MutUnsafeAnyOrigin],
    scale: UnsafePointer[Scalar[in_dtype], MutUnsafeAnyOrigin],
    shift: UnsafePointer[Scalar[in_dtype], MutUnsafeAnyOrigin],
    output: UnsafePointer[Scalar[in_dtype], MutUnsafeAnyOrigin],
    num_rows: Int,
    row_length: Int,
    param_row_stride: Int,
):
    """Independent scalar oracle: `(1 + scale) * x + shift` in fp64, rounded out.
    """
    for r in range(num_rows):
        for j in range(row_length):
            var xi = x[r * row_length + j].cast[F64]()
            var si = scale[r * param_row_stride + j].cast[F64]()
            var shi = shift[r * param_row_stride + j].cast[F64]()
            var y = (Float64(1.0) + si) * xi + shi
            output[r * row_length + j] = y.cast[in_dtype]()


def reference_gate[
    in_dtype: DType
](
    x: UnsafePointer[Scalar[in_dtype], MutUnsafeAnyOrigin],
    y: UnsafePointer[Scalar[in_dtype], MutUnsafeAnyOrigin],
    gate: UnsafePointer[Scalar[in_dtype], MutUnsafeAnyOrigin],
    output: UnsafePointer[Scalar[in_dtype], MutUnsafeAnyOrigin],
    num_rows: Int,
    row_length: Int,
    gate_row_stride: Int,
):
    """Independent scalar oracle: `x + gate * y` in fp64, rounded out."""
    for r in range(num_rows):
        for j in range(row_length):
            var xi = x[r * row_length + j].cast[F64]()
            var yi = y[r * row_length + j].cast[F64]()
            var gi = gate[r * gate_row_stride + j].cast[F64]()
            output[r * row_length + j] = (xi + gi * yi).cast[in_dtype]()


@always_inline
def is_close(got: Float64, want: Float64, atol: Float64, rtol: Float64) -> Bool:
    # `got == want` short-circuits equal (including matching +/-inf from fp16
    # saturation at large magnitude); the relative band absorbs the fp16 ulp.
    if got == want:
        return True
    return abs(got - want) <= atol + rtol * abs(want)


def _compare[
    in_dtype: DType
](
    got: UnsafePointer[Scalar[in_dtype], MutUnsafeAnyOrigin],
    want: UnsafePointer[Scalar[in_dtype], MutUnsafeAnyOrigin],
    n: Int,
    atol: Float64,
    rtol: Float64,
    mut worst_abs: Float64,
    mut worst_rel: Float64,
) -> Bool:
    var ok = True
    for k in range(n):
        var g = got[k].cast[F64]()
        var w = want[k].cast[F64]()
        if g != g:  # NaN never appears in a correct output
            ok = False
            continue
        if not is_close(g, w, atol, rtol):
            ok = False
        var d = abs(g - w)
        worst_abs = max(worst_abs, d)
        if abs(w) > 1.0:
            worst_rel = max(worst_rel, d / abs(w))
    return ok


def run_modulate_case[
    in_dtype: DType
](
    num_rows: Int,
    row_length: Int,
    param_row_stride: Int,
    x_pattern: Int,
    scale_pattern: Int,
    atol: Float64,
    rtol: Float64,
    mut worst_abs: Float64,
    mut worst_rel: Float64,
    name: String,
) -> Bool:
    var n = num_rows * row_length
    var param_len = (num_rows - 1) * param_row_stride + row_length
    var x = alloc[Scalar[in_dtype]](n).as_unsafe_any_origin()
    var scale = alloc[Scalar[in_dtype]](param_len).as_unsafe_any_origin()
    var shift = alloc[Scalar[in_dtype]](param_len).as_unsafe_any_origin()
    var got = alloc[Scalar[in_dtype]](n).as_unsafe_any_origin()
    var want = alloc[Scalar[in_dtype]](n).as_unsafe_any_origin()

    fill_x[in_dtype](x, n, x_pattern)
    fill_param[in_dtype](scale, param_len, scale_pattern)
    fill_param[in_dtype](shift, param_len, 3)  # independent shift

    ada_modulate_2d[in_dtype](
        _rows_tile[in_dtype](x, num_rows, row_length),
        _param_tile[in_dtype](scale, num_rows, row_length, param_row_stride),
        _param_tile[in_dtype](shift, num_rows, row_length, param_row_stride),
        _rows_tile[in_dtype](got, num_rows, row_length),
    )
    reference_modulate[in_dtype](
        x, scale, shift, want, num_rows, row_length, param_row_stride
    )

    var ok = _compare[in_dtype](got, want, n, atol, rtol, worst_abs, worst_rel)
    if not ok:
        print(
            "  FAIL |",
            name,
            "| rows=",
            num_rows,
            "cols=",
            row_length,
            "stride=",
            param_row_stride,
            "x=",
            x_pattern,
            "scale=",
            scale_pattern,
        )

    x.free()
    scale.free()
    shift.free()
    got.free()
    want.free()
    return ok


def run_gate_case[
    in_dtype: DType
](
    num_rows: Int,
    row_length: Int,
    gate_row_stride: Int,
    x_pattern: Int,
    gate_pattern: Int,
    atol: Float64,
    rtol: Float64,
    mut worst_abs: Float64,
    mut worst_rel: Float64,
    name: String,
) -> Bool:
    var n = num_rows * row_length
    var gate_len = (num_rows - 1) * gate_row_stride + row_length
    var x = alloc[Scalar[in_dtype]](n).as_unsafe_any_origin()
    var yv = alloc[Scalar[in_dtype]](n).as_unsafe_any_origin()
    var gate = alloc[Scalar[in_dtype]](gate_len).as_unsafe_any_origin()
    var got = alloc[Scalar[in_dtype]](n).as_unsafe_any_origin()
    var want = alloc[Scalar[in_dtype]](n).as_unsafe_any_origin()

    fill_x[in_dtype](x, n, x_pattern)
    fill_x[in_dtype](yv, n, (x_pattern + 2) % 4)  # independent sub-block output
    fill_param[in_dtype](gate, gate_len, gate_pattern)

    gate_residual_2d[in_dtype](
        _rows_tile[in_dtype](x, num_rows, row_length),
        _rows_tile[in_dtype](yv, num_rows, row_length),
        _param_tile[in_dtype](gate, num_rows, row_length, gate_row_stride),
        _rows_tile[in_dtype](got, num_rows, row_length),
    )
    reference_gate[in_dtype](
        x, yv, gate, want, num_rows, row_length, gate_row_stride
    )

    var ok = _compare[in_dtype](got, want, n, atol, rtol, worst_abs, worst_rel)
    if not ok:
        print(
            "  FAIL |",
            name,
            "| rows=",
            num_rows,
            "cols=",
            row_length,
            "stride=",
            gate_row_stride,
            "x=",
            x_pattern,
            "gate=",
            gate_pattern,
        )

    x.free()
    yv.free()
    gate.free()
    got.free()
    want.free()
    return ok


def anchor_modulate_scale_neg1[in_dtype: DType]() -> Bool:
    """`scale == -1` must give `out == shift` bit-exact, independent of the
    oracle."""
    var num_rows = 3
    var row_length = 6144
    var n = num_rows * row_length
    var x = alloc[Scalar[in_dtype]](n).as_unsafe_any_origin()
    var scale = alloc[Scalar[in_dtype]](n).as_unsafe_any_origin()
    var shift = alloc[Scalar[in_dtype]](n).as_unsafe_any_origin()
    var got = alloc[Scalar[in_dtype]](n).as_unsafe_any_origin()

    fill_x[in_dtype](x, n, 1)  # large x: (1 + -1)*x must still vanish
    fill_param[in_dtype](scale, n, 0)  # exactly -1
    fill_param[in_dtype](shift, n, 3)

    ada_modulate_2d[in_dtype](
        _rows_tile[in_dtype](x, num_rows, row_length),
        _param_tile[in_dtype](scale, num_rows, row_length, n // num_rows),
        _param_tile[in_dtype](shift, num_rows, row_length, n // num_rows),
        _rows_tile[in_dtype](got, num_rows, row_length),
    )

    var mism = 0
    for k in range(n):
        if got[k] != shift[k]:
            mism += 1
    var ok = mism == 0
    print(
        "PASS" if ok else "FAIL",
        "| anchor scale=-1 -> out==shift | mismatches=",
        mism,
    )

    x.free()
    scale.free()
    shift.free()
    got.free()
    return ok


def anchor_modulate_identity[in_dtype: DType]() -> Bool:
    """`scale == 0` and `shift == 0` must give `out == x` bit-exact: the affine's
    identity, independent of the oracle. A spurious offset or a `1 + scale`
    miscount breaks it."""
    var num_rows = 3
    var row_length = 6144
    var n = num_rows * row_length
    var x = alloc[Scalar[in_dtype]](n).as_unsafe_any_origin()
    var scale = alloc[Scalar[in_dtype]](n).as_unsafe_any_origin()
    var shift = alloc[Scalar[in_dtype]](n).as_unsafe_any_origin()
    var got = alloc[Scalar[in_dtype]](n).as_unsafe_any_origin()

    fill_x[in_dtype](x, n, 0)  # ordinary mixed-sign activation
    fill_param[in_dtype](scale, n, 1)  # exactly 0
    fill_param[in_dtype](shift, n, 1)  # exactly 0

    ada_modulate_2d[in_dtype](
        _rows_tile[in_dtype](x, num_rows, row_length),
        _param_tile[in_dtype](scale, num_rows, row_length, n // num_rows),
        _param_tile[in_dtype](shift, num_rows, row_length, n // num_rows),
        _rows_tile[in_dtype](got, num_rows, row_length),
    )

    var mism = 0
    for k in range(n):
        if got[k] != x[k]:
            mism += 1
    var ok = mism == 0
    print(
        "PASS" if ok else "FAIL",
        "| anchor scale=0,shift=0 -> out==x | mismatches=",
        mism,
    )

    x.free()
    scale.free()
    shift.free()
    got.free()
    return ok


def anchor_gate_zero[in_dtype: DType]() -> Bool:
    """`gate == 0` must give `out == x` bit-exact, independent of the oracle."""
    var num_rows = 3
    var row_length = 6144
    var n = num_rows * row_length
    var x = alloc[Scalar[in_dtype]](n).as_unsafe_any_origin()
    var yv = alloc[Scalar[in_dtype]](n).as_unsafe_any_origin()
    var gate = alloc[Scalar[in_dtype]](n).as_unsafe_any_origin()
    var got = alloc[Scalar[in_dtype]](n).as_unsafe_any_origin()

    fill_x[in_dtype](x, n, 3)
    fill_x[in_dtype](yv, n, 1)  # large y: 0 * y must still vanish
    fill_param[in_dtype](gate, n, 1)  # exactly 0

    gate_residual_2d[in_dtype](
        _rows_tile[in_dtype](x, num_rows, row_length),
        _rows_tile[in_dtype](yv, num_rows, row_length),
        _param_tile[in_dtype](gate, num_rows, row_length, n // num_rows),
        _rows_tile[in_dtype](got, num_rows, row_length),
    )

    var mism = 0
    for k in range(n):
        if got[k] != x[k]:
            mism += 1
    var ok = mism == 0
    print(
        "PASS" if ok else "FAIL",
        "| anchor gate=0 -> out==x | mismatches=",
        mism,
    )

    x.free()
    yv.free()
    gate.free()
    got.free()
    return ok


def run_suite[in_dtype: DType](label: String) -> Int:
    print("==== modulation suite:", label, "====")
    # The kernel folds in fp32 and the oracle in fp64, so the two round to the
    # same fp16/bf16 value except at a rare halfway straddle; the band absorbs
    # that one-ulp disagreement. bf16's 8-bit mantissa gets the looser band.
    var atol: Float64
    var rtol: Float64
    comptime if in_dtype == DType.float16:
        atol = 3.0e-3
        rtol = 3.0e-3
    else:
        atol = 1.5e-2
        rtol = 1.5e-2

    # exact-vector, tail, tiny (< 64-lane vector), multi-row, FLUX D=6144, and
    # large tail. 6144 % 64 == 0 (no tail); 6145/6143/4097/127/65 exercise it.
    var shape_rows = [1, 1, 1, 4, 2, 1, 3, 1]
    var shape_cols = [16, 64, 127, 128, 65, 6144, 6145, 4097]

    var failures = 0
    var cases = 0
    var worst_abs = Float64(0)
    var worst_rel = Float64(0)

    # Modulation: both param layouts (shared [D] via stride 0; materialized
    # [rows, D] via stride == row_length), spanning x and scale patterns.
    var mod_x = [0, 1, 3]
    var mod_scale = [1, 2, 3, 4]
    for si in range(len(shape_rows)):
        var rows = shape_rows[si]
        var cols = shape_cols[si]
        for stride_mode in range(2):
            var stride = 0 if stride_mode == 0 else cols
            for xi in range(len(mod_x)):
                for pi in range(len(mod_scale)):
                    cases += 1
                    if not run_modulate_case[in_dtype](
                        rows,
                        cols,
                        stride,
                        mod_x[xi],
                        mod_scale[pi],
                        atol,
                        rtol,
                        worst_abs,
                        worst_rel,
                        "modulate",
                    ):
                        failures += 1

    # Gated residual, per-channel gate: shared vs materialized layouts.
    var gate_x = [0, 3]
    var gate_p = [1, 2, 3, 4]
    for si in range(len(shape_rows)):
        var rows = shape_rows[si]
        var cols = shape_cols[si]
        for stride_mode in range(2):
            var stride = 0 if stride_mode == 0 else cols
            for xi in range(len(gate_x)):
                for pi in range(len(gate_p)):
                    cases += 1
                    if not run_gate_case[in_dtype](
                        rows,
                        cols,
                        stride,
                        gate_x[xi],
                        gate_p[pi],
                        atol,
                        rtol,
                        worst_abs,
                        worst_rel,
                        "gate",
                    ):
                        failures += 1

    if not anchor_modulate_identity[in_dtype]():
        failures += 1
    if not anchor_modulate_scale_neg1[in_dtype]():
        failures += 1
    if not anchor_gate_zero[in_dtype]():
        failures += 1

    print(
        "  cases=",
        cases,
        "| failures=",
        failures,
        "| worst_abs=",
        worst_abs,
        "| worst_rel=",
        worst_rel,
    )
    return failures


def run_striped[
    in_dtype: DType
](name: String, num_rows: Int, row_length: Int, param_stride: Int) -> Int:
    """Checks the launch-striped split against the whole-tile kernels.

    The four compute threads each run `ada_modulate_striped` /
    `gate_residual_striped` over the rows their launch index owns; the union must
    reproduce, element for element, what `ada_modulate_2d` / `gate_residual_2d`
    produce over the whole tile. Exact equality rather than a tolerance: both
    arms run the same row body, so a difference is a partition bug -- a row
    computed twice, or none at all -- not arithmetic. A sentinel pre-fill makes
    an uncovered row a failure rather than a silent zero.

    This is the shipped path. `mojo_ada_modulate_*` / `mojo_gate_residual_*` in
    `npu_kernel.mojo` read the launch index off the shim and call exactly these.
    `param_stride` is driven at both 0 (the FLUX broadcast) and `row_length` (the
    materialized params), because a striped partition reading a broadcast row is
    the case where a stride mistake would hide.
    """
    var failures = 0
    comptime SENTINEL = Scalar[in_dtype](-31000.0)

    var count = num_rows * row_length
    var param_count = (num_rows - 1) * param_stride + row_length

    var xb = alloc[Scalar[in_dtype]](count).as_unsafe_any_origin()
    var yb = alloc[Scalar[in_dtype]](count).as_unsafe_any_origin()
    var pb = alloc[Scalar[in_dtype]](param_count).as_unsafe_any_origin()
    for i in range(count):
        xb.store(i, Scalar[in_dtype](Float64(i % 97) / 97.0 * 4.0 - 2.0))
        yb.store(i, Scalar[in_dtype](Float64(i % 89) / 89.0 * 3.0 - 1.5))
    for i in range(param_count):
        pb.store(i, Scalar[in_dtype](Float64(i % 71) / 71.0 * 2.0 - 1.0))

    var whole = alloc[Scalar[in_dtype]](count).as_unsafe_any_origin()
    var striped_out = alloc[Scalar[in_dtype]](count).as_unsafe_any_origin()

    var x_t = _rows_tile[in_dtype](xb, num_rows, row_length)
    var y_t = _rows_tile[in_dtype](yb, num_rows, row_length)
    var p_t = _param_tile[in_dtype](pb, num_rows, row_length, param_stride)

    # adaLN modulation.
    for i in range(count):
        whole.store(i, SENTINEL)
        striped_out.store(i, SENTINEL)
    ada_modulate_2d[in_dtype](
        x_t, p_t, p_t, _rows_tile[in_dtype](whole, num_rows, row_length)
    )
    for t in range(5):
        ada_modulate_striped[in_dtype](
            x_t,
            p_t,
            p_t,
            _rows_tile[in_dtype](striped_out, num_rows, row_length),
            UInt32(t),
        )
    var ada_bad = 0
    var ada_uncovered = 0
    for i in range(count):
        if whole.load(i) != striped_out.load(i):
            ada_bad += 1
        if striped_out.load(i) == SENTINEL:
            ada_uncovered += 1
    if ada_bad != 0 or ada_uncovered != 0:
        print(
            "  FAIL",
            name,
            "ada striped mismatch elems=",
            ada_bad,
            "uncovered=",
            ada_uncovered,
        )
        failures += 1

    # Gated residual, including the fp16 clamp arm.
    for i in range(count):
        whole.store(i, SENTINEL)
        striped_out.store(i, SENTINEL)
    gate_residual_2d[in_dtype](
        x_t, y_t, p_t, _rows_tile[in_dtype](whole, num_rows, row_length)
    )
    for t in range(5):
        gate_residual_striped[in_dtype](
            x_t,
            y_t,
            p_t,
            _rows_tile[in_dtype](striped_out, num_rows, row_length),
            UInt32(t),
        )
    var gate_bad = 0
    var gate_uncovered = 0
    for i in range(count):
        if whole.load(i) != striped_out.load(i):
            gate_bad += 1
        if striped_out.load(i) == SENTINEL:
            gate_uncovered += 1
    if gate_bad != 0 or gate_uncovered != 0:
        print(
            "  FAIL",
            name,
            "gate striped mismatch elems=",
            gate_bad,
            "uncovered=",
            gate_uncovered,
        )
        failures += 1

    # The clamped arm, which is a separate manifest row driving the same body.
    for i in range(count):
        whole.store(i, SENTINEL)
        striped_out.store(i, SENTINEL)
    gate_residual_2d[in_dtype, clamp_fp16=True](
        x_t, y_t, p_t, _rows_tile[in_dtype](whole, num_rows, row_length)
    )
    for t in range(5):
        gate_residual_striped[in_dtype, clamp_fp16=True](
            x_t,
            y_t,
            p_t,
            _rows_tile[in_dtype](striped_out, num_rows, row_length),
            UInt32(t),
        )
    var clamp_bad = 0
    for i in range(count):
        if whole.load(i) != striped_out.load(i):
            clamp_bad += 1
    if clamp_bad != 0:
        print("  FAIL", name, "gate clamp striped mismatch elems=", clamp_bad)
        failures += 1

    if failures == 0:
        print(
            "  ok  ",
            name,
            "striped == whole-tile, rows",
            num_rows,
            "param_stride",
            param_stride,
        )

    xb.free()
    yb.free()
    pb.free()
    whole.free()
    striped_out.free()
    return failures


def run_fused_ln[
    in_dtype: DType
](name: String, num_rows: Int, row_length: Int, param_stride: Int) -> Int:
    """The fused arm's bit-identity claim, proven on host.

    Sequential arm: the standalone no-affine layer norm (`_layer_norm`, the
    shipped kernel body) into a temp buffer, then `ada_modulate_2d` over it.
    Fused arm: `_ada_modulate_ln_rows` / `ada_modulate_ln_striped` on the
    PRE-norm input. Every output element must match EXACTLY — the fused path
    rounds the normalized value to `in_dtype` before modulating, which is the
    bit the sequential path stores and reloads. A tolerance here would hide
    the exact defect the device gate is built to refuse.
    """
    comptime SENTINEL = Scalar[in_dtype](-1234.5)
    var eps = Float32(1e-6)
    var n = num_rows * row_length
    var param_len = (num_rows - 1) * param_stride + row_length
    var x = alloc[Scalar[in_dtype]](n).as_unsafe_any_origin()
    var scale = alloc[Scalar[in_dtype]](param_len).as_unsafe_any_origin()
    var shift = alloc[Scalar[in_dtype]](param_len).as_unsafe_any_origin()
    var norm_tmp = alloc[Scalar[in_dtype]](n).as_unsafe_any_origin()
    var out_seq = alloc[Scalar[in_dtype]](n).as_unsafe_any_origin()
    var out_fused = alloc[Scalar[in_dtype]](n).as_unsafe_any_origin()

    fill_x[in_dtype](x, n, 3)
    fill_param[in_dtype](scale, param_len, 3)
    fill_param[in_dtype](shift, param_len, 4)

    var x_t = _rows_tile[in_dtype](x, num_rows, row_length)
    var scale_t = _param_tile[in_dtype](
        scale, num_rows, row_length, param_stride
    )
    var shift_t = _param_tile[in_dtype](
        shift, num_rows, row_length, param_stride
    )
    var norm_t = _rows_tile[in_dtype](norm_tmp, num_rows, row_length)

    for t in range(4):
        _layer_norm(norm_t, x_t, UInt32(t), eps)
    ada_modulate_2d[in_dtype](
        norm_t,
        scale_t,
        shift_t,
        _rows_tile[in_dtype](out_seq, num_rows, row_length),
    )

    var failures = 0

    _ada_modulate_ln_rows[in_dtype](
        x_t,
        scale_t,
        shift_t,
        _rows_tile[in_dtype](out_fused, num_rows, row_length),
        eps,
        0,
        num_rows,
    )
    var whole_bad = 0
    for i in range(n):
        if out_seq[i] != out_fused[i]:
            whole_bad += 1
    if whole_bad != 0:
        print("  FAIL", name, "fused-LN whole-tile mismatch elems=", whole_bad)
        failures += 1

    for i in range(n):
        out_fused[i] = SENTINEL
    for t in range(5):
        ada_modulate_ln_striped[in_dtype](
            x_t,
            scale_t,
            shift_t,
            _rows_tile[in_dtype](out_fused, num_rows, row_length),
            eps,
            UInt32(t),
        )
    var striped_bad = 0
    var uncovered = 0
    for i in range(n):
        if out_seq[i] != out_fused[i]:
            striped_bad += 1
        if out_fused[i] == SENTINEL:
            uncovered += 1
    if striped_bad != 0 or uncovered != 0:
        print(
            "  FAIL",
            name,
            "fused-LN striped mismatch elems=",
            striped_bad,
            "uncovered=",
            uncovered,
        )
        failures += 1

    if failures == 0:
        print("  ok  ", name, "fused == layernorm;modulate, bit-exact")

    x.free()
    scale.free()
    shift.free()
    norm_tmp.free()
    out_seq.free()
    out_fused.free()
    return failures


def main() raises:
    var failures = 0
    failures += run_suite[DType.float16]("fp16")
    failures += run_suite[DType.bfloat16]("bf16")

    # The fused no-affine-LN arm (SKYE_ADA_FUSED_LN): bit-exact against the
    # two-kernel sequence, at the FLUX hidden dim, a ragged tail length, a
    # sub-thread-count row count, and both param layouts.
    print("==== fused LN + adaLN (bit-exact vs sequential) ====")
    failures += run_fused_ln[DType.float16]("fp16 flux bcast", 7, 6144, 0)
    failures += run_fused_ln[DType.float16]("fp16 ragged dense", 5, 333, 333)
    failures += run_fused_ln[DType.float16]("fp16 tiny bcast", 3, 64, 0)
    failures += run_fused_ln[DType.bfloat16]("bf16 bcast", 4, 96, 0)

    # The launch-striped partition the shipped entries use. Row counts that do
    # and do not divide by the four compute threads, and one below the thread
    # count so some indices get an empty run; each at both param strides.
    print("==== launch-striped partition ====")
    for ref rows in [1, 3, 4, 7, 8, 13]:
        failures += run_striped[DType.float16](
            "fp16 rows=" + String(rows) + " bcast", rows, 128, 0
        )
        failures += run_striped[DType.float16](
            "fp16 rows=" + String(rows) + " dense", rows, 128, 128
        )
        failures += run_striped[DType.bfloat16](
            "bf16 rows=" + String(rows) + " bcast", rows, 96, 0
        )

    print()
    if failures == 0:
        print("ALL MODULATION TESTS PASSED")
    else:
        raise Error(String(failures) + " modulation test case(s) FAILED")
