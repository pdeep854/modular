"""Off-device algorithm test for the HVX no-affine LayerNorm kernel.

Runs the kernel's own `_layer_norm` (which drives `LayerNorm3Pass`) on the host
and compares against an independent float64 reference. This proves the ALGORITHM
— the two-pass mean/variance, the mean subtraction, the eps guard, the tail
handling, the row split across launch threads — is numerically correct. It says
nothing about whether the device emits vector instructions for it: the SIMD width
resolves the same on host and device, so a scalar host build would pass this too.
Vectorization is settled separately, by disassembling the v81 object (build.sh's
HVX-mnemonic count) and ultimately on silicon.

The reference accumulates in float64 where the kernel accumulates in reassociated
float32, so the two disagree by ~1e-7 before the down-cast — far below the fp16 /
bf16 output resolution, which is why the tolerance is set at that resolution and
the reported max errors sit well inside it.
"""

from hexagon_ondevice.layernorm_kernel import (
    LayerNorm3Pass,
    RowMajor2D,
    _layer_norm,
)
from layout.coord import Coord
from layout.tile_layout import row_major

from std.math import isfinite, sqrt
from std.memory import alloc

comptime SENTINEL: Float32 = 30000.0
"""A finite value normalized output never takes, pre-written to every output
element so an unwritten (missing) one is detectable after the launch."""


def _fill_seeded[
    dtype: DType
](
    ptr: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin],
    n: Int,
    seed: UInt32,
    scale: Float32,
):
    """Deterministic pseudo-random fill in roughly [-scale, +scale]."""
    var state = seed | 1
    for i in range(n):
        state = state * 1664525 + 1013904223
        var u = Float32(Int(state >> 8) & 0xFFFF) / Float32(0xFFFF)
        ptr.store(i, ((u * 2.0 - 1.0) * scale).cast[dtype]())


def _fill_ramp[
    dtype: DType
](
    ptr: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin],
    num_rows: Int,
    length: Int,
):
    """A linear ramp across each row, offset per row — smoothly varying, so the
    mean is far from zero and the variance is well conditioned."""
    for r in range(num_rows):
        for i in range(length):
            var t = Float32(i) / Float32(length)
            ptr.store(
                r * length + i,
                ((t * 4.0 - 2.0) + Float32(r) * 0.5).cast[dtype](),
            )


def _fill_const[
    dtype: DType
](ptr: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin], n: Int, c: Float32):
    for i in range(n):
        ptr.store(i, c.cast[dtype]())


def _fill_sentinel[
    dtype: DType
](ptr: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin], n: Int):
    for i in range(n):
        ptr.store(i, SENTINEL.cast[dtype]())


def _ref_layer_norm[
    dtype: DType
](
    out_ref: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin],
    input: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin],
    num_rows: Int,
    length: Int,
    eps: Float32,
):
    """Independent float64 reference: mean, then variance about it, then
    normalize. Reads the same stored dtype inputs the kernel reads."""
    for r in range(num_rows):
        var base = r * length

        var s = Float64(0)
        for i in range(length):
            s += Float64(input.load(base + i).cast[DType.float32]())
        var mean = s / Float64(length)

        var vs = Float64(0)
        for i in range(length):
            var d = Float64(input.load(base + i).cast[DType.float32]()) - mean
            vs += d * d
        var variance = vs / Float64(length)

        var inv_std = Float64(1) / sqrt(variance + Float64(eps))
        for i in range(length):
            var o = (
                Float64(input.load(base + i).cast[DType.float32]()) - mean
            ) * inv_std
            out_ref.store(base + i, o.cast[dtype]())


def _run_case[
    dtype: DType
](
    name: String,
    num_rows: Int,
    length: Int,
    input: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin],
    eps: Float32,
    atol: Float32,
    rtol: Float32,
) -> Bool:
    """Drives the real striped `_layer_norm` over the four compute threads, then
    scores it against the float64 reference. Returns True on pass."""
    var count = num_rows * length
    var out = alloc[Scalar[dtype]](count).as_unsafe_any_origin()
    var reference = alloc[Scalar[dtype]](count).as_unsafe_any_origin()

    _fill_sentinel[dtype](out, count)
    _ref_layer_norm[dtype](reference, input, num_rows, length, eps)

    # Stands in for the C-ABI seam: the kernel's entry points build these tiles
    # from the skel's pointers and extents, and `_layer_norm` takes them built.
    var extent = row_major(Coord(num_rows, length))
    var dst = RowMajor2D[dtype](ptr=out, layout=extent)
    var src = RowMajor2D[dtype](ptr=input, layout=extent)

    # Each compute thread writes the rows its launch index owns; the union covers
    # every row exactly once. Threads 4+ (the DMA thread) take none and return.
    for t in range(4):
        _layer_norm(dst, src, UInt32(t), eps)

    var max_err = Float32(0)
    var max_abs = Float32(0)
    var missing = 0
    var nonfinite = 0
    for i in range(count):
        var o = out.load(i).cast[DType.float32]()
        var e = reference.load(i).cast[DType.float32]()
        if not isfinite(o):
            nonfinite += 1
            continue
        var s = SENTINEL
        if (o - s if o > s else s - o) < 1.0:
            missing += 1
        var abs_e = e if e >= 0 else -e
        if abs_e > max_abs:
            max_abs = abs_e
        var diff = o - e
        if diff < 0:
            diff = -diff
        if diff > max_err:
            max_err = diff

    var tol = atol + rtol * max_abs
    var ok = (max_err <= tol) and (missing == 0) and (nonfinite == 0)

    print(
        name,
        "rows",
        num_rows,
        "D",
        length,
        "-> ",
        "PASS" if ok else "FAIL",
        "(max_err=",
        max_err,
        "tol=",
        tol,
        "max|out|=",
        max_abs,
        "missing=",
        missing,
        "nonfinite=",
        nonfinite,
        ")",
    )

    out.free()
    reference.free()
    return ok


def _cases[dtype: DType](label: String, atol: Float32, rtol: Float32) -> Bool:
    """Runs the full shape/edge matrix for one dtype."""
    print("---", label, "---")
    var eps = Float32(1e-6)
    var all_ok = True

    # Real FLUX MMDiT width, single and multi-row.
    var big = alloc[Scalar[dtype]](8 * 6144).as_unsafe_any_origin()
    _fill_seeded[dtype](big, 8 * 6144, 0xF10C, 3.0)
    all_ok = (
        _run_case[dtype]("D6144 x1   ", 1, 6144, big, eps, atol, rtol)
        and all_ok
    )
    all_ok = (
        _run_case[dtype]("D6144 x8   ", 8, 6144, big, eps, atol, rtol)
        and all_ok
    )
    big.free()

    # Small whole-vector width (64 lanes) and two vectors, multi-row so the row
    # split across four threads is exercised on counts that do and do not divide.
    var small = alloc[Scalar[dtype]](12 * 128).as_unsafe_any_origin()
    _fill_seeded[dtype](small, 12 * 128, 0x5EED, 3.0)
    all_ok = (
        _run_case[dtype]("D64 x1     ", 1, 64, small, eps, atol, rtol)
        and all_ok
    )
    all_ok = (
        _run_case[dtype]("D128 x12   ", 12, 128, small, eps, atol, rtol)
        and all_ok
    )
    all_ok = (
        _run_case[dtype]("D128 x10   ", 10, 128, small, eps, atol, rtol)
        and all_ok
    )
    small.free()

    # Tails: widths that are not a multiple of 64, including one below a full
    # vector, so the masked partial-vector path runs in every pass.
    var tail = alloc[Scalar[dtype]](4 * 6145).as_unsafe_any_origin()
    _fill_seeded[dtype](tail, 4 * 6145, 0x7A11, 3.0)
    all_ok = (
        _run_case[dtype]("tail D65   ", 1, 65, tail, eps, atol, rtol) and all_ok
    )
    all_ok = (
        _run_case[dtype]("tail D100  ", 1, 100, tail, eps, atol, rtol)
        and all_ok
    )
    all_ok = (
        _run_case[dtype]("tail D6145 ", 4, 6145, tail, eps, atol, rtol)
        and all_ok
    )
    tail.free()

    # Ramp: smoothly varying, large nonzero mean — a different regime from the
    # zero-mean random fills.
    var ramp = alloc[Scalar[dtype]](3 * 6144).as_unsafe_any_origin()
    _fill_ramp[dtype](ramp, 3, 6144)
    all_ok = (
        _run_case[dtype]("ramp D6144 ", 3, 6144, ramp, eps, atol, rtol)
        and all_ok
    )
    ramp.free()

    # Constant row: variance 0, so the eps guard is the only thing between the
    # reciprocal and a divide-by-zero. Output must be exactly 0, finite.
    var const_row = alloc[Scalar[dtype]](2 * 6144).as_unsafe_any_origin()
    _fill_const[dtype](const_row, 2 * 6144, 1.5)
    all_ok = (
        _run_case[dtype]("const D6144", 2, 6144, const_row, eps, atol, rtol)
        and all_ok
    )
    const_row.free()

    # Large magnitude: a sum of squares that overflows fp16 (~6e4 max) but not the
    # fp32 accumulator, so a passing result is direct evidence the accumulation is
    # fp32. Output stays normalized regardless of input scale.
    var large = alloc[Scalar[dtype]](2 * 6144).as_unsafe_any_origin()
    _fill_seeded[dtype](large, 2 * 6144, 0xB16, 2000.0)
    all_ok = (
        _run_case[dtype]("large D6144", 2, 6144, large, eps, atol, rtol)
        and all_ok
    )
    large.free()

    return all_ok


def main() raises:
    var all_ok = True

    # fp16: ~11 mantissa bits, ulp ~1e-3 at |out|~4.
    all_ok = _cases[DType.float16]("float16", 2e-2, 1e-2) and all_ok

    # bf16: 8 mantissa bits, ulp ~2e-2 at |out|~4 — a coarser tolerance for a
    # coarser type, still far tighter than any algorithm bug would clear.
    all_ok = _cases[DType.bfloat16]("bfloat16", 8e-2, 2e-2) and all_ok

    print("")
    if all_ok:
        print("PASS")
    else:
        print("FAIL")
        raise Error("layernorm algorithm test failed")
