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
# The two fp32 table builders of the FLUX.2 denoise preamble: the timestep
# sinusoidal embedding (diffusers `get_timestep_embedding`) and the per-axis
# rope (cos, sin) table (diffusers `get_1d_rotary_pos_embed`, in the pair
# layout `pack_flux2_rope` consumes and the signed rope kernels read).
#
# Everything here is deliberately SCALAR, which for these two kernels is a
# correctness feature rather than a missed optimization. Vector fp32 on this
# part lowers through qfloat (`+hvx-qfloat`), whose extra-precision multiply
# and convert do not round like one IEEE op each -- the rope dispatch check
# carries a tolerance for exactly that. A scalar fp32/fp64 op is one IEEE
# instruction on both this DSP and any host, so a table built here is
# bit-identical to the same arithmetic run off-device, and the device check
# can demand exact equality instead of a tolerance wide enough to hide a real
# error. The tables are also tiny and built once per generation (rope) or once
# per step over a handful of rows (timestep), so there is no throughput to buy
# back. llama.cpp's Hexagon backend builds its rope cache the same way:
# scalar `cosf`/`sinf` per element, vector code only in the rope *application*.
#
# sin/cos/exp are this file's own, not libm's or `std.math`'s, for two
# reasons. The skel links no libm, so `llvm.sin` -- what `std.math.sin`
# lowers to off-GPU -- would scalarize into an undefined `sinf` and the
# library would refuse to load. And the host reference has to reproduce the
# kernel bit for bit, which it can only do if the op sequence is spelled out
# here: double-precision Cody-Waite reduction and Taylor cores built from
# plain IEEE mul/add/sub (no fma anywhere, so there is nothing a backend
# could contract differently), rounded to fp32 once at the end. Every
# constant is fdlibm's, written to full precision.

from std.ffi import external_call
from std.math import ceildiv
from std.memory import bitcast

comptime F32 = DType.float32
comptime F64 = DType.float64
comptime F16 = DType.float16
comptime I64 = DType.int64

comptime TWO_OVER_PI = Float64(0.63661977236758138243)
"""The double-precision 2/pi (0x3FE45F306DC9C883)."""

comptime RNE_MAGIC = Float64(6755399441055744.0)
"""1.5 * 2^52: adding and subtracting it rounds a double in (-2^51, 2^51) to
the nearest integer (ties to even) using two exact-width IEEE adds, with no
`round`/`rint` call for a bare-metal target to miss."""

comptime PIO2_HI = Float64(1.57079632673412561417)
"""The first 33 bits of pi/2 (0x3FF921FB54400000). With at most 20 quadrant
bits (see `_sincos_f32`), `n * PIO2_HI` is exact: 20 + 33 <= 53."""

comptime PIO2_LO = Float64(6.0771005065061922493e-11)
"""The tail pi/2 - PIO2_HI (0x3DD0B4611A626331)."""

comptime LOG2_E = Float64(1.4426950408889633870)
"""The double-precision log2(e) (0x3FF71547652B82FE)."""

comptime LN2_HI = Float64(0.69314718036912381649)
"""The first 33 bits of ln(2) (0x3FE62E42FEE00000); `n * LN2_HI` is exact for
the |n| <= 128 this file's exp ever produces."""

comptime LN2_LO = Float64(1.9082149292705877000e-10)
"""The tail ln(2) - LN2_HI (0x3DEA39EF35793C76)."""

comptime SINCOS_MAX_ARG = Float32(823549.0)
"""|theta| bound for `_sincos_f32`: pi/4 * 2^20, past which the quadrant
count outgrows the 20 bits that keep `n * PIO2_HI` exact. The dispatch stubs
bound positions and frequencies well inside it."""


@always_inline
def _rne_integer(x: Float64) -> Float64:
    """Rounds to the nearest integer, ties to even, for |x| < 2^51."""
    return (x + RNE_MAGIC) - RNE_MAGIC


@always_inline
def _sin_core(r: Float64) -> Float64:
    """Evaluates sin(r) for |r| <= pi/4 + epsilon: fdlibm's kernel polynomial.

    Odd in `r` exactly -- every term carries an odd power and IEEE negation is
    exact -- which is what makes the table kernels' sign symmetry a bit-exact
    invariant rather than an approximate one.
    """
    var z = r * r
    var p = Float64(1.58969099521155010221e-10)
    p = p * z + -2.50507602534068634195e-08
    p = p * z + 2.75573137070700676789e-06
    p = p * z + -1.98412698298579493134e-04
    p = p * z + 8.33333333332248946124e-03
    p = p * z + -1.66666666666666324348e-01
    return r + r * (z * p)


@always_inline
def _cos_core(r: Float64) -> Float64:
    """Evaluates cos(r) for |r| <= pi/4 + epsilon: fdlibm's kernel polynomial, even in
    `r` exactly."""
    var z = r * r
    var p = Float64(-1.13596475577881948265e-11)
    p = p * z + 2.08757232129817482790e-09
    p = p * z + -2.75573143513906633035e-07
    p = p * z + 2.48015872894767294178e-05
    p = p * z + -1.38888888888741095749e-03
    p = p * z + 4.16666666666666019037e-02
    return 1.0 - (z * 0.5 - z * z * p)


@always_inline
def _sincos_f32(theta: Float32) -> SIMD[F32, 2]:
    """Returns (sin, cos) of an fp32 angle, each correctly rounded for |theta| <=
    `SINCOS_MAX_ARG`.

    The double-precision path is the determinism argument, not an accuracy
    flourish: reduction error ~2^-62 and core error ~2^-52 sit far below
    fp32's 2^-24 quantum, so the one rounding that can differ between targets
    is the final cast -- and both targets round it the same way. Every
    intermediate op is a single IEEE mul/add/sub, so the sequence has no
    contraction or libcall for a target to resolve differently.
    """
    var x = Float64(theta)
    var n = _rne_integer(x * TWO_OVER_PI)
    var quadrant = Int64(n) & 3

    # Cody-Waite: the high product is exact (33-bit constant, <= 20-bit n), so
    # the subtraction cancels without error; only the low tail's product
    # rounds, ~2^-62 from the result.
    var r = x - n * PIO2_HI
    var tail = n * PIO2_LO
    r = r - tail

    var s = _sin_core(r)
    var c = _cos_core(r)

    if quadrant == 0:
        return SIMD[F32, 2](Float32(s), Float32(c))
    elif quadrant == 1:
        return SIMD[F32, 2](Float32(c), Float32(-s))
    elif quadrant == 2:
        return SIMD[F32, 2](Float32(-s), Float32(-c))
    else:
        return SIMD[F32, 2](Float32(-c), Float32(s))


@always_inline
def _exp_f32(arg: Float32) -> Float32:
    """Computes e^arg for fp32, correctly rounded over the band domain [-92, 0].

    The same double-internal, plain-mul/add construction as `_sincos_f32`:
    reduce against ln(2) split exactly, run a Taylor core on |r| <= ln(2)/2,
    scale by a bit-assembled 2^n, round to fp32 once. Domain-limited to what
    the sinusoid band needs (n in [-133, 0], results normal in fp64), so no
    overflow or denormal handling.
    """
    var x = Float64(arg)
    var n = _rne_integer(x * LOG2_E)
    var r = x - n * LN2_HI
    var tail = n * LN2_LO
    r = r - tail

    # Taylor to r^12/12!: tail < 2^-52 of the result for |r| <= 0.347.
    var p = Float64(2.0876756987868098979e-09)
    p = p * r + 2.5052108385441718775e-08
    p = p * r + 2.7557319223985890653e-07
    p = p * r + 2.7557319223985890653e-06
    p = p * r + 2.4801587301587301587e-05
    p = p * r + 1.9841269841269841270e-04
    p = p * r + 1.3888888888888888889e-03
    p = p * r + 8.3333333333333333333e-03
    p = p * r + 4.1666666666666666667e-02
    p = p * r + 1.6666666666666666667e-01
    p = p * r + 0.5
    p = p * r + 1.0
    p = p * r + 1.0

    # 2^n assembled in the exponent field; n stays comfortably inside
    # fp64's normal range on this domain.
    var scale = _pow2_f64(Int64(n))
    return Float32(p * scale)


@always_inline
def _pow2_f64(n: Int64) -> Float64:
    """Computes 2^n for |n| small enough that the result is a normal double."""
    var bits = Scalar[DType.uint64](Int64(1023) + n) << 52
    return bitcast[F64](bits)


@always_inline
def _sinusoid_embed_rows(
    output: UnsafePointer[Scalar[F16], MutUnsafeAnyOrigin],
    t: UnsafePointer[Scalar[F32], MutUnsafeAnyOrigin],
    half_dim: Int,
    time_scale: Float32,
    neg_ln_max_period: Float32,
    flip: Int,
    start_row: Int,
    end_row: Int,
):
    """Builds the rows `[start_row, end_row)` of the timestep sinusoidal
    embedding, `[num_rows] -> [num_rows, 2*half_dim]` fp16.

    Row `b` holds `cos(arg_b * band_i)` over one half and `sin(arg_b *
    band_i)` over the other, `flip != 0` putting the cosines first (FLUX's
    `flip_sin_to_cos=True`); `band_i = exp(neg_ln_max_period * i /
    half_dim)`. Two measured quirks of the staged graph are reproduced
    deliberately rather than "fixed": the argument is `t * time_scale` pushed
    through an fp16 round-trip (the reference numerics path -- the module
    casts the scaled timestep to its dtype before embedding it in fp32), and
    the band exponent folds as `(coeff * i) / half_dim`, multiply before
    divide. `neg_ln_max_period` arrives precomputed because the graph's own
    `log(max_period)` is constant-folded before the kernel exists, and the
    skel links no libm to compute one.
    """
    var cos_base = 0 if flip != 0 else half_dim
    var sin_base = half_dim if flip != 0 else 0
    for b in range(start_row, end_row):
        var scaled = t[b] * time_scale
        var arg = scaled.cast[F16]().cast[F32]()
        var out_row = output + b * 2 * half_dim
        for i in range(half_dim):
            var exponent = (neg_ln_max_period * Float32(i)) / Float32(half_dim)
            var band = _exp_f32(exponent)
            var sc = _sincos_f32(arg * band)
            out_row[cos_base + i] = sc[1].cast[F16]()
            out_row[sin_base + i] = sc[0].cast[F16]()


def sinusoid_embed_2d(
    output: UnsafePointer[Scalar[F16], MutUnsafeAnyOrigin],
    t: UnsafePointer[Scalar[F32], MutUnsafeAnyOrigin],
    num_rows: Int,
    half_dim: Int,
    time_scale: Float32,
    neg_ln_max_period: Float32,
    flip: Int,
):
    """Builds the whole timestep sinusoidal embedding table."""
    _sinusoid_embed_rows(
        output, t, half_dim, time_scale, neg_ln_max_period, flip, 0, num_rows
    )


@always_inline
def _rope_table_rows(
    output: UnsafePointer[Scalar[F32], MutUnsafeAnyOrigin],
    positions: UnsafePointer[Scalar[I64], MutUnsafeAnyOrigin],
    freqs: UnsafePointer[Scalar[F32], MutUnsafeAnyOrigin],
    half_dim: Int,
    pos_stride: Int,
    start_row: Int,
    end_row: Int,
):
    """Builds the rows `[start_row, end_row)` of one rope axis's table,
    `[num_rows, 2*half_dim]` fp32 in the (cos, sin) pair layout the signed rope
    kernels read: `out[s, 2k] = cos(pos_s * freq_k)`, `out[s, 2k + 1] =
    sin(pos_s * freq_k)`.

    `positions` are the raw si64 ids -- the cast to fp32 happens at the load,
    so the graph's standalone cast kernel disappears into this one -- and
    step by `pos_stride`, which lets one dispatch read its axis's column
    straight out of the packed `[rows, num_axes]` id tensor with no slice
    materialized. `freqs` is the `[half_dim]` frequency row the graph
    constant-folds; nothing here recomputes it.
    """
    for s in range(start_row, end_row):
        var pos = positions[s * pos_stride].cast[F32]()
        var out_row = output + s * 2 * half_dim
        for k in range(half_dim):
            var sc = _sincos_f32(pos * freqs[k])
            out_row[2 * k] = sc[1]
            out_row[2 * k + 1] = sc[0]


def rope_table_2d(
    output: UnsafePointer[Scalar[F32], MutUnsafeAnyOrigin],
    positions: UnsafePointer[Scalar[I64], MutUnsafeAnyOrigin],
    freqs: UnsafePointer[Scalar[F32], MutUnsafeAnyOrigin],
    num_rows: Int,
    half_dim: Int,
    pos_stride: Int,
):
    """Builds one rope axis's whole table."""
    _rope_table_rows(
        output, positions, freqs, half_dim, pos_stride, 0, num_rows
    )


# ===--------------------------------------------------------------------------===
# Device services + the launch-striped entries.
#
# The `external_call`s are the same `qshim` queries `npu_kernel.mojo` spells out,
# repeated here because a standalone `mojo build --emit=object` for the bare-metal
# Hexagon triple cannot reach the overlay. They are undefined until the link binds
# them against `qshim_qurt.c`; a host build that never calls an entry leaves them
# unreferenced, which is what lets the two table builders be tested off-device.
#
# The entries below are what the FLUX dispatch path reaches (megakernel slice 4):
# the rows divide across the launch's four compute threads by launch index, so
# each participant builds the run of rows it owns and the union covers every row
# exactly once. Both are pure per-row functions of their inputs, so the split
# needs no reduction or barrier.
# ===--------------------------------------------------------------------------===

comptime NUM_COMPUTE_THREADS: UInt32 = 4
"""Threads of a core that compute. The fifth is the part's DMA thread."""

comptime QSHIM_QUERY_THREAD_ID: UInt32 = 0
comptime QSHIM_QUERY_NUM_THREADS: UInt32 = 1


@always_inline
def _qshim_query(attribute: UInt32) -> Int64:
    """Reads one device attribute, as the overlay's `_qshim_query_u8` does."""
    var value: Int64 = 0
    _ = external_call["qshimQuery", UInt32](attribute, UnsafePointer(to=value))
    return value


@always_inline
def _npu_thread_idx() -> Int:
    """Index of this thread within the core, 0-based (narrowed to 8 bits)."""
    return Int(UInt8(_qshim_query(QSHIM_QUERY_THREAD_ID)))


@always_inline
def _qurt_thread_id() -> UInt32:
    """The RTOS thread this code is running on, read only to tell threads
    apart — a launch index alone proves nothing about how many threads back
    those indices."""
    return external_call["qurt_thread_get_id", UInt32]()


@always_inline
def _thread_rows(num_rows: Int, thread_id: UInt32) -> Tuple[Int, Int]:
    """The `[start, end)` run of rows this launch index owns.

    Each of the four compute threads owns a contiguous run; the fifth (DMA)
    thread and any launch index past the row count get the empty run `(0, 0)`,
    which the row body then loops zero times over. The split stays hand-written:
    `.distribute()` would give a *cyclic* partition, and this part wants blocked
    (KB `kernels/distribute-is-cyclic-this-part-wants-blocked`).

    `thread_id` is an argument rather than a shim read so the partition is
    testable off-device; only the entries below read the shim.
    """
    if thread_id >= NUM_COMPUTE_THREADS:
        return (0, 0)
    var rows_per_thread = ceildiv(num_rows, Int(NUM_COMPUTE_THREADS))
    var start_row = Int(thread_id) * rows_per_thread
    if start_row >= num_rows:
        return (0, 0)
    var end_row = start_row + rows_per_thread
    if end_row > num_rows:
        end_row = num_rows
    return (start_row, end_row)


@always_inline
def sinusoid_embed_striped(
    output: UnsafePointer[Scalar[F16], MutUnsafeAnyOrigin],
    t: UnsafePointer[Scalar[F32], MutUnsafeAnyOrigin],
    num_rows: Int,
    half_dim: Int,
    time_scale: Float32,
    neg_ln_max_period: Float32,
    flip: Int,
    thread_id: UInt32,
):
    """Builds the embedding rows `thread_id` owns."""
    var rows = _thread_rows(num_rows, thread_id)
    _sinusoid_embed_rows(
        output,
        t,
        half_dim,
        time_scale,
        neg_ln_max_period,
        flip,
        rows[0],
        rows[1],
    )


@always_inline
def rope_table_striped(
    output: UnsafePointer[Scalar[F32], MutUnsafeAnyOrigin],
    positions: UnsafePointer[Scalar[I64], MutUnsafeAnyOrigin],
    freqs: UnsafePointer[Scalar[F32], MutUnsafeAnyOrigin],
    num_rows: Int,
    half_dim: Int,
    pos_stride: Int,
    thread_id: UInt32,
):
    """Builds the rope table rows `thread_id` owns."""
    var rows = _thread_rows(num_rows, thread_id)
    _rope_table_rows(
        output, positions, freqs, half_dim, pos_stride, rows[0], rows[1]
    )


@always_inline
def _sincos_witness(
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    thread_idx: Int,
):
    """The softmax/silu witness layout with both width slots deliberately 1:
    these kernels are scalar by design, and the exact-answer comparison is
    what proves the arithmetic stayed scalar-IEEE (qfloat vector arithmetic
    could not reproduce the bits)."""
    if thread_idx == 0:
        witness.store(0, UInt32(1))
        witness.store(1, UInt32(1))
        witness.store(2, UInt32(_qshim_query(QSHIM_QUERY_NUM_THREADS)))
    witness.store(3 + thread_idx, _qurt_thread_id())


@always_inline
def sinusoid_embed_entry(
    output: UnsafePointer[Scalar[F16], MutUnsafeAnyOrigin],
    t: UnsafePointer[Scalar[F32], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    num_rows: Int,
    half_dim: Int,
    time_scale: Float32,
    neg_ln_max_period: Float32,
    flip: Int,
):
    """Timestep sinusoidal embedding over the rows this launch index owns."""
    var thread_idx = _npu_thread_idx()
    _sincos_witness(witness, thread_idx)
    sinusoid_embed_striped(
        output,
        t,
        num_rows,
        half_dim,
        time_scale,
        neg_ln_max_period,
        flip,
        UInt32(thread_idx),
    )


@always_inline
def rope_table_entry(
    output: UnsafePointer[Scalar[F32], MutUnsafeAnyOrigin],
    positions: UnsafePointer[Scalar[I64], MutUnsafeAnyOrigin],
    freqs: UnsafePointer[Scalar[F32], MutUnsafeAnyOrigin],
    witness: UnsafePointer[Scalar[DType.uint32], MutUnsafeAnyOrigin],
    num_rows: Int,
    half_dim: Int,
    pos_stride: Int,
):
    """One rope axis's table over the rows this launch index owns."""
    var thread_idx = _npu_thread_idx()
    _sincos_witness(witness, thread_idx)
    rope_table_striped(
        output,
        positions,
        freqs,
        num_rows,
        half_dim,
        pos_stride,
        UInt32(thread_idx),
    )
