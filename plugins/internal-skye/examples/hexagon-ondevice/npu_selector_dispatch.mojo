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
"""Dispatches pre-signed on-device NPU selectors from Mojo.

The `mo.*` registrations in `max/kernels/src/hexagon_ondevice/` are host-side
Mojo stubs that reach a kernel through `load_function(function_name=..., asm="")`
plus `enqueue_function`, on the argument that `QuRTDeviceContext::loadFunction`
resolves a name against `NPUKernels.def` and ignores the code. That argument had
never been run: every dispatch confirmed on the part so far came from the C++
drivers beside this file. This runs it.

Each selector below mirrors one registration's launch exactly -- the same
argument order, the same grid of one core, the same low-bit thread bitmap, the
same trailing witness buffer. `hexagon_ondevice/dispatch.mojo` and
`runtime/lib/qurt/NPUKernels.def` are the source of truth; this driver restates
the contract rather than importing it because it cross-compiles for Windows
ARM64 against the base stdlib alone.

Four of the six registered families, so a failure is attributable: they share the
whole Mojo dispatch path and differ only in the kernel, so some passing and
others failing places the fault in the kernel and not in the path. `groupnorm`
and `mha_no_cache` are left out because the value of another case here is the
dispatch it exercises, and a host reference for an online-softmax attention or a
grouped normalization is where a wrong reference would start manufacturing false
failures instead.

Three guards run unconditionally on every output, because a run that prints a
number is not a run that computed one: no element may be left at the seed the
output was staged with, every element must be finite, and the output must carry
more than one distinct value.
"""

from std.gpu.host import DeviceContext, Dim
from std.math import exp, sqrt


# One FastRPC session addresses one NSP core, so a launch can only name that
# core; a wider grid is refused by the host packer.
comptime NPU_GRID_CORES: Int = 1

# The block's X extent is a bitmap of the core's threads, which must set its low
# bits with no gaps. Five threads: four the kernels compute on, plus the DMA one.
comptime NPU_THREAD_MASK: Int = 0x1F

# Vector width, accumulator width, participant count, one id per thread, and
# then the per-thread cycle bracket `mojo_matmul_f16` (`npu_kernel.mojo:1978`)
# and `mojo_mha_no_cache_f16` (`:3101`) write UNCONDITIONALLY -- no runtime
# gate, no build flag. One size serves every selector this file dispatches, so
# it holds the largest writer: at `3 + 5` the `matmul` leg allocated 8 words
# against the 32 that kernel writes, and every launch of it put 96 bytes past
# the allocation.
comptime NPU_WITNESS_WORDS: Int = 3 + 5 + 4 * 6

# See `hexagon_ondevice/dispatch.mojo` for why the allocation is in bytes
# (P5 M4.2 order 091): the allocator consumes this argument as a byte size.
comptime NPU_WITNESS_BYTES: Int = NPU_WITNESS_WORDS * 4

# What an output is staged with, so an element the kernel never wrote is
# distinguishable from one it did. Outside the range of any correct result here.
comptime SEED: Float32 = -7.0

comptime DT = DType.float16

# Half precision carries ~11 bits of mantissa, so a relative error this size
# admits a different summation order and nothing more.
comptime TOLERANCE: Float32 = 4e-3

# Paired with the relative one for outputs that pass near zero, where a relative
# error is dominated by half precision's own spacing rather than by the
# computation. A softmax row of this width sits around 1/64.
comptime ABSOLUTE_TOLERANCE: Float32 = 1e-3


def npu_selector_sig():
    """Stands in for the signature `load_function` asks for.

    `DeviceExternalFunction` carries no argument types, so the signature never
    reaches the launch: arguments are packed positionally by width and the host
    packer checks them against the selector's declared arity.
    """
    pass


def report(
    name: String,
    residual: Int,
    nonfinite: Int,
    distinct: Int,
    wrong: Int,
    worst_abs: Float32,
    worst_rel: Float32,
) -> Bool:
    # Both errors, neither conditioned on the other: a single figure that falls
    # back to the absolute one near zero reads as bit-exactness when it is only
    # smallness, and a reader cannot tell which they were shown.
    print(
        name,
        ": residual=",
        residual,
        " non_finite=",
        nonfinite,
        " distinct=",
        distinct,
        " wrong=",
        wrong,
        " max_abs_err=",
        worst_abs,
        " max_rel_err=",
        worst_rel,
        sep="",
    )
    var ok = residual == 0 and nonfinite == 0 and distinct > 1 and wrong == 0
    print(name, ": PASS" if ok else ": FAIL", sep="")
    return ok


# Rules out NaN and both infinities without a math import: only a NaN compares
# unequal to itself, and only an infinity has a non-zero self-difference.
def is_finite(v: Scalar[DT]) -> Bool:
    return v == v and v - v == 0


def absolute(v: Float32) -> Float32:
    return v if v >= 0 else -v


def relative(err: Float32, want: Float32) -> Float32:
    """`err` against the magnitude it was measured on, or itself at zero."""
    var scale = absolute(want)
    return err / scale if scale > 0 else err


def within_tolerance(err: Float32, rel: Float32) -> Bool:
    """Either bound is enough, which is what makes a near-zero output judgeable.

    A softmax row of this width sits around 1/64, where half precision's own
    spacing already exceeds the relative bound; requiring the relative one there
    would fail a correct kernel.
    """
    return err <= ABSOLUTE_TOLERANCE or rel <= TOLERANCE


def run_rms_norm(ctx: DeviceContext) raises -> Bool:
    """Mirrors `RMSNorm.execute`'s launch in `registrations.mojo`."""
    comptime ROWS = 3
    comptime LENGTH = 64
    comptime EPS: Float32 = 1e-6
    comptime OFFSET: Float32 = 0.0

    var out_host = ctx.enqueue_create_host_buffer[DT](ROWS * LENGTH)
    var in_host = ctx.enqueue_create_host_buffer[DT](ROWS * LENGTH)
    var gamma_host = ctx.enqueue_create_host_buffer[DT](LENGTH)
    var w_host = ctx.enqueue_create_host_buffer[DType.uint32](NPU_WITNESS_BYTES)

    for row in range(ROWS):
        for i in range(LENGTH):
            in_host[row * LENGTH + i] = Scalar[DT]((row + i) % 4 + 1)
    for i in range(LENGTH):
        gamma_host[i] = Scalar[DT](Float32(i % 3 + 1) * 0.5)
    for i in range(ROWS * LENGTH):
        out_host[i] = Scalar[DT](SEED)
    for i in range(NPU_WITNESS_WORDS):
        w_host[i] = 0

    var out_dev = ctx.enqueue_create_buffer[DT](ROWS * LENGTH)
    var in_dev = ctx.enqueue_create_buffer[DT](ROWS * LENGTH)
    var gamma_dev = ctx.enqueue_create_buffer[DT](LENGTH)
    var w_dev = ctx.enqueue_create_buffer[DType.uint32](NPU_WITNESS_BYTES)
    out_dev.enqueue_copy_from(out_host)
    in_dev.enqueue_copy_from(in_host)
    gamma_dev.enqueue_copy_from(gamma_host)
    w_dev.enqueue_copy_from(w_host)
    ctx.synchronize()

    var f = ctx.load_function[npu_selector_sig](
        function_name=String("rms_norm"), asm=String("")
    )
    ctx.enqueue_function(
        f,
        out_dev.unsafe_ptr(),
        in_dev.unsafe_ptr(),
        gamma_dev.unsafe_ptr(),
        w_dev.unsafe_ptr(),
        Int32(ROWS),
        Int32(LENGTH),
        EPS,
        OFFSET,
        Int32(1),
        grid_dim=Dim(NPU_GRID_CORES),
        block_dim=Dim(NPU_THREAD_MASK),
    )
    ctx.synchronize()

    out_dev.enqueue_copy_to(out_host)
    w_dev.enqueue_copy_to(w_host)
    ctx.synchronize()
    print(
        "rms_norm witness: simd_width=",
        w_host[0],
        " accum_width=",
        w_host[1],
        " participants=",
        w_host[2],
        sep="",
    )

    var residual = 0
    var nonfinite = 0
    var distinct = 0
    var wrong = 0
    var worst_abs = Float32(0)
    var worst_rel = Float32(0)
    for i in range(ROWS * LENGTH):
        var got = out_host[i]
        if Float32(got) == SEED:
            residual += 1
        if not is_finite(got):
            nonfinite += 1
        var seen = False
        for j in range(i):
            if out_host[j] == got:
                seen = True
        if not seen:
            distinct += 1

    for row in range(ROWS):
        var mean_square = Float32(0)
        for i in range(LENGTH):
            var x = Float32(in_host[row * LENGTH + i])
            mean_square += x * x
        mean_square /= Float32(LENGTH)
        var scale = Float32(1.0) / sqrt(mean_square + EPS)
        for i in range(LENGTH):
            var want = (
                Float32(in_host[row * LENGTH + i])
                * scale
                * (Float32(gamma_host[i]) + OFFSET)
            )
            var err = absolute(Float32(out_host[row * LENGTH + i]) - want)
            var rel = relative(err, want)
            if err > worst_abs:
                worst_abs = err
            if rel > worst_rel:
                worst_rel = rel
            if not within_tolerance(err, rel):
                wrong += 1

    return report(
        "rms_norm", residual, nonfinite, distinct, wrong, worst_abs, worst_rel
    )


def run_matmul(ctx: DeviceContext) raises -> Bool:
    """Mirrors `Matmul.execute`'s launch in `registrations.mojo`."""
    comptime ROWS = 4
    comptime COLS = 8

    # The kernel reads the inner extent a vector at a time, so the manifest
    # requires a multiple of 64.
    comptime INNER = 64

    var c_host = ctx.enqueue_create_host_buffer[DT](ROWS * COLS)
    var a_host = ctx.enqueue_create_host_buffer[DT](ROWS * INNER)
    var b_host = ctx.enqueue_create_host_buffer[DT](COLS * INNER)
    var w_host = ctx.enqueue_create_host_buffer[DType.uint32](NPU_WITNESS_WORDS)

    # Small integers: the largest dot product is 64 * 3 * 3 = 576, and every
    # partial sum below it is an integer half precision represents exactly.
    for row in range(ROWS):
        for k in range(INNER):
            a_host[row * INNER + k] = Scalar[DT]((row + k) % 4)
    for col in range(COLS):
        for k in range(INNER):
            b_host[col * INNER + k] = Scalar[DT]((col + k) % 3 + 1)
    for i in range(ROWS * COLS):
        c_host[i] = Scalar[DT](SEED)
    for i in range(NPU_WITNESS_WORDS):
        w_host[i] = 0

    var c_dev = ctx.enqueue_create_buffer[DT](ROWS * COLS)
    var a_dev = ctx.enqueue_create_buffer[DT](ROWS * INNER)
    var b_dev = ctx.enqueue_create_buffer[DT](COLS * INNER)
    var w_dev = ctx.enqueue_create_buffer[DType.uint32](NPU_WITNESS_BYTES)
    c_dev.enqueue_copy_from(c_host)
    a_dev.enqueue_copy_from(a_host)
    b_dev.enqueue_copy_from(b_host)
    w_dev.enqueue_copy_from(w_host)
    ctx.synchronize()

    var f = ctx.load_function[npu_selector_sig](
        function_name=String("matmul"), asm=String("")
    )
    ctx.enqueue_function(
        f,
        c_dev.unsafe_ptr(),
        a_dev.unsafe_ptr(),
        b_dev.unsafe_ptr(),
        w_dev.unsafe_ptr(),
        Int32(ROWS),
        Int32(COLS),
        Int32(INNER),
        grid_dim=Dim(NPU_GRID_CORES),
        block_dim=Dim(NPU_THREAD_MASK),
    )
    ctx.synchronize()

    c_dev.enqueue_copy_to(c_host)
    w_dev.enqueue_copy_to(w_host)
    ctx.synchronize()
    print(
        "matmul witness: simd_width=",
        w_host[0],
        " accum_width=",
        w_host[1],
        " participants=",
        w_host[2],
        sep="",
    )

    var residual = 0
    var nonfinite = 0
    var distinct = 0
    var wrong = 0
    var worst_abs = Float32(0)
    var worst_rel = Float32(0)
    for i in range(ROWS * COLS):
        var got = c_host[i]
        if Float32(got) == SEED:
            residual += 1
        if not is_finite(got):
            nonfinite += 1
        var seen = False
        for j in range(i):
            if c_host[j] == got:
                seen = True
        if not seen:
            distinct += 1

    for row in range(ROWS):
        for col in range(COLS):
            var want = Float32(0)
            for k in range(INNER):
                want += Float32(a_host[row * INNER + k]) * Float32(
                    b_host[col * INNER + k]
                )
            # Exact by construction, so anything above the tolerance is a
            # different computation and not a rounding difference.
            var err = absolute(Float32(c_host[row * COLS + col]) - want)
            var rel = relative(err, want)
            if err > worst_abs:
                worst_abs = err
            if rel > worst_rel:
                worst_rel = rel
            if not within_tolerance(err, rel):
                wrong += 1

    return report(
        "matmul", residual, nonfinite, distinct, wrong, worst_abs, worst_rel
    )


def run_softmax(ctx: DeviceContext) raises -> Bool:
    """Mirrors `Softmax.execute`'s launch in `registrations.mojo`."""
    comptime ROWS = 3
    comptime LENGTH = 64

    var out_host = ctx.enqueue_create_host_buffer[DT](ROWS * LENGTH)
    var in_host = ctx.enqueue_create_host_buffer[DT](ROWS * LENGTH)
    var w_host = ctx.enqueue_create_host_buffer[DType.uint32](NPU_WITNESS_WORDS)

    # A small integer span, so the exponentials stay well inside half
    # precision's range and the row sums are not dominated by one term.
    for row in range(ROWS):
        for i in range(LENGTH):
            in_host[row * LENGTH + i] = Scalar[DT]((row + i) % 5 - 2)
    for i in range(ROWS * LENGTH):
        out_host[i] = Scalar[DT](SEED)
    for i in range(NPU_WITNESS_WORDS):
        w_host[i] = 0

    var out_dev = ctx.enqueue_create_buffer[DT](ROWS * LENGTH)
    var in_dev = ctx.enqueue_create_buffer[DT](ROWS * LENGTH)
    var w_dev = ctx.enqueue_create_buffer[DType.uint32](NPU_WITNESS_BYTES)
    out_dev.enqueue_copy_from(out_host)
    in_dev.enqueue_copy_from(in_host)
    w_dev.enqueue_copy_from(w_host)
    ctx.synchronize()

    var f = ctx.load_function[npu_selector_sig](
        function_name=String("softmax"), asm=String("")
    )
    ctx.enqueue_function(
        f,
        out_dev.unsafe_ptr(),
        in_dev.unsafe_ptr(),
        w_dev.unsafe_ptr(),
        Int32(ROWS),
        Int32(LENGTH),
        grid_dim=Dim(NPU_GRID_CORES),
        block_dim=Dim(NPU_THREAD_MASK),
    )
    ctx.synchronize()

    out_dev.enqueue_copy_to(out_host)
    w_dev.enqueue_copy_to(w_host)
    ctx.synchronize()
    print(
        "softmax witness: simd_width=",
        w_host[0],
        " accum_width=",
        w_host[1],
        " participants=",
        w_host[2],
        sep="",
    )

    var residual = 0
    var nonfinite = 0
    var distinct = 0
    var wrong = 0
    var worst_abs = Float32(0)
    var worst_rel = Float32(0)
    for i in range(ROWS * LENGTH):
        var got = out_host[i]
        if Float32(got) == SEED:
            residual += 1
        if not is_finite(got):
            nonfinite += 1
        var seen = False
        for j in range(i):
            if out_host[j] == got:
                seen = True
        if not seen:
            distinct += 1

    for row in range(ROWS):
        var peak = Float32(in_host[row * LENGTH])
        for i in range(LENGTH):
            var x = Float32(in_host[row * LENGTH + i])
            if x > peak:
                peak = x
        var total = Float32(0)
        for i in range(LENGTH):
            total += exp(Float32(in_host[row * LENGTH + i]) - peak)
        for i in range(LENGTH):
            var want = exp(Float32(in_host[row * LENGTH + i]) - peak) / total
            var err = absolute(Float32(out_host[row * LENGTH + i]) - want)
            var rel = relative(err, want)
            if err > worst_abs:
                worst_abs = err
            if rel > worst_rel:
                worst_rel = rel
            if not within_tolerance(err, rel):
                wrong += 1

    return report(
        "softmax", residual, nonfinite, distinct, wrong, worst_abs, worst_rel
    )


def run_layernorm(ctx: DeviceContext) raises -> Bool:
    """Mirrors `LayerNorm.execute`'s launch in `registrations.mojo`.

    The selector applies no affine, and the registration passes neither `gamma`
    nor `beta`, so the reference here has none either.
    """
    comptime ROWS = 3
    comptime LENGTH = 64
    comptime EPS: Float32 = 1e-5

    var out_host = ctx.enqueue_create_host_buffer[DT](ROWS * LENGTH)
    var in_host = ctx.enqueue_create_host_buffer[DT](ROWS * LENGTH)
    var w_host = ctx.enqueue_create_host_buffer[DType.uint32](NPU_WITNESS_WORDS)

    for row in range(ROWS):
        for i in range(LENGTH):
            in_host[row * LENGTH + i] = Scalar[DT]((row + i) % 7 + 1)
    for i in range(ROWS * LENGTH):
        out_host[i] = Scalar[DT](SEED)
    for i in range(NPU_WITNESS_WORDS):
        w_host[i] = 0

    var out_dev = ctx.enqueue_create_buffer[DT](ROWS * LENGTH)
    var in_dev = ctx.enqueue_create_buffer[DT](ROWS * LENGTH)
    var w_dev = ctx.enqueue_create_buffer[DType.uint32](NPU_WITNESS_BYTES)
    out_dev.enqueue_copy_from(out_host)
    in_dev.enqueue_copy_from(in_host)
    w_dev.enqueue_copy_from(w_host)
    ctx.synchronize()

    var f = ctx.load_function[npu_selector_sig](
        function_name=String("layernorm"), asm=String("")
    )
    ctx.enqueue_function(
        f,
        out_dev.unsafe_ptr(),
        in_dev.unsafe_ptr(),
        w_dev.unsafe_ptr(),
        Int32(ROWS),
        Int32(LENGTH),
        EPS,
        grid_dim=Dim(NPU_GRID_CORES),
        block_dim=Dim(NPU_THREAD_MASK),
    )
    ctx.synchronize()

    out_dev.enqueue_copy_to(out_host)
    w_dev.enqueue_copy_to(w_host)
    ctx.synchronize()
    print(
        "layernorm witness: simd_width=",
        w_host[0],
        " accum_width=",
        w_host[1],
        " participants=",
        w_host[2],
        sep="",
    )

    var residual = 0
    var nonfinite = 0
    var distinct = 0
    var wrong = 0
    var worst_abs = Float32(0)
    var worst_rel = Float32(0)
    for i in range(ROWS * LENGTH):
        var got = out_host[i]
        if Float32(got) == SEED:
            residual += 1
        if not is_finite(got):
            nonfinite += 1
        var seen = False
        for j in range(i):
            if out_host[j] == got:
                seen = True
        if not seen:
            distinct += 1

    for row in range(ROWS):
        var mean = Float32(0)
        for i in range(LENGTH):
            mean += Float32(in_host[row * LENGTH + i])
        mean /= Float32(LENGTH)
        var variance = Float32(0)
        for i in range(LENGTH):
            var centered = Float32(in_host[row * LENGTH + i]) - mean
            variance += centered * centered
        variance /= Float32(LENGTH)
        var scale = Float32(1.0) / sqrt(variance + EPS)
        for i in range(LENGTH):
            var want = (Float32(in_host[row * LENGTH + i]) - mean) * scale
            var err = absolute(Float32(out_host[row * LENGTH + i]) - want)
            var rel = relative(err, want)
            if err > worst_abs:
                worst_abs = err
            if rel > worst_rel:
                worst_rel = rel
            if not within_tolerance(err, rel):
                wrong += 1

    return report(
        "layernorm", residual, nonfinite, distinct, wrong, worst_abs, worst_rel
    )


def main() raises:
    var ctx = DeviceContext(api=String("npu"))
    print("context: ok")

    var passed = 0
    var failed = 0

    # Each selector is attempted even after one fails, and a raise is caught per
    # selector: they share the whole dispatch path, so which ones fail is what
    # separates a broken path from a broken kernel.
    try:
        if run_rms_norm(ctx):
            passed += 1
        else:
            failed += 1
    except error:
        print("rms_norm: raised:", error)
        failed += 1
    try:
        if run_layernorm(ctx):
            passed += 1
        else:
            failed += 1
    except error:
        print("layernorm: raised:", error)
        failed += 1
    try:
        if run_softmax(ctx):
            passed += 1
        else:
            failed += 1
    except error:
        print("softmax: raised:", error)
        failed += 1
    try:
        if run_matmul(ctx):
            passed += 1
        else:
            failed += 1
    except error:
        print("matmul: raised:", error)
        failed += 1

    print("selectors passed:", passed, "failed:", failed)
    print("PASS" if failed == 0 else "FAIL")
    if failed != 0:
        raise Error("an on-device selector dispatched from Mojo did not match")
