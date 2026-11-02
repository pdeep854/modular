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
"""End-to-end integration test: FP32 vector-add on Hexagon CDSP.

Exercises three consecutive dispatches through `DeviceContext(api="npu")`:

  1. `env_probe`      (selector 2) — confirms the DSP session is live and
                                     reports the HVX/thread configuration.
  2. `vadd`           (selector 0) — elementwise FP32 a + b = c, at four
                                     sizes that exercise both the HVX-vectorised
                                     body and the scalar tail.
  3. `vscale_offset`  (selector 1) — elementwise FP32 fused multiply-add
                                     out[i] = scale * in[i] + offset, exact
                                     integer values so the reference is bit-exact.

The test format mirrors `npu_selector_dispatch.mojo`: every launched buffer is
seeded with a sentinel outside the valid result range so an element the kernel
never touched is distinguishable from one it computed, each case runs all three
health guards (residual, nonfinite, distinct), and exact FP32 arithmetic on small
integers is used so the "wrong" counter catches any computation error rather than
a rounding difference.

Wire contract (from `runtime/lib/qurt/NPUKernels.def`):
  vadd           selector 0   buffers: lhs, rhs, result        scalar: n (Int32)
  vscale_offset  selector 1   buffers: input, output           scalars: scale, offset, n
  env_probe      selector 2   buffers: report (uint32)         scalar: word_capacity
"""

from std.gpu.host import DeviceContext, Dim


# One FastRPC session addresses one NSP core.
comptime NPU_GRID_CORES: Int = 1

# Thread bitmap: five threads (four compute + one DMA), low bits set with no gap.
comptime NPU_THREAD_MASK: Int = 0x1F

# Words the env_probe kernel writes: HVX width (bytes), thread count, DSP version,
# HMX present flag, and one word per thread for its hardware thread id. Keep
# headroom so a newer skel that writes more words is truncated rather than faulting.
comptime ENV_PROBE_WORDS: Int = 32

# Seed value for output buffers: outside the range of any correct FP32 result
# produced by these kernels (all inputs are small positive integers).
comptime SEED_F32: Float32 = -999.0
comptime SEED_U32: UInt32 = 0xDEADBEEF


def npu_sig():
    """Dummy signature for load_function.

    `DeviceExternalFunction` carries no argument types; the selector is resolved
    by name against NPUKernels.def and the asm argument is ignored entirely.
    """
    pass


def absolute_f32(v: Float32) -> Float32:
    return v if v >= 0 else -v


def is_finite_f32(v: Float32) -> Bool:
    return v == v and v - v == 0


def report_case(
    name: String,
    n: Int,
    residual: Int,
    nonfinite: Int,
    distinct: Int,
    wrong: Int,
) -> Bool:
    var ok = residual == 0 and nonfinite == 0 and distinct > 1 and wrong == 0
    print(
        name,
        " n=",
        n,
        ": residual=",
        residual,
        " nonfinite=",
        nonfinite,
        " distinct=",
        distinct,
        " wrong=",
        wrong,
        " -> ",
        "PASS" if ok else "FAIL",
        sep="",
    )
    return ok


# ---------------------------------------------------------------------------
# env_probe: confirm the DSP session is live
# ---------------------------------------------------------------------------


def run_env_probe(ctx: DeviceContext) raises -> Bool:
    """Dispatches selector 2 and prints the DSP environment report.

    Words 0-3 carry the HVX register width (bytes), the active thread count,
    the Hexagon architecture version, and whether the HMX unit is present. The
    remaining words hold per-thread hardware ids written by each participant.
    A zero in any slot means the kernel left it unwritten; a skel that predates
    the env_probe row refuses it by name and the session survives.
    """
    var report_host = ctx.enqueue_create_host_buffer[DType.uint32](
        ENV_PROBE_WORDS
    )
    for i in range(ENV_PROBE_WORDS):
        report_host[i] = SEED_U32

    var report_dev = ctx.enqueue_create_buffer[DType.uint32](ENV_PROBE_WORDS)
    report_dev.enqueue_copy_from(report_host)
    ctx.synchronize()

    var f = ctx.load_function[npu_sig](
        function_name=String("env_probe"), asm=String("")
    )
    ctx.enqueue_function(
        f,
        report_dev.unsafe_ptr(),
        Int32(ENV_PROBE_WORDS),
        grid_dim=Dim(NPU_GRID_CORES),
        block_dim=Dim(NPU_THREAD_MASK),
    )
    ctx.synchronize()

    report_dev.enqueue_copy_to(report_host)
    ctx.synchronize()

    # Check at least the first word was written.
    var wrote_anything = False
    for i in range(ENV_PROBE_WORDS):
        if report_host[i] != SEED_U32:
            wrote_anything = True

    print(
        "env_probe: hvx_bytes=",
        report_host[0],
        " threads=",
        report_host[1],
        " arch_ver=",
        report_host[2],
        " hmx_present=",
        report_host[3],
        sep="",
    )
    var ok = wrote_anything
    print("env_probe: ", "PASS" if ok else "FAIL (no words written)", sep="")
    return ok


# ---------------------------------------------------------------------------
# vadd: FP32 vector-add, four sizes
# ---------------------------------------------------------------------------


def run_vadd_size(ctx: DeviceContext, n: Int) raises -> Bool:
    """One vector-add dispatch at the given element count.

    Inputs are small positive integers (a[i] = i % 7 + 1, b[i] = i % 5 + 1)
    so every sum is an exact FP32 integer and the reference is bit-exact.
    """
    var a_host = ctx.enqueue_create_host_buffer[DType.float32](n)
    var b_host = ctx.enqueue_create_host_buffer[DType.float32](n)
    var c_host = ctx.enqueue_create_host_buffer[DType.float32](n)

    for i in range(n):
        a_host[i] = Float32(i % 7 + 1)
        b_host[i] = Float32(i % 5 + 1)
        c_host[i] = SEED_F32

    var a_dev = ctx.enqueue_create_buffer[DType.float32](n)
    var b_dev = ctx.enqueue_create_buffer[DType.float32](n)
    var c_dev = ctx.enqueue_create_buffer[DType.float32](n)
    a_dev.enqueue_copy_from(a_host)
    b_dev.enqueue_copy_from(b_host)
    c_dev.enqueue_copy_from(c_host)
    ctx.synchronize()

    var f = ctx.load_function[npu_sig](
        function_name=String("vadd"), asm=String("")
    )
    ctx.enqueue_function(
        f,
        a_dev.unsafe_ptr(),
        b_dev.unsafe_ptr(),
        c_dev.unsafe_ptr(),
        Int32(n),
        grid_dim=Dim(NPU_GRID_CORES),
        block_dim=Dim(NPU_THREAD_MASK),
    )
    ctx.synchronize()

    c_dev.enqueue_copy_to(c_host)
    ctx.synchronize()

    var residual = 0
    var nonfinite = 0
    var distinct = 0
    var wrong = 0

    for i in range(n):
        var got = c_host[i]
        if got == SEED_F32:
            residual += 1
        if not is_finite_f32(got):
            nonfinite += 1
        var seen = False
        for j in range(i):
            if c_host[j] == got:
                seen = True
        if not seen:
            distinct += 1
        # Reference is exact: a + b is a small integer representable in FP32.
        var want = Float32(i % 7 + 1) + Float32(i % 5 + 1)
        if got != want:
            wrong += 1

    return report_case("vadd", n, residual, nonfinite, distinct, wrong)


def run_vadd(ctx: DeviceContext) raises -> Bool:
    """Runs vector-add at four sizes covering HVX-vectorised and tail paths.

    The HVX body in `npu_kernel.mojo` processes LANES=32 elements per iteration
    (128-byte register / 4-byte float). The sizes below cover:
      32  — exactly one HVX vector, no tail
      64  — two vectors, no tail
      100 — three vectors + 4-element tail
      256 — eight vectors, no tail
    """
    var ok = True
    var sizes: List[Int] = [32, 64, 100, 256]
    for n in sizes:
        if not run_vadd_size(ctx, n[]):
            ok = False
    return ok


# ---------------------------------------------------------------------------
# vscale_offset: FP32 fused multiply-add
# ---------------------------------------------------------------------------


def run_vscale_offset(ctx: DeviceContext) raises -> Bool:
    """Dispatches selector 1: out[i] = scale * in[i] + offset.

    Uses scale=3.0, offset=10.0, input[i]=i%8+1 so every output is a small
    integer exact in FP32.
    """
    comptime N = 128
    comptime SCALE: Float32 = 3.0
    comptime OFFSET: Float32 = 10.0

    var in_host = ctx.enqueue_create_host_buffer[DType.float32](N)
    var out_host = ctx.enqueue_create_host_buffer[DType.float32](N)

    for i in range(N):
        in_host[i] = Float32(i % 8 + 1)
        out_host[i] = SEED_F32

    var in_dev = ctx.enqueue_create_buffer[DType.float32](N)
    var out_dev = ctx.enqueue_create_buffer[DType.float32](N)
    in_dev.enqueue_copy_from(in_host)
    out_dev.enqueue_copy_from(out_host)
    ctx.synchronize()

    var f = ctx.load_function[npu_sig](
        function_name=String("vscale_offset"), asm=String("")
    )
    ctx.enqueue_function(
        f,
        in_dev.unsafe_ptr(),
        out_dev.unsafe_ptr(),
        SCALE,
        OFFSET,
        Int32(N),
        grid_dim=Dim(NPU_GRID_CORES),
        block_dim=Dim(NPU_THREAD_MASK),
    )
    ctx.synchronize()

    out_dev.enqueue_copy_to(out_host)
    ctx.synchronize()

    var residual = 0
    var nonfinite = 0
    var distinct = 0
    var wrong = 0

    for i in range(N):
        var got = out_host[i]
        if got == SEED_F32:
            residual += 1
        if not is_finite_f32(got):
            nonfinite += 1
        var seen = False
        for j in range(i):
            if out_host[j] == got:
                seen = True
        if not seen:
            distinct += 1
        var want = SCALE * Float32(i % 8 + 1) + OFFSET
        if got != want:
            wrong += 1

    return report_case("vscale_offset", N, residual, nonfinite, distinct, wrong)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() raises:
    var ctx = DeviceContext(api=String("npu"))
    print("context: ok")

    var passed = 0
    var failed = 0

    # env_probe first: if the session is not live, the compute cases will also
    # fail and the probe's output names the reason.
    try:
        if run_env_probe(ctx):
            passed += 1
        else:
            failed += 1
    except error:
        print("env_probe: raised:", error)
        failed += 1

    # Each compute case is attempted even after a previous failure: they share
    # the same dispatch path, so which ones fail and which pass narrows the fault.
    try:
        if run_vadd(ctx):
            passed += 1
        else:
            failed += 1
    except error:
        print("vadd: raised:", error)
        failed += 1

    try:
        if run_vscale_offset(ctx):
            passed += 1
        else:
            failed += 1
    except error:
        print("vscale_offset: raised:", error)
        failed += 1

    print("passed:", passed, "failed:", failed)
    print("PASS" if failed == 0 else "FAIL")
    if failed != 0:
        raise Error("one or more NPU dispatches did not match the CPU reference")
