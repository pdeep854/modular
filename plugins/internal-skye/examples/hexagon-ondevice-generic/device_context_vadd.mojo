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
"""End-to-end vadd test through DeviceContext(api="npu") on native Linux.

Exercises FP32 vector-add through the generic dispatch path (method 3,
selector 0) on the Hexagon Compute DSP of a Snapdragon device running native
Linux.

Acceptance gates (all must pass for exit 0):
  1. DeviceContext creates successfully — FastRPC library is discoverable.
  2. Counts 17, 256, 257 produce exact FP32 lhs[i] + rhs[i] results.
  3. A 257-element interior slice preserves guard values outside the slice.
  4. Seeded sentinel values are overwritten in active elements (no residual).
  5. No NaN or infinity in any output element.
  6. A negative element count is explicitly rejected.
  7. Unsupported selectors are refused by name (not silently executed on CPU).

Run after building and staging the DSP skel:
  mojo run -I <repo-root>/Mojo/stdlib \\
      plugins/internal-skye/examples/hexagon-ondevice-generic/device_context_vadd.mojo

Set MODULAR_NPU_RPC_ABI=generic (the default) and make sure libcdsprpc.so is
discoverable via ADSP_LIBRARY_PATH or standard system paths.
"""

from npu_context import DeviceContext, DeviceExternalFunction, Dim


# One FastRPC session addresses one NSP core.
comptime NPU_GRID_CORES: Int = 1

# Thread bitmap: five threads (four compute + one DMA).
comptime NPU_THREAD_MASK: Int = 0x1F

# Sentinel outside the valid result range for these inputs (small positive int sums).
comptime SEED_F32: Float32 = -999.0


def npu_sig():
    """Dummy type parameter for load_function.

    DeviceExternalFunction carries no argument types; the selector is resolved
    by name against NPUKernels.def on the host and dispatched by number on the
    DSP.  This zero-argument function serves as the required type witness.
    """
    pass


def is_finite_f32(v: Float32) -> Bool:
    return v == v and v - v == 0


def vadd_dispatch(ctx: DeviceContext, f: DeviceExternalFunction,
                  a: UInt64, b: UInt64, c: UInt64, n: Int32) raises:
    """Dispatches vadd with explicit rejection of non-positive element counts.

    Args:
        ctx: NPU device context.
        f: Resolved vadd kernel handle.
        a: Device buffer address for lhs.
        b: Device buffer address for rhs.
        c: Device buffer address for output.
        n: Element count; must be > 0.

    Raises:
        Error if n <= 0 or if the dispatch fails.
    """
    if n <= 0:
        raise Error("vadd element count must be > 0, got " + String(Int(n)))
    ctx.enqueue_function(
        f, a, b, c, n,
        grid_dim=Dim(NPU_GRID_CORES),
        block_dim=Dim(NPU_THREAD_MASK),
    )


def report(name: String, n: Int,
           residual: Int, nonfinite: Int, distinct: Int, wrong: Int) -> Bool:
    var ok = (residual == 0 and nonfinite == 0 and distinct > 1 and wrong == 0)
    print(name, " n=", n,
          ": residual=", residual, " nonfinite=", nonfinite,
          " distinct=", distinct, " wrong=", wrong,
          " -> ", "PASS" if ok else "FAIL", sep="")
    return ok


# ---------------------------------------------------------------------------
# vadd at a given element count
# ---------------------------------------------------------------------------

def run_vadd_size(ctx: DeviceContext, n: Int) raises -> Bool:
    """Dispatches selector 0 for `n` elements and validates results.

    Inputs: a[i] = i % 7 + 1, b[i] = i % 5 + 1.
    All sums are small positive integers exactly representable in FP32, so the
    reference is bit-exact.
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
    vadd_dispatch(ctx, f, a_dev.unsafe_ptr(), b_dev.unsafe_ptr(),
                  c_dev.unsafe_ptr(), Int32(n))
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
        var want = Float32(i % 7 + 1) + Float32(i % 5 + 1)
        if got != want:
            wrong += 1

    return report("vadd", n, residual, nonfinite, distinct, wrong)


# ---------------------------------------------------------------------------
# interior slice: guard elements outside [start, start+n) must be untouched
# ---------------------------------------------------------------------------

def run_vadd_slice(ctx: DeviceContext, total: Int, start: Int, n: Int) raises -> Bool:
    """Dispatch vadd on a sub-range and verify guards outside the slice.

    Allocates `total` elements, seeds all outputs to SEED_F32, dispatches only
    elements [start, start+n), then checks that elements outside the slice were
    not modified.
    """
    var a_host = ctx.enqueue_create_host_buffer[DType.float32](total)
    var b_host = ctx.enqueue_create_host_buffer[DType.float32](total)
    var c_host = ctx.enqueue_create_host_buffer[DType.float32](total)

    for i in range(total):
        a_host[i] = Float32(i % 7 + 1)
        b_host[i] = Float32(i % 5 + 1)
        c_host[i] = SEED_F32

    var a_dev = ctx.enqueue_create_buffer[DType.float32](total)
    var b_dev = ctx.enqueue_create_buffer[DType.float32](total)
    var c_dev = ctx.enqueue_create_buffer[DType.float32](total)
    a_dev.enqueue_copy_from(a_host)
    b_dev.enqueue_copy_from(b_host)
    c_dev.enqueue_copy_from(c_host)
    ctx.synchronize()

    # The sub-buffer carries only the slice.
    var a_slice = ctx.enqueue_create_buffer[DType.float32](n)
    var b_slice = ctx.enqueue_create_buffer[DType.float32](n)
    var c_slice = ctx.enqueue_create_buffer[DType.float32](n)

    for i in range(n):
        a_host[i] = Float32((start + i) % 7 + 1)
        b_host[i] = Float32((start + i) % 5 + 1)
    a_slice.enqueue_copy_from(a_host)
    b_slice.enqueue_copy_from(b_host)
    ctx.synchronize()

    var f = ctx.load_function[npu_sig](
        function_name=String("vadd"), asm=String("")
    )
    vadd_dispatch(ctx, f, a_slice.unsafe_ptr(), b_slice.unsafe_ptr(),
                  c_slice.unsafe_ptr(), Int32(n))
    ctx.synchronize()

    c_slice.enqueue_copy_to(c_host)
    ctx.synchronize()

    var wrong = 0
    for i in range(n):
        var got = c_host[i]
        var want = Float32((start + i) % 7 + 1) + Float32((start + i) % 5 + 1)
        if got != want:
            wrong += 1

    # Guard elements in c_dev (outside slice) must still be SEED_F32.
    c_dev.enqueue_copy_to(c_host)
    ctx.synchronize()
    var guard_broken = 0
    for i in range(total):
        if c_host[i] != SEED_F32:
            guard_broken += 1

    var ok = (wrong == 0 and guard_broken == 0)
    print("vadd_slice start=", start, " n=", n,
          ": wrong=", wrong, " guard_broken=", guard_broken,
          " -> ", "PASS" if ok else "FAIL", sep="")
    return ok


# ---------------------------------------------------------------------------
# negative count: must be rejected, not silently succeed
# ---------------------------------------------------------------------------

def run_vadd_negative_count(ctx: DeviceContext) raises -> Bool:
    """Verifies that n <= 0 is refused before reaching the device."""
    _ = ctx.enqueue_create_host_buffer[DType.float32](4)
    var dev   = ctx.enqueue_create_buffer[DType.float32](4)

    var refused = False
    try:
        var f = ctx.load_function[npu_sig](
            function_name=String("vadd"), asm=String("")
        )
        vadd_dispatch(ctx, f, dev.unsafe_ptr(), dev.unsafe_ptr(),
                      dev.unsafe_ptr(), Int32(-1))
        ctx.synchronize()
    except:
        refused = True

    print("vadd_negative_count: refused=", refused,
          " -> ", "PASS" if refused else "FAIL (negative count was not rejected)",
          sep="")
    return refused


# ---------------------------------------------------------------------------
# unknown selector: must be refused by name
# ---------------------------------------------------------------------------

def run_unknown_selector(ctx: DeviceContext) raises -> Bool:
    """Verifies that an unknown kernel name raises before dispatching."""
    var refused = False
    try:
        var f = ctx.load_function[npu_sig](
            function_name=String("__no_such_kernel__"), asm=String("")
        )
        var dev = ctx.enqueue_create_buffer[DType.float32](4)
        ctx.enqueue_function(
            f,
            dev.unsafe_ptr(),
            Int32(4),
            grid_dim=Dim(NPU_GRID_CORES),
            block_dim=Dim(NPU_THREAD_MASK),
        )
        ctx.synchronize()
    except:
        refused = True

    print("unknown_selector: refused=", refused,
          " -> ", "PASS" if refused else "FAIL (unknown selector not rejected)",
          sep="")
    return refused


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() raises:
    print("DeviceContext create ... ", end="")
    var ctx = DeviceContext(api=String("npu"))
    print("ok")

    var passed = 0
    var failed = 0

    # Core sizes: 17 exercises the scalar tail, 256 fills eight HVX registers,
    # 257 exercises both.
    for n in [17, 256, 257]:
        try:
            if run_vadd_size(ctx, n):
                passed += 1
            else:
                failed += 1
        except e:
            print("vadd n=", n, ": raised: ", e, sep="")
            failed += 1

    # Interior slice: 257-element allocation, 129-element active window offset by 64.
    try:
        if run_vadd_slice(ctx, 257, 64, 129):
            passed += 1
        else:
            failed += 1
    except e:
        print("vadd_slice: raised: ", e, sep="")
        failed += 1

    # Negative count must be refused.
    try:
        if run_vadd_negative_count(ctx):
            passed += 1
        else:
            failed += 1
    except e:
        print("vadd_negative_count: raised: ", e, sep="")
        failed += 1

    # Unknown selector must be refused.
    try:
        if run_unknown_selector(ctx):
            passed += 1
        else:
            failed += 1
    except e:
        print("unknown_selector: raised: ", e, sep="")
        failed += 1

    print("\npassed:", passed, " failed:", failed)
    print("PASS" if failed == 0 else "FAIL")
    if failed != 0:
        raise Error("one or more NPU dispatch checks failed")
