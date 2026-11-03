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
"""Single-device characterization of the full DMA + crouton + HMX pipeline.

This is the three-way intersection with the HMX matrix engine in the loop: the
shared `hmx_matmul_kernel` runs the whole chain on one NSP — DMA both operands
DDR->VTCM (thread 4), pack A and B row-major->crouton (HVX 0-3), run the matmul
on the crouton-formatted tiles (HMX, thread 4), unpack C crouton->row-major
(HVX 0-3), and DMA C VTCM->DDR (thread 4).

That kernel is otherwise only exercised by the `manual`, multi-device
tensor-parallel tests, so until now there was no single-device, non-manual
regression that all three pieces line up. This test fills that gap and, beyond a
pass/fail, characterizes FP16 accuracy across an aligned shape sweep (error
grows with the contraction dimension K, where the crouton-tiled accumulation
happens), against an FP32-accumulating CPU reference.

Shapes are kept small so A, B, C and their crouton copies all fit in VTCM at
once (the single-tile kernel's constraint).
"""

from common import print_separator, print_test_header
from qaic_test_utils.rng import Xorshift32
from qaic_test_utils import (
    alloc_dev_fp16,
    compile_and_load_hmx_matmul,
    run_hmx_matmul,
    cpu_matmul_f16,
    verify_matmul_f16,
    print_verify_result,
)

from std.memory import UnsafePointer, alloc
from std.testing import TestSuite, assert_true

from max.gpu.host import DeviceContext
from max._plugin.qaic.host.qaic import (
    qaic_get_jit_device,
    qaic_release_jit_buffer,
    qaic_release_jit_function,
    qaic_copy_jit_buffer,
    QAIC_COPY_TO_DEVICE,
    QAIC_COPY_FROM_DEVICE,
)


def run_pipeline(M: Int, K: Int, N: Int, ctx: DeviceContext) raises:
    print("  M x K x N =", M, "x", K, "x", N)

    var jit_device = qaic_get_jit_device(ctx)
    var jit_func = compile_and_load_hmx_matmul(ctx)

    var a_elems = M * K
    var b_elems = K * N
    var c_elems = M * N

    var a_buf = alloc_dev_fp16(ctx, jit_device, a_elems)
    var b_buf = alloc_dev_fp16(ctx, jit_device, b_elems)
    var c_buf = alloc_dev_fp16(ctx, jit_device, c_elems)

    var a_host = alloc[Float16](a_elems)
    var b_host = alloc[Float16](b_elems)
    var c_host = alloc[Float16](c_elems)
    var ref_host = alloc[Float16](c_elems)

    var rng = Xorshift32.from_time()
    rng.fill_fp16(a_host, a_elems)
    rng.fill_fp16(b_host, b_elems)

    qaic_copy_jit_buffer(
        ctx,
        a_buf,
        a_host.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(a_elems * 2),
        QAIC_COPY_TO_DEVICE,
    )
    qaic_copy_jit_buffer(
        ctx,
        b_buf,
        b_host.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(b_elems * 2),
        QAIC_COPY_TO_DEVICE,
    )

    run_hmx_matmul(ctx, jit_device, jit_func, c_buf, a_buf, b_buf, M, K, N)

    qaic_copy_jit_buffer(
        ctx,
        c_buf,
        c_host.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(c_elems * 2),
        QAIC_COPY_FROM_DEVICE,
    )

    cpu_matmul_f16(a_host, b_host, ref_host, M, K, N)

    # Relative tolerance scaled mildly with K to absorb FP16 accumulation noise.
    var rel_tol = Float32(0.05)
    var result = verify_matmul_f16(c_host, ref_host, c_elems, rel_tol=rel_tol)
    print_verify_result(result, c_elems)

    a_host.unsafe_free()
    b_host.unsafe_free()
    c_host.unsafe_free()
    ref_host.unsafe_free()
    qaic_release_jit_buffer(ctx, a_buf)
    qaic_release_jit_buffer(ctx, b_buf)
    qaic_release_jit_buffer(ctx, c_buf)
    qaic_release_jit_function(jit_func)

    assert_true(result.passed())


def test_hmx_pipeline_square() raises:
    print_test_header("DMA+crouton+HMX matmul: square shapes")
    var ctx = DeviceContext(device_id=0, api="qaic")
    run_pipeline(32, 32, 32, ctx)
    run_pipeline(64, 64, 64, ctx)
    run_pipeline(128, 128, 128, ctx)


def test_hmx_pipeline_rectangular() raises:
    print_test_header("DMA+crouton+HMX matmul: rectangular shapes")
    var ctx = DeviceContext(device_id=0, api="qaic")
    run_pipeline(32, 64, 128, ctx)
    run_pipeline(128, 32, 64, ctx)
    run_pipeline(64, 128, 32, ctx)
    run_pipeline(256, 128, 64, ctx)


def test_hmx_pipeline_k_scaling() raises:
    # Same M, N; increasing K to characterize accumulation error growth.
    print_test_header("DMA+crouton+HMX matmul: K-accumulation sweep")
    var ctx = DeviceContext(device_id=0, api="qaic")
    run_pipeline(64, 32, 64, ctx)
    run_pipeline(64, 128, 64, ctx)
    run_pipeline(64, 512, 64, ctx)
    print_separator()
    print("TEST PASSED!")
    print_separator()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
