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
"""Compile-and-run helpers for QAIC kernel execution in tests."""

from std.memory import UnsafePointer, alloc
from max.gpu.host import DeviceContext, Dim

# Output slots for parallel reductions (one per worker).
# AI100: 16 NSPs × 5 HVX threads = 80 workers max.
# Sized with headroom for future devices.
comptime MAX_NPU_WORKERS: Int = 256
"""Output slots for parallel reductions (one per worker)."""


def compile_and_run_unary[
    kernel_fn: def(
        UnsafePointer[Float32, MutUnsafeAnyOrigin],
        UnsafePointer[Float32, MutUnsafeAnyOrigin],
    ) thin -> None
](
    ctx: DeviceContext,
    input_host: UnsafePointer[Float32, MutUntrackedOrigin],
    n: Int,
    num_cores: Int = 0,
    thread_mask: Int = 0,
) raises -> UnsafePointer[Float32, MutUntrackedOrigin]:
    """Compiles and runs a unary kernel on QAIC, returning the output buffer.

    Handles device buffer creation, host-to-device copy, compilation,
    execution, and device-to-host copy-back.  The caller is responsible
    for freeing the returned output pointer.

    Parameters:
        kernel_fn: A kernel function with signature
            ``fn(input, output) -> None``.

    Args:
        ctx: The QAIC device context.
        input_host: Pointer to host input data (n elements).
        n: Number of elements.
        num_cores: Number of NPU cores (0 = all available).
        thread_mask: HVX thread bitmask (0 = all threads).

    Returns:
        A pointer to the host output buffer (n elements).

    Raises:
        If device buffer creation, copy, compilation, or execution fails.
    """
    var input_dev = ctx.enqueue_create_buffer[DType.float32](n)
    var output_dev = ctx.enqueue_create_buffer[DType.float32](n)

    ctx.enqueue_copy(input_dev, input_host)

    var func = ctx.compile_function[kernel_fn]()
    ctx.enqueue_function(
        func,
        input_dev,
        output_dev,
        grid_dim=Dim(num_cores),
        block_dim=Dim(thread_mask),
    )

    var output_host = alloc[Float32](n)
    ctx.enqueue_copy(output_host, output_dev)
    ctx.synchronize()

    return output_host


def compile_and_run_binary[
    kernel_fn: def(
        UnsafePointer[Float32, MutUnsafeAnyOrigin],
        UnsafePointer[Float32, MutUnsafeAnyOrigin],
        UnsafePointer[Float32, MutUnsafeAnyOrigin],
    ) thin -> None
](
    ctx: DeviceContext,
    a_host: UnsafePointer[Float32, MutUntrackedOrigin],
    b_host: UnsafePointer[Float32, MutUntrackedOrigin],
    n: Int,
    num_cores: Int = 0,
    thread_mask: Int = 0,
) raises -> UnsafePointer[Float32, MutUntrackedOrigin]:
    """Compiles and runs a binary kernel on QAIC, returning the output buffer.

    Handles device buffer creation for three buffers (a, b, output),
    host-to-device copies, compilation, execution, and device-to-host
    copy-back.  The caller is responsible for freeing the returned output
    pointer.

    Parameters:
        kernel_fn: A kernel function with signature
            ``fn(a, b, output) -> None``.

    Args:
        ctx: The QAIC device context.
        a_host: Pointer to host input A data (n elements).
        b_host: Pointer to host input B data (n elements).
        n: Number of elements.
        num_cores: Number of NPU cores (0 = all available).
        thread_mask: HVX thread bitmask (0 = all threads).

    Returns:
        A pointer to the host output buffer (n elements).

    Raises:
        If device buffer creation, copy, compilation, or execution fails.
    """
    var a_dev = ctx.enqueue_create_buffer[DType.float32](n)
    var b_dev = ctx.enqueue_create_buffer[DType.float32](n)
    var output_dev = ctx.enqueue_create_buffer[DType.float32](n)

    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    var func = ctx.compile_function[kernel_fn]()
    ctx.enqueue_function(
        func,
        a_dev,
        b_dev,
        output_dev,
        grid_dim=Dim(num_cores),
        block_dim=Dim(thread_mask),
    )

    var output_host = alloc[Float32](n)
    ctx.enqueue_copy(output_host, output_dev)
    ctx.synchronize()

    return output_host


def compile_and_run_reduction[
    kernel_fn: def(
        UnsafePointer[Float32, MutUnsafeAnyOrigin],
        UnsafePointer[Float32, MutUnsafeAnyOrigin],
    ) thin -> None
](
    ctx: DeviceContext,
    input_host: UnsafePointer[Float32, MutUntrackedOrigin],
    n: Int,
    num_cores: Int = 0,
    thread_mask: Int = 0,
) raises -> UnsafePointer[Float32, MutUntrackedOrigin]:
    """Compiles and runs a reduction kernel on QAIC, returning a 1-element output.

    Same as `compile_and_run_unary` except the output buffer is a single
    element, suitable for kernels that reduce an input array to a scalar.
    The caller is responsible for freeing the returned output pointer.

    Parameters:
        kernel_fn: A kernel function with signature
            ``fn(input, output) -> None`` where output has 1 element.

    Args:
        ctx: The QAIC device context.
        input_host: Pointer to host input data (n elements).
        n: Number of input elements.
        num_cores: Number of NPU cores (0 = all available).
        thread_mask: HVX thread bitmask (0 = all threads).

    Returns:
        A pointer to the host output buffer (1 element).

    Raises:
        If device buffer creation, copy, compilation, or execution fails.
    """
    var input_dev = ctx.enqueue_create_buffer[DType.float32](n)
    var output_dev = ctx.enqueue_create_buffer[DType.float32](1)

    ctx.enqueue_copy(input_dev, input_host)

    var func = ctx.compile_function[kernel_fn]()
    ctx.enqueue_function(
        func,
        input_dev,
        output_dev,
        grid_dim=Dim(num_cores),
        block_dim=Dim(thread_mask),
    )

    var output_host = alloc[Float32](1)
    ctx.enqueue_copy(output_host, output_dev)
    ctx.synchronize()

    return output_host


def compile_and_run_parallel_reduction[
    kernel_fn: def(
        UnsafePointer[Float32, MutUnsafeAnyOrigin],
        UnsafePointer[Float32, MutUnsafeAnyOrigin],
    ) thin -> None
](
    ctx: DeviceContext,
    input_host: UnsafePointer[Float32, MutUntrackedOrigin],
    n: Int,
    identity: Float32 = Float32(0.0),
    num_cores: Int = 0,
    thread_mask: Int = 0,
) raises -> UnsafePointer[Float32, MutUntrackedOrigin]:
    """Runs a parallel reduction kernel across all NPU workers.

    Each worker computes a partial result for its work range and writes
    it to ``output[npu_global_idx()]``.  Unused slots are pre-filled with
    ``identity`` so the caller can combine all MAX_NPU_WORKERS elements
    without knowing how many workers ran.

    Parameters:
        kernel_fn: A kernel function with signature
            ``fn(input, partials_output) -> None``.

    Args:
        ctx: The QAIC device context.
        input_host: Pointer to host input data (n elements).
        n: Number of input elements.
        identity: Fill value for unused output slots (e.g. 0 for sum,
            1 for product).
        num_cores: Number of NPU cores (0 = all available).
        thread_mask: HVX thread bitmask (0 = all threads).

    Returns:
        A pointer to per-worker partial results (MAX_NPU_WORKERS elements).

    Raises:
        If device buffer creation, copy, compilation, or execution fails.
    """
    var input_dev = ctx.enqueue_create_buffer[DType.float32](n)
    var output_dev = ctx.enqueue_create_buffer[DType.float32](MAX_NPU_WORKERS)

    var output_init = alloc[Float32](MAX_NPU_WORKERS)
    for i in range(MAX_NPU_WORKERS):
        output_init[i] = identity

    ctx.enqueue_copy(input_dev, input_host)
    ctx.enqueue_copy(output_dev, output_init)

    var func = ctx.compile_function[kernel_fn]()
    ctx.enqueue_function(
        func,
        input_dev,
        output_dev,
        grid_dim=Dim(num_cores),
        block_dim=Dim(thread_mask),
    )

    var output_host = alloc[Float32](MAX_NPU_WORKERS)
    ctx.enqueue_copy(output_host, output_dev)
    ctx.synchronize()

    output_init.unsafe_free()

    return output_host
