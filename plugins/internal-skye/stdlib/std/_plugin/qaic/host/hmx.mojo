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
"""QAIC HMX (Hexagon Matrix eXtension) library bindings for Mojo.

This module provides Mojo bindings for the Qualcomm HMX matmul library
(libQAicHexagonHMXV68.a on aic100, libQAicHexagonHMXV81.a on aic200; the
compiler links the arch-matching one). These functions run on the Hexagon DSP
within QAIC accelerators and use the HMX hardware unit for accelerated FP16
matrix multiplication.

The HMX matmul flow requires:
1. Converting input matrices from row-major to "crouton" tiled format
2. Performing matmul on HMX hardware (operates on crouton data)
3. Converting output from crouton back to row-major

Thread model:
- HVX threads (0-3): handle crouton format conversions
- HMX thread (4): handles the actual matmul computation
- DMA thread (4): handles data movement between DDR and VTCM

Reference headers:
- /opt/qti-aic/dev/inc/jit/QAicHexagonHMX.h
- /opt/qti-aic/dev/inc/jit/QAicHexagonHMXDefs.h

IMPORTANT: These functions are only callable from device code compiled for
the QAIC target (hexagon-unknown-linux-musl). They will not work in host code.
"""

from std.ffi import external_call
from std.memory import UnsafePointer
from .qshim import qshim_sync_nsps


# ===-----------------------------------------------------------------------===#
# Constants
# ===-----------------------------------------------------------------------===#

comptime HMX_ALIGNMENT: Int = 2048
"""Alignment for HMX crouton buffers (32x32 FP16 tile = 2048 bytes)."""

comptime HMX_INPUT_ALIGNMENT: Int = 128
"""Alignment for LHS input matrix in VTCM."""

comptime HMX_WEIGHT_ALIGNMENT: Int = 128
"""Alignment for RHS weight matrix in VTCM."""

comptime HMX_CROUTON_TILE_Y: Int = 32
"""Crouton tile size along Y dimension."""

comptime HMX_CROUTON_TILE_D: Int = 32
"""Crouton tile size along D dimension."""

comptime HMX_CROUTON_TILE_X: Int = 1
"""Crouton tile size along X dimension."""


# ===-----------------------------------------------------------------------===#
# QAicHMXDims - Matches C++ QAicHMXDims class memory layout
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct QAicHMXDims(TrivialRegisterPassable):
    """Dimension descriptor for HMX tensor layouts.

    Matches the memory layout of the C++ `QAicHMXDims<Layout>` class:
    5 consecutive int32 fields (20 bytes total).

    Uses 5 fields (not 3) to prevent SROA from decomposing the struct
    when its address is taken.
    """

    var n_dim: Int32
    var x_dim: Int32
    var y_dim: Int32
    var d_dim: Int32
    var elem_size: Int32


def hmx_flat_dims_yd(m: Int, k: Int) -> QAicHMXDims:
    """Create flat (row-major) HMX dims for a [M, K] matrix.

    For HMXShapeKind::YD: N=1, X=1, Y=M, D=K.
    """
    return QAicHMXDims(
        n_dim=Int32(1),
        x_dim=Int32(1),
        y_dim=Int32(m),
        d_dim=Int32(k),
        elem_size=Int32(2),
    )


def hmx_crouton_dims_from_flat(flat: QAicHMXDims) -> QAicHMXDims:
    """Convert flat dims to crouton (tiled) dims.

    Aligns Y and D up to multiples of 32, then divides by tile size.
    """
    var y_aligned = _align_to(Int(flat.y_dim), HMX_CROUTON_TILE_Y)
    var d_aligned = _align_to(Int(flat.d_dim), HMX_CROUTON_TILE_D)
    return QAicHMXDims(
        n_dim=flat.n_dim,
        x_dim=flat.x_dim,
        y_dim=Int32(y_aligned // HMX_CROUTON_TILE_Y),
        d_dim=Int32(d_aligned // HMX_CROUTON_TILE_D),
        elem_size=flat.elem_size,
    )


def hmx_crouton_size_in_bytes(crouton: QAicHMXDims) -> Int:
    """Compute total size in bytes for a crouton-layout tensor."""
    var total_elements = (
        Int(crouton.n_dim)
        * Int(crouton.x_dim)
        * Int(crouton.y_dim)
        * Int(crouton.d_dim)
        * HMX_CROUTON_TILE_X
        * HMX_CROUTON_TILE_Y
        * HMX_CROUTON_TILE_D
    )
    return total_elements * Int(crouton.elem_size)


def _align_to(value: Int, alignment: Int) -> Int:
    """Align value up to the nearest multiple of alignment (power of 2)."""
    return (value + alignment - 1) & ~(alignment - 1)


# ===-----------------------------------------------------------------------===#
# HMX Library Function Bindings
# ===-----------------------------------------------------------------------===#


@no_inline
def hmx_matmul_bias_init_16b(is_bf16: Bool = False) -> UInt32:
    """Initialize HMX bias registers for 16-bit crouton matmul.

    Must be called once at kernel start by the HMX thread only. `is_bf16`
    selects the operand interpretation.
    """
    return external_call["qaic_hmx_matmul_bias_init_16b", UInt32](
        Int32(1) if is_bf16 else Int32(0),
    )


@no_inline
def hmx_rm_matmul_lhs_to_crouton_16b(
    dst: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    src: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    crouton_dims: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    flat_dims: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    thread_id: Int32,
) -> UInt32:
    """Convert row-major LHS tensor to crouton format (HVX threads only)."""
    return external_call[
        "qaic_hmx_rm_matmul_lhs_to_crouton_16b",
        UInt32,
    ](dst, src, crouton_dims, flat_dims, thread_id)


@no_inline
def hmx_rm_matmul_lhs_to_crouton_8b(
    dst: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    src: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    crouton_dims: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    flat_dims: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    thread_id: Int32,
) -> UInt32:
    """Convert row-major boolean LHS tensor to 16-bit crouton attention mask (HVX threads only).

    Maps boolean elements: false → -inf, true → 0. Used to prepare boolean
    attention masks for HMX-based scaled-attention kernels.
    """
    return external_call[
        "qaic_hmx_rm_matmul_lhs_to_crouton_8b",
        UInt32,
    ](dst, src, crouton_dims, flat_dims, thread_id)


@no_inline
def hmx_cm_matmul_lhs_to_crouton_16b(
    dst: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    src: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    vtcm_scratch: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    crouton_dims: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    flat_dims: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    thread_id: Int32,
) -> UInt32:
    """Convert column-major 16-bit LHS tensor to crouton format (HVX threads only).

    vtcm_scratch must be VTCM-resident with size:
        (numThreads * HVX_VectorSize) + HVX_VectorSize bytes, aligned to HVX_VectorSize.
    """
    return external_call[
        "qaic_hmx_cm_matmul_lhs_to_crouton_16b",
        UInt32,
    ](dst, src, vtcm_scratch, crouton_dims, flat_dims, thread_id)


@no_inline
def hmx_rm_matmul_rhs_to_crouton_16b(
    dst: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    src: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    dims3: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    thread_id: Int32,
) -> UInt32:
    """Convert row-major RHS tensor to crouton format (HVX threads only).

    dims3 points to int[3] = {batch, k, n}.
    """
    return external_call[
        "qaic_hmx_rm_matmul_rhs_to_crouton_16b",
        UInt32,
    ](dst, src, dims3, thread_id)


@no_inline
def hmx_cm_matmul_rhs_to_crouton_16b(
    dst: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    src: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    dims3: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    thread_id: Int32,
) -> UInt32:
    """Convert column-major RHS tensor to crouton format (HVX threads only).

    dims3 points to int[3] = {batch, k, n}.
    """
    return external_call[
        "qaic_hmx_cm_matmul_rhs_to_crouton_16b",
        UInt32,
    ](dst, src, dims3, thread_id)


@no_inline
def hmx_get_neg_identity_crouton_hf(
    dst: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    thread_id: Int32,
) -> UInt32:
    """Populate a single 32x32 FP16 crouton tile with the negated identity.

    Required as the `neg_ident_w` argument to `hmx_matmul_prs_with_crouton_hf`.
    Wraps `qaic_hmx_get_neg_identity_crouton_hf` from libQAicHexagonHMXV68.a,
    which fills `dst` from an internal lookup table.

    The signature here is inferred from the symbol's mangled name (it uses an
    internal `neg_iden_lut`) and the convention used by sibling HVX-thread
    helpers such as `hmx_rm_matmul_rhs_to_crouton_16b`: the QAIC reference
    header `QAicHexagonHMX.h` is not installed on this system, so this
    binding is provisional and may need to be revised once the header is
    available.

    Args:
        dst: VTCM-resident buffer of `HMX_ALIGNMENT` (2048) bytes,
            `HMX_ALIGNMENT`-aligned, to receive the negated-identity crouton.
        thread_id: The calling HVX thread's ID. All four HVX threads (0-3)
            should call this; the library presumably distributes work by
            `thread_id`.
    """
    return external_call[
        "qaic_hmx_get_neg_identity_crouton_hf",
        UInt32,
    ](dst, thread_id)


@no_inline
def hmx_matmul_with_crouton_hf(
    dst: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    lhs: UnsafePointer[
        UnsafePointer[UInt8, MutUnsafeAnyOrigin], MutUnsafeAnyOrigin
    ],
    rhs: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    out_flat_dims: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    out_crouton_dims: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    lhs_crouton_dims: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    wt_dims: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    clear_acc: Int32,
    write_dst: Int32,
    is_bf16: Bool = False,
) -> UInt32:
    """Execute HMX matrix multiplication on crouton-formatted tensors.

    Must be called by the HMX thread ONLY. `is_bf16` selects the operand
    interpretation.
    """
    return external_call[
        "qaic_hmx_matmul_with_crouton_16b",
        UInt32,
    ](
        dst,
        lhs,
        rhs,
        out_flat_dims,
        out_crouton_dims,
        lhs_crouton_dims,
        wt_dims,
        clear_acc,
        write_dst,
        Int32(1) if is_bf16 else Int32(0),
    )


@no_inline
def hmx_matmul_prs_with_crouton_hf(
    dst: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    lhs: UnsafePointer[
        UnsafePointer[UInt8, MutUnsafeAnyOrigin], MutUnsafeAnyOrigin
    ],
    rhs: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    low_order_bits: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    neg_ident_w: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    out_flat_dims: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    out_crouton_dims: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    lhs_crouton_dims: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    wt_dims: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    clear_acc: Int32,
    write_dst: Int32,
) -> UInt32:
    """Execute HMX partial-result-sum (PRS) matrix multiplication on crouton tensors.

    Accumulates partial results across multiple LHS tiles. `low_order_bits`
    receives the low-order accumulator bits; `neg_ident_w` is the negated
    identity weight used by the PRS reduction.

    Must be called by the HMX thread ONLY.
    """
    return external_call[
        "qaic_hmx_matmul_prs_with_crouton_16b",
        UInt32,
    ](
        dst,
        lhs,
        rhs,
        low_order_bits,
        neg_ident_w,
        out_flat_dims,
        out_crouton_dims,
        lhs_crouton_dims,
        wt_dims,
        clear_acc,
        write_dst,
        Int32(0),  # isBF16; pass 0 for the fp16 path.
    )


@no_inline
def hmx_matmul_crouton_to_rm_16b(
    dst: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    src: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    flat_dims: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    crouton_dims: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    thread_id: Int32,
) -> UInt32:
    """Convert crouton-format output tensor back to row-major (HVX threads only).
    """
    return external_call[
        "qaic_hmx_matmul_crouton_to_rm_16b",
        UInt32,
    ](dst, src, flat_dims, crouton_dims, thread_id)


@no_inline
def hmx_softmax_with_crouton_hf(
    src: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    dst: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    scratch: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    flat_dims: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
    thread_id: Int32,
) -> UInt32:
    """Compute softmax on a crouton-format tensor in-place (HVX threads only).

    Must be called by all 4 HVX threads (0-3) simultaneously. The library
    distributes Y-tiles across the 4 threads and uses a hardcoded count=4
    barrier, so the input shape must have M >= 128 (≥4 Y-tiles of 32 rows
    each) to ensure every thread receives at least one tile.

    scratch must be VTCM-resident with size:
        2 * DEV_ATTR_QSHIM_NUM_HVX_UNITS * HVX_VECTOR_SIZE
        = 2 * 4 * 128 = 1024 bytes, aligned to HVX_VECTOR_SIZE.
    """
    return external_call[
        "qaic_softmax_with_crouton_16b",
        UInt32,
    ](src, dst, scratch, flat_dims, thread_id, False)


# ===-----------------------------------------------------------------------===#
# HMX Thread Synchronization
# ===-----------------------------------------------------------------------===#


@no_inline
def hmx_sync_hvx_threads(thread_id: UInt32):
    """4-participant HVX-only barrier (threads 0-3).

    Wraps `qaicSyncHVXThread`, which uses a hardcoded participant count of 4.
    All four HVX threads must call this; thread 4 (HMX/DMA) does NOT
    participate and must not call this function.

    Library functions such as `qaic_softmax_with_crouton_16b` call this
    internally — callers do not need to add extra calls around them.
    """
    external_call["qaicSyncHVXThread", NoneType](thread_id)


@no_inline
def hmx_sync_hvx_and_hmx_threads(thread_id: UInt32):
    """Synchronize HVX and HMX threads (barrier)."""
    external_call["qaicSyncHVXAndHMXThreads", NoneType](thread_id)


@no_inline
def hmx_sync_hvx_and_dma_threads(thread_id: UInt32):
    """Synchronize HVX and DMA threads (barrier)."""
    external_call["qaicSyncHVXAndDMAThreads", NoneType](thread_id)


@always_inline
def hmx_sync_hvx_dma_then_all_nsps(
    thread_id: UInt32,
    is_hmx_thread: Bool,
    nsp_mask: UInt32,
) -> UInt32:
    """Synchronize all local HMX/HVX/DMA work across all participating NSPs.

    This helper matches the standard HMX thread model where the HMX thread is
    also the designated DMA thread for the NSP. It implements the safe
    local -> global -> local barrier sequence around `qshim_sync_nsps()`:

    1. `hmx_sync_hvx_and_dma_threads()` closes the local stage.
    2. The HMX/DMA thread performs the cross-NSP barrier.
    3. `hmx_sync_hvx_and_hmx_threads()` releases the next stage.

    Args:
        thread_id: The calling thread's ID.
        is_hmx_thread: True for the designated HMX/DMA thread that should call
            `qshim_sync_nsps()`.
        nsp_mask: Bitmask of NSPs that participate in the cross-NSP barrier.

    Returns:
        Status returned by `qshim_sync_nsps()` on the HMX/DMA thread,
        otherwise 0.
    """
    hmx_sync_hvx_and_dma_threads(thread_id)

    var status: UInt32 = 0
    if is_hmx_thread:
        status = qshim_sync_nsps(nsp_mask)

    hmx_sync_hvx_and_hmx_threads(thread_id)
    return status
