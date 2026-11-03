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
"""QAIC QShim device runtime bindings for Mojo.

This module provides Mojo bindings for the Qualcomm AI Cloud (QAIC) QShim
device runtime API. These functions are available to device code running on
the Hexagon DSP within QAIC accelerators.

The QShim API provides low-level access to:
- Thread synchronization primitives
- DMA (Direct Memory Access) operations for efficient data movement
- VTCM (Vector Tightly Coupled Memory) management
- Hardware random number generation
- Profiling and logging utilities
- Execution context information (core/thread IDs)

IMPORTANT: These functions are only callable from device code compiled for
the QAIC target (hexagon-unknown-linux-musl). They will not work in host code.

Reference: /opt/qti-aic/dev/inc/jit/jit_qshim_api.h
"""

from std.ffi import external_call

from std.collections import OptionalReg
from std.collections.string.string_span import _get_kgen_string
from std.memory import UnsafePointer


# ===-----------------------------------------------------------------------===#
# Constants and Types
# ===-----------------------------------------------------------------------===#


comptime INVALID_UDMA_HANDLE: UInt32 = 0xFFFFFFFF
"""Invalid UDMA handle value returned on submission failure."""


# Log mask values for qshim_log()
comptime LOG_MASK_NONE: UInt32 = 0x0
"""No logging."""

comptime LOG_MASK_FATAL: UInt32 = 0x1
"""Fatal error logging."""

comptime LOG_MASK_ERROR: UInt32 = 0x2
"""Error logging."""

comptime LOG_MASK_WARN: UInt32 = 0x4
"""Warning logging."""

comptime LOG_MASK_INFO: UInt32 = 0x8
"""Informational logging."""

comptime LOG_MASK_DEBUG: UInt32 = 0x10
"""Debug logging."""

comptime LOG_MASK_INVALID: UInt32 = 0x7F
"""Invalid log mask."""


# ECC memory types
comptime ECC_MEM_VTCM: UInt32 = 0
"""VTCM memory type for ECC error checking."""


@fieldwise_init
struct QShimUdmaDescCommonAttrs(TrivialRegisterPassable):
    """Common attributes for UDMA (Micro DMA) descriptors.

    This struct allows specifying ordering constraints for DMA transfers.

    Attributes:
        order_and_reserved: Bit field containing (SDK 1.22 layout):
            - Bit 0 (order): DMA ordering point.
                * 0: No ordering requirements with previous DMAs.
                * 1: Previous DMA must complete before this one starts.
            - Bit 1 (bypassOverride): Override default UDMA cache-bypass
                behavior. Must be 0 on AI 100 (use default per-memory-type
                bypass policy).
            - Bit 2 (destBypass): Destination cache-bypass; only effective
                when bypassOverride = 1.
            - Bit 3 (srcBypass): Source cache-bypass; only effective when
                bypassOverride = 1.
            - Bits 4-31: Reserved.

        Setting `order_and_reserved` to only 0 or 1 keeps bits 1-3 clear, so
        bypassOverride stays 0 as required.

    Example:
        ```mojo
        # Create descriptor with ordering requirement
        var attrs = QShimUdmaDescCommonAttrs(order_and_reserved=1)

        # Submit DMA that must wait for previous transfers
        var handle = qshim_linear_udma_submit(
            thread_id, src, size, dst,
            UnsafePointer(to=attrs).as_unsafe_any_origin(), True, status_ptr
        )
        ```

    Note:
        By default, DMAs are NOT guaranteed to execute in order. Set the order
        bit when ordering between DMA descriptors is required.
    """

    var order_and_reserved: UInt32


@fieldwise_init
struct QShim2DUdmaDescAttrs(TrivialRegisterPassable):
    """Attributes for 2D UDMA (Micro DMA) transfers.

    2D DMA allows transferring rectangular regions of memory with different
    source and destination strides, useful for matrix operations and image
    processing.

    Attributes:
        desc_type_reserved: Descriptor type (only Type0 supported) and reserved bits.
        height_width: Packed field: height (bits 0-15) and width (bits 16-31).
            - height: Number of rows in the region (max 65535).
            - width: Width of each row in bytes (max 65535).
        dest_stride_src_stride: Packed field for stride values.
            - dest_stride (bits 0-15): Bytes between destination rows.
            - src_stride (bits 16-31): Bytes between source rows.
        reserved: Reserved field.
        common_attrs: Pointer to common attributes (can be null).

    Example:
        ```mojo
        # Transfer a 256x128 region with custom strides
        var attrs = QShim2DUdmaDescAttrs(
            desc_type_reserved=0,
            height_width=(128 << 16) | 256,  # height=128, width=256 bytes
            dest_stride_src_stride=(512 << 16) | 1024,  # dst_stride=512, src_stride=1024
            reserved=0,
            common_attrs={},
        )
        ```
    """

    var desc_type_reserved: UInt32
    var height_width: UInt32
    var dest_stride_src_stride: UInt32
    var reserved: UInt32

    @__allow_legacy_any_origin_fields
    var common_attrs: OptionalReg[
        UnsafePointer[QShimUdmaDescCommonAttrs, MutAnyOrigin]
    ]


@fieldwise_init
struct QShimThreadSyncStruct(TrivialRegisterPassable):
    """Thread synchronization structure for qshim_sync_threads().

    This struct must be allocated in global memory (not on the stack) and
    shared across all threads that need to synchronize.

    Attributes:
        thread_mask: Bitmask of threads to synchronize. Bit N corresponds to
            thread ID N. For example, 0x1F syncs threads 0-4.
        data: Internal synchronization data (do not modify).

    Example:
        ```mojo
        # Allocate sync point in kernel global memory (not on the stack)
        var sync_ptr = qshim_create_thread_sync_point[thread_mask=0x1F]()

        # In kernel, each thread calls:
        var status = qshim_sync_threads(thread_id, sync_ptr)
        ```

    Warning:
        - The sync_point MUST be in global memory, NOT on the thread's stack.
          Use `qshim_create_thread_sync_point` to allocate it correctly.
        - Do not modify thread_mask after initialization until sync completes.
        - After sync completes, thread_mask returns to its original value and
          the struct can be reused.
    """

    var thread_mask: Int32
    var data: Int32


# ===-----------------------------------------------------------------------===#
# AicJitPtr ABI helper (SDK 1.23+)
# ===-----------------------------------------------------------------------===#


@always_inline
def _to_aic_jit_ptr(p: UnsafePointer[UInt8, MutAnyOrigin]) -> UInt64:
    """Widen a device pointer to the `AicJitPtr` (`uint64_t`) qshim ABI.

    SDK 1.23 changed every qshim pointer parameter from `void *` to
    `AicJitPtr`, which is a `uint64_t` (to support 64-bit AI 200 addressing).
    The NSP address space is 32-bit, so the address lives in the low 32 bits
    and the high 32 bits are zero: truncate to 32 bits, then zero-extend to
    64. Passing a bare 32-bit pointer instead of a 64-bit value would
    misalign the register pair the callee reads for the `uint64_t` argument.
    """
    return UInt64(UInt32(Int(p)))


# ===-----------------------------------------------------------------------===#
# Thread Synchronization
# ===-----------------------------------------------------------------------===#


def qshim_create_thread_sync_point[
    thread_mask: Int32, name: StaticString = "_sync_point"
]() -> UnsafePointer[QShimThreadSyncStruct, MutUntrackedOrigin]:
    """Creates a `QShimThreadSyncStruct` in kernel global memory.

    Allocates the sync struct using `pop.global_alloc` with an initializer
    so `thread_mask` is placed in the `.data` section at compile time,
    eliminating the per-thread runtime store.

    Parameters:
        thread_mask: Bitmask of threads to synchronize. Bit N corresponds
            to thread ID N. For example, `0x1F` syncs threads 0-4.
        name: Symbol name for the global allocation. Acts as the unique
            identifier — calls with the same name return the same address.

    Returns:
        Pointer to an initialized `QShimThreadSyncStruct` in global memory.

    Example:
        ```mojo
        var sync_ptr = qshim_create_thread_sync_point[thread_mask=0x1F]()
        _ = qshim_sync_threads(thread_id, sync_ptr)
        ```
    """
    return {
        _mlir_value = __mlir_op.`pop.global_alloc`[
            name=_get_kgen_string[name, "._aic_syncpoint"](),
            count=Int(1).__mlir_index__(),
            _type=UnsafePointer[
                QShimThreadSyncStruct, MutUntrackedOrigin
            ]._mlir_type,
            alignment=Int(4).__mlir_index__(),
            initializer=QShimThreadSyncStruct(thread_mask=thread_mask, data=0),
        ]()
    }


@always_inline
def qshim_sync_threads(
    thread_id: UInt32,
    sync_point: UnsafePointer[QShimThreadSyncStruct, MutAnyOrigin],
) -> UInt32:
    """Synchronize multiple HVX threads at a barrier.

    All threads specified in the sync_point's thread_mask must call this
    function. Threads block until all participating threads reach the barrier,
    then all are released simultaneously.

    Args:
        thread_id: The calling thread's ID (0-4 on hexagonv68).
        sync_point: Pointer to a shared QShimThreadSyncStruct in global memory.
            The thread_mask field specifies which threads participate.

    Returns:
        0 on success, non-zero on error (null pointer or zero thread_mask).

    Example:
        ```mojo
        def my_kernel(entry_config: ..., pointer_array: ...) -> UInt32:
            var tid = entry_config[].thread_id

            # Allocate sync point in kernel global memory
            var sync_ptr = qshim_create_thread_sync_point[thread_mask=0x1F]()

            # Phase 1: Each thread does independent work
            do_work_phase1()

            # Barrier: Wait for all threads
            _ = qshim_sync_threads(tid, sync_ptr)

            # Phase 2: All threads continue together
            do_work_phase2()
            return 0
        ```

    When to use:
        - Coordinating multi-phase algorithms where all threads must complete
          one phase before starting the next.
        - Ensuring data written by one thread is visible to others.
        - Implementing producer-consumer patterns within a kernel.

    Performance note:
        Thread synchronization has overhead. Minimize sync points when possible
        by restructuring algorithms to reduce inter-thread dependencies.
    """
    return external_call[
        "qshimSyncThreads",
        UInt32,
    ](thread_id, sync_point)


# ===-----------------------------------------------------------------------===#
# DMA Operations
# ===-----------------------------------------------------------------------===#


@always_inline
def qshim_linear_udma_submit(
    thread_id: UInt32,
    src: UnsafePointer[UInt8, MutAnyOrigin],
    size: UInt32,
    dst: UnsafePointer[UInt8, MutAnyOrigin],
    udma_desc_attrs: Optional[
        UnsafePointer[QShimUdmaDescCommonAttrs, MutAnyOrigin]
    ],
    require_handle: Bool,
    status: UnsafePointer[UInt32, MutAnyOrigin],
) -> UInt32:
    """Submit a linear (1D) DMA transfer request.

    Initiates an asynchronous memory copy from src to dst. The transfer runs
    in the background while the thread continues execution.

    Args:
        thread_id: The calling thread's ID.
        src: Source memory address.
        size: Number of bytes to transfer.
        dst: Destination memory address.
        udma_desc_attrs: Optional pointer to descriptor attributes for ordering
            control. Pass null for default behavior.
        require_handle: If True, returns a handle that can be waited on with
            qshim_udma_wait(). If False, returns INVALID_UDMA_HANDLE.
        status: Output pointer for submission status (0 = success).

    Returns:
        UDMA handle if require_handle is True and submission succeeds,
        otherwise INVALID_UDMA_HANDLE.

    Example:
        ```mojo
        var status: UInt32 = 0

        # Start async DMA transfer
        var handle = qshim_linear_udma_submit(
            thread_id,
            src_ptr,
            1024,  # Transfer 1KB
            dst_ptr,
            {},  # No special attrs (Optional None)
            True,  # We want to wait on this
            UnsafePointer(to=status).as_unsafe_any_origin(),
        )

        if status == 0:
            # Do other work while DMA runs...
            do_computation()

            # Wait for DMA to complete
            _ = qshim_udma_wait(handle)
        ```

    When to use:
        - Moving data between DDR and VTCM (fast on-chip memory).
        - Prefetching data for upcoming computations.
        - Double-buffering: process one buffer while loading the next.
        - Any large memory copy where you can overlap with computation.

    Performance tips:
        - Align src and dst to 128 bytes for best performance.
        - Use VTCM as destination for data that will be accessed repeatedly.
        - Start DMA early to hide latency behind computation.
    """
    # Pass the src/dst device addresses as 32-bit values to a hexagon-clang
    # trampoline that widens them to the 64-bit `AicJitPtr` the SDK qshim
    # expects. Passing a `UInt64` directly (via `_to_aic_jit_ptr`) mis-delivers
    # the interleaved 64-bit arg across the Mojo `external_call` boundary on the
    # 32-bit Hexagon C ABI, landing the address in the wrong half of the
    # register pair (SDK then rejects it as HSV39). The trampoline is compiled
    # by hexagon-clang and linked in by `M_KGEN_createSharedObject`.
    return external_call[
        "skye_udma_submit_linear_lo32",
        UInt32,
    ](
        thread_id,
        UInt32(Int(src)),
        size,
        UInt32(Int(dst)),
        udma_desc_attrs,
        require_handle,
        status,
    )


@always_inline
def qshim_2d_udma_submit(
    thread_id: UInt32,
    src: UnsafePointer[UInt8, MutAnyOrigin],
    dst: UnsafePointer[UInt8, MutAnyOrigin],
    udma_desc_attrs: UnsafePointer[QShim2DUdmaDescAttrs, MutAnyOrigin],
    require_handle: Bool,
    status: UnsafePointer[UInt32, MutAnyOrigin],
) -> UInt32:
    """Submit a 2D DMA transfer request.

    Transfers a rectangular region of memory with potentially different source
    and destination strides. Useful for matrix operations and image processing
    where data layouts differ between source and destination.

    Args:
        thread_id: The calling thread's ID.
        src: Source memory address (top-left corner of region).
        dst: Destination memory address (top-left corner).
        udma_desc_attrs: Pointer to 2D descriptor attributes specifying
            width, height, and stride values. Required (cannot be null).
        require_handle: If True, returns a waitable handle.
        status: Output pointer for submission status.

    Returns:
        UDMA handle if require_handle is True and submission succeeds,
        otherwise INVALID_UDMA_HANDLE.

    Example:
        ```mojo
        # Transfer a 64x32 tile from a larger matrix
        var attrs = QShim2DUdmaDescAttrs(
            desc_type_reserved=0,
            height_width=(32 << 16) | (64 * 4),  # 32 rows, 64 floats = 256 bytes
            dest_stride_src_stride=(256 << 16) | 4096,  # dst packed, src has 1024-float stride
            reserved=0,
            common_attrs={},
        )

        var status: UInt32 = 0
        var handle = qshim_2d_udma_submit(
            thread_id, src_ptr, dst_ptr,
            UnsafePointer(to=attrs).as_unsafe_any_origin(), True,
            UnsafePointer(to=status).as_unsafe_any_origin()
        )
        ```

    When to use:
        - Extracting tiles from larger matrices for tiled computation.
        - Converting between row-major and column-major layouts.
        - Copying image regions with padding/cropping.
        - Gathering scattered data into contiguous VTCM buffers.
    """
    # See `qshim_linear_udma_submit`: pass the addresses as 32-bit values to a
    # hexagon-clang trampoline that widens them to `AicJitPtr`, dodging the
    # Hexagon 64-bit-arg ABI mismatch in Mojo's `external_call` lowering.
    return external_call[
        "skye_udma_submit_2d_lo32",
        UInt32,
    ](
        thread_id,
        UInt32(Int(src)),
        UInt32(Int(dst)),
        udma_desc_attrs,
        require_handle,
        status,
    )


@always_inline
def qshim_udma_wait(handle: UInt32) -> UInt32:
    """Wait for a DMA transfer to complete.

    Blocks the calling thread until the DMA transfer associated with the
    given handle finishes.

    Args:
        handle: The UDMA handle returned from qshim_linear_udma_submit() or
            qshim_2d_udma_submit().

    Returns:
        0 on success, non-zero on error (including INVALID_UDMA_HANDLE).

    Example:
        ```mojo
        # Submit multiple DMAs
        var handle1 = qshim_linear_udma_submit(tid, src1, size1, dst1, ...)
        var handle2 = qshim_linear_udma_submit(tid, src2, size2, dst2, ...)

        # Do computation while DMAs run
        compute_something()

        # Wait for both to complete
        _ = qshim_udma_wait(handle1)
        _ = qshim_udma_wait(handle2)

        # Now safe to use dst1 and dst2
        ```

    Note:
        Only call this with handles obtained from DMA submit functions where
        require_handle=True. Waiting on INVALID_UDMA_HANDLE returns an error.
    """
    return external_call["qshimUDmaWait", UInt32](handle)


# ===-----------------------------------------------------------------------===#
# VTCM (Vector Tightly Coupled Memory) Management
# ===-----------------------------------------------------------------------===#


@always_inline
def qshim_get_base_vtcm_addr() -> UnsafePointer[UInt8, MutAnyOrigin]:
    """Get the base address of VTCM (Vector Tightly Coupled Memory).

    VTCM is fast on-chip memory accessible to HVX vector operations with
    single-cycle latency. Use it for frequently accessed data.

    Returns:
        Pointer to the start of VTCM memory.

    Example:
        ```mojo
        var vtcm_base = qshim_get_base_vtcm_addr()
        var vtcm_size = qshim_get_vtcm_size()

        # Allocate space in VTCM for a tile
        var tile_offset = 0
        var tile = vtcm_base.offset(tile_offset).bitcast[Float32]()

        # DMA data into VTCM
        _ = qshim_linear_udma_submit(tid, ddr_ptr, tile_size, vtcm_base, ...)
        ```

    When to use:
        - Store working data for HVX vector operations.
        - Create scratch buffers for intermediate results.
        - Implement software-managed cache for hot data.

    Warning:
        VTCM is shared among all threads on a core. Coordinate usage to avoid
        conflicts. Use thread_id-based offsets to partition VTCM among threads.
    """
    return external_call[
        "qshimGetBaseVtcmAddr", UnsafePointer[UInt8, MutAnyOrigin]
    ]()


@always_inline
def qshim_get_vtcm_size() -> UInt32:
    """Get the size of VTCM in bytes.

    Returns:
        VTCM size in bytes (8 MiB on hexagonv68, 16 MiB on hexagonv73).

    Example:
        ```mojo
        var vtcm_size = qshim_get_vtcm_size()
        var num_threads = qshim_get_num_threads()

        # Partition VTCM among threads
        var per_thread_vtcm = vtcm_size // Int(num_threads)
        var my_vtcm_offset = Int(thread_id) * per_thread_vtcm
        ```
    """
    # SDK 1.22: individual getters removed from JIT runtime; use qshimQuery.
    var val: Int64 = 0
    _ = external_call[
        "qshimQuery",
        UInt32,
    ](
        UInt32(4), UnsafePointer(to=val)
    )  # DEV_ATTR_QSHIM_VTCM_SIZE = 4
    return UInt32(val)


# ===-----------------------------------------------------------------------===#
# Memory Mapping
# ===-----------------------------------------------------------------------===#


@always_inline
def qshim_mmap(
    buf_handle: UnsafePointer[UInt8, MutAnyOrigin],
    offset: UInt64,
    size: UInt32,
    flag: UInt32,
    status: UnsafePointer[UInt32, MutAnyOrigin],
) -> UnsafePointer[UInt8, MutAnyOrigin]:
    """Map a buffer region into the kernel's virtual address space.

    Creates a virtual memory mapping for a buffer handle received through
    the kernel's pointer_array argument.

    Args:
        buf_handle: Buffer handle from the pointer_array.
        offset: Offset into the buffer to start mapping.
        size: Size of the region to map in bytes.
        flag: Mapping attributes (e.g., cacheable).
        status: Output pointer for status code (0 = success).

    Returns:
        Virtual address of the mapped region, or null on failure.

    Example:
        ```mojo
        var status: UInt32 = 0
        var mapped_ptr = qshim_mmap(
            buf_handle, 0, 4096, 0, UnsafePointer(to=status).as_unsafe_any_origin()
        )
        if status == 0 and mapped_ptr:
            # Access buffer through mapped_ptr
            var data = mapped_ptr.bitcast[Float32]()
            # ...

            # Unmap when done
            qshim_unmap(mapped_ptr)
        ```

    When to use:
        - Accessing buffers that weren't pre-mapped by the runtime.
        - Mapping specific regions of large buffers on demand.
        - Working with buffers that require specific caching behavior.

    Note:
        Always call qshim_unmap() when done to release the mapping.
    """
    var va_addr = external_call[
        "qshimMmap",
        UInt64,
    ](buf_handle, offset, size, flag, status)
    return UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(va_addr))


@always_inline
def qshim_unmap(va_addr: UnsafePointer[UInt8, MutAnyOrigin]):
    """Unmap a previously mapped buffer region.

    Releases a virtual memory mapping created by qshim_mmap().

    Args:
        va_addr: Virtual address returned by qshim_mmap().

    Example:
        ```mojo
        var mapped = qshim_mmap(handle, 0, size, 0, status_ptr)
        # ... use mapped memory ...
        qshim_unmap(mapped)
        ```
    """
    external_call["qshimUnmap", NoneType](_to_aic_jit_ptr(va_addr))


# ===-----------------------------------------------------------------------===#
# Profiling and Logging
# ===-----------------------------------------------------------------------===#


@always_inline
def qshim_profiling_entry_start[
    O: Origin
](thread_id: UInt32, event_name: UnsafePointer[UInt8, O]):
    """Mark the start of a profiling event.

    Records a timestamp for the start of a named event, which appears in
    Chrome trace format profiling output.

    Parameters:
        O: The origin of the event name pointer.

    Args:
        thread_id: The calling thread's ID.
        event_name: Name of the profiling event (null-terminated string).

    Example:
        ```mojo
        # Profile a computation section
        qshim_profiling_entry_start(thread_id, "matmul_tile".unsafe_ptr())
        compute_matmul_tile()
        qshim_profiling_entry_end(thread_id, "matmul_tile".unsafe_ptr())
        ```

    When to use:
        - Identifying performance bottlenecks in kernels.
        - Measuring time spent in different algorithm phases.
        - Comparing optimization approaches.

    Note:
        There is approximately 2μs overhead for writing the timestamp.
        Use sparingly in hot paths. Always pair with qshim_profiling_entry_end()
        using the same event name.
    """
    external_call[
        "qshimProfilingEntryStart",
        NoneType,
    ](thread_id, event_name)


@always_inline
def qshim_profiling_entry_end[
    O: Origin
](thread_id: UInt32, event_name: UnsafePointer[UInt8, O]):
    """Mark the end of a profiling event.

    Records a timestamp for the end of a named event started with
    qshim_profiling_entry_start().

    Parameters:
        O: The origin of the event name pointer.

    Args:
        thread_id: The calling thread's ID.
        event_name: Name of the profiling event (must match start event).

    Note:
        Must be called after qshim_profiling_entry_start() with the same
        event_name, otherwise the trace output will be malformed.
    """
    external_call[
        "qshimProfilingEntryEnd",
        NoneType,
    ](thread_id, event_name)


# ===-----------------------------------------------------------------------===#
# Logging
# ===-----------------------------------------------------------------------===#
#
# The qshimLog C function has signature:
#   void qshimLog(uint32_t mask, uint32_t dataCount, const char *formatStr, ...)
#
# Since the Hexagon target doesn't support variadic kgen.pack passing, we provide
# fixed-argument overloads for common use cases (0-4 format arguments).


@always_inline
def qshim_log[O: Origin](mask: UInt32, message: UnsafePointer[UInt8, O]):
    """Logs a simple message with no format arguments.

    Parameters:
        O: The origin of the message pointer.

    Args:
        mask: Log level mask (e.g., LOG_MASK_ERROR, LOG_MASK_DEBUG).
        message: The message string (null-terminated).

    Note:
        This function is only available in device code on Qualcomm accelerators.
    """
    external_call[
        "qshimLog",
        NoneType,
    ](mask, UInt32(0), message)


@always_inline
def qshim_log[
    O: Origin
](mask: UInt32, format_str: UnsafePointer[UInt8, O], arg0: UInt32):
    """Logs a formatted message with 1 UInt32 argument.

    Parameters:
        O: The origin of the format string pointer.

    Args:
        mask: Log level mask (e.g., LOG_MASK_ERROR, LOG_MASK_DEBUG).
        format_str: C-style format string (null-terminated).
        arg0: First format argument.

    Note:
        This function is only available in device code on Qualcomm accelerators.
        Format specifiers follow C printf conventions (use %u for UInt32).
    """
    external_call[
        "qshimLog",
        NoneType,
    ](mask, UInt32(1), format_str, arg0)


@always_inline
def qshim_log[
    O: Origin
](
    mask: UInt32,
    format_str: UnsafePointer[UInt8, O],
    arg0: UInt32,
    arg1: UInt32,
):
    """Logs a formatted message with 2 UInt32 arguments.

    Parameters:
        O: The origin of the format string pointer.

    Args:
        mask: Log level mask (e.g., LOG_MASK_ERROR, LOG_MASK_DEBUG).
        format_str: C-style format string (null-terminated).
        arg0: First format argument.
        arg1: Second format argument.

    Note:
        This function is only available in device code on Qualcomm accelerators.
        Format specifiers follow C printf conventions (use %u for UInt32).
    """
    external_call[
        "qshimLog",
        NoneType,
    ](mask, UInt32(2), format_str, arg0, arg1)


@always_inline
def qshim_log[
    O: Origin
](
    mask: UInt32,
    format_str: UnsafePointer[UInt8, O],
    arg0: UInt32,
    arg1: UInt32,
    arg2: UInt32,
):
    """Logs a formatted message with 3 UInt32 arguments.

    Parameters:
        O: The origin of the format string pointer.

    Args:
        mask: Log level mask (e.g., LOG_MASK_ERROR, LOG_MASK_DEBUG).
        format_str: C-style format string (null-terminated).
        arg0: First format argument.
        arg1: Second format argument.
        arg2: Third format argument.

    Note:
        This function is only available in device code on Qualcomm accelerators.
        Format specifiers follow C printf conventions (use %u for UInt32).
    """
    external_call[
        "qshimLog",
        NoneType,
    ](mask, UInt32(3), format_str, arg0, arg1, arg2)


@always_inline
def qshim_log[
    O: Origin
](
    mask: UInt32,
    format_str: UnsafePointer[UInt8, O],
    arg0: UInt32,
    arg1: UInt32,
    arg2: UInt32,
    arg3: UInt32,
):
    """Logs a formatted message with 4 UInt32 arguments.

    Parameters:
        O: The origin of the format string pointer.

    Args:
        mask: Log level mask (e.g., LOG_MASK_ERROR, LOG_MASK_DEBUG).
        format_str: C-style format string (null-terminated).
        arg0: First format argument.
        arg1: Second format argument.
        arg2: Third format argument.
        arg3: Fourth format argument.

    Note:
        This function is only available in device code on Qualcomm accelerators.
        Format specifiers follow C printf conventions (use %u for UInt32).
    """
    external_call[
        "qshimLog",
        NoneType,
    ](mask, UInt32(4), format_str, arg0, arg1, arg2, arg3)


# ===-----------------------------------------------------------------------===#
# Profiling with Integer Event IDs (PIC-compatible)
# ===-----------------------------------------------------------------------===#
#
# The string-based profiling functions (qshim_profiling_entry_start/end) require
# passing string literal pointers, which generates R_HEX_32_6_X relocations
# incompatible with position-independent code on Hexagon.
#
# These integer-based wrappers provide a PIC-compatible alternative by using
# predefined event ID constants instead of runtime string pointers.


@always_inline
def qshim_profiling_start_event(thread_id: UInt32, event_id: UInt32):
    """Mark the start of a profiling event using an integer ID.

    This is a PIC-compatible alternative to qshim_profiling_entry_start that
    avoids string literal relocations by using integer event IDs.

    Args:
        thread_id: The calling thread's ID.
        event_id: Numeric event identifier (user-defined, 0-255 recommended).

    Example:
        ```mojo
        # Define event IDs as constants
        comptime EVENT_LOAD: UInt32 = 0
        comptime EVENT_COMPUTE: UInt32 = 1
        comptime EVENT_STORE: UInt32 = 2

        def my_kernel(...) -> UInt32:
            var tid = entry_config[].thread_id

            qshim_profiling_start_event(tid, EVENT_COMPUTE)
            do_computation()
            qshim_profiling_end_event(tid, EVENT_COMPUTE)
            return 0
        ```

    Note:
        Event IDs appear as numeric values in trace output. Map them to
        meaningful names when analyzing profiling data.
    """
    # Use the underlying profiling API with a minimal static buffer
    # The event_id is encoded as a single-digit string to minimize overhead
    external_call["qshimProfilingEntryStart", NoneType](thread_id, event_id)


@always_inline
def qshim_profiling_end_event(thread_id: UInt32, event_id: UInt32):
    """Mark the end of a profiling event using an integer ID.

    This is a PIC-compatible alternative to qshim_profiling_entry_end.

    Args:
        thread_id: The calling thread's ID.
        event_id: Numeric event identifier (must match the start event).
    """
    external_call["qshimProfilingEntryEnd", NoneType](thread_id, event_id)


# ===-----------------------------------------------------------------------===#
# Execution Context Information
# ===-----------------------------------------------------------------------===#


@always_inline
def _qshim_query_u8(query_type: UInt32) -> UInt8:
    """Internal helper: call qshimQuery and return result as UInt8."""
    var val: Int64 = 0
    _ = external_call[
        "qshimQuery",
        UInt32,
    ](query_type, UnsafePointer(to=val))
    return UInt8(val)


@always_inline
def qshim_get_num_threads() -> UInt8:
    """Get the number of HVX threads executing the kernel.

    Returns:
        Thread count: 1-5 on hexagonv68.

    Example:
        ```mojo
        var num_threads = Int(qshim_get_num_threads())
        var work_per_thread = total_work // num_threads
        ```

    Note:
        This returns the actual thread count, which may differ from the
        thread_mask used at dispatch time.
    """
    return _qshim_query_u8(1)  # DEV_ATTR_QSHIM_NUM_THREADS


@always_inline
def qshim_get_core_id() -> UInt8:
    """Get the NSP core ID executing the current thread.

    Returns:
        Core ID: 0-15 on hexagonv68.

    Example:
        ```mojo
        var core_id = Int(qshim_get_core_id())
        var thread_id = Int(qshim_get_thread_id())
        var global_id = core_id * Int(qshim_get_num_threads()) + thread_id
        ```
    """
    return _qshim_query_u8(2)  # DEV_ATTR_QSHIM_CORE_ID


@always_inline
def qshim_get_thread_id() -> UInt8:
    """Get the HVX thread ID within the current core.

    Returns:
        Thread ID: 0-4 on hexagonv68.

    Example:
        ```mojo
        var tid = Int(qshim_get_thread_id())
        # Use tid for thread-local work partitioning
        ```
    """
    return _qshim_query_u8(0)  # DEV_ATTR_QSHIM_THREAD_ID


@always_inline
def qshim_get_num_cores() -> UInt8:
    """Get the number of NSP cores executing the kernel.

    Returns:
        Core count: 1-16 on hexagonv68.

    Example:
        ```mojo
        var num_cores = Int(qshim_get_num_cores())
        var num_threads = Int(qshim_get_num_threads())
        var total_workers = num_cores * num_threads
        ```

    Note:
        Use this with qshim_get_num_threads() to compute total parallelism.
    """
    return _qshim_query_u8(3)  # DEV_ATTR_QSHIM_NUM_CORES


@always_inline
def qshim_get_num_hvx_units() -> UInt8:
    """Get the number of HVX units available.

    Returns:
        Number of HVX units on the device.

    Note:
        This indicates hardware capability, not currently active units.
    """
    return _qshim_query_u8(5)  # DEV_ATTR_QSHIM_NUM_HVX_UNITS


# ===-----------------------------------------------------------------------===#
# ECC Error Monitoring
# ===-----------------------------------------------------------------------===#


@always_inline
def qshim_get_ecc_error_count(
    mem_type: UInt32,
    ce_count: UnsafePointer[UInt32, MutAnyOrigin],
    ue_count: UnsafePointer[UInt32, MutAnyOrigin],
) -> UInt32:
    """Get ECC (Error Correcting Code) error counts for memory.

    Retrieves correctable and uncorrectable error counts for the specified
    memory type.

    Args:
        mem_type: Memory type (use ECC_MEM_VTCM for VTCM).
        ce_count: Output pointer for correctable error count.
        ue_count: Output pointer for uncorrectable error count.

    Returns:
        0 on success, error code on failure:
        - JIT_DEV_QSHIM_ERROR_ECC_INVALID_PARAMETER: Invalid mem_type or null pointers.
        - JIT_DEV_QSHIM_ERROR_ECC_GET_ERROR_COUNT_ECC_DISABLED: ECC is disabled.
        - JIT_DEV_QSHIM_ERROR_ECC_GET_ERROR_COUNT_FAIL: General failure.

    Example:
        ```mojo
        var ce: UInt32 = 0
        var ue: UInt32 = 0
        var status = qshim_get_ecc_error_count(
            ECC_MEM_VTCM, UnsafePointer(to=ce), UnsafePointer(to=ue)
        )
        if status == 0:
            if ue > 0:
                # Handle uncorrectable errors
                pass
        ```

    When to use:
        - Monitoring memory health in long-running applications.
        - Detecting potential hardware issues.
        - Implementing error recovery strategies.
    """
    return external_call[
        "qshimGetEccErrorCount",
        UInt32,
    ](mem_type, ce_count, ue_count)


# ===-----------------------------------------------------------------------===#
# Multicast Operations (SDK 1.22+)
# ===-----------------------------------------------------------------------===#

# Multicast synchronization points.
# Multiple multicast transfers using the same syncpoint with overlapping
# NSP bitmaps simultaneously leads to undefined behavior.
comptime MCSP_0: UInt32 = 0
comptime MCSP_1: UInt32 = 1
comptime MCSP_2: UInt32 = 2
comptime MCSP_3: UInt32 = 3
comptime MCSP_4: UInt32 = 4
comptime MCSP_5: UInt32 = 5
comptime MCSP_6: UInt32 = 6
comptime MCSP_7: UInt32 = 7
comptime MCSP_8: UInt32 = 8
comptime MCSP_9: UInt32 = 9
comptime MCSP_10: UInt32 = 10
comptime MCSP_11: UInt32 = 11
comptime MCSP_12: UInt32 = 12
comptime MCSP_13: UInt32 = 13
comptime MCSP_14: UInt32 = 14
comptime MCSP_15: UInt32 = 15


@always_inline
def qshim_multicast_xfer(
    thread_id: UInt32,
    src: UnsafePointer[UInt8, MutAnyOrigin],
    dst_offsets: UnsafePointer[
        UnsafePointer[UInt32, MutAnyOrigin], MutAnyOrigin
    ],
    dst_nsp_mask: UInt32,
    syncpoint: UInt32,
    size: UInt32,
) -> UInt32:
    """Broadcast data from the calling NSP's VTCM to multiple destination NSPs.

    Initiates a multicast DMA that copies `size` bytes from `src` on the
    calling NSP to the VTCM of each destination NSP specified in
    `dst_nsp_mask`. Destination NSPs call `qshim_wait_on_multicast()` with
    the same syncpoint to wait for the data to arrive.

    Args:
        thread_id: The calling thread's ID.
        src: Source VTCM address on the calling NSP.
        dst_offsets: Array of pointers to destination VTCM offsets, one per
            NSP in the system. Element `i` points to the VTCM offset for
            NSP `i`. The array size must match the total number of NSPs.
            Only entries for NSPs set in `dst_nsp_mask` are read.
        dst_nsp_mask: Bitmask of destination NSPs. Bit 0 = NSP0, bit 1 = NSP1,
            and so on.
        syncpoint: Multicast synchronization point (MCSP_0 through MCSP_15).
            Do not reuse a syncpoint whose wait has not yet completed.
            Concurrent use of the same syncpoint with overlapping NSP masks
            leads to undefined behavior.
        size: Number of bytes to copy from `src` to each destination.

    Returns:
        0 on success, non-zero on error.

    Example:
        ```mojo
        # On source NSP (e.g., NSP 0): broadcast 4KB to NSPs 1 and 2
        var vtcm = qshim_get_base_vtcm_addr()
        var src = vtcm  # source data at VTCM base

        # Build destination offset array (one entry per NSP in the system)
        var num_nsps = Int(qshim_get_num_cores())
        var offsets = UnsafePointer[
            UnsafePointer[UInt32, MutAnyOrigin], MutAnyOrigin
        ].alloc(num_nsps)
        var dst_off_1: UInt32 = 0  # NSP 1 receives at VTCM offset 0
        var dst_off_2: UInt32 = 0  # NSP 2 receives at VTCM offset 0
        offsets[1] = UnsafePointer(to=dst_off_1)
        offsets[2] = UnsafePointer(to=dst_off_2)

        _ = qshim_multicast_xfer(tid, src, offsets, 0x6, MCSP_0, 4096)

        # On destination NSPs 1 and 2:
        _ = qshim_wait_on_multicast(MCSP_0)
        # Data is now available in VTCM at the specified offsets
        ```

    Warning:
        This function must be called from the source NSP. The source NSP
        should NOT be included in `dst_nsp_mask`.
    """
    return external_call[
        "qshimMulticastXfer",
        UInt32,
    ](
        thread_id,
        _to_aic_jit_ptr(src),
        dst_offsets,
        dst_nsp_mask,
        syncpoint,
        size,
    )


@always_inline
def qshim_wait_on_multicast(syncpoint: UInt32) -> UInt32:
    """Wait for a multicast operation to complete.

    Blocks until the multicast transfer associated with the given syncpoint
    finishes on the receiving NSP.

    Args:
        syncpoint: Multicast synchronization point (MCSP_0 through MCSP_15).

    Returns:
        0 on success, non-zero on failure.

    Example:
        ```mojo
        # On source NSP: initiate multicast
        _ = qshim_multicast_xfer(tid, src, dst_offsets, mask, MCSP_0, size)

        # On destination NSPs: wait for data arrival
        _ = qshim_wait_on_multicast(MCSP_0)
        ```
    """
    return external_call["qshimWaitOnMulticast", UInt32](syncpoint)


# ===-----------------------------------------------------------------------===#
# NSP Synchronization (recent SDK)
# ===-----------------------------------------------------------------------===#


@always_inline
def qshim_sync_nsps(nsp_mask: UInt32) -> UInt32:
    """Synchronize multiple NSP cores at a barrier.

    All NSPs specified in the bitmask must have one designated thread call
    this function. The call blocks until all participating NSPs reach the
    barrier, but it does NOT wait for non-designated threads on each NSP.
    Use `qshim_sync_all_threads_all_nsps()` to create a full stage barrier
    across all participating threads on all participating NSPs.

    Args:
        nsp_mask: Bitmask of NSPs to synchronize. Bit 0 = NSP0, bit 1 = NSP1,
            and so on.

    Returns:
        0 on success, non-zero on failure.

    Example:
        ```mojo
        # Synchronize all threads across NSPs 0 and 1.
        var sync_ptr = qshim_create_thread_sync_point[thread_mask=0x1F]()
        _ = qshim_sync_all_threads_all_nsps(tid, 0x3, sync_ptr)
        ```

    Warning:
        Only ONE thread per NSP should call this function. Calling
        `qshim_sync_threads()` only after `qshim_sync_nsps()` is not enough to
        form a full cross-NSP stage barrier, because non-designated threads on
        another NSP may still be executing the previous stage.
    """
    return external_call["qshimSyncNSPs", UInt32](nsp_mask)


@always_inline
def qshim_sync_all_threads_all_nsps(
    thread_id: UInt32,
    nsp_mask: UInt32,
    sync_point: UnsafePointer[QShimThreadSyncStruct, MutAnyOrigin],
    leader_thread_id: UInt32 = 0,
) -> UInt32:
    """Synchronize all participating threads across all participating NSPs.

    This helper implements the required hierarchical barrier:

    1. All local threads reach `qshim_sync_threads()` so the designated thread
       cannot enter `qshim_sync_nsps()` early.
    2. One designated thread per NSP calls `qshim_sync_nsps()`.
    3. All local threads synchronize again before starting the next stage.

    The same sync point is reused for both local barriers. This is safe because
    `qshim_sync_threads()` restores the original thread mask when a barrier
    completes, allowing sequential reuse of the sync struct.

    Args:
        thread_id: The calling thread's ID (0-4 on hexagonv68).
        nsp_mask: Bitmask of NSPs that participate in the cross-NSP barrier.
        sync_point: Shared local thread barrier used before and after the
            designated thread calls `qshim_sync_nsps()`.
        leader_thread_id: The local thread that participates in
            `qshim_sync_nsps()`. Defaults to thread 0.

    Returns:
        Status returned by `qshim_sync_nsps()` on the designated thread,
        otherwise 0.

    Example:
        ```mojo
        var sync_ptr = qshim_create_thread_sync_point[thread_mask=0x1F]()
        _ = qshim_sync_all_threads_all_nsps(tid, 0x3, sync_ptr)
        ```
    """
    _ = qshim_sync_threads(thread_id, sync_point)

    var status: UInt32 = 0
    if thread_id == leader_thread_id:
        status = qshim_sync_nsps(nsp_mask)

    _ = qshim_sync_threads(thread_id, sync_point)
    return status


# ===-----------------------------------------------------------------------===#
# Generic Hardware Query (SDK 1.22+)
# ===-----------------------------------------------------------------------===#

# Query type constants for qshim_query()
comptime QSHIM_QUERY_THREAD_ID: UInt32 = 0
"""Query the current thread ID."""

comptime QSHIM_QUERY_NUM_THREADS: UInt32 = 1
"""Query the number of threads."""

comptime QSHIM_QUERY_CORE_ID: UInt32 = 2
"""Query the current core ID."""

comptime QSHIM_QUERY_NUM_CORES: UInt32 = 3
"""Query the number of cores."""

comptime QSHIM_QUERY_VTCM_SIZE: UInt32 = 4
"""Query the VTCM size in bytes."""

comptime QSHIM_QUERY_NUM_HVX_UNITS: UInt32 = 5
"""Query the number of HVX units."""

comptime QSHIM_QUERY_HMX_THREAD_ID: UInt32 = 6
"""Query the HMX thread ID."""


@always_inline
def qshim_query(
    query_type: UInt32,
    out_val: UnsafePointer[Int64, MutAnyOrigin],
) -> UInt32:
    """Query NSP hardware details using a generic query interface.

    This is a unified query API that can retrieve various hardware
    attributes. For common queries, prefer the dedicated functions
    (qshim_get_num_threads, qshim_get_core_id, etc.) which are simpler.

    Args:
        query_type: The attribute to query (QSHIM_QUERY_* constants).
        out_val: Output pointer for the queried value.

    Returns:
        0 on success, non-zero on failure (e.g., unsupported query type).

    Example:
        ```mojo
        var hmx_tid: Int64 = 0
        var status = qshim_query(QSHIM_QUERY_HMX_THREAD_ID, UnsafePointer(to=hmx_tid))
        if status == 0:
            # Use hmx_tid for HMX-specific work partitioning
            pass
        ```
    """
    return external_call[
        "qshimQuery",
        UInt32,
    ](query_type, out_val)
