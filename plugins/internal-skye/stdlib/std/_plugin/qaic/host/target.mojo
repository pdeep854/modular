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
"""QAIC (Qualcomm AI) compilation target definitions for Mojo.

This module provides target definitions for compiling Mojo code to run on
Qualcomm AI accelerators.
"""

from std.compile import compile_info, CompiledFunctionInfo
from std.gpu.host import get_gpu_target
from std.gpu.host.info import GPUInfo
from std.reflection import reflect, get_linkage_name
from std.reflection.reflect import _field_types_of
from std.collections import Array
from std.ffi import external_call
from std.memory import alloc, stack_allocation, unsafe_memcpy
from std.sys import align_of, size_of
from std.sys.info import _accelerator_arch, _TargetType, is_triple
from std.utils.index import IndexList

from std.builtin.variadics import _call_with_dynamic_pack_pointers

from . import qshim


# ===-----------------------------------------------------------------------===#
# QAIC kernel-side unwinding (setjmp/longjmp)
# ===-----------------------------------------------------------------------===#
#
# Hexagon `llvm.intr.trap` hangs the device until the QAIC watchdog fires
# (~hours), so a fatal kernel must instead return non-zero. The wrapper does
# `setjmp` before invoking the user kernel and `abort()` does `longjmp` back.
# The active jmp_buf is stored per-thread in `qaic_unwind.c` (Mojo's stdlib
# build forbids module-level mutable state).
#
# Hexagon `setjmp` saves 15 ints (r16-r27, r29-r31 = 60 bytes); 64 ints
# allocated for padding.

comptime _QAIC_JMP_BUF_INTS = 64


struct QaicUnwindGuard:
    """Context manager for QAIC kernel-side setjmp/longjmp error handling.

    Arms a per-thread setjmp trap before the user kernel runs and
    auto-disarms on scope exit regardless of how the scope is left.

    `__enter__` MUST be `@always_inline` so that the `setjmp` instruction
    is emitted inside the wrapper's own stack frame.  If it were in a
    separate (non-inlined) callee, the callee's frame would be gone by the
    time `longjmp` fires and the jump would land in a dead frame.

    When `longjmp` fires the CPU jumps back to the `setjmp` instruction
    (which is inlined into the `with` preamble).  `setjmp` returns non-zero,
    `__enter__` returns True if `longjmp` fired (kernel aborted), which the
    caller uses to return `JIT_DEV_ERROR_FATAL`.

    Usage:
        with QaicUnwindGuard() as tripped:
            if tripped:
                # JIT_DEV_ERROR_FATAL: host surfaces this on `synchronize()`.
                return 14
            user_kernel(...)
        return 0
    """

    var _jmpbuf: Array[Int32, _QAIC_JMP_BUF_INTS]

    def __init__(out self):
        self._jmpbuf = Array[Int32, _QAIC_JMP_BUF_INTS](fill=0)

    @always_inline
    def __enter__(mut self) -> Bool:
        external_call["__qaic_unwind_set", NoneType](self._jmpbuf.unsafe_ptr())
        var rc = __mlir_op.`pop.external_call`[
            func=__mlir_attr[`"setjmp" : !kgen.string`],
            funcAttrs=__mlir_attr.`["returns_twice"]`,
            _type=Int32,
        ](self._jmpbuf.unsafe_ptr())
        return rc != 0

    @always_inline
    def __exit__(mut self):
        external_call["__qaic_unwind_clear", NoneType]()


@always_inline("nodebug")
def _is_npu_target[t: _TargetType]() -> Bool:
    """Returns True if the target is a QAIC (Qualcomm NPU) target.

    QAIC targets use the Hexagon triple, distinguishing them from CPU and
    other GPU targets at compile time. This predicate is the dispatch hook
    for unified APIs (`compile_function`, `enqueue_function`) that take an
    NPU branch when `t` is a QAIC target.

    Parameters:
        t: The target to check.

    Returns:
        True if `t` targets the QAIC NPU.
    """
    return is_triple["hexagon-unknown-linux-musl", t]()


@always_inline("nodebug")
def _is_ondevice_npu_target[t: _TargetType]() -> Bool:
    """Returns True if the target is the on-device (integrated) NPU.

    The two parts share this plugin — the on-device NPU's target declares
    `stdlib_plugin = "qaic"`, so every stdlib dispatch site resolves to
    `QAICBackend` for both — and they differ in how a kernel reaches the
    device. The card JIT-compiles one and loads it; the on-device part loads
    only names its signed library already carries, so a hook that generates
    device code serves one and not the other. The triple is what separates
    them: QuRT loads a bare-metal ELF where the card loads a musl image.

    This is the complement of `_is_npu_target` rather than its negation:
    `_is_npu_target` is False here, and a hook that only asks that question
    cannot tell this part from a GPU.

    Parameters:
        t: The target to check.

    Returns:
        True if `t` targets the on-device NPU.
    """
    return is_triple["hexagon-unknown-unknown-elf", t]()


# ===-----------------------------------------------------------------------===#
# QAIC JIT Entry Point Types
# ===-----------------------------------------------------------------------===#
#
# These types match the C definitions in the QAIC SDK:
#   /opt/qti-aic/dev/inc/jit/jit_dev_exe_function.h
#
# Hardware limits from /opt/qti-aic/dev/inc/jit/jit_qshim_api.h:
#   AI100: 1-16 cores (core_id 0-15), 1-5 threads (thread_id 0-4)


@fieldwise_init
struct AicJitEntryPointConfig(TrivialRegisterPassable):
    """Configuration passed to JIT kernel entry points.

    This struct is passed by the QAIC runtime to each kernel invocation,
    providing information about the execution context.

    Attributes:
        core_id: The NSP (Neural Signal Processor) core ID.
        num_cores: Total number of NSP cores executing this batch.
        thread_id: The HVX thread ID within the core.
        num_threads: Total number of HVX threads executing per core.
    """

    var core_id: UInt32
    var num_cores: UInt32
    var thread_id: UInt32
    var num_threads: UInt32


@fieldwise_init
struct NpuGridConfig(TrivialRegisterPassable):
    """CUDA-style logical launch parameters for NPU kernels.

    Injected at FAM slot 0 by `enqueue_function` whenever the user
    kernel takes an `UnsafePointer[NpuGridConfig, ...]` as its first
    declared argument. Kernels use these values to implement
    element-level indexing independently of the physical hardware
    partitioning.

    The QAIC hardware is always launched at full capacity regardless of
    these values. The kernel combines NpuGridConfig with entry_config
    (core_id, thread_id) to map hardware threads onto logical elements.

    NpuGridConfig occupies FAM slot 0; user buffer arguments follow in
    slots 1..N.

    Example:
        ```mojo
        # slot 0 → grid config (always first in v2 kernels)
        var cfg        = npu_get_grid_config(pointer_array)
        # slots 1..N → user buffers
        var input      = npu_get_buffer[Float32](pointer_array, 1)
        var output     = npu_get_buffer[Float32](pointer_array, 2)

        var block_idx  = Int(entry_config[].core_id)
        var thread_idx = Int(entry_config[].thread_id)
        var num_cores  = Int(entry_config[].num_cores)

        # block-stride loop: each core handles every num_cores-th block
        for b in range(block_idx, Int(cfg.num_blocks), num_cores):
            var i = b * Int(cfg.block_dim) + thread_idx
            if i < Int(cfg.grid_dim):
                output[i] = compute(input[i])
        ```

    Attributes:
        grid_dim:   Total number of elements in the problem (e.g. 1024).
        block_dim:  Logical block size requested by the caller (e.g. 5).
        num_blocks: ceil(grid_dim / block_dim), precomputed on the host.
    """

    var grid_dim: UInt32
    var block_dim: UInt32
    var num_blocks: UInt32


struct AicJitPointerArray(TrivialRegisterPassable):
    """Array of buffer pointers passed to JIT kernels.

    The QAIC runtime passes buffer pointers through this structure.
    Kernels access their input/output buffers via the pointers array.

    IMPORTANT - Flexible Array Member (FAM):
        This struct uses a C99 Flexible Array Member (FAM). The term "FAM" refers
        to a C feature (ISO C99 §6.7.2.1) where a struct's last member is an array
        of unspecified size (`type name[]`). The array data is stored INLINE after
        the struct, not as a pointer to elsewhere.

        C definition from QAIC SDK (jit_dev_exe_function.h):
            typedef uint64_t AicJitPtr;  // SDK 1.23: pointers[] are AicJitPtr
            typedef struct AicJitPointerArray {
                uint32_t numPointers;
                uint32_t reserved;
                AicJitPtr pointers[];  // FAM - the [] with no size is the key indicator
            } AicJitPointerArray;

        SDK 1.23 widened each pointer from `void *` (`AicPtr`) to a fixed
        `uint64_t` (`AicJitPtr`, to support 64-bit AI 200 addressing) and
        inserted a `reserved` word so the array is 8-byte aligned. On the
        32-bit NSP the device address occupies the low 32 bits of each slot
        and the high 32 bits are zero. This IS an ABI change versus the
        prior 4-byte `void *pointers[]` layout.

    FAM vs Regular Pointer - Why This Matters:

        WITHOUT FAM (regular pointer to separate array):
            ```c
            struct { uint32_t n; void **pointers; };  // pointer TO array
            ```
            Memory: TWO separate allocations
            ┌───────┬──────────┐
            │   n   │ pointers ────────┐
            └───────┴──────────┘       │
                                       ▼
                           ┌──────┬──────┬──────┐
                           │ ptr0 │ ptr1 │ ptr2 │  (elsewhere)
                           └──────┴──────┴──────┘

        WITH FAM (inline array) - What QAIC uses:
            ```c
            struct { uint32_t n; void *pointers[]; };  // array IS inline
            ```
            Memory: ONE contiguous block
            ┌───────┬──────┬──────┬──────┐
            │   n   │ ptr0 │ ptr1 │ ptr2 │  (all together)
            └───────┴──────┴──────┴──────┘

        Why QAIC uses FAM:
        - One allocation instead of two (simpler, faster)
        - Better cache locality (contiguous memory)
        - Easier to pass to device (single buffer)

    Memory layout (SDK 1.23, AicJitPtr is a 8-byte uint64_t):
        [numPointers:4][reserved:4][ptr0:8][ptr1:8][ptr2:8]...
        offset:   0          4         8      16      24

    How to access pointers in kernel code:
        The Mojo struct cannot directly represent a FAM, so kernels must compute
        byte offsets from the struct base to access the inline pointer array.
        Viewing the array as UInt64 words, the header (numPointers + reserved)
        is one 8-byte word and each pointer is one 8-byte word, so pointer `i`
        is at word `i + 1`:

        ```mojo
        var base = pointer_array.bitcast[UInt64]()
        var ptr0_addr = (base + 1)[]  # offset 8  = pointers[0]
        var ptr1_addr = (base + 2)[]  # offset 16 = pointers[1]
        var input = UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=Int(ptr0_addr)
        )
        ```

    Debugging note:
        This FAM layout was discovered when kernels produced incorrect output
        (all zeros). Disassembly showed correct offset access, but the
        struct was treating the `pointers` field as a pointer-to-array instead
        of inline data. The fix was to access pointers via byte offsets.

    Attributes:
        num_pointers: Number of pointers in the array.
        reserved: Padding word (SDK 1.23) that 8-byte-aligns `pointers`.
        pointers: First pointer in the inline array (offset 8 from struct
            base). A 64-bit `AicJitPtr`; on the 32-bit NSP the address is in
            the low 32 bits.
    """

    var num_pointers: UInt32
    var reserved: UInt32
    # This field exists at offset 8, which is pointers[0] in the C FAM layout.
    # It provides convenient access to the first pointer, but for additional
    # pointers, use byte offset computation as shown in the docstring above.
    var pointers: UInt64


# ===-----------------------------------------------------------------------===#
# QAIC Kernel Function Type
# ===-----------------------------------------------------------------------===#

# The expected signature for QAIC JIT kernel entry points:
# def(entry_config: UnsafePointer[AicJitEntryPointConfig],
#    pointer_array: UnsafePointer[AicJitPointerArray]) -> UInt32
#
# Return value: 0 for success, non-zero for error


# ===-----------------------------------------------------------------------===#
# QAIC JIT Runtime Libraries
# ===-----------------------------------------------------------------------===#
#
# The QAIC JIT compiler automatically links the Hexagon runtime libraries
# (libgcc.a and libc.a) from the Qualcomm SDK when producing shared objects.
# This provides full support for:
#
#   Division operations:
#     - Integer division (//), modulo (%), etc.
#     - Implemented via __hexagon_divsi3, __hexagon_udivsi3, etc.
#
#   Math functions (from libc.a):
#     - Trigonometric: cosf, sinf, tanf, asinf, acosf, atanf
#     - Exponential: expf, logf, powf
#     - Other: sqrtf, fmodf, floorf, ceilf
#
# The libraries are found automatically based on the hexagon-link location
# in the Qualcomm SDK (typically /opt/qti-aic/dev/hexagon_tools-*/target/
# hexagon/lib/v68/G0/pic/).
#
# Performance note: Division by compile-time CONSTANTS is highly optimized.
# The compiler transforms `x // 7` into multiplication by magic constants,
# which is faster than calling the division helper.


# ===-----------------------------------------------------------------------===#
# QAIC JIT Tips - Power-of-2 Division
# ===-----------------------------------------------------------------------===#
#
# While integer division is now fully supported, using bit-shifting for
# power-of-2 divisors is still faster (single instruction vs function call).
#
# Example - Multi-threaded work distribution:
#     ```mojo
#     # Define thread count as power of 2
#     comptime LOG2_THREADS: Int = 2  # 4 threads
#     comptime NUM_THREADS: Int = 1 << LOG2_THREADS  # 4
#
#     def my_kernel(...) -> UInt32:
#         var thread_id = Int(config.thread_id)
#         var size = Int(size_ptr[])
#
#         # Use bit shift for power-of-2 division (faster)
#         var chunk_size = size >> LOG2_THREADS  # size / 4
#         var start = thread_id * chunk_size
#         var end = start + chunk_size
#         if thread_id == NUM_THREADS - 1:
#             end = size  # Last thread handles remainder
#
#         for i in range(start, end):
#             # ... process elements ...
#     ```
#
# For non-power-of-2 divisors, regular division (//) works fine:
#     var chunk_size = size // num_threads  # Works!


# ===-----------------------------------------------------------------------===#
# HVX Constants
# ===-----------------------------------------------------------------------===#

comptime HVX_VECTOR_SIZE: Int = 128
"""Size of an HVX vector in bytes (1024 bits)."""

comptime FLOAT32_PER_HVX: Int = 32
"""Number of float32 elements per HVX vector."""

comptime FLOAT16_PER_HVX: Int = 64
"""Number of float16 elements per HVX vector."""

comptime INT32_PER_HVX: Int = 32
"""Number of int32 elements per HVX vector."""

comptime INT16_PER_HVX: Int = 64
"""Number of int16 elements per HVX vector."""

comptime INT8_PER_HVX: Int = 128
"""Number of int8 elements per HVX vector."""


# ===-----------------------------------------------------------------------===#
# NSP Info
# ===-----------------------------------------------------------------------===#

comptime NPU_NUM_THREADS: Int = GPUInfo.from_name[
    _accelerator_arch()
]().threads_per_multiprocessor
"""Number of hardware threads per NSP core on the configured accelerator."""

comptime NPU_NUM_CORES: Int = GPUInfo.from_name[_accelerator_arch()]().sm_count
"""Number of NSP cores on the configured accelerator."""

comptime NPU_ALL_THREADS: Int = (1 << NPU_NUM_THREADS) - 1
"""Thread bitmask selecting every thread of an NSP core.

Hardware dispatch takes a per-core thread bitmask, not a thread count, so
"all threads" is the low `NPU_NUM_THREADS` bits set (`0x1F` on aic100).
"""


@always_inline("nodebug")
def is_aic100[target_arch: StaticString = _accelerator_arch()]() -> Bool:
    return "aic100" in target_arch


@always_inline("nodebug")
def is_aic200[target_arch: StaticString = _accelerator_arch()]() -> Bool:
    return "aic200" in target_arch


@always_inline("nodebug")
def is_qaic[target_arch: StaticString = _accelerator_arch()]() -> Bool:
    return is_aic100[target_arch]() or is_aic200[target_arch]()


@always_inline("nodebug")
def num_threads[target_arch: StaticString = _accelerator_arch()]() -> Int:
    return GPUInfo.from_name[target_arch]().threads_per_multiprocessor


@always_inline("nodebug")
def num_hvx_threads[target_arch: StaticString = _accelerator_arch()]() -> Int:
    comptime if is_aic100[target_arch]():
        return 4
    elif is_aic200[target_arch]():
        return 8
    else:
        comptime assert False, "unknown target for hvx threads"


@always_inline("nodebug")
def is_hvx[
    target_arch: StaticString = _accelerator_arch()
](thread_id: Int) -> Bool:
    return thread_id < num_hvx_threads[target_arch]()


@always_inline("nodebug")
def is_hmx[
    target_arch: StaticString = _accelerator_arch()
](thread_id: Int) -> Bool:
    return thread_id == num_hvx_threads[target_arch]()


@always_inline("nodebug")
def hmx_thread_id[target_arch: StaticString = _accelerator_arch()]() -> Int:
    comptime if is_aic100[target_arch]():
        return 4
    elif is_aic200[target_arch]():
        return 8
    else:
        comptime assert False, "unknown target for hmx thread id"


@always_inline("nodebug")
def dma_thread_id[target_arch: StaticString = _accelerator_arch()]() -> Int:
    comptime if is_aic100[target_arch]():
        return 4
    elif is_aic200[target_arch]():
        return 9
    else:
        comptime assert False, "unknown target for dma thread id"


@always_inline("nodebug")
def is_hmx_dtype[
    target_arch: StaticString = _accelerator_arch(), //, dtype: DType
]() -> Bool:
    comptime if is_aic100[target_arch]():
        return dtype in (DType.float16, DType.int8)

    elif is_aic200[target_arch]():
        return dtype in (
            DType.float16,
            DType.int8,
            DType.bfloat16,
            DType.float8_e4m3fn,
        )
    else:
        comptime assert False, "unknown target for hmx dtype"


@always_inline("nodebug")
def nsp_mask(num_cores: Int = npu_num_cores()) -> Int:
    debug_assert(
        num_cores <= NPU_NUM_CORES,
        "nsp_mask: num_cores exceeds the NSP count of this accelerator",
    )
    return (1 << num_cores) - 1


@always_inline("nodebug")
def thread_mask[
    target_arch: StaticString = _accelerator_arch(),
]() -> Int:
    return (1 << num_threads[target_arch]()) - 1


@always_inline("nodebug")
def hvx_mask[target_arch: StaticString = _accelerator_arch()]() -> Int:
    return (1 << num_hvx_threads[target_arch]()) - 1


@always_inline("nodebug")
def hvx_and_dma_mask[target_arch: StaticString = _accelerator_arch()]() -> Int:
    return hvx_mask[target_arch]() | (1 << dma_thread_id[target_arch]())


@always_inline("nodebug")
def hvx_and_hmx_mask[target_arch: StaticString = _accelerator_arch()]() -> Int:
    return hvx_mask[target_arch]() | (1 << hmx_thread_id[target_arch]())


@always_inline("nodebug")
def hmx_and_dma_mask[target_arch: StaticString = _accelerator_arch()]() -> Int:
    return (1 << hmx_thread_id[target_arch]()) | (
        1 << dma_thread_id[target_arch]()
    )


# ===-----------------------------------------------------------------------===#
# NPU Buffer Helpers - Simplify pointer_array access
# ===-----------------------------------------------------------------------===#


@always_inline
def npu_get_buffer[
    T: AnyType
](
    pointer_array: UnsafePointer[AicJitPointerArray, MutAnyOrigin], index: Int
) -> UnsafePointer[T, MutAnyOrigin]:
    """Extract a typed buffer pointer from the pointer_array.

    This helper simplifies the FAM (Flexible Array Member) access pattern
    for QAIC kernels.

    Safety:
        This function performs a debug assertion that `index` is within bounds
        of the pointer array. In release builds, no bounds checking is performed
        and callers must ensure `index < pointer_array[].num_pointers`.

    Parameters:
        T: The element type of the buffer.

    Args:
        pointer_array: The pointer_array passed to the kernel.
        index: The buffer index (0-based). Must be less than num_pointers.

    Returns:
        A typed pointer to the buffer data.

    Example:
        ```mojo
        def my_kernel(
            entry_config: UnsafePointer[AicJitEntryPointConfig, MutAnyOrigin],
            pointer_array: UnsafePointer[AicJitPointerArray, MutAnyOrigin],
        ) -> UInt32:
            var input = npu_get_buffer[Float32](pointer_array, 0)
            var output = npu_get_buffer[Float32](pointer_array, 1)
            # ... use input and output ...
            return 0
        ```
    """
    debug_assert(
        index >= 0 and index < Int(pointer_array[].num_pointers),
        "npu_get_buffer: index out of bounds",
    )
    var base = pointer_array.bitcast[UInt64]()
    var ptr_addr = (base + _npu_fam_word(index))[]
    return UnsafePointer[T, MutAnyOrigin](unsafe_from_address=Int(ptr_addr))


@always_inline
def npu_get_grid_config(
    pointer_array: UnsafePointer[AicJitPointerArray, MutAnyOrigin],
) -> NpuGridConfig:
    """Return the logical launch parameters injected by `enqueue_function`.

    NpuGridConfig is stored in FAM slot 0 whenever the kernel takes
    an `UnsafePointer[NpuGridConfig, ...]` as its first declared
    argument. User buffer arguments follow in slots 1..N and must be
    accessed with `npu_get_buffer(..., i+1)`. Calling this from a
    kernel that does not declare `NpuGridConfig` is undefined
    behaviour.

    Args:
        pointer_array: The pointer_array received by the kernel entry point.

    Returns:
        The NpuGridConfig holding grid_dim, block_dim, and num_blocks.
    """
    return npu_get_buffer[NpuGridConfig](pointer_array, 0)[]


# ===-----------------------------------------------------------------------===#
# Type Detection Helpers
# ===-----------------------------------------------------------------------===#


def looks_like_pointer[T: AnyType]() -> Bool:
    """Detect Pointer types by their base type name.

    Parameters:
        T: The type to check.

    Returns:
        True if T is an Pointer type.
    """
    comptime base_name = reflect[T].base_name()
    return base_name == "Pointer"


def looks_like_tile_tensor[T: AnyType]() -> Bool:
    """Detect TileTensor types by their base type name.

    Parameters:
        T: The type to check.

    Returns:
        True if T is a TileTensor type.
    """
    comptime base_name = reflect[T].base_name()
    return base_name == "TileTensor"


comptime _static_tile_tensor_size_matches[T: AnyType] = (
    looks_like_tile_tensor[T]()
    and size_of[T]() == size_of[UnsafePointer[NoneType, MutAnyOrigin]]()
)
"""True when T is a TileTensor whose struct is exactly one pointer in size.

On host (64-bit):   size_of[T]() == 8 (one 64-bit pointer, zero layout bytes)
On Hexagon (32-bit): size_of[T]() == 4 (one 32-bit pointer, zero layout bytes)
"""


def looks_like_static_tile_tensor[T: AnyType]() -> Bool:
    """Detect TileTensor types whose struct is pointer-only (all-static layout).

    A static TileTensor has all ComptimeInt layout dimensions, so no runtime
    layout bytes are stored.  Its struct is exactly one pointer in size:
      - On the host (64-bit):    size_of[T]() == 8
      - On Hexagon (32-bit):     size_of[T]() == 4

    The comparison against `size_of[UnsafePointer[NoneType, MutAnyOrigin]]()`
    is target-aware, so this function returns the correct answer whether it is
    evaluated on the host or inside Hexagon-compiled code.

    Parameters:
        T: The type to check.

    Returns:
        True if T is a TileTensor with a zero-sized (all compile-time) layout.
    """
    return _static_tile_tensor_size_matches[T]


def _is_npu_capture_leaf[T: AnyType]() -> Bool:
    """True if `T` is a pointer-like capture leaf consuming one FAM slot:
    an `UnsafePointer` or a (static or dynamic) `TileTensor`."""
    return (
        looks_like_static_tile_tensor[T]()
        or looks_like_tile_tensor[T]()
        or looks_like_pointer[T]()
    )


# ===-----------------------------------------------------------------------===#
# NPU Closure Serialization
# ===-----------------------------------------------------------------------===#


def _serialize_npu_reg_tuple_payload[
    elem_types: TypeList[Trait=AnyType, ...],
](
    src: UnsafePointer[UInt8, ImmutAnyOrigin],
    dest: UnsafePointer[UInt8, MutAnyOrigin],
) raises:
    """Walks a `_RegTuple`'s underlying mlir-struct payload by iterating
    the wrapped element types directly.

    `_RegTuple<*element_types>` stores its values in a single
    `_mlir_value: !kgen.struct<element_types isParamPack>` field whose
    interior `reflect.is_struct()` rejects. This helper sidesteps that
    by taking the element type list as a `TypeList` (obtained via
    `_field_types_of` on the mlir struct) and walking the bytes
    sequentially with target-aware sizing.

    Padding: contiguous `Int` / `Int32` / `ComptimeInt` elements have
    matching alignment on both targets, so cumulative-size offset
    arithmetic is correct for those cases — which covers every
    `CoordLike` element type currently used by `Layout`.

    Parameters:
        elem_types: The mlir struct's field types (one entry per
            `_RegTuple` element).

    Args:
        src: Pointer to the host bytes at the start of the
            `_RegTuple`'s mlir-struct payload.
        dest: Pointer to the device-target buffer at the corresponding
            position.
    """
    var host_off: Int = 0
    var dev_off: Int = 0
    comptime host_int_sz = size_of[Int]()
    comptime dev_int_sz = size_of[Int, target=get_gpu_target()]()
    comptime for i in range(elem_types.length):
        comptime ElemType = elem_types[i]
        comptime host_sz = size_of[ElemType]()
        comptime dev_sz = size_of[ElemType, target=get_gpu_target()]()

        # We can't directly compare `ElemType` against Mojo's `Int` because
        # `_field_types_of` on a `!kgen.struct<>` reports the underlying
        # `!pop.scalar<index>` mlir element type rather than the user-facing
        # `Int` struct that wraps it. Fall back to size-based dispatch,
        # which covers every `CoordLike` element shape `Layout` currently
        # produces: `Int` (8B → 4B), `ComptimeInt[*]` (0B → 0B), and
        # fixed-width scalars (`Scalar[DType.int32]` etc., NB → NB).
        comptime if host_sz == 0 and dev_sz == 0:
            # Static / zero-sized element — skip.
            pass
        elif host_sz == host_int_sz and dev_sz == dev_int_sz:
            # `Int`-shaped: 64-bit host → 32-bit device, range-checked.
            var v = (src + host_off).bitcast[Int]()[]
            if v != Int(Int32(v)):
                raise "TileTensor dim does not fit in Int32."
            (dest + dev_off).bitcast[Int32]()[] = Int32(v)
        elif host_sz == dev_sz:
            # Same size on both targets — copy verbatim.
            unsafe_memcpy(
                dest=(dest + dev_off).unsafe_mut_cast[True](),
                src=(src + host_off).unsafe_mut_cast[False](),
                count=host_sz,
            )
        else:
            comptime assert (
                False
            ), "unsupported `_RegTuple` element type for NPU serialization"

        host_off += host_sz
        dev_off += dev_sz


def _serialize_npu_struct[
    T: AnyType
](
    src: UnsafePointer[UInt8, ImmutAnyOrigin],
    dest: UnsafePointer[UInt8, MutAnyOrigin],
) raises:
    """Recursively serialize a struct's fields into a device-target buffer.

    Walks each field of `T` and copies it into `dest` at its Hexagon-target
    offset, applying type-specific handling:
    - Same host/device size: copied verbatim.
    - UnsafePointer: left zeroed; device wrapper patches via FAM.
    - Int: truncated to Int32 with a runtime range check.
    - `_RegTuple<...>`: dispatched to `_serialize_npu_reg_tuple_payload`,
      which walks the wrapped mlir-struct elements directly. (Generic
      recursion through `reflect.is_struct()` rejects mlir struct types.)
    - Nested struct: recurses into the struct's own fields.

    Parameters:
        T: The struct type to serialize.

    Args:
        src: Pointer to the host-side struct bytes.
        dest: Pointer to the zeroed device-target buffer to write into.
    """
    comptime r = reflect[T]
    # A `RegisterPassable` closure with no captures collapses to a non-struct
    # representation; there's nothing to walk and the device blob is empty.
    # Short-circuit instead of asserting.
    comptime if not r.is_struct():
        return
    comptime for i in range(r.field_count()):
        comptime FieldType = r.field_types()[i]
        comptime host_off = r.field_offset[index=i]()
        comptime dev_off = r.field_offset[index=i, target=get_gpu_target()]()

        comptime if size_of[FieldType, target=get_gpu_target()]() == size_of[
            FieldType
        ]():
            # Same size on host and device — copy verbatim.
            unsafe_memcpy(
                dest=(dest + dev_off).unsafe_mut_cast[True](),
                src=(src + host_off).unsafe_mut_cast[False](),
                count=size_of[FieldType](),
            )
        elif looks_like_pointer[FieldType]():
            # Reserve 4-byte pointer field; wrapper patches at runtime.
            pass
        elif FieldType == Int:
            # Truncate Int (64-bit on host) to Int32 (32-bit on device).
            var host_int = (src + host_off).bitcast[Int]()[]
            if host_int != Int(Int32(host_int)):
                raise "Captured `Int` value does not fit in `Int32`."
            (dest + dev_off).bitcast[Int32]()[] = Int32(host_int)
        elif reflect[FieldType].base_name() == "_RegTuple":
            # `_RegTuple`'s `_mlir_value` is an opaque `!kgen.struct<>`
            # that `reflect.is_struct()` rejects. The mlir struct sits at
            # offset 0 inside `_RegTuple`, and its element types are the
            # `_RegTuple`'s variadic parameters laid out sequentially —
            # extract them via `_field_types_of` and walk the bytes.
            _serialize_npu_reg_tuple_payload[
                _field_types_of[reflect[FieldType].field_types()[0]]()
            ](src + host_off, dest + dev_off)
        elif reflect[FieldType].is_struct():
            _serialize_npu_struct[FieldType](src + host_off, dest + dev_off)
        elif r.base_name() == "_PointerNicheStorage":
            # TODO(HWB-232): Handle this properly, currently only used by
            # `mo.gather`, to pass an optional pointer for CPU devices so it's
            # safe to skip here.
            pass
        else:
            comptime assert False, String(
                t"closure capture has different size on host and device; wrap"
                t" pointers as UnsafePointer or restructure the capture\nType:"
                t" {r.base_name()}, Field Type:"
                t" {reflect[FieldType].base_name()}"
            )


def _build_npu_closure_bytes[
    ClosureType: AnyType,
](closure: ClosureType) raises -> UnsafePointer[UInt8, MutAnyOrigin]:
    """Serialize a closure into a host buffer sized for the NPU device target.

    Allocates a zeroed host buffer equal to the closure's device-target
    footprint, then delegates field serialization to `_serialize_npu_struct`,
    which recurses into nested structs.

    Host and device offsets are computed independently to handle layout
    differences (4-byte pointers, Hexagon alignment) — they diverge whenever
    the captures struct has internal padding holes.

    The whole struct is serialized from offset 0: `_serialize_npu_struct`
    recurses into a compiler closure's field-0 captures sub-struct and a
    named register-passable adapter's own capture fields alike.

    The caller is responsible for freeing the returned pointer.

    Parameters:
        ClosureType: The closure type whose captures to serialize.

    Args:
        closure: The closure instance to serialize.

    Returns:
        A heap-allocated UInt8 buffer of size
        `size_of[ClosureType, target=get_gpu_target()]()` containing the
        serialized capture data.
    """
    comptime r = reflect[ClosureType]
    comptime dev_sz = size_of[ClosureType, target=get_gpu_target()]()
    var dev_closure_bytes = alloc[UInt8](dev_sz)
    for j in range(dev_sz):
        dev_closure_bytes[j] = 0

    # An empty-captures `RegisterPassable` closure can reflect as a
    # non-struct or as a struct with no fields. Nothing to walk in either
    # case — return the zeroed device blob.
    comptime if not r.is_struct() or r.field_count() < 1:
        return dev_closure_bytes.as_unsafe_any_origin()

    var closure_host_ptr = UnsafePointer(to=closure).bitcast[UInt8]()
    _serialize_npu_struct[ClosureType](
        closure_host_ptr.as_unsafe_any_origin(),
        dev_closure_bytes.as_unsafe_any_origin(),
    )

    return dev_closure_bytes.as_unsafe_any_origin()


# ===-----------------------------------------------------------------------===#
# NPU Tensor Metadata
# ===-----------------------------------------------------------------------===#
#
# When a dynamic TileTensor is passed to an NPU kernel, it occupies two FAM
# slots:
#   Slot N:   data buffer (device address of tensor data)
#   Slot N+1: metadata buffer, patched on device
#
# The metadata buffer's byte-layout matches the tensor struct on 32-bit Hexagon:
#   TileTensor:   [ptr:4][layout bytes]
#
# The ptr field is set to 0 on the host and patched on device with the
# actual data buffer address from slot N.
#
# Static TileTensors (all ComptimeInt layout dims) are a special case: their
# struct on 32-bit Hexagon is just a 4-byte pointer — identical to the data
# address already in FAM slot N.  For these tensors no metadata buffer is
# needed; they use a single FAM slot like an UnsafePointer.


# ===-----------------------------------------------------------------------===#
# NPU FAM Slot ABI Constants
# ===-----------------------------------------------------------------------===#
#
# The QAIC pointer array (SDK 1.23) is a UInt64 array: the first 8-byte word
# is the header (numPointers + reserved), followed by one 8-byte `AicJitPtr`
# word per FAM slot. Slot 0 is the grid-config buffer, slot 1 the serialized
# closure blob, and (for rank > 1 indexed closures) slot 2 the per-axis
# extents buffer. Pointer/tile captures occupy the slots after the reserved
# ones. Reads go through `pointer_array.bitcast[UInt64]()`.

comptime _NPU_FAM_HEADER_WORDS: Int = 1
"""Leading header word (numPointers + reserved) preceding slot 0, as a UInt64
word in the pointer array."""

comptime _NPU_FAM_GRID_SLOT: Int = 0
"""FAM slot holding the `NpuGridConfig` buffer."""

comptime _NPU_FAM_BLOB_SLOT: Int = 1
"""FAM slot holding the serialized closure blob."""

comptime _NPU_FAM_EXTENTS_SLOT: Int = 2
"""FAM slot holding the per-axis extents buffer (rank > 1 indexed closures)."""


def _npu_fam_word(slot: Int) -> Int:
    """Word offset of FAM `slot` in the pointer array, past the header word."""
    return slot + _NPU_FAM_HEADER_WORDS


def _npu_capture_base_slot[rank: Int = 1]() -> Int:
    """First FAM slot for pointer/tile captures.

    rank > 1 indexed closures reserve an additional slot for the extents.
    """
    return (_NPU_FAM_EXTENTS_SLOT + 1) if rank > 1 else (_NPU_FAM_BLOB_SLOT + 1)


# ===-----------------------------------------------------------------------===#
# NPU FAM Slot Helpers
# ===-----------------------------------------------------------------------===#
#
# Dynamic TileTensor arguments occupy 2 FAM slots (data + metadata).  Static
# TileTensors and all other argument types occupy 1 slot.  These compile-time
# helpers compute the FAM slot count for each kernel argument.


def npu_slot_count[T: AnyType]() -> Int:
    """Number of FAM slots required for argument type T.

    - Dynamic TileTensor (runtime layout dims): 2 slots.
    - Static TileTensor (all compile-time layout dims): 1 slot.
      The Hexagon struct is pointer-only, so the data FAM slot is sufficient.
    - Everything else (UnsafePointer, scalars): 1 slot.

    Parameters:
        T: The argument type to check.

    Returns:
        2 for dynamic TileTensor, 1 for everything else.
    """
    comptime if looks_like_static_tile_tensor[T]():
        return 1
    elif looks_like_tile_tensor[T]():
        return 2
    else:
        return 1


def npu_total_slots[
    declared_arg_types: TypeList[Trait=AnyType, ...],
]() -> Int:
    """Total number of FAM slots for all arguments.

    Sum of `npu_slot_count` for each type:
    - Dynamic TileTensor: 2 slots
    - Static TileTensor / UnsafePointer / scalar: 1 slot

    Parameters:
        declared_arg_types: The kernel's argument types.

    Returns:
        Total FAM slot count for the kernel's argument list.
    """
    var total = 0
    comptime for i in range(declared_arg_types.length):
        total += npu_slot_count[declared_arg_types[i]]()
    return total


# ===-----------------------------------------------------------------------===#
# NPU Wrapper Generator
# ===-----------------------------------------------------------------------===#
#
# Generates a QAIC-compliant wrapper around a user-friendly kernel function.
# Users write clean kernels with typed parameters (UnsafePointer, TileTensor,
# or scalars) and make_npu_wrapper automatically handles the
# (entry_config, pointer_array) -> UInt32 boilerplate.
#
# The variadic version replaces the previous 9 manual overloads (0-8 args)
# with a single function that handles any number of arguments of any type.
#
# Example with UnsafePointer args:
#     def vector_add(
#         a: UnsafePointer[Float32, MutAnyOrigin],
#         b: UnsafePointer[Float32, MutAnyOrigin],
#         result: UnsafePointer[Float32, MutAnyOrigin],
#     ):
#         result[0] = a[0] + b[0]
#
#     var kernel = ctx.compile_function[vector_add]()
#
# Example with TileTensor args:
#     def matmul(A: TileTensor[...], B: TileTensor[...], C: TileTensor[...]):
#         ...
#
#     var kernel = ctx.compile_function[matmul]()


@__name(t"__npuw_{get_linkage_name[user_def]()}")
def make_npu_wrapper[
    declared_arg_types: TypeList[Trait=AnyType, ...],
    //,
    user_def: def(* args: * declared_arg_types) thin -> None,
](
    entry_config: UnsafePointer[AicJitEntryPointConfig, MutAnyOrigin],
    pointer_array: UnsafePointer[AicJitPointerArray, MutAnyOrigin],
) -> UInt32:
    """Generate a QAIC-compliant wrapper for a variadic NPU kernel.

    Handles any combination of UnsafePointer, TileTensor, and scalar
    argument types by extracting each from the QAIC FAM pointer array
    using type-appropriate marshalling:

    - **UnsafePointer**: 1 FAM slot. Contains the 32-bit device address
      (the pointer value). Points into the FAM slot so dereferencing
      reads the UnsafePointer value.
    - **Static TileTensor**: 1 FAM slot. The Hexagon struct is pointer-only;
      the FAM slot itself holds the device address.
    - **Dynamic TileTensor**: 2 FAM slots. Slot N contains the data buffer
      device address, slot N+1 contains the metadata buffer address. The
      wrapper patches the metadata buffer's ptr_placeholder with the data
      address, producing a valid tensor struct in device memory.
    - **Scalar/struct**: 1 FAM slot. Contains a pointer to the value.

    Parameters:
        declared_arg_types: The argument types of the user kernel (inferred).
        user_def: The user's kernel function with typed parameters.

    Args:
        entry_config: QAIC entry point configuration (unused, for signature
            compliance).
        pointer_array: QAIC pointer array containing buffer addresses.

    Returns:
        0 on success, or `JIT_DEV_ERROR_FATAL` (14) if the kernel called
        `abort()` (which longjmps back here on QAIC).
    """
    _ = entry_config

    var base = pointer_array.bitcast[UInt64]()
    var fam_idx: Int = 0

    def make_elem[
        i: Int
    ](out result: Pointer[declared_arg_types[i], MutAnyOrigin]) {
        mut fam_idx, imm base
    }:
        comptime ArgType = declared_arg_types[i]
        comptime ArgPtrType = Pointer[ArgType, MutAnyOrigin]

        comptime if looks_like_static_tile_tensor[ArgType]():
            # Static TileTensor: 1 FAM slot.
            #
            # The Hexagon struct for a static TileTensor is just a 4-byte
            # pointer.  The 8-byte FAM slot holds the device address in its
            # low 32 bits (little-endian), which IS the complete struct.
            # We point this argument at the FAM slot so the kernel reads the
            # 4-byte pointer from the slot's low word, same as for
            # UnsafePointer args — no metadata buffer or patching needed.
            result = rebind[ArgPtrType](base + _npu_fam_word(fam_idx))
            fam_idx += 1
        elif looks_like_tile_tensor[ArgType]():
            # Dynamic TileTensor: 2 FAM slots
            #   Slot fam_idx:     data buffer device address
            #   Slot fam_idx + 1: metadata buffer device address
            #
            # The metadata buffer contains serialized tensor metadata
            # whose first 4 bytes (ptr_placeholder) we overwrite with
            # the data address.  After patching, the buffer is a valid
            # tensor struct on 32-bit Hexagon.
            #
            # FAM slots are 8-byte AicJitPtr words (SDK 1.23); the device
            # address is in the low 32 bits, so truncate to UInt32 for the
            # 4-byte placeholder.
            var data_addr = (base + _npu_fam_word(fam_idx))[]
            var meta_addr = (base + _npu_fam_word(fam_idx + 1))[]

            # Patch ptr_placeholder with the real data device address.
            UnsafePointer[UInt32, MutAnyOrigin](
                unsafe_from_address=Int(meta_addr)
            )[] = UInt32(data_addr)

            # Point into the (now-patched) metadata buffer.
            result = rebind[ArgPtrType](
                UnsafePointer[NoneType, MutAnyOrigin](
                    unsafe_from_address=Int(meta_addr)
                )
            )
            fam_idx += 2
        elif looks_like_pointer[ArgType]():
            # 8-byte FAM slot holding the 32-bit device address in its low
            # word.  Point into the FAM slot itself; the kernel reads the
            # 4-byte UnsafePointer value from the slot's low word.
            result = rebind[ArgPtrType](base + _npu_fam_word(fam_idx))
            fam_idx += 1
        else:
            # FAM slot → pointer to scalar value
            var addr = (base + _npu_fam_word(fam_idx))[]
            result = rebind[ArgPtrType](
                UnsafePointer[NoneType, MutAnyOrigin](
                    unsafe_from_address=Int(addr)
                )
            )
            fam_idx += 1

    with QaicUnwindGuard() as tripped:
        if tripped:
            # JIT_DEV_ERROR_FATAL: host surfaces this on `synchronize()`.
            return 14

        _call_with_dynamic_pack_pointers[user_def](make_elem)
    return 0


# ===-----------------------------------------------------------------------===#
# NPU Closure Capture Pointer Walker
# ===-----------------------------------------------------------------------===#
#
# Every device-side closure wrapper has to patch the FAM-slot addresses for
# pointer-like captures back into the device closure blob before invoking the
# closure. A capture is "pointer-like" when its leaf is an `UnsafePointer` or a
# `TileTensor` (whose first field is the data pointer). Captures can be
# arbitrarily nested through plain structs and nested RegisterPassable
# closures (e.g. `foreach`'s RegisterPassable wrapper captures a
# `ManagedTensorSlice` whose `_ptr` field is the leaf pointer).
#
# The walk has no depth cap: it stops at pointer-like leaves and bottoms out
# on the finite acyclic by-value field tree, reaching exactly the leaves
# `_serialize_npu_struct` lays out. The count/fill/patch walks have to stay
# slot-for-slot consistent with that serializer, so they recurse the same way.


trait NpuCaptureLeafVisitor(Copyable, Movable):
    """Per-leaf policy for `_walk_npu_capture_leaves`.

    The walker handles the (shared) field traversal and recursion; the
    visitor handles the per-pointer-leaf action (count, fill a FAM slot,
    or patch a device pointer).
    """

    def visit_leaf(mut self, field_off: Int):
        """Called once per pointer-like leaf, at the leaf's byte offset
        (host offset on the host, device offset on the device — set by the
        compile target)."""
        ...


def _walk_npu_capture_leaves[
    T: AnyType,
    V: NpuCaptureLeafVisitor,
](mut visitor: V, base_off: Int):
    """Walk struct `T`'s pointer-like capture leaves, invoking `visitor`.

    The single traversal shared by the FAM count, fill, and device-patch
    walks. Visits every `UnsafePointer` / `TileTensor` leaf in field order,
    descending into nested RegisterPassable closures (field-0 captures) and
    plain structs down to the genuine pointer leaves. `field_offset` is
    ambient: the same call resolves to host offsets on the host and device
    offsets on the device, so FAM slot N maps to the same leaf on both sides.

    Parameters:
        T: The struct type to walk.
        V: The leaf-visitor policy.

    Args:
        visitor: The leaf policy; mutated as leaves are visited.
        base_off: Byte offset of `T` within the enclosing buffer/value.
    """
    comptime r = reflect[T]
    comptime if not r.is_struct():
        return
    comptime for i in range(r.field_count()):
        comptime FieldType = r.field_types()[i]
        comptime rel_off = r.field_offset[index=i]()
        var field_off = base_off + rel_off

        comptime if _is_npu_capture_leaf[FieldType]():
            visitor.visit_leaf(field_off)
        elif reflect[FieldType].base_name() == "_RegTuple" or reflect[
            FieldType
        ].base_name() == "_PointerNicheStorage":
            # Skip fields with reflection issues, these don't contain leaves.
            pass
        elif reflect[FieldType].is_struct():
            # Struct, could be a regular struct or a closure.
            _walk_npu_capture_leaves[FieldType, V](visitor, field_off)
        # else: scalar / opaque — no leaf, no recursion.


@fieldwise_init
struct _NpuPointerPatcher(NpuCaptureLeafVisitor):
    """Device-side leaf visitor: write each FAM slot's 32-bit device address
    into the zeroed 4-byte field at the leaf's (device) offset."""

    var blob_base: UnsafePointer[UInt8, MutUntrackedOrigin]
    var fam_base: UnsafePointer[UInt64, MutUntrackedOrigin]
    var fam_idx: Int

    def visit_leaf(mut self, field_off: Int):
        # FAM slots are 8-byte AicJitPtr words (SDK 1.23); the blob's pointer
        # field is a 4-byte device pointer, so truncate to the low 32 bits.
        (self.blob_base + field_off).bitcast[UInt32]()[] = UInt32(
            (self.fam_base + _npu_fam_word(self.fam_idx))[]
        )
        self.fam_idx += 1


def _patch_npu_capture_pointers[
    T: AnyType,
](
    blob_base: UnsafePointer[UInt8, MutAnyOrigin],
    fam_base: UnsafePointer[UInt64, MutAnyOrigin],
    base_off: Int,
    mut fam_idx: Int,
):
    """Walk struct `T` and patch device-side closure-blob pointer fields.

    Walks every field of struct `T` starting at `base_off` bytes into the
    device closure blob. At each pointer-like leaf (`UnsafePointer` or
    `TileTensor`) the next FAM slot's 32-bit device address is written
    into the corresponding 4-byte field; `fam_idx` is advanced. Non-leaf
    structs (plain structs and nested RegisterPassable closures whose
    first field is the captures sub-struct) are descended into down to the
    genuine pointer leaves. Scalar fields are left as serialized.

    Field-order parity with the host-side count/fill visitors is
    guaranteed by the shared `_walk_npu_capture_leaves`, so FAM slot N
    corresponds to the same leaf pointer on both sides.

    Parameters:
        T: The struct type whose pointer-like fields to patch.

    Args:
        blob_base: Base pointer of the device closure blob.
        fam_base: Base pointer of the FAM `UInt64` array.
        base_off: Byte offset of `T` inside the closure blob.
        fam_idx: In/out FAM slot index. Each pointer-like leaf consumes
            one slot.
    """
    var visitor = _NpuPointerPatcher(
        blob_base.unsafe_origin_cast[MutUntrackedOrigin](),
        fam_base.unsafe_origin_cast[MutUntrackedOrigin](),
        fam_idx,
    )
    _walk_npu_capture_leaves[T, _NpuPointerPatcher](visitor, base_off)
    fam_idx = visitor.fam_idx


def make_npu_wrapper_closure[
    ClosureType: RegisterPassable & def() -> None,
](
    entry_config: UnsafePointer[AicJitEntryPointConfig, MutAnyOrigin],
    pointer_array: UnsafePointer[AicJitPointerArray, MutAnyOrigin],
) -> UInt32:
    """Generate a QAIC-compliant device-side wrapper for a zero-parameter closure kernel.

    The closure captures all runtime state it needs. FAM slot 0 holds the
    `NpuGridConfig` buffer used by the QAIC runtime to dispatch threads
    and cores; slot 1 holds the serialized closure struct (scalar
    captures embedded). Subsequent FAM slots hold device addresses for
    pointer/tensor captures, which are patched back into the closure
    before invocation.

    Type-based FAM slot assignment for captures:

    - **Scalar/struct**: embedded in the closure blob, no extra FAM slot.
    - **UnsafePointer**: 1 extra FAM slot. Contains the 32-bit device address,
      which is written directly into the closure's pointer field.
    - **Static TileTensor**: 1 extra FAM slot. Device address written into the
      closure's tensor field (struct is pointer-sized on 32-bit Hexagon).
    - **Dynamic TileTensor**: 1 extra FAM slot. Data address written directly
      into the closure's ptr field; the layout bytes are already inline in the
      closure blob.

    Parameters:
        ClosureType: A `RegisterPassable` callable with no parameters that
            captures all needed state.

    Args:
        entry_config: Unused; present for QAIC signature compliance.
        pointer_array: FAM pointer array. Slot 0 = NpuGridConfig buffer;
            slot 1 = closure blob; slots 2..N = device addresses for
            pointer/tensor captures, in the order they appear in the
            closure's capture list.

    Returns:
        Always returns 0 (success).
    """
    _ = entry_config
    # Copy the closure blob out of JIT local storage (which is not aligned to
    # the closure's required alignment) into a correctly-aligned stack buffer,
    # then patch + execute from the copy.
    comptime _blob_sz = size_of[ClosureType]()
    var _blob_aligned = stack_allocation[
        _blob_sz, UInt8, alignment=align_of[ClosureType]()
    ]()
    unsafe_memcpy(
        dest=_blob_aligned,
        src=npu_get_buffer[UInt8](pointer_array, 1),
        count=_blob_sz,
    )
    var closure_ptr = _blob_aligned.bitcast[ClosureType]()
    var base = pointer_array.bitcast[UInt64]()
    # Slots 0 and 1 are reserved for grid config and the closure blob;
    # pointer/tensor captures start at slot 2.
    var fam_idx: Int = _npu_capture_base_slot()

    _patch_npu_capture_pointers[ClosureType](
        closure_ptr.bitcast[UInt8]().as_unsafe_any_origin(), base, 0, fam_idx
    )

    with QaicUnwindGuard() as tripped:
        if tripped:
            # JIT_DEV_ERROR_FATAL: host surfaces this on `synchronize()`.
            return 14
        closure_ptr[]()
    return 0


def make_npu_wrapper_closure_with_grid_config[
    ClosureType: RegisterPassable
    & def(cfg: UnsafePointer[NpuGridConfig, MutAnyOrigin]) -> None,
](
    entry_config: UnsafePointer[AicJitEntryPointConfig, MutAnyOrigin],
    pointer_array: UnsafePointer[AicJitPointerArray, MutAnyOrigin],
) -> UInt32:
    """Device-side wrapper for a closure that takes an `NpuGridConfig*` arg.

    Mirrors `make_npu_wrapper_closure` but invokes the closure body with
    the FAM-slot-0 grid-config buffer's device address. The closure can
    then read `grid_dim`, `block_dim`, and `num_blocks` to compute its
    own work partitioning.

    FAM layout matches `make_npu_wrapper_closure`: slot 0 is the
    `NpuGridConfig` buffer, slot 1 is the closure blob, slots 2..N are
    pointer/tile capture addresses in capture-list order.

    Parameters:
        ClosureType: A `RegisterPassable` callable taking a single
            `UnsafePointer[NpuGridConfig, MutAnyOrigin]` argument.

    Args:
        entry_config: Unused; present for QAIC signature compliance.
        pointer_array: FAM pointer array (see layout above).

    Returns:
        Always returns 0 (success).
    """
    _ = entry_config
    var grid_cfg_ptr = npu_get_buffer[NpuGridConfig](pointer_array, 0)
    # Copy the closure blob out of JIT local storage (which is not aligned to
    # the closure's required alignment) into a correctly-aligned stack buffer,
    # then patch + execute from the copy.
    comptime _blob_sz = size_of[ClosureType]()
    var _blob_aligned = stack_allocation[
        _blob_sz, UInt8, alignment=align_of[ClosureType]()
    ]()
    unsafe_memcpy(
        dest=_blob_aligned,
        src=npu_get_buffer[UInt8](pointer_array, 1),
        count=_blob_sz,
    )
    var closure_ptr = _blob_aligned.bitcast[ClosureType]()
    var base = pointer_array.bitcast[UInt64]()
    var fam_idx: Int = _npu_capture_base_slot()

    _patch_npu_capture_pointers[ClosureType](
        closure_ptr.bitcast[UInt8]().as_unsafe_any_origin(), base, 0, fam_idx
    )

    with QaicUnwindGuard() as tripped:
        if tripped:
            return 14
        closure_ptr[](grid_cfg_ptr)
    return 0


def _make_npu_index_closure_wrapper[
    rank: Int,
    ClosureType: def[width: Int, rank: Int, alignment: Int = 1](
        IndexList[rank]
    ) -> None,
    simd_width: Int = 1,
](
    entry_config: UnsafePointer[AicJitEntryPointConfig, MutAnyOrigin],
    pointer_array: UnsafePointer[AicJitPointerArray, MutAnyOrigin],
) -> UInt32:
    """Device-side wrapper for a parametric indexed-closure NPU kernel.

    Mirrors `make_npu_wrapper_closure` for a parametric closure of shape
    `def[width, rank, alignment](IndexList[rank]) -> None` satisfying
    `RegisterPassable`. Reads `NpuGridConfig.grid_dim` from FAM slot 0 as the total
    element count, distributes work across NPU workers via
    `npu_work_range`, and (for `rank > 1`) de-linearizes each worker's
    flat index into an `IndexList[rank]` via the per-axis extents at
    FAM slot 2. Walks each worker's range in `simd_width`-wide chunks,
    invoking the body at `width=simd_width` for the bulk plus `width=1`
    per element for the tail. The body is responsible for using `width`
    in its `load[width=W]` / `store[width=W]` operations. Pointer/tensor
    captures are patched back into the closure's captures struct from
    their FAM slots before the loop.

    FAM layout (rank == 1):
    - Slot 0: `NpuGridConfig` (`grid_dim` carries the element count).
    - Slot 1: serialized closure blob.
    - Slots 2..N: pointer/tensor capture addresses in capture-list order.

    FAM layout (rank > 1):
    - Slot 0: `NpuGridConfig` (`grid_dim` carries the total element count).
    - Slot 1: serialized closure blob.
    - Slot 2: per-axis extents buffer (`rank` × UInt32, row-major).
    - Slots 3..N: pointer/tensor capture addresses in capture-list order.

    Parameters:
        rank: The rank of the index space.
        ClosureType: A parametric callable satisfying `RegisterPassable`
            and `def[width: Int, rank: Int, alignment: Int = 1](IndexList[rank]) -> None`.
        simd_width: Number of contiguous elements processed per body
            invocation in the bulk path. Defaults to 1 (scalar).

    Args:
        entry_config: Unused; present for QAIC signature compliance.
        pointer_array: FAM pointer array (see layout above).

    Returns:
        Always returns 0 (success).
    """
    _ = entry_config
    var grid_config = npu_get_grid_config(pointer_array)
    var size = Int(grid_config.grid_dim)
    # Copy the closure blob out of JIT local storage (which is not aligned to
    # the closure's required alignment) into a correctly-aligned stack buffer,
    # then patch + execute from the copy.
    comptime _blob_sz = size_of[ClosureType]()
    var _blob_aligned = stack_allocation[
        _blob_sz, UInt8, alignment=align_of[ClosureType]()
    ]()
    unsafe_memcpy(
        dest=_blob_aligned,
        src=npu_get_buffer[UInt8](pointer_array, 1),
        count=_blob_sz,
    )
    var closure_ptr = _blob_aligned.bitcast[ClosureType]()
    var base = pointer_array.bitcast[UInt64]()

    # Pointer/tile captures start at slot 2 when rank == 1, otherwise at
    # slot 3 (slot 2 carries the per-axis extents buffer).
    comptime fam_pointer_start: Int = _npu_capture_base_slot[rank]()
    var fam_idx: Int = fam_pointer_start

    _patch_npu_capture_pointers[ClosureType](
        closure_ptr.bitcast[UInt8]().as_unsafe_any_origin(), base, 0, fam_idx
    )

    comptime if rank == 1:
        var (start, end) = npu_work_range(size)
        var i = start
        # Alignment prefix: scalar elements until `i` reaches the next
        # simd_width-aligned boundary.  Without this, workers whose
        # `start` is not a multiple of `simd_width` (which happens
        # whenever chunk_size % simd_width != 0) would execute a bulk
        # iteration that spans two logical rows, reading the wrong gather
        # index for the overflow elements.
        while i % simd_width != 0 and i < end:
            closure_ptr[][1, rank](IndexList[rank](i))
            i += 1
        # Bulk: simd_width-wide chunks (i is now simd_width-aligned).
        while i + simd_width <= end:
            closure_ptr[][simd_width, rank](IndexList[rank](i))
            i += simd_width
        # Tail: per-element width=1.
        while i < end:
            closure_ptr[][1, rank](IndexList[rank](i))
            i += 1
    else:
        # Read per-axis extents from FAM slot 2.
        var extents_addr = (base + _npu_fam_word(_NPU_FAM_EXTENTS_SLOT))[]
        var extents_ptr = UnsafePointer[UInt32, MutAnyOrigin](
            unsafe_from_address=Int(extents_addr)
        )

        var extents = IndexList[rank]()
        comptime for d in range(rank):
            extents[d] = Int(extents_ptr[d])

        @always_inline
        def _delinearize(linear_i: Int) {imm} -> IndexList[rank]:
            var idx = IndexList[rank]()
            var rem = linear_i

            comptime for d_rev in range(rank - 1, 0, -1):
                idx[d_rev] = rem % extents[d_rev]
                rem //= extents[d_rev]
            idx[0] = rem
            return idx

        @always_inline
        def _next_row(idx: IndexList[rank]) {imm} -> IndexList[rank]:
            """`idx` advanced to the start of the following row."""
            var out = idx
            out[rank - 1] = 0

            comptime for d_rev in range(rank - 2, 0, -1):
                out[d_rev] += 1
                if out[d_rev] < extents[d_rev]:
                    return out
                out[d_rev] = 0
            out[0] += 1
            return out

        # Linear range this thread will process
        var (start, end) = npu_work_range(size)
        if start >= end:
            return 0

        var idx = _delinearize(start)
        var col_i = idx[rank - 1]
        var linear_i = start

        while linear_i < end:
            var row_end = min(extents[rank - 1], col_i + end - linear_i)
            linear_i += row_end - col_i

            while col_i + simd_width <= row_end:
                idx[rank - 1] = col_i
                closure_ptr[][simd_width, rank](idx)
                col_i += simd_width

            while col_i < row_end:
                idx[rank - 1] = col_i
                closure_ptr[][1, rank](idx)
                col_i += 1

            col_i = 0
            if linear_i < end:
                idx = _next_row(idx)

    return 0


# ===-----------------------------------------------------------------------===#
# compile_info_npu: Auto-wrapping compile_info for NPU kernels
# ===-----------------------------------------------------------------------===#

# The wrapped kernel function type used by all make_npu_wrapper overloads.
comptime _NpuWrappedKernelType = def(
    entry_config: UnsafePointer[AicJitEntryPointConfig, MutAnyOrigin],
    pointer_array: UnsafePointer[AicJitPointerArray, MutAnyOrigin],
) thin -> UInt32


@always_inline
def compile_info_npu[
    kernel: _NpuWrappedKernelType,
    /,
    *,
    emission_kind: StaticString = "object",
    link_options: StaticString = "",
]() -> CompiledFunctionInfo[_NpuWrappedKernelType, kernel, get_gpu_target()]:
    """Compile a raw QAIC kernel (no wrapping needed).

    Parameters:
        kernel: A kernel with the raw QAIC signature.
        emission_kind: Output format ("object", "asm", "llvm", "llvm-opt").
        link_options: Additional linker flags and options as a string.

    Returns:
        A CompiledFunctionInfo with the compiled ELF, function name, etc.
    """
    return compile_info[
        kernel,
        emission_kind=emission_kind,
        target=get_gpu_target(),
        link_options=link_options,
    ]()


@always_inline
def compile_info_npu[
    declared_arg_types: TypeList[Trait=AnyType, ...],
    //,
    user_def: def(* args: * declared_arg_types) thin -> None,
    /,
    *,
    emission_kind: StaticString = "object",
    link_options: StaticString = "",
]() -> CompiledFunctionInfo[
    _NpuWrappedKernelType, make_npu_wrapper[user_def], get_gpu_target()
]:
    """Compile a variadic NPU kernel, auto-wrapping with make_npu_wrapper.

    Accepts kernels with any combination of UnsafePointer, TileTensor,
    and scalar argument types. The compiler infers declared_arg_types
    from the user_def signature.

    Parameters:
        declared_arg_types: The argument types of the user kernel (inferred).
        user_def: The user's kernel function with typed parameters.
        emission_kind: Output format ("object", "asm", "llvm", "llvm-opt").
        link_options: Additional linker flags and options as a comma separated string.

    Returns:
        A CompiledFunctionInfo with the compiled ELF, function name, etc.
    """
    return compile_info_npu[
        make_npu_wrapper[user_def],
        emission_kind=emission_kind,
        link_options=link_options,
    ]()


# ===-----------------------------------------------------------------------===#
# NPU Barrier Synchronization
# ===-----------------------------------------------------------------------===#


@always_inline
def npu_barrier(
    sync_point: UnsafePointer[qshim.QShimThreadSyncStruct, MutAnyOrigin],
) -> UInt32:
    """Synchronize all HVX threads within an NPU core (like GPU barrier()).

    This function synchronizes all threads specified in the sync_point's
    thread_mask. All threads block until all participating threads reach
    this barrier.

    Args:
        sync_point: Pointer to a shared QShimThreadSyncStruct in global memory.
            Must be initialized with thread_mask before first use.

    Returns:
        0 on success, non-zero on error.

    Example:
        ```mojo
        # Global sync point (passed via buffer)
        var sync = npu_get_buffer[QShimThreadSyncStruct](pointer_array, 3)
        sync[].thread_mask = 0x1F  # Sync all 5 threads
        sync[].data = 0

        # In kernel loop:
        npu_barrier(sync)  # All threads sync here
        ```
    """
    var thread_id = UInt32(npu_thread_idx())
    return qshim.qshim_sync_threads(thread_id, sync_point)


# ===-----------------------------------------------------------------------===#
# NPU Thread Context Primitives
# ===-----------------------------------------------------------------------===#
#
# These functions use qshim runtime calls internally, eliminating
# the need for kernels to receive and pass entry_config explicitly.
#
# This enables user-friendly kernel signatures:
#     def my_kernel(input: TileTensor[...], output: TileTensor[...]):
#         var idx = npu_global_idx()  # No entry_config needed!
#         ...
#
# The qshim functions (qshim_get_thread_id, qshim_get_core_id, etc.) are
# external calls to the QAIC runtime that work without parameters - similar
# to how GPU uses LLVM intrinsics for block/thread IDs.


@always_inline
def npu_core_idx() -> Int:
    """Returns the NSP core index (equivalent to block_idx.x on GPU).

    This parameterless version uses qshim runtime calls internally,
    allowing use in user-friendly kernel signatures without entry_config.

    Returns:
        The core index (0 to num_cores-1).
    """
    return Int(qshim.qshim_get_core_id())


@always_inline
def npu_thread_idx() -> Int:
    """Returns the HVX thread index within the core (equivalent to thread_idx.x on GPU).

    This parameterless version uses qshim runtime calls internally.

    Returns:
        The thread index (0 to num_threads-1).
    """
    return Int(qshim.qshim_get_thread_id())


@always_inline
def npu_num_cores() -> Int:
    """Returns the total number of NSP cores (equivalent to grid_dim.x on GPU).

    This parameterless version uses qshim runtime calls internally.

    Returns:
        The number of cores.
    """
    return Int(qshim.qshim_get_num_cores())


@always_inline
def npu_num_threads() -> Int:
    """Returns the number of HVX threads per core (equivalent to block_dim.x on GPU).

    This parameterless version uses qshim runtime calls internally.

    Returns:
        The number of threads per core.
    """
    return Int(qshim.qshim_get_num_threads())


@always_inline
def npu_global_idx() -> Int:
    """Returns the global worker ID (equivalent to global_idx.x on GPU).

    Computes: core_id * num_threads + thread_id

    This parameterless version uses qshim runtime calls internally,
    allowing use in user-friendly kernel signatures without entry_config.

    Returns:
        The global worker index.
    """
    var core_id = Int(qshim.qshim_get_core_id())
    var thread_id = Int(qshim.qshim_get_thread_id())
    var num_threads = Int(qshim.qshim_get_num_threads())
    return core_id * num_threads + thread_id


@always_inline
def npu_num_workers() -> Int:
    """Returns the total number of workers (equivalent to grid_dim.x * block_dim.x on GPU).

    Computes: num_cores * num_threads

    This parameterless version uses qshim runtime calls internally.

    Returns:
        The total number of workers.
    """
    var num_cores = Int(qshim.qshim_get_num_cores())
    var num_threads = Int(qshim.qshim_get_num_threads())
    return num_cores * num_threads


# ===-----------------------------------------------------------------------===#
# GPU-like Aliases for NPU Programming
# ===-----------------------------------------------------------------------===#
#
# These aliases provide familiar GPU naming for developers transitioning from
# GPU to NPU programming. The underlying concepts map as follows:
#
#   GPU block ↔ NPU core (NSP)
#   GPU thread ↔ NPU thread (HVX)
#   GPU grid ↔ NPU cores array
#   GPU block_dim ↔ NPU threads_per_core


@always_inline
def npu_block_idx() -> Int:
    """Returns the core index (GPU-like alias for npu_core_idx).

    Parameterless version using qshim runtime calls.

    Returns:
        The core index (0 to num_cores-1).
    """
    return npu_core_idx()


@always_inline
def npu_grid_dim() -> Int:
    """Returns the number of cores (GPU-like alias for npu_num_cores).

    Parameterless version using qshim runtime calls.

    Returns:
        The number of cores.
    """
    return npu_num_cores()


@always_inline
def npu_block_dim() -> Int:
    """Returns threads per core (GPU-like alias for npu_num_threads).

    Parameterless version using qshim runtime calls.

    Returns:
        The number of threads per core.
    """
    return npu_num_threads()


# ===-----------------------------------------------------------------------===#
# NPU 2D Indexing and Work Range Helpers
# ===-----------------------------------------------------------------------===#
#
# These helpers provide 2D indexing similar to GPU's global_idx.x/y pattern.
# Since NPU only has a linear worker index, 2D coordinates must be computed
# from the linear index and grid dimensions.


@always_inline
def npu_global_idx_2d(width: Int) -> Tuple[Int, Int]:
    """Returns 2D global indices (x, y) for 2D grid computations.

    Computes x = worker_id % width, y = worker_id // width, similar to
    GPU's global_idx.x and global_idx.y but computed from linear worker index.

    Note: Unlike GPU where each thread typically processes one element, NPU
    workers usually need to process multiple elements. Use npu_work_range_2d()
    for typical NPU work distribution patterns.

    Args:
        width: The width of the 2D grid (number of columns).

    Returns:
        A tuple (x, y) where x is the column index and y is the row index.

    Example:
        ```mojo
        def my_kernel(input: ..., output: ...):
            var (x, y) = npu_global_idx_2d(WIDTH)
            if x < WIDTH and y < HEIGHT:
                output[y * WIDTH + x] = compute(x, y)
        ```
    """
    return divmod(npu_global_idx(), width)


@always_inline
def npu_work_range(total_size: Int) -> Tuple[Int, Int]:
    """Returns the (start, end) range of work for this worker.

    This is the standard NPU work distribution pattern where each worker
    processes a contiguous chunk of elements. The last worker handles
    any remainder.

    Args:
        total_size: Total number of elements to process.

    Returns:
        A tuple (start, end) representing the half-open range [start, end)
        that this worker should process.

    Example:
        ```mojo
        def my_kernel(input: ..., output: ...):
            var (start, end) = npu_work_range(VECTOR_SIZE)
            for i in range(start, end):
                output[i] = input[i] * 2.0
        ```
    """
    var worker_id = Int(npu_global_idx())
    var total_workers = Int(npu_num_workers())

    # Compute chunk size, rounding up to ensure all elements are covered
    var chunk_size = (total_size + total_workers - 1) // total_workers
    var start = worker_id * chunk_size
    var end = min(start + chunk_size, total_size)

    return (start, end)


@always_inline
def npu_work_range_2d(width: Int, height: Int) -> Tuple[Int, Int]:
    """Returns the (start, end) range of linear indices for 2D work distribution.

    For 2D grids, this distributes work by linearizing the 2D indices.
    Workers process contiguous chunks of the linearized space.
    Convert linear indices to 2D using: row = idx // width, col = idx % width.

    Args:
        width: The width of the 2D grid (number of columns).
        height: The height of the 2D grid (number of rows).

    Returns:
        A tuple (start, end) of linear indices to process.

    Example:
        ```mojo
        def my_kernel(input: ..., output: ...):
            var (start, end) = npu_work_range_2d(WIDTH, HEIGHT)
            for idx in range(start, end):
                var row = idx // WIDTH
                var col = idx % WIDTH
                output[row, col] = process(input[row, col])
        ```
    """
    return npu_work_range(width * height)


# ===-----------------------------------------------------------------------===#
# NPU Worker Context - Convenient struct for kernel initialization
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct NPUWorkerContext(Copyable, Movable):
    """Convenient container for NPU worker information.

    This struct captures all the commonly needed worker context at kernel
    start, reducing boilerplate code. It provides both linear and optional
    2D indexing.

    Example:
        ```mojo
        def my_kernel(
            entry_config: UnsafePointer[AicJitEntryPointConfig, MutAnyOrigin],
            pointer_array: UnsafePointer[AicJitPointerArray, MutAnyOrigin],
        ) -> UInt32:
            var ctx = NPUWorkerContext(entry_config)
            var (start, end) = ctx.work_range(VECTOR_SIZE)

            for i in range(start, end):
                # Process element i
                pass

            return 0
        ```
    """

    var worker_id: Int
    """Global worker ID (0 to total_workers-1)."""

    var total_workers: Int
    """Total number of workers (num_cores * num_threads)."""

    var core_id: Int
    """NSP core ID (0 to num_cores-1)."""

    var thread_id: Int
    """HVX thread ID within core (0 to num_threads-1)."""

    var num_cores: Int
    """Number of NSP cores."""

    var num_threads: Int
    """Number of HVX threads per core."""

    def __init__(
        out self,
        entry_config: UnsafePointer[AicJitEntryPointConfig, MutAnyOrigin],
    ):
        """Initialize worker context from entry_config.

        Args:
            entry_config: The entry_config pointer passed to the kernel.
        """
        var config = entry_config[]
        self.core_id = Int(config.core_id)
        self.thread_id = Int(config.thread_id)
        self.num_cores = Int(config.num_cores)
        self.num_threads = Int(config.num_threads)
        self.worker_id = self.core_id * self.num_threads + self.thread_id
        self.total_workers = self.num_cores * self.num_threads

    def work_range(self, total_size: Int) -> Tuple[Int, Int]:
        """Returns the (start, end) range of work for this worker.

        Args:
            total_size: Total number of elements to process.

        Returns:
            A tuple (start, end) representing the half-open range [start, end).
        """
        var chunk_size = (
            total_size + self.total_workers - 1
        ) // self.total_workers
        var start = self.worker_id * chunk_size
        var end = min(start + chunk_size, total_size)
        return (start, end)

    def work_range_2d(self, width: Int, height: Int) -> Tuple[Int, Int]:
        """Returns the (start, end) range for 2D work distribution.

        Args:
            width: The width of the 2D grid.
            height: The height of the 2D grid.

        Returns:
            A tuple (start, end) of linear indices to process.
        """
        return self.work_range(width * height)
