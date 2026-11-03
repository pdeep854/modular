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
"""Probe on-device heap allocation behavior from a Mojo NPU kernel.

Ported from the QAicHeapProbeTest.cpp / kernel_heap_probe.cpp pair in the
qaic-jit-research repo. Characterizes the shipped Hexagon libc
`malloc/free/calloc/realloc/memalign/aligned_alloc/posix_memalign/sbrk`
entry points reachable from JIT kernels.

Twelve sections are exercised, mirroring the C++ original:

1. heap_init                 — initial sbrk(0) probe (libc self-initializes
   on first malloc; calling dkw_malloc_init directly hangs this device)
2. heap_malloc_smoke         — malloc/write/read/free round-trip
3. heap_calloc_smoke         — calloc zero-init check
4. heap_realloc_smoke        — realloc preserves initial bytes
5. heap_memalign_smoke       — memalign alignment check
   heap_aligned_alloc_smoke  — aligned_alloc alignment check
   heap_posix_memalign_smoke — posix_memalign alignment + rc check
6. heap_sbrk_probe           — sbrk(0) / sbrk(+N) / sbrk(-N) characterization
7. heap_size_probe           — chunk-wise malloc-until-NULL ceiling
8. heap_many_small_loop      — 10k alloc/free cycles, drift + reuse
9. heap_write_marker/read    — intra-CB persistence (two dispatches, one CB)
10. heap_write_marker/read   — inter-CB persistence (two submissions)
11. heap_concurrent_malloc   — 5-thread concurrent malloc (threadBitmap=0x1F)

The C++ `heap_cpp_new_delete` section is NOT ported: Mojo has no
`operator new`/`delete` syntax and the Itanium C++ ABI mangled symbols
`_Znwj`/`_Znaj` are not in libc.a (they're defined in the C++ operator
source via a shim over malloc/free). The C-level allocators exercised in
sections 2-4 already cover the same malloc/free mechanics the shim would
be measuring.

All probes live in a single Mojo kernel selected by `meta_buf[0]` so that
the same .so (and its BSS globals — the persistence marker, the size
probe's slot array) is shared across dispatches. A single
`compile_info_npu` + `qaic_load_jit_function` at setup suffices.

This test is how this worktree discovered that the Mojo compiler plugin's
`-no-undefined` link line needs sys_sbrk / sys_Mtx* / sys_Tls* / sys_remove
pre-stubbed via `--defsym=0` (see `KGEN/lib/Compiler/ObjectCompiler.cpp`),
matching how qshim imports are already handled.

To run (requires QAIC hardware):
    mojo test_qaic_heap_probe.mojo
"""

from common import print_separator, print_test_header

from std.collections.string.string_span import _get_kgen_string
from std.ffi import external_call
from std.memory import UnsafePointer, alloc
from std.memory.unsafe_pointer import pointer_to_int
from std.sys import size_of
from std.sys.info import align_of
from std.testing import TestSuite, assert_equal

from max.gpu.host import DeviceContext
from std._plugin.qaic.host.target import compile_info_npu
from max._plugin.qaic.host.qaic import (
    QAicBuf,
    QAicCommandBuffer,
    QAicDescriptorIndex,
    QAicDevice,
    QAicJitFunction,
    QAicStream,
    QAIC_BUFFER_TYPE_DEV_DDR,
    QAIC_COPY_FROM_DEVICE,
    QAIC_COPY_TO_DEVICE,
    QAIC_DEV_BUF_IO_RW,
    QAIC_DEV_BUF_RESIDENT_KEEP,
    qaic_allocate_jit_buffer,
    qaic_begin_jit_command_buffer,
    qaic_copy_jit_buffer,
    qaic_create_jit_command_buffer,
    qaic_create_jit_stream,
    qaic_dispatch_jit_command_buffer,
    qaic_end_jit_command_buffer,
    qaic_execute_jit_function,
    qaic_get_jit_device,
    qaic_load_jit_function,
    qaic_push_jit_buffer_descriptor,
    qaic_release_jit_buffer,
    qaic_release_jit_command_buffer,
    qaic_release_jit_function,
    qaic_release_jit_stream,
    qaic_submit_and_wait_jit_command_buffers,
)
from std._plugin.qaic.host.qshim import qshim_get_thread_id


# ===-----------------------------------------------------------------------===#
# Probe IDs (kernel-side switch selector, stored in meta_buf[0])
# ===-----------------------------------------------------------------------===#

comptime PROBE_INIT: UInt32 = 0
comptime PROBE_MALLOC_SMOKE: UInt32 = 1
comptime PROBE_CALLOC_SMOKE: UInt32 = 2
comptime PROBE_REALLOC_SMOKE: UInt32 = 3
comptime PROBE_MEMALIGN_SMOKE: UInt32 = 4
comptime PROBE_ALIGNED_ALLOC_SMOKE: UInt32 = 5
comptime PROBE_POSIX_MEMALIGN_SMOKE: UInt32 = 6
comptime PROBE_SBRK: UInt32 = 7
comptime PROBE_SIZE_PROBE: UInt32 = 8
comptime PROBE_MANY_SMALL: UInt32 = 9
comptime PROBE_WRITE_MARKER: UInt32 = 10
comptime PROBE_READ_MARKER: UInt32 = 11
comptime PROBE_CONCURRENT_MALLOC: UInt32 = 12


# ===-----------------------------------------------------------------------===#
# Host constants
# ===-----------------------------------------------------------------------===#

comptime DEFAULT_ALIGNMENT: UInt32 = 128
comptime ALL_COMPUTE_UNITS: UInt64 = 0xFFFFFFFFFFFFFFFF
comptime SINGLE_THREAD_MASK: UInt32 = 0x01
comptime FIVE_THREAD_MASK: UInt32 = 0x1F


# ===-----------------------------------------------------------------------===#
# BSS helpers — `pop.global_alloc` wrapper for zero-initialized kernel globals.
# Matches the pattern used by `_static_alloc` in
# test_qshim_multicast_alloc_repro.mojo.
# ===-----------------------------------------------------------------------===#


@always_inline
def _bss_alloc[
    type: AnyType,
    /,
    count: Int,
    *,
    alignment: Int = align_of[type](),
    name: StringSlice[ImmStaticOrigin],
]() -> UnsafePointer[type, MutUntrackedOrigin]:
    """Zero-initialized kernel global (BSS) of `count` elements of `type`.

    The allocation persists for the lifetime of the loaded .so, so values
    written by one dispatch are visible to a later dispatch targeting the
    same `QAicDeviceExecutable`.
    """
    return {
        _mlir_value = __mlir_op.`pop.global_alloc`[
            name=_get_kgen_string[name](),
            count=count.__mlir_index__(),
            _type=UnsafePointer[type, MutUntrackedOrigin]._mlir_type,
            alignment=alignment.__mlir_index__(),
        ]()
    }


# ===-----------------------------------------------------------------------===#
# Kernel-side libc shims (resolved at device-load time by the QAIC JIT loader
# via `--defsym=sys_*=0` stubs the Mojo compiler plugin now emits).
# ===-----------------------------------------------------------------------===#


@always_inline
def _dev_malloc(
    size: UInt32,
) -> Optional[UnsafePointer[UInt8, MutUnsafeAnyOrigin]]:
    return external_call[
        "malloc", Optional[UnsafePointer[UInt8, MutUnsafeAnyOrigin]], UInt32
    ](size)


@always_inline
def _dev_free(ptr: UnsafePointer[UInt8, MutUnsafeAnyOrigin]):
    external_call["free", NoneType, UnsafePointer[UInt8, MutUnsafeAnyOrigin]](
        ptr
    )


@always_inline
def _dev_calloc(
    nmemb: UInt32, size: UInt32
) -> Optional[UnsafePointer[UInt8, MutUnsafeAnyOrigin]]:
    return external_call[
        "calloc",
        Optional[UnsafePointer[UInt8, MutUnsafeAnyOrigin]],
        UInt32,
        UInt32,
    ](nmemb, size)


@always_inline
def _dev_realloc(
    ptr: UnsafePointer[UInt8, MutUnsafeAnyOrigin], size: UInt32
) -> Optional[UnsafePointer[UInt8, MutUnsafeAnyOrigin]]:
    return external_call[
        "realloc",
        Optional[UnsafePointer[UInt8, MutUnsafeAnyOrigin]],
        UnsafePointer[UInt8, MutUnsafeAnyOrigin],
        UInt32,
    ](ptr, size)


@always_inline
def _dev_memalign(
    alignment: UInt32, size: UInt32
) -> Optional[UnsafePointer[UInt8, MutUnsafeAnyOrigin]]:
    return external_call[
        "memalign",
        Optional[UnsafePointer[UInt8, MutUnsafeAnyOrigin]],
        UInt32,
        UInt32,
    ](alignment, size)


@always_inline
def _dev_aligned_alloc(
    alignment: UInt32, size: UInt32
) -> Optional[UnsafePointer[UInt8, MutUnsafeAnyOrigin]]:
    return external_call[
        "aligned_alloc",
        Optional[UnsafePointer[UInt8, MutUnsafeAnyOrigin]],
        UInt32,
        UInt32,
    ](alignment, size)


@always_inline
def _dev_posix_memalign(
    memptr: UnsafePointer[
        Optional[UnsafePointer[UInt8, MutUnsafeAnyOrigin]], MutUnsafeAnyOrigin
    ],
    alignment: UInt32,
    size: UInt32,
) -> Int32:
    return external_call[
        "posix_memalign",
        Int32,
        UnsafePointer[
            Optional[UnsafePointer[UInt8, MutUnsafeAnyOrigin]],
            MutUnsafeAnyOrigin,
        ],
        UInt32,
        UInt32,
    ](memptr, alignment, size)


@always_inline
def _dev_sbrk(incr: Int32) -> UnsafePointer[UInt8, MutUnsafeAnyOrigin]:
    return external_call[
        "sbrk", UnsafePointer[UInt8, MutUnsafeAnyOrigin], Int32
    ](incr)


@always_inline
def _ptr_to_u32(p: UnsafePointer[UInt8, MutUnsafeAnyOrigin]) -> UInt32:
    """Truncate a 32-bit Hexagon pointer to a UInt32 for host-readable output.
    """
    return UInt32(Int(p))


@always_inline
def _opt_ptr_to_u32[
    T: AnyType,
    //,
](p: Optional[UnsafePointer[T, MutUnsafeAnyOrigin]]) -> UInt32:
    """`_ptr_to_u32` for an optionally-null pointer. Returns 0 when None."""
    return UInt32(pointer_to_int(p))


# ===-----------------------------------------------------------------------===#
# Probe bodies — one @always_inline helper per section, keyed by PROBE_*.
# All bodies write to `out` (UInt32[8]) except the concurrent probe (UInt32[32])
# and the read_marker probe (UInt32[16]); they never read `out` before writing.
# ===-----------------------------------------------------------------------===#


@always_inline
def _probe_init(output: UnsafePointer[UInt32, MutUnsafeAnyOrigin]):
    """`heap_init` — initial break probe.

    Unlike the C++ original, we don't call `dkw_malloc_init()`: libc `malloc`
    lazy-initializes on first call on this device, and explicitly invoking
    the init entry point hangs the NSP (it appears to wait on a TLS /
    syscall backend that the runtime doesn't satisfy synchronously).

    output[0] = sbrk(0) initial break
    output[1] = 0xCAFEBABE sentinel
    output[2] = sizeof(void*)
    output[3] = 0
    """
    output[0] = _ptr_to_u32(_dev_sbrk(0))
    output[1] = 0xCAFEBABE
    output[2] = UInt32(size_of[UnsafePointer[UInt8, MutUnsafeAnyOrigin]]())
    output[3] = 0


@always_inline
def _probe_malloc_smoke(output: UnsafePointer[UInt32, MutUnsafeAnyOrigin]):
    """`heap_malloc_smoke` — malloc(64B) / write / read / free round-trip.

    output[0] = ptr, output[1] = ptr & 63, output[2] = verify (1/0),
    output[3] = first mismatch idx, output[4] = mismatch observed,
    output[5] = sbrk after malloc, output[6] = sbrk after free,
    output[7] = 0xDEADBEEF sentinel
    """
    var n: UInt32 = 16  # 64 bytes of UInt32
    var opt_p = _dev_malloc(n * 4)
    output[0] = _opt_ptr_to_u32(opt_p)
    output[1] = output[0] & 63
    output[2] = 0
    output[3] = 0xFFFFFFFF
    output[4] = 0

    if opt_p:
        var p = opt_p.unsafe_value().bitcast[UInt32]()
        for i in range(Int(n)):
            p[i] = UInt32(0xDEADBEEF) ^ UInt32(i)
        var ok: UInt32 = 1
        for i in range(Int(n)):
            var v = p[i]
            if v != (UInt32(0xDEADBEEF) ^ UInt32(i)):
                ok = 0
                output[3] = UInt32(i)
                output[4] = v
                break
        output[2] = ok
        output[5] = _ptr_to_u32(_dev_sbrk(0))
        _dev_free(p.bitcast[UInt8]().as_unsafe_any_origin())
        output[6] = _ptr_to_u32(_dev_sbrk(0))
    else:
        output[5] = _ptr_to_u32(_dev_sbrk(0))
        output[6] = output[5]

    output[7] = 0xDEADBEEF


@always_inline
def _probe_calloc_smoke(output: UnsafePointer[UInt32, MutUnsafeAnyOrigin]):
    """`heap_calloc_smoke` — calloc(256, 4) zero-init check.

    output[0] = ptr, output[1] = all-zero (1/0),
    output[2] = first non-zero offset, output[3] = first non-zero byte value,
    output[4] = bytes requested, output[7] = 0xC0FFEE00 sentinel
    """
    var nmemb: UInt32 = 256
    var sz: UInt32 = 4
    var total: UInt32 = nmemb * sz

    var opt_p = _dev_calloc(nmemb, sz)
    output[0] = _opt_ptr_to_u32(opt_p)
    output[1] = 0
    output[2] = 0xFFFFFFFF
    output[3] = 0
    output[4] = total

    if opt_p:
        var p = opt_p.unsafe_value()
        var clean: UInt32 = 1
        for i in range(Int(total)):
            if p[i] != 0:
                clean = 0
                output[2] = UInt32(i)
                output[3] = UInt32(p[i])
                break
        output[1] = clean
        _dev_free(p)

    output[7] = 0xC0FFEE00


@always_inline
def _probe_realloc_smoke(output: UnsafePointer[UInt32, MutUnsafeAnyOrigin]):
    """`heap_realloc_smoke` — malloc, fill, realloc, verify preservation.

    output[0] = ptr_before, output[1] = ptr_after, output[2] = preserved (1/0),
    output[3] = first mismatch idx, output[4] = mismatch value,
    output[5] = old size, output[6] = new size, output[7] = 0xABCDEF01 sentinel
    """
    var old_n: UInt32 = 32
    var new_n: UInt32 = 128

    var opt_p = _dev_malloc(old_n * 4)
    output[0] = _opt_ptr_to_u32(opt_p)
    output[5] = old_n * 4
    output[6] = new_n * 4

    if not opt_p:
        output[7] = 0xABCDEF01
        return

    var p = opt_p.unsafe_value().bitcast[UInt32]()
    for i in range(Int(old_n)):
        p[i] = UInt32(0x10000) + UInt32(i)

    var opt_q = _dev_realloc(
        p.bitcast[UInt8]().as_unsafe_any_origin(), new_n * 4
    )
    output[1] = _opt_ptr_to_u32(opt_q)

    if not opt_q:
        output[2] = 0
        output[3] = 0xFFFFFFFF
        _dev_free(p.bitcast[UInt8]().as_unsafe_any_origin())
        output[7] = 0xABCDEF01
        return

    var q = opt_q.unsafe_value().bitcast[UInt32]()
    var ok: UInt32 = 1
    output[3] = 0xFFFFFFFF
    for i in range(Int(old_n)):
        var v = q[i]
        if v != (UInt32(0x10000) + UInt32(i)):
            ok = 0
            output[3] = UInt32(i)
            output[4] = v
            break
    output[2] = ok

    _dev_free(q.bitcast[UInt8]().as_unsafe_any_origin())
    output[7] = 0xABCDEF01


@always_inline
def _align_result(
    output: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
    opt_p: Optional[UnsafePointer[UInt8, MutUnsafeAnyOrigin]],
    alignment: UInt32,
    size: UInt32,
    extra: UInt32,
):
    """Shared writeback for the three aligned-allocator smokes."""
    output[0] = _opt_ptr_to_u32(opt_p)
    output[1] = alignment
    output[2] = size
    if opt_p:
        # 0 = aligned
        output[3] = UInt32(Int(opt_p.unsafe_value())) & (alignment - 1)
        output[4] = 1
    else:
        output[3] = 0xFFFFFFFF
        output[4] = 0
    output[5] = extra


@always_inline
def _probe_memalign_smoke(output: UnsafePointer[UInt32, MutUnsafeAnyOrigin]):
    """`heap_memalign_smoke` — memalign(64, 256) alignment check.

    sentinel = 0xAAAAAAAA
    """
    var alignment: UInt32 = 64
    var size: UInt32 = 256
    var opt_p = _dev_memalign(alignment, size)
    _align_result(output, opt_p, alignment, size, 0)
    if opt_p:
        _dev_free(opt_p.unsafe_value())
    output[7] = 0xAAAAAAAA


@always_inline
def _probe_aligned_alloc_smoke(
    output: UnsafePointer[UInt32, MutUnsafeAnyOrigin]
):
    """`heap_aligned_alloc_smoke` — aligned_alloc(128, 256) alignment check.

    sentinel = 0xBBBBBBBB
    """
    var alignment: UInt32 = 128
    var size: UInt32 = 256
    var opt_p = _dev_aligned_alloc(alignment, size)
    _align_result(output, opt_p, alignment, size, 0)
    if opt_p:
        _dev_free(opt_p.unsafe_value())
    output[7] = 0xBBBBBBBB


@always_inline
def _probe_posix_memalign_smoke(
    output: UnsafePointer[UInt32, MutUnsafeAnyOrigin]
):
    """`heap_posix_memalign_smoke` — posix_memalign(&p, 512, 1024).

    output[5] carries the posix_memalign return code (0 on success).
    sentinel = 0xCCCCCCCC
    """
    var alignment: UInt32 = 512
    var size: UInt32 = 1024
    var opt_p = Optional[UnsafePointer[UInt8, MutUnsafeAnyOrigin]]()
    var rc = _dev_posix_memalign(
        UnsafePointer(to=opt_p).as_unsafe_any_origin(), alignment, size
    )
    _align_result(output, opt_p, alignment, size, UInt32(rc))
    if opt_p:
        _dev_free(opt_p.unsafe_value())
    output[7] = 0xCCCCCCCC


@always_inline
def _probe_sbrk(output: UnsafePointer[UInt32, MutUnsafeAnyOrigin]):
    """`heap_sbrk_probe` — sbrk(0), sbrk(+4096), sbrk(-4096).

    output[0..6] = raw sbrk observations + deltas, output[7] = 0xDDDDDDDD sentinel.
    """
    var b0 = _ptr_to_u32(_dev_sbrk(0))
    var r1 = _ptr_to_u32(_dev_sbrk(4096))
    var b1 = _ptr_to_u32(_dev_sbrk(0))
    var r2 = _ptr_to_u32(_dev_sbrk(-4096))
    var b2 = _ptr_to_u32(_dev_sbrk(0))

    output[0] = b0
    output[1] = r1
    output[2] = b1
    output[3] = b1 - b0
    output[4] = r2
    output[5] = b2
    output[6] = b2 - b0
    output[7] = 0xDDDDDDDD


# Slot cap for size_probe. 8192 * 4 B = 32 KB of pointer bookkeeping.
comptime SIZE_PROBE_MAX_SLOTS: Int = 8192


@always_inline
def _probe_size(
    output: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
    meta: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    """`heap_size_probe` — malloc(chunk) loop until NULL or slot cap.

    Chunk size is read from meta[1]. All slots are freed on exit.

    output[0]=chunk, output[1]=count, output[2]=count*chunk,
    output[3]=sbrk before, output[4]=sbrk at peak, output[5]=sbrk after free,
    output[6]=0xFFFFFFFF iff slot cap hit, output[7]=0xEEEEEEEE sentinel
    """
    var chunk = meta[1]
    if chunk == 0:
        chunk = 4096

    var slots = _bss_alloc[
        UnsafePointer[UInt8, MutUnsafeAnyOrigin],
        SIZE_PROBE_MAX_SLOTS,
        name="heap_probe_size_slots",
    ]()

    var brk0 = _ptr_to_u32(_dev_sbrk(0))

    var count: UInt32 = 0
    var cap_hit: UInt32 = 0
    while Int(count) < SIZE_PROBE_MAX_SLOTS:
        var opt_p = _dev_malloc(chunk)
        if not opt_p:
            break
        slots[Int(count)] = opt_p.unsafe_value()
        count += 1
    if Int(count) == SIZE_PROBE_MAX_SLOTS:
        cap_hit = 0xFFFFFFFF

    var brk_peak = _ptr_to_u32(_dev_sbrk(0))

    for i in range(Int(count)):
        _dev_free(slots[i])
        slots[i] = UnsafePointer[UInt8, MutUnsafeAnyOrigin].unsafe_dangling()

    var brk_end = _ptr_to_u32(_dev_sbrk(0))

    output[0] = chunk
    output[1] = count
    output[2] = count * chunk
    output[3] = brk0
    output[4] = brk_peak
    output[5] = brk_end
    output[6] = cap_hit
    output[7] = 0xEEEEEEEE


@always_inline
def _probe_many_small(
    output: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
    meta: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    """`heap_many_small_loop` — N × malloc(size)+touch+free cycles.

    Reads iters from meta[1] and size from meta[2].

    output[0..2]=iters,size,failures  output[3..4]=sbrk before/after
    output[5]=drift  output[6]=same-ptr count  output[7]=0xF00DF00D sentinel
    """
    var iters = meta[1]
    var size = meta[2]
    if size == 0:
        size = 64

    var brk0 = _ptr_to_u32(_dev_sbrk(0))
    var failures: UInt32 = 0
    var same_ptr: UInt32 = 0
    var first_ptr = Optional[UnsafePointer[UInt8, MutUnsafeAnyOrigin]]()

    for i in range(Int(iters)):
        var opt_p = _dev_malloc(size)
        if not opt_p:
            failures += 1
            continue
        if i == 0:
            first_ptr = opt_p
        if opt_p == first_ptr:
            same_ptr += 1
        var p = opt_p.unsafe_value()
        p[0] = UInt8(i & 0xFF)
        _dev_free(p)

    var brk1 = _ptr_to_u32(_dev_sbrk(0))

    output[0] = iters
    output[1] = size
    output[2] = failures
    output[3] = brk0
    output[4] = brk1
    output[5] = brk1 - brk0
    output[6] = same_ptr
    output[7] = 0xF00DF00D


# BSS marker struct — survives across dispatches targeting the same loaded .so,
# so heap_write_marker and heap_read_marker can exchange the pointer.
struct HeapMarkerState(Copyable, Movable):
    """BSS persistence slot for the write_marker/read_marker pair."""

    # `ptr` is nullable: read_marker may run before write_marker has
    # successfully allocated (or at all), so the field has genuine
    # "no pointer yet" state that must be distinguishable from a real
    # allocation.
    @__allow_legacy_any_origin_fields
    var ptr: Optional[UnsafePointer[UInt32, MutUnsafeAnyOrigin]]
    var pattern: UInt32
    var size: UInt32

    def __init__(out self):
        self.ptr = {}
        self.pattern = 0
        self.size = 0


@always_inline
def _probe_write_marker(
    output: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
    meta: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    """`heap_write_marker` — malloc numWords*4, fill with pattern^i, leak ptr.

    meta[1]=numWords (clamped to 4096), meta[2]=pattern.

    output[0..2]=ptr,numWords,pattern  output[3]=sbrk_after
    output[4..6]=first three words written  output[7]=0xBABEFACE sentinel
    """
    var num_words = meta[1]
    var pattern = meta[2]
    if num_words == 0:
        num_words = 64
    if num_words > 4096:
        num_words = 4096

    var opt_p = _dev_malloc(num_words * 4)
    output[0] = _opt_ptr_to_u32(opt_p)
    output[1] = num_words
    output[2] = pattern

    var marker = _bss_alloc[HeapMarkerState, 1, name="heap_probe_marker"]()

    if opt_p:
        var p = opt_p.unsafe_value().bitcast[UInt32]()
        for i in range(Int(num_words)):
            p[i] = pattern ^ UInt32(i)
        marker[].ptr = p
        marker[].pattern = pattern
        marker[].size = num_words
        output[4] = p[0]
        output[5] = p[1]
        output[6] = p[2]
    else:
        output[4] = 0
        output[5] = 0
        output[6] = 0

    output[3] = _ptr_to_u32(_dev_sbrk(0))
    output[7] = 0xBABEFACE


@always_inline
def _probe_read_marker(output: UnsafePointer[UInt32, MutUnsafeAnyOrigin]):
    """`heap_read_marker` — read through the BSS pointer saved by write_marker.

    out (UInt32[16]):
      [0]=saved ptr, [1]=saved pattern, [2]=saved size,
      [3]=sbrk(0), [4]=match count, [5]=first mismatch idx,
      [6]=first mismatch value, [7]=0xDEFEC8ED sentinel
    """
    var marker = _bss_alloc[HeapMarkerState, 1, name="heap_probe_marker"]()

    var opt_p = marker[].ptr
    var pattern = marker[].pattern
    var n = marker[].size

    output[0] = _opt_ptr_to_u32(opt_p)
    output[1] = pattern
    output[2] = n
    output[3] = _ptr_to_u32(_dev_sbrk(0))
    output[5] = 0xFFFFFFFF
    output[6] = 0

    var matches: UInt32 = 0
    if opt_p and n < 4096:
        var p = opt_p.unsafe_value()
        for i in range(Int(n)):
            var v = p[i]
            if v == (pattern ^ UInt32(i)):
                matches += 1
            elif output[5] == 0xFFFFFFFF:
                output[5] = UInt32(i)
                output[6] = v
    output[4] = matches
    output[7] = 0xDEFEC8ED


@always_inline
def _probe_concurrent_malloc(output: UnsafePointer[UInt32, MutUnsafeAnyOrigin]):
    """`heap_concurrent_malloc` — per-thread malloc/fill/verify/free.

    Dispatched with threadBitmap=0x1F; each thread writes 4 words into
    output[tid*4 .. tid*4+3]:
      [+0] ok (1/0)  [+1] ptr  [+2] size  [+3] mismatch idx (-1 on success)
    """
    var tid = UInt32(qshim_get_thread_id())
    var base = Int(tid) * 4
    var size = UInt32(64) * (tid + 1)

    var opt_p = _dev_malloc(size)
    output[base + 0] = 0
    output[base + 1] = _opt_ptr_to_u32(opt_p)
    output[base + 2] = size
    output[base + 3] = 0xFFFFFFFF

    if not opt_p:
        return

    var p = opt_p.unsafe_value()
    var pat = UInt8((UInt32(0xA0) | tid) & 0xFF)
    for i in range(Int(size)):
        p[i] = pat

    var ok: UInt32 = 1
    for i in range(Int(size)):
        if p[i] != pat:
            ok = 0
            output[base + 3] = UInt32(i)
            break
    output[base + 0] = ok

    _dev_free(p)


# ===-----------------------------------------------------------------------===#
# Single-entry kernel: dispatcher on meta[0].
# ===-----------------------------------------------------------------------===#


def heap_probe_kernel(
    out_buf: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
    meta_buf: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    """Probe-dispatch kernel. meta_buf[0] selects the probe body."""
    var probe_id = meta_buf[0]

    if probe_id == PROBE_INIT:
        _probe_init(out_buf)
    elif probe_id == PROBE_MALLOC_SMOKE:
        _probe_malloc_smoke(out_buf)
    elif probe_id == PROBE_CALLOC_SMOKE:
        _probe_calloc_smoke(out_buf)
    elif probe_id == PROBE_REALLOC_SMOKE:
        _probe_realloc_smoke(out_buf)
    elif probe_id == PROBE_MEMALIGN_SMOKE:
        _probe_memalign_smoke(out_buf)
    elif probe_id == PROBE_ALIGNED_ALLOC_SMOKE:
        _probe_aligned_alloc_smoke(out_buf)
    elif probe_id == PROBE_POSIX_MEMALIGN_SMOKE:
        _probe_posix_memalign_smoke(out_buf)
    elif probe_id == PROBE_SBRK:
        _probe_sbrk(out_buf)
    elif probe_id == PROBE_SIZE_PROBE:
        _probe_size(out_buf, meta_buf)
    elif probe_id == PROBE_MANY_SMALL:
        _probe_many_small(out_buf, meta_buf)
    elif probe_id == PROBE_WRITE_MARKER:
        _probe_write_marker(out_buf, meta_buf)
    elif probe_id == PROBE_READ_MARKER:
        _probe_read_marker(out_buf)
    elif probe_id == PROBE_CONCURRENT_MALLOC:
        _probe_concurrent_malloc(out_buf)


# ===-----------------------------------------------------------------------===#
# Host-side helpers
# ===-----------------------------------------------------------------------===#


def _alloc_ddr(
    ctx: DeviceContext, jit_device: QAicDevice, size_bytes: UInt64
) raises -> QAicBuf:
    return qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        size_bytes,
        DEFAULT_ALIGNMENT,
        ALL_COMPUTE_UNITS,
    )


def _fill_zero_u32(host: UnsafePointer[UInt32, MutUntrackedOrigin], n: Int):
    for i in range(n):
        host[i] = 0


def _hex32(v: UInt32) -> String:
    """Best-effort hex formatter (no leading zero padding — prefix with 0x)."""
    return "0x" + hex(v)


# ===-----------------------------------------------------------------------===#
# Probe drivers — one per section. Each runs a single dispatch (or two, for
# the persistence probes) and returns True on PASS, False on FAIL. Some
# sections are characterization-only and always return True with a printed
# INFORMATIONAL label — matching the C++ source.
# ===-----------------------------------------------------------------------===#


def _dispatch_probe_simple(
    ctx: DeviceContext,
    jit_device: QAicDevice,
    jit_func: QAicJitFunction,
    probe_id: UInt32,
    out_words: Int,
    meta1: UInt32 = 0,
    meta2: UInt32 = 0,
    thread_mask: UInt32 = SINGLE_THREAD_MASK,
) raises -> UnsafePointer[UInt32, MutUntrackedOrigin]:
    """Run one dispatch of `heap_probe_kernel` and return the `out_words`-entry
    result array (caller must `.unsafe_free()` it).

    Allocates + initializes + frees fresh out/meta buffers per call. The meta
    buffer is always 4 words (probe_id, meta1, meta2, 0) regardless of how
    many the probe reads.
    """
    var out_bytes = UInt64(out_words) * 4
    var meta_words = 4
    var meta_bytes = UInt64(meta_words) * 4

    var out_buf = _alloc_ddr(ctx, jit_device, out_bytes)
    var meta_buf = _alloc_ddr(ctx, jit_device, meta_bytes)

    var zeros = alloc[UInt32](out_words)
    _fill_zero_u32(zeros, out_words)
    qaic_copy_jit_buffer(
        ctx,
        out_buf,
        zeros.bitcast[UInt8]().as_unsafe_any_origin(),
        out_bytes,
        QAIC_COPY_TO_DEVICE,
    )
    zeros.unsafe_free()

    var meta = alloc[UInt32](meta_words)
    meta[0] = probe_id
    meta[1] = meta1
    meta[2] = meta2
    meta[3] = 0
    qaic_copy_jit_buffer(
        ctx,
        meta_buf,
        meta.bitcast[UInt8]().as_unsafe_any_origin(),
        meta_bytes,
        QAIC_COPY_TO_DEVICE,
    )
    meta.unsafe_free()

    var buffers = alloc[QAicBuf](2)
    buffers[0] = out_buf
    buffers[1] = meta_buf
    qaic_execute_jit_function(
        ctx, jit_func, buffers.as_unsafe_any_origin(), 2, 1, thread_mask
    )
    buffers.unsafe_free()

    var results = alloc[UInt32](out_words)
    qaic_copy_jit_buffer(
        ctx,
        out_buf,
        results.bitcast[UInt8]().as_unsafe_any_origin(),
        out_bytes,
        QAIC_COPY_FROM_DEVICE,
    )

    qaic_release_jit_buffer(ctx, out_buf)
    qaic_release_jit_buffer(ctx, meta_buf)
    return results


def run_init(
    ctx: DeviceContext, jit_device: QAicDevice, jit_func: QAicJitFunction
) raises -> Bool:
    print()
    print("=== Section 1: heap_init ===")
    var output = _dispatch_probe_simple(
        ctx, jit_device, jit_func, PROBE_INIT, 8
    )
    print("  sbrk(0) initial:", _hex32(output[0]))
    print("  sentinel:       ", _hex32(output[1]))
    print("  sizeof(void*):  ", output[2])
    var ok = (output[1] == UInt32(0xCAFEBABE)) and (output[2] == 4)
    print("  Result:", "PASS" if ok else "FAIL")
    output.unsafe_free()
    return ok


def run_malloc_smoke(
    ctx: DeviceContext, jit_device: QAicDevice, jit_func: QAicJitFunction
) raises -> Bool:
    print()
    print("=== Section 2a: heap_malloc_smoke ===")
    var output = _dispatch_probe_simple(
        ctx, jit_device, jit_func, PROBE_MALLOC_SMOKE, 8
    )
    print("  ptr:               ", _hex32(output[0]))
    print("  ptr & 63:          ", output[1])
    print("  verify:            ", "OK" if output[2] == 1 else "FAIL")
    if output[2] == 0:
        print(
            "    first mismatch idx=",
            output[3],
            " observed=",
            _hex32(output[4]),
        )
    print("  sbrk after malloc: ", _hex32(output[5]))
    print("  sbrk after free:   ", _hex32(output[6]))
    print("  sentinel:          ", _hex32(output[7]))
    var ok = (
        (output[0] != 0)
        and (output[2] == 1)
        and (output[7] == UInt32(0xDEADBEEF))
    )
    print("  Result:", "PASS" if ok else "FAIL")
    output.unsafe_free()
    return ok


def run_calloc_smoke(
    ctx: DeviceContext, jit_device: QAicDevice, jit_func: QAicJitFunction
) raises -> Bool:
    print()
    print("=== Section 2b: heap_calloc_smoke ===")
    var output = _dispatch_probe_simple(
        ctx, jit_device, jit_func, PROBE_CALLOC_SMOKE, 8
    )
    print("  ptr:             ", _hex32(output[0]))
    print("  zero-init:       ", "OK" if output[1] == 1 else "FAIL")
    if output[1] == 0:
        print("    first nonzero byte offset=", output[2], " value=", output[3])
    print("  bytes requested: ", output[4])
    print("  sentinel:        ", _hex32(output[7]))
    var ok = (
        (output[0] != 0)
        and (output[1] == 1)
        and (output[7] == UInt32(0xC0FFEE00))
    )
    print("  Result:", "PASS" if ok else "FAIL")
    output.unsafe_free()
    return ok


def run_realloc_smoke(
    ctx: DeviceContext, jit_device: QAicDevice, jit_func: QAicJitFunction
) raises -> Bool:
    print()
    print("=== Section 2c: heap_realloc_smoke ===")
    var output = _dispatch_probe_simple(
        ctx, jit_device, jit_func, PROBE_REALLOC_SMOKE, 8
    )
    print("  ptr_before:", _hex32(output[0]))
    print("  ptr_after: ", _hex32(output[1]))
    print("  preserved: ", "OK" if output[2] == 1 else "FAIL")
    if output[2] == 0:
        print(
            "    first mismatch idx=", output[3], " value=", _hex32(output[4])
        )
    print("  old/new:   ", output[5], "/", output[6], "bytes")
    print("  sentinel:  ", _hex32(output[7]))
    var ok = (
        (output[1] != 0)
        and (output[2] == 1)
        and (output[7] == UInt32(0xABCDEF01))
    )
    print("  Result:", "PASS" if ok else "FAIL")
    output.unsafe_free()
    return ok


def _print_align_block(
    label: String,
    output: UnsafePointer[UInt32, MutUntrackedOrigin],
    sentinel: UInt32,
) -> Bool:
    print("  ", label, ":")
    print(
        "    ptr=", _hex32(output[0]), " align=", output[1], " size=", output[2]
    )
    print("    ptr %% align =", output[3], " non-null=", output[4])
    print("    sentinel=", _hex32(output[7]))
    var ok = (output[4] == 1) and (output[3] == 0) and (output[7] == sentinel)
    print("    Result:", "PASS" if ok else "FAIL")
    return ok


def run_aligned(
    ctx: DeviceContext, jit_device: QAicDevice, jit_func: QAicJitFunction
) raises -> Bool:
    print()
    print("=== Section 2d: aligned allocators ===")
    var a = _dispatch_probe_simple(
        ctx, jit_device, jit_func, PROBE_MEMALIGN_SMOKE, 8
    )
    var ok_a = _print_align_block("heap_memalign_smoke", a, UInt32(0xAAAAAAAA))
    a.unsafe_free()

    var b = _dispatch_probe_simple(
        ctx, jit_device, jit_func, PROBE_ALIGNED_ALLOC_SMOKE, 8
    )
    var ok_b = _print_align_block(
        "heap_aligned_alloc_smoke", b, UInt32(0xBBBBBBBB)
    )
    b.unsafe_free()

    var c = _dispatch_probe_simple(
        ctx, jit_device, jit_func, PROBE_POSIX_MEMALIGN_SMOKE, 8
    )
    print("    posix_memalign rc=", Int32(c[5]))
    var ok_c = _print_align_block(
        "heap_posix_memalign_smoke", c, UInt32(0xCCCCCCCC)
    )
    c.unsafe_free()

    return ok_a and ok_b and ok_c


def run_sbrk(
    ctx: DeviceContext, jit_device: QAicDevice, jit_func: QAicJitFunction
) raises -> Bool:
    print()
    print("=== Section 3: heap_sbrk_probe ===")
    var output = _dispatch_probe_simple(
        ctx, jit_device, jit_func, PROBE_SBRK, 8
    )
    print("  sbrk(0)       :     ", _hex32(output[0]))
    print("  sbrk(+4096) ret:    ", _hex32(output[1]))
    print(
        "  sbrk(0) after grow: ",
        _hex32(output[2]),
        " (delta",
        Int32(output[3]),
        ")",
    )
    print("  sbrk(-4096) ret:    ", _hex32(output[4]))
    print(
        "  sbrk(0) after shrink:",
        _hex32(output[5]),
        " (delta",
        Int32(output[6]),
        ")",
    )
    print("  sentinel:            ", _hex32(output[7]))
    # Runtime may or may not honor user-level sbrk growth. Sentinel proves the
    # entry point ran; that's what we assert.
    var ok = output[7] == UInt32(0xDDDDDDDD)
    print("  Result:", "INFORMATIONAL" if ok else "FAIL")
    output.unsafe_free()
    return ok


def _run_size_probe(
    ctx: DeviceContext,
    jit_device: QAicDevice,
    jit_func: QAicJitFunction,
    chunk: UInt32,
) raises:
    var output = _dispatch_probe_simple(
        ctx, jit_device, jit_func, PROBE_SIZE_PROBE, 8, meta1=chunk
    )
    var total: UInt64 = UInt64(output[1]) * UInt64(output[0])
    var kb = total >> 10
    var mb = total >> 20
    var cap_str = " [slot-cap hit]" if output[6] == UInt32(0xFFFFFFFF) else ""
    print(
        "  chunk=",
        output[0],
        " count=",
        output[1],
        " total=",
        total,
        "B (",
        kb,
        " KB,",
        mb,
        " MB)  sbrk(0) pre=",
        _hex32(output[3]),
        " peak=",
        _hex32(output[4]),
        " post=",
        _hex32(output[5]),
        cap_str,
    )
    output.unsafe_free()


def run_size_probe(
    ctx: DeviceContext, jit_device: QAicDevice, jit_func: QAicJitFunction
) raises -> Bool:
    print()
    print("=== Section 4: heap_size_probe (heap ceiling) ===")
    _run_size_probe(ctx, jit_device, jit_func, UInt32(4096))
    _run_size_probe(ctx, jit_device, jit_func, UInt32(65536))
    _run_size_probe(ctx, jit_device, jit_func, UInt32(1) << 20)
    _run_size_probe(ctx, jit_device, jit_func, UInt32(4) << 20)
    print("  Result: INFORMATIONAL")
    return True


def run_many_small(
    ctx: DeviceContext, jit_device: QAicDevice, jit_func: QAicJitFunction
) raises -> Bool:
    print()
    print("=== Section 5: heap_many_small_loop ===")
    var iters: UInt32 = 10000
    var size: UInt32 = 64
    var output = _dispatch_probe_simple(
        ctx,
        jit_device,
        jit_func,
        PROBE_MANY_SMALL,
        8,
        meta1=iters,
        meta2=size,
    )
    print(
        "  iters=",
        output[0],
        " size=",
        output[1],
        " failures=",
        output[2],
    )
    print(
        "  sbrk before/after:",
        _hex32(output[3]),
        "/",
        _hex32(output[4]),
        " drift=",
        Int32(output[5]),
        "B",
    )
    print(
        "  same-ptr returns:",
        output[6],
        "/",
        output[0],
        " (high = good reuse)",
    )
    print("  sentinel:", _hex32(output[7]))
    var ok = (output[2] == 0) and (output[7] == UInt32(0xF00DF00D))
    print("  Result:", "PASS" if ok else "FAIL")
    output.unsafe_free()
    return ok


def run_persistence_intra_cb(
    ctx: DeviceContext,
    jit_device: QAicDevice,
    stream: QAicStream,
    jit_func: QAicJitFunction,
) raises -> Bool:
    print()
    print("=== Section 6a: persistence intra-CB (two dispatches) ===")

    # Pair of write then read in a single CB.
    var w_out_bytes: UInt64 = 8 * 4
    var r_out_bytes: UInt64 = 16 * 4
    var meta_bytes: UInt64 = 4 * 4

    var w_out = _alloc_ddr(ctx, jit_device, w_out_bytes)
    var w_meta = _alloc_ddr(ctx, jit_device, meta_bytes)
    var r_out = _alloc_ddr(ctx, jit_device, r_out_bytes)
    var r_meta = _alloc_ddr(ctx, jit_device, meta_bytes)

    var zw = alloc[UInt32](8)
    _fill_zero_u32(zw, 8)
    qaic_copy_jit_buffer(
        ctx,
        w_out,
        zw.bitcast[UInt8]().as_unsafe_any_origin(),
        w_out_bytes,
        QAIC_COPY_TO_DEVICE,
    )
    zw.unsafe_free()
    var zr = alloc[UInt32](16)
    _fill_zero_u32(zr, 16)
    qaic_copy_jit_buffer(
        ctx,
        r_out,
        zr.bitcast[UInt8]().as_unsafe_any_origin(),
        r_out_bytes,
        QAIC_COPY_TO_DEVICE,
    )
    zr.unsafe_free()

    var w_meta_host = alloc[UInt32](4)
    w_meta_host[0] = PROBE_WRITE_MARKER
    w_meta_host[1] = 128
    w_meta_host[2] = UInt32(0xFEEDFACE)
    w_meta_host[3] = 0
    qaic_copy_jit_buffer(
        ctx,
        w_meta,
        w_meta_host.bitcast[UInt8]().as_unsafe_any_origin(),
        meta_bytes,
        QAIC_COPY_TO_DEVICE,
    )
    w_meta_host.unsafe_free()

    var r_meta_host = alloc[UInt32](4)
    r_meta_host[0] = PROBE_READ_MARKER
    r_meta_host[1] = 0
    r_meta_host[2] = 0
    r_meta_host[3] = 0
    qaic_copy_jit_buffer(
        ctx,
        r_meta,
        r_meta_host.bitcast[UInt8]().as_unsafe_any_origin(),
        meta_bytes,
        QAIC_COPY_TO_DEVICE,
    )
    r_meta_host.unsafe_free()

    var cb = qaic_create_jit_command_buffer(ctx, jit_device, 0x1)
    qaic_begin_jit_command_buffer(ctx, cb)

    var args1 = alloc[QAicDescriptorIndex](2)
    args1[0] = qaic_push_jit_buffer_descriptor(
        ctx, cb, w_out, 0, w_out_bytes, QAIC_DEV_BUF_RESIDENT_KEEP
    )
    args1[1] = qaic_push_jit_buffer_descriptor(
        ctx, cb, w_meta, 0, meta_bytes, QAIC_DEV_BUF_RESIDENT_KEEP
    )
    qaic_dispatch_jit_command_buffer(
        ctx, cb, jit_func, args1.as_unsafe_any_origin(), 2, SINGLE_THREAD_MASK
    )

    var args2 = alloc[QAicDescriptorIndex](2)
    args2[0] = qaic_push_jit_buffer_descriptor(
        ctx, cb, r_out, 0, r_out_bytes, QAIC_DEV_BUF_RESIDENT_KEEP
    )
    args2[1] = qaic_push_jit_buffer_descriptor(
        ctx, cb, r_meta, 0, meta_bytes, QAIC_DEV_BUF_RESIDENT_KEEP
    )
    qaic_dispatch_jit_command_buffer(
        ctx, cb, jit_func, args2.as_unsafe_any_origin(), 2, SINGLE_THREAD_MASK
    )

    qaic_end_jit_command_buffer(ctx, cb)

    var cb_list = alloc[QAicCommandBuffer](1)
    cb_list[0] = cb
    qaic_submit_and_wait_jit_command_buffers(
        ctx, stream, cb_list.as_unsafe_any_origin(), 1
    )
    cb_list.unsafe_free()
    args1.unsafe_free()
    args2.unsafe_free()

    var r_host = alloc[UInt32](16)
    qaic_copy_jit_buffer(
        ctx,
        r_out,
        r_host.bitcast[UInt8]().as_unsafe_any_origin(),
        r_out_bytes,
        QAIC_COPY_FROM_DEVICE,
    )

    print(
        "  [Note: per qaic-jit-research CLAUDE.md #2, the read dispatch's DDR"
    )
    print("   writes in a multi-dispatch CB can be dropped, so this sub-test")
    print("   is largely a stability check — did the dispatch hang?]")
    print(
        "  read saw ptr=",
        _hex32(r_host[0]),
        " pattern=",
        _hex32(r_host[1]),
        " n=",
        r_host[2],
        " matches=",
        r_host[4],
    )

    r_host.unsafe_free()
    qaic_release_jit_command_buffer(ctx, cb)
    qaic_release_jit_buffer(ctx, w_out)
    qaic_release_jit_buffer(ctx, w_meta)
    qaic_release_jit_buffer(ctx, r_out)
    qaic_release_jit_buffer(ctx, r_meta)

    print("  Result: INFORMATIONAL (did not hang)")
    return True


def run_persistence_inter_cb(
    ctx: DeviceContext,
    jit_device: QAicDevice,
    jit_func: QAicJitFunction,
) raises -> Bool:
    print()
    print("=== Section 6b: persistence inter-CB (two CBs) ===")

    var w_output = _dispatch_probe_simple(
        ctx,
        jit_device,
        jit_func,
        PROBE_WRITE_MARKER,
        8,
        meta1=UInt32(128),
        meta2=UInt32(0xCABBA9E5),
    )
    print(
        "  CB1 write: ptr=",
        _hex32(w_output[0]),
        " n=",
        w_output[1],
        " pattern=",
        _hex32(w_output[2]),
        " sbrk_after=",
        _hex32(w_output[3]),
    )
    print(
        "  first 3 words:",
        _hex32(w_output[4]),
        ",",
        _hex32(w_output[5]),
        ",",
        _hex32(w_output[6]),
    )

    var r_output = _dispatch_probe_simple(
        ctx, jit_device, jit_func, PROBE_READ_MARKER, 16
    )
    print(
        "  CB2 read: saved_ptr=",
        _hex32(r_output[0]),
        " pattern=",
        _hex32(r_output[1]),
        " n=",
        r_output[2],
        " sbrk_now=",
        _hex32(r_output[3]),
    )
    if r_output[5] != UInt32(0xFFFFFFFF):
        print(
            "  matches=",
            r_output[4],
            "/",
            r_output[2],
            " (first mismatch idx=",
            r_output[5],
            " val=",
            _hex32(r_output[6]),
            ")",
        )
    else:
        print("  matches=", r_output[4], "/", r_output[2])

    var persists = (
        (r_output[0] != 0)
        and (r_output[2] != 0)
        and (r_output[4] == r_output[2])
    )
    var ptr_gone = r_output[0] == 0
    if persists:
        print("  Heap state (BSS marker + malloc'd buffer) PERSISTS across CBs")
    elif ptr_gone:
        print("  BSS state reset between CBs (pointer is 0)")
    else:
        print("  Pointer preserved but contents gone / garbage")
    print("  Result: INFORMATIONAL")
    w_output.unsafe_free()
    r_output.unsafe_free()
    return True


def run_concurrent(
    ctx: DeviceContext, jit_device: QAicDevice, jit_func: QAicJitFunction
) raises -> Bool:
    print()
    print("=== Section 7: heap_concurrent_malloc (5 threads) ===")
    var output = _dispatch_probe_simple(
        ctx,
        jit_device,
        jit_func,
        PROBE_CONCURRENT_MALLOC,
        32,
        thread_mask=FIVE_THREAD_MASK,
    )
    var all_ok = True
    for t in range(5):
        var base = t * 4
        var ok = output[base + 0]
        var p = output[base + 1]
        var sz = output[base + 2]
        var mis = output[base + 3]
        var mis_str: String = "none" if mis == UInt32(0xFFFFFFFF) else String(
            mis
        )
        print(
            "  thread",
            t,
            ": ok=",
            ok,
            " ptr=",
            _hex32(p),
            " size=",
            sz,
            " mismatch=",
            mis_str,
        )
        if ok != 1 or p == 0:
            all_ok = False
    print("  Result:", "PASS" if all_ok else "FAIL")
    output.unsafe_free()
    return all_ok


# ===-----------------------------------------------------------------------===#
# Main
# ===-----------------------------------------------------------------------===#


def test_qaic_heap_probe() raises:
    print_test_header("QAIC Heap Probe Test (Mojo port)")
    print()

    print("Step 1: Compiling heap_probe_kernel to Hexagon ELF...")
    var elf = compile_info_npu[heap_probe_kernel]()
    print("  ELF size:", elf.asm.byte_length(), "bytes")
    print("  Function name:", elf.function_name)
    print()

    print("Step 2: Creating QAIC DeviceContext...")
    var ctx = DeviceContext(device_id=0, api="qaic")
    var jit_device = qaic_get_jit_device(ctx)
    var stream = qaic_create_jit_stream(ctx, jit_device)
    print("  Context + JIT device + stream ready")
    print()

    print("Step 3: Loading JIT function...")
    var jit_func = qaic_load_jit_function(
        ctx,
        elf.function_name,
        elf.asm.unsafe_ptr().bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(elf.asm.byte_length()),
    )
    print("  JIT function loaded")
    print()

    var passed = 0
    var failed = 0
    var total = 0

    @__parameter
    def record(r: Bool):
        total += 1
        if r:
            passed += 1
        else:
            failed += 1

    record(run_init(ctx, jit_device, jit_func))
    record(run_malloc_smoke(ctx, jit_device, jit_func))
    record(run_calloc_smoke(ctx, jit_device, jit_func))
    record(run_realloc_smoke(ctx, jit_device, jit_func))
    record(run_aligned(ctx, jit_device, jit_func))
    record(run_sbrk(ctx, jit_device, jit_func))
    record(run_size_probe(ctx, jit_device, jit_func))
    record(run_many_small(ctx, jit_device, jit_func))
    record(run_persistence_intra_cb(ctx, jit_device, stream, jit_func))
    record(run_persistence_inter_cb(ctx, jit_device, jit_func))
    record(run_concurrent(ctx, jit_device, jit_func))

    qaic_release_jit_function(jit_func)
    qaic_release_jit_stream(ctx, stream)

    print()
    print_separator()
    print("Summary:", passed, "passed,", failed, "failed, out of", total)
    print_separator()

    assert_equal(failed, 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
