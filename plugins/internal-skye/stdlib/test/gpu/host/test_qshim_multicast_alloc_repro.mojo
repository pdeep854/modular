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
"""Reproducer for external-qualcomm issue #26: qshim multicast dst_offsets
allocator behavior.

Mirrors the single working case from `test_qshim_multicast_success.mojo`
(scalar-pointer dst_offsets contract, NSP0 -> NSP1), with four *minimal
variants* of the `dst_offsets` pointer array:

1. `inline_array`  — `Array` on the kernel stack (the pattern used by
   `test_qshim_multicast_success.mojo`).
2. `global_alloc`  — `pop.global_alloc` (the `MulticastOps._multicast`
   helper / `static_alloc` workaround in external-qualcomm's `utils.mojo`).
3. `heap_alloc`    — `std.memory.alloc(...)` called from device code
   (the pattern from the `qshim_multicast_xfer` docstring example, and
   from the GH #26 report).
4. `libc_malloc`   — `external_call["malloc", ...]` directly from device
   code, bypassing Mojo's `alloc` path.

Findings (demonstrated by this test):

* All four variants successfully multicast 256 bytes from NSP0 to NSP1.
  The `dst_offsets` array can live on the kernel stack, in a
  `.data`-section global, on the libc-managed heap via direct `malloc`,
  or on the libc-managed heap via Mojo's `std.memory.alloc`.

* Variant 4 (direct libc `malloc`) works because the Mojo compiler
  plugin's hexagon-link invocation uses
  `--unresolved-symbols=ignore-in-object-files`, leaving libc's
  `sys_sbrk` / `sys_Mtx*` / `sys_Tls*` imports as dynamic-symbol
  references that the QAIC device JIT loader binds at kernel-load time
  (matching how the shipped `libQAicV68Ops.so` arranges its own imports).
  See `test_qaic_heap_probe.mojo` for a broader characterization of
  on-device heap behavior.

* Variant 3 (`std.memory.alloc`) works because `pop.aligned_alloc`
  (the op `std.memory.alloc` lowers to) is now routed to libc `memalign`
  on `isHexagonTriple` targets and `pop.aligned_free` is routed to libc
  `free`. See `ConvertPOPAlignedAlloc` / `ConvertPOPAlignedFree` in
  `KGEN/lib/KGENToLLVM/LowerPOPToLLVM.cpp`.
"""

from common import print_test_header

from std.collections.string.string_span import _get_kgen_string
from std.ffi import external_call
from std.sys import size_of
from std.sys.info import align_of
from std.sys._libc import free
from std.memory import UnsafePointer, alloc
from std.testing import TestSuite, assert_true

from max.gpu.host import DeviceContext
from std._plugin.qaic.host.target import compile_info_npu
from max._plugin.qaic.host.qaic import (
    QAicBuf,
    QAicBufferDescriptor,
    QAicDevice,
    QAicJitFunction,
    QAIC_BUFFER_TYPE_DEV_DDR,
    QAIC_COPY_FROM_DEVICE,
    QAIC_COPY_TO_DEVICE,
    QAIC_DEV_BUF_ADDR_MODE_HSV32,
    QAIC_DEV_BUF_AUTO_MAP,
    QAIC_DEV_BUF_IO_RW,
    QAIC_DEV_BUF_RESIDENT_KEEP,
    qaic_allocate_jit_buffer,
    qaic_copy_jit_buffer,
    qaic_execute_jit_function,
    qaic_get_jit_device,
    qaic_load_jit_function,
    qaic_release_jit_buffer,
    qaic_release_jit_function,
)
from std._plugin.qaic.host.qshim import (
    qshim_get_base_vtcm_addr,
    qshim_get_core_id,
    qshim_get_thread_id,
    qshim_multicast_xfer,
    qshim_sync_nsps,
    qshim_wait_on_multicast,
)


# ===-----------------------------------------------------------------------===#
# Shared config (matches the simplest `test_qshim_multicast_success` case:
# scalar-pointer dst_offsets contract, NSP0 -> NSP1, 256-byte transfer).
# ===-----------------------------------------------------------------------===#

comptime MAX_NSPS: Int = 16
comptime UINT32_BYTES: Int = 4
comptime ALIGNMENT: UInt32 = 128
comptime ALL_COMPUTE_UNITS: UInt64 = 0xFFFFFFFFFFFFFFFF
comptime SINGLE_THREAD_MASK: UInt32 = 0x01
comptime LAYOUT_GAP_BYTES: Int = 0x1000

# participant_mask = NSP0 + NSP1, source = NSP0, dest_mask = NSP1.
comptime PARTICIPANT_MASK: UInt32 = 0x3
comptime SOURCE_NSP: Int = 0
comptime DEST_MASK: UInt32 = 0x2
comptime SYNCPOINT: UInt32 = 0
comptime TRANSFER_SIZE: Int = 256
comptime SEED_BASE: UInt32 = 0x13579BDF

# Per-NSP VTCM layout for this fixed scenario.
comptime CHUNK: Int = ((TRANSFER_SIZE + 63) // 64) * 64
comptime SRC_OFFSET: Int = LAYOUT_GAP_BYTES
comptime DST_BASE: Int = (
    (SRC_OFFSET + CHUNK + LAYOUT_GAP_BYTES + 63) // 64
) * 64

# The kernel only cares about four flags in the result buffer.
comptime RESULT_WORDS: Int = 4
comptime RESULT_MAGIC_IDX: Int = 0
comptime RESULT_XFER_STATUS_IDX: Int = 1
comptime RESULT_WAIT_STATUS_IDX: Int = 2
comptime RESULT_BYTES_MATCHED_IDX: Int = 3

comptime RESULT_BYTES: Int = RESULT_WORDS * UINT32_BYTES
comptime RESULT_BUFFER_WORDS: Int = MAX_NSPS * RESULT_WORDS
comptime RESULT_BUFFER_BYTES: Int = RESULT_BUFFER_WORDS * UINT32_BYTES

comptime RESULT_MAGIC: UInt32 = 0x4D435354  # "MCST"
comptime RESULT_NOT_RUN: UInt32 = 0xFFFFFFFF


# ===-----------------------------------------------------------------------===#
# Shared helpers used from device code.
# ===-----------------------------------------------------------------------===#


@always_inline
def pattern_byte(index: UInt32) -> UInt8:
    var mix = SEED_BASE + index * 0x9E3779B9
    mix = mix ^ (mix >> 16)
    return UInt8(mix & 0xFF)


@always_inline
def run_multicast_body(
    result_words: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
    dst_ptrs: UnsafePointer[
        UnsafePointer[UInt32, MutUnsafeAnyOrigin], MutUnsafeAnyOrigin
    ],
):
    """Body shared by all four kernel variants.

    Fills the VTCM source with a deterministic pattern (on NSP0), runs the
    multicast via the provided `dst_ptrs` array, waits on NSP1, and reports
    (magic, xfer_status, wait_status, bytes_matched) into `result_words`.
    """
    var core_id = Int(qshim_get_core_id())
    var thread_id = UInt32(qshim_get_thread_id())

    for i in range(RESULT_WORDS):
        result_words[i] = RESULT_NOT_RUN
    result_words[RESULT_MAGIC_IDX] = RESULT_MAGIC

    var vtcm_base = qshim_get_base_vtcm_addr()
    var src = vtcm_base + SRC_OFFSET
    var dst = vtcm_base + DST_BASE

    for i in range(TRANSFER_SIZE):
        dst[i] = UInt8(0xA5)

    if core_id == SOURCE_NSP:
        for i in range(TRANSFER_SIZE):
            src[i] = pattern_byte(UInt32(i))

    _ = qshim_sync_nsps(PARTICIPANT_MASK)

    if core_id == SOURCE_NSP:
        result_words[RESULT_XFER_STATUS_IDX] = qshim_multicast_xfer(
            thread_id,
            src,
            dst_ptrs,
            DEST_MASK,
            SYNCPOINT,
            UInt32(TRANSFER_SIZE),
        )

    _ = qshim_sync_nsps(PARTICIPANT_MASK)

    if core_id == 1:
        result_words[RESULT_WAIT_STATUS_IDX] = qshim_wait_on_multicast(
            SYNCPOINT
        )

    _ = qshim_sync_nsps(PARTICIPANT_MASK)

    var bytes_matched: UInt32 = 0
    if core_id == 1:
        for i in range(TRANSFER_SIZE):
            if dst[i] == pattern_byte(UInt32(i)):
                bytes_matched += 1
    result_words[RESULT_BYTES_MATCHED_IDX] = bytes_matched


@always_inline
def _static_alloc[
    type: AnyType,
    /,
    count: Int,
    *,
    alignment: Int = align_of[type](),
    name: StringSlice[ImmStaticOrigin],
]() -> UnsafePointer[type, MutUntrackedOrigin]:
    """Mirror of `static_alloc` from external-qualcomm's utils.mojo."""
    return {
        _mlir_value = __mlir_op.`pop.global_alloc`[
            name=_get_kgen_string[name](),
            count=count.__mlir_index__(),
            _type=UnsafePointer[type, MutUnsafeAnyOrigin]._mlir_type,
            alignment=alignment.__mlir_index__(),
        ]()
    }


# ===-----------------------------------------------------------------------===#
# Kernel variants
# ===-----------------------------------------------------------------------===#


def kernel_inline_array(
    result_words: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    var dst_offsets = Array[UInt32, MAX_NSPS](uninitialized=True)
    var dst_ptrs = Array[UnsafePointer[UInt32, MutUnsafeAnyOrigin], MAX_NSPS](
        uninitialized=True
    )
    for i in range(MAX_NSPS):
        dst_offsets[i] = UInt32(DST_BASE)
        dst_ptrs[i] = UnsafePointer[UInt32, MutUnsafeAnyOrigin](
            unsafe_from_address=Int(UnsafePointer(to=dst_offsets[i]))
        )

    run_multicast_body(
        result_words, dst_ptrs.unsafe_ptr().as_unsafe_any_origin()
    )


def kernel_global_alloc(
    result_words: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    var dst_offset_value: UInt32 = UInt32(DST_BASE)
    var dst_ptrs = _static_alloc[
        UnsafePointer[UInt32, MutUnsafeAnyOrigin],
        MAX_NSPS,
        name="mc_repro_dst_ptrs",
    ]()
    for i in range(MAX_NSPS):
        dst_ptrs[i] = UnsafePointer(to=dst_offset_value).as_unsafe_any_origin()

    run_multicast_body(result_words, dst_ptrs.as_unsafe_any_origin())


def kernel_heap_alloc(
    result_words: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    var dst_offset_value: UInt32 = UInt32(DST_BASE)
    var dst_ptrs = alloc[UnsafePointer[UInt32, MutUnsafeAnyOrigin]](MAX_NSPS)
    for i in range(MAX_NSPS):
        dst_ptrs[i] = UnsafePointer(to=dst_offset_value).as_unsafe_any_origin()

    run_multicast_body(result_words, dst_ptrs.as_unsafe_any_origin())
    # No `.unsafe_free()`: mirrors the leak in the `qshim_multicast_xfer` docstring example.


def kernel_libc_malloc(
    result_words: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    var dst_offset_value: UInt32 = UInt32(DST_BASE)
    var n_bytes = UInt32(
        MAX_NSPS * size_of[UnsafePointer[UInt32, MutUnsafeAnyOrigin]]()
    )
    var raw = external_call[
        "malloc", UnsafePointer[UInt8, MutUnsafeAnyOrigin], UInt32
    ](n_bytes)
    var dst_ptrs = raw.bitcast[UnsafePointer[UInt32, MutUnsafeAnyOrigin]]()
    for i in range(MAX_NSPS):
        dst_ptrs[i] = UnsafePointer(to=dst_offset_value).as_unsafe_any_origin()

    run_multicast_body(result_words, dst_ptrs)

    free(raw.bitcast[NoneType]())


# ===-----------------------------------------------------------------------===#
# Host-side driver
# ===-----------------------------------------------------------------------===#


def allocate_ddr_buffer(
    ctx: DeviceContext, jit_device: QAicDevice, size_bytes: Int
) raises -> QAicBuf:
    return qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        UInt64(size_bytes),
        ALIGNMENT,
        ALL_COMPUTE_UNITS,
    )


def build_descriptors(
    num_cores: Int,
    result_buf: QAicBuf,
) -> UnsafePointer[QAicBufferDescriptor, MutUntrackedOrigin]:
    var descs = alloc[QAicBufferDescriptor](num_cores)
    for nsp in range(num_cores):
        descs[nsp] = QAicBufferDescriptor(
            result_buf,
            UInt64(nsp * RESULT_BYTES),
            UInt64(RESULT_BYTES),
            QAIC_DEV_BUF_RESIDENT_KEEP,
            QAIC_DEV_BUF_AUTO_MAP,
            QAIC_DEV_BUF_ADDR_MODE_HSV32,
        )
    return descs


def verify_results(
    title: String,
    ctx: DeviceContext,
    result_buf: QAicBuf,
    num_cores: Int,
) raises -> Bool:
    var results = alloc[UInt32](RESULT_BUFFER_WORDS)
    qaic_copy_jit_buffer(
        ctx,
        result_buf,
        results.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(RESULT_BUFFER_BYTES),
        QAIC_COPY_FROM_DEVICE,
        0,
    )
    var ok = True
    for nsp in range(num_cores):
        var base = nsp * RESULT_WORDS
        print(
            "    NSP",
            nsp,
            " magic=",
            hex(results[base + RESULT_MAGIC_IDX]),
            " xfer=",
            results[base + RESULT_XFER_STATUS_IDX],
            " wait=",
            results[base + RESULT_WAIT_STATUS_IDX],
            " bytes=",
            results[base + RESULT_BYTES_MATCHED_IDX],
        )
        if results[base + RESULT_MAGIC_IDX] != RESULT_MAGIC:
            ok = False
        if nsp == SOURCE_NSP:
            if results[base + RESULT_XFER_STATUS_IDX] != 0:
                ok = False
        if nsp == 1:
            if results[base + RESULT_WAIT_STATUS_IDX] != 0:
                ok = False
            if results[base + RESULT_BYTES_MATCHED_IDX] != UInt32(
                TRANSFER_SIZE
            ):
                ok = False
    results.unsafe_free()
    print("    Result:", "PASS" if ok else "FAIL", "(", title, ")")
    return ok


# ===-----------------------------------------------------------------------===#
# Test cases — one per variant, each directly references its kernel.
# ===-----------------------------------------------------------------------===#


def test_inline_array_variant() raises:
    print_test_header("repro: Array dst_offsets (stack)")
    var ctx = DeviceContext(device_id=0, api="qaic")
    var device = qaic_get_jit_device(ctx)

    var elf = compile_info_npu[kernel_inline_array]()
    var func = qaic_load_jit_function(
        ctx,
        elf.function_name,
        elf.asm.unsafe_ptr().bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(elf.asm.byte_length()),
    )

    var num_cores = 2
    var result_buf = allocate_ddr_buffer(ctx, device, RESULT_BUFFER_BYTES)
    var result_init = alloc[UInt32](RESULT_BUFFER_WORDS)
    for i in range(RESULT_BUFFER_WORDS):
        result_init[i] = RESULT_NOT_RUN
    qaic_copy_jit_buffer(
        ctx,
        result_buf,
        result_init.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(RESULT_BUFFER_BYTES),
        QAIC_COPY_TO_DEVICE,
        0,
    )
    result_init.unsafe_free()

    var descs = build_descriptors(num_cores, result_buf)
    qaic_execute_jit_function(
        ctx,
        func,
        descs.as_unsafe_any_origin(),
        1,
        UInt32(num_cores),
        SINGLE_THREAD_MASK,
    )
    descs.unsafe_free()

    assert_true(verify_results("Array", ctx, result_buf, num_cores))
    qaic_release_jit_buffer(ctx, result_buf)
    qaic_release_jit_function(func)


def test_global_alloc_variant() raises:
    print_test_header("repro: pop.global_alloc dst_offsets (static)")
    var ctx = DeviceContext(device_id=0, api="qaic")
    var device = qaic_get_jit_device(ctx)

    var elf = compile_info_npu[kernel_global_alloc]()
    var func = qaic_load_jit_function(
        ctx,
        elf.function_name,
        elf.asm.unsafe_ptr().bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(elf.asm.byte_length()),
    )

    var num_cores = 2
    var result_buf = allocate_ddr_buffer(ctx, device, RESULT_BUFFER_BYTES)
    var result_init = alloc[UInt32](RESULT_BUFFER_WORDS)
    for i in range(RESULT_BUFFER_WORDS):
        result_init[i] = RESULT_NOT_RUN
    qaic_copy_jit_buffer(
        ctx,
        result_buf,
        result_init.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(RESULT_BUFFER_BYTES),
        QAIC_COPY_TO_DEVICE,
        0,
    )
    result_init.unsafe_free()

    var descs = build_descriptors(num_cores, result_buf)
    qaic_execute_jit_function(
        ctx,
        func,
        descs.as_unsafe_any_origin(),
        1,
        UInt32(num_cores),
        SINGLE_THREAD_MASK,
    )
    descs.unsafe_free()

    assert_true(verify_results("pop.global_alloc", ctx, result_buf, num_cores))
    qaic_release_jit_buffer(ctx, result_buf)
    qaic_release_jit_function(func)


def test_heap_alloc_variant() raises:
    print_test_header("repro: std.memory.alloc dst_offsets (device heap)")
    var ctx = DeviceContext(device_id=0, api="qaic")
    var device = qaic_get_jit_device(ctx)

    var elf = compile_info_npu[kernel_heap_alloc]()
    var func = qaic_load_jit_function(
        ctx,
        elf.function_name,
        elf.asm.unsafe_ptr().bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(elf.asm.byte_length()),
    )

    var num_cores = 2
    var result_buf = allocate_ddr_buffer(ctx, device, RESULT_BUFFER_BYTES)
    var result_init = alloc[UInt32](RESULT_BUFFER_WORDS)
    for i in range(RESULT_BUFFER_WORDS):
        result_init[i] = RESULT_NOT_RUN
    qaic_copy_jit_buffer(
        ctx,
        result_buf,
        result_init.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(RESULT_BUFFER_BYTES),
        QAIC_COPY_TO_DEVICE,
        0,
    )
    result_init.unsafe_free()

    var descs = build_descriptors(num_cores, result_buf)
    qaic_execute_jit_function(
        ctx,
        func,
        descs.as_unsafe_any_origin(),
        1,
        UInt32(num_cores),
        SINGLE_THREAD_MASK,
    )
    descs.unsafe_free()

    assert_true(verify_results("std.memory.alloc", ctx, result_buf, num_cores))
    qaic_release_jit_buffer(ctx, result_buf)
    qaic_release_jit_function(func)


def test_libc_malloc_variant() raises:
    print_test_header("repro: libc malloc dst_offsets (device heap, direct)")
    var ctx = DeviceContext(device_id=0, api="qaic")
    var device = qaic_get_jit_device(ctx)

    var elf = compile_info_npu[kernel_libc_malloc]()
    var func = qaic_load_jit_function(
        ctx,
        elf.function_name,
        elf.asm.unsafe_ptr().bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(elf.asm.byte_length()),
    )

    var num_cores = 2
    var result_buf = allocate_ddr_buffer(ctx, device, RESULT_BUFFER_BYTES)
    var result_init = alloc[UInt32](RESULT_BUFFER_WORDS)
    for i in range(RESULT_BUFFER_WORDS):
        result_init[i] = RESULT_NOT_RUN
    qaic_copy_jit_buffer(
        ctx,
        result_buf,
        result_init.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(RESULT_BUFFER_BYTES),
        QAIC_COPY_TO_DEVICE,
        0,
    )
    result_init.unsafe_free()

    var descs = build_descriptors(num_cores, result_buf)
    qaic_execute_jit_function(
        ctx,
        func,
        descs.as_unsafe_any_origin(),
        1,
        UInt32(num_cores),
        SINGLE_THREAD_MASK,
    )
    descs.unsafe_free()

    assert_true(verify_results("libc malloc", ctx, result_buf, num_cores))
    qaic_release_jit_buffer(ctx, result_buf)
    qaic_release_jit_function(func)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
