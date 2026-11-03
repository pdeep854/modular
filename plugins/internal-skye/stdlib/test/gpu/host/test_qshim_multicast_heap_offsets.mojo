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
"""Regression test for `modular/external-qualcomm#26`.

GH#26 reported that `qshim_multicast_xfer` hung (with "double free" log
entries) when the `dst_offsets` array was allocated with Mojo's `alloc()`,
per the docstring example of `qshim_multicast_xfer`. The reporter's
workaround was `pop.global_alloc` (a `static_alloc[...]` helper, mirrored
in `external-qualcomm/mojo/examples/qualcomm-kernels/matmul/utils.mojo`'s
`MulticastOps._multicast`).

Two root causes — both now fixed in the plugin repo:

1. **Host-side double-free on error propagation.** Commit 46b42ed
   (`[runtime] Fix double-free in QAIC plugin raiseError (#173)`)
   eliminated the "double free" message + hang the reporter saw: when
   `qshim_multicast_xfer` returned an error, the plugin previously
   double-freed the error string at thread exit, crashing the host.

2. **NSP-side link failure for any libc heap call.** The Mojo compiler
   plugin's hexagon-link invocation used `-no-undefined`, which rejected
   libc.a's `sys_sbrk` / `sys_Mtx*` / `sys_Tls*` imports at link time.
   Today's change swaps it for `--unresolved-symbols=ignore-in-object-files`
   (see `KGEN/lib/Compiler/ObjectCompiler.cpp`), letting the QAIC device
   JIT loader bind those syscalls at kernel-load time — the same way the
   shipped `libQAicV68Ops.so` does. `external_call["malloc", ...]` from
   Mojo device code now works end-to-end.

This test exercises the exact external-qualcomm `MulticastOps._multicast`
call pattern with **heap-allocated offsets** (via `external_call["malloc",
...]`, functionally equivalent to what the user wanted to write as
`alloc[UnsafePointer[UInt32, MutUnsafeAnyOrigin]](num_nsps)`). It runs a
one-to-many multicast NSP0 -> NSP1..3 and asserts that:

- `qshim_multicast_xfer` returns 0 (no double-free, no hang).
- `qshim_wait_on_multicast` returns 0 on every destination.
- Bytes arrive unchanged at each destination's VTCM.

For additional coverage of allocator alternatives (stack `Array`,
`pop.global_alloc`, the broken `std.memory.alloc` variant 3, and the
`external_call["malloc"]` path tested here) see
`test_qshim_multicast_alloc_repro.mojo`.

Outstanding (tracked separately): `std.memory.alloc` in NSP kernels
still fails to link because it lowers to `KGEN_CompilerRT_AlignedAlloc`
which isn't defined on Hexagon — that's a Mojo lowering issue, not a
qshim issue. Users who want a `new`-style heap allocation from a Mojo
kernel today should call `external_call["malloc", ...]` directly, as
this test does.

To run (requires QAIC hardware):
    mojo test_qshim_multicast_heap_offsets.mojo
"""

from common import print_test_header

from std.ffi import external_call
from std.memory import UnsafePointer, alloc
from std.sys import size_of
from std.sys._libc import free
from std.sys.info import align_of
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
# Scenario constants (mirrors external-qualcomm MulticastOps fanout)
# ===-----------------------------------------------------------------------===#

comptime NUM_NSPS: Int = 4
comptime SOURCE_NSP: Int = 0
comptime PARTICIPANT_MASK: UInt32 = 0xF  # NSPs 0..3
comptime DEST_MASK: UInt32 = 0xE  # NSPs 1..3 (exclude source)
comptime SYNCPOINT: UInt32 = 0

comptime TRANSFER_SIZE: Int = 1024
comptime VTCM_SRC_OFFSET: Int = 0x1000
comptime VTCM_DST_OFFSET: Int = 0x3000
comptime SEED_BASE: UInt32 = 0x5C0FFEE5

comptime SINGLE_THREAD_MASK: UInt32 = 0x01
comptime ALIGNMENT: UInt32 = 128
comptime ALL_COMPUTE_UNITS: UInt64 = 0xFFFFFFFFFFFFFFFF

# Per-NSP result layout (UInt32 words)
comptime RESULT_MAGIC_IDX: Int = 0
comptime RESULT_CORE_ID_IDX: Int = 1
comptime RESULT_XFER_STATUS_IDX: Int = 2
comptime RESULT_WAIT_STATUS_IDX: Int = 3
comptime RESULT_BYTES_MATCHED_IDX: Int = 4
comptime RESULT_FIRST_FAIL_IDX: Int = 5
comptime RESULT_OFFSETS_PTR_IDX: Int = 6
comptime RESULT_FREE_OK_IDX: Int = 7
comptime RESULT_WORDS: Int = 8
comptime RESULT_BYTES: Int = RESULT_WORDS * 4
comptime RESULT_BUFFER_WORDS: Int = NUM_NSPS * RESULT_WORDS
comptime RESULT_BUFFER_BYTES: Int = RESULT_BUFFER_WORDS * 4

comptime RESULT_MAGIC: UInt32 = 0x47483236  # "GH26"
comptime RESULT_NOT_RUN: UInt32 = 0xFFFFFFFF
comptime RESULT_SKIPPED: UInt32 = 0xFFFFFFFE


@always_inline
def pattern_byte(i: UInt32) -> UInt8:
    var mix = SEED_BASE + i * 0x9E3779B9
    mix = mix ^ (mix >> 16)
    mix = mix ^ (mix >> 8)
    return UInt8(mix & 0xFF)


# ===-----------------------------------------------------------------------===#
# Device-side shims for libc allocation (work after today's plugin fix)
# ===-----------------------------------------------------------------------===#


@always_inline
def _dev_malloc(size: UInt32) -> UnsafePointer[UInt8, MutUnsafeAnyOrigin]:
    return external_call[
        "malloc", UnsafePointer[UInt8, MutUnsafeAnyOrigin], UInt32
    ](size)


@always_inline
def _dev_free(ptr: UnsafePointer[UInt8, MutUnsafeAnyOrigin]):
    free(ptr.bitcast[NoneType]())


# ===-----------------------------------------------------------------------===#
# Kernel: multicast with heap-allocated dst_offsets array (GH#26 pattern)
# ===-----------------------------------------------------------------------===#
#
# Mirrors MulticastOps._multicast from external-qualcomm but uses
# `external_call["malloc", ...]` for the offsets array instead of
# `pop.global_alloc`. This is the pattern the reporter originally wanted
# to write as `alloc[UnsafePointer[UInt32, MutUnsafeAnyOrigin]](num_nsps)`.


def heap_multicast_probe(
    result_words: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    for i in range(RESULT_WORDS):
        result_words[i] = RESULT_NOT_RUN
    result_words[RESULT_MAGIC_IDX] = RESULT_MAGIC
    result_words[RESULT_FREE_OK_IDX] = 0

    var core_id = Int(qshim_get_core_id())
    var core_id_u32 = UInt32(core_id)
    var thread_id = UInt32(qshim_get_thread_id())

    result_words[RESULT_CORE_ID_IDX] = core_id_u32

    # Every participating NSP seeds its VTCM destination with a sentinel so
    # we can tell "data arrived" from "data was always this value". Source
    # additionally fills the VTCM source region with the reference pattern.
    var vtcm_base = qshim_get_base_vtcm_addr()
    var src = vtcm_base + VTCM_SRC_OFFSET
    var dst = vtcm_base + VTCM_DST_OFFSET

    for i in range(TRANSFER_SIZE):
        dst[i] = UInt8(0xA5)

    if core_id == SOURCE_NSP:
        for i in range(TRANSFER_SIZE):
            src[i] = pattern_byte(UInt32(i))

    # Pre-transfer cross-NSP barrier.
    _ = qshim_sync_nsps(PARTICIPANT_MASK)

    # Only the source NSP builds the offsets array + submits the multicast.
    # Destinations just wait.
    if core_id == SOURCE_NSP:
        # Heap-allocate the offsets array of pointer-to-UInt32, matching
        # external-qualcomm MulticastOps._multicast's `static_alloc[...]`
        # layout exactly — same element type, same count, just coming from
        # libc malloc instead of `pop.global_alloc`. This is the core
        # GH#26 assertion: this allocation pattern MUST work without
        # tripping the plugin's old raiseError double-free path.
        var elem_size = UInt32(
            size_of[UnsafePointer[UInt32, MutUnsafeAnyOrigin]]()
        )
        var total_size = elem_size * UInt32(NUM_NSPS)
        var raw = _dev_malloc(total_size)
        var offsets_table = raw.bitcast[
            UnsafePointer[UInt32, MutUnsafeAnyOrigin]
        ]()
        result_words[RESULT_OFFSETS_PTR_IDX] = UInt32(Int(raw))

        # Each destination NSP receives the broadcast at the same offset.
        # `dst_offset_value` is a stack-local UInt32 whose address the
        # offsets table points to — same contract MulticastOps uses.
        var dst_offset_value: UInt32 = UInt32(VTCM_DST_OFFSET)
        for i in range(NUM_NSPS):
            if i == SOURCE_NSP:
                # Source NSP isn't in DEST_MASK so this slot is never read;
                # use a dangling placeholder to satisfy the array's element
                # type.
                offsets_table[i] = UnsafePointer[
                    UInt32, MutUnsafeAnyOrigin
                ].unsafe_dangling()
            else:
                offsets_table[i] = UnsafePointer(
                    to=dst_offset_value
                ).as_unsafe_any_origin()

        result_words[RESULT_XFER_STATUS_IDX] = qshim_multicast_xfer(
            thread_id,
            src,
            offsets_table,
            DEST_MASK,
            SYNCPOINT,
            UInt32(TRANSFER_SIZE),
        )

        # Free the heap-allocated offsets. The multicast descriptor has
        # already been pushed to the DMA engine by this point, so it's
        # safe to release. If libc malloc/free were unsafe here (the
        # reporter's feared failure mode), this free would trigger the
        # old double-free path or otherwise misbehave. We record a
        # success flag that the host checks.
        _dev_free(raw)
        result_words[RESULT_FREE_OK_IDX] = 1

    # Mid-transfer barrier, then destinations wait on the multicast
    # completion semaphore.
    _ = qshim_sync_nsps(PARTICIPANT_MASK)

    if core_id != SOURCE_NSP:
        result_words[RESULT_WAIT_STATUS_IDX] = qshim_wait_on_multicast(
            SYNCPOINT
        )

    _ = qshim_sync_nsps(PARTICIPANT_MASK)

    # Verify: destinations should see `pattern_byte(i)`; source keeps
    # whatever it wrote (also `pattern_byte(i)`) and verifies its own
    # copy survived.
    var bytes_matched: UInt32 = 0
    var first_fail: UInt32 = RESULT_NOT_RUN
    if core_id == SOURCE_NSP:
        for i in range(TRANSFER_SIZE):
            if src[i] == pattern_byte(UInt32(i)):
                bytes_matched += 1
            elif first_fail == RESULT_NOT_RUN:
                first_fail = UInt32(i)
    else:
        for i in range(TRANSFER_SIZE):
            if dst[i] == pattern_byte(UInt32(i)):
                bytes_matched += 1
            elif first_fail == RESULT_NOT_RUN:
                first_fail = UInt32(i)

    result_words[RESULT_BYTES_MATCHED_IDX] = bytes_matched
    result_words[RESULT_FIRST_FAIL_IDX] = first_fail


# ===-----------------------------------------------------------------------===#
# Host driver
# ===-----------------------------------------------------------------------===#


def _alloc_ddr(
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


def _build_per_core_descs(
    num_cores: Int, result_buf: QAicBuf
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


def _verify(
    results: UnsafePointer[UInt32, MutUntrackedOrigin],
) -> Bool:
    var ok = True
    for nsp in range(NUM_NSPS):
        var base = nsp * RESULT_WORDS
        var magic = results[base + RESULT_MAGIC_IDX]
        var core_id = results[base + RESULT_CORE_ID_IDX]
        var xfer = results[base + RESULT_XFER_STATUS_IDX]
        var wait = results[base + RESULT_WAIT_STATUS_IDX]
        var matched = results[base + RESULT_BYTES_MATCHED_IDX]
        var first_fail = results[base + RESULT_FIRST_FAIL_IDX]
        var offsets_ptr = results[base + RESULT_OFFSETS_PTR_IDX]
        var free_ok = results[base + RESULT_FREE_OK_IDX]

        var xfer_str: String = "n/a" if xfer == RESULT_NOT_RUN else String(xfer)
        var wait_str: String = "n/a" if wait == RESULT_NOT_RUN else String(wait)
        var first_fail_str: String = (
            "none" if first_fail == RESULT_NOT_RUN else String(first_fail)
        )
        var offsets_ptr_str: String = (
            "n/a" if offsets_ptr == RESULT_NOT_RUN else hex(offsets_ptr)
        )

        print(
            "  NSP",
            nsp,
            " magic=",
            hex(magic),
            " core_id=",
            core_id,
            " xfer=",
            xfer_str,
            " wait=",
            wait_str,
            " matched=",
            matched,
            "/",
            TRANSFER_SIZE,
            " first_fail=",
            first_fail_str,
            " offsets_ptr=",
            offsets_ptr_str,
            " free_ok=",
            free_ok,
        )

        if magic != RESULT_MAGIC:
            ok = False
        if core_id != UInt32(nsp):
            ok = False
        if matched != UInt32(TRANSFER_SIZE):
            ok = False

        if nsp == SOURCE_NSP:
            # Source performs the xfer and must succeed + must successfully
            # free the heap-allocated offsets array.
            if xfer != 0:
                ok = False
            if free_ok != 1:
                ok = False
            if offsets_ptr == 0 or offsets_ptr == RESULT_NOT_RUN:
                ok = False
            # Source doesn't call wait; wait slot stays NOT_RUN.
            if wait != RESULT_NOT_RUN:
                ok = False
        else:
            # Destinations wait on the multicast syncpoint and must
            # receive a successful status + the sent bytes.
            if wait != 0:
                ok = False
            if xfer != RESULT_NOT_RUN:
                ok = False
    return ok


def _run_one_dispatch(title: String) raises -> Bool:
    print_test_header(title)
    var ctx = DeviceContext(device_id=0, api="qaic")
    var jit_device = qaic_get_jit_device(ctx)

    var elf = compile_info_npu[heap_multicast_probe]()
    var jit_func = qaic_load_jit_function(
        ctx,
        elf.function_name,
        elf.asm.unsafe_ptr().bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(elf.asm.byte_length()),
    )

    var result_buf = _alloc_ddr(ctx, jit_device, RESULT_BUFFER_BYTES)

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

    var descs = _build_per_core_descs(NUM_NSPS, result_buf)
    qaic_execute_jit_function(
        ctx,
        jit_func,
        descs.as_unsafe_any_origin(),
        1,
        UInt32(NUM_NSPS),
        SINGLE_THREAD_MASK,
    )
    descs.unsafe_free()

    var host = alloc[UInt32](RESULT_BUFFER_WORDS)
    qaic_copy_jit_buffer(
        ctx,
        result_buf,
        host.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(RESULT_BUFFER_BYTES),
        QAIC_COPY_FROM_DEVICE,
        0,
    )
    var ok = _verify(host)
    host.unsafe_free()

    qaic_release_jit_buffer(ctx, result_buf)
    qaic_release_jit_function(jit_func)
    print("  Result:", "PASS" if ok else "FAIL")
    return ok


# ===-----------------------------------------------------------------------===#
# Tests
# ===-----------------------------------------------------------------------===#


def test_heap_offsets_multicast() raises:
    """GH#26: qshim_multicast_xfer works with heap-allocated dst_offsets.

    One-to-many multicast NSP0 -> NSP1..3 using an offsets array allocated
    via libc malloc from the kernel. Asserts no double-free / hang (the
    original symptom), that destination NSPs see the sent bytes intact,
    and that the kernel can free the offsets array cleanly after the
    multicast descriptor is pushed.
    """
    assert_true(
        _run_one_dispatch(
            "GH#26 regression: heap-allocated dst_offsets via"
            " external_call[malloc]"
        )
    )


def test_heap_offsets_multicast_repeat() raises:
    """Run the same multicast twice back-to-back.

    If libc malloc/free in the kernel had any lingering double-free or
    use-after-free across repeated multicasts (the sort of symptom that
    originally showed up in GH#26 as intermittent hangs), a second
    dispatch against a fresh .so load would flush it out. Each dispatch
    re-creates its own DeviceContext so the plugin-side state is also
    reset between runs.
    """
    assert_true(_run_one_dispatch("GH#26 regression: first dispatch"))
    assert_true(_run_one_dispatch("GH#26 regression: repeat dispatch"))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
