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
"""Positive qshim multicast regression tests for QAIC JIT dispatch.

This test ports the successful `qshimMulticastXfer()` capabilities recently
validated in `qaic-jit-research` into a Mojo regression that runs through the
plugin repo's multi-NSP JIT execution path without constructing explicit
command buffers.

Covered positive cases:
- scalar-pointer `dst_offsets` contract (`NSP0 -> NSP1`)
- byte-granular through large transfer sizes
- one-to-many fanout with the same VTCM offset
- one-to-many fanout with distinct per-NSP destination offsets
- sparse low-ID pair (`NSP0 -> NSP3`)
- sequential syncpoint reuse after each wait completes
- odd-size transfer with unaligned source/destination offsets

To run (requires QAIC hardware):
    mojo test_qshim_multicast_success.mojo
"""

from common import print_test_header

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
    qshim_get_num_cores,
    qshim_get_thread_id,
    qshim_get_vtcm_size,
    qshim_multicast_xfer,
    qshim_sync_nsps,
    qshim_wait_on_multicast,
)


# ===-----------------------------------------------------------------------===#
# Probe metadata / result layout
# ===-----------------------------------------------------------------------===#

comptime MAX_NSPS: Int = 16
comptime MAX_ROUNDS: Int = 2
comptime UINT32_BYTES: Int = 4
comptime ALIGNMENT: UInt32 = 128
comptime ALL_COMPUTE_UNITS: UInt64 = 0xFFFFFFFFFFFFFFFF
comptime SINGLE_THREAD_MASK: UInt32 = 0x01
comptime LAYOUT_GAP_BYTES: Int = 0x1000

comptime ROLE_PARTICIPANT: UInt32 = 1 << 0
comptime ROLE_SOURCE: UInt32 = 1 << 1
comptime ROLE_DESTINATION: UInt32 = 1 << 2
comptime ROLE_WAITER: UInt32 = 1 << 3

comptime RESULT_MAGIC: UInt32 = 0x4D435354  # "MCST"
comptime RESULT_NOT_RUN: UInt32 = 0xFFFFFFFF
comptime RESULT_SKIPPED: UInt32 = 0xFFFFFFFE

comptime META_PARTICIPANT_MASK_IDX: Int = 0
comptime META_SOURCE_NSP_IDX: Int = 1
comptime META_DEST_MASK_IDX: Int = 2
comptime META_WAIT_MASK_IDX: Int = 3
comptime META_SYNCPOINT_IDX: Int = 4
comptime META_TRANSFER_SIZE_IDX: Int = 5
comptime META_SRC_OFFSET_IDX: Int = 6
comptime META_CONTROL_OFFSET_IDX: Int = 7
comptime META_ROUNDS_IDX: Int = 8
comptime META_SEED_BASE_IDX: Int = 9
comptime META_DST_OFFSETS_BASE_IDX: Int = 16
comptime META_WORDS: Int = META_DST_OFFSETS_BASE_IDX + MAX_NSPS
comptime META_BYTES: Int = META_WORDS * UINT32_BYTES

comptime RESULT_MAGIC_IDX: Int = 0
comptime RESULT_CORE_ID_IDX: Int = 1
comptime RESULT_NUM_CORES_IDX: Int = 2
comptime RESULT_THREAD_ID_IDX: Int = 3
comptime RESULT_ROLES_IDX: Int = 4
comptime RESULT_VTCM_SIZE_IDX: Int = 5
comptime RESULT_PARTICIPANT_MASK_IDX: Int = 6
comptime RESULT_DEST_MASK_IDX: Int = 7
comptime RESULT_TRANSFER_SIZE_IDX: Int = 8
comptime RESULT_DST_OFFSET_IDX: Int = 9
comptime RESULT_CONTROL_OFFSET_IDX: Int = 10
comptime RESULT_BYTES_MATCHED_IDX: Int = 11
comptime RESULT_CONTROL_BYTES_MATCHED_IDX: Int = 12
comptime RESULT_SRC_BYTES_MATCHED_IDX: Int = 13
comptime RESULT_FIRST_FAIL_BYTE_IDX: Int = 14
comptime RESULT_ROUNDS_COMPLETED_IDX: Int = 15
comptime RESULT_WAIT_CALLS_COMPLETED_IDX: Int = 16
comptime RESULT_XFER_STATUS_BASE_IDX: Int = 17
comptime RESULT_WAIT_STATUS_BASE_IDX: Int = 19
comptime RESULT_SYNC_PRE_BASE_IDX: Int = 21
comptime RESULT_SYNC_MID_BASE_IDX: Int = 23
comptime RESULT_SYNC_POST_BASE_IDX: Int = 25
comptime RESULT_WORDS: Int = 27
comptime RESULT_BYTES: Int = RESULT_WORDS * UINT32_BYTES
comptime RESULT_BUFFER_WORDS: Int = MAX_NSPS * RESULT_WORDS
comptime RESULT_BUFFER_BYTES: Int = RESULT_BUFFER_WORDS * UINT32_BYTES


# ===-----------------------------------------------------------------------===#
# Shared helpers
# ===-----------------------------------------------------------------------===#


@always_inline
def mask_has(mask: UInt32, bit: Int) -> Bool:
    return (mask & UInt32(1 << bit)) != 0


@always_inline
def align_up(value: Int, alignment: Int) -> Int:
    return ((value + alignment - 1) // alignment) * alignment


@always_inline
def pattern_byte(seed_base: UInt32, round: UInt32, index: UInt32) -> UInt8:
    var mix = seed_base + round * 0x11111111 + index * 0x9E3779B9
    mix = mix ^ (mix >> 16)
    mix = mix ^ (mix >> 8)
    return UInt8(mix & 0xFF)


@always_inline
def dest_sentinel_byte(core_id: UInt32, index: UInt32) -> UInt8:
    return UInt8((0xA5 ^ (core_id * 17) ^ index) & 0xFF)


@always_inline
def control_sentinel_byte(core_id: UInt32, index: UInt32) -> UInt8:
    return UInt8((0x5A ^ (core_id * 29) ^ (index * 3)) & 0xFF)


# ===-----------------------------------------------------------------------===#
# Kernel: qshim multicast probe
# ===-----------------------------------------------------------------------===#


def qshim_multicast_probe(
    meta: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
    result_words: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    for i in range(RESULT_WORDS):
        result_words[i] = RESULT_NOT_RUN

    result_words[RESULT_MAGIC_IDX] = RESULT_MAGIC
    result_words[RESULT_FIRST_FAIL_BYTE_IDX] = RESULT_NOT_RUN
    result_words[RESULT_SRC_BYTES_MATCHED_IDX] = RESULT_SKIPPED

    var core_id = Int(qshim_get_core_id())
    var core_id_u32 = UInt32(core_id)
    var thread_id = UInt32(qshim_get_thread_id())
    var num_cores = UInt32(qshim_get_num_cores())
    var vtcm_size = qshim_get_vtcm_size()

    var participant_mask = meta[META_PARTICIPANT_MASK_IDX]
    var source_nsp = Int(meta[META_SOURCE_NSP_IDX])
    var dest_mask = meta[META_DEST_MASK_IDX]
    var wait_mask = meta[META_WAIT_MASK_IDX]
    var syncpoint = meta[META_SYNCPOINT_IDX]
    var transfer_size = Int(meta[META_TRANSFER_SIZE_IDX])
    var src_offset = Int(meta[META_SRC_OFFSET_IDX])
    var control_offset = Int(meta[META_CONTROL_OFFSET_IDX])
    var rounds = Int(meta[META_ROUNDS_IDX])
    var seed_base = meta[META_SEED_BASE_IDX]
    var dst_offset = meta[META_DST_OFFSETS_BASE_IDX + core_id]

    if rounds < 1:
        rounds = 1
    if rounds > MAX_ROUNDS:
        rounds = MAX_ROUNDS

    var roles: UInt32 = 0
    if mask_has(participant_mask, core_id):
        roles |= ROLE_PARTICIPANT
    if core_id == source_nsp:
        roles |= ROLE_SOURCE
    if mask_has(dest_mask, core_id):
        roles |= ROLE_DESTINATION
    if mask_has(wait_mask, core_id):
        roles |= ROLE_WAITER

    result_words[RESULT_CORE_ID_IDX] = core_id_u32
    result_words[RESULT_NUM_CORES_IDX] = num_cores
    result_words[RESULT_THREAD_ID_IDX] = thread_id
    result_words[RESULT_ROLES_IDX] = roles
    result_words[RESULT_VTCM_SIZE_IDX] = vtcm_size
    result_words[RESULT_PARTICIPANT_MASK_IDX] = participant_mask
    result_words[RESULT_DEST_MASK_IDX] = dest_mask
    result_words[RESULT_TRANSFER_SIZE_IDX] = UInt32(transfer_size)
    result_words[RESULT_DST_OFFSET_IDX] = dst_offset
    result_words[RESULT_CONTROL_OFFSET_IDX] = UInt32(control_offset)

    if (roles & ROLE_PARTICIPANT) == 0:
        return

    var vtcm_base = qshim_get_base_vtcm_addr()
    var src = vtcm_base + src_offset
    var dst = vtcm_base + Int(dst_offset)
    var control = vtcm_base + control_offset

    var dst_offsets = Array[UInt32, MAX_NSPS](uninitialized=True)
    var dst_ptrs = Array[UnsafePointer[UInt32, MutUnsafeAnyOrigin], MAX_NSPS](
        uninitialized=True
    )
    for i in range(MAX_NSPS):
        dst_offsets[i] = meta[META_DST_OFFSETS_BASE_IDX + i]
        dst_ptrs[i] = UnsafePointer[UInt32, MutUnsafeAnyOrigin](
            unsafe_from_address=Int(UnsafePointer(to=dst_offsets[i]))
        )

    var wait_calls_completed = 0
    for round in range(rounds):
        for i in range(transfer_size):
            var byte_index = UInt32(i)
            dst[i] = dest_sentinel_byte(core_id_u32, byte_index)
            control[i] = control_sentinel_byte(core_id_u32, byte_index)

        if core_id == source_nsp:
            for i in range(transfer_size):
                src[i] = pattern_byte(seed_base, UInt32(round), UInt32(i))

        result_words[RESULT_SYNC_PRE_BASE_IDX + round] = qshim_sync_nsps(
            participant_mask
        )

        if core_id == source_nsp:
            result_words[
                RESULT_XFER_STATUS_BASE_IDX + round
            ] = qshim_multicast_xfer(
                thread_id,
                src,
                dst_ptrs.unsafe_ptr().as_unsafe_any_origin(),
                dest_mask,
                syncpoint,
                UInt32(transfer_size),
            )

        result_words[RESULT_SYNC_MID_BASE_IDX + round] = qshim_sync_nsps(
            participant_mask
        )

        if mask_has(wait_mask, core_id):
            result_words[
                RESULT_WAIT_STATUS_BASE_IDX + wait_calls_completed
            ] = qshim_wait_on_multicast(syncpoint)
            wait_calls_completed += 1

        result_words[RESULT_SYNC_POST_BASE_IDX + round] = qshim_sync_nsps(
            participant_mask
        )
        result_words[RESULT_ROUNDS_COMPLETED_IDX] = UInt32(round + 1)

    var verify_round = rounds - 1
    var bytes_matched: UInt32 = 0
    var control_bytes_matched: UInt32 = 0
    var src_bytes_matched: UInt32 = 0
    var first_fail_byte = RESULT_NOT_RUN

    for i in range(transfer_size):
        var index_u32 = UInt32(i)
        var expected_dst = dest_sentinel_byte(core_id_u32, index_u32)
        if mask_has(dest_mask, core_id):
            expected_dst = pattern_byte(
                seed_base, UInt32(verify_round), index_u32
            )

        if dst[i] == expected_dst:
            bytes_matched += 1
        elif first_fail_byte == RESULT_NOT_RUN:
            first_fail_byte = index_u32

        if control[i] == control_sentinel_byte(core_id_u32, index_u32):
            control_bytes_matched += 1

        if core_id == source_nsp:
            if src[i] == pattern_byte(
                seed_base, UInt32(verify_round), index_u32
            ):
                src_bytes_matched += 1

    result_words[RESULT_BYTES_MATCHED_IDX] = bytes_matched
    result_words[RESULT_CONTROL_BYTES_MATCHED_IDX] = control_bytes_matched
    result_words[RESULT_SRC_BYTES_MATCHED_IDX] = (
        src_bytes_matched if core_id == source_nsp else RESULT_SKIPPED
    )
    result_words[RESULT_FIRST_FAIL_BYTE_IDX] = first_fail_byte
    result_words[RESULT_WAIT_CALLS_COMPLETED_IDX] = UInt32(wait_calls_completed)


# ===-----------------------------------------------------------------------===#
# Host-side helpers
# ===-----------------------------------------------------------------------===#


def load_multicast_probe_jit_function(
    ctx: DeviceContext,
) raises -> QAicJitFunction:
    var elf = compile_info_npu[qshim_multicast_probe]()
    return qaic_load_jit_function(
        ctx,
        elf.function_name,
        elf.asm.unsafe_ptr().bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(elf.asm.byte_length()),
    )


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


def upload_words(
    ctx: DeviceContext,
    buffer: QAicBuf,
    data: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
    num_words: Int,
    offset: UInt64 = 0,
) raises:
    qaic_copy_jit_buffer(
        ctx,
        buffer,
        data.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(num_words * UINT32_BYTES),
        QAIC_COPY_TO_DEVICE,
        offset,
    )


def download_words(
    ctx: DeviceContext,
    buffer: QAicBuf,
    num_words: Int,
    offset: UInt64 = 0,
) raises -> UnsafePointer[UInt32, MutUntrackedOrigin]:
    var host = alloc[UInt32](num_words)
    qaic_copy_jit_buffer(
        ctx,
        buffer,
        host.bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(num_words * UINT32_BYTES),
        QAIC_COPY_FROM_DEVICE,
        offset,
    )
    return host


def build_multicast_descriptors(
    num_cores: Int,
    meta_buf: QAicBuf,
    result_buf: QAicBuf,
) -> UnsafePointer[QAicBufferDescriptor, MutUntrackedOrigin]:
    var descs = alloc[QAicBufferDescriptor](num_cores * 2)
    for nsp in range(num_cores):
        var base = nsp * 2
        descs[base] = QAicBufferDescriptor(
            meta_buf,
            0,
            UInt64(META_BYTES),
            QAIC_DEV_BUF_RESIDENT_KEEP,
            QAIC_DEV_BUF_AUTO_MAP,
            QAIC_DEV_BUF_ADDR_MODE_HSV32,
        )
        descs[base + 1] = QAicBufferDescriptor(
            result_buf,
            UInt64(nsp * RESULT_BYTES),
            UInt64(RESULT_BYTES),
            QAIC_DEV_BUF_RESIDENT_KEEP,
            QAIC_DEV_BUF_AUTO_MAP,
            QAIC_DEV_BUF_ADDR_MODE_HSV32,
        )
    return descs


def highest_set_bit(mask: UInt32) -> Int:
    var highest = 0
    for bit in range(MAX_NSPS):
        if mask_has(mask, bit):
            highest = bit
    return highest


def make_transfer_meta(
    participant_mask: UInt32,
    source_nsp: Int,
    dest_mask: UInt32,
    transfer_size: Int,
    syncpoint: UInt32,
    same_offsets: Bool = False,
    rounds: Int = 1,
) -> UnsafePointer[UInt32, MutUntrackedOrigin]:
    var meta = alloc[UInt32](META_WORDS)
    for i in range(META_WORDS):
        meta[i] = 0

    var effective_rounds = rounds
    if effective_rounds < 1:
        effective_rounds = 1
    if effective_rounds > MAX_ROUNDS:
        effective_rounds = MAX_ROUNDS

    meta[META_PARTICIPANT_MASK_IDX] = participant_mask
    meta[META_SOURCE_NSP_IDX] = UInt32(source_nsp)
    meta[META_DEST_MASK_IDX] = dest_mask
    meta[META_WAIT_MASK_IDX] = dest_mask
    meta[META_SYNCPOINT_IDX] = syncpoint
    meta[META_TRANSFER_SIZE_IDX] = UInt32(transfer_size)
    meta[META_ROUNDS_IDX] = UInt32(effective_rounds)
    meta[META_SEED_BASE_IDX] = (
        0x13579BDF + syncpoint * 257 + UInt32(transfer_size)
    )

    var active_mask = participant_mask | dest_mask | UInt32(1 << source_nsp)
    var num_nsps = highest_set_bit(active_mask) + 1
    var chunk = align_up(transfer_size, 64)
    var src_offset = LAYOUT_GAP_BYTES
    var dst_base = align_up(src_offset + chunk + LAYOUT_GAP_BYTES, 64)

    for i in range(MAX_NSPS):
        meta[META_DST_OFFSETS_BASE_IDX + i] = UInt32(dst_base)

    var cursor: Int
    if same_offsets:
        cursor = dst_base + chunk + LAYOUT_GAP_BYTES
    else:
        for i in range(num_nsps):
            meta[META_DST_OFFSETS_BASE_IDX + i] = UInt32(
                dst_base + i * (chunk + LAYOUT_GAP_BYTES)
            )
        cursor = dst_base + num_nsps * (chunk + LAYOUT_GAP_BYTES)

    meta[META_SRC_OFFSET_IDX] = UInt32(src_offset)
    meta[META_CONTROL_OFFSET_IDX] = UInt32(align_up(cursor, 64))
    return meta


def status_text(status: UInt32) -> String:
    if status == 0:
        return "0(success)"
    if status == RESULT_NOT_RUN:
        return "n/a"
    if status == RESULT_SKIPPED:
        return "skip"
    if status == 530:
        return "530(multicast inval)"
    if status == 531:
        return "531(wait inval)"
    if status == 533:
        return "533(sync inval)"
    return String(status)


def role_text(roles: UInt32) -> String:
    var text = ""
    if (roles & ROLE_PARTICIPANT) != 0:
        text += "P"
    if (roles & ROLE_SOURCE) != 0:
        text += "S"
    if (roles & ROLE_DESTINATION) != 0:
        text += "D"
    if (roles & ROLE_WAITER) != 0:
        text += "W"
    if text == "":
        return "-"
    return text


def verify_case_results(
    title: String,
    meta: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
    results: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
) -> Bool:
    var participant_mask = meta[META_PARTICIPANT_MASK_IDX]
    var source_nsp = Int(meta[META_SOURCE_NSP_IDX])
    var dest_mask = meta[META_DEST_MASK_IDX]
    var transfer_size = meta[META_TRANSFER_SIZE_IDX]
    var rounds = Int(meta[META_ROUNDS_IDX])

    print("  case:", title)
    print(
        "  participants=",
        hex(participant_mask),
        " dest=",
        hex(dest_mask),
        " syncpoint=",
        meta[META_SYNCPOINT_IDX],
        " bytes=",
        transfer_size,
    )

    var ok = True
    for nsp in range(MAX_NSPS):
        if not mask_has(participant_mask, nsp):
            continue

        var base = nsp * RESULT_WORDS
        var roles = results[base + RESULT_ROLES_IDX]
        print(
            "  NSP",
            nsp,
            " roles=",
            role_text(roles),
            " xfer0=",
            status_text(results[base + RESULT_XFER_STATUS_BASE_IDX]),
            " xfer1=",
            status_text(results[base + RESULT_XFER_STATUS_BASE_IDX + 1]),
            " wait0=",
            status_text(results[base + RESULT_WAIT_STATUS_BASE_IDX]),
            " wait1=",
            status_text(results[base + RESULT_WAIT_STATUS_BASE_IDX + 1]),
            " bytes=",
            results[base + RESULT_BYTES_MATCHED_IDX],
            "/",
            transfer_size,
            " ctrl=",
            results[base + RESULT_CONTROL_BYTES_MATCHED_IDX],
            "/",
            transfer_size,
            " first_fail=",
            results[base + RESULT_FIRST_FAIL_BYTE_IDX],
        )

        if results[base + RESULT_MAGIC_IDX] != RESULT_MAGIC:
            ok = False
            continue

        if results[base + RESULT_CORE_ID_IDX] != UInt32(nsp):
            ok = False
        if (roles & ROLE_PARTICIPANT) == 0:
            ok = False

        if nsp == source_nsp:
            if (roles & ROLE_SOURCE) == 0:
                ok = False
        else:
            if (roles & ROLE_SOURCE) != 0:
                ok = False

        if mask_has(dest_mask, nsp):
            if (roles & ROLE_DESTINATION) == 0 or (roles & ROLE_WAITER) == 0:
                ok = False
        else:
            if (roles & ROLE_DESTINATION) != 0 or (roles & ROLE_WAITER) != 0:
                ok = False

        if results[base + RESULT_TRANSFER_SIZE_IDX] != transfer_size:
            ok = False
        if results[base + RESULT_ROUNDS_COMPLETED_IDX] != UInt32(rounds):
            ok = False

        var expected_wait_calls = UInt32(
            rounds if mask_has(dest_mask, nsp) else 0
        )
        if (
            results[base + RESULT_WAIT_CALLS_COMPLETED_IDX]
            != expected_wait_calls
        ):
            ok = False

        if results[base + RESULT_BYTES_MATCHED_IDX] != transfer_size:
            ok = False
        if results[base + RESULT_CONTROL_BYTES_MATCHED_IDX] != transfer_size:
            ok = False

        if nsp == source_nsp:
            if results[base + RESULT_SRC_BYTES_MATCHED_IDX] != transfer_size:
                ok = False
        else:
            if results[base + RESULT_SRC_BYTES_MATCHED_IDX] != RESULT_SKIPPED:
                ok = False

        if results[base + RESULT_BYTES_MATCHED_IDX] == transfer_size:
            if results[base + RESULT_FIRST_FAIL_BYTE_IDX] != RESULT_NOT_RUN:
                ok = False
        else:
            if results[base + RESULT_FIRST_FAIL_BYTE_IDX] == RESULT_NOT_RUN:
                ok = False

        for round in range(MAX_ROUNDS):
            var expected_sync = UInt32(0 if round < rounds else RESULT_NOT_RUN)
            if (
                results[base + RESULT_SYNC_PRE_BASE_IDX + round]
                != expected_sync
            ):
                ok = False
            if (
                results[base + RESULT_SYNC_MID_BASE_IDX + round]
                != expected_sync
            ):
                ok = False
            if (
                results[base + RESULT_SYNC_POST_BASE_IDX + round]
                != expected_sync
            ):
                ok = False

            var expected_xfer = RESULT_NOT_RUN
            if round < rounds and nsp == source_nsp:
                expected_xfer = 0
            if (
                results[base + RESULT_XFER_STATUS_BASE_IDX + round]
                != expected_xfer
            ):
                ok = False

            var expected_wait = RESULT_NOT_RUN
            if round < rounds and mask_has(dest_mask, nsp):
                expected_wait = 0
            if (
                results[base + RESULT_WAIT_STATUS_BASE_IDX + round]
                != expected_wait
            ):
                ok = False

    print("  Result:", "PASS" if ok else "FAIL")
    return ok


def run_case(
    title: String, meta: UnsafePointer[UInt32, MutUnsafeAnyOrigin]
) raises -> Bool:
    var case_ctx = DeviceContext(device_id=0, api="qaic")
    var case_device = qaic_get_jit_device(case_ctx)
    var case_func = load_multicast_probe_jit_function(case_ctx)

    var meta_buf = allocate_ddr_buffer(case_ctx, case_device, META_BYTES)
    var result_buf = allocate_ddr_buffer(
        case_ctx, case_device, RESULT_BUFFER_BYTES
    )

    var result_init = alloc[UInt32](RESULT_BUFFER_WORDS)
    for i in range(RESULT_BUFFER_WORDS):
        result_init[i] = RESULT_NOT_RUN

    upload_words(case_ctx, meta_buf, meta, META_WORDS)
    upload_words(
        case_ctx,
        result_buf,
        result_init.as_unsafe_any_origin(),
        RESULT_BUFFER_WORDS,
    )
    result_init.unsafe_free()

    var participant_mask = meta[META_PARTICIPANT_MASK_IDX]
    var num_cores = highest_set_bit(participant_mask) + 1
    var descs = build_multicast_descriptors(num_cores, meta_buf, result_buf)
    qaic_execute_jit_function(
        case_ctx,
        case_func,
        descs.as_unsafe_any_origin(),
        2,
        UInt32(num_cores),
        SINGLE_THREAD_MASK,
    )
    descs.unsafe_free()

    var results = download_words(case_ctx, result_buf, RESULT_BUFFER_WORDS)
    var ok = verify_case_results(title, meta, results.as_unsafe_any_origin())
    results.unsafe_free()

    qaic_release_jit_buffer(case_ctx, meta_buf)
    qaic_release_jit_buffer(case_ctx, result_buf)
    qaic_release_jit_function(case_func)
    return ok


# ===-----------------------------------------------------------------------===#
# Case groups
# ===-----------------------------------------------------------------------===#


def case_scalar_ptr_contract() raises -> Bool:
    var meta = make_transfer_meta(0x3, 0, 0x2, 256, 0)
    var ok = run_case(
        "scalar-pointer dst_offsets contract (NSP0 -> NSP1)",
        meta.as_unsafe_any_origin(),
    )
    meta.unsafe_free()
    return ok


def case_size_sweep() raises -> Bool:
    var sizes = Array[UInt32, 8](uninitialized=True)
    sizes[0] = 1
    sizes[1] = 3
    sizes[2] = 7
    sizes[3] = 31
    sizes[4] = 255
    sizes[5] = 256
    sizes[6] = 4096
    sizes[7] = 1048576

    var ok = True
    for i in range(8):
        var size = Int(sizes[i])
        var meta = make_transfer_meta(0x3, 0, 0x2, size, 1)
        if not run_case(
            "size sweep: " + String(size) + " bytes",
            meta.as_unsafe_any_origin(),
        ):
            ok = False
        meta.unsafe_free()
    return ok


def case_same_offset_fanout() raises -> Bool:
    var meta = make_transfer_meta(0xF, 0, 0xE, 64 * 1024, 2, same_offsets=True)
    var ok = run_case(
        "one-to-many fanout with same VTCM offset",
        meta.as_unsafe_any_origin(),
    )
    meta.unsafe_free()
    return ok


def case_distinct_offset_fanout() raises -> Bool:
    var meta = make_transfer_meta(0xF, 0, 0xE, 64 * 1024, 3)
    var ok = run_case(
        "one-to-many fanout with distinct destination offsets",
        meta.as_unsafe_any_origin(),
    )
    meta.unsafe_free()
    return ok


def case_sparse_pair() raises -> Bool:
    var meta = make_transfer_meta(0x9, 0, 0x8, 4096, 4)
    var ok = run_case(
        "sparse low-ID pair (NSP0 -> NSP3)",
        meta.as_unsafe_any_origin(),
    )
    meta.unsafe_free()
    return ok


def case_syncpoint_reuse() raises -> Bool:
    var first_meta = make_transfer_meta(0x3, 0, 0x2, 4096, 1)
    var ok = run_case(
        "syncpoint reuse: first transfer on MCSP_1",
        first_meta.as_unsafe_any_origin(),
    )

    first_meta.unsafe_free()
    if not ok:
        return False

    var second_meta = make_transfer_meta(0x3, 0, 0x2, 8192, 1)
    ok = run_case(
        "syncpoint reuse: second transfer after full wait on MCSP_1",
        second_meta.as_unsafe_any_origin(),
    )
    second_meta.unsafe_free()
    return ok


def case_unaligned_offsets() raises -> Bool:
    var meta = make_transfer_meta(0x3, 0, 0x2, 37, 3)
    meta[META_SRC_OFFSET_IDX] += 3
    meta[META_DST_OFFSETS_BASE_IDX + 0] += 1
    meta[META_DST_OFFSETS_BASE_IDX + 1] += 5
    meta[META_CONTROL_OFFSET_IDX] += 7
    var ok = run_case(
        "odd-size transfer with unaligned source/destination offsets",
        meta.as_unsafe_any_origin(),
    )
    meta.unsafe_free()
    return ok


# ===-----------------------------------------------------------------------===#
# Main
# ===-----------------------------------------------------------------------===#


def run_named_subtest(group: String) -> Bool:
    if group == "scalar":
        print("Sub-test A: scalar-pointer dst_offsets contract")
        try:
            return case_scalar_ptr_contract()
        except e:
            print("  FAIL:", e)
            return False

    if group == "sizes":
        print("Sub-test B: size sweep (byte-granular through large)")
        try:
            return case_size_sweep()
        except e:
            print("  FAIL:", e)
            return False

    if group == "fanout-same":
        print("Sub-test C: same-offset fanout to NSP1..NSP3")
        try:
            return case_same_offset_fanout()
        except e:
            print("  FAIL:", e)
            return False

    if group == "fanout-distinct":
        print("Sub-test D: distinct-offset fanout to NSP1..NSP3")
        try:
            return case_distinct_offset_fanout()
        except e:
            print("  FAIL:", e)
            return False

    if group == "sparse":
        print("Sub-test E: sparse low-ID pair NSP0 -> NSP3")
        try:
            return case_sparse_pair()
        except e:
            print("  FAIL:", e)
            return False

    if group == "reuse":
        print("Sub-test F: sequential syncpoint reuse")
        try:
            return case_syncpoint_reuse()
        except e:
            print("  FAIL:", e)
            return False

    if group == "unaligned":
        print("Sub-test G: odd-size transfer with unaligned offsets")
        try:
            return case_unaligned_offsets()
        except e:
            print("  FAIL:", e)
            return False

    print("Unknown sub-test group:", group)
    return False


def assert_multicast_subtest(title: String, group: String) raises:
    print_test_header(title)
    assert_true(run_named_subtest(group))


def test_scalar_ptr_contract() raises:
    assert_multicast_subtest(
        "qshim Multicast: scalar-pointer dst_offsets contract",
        "scalar",
    )


def test_size_sweep() raises:
    assert_multicast_subtest(
        "qshim Multicast: size sweep",
        "sizes",
    )


def test_same_offset_fanout() raises:
    assert_multicast_subtest(
        "qshim Multicast: same-offset fanout",
        "fanout-same",
    )


def test_distinct_offset_fanout() raises:
    assert_multicast_subtest(
        "qshim Multicast: distinct-offset fanout",
        "fanout-distinct",
    )


def test_sparse_pair() raises:
    assert_multicast_subtest(
        "qshim Multicast: sparse low-ID pair",
        "sparse",
    )


def test_syncpoint_reuse() raises:
    assert_multicast_subtest(
        "qshim Multicast: sequential syncpoint reuse",
        "reuse",
    )


def test_unaligned_offsets() raises:
    assert_multicast_subtest(
        "qshim Multicast: unaligned offsets",
        "unaligned",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
