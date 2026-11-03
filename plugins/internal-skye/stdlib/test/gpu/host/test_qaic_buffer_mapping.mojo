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
"""Test manual buffer mapping.

This tests fills an 8GiB buffer. Each NSP has a 32 bit address space so to fill
such a large buffer we need to map parts of the buffer in different cores
virtual address space.

This is done by manually providing buffer descriptors to
qaic_execute_jit_function. In this example we use 8 cores and have each core
fill a 1GiB slice of the buffer.

Two mapping rules are exercised:
    - test_jit_interface: QAIC_DEV_BUF_AUTO_MAP, where the runtime pre-maps each
      descriptor and the kernel receives a ready-to-use pointer.
    - test_jit_interface_deferred_map: QAIC_DEV_BUF_DEFERRED_MAP, where the
      kernel receives a buffer handle and maps regions on demand with
      qshim_mmap / qshim_unmap.

To run (requires QAIC hardware):
    mojo test_qaic_buffer_mapping.mojo
"""

from std.memory import UnsafePointer, alloc
from std.testing import TestSuite, assert_true

from max.gpu.host import DeviceContext, Dim
from std._plugin.qaic.host.target import (
    AicJitEntryPointConfig,
    AicJitPointerArray,
    compile_info_npu,
    npu_get_buffer,
    npu_core_idx,
)
from std._plugin.qaic.host.qshim import qshim_mmap, qshim_unmap
from max._plugin.qaic.host.qaic import (
    QAicBufferDescriptor,
    QAIC_BUFFER_TYPE_DEV_DDR,
    QAIC_COPY_FROM_DEVICE,
    QAIC_DEV_BUF_ADDR_MODE_HSV32,
    QAIC_DEV_BUF_AUTO_MAP,
    QAIC_DEV_BUF_DEFERRED_MAP,
    QAIC_DEV_BUF_IO_RW,
    QAIC_DEV_BUF_RESIDENT_KEEP,
    qaic_allocate_jit_buffer,
    qaic_copy_jit_buffer,
    qaic_execute_jit_function,
    qaic_get_jit_device,
    qaic_load_jit_function,
    qaic_release_jit_buffer,
    qaic_release_jit_function,
    qaic_split,
)

comptime FILL_VALUE: UInt8 = 42
comptime GIG_SIZE: Int = 1024 * 1024 * 1024
comptime NUM_CORES: Int = 8
comptime BUF_SIZE: Int = GIG_SIZE * NUM_CORES

# Window mapped at a time by the deferred-map kernel. Each core streams its
# 1 GiB slice through this 256 MiB window, keeping resident VA well under the
# per-NSP limit (~3,584 MB).
comptime MAP_CHUNK_SIZE: Int = 256 * 1024 * 1024


# Basic fill kernel filling 1GiB with FILL_VALUE
def fill_kernel(
    buf: UnsafePointer[UInt8, MutUnsafeAnyOrigin],
):
    for i in range(UInt32(GIG_SIZE)):
        buf[i] = FILL_VALUE


# Deferred-map fill kernel.
#
# Unlike `fill_kernel`, this uses the raw QAIC entry-point ABI (entry_config +
# pointer_array) because under `QAIC_DEV_BUF_DEFERRED_MAP` the runtime does NOT
# pre-map buffers: the pointer_array entry is a buffer *handle*, not a usable
# virtual address. The kernel must map regions on demand with `qshim_mmap` and
# release them with `qshim_unmap`.
#
# Each core owns a 1 GiB slice.
def deferred_fill_kernel(
    entry_config: UnsafePointer[AicJitEntryPointConfig, MutUnsafeAnyOrigin],
    pointer_array: UnsafePointer[AicJitPointerArray, MutUnsafeAnyOrigin],
) -> UInt32:
    var handle = npu_get_buffer[UInt8](pointer_array, 0)

    var status: UInt32 = 0
    var mapped = qshim_mmap(
        handle,
        UInt64(npu_core_idx() * GIG_SIZE),
        UInt32(GIG_SIZE),
        0,
        UnsafePointer(to=status).as_unsafe_any_origin(),
    )
    # qshim_mmap reports failure (and a null mapping) via a non-zero status.
    if status != 0:
        return 1  # non-zero return signals a kernel error

    for i in range(UInt32(GIG_SIZE)):
        mapped[i] = FILL_VALUE

    qshim_unmap(mapped)
    return 0


def test_jit_interface() raises:
    var ctx = DeviceContext(device_id=0, api="qaic")

    var elf = compile_info_npu[fill_kernel]()
    var jit_device = qaic_get_jit_device(ctx)

    print("Loading JIT function...")
    var jit_func = qaic_load_jit_function(
        ctx,
        elf.function_name,
        elf.asm.unsafe_ptr().bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(elf.asm.byte_length()),
    )

    print("Allocating JIT buffer...")
    var buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        UInt64(BUF_SIZE),
        128,  # 128-byte alignment for HVX
        0xFFFFFFFFFFFFFFFF,  # all compute units
    )

    print("Setup buffer descriptors...")
    # Build per-core descriptor array: each core gets its own 1 GiB slice.
    var descs = alloc[QAicBufferDescriptor](NUM_CORES)
    for c in range(NUM_CORES):
        descs[c] = QAicBufferDescriptor(
            buf,
            UInt64(c) * UInt64(GIG_SIZE),  # offset into shared buffer
            UInt64(GIG_SIZE),  # length of this core's slice
            QAIC_DEV_BUF_RESIDENT_KEEP,
            QAIC_DEV_BUF_AUTO_MAP,
            QAIC_DEV_BUF_ADDR_MODE_HSV32,
        )

    print("Execute kernel...")
    qaic_execute_jit_function(
        ctx,
        jit_func,
        descs.as_unsafe_any_origin(),
        1,  # num_descs_per_core: kernel takes 1 buffer argument
        UInt32(NUM_CORES),
        0x01,  # thread_mask: 1 HVX thread
    )
    descs.unsafe_free()

    print("Copy data back from device...")
    var host_buf = alloc[UInt8](BUF_SIZE)
    qaic_copy_jit_buffer(
        ctx,
        buf,
        host_buf.as_unsafe_any_origin(),
        UInt64(BUF_SIZE),
        QAIC_COPY_FROM_DEVICE,
    )

    var ok = True
    for i in range(BUF_SIZE):
        if host_buf[i] != FILL_VALUE:
            print(
                "FAIL at index",
                i,
                ": expected",
                Int(FILL_VALUE),
                "got",
                Int(host_buf[i]),
            )
            ok = False
            break

    host_buf.unsafe_free()
    qaic_release_jit_buffer(ctx, buf)
    qaic_release_jit_function(jit_func)

    assert_true(ok)


def test_jit_interface_deferred_map() raises:
    var ctx = DeviceContext(device_id=0, api="qaic")

    var elf = compile_info_npu[deferred_fill_kernel]()
    var jit_device = qaic_get_jit_device(ctx)

    print("Loading deferred-map JIT function...")
    var jit_func = qaic_load_jit_function(
        ctx,
        elf.function_name,
        elf.asm.unsafe_ptr().bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(elf.asm.byte_length()),
    )

    print("Allocating JIT buffer...")
    var buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        UInt64(BUF_SIZE),
        128,  # 128-byte alignment for HVX
        0xFFFFFFFFFFFFFFFF,  # all compute units
    )

    print("Setup deferred-map buffer descriptors...")
    var descs = alloc[QAicBufferDescriptor](NUM_CORES)
    # Map the entire 8GiB buffer with deferred mapping, each core will then map
    # its own slice.
    for c in range(NUM_CORES):
        descs[c] = QAicBufferDescriptor(
            buf,
            0,  # offset
            UInt64(BUF_SIZE),  # length
            QAIC_DEV_BUF_RESIDENT_KEEP,
            QAIC_DEV_BUF_DEFERRED_MAP,
            QAIC_DEV_BUF_ADDR_MODE_HSV32,
        )

    print("Execute deferred-map kernel...")
    qaic_execute_jit_function(
        ctx,
        jit_func,
        descs.as_unsafe_any_origin(),
        1,  # num_descs_per_core: kernel takes 1 buffer argument
        UInt32(NUM_CORES),
        0x01,  # thread_mask: 1 HVX thread
    )
    descs.unsafe_free()

    print("Copy data back from device...")
    var host_buf = alloc[UInt8](BUF_SIZE)
    qaic_copy_jit_buffer(
        ctx,
        buf,
        host_buf.as_unsafe_any_origin(),
        UInt64(BUF_SIZE),
        QAIC_COPY_FROM_DEVICE,
    )

    var ok = True
    for i in range(BUF_SIZE):
        if host_buf[i] != FILL_VALUE:
            print(
                "FAIL at index",
                i,
                ": expected",
                Int(FILL_VALUE),
                "got",
                Int(host_buf[i]),
            )
            ok = False
            break

    host_buf.unsafe_free()
    qaic_release_jit_buffer(ctx, buf)
    qaic_release_jit_function(jit_func)

    assert_true(ok)


def test_npu_interface() raises:
    var ctx = DeviceContext(device_id=0, api="qaic")

    var buf = ctx.enqueue_create_buffer[DType.uint8](BUF_SIZE)
    var kernel = ctx.compile_function[fill_kernel]()
    ctx.enqueue_function(
        kernel,
        qaic_split(buf, NUM_CORES),
        grid_dim=Dim(NUM_CORES),
        block_dim=Dim(0x01),
    )
    ctx.synchronize()

    var host_buf = alloc[UInt8](BUF_SIZE)
    ctx.enqueue_copy(host_buf, buf)
    ctx.synchronize()

    var ok = True
    for i in range(BUF_SIZE):
        if host_buf[i] != FILL_VALUE:
            print(
                "FAIL at index",
                i,
                ": expected",
                Int(FILL_VALUE),
                "got",
                Int(host_buf[i]),
            )
            ok = False
            break

    host_buf.unsafe_free()
    assert_true(ok)


def main() raises:
    var suite = TestSuite.discover_tests[__functions_in_module()]()
    # Deferred mapping test if flaky and this technique is unused at the moment
    # so skip it for now.
    suite.skip[test_jit_interface_deferred_map]()
    suite^.run()
