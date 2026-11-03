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
"""Exercise manual QAIC command-buffer behavior in one consolidated mojo_test.

The scenarios below cover the same behaviors as the original audit-style tests:
cuBitMap validation, command-buffer reuse, mutable device buffers, baked local
storage, and batch submission.
"""

from common import print_separator, print_test_header

from max.gpu.host import DeviceContext
from std._plugin.qaic.host.target import compile_info_npu
from max._plugin.qaic.host.qaic import (
    QAicCommandBuffer,
    QAicComputeUnitBitMap,
    QAicDescriptorIndex,
    QAicDevice,
    QAicJitFunction,
    QAicStream,
    QAicThreadBitmap,
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
    qaic_get_jit_device,
    qaic_load_jit_function,
    qaic_push_jit_buffer_descriptor,
    qaic_push_jit_local_storage,
    qaic_release_jit_buffer,
    qaic_release_jit_command_buffer,
    qaic_release_jit_function,
    qaic_release_jit_stream,
    qaic_submit_and_wait_jit_command_buffers,
)
from std.memory import UnsafePointer, alloc
from std.sys import size_of
from std.testing import TestSuite, assert_true

comptime DEFAULT_ALIGNMENT: UInt32 = 128
comptime ALL_COMPUTE_UNITS: QAicComputeUnitBitMap = 0xFFFFFFFFFFFFFFFF
comptime SINGLE_THREAD_MASK: QAicThreadBitmap = 0x01
comptime FLOAT32_BYTES: UInt64 = 4


def scalar_muladd_kernel(
    output: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    input_a: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    input_b: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    bias_ptr: UnsafePointer[Float32, MutUnsafeAnyOrigin],
    num_elements_ptr: UnsafePointer[UInt32, MutUnsafeAnyOrigin],
):
    var bias = bias_ptr[]
    var num_elements = Int(num_elements_ptr[])
    for i in range(num_elements):
        output[i] = input_a[i] * input_b[i] + bias


@always_inline
def count_constant_f32_mismatches(
    data: UnsafePointer[Float32, MutUntrackedOrigin],
    start: Int,
    num_elements: Int,
    expected: Float32,
    atol: Float32 = 0.001,
) -> Int:
    var mismatches = 0
    for i in range(start, start + num_elements):
        if abs(data[i] - expected) > atol:
            mismatches += 1
    return mismatches


def verify_constant_region(
    label: String,
    output: UnsafePointer[Float32, MutUntrackedOrigin],
    start: Int,
    num_elements: Int,
    expected: Float32,
) -> Bool:
    var mismatches = count_constant_f32_mismatches(
        output, start, num_elements, expected
    )
    print(
        "  ["
        + label
        + "] out["
        + String(start)
        + "]="
        + String(output[start])
        + " expected="
        + String(expected)
        + " mismatches="
        + String(mismatches)
    )
    return mismatches == 0


def fill_constant_f32(
    data: UnsafePointer[Float32, MutUntrackedOrigin],
    num_elements: Int,
    value: Float32,
):
    for i in range(num_elements):
        data[i] = value


def run_cubitmap_validation_test(
    ctx: DeviceContext,
    jit_device: QAicDevice,
    stream: QAicStream,
    jit_func: QAicJitFunction,
) raises -> Bool:
    print("Scenario 1: command-buffer cuBitMap validation")
    var passed = 0
    var fail = 0

    print("Sub-test A: valid single-bit cuBitMap 0x1 is accepted")
    try:
        var cmd_buffer = qaic_create_jit_command_buffer(ctx, jit_device, 0x1)
        qaic_release_jit_command_buffer(ctx, cmd_buffer)
        print("  PASS")
        passed += 1
    except e:
        print("  FAIL:", e)
        fail += 1
    print()

    print("Sub-test B: valid single-bit cuBitMap 0x2 is accepted")
    try:
        var cmd_buffer = qaic_create_jit_command_buffer(ctx, jit_device, 0x2)
        qaic_release_jit_command_buffer(ctx, cmd_buffer)
        print("  PASS")
        passed += 1
    except e:
        print("  FAIL:", e)
        fail += 1
    print()

    # The current runtime aborts after back-to-back invalid create attempts, so
    # keep a single representative rejection case in this integration test.
    print("Sub-test C: multi-bit cuBitMap 0x3 is rejected")
    try:
        var cmd_buffer = qaic_create_jit_command_buffer(ctx, jit_device, 0x3)
        qaic_release_jit_command_buffer(ctx, cmd_buffer)
        print("  FAIL: multi-bit cuBitMap unexpectedly succeeded")
        fail += 1
    except e:
        print("  PASS:", e)
        passed += 1
    print()

    print("Sub-test D: functional execution still works with single-bit 0x1")
    var num_elements = 512
    var byte_size = UInt64(num_elements) * FLOAT32_BYTES
    var output_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        byte_size,
        DEFAULT_ALIGNMENT,
        ALL_COMPUTE_UNITS,
    )
    var input_a_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        byte_size,
        DEFAULT_ALIGNMENT,
        ALL_COMPUTE_UNITS,
    )
    var input_b_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        byte_size,
        DEFAULT_ALIGNMENT,
        ALL_COMPUTE_UNITS,
    )
    var input_a_host = alloc[Float32](num_elements)
    var input_b_host = alloc[Float32](num_elements)
    var output = alloc[Float32](num_elements)

    fill_constant_f32(input_a_host, num_elements, 4.0)
    qaic_copy_jit_buffer(
        ctx,
        input_a_buf,
        input_a_host.bitcast[UInt8]().as_unsafe_any_origin(),
        byte_size,
        QAIC_COPY_TO_DEVICE,
        0,
    )
    fill_constant_f32(input_b_host, num_elements, 5.0)
    qaic_copy_jit_buffer(
        ctx,
        input_b_buf,
        input_b_host.bitcast[UInt8]().as_unsafe_any_origin(),
        byte_size,
        QAIC_COPY_TO_DEVICE,
        0,
    )

    var functional_cb = qaic_create_jit_command_buffer(ctx, jit_device, 0x1)
    qaic_begin_jit_command_buffer(ctx, functional_cb)

    var arg_indices = Array[QAicDescriptorIndex, 5](uninitialized=True)
    arg_indices[0] = qaic_push_jit_buffer_descriptor(
        ctx,
        functional_cb,
        output_buf,
        0,
        byte_size,
        QAIC_DEV_BUF_RESIDENT_KEEP,
    )
    arg_indices[1] = qaic_push_jit_buffer_descriptor(
        ctx,
        functional_cb,
        input_a_buf,
        0,
        byte_size,
        QAIC_DEV_BUF_RESIDENT_KEEP,
    )
    arg_indices[2] = qaic_push_jit_buffer_descriptor(
        ctx,
        functional_cb,
        input_b_buf,
        0,
        byte_size,
        QAIC_DEV_BUF_RESIDENT_KEEP,
    )
    var bias_copy: Float32 = 2.0
    arg_indices[3] = qaic_push_jit_local_storage(
        ctx,
        functional_cb,
        UnsafePointer(to=bias_copy).bitcast[UInt8]().as_unsafe_any_origin(),
        UInt32(size_of[Float32]()),
    )
    var num_elements_u32 = UInt32(num_elements)
    arg_indices[4] = qaic_push_jit_local_storage(
        ctx,
        functional_cb,
        UnsafePointer(to=num_elements_u32)
        .bitcast[UInt8]()
        .as_unsafe_any_origin(),
        UInt32(size_of[UInt32]()),
    )

    qaic_dispatch_jit_command_buffer(
        ctx,
        functional_cb,
        jit_func,
        arg_indices.unsafe_ptr().as_unsafe_any_origin(),
        UInt32(5),
        SINGLE_THREAD_MASK,
    )
    qaic_end_jit_command_buffer(ctx, functional_cb)

    var single_submit = Array[QAicCommandBuffer, 1](uninitialized=True)
    single_submit[0] = functional_cb
    qaic_submit_and_wait_jit_command_buffers(
        ctx,
        stream,
        single_submit.unsafe_ptr().as_unsafe_any_origin(),
        UInt32(1),
    )
    qaic_copy_jit_buffer(
        ctx,
        output_buf,
        output.bitcast[UInt8]().as_unsafe_any_origin(),
        byte_size,
        QAIC_COPY_FROM_DEVICE,
        0,
    )
    var mismatches = count_constant_f32_mismatches(
        output, 0, num_elements, 22.0
    )
    print("  out[0]=", output[0], " expected=22.0 mismatches=", mismatches)
    if mismatches == 0:
        passed += 1
    else:
        fail += 1
    output.unsafe_free()
    input_a_host.unsafe_free()
    input_b_host.unsafe_free()
    qaic_release_jit_command_buffer(ctx, functional_cb)
    qaic_release_jit_buffer(ctx, output_buf)
    qaic_release_jit_buffer(ctx, input_a_buf)
    qaic_release_jit_buffer(ctx, input_b_buf)
    print()

    print("Scenario 1 result:", passed, "passed,", fail, "failed")
    print_separator()
    return fail == 0


def run_reuse_test(
    ctx: DeviceContext,
    jit_device: QAicDevice,
    stream: QAicStream,
    jit_func: QAicJitFunction,
) raises -> Bool:
    print("Scenario 2: command-buffer reuse with mutable inputs")
    var num_elements = 2048
    var byte_size = UInt64(num_elements) * FLOAT32_BYTES
    var bias: Float32 = 2.0
    var cu_bitmap: QAicComputeUnitBitMap = 0x1

    print("Configuration:")
    print("  Elements:", num_elements)
    print("  Bias:", bias)
    print("  Compute unit bitmap: 0x1")
    print()

    var output_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        byte_size,
        DEFAULT_ALIGNMENT,
        ALL_COMPUTE_UNITS,
    )
    var input_a_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        byte_size,
        DEFAULT_ALIGNMENT,
        ALL_COMPUTE_UNITS,
    )
    var input_b_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        byte_size,
        DEFAULT_ALIGNMENT,
        ALL_COMPUTE_UNITS,
    )
    var input_a_host = alloc[Float32](num_elements)
    var input_b_host = alloc[Float32](num_elements)
    var output = alloc[Float32](num_elements)

    var passed = 0
    var fail = 0

    print("Sub-test A: initial submit (a=3, b=4 -> 14.0)")
    fill_constant_f32(input_a_host, num_elements, 3.0)
    qaic_copy_jit_buffer(
        ctx,
        input_a_buf,
        input_a_host.bitcast[UInt8]().as_unsafe_any_origin(),
        byte_size,
        QAIC_COPY_TO_DEVICE,
        0,
    )
    fill_constant_f32(input_b_host, num_elements, 4.0)
    qaic_copy_jit_buffer(
        ctx,
        input_b_buf,
        input_b_host.bitcast[UInt8]().as_unsafe_any_origin(),
        byte_size,
        QAIC_COPY_TO_DEVICE,
        0,
    )
    var cmd_buffer = qaic_create_jit_command_buffer(ctx, jit_device, cu_bitmap)
    qaic_begin_jit_command_buffer(ctx, cmd_buffer)

    var arg_indices = Array[QAicDescriptorIndex, 5](uninitialized=True)
    arg_indices[0] = qaic_push_jit_buffer_descriptor(
        ctx,
        cmd_buffer,
        output_buf,
        0,
        byte_size,
        QAIC_DEV_BUF_RESIDENT_KEEP,
    )
    arg_indices[1] = qaic_push_jit_buffer_descriptor(
        ctx,
        cmd_buffer,
        input_a_buf,
        0,
        byte_size,
        QAIC_DEV_BUF_RESIDENT_KEEP,
    )
    arg_indices[2] = qaic_push_jit_buffer_descriptor(
        ctx,
        cmd_buffer,
        input_b_buf,
        0,
        byte_size,
        QAIC_DEV_BUF_RESIDENT_KEEP,
    )
    var bias_copy = bias
    arg_indices[3] = qaic_push_jit_local_storage(
        ctx,
        cmd_buffer,
        UnsafePointer(to=bias_copy).bitcast[UInt8]().as_unsafe_any_origin(),
        UInt32(size_of[Float32]()),
    )
    var num_elements_u32 = UInt32(num_elements)
    arg_indices[4] = qaic_push_jit_local_storage(
        ctx,
        cmd_buffer,
        UnsafePointer(to=num_elements_u32)
        .bitcast[UInt8]()
        .as_unsafe_any_origin(),
        UInt32(size_of[UInt32]()),
    )

    qaic_dispatch_jit_command_buffer(
        ctx,
        cmd_buffer,
        jit_func,
        arg_indices.unsafe_ptr().as_unsafe_any_origin(),
        UInt32(5),
        SINGLE_THREAD_MASK,
    )
    qaic_end_jit_command_buffer(ctx, cmd_buffer)

    var single_submit = Array[QAicCommandBuffer, 1](uninitialized=True)
    single_submit[0] = cmd_buffer
    qaic_submit_and_wait_jit_command_buffers(
        ctx,
        stream,
        single_submit.unsafe_ptr().as_unsafe_any_origin(),
        UInt32(1),
    )
    qaic_copy_jit_buffer(
        ctx,
        output_buf,
        output.bitcast[UInt8]().as_unsafe_any_origin(),
        byte_size,
        QAIC_COPY_FROM_DEVICE,
        0,
    )
    if verify_constant_region("Run1", output, 0, num_elements, 14.0):
        passed += 1
    else:
        fail += 1
    print()

    print("Sub-test B: reuse same CB after mutating a=1 (expected 6.0)")
    fill_constant_f32(input_a_host, num_elements, 1.0)
    qaic_copy_jit_buffer(
        ctx,
        input_a_buf,
        input_a_host.bitcast[UInt8]().as_unsafe_any_origin(),
        byte_size,
        QAIC_COPY_TO_DEVICE,
        0,
    )
    qaic_submit_and_wait_jit_command_buffers(
        ctx,
        stream,
        single_submit.unsafe_ptr().as_unsafe_any_origin(),
        UInt32(1),
    )
    qaic_copy_jit_buffer(
        ctx,
        output_buf,
        output.bitcast[UInt8]().as_unsafe_any_origin(),
        byte_size,
        QAIC_COPY_FROM_DEVICE,
        0,
    )
    if verify_constant_region("Run2", output, 0, num_elements, 6.0):
        passed += 1
    else:
        fail += 1
    print()

    print("Sub-test C: reuse same CB after mutating b=10 (expected 12.0)")
    fill_constant_f32(input_b_host, num_elements, 10.0)
    qaic_copy_jit_buffer(
        ctx,
        input_b_buf,
        input_b_host.bitcast[UInt8]().as_unsafe_any_origin(),
        byte_size,
        QAIC_COPY_TO_DEVICE,
        0,
    )
    qaic_submit_and_wait_jit_command_buffers(
        ctx,
        stream,
        single_submit.unsafe_ptr().as_unsafe_any_origin(),
        UInt32(1),
    )
    qaic_copy_jit_buffer(
        ctx,
        output_buf,
        output.bitcast[UInt8]().as_unsafe_any_origin(),
        byte_size,
        QAIC_COPY_FROM_DEVICE,
        0,
    )
    if verify_constant_region("Run3", output, 0, num_elements, 12.0):
        passed += 1
    else:
        fail += 1
    print()

    print("Sub-test D: reuse same CB after mutating a=5, b=5 (expected 27.0)")
    fill_constant_f32(input_a_host, num_elements, 5.0)
    qaic_copy_jit_buffer(
        ctx,
        input_a_buf,
        input_a_host.bitcast[UInt8]().as_unsafe_any_origin(),
        byte_size,
        QAIC_COPY_TO_DEVICE,
        0,
    )
    fill_constant_f32(input_b_host, num_elements, 5.0)
    qaic_copy_jit_buffer(
        ctx,
        input_b_buf,
        input_b_host.bitcast[UInt8]().as_unsafe_any_origin(),
        byte_size,
        QAIC_COPY_TO_DEVICE,
        0,
    )
    qaic_submit_and_wait_jit_command_buffers(
        ctx,
        stream,
        single_submit.unsafe_ptr().as_unsafe_any_origin(),
        UInt32(1),
    )
    qaic_copy_jit_buffer(
        ctx,
        output_buf,
        output.bitcast[UInt8]().as_unsafe_any_origin(),
        byte_size,
        QAIC_COPY_FROM_DEVICE,
        0,
    )
    if verify_constant_region("Run4", output, 0, num_elements, 27.0):
        passed += 1
    else:
        fail += 1
    print()

    print("Sub-test E: reuse same CB after mutating a=0, b=0 (expected 2.0)")
    fill_constant_f32(input_a_host, num_elements, 0.0)
    qaic_copy_jit_buffer(
        ctx,
        input_a_buf,
        input_a_host.bitcast[UInt8]().as_unsafe_any_origin(),
        byte_size,
        QAIC_COPY_TO_DEVICE,
        0,
    )
    fill_constant_f32(input_b_host, num_elements, 0.0)
    qaic_copy_jit_buffer(
        ctx,
        input_b_buf,
        input_b_host.bitcast[UInt8]().as_unsafe_any_origin(),
        byte_size,
        QAIC_COPY_TO_DEVICE,
        0,
    )
    qaic_submit_and_wait_jit_command_buffers(
        ctx,
        stream,
        single_submit.unsafe_ptr().as_unsafe_any_origin(),
        UInt32(1),
    )
    qaic_copy_jit_buffer(
        ctx,
        output_buf,
        output.bitcast[UInt8]().as_unsafe_any_origin(),
        byte_size,
        QAIC_COPY_FROM_DEVICE,
        0,
    )
    if verify_constant_region("Run5", output, 0, num_elements, 2.0):
        passed += 1
    else:
        fail += 1
    print()

    qaic_release_jit_command_buffer(ctx, cmd_buffer)

    print("Sub-test F: rebuilt baseline with final inputs (expected 2.0)")
    var rebuilt_cmd_buffer = qaic_create_jit_command_buffer(
        ctx, jit_device, cu_bitmap
    )
    qaic_begin_jit_command_buffer(ctx, rebuilt_cmd_buffer)

    var rebuilt_arg_indices = Array[QAicDescriptorIndex, 5](uninitialized=True)
    rebuilt_arg_indices[0] = qaic_push_jit_buffer_descriptor(
        ctx,
        rebuilt_cmd_buffer,
        output_buf,
        0,
        byte_size,
        QAIC_DEV_BUF_RESIDENT_KEEP,
    )
    rebuilt_arg_indices[1] = qaic_push_jit_buffer_descriptor(
        ctx,
        rebuilt_cmd_buffer,
        input_a_buf,
        0,
        byte_size,
        QAIC_DEV_BUF_RESIDENT_KEEP,
    )
    rebuilt_arg_indices[2] = qaic_push_jit_buffer_descriptor(
        ctx,
        rebuilt_cmd_buffer,
        input_b_buf,
        0,
        byte_size,
        QAIC_DEV_BUF_RESIDENT_KEEP,
    )
    var rebuilt_bias_copy = bias
    rebuilt_arg_indices[3] = qaic_push_jit_local_storage(
        ctx,
        rebuilt_cmd_buffer,
        UnsafePointer(to=rebuilt_bias_copy)
        .bitcast[UInt8]()
        .as_unsafe_any_origin(),
        UInt32(size_of[Float32]()),
    )
    var rebuilt_num_elements_u32 = UInt32(num_elements)
    rebuilt_arg_indices[4] = qaic_push_jit_local_storage(
        ctx,
        rebuilt_cmd_buffer,
        UnsafePointer(to=rebuilt_num_elements_u32)
        .bitcast[UInt8]()
        .as_unsafe_any_origin(),
        UInt32(size_of[UInt32]()),
    )

    qaic_dispatch_jit_command_buffer(
        ctx,
        rebuilt_cmd_buffer,
        jit_func,
        rebuilt_arg_indices.unsafe_ptr().as_unsafe_any_origin(),
        UInt32(5),
        SINGLE_THREAD_MASK,
    )
    qaic_end_jit_command_buffer(ctx, rebuilt_cmd_buffer)

    single_submit[0] = rebuilt_cmd_buffer
    qaic_submit_and_wait_jit_command_buffers(
        ctx,
        stream,
        single_submit.unsafe_ptr().as_unsafe_any_origin(),
        UInt32(1),
    )
    qaic_copy_jit_buffer(
        ctx,
        output_buf,
        output.bitcast[UInt8]().as_unsafe_any_origin(),
        byte_size,
        QAIC_COPY_FROM_DEVICE,
        0,
    )
    if verify_constant_region("Run6-rebuild", output, 0, num_elements, 2.0):
        passed += 1
    else:
        fail += 1
    qaic_release_jit_command_buffer(ctx, rebuilt_cmd_buffer)
    print()

    output.unsafe_free()
    input_a_host.unsafe_free()
    input_b_host.unsafe_free()
    qaic_release_jit_buffer(ctx, output_buf)
    qaic_release_jit_buffer(ctx, input_a_buf)
    qaic_release_jit_buffer(ctx, input_b_buf)

    print("Scenario 2 result:", passed, "passed,", fail, "failed")
    print_separator()
    return fail == 0


def run_mutable_device_buffer_test(
    ctx: DeviceContext,
    jit_device: QAicDevice,
    stream: QAicStream,
    jit_func: QAicJitFunction,
) raises -> Bool:
    print("Scenario 3: mutable device buffers across command-buffer reuse")
    var num_elements = 768
    var byte_size = UInt64(num_elements) * FLOAT32_BYTES
    var bias: Float32 = 1.0
    var cu_bitmap: QAicComputeUnitBitMap = 0x1

    print("Configuration:")
    print("  Elements:", num_elements)
    print("  Bias:", bias)
    print("  Compute unit bitmap: 0x1")
    print()

    var output_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        byte_size,
        DEFAULT_ALIGNMENT,
        ALL_COMPUTE_UNITS,
    )
    var input_a_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        byte_size,
        DEFAULT_ALIGNMENT,
        ALL_COMPUTE_UNITS,
    )
    var input_b_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        byte_size,
        DEFAULT_ALIGNMENT,
        ALL_COMPUTE_UNITS,
    )

    var host_a = alloc[Float32](num_elements)
    var host_b = alloc[Float32](num_elements)
    var host_output = alloc[Float32](num_elements)

    var cmd_buffer = qaic_create_jit_command_buffer(ctx, jit_device, cu_bitmap)
    qaic_begin_jit_command_buffer(ctx, cmd_buffer)

    var arg_indices = Array[QAicDescriptorIndex, 5](uninitialized=True)
    arg_indices[0] = qaic_push_jit_buffer_descriptor(
        ctx,
        cmd_buffer,
        output_buf,
        0,
        byte_size,
        QAIC_DEV_BUF_RESIDENT_KEEP,
    )
    arg_indices[1] = qaic_push_jit_buffer_descriptor(
        ctx,
        cmd_buffer,
        input_a_buf,
        0,
        byte_size,
        QAIC_DEV_BUF_RESIDENT_KEEP,
    )
    arg_indices[2] = qaic_push_jit_buffer_descriptor(
        ctx,
        cmd_buffer,
        input_b_buf,
        0,
        byte_size,
        QAIC_DEV_BUF_RESIDENT_KEEP,
    )
    var bias_copy = bias
    arg_indices[3] = qaic_push_jit_local_storage(
        ctx,
        cmd_buffer,
        UnsafePointer(to=bias_copy).bitcast[UInt8]().as_unsafe_any_origin(),
        UInt32(size_of[Float32]()),
    )
    var num_elements_u32 = UInt32(num_elements)
    arg_indices[4] = qaic_push_jit_local_storage(
        ctx,
        cmd_buffer,
        UnsafePointer(to=num_elements_u32)
        .bitcast[UInt8]()
        .as_unsafe_any_origin(),
        UInt32(size_of[UInt32]()),
    )

    qaic_dispatch_jit_command_buffer(
        ctx,
        cmd_buffer,
        jit_func,
        arg_indices.unsafe_ptr().as_unsafe_any_origin(),
        UInt32(5),
        SINGLE_THREAD_MASK,
    )
    qaic_end_jit_command_buffer(ctx, cmd_buffer)

    var single_submit = Array[QAicCommandBuffer, 1](uninitialized=True)
    single_submit[0] = cmd_buffer

    var passed = 0
    var fail = 0

    print("Sub-test A: initial submit a=3, b=7 -> 22.0")
    fill_constant_f32(host_a, num_elements, 3.0)
    fill_constant_f32(host_b, num_elements, 7.0)
    qaic_copy_jit_buffer(
        ctx,
        input_a_buf,
        host_a.bitcast[UInt8]().as_unsafe_any_origin(),
        byte_size,
        QAIC_COPY_TO_DEVICE,
        0,
    )
    qaic_copy_jit_buffer(
        ctx,
        input_b_buf,
        host_b.bitcast[UInt8]().as_unsafe_any_origin(),
        byte_size,
        QAIC_COPY_TO_DEVICE,
        0,
    )
    qaic_submit_and_wait_jit_command_buffers(
        ctx,
        stream,
        single_submit.unsafe_ptr().as_unsafe_any_origin(),
        UInt32(1),
    )
    qaic_copy_jit_buffer(
        ctx,
        output_buf,
        host_output.bitcast[UInt8]().as_unsafe_any_origin(),
        byte_size,
        QAIC_COPY_FROM_DEVICE,
        0,
    )
    if verify_constant_region("R1", host_output, 0, num_elements, 22.0):
        passed += 1
    else:
        fail += 1
    print()

    print("Sub-test B: mutate only a=10 -> 71.0")
    fill_constant_f32(host_a, num_elements, 10.0)
    qaic_copy_jit_buffer(
        ctx,
        input_a_buf,
        host_a.bitcast[UInt8]().as_unsafe_any_origin(),
        byte_size,
        QAIC_COPY_TO_DEVICE,
        0,
    )
    qaic_submit_and_wait_jit_command_buffers(
        ctx,
        stream,
        single_submit.unsafe_ptr().as_unsafe_any_origin(),
        UInt32(1),
    )
    qaic_copy_jit_buffer(
        ctx,
        output_buf,
        host_output.bitcast[UInt8]().as_unsafe_any_origin(),
        byte_size,
        QAIC_COPY_FROM_DEVICE,
        0,
    )
    if verify_constant_region("R2", host_output, 0, num_elements, 71.0):
        passed += 1
    else:
        fail += 1
    print()

    print("Sub-test C: mutate only b=2 -> 21.0")
    fill_constant_f32(host_b, num_elements, 2.0)
    qaic_copy_jit_buffer(
        ctx,
        input_b_buf,
        host_b.bitcast[UInt8]().as_unsafe_any_origin(),
        byte_size,
        QAIC_COPY_TO_DEVICE,
        0,
    )
    qaic_submit_and_wait_jit_command_buffers(
        ctx,
        stream,
        single_submit.unsafe_ptr().as_unsafe_any_origin(),
        UInt32(1),
    )
    qaic_copy_jit_buffer(
        ctx,
        output_buf,
        host_output.bitcast[UInt8]().as_unsafe_any_origin(),
        byte_size,
        QAIC_COPY_FROM_DEVICE,
        0,
    )
    if verify_constant_region("R3", host_output, 0, num_elements, 21.0):
        passed += 1
    else:
        fail += 1
    print()

    print("Sub-test D: overwrite output with 999 before resubmit -> 21.0")
    fill_constant_f32(host_output, num_elements, 999.0)
    qaic_copy_jit_buffer(
        ctx,
        output_buf,
        host_output.bitcast[UInt8]().as_unsafe_any_origin(),
        byte_size,
        QAIC_COPY_TO_DEVICE,
        0,
    )
    qaic_submit_and_wait_jit_command_buffers(
        ctx,
        stream,
        single_submit.unsafe_ptr().as_unsafe_any_origin(),
        UInt32(1),
    )
    qaic_copy_jit_buffer(
        ctx,
        output_buf,
        host_output.bitcast[UInt8]().as_unsafe_any_origin(),
        byte_size,
        QAIC_COPY_FROM_DEVICE,
        0,
    )
    if verify_constant_region("R4", host_output, 0, num_elements, 21.0):
        passed += 1
    else:
        fail += 1
    print()

    print("Sub-test E: mutate only a=0 -> 1.0")
    fill_constant_f32(host_a, num_elements, 0.0)
    qaic_copy_jit_buffer(
        ctx,
        input_a_buf,
        host_a.bitcast[UInt8]().as_unsafe_any_origin(),
        byte_size,
        QAIC_COPY_TO_DEVICE,
        0,
    )
    qaic_submit_and_wait_jit_command_buffers(
        ctx,
        stream,
        single_submit.unsafe_ptr().as_unsafe_any_origin(),
        UInt32(1),
    )
    qaic_copy_jit_buffer(
        ctx,
        output_buf,
        host_output.bitcast[UInt8]().as_unsafe_any_origin(),
        byte_size,
        QAIC_COPY_FROM_DEVICE,
        0,
    )
    if verify_constant_region("R5", host_output, 0, num_elements, 1.0):
        passed += 1
    else:
        fail += 1
    print()

    host_a.unsafe_free()
    host_b.unsafe_free()
    host_output.unsafe_free()
    qaic_release_jit_command_buffer(ctx, cmd_buffer)
    qaic_release_jit_buffer(ctx, output_buf)
    qaic_release_jit_buffer(ctx, input_a_buf)
    qaic_release_jit_buffer(ctx, input_b_buf)

    print("Scenario 3 result:", passed, "passed,", fail, "failed")
    print_separator()
    return fail == 0


def run_local_storage_test(
    ctx: DeviceContext,
    jit_device: QAicDevice,
    stream: QAicStream,
    jit_func: QAicJitFunction,
) raises -> Bool:
    print("Scenario 4: local storage is baked at record time")
    var total_elements = 2048
    var partial_elements = 512
    var byte_size = UInt64(total_elements) * FLOAT32_BYTES
    var sentinel: Float32 = -777.0
    var cu_bitmap: QAicComputeUnitBitMap = 0x1

    print("Configuration:")
    print("  Total elements:", total_elements)
    print("  Initial baked num_elements:", partial_elements)
    print("  Initial baked bias: 5.0")
    print()

    var output_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        byte_size,
        DEFAULT_ALIGNMENT,
        ALL_COMPUTE_UNITS,
    )
    var input_a_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        byte_size,
        DEFAULT_ALIGNMENT,
        ALL_COMPUTE_UNITS,
    )
    var input_b_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        byte_size,
        DEFAULT_ALIGNMENT,
        ALL_COMPUTE_UNITS,
    )
    var input_a_host = alloc[Float32](total_elements)
    var input_b_host = alloc[Float32](total_elements)
    var output = alloc[Float32](total_elements)

    fill_constant_f32(input_a_host, total_elements, 2.0)
    qaic_copy_jit_buffer(
        ctx,
        input_a_buf,
        input_a_host.bitcast[UInt8]().as_unsafe_any_origin(),
        byte_size,
        QAIC_COPY_TO_DEVICE,
        0,
    )
    fill_constant_f32(input_b_host, total_elements, 3.0)
    qaic_copy_jit_buffer(
        ctx,
        input_b_buf,
        input_b_host.bitcast[UInt8]().as_unsafe_any_origin(),
        byte_size,
        QAIC_COPY_TO_DEVICE,
        0,
    )

    var host_bias: Float32 = 5.0
    var host_num_elements: Int = partial_elements
    var partial_byte_size = UInt64(host_num_elements) * FLOAT32_BYTES
    var cmd_buffer = qaic_create_jit_command_buffer(ctx, jit_device, cu_bitmap)
    qaic_begin_jit_command_buffer(ctx, cmd_buffer)

    var arg_indices = Array[QAicDescriptorIndex, 5](uninitialized=True)
    arg_indices[0] = qaic_push_jit_buffer_descriptor(
        ctx,
        cmd_buffer,
        output_buf,
        0,
        partial_byte_size,
        QAIC_DEV_BUF_RESIDENT_KEEP,
    )
    arg_indices[1] = qaic_push_jit_buffer_descriptor(
        ctx,
        cmd_buffer,
        input_a_buf,
        0,
        partial_byte_size,
        QAIC_DEV_BUF_RESIDENT_KEEP,
    )
    arg_indices[2] = qaic_push_jit_buffer_descriptor(
        ctx,
        cmd_buffer,
        input_b_buf,
        0,
        partial_byte_size,
        QAIC_DEV_BUF_RESIDENT_KEEP,
    )
    var initial_bias_copy = host_bias
    arg_indices[3] = qaic_push_jit_local_storage(
        ctx,
        cmd_buffer,
        UnsafePointer(to=initial_bias_copy)
        .bitcast[UInt8]()
        .as_unsafe_any_origin(),
        UInt32(size_of[Float32]()),
    )
    var initial_num_elements_u32 = UInt32(host_num_elements)
    arg_indices[4] = qaic_push_jit_local_storage(
        ctx,
        cmd_buffer,
        UnsafePointer(to=initial_num_elements_u32)
        .bitcast[UInt8]()
        .as_unsafe_any_origin(),
        UInt32(size_of[UInt32]()),
    )

    qaic_dispatch_jit_command_buffer(
        ctx,
        cmd_buffer,
        jit_func,
        arg_indices.unsafe_ptr().as_unsafe_any_origin(),
        UInt32(5),
        SINGLE_THREAD_MASK,
    )
    qaic_end_jit_command_buffer(ctx, cmd_buffer)

    var single_submit = Array[QAicCommandBuffer, 1](uninitialized=True)
    single_submit[0] = cmd_buffer

    var passed = 0
    var fail = 0

    print("Sub-test A: initial run uses baked bias=5 and num_elements=512")
    fill_constant_f32(output, total_elements, sentinel)
    qaic_copy_jit_buffer(
        ctx,
        output_buf,
        output.bitcast[UInt8]().as_unsafe_any_origin(),
        byte_size,
        QAIC_COPY_TO_DEVICE,
        0,
    )
    qaic_submit_and_wait_jit_command_buffers(
        ctx,
        stream,
        single_submit.unsafe_ptr().as_unsafe_any_origin(),
        UInt32(1),
    )
    qaic_copy_jit_buffer(
        ctx,
        output_buf,
        output.bitcast[UInt8]().as_unsafe_any_origin(),
        byte_size,
        QAIC_COPY_FROM_DEVICE,
        0,
    )
    var first_errors = count_constant_f32_mismatches(
        output, 0, partial_elements, 11.0
    )
    var rest_errors = count_constant_f32_mismatches(
        output, partial_elements, total_elements - partial_elements, sentinel
    )
    var ok = first_errors == 0 and rest_errors == 0
    print(
        "  first[0]=",
        output[0],
        " first_errors=",
        first_errors,
        " rest[512]=",
        output[partial_elements],
        " rest_errors=",
        rest_errors,
    )
    if ok:
        passed += 1
    else:
        fail += 1
    print()

    print("Sub-test B: changing host bias to 100 does not affect baked CB")
    host_bias = 100.0
    fill_constant_f32(output, total_elements, sentinel)
    qaic_copy_jit_buffer(
        ctx,
        output_buf,
        output.bitcast[UInt8]().as_unsafe_any_origin(),
        byte_size,
        QAIC_COPY_TO_DEVICE,
        0,
    )
    qaic_submit_and_wait_jit_command_buffers(
        ctx,
        stream,
        single_submit.unsafe_ptr().as_unsafe_any_origin(),
        UInt32(1),
    )
    qaic_copy_jit_buffer(
        ctx,
        output_buf,
        output.bitcast[UInt8]().as_unsafe_any_origin(),
        byte_size,
        QAIC_COPY_FROM_DEVICE,
        0,
    )
    ok = count_constant_f32_mismatches(output, 0, partial_elements, 11.0) == 0
    print(
        "  out[0]=",
        output[0],
        " expected 11.0 if immutable, 106.0 if mutable",
    )
    if ok:
        passed += 1
    else:
        fail += 1
    print()

    print(
        "Sub-test C: changing host num_elements to 2048 does not affect"
        " baked CB"
    )
    host_num_elements = total_elements
    fill_constant_f32(output, total_elements, sentinel)
    qaic_copy_jit_buffer(
        ctx,
        output_buf,
        output.bitcast[UInt8]().as_unsafe_any_origin(),
        byte_size,
        QAIC_COPY_TO_DEVICE,
        0,
    )
    qaic_submit_and_wait_jit_command_buffers(
        ctx,
        stream,
        single_submit.unsafe_ptr().as_unsafe_any_origin(),
        UInt32(1),
    )
    qaic_copy_jit_buffer(
        ctx,
        output_buf,
        output.bitcast[UInt8]().as_unsafe_any_origin(),
        byte_size,
        QAIC_COPY_FROM_DEVICE,
        0,
    )
    var written_count = total_elements - count_constant_f32_mismatches(
        output, 0, total_elements, 11.0
    )
    ok = written_count == partial_elements
    print(
        "  matches at 11.0=",
        written_count,
        " out[0]=",
        output[0],
        " out[511]=",
        output[partial_elements - 1],
        " out[512]=",
        output[partial_elements],
    )
    if ok:
        passed += 1
    else:
        fail += 1
    print()

    qaic_release_jit_command_buffer(ctx, cmd_buffer)

    print("Sub-test D: rebuilt CB observes new bias=100 and num_elements=2048")
    var rebuilt_byte_size = UInt64(host_num_elements) * FLOAT32_BYTES
    var rebuilt_cmd_buffer = qaic_create_jit_command_buffer(
        ctx, jit_device, cu_bitmap
    )
    qaic_begin_jit_command_buffer(ctx, rebuilt_cmd_buffer)

    var rebuilt_arg_indices = Array[QAicDescriptorIndex, 5](uninitialized=True)
    rebuilt_arg_indices[0] = qaic_push_jit_buffer_descriptor(
        ctx,
        rebuilt_cmd_buffer,
        output_buf,
        0,
        rebuilt_byte_size,
        QAIC_DEV_BUF_RESIDENT_KEEP,
    )
    rebuilt_arg_indices[1] = qaic_push_jit_buffer_descriptor(
        ctx,
        rebuilt_cmd_buffer,
        input_a_buf,
        0,
        rebuilt_byte_size,
        QAIC_DEV_BUF_RESIDENT_KEEP,
    )
    rebuilt_arg_indices[2] = qaic_push_jit_buffer_descriptor(
        ctx,
        rebuilt_cmd_buffer,
        input_b_buf,
        0,
        rebuilt_byte_size,
        QAIC_DEV_BUF_RESIDENT_KEEP,
    )
    var rebuilt_bias_copy = host_bias
    rebuilt_arg_indices[3] = qaic_push_jit_local_storage(
        ctx,
        rebuilt_cmd_buffer,
        UnsafePointer(to=rebuilt_bias_copy)
        .bitcast[UInt8]()
        .as_unsafe_any_origin(),
        UInt32(size_of[Float32]()),
    )
    var rebuilt_num_elements_u32 = UInt32(host_num_elements)
    rebuilt_arg_indices[4] = qaic_push_jit_local_storage(
        ctx,
        rebuilt_cmd_buffer,
        UnsafePointer(to=rebuilt_num_elements_u32)
        .bitcast[UInt8]()
        .as_unsafe_any_origin(),
        UInt32(size_of[UInt32]()),
    )

    qaic_dispatch_jit_command_buffer(
        ctx,
        rebuilt_cmd_buffer,
        jit_func,
        rebuilt_arg_indices.unsafe_ptr().as_unsafe_any_origin(),
        UInt32(5),
        SINGLE_THREAD_MASK,
    )
    qaic_end_jit_command_buffer(ctx, rebuilt_cmd_buffer)

    single_submit[0] = rebuilt_cmd_buffer
    fill_constant_f32(output, total_elements, sentinel)
    qaic_copy_jit_buffer(
        ctx,
        output_buf,
        output.bitcast[UInt8]().as_unsafe_any_origin(),
        byte_size,
        QAIC_COPY_TO_DEVICE,
        0,
    )
    qaic_submit_and_wait_jit_command_buffers(
        ctx,
        stream,
        single_submit.unsafe_ptr().as_unsafe_any_origin(),
        UInt32(1),
    )
    qaic_copy_jit_buffer(
        ctx,
        output_buf,
        output.bitcast[UInt8]().as_unsafe_any_origin(),
        byte_size,
        QAIC_COPY_FROM_DEVICE,
        0,
    )
    ok = count_constant_f32_mismatches(output, 0, total_elements, 106.0) == 0
    print(
        "  out[0]=",
        output[0],
        " out[2047]=",
        output[total_elements - 1],
        " expected 106.0",
    )
    if ok:
        passed += 1
    else:
        fail += 1
    qaic_release_jit_command_buffer(ctx, rebuilt_cmd_buffer)
    print()

    output.unsafe_free()
    input_a_host.unsafe_free()
    input_b_host.unsafe_free()
    qaic_release_jit_buffer(ctx, output_buf)
    qaic_release_jit_buffer(ctx, input_a_buf)
    qaic_release_jit_buffer(ctx, input_b_buf)

    print("Scenario 4 result:", passed, "passed,", fail, "failed")
    print_separator()
    return fail == 0


def run_batch_test(
    ctx: DeviceContext,
    jit_device: QAicDevice,
    stream: QAicStream,
    jit_func: QAicJitFunction,
) raises -> Bool:
    print("Scenario 5: batch submission with shared and partitioned buffers")
    var passed = 0
    var fail = 0

    print("Sub-test A: shared RESIDENT_KEEP inputs with separate outputs")
    var shared_elements = 4096
    var shared_bytes = UInt64(shared_elements) * FLOAT32_BYTES
    var a_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        shared_bytes,
        DEFAULT_ALIGNMENT,
        ALL_COMPUTE_UNITS,
    )
    var b_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        shared_bytes,
        DEFAULT_ALIGNMENT,
        ALL_COMPUTE_UNITS,
    )
    var out0_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        shared_bytes,
        DEFAULT_ALIGNMENT,
        ALL_COMPUTE_UNITS,
    )
    var out1_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        shared_bytes,
        DEFAULT_ALIGNMENT,
        ALL_COMPUTE_UNITS,
    )
    var a_host = alloc[Float32](shared_elements)
    var b_host = alloc[Float32](shared_elements)
    var out0 = alloc[Float32](shared_elements)
    var out1 = alloc[Float32](shared_elements)

    fill_constant_f32(a_host, shared_elements, 3.0)
    qaic_copy_jit_buffer(
        ctx,
        a_buf,
        a_host.bitcast[UInt8]().as_unsafe_any_origin(),
        shared_bytes,
        QAIC_COPY_TO_DEVICE,
        0,
    )
    fill_constant_f32(b_host, shared_elements, 5.0)
    qaic_copy_jit_buffer(
        ctx,
        b_buf,
        b_host.bitcast[UInt8]().as_unsafe_any_origin(),
        shared_bytes,
        QAIC_COPY_TO_DEVICE,
        0,
    )
    fill_constant_f32(out0, shared_elements, 0.0)
    qaic_copy_jit_buffer(
        ctx,
        out0_buf,
        out0.bitcast[UInt8]().as_unsafe_any_origin(),
        shared_bytes,
        QAIC_COPY_TO_DEVICE,
        0,
    )
    fill_constant_f32(out1, shared_elements, 0.0)
    qaic_copy_jit_buffer(
        ctx,
        out1_buf,
        out1.bitcast[UInt8]().as_unsafe_any_origin(),
        shared_bytes,
        QAIC_COPY_TO_DEVICE,
        0,
    )

    var shared_batch = Array[QAicCommandBuffer, 2](uninitialized=True)
    shared_batch[0] = qaic_create_jit_command_buffer(ctx, jit_device, 0x1)
    qaic_begin_jit_command_buffer(ctx, shared_batch[0])

    var shared0_arg_indices = Array[QAicDescriptorIndex, 5](uninitialized=True)
    shared0_arg_indices[0] = qaic_push_jit_buffer_descriptor(
        ctx,
        shared_batch[0],
        out0_buf,
        0,
        shared_bytes,
        QAIC_DEV_BUF_RESIDENT_KEEP,
    )
    shared0_arg_indices[1] = qaic_push_jit_buffer_descriptor(
        ctx,
        shared_batch[0],
        a_buf,
        0,
        shared_bytes,
        QAIC_DEV_BUF_RESIDENT_KEEP,
    )
    shared0_arg_indices[2] = qaic_push_jit_buffer_descriptor(
        ctx,
        shared_batch[0],
        b_buf,
        0,
        shared_bytes,
        QAIC_DEV_BUF_RESIDENT_KEEP,
    )
    var shared0_bias: Float32 = 1.0
    shared0_arg_indices[3] = qaic_push_jit_local_storage(
        ctx,
        shared_batch[0],
        UnsafePointer(to=shared0_bias).bitcast[UInt8]().as_unsafe_any_origin(),
        UInt32(size_of[Float32]()),
    )
    var shared_num_elements_u32 = UInt32(shared_elements)
    shared0_arg_indices[4] = qaic_push_jit_local_storage(
        ctx,
        shared_batch[0],
        UnsafePointer(to=shared_num_elements_u32)
        .bitcast[UInt8]()
        .as_unsafe_any_origin(),
        UInt32(size_of[UInt32]()),
    )

    qaic_dispatch_jit_command_buffer(
        ctx,
        shared_batch[0],
        jit_func,
        shared0_arg_indices.unsafe_ptr().as_unsafe_any_origin(),
        UInt32(5),
        SINGLE_THREAD_MASK,
    )
    qaic_end_jit_command_buffer(ctx, shared_batch[0])

    shared_batch[1] = qaic_create_jit_command_buffer(ctx, jit_device, 0x2)
    qaic_begin_jit_command_buffer(ctx, shared_batch[1])

    var shared1_arg_indices = Array[QAicDescriptorIndex, 5](uninitialized=True)
    shared1_arg_indices[0] = qaic_push_jit_buffer_descriptor(
        ctx,
        shared_batch[1],
        out1_buf,
        0,
        shared_bytes,
        QAIC_DEV_BUF_RESIDENT_KEEP,
    )
    shared1_arg_indices[1] = qaic_push_jit_buffer_descriptor(
        ctx,
        shared_batch[1],
        a_buf,
        0,
        shared_bytes,
        QAIC_DEV_BUF_RESIDENT_KEEP,
    )
    shared1_arg_indices[2] = qaic_push_jit_buffer_descriptor(
        ctx,
        shared_batch[1],
        b_buf,
        0,
        shared_bytes,
        QAIC_DEV_BUF_RESIDENT_KEEP,
    )
    var shared1_bias: Float32 = 2.0
    shared1_arg_indices[3] = qaic_push_jit_local_storage(
        ctx,
        shared_batch[1],
        UnsafePointer(to=shared1_bias).bitcast[UInt8]().as_unsafe_any_origin(),
        UInt32(size_of[Float32]()),
    )
    shared1_arg_indices[4] = qaic_push_jit_local_storage(
        ctx,
        shared_batch[1],
        UnsafePointer(to=shared_num_elements_u32)
        .bitcast[UInt8]()
        .as_unsafe_any_origin(),
        UInt32(size_of[UInt32]()),
    )

    qaic_dispatch_jit_command_buffer(
        ctx,
        shared_batch[1],
        jit_func,
        shared1_arg_indices.unsafe_ptr().as_unsafe_any_origin(),
        UInt32(5),
        SINGLE_THREAD_MASK,
    )
    qaic_end_jit_command_buffer(ctx, shared_batch[1])

    qaic_submit_and_wait_jit_command_buffers(
        ctx, stream, shared_batch.unsafe_ptr().as_unsafe_any_origin(), UInt32(2)
    )
    qaic_copy_jit_buffer(
        ctx,
        out0_buf,
        out0.bitcast[UInt8]().as_unsafe_any_origin(),
        shared_bytes,
        QAIC_COPY_FROM_DEVICE,
        0,
    )
    qaic_copy_jit_buffer(
        ctx,
        out1_buf,
        out1.bitcast[UInt8]().as_unsafe_any_origin(),
        shared_bytes,
        QAIC_COPY_FROM_DEVICE,
        0,
    )
    var ok = verify_constant_region(
        "NSP0 initial", out0, 0, shared_elements, 16.0
    ) and verify_constant_region("NSP1 initial", out1, 0, shared_elements, 17.0)
    if ok:
        passed += 1
    else:
        fail += 1
    print()

    print("Sub-test B: re-submit same batch after mutating shared input a=4")
    fill_constant_f32(a_host, shared_elements, 4.0)
    qaic_copy_jit_buffer(
        ctx,
        a_buf,
        a_host.bitcast[UInt8]().as_unsafe_any_origin(),
        shared_bytes,
        QAIC_COPY_TO_DEVICE,
        0,
    )
    qaic_submit_and_wait_jit_command_buffers(
        ctx, stream, shared_batch.unsafe_ptr().as_unsafe_any_origin(), UInt32(2)
    )
    qaic_copy_jit_buffer(
        ctx,
        out0_buf,
        out0.bitcast[UInt8]().as_unsafe_any_origin(),
        shared_bytes,
        QAIC_COPY_FROM_DEVICE,
        0,
    )
    qaic_copy_jit_buffer(
        ctx,
        out1_buf,
        out1.bitcast[UInt8]().as_unsafe_any_origin(),
        shared_bytes,
        QAIC_COPY_FROM_DEVICE,
        0,
    )
    ok = verify_constant_region(
        "NSP0 reused", out0, 0, shared_elements, 21.0
    ) and verify_constant_region("NSP1 reused", out1, 0, shared_elements, 22.0)
    if ok:
        passed += 1
    else:
        fail += 1
    out0.unsafe_free()
    out1.unsafe_free()
    a_host.unsafe_free()
    b_host.unsafe_free()
    qaic_release_jit_command_buffer(ctx, shared_batch[0])
    qaic_release_jit_command_buffer(ctx, shared_batch[1])
    qaic_release_jit_buffer(ctx, a_buf)
    qaic_release_jit_buffer(ctx, b_buf)
    qaic_release_jit_buffer(ctx, out0_buf)
    qaic_release_jit_buffer(ctx, out1_buf)
    print()

    print("Sub-test C: offset-partitioned batch over one shared buffer")
    var region_elements = 4096
    var region_bytes = UInt64(region_elements) * FLOAT32_BYTES
    var total_bytes = region_bytes * 6
    var partitioned_buf = qaic_allocate_jit_buffer(
        ctx,
        jit_device,
        QAIC_BUFFER_TYPE_DEV_DDR,
        QAIC_DEV_BUF_IO_RW,
        total_bytes,
        DEFAULT_ALIGNMENT,
        ALL_COMPUTE_UNITS,
    )
    var partitioned_upload = alloc[Float32](region_elements)
    var region0 = alloc[Float32](region_elements)
    var region1 = alloc[Float32](region_elements)

    fill_constant_f32(partitioned_upload, region_elements, 2.0)
    qaic_copy_jit_buffer(
        ctx,
        partitioned_buf,
        partitioned_upload.bitcast[UInt8]().as_unsafe_any_origin(),
        region_bytes,
        QAIC_COPY_TO_DEVICE,
        0,
    )
    fill_constant_f32(partitioned_upload, region_elements, 4.0)
    qaic_copy_jit_buffer(
        ctx,
        partitioned_buf,
        partitioned_upload.bitcast[UInt8]().as_unsafe_any_origin(),
        region_bytes,
        QAIC_COPY_TO_DEVICE,
        region_bytes,
    )
    fill_constant_f32(partitioned_upload, region_elements, 0.0)
    qaic_copy_jit_buffer(
        ctx,
        partitioned_buf,
        partitioned_upload.bitcast[UInt8]().as_unsafe_any_origin(),
        region_bytes,
        QAIC_COPY_TO_DEVICE,
        region_bytes * 2,
    )
    fill_constant_f32(partitioned_upload, region_elements, 7.0)
    qaic_copy_jit_buffer(
        ctx,
        partitioned_buf,
        partitioned_upload.bitcast[UInt8]().as_unsafe_any_origin(),
        region_bytes,
        QAIC_COPY_TO_DEVICE,
        region_bytes * 3,
    )
    fill_constant_f32(partitioned_upload, region_elements, 3.0)
    qaic_copy_jit_buffer(
        ctx,
        partitioned_buf,
        partitioned_upload.bitcast[UInt8]().as_unsafe_any_origin(),
        region_bytes,
        QAIC_COPY_TO_DEVICE,
        region_bytes * 4,
    )
    fill_constant_f32(partitioned_upload, region_elements, 0.0)
    qaic_copy_jit_buffer(
        ctx,
        partitioned_buf,
        partitioned_upload.bitcast[UInt8]().as_unsafe_any_origin(),
        region_bytes,
        QAIC_COPY_TO_DEVICE,
        region_bytes * 5,
    )

    var partitioned_batch = Array[QAicCommandBuffer, 2](uninitialized=True)
    partitioned_batch[0] = qaic_create_jit_command_buffer(ctx, jit_device, 0x1)
    qaic_begin_jit_command_buffer(ctx, partitioned_batch[0])

    var partitioned0_arg_indices = Array[QAicDescriptorIndex, 5](
        uninitialized=True
    )
    partitioned0_arg_indices[0] = qaic_push_jit_buffer_descriptor(
        ctx,
        partitioned_batch[0],
        partitioned_buf,
        region_bytes * 2,
        region_bytes,
        QAIC_DEV_BUF_RESIDENT_KEEP,
    )
    partitioned0_arg_indices[1] = qaic_push_jit_buffer_descriptor(
        ctx,
        partitioned_batch[0],
        partitioned_buf,
        0,
        region_bytes,
        QAIC_DEV_BUF_RESIDENT_KEEP,
    )
    partitioned0_arg_indices[2] = qaic_push_jit_buffer_descriptor(
        ctx,
        partitioned_batch[0],
        partitioned_buf,
        region_bytes,
        region_bytes,
        QAIC_DEV_BUF_RESIDENT_KEEP,
    )
    var partitioned0_bias: Float32 = 10.0
    partitioned0_arg_indices[3] = qaic_push_jit_local_storage(
        ctx,
        partitioned_batch[0],
        UnsafePointer(to=partitioned0_bias)
        .bitcast[UInt8]()
        .as_unsafe_any_origin(),
        UInt32(size_of[Float32]()),
    )
    var region_elements_u32 = UInt32(region_elements)
    partitioned0_arg_indices[4] = qaic_push_jit_local_storage(
        ctx,
        partitioned_batch[0],
        UnsafePointer(to=region_elements_u32)
        .bitcast[UInt8]()
        .as_unsafe_any_origin(),
        UInt32(size_of[UInt32]()),
    )

    qaic_dispatch_jit_command_buffer(
        ctx,
        partitioned_batch[0],
        jit_func,
        partitioned0_arg_indices.unsafe_ptr().as_unsafe_any_origin(),
        UInt32(5),
        SINGLE_THREAD_MASK,
    )
    qaic_end_jit_command_buffer(ctx, partitioned_batch[0])

    partitioned_batch[1] = qaic_create_jit_command_buffer(ctx, jit_device, 0x2)
    qaic_begin_jit_command_buffer(ctx, partitioned_batch[1])

    var partitioned1_arg_indices = Array[QAicDescriptorIndex, 5](
        uninitialized=True
    )
    partitioned1_arg_indices[0] = qaic_push_jit_buffer_descriptor(
        ctx,
        partitioned_batch[1],
        partitioned_buf,
        region_bytes * 5,
        region_bytes,
        QAIC_DEV_BUF_RESIDENT_KEEP,
    )
    partitioned1_arg_indices[1] = qaic_push_jit_buffer_descriptor(
        ctx,
        partitioned_batch[1],
        partitioned_buf,
        region_bytes * 3,
        region_bytes,
        QAIC_DEV_BUF_RESIDENT_KEEP,
    )
    partitioned1_arg_indices[2] = qaic_push_jit_buffer_descriptor(
        ctx,
        partitioned_batch[1],
        partitioned_buf,
        region_bytes * 4,
        region_bytes,
        QAIC_DEV_BUF_RESIDENT_KEEP,
    )
    var partitioned1_bias: Float32 = -1.0
    partitioned1_arg_indices[3] = qaic_push_jit_local_storage(
        ctx,
        partitioned_batch[1],
        UnsafePointer(to=partitioned1_bias)
        .bitcast[UInt8]()
        .as_unsafe_any_origin(),
        UInt32(size_of[Float32]()),
    )
    partitioned1_arg_indices[4] = qaic_push_jit_local_storage(
        ctx,
        partitioned_batch[1],
        UnsafePointer(to=region_elements_u32)
        .bitcast[UInt8]()
        .as_unsafe_any_origin(),
        UInt32(size_of[UInt32]()),
    )

    qaic_dispatch_jit_command_buffer(
        ctx,
        partitioned_batch[1],
        jit_func,
        partitioned1_arg_indices.unsafe_ptr().as_unsafe_any_origin(),
        UInt32(5),
        SINGLE_THREAD_MASK,
    )
    qaic_end_jit_command_buffer(ctx, partitioned_batch[1])

    qaic_submit_and_wait_jit_command_buffers(
        ctx,
        stream,
        partitioned_batch.unsafe_ptr().as_unsafe_any_origin(),
        UInt32(2),
    )
    qaic_copy_jit_buffer(
        ctx,
        partitioned_buf,
        region0.bitcast[UInt8]().as_unsafe_any_origin(),
        region_bytes,
        QAIC_COPY_FROM_DEVICE,
        region_bytes * 2,
    )
    qaic_copy_jit_buffer(
        ctx,
        partitioned_buf,
        region1.bitcast[UInt8]().as_unsafe_any_origin(),
        region_bytes,
        QAIC_COPY_FROM_DEVICE,
        region_bytes * 5,
    )
    ok = verify_constant_region(
        "Partitioned NSP0", region0, 0, region_elements, 18.0
    ) and verify_constant_region(
        "Partitioned NSP1", region1, 0, region_elements, 20.0
    )
    if ok:
        passed += 1
    else:
        fail += 1
    partitioned_upload.unsafe_free()
    region0.unsafe_free()
    region1.unsafe_free()
    qaic_release_jit_command_buffer(ctx, partitioned_batch[0])
    qaic_release_jit_command_buffer(ctx, partitioned_batch[1])
    qaic_release_jit_buffer(ctx, partitioned_buf)
    print()

    print("Scenario 5 result:", passed, "passed,", fail, "failed")
    print_separator()
    return fail == 0


def run_named_command_buffer_scenario(group: String) raises -> Bool:
    var ctx = DeviceContext(device_id=0, api="qaic")
    var jit_device = qaic_get_jit_device(ctx)
    var stream = qaic_create_jit_stream(ctx, jit_device)
    var elf = compile_info_npu[scalar_muladd_kernel]()
    var jit_func = qaic_load_jit_function(
        ctx,
        elf.function_name,
        elf.asm.unsafe_ptr().bitcast[UInt8]().as_unsafe_any_origin(),
        UInt64(elf.asm.byte_length()),
    )

    var ok = False
    if group == "cubitmap":
        ok = run_cubitmap_validation_test(ctx, jit_device, stream, jit_func)
    elif group == "reuse":
        ok = run_reuse_test(ctx, jit_device, stream, jit_func)
    elif group == "mutable-buffers":
        ok = run_mutable_device_buffer_test(ctx, jit_device, stream, jit_func)
    elif group == "local-storage":
        ok = run_local_storage_test(ctx, jit_device, stream, jit_func)
    elif group == "batch":
        ok = run_batch_test(ctx, jit_device, stream, jit_func)
    else:
        print("Unknown command-buffer scenario:", group)

    qaic_release_jit_function(jit_func)
    qaic_release_jit_stream(ctx, stream)
    return ok


def assert_command_buffer_scenario(title: String, group: String) raises:
    print_test_header(title)
    assert_true(run_named_command_buffer_scenario(group))


def test_cubitmap_validation() raises:
    assert_command_buffer_scenario(
        "QAIC Command Buffer: cuBitMap validation",
        "cubitmap",
    )


def test_command_buffer_reuse() raises:
    assert_command_buffer_scenario(
        "QAIC Command Buffer: command-buffer reuse",
        "reuse",
    )


def test_mutable_device_buffers() raises:
    assert_command_buffer_scenario(
        "QAIC Command Buffer: mutable device buffers",
        "mutable-buffers",
    )


def test_local_storage() raises:
    assert_command_buffer_scenario(
        "QAIC Command Buffer: baked local storage",
        "local-storage",
    )


def test_batch_submission() raises:
    assert_command_buffer_scenario(
        "QAIC Command Buffer: batch submission",
        "batch",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
