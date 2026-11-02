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
"""Minimal DeviceContext/Dim stubs for NPU testing on Mojo 1.3.0.dev0.

Provides only the API surface used by the NPU integration tests:
  DeviceContext(api="npu"), HostBuffer, DeviceBuffer, DeviceExternalFunction,
  Dim, enqueue_create_host_buffer, enqueue_create_buffer, enqueue_copy_from,
  enqueue_copy_to, synchronize, load_function, enqueue_function.

All other DeviceContext functionality (GPU kernels, CUDA, HIP, Metal) is
deliberately absent here; use the full device_context.mojo for that.
"""

from memory import UnsafePointer
from sys import external_call, sizeof
from sys.ffi import OpaquePointer


# ---------------------------------------------------------------------------
# Opaque C-runtime handle types
# ---------------------------------------------------------------------------

alias _CtxPtr = OpaquePointer
alias _FuncPtr = OpaquePointer


def _cstr(s: String) -> UnsafePointer[UInt8]:
    return s.unsafe_ptr()


def _checked(err: UnsafePointer[UInt8]) raises:
    if not err:
        return
    raise Error(String(StringRef(err)))


# ---------------------------------------------------------------------------
# Dim
# ---------------------------------------------------------------------------


struct Dim(Copyable):
    """Grid / block dimension (1-D, 2-D, or 3-D)."""

    var _x: Int32
    var _y: Int32
    var _z: Int32

    def __init__(out self, x: Int):
        self._x = Int32(x)
        self._y = Int32(1)
        self._z = Int32(1)

    def __init__(out self, x: Int, y: Int):
        self._x = Int32(x)
        self._y = Int32(y)
        self._z = Int32(1)

    def __init__(out self, x: Int, y: Int, z: Int):
        self._x = Int32(x)
        self._y = Int32(y)
        self._z = Int32(z)

    def x(self) -> Int32:
        return self._x

    def y(self) -> Int32:
        return self._y

    def z(self) -> Int32:
        return self._z


# ---------------------------------------------------------------------------
# HostBuffer — pinned host memory accessible by both host and NPU
# ---------------------------------------------------------------------------


struct HostBuffer[dtype: DType]:
    var _ptr: UnsafePointer[Scalar[dtype]]
    var _n: Int

    def __init__(out self, ptr: UnsafePointer[Scalar[dtype]], n: Int):
        self._ptr = ptr
        self._n = n

    def __getitem__(self, i: Int) -> Scalar[dtype]:
        return self._ptr[i]

    def __setitem__(mut self, i: Int, v: Scalar[dtype]):
        self._ptr[i] = v

    def unsafe_ptr(self) -> UnsafePointer[Scalar[dtype]]:
        return self._ptr


# ---------------------------------------------------------------------------
# DeviceBuffer — on-device memory
# ---------------------------------------------------------------------------


struct DeviceBuffer[dtype: DType]:
    var _ptr: UnsafePointer[Scalar[dtype]]
    var _n: Int

    def __init__(out self, ptr: UnsafePointer[Scalar[dtype]], n: Int):
        self._ptr = ptr
        self._n = n

    def unsafe_ptr(self) -> UnsafePointer[Scalar[dtype]]:
        return self._ptr

    def enqueue_copy_from(self, src: HostBuffer[dtype]):
        pass  # no-op: on NPU, host and device share memory

    def enqueue_copy_to(self, mut dst: HostBuffer[dtype]):
        pass  # no-op: on NPU, host and device share memory


# ---------------------------------------------------------------------------
# DeviceExternalFunction
# ---------------------------------------------------------------------------


struct DeviceExternalFunction:
    var _handle: _FuncPtr

    def __init__(out self, handle: _FuncPtr):
        self._handle = handle


# ---------------------------------------------------------------------------
# DeviceContext — wraps AsyncRT_DeviceContext
# ---------------------------------------------------------------------------


struct DeviceContext:
    var _handle: _CtxPtr

    def __init__(out self, device_id: Int = 0, *, var api: String = "npu") raises:
        var result: _CtxPtr = _CtxPtr()
        var api_cstr = api.unsafe_ptr()
        _checked(
            external_call[
                "AsyncRT_DeviceContext_create",
                UnsafePointer[UInt8],
                UnsafePointer[_CtxPtr],
                UnsafePointer[UInt8],
                Int32,
            ](
                UnsafePointer(to=result),
                api_cstr,
                Int32(device_id),
            )
        )
        self._handle = result

    def enqueue_create_host_buffer[
        dtype: DType
    ](self, n: Int) -> HostBuffer[dtype]:
        var bytes = n * sizeof[Scalar[dtype]]()
        var ptr = external_call[
            "AsyncRT_DeviceContext_allocateHostBuffer",
            UnsafePointer[Scalar[dtype]],
            _CtxPtr,
            Int,
        ](self._handle, bytes)
        return HostBuffer[dtype](ptr, n)

    def enqueue_create_buffer[
        dtype: DType
    ](self, n: Int) -> DeviceBuffer[dtype]:
        var bytes = n * sizeof[Scalar[dtype]]()
        var ptr = external_call[
            "AsyncRT_DeviceContext_allocateBuffer",
            UnsafePointer[Scalar[dtype]],
            _CtxPtr,
            Int,
        ](self._handle, bytes)
        return DeviceBuffer[dtype](ptr, n)

    def synchronize(self) raises:
        _checked(
            external_call[
                "AsyncRT_DeviceContext_synchronize",
                UnsafePointer[UInt8],
                _CtxPtr,
            ](self._handle)
        )

    def load_function[
        sig: AnyType
    ](
        self,
        *,
        function_name: String,
        asm: String,
    ) raises -> DeviceExternalFunction:
        var func_result: _FuncPtr = _FuncPtr()
        var empty = String("")
        _checked(
            external_call[
                "AsyncRT_DeviceContext_loadFunction",
                UnsafePointer[UInt8],
                UnsafePointer[_FuncPtr],
                _CtxPtr,
                UnsafePointer[UInt8],
                UnsafePointer[UInt8],
                UnsafePointer[UInt8],
                Int,
                Int32,
                UnsafePointer[UInt8],
                Int32,
            ](
                UnsafePointer(to=func_result),
                self._handle,
                empty.unsafe_ptr(),           # module_name
                function_name.unsafe_ptr(),   # function_name
                asm.unsafe_ptr(),             # data
                Int(asm.byte_length()),       # dataLen
                Int32(0),                     # maxDynamicSharedBytes
                empty.unsafe_ptr(),           # debugLevel
                Int32(3),                     # optimizationLevel
            )
        )
        return DeviceExternalFunction(func_result)

    def enqueue_function[
        *Ts: AnyType
    ](
        self,
        f: DeviceExternalFunction,
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
    ) raises:
        # Pack each argument as an OpaquePointer to its value on the stack.
        var n = __mlir_op.`pop.variadic.size`(args)
        var ptrs = UnsafePointer[OpaquePointer].alloc(n)
        # We can't iterate variadic args in a loop in 1.3.0; use a comptime unroll.

        @parameter
        for i in range(n):
            var arg = args[i]
            ptrs[i] = OpaquePointer(UnsafePointer(to=arg).bitcast[NoneType]())

        _checked(
            external_call[
                "AsyncRT_DeviceContext_enqueueFunctionDirect",
                UnsafePointer[UInt8],
                _CtxPtr,
                _FuncPtr,
                UInt32,
                UInt32,
                UInt32,
                UInt32,
                UInt32,
                UInt32,
                UInt32,
                OpaquePointer,
                UInt32,
                UnsafePointer[OpaquePointer],
                UInt32,
                OpaquePointer,
            ](
                self._handle,
                f._handle,
                UInt32(grid_dim.x()),
                UInt32(grid_dim.y()),
                UInt32(grid_dim.z()),
                UInt32(block_dim.x()),
                UInt32(block_dim.y()),
                UInt32(block_dim.z()),
                UInt32(0),        # shared_mem_bytes
                OpaquePointer(),  # attributes (null)
                UInt32(0),        # num_attributes
                ptrs,
                UInt32(n),
                OpaquePointer(),  # arg_sizes (null)
            )
        )
        ptrs.free()
