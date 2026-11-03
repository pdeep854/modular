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
"""Standalone NPU context stub for the hexagon-ondevice-generic example.

A self-contained module that works with the pip-installed mojo (1.2.0+).
Uses only stable 1.2.0 stdlib APIs:
  - std.ffi.external_call / c_uint
  - std.memory.Pointer (via Pointer(to=var) for address capture)
  - std.collections.List for buffer storage (gives unsafe_ptr() and [])
  - Int for opaque FFI handle storage
  - UInt64 as external_call return for null/non-null error checks

Provides: DeviceContext, HostBuffer, DeviceBuffer, DeviceExternalFunction, Dim.
"""

from std.ffi import external_call, c_uint
from std.memory import Pointer
from std.collections import List
from std.sys import size_of


# ---------------------------------------------------------------------------
# Dim
# ---------------------------------------------------------------------------


struct Dim(Copyable):
    """Grid or block dimension (one, two, or three axes)."""

    var _x: Int
    var _y: Int
    var _z: Int

    def __init__(out self, x: Int):
        self._x = x
        self._y = 1
        self._z = 1

    def __init__(out self, x: Int, y: Int):
        self._x = x
        self._y = y
        self._z = 1

    def __init__(out self, x: Int, y: Int, z: Int):
        self._x = x
        self._y = y
        self._z = z

    def x(self) -> Int:
        return self._x

    def y(self) -> Int:
        return self._y

    def z(self) -> Int:
        return self._z


# ---------------------------------------------------------------------------
# HostBuffer
# ---------------------------------------------------------------------------


struct HostBuffer[dtype: DType](Copyable):
    """Host-accessible buffer backed by a List allocation."""

    var _data: List[Scalar[Self.dtype]]

    def __init__(out self, n: Int):
        self._data = List[Scalar[Self.dtype]](capacity=n)
        for _ in range(n):
            self._data.append(Scalar[Self.dtype](0))

    def __getitem__(self, i: Int) -> Scalar[Self.dtype]:
        return self._data[i]

    def __setitem__(mut self, i: Int, v: Scalar[Self.dtype]):
        self._data[i] = v

    def unsafe_ptr(self) -> UInt64:
        """Returns the raw address of the backing storage."""
        return UInt64(Int(self._data.unsafe_ptr()))

    def size(self) -> Int:
        return len(self._data)


# ---------------------------------------------------------------------------
# DeviceBuffer
# ---------------------------------------------------------------------------


struct DeviceBuffer[dtype: DType](Copyable):
    """On-device buffer backed by an AsyncRT rpcmem allocation.

    Memory is allocated by the DeviceContext via rpcmem_alloc so that the NPU
    kernel dispatch path can locate the buffer by address through
    allocationFrom().  Plain heap memory (e.g. List) is rejected at dispatch
    time because the QuRT device only tracks rpcmem-allocated regions.
    """

    var _ctx: Int      # DeviceContext* stored as Int for FFI
    var _ptr: UInt64   # rpcmem address returned by createBuffer_async
    var _n: Int        # element count

    def __init__(out self, ctx: Int, n: Int) raises:
        var buf: Int = 0
        var ptr: Int = 0
        var err = external_call[
            "AsyncRT_DeviceContext_createBuffer_async",
            UInt64,
        ](
            Pointer(to=buf),
            Pointer(to=ptr),
            ctx,
            n,
            size_of[Scalar[Self.dtype]](),
        )
        if err != 0:
            raise Error("AsyncRT_DeviceContext_createBuffer_async failed")
        self._ctx = ctx
        self._ptr = UInt64(ptr)
        self._n = n

    def unsafe_ptr(self) -> UInt64:
        """Returns the rpcmem address passed to the NPU kernel as a buffer arg."""
        return self._ptr

    def size(self) -> Int:
        return self._n

    def enqueue_copy_from(mut self, src: HostBuffer[Self.dtype]):
        var bytes = self._n * size_of[Scalar[Self.dtype]]()
        _ = external_call["AsyncRT_DeviceContext_HtoD_async_sized", UInt64](
            self._ctx, self._ptr, src.unsafe_ptr(), bytes
        )

    def enqueue_copy_to(self, mut dst: HostBuffer[Self.dtype]):
        var bytes = self._n * size_of[Scalar[Self.dtype]]()
        _ = external_call["AsyncRT_DeviceContext_DtoH_async_sized", UInt64](
            self._ctx, dst.unsafe_ptr(), self._ptr, bytes
        )


# ---------------------------------------------------------------------------
# DeviceExternalFunction
# ---------------------------------------------------------------------------


struct DeviceExternalFunction(Copyable):
    """Handle to a kernel resolved by name via AsyncRT loadFunction."""

    var _handle: Int

    def __init__(out self, handle: Int):
        self._handle = handle


# ---------------------------------------------------------------------------
# DeviceContext
# ---------------------------------------------------------------------------


struct DeviceContext(Copyable):
    """Host-side handle to the NPU (Hexagon CDSP) via AsyncRT."""

    var _handle: Int

    def __init__(out self, *, api: String) raises:
        """Creates an AsyncRT device context for the given API string.

        Args:
            api: Backend name, e.g. "npu".

        Raises:
            Error if AsyncRT_DeviceContext_create returns a non-null error.
        """
        var handle: Int = 0
        var err = external_call[
            "AsyncRT_DeviceContext_create",
            UInt64,
        ](
            Pointer(to=handle),
            api.unsafe_ptr(),
            Int32(0),
        )
        if err != 0:
            raise Error("AsyncRT_DeviceContext_create failed")
        self._handle = handle

    def enqueue_create_host_buffer[dtype: DType](
        self, n: Int
    ) -> HostBuffer[dtype]:
        """Allocates a host-accessible buffer of `n` elements.

        Args:
            n: Number of elements.

        Returns:
            A `HostBuffer` wrapping the allocation.
        """
        return HostBuffer[dtype](n)

    def enqueue_create_buffer[dtype: DType](
        self, n: Int
    ) raises -> DeviceBuffer[dtype]:
        """Allocates a device buffer of `n` elements via rpcmem.

        Args:
            n: Number of elements.

        Returns:
            A `DeviceBuffer` wrapping the rpcmem allocation.

        Raises:
            Error if AsyncRT_DeviceContext_createBuffer_async fails.
        """
        return DeviceBuffer[dtype](self._handle, n)

    def synchronize(self) raises:
        """Blocks until all enqueued NPU operations complete.

        Raises:
            Error if AsyncRT_DeviceContext_synchronize returns a non-null error.
        """
        var err = external_call[
            "AsyncRT_DeviceContext_synchronize",
            UInt64,
        ](self._handle)
        if err != 0:
            raise Error("AsyncRT_DeviceContext_synchronize failed")

    def load_function[
        func_type: TrivialRegisterPassable,
        //,
        sig: func_type,
        /,
    ](
        self,
        *,
        function_name: String,
        asm: String,
    ) raises -> DeviceExternalFunction:
        """Resolves a kernel by name from NPUKernels.def via AsyncRT.

        Parameters:
            sig: Unused type parameter; kept for API compatibility.

        Args:
            function_name: Name of the NPU kernel selector to resolve.
            asm: Optional assembly bytes; pass `""` for name-only lookup.

        Returns:
            A `DeviceExternalFunction` wrapping the resolved kernel handle.

        Raises:
            Error if AsyncRT_DeviceContext_loadFunction returns a non-null error.
        """
        var func_handle: Int = 0
        var empty = String("")
        var err = external_call[
            "AsyncRT_DeviceContext_loadFunction",
            UInt64,
        ](
            Pointer(to=func_handle),
            self._handle,
            empty.unsafe_ptr(),
            function_name.unsafe_ptr(),
            asm.unsafe_ptr(),
            Int(asm.byte_length()),
            Int32(0),
            empty.unsafe_ptr(),
            Int32(3),
        )
        if err != 0:
            raise Error("AsyncRT_DeviceContext_loadFunction failed")
        return DeviceExternalFunction(func_handle)

    def enqueue_function[*Ts: AnyType](
        self,
        f: DeviceExternalFunction,
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
    ) raises:
        """Dispatches `f` on the NPU with the given grid/block dimensions.

        Parameters:
            *Ts: Compile-time types of the variadic arguments.

        Args:
            f: Kernel handle from `load_function`.
            *args: Arguments forwarded to the kernel.
            grid_dim: Number of grid cores to launch.
            block_dim: Thread mask (low bits map to CDSP hardware threads).

        Raises:
            Error if AsyncRT_DeviceContext_enqueueFunctionDirect fails.
        """
        comptime num_args = Ts.length
        var dense_args = List[UInt64](capacity=num_args)
        # argSizes must be non-null when the kernel has scalar arguments:
        # QuRTDeviceContext packs scalars by byte width using this array.
        var arg_sizes = List[UInt64](capacity=num_args)
        comptime for i in range(num_args):
            dense_args.append(UInt64(Int(Pointer(to=args[i]))))
            arg_sizes.append(UInt64(size_of[Ts[i]]()))
        var err = external_call[
            "AsyncRT_DeviceContext_enqueueFunctionDirect",
            UInt64,
        ](
            self._handle,
            f._handle,
            c_uint(grid_dim._x),
            c_uint(1),
            c_uint(1),
            c_uint(block_dim._x),
            c_uint(1),
            c_uint(1),
            c_uint(0),
            UInt64(0),
            c_uint(0),
            dense_args.unsafe_ptr(),
            c_uint(num_args),
            arg_sizes.unsafe_ptr(),
        )
        if err != 0:
            raise Error("AsyncRT_DeviceContext_enqueueFunctionDirect failed")
