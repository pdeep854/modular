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
"""Minimal DeviceContext stub for NPU testing on Mojo 1.3.0.dev0.

Provides: DeviceContext, HostBuffer, DeviceBuffer, DeviceExternalFunction, Dim.
Only the subset of the API used by the hexagon on-device NPU tests is here.
"""

from std.collections import InlineArray
from std.ffi import external_call, c_uint
from std.memory import Pointer
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
    """Host-accessible buffer.

    On Hexagon/CDSP the host and device share physical memory, so the same
    allocation serves both `enqueue_create_host_buffer` and
    `enqueue_create_buffer`.
    """

    var _ptr: Pointer[Scalar[Self.dtype], MutAnyOrigin]
    var _n: Int

    def __init__(
        out self,
        ptr: Pointer[Scalar[Self.dtype], MutAnyOrigin],
        n: Int,
    ):
        self._ptr = ptr
        self._n = n

    def __getitem__(self, i: Int) -> Scalar[Self.dtype]:
        return self._ptr[i]

    def __setitem__(mut self, i: Int, v: Scalar[Self.dtype]):
        self._ptr[i] = v

    def unsafe_ptr(self) -> Pointer[Scalar[Self.dtype], MutAnyOrigin]:
        return self._ptr


# ---------------------------------------------------------------------------
# DeviceBuffer
# ---------------------------------------------------------------------------


struct DeviceBuffer[dtype: DType](Copyable):
    """On-device buffer.

    On Hexagon/CDSP the host and device share the same physical memory, but
    `enqueue_create_host_buffer` and `enqueue_create_buffer` produce separate
    allocations, so the copy methods perform an element-wise CPU copy.
    """

    var _ptr: Pointer[Scalar[Self.dtype], MutAnyOrigin]
    var _n: Int

    def __init__(
        out self,
        ptr: Pointer[Scalar[Self.dtype], MutAnyOrigin],
        n: Int,
    ):
        self._ptr = ptr
        self._n = n

    def unsafe_ptr(self) -> Pointer[Scalar[Self.dtype], MutAnyOrigin]:
        return self._ptr

    def enqueue_copy_from(self, src: HostBuffer[Self.dtype]):
        var n = self._n if self._n < src._n else src._n
        for i in range(n):
            self._ptr[i] = src._ptr[i]

    def enqueue_copy_to(self, mut dst: HostBuffer[Self.dtype]):
        var n = self._n if self._n < dst._n else dst._n
        for i in range(n):
            dst._ptr[i] = self._ptr[i]


# ---------------------------------------------------------------------------
# DeviceExternalFunction
# ---------------------------------------------------------------------------


struct DeviceExternalFunction(Copyable):
    """Handle to a kernel resolved by name via AsyncRT loadFunction."""

    var _handle: Pointer[NoneType, MutAnyOrigin]

    def __init__(out self, handle: Pointer[NoneType, MutAnyOrigin]):
        self._handle = handle


# ---------------------------------------------------------------------------
# DeviceContext
# ---------------------------------------------------------------------------


struct DeviceContext(Copyable):
    """Host-side handle to the NPU (Hexagon CDSP) via AsyncRT."""

    var _handle: Pointer[NoneType, MutAnyOrigin]

    def __init__(out self, *, api: String) raises:
        """Creates an AsyncRT device context for the given API string.

        Args:
            api: Backend name, e.g. "npu".

        Raises:
            Error if AsyncRT_DeviceContext_create returns a non-null error.
        """
        var handle = Pointer[NoneType, MutAnyOrigin]()
        var err = external_call[
            "AsyncRT_DeviceContext_create",
            Pointer[UInt8, MutAnyOrigin],
        ](
            Pointer(to=handle),
            api.unsafe_ptr(),
            Int32(0),
        )
        if err:
            raise Error("AsyncRT_DeviceContext_create failed")
        self._handle = handle

    def enqueue_create_host_buffer[dtype: DType](
        self, n: Int
    ) -> HostBuffer[dtype]:
        """Allocates a host-accessible buffer of `n` elements.

        On Hexagon/CDSP shared memory this allocation is also device-visible.

        Args:
            n: Number of elements.

        Returns:
            A `HostBuffer` wrapping the allocation.
        """
        var ptr = Pointer[Scalar[dtype]].alloc(n)
        return HostBuffer[dtype](
            ptr.unsafe_mut_cast[True]().as_unsafe_any_origin(), n
        )

    def enqueue_create_buffer[dtype: DType](
        self, n: Int
    ) -> DeviceBuffer[dtype]:
        """Allocates a device buffer of `n` elements.

        On Hexagon/CDSP shared memory this allocation is also host-visible.

        Args:
            n: Number of elements.

        Returns:
            A `DeviceBuffer` wrapping the allocation.
        """
        var ptr = Pointer[Scalar[dtype]].alloc(n)
        return DeviceBuffer[dtype](
            ptr.unsafe_mut_cast[True]().as_unsafe_any_origin(), n
        )

    def synchronize(self) raises:
        """Blocks until all enqueued NPU operations complete.

        Raises:
            Error if AsyncRT_DeviceContext_synchronize returns a non-null error.
        """
        var err = external_call[
            "AsyncRT_DeviceContext_synchronize",
            Pointer[UInt8, MutAnyOrigin],
        ](self._handle)
        if err:
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

        Passing `asm=""` makes the runtime look the name up in its built-in
        registry rather than JIT-compiling provided assembly.

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
        var func_handle = Pointer[NoneType, MutAnyOrigin]()
        var empty = String("")
        var err = external_call[
            "AsyncRT_DeviceContext_loadFunction",
            Pointer[UInt8, MutAnyOrigin],
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
        if err:
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

        Each argument is forwarded by address via the FAM protocol:
        `args[i]` points at the i-th user argument value.

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
        var dense_args = InlineArray[Pointer[NoneType, MutAnyOrigin], num_args](
            uninitialized=True
        )
        comptime for i in range(num_args):
            dense_args[i] = (
                Pointer(to=args[i])
                .unsafe_bitcast[NoneType]()
                .unsafe_mut_cast[True]()
                .as_unsafe_any_origin()
            )
        var err = external_call[
            "AsyncRT_DeviceContext_enqueueFunctionDirect",
            Pointer[UInt8, MutAnyOrigin],
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
            Pointer[NoneType, MutAnyOrigin](),
            c_uint(0),
            dense_args.unsafe_ptr(),
            c_uint(num_args),
            Pointer[UInt64, MutAnyOrigin](),
        )
        if err:
            raise Error("AsyncRT_DeviceContext_enqueueFunctionDirect failed")
