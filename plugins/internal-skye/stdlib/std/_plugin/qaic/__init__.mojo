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
"""QAIC vendor backend.

Implements the `PluginHooks` trait for Qualcomm Cloud AI (Hexagon/HVX)
targets. The stdlib dispatch sites consult `std._plugin.CurrentPlugin`;
the plugin overlay re-aliases that to `QAICBackend`, so every hook that
`QAICBackend` populates with a concrete callable replaces the stdlib default
for QAIC builds.
"""

from std.collections import OptionalReg
from std.collections.string.string_span import (
    _get_kgen_string,
)
from std.memory import UnsafePointer
from std.reflection import SourceLocation
from std.ffi import external_call
from std.sys import llvm_intrinsic, stderr
from std.sys.info import _TargetType, is_triple, _current_target
from std.ffi import CStringSlice

from .algorithm_backend import _reduce_generator_npu
from .host.intrinsics_hvx import (
    hexagon_runtime_exp__vf,
    hexagon_runtime_tanh__vf,
)
from .host.qshim import (
    LOG_MASK_ERROR,
    LOG_MASK_FATAL,
    LOG_MASK_INFO,
    qshim_log,
)

from std._plugin._trait import (
    PluginHooks,
    _ReduceGeneratorPluginHookFnType,
    _PrintEmitPluginHookFnType,
)

# ===-----------------------------------------------------------------------===#
# QAICBackend
# ===-----------------------------------------------------------------------===#


def _is_qaic[target: _TargetType = _current_target()]() -> Bool:
    """Returns True if the provided target is supported by the QAIC plugin."""
    return is_triple["hexagon-unknown-linux-musl", target]()


def _exp[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Hexagon-runtime-provided `exp` for HVX-sized float vectors.

    Falls back to the generic `llvm.exp` intrinsic for shapes that the
    Hexagon runtime does not accelerate.
    """
    comptime if (dtype == DType.float32 and width == 32) or (
        dtype == DType.float16 and width == 64
    ):
        return hexagon_runtime_exp__vf[dtype, width](x)
    else:
        return llvm_intrinsic[
            "llvm.exp", SIMD[dtype, width], has_side_effect=False
        ](x)


def _tanh[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Hexagon-runtime-provided `tanh` for HVX-sized float vectors.

    Falls back to the generic `llvm.tanh` intrinsic for shapes that the
    Hexagon runtime does not accelerate.
    """
    comptime assert (dtype == DType.float32 and width == 32) or (
        dtype == DType.float16 and width == 64
    )
    return hexagon_runtime_tanh__vf[dtype, width](x)


def _stack_allocation[
    count: Int,
    type: AnyType,
    /,
    name: Optional[StaticString],
    alignment: Int,
    address_space: AddressSpace,
]() -> Pointer[type, MutUntrackedOrigin, address_space=address_space]:
    comptime assert (
        address_space == AddressSpace.SHARED
    ), "QAIC Backend only handles SHARED address space stack allocation"
    # On Hexagon/QAIC, the SHARED address space maps to VTCM (Vector
    # Tightly Coupled Memory). The mojo compiler assigns each allocation a
    # unique byte offset within the shared VTCM region at module scope
    # and lowers the allocation to a runtime call to
    # qshimGetBaseVtcmAddr() + the pre-computed offset.
    comptime global_name = name.value() if name else "_global_alloc"
    return {
        _mlir_value = __mlir_op.`pop.global_alloc`[
            name=_get_kgen_string[global_name](),
            count=count.__mlir_index__(),
            memoryType=__mlir_attr.`#pop<global_alloc_addr_space gpu_shared>`,
            _type=Pointer[
                type, MutUntrackedOrigin, address_space=address_space
            ]._mlir_type,
            alignment=alignment.__mlir_index__(),
        ]()
    }


def _unsafe_dangling[alignment: Int]() -> Int:
    """`UnsafePointer.unsafe_dangling()` address for QAIC targets.

    Skye needs to map pointers so we need to detect dangling pointers and
    handle them differently, use 1 as sentinel. UnsafePointer shouldn't be
    constructed with 0. The default of using the alignment makes it harder to
    detect that the pointer is dangling.
    """
    return 1


@always_inline
def _qshim_mask_for(file: FileDescriptor) -> UInt32:
    """Maps `stderr` to ERROR and everything else to INFO, so
    `print(..., file=stderr)` lands as an Error-tagged ULog record."""
    return LOG_MASK_ERROR if file.value == stderr.value else LOG_MASK_INFO


def _print_emit[O: Origin](cstr: CStringSlice[O], file: FileDescriptor):
    """QAIC `print()` byte emitter — routes through `qshim_log`. The
    explicit `length` argument is unused because `qshim_log` takes a
    nul-terminated buffer; it's part of the hook signature for backends
    that need it."""
    qshim_log(_qshim_mask_for(file), cstr.ptr().unsafe_bitcast[UInt8]())


def _abort():
    """`abort()` override: longjmp back to the wrapper's `setjmp`, making
    the kernel return `JIT_DEV_ERROR_FATAL`. If no kernel is in flight
    the per-thread slot is null and we return so `abort()` falls through
    to its default trap."""
    var buf = external_call[
        "__qaic_unwind_get", UnsafePointer[Int32, MutUnsafeAnyOrigin]
    ]()
    if Int(buf) != 0:
        __mlir_op.`pop.external_call`[
            func=__mlir_attr[`"longjmp" : !kgen.string`],
            funcAttrs=__mlir_attr.`["noreturn"]`,
            _type=None,
        ](buf, Int32(1))


def _debug_assert_emit[
    O: Origin
](message: UnsafePointer[Byte, O], length: Int, loc: SourceLocation) -> None:
    """Emit assertion message via `qshim_log`. Body must not call any code
    that uses `debug_assert` (`_WriteBufferHeap`, `_libc.malloc`,
    `Int.write_to`, ...) — those re-enter this function and deadlock the
    compiler. Until we have a `debug_assert`-free formatter (e.g. a C
    `snprintf` + `qshimLog` wrapper), the file:line:col prefix is dropped.
    """
    _ = length
    _ = loc
    # NB: QShim mirrors `LOG_MASK_FATAL` records (the host trace shows two
    # entries per call). This is QAIC SDK behavior, not a bug here.
    qshim_log(LOG_MASK_FATAL, message)


struct QAICBackend(PluginHooks):
    """`PluginHooks` implementation for Qualcomm QAIC (Hexagon) targets."""

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["qaic"]()

    comptime exp_fn: OptionalReg[
        def[
            dtype: DType, width: SIMDLength, //
        ](SIMD[dtype, width]) thin -> SIMD[dtype, width]
    ] = _exp

    comptime tanh_fn[dtype: DType, width: Int]: OptionalReg[
        def[
            dtype: DType, width: SIMDLength, //
        ](SIMD[dtype, width]) thin -> SIMD[dtype, width]
    ] = _tanh if (dtype == DType.float32 and width == 32) or (
        dtype == DType.float16 and width == 64
    ) else None

    comptime stack_allocation_fn[address_space: AddressSpace]: OptionalReg[
        def[
            count: Int,
            type: AnyType,
            /,
            name: Optional[StaticString],
            alignment: Int,
        ]() thin -> Pointer[
            type, MutUntrackedOrigin, address_space=address_space
        ]
    ] = _stack_allocation[
        address_space=address_space, ...
    ] if address_space == AddressSpace.SHARED else None

    comptime address_space_fn[name: StaticString]: OptionalReg[
        AddressSpace
    ] = None

    # Unconditionally change the `unsafe_dangling` behavior when the QAIC
    # plugin is available. This is necessary so that dangling pointers created
    # on the host and passed to the device can be detected, to avoid
    # incorrectly trying to map them.
    comptime unsafe_dangling_fn: OptionalReg[
        def[alignment: Int]() thin -> Int
    ] = _unsafe_dangling

    comptime print_emit_fn: OptionalReg[
        _PrintEmitPluginHookFnType
    ] = _print_emit

    comptime reduce_generator_fn: OptionalReg[
        _ReduceGeneratorPluginHookFnType
    ] = (
        # FIXME(MOCO-3856): This fn type `rebind` should not be necessary.
        rebind[_ReduceGeneratorPluginHookFnType](_reduce_generator_npu)
    )

    @staticmethod
    def abort_fn():
        _abort()

    @staticmethod
    def debug_assert_emit_fn[
        O: Origin
    ](message: UnsafePointer[Byte, O], length: Int, loc: SourceLocation):
        _debug_assert_emit(message, length, loc)

    comptime _handles_debug_assert: Bool = True
