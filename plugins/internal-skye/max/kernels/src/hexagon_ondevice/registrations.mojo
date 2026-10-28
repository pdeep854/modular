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
"""`mo.*` lowerings onto the on-device NPU's code-signed kernel set.

Scoped by `type="npu", api="npu"` so these coexist with the discrete card's
wildcard registrations in `../qaic`: both parts report the label `"npu"`, and
only the API tells them apart (`QuRTDevice::api` is `"npu"`, `QAICDevice::api`
is `"qaic"`). `MojoRegistry::getResolvedKernel` picks the more specific match
and makes an ambiguity a compile error rather than a silent wrong pick.

Selectors the kernel set carries but nothing here reaches yet, and why:

- `vscale_offset` is the bring-up scale-and-shift sibling of `vadd`; no
  staged op lowers to it.

`vadd` left that list when `Add` below grew its device arm: under
`MODULAR_NPU_ADD_DEVICE=1` a bare fp32 add dispatches it, and fp16/bf16 adds
dispatch `gate_residual`'s ones-gate arm the same way. `silu` left it when
`Silu` took over the elementwise SIMD lambda the compiler would otherwise
register -- that lambda is device code that does not compile here, so a bare
`silu` was a hard refusal rather than a slow path, which is what made one
dispatch per activation the better trade.

Four registrations here claim a `mo.composite.*` the graph compiler mints
rather than an operator a model author wrote: `gated_residual`,
`gated_residual_clamp`, `ada_modulate` and `swiglu`. Their expressions have no
`mo.*` operator, and on this target the unfused ops reach no kernel at all --
the graph compiler folds them into a neighbouring kernel's epilogue, which a
pre-signed kernel cannot take. The matching rules are in
`GraphCompiler/lib/MODialect/Patterns/Fusion.cpp`.

`Concat` and `Split` are the two registrations here that are not dispatches at
all: device memory on this transport is host memory, so pure data movement is a
host copy and needs no selector.

`npu_view_copy` is not a registration of an `mo.*` op at all, and that is the
point. Every view kernel -- `mo.slice`, `mo.transpose`,
`mo.static.broadcast_to`, `mo.static.reshape` -- lowers to an alias plus a
`mogg.call.materialize`, and the generic materializer is generated device code
this target cannot load. Registering the *materializer* for this device rather
than an `execute` for each view op closes all of them at once and leaves each op
a view, so an alias the compiler can fold still costs nothing.

`mo.cast` is counted with movement and is not: it converts every element between
two dtypes, so there is nothing to copy -- fp16 and bf16 are the same width and
different numbers. It is an elementwise SIMD lambda for the same reason `vadd`
and `silu` are, and it remains a genuine device-code gap.
"""

import extensibility
from extensibility import (
    InputTensor,
    InputVariadicTensors,
    OutputTensor,
    OutputVariadicTensors,
    _FusedComputeOutputTensor,
    _FusedOutputTensor,
    register_internal,
)
from builtin_primitives.primitives import OwnedByteBuffer
from std.collections import Array
from std.collections.string import atol
from std.ffi import external_call, _get_global_or_null
from max.gpu.host import DeviceContext, Dim
from std.math import min, round, sqrt
from std.memory import AddressSpace, alloc, unsafe_memcpy
from std.memory.alloc import unsafe_alloc
from std.os import getenv
from std.utils.index import IndexList

from .dispatch import (
    NPU_ARENA_BYTES,
    NPU_ARENA_HDR_CURSOR,
    NPU_ARENA_HDR_MAGIC,
    NPU_ARENA_HDR_SLOTS,
    NPU_ARENA_HDR_STRIDE,
    NPU_ARENA_HDR_WITNESS_BYTES,
    NPU_ARENA_HEADER_BYTES,
    NPU_ARENA_MAGIC,
    NPU_ARENA_SLOTS,
    NPU_ARENA_SLOT_BYTES,
    NPU_GRID_CORES,
    NPU_GUARD_ACCEPTS,
    NPU_GUARD_ACCEPT_COUNT,
    NPU_GUARD_CLASS,
    NPU_GUARD_CLASS_ATTRIBUTION,
    NPU_GUARD_CLASS_CORRECTNESS,
    NPU_GUARD_COND_NONE,
    NPU_GUARD_COND_VALUE,
    NPU_GUARD_COND_WORD,
    NPU_GUARD_HDR_COUNT,
    NPU_GUARD_HDR_MAGIC,
    NPU_GUARD_HDR_OP_LEN,
    NPU_GUARD_HDR_OP_PTR,
    NPU_GUARD_HDR_WORDS,
    NPU_GUARD_MAGIC,
    NPU_GUARD_WORD,
    NPU_GUARD_WORDS,
    NPU_MATMUL_PATH_HMX_RAN,
    NPU_MATMUL_PATH_UNSET,
    NPU_RELAYOUT_PATH_RAN,
    NPU_THREAD_MASK,
    NPU_UNPACK_ARM_BIAS,
    NPU_UNPACK_ARM_BIAS_RESID,
    NPU_UNPACK_ARM_NATIVE,
    NPU_UNPACK_ARM_PAIRED,
    NPU_UNPACK_ARM_SCALAR_OFF,
    NPU_UNPACK_ARM_SCALAR_REFUSED,
    NPU_UNPACK_ARM_UNSET,
    NPU_UNPACK_ARM_VECTOR,
    NPU_WITNESS_BYTES,
    NPU_WITNESS_MATMUL_PATH,
    NPU_WITNESS_RELAYOUT_PATH,
    NPU_WITNESS_UNPACK_ARM,
    npu_selector_sig,
)
from layout.coord import Coord
from layout.tile_layout import row_major

from .hmx_crouton import (
    HMX_TILE,
    HmxRows2D,
    hmx_native_weight_croutons,
    hmx_panel,
    hmx_repack_weight_cpm,
)
from .matmul_kernel import matmul_bt_hmx_accepts, mm_conv_fused_supported
from .rms_norm_kernel import (
    RMS_ROPE_MAGIC,
    RMS_ROPE_MAX_HEAD,
    RMS_WITNESS_ARM_WORD,
)


def _fused_conv_supported(c_in: Int) -> Bool:
    """Whether `conv_im2col` can serve a convolution with `c_in` channels.

    Delegates to the kernel core's own predicate so the dispatcher and the
    kernel cannot disagree about which shapes the fused gather is defined for —
    a contract a signed kernel has no way to refuse for itself.
    """
    return mm_conv_fused_supported(c_in)


def _require_dense[
    rank: Int
](
    shape: IndexList[rank],
    strides: IndexList[rank],
    op: StaticString,
    name: StaticString,
) raises:
    """Refuses an operand whose elements are not densely row-major.

    A selector reads its operands as flat contiguous blocks, so a stub that
    addresses part of one -- a batch slice, a concat block -- reaches it by
    pointer arithmetic over the extents. That is the same memory only when the
    strides are the dense row-major ones, and they need not be: MAX registers
    `mo.static.broadcast_to`, `mo.static.reshape`, `mo.transpose` and `mo.slice`
    as *view* kernels, so an operand can arrive as a stride-0 or otherwise
    strided alias of a smaller buffer. A batched matmul whose weight is one
    rank-2 tensor broadcast across the batch is exactly that shape, and walking
    it as dense would read past the allocation and compute from whatever
    followed. Refusing names the cause where a wrong number would not.

    An axis of extent 1 is never stepped, so its stride cannot be observed and
    any value for it is dense -- which is the batch of 1 a diffusion graph
    actually stages, and it must not be refused.
    """
    for i in range(rank):
        # An empty operand has no element to address, so none of its strides is
        # observable and there is nothing to refuse.
        if shape[i] == 0:
            return

    var expected = 1
    for i in range(rank - 1, -1, -1):
        var extent = shape[i]
        if extent != 1 and strides[i] != expected:
            raise Error(
                op,
                " on the on-device NPU: ",
                name,
                (
                    " is not densely row-major, so the selector cannot be"
                    " reached by pointer arithmetic; axis "
                ),
                i,
                " has stride ",
                strides[i],
                " where a dense operand of this shape would have ",
                expected,
            )
        expected *= extent


def _require_pitched_rows[
    rank: Int
](
    shape: IndexList[rank],
    strides: IndexList[rank],
    op: StaticString,
    name: StaticString,
) raises -> Int:
    """Accepts a dense operand or a row-pitched column slice of one, and
    returns the element pitch between consecutive rows.

    A selector that walks its operand row by row needs one stride beyond the
    dense ones: the row axis's. A column slice of a wider fused projection —
    the layout FLUX.2's single-stream blocks hand `mo.composite.swiglu` at
    twenty of its thirty sites — arrives as trailing stride 1, row stride
    equal to the projection's width, and its start already folded into the
    pointer (a view carries its offset in its pointer, and the launch path
    sends every buffer's offset within its allocation). Leading axes must
    fold contiguously onto those pitched rows, so one pitch addresses every
    row. Anything else — a broadcast's stride-0 axis, a transpose, a batch
    slice with its own pitch — still refuses, exactly as `_require_dense`
    would, because no single pitch reaches its elements.

    An axis of extent 1 is never stepped, so any stride there is accepted;
    an empty operand has nothing to address and reports the dense pitch.
    """
    var dense_pitch = shape[rank - 1] if rank > 0 else 0
    for i in range(rank):
        if shape[i] == 0:
            return dense_pitch
    if rank == 1:
        if strides[0] != 1:
            raise Error(
                op,
                " on the on-device NPU: ",
                name,
                " must be contiguous along its only axis; got stride ",
                strides[0],
            )
        return dense_pitch

    if shape[rank - 1] != 1 and strides[rank - 1] != 1:
        raise Error(
            op,
            " on the on-device NPU: ",
            name,
            " must be contiguous along its trailing axis; got stride ",
            strides[rank - 1],
        )

    var row_pitch = dense_pitch
    if shape[rank - 2] != 1:
        row_pitch = strides[rank - 2]
        if row_pitch < dense_pitch:
            raise Error(
                op,
                " on the on-device NPU: ",
                name,
                " has row stride ",
                row_pitch,
                " shorter than its rows' ",
                dense_pitch,
                " elements, so consecutive rows overlap",
            )

    var expected = shape[rank - 2] * row_pitch
    for i in range(rank - 3, -1, -1):
        if shape[i] != 1 and strides[i] != expected:
            raise Error(
                op,
                " on the on-device NPU: ",
                name,
                (
                    " does not fold onto row-pitched storage, so one pitch"
                    " cannot address it; axis "
                ),
                i,
                " has stride ",
                strides[i],
                " where the pitched rows continue at ",
                expected,
            )
        expected *= shape[i]
    return row_pitch


comptime _NPU_ARENA_KEY: StaticString = "npu_witness_arena"

comptime _NPUArenaPtr = Pointer[UInt32, MutUnsafeAnyOrigin]


def _arena_ptr(cached: Pointer[NoneType, MutUntrackedOrigin]) -> _NPUArenaPtr:
    """The cached arena address as a pointer this file may index.

    The runtime's global table hands back a checked `Pointer`, and the arena is
    indexed word by word, so the address is what crosses -- the same shape the
    kernels' own `witness` argument has.
    """
    return _NPUArenaPtr(unsafe_from_address=Int(cached))


def _witness_arena(ctx: DeviceContext) raises -> _NPUArenaPtr:
    """The process-lifetime witness allocation, created on first use.

    Cached in the runtime's global table rather than in a struct field, because
    a registration is a static `execute` with nowhere to keep state; this is the
    same mechanism `get_hilbert_lut_with_cache` uses to hold a device buffer for
    the life of the process (`linalg/utils_gpu.mojo`). The buffer's ownership is
    dropped with `take_ptr` on purpose: the arena is never freed, and a free
    would flush the batch it exists to protect.

    Keyed to the PROCESS, and so to the device rather than to a context: a
    launch's operands are resolved against allocations the DEVICE handed out,
    which every context of it shares, and this backend addresses one NSP per
    process. A second device would need a key per device.

    Sized in the same units the per-launch witness asked for -- see
    `NPU_WITNESS_BYTES`, whose comment records that this argument is honoured as
    bytes today and would over-allocate rather than under-allocate if that ever
    changed.
    """
    var cached = _get_global_or_null(_NPU_ARENA_KEY)
    if not cached:
        var buf = ctx.enqueue_create_buffer[DType.uint32](NPU_ARENA_BYTES)
        var fresh = buf.unsafe_ptr()
        external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
            StringSlice(_NPU_ARENA_KEY), fresh.unsafe_bitcast[NoneType]()
        )
        _ = buf.take_ptr()
        cached = _get_global_or_null(_NPU_ARENA_KEY)
        var header = _arena_ptr(cached.unsafe_value())
        # Written before the magic, so a reader that sees the magic sees a
        # complete header. The transport checks these against its own batch cap
        # and refuses a host whose arena is too small to keep one slot per
        # in-flight launch.
        header[unsafe_offset=NPU_ARENA_HDR_SLOTS] = UInt32(NPU_ARENA_SLOTS)
        header[unsafe_offset=NPU_ARENA_HDR_STRIDE] = UInt32(
            NPU_ARENA_SLOT_BYTES
        )
        header[unsafe_offset=NPU_ARENA_HDR_WITNESS_BYTES] = UInt32(
            NPU_WITNESS_BYTES
        )
        header[unsafe_offset=NPU_ARENA_HDR_CURSOR] = 0
        header[unsafe_offset=NPU_ARENA_HDR_MAGIC] = NPU_ARENA_MAGIC
    return _arena_ptr(cached.unsafe_value())


def _refuse_batched_graph_path(batch_requested: Bool) raises:
    """Refuses `MODULAR_NPU_BATCH=1` on the graph path, which it corrupts.

    A KNOWN-CORRUPTING MODE REACHABLE BY ENV VAR IS THE ONE THAT SILENTLY
    BECOMES A BASELINE. Batching is correct for a launch sequence with no host
    reader in it -- the hand drivers, where it was developed and measured -- and
    wrong for this one, so the refusal belongs where the graph path can
    recognise itself rather than in the transport, which cannot tell its callers
    apart.

    THE MECHANISM (run 186A; PR #447 gates 7-8 lock it with no device). Several
    registrations in this file execute ON THE HOST and reach their operands
    through `unsafe_ptr()` alone -- `Concat`, `Split`, `Cast`, `Gather`, the
    elementwise host walks, `npu_view_copy`'s materializer, `mgp.buffer.concat`.
    A host op asks the transport for nothing, so nothing flushes and nothing
    orders it against a launch still held in the batch: it reads the producer's
    PRE-LAUNCH bytes. Every status is 0 and every witness guard passes -- the
    kernels that ran did run correctly, on wrong inputs, which a family-2 guard
    cannot see by construction -- and the render is deterministically wrong. 186A
    measured five wrong stage hashes that way, twice, identically.

    `QuRTDeviceContext.h`'s `batching()` comment predicted exactly this caller
    class and scoped batching to sequences known to contain none. This is that
    scope, enforced.

    Not a fix and not a policy: the fix disposition (flush at every host reader,
    or hold only inside a recorded region) is open, and whichever lands deletes
    this. Takes the flag's value rather than reading it, so the refusal is
    testable without touching the environment.
    """
    if not batch_requested:
        return
    raise Error(
        "MODULAR_NPU_BATCH=1 is refused on the on-device NPU's graph path: it"
        " corrupts results silently. Ops this target runs on the host"
        " (mo.concat, mo.split, mo.cast, mo.gather, the elementwise walks, a"
        " materializing view) read their operands straight out of device memory"
        " without going through the transport, so a launch still held in the"
        " batch has not written them yet and the host op computes from"
        " pre-launch bytes -- with every status 0 and every witness guard"
        " passing. Run 186A measured five deterministically wrong stage hashes"
        " this way; PR #447's gates 7 and 8 lock the mechanism. Unset"
        " MODULAR_NPU_BATCH to run the graph; batching stays available to"
        " launch sequences with no host reader in them."
    )


def _refuse_megastep_graph_path(megastep_requested: Bool) raises:
    """Refuses `MODULAR_NPU_MEGASTEP=1` on the graph path: 186A, held longer.

    A HELD STEP TABLE IS CORRUPTED BY THE SAME READERS AS A HELD BATCH. The
    mechanism is `_refuse_batched_graph_path`'s docstring: this file's host ops
    reach their operands through `unsafe_ptr()` alone, so nothing flushes and
    nothing orders them against a launch still held. The megastep makes the
    window WIDER, deliberately: it holds the whole step (up to 1024 launches
    against the batch's measured width 8.5), `memAlloc` does not flush the
    table, and device frees are parked under it until the flush -- the hold was
    built to survive this file's witness bracketing, which under a batch would
    have cut it. Every host reader inside that window computes from pre-launch
    bytes with rc=0 and every witness guard passing.

    The step table also carries a second exposure the batch does not: only
    init-constant-classified allocations may be table entries, and that
    classification trusts the process-global source-range map in
    `npu_constant_ranges.cpp`, which only grows -- `Forget` is written and
    nothing calls it, so a stale range from a previous stage can bless a dirty
    buffer as constant and its maintenance is skipped. That guard is the named
    precondition on every holding arm; this refusal stands until it lands and
    holding is scoped to a declared recorded region, the adjudicated
    disposition (`MEGAKERNEL_ASSESSMENT.md` section 4 names this refusal as
    guard 1 of its two; the recorded-region work deletes this).

    Same shape as the batch refusal on purpose: takes the flag's value rather
    than reading it, so the refusal is testable without touching the
    environment; the caller reads it exactly the way `npuMegastepRequested`
    does (the value is exactly "1"). The hand drivers and the transport's own
    megastep lifecycle/replay tests are untouched.
    """
    if not megastep_requested:
        return
    raise Error(
        "MODULAR_NPU_MEGASTEP=1 is refused on the on-device NPU's graph path:"
        " it corrupts results silently the same way MODULAR_NPU_BATCH does"
        " (run 186A; PR #447 locks the mechanism, #448 refuses the batch), and"
        " it holds longer. Ops this target runs on the host (mo.concat,"
        " mo.split, mo.cast, mo.gather, the elementwise walks, a materializing"
        " view) read their operands straight out of device memory without"
        " going through the transport, and a step table still holding the"
        " step's launches has not written them yet: memAlloc does not flush it"
        " and memFree parks, so the host op computes from pre-launch bytes --"
        " with every status 0 and every witness guard passing. The table's"
        " constant-ranges classification (npu_constant_ranges.cpp) is also"
        " never forgotten across stages. Unset MODULAR_NPU_MEGASTEP to run the"
        " graph; the refusal stands until the constant-ranges guard lands and"
        " holding is scoped to a declared recorded region."
    )


def _witness_slot(ctx: DeviceContext) raises -> _NPUArenaPtr:
    """Takes the next arena slot for one launch, and returns its witness base.

    ONE CALL PER LAUNCH, NOT PER OP: `BatchMatmul` and both `Conv` arms issue N
    launches against what used to be one buffer, and a slot shared across two
    launches in one batch would alias their reports -- and, on the HMX arm,
    corrupt the kernel's own rendezvous (see `NPU_ARENA_MAGIC`'s comment).

    The cursor wraps at `NPU_ARENA_SLOTS`, which is at least the transport's
    batch cap, so any window of consecutive launches short enough to be one
    batch gets distinct slots. Not atomic, and it does not need to be: the
    transport's own launch path already assumes a single submitter per context
    -- `enqueueFunctionExecDirect` appends to `batch_` under no lock.

    The slot is cleared rather than merely seeded. A reused slot whose stale
    words survive is the aliasing failure the VAE conv probe already warns
    about: plausible numbers belonging to an earlier launch. Clearing is also
    what seeds slots 7 and 22, since both `UNSET` codes are zero.
    """
    # Every graph-path launch takes a slot here and nothing else does, so this
    # is the one place the graph path can refuse a transport mode on its own
    # behalf. Read the same way the transport reads them -- the value is
    # exactly "1", not merely set (`npuBatchRequested`,
    # `npuMegastepRequested`).
    _refuse_batched_graph_path(getenv("MODULAR_NPU_BATCH") == "1")
    _refuse_megastep_graph_path(getenv("MODULAR_NPU_MEGASTEP") == "1")

    # The chain hold's backstop: every device launch takes a slot here before
    # it enqueues, so a held matmul member runs ahead of whatever this launch
    # is about to compute. The chain route's own calls are no-ops by then --
    # it clears the pending record before it takes slots of its own.
    _matmul_chain_flush()

    var arena = _witness_arena(ctx)
    var cursor = Int(arena[unsafe_offset=NPU_ARENA_HDR_CURSOR])
    arena[unsafe_offset=NPU_ARENA_HDR_CURSOR] = UInt32(
        (cursor + 1) % NPU_ARENA_SLOTS
    )
    var slot = arena.unsafe_offset(
        NPU_ARENA_HEADER_BYTES // 4 + cursor * (NPU_ARENA_SLOT_BYTES // 4)
    )
    for i in range(NPU_ARENA_SLOT_BYTES // 4):
        slot[unsafe_offset=i] = 0
    return slot


def _guard_record(slot: _NPUArenaPtr) -> _NPUArenaPtr:
    """The host-only half of a slot: the guard record the flush reads.

    Past the witness the device is given, so no kernel addresses it -- a launch
    carries the witness base and writes `NPU_WITNESS_WORDS` from there.
    """
    return slot.unsafe_offset(NPU_WITNESS_BYTES // 4)


def _arm_guard(
    slot: _NPUArenaPtr,
    guard_class: UInt32,
    word: Int,
    cond_word: Int,
    cond_value: UInt32,
    op: StaticString,
) -> _NPUArenaPtr:
    """Opens a guard record on `slot` and returns its accept list.

    The caller fills the accept list, which is what keeps the device's codes in
    this file: the transport evaluates "is the word one of these" and knows
    none of them. `MATMUL_PATH_*` and `HMX_UNPACK_ARM_*` belong to the signed
    kernel set, and this side is only ever their reader.
    """
    var record = _guard_record(slot)
    var index = Int(record[unsafe_offset=NPU_GUARD_HDR_COUNT])
    record[unsafe_offset=NPU_GUARD_HDR_COUNT] = UInt32(index + 1)
    record[unsafe_offset=NPU_GUARD_HDR_OP_LEN] = UInt32(op.byte_length())
    var op_ptr = UInt64(Int(op.unsafe_ptr()))
    record[unsafe_offset=NPU_GUARD_HDR_OP_PTR] = UInt32(op_ptr & 0xFFFFFFFF)
    record[unsafe_offset=NPU_GUARD_HDR_OP_PTR + 1] = UInt32(op_ptr >> 32)
    var entry = record.unsafe_offset(
        (NPU_GUARD_HDR_WORDS + index * NPU_GUARD_WORDS)
    )
    entry[unsafe_offset=NPU_GUARD_CLASS] = guard_class
    entry[unsafe_offset=NPU_GUARD_WORD] = UInt32(word)
    entry[unsafe_offset=NPU_GUARD_COND_WORD] = UInt32(cond_word)
    entry[unsafe_offset=NPU_GUARD_COND_VALUE] = cond_value
    # Last, and after the entry it counts: the transport reads a record only
    # once this word is present, so a partially written one is never evaluated.
    record[unsafe_offset=NPU_GUARD_HDR_MAGIC] = NPU_GUARD_MAGIC
    return entry


def _accept(entry: _NPUArenaPtr, value: UInt32):
    """Adds one accepted code to a guard armed by `_arm_guard`."""
    var count = Int(entry[unsafe_offset=NPU_GUARD_ACCEPT_COUNT])
    entry[unsafe_offset=NPU_GUARD_ACCEPTS + count] = value
    entry[unsafe_offset=NPU_GUARD_ACCEPT_COUNT] = UInt32(count + 1)


def _arm_matmul_ran(slot: _NPUArenaPtr, op: StaticString):
    """Arms `_require_matmul_ran` on this launch, deferred to the flush.

    THE DEFERRAL IS THE POINT. Read straight after the enqueue this guard sees
    an unwritten slot under batching -- the value is late, not lost (PR #442) --
    so it would raise on a launch that runs correctly moments later. Evaluated
    at the flush it reads what the kernel actually wrote, before any consumer
    can observe the output: every route by which host code or another op can
    see a launch's result sends the batch first.

    CORRECTNESS, so a refusal invalidates the whole flush. That is not a policy
    this file chose: the transport already fails a batch at the refusing record
    when the device raises the RPC status (`mojo_npu_imp.c:1058-1062`), and this
    guard is standing in for a status word `_matmul_stripe_hmx` does not raise
    on the paths that write a refusal code and return success
    (`npu_kernel.mojo:2784`, `:2802`, `:2807`). The day those paths call
    `mojo_npu_launch_fail` the way the attention arm already does
    (`:4342`, `:4385`, `:4562`), this guard can be deleted outright -- the
    accept set below and `_require_matmul_ran`'s docstring are what a reader
    doing that will need.

    The accepted codes are the seed (the vector nest ran and left the slot
    alone) and `HMX1`. Anything else is a refusal, tested that way rather than
    against a list of known failures so a code this host has not heard of is
    still loud -- `_require_matmul_ran` states why, and the two must agree.
    """
    var entry = _arm_guard(
        slot,
        NPU_GUARD_CLASS_CORRECTNESS,
        NPU_WITNESS_MATMUL_PATH,
        NPU_GUARD_COND_NONE,
        0,
        op,
    )
    _accept(entry, NPU_MATMUL_PATH_UNSET)
    _accept(entry, NPU_MATMUL_PATH_HMX_RAN)


def _arm_unpack_arm_known(slot: _NPUArenaPtr, op: StaticString):
    """Arms `_require_unpack_arm_known` on this launch, deferred to the flush.

    ATTRIBUTION, so this one records and the render continues. The output is
    correct whichever arm ran -- the seam's arms are bit-identical and the guard
    accepts `SCALAR_REFUSED`, the code that means a build which asked for the
    vector arm measured nothing. What it catches is a skel newer than this host.
    Letting that abort a render is the regression-gate-becomes-correctness-gate
    trap, and it is the failure this reader was written to avoid in the first
    place (see `_require_unpack_arm_known`, which crosses before its writer for
    the same reason).

    Conditional on slot 7 holding `HMX1`, because slot 22 is the seam's only on
    the HMX arm: off that arm the HVX nest owns words 8..31 and leaves a utimer
    tick there, and validating a tick as an arm code would take down every
    render on the shipping default build.
    """
    var entry = _arm_guard(
        slot,
        NPU_GUARD_CLASS_ATTRIBUTION,
        NPU_WITNESS_UNPACK_ARM,
        NPU_WITNESS_MATMUL_PATH,
        NPU_MATMUL_PATH_HMX_RAN,
        op,
    )
    _accept(entry, NPU_UNPACK_ARM_UNSET)
    _accept(entry, NPU_UNPACK_ARM_SCALAR_OFF)
    _accept(entry, NPU_UNPACK_ARM_VECTOR)
    _accept(entry, NPU_UNPACK_ARM_SCALAR_REFUSED)
    _accept(entry, NPU_UNPACK_ARM_PAIRED)
    _accept(entry, NPU_UNPACK_ARM_NATIVE)
    _accept(entry, NPU_UNPACK_ARM_BIAS)
    _accept(entry, NPU_UNPACK_ARM_BIAS_RESID)


def _require_matmul_ran(path: UInt32, op: StaticString) raises:
    """Refuses a `matmul` dispatch whose kernel wrote no output.

    An HMX build that accepts a shape and then cannot run it -- no VTCM grant, a
    lock the manager refused, a rendezvous past its deadline -- refuses OUTRIGHT
    rather than dropping to the vector nest, because reporting the vector unit's
    numbers as the array's is the one failure that path must not have. It says so
    in witness slot 7 and returns success, so a stub that does not read the slot
    hands the graph whatever the output allocation held. That is the cold
    critique's C1 on this side of the transport.

    `NPU_MATMUL_PATH_UNSET` is the seed this stub wrote before the launch, and it
    means the vector nest ran: that arm leaves the slot alone. Anything that is
    neither the seed nor `HMX1` is a refusal, tested that way rather than against
    a list so a code this file has not heard of is still loud.
    """
    if path == NPU_MATMUL_PATH_UNSET or path == NPU_MATMUL_PATH_HMX_RAN:
        return
    raise Error(
        op,
        (
            " on the on-device NPU: the matmul kernel refused the matrix array"
            " and wrote no output (witness slot 7 = "
        ),
        path,
        "); see MATMUL_PATH_* in npu_kernel.mojo for what the code names",
    )


def _require_unpack_arm_known(
    path: UInt32, arm: UInt32, op: StaticString
) raises:
    """Refuses a dispatch whose unpack seam echoed a code this host cannot name.

    THE READER HALF of the arm witness, and the reason the writer is worth
    having. `SKYE_HMX_UNPACK_VEC`'s failure mode is a silent admission early-out:
    the vector arm is refused, the scalar arm runs, the output is bit-identical
    and the A/B measures the arm it did not select while looking like a clean
    null. The seam echoes which arm ran (`MATMUL_WIT_UNPACK_ARM`) and this is
    where a render consumes it -- the graph path is what a render runs, so a
    reader anywhere else is not on this question's path.

    SLOT 22 IS THE SEAM'S ONLY ON THE HMX ARM, which is why `path` is an
    argument. The word is free by the mutual exclusion `_matmul_hmx_sweep`
    documents -- the array arm `return`s at `mojo_matmul_f16`'s
    `matmul_bt_hmx_accepts` branch before the HVX nest that owns words 8..31 --
    and the OTHER side of that exclusion is that the nest DOES own word 22, as
    thread 2's utimer bracket (`witness + 8 + thread_idx * MHA_PROBE_STRIDE`,
    slot 2). A shape the array declines therefore leaves a tick count here, not
    an arm code: on the 24 M=1 matmul dispatches of every production render,
    order 149 leg B read 1965..77873 here in the flag-ON build and 2058..77904 in
    the flag-OFF one -- the nest writes the word whatever the arm is set to, so
    this is the shipping default's problem and not an experiment's.
    Reading the word on that arm is reading the nest's clock and raising on it.
    So the reader keys on the same discriminator the writer does, and `HMX1` in
    slot 7 is exactly it: `_matmul_hmx_sweep` is unreachable unless slot 7 already
    holds it. Off that arm there is nothing here to validate, and the seed cannot
    survive to say so -- the nest overwrites it too.

    `NPU_UNPACK_ARM_UNSET` is the seed and is ACCEPTED: the write is
    comptime-gated with the arm, so an untouched slot is a flag-off build or a
    skel predating the word, neither of which is a fault. The two scalar codes
    are accepted and distinct on purpose -- REFUSED is the one that means a build
    which asked for the vector arm measured nothing, and a caller reading the
    word gets to tell that from a plain flag-off run.

    `NPU_UNPACK_ARM_PAIRED` is accepted BEFORE any build emits it, which is the
    only order that works: this raises during a render, so a skel whose unpack
    reports a code its host predates takes down the first matmul of the first
    render even though the arm ran correctly. The reader crosses first and waits.

    Tested against the codes this file knows rather than against a list of
    failures, so a code from a newer skel is loud here instead of silently
    scoring as one of these.
    """
    if path != NPU_MATMUL_PATH_HMX_RAN:
        return
    if (
        arm == NPU_UNPACK_ARM_UNSET
        or arm == NPU_UNPACK_ARM_SCALAR_OFF
        or arm == NPU_UNPACK_ARM_VECTOR
        or arm == NPU_UNPACK_ARM_SCALAR_REFUSED
        or arm == NPU_UNPACK_ARM_PAIRED
        or arm == NPU_UNPACK_ARM_NATIVE
        or arm == NPU_UNPACK_ARM_BIAS
        or arm == NPU_UNPACK_ARM_BIAS_RESID
    ):
        return
    raise Error(
        op,
        (
            " on the on-device NPU: the unpack seam echoed an arm code this"
            " host does not know (witness slot 22 = "
        ),
        arm,
        "); see HMX_UNPACK_ARM_* in hmx_crouton.mojo for what the code names",
    )


def _stage_k_padded_rows[
    dtype: DType, dst_origin: MutOrigin, src_origin: MutOrigin, //
](
    dst: Pointer[Scalar[dtype], dst_origin],
    src: Pointer[Scalar[dtype], src_origin],
    rows: Int,
    k: Int,
    k_pad: Int,
):
    """Stages `[rows, k]` row-major data into a `[rows, k_pad]` buffer.

    Zero-padding the contraction axis is the identity on the result: every
    padded lane pairs a zero of A with a zero of B and contributes exactly
    nothing to the accumulation. The `matmul` selector reads K in 64-element
    vectors, so a ragged K is dispatched at the next multiple of 64 through
    this staging -- the same trick `mo.conv` plays on its im2col rows, and
    the one the silicon-certified hand driver plays on `post_quant_conv`.

    Args:
        dst: The `[rows, k_pad]` destination; every element is written, so
            the buffer needs no prior fill.
        src: The `[rows, k]` row-major source.
        rows: How many rows to stage.
        k: The source row length.
        k_pad: The destination row pitch; at least `k`.
    """
    for i in range(rows):
        unsafe_memcpy(
            dest=dst.unsafe_offset(i * k_pad),
            src=src.unsafe_offset(i * k),
            count=k,
        )
        for j in range(k, k_pad):
            dst[unsafe_offset=i * k_pad + j] = 0


def _stage_scaled_rows[
    dtype: DType, dst_origin: MutOrigin, src_origin: MutOrigin, //
](
    dst: Pointer[Scalar[dtype], dst_origin],
    src: Pointer[Scalar[dtype], src_origin],
    rows: Int,
    k: Int,
    k_pad: Int,
    scale: Float32,
):
    """Stages `[rows, k]` row-major data into `[rows, k_pad]`, folding `scale`.

    The decomposed attention's scale fold: the signed `softmax` selector takes
    no scale scalar, so the hand driver folds the softmax scale host-side into
    `to_q` in fp32 (`npu_vae_decoder_dispatch.cpp`, `walkAttention`). Folding
    into the staged activation instead is the same operator with one extra
    `dtype` rounding: the multiply happens in fp32 and rounds once on store.
    The zero tail is `_stage_k_padded_rows`'s contraction identity, so a
    ragged row length dispatches at the next multiple of 64.

    Args:
        dst: The `[rows, k_pad]` destination; every element is written, so
            the buffer needs no prior fill.
        src: The `[rows, k]` row-major source.
        rows: How many rows to stage.
        k: The source row length.
        k_pad: The destination row pitch; at least `k`.
        scale: The factor folded into every staged element.
    """
    for i in range(rows):
        for j in range(k):
            dst[unsafe_offset=i * k_pad + j] = (
                src[unsafe_offset=i * k + j].cast[DType.float32]() * scale
            ).cast[dtype]()
        for j in range(k, k_pad):
            dst[unsafe_offset=i * k_pad + j] = 0


def _stage_transposed_rows[
    dtype: DType, dst_origin: MutOrigin, src_origin: MutOrigin, //
](
    dst: Pointer[Scalar[dtype], dst_origin],
    src: Pointer[Scalar[dtype], src_origin],
    rows: Int,
    cols: Int,
    rows_pad: Int,
):
    """Stages `[rows, cols]` row-major data transposed into `[cols, rows_pad]`.

    The hand driver's `transposeStrided`: V arrives token-major `[seq, dim]`
    and the PV matmul contracts over the sequence, so V becomes the
    `[N=dim, K=seq]` b-transposed weight the `matmul` selector reads. The
    zeroed tail carries the same contraction identity as
    `_stage_k_padded_rows` -- a ragged sequence pairs zero probability
    columns with these zero columns and adds exactly nothing.

    Args:
        dst: The `[cols, rows_pad]` destination; every element is written, so
            the buffer needs no prior fill.
        src: The `[rows, cols]` row-major source.
        rows: The source row count, the destination's inner extent.
        cols: The source row length, the destination's row count.
        rows_pad: The destination row pitch; at least `rows`.
    """
    for c in range(cols):
        for r in range(rows):
            dst[unsafe_offset=c * rows_pad + r] = src[
                unsafe_offset=r * cols + c
            ]
        for r in range(rows, rows_pad):
            dst[unsafe_offset=c * rows_pad + r] = 0


comptime _WEIGHT_NATIVE_ENV = "MODULAR_NPU_WEIGHT_NATIVE"


def _stage_native_weight(m: Int, n: Int, k_pad: Int) -> Bool:
    """Whether a staged B operand must carry the crouton byte order.

    The `matmul` selector's weight layout is a skel build define
    (`SKYE_HMX_WEIGHTS_NATIVE`) no host code can read, so the serve declares
    it through the environment: strictly "1", the same reading
    `npu_registry_load.h` gives the same variable, and any other value keeps
    the row-major staging every non-native world reads. The shape half
    mirrors the kernel's own arm gate -- a dispatch the array refuses runs
    the row-major HVX nest, and croutons staged for that nest would be the
    same silent mismatch this predicate exists to remove. Read per dispatch:
    a `getenv` beside a transport that costs a second is free, and one MEF
    outlives any single serve's environment.

    Args:
        m: Output rows of the dispatch.
        n: Output columns, the staged operand's row count.
        k_pad: The reduction extent as dispatched; a multiple of 64.

    Returns:
        True when the staged rows must be repacked into croutons.
    """
    if getenv(_WEIGHT_NATIVE_ENV) != "1":
        return False
    return matmul_bt_hmx_accepts(m, n, k_pad, 0, n)


comptime _CONV_HMX_ENV = "MODULAR_NPU_CONV_HMX"


def _conv_hmx_requested() -> Bool:
    """Whether the fused conv dispatches the array selector instead of HVX.

    Read per dispatch for the same reason `_stage_native_weight` reads its
    variable there: one MEF outlives any single serve's environment. ON is
    the shipped posture -- the array arm certified at orders 319/320
    (byte-tier control, census 31+1, twin x4 boots, wall -22.9%); an
    explicit "0" restores selector 36, keeping the one-variable A/B and the
    certified fallback reachable on any serve without a re-mint.
    """
    return getenv(_CONV_HMX_ENV) != "0"


def _crouton_pack_rows[
    dtype: DType, dst_origin: MutOrigin, src_origin: MutOrigin, //
](
    dst: Pointer[Scalar[dtype], dst_origin],
    src: Pointer[Scalar[dtype], src_origin],
    n: Int,
    k: Int,
):
    """Repacks row-major `[n, k]` rows into the column-panel-major artefact.

    The kernels' own `hmx_repack_weight_cpm` -- the crouton order's one
    implementation in the tree -- over the same re-origination seam its read
    side (`hmx_native_weight_panel`) makes. Same element count in and out;
    only the meaning of the byte order changes, which is why the caller
    gates on `_stage_native_weight` rather than anything here being able to
    check.

    Args:
        dst: The `[n, k]` destination; receives every element, crouton-ordered.
        src: The `[n, k]` row-major source.
        n: Output columns of the matmul this weight serves; a multiple of 32.
        k: The reduction extent; a multiple of 64.
    """
    comptime assert (
        dtype == DType.float16
    ), "the crouton weight artefact is the fp16 arm's"
    hmx_repack_weight_cpm[AddressSpace.GENERIC](
        hmx_panel[AddressSpace.GENERIC](
            Pointer[Scalar[DType.float16], MutUntrackedOrigin](
                unsafe_from_address=Int(dst)
            ),
            hmx_native_weight_croutons(n, k),
        ),
        HmxRows2D(
            ptr=Pointer[Scalar[DType.float16], MutUnsafeAnyOrigin](
                unsafe_from_address=Int(src)
            ),
            layout=row_major(Coord(n, k)),
        ),
        n,
        k,
    )


def _stage_weight_rows[
    dtype: DType, dst_origin: MutOrigin, src_origin: MutOrigin, //
](
    dst: Pointer[Scalar[dtype], dst_origin],
    src: Pointer[Scalar[dtype], src_origin],
    rows: Int,
    k: Int,
    k_pad: Int,
    m: Int,
):
    """Stages `[rows, k]` row-major B into `[rows, k_pad]`, layout-matched.

    `_stage_k_padded_rows` with the layout question answered: on an
    array-native serve (`_stage_native_weight`) the staged rows additionally
    take the crouton byte order the skel's feeders read, because selector 24
    has no runtime layout arm -- a row-major B on a native skel computes
    finite, well-scaled, entirely wrong numbers (the 306 a2/a3 conviction).
    Zero-padding first keeps the contraction identity; the pad rides into
    the croutons like any other zero element.

    Args:
        dst: The `[rows, k_pad]` destination; every element is written.
        src: The `[rows, k]` row-major source.
        rows: How many rows to stage (the matmul's N).
        k: The source row length.
        k_pad: The destination row pitch; at least `k`, a multiple of 64.
        m: Output rows of the dispatch this operand feeds.
    """
    if not _stage_native_weight(m, rows, k_pad):
        _stage_k_padded_rows(dst, src, rows, k, k_pad)
        return
    if k == k_pad:
        _crouton_pack_rows(dst, src, rows, k_pad)
        return
    var staged = unsafe_alloc[Scalar[dtype]](rows * k_pad)
    _stage_k_padded_rows(staged, src, rows, k, k_pad)
    _crouton_pack_rows(dst, staged, rows, k_pad)
    staged.unsafe_free()


def _stage_transposed_weight[
    dtype: DType, dst_origin: MutOrigin, src_origin: MutOrigin, //
](
    dst: Pointer[Scalar[dtype], dst_origin],
    src: Pointer[Scalar[dtype], src_origin],
    rows: Int,
    cols: Int,
    rows_pad: Int,
    m: Int,
):
    """`_stage_transposed_rows` with the layout question answered.

    The decomposed attention's V^T is a runtime-built `[N = cols,
    K = rows_pad]` weight operand, so it owes the serving skel the same
    layout `_stage_weight_rows` owes for a registry row -- the crouton
    contract binds every B the selector reads, not only checkpoint weights.

    Args:
        dst: The `[cols, rows_pad]` destination; every element is written.
        src: The `[rows, cols]` row-major source.
        rows: The source row count, the destination's inner extent.
        cols: The source row length, the destination's row count.
        rows_pad: The destination row pitch; at least `rows`.
        m: Output rows of the dispatch this operand feeds.
    """
    if not _stage_native_weight(m, cols, rows_pad):
        _stage_transposed_rows(dst, src, rows, cols, rows_pad)
        return
    var staged = unsafe_alloc[Scalar[dtype]](cols * rows_pad)
    _stage_transposed_rows(staged, src, rows, cols, rows_pad)
    _crouton_pack_rows(dst, staged, cols, rows_pad)
    staged.unsafe_free()


comptime _MATMUL_CHAIN_ENV = "MODULAR_NPU_MATMUL_CHAIN"

comptime _CHAIN_STATE_KEY: StaticString = "npu_matmul_chain_pending"

comptime _CHAIN_WORDS = 12
"""live flag, then c/a/b/witness addresses (two words each), then m, n, k."""

comptime _CHAIN_TAKE_CLAIMED: UInt32 = 1
comptime _CHAIN_TAKE_FIRED: UInt32 = 2


def _matmul_chain_requested() -> Bool:
    """Whether adjacent independent matmuls pair into `matmul_chain2` (48).

    Strictly "1", read per dispatch, unset the shipping default -- the
    `_add_device_requested` contract. Valid only against a catalogue whose
    skel carries selector 48; an older signed skel refuses the row by name
    and the caller keeps selector 24 singletons.
    """
    return getenv(_MATMUL_CHAIN_ENV) == "1"


def _refuse_chain_with_queue(chain: Bool, queue: Bool) raises:
    """Refuses the chain hold under `MODULAR_NPU_QUEUE=1`.

    The hold's soundness argument is the synchronous transport's: a held
    member is dispatched before any seam through which its output can be
    observed, and `dispatchSingle` completes before it returns. The queue
    makes the second half untrue -- a fired member is merely ENQUEUED -- so
    the pair of knobs is refused by value, the batch-refusal shape, rather
    than reasoned about per site.
    """
    if not (chain and queue):
        return
    raise Error(
        "MODULAR_NPU_MATMUL_CHAIN=1 is refused with MODULAR_NPU_QUEUE=1: the"
        " chain hold's flush discipline assumes the blocking transport, where"
        " a fired member has RUN when the fire returns. Unset one of the two."
    )


def _chain_state_or_null() -> Optional[Pointer[NoneType, MutUntrackedOrigin]]:
    return _get_global_or_null(_CHAIN_STATE_KEY)


def _chain_state(ctx: DeviceContext) raises -> _NPUArenaPtr:
    """The one pending-member record: the witness arena's own global idiom."""
    var cached = _chain_state_or_null()
    if not cached:
        var buf = ctx.enqueue_create_buffer[DType.uint32](_CHAIN_WORDS)
        var fresh = buf.unsafe_ptr()
        for i in range(_CHAIN_WORDS):
            fresh[unsafe_offset=i] = 0
        external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
            StringSlice(_CHAIN_STATE_KEY), fresh.unsafe_bitcast[NoneType]()
        )
        _ = buf.take_ptr()
        cached = _chain_state_or_null()
    return _arena_ptr(cached.unsafe_value())


def _chain_addr_store(state: _NPUArenaPtr, word: Int, addr: Int):
    state[unsafe_offset=word] = UInt32(addr & 0xFFFFFFFF)
    state[unsafe_offset=word + 1] = UInt32((addr >> 32) & 0xFFFFFFFF)


def _chain_addr_load(state: _NPUArenaPtr, word: Int) -> Int:
    return Int(state[unsafe_offset=word]) | (
        Int(state[unsafe_offset=word + 1]) << 32
    )


def _matmul_chain_flush():
    """Fires a held chain member, if any. The 186A closure, stub half.

    Called at the top of EVERY registration in this file before it touches
    an operand -- host walks included, which is exactly the reader class
    that corrupts a held batch (`_refuse_batched_graph_path`) -- and needs
    no context: the held launch lives fully packed on the transport side,
    and `npuChainFire` runs it through `dispatchSingle`. The transport also
    fires it itself at every `flushBatch` seam (copies, synchronize, alloc,
    free, teardown), which is what closes the window a graph's LAST
    registration would otherwise leave open.
    """
    var cached = _chain_state_or_null()
    if not cached:
        return
    var state = _arena_ptr(cached.unsafe_value())
    if state[unsafe_offset=0] == 0:
        return
    state[unsafe_offset=0] = 0
    _ = external_call["npuChainFire", UInt32]()


comptime _ChainGemvPtr = Pointer[Scalar[DType.float16], MutUnsafeAnyOrigin]


def _chain_dense_2d(shape: IndexList[2], strides: IndexList[2]) -> Bool:
    """Whether a rank-2 view is dense row-major, without raising.

    The independence proof below is byte-range arithmetic over
    `base + rows * cols * 2`, which is only the tensor's true extent when
    the view is dense; anything else declines the chain and keeps the
    incumbent path.
    """
    return strides[1] == 1 and strides[0] == shape[1]


def _matmul_chain2_independent(
    c0: Int,
    c0_bytes: Int,
    a0: Int,
    a0_bytes: Int,
    b0: Int,
    b0_bytes: Int,
    c1: Int,
    c1_bytes: Int,
    a1: Int,
    a1_bytes: Int,
    b1: Int,
    b1_bytes: Int,
) -> Bool:
    """Whether two adjacent matmuls are provably independent, by byte range.

    The proof the grouper needs is exactly the scoping's dependency analysis
    (`MATMUL_CHAIN_SCOPING.md` section 2, read off `block_stack.mojo`'s
    block programs): two neighbors chain only when NEITHER reads or writes
    the other's output. In dispatch-time terms, over dense fp16 extents:

    - RAW: member 1's `A` or `B` overlaps member 0's `C` -- refused.
    - WAR: member 1's `C` overlaps member 0's `A` or `B` -- refused (the
      members share one launch, so both orders are hazards).
    - WAW: the two `C` ranges overlap -- refused.

    Shared INPUTS (`A0 == A1`, the encoder q/k/v shape) overlap freely.
    Anything not provable from the pointers and extents in hand is a
    refusal, never an assumption -- order 280's conservation law is held by
    refusing the pair, not by reordering around it.
    """

    def overlaps(lo0: Int, len0: Int, lo1: Int, len1: Int) -> Bool:
        return lo0 < lo1 + len1 and lo1 < lo0 + len0

    if overlaps(c0, c0_bytes, a1, a1_bytes):
        return False
    if overlaps(c0, c0_bytes, b1, b1_bytes):
        return False
    if overlaps(c1, c1_bytes, a0, a0_bytes):
        return False
    if overlaps(c1, c1_bytes, b0, b0_bytes):
        return False
    if overlaps(c0, c0_bytes, c1, c1_bytes):
        return False
    return True


def _matmul_chain2_eligible(m: Int, n: Int, k: Int) -> Bool:
    """Whether one member is chain-eligible: the DEVICE's own array predicate.

    `matmul_bt_hmx_accepts` is imported rather than mirrored so the two
    sides cannot drift; a member that passes here and still declines on the
    device (a non-HMX skel) falls to the chain kernel's nest arm for BOTH
    members -- correct, never mixed. M=1 gemvs and every other
    predicate-declined shape stay selector-24 singletons this batch.
    """
    return matmul_bt_hmx_accepts(m, n, k, 0, n)


def _matmul_chain_route(
    ctx: DeviceContext,
    c_ptr: Int,
    a_ptr: Int,
    b_ptr: Int,
    m: Int,
    n: Int,
    k: Int,
    dense: Bool,
) raises -> Bool:
    """Routes one matmul dispatch through the chain grouper.

    True: the dispatch was consumed (deferred as a pending member, or paired
    with the held one into a `matmul_chain2` launch). False: the caller owns
    the dispatch, and any previously held member has been flushed ahead of
    it -- singletons fall through to selector 24 unchanged.
    """
    var chain_on = _matmul_chain_requested()
    var cached = _chain_state_or_null()
    if not cached and not chain_on:
        return False
    _refuse_chain_with_queue(chain_on, getenv("MODULAR_NPU_QUEUE") == "1")

    var state = _chain_state(ctx)
    var incoming_ok = (
        chain_on and dense and _matmul_chain2_eligible(m, n, k)
    )

    if state[unsafe_offset=0] != 0:
        var c0 = _chain_addr_load(state, 1)
        var a0 = _chain_addr_load(state, 3)
        var b0 = _chain_addr_load(state, 5)
        var m0 = Int(state[unsafe_offset=9])
        var n0 = Int(state[unsafe_offset=10])
        var k0 = Int(state[unsafe_offset=11])
        if incoming_ok and _matmul_chain2_independent(
            c0,
            m0 * n0 * 2,
            a0,
            m0 * k0 * 2,
            b0,
            n0 * k0 * 2,
            c_ptr,
            m * n * 2,
            a_ptr,
            m * k * 2,
            b_ptr,
            n * k * 2,
        ):
            state[unsafe_offset=0] = 0
            var take = external_call["npuChainTakeHeld", UInt32]()
            if take == _CHAIN_TAKE_CLAIMED:
                var slot0 = _NPUArenaPtr(
                    unsafe_from_address=_chain_addr_load(state, 7)
                )
                _matmul_chain2_dispatch(
                    ctx,
                    c0,
                    a0,
                    b0,
                    slot0,
                    c_ptr,
                    a_ptr,
                    b_ptr,
                    m0,
                    n0,
                    k0,
                    m,
                    n,
                    k,
                )
                return True
            # FIRED: a flush seam ran the held member already; the incoming
            # matmul starts a fresh hold below. Any other answer is a
            # stub/transport skew, resolved the same conservative way.
        else:
            _matmul_chain_flush()

    if not incoming_ok:
        return False

    var slot = _witness_slot(ctx)
    _arm_matmul_ran(slot, "mo.matmul.chain0")
    _arm_unpack_arm_known(slot, "mo.matmul.chain0")
    var f = ctx.load_function[npu_selector_sig](
        function_name=String("matmul"), asm=String("")
    )
    state[unsafe_offset=0] = 1
    _chain_addr_store(state, 1, c_ptr)
    _chain_addr_store(state, 3, a_ptr)
    _chain_addr_store(state, 5, b_ptr)
    _chain_addr_store(state, 7, Int(slot))
    state[unsafe_offset=9] = UInt32(m)
    state[unsafe_offset=10] = UInt32(n)
    state[unsafe_offset=11] = UInt32(k)
    external_call["npuChainHoldNext", NoneType]()
    ctx.enqueue_function(
        f,
        _ChainGemvPtr(unsafe_from_address=c_ptr),
        _ChainGemvPtr(unsafe_from_address=a_ptr),
        _ChainGemvPtr(unsafe_from_address=b_ptr),
        slot,
        Int32(m),
        Int32(n),
        Int32(k),
        grid_dim=Dim(NPU_GRID_CORES),
        block_dim=Dim(NPU_THREAD_MASK),
    )
    return True


def _matmul_chain2_dispatch(
    ctx: DeviceContext,
    c0: Int,
    a0: Int,
    b0: Int,
    slot0: _NPUArenaPtr,
    c1: Int,
    a1: Int,
    b1: Int,
    m0: Int,
    n0: Int,
    k0: Int,
    m1: Int,
    n1: Int,
    k1: Int,
) raises:
    """One `matmul_chain2` launch: the claimed member 0 plus the incoming
    member 1, each with its own armed arena slot. The member-distinct op
    names are the attribution: the transport's fail-fast error carries the
    guard record's own op string, so a mid-chain failure names its member.
    """
    var slot1 = _witness_slot(ctx)
    _arm_matmul_ran(slot1, "mo.matmul.chain1")
    _arm_unpack_arm_known(slot1, "mo.matmul.chain1")
    var f = ctx.load_function[npu_selector_sig](
        function_name=String("matmul_chain2"), asm=String("")
    )
    ctx.enqueue_function(
        f,
        _ChainGemvPtr(unsafe_from_address=c0),
        _ChainGemvPtr(unsafe_from_address=a0),
        _ChainGemvPtr(unsafe_from_address=b0),
        slot0,
        _ChainGemvPtr(unsafe_from_address=c1),
        _ChainGemvPtr(unsafe_from_address=a1),
        _ChainGemvPtr(unsafe_from_address=b1),
        slot1,
        Int32(m0),
        Int32(n0),
        Int32(k0),
        Int32(m1),
        Int32(n1),
        Int32(k1),
        grid_dim=Dim(NPU_GRID_CORES),
        block_dim=Dim(NPU_THREAD_MASK),
    )


@extensibility.register("mo.matmul", type="npu", api="npu")
struct Matmul:
    """Dispatches the `matmul` selector, `C[M, N] = A[M, K] @ B[N, K]^T`."""

    @staticmethod
    def execute[
        transpose_b: Bool,
        packed_b: Bool,
        has_epilogue_fusion: Bool,
        target: StaticString,
        _trace_name: StaticString,
    ](
        c: _FusedComputeOutputTensor[rank=2, ...],
        a: InputTensor[dtype=c.dtype, rank=2, ...],
        b: InputTensor[dtype=c.dtype, rank=2, ...],
        ctx: DeviceContext,
    ) capturing raises:
        comptime assert transpose_b, (
            "mo.matmul on the on-device NPU: the `matmul` selector reads B as"
            " [N, K], so transpose_b must be set"
        )
        comptime assert (
            not packed_b
        ), "mo.matmul on the on-device NPU: packed_b is not supported"
        comptime assert (
            c.dtype == DType.float16
        ), "mo.matmul on the on-device NPU: fp16 only"
        # A fused epilogue would have to be generated as device code, and
        # generating it is not what stops us: compiling a kernel inside a live
        # host process, cataloguing and signing it unprivileged, and loading it
        # through FastRPC's own privileged loader all work today. Two other
        # mechanisms make it unloadable. The window to introduce a *new* library
        # filename closes at the session's first open attempt, so a kernel
        # produced at graph-compile time is deliverable only if its filename was
        # reserved and primed before then. And `loadFunction` resolves by name
        # against `NPUKernels.def`, which is compiled into both the host packer
        # and the device trampoline, so a freshly generated kernel has no row to
        # match: what blocks it is the name lookup, not the code generation.
        # Both are sequencing and lookup problems rather than an absent
        # capability, and whether a name-based path could bypass the static
        # table is unmeasured. Refusing at compile time names the cause; fusion
        # stays deferred on magnitude, not on feasibility.
        comptime assert not has_epilogue_fusion, (
            "mo.matmul on the on-device NPU: an epilogue cannot be fused into a"
            " pre-signed kernel"
        )

        var m = Int(a.dim_size(0))
        var k = Int(a.dim_size(1))
        var n = Int(c.dim_size(1))
        # Ahead of the k-pad staging below, which READS `a` and `b` on the
        # host: the route either consumes this dispatch (deferred or paired)
        # or has flushed any held member before the first operand read.
        if _matmul_chain_route(
            ctx,
            Int(c.unsafe_ptr()),
            Int(a.unsafe_ptr()),
            Int(b.unsafe_ptr()),
            m,
            n,
            k,
            _chain_dense_2d(c.shape(), c.strides())
            and _chain_dense_2d(a.shape(), a.strides())
            and _chain_dense_2d(b.shape(), b.strides()),
        ):
            return
        if k % 64 != 0:
            _require_dense(a.shape(), a.strides(), "mo.matmul", "A")
            _require_dense(b.shape(), b.strides(), "mo.matmul", "B")
            var k_pad = ((k + 63) // 64) * 64
            var a_padded = ctx.enqueue_create_buffer[c.dtype](m * k_pad)
            var b_padded = ctx.enqueue_create_buffer[c.dtype](n * k_pad)
            _stage_k_padded_rows(
                a_padded.unsafe_ptr(), a.unsafe_ptr(), m, k, k_pad
            )
            _stage_weight_rows(
                b_padded.unsafe_ptr(), b.unsafe_ptr(), n, k, k_pad, m
            )
            var f_pad = ctx.load_function[npu_selector_sig](
                function_name=String("matmul"), asm=String("")
            )
            var pad_witness = _witness_slot(ctx)
            _arm_matmul_ran(pad_witness, "mo.matmul")
            _arm_unpack_arm_known(pad_witness, "mo.matmul")
            ctx.enqueue_function(
                f_pad,
                c.unsafe_ptr(),
                a_padded.unsafe_ptr(),
                b_padded.unsafe_ptr(),
                pad_witness,
                Int32(m),
                Int32(n),
                Int32(k_pad),
                grid_dim=Dim(NPU_GRID_CORES),
                block_dim=Dim(NPU_THREAD_MASK),
            )
            _ = a_padded^
            _ = b_padded^
            return

        var f = ctx.load_function[npu_selector_sig](
            function_name=String("matmul"), asm=String("")
        )
        # Armed before the launch and evaluated at the flush, not read back
        # here. The read-after-enqueue this replaced rested on "an invocation on
        # this transport completes before it returns", which batching is exactly
        # what makes untrue: the slot reads unwritten, so the guard would refuse
        # a launch that runs correctly moments later (PR #442).
        var witness_ptr = _witness_slot(ctx)
        _arm_matmul_ran(witness_ptr, "mo.matmul")
        _arm_unpack_arm_known(witness_ptr, "mo.matmul")
        ctx.enqueue_function(
            f,
            c.unsafe_ptr(),
            a.unsafe_ptr(),
            b.unsafe_ptr(),
            witness_ptr,
            Int32(m),
            Int32(n),
            Int32(k),
            grid_dim=Dim(NPU_GRID_CORES),
            block_dim=Dim(NPU_THREAD_MASK),
        )


@extensibility.register("mo.batch_matmul", type="npu", api="npu")
struct BatchMatmul:
    """Dispatches the `matmul` selector once per leading-dimension slice.

    `mo.matmul` is rank-2 by op contract, so a Linear applied to a
    `[batch, seq, dim]` activation legalizes to this op instead and `Matmul`
    above is reached only where the activation is already rank 2 -- the timestep
    and modulation projections of a diffusion graph, a minority of its matmuls.
    The kernel is the same either way, so this needs no device code: it walks
    the leading dims and dispatches each `[M, K] @ [N, K]^T` slice, which is
    what the
    discrete card's overlay does for the same reason
    (`../qaic/batch_matmul.mojo`).

    One dispatch per slice rather than one for the whole batch, because the
    signed kernel set has no batched entry point and no row of `NPUKernels.def`
    to name one by. At the batch of 1 a diffusion graph stages, that is a single
    dispatch and identical work to `Matmul`.
    """

    @staticmethod
    def execute[
        has_epilogue_fusion: Bool,
        rank: Int,
        transpose_b: Bool,
        target: StaticString,
    ](
        c: _FusedComputeOutputTensor[rank=rank, ...],
        a: InputTensor[dtype=c.dtype, rank=rank, ...],
        b: InputTensor[dtype=c.dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        comptime assert transpose_b, (
            "mo.batch_matmul on the on-device NPU: the `matmul` selector"
            " reads B as [N, K], so transpose_b must be set"
        )
        # No `matmul_bf16` row exists, so unlike the two norms this cannot be
        # widened by naming a second selector: bf16 here needs device code.
        comptime assert (
            c.dtype == DType.float16
        ), "mo.batch_matmul on the on-device NPU: fp16 only"
        comptime assert not has_epilogue_fusion, (
            "mo.batch_matmul on the on-device NPU: an epilogue cannot be fused"
            " into a pre-signed kernel"
        )
        comptime assert rank >= 2, (
            "mo.batch_matmul on the on-device NPU: the `matmul` selector"
            " reads a 2-D slice, so rank must be at least 2"
        )

        var m = Int(a.dim_size(rank - 2))
        var k = Int(a.dim_size(rank - 1))
        var n = Int(b.dim_size(rank - 2))
        if Int(b.dim_size(rank - 1)) != k:
            raise Error(
                (
                    "mo.batch_matmul on the on-device NPU: B is read as [N, K],"
                    " so its inner extent must match A's, got "
                ),
                Int(b.dim_size(rank - 1)),
                " against ",
                k,
            )
        if Int(c.dim_size(rank - 2)) != m or Int(c.dim_size(rank - 1)) != n:
            raise Error(
                (
                    "mo.batch_matmul on the on-device NPU: the output must be"
                    " [M, N] per slice, got "
                ),
                Int(c.dim_size(rank - 2)),
                " by ",
                Int(c.dim_size(rank - 1)),
                " against ",
                m,
                " by ",
                n,
            )
        # The op admits batch extents that broadcast against each other, while
        # the slice arithmetic below steps all three operands by the same index.
        # An operand carrying one batch where another carries several would be
        # read at the wrong offset for every slice past the first.
        var num_batches = 1
        for i in range(rank - 2):
            var extent = Int(a.dim_size(i))
            if Int(b.dim_size(i)) != extent or Int(c.dim_size(i)) != extent:
                raise Error(
                    (
                        "mo.batch_matmul on the on-device NPU: the operands'"
                        " batch extents must agree, so a broadcast batch"
                        " needs a stub that strides them separately; axis "
                    ),
                    i,
                    " is ",
                    extent,
                    ", ",
                    Int(b.dim_size(i)),
                    " and ",
                    Int(c.dim_size(i)),
                )
            num_batches *= extent

        _require_dense(c.shape(), c.strides(), "mo.batch_matmul", "the output")
        _require_dense(a.shape(), a.strides(), "mo.batch_matmul", "A")
        _require_dense(b.shape(), b.strides(), "mo.batch_matmul", "B")

        var a_slice = m * k
        var b_slice = n * k
        var c_slice = m * n
        var a_ptr = a.unsafe_ptr()
        var b_ptr = b.unsafe_ptr()
        var c_ptr = c.unsafe_ptr()

        if k % 64 != 0:
            # The staging below READS `a` and `b` on the host; a held chain
            # member must run first. (`enqueue_create_buffer` also flushes
            # through the transport seam, but the read is what binds.)
            _matmul_chain_flush()
            var k_pad = ((k + 63) // 64) * 64
            var a_padded = ctx.enqueue_create_buffer[c.dtype](
                num_batches * m * k_pad
            )
            var b_padded = ctx.enqueue_create_buffer[c.dtype](
                num_batches * n * k_pad
            )
            _stage_k_padded_rows(
                a_padded.unsafe_ptr(), a_ptr, num_batches * m, k, k_pad
            )
            # Per slice, not over `num_batches * n` rows at once: the crouton
            # artefact is column-panel-major within ONE `[n, k_pad]` operand,
            # so each slice is its own repack even though the row-major
            # staging would have been a single pass.
            for i in range(num_batches):
                _stage_weight_rows(
                    b_padded.unsafe_ptr().unsafe_offset(i * n * k_pad),
                    b_ptr.unsafe_offset(i * b_slice),
                    n,
                    k,
                    k_pad,
                    m,
                )
            var f_pad = ctx.load_function[npu_selector_sig](
                function_name=String("matmul"), asm=String("")
            )
            for i in range(num_batches):
                var pad_witness = _witness_slot(ctx)
                _arm_matmul_ran(pad_witness, "mo.batch_matmul")
                _arm_unpack_arm_known(pad_witness, "mo.batch_matmul")
                ctx.enqueue_function(
                    f_pad,
                    c_ptr.unsafe_offset(i * c_slice),
                    a_padded.unsafe_ptr().unsafe_offset(i * m * k_pad),
                    b_padded.unsafe_ptr().unsafe_offset(i * n * k_pad),
                    pad_witness,
                    Int32(m),
                    Int32(n),
                    Int32(k_pad),
                    grid_dim=Dim(NPU_GRID_CORES),
                    block_dim=Dim(NPU_THREAD_MASK),
                )
            _ = a_padded^
            _ = b_padded^
            return

        var f = ctx.load_function[npu_selector_sig](
            function_name=String("matmul"), asm=String("")
        )
        for i in range(num_batches):
            # Adjacent slices of one batch_matmul are independent by
            # construction (disjoint slices of dense operands), so the route
            # pairs them exactly as it pairs adjacent ops; at the production
            # leading extent of 1 this is the cross-op path.
            if _matmul_chain_route(
                ctx,
                Int(c_ptr.unsafe_offset(i * c_slice)),
                Int(a_ptr.unsafe_offset(i * a_slice)),
                Int(b_ptr.unsafe_offset(i * b_slice)),
                m,
                n,
                k,
                True,
            ):
                continue
            # A SLOT PER SLICE, not one witness across them. The old comment
            # here reasoned that the dispatches cannot overlap, so the last
            # report was the one worth reading -- and then re-seeded per slice
            # anyway, because a slice the array refused followed by one it ran
            # would report the run and leave the refused slice's output
            # unwritten. Under batching every slice IS in flight at once, so
            # that re-seed is not enough and the slot has to be.
            var witness_ptr = _witness_slot(ctx)
            _arm_matmul_ran(witness_ptr, "mo.batch_matmul")
            _arm_unpack_arm_known(witness_ptr, "mo.batch_matmul")
            ctx.enqueue_function(
                f,
                c_ptr.unsafe_offset(i * c_slice),
                a_ptr.unsafe_offset(i * a_slice),
                b_ptr.unsafe_offset(i * b_slice),
                witness_ptr,
                Int32(m),
                Int32(n),
                Int32(k),
                grid_dim=Dim(NPU_GRID_CORES),
                block_dim=Dim(NPU_THREAD_MASK),
            )


@always_inline
def _rms_norm_selector[dtype: DType]() -> StaticString:
    """Names the signed `rms_norm` row compiled for `dtype`.

    A second selector rather than a flag on one, for the reason
    `_layernorm_selector` gives: the device library's kernel set is fixed when it
    is signed, so a dtype reaches an entry point built for it or none. Both rows
    take the same buffers and scalars, so the name is the only thing that differs
    between the two dispatches -- which is also what makes a right answer
    evidence of the right pick, the wrong kernel reading the same bytes as a
    different number entirely.

    `comptime if` rather than a conditional expression so that one name reaches
    the object; see `_layernorm_selector` for what an expression costs.
    """
    comptime if dtype == DType.float16:
        return "rms_norm"
    else:
        return "rms_norm_bf16"


@extensibility.register("mo.reduce.rms_norm", type="npu", api="npu")
struct RMSNorm:
    """Dispatches an `rms_norm` selector over the last axis.

    `NPUKernels.def` carries a bf16 twin of this selector at the same wire shape,
    so the dtype picks between two signed kernels.
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        multiply_before_cast: Bool = True,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        gamma: InputTensor[dtype=dtype, rank=1, ...],
        epsilon: Float32,
        weight_offset: Scalar[dtype=dtype],
        ctx: DeviceContext,
    ) capturing raises:
        comptime assert (
            dtype == DType.float16 or dtype == DType.bfloat16
        ), "mo.reduce.rms_norm on the on-device NPU: fp16 or bf16 only"
        if output.shape() != input.shape():
            raise Error(
                "mo.reduce.rms_norm on the on-device NPU: input and output"
                " shapes differ"
            )

        var length = Int(input.dim_size(rank - 1))
        var num_rows = 1
        for i in range(rank - 1):
            num_rows *= Int(input.dim_size(i))

        var f = ctx.load_function[npu_selector_sig](
            function_name=String(_rms_norm_selector[dtype]()), asm=String("")
        )
        # `weight_offset` travels single-precision and the kernel narrows it
        # itself, keeping the half-precision element type where the arithmetic
        # is. `multiply_before_cast` is a kernel parameter both instantiations
        # of which are compiled, so it is selected by this scalar.
        ctx.enqueue_function(
            f,
            output.unsafe_ptr(),
            input.unsafe_ptr(),
            gamma.unsafe_ptr(),
            _witness_slot(ctx),
            Int32(num_rows),
            Int32(length),
            epsilon,
            Float32(weight_offset),
            Int32(1) if multiply_before_cast else Int32(0),
            grid_dim=Dim(NPU_GRID_CORES),
            block_dim=Dim(NPU_THREAD_MASK),
        )


def _require_identity_affine(
    weight: InputTensor[rank=1, ...],
    identity: Scalar[dtype=weight.dtype],
    name: StaticString,
) raises:
    """Refuses an affine weight the `layernorm` selector would drop.

    The selector normalizes and stops there: `NPUKernels.def` gives it output,
    input and witness, and no weights. `mo.reduce.layer_norm` is
    `gamma * normed + beta`, so weights that are not the identity would be
    dropped and the result would come back as though the graph had asked for
    none. The values decide rather than the signature, because a graph that
    means no affine still arrives carrying real tensors -- `np.ones` /
    `np.zeros` from `ConstantLayerNorm`, a `broadcast_to` of 1.0 / 0.0 from the
    DiT blocks -- and that case the selector computes exactly. There is no
    signature to refuse at compile time instead, the weights being graph values.

    Reading them on the host is what this transport allows where a discrete card
    would not: its memory is `rpcmem` shared with this process and an invocation
    completes before it returns, so there is neither a copy to stage nor
    outstanding work to wait on (`QuRTDeviceContext::enqueueCopyToDevice`,
    `::synchronize`). Indexed rather than pointer-linear, so that a broadcast
    weight's zero stride is honoured.
    """
    for i in range(Int(weight.dim_size(0))):
        if weight[i] != identity:
            raise Error(
                (
                    "mo.reduce.layer_norm on the on-device NPU: the `layernorm`"
                    " selector applies no affine, so "
                ),
                name,
                " must be ",
                identity,
                " throughout; element ",
                i,
                " is ",
                weight[i],
            )


@always_inline
def _layernorm_selector[dtype: DType]() -> StaticString:
    """Names the signed `layernorm` row compiled for `dtype`.

    A second selector rather than a flag on one, because the device library's
    kernel set is fixed when it is signed: a dtype reaches an entry point built
    for it or none. Both rows take the same buffers and scalars, so the name is
    the only thing that differs between the two dispatches.

    `comptime if` rather than a conditional expression so that one name reaches
    the object. Both branches of an expression are emitted as constants even
    when the condition folds, which would put both selector names in every
    exported MEF and leave the artifact unable to say which one a graph
    dispatches.
    """
    comptime if dtype == DType.float16:
        return "layernorm"
    else:
        return "layernorm_bf16"


@extensibility.register("mo.reduce.layer_norm", type="npu", api="npu")
struct LayerNorm:
    """Dispatches a `layernorm` selector, refusing the affine it drops.

    The only registration here that is not fp16-only: `NPUKernels.def` carries a
    bf16 twin of this selector, so the dtype picks between two signed kernels.
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        gamma: InputTensor[dtype=dtype, rank=1, ...],
        beta: InputTensor[dtype=dtype, rank=1, ...],
        epsilon: Float32,
        ctx: DeviceContext,
    ) capturing raises:
        comptime assert (
            dtype == DType.float16 or dtype == DType.bfloat16
        ), "mo.reduce.layer_norm on the on-device NPU: fp16 or bf16 only"
        if output.shape() != input.shape():
            raise Error(
                "mo.reduce.layer_norm on the on-device NPU: input and output"
                " shapes differ"
            )
        _require_identity_affine(gamma, 1, "gamma")
        _require_identity_affine(beta, 0, "beta")

        var length = Int(input.dim_size(rank - 1))
        var num_rows = 1
        for i in range(rank - 1):
            num_rows *= Int(input.dim_size(i))

        var f = ctx.load_function[npu_selector_sig](
            function_name=String(_layernorm_selector[dtype]()), asm=String("")
        )
        ctx.enqueue_function(
            f,
            output.unsafe_ptr(),
            input.unsafe_ptr(),
            _witness_slot(ctx),
            Int32(num_rows),
            Int32(length),
            epsilon,
            grid_dim=Dim(NPU_GRID_CORES),
            block_dim=Dim(NPU_THREAD_MASK),
        )


@extensibility.register("mo.reduce.softmax", type="npu", api="npu")
struct Softmax:
    """Dispatches the `softmax` selector over the last axis.

    The input is a plain `InputTensor` rather than a fused one so that
    `unsafe_ptr()` addresses a materialized buffer: a prologue fusion would
    produce its data from a lambda the device cannot run.
    """

    @staticmethod
    def execute[
        target: StaticString,
        axis: Int,
    ](
        output: OutputTensor,
        input: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        comptime assert (
            output.dtype == DType.float16
        ), "mo.reduce.softmax on the on-device NPU: fp16 only"
        comptime assert (
            axis == output.rank - 1
        ), "mo.reduce.softmax on the on-device NPU: last-axis only"

        var length = Int(output.dim_size(output.rank - 1))
        var num_rows = Int(output.size()) // length

        var f = ctx.load_function[npu_selector_sig](
            function_name=String("softmax"), asm=String("")
        )
        ctx.enqueue_function(
            f,
            output.unsafe_ptr(),
            input.unsafe_ptr(),
            _witness_slot(ctx),
            Int32(num_rows),
            Int32(length),
            grid_dim=Dim(NPU_GRID_CORES),
            block_dim=Dim(NPU_THREAD_MASK),
        )


@always_inline
def _groupnorm_selector[dtype: DType]() -> StaticString:
    """Names the signed `groupnorm` row compiled for `dtype`.

    The same two-row arrangement as `_layernorm_selector`, for the same reason:
    the kernel set is fixed when the library is signed, so a dtype reaches a row
    built for it or none. Both rows take the same buffers and scalars, so the
    name is all that differs between the two dispatches.

    `comptime if` rather than a conditional expression so that one name reaches
    the object, as `_layernorm_selector` explains.
    """
    comptime if dtype == DType.float16:
        return "groupnorm"
    else:
        return "groupnorm_bf16"


@extensibility.register("mo.reduce.group_norm", type="npu", api="npu")
struct GroupNorm:
    """Dispatches a `groupnorm` selector over `[N, C, spatial]`.

    Like `LayerNorm`, and unlike the rest of this file, not fp16-only:
    `NPUKernels.def` carries a bf16 twin of this selector, so the dtype picks
    between two signed kernels.
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        gamma: InputTensor[dtype=dtype, rank=1, ...],
        beta: InputTensor[dtype=dtype, rank=1, ...],
        epsilon: Float32,
        num_groups: Int32,
        ctx: DeviceContext,
    ) capturing raises:
        # A held matmul-chain member must run before this op touches any
        # operand: this is the 186A host-reader class, closed per site.
        _matmul_chain_flush()
        comptime assert (
            dtype == DType.float16 or dtype == DType.bfloat16
        ), "mo.reduce.group_norm on the on-device NPU: fp16 or bf16 only"
        comptime assert rank >= 3, (
            "mo.reduce.group_norm on the on-device NPU: the selector reads [N,"
            " C, spatial], so rank must be at least 3"
        )
        if output.shape() != input.shape():
            raise Error(
                "mo.reduce.group_norm on the on-device NPU: input and output"
                " shapes differ"
            )

        var n = Int(input.dim_size(0))
        var channels = Int(input.dim_size(1))
        var spatial = 1
        for i in range(2, rank):
            spatial *= Int(input.dim_size(i))

        # The selector sizes a group as `C // num_groups` and walks the groups
        # as equal contiguous blocks, so an indivisible channel count would
        # leave the remainder unnormalized instead of failing -- the same reason
        # MAX's own `group_norm` refuses it rather than rounding.
        if channels % Int(num_groups) != 0:
            raise Error(
                (
                    "mo.reduce.group_norm on the on-device NPU: channels must"
                    " divide into num_groups, got "
                ),
                channels,
                " channels in ",
                num_groups,
                " groups",
            )

        var f = ctx.load_function[npu_selector_sig](
            function_name=String(_groupnorm_selector[dtype]()), asm=String("")
        )
        ctx.enqueue_function(
            f,
            output.unsafe_ptr(),
            input.unsafe_ptr(),
            gamma.unsafe_ptr(),
            beta.unsafe_ptr(),
            _witness_slot(ctx),
            Int32(n),
            Int32(channels),
            Int32(spatial),
            num_groups,
            epsilon,
            grid_dim=Dim(NPU_GRID_CORES),
            block_dim=Dim(NPU_THREAD_MASK),
        )


def _per_channel_rows[
    rank: Int
](
    param: InputTensor[rank=rank, ...],
    row_length: Int,
    outer: Int,
    op: StaticString,
    name: StaticString,
) raises -> Int:
    """Returns how many rows a per-channel parameter carries: 1, or `outer`.

    The composite's verifier already holds the shape to the trailing axis with
    at most one row per outermost index. Reading the extents again here keeps
    both halves of the contract stated where each is enforced: the offsets the
    dispatch steps the parameter by are derived from this, and a wrong one
    reads past the allocation.
    """
    if Int(param.dim_size(rank - 1)) != row_length:
        raise Error(
            op,
            " on the on-device NPU: ",
            name,
            " must carry one element per channel; got ",
            Int(param.dim_size(rank - 1)),
            " for a row length of ",
            row_length,
        )
    var rows = 1
    for i in range(rank - 1):
        rows *= Int(param.dim_size(i))
    # The per-outer offset arithmetic steps the parameter one row per
    # outermost index, so the rows must come from that axis alone.
    if rows != 1 and (rows != outer or Int(param.dim_size(0)) != rows):
        raise Error(
            op,
            " on the on-device NPU: ",
            name,
            " must carry one row, or one per outermost index; got ",
            rows,
            " rows against ",
            outer,
        )
    return rows


@always_inline
def _rope_selector[dtype: DType, interleaved: Bool]() -> StaticString:
    """Names the signed rope row for `dtype` and the pairing convention.

    Four rows rather than a flag on one, for the reason `_layernorm_selector`
    gives: the device library's kernel set is fixed when it is signed, so a
    dtype reaches an entry point built for it or none. All four take the same
    buffers and scalars, so the name is the only thing that differs between the
    dispatches.

    `comptime if` throughout rather than a conditional expression, so that one
    name reaches the object -- see `_layernorm_selector` for what an expression
    would cost the exported artifact. It is also what keeps the two bf16 names
    out of an fp16 graph's artifact entirely: `NPUKernels.def` carries the fp16
    rows, and the bf16 twins arrive with the bf16 kernel work, so until then a
    bf16 graph fails at `loadFunction` by name rather than silently.
    """
    comptime if interleaved:
        comptime if dtype == DType.float16:
            return "rope_interleaved"
        else:
            return "rope_interleaved_bf16"
    else:
        comptime if dtype == DType.float16:
            return "rope_half"
        else:
            return "rope_half_bf16"


@extensibility.register(
    "mo.rope.ragged.with_position_id", type="npu", api="npu"
)
struct RopeRaggedWithPositionId[interleaved: Bool]:
    """Dispatches a rope selector over `[tokens, heads, head_size]`.

    The signature conversion the file header names as what stands between the
    graph's ragged RoPE and the dense selectors. Three of its four parts turn
    out to be identities:

    - `freqs_cis` already arrives as the selector's own layout. The built-in
      loads `freqs_cis[position_ids[t], head_dim_idx]` and deinterleaves it into
      cos and sin (`max/kernels/src/nn/rope.mojo`, `_rope`), which is exactly
      "cos/sin interleaved per pair" -- and FLUX.2 stages it fp32, the dtype the
      selector's `const float *` reads.
    - the `interleaved` convention maps one to one: the built-in's interleaved
      branch pairs `(2p, 2p+1)` and its `rotate_half` branch pairs `p` with
      `p + half`, which are the two selectors.
    - the ragged framing is inert here rather than merely degenerate.
      `rope.mojo` computes a position from `input_row_offsets` and `start_pos`
      and then *overwrites* it with `position_ids[t]` whenever position ids are
      present, so for this op neither operand can change a result. FLUX.2
      synthesizes both (`max/python/max/nn/kernels.py`,
      `rope_ragged_with_position_ids`) precisely because they are unused.

    What is real is that `position_ids` is per *token* while the selector's
    `positions` is per *row*, and a row is one head of one token. The expansion
    is a host fill, which this transport allows for the reason
    `_require_identity_affine` records: device memory is `rpcmem` shared with
    this process.

    Plain tensors on both sides. `Softmax` gives the reason for the input side;
    the output side needs it for the same reason, and `mo.reduce.layer_norm`
    already narrows a `FusedOutputTensor` to a plain one here and runs exact on
    silicon. A fused epilogue stores through a lambda, and `unsafe_ptr()` has to
    address a buffer the device can write.

    Parameters:
        interleaved: Whether RoPE pairs adjacent components rather than
            splitting them into halves, as the built-in registration's parameter
            of the same name does.
    """

    @staticmethod
    def execute[
        dtype: DType,
        freq_dtype: DType,
        //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=3, ...],
        x: InputTensor[dtype=dtype, rank=3, ...],
        input_row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        start_pos: InputTensor[dtype=DType.uint32, rank=1, ...],
        freqs_cis: InputTensor[dtype=freq_dtype, rank=2, ...],
        position_ids: InputTensor[dtype=DType.uint32, rank=2, ...],
        ctx: DeviceContext,
    ) capturing raises:
        comptime assert dtype == DType.float16 or dtype == DType.bfloat16, (
            "mo.rope.ragged.with_position_id on the on-device NPU: fp16 or bf16"
            " only"
        )
        comptime assert freq_dtype == DType.float32, (
            "mo.rope.ragged.with_position_id on the on-device NPU: the rope"
            " selectors read cos/sin as fp32"
        )

        if output.shape() != x.shape():
            raise Error(
                "mo.rope.ragged.with_position_id on the on-device NPU: input"
                " and output shapes differ"
            )
        _require_dense(
            x.shape(),
            x.strides(),
            "mo.rope.ragged.with_position_id",
            "the input",
        )
        _require_dense(
            output.shape(),
            output.strides(),
            "mo.rope.ragged.with_position_id",
            "the output",
        )
        _require_dense(
            freqs_cis.shape(),
            freqs_cis.strides(),
            "mo.rope.ragged.with_position_id",
            "freqs_cis",
        )

        var tokens = Int(x.dim_size(0))
        var heads = Int(x.dim_size(1))
        var head_size = Int(x.dim_size(2))
        var rope_dim = Int(freqs_cis.dim_size(1))

        # A partial rope leaves the leading `head_size - rope_dim` elements
        # unrotated, which the selector has no way to express: it reads one
        # cos/sin row per head vector and rotates all of it. Refusing names the
        # cause where rotating the unroped region would return plausible
        # numbers.
        if rope_dim != head_size:
            raise Error(
                (
                    "mo.rope.ragged.with_position_id on the on-device NPU: the"
                    " rope selectors rotate the whole head vector, so"
                    " freqs_cis' rope_dim must equal head_size, got "
                ),
                rope_dim,
                " against ",
                head_size,
            )
        if head_size % 2 != 0:
            raise Error(
                (
                    "mo.rope.ragged.with_position_id on the on-device NPU: rope"
                    " pairs elements, so head_size must be even, got "
                ),
                head_size,
            )

        var max_position = Int(freqs_cis.dim_size(0))
        var stated_tokens = Int(position_ids.dim_size(1))
        if stated_tokens < tokens:
            raise Error(
                (
                    "mo.rope.ragged.with_position_id on the on-device NPU:"
                    " position_ids carries "
                ),
                stated_tokens,
                " positions for ",
                tokens,
                " tokens",
            )

        # One freqs row per head vector: `positions[t * heads + h]` is token
        # `t`'s position, which is what the selector multiplies by `head_size`
        # to reach a row. Read from row 0 of `position_ids` and indexed rather
        # than pointer-linear, as `rope.mojo` reads it and as
        # `_require_identity_affine` reads its weight -- a broadcast row's zero
        # stride is then honoured.
        var num_rows = tokens * heads
        var positions = ctx.enqueue_create_buffer[DType.uint32](num_rows)
        var positions_ptr = positions.unsafe_ptr()
        for t in range(tokens):
            var pos = position_ids[0, t]
            # Out of range would read whatever follows the freqs allocation and
            # rope with it. The bound is free here: the expansion already visits
            # every position.
            if Int(pos) >= max_position:
                raise Error(
                    (
                        "mo.rope.ragged.with_position_id on the on-device NPU:"
                        " position_ids["
                    ),
                    t,
                    "] is ",
                    pos,
                    ", past freqs_cis' ",
                    max_position,
                    " rows",
                )
            for h in range(heads):
                positions_ptr[unsafe_offset=t * heads + h] = pos

        var f = ctx.load_function[npu_selector_sig](
            function_name=String(_rope_selector[dtype, Self.interleaved]()),
            asm=String(""),
        )
        ctx.enqueue_function(
            f,
            output.unsafe_ptr(),
            x.unsafe_ptr(),
            freqs_cis.unsafe_ptr(),
            positions_ptr,
            _witness_slot(ctx),
            Int32(num_rows),
            Int32(head_size),
            grid_dim=Dim(NPU_GRID_CORES),
            block_dim=Dim(NPU_THREAD_MASK),
        )
        # Neither operand reaches the selector: see the struct docstring for why
        # the built-in cannot let them change a result.
        _ = input_row_offsets
        _ = start_pos
        _ = positions^


comptime _ROPE_FUSED_STATE_KEY: StaticString = "npu_rope_fused_norm_state"

comptime _RopeStatePtr = Pointer[UInt64, MutUnsafeAnyOrigin]

comptime _ROPE_ARM_UNPROBED: UInt64 = 0
comptime _ROPE_ARM_FUSED: UInt64 = 1
comptime _ROPE_ARM_LEGACY: UInt64 = 2

comptime _ROPE_ST_ARM = 0
comptime _ROPE_ST_PACK = 1
comptime _ROPE_ST_PACK_WORDS = 2
comptime _ROPE_ST_KEY_FREQS = 3
comptime _ROPE_ST_KEY_POS = 4
comptime _ROPE_ST_ROWS = 5
comptime _ROPE_ST_LENGTH = 6
comptime _ROPE_ST_SCRATCH = 7
comptime _ROPE_ST_HDR_OFF = 8
comptime _ROPE_ST_POS_OFF = 9
comptime _ROPE_ST_FREQS_OFF = 10
comptime _ROPE_ST_SAMPLE = 11
comptime _ROPE_ST_WORDS = 12

comptime _ROPE_PACK_SEED: UInt32 = 0xDEADDEAD


def _rope_fused_state() -> _RopeStatePtr:
    """The process-lifetime state of the ROPE_FUSED_NORM host half.

    The same global-table mechanism as `_witness_arena`, for the same reason: a
    registration is a static `execute` with nowhere to keep state, and the
    probed arm, the rope pack and the legacy scratch must outlive every
    dispatch. Never freed.
    """
    var cached = _get_global_or_null(_ROPE_FUSED_STATE_KEY)
    if not cached:
        var fresh = unsafe_alloc[UInt64](_ROPE_ST_WORDS)
        for i in range(_ROPE_ST_WORDS):
            fresh[unsafe_offset=i] = 0
        external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
            StringSlice(_ROPE_FUSED_STATE_KEY), fresh.unsafe_bitcast[NoneType]()
        )
        cached = _get_global_or_null(_ROPE_FUSED_STATE_KEY)
    return _RopeStatePtr(unsafe_from_address=Int(cached.unsafe_value()))


def _rope_pack_layout(rows: Int, freq_words: Int) -> IndexList[4]:
    """Byte offsets of the pack's regions: (header, positions, freqs, total).

    One mapped allocation, everything witness-relative, because a byte offset
    survives the transport only within one allocation
    (`QuRTDeviceContext.cpp`'s per-argument buffer resolution). The layout is
    the block-stack host's with a single header: the graph path's rope rows
    always start at its own positions table's row 0, so the double block's
    second header has no counterpart here.
    """
    var hdr = ((NPU_WITNESS_BYTES + 127) // 128) * 128
    var pos = hdr + 128
    var freqs = ((pos + rows * 4 + 127) // 128) * 128
    var total = freqs + freq_words * 4
    return IndexList[4](hdr, pos, freqs, total)


def _rope_pack_sample(
    position_ids: InputTensor[dtype=DType.uint32, rank=2, ...],
    tokens: Int,
    freqs: Pointer[UInt32, MutUnsafeAnyOrigin],
    freq_words: Int,
) -> UInt64:
    """A cheap content fingerprint over both tables' first and last words.

    The pack caches by source pointer, and a freed-and-reallocated step input
    can land at the same address with different bytes. Sampling the edges
    catches real table drift for ~256 bytes of reads per dispatch.
    TODO(skye): a full hash if a graph ever stages tables that differ only in
    the interior; FLUX's are step-invariant.
    """
    var h: UInt64 = 0xCBF29CE484222325
    comptime FNV: UInt64 = 0x100000001B3
    for i in range(min(16, freq_words)):
        h = (h ^ UInt64(freqs[unsafe_offset=i])) * FNV
        h = (h ^ UInt64(freqs[unsafe_offset=freq_words - 1 - i])) * FNV
    for i in range(min(16, tokens)):
        h = (h ^ UInt64(position_ids[0, i])) * FNV
        h = (h ^ UInt64(position_ids[0, tokens - 1 - i])) * FNV
    return h


def _rope_pack_write_header(
    pack: Pointer[UInt32, MutUnsafeAnyOrigin],
    layout: IndexList[4],
    length: Int,
    freq_rows: Int,
):
    """Seeds the witness words and writes the fused-contract header."""
    for i in range(NPU_WITNESS_BYTES // 4):
        pack[unsafe_offset=i] = _ROPE_PACK_SEED
    var hdr = layout[0] // 4
    pack[unsafe_offset=hdr] = UInt32(RMS_ROPE_MAGIC)
    pack[unsafe_offset=hdr + 1] = UInt32(layout[1])
    pack[unsafe_offset=hdr + 2] = UInt32(layout[2])
    pack[unsafe_offset=hdr + 3] = UInt32(length)
    pack[unsafe_offset=hdr + 4] = UInt32(freq_rows)
    pack[unsafe_offset=hdr + 5] = 0
    pack[unsafe_offset=hdr + 6] = 0
    pack[unsafe_offset=hdr + 7] = 0


def _rope_fused_probe(
    ctx: DeviceContext, length: Int, epsilon: Float32
) raises -> UInt64:
    """One-row probe dispatch: the skel says which arm it computes.

    THE ARM IS ECHOED, NOT ASSUMED (order 229, via #475's redesign): a complete
    fused contract on scratch, and witness word `RMS_WITNESS_ARM_WORD` read
    back -- rpcmem is host-coherent and the transport is synchronous, so the
    readback is a plain load. A flag-off skel never writes the word, so the
    surviving seed is the legacy verdict; 2 means a fused skel REFUSED the very
    contract production would use, which no render may proceed past.
    """
    var f = ctx.load_function[npu_selector_sig](
        function_name=String(_rms_norm_selector[DType.float16]()),
        asm=String(""),
    )
    var layout = _rope_pack_layout(1, length)
    var pack_buf = ctx.enqueue_create_buffer[DType.uint32](layout[3] // 4)
    var pack = Pointer[UInt32, MutUnsafeAnyOrigin](
        unsafe_from_address=Int(pack_buf.unsafe_ptr())
    )
    _rope_pack_write_header(pack, layout, length, 1)
    pack[unsafe_offset=layout[1] // 4] = 0
    for i in range(length):
        pack[unsafe_offset=layout[2] // 4 + i] = 0
    var out_buf = ctx.enqueue_create_buffer[DType.float16](length)
    var x_buf = ctx.enqueue_create_buffer[DType.float16](length)
    var gamma_buf = ctx.enqueue_create_buffer[DType.float16](length)
    for i in range(length):
        out_buf.unsafe_ptr()[unsafe_offset=i] = Scalar[DType.float16](0)
        x_buf.unsafe_ptr()[unsafe_offset=i] = Scalar[DType.float16](0)
        gamma_buf.unsafe_ptr()[unsafe_offset=i] = Scalar[DType.float16](0)
    ctx.enqueue_function(
        f,
        out_buf.unsafe_ptr(),
        x_buf.unsafe_ptr(),
        gamma_buf.unsafe_ptr(),
        pack,
        Int32(1),
        Int32(length),
        epsilon,
        Float32(0),
        Int32(layout[0]),
        grid_dim=Dim(NPU_GRID_CORES),
        block_dim=Dim(NPU_THREAD_MASK),
    )
    var echoed = pack[unsafe_offset=RMS_WITNESS_ARM_WORD]
    if echoed == 2:
        raise Error(
            "fused_rms_norm_rope: the fused skel refused the production"
            " contract at the arm probe -- host/kernel header mismatch, fix"
            " before any render"
        )
    if echoed == 1:
        print(
            "graph-path qk-norm arm from the skel witness: rope-fused=1 --"
            " rope dispatches fused into selector 5"
        )
        return _ROPE_ARM_FUSED
    if echoed == _ROPE_PACK_SEED:
        print(
            "graph-path qk-norm arm from the skel witness: rope-fused=0 --"
            " sequential norm+rope pair kept"
        )
        return _ROPE_ARM_LEGACY
    raise Error(
        (
            "fused_rms_norm_rope: the arm probe's witness word is"
            " neither 1, 2 nor the seed -- unrecognized skel echo "
        ),
        Int(echoed),
    )


@extensibility.register("fused_rms_norm_rope", type="npu")
struct RMSNormRope:
    """Dispatches the ROPE_FUSED_NORM contract: qk rms_norm and interleaved
    rope as one selector-5 pass, host half of `rms_norm_kernel.mojo`'s fused
    arm.

    `api` is deliberately a wildcard, unlike this file's `mo.*` rows (window
    285): this op is minted by `ops.custom` at GRAPH BUILD, and the add-time
    kernel check queries device info before the compile holds the device --
    `("npu", "unknown", "unknown", "unknown")` -- so an `api="npu"` pin can
    never match there and the production export refuses to stage. Pinning
    `type` keeps cpu/gpu targets refused; the name is unique across the
    plugin packages, so the wildcard creates no resolution ambiguity (the
    qaic production ops register fully bare for the same reason).

    This registration is the PRODUCTION engagement path the #487 device window
    lacked (the fourth real-correct-unreachable-arm instance): the render
    runner's denoise stage executes the MEF, the MEF's staged form of the
    single-stream qk-norm+rope pair is this one composite, and this launcher
    builds the rope pack, self-configures from the skel's arm echo, and either
    fuses or keeps the sequential pair. Nothing here trusts a build flag at
    run time.

    Wire discipline: selector 5's C ABI is frozen at 4 buffers + 5 scalars, so
    the positions and freqs tables ride inside the one allocation the witness
    argument names, located by a byte offset passed in the
    `multiply_before_cast` scalar slot (ctl). Legacy ctl values 0/1 keep their
    meaning, which is what makes both mispairing directions inert: a fused
    skel under a host that never opted in runs the plain norm per dispatch,
    and a legacy skel under this launcher is detected by the probe and served
    the sequential pair.

    The emitter (`test_flux_denoise_mef.py`'s staging patch) only mints this
    op where the norm's output feeds the rope directly, with
    `multiply_before_cast=True`, `weight_offset=0` and `interleaved=True` --
    the narrowing FUSION_S2 signs. The dual-stream blocks' norm-concat-rope
    chains stay on the unfused ops.
    """

    @staticmethod
    def execute[
        dtype: DType,
        freq_dtype: DType,
        //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=3, ...],
        x: InputTensor[dtype=dtype, rank=3, ...],
        gamma: InputTensor[dtype=dtype, rank=1, ...],
        freqs_cis: InputTensor[dtype=freq_dtype, rank=2, ...],
        position_ids: InputTensor[dtype=DType.uint32, rank=2, ...],
        epsilon: Float32,
        ctx: DeviceContext,
    ) capturing raises:
        # A held matmul-chain member must run before this op touches any
        # operand: this is the 186A host-reader class, closed per site.
        _matmul_chain_flush()
        comptime assert dtype == DType.float16, (
            "fused_rms_norm_rope on the on-device NPU: fp16 only -- the"
            " fused selector-5 arm exists for the production f16 row"
        )
        comptime assert freq_dtype == DType.float32, (
            "fused_rms_norm_rope on the on-device NPU: the rope tables"
            " are read as fp32"
        )
        _refuse_batched_graph_path(getenv("MODULAR_NPU_BATCH") == "1")
        _refuse_megastep_graph_path(getenv("MODULAR_NPU_MEGASTEP") == "1")

        if output.shape() != x.shape():
            raise Error(
                "fused_rms_norm_rope on the on-device NPU: input and"
                " output shapes differ"
            )
        _require_dense(
            x.shape(), x.strides(), "fused_rms_norm_rope", "the input"
        )
        _require_dense(
            output.shape(),
            output.strides(),
            "fused_rms_norm_rope",
            "the output",
        )
        _require_dense(
            freqs_cis.shape(),
            freqs_cis.strides(),
            "fused_rms_norm_rope",
            "freqs_cis",
        )

        var tokens = Int(x.dim_size(0))
        var heads = Int(x.dim_size(1))
        var length = Int(x.dim_size(2))
        var rows = tokens * heads
        if Int(gamma.dim_size(0)) != length:
            raise Error(
                "fused_rms_norm_rope on the on-device NPU: gamma must"
                " carry one weight per head element"
            )
        if Int(freqs_cis.dim_size(1)) != length or length % 2 != 0:
            raise Error(
                (
                    "fused_rms_norm_rope on the on-device NPU: the fused"
                    " kernel rotates the whole even-sized head vector, got"
                    " head_size "
                ),
                length,
                " against freqs_cis row length ",
                Int(freqs_cis.dim_size(1)),
            )
        if length > RMS_ROPE_MAX_HEAD:
            raise Error(
                "fused_rms_norm_rope on the on-device NPU: head_size ",
                length,
                " is past the fused kernel's staging cap ",
                RMS_ROPE_MAX_HEAD,
            )
        var freq_rows = Int(freqs_cis.dim_size(0))
        if Int(position_ids.dim_size(1)) < tokens:
            raise Error(
                (
                    "fused_rms_norm_rope on the on-device NPU:"
                    " position_ids carries "
                ),
                Int(position_ids.dim_size(1)),
                " positions for ",
                tokens,
                " tokens",
            )
        # Indexed rather than pointer-linear, as the unfused rope registration
        # reads it -- a broadcast row's zero stride is then honoured.
        for t in range(tokens):
            if Int(position_ids[0, t]) >= freq_rows:
                raise Error(
                    "fused_rms_norm_rope on the on-device NPU: position_ids[",
                    t,
                    "] is ",
                    Int(position_ids[0, t]),
                    ", past freqs_cis' ",
                    freq_rows,
                    " rows",
                )

        var f = ctx.load_function[npu_selector_sig](
            function_name=String(_rms_norm_selector[dtype]()), asm=String("")
        )
        var state = _rope_fused_state()

        # The skel is asked once per process which arm it computes; every
        # fused/sequential decision below flows from that echo, never from a
        # build knob (the 229 class).
        if state[unsafe_offset=_ROPE_ST_ARM] == _ROPE_ARM_UNPROBED:
            state[unsafe_offset=_ROPE_ST_ARM] = _rope_fused_probe(
                ctx, length, epsilon
            )

        # The pack is rebuilt in place when the source tables move or their
        # edge fingerprint drifts; its geometry is pinned by the first
        # dispatch because the allocation is process-lifetime.
        var freq_words = freq_rows * length
        var layout = _rope_pack_layout(rows, freq_words)
        if state[unsafe_offset=_ROPE_ST_PACK] == 0:
            var pack_buf = ctx.enqueue_create_buffer[DType.uint32](
                layout[3] // 4
            )
            state[unsafe_offset=_ROPE_ST_PACK] = UInt64(
                Int(pack_buf.unsafe_ptr())
            )
            state[unsafe_offset=_ROPE_ST_PACK_WORDS] = UInt64(layout[3] // 4)
            state[unsafe_offset=_ROPE_ST_ROWS] = UInt64(rows)
            state[unsafe_offset=_ROPE_ST_LENGTH] = UInt64(length)
            state[unsafe_offset=_ROPE_ST_HDR_OFF] = UInt64(layout[0])
            state[unsafe_offset=_ROPE_ST_POS_OFF] = UInt64(layout[1])
            state[unsafe_offset=_ROPE_ST_FREQS_OFF] = UInt64(layout[2])
            _ = pack_buf.take_ptr()
        elif state[unsafe_offset=_ROPE_ST_ROWS] != UInt64(rows) or state[
            unsafe_offset=_ROPE_ST_LENGTH
        ] != UInt64(length):
            raise Error(
                "fused_rms_norm_rope on the on-device NPU: a second"
                " geometry reached the process-lifetime rope pack -- one"
                " graph geometry per process is the supported shape"
            )
        var pack = Pointer[UInt32, MutUnsafeAnyOrigin](
            unsafe_from_address=Int(state[unsafe_offset=_ROPE_ST_PACK])
        )
        var freqs_words_ptr = freqs_cis.unsafe_ptr().unsafe_bitcast[UInt32]()
        var sample = _rope_pack_sample(
            position_ids, tokens, freqs_words_ptr, freq_words
        )
        if (
            state[unsafe_offset=_ROPE_ST_KEY_FREQS]
            != UInt64(Int(freqs_words_ptr))
            or state[unsafe_offset=_ROPE_ST_KEY_POS]
            != UInt64(Int(position_ids.unsafe_ptr()))
            or state[unsafe_offset=_ROPE_ST_SAMPLE] != sample
        ):
            _rope_pack_write_header(pack, layout, length, freq_rows)
            var pos = pack.unsafe_offset(layout[1] // 4)
            for t in range(tokens):
                var p = UInt32(position_ids[0, t])
                for h in range(heads):
                    pos[unsafe_offset=t * heads + h] = p
            var freqs_dst = pack.unsafe_offset(layout[2] // 4)
            for i in range(freq_words):
                freqs_dst[unsafe_offset=i] = freqs_words_ptr[unsafe_offset=i]
            state[unsafe_offset=_ROPE_ST_KEY_FREQS] = UInt64(
                Int(freqs_words_ptr)
            )
            state[unsafe_offset=_ROPE_ST_KEY_POS] = UInt64(
                Int(position_ids.unsafe_ptr())
            )
            state[unsafe_offset=_ROPE_ST_SAMPLE] = sample

        if state[unsafe_offset=_ROPE_ST_ARM] == _ROPE_ARM_FUSED:
            # Treatment delivery is per dispatch, not per process: the arm
            # word is re-seeded before and read after every launch, so a
            # fused dispatch that silently ran the plain norm cannot pass.
            pack[unsafe_offset=RMS_WITNESS_ARM_WORD] = _ROPE_PACK_SEED
            ctx.enqueue_function(
                f,
                output.unsafe_ptr(),
                x.unsafe_ptr(),
                gamma.unsafe_ptr(),
                pack,
                Int32(rows),
                Int32(length),
                epsilon,
                Float32(0),
                Int32(layout[0]),
                grid_dim=Dim(NPU_GRID_CORES),
                block_dim=Dim(NPU_THREAD_MASK),
            )
            var echoed = pack[unsafe_offset=RMS_WITNESS_ARM_WORD]
            if echoed != 1:
                raise Error(
                    (
                        "fused_rms_norm_rope on the on-device NPU: a"
                        " fused dispatch echoed arm "
                    ),
                    Int(echoed),
                    (
                        " where the probe promised 1 -- the fused path must"
                        " never half-run"
                    ),
                )
            return

        # Legacy skel: the sequential pair, relocated -- the plain norm lands
        # in a launcher-owned scratch row set and selector 9 reads it back,
        # the exact arithmetic the unfused graph performs. The rope tables
        # ride the pack as interior pointers, which the transport folds.
        if state[unsafe_offset=_ROPE_ST_SCRATCH] == 0:
            var scratch_buf = ctx.enqueue_create_buffer[DType.float16](
                rows * length
            )
            state[unsafe_offset=_ROPE_ST_SCRATCH] = UInt64(
                Int(scratch_buf.unsafe_ptr())
            )
            _ = scratch_buf.take_ptr()
        var scratch = Pointer[Float16, MutUnsafeAnyOrigin](
            unsafe_from_address=Int(state[unsafe_offset=_ROPE_ST_SCRATCH])
        )
        ctx.enqueue_function(
            f,
            scratch,
            x.unsafe_ptr(),
            gamma.unsafe_ptr(),
            _witness_slot(ctx),
            Int32(rows),
            Int32(length),
            epsilon,
            Float32(0),
            Int32(1),
            grid_dim=Dim(NPU_GRID_CORES),
            block_dim=Dim(NPU_THREAD_MASK),
        )
        var rope_f = ctx.load_function[npu_selector_sig](
            function_name=String(_rope_selector[dtype, True]()),
            asm=String(""),
        )
        ctx.enqueue_function(
            rope_f,
            output.unsafe_ptr(),
            scratch,
            (
                pack.unsafe_offset(
                    Int(state[unsafe_offset=_ROPE_ST_FREQS_OFF]) // 4
                )
            ).unsafe_bitcast[Float32](),
            pack.unsafe_offset(Int(state[unsafe_offset=_ROPE_ST_POS_OFF]) // 4),
            _witness_slot(ctx),
            Int32(rows),
            Int32(length),
            grid_dim=Dim(NPU_GRID_CORES),
            block_dim=Dim(NPU_THREAD_MASK),
        )


@always_inline
def _gate_residual_selector[dtype: DType]() -> StaticString:
    """Names the signed `gate_residual` row compiled for `dtype`.

    The same two-row arrangement as `_layernorm_selector`, for the same reason:
    the kernel set is fixed when the library is signed, so a dtype reaches a row
    built for it or none. Both rows take the same buffers and scalars, so the
    name is all that differs between the two dispatches.

    `comptime if` rather than a conditional expression so that one name reaches
    the object, as `_layernorm_selector` explains.
    """
    comptime if dtype == DType.float16:
        return "gate_residual"
    else:
        return "gate_residual_bf16"


@extensibility.register("mo.composite.gated_residual", type="npu", api="npu")
struct GatedResidual:
    """Dispatches a `gate_residual` selector for `out = x + gate * y`.

    The one registration here that claims a composite the graph compiler mints
    rather than an operator the model author wrote. `GatedResidualPattern`
    (`GraphCompiler/lib/MODialect/Patterns/Fusion.cpp`) collapses
    `mo.add(x, mo.mul(mo.static.broadcast_to(gate), y))` into it, because the
    unfused pair reaches no kernel on this target: the graph compiler would
    otherwise fold both ops into the producing matmul's epilogue, and a signed
    kernel cannot take one.

    Not fp16-only: `NPUKernels.def` carries a bf16 twin, which is the dtype
    FLUX.2 runs in.
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        gate_rank: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        x: InputTensor[dtype=dtype, rank=rank, ...],
        y: InputTensor[dtype=dtype, rank=rank, ...],
        gate: InputTensor[dtype=dtype, rank=gate_rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        comptime assert (
            dtype == DType.float16 or dtype == DType.bfloat16
        ), "mo.composite.gated_residual on the on-device NPU: fp16 or bf16 only"

        if output.shape() != x.shape() or output.shape() != y.shape():
            raise Error(
                "mo.composite.gated_residual on the on-device NPU: x, y and the"
                " output must share one shape"
            )
        _require_dense(
            output.shape(),
            output.strides(),
            "mo.composite.gated_residual",
            "the output",
        )
        _require_dense(
            x.shape(), x.strides(), "mo.composite.gated_residual", "x"
        )
        _require_dense(
            y.shape(), y.strides(), "mo.composite.gated_residual", "y"
        )
        _require_dense(
            gate.shape(),
            gate.strides(),
            "mo.composite.gated_residual",
            "the gate",
        )

        var row_length = Int(output.dim_size(rank - 1))
        var num_rows = 1
        for i in range(rank - 1):
            num_rows *= Int(output.dim_size(i))
        var outer = Int(output.dim_size(0)) if rank > 1 else 1

        var gate_rows = _per_channel_rows(
            gate, row_length, outer, "mo.composite.gated_residual", "the gate"
        )

        var f = ctx.load_function[npu_selector_sig](
            function_name=String(_gate_residual_selector[dtype]()),
            asm=String(""),
        )
        if gate_rows == 1:
            ctx.enqueue_function(
                f,
                output.unsafe_ptr(),
                x.unsafe_ptr(),
                y.unsafe_ptr(),
                gate.unsafe_ptr(),
                _witness_slot(ctx),
                Int32(num_rows),
                Int32(row_length),
                Int32(0),
                grid_dim=Dim(NPU_GRID_CORES),
                block_dim=Dim(NPU_THREAD_MASK),
            )
        else:
            # One gate row per outermost index (the symbolic batch a diffusion
            # graph stages). The signed kernel steps its parameter linearly per
            # row, which cannot express per-batch rows in one dispatch, so this
            # walks the outermost axis as `BatchMatmul` walks its slices -- at
            # the batch of 1 the loop is the single dispatch above. A witness
            # SLOT per slice, as `BatchMatmul` takes one: nothing reads this
            # op's witness today, but a slot is per launch by construction and
            # a shared one would alias the slices' reports.
            var rows_per_outer = num_rows // outer
            var slice_elems = rows_per_outer * row_length
            for b in range(outer):
                ctx.enqueue_function(
                    f,
                    output.unsafe_ptr().unsafe_offset(b * slice_elems),
                    x.unsafe_ptr().unsafe_offset(b * slice_elems),
                    y.unsafe_ptr().unsafe_offset(b * slice_elems),
                    gate.unsafe_ptr().unsafe_offset(b * row_length),
                    _witness_slot(ctx),
                    Int32(rows_per_outer),
                    Int32(row_length),
                    Int32(0),
                    grid_dim=Dim(NPU_GRID_CORES),
                    block_dim=Dim(NPU_THREAD_MASK),
                )


@extensibility.register(
    "mo.composite.gated_residual_clamp", type="npu", api="npu"
)
struct GatedResidualClamp:
    """Dispatches the signed `gate_residual_clamp` selector for
    `out = clamp(x + gate * y, -65504, 65504)`.

    `GatedResidualClampPattern` mints the composite from the fp16-saturated
    sub-block closes -- `min(max(x + gate * y, -65504), 65504)`, the
    `saturate_to_float16` guard of `flux2.py` -- which #166's clamp-absence
    guard kept refusing whole so no site would be half-claimed. The kernel
    owns the bounds, so the wire shape and scalars are exactly
    `gate_residual`'s and only the selector name differs.

    fp16 only, unlike its unclamped sibling: bf16 carries fp32's exponent
    range, so the bound cannot bind, the reference never stages the guard at
    bf16, and `NPUKernels.def` carries no bf16 row -- one literal selector
    name rather than a dtype helper.
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        gate_rank: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        x: InputTensor[dtype=dtype, rank=rank, ...],
        y: InputTensor[dtype=dtype, rank=rank, ...],
        gate: InputTensor[dtype=dtype, rank=gate_rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        comptime assert (
            dtype == DType.float16
        ), "mo.composite.gated_residual_clamp on the on-device NPU: fp16 only"

        if output.shape() != x.shape() or output.shape() != y.shape():
            raise Error(
                "mo.composite.gated_residual_clamp on the on-device NPU: x, y"
                " and the output must share one shape"
            )
        _require_dense(
            output.shape(),
            output.strides(),
            "mo.composite.gated_residual_clamp",
            "the output",
        )
        _require_dense(
            x.shape(), x.strides(), "mo.composite.gated_residual_clamp", "x"
        )
        _require_dense(
            y.shape(), y.strides(), "mo.composite.gated_residual_clamp", "y"
        )
        _require_dense(
            gate.shape(),
            gate.strides(),
            "mo.composite.gated_residual_clamp",
            "the gate",
        )

        var row_length = Int(output.dim_size(rank - 1))
        var num_rows = 1
        for i in range(rank - 1):
            num_rows *= Int(output.dim_size(i))
        var outer = Int(output.dim_size(0)) if rank > 1 else 1

        var gate_rows = _per_channel_rows(
            gate,
            row_length,
            outer,
            "mo.composite.gated_residual_clamp",
            "the gate",
        )

        var f = ctx.load_function[npu_selector_sig](
            function_name=String("gate_residual_clamp"), asm=String("")
        )
        if gate_rows == 1:
            ctx.enqueue_function(
                f,
                output.unsafe_ptr(),
                x.unsafe_ptr(),
                y.unsafe_ptr(),
                gate.unsafe_ptr(),
                _witness_slot(ctx),
                Int32(num_rows),
                Int32(row_length),
                Int32(0),
                grid_dim=Dim(NPU_GRID_CORES),
                block_dim=Dim(NPU_THREAD_MASK),
            )
        else:
            # One gate row per outermost index, walked exactly as
            # `GatedResidual` walks it -- see there for why one dispatch
            # cannot express per-batch parameter rows.
            var rows_per_outer = num_rows // outer
            var slice_elems = rows_per_outer * row_length
            for b in range(outer):
                ctx.enqueue_function(
                    f,
                    output.unsafe_ptr().unsafe_offset(b * slice_elems),
                    x.unsafe_ptr().unsafe_offset(b * slice_elems),
                    y.unsafe_ptr().unsafe_offset(b * slice_elems),
                    gate.unsafe_ptr().unsafe_offset(b * row_length),
                    _witness_slot(ctx),
                    Int32(rows_per_outer),
                    Int32(row_length),
                    Int32(0),
                    grid_dim=Dim(NPU_GRID_CORES),
                    block_dim=Dim(NPU_THREAD_MASK),
                )


@always_inline
def _ada_modulate_selector[dtype: DType]() -> StaticString:
    """Names the signed `ada_modulate` row compiled for `dtype`.

    `comptime if` rather than a conditional expression so that one name reaches
    the object, as `_layernorm_selector` explains.
    """
    comptime if dtype == DType.float16:
        return "ada_modulate"
    else:
        return "ada_modulate_bf16"


@extensibility.register("mo.composite.ada_modulate", type="npu", api="npu")
struct AdaModulate:
    """Dispatches an `ada_modulate` selector for `(1 + scale) * x + shift`.

    A composite the graph compiler mints rather than an operator the model
    author wrote, as `GatedResidual` is: `AdaModulatePattern` collapses the
    scale's `1 +`, the multiply and the shift's add into it.

    **The one is the kernel's, not the scale's.** The selector computes
    `(1 + scale) * x + shift`, so `scale` here is the raw modulation parameter;
    the pattern is what guarantees the graph asked for that one, by consuming
    an `mo.add` against a constant one and refusing to fire without it.

    Not fp16-only: `NPUKernels.def` carries a bf16 twin, which is the dtype
    FLUX.2 runs in.
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        scale_rank: Int,
        shift_rank: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        x: InputTensor[dtype=dtype, rank=rank, ...],
        scale: InputTensor[dtype=dtype, rank=scale_rank, ...],
        shift: InputTensor[dtype=dtype, rank=shift_rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        comptime assert (
            dtype == DType.float16 or dtype == DType.bfloat16
        ), "mo.composite.ada_modulate on the on-device NPU: fp16 or bf16 only"

        if output.shape() != x.shape():
            raise Error(
                "mo.composite.ada_modulate on the on-device NPU: x and the"
                " output must share one shape"
            )
        _require_dense(
            output.shape(),
            output.strides(),
            "mo.composite.ada_modulate",
            "the output",
        )
        _require_dense(x.shape(), x.strides(), "mo.composite.ada_modulate", "x")
        _require_dense(
            scale.shape(),
            scale.strides(),
            "mo.composite.ada_modulate",
            "the scale",
        )
        _require_dense(
            shift.shape(),
            shift.strides(),
            "mo.composite.ada_modulate",
            "the shift",
        )

        var row_length = Int(output.dim_size(rank - 1))
        var num_rows = 1
        for i in range(rank - 1):
            num_rows *= Int(output.dim_size(i))
        var outer = Int(output.dim_size(0)) if rank > 1 else 1

        var scale_rows = _per_channel_rows(
            scale, row_length, outer, "mo.composite.ada_modulate", "the scale"
        )
        var shift_rows = _per_channel_rows(
            shift, row_length, outer, "mo.composite.ada_modulate", "the shift"
        )

        var f = ctx.load_function[npu_selector_sig](
            function_name=String(_ada_modulate_selector[dtype]()),
            asm=String(""),
        )
        if scale_rows == 1 and shift_rows == 1:
            ctx.enqueue_function(
                f,
                output.unsafe_ptr(),
                x.unsafe_ptr(),
                scale.unsafe_ptr(),
                shift.unsafe_ptr(),
                _witness_slot(ctx),
                Int32(num_rows),
                Int32(row_length),
                Int32(0),
                grid_dim=Dim(NPU_GRID_CORES),
                block_dim=Dim(NPU_THREAD_MASK),
            )
        else:
            # One parameter row per outermost index (the symbolic batch a
            # diffusion graph stages); either parameter may also still be a
            # single shared row, so each steps by its own count. The signed
            # kernel steps its parameters linearly per row, which cannot
            # express per-batch rows in one dispatch, so this walks the
            # outermost axis as `BatchMatmul` walks its slices -- at the batch
            # of 1 the loop is the single dispatch above. A witness SLOT per
            # slice, as `BatchMatmul` takes one.
            var rows_per_outer = num_rows // outer
            var slice_elems = rows_per_outer * row_length
            var scale_step = row_length if scale_rows != 1 else 0
            var shift_step = row_length if shift_rows != 1 else 0
            for b in range(outer):
                ctx.enqueue_function(
                    f,
                    output.unsafe_ptr().unsafe_offset(b * slice_elems),
                    x.unsafe_ptr().unsafe_offset(b * slice_elems),
                    scale.unsafe_ptr().unsafe_offset(b * scale_step),
                    shift.unsafe_ptr().unsafe_offset(b * shift_step),
                    _witness_slot(ctx),
                    Int32(rows_per_outer),
                    Int32(row_length),
                    Int32(0),
                    grid_dim=Dim(NPU_GRID_CORES),
                    block_dim=Dim(NPU_THREAD_MASK),
                )


@always_inline
def _swiglu_selector[dtype: DType]() -> StaticString:
    """Names the signed `swiglu` row compiled for `dtype`.

    `comptime if` rather than a conditional expression so that one name reaches
    the object, as `_layernorm_selector` explains.
    """
    comptime if dtype == DType.float16:
        return "swiglu"
    else:
        return "swiglu_bf16"


@extensibility.register("mo.composite.swiglu", type="npu", api="npu")
@extensibility.accepts_pitched_rows
struct SwiGLU:
    """Dispatches a `swiglu` selector for `silu(gate) * up`.

    The input is the **packed** projection, `[rows, 2 * inner_dim]`, because
    that is what the selector reads: it takes the first `inner_dim` of a row as
    the gate and the next as the up projection, which is the layout
    `chunk(x, 2, axis=-1)` produces. `SwiGLUPattern` is what turns the two
    halves back into the packed row the graph had before it split them.

    The packed rows need not be dense: at twenty of FLUX.2's thirty sites the
    pair is a column slice of the single-stream block's wider fused
    projection, and `@extensibility.accepts_pitched_rows` is what lets that
    view cross as-is — its start in the pointer, its row pitch as the
    selector's third scalar — instead of materializing through the host
    strided walk first. The decorator is the compiler half of the contract;
    `_require_pitched_rows` below is the dispatch half that refuses any
    layout wider than one row pitch.

    Not fp16-only: `NPUKernels.def` carries a bf16 twin, which is the dtype
    FLUX.2 runs in.
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        comptime assert (
            dtype == DType.float16 or dtype == DType.bfloat16
        ), "mo.composite.swiglu on the on-device NPU: fp16 or bf16 only"

        _require_dense(
            output.shape(),
            output.strides(),
            "mo.composite.swiglu",
            "the output",
        )
        var row_pitch = _require_pitched_rows(
            input.shape(), input.strides(), "mo.composite.swiglu", "the input"
        )

        var inner_dim = Int(output.dim_size(rank - 1))
        var num_rows = 1
        for i in range(rank - 1):
            num_rows *= Int(output.dim_size(i))

        # The verifier already holds the input to twice the output's trailing
        # extent and the same shape elsewhere. Reading it again is what keeps a
        # row the selector would walk off the end of from reaching it:
        # `inner_dim` is where it finds each row's up half.
        if Int(input.dim_size(rank - 1)) != 2 * inner_dim:
            raise Error(
                (
                    "mo.composite.swiglu on the on-device NPU: the input's"
                    " trailing extent must be twice the output's; got "
                ),
                Int(input.dim_size(rank - 1)),
                " for an inner dimension of ",
                inner_dim,
            )
        for i in range(rank - 1):
            if input.dim_size(i) != output.dim_size(i):
                raise Error(
                    (
                        "mo.composite.swiglu on the on-device NPU: the input"
                        " and the output must agree off the trailing axis;"
                        " axis "
                    ),
                    i,
                    " is ",
                    Int(input.dim_size(i)),
                    " against ",
                    Int(output.dim_size(i)),
                )

        var f = ctx.load_function[npu_selector_sig](
            function_name=String(_swiglu_selector[dtype]()), asm=String("")
        )
        ctx.enqueue_function(
            f,
            output.unsafe_ptr(),
            input.unsafe_ptr(),
            _witness_slot(ctx),
            Int32(num_rows),
            Int32(inner_dim),
            Int32(row_pitch),
            grid_dim=Dim(NPU_GRID_CORES),
            block_dim=Dim(NPU_THREAD_MASK),
        )


@always_inline
def _silu_selector[dtype: DType]() -> StaticString:
    """Names the signed `silu` row compiled for `dtype`.

    `comptime if` rather than a conditional expression so that one name reaches
    the object, as `_layernorm_selector` explains.
    """
    comptime if dtype == DType.float16:
        return "silu"
    else:
        return "silu_bf16"


@extensibility.register("mo.silu", type="npu", api="npu")
struct Silu:
    """Dispatches a `silu` selector, one launch per activation.

    Unlike the composites above this claims an operator the model author wrote.
    The compiler otherwise registers `mo.silu` as an elementwise SIMD lambda for
    its fusion machinery rather than an `execute` kernel, and on this target
    that lambda is device code that does not compile -- so a bare `silu` is a
    hard compile refusal, not a slow path. Taking it over breaks each one out
    into a dispatch of its own, which is the trade: five dispatches a step
    against a graph that does not compile at all.

    The gated form is not this. `silu(gate) * up` reaches `mo.composite.swiglu`
    in one launch, and `SwiGLUPattern` claims those before this is consulted.
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        comptime assert (
            dtype == DType.float16 or dtype == DType.bfloat16
        ), "mo.silu on the on-device NPU: fp16 or bf16 only"

        if output.shape() != input.shape():
            raise Error(
                "mo.silu on the on-device NPU: input and output shapes differ"
            )
        _require_dense(
            output.shape(), output.strides(), "mo.silu", "the output"
        )
        _require_dense(input.shape(), input.strides(), "mo.silu", "the input")

        var row_length = Int(output.dim_size(rank - 1))
        var num_rows = 1
        for i in range(rank - 1):
            num_rows *= Int(output.dim_size(i))

        var f = ctx.load_function[npu_selector_sig](
            function_name=String(_silu_selector[dtype]()), asm=String("")
        )
        ctx.enqueue_function(
            f,
            output.unsafe_ptr(),
            input.unsafe_ptr(),
            _witness_slot(ctx),
            Int32(num_rows),
            Int32(row_length),
            grid_dim=Dim(NPU_GRID_CORES),
            block_dim=Dim(NPU_THREAD_MASK),
        )


comptime MHA_RT_ENV = "MODULAR_NPU_MHA_RT"
"""The one variable that selects the `mha_no_cache` arm, for both its callers.

`npu_attention_dispatch.cpp` reads this name for the standalone driver; this
module reads it for the graph, so a device leg is one binary and one variable
rather than two skels."""

comptime MHA_RT_ARM_MAX = 3
"""The highest arm `mojo_mha_no_cache_f16` defines (`MHA_RT_TEAM`).

Spelled here rather than imported: `npu_kernel.mojo` is the DEVICE compile's
entry and no package member imports it (`build.sh`'s `DEVICE_PKG_DENY`), so
reaching into it from this host layer would pull the whole kernel into every
graph compile to read one integer. The C++ driver spells its own copy for the
same reason. The arms are contiguous from 0, so one bound checks them all."""


comptime MHA_RT_DEFAULT = 3
"""The arm an UNSET `MODULAR_NPU_MHA_RT` selects: `MHA_RT_TEAM`, the production
arm since order 123.

Spelled apart from `MHA_RT_ARM_MAX` deliberately, though both are 3 today. One
is "the highest arm the kernel defines" and the other is "the arm we ship"; a
fifth arm added tomorrow must move the bound without moving the default."""


def _mha_rt_selector(spec: StringSlice) raises -> Int32:
    """Reads the `rt` arm selector the `mha_no_cache` launch carries.

    `rt` picks which arm computes: `MHA_RT_HVX` the shipped vector path,
    `MHA_RT_DARK` a QK^T tile on the matrix array beside it, `MHA_RT_FLASH`
    head 0 on the array, `MHA_RT_TEAM` every head on the array.

    AN UNSET VARIABLE IS `MHA_RT_TEAM` since the default wave. Order 123
    claimed attention in production on that arm -- renders 301 and 308 s, Tier
    B 52.986 dB, the pair bitwise identical -- and it has been named on the
    command line of every render since, which made the production arm a thing
    an operator had to remember. `MODULAR_NPU_MHA_RT=0` still selects the
    vector path, so the control arm is a name rather than a source edit.

    A skel built `SKYE_HMX_ATTN=0` receiving this runs the vector arm anyway,
    correct and slow, because the array arm is not in the object -- order 140's
    withheld control proved that by digest. `build.sh` now defaults `HMX_ATTN`
    on for the same reason this defaults to `MHA_RT_TEAM`, so the two agree
    unless an operator names one of them.

    ORDER 121 IS WHY THIS IS READ HERE. Its gate proved the array arm correct
    over all 24 heads and 3.9x faster on the standalone driver, and its render
    still took the vector path -- because this dispatch passed a literal 0 and
    `MODULAR_NPU_MHA_RT` reached only the driver. That is the third instance in
    this arc of the print-vs-dispatch class, where a selector is reported or
    reserved somewhere the launch does not read: #349 dropped the driver's own
    read while keeping its `rt=` print, #350 restored it, and the graph path was
    left behind both times.

    A value outside the arms is refused rather than clamped. The driver's
    `strtol` yields 0 for an unparsable spelling, which would make a typo render
    the vector path while the operator believed the array ran -- the same class
    this plumbing exists to close.

    Args:
        spec: The variable's value, empty when it is unset.

    Returns:
        The `rt` scalar to launch with.

    Raises:
        If `spec` is neither empty nor an arm the kernel defines.
    """
    if not spec:
        return Int32(MHA_RT_DEFAULT)

    # An unparsable spelling lands on the same refusal as an out-of-range one,
    # which is the point: both are an operator asking for an arm that does not
    # exist, and neither may quietly become the vector path.
    var rt: Int
    try:
        rt = atol(spec)
    except:
        rt = -1
    if rt < 0 or rt > MHA_RT_ARM_MAX:
        raise Error(
            "mo.mha.no_cache on the on-device NPU: ",
            MHA_RT_ENV,
            " must name an arm in [0, ",
            MHA_RT_ARM_MAX,
            "]; got ",
            spec,
        )
    return Int32(rt)


comptime MHA_MASKED_RT_ENV = "MODULAR_NPU_MHA_MASKED_RT"
"""The variable that selects the `mha_masked` arm.

Its OWN variable, not `MHA_RT_ENV`. The two entries define different arm sets --
the masked one has no `MHA_RT_DARK` or `MHA_RT_FLASH`, because those are slice
2/3 probes of the array and the array is probed by the entry that already has
them -- so one shared variable would make a dense diagnostic (`=1`) refuse the
encoder's dispatch, or, worse, quietly render its vector arm. Separate names let
the render's arm and the encoder's be chosen independently, which is what a leg
that moves one of them at a time needs."""

comptime MHA_MASKED_RT_TEAM = 3
"""The one non-vector arm `mojo_mha_masked_f16` defines (`MHA_RT_TEAM`).

Spelled here for `MHA_RT_ARM_MAX`'s reason: `npu_kernel.mojo` is the device
compile's entry and no package member may import it. `test_mha_rt_selector.mojo`
is where this copy and the kernel's meet."""


def _mha_masked_rt_selector(spec: StringSlice) raises -> Int32:
    """Reads the `rt` arm selector the `mha_masked` launch carries.

    `MHA_RT_HVX` (0) is the vector path. `MHA_RT_TEAM` (3) puts every head on
    the matrix array with the additive bias applied at the softmax seam, and is
    what an UNSET variable selects since the default wave: order 140 collected
    that arm in the render -- pair bitwise identical, Tier B 52.778 dB, the
    encoder stage 31.4 -> 15.5 s and the same 28 masked dispatches 606.8 ->
    75.0 ms each, zero failures. `MODULAR_NPU_MHA_MASKED_RT=0` still selects
    the vector path.

    THE ARMS BETWEEN THEM ARE REFUSED, NOT PASSED THROUGH. `MHA_RT_DARK` and
    `MHA_RT_FLASH` are arms of the DENSE entry; the masked entry has no branch
    for them, so a launch carrying one would run the vector arm and report a
    selector the operator believes engaged the array. That is the same class as
    the literal this function replaces, one level down -- so 1 and 2 raise here
    rather than reaching a kernel with nothing to do with them.

    Args:
        spec: The variable's value, empty when it is unset.

    Returns:
        The `rt` scalar to launch with.

    Raises:
        If `spec` is neither empty nor an arm the masked kernel defines.
    """
    if not spec:
        return Int32(MHA_MASKED_RT_TEAM)

    var rt: Int
    try:
        rt = atol(spec)
    except:
        rt = -1
    if rt != 0 and rt != MHA_MASKED_RT_TEAM:
        raise Error(
            "mo.composite.masked_flash_attention_gpu on the on-device NPU: ",
            MHA_MASKED_RT_ENV,
            " must name an arm the masked kernel defines (0 or ",
            MHA_MASKED_RT_TEAM,
            "); got ",
            spec,
        )
    return Int32(rt)


@extensibility.register("mo.mha.no_cache", type="npu", api="npu")
struct FlashAttention:
    """Dispatches the `mha_no_cache` selector, dense and non-causal."""

    @staticmethod
    def execute[
        rank: Int,
        //,
        target: StaticString,
        mask_str: StaticString,
        local_window_size: Int = -1,
    ](
        output: OutputTensor[rank=rank, ...],
        q: InputTensor[dtype=output.dtype, rank=rank, ...],
        k: InputTensor[dtype=output.dtype, rank=rank, ...],
        v: InputTensor[dtype=output.dtype, rank=rank, ...],
        scale: Float32,
        ctx: DeviceContext,
    ) raises:
        # A held matmul-chain member must run before this op touches any
        # operand: this is the 186A host-reader class, closed per site.
        _matmul_chain_flush()
        comptime assert (
            rank == 4
        ), "mo.mha.no_cache on the on-device NPU: BSHD inputs only"
        comptime assert (
            output.dtype == DType.float16
        ), "mo.mha.no_cache on the on-device NPU: fp16 only"
        # `"null"` is `MaskName.NULL`, spelled literally to avoid depending on
        # the attention package for one string.
        comptime assert mask_str == "null", (
            "mo.mha.no_cache on the on-device NPU: the `mha_no_cache` selector"
            " applies no mask; a masked graph needs the `mha_masked` selector,"
            " which is not reached yet"
        )
        comptime assert (
            local_window_size == -1
        ), "mo.mha.no_cache on the on-device NPU: no sliding window"

        # The selector reads Q/K/V/O as `[seq_len, num_heads, 128]`, which is
        # BSHD's own layout once the batch is one and the head dim is the 128
        # the kernel is written for. A batch above 1 needs a copy the stub
        # does not make, so it is refused rather than reinterpreted; a
        # single head of any other dim takes the decomposed path below.
        var batch = Int(q.dim_size(0))
        var seq_len = Int(q.dim_size(1))
        var num_heads = Int(q.dim_size(2))
        var head_dim = Int(q.dim_size(3))
        if batch != 1:
            raise Error(
                "mo.mha.no_cache on the on-device NPU: batch must be 1, got ",
                batch,
            )
        if head_dim != 128 and num_heads != 1:
            # A per-head slice of BSHD is strided and nothing stages one
            # today, so the decomposition below serves exactly one head.
            raise Error(
                (
                    "mo.mha.no_cache on the on-device NPU: a head dim off the"
                    " flash kernel's 128 is decomposed onto the signed"
                    " matmul/softmax/matmul selectors for one head only; got "
                ),
                num_heads,
                " heads of ",
                head_dim,
            )

        # One `seq_len` and one `num_heads` reach the selector for all of Q, K
        # and V, so it strides them alike, while the op is wider: it admits K/V
        # as `BShD` for grouped attention and reads its key count from K rather
        # than Q. The unequal-length case is not hypothetical, and the mask
        # check above does not exclude it -- Wan's cross attention builds this
        # op over text K/V and latent Q under the same `NULL_MASK`. Either shape
        # would be read through Q's extents, against the wrong elements.
        if k.shape() != q.shape() or v.shape() != q.shape():
            raise Error(
                "mo.mha.no_cache on the on-device NPU: the `mha_no_cache`"
                " selector strides Q, K and V alike, so all three must have"
                " one shape; grouped or cross attention needs a stub that"
                " passes their extents separately"
            )

        if head_dim != 128:
            # The silicon-certified hand driver's decomposition
            # (`npu_vae_decoder_dispatch.cpp`, `walkAttention`) moved to the
            # seam the graph crosses: `matmul` (QK^T) -> `softmax` -> `matmul`
            # (PV), three already-signed selectors, with the scale folded
            # host-side into a staged Q. It computes the same attention at
            # the real head dim: one head's QK logit is a SINGLE dot product
            # over all `head_dim` lanes, so splitting a 512-wide head into
            # four 128-wide heads would softmax four partial dots separately
            # -- a different function, not a reshaping. The flash kernel's
            # own header defers a non-128 instantiation for the same reason
            # this arm exists: new kernel bytes force a catalogue re-sign,
            # and its per-head thread split would idle three of four threads
            # on one head, where these matmuls stripe all four plus the
            # array.
            _require_dense(q.shape(), q.strides(), "mo.mha.no_cache", "Q")
            _require_dense(k.shape(), k.strides(), "mo.mha.no_cache", "K")
            _require_dense(v.shape(), v.strides(), "mo.mha.no_cache", "V")
            _require_dense(
                output.shape(),
                output.strides(),
                "mo.mha.no_cache",
                "the output",
            )

            var d_pad = ((head_dim + 63) // 64) * 64
            var s_pad = ((seq_len + 63) // 64) * 64

            # Q is staged regardless (the scale fold), so its pad is free.
            var q_scaled = ctx.enqueue_create_buffer[output.dtype](
                seq_len * d_pad
            )
            _stage_scaled_rows(
                q_scaled.unsafe_ptr(),
                q.unsafe_ptr(),
                seq_len,
                head_dim,
                d_pad,
                scale,
            )

            var scores = ctx.enqueue_create_buffer[output.dtype](
                seq_len * seq_len
            )
            var f_matmul = ctx.load_function[npu_selector_sig](
                function_name=String("matmul"), asm=String("")
            )
            var qk_witness = _witness_slot(ctx)
            _arm_matmul_ran(qk_witness, "mo.mha.no_cache")
            _arm_unpack_arm_known(qk_witness, "mo.mha.no_cache")
            # Dense `[1, S, 1, D]` is `[S, D]` row-major: token-major K IS
            # the `[N=S, K=D]` b-transposed weight, used verbatim with no
            # transpose -- the driver's own trick. Verbatim only holds where
            # the skel reads row-major; an array-native serve owes the
            # selector croutons, so K is then staged like a ragged head dim.
            if head_dim == d_pad and not _stage_native_weight(
                seq_len, seq_len, d_pad
            ):
                ctx.enqueue_function(
                    f_matmul,
                    scores.unsafe_ptr(),
                    q_scaled.unsafe_ptr(),
                    k.unsafe_ptr(),
                    qk_witness,
                    Int32(seq_len),
                    Int32(seq_len),
                    Int32(d_pad),
                    grid_dim=Dim(NPU_GRID_CORES),
                    block_dim=Dim(NPU_THREAD_MASK),
                )
            else:
                var k_padded = ctx.enqueue_create_buffer[output.dtype](
                    seq_len * d_pad
                )
                _stage_weight_rows(
                    k_padded.unsafe_ptr(),
                    k.unsafe_ptr(),
                    seq_len,
                    head_dim,
                    d_pad,
                    seq_len,
                )
                ctx.enqueue_function(
                    f_matmul,
                    scores.unsafe_ptr(),
                    q_scaled.unsafe_ptr(),
                    k_padded.unsafe_ptr(),
                    qk_witness,
                    Int32(seq_len),
                    Int32(seq_len),
                    Int32(d_pad),
                    grid_dim=Dim(NPU_GRID_CORES),
                    block_dim=Dim(NPU_THREAD_MASK),
                )
                _ = k_padded^

            # One dispatch over all `seq_len` rows of length `seq_len`; the
            # selector accumulates in fp32, designed for rows this long.
            var probs = ctx.enqueue_create_buffer[output.dtype](
                seq_len * seq_len
            )
            var f_softmax = ctx.load_function[npu_selector_sig](
                function_name=String("softmax"), asm=String("")
            )
            ctx.enqueue_function(
                f_softmax,
                probs.unsafe_ptr(),
                scores.unsafe_ptr(),
                _witness_slot(ctx),
                Int32(seq_len),
                Int32(seq_len),
                grid_dim=Dim(NPU_GRID_CORES),
                block_dim=Dim(NPU_THREAD_MASK),
            )

            var v_t = ctx.enqueue_create_buffer[output.dtype](head_dim * s_pad)
            _stage_transposed_weight(
                v_t.unsafe_ptr(),
                v.unsafe_ptr(),
                seq_len,
                head_dim,
                s_pad,
                seq_len,
            )

            var pv_witness = _witness_slot(ctx)
            _arm_matmul_ran(pv_witness, "mo.mha.no_cache")
            _arm_unpack_arm_known(pv_witness, "mo.mha.no_cache")
            if seq_len == s_pad:
                ctx.enqueue_function(
                    f_matmul,
                    output.unsafe_ptr(),
                    probs.unsafe_ptr(),
                    v_t.unsafe_ptr(),
                    pv_witness,
                    Int32(seq_len),
                    Int32(head_dim),
                    Int32(s_pad),
                    grid_dim=Dim(NPU_GRID_CORES),
                    block_dim=Dim(NPU_THREAD_MASK),
                )
            else:
                # The probs are host-readable the moment the softmax launch
                # returns: an invocation on this transport completes before
                # it returns. The conv im2col arm documents the same contract
                # and its batching caveat; this staged read inherits both.
                var probs_padded = ctx.enqueue_create_buffer[output.dtype](
                    seq_len * s_pad
                )
                _stage_k_padded_rows(
                    probs_padded.unsafe_ptr(),
                    probs.unsafe_ptr(),
                    seq_len,
                    seq_len,
                    s_pad,
                )
                ctx.enqueue_function(
                    f_matmul,
                    output.unsafe_ptr(),
                    probs_padded.unsafe_ptr(),
                    v_t.unsafe_ptr(),
                    pv_witness,
                    Int32(seq_len),
                    Int32(head_dim),
                    Int32(s_pad),
                    grid_dim=Dim(NPU_GRID_CORES),
                    block_dim=Dim(NPU_THREAD_MASK),
                )
                _ = probs_padded^
            _ = q_scaled^
            _ = scores^
            _ = probs^
            _ = v_t^
            return

        var f = ctx.load_function[npu_selector_sig](
            function_name=String("mha_no_cache"), asm=String("")
        )
        # Read per launch rather than once: nothing here outlives a dispatch,
        # and a `getenv` beside a transport that costs a second is free.
        var rt_spec = getenv(MHA_RT_ENV)
        ctx.enqueue_function(
            f,
            output.unsafe_ptr(),
            q.unsafe_ptr(),
            k.unsafe_ptr(),
            v.unsafe_ptr(),
            _witness_slot(ctx),
            Int32(seq_len),
            Int32(num_heads),
            scale,
            _mha_rt_selector(rt_spec),
            grid_dim=Dim(NPU_GRID_CORES),
            block_dim=Dim(NPU_THREAD_MASK),
        )


@extensibility.register(
    "mo.composite.masked_flash_attention_gpu", type="npu", api="npu"
)
struct MaskedFlashAttention:
    """Dispatches the `mha_masked` selector: dense attention plus an additive
    score bias.

    No pattern mints this op: `masked_flash_attention_gpu`
    (`max/python/max/nn/kernels.py`) stages it directly, and its only other
    registration is the GPU-only builtin (`builtin_kernels/attention.mojo`,
    `comptime assert is_gpu`), so without this route the Qwen3 encoder's
    attention has no NPU lowering at all -- the gap the file header carried
    since the selector was signed. The selector is `mha_no_cache` with one
    extra buffer: a `[seq_len, seq_len]` fp32 additive bias, 0 where visible
    and large-negative where masked, shared across heads and added to each
    score before the online-softmax max (`NPUKernels.def`, selector 23).
    That is exactly the operand the encoder builds
    (`causal_attention_mask_with_token_mask`) and squeezes to rank 3 before
    this op.
    """

    @staticmethod
    def execute[
        target: StaticString, rank: Int
    ](
        output: OutputTensor[rank=rank, ...],
        q: InputTensor[dtype=output.dtype, rank=rank, ...],
        k: InputTensor[dtype=output.dtype, rank=rank, ...],
        v: InputTensor[dtype=output.dtype, rank=rank, ...],
        mask: InputTensor,
        scale: Float32,
        ctx: DeviceContext,
    ) raises:
        comptime assert rank == 4, (
            "mo.composite.masked_flash_attention_gpu on the on-device NPU:"
            " BSHD inputs only"
        )
        comptime assert output.dtype == DType.float16, (
            "mo.composite.masked_flash_attention_gpu on the on-device NPU:"
            " fp16 only"
        )
        # The selector reads `bias[query, key]` rows shared by every head, so
        # a rank-4 per-head mask has no lowering here; and it reads them as
        # `const float *`, so a half-precision mask would be reinterpreted
        # rather than converted.
        comptime assert mask.rank == 3, (
            "mo.composite.masked_flash_attention_gpu on the on-device NPU:"
            " the `mha_masked` selector shares one [S, S] bias across heads,"
            " so the mask must be rank 3"
        )
        comptime assert mask.dtype == DType.float32, (
            "mo.composite.masked_flash_attention_gpu on the on-device NPU:"
            " the `mha_masked` selector reads a single-precision bias"
        )

        var batch = Int(q.dim_size(0))
        var seq_len = Int(q.dim_size(1))
        var num_heads = Int(q.dim_size(2))
        var head_dim = Int(q.dim_size(3))
        if batch != 1:
            raise Error(
                (
                    "mo.composite.masked_flash_attention_gpu on the on-device"
                    " NPU: batch must be 1, got "
                ),
                batch,
            )
        if head_dim != 128:
            raise Error(
                (
                    "mo.composite.masked_flash_attention_gpu on the on-device"
                    " NPU: head dim must be 128, got "
                ),
                head_dim,
            )

        # One `seq_len` and one `num_heads` reach the selector for Q, K and V
        # alike. Grouped attention never arrives here with fewer KV heads: the
        # encoder materializes KV to the full head count before this op
        # (`EncoderAttention._repeat_kv`), and anything else would be read
        # through Q's extents against the wrong elements.
        if k.shape() != q.shape() or v.shape() != q.shape():
            raise Error(
                "mo.composite.masked_flash_attention_gpu on the on-device"
                " NPU: the `mha_masked` selector strides Q, K and V alike, so"
                " all three must have one shape; grouped or cross attention"
                " needs a stub that passes their extents separately"
            )
        if (
            Int(mask.dim_size(0)) != 1
            or Int(mask.dim_size(1)) != seq_len
            or Int(mask.dim_size(2)) != seq_len
        ):
            raise Error(
                (
                    "mo.composite.masked_flash_attention_gpu on the on-device"
                    " NPU: the bias must be [1, seq_len, seq_len]; got ["
                ),
                Int(mask.dim_size(0)),
                ", ",
                Int(mask.dim_size(1)),
                ", ",
                Int(mask.dim_size(2)),
                "] against seq_len ",
                seq_len,
            )
        # The bias is the one operand that routinely arrives as a view (the
        # encoder squeezes a [1, 1, S, S] graph input), so its layout is
        # checked rather than assumed: the selector walks it as one dense
        # [S, S] block.
        _require_dense(
            mask.shape(),
            mask.strides(),
            "mo.composite.masked_flash_attention_gpu",
            "the bias",
        )

        var f = ctx.load_function[npu_selector_sig](
            function_name=String("mha_masked"), asm=String("")
        )
        # Derived ONCE and dispatched, rather than derived and then re-spelled
        # as a literal beside the launch. That re-spelling is the whole
        # print-vs-dispatch class -- five instances on the books, and this
        # dispatch was one of them until now: it passed `Int32(0)` under a
        # comment calling `rt` unused, so an encoder built with `HMX_ATTN=1`
        # reached the vector arm no matter what the operator set.
        var rt = _mha_masked_rt_selector(getenv(MHA_MASKED_RT_ENV))
        ctx.enqueue_function(
            f,
            output.unsafe_ptr(),
            q.unsafe_ptr(),
            k.unsafe_ptr(),
            v.unsafe_ptr(),
            mask.unsafe_ptr(),
            _witness_slot(ctx),
            Int32(seq_len),
            Int32(num_heads),
            scale,
            rt,
            grid_dim=Dim(NPU_GRID_CORES),
            block_dim=Dim(NPU_THREAD_MASK),
        )


def _require_matching_off_axis[
    rank: Int
](
    part: IndexList[rank],
    whole: IndexList[rank],
    axis: Int,
    op: StaticString,
    name: StaticString,
) raises:
    """Refuses a part whose extents off the joined axis differ from the whole's.

    One trailing-block size is derived from the joined tensor and applied to
    every part, so a part disagreeing anywhere but on the axis would be copied
    at the wrong stride -- silently, the byte count still being a plausible one.
    """
    for i in range(rank):
        if i != axis and part[i] != whole[i]:
            raise Error(
                op,
                " on the on-device NPU: ",
                name,
                " must match the joined tensor on every axis but ",
                axis,
                "; axis ",
                i,
                " is ",
                part[i],
                " against ",
                whole[i],
            )


def _copy_view_to_dense[
    dtype: DType, rank: Int
](
    output: OutputTensor[dtype=dtype, rank=rank, ...],
    input: InputTensor[dtype=dtype, rank=rank, ...],
    op: StaticString,
) raises:
    """Reads a strided `input` into a dense `output`, on the host.

    The input is deliberately *not* required to be dense: a view is exactly
    where an interior pointer and a non-unit stride arise, so the region is read
    through the input's own strides. A view also carries its offset in its
    pointer and its step in its strides, which is why no starts or steps are
    passed -- the layout the compiler already computed says everything. The
    output is a fresh buffer and is required to be dense, since it is filled
    linearly.
    """
    _require_dense(output.shape(), output.strides(), op, "the output")

    var count = Int(output.size())
    if count == 0:
        return

    var in_strides = input.strides()

    # A unit innermost stride makes each output row one run of the input, so the
    # walk moves rows rather than elements. That is the shape a chunked
    # projection takes, and it is the difference between a memcpy per row and a
    # load per element. A broadcast leaves a 0 there and falls to the scalar
    # arm, which is correct rather than merely safe: the run would repeat one
    # element.
    var last = rank - 1
    var run = Int(output.dim_size(last))
    var by_run = in_strides[last] == 1
    var rows = count // run if by_run else count

    var in_ptr = input.unsafe_ptr()
    var out_ptr = output.unsafe_ptr()
    var index = IndexList[rank]()
    for _ in range(rows):
        var offset = 0
        for a in range(rank):
            offset += index[a] * in_strides[a]
        if by_run:
            unsafe_memcpy(
                dest=out_ptr, src=in_ptr.unsafe_offset(offset), count=run
            )
            out_ptr = out_ptr.unsafe_offset(run)
        else:
            out_ptr[unsafe_offset=0] = in_ptr[unsafe_offset=offset]
            out_ptr = out_ptr.unsafe_offset(1)

        # Odometer over the axes the copy above did not consume.
        for a in range(last - 1 if by_run else last, -1, -1):
            index[a] += 1
            if index[a] < Int(output.dim_size(a)):
                break
            index[a] = 0


comptime _RELAYOUT_DEVICE_ENV = "MODULAR_NPU_RELAYOUT_DEVICE"

# Below this the 1.211 ms invoke floor rivals the walk itself (11.9 ns/elem
# fitted, HOST_WALK_CENSUS.md section 4 -- ~100 K elements to break even on
# the scalar arm); the census's smallest routable site is 131072 elements.
comptime _RELAYOUT_MIN_ELEMS = 32768


def _relayout_device_requested() -> Bool:
    """Whether this serve routes materializing permutes onto selector 47.

    Strictly "1", read per dispatch, unset the shipping default -- the host
    walk -- so the A/B is one artifact and an environment variable apart and
    the revert is unsetting it (`_add_device_requested`'s contract). Valid
    only against a catalogue carrying the relayout row: an older signed skel
    refuses the selector by name and the launch fails LOUDLY at the flush,
    never silently -- flip the knob after the window certifies, not before.
    """
    return getenv(_RELAYOUT_DEVICE_ENV) == "1"


def _arm_relayout_ran(slot: _NPUArenaPtr, op: StaticString):
    """Arms word 7 == RAN on a relayout launch, deferred to the flush.

    CORRECTNESS, not attribution: the row's REFUSED code means the kernel
    wrote NOTHING, so an accepted flush would hand the graph whatever the
    output allocation held -- `_require_matmul_ran`'s exact failure class.
    Accepting only RAN also makes the guard a live control: a host/kernel
    extent disagreement fails the render at the flush rather than feeding
    the next op stale bytes.
    """
    var entry = _arm_guard(
        slot,
        NPU_GUARD_CLASS_CORRECTNESS,
        NPU_WITNESS_RELAYOUT_PATH,
        NPU_GUARD_COND_NONE,
        0,
        op,
    )
    _accept(entry, NPU_RELAYOUT_PATH_RAN)


def _relayout_fold[
    rank: Int
](
    dims: IndexList[rank], in_strides: IndexList[rank]
) -> Tuple[Bool, Int, Int, Int]:
    """Folds a dense output's axes against its input view's strides, and
    names the batched 2-D transpose if that is what remains.

    Returns `(routable, batch, rows, cols)` in the relayout row's own terms:
    the view reads a dense `[batch, rows, cols]` input and the copy writes a
    dense `[batch, cols, rows]` output. The fold is the standard contiguity
    merge -- size-1 axes drop, adjacent axes coalesce when `stride[a] ==
    stride[a+1] * dim[a+1]` -- and the routable patterns after it are exactly
    `[cols, rows]` at strides `[1, cols]`, with an optional leading batch
    axis at stride `rows * cols`. A zero stride (broadcast) or a third
    swapped group (the 6-D unpatchify) never folds to either, which is what
    keeps this a shape statement rather than a heuristic.

    A pure function of the two lists so the harness can drive it over the
    census's stride table without a tensor or a transport behind it.
    """
    var merged_d = IndexList[rank]()
    var merged_s = IndexList[rank]()
    var m = 0
    for a in range(rank):
        var d = dims[a]
        var st = in_strides[a]
        if d == 1:
            continue
        if d <= 0 or st <= 0:
            return (False, 0, 0, 0)
        if m > 0 and merged_s[m - 1] == st * d:
            merged_d[m - 1] = merged_d[m - 1] * d
            merged_s[m - 1] = st
            continue
        merged_d[m] = d
        merged_s[m] = st
        m += 1

    if m == 2:
        var cols = merged_d[0]
        var rows = merged_d[1]
        if merged_s[0] != 1 or merged_s[1] != cols:
            return (False, 0, 0, 0)
        return (True, 1, rows, cols)
    if m == 3:
        var batch = merged_d[0]
        var cols = merged_d[1]
        var rows = merged_d[2]
        if (
            merged_s[1] != 1
            or merged_s[2] != cols
            or merged_s[0] != rows * cols
        ):
            return (False, 0, 0, 0)
        return (True, batch, rows, cols)
    return (False, 0, 0, 0)


def _relayout_device_dispatch[
    dtype: DType, rank: Int, //
](
    z: OutputTensor[dtype=dtype, rank=rank, ...],
    x: InputTensor[dtype=dtype, rank=rank, ...],
    ctx: DeviceContext,
) raises -> Bool:
    """Dispatches one materializing permute as the `relayout` selector, and
    reports whether it did; False keeps the copy on the host walk.

    The routable class is exactly what the kernel computes: a BATCHED 2-D
    TRANSPOSE. The output's dense axes are folded against the input view's
    strides (the standard contiguity merge -- adjacent axes coalesce when
    `stride[a] == stride[a+1] * dim[a+1]`), and the copy routes only when
    what remains is `[J, I]` read at strides `[1, J]`, with an optional
    leading batch axis at stride `I * J`. That covers both directions of the
    census's conv wraps (NCHW -> NHWC folds to `(N, C, H*W)`, NHWC -> NCHW to
    `(N, H*W, C)`) and the attention/tail seq permutes, and it structurally
    excludes broadcasts (a zero stride never merges to `[1, J]`) and the 6-D
    unpatchify (three swapped groups do not fold to two).

    Declines -- False, never raising, since the walk is correct for
    everything -- when the serve did not ask, when the dtype is not the
    row's fp16, when the fold fails, when the tensor is under
    `_RELAYOUT_MIN_ELEMS`, or when an extent overflows the wire's Int32
    scalars. A non-dense output raises, exactly as the walk it replaces
    would have.
    """
    comptime if dtype != DType.float16 or rank < 2:
        return False
    else:
        if not _relayout_device_requested():
            return False
        var count = Int(z.size())
        if count < _RELAYOUT_MIN_ELEMS or count > 0x7FFFFFFF:
            return False
        _require_dense(
            z.shape(), z.strides(), "a materializing view", "the output"
        )

        var fold = _relayout_fold(z.shape(), x.strides())
        if not fold[0]:
            return False
        var batch = fold[1]
        var rows = fold[2]
        var cols = fold[3]

        var f = ctx.load_function[npu_selector_sig](
            function_name=String("relayout"), asm=String("")
        )
        var slot = _witness_slot(ctx)
        _arm_relayout_ran(slot, "a materializing view")
        ctx.enqueue_function(
            f,
            z.unsafe_ptr(),
            x.unsafe_ptr(),
            slot,
            Int32(batch),
            Int32(rows),
            Int32(cols),
            grid_dim=Dim(NPU_GRID_CORES),
            block_dim=Dim(NPU_THREAD_MASK),
        )
        return True


comptime _UPSAMPLE_FOLD_ENV = "MODULAR_NPU_UPSAMPLE_FOLD"

comptime _UPSAMPLE_FOLD_DIAG_ENV = "MODULAR_NPU_UPSAMPLE_FOLD_DIAG"

comptime _UP_TRACE_KEY: StaticString = "npu_upsample_trace"

comptime _UP_TRACE_RECORDS = 8
comptime _UP_TRACE_WORDS = 8
"""Per record: kind, out pointer (2 words), and four extents. Eight records
round-robin; the graph holds one live site at a time, so the depth is slack,
not a requirement."""

comptime _UP_KIND_EXPANDED: UInt32 = 1
comptime _UP_KIND_ONES: UInt32 = 2
comptime _UP_KIND_PRODUCT: UInt32 = 3
comptime _UP_KIND_PINNED: UInt32 = 4


def _upsample_fold_requested() -> Bool:
    """Whether this serve folds the reshape x broadcast-ones upsample into
    the conv-epilogue row's gather (`upsample_log2 = 1` on selector 45).

    Strictly "1", read per dispatch, unset the shipping default -- today's
    walks -- the `_add_device_requested` contract. Valid only against a
    catalogue whose selector 45 ACCEPTS `upsample_log2 == 1` (this window's
    skel); against the declared-and-refused vintage the dispatch fails
    loudly at the flush (slot 7 code 8 fails the correctness guard), never
    silently.
    """
    return getenv(_UPSAMPLE_FOLD_ENV) == "1"


def _upsample_fold_diag_requested() -> Bool:
    """Whether the EAGER fold arm may still dispatch: diagnostic-only since
    the graph composite (`npu_upsample_conv`) became the production fold.

    The eager arm re-proves the kernel leg on a serve whose MEF predates the
    composite staging, and only on top of the fold knob itself -- this flag
    alone arms nothing.
    """
    return getenv(_UPSAMPLE_FOLD_DIAG_ENV) == "1"


def _up_trace(ctx: DeviceContext) raises -> _NPUArenaPtr:
    """The upsample recognition trace: the witness arena's own idiom, sized
    to a page of records.

    WHAT A RECORD IS, and pointedly is not: a record says "the tensor at
    this pointer was OBSERVED being built as a nearest-2x upsample" -- a
    recognition, never a data source. No stage ever reads payload through a
    recorded pointer later than the op that recorded it: the fold's staging
    reads only the conv's OWN live input (`_upsample_fold_dispatch`), so a
    stale record can at worst mis-route a conv whose input pointer AND
    rank-4 extents both collide with a dead site's -- and records are popped
    on first lookup and cleared at each new site's head to keep even that
    window one op wide. The eager op stream cannot prove consumers the way
    the graph pattern (design 2.1's `Fusion.cpp` half, post-window) will;
    this registry exists so the kernel arm is reachable and certifiable
    NOW, behind its own default-off knob, and none of it sits on the
    composite's future path.
    """
    var cached = _get_global_or_null(_UP_TRACE_KEY)
    if not cached:
        var buf = ctx.enqueue_create_buffer[DType.uint32](
            4 + _UP_TRACE_RECORDS * _UP_TRACE_WORDS
        )
        var fresh = buf.unsafe_ptr()
        for i in range(4 + _UP_TRACE_RECORDS * _UP_TRACE_WORDS):
            fresh[unsafe_offset=i] = 0
        external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
            StringSlice(_UP_TRACE_KEY), fresh.unsafe_bitcast[NoneType]()
        )
        _ = buf.take_ptr()
        cached = _get_global_or_null(_UP_TRACE_KEY)
    return _arena_ptr(cached.unsafe_value())


def _up_record(
    trace: _NPUArenaPtr,
    kind: UInt32,
    out_addr: Int,
    e0: Int,
    e1: Int,
    e2: Int,
    e3: Int,
):
    """Writes one recognition, overwriting any record with the same pointer
    (the buffer was rebuilt; the old claim is dead)."""
    var slot = -1
    for i in range(_UP_TRACE_RECORDS):
        var base = 4 + i * _UP_TRACE_WORDS
        if (
            trace[unsafe_offset=base] != 0
            and Int(trace[unsafe_offset=base + 1])
            | (Int(trace[unsafe_offset=base + 2]) << 32) == out_addr
        ):
            slot = i
            break
    if slot < 0:
        slot = Int(trace[unsafe_offset=0]) % _UP_TRACE_RECORDS
        trace[unsafe_offset=0] = UInt32(
            (Int(trace[unsafe_offset=0]) + 1) % _UP_TRACE_RECORDS
        )
    var base = 4 + slot * _UP_TRACE_WORDS
    trace[unsafe_offset=base] = kind
    trace[unsafe_offset=base + 1] = UInt32(out_addr & 0xFFFFFFFF)
    trace[unsafe_offset=base + 2] = UInt32((out_addr >> 32) & 0xFFFFFFFF)
    trace[unsafe_offset=base + 3] = UInt32(e0)
    trace[unsafe_offset=base + 4] = UInt32(e1)
    trace[unsafe_offset=base + 5] = UInt32(e2)
    trace[unsafe_offset=base + 6] = UInt32(e3)


def _up_take(
    trace: _NPUArenaPtr, kind: UInt32, addr: Int, pop: Bool
) -> Tuple[Bool, Int, Int, Int, Int]:
    """Finds a recognition by kind and pointer; `pop` consumes it."""
    for i in range(_UP_TRACE_RECORDS):
        var base = 4 + i * _UP_TRACE_WORDS
        if trace[unsafe_offset=base] != kind:
            continue
        if (
            Int(trace[unsafe_offset=base + 1])
            | (Int(trace[unsafe_offset=base + 2]) << 32)
        ) != addr:
            continue
        var e0 = Int(trace[unsafe_offset=base + 3])
        var e1 = Int(trace[unsafe_offset=base + 4])
        var e2 = Int(trace[unsafe_offset=base + 5])
        var e3 = Int(trace[unsafe_offset=base + 6])
        if pop:
            trace[unsafe_offset=base] = 0
        return (True, e0, e1, e2, e3)
    return (False, 0, 0, 0, 0)


def _up_clear(trace: _NPUArenaPtr):
    """Drops every recognition: a new site's head starts a clean table."""
    for i in range(_UP_TRACE_RECORDS):
        trace[unsafe_offset=4 + i * _UP_TRACE_WORDS] = 0


def _upsample_note_materialize[
    dtype: DType, rank: Int, //
](
    z: OutputTensor[dtype=dtype, rank=rank, ...],
    x: InputTensor[dtype=dtype, rank=rank, ...],
    ctx: DeviceContext,
) raises:
    """Recognizes the upsample idiom's two materialize legs and the conv
    wrap that follows them, off the exact shapes `interpolate_2d_nearest`
    authors (`vae_flux2.py:724-763`).

    Three signatures, each a strict stride statement (anything else is not
    the idiom and records nothing):

    - EXPANDED: rank-6 `[N, C, H, 2, W, 2]` reading `[N, C, H, W]` dense
      with stride 0 on both duplicated axes -- `x_reshaped` broadcast.
    - ONES: the same rank-6 extents reading a 4-element source (strides
      `[0, 0, 0, 2, 0, 1]`), all four elements exactly 1.0 -- the
      broadcast-ones operand, read HERE while it is this op's own live
      input.
    - PINNED: rank-4 `[N, 2H, 2W, C]` reading `[N, C, 2H, 2W]` dense (the
      conv wrap's pin) whose base pointer carries a PRODUCT recognition
      with agreeing extents.

    Recording happens AFTER the walk produced the dense output, so every
    consumer of these tensors is untouched whether or not a conv ever folds.
    """
    comptime if dtype != DType.float16:
        return
    comptime if rank == 6:
        var n = Int(z.dim_size(0))
        var c = Int(z.dim_size(1))
        var h = Int(z.dim_size(2))
        var w = Int(z.dim_size(4))
        if Int(z.dim_size(3)) != 2 or Int(z.dim_size(5)) != 2:
            return
        var s = x.strides()
        var trace = _up_trace(ctx)
        if (
            s[3] == 0
            and s[5] == 0
            and s[4] == 1
            and s[2] == w
            and s[1] == h * w
            and (n == 1 or s[0] == c * h * w)
        ):
            # A new site begins at its expansion; drop stale recognitions.
            _up_clear(trace)
            _up_record(
                trace,
                _UP_KIND_EXPANDED,
                Int(z.unsafe_ptr()),
                n,
                c,
                h,
                w,
            )
            return
        if s[0] == 0 and s[1] == 0 and s[2] == 0 and s[3] == 2 and s[4] == 0 and s[5] == 1:
            var src = x.unsafe_ptr()
            for i in range(4):
                if src[unsafe_offset=i] != Scalar[dtype](1):
                    return
            _up_record(
                trace, _UP_KIND_ONES, Int(z.unsafe_ptr()), n, c, h, w
            )
        return
    comptime if rank == 4:
        var n4 = Int(z.dim_size(0))
        var h2 = Int(z.dim_size(1))
        var w2 = Int(z.dim_size(2))
        var c4 = Int(z.dim_size(3))
        var s4 = x.strides()
        if not (
            s4[3] == h2 * w2
            and s4[1] == w2
            and s4[2] == 1
            and (n4 == 1 or s4[0] == c4 * h2 * w2)
        ):
            return
        var trace = _up_trace(ctx)
        var prod = _up_take(
            trace, _UP_KIND_PRODUCT, Int(x.unsafe_ptr()), False
        )
        if (
            prod[0]
            and prod[1] == n4
            and prod[2] == c4
            and 2 * prod[3] == h2
            and 2 * prod[4] == w2
        ):
            _up_record(
                trace,
                _UP_KIND_PINNED,
                Int(z.unsafe_ptr()),
                n4,
                c4,
                prod[3],
                prod[4],
            )


def _upsample_note_mul[
    dtype: DType, rank: Int, //
](
    output: OutputTensor[dtype=dtype, rank=rank, ...],
    x: InputTensor[dtype=dtype, rank=rank, ...],
    y: InputTensor[dtype=dtype, rank=rank, ...],
    ctx: DeviceContext,
) raises:
    """Joins the idiom's mul: EXPANDED times ONES (either order) makes the
    product a recognized upsample. The reshape that follows is a view, so
    the PRODUCT pointer is what the conv wrap's pin will read."""
    comptime if dtype != DType.float16 or rank != 6:
        return
    else:
        var trace = _up_trace(ctx)
        var ex = _up_take(trace, _UP_KIND_EXPANDED, Int(x.unsafe_ptr()), False)
        var ones_addr = Int(y.unsafe_ptr())
        if not ex[0]:
            ex = _up_take(trace, _UP_KIND_EXPANDED, Int(y.unsafe_ptr()), False)
            ones_addr = Int(x.unsafe_ptr())
        if not ex[0]:
            return
        if not _up_take(trace, _UP_KIND_ONES, ones_addr, False)[0]:
            return
        if (
            Int(output.dim_size(0)) != ex[1]
            or Int(output.dim_size(1)) != ex[2]
            or Int(output.dim_size(2)) != ex[3]
            or Int(output.dim_size(4)) != ex[4]
        ):
            return
        _up_record(
            trace,
            _UP_KIND_PRODUCT,
            Int(output.unsafe_ptr()),
            ex[1],
            ex[2],
            ex[3],
            ex[4],
        )


@register_internal("mogg.call.materialize.npu")
def npu_view_copy[
    dtype: DType,
    rank: Int,
    //,
    *,
    target: StaticString,
    _trace_name: StaticString = "mogg.npu_view_copy",
](
    z: OutputTensor[dtype=dtype, rank=rank, ...],
    x: InputTensor[dtype=dtype, rank=rank, ...],
    ctx: DeviceContext,
) raises:
    """Materializes a view on this device, replacing the generic copy.

    Every view kernel -- `mo.slice`, `mo.transpose`, `mo.static.broadcast_to`,
    and the `mo.split` that lowers through slice views -- starts as an alias
    plus one materialize, and `mogg-fuse-views` erases the materialize wherever
    it can fold the alias away. What survives runs this. The generic
    `view_copy_impl` cannot serve it here: it is `foreach` over `elementwise`,
    which is generated device code, and this accelerator has no JIT to load it.

    So the copy is a host walk, for the reason `Concat` and `Split` give:
    device memory on this transport is host memory. It is registered under
    `mogg.call.materialize.npu` and reached only through
    `MojoRegistry::lookupPrimitiveForDevice`, so no other device sees it.

    Registering this rather than an `execute` for `mo.slice` is what keeps the
    ops themselves untouched. An `execute` would win kernel resolution and take
    `mo.slice` out of the view machinery on this device, turning every alias
    into a copy -- including the 162 whole-tensor slices a FLUX denoise step
    folds for free, 100 of which slice a `[3072, 3072]` weight.

    Under `MODULAR_NPU_RELAYOUT_DEVICE=1` the transpose class -- the batched
    2-D fold `_relayout_device_dispatch` names, 77 sites and 80.6% of the
    measured non-dispatch pool -- goes to the signed `relayout` row instead,
    and everything it declines (broadcasts, the 6-D unpatchify, small
    tensors, non-fp16) keeps this walk unchanged.
    """
    # A held matmul-chain member must run before this op touches any
    # operand: this is the 186A host-reader class, closed per site.
    _matmul_chain_flush()
    if _relayout_device_dispatch(z, x, ctx):
        if _upsample_fold_requested():
            _upsample_note_materialize(z, x, ctx)
        return
    _copy_view_to_dense(z, x, "a materializing view")
    if _upsample_fold_requested():
        _upsample_note_materialize(z, x, ctx)


@register_internal("mgp.buffer.concat.npu")
def npu_buffer_concat[
    bDevice: StaticString
](
    output: OwnedByteBuffer,
    inputs: Array[OwnedByteBuffer, ...],
    call_ctx: DeviceContext,
) raises:
    """Joins whole buffers back to back on the host, replacing the generic.

    `mgp.buffer.concat` is what an axis-0 `mo.concat` becomes when
    `ConcatInPlacePattern` claims it and `ConcatInPlacePass` then cannot
    replace the copy with a borrow -- the FLUX graph's packed-ids join is
    exactly that shape: its operands are squeezed graph inputs, whose
    buffers are borrowed rather than locally allocated. The generic
    primitive serves the op with `nn.concat`, whose non-CPU arm is generated
    device code, and this accelerator has no JIT to load it.

    So the join is a host copy, for the reason `Concat` and `Split` give:
    device memory on this transport is host memory. It is registered under
    `mgp.buffer.concat.npu` and reached only through
    `MojoRegistry::lookupPrimitiveForDevice`, so no other device sees it.

    Byte-blind by construction: a buffer join has no axis and no dtype, so
    the slab arithmetic `mo.concat`'s stub needs has no analogue here.
    """
    # A held matmul-chain member must run before this op touches any
    # operand: this is the 186A host-reader class, closed per site.
    _matmul_chain_flush()
    var total = 0
    for i in range(len(inputs)):
        total += inputs[i].size()
    if total != output.size():
        raise Error(
            "mgp.buffer.concat on the on-device NPU: the output holds ",
            output.size(),
            " bytes but the inputs supply ",
            total,
        )
    var out_ptr = output.unsafe_ptr()
    var offset = 0
    for i in range(len(inputs)):
        var n = inputs[i].size()
        if n != 0:
            unsafe_memcpy(
                dest=out_ptr.unsafe_offset(offset),
                src=inputs[i].unsafe_ptr(),
                count=n,
            )
        offset += n


@extensibility.register("mo.concat", type="npu", api="npu")
struct Concat:
    """Copies each input into its slice of the output, on the host.

    Pure data movement, and the only registration here that reaches no selector:
    device memory on this transport is host memory, which
    `_require_identity_affine` already relies on to read a weight, so the join
    is `outer` contiguous copies per input rather than a dispatch. That is also
    what the hand-written driver does -- its plan marks the joint concat and the
    text drop as host copies with no selector at all.

    Plain tensors rather than the built-in's fused ones, for the reason
    `Softmax` gives: `unsafe_ptr()` has to address a materialized buffer, and a
    fused operand would produce its data from a lambda the device cannot run.

    No dtype gate, unlike every dispatching stub above. There is no selector
    that
    had to be compiled for a dtype, and a copy does not interpret what it moves:
    element counts are in units of the storage type, so a packed encoding
    travels correctly as its packed type.
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        axis: Int,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        inputs: InputVariadicTensors[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        # A held matmul-chain member must run before this op touches any
        # operand: this is the 186A host-reader class, closed per site.
        _matmul_chain_flush()
        comptime ax = axis if axis >= 0 else axis + rank
        comptime assert (
            0 <= ax < rank
        ), "mo.concat on the on-device NPU: axis out of range"

        # `inner` is the contiguous trailing block that one step along the axis
        # covers; `outer` is the number of independent slabs before it. At axis
        # 0 there is a single slab and each input is one contiguous copy.
        var inner = 1
        for i in range(ax + 1, rank):
            inner *= Int(output.dim_size(i))
        var outer = 1
        for i in range(ax):
            outer *= Int(output.dim_size(i))
        var out_axis = Int(output.dim_size(ax))
        var out_slab = out_axis * inner

        _require_dense(
            output.shape(), output.strides(), "mo.concat", "the output"
        )
        var out_ptr = output.unsafe_ptr()

        var offset = 0
        comptime for j in range(inputs.size):
            var extent = Int(inputs[j].dim_size(ax))
            _require_matching_off_axis(
                inputs[j].shape(), output.shape(), ax, "mo.concat", "an input"
            )
            _require_dense(
                inputs[j].shape(), inputs[j].strides(), "mo.concat", "an input"
            )
            # An empty input contributes nothing and its pointer need not
            # address anything, so it is skipped rather than copied as zero
            # bytes.
            if extent != 0:
                var src = inputs[j].unsafe_ptr()
                var block = extent * inner
                for slab in range(outer):
                    unsafe_memcpy(
                        dest=out_ptr.unsafe_offset(
                            slab * out_slab
                        ).unsafe_offset(offset * inner),
                        src=src.unsafe_offset(slab * block),
                        count=block,
                    )
            offset += extent

        if offset != out_axis:
            raise Error(
                (
                    "mo.concat on the on-device NPU: the inputs must tile the"
                    " output along the axis, got "
                ),
                offset,
                " against ",
                out_axis,
            )


@extensibility.register("mo.split", type="npu", api="npu")
struct Split:
    """Copies each output's slice out of the input, on the host.

    The dual of `Concat` and a host copy for the same reason: where the join
    reads dense inputs into a strided output, this reads a strided input into
    dense outputs. `split_sizes` is accepted to match the op's signature and
    otherwise unused, the per-output extents already being carried by the result
    shapes
    -- which is how the built-in reads them too.
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        _trace_name: StaticString,
        axis: Int,
    ](
        output: OutputVariadicTensors[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        split_sizes: InputTensor[rank=1, ...],
        ctx: DeviceContext,
    ) capturing raises:
        # A held matmul-chain member must run before this op touches any
        # operand: this is the 186A host-reader class, closed per site.
        _matmul_chain_flush()
        comptime ax = axis if axis >= 0 else axis + rank
        comptime assert (
            0 <= ax < rank
        ), "mo.split on the on-device NPU: axis out of range"

        var inner = 1
        for i in range(ax + 1, rank):
            inner *= Int(input.dim_size(i))
        var outer = 1
        for i in range(ax):
            outer *= Int(input.dim_size(i))
        var in_axis = Int(input.dim_size(ax))
        var in_slab = in_axis * inner

        _require_dense(input.shape(), input.strides(), "mo.split", "the input")
        var in_ptr = input.unsafe_ptr()

        var offset = 0
        comptime for j in range(output.size):
            var extent = Int(output[j].dim_size(ax))
            _require_matching_off_axis(
                output[j].shape(), input.shape(), ax, "mo.split", "an output"
            )
            _require_dense(
                output[j].shape(), output[j].strides(), "mo.split", "an output"
            )
            if extent != 0:
                var dst = output[j].unsafe_ptr()
                var block = extent * inner
                for slab in range(outer):
                    unsafe_memcpy(
                        dest=dst.unsafe_offset(slab * block),
                        src=in_ptr.unsafe_offset(slab * in_slab).unsafe_offset(
                            offset * inner
                        ),
                        count=block,
                    )
            offset += extent

        if offset != in_axis:
            raise Error(
                (
                    "mo.split on the on-device NPU: the outputs must tile the"
                    " input along the axis, got "
                ),
                offset,
                " against ",
                in_axis,
            )


@extensibility.register("mo.cast", type="npu", api="npu")
struct Cast:
    """Converts every element between two dtypes, on the host.

    The third registration here to reach no selector, and it reaches none for a
    different reason than `Concat` and `Split` do. Those are pure movement, so
    there is nothing for a device to compute. This is real elementwise work --
    fp16 and bf16 are the same width and different numbers -- but the signed
    kernel set carries no cast entry point at all, so there is nothing to
    dispatch to. Device memory on this transport is host memory, which is what
    lets the conversion happen here instead.

    Taking `mo.cast` over is the same move `Silu` makes and for the same reason:
    the compiler otherwise lowers it to an elementwise SIMD lambda, which on this
    target is device code that does not compile, so a graph carrying a single
    cast is refused outright rather than merely slowed.

    Because it dispatches nothing, no selector name reaches the artifact and the
    kernel package's path does not either -- the conversion is inlined into the
    host sequence the runtime executes. `mo.cast` in the artifact and this
    stub's own refusal message are what tie a lowering to this file;
    `test_npu_graph_cast.py` asserts both.

    TODO(skye): this walks the tensor on the host, vectorized but without HVX.
    A signed `cast` selector would move it onto the device; the trigger is the
    next re-sign of the kernel library, since appending a row is what a re-sign
    costs and nothing else here needs one.

    No dtype gate. There is no selector that had to be compiled for a dtype, and
    the conversion is whatever the two element types say it is.
    """

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor,
        input: InputTensor[rank=output.rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        # A held matmul-chain member must run before this op touches any
        # operand: this is the 186A host-reader class, closed per site.
        _matmul_chain_flush()
        if output.shape() != input.shape():
            raise Error(
                "mo.cast on the on-device NPU: input and output shapes differ"
            )
        # Dense on both sides, as `Concat` requires: the walk below is pointer
        # linear, and a view operand's stride-0 alias is not the same memory.
        _require_dense(
            output.shape(), output.strides(), "mo.cast", "the output"
        )
        _require_dense(input.shape(), input.strides(), "mo.cast", "the input")

        var count = Int(output.size())
        var src = input.unsafe_ptr()
        var dst = output.unsafe_ptr()

        # A host vector width rather than the device's: this loop runs on the
        # ARM64 host, so it is the host's SIMD that applies.
        comptime W = 8
        var vectorized_end = count - (count % W)
        for i in range(0, vectorized_end, W):
            dst.unsafe_store[W](i, src.unsafe_load[W](i).cast[output.dtype]())
        for i in range(vectorized_end, count):
            dst.unsafe_store[1](i, src.unsafe_load[1](i).cast[output.dtype]())


def _binary_host_walk[
    dtype: DType,
    rank: Int,
    //,
    op: StaticString,
](
    output: OutputTensor[dtype=dtype, rank=rank, ...],
    x: InputTensor[dtype=dtype, rank=rank, ...],
    y: InputTensor[dtype=dtype, rank=rank, ...],
) raises:
    """Computes a binary elementwise op on the host, `Cast`'s move widened.

    The compiler lowers a bare binary elementwise op to a fused-region SIMD
    lambda, which on this target is generated device code that cannot load, so
    a graph carrying one is refused outright. The signed set has no plain
    add/mul row to dispatch either -- the residual and bias shapes that
    dominate a model go through the composite rules and their signed kernels
    -- so what remains is the long tail (a rope application's products, a BN
    denormalize, an output range map), each a handful of sites per graph.
    Device memory on this transport is host memory, which is what lets the
    tail run here, exactly as `Cast`, `Concat` and the view materializer
    already do.

    TODO(skye): a host walk, vectorized but without HVX. The rope application
    and the SwiGLU product deserve signed rows (`rope_half` exists; SwiGLU
    needs the packed-halves concat the hand driver does) -- the trigger is
    phase 2's dispatch work, where moving the tail on-device is measured
    rather than assumed.

    Dense on every side, as `Cast` requires: a spread or strided operand
    arrives materialized by `npu_view_copy`, and one that does not is refused
    by name rather than walked as the wrong memory.
    """
    if output.shape() != x.shape() or output.shape() != y.shape():
        raise Error(
            op,
            (
                " on the on-device NPU: the host walk takes one shape across"
                " both inputs and the output"
            ),
        )
    _require_dense(output.shape(), output.strides(), op, "the output")
    _require_dense(x.shape(), x.strides(), op, "x")
    _require_dense(y.shape(), y.strides(), op, "y")

    var count = Int(output.size())
    var xs = x.unsafe_ptr()
    var ys = y.unsafe_ptr()
    var dst = output.unsafe_ptr()

    # A host vector width rather than the device's: this loop runs on the
    # ARM64 host, so it is the host's SIMD that applies. The op switch is
    # comptime, so each registration compiles to the one loop body it names.
    comptime W = 8
    var vectorized_end = count - (count % W)
    comptime if op == "mo.add":
        for i in range(0, vectorized_end, W):
            dst.unsafe_store[W](i, xs.unsafe_load[W](i) + ys.unsafe_load[W](i))
        for i in range(vectorized_end, count):
            dst.unsafe_store[1](i, xs.unsafe_load[1](i) + ys.unsafe_load[1](i))
    elif op == "mo.sub":
        for i in range(0, vectorized_end, W):
            dst.unsafe_store[W](i, xs.unsafe_load[W](i) - ys.unsafe_load[W](i))
        for i in range(vectorized_end, count):
            dst.unsafe_store[1](i, xs.unsafe_load[1](i) - ys.unsafe_load[1](i))
    elif op == "mo.mul":
        for i in range(0, vectorized_end, W):
            dst.unsafe_store[W](i, xs.unsafe_load[W](i) * ys.unsafe_load[W](i))
        for i in range(vectorized_end, count):
            dst.unsafe_store[1](i, xs.unsafe_load[1](i) * ys.unsafe_load[1](i))
    elif op == "mo.min":
        for i in range(0, vectorized_end, W):
            dst.unsafe_store[W](
                i, min(xs.unsafe_load[W](i), ys.unsafe_load[W](i))
            )
        for i in range(vectorized_end, count):
            dst.unsafe_store[1](
                i, min(xs.unsafe_load[1](i), ys.unsafe_load[1](i))
            )
    elif op == "mo.max":
        for i in range(0, vectorized_end, W):
            dst.unsafe_store[W](
                i, max(xs.unsafe_load[W](i), ys.unsafe_load[W](i))
            )
        for i in range(vectorized_end, count):
            dst.unsafe_store[1](
                i, max(xs.unsafe_load[1](i), ys.unsafe_load[1](i))
            )
    else:
        comptime assert False, "unmapped binary host-walk op"


def _unary_host_walk[
    dtype: DType,
    rank: Int,
    //,
    op: StaticString,
](
    output: OutputTensor[dtype=dtype, rank=rank, ...],
    x: InputTensor[dtype=dtype, rank=rank, ...],
) raises:
    """Computes a unary elementwise op on the host; `_binary_host_walk` says
    why the tail runs here. The bodies are the same `std.math` calls the
    compiler's own elementwise lambdas inline, so a value computed on this
    route matches one computed on any other target's."""
    if output.shape() != x.shape():
        raise Error(
            op,
            (
                " on the on-device NPU: the host walk takes one shape for the"
                " input and the output"
            ),
        )
    _require_dense(output.shape(), output.strides(), op, "the output")
    _require_dense(x.shape(), x.strides(), op, "the input")

    var count = Int(output.size())
    var xs = x.unsafe_ptr()
    var dst = output.unsafe_ptr()

    comptime W = 8
    var vectorized_end = count - (count % W)
    comptime if op == "mo.sqrt":
        for i in range(0, vectorized_end, W):
            dst.unsafe_store[W](i, sqrt(xs.unsafe_load[W](i)))
        for i in range(vectorized_end, count):
            dst.unsafe_store[1](i, sqrt(xs.unsafe_load[1](i)))
    elif op == "mo.round":
        for i in range(0, vectorized_end, W):
            dst.unsafe_store[W](i, round(xs.unsafe_load[W](i)))
        for i in range(vectorized_end, count):
            dst.unsafe_store[1](i, round(xs.unsafe_load[1](i)))
    elif op == "mo.relu":
        for i in range(0, vectorized_end, W):
            dst.unsafe_store[W](i, max(xs.unsafe_load[W](i), 0))
        for i in range(vectorized_end, count):
            dst.unsafe_store[1](i, max(xs.unsafe_load[1](i), 0))
    else:
        comptime assert False, "unmapped unary host-walk op"


comptime _ADD_DEVICE_ENV = "MODULAR_NPU_ADD_DEVICE"

# The ones gate is staged per dispatch, so its length is this route's own
# cost; the cap keeps that staging proportional to what one dispatch saves.
# Every trailing axis the packed-32 VAE census stages is <= 512.
comptime _ADD_ONES_GATE_CAP = 16384


def _add_device_requested() -> Bool:
    """Whether this serve routes bare `mo.add`s onto the signed kernel set.

    Strictly "1", the reading `_stage_native_weight` gives its own variable,
    and read per dispatch for the same reason: one MEF outlives any single
    serve's environment. Unset is the shipping default -- the host walk --
    so the two arms of an A/B are one artifact and an environment variable
    apart, and the revert is unsetting it.
    """
    return getenv(_ADD_DEVICE_ENV) == "1"


def _add_gate_residual_dispatch[
    dtype: DType,
    rank: Int,
    //,
](
    output: OutputTensor[dtype=dtype, rank=rank, ...],
    x: InputTensor[dtype=dtype, rank=rank, ...],
    y: InputTensor[dtype=dtype, rank=rank, ...],
    ctx: DeviceContext,
) raises -> Bool:
    """Dispatches one bare fp16/bf16 add as `gate_residual` with a ones gate,
    and reports whether it did; False keeps the add on the host walk.

    The signed set carries no plain half-precision add row and rows are
    append-only between re-sign windows -- but `gate_residual` (selector 19,
    bf16 twin at 20) already computes `out = x + gate * y` with
    `gate_row_stride == 0` sharing one length-D gate across every row
    (`modulation_kernel.mojo`, `gate_residual_2d`). A staged ones gate makes
    that a plain elementwise add exactly: the kernel accumulates in fp32
    (`ACCUM`), where `1.0 * y` is exact, and its one rounding on store is the
    correctly rounded half-precision sum -- the value the host walk's native
    add produces. Staging the small operand a selector needs is `Matmul`'s
    own k-pad move, and the buffer is dropped after the enqueue the same way.

    Declines -- returning False rather than raising, since the host walk is
    correct for everything and owns the refusal messages -- when the serve
    did not ask (`MODULAR_NPU_ADD_DEVICE`), when the shapes differ (the walk
    refuses those by name), when the tensor is empty, when the extents
    overflow the wire's Int32 scalars or the kernel's own 32-bit row
    arithmetic, or when the trailing axis is past `_ADD_ONES_GATE_CAP`.
    Density is required rather than declined: the walk would refuse a
    non-dense operand identically, so raising here changes nothing
    observable.
    """
    comptime assert (
        dtype == DType.float16 or dtype == DType.bfloat16
    ), "the gate_residual selectors are fp16 and bf16 (NPUKernels.def 19/20)"
    comptime assert (
        rank >= 2
    ), "a rank-1 add has no leading axis to take the row count from"
    if not _add_device_requested():
        return False
    if output.shape() != x.shape() or output.shape() != y.shape():
        return False
    var row_length = Int(output.dim_size(rank - 1))
    if row_length <= 0 or row_length > _ADD_ONES_GATE_CAP:
        return False
    var num_rows = 1
    for i in range(rank - 1):
        num_rows *= Int(output.dim_size(i))
    if num_rows <= 0 or num_rows > 0x7FFFFFFF // row_length:
        return False
    _require_dense(output.shape(), output.strides(), "mo.add", "the output")
    _require_dense(x.shape(), x.strides(), "mo.add", "x")
    _require_dense(y.shape(), y.strides(), "mo.add", "y")

    var gate = ctx.enqueue_create_buffer[dtype](row_length)
    var gate_ptr = gate.unsafe_ptr()
    for j in range(row_length):
        gate_ptr[unsafe_offset=j] = Scalar[dtype](1)

    var f = ctx.load_function[npu_selector_sig](
        function_name=String(_gate_residual_selector[dtype]()), asm=String("")
    )
    ctx.enqueue_function(
        f,
        output.unsafe_ptr(),
        x.unsafe_ptr(),
        y.unsafe_ptr(),
        gate_ptr,
        _witness_slot(ctx),
        Int32(num_rows),
        Int32(row_length),
        Int32(0),
        grid_dim=Dim(NPU_GRID_CORES),
        block_dim=Dim(NPU_THREAD_MASK),
    )
    _ = gate^
    return True


def _add_vadd_dispatch[
    dtype: DType,
    rank: Int,
    //,
](
    output: OutputTensor[dtype=dtype, rank=rank, ...],
    x: InputTensor[dtype=dtype, rank=rank, ...],
    y: InputTensor[dtype=dtype, rank=rank, ...],
    ctx: DeviceContext,
) raises -> Bool:
    """Dispatches one bare fp32 add as the `vadd` selector, and reports
    whether it did; False keeps the add on the host walk.

    `vadd` (selector 0) is the flat fp32 elementwise add: lhs, rhs and result
    buffers in that order, one element-count scalar, no witness buffer. The
    slot is still taken -- every graph-path launch takes one, and the take is
    where the transport-mode refusals live (`_witness_slot`) -- it just has
    no argument slot to travel in, so nothing writes it and the flush reads
    no guard from it. Declines mirror `_add_gate_residual_dispatch`'s, minus
    the gate staging this route does not need.
    """
    comptime assert (
        dtype == DType.float32
    ), "the vadd selector is fp32 (NPUKernels.def selector 0)"
    if not _add_device_requested():
        return False
    if output.shape() != x.shape() or output.shape() != y.shape():
        return False
    var count = Int(output.size())
    if count <= 0 or count > 0x7FFFFFFF:
        return False
    _require_dense(output.shape(), output.strides(), "mo.add", "the output")
    _require_dense(x.shape(), x.strides(), "mo.add", "x")
    _require_dense(y.shape(), y.strides(), "mo.add", "y")

    var f = ctx.load_function[npu_selector_sig](
        function_name=String("vadd"), asm=String("")
    )
    _ = _witness_slot(ctx)
    ctx.enqueue_function(
        f,
        x.unsafe_ptr(),
        y.unsafe_ptr(),
        output.unsafe_ptr(),
        Int32(count),
        grid_dim=Dim(NPU_GRID_CORES),
        block_dim=Dim(NPU_THREAD_MASK),
    )
    return True


@extensibility.register("mo.add", type="npu", api="npu")
struct Add:
    """The bare-add tail: signed selectors when the serve asks, else the host.

    `MODULAR_NPU_ADD_DEVICE=1` routes eligible same-shape dense adds onto the
    signed kernel set -- fp16/bf16 through `gate_residual`'s ones-gate arm,
    fp32 through `vadd` -- and anything either route declines walks on the
    host exactly as before. The composite rules still own the gated shapes
    they claim; what this moves is the bare tail, which the packed-32 VAE
    decode stages 58 of (`rmo.add`: every residual close, the upsample joins,
    and the output head's range-map adds).
    """

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor,
        x: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        y: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        # A held matmul-chain member must run before this op touches any
        # operand: this is the 186A host-reader class, closed per site.
        _matmul_chain_flush()
        comptime if (
            output.dtype == DType.float16 or output.dtype == DType.bfloat16
        ) and output.rank >= 2:
            if _add_gate_residual_dispatch(output, x, y, ctx):
                return
        comptime if output.dtype == DType.float32:
            if _add_vadd_dispatch(output, x, y, ctx):
                return
        _binary_host_walk[op="mo.add"](output, x, y)


@extensibility.register("mo.sub", type="npu", api="npu")
struct Sub:
    """The rope application's difference half, on the host."""

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor,
        x: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        y: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        # A held matmul-chain member must run before this op touches any
        # operand: this is the 186A host-reader class, closed per site.
        _matmul_chain_flush()
        _binary_host_walk[op="mo.sub"](output, x, y)


@extensibility.register("mo.mul", type="npu", api="npu")
struct Mul:
    """The bare-product tail (rope, SwiGLU's two-projection spelling, the
    range map), on the host.

    Under `MODULAR_NPU_UPSAMPLE_FOLD=1` the walk also NOTES the upsample
    idiom's join (EXPANDED times ONES) after producing it -- recognition
    only, the product's bytes and every consumer unchanged."""

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor,
        x: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        y: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        # A held matmul-chain member must run before this op touches any
        # operand: this is the 186A host-reader class, closed per site.
        _matmul_chain_flush()
        _binary_host_walk[op="mo.mul"](output, x, y)
        if _upsample_fold_requested():
            _upsample_note_mul(output, x, y, ctx)


@extensibility.register("mo.min", type="npu", api="npu")
struct Min:
    """The range map's upper clamp, on the host."""

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor,
        x: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        y: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        # A held matmul-chain member must run before this op touches any
        # operand: this is the 186A host-reader class, closed per site.
        _matmul_chain_flush()
        _binary_host_walk[op="mo.min"](output, x, y)


@extensibility.register("mo.max", type="npu", api="npu")
struct Max:
    """The saturation arm a canonicalizer did not turn into `mo.relu`."""

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor,
        x: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        y: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        # A held matmul-chain member must run before this op touches any
        # operand: this is the 186A host-reader class, closed per site.
        _matmul_chain_flush()
        _binary_host_walk[op="mo.max"](output, x, y)


@extensibility.register("mo.sqrt", type="npu", api="npu")
struct Sqrt:
    """The BN denormalize's per-channel std, on the host."""

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor,
        x: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        # A held matmul-chain member must run before this op touches any
        # operand: this is the 186A host-reader class, closed per site.
        _matmul_chain_flush()
        _unary_host_walk[op="mo.sqrt"](output, x)


@extensibility.register("mo.round", type="npu", api="npu")
struct Round:
    """The range map's round-before-uint8, on the host."""

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor,
        x: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        # A held matmul-chain member must run before this op touches any
        # operand: this is the 186A host-reader class, closed per site.
        _matmul_chain_flush()
        _unary_host_walk[op="mo.round"](output, x)


@extensibility.register("mo.relu", type="npu", api="npu")
struct Relu:
    """The range map's lower clamp, canonicalized from `max(x, 0)`."""

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor,
        x: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        # A held matmul-chain member must run before this op touches any
        # operand: this is the 186A host-reader class, closed per site.
        _matmul_chain_flush()
        _unary_host_walk[op="mo.relu"](output, x)


@extensibility.register("mo.gather", type="npu", api="npu")
struct Gather:
    """The embedding lookup, as the host row-copy the P1 design names.

    The one gather the render stages: the Qwen3 encoder's `embed_tokens`,
    `[vocab, hidden]` fp16 indexed by `[seq]` token ids on axis 0. Pure
    movement -- each output row is one table row -- so it joins `Concat`,
    `Split` and the view materializer on the host rather than costing a
    signed kernel ("gather is a host walk", `EXECUTION_PLAN.md` P1.1). Any
    other gather shape is refused by name so a new consumer runs the census
    rather than inheriting a walk written for this one.
    """

    @staticmethod
    def execute[
        axis: Int,
        target: StaticString,
    ](
        output: OutputTensor,
        input: InputTensor[dtype=output.dtype, ...],
        indices: InputTensor,
        ctx: DeviceContext,
    ) capturing raises:
        # A held matmul-chain member must run before this op touches any
        # operand: this is the 186A host-reader class, closed per site.
        _matmul_chain_flush()
        comptime assert axis == 0, (
            "mo.gather on the on-device NPU: the host walk copies table rows,"
            " so only axis 0 is served"
        )
        comptime assert (
            input.rank == 2 and indices.rank == 1 and output.rank == 2
        ), (
            "mo.gather on the on-device NPU: served as [vocab, hidden] rows by"
            " a rank-1 index; anything else needs its own walk"
        )
        comptime assert indices.dtype in (
            DType.int32,
            DType.int64,
            DType.uint32,
        ), "mo.gather on the on-device NPU: integer indices only"

        _require_dense(
            output.shape(), output.strides(), "mo.gather", "the output"
        )
        _require_dense(input.shape(), input.strides(), "mo.gather", "the table")
        _require_dense(
            indices.shape(), indices.strides(), "mo.gather", "the indices"
        )

        var rows = Int(indices.dim_size(0))
        var vocab = Int(input.dim_size(0))
        var width = Int(input.dim_size(1))
        if Int(output.dim_size(0)) != rows or Int(output.dim_size(1)) != width:
            raise Error(
                "mo.gather on the on-device NPU: the output must be one table"
                " row per index"
            )

        var table = input.unsafe_ptr()
        var ids = indices.unsafe_ptr()
        var dst = output.unsafe_ptr()
        for i in range(rows):
            var row = Int(ids[unsafe_offset=i])
            # A loud bound beats reading whatever follows the table: the ids
            # are runtime data, and this transport's device memory is host
            # memory, so the check costs a compare.
            if row < 0 or row >= vocab:
                raise Error(
                    "mo.gather on the on-device NPU: index ",
                    row,
                    " is out of the table's ",
                    vocab,
                    " rows",
                )
            unsafe_memcpy(
                dest=dst.unsafe_offset(i * width),
                src=table.unsafe_offset(row * width),
                count=width,
            )


def _permute_filter_fused_k[
    dtype: DType, dst_origin: MutOrigin, src_origin: MutOrigin, //
](
    dst: Pointer[Scalar[dtype], dst_origin],
    src: Pointer[Scalar[dtype], src_origin],
    c_out: Int,
    c_in: Int,
    r: Int,
    s: Int,
):
    """Permutes an `[F, C, R, S]` filter into the fused arm's k order.

    The fused selectors gather A in `(r*S + s)*C_in + c` order -- channel
    FASTEST within a tap -- where the FCRS checkpoint is channel-slowest, so
    the `[C_out, K]` b-transposed rows the kernel reads are this permutation
    of the checkpoint bytes. `C_out * K` element moves, the same size as the
    zero-padded copy the materialising arm stages per dispatch.

    Args:
        dst: The `[c_out, c_in * r * s]` destination; every element written.
        src: The `[c_out, c_in, r, s]` row-major checkpoint filter.
        c_out: Filter count, the destination's row count.
        c_in: Input channels; channel-fastest within each destination tap.
        r: Filter height.
        s: Filter width.
    """
    var k = c_in * r * s
    for n in range(c_out):
        for cc in range(c_in):
            for rr in range(r):
                for ss in range(s):
                    dst[unsafe_offset=n * k + (rr * s + ss) * c_in + cc] = src[
                        unsafe_offset=((n * c_in + cc) * r + rr) * s + ss
                    ]


def _upsample_fold_dispatch[
    dtype: DType,
    out_origin: MutOrigin,
    in_origin: MutOrigin,
    f_origin: MutOrigin,
    //,
](
    out_base: Pointer[Scalar[dtype], out_origin],
    in_ptr: Pointer[Scalar[dtype], in_origin],
    f_src: Pointer[Scalar[dtype], f_origin],
    batch: Int,
    h_in: Int,
    w_in: Int,
    c_in: Int,
    c_out: Int,
    r: Int,
    s: Int,
    pad_top: Int,
    pad_left: Int,
    h_out: Int,
    w_out: Int,
    m: Int,
    k: Int,
    ctx: DeviceContext,
) raises -> Bool:
    """Dispatches one recognized upsample-fed conv as selector 45 with
    `upsample_log2 = 1`, and reports whether it did; False keeps today's
    route untouched.

    The admission is deliberately the CONJUNCTION of the recognition and
    the arm that implements the fold: a PINNED recognition whose extents
    agree with this conv's operands, the fused-conv contract
    (`C_in % 64 == 0`), the serve asking for the array
    (`MODULAR_NPU_CONV_HMX`), and `matmul_bt_hmx_accepts` in the fold's own
    coordinates (m, k are unchanged by it). Anything short of all four
    declines here, because the kernel refuses a fold it cannot run on the
    array (slot 7 code 8) rather than degrading -- and a declined dispatch
    would fail the correctness guard with the output absent. Since the graph
    composite became the production fold, an admitted recognition ALSO
    demands `MODULAR_NPU_UPSAMPLE_FOLD_DIAG=1` or it logs and stands down
    (`_upsample_fold_diag_requested`).

    THE SMALL MAP IS STAGED FROM THE CONV'S OWN LIVE INPUT, not from any
    recorded pointer: the input IS the materialized 2x map (the walks all
    still ran; the knob buys certification, the pool win lands with the
    graph composite -- `UpsampleConv` below), so its even rows and columns
    ARE the pre-upsample map, bit-for-bit. Nothing is ever read through a
    pointer that could have died -- `_up_trace`'s contract. Filter staging
    and the launch itself are `_launch_upsample_conv_ep`, shared with the
    composite's registration.
    """
    if not _upsample_fold_requested():
        return False
    if not _conv_hmx_requested():
        return False
    if not matmul_bt_hmx_accepts(m, c_out, k, 0, c_out):
        return False
    var trace = _up_trace(ctx)
    var pin = _up_take(trace, _UP_KIND_PINNED, Int(in_ptr), True)
    if not pin[0]:
        return False
    var h_small = pin[3]
    var w_small = pin[4]
    if (
        pin[1] != batch
        or pin[2] != c_in
        or 2 * h_small != h_in
        or 2 * w_small != w_in
    ):
        return False
    # Post-recognition so a production serve (fold knob armed for the
    # composite) never logs on the convs the graph already folded.
    if not _upsample_fold_diag_requested():
        print(
            "npu upsample fold: eager arm recognized this conv but stands"
            " down -- the graph composite (npu_upsample_conv) is the"
            " production fold; set MODULAR_NPU_UPSAMPLE_FOLD_DIAG=1 to"
            " dispatch the eager arm for diagnosis"
        )
        return False

    # The pre-upsample map, subsampled out of this op's own dense input:
    # rows 2h, columns 2w, all C channels per pick -- one C-length run each.
    var small = ctx.enqueue_create_buffer[dtype](
        batch * h_small * w_small * c_in
    )
    var small_ptr = small.unsafe_ptr()
    for b in range(batch):
        var src_plane = b * h_in * w_in * c_in
        var dst_plane = b * h_small * w_small * c_in
        for h in range(h_small):
            var src_row = src_plane + (2 * h) * w_in * c_in
            var dst_row = dst_plane + h * w_small * c_in
            for w in range(w_small):
                unsafe_memcpy(
                    dest=small_ptr.unsafe_offset(dst_row + w * c_in),
                    src=in_ptr.unsafe_offset(src_row + (2 * w) * c_in),
                    count=c_in,
                )

    _launch_upsample_conv_ep(
        out_base,
        small_ptr,
        f_src,
        batch,
        h_small,
        w_small,
        c_in,
        c_out,
        r,
        s,
        pad_top,
        pad_left,
        h_out,
        w_out,
        m,
        k,
        "mo.conv upsample fold",
        ctx,
    )
    _ = small^
    return True


def _launch_upsample_conv_ep[
    dtype: DType,
    out_origin: MutOrigin,
    in_origin: MutOrigin,
    f_origin: MutOrigin,
    //,
](
    out_base: Pointer[Scalar[dtype], out_origin],
    small_ptr: Pointer[Scalar[dtype], in_origin],
    f_src: Pointer[Scalar[dtype], f_origin],
    batch: Int,
    h_small: Int,
    w_small: Int,
    c_in: Int,
    c_out: Int,
    r: Int,
    s: Int,
    pad_top: Int,
    pad_left: Int,
    h_out: Int,
    w_out: Int,
    m: Int,
    k: Int,
    op: StaticString,
    ctx: DeviceContext,
) raises:
    """Stages the filter artefact and dispatches selector 45 with
    `upsample_log2 = 1` over an already-small `[batch, h_small, w_small,
    c_in]` map, one launch per batch index.

    The tail both fold arms share: the eager recognizer hands it the map it
    subsampled out of the conv's materialized input, and the graph
    composite's registration hands it the conv's own live input, which IS
    the pre-upsample map when the composite staged. Admission is the
    caller's job -- everything here assumes the shape was accepted.

    The filter stages exactly as the selector-40 route stages it -- fused-k
    permute, croutons when the skel is array-native -- plus the ep row's
    bias tail, ZERO-FILLED: this graph's bias lands in a separate `mo.add`
    the fold does not claim, and a zero row through the fp16 widen-add is
    the identity (the sel-41 rule: unconditional, never a branch).
    """
    var artefact = ctx.enqueue_create_buffer[dtype](c_out * k + c_out)
    var a_dst = artefact.unsafe_ptr()
    if _stage_native_weight(m, c_out, k):
        var permuted = unsafe_alloc[Scalar[dtype]](c_out * k)
        _permute_filter_fused_k(permuted, f_src, c_out, c_in, r, s)
        _crouton_pack_rows(a_dst, permuted, c_out, k)
        permuted.unsafe_free()
    else:
        _permute_filter_fused_k(a_dst, f_src, c_out, c_in, r, s)
    for j in range(c_out):
        a_dst[unsafe_offset=c_out * k + j] = Scalar[dtype](0)

    # The row's residual and gn_ab buffers, bound and unread on this arm
    # (resid_present = 0; the phase-ii fold has not landed). Two distinct
    # allocations rather than one aliased pointer: the wire binds them as
    # two buffers, and the dispatch's own argument checking says so.
    var inert_r = ctx.enqueue_create_buffer[dtype](HMX_TILE)
    var inert_r_ptr = inert_r.unsafe_ptr()
    var inert_g = ctx.enqueue_create_buffer[dtype](HMX_TILE)
    var inert_g_ptr = inert_g.unsafe_ptr()
    for j in range(HMX_TILE):
        inert_r_ptr[unsafe_offset=j] = Scalar[dtype](0)
        inert_g_ptr[unsafe_offset=j] = Scalar[dtype](0)

    var f = ctx.load_function[npu_selector_sig](
        function_name=String("conv_im2col_hmx_ep"), asm=String("")
    )
    for b in range(batch):
        var witness_ptr = _witness_slot(ctx)
        _arm_matmul_ran(witness_ptr, op)
        _arm_unpack_arm_known(witness_ptr, op)
        ctx.enqueue_function(
            f,
            out_base.unsafe_offset(b * m * c_out),
            small_ptr.unsafe_offset(b * h_small * w_small * c_in),
            a_dst,
            witness_ptr,
            inert_r_ptr,
            inert_g_ptr,
            Int32(h_small),
            Int32(w_small),
            Int32(c_in),
            Int32(c_out),
            Int32(r),
            Int32(s),
            Int32(pad_top),
            Int32(pad_left),
            Int32(h_out),
            Int32(w_out),
            Int32(0),
            Int32(1),
            grid_dim=Dim(NPU_GRID_CORES),
            block_dim=Dim(NPU_THREAD_MASK),
        )
    _ = artefact^
    _ = inert_r^
    _ = inert_g^


@extensibility.register("npu_upsample_conv", type="npu")
struct UpsampleConv:
    """Claims the `npu_upsample_conv` custom op the VAE authoring folds a
    recognized `interpolate_2d_nearest -> conv` pair into
    (`vae_flux2._maybe_fold_npu_upsample_conv`, behind
    `MODULAR_NPU_UPSAMPLE_FOLD=1`), and dispatches signed selector 45 with
    `upsample_log2 = 1` over the pre-upsample map.

    The graph half of the phase-iii fold: the eager recognizer under the
    same knob proves and certifies the kernel arm but runs AFTER the
    upsample's materialization walks already executed, because the op
    stream cannot see consumers. The composite is staged only where the
    graph could prove them, so here the input IS the small map and the
    walks never staged. Stride and dilation are structurally 1: the
    recognizer only folds the stride-1 wrap `Conv2d(permute=True)` authors,
    so the op carries no stride operand at all.

    `api` is deliberately a wildcard, `RMSNormRope`'s window-285 rule: an
    op minted by `ops.custom` at graph build is checked against device info
    the compile does not hold yet, so an `api="npu"` pin can never match
    there. Pinning `type` keeps cpu/gpu targets refused, and the name is
    unique across the plugin packages.

    Everything the eager arm declines on is a RAISE here, not a fallback:
    once the composite staged, the materialization ops are gone, so a
    dispatch that cannot run the array has nothing to degrade to -- a
    mint/serve knob skew or an ineligible shape must fail loudly at the
    first dispatch rather than compute anything else.
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        //,
        pad_top: Int,
        pad_bottom: Int,
        pad_left: Int,
        pad_right: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        filter: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        comptime assert dtype == DType.float16, (
            "npu_upsample_conv on the on-device NPU: fp16 only -- selector"
            " 45's upsample arm exists for the production f16 row"
        )
        comptime assert rank == 4, (
            "npu_upsample_conv on the on-device NPU: NHWC map and FCRS"
            " filter are rank 4"
        )

        if not _upsample_fold_requested():
            raise Error(
                "npu_upsample_conv: this MEF was staged with"
                " MODULAR_NPU_UPSAMPLE_FOLD=1 but the serve does not arm"
                " it; the fold's staging and its dispatch are one knob"
            )
        if not _conv_hmx_requested():
            raise Error(
                "npu_upsample_conv: selector 45 runs on the array only;"
                " arm MODULAR_NPU_CONV_HMX=1 -- the composite has no HVX"
                " fallback by construction"
            )

        _require_dense(
            output.shape(), output.strides(), "npu_upsample_conv", "the output"
        )
        _require_dense(
            input.shape(), input.strides(), "npu_upsample_conv", "the input"
        )
        _require_dense(
            filter.shape(), filter.strides(), "npu_upsample_conv", "the filter"
        )

        var batch = Int(input.dim_size(0))
        var h_small = Int(input.dim_size(1))
        var w_small = Int(input.dim_size(2))
        var c_in = Int(input.dim_size(3))
        var c_out = Int(filter.dim_size(0))
        var r = Int(filter.dim_size(2))
        var s = Int(filter.dim_size(3))
        var h_out = Int(output.dim_size(1))
        var w_out = Int(output.dim_size(2))

        if Int(filter.dim_size(1)) != c_in:
            raise Error(
                "npu_upsample_conv: the filter's C extent must be the"
                " input's channel count"
            )
        if (
            Int(output.dim_size(0)) != batch
            or Int(output.dim_size(3)) != c_out
            or h_out != 2 * h_small + pad_top + pad_bottom - (r - 1)
            or w_out != 2 * w_small + pad_left + pad_right - (s - 1)
        ):
            raise Error(
                "npu_upsample_conv: output extents do not match a stride-1"
                " convolution of the 2x-upsampled input"
            )

        var m = h_out * w_out
        var k = c_in * r * s
        if not _fused_conv_supported(c_in):
            raise Error(
                (
                    "npu_upsample_conv: the fused gather needs C_in % 64 == 0;"
                    " the staging recognizer should never have folded C_in "
                ),
                c_in,
            )
        if not matmul_bt_hmx_accepts(m, c_out, k, 0, c_out):
            raise Error(
                "npu_upsample_conv: the array refuses this shape"
                " (matmul_bt_hmx_accepts); the staging recognizer and this"
                " dispatch disagree on the admission"
            )

        _launch_upsample_conv_ep(
            output.unsafe_ptr(),
            input.unsafe_ptr(),
            filter.unsafe_ptr(),
            batch,
            h_small,
            w_small,
            c_in,
            c_out,
            r,
            s,
            pad_top,
            pad_left,
            h_out,
            w_out,
            m,
            k,
            "npu_upsample_conv",
            ctx,
        )


@extensibility.register("mo.conv", type="npu", api="npu")
struct Conv:
    """Serves a stride-1 convolution as a host im2col feeding the signed
    matmul, the P1 design's route (`VAE_DECODER.md`; conv is not new math on
    this port, it is the b-transposed matmul plus a gather).

    The 1x1 convolutions never arrive: `ConvToMatmulPattern` rewrites them to
    bare `mo.matmul` ahead of resolution. What lands here is the decoder's
    3x3 stride-1 stack, whose generic builtin binding is generated device
    code -- the compile survives elaboration and then asks `compileOffloads`
    for the bare-metal triple no engine build carries, the same wall the
    denoise graph's buffer concat hit.

    The mapping is MAX's own GPU lowering with the filter's own layout as the
    contraction order: `M = H_out * W_out` output pixels per batch,
    `K = C_in * R * S` in FCRS order so the `[F, C, R, S]` filter IS the
    selector's `[N, K]` b-transposed operand with no transpose, and the
    im2col gather zero-fills the padding taps on the host, per batch index,
    into a scratch the dispatch reads. The selector requires K to be a
    multiple of 64; a narrow conv (the decoder's `conv_in`, K = 288) is
    zero-padded on K -- zero taps against zero filter rows add exactly
    nothing -- with the filter staged into padded rows.

    Every convolution whose `C_in` is a multiple of 64 now takes the FUSED
    arm instead: `conv_im2col` gathers A from the NHWC input map inside the
    kernel, so no im2col scratch is allocated, filled or transported. The
    materialising path below survives only for `conv_in` (K = 288, C_in = 32),
    which is 0.05% of the decode's MACs and the one shape the contiguous gather
    is not defined for.

    TODO(skye): the fused arm still permutes the FCRS filter into the gather's
    `(r, s, c)` k order per dispatch, and the selector-40 route on an
    array-native serve adds the crouton repack as a second same-size pass.
    Order 318 measured the recorder's per-launch prep at 0.02-0.04 ms/call on
    BOTH selectors -- four orders of magnitude under a `C_out * K` host pass
    -- so the staging is not being paid per dispatch as written and a
    weight-load hoist has no measurable prize; where the runtime actually
    absorbs it is not established. Re-price before hoisting anything.
    """

    @staticmethod
    def execute[
        input_layout: StaticString,
        filter_layout: StaticString,
        has_epilogue_fusion: Bool,
        target: StaticString,
        _trace_name: StaticString,
    ](
        output: _FusedOutputTensor[...],
        input: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        filter: InputTensor[dtype=output.dtype, ...],
        strides: InputTensor,
        dilation: InputTensor,
        paddings: InputTensor,
        num_groups: Scalar,
        ctx: DeviceContext,
    ) capturing raises:
        # A held matmul-chain member must run before this op touches any
        # operand: this is the 186A host-reader class, closed per site.
        _matmul_chain_flush()
        comptime assert (
            output.dtype == DType.float16 and input.dtype == DType.float16
        ), "mo.conv on the on-device NPU: fp16 only"
        comptime assert not has_epilogue_fusion, (
            "mo.conv on the on-device NPU: an epilogue cannot be fused into a"
            " pre-signed kernel"
        )
        comptime assert (
            input_layout == "NHWC" and filter_layout == "FCRS"
        ), "mo.conv on the on-device NPU: NHWC activations, FCRS filter"
        comptime assert (
            input.rank == 4 and filter.rank == 4
        ), "mo.conv on the on-device NPU: 2D convolutions only"

        if Int(num_groups) != 1:
            raise Error(
                (
                    "mo.conv on the on-device NPU: grouped convolution has no"
                    " lowering here; got groups "
                ),
                Int(num_groups),
            )
        for i in range(Int(strides.size())):
            if (
                Int(strides._ptr[unsafe_offset=i]) != 1
                or Int(dilation._ptr[unsafe_offset=i]) != 1
            ):
                raise Error(
                    "mo.conv on the on-device NPU: stride and dilation must"
                    " be 1 (the decoder is stride-1 by design)"
                )

        _require_dense(
            output.shape(), output.strides(), "mo.conv", "the output"
        )
        _require_dense(input.shape(), input.strides(), "mo.conv", "the input")
        _require_dense(
            filter.shape(), filter.strides(), "mo.conv", "the filter"
        )

        var batch = Int(input.dim_size(0))
        var h_in = Int(input.dim_size(1))
        var w_in = Int(input.dim_size(2))
        var c_in = Int(input.dim_size(3))
        var c_out = Int(filter.dim_size(0))
        var r = Int(filter.dim_size(2))
        var s = Int(filter.dim_size(3))
        var pad_top = Int(paddings._ptr[unsafe_offset=0])
        var pad_left = Int(paddings._ptr[unsafe_offset=2])
        var h_out = Int(output.dim_size(1))
        var w_out = Int(output.dim_size(2))

        if Int(filter.dim_size(1)) != c_in:
            raise Error(
                "mo.conv on the on-device NPU: the filter's C extent must be"
                " the input's channel count"
            )
        if (
            Int(output.dim_size(0)) != batch
            or Int(output.dim_size(3)) != c_out
            or h_out
            != h_in + pad_top + Int(paddings._ptr[unsafe_offset=1]) - (r - 1)
            or w_out
            != w_in + pad_left + Int(paddings._ptr[unsafe_offset=3]) - (s - 1)
        ):
            raise Error(
                "mo.conv on the on-device NPU: output extents do not match a"
                " stride-1 convolution of these operands"
            )

        var m = h_out * w_out
        var k = c_in * r * s

        # The FUSED arm: no im2col matrix at all. The `conv_im2col` selector
        # gathers each A vector out of the NHWC input map inside the kernel, so
        # the `M x K_pad` scratch below — 1.208 GB for `up2.resize_conv` at
        # 512x512, and over the transport's 2 GiB single-allocation cap at
        # 1024x1024 — is never allocated, never filled by a host loop, and never
        # crossed. `C_in % 64 == 0` is what makes each gathered vector one
        # contiguous 128-byte load; the decoder's `conv_in` (C_in = 32, 0.05% of
        # the decode's MACs) is the one convolution that falls through to the
        # materialising path below.
        #
        # The filter needs the fused arm's k order, `(r*S + s)*C_in + c` —
        # channel FASTEST — where the FCRS checkpoint is channel-slowest, so it
        # is permuted here. That permutation is `C_out * K` elements, the same
        # size as the zero-padded copy the materialising arm already stages per
        # dispatch, and belongs at weight load once phase 2 owns the weights.
        if _fused_conv_supported(c_in):
            # The upsample fold first: a conv whose input was recognized as
            # the reshape x broadcast-ones nearest-2x (design phase iii)
            # dispatches selector 45 with `upsample_log2 = 1` over the small
            # map, when the serve asked (`MODULAR_NPU_UPSAMPLE_FOLD=1`) and
            # the array admits the shape. Everything it declines continues
            # below exactly as before.
            if _upsample_fold_dispatch(
                output.unsafe_ptr(),
                input.unsafe_ptr(),
                filter.unsafe_ptr(),
                batch,
                h_in,
                w_in,
                c_in,
                c_out,
                r,
                s,
                pad_top,
                pad_left,
                h_out,
                w_out,
                m,
                k,
                ctx,
            ):
                return
            # Selector 40 is dispatched only for shapes the array will RUN.
            # The kernel's own gate is `geom.usable() and
            # matmul_bt_hmx_accepts` (`npu_kernel.mojo:3495`; `usable` is
            # vacuous on this path -- `c_in % 64 == 0` implies the channel
            # bound and every other refusing extent zeroes the payload), and
            # a shape it declines falls to the HVX nest INSIDE the kernel --
            # where witness word 7 still holds the launch's fifth thread's
            # `_qurt_thread_id()` from the conv entries' UNBOUNDED stamp
            # (`npu_kernel.mojo:3486`; the matmul entry bounds the same stamp
            # to compute threads for exactly this collision, `:3207-3216`,
            # P5 M2.4). A declined dispatch therefore cannot present a
            # readable witness on the signed skels, so the decline happens
            # HERE, at routing (order 318: `conv_out`, c_out = 3, left word 7
            # = 0x1081AA7 / 0x1901AA7 across the two skels -- a thread id,
            # not a code), and a declined conv keeps selector 36, today's
            # exact path. Bounding the conv entries' stamp like matmul's is
            # queued for the next re-sign window.
            var conv_hmx = _conv_hmx_requested() and matmul_bt_hmx_accepts(
                m, c_out, k, 0, c_out
            )
            var filter_rsc = ctx.enqueue_create_buffer[output.dtype](c_out * k)
            var f_dst = filter_rsc.unsafe_ptr()
            var f_src = filter.unsafe_ptr()
            # Selector 40 reads B per the SKEL's weight layout with no layout
            # argument on the wire (mismatch class #366), so a routed
            # dispatch stages croutons exactly when the serve declares the
            # array-native skel: `_stage_native_weight` composes that
            # declaration with the same `matmul_bt_hmx_accepts` the routing
            # above and the kernel both test -- one predicate, three readers,
            # no seam for disagreement.
            if conv_hmx and _stage_native_weight(m, c_out, k):
                var permuted = unsafe_alloc[Scalar[output.dtype]](c_out * k)
                _permute_filter_fused_k(permuted, f_src, c_out, c_in, r, s)
                _crouton_pack_rows(f_dst, permuted, c_out, k)
                permuted.unsafe_free()
            else:
                _permute_filter_fused_k(f_dst, f_src, c_out, c_in, r, s)

            # `conv_im2col_hmx` (selector 40) takes the IDENTICAL buffers and
            # scalars and calls `_matmul_stripe_hmx`, inheriting every code
            # `_arm_matmul_ran` exists to catch -- family-2 refusals written
            # beside a successful return (`WITNESS_GUARD_CLASSIFICATION.md`
            # row 15's latent case). So the array selector is dispatched only
            # with the correctness guard armed per launch, exactly as the
            # materialising arm below arms its matmul. Selector 36 keeps the
            # bare slot: the pure-HVX body has no refusal to miss, and a
            # guard that cannot fail is not a control.
            var selector = String("conv_im2col")
            if conv_hmx:
                selector = String("conv_im2col_hmx")
            var conv_f = ctx.load_function[npu_selector_sig](
                function_name=selector, asm=String("")
            )
            var in_base = input.unsafe_ptr()
            var out_base = output.unsafe_ptr()
            for b in range(batch):
                var witness_ptr = _witness_slot(ctx)
                if conv_hmx:
                    _arm_matmul_ran(witness_ptr, "mo.conv")
                    _arm_unpack_arm_known(witness_ptr, "mo.conv")
                ctx.enqueue_function(
                    conv_f,
                    out_base.unsafe_offset(b * m * c_out),
                    in_base.unsafe_offset(b * h_in * w_in * c_in),
                    f_dst,
                    witness_ptr,
                    Int32(h_in),
                    Int32(w_in),
                    Int32(c_in),
                    Int32(c_out),
                    Int32(r),
                    Int32(s),
                    Int32(pad_top),
                    Int32(pad_left),
                    Int32(h_out),
                    Int32(w_out),
                    grid_dim=Dim(NPU_GRID_CORES),
                    block_dim=Dim(NPU_THREAD_MASK),
                )
            _ = filter_rsc^
            return

        # The selector reads K in 64-element vectors; zero taps against zero
        # filter rows contribute exactly nothing, so padding K is the
        # identity on the result.
        var k_pad = ((k + 63) // 64) * 64

        var scratch = ctx.enqueue_create_buffer[output.dtype](m * k_pad)
        var a = scratch.unsafe_ptr()

        # The filter is staged once per dispatch rather than passed through:
        # when K needs padding the copy is load-bearing (zero rows under zero
        # taps), and when it does not, one buffer type serves both arms.
        # TODO(skye): pass the checkpoint filter pointer straight through for
        # the aligned case once phase 2 measures the copy; it is ~5 MB per
        # dispatch at the widest stage.
        var padded_filter = ctx.enqueue_create_buffer[output.dtype](
            c_out * k_pad
        )
        var weights = padded_filter.unsafe_ptr()
        # The FCRS filter's own flatten IS the im2col's contraction order, so
        # row-per-row staging with a zero K tail is the whole adaptation --
        # plus the crouton repack on an array-native serve, which this arm
        # owes the selector exactly as the k-pad matmul path does.
        _stage_weight_rows(weights, filter.unsafe_ptr(), c_out, k, k_pad, m)

        var f = ctx.load_function[npu_selector_sig](
            function_name=String("matmul"), asm=String("")
        )

        var src = input.unsafe_ptr()
        var dst_base = output.unsafe_ptr()
        for b in range(batch):
            var plane = src.unsafe_offset(b * h_in * w_in * c_in)
            for y in range(h_out):
                for x in range(w_out):
                    var row = a.unsafe_offset((y * w_out + x) * k_pad)
                    for c in range(c_in):
                        for rr in range(r):
                            var yy = y + rr - pad_top
                            for ss in range(s):
                                var xx = x + ss - pad_left
                                var tap = Scalar[output.dtype](0)
                                if (
                                    yy >= 0
                                    and yy < h_in
                                    and xx >= 0
                                    and xx < w_in
                                ):
                                    tap = plane[
                                        unsafe_offset=(yy * w_in + xx) * c_in
                                        + c
                                    ]
                                row[unsafe_offset=(c * r + rr) * s + ss] = tap
                    for j in range(k, k_pad):
                        row[unsafe_offset=j] = 0
            # One `a` scratch still serves every batch index in sequence, on the
            # rationale batching removes: an invocation on this transport
            # completes before it returns. The witness no longer relies on that
            # (a slot per launch, below), but the im2col scratch does, so
            # BATCHING THIS ARM NEEDS A SCRATCH PER BATCH INDEX FIRST -- held
            # launches would all read the last fill. Filed rather than fixed
            # here: it is a buffer-lifetime change, not a witness one.
            var witness_ptr = _witness_slot(ctx)
            _arm_matmul_ran(witness_ptr, "mo.conv")
            ctx.enqueue_function(
                f,
                dst_base.unsafe_offset(b * m * c_out),
                a,
                weights,
                witness_ptr,
                Int32(m),
                Int32(c_out),
                Int32(k_pad),
                grid_dim=Dim(NPU_GRID_CORES),
                block_dim=Dim(NPU_THREAD_MASK),
            )
        _ = scratch^
        _ = padded_filter^


def _require_sincos_domain(largest_angle: Float64, op: StaticString) raises:
    """Refuses an angle past the table kernels' reduction bound.

    `sinusoid_embed` and `rope_table` are correctly rounded for |theta| <=
    pi/4 * 2^20 (`SINCOS_MAX_ARG` in `sincos_tables.mojo`); past it the
    Cody-Waite high product stops being exact and the error grows silently.
    Both operands are host-visible memory on this transport, so the largest
    angle a dispatch can produce is checkable here, and a loud refusal beats
    a plausible wrong table. FLUX sits three orders of magnitude inside the
    bound (positions < 4608, frequencies <= 1, timesteps in [0, 1] scaled by
    1000).
    """
    comptime SINCOS_MAX_ARG = Float64(823549.0)
    if largest_angle > SINCOS_MAX_ARG:
        raise Error(
            op,
            (
                " on the on-device NPU: the largest angle this dispatch"
                " could produce is "
            ),
            largest_angle,
            ", past the kernel's correctly-rounded reduction bound of ",
            SINCOS_MAX_ARG,
        )


@extensibility.register("mo.composite.sinusoid_embed", type="npu", api="npu")
struct SinusoidEmbed:
    """Dispatches the `sinusoid_embed` selector: the timestep sinusoidal
    embedding table, `[rows]` fp32 timesteps to `[rows, 2*half_dim]` fp16.

    A composite the graph compiler will mint rather than an operator the model
    author writes: the staged form is a chain of generated-code stubs (a
    scaled-timestep fp16 round-trip, an `mo.range`-rooted frequency band, an
    outer product, `mo.sin`/`mo.cos`, a flip-concat and a cast), none of which
    reaches a signed kernel on this target. The whole chain is one table with
    one name.

    `time_scale` and `neg_ln_max_period` arrive as the graph's own folded
    constants -- the scale multiplying the timestep, and the constant
    multiplying the band's `range` (the reference's `-log(max_period)`,
    already folded, which is also why no log is computed here or on the
    device). `flip` puts the cosines first, `flip_sin_to_cos=True` in the
    reference and FLUX's setting.
    """

    @staticmethod
    def execute[
        flip: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=DType.float16, rank=2, ...],
        t: InputTensor[dtype=DType.float32, rank=1, ...],
        time_scale: Float32,
        neg_ln_max_period: Float32,
        ctx: DeviceContext,
    ) capturing raises:
        # A held matmul-chain member must run before this op touches any
        # operand: this is the 186A host-reader class, closed per site.
        _matmul_chain_flush()
        var num_rows = Int(output.dim_size(0))
        var cols = Int(output.dim_size(1))
        if Int(t.dim_size(0)) != num_rows:
            raise Error(
                (
                    "mo.composite.sinusoid_embed on the on-device NPU: one"
                    " output row per timestep, got "
                ),
                num_rows,
                " rows against ",
                Int(t.dim_size(0)),
                " timesteps",
            )
        if cols % 2 != 0 or cols == 0:
            raise Error(
                (
                    "mo.composite.sinusoid_embed on the on-device NPU: the"
                    " output holds a cos half and a sin half, so its width"
                    " must be a positive even number, got "
                ),
                cols,
            )
        _require_dense(
            output.shape(),
            output.strides(),
            "mo.composite.sinusoid_embed",
            "the output",
        )
        _require_dense(
            t.shape(), t.strides(), "mo.composite.sinusoid_embed", "t"
        )

        # The largest angle is max|t| * time_scale (the band peaks at 1).
        # The timesteps are host-visible, so the bound is checked on the
        # values rather than assumed.
        var t_ptr = t.unsafe_ptr()
        var largest = Float64(0)
        for b in range(num_rows):
            var a = Float64(t_ptr[unsafe_offset=b]) * Float64(time_scale)
            if a < 0:
                a = -a
            if a > largest:
                largest = a
        _require_sincos_domain(largest, "mo.composite.sinusoid_embed")

        var f = ctx.load_function[npu_selector_sig](
            function_name=String("sinusoid_embed"), asm=String("")
        )
        ctx.enqueue_function(
            f,
            output.unsafe_ptr(),
            t_ptr,
            _witness_slot(ctx),
            Int32(num_rows),
            Int32(cols // 2),
            time_scale,
            neg_ln_max_period,
            Int32(1) if flip != 0 else Int32(0),
            grid_dim=Dim(NPU_GRID_CORES),
            block_dim=Dim(NPU_THREAD_MASK),
        )


@extensibility.register("mo.composite.rope_table", type="npu", api="npu")
struct RopeTable:
    """Dispatches the `rope_table` selector: one rope axis's fp32 (cos, sin)
    table in the interleaved pair layout the signed rope kernels read.

    A composite the graph compiler will mint, as `SinusoidEmbed` is: the
    staged form is a cast of the packed id tensor, a per-axis column slice,
    an outer product against a constant-folded frequency row, and
    `mo.sin`/`mo.cos` stubs, none of which reaches a signed kernel here.

    `positions` is the axis's si64 id column and is the one operand this
    package accepts as a strided view: the selector takes the row stride as
    a scalar, so the column is read straight out of the packed
    `[rows, num_axes]` id tensor with no cast kernel and no slice
    materialized -- which is the point of the composite.
    """

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor[dtype=DType.float32, rank=2, ...],
        positions: InputTensor[dtype=DType.int64, rank=1, ...],
        freqs: InputTensor[dtype=DType.float32, rank=1, ...],
        ctx: DeviceContext,
    ) capturing raises:
        # A held matmul-chain member must run before this op touches any
        # operand: this is the 186A host-reader class, closed per site.
        _matmul_chain_flush()
        var num_rows = Int(output.dim_size(0))
        var cols = Int(output.dim_size(1))
        var half_dim = Int(freqs.dim_size(0))
        if Int(positions.dim_size(0)) != num_rows:
            raise Error(
                (
                    "mo.composite.rope_table on the on-device NPU: one output"
                    " row per position, got "
                ),
                num_rows,
                " rows against ",
                Int(positions.dim_size(0)),
                " positions",
            )
        if cols != 2 * half_dim:
            raise Error(
                (
                    "mo.composite.rope_table on the on-device NPU: the output"
                    " holds one (cos, sin) pair per frequency, so its width"
                    " must be twice the frequency count; got "
                ),
                cols,
                " against ",
                half_dim,
                " frequencies",
            )
        _require_dense(
            output.shape(),
            output.strides(),
            "mo.composite.rope_table",
            "the output",
        )
        _require_dense(
            freqs.shape(),
            freqs.strides(),
            "mo.composite.rope_table",
            "the frequencies",
        )

        # A negative stride would walk the ids backwards out of the
        # allocation; zero (a broadcast row) and any positive pitch are what
        # the selector's pointer arithmetic means.
        var pos_stride = positions.strides()[0]
        if pos_stride < 0:
            raise Error(
                (
                    "mo.composite.rope_table on the on-device NPU: the"
                    " position column's stride must be non-negative, got "
                ),
                pos_stride,
            )

        # Both operands are host-visible, so the angle bound is checked on
        # the values rather than assumed.
        var pos_ptr = positions.unsafe_ptr()
        var freqs_ptr = freqs.unsafe_ptr()
        var largest_pos = Float64(0)
        for s in range(num_rows):
            var p = Float64(pos_ptr[unsafe_offset=s * pos_stride])
            if p < 0:
                p = -p
            if p > largest_pos:
                largest_pos = p
        var largest_freq = Float64(0)
        for k in range(half_dim):
            var q = Float64(freqs_ptr[unsafe_offset=k])
            if q < 0:
                q = -q
            if q > largest_freq:
                largest_freq = q
        _require_sincos_domain(
            largest_pos * largest_freq, "mo.composite.rope_table"
        )

        var f = ctx.load_function[npu_selector_sig](
            function_name=String("rope_table"), asm=String("")
        )
        ctx.enqueue_function(
            f,
            output.unsafe_ptr(),
            pos_ptr,
            freqs_ptr,
            _witness_slot(ctx),
            Int32(num_rows),
            Int32(half_dim),
            Int32(pos_stride),
            grid_dim=Dim(NPU_GRID_CORES),
            block_dim=Dim(NPU_THREAD_MASK),
        )
