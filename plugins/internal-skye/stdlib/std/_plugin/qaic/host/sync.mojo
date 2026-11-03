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
"""Native thread-group and cross-NSP barriers for QAIC kernels.

These are Mojo native counting barriers built on atomics,
a global syncpoint, and the Hexagon `syncht` / `pause` instructions.

Two rules govern every barrier below:

1. **Exactly the threads named by the mask may call it.** A thread outside the
   mask that calls in corrupts the count; a thread inside the mask that skips
   the call deadlocks the rest.

2. **Distinct participant sets need distinct syncpoints.** Syncpoint identity
   is the `name` parameter — two barriers sharing a name share an address, so
   two different masks under one name will deadlock.
"""

from std.atomic import Atomic
from std.bit import pop_count
from std.collections.string.string_span import _get_kgen_string
from std.memory import UnsafePointer
from std.sys import inlined_assembly
from std.sys.info import _accelerator_arch

from . import qshim
from .target import (
    dma_thread_id,
    hmx_and_dma_mask,
    hmx_thread_id,
    hvx_and_dma_mask,
    hvx_and_hmx_mask,
    hvx_mask,
    is_qaic,
    npu_num_cores,
    nsp_mask,
    thread_mask,
)


# ===-----------------------------------------------------------------------===#
# Low level Hexagon helpers
# ===-----------------------------------------------------------------------===#


@always_inline
def _syncht[target_arch: StaticString = _accelerator_arch()]():
    """Issues a Hexagon `syncht` (synchronize transactions) instruction."""
    comptime assert is_qaic[
        target_arch
    ](), "unknown behavior on non QAIC target"
    inlined_assembly[
        "{syncht;\n}",
        NoneType,
        constraints="~{memory}",
        has_side_effect=True,
    ]()


@always_inline
def _relax[
    target_arch: StaticString = _accelerator_arch(),
    //,
    long_pause: Bool = False,
]():
    """Issues a Hexagon `pause` to reduce power while spinning.

    Parameters:
        long_pause: Use the longer pause, for ms-scale waits.
    """
    comptime assert is_qaic[
        target_arch
    ](), "unknown behavior on non QAIC target"
    comptime if long_pause:
        inlined_assembly[
            "{pause(#0xFF);\n}",
            NoneType,
            constraints="~{memory}",
            has_side_effect=True,
        ]()
        return

    inlined_assembly[
        "{pause(#0x08);\n}",
        NoneType,
        constraints="~{memory}",
        has_side_effect=True,
    ]()


@always_inline
def participates[mask: Int](thread_id: UInt32) -> Bool:
    """Returns True if `thread_id` is named by `mask`.

    Parameters:
        mask: The thread mask to test against.

    Args:
        thread_id: The thread id to test.

    Returns:
        True if bit `thread_id` of `mask` is set.
    """
    return (mask & (1 << Int(thread_id))) != 0


# ===-----------------------------------------------------------------------===#
# Syncpoint allocation
# ===-----------------------------------------------------------------------===#


@always_inline
def create_syncpoint[
    name: StaticString = "_sync_point",
]() -> UnsafePointer[Int32, MutUntrackedOrigin]:
    """Creates a zero-initialized barrier counter in kernel global memory.

    The counter must live in global memory, not on a thread's stack, since
    every participating thread accesses the same address.

    Parameters:
        name: Symbol name for the global allocation. Acts as the unique
            identifier — calls with the same name return the same address, so
            barriers over different participant sets must use different names.

    Returns:
        Pointer to a zeroed `Int32` counter in global memory.
    """
    return {
        _mlir_value = __mlir_op.`pop.global_alloc`[
            name=_get_kgen_string[name, "._aic_syncpoint"](),
            count=Int(1).__mlir_index__(),
            _type=UnsafePointer[Int32, MutUntrackedOrigin]._mlir_type,
            alignment=Int(128).__mlir_index__(),
            initializer=Int32(0),
        ]()
    }


# ===-----------------------------------------------------------------------===#
# Core primitive
# ===-----------------------------------------------------------------------===#


@no_inline
def qaic_sync_threads[
    long_pause: Bool = False
](sync_point: UnsafePointer[Int32, MutUntrackedOrigin], num_threads: Int32,):
    """Synchronizes `num_threads` threads at a shared `sync_point`.

    A sense reversing counting barrier. The counter runs up as threads arrive;
    the last arriver drives it negative to release the waiters, and threads
    then count it back to zero on the way out. That departure phase is what
    makes the syncpoint safely reusable: a thread racing ahead into the next
    barrier instance blocks in step 1 until the previous instance has fully
    drained, so a fast thread cannot corrupt the count of a barrier its peers
    have not yet left.

    Parameters:
        long_pause: Use the longer Hexagon `pause` while spinning.

    Args:
        sync_point: Shared counter from `create_syncpoint`.
        num_threads: Exact number of threads that will arrive. A count that
            exceeds the real number of callers deadlocks.
    """
    # Step 1: wait until the previous barrier instance has fully exited.
    while Atomic[Int32].load(sync_point) < 0:
        _relax[long_pause]()

    # Step 2: arrive. `fetch_add` returns the value *before* the increment.
    var count = Atomic[Int32].fetch_add(sync_point, 1) + 1

    if count == num_threads:
        # Ensure all prior memory operations complete before releasing others.
        _syncht()

        # Step 3a: last arriver flips the counter negative to release waiters.
        _ = Atomic[Int32].fetch_add(sync_point, -2 * num_threads)

    else:
        # Step 3b: wait until the last arriver flips the counter negative.
        while Atomic[Int32].load(sync_point) >= 0:
            _relax[long_pause]()

    # Step 4: departure ack. The thread that restores the counter to 0 readies
    # the barrier for reuse.
    var departure_count = Atomic[Int32].fetch_add(sync_point, 1) + 1
    if departure_count == 0:
        _syncht()


@always_inline
def _sync_threads[
    name: StaticString, mask: Int, long_pause: Bool = False
](thread_id: UInt32):
    """Barrier over exactly the threads named by `mask`.

    Parameters:
        name: Syncpoint symbol name; must be unique per participant set.
        mask: The participating threads.
        long_pause: Use the longer Hexagon `pause` while spinning.

    Args:
        thread_id: The calling thread's id.
    """
    comptime assert mask != 0, "empty sync mask has no participants"
    debug_assert(
        participates[mask](thread_id),
        "_sync_threads: thread does not participate in mask",
    )

    comptime num_threads = Int32(pop_count(mask))

    # A single-participant barrier has nothing to wait for; the ordering
    # guarantee alone is what the caller needs.
    comptime if num_threads == 1:
        _syncht()
        return

    qaic_sync_threads[long_pause](create_syncpoint[name](), num_threads)


# ===-----------------------------------------------------------------------===#
# Thread-group barriers
# ===-----------------------------------------------------------------------===#


@always_inline
def sync_threads[
    target_arch: StaticString = _accelerator_arch(),
    //,
    mask: Int = thread_mask[target_arch](),
    long_pause: Bool = True,
    *,
    name: StaticString = String(mask) + "_sync_all_syncpoint",
](thread_id: UInt32):
    """Barrier over the threads set in the `mask`.

    With no parameters, synchronizes every thread the launch dispatches per
    core. Pass an explicit `name` and `mask` to synchronize a distinct subset;
    the name identifies a syncpoint and must be unique per participant set, or
    barriers over different masks under one name will deadlock.

    Parameters:
        target_arch: Accelerator arch string; defaults to the compile target.
        mask: The exact set of threads calling this barrier.
        long_pause: Use the longer Hexagon `pause` while spinning.
        name: Syncpoint symbol name.

    Args:
        thread_id: The calling thread's id.
    """
    _sync_threads[name, mask, long_pause](thread_id)


@always_inline
def sync_hvx_threads[
    target_arch: StaticString = _accelerator_arch(),
](thread_id: UInt32):
    """HVX-only barrier.

    All HVX threads must call this. The HMX and DMA threads do not participate
    and must not call it.

    Parameters:
        target_arch: Accelerator arch string; defaults to the compile target.

    Args:
        thread_id: The calling thread's id.
    """
    _sync_threads["_sync_hvx_syncpoint", hvx_mask[target_arch](), False](
        thread_id
    )


@always_inline
def sync_hvx_and_hmx_threads[
    target_arch: StaticString = _accelerator_arch(),
](thread_id: UInt32):
    """Barrier over the HVX threads plus the HMX thread.

    Parameters:
        target_arch: Accelerator arch string; defaults to the compile target.

    Args:
        thread_id: The calling thread's id.
    """
    _sync_threads[
        "_sync_hvx_hmx_syncpoint", hvx_and_hmx_mask[target_arch](), False
    ](thread_id)


@always_inline
def sync_hvx_and_dma_threads[
    target_arch: StaticString = _accelerator_arch(),
](thread_id: UInt32):
    """Barrier over the HVX threads plus the DMA thread.

    Parameters:
        target_arch: Accelerator arch string; defaults to the compile target.

    Args:
        thread_id: The calling thread's id.
    """
    _sync_threads[
        "_sync_hvx_dma_syncpoint", hvx_and_dma_mask[target_arch](), True
    ](thread_id)


@always_inline
def sync_hmx_and_dma_threads[
    target_arch: StaticString = _accelerator_arch(),
](thread_id: UInt32):
    """Barrier over the HMX thread plus the DMA thread.

    Parameters:
        target_arch: Accelerator arch string; defaults to the compile target.

    Args:
        thread_id: The calling thread's id.
    """
    _sync_threads[
        "_sync_hmx_dma_syncpoint", hmx_and_dma_mask[target_arch](), True
    ](thread_id)


# ===-----------------------------------------------------------------------===#
# Cross-NSP barriers
# ===-----------------------------------------------------------------------===#
#
# Each is local -> global -> local: close the local stage, have one delegate
# per NSP announce this core to its peers, then release the next stage.


@always_inline
def sync_threads_across_nsps[
    target_arch: StaticString = _accelerator_arch(),
    //,
    mask: Int = thread_mask[target_arch](),
    long_pause: Bool = True,
    *,
    name: StaticString = String(mask) + "_sync_nsps_syncpoint",
](thread_id: UInt32, cores: Int = npu_num_cores()):
    """All-thread barrier across the participating NSP cores.

    Every thread the launch dispatches must call this.

    Parameters:
        target_arch: Accelerator arch string; defaults to the compile target.
        mask: The exact set of threads calling this barrier.
        long_pause: Use the longer Hexagon `pause` while spinning.
        name: Syncpoint symbol name.

    Args:
        thread_id: The calling thread's id.
        cores: Number of participating NSP cores; defaults to this dispatch's
            core count.
    """
    var _nsp_mask = nsp_mask(cores)

    sync_threads[mask, long_pause, name=name](thread_id)
    if thread_id == UInt32(dma_thread_id[target_arch]()):
        _ = qshim.qshim_sync_nsps(UInt32(_nsp_mask))
    sync_threads[mask, long_pause, name=name](thread_id)


@always_inline
def sync_hvx_threads_across_nsps[
    target_arch: StaticString = _accelerator_arch(),
](thread_id: UInt32, cores: Int = npu_num_cores()):
    """HVX-only barrier across the participating NSP cores.

    Parameters:
        target_arch: Accelerator arch string; defaults to the compile target.

    Args:
        thread_id: The calling thread's id.
        cores: Number of participating NSP cores; defaults to this dispatch's
            core count.
    """
    var mask = nsp_mask(cores)

    sync_hvx_threads[target_arch](thread_id)
    if thread_id == 0:
        _ = qshim.qshim_sync_nsps(UInt32(mask))
    sync_hvx_threads[target_arch](thread_id)


@always_inline
def sync_hvx_and_hmx_threads_across_nsps[
    target_arch: StaticString = _accelerator_arch(),
](thread_id: UInt32, cores: Int = npu_num_cores()):
    """HVX + HMX barrier across the participating NSP cores.

    The HMX thread is the delegate: it owns the transfers this barrier fences.

    Parameters:
        target_arch: Accelerator arch string; defaults to the compile target.

    Args:
        thread_id: The calling thread's id.
        cores: Number of participating NSP cores; defaults to this dispatch's
            core count.
    """
    var mask = nsp_mask(cores)

    sync_hvx_and_hmx_threads[target_arch](thread_id)
    if thread_id == UInt32(hmx_thread_id[target_arch]()):
        _ = qshim.qshim_sync_nsps(UInt32(mask))
    sync_hvx_and_hmx_threads[target_arch](thread_id)


@always_inline
def sync_hvx_and_dma_threads_across_nsps[
    target_arch: StaticString = _accelerator_arch(),
](thread_id: UInt32, cores: Int = npu_num_cores()):
    """HVX + DMA barrier across the participating NSP cores.

    The DMA thread is the delegate: it owns the transfers this barrier fences.

    Parameters:
        target_arch: Accelerator arch string; defaults to the compile target.

    Args:
        thread_id: The calling thread's id.
        cores: Number of participating NSP cores; defaults to this dispatch's
            core count.
    """
    var mask = nsp_mask(cores)

    sync_hvx_and_dma_threads[target_arch](thread_id)
    if thread_id == UInt32(dma_thread_id[target_arch]()):
        _ = qshim.qshim_sync_nsps(UInt32(mask))
    sync_hvx_and_dma_threads[target_arch](thread_id)


@always_inline
def sync_hmx_and_dma_threads_across_nsps[
    target_arch: StaticString = _accelerator_arch(),
](thread_id: UInt32, cores: Int = npu_num_cores()):
    """HMX + DMA barrier across the participating NSP cores.

    Parameters:
        target_arch: Accelerator arch string; defaults to the compile target.

    Args:
        thread_id: The calling thread's id.
        cores: Number of participating NSP cores; defaults to this dispatch's
            core count.
    """
    var mask = nsp_mask(cores)

    sync_hmx_and_dma_threads[target_arch](thread_id)
    if thread_id == UInt32(hmx_thread_id[target_arch]()):
        _ = qshim.qshim_sync_nsps(UInt32(mask))
    sync_hmx_and_dma_threads[target_arch](thread_id)
