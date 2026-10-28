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
"""The launch contract every on-device NPU registration shares.

A registration in this package is a host-side dispatch stub, not a kernel. The
kernel already lives in the code-signed device library, so the stub resolves a
selector by name and launches it: `QuRTDeviceContext::loadFunction` matches the
name against `runtime/lib/qurt/NPUKernels.def` and ignores the `asm` argument
entirely. So a kernel this package cannot *name* does not exist for it, and the
ceiling is that static table rather than an inability to produce device code:
compiling a kernel inside a live host process, cataloguing and signing it
unprivileged, and loading it through FastRPC's own privileged loader all work.
What a freshly generated kernel lacks is a row to resolve against, and a library
filename reserved before the session's first open attempt.
"""


# One FastRPC session addresses one NSP core, so a launch can only name that
# core; `QuRTDeviceContext::enqueueFunctionExecDirect` rejects a wider grid.
comptime NPU_GRID_CORES: Int = 1

# The block's X extent is a bitmap of the core's threads, which must set its low
# bits with no gaps -- a participant's launch index is its position in the mask.
# Five threads: four the kernels compute on, plus the part's DMA thread.
comptime NPU_THREAD_MASK: Int = 0x1F

# Words a kernel reports through its trailing witness buffer: the vector width,
# the accumulator width, the participant count, one id per thread, and then the
# per-thread cycle bracket `mojo_matmul_f16` (`npu_kernel.mojo:1978`) and
# `mojo_mha_no_cache_f16` (`:3101`) write UNCONDITIONALLY -- no runtime gate, no
# build flag. One size serves every registration, so it holds the largest
# writer: at `3 + 5` the `mo.matmul`, `mo.batch_matmul`, `mo.conv` and flash
# attention stubs allocated 8 words against the 32 those two kernels write, and
# every graph launch of them put 96 bytes past the allocation. The stub
# allocates it because it is a property of the launch, not a graph value.
comptime NPU_WITNESS_WORDS: Int = 3 + 5 + 4 * 6

# Bytes to ASK the allocator for, and the fix order 091 measured.
#
# The comment above records the 8-words-against-32-written overrun and says it
# was fixed by raising the word count. It was not: the device buffers below are
# allocated through a path that consumes this argument as a BYTE size and drops
# the dtype -- 091 read `asz = 32` on every witness slot of 1612 render launches
# (`{32: 1612}`), against `M*K*2` exactly for the operand slots in the same dump,
# which settles that `asz` is bytes. So the count went 8 -> 32 while the
# ALLOCATION went 8 bytes -> 32 bytes, and the two kernels that write 32 words
# have been running 96 bytes past the end ever since, exactly as written above.
#
# Multiplying here is correct under the semantics 091 measured and SAFE under the
# semantics the API documents (`enqueue_create_buffer`'s `size` is "the number of
# elements"): if the argument is ever honoured as elements this over-allocates to
# 128 words rather than under-allocating, and 32 words still fit. The ambiguity
# itself is a driver question and is filed, not resolved here.
comptime NPU_WITNESS_BYTES: Int = NPU_WITNESS_WORDS * 4

# Which unit computed a `matmul` dispatch, and the only place a refusal is
# reported: an HMX build that accepts a shape and then cannot run it writes
# NOTHING to the output and still returns success (`_matmul_stripe_hmx`,
# `npu_kernel.mojo`). A stub that allocates this buffer and drops it therefore
# hands the graph whatever the allocation held. Reading it is the graph path's
# half of the cold critique's C1.
#
# The codes are the device's (`MATMUL_PATH_*`) and this is a READER of them --
# the witness protocol, its slot positions and its widths belong to the signed
# kernel set and are not a host stub's to move.
comptime NPU_WITNESS_MATMUL_PATH: Int = 7

# What the host seeds the slot with, and so what "the vector nest ran" looks
# like: the kernel writes the slot on its HMX arm only, and the stub cannot tell
# an unwritten slot from a refusal unless it put a known value there first.
comptime NPU_MATMUL_PATH_UNSET: UInt32 = 0

# `MATMUL_PATH_HMX_RAN`, 'HMX1': the matrix array computed the output.
comptime NPU_MATMUL_PATH_HMX_RAN: UInt32 = 0x484D5831

# Which arm of the crouton unpack seam ran (`MATMUL_WIT_UNPACK_ARM`). Written
# only on the HMX arm, which returns before the HVX nest that owns words 8..31,
# so 22 is free there -- see the word's doc comment in `npu_kernel.mojo` before
# reusing it. This side is a READER of the device's `HMX_UNPACK_ARM_*` codes;
# the slot and the widths belong to the signed kernel set.
comptime NPU_WITNESS_UNPACK_ARM: Int = 22

# What the host seeds the slot with. The write is comptime-gated with
# `SKYE_HMX_UNPACK_VEC`, so an unchanged slot means "this skel has no unpack-arm
# instrument" -- a flag-off build or one predating the word -- and NOT a refusal.
# Without a seed the stub cannot tell that from a device answer.
comptime NPU_UNPACK_ARM_UNSET: UInt32 = 0

# The codes the seam can report. Scalar-because-off and scalar-because-REFUSED
# are deliberately distinct: both mean the vector arm did not run, and only the
# second means a build that asked for it measured nothing.
comptime NPU_UNPACK_ARM_SCALAR_OFF: UInt32 = 1
comptime NPU_UNPACK_ARM_VECTOR: UInt32 = 2
comptime NPU_UNPACK_ARM_SCALAR_REFUSED: UInt32 = 3

# PAIRED has no writer until slice 2 lands. It is accepted here first because the
# reader raises on an unknown code during a real render, so a writer that shipped
# ahead of its reader would take down the first matmul of the first render.
comptime NPU_UNPACK_ARM_PAIRED: UInt32 = 4

# NATIVE is the `output_native` readout (annihilation rung 1): the tile went to
# `C` as a crouton, no unpack arm ran. No shipped skel configuration emits it —
# the flag is a bare comptime parameter only the rung-1 microbench sets — but
# the reader crosses first, per PAIRED's rule above.
comptime NPU_UNPACK_ARM_NATIVE: UInt32 = 5

# BIAS / BIAS_RESID are the conv epilogue readouts (`conv_im2col_hmx_ep`, the
# re-sign-window sibling of selector 40): the unpack added the artefact's
# per-channel bias tail, and RESID also accumulated the row's fifth buffer.
# Reader crosses first, per PAIRED's rule above; the writer rides the same
# re-sign window as the row.
comptime NPU_UNPACK_ARM_BIAS: UInt32 = 6
comptime NPU_UNPACK_ARM_BIAS_RESID: UInt32 = 7

# The relayout row's path word (`relayout_kernel.mojo`, selector 47): the same
# word 7 the matmul family reports through, with the row's own two-code
# vocabulary. RAN means the walk covered the declared extents; REFUSED means a
# non-positive extent and NOTHING WRITTEN, so the guard that reads it is
# CORRECTNESS -- an output the kernel never produced must fail the flush, not
# feed the next op. This side is a reader; the codes belong to the signed row.
comptime NPU_WITNESS_RELAYOUT_PATH: Int = 7
comptime NPU_RELAYOUT_PATH_RAN: UInt32 = 1
comptime NPU_RELAYOUT_PATH_REFUSED: UInt32 = 2


# ===----------------------------------------------------------------------=== #
# The witness arena — one process-lifetime allocation, one slot per in-flight
# launch. Layout shared with `QuRTDeviceContext::witnessGuardsOf`, which is the
# only other reader; keep the two in step through the magic words below.
# ===----------------------------------------------------------------------=== #

# WHY THE ARENA EXISTS, so a later reader does not mistake it for tidying.
# `enqueue_create_buffer` reaches `memAlloc` and destroying the buffer reaches
# `memFree`, and BOTH flush the batch (`QuRTDeviceContext.cpp:2162`, `:2180`).
# A witness allocated before each launch and freed after it therefore sends
# whatever the batch was holding, twice per launch, which is what pins the mean
# batch width near one (`npu_witness_alloc_count_test.cpp`). Taking a slot out
# of an allocation that already exists reaches neither call, so the flush pair
# disappears and the batch can hold. That is the whole measured product; the
# guard protocol below is what makes it safe.
#
# ONE SLOT PER IN-FLIGHT LAUNCH IS A CORRECTNESS REQUIREMENT, NOT A BUDGET
# CHOICE. Witness slot 7 is an intra-launch cross-thread channel on the HMX
# arm: the array thread publishes the lock outcome there before the first
# rendezvous (`npu_kernel.mojo:2802`) and every feeder reads it after (`:2807`)
# to decide whether to enter the sweep. A slot shared by two in-flight launches
# would corrupt the kernel's own rendezvous protocol, not merely the host's
# reading of it -- a second launch's array thread writing the word while the
# first launch's feeders read it steers those feeders at random. A future
# "share one slot to save arena bytes" simplification would look harmless and
# would not be. At 128 bytes a slot the whole arena is 16 KiB; there is no
# bytes argument to have.
comptime NPU_ARENA_MAGIC: UInt32 = 0x5749544E  # 'WITN'

# Slots, which must be at least the transport's batch cap
# (`kNPUBatchMaxLaunches`, `QuRTDeviceContext.cpp:168`) -- NOT
# `kNPUMaxBufferSlots`, which is the per-launch buffer-operand count and an
# unrelated 8. The host takes slots with a cursor that wraps here, so two
# launches collide only if more than this many are in flight at once, and a
# batch never holds more than the cap. The transport re-checks this number
# against its own cap when it reads the header and refuses a skewed host.
comptime NPU_ARENA_SLOTS: Int = 64

# A slot is the witness the launch carries, then a host-only guard record the
# device never addresses: the launch's operand is the witness base and a kernel
# writes `NPU_WITNESS_WORDS` from there.
comptime NPU_ARENA_GUARD_BYTES: Int = 128
comptime NPU_ARENA_SLOT_BYTES: Int = NPU_WITNESS_BYTES + NPU_ARENA_GUARD_BYTES
comptime NPU_ARENA_HEADER_BYTES: Int = 128
comptime NPU_ARENA_BYTES: Int = (
    NPU_ARENA_HEADER_BYTES + NPU_ARENA_SLOTS * NPU_ARENA_SLOT_BYTES
)

# Header words. The cursor lives here rather than in a Mojo global because the
# arena pointer is the only thing the global cache carries.
comptime NPU_ARENA_HDR_MAGIC: Int = 0
comptime NPU_ARENA_HDR_SLOTS: Int = 1
comptime NPU_ARENA_HDR_STRIDE: Int = 2
comptime NPU_ARENA_HDR_WITNESS_BYTES: Int = 3
comptime NPU_ARENA_HDR_CURSOR: Int = 4

# A guard record, written by the stub before the launch and read by the flush
# after it. The magic is what lets the transport tell an arena slot from any
# other trailing buffer -- the hand drivers and the C++ tests pass witnesses
# that are not arena slots, and those must arm nothing.
comptime NPU_GUARD_MAGIC: UInt32 = 0x47554152  # 'GUAR'
comptime NPU_GUARD_MAX: Int = 2
comptime NPU_GUARD_WORDS: Int = 12
comptime NPU_GUARD_HDR_MAGIC: Int = 0
comptime NPU_GUARD_HDR_COUNT: Int = 1
comptime NPU_GUARD_HDR_OP_LEN: Int = 2
comptime NPU_GUARD_HDR_OP_PTR: Int = 4  # and 5: a 64-bit host pointer
comptime NPU_GUARD_HDR_WORDS: Int = 8
comptime NPU_GUARD_CLASS: Int = 0
comptime NPU_GUARD_WORD: Int = 1
comptime NPU_GUARD_COND_WORD: Int = 2
comptime NPU_GUARD_COND_VALUE: Int = 3
comptime NPU_GUARD_ACCEPT_COUNT: Int = 4
comptime NPU_GUARD_ACCEPTS: Int = 5
comptime NPU_GUARD_MAX_ACCEPTS: Int = NPU_GUARD_WORDS - NPU_GUARD_ACCEPTS

# In the condition-word field: the guard applies to every launch that armed it.
# A real word index is never this, and the transport treats it as no condition.
comptime NPU_GUARD_COND_NONE: Int = 0xFFFFFFFF

# THE TWO CLASSES, AND THE WHOLE REASON THE TABLE IN
# `.agentwork/flux-npu/WITNESS_GUARD_CLASSIFICATION.md` HAD TO BE WRITTEN FIRST.
#
# CORRECTNESS: the kernel accepted the shape, refused the unit and wrote no
# output, while returning success -- so this guard stands in for a status word
# the device did not raise, and it must invalidate the flush exactly as a
# nonzero status would (`mojo_npu_imp.c:1058-1062`). Delete it the day those
# kernel paths call `mojo_npu_launch_fail` themselves.
#
# ATTRIBUTION: the output is correct and only the measurement is void. Letting
# one of these abort a render is the regression-gate-becomes-correctness-gate
# trap; they record loudly and the render continues.
comptime NPU_GUARD_CLASS_CORRECTNESS: UInt32 = 1
comptime NPU_GUARD_CLASS_ATTRIBUTION: UInt32 = 2


def npu_selector_sig():
    """Stands in for the signature `load_function` asks for.

    `DeviceExternalFunction` carries no argument types, so the signature
    parameter never reaches the launch: `enqueue_function` packs arguments
    positionally by width, and it is the host packer that checks them against
    the selector's declared arity (`QuRTDeviceContext.cpp`, `argCount ==
    bufferCount + scalarCount`). Spelling one signature per selector here would
    add a second copy of an ABI that `NPUKernels.def` deliberately keeps in one
    place, so every stub shares this one.
    """
    pass
