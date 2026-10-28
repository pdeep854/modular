//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the on-device NPU DeviceContext.
//
//===----------------------------------------------------------------------===//

#include "QuRTDeviceContext.h"

#include "NPUDispatchRecorder.h"
#include "npu_constant_ranges.h"

#include "MLRT/Driver/DeviceContext/ErrorMacros.h"
#include "Support/ErrorOr.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/bit.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace M::Driver {

namespace {

/// One kernel of the pre-built skel, as its manifest declares it.
///
/// A FastRPC invocation is typed by the interface definition the skel was
/// generated from, so a typed entry point per kernel would mean a wire layout
/// per kernel. The skel instead carries one deliberately untyped entry point —
/// a selector, a packed block of scalars, and a run of buffer slots — and this
/// records only how a launch's arguments divide between the block and the
/// slots.
struct NPUKernel {
  const char *name;
  /// What the device trampoline switches on, as the manifest declares it.
  uint32_t selector;
  /// Leading arguments of the launch. Each takes a slot of its own, because the
  /// device is handed a mapping per slot rather than an address it could read
  /// out of the block.
  uint32_t bufferCount;
  /// The arguments after those, which travel in the packed block.
  uint32_t scalarCount;
};

// The kernels the signed library was built with. `call` describes the device
// side and is dropped here.
#define NPU_KERNEL(name, selector, bufferCount, scalarCount, call)             \
  {#name, selector, bufferCount, scalarCount},
constexpr NPUKernel kNPUKernels[] = {
#include "NPUKernels.def"
};

/// Holds when every row's declared selector is still its position, which is the
/// manifest's one ordering rule. Checked here rather than trusted because the
/// numbering is a signed library's ABI and moving a row costs a re-sign of
/// every kernel after it; the device trampoline asserts the same thing over its
/// own expansion of the same file.
constexpr bool selectorsAreTheirPositions() {
  uint32_t position = 0;
  for (const NPUKernel &kernel : kNPUKernels)
    if (kernel.selector != position++)
      return false;
  return true;
}
static_assert(selectorsAreTheirPositions(),
              "NPUKernels.def: a row's declared selector is no longer its "
              "position, so a row was inserted, reordered, or removed. Every "
              "selector after it now names a different kernel than the signed "
              "libraries were built with -- append instead.");

const NPUKernel *findKernel(StringRef name) {
  for (const NPUKernel &kernel : kNPUKernels)
    if (name == kernel.name)
      return &kernel;
  return nullptr;
}

/// What the device trampoline switches on, which both halves read out of the
/// same file rather than agreeing by hand.
uint32_t selectorOf(const NPUKernel &kernel) { return kernel.selector; }

/// Method index of the generic entry point: after the interface's own open and
/// close, and after the typed `vadd` the direct-transport example drives.
constexpr uint32_t kNPUDispatchMethodID = 3;

/// Buffer slots the generic entry point declares. Fixed here because a FastRPC
/// method's slots come from the interface definition the skel was generated
/// from, so widening it is an edit to both halves rather than a caller's
/// choice.
constexpr uint32_t kNPUMaxBufferSlots = 8;

/// Width of one scalar's place in the packed block. One fixed width rather than
/// each scalar's own, so an offset is an index times this and neither half has
/// to reproduce the other's alignment arithmetic.
constexpr uint32_t kNPUScalarSlotBytes = 8;

/// Threads one launch may name. The device spreads a launch over a QuRT thread
/// per participant and gives each a descriptor slot of its own for linear uDMA,
/// so this is that table's size — `MOJO_NPU_MAX_LAUNCH_THREADS` in the device
/// library's `mojo_npu_launch.h`. Refused here as well as there so an
/// over-wide mask is reported as the launch's fault rather than as a failed
/// invocation.
constexpr uint32_t kNPUMaxLaunchThreads = 8;

/// The launch word carries the participant mask in its low byte and how the
/// launch is to be instrumented above it — `NPU_LAUNCH_TIME` in the device
/// library's `mojo_npu_imp.c`, which the signed skel already implements. Set
/// only when a timing slot travels with the launch; a production dispatch
/// sends the mask alone and takes the path it always did.
constexpr uint32_t kNPULaunchTime = 0x00010000u;

/// The timing slot holds the per-launch stamps and, when the skel was built
/// with the phase probe, the within-dispatch phase block appended after them
/// and the session sync-counter block after that. One size for all: the slot
/// is latched for the process, so sizing it by whichever record the first
/// caller wanted would truncate the others.
constexpr uint32_t kNPUTimingSlotBytes = sizeof(NPULaunchDeviceTiming) +
                                         sizeof(NPULaunchPhases) +
                                         sizeof(NPULaunchSyncReport);

/// What a stamps-only request carries instead: the sync block rides even
/// without the phase block, because the counters are the witness order 243
/// could not read and the phase record is 30x its size. The skel places the
/// sync block right after the stamps when the slot cannot hold the phase
/// record whole, so the offset the host reads at is decided by the size it
/// sent (`qshimPhaseCopy` in qshim_qurt.c states the layout rule).
constexpr uint32_t kNPUStampSlotBytes =
    sizeof(NPULaunchDeviceTiming) + sizeof(NPULaunchSyncReport);

/// What the pipe-egress request (`MODULAR_NPU_PERF_PIPE`) appends to either
/// shape above: the attention pipe session block rides after the sync block,
/// by the same size-decides-layout rule, only when the host sized the slot
/// for it. Kept out of the default sizes so a run that did not ask sends the
/// slot it always sent and measures exactly what it always measured.
constexpr uint32_t kNPUPipeSlotExtraBytes = sizeof(NPULaunchPipeReport);

/// Whether launches travel over the shared-memory queue instead of one blocking
/// invocation each. One variable so a device A/B moves one thing: both
/// transports are compiled in, and the synchronous one is what an unset
/// environment gets.
bool npuQueueRequested() {
  const char *value = getenv("MODULAR_NPU_QUEUE");
  return value != nullptr && value[0] == '1' && value[1] == '\0';
}

/// Whether the environment asked for batched dispatch. Read apart from
/// `batching()` because the two-transports refusal below has to fire on the
/// request itself: `batching()` also folds in the IO recorder, and a refusal
/// that a recorder could silence would let both flags travel to the device leg
/// unnoticed.
bool npuBatchRequested() {
  const char *value = getenv("MODULAR_NPU_BATCH");
  return value != nullptr && value[0] == '1' && value[1] == '\0';
}

/// The scalar block carries a list of launches rather than one kernel's
/// arguments, and the device runs them back to back behind one crossing.
///
/// The block's layout is restated here from the device library's
/// `mojo_npu_imp.c` for the same reason the rest of the wire is: the two halves
/// are separate builds signed apart, so a layout that stopped agreeing has to
/// break something. `mojo_npu.idl` is untouched -- the block is
/// length-delimited and the device checks its length with `>=`, which is what
/// already lets a launch append per-buffer offsets to it -- so the interface's
/// md5, and with it every generated stub, is unmoved.
constexpr uint32_t kNPULaunchBatch = 0x00040000u;

/// 'NBTC', opening the block.
constexpr uint32_t kNPUBatchMagic = 0x4E425443u;

/// magic, count, report slot, reserved.
constexpr uint32_t kNPUBatchHeaderBytes = 16;

/// selector, thread mask, argument offset, slot map, arity, reserved.
constexpr uint32_t kNPUBatchRecordBytes = 24;

/// Bounds the device's loop against a corrupt count. The eight slots bound a
/// real batch long before this does.
constexpr uint32_t kNPUBatchMaxLaunches = 64;

/// No timing slot travelled with the batch.
constexpr uint32_t kNPUBatchNoSlot = 0xFFFFFFFFu;

// ===--------------------------------------------------------------------===//
// The witness arena, as this side reads it.
//
// The graph stubs take a witness out of one process-lifetime allocation
// instead of creating and destroying a buffer around every launch, because
// `memAlloc` and `memFree` each flush (`:2162`, `:2180`) and a pair of those
// per launch is what pins the batch at width one. Slot layout and these
// constants are written in `max/kernels/src/hexagon_ondevice/dispatch.mojo`;
// restated here rather than shared, as the wire layout in
// `npu_fake_transport.c` is, because the two are separate builds and a layout
// that stopped agreeing has to break something. The magics are what break it:
// a slot whose record does not carry `kNPUGuardMagic` arms nothing, which is
// also how every non-arena witness -- the hand drivers', the tests' -- stays
// unguarded.
//
// What this side evaluates is deliberately generic: "is this witness word one
// of these values, and only when that other word holds that value". The codes
// themselves (`MATMUL_PATH_*`, `HMX_UNPACK_ARM_*`) belong to the code-signed
// kernel set and stay on the stub side, which is their only reader.
// ===--------------------------------------------------------------------===//

/// 'WITN', at the head of the arena allocation.
constexpr uint32_t kNPUArenaMagic = 0x5749544Eu;

/// 'GUAR', at the head of a slot's guard record.
constexpr uint32_t kNPUGuardMagic = 0x47554152u;

/// The witness a launch is given; the guard record follows it.
constexpr uint64_t kNPUArenaWitnessBytes = 128;
constexpr uint64_t kNPUArenaGuardBytes = 128;

/// Arena header words: magic, slot count, slot stride, witness bytes, cursor.
/// The cursor is the stub's; this side reads the three it must agree about.
constexpr uint32_t kNPUArenaMagicWord = 0;
constexpr uint32_t kNPUArenaSlotsWord = 1;
constexpr uint32_t kNPUArenaStrideWord = 2;
constexpr uint32_t kNPUArenaWitnessBytesWord = 3;

/// Guard record header: magic, guard count, op name length, pad, then the op
/// name's address in two words.
constexpr uint32_t kNPUGuardRecordMagicWord = 0;
constexpr uint32_t kNPUGuardRecordCountWord = 1;
constexpr uint32_t kNPUGuardRecordOpLenWord = 2;
constexpr uint32_t kNPUGuardRecordOpPtrWord = 4;
constexpr uint32_t kNPUGuardHeaderWords = 8;

/// One guard: class, witness word, condition word, condition value, accept
/// count, then the accepted values.
constexpr uint32_t kNPUGuardClassWord = 0;
constexpr uint32_t kNPUGuardWordWord = 1;
constexpr uint32_t kNPUGuardCondWordWord = 2;
constexpr uint32_t kNPUGuardCondValueWord = 3;
constexpr uint32_t kNPUGuardAcceptCountWord = 4;
constexpr uint32_t kNPUGuardAcceptsWord = 5;
constexpr uint32_t kNPUGuardWords = 12;
constexpr uint32_t kNPUGuardMax = 2;
constexpr uint32_t kNPUGuardMaxAccepts = kNPUGuardWords - kNPUGuardAcceptsWord;

/// The longest op name this side will read out of a guard record. `mo.*` names
/// are short; a length past this is a corrupt record, not a long op.
constexpr uint32_t kNPUGuardOpNameMax = 64;

/// In a guard's condition-word field: it applies to every launch that armed it.
constexpr uint32_t kNPUGuardCondNone = 0xFFFFFFFFu;

/// The kernel refused the unit and wrote no output while returning success, so
/// this guard stands in for a status word the device did not raise: it must
/// invalidate the flush exactly as a nonzero status does.
constexpr uint32_t kNPUGuardClassCorrectness = 1;

/// The output is correct and only the measurement is void. Recorded loudly,
/// never fatal -- an instrument that can abort a render is a regression gate
/// that has become a correctness gate.
constexpr uint32_t kNPUGuardClassAttribution = 2;

/// The selector a batch travels under: deliberately not a manifest row, so a
/// skel built before batching refuses the call by name instead of decoding the
/// block as one kernel's arguments.
constexpr uint32_t kNPUBatchSelector = 0xFFFFFFFFu;

/// Writes one wire word, which the block carries unaligned.
void writeWireWord(uint8_t *at, uint32_t value) {
  memcpy(at, &value, sizeof(value));
}

/// Whether launches are held for a whole-step table: the ENTIRE host-ordered
/// launch list of a step sent as one dispatch, walked on the device --
/// llama.cpp's op-table shape (up to 1024 ops per packet) reached over the
/// blocking transport. Strictly "1", like every other arm.
bool npuMegastepRequested() {
  const char *value = getenv("MODULAR_NPU_MEGASTEP");
  return value != nullptr && value[0] == '1' && value[1] == '\0';
}

/// The matmul-chain hold: the host half of selector 48
/// (`MODULAR_NPU_MATMUL_CHAIN`, `registrations.mojo`). The stub holds exactly
/// ONE fully-packed sel-24 launch between two adjacent matmul registrations,
/// so an independent successor can claim it back and the pair can go out as
/// one `matmul_chain2` launch. Unlike the batch, whose graph-path corruption
/// class is a HOST READER the transport never hears about (186A), the stub
/// closes that class itself: every registration in `registrations.mojo`
/// fires the hold before touching any operand, and this side fires it at
/// `flushBatch` -- every entry point through which the host can observe
/// device memory -- so the hold cannot outlive the window between two
/// registrations plus the copy/sync/alloc/free seams. File-static rather
/// than per-context because the stub reaches it through bare C symbols (the
/// PE JIT resolves export names, not instances), under the same
/// single-submitter assumption `enqueueFunctionExecDirect` already documents
/// for `batch_`.
static QuRTDeviceContext *gChainHolder = nullptr;
static bool gChainHoldNext = false;
static bool gChainHeldFired = false;

/// The step table's wire, restated from the device library's `mojo_npu_imp.c`
/// for the same reason the batch's is: two halves signed apart, so a layout
/// that stopped agreeing has to break something. The batch's block cannot
/// carry a step: its operands are named by one of eight declared slots, and a
/// step's allocation union is over a hundred wide (every weight is its own
/// allocation), which is why the recorded 398-launch step batches into ~47
/// calls. A megastep record therefore names each operand either by call slot
/// or by an entry in a session table the skel keeps across calls -- filled by
/// register-mode dispatches that present each allocation as an ordinary
/// interface pointer argument, once per session.
///
/// Only allocations classified init-constant (the same provenance test the
/// queue keys descriptor maintenance off) may be named by table entry: the
/// registration presentation flushes the host's one init write, and nothing
/// dirties the allocation after it, so the run call's omission of per-call
/// maintenance takes away nothing. Everything else rides the run call's own
/// slots with the vendor's full maintenance, and when that union outgrows the
/// slots the table flushes -- the step splits rather than the claim widening.
///
/// 'NMGS', opening the block.
constexpr uint32_t kNPUMegastepMagic = 0x4E4D4753u;

/// magic, count, report slot, mode.
constexpr uint32_t kNPUMegastepHeaderBytes = 16;

/// selector, thread mask, argument offset, arity, then eight 16-bit operand
/// references.
constexpr uint32_t kNPUMegastepRecordBytes = 32;

/// register entry: table index, slot of the registering call.
constexpr uint32_t kNPUMegastepEntryBytes = 8;

constexpr uint32_t kNPUMegastepModeRegister = 0;
constexpr uint32_t kNPUMegastepModeRun = 1;

/// Bounds one table, mirroring the device's own bound against a corrupt
/// count. A denoise step is ~403 launches, so a whole step fits with headroom.
constexpr uint32_t kNPUMegastepMaxLaunches = 1024;

/// Session-table entries the device keeps; the real model registers ~170.
constexpr uint32_t kNPUMegastepTableEntries = 512;

/// Set in an operand reference to name the session table.
constexpr uint16_t kNPUMegastepRefTable = 0x8000;

/// Whether registry-sourced write-once buffers may carry the init-once
/// attribute that skips the driver's flush. Strictly "1", like the queue and
/// mmap arms, and read per call rather than latched: the flag is the single
/// variable of the on-device A/B, and reading it fresh keeps one process's
/// control leg honest.
bool npuWeightsNoFlushRequested() {
  const char *value = getenv("MODULAR_NPU_WEIGHTS_NOFLUSH");
  return value != nullptr && value[0] == '1' && value[1] == '\0';
}

/// Whether an allocation is mapped to the DSP once, for its life, instead of
/// once per call by the driver underneath `remote_handle64_invoke`. Strictly
/// "1", like every other arm, and read per call rather than latched -- the
/// flag is the single variable of the on-device A/B.
///
/// The shipped ggml-hexagon backend keeps its shared buffers mapped for the
/// session this way (`ggml-hexagon.cpp:359-378`: `rpcmem_alloc2`,
/// `rpcmem_to_fd`, then `fastrpc_mmap` inline at the tail of `alloc`, torn
/// down only from buffer destruction), and this transport already states the
/// same claim from the other side: "that mapping is what the synchronous path
/// got for free, per call, from the driver" (`NPUQueueTransport.cpp:758-764`).
bool npuMapOnceRequested() {
  const char *value = getenv("MODULAR_NPU_MAP_ONCE");
  return value != nullptr && value[0] == '1' && value[1] == '\0';
}

/// Whether freed device chunks are parked in a size-keyed pool and re-issued
/// address-stable, instead of each step's buffers dying by `rpcmem_free` and
/// being re-created. Strictly "1" and read per call, like every other arm.
///
/// The mechanism it deletes: the vendor driver charges a byte-proportional
/// first-presentation cost (~4.17 GiB/s, measured three independent times:
/// order 041's weight walk, order 151's fresh-vs-held VAE buffers, order
/// 238's matmul-at-floor once the weights became process-lifetime) the first
/// time a buffer handle is presented in an invoke, and the ~1 ms floor after.
/// A freed-and-reallocated buffer is a fresh handle over the same bytes, so
/// the engine's per-step step-buffer churn re-pays ~1 GiB of that walk every
/// step, inside the blocking invoke of whichever launch presents the buffers
/// first (SINUSOID_DIAGNOSIS.md, 2026-08-16). Parking keeps the handle alive
/// across steps, which is the endgame the weights already measure at rev 8.
///
/// Adopted, not invented: the reference backend allocates session-lifetime
/// buffers and maps once (`ggml-hexagon.cpp:359-378`), and this file made the
/// identical move for the witness arena, with the reason written on it. The
/// `NPU_` prefix (not `MODULAR_NPU_`) marks it recipe-file-controlled like
/// the VAE walk levers: arming is one env row in the run recipe — no skel
/// change, no manifest row, no re-sign window.
bool npuStepAllocPoolRequested() {
  const char *value = getenv("NPU_STEP_ALLOC_POOL");
  return value != nullptr && value[0] == '1' && value[1] == '\0';
}

/// Whether the session votes itself out of power management: the host's
/// FastRPC latency vote and the DSP's session-long HAP_power vote, together
/// behind one variable so a device A/B moves one thing.
///
/// Default ON, unlike every other arm, on two measured results: the vote is
/// worth 1.9% of denoise execute at bit-identical outputs (order 073, Welch
/// t = -9.36), and the HMX lock grant REQUIRES a live session vote -- seven
/// windows of QURT_EINVALID refusals ended when the vote alone flipped the
/// production lock to a grant (order 077). `MODULAR_NPU_POWER=0` is the
/// kill-switch, restoring the unvoted session -- and, with it, no HMX.
bool npuPowerRequested() {
  const char *value = getenv("MODULAR_NPU_POWER");
  return !(value != nullptr && value[0] == '0' && value[1] == '\0');
}

/// Names the skel and the entry point the loader dispatches through. The path
/// is bare, so WHICH library loads is decided by the FastRPC loader's own
/// default search order — measured on the target box (order-025 step 1) to be
/// neither the vendor-library directory, nor ADSP_LIBRARY_PATH, nor any
/// registry key. The 2026-08-07 device legs (order-029 record) settled the
/// rest: the loader refuses ABSOLUTE file:/// URIs as a class -- a known-good
/// tuple named by absolute path returns AEE_EUNABLETOLOAD -- and the bare
/// name binds the correct library once the host-side kernel list is current
/// (the historic "wrong tuple binds" readings were a stale list compiled
/// into the probing exe, not loader behavior).
constexpr const char *kSkelURI =
    "file:///libmojo_npu_skel.so?mojo_npu_skel_handle_invoke"
    "&_modver=1.0&_dom=cdsp";

/// The URI the open uses: always kSkelURI. MOJO_NPU_SKEL_DIR once re-rooted
/// the bare name to an absolute path; the platform refuses that class of URI
/// (order-029, leg 1), so a set override now fails the open up front with
/// the finding, rather than sending a URI the loader will refuse -- a stale
/// env var must be loud, not a mystery refusal.
ErrorOr<const char *> skelOpenURI() {
  const char *dir = getenv("MOJO_NPU_SKEL_DIR");
  if (dir != nullptr && *dir != '\0')
    return Error(
        "MOJO_NPU_SKEL_DIR is set, but absolute-path skel URIs are refused "
        "by the FastRPC loader on this platform (order-029 record); unset it "
        "-- the bare name binds the staged tuple when the host kernel list "
        "is current");
  return kSkelURI;
}

/// What a session knows about its power vote and the skel it opened, latched
/// at skel open and consulted before any HMX-capable dispatch.
struct PowerVoteOutcome {
  /// The DSP vote is placed and held: the `power_vote` dispatch returned
  /// success and `HAP_power_set` was resident. The host latency half is
  /// deliberately not part of liveness -- order 077 proved the DSP half alone
  /// flips the HMX lock from refusal to grant.
  bool live = false;
  /// Whether the skel routes any kernel to the HMX array: 1 when the vote
  /// report carried the capability word, 0 when it did not, -1 when no report
  /// came back (dispatch refused or the buffer could not be allocated).
  int skelHmx = -1;
  /// Why `live` is false, phrased for the dispatch-time refusal.
  std::string reason;
};

/// 'HMX1' in the vote report's word 5: this skel was built with
/// `HMX_MATMUL=1` and routes eligible dispatches to the matrix array. The
/// same marker the matmul witness uses for "the array computed this". A skel
/// that predates the word leaves it zero, and a skel that predates the row
/// refuses the dispatch -- both read as "not an HMX build", which is sound
/// because every HMX-capable skel postdates both (rows are append-only).
constexpr uint32_t kNPUPowerVoteHmxMarker = 0x484D5831;

/// Words the vote's report carries; 0-4 are the vote statuses, 5 the HMX
/// capability marker.
constexpr uint32_t kNPUPowerVoteReportWords = 8;

/// Dispatched-but-unanswerable sentinel for `dispatchPowerVoteRow`: the
/// report buffer could not be allocated, so nothing crossed the wire.
constexpr uint32_t kNPUPowerVoteNoReport = 0xFFFFFFFFu;

/// Sends one `power_vote` dispatch (`enable` 1 places the session vote, 0
/// releases/queries) and copies the kernel's report back into `report`.
/// Returns the invocation's status, or `kNPUPowerVoteNoReport` when the
/// report buffer could not be allocated. Hand-rolled against the driver
/// rather than routed through `dispatchSingle`: the vote must precede the
/// launches it covers and runs under the context lock, where re-entering the
/// dispatch path would deadlock.
uint32_t dispatchPowerVoteRow(QuRTDriver &driver, uint64_t handle,
                              uint32_t enable,
                              uint32_t report[kNPUPowerVoteReportWords]) {
  memset(report, 0, kNPUPowerVoteReportWords * sizeof(uint32_t));
  const NPUKernel *kernel = findKernel("power_vote");
  if (kernel == nullptr)
    return kNPUPowerVoteNoReport;

  auto *wire = static_cast<uint32_t *>(
      driver.rpcmemAlloc(kQuRTHeapIDSystem, kQuRTRpcmemDefaultFlags,
                         kNPUPowerVoteReportWords * sizeof(uint32_t)));
  if (wire == nullptr)
    return kNPUPowerVoteNoReport;
  memset(wire, 0, kNPUPowerVoteReportWords * sizeof(uint32_t));

  // The wire a single dispatch sends, at this kernel's arity: two scalar
  // slots (enable, word capacity), then the one buffer's offset slot.
  uint32_t scalars[6] = {};
  scalars[0] = enable;
  scalars[2] = kNPUPowerVoteReportWords;

  llvm::SmallVector<uint32_t, 19> primIn(3 + 2 * kNPUMaxBufferSlots, 0);
  primIn[0] = selectorOf(*kernel);
  primIn[1] = 0x1;
  primIn[2] = sizeof(scalars);
  primIn[3] = kNPUPowerVoteReportWords * sizeof(uint32_t);
  primIn[4] = kNPUPowerVoteReportWords * sizeof(uint32_t);

  llvm::SmallVector<QuRTRemoteBuf, 18> slots(2 + 2 * kNPUMaxBufferSlots,
                                             QuRTRemoteBuf{nullptr, 0});
  slots[0] = {primIn.data(), primIn.size() * sizeof(uint32_t)};
  slots[1] = {scalars, sizeof(scalars)};
  slots[2] = {wire, kNPUPowerVoteReportWords * sizeof(uint32_t)};
  slots[2 + kNPUMaxBufferSlots] = {wire,
                                   kNPUPowerVoteReportWords * sizeof(uint32_t)};

  uint32_t status = static_cast<uint32_t>(
      driver.handleInvoke(handle,
                          qurtScalars(kNPUDispatchMethodID,
                                      /*inBufs=*/2 + kNPUMaxBufferSlots,
                                      /*outBufs=*/kNPUMaxBufferSlots),
                          slots.data()));
  memcpy(report, wire, kNPUPowerVoteReportWords * sizeof(uint32_t));
  driver.rpcmemFree(wire);
  return status;
}

/// The session-long power vote, placed once right after the skel opens
/// (default; `MODULAR_NPU_POWER=0` disables). Two halves behind the one
/// variable, each self-reporting so an A/B leg can say which of them it
/// actually got:
///
///  - the host half is the FastRPC QoS vote (`DSPRPC_CONTROL_LATENCY`) the
///    shipped ggml-hexagon backend sets right after its session open
///    (`ggml-hexagon.cpp:1786-1793`), keeping the RPC path out of deep LPM;
///  - the DSP half is one `power_vote` dispatch, whose kernel holds the
///    HAP_power votes that backend pins for its whole session
///    (`htp/main.c:426-505`) until `mojo_npu_close` releases them.
///
/// Every failure is a named report rather than an error: a session without
/// its vote is slower, not wrong, and the one thing a silent skip would cost
/// is the A/B's meaning. The exception is HMX, whose lock the platform
/// refuses in an unvoted session (order 077) -- the returned outcome is what
/// `guardHmxDispatch` turns into a dispatch-time refusal. Never dispatched
/// through the queue or a held batch -- the vote must be placed before the
/// launches it covers, and this runs under the context lock the queue's own
/// open takes.
PowerVoteOutcome voteNPUPower(QuRTDriver &driver, uint64_t handle) {
  if (driver.handleControl == nullptr) {
    fprintf(stderr,
            "npu power: MODULAR_NPU_POWER on, but this machine's FastRPC "
            "library exports no remote_handle64_control; host latency vote "
            "skipped\n");
  } else {
    QuRTRemoteLatency latency = {/*enable=*/1, /*latency=*/100};
    int rc = driver.handleControl(handle, kQuRTControlLatency, &latency,
                                  sizeof(latency));
    if (rc == 0)
      fprintf(stderr, "npu power: host latency vote set\n");
    else
      fprintf(stderr, "npu power: host latency vote refused: %s\n",
              qurtStatusMessage(static_cast<uint32_t>(rc)).c_str());
  }

  PowerVoteOutcome outcome;
  if (findKernel("power_vote") == nullptr) {
    fprintf(stderr, "npu power: no power_vote row in this host's manifest; "
                    "dsp vote skipped\n");
    outcome.reason = "this host's manifest carries no power_vote row";
    return outcome;
  }

  uint32_t report[kNPUPowerVoteReportWords];
  uint32_t status = dispatchPowerVoteRow(driver, handle, /*enable=*/1, report);
  if (status == kNPUPowerVoteNoReport) {
    fprintf(stderr, "npu power: could not allocate the vote's report buffer; "
                    "dsp vote skipped\n");
    outcome.reason = "the vote's report buffer could not be allocated";
    return outcome;
  }

  if (status != 0) {
    fprintf(stderr,
            "npu power: dsp vote refused by the signed skel: %s -- it "
            "predates the power_vote row\n",
            qurtStatusMessage(status).c_str());
    outcome.reason = "the dsp vote was refused by the signed skel";
    return outcome;
  }

  outcome.skelHmx = report[5] == kNPUPowerVoteHmxMarker ? 1 : 0;
  if (report[0] == 0) {
    fprintf(stderr, "npu power: dsp vote unavailable: HAP_power_set is not "
                    "resident in this protection domain\n");
    outcome.reason = "HAP_power_set is not resident in this protection domain";
    return outcome;
  }

  fprintf(stderr,
          "npu power: dsp vote max (dcvs off, sleep off, hmx clock pinned) "
          "-- apptype=%d dcvs=%d hvx=%d hmx=%d\n",
          static_cast<int>(report[1]), static_cast<int>(report[2]),
          static_cast<int>(report[3]), static_cast<int>(report[4]));
  outcome.live = true;
  return outcome;
}

/// The kernels whose device bodies may take the HMX array instead of HVX, by
/// manifest name: `matmul` routes its stripes to the array on an
/// `HMX_MATMUL=1` build (`MM_PATH_HMX`, `matmul_kernel.mojo`),
/// `hmx_int8_probe` exists only to exercise it (`mojo_hmx_int8_probe`,
/// `npu_kernel.mojo`), and both attention selectors take it on an
/// `HMX_ATTN=1` build through their `rt` arm (`_mha_hmx_team`,
/// `npu_kernel.mojo`). Whether the *loaded* skel is such a build is the vote
/// report's capability word, not this list -- this list only scopes which
/// dispatches consult it. Extend it when a new kernel gains an HMX path.
///
/// THE ATTENTION ROWS WERE MISSING FROM ORDERS 118-123, when the array arm was
/// added, until order 134 caught what that costs. A dispatch this function
/// declines never reaches `guardHmxDispatch`'s vote check, so an attention
/// launch that takes `qurt_hmx_lock` in a session where no `matmul` has
/// dispatched yet asks for the array with no power vote behind it -- and the
/// vote is the grant's discriminator (order 077). The grant then fails, the
/// device body marks the launch, and the host sees the first dispatch fail.
/// A session that happened to run a matmul first was masked, because the vote
/// goes live once and stays live.
bool npuKernelMayEngageHmx(StringRef name) {
  return name == "matmul" || name == "hmx_int8_probe" ||
         name == "mha_no_cache" || name == "mha_masked";
}

} // namespace

//===----------------------------------------------------------------------===//
// QuRTDevice
//===----------------------------------------------------------------------===//

QuRTDevice::QuRTDevice(int id, Device::MemInfo memInfo)
    : Device(api, "NPU", id, memInfo, M::kPreferredMemoryAlignment) {}

QuRTDevice::~QuRTDevice() { clearMemoryManager(); }

ErrorOr<DeviceRef> QuRTDevice::create(int id) {
  // Only one on-device NPU exists per part.
  if (id != 0)
    return Error("NPU device id must be 0, got " + std::to_string(id));

  // Binding the transport is what proves the device is there. Deliberately does
  // not open the kernel library: see QuRTDeviceContext::skelHandle.
  ErrorOr<QuRTDriver &> driver = QuRTDriver::get();
  if (driver.isError())
    return driver.takeError();

  // Process-lifetime, like every other backend's device: the base Device owns a
  // host MemoryManager whose destructor asserts it holds no chunks, and only
  // the device-side manager has a pre-teardown clear hook. A per-request device
  // would trip that assertion as soon as a host buffer had been allocated.
  //
  // MemInfo is left zero because device memory here is host memory: there is no
  // separate budget to report, and the allocator must not cache against one.
  static QuRTDevice *device = new QuRTDevice(id, Device::MemInfo());
  return DeviceRef::copy(device);
}

ErrorOr<int> QuRTDevice::getApiVersion() const {
  RETURN_ERROR("the on-device NPU transport reports no API version");
}

ErrorOr<DriverVersion &> QuRTDevice::getDriverVersion() const {
  RETURN_ERROR("the on-device NPU transport reports no driver version");
}

ErrorOrSuccess QuRTDevice::isCompatible() const {
  ErrorOr<QuRTDriver &> driver = QuRTDriver::get();
  if (driver.isError())
    RETURN_ERROR(driver.getError());
  return success();
}

ErrorOr<int> QuRTDevice::computeCapability() const {
  RETURN_ERROR("the on-device NPU reports no compute capability");
}

ErrorOr<bool> QuRTDevice::canAccess(const Device &peer) const { return false; }

ErrorOr<int> QuRTDevice::getAttribute(Attribute attr) const {
  // TODO(fabio): the device-side shim can answer thread and scratch-memory
  // queries, but only from a kernel already running on the device; wire these
  // through once an entry point exists to ask on the host's behalf.
  RETURN_ERROR("the on-device NPU exposes no device attributes to the host");
}

ErrorOr<std::string> QuRTDevice::getPciBusId() const {
  RETURN_ERROR("the on-device NPU is not on a PCI bus");
}

int QuRTDevice::deviceCount() const { return 1; }

ErrorOrSuccess QuRTDevice::chunkAlloc(void **ptr, size_t byteSize,
                                      bool isHost) {
  ErrorOr<QuRTDriver &> driver = QuRTDriver::get();
  if (driver.isError())
    RETURN_ERROR(driver.getError());

  // The allocator takes a signed 32-bit size.
  //
  // TODO(fabio): allocations of 2 GiB and over are rejected rather than split;
  // move to the transport's 64-bit allocation entry point when a model needs a
  // single buffer that large.
  if (byteSize > 0x7FFFFFFFu)
    RETURN_ERROR("NPU allocation of " + std::to_string(byteSize) +
                 " bytes exceeds the transport's 2 GiB limit");

  if (npuMapOnceRequested() && npuQueueRequested())
    RETURN_ERROR("MODULAR_NPU_MAP_ONCE=1 and MODULAR_NPU_QUEUE=1 both map "
                 "every allocation to the DSP, and the queue's own table "
                 "(NPUQueueTransport::mapAllocation) unmaps what it mapped; "
                 "set exactly one");

  // Exact-size pool hit: the parked chunk is re-issued with its rpcmem handle
  // never having died, so the driver's first-presentation work for it is
  // already done. It cannot carry a stale mapping, attribute or constant
  // class — parking runs `chunkFree`'s erasures verbatim before the block
  // reaches the pool — and it cannot be zero-filled: rpcmem gives no zeroing
  // guarantee (npu_fake_transport.c states the contract), so a caller that
  // needed zeros never had them. A miss falls through to the raw path and is
  // counted, so the exit census can say how much of the run the pool served.
  if (npuStepAllocPoolRequested()) {
    std::lock_guard<std::mutex> lock(allocationsMu_);
    auto bucket = stepPoolFree_.find(byteSize);
    if (bucket != stepPoolFree_.end() && !bucket->second.empty()) {
      void *block = bucket->second.back();
      bucket->second.pop_back();
      --stepPoolParkedBlocks_;
      stepPoolParkedBytes_ -= byteSize;
      ++stepPoolReuses_;
      stepPoolReusedBytes_ += byteSize;
      // The treatment's self-report, like map-once's: one line at first
      // reuse, so an A/B leg whose stderr lacks it is known untreated.
      if (!stepPoolAnnounced_) {
        stepPoolAnnounced_ = true;
        fprintf(stderr, "npu step-pool: armed (first reuse %p, %zu bytes)\n",
                block, byteSize);
      }
      allocations_[reinterpret_cast<uintptr_t>(block)] = byteSize;
      *ptr = block;
      return success();
    }
    ++stepPoolRawWhileArmed_;
  }

  *ptr = driver->rpcmemAlloc(kQuRTHeapIDSystem, kQuRTRpcmemDefaultFlags,
                             static_cast<int>(byteSize));
  if (*ptr == nullptr)
    RETURN_ERROR("NPU allocation of " + std::to_string(byteSize) +
                 " bytes failed");

  std::lock_guard<std::mutex> lock(allocationsMu_);
  // A fresh allocation already carrying a mapping means the previous owner of
  // this address was freed without one, and rpcmem hands addresses back: the
  // stale descriptor would be resolved for the new buffer. Checked against the
  // table rather than behind the flag -- when nothing was ever mapped the map
  // is empty and this cannot fire, and when the flag was flipped mid-process
  // it still can.
  if (!mappedFDs_.empty() &&
      mappedFDs_.count(reinterpret_cast<uintptr_t>(*ptr)) != 0) {
    driver->rpcmemFree(*ptr);
    void *stale = *ptr;
    *ptr = nullptr;
    RETURN_ERROR("the NPU allocation at " +
                 std::to_string(reinterpret_cast<uintptr_t>(stale)) +
                 " is already mapped, so a mapping outlived its buffer");
  }
  allocations_[reinterpret_cast<uintptr_t>(*ptr)] = byteSize;
  return success();
}

ErrorOrSuccess QuRTDevice::mapAllocationOnce(QuRTDriver &driver, void *base,
                                             size_t byteSize) {
  // Refusing beats proceeding bare, the way the NOFLUSH probe already refuses:
  // an arm that silently ran without its treatment is indistinguishable from
  // its own control.
  if (driver.rpcmemToFD == nullptr || driver.fastrpcMmap == nullptr)
    RETURN_ERROR("MODULAR_NPU_MAP_ONCE is set, but this machine's FastRPC "
                 "library exports no rpcmem_to_fd/fastrpc_mmap, so an "
                 "allocation cannot be mapped for its lifetime");

  auto address = reinterpret_cast<uintptr_t>(base);
  {
    // Idempotent on the allocation already being mapped, the way the
    // reference's `alloc` opens with `if (this->base) return;`
    // (`ggml-hexagon.cpp:359`). Every launch naming this allocation reaches
    // here, and only the first may map.
    std::lock_guard<std::mutex> lock(allocationsMu_);
    if (mappedFDs_.count(address) != 0)
      return success();
  }

  int fd = driver.rpcmemToFD(base);
  if (fd < 0)
    RETURN_ERROR("rpcmem_to_fd refused the allocation at " +
                 std::to_string(address) + ": returned " + std::to_string(fd) +
                 ", so it cannot be mapped for its lifetime");

  // MAP_FD, not MAP_STATIC: the vendor's buffer bookkeeping is fd-keyed and a
  // STATIC mapping never lands in it -- order 081 measured the DSP read
  // refusing with AEE_ENOSUCHMAP under STATIC, at packet-write time rather
  // than at map time. The full argument is on the constant's definition.
  int rc = driver.fastrpcMmap(kQuRTComputeDomainID, fd, base, /*offset=*/0,
                              byteSize, NPU_FASTRPC_MAP_FD);
  if (rc != 0)
    RETURN_ERROR("fastrpc_mmap of the " + std::to_string(byteSize) +
                 "-byte NPU allocation at " + std::to_string(address) +
                 " failed: " + dspQueueStatusMessage(rc));

  {
    std::lock_guard<std::mutex> lock(allocationsMu_);
    mappedFDs_[address] = fd;
  }
  // The treatment's self-report, like the NOFLUSH probe's: an on-device A/B
  // leg is judged by these lines, whose count is the mechanism check -- one
  // pair per allocation for the run, not one per launch.
  fprintf(stderr, "npu map-once: mapped %p (%zu bytes) fd %d\n", base, byteSize,
          fd);
  return success();
}

namespace {

/// The device whose step pool reports at process exit. One pointer, not a
/// registry: the device is a process-lifetime singleton (`create`'s leaked
/// static), so it outlives the hook and there is never a second.
QuRTDevice *stepPoolCensusDevice = nullptr;

void reportStepPoolCensusAtExit() {
  if (stepPoolCensusDevice != nullptr)
    stepPoolCensusDevice->reportStepPoolCensus();
}

} // namespace

ErrorOrSuccess QuRTDevice::chunkFree(void *ptr, bool isHost) {
  ErrorOr<QuRTDriver &> driver = QuRTDriver::get();
  if (driver.isError())
    RETURN_ERROR(driver.getError());
  {
    std::lock_guard<std::mutex> lock(allocationsMu_);
    auto base = reinterpret_cast<uintptr_t>(ptr);
    auto it = allocations_.find(base);
    // Deregister (fd -1, per the vendor header) before the free: the driver
    // keys registrations by address range and rpcmem hands addresses back, so
    // an attribute left behind would silently ride a future allocation at the
    // same address.
    if (attributed_.erase(base) != 0 && it != allocations_.end() &&
        driver->registerBufAttr != nullptr)
      driver->registerBufAttr(ptr, it->second, /*fd=*/-1, kQuRTAttrNone);
    // Unmapped before the free and address-exact, for the reason the attribute
    // deregistration above gives: rpcmem hands addresses back, and a mapping
    // left behind would be resolved for whatever lands at this address next.
    // Driven by the table rather than by the flag, so a flag flipped
    // mid-process still unmaps exactly what it mapped.
    if (auto mapping = mappedFDs_.find(base); mapping != mappedFDs_.end()) {
      if (driver->fastrpcMunmap != nullptr)
        driver->fastrpcMunmap(kQuRTComputeDomainID, mapping->second, ptr,
                              it != allocations_.end() ? it->second : 0);
      fprintf(stderr, "npu map-once: unmapped %p fd %d\n", ptr,
              mapping->second);
      mappedFDs_.erase(mapping);
    }
    // rpcmem hands addresses back, so a class left behind would silently ride
    // a future allocation at the same address -- the same hazard the attribute
    // erase above guards against.
    constantBases_.erase(base);
    // Parking replaces only the `rpcmem_free` below: every erasure above ran
    // exactly as on a real free, so a parked block is as bare as a fresh one
    // -- no attribute, no mapping, no constant class -- and only its rpcmem
    // handle survives, which is the pool's whole point. A pointer this device
    // never allocated has no size to key a bucket on and takes the raw path.
    if (npuStepAllocPoolRequested() && it != allocations_.end()) {
      size_t byteSize = it->second;
      allocations_.erase(it);
      stepPoolFree_[byteSize].push_back(ptr);
      ++stepPoolParks_;
      ++stepPoolParkedBlocks_;
      stepPoolParkedBytes_ += byteSize;
      if (stepPoolParkedBlocks_ > stepPoolHighWaterBlocks_)
        stepPoolHighWaterBlocks_ = stepPoolParkedBlocks_;
      if (stepPoolParkedBytes_ > stepPoolHighWaterBytes_)
        stepPoolHighWaterBytes_ = stepPoolParkedBytes_;
      if (!stepPoolExitArmed_) {
        stepPoolExitArmed_ = true;
        stepPoolCensusDevice = this;
        atexit(reportStepPoolCensusAtExit);
      }
      return success();
    }
    if (it != allocations_.end())
      allocations_.erase(it);
  }
  driver->rpcmemFree(ptr);
  return success();
}

void QuRTDevice::reportStepPoolCensus() {
  std::lock_guard<std::mutex> lock(allocationsMu_);
  fprintf(stderr,
          "npu step-pool: census: high-water %llu blocks / %llu bytes parked; "
          "served %llu reuses / %llu bytes; %llu raw allocs while armed; %llu "
          "parks\n",
          static_cast<unsigned long long>(stepPoolHighWaterBlocks_),
          static_cast<unsigned long long>(stepPoolHighWaterBytes_),
          static_cast<unsigned long long>(stepPoolReuses_),
          static_cast<unsigned long long>(stepPoolReusedBytes_),
          static_cast<unsigned long long>(stepPoolRawWhileArmed_),
          static_cast<unsigned long long>(stepPoolParks_));
}

ErrorOrSuccess QuRTDevice::noteHostWrite(void *dst, const void *src,
                                         size_t byteSize) {
  // The queue transport wants the classification without the attribute: its
  // descriptor flags are per class, chosen at `send`, so a queue-enabled run
  // must keep `constantBases_` current even with the NOFLUSH probe off. The
  // step table reads the same class -- only a classified allocation may be
  // named by session-table reference -- so it keeps the bookkeeping live too.
  bool applyAttr = npuWeightsNoFlushRequested();
  if (!applyAttr && !npuQueueRequested() && !npuMegastepRequested())
    return success();

  auto address = reinterpret_cast<uintptr_t>(dst);
  std::lock_guard<std::mutex> lock(allocationsMu_);
  // `allocationFrom`'s containment lookup, inlined because that helper takes
  // this lock. A destination outside every allocation is a plain host buffer:
  // nothing to classify.
  auto after = allocations_.upper_bound(address);
  if (after == allocations_.begin())
    return success();
  auto [base, allocBytes] = *std::prev(after);
  if (address - base >= allocBytes)
    return success();

  // Init-written means: this write fills its whole allocation and every source
  // byte is memory the loader declared constant. Anything else -- a partial
  // write, a fill, an undeclared source -- makes the buffer ordinary again.
  bool initWritten = address == base && byteSize == allocBytes &&
                     src != nullptr &&
                     MODULAR_NPUConstantHostRangeCovers(src, byteSize) != 0;
  if (initWritten)
    constantBases_.insert(base);
  else
    constantBases_.erase(base);
  if (!applyAttr)
    return success();
  bool attributed = attributed_.count(base) != 0;
  if (initWritten == attributed)
    return success();
  RETURN_IF_ERROR(setBufferAttr(
      base, allocBytes, initWritten ? kQuRTAttrForceNoFlush : kQuRTAttrNone));
  if (initWritten)
    attributed_.insert(base);
  else
    attributed_.erase(base);
  // The treatment's self-report: an on-device A/B leg is judged by these lines
  // (count and direction), the same way the skel-open line names the tuple.
  fprintf(stderr, "npu weights noflush: %s %p (%zu bytes)\n",
          initWritten ? "applied to" : "withdrawn from",
          reinterpret_cast<void *>(base), allocBytes);
  return success();
}

bool QuRTDevice::isDeclaredConstant(const void *base) const {
  std::lock_guard<std::mutex> lock(allocationsMu_);
  return constantBases_.count(reinterpret_cast<uintptr_t>(base)) != 0;
}

ErrorOrSuccess QuRTDevice::setBufferAttr(uintptr_t base, size_t byteSize,
                                         int attr) {
  ErrorOr<QuRTDriver &> driver = QuRTDriver::get();
  if (driver.isError())
    RETURN_ERROR(driver.getError());
  // Refusing beats proceeding bare: with the flag set, a silently unset
  // attribute would make an A/B leg indistinguishable from its own control.
  if (driver->registerBufAttr == nullptr)
    RETURN_ERROR("MODULAR_NPU_WEIGHTS_NOFLUSH is set, but this machine's "
                 "FastRPC library exports no remote_register_buf_attr2");
  if (driver->rpcmemToFD == nullptr)
    RETURN_ERROR("MODULAR_NPU_WEIGHTS_NOFLUSH is set, but this machine's "
                 "FastRPC library exports no rpcmem_to_fd");
  int fd = driver->rpcmemToFD(reinterpret_cast<void *>(base));
  if (fd < 0)
    RETURN_ERROR("rpcmem_to_fd refused the allocation at " +
                 std::to_string(base) + ": returned " + std::to_string(fd) +
                 ", so its buffer attribute cannot be set");
  driver->registerBufAttr(reinterpret_cast<void *>(base), byteSize, fd, attr);
  return success();
}

ErrorOr<QuRTDevice::Allocation>
QuRTDevice::allocationFrom(const void *ptr) const {
  auto address = reinterpret_cast<uintptr_t>(ptr);
  std::lock_guard<std::mutex> lock(allocationsMu_);
  auto after = allocations_.upper_bound(address);
  if (after != allocations_.begin()) {
    auto [base, byteSize] = *std::prev(after);
    if (address - base < byteSize)
      return Allocation{base, byteSize};
  }
  RETURN_ERROR("no NPU allocation holds address " + std::to_string(address) +
               ", so no length can be stated for it: only memory this device "
               "allocated is reachable from a kernel");
}

ErrorOr<size_t> QuRTDevice::allocationBytesFrom(const void *ptr) const {
  DEFINE_OR_RETURN_ERROR(Allocation, allocation, allocationFrom(ptr));
  return allocation.byteSize -
         (reinterpret_cast<uintptr_t>(ptr) - allocation.base);
}

ErrorOr<Device::MemInfo> QuRTDevice::getMemoryInfo() const {
  // Shared with the host, so a device-local figure would be a fiction. Reported
  // as unknown, which is also what keeps the MemoryManager from pooling against
  // a budget that does not exist.
  return Device::MemInfo();
}

ErrorOr<void *> QuRTDevice::getCurrentDriverContext() const {
  return static_cast<void *>(nullptr);
}

ErrorOrSuccess QuRTDevice::setCurrentDriverContext(void *ctx) {
  // The transport keeps no per-thread context to switch.
  return success();
}

ErrorOr<DeviceContextRef> QuRTDevice::createContext() {
  return QuRTDeviceContextRef::create(*this);
}

DLDevice QuRTDevice::getDLDevice(bool /*pinned*/) const {
  // Device and host share the allocation, so the pinned variant is the same.
  return {kDLHexagon, id()};
}

//===----------------------------------------------------------------------===//
// QuRTDeviceStream
//===----------------------------------------------------------------------===//

QuRTDeviceStream::QuRTDeviceStream(QuRTDeviceContext &ctx)
    : DeviceStream(ctx) {}

ErrorOrSuccess QuRTDeviceStream::synchronize() {
  return const_cast<DeviceContext &>(context_()).synchronize();
}

ErrorOr<DeviceEventRef> QuRTDeviceStream::enqueueEvent() const {
  RETURN_ERROR("the on-device NPU transport has no events: an invocation "
               "completes before it returns");
}

ErrorOrSuccess QuRTDeviceStream::eventRecord(const DeviceEvent &event) const {
  return success();
}

ErrorOrSuccess QuRTDeviceStream::waitEvent(const DeviceEvent &event,
                                           unsigned int flags) const {
  return success();
}

ErrorOrSuccess
QuRTDeviceStream::enqueueWaitForEvent(const DeviceEvent &event) const {
  return success();
}

ErrorOrSuccess QuRTDeviceStream::enqueueFunctionExecDirect(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    void *attrs, uint32_t numAttrs, void **args, uint32_t argCount,
    const size_t *argSizes) const {
  return const_cast<DeviceContext &>(context_())
      .enqueueFunctionExecDirect(func, gridX, gridY, gridZ, blockX, blockY,
                                 blockZ, sharedMemBytes, attrs, numAttrs, args,
                                 argCount, argSizes);
}

void *QuRTDeviceStream::ptr() const { return nullptr; }

//===----------------------------------------------------------------------===//
// QuRTDeviceFunction
//===----------------------------------------------------------------------===//

QuRTDeviceFunction::QuRTDeviceFunction(QuRTDeviceContext &ctx, std::string name,
                                       uint32_t methodID)
    : DeviceFunction(ctx), name_(std::move(name)), methodID_(methodID) {}

ErrorOrSuccess QuRTDeviceFunction::copyToNamedConstant(const char *name,
                                                       const void *data,
                                                       size_t byteSize) {
  RETURN_ERROR("the on-device NPU skel exposes no named constants");
}

ErrorOr<int32_t> QuRTDeviceFunction::getAttribute(int32_t code) {
  RETURN_ERROR("the on-device NPU skel exposes no function attributes");
}

//===----------------------------------------------------------------------===//
// QuRTDeviceContext
//===----------------------------------------------------------------------===//

QuRTDeviceContext::QuRTDeviceContext(QuRTDevice &device)
    : DeviceContext(device, TypeID::get<QuRTDeviceContext>()) {}

QuRTDeviceContext::~QuRTDeviceContext() {
  // A launch held at teardown still has to run: the graph above may already
  // have read its output through a mapping this context does not mediate.
  // Sent before the queue is drained -- submit, then wait -- though the
  // two-transports refusal means at most one of these has work. `flushBatch`
  // also fires a held chain member, after which the process-wide holder must
  // stop naming this context.
  (void)flushBatch();
  if (gChainHolder == this) {
    gChainHolder = nullptr;
    gChainHoldNext = false;
  }
  // Before anything is freed: a frame still in flight names allocations this
  // context is about to hand back.
  (void)drainQueue();
  // After both, so it sees whatever either recorded. A pending error nobody
  // ever checked is the one real gap in the inherited mechanism: without this,
  // a witness guard that fired on the last flush of a process is dropped in
  // silence. A destructor cannot raise, so it shouts.
  ErrorOrSuccess dropped = checkPendingError();
  if (dropped.isError())
    fprintf(stderr,
            "npu witness: a deferred error was still pending at context "
            "teardown and no caller ever reported it: %s\n",
            dropped.getError());
  if (timingSlot_ != nullptr)
    (void)static_cast<QuRTDevice &>(device()).chunkFree(timingSlot_,
                                                        /*isHost=*/false);
  if (skelHandle_ == 0)
    return;
  ErrorOr<QuRTDriver &> driver = QuRTDriver::get();
  if (!driver.isError())
    driver->handleClose(skelHandle_);
}

ErrorOr<uint64_t> QuRTDeviceContext::skelHandle() {
  std::lock_guard<std::mutex> lock(mu_);
  if (skelHandle_ != 0)
    return skelHandle_;

  ErrorOr<QuRTDriver &> driver = QuRTDriver::get();
  if (driver.isError())
    RETURN_ERROR(driver.getError());

  // Must precede the open. A zero status only records the request; the device
  // can still refuse the load below. A refusal is not fatal on its own either,
  // since a signed catalogue loads without the request, so it is carried into
  // the open's diagnosis rather than returned here.
  QuRTUnsignedModule request = {static_cast<int>(kQuRTComputeDomainID), 1};
  uint32_t sessionStatus = static_cast<uint32_t>(driver->sessionControl(
      kQuRTControlUnsignedModule, &request, sizeof(request)));

  DEFINE_OR_RETURN_ERROR(const char *, rawURI, skelOpenURI());
  std::string uri(rawURI);
  // Stated before the open and unconditionally: which tuple a leg bound is
  // the first question of every on-device A/B, and a refused open must still
  // leave the URI it refused on record.
  fprintf(stderr, "npu skel open: %s\n", uri.c_str());

  uint64_t handle = 0;
  uint32_t status =
      static_cast<uint32_t>(driver->handleOpen(uri.c_str(), &handle));
  if (status != 0 || handle == 0)
    RETURN_ERROR(
        "the NPU refused to load the kernel library: " +
        qurtStatusMessage(status) +
        ". Its code-signing catalogue is regenerated on every build of it, so "
        "it must be signed again after one; an unsigned or stale catalogue is "
        "reported as AEE_EUNABLETOLOAD." +
        (sessionStatus == 0
             ? std::string()
             : " Unsigned-module support was refused too, with " +
                   qurtStatusMessage(sessionStatus) + "."));

  skelHandle_ = handle;

  // After the open, before any launch: the vote covers every dispatch this
  // session makes, and llama.cpp's backend places it at the same point --
  // session up, no op sent yet. Reports rather than fails: an unvoted
  // session is slower, not wrong, for every kernel except the HMX-capable
  // ones, whose dispatch `guardHmxDispatch` refuses by name against the
  // outcome latched here.
  if (npuPowerRequested()) {
    PowerVoteOutcome outcome = voteNPUPower(*driver, handle);
    powerVoteLive_ = outcome.live;
    powerVoteReason_ = outcome.reason;
    skelHmx_ = outcome.skelHmx;
  } else {
    powerVoteLive_ = false;
    powerVoteReason_ = "MODULAR_NPU_POWER=0";
    // Deliberately left unknown: resolving it costs a dispatch, and a session
    // that never launches an HMX-capable kernel should stay wire-identical to
    // the unvoted sessions every pre-vote record was taken on.
    skelHmx_ = -1;
  }

  return skelHandle_;
}

ErrorOrSuccess QuRTDeviceContext::guardHmxDispatch(const char *kernelName) {
  if (!npuKernelMayEngageHmx(kernelName))
    return success();

  // Opens the session if this is the first dispatch: the vote (or its
  // absence) is decided at open, and this launch would open it moments later
  // anyway.
  DEFINE_OR_RETURN_ERROR(uint64_t, handle, skelHandle());

  std::lock_guard<std::mutex> lock(mu_);
  if (powerVoteLive_)
    return success();

  // An unvoted session may still be fine: only an HMX build has a lock to
  // lose. With the vote off nothing has asked the skel yet, so ask now, once
  // -- `enable=0` places no vote and touches no HAP_power state (the device
  // body releases, which holds nothing here); a skel predating the row
  // refuses it harmlessly, and predates HMX with it.
  if (skelHmx_ == -1) {
    ErrorOr<QuRTDriver &> driver = QuRTDriver::get();
    if (driver.isError())
      RETURN_ERROR(driver.getError());
    uint32_t report[kNPUPowerVoteReportWords];
    uint32_t status =
        dispatchPowerVoteRow(*driver, handle, /*enable=*/0, report);
    skelHmx_ = status == 0 && report[5] == kNPUPowerVoteHmxMarker ? 1 : 0;
  }
  if (skelHmx_ != 1)
    return success();

  RETURN_ERROR("NPU kernel \"" + std::string(kernelName) +
               "\" routes to the HMX array in this kernel library, and this "
               "session's power vote is not live (" +
               powerVoteReason_ +
               "). An unvoted session's HMX lock is refused by the platform "
               "as QURT_EINVALID -- seven windows of refusals ended when the "
               "session vote alone flipped the lock to a grant (P5 M2.4 "
               "order 077). HMX requires the session power vote; "
               "MODULAR_NPU_POWER=0 disables HMX.");
}

namespace {

/// Hash of the destination's leading window before an operation overwrites
/// it, or 0 with recording off. Split from `recordHostOp` so the operation
/// itself stays between the two captures.
uint64_t preHashFor(const void *dst, size_t byteSize) {
  NPUDispatchRecorder *recorder = NPUDispatchRecorder::get();
  return recorder == nullptr ? 0 : recorder->windowHashOf(dst, byteSize);
}

/// Records one runtime-mediated host operation. The destination's allocation
/// is resolved through the device so the record carries a cross-run identity;
/// a destination outside every allocation (a host buffer) is recorded
/// without one.
void recordHostOp(QuRTDevice &device, const char *kind, const void *dst,
                  const void *src, size_t byteSize, uint64_t dstPreHash) {
  NPUDispatchRecorder *recorder = NPUDispatchRecorder::get();
  if (recorder == nullptr)
    return;
  uintptr_t base = 0;
  size_t allocBytes = 0;
  ErrorOr<QuRTDevice::Allocation> allocation = device.allocationFrom(dst);
  if (!allocation.isError()) {
    base = allocation.get().base;
    allocBytes = allocation.get().byteSize;
  }
  recorder->recordHostOp(kind, dst, base, allocBytes, src, byteSize,
                         dstPreHash);
}

} // namespace

ErrorOrSuccess QuRTDeviceContext::enqueueCopyToDevice(void *dst,
                                                      const void *src,
                                                      size_t byteSize) {
  // Device memory is this process's memory, so a host touch of it races any
  // launch still in flight -- in both directions, since a queued launch both
  // reads and writes. Sending the held batch and then draining the queue is
  // what keeps "the runtime moved these bytes" meaning the same thing on
  // every transport. No-ops with both off, and free in the graph path, where
  // every copy precedes the first launch of the step.
  RETURN_IF_ERROR(flushBatch());
  RETURN_IF_ERROR(drainQueue());
  RETURN_IF_ERROR(
      static_cast<QuRTDevice &>(device()).noteHostWrite(dst, src, byteSize));
  megastepForgetWrite(dst);
  uint64_t pre = preHashFor(dst, byteSize);
  // Device memory is mapped into this process, so the transfer is a copy within
  // one address space rather than a DMA across a bus.
  memcpy(dst, src, byteSize);
  recordHostOp(static_cast<QuRTDevice &>(device()), "copy_to", dst, src,
               byteSize, pre);
  return success();
}

ErrorOrSuccess QuRTDeviceContext::enqueueCopyFromDevice(void *dst,
                                                        const void *src,
                                                        size_t byteSize) {
  // The same barrier as `enqueueCopyToDevice`, for the same reason: submit
  // what is held, then wait for what is in flight.
  RETURN_IF_ERROR(flushBatch());
  // AFTER the flush, so a guard this very flush recorded is reported before
  // the caller reads the output it was recorded about.
  RETURN_IF_PENDING_ERROR(*this);
  RETURN_IF_ERROR(drainQueue());
  // The destination is normally a plain host buffer, but nothing enforces
  // that on a backend whose device memory is host-mapped, and a device-side
  // source is never declared constant -- so this can only ever withdraw.
  RETURN_IF_ERROR(
      static_cast<QuRTDevice &>(device()).noteHostWrite(dst, src, byteSize));
  megastepForgetWrite(dst);
  uint64_t pre = preHashFor(dst, byteSize);
  memcpy(dst, src, byteSize);
  recordHostOp(static_cast<QuRTDevice &>(device()), "copy_from", dst, src,
               byteSize, pre);
  return success();
}

ErrorOrSuccess QuRTDeviceContext::memCopyDtoD(DeviceBuffer &dst,
                                              DeviceBuffer &src,
                                              size_t srcByteOffset,
                                              size_t srcByteSize,
                                              bool crossStreamSync) {
  RETURN_IF_ERROR(flushBatch());
  RETURN_IF_ERROR(drainQueue());
  RETURN_IF_ERROR(static_cast<QuRTDevice &>(device()).noteHostWrite(
      dst.ptr(), static_cast<const uint8_t *>(src.ptr()) + srcByteOffset,
      srcByteSize));
  megastepForgetWrite(dst.ptr());
  uint64_t pre = preHashFor(dst.ptr(), srcByteSize);
  memcpy(dst.ptr(), static_cast<const uint8_t *>(src.ptr()) + srcByteOffset,
         srcByteSize);
  recordHostOp(static_cast<QuRTDevice &>(device()), "copy_dtod", dst.ptr(),
               static_cast<const uint8_t *>(src.ptr()) + srcByteOffset,
               srcByteSize, pre);
  return success();
}

ErrorOrSuccess QuRTDeviceContext::enqueueSetMemory(void *dst, size_t byteSize,
                                                   uint64_t val,
                                                   size_t valSize) {
  RETURN_IF_ERROR(flushBatch());
  RETURN_IF_ERROR(drainQueue());
  if (valSize == 0 || valSize > sizeof(val))
    RETURN_ERROR("NPU fill value must be 1 to " + std::to_string(sizeof(val)) +
                 " bytes wide, got " + std::to_string(valSize));
  if (byteSize % valSize != 0)
    RETURN_ERROR("NPU fill of " + std::to_string(byteSize) +
                 " bytes is not a whole number of " + std::to_string(valSize) +
                 "-byte values");
  RETURN_IF_ERROR(static_cast<QuRTDevice &>(device()).noteHostWrite(
      dst, /*src=*/nullptr, byteSize));
  megastepForgetWrite(dst);
  auto *out = static_cast<uint8_t *>(dst);
  uint64_t pre = preHashFor(dst, byteSize);
  for (size_t offset = 0; offset < byteSize; offset += valSize)
    memcpy(out + offset, &val, valSize);
  recordHostOp(static_cast<QuRTDevice &>(device()), "fill", dst,
               /*src=*/nullptr, byteSize, pre);
  return success();
}

ErrorOr<DeviceFunctionRef>
QuRTDeviceContext::loadFunction(StringRef moduleName, StringRef functionName,
                                StringRef code, int32_t maxDynamicSharedBytes,
                                StringRef debugLevel,
                                int32_t optimizationLevel) {
  if (findKernel(functionName) == nullptr) {
    std::string known;
    for (const NPUKernel &candidate : kNPUKernels)
      known += (known.empty() ? "" : ", ") + std::string(candidate.name);
    RETURN_ERROR("the NPU kernel library carries no kernel \"" +
                 functionName.str() + "\"; its manifest declares: " + known +
                 ". Adding one means rebuilding and re-signing the library.");
  }

  // `code` is ignored on purpose: the kernel is already in the device library,
  // and every kernel is reached through the same entry point, so the load is
  // the name resolution above and nothing else.
  return DeviceFunctionRef::take(
      new QuRTDeviceFunction(*this, functionName.str(), kNPUDispatchMethodID));
}

ErrorOrSuccess QuRTDeviceContext::enqueueFunctionExecDirect(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    void *attrs, uint32_t numAttrs, void **args, uint32_t argCount,
    const size_t *argSizes) {
  // Before the launch is packed: a guard recorded by the previous flush is
  // reported here, so a refused launch's output is never an operand of the
  // next one. This is the report site the flush defers to.
  RETURN_IF_PENDING_ERROR(*this);

  // First, so the host prep this function does — validating the geometry,
  // packing the scalars, resolving each operand's allocation — is inside the
  // measured span rather than silently outside it.
  NPUDispatchTimeline *timeline = NPUDispatchTimeline::get();
  uint64_t entryNanos = timeline ? NPUDispatchTimeline::nowNanos() : 0;

  // Refused by name, the way the queue already refuses the IO recorder and
  // the device timing report: every device leg moves one variable, and how a
  // held batch composes with a live queue is a design question (the M3.1-R
  // window's), not a precedence this function may quietly pick.
  if (npuBatchRequested() && npuQueueRequested())
    RETURN_ERROR("MODULAR_NPU_BATCH=1 and MODULAR_NPU_QUEUE=1 name two "
                 "transports for the same launch; set exactly one");
  if (npuMegastepRequested() && npuBatchRequested())
    RETURN_ERROR("MODULAR_NPU_MEGASTEP=1 and MODULAR_NPU_BATCH=1 name two "
                 "transports for the same launch; set exactly one of "
                 "MODULAR_NPU_MEGASTEP, MODULAR_NPU_BATCH, MODULAR_NPU_QUEUE");
  if (npuMegastepRequested() && npuQueueRequested())
    RETURN_ERROR("MODULAR_NPU_MEGASTEP=1 and MODULAR_NPU_QUEUE=1 name two "
                 "transports for the same launch; set exactly one of "
                 "MODULAR_NPU_MEGASTEP, MODULAR_NPU_BATCH, MODULAR_NPU_QUEUE");

  const NPUKernel *kernel = findKernel(func.name());
  if (kernel == nullptr)
    RETURN_ERROR("no manifest entry for NPU kernel \"" + func.name() + "\"");

  // Before the launch is packed, held, or queued, so every transport inherits
  // it: a kernel that may take the HMX array must not be dispatched into a
  // session whose power vote is not live -- the lock would refuse and the
  // kernel would write nothing, mysteriously (P5 M2.4 order 077).
  RETURN_IF_ERROR(guardHmxDispatch(kernel->name));

  // Geometry as the discrete part reads it: the grid counts NSP cores and the
  // block's X is a bitmap of the threads within one. A FastRPC session
  // addresses a single NSP, so the grid can only be that one core, while the
  // mask is carried to the device and spread over a thread per bit.
  if (gridX * gridY * gridZ > 1)
    RETURN_ERROR("the on-device NPU exposes one core per session, so a launch "
                 "grid of " +
                 std::to_string(gridX * gridY * gridZ) +
                 " cores cannot be dispatched");
  if (blockY > 1 || blockZ > 1)
    RETURN_ERROR("the on-device NPU takes its launch geometry as a thread "
                 "bitmap in the block's X extent, so Y and Z must be 1");

  // Deliberately not read as "every thread", which is what a zero mask means on
  // the discrete part: what a zero would have to expand to here is the
  // participant count, and that count is what the device sizes a barrier by and
  // what a kernel divides its range by. Naming it is the caller's business.
  uint32_t threadMask = blockX;
  if (threadMask == 0)
    RETURN_ERROR("an NPU launch names its threads as a bitmap in the block's X "
                 "extent, and 0 names none of them");
  if (llvm::popcount(threadMask) > static_cast<int>(kNPUMaxLaunchThreads))
    RETURN_ERROR("an NPU launch takes at most " +
                 std::to_string(kNPUMaxLaunchThreads) + " threads, and mask " +
                 std::to_string(threadMask) + " names " +
                 std::to_string(llvm::popcount(threadMask)));

  // Only a mask of the low bits, because a participant's launch index here is
  // its position among the participants while a bit's position on the discrete
  // part is a hardware thread number. The two agree exactly when the set bits
  // start at 0 with no gaps, which every mask in use does. A sparser one would
  // be renumbered instead of honoured, and a kernel that reads its index as a
  // hardware thread number would then do the wrong work without saying so.
  if ((threadMask & (threadMask + 1)) != 0)
    RETURN_ERROR("an NPU launch numbers its threads by position within the "
                 "mask, so the mask must set its low bits and no others; " +
                 std::to_string(threadMask) + " leaves a gap");

  uint32_t expected = kernel->bufferCount + kernel->scalarCount;
  if (argCount != expected)
    RETURN_ERROR("NPU kernel \"" + func.name() + "\" takes " +
                 std::to_string(expected) + " arguments, got " +
                 std::to_string(argCount));
  if (kernel->bufferCount > kNPUMaxBufferSlots)
    RETURN_ERROR("NPU kernel \"" + func.name() + "\" takes " +
                 std::to_string(kernel->bufferCount) +
                 " buffers, past the entry point's " +
                 std::to_string(kNPUMaxBufferSlots) + " slots");
  if (argSizes == nullptr && kernel->scalarCount > 0)
    RETURN_ERROR("dispatching NPU kernel \"" + func.name() +
                 "\" needs argument sizes: its scalars are packed by width");

  // Every scalar is widened into a slot of its own by copying into the low
  // bytes of a zeroed one, which assumes a little-endian host — as every target
  // this backend builds for is.
  static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__);
  // One slot per scalar, then one per buffer for its offset within the
  // allocation the device will map. The offsets ride here rather than in a new
  // interface parameter because the block is length-delimited and the device
  // checks its length with `>=`, so a longer one reaches an older skel
  // harmlessly and this needs no change to `mojo_npu.idl` -- whose md5 gates
  // whether a foreign host may invoke the skel at all.
  PendingLaunch pending;
  pending.name = kernel->name;
  pending.selector = selectorOf(*kernel);
  pending.threadMask = threadMask;
  pending.bufferCount = kernel->bufferCount;
  pending.scalarCount = kernel->scalarCount;
  pending.entryNanos = entryNanos;
  pending.scalars.assign(
      (kernel->scalarCount + kernel->bufferCount) * kNPUScalarSlotBytes, 0);
  for (uint32_t i = 0; i < kernel->scalarCount; ++i) {
    size_t byteSize = argSizes[kernel->bufferCount + i];
    if (byteSize == 0 || byteSize > kNPUScalarSlotBytes)
      RETURN_ERROR("scalar argument " + std::to_string(i) +
                   " of NPU kernel \"" + func.name() + "\" is " +
                   std::to_string(byteSize) +
                   " bytes wide, which does not fit a " +
                   std::to_string(kNPUScalarSlotBytes) + "-byte slot");
    memcpy(pending.scalars.data() + i * kNPUScalarSlotBytes,
           args[kernel->bufferCount + i], byteSize);
  }

  auto &npuDevice = static_cast<QuRTDevice &>(device());
  pending.allocations.resize(kernel->bufferCount);
  for (uint32_t i = 0; i < kernel->bufferCount; ++i) {
    void *buffer = nullptr;
    memcpy(&buffer, args[i], sizeof(buffer));

    // The allocation the address falls in, which is the granularity FastRPC
    // registers memory at; passing an address this device never handed out is
    // refused here. The device is given that allocation and told the offset
    // separately, because the offset does not survive the call: FastRPC hands
    // the far side the mapping's base, so a resolved interior address is read
    // and written at the wrong place -- measured, by dispatching base+N and
    // getting back the answer for base.
    DEFINE_OR_RETURN_ERROR(QuRTDevice::Allocation, allocation,
                           npuDevice.allocationFrom(buffer));
    uint64_t offset = reinterpret_cast<uintptr_t>(buffer) - allocation.base;
    memcpy(pending.scalars.data() +
               (kernel->scalarCount + i) * kNPUScalarSlotBytes,
           &offset, sizeof(offset));
    pending.allocations[i] = allocation;
  }

  // Map-once maps HERE, not at allocation. `fastrpc_mmap` is itself an RPC on
  // the domain, and a domain with no session open refuses it with AEE_ERPC --
  // order 163 measured 512 at every length and in both modes, from a driver
  // that mapped at allocation time. `QuRTDevice::chunkAlloc` cannot do better:
  // the device deliberately does not open the kernel library (`create`, :515),
  // so at allocation time there is no session to map into. This is the same
  // ordering the queue transport already gets on silicon -- `queueTransport`
  // opens the skel before creating the channel, and `mapAllocation` runs later
  // still, on the send path -- and the same one the reference gets, whose
  // session is open at backend init before any buffer exists.
  //
  // Consequence for the reference's fused alloc-and-map
  // (`ggml-hexagon.cpp:378`, brief section 4.2): we cannot fuse them, so the
  // invariant is held the other way. An allocation is unmapped until its first
  // launch and mapped after; there is no mapped-but-unallocated state because
  // teardown is address-exact in `chunkFree`, and no allocated-but-unmapped
  // launch because this runs before the transport fork below.
  if (npuMapOnceRequested()) {
    DEFINE_OR_RETURN_ERROR(uint64_t, session, skelHandle());
    (void)session;
    ErrorOr<QuRTDriver &> driver = QuRTDriver::get();
    if (driver.isError())
      RETURN_ERROR(driver.getError());
    for (const QuRTDevice::Allocation &allocation : pending.allocations)
      RETURN_IF_ERROR(npuDevice.mapAllocationOnce(
          *driver, reinterpret_cast<void *>(allocation.base),
          allocation.byteSize));
  }

  // The matmul-chain hold intercepts here, after the launch is fully packed
  // and mapped, so the held copy is exactly what `dispatchSingle` would have
  // sent -- an autonomous fire at a flush seam is byte-identical to the
  // launch the stub deferred.
  if (gChainHoldNext) {
    gChainHoldNext = false;
    gChainHolder = this;
    gChainHeldFired = false;
    chainHeld_ = std::move(pending);
    return success();
  }

  if (megastepping())
    return holdForMegastep(pending);
  if (!batching())
    return dispatchSingle(pending);

  // A launch may only name allocations this call declares, so what bounds a
  // batch is the union of its launches' allocations, not its length. One more
  // than fits sends what is held and opens a batch of its own.
  uint32_t slotBudget = kNPUMaxBufferSlots;
  if (timeline != nullptr && timeline->deviceTimingRequested())
    --slotBudget;

  std::vector<QuRTDevice::Allocation> trial = batchSlots_;
  std::vector<uint8_t> slotIndex;
  if (batch_.size() >= kNPUBatchMaxLaunches ||
      !assignSlots(pending, slotBudget, trial, slotIndex)) {
    RETURN_IF_ERROR(flushBatch());
    trial.clear();
    // A kernel with an operand per slot cannot share the call with the timing
    // report, so it goes alone rather than being refused.
    if (!assignSlots(pending, slotBudget, trial, slotIndex))
      return dispatchSingle(pending);
  }

  pending.slotIndex = std::move(slotIndex);
  batchSlots_ = std::move(trial);
  batch_.push_back(std::move(pending));
  return success();
}

bool QuRTDeviceContext::assignSlots(const PendingLaunch &launch,
                                    uint32_t budget,
                                    std::vector<QuRTDevice::Allocation> &slots,
                                    std::vector<uint8_t> &slotIndex) {
  slotIndex.assign(launch.bufferCount, 0);
  for (uint32_t i = 0; i < launch.bufferCount; ++i) {
    // Identified by base: an allocation is what FastRPC registers and maps, so
    // two operands inside one of them are one slot and one mapping, whatever
    // their offsets.
    size_t at = 0;
    while (at < slots.size() && slots[at].base != launch.allocations[i].base)
      ++at;
    if (at == slots.size()) {
      if (slots.size() >= budget)
        return false;
      slots.push_back(launch.allocations[i]);
    }
    slotIndex[i] = static_cast<uint8_t>(at);
  }
  return true;
}

ErrorOrSuccess QuRTDeviceContext::dispatchSingle(PendingLaunch &launch) {
  NPUDispatchTimeline *timeline = NPUDispatchTimeline::get();

  // Mirrors the primary-input block the interface definition generates for the
  // entry point: its two plain arguments, the scalar block's own length, then a
  // length per buffer slot — twice each, since a slot is declared in both
  // directions and carries one length per direction.
  llvm::SmallVector<uint32_t, 19> primIn(3 + 2 * kNPUMaxBufferSlots, 0);
  primIn[0] = launch.selector;
  primIn[1] = launch.threadMask;
  primIn[2] = static_cast<uint32_t>(launch.scalars.size());

  // The block, then every buffer slot in both directions. Unused slots stay
  // empty: the entry point declares a fixed run of them and a kernel takes the
  // leading ones.
  llvm::SmallVector<QuRTRemoteBuf, 18> slots(2 + 2 * kNPUMaxBufferSlots,
                                             QuRTRemoteBuf{nullptr, 0});
  slots[0] = {primIn.data(), primIn.size() * sizeof(uint32_t)};
  slots[1] = {launch.scalars.data(), launch.scalars.size()};

  for (uint32_t i = 0; i < launch.bufferCount; ++i) {
    // Fits a length field because a single allocation is capped well below 4
    // GiB; see `chunkAlloc`.
    const QuRTDevice::Allocation &allocation = launch.allocations[i];
    auto *base = reinterpret_cast<void *>(allocation.base);
    primIn[3 + 2 * i] = static_cast<uint32_t>(allocation.byteSize);
    primIn[4 + 2 * i] = static_cast<uint32_t>(allocation.byteSize);
    slots[2 + i] = {base, allocation.byteSize};
    slots[2 + kNPUMaxBufferSlots + i] = {base, allocation.byteSize};
  }

  // The skel writes its timing report into the slot after the kernel's own
  // buffers, and reads the launch word's instrumentation bits to know it was
  // asked. No registered kernel takes more than six of the eight slots, so
  // that slot is always free -- but the manifest is what says so, and a kernel
  // appended with eight would go untimed rather than have an operand
  // displaced. The participant mask is validated above in its own right, so a
  // geometry that would be refused is still refused with the flag set.
  NPULaunchDeviceTiming deviceTiming = {};
  NPULaunchPhases devicePhases = {};
  NPULaunchSyncReport deviceSync = {};
  NPULaunchPipeReport devicePipe = {};
  bool deviceTimed = false;
  uint32_t reportBytes = 0;
  if (timeline != nullptr && timeline->deviceTimingRequested() &&
      launch.bufferCount < kNPUMaxBufferSlots) {
    DEFINE_OR_RETURN_ERROR(void *, report, timingSlot());
    // Only the single-dispatch path asks for phases. A batch and a megastep run
    // several kernels behind one report and the device's record is
    // launch-scoped, so a phase block there would be a superposition of kernels
    // with no way to attribute it -- worse than absent.
    reportBytes = timeline->devicePhasesRequested() ? kNPUTimingSlotBytes
                                                    : kNPUStampSlotBytes;
    if (timeline->devicePipeRequested())
      reportBytes += kNPUPipeSlotExtraBytes;
    memset(report, 0, reportBytes);
    uint32_t slot = launch.bufferCount;
    primIn[3 + 2 * slot] = reportBytes;
    primIn[4 + 2 * slot] = reportBytes;
    slots[2 + slot] = {report, reportBytes};
    slots[2 + kNPUMaxBufferSlots + slot] = {report, reportBytes};
    // Deliberately no offset slot for it: the device folds offsets back only
    // for the kernel's own buffers, so the report lands at the mapping's base,
    // which is this allocation's base.
    primIn[1] |= kNPULaunchTime;
    deviceTimed = true;
  }
  uint32_t wireMask = primIn[1];

  if (npuQueueRequested()) {
    // The instruments below bracket a launch that has already finished when the
    // call returns, which is exactly what the queue stops being true. Refused
    // rather than silently reporting the wrong thing: an IO recorder would hash
    // buffers a launch has not written yet, and the device timing report would
    // be read before the device wrote it.
    if (NPUDispatchRecorder::get() != nullptr)
      RETURN_ERROR("the NPU IO recorder needs a launch to be finished when the "
                   "dispatch returns, which MODULAR_NPU_QUEUE=1 makes untrue; "
                   "record over the synchronous transport");
    if (deviceTimed)
      RETURN_ERROR("the NPU device timing report is read when the dispatch "
                   "returns, which MODULAR_NPU_QUEUE=1 makes too early; time "
                   "over the synchronous transport");

    // The same operands the slot list carries, as base and length -- so the
    // frame declares the bytes the per-call path declared, and the union of a
    // frame's launches is the maintenance one frame does instead of many. The
    // class rides along: a reference the loader's declarations classify
    // constant gets a descriptor with no maintenance flags, which is the
    // per-frame O(model)-to-O(frame) saving the descriptor split exists for.
    auto &npuDevice = static_cast<QuRTDevice &>(device());
    llvm::SmallVector<NPUQueueBufferRef, 8> refs;
    for (uint32_t i = 0; i < launch.bufferCount; ++i)
      refs.push_back({slots[2 + i].pv, slots[2 + i].nLen,
                      npuDevice.isDeclaredConstant(slots[2 + i].pv)});

    NPUQueueLaunch queued;
    queued.selector = launch.selector;
    queued.launchWord = wireMask;
    queued.scalars = launch.scalars.data();
    queued.scalarBytes = launch.scalars.size();
    queued.buffers = refs.data();
    queued.bufferCount = refs.size();

    DEFINE_OR_RETURN_ERROR(NPUQueueTransport *, transport, queueTransport());
    uint64_t preEnqueue = timeline ? NPUDispatchTimeline::nowNanos() : 0;
    RETURN_IF_ERROR(transport->enqueue(queued));
    if (timeline != nullptr)
      timeline->recordLaunch(launch.name, launch.selector, wireMask,
                             /*status=*/0, launch.entryNanos, preEnqueue,
                             NPUDispatchTimeline::nowNanos(),
                             NPUDispatchTimeline::nowNanos(),
                             /*deviceTiming=*/nullptr);
    return success();
  }

  DEFINE_OR_RETURN_ERROR(uint64_t, handle, skelHandle());
  ErrorOr<QuRTDriver &> driver = QuRTDriver::get();
  if (driver.isError())
    RETURN_ERROR(driver.getError());

  // Bracketing the invocation is exact here because it is synchronous: the
  // post-hashes are of what this launch left behind, with nothing in flight.
  NPUDispatchRecorder *recorder = NPUDispatchRecorder::get();
  uint64_t recordToken = 0;
  if (recorder != nullptr)
    recordToken = recorder->beginLaunch(
        launch.name, launch.selector, wireMask, launch.scalars.data(),
        launch.scalars.size(), launch.scalarCount, launch.bufferCount,
        slots.data() + 2);

  // Immediately around the call and nothing else, so what lands in `invoke` is
  // the round trip: the IO recorder's hashing, when it is also on, is charged
  // to prep and post where the summary tool can say so.
  uint64_t preInvoke = timeline ? NPUDispatchTimeline::nowNanos() : 0;
  uint32_t status = static_cast<uint32_t>(
      driver->handleInvoke(handle,
                           qurtScalars(kNPUDispatchMethodID,
                                       /*inBufs=*/2 + kNPUMaxBufferSlots,
                                       /*outBufs=*/kNPUMaxBufferSlots),
                           slots.data()));
  uint64_t postInvoke = timeline ? NPUDispatchTimeline::nowNanos() : 0;

  bool devicePhased = false;
  bool deviceSynced = false;
  bool devicePiped = false;
  if (deviceTimed) {
    const auto *reportBase =
        static_cast<const uint8_t *>(slots[2 + launch.bufferCount].pv);
    memcpy(&deviceTiming, reportBase, sizeof(deviceTiming));
    if (reportBytes >= kNPUTimingSlotBytes) {
      memcpy(&devicePhases, reportBase + sizeof(deviceTiming),
             sizeof(devicePhases));
      // A skel built without the probe leaves the block as this host zeroed it.
      // Reported as absent rather than as a set of zeros, which would otherwise
      // read as a participant that ran and did nothing.
      for (uint32_t t = 0; t < kNPUPhaseMaxThreads; ++t)
        devicePhased |= devicePhases.thread[t].marks != 0;
    }
    // The sync block sits after the phase record when the slot held it whole
    // and directly after the stamps when not -- the same size-decides-layout
    // rule the skel writes by. The magic answers for skels that never wrote
    // it: the block absent beats a session that looks idle.
    uint32_t syncOffset =
        sizeof(deviceTiming) +
        (reportBytes >= kNPUTimingSlotBytes ? sizeof(devicePhases) : 0);
    if (reportBytes >= syncOffset + sizeof(deviceSync)) {
      memcpy(&deviceSync, reportBase + syncOffset, sizeof(deviceSync));
      deviceSynced = deviceSync.magic == kNPUSyncReportMagic &&
                     deviceSync.version == kNPUSyncReportVersion;
    }
    // The pipe block sits third, and only the pipe-egress request sizes the
    // slot to reach it. Same magic rule again: a skel built without the
    // pipe-array flag leaves this host's zeroes, and absent beats a ledger
    // that looks idle.
    uint32_t pipeOffset = syncOffset + sizeof(deviceSync);
    if (reportBytes >= pipeOffset + sizeof(devicePipe)) {
      memcpy(&devicePipe, reportBase + pipeOffset, sizeof(devicePipe));
      devicePiped = devicePipe.magic == kNPUPipeReportMagic &&
                    devicePipe.version == kNPUPipeReportVersion;
    }
  }
  if (recorder != nullptr)
    recorder->endLaunch(recordToken, status);
  if (timeline != nullptr)
    timeline->recordLaunch(launch.name, launch.selector, wireMask, status,
                           launch.entryNanos, preInvoke, postInvoke,
                           NPUDispatchTimeline::nowNanos(),
                           deviceTimed ? &deviceTiming : nullptr,
                           devicePhased ? &devicePhases : nullptr,
                           deviceSynced ? &deviceSync : nullptr,
                           devicePiped ? &devicePipe : nullptr);
  if (status != 0)
    RETURN_ERROR("NPU invocation of \"" + std::string(launch.name) +
                 "\" failed: " + qurtStatusMessage(status));
  // Recorded rather than returned even here, where this call could raise it:
  // one policy for both transports, so a guard's report does not change shape
  // when `MODULAR_NPU_BATCH` is set. The next transport call reports it, which
  // on this path is at most one op later and still ahead of any consumer.
  RECORD_PENDING_ERROR(*this, evaluateWitnessGuards(launch, 0, 1));
  return success();
}

ErrorOr<NPUQueueTransport *> QuRTDeviceContext::queueTransport() {
  // Opened before the lock, not under it: the queue is handed to the skel
  // through `start`, a method on this handle, and `skelHandle` takes the same
  // mutex.
  DEFINE_OR_RETURN_ERROR(uint64_t, handle, skelHandle());
  std::lock_guard<std::mutex> lock(mu_);
  if (queue_ == nullptr) {
    DEFINE_OR_RETURN_ERROR(std::unique_ptr<NPUQueueChannel>, channel,
                           createNPUDspQueueChannel(handle));
    queue_ = std::make_unique<NPUQueueTransport>(std::move(channel));
  }
  return queue_.get();
}

ErrorOrSuccess QuRTDeviceContext::drainQueue() {
  std::lock_guard<std::mutex> lock(mu_);
  if (queue_ == nullptr)
    return success();
  return queue_->drain();
}

ErrorOrSuccess QuRTDeviceContext::fireHeldChain() {
  if (!chainHeld_)
    return success();
  PendingLaunch launch = std::move(*chainHeld_);
  chainHeld_.reset();
  gChainHeldFired = true;
  // `dispatchSingle` evaluates the held member's own witness guards and
  // records any refusal through the pending-error latch, exactly as an
  // un-deferred singleton would have.
  return dispatchSingle(launch);
}

bool QuRTDeviceContext::takeHeldChain() {
  if (!chainHeld_)
    return false;
  chainHeld_.reset();
  return true;
}

extern "C" {
/// The next `enqueueFunctionExecDirect` on this process's context is held
/// instead of dispatched. One-shot; the stub sets it immediately before the
/// enqueue it defers.
void npuChainHoldNext(void) { gChainHoldNext = true; }

/// Claims the held launch back for pairing. 1: claimed (the caller owns the
/// pair dispatch; the held copy is discarded). 2: a flush seam already fired
/// it (the stub's record is stale; treat the incoming matmul as fresh).
/// 0: nothing held and nothing fired -- the stub and this side disagree,
/// which the stub treats as "flushed" and re-records.
uint32_t npuChainTakeHeld(void) {
  gChainHoldNext = false;
  if (gChainHolder != nullptr && gChainHolder->takeHeldChain())
    return 1;
  if (gChainHeldFired) {
    gChainHeldFired = false;
    return 2;
  }
  return 0;
}

/// Fires the held launch now, if any. Nonzero only on a hard dispatch error;
/// witness-guard refusals ride the pending-error latch inside
/// `dispatchSingle` and surface at the next context call, the same delivery
/// every flushed launch gets.
uint32_t npuChainFire(void) {
  gChainHoldNext = false;
  if (gChainHolder == nullptr)
    return 0;
  ErrorOrSuccess result = gChainHolder->fireHeldChain();
  if (result.isError()) {
    fprintf(stderr, "npu matmul chain: held member failed at fire: %s\n",
            result.getError());
    return 1;
  }
  return 0;
}
} // extern "C"

ErrorOrSuccess
QuRTDeviceContext::evaluateWitnessGuards(const PendingLaunch &launch,
                                         size_t position, size_t width) {
  if (launch.bufferCount == 0)
    return success();
  // The chain row (`matmul_chain2`) carries one witness PER MEMBER -- slots 3
  // and 7 -- and the transport must walk both or member 0's correctness guard
  // is dead. Member 0 walks FIRST so a fail-fast names the earliest failing
  // member, the same rule the device batch applies by stopping at its
  // refusing record.
  static const NPUKernel *chainRow = findKernel("matmul_chain2");
  if (chainRow != nullptr && launch.selector == selectorOf(*chainRow) &&
      launch.bufferCount == chainRow->bufferCount)
    RETURN_IF_ERROR(evaluateWitnessGuardsAt(launch, 3, position, width));
  return evaluateWitnessGuardsAt(launch, launch.bufferCount - 1, position,
                                 width);
}

ErrorOrSuccess
QuRTDeviceContext::evaluateWitnessGuardsAt(const PendingLaunch &launch,
                                           uint32_t operand, size_t position,
                                           size_t width) {
  // The witness is every NPU kernel's last buffer slot, by the `NPU_KERNEL`
  // convention the dispatch recorder already relies on -- except the chain
  // row's member-0 slot, which the wrapper above names explicitly.
  const QuRTDevice::Allocation &allocation = launch.allocations[operand];
  uint64_t offset = 0;
  memcpy(&offset,
         launch.scalars.data() +
             (launch.scalarCount + operand) * kNPUScalarSlotBytes,
         sizeof(offset));
  if (offset + kNPUArenaWitnessBytes + kNPUArenaGuardBytes >
      allocation.byteSize)
    return success();

  const auto *witness = reinterpret_cast<const uint32_t *>(
      static_cast<uintptr_t>(allocation.base + offset));
  const uint32_t *record = witness + kNPUArenaWitnessBytes / sizeof(uint32_t);
  // Not an arena slot: a hand driver's witness, a test's, or a kernel appended
  // with a trailing buffer that is not one. Nothing was armed, so nothing is
  // checked -- which is the whole reason the record carries a magic.
  if (record[kNPUGuardRecordMagicWord] != kNPUGuardMagic)
    return success();

  // The arena's own header, checked here rather than at creation because this
  // is the first point both sides exist. A stub whose arena is shorter than
  // the batch cap would hand two in-flight launches one slot, and slot 7 is a
  // cross-thread channel inside the HMX arm's launch -- so that is a corrupted
  // rendezvous, not merely a confused report.
  const auto *header = reinterpret_cast<const uint32_t *>(
      static_cast<uintptr_t>(allocation.base));
  if (header[kNPUArenaMagicWord] != kNPUArenaMagic)
    RETURN_ERROR("a witness guard was armed in a buffer that is not a witness "
                 "arena; the host stub and this transport disagree about the "
                 "arena layout");
  if (header[kNPUArenaSlotsWord] < kNPUBatchMaxLaunches)
    RETURN_ERROR("the witness arena carries " +
                 std::to_string(header[kNPUArenaSlotsWord]) +
                 " slots against a batch cap of " +
                 std::to_string(kNPUBatchMaxLaunches) +
                 "; two launches of one batch would share a slot");
  if (header[kNPUArenaWitnessBytesWord] != kNPUArenaWitnessBytes)
    RETURN_ERROR("the witness arena declares a " +
                 std::to_string(header[kNPUArenaWitnessBytesWord]) +
                 "-byte witness against this transport's " +
                 std::to_string(kNPUArenaWitnessBytes));
  // The stride is what turns an offset back into a slot index, and a
  // disagreement here would put every guard record in the wrong place while
  // both magics still matched.
  if (header[kNPUArenaStrideWord] !=
      kNPUArenaWitnessBytes + kNPUArenaGuardBytes)
    RETURN_ERROR("the witness arena strides its slots every " +
                 std::to_string(header[kNPUArenaStrideWord]) +
                 " bytes against this transport's " +
                 std::to_string(kNPUArenaWitnessBytes + kNPUArenaGuardBytes));

  // The originating op, which is the whole product of arming a guard per
  // launch: the kernel name is `matmul` for `mo.matmul`, `mo.batch_matmul` and
  // `mo.conv` alike, and naming the batch names none of them.
  std::string op = launch.name;
  uint64_t opAddress =
      static_cast<uint64_t>(record[kNPUGuardRecordOpPtrWord]) |
      (static_cast<uint64_t>(record[kNPUGuardRecordOpPtrWord + 1]) << 32);
  uint32_t opLength = record[kNPUGuardRecordOpLenWord];
  if (opAddress != 0 && opLength > 0 && opLength <= kNPUGuardOpNameMax)
    op.assign(reinterpret_cast<const char *>(static_cast<uintptr_t>(opAddress)),
              opLength);

  uint32_t slot = static_cast<uint32_t>(
      offset / (kNPUArenaWitnessBytes + kNPUArenaGuardBytes));
  uint32_t guards =
      std::min<uint32_t>(record[kNPUGuardRecordCountWord], kNPUGuardMax);
  for (uint32_t g = 0; g < guards; ++g) {
    const uint32_t *guard = record + kNPUGuardHeaderWords + g * kNPUGuardWords;
    uint32_t word = guard[kNPUGuardWordWord];
    uint32_t condWord = guard[kNPUGuardCondWordWord];
    if (word >= kNPUArenaWitnessBytes / sizeof(uint32_t))
      continue;
    if (condWord != kNPUGuardCondNone) {
      if (condWord >= kNPUArenaWitnessBytes / sizeof(uint32_t))
        continue;
      if (witness[condWord] != guard[kNPUGuardCondValueWord])
        continue;
    }

    uint32_t value = witness[word];
    uint32_t accepts = std::min<uint32_t>(guard[kNPUGuardAcceptCountWord],
                                          kNPUGuardMaxAccepts);
    bool accepted = false;
    for (uint32_t a = 0; a < accepts; ++a)
      accepted = accepted || value == guard[kNPUGuardAcceptsWord + a];
    if (accepted)
      continue;

    std::string what =
        op + " on the on-device NPU: kernel \"" + std::string(launch.name) +
        "\" left witness word " + std::to_string(word) + " = " +
        std::to_string(value) + ", which this host does not accept (launch " +
        std::to_string(position + 1) + " of " + std::to_string(width) +
        " in the flush, arena slot " + std::to_string(slot) + ")";
    if (guard[kNPUGuardClassWord] == kNPUGuardClassCorrectness) {
      // Fail-fast on the first one, matching what the transport already does
      // when the device DOES raise the status: `mojo_npu_imp.c` stops the
      // device batch at the refusing record, so the remaining launches never
      // ran either. The two refusal families then look identical to the graph.
      RETURN_ERROR(what + "; the kernel refused the unit and wrote no output");
    }
    fprintf(stderr, "npu witness: %s\n", what.c_str());
  }
  return success();
}

ErrorOrSuccess QuRTDeviceContext::flushBatch() {
  // The chain hold drains ahead of everything: this is called from every
  // entry point through which the host can observe device memory, and a held
  // chain member is a launch whose output has not been written yet.
  RETURN_IF_ERROR(fireHeldChain());
  // The raw latch, not `megastepping()`: this runs on every flush call site,
  // including allocation before any dispatch, and must not decide the mode as
  // a side effect. Launches are only ever held after a hold path latched it.
  if (megastepping_ == 1)
    return flushMegastep();
  if (batch_.empty())
    return success();

  // Taken before anything can fail, so a flush that errors leaves nothing held
  // to be sent a second time by the next one.
  std::vector<PendingLaunch> launches;
  std::vector<QuRTDevice::Allocation> slotAllocations;
  launches.swap(batch_);
  slotAllocations.swap(batchSlots_);

  // A batch of one is a single dispatch, and sending it as one keeps the
  // common case on the wire that has silicon behind it.
  if (launches.size() == 1)
    return dispatchSingle(launches[0]);

  // The header, then a record per launch, then each launch's own argument area
  // — its scalars followed by one offset per buffer, byte for byte the block a
  // single dispatch sends, so the kernel reads identical arguments either way.
  std::vector<uint8_t> block(
      kNPUBatchHeaderBytes + launches.size() * kNPUBatchRecordBytes, 0);
  for (size_t i = 0; i < launches.size(); ++i) {
    PendingLaunch &launch = launches[i];
    auto argsOffset = static_cast<uint32_t>(block.size());
    block.insert(block.end(), launch.scalars.begin(), launch.scalars.end());

    uint32_t slotMap = 0;
    for (uint32_t j = 0; j < launch.bufferCount; ++j)
      slotMap |= static_cast<uint32_t>(launch.slotIndex[j]) << (4 * j);

    uint8_t *record =
        block.data() + kNPUBatchHeaderBytes + i * kNPUBatchRecordBytes;
    writeWireWord(record, launch.selector);
    writeWireWord(record + 4, launch.threadMask);
    writeWireWord(record + 8, argsOffset);
    writeWireWord(record + 12, slotMap);
    // The arity the host laid this record out with. The device checks it
    // against its own expansion of the manifest, so a host built against a
    // newer one is refused rather than read at the wrong offsets.
    writeWireWord(record + 16, launch.bufferCount | (launch.scalarCount << 16));
  }

  llvm::SmallVector<uint32_t, 19> primIn(3 + 2 * kNPUMaxBufferSlots, 0);
  primIn[0] = kNPUBatchSelector;
  // The device takes each launch's geometry from its own record; the low byte
  // carries the first launch's so the word is never one an older skel would
  // read as naming no threads.
  primIn[1] = kNPULaunchBatch | (launches.front().threadMask & 0xFFu);
  primIn[2] = static_cast<uint32_t>(block.size());

  llvm::SmallVector<QuRTRemoteBuf, 18> slots(2 + 2 * kNPUMaxBufferSlots,
                                             QuRTRemoteBuf{nullptr, 0});
  slots[0] = {primIn.data(), primIn.size() * sizeof(uint32_t)};
  for (size_t i = 0; i < slotAllocations.size(); ++i) {
    auto *base = reinterpret_cast<void *>(slotAllocations[i].base);
    size_t byteSize = slotAllocations[i].byteSize;
    primIn[3 + 2 * i] = static_cast<uint32_t>(byteSize);
    primIn[4 + 2 * i] = static_cast<uint32_t>(byteSize);
    slots[2 + i] = {base, byteSize};
    slots[2 + kNPUMaxBufferSlots + i] = {base, byteSize};
  }

  NPUDispatchTimeline *timeline = NPUDispatchTimeline::get();
  NPULaunchDeviceTiming deviceTiming = {};
  uint32_t reportSlot = kNPUBatchNoSlot;
  if (timeline != nullptr && timeline->deviceTimingRequested() &&
      slotAllocations.size() < kNPUMaxBufferSlots) {
    DEFINE_OR_RETURN_ERROR(void *, report, timingSlot());
    memset(report, 0, sizeof(NPULaunchDeviceTiming));
    reportSlot = static_cast<uint32_t>(slotAllocations.size());
    primIn[3 + 2 * reportSlot] = sizeof(NPULaunchDeviceTiming);
    primIn[4 + 2 * reportSlot] = sizeof(NPULaunchDeviceTiming);
    slots[2 + reportSlot] = {report, sizeof(NPULaunchDeviceTiming)};
    slots[2 + kNPUMaxBufferSlots + reportSlot] = {
        report, sizeof(NPULaunchDeviceTiming)};
    primIn[1] |= kNPULaunchTime;
  }

  // Last, so the report slot is known: the block is addressed by `slots[1]`
  // below and nothing resizes it after this.
  writeWireWord(block.data(), kNPUBatchMagic);
  writeWireWord(block.data() + 4, static_cast<uint32_t>(launches.size()));
  writeWireWord(block.data() + 8, reportSlot);
  slots[1] = {block.data(), block.size()};
  uint32_t wireMask = primIn[1];

  DEFINE_OR_RETURN_ERROR(uint64_t, handle, skelHandle());
  ErrorOr<QuRTDriver &> driver = QuRTDriver::get();
  if (driver.isError())
    RETURN_ERROR(driver.getError());

  uint64_t preInvoke = timeline ? NPUDispatchTimeline::nowNanos() : 0;
  uint32_t status = static_cast<uint32_t>(
      driver->handleInvoke(handle,
                           qurtScalars(kNPUDispatchMethodID,
                                       /*inBufs=*/2 + kNPUMaxBufferSlots,
                                       /*outBufs=*/kNPUMaxBufferSlots),
                           slots.data()));
  uint64_t postInvoke = timeline ? NPUDispatchTimeline::nowNanos() : 0;

  if (reportSlot != kNPUBatchNoSlot)
    memcpy(&deviceTiming, slots[2 + reportSlot].pv, sizeof(deviceTiming));
  if (timeline != nullptr)
    // One record for the batch, from the first held launch's entry: the gap
    // before it is device idle exactly as it is for a single dispatch, while
    // the launches inside it have no separately observable time.
    timeline->recordLaunch("batch", kNPUBatchSelector, wireMask, status,
                           launches.front().entryNanos, preInvoke, postInvoke,
                           NPUDispatchTimeline::nowNanos(),
                           reportSlot != kNPUBatchNoSlot ? &deviceTiming
                                                         : nullptr);
  if (status != 0)
    RETURN_ERROR("NPU invocation of a batch of " +
                 std::to_string(launches.size()) +
                 " launches failed: " + qurtStatusMessage(status));

  // THE FLUSH DETECTS AND RECORDS; THE NEXT RAISING CALLER REPORTS. Not
  // returned from here: this runs on every flush call site, including
  // `memFree`'s, where the free can come from a destructor that cannot raise
  // (`:2176`). The sticky slot is the base class's, first-error-wins, and it is
  // reported at the `RETURN_IF_PENDING_ERROR` that opens the next launch, copy
  // or synchronize -- so no consumer can observe the refused launch's output,
  // which is the property that makes deferral safe rather than weaker.
  for (size_t i = 0; i < launches.size(); ++i)
    RECORD_PENDING_ERROR(
        *this, evaluateWitnessGuards(launches[i], i, launches.size()));
  return success();
}

bool QuRTDeviceContext::batching() {
  // LATCHED PER CONTEXT, ON ITS FIRST LAUNCH -- while `npuBatchRequested()`
  // re-reads `getenv` every call. So a test that flips MODULAR_NPU_BATCH
  // between two arms and reuses one context silently runs BOTH on whichever
  // transport the first launch latched, and reads as a clean A/B. Give each
  // arm its own context (`createContext()`), or fork, as the queue tests do
  // for their own process-level bindings. Cost of not knowing: one false red
  // and nearly one false green in `npu_witness_alloc_count_test`.
  if (batching_ < 0) {
    // Off while the IO recorder is on, and not as a convenience: the recorder
    // hashes each operand either side of the launch that touched it, which a
    // batch has no boundary for. It is also the reference leg of the
    // batched-versus-per-call comparison, so it has to keep sending one launch
    // per call.
    batching_ =
        (npuBatchRequested() && NPUDispatchRecorder::get() == nullptr) ? 1 : 0;
    // The device leg's positive self-report: a treatment leg that cannot say
    // it is the treatment gets compared against nothing.
    if (batching_ == 1)
      fprintf(stderr, "npu batch: active (MODULAR_NPU_BATCH=1)\n");
  }
  return batching_ != 0;
}

bool QuRTDeviceContext::megastepping() {
  if (megastepping_ < 0)
    // Off while the IO recorder is on, exactly as batching is: the recorder
    // brackets one finished launch per call, which a table has no boundary
    // for, and the per-call leg it keeps sending is the reference the tabled
    // leg is compared against.
    megastepping_ =
        (npuMegastepRequested() && NPUDispatchRecorder::get() == nullptr) ? 1
                                                                          : 0;
  return megastepping_ != 0;
}

ErrorOrSuccess QuRTDeviceContext::holdForMegastep(PendingLaunch &pending) {
  // A flush error can be swallowed upstream -- a free from a destructor
  // cannot raise -- and a render that continues past one computes from
  // kernels that never ran. Refusing the next launch is what turns that into
  // a loud stop instead of an all-zero output (order 087's failure mode).
  if (!megastepPoison_.empty())
    RETURN_ERROR("an earlier megastep flush failed and its error could not "
                 "be raised where it happened: " +
                 megastepPoison_ +
                 " -- refusing further launches rather than rendering from "
                 "kernels that never ran");
  if (batch_.size() >= kNPUMegastepMaxLaunches)
    RETURN_IF_ERROR(flushMegastep());

  auto &npuDevice = static_cast<QuRTDevice &>(device());
  uint32_t slotBudget = kNPUMaxBufferSlots;
  NPUDispatchTimeline *timeline = NPUDispatchTimeline::get();
  if (timeline != nullptr && timeline->deviceTimingRequested())
    --slotBudget;

  // Two attempts, the second against a table just flushed empty. A launch's
  // maintained operands are bounded by its own arity, which the manifest caps
  // under the slots, so the retry can only fail for a kernel wider than the
  // interface -- which the arity checks refused long before here.
  for (int attempt = 0; attempt < 2; ++attempt) {
    std::vector<QuRTDevice::Allocation> trial = megastepSlots_;
    std::vector<std::pair<uint16_t, QuRTDevice::Allocation>> newEntries;
    std::vector<uint16_t> refs(pending.bufferCount, 0);
    uint16_t nextIndex = megastepNextIndex_;
    bool fits = true;

    for (uint32_t i = 0; i < pending.bufferCount; ++i) {
      const QuRTDevice::Allocation &allocation = pending.allocations[i];

      // A session-table reference is sound only for an allocation nothing
      // will write again: classified init-constant NOW (the class is
      // revocable, and `megastepForgetWrite` drops the entry when it is), and
      // registered while the table has room. Everything else takes a
      // maintained slot of the call itself.
      if (npuDevice.isDeclaredConstant(
              reinterpret_cast<void *>(allocation.base))) {
        auto it = megastepIndexByBase_.find(allocation.base);
        if (it != megastepIndexByBase_.end()) {
          refs[i] = kNPUMegastepRefTable | it->second;
          continue;
        }
        uint16_t pendingIndex = kNPUMegastepTableEntries;
        for (const auto &entry : newEntries)
          if (entry.second.base == allocation.base)
            pendingIndex = entry.first;
        if (pendingIndex != kNPUMegastepTableEntries) {
          refs[i] = kNPUMegastepRefTable | pendingIndex;
          continue;
        }
        if (nextIndex < kNPUMegastepTableEntries) {
          refs[i] = kNPUMegastepRefTable | nextIndex;
          newEntries.push_back({nextIndex, allocation});
          ++nextIndex;
          continue;
        }
        // Table exhausted: the allocation rides a maintained slot instead.
        // Indices are never reused, so exhaustion degrades throughput, not
        // correctness.
      }

      size_t at = 0;
      while (at < trial.size() && trial[at].base != allocation.base)
        ++at;
      if (at == trial.size()) {
        if (trial.size() >= slotBudget) {
          fits = false;
          break;
        }
        trial.push_back(allocation);
      }
      refs[i] = static_cast<uint16_t>(at);
    }

    if (fits) {
      for (const auto &entry : newEntries) {
        megastepIndexByBase_[entry.second.base] = entry.first;
        megastepUnregistered_.push_back(entry);
      }
      megastepNextIndex_ = nextIndex;
      pending.refs = std::move(refs);
      megastepSlots_ = std::move(trial);
      batch_.push_back(std::move(pending));
      return success();
    }
    RETURN_IF_ERROR(flushMegastep());
  }
  return dispatchSingle(pending);
}

ErrorOrSuccess QuRTDeviceContext::flushMegastep() {
  // On every exit and after any held table was sent: a parked free's
  // allocation had to stay whole exactly until the launches naming it ran.
  auto drainOnExit = llvm::scope_exit([this] { drainMegastepFrees(); });

  if (batch_.empty() && megastepUnregistered_.empty())
    return success();

  // The transport row, out of the same manifest expansion every selector
  // comes from. Always present: the row is compiled into this binary.
  const NPUKernel *row = findKernel("megastep");
  if (row == nullptr)
    RETURN_ERROR("the NPU kernel manifest carries no megastep row");
  uint32_t selector = selectorOf(*row);

  // Taken before anything can fail, so a flush that errors leaves nothing
  // held to be sent a second time by the next one.
  std::vector<PendingLaunch> launches;
  std::vector<QuRTDevice::Allocation> slotAllocations;
  std::vector<std::pair<uint16_t, QuRTDevice::Allocation>> registrations;
  launches.swap(batch_);
  slotAllocations.swap(megastepSlots_);
  registrations.swap(megastepUnregistered_);

  DEFINE_OR_RETURN_ERROR(uint64_t, handle, skelHandle());
  ErrorOr<QuRTDriver &> driver = QuRTDriver::get();
  if (driver.isError())
    RETURN_ERROR(driver.getError());
  NPUDispatchTimeline *timeline = NPUDispatchTimeline::get();

  // Registrations first: a run block may name any entry registered this
  // session, including ones registered within this flush. Each group
  // presents its allocations as ordinary interface pointer arguments, which
  // is the once-per-session maintenance a table reference then lives off.
  for (size_t at = 0; at < registrations.size(); at += kNPUMaxBufferSlots) {
    size_t group =
        std::min<size_t>(kNPUMaxBufferSlots, registrations.size() - at);
    std::vector<uint8_t> block(kNPUMegastepHeaderBytes +
                               group * kNPUMegastepEntryBytes);
    writeWireWord(block.data(), kNPUMegastepMagic);
    writeWireWord(block.data() + 4, static_cast<uint32_t>(group));
    writeWireWord(block.data() + 8, kNPUBatchNoSlot);
    writeWireWord(block.data() + 12, kNPUMegastepModeRegister);

    llvm::SmallVector<uint32_t, 19> primIn(3 + 2 * kNPUMaxBufferSlots, 0);
    primIn[0] = selector;
    primIn[1] = 1;
    primIn[2] = static_cast<uint32_t>(block.size());
    llvm::SmallVector<QuRTRemoteBuf, 18> slots(2 + 2 * kNPUMaxBufferSlots,
                                               QuRTRemoteBuf{nullptr, 0});
    slots[0] = {primIn.data(), primIn.size() * sizeof(uint32_t)};
    slots[1] = {block.data(), block.size()};
    for (size_t j = 0; j < group; ++j) {
      const auto &[index, allocation] = registrations[at + j];
      uint8_t *entry =
          block.data() + kNPUMegastepHeaderBytes + j * kNPUMegastepEntryBytes;
      writeWireWord(entry, index);
      writeWireWord(entry + 4, static_cast<uint32_t>(j));
      auto *base = reinterpret_cast<void *>(allocation.base);
      primIn[3 + 2 * j] = static_cast<uint32_t>(allocation.byteSize);
      primIn[4 + 2 * j] = static_cast<uint32_t>(allocation.byteSize);
      slots[2 + j] = {base, allocation.byteSize};
      slots[2 + kNPUMaxBufferSlots + j] = {base, allocation.byteSize};
    }

    uint64_t preInvoke = timeline ? NPUDispatchTimeline::nowNanos() : 0;
    uint32_t status = static_cast<uint32_t>(
        driver->handleInvoke(handle,
                             qurtScalars(kNPUDispatchMethodID,
                                         /*inBufs=*/2 + kNPUMaxBufferSlots,
                                         /*outBufs=*/kNPUMaxBufferSlots),
                             slots.data()));
    if (timeline != nullptr)
      timeline->recordLaunch("megastep_register", selector, primIn[1], status,
                             preInvoke, preInvoke,
                             NPUDispatchTimeline::nowNanos(),
                             NPUDispatchTimeline::nowNanos(),
                             /*deviceTiming=*/nullptr);
    if (status != 0) {
      // The device never saw these entries, so the next hold must not name
      // them: take them back out of the mirror before reporting.
      for (size_t j = at; j < registrations.size(); ++j)
        megastepIndexByBase_.erase(registrations[j].second.base);
      // Recorded before the return because the caller may be unable to raise
      // it -- a free runs from destructors -- and said on stderr for the same
      // reason; the next hold refuses on the record.
      megastepPoison_ = "megastep registration of " + std::to_string(group) +
                        " allocations failed: " + qurtStatusMessage(status);
      fprintf(stderr, "npu megastep: FLUSH FAILED: %s\n",
              megastepPoison_.c_str());
      RETURN_ERROR("NPU megastep registration of " + std::to_string(group) +
                   " allocations failed: " + qurtStatusMessage(status) +
                   " -- a signed skel predating the megastep manifest row "
                   "refuses the selector by name");
    }
  }
  // Once, at the first registration flush: order 087 showed the per-flush
  // repeat collapses under log dedup into exactly the wrong reading (one
  // small flush instead of many). The ledger -- registrations are zero-launch
  // invocations -- carries every later registration.
  if (!registrations.empty() && !megastepRegisterAnnounced_) {
    fprintf(stderr, "npu megastep: registered %zu constant allocations\n",
            registrations.size());
    megastepRegisterAnnounced_ = true;
  }

  if (launches.empty())
    return success();

  // A table of one is not a transport: it goes out on the per-call wire --
  // the path with silicon behind it -- rather than as a degenerate one-entry
  // walk. This is the floor under every regime that keeps the table from
  // forming (order 087: a free after every launch), and it must degrade to
  // the synchronous answer, never to a wrong one.
  if (launches.size() == 1) {
    if (!megastepDegradeAnnounced_) {
      fprintf(stderr,
              "npu megastep: table degenerated to one launch; using per-call "
              "dispatch\n");
      megastepDegradeAnnounced_ = true;
    }
    return dispatchSingle(launches.front());
  }

  // The device leg's positive self-report, with the table's width: a
  // treatment leg that cannot say it is the treatment gets compared against
  // nothing.
  if (!megastepAnnounced_) {
    fprintf(stderr,
            "npu megastep: active (MODULAR_NPU_MEGASTEP=1, %zu launches "
            "tabled)\n",
            launches.size());
    megastepAnnounced_ = true;
  }

  // The header, a record per launch, then each launch's own argument area --
  // byte for byte the block a single dispatch sends, so a kernel reads
  // identical arguments on every transport.
  std::vector<uint8_t> block(
      kNPUMegastepHeaderBytes + launches.size() * kNPUMegastepRecordBytes, 0);
  for (size_t i = 0; i < launches.size(); ++i) {
    PendingLaunch &launch = launches[i];
    auto argsOffset = static_cast<uint32_t>(block.size());
    block.insert(block.end(), launch.scalars.begin(), launch.scalars.end());

    uint8_t *record =
        block.data() + kNPUMegastepHeaderBytes + i * kNPUMegastepRecordBytes;
    writeWireWord(record, launch.selector);
    writeWireWord(record + 4, launch.threadMask);
    writeWireWord(record + 8, argsOffset);
    writeWireWord(record + 12, launch.bufferCount | (launch.scalarCount << 16));
    for (uint32_t j = 0; j < launch.bufferCount; ++j)
      memcpy(record + 16 + 2 * j, &launch.refs[j], sizeof(uint16_t));
  }

  // The row declares one buffer, so the wire always carries one: a step whose
  // operands all went to the session table still presents its first launch's
  // first allocation, which costs one redundant maintenance pass and keeps
  // the declared arity honest.
  if (slotAllocations.empty())
    slotAllocations.push_back(launches.front().allocations.front());

  llvm::SmallVector<uint32_t, 19> primIn(3 + 2 * kNPUMaxBufferSlots, 0);
  primIn[0] = selector;
  primIn[1] = 1;
  primIn[2] = static_cast<uint32_t>(block.size());

  llvm::SmallVector<QuRTRemoteBuf, 18> slots(2 + 2 * kNPUMaxBufferSlots,
                                             QuRTRemoteBuf{nullptr, 0});
  slots[0] = {primIn.data(), primIn.size() * sizeof(uint32_t)};
  for (size_t i = 0; i < slotAllocations.size(); ++i) {
    auto *base = reinterpret_cast<void *>(slotAllocations[i].base);
    size_t byteSize = slotAllocations[i].byteSize;
    primIn[3 + 2 * i] = static_cast<uint32_t>(byteSize);
    primIn[4 + 2 * i] = static_cast<uint32_t>(byteSize);
    slots[2 + i] = {base, byteSize};
    slots[2 + kNPUMaxBufferSlots + i] = {base, byteSize};
  }

  NPULaunchDeviceTiming deviceTiming = {};
  uint32_t reportSlot = kNPUBatchNoSlot;
  if (timeline != nullptr && timeline->deviceTimingRequested() &&
      slotAllocations.size() < kNPUMaxBufferSlots) {
    DEFINE_OR_RETURN_ERROR(void *, report, timingSlot());
    memset(report, 0, sizeof(NPULaunchDeviceTiming));
    reportSlot = static_cast<uint32_t>(slotAllocations.size());
    primIn[3 + 2 * reportSlot] = sizeof(NPULaunchDeviceTiming);
    primIn[4 + 2 * reportSlot] = sizeof(NPULaunchDeviceTiming);
    slots[2 + reportSlot] = {report, sizeof(NPULaunchDeviceTiming)};
    slots[2 + kNPUMaxBufferSlots + reportSlot] = {
        report, sizeof(NPULaunchDeviceTiming)};
    primIn[1] |= kNPULaunchTime;
  }

  writeWireWord(block.data(), kNPUMegastepMagic);
  writeWireWord(block.data() + 4, static_cast<uint32_t>(launches.size()));
  writeWireWord(block.data() + 8, reportSlot);
  writeWireWord(block.data() + 12, kNPUMegastepModeRun);
  slots[1] = {block.data(), block.size()};
  uint32_t wireMask = primIn[1];

  uint64_t preInvoke = timeline ? NPUDispatchTimeline::nowNanos() : 0;
  uint32_t status = static_cast<uint32_t>(
      driver->handleInvoke(handle,
                           qurtScalars(kNPUDispatchMethodID,
                                       /*inBufs=*/2 + kNPUMaxBufferSlots,
                                       /*outBufs=*/kNPUMaxBufferSlots),
                           slots.data()));
  uint64_t postInvoke = timeline ? NPUDispatchTimeline::nowNanos() : 0;

  if (reportSlot != kNPUBatchNoSlot)
    memcpy(&deviceTiming, slots[2 + reportSlot].pv, sizeof(deviceTiming));
  if (timeline != nullptr)
    timeline->recordLaunch(
        "megastep", selector, wireMask, status, launches.front().entryNanos,
        preInvoke, postInvoke, NPUDispatchTimeline::nowNanos(),
        reportSlot != kNPUBatchNoSlot ? &deviceTiming : nullptr);
  if (status != 0) {
    // As above: recorded and said before the return, because the flush that
    // carried this table may have been triggered somewhere that cannot raise.
    megastepPoison_ = "megastep of " + std::to_string(launches.size()) +
                      " launches failed: " + qurtStatusMessage(status);
    fprintf(stderr, "npu megastep: FLUSH FAILED: %s\n",
            megastepPoison_.c_str());
    RETURN_ERROR("NPU invocation of a megastep of " +
                 std::to_string(launches.size()) +
                 " launches failed: " + qurtStatusMessage(status));
  }

  // The same guards, on the third transport. A correctness guard that silently
  // stops running under one transport is worse than not having it: the graph
  // path looks guarded and is not.
  for (size_t i = 0; i < launches.size(); ++i)
    RECORD_PENDING_ERROR(
        *this, evaluateWitnessGuards(launches[i], i, launches.size()));
  return success();
}

void QuRTDeviceContext::megastepForgetWrite(const void *dst) {
  if (megastepIndexByBase_.empty() && megastepUnregistered_.empty())
    return;
  // A plain host destination falls in no device allocation and drops nothing.
  ErrorOr<QuRTDevice::Allocation> allocation =
      static_cast<QuRTDevice &>(device()).allocationFrom(dst);
  if (allocation.isError())
    return;
  megastepIndexByBase_.erase(allocation->base);
  // A registration still pending for this allocation must not be sent: with
  // frees deferred, a free can now precede the flush that would have
  // presented it.
  for (size_t at = 0; at < megastepUnregistered_.size();)
    if (megastepUnregistered_[at].second.base == allocation->base)
      megastepUnregistered_.erase(megastepUnregistered_.begin() +
                                  static_cast<long>(at));
    else
      ++at;
}

void QuRTDeviceContext::drainMegastepFrees() {
  std::vector<void *> parked;
  parked.swap(megastepDeferredFrees_);
  for (void *ptr : parked) {
    ErrorOrSuccess freed =
        static_cast<QuRTDevice &>(device()).chunkFree(ptr, /*isHost=*/false);
    // A failed free of a dead buffer is a leak, not a corruption; said, not
    // raised, because this runs on every flush exit including error paths.
    if (freed.isError())
      fprintf(stderr, "npu megastep: deferred free failed: %s\n",
              freed.getError());
  }
}

ErrorOr<void *> QuRTDeviceContext::timingSlot() {
  std::lock_guard<std::mutex> lock(mu_);
  if (timingSlot_ != nullptr)
    return timingSlot_;
  // Through the device's own allocator, because a dispatch refuses any pointer
  // outside an allocation this device handed out.
  // Sized for EVERY record unconditionally, rather than for whichever the
  // environment asked for: this allocation is latched for the process, and a
  // slot sized by the first caller would silently truncate a later one. The
  // device writes phases only into a slot long enough for them, so an
  // over-sized slot costs the mapping and nothing else.
  RETURN_IF_ERROR(static_cast<QuRTDevice &>(device()).chunkAlloc(
      &timingSlot_, kNPUTimingSlotBytes + kNPUPipeSlotExtraBytes,
      /*isHost=*/false));
  return timingSlot_;
}

ErrorOr<DeviceStreamRef> QuRTDeviceContext::stream() const {
  return DeviceStreamRef::take(
      new QuRTDeviceStream(const_cast<QuRTDeviceContext &>(*this)));
}

ErrorOrSuccess QuRTDeviceContext::synchronize() {
  // Submit what is held, then wait for what is in flight: a batch still
  // filling is sent, and with the queue on this is the graph boundary. The
  // two-transports refusal means at most one has work.
  RETURN_IF_ERROR(flushBatch());
  // Similar to CUDA's behaviour, and to Metal's own `synchronize`: a deferred
  // error surfaces at the next synchronizing call. Here that is also the graph
  // boundary, so a witness guard recorded anywhere in the step is raised before
  // the step is called done.
  RETURN_IF_PENDING_ERROR(*this);
  return drainQueue();
}

ErrorOr<DeviceEventRef> QuRTDeviceContext::enqueueEvent() {
  RETURN_ERROR("the on-device NPU transport has no events: an invocation "
               "completes before it returns");
}

// Recording an event and waiting on one both assert that what came before is
// done, which on this transport was true by construction until a batch could be
// held back. So each of them sends what is held; none of them waits for
// anything else.
ErrorOrSuccess QuRTDeviceContext::enqueueEvent(const DeviceEvent &event) {
  return flushBatch();
}

ErrorOrSuccess QuRTDeviceContext::enqueueWaitFor(DeviceEvent &event) {
  return flushBatch();
}

ErrorOrSuccess QuRTDeviceContext::enqueueWaitFor(DeviceContext &ctx) {
  return flushBatch();
}

ErrorOr<DeviceTimerRef> QuRTDeviceContext::startTimer() {
  RETURN_ERROR("the on-device NPU transport exposes no device timer");
}

ErrorOr<std::chrono::nanoseconds>
QuRTDeviceContext::stopTimer(const DeviceTimer &timer) {
  RETURN_ERROR("the on-device NPU transport exposes no device timer");
}

ErrorOr<Device::MemInfo> QuRTDeviceContext::getDeviceMemoryInfo() const {
  return device().getMemoryInfo();
}

// Both send what is held first: a held launch names its operands by the
// allocation they fall in, so an allocation freed or reused under it would be
// mapped at a length it no longer has.
ErrorOrSuccess QuRTDeviceContext::memAlloc(void **ptr, size_t byteSize,
                                           Device::AllocMode mode) {
  // A fresh allocation cannot be named by any held launch, so the step table
  // holds across it. The graph path allocates a witness before every launch
  // (registrations.mojo), and flushing here is half of what shredded order
  // 087's tables to width ~1.7; the batch keeps its conservative flush, whose
  // behavior is pinned elsewhere. A deferred free cannot alias the new chunk:
  // the parked base has not reached rpcmem_free, so the driver cannot hand it
  // back yet.
  if (megastepping_ != 1)
    RETURN_IF_ERROR(flushBatch());
  return static_cast<QuRTDevice &>(device()).chunkAlloc(ptr, byteSize,
                                                        /*isHost=*/false);
}

ErrorOrSuccess QuRTDeviceContext::memFree(void *ptr, bool isHost) {
  // Under the step table a device free is PARKED, not performed: the graph
  // path frees a fresh witness after every launch, and an immediate free
  // both cuts the table (the other half of order 087's width-1.7) and, when
  // the free comes from a destructor, hands any flush error to a caller that
  // cannot raise it. The allocation stays whole until the flush that sends
  // the launches naming it; the table entry is dropped now, while the
  // allocation still resolves, because rpcmem hands addresses back.
  if (megastepping_ == 1 && !isHost) {
    megastepForgetWrite(ptr);
    megastepDeferredFrees_.push_back(ptr);
    return success();
  }
  RETURN_IF_ERROR(flushBatch());
  megastepForgetWrite(ptr);
  return static_cast<QuRTDevice &>(device()).chunkFree(ptr, isHost);
}

void *QuRTDeviceContext::driverContext() const { return nullptr; }

QuRTDeviceContext::MMapFileImplResult
QuRTDeviceContext::memmapFileImpl(const std::filesystem::path &filename) const {
  // TODO(fabio): weights cannot be mapped straight into device-visible memory,
  // so a model has to be staged through a host buffer; implement once the
  // transport gains a way to register an existing mapping.
  return Error("the on-device NPU cannot memory-map a file into device memory");
}

//===----------------------------------------------------------------------===//
// Entry point
//===----------------------------------------------------------------------===//

// The seam `Device::create` calls for an "npu" request. Declared in
// DeviceContext.h behind the same MODULAR_STATIC_NPU_BACKEND that the build
// defines where it supplies this library.
ErrorOr<DeviceRef> createNPUDevice(int id) { return QuRTDevice::create(id); }

} // namespace M::Driver
