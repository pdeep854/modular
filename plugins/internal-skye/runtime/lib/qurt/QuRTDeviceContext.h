//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// A DeviceContext for the on-device NPU, reached over FastRPC.
//
// Named for the real-time OS the kernels execute under, matching the
// per-runtime naming of CUDADeviceContext and MetalDeviceContext, and
// distinguishing it from the discrete accelerator's device context: that one
// drives its part through the vendor SDK and a JIT command-buffer model which
// has no counterpart over FastRPC.
//
// Shape of the backend, since it differs from every other one in tree: device
// memory here is host-mapped shared memory, so a "copy" is a memcpy and there
// is no separate address space. Dispatch is a remote invocation of an entry
// point in a skel that was built and code-signed ahead of time, not a JIT
// compile.
//
//===----------------------------------------------------------------------===//

#ifndef SKYE_RUNTIME_QURT_QURTDEVICECONTEXT_H
#define SKYE_RUNTIME_QURT_QURTDEVICECONTEXT_H

#include "MLRT/Driver/DeviceContext/CompilationDevice.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "NPUQueueTransport.h"
#include "QuRTDriver.h"
#include "Support/ErrorOr.h"
#include "Support/RCRef.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace M::Driver {

class QuRTDevice;
class QuRTDeviceContext;
class QuRTDeviceFunction;
class QuRTDeviceStream;

using QuRTDeviceRef = RCRef<QuRTDevice>;
using QuRTDeviceContextRef = RCRef<QuRTDeviceContext>;
using QuRTDeviceFunctionRef = RCRef<QuRTDeviceFunction>;
using QuRTDeviceStreamRef = RCRef<QuRTDeviceStream>;

class QuRTDevice : public Device {
public:
  static constexpr API api = Device::npuAPI;

  QuRTDevice(int id, Device::MemInfo memInfo);
  ~QuRTDevice() override;

  /// Builds device `id`, or reports why the transport is unreachable. Called
  /// through `M::Driver::createNPUDevice`, the seam `Device::create` uses.
  static ErrorOr<DeviceRef> create(int id);

  ErrorOr<int> getApiVersion() const override;
  ErrorOr<DriverVersion &> getDriverVersion() const override;
  ErrorOrSuccess isCompatible() const override;
  ErrorOrSuccess runHealthcheck() const override { return success(); }
  ErrorOr<int> computeCapability() const override;

  ErrorOr<bool> canAccess(const Device &peer) const override;
  ErrorOr<int> getAttribute(Attribute attr) const override;
  ErrorOr<std::string> getPciBusId() const override;
  int deviceCount() const override;

  ErrorOrSuccess chunkAlloc(void **ptr, size_t byteSize, bool isHost) override;
  ErrorOrSuccess chunkFree(void *ptr, bool isHost) override;

  /// Reclassifies the allocation `dst` falls in ahead of a host write of
  /// `byteSize` bytes from `src`, and applies or withdraws the init-once buffer
  /// attribute accordingly. Every runtime-mediated write to device memory calls
  /// this first, which is what makes the classification revocable: a buffer the
  /// host writes again stops being init-written before the write lands, so the
  /// driver flushes it as it always did.
  ///
  /// A no-op unless `MODULAR_NPU_WEIGHTS_NOFLUSH` or `MODULAR_NPU_QUEUE` is
  /// set -- the queue keeps the classification without the attribute, because
  /// its per-reference descriptor flags are chosen by the same provenance test.
  /// Errors name a missing vendor entry point or a refused descriptor rather
  /// than silently leaving the attribute unset, which would make an A/B leg
  /// indistinguishable from its own control.
  ErrorOrSuccess noteHostWrite(void *dst, const void *src, size_t byteSize);

  /// Whether the allocation at `base` is currently classified init-constant:
  /// written whole, once, from loader-declared constant memory, and not
  /// rewritten since. The queue transport marks such references so their
  /// descriptors carry no per-frame cache maintenance.
  bool isDeclaredConstant(const void *base) const;

  /// Bytes from `ptr` to the end of the allocation it falls in, or an error
  /// when it falls in none. A dispatch states a length for every argument slot,
  /// while a launch argument is a bare address, so the length comes from the
  /// allocation rather than from the argument vector.
  ErrorOr<size_t> allocationBytesFrom(const void *ptr) const;

  /// The whole allocation `ptr` falls in: its base and its size.
  ///
  /// A launch needs both halves, not just a length. FastRPC registers memory an
  /// allocation at a time and hands the device the *mapping's* base, so an
  /// address's offset within its allocation does not survive the call and has
  /// to be sent separately for the device to add back.
  struct Allocation {
    uintptr_t base;
    size_t byteSize;
  };
  ErrorOr<Allocation> allocationFrom(const void *ptr) const;

  /// Maps `base` to the compute DSP for the rest of the allocation's life, and
  /// is a no-op once it has. Idempotent because the caller is the dispatch
  /// path: every launch naming the allocation reaches it, and only the first
  /// may map.
  ///
  /// Public, and not called from `chunkAlloc`, because `fastrpc_mmap` is an RPC
  /// on the domain and this device deliberately opens no session (`create`) --
  /// only a context that has opened the skel may map. Mapping at allocation
  /// time is what order 163 measured returning AEE_ERPC.
  ErrorOrSuccess mapAllocationOnce(QuRTDriver &driver, void *base,
                                   size_t byteSize);

  ErrorOr<Device::MemInfo> getMemoryInfo() const override;

  ErrorOr<void *> getCurrentDriverContext() const override;
  ErrorOrSuccess setCurrentDriverContext(void *ctx) override;

  ErrorOr<DeviceContextRef> createContext() override;

  DLDevice getDLDevice(bool /*pinned*/) const override;

private:
  /// Sets `base`'s buffer attribute, or says why it could not.
  ErrorOrSuccess setBufferAttr(uintptr_t base, size_t byteSize, int attr);

  /// Every allocation this device made, by base address, so a dispatch can
  /// bound the slot it passes one in.
  mutable std::mutex allocationsMu_;
  std::map<uintptr_t, size_t> allocations_;
  /// Allocations currently carrying a non-default buffer attribute, so a host
  /// write can take it back off. Guarded by `allocationsMu_`.
  std::set<uintptr_t> attributed_;
  /// Allocations classified init-constant, tracked apart from `attributed_`
  /// because the classification serves two consumers with different gates: the
  /// NOFLUSH probe registers a driver attribute, the queue only reads the
  /// class. Guarded by `allocationsMu_`.
  std::set<uintptr_t> constantBases_;
  /// The descriptor each allocation is mapped to the DSP under, for the
  /// allocations `MODULAR_NPU_MAP_ONCE` mapped. Teardown is keyed off this
  /// table and not off the flag, so a flag flipped mid-process still unmaps
  /// exactly what it mapped. Guarded by `allocationsMu_`.
  std::map<uintptr_t, int> mappedFDs_;

  /// The step-buffer pool behind `NPU_STEP_ALLOC_POOL=1`: chunks parked by
  /// exact byte size at `chunkFree` instead of handed to `rpcmem_free`, and
  /// re-issued address-stable on the next same-size `chunkAlloc`. The point is
  /// the rpcmem handle's lifetime: the vendor driver charges a
  /// byte-proportional first-presentation cost per handle, so the engine's
  /// per-step free/realloc of its step buffers re-pays it every step inside
  /// the step's first launch — parking keeps the handle alive and the cost is
  /// paid once per process, the same endgame the process-lifetime weights
  /// measure. Exact-size keying is complete: a block re-issued only for the
  /// byte size it was allocated at carries the identical rpcmem guarantees.
  /// All of it guarded by `allocationsMu_`, beside the tables `chunkFree`'s
  /// erasures maintain — a parked block passes those erasures verbatim, so
  /// nothing (mapping, attribute, constant class) rides a re-issued block.
  std::map<size_t, std::vector<void *>> stepPoolFree_;
  uint64_t stepPoolParkedBlocks_ = 0;
  uint64_t stepPoolParkedBytes_ = 0;
  uint64_t stepPoolHighWaterBlocks_ = 0;
  uint64_t stepPoolHighWaterBytes_ = 0;
  uint64_t stepPoolParks_ = 0;
  uint64_t stepPoolReuses_ = 0;
  uint64_t stepPoolReusedBytes_ = 0;
  uint64_t stepPoolRawWhileArmed_ = 0;
  bool stepPoolAnnounced_ = false;
  bool stepPoolExitArmed_ = false;

public:
  /// Writes the pool's census to stderr: high-water parked blocks/bytes, and
  /// how many allocations the pool served against how many went to rpcmem
  /// while armed. Public because it runs from the `atexit` hook the first park
  /// arms — the device is a process-lifetime singleton, so the hook outlives
  /// nothing. The line is the run's delivery receipt: a treatment leg whose
  /// stderr carries no census was not treated, and the footprint ceiling is
  /// reported rather than assumed.
  void reportStepPoolCensus();
};

/// Streams have no counterpart in this transport: an invocation blocks until
/// the device is done, so ordering is already total and every operation below
/// is either a no-op or an error naming what is missing.
class QuRTDeviceStream : public DeviceStream {
public:
  explicit QuRTDeviceStream(QuRTDeviceContext &ctx);
  ~QuRTDeviceStream() override = default;

  ErrorOrSuccess synchronize() override;
  ErrorOr<DeviceEventRef> enqueueEvent() const override;
  ErrorOrSuccess eventRecord(const DeviceEvent &event) const override;
  ErrorOrSuccess waitEvent(const DeviceEvent &event,
                           unsigned int flags) const override;
  ErrorOrSuccess enqueueWaitForEvent(const DeviceEvent &event) const override;
  ErrorOrSuccess enqueueFunctionExecDirect(
      const DeviceFunction &func, uint32_t gridX, uint32_t gridY,
      uint32_t gridZ, uint32_t blockX, uint32_t blockY, uint32_t blockZ,
      uint32_t sharedMemBytes, void *attrs, uint32_t numAttrs, void **args,
      uint32_t argCount = 0, const size_t *argSizes = nullptr) const override;
  void *ptr() const override;
};

/// A kernel of the loaded skel. Every kernel shares the skel's one generic
/// entry point, so the method index below is the same for all of them and it is
/// the name that picks the kernel out; see `NPUKernels.def`.
class QuRTDeviceFunction : public DeviceFunction {
public:
  QuRTDeviceFunction(QuRTDeviceContext &ctx, std::string name,
                     uint32_t methodID);
  ~QuRTDeviceFunction() override = default;

  const std::string &name() const override { return name_; }
  uint32_t methodID() const { return methodID_; }

  ErrorOrSuccess copyToNamedConstant(const char *name, const void *data,
                                     size_t byteSize) override;
  ErrorOr<int32_t> getAttribute(int32_t code) override;

private:
  std::string name_;
  uint32_t methodID_;
};

class QuRTDeviceContext : public DeviceContext {
public:
  explicit QuRTDeviceContext(QuRTDevice &device);
  ~QuRTDeviceContext() override;

  ErrorOrSuccess enqueueCopyToDevice(void *dst, const void *src,
                                     size_t byteSize) override;
  ErrorOrSuccess enqueueCopyFromDevice(void *dst, const void *src,
                                       size_t byteSize) override;
  ErrorOrSuccess memCopyDtoD(DeviceBuffer &dst, DeviceBuffer &src,
                             size_t srcByteOffset, size_t srcByteSize,
                             bool crossStreamSync) override;
  ErrorOrSuccess enqueueSetMemory(void *dst, size_t byteSize, uint64_t val,
                                  size_t valSize) override;

  ErrorOr<DeviceFunctionRef>
  loadFunction(StringRef moduleName, StringRef functionName, StringRef code,
               int32_t maxDynamicSharedBytes, StringRef debugLevel,
               int32_t optimizationLevel) override;
  ErrorOrSuccess enqueueFunctionExecDirect(
      const DeviceFunction &func, uint32_t gridX, uint32_t gridY,
      uint32_t gridZ, uint32_t blockX, uint32_t blockY, uint32_t blockZ,
      uint32_t sharedMemBytes, void *attrs, uint32_t numAttrs, void **args,
      uint32_t argCount = 0, const size_t *argSizes = nullptr) override;

  ErrorOr<DeviceStreamRef> stream() const override;
  ErrorOrSuccess synchronize() override;

  ErrorOr<DeviceEventRef> enqueueEvent() override;
  ErrorOrSuccess enqueueEvent(const DeviceEvent &event) override;
  ErrorOrSuccess enqueueWaitFor(DeviceEvent &event) override;
  ErrorOrSuccess enqueueWaitFor(DeviceContext &ctx) override;

  ErrorOr<DeviceTimerRef> startTimer() override;
  ErrorOr<std::chrono::nanoseconds>
  stopTimer(const DeviceTimer &timer) override;

  ErrorOr<Device::MemInfo> getDeviceMemoryInfo() const override;

  /// Opens the skel on first use and returns its handle. Separate from context
  /// creation on purpose: this is the only step that needs a valid code
  /// signature, so buffers and copies stay usable when the catalogue is stale,
  /// and the signing failure surfaces at dispatch where it can be named.
  ErrorOr<uint64_t> skelHandle();

private:
  ErrorOrSuccess memAlloc(void **ptr, size_t byteSize,
                          Device::AllocMode mode) override;
  ErrorOrSuccess memFree(void *ptr, bool isHost) override;

  void *driverContext() const override;

  MMapFileImplResult
  memmapFileImpl(const std::filesystem::path &filename) const override;

  /// The device allocation a launch's timing report is written into, allocated
  /// on first use and reused for every launch after. One per context rather
  /// than one per launch because allocating is an rpcmem round trip and would
  /// be charged to the thing being measured.
  ErrorOr<void *> timingSlot();

  /// The queue transport, opened on first dispatch when `MODULAR_NPU_QUEUE=1`.
  /// Null when the env var is unset, which is when every launch takes the
  /// blocking path this class was built around.
  ErrorOr<NPUQueueTransport *> queueTransport();

  /// Waits for every queued launch to finish. A no-op with the queue off, and
  /// the whole of what makes the graph boundary synchronous with it on: a host
  /// read of device memory, a device-to-host copy or an explicit synchronize
  /// must not see memory a launch still in flight is writing.
  ErrorOrSuccess drainQueue();

  /// A launch packed for the wire but not yet sent, held while the batch it
  /// will travel in fills.
  struct PendingLaunch {
    /// Points into the static kernel manifest, so it outlives the launch.
    const char *name = nullptr;
    uint32_t selector = 0;
    uint32_t threadMask = 0;
    uint32_t bufferCount = 0;
    uint32_t scalarCount = 0;
    /// The kernel's scalars, then one 8-byte offset per buffer -- the same
    /// block a single dispatch sends, so a batched launch and a per-call
    /// launch read identical arguments.
    std::vector<uint8_t> scalars;
    /// The allocation each buffer operand falls in, in operand order.
    std::vector<QuRTDevice::Allocation> allocations;
    /// Which of the call's declared slots carries each operand. Empty until
    /// the launch joins a batch.
    std::vector<uint8_t> slotIndex;
    /// Megastep operand references: a call slot, or a session table entry
    /// under the table bit. Empty until the launch joins a step table.
    std::vector<uint16_t> refs;
    uint64_t entryNanos = 0;
  };

  /// Whether launches are held for a batched dispatch. Latched on first call
  /// from `MODULAR_NPU_BATCH`; off unless it is "1", and off whenever the IO
  /// recorder is on.
  ///
  /// Opt-in because holding a launch back weakens the one guarantee this
  /// transport gave for free: that a dispatch has run by the time
  /// `enqueueFunctionExecDirect` returns. Every route by which this class can
  /// be asked to observe device memory sends the batch first, but device
  /// memory here IS host memory, so a caller holding a device pointer can read
  /// an operand without asking this class anything — and a kernel that runs on
  /// the host while claiming a device is exactly such a caller. For a launch
  /// sequence that is known to contain no such reader, batching is
  /// transparent; for the graph path the durable seam is MAX's own
  /// `DeviceContext::createGraphBuilder()`, whose recorded region carries that
  /// property by construction.
  bool batching();

  /// Which declared slot each of `launch`'s operands takes, appending to
  /// `slots` the allocations it does not already carry. False when the union
  /// would pass `budget`, at which point `slots` is left partly extended --
  /// callers work on a copy for exactly that reason.
  static bool assignSlots(const PendingLaunch &launch, uint32_t budget,
                          std::vector<QuRTDevice::Allocation> &slots,
                          std::vector<uint8_t> &slotIndex);

  /// Sends one launch on its own: the wire every dispatch used before batching
  /// existed, and the wire every dispatch still uses with it off.
  ErrorOrSuccess dispatchSingle(PendingLaunch &launch);

  /// Sends every held launch as one invocation carrying a launch list, and
  /// leaves nothing held. Called from each entry point through which the host
  /// could observe device memory, so a launch cannot be observed before it ran.
  /// With the step table on it forwards to `flushMegastep`, so every flush
  /// call site covers both holds without naming both.
  ErrorOrSuccess flushBatch();

  /// Evaluates the deferred witness guards a graph stub armed on `launch`'s
  /// arena slot, once the launch has actually run. `position` and `width`
  /// name the launch within the flush that just completed, so a refusal names
  /// which of N it was. Returns an error only for a CORRECTNESS guard;
  /// attribution guards record loudly and the render continues. The
  /// `matmul_chain2` row carries one witness per member, and this walks both.
  ErrorOrSuccess evaluateWitnessGuards(const PendingLaunch &launch,
                                       size_t position, size_t width);

  /// One member witness's guards: `operand` names the buffer slot the walk
  /// reads. Every row's witness is its last slot; the chain row also has one
  /// at slot 3, which `evaluateWitnessGuards` walks first.
  ErrorOrSuccess evaluateWitnessGuardsAt(const PendingLaunch &launch,
                                         uint32_t operand, size_t position,
                                         size_t width);

public:
  /// Dispatches the launch the matmul-chain stub asked to hold, if one is
  /// still held. Called from `flushBatch` -- every seam through which the
  /// host can observe device memory -- and from the stub's own C entries.
  ErrorOrSuccess fireHeldChain();

  /// Discards the held chain launch and reports whether one was held; the
  /// stub calls this when it pairs the held member into a `matmul_chain2`
  /// dispatch of its own.
  bool takeHeldChain();

private:
  /// The matmul-chain hold: at most one launch, held between two adjacent
  /// matmul registrations by `MODULAR_NPU_MATMUL_CHAIN=1`.
  std::optional<PendingLaunch> chainHeld_;

  /// Whether launches are held for a whole-step table. Latched on first call
  /// from `MODULAR_NPU_MEGASTEP`, off whenever the IO recorder is on -- the
  /// recorder brackets one finished launch per call, which a table has no
  /// boundary for, and its per-call leg is the reference the table is judged
  /// against.
  bool megastepping();

  /// Holds `launch` for the step table: constant-classified operands become
  /// session-table references (registered at the next flush), everything else
  /// takes one of the call's own maintained slots. Flushes first when the
  /// maintained union or the launch count would pass its bound.
  ErrorOrSuccess holdForMegastep(PendingLaunch &launch);

  /// Sends pending registrations, then every held launch as ONE invocation
  /// carrying the whole table, and leaves nothing held.
  ErrorOrSuccess flushMegastep();

  /// Drops the session-table entry for the allocation `dst` falls in, if any,
  /// and any registration still pending for it. Called on every host write
  /// into device memory and on every free: a rewritten or re-issued
  /// allocation must be re-registered -- registration is the presentation
  /// that flushes the host's write -- rather than named by a reference whose
  /// maintenance already happened.
  void megastepForgetWrite(const void *dst);

  /// Hands every deferred free to the device allocator. Called on every
  /// `flushMegastep` exit, after any held table was sent: a freed allocation
  /// a held launch names stays whole until the table it rides is dispatched.
  void drainMegastepFrees();

  /// Refuses, by name and before any transport sees it, a dispatch of an
  /// HMX-capable kernel into a session whose power vote is not live. The HMX
  /// lock grant requires a live session vote (P5 M2.4 order 077: seven
  /// windows of QURT_EINVALID refusals, flipped to a grant by the vote
  /// alone), so dispatching without one produces a kernel that computes
  /// nothing and says so only in a witness word -- the mystery this refusal
  /// replaces. A no-op for every other kernel, and for HVX-only skels, which
  /// it recognizes through the vote report's capability word.
  ErrorOrSuccess guardHmxDispatch(const char *kernelName);

  mutable std::mutex mu_;
  uint64_t skelHandle_ = 0;

  /// The session power vote's outcome, latched at skel open under `mu_`:
  /// whether the DSP vote is live, why not when not, and whether the loaded
  /// skel routes any kernel to the HMX array (-1 until a vote report answers
  /// it; resolved lazily by `guardHmxDispatch` when the vote is off).
  bool powerVoteLive_ = false;
  std::string powerVoteReason_;
  int skelHmx_ = -1;
  void *timingSlot_ = nullptr;
  std::unique_ptr<NPUQueueTransport> queue_;

  /// Launches held for the next batch, and the distinct allocations they name
  /// in the slot order the call will declare them in. A launch names its
  /// operands by slot rather than by address, because a kernel may only
  /// dereference a pointer that arrived in a declared slot -- which is also
  /// what bounds a batch, and what stops it declaring one allocation twice.
  std::vector<PendingLaunch> batch_;
  std::vector<QuRTDevice::Allocation> batchSlots_;
  int batching_ = -1;

  /// Step-table state, live only with `MODULAR_NPU_MEGASTEP=1` (the transport
  /// refusals keep it and the batch from ever holding at once, so `batch_` is
  /// shared). The map mirrors the device's session table: which allocation
  /// holds which entry, so a step can name a weight registered steps ago
  /// without re-presenting it.
  std::vector<QuRTDevice::Allocation> megastepSlots_;
  std::map<uintptr_t, uint16_t> megastepIndexByBase_;
  std::vector<std::pair<uint16_t, QuRTDevice::Allocation>>
      megastepUnregistered_;
  /// Device frees parked while launches are held: the graph path frees a
  /// fresh witness after every launch, and an immediate free would cut the
  /// table there (order 087's width-1.7 tables). Freed on every flush, after
  /// any held table was sent.
  std::vector<void *> megastepDeferredFrees_;
  /// A flush error that may have been swallowed upstream -- a free from a
  /// destructor cannot raise -- recorded so the next launch refuses loudly
  /// instead of the render continuing from kernels that never ran.
  std::string megastepPoison_;
  uint16_t megastepNextIndex_ = 0;
  int megastepping_ = -1;
  bool megastepAnnounced_ = false;
  bool megastepDegradeAnnounced_ = false;
  bool megastepRegisterAnnounced_ = false;
};

} // namespace M::Driver

#endif // SKYE_RUNTIME_QURT_QURTDEVICECONTEXT_H
