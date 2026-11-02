//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Per-dispatch IO recording for the on-device NPU launch path.
//
// Every kernel launch the graph path makes funnels through
// `QuRTDeviceContext::enqueueFunctionExecDirect`, so a record taken there sees
// the whole dispatch chain of a compiled model without rebuilding the MEF or
// re-signing the device library. Each launch record carries what the wire
// carries -- the selector, the thread mask, each buffer slot's allocation and
// its offset as packed into the scalar block, the scalar values -- plus a
// content hash of each operand's leading window before and after the
// invocation. The runtime's own copies and fills are recorded too, so a byte
// that changes between two launches is attributable to something.
//
// The offsets are decoded back out of the packed scalar block rather than
// taken from the packer's inputs, so the record is of what was SENT: a packing
// bug shows up in the record instead of being masked by it.
//
// What a window hash can and cannot localize: a divergence in an operand's
// leading `window` bytes (default 64 KiB) is caught at the first dispatch that
// carries it; a corruption confined to bytes past the window is invisible
// unless MODULAR_NPU_DISPATCH_DUMP_WINDOW=0 extends the hash to the whole
// remaining allocation span, at proportionate hashing cost. The all-zero flag
// is computed over the same window.
//
// Recording is off unless MODULAR_NPU_DISPATCH_DUMP names a directory; when
// off, the cost is one null check per dispatch. Records are line-delimited
// JSON, one file per process, flushed per record so a killed run keeps what it
// saw. Environment:
//
//   MODULAR_NPU_DISPATCH_DUMP        directory to write into (master switch)
//   MODULAR_NPU_DISPATCH_DUMP_WINDOW hashed bytes per operand, 0 = to the end
//                                    of the allocation (default 65536)
//   MODULAR_NPU_DISPATCH_DUMP_BYTES  "1" also dumps each hashed window to a
//                                    .bin file (pre and post)
//   MODULAR_NPU_DISPATCH_DUMP_BYTES_CAP
//                                    total .bin bytes before dumping stops
//                                    (default 256 MiB); hashing continues
//
// Allocation identity across runs: raw bases differ per process, so each
// record also carries a small ordinal assigned to each allocation base in
// first-seen order. Two runs of the same compiled model allocate in the same
// order, so ordinals line up where raw addresses cannot. A base freed and
// reissued keeps its first ordinal; no current caller frees mid-run.
//
//===----------------------------------------------------------------------===//

#ifndef SKYE_RUNTIME_QURT_NPUDISPATCHRECORDER_H
#define SKYE_RUNTIME_QURT_NPUDISPATCHRECORDER_H

#include "QuRTDriver.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <mutex>
#include <string>
#include <vector>

/// Names a phase of the process on the perf timeline, from C. Defined in
/// NPUDispatchRecorder.cpp; declared here so the definition is checked against
/// what a C driver links to. A no-op when `MODULAR_NPU_PERF_LOG` is unset.
extern "C" void npu_perf_mark(const char *phase);

namespace M::Driver {

/// Words of a witness slot that may be read, given the slot's allocation and
/// the operand's offset into it. Bytes in, words out, and the units are the
/// point: P5 M4.2 order 091 found the graph path allocating the witness at 32
/// BYTES while the kernels write 32 WORDS, so a reader that trusts a word count
/// it believes rather than the bytes the slot has reads off the end.
uint32_t npuWitnessWordsToReport(uint64_t allocBytes, uint64_t offset);

class NPUDispatchRecorder {
public:
  /// The process-wide recorder, or nullptr when recording is off. Latched from
  /// the environment on first call, so the check a disabled dispatch pays is
  /// this null test.
  static NPUDispatchRecorder *get();

  /// Captures operand identities and pre-invocation hashes for one launch.
  /// `scalarBlock` is the packed block exactly as it will cross, holding
  /// `scalarCount` scalar slots then `bufferCount` offset slots; `bufferSlots`
  /// are the buffer entries handed to the transport (allocation base and
  /// size). Returns a token to close the record with.
  uint64_t beginLaunch(const char *name, uint32_t selector, uint32_t threadMask,
                       const uint8_t *scalarBlock, size_t scalarBlockBytes,
                       uint32_t scalarCount, uint32_t bufferCount,
                       const QuRTRemoteBuf *bufferSlots);

  /// Hashes the same windows after the invocation returned and writes the
  /// record, including the transport's status: a refused launch is a record,
  /// not a gap.
  void endLaunch(uint64_t token, uint32_t status);

  /// Hashes `byteSize` leading bytes (bounded by the window) of a destination
  /// about to be overwritten. Pair with `recordHostOp` so the record carries
  /// what was there before.
  uint64_t windowHashOf(const void *ptr, size_t byteSize);

  /// Records one runtime-mediated host operation: a copy in either direction,
  /// a device-to-device copy, or a fill. `src` may be null (fills), and the
  /// destination's allocation is 0/0 when it is not device memory: only a
  /// device allocation has a cross-run identity worth recording.
  void recordHostOp(const char *kind, const void *dst, uintptr_t dstAllocBase,
                    size_t dstAllocBytes, const void *src, size_t byteSize,
                    uint64_t dstPreHash);

private:
  struct SlotCapture {
    uintptr_t base;
    uint64_t allocOrdinal;
    uint64_t allocBytes;
    uint64_t offset;
    uint64_t hashedBytes;
    uint64_t preHash;
    bool preZero;
    uint64_t postHash;
    bool postZero;
  };

  struct PendingLaunch {
    std::string name;
    uint32_t selector;
    uint32_t threadMask;
    std::vector<uint64_t> scalars;
    std::vector<SlotCapture> slots;
  };

  explicit NPUDispatchRecorder(std::string directory);

  uint64_t ordinalOf(uintptr_t base);
  void hashWindow(const void *ptr, size_t available, uint64_t &hash,
                  bool &allZero, uint64_t &hashedBytes);
  void dumpWindowBytes(uint64_t seq, size_t slot, const char *leg,
                       const void *ptr, size_t byteCount);
  void writeLaunchRecord(uint64_t seq, const PendingLaunch &launch,
                         uint32_t status);

  std::mutex mu_;
  FILE *file_ = nullptr;
  std::string directory_;
  uint64_t nextSeq_ = 0;
  uint64_t window_ = 65536;
  bool dumpBytes_ = false;
  uint64_t byteCap_ = 256ull << 20;
  uint64_t bytesWritten_ = 0;
  bool byteCapReported_ = false;
  std::map<uintptr_t, uint64_t> allocOrdinals_;
  std::map<uint64_t, PendingLaunch> pending_;
};

/// Stamps for the timing report `mojo_npu_imp.c` writes when a launch asks for
/// one. Raw counter reads, not differences, because a counter that did not
/// advance at all is itself a finding and a difference hides it.
struct NPULaunchDeviceTiming {
  uint64_t entryPcycles;
  uint64_t threadsCreatedPcycles;
  uint64_t kernelReturnedPcycles;
  uint64_t exitPcycles;
  uint64_t entryQtimer;
  uint64_t exitQtimer;
};

/// One participant's WITHIN-dispatch accounting, appended after the struct
/// above in the same slot (`npu_phase_probe.h`, which owns the layout and
/// static-asserts its size). The timing struct is deliberately left
/// byte-identical, so a caller that asks for the shorter slot still gets the
/// per-launch stamps and simply no phases.
///
/// Where `NPULaunchDeviceTiming` says how long a launch took, this says where
/// the time went inside it: barrier vs MMA vs the steps the schedule gives the
/// array no work in. The buckets partition the span between a thread's first
/// and last mark, so a share computed from them has a denominator that is
/// stated rather than assumed.
constexpr uint32_t kNPUPhaseCount = 10;
constexpr uint32_t kNPUWaitHistBuckets = 8;
constexpr uint32_t kNPUPhaseMaxThreads = 8;

struct NPULaunchPhaseThread {
  uint64_t pcycles[kNPUPhaseCount];
  uint64_t qticks[kNPUPhaseCount];
  uint64_t waits;
  uint64_t lastIn;
  uint64_t spinExits;
  uint64_t napExits;
  uint64_t giveUps;
  uint64_t naps;
  uint64_t waitHist[kNPUWaitHistBuckets];
  uint64_t marks;
  uint64_t openPcycle;
  uint64_t openQtick;
  uint32_t phase;
  uint32_t pad;
};

struct NPULaunchPhases {
  NPULaunchPhaseThread thread[kNPUPhaseMaxThreads];
};

static_assert(sizeof(NPULaunchPhaseThread) == 304,
              "must match npu_phase_thread_t in npu_phase_probe.h");

/// The rendezvous' session counters, as the skel ships them in the report
/// slot after the phase record -- or alone, when only the stamps were asked
/// for (`npu_sync_report.h` carries the layout and the why; order 243's
/// unreadable counter witness is the why). Counters are session-cumulative:
/// diff consecutive rows for per-render deltas, read the last row for the
/// session. The magic distinguishes a block the skel wrote from the zeroes
/// this host sent: a skel without the probe reports ABSENT, not idle.
constexpr uint32_t kNPUSyncReportMagic = 0x434E5953u;
constexpr uint32_t kNPUSyncReportVersion = 1u;

struct NPULaunchSyncReport {
  uint32_t magic;
  uint32_t version;
  uint64_t waits;
  uint64_t lastIn;
  uint64_t spinExits;
  uint64_t napExits;
  uint64_t giveUps;
  uint64_t naps;
  uint64_t waitQticks;
  uint64_t napWaitQticks;
  uint64_t budgetHi;
};

static_assert(sizeof(NPULaunchSyncReport) == 80,
              "must match npu_sync_report_wire_t in npu_sync_report.h");

/// Where a denoise step's wall clock went, recorded at the one place every
/// kernel launch passes through and at the phases a process crosses on the way
/// there.
///
/// The launch path is one blocking FastRPC round trip per kernel with nothing
/// in flight across it (`QuRTDeviceContext::synchronize` is a no-op for exactly
/// that reason), so a host-side bracket around the invoke is not an
/// approximation of the device's occupancy -- it is the whole of it, and the
/// time between one launch's return and the next launch's invoke is time the
/// NSP provably spent idle. That gap is the number the port has never had.
///
/// Three host spans per launch, so the gap is attributable rather than merely
/// visible: `prep` is everything before the wire (validating the geometry,
/// packing scalars, resolving each operand's allocation), `invoke` is the
/// blocking round trip, `post` is the return path. Whatever is left between one
/// launch's `post` and the next's `prep` belongs to the graph runtime above
/// this layer, and the summary tool names it rather than absorbing it.
///
/// With `MODULAR_NPU_PERF_DEVICE=1` the launch also carries the skel's own
/// timing report back, which splits `invoke` into what the NSP was busy with
/// and what the transport cost. That needs no interface change and no re-sign:
/// the signed skel already implements it (`NPU_LAUNCH_TIME` in
/// `mojo_npu_imp.c`), and the report rides in the buffer slot after the
/// kernel's own -- a slot the manifest guarantees is free, since no registered
/// kernel takes more than six of the eight. It does cost one more mapped slot
/// per call, so it perturbs what it measures; running with it off and on is
/// itself the measurement of what a mapped slot costs.
///
/// Recording is off unless `MODULAR_NPU_PERF_LOG` names a directory; when off,
/// the cost is one null check per dispatch. Records are line-delimited JSON,
/// one file per process. Environment:
///
///   MODULAR_NPU_PERF_LOG     directory to write into (master switch)
///   MODULAR_NPU_PERF_DEVICE  "1" also asks the skel for its per-launch report
///                            (stamps plus, from a probe-built skel, the
///                            session sync-counter block)
///   MODULAR_NPU_PERF_PHASES  "1" additionally asks for the within-dispatch
///                            phase block; implies MODULAR_NPU_PERF_DEVICE
///   MODULAR_NPU_PERF_FLUSH   "1" flushes every record instead of buffering,
///                            for a run expected to be killed; costs the flush
///                            inside the inter-launch gap it is measuring
class NPUDispatchTimeline {
public:
  /// The process-wide timeline, or nullptr when timing is off. Latched from the
  /// environment on first call.
  static NPUDispatchTimeline *get();

  /// Monotonic nanoseconds. Always callable, so a caller can stamp before
  /// deciding whether anything will read the stamp.
  static uint64_t nowNanos();

  /// Whether launches should carry the skel's timing report back.
  bool deviceTimingRequested() const { return deviceTiming_; }

  /// Whether the report should also carry the within-dispatch phase block. A
  /// superset of `deviceTimingRequested`: the phases ride the same slot, so
  /// asking for them without the stamps is not expressible and is not refused,
  /// it simply implies them.
  bool devicePhasesRequested() const { return devicePhases_; }

  /// One launch, as the four stamps around it. `device` is null when the launch
  /// carried no report or the skel wrote none; `phases` and `sync` are null
  /// additionally when the skel was built without the probe, which is the
  /// shipping build.
  void recordLaunch(const char *name, uint32_t selector, uint32_t wireMask,
                    uint32_t status, uint64_t entryNanos, uint64_t preInvoke,
                    uint64_t postInvoke, uint64_t exitNanos,
                    const NPULaunchDeviceTiming *device,
                    const NPULaunchPhases *phases = nullptr,
                    const NPULaunchSyncReport *sync = nullptr);

  /// Names a phase the process crossed. The per-step fixed cost the one-process
  /// -per-denoise-step constraint imposes is the span between these.
  void mark(const char *phase);

  /// Writes what is buffered. Registered with `atexit`; also reachable for a
  /// test that reads the file back while the process still runs.
  void flush();

private:
  explicit NPUDispatchTimeline(std::string directory);

  /// Appends `line` under the lock, flushing if the buffer is large or every
  /// record was asked for.
  void append(const std::string &line);
  void flushLocked();

  std::mutex mu_;
  FILE *file_ = nullptr;
  std::string buffer_;
  uint64_t originNanos_ = 0;
  uint64_t nextSeq_ = 0;
  bool deviceTiming_ = false;
  bool devicePhases_ = false;
  bool flushEvery_ = false;
};

} // namespace M::Driver

#endif // SKYE_RUNTIME_QURT_NPUDISPATCHRECORDER_H
