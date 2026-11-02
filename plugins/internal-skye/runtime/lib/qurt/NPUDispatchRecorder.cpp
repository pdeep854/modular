//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements per-dispatch IO recording for the NPU launch path.
//
//===----------------------------------------------------------------------===//

#include "NPUDispatchRecorder.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>

#ifdef _WIN32
#include <process.h>
#define NPU_RECORDER_GETPID _getpid
#else
#include <unistd.h>
#define NPU_RECORDER_GETPID getpid
#endif

namespace M::Driver {

namespace {

// The witness is every NPU kernel's LAST buffer slot, by the `NPU_KERNEL`
// declarations in `NPUKernels.def`. P5 M4.2 order 091: keying on its SIZE
// instead was wrong twice over -- the graph path was allocating 32 bytes, not
// 32 words, so an exact-size key never matched on 1612 of 1612 launches, and
// had it matched it would have read 18 words out of an 8-word buffer. The
// position is what the manifest actually guarantees; the size is what the
// allocation bug was in.
//
// 18 was the highest word any consumer had asked for (`MATMUL_WIT_ARM`). Order
// 149 leg B wanted word 22 and found the report silently one constant short of
// it -- the reader half of #393 VALIDATES the arm code and nothing on the render
// path REPORTS it, so the leg had to rebuild this library out of a worktree to
// read the word its own order was written around. 32 is `NPU_WITNESS_WORDS`, the
// whole buffer, so the cap stops being a thing to re-raise per word;
// `npuWitnessWordsToReport` still clamps to what the slot actually holds, which
// is what order 091's 8-word allocation needed.
constexpr uint32_t kNPUWitnessReported = 32;
} // namespace

uint32_t npuWitnessWordsToReport(uint64_t allocBytes, uint64_t offset) {
  uint64_t available = offset <= allocBytes ? allocBytes - offset : 0;
  uint64_t words = available / sizeof(uint32_t);
  return static_cast<uint32_t>(words > kNPUWitnessReported ? kNPUWitnessReported
                                                           : words);
}

namespace {

/// Word-wide FNV-1a fold: 8 bytes per multiply instead of one, because a
/// window hash runs over every operand of every dispatch and the byte-serial
/// form is multiply-latency-bound eight times over. Not interchangeable with
/// standard FNV-1a; nothing recomputes it outside this file -- comparisons are
/// hash-to-hash and the all-zero question is answered by the flag captured
/// alongside.
constexpr uint64_t kHashSeed = 14695981039346656037ull;
constexpr uint64_t kHashPrime = 1099511628211ull;

uint64_t envByteCount(const char *name, uint64_t fallback) {
  const char *value = getenv(name);
  if (value == nullptr || *value == '\0')
    return fallback;
  return strtoull(value, nullptr, 10);
}

} // namespace

NPUDispatchRecorder *NPUDispatchRecorder::get() {
  static NPUDispatchRecorder *instance = []() -> NPUDispatchRecorder * {
    const char *directory = getenv("MODULAR_NPU_DISPATCH_DUMP");
    if (directory == nullptr || *directory == '\0')
      return nullptr;
    auto *recorder = new NPUDispatchRecorder(directory);
    if (recorder->file_ == nullptr) {
      // A recorder that cannot write must not fail dispatches; say why once
      // and stay off.
      fprintf(stderr,
              "npu dispatch recording disabled: cannot open a dump file "
              "under %s\n",
              directory);
      delete recorder;
      return nullptr;
    }
    return recorder;
  }();
  return instance;
}

NPUDispatchRecorder::NPUDispatchRecorder(std::string directory)
    : directory_(std::move(directory)) {
  window_ = envByteCount("MODULAR_NPU_DISPATCH_DUMP_WINDOW", window_);
  const char *bytes = getenv("MODULAR_NPU_DISPATCH_DUMP_BYTES");
  dumpBytes_ = bytes != nullptr && strcmp(bytes, "1") == 0;
  byteCap_ = envByteCount("MODULAR_NPU_DISPATCH_DUMP_BYTES_CAP", byteCap_);

  std::error_code errorCode;
  std::filesystem::create_directories(directory_, errorCode);
  if (dumpBytes_)
    std::filesystem::create_directories(directory_ + "/bytes", errorCode);

  std::string path = directory_ + "/npu_iodump_" +
                     std::to_string(NPU_RECORDER_GETPID()) + ".jsonl";
  file_ = fopen(path.c_str(), "w");
  if (file_ == nullptr)
    return;
  fprintf(file_,
          "{\"k\":\"meta\",\"v\":1,\"pid\":%d,\"window\":%llu,"
          "\"bytes\":%d,\"byteCap\":%llu}\n",
          NPU_RECORDER_GETPID(), static_cast<unsigned long long>(window_),
          dumpBytes_ ? 1 : 0, static_cast<unsigned long long>(byteCap_));
  fflush(file_);
}

uint64_t NPUDispatchRecorder::ordinalOf(uintptr_t base) {
  auto [it, inserted] = allocOrdinals_.try_emplace(base, allocOrdinals_.size());
  return it->second;
}

void NPUDispatchRecorder::hashWindow(const void *ptr, size_t available,
                                     uint64_t &hash, bool &allZero,
                                     uint64_t &hashedBytes) {
  size_t count = available;
  if (window_ != 0 && count > window_)
    count = window_;
  hashedBytes = count;
  hash = kHashSeed;
  allZero = true;
  if (ptr == nullptr || count == 0) {
    allZero = false;
    return;
  }

  const auto *bytes = static_cast<const uint8_t *>(ptr);
  uint64_t accumulated = 0;
  size_t whole = count & ~size_t{7};
  for (size_t i = 0; i < whole; i += 8) {
    uint64_t word;
    memcpy(&word, bytes + i, sizeof(word));
    accumulated |= word;
    hash = (hash ^ word) * kHashPrime;
  }
  for (size_t i = whole; i < count; ++i) {
    accumulated |= bytes[i];
    hash = (hash ^ bytes[i]) * kHashPrime;
  }
  allZero = accumulated == 0;
}

void NPUDispatchRecorder::dumpWindowBytes(uint64_t seq, size_t slot,
                                          const char *leg, const void *ptr,
                                          size_t byteCount) {
  if (!dumpBytes_ || ptr == nullptr || byteCount == 0)
    return;
  if (bytesWritten_ + byteCount > byteCap_) {
    if (!byteCapReported_) {
      byteCapReported_ = true;
      fprintf(file_, "{\"k\":\"bytecap\",\"seq\":%llu,\"written\":%llu}\n",
              static_cast<unsigned long long>(seq),
              static_cast<unsigned long long>(bytesWritten_));
      fflush(file_);
    }
    return;
  }
  char name[64];
  snprintf(name, sizeof(name), "/bytes/d%llu_s%zu_%s.bin",
           static_cast<unsigned long long>(seq), slot, leg);
  std::string path = directory_ + name;
  FILE *out = fopen(path.c_str(), "wb");
  if (out == nullptr)
    return;
  bytesWritten_ += fwrite(ptr, 1, byteCount, out);
  fclose(out);
}

uint64_t NPUDispatchRecorder::beginLaunch(
    const char *name, uint32_t selector, uint32_t threadMask,
    const uint8_t *scalarBlock, size_t scalarBlockBytes, uint32_t scalarCount,
    uint32_t bufferCount, const QuRTRemoteBuf *bufferSlots) {
  std::lock_guard<std::mutex> lock(mu_);
  uint64_t seq = nextSeq_++;
  PendingLaunch launch;
  launch.name = name;
  launch.selector = selector;
  launch.threadMask = threadMask;

  launch.scalars.reserve(scalarCount);
  for (uint32_t i = 0; i < scalarCount; ++i) {
    uint64_t value = 0;
    if ((i + 1) * 8 <= scalarBlockBytes)
      memcpy(&value, scalarBlock + i * 8, sizeof(value));
    launch.scalars.push_back(value);
  }

  launch.slots.reserve(bufferCount);
  for (uint32_t i = 0; i < bufferCount; ++i) {
    SlotCapture capture = {};
    capture.base = reinterpret_cast<uintptr_t>(bufferSlots[i].pv);
    capture.allocOrdinal = ordinalOf(capture.base);
    capture.allocBytes = bufferSlots[i].nLen;
    // The offset as packed for the wire, in the slot after the scalars; read
    // back rather than passed in so the record is of what crossed.
    size_t offsetSlot = (scalarCount + i) * 8;
    if (offsetSlot + 8 <= scalarBlockBytes)
      memcpy(&capture.offset, scalarBlock + offsetSlot, sizeof(capture.offset));

    size_t available = 0;
    if (capture.offset <= capture.allocBytes)
      available = capture.allocBytes - capture.offset;
    const void *window =
        reinterpret_cast<const void *>(capture.base + capture.offset);
    hashWindow(window, available, capture.preHash, capture.preZero,
               capture.hashedBytes);
    dumpWindowBytes(seq, i, "pre", window, capture.hashedBytes);
    launch.slots.push_back(capture);
  }

  // P5 M4.2 order 088: the ISSUE half of the record, written and flushed BEFORE
  // the invoke crosses. Until now a launch record existed only once the call
  // returned, so a dispatch that was issued and never came back was
  // indistinguishable on the wire from one the host never issued -- which is
  // exactly the pair 088's stall could not separate (`seq=691` returned, seq
  // 692 simply absent). An `issue` with no matching `launch` now means the
  // device has it and is not giving it back; no `issue` at all means the host
  // stopped.
  if (file_ != nullptr)
    fprintf(file_,
            "{\"k\":\"issue\",\"seq\":%llu,\"name\":\"%s\",\"sel\":%u,"
            "\"mask\":%u}\n",
            static_cast<unsigned long long>(seq), launch.name.c_str(), selector,
            threadMask);
  if (file_ != nullptr)
    fflush(file_);

  pending_.emplace(seq, std::move(launch));
  return seq;
}

void NPUDispatchRecorder::endLaunch(uint64_t token, uint32_t status) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = pending_.find(token);
  if (it == pending_.end())
    return;
  PendingLaunch &launch = it->second;
  for (size_t i = 0; i < launch.slots.size(); ++i) {
    SlotCapture &capture = launch.slots[i];
    uint64_t hashedBytes = 0;
    const void *window =
        reinterpret_cast<const void *>(capture.base + capture.offset);
    size_t available = 0;
    if (capture.offset <= capture.allocBytes)
      available = capture.allocBytes - capture.offset;
    hashWindow(window, available, capture.postHash, capture.postZero,
               hashedBytes);
    dumpWindowBytes(token, i, "post", window, hashedBytes);
  }
  writeLaunchRecord(token, launch, status);
  pending_.erase(it);
}

void NPUDispatchRecorder::writeLaunchRecord(uint64_t seq,
                                            const PendingLaunch &launch,
                                            uint32_t status) {
  fprintf(file_,
          "{\"k\":\"launch\",\"seq\":%llu,\"name\":\"%s\",\"sel\":%u,"
          "\"mask\":%u,\"status\":%u,\"scalars\":[",
          static_cast<unsigned long long>(seq), launch.name.c_str(),
          launch.selector, launch.threadMask, status);
  for (size_t i = 0; i < launch.scalars.size(); ++i)
    fprintf(file_, "%s\"0x%llx\"", i == 0 ? "" : ",",
            static_cast<unsigned long long>(launch.scalars[i]));
  // P5 M4.2 order 088: the witness words, verbatim, for any slot shaped like
  // one. The kernels have always written them and the render path has never
  // read them, so a stall there could say nothing about which unit ran, which
  // role plan was used, or -- since #303 -- whether the array thread was inside
  // the HMX lock or inside a rendezvous. Keyed on the slot's SIZE rather than
  // on a selector, so every kernel that carries a witness reports one.
  if (!launch.slots.empty()) {
    const SlotCapture &capture = launch.slots.back();
    // Bounded by what the slot actually holds, never by what the kernel is
    // believed to write: a short witness truncates and says so by its length,
    // which is how order 091's 8-word allocation would have presented as a
    // finding instead of as a buffer overread.
    uint64_t available = capture.offset <= capture.allocBytes
                             ? capture.allocBytes - capture.offset
                             : 0;
    uint32_t words =
        npuWitnessWordsToReport(capture.allocBytes, capture.offset);
    if (words > 0) {
      const uint32_t *witness =
          reinterpret_cast<const uint32_t *>(capture.base + capture.offset);
      fprintf(file_, "],\"witBytes\":%llu,\"wit\":[",
              static_cast<unsigned long long>(available));
      for (uint32_t i = 0; i < words; ++i)
        fprintf(file_, "%s%u", i == 0 ? "" : ",", witness[i]);
    }
  }
  fprintf(file_, "],\"bufs\":[");
  for (size_t i = 0; i < launch.slots.size(); ++i) {
    const SlotCapture &capture = launch.slots[i];
    fprintf(file_,
            "%s{\"slot\":%zu,\"alloc\":%llu,\"base\":\"0x%llx\","
            "\"asz\":%llu,\"off\":%llu,\"n\":%llu,"
            "\"pre\":\"0x%llx\",\"preZ\":%d,"
            "\"post\":\"0x%llx\",\"postZ\":%d}",
            i == 0 ? "" : ",", i,
            static_cast<unsigned long long>(capture.allocOrdinal),
            static_cast<unsigned long long>(capture.base),
            static_cast<unsigned long long>(capture.allocBytes),
            static_cast<unsigned long long>(capture.offset),
            static_cast<unsigned long long>(capture.hashedBytes),
            static_cast<unsigned long long>(capture.preHash),
            capture.preZero ? 1 : 0,
            static_cast<unsigned long long>(capture.postHash),
            capture.postZero ? 1 : 0);
  }
  fprintf(file_, "]}\n");
  fflush(file_);
}

uint64_t NPUDispatchRecorder::windowHashOf(const void *ptr, size_t byteSize) {
  std::lock_guard<std::mutex> lock(mu_);
  uint64_t hash = 0;
  uint64_t hashedBytes = 0;
  bool allZero = false;
  hashWindow(ptr, byteSize, hash, allZero, hashedBytes);
  return hash;
}

void NPUDispatchRecorder::recordHostOp(const char *kind, const void *dst,
                                       uintptr_t dstAllocBase,
                                       size_t dstAllocBytes, const void *src,
                                       size_t byteSize, uint64_t dstPreHash) {
  std::lock_guard<std::mutex> lock(mu_);
  uint64_t seq = nextSeq_++;

  uint64_t dstPost = 0;
  uint64_t dstHashed = 0;
  bool dstZero = false;
  hashWindow(dst, byteSize, dstPost, dstZero, dstHashed);

  uint64_t srcHash = 0;
  uint64_t srcHashed = 0;
  bool srcZero = false;
  if (src != nullptr)
    hashWindow(src, byteSize, srcHash, srcZero, srcHashed);

  // -1 marks a destination outside any device allocation (a host buffer), so
  // the differ never mistakes it for allocation 0.
  long long dstAlloc = -1;
  unsigned long long dstOffset = 0;
  if (dstAllocBase != 0) {
    dstAlloc = static_cast<long long>(ordinalOf(dstAllocBase));
    dstOffset = reinterpret_cast<uintptr_t>(dst) - dstAllocBase;
  }

  fprintf(file_,
          "{\"k\":\"%s\",\"seq\":%llu,\"n\":%llu,"
          "\"dst\":\"0x%llx\",\"dstAlloc\":%lld,\"dstAsz\":%llu,"
          "\"dstOff\":%llu,\"src\":\"0x%llx\",\"hashed\":%llu,"
          "\"srcHash\":\"0x%llx\",\"dstPre\":\"0x%llx\","
          "\"dstPost\":\"0x%llx\",\"dstPostZ\":%d}\n",
          kind, static_cast<unsigned long long>(seq),
          static_cast<unsigned long long>(byteSize),
          static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(dst)),
          dstAlloc, static_cast<unsigned long long>(dstAllocBytes), dstOffset,
          static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(src)),
          static_cast<unsigned long long>(dstHashed),
          static_cast<unsigned long long>(srcHash),
          static_cast<unsigned long long>(dstPreHash),
          static_cast<unsigned long long>(dstPost), dstZero ? 1 : 0);
  fflush(file_);
}

//===----------------------------------------------------------------------===//
// NPUDispatchTimeline
//===----------------------------------------------------------------------===//

namespace {

/// Buffered bytes before a write goes out. Small enough that a run killed
/// mid-step loses at most this much, large enough that the write itself is not
/// what the inter-launch gap is measuring.
constexpr size_t kTimelineFlushBytes = 1 << 16;

bool envIsOne(const char *name) {
  const char *value = getenv(name);
  return value != nullptr && strcmp(value, "1") == 0;
}

} // namespace

uint64_t NPUDispatchTimeline::nowNanos() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

NPUDispatchTimeline *NPUDispatchTimeline::get() {
  static NPUDispatchTimeline *instance = []() -> NPUDispatchTimeline * {
    const char *directory = getenv("MODULAR_NPU_PERF_LOG");
    if (directory == nullptr || *directory == '\0')
      return nullptr;
    auto *timeline = new NPUDispatchTimeline(directory);
    if (timeline->file_ == nullptr) {
      // A timeline that cannot write must not fail dispatches.
      fprintf(stderr,
              "npu perf timing disabled: cannot open a log file under %s\n",
              directory);
      delete timeline;
      return nullptr;
    }
    // Never destroyed, like the recorder, so the tail is written here rather
    // than by a destructor that does not run.
    atexit([]() {
      if (NPUDispatchTimeline *live = NPUDispatchTimeline::get())
        live->flush();
    });
    return timeline;
  }();
  return instance;
}

NPUDispatchTimeline::NPUDispatchTimeline(std::string directory) {
  devicePhases_ = envIsOne("MODULAR_NPU_PERF_PHASES");
  // The phase block rides the device timing slot, so asking for it is
  // asking for that slot too. Implied rather than refused: a run that set
  // only the phase variable and got nothing back would look exactly like a
  // skel built without the probe.
  deviceTiming_ = devicePhases_ || envIsOne("MODULAR_NPU_PERF_DEVICE");
  flushEvery_ = envIsOne("MODULAR_NPU_PERF_FLUSH");

  std::error_code errorCode;
  std::filesystem::create_directories(directory, errorCode);
  std::string path = directory + "/npu_perf_" +
                     std::to_string(NPU_RECORDER_GETPID()) + ".jsonl";
  file_ = fopen(path.c_str(), "w");
  if (file_ == nullptr)
    return;

  originNanos_ = nowNanos();
  // The wall clock is recorded once, beside the monotonic origin, so records
  // from the several processes one denoise takes can be laid end to end; every
  // stamp after this is monotonic and relative, which no clock adjustment can
  // move.
  auto epochNanos = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count();
  char line[256];
  snprintf(line, sizeof(line),
           "{\"k\":\"meta\",\"v\":1,\"pid\":%d,\"epoch_ns\":%lld,"
           "\"device\":%d,\"iodump\":%d}\n",
           NPU_RECORDER_GETPID(), static_cast<long long>(epochNanos),
           deviceTiming_ ? 1 : 0, getenv("MODULAR_NPU_DISPATCH_DUMP") ? 1 : 0);
  buffer_ = line;
  flush();
}

void NPUDispatchTimeline::append(const std::string &line) {
  std::lock_guard<std::mutex> lock(mu_);
  buffer_ += line;
  if (flushEvery_ || buffer_.size() >= kTimelineFlushBytes)
    flushLocked();
}

void NPUDispatchTimeline::flush() {
  std::lock_guard<std::mutex> lock(mu_);
  flushLocked();
}

void NPUDispatchTimeline::flushLocked() {
  if (file_ == nullptr || buffer_.empty())
    return;
  fwrite(buffer_.data(), 1, buffer_.size(), file_);
  fflush(file_);
  buffer_.clear();
}

void NPUDispatchTimeline::recordLaunch(const char *name, uint32_t selector,
                                       uint32_t wireMask, uint32_t status,
                                       uint64_t entryNanos, uint64_t preInvoke,
                                       uint64_t postInvoke, uint64_t exitNanos,
                                       const NPULaunchDeviceTiming *device,
                                       const NPULaunchPhases *phases,
                                       const NPULaunchSyncReport *sync) {
  uint64_t seq = 0;
  {
    std::lock_guard<std::mutex> lock(mu_);
    seq = nextSeq_++;
  }

  char line[512];
  int written = snprintf(
      line, sizeof(line),
      "{\"k\":\"launch\",\"seq\":%llu,\"name\":\"%s\",\"sel\":%u,"
      "\"mask\":%u,\"status\":%u,\"t\":%llu,\"prep\":%llu,\"invoke\":%llu,"
      "\"post\":%llu",
      static_cast<unsigned long long>(seq), name, selector, wireMask, status,
      static_cast<unsigned long long>(entryNanos - originNanos_),
      static_cast<unsigned long long>(preInvoke - entryNanos),
      static_cast<unsigned long long>(postInvoke - preInvoke),
      static_cast<unsigned long long>(exitNanos - postInvoke));
  std::string record(line, written > 0 ? static_cast<size_t>(written) : 0);
  if (device != nullptr) {
    snprintf(line, sizeof(line), ",\"dev\":[%llu,%llu,%llu,%llu,%llu,%llu]",
             static_cast<unsigned long long>(device->entryPcycles),
             static_cast<unsigned long long>(device->threadsCreatedPcycles),
             static_cast<unsigned long long>(device->kernelReturnedPcycles),
             static_cast<unsigned long long>(device->exitPcycles),
             static_cast<unsigned long long>(device->entryQtimer),
             static_cast<unsigned long long>(device->exitQtimer));
    record += line;
  }
  if (phases != nullptr) {
    // One array per participant, and only for participants that actually
    // marked: a thread with no marks was never instrumented, and emitting its
    // zeros beside real numbers is how a reader comes to average them.
    //
    // Field order is the struct's, stated once here because the reader is a
    // position-decoded array rather than named keys -- keeping a launch record
    // to one line, which is what makes the log greppable at 403 launches a
    // step. `npu_perf_summary.py` names the columns.
    std::string block;
    for (uint32_t t = 0; t < kNPUPhaseMaxThreads; ++t) {
      const NPULaunchPhaseThread &th = phases->thread[t];
      if (th.marks == 0)
        continue;
      snprintf(line, sizeof(line), "%s[%u", block.empty() ? "" : ",", t);
      block += line;
      for (uint32_t p = 0; p < kNPUPhaseCount; ++p) {
        snprintf(line, sizeof(line), ",%llu,%llu",
                 static_cast<unsigned long long>(th.pcycles[p]),
                 static_cast<unsigned long long>(th.qticks[p]));
        block += line;
      }
      snprintf(line, sizeof(line), ",%llu,%llu,%llu,%llu,%llu,%llu",
               static_cast<unsigned long long>(th.waits),
               static_cast<unsigned long long>(th.lastIn),
               static_cast<unsigned long long>(th.spinExits),
               static_cast<unsigned long long>(th.napExits),
               static_cast<unsigned long long>(th.giveUps),
               static_cast<unsigned long long>(th.naps));
      block += line;
      for (uint32_t b = 0; b < kNPUWaitHistBuckets; ++b) {
        snprintf(line, sizeof(line), ",%llu",
                 static_cast<unsigned long long>(th.waitHist[b]));
        block += line;
      }
      block += "]";
    }
    if (!block.empty()) {
      record += ",\"ph\":[";
      record += block;
      record += "]";
    }
  }
  if (sync != nullptr) {
    // Named keys, unlike `ph`: one small object per record, and the readers
    // that need it (the park-cost verdict above all) are jq one-liners on the
    // lane, not the summary tool. Values are session-cumulative by contract.
    snprintf(line, sizeof(line),
             ",\"sync\":{\"waits\":%llu,\"last_in\":%llu,\"spin\":%llu,"
             "\"nap\":%llu,\"giveup\":%llu,\"naps\":%llu,\"wait_qt\":%llu,"
             "\"nap_qt\":%llu,\"budget_hi\":%llu}",
             static_cast<unsigned long long>(sync->waits),
             static_cast<unsigned long long>(sync->lastIn),
             static_cast<unsigned long long>(sync->spinExits),
             static_cast<unsigned long long>(sync->napExits),
             static_cast<unsigned long long>(sync->giveUps),
             static_cast<unsigned long long>(sync->naps),
             static_cast<unsigned long long>(sync->waitQticks),
             static_cast<unsigned long long>(sync->napWaitQticks),
             static_cast<unsigned long long>(sync->budgetHi));
    record += line;
  }
  record += "}\n";
  append(record);
}

void NPUDispatchTimeline::mark(const char *phase) {
  char line[256];
  snprintf(line, sizeof(line), "{\"k\":\"mark\",\"phase\":\"%s\",\"t\":%llu}\n",
           phase, static_cast<unsigned long long>(nowNanos() - originNanos_));
  append(line);
  // A mark is a phase boundary, which is exactly where a run that dies should
  // still have said how far it got.
  flush();
}

} // namespace M::Driver

extern "C" void npu_perf_mark(const char *phase) {
  if (M::Driver::NPUDispatchTimeline *timeline =
          M::Driver::NPUDispatchTimeline::get())
    timeline->mark(phase);
}
