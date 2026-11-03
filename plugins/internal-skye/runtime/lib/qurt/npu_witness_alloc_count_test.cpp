//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
// What a per-launch scratch buffer costs a batch.
//
// Every registered graph op USED TO allocate a witness buffer before its launch
// and destroy it after. The MemoryManager is disabled by construction on this
// backend (`QuRTDeviceContext.cpp:802-807`), so each of those was a live
// `rpcmem_alloc` and a live `rpcmem_free` -- and `memAlloc`/`memFree` each call
// `flushBatch()` (`:2162`, `:2180`), which sends whatever the batch was holding.
//
// This measurement is what put the witness arena in: the stubs now take a slot
// out of one process-lifetime allocation (`_witness_slot`, `registrations.mojo`)
// and the pair is gone. The arms below are kept as the standing proof of WHY --
// re-introducing a per-launch buffer anywhere on the launch path costs the batch
// its width, and this is the test that says so without a device.
//
// The audit prices the pairs at [MODEL] 0.3-27 s, a range wide enough to be
// either noise or half the stage, and the per-call cost is a device question.
// The COUNT is not: it is structural, it is disputed, and it decides whether
// BATCH/QUEUE/MEGASTEP can hold anything at all. This measures it with no
// device, by counting at the one chokepoint every allocation passes through.
//
// Two arms over the same launch sequence, `MODULAR_NPU_BATCH=1` on both:
//
//   * PER-LAUNCH -- a scratch buffer created before each dispatch and released
//     after, which is the shape the graph path has today;
//   * PROCESS-LIFETIME -- one scratch buffer created once and reused, which is
//     the shape the audit proposes.
//
// The difference in mean batch width is the finding.
//
//===----------------------------------------------------------------------===//

#include "QuRTDeviceContext.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <string>

using namespace M;
using namespace M::Driver;

namespace {

int failures = 0;

void expect(bool condition, const std::string &what) {
  printf("%s: %s\n", condition ? "ok" : "FAIL", what.c_str());
  if (!condition)
    ++failures;
}

/// Mirrors `npu_fake_transport.c`'s order. Restated rather than shared, for the
/// same reason the wire layout there is.
struct Stats {
  uint64_t allocs, frees, allocBytes, batches, batchedLaunches, singleLaunches;

  double meanWidth() const {
    return batches == 0 ? 0.0
                        : static_cast<double>(batchedLaunches) /
                              static_cast<double>(batches);
  }
};

using StatsFn = void (*)(uint64_t *, int);
using ResetFn = void (*)();

StatsFn statsFn = nullptr;
ResetFn resetFn = nullptr;

Stats readStats() {
  uint64_t raw[6] = {};
  statsFn(raw, 6);
  return Stats{raw[0], raw[1], raw[2], raw[3], raw[4], raw[5]};
}

constexpr size_t kWitnessBytes = 512;
constexpr size_t kOperandBytes = 4096;
constexpr int kLaunches = 8;

/// `vadd`, whose manifest row is three buffers and one scalar. The witness
/// stands in for the graph path's: a small buffer that is not an operand of
/// interest, allocated on one schedule or another.
int runArm(DeviceContext &ctx, bool perLaunch, Stats &out) {
  ErrorOr<DeviceBufferRef> arena = ctx.enqueueCreateBuffer(kOperandBytes * 4);
  if (arena.isError()) {
    printf("arena: %s\n", arena.getError());
    return 1;
  }
  auto *bytes = static_cast<uint8_t *>((*arena)->ptr());

  DeviceBufferRef held;
  if (!perLaunch) {
    ErrorOr<DeviceBufferRef> witness = ctx.enqueueCreateBuffer(kWitnessBytes);
    if (witness.isError()) {
      printf("witness: %s\n", witness.getError());
      return 1;
    }
    held = witness.takeValue();
  }

  resetFn();
  for (int i = 0; i < kLaunches; ++i) {
    DeviceBufferRef perCall;
    if (perLaunch) {
      ErrorOr<DeviceBufferRef> witness = ctx.enqueueCreateBuffer(kWitnessBytes);
      if (witness.isError()) {
        printf("witness: %s\n", witness.getError());
        return 1;
      }
      perCall = witness.takeValue();
    }
    void *a = bytes;
    void *b = bytes + kOperandBytes;
    void *c = perLaunch ? perCall->ptr() : held->ptr();
    uint32_t count = 64;
    void *args[4] = {&a, &b, &c, &count};
    size_t sizes[4] = {sizeof(void *), sizeof(void *), sizeof(void *),
                       sizeof(uint32_t)};
    ErrorOr<DeviceFunctionRef> vadd = ctx.loadFunction("", "vadd", "");
    if (vadd.isError()) {
      printf("loadFunction: %s\n", vadd.getError());
      return 1;
    }
    ErrorOrSuccess result = ctx.enqueueFunctionExecDirect(
        **vadd, /*gridX=*/1, /*gridY=*/1, /*gridZ=*/1, /*blockX=*/1,
        /*blockY=*/1, /*blockZ=*/1, /*sharedMemBytes=*/0, /*attrs=*/nullptr,
        /*numAttrs=*/0, args, /*argCount=*/4, sizes);
    if (result.isError()) {
      printf("dispatch: %s\n", result.getError());
      return 1;
    }
    // Released here in the per-launch arm, exactly as a registered op's
    // witness is destroyed when its launch returns.
  }
  ErrorOrSuccess sync = ctx.synchronize();
  if (sync.isError()) {
    printf("synchronize: %s\n", sync.getError());
    return 1;
  }
  out = readStats();
  return 0;
}

} // namespace

int main() {
  const char *fakeLib = getenv("NPU_FAKE_TRANSPORT");
  if (fakeLib == nullptr || *fakeLib == '\0') {
    printf("FAIL: NPU_FAKE_TRANSPORT is not set\n");
    return 1;
  }
  setenv("MODULAR_NPU_TRANSPORT_LIB", fakeLib, 1);
  setenv("MODULAR_NPU_BATCH", "1", 1);
  setenv("MODULAR_NPU_POWER", "0", 1);

  ErrorOr<DeviceRef> device = QuRTDevice::create(0);
  if (device.isError()) {
    printf("FAIL: device: %s\n", device.getError());
    return 1;
  }
  ErrorOr<DeviceContextRef> context = (*device)->createContext();
  if (context.isError()) {
    printf("FAIL: context: %s\n", context.getError());
    return 1;
  }

  // The driver has the library open; this resolves the same image rather than
  // loading a second copy, so the counters read are the ones being written.
  void *handle = dlopen(fakeLib, RTLD_NOW);
  if (handle == nullptr) {
    printf("FAIL: dlopen %s: %s\n", fakeLib, dlerror());
    return 1;
  }
  statsFn =
      reinterpret_cast<StatsFn>(dlsym(handle, "npu_fake_transport_stats"));
  resetFn = reinterpret_cast<ResetFn>(
      dlsym(handle, "npu_fake_transport_stats_reset"));
  if (statsFn == nullptr || resetFn == nullptr) {
    printf("FAIL: the stand-in exports no counters\n");
    return 1;
  }

  Stats perLaunch{}, lifetime{};
  if (runArm(**context, /*perLaunch=*/true, perLaunch) != 0)
    return 1;
  if (runArm(**context, /*perLaunch=*/false, lifetime) != 0)
    return 1;

  printf("per-launch witness:      %llu allocs, %llu frees, %llu batches over "
         "%llu batched launches (mean width %.2f), %llu sent singly\n",
         (unsigned long long)perLaunch.allocs,
         (unsigned long long)perLaunch.frees,
         (unsigned long long)perLaunch.batches,
         (unsigned long long)perLaunch.batchedLaunches, perLaunch.meanWidth(),
         (unsigned long long)perLaunch.singleLaunches);
  printf("process-lifetime witness: %llu allocs, %llu frees, %llu batches over "
         "%llu batched launches (mean width %.2f), %llu sent singly\n",
         (unsigned long long)lifetime.allocs,
         (unsigned long long)lifetime.frees,
         (unsigned long long)lifetime.batches,
         (unsigned long long)lifetime.batchedLaunches, lifetime.meanWidth(),
         (unsigned long long)lifetime.singleLaunches);

  // The count, which is the structural claim: one allocation and one free per
  // launch, against none.
  expect(perLaunch.allocs >= kLaunches && perLaunch.frees >= kLaunches,
         "the per-launch arm allocates and frees once per launch");
  expect(lifetime.allocs == 0 && lifetime.frees == 0,
         "the process-lifetime arm allocates nothing inside the launch loop");

  // The consequence, which is why the count matters: an allocation between two
  // launches flushes the batch, so the batch cannot grow.
  expect(lifetime.meanWidth() > perLaunch.meanWidth(),
         "the per-launch allocation is what caps the batch width");

  // The SECOND block, and it is independent of the first. Every registered op
  // seeds its witness, enqueues, and reads the slot back immediately -- the
  // registration's own comment says why that is sound: "an invocation on this
  // transport completes before it returns". Batching is exactly what makes
  // that untrue. So hoisting the witness removes the flush and lands the graph
  // path on this, where the guard reads a slot the kernel has not written yet.
  {
    constexpr uint32_t kUnset = 0xFFFFFFFFu;
    // A context of its own per arm: `batching_` is latched on a context's
    // first launch (`QuRTDeviceContext.cpp:1696`), so the transport cannot be
    // changed under one that has already dispatched.
    auto seedEnqueueRead = [&](bool batch, uint32_t *afterFlushOut) {
      setenv("MODULAR_NPU_BATCH", batch ? "1" : "0", 1);
      ErrorOr<DeviceContextRef> own = (*device)->createContext();
      if (own.isError())
        return kUnset;
      DeviceContext &ctx = **own;
      ErrorOr<DeviceBufferRef> witness = ctx.enqueueCreateBuffer(kWitnessBytes);
      ErrorOr<DeviceBufferRef> arena =
          ctx.enqueueCreateBuffer(kOperandBytes * 4);
      if (witness.isError() || arena.isError())
        return kUnset;
      auto *slot = static_cast<uint32_t *>((*witness)->ptr());
      auto *bytes = static_cast<uint8_t *>((*arena)->ptr());
      *slot = kUnset;
      void *a = bytes;
      void *b = bytes + kOperandBytes;
      void *c = (*witness)->ptr();
      uint32_t count = 64;
      void *args[4] = {&a, &b, &c, &count};
      size_t sizes[4] = {sizeof(void *), sizeof(void *), sizeof(void *),
                         sizeof(uint32_t)};
      ErrorOr<DeviceFunctionRef> vadd = ctx.loadFunction("", "vadd", "");
      if (vadd.isError())
        return kUnset;
      ErrorOrSuccess r = ctx.enqueueFunctionExecDirect(
          **vadd, 1, 1, 1, 1, 1, 1, 0, nullptr, 0, args, 4, sizes);
      if (r.isError())
        return kUnset;
      // Exactly what a registration does next: read the slot, with nothing
      // between the enqueue and the read.
      const uint32_t seen = *slot;
      ErrorOrSuccess flushed = ctx.synchronize();
      (void)flushed;
      if (afterFlushOut != nullptr)
        *afterFlushOut = *slot;
      return seen;
    };

    uint32_t afterFlush = kUnset;
    const uint32_t synchronous = seedEnqueueRead(/*batch=*/false, nullptr);
    const uint32_t batched = seedEnqueueRead(/*batch=*/true, &afterFlush);
    setenv("MODULAR_NPU_BATCH", "1", 1);

    printf("witness read straight after enqueue: synchronous 0x%08x, batched "
           "0x%08x, batched after the flush 0x%08x (unset is 0x%08x)\n",
           synchronous, batched, afterFlush, kUnset);
    expect(synchronous != kUnset,
           "on the synchronous transport the witness is written by the time "
           "the enqueue returns -- which is what every registration assumes");
    expect(batched == kUnset,
           "under batching the same read sees an unwritten slot: the launch "
           "has not run, so a read-after-enqueue guard reads nothing");
    expect(afterFlush != kUnset,
           "the slot is written once the batch is flushed -- the value is not "
           "lost, it is late");
  }

  if (failures != 0) {
    printf("%d check(s) failed\n", failures);
    return 1;
  }
  printf("all checks passed\n");
  return 0;
}
