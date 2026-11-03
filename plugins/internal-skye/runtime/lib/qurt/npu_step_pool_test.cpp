//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// NPU_STEP_ALLOC_POOL at the seam it moves: a freed chunk is parked by exact
// byte size and re-issued address-stable, instead of dying by `rpcmem_free`
// and being re-created — which re-creates its handle and re-pays the vendor
// driver's byte-proportional first-presentation charge every step
// (SINUSOID_DIAGNOSIS.md). Every claim below is judged by the driver calls
// that actually crossed the seam, counted at the stand-in's own chokepoint
// (`npu_fake_transport_stats`), plus the mapping log for the one leg where
// the pool must compose with MODULAR_NPU_MAP_ONCE.
//
// The properties, in the order a device leg would meet them:
//
//   * flag OFF, same binary, is the negative control: every free reaches
//     `rpcmem_free` and every allocation `rpcmem_alloc`, bit for bit today's
//     path — INCLUDING with a warm pool parked by an earlier armed phase, so
//     a flag flipped off mid-process cannot be half-treated;
//   * flag ON: free parks (no `rpcmem_free`), and the next same-size
//     allocation is THE SAME ADDRESS with no `rpcmem_alloc` — the handle
//     never died, which is the whole lever;
//   * buckets are exact-size: a different size after a park is a fresh
//     allocation, never a re-cut of the parked block, and the parked block
//     is still there for its own size afterwards;
//   * a dispatch through a re-issued block leaves byte-identical output to a
//     dispatch through a fresh block — the differential the stand-in's
//     deterministic writes make checkable;
//   * with MAP_ONCE on top, parking unmaps address-exact and the re-issued
//     block is mapped AFRESH on its first launch: `chunkFree`'s erasures run
//     verbatim on park, so no mapping (and by the same code path no
//     attribute, no constant class) outlives the park and rides the reuse —
//     and `chunkAlloc`'s mapping-outlived-buffer refusal never fires.
//
// Every leg runs kRounds times in one process: reuse is exactly the state
// hazard class that passes once and fails on the second pass, so the loop is
// part of the assertion. Rounds after the first also flip alloc-from-raw to
// alloc-from-pool, which is asserted per round rather than averaged away.
//
//===----------------------------------------------------------------------===//

#include "QuRTDeviceContext.h"

#include <dlfcn.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace M;
using namespace M::Driver;

namespace {

int failures = 0;

void expect(bool condition, const std::string &what) {
  if (!condition) {
    printf("FAIL: %s\n", what.c_str());
    ++failures;
  }
}

constexpr int kRounds = 3;

// One distinct size per leg, because the pool is process-lifetime state in a
// process-lifetime device: a leg that shared a size with another would be
// asserting against the other leg's parked blocks.
constexpr size_t kOffLegBytes = 64 << 10;
constexpr size_t kReuseLegBytes = 96 << 10;
constexpr size_t kMixedLegBytesA = 128 << 10;
constexpr size_t kMixedLegBytesB = 160 << 10;
constexpr size_t kMapLegBytes = 192 << 10;
constexpr size_t kBytesLegBytes = 224 << 10;

using StatsFn = void (*)(uint64_t *, int);
StatsFn statsFn = nullptr;

struct RawCalls {
  uint64_t allocs;
  uint64_t frees;
};

RawCalls rawCalls() {
  uint64_t out[2] = {0, 0};
  statsFn(out, 2);
  return {out[0], out[1]};
}

std::string logPath;

std::vector<std::string> takeLog() {
  std::vector<std::string> lines;
  std::ifstream in(logPath);
  for (std::string line; std::getline(in, line);)
    if (!line.empty())
      lines.push_back(line);
  in.close();
  std::ofstream(logPath, std::ios::trunc).close();
  return lines;
}

std::string ptrString(const void *ptr) {
  char text[32];
  snprintf(text, sizeof(text), "%p", ptr);
  return text;
}

/// One `vadd` launch naming `base` for all three of its buffer operands, the
/// sibling tests' shape.
ErrorOrSuccess launchOn(DeviceContext &ctx, const DeviceFunction &fn,
                        void *base) {
  uint32_t count = 256;
  void *a = base;
  void *args[4] = {&a, &a, &a, &count};
  size_t sizes[4] = {sizeof(void *), sizeof(void *), sizeof(void *),
                     sizeof(uint32_t)};
  RETURN_IF_ERROR(ctx.enqueueFunctionExecDirect(
      fn, /*gridX=*/1, /*gridY=*/1, /*gridZ=*/1, /*blockX=*/1, /*blockY=*/1,
      /*blockZ=*/1, /*sharedMemBytes=*/0, /*attrs=*/nullptr, /*numAttrs=*/0,
      args, /*argCount=*/4, sizes));
  return ctx.synchronize();
}

/// Allocates `bytes`, hands back the buffer, and asserts whether the request
/// crossed the driver seam as a raw `rpcmem_alloc` or was served without one.
DeviceBufferRef allocExpecting(DeviceContext &ctx, size_t bytes, bool raw,
                               const std::string &when) {
  RawCalls before = rawCalls();
  ErrorOr<DeviceBufferRef> buffer = ctx.enqueueCreateBuffer(bytes);
  expect(!buffer.isError(), when + ": allocation failed");
  if (buffer.isError())
    return DeviceBufferRef();
  RawCalls after = rawCalls();
  expect(after.allocs == before.allocs + (raw ? 1 : 0),
         when +
             (raw ? ": expected one rpcmem_alloc, saw "
                  : ": expected no rpcmem_alloc, saw ") +
             std::to_string(after.allocs - before.allocs));
  expect(after.frees == before.frees, when + ": an allocation freed something");
  return buffer.takeValue();
}

/// Releases `ref` and asserts whether the free crossed the seam as a raw
/// `rpcmem_free` (flag off) or was parked (flag on).
void freeExpecting(DeviceBufferRef &ref, bool raw, const std::string &when) {
  RawCalls before = rawCalls();
  ref.reset();
  RawCalls after = rawCalls();
  expect(after.frees == before.frees + (raw ? 1 : 0),
         when +
             (raw ? ": expected one rpcmem_free, saw "
                  : ": expected a park (no rpcmem_free), saw ") +
             std::to_string(after.frees - before.frees));
  expect(after.allocs == before.allocs, when + ": a free allocated something");
}

/// Flag OFF is the raw path, verbatim. The warm variant of this claim — a
/// pool already holding a block of the requested size must still not serve a
/// disarmed process — runs in `main` after the armed legs have parked.
void runOffLeg(DeviceContext &ctx, int round) {
  const std::string when = "off leg round " + std::to_string(round);
  unsetenv("NPU_STEP_ALLOC_POOL");
  DeviceBufferRef a = allocExpecting(ctx, kOffLegBytes, /*raw=*/true, when);
  freeExpecting(a, /*raw=*/true, when + " first free");
  DeviceBufferRef b =
      allocExpecting(ctx, kOffLegBytes, /*raw=*/true, when + " realloc");
  freeExpecting(b, /*raw=*/true, when + " second free");

  // Anything but exactly "1" is off, so a leg cannot be half-treated by a
  // stray value.
  setenv("NPU_STEP_ALLOC_POOL", "0", 1);
  DeviceBufferRef c =
      allocExpecting(ctx, kOffLegBytes, /*raw=*/true, when + " flag-zero");
  freeExpecting(c, /*raw=*/true, when + " flag-zero free");
  unsetenv("NPU_STEP_ALLOC_POOL");
}

/// The lever itself: same size, free then alloc, same address, no raw calls.
void runReuseLeg(DeviceContext &ctx, int round) {
  const std::string when = "reuse leg round " + std::to_string(round);
  setenv("NPU_STEP_ALLOC_POOL", "1", 1);

  // Round 0 has an empty bucket for this size and must go raw — the once-per
  // -process cost the diagnosis prices. Later rounds are served by the block
  // the previous round parked.
  DeviceBufferRef first =
      allocExpecting(ctx, kReuseLegBytes, /*raw=*/round == 0, when);
  void *base = first ? first->ptr() : nullptr;
  freeExpecting(first, /*raw=*/false, when + " park");

  DeviceBufferRef again =
      allocExpecting(ctx, kReuseLegBytes, /*raw=*/false, when + " reuse");
  expect(again && again->ptr() == base,
         when + ": reuse moved: parked " + ptrString(base) + ", got " +
             ptrString(again ? again->ptr() : nullptr));
  freeExpecting(again, /*raw=*/false, when + " re-park");
  unsetenv("NPU_STEP_ALLOC_POOL");
}

/// Buckets are exact-size: a different size never re-cuts a parked block, and
/// the parked block survives for its own size.
void runMixedLeg(DeviceContext &ctx, int round) {
  const std::string when = "mixed leg round " + std::to_string(round);
  setenv("NPU_STEP_ALLOC_POOL", "1", 1);

  DeviceBufferRef a =
      allocExpecting(ctx, kMixedLegBytesA, /*raw=*/round == 0, when + " A");
  void *baseA = a ? a->ptr() : nullptr;
  freeExpecting(a, /*raw=*/false, when + " park A");

  // The other size must not be served from A's bucket.
  DeviceBufferRef b =
      allocExpecting(ctx, kMixedLegBytesB, /*raw=*/round == 0, when + " B");
  if (round == 0)
    expect(b && b->ptr() != baseA,
           when + ": a fresh size was cut from the parked block");

  // A's block is still whole and still A-sized.
  DeviceBufferRef a2 =
      allocExpecting(ctx, kMixedLegBytesA, /*raw=*/false, when + " A reuse");
  expect(a2 && a2->ptr() == baseA,
         when + ": A's bucket lost its block to the B allocation");

  freeExpecting(b, /*raw=*/false, when + " park B");
  freeExpecting(a2, /*raw=*/false, when + " re-park A");
  unsetenv("NPU_STEP_ALLOC_POOL");
}

/// Pool x MAP_ONCE: parking unmaps address-exact (the erasures run verbatim),
/// and the re-issued block maps AFRESH on its first launch instead of riding
/// the parked lifetime's mapping. This is the observable half of the
/// bookkeeping-hygiene hazard — attribute and constant class leave by the
/// same erasure block this leg proves runs on park.
void runMapOnceLeg(DeviceContext &ctx, const DeviceFunction &fn, int round) {
  const std::string when = "map-once leg round " + std::to_string(round);
  setenv("NPU_STEP_ALLOC_POOL", "1", 1);
  setenv("MODULAR_NPU_MAP_ONCE", "1", 1);
  (void)takeLog();

  DeviceBufferRef buffer =
      allocExpecting(ctx, kMapLegBytes, /*raw=*/round == 0, when);
  if (!buffer)
    return;
  void *base = buffer->ptr();
  expect(takeLog().empty(), when + ": allocating mapped something");

  expect(!launchOn(ctx, fn, base).isError(), when + ": first dispatch failed");
  {
    std::vector<std::string> lines = takeLog();
    expect(lines.size() == 2 &&
               lines[0].rfind("to_fd " + ptrString(base), 0) == 0,
           when + ": first launch did not map the block (" +
               std::to_string(lines.size()) + " seam lines)");
  }

  // Parking must unmap, address-exact, exactly as a real free would.
  freeExpecting(buffer, /*raw=*/false, when + " park");
  {
    std::vector<std::string> lines = takeLog();
    expect(lines.size() == 1 &&
               lines[0].rfind("munmap " + ptrString(base), 0) == 0,
           when + ": the park did not unmap the block");
  }

  // The reuse is the same address with no raw alloc — and chunkAlloc's
  // mapping-outlived-buffer refusal must NOT fire, which is what this
  // allocation succeeding asserts.
  DeviceBufferRef reused =
      allocExpecting(ctx, kMapLegBytes, /*raw=*/false, when + " reuse");
  expect(reused && reused->ptr() == base, when + ": reuse moved");
  expect(takeLog().empty(), when + ": the reuse mapped at alloc time");

  // ...and its first launch maps it afresh: the parked mapping is dead, not
  // inherited.
  expect(!launchOn(ctx, fn, base).isError(), when + ": reuse dispatch failed");
  {
    std::vector<std::string> lines = takeLog();
    expect(lines.size() == 2,
           when + ": the re-issued block was not mapped afresh (" +
               std::to_string(lines.size()) + " seam lines)");
  }

  freeExpecting(reused, /*raw=*/false, when + " re-park");
  (void)takeLog();
  unsetenv("MODULAR_NPU_MAP_ONCE");
  unsetenv("NPU_STEP_ALLOC_POOL");
}

/// A dispatch through a re-issued block leaves the bytes a dispatch through a
/// fresh block leaves. rpcmem gives no zeroing guarantee, so the contract a
/// kernel may hold is "my output is what I wrote", and the stand-in's
/// deterministic writes make that comparable — the pooled arm is even seeded
/// with garbage first, so inheriting parked bytes would be caught if the
/// "kernel" failed to overwrite.
void runBytesLeg(DeviceContext &ctx, const DeviceFunction &fn, int round) {
  const std::string when = "bytes leg round " + std::to_string(round);
  setenv("NPU_STEP_ALLOC_POOL", "1", 1);

  uint8_t fresh[64];
  uint8_t pooled[64];
  static uint8_t garbage[kBytesLegBytes];
  memset(garbage, 0x5A, sizeof(garbage));

  DeviceBufferRef first =
      allocExpecting(ctx, kBytesLegBytes, /*raw=*/round == 0, when);
  if (!first)
    return;
  expect(!launchOn(ctx, fn, first->ptr()).isError(),
         when + ": fresh dispatch failed");
  expect(
      !ctx.enqueueCopyFromDevice(fresh, first->ptr(), sizeof(fresh)).isError(),
      when + ": fresh readback failed");
  freeExpecting(first, /*raw=*/false, when + " park");

  DeviceBufferRef reused =
      allocExpecting(ctx, kBytesLegBytes, /*raw=*/false, when + " reuse");
  if (!reused)
    return;
  expect(!ctx.enqueueCopyToDevice(reused->ptr(), garbage, kBytesLegBytes)
              .isError(),
         when + ": seeding failed");
  expect(!launchOn(ctx, fn, reused->ptr()).isError(),
         when + ": pooled dispatch failed");
  expect(!ctx.enqueueCopyFromDevice(pooled, reused->ptr(), sizeof(pooled))
              .isError(),
         when + ": pooled readback failed");
  expect(memcmp(fresh, pooled, sizeof(fresh)) == 0,
         when + ": a dispatch through the pooled block left different bytes");
  freeExpecting(reused, /*raw=*/false, when + " re-park");
  unsetenv("NPU_STEP_ALLOC_POOL");
}

} // namespace

int main() {
  const char *fake = getenv("NPU_FAKE_TRANSPORT");
  const char *tmp = getenv("TEST_TMPDIR");
  if (fake == nullptr || tmp == nullptr) {
    printf("FAIL: NPU_FAKE_TRANSPORT and TEST_TMPDIR must be set\n");
    return 1;
  }
  setenv("MODULAR_NPU_TRANSPORT_LIB",
         std::filesystem::absolute(fake).string().c_str(), 1);
  logPath = std::string(tmp) + "/mapping_calls.log";
  setenv("NPU_FAKE_TRANSPORT_ATTR_LOG", logPath.c_str(), 1);
  // The vote has its own test, and its dispatch would put calls of its own on
  // the seams this one counts.
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
  DeviceContext &ctx = **context;
  ErrorOr<DeviceFunctionRef> vadd = ctx.loadFunction("", "vadd", "");
  if (vadd.isError()) {
    printf("FAIL: loadFunction vadd: %s\n", vadd.getError());
    return 1;
  }

  // The driver has the library open; this resolves the same image rather than
  // loading a second copy, so the counters read are the ones being written.
  void *handle = dlopen(fake, RTLD_NOW);
  if (handle == nullptr) {
    printf("FAIL: dlopen %s: %s\n", fake, dlerror());
    return 1;
  }
  statsFn =
      reinterpret_cast<StatsFn>(dlsym(handle, "npu_fake_transport_stats"));
  if (statsFn == nullptr) {
    printf("FAIL: the stand-in exports no counters\n");
    return 1;
  }
  (void)takeLog();

  for (int round = 0; round < kRounds; ++round) {
    runOffLeg(ctx, round);
    runReuseLeg(ctx, round);
    runMixedLeg(ctx, round);
    runMapOnceLeg(ctx, **vadd, round);
    runBytesLeg(ctx, **vadd, round);
    // The pool now holds a block of the reuse leg's size, parked while armed.
    // A disarmed process must go raw anyway — the flag-off path may not even
    // look at the pool, warm or not.
    const std::string when = "warm off leg round " + std::to_string(round);
    DeviceBufferRef raw =
        allocExpecting(ctx, kReuseLegBytes, /*raw=*/true, when);
    freeExpecting(raw, /*raw=*/true, when + " free");
  }

  if (failures != 0) {
    printf("%d check(s) failed\n", failures);
    return 1;
  }
  printf("all checks passed\n");
  return 0;
}
