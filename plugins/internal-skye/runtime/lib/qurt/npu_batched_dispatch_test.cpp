//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Batched dispatch, judged the only way a batched transport can be judged: a
// differential.
//
// A batch is not a faster way to do the same thing unless it does the same
// thing, and the way it stops doing the same thing is not a fault. It is a
// launch reading an operand at the wrong offset, or out of a slot another
// launch's operand also landed in — wrong numbers, from a run that returns
// zero. So the test is not a benchmark and not a self-consistency check: the
// same chain of launches runs through the per-call path and through the
// batched path, into arenas that start identical, and every byte has to match.
//
// Three controls, because a differential that cannot fail proves nothing:
//
//   * one launch's scalar is changed in the batched leg, which MUST mismatch —
//     otherwise the comparison is not reading the dispatched bytes at all;
//   * the batched leg has to actually batch, which the stand-in transport
//     reports by counting launches per invocation rather than by a clock;
//   * a batch sent to a transport that predates batching has to be REFUSED,
//     since a signed skel older than the host is the one configuration where
//     a silently misread block is reachable in production.
//
// What this cannot test is the reason batching is delicate: the driver's
// per-call cache maintenance. A host has no NSP caches to leave stale, so the
// coherence half of the argument is settled on silicon, by the same shape of
// differential, with the batch boundary's maintenance removed in a control run
// that must mismatch.
//
//===----------------------------------------------------------------------===//

#include "QuRTDeviceContext.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
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

constexpr size_t kArenaBytes = 1 << 16;

// A second allocation, so a chain can name operands the first one does not
// carry: what bounds a batch is the number of distinct allocations its
// launches name, and a chain confined to one would never exercise that.
constexpr size_t kSideBytes = 1 << 12;

int childFail(const char *stage, const char *error) {
  printf("child %s: %s\n", stage, error);
  return 1;
}

/// The chain both legs run: launches that share operands, launches that do not,
/// and operands at a nonzero offset within their allocation — the three things
/// a slot map has to get right.
int runChain(DeviceContext &ctx, uint8_t *arena, uint8_t *side, bool perturb) {
  ErrorOr<DeviceFunctionRef> vadd = ctx.loadFunction("", "vadd", "");
  if (vadd.isError())
    return childFail("loadFunction vadd", vadd.getError());
  ErrorOr<DeviceFunctionRef> scale = ctx.loadFunction("", "vscale_offset", "");
  if (scale.isError())
    return childFail("loadFunction vscale_offset", scale.getError());

  void *a = arena;
  void *b = arena + 4096;
  void *c = arena + 8192;
  void *d = side;
  void *e = side + 1024;

  // One launch's destination, moved 128 bytes within the allocation it already
  // named. That is the divergence a mis-built slot map or a dropped offset
  // actually produces: the right allocation, the wrong place inside it, and a
  // dispatch that still returns zero.
  void *aShifted = perturb ? arena + 128 : arena;

  uint32_t count = 256;
  float scaleBy = 2.0f;
  float offsetBy = 1.0f;

  // vadd takes three buffers and a count; vscale_offset takes two buffers, two
  // floats and a count.
  void *addArgs[4] = {&a, &b, &c, &count};
  size_t addSizes[4] = {sizeof(void *), sizeof(void *), sizeof(void *),
                        sizeof(uint32_t)};
  void *addAgainArgs[4] = {&b, &c, &aShifted, &count};
  void *crossArgs[4] = {&c, &d, &e, &count};
  void *scaleArgs[5] = {&d, &b, &scaleBy, &offsetBy, &count};
  size_t scaleSizes[5] = {sizeof(void *), sizeof(void *), sizeof(float),
                          sizeof(float), sizeof(uint32_t)};

  struct Launch {
    DeviceFunction *function;
    void **args;
    size_t *sizes;
    uint32_t argCount;
  };
  Launch chain[] = {
      {&**vadd, addArgs, addSizes, 4},
      {&**vadd, addAgainArgs, addSizes, 4},
      {&**scale, scaleArgs, scaleSizes, 5},
      {&**vadd, crossArgs, addSizes, 4},
      {&**vadd, addArgs, addSizes, 4},
  };

  for (const Launch &launch : chain) {
    ErrorOrSuccess result = ctx.enqueueFunctionExecDirect(
        *launch.function, /*gridX=*/1, /*gridY=*/1, /*gridZ=*/1, /*blockX=*/1,
        /*blockY=*/1, /*blockZ=*/1, /*sharedMemBytes=*/0, /*attrs=*/nullptr,
        /*numAttrs=*/0, launch.args, launch.argCount, launch.sizes);
    if (result.isError())
      return childFail("dispatch", result.getError());
  }
  // What a graph does before it reads an output, and what a held batch has to
  // be sent by.
  ErrorOrSuccess sync = ctx.synchronize();
  if (sync.isError())
    return childFail("synchronize", sync.getError());
  return 0;
}

int runChild(const char *resultPath, bool perturb) {
  ErrorOr<DeviceRef> device = QuRTDevice::create(0);
  if (device.isError())
    return childFail("device", device.getError());
  ErrorOr<DeviceContextRef> context = (*device)->createContext();
  if (context.isError())
    return childFail("context", context.getError());
  DeviceContext &ctx = **context;

  ErrorOr<DeviceBufferRef> arena = ctx.enqueueCreateBuffer(kArenaBytes);
  if (arena.isError())
    return childFail("arena", arena.getError());
  ErrorOr<DeviceBufferRef> side = ctx.enqueueCreateBuffer(kSideBytes);
  if (side.isError())
    return childFail("side", side.getError());

  auto *arenaBytes = static_cast<uint8_t *>((*arena)->ptr());
  auto *sideBytes = static_cast<uint8_t *>((*side)->ptr());
  int status = runChain(ctx, arenaBytes, sideBytes, perturb);
  if (status != 0)
    return status;

  std::ofstream out(resultPath, std::ios::binary);
  out.write(reinterpret_cast<const char *>(arenaBytes), kArenaBytes);
  out.write(reinterpret_cast<const char *>(sideBytes), kSideBytes);
  return out.good() ? 0 : 1;
}

/// With both transport flags set, the first dispatch must be refused with a
/// message naming both — a device leg is supposed to move exactly one
/// variable, and how a held batch composes with a live queue is a design
/// question the dispatch path may not quietly answer with a precedence.
/// Exits 0 only on the named refusal.
int runChildBothFlags() {
  ErrorOr<DeviceRef> device = QuRTDevice::create(0);
  if (device.isError())
    return childFail("device", device.getError());
  ErrorOr<DeviceContextRef> context = (*device)->createContext();
  if (context.isError())
    return childFail("context", context.getError());
  DeviceContext &ctx = **context;

  ErrorOr<DeviceBufferRef> arena = ctx.enqueueCreateBuffer(kArenaBytes);
  if (arena.isError())
    return childFail("arena", arena.getError());
  ErrorOr<DeviceFunctionRef> vadd = ctx.loadFunction("", "vadd", "");
  if (vadd.isError())
    return childFail("loadFunction vadd", vadd.getError());

  auto *bytes = static_cast<uint8_t *>((*arena)->ptr());
  void *a = bytes;
  void *b = bytes + 4096;
  void *c = bytes + 8192;
  uint32_t count = 256;
  void *args[4] = {&a, &b, &c, &count};
  size_t sizes[4] = {sizeof(void *), sizeof(void *), sizeof(void *),
                     sizeof(uint32_t)};

  ErrorOrSuccess result = ctx.enqueueFunctionExecDirect(
      **vadd, /*gridX=*/1, /*gridY=*/1, /*gridZ=*/1, /*blockX=*/1,
      /*blockY=*/1, /*blockZ=*/1, /*sharedMemBytes=*/0, /*attrs=*/nullptr,
      /*numAttrs=*/0, args, /*argCount=*/4, sizes);
  if (!result.isError())
    return childFail("both-flags", "a dispatch with MODULAR_NPU_BATCH=1 and "
                                   "MODULAR_NPU_QUEUE=1 was accepted");
  std::string error(result.getError());
  if (error.find("MODULAR_NPU_BATCH") == std::string::npos ||
      error.find("MODULAR_NPU_QUEUE") == std::string::npos)
    return childFail(
        "both-flags",
        ("the refusal does not name both flags: " + error).c_str());
  return 0;
}

//===----------------------------------------------------------------------===//
// Orchestrator
//===----------------------------------------------------------------------===//

struct Leg {
  bool batched = false;
  bool perturb = false;
  bool transportPredatesBatching = false;
  bool bothFlags = false;
  std::string resultPath;
  std::string countPath;
};

int spawnChild(const std::string &self, const std::string &fakeLib,
               const Leg &leg) {
  pid_t pid = fork();
  if (pid < 0)
    return -1;
  if (pid == 0) {
    setenv("MODULAR_NPU_TRANSPORT_LIB", fakeLib.c_str(), 1);
    setenv("MODULAR_NPU_BATCH", leg.batched ? "1" : "0", 1);
    // The session vote is on by default and would land in the invocation
    // ledger; this test moves one variable -- the transport -- and the vote
    // has its own (npu_power_vote_test).
    setenv("MODULAR_NPU_POWER", "0", 1);
    setenv("NPU_FAKE_TRANSPORT_COUNT", leg.countPath.c_str(), 1);
    if (leg.transportPredatesBatching)
      setenv("NPU_FAKE_TRANSPORT_NO_BATCH", "1", 1);
    if (leg.bothFlags)
      setenv("MODULAR_NPU_QUEUE", "1", 1);
    const char *mode = leg.bothFlags ? "--child-both-flags"
                       : leg.perturb ? "--child-perturbed"
                                     : "--child";
    execl(self.c_str(), self.c_str(), mode, leg.resultPath.c_str(),
          static_cast<char *>(nullptr));
    _exit(127);
  }
  int status = 0;
  if (waitpid(pid, &status, 0) != pid)
    return -1;
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

std::vector<uint8_t> readBytes(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)),
                              std::istreambuf_iterator<char>());
}

/// Invocations the leg made, and the largest number of launches any one of them
/// carried. One invocation per launch is the per-call path; anything else is a
/// batch that formed.
void readCounts(const std::string &path, size_t *invocations, uint32_t *widest,
                uint32_t *total) {
  *invocations = 0;
  *widest = 0;
  *total = 0;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty())
      continue;
    auto launches = static_cast<uint32_t>(strtoul(line.c_str(), nullptr, 10));
    ++*invocations;
    *total += launches;
    if (launches > *widest)
      *widest = launches;
  }
}

int runOrchestrator(const char *self) {
  const char *fake = getenv("NPU_FAKE_TRANSPORT");
  const char *tmp = getenv("TEST_TMPDIR");
  if (fake == nullptr || tmp == nullptr) {
    printf("FAIL: NPU_FAKE_TRANSPORT and TEST_TMPDIR must be set\n");
    return 1;
  }
  char resolved[4096];
  std::string fakeLib =
      realpath(fake, resolved) != nullptr ? resolved : std::string(fake);
  std::string base(tmp);

  Leg perCall{false,
              false,
              false,
              false,
              base + "/perCall.bin",
              base + "/perCall.count"};
  Leg batched{true,
              false,
              false,
              false,
              base + "/batched.bin",
              base + "/batched.count"};
  Leg perturbed{true,
                true,
                false,
                false,
                base + "/perturbed.bin",
                base + "/perturbed.count"};
  Leg older{
      true, false, true, false, base + "/older.bin", base + "/older.count"};

  expect(spawnChild(self, fakeLib, perCall) == 0, "per-call leg exited 0");
  expect(spawnChild(self, fakeLib, batched) == 0, "batched leg exited 0");
  expect(spawnChild(self, fakeLib, perturbed) == 0, "perturbed leg exited 0");
  if (failures != 0)
    return 1;

  size_t invocations = 0;
  uint32_t widest = 0;
  uint32_t launches = 0;
  readCounts(perCall.countPath, &invocations, &widest, &launches);
  expect(invocations == 5 && widest == 1,
         "the per-call leg sent one launch per invocation");

  readCounts(batched.countPath, &invocations, &widest, &launches);
  expect(launches == 5, "the batched leg ran every launch of the chain");
  expect(widest > 1, "the batched leg never formed a batch, so the comparison "
                     "below is per-call against per-call");
  expect(invocations < 5, "the batched leg saved no invocation");
  printf("batched leg: %zu invocations for %u launches, widest %u\n",
         invocations, launches, widest);

  std::vector<uint8_t> perCallBytes = readBytes(perCall.resultPath);
  std::vector<uint8_t> batchedBytes = readBytes(batched.resultPath);
  expect(perCallBytes.size() == kArenaBytes + kSideBytes,
         "the per-call leg wrote a full result");
  expect(perCallBytes == batchedBytes,
         "batched and per-call dispatch of the same chain differ");

  // The control: the comparison above has to be capable of failing, and one
  // changed scalar in one launch of five is about the smallest divergence a
  // real mis-batching would produce.
  std::vector<uint8_t> perturbedBytes = readBytes(perturbed.resultPath);
  expect(perturbedBytes.size() == perCallBytes.size(),
         "the perturbed leg wrote a full result");
  expect(perturbedBytes != perCallBytes,
         "a deliberately altered launch produced identical bytes: the "
         "comparison is not reading what the dispatches wrote");

  // The other control: an older signed library must refuse a batch rather than
  // decode its block as one kernel's arguments.
  expect(spawnChild(self, fakeLib, older) != 0,
         "a transport predating batching accepted a batch instead of refusing "
         "it");

  // Two transports requested at once must be refused by a message naming both
  // flags; the child exits 0 only on that named refusal.
  Leg bothFlags{
      true, false, false, true, base + "/both.bin", base + "/both.count"};
  expect(spawnChild(self, fakeLib, bothFlags) == 0,
         "MODULAR_NPU_BATCH=1 with MODULAR_NPU_QUEUE=1 was not refused by "
         "name");

  return failures == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char **argv) {
  if (argc == 3 && strcmp(argv[1], "--child") == 0)
    return runChild(argv[2], /*perturb=*/false);
  if (argc == 3 && strcmp(argv[1], "--child-perturbed") == 0)
    return runChild(argv[2], /*perturb=*/true);
  if (argc == 3 && strcmp(argv[1], "--child-both-flags") == 0)
    return runChildBothFlags();
  int status = runOrchestrator(argv[0]);
  printf(status == 0 ? "PASS\n" : "FAILED\n");
  return status;
}
