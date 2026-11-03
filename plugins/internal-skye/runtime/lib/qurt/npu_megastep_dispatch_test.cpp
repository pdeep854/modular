//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// The step table, judged the way the batch was judged: a differential. The
// same chain of launches runs through the per-call path and through the
// megastep path, into arenas that start identical, and every byte has to
// match — including the bytes of ten constant-classified "weight"
// allocations, each named by session-table reference in the tabled leg and
// each written at a different offset, so a reference resolved to the wrong
// table entry lands at an offset that cannot also be right.
//
// What the megastep claims beyond the batch, and what is pinned here:
//
//   * the chain names TWELVE distinct allocations, past the eight slots that
//     bound a batch, and still travels as ONE run invocation — the tabled
//     leg's count file must show every launch behind a single crossing, with
//     the registrations as separate zero-launch invocations before it;
//   * the activation self-report names the flag and the table's width;
//   * a transport predating the manifest row refuses the selector by name;
//   * MEGASTEP x BATCH and MEGASTEP x QUEUE are refused at the first
//     dispatch by a message naming both flags and the exactly-one-of-three
//     rule (BATCH x QUEUE is pinned by npu_batched_dispatch_test).
//
// What this cannot test is the coherence half of the table's argument — that
// registration's one presentation is the only maintenance an init-once
// allocation needs. A host has no NSP caches to leave stale; that half is
// settled on silicon by the same differential, as the batch's was.
//
//===----------------------------------------------------------------------===//

#include "QuRTDeviceContext.h"
#include "npu_constant_ranges.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
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
constexpr size_t kSideBytes = 1 << 12;

/// Ten, so the chain's allocation union (with the arena and the side buffer)
/// is wider than the eight slots that bound a batch: the point the table
/// exists to pass.
constexpr size_t kWeights = 10;
constexpr size_t kWeightBytes = 4096;

/// The host memory the weights are copied from, declared constant to the
/// classifier the way a loader declares a registry blob. Static so the
/// declaration outlives every copy.
uint8_t weightBlob[kWeights * kWeightBytes];

int childFail(const char *stage, const char *error) {
  printf("child %s: %s\n", stage, error);
  return 1;
}

/// The chain both legs run: ten launches each naming its own weight at its
/// own offset, then two naming no weight at all — so the table carries mixed
/// references, launches that share the maintained slots, and one launch whose
/// operands are slots only.
int runChain(DeviceContext &ctx, uint8_t *arena, uint8_t *side,
             uint8_t **weights, bool perturb) {
  ErrorOr<DeviceFunctionRef> vadd = ctx.loadFunction("", "vadd", "");
  if (vadd.isError())
    return childFail("loadFunction vadd", vadd.getError());

  uint32_t count = 64;
  for (size_t k = 0; k < kWeights; ++k) {
    void *a = arena + 512 * k;
    // Each weight read at a different interior offset: a session-table
    // reference resolved to the wrong entry then writes at an offset that is
    // wrong for that entry, which the byte comparison sees. The perturbed leg
    // moves one of these by 64 — the smallest divergence a mis-resolved or
    // mis-folded reference produces.
    void *b = weights[k] + 128 * k + (perturb && k == 3 ? 64 : 0);
    void *c = side + 64 * k;
    void *args[4] = {&a, &b, &c, &count};
    size_t sizes[4] = {sizeof(void *), sizeof(void *), sizeof(void *),
                       sizeof(uint32_t)};
    ErrorOrSuccess result = ctx.enqueueFunctionExecDirect(
        **vadd, /*gridX=*/1, /*gridY=*/1, /*gridZ=*/1, /*blockX=*/1,
        /*blockY=*/1, /*blockZ=*/1, /*sharedMemBytes=*/0, /*attrs=*/nullptr,
        /*numAttrs=*/0, args, /*argCount=*/4, sizes);
    if (result.isError())
      return childFail("dispatch", result.getError());
  }

  void *pure[2][3] = {{arena, arena + 4096, arena + 8192},
                      {side, arena + 2048, side + 2048}};
  for (auto &operands : pure) {
    void *args[4] = {&operands[0], &operands[1], &operands[2], &count};
    size_t sizes[4] = {sizeof(void *), sizeof(void *), sizeof(void *),
                       sizeof(uint32_t)};
    ErrorOrSuccess result = ctx.enqueueFunctionExecDirect(
        **vadd, /*gridX=*/1, /*gridY=*/1, /*gridZ=*/1, /*blockX=*/1,
        /*blockY=*/1, /*blockZ=*/1, /*sharedMemBytes=*/0, /*attrs=*/nullptr,
        /*numAttrs=*/0, args, /*argCount=*/4, sizes);
    if (result.isError())
      return childFail("dispatch", result.getError());
  }

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

  // What a loader does for a registry blob: declare the host source constant,
  // then fill each device allocation whole from it — the provenance that
  // classifies an allocation init-once and makes it eligible for a
  // session-table reference in the tabled leg.
  MODULAR_NPUDeclareConstantHostRange(weightBlob, sizeof(weightBlob));
  std::vector<DeviceBufferRef> weightBuffers;
  uint8_t *weights[kWeights];
  for (size_t k = 0; k < kWeights; ++k) {
    ErrorOr<DeviceBufferRef> weight = ctx.enqueueCreateBuffer(kWeightBytes);
    if (weight.isError())
      return childFail("weight", weight.getError());
    ErrorOrSuccess copied = ctx.enqueueCopyToDevice(
        (*weight)->ptr(), weightBlob + k * kWeightBytes, kWeightBytes);
    if (copied.isError())
      return childFail("copy weight", copied.getError());
    weights[k] = static_cast<uint8_t *>((*weight)->ptr());
    weightBuffers.push_back(std::move(*weight));
  }

  auto *arenaBytes = static_cast<uint8_t *>((*arena)->ptr());
  auto *sideBytes = static_cast<uint8_t *>((*side)->ptr());
  int status = runChain(ctx, arenaBytes, sideBytes, weights, perturb);
  if (status != 0)
    return status;

  std::ofstream out(resultPath, std::ios::binary);
  out.write(reinterpret_cast<const char *>(arenaBytes), kArenaBytes);
  out.write(reinterpret_cast<const char *>(sideBytes), kSideBytes);
  for (size_t k = 0; k < kWeights; ++k)
    out.write(reinterpret_cast<const char *>(weights[k]), kWeightBytes);
  return out.good() ? 0 : 1;
}

/// With the step table and a second transport both requested, the first
/// dispatch must be refused with a message naming both flags and the
/// exactly-one-of-three rule. Exits 0 only on that named refusal.
int runChildPairFlags(const char *partner) {
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
  uint32_t count = 64;
  void *args[4] = {&a, &b, &c, &count};
  size_t sizes[4] = {sizeof(void *), sizeof(void *), sizeof(void *),
                     sizeof(uint32_t)};

  ErrorOrSuccess result = ctx.enqueueFunctionExecDirect(
      **vadd, /*gridX=*/1, /*gridY=*/1, /*gridZ=*/1, /*blockX=*/1,
      /*blockY=*/1, /*blockZ=*/1, /*sharedMemBytes=*/0, /*attrs=*/nullptr,
      /*numAttrs=*/0, args, /*argCount=*/4, sizes);
  if (!result.isError())
    return childFail("pair-flags", "a dispatch with two transports requested "
                                   "was accepted");
  std::string error(result.getError());
  if (error.find("MODULAR_NPU_MEGASTEP=1") == std::string::npos ||
      error.find(std::string(partner) + "=1") == std::string::npos ||
      error.find("set exactly one of MODULAR_NPU_MEGASTEP, MODULAR_NPU_BATCH, "
                 "MODULAR_NPU_QUEUE") == std::string::npos)
    return childFail(
        "pair-flags",
        ("the refusal does not name both flags and the exactly-one rule: " +
         error)
            .c_str());
  return 0;
}

//===----------------------------------------------------------------------===//
// Orchestrator
//===----------------------------------------------------------------------===//

struct Leg {
  bool megastep = false;
  bool perturb = false;
  bool transportPredatesMegastep = false;
  const char *pairFlag = nullptr;
  std::string resultPath;
  std::string countPath;
  std::string stderrPath;
};

int spawnChild(const std::string &self, const std::string &fakeLib,
               const Leg &leg) {
  pid_t pid = fork();
  if (pid < 0)
    return -1;
  if (pid == 0) {
    setenv("MODULAR_NPU_TRANSPORT_LIB", fakeLib.c_str(), 1);
    setenv("MODULAR_NPU_MEGASTEP", leg.megastep ? "1" : "0", 1);
    // The session vote is on by default and would land in the invocation
    // ledger; this test moves one variable -- the transport -- and the vote
    // has its own (npu_power_vote_test).
    setenv("MODULAR_NPU_POWER", "0", 1);
    setenv("NPU_FAKE_TRANSPORT_COUNT", leg.countPath.c_str(), 1);
    if (leg.transportPredatesMegastep)
      setenv("NPU_FAKE_TRANSPORT_NO_MEGASTEP", "1", 1);
    if (leg.pairFlag != nullptr)
      setenv(leg.pairFlag, "1", 1);
    int fd = open(leg.stderrPath.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (fd >= 0) {
      dup2(fd, 2);
      close(fd);
    }
    const char *mode = leg.pairFlag != nullptr ? "--child-pair"
                       : leg.perturb          ? "--child-perturbed"
                                              : "--child";
    execl(self.c_str(), self.c_str(), mode, leg.resultPath.c_str(),
          leg.pairFlag != nullptr ? leg.pairFlag : "",
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

/// Invocations the leg made, the launches each carried, and the widest one. A
/// registration is an invocation carrying zero launches, so the tabled leg's
/// ledger separates the session's registrations from the step's one run.
void readCounts(const std::string &path, size_t *invocations, uint32_t *widest,
                uint32_t *total, size_t *empty) {
  *invocations = 0;
  *widest = 0;
  *total = 0;
  *empty = 0;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty())
      continue;
    auto launches = static_cast<uint32_t>(strtoul(line.c_str(), nullptr, 10));
    ++*invocations;
    *total += launches;
    if (launches == 0)
      ++*empty;
    if (launches > *widest)
      *widest = launches;
  }
}

bool fileHasLine(const std::string &path, const std::string &needle) {
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line))
    if (line.find(needle) != std::string::npos)
      return true;
  return false;
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

  auto leg = [&](const char *name) {
    Leg l;
    l.resultPath = base + "/" + name + ".bin";
    l.countPath = base + "/" + name + ".count";
    l.stderrPath = base + "/" + name + ".stderr";
    return l;
  };

  Leg perCall = leg("perCall");
  Leg tabled = leg("tabled");
  tabled.megastep = true;
  Leg perturbed = leg("perturbed");
  perturbed.megastep = true;
  perturbed.perturb = true;
  Leg older = leg("older");
  older.megastep = true;
  older.transportPredatesMegastep = true;

  expect(spawnChild(self, fakeLib, perCall) == 0, "per-call leg exited 0");
  expect(spawnChild(self, fakeLib, tabled) == 0, "megastep leg exited 0");
  expect(spawnChild(self, fakeLib, perturbed) == 0, "perturbed leg exited 0");
  if (failures != 0)
    return 1;

  constexpr uint32_t kChain = kWeights + 2;
  size_t invocations = 0, empty = 0;
  uint32_t widest = 0, launches = 0;
  readCounts(perCall.countPath, &invocations, &widest, &launches, &empty);
  expect(invocations == kChain && widest == 1,
         "the per-call leg sent one launch per invocation");

  readCounts(tabled.countPath, &invocations, &widest, &launches, &empty);
  expect(launches == kChain, "the tabled leg ran every launch of the chain");
  expect(widest == kChain,
         "the whole chain travelled as ONE run invocation: twelve distinct "
         "allocations is past what eight slots can union, so anything less "
         "means the table split");
  expect(empty == 2, "ten weights registered as two zero-launch invocations "
                     "(eight per call, then two)");
  expect(invocations == 3,
         "the tabled leg made exactly two registrations and one run");
  printf("tabled leg: %zu invocations for %u launches, widest %u, "
         "%zu registrations\n",
         invocations, launches, widest, empty);

  expect(fileHasLine(tabled.stderrPath,
                     "npu megastep: active (MODULAR_NPU_MEGASTEP=1, 12 "
                     "launches tabled)"),
         "the activation self-report names the flag and the table's width");
  expect(fileHasLine(tabled.stderrPath,
                     "npu megastep: registered 10 constant allocations"),
         "the registration self-report names the session table's fill");

  std::vector<uint8_t> perCallBytes = readBytes(perCall.resultPath);
  std::vector<uint8_t> tabledBytes = readBytes(tabled.resultPath);
  expect(perCallBytes.size() ==
             kArenaBytes + kSideBytes + kWeights * kWeightBytes,
         "the per-call leg wrote a full result");
  expect(perCallBytes == tabledBytes,
         "megastep and per-call dispatch of the same chain differ");

  // The control: one operand of one launch of twelve, moved 64 bytes within
  // the weight it already named, MUST mismatch — otherwise the comparison is
  // not reading the bytes the table-referenced dispatches wrote.
  std::vector<uint8_t> perturbedBytes = readBytes(perturbed.resultPath);
  expect(perturbedBytes.size() == perCallBytes.size(),
         "the perturbed leg wrote a full result");
  expect(perturbedBytes != perCallBytes,
         "a deliberately altered launch produced identical bytes: the "
         "comparison is not reading what the dispatches wrote");

  // A signed library predating the manifest row must refuse the selector
  // rather than decode the block as one kernel's arguments.
  expect(spawnChild(self, fakeLib, older) != 0,
         "a transport predating the megastep row accepted its selector "
         "instead of refusing it");

  // The transport-exclusion matrix's two new pairs; BATCH x QUEUE is pinned
  // where it landed, in npu_batched_dispatch_test.
  Leg pairBatch = leg("pairBatch");
  pairBatch.megastep = true;
  pairBatch.pairFlag = "MODULAR_NPU_BATCH";
  expect(spawnChild(self, fakeLib, pairBatch) == 0,
         "MODULAR_NPU_MEGASTEP=1 with MODULAR_NPU_BATCH=1 was not refused by "
         "name");
  Leg pairQueue = leg("pairQueue");
  pairQueue.megastep = true;
  pairQueue.pairFlag = "MODULAR_NPU_QUEUE";
  expect(spawnChild(self, fakeLib, pairQueue) == 0,
         "MODULAR_NPU_MEGASTEP=1 with MODULAR_NPU_QUEUE=1 was not refused by "
         "name");

  return failures == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char **argv) {
  if (argc >= 3 && strcmp(argv[1], "--child") == 0)
    return runChild(argv[2], /*perturb=*/false);
  if (argc >= 3 && strcmp(argv[1], "--child-perturbed") == 0)
    return runChild(argv[2], /*perturb=*/true);
  if (argc >= 4 && strcmp(argv[1], "--child-pair") == 0)
    return runChildPairFlags(argv[3]);
  int status = runOrchestrator(argv[0]);
  printf(status == 0 ? "PASS\n" : "FAILED\n");
  return status;
}
