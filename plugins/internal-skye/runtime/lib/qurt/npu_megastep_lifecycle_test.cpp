//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// The megastep session table over TIME, which the one-step differential
// cannot see: registration is once per session, not once per step, and every
// event that could make a table entry lie -- a rewrite, a free, a
// reallocation -- has to force a re-registration rather than a stale read.
// The count ledger is the observable: a registration is a zero-launch
// invocation, so the exact line sequence says when the session presented an
// allocation and when it trusted the table.
//
// Three legs:
//
//   * LIFECYCLE -- one weight across four steps: registered once, trusted on
//     the identical second step (no registration line), re-registered after
//     a whole rewrite (the forget hook dropped the entry), re-registered
//     after a free and a fresh allocation (rpcmem hands addresses back, so
//     the entry must not survive the base).
//   * OVERFLOW -- ten mutable allocations against an eight-slot budget: the
//     step SPLITS into two run invocations rather than widening the claim,
//     and the split leg's bytes still match the per-call leg's exactly.
//   * MARKS -- the serve exe brackets each execute in step_<tag> perf marks
//     (PersistentStageSession R5); the megastep's timeline records must land
//     INSIDE the step's brackets, or the serve A/B loses per-step
//     attribution the moment the flag turns on.
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
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

extern "C" void npu_perf_mark(const char *phase);

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

int childFail(const char *stage, const std::string &error) {
  printf("child %s: %s\n", stage, error.c_str());
  return 1;
}

constexpr size_t kWeightBytes = 4096;
constexpr size_t kArenaBytes = 1 << 15;
uint8_t weightBlob[kWeightBytes];

int dispatch(DeviceContext &ctx, DeviceFunction &kernel, void *a, void *b,
             void *c) {
  uint32_t count = 64;
  void *args[4] = {&a, &b, &c, &count};
  size_t sizes[4] = {sizeof(void *), sizeof(void *), sizeof(void *),
                     sizeof(uint32_t)};
  ErrorOrSuccess result = ctx.enqueueFunctionExecDirect(
      kernel, /*gridX=*/1, /*gridY=*/1, /*gridZ=*/1, /*blockX=*/1,
      /*blockY=*/1, /*blockZ=*/1, /*sharedMemBytes=*/0, /*attrs=*/nullptr,
      /*numAttrs=*/0, args, /*argCount=*/4, sizes);
  if (result.isError())
    return childFail("dispatch", result.getError());
  return 0;
}

/// One step: three launches naming the weight at distinct offsets, then the
/// synchronize a graph boundary does.
int step(DeviceContext &ctx, DeviceFunction &kernel, uint8_t *arena,
         uint8_t *weight) {
  for (int i = 0; i < 3; ++i)
    if (dispatch(ctx, kernel, arena + 1024 * i, weight + 256 * i,
                 arena + 8192 + 64 * i) != 0)
      return 1;
  ErrorOrSuccess sync = ctx.synchronize();
  if (sync.isError())
    return childFail("synchronize", sync.getError());
  return 0;
}

int runChildLifecycle() {
  ErrorOr<DeviceRef> device = QuRTDevice::create(0);
  if (device.isError())
    return childFail("device", device.getError());
  ErrorOr<DeviceContextRef> context = (*device)->createContext();
  if (context.isError())
    return childFail("context", context.getError());
  DeviceContext &ctx = **context;
  ErrorOr<DeviceFunctionRef> vadd = ctx.loadFunction("", "vadd", "");
  if (vadd.isError())
    return childFail("loadFunction", vadd.getError());
  ErrorOr<DeviceBufferRef> arena = ctx.enqueueCreateBuffer(kArenaBytes);
  if (arena.isError())
    return childFail("arena", arena.getError());
  auto *arenaBytes = static_cast<uint8_t *>((*arena)->ptr());

  MODULAR_NPUDeclareConstantHostRange(weightBlob, sizeof(weightBlob));

  {
    ErrorOr<DeviceBufferRef> weight = ctx.enqueueCreateBuffer(kWeightBytes);
    if (weight.isError())
      return childFail("weight", weight.getError());
    auto *weightBytes = static_cast<uint8_t *>((*weight)->ptr());
    ErrorOrSuccess filled =
        ctx.enqueueCopyToDevice(weightBytes, weightBlob, kWeightBytes);
    if (filled.isError())
      return childFail("fill", filled.getError());

    // Step 1 registers; the identical step 2 must NOT -- the table is the
    // point, and a re-presentation per step would be the writeback the
    // session table exists to retire.
    if (step(ctx, **vadd, arenaBytes, weightBytes) != 0)
      return 1;
    if (step(ctx, **vadd, arenaBytes, weightBytes) != 0)
      return 1;

    // A whole rewrite keeps the classification but must drop the entry: the
    // registration is the presentation that flushes the write, so trusting
    // the old entry would read bytes the driver never cleaned.
    ErrorOrSuccess rewrite =
        ctx.enqueueCopyToDevice(weightBytes, weightBlob, kWeightBytes);
    if (rewrite.isError())
      return childFail("rewrite", rewrite.getError());
    if (step(ctx, **vadd, arenaBytes, weightBytes) != 0)
      return 1;
  }

  // Freed above by the scope; a fresh allocation -- rpcmem may hand the same
  // base back -- must register as itself, not inherit the dead entry.
  ErrorOr<DeviceBufferRef> second = ctx.enqueueCreateBuffer(kWeightBytes);
  if (second.isError())
    return childFail("second", second.getError());
  auto *secondBytes = static_cast<uint8_t *>((*second)->ptr());
  ErrorOrSuccess filled =
      ctx.enqueueCopyToDevice(secondBytes, weightBlob, kWeightBytes);
  if (filled.isError())
    return childFail("fill second", filled.getError());
  return step(ctx, **vadd, arenaBytes, secondBytes);
}

int runChildOverflow(const char *resultPath, bool megastep) {
  ErrorOr<DeviceRef> device = QuRTDevice::create(0);
  if (device.isError())
    return childFail("device", device.getError());
  ErrorOr<DeviceContextRef> context = (*device)->createContext();
  if (context.isError())
    return childFail("context", context.getError());
  DeviceContext &ctx = **context;
  ErrorOr<DeviceFunctionRef> vadd = ctx.loadFunction("", "vadd", "");
  if (vadd.isError())
    return childFail("loadFunction", vadd.getError());

  // Ten mutable allocations and a shared one: nothing declared constant, so
  // every operand needs a maintained slot and the union passes eight inside
  // one step.
  constexpr size_t kMutable = 10;
  std::vector<DeviceBufferRef> buffers;
  uint8_t *bases[kMutable];
  for (size_t i = 0; i < kMutable; ++i) {
    ErrorOr<DeviceBufferRef> buffer = ctx.enqueueCreateBuffer(kWeightBytes);
    if (buffer.isError())
      return childFail("alloc", buffer.getError());
    bases[i] = static_cast<uint8_t *>((*buffer)->ptr());
    buffers.push_back(std::move(*buffer));
  }
  ErrorOr<DeviceBufferRef> shared = ctx.enqueueCreateBuffer(kArenaBytes);
  if (shared.isError())
    return childFail("shared", shared.getError());
  auto *sharedBytes = static_cast<uint8_t *>((*shared)->ptr());

  (void)megastep;
  for (size_t i = 0; i < kMutable; ++i)
    if (dispatch(ctx, **vadd, bases[i], bases[i] + 512,
                 sharedBytes + 128 * i) != 0)
      return 1;
  ErrorOrSuccess sync = ctx.synchronize();
  if (sync.isError())
    return childFail("synchronize", sync.getError());

  std::ofstream out(resultPath, std::ios::binary);
  for (size_t i = 0; i < kMutable; ++i)
    out.write(reinterpret_cast<const char *>(bases[i]), kWeightBytes);
  out.write(reinterpret_cast<const char *>(sharedBytes), kArenaBytes);
  return out.good() ? 0 : 1;
}

/// The graph path's exact per-launch bracketing (registrations.mojo): a fresh
/// witness allocated before every launch and freed right after it. Order 087
/// showed this shreds an eager-flushing table to width ~1; with allocation
/// no longer flushing and frees deferred, the whole chain must travel as one
/// run.
int runChildCadence(const char *resultPath) {
  ErrorOr<DeviceRef> device = QuRTDevice::create(0);
  if (device.isError())
    return childFail("device", device.getError());
  ErrorOr<DeviceContextRef> context = (*device)->createContext();
  if (context.isError())
    return childFail("context", context.getError());
  DeviceContext &ctx = **context;
  ErrorOr<DeviceFunctionRef> vadd = ctx.loadFunction("", "vadd", "");
  if (vadd.isError())
    return childFail("loadFunction", vadd.getError());
  ErrorOr<DeviceBufferRef> arena = ctx.enqueueCreateBuffer(kArenaBytes);
  if (arena.isError())
    return childFail("arena", arena.getError());
  auto *arenaBytes = static_cast<uint8_t *>((*arena)->ptr());

  MODULAR_NPUDeclareConstantHostRange(weightBlob, sizeof(weightBlob));
  ErrorOr<DeviceBufferRef> weight = ctx.enqueueCreateBuffer(kWeightBytes);
  if (weight.isError())
    return childFail("weight", weight.getError());
  auto *weightBytes = static_cast<uint8_t *>((*weight)->ptr());
  ErrorOrSuccess filled =
      ctx.enqueueCopyToDevice(weightBytes, weightBlob, kWeightBytes);
  if (filled.isError())
    return childFail("fill", filled.getError());

  for (int i = 0; i < 6; ++i) {
    ErrorOr<DeviceBufferRef> witness = ctx.enqueueCreateBuffer(128);
    if (witness.isError())
      return childFail("witness", witness.getError());
    if (dispatch(ctx, **vadd, arenaBytes + 1024 * i,
                 weightBytes + 256 * (i % 3),
                 static_cast<uint8_t *>((*witness)->ptr())) != 0)
      return 1;
    // The witness ref drops here, freeing it while the launch is still held
    // -- the exact lifetime the graph path gives every launch's witness.
  }
  ErrorOrSuccess sync = ctx.synchronize();
  if (sync.isError())
    return childFail("synchronize", sync.getError());

  std::ofstream out(resultPath, std::ios::binary);
  out.write(reinterpret_cast<const char *>(arenaBytes), kArenaBytes);
  out.write(reinterpret_cast<const char *>(weightBytes), kWeightBytes);
  return out.good() ? 0 : 1;
}

/// One launch, then the graph boundary: the table cannot form, and the step
/// must go out on the per-call wire rather than as a one-entry walk.
int runChildFallback(const char *resultPath) {
  ErrorOr<DeviceRef> device = QuRTDevice::create(0);
  if (device.isError())
    return childFail("device", device.getError());
  ErrorOr<DeviceContextRef> context = (*device)->createContext();
  if (context.isError())
    return childFail("context", context.getError());
  DeviceContext &ctx = **context;
  ErrorOr<DeviceFunctionRef> vadd = ctx.loadFunction("", "vadd", "");
  if (vadd.isError())
    return childFail("loadFunction", vadd.getError());
  ErrorOr<DeviceBufferRef> arena = ctx.enqueueCreateBuffer(kArenaBytes);
  if (arena.isError())
    return childFail("arena", arena.getError());
  auto *arenaBytes = static_cast<uint8_t *>((*arena)->ptr());

  MODULAR_NPUDeclareConstantHostRange(weightBlob, sizeof(weightBlob));
  ErrorOr<DeviceBufferRef> weight = ctx.enqueueCreateBuffer(kWeightBytes);
  if (weight.isError())
    return childFail("weight", weight.getError());
  auto *weightBytes = static_cast<uint8_t *>((*weight)->ptr());
  ErrorOrSuccess filled =
      ctx.enqueueCopyToDevice(weightBytes, weightBlob, kWeightBytes);
  if (filled.isError())
    return childFail("fill", filled.getError());

  if (dispatch(ctx, **vadd, arenaBytes, weightBytes + 256, arenaBytes + 8192) !=
      0)
    return 1;
  ErrorOrSuccess sync = ctx.synchronize();
  if (sync.isError())
    return childFail("synchronize", sync.getError());

  std::ofstream out(resultPath, std::ios::binary);
  out.write(reinterpret_cast<const char *>(arenaBytes), kArenaBytes);
  out.write(reinterpret_cast<const char *>(weightBytes), kWeightBytes);
  return out.good() ? 0 : 1;
}

/// A device that refuses the run (NPU_FAKE_TRANSPORT_REFUSE_MEGASTEP_RUN):
/// the flush error must surface at the synchronize AND poison the context so
/// the next launch refuses too -- the path a destructor-swallowed error takes
/// to a loud stop instead of an all-zero render.
int runChildPoison() {
  ErrorOr<DeviceRef> device = QuRTDevice::create(0);
  if (device.isError())
    return childFail("device", device.getError());
  ErrorOr<DeviceContextRef> context = (*device)->createContext();
  if (context.isError())
    return childFail("context", context.getError());
  DeviceContext &ctx = **context;
  ErrorOr<DeviceFunctionRef> vadd = ctx.loadFunction("", "vadd", "");
  if (vadd.isError())
    return childFail("loadFunction", vadd.getError());
  ErrorOr<DeviceBufferRef> arena = ctx.enqueueCreateBuffer(kArenaBytes);
  if (arena.isError())
    return childFail("arena", arena.getError());
  auto *arenaBytes = static_cast<uint8_t *>((*arena)->ptr());

  // Two launches, so the flush is a real run rather than the fallback.
  if (dispatch(ctx, **vadd, arenaBytes, arenaBytes + 4096,
               arenaBytes + 8192) != 0)
    return 1;
  if (dispatch(ctx, **vadd, arenaBytes + 512, arenaBytes + 4608,
               arenaBytes + 8704) != 0)
    return 1;
  ErrorOrSuccess sync = ctx.synchronize();
  if (!sync.isError())
    return childFail("poison", "a refused megastep run reported success");
  std::string error(sync.getError());
  if (error.find("megastep of 2 launches failed") == std::string::npos)
    return childFail("poison",
                     ("the flush failure does not name itself: " + error)
                         .c_str());

  // The next launch must refuse on the recorded poison, not proceed: this is
  // what stops a render whose flush error was swallowed by a destructor.
  ErrorOr<DeviceBufferRef> more = ctx.enqueueCreateBuffer(kWeightBytes);
  if (more.isError())
    return childFail("alloc after poison", more.getError());
  uint8_t *moreBytes = static_cast<uint8_t *>((*more)->ptr());
  void *a = moreBytes;
  void *b = moreBytes + 512;
  void *c = moreBytes + 1024;
  uint32_t count = 64;
  void *args[4] = {&a, &b, &c, &count};
  size_t sizes[4] = {sizeof(void *), sizeof(void *), sizeof(void *),
                     sizeof(uint32_t)};
  ErrorOrSuccess result = ctx.enqueueFunctionExecDirect(
      **vadd, /*gridX=*/1, /*gridY=*/1, /*gridZ=*/1, /*blockX=*/1,
      /*blockY=*/1, /*blockZ=*/1, /*sharedMemBytes=*/0, /*attrs=*/nullptr,
      /*numAttrs=*/0, args, /*argCount=*/4, sizes);
  if (!result.isError())
    return childFail("poison", "a launch after a failed flush was accepted");
  std::string refused(result.getError());
  if (refused.find("an earlier megastep flush failed") == std::string::npos)
    return childFail(
        "poison",
        ("the post-failure refusal does not name the cause: " + refused)
            .c_str());
  return 0;
}

int runChildMarks() {
  ErrorOr<DeviceRef> device = QuRTDevice::create(0);
  if (device.isError())
    return childFail("device", device.getError());
  ErrorOr<DeviceContextRef> context = (*device)->createContext();
  if (context.isError())
    return childFail("context", context.getError());
  DeviceContext &ctx = **context;
  ErrorOr<DeviceFunctionRef> vadd = ctx.loadFunction("", "vadd", "");
  if (vadd.isError())
    return childFail("loadFunction", vadd.getError());
  ErrorOr<DeviceBufferRef> arena = ctx.enqueueCreateBuffer(kArenaBytes);
  if (arena.isError())
    return childFail("arena", arena.getError());
  auto *arenaBytes = static_cast<uint8_t *>((*arena)->ptr());

  MODULAR_NPUDeclareConstantHostRange(weightBlob, sizeof(weightBlob));
  ErrorOr<DeviceBufferRef> weight = ctx.enqueueCreateBuffer(kWeightBytes);
  if (weight.isError())
    return childFail("weight", weight.getError());
  auto *weightBytes = static_cast<uint8_t *>((*weight)->ptr());
  ErrorOrSuccess filled =
      ctx.enqueueCopyToDevice(weightBytes, weightBlob, kWeightBytes);
  if (filled.isError())
    return childFail("fill", filled.getError());

  // The serve exe's R5 brackets, emitted here the way the exec driver emits
  // them: everything the step did -- registrations included -- must land
  // between them.
  npu_perf_mark("step_00_start");
  int status = step(ctx, **vadd, arenaBytes, weightBytes);
  npu_perf_mark("step_00_done");
  return status;
}

//===----------------------------------------------------------------------===//
// Orchestrator
//===----------------------------------------------------------------------===//

struct Leg {
  const char *mode;
  bool megastep = true;
  bool refuseRun = false;
  std::string resultPath;
  std::string countPath;
  std::string stderrPath;
  std::string perfDir;
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
    if (!leg.perfDir.empty())
      setenv("MODULAR_NPU_PERF_LOG", leg.perfDir.c_str(), 1);
    if (leg.refuseRun)
      setenv("NPU_FAKE_TRANSPORT_REFUSE_MEGASTEP_RUN", "1", 1);
    int fd = open(leg.stderrPath.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (fd >= 0) {
      dup2(fd, 2);
      close(fd);
    }
    execl(self.c_str(), self.c_str(), leg.mode, leg.resultPath.c_str(),
          static_cast<char *>(nullptr));
    _exit(127);
  }
  int status = 0;
  if (waitpid(pid, &status, 0) != pid)
    return -1;
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

std::vector<std::string> readLines(const std::string &path) {
  std::ifstream in(path);
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(in, line))
    if (!line.empty())
      lines.push_back(line);
  return lines;
}

std::vector<uint8_t> readBytes(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)),
                              std::istreambuf_iterator<char>());
}

size_t countMatches(const std::vector<std::string> &lines,
                    const std::string &needle) {
  size_t matches = 0;
  for (const std::string &line : lines)
    if (line.find(needle) != std::string::npos)
      ++matches;
  return matches;
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

  auto leg = [&](const char *name, const char *mode) {
    Leg l;
    l.mode = mode;
    l.resultPath = base + "/" + name + ".bin";
    l.countPath = base + "/" + name + ".count";
    l.stderrPath = base + "/" + name + ".stderr";
    return l;
  };

  // LIFECYCLE: the exact invocation sequence over four steps.
  Leg lifecycle = leg("lifecycle", "--child-lifecycle");
  expect(spawnChild(self, fakeLib, lifecycle) == 0, "lifecycle leg exited 0");
  std::vector<std::string> counts = readLines(lifecycle.countPath);
  const std::vector<std::string> wanted = {"0", "3", "3", "0", "3", "0", "3"};
  expect(counts == wanted,
         "the session ledger reads register/run/run/re-register-after-"
         "rewrite/run/re-register-after-free/run");
  std::vector<std::string> reports = readLines(lifecycle.stderrPath);
  // Once, latched: order 087 showed the per-flush repeat collapses under log
  // dedup into exactly the wrong reading. The re-registrations stay proven by
  // the ledger's later zero-launch invocations above.
  expect(countMatches(reports, "npu megastep: registered 1 constant "
                               "allocations") == 1,
         "the registration self-report announces once and leaves the ledger "
         "to carry the re-registrations");

  // OVERFLOW: the step splits rather than widening the claim, and the split
  // bytes still match per-call.
  Leg overflowPerCall = leg("overflowPerCall", "--child-overflow");
  overflowPerCall.megastep = false;
  Leg overflowTabled = leg("overflowTabled", "--child-overflow");
  expect(spawnChild(self, fakeLib, overflowPerCall) == 0,
         "overflow per-call leg exited 0");
  expect(spawnChild(self, fakeLib, overflowTabled) == 0,
         "overflow megastep leg exited 0");
  counts = readLines(overflowTabled.countPath);
  expect(counts == std::vector<std::string>({"7", "3"}),
         "eleven mutable allocations against eight slots split the step "
         "into runs of 7 and 3");
  expect(readBytes(overflowPerCall.resultPath) ==
                 readBytes(overflowTabled.resultPath) &&
             !readBytes(overflowPerCall.resultPath).empty(),
         "the split step's bytes differ from per-call");

  // CADENCE: the graph path's witness-per-launch bracketing must not shred
  // the table -- allocation holds across it, frees are deferred to the flush.
  Leg cadencePerCall = leg("cadencePerCall", "--child-cadence");
  cadencePerCall.megastep = false;
  Leg cadenceTabled = leg("cadenceTabled", "--child-cadence");
  expect(spawnChild(self, fakeLib, cadencePerCall) == 0,
         "cadence per-call leg exited 0");
  expect(spawnChild(self, fakeLib, cadenceTabled) == 0,
         "cadence megastep leg exited 0");
  counts = readLines(cadenceTabled.countPath);
  expect(counts == std::vector<std::string>({"0", "6"}),
         "a witness allocated before and freed after every launch still "
         "leaves one six-launch run (order 087's cadence)");
  expect(countMatches(readLines(cadenceTabled.stderrPath),
                      "npu megastep: active (MODULAR_NPU_MEGASTEP=1, 6 "
                      "launches tabled)") == 1,
         "the cadence leg's table formed at full width");
  expect(readBytes(cadencePerCall.resultPath) ==
                 readBytes(cadenceTabled.resultPath) &&
             !readBytes(cadenceTabled.resultPath).empty(),
         "the cadence leg's bytes differ from per-call");

  // FALLBACK: a table of one is not a transport; it takes the per-call wire
  // and says so, and never claims to be an active megastep.
  Leg fallbackPerCall = leg("fallbackPerCall", "--child-fallback");
  fallbackPerCall.megastep = false;
  Leg fallbackTabled = leg("fallbackTabled", "--child-fallback");
  expect(spawnChild(self, fakeLib, fallbackPerCall) == 0,
         "fallback per-call leg exited 0");
  expect(spawnChild(self, fakeLib, fallbackTabled) == 0,
         "fallback megastep leg exited 0");
  counts = readLines(fallbackTabled.countPath);
  expect(counts == std::vector<std::string>({"0", "1"}),
         "the one-launch step went out per-call after its registration");
  {
    std::vector<std::string> lines = readLines(fallbackTabled.stderrPath);
    expect(countMatches(lines, "npu megastep: table degenerated to one "
                               "launch; using per-call dispatch") == 1,
           "the degrade is announced once");
    expect(countMatches(lines, "launches tabled") == 0,
           "a degenerate table must not claim to be an active megastep");
  }
  expect(readBytes(fallbackPerCall.resultPath) ==
                 readBytes(fallbackTabled.resultPath) &&
             !readBytes(fallbackTabled.resultPath).empty(),
         "the fallback's bytes differ from per-call");

  // POISON: a refused run surfaces at the flush AND stops the next launch --
  // the loud version of order 087's silent zeros.
  Leg poison = leg("poison", "--child-poison");
  poison.refuseRun = true;
  expect(spawnChild(self, fakeLib, poison) == 0,
         "the refused run was not surfaced and latched (poison leg)");
  expect(countMatches(readLines(poison.stderrPath),
                      "npu megastep: FLUSH FAILED:") >= 1,
         "the flush failure is said on stderr where no caller can swallow "
         "it");

  // MARKS: the megastep's timeline records land inside the step brackets.
  Leg marks = leg("marks", "--child-marks");
  marks.perfDir = base + "/perf";
  mkdir(marks.perfDir.c_str(), 0755);
  expect(spawnChild(self, fakeLib, marks) == 0, "marks leg exited 0");
  std::string perfPath;
  {
    std::string prefix = marks.perfDir + "/npu_perf_";
    // One child, one file: the pid is unknown but the directory holds only
    // this leg's log.
    FILE *ls = popen(("ls " + marks.perfDir).c_str(), "r");
    char name[256] = {};
    if (ls != nullptr && fscanf(ls, "%255s", name) == 1)
      perfPath = marks.perfDir + "/" + name;
    if (ls != nullptr)
      pclose(ls);
  }
  std::vector<std::string> perf = readLines(perfPath);
  long start = -1, registered = -1, run = -1, done = -1;
  for (size_t i = 0; i < perf.size(); ++i) {
    if (perf[i].find("\"phase\":\"step_00_start\"") != std::string::npos)
      start = static_cast<long>(i);
    if (perf[i].find("\"name\":\"megastep_register\"") != std::string::npos)
      registered = static_cast<long>(i);
    if (perf[i].find("\"name\":\"megastep\"") != std::string::npos)
      run = static_cast<long>(i);
    if (perf[i].find("\"phase\":\"step_00_done\"") != std::string::npos)
      done = static_cast<long>(i);
  }
  expect(start >= 0 && registered > start && run > registered && done > run,
         "the step marks bracket the megastep's records (start=" +
             std::to_string(start) + " register=" + std::to_string(registered) +
             " run=" + std::to_string(run) + " done=" + std::to_string(done) +
             "): per-step attribution holds with the flag on");

  return failures == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char **argv) {
  if (argc >= 2 && strcmp(argv[1], "--child-lifecycle") == 0)
    return runChildLifecycle();
  if (argc >= 3 && strcmp(argv[1], "--child-overflow") == 0)
    return runChildOverflow(argv[2],
                            getenv("MODULAR_NPU_MEGASTEP") != nullptr);
  if (argc >= 3 && strcmp(argv[1], "--child-cadence") == 0)
    return runChildCadence(argv[2]);
  if (argc >= 3 && strcmp(argv[1], "--child-fallback") == 0)
    return runChildFallback(argv[2]);
  if (argc >= 2 && strcmp(argv[1], "--child-poison") == 0)
    return runChildPoison();
  if (argc >= 2 && strcmp(argv[1], "--child-marks") == 0)
    return runChildMarks();
  int status = runOrchestrator(argv[0]);
  printf(status == 0 ? "PASS\n" : "FAILED\n");
  return status;
}
