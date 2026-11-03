//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// The megastep against the REAL step, not a synthetic chain: the recorded
// whole-graph denoise step (order-041/071 evidence, 398 launches, 185
// allocations, every weight its own allocation) distilled by
// npu_megastep_replay_fixture.py and driven through the real table builder
// against the stand-in transport.
//
// What the synthetic differential cannot claim and this does:
//
//   * the real selector mix and the real allocation pattern -- 172
//     allocations classify init-constant under the runtime's own provenance
//     test when their copies come from a declared range, exactly the
//     registry-weight shape the fixture derives from the record;
//   * the whole 398-launch step travels as ONE run invocation after exactly
//     ceil(172/8) = 22 registration calls -- the fixture's header states the
//     expectation and the count ledger has to match it;
//   * the real step's mutable union (the activation arena, four witness
//     buffers, one staging buffer) fits the call's own slots, which is the
//     load-bearing claim behind "one dispatch per step";
//   * byte-for-byte equality against the per-call leg across every
//     allocation, with a perturbed control that must fail.
//
// The fixture preserves launch order, selectors, kernel scalars, allocation
// identity and offset distinctness, and scales byte sizes down (453 MB of
// arena does not belong in a host test); see the generator's docstring for
// exactly what is preserved and why that is sufficient for this transport.
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
#include <map>
#include <sstream>
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

int childFail(const char *stage, const std::string &error) {
  printf("child %s: %s\n", stage, error.c_str());
  return 1;
}

/// One source for every copy the record classifies constant, declared the way
/// a loader declares a registry blob; one undeclared source for everything
/// else. Static so the declaration outlives every copy.
constexpr size_t kSourceBytes = 1 << 18;
uint8_t declaredBlob[kSourceBytes];
uint8_t scratchBlob[kSourceBytes];

struct Fixture {
  size_t allocations = 0;
  size_t launches = 0;
  size_t constants = 0;
  size_t registrationCalls = 0;
  size_t runs = 0;
  std::vector<std::string> lines;
};

bool readFixture(const char *path, Fixture &fixture) {
  std::ifstream in(path);
  std::string line;
  if (!std::getline(in, line))
    return false;
  char tag = 0;
  std::istringstream header(line);
  if (!(header >> tag >> fixture.allocations >> fixture.launches >>
        fixture.constants >> fixture.registrationCalls >> fixture.runs) ||
      tag != 'H')
    return false;
  while (std::getline(in, line))
    if (!line.empty())
      fixture.lines.push_back(line);
  return true;
}

/// Replays the fixture: allocations, then copies and launches in recorded
/// order, then the synchronize the graph boundary does. `perturbAt` moves the
/// first operand of that launch by 64 bytes -- the smallest divergence a
/// mis-resolved table reference produces.
int runChild(const char *resultPath, long perturbAt) {
  const char *fixturePath = getenv("NPU_MEGASTEP_FIXTURE");
  if (fixturePath == nullptr)
    return childFail("fixture", "NPU_MEGASTEP_FIXTURE must be set");
  Fixture fixture;
  if (!readFixture(fixturePath, fixture))
    return childFail("fixture", "unreadable header");

  ErrorOr<DeviceRef> device = QuRTDevice::create(0);
  if (device.isError())
    return childFail("device", device.getError());
  ErrorOr<DeviceContextRef> context = (*device)->createContext();
  if (context.isError())
    return childFail("context", context.getError());
  DeviceContext &ctx = **context;

  MODULAR_NPUDeclareConstantHostRange(declaredBlob, sizeof(declaredBlob));

  std::map<long, DeviceBufferRef> buffers;
  std::map<long, size_t> sizes;
  std::map<std::string, DeviceFunctionRef> kernels;
  long launchIndex = 0;

  for (const std::string &line : fixture.lines) {
    std::istringstream fields(line);
    char tag = 0;
    fields >> tag;
    if (tag == 'A') {
      long id = 0;
      size_t byteSize = 0;
      fields >> id >> byteSize;
      if (byteSize > kSourceBytes)
        return childFail("alloc", "scaled allocation outgrew the sources");
      ErrorOr<DeviceBufferRef> buffer = ctx.enqueueCreateBuffer(byteSize);
      if (buffer.isError())
        return childFail("alloc", buffer.getError());
      sizes[id] = byteSize;
      buffers.emplace(id, std::move(*buffer));
    } else if (tag == 'C') {
      long id = 0;
      size_t offset = 0, count = 0;
      char source = 0;
      fields >> id >> offset >> count >> source;
      auto *dst = static_cast<uint8_t *>(buffers.at(id)->ptr()) + offset;
      const uint8_t *src = source == 'd' ? declaredBlob : scratchBlob;
      ErrorOrSuccess copied = ctx.enqueueCopyToDevice(dst, src, count);
      if (copied.isError())
        return childFail("copy", copied.getError());
    } else if (tag == 'L') {
      std::string name;
      unsigned sel = 0, mask = 0;
      size_t bufCount = 0;
      fields >> name >> sel >> mask >> bufCount;
      auto found = kernels.find(name);
      if (found == kernels.end()) {
        ErrorOr<DeviceFunctionRef> kernel = ctx.loadFunction("", name, "");
        if (kernel.isError())
          return childFail("loadFunction", kernel.getError());
        found = kernels.emplace(name, std::move(*kernel)).first;
      }

      void *operands[8] = {};
      uint64_t scalars[16] = {};
      void *args[24] = {};
      size_t argSizes[24] = {};
      for (size_t i = 0; i < bufCount; ++i) {
        std::string ref;
        fields >> ref;
        size_t colon = ref.find(':');
        long id = strtol(ref.substr(0, colon).c_str(), nullptr, 10);
        size_t offset = strtoul(ref.substr(colon + 1).c_str(), nullptr, 10);
        if (launchIndex == perturbAt && i == 0)
          offset += 64;
        operands[i] = static_cast<uint8_t *>(buffers.at(id)->ptr()) + offset;
        args[i] = &operands[i];
        argSizes[i] = sizeof(void *);
      }
      size_t scalarCount = 0;
      fields >> scalarCount;
      for (size_t i = 0; i < scalarCount; ++i) {
        fields >> scalars[i];
        args[bufCount + i] = &scalars[i];
        argSizes[bufCount + i] = sizeof(uint64_t);
      }

      ErrorOrSuccess result = ctx.enqueueFunctionExecDirect(
          *found->second, /*gridX=*/1, /*gridY=*/1, /*gridZ=*/1,
          /*blockX=*/mask, /*blockY=*/1, /*blockZ=*/1, /*sharedMemBytes=*/0,
          /*attrs=*/nullptr, /*numAttrs=*/0, args,
          static_cast<uint32_t>(bufCount + scalarCount), argSizes);
      if (result.isError())
        return childFail("dispatch", result.getError());
      ++launchIndex;
    } else {
      return childFail("fixture", "unknown record: " + line);
    }
  }

  ErrorOrSuccess sync = ctx.synchronize();
  if (sync.isError())
    return childFail("synchronize", sync.getError());

  std::ofstream out(resultPath, std::ios::binary);
  for (const auto &[id, buffer] : buffers)
    out.write(static_cast<const char *>(buffer->ptr()),
              static_cast<std::streamsize>(sizes.at(id)));
  return out.good() ? 0 : 1;
}

//===----------------------------------------------------------------------===//
// Orchestrator
//===----------------------------------------------------------------------===//

struct Leg {
  bool megastep = false;
  long perturbAt = -1;
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
    int fd = open(leg.stderrPath.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (fd >= 0) {
      dup2(fd, 2);
      close(fd);
    }
    std::string perturb = std::to_string(leg.perturbAt);
    execl(self.c_str(), self.c_str(), "--child", leg.resultPath.c_str(),
          perturb.c_str(), static_cast<char *>(nullptr));
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
  const char *fixturePath = getenv("NPU_MEGASTEP_FIXTURE");
  if (fake == nullptr || tmp == nullptr || fixturePath == nullptr) {
    printf("FAIL: NPU_FAKE_TRANSPORT, NPU_MEGASTEP_FIXTURE and TEST_TMPDIR "
           "must be set\n");
    return 1;
  }
  Fixture fixture;
  if (!readFixture(fixturePath, fixture)) {
    printf("FAIL: fixture header unreadable\n");
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
  perturbed.perturbAt = static_cast<long>(fixture.launches / 2);

  expect(spawnChild(self, fakeLib, perCall) == 0, "per-call leg exited 0");
  expect(spawnChild(self, fakeLib, tabled) == 0, "megastep leg exited 0");
  expect(spawnChild(self, fakeLib, perturbed) == 0, "perturbed leg exited 0");
  if (failures != 0)
    return 1;

  // The recorded step carries matmuls, and this unvoted (=0) session answers
  // the driver's one HMX-capability query before the first of them -- one
  // extra single-launch invocation per leg, exactly what a =0 render on an
  // HVX tuple sends (npu_power_vote_test pins the query itself).
  constexpr uint32_t kHmxQuery = 1;

  size_t invocations = 0, empty = 0;
  uint32_t widest = 0, launches = 0;
  readCounts(perCall.countPath, &invocations, &widest, &launches, &empty);
  expect(invocations == fixture.launches + kHmxQuery && widest == 1,
         "the per-call leg sent one launch per invocation");

  readCounts(tabled.countPath, &invocations, &widest, &launches, &empty);
  expect(launches == fixture.launches + kHmxQuery,
         "the tabled leg ran every launch of the recorded step");
  expect(empty == fixture.registrationCalls,
         "the registration ledger matches the record's constant "
         "classification (" +
             std::to_string(fixture.constants) + " allocations in " +
             std::to_string(fixture.registrationCalls) + " calls)");
  expect(invocations == fixture.registrationCalls + fixture.runs + kHmxQuery,
         "the whole recorded step travelled as " +
             std::to_string(fixture.runs) + " run invocation(s)");
  expect(widest == fixture.launches / fixture.runs ||
             widest <= fixture.launches,
         "run width is consistent with the expected split");
  printf("replay: %zu invocations for %u launches, widest %u, "
         "%zu registrations\n",
         invocations, launches, widest, empty);

  expect(fileHasLine(tabled.stderrPath,
                     "npu megastep: active (MODULAR_NPU_MEGASTEP=1, " +
                         std::to_string(fixture.launches) +
                         " launches tabled)"),
         "the activation self-report names the real step's width");
  expect(fileHasLine(tabled.stderrPath,
                     "npu megastep: registered " +
                         std::to_string(fixture.constants) +
                         " constant allocations"),
         "the registration self-report names the record's constant count");

  std::vector<uint8_t> perCallBytes = readBytes(perCall.resultPath);
  std::vector<uint8_t> tabledBytes = readBytes(tabled.resultPath);
  expect(!perCallBytes.empty(), "the per-call leg wrote a result");
  expect(perCallBytes == tabledBytes,
         "megastep and per-call replay of the recorded step differ");

  std::vector<uint8_t> perturbedBytes = readBytes(perturbed.resultPath);
  expect(perturbedBytes.size() == perCallBytes.size(),
         "the perturbed leg wrote a full result");
  expect(perturbedBytes != perCallBytes,
         "a deliberately altered launch produced identical bytes: the "
         "comparison is not reading what the replay wrote");

  return failures == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char **argv) {
  if (argc == 4 && strcmp(argv[1], "--child") == 0)
    return runChild(argv[2], strtol(argv[3], nullptr, 10));
  int status = runOrchestrator(argv[0]);
  printf(status == 0 ? "PASS\n" : "FAILED\n");
  return status;
}
