//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Host-side proof that the perf timeline measures what it claims, before it is
// ever pointed at a device.
//
// A harness whose first outing is also its debut is a harness nobody can
// disagree with, so this spends time in places whose duration is known by
// construction and checks the record puts it back where it was spent. The
// stand-in transport blocks for `NPU_FAKE_TRANSPORT_INVOKE_US` inside the
// invocation and reports `NPU_FAKE_TRANSPORT_DEVICE_US` as device-side; the
// child sleeps a known amount BETWEEN launches, which is the term that matters
// most on this transport -- one blocking round trip per kernel means the gap
// between them is time the NSP provably spent idle, and a harness that smeared
// that gap into the launch would hide the port's biggest number.
//
// The control that could fail: the same chain runs with the timeline off, and
// must produce no file and no error. An instrument that is not free when off
// is one nobody leaves in.
//
//===----------------------------------------------------------------------===//

#include "NPUDispatchRecorder.h"
#include "QuRTDeviceContext.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <sys/wait.h>
#include <thread>
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

constexpr size_t kArenaBytes = 1 << 20;

// What is injected, and therefore what the record has to find. The invoke and
// gap are milliseconds apart from the noise floor of a host clock, so the
// assertions can be tight without being flaky; the device figure is a tenth of
// the invoke so a report that leaked into the wrong term would be obvious.
constexpr long kInvokeMicros = 20000;
constexpr long kDeviceMicros = 2000;
constexpr long kGapMicros = 40000;
constexpr uint64_t kLaunches = 3;

// A host under load can only make a span longer, never shorter, so the lower
// bound is exact and the upper one carries the slack. Sized generously because
// a sleep is a floor, not a duration: `nanosleep(20 ms)` measures 24-30 ms on a
// macOS host with timer coalescing, which is the harness reading correctly and
// would otherwise read as a flake. Still well under what the failure mode
// costs -- a launch that had absorbed the 40 ms inter-launch gap would read
// 60 ms and be caught.
constexpr uint64_t kSlackNanos = 30000000;

int childFail(const char *stage, const char *error) {
  printf("child %s: %s\n", stage, error);
  return 1;
}

int runChild() {
  npu_perf_mark("process_start");

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
  auto *arenaBytes = static_cast<uint8_t *>((*arena)->ptr());

  ErrorOr<DeviceFunctionRef> vadd = ctx.loadFunction("", "vadd", "");
  if (vadd.isError())
    return childFail("loadFunction", vadd.getError());

  npu_perf_mark("model_init");

  void *buffers[3] = {arenaBytes, arenaBytes + 98304, arenaBytes + 196608};
  uint32_t count = 1024;
  void *args[4] = {&buffers[0], &buffers[1], &buffers[2], &count};
  size_t argSizes[4] = {sizeof(void *), sizeof(void *), sizeof(void *),
                        sizeof(uint32_t)};

  for (uint64_t i = 0; i < kLaunches; ++i) {
    if (i != 0)
      std::this_thread::sleep_for(std::chrono::microseconds(kGapMicros));
    ErrorOrSuccess result = ctx.enqueueFunctionExecDirect(
        **vadd, /*gridX=*/1, /*gridY=*/1, /*gridZ=*/1, /*blockX=*/1,
        /*blockY=*/1, /*blockZ=*/1, /*sharedMemBytes=*/0, /*attrs=*/nullptr,
        /*numAttrs=*/0, args, 4, argSizes);
    if (result.isError())
      return childFail("dispatch", result.getError());
  }

  npu_perf_mark("process_exit");
  return 0;
}

//===----------------------------------------------------------------------===//
// Orchestrator
//===----------------------------------------------------------------------===//

int spawnChild(const std::string &self, const std::string &fakeLib,
               const char *logDir, bool deviceTiming) {
  pid_t pid = fork();
  if (pid < 0)
    return -1;
  if (pid == 0) {
    setenv("MODULAR_NPU_TRANSPORT_LIB", fakeLib.c_str(), 1);
    setenv("NPU_FAKE_TRANSPORT_INVOKE_US",
           std::to_string(kInvokeMicros).c_str(), 1);
    setenv("NPU_FAKE_TRANSPORT_DEVICE_US",
           std::to_string(kDeviceMicros).c_str(), 1);
    if (logDir != nullptr)
      setenv("MODULAR_NPU_PERF_LOG", logDir, 1);
    else
      unsetenv("MODULAR_NPU_PERF_LOG");
    if (deviceTiming)
      setenv("MODULAR_NPU_PERF_DEVICE", "1", 1);
    else
      unsetenv("MODULAR_NPU_PERF_DEVICE");
    execl(self.c_str(), self.c_str(), "--child", nullptr);
    _exit(127);
  }
  int status = 0;
  waitpid(pid, &status, 0);
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

struct Record {
  std::string kind;
  std::string name;
  uint64_t at = 0;
  uint64_t prep = 0;
  uint64_t invoke = 0;
  uint64_t post = 0;
  bool hasDevice = false;
  uint64_t deviceNanos = 0;
  bool hasSync = false;
  uint64_t syncWaits = 0;
  uint64_t syncNapExits = 0;
  uint64_t syncNapQticks = 0;
  uint64_t syncBudgetHi = 0;
};

/// The one number a JSON field carries, found by name. Deliberately not a JSON
/// parser: these records are written by one `snprintf` in one file, and a
/// reader that reproduces their shape would pass even if that shape changed.
bool fieldOf(const std::string &line, const char *key, uint64_t &out) {
  std::string needle = std::string("\"") + key + "\":";
  size_t at = line.find(needle);
  if (at == std::string::npos)
    return false;
  out = strtoull(line.c_str() + at + needle.size(), nullptr, 10);
  return true;
}

std::string stringFieldOf(const std::string &line, const char *key) {
  std::string needle = std::string("\"") + key + "\":\"";
  size_t at = line.find(needle);
  if (at == std::string::npos)
    return {};
  size_t start = at + needle.size();
  size_t end = line.find('"', start);
  return line.substr(start, end - start);
}

std::vector<Record> readTimeline(const std::string &directory) {
  std::vector<Record> records;
  for (const auto &entry : std::filesystem::directory_iterator(directory)) {
    std::ifstream file(entry.path());
    std::string line;
    while (std::getline(file, line)) {
      Record record;
      record.kind = stringFieldOf(line, "k");
      if (record.kind == "mark") {
        record.name = stringFieldOf(line, "phase");
        fieldOf(line, "t", record.at);
      } else if (record.kind == "launch") {
        record.name = stringFieldOf(line, "name");
        fieldOf(line, "t", record.at);
        fieldOf(line, "prep", record.prep);
        fieldOf(line, "invoke", record.invoke);
        fieldOf(line, "post", record.post);
        size_t dev = line.find("\"dev\":[");
        if (dev != std::string::npos) {
          // The qtimer pair is the last two of the six.
          const char *cursor = line.c_str() + dev + 7;
          uint64_t stamps[6] = {};
          for (uint64_t &stamp : stamps) {
            stamp = strtoull(cursor, const_cast<char **>(&cursor), 10);
            if (*cursor == ',')
              ++cursor;
          }
          record.hasDevice = true;
          record.deviceNanos =
              (stamps[5] - stamps[4]) * 1000000000ull / 19200000ull;
        }
        if (line.find("\"sync\":{") != std::string::npos) {
          record.hasSync = true;
          fieldOf(line, "waits", record.syncWaits);
          fieldOf(line, "nap", record.syncNapExits);
          fieldOf(line, "nap_qt", record.syncNapQticks);
          fieldOf(line, "budget_hi", record.syncBudgetHi);
        }
      } else {
        continue;
      }
      records.push_back(record);
    }
  }
  return records;
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
  std::string hostLog = base + "/perfHost";
  std::string deviceLog = base + "/perfDevice";
  std::string offLog = base + "/perfOff";

  expect(spawnChild(self, fakeLib, hostLog.c_str(), false) == 0,
         "host-timed run exited 0");
  expect(spawnChild(self, fakeLib, deviceLog.c_str(), true) == 0,
         "device-timed run exited 0");
  expect(spawnChild(self, fakeLib, nullptr, false) == 0,
         "untimed run exited 0");
  if (failures)
    return 1;

  // The control: off means off. No directory, no file, and the dispatches all
  // succeeded above.
  expect(!std::filesystem::exists(offLog),
         "the timeline writes nothing when MODULAR_NPU_PERF_LOG is unset");

  std::vector<Record> host = readTimeline(hostLog);
  std::vector<Record> launches;
  std::vector<Record> marks;
  for (const Record &record : host)
    (record.kind == "launch" ? launches : marks).push_back(record);

  expect(launches.size() == kLaunches,
         "every launch produced a record (" + std::to_string(launches.size()) +
             " of " + std::to_string(kLaunches) + ")");
  expect(marks.size() == 3, "every phase mark was written");
  if (failures)
    return 1;

  expect(marks[0].name == "process_start" && marks[1].name == "model_init" &&
             marks[2].name == "process_exit",
         "phase marks are in the order they were made");
  expect(marks[0].at <= marks[1].at && marks[1].at <= marks[2].at,
         "phase marks advance monotonically");

  const uint64_t injectedInvoke = uint64_t(kInvokeMicros) * 1000;
  const uint64_t injectedGap = uint64_t(kGapMicros) * 1000;

  for (const Record &record : launches) {
    expect(record.name == "vadd", "the record names the kernel dispatched");
    // The transport blocked for a known time; the span around it must contain
    // that time and little else.
    expect(record.invoke >= injectedInvoke,
           "invoke covers what the transport blocked for");
    expect(record.invoke < injectedInvoke + kSlackNanos,
           "invoke is the blocking call and not its neighbours (" +
               std::to_string(record.invoke) + " ns)");
    // Prep and post do real work -- packing, allocation lookup, writing the
    // record -- but must not have absorbed the injected wait.
    expect(record.prep < injectedInvoke,
           "prep is host packing, not the invocation");
    expect(record.post < injectedInvoke,
           "post is the return path, not the invocation");
  }

  // The number this instrument exists for: between one launch's return and the
  // next launch's entry, the device is idle. The child slept a known amount
  // there, and the timeline has to attribute it there rather than to a launch.
  for (size_t i = 1; i < launches.size(); ++i) {
    uint64_t previousExit = launches[i - 1].at + launches[i - 1].prep +
                            launches[i - 1].invoke + launches[i - 1].post;
    uint64_t gap = launches[i].at - previousExit;
    expect(gap >= injectedGap,
           "the inter-launch gap covers the host time between dispatches");
    expect(gap < injectedGap + kSlackNanos,
           "the inter-launch gap is that time and not more (" +
               std::to_string(gap) + " ns)");
  }

  expect(!launches[0].hasDevice,
         "no device report rides along unless it was asked for");
  expect(!launches[0].hasSync,
         "no sync block rides along unless the report slot travels");

  // With the report asked for, the launch carries the device's own reading
  // back, which is what splits the blocking invoke into NSP-busy and transport.
  std::vector<Record> device = readTimeline(deviceLog);
  size_t timed = 0;
  const uint64_t injectedDevice = uint64_t(kDeviceMicros) * 1000;
  for (const Record &record : device) {
    if (record.kind != "launch")
      continue;
    ++timed;
    expect(record.hasDevice, "MODULAR_NPU_PERF_DEVICE=1 brings a report back");
    // Converted from the qtimer's 19.2 MHz, so a tick of rounding is expected
    // and nothing more.
    expect(record.deviceNanos + 1000 >= injectedDevice &&
               record.deviceNanos <= injectedDevice + 1000,
           "the device figure is what the transport reported (" +
               std::to_string(record.deviceNanos) + " ns)");
    expect(record.invoke > record.deviceNanos,
           "the blocking invoke is longer than the device was busy, which is "
           "the transport cost the split exists to expose");
    // The sync block crossed the same slot and landed under its own key, with
    // the transport's numbers intact -- the egress order 243 could not find.
    expect(record.hasSync,
           "MODULAR_NPU_PERF_DEVICE=1 brings the sync counters back");
    expect(record.syncWaits == 9 && record.syncNapExits == 2 &&
               record.syncNapQticks == 96000 && record.syncBudgetHi == 32000,
           "the sync counters are the transport's, unmangled");
  }
  expect(timed == kLaunches, "every launch carried a report");

  if (failures == 0)
    printf("PASS: the perf timeline attributes injected time correctly\n");
  return failures == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "--child") == 0)
    return runChild();
  return runOrchestrator(argv[0]);
}
