//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// The MODULAR_NPU_POWER session vote, judged at the two seams it crosses: the
// control calls that reached the stand-in transport, and the self-reports on
// stderr. One leg per process because the driver binding and the skel handle
// latch for a process's life, exactly as they do in production.
//
// The claims, in the order a device leg would meet them. The vote is ON by
// default (orders 073 and 077): an unset flag votes the host latency once
// (DSPRPC_CONTROL_LATENCY, enable 1) and dispatches power_vote once before
// the first kernel, and says both things on stderr; `MODULAR_NPU_POWER=0` is
// the kill-switch, leaving both seams untouched with not one extra
// invocation for a session that dispatches no HMX-capable kernel; and a
// driver exporting no remote_handle64_control -- a genuinely absent symbol,
// not a refusing stub -- is reported by name while the DSP half and the
// launches themselves proceed. A silent skip in any of these is the one
// failure mode this flag cannot have: an A/B leg that cannot say which
// treatment it got is indistinguishable from its own control.
//
// Layered on that, the unvoted-HMX refusal (P5 M2.4 order 077: the HMX lock
// grant requires a live session vote): a dispatch of an HMX-capable kernel
// into a session whose vote is not live -- killed by =0, or failed for want
// of HAP_power_set -- is refused by name before any transport sees it, but
// ONLY when the loaded skel reports the HMX capability word; an HVX-only
// build keeps dispatching matmul under =0, because that is every recorded
// A/B's control arm.
//
//===----------------------------------------------------------------------===//

#include "QuRTDeviceContext.h"

#include <array>
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

/// One vadd through the real dispatch path, which is what opens the skel and
/// -- with the vote on -- places it ahead of the kernel. With `matmul` also
/// requested, follows with one HMX-capable dispatch: exit 3 with the refusal
/// on stderr when the guard refuses it, 0 when it dispatches.
int runChild(bool matmul) {
  ErrorOr<DeviceRef> device = QuRTDevice::create(0);
  if (device.isError()) {
    printf("child device: %s\n", device.getError());
    return 1;
  }
  ErrorOr<DeviceContextRef> context = (*device)->createContext();
  if (context.isError()) {
    printf("child context: %s\n", context.getError());
    return 1;
  }
  DeviceContext &ctx = **context;

  ErrorOr<DeviceBufferRef> arena = ctx.enqueueCreateBuffer(16384);
  if (arena.isError()) {
    printf("child arena: %s\n", arena.getError());
    return 1;
  }
  ErrorOr<DeviceFunctionRef> vadd = ctx.loadFunction("", "vadd", "");
  if (vadd.isError()) {
    printf("child loadFunction: %s\n", vadd.getError());
    return 1;
  }

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
  if (result.isError()) {
    printf("child dispatch: %s\n", result.getError());
    return 1;
  }
  if (!matmul)
    return 0;

  ErrorOr<DeviceFunctionRef> fn = ctx.loadFunction("", "matmul", "");
  if (fn.isError()) {
    printf("child loadFunction matmul: %s\n", fn.getError());
    return 1;
  }
  void *mc = bytes;
  void *ma = bytes + 4096;
  void *mb = bytes + 8192;
  void *mw = bytes + 12288;
  uint32_t m = 4, n = 4, k = 64;
  void *margs[7] = {&mc, &ma, &mb, &mw, &m, &n, &k};
  size_t msizes[7] = {sizeof(void *), sizeof(void *), sizeof(void *),
                      sizeof(void *), sizeof(uint32_t), sizeof(uint32_t),
                      sizeof(uint32_t)};
  ErrorOrSuccess mm = ctx.enqueueFunctionExecDirect(
      **fn, /*gridX=*/1, /*gridY=*/1, /*gridZ=*/1, /*blockX=*/1,
      /*blockY=*/1, /*blockZ=*/1, /*sharedMemBytes=*/0, /*attrs=*/nullptr,
      /*numAttrs=*/0, margs, /*argCount=*/7, msizes);
  if (mm.isError()) {
    fprintf(stderr, "child matmul: %s\n", mm.getError());
    return 3;
  }
  return 0;
}

struct Leg {
  /// The MODULAR_NPU_POWER value the leg runs under, or null for unset.
  const char *power = nullptr;
  std::string fakeLib;
  std::string stderrPath;
  std::string logPath;
  std::string countPath;
  /// Whether the child follows the vadd with one HMX-capable dispatch.
  bool matmul = false;
  /// Which skel the stand-in transport plays: an HMX_MATMUL=1 build, a
  /// protection domain without HAP_power_set, a skel predating the
  /// power_vote row.
  bool hmxBuild = false;
  bool powerNotResident = false;
  bool noPowerVoteRow = false;
};

int spawnChild(const std::string &self, const Leg &leg) {
  pid_t pid = fork();
  if (pid < 0)
    return -1;
  if (pid == 0) {
    setenv("MODULAR_NPU_TRANSPORT_LIB", leg.fakeLib.c_str(), 1);
    setenv("NPU_FAKE_TRANSPORT_ATTR_LOG", leg.logPath.c_str(), 1);
    setenv("NPU_FAKE_TRANSPORT_COUNT", leg.countPath.c_str(), 1);
    if (leg.power != nullptr)
      setenv("MODULAR_NPU_POWER", leg.power, 1);
    else
      unsetenv("MODULAR_NPU_POWER");
    if (leg.hmxBuild)
      setenv("NPU_FAKE_TRANSPORT_HMX_BUILD", "1", 1);
    if (leg.powerNotResident)
      setenv("NPU_FAKE_TRANSPORT_POWER_NOT_RESIDENT", "1", 1);
    if (leg.noPowerVoteRow)
      setenv("NPU_FAKE_TRANSPORT_NO_POWER_VOTE", "1", 1);
    int fd = open(leg.stderrPath.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (fd >= 0) {
      dup2(fd, 2);
      close(fd);
    }
    execl(self.c_str(), self.c_str(),
          leg.matmul ? "--child-matmul" : "--child",
          static_cast<char *>(nullptr));
    _exit(127);
  }
  int status = 0;
  if (waitpid(pid, &status, 0) != pid)
    return -1;
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

std::vector<std::string> readLines(const std::string &path) {
  std::vector<std::string> lines;
  std::ifstream in(path);
  for (std::string line; std::getline(in, line);)
    if (!line.empty())
      lines.push_back(line);
  return lines;
}

bool anyContains(const std::vector<std::string> &lines,
                 const std::string &needle) {
  for (const std::string &line : lines)
    if (line.find(needle) != std::string::npos)
      return true;
  return false;
}

size_t countPrefixed(const std::vector<std::string> &lines,
                     const std::string &prefix) {
  size_t count = 0;
  for (const std::string &line : lines)
    if (line.rfind(prefix, 0) == 0)
      ++count;
  return count;
}

int runOrchestrator(const char *self) {
  const char *fake = getenv("NPU_FAKE_TRANSPORT");
  const char *fakeNoControl = getenv("NPU_FAKE_TRANSPORT_NO_CONTROL");
  const char *tmp = getenv("TEST_TMPDIR");
  if (fake == nullptr || fakeNoControl == nullptr || tmp == nullptr) {
    printf("FAIL: NPU_FAKE_TRANSPORT, NPU_FAKE_TRANSPORT_NO_CONTROL and "
           "TEST_TMPDIR must be set\n");
    return 1;
  }
  char resolved[4096];
  std::string fakeLib =
      realpath(fake, resolved) != nullptr ? resolved : std::string(fake);
  std::string noControlLib = realpath(fakeNoControl, resolved) != nullptr
                                 ? resolved
                                 : std::string(fakeNoControl);
  std::string base(tmp);

  auto legPaths = [&base](const char *name) {
    return std::array<std::string, 3>{base + "/" + name + ".stderr",
                                      base + "/" + name + ".log",
                                      base + "/" + name + ".count"};
  };

  auto pathsOff = legPaths("off");
  Leg off{nullptr, fakeLib, pathsOff[0], pathsOff[1], pathsOff[2]};
  auto pathsZero = legPaths("zero");
  Leg zero{"0", fakeLib, pathsZero[0], pathsZero[1], pathsZero[2]};
  auto pathsOn = legPaths("on");
  Leg on{"1", fakeLib, pathsOn[0], pathsOn[1], pathsOn[2]};
  auto pathsNoCtl = legPaths("noctl");
  Leg noControl{"1", noControlLib, pathsNoCtl[0], pathsNoCtl[1],
                pathsNoCtl[2]};

  auto pathsHmxVoted = legPaths("hmxvoted");
  Leg hmxVoted{nullptr, fakeLib, pathsHmxVoted[0], pathsHmxVoted[1],
               pathsHmxVoted[2], /*matmul=*/true, /*hmxBuild=*/true};
  auto pathsHmxZero = legPaths("hmxzero");
  Leg hmxZero{"0", fakeLib, pathsHmxZero[0], pathsHmxZero[1], pathsHmxZero[2],
              /*matmul=*/true, /*hmxBuild=*/true};
  auto pathsHmxDead = legPaths("hmxdead");
  Leg hmxDead{nullptr, fakeLib, pathsHmxDead[0], pathsHmxDead[1],
              pathsHmxDead[2], /*matmul=*/true, /*hmxBuild=*/true,
              /*powerNotResident=*/true};
  auto pathsHvxZero = legPaths("hvxzero");
  Leg hvxZero{"0", fakeLib, pathsHvxZero[0], pathsHvxZero[1], pathsHvxZero[2],
              /*matmul=*/true};
  auto pathsOldSkel = legPaths("oldskel");
  Leg oldSkel{nullptr, fakeLib, pathsOldSkel[0], pathsOldSkel[1],
              pathsOldSkel[2], /*matmul=*/true, /*hmxBuild=*/false,
              /*powerNotResident=*/false, /*noPowerVoteRow=*/true};

  expect(spawnChild(self, off) == 0, "flag-unset leg exited 0");
  expect(spawnChild(self, zero) == 0, "flag-zero leg exited 0");
  expect(spawnChild(self, on) == 0, "flag-on leg exited 0");
  expect(spawnChild(self, noControl) == 0, "no-control leg exited 0");
  expect(spawnChild(self, hmxVoted) == 0, "voted HMX-build leg exited 0");
  expect(spawnChild(self, hmxZero) == 3,
         "unvoted (=0) HMX-build matmul was not refused");
  expect(spawnChild(self, hmxDead) == 3,
         "vote-failed HMX-build matmul was not refused");
  expect(spawnChild(self, hvxZero) == 0,
         "unvoted (=0) HVX-only matmul did not dispatch -- the control arm "
         "of every recorded A/B just broke");
  expect(spawnChild(self, oldSkel) == 0,
         "a skel predating the power_vote row did not keep dispatching");
  if (failures != 0)
    return 1;

  // Default on: both halves vote once, before the kernel, and both say so --
  // pinned unset and "1" alike, because the flip makes unset the voted arm.
  for (const Leg *leg : {&off, &on}) {
    std::vector<std::string> err = readLines(leg->stderrPath);
    expect(anyContains(err, "npu power: host latency vote set"),
           "vote on, but the host latency vote did not report itself");
    expect(anyContains(err, "npu power: dsp vote max (dcvs off, sleep off, "
                            "hmx clock pinned)"),
           "vote on, but the dsp vote did not report itself");

    std::vector<std::string> log = readLines(leg->logPath);
    expect(countPrefixed(log, "control ") == 1,
           "vote on, but remote_handle64_control was not called exactly once");
    for (const std::string &line : log) {
      if (line.rfind("control ", 0) != 0)
        continue;
      expect(line.find(" 1 1 100") != std::string::npos,
             "the latency vote's payload is not req=1 enable=1 latency=100: " +
                 line);
    }
    expect(readLines(leg->countPath).size() == 2,
           "vote on, but the vote dispatch and the vadd were not exactly two "
           "invocations");
  }

  // The kill-switch: neither seam touched, not one extra invocation, and
  // nothing to say -- pinned, because the unvoted arm of every device A/B is
  // this, and it must stay wire-identical to the pre-vote recordings for a
  // session that dispatches no HMX-capable kernel.
  {
    std::vector<std::string> err = readLines(zero.stderrPath);
    expect(!anyContains(err, "npu power:"),
           "kill-switch set, but a power self-report was printed");
    expect(countPrefixed(readLines(zero.logPath), "control ") == 0,
           "kill-switch set, but remote_handle64_control was called");
    expect(readLines(zero.countPath).size() == 1,
           "kill-switch set, but the leg made more than the one vadd "
           "invocation");
  }

  // A driver without the symbol: the absence is named, and everything else
  // -- the DSP half, the launch itself -- still happens.
  {
    std::vector<std::string> err = readLines(noControl.stderrPath);
    expect(anyContains(err, "exports no remote_handle64_control"),
           "missing remote_handle64_control was not reported by name");
    expect(!anyContains(err, "npu power: host latency vote set"),
           "the host latency vote claims to be set with no symbol to set it");
    expect(anyContains(err, "npu power: dsp vote max (dcvs off, sleep off, "
                            "hmx clock pinned)"),
           "a missing host symbol silenced the independent dsp vote");
    expect(countPrefixed(readLines(noControl.logPath), "control ") == 0,
           "a control call crossed a seam whose symbol does not exist");
    expect(readLines(noControl.countPath).size() == 2,
           "the no-control leg did not still dispatch the vote and the vadd");
  }

  // An HMX build with the vote live dispatches matmul like any other kernel:
  // vote, vadd, matmul -- three invocations, no refusal.
  {
    std::vector<std::string> err = readLines(hmxVoted.stderrPath);
    expect(!anyContains(err, "child matmul:"),
           "a voted session refused an HMX-capable dispatch");
    expect(readLines(hmxVoted.countPath).size() == 3,
           "the voted HMX leg was not exactly vote + vadd + matmul");
  }

  // The unvoted-HMX refusal, both ways a session loses its vote. The wording
  // is pinned because the whole point is that the next person reads the fix
  // in the error instead of spending seven windows on witness words.
  {
    std::vector<std::string> err = readLines(hmxZero.stderrPath);
    expect(anyContains(err, "child matmul:"),
           "the =0 HMX leg's matmul refusal did not reach the child");
    expect(anyContains(err, "(MODULAR_NPU_POWER=0)"),
           "the =0 refusal does not name the kill-switch as the reason");
    expect(anyContains(err, "HMX requires the session power vote; "
                            "MODULAR_NPU_POWER=0 disables HMX"),
           "the =0 refusal does not carry the flag-and-fix sentence");
    expect(anyContains(err, "P5 M2.4 order 077"),
           "the =0 refusal does not cite the 077 finding");
    // The vadd, then the one enable=0 capability query the guard sends: the
    // refused matmul itself never crosses.
    expect(readLines(hmxZero.countPath).size() == 2,
           "the =0 HMX leg was not exactly vadd + capability query");
  }
  {
    std::vector<std::string> err = readLines(hmxDead.stderrPath);
    expect(anyContains(err, "npu power: dsp vote unavailable"),
           "the not-resident leg's vote did not report its own failure");
    expect(anyContains(err,
                       "HAP_power_set is not resident in this protection "
                       "domain). An unvoted session's HMX lock is refused"),
           "the vote-failed refusal does not name the failed vote as the "
           "reason");
  }

  // The scoping that keeps the control arms alive: an HVX-only skel answers
  // the capability query with no marker, and matmul dispatches unvoted.
  {
    std::vector<std::string> err = readLines(hvxZero.stderrPath);
    expect(!anyContains(err, "child matmul:"),
           "an HVX-only build's matmul was refused under =0");
    expect(readLines(hvxZero.countPath).size() == 3,
           "the HVX =0 leg was not exactly vadd + capability query + matmul");
  }

  // A skel predating the row refuses the vote by name; it predates HMX with
  // it, so dispatches proceed. The refused vote and the refused query never
  // reach the count.
  {
    std::vector<std::string> err = readLines(oldSkel.stderrPath);
    expect(anyContains(err, "predates the power_vote row"),
           "the old-skel vote refusal was not reported by name");
    expect(!anyContains(err, "child matmul:"),
           "an old skel's matmul was refused for want of a vote it cannot "
           "carry");
    expect(readLines(oldSkel.countPath).size() == 2,
           "the old-skel leg was not exactly vadd + matmul");
  }

  if (failures == 0)
    printf("PASS: npu_power_vote_test\n");
  return failures == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char **argv) {
  if (argc > 1 && strcmp(argv[1], "--child") == 0)
    return runChild(/*matmul=*/false);
  if (argc > 1 && strcmp(argv[1], "--child-matmul") == 0)
    return runChild(/*matmul=*/true);
  return runOrchestrator(argv[0]);
}
