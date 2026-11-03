//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// MODULAR_NPU_MAP_ONCE at the seam it moves: an allocation is mapped to the
// DSP once, for its life, instead of once per call by the driver underneath
// `remote_handle64_invoke`. Every claim below is judged by the FastRPC calls
// that actually crossed the driver seam, which the stand-in appends to a log
// file -- the same observable the NOFLUSH probe's test reads.
//
// The properties, in the order a device leg would meet them:
//
//   * flag OFF, same binary, is the negative control: the mapping seam is
//     never touched, so the control arm is today's driver behaviour bit for
//     bit;
//   * ALLOCATING MAPS NOTHING. `fastrpc_mmap` is itself an RPC on the domain,
//     and order 163 measured it returning AEE_ERPC (512) at every length from
//     a driver that mapped at allocation time, because no session was open
//     yet. The mapping belongs at the first launch naming the allocation,
//     after the session the dispatch path opens -- the ordering the queue
//     transport already gets on silicon;
//   * the first launch maps it exactly once, in FASTRPC_MAP_FD and no other
//     mode -- order 081 measured MAP_STATIC refusing the DSP read with
//     AEE_ENOSUCHMAP at packet-write time, so the mode is the hazard;
//   * further launches, copies and readbacks add no further mapping call,
//     which is the whole lever: once per allocation, not once per launch;
//   * a free unmaps address-exact, and a fresh allocation afterwards is mapped
//     again rather than riding the freed one's descriptor. rpcmem hands
//     addresses back, so this is the lifecycle test's shape
//     (`npu_megastep_lifecycle_test.cpp:20`) applied to mappings;
//   * a driver whose descriptors refuse fails the allocation by name, never
//     proceeds bare;
//   * MODULAR_NPU_QUEUE, which maps every allocation through its own fd-keyed
//     table and unmaps what it mapped, is refused by name rather than
//     double-mapping.
//
// Every leg runs kRounds times in one process. Both hazards this lever risks
// -- the mapping mode and address reuse -- present intermittently, so a single
// green pass is not evidence; the loop is part of the assertion, not a
// flake-hunting convenience.
//
//===----------------------------------------------------------------------===//

#include "QuRTDeviceContext.h"

#include "NPUDspQueueABI.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
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

constexpr size_t kBufferBytes = 64 << 10;

/// Enough rounds for an intermittent hazard to present: the address-reuse and
/// mapping-mode failures this lever risks pass twice and fail on the third.
constexpr int kRounds = 3;

std::string logPath;

/// The driver calls since the last take, in order, log truncated after.
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

/// The to_fd/mmap pair one allocation makes: the same buffer in both lines,
/// the descriptor the stand-in answered echoed back to it, the allocation's
/// own length, and the mapping mode. Returns the descriptor so a later unmap
/// can be tied to this map without leaning on call order.
std::string expectMapPair(const std::vector<std::string> &lines,
                          const void *buffer, const std::string &when) {
  if (lines.size() != 2) {
    expect(false, when + ": expected a to_fd/mmap pair, got " +
                      std::to_string(lines.size()) + " lines");
    return {};
  }
  std::string p = ptrString(buffer);
  std::string fd;
  {
    std::istringstream first(lines[0]);
    std::string word, ptr, arrow;
    first >> word >> ptr >> arrow >> fd;
    expect(word == "to_fd" && ptr == p,
           when +
               ": first call is not to_fd on the mapped buffer: " + lines[0]);
  }
  std::string wanted = "mmap " + p + " " + fd + " " +
                       std::to_string(kBufferBytes) + " " +
                       std::to_string(NPU_FASTRPC_MAP_FD);
  expect(lines[1] == wanted,
         when + ": expected \"" + wanted + "\", got \"" + lines[1] + "\"");
  return fd;
}

void expectUnmap(const std::vector<std::string> &lines, const void *buffer,
                 const std::string &fd, const std::string &when) {
  std::string wanted = "munmap " + ptrString(buffer) + " " + fd + " " +
                       std::to_string(kBufferBytes) + " 0";
  expect(lines.size() == 1 && lines[0] == wanted,
         when + ": expected exactly \"" + wanted + "\", got " +
             std::to_string(lines.size()) + " lines" +
             (lines.empty() ? "" : " starting \"" + lines[0] + "\""));
}

/// One `vadd` launch naming `base` for all three of its buffer operands, which
/// also exercises the idempotence: three operands, one allocation, one map.
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

/// One buffer's whole life under the flag, driven through the paths a decode
/// uses: stage in, launch, read back, free.
void runRound(DeviceContext &ctx, const DeviceFunction &fn, int round) {
  const std::string when = "round " + std::to_string(round);
  static uint8_t staging[kBufferBytes];
  memset(staging, 0x27, sizeof(staging));

  // The negative control, in this same binary and this same round: with the
  // flag off nothing reaches the mapping seam.
  unsetenv("MODULAR_NPU_MAP_ONCE");
  {
    ErrorOr<DeviceBufferRef> control = ctx.enqueueCreateBuffer(kBufferBytes);
    expect(!control.isError(), when + ": control buffer failed");
    if (control.isError())
      return;
    DeviceBufferRef ref = control.takeValue();
    expect(
        !ctx.enqueueCopyToDevice(ref->ptr(), staging, kBufferBytes).isError(),
        when + ": control copy failed");
    expect(!launchOn(ctx, fn, ref->ptr()).isError(),
           when + ": control dispatch failed");
  }
  expect(takeLog().empty(),
         when + ": the flag is off, but the mapping seam was touched");

  // Anything but exactly "1" is off, so a leg cannot be half-treated by a
  // stray value.
  setenv("MODULAR_NPU_MAP_ONCE", "0", 1);
  {
    ErrorOr<DeviceBufferRef> control = ctx.enqueueCreateBuffer(kBufferBytes);
    expect(!control.isError(), when + ": flag-zero buffer failed");
    if (!control.isError())
      expect(!launchOn(ctx, fn, (*control)->ptr()).isError(),
             when + ": flag-zero dispatch failed");
  }
  expect(takeLog().empty(), when + ": MODULAR_NPU_MAP_ONCE=0 acted as set");

  setenv("MODULAR_NPU_MAP_ONCE", "1", 1);
  ErrorOr<DeviceBufferRef> buffer = ctx.enqueueCreateBuffer(kBufferBytes);
  expect(!buffer.isError(), when + ": mapped buffer failed");
  if (buffer.isError())
    return;
  DeviceBufferRef ref = buffer.takeValue();
  void *base = ref->ptr();

  // ORDER 163's REGRESSION GUARD, and the reason this test exists in this
  // shape. Allocating must NOT map: `fastrpc_mmap` is an RPC, and at allocation
  // time no session is open on the domain, which is what returned AEE_ERPC
  // (512) at every length on silicon. Staging into the buffer must not map
  // either -- a copy opens no session.
  expect(!ctx.enqueueCopyToDevice(base, staging, kBufferBytes).isError(),
         when + ": staging copy failed");
  expect(takeLog().empty(),
         when + ": something mapped before the first launch, which is order "
                "163's AEE_ERPC");

  // The first launch naming the allocation is what maps it, after the session
  // the dispatch path opens.
  expect(!launchOn(ctx, fn, base).isError(), when + ": first dispatch failed");
  std::string fd = expectMapPair(takeLog(), base, when + " first launch");

  // The lever itself: further launches, copies and readbacks over a mapped
  // allocation map nothing. "Once per allocation, not once per launch."
  static uint8_t readback[kBufferBytes];
  expect(!launchOn(ctx, fn, base).isError(), when + ": second dispatch failed");
  expect(!ctx.enqueueCopyToDevice(base, staging, kBufferBytes).isError(),
         when + ": restaging failed");
  expect(!ctx.enqueueCopyFromDevice(readback, base, kBufferBytes).isError(),
         when + ": readback failed");
  expect(takeLog().empty(),
         when + ": using a mapped allocation mapped it again");

  // Freeing unmaps address-exact...
  ref.reset();
  expectUnmap(takeLog(), base, fd, when + " free");

  // ...and the next allocation is mapped afresh on ITS first launch. rpcmem
  // hands addresses back, so this is the leg that catches a mapping outliving
  // its buffer: when the base repeats the descriptor repeats with it, and a
  // stale table entry would make this allocation silently ride the freed one's
  // mapping -- `chunkAlloc` refuses that outright.
  ErrorOr<DeviceBufferRef> reused = ctx.enqueueCreateBuffer(kBufferBytes);
  expect(!reused.isError(), when + ": reallocation failed");
  if (!reused.isError()) {
    DeviceBufferRef fresh = reused.takeValue();
    expect(takeLog().empty(), when + ": the reallocation mapped at alloc time");
    expect(!launchOn(ctx, fn, fresh->ptr()).isError(),
           when + ": reallocation dispatch failed");
    expectMapPair(takeLog(), fresh->ptr(), when + " reallocation first launch");
  }
  (void)takeLog();
  unsetenv("MODULAR_NPU_MAP_ONCE");
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
  // the seam this one reads.
  setenv("MODULAR_NPU_POWER", "0", 1);

  // The SDK's own enum: STATIC 0, RESERVED 1, FD 2 (lane evidence 068). The
  // mode is the hazard, so the constant this transport maps under is checked
  // here too rather than only where the queue asserts it.
  expect(NPU_FASTRPC_MAP_FD == 2,
         "NPU_FASTRPC_MAP_FD is the SDK's FASTRPC_MAP_FD");

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
  (void)takeLog();

  for (int round = 0; round < kRounds; ++round)
    runRound(ctx, **vadd, round);

  // A driver whose descriptors refuse fails the LAUNCH by name -- the mapping
  // now happens there, not at allocation. With the flag set, silently
  // dispatching without the mapping would make an A/B leg indistinguishable
  // from its own control.
  setenv("MODULAR_NPU_MAP_ONCE", "1", 1);
  ErrorOr<DeviceBufferRef> noFd = ctx.enqueueCreateBuffer(kBufferBytes);
  expect(!noFd.isError(), "the no-descriptor leg's buffer failed");
  if (!noFd.isError()) {
    setenv("NPU_FAKE_TRANSPORT_NO_FD", "1", 1);
    ErrorOrSuccess refused = launchOn(ctx, **vadd, (*noFd)->ptr());
    expect(refused.isError(), "a refused descriptor did not fail the launch");
    if (refused.isError())
      expect(std::string(refused.getError()).find("rpcmem_to_fd") !=
                 std::string::npos,
             "the refusal does not name rpcmem_to_fd");
    unsetenv("NPU_FAKE_TRANSPORT_NO_FD");
  }
  (void)takeLog();

  // Two owners of the same mapping is a design question, not a precedence this
  // allocator may quietly pick -- refused the way the transports already
  // refuse each other.
  setenv("MODULAR_NPU_QUEUE", "1", 1);
  ErrorOr<DeviceBufferRef> both = ctx.enqueueCreateBuffer(kBufferBytes);
  expect(both.isError(), "MAP_ONCE with QUEUE was not refused");
  if (both.isError())
    expect(std::string(both.getError()).find("MODULAR_NPU_QUEUE") !=
               std::string::npos,
           "the refusal does not name the other transport");
  unsetenv("MODULAR_NPU_QUEUE");
  unsetenv("MODULAR_NPU_MAP_ONCE");
  (void)takeLog();

  if (failures != 0) {
    printf("%d check(s) failed\n", failures);
    return 1;
  }
  printf("all checks passed\n");
  return 0;
}
