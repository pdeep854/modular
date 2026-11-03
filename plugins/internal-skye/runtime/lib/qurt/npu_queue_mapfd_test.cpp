//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// The queue channel's buffer mapping, held to the mode silicon accepts.
//
// Order 081 localized the queue transport's residual to one call: with every
// buffer fastrpc_mmap'd FASTRPC_MAP_STATIC, the DSP's read refuses the very
// first packet with AEE_ENOSUCHMAP -- the vendor's buffer bookkeeping is
// keyed by file descriptor and a STATIC mapping never lands in the table the
// queue resolves from. The stand-in transport now models that bookkeeping
// (fd-keyed, populated only by FASTRPC_MAP_FD), so the mapping mode is a
// host-visible property instead of a device-only one:
//
//   * the queue leg dispatches through the real channel against the fake and
//     must complete -- which it can only do if every descriptor's fd was
//     mapped with the mode the bookkeeping accepts;
//   * the static-world leg is the control that can fail: the fake records
//     mappings the way 081's firmware treated STATIC ones (absent), and the
//     dispatch must be REFUSED with ENOSUCHMAP named -- the exact silicon
//     presentation, reproduced where a test can see it.
//
// Legs run in forked children because the transport library and the queue
// flag are process-level bindings.
//
//===----------------------------------------------------------------------===//

#include "QuRTDeviceContext.h"

#include "NPUDspQueueABI.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

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

int childFail(const char *stage, const char *error) {
  printf("child %s: %s\n", stage, error);
  return 1;
}

/// Two dispatches with a drain between them: the second proves the completion
/// came back and the ring slot was recycled, not just that a frame left.
int runQueueChild() {
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

  for (int round = 0; round < 2; ++round) {
    ErrorOrSuccess result = ctx.enqueueFunctionExecDirect(
        **vadd, /*gridX=*/1, /*gridY=*/1, /*gridZ=*/1, /*blockX=*/1,
        /*blockY=*/1, /*blockZ=*/1, /*sharedMemBytes=*/0, /*attrs=*/nullptr,
        /*numAttrs=*/0, args, /*argCount=*/4, sizes);
    if (result.isError())
      return childFail("dispatch", result.getError());
    ErrorOrSuccess sync = ctx.synchronize();
    if (sync.isError())
      return childFail("synchronize", sync.getError());
  }
  return 0;
}

/// Exits 0 only when the dispatch is refused AND the refusal names the
/// mapping class -- a static-world leg that succeeded, or failed for any
/// other reason, is a test that stopped watching the layer 081 paid for.
int runStaticWorldChild() {
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
  ErrorOrSuccess sync =
      result.isError() ? std::move(result) : ctx.synchronize();
  if (!sync.isError())
    return childFail("static-world",
                     "a queue dispatch with unresolvable mappings succeeded");
  std::string error(sync.getError());
  if (error.find("ENOSUCHMAP") == std::string::npos)
    return childFail(
        "static-world",
        ("the refusal does not name the mapping class: " + error).c_str());
  return 0;
}

int spawnChild(const std::string &self, const std::string &fakeLib,
               const char *mode, bool staticWorld) {
  pid_t pid = fork();
  if (pid < 0)
    return -1;
  if (pid == 0) {
    setenv("MODULAR_NPU_TRANSPORT_LIB", fakeLib.c_str(), 1);
    setenv("MODULAR_NPU_QUEUE", "1", 1);
    // One variable: the vote has its own test, and the recorder and timing
    // instruments are refused on the queue path by design.
    setenv("MODULAR_NPU_POWER", "0", 1);
    if (staticWorld)
      setenv("NPU_FAKE_TRANSPORT_STATIC_WORLD", "1", 1);
    execl(self.c_str(), self.c_str(), mode, static_cast<char *>(nullptr));
    _exit(127);
  }
  int status = 0;
  if (waitpid(pid, &status, 0) != pid)
    return -1;
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

} // namespace

int main(int argc, char **argv) {
  if (argc > 1 && strcmp(argv[1], "--child-queue") == 0)
    return runQueueChild();
  if (argc > 1 && strcmp(argv[1], "--child-static-world") == 0)
    return runStaticWorldChild();

  // The SDK's own enum: STATIC 0, RESERVED 1, FD 2 (lane evidence 068). A
  // restated constant that drifted would map every buffer in a mode the
  // bookkeeping ignores -- 081's failure, reintroduced silently.
  expect(NPU_FASTRPC_MAP_FD == 2,
         "NPU_FASTRPC_MAP_FD is the SDK's FASTRPC_MAP_FD");

  const char *fakeLib = getenv("NPU_FAKE_TRANSPORT");
  if (fakeLib == nullptr || *fakeLib == '\0') {
    printf("FAIL: NPU_FAKE_TRANSPORT is not set\n");
    return 1;
  }

  expect(spawnChild(argv[0], fakeLib, "--child-queue",
                    /*staticWorld=*/false) == 0,
         "the queue leg completes against fd-keyed bookkeeping");
  expect(spawnChild(argv[0], fakeLib, "--child-static-world",
                    /*staticWorld=*/true) == 0,
         "the static-world leg is refused with ENOSUCHMAP named");

  if (failures != 0) {
    printf("%d check(s) failed\n", failures);
    return 1;
  }
  printf("all checks passed\n");
  return 0;
}
