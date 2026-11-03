//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Where a stale skel becomes visible, and where it does not.
//
// `loadFunction` resolves a kernel name against `kNPUKernels`, the table
// compiled into THIS process beside its own `NPUKernels.def`. The signed skel's
// table is a different table on the far side of FastRPC, and nothing on the
// wire carries it. So a driver built with an appended row always loads that row
// successfully, whatever the device holds, and a caller that reads "the load
// succeeded" as "the kernel resolved" has read the host's manifest back to
// itself -- the defect WS4 found behind the VAE decoder's F4 fallback, where
// readiness keyed on the load could never be false and the host branches were
// unreachable.
//
// The first moment the asymmetry is observable is the INVOKE, which the device
// refuses at its argument check. Three legs, one variable each:
//
//   * the load leg: an appended row loads fine against a stand-in that does not
//     carry it, and a name no manifest carries is refused -- so the load check
//     is real, and it is a check on the HOST's table;
//   * the refusal leg: the same function's dispatch fails when the stand-in
//     refuses that selector, which is what a driver may key readiness on;
//   * the control: with the refusal cleared and nothing else changed, that same
//     dispatch succeeds. Without it, "the invoke failed" would be evidence
//     about this test's argument packing rather than about the selector.
//
// MODEL, NOT MEASUREMENT (order 162's caveat): the stand-in reproduces the
// device's refusal shape -- kBadParm from the entry point, before any write --
// from `mojo_npu_imp.c`. It is not silicon, and the silicon presentation it
// restates is order 164's R7.
//
//===----------------------------------------------------------------------===//

#include "QuRTDeviceContext.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

/// `NPUKernels.def:527` -- appended last, so it is the row a skel signed for
/// any earlier window lacks.
constexpr uint32_t kTransposeBiasSelector = 41;

constexpr size_t kArenaBytes = 1 << 16;

/// The operand list `DeviceEngine::transposeBias` sends: output, input, bias,
/// witness, then `R` and `N`. Restated here rather than shared, so an arity
/// that stopped agreeing with the manifest breaks this test too.
ErrorOrSuccess dispatchTransposeBias(DeviceContext &ctx,
                                     const DeviceFunction &fn, uint8_t *arena) {
  void *out = arena;
  void *in = arena + 16384;
  void *bias = arena + 32768;
  void *witness = arena + 49152;
  int32_t r = 8, n = 8;
  void *args[6] = {&out, &in, &bias, &witness, &r, &n};
  size_t sizes[6] = {sizeof(void *), sizeof(void *), sizeof(void *),
                     sizeof(void *), sizeof(r),      sizeof(n)};
  ErrorOrSuccess result = ctx.enqueueFunctionExecDirect(
      fn, /*gridX=*/1, /*gridY=*/1, /*gridZ=*/1, /*blockX=*/1, /*blockY=*/1,
      /*blockZ=*/1, /*sharedMemBytes=*/0, /*attrs=*/nullptr, /*numAttrs=*/0,
      args, /*argCount=*/6, sizes);
  if (result.isError())
    return result;
  return ctx.synchronize();
}

} // namespace

int main() {
  const char *fakeLib = getenv("NPU_FAKE_TRANSPORT");
  if (fakeLib == nullptr || *fakeLib == '\0') {
    printf("FAIL: NPU_FAKE_TRANSPORT is not set\n");
    return 1;
  }
  setenv("MODULAR_NPU_TRANSPORT_LIB", fakeLib, 1);
  // One variable per leg: the vote has its own test, and an unvoted session
  // would add a second reason a dispatch could be refused.
  setenv("MODULAR_NPU_POWER", "0", 1);
  setenv("NPU_FAKE_TRANSPORT_NO_SELECTOR",
         std::to_string(kTransposeBiasSelector).c_str(), 1);

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

  ErrorOr<DeviceBufferRef> arena = ctx.enqueueCreateBuffer(kArenaBytes);
  if (arena.isError()) {
    printf("FAIL: arena: %s\n", arena.getError());
    return 1;
  }
  auto *bytes = static_cast<uint8_t *>((*arena)->ptr());

  ErrorOr<DeviceFunctionRef> loaded =
      ctx.loadFunction("", "transpose_bias", "");
  expect(!loaded.isError(),
         "loadFunction resolves an appended row against a device that refuses "
         "its selector -- the load cannot see the skel");
  if (loaded.isError()) {
    printf("  %s\n", loaded.getError());
    return 1;
  }

  ErrorOr<DeviceFunctionRef> absent =
      ctx.loadFunction("", "no_such_kernel", "");
  expect(absent.isError(),
         "loadFunction still refuses a name no manifest carries -- its check "
         "is real, and it is a check on this process's kNPUKernels");

  ErrorOrSuccess refused = dispatchTransposeBias(ctx, **loaded, bytes);
  expect(refused.isError(),
         "the invoke is where a skel without selector 41 is observable");
  if (refused.isError())
    printf("  refusal: %s\n", refused.getError());

  // The VAE decoder's latch tells a missing row from a transport fault by
  // this spelling (`DeviceEngine::transposeBias`), because the driver is
  // handed an error string and not a status. `qurtStatusMessage` renaming
  // 0x8000040E would leave that comparison silently never matching -- a
  // permanent no-latch -- so the name is locked here, at the seam that
  // produces it.
  expect(refused.isError() &&
             std::string(refused.getError()).find("AEE_EBADPARM") !=
                 std::string::npos,
         "the refusal names AEE_EBADPARM, which is what the VAE decoder's "
         "latch matches on");

  unsetenv("NPU_FAKE_TRANSPORT_NO_SELECTOR");
  ErrorOrSuccess served = dispatchTransposeBias(ctx, **loaded, bytes);
  expect(!served.isError(),
         "control: the same dispatch, with only the refusal cleared, succeeds");
  if (served.isError())
    printf("  %s\n", served.getError());

  if (failures != 0) {
    printf("%d check(s) failed\n", failures);
    return 1;
  }
  printf("all checks passed\n");
  return 0;
}
