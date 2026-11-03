//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// The weights-NOFLUSH classification, driven through the real copy paths:
// `QuRTDevice::create` binds the stand-in transport through the
// MODULAR_NPU_TRANSPORT_LIB seam, the loader's seam declares a constant host
// range, and every claim below is judged by the buffer-attribute calls that
// actually crossed the driver seam, appended by the stand-in to a log file.
//
// The properties that matter, in the order a device leg would meet them: the
// flag off means the seam is never touched (the control leg must be today's
// driver behaviour, bit for bit); a whole-allocation copy from declared memory
// applies FASTRPC_ATTR_FORCE_NOFLUSH (32) exactly once; any later
// runtime-mediated write that is not that -- a partial copy, a fill, an
// undeclared source -- withdraws it (0) before the write lands, which is the
// revocation the alloc-reuse pool leans on; a free deregisters (fd -1) so no
// attribute outlives its allocation onto a recycled address; and a driver
// whose descriptors refuse must fail the copy by name, never proceed bare --
// a leg that silently dropped its treatment is indistinguishable from its own
// control.
//
//===----------------------------------------------------------------------===//

#include "QuRTDeviceContext.h"
#include "npu_constant_ranges.h"

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

constexpr size_t kWeightBytes = 64 << 10;

std::string logPath;

/// The attribute calls since the last take, in order, log truncated after.
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

/// Checks one to_fd/attr2 pair: the same buffer in both lines, the fd the
/// stand-in answered echoed back to it, and the attribute value asked for.
void expectPair(const std::vector<std::string> &lines, const void *buffer,
                int attr, const std::string &when) {
  if (lines.size() != 2) {
    expect(false, when + ": expected a to_fd/attr2 pair, got " +
                      std::to_string(lines.size()) + " lines");
    return;
  }
  std::string p = ptrString(buffer);
  std::string fd;
  {
    std::istringstream first(lines[0]);
    std::string word, ptr, arrow;
    first >> word >> ptr >> arrow >> fd;
    expect(word == "to_fd" && ptr == p, when +
                                            ": first call is not to_fd on "
                                            "the classified buffer: " +
                                            lines[0]);
  }
  std::string wanted = "attr2 " + p + " " + std::to_string(kWeightBytes) + " " +
                       fd + " " + std::to_string(attr);
  expect(lines[1] == wanted,
         when + ": expected \"" + wanted + "\", got \"" + lines[1] + "\"");
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
  logPath = std::string(tmp) + "/attr_calls.log";
  setenv("NPU_FAKE_TRANSPORT_ATTR_LOG", logPath.c_str(), 1);

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

  // The loader's half of the contract: one blob of weights, declared; another
  // buffer of the same size and shape that nobody declared.
  static uint8_t weightsBlob[kWeightBytes];
  static uint8_t undeclaredBlob[kWeightBytes];
  memset(weightsBlob, 0x5A, sizeof(weightsBlob));
  MODULAR_NPUDeclareConstantHostRange(weightsBlob, sizeof(weightsBlob));

  ErrorOr<DeviceBufferRef> weights = ctx.enqueueCreateBuffer(kWeightBytes);
  if (weights.isError()) {
    printf("FAIL: buffer: %s\n", weights.getError());
    return 1;
  }
  void *w = (*weights)->ptr();
  (void)takeLog();

  // Flag off: the seam is never touched, whatever the copy looks like.
  unsetenv("MODULAR_NPU_WEIGHTS_NOFLUSH");
  expect(!ctx.enqueueCopyToDevice(w, weightsBlob, kWeightBytes).isError(),
         "flag-off copy failed");
  expect(takeLog().empty(), "flag off, but the attribute seam was touched");

  // Anything but exactly "1" is off.
  setenv("MODULAR_NPU_WEIGHTS_NOFLUSH", "0", 1);
  expect(!ctx.enqueueCopyToDevice(w, weightsBlob, kWeightBytes).isError(),
         "flag-zero copy failed");
  expect(takeLog().empty(), "MODULAR_NPU_WEIGHTS_NOFLUSH=0 acted as set");

  // A whole-allocation copy from declared memory is init-written: apply.
  setenv("MODULAR_NPU_WEIGHTS_NOFLUSH", "1", 1);
  expect(!ctx.enqueueCopyToDevice(w, weightsBlob, kWeightBytes).isError(),
         "qualifying copy failed");
  expectPair(takeLog(), w, 32, "qualifying whole copy");

  // The same qualifying copy again changes nothing: no churn on the seam.
  expect(!ctx.enqueueCopyToDevice(w, weightsBlob, kWeightBytes).isError(),
         "repeated qualifying copy failed");
  expect(takeLog().empty(), "a repeated qualifying copy re-registered");

  // A partial host write revokes before the write lands.
  expect(!ctx.enqueueCopyToDevice(w, weightsBlob, 1024).isError(),
         "partial copy failed");
  expectPair(takeLog(), w, 0, "partial copy over an attributed buffer");

  // An interior write to an unattributed buffer stays off the seam: the
  // steady-state dispatch traffic must not pay a driver call per copy.
  expect(!ctx.enqueueCopyToDevice(static_cast<uint8_t *>(w) + 512, weightsBlob,
                                  1024)
              .isError(),
         "interior copy failed");
  expect(takeLog().empty(), "an interior copy touched the attribute seam");

  // Re-qualify, then a fill revokes the same way a partial copy does.
  expect(!ctx.enqueueCopyToDevice(w, weightsBlob, kWeightBytes).isError(),
         "re-qualifying copy failed");
  expectPair(takeLog(), w, 32, "re-qualifying whole copy");
  expect(!ctx.enqueueSetMemory(w, kWeightBytes, 0, 8).isError(), "fill failed");
  expectPair(takeLog(), w, 0, "fill over an attributed buffer");

  // A whole-allocation copy from memory nobody declared never qualifies --
  // size and shape do not make a weight; provenance does.
  expect(!ctx.enqueueCopyToDevice(w, undeclaredBlob, kWeightBytes).isError(),
         "undeclared-source copy failed");
  expect(takeLog().empty(), "an undeclared source was classified as constant");

  // ... and arriving over an attributed buffer, it revokes.
  expect(!ctx.enqueueCopyToDevice(w, weightsBlob, kWeightBytes).isError(),
         "qualify-before-revoke copy failed");
  expectPair(takeLog(), w, 32, "qualify before undeclared revoke");
  expect(!ctx.enqueueCopyToDevice(w, undeclaredBlob, kWeightBytes).isError(),
         "undeclared revoke copy failed");
  expectPair(takeLog(), w, 0, "undeclared source over an attributed buffer");

  // A copy out of the device only reads device memory; the host-side
  // destination is outside every allocation and classifies nothing.
  static uint8_t hostOut[kWeightBytes];
  expect(!ctx.enqueueCopyFromDevice(hostOut, w, kWeightBytes).isError(),
         "copy from device failed");
  expect(takeLog().empty(), "a copy to a host buffer touched the seam");

  // Freeing an attributed buffer deregisters it (fd -1): no attribute may
  // outlive its allocation onto whatever lands at that address next.
  {
    ErrorOr<DeviceBufferRef> scoped = ctx.enqueueCreateBuffer(kWeightBytes);
    expect(!scoped.isError(), "scoped buffer failed");
    void *s = nullptr;
    if (!scoped.isError()) {
      DeviceBufferRef ref = scoped.takeValue();
      s = ref->ptr();
      expect(!ctx.enqueueCopyToDevice(s, weightsBlob, kWeightBytes).isError(),
             "scoped qualifying copy failed");
      expectPair(takeLog(), s, 32, "scoped qualifying copy");
    }
    if (s != nullptr) {
      std::vector<std::string> lines = takeLog();
      std::string wanted = "attr2 " + ptrString(s) + " " +
                           std::to_string(kWeightBytes) + " -1 0";
      expect(lines.size() == 1 && lines[0] == wanted,
             "free of an attributed buffer did not deregister: expected \"" +
                 wanted + "\"");
    }
  }

  // A driver whose descriptors refuse fails the copy by name: with the flag
  // set, silently proceeding without the attribute would make an A/B leg
  // indistinguishable from its own control.
  ErrorOr<DeviceBufferRef> second = ctx.enqueueCreateBuffer(kWeightBytes);
  expect(!second.isError(), "second buffer failed");
  if (!second.isError()) {
    void *s = (*second)->ptr();
    setenv("NPU_FAKE_TRANSPORT_NO_FD", "1", 1);
    ErrorOrSuccess refused =
        ctx.enqueueCopyToDevice(s, weightsBlob, kWeightBytes);
    expect(refused.isError(), "a refused descriptor did not fail the copy");
    if (refused.isError())
      expect(std::string(refused.getError()).find("rpcmem_to_fd") !=
                 std::string::npos,
             "the refusal does not name rpcmem_to_fd");
    unsetenv("NPU_FAKE_TRANSPORT_NO_FD");
    (void)takeLog();
  }

  unsetenv("MODULAR_NPU_WEIGHTS_NOFLUSH");
  if (failures == 0)
    printf("PASS: npu_weights_noflush_test\n");
  return failures == 0 ? 0 : 1;
}
