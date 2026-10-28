//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the host-side FastRPC binding.
//
//===----------------------------------------------------------------------===//

#include "QuRTDriver.h"

#include "Support/DLFcn.h"
#include "llvm/ADT/StringRef.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <type_traits>

#ifdef _WIN32
#include <windows.h>
#endif

namespace M::Driver {

namespace {

// The service whose driver package carries the NPU's FastRPC library. An
// identically named library ships with the audio DSP's package and refuses this
// domain, so the path is resolved from the service entry rather than guessed:
// the package directory carries a per-install hash that differs per machine.
constexpr const char *kServiceKey =
    "SYSTEM\\CurrentControlSet\\Services\\qcnspmcdm";
constexpr const char *kLibraryName = "libcdsprpc.dll";

#ifdef _WIN32

/// Reads `ImagePath` for the NPU service and returns the directory holding it.
ErrorOr<std::string> resolveDriverPackageDirectory() {
  char imagePath[MAX_PATH] = {0};
  DWORD byteSize = sizeof(imagePath);
  LSTATUS status = RegGetValueA(HKEY_LOCAL_MACHINE, kServiceKey, "ImagePath",
                                RRF_RT_REG_SZ | RRF_RT_REG_EXPAND_SZ, nullptr,
                                imagePath, &byteSize);
  if (status != ERROR_SUCCESS)
    return Error("no NPU driver service registered: reading ImagePath from "
                 "HKLM\\" +
                 std::string(kServiceKey) + " failed with " +
                 std::to_string(status));

  std::string path(imagePath);
  // The value is a kernel object path, not a Win32 one.
  constexpr llvm::StringLiteral kSystemRoot = "\\SystemRoot\\";
  if (StringRef(path).starts_with_insensitive(kSystemRoot)) {
    char windowsDir[MAX_PATH] = {0};
    if (GetSystemWindowsDirectoryA(windowsDir, sizeof(windowsDir)) == 0)
      return Error("GetSystemWindowsDirectoryA failed with " +
                   std::to_string(GetLastError()));
    path = std::string(windowsDir) + "\\" + path.substr(kSystemRoot.size());
  }

  size_t separator = path.find_last_of("\\/");
  if (separator == std::string::npos)
    return Error("NPU driver ImagePath has no directory component: " + path);
  return path.substr(0, separator);
}

#endif // _WIN32

/// Binds every entry point out of the library at `libraryPath`, which lives in
/// `directory`. Split from `bindDriver` so the OS binding and the test seam
/// resolve their paths differently but bind identically.
ErrorOr<QuRTDriver> bindLibrary(const std::string &libraryPath,
                                std::string directory) {
  QuRTDriver driver;
  driver.libraryDirectory = std::move(directory);

  void *handle = dlopen(libraryPath.c_str(), RTLD_NOW);
  if (handle == nullptr)
    return Error("failed to load the NPU FastRPC library " + libraryPath +
                 ": " + dlerror());

  // Every entry point is required: a partial binding would fail later at a
  // call site that cannot say what was missing.
  const char *missing = nullptr;
  auto bind = [&](auto &slot, const char *symbol) {
    void *address = dlsym(handle, symbol);
    if (address == nullptr) {
      if (missing == nullptr)
        missing = symbol;
      return;
    }
    slot = reinterpret_cast<std::decay_t<decltype(slot)>>(address);
  };

  // Bound but not required, and recorded rather than refused: only the queue
  // transport calls these, and refusing the whole library for their absence
  // would take the synchronous path down with them.
  auto bindQueue = [&](auto &slot, const char *symbol) {
    void *address = dlsym(handle, symbol);
    if (address == nullptr) {
      if (driver.missingQueueSymbol == nullptr)
        driver.missingQueueSymbol = symbol;
      return;
    }
    slot = reinterpret_cast<std::decay_t<decltype(slot)>>(address);
  };

  bind(driver.rpcmemInit, "rpcmem_init");
  bind(driver.rpcmemAlloc, "rpcmem_alloc");
  bind(driver.rpcmemFree, "rpcmem_free");
  bind(driver.sessionControl, "remote_session_control");
  bind(driver.handleOpen, "remote_handle64_open");
  bind(driver.handleInvoke, "remote_handle64_invoke");
  bind(driver.handleClose, "remote_handle64_close");

  // Optional, and deliberately bound after `missing` would have been set by a
  // required symbol: a library without these is still a working transport.
  auto bindOptional = [&](auto &slot, const char *symbol) {
    if (void *address = dlsym(handle, symbol))
      slot = reinterpret_cast<std::decay_t<decltype(slot)>>(address);
  };
  bindOptional(driver.rpcmemToFD, "rpcmem_to_fd");
  bindOptional(driver.registerBufAttr, "remote_register_buf_attr2");
  bindOptional(driver.handleControl, "remote_handle64_control");

  if (missing != nullptr) {
    dlclose(handle);
    return Error("the NPU FastRPC library " + libraryPath + " exports no " +
                 missing);
  }

  bindQueue(driver.rpcmemToFD, "rpcmem_to_fd");
  bindQueue(driver.fastrpcMmap, "fastrpc_mmap");
  bindQueue(driver.fastrpcMunmap, "fastrpc_munmap");
  bindQueue(driver.dspqueueCreate, "dspqueue_create");
  bindQueue(driver.dspqueueExport, "dspqueue_export");
  bindQueue(driver.dspqueueClose, "dspqueue_close");
  bindQueue(driver.dspqueueWrite, "dspqueue_write");
  bindQueue(driver.dspqueueReadNoblock, "dspqueue_read_noblock");

  driver.rpcmemInit();
  return driver;
}

ErrorOr<QuRTDriver> bindDriver() {
  // Test seam: an explicitly named library replaces the OS's transport, so the
  // host half of the dispatch path -- packing, offsets, recording -- can run
  // where no NPU exists. A stand-in cannot execute kernels; nothing that
  // reaches a device honours this.
  if (const char *substitute = getenv("MODULAR_NPU_TRANSPORT_LIB");
      substitute != nullptr && *substitute != '\0') {
    std::string path(substitute);
    size_t separator = path.find_last_of("\\/");
    return bindLibrary(path, separator == std::string::npos
                                 ? std::string(".")
                                 : path.substr(0, separator));
  }

#ifndef _WIN32
  // The on-device NPU is reached through the OS's own FastRPC library. Nothing
  // stops a POSIX port beyond a second name-resolution path, but the milestone
  // this backend exists for is Windows on ARM64.
  //
  // TODO(fabio): a POSIX host cannot reach the on-device NPU at all; give
  // `resolveDriverPackageDirectory` a POSIX arm once a Linux host needs it.
  return Error("the on-device NPU backend requires Windows");
#else
  ErrorOr<std::string> directory = resolveDriverPackageDirectory();
  if (directory.isError())
    return directory.takeError();
  return bindLibrary(*directory + "\\" + kLibraryName, *directory);
#endif // _WIN32
}

} // namespace

std::string qurtStatusMessage(uint32_t status) {
  // Vendor statuses are an offset plus an index into its error enum. Only the
  // ones this transport can actually produce are named; anything else keeps its
  // raw value rather than being mislabelled.
  const char *name = nullptr;
  switch (status) {
  case 0x00000000:
    name = "SUCCESS";
    break;
  case 0x80000406:
    name = "AEE_EUNABLETOLOAD";
    break;
  case 0x8000040E:
    name = "AEE_EBADPARM";
    break;
  case 0x80000427:
    name = "AEE_ENOSUCH";
    break;
  case 0x80000442:
    name = "AEE_ENOSUCHSYMBOL";
    break;
  case 0x80000472:
    name = "AEE_ECONNREFUSED";
    break;
  case 0x80000473:
    name = "AEE_ENOSESSION";
    break;
  default:
    break;
  }

  char raw[16] = {0};
  snprintf(raw, sizeof(raw), "0x%08x", status);
  return name == nullptr ? std::string(raw)
                         : std::string(name) + " (" + raw + ")";
}

ErrorOr<QuRTDriver &> QuRTDriver::get() {
  static std::mutex mu;
  static std::optional<QuRTDriver> driver;
  static std::string bindError;

  std::lock_guard<std::mutex> lock(mu);
  if (!driver.has_value() && bindError.empty()) {
    ErrorOr<QuRTDriver> bound = bindDriver();
    if (bound.isError())
      bindError = bound.getError();
    else
      driver = bound.takeValue();
  }
  if (!driver.has_value())
    return Error(bindError);
  return *driver;
}

} // namespace M::Driver
