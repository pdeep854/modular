//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Host-side binding for the FastRPC transport that reaches the on-device NPU.
//
// The vendor library is a flat C API loaded with dlopen/dlsym, so nothing here
// links against the device SDK and no C++ ABI crosses into the vendor's
// Microsoft-built DLL. That is also why the two structs and the scalar-packing
// macro below are restated rather than included: the SDK has no Linux-aarch64
// build, so its headers are not available to this cross compile.
//
//===----------------------------------------------------------------------===//

#ifndef SKYE_RUNTIME_QURT_QURTDRIVER_H
#define SKYE_RUNTIME_QURT_QURTDRIVER_H

#include "NPUDspQueueABI.h"
#include "Support/ErrorOr.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace M::Driver {

/// One argument slot of a FastRPC invocation: the vendor's `remote_buf`.
struct QuRTRemoteBuf {
  void *pv;
  size_t nLen;
};

/// Payload of the unsigned-protection-domain session request.
struct QuRTUnsignedModule {
  int domain;
  int enable;
};

/// The compute DSP domain the NPU lives on.
constexpr uint32_t kQuRTComputeDomainID = 3;

/// `remote_session_control` request selecting unsigned-module support.
constexpr uint32_t kQuRTControlUnsignedModule = 2;

/// rpcmem heap and flags for plain host-visible shared memory.
constexpr int kQuRTHeapIDSystem = 25;
constexpr uint32_t kQuRTRpcmemDefaultFlags = 1;

/// Buffer attributes for `remote_register_buf_attr2`, restated from the
/// vendor's `remote.h` for the same reason `remote_buf` is. Only the two the
/// transport can set are named: `NONE` restores the driver's own cache
/// maintenance, `FORCE_NOFLUSH` makes the driver skip flushing that buffer and
/// hands the obligation to the client.
constexpr int kQuRTAttrNone = 0;
constexpr int kQuRTAttrForceNoFlush = 32;

/// `remote_handle64_control` request keeping the RPC path out of deep
/// low-power modes for this handle's session -- `DSPRPC_CONTROL_LATENCY` and
/// its payload, restated from the vendor's `remote.h`. `enable = 1` selects
/// the PM-QoS mode, which is the mode the shipped ggml-hexagon backend votes
/// right after its session open; `latency` is that mode's target in
/// microseconds and only the adaptive mode reads it as more than a hint.
constexpr uint32_t kQuRTControlLatency = 1;
struct QuRTRemoteLatency {
  uint32_t enable;
  uint32_t latency;
};

/// Packs an invocation descriptor, mirroring the vendor's
/// `REMOTE_SCALARS_MAKEX`.
constexpr uint32_t qurtScalars(uint32_t method, uint32_t inBufs,
                               uint32_t outBufs) {
  return (method & 0x1F) << 24 | (inBufs & 0xFF) << 16 | (outBufs & 0xFF) << 8;
}

/// Renders a vendor status as a symbolic name plus its raw value, so a failure
/// reads as a diagnosis rather than a hex number. Guessing at these wastes
/// hours; the offsets come from the SDK's error header.
std::string qurtStatusMessage(uint32_t status);

/// The dynamically bound FastRPC entry points, resolved once per process.
///
/// Deliberately the smallest set the transport needs. Session lifetime
/// (`init`, `sessionControl`) is separated from skel lifetime
/// (`handleOpen`/`handleClose`) because only the latter requires a signed
/// catalogue, so buffers and copies stay usable when signing is stale.
struct QuRTDriver {
  /// Loads the vendor library and binds its entry points. Idempotent; the
  /// first call's outcome is cached, including its failure.
  static ErrorOr<QuRTDriver &> get();

  /// Directory holding the vendor library, which is also where the loader
  /// looks for a skel named by a bare `file:///` URI.
  std::string libraryDirectory;

  void (*rpcmemInit)() = nullptr;
  void *(*rpcmemAlloc)(int heapID, uint32_t flags, int byteSize) = nullptr;
  void (*rpcmemFree)(void *ptr) = nullptr;
  int (*sessionControl)(uint32_t requestID, void *data,
                        uint32_t dataLen) = nullptr;
  int (*handleOpen)(const char *uri, uint64_t *handle) = nullptr;
  int (*handleInvoke)(uint64_t handle, uint32_t scalars,
                      QuRTRemoteBuf *args) = nullptr;
  int (*handleClose)(uint64_t handle) = nullptr;

  /// Optional: the buffer-attribute pair. The vendor documents both as weak
  /// symbols absent from some library versions, so a null here is a fact about
  /// this machine's driver rather than a binding failure — every caller is
  /// opt-in and reports the absence by name instead of proceeding without it.
  int (*rpcmemToFD)(void *ptr) = nullptr;
  void (*registerBufAttr)(void *buf, size_t byteSize, int fd,
                          int attr) = nullptr;

  /// Optional, for the same reason: only the `MODULAR_NPU_POWER` latency
  /// vote (on by default) calls it, and a library without it still serves
  /// every transport. A null becomes a named report at the vote, never a
  /// crash.
  int (*handleControl)(uint64_t handle, uint32_t req, void *data,
                       uint32_t dataLen) = nullptr;

  /// The queue transport's entry points, and the two mapping calls it needs.
  ///
  /// Optional, unlike everything above: a driver that exports none of them
  /// still serves the synchronous transport perfectly, and that is the path an
  /// unset `MODULAR_NPU_QUEUE` takes. A null here becomes a named refusal from
  /// `createNPUDspQueueChannel`, which is where the caller asked for the queue
  /// by name and so is where a missing symbol is a diagnosis rather than a
  /// crash.
  int (*fastrpcMmap)(int domain, int fd, void *addr, int offset, size_t length,
                     int flags) = nullptr;
  int (*fastrpcMunmap)(int domain, int fd, void *addr, size_t length) = nullptr;
  int (*dspqueueCreate)(int domain, uint32_t flags, uint32_t requestQueueBytes,
                        uint32_t responseQueueBytes,
                        void (*packetCallback)(void *, int, void *),
                        void (*errorCallback)(void *, int, void *),
                        void *callbackContext, void **queue) = nullptr;
  int (*dspqueueExport)(void *queue, uint64_t *queueID) = nullptr;
  int (*dspqueueClose)(void *queue) = nullptr;
  int (*dspqueueWrite)(void *queue, uint32_t flags, uint32_t bufferCount,
                       npu_dspqueue_buffer_t *buffers, uint32_t messageBytes,
                       const uint8_t *message,
                       uint32_t timeoutMicros) = nullptr;
  int (*dspqueueReadNoblock)(void *queue, uint32_t *flags, uint32_t maxBuffers,
                             uint32_t *bufferCount,
                             npu_dspqueue_buffer_t *buffers,
                             uint32_t maxMessageBytes, uint32_t *messageBytes,
                             uint8_t *message) = nullptr;

  /// Names the first queue entry point this driver does not export, or null
  /// when it exports all of them.
  const char *missingQueueSymbol = nullptr;
};

} // namespace M::Driver

#endif // SKYE_RUNTIME_QURT_QURTDRIVER_H
