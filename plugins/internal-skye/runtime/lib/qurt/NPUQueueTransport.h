//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Carrying launches to the NPU over a shared-memory queue instead of one
// blocking FastRPC invocation each.
//
// Why a second transport exists at all: the synchronous path costs a fixed
// 1.21 ms per launch and, more importantly, serialises host prep against device
// compute. On the VAE stage that serialisation is 48.6 s of a 119.0 s window --
// larger than the whole transport term -- and no amount of batching reaches it,
// because a batch still prepares every launch before it blocks on all of them.
// A queue lets the host pack launch n+1 while the DSP runs launch n, which
// turns the stage's `prep + compute` into `max(prep, compute)`. See
// `.agentwork/flux-npu/BATCHING_ROUTE.md` section 5.
//
// Layering, and the reason for it: everything with a *rule* in it -- the frame
// layout, the fit-or-flush discipline, the union of allocations a frame
// declares, the in-flight ring, the flush-and-wait at a graph boundary -- lives
// here and is driven through an abstract channel, so all of it is exercised on
// a host with no NPU. The vendor ABI is confined to the one channel that speaks
// it, which is compiled only where the SDK's own `dspqueue.h` is on the include
// path. Nothing in this file restates a vendor struct from memory: the queue
// hands the DLL a buffer descriptor it dereferences, so a layout that drifted
// would be a wrong pointer rather than a failed call.
//
//===----------------------------------------------------------------------===//

#ifndef SKYE_RUNTIME_QURT_NPUQUEUETRANSPORT_H
#define SKYE_RUNTIME_QURT_NPUQUEUETRANSPORT_H

#include "NPUDspQueueABI.h"
#include "NPUQueueWire.h"
#include "Support/ErrorOr.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace M::Driver {

/// Renders a host-side AEE status, for the FastRPC calls whose return values
/// use the host's numbering rather than the DSP's -- `dspqueue_*` and
/// `fastrpc_mmap`. Shared with the synchronous path's map-once arm, which
/// reaches the same `fastrpc_mmap` return list: order 163 spent an
/// adjudication round on a bare "512" this would have named.
std::string dspQueueStatusMessage(int status);

/// One allocation a frame's launches operate on, as the host knows it: the base
/// FastRPC would map and the length the synchronous path declares in the entry
/// point's primary-input block.
struct NPUQueueBufferRef {
  void *base = nullptr;
  size_t byteSize = 0;
  /// True when the allocation was classified init-constant -- written whole,
  /// once, from memory the loader declared constant (the same provenance test
  /// the weights-NOFLUSH probe applies). The descriptor for such a reference
  /// carries no cache-maintenance flags: the host wrote it before the first
  /// frame and never again, so per-frame maintenance on it is O(model) work
  /// for bytes that cannot be stale. The shipped practice this follows is
  /// ggml-hexagon, which never puts a weight in a descriptor's maintenance
  /// path at all (`GGML_HEXAGON_LESSONS.md` section 3).
  bool constant = false;
};

/// One launch, in the shape the packer needs and no other. Deliberately the
/// same tuple `QuRTDeviceContext::enqueueFunctionExecDirect` builds for the
/// synchronous path -- the scalar block arrives already packed, offsets in its
/// tail -- so the two transports cannot diverge in what a launch *is*.
struct NPUQueueLaunch {
  uint32_t selector = 0;
  /// The entry point's second argument verbatim: participant mask in the low
  /// byte, instrumentation bits above it.
  uint32_t launchWord = 0;
  const uint8_t *scalars = nullptr;
  size_t scalarBytes = 0;
  const NPUQueueBufferRef *buffers = nullptr;
  size_t bufferCount = 0;
};

/// Where a packed frame goes. Abstract for the reason in the file comment, and
/// narrow on purpose: a transport that can only send a frame and collect a
/// response is one whose ordering is a property of the queue rather than of the
/// caller.
class NPUQueueChannel {
public:
  virtual ~NPUQueueChannel();

  /// Sends one frame, with the union of its launches' allocations attached as
  /// buffer references. The references are what carry the mapping and the cache
  /// maintenance the FastRPC driver used to do per call; attaching the union is
  /// what makes a frame declare the same bytes the individual launches did.
  virtual ErrorOrSuccess send(const void *frame, size_t frameBytes,
                              const NPUQueueBufferRef *refs,
                              size_t refCount) = 0;

  /// Collects one completed frame's id and status. Reports `false` when the
  /// timeout passed with nothing to collect, which is not an error: a caller
  /// polling for room in the ring and a caller draining at a graph boundary
  /// want the same call and differ only in what they do with a miss.
  virtual ErrorOr<bool> receive(uint32_t &batchID, int32_t &status,
                                uint32_t timeoutMicros) = 0;

  /// Fetches the drain's post-mortem report over the synchronous transport
  /// and renders it to stderr, when the channel has a device to ask. Called
  /// by the transport at the moment it refuses -- a receive that gave up, or
  /// a drain refusal -- because that is exactly when the queue itself can no
  /// longer carry the answer. Best-effort by contract: it must not replace
  /// the refusal it accompanies, and a channel with no device (the tests'
  /// loopback) does nothing.
  virtual void dumpReport() {}
};

/// Lays out one packet's buffer descriptors: the frame at index 0, then the
/// frame's allocations in the order its buffer table names them.
///
/// Separated from the channel because it is the whole of what the two halves
/// have to agree on and none of what needs a device. The drain loop's resolve
/// table is this array minus its first entry, and that shift is the single
/// thing a test has to hold the two sides to -- so the array is built here, by
/// a function a host can call, rather than inline in a `send` that only runs on
/// the box.
///
/// `refFDs` carries one FastRPC file descriptor per reference; the descriptor
/// names a buffer by fd, and the same fd must be the one it was mapped with.
///
/// Flags are per class: the frame and every non-constant reference carry the
/// maintenance pair, a reference marked `constant` carries none. `flagAllRefs`
/// nonzero suppresses the distinction and flags everything -- the A/B control
/// for pricing what the per-class default saves, reachable through
/// `MODULAR_NPU_QUEUE_WFLAGS=all` on the real channel.
///
/// Returns the number of descriptors written, or zero when `capacity` cannot
/// hold them.
size_t npuDspQueueDescribeFrame(void *frame, size_t frameBytes, int frameFD,
                                const NPUQueueBufferRef *refs,
                                const int *refFDs, size_t refCount,
                                npu_dspqueue_buffer_t *out, size_t capacity,
                                int flagAllRefs);

/// Binds the vendor queue, or reports why it is unreachable. Returns a channel
/// whose `send` reaches the DSP. Separate from the transport so the transport
/// has no build-time dependency on the SDK.
ErrorOr<std::unique_ptr<NPUQueueChannel>>
createNPUDspQueueChannel(uint64_t skelHandle);

/// Batches launches into frames and keeps a bounded number of them in flight.
///
/// Ordering is total and is the queue's: frames are sent in the order they were
/// filled and the device drains them in that order, so a launch's dependence on
/// an earlier launch's output needs nothing from the caller -- which is what
/// lets this replace a transport whose ordering came from blocking.
class NPUQueueTransport {
public:
  /// Frames the host will leave outstanding before it waits for one back.
  ///
  /// Four, not the vendor's sixteen. Their depth serves a decode loop that
  /// submits from several graphs and wants the DSP fed across any host hiccup;
  /// this is a single-model FIFO lane where one graph submits, in order, and
  /// every graph boundary drains the ring anyway. Depth only has to cover the
  /// host's lookahead, and the measured lookahead is small: the VAE stage
  /// spends 2.393 ms/launch on host gap against 2.724 ms/launch of device, so a
  /// depth of two already hides it and four is the margin for jitter. Past that
  /// each extra frame is another pinned descriptor block and a longer
  /// worst-case drain at the boundary, for no more overlap.
  static constexpr uint32_t kBatchesInFlight = 4;

  /// Scalar bytes one frame reserves. A launch's block is its kernel's scalars
  /// plus one 8-byte offset per operand, so this is a cap on the frame rather
  /// than on any kernel: a launch that does not fit the remainder starts the
  /// next frame.
  static constexpr uint32_t kScalarArenaBytes = 256u * 1024u;

  explicit NPUQueueTransport(std::unique_ptr<NPUQueueChannel> channel);
  ~NPUQueueTransport();

  /// Adds one launch to the open frame, sending that frame first when the
  /// launch does not fit. Never splits a launch across frames: a frame is the
  /// unit the device's cache bracket covers, so half a launch in one is half a
  /// launch whose operands were not declared.
  ErrorOrSuccess enqueue(const NPUQueueLaunch &launch);

  /// Sends the open frame, if there is one. Does not wait for it.
  ErrorOrSuccess flush();

  /// Sends the open frame and waits for every frame in flight. This is the
  /// graph boundary: after it returns, every launch this transport accepted has
  /// run and its outputs are visible to the host.
  ErrorOrSuccess drain();

  /// Launches accepted since construction, and frames sent. Both are what a
  /// test asserts ordering and batching against, and what an A/B reports.
  uint64_t launchesAccepted() const { return launchesAccepted_; }
  uint64_t framesSent() const { return framesSent_; }

private:
  /// One frame as it fills, plus the block it is serialised into. Frames are
  /// reused round-robin and their storage is reserved once, so steady-state
  /// batching allocates nothing.
  struct Frame {
    std::vector<npu_queue_launch_t> launches;
    /// The union of the launches' allocations, in the order first seen. A
    /// launch's `buffer_index` entries index this.
    std::vector<NPUQueueBufferRef> refs;
    std::vector<uint8_t> scalars;
    std::vector<uint8_t> block;
    uint32_t batchID = 0;
    bool inFlight = false;

    /// Filling, as opposed to sent or empty. A frame keeps its contents while
    /// it is in flight -- they are what the device is reading -- so "has
    /// launches" alone would call a frame that is already out an open one, and
    /// a flush would send it twice.
    bool isOpen() const { return !inFlight && !launches.empty(); }
  };

  ErrorOrSuccess sendFrame(Frame &frame);
  /// Waits until at most `atMost` frames remain in flight.
  ErrorOrSuccess waitUntilInFlight(uint32_t atMost);
  /// Serialises a filled frame into its block, in the layout `NPUQueueWire.h`
  /// states. The one place the layout is written, as `npu_queue_frame_valid` is
  /// the one place it is checked.
  void serialize(Frame &frame);

  /// The buffer table index for `ref` in `frame`, adding it when the frame does
  /// not already declare that allocation. This is the union rule the batch
  /// planner replays, and it is why a frame of launches over one arena declares
  /// one buffer rather than one per launch.
  static uint32_t referenceIndexFor(Frame &frame, const NPUQueueBufferRef &ref);

  std::unique_ptr<NPUQueueChannel> channel_;
  Frame frames_[kBatchesInFlight];
  uint32_t openIndex_ = 0;
  uint32_t inFlight_ = 0;
  uint32_t nextBatchID_ = 1;

  /// Microseconds one `receive` may wait before the transport refuses by name
  /// instead of waiting forever. Read from `MODULAR_NPU_QUEUE_RECV_GIVEUP_S`
  /// at construction, default 120 s; `0` opts back into the unbounded wait.
  /// Bounded by default because order 074's presentation -- a clean bind, a
  /// clean `dspqueue_write`, and then silence -- reached the host as an
  /// unbounded poll with no output at all, and a silent infinite hang must
  /// not be reachable again.
  uint32_t receiveGiveupMicros_ = 0;

  uint64_t launchesAccepted_ = 0;
  uint64_t framesSent_ = 0;
};

} // namespace M::Driver

#endif // SKYE_RUNTIME_QURT_NPUQUEUETRANSPORT_H
