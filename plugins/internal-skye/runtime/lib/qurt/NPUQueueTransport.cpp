//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the queue transport: framing, the fit-or-flush
// discipline, the union of allocations a frame declares, and the bounded ring
// of frames in flight.
//
//===----------------------------------------------------------------------===//

#include "NPUQueueTransport.h"

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/Driver/DeviceContext/ErrorMacros.h"
#include "QuRTDriver.h"

#include <cstdlib>
#include <cstring>
#include <string>

namespace M::Driver {

NPUQueueChannel::~NPUQueueChannel() = default;

namespace {

/// Names a frame's failure. The drain refuses a malformed frame with a code of
/// its own before any kernel runs, and those codes are not vendor statuses --
/// rendering one as `0xfffffffe` would send a reader looking through the SDK's
/// error enum for a check this side wrote.
std::string queueStatusMessage(int32_t status) {
  switch (status) {
  case NPU_QUEUE_BAD_FRAME:
    return "the frame was malformed or from a newer layout than this skel "
           "reads";
  case NPU_QUEUE_BAD_BUFFER_INDEX:
    return "a launch named an operand outside the frame's own buffer table";
  case NPU_QUEUE_BAD_SCALARS:
    return "a launch's scalar block reached past the frame's arena";
  case NPU_QUEUE_BAD_LAUNCH_SLOTS:
    return "a launch named more operands than the entry point has slots";
  case NPU_QUEUE_BAD_READ:
    return "the drain's own vendor read failed, so the packet could not be "
           "read on the device at all (the rc is in the queue report)";
  default:
    return qurtStatusMessage(static_cast<uint32_t>(status));
  }
}

/// The receive bound, in the vendor timeout's unit. Default 120 s: a full
/// frame is ~3 s of device work at the recorded worst, so the default is an
/// order of magnitude of headroom, not a tuning point. `0` opts back into
/// waiting forever, which is spelled as the vendor's `TIMEOUT_NONE` rather
/// than left to a zero the channel would read as "do not wait at all".
uint32_t receiveGiveupMicrosFromEnv() {
  unsigned long seconds = 120;
  if (const char *env = getenv("MODULAR_NPU_QUEUE_RECV_GIVEUP_S"))
    if (*env != '\0')
      seconds = strtoul(env, nullptr, 10);
  if (seconds == 0)
    return NPU_DSPQUEUE_TIMEOUT_NONE;
  // The vendor timeout is 32-bit microseconds, so the largest expressible
  // finite bound is ~4294 s; a value clamped here is still a bound.
  if (seconds > 4000)
    seconds = 4000;
  return static_cast<uint32_t>(seconds) * 1000000u;
}

} // namespace

NPUQueueTransport::NPUQueueTransport(std::unique_ptr<NPUQueueChannel> channel)
    : channel_(std::move(channel)),
      receiveGiveupMicros_(receiveGiveupMicrosFromEnv()) {
  // Reserved once so that steady-state batching -- the case this transport
  // exists for -- neither allocates nor moves a block the channel may still be
  // reading from.
  for (Frame &frame : frames_) {
    frame.launches.reserve(NPU_QUEUE_MAX_LAUNCHES);
    frame.refs.reserve(NPU_QUEUE_MAX_BUFFERS);
    frame.scalars.reserve(kScalarArenaBytes);
    frame.block.reserve(npu_queue_frame_bytes(
        NPU_QUEUE_MAX_LAUNCHES, NPU_QUEUE_MAX_BUFFERS, kScalarArenaBytes));
  }
}

NPUQueueTransport::~NPUQueueTransport() = default;

uint32_t NPUQueueTransport::referenceIndexFor(Frame &frame,
                                              const NPUQueueBufferRef &ref) {
  // Linear because the union is bounded at 64 and a frame's launches overwhelm-
  // ingly repeat the same few arenas: the whole denoise activation set is one
  // 0.96 GiB allocation, so the common answer is index 0 on the first compare.
  for (uint32_t i = 0; i < frame.refs.size(); ++i)
    if (frame.refs[i].base == ref.base)
      return i;
  frame.refs.push_back(ref);
  return static_cast<uint32_t>(frame.refs.size() - 1);
}

ErrorOrSuccess NPUQueueTransport::enqueue(const NPUQueueLaunch &launch) {
  if (launch.bufferCount > NPU_QUEUE_BUFFER_SLOTS)
    RETURN_ERROR("an NPU queue launch takes at most " +
                 std::to_string(NPU_QUEUE_BUFFER_SLOTS) + " buffers, got " +
                 std::to_string(launch.bufferCount));
  if (launch.scalarBytes > kScalarArenaBytes)
    RETURN_ERROR("an NPU queue launch's scalar block is " +
                 std::to_string(launch.scalarBytes) +
                 " bytes, past the frame's whole arena of " +
                 std::to_string(kScalarArenaBytes));

  // The slot this launch goes in may still be out, which at depth four means
  // the device is that many frames behind. This is the whole of the
  // backpressure, and it belongs here rather than in `flush` so the ring can
  // hold its depth: it collects one completion, not the whole ring.
  while (frames_[openIndex_].inFlight)
    RETURN_IF_ERROR(waitUntilInFlight(inFlight_ - 1));
  Frame &frame = frames_[openIndex_];

  // Fit before add, never a partial launch in a frame. What the launch would
  // add to the union is counted against the cap without being committed, so a
  // refusal leaves the open frame exactly as it was.
  uint32_t wouldAdd = 0;
  for (size_t i = 0; i < launch.bufferCount; ++i) {
    bool known = false;
    for (const NPUQueueBufferRef &ref : frame.refs)
      if (ref.base == launch.buffers[i].base)
        known = true;
    // A launch naming one allocation twice adds it once, which is the same
    // answer `referenceIndexFor` gives.
    for (size_t j = 0; j < i; ++j)
      if (launch.buffers[j].base == launch.buffers[i].base)
        known = true;
    if (!known)
      ++wouldAdd;
  }

  bool fits = frame.launches.size() < NPU_QUEUE_MAX_LAUNCHES &&
              frame.refs.size() + wouldAdd <= NPU_QUEUE_MAX_BUFFERS &&
              frame.scalars.size() + launch.scalarBytes <= kScalarArenaBytes;
  if (!fits && frame.isOpen()) {
    RETURN_IF_ERROR(flush());
    return enqueue(launch);
  }
  if (!fits)
    RETURN_ERROR("an NPU queue launch fits in no frame: " +
                 std::to_string(launch.bufferCount) + " buffers and " +
                 std::to_string(launch.scalarBytes) + " scalar bytes");

  npu_queue_launch_t record = {};
  record.selector = launch.selector;
  record.launch_word = launch.launchWord;
  record.scalar_offset = static_cast<uint32_t>(frame.scalars.size());
  record.scalar_bytes = static_cast<uint32_t>(launch.scalarBytes);
  record.buffer_count = static_cast<uint32_t>(launch.bufferCount);
  record.status = 0;
  for (uint32_t i = 0; i < NPU_QUEUE_BUFFER_SLOTS; ++i)
    record.buffer_index[i] = NPU_QUEUE_NO_BUFFER;
  for (size_t i = 0; i < launch.bufferCount; ++i)
    record.buffer_index[i] =
        static_cast<uint16_t>(referenceIndexFor(frame, launch.buffers[i]));

  // Copied rather than referenced: the caller's block is a stack local of the
  // dispatch that is about to return, and the frame outlives it by definition
  // now that the send does not block.
  frame.scalars.insert(frame.scalars.end(), launch.scalars,
                       launch.scalars + launch.scalarBytes);
  frame.launches.push_back(record);
  ++launchesAccepted_;
  return success();
}

void NPUQueueTransport::serialize(Frame &frame) {
  const auto launchCount = static_cast<uint32_t>(frame.launches.size());
  const auto bufferCount = static_cast<uint32_t>(frame.refs.size());
  const auto scalarBytes = static_cast<uint32_t>(frame.scalars.size());

  npu_queue_frame_header_t header = {};
  header.magic = NPU_QUEUE_FRAME_MAGIC;
  header.version = NPU_QUEUE_FRAME_VERSION;
  header.batch_id = frame.batchID;
  header.launch_count = launchCount;
  header.buffer_count = bufferCount;
  header.flags = 0;
  header.buffer_table_offset = sizeof(npu_queue_frame_header_t);
  header.launch_table_offset =
      header.buffer_table_offset + bufferCount * sizeof(npu_queue_buffer_t);
  header.scalar_arena_offset =
      header.launch_table_offset + launchCount * sizeof(npu_queue_launch_t);
  header.scalar_arena_bytes = scalarBytes;
  header.frame_bytes =
      npu_queue_frame_bytes(launchCount, bufferCount, scalarBytes);

  // Zero-filled to the aligned length: the tail past the arena is padding the
  // device does not read, and leaving a previous frame's bytes there would make
  // two identical batches produce different frames.
  frame.block.assign(header.frame_bytes, 0);
  uint8_t *out = frame.block.data();
  memcpy(out, &header, sizeof(header));

  auto *buffers =
      reinterpret_cast<npu_queue_buffer_t *>(out + header.buffer_table_offset);
  for (uint32_t i = 0; i < bufferCount; ++i) {
    // The reference index is the position in the attachment list `send` builds
    // from the same vector, so the two are the same numbering by construction.
    buffers[i].reference = i;
    buffers[i].bytes = static_cast<uint32_t>(frame.refs[i].byteSize);
  }
  memcpy(out + header.launch_table_offset, frame.launches.data(),
         launchCount * sizeof(npu_queue_launch_t));
  memcpy(out + header.scalar_arena_offset, frame.scalars.data(), scalarBytes);
}

ErrorOrSuccess NPUQueueTransport::sendFrame(Frame &frame) {
  serialize(frame);
  RETURN_IF_ERROR(channel_->send(frame.block.data(), frame.block.size(),
                                 frame.refs.data(), frame.refs.size()));
  frame.inFlight = true;
  ++inFlight_;
  ++framesSent_;
  return success();
}

ErrorOrSuccess NPUQueueTransport::waitUntilInFlight(uint32_t atMost) {
  while (inFlight_ > atMost) {
    uint32_t batchID = 0;
    int32_t status = 0;
    // Blocking, because the caller is either out of ring or at a graph
    // boundary; in both cases there is nothing else for this thread to do and
    // a poll would only spin. The device bounds the wait in the healthy case;
    // the giveup bounds it in the 074 case, where the device accepted the
    // frame and then went silent.
    DEFINE_OR_RETURN_ERROR(
        bool, got, channel_->receive(batchID, status, receiveGiveupMicros_));
    if (!got) {
      channel_->dumpReport();
      RETURN_ERROR(
          "the NPU queue got no completion within " +
          std::to_string(receiveGiveupMicros_ / 1000000u) + " s with " +
          std::to_string(inFlight_) + " frame(s) in flight (" +
          std::to_string(framesSent_) +
          " sent this session). The device accepted the frame and went "
          "silent; its own account of why is in the queue report above. "
          "MODULAR_NPU_QUEUE_RECV_GIVEUP_S tunes this bound, 0 waits forever");
    }

    // The drain's read-failure refusal: batch id zero names no frame by
    // construction (ids are monotonic from one), and the status says the
    // packet at the head of the device's queue could not be read at all.
    // Recognised before the ring match so it is reported as what it is
    // rather than as "a frame this transport did not send".
    if (batchID == 0 && status == NPU_QUEUE_BAD_READ) {
      channel_->dumpReport();
      RETURN_ERROR("the NPU drain refused a packet: " +
                   queueStatusMessage(status));
    }

    Frame *completed = nullptr;
    for (Frame &candidate : frames_)
      if (candidate.inFlight && candidate.batchID == batchID)
        completed = &candidate;
    if (completed == nullptr)
      RETURN_ERROR("the NPU queue completed frame " + std::to_string(batchID) +
                   ", which this transport did not send");

    // Recycled where it completes, so a frame that is not in flight is always
    // ready to fill and `isOpen` needs no third state.
    completed->inFlight = false;
    completed->launches.clear();
    completed->refs.clear();
    completed->scalars.clear();
    --inFlight_;
    if (status != 0)
      RETURN_ERROR("an NPU launch in queue frame " + std::to_string(batchID) +
                   " failed: " + queueStatusMessage(status));
  }
  return success();
}

ErrorOrSuccess NPUQueueTransport::flush() {
  Frame &frame = frames_[openIndex_];
  if (!frame.isOpen())
    return success();

  frame.batchID = nextBatchID_++;
  RETURN_IF_ERROR(sendFrame(frame));

  // Advance and return without waiting. A flush that waited for the next slot
  // would leave the ring one frame short of its depth: after this the device
  // may legitimately have every frame outstanding, and the producer has no
  // reason to stall until it needs a slot back.
  openIndex_ = (openIndex_ + 1) % kBatchesInFlight;
  return success();
}

ErrorOrSuccess NPUQueueTransport::drain() {
  RETURN_IF_ERROR(flush());
  return waitUntilInFlight(0);
}

} // namespace M::Driver

//===----------------------------------------------------------------------===//
// The vendor-backed channel
//===----------------------------------------------------------------------===//
//
// Everything above is ours and runs anywhere. This is the one place the vendor
// ABI is spoken.
//
// It used to refuse to be written, because `dspqueue_write` hands the DLL a
// buffer descriptor it dereferences and the descriptor's field layout was not
// on record -- a layout restated from memory that had drifted would be a wrong
// pointer rather than a failed call. That is no longer the state of the world:
// `struct dspqueue_buffer` is public in `dspqueue.h` and is four `uint32_t`
// plus a `union {void *; uint64_t}`, 24 bytes at 8-byte alignment, asserted
// against the real header by the SDK's own hexagon-clang with a control leg
// that failed as it should. `NPUDspQueueABI.h` carries that layout and the
// assertions that hold this build to it; the device compile, where the real
// header *is* readable, asserts the restatement against it.
//
// So this is compiled unconditionally, exactly like the rest of `QuRTDriver`:
// the entry points are bound with `dlsym` and none of the SDK's headers are on
// the host include path -- there is no Hexagon SDK build for this host and
// there never will be. `MODULAR_NPU_DSPQUEUE` is gone; a driver that exports no
// `dspqueue_*` is reported by name at `createNPUDspQueueChannel`, which is
// where the caller asked for this transport and so where a missing symbol is a
// diagnosis rather than a crash.

#include "NPUDspQueueABI.h"

#include <chrono>
#include <map>
#include <thread>

namespace M::Driver {

size_t npuDspQueueDescribeFrame(void *frame, size_t frameBytes, int frameFD,
                                const NPUQueueBufferRef *refs,
                                const int *refFDs, size_t refCount,
                                npu_dspqueue_buffer_t *out, size_t capacity,
                                int flagAllRefs) {
  if (capacity < refCount + 1)
    return 0;

  // Zeroed rather than field-assigned, because `ptr` and `address` overlay each
  // other: on a 32-bit endpoint a write to `ptr` leaves the high four bytes of
  // the union whatever they were. This host is 64-bit and the write fills the
  // slot, but the descriptor is the vendor's shape and not ours, and a struct
  // whose correctness depends on the host's pointer width is one that breaks
  // silently on the endpoint where it does not.
  memset(out, 0, (refCount + 1) * sizeof(npu_dspqueue_buffer_t));

  // The frame first, at packet index 0, which is what makes the drain loop's
  // resolve table `bases + 1`. It is a buffer and not the packet's message
  // because a message is copied into the queue's own memory and a full frame is
  // a quarter of a megabyte.
  out[0].fd = static_cast<uint32_t>(frameFD);
  out[0].size = static_cast<uint32_t>(frameBytes);
  out[0].offset = 0;
  out[0].flags = NPU_DSPQUEUE_TRANSPORT_BUFFER_FLAGS;
  out[0].ptr = frame;

  for (size_t i = 0; i < refCount; ++i) {
    out[i + 1].fd = static_cast<uint32_t>(refFDs[i]);
    out[i + 1].size = static_cast<uint32_t>(refs[i].byteSize);
    out[i + 1].offset = 0;
    // Maintenance is per class, not per descriptor: a declared-constant
    // reference (a weight) was written once at init and never again, so
    // flushing the host's caches for it and invalidating the device's costs
    // O(model) bytes per frame for lines that cannot be stale. The shipped
    // ggml-hexagon backend makes the same split harder still -- its data
    // tensors never ride a descriptor and only the one metadata block carries
    // the pair (`GGML_HEXAGON_LESSONS.md` section 3, their host `:1477` and
    // DSP `main.c:1091-1092`). `flagAllRefs` is the A/B control that restores
    // the old everything-flagged behaviour.
    out[i + 1].flags = (refs[i].constant && !flagAllRefs)
                           ? 0
                           : NPU_DSPQUEUE_TRANSPORT_BUFFER_FLAGS;
    out[i + 1].ptr = refs[i].base;
  }
  return refCount + 1;
}

/// Renders a host-side AEE status. Deliberately not `qurtStatusMessage`, which
/// reads the DSP's numbering: `AEE_EOFFSET` is `0x80000400` under `__hexagon__`
/// and zero on the host, so the same symbolic error is `0x8000040E` on one side
/// and `14` on the other. Only the codes the dspqueue and fastrpc_mmap return
/// lists actually name are here; anything else is reported as its number, which
/// is at least not a wrong name.
std::string dspQueueStatusMessage(int status) {
  const char *name = nullptr;
  switch (status) {
  case 1:
    name = "AEE_EFAILED";
    break;
  case 2:
    name = "AEE_ENOMEMORY";
    break;
  case 12:
    name = "AEE_EEXPIRED";
    break;
  case 13:
    name = "AEE_EBADSTATE";
    break;
  case 14:
    name = "AEE_EBADPARM";
    break;
  case 16:
    name = "AEE_EBADITEM";
    break;
  case 20:
    name = "AEE_EUNSUPPORTED";
    break;
  case 32:
    name = "AEE_EITEMBUSY";
    break;
  case 46:
    name = "AEE_EINTERRUPTED";
    break;
  case 72:
    // The one worth naming in full: it means a buffer reached a packet without
    // having been mapped, which is the precondition `dspqueue.h` states twice.
    name = "AEE_ENOSUCHMAP (a buffer in the packet was never fastrpc_mmap'd)";
    break;
  case 512:
    // The other one worth naming in full: the RPC layer itself failed, which
    // says nothing about the call's merits. Order 163 met it from
    // `fastrpc_mmap` at every length and both modes, because the domain had no
    // open session yet -- so the cause it points at is the call's ORDER, not
    // its arguments.
    name = "AEE_ERPC (the RPC layer failed; is a session open on this domain?)";
    break;
  case NPU_DSPQUEUE_EWOULDBLOCK:
    name = "AEE_EWOULDBLOCK";
    break;
  default:
    return "status " + std::to_string(status);
  }
  return std::string(name) + " (" + std::to_string(status) + ")";
}

namespace {

/// Method indices in `mojo_npu.idl`. `dispatch` is 3 and is marshalled by hand
/// everywhere, so these two are its successors in declaration order -- which is
/// exactly why the IDL appends them rather than inserting.
constexpr uint32_t kNPUStartMethodID = 4;
constexpr uint32_t kNPUStopMethodID = 5;
/// The generic entry point itself, for the one dispatch this file sends: the
/// post-mortem report read-back rides the synchronous transport because it is
/// asked for exactly when the queue has stopped answering.
constexpr uint32_t kNPUDispatchMethodID = 3;
constexpr uint32_t kNPUMaxBufferSlots = 8;

/// The `queue_report` selector, from the manifest rather than by hand -- the
/// same expansion the device trampoline and the host kernel list read, so the
/// three numberings cannot drift.
enum {
#define NPU_KERNEL(name, selector, bufferCount, scalarCount, call)             \
  kNPUSelector_##name = selector,
#include "NPUKernels.def"
};

/// Queue memory the vendor allocates for each direction, sized from the packet
/// and the ring rather than picked: twice the in-flight depth in packets plus
/// slack, which is the shipped ggml-hexagon recipe at their depth of sixteen
/// (`GGML_HEXAGON_LESSONS.md` section 1, their `:1798-1799`) applied to this
/// ring's depth of four. A packet is its descriptor block -- 65 descriptors at
/// 24 bytes -- so the arithmetic keeps the write path out of
/// `dspqueue_write`'s blocking-for-room case by construction even if the
/// descriptor count grows.
constexpr uint32_t kPacketDescriptorBytes =
    NPU_QUEUE_MAX_PACKET_BUFFERS * sizeof(npu_dspqueue_buffer_t);
constexpr uint32_t kRequestQueueBytes =
    kPacketDescriptorBytes * NPUQueueTransport::kBatchesInFlight * 2 + 1024;
constexpr uint32_t kResponseQueueBytes =
    (kPacketDescriptorBytes + sizeof(npu_queue_response_t)) *
        NPUQueueTransport::kBatchesInFlight * 2 +
    1024;

/// Polls before `receive` stops spinning and starts sleeping. A completion the
/// device is about to post arrives inside a few microseconds and a sleep would
/// round that up to the host scheduler's tick; a completion that is a whole
/// launch away is worth a syscall to wait for. The first is the ring-full case,
/// the second is the graph boundary.
constexpr uint32_t kSpinPolls = 64;
constexpr auto kPollSleep = std::chrono::microseconds(200);

/// The vendor queue, bound through the entry points `QuRTDriver` resolved.
class DspQueueChannel : public NPUQueueChannel {
public:
  DspQueueChannel(QuRTDriver &driver, uint64_t skelHandle, void *queue,
                  bool flagAllRefs)
      : driver_(driver), skelHandle_(skelHandle), queue_(queue),
        flagAllRefs_(flagAllRefs) {}

  ~DspQueueChannel() override {
    // Stopped before the queue is closed, and the queue closed before the
    // mappings are dropped. A drain thread still reading a queue that has been
    // closed is what makes a wedged protection domain stick, and on this box a
    // wedged CDSP is a reboot a human has to authorise.
    QuRTRemoteBuf unused{};
    driver_.handleInvoke(skelHandle_, qurtScalars(kNPUStopMethodID, 0, 0),
                         &unused);
    driver_.dspqueueClose(queue_);
    for (const auto &[fd, mapping] : mapped_)
      driver_.fastrpcMunmap(kQuRTComputeDomainID, fd, mapping.base,
                            mapping.byteSize);
    for (Staging &staging : staging_)
      if (staging.base != nullptr)
        driver_.rpcmemFree(staging.base);
  }

  /// Allocates the frame blocks and maps them. Separate from the constructor so
  /// a failure is a returned error rather than a half-built channel.
  ErrorOrSuccess bind() {
    const size_t frameCapacity =
        npu_queue_frame_bytes(NPU_QUEUE_MAX_LAUNCHES, NPU_QUEUE_MAX_BUFFERS,
                              NPUQueueTransport::kScalarArenaBytes);
    for (Staging &staging : staging_) {
      // Allocated at the frame's maximum rather than grown, so that a `send` in
      // steady state neither allocates nor maps -- which is the whole of what
      // this transport is for.
      staging.base =
          driver_.rpcmemAlloc(kQuRTHeapIDSystem, kQuRTRpcmemDefaultFlags,
                              static_cast<int>(frameCapacity));
      if (staging.base == nullptr)
        RETURN_ERROR("the NPU queue could not allocate a " +
                     std::to_string(frameCapacity) + "-byte frame block");
      staging.byteSize = frameCapacity;
      DEFINE_OR_RETURN_ERROR(int, fd,
                             mapAllocation(staging.base, frameCapacity));
      staging.fd = fd;
    }
    return success();
  }

  ErrorOrSuccess send(const void *frame, size_t frameBytes,
                      const NPUQueueBufferRef *refs, size_t refCount) override {
    if (refCount > NPU_QUEUE_MAX_BUFFERS)
      RETURN_ERROR("an NPU queue frame declared " + std::to_string(refCount) +
                   " allocations, past the packet's " +
                   std::to_string(NPU_QUEUE_MAX_BUFFERS));

    Staging *staging = nullptr;
    for (Staging &candidate : staging_)
      if (candidate.batchID == 0) {
        staging = &candidate;
        break;
      }
    if (staging == nullptr)
      RETURN_ERROR("the NPU queue has no free frame block, so more frames are "
                   "in flight than the ring allows");
    if (frameBytes > staging->byteSize)
      RETURN_ERROR("an NPU queue frame is " + std::to_string(frameBytes) +
                   " bytes, past the block's " +
                   std::to_string(staging->byteSize));

    // Copied into mapped memory rather than sent from the transport's own
    // block: a descriptor names a buffer by FastRPC file descriptor, and the
    // transport's block is plain heap that has none.
    memcpy(staging->base, frame, frameBytes);

    int refFDs[NPU_QUEUE_MAX_BUFFERS];
    for (size_t i = 0; i < refCount; ++i) {
      DEFINE_OR_RETURN_ERROR(int, fd,
                             mapAllocation(refs[i].base, refs[i].byteSize));
      refFDs[i] = fd;
    }

    npu_dspqueue_buffer_t descriptors[NPU_QUEUE_MAX_PACKET_BUFFERS];
    size_t count = npuDspQueueDescribeFrame(
        staging->base, frameBytes, staging->fd, refs, refFDs, refCount,
        descriptors, NPU_QUEUE_MAX_PACKET_BUFFERS, flagAllRefs_ ? 1 : 0);
    if (count == 0)
      RETURN_ERROR("an NPU queue frame did not fit the packet's descriptors");

    // No message: the frame is a buffer, and every field the drain loop reads
    // before it trusts the frame is in the frame's own header. A null message
    // is what the header's own "zero if the packet contains no message" allows
    // -- said of this parameter, and not of the read side's maximum.
    int rc = driver_.dspqueueWrite(queue_, NPU_DSPQUEUE_PACKET_FLAG_BUFFERS,
                                   static_cast<uint32_t>(count), descriptors,
                                   /*messageBytes=*/0, /*message=*/nullptr,
                                   NPU_DSPQUEUE_TIMEOUT_NONE);
    if (rc != 0)
      RETURN_ERROR("dspqueue_write of an NPU launch frame failed: " +
                   dspQueueStatusMessage(rc));

    const auto *header = static_cast<const npu_queue_frame_header_t *>(frame);
    staging->batchID = header->batch_id;
    return success();
  }

  void dumpReport() override {
    // One dispatch of the `queue_report` selector, marshalled by hand the way
    // the power vote's is: the wire a single launch sends, at this row's
    // arity of one buffer and one scalar (the word capacity).
    constexpr uint32_t kWords = NPU_QUEUE_REPORT_WORDS;
    auto *words = static_cast<uint32_t *>(driver_.rpcmemAlloc(
        kQuRTHeapIDSystem, kQuRTRpcmemDefaultFlags, kWords * sizeof(uint32_t)));
    if (words == nullptr) {
      fprintf(stderr, "npu queue report: unavailable (no report buffer)\n");
      return;
    }
    memset(words, 0, kWords * sizeof(uint32_t));

    // One scalar slot for the capacity, one offset slot for the buffer.
    uint32_t scalars[4] = {};
    scalars[0] = kWords;

    uint32_t primIn[3 + 2 * kNPUMaxBufferSlots] = {};
    primIn[0] = kNPUSelector_queue_report;
    primIn[1] = 0x1;
    primIn[2] = sizeof(scalars);
    primIn[3] = kWords * sizeof(uint32_t);
    primIn[4] = kWords * sizeof(uint32_t);

    QuRTRemoteBuf slots[2 + 2 * kNPUMaxBufferSlots] = {};
    slots[0] = {primIn, sizeof(primIn)};
    slots[1] = {scalars, sizeof(scalars)};
    slots[2] = {words, kWords * sizeof(uint32_t)};
    slots[2 + kNPUMaxBufferSlots] = {words, kWords * sizeof(uint32_t)};

    uint32_t status = static_cast<uint32_t>(driver_.handleInvoke(
        skelHandle_,
        qurtScalars(kNPUDispatchMethodID, /*inBufs=*/2 + kNPUMaxBufferSlots,
                    /*outBufs=*/kNPUMaxBufferSlots),
        slots));
    if (status != 0) {
      fprintf(stderr,
              "npu queue report: unavailable -- the signed skel refused the "
              "queue_report selector (%s); it predates the row\n",
              qurtStatusMessage(status).c_str());
      driver_.rpcmemFree(words);
      return;
    }
    if (words[NPU_QUEUE_REPORT_MAGIC_WORD] != NPU_QUEUE_REPORT_MAGIC) {
      fprintf(stderr, "npu queue report: not initialized -- the skel's start "
                      "never ran in this session\n");
      driver_.rpcmemFree(words);
      return;
    }
    // One protocol-greppable line per group, every 074 candidate separated:
    // import mode and rc (candidate 1), the echoed queue id (candidate 3),
    // the read-rc split and pending depth (would-block-forever vs a refused
    // head packet), and the last frame's magic/parse words (stale cache vs
    // framing, candidates 4 and 2).
    fprintf(stderr, "npu queue report: state %u, import %s (rc %d)\n",
            words[NPU_QUEUE_REPORT_STATE],
            words[NPU_QUEUE_REPORT_IMPORT_MODE] ? "callback-fallback"
                                                : "null-callbacks",
            static_cast<int32_t>(words[NPU_QUEUE_REPORT_IMPORT_RC]));
    fprintf(
        stderr, "npu queue report: queue id as decoded %llu\n",
        static_cast<unsigned long long>(
            (static_cast<uint64_t>(words[NPU_QUEUE_REPORT_QUEUE_ID_HI]) << 32) |
            words[NPU_QUEUE_REPORT_QUEUE_ID_LO]));
    fprintf(stderr,
            "npu queue report: wakes %u, empty reads %u, frames seen %u, "
            "last read rc %d (x%u)\n",
            words[NPU_QUEUE_REPORT_WAKES], words[NPU_QUEUE_REPORT_EMPTY_READS],
            words[NPU_QUEUE_REPORT_FRAMES_SEEN],
            static_cast<int32_t>(words[NPU_QUEUE_REPORT_LAST_READ_RC]),
            words[NPU_QUEUE_REPORT_READ_RC_REPEATS]);
    fprintf(stderr,
            "npu queue report: last packet flags 0x%x, %u buffers, %u message "
            "bytes, frame %u bytes, frame ptr class %u\n",
            words[NPU_QUEUE_REPORT_LAST_PACKET_FLAGS],
            words[NPU_QUEUE_REPORT_LAST_PACKET_BUFFERS],
            words[NPU_QUEUE_REPORT_LAST_PACKET_MESSAGE_BYTES],
            words[NPU_QUEUE_REPORT_LAST_PACKET_FRAME_BYTES],
            words[NPU_QUEUE_REPORT_FRAME_PTR_CLASS]);
    fprintf(stderr,
            "npu queue report: last frame magic 0x%x version %u batch %u, "
            "parse status %d\n",
            words[NPU_QUEUE_REPORT_LAST_FRAME_MAGIC],
            words[NPU_QUEUE_REPORT_LAST_FRAME_VERSION],
            words[NPU_QUEUE_REPORT_LAST_FRAME_BATCH],
            static_cast<int32_t>(words[NPU_QUEUE_REPORT_LAST_PARSE_STATUS]));
    fprintf(stderr,
            "npu queue report: responses %u, last write rc %d, refusals %u, "
            "pending packets %u (%u samples)\n",
            words[NPU_QUEUE_REPORT_RESPONSES_WRITTEN],
            static_cast<int32_t>(words[NPU_QUEUE_REPORT_LAST_RESPONSE_RC]),
            words[NPU_QUEUE_REPORT_REFUSALS_SENT],
            words[NPU_QUEUE_REPORT_PENDING_PACKETS],
            words[NPU_QUEUE_REPORT_PENDING_SAMPLES]);
    // Appended, and only when the vendor's error callback ever fired, so the
    // established report lines keep their exact shape for the leg protocols
    // that grep them.
    if (words[NPU_QUEUE_REPORT_ERROR_CB_COUNT] != 0)
      fprintf(stderr, "npu queue report: error callback rc %d (x%u)\n",
              static_cast<int32_t>(words[NPU_QUEUE_REPORT_ERROR_CB_RC]),
              words[NPU_QUEUE_REPORT_ERROR_CB_COUNT]);
    driver_.rpcmemFree(words);
  }

  ErrorOr<bool> receive(uint32_t &batchID, int32_t &status,
                        uint32_t timeoutMicros) override {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::microseconds(timeoutMicros);
    for (uint32_t poll = 0;; ++poll) {
      uint32_t flags = 0;
      uint32_t bufferCount = 0;
      uint32_t messageBytes = 0;
      npu_dspqueue_buffer_t descriptors[NPU_QUEUE_MAX_PACKET_BUFFERS];
      npu_queue_response_t response = {};

      // `_noblock` and a poll rather than `dspqueue_read`: blocking reads
      // misbehave on Windows-on-Snapdragon, which the reference implementation
      // works around the same way (`htp-drv.cpp:175-183`).
      int rc = driver_.dspqueueReadNoblock(
          queue_, &flags, NPU_QUEUE_MAX_PACKET_BUFFERS, &bufferCount,
          descriptors, sizeof(response), &messageBytes,
          reinterpret_cast<uint8_t *>(&response));
      if (rc == 0) {
        if (messageBytes != sizeof(response))
          RETURN_ERROR("an NPU queue completion carried " +
                       std::to_string(messageBytes) +
                       " message bytes, not the " +
                       std::to_string(sizeof(response)) + " a response is");
        batchID = response.batch_id;
        status = response.status;
        for (Staging &staging : staging_)
          if (staging.batchID == batchID)
            staging.batchID = 0;
        return true;
      }
      if (rc != NPU_DSPQUEUE_EWOULDBLOCK)
        RETURN_ERROR("dspqueue_read_noblock failed: " +
                     dspQueueStatusMessage(rc));

      // `DSPQUEUE_TIMEOUT_NONE` is the transport's "wait for the device", and
      // it is the same bit pattern as the largest finite timeout, so it is
      // spelled out rather than left to overflow into a deadline.
      if (timeoutMicros != NPU_DSPQUEUE_TIMEOUT_NONE &&
          std::chrono::steady_clock::now() >= deadline)
        return false;
      if (poll < kSpinPolls)
        std::this_thread::yield();
      else
        std::this_thread::sleep_for(kPollSleep);
    }
  }

private:
  struct Staging {
    void *base = nullptr;
    size_t byteSize = 0;
    int fd = 0;
    /// The frame this block is carrying, or zero when it is free. Batch ids are
    /// monotonic from one, so zero is not a frame.
    uint32_t batchID = 0;
  };

  struct Mapping {
    void *base = nullptr;
    size_t byteSize = 0;
  };

  /// The file descriptor for `base`, mapping the allocation on first use.
  ///
  /// `dspqueue.h` states the precondition twice -- a buffer must be mapped with
  /// `fastrpc_mmap` before a packet may name it, or the write fails with
  /// `AEE_ENOSUCHMAP` -- and that mapping is what the synchronous path got for
  /// free, per call, from the driver. Doing it once per allocation instead of
  /// once per launch is the other half of what this transport is worth.
  ///
  /// Keyed by descriptor and not by address, because a descriptor names a
  /// buffer by descriptor and the vendor's unmap is by descriptor too.
  ///
  /// TODO(fabio): an allocation freed and reallocated can reuse its file
  /// descriptor, and this cache would then skip mapping the new one. Nothing in
  /// the queue's A/B frees while frames are in flight; wire `chunkFree` through
  /// to here before a serving path uses this transport.
  ErrorOr<int> mapAllocation(void *base, size_t byteSize) {
    int fd = driver_.rpcmemToFD(base);
    if (fd < 0)
      RETURN_ERROR("rpcmem_to_fd found no NPU allocation at " +
                   std::to_string(reinterpret_cast<uintptr_t>(base)));
    auto [entry, added] = mapped_.try_emplace(fd, Mapping{base, byteSize});
    if (!added)
      return fd;

    // MAP_FD, not MAP_STATIC: the vendor's buffer bookkeeping is fd-keyed
    // and a STATIC mapping never lands in it -- order 081 measured the DSP
    // read refusing the very first packet with AEE_ENOSUCHMAP under STATIC.
    // The full argument is on the constant's definition.
    int rc = driver_.fastrpcMmap(kQuRTComputeDomainID, fd, base, /*offset=*/0,
                                 byteSize, NPU_FASTRPC_MAP_FD);
    if (rc != 0) {
      mapped_.erase(entry);
      RETURN_ERROR("fastrpc_mmap of the " + std::to_string(byteSize) +
                   "-byte NPU allocation a queue frame names failed: " +
                   dspQueueStatusMessage(rc));
    }
    return fd;
  }

  QuRTDriver &driver_;
  uint64_t skelHandle_;
  void *queue_;
  /// `MODULAR_NPU_QUEUE_WFLAGS=all` at bind: every reference keeps the
  /// maintenance pair, constant or not. The B leg of the per-class A/B.
  bool flagAllRefs_;
  Staging staging_[NPUQueueTransport::kBatchesInFlight];
  std::map<int, Mapping> mapped_;
};

} // namespace

ErrorOr<std::unique_ptr<NPUQueueChannel>>
createNPUDspQueueChannel(uint64_t skelHandle) {
  ErrorOr<QuRTDriver &> driver = QuRTDriver::get();
  if (driver.isError())
    RETURN_ERROR(driver.getError());
  // Reported rather than silently falling back, because the caller asked for
  // this transport by name: `MODULAR_NPU_QUEUE=1` is an A/B knob, and an A/B
  // whose B leg quietly ran the A path is worse than one that refused.
  if (driver->missingQueueSymbol != nullptr)
    RETURN_ERROR("this NPU driver exports no " +
                 std::string(driver->missingQueueSymbol) +
                 ", so launches cannot travel over a queue on this box. The "
                 "transport's framing, batching and ring are built in either "
                 "way and are what the device-free tests cover");

  void *queue = nullptr;
  int rc = driver->dspqueueCreate(kQuRTComputeDomainID, /*flags=*/0,
                                  kRequestQueueBytes, kResponseQueueBytes,
                                  /*packetCallback=*/nullptr,
                                  /*errorCallback=*/nullptr,
                                  /*callbackContext=*/nullptr, &queue);
  if (rc != 0)
    RETURN_ERROR("dspqueue_create for the NPU failed: " +
                 dspQueueStatusMessage(rc));

  uint64_t queueID = 0;
  rc = driver->dspqueueExport(queue, &queueID);
  if (rc != 0) {
    driver->dspqueueClose(queue);
    RETURN_ERROR("dspqueue_export for the NPU failed: " +
                 dspQueueStatusMessage(rc));
  }
  // Rendered exactly like the report's "queue id as decoded" line, plus hex:
  // the 076 leg echoed a decoded id whose low word was zero -- the shape a
  // 32-bit skew would produce -- and could not adjudicate it, because nothing
  // printed the id this side actually exported. These two lines are that
  // comparison, for free, on every bind.
  fprintf(stderr, "npu queue exported id %llu (0x%016llx)\n",
          static_cast<unsigned long long>(queueID),
          static_cast<unsigned long long>(queueID));

  // Stated before anything can fail past here, like the skel-open line: which
  // flag policy a leg ran under is the first question of the WFLAGS A/B, and
  // it must be on record even for a bind that then refuses.
  const char *wflags = getenv("MODULAR_NPU_QUEUE_WFLAGS");
  bool flagAllRefs = wflags != nullptr && strcmp(wflags, "all") == 0;
  fprintf(stderr, "npu queue flags: %s\n",
          flagAllRefs ? "all references carry the maintenance pair "
                        "(MODULAR_NPU_QUEUE_WFLAGS=all)"
                      : "per-class (constant references unflagged)");

  auto channel = std::make_unique<DspQueueChannel>(*driver, skelHandle, queue,
                                                   flagAllRefs);
  RETURN_IF_ERROR(channel->bind());

  // `start` is the skel's, and it is invoked by hand at its IDL index for the
  // same reason `dispatch` is. A host built against a skel that predates the
  // method fails here -- loudly, at the one call that is new -- rather than
  // somewhere downstream.
  QuRTRemoteBuf startArgs[1] = {{&queueID, sizeof(queueID)}};
  uint32_t status = static_cast<uint32_t>(driver->handleInvoke(
      skelHandle, qurtScalars(kNPUStartMethodID, /*inBufs=*/1, /*outBufs=*/0),
      startArgs));
  if (status != 0)
    RETURN_ERROR("the NPU skel refused to start its queue drain: " +
                 qurtStatusMessage(status));
  return std::unique_ptr<NPUQueueChannel>(std::move(channel));
}

} // namespace M::Driver
