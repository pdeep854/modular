//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// The queue transport, driven end to end with no device.
//
// Both halves are the real ones: launches go into the real `NPUQueueTransport`,
// are described into real `dspqueue_buffer` descriptors by the real
// `npuDspQueueDescribeFrame`, and come out of the real `npu_queue_run_packet`,
// the same function the skel's drain loop calls. What sits between them is a
// loopback channel that does what the vendor queue does and nothing else -- it
// hands the packet to the far side and remembers the completion -- so what is
// being tested is the framing, not a model of it.
//
// In particular the packet's descriptor array is the real one, frame at index
// zero and allocations after it. That offset is the seam between the two
// halves: a resolve table that did not skip the frame would map every operand
// one allocation early, which is a wrong render rather than a failed call, and
// it is the one thing a device-free test can still hold both sides to.
//
// The claim under test is the one a transport swap has to earn: a launch that
// travelled over the queue is the launch that would have travelled over the
// blocking invocation. Not "similar" -- the selector, the launch word, every
// byte of the packed scalar block (offsets in its tail included) and every
// operand's base and declared length, in the order they were dispatched.
//
// It fails on the mistakes that matter here rather than on a checksum: a frame
// whose tables are laid out differently, a union that maps a launch's operand
// to the wrong allocation, launches reordered within a frame or across frames,
// a flush that runs launches early or leaves them unrun, and a ring that lets
// more frames out than it is allowed to.
//
//===----------------------------------------------------------------------===//

#include "NPUQueueTransport.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

extern "C" {
#include "mojo_npu_queue.h"
}

namespace {

int failures = 0;

void expect(bool condition, const std::string &what) {
  if (!condition) {
    std::printf("FAIL: %s\n", what.c_str());
    ++failures;
  }
}

/// One launch as the far side saw it. This is the comparison subject: it is
/// what the device trampoline is handed, so equality with what was dispatched
/// is the whole of "the transport changed nothing".
struct SeenLaunch {
  uint32_t selector = 0;
  uint32_t launchWord = 0;
  std::vector<uint8_t> scalars;
  std::vector<const void *> bases;
  std::vector<int> lengths;
};

struct Loopback;

/// The far side's launch runner. Records rather than computes: the kernels are
/// not what this is about, and a recorder is what lets the comparison be exact.
int32_t recordLaunch(void *context, uint32_t selector, uint32_t launchWord,
                     const uint8_t *scalars, int scalarsLen, uint8_t **buffers,
                     const int *lengths);

/// FastRPC descriptors, as this test hands them out. Never opened: the
/// transport only carries a descriptor from `rpcmem_to_fd` to the packet, and
/// what a test can check is that it carries the right one to the right slot.
constexpr int kFrameFD = 900;
int fakeFD(size_t index) { return static_cast<int>(index) + 1; }

/// Stands in for the vendor queue: it delivers frames in order and completes
/// them when told to, which is the only behaviour the transport depends on.
///
/// `deferCompletions` is what makes the ring observable. With it off a frame
/// completes inside `send`, so nothing is ever in flight and the ring is never
/// exercised; with it on, completions are held back until a test releases them,
/// and the transport has to block.
struct Loopback : M::Driver::NPUQueueChannel {
  std::vector<SeenLaunch> seen;
  /// Every frame this queue was handed, kept alive: a deque, because the
  /// descriptors hold pointers into these and a vector would move them.
  std::deque<std::vector<uint8_t>> deliveredFrames;
  std::vector<uint32_t> framesDelivered;
  std::deque<std::pair<uint32_t, int32_t>> completions;
  bool deferCompletions = false;
  uint32_t maxOutstanding = 0;
  uint32_t outstanding = 0;
  /// Fails the launch at this index, counting from the first ever seen.
  int failAtLaunch = -1;
  int32_t frameStatus = 0;
  /// The last packet's descriptors, kept so a test can read what the far side
  /// was actually handed rather than what it was meant to be handed.
  npu_dspqueue_buffer_t descriptors[NPU_QUEUE_MAX_PACKET_BUFFERS] = {};
  size_t descriptorCount = 0;
  /// Mirrors the channel's `MODULAR_NPU_QUEUE_WFLAGS=all` reading.
  int flagAllRefs = 0;

  M::ErrorOrSuccess send(const void *frame, size_t frameBytes,
                         const M::Driver::NPUQueueBufferRef *refs,
                         size_t refCount) override {
    // Copied before it is read, because the vendor queue delivers a frame the
    // sender no longer owns; reading the sender's own block would hide a
    // lifetime bug rather than find one. Kept afterwards for the same reason
    // the channel keeps its staging block until the completion comes back: a
    // packet's descriptor points into it, and the far side is still reading.
    deliveredFrames.emplace_back(static_cast<const uint8_t *>(frame),
                                 static_cast<const uint8_t *>(frame) +
                                     frameBytes);
    std::vector<uint8_t> &delivered = deliveredFrames.back();
    std::vector<int> fds;
    for (size_t i = 0; i < refCount; ++i)
      fds.push_back(fakeFD(i));

    size_t count = M::Driver::npuDspQueueDescribeFrame(
        delivered.data(), delivered.size(), kFrameFD, refs, fds.data(),
        refCount, descriptors, NPU_QUEUE_MAX_PACKET_BUFFERS, flagAllRefs);
    if (count != refCount + 1)
      return M::Error("the frame did not fit the packet's descriptors");
    descriptorCount = count;

    npu_queue_response_t response = {};
    npu_queue_run_packet(descriptors, static_cast<uint32_t>(count),
                         recordLaunch, this, &response);
    frameStatus = response.status;
    framesDelivered.push_back(response.batch_id);

    ++outstanding;
    if (outstanding > maxOutstanding)
      maxOutstanding = outstanding;
    completions.emplace_back(response.batch_id, response.status);
    return M::success();
  }

  M::ErrorOr<bool> receive(uint32_t &batchID, int32_t &status,
                           uint32_t /*timeoutMicros*/) override {
    // What the device drain sends when its own vendor read fails: one
    // message-only completion, batch id zero, `NPU_QUEUE_BAD_READ`.
    if (refuseAsBadRead) {
      batchID = 0;
      status = NPU_QUEUE_BAD_READ;
      return true;
    }
    if (completions.empty())
      return false;
    // Held back until released, so a test can put the transport under its own
    // ring bound and see it block.
    if (deferCompletions && released == 0)
      return false;
    if (released > 0)
      --released;
    batchID = completions.front().first;
    status = completions.front().second;
    completions.pop_front();
    --outstanding;
    return true;
  }

  /// Lets `count` more completions through.
  void release(uint32_t count) { released += count; }

  uint32_t released = 0;
  bool refuseAsBadRead = false;
};

int32_t recordLaunch(void *context, uint32_t selector, uint32_t launchWord,
                     const uint8_t *scalars, int scalarsLen, uint8_t **buffers,
                     const int *lengths) {
  auto *loop = static_cast<Loopback *>(context);
  SeenLaunch launch;
  launch.selector = selector;
  launch.launchWord = launchWord;
  launch.scalars.assign(scalars, scalars + scalarsLen);
  for (uint32_t i = 0; i < NPU_QUEUE_BUFFER_SLOTS; ++i) {
    if (buffers[i] == nullptr)
      continue;
    launch.bases.push_back(buffers[i]);
    launch.lengths.push_back(lengths[i]);
  }
  int index = static_cast<int>(loop->seen.size());
  loop->seen.push_back(std::move(launch));
  return index == loop->failAtLaunch ? -7 : 0;
}

/// A launch built the way `enqueueFunctionExecDirect` builds one: a packed
/// scalar block whose tail carries an 8-byte offset per operand.
struct Dispatched {
  uint32_t selector;
  uint32_t launchWord;
  std::vector<uint8_t> scalars;
  std::vector<M::Driver::NPUQueueBufferRef> buffers;

  M::Driver::NPUQueueLaunch wire() const {
    M::Driver::NPUQueueLaunch launch;
    launch.selector = selector;
    launch.launchWord = launchWord;
    launch.scalars = scalars.data();
    launch.scalarBytes = scalars.size();
    launch.buffers = buffers.data();
    launch.bufferCount = buffers.size();
    return launch;
  }
};

Dispatched
makeLaunch(uint32_t selector, uint32_t launchWord, uint32_t scalarCount,
           const std::vector<M::Driver::NPUQueueBufferRef> &buffers) {
  Dispatched launch;
  launch.selector = selector;
  launch.launchWord = launchWord;
  launch.buffers = buffers;
  launch.scalars.assign((scalarCount + buffers.size()) * 8, 0);
  // Deterministic in the selector and the position, so a block that arrived
  // shifted, truncated or from another launch is not equal to this one.
  for (size_t i = 0; i < launch.scalars.size(); ++i)
    launch.scalars[i] = static_cast<uint8_t>(0x5Au ^ (selector * 31u) ^ i);
  return launch;
}

void expectSame(const Dispatched &sent, const SeenLaunch &got,
                const std::string &what) {
  expect(sent.selector == got.selector, what + ": selector");
  expect(sent.launchWord == got.launchWord, what + ": launch word");
  expect(sent.scalars == got.scalars, what + ": scalar block byte-for-byte");
  expect(sent.buffers.size() == got.bases.size(), what + ": operand count");
  for (size_t i = 0; i < sent.buffers.size() && i < got.bases.size(); ++i) {
    expect(sent.buffers[i].base == got.bases[i],
           what + ": operand " + std::to_string(i) + " base");
    expect(static_cast<int>(sent.buffers[i].byteSize) == got.lengths[i],
           what + ": operand " + std::to_string(i) + " declared length");
  }
}

/// Addresses that are never dereferenced: the transport treats an allocation
/// base as an identity, and the far side only records it.
void *fakeAllocation(uintptr_t index) {
  return reinterpret_cast<void *>(0x100000u * (index + 1));
}

//===----------------------------------------------------------------------===//

/// The headline: a run of launches over shared arenas comes out the far side
/// unchanged and in order, in one frame.
void testRoundTripIsExactAndOrdered() {
  auto channel = std::make_unique<Loopback>();
  Loopback *loop = channel.get();
  M::Driver::NPUQueueTransport transport(std::move(channel));

  M::Driver::NPUQueueBufferRef arena{fakeAllocation(0), 0x40000};
  M::Driver::NPUQueueBufferRef weights{fakeAllocation(1), 0x8000};
  M::Driver::NPUQueueBufferRef witness{fakeAllocation(2), 64};

  std::vector<Dispatched> sent;
  for (uint32_t i = 0; i < 16; ++i)
    sent.push_back(makeLaunch(/*selector=*/i + 3, /*launchWord=*/(i % 4) + 1,
                              /*scalarCount=*/i % 5,
                              {arena, weights, witness}));

  for (const Dispatched &launch : sent)
    expect(!transport.enqueue(launch.wire()).isError(), "enqueue accepted");

  // Nothing has run: an enqueue that dispatched immediately would be the
  // blocking transport wearing a queue's name.
  expect(loop->seen.empty(), "enqueue alone runs nothing");
  expect(transport.framesSent() == 0, "enqueue alone sends no frame");

  expect(!transport.drain().isError(), "drain succeeded");
  expect(transport.framesSent() == 1, "one frame for sixteen launches");
  expect(loop->seen.size() == sent.size(), "every launch arrived");
  for (size_t i = 0; i < sent.size() && i < loop->seen.size(); ++i)
    expectSame(sent[i], loop->seen[i], "launch " + std::to_string(i));
}

/// The union rule: launches over the same allocations declare it once, and an
/// operand still resolves to the allocation its own launch named.
void testUnionCollapsesRepeatedAllocations() {
  auto channel = std::make_unique<Loopback>();
  Loopback *loop = channel.get();
  M::Driver::NPUQueueTransport transport(std::move(channel));

  // Three arenas, each launch naming a different pair, so a frame that
  // deduplicated by position rather than by identity would cross them.
  M::Driver::NPUQueueBufferRef a{fakeAllocation(0), 0x1000};
  M::Driver::NPUQueueBufferRef b{fakeAllocation(1), 0x2000};
  M::Driver::NPUQueueBufferRef c{fakeAllocation(2), 0x3000};

  std::vector<Dispatched> sent = {
      makeLaunch(1, 1, 2, {a, b}),
      makeLaunch(2, 1, 2, {b, c}),
      makeLaunch(3, 1, 2, {c, a}),
      makeLaunch(4, 1, 2, {a, a}),
  };
  for (const Dispatched &launch : sent)
    expect(!transport.enqueue(launch.wire()).isError(), "enqueue accepted");
  expect(!transport.drain().isError(), "drain succeeded");

  expect(transport.framesSent() == 1, "one frame");
  for (size_t i = 0; i < sent.size() && i < loop->seen.size(); ++i)
    expectSame(sent[i], loop->seen[i], "union launch " + std::to_string(i));
}

/// Ordering across frames, which is the property the graph depends on and the
/// one a ring is most likely to break.
void testOrderIsKeptAcrossFrames() {
  auto channel = std::make_unique<Loopback>();
  Loopback *loop = channel.get();
  M::Driver::NPUQueueTransport transport(std::move(channel));

  // One distinct allocation per launch, so the union fills and the transport
  // has to start a new frame -- more frames than the ring holds.
  std::vector<Dispatched> sent;
  for (uint32_t i = 0; i < NPU_QUEUE_MAX_BUFFERS * 3; ++i) {
    M::Driver::NPUQueueBufferRef own{fakeAllocation(i), 0x1000 + i};
    sent.push_back(makeLaunch(i + 1, 1, /*scalarCount=*/2, {own}));
  }
  for (const Dispatched &launch : sent)
    expect(!transport.enqueue(launch.wire()).isError(), "enqueue accepted");
  expect(!transport.drain().isError(), "drain succeeded");

  expect(transport.framesSent() == 3, "one frame per full union");
  expect(loop->seen.size() == sent.size(), "every launch arrived");
  for (size_t i = 0; i < sent.size() && i < loop->seen.size(); ++i)
    expectSame(sent[i], loop->seen[i], "ordered launch " + std::to_string(i));

  // Frames themselves in order, by the id the ring recycles.
  for (size_t i = 1; i < loop->framesDelivered.size(); ++i)
    expect(loop->framesDelivered[i - 1] < loop->framesDelivered[i],
           "frame " + std::to_string(i) + " delivered after its predecessor");
}

/// The ring bound. Completions are held back, so the transport may not send
/// more than it is allowed to leave outstanding.
void testRingBoundsFramesInFlight() {
  auto channel = std::make_unique<Loopback>();
  Loopback *loop = channel.get();
  loop->deferCompletions = true;
  M::Driver::NPUQueueTransport transport(std::move(channel));

  for (uint32_t frame = 0;
       frame < M::Driver::NPUQueueTransport::kBatchesInFlight; ++frame) {
    M::Driver::NPUQueueBufferRef own{fakeAllocation(frame), 0x1000};
    Dispatched launch = makeLaunch(frame + 1, 1, 1, {own});
    expect(!transport.enqueue(launch.wire()).isError(), "enqueue accepted");
    expect(!transport.flush().isError(), "flush succeeded");
  }
  expect(loop->outstanding == M::Driver::NPUQueueTransport::kBatchesInFlight,
         "the ring filled to its bound");
  expect(loop->maxOutstanding <= M::Driver::NPUQueueTransport::kBatchesInFlight,
         "the ring never exceeded its bound");

  loop->release(M::Driver::NPUQueueTransport::kBatchesInFlight);
  expect(!transport.drain().isError(), "drain succeeded once released");
  expect(loop->outstanding == 0, "drain left nothing in flight");
}

/// A launch that fails stops its frame. Continuing would run kernels over
/// inputs an earlier launch never wrote.
void testFailureStopsTheFrame() {
  auto channel = std::make_unique<Loopback>();
  Loopback *loop = channel.get();
  loop->failAtLaunch = 2;
  M::Driver::NPUQueueTransport transport(std::move(channel));

  M::Driver::NPUQueueBufferRef arena{fakeAllocation(0), 0x1000};
  for (uint32_t i = 0; i < 8; ++i) {
    Dispatched launch = makeLaunch(i + 1, 1, 1, {arena});
    expect(!transport.enqueue(launch.wire()).isError(), "enqueue accepted");
  }
  expect(transport.drain().isError(), "a failed launch fails the drain");
  expect(loop->seen.size() == 3, "the frame stopped at the failing launch");
}

//===----------------------------------------------------------------------===//
// The drain logic on its own: what it must refuse.
//
// These build the frame by hand, because a frame the packer produced is by
// construction one the reader accepts -- and the reader's job is the frames the
// packer did not produce.
//===----------------------------------------------------------------------===//

std::vector<uint8_t> oneLaunchFrame() {
  auto channel = std::make_unique<Loopback>();
  Loopback *loop = channel.get();
  M::Driver::NPUQueueTransport transport(std::move(channel));
  M::Driver::NPUQueueBufferRef arena{fakeAllocation(0), 0x1000};
  Dispatched launch = makeLaunch(5, 1, 2, {arena});
  (void)transport.enqueue(launch.wire());
  (void)transport.drain();
  expect(loop->seen.size() == 1, "the hand-edited frame started out valid");
  // Rebuilt rather than captured, so the bytes edited below are a frame the
  // packer really produces.
  std::vector<uint8_t> frame(
      npu_queue_frame_bytes(1, 1, launch.scalars.size()));
  auto *header = reinterpret_cast<npu_queue_frame_header_t *>(frame.data());
  header->magic = NPU_QUEUE_FRAME_MAGIC;
  header->version = NPU_QUEUE_FRAME_VERSION;
  header->batch_id = 1;
  header->launch_count = 1;
  header->buffer_count = 1;
  header->buffer_table_offset = sizeof(npu_queue_frame_header_t);
  header->launch_table_offset =
      header->buffer_table_offset + sizeof(npu_queue_buffer_t);
  header->scalar_arena_offset =
      header->launch_table_offset + sizeof(npu_queue_launch_t);
  header->scalar_arena_bytes = static_cast<uint32_t>(launch.scalars.size());
  header->frame_bytes = static_cast<uint32_t>(frame.size());
  auto *buffers = reinterpret_cast<npu_queue_buffer_t *>(
      frame.data() + header->buffer_table_offset);
  buffers[0].reference = 0;
  buffers[0].bytes = 0x1000;
  auto *launches = reinterpret_cast<npu_queue_launch_t *>(
      frame.data() + header->launch_table_offset);
  launches[0].selector = 5;
  launches[0].launch_word = 1;
  launches[0].scalar_offset = 0;
  launches[0].scalar_bytes = static_cast<uint32_t>(launch.scalars.size());
  launches[0].buffer_count = 1;
  for (uint32_t i = 0; i < NPU_QUEUE_BUFFER_SLOTS; ++i)
    launches[0].buffer_index[i] = NPU_QUEUE_NO_BUFFER;
  launches[0].buffer_index[0] = 0;
  return frame;
}

/// Runs a hand-built frame the way the drain loop runs one: as the first buffer
/// of a packet whose other descriptors are the frame's allocations. `arrived`
/// is what the queue reports for that first descriptor, which is how a frame
/// shorter than its header claims reaches the reader.
int32_t runBare(std::vector<uint8_t> &frame, Loopback &sink,
                size_t arrived = 0) {
  M::Driver::NPUQueueBufferRef arena{fakeAllocation(0), 0x1000};
  int fd = fakeFD(0);
  npu_dspqueue_buffer_t descriptors[2];
  size_t count = M::Driver::npuDspQueueDescribeFrame(
      frame.data(), arrived == 0 ? frame.size() : arrived, kFrameFD, &arena,
      &fd, 1, descriptors, 2, /*flagAllRefs=*/0);
  expect(count == 2, "a one-allocation frame describes two buffers");
  npu_queue_response_t response = {};
  npu_queue_run_packet(descriptors, static_cast<uint32_t>(count), recordLaunch,
                       &sink, &response);
  return response.status;
}

void testDrainRefusesWhatItCannotTrust() {
  Loopback sink;

  std::vector<uint8_t> good = oneLaunchFrame();
  expect(runBare(good, sink) == NPU_QUEUE_OK, "a well-formed frame runs");
  expect(sink.seen.size() == 1, "and runs exactly its one launch");

  auto headerOf = [](std::vector<uint8_t> &f) {
    return reinterpret_cast<npu_queue_frame_header_t *>(f.data());
  };
  auto launchOf = [&](std::vector<uint8_t> &f) {
    return reinterpret_cast<npu_queue_launch_t *>(
        f.data() + headerOf(f)->launch_table_offset);
  };

  // A skel that decoded a frame it did not understand would read the new layout
  // as the old one, in place, and dispatch from it.
  std::vector<uint8_t> badMagic = oneLaunchFrame();
  headerOf(badMagic)->magic ^= 1u;
  expect(runBare(badMagic, sink) == NPU_QUEUE_BAD_FRAME, "magic is checked");

  std::vector<uint8_t> badVersion = oneLaunchFrame();
  headerOf(badVersion)->version += 1;
  expect(runBare(badVersion, sink) == NPU_QUEUE_BAD_FRAME,
         "version is checked");

  // An unknown frame-wide flag is a batch that means something this reader does
  // not do.
  std::vector<uint8_t> badFlags = oneLaunchFrame();
  headerOf(badFlags)->flags = 1u;
  expect(runBare(badFlags, sink) == NPU_QUEUE_BAD_FRAME,
         "an unknown frame flag is refused, not ignored");

  // The tables must be where the header says and inside the frame.
  std::vector<uint8_t> badLayout = oneLaunchFrame();
  headerOf(badLayout)->scalar_arena_offset += 8;
  expect(runBare(badLayout, sink) == NPU_QUEUE_BAD_FRAME,
         "a shifted table is refused");

  std::vector<uint8_t> truncated = oneLaunchFrame();
  expect(runBare(truncated, sink, truncated.size() - 8) == NPU_QUEUE_BAD_FRAME,
         "a frame longer than what arrived is refused");

  // An operand index past the frame's own union would resolve to whatever
  // followed the table.
  std::vector<uint8_t> badIndex = oneLaunchFrame();
  launchOf(badIndex)->buffer_index[0] = 4;
  expect(runBare(badIndex, sink) == NPU_QUEUE_BAD_BUFFER_INDEX,
         "an out-of-range operand index is refused");

  // Scalars reaching past the arena would read the frame's zero padding and
  // dispatch a kernel with a plausible-looking empty block.
  std::vector<uint8_t> badScalars = oneLaunchFrame();
  launchOf(badScalars)->scalar_bytes += 8;
  expect(runBare(badScalars, sink) == NPU_QUEUE_BAD_SCALARS,
         "scalars past the arena are refused");

  std::vector<uint8_t> badSlots = oneLaunchFrame();
  launchOf(badSlots)->buffer_count = NPU_QUEUE_BUFFER_SLOTS + 1;
  expect(runBare(badSlots, sink) == NPU_QUEUE_BAD_LAUNCH_SLOTS,
         "more operands than the entry point has slots is refused");

  // The control: every refusal above ran nothing beyond the one good frame.
  expect(sink.seen.size() == 1, "a refused frame runs no launch");
}

//===----------------------------------------------------------------------===//
// The packet, as the vendor library will read it.
//===----------------------------------------------------------------------===//

/// Every field of every descriptor, against what the allocation actually is.
/// The vendor dereferences this array, so a field in the wrong place is a wrong
/// pointer and not a failed call -- which is why the layout is asserted at
/// compile time and its contents here.
void testDescriptorsNameTheRightBuffers() {
  auto channel = std::make_unique<Loopback>();
  Loopback *loop = channel.get();
  M::Driver::NPUQueueTransport transport(std::move(channel));

  std::vector<M::Driver::NPUQueueBufferRef> arenas = {
      {fakeAllocation(0), 0x40000},
      {fakeAllocation(1), 0x8000},
      {fakeAllocation(2), 64},
  };
  Dispatched launch = makeLaunch(7, 1, 3, arenas);
  expect(!transport.enqueue(launch.wire()).isError(), "enqueue accepted");
  expect(!transport.drain().isError(), "drain succeeded");

  expect(loop->descriptorCount == arenas.size() + 1,
         "the packet carries one descriptor per allocation, plus the frame");

  // The frame is index zero, and it is a real frame: reading its magic through
  // the descriptor is what says the pointer the vendor would dereference is the
  // pointer the packer wrote.
  const npu_dspqueue_buffer_t &frame = loop->descriptors[0];
  expect(frame.fd == static_cast<uint32_t>(kFrameFD), "the frame names its fd");
  expect(frame.offset == 0, "the frame starts at its allocation's base");
  expect(frame.ptr != nullptr &&
             static_cast<const npu_queue_frame_header_t *>(frame.ptr)->magic ==
                 NPU_QUEUE_FRAME_MAGIC,
         "descriptor zero points at the frame");
  expect(
      frame.size ==
          static_cast<const npu_queue_frame_header_t *>(frame.ptr)->frame_bytes,
      "the frame's descriptor declares the frame's own length");

  for (size_t i = 0; i < arenas.size(); ++i) {
    const npu_dspqueue_buffer_t &got = loop->descriptors[i + 1];
    std::string what = "allocation " + std::to_string(i);
    expect(got.ptr == arenas[i].base, what + " descriptor base");
    expect(got.size == arenas[i].byteSize, what + " descriptor length");
    expect(got.fd == static_cast<uint32_t>(fakeFD(i)), what + " descriptor fd");
    // An offset would double-count: the frame's launches already carry their
    // offset within the allocation in the scalar block's tail, and the device
    // folds it back there.
    expect(got.offset == 0, what + " descriptor offset");
  }
}

/// The cache maintenance, which is the whole of what the frame gives up by not
/// being a FastRPC call. Sender-flush plus recipient-invalidate, and nothing
/// else -- `REF`/`DEREF` refcount a mapping this transport holds for the
/// allocation's lifetime, so taking one per packet would only have to be
/// dropped per packet.
void testDescriptorFlagsAreTheMaintenanceTheDriverStopsDoing() {
  auto channel = std::make_unique<Loopback>();
  Loopback *loop = channel.get();
  M::Driver::NPUQueueTransport transport(std::move(channel));

  M::Driver::NPUQueueBufferRef arena{fakeAllocation(0), 0x1000};
  Dispatched launch = makeLaunch(1, 1, 1, {arena});
  expect(!transport.enqueue(launch.wire()).isError(), "enqueue accepted");
  expect(!transport.drain().isError(), "drain succeeded");

  for (size_t i = 0; i < loop->descriptorCount; ++i) {
    uint32_t flags = loop->descriptors[i].flags;
    std::string what = "descriptor " + std::to_string(i);
    expect((flags & NPU_DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER) != 0,
           what + " flushes the sender's caches");
    expect((flags & NPU_DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT) != 0,
           what + " invalidates the recipient's caches");
    expect((flags & (NPU_DSPQUEUE_BUFFER_FLAG_REF |
                     NPU_DSPQUEUE_BUFFER_FLAG_DEREF)) == 0,
           what + " takes no per-packet reference");
    // The vendor reserves 0x01, 0x02 and everything from 0x100 up, and a
    // reserved bit set is a packet meaning something this side did not intend.
    expect((flags & ~0xFCu) == 0, what + " sets no reserved bit");
  }
}

/// The seam. A resolve table that did not skip the frame would hand every
/// launch the frame's own address in place of its first operand -- and the
/// frame is a real, mapped, plausible pointer, so nothing downstream would
/// fault. It would render.
void testOperandsResolvePastTheFrame() {
  auto channel = std::make_unique<Loopback>();
  Loopback *loop = channel.get();
  M::Driver::NPUQueueTransport transport(std::move(channel));

  M::Driver::NPUQueueBufferRef first{fakeAllocation(0), 0x1000};
  M::Driver::NPUQueueBufferRef second{fakeAllocation(1), 0x2000};
  Dispatched launch = makeLaunch(3, 1, 2, {first, second});
  expect(!transport.enqueue(launch.wire()).isError(), "enqueue accepted");
  expect(!transport.drain().isError(), "drain succeeded");

  expect(loop->seen.size() == 1, "the launch ran");
  if (loop->seen.empty())
    return;
  const void *frame = loop->descriptors[0].ptr;
  for (const void *base : loop->seen[0].bases)
    expect(base != frame, "no operand resolved to the frame itself");
  expect(loop->seen[0].bases.size() == 2 &&
             loop->seen[0].bases[0] == first.base,
         "the first operand is the first allocation, not the frame");
}

/// A packet with more allocations than the descriptor array holds is refused
/// before it is written, not clipped. Clipping would send a frame whose buffer
/// table names allocations the packet never carried.
void testDescriptorCapacityIsRefusedNotClipped() {
  M::Driver::NPUQueueBufferRef refs[2] = {{fakeAllocation(0), 0x1000},
                                          {fakeAllocation(1), 0x2000}};
  int fds[2] = {fakeFD(0), fakeFD(1)};
  npu_dspqueue_buffer_t descriptors[3] = {};
  std::vector<uint8_t> frame(64, 0);

  expect(M::Driver::npuDspQueueDescribeFrame(
             frame.data(), frame.size(), kFrameFD, refs, fds, 2, descriptors, 2,
             /*flagAllRefs=*/0) == 0,
         "two allocations do not fit two descriptors");
  expect(descriptors[0].ptr == nullptr,
         "a refused description writes nothing at all");
  expect(M::Driver::npuDspQueueDescribeFrame(
             frame.data(), frame.size(), kFrameFD, refs, fds, 2, descriptors, 3,
             /*flagAllRefs=*/0) == 3,
         "two allocations fit three descriptors");
}

/// The maintenance split: a declared-constant reference's descriptor carries
/// no flags -- the host wrote it once at init, so per-frame maintenance on it
/// is O(model) bytes for lines that cannot be stale -- while the frame and
/// every host-written reference keep the pair. The wiring under test is the
/// classification's, end to end through the real packer and the real
/// describe: misassigning one reference's class is a wrong descriptor, and a
/// wrong descriptor is silent corruption on silicon, not a failed call.
void testConstantReferencesRideUnflagged() {
  auto channel = std::make_unique<Loopback>();
  Loopback *loop = channel.get();
  M::Driver::NPUQueueTransport transport(std::move(channel));

  M::Driver::NPUQueueBufferRef activations{fakeAllocation(0), 0x1000, false};
  M::Driver::NPUQueueBufferRef weights{fakeAllocation(1), 0x2000, true};
  Dispatched launch = makeLaunch(1, 1, 1, {activations, weights});
  expect(!transport.enqueue(launch.wire()).isError(), "enqueue accepted");
  expect(!transport.drain().isError(), "drain succeeded");
  expect(loop->descriptorCount == 3, "frame plus two references");

  constexpr uint32_t kPair = NPU_DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER |
                             NPU_DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
  expect(loop->descriptors[0].flags == kPair,
         "the frame carries the maintenance pair");
  expect(loop->descriptors[1].flags == kPair,
         "a host-written reference carries the maintenance pair");
  expect(loop->descriptors[2].flags == 0,
         "a declared-constant reference carries no maintenance");
}

/// `MODULAR_NPU_QUEUE_WFLAGS=all`, at the packer level the channel drives it
/// through: the override exists so a device A/B can price what the per-class
/// default saves, and a control leg that silently kept the per-class split
/// would be indistinguishable from its own treatment.
void testWflagsAllOverridesTheClassSplit() {
  auto channel = std::make_unique<Loopback>();
  Loopback *loop = channel.get();
  loop->flagAllRefs = 1;
  M::Driver::NPUQueueTransport transport(std::move(channel));

  M::Driver::NPUQueueBufferRef weights{fakeAllocation(1), 0x2000, true};
  Dispatched launch = makeLaunch(1, 1, 1, {weights});
  expect(!transport.enqueue(launch.wire()).isError(), "enqueue accepted");
  expect(!transport.drain().isError(), "drain succeeded");

  constexpr uint32_t kPair = NPU_DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER |
                             NPU_DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
  expect(loop->descriptors[1].flags == kPair,
         "WFLAGS=all keeps the pair on a constant reference");
}

/// The device import's retry decision and its self-report tokens, which are
/// host-buildable precisely so this test can hold them. Null callbacks bind
/// first; only the reference implementation's named refusal warrants the
/// callback-mode retry; any other refusal is final.
void testImportFallbackDecision() {
  const int kBadParm = 14;
  expect(npu_queue_import_retry(0, kBadParm) == 0,
         "a successful null-callback import stands");
  expect(npu_queue_import_retry(kBadParm, kBadParm) == 1,
         "the named refusal warrants the callback-mode retry");
  expect(npu_queue_import_retry(1 /*AEE_EFAILED*/, kBadParm) == 0,
         "any other refusal is final, not retried in another mode");

  expect(std::string(npu_queue_import_mode_name(0)) == "null-callbacks",
         "the first mode names itself");
  expect(std::string(npu_queue_import_mode_name(1)) == "callback-fallback",
         "the fallback names itself");
}

/// The whole self-report line, built without libc stdio because the device
/// libc's vsnprintf imports a heap/lock/TLS layer the protection domain does
/// not provide -- the formatting FARF alone made the skel unloadable (lane
/// evidence 072). The transport protocol greps these exact strings from the
/// mini-dm capture, so they are pinned byte for byte, %d semantics included.
void testImportMessageIsExactAndBounded() {
  char buffer[NPU_QUEUE_IMPORT_MSG_BYTES];

  int length = npu_queue_import_message(buffer, sizeof(buffer), 0, 0);
  expect(std::string(buffer) == "npu queue import: null-callbacks (rc 0)",
         "the null-callbacks line is byte-exact");
  expect(length == (int)std::strlen(buffer),
         "the returned length is the line's");

  npu_queue_import_message(buffer, sizeof(buffer), 1, 39);
  expect(std::string(buffer) == "npu queue import: callback-fallback (rc 39)",
         "the fallback line is byte-exact");

  /* AEE codes are negative as a 32-bit int, and FARF printed them with %d. */
  npu_queue_import_message(buffer, sizeof(buffer), 1, (int)0x80000406u);
  expect(std::string(buffer) ==
             "npu queue import: callback-fallback (rc -2147482618)",
         "an AEE code keeps its %d rendering");

  length = npu_queue_import_message(buffer, sizeof(buffer), 1, INT32_MIN);
  expect(std::string(buffer) ==
             "npu queue import: callback-fallback (rc -2147483648)",
         "INT_MIN has no special case");
  expect(length < NPU_QUEUE_IMPORT_MSG_BYTES,
         "the longest line fits the declared capacity");

  /* The bound is enforced, not assumed: truncation stays NUL-terminated. */
  char tight[8];
  length = npu_queue_import_message(tight, (int)sizeof(tight), 0, 0);
  expect(std::string(tight) == "npu que" && length == 7,
         "truncation stops at capacity - 1 and terminates");
  expect(npu_queue_import_message(tight, 0, 0, 0) == 0,
         "zero capacity writes nothing");
}

/// The response, which is what lets the host free the right ring slot. A frame
/// the drain refuses still names itself: a completion without a batch id would
/// strand the block that frame was carried in.
void testResponseIdentifiesEvenARefusedFrame() {
  Loopback sink;
  std::vector<uint8_t> frame = oneLaunchFrame();
  const uint32_t batchID =
      reinterpret_cast<npu_queue_frame_header_t *>(frame.data())->batch_id;
  reinterpret_cast<npu_queue_frame_header_t *>(frame.data())->magic ^= 1u;

  M::Driver::NPUQueueBufferRef arena{fakeAllocation(0), 0x1000};
  int fd = fakeFD(0);
  npu_dspqueue_buffer_t descriptors[2];
  size_t count = M::Driver::npuDspQueueDescribeFrame(
      frame.data(), frame.size(), kFrameFD, &arena, &fd, 1, descriptors, 2,
      /*flagAllRefs=*/0);
  npu_queue_response_t response = {};
  npu_queue_run_packet(descriptors, static_cast<uint32_t>(count), recordLaunch,
                       &sink, &response);

  expect(response.status == NPU_QUEUE_BAD_FRAME, "the frame was refused");
  expect(response.batch_id == batchID, "and the response still names it");
}

/// The receive bound, which is what turns the order-074 presentation -- a
/// clean bind, a clean write, then silence -- from an unbounded silent hang
/// into a refusal that names itself, its bound, and the knob. Bounded BY
/// DEFAULT: a channel that never completes must produce an error with no
/// environment set at all.
void testReceiveGivesUpByDefaultAndNamesTheKnob() {
  unsetenv("MODULAR_NPU_QUEUE_RECV_GIVEUP_S");
  auto channel = std::make_unique<Loopback>();
  channel->deferCompletions = true;
  M::Driver::NPUQueueTransport transport(std::move(channel));

  M::Driver::NPUQueueBufferRef arena{fakeAllocation(0), 0x1000};
  Dispatched launch = makeLaunch(1, 1, 1, {arena});
  expect(!transport.enqueue(launch.wire()).isError(), "enqueue accepted");
  M::ErrorOrSuccess drained = transport.drain();
  expect(drained.isError(), "a device that never answers fails the drain");
  if (drained.isError()) {
    std::string message = drained.getError();
    expect(message.find("MODULAR_NPU_QUEUE_RECV_GIVEUP_S") != std::string::npos,
           "the refusal names the knob");
    expect(message.find("120 s") != std::string::npos,
           "the default bound is 120 s and the refusal says so");
    expect(message.find("queue report") != std::string::npos,
           "the refusal points at the drain's own account");
  }

  setenv("MODULAR_NPU_QUEUE_RECV_GIVEUP_S", "7", 1);
  auto tuned = std::make_unique<Loopback>();
  tuned->deferCompletions = true;
  M::Driver::NPUQueueTransport tunedTransport(std::move(tuned));
  expect(!tunedTransport.enqueue(launch.wire()).isError(), "enqueue accepted");
  M::ErrorOrSuccess tunedDrain = tunedTransport.drain();
  expect(tunedDrain.isError() &&
             std::string(tunedDrain.getError()).find("7 s") !=
                 std::string::npos,
         "the bound is the environment's, and the refusal says which");
  unsetenv("MODULAR_NPU_QUEUE_RECV_GIVEUP_S");
}

/// The drain's read-failure refusal, host side: a completion with batch id
/// zero and `NPU_QUEUE_BAD_READ` is the device saying "I received a packet I
/// cannot read", and it must be reported as that -- not as a frame this
/// transport did not send, and above all not dropped back into the wait.
void testDrainReadRefusalIsNamed() {
  auto channel = std::make_unique<Loopback>();
  channel->refuseAsBadRead = true;
  M::Driver::NPUQueueTransport transport(std::move(channel));

  M::Driver::NPUQueueBufferRef arena{fakeAllocation(0), 0x1000};
  Dispatched launch = makeLaunch(1, 1, 1, {arena});
  expect(!transport.enqueue(launch.wire()).isError(), "enqueue accepted");
  M::ErrorOrSuccess drained = transport.drain();
  expect(drained.isError(), "a drain refusal fails the drain");
  if (drained.isError()) {
    std::string message = drained.getError();
    expect(message.find("refused a packet") != std::string::npos,
           "the refusal is reported as the drain's");
    expect(message.find("vendor read failed") != std::string::npos,
           "the refusal says what failed on the device");
  }
}

/// The post-mortem report's word layout, held host-side so a device leg reads
/// words whose meaning a test already pinned. These are the words that
/// separate the 074 candidates: the read-rc pair, the packet's shape, and the
/// last frame's magic/version/batch words.
void testReportWordsPinTheWireLayout() {
  npu_queue_report_reset();
  uint32_t words[NPU_QUEUE_REPORT_WORDS] = {};
  mojo_npu_queue_report(words, NPU_QUEUE_REPORT_WORDS);
  expect(words[NPU_QUEUE_REPORT_MAGIC_WORD] == NPU_QUEUE_REPORT_MAGIC,
         "a reset report leads with the magic");
  expect(words[NPU_QUEUE_REPORT_STATE] == NPU_QUEUE_STATE_UNSTARTED,
         "a reset report has not started");

  // Consecutive same-rc reads count up; a different rc restarts the count.
  npu_queue_report_note_read_error(72);
  npu_queue_report_note_read_error(72);
  npu_queue_report_note_read_error(14);
  mojo_npu_queue_report(words, NPU_QUEUE_REPORT_WORDS);
  expect(words[NPU_QUEUE_REPORT_LAST_READ_RC] == 14u,
         "the last read rc is the latest");
  expect(words[NPU_QUEUE_REPORT_READ_RC_REPEATS] == 1u,
         "a new rc restarts the repeat count");

  // A received packet leaves its shape and the frame header words behind,
  // recorded before the frame runs.
  std::vector<uint8_t> frame = oneLaunchFrame();
  M::Driver::NPUQueueBufferRef arena{fakeAllocation(0), 0x1000};
  int fd = fakeFD(0);
  npu_dspqueue_buffer_t descriptors[2];
  size_t count = M::Driver::npuDspQueueDescribeFrame(
      frame.data(), frame.size(), kFrameFD, &arena, &fd, 1, descriptors, 2,
      /*flagAllRefs=*/0);
  npu_queue_report_note_packet(0x2, static_cast<uint32_t>(count), 0,
                               descriptors);
  mojo_npu_queue_report(words, NPU_QUEUE_REPORT_WORDS);
  expect(words[NPU_QUEUE_REPORT_FRAMES_SEEN] == 1u, "the packet was counted");
  expect(words[NPU_QUEUE_REPORT_LAST_PACKET_BUFFERS] == 2u,
         "the packet's descriptor count was recorded");
  expect(words[NPU_QUEUE_REPORT_LAST_PACKET_FRAME_BYTES] == frame.size(),
         "the frame descriptor's size was recorded");
  expect(words[NPU_QUEUE_REPORT_FRAME_PTR_CLASS] == 2u,
         "the frame pointer resolved and was classified readable");
  expect(words[NPU_QUEUE_REPORT_LAST_FRAME_MAGIC] == NPU_QUEUE_FRAME_MAGIC,
         "the frame's magic was read back");
  expect(words[NPU_QUEUE_REPORT_LAST_FRAME_VERSION] == NPU_QUEUE_FRAME_VERSION,
         "the frame's version was read back");
  expect(words[NPU_QUEUE_REPORT_LAST_FRAME_BATCH] == 1u,
         "the frame's batch id was read back");

  // A frame descriptor that resolved to null is classified, not chased.
  descriptors[0].ptr = nullptr;
  npu_queue_report_note_packet(0x2, static_cast<uint32_t>(count), 0,
                               descriptors);
  mojo_npu_queue_report(words, NPU_QUEUE_REPORT_WORDS);
  expect(words[NPU_QUEUE_REPORT_FRAME_PTR_CLASS] == 1u,
         "a null frame pointer is class 1");

  // A short reader truncates rather than overruns.
  uint32_t small[6] = {0, 0, 0, 0, 0xDEADBEEFu, 0xDEADBEEFu};
  mojo_npu_queue_report(small, 4);
  expect(small[0] == NPU_QUEUE_REPORT_MAGIC && small[4] == 0xDEADBEEFu &&
             small[5] == 0xDEADBEEFu,
         "a four-word capacity writes four words and no more");
}

/// A scripted vendor for `npu_queue_service_step`: each call to read pops the
/// next scripted answer, and every write is captured whole. This is the
/// callback-mode drain's world -- the pass has no thread and no sleep, only
/// the vendor's answers -- so what these tests hold is exactly what the
/// packet callback runs on the device.
struct ScriptedVendor {
  struct Read {
    int rc = 0;
    std::vector<npu_dspqueue_buffer_t> descriptors;
  };
  struct Write {
    uint32_t flags = 0;
    std::vector<npu_dspqueue_buffer_t> descriptors;
    npu_queue_response_t response = {};
    int rc = 0;
  };
  std::deque<Read> reads;
  std::vector<Write> writes;
  int writeRC = 0;
  int readCalls = 0;

  static int read(void *queue, uint32_t *flags, uint32_t maxBuffers,
                  uint32_t *numBuffers, npu_dspqueue_buffer_t *buffers,
                  uint32_t maxMessageBytes, uint32_t *messageBytes,
                  uint8_t *message) {
    (void)maxMessageBytes;
    (void)message;
    auto *self = static_cast<ScriptedVendor *>(queue);
    ++self->readCalls;
    if (self->reads.empty())
      return NPU_DSPQUEUE_EWOULDBLOCK;
    Read next = self->reads.front();
    self->reads.pop_front();
    if (next.rc != 0)
      return next.rc;
    if (next.descriptors.size() > maxBuffers)
      return 14 /*AEE_EBADPARM, host numbering*/;
    *flags = NPU_DSPQUEUE_PACKET_FLAG_BUFFERS;
    *numBuffers = static_cast<uint32_t>(next.descriptors.size());
    for (size_t i = 0; i < next.descriptors.size(); ++i)
      buffers[i] = next.descriptors[i];
    *messageBytes = 0;
    return 0;
  }

  static int write(void *queue, uint32_t flags, uint32_t numBuffers,
                   npu_dspqueue_buffer_t *buffers, uint32_t messageBytes,
                   const uint8_t *message, uint32_t timeoutMicros) {
    (void)timeoutMicros;
    auto *self = static_cast<ScriptedVendor *>(queue);
    Write seen;
    seen.flags = flags;
    seen.descriptors.assign(buffers, buffers + numBuffers);
    if (messageBytes == sizeof(npu_queue_response_t))
      memcpy(&seen.response, message, sizeof(seen.response));
    seen.rc = self->writeRC;
    self->writes.push_back(seen);
    return self->writeRC;
  }

  npu_queue_channel_ops_t ops() {
    npu_queue_channel_ops_t table;
    table.queue = this;
    table.read_noblock = &ScriptedVendor::read;
    table.write = &ScriptedVendor::write;
    table.sample_pending = nullptr;
    return table;
  }
};

/// The callback-mode service pass, end to end: a packet in, the frame run, a
/// completion out with the request's own per-descriptor flags echoed, then
/// EMPTY when the queue has nothing more. One pass per packet is the whole of
/// what the device's packet callback loops over.
void testServiceStepServicesAPacketAndAnswersIt() {
  npu_queue_report_reset();
  Loopback sink;
  std::vector<uint8_t> frame = oneLaunchFrame();
  const uint32_t batchID =
      reinterpret_cast<npu_queue_frame_header_t *>(frame.data())->batch_id;

  M::Driver::NPUQueueBufferRef arena{fakeAllocation(0), 0x1000, true};
  int fd = fakeFD(0);
  ScriptedVendor vendor;
  ScriptedVendor::Read packet;
  packet.descriptors.resize(2);
  size_t count = M::Driver::npuDspQueueDescribeFrame(
      frame.data(), frame.size(), kFrameFD, &arena, &fd, 1,
      packet.descriptors.data(), 2, /*flagAllRefs=*/0);
  expect(count == 2, "the scripted packet describes two buffers");
  vendor.reads.push_back(packet);

  npu_queue_channel_ops_t ops = vendor.ops();
  int rc = npu_queue_service_step(&ops, recordLaunch, &sink);
  expect(rc == NPU_QUEUE_SERVICE_SERVICED, "the pass consumed the packet");
  expect(sink.seen.size() == 1, "and ran its one launch");
  expect(vendor.writes.size() == 1, "one completion went back");
  if (vendor.writes.size() == 1 && vendor.writes[0].descriptors.size() == 2) {
    expect(vendor.writes[0].response.batch_id == batchID,
           "the completion names the frame");
    expect(vendor.writes[0].response.status == NPU_QUEUE_OK,
           "and reports its status");
    expect(vendor.writes[0].descriptors[0].flags ==
               NPU_DSPQUEUE_TRANSPORT_BUFFER_FLAGS,
           "the frame keeps its maintenance pair on the way back");
    expect(vendor.writes[0].descriptors[1].flags == 0,
           "a constant reference rides back bare, echoed not re-flagged");
  } else {
    expect(false, "the completion carried the packet's two buffers");
  }

  rc = npu_queue_service_step(&ops, recordLaunch, &sink);
  expect(rc == NPU_QUEUE_SERVICE_EMPTY, "an empty queue reads empty");
  expect(npu_queue_report_get(NPU_QUEUE_REPORT_FRAMES_SEEN) == 1u,
         "the report counted the one frame");
}

/// The refusal-completion guarantee, which 076 proved matters in callback
/// mode: a read the vendor refuses outright still answers the host once --
/// message-only `NPU_QUEUE_BAD_READ` -- and then latches, so no later pass
/// touches the vendor again. The latch is the report's own state word.
void testServiceStepRefusesOnceAndLatches() {
  npu_queue_report_reset();
  Loopback sink;
  ScriptedVendor vendor;
  ScriptedVendor::Read refused;
  refused.rc = 13 /*AEE_EBADSTATE, host numbering*/;
  vendor.reads.push_back(refused);

  npu_queue_channel_ops_t ops = vendor.ops();
  int rc = npu_queue_service_step(&ops, recordLaunch, &sink);
  expect(rc == NPU_QUEUE_SERVICE_FATAL, "a refused read latches the pass");
  expect(vendor.writes.size() == 1, "exactly one refusal went back");
  if (vendor.writes.size() == 1) {
    expect(vendor.writes[0].response.status == NPU_QUEUE_BAD_READ,
           "and it is the BAD_READ refusal");
    expect(vendor.writes[0].descriptors.empty(),
           "the refusal is message-only: the buffers are what could not be "
           "read");
  }
  expect(npu_queue_report_get(NPU_QUEUE_REPORT_STATE) ==
             NPU_QUEUE_STATE_READ_FATAL,
         "the latch is the report's state word");

  int readsBefore = vendor.readCalls;
  rc = npu_queue_service_step(&ops, recordLaunch, &sink);
  expect(rc == NPU_QUEUE_SERVICE_FATAL, "a latched pass stays latched");
  expect(vendor.readCalls == readsBefore, "and never touches the vendor again");
  expect(vendor.writes.size() == 1, "no second refusal is sent");
}

/// A completion that cannot be written latches too: the host will wait on the
/// frame forever, and the report plus the host's bounded receive are the only
/// channels left to say so.
void testServiceStepLatchesWhenTheAnswerCannotLeave() {
  npu_queue_report_reset();
  Loopback sink;
  std::vector<uint8_t> frame = oneLaunchFrame();
  M::Driver::NPUQueueBufferRef arena{fakeAllocation(0), 0x1000};
  int fd = fakeFD(0);
  ScriptedVendor vendor;
  vendor.writeRC = 13 /*AEE_EBADSTATE, host numbering*/;
  ScriptedVendor::Read packet;
  packet.descriptors.resize(2);
  M::Driver::npuDspQueueDescribeFrame(frame.data(), frame.size(), kFrameFD,
                                      &arena, &fd, 1, packet.descriptors.data(),
                                      2,
                                      /*flagAllRefs=*/0);
  vendor.reads.push_back(packet);

  npu_queue_channel_ops_t ops = vendor.ops();
  int rc = npu_queue_service_step(&ops, recordLaunch, &sink);
  expect(rc == NPU_QUEUE_SERVICE_FATAL, "an undeliverable completion latches");
  expect(npu_queue_report_get(NPU_QUEUE_REPORT_STATE) ==
             NPU_QUEUE_STATE_RESPONSE_FATAL,
         "as the response-fatal state");
  expect(sink.seen.size() == 1,
         "the frame itself ran; it is the answer that could not leave");
}

} // namespace

int main() {
  testRoundTripIsExactAndOrdered();
  testUnionCollapsesRepeatedAllocations();
  testOrderIsKeptAcrossFrames();
  testRingBoundsFramesInFlight();
  testFailureStopsTheFrame();
  testDrainRefusesWhatItCannotTrust();
  testDescriptorsNameTheRightBuffers();
  testDescriptorFlagsAreTheMaintenanceTheDriverStopsDoing();
  testOperandsResolvePastTheFrame();
  testDescriptorCapacityIsRefusedNotClipped();
  testResponseIdentifiesEvenARefusedFrame();
  testConstantReferencesRideUnflagged();
  testWflagsAllOverridesTheClassSplit();
  testImportFallbackDecision();
  testImportMessageIsExactAndBounded();
  testReceiveGivesUpByDefaultAndNamesTheKnob();
  testDrainReadRefusalIsNamed();
  testReportWordsPinTheWireLayout();
  testServiceStepServicesAPacketAndAnswersIt();
  testServiceStepRefusesOnceAndLatches();
  testServiceStepLatchesWhenTheAnswerCannotLeave();

  if (failures != 0) {
    std::printf("%d check(s) failed\n", failures);
    return 1;
  }
  std::printf("all checks passed\n");
  return 0;
}
