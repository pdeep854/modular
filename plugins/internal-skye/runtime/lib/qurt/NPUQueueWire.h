//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// The frame a batch of NPU launches travels in when the transport is a
// shared-memory queue rather than one FastRPC invocation per launch.
//
// Read by both halves -- the host packer in NPUQueueTransport.cpp and the
// device drain loop in mojo_npu_queue.c -- from this one file, the way
// NPUKernels.def is read by both, and staged beside the skel sources by
// examples/hexagon-ondevice/build.sh for the same reason. Two independent
// restatements of a layout the DSP reads *in place* is a class of bug no test
// finds until a render is silently wrong.
//
// The frame carries a launch tuple of exactly the shape the synchronous path
// packs into the generic entry point's IDL slots today: selector, launch word,
// the packed scalar block with the per-buffer offsets already in its tail, and
// one buffer reference per operand. Nothing about a launch changes because it
// arrived over a queue -- in particular the scalar block is copied verbatim, so
// the device folds offsets back with the same `npu_apply_buffer_offsets` it
// always did, and the witness/timing slot keeps its position after the kernel's
// own buffers.
//
// Tensor bytes never travel in the frame. A buffer is named by its index into
// the message's own buffer references, which the queue maps once; the frame
// carries the index and the allocation's length, and the DSP resolves the base.
//
//===----------------------------------------------------------------------===//

#ifndef SKYE_RUNTIME_QURT_NPUQUEUEWIRE_H
#define SKYE_RUNTIME_QURT_NPUQUEUEWIRE_H

#include <stdint.h>

/// Leads every frame. A skel built before a layout change decodes the magic and
/// the version and refuses, rather than reading the new layout as the old one:
/// the frame is read in place out of shared memory, so a mismatch that is not
/// refused is a wrong render and not a failed call.
#define NPU_QUEUE_FRAME_MAGIC 0x4E505551u /* "NPUQ" */
#define NPU_QUEUE_FRAME_VERSION 1u

/// Buffer slots one launch may name, matching the generic entry point's slot
/// count so a launch's operand list is unchanged by the transport.
#define NPU_QUEUE_BUFFER_SLOTS 8

/// Absent operand, in a launch's slot table.
#define NPU_QUEUE_NO_BUFFER 0xFFFFu

/// Launches one frame may carry. Not a tuning knob: it is what bounds the
/// frame's own descriptor block, which is pinned per in-flight batch. The
/// longest run a real sequence produces is far below it -- 398 launches for a
/// whole denoise step, 20,326 for the VAE stage -- so this caps the pinned
/// block rather than the schedule.
#define NPU_QUEUE_MAX_LAUNCHES 1024

/// Distinct allocations one frame may reference. This is the union of its
/// launches' operands, the same rule the batch planner replays, and it must
/// equal the device-side resolve table's size or a frame can name a buffer the
/// drain loop has no room for.
#define NPU_QUEUE_MAX_BUFFERS 64

/// Buffer references one queue packet carries: the frame's own allocations,
/// plus one for the frame itself.
///
/// The frame does not travel as the packet's message. A message is copied into
/// the queue's own memory and a full frame is a quarter of a megabyte, so it
/// rides as a buffer reference like any other allocation -- attached first, at
/// packet index 0. That offset is the whole reason this constant is not
/// `NPU_QUEUE_MAX_BUFFERS`: a reader that sized its descriptor array by the
/// frame's buffer count would have the vendor's read refuse a full frame with
/// `AEE_EBADPARM`, and one that forgot the shift would resolve every operand
/// one allocation early.
#define NPU_QUEUE_MAX_PACKET_BUFFERS (NPU_QUEUE_MAX_BUFFERS + 1)

/// Every descriptor is a whole number of 8-byte words so the DSP can read the
/// frame in place at natural alignment. Checked on both sides.
#define NPU_QUEUE_ALIGN 8u

/// Why the drain refused a frame, when it did. Part of the wire rather than of
/// either side: these travel back as a launch's status, so the host renders
/// them, and rendering one through the vendor's error enum would send a reader
/// looking for a check this side wrote. Negative, so they cannot collide with
/// an `AEEResult` a kernel returned.
enum {
  NPU_QUEUE_OK = 0,
  NPU_QUEUE_BAD_FRAME = -1,
  NPU_QUEUE_BAD_BUFFER_INDEX = -2,
  NPU_QUEUE_BAD_SCALARS = -3,
  NPU_QUEUE_BAD_LAUNCH_SLOTS = -4,
  /// The drain's own vendor read failed with something other than
  /// would-block, so the packet at the head of the queue cannot be read on
  /// the device at all. Travels back as a message-only completion with batch
  /// id zero -- there is no frame to echo -- because a packet the drain
  /// received and cannot answer is otherwise an unbounded silent host wait
  /// (the order-074 presentation). The vendor rc itself is in the queue
  /// report, not here: a completion carries one status and the host renders
  /// wire codes, not the DSP's AEE numbering.
  NPU_QUEUE_BAD_READ = -5
};

/// The drain's post-mortem account, readable from the host over the
/// synchronous transport (the `queue_report` selector) after the queue has
/// gone silent -- which is exactly when the queue itself cannot carry it.
/// Plain uint32 words in skel .bss, the `power_vote` report-buffer pattern
/// and deliberately not the witness protocol.
///
/// One device leg separates the 074 candidates with it: the import mode and
/// rc plus the echoed queue id close the import-mode and id-truncation
/// questions; the read-rc words split "the packet never became visible"
/// (would-block forever, pending 0) from "the packet arrived and the vendor
/// read refuses it" (ENOSUCHMAP/EBADPARM class, pending >= 1); the last
/// frame's magic/version/parse words split a stale-cache frame from a
/// framing mismatch.
#define NPU_QUEUE_REPORT_MAGIC 0x4E505152u /* "NPQR" */
enum {
  NPU_QUEUE_STATE_UNSTARTED = 0,
  NPU_QUEUE_STATE_IMPORT_FAILED = 1,
  NPU_QUEUE_STATE_RUNNING = 2,
  NPU_QUEUE_STATE_STOPPED = 3,
  /// The drain latched after a vendor read failed; it answered with one
  /// `NPU_QUEUE_BAD_READ` refusal and parked until `stop`.
  NPU_QUEUE_STATE_READ_FATAL = 4,
  /// The drain latched after a response write failed: completions can no
  /// longer reach the host, so running further frames would only hide that.
  NPU_QUEUE_STATE_RESPONSE_FATAL = 5,
  NPU_QUEUE_STATE_THREAD_FAILED = 6
};
enum {
  NPU_QUEUE_REPORT_MAGIC_WORD = 0,
  NPU_QUEUE_REPORT_STATE = 1,
  /// 0 null-callbacks, 1 callback-fallback -- the line no host leg could
  /// capture without a mini-dm, now a word it reads back.
  NPU_QUEUE_REPORT_IMPORT_MODE = 2,
  NPU_QUEUE_REPORT_IMPORT_RC = 3,
  /// The id `start` decoded, echoed so the host can compare it to the one it
  /// sent: equal words end the uint64-narrowing question without a FARF.
  NPU_QUEUE_REPORT_QUEUE_ID_LO = 4,
  NPU_QUEUE_REPORT_QUEUE_ID_HI = 5,
  NPU_QUEUE_REPORT_WAKES = 6,
  NPU_QUEUE_REPORT_EMPTY_READS = 7,
  NPU_QUEUE_REPORT_FRAMES_SEEN = 8,
  NPU_QUEUE_REPORT_LAST_READ_RC = 9,
  NPU_QUEUE_REPORT_READ_RC_REPEATS = 10,
  NPU_QUEUE_REPORT_LAST_PACKET_FLAGS = 11,
  NPU_QUEUE_REPORT_LAST_PACKET_BUFFERS = 12,
  NPU_QUEUE_REPORT_LAST_PACKET_MESSAGE_BYTES = 13,
  NPU_QUEUE_REPORT_LAST_PACKET_FRAME_BYTES = 14,
  /// 0 no packet yet, 1 the frame descriptor resolved to a null local
  /// address, 2 it resolved to an address this side can read.
  NPU_QUEUE_REPORT_FRAME_PTR_CLASS = 15,
  NPU_QUEUE_REPORT_LAST_FRAME_MAGIC = 16,
  NPU_QUEUE_REPORT_LAST_FRAME_VERSION = 17,
  NPU_QUEUE_REPORT_LAST_FRAME_BATCH = 18,
  /// `npu_queue_run_packet`'s status for the last packet, as a bit pattern.
  NPU_QUEUE_REPORT_LAST_PARSE_STATUS = 19,
  NPU_QUEUE_REPORT_RESPONSES_WRITTEN = 20,
  NPU_QUEUE_REPORT_LAST_RESPONSE_RC = 21,
  NPU_QUEUE_REPORT_REFUSALS_SENT = 22,
  /// `dspqueue_get_stat(READ_QUEUE_PACKETS)`, sampled on a read error, on
  /// `stop`, and periodically while the queue reads empty. Pending >= 1 under
  /// would-block-forever is a packet the signalling never surfaced; pending 0
  /// is a packet that never reached this side's queue at all. Valid only when
  /// the sample count below is nonzero -- the stat entry is bound weak, since
  /// its residency in this protection domain is unproven (072 load class).
  NPU_QUEUE_REPORT_PENDING_PACKETS = 23,
  NPU_QUEUE_REPORT_PENDING_SAMPLES = 24,
  /// The vendor's error callback, recorded and counted but never acted on:
  /// the reference implementation's own handler is a log line with the
  /// comment "No errors expected on the DSP" (`main.c:649-652`), and a
  /// callback-context teardown would race the very thread that called us.
  NPU_QUEUE_REPORT_ERROR_CB_RC = 25,
  NPU_QUEUE_REPORT_ERROR_CB_COUNT = 26,
  NPU_QUEUE_REPORT_WORDS = 28
};

/// One frame: a header, then the buffer table, the launch table, and the scalar
/// arena, each at a byte offset the header states rather than at a position a
/// reader recomputes. Stating the offsets is what lets a later version append a
/// table without moving the ones before it.
typedef struct {
  uint32_t magic;
  uint32_t version;
  /// Identifies the frame in the response, and is what the host's in-flight
  /// ring recycles. Monotonic within a session.
  uint32_t batch_id;
  uint32_t launch_count;
  uint32_t buffer_count;
  /// Reserved for frame-wide behaviour. Zero today; a reader must refuse bits
  /// it does not know, because an ignored flag is a silently different batch.
  uint32_t flags;
  uint32_t buffer_table_offset;
  uint32_t launch_table_offset;
  uint32_t scalar_arena_offset;
  uint32_t scalar_arena_bytes;
  uint32_t frame_bytes;
  uint32_t reserved;
} npu_queue_frame_header_t;

/// One distinct allocation the frame's launches operate on. `reference` indexes
/// the queue message's own buffer references, which is what carries the mapping
/// and the cache-maintenance flags; `bytes` is the allocation's length exactly
/// as the synchronous path declares it in the entry point's primary-input
/// block, so a device-side length check reads the same number either way.
typedef struct {
  uint32_t reference;
  uint32_t bytes;
} npu_queue_buffer_t;

/// One launch. `launch_word` is the entry point's second argument verbatim --
/// participant mask in the low byte, instrumentation bits above it -- so the
/// witness protocol, the timing report and the skip-compute stage all behave as
/// they do on the synchronous path.
///
/// `status` is written by the drain loop, not the host: a frame is also the
/// response, so a per-launch result needs no second table.
typedef struct {
  uint32_t selector;
  uint32_t launch_word;
  uint32_t scalar_offset;
  uint32_t scalar_bytes;
  uint32_t buffer_count;
  int32_t status;
  uint16_t buffer_index[NPU_QUEUE_BUFFER_SLOTS];
} npu_queue_launch_t;

/// The response packet's message: which frame finished, and how.
///
/// The batch id travels back rather than being inferred from the response's
/// buffer references, because inference is what makes a lost or reordered
/// completion look like a successful one. The host matches this against its own
/// ring and refuses an id it did not send.
typedef struct {
  uint32_t batch_id;
  int32_t status;
} npu_queue_response_t;

/// Bytes a frame occupies for the given counts. The one place the layout is
/// computed, so the packer and the reader cannot disagree about where a table
/// ends.
static inline uint32_t npu_queue_frame_bytes(uint32_t launch_count,
                                             uint32_t buffer_count,
                                             uint32_t scalar_arena_bytes) {
  uint32_t total = (uint32_t)sizeof(npu_queue_frame_header_t);
  total += buffer_count * (uint32_t)sizeof(npu_queue_buffer_t);
  total += launch_count * (uint32_t)sizeof(npu_queue_launch_t);
  total += scalar_arena_bytes;
  return (total + NPU_QUEUE_ALIGN - 1u) & ~(NPU_QUEUE_ALIGN - 1u);
}

/// Holds when a frame's tables are inside it and in the order the packer wrote
/// them. Every field a reader indexes by is checked here once, so the read loop
/// can be a loop rather than a chain of bounds tests.
static inline int npu_queue_frame_valid(const npu_queue_frame_header_t *header,
                                        uint32_t received_bytes) {
  if (header == 0 || received_bytes < sizeof(npu_queue_frame_header_t))
    return 0;
  if (header->magic != NPU_QUEUE_FRAME_MAGIC ||
      header->version != NPU_QUEUE_FRAME_VERSION || header->flags != 0)
    return 0;
  if (header->launch_count == 0 ||
      header->launch_count > NPU_QUEUE_MAX_LAUNCHES)
    return 0;
  if (header->buffer_count > NPU_QUEUE_MAX_BUFFERS)
    return 0;
  if (header->frame_bytes > received_bytes)
    return 0;
  if (header->frame_bytes != npu_queue_frame_bytes(header->launch_count,
                                                   header->buffer_count,
                                                   header->scalar_arena_bytes))
    return 0;
  if ((header->buffer_table_offset | header->launch_table_offset |
       header->scalar_arena_offset) &
      (NPU_QUEUE_ALIGN - 1u))
    return 0;
  if (header->buffer_table_offset != sizeof(npu_queue_frame_header_t))
    return 0;
  if (header->launch_table_offset !=
      header->buffer_table_offset +
          header->buffer_count * (uint32_t)sizeof(npu_queue_buffer_t))
    return 0;
  if (header->scalar_arena_offset !=
      header->launch_table_offset +
          header->launch_count * (uint32_t)sizeof(npu_queue_launch_t))
    return 0;
  return header->scalar_arena_offset + header->scalar_arena_bytes <=
         header->frame_bytes;
}

#endif // SKYE_RUNTIME_QURT_NPUQUEUEWIRE_H
