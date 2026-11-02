/* The device side of the queue transport.
 *
 * Two halves, deliberately separated by what they need to run:
 *
 *  - `npu_queue_run_packet` and `npu_queue_run_frame` resolve a packet's buffer
 *    descriptors and run the frame's launches. Plain memory, no QuRT, no vendor
 *    queue; they build and are tested on a host.
 *  - the resident drain loop, which imports the queue the host exported, waits
 *    on it, hands each frame to the function above, and writes the response.
 *    It needs the device and is compiled only there.
 *
 * The launch a frame carries is the launch the synchronous entry point carries:
 * both call `npu_run_launch`. Nothing about a kernel, its witness slot, its
 * timing report or its participant fan-out changes with the transport.
 */

#include "mojo_npu_queue.h"

/* One launch's slots, resolved from the frame's buffer table. A launch names at
   most the entry point's slot count, so this is a fixed array rather than an
   allocation -- the protection domain's C library has no heap. */
typedef struct {
  uint8_t *buffers[NPU_QUEUE_BUFFER_SLOTS];
  int lengths[NPU_QUEUE_BUFFER_SLOTS];
} npu_queue_slots_t;

int32_t npu_queue_run_frame(void *frame, uint32_t received_bytes,
                            uint8_t *const *bases, uint32_t base_count,
                            npu_queue_launch_fn run, void *context) {
  npu_queue_frame_header_t *header = (npu_queue_frame_header_t *)frame;
  if (!npu_queue_frame_valid(header, received_bytes))
    return NPU_QUEUE_BAD_FRAME;
  /* The host's union and this side's resolved table have to be the same table.
     A frame naming more buffers than were resolved is refused rather than
     clamped: a clamp would run the launches whose operands did resolve. */
  if (header->buffer_count > base_count || bases == 0 || run == 0)
    return NPU_QUEUE_BAD_FRAME;

  uint8_t *bytes = (uint8_t *)frame;
  const npu_queue_buffer_t *buffers =
      (const npu_queue_buffer_t *)(bytes + header->buffer_table_offset);
  npu_queue_launch_t *launches =
      (npu_queue_launch_t *)(bytes + header->launch_table_offset);
  const uint8_t *arena = bytes + header->scalar_arena_offset;

  for (uint32_t i = 0; i < header->launch_count; ++i) {
    npu_queue_launch_t *launch = &launches[i];

    if (launch->buffer_count > NPU_QUEUE_BUFFER_SLOTS) {
      launch->status = NPU_QUEUE_BAD_LAUNCH_SLOTS;
      return launch->status;
    }
    /* Checked against the arena rather than against the frame: the arena's end
       is where the padding starts, and a launch reading into padding would read
       zeroes and dispatch a kernel with a plausible-looking empty block. */
    if (launch->scalar_offset > header->scalar_arena_bytes ||
        launch->scalar_bytes >
            header->scalar_arena_bytes - launch->scalar_offset) {
      launch->status = NPU_QUEUE_BAD_SCALARS;
      return launch->status;
    }

    npu_queue_slots_t slots;
    for (uint32_t slot = 0; slot < NPU_QUEUE_BUFFER_SLOTS; ++slot) {
      slots.buffers[slot] = 0;
      slots.lengths[slot] = 0;
    }
    for (uint32_t slot = 0; slot < launch->buffer_count; ++slot) {
      uint16_t index = launch->buffer_index[slot];
      if (index >= header->buffer_count) {
        launch->status = NPU_QUEUE_BAD_BUFFER_INDEX;
        return launch->status;
      }
      slots.buffers[slot] = bases[index];
      /* The length the host declared for the allocation, which is the same
         number the synchronous path puts in the entry point's primary-input
         block -- so a kernel's own length check reads the same value whichever
         transport carried it. */
      slots.lengths[slot] = (int)buffers[index].bytes;
    }

    /* The scalar block is passed exactly as the host packed it, offsets in its
       tail included, so `npu_apply_buffer_offsets` inside the launch folds them
       back the way it always has. */
    launch->status =
        run(context, launch->selector, launch->launch_word,
            arena + launch->scalar_offset, (int)launch->scalar_bytes,
            slots.buffers, slots.lengths);
    if (launch->status != 0)
      return launch->status;
  }
  return NPU_QUEUE_OK;
}

int32_t npu_queue_run_packet(const npu_dspqueue_buffer_t *descriptors,
                             uint32_t descriptor_count, npu_queue_launch_fn run,
                             void *context, npu_queue_response_t *response) {
  response->batch_id = 0;
  response->status = NPU_QUEUE_BAD_FRAME;
  /* One for the frame and at least nothing else: a packet with no descriptors
     carries no frame, and a frame is the only thing this side can act on. */
  if (descriptors == 0 || descriptor_count == 0)
    return response->status;

  uint8_t *bases[NPU_QUEUE_MAX_BUFFERS];
  uint32_t base_count = descriptor_count - 1u;
  if (base_count > NPU_QUEUE_MAX_BUFFERS)
    return response->status;
  for (uint32_t i = 0; i < base_count; ++i)
    bases[i] = (uint8_t *)descriptors[i + 1].ptr;

  /* Read before the frame runs, because a launch's status is written back into
     the frame and a frame that failed validation has no id to report -- but one
     that ran must be identified even if a launch in it failed. */
  void *frame = descriptors[0].ptr;
  if (frame != 0 && descriptors[0].size >= sizeof(npu_queue_frame_header_t))
    response->batch_id = ((const npu_queue_frame_header_t *)frame)->batch_id;

  response->status = npu_queue_run_frame(frame, descriptors[0].size, bases,
                                         base_count, run, context);
  return response->status;
}

int npu_queue_import_retry(int rc, int bad_parm) { return rc == bad_parm; }

const char *npu_queue_import_mode_name(int used_fallback) {
  return used_fallback ? "callback-fallback" : "null-callbacks";
}

static int npu_queue_append(char *out, int capacity, int pos,
                            const char *text) {
  while (*text != '\0' && pos + 1 < capacity)
    out[pos++] = *text++;
  out[pos] = '\0';
  return pos;
}

/* Signed decimal by unsigned magnitude, so INT_MIN needs no special case. */
static int npu_queue_append_decimal(char *out, int capacity, int pos,
                                    int value) {
  char digits[10];
  unsigned magnitude = (unsigned)value;
  int count = 0;
  if (value < 0) {
    magnitude = 0u - magnitude;
    if (pos + 1 < capacity)
      out[pos++] = '-';
  }
  do {
    digits[count++] = (char)('0' + (magnitude % 10u));
    magnitude /= 10u;
  } while (magnitude != 0u);
  while (count > 0 && pos + 1 < capacity)
    out[pos++] = digits[--count];
  out[pos] = '\0';
  return pos;
}

int npu_queue_import_message(char *out, int capacity, int used_fallback,
                             int rc) {
  if (out == 0 || capacity <= 0)
    return 0;
  int pos = npu_queue_append(out, capacity, 0, "npu queue import: ");
  pos = npu_queue_append(out, capacity, pos,
                         npu_queue_import_mode_name(used_fallback));
  pos = npu_queue_append(out, capacity, pos, " (rc ");
  pos = npu_queue_append_decimal(out, capacity, pos, rc);
  return npu_queue_append(out, capacity, pos, ")");
}

/* ===------------------------------------------------------------------=== */
/* The post-mortem report                                                   */
/* ===------------------------------------------------------------------=== */

/* Volatile because the drain thread writes it while an RPC thread may be
   reading it back through the `queue_report` selector. The words are counters
   and latches a post-mortem reads for magnitude and identity, so torn
   cross-word ordering costs nothing; what matters is that the compiler never
   caches a store the reader was created to see. */
static volatile uint32_t g_queue_report[NPU_QUEUE_REPORT_WORDS];

void npu_queue_report_reset(void) {
  for (int word = 0; word < NPU_QUEUE_REPORT_WORDS; ++word)
    g_queue_report[word] = 0;
  g_queue_report[NPU_QUEUE_REPORT_MAGIC_WORD] = NPU_QUEUE_REPORT_MAGIC;
}

void npu_queue_report_set(int word, uint32_t value) {
  if (word >= 0 && word < NPU_QUEUE_REPORT_WORDS)
    g_queue_report[word] = value;
}

void npu_queue_report_add(int word, uint32_t amount) {
  if (word >= 0 && word < NPU_QUEUE_REPORT_WORDS)
    g_queue_report[word] += amount;
}

uint32_t npu_queue_report_get(int word) {
  if (word >= 0 && word < NPU_QUEUE_REPORT_WORDS)
    return g_queue_report[word];
  return 0;
}

void npu_queue_report_note_packet(uint32_t flags, uint32_t buffer_count,
                                  uint32_t message_bytes,
                                  const npu_dspqueue_buffer_t *descriptors) {
  npu_queue_report_add(NPU_QUEUE_REPORT_FRAMES_SEEN, 1);
  npu_queue_report_set(NPU_QUEUE_REPORT_LAST_PACKET_FLAGS, flags);
  npu_queue_report_set(NPU_QUEUE_REPORT_LAST_PACKET_BUFFERS, buffer_count);
  npu_queue_report_set(NPU_QUEUE_REPORT_LAST_PACKET_MESSAGE_BYTES,
                       message_bytes);
  if (descriptors == 0 || buffer_count == 0) {
    npu_queue_report_set(NPU_QUEUE_REPORT_FRAME_PTR_CLASS, 1);
    return;
  }
  npu_queue_report_set(NPU_QUEUE_REPORT_LAST_PACKET_FRAME_BYTES,
                       descriptors[0].size);
  /* The frame header words are recorded before the packet runs, so a parse
     that faults still leaves behind the one thing that classifies it: a
     garbage magic is a stale or unresolved mapping, a good magic with a
     refused parse is a framing question. */
  const npu_queue_frame_header_t *header =
      (const npu_queue_frame_header_t *)descriptors[0].ptr;
  if (header == 0 ||
      descriptors[0].size < (uint32_t)sizeof(npu_queue_frame_header_t)) {
    npu_queue_report_set(NPU_QUEUE_REPORT_FRAME_PTR_CLASS, 1);
    return;
  }
  npu_queue_report_set(NPU_QUEUE_REPORT_FRAME_PTR_CLASS, 2);
  npu_queue_report_set(NPU_QUEUE_REPORT_LAST_FRAME_MAGIC, header->magic);
  npu_queue_report_set(NPU_QUEUE_REPORT_LAST_FRAME_VERSION, header->version);
  npu_queue_report_set(NPU_QUEUE_REPORT_LAST_FRAME_BATCH, header->batch_id);
}

void npu_queue_report_note_read_error(int rc) {
  uint32_t last = g_queue_report[NPU_QUEUE_REPORT_LAST_READ_RC];
  if (last == (uint32_t)rc) {
    npu_queue_report_add(NPU_QUEUE_REPORT_READ_RC_REPEATS, 1);
  } else {
    npu_queue_report_set(NPU_QUEUE_REPORT_LAST_READ_RC, (uint32_t)rc);
    npu_queue_report_set(NPU_QUEUE_REPORT_READ_RC_REPEATS, 1);
  }
}

void mojo_npu_queue_report(uint32_t *out, int word_capacity) {
  if (out == 0)
    return;
  int words = word_capacity < NPU_QUEUE_REPORT_WORDS ? word_capacity
                                                     : NPU_QUEUE_REPORT_WORDS;
  for (int word = 0; word < words; ++word)
    out[word] = g_queue_report[word];
}

/* Empty reads between periodic pending-packet samples: ~5 s at the drain's
   wait, frequent enough to catch a stuck packet, far too rare to matter. */
#define NPU_QUEUE_PENDING_SAMPLE_READS 500u

int npu_queue_service_step(const npu_queue_channel_ops_t *ops,
                           npu_queue_launch_fn run, void *context) {
  /* The latch is the report's state word: a pass that latched must never
     touch the vendor again, in either drain mode, and the same word is what
     the post-mortem and the host tests read. */
  uint32_t state = npu_queue_report_get(NPU_QUEUE_REPORT_STATE);
  if (state == NPU_QUEUE_STATE_READ_FATAL ||
      state == NPU_QUEUE_STATE_RESPONSE_FATAL)
    return NPU_QUEUE_SERVICE_FATAL;

  uint32_t flags = 0;
  uint32_t buffer_count = 0;
  uint32_t message_bytes = 0;
  /* Sized for the frame's descriptor plus the frame's own allocations. Sizing
     it by the frame's buffer count instead would have the vendor's read
     refuse a full frame outright with AEE_EBADPARM ("The packet is too large
     to fit in the provided buffers"). */
  npu_dspqueue_buffer_t buffers[NPU_QUEUE_MAX_PACKET_BUFFERS];

  /* A submission carries no message -- everything this side reads before it
     trusts a frame is in the frame's own header -- but a real buffer is
     passed anyway rather than a null and a zero maximum. `dspqueue_write`
     documents "zero if the packet contains no message"; the read side
     documents no such thing about its maximum, and a null the library chose
     to validate would refuse every packet. */
  uint8_t message[sizeof(npu_queue_response_t)];
  npu_queue_report_add(NPU_QUEUE_REPORT_WAKES, 1);
  int rc = ops->read_noblock(ops->queue, &flags, NPU_QUEUE_MAX_PACKET_BUFFERS,
                             &buffer_count, buffers, sizeof(message),
                             &message_bytes, message);
  if (rc == NPU_DSPQUEUE_EWOULDBLOCK) {
    npu_queue_report_add(NPU_QUEUE_REPORT_EMPTY_READS, 1);
    if (ops->sample_pending != 0 &&
        npu_queue_report_get(NPU_QUEUE_REPORT_EMPTY_READS) %
                NPU_QUEUE_PENDING_SAMPLE_READS ==
            0u)
      ops->sample_pending(ops->queue);
    return NPU_QUEUE_SERVICE_EMPTY;
  }
  if (rc != 0) {
    /* Everything else the read can return -- EBADPARM, ENOSUCHMAP, EBADITEM,
       EBADSTATE -- is persistent for the packet at the head of the queue, so
       retrying it is the unbounded silent host wait order 074 recorded. The
       protocol instead: record it, sample the queue depth while the evidence
       is live, answer the host once with a refusal it can render, and latch.

       The refusal is message-only. The packet's buffers were exactly what
       could not be read, so there is nothing to echo, and a response needs
       no buffer resolution to travel. Non-blocking, because a drain that
       blocked in `write` while latching could hang `stop`'s join -- and a
       full response ring here would mean completions were already failing. */
    npu_queue_report_note_read_error(rc);
    if (ops->sample_pending != 0)
      ops->sample_pending(ops->queue);
    npu_queue_response_t refusal;
    refusal.batch_id = 0;
    refusal.status = NPU_QUEUE_BAD_READ;
    int wrc = ops->write(ops->queue, NPU_DSPQUEUE_PACKET_FLAG_MESSAGE,
                         /*num_buffers=*/0, /*buffers=*/0, sizeof(refusal),
                         (const uint8_t *)&refusal, /*timeout_us=*/0);
    npu_queue_report_add(NPU_QUEUE_REPORT_REFUSALS_SENT, 1);
    npu_queue_report_set(NPU_QUEUE_REPORT_LAST_RESPONSE_RC, (uint32_t)wrc);
    npu_queue_report_set(NPU_QUEUE_REPORT_STATE, NPU_QUEUE_STATE_READ_FATAL);
    return NPU_QUEUE_SERVICE_FATAL;
  }

  /* The frame rides in the packet's first buffer reference, which the queue
     has already invalidated on this side; `npu_queue_run_packet` is where
     that convention is written down. */
  npu_queue_report_note_packet(flags, buffer_count, message_bytes, buffers);
  npu_queue_response_t response;
  npu_queue_run_packet(buffers, buffer_count, run, context, &response);
  npu_queue_report_set(NPU_QUEUE_REPORT_LAST_PARSE_STATUS,
                       (uint32_t)response.status);

  /* The buffers go back carrying each descriptor's own flags, echoed: the
     host chose a maintenance class per reference -- the frame and
     host-written data flagged, declared-constant weights bare -- and
     "sender"/"recipient" are relative to the packet, so the same pair means
     the mirror image on the way back. Echoing is what keeps the response's
     maintenance the request's class split rather than a second policy; the
     shipped backend's responses flag only its metadata block the same way
     (`GGML_HEXAGON_LESSONS.md` section 3, their `main.c:1091-1092`). */
  int wrc = ops->write(ops->queue,
                       NPU_DSPQUEUE_PACKET_FLAG_MESSAGE |
                           NPU_DSPQUEUE_PACKET_FLAG_BUFFERS,
                       buffer_count, buffers, sizeof(response),
                       (const uint8_t *)&response, NPU_DSPQUEUE_TIMEOUT_NONE);
  npu_queue_report_add(NPU_QUEUE_REPORT_RESPONSES_WRITTEN, 1);
  npu_queue_report_set(NPU_QUEUE_REPORT_LAST_RESPONSE_RC, (uint32_t)wrc);
  if (wrc != 0) {
    /* A completion that did not reach the queue is a frame the host will
       wait on forever; there is no channel left to say so, which is what
       the report and the host's bounded receive are for. */
    npu_queue_report_set(NPU_QUEUE_REPORT_STATE,
                         NPU_QUEUE_STATE_RESPONSE_FATAL);
    return NPU_QUEUE_SERVICE_FATAL;
  }
  return NPU_QUEUE_SERVICE_SERVICED;
}

/* ===------------------------------------------------------------------=== */
/* The resident loop                                                        */
/* ===------------------------------------------------------------------=== */

/* Compiled only for the device, where the SDK's own `dspqueue.h` is on the
   include path -- which makes this the one translation unit in the port that
   can hold the host's restated descriptor to the vendor's real one. It does,
   below, and a drift fails this build rather than corrupting a pointer on
   device. */
#ifdef __hexagon__

#include "AEEStdDef.h"
#include "HAP_farf.h"
#include "dspqueue.h"
#include "mojo_npu.h"
#include "qurt.h"

/* The level FARF(ALWAYS, ...) logs at, so the pre-built line below reaches the
   same mini-dm capture the macro's output did. The SDK's HAP_farf.h defines
   it; the fallback only keeps an older header from failing this compile. */
#ifndef FARF_ALWAYS_LEVEL
#define FARF_ALWAYS_LEVEL HAP_LEVEL_HIGH
#endif

NPU_DSPQUEUE_ASSERT_MATCHES_SDK();

extern AEEResult npu_run_launch(uint32 selector, uint32 thread_mask,
                                const uint8 *scalars, int scalars_len,
                                uint8 **buffers, const int *lengths);

/* The launch runner the device passes to the decode above: the same entry the
   synchronous dispatch calls, adapted to the callback's shape. */
static int32_t npu_queue_run_launch(void *context, uint32_t selector,
                                    uint32_t launch_word,
                                    const uint8_t *scalars, int scalars_len,
                                    uint8_t **buffers, const int *lengths) {
  (void)context;
  return (int32_t)npu_run_launch(selector, launch_word, scalars, scalars_len,
                                 buffers, lengths);
}

/* The session the host opened with `start`, and the flag `stop` clears.

   `running` is read by the drain thread and written by an RPC thread, so it is
   volatile: the loop's exit is the only thing standing between a `stop` and a
   protection domain that has to be torn down by rebooting the box. */
static dspqueue_t g_queue;
static qurt_thread_t g_drain_thread;
/* Whether this session spawned its own drain thread (null-callbacks mode).
   Callback mode spawns none -- the vendor callback thread is the drain --
   so `stop` must know which teardown it owes. */
static int g_have_drain_thread;
static volatile int g_running;

/* Stack for the drain thread. Static for the same reason the launch stacks are:
   this protection domain's C library has no heap. */
#define NPU_QUEUE_STACK_BYTES (2 * 16384)
static uint8 g_drain_stack[NPU_QUEUE_STACK_BYTES] __attribute__((aligned(128)));

/* How long a drain wait blocks before looking at `running` again. The loop must
   not spin: a spinning thread holds a hardware thread that a launch's
   participants need, and the participants are what a launch waits on at its
   barrier -- so a spin here starves the work it is waiting for. Blocking in the
   queue is what yields the thread; the timeout exists only so `stop` is
   observed within a bounded time rather than never. */
#define NPU_QUEUE_WAIT_MICROS 10000u

/* The queue-depth statistic, bound weak: its residency in this protection
   domain is unproven (the 072 load class), and a missing optional probe must
   not cost the skel its load. WEAK UND stays importable by construction under
   `npu_skel_symbol_gate.sh`. */
#pragma weak dspqueue_get_stat

/* Samples how many packets sit unread in this side's queue, when the stat
   entry exists. It is the one word that splits the two silent-hang classes: a
   read that says would-block forever while a packet is pending is signalling
   that never surfaced it; pending zero is a packet that never reached this
   side at all. */
static void npu_queue_report_sample_pending(void *queue) {
  if (0 == (dspqueue_get_stat))
    return;
  uint64 pending = 0;
  if (dspqueue_get_stat((dspqueue_t)queue, DSPQUEUE_STAT_READ_QUEUE_PACKETS,
                        &pending) != 0)
    return;
  npu_queue_report_set(NPU_QUEUE_REPORT_PENDING_PACKETS, (uint32_t)pending);
  npu_queue_report_add(NPU_QUEUE_REPORT_PENDING_SAMPLES, 1);
}

/* The vendor calls behind the service pass's ops table. The casts are safe
   because this translation unit asserts the restated descriptor identical to
   the vendor's. */
static int npu_queue_dsp_read(void *queue, uint32_t *flags,
                              uint32_t max_buffers, uint32_t *num_buffers,
                              npu_dspqueue_buffer_t *buffers,
                              uint32_t max_message_bytes,
                              uint32_t *message_bytes, uint8_t *message) {
  return dspqueue_read_noblock((dspqueue_t)queue, flags, max_buffers,
                               num_buffers, (struct dspqueue_buffer *)buffers,
                               max_message_bytes, message_bytes, message);
}

static int npu_queue_dsp_write(void *queue, uint32_t flags,
                               uint32_t num_buffers,
                               npu_dspqueue_buffer_t *buffers,
                               uint32_t message_bytes, const uint8_t *message,
                               uint32_t timeout_us) {
  return dspqueue_write((dspqueue_t)queue, flags, num_buffers,
                        (struct dspqueue_buffer *)buffers, message_bytes,
                        message, timeout_us);
}

static npu_queue_channel_ops_t g_dspqueue_ops;

static void npu_queue_drain(void *arg) {
  (void)arg;
  while (g_running) {
    /* Everything queued, greedily: a serviced packet may have peers behind
       it, and the frame after it is exactly the overlap this transport is
       for. EMPTY and FATAL both end at the wait -- a latched drain only
       watches for `stop`, and the wait is what bounds how long `stop` can go
       unobserved. */
    int rc = npu_queue_service_step(&g_dspqueue_ops, npu_queue_run_launch, 0);
    if (rc == NPU_QUEUE_SERVICE_SERVICED)
      continue;
    qurt_timer_sleep(NPU_QUEUE_WAIT_MICROS);
  }
  if (g_queue_report[NPU_QUEUE_REPORT_STATE] == NPU_QUEUE_STATE_RUNNING)
    npu_queue_report_set(NPU_QUEUE_REPORT_STATE, NPU_QUEUE_STATE_STOPPED);
  qurt_thread_exit(0);
}

/* The callback-mode burst: the reference implementation's `process_ops` poll
   backoff verbatim (`main.c:1101-1103,1121-1128`) -- packet signalling
   coalesces, so after the queue reads empty the pass lingers through a short
   poll burst for packets arriving on the callback's heels, then returns the
   vendor its thread. */
#define NPU_QUEUE_BURST_POLLS 100u
#define NPU_QUEUE_BURST_SLEEP_MICROS 100u

/* Packet callback, registered only when the null-callbacks import was refused.
   The 076 leg proved the alternative wrong on this silicon: with callbacks
   registered, this side's read/write pair is legal only from the callback's
   own context -- a resident drain thread calling the same pair got
   AEE_EBADSTATE on both -- so in this mode the vendor's callback thread IS
   the drain, exactly the reference implementation's structure
   (`htp_packet_callback` -> `process_ops`, their `main.c:1148-1156`; no
   thread of their own is spawned in callback mode, `main.c:362,547`). */
static void npu_queue_packet_callback(dspqueue_t queue, AEEResult error,
                                      void *context) {
  (void)queue;
  (void)error;
  (void)context;
  uint32 polls = NPU_QUEUE_BURST_POLLS;
  while (g_running) {
    int rc = npu_queue_service_step(&g_dspqueue_ops, npu_queue_run_launch, 0);
    if (rc == NPU_QUEUE_SERVICE_SERVICED) {
      polls = NPU_QUEUE_BURST_POLLS;
      continue;
    }
    if (rc == NPU_QUEUE_SERVICE_FATAL)
      return;
    if (--polls == 0)
      return;
    qurt_timer_sleep(NPU_QUEUE_BURST_SLEEP_MICROS);
  }
}

/* Recorded and counted, never acted on: the reference implementation's own
   handler is a log line with the comment "No errors expected on the DSP"
   (their `main.c:649-652`), and a teardown from callback context would race
   the thread that called us. The report is where a post-mortem reads it. */
static void npu_queue_error_callback(dspqueue_t queue, AEEResult error,
                                     void *context) {
  (void)queue;
  (void)context;
  npu_queue_report_set(NPU_QUEUE_REPORT_ERROR_CB_RC, (uint32_t)error);
  npu_queue_report_add(NPU_QUEUE_REPORT_ERROR_CB_COUNT, 1);
}

AEEResult mojo_npu_start(remote_handle64 handle, uint64 dsp_queue_id) {
  (void)handle;
  if (g_running)
    return AEE_EALREADY;
  /* Reset first and echo the id as decoded, so even a session that dies right
     here leaves the two words that close the id-truncation question: the host
     compares them to the id it sent. */
  npu_queue_report_reset();
  npu_queue_report_set(NPU_QUEUE_REPORT_QUEUE_ID_LO, (uint32_t)dsp_queue_id);
  npu_queue_report_set(NPU_QUEUE_REPORT_QUEUE_ID_HI,
                       (uint32_t)(dsp_queue_id >> 32));
  /* The id is 64 bits on both sides of `dspqueue_export`/`dspqueue_import`;
     narrowing it here would import a queue that was never exported.

     Null callbacks first -- no vendor threads next to the drain -- then the
     reference implementation's fallback: some firmware refuses the null with
     AEE_EBADPARM and wants callback mode (`GGML_HEXAGON_LESSONS.md` section 1,
     their `htp/main.c:327-333`); the 076 leg measured that refusal on this
     silicon. The mode decides who the drain IS, not merely who wakes it:
     076 also measured that with callbacks registered, this side's explicit
     read/write pair is AEE_EBADSTATE from any thread but the callback's own,
     so callback mode services frames from the callback and spawns no thread.

     `g_running` is raised before the callback-mode import because the vendor
     may deliver the first callback on its own thread the moment the import
     binds, and a service pass must never observe the session as stopped
     while `start` is still on its way out. */
  g_running = 1;
  int rc = dspqueue_import(dsp_queue_id, 0, 0, 0, &g_queue);
  int used_fallback = 0;
  if (npu_queue_import_retry(rc, AEE_EBADPARM)) {
    used_fallback = 1;
    rc = dspqueue_import(dsp_queue_id, npu_queue_packet_callback,
                         npu_queue_error_callback, 0, &g_queue);
  }
  npu_queue_report_set(NPU_QUEUE_REPORT_IMPORT_MODE, (uint32_t)used_fallback);
  npu_queue_report_set(NPU_QUEUE_REPORT_IMPORT_RC, (uint32_t)rc);
  npu_queue_report_set(NPU_QUEUE_REPORT_STATE,
                       rc == 0 ? NPU_QUEUE_STATE_RUNNING
                               : NPU_QUEUE_STATE_IMPORT_FAILED);
  /* The self-report line order 074's protocol greps per leg, routed the way
     FARF(ALWAYS, ...) routes -- same resident entries, same level -- but with
     the string pre-built. The FARF macro itself is unusable here: it emits
     the header's inline `vsnprintf` fallback alongside the resident call, and
     libc's `vsnprintf` imports a heap/lock/TLS layer the protection domain
     does not provide, so the skel refuses to load (lane evidence 072). Both
     entries are weak and proven resident on this device (072 control and
     grouplink arms); the null check mirrors the FARF macro's own, and the
     `else` calls `HAP_debug` unconditionally exactly as the macro's fallback
     does. */
  char message[NPU_QUEUE_IMPORT_MSG_BYTES];
  npu_queue_import_message(message, (int)sizeof(message), used_fallback, rc);
  if (0 == (HAP_debug_v2))
    HAP_debug(message, FARF_ALWAYS_LEVEL, __FILE__, __LINE__);
  else
    HAP_debug_v2(FARF_ALWAYS_LEVEL, __FILE__, __LINE__, "%s", message);
  if (rc != 0) {
    g_running = 0;
    return AEE_EFAILED;
  }

  g_dspqueue_ops.queue = (void *)g_queue;
  g_dspqueue_ops.read_noblock = npu_queue_dsp_read;
  g_dspqueue_ops.write = npu_queue_dsp_write;
  g_dspqueue_ops.sample_pending = npu_queue_report_sample_pending;

  /* Callback mode is fully bound at this point: the vendor's callback thread
     is the drain, so there is no thread to create and nothing further to
     start. The join in `stop` is conditional on the same fact. */
  if (used_fallback) {
    g_have_drain_thread = 0;
    return 0;
  }

  qurt_thread_attr_t attr;
  qurt_thread_attr_init(&attr);
  qurt_thread_attr_set_name(&attr, "mojo_npu_q");
  /* The priority a launch's participants are created at is the priority of the
     thread that creates them, and with a queue that thread is this one rather
     than an RPC thread. Pinning it to this call's priority is what keeps a
     participant from being denied the hardware thread its peers wait for at a
     barrier -- the same rule the synchronous fan-out follows, now that the two
     no longer share a thread. */
  qurt_thread_attr_set_priority(
      &attr, (unsigned short)qurt_thread_get_priority(qurt_thread_get_id()));
  qurt_thread_attr_set_stack_addr(&attr, g_drain_stack);
  qurt_thread_attr_set_stack_size(&attr, NPU_QUEUE_STACK_BYTES);

  if (qurt_thread_create(&g_drain_thread, &attr, npu_queue_drain, 0) !=
      QURT_EOK) {
    g_running = 0;
    npu_queue_report_set(NPU_QUEUE_REPORT_STATE, NPU_QUEUE_STATE_THREAD_FAILED);
    dspqueue_close(g_queue);
    return AEE_EQURTTHREADCREATE;
  }
  g_have_drain_thread = 1;
  return 0;
}

AEEResult mojo_npu_stop(remote_handle64 handle) {
  (void)handle;
  if (!g_running)
    return 0;
  /* Joined rather than signalled and forgotten. A drain thread still running
     when the session is torn down is what makes a wedged protection domain
     stick, and on this box a wedged CDSP is a reboot a human has to authorise.
   */
  g_running = 0;
  if (g_have_drain_thread) {
    int unused = 0;
    (void)qurt_thread_join(g_drain_thread, &unused);
    g_have_drain_thread = 0;
  }
  /* In callback mode there is no thread of ours to join: the drain is the
     vendor's callback thread, `g_running` stops the next service pass, and
     the close below is what ends deliveries -- the same teardown the
     reference implementation ships (killed flag, join only the thread it
     spawned, close; their `main.c:573-585`). A callback mid-pass when the
     flag drops finishes its current packet and returns at the next check,
     which is also their shipped window. */
  npu_queue_report_sample_pending((void *)g_queue);
  if (g_queue_report[NPU_QUEUE_REPORT_STATE] == NPU_QUEUE_STATE_RUNNING)
    npu_queue_report_set(NPU_QUEUE_REPORT_STATE, NPU_QUEUE_STATE_STOPPED);
  dspqueue_close(g_queue);
  return 0;
}

#endif /* __hexagon__ */
