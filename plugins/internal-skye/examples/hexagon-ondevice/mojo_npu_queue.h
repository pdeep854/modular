/* Draining a frame of launches that arrived over the shared-memory queue.
 *
 * Split from the resident loop that feeds it so the decode -- which is where
 * the bounds live, and where a bug is a wrong pointer rather than a failed call
 * -- is a plain function over plain memory and can be built and tested on a
 * host with no NPU. The loop around it needs QuRT and the vendor queue and can
 * only run on the device; this cannot, and does not.
 */

#ifndef MOJO_NPU_QUEUE_H
#define MOJO_NPU_QUEUE_H

#include "NPUDspQueueABI.h"
#include "NPUQueueWire.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Runs one launch. The device passes `npu_run_launch`; a host build passes a
   recorder, which is what lets the drain logic be tested without a kernel.
   `buffers` is written through, so the caller owns it. */
typedef int32_t (*npu_queue_launch_fn)(void *context, uint32_t selector,
                                       uint32_t launch_word,
                                       const uint8_t *scalars, int scalars_len,
                                       uint8_t **buffers, const int *lengths);

/* Runs every launch in `frame`, in order, and writes each one's result back
   into the frame's own launch record so the response needs no second table.
   Stops at the first failure: a batch's later launches read what its earlier
   ones wrote, so continuing past one would run kernels over garbage and report
   a status for work that never had valid inputs.
 *
 * `bases` is the frame's buffer table resolved to this side's addresses, one
 * per entry, in the table's own order. Returns 0, or the first failing status.
 */
int32_t npu_queue_run_frame(void *frame, uint32_t received_bytes,
                            uint8_t *const *bases, uint32_t base_count,
                            npu_queue_launch_fn run, void *context);

/* Runs one queue packet: resolves its descriptors and hands the result to
   `npu_queue_run_frame`, then fills the response the host reads.

   The frame is the packet's FIRST buffer reference, so the frame's own buffer
   table is resolved from `descriptors + 1` -- the one place that shift is
   written down. It is here, and not in the resident loop, because the loop is
   the half a host cannot build and this is the half where getting the shift
   wrong resolves every operand one allocation early. Returns the status it also
   writes into `response`. */
int32_t npu_queue_run_packet(const npu_dspqueue_buffer_t *descriptors,
                             uint32_t descriptor_count, npu_queue_launch_fn run,
                             void *context, npu_queue_response_t *response);

/* The two vendor calls one service pass makes, behind an ops table so the
   pass itself is host-buildable. On the device the table wraps
   `dspqueue_read_noblock` and `dspqueue_write`; in a host test it is a fake
   that scripts the vendor's answers. The pass is the shared core of BOTH
   drain modes -- the resident thread's loop and the packet callback's burst
   -- which is what keeps the two modes one protocol instead of two. */
typedef struct npu_queue_channel_ops {
  void *queue;
  int (*read_noblock)(void *queue, uint32_t *flags, uint32_t max_buffers,
                      uint32_t *num_buffers, npu_dspqueue_buffer_t *buffers,
                      uint32_t max_message_bytes, uint32_t *message_bytes,
                      uint8_t *message);
  int (*write)(void *queue, uint32_t flags, uint32_t num_buffers,
               npu_dspqueue_buffer_t *buffers, uint32_t message_bytes,
               const uint8_t *message, uint32_t timeout_us);
  /* Optional: samples the unread-packet depth into the report. Null when the
     channel has no such probe -- on the device it is a weak-bound stat entry
     whose residency is unproven (the 072 load class). */
  void (*sample_pending)(void *queue);
} npu_queue_channel_ops_t;

/* One service pass: read one packet, run it, answer it. Returns SERVICED when
   a packet was consumed (there may be more), EMPTY when the queue had none,
   FATAL when the drain latched -- a read the vendor refused outright or a
   response that could not be written. A latched pass answers the host once
   with an `NPU_QUEUE_BAD_READ` refusal before latching, and every later pass
   returns FATAL without touching the vendor again: retrying a read that
   refuses the head packet is the unbounded silent host wait order 074
   recorded. The latch is the report's own state word, so it survives into
   the post-mortem and is visible to a host test. */
#define NPU_QUEUE_SERVICE_SERVICED 0
#define NPU_QUEUE_SERVICE_EMPTY 1
#define NPU_QUEUE_SERVICE_FATAL (-1)
int npu_queue_service_step(const npu_queue_channel_ops_t *ops,
                           npu_queue_launch_fn run, void *context);

/* Whether a refused `dspqueue_import` with null callbacks warrants a retry in
   callback mode. Null callbacks are tried first because they keep the vendor
   library from starting threads of its own next to the drain, and some
   firmware refuses the null with exactly `bad_parm` -- the sequence the
   shipped ggml-hexagon backend encodes (`GGML_HEXAGON_LESSONS.md` section 1,
   their `htp/main.c:327-333`). Any other refusal is final: retrying it in
   another mode would mask a real error behind a mode change.

   Host-buildable so the decision -- not a transcript of it -- is what the
   framing tests hold. `bad_parm` is a parameter because the AEE numbering
   differs between the DSP and a host test (`AEE_EOFFSET`). */
int npu_queue_import_retry(int rc, int bad_parm);

/* The self-report token for the mode that bound, for the `npu queue import:`
   line: "null-callbacks" or "callback-fallback". Which mode a leg ran under is
   the first question of an import A/B, so the line prints unconditionally. */
const char *npu_queue_import_mode_name(int used_fallback);

/* Builds the whole `npu queue import:` line -- exactly
   `npu queue import: <mode> (rc <n>)`, `<n>` in %d form -- into `out`,
   NUL-terminated and truncated at `capacity - 1`. Returns the length written.

   Built by hand rather than formatted because the device must not format:
   FARF's formatting path is a header inline over libc's `vsnprintf`, which
   drags the heap and lock/TLS layers (`sys_sbrk`, `sys_Mtx*`, `sys_Tls*`)
   into the link, none of them resident in the protection domain -- so the one
   log line the transport protocol mandates was what made the skel refuse to
   load (lane evidence 072, arm `nofarf` against `base`). Host-buildable so
   the exact strings the protocol greps from the mini-dm capture are pinned by
   a test rather than by transcript. */
#define NPU_QUEUE_IMPORT_MSG_BYTES 64
int npu_queue_import_message(char *out, int capacity, int used_fallback,
                             int rc);

/* The drain's post-mortem report: `NPU_QUEUE_REPORT_WORDS` uint32 words whose
   indices `NPUQueueWire.h` names, written by the resident loop as it goes and
   read back by the host over the synchronous transport through the
   `queue_report` selector -- after a hang, which is exactly when the queue
   itself cannot say what happened. Host-buildable (it is plain memory) so the
   word layout is pinned by a test rather than by a device leg. */
void npu_queue_report_reset(void);
void npu_queue_report_set(int word, uint32_t value);
void npu_queue_report_add(int word, uint32_t amount);
uint32_t npu_queue_report_get(int word);

/* Records one received packet's shape, and -- when the frame descriptor
   resolved to a readable address -- the frame header words a post-mortem
   needs: magic, version, batch id. Called before the packet runs, so a parse
   that never returns still leaves the packet's account behind. */
void npu_queue_report_note_packet(uint32_t flags, uint32_t buffer_count,
                                  uint32_t message_bytes,
                                  const npu_dspqueue_buffer_t *descriptors);

/* Records a vendor read failure: the rc, and how many consecutive reads
   returned it. */
void npu_queue_report_note_read_error(int rc);

/* The `queue_report` selector's body: copies the report into `out`, at most
   `word_capacity` words, so a shorter reader truncates rather than overruns. */
void mojo_npu_queue_report(uint32_t *out, int word_capacity);

#ifdef __cplusplus
}
#endif

#endif /* MOJO_NPU_QUEUE_H */
