//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// The one vendor struct the queue transport dereferences, restated once.
//
// It has to be restated rather than included: the host half is built where the
// Hexagon SDK has no library at all -- `QuRTDriver.h` says why, and binds every
// FastRPC entry point with `dlsym` for the same reason -- so `dspqueue.h` is
// not on that include path and never will be. What the SDK's own headers settle
// is therefore recorded here as *numbers*, and the numbers are checked twice:
//
//  - here, on every translation unit that includes this file, host and device
//    alike, so a restatement that drifts fails the build rather than handing a
//    wrong pointer to a DLL;
//  - and on the device, against the real `struct dspqueue_buffer`, by expanding
//    `NPU_DSPQUEUE_ASSERT_MATCHES_SDK()` where `dspqueue.h` *is* on the include
//    path (`mojo_npu_queue.c`). That is the leg that makes the numbers below a
//    claim about the vendor's header and not just about themselves.
//
// The layout is not a guess and was not read off by eye: it is
// `dspqueue.h:125-138` of Hexagon SDK 6.6.0.0
// (sha256 157492952bf1278c...), asserted against that header by the SDK's own
// hexagon-clang 19.0.07 with a deliberately-wrong control leg that the compiler
// refused. Evidence:
// `.agentwork/flux-npu/lane-evidence/068/dspqueue-truth.txt`.
//
//===----------------------------------------------------------------------===//

#ifndef SKYE_RUNTIME_QURT_NPUDSPQUEUEABI_H
#define SKYE_RUNTIME_QURT_NPUDSPQUEUEABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define NPU_DSPQUEUE_STATIC_ASSERT(cond, message) static_assert(cond, message)
#define NPU_DSPQUEUE_ALIGNOF(type) alignof(type)
#else
#define NPU_DSPQUEUE_STATIC_ASSERT(cond, message) _Static_assert(cond, message)
#define NPU_DSPQUEUE_ALIGNOF(type) _Alignof(type)
#endif

/// `struct dspqueue_buffer`, verbatim. Field for field and comment for comment
/// the vendor's, because the value of a restatement is entirely in its being
/// the same text: a paraphrase is a second source to keep in sync.
typedef struct {
  uint32_t fd;     /**< Buffer file descriptor */
  uint32_t size;   /**< Buffer size in bytes. The client can set this field
                        to zero when writing packets; in this case the
                        framework will set the field to the size of the
                        buffer as mapped. */
  uint32_t offset; /**< Offset within the buffer in bytes as allocated and
                        mapped. The virtual address #ptr includes the offset */
  uint32_t flags;  /**< Buffer flags, see enum #dspqueue_buffer_flags */
  union {
    void *ptr; /**< Buffer virtual address; NULL if not mapped in the local
                    context */
    uint64_t address;
  };
} npu_dspqueue_buffer_t;

// Measured, not assumed. The union carries a `uint64_t`, so it is 8 bytes at
// 8-byte alignment on an ILP32 device and on an LP64 host alike and the total
// is 24 either way -- which is what lets one restatement serve both halves.
NPU_DSPQUEUE_STATIC_ASSERT(sizeof(npu_dspqueue_buffer_t) == 24,
                           "dspqueue_buffer is 24 bytes");
NPU_DSPQUEUE_STATIC_ASSERT(NPU_DSPQUEUE_ALIGNOF(npu_dspqueue_buffer_t) == 8,
                           "dspqueue_buffer aligns to 8");
NPU_DSPQUEUE_STATIC_ASSERT(offsetof(npu_dspqueue_buffer_t, fd) == 0,
                           "dspqueue_buffer.fd is at 0");
NPU_DSPQUEUE_STATIC_ASSERT(offsetof(npu_dspqueue_buffer_t, size) == 4,
                           "dspqueue_buffer.size is at 4");
NPU_DSPQUEUE_STATIC_ASSERT(offsetof(npu_dspqueue_buffer_t, offset) == 8,
                           "dspqueue_buffer.offset is at 8");
NPU_DSPQUEUE_STATIC_ASSERT(offsetof(npu_dspqueue_buffer_t, flags) == 12,
                           "dspqueue_buffer.flags is at 12");
NPU_DSPQUEUE_STATIC_ASSERT(offsetof(npu_dspqueue_buffer_t, ptr) == 16,
                           "dspqueue_buffer.ptr is at 16");
NPU_DSPQUEUE_STATIC_ASSERT(offsetof(npu_dspqueue_buffer_t, address) == 16,
                           "dspqueue_buffer.address overlays .ptr");

/// Expanded where the SDK's own `dspqueue.h` is on the include path, which on
/// this port is the device compile. Without this leg the assertions above are a
/// closed loop: they check the restatement against itself.
#define NPU_DSPQUEUE_ASSERT_MATCHES_SDK()                                      \
  NPU_DSPQUEUE_STATIC_ASSERT(sizeof(npu_dspqueue_buffer_t) ==                  \
                                 sizeof(struct dspqueue_buffer),               \
                             "restated dspqueue_buffer changed size");         \
  NPU_DSPQUEUE_STATIC_ASSERT(NPU_DSPQUEUE_ALIGNOF(npu_dspqueue_buffer_t) ==    \
                                 NPU_DSPQUEUE_ALIGNOF(struct dspqueue_buffer), \
                             "restated dspqueue_buffer changed alignment");    \
  NPU_DSPQUEUE_STATIC_ASSERT(offsetof(npu_dspqueue_buffer_t, fd) ==            \
                                 offsetof(struct dspqueue_buffer, fd),         \
                             "restated dspqueue_buffer.fd moved");             \
  NPU_DSPQUEUE_STATIC_ASSERT(offsetof(npu_dspqueue_buffer_t, size) ==          \
                                 offsetof(struct dspqueue_buffer, size),       \
                             "restated dspqueue_buffer.size moved");           \
  NPU_DSPQUEUE_STATIC_ASSERT(offsetof(npu_dspqueue_buffer_t, offset) ==        \
                                 offsetof(struct dspqueue_buffer, offset),     \
                             "restated dspqueue_buffer.offset moved");         \
  NPU_DSPQUEUE_STATIC_ASSERT(offsetof(npu_dspqueue_buffer_t, flags) ==         \
                                 offsetof(struct dspqueue_buffer, flags),      \
                             "restated dspqueue_buffer.flags moved");          \
  NPU_DSPQUEUE_STATIC_ASSERT(offsetof(npu_dspqueue_buffer_t, ptr) ==           \
                                 offsetof(struct dspqueue_buffer, ptr),        \
                             "restated dspqueue_buffer.ptr moved");            \
  NPU_DSPQUEUE_STATIC_ASSERT(NPU_DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER ==          \
                                 DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER,            \
                             "FLUSH_SENDER changed value");                    \
  NPU_DSPQUEUE_STATIC_ASSERT(NPU_DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT ==  \
                                 DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT,    \
                             "INVALIDATE_RECIPIENT changed value");            \
  NPU_DSPQUEUE_STATIC_ASSERT(NPU_DSPQUEUE_PACKET_FLAG_BUFFERS ==               \
                                 DSPQUEUE_PACKET_FLAG_BUFFERS,                 \
                             "PACKET_FLAG_BUFFERS changed value");             \
  NPU_DSPQUEUE_STATIC_ASSERT(NPU_DSPQUEUE_PACKET_FLAG_MESSAGE ==               \
                                 DSPQUEUE_PACKET_FLAG_MESSAGE,                 \
                             "PACKET_FLAG_MESSAGE changed value")

/// `enum dspqueue_buffer_flags`, the subset this transport sets, with the
/// vendor's own doc comments -- because the choice of which bits to set is the
/// whole of the cache maintenance the FastRPC driver used to do per call, and a
/// reader checking that choice should not have to leave this file.
///
/// Cache maintenance is per packet and per direction, and the writer of the
/// packet chooses it. "Sender" and "recipient" are relative to the packet, so
/// the same two bits mean host-flush/device-invalidate on a submission and
/// device-flush/host-invalidate on the response.
#define NPU_DSPQUEUE_BUFFER_FLAG_REF 0x00000004u
#define NPU_DSPQUEUE_BUFFER_FLAG_DEREF 0x00000008u
/**< Flush buffer from sender caches */
#define NPU_DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER 0x00000010u
/**< Invalidate buffer from sender caches */
#define NPU_DSPQUEUE_BUFFER_FLAG_INVALIDATE_SENDER 0x00000020u
/**< Flush buffer from recipient caches */
#define NPU_DSPQUEUE_BUFFER_FLAG_FLUSH_RECIPIENT 0x00000040u
/**< Invalidate buffer from recipient caches */
#define NPU_DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT 0x00000080u

/// What a packet this transport writes carries: the sender flushes what it
/// wrote so the recipient can see it, and the recipient invalidates so it does
/// not read a stale line of its own. Both halves send this same pair, because
/// both halves both read and write the buffers a frame names -- the host packs
/// inputs and reads outputs, the device reads inputs and writes outputs -- and
/// which of the two a given allocation is is the kernel's business, not the
/// transport's.
///
/// `REF`/`DEREF` are deliberately not set. They refcount a *previously mapped*
/// buffer; the mapping itself is `fastrpc_mmap`'s and lasts as long as the
/// allocation does, so a reference taken per packet would only have to be
/// dropped per packet.
#define NPU_DSPQUEUE_TRANSPORT_BUFFER_FLAGS                                    \
  (NPU_DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER |                                     \
   NPU_DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT)

/// `enum dspqueue_packet_flags`, the two this transport uses.
#define NPU_DSPQUEUE_PACKET_FLAG_MESSAGE 0x0001u
#define NPU_DSPQUEUE_PACKET_FLAG_BUFFERS 0x0002u

/// `DSPQUEUE_TIMEOUT_NONE`: block until the call can be served.
#define NPU_DSPQUEUE_TIMEOUT_NONE 0xffffffffu

/// `AEE_EWOULDBLOCK`. One of the two error codes `AEEStdErr.h` does *not* build
/// out of `AEE_EOFFSET`, so unlike every other AEE code it reads the same on
/// the host and on the DSP -- which is what lets one constant serve the drain
/// loop and the host's poll.
#define NPU_DSPQUEUE_EWOULDBLOCK 516

/// `FASTRPC_MAP_FD`, the mapping a queue buffer needs -- measured, not argued.
///
/// `dspqueue.h` states the precondition twice: "Buffers must be mapped to the
/// DSP with fastrpc_mmap() before they can be used in queue packets", or the
/// write and the read fail with `AEE_ENOSUCHMAP`. The first pick here was
/// `FASTRPC_MAP_STATIC` (0), on the argument that only STATIC's
/// stable-remote-address contract makes the descriptor's `ptr` meaningful on
/// the far side. Order 081 refuted that on silicon: under STATIC the DSP's
/// `dspqueue_read_noblock` returns `AEE_ENOSUCHMAP` on the very first packet,
/// before populating anything -- the library's buffer bookkeeping is keyed by
/// FILE DESCRIPTOR ("buffer references are tracked based on the buffer file
/// descriptor", the struct's own doc, and `fd` is its first field for exactly
/// that reason), and a STATIC mapping evidently never lands in the fd table
/// this firmware's read resolves from. The descriptor's `ptr` is not ours to
/// make meaningful: the header defines it per side ("NULL if not mapped in
/// the local context") and the READER's library fills it from its own
/// fd-keyed mapping -- which is why the shipped ggml-hexagon backend maps its
/// one descriptor-riding buffer `FASTRPC_MAP_FD` and then reads `dbuf->ptr`
/// on the DSP as-is (`GGML_HEXAGON_LESSONS.md` section 3, their `:327-333`
/// pinned arena and `main.c:1000` `m_ptr = dbuf->ptr`). No device-side
/// `HAP_mmap*` call is needed for descriptor-riding buffers; their HAP mmap
/// cache serves only fds shipped INSIDE their metadata block, invisible to
/// dspqueue, and this transport has none of those.
#define NPU_FASTRPC_MAP_FD 2

#endif // SKYE_RUNTIME_QURT_NPUDSPQUEUEABI_H
