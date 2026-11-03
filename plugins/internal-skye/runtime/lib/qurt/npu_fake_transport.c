//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// A stand-in for the vendor FastRPC library, loaded through the
// MODULAR_NPU_TRANSPORT_LIB test seam so the host half of the NPU dispatch
// path can run where no device exists.
//
// It restates the generic entry point's wire layout from the receiving side --
// deliberately not shared with QuRTDeviceContext.cpp, so a packing change that
// breaks the contract breaks a test instead of updating both halves at once:
// pra[0] is the primary-input block (selector, thread mask, scalar-block
// length, then two lengths per buffer slot), pra[1] is the packed scalar block
// whose last bufferCount 8-byte words are each buffer's offset within its
// allocation, and pra[2+i] are the buffer slots, allocation-based.
//
// An invocation "executes" by writing a short pattern, deterministic in the
// selector and the slot, at each operand's offset -- so a recorder's
// post-hashes move the way a real kernel's would, identically on every run.
// Real kernels do not write their inputs; this writes every slot because the
// wire does not say which are outputs, and the tests account for it.
//
// Memory is plain aligned malloc, zero-filled: rpcmem gives no zeroing
// guarantee, but a deterministic starting state is what makes two runs'
// pre-hashes comparable in a test.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
  void *pv;
  size_t nLen;
} remote_buf;

enum {
  kBufferSlots = 8,
  kDispatchMethod = 3,
  kBadParm = 0x8000040E,
  /* `AEE_ECONNREFUSED`, as `qurtStatusMessage` names it: a status that is NOT
     the missing-row refusal, for tests that must tell the two apart. */
  kConnRefused = 0x80000472,
  /* `NPU_LAUNCH_TIME` in mojo_npu_imp.c, restated here for the same reason the
     wire layout is: a flag that stopped agreeing should break a test. */
  kLaunchTime = 0x00010000,
  /* `NPU_LAUNCH_BATCH`, and the block a batch sends instead of one kernel's
     scalars: a 16-byte header (magic, count, report slot, reserved) then a
     24-byte record per launch (selector, thread mask, argument offset, slot
     map, arity, reserved). Restated, not shared. */
  kLaunchBatch = 0x00040000,
  kBatchMagic = 0x4E425443,
  kBatchHeaderBytes = 16,
  kBatchRecordBytes = 24,
  kBatchNoSlot = -1,
};

/* The selector a batch travels under. A skel built before batching finds no
   manifest row for it and refuses; NPU_FAKE_TRANSPORT_NO_BATCH makes this
   stand-in do the same, which is how a test can check that a batch reaching an
   older library is refused rather than mis-run. */
#define kBatchSelector 0xFFFFFFFFu

/* The megastep row's selector and block, restated from NPUKernels.def and
   mojo_npu_imp.c like everything else on this wire: a 16-byte header (magic,
   count, report slot, mode), then either 8-byte register entries (table
   index, slot) or 32-byte run records (selector, thread mask, argument
   offset, arity, eight 16-bit operand references -- call slot, or session
   table entry under the high bit). NPU_FAKE_TRANSPORT_NO_MEGASTEP emulates a
   skel signed before the row existed, which refuses the selector by name. */
enum {
  kMegastepSelector = 39,
  kMegastepMagic = 0x4E4D4753,
  kMegastepHeaderBytes = 16,
  kMegastepRecordBytes = 32,
  kMegastepEntryBytes = 8,
  kMegastepModeRegister = 0,
  kMegastepModeRun = 1,
  kMegastepMaxLaunches = 1024,
  kMegastepTableEntries = 512,
  kMegastepRefTable = 0x8000,
};

/* The power_vote row's selector and report shape, restated from
   NPUKernels.def and mojo_npu_imp.c. The stand-in answers it the way the
   device body does -- word 0 residency, words 1-4 zero-status votes, word 5
   the HMX capability marker -- under three knobs:
   NPU_FAKE_TRANSPORT_NO_POWER_VOTE emulates a skel signed before the row
   existed (refuses the selector by name), NPU_FAKE_TRANSPORT_POWER_NOT_RESIDENT
   a protection domain without HAP_power_set (word 0 reads 0), and
   NPU_FAKE_TRANSPORT_HMX_BUILD an HMX_MATMUL=1 build (word 5 reads 'HMX1'). */
enum {
  kPowerVoteSelector = 37,
  kPowerVoteHmxMarker = 0x484D5831,
};

/* The qtimer's fixed rate, as the device reports it. */
#define kFakeQtimerHz 19200000

/* How long an invocation blocks, and how much of that it claims the device was
   busy for, in microseconds. Both zero by default, so the recorder tests that
   predate this see the invocation they always did. A harness that says where a
   launch's time went has to be checkable against time that was actually spent
   somewhere known, which is what these two inject. */
static long fake_env_us(const char *name) {
  const char *value = getenv(name);
  if (value == NULL || *value == '\0')
    return 0;
  return strtol(value, NULL, 10);
}

static void fake_sleep_us(long micros) {
  if (micros <= 0)
    return;
  struct timespec request;
  request.tv_sec = micros / 1000000;
  request.tv_nsec = (micros % 1000000) * 1000;
  while (nanosleep(&request, &request) == -1)
    ;
}

static int fake_env_flag(const char *name) {
  const char *value = getenv(name);
  return value != NULL && value[0] == '1' && value[1] == '\0';
}

/* The buffer-attribute pair the weights-NOFLUSH classification binds as
   optional symbols. Each call appends one line to the file named by
   NPU_FAKE_TRANSPORT_ATTR_LOG, which is the test's observable: what the
   classifier decided is exactly what crossed this seam, in order. The fd is
   pointer-derived so a log line ties an attr2 call to the to_fd answer for
   the same buffer without leaning on call order. NPU_FAKE_TRANSPORT_NO_FD=1
   makes rpcmem_to_fd refuse, standing in for a driver without descriptors. */
static void fake_attr_log(const char *fmt, const void *ptr, long long a,
                          long long b, long long c) {
  const char *path = getenv("NPU_FAKE_TRANSPORT_ATTR_LOG");
  if (path == NULL || *path == '\0')
    return;
  FILE *log = fopen(path, "a");
  if (log == NULL)
    return;
  fprintf(log, fmt, ptr, a, b, c);
  fclose(log);
}

static int fake_fd_for(const void *ptr) {
  return (int)(((uintptr_t)ptr >> 4) & 0xFFFFF) | 1;
}

__attribute__((visibility("default"))) int rpcmem_to_fd(void *ptr) {
  if (fake_env_flag("NPU_FAKE_TRANSPORT_NO_FD"))
    return -1;
  int fd = fake_fd_for(ptr);
  fake_attr_log("to_fd %p -> %lld\n", ptr, (long long)fd, 0, 0);
  return fd;
}

/* ===------------------------------------------------------------------=== */
/* The vendor queue, modeled to the depth order 081 paid for                */
/* ===------------------------------------------------------------------=== */

/* The one behaviour worth standing in for is the one silicon refused on:
   dspqueue's buffer bookkeeping is keyed by FILE DESCRIPTOR and populated by
   `fastrpc_mmap` with `FASTRPC_MAP_FD` -- a packet naming any other fd is
   refused with AEE_ENOSUCHMAP (host numbering 72), which is order 081's
   presentation (a STATIC mapping never lands in the fd table). The write-side
   check models the same bookkeeping the DSP read resolves from; the header
   documents ENOSUCHMAP on both calls with the same sentence. Restated, not
   shared: the frame header offsets and the response shape are the wire's, and
   a layout change should break this test, not update both halves at once.

   One queue, one packet in flight, completion answered on the next read --
   the depth a host integration test needs and no more. No launches run; the
   completion echoes the frame's own batch id with status 0, which is exactly
   what the transport needs to free its ring slot. */

enum {
  kFakeMapFD = 2,       /* FASTRPC_MAP_FD */
  kFakeENoSuchMap = 72, /* AEE_ENOSUCHMAP, host numbering */
  kFakeEWouldBlock = 516,
  kFakeMappings = 64,
  kFakeFrameBatchOffset = 8, /* npu_queue_frame_header_t.batch_id */
};

typedef struct {
  uint32_t fd;
  uint32_t size;
  uint32_t offset;
  uint32_t flags;
  union {
    void *ptr;
    uint64_t address;
  };
} fake_dspqueue_buffer;

static struct {
  int fd;
  int flags;
} fake_mapped[kFakeMappings];
static int fake_mapped_count;

static struct {
  uint32_t batch_id;
  int32_t status;
} fake_queue_response;
static int fake_queue_response_ready;
static int fake_queue_open;

__attribute__((visibility("default"))) int fastrpc_mmap(int domain, int fd,
                                                        void *addr, int offset,
                                                        size_t length,
                                                        int flags) {
  (void)domain;
  (void)offset;
  /* Logged with the mode that was ASKED for, before the static world takes it
     away: whether a caller maps once and in which mode is the host-visible
     property a map-once leg is judged by. */
  fake_attr_log("mmap %p %lld %lld %lld\n", addr, (long long)fd,
                (long long)length, (long long)flags);
  /* NPU_FAKE_TRANSPORT_STATIC_WORLD stands in for order 081's firmware: the
     mapping call succeeds, but whatever mode was requested never lands in the
     fd table the queue resolves from -- so the write refuses ENOSUCHMAP, the
     silicon presentation, host-visible. */
  if (fake_env_flag("NPU_FAKE_TRANSPORT_STATIC_WORLD"))
    flags = 0;
  for (int i = 0; i < fake_mapped_count; ++i)
    if (fake_mapped[i].fd == fd) {
      fake_mapped[i].flags = flags;
      return 0;
    }
  if (fake_mapped_count >= kFakeMappings)
    return kFakeENoSuchMap;
  fake_mapped[fake_mapped_count].fd = fd;
  fake_mapped[fake_mapped_count].flags = flags;
  ++fake_mapped_count;
  return 0;
}

__attribute__((visibility("default"))) int
fastrpc_munmap(int domain, int fd, void *addr, size_t length) {
  (void)domain;
  fake_attr_log("munmap %p %lld %lld %lld\n", addr, (long long)fd,
                (long long)length, 0);
  for (int i = 0; i < fake_mapped_count; ++i)
    if (fake_mapped[i].fd == fd) {
      fake_mapped[i] = fake_mapped[fake_mapped_count - 1];
      --fake_mapped_count;
      return 0;
    }
  return kFakeENoSuchMap;
}

static int fake_fd_resolvable(uint32_t fd) {
  for (int i = 0; i < fake_mapped_count; ++i)
    if (fake_mapped[i].fd == (int)fd)
      return fake_mapped[i].flags == kFakeMapFD;
  return 0;
}

__attribute__((visibility("default"))) int
dspqueue_create(int domain, uint32_t flags, uint32_t reqBytes,
                uint32_t rspBytes, void (*packetCallback)(void *, int, void *),
                void (*errorCallback)(void *, int, void *), void *context,
                void **queue) {
  (void)domain;
  (void)flags;
  (void)reqBytes;
  (void)rspBytes;
  (void)packetCallback;
  (void)errorCallback;
  (void)context;
  fake_queue_open = 1;
  fake_queue_response_ready = 0;
  *queue = &fake_queue_open;
  return 0;
}

__attribute__((visibility("default"))) int dspqueue_export(void *queue,
                                                           uint64_t *queueID) {
  (void)queue;
  /* Low word zero on purpose: 081 measured that shape as a real property of
     the id on this firmware, and a test id should look like a real one. */
  *queueID = 0x0000358900000000ULL;
  return 0;
}

__attribute__((visibility("default"))) int dspqueue_close(void *queue) {
  (void)queue;
  fake_queue_open = 0;
  return 0;
}

__attribute__((visibility("default"))) int
dspqueue_write(void *queue, uint32_t flags, uint32_t bufferCount,
               fake_dspqueue_buffer *buffers, uint32_t messageBytes,
               const uint8_t *message, uint32_t timeoutMicros) {
  (void)queue;
  (void)flags;
  (void)messageBytes;
  (void)message;
  (void)timeoutMicros;
  /* The fd-keyed refusal the whole model exists for. */
  for (uint32_t i = 0; i < bufferCount; ++i)
    if (!fake_fd_resolvable(buffers[i].fd))
      return kFakeENoSuchMap;
  if (bufferCount == 0 || buffers[0].ptr == NULL)
    return kFakeENoSuchMap;
  const uint8_t *frame = (const uint8_t *)buffers[0].ptr;
  memcpy(&fake_queue_response.batch_id, frame + kFakeFrameBatchOffset,
         sizeof(uint32_t));
  fake_queue_response.status = 0;
  fake_queue_response_ready = 1;
  return 0;
}

__attribute__((visibility("default"))) int
dspqueue_read_noblock(void *queue, uint32_t *flags, uint32_t maxBuffers,
                      uint32_t *bufferCount, fake_dspqueue_buffer *buffers,
                      uint32_t maxMessageBytes, uint32_t *messageBytes,
                      uint8_t *message) {
  (void)queue;
  (void)maxBuffers;
  (void)buffers;
  if (!fake_queue_response_ready)
    return kFakeEWouldBlock;
  if (maxMessageBytes < sizeof(fake_queue_response))
    return kFakeENoSuchMap;
  *flags = 0x0001; /* message */
  *bufferCount = 0;
  *messageBytes = sizeof(fake_queue_response);
  memcpy(message, &fake_queue_response, sizeof(fake_queue_response));
  fake_queue_response_ready = 0;
  return 0;
}

__attribute__((visibility("default"))) void
remote_register_buf_attr2(void *buf, size_t size, int fd, int attr) {
  fake_attr_log("attr2 %p %lld %lld %lld\n", buf, (long long)size,
                (long long)fd, (long long)attr);
}

__attribute__((visibility("default"))) void rpcmem_init(void) {}

/* Allocation and batch-width bookkeeping. The host's MemoryManager is disabled
   by construction on this backend, so every device buffer is one rpcmem_alloc
   here and every release one rpcmem_free -- which makes this the one place a
   host test can count them without a device. Read through
   `npu_fake_transport_stats`, since this library is dlopen'd into the calling
   process rather than talking to one. */
static uint64_t fake_allocs;
static uint64_t fake_frees;
static uint64_t fake_alloc_bytes;
static uint64_t fake_batches;
static uint64_t fake_batched_launches;
static uint64_t fake_single_launches;

__attribute__((visibility("default"))) void
npu_fake_transport_stats(uint64_t *out, int count) {
  const uint64_t all[] = {fake_allocs,           fake_frees,
                          fake_alloc_bytes,      fake_batches,
                          fake_batched_launches, fake_single_launches};
  const int have = (int)(sizeof(all) / sizeof(*all));
  for (int i = 0; i < count && i < have; ++i)
    out[i] = all[i];
}

__attribute__((visibility("default"))) void
npu_fake_transport_stats_reset(void) {
  fake_allocs = 0;
  fake_frees = 0;
  fake_alloc_bytes = 0;
  fake_batches = 0;
  fake_batched_launches = 0;
  fake_single_launches = 0;
}

__attribute__((visibility("default"))) void *
rpcmem_alloc(int heapid, uint32_t flags, int size) {
  void *ptr = NULL;
  if (size <= 0 || posix_memalign(&ptr, 4096, (size_t)size) != 0)
    return NULL;
  memset(ptr, 0, (size_t)size);
  ++fake_allocs;
  fake_alloc_bytes += (uint64_t)size;
  return ptr;
}

__attribute__((visibility("default"))) void rpcmem_free(void *ptr) {
  ++fake_frees;
  free(ptr);
}

__attribute__((visibility("default"))) int
remote_session_control(uint32_t req, void *data, uint32_t datalen) {
  return 0;
}

__attribute__((visibility("default"))) int
remote_handle64_open(const char *name, uint64_t *handle) {
  *handle = 0x5AFE;
  return 0;
}

__attribute__((visibility("default"))) int remote_handle64_close(uint64_t h) {
  return 0;
}

/* The QoS control the MODULAR_NPU_POWER host half votes through, logged to
   the same seam file the attribute calls use. Compiled out of the
   no-control variant (see BUILD.bazel), which is how a test exercises the
   named-absence report against a genuinely missing dlsym rather than a
   refusing stub. */
#ifndef NPU_FAKE_NO_CONTROL
__attribute__((visibility("default"))) int
remote_handle64_control(uint64_t handle, uint32_t req, void *data,
                        uint32_t datalen) {
  uint32_t enable = 0;
  uint32_t latency = 0;
  if (data != NULL && datalen >= 2 * sizeof(uint32_t)) {
    memcpy(&enable, data, sizeof(enable));
    memcpy(&latency, (uint8_t *)data + sizeof(enable), sizeof(latency));
  }
  fake_attr_log("control %p %lld %lld %lld\n", (void *)(uintptr_t)handle,
                (long long)req, (long long)enable, (long long)latency);
  return 0;
}
#endif

/* The kernel's own scalars folded to one byte, so the pattern below depends
   on them: a transport that delivers the right operands at the right offsets
   but mangles the packed scalar block -- the failure a real kernel would
   express as computing on garbage extents -- breaks the differential instead
   of passing it. */
static uint8_t fake_scalar_hash(const uint8_t *scalars, size_t bytes) {
  uint8_t hash = 0;
  for (size_t j = 0; j < bytes; ++j)
    hash = (uint8_t)((hash * 31u) ^ scalars[j]);
  return hash;
}

/* What an "execution" leaves behind, in one place because a batched launch and
   a per-call launch have to leave the same bytes or the differential test is
   comparing two computations rather than two transports. */
static int fake_write_operand(uint8_t *base, size_t allocBytes, uint64_t offset,
                              uint32_t selector, uint32_t operand,
                              uint8_t scalarHash) {
  if (base == NULL || offset >= allocBytes)
    return 0;
  size_t count = allocBytes - (size_t)offset;
  if (count > 64)
    count = 64;
  for (size_t j = 0; j < count; ++j)
    base[offset + j] = (uint8_t)(0xA5u ^ (selector * 7u) ^ (operand * 13u) ^
                                 scalarHash ^ (uint32_t)j);
  return 1;
}

/* The witness words a test needs the "kernel" to leave behind. The generic
   pattern above fills the first 64 bytes of every operand, which lands
   deterministic garbage in the witness slots a graph stub reads -- useful as a
   refusal, useless as a pass. These two knobs write named words over the
   pattern, in the LAST operand (the witness, by the NPU_KERNEL convention):

     NPU_FAKE_TRANSPORT_WITNESS_ALL="7=1212895793,22=2"
     NPU_FAKE_TRANSPORT_WITNESS_AT="2:7=3"

   ALL applies to every launch of the invoke; AT to one launch position within
   it, which is how a test presents a refusal in the MIDDLE of a batch. */
static void fake_apply_witness(const char *spec, uint8_t *base,
                               size_t allocBytes, uint64_t offset) {
  if (spec == NULL || base == NULL)
    return;
  while (*spec != '\0') {
    char *end = NULL;
    unsigned long word = strtoul(spec, &end, 0);
    if (end == spec || *end != '=')
      return;
    spec = end + 1;
    unsigned long value = strtoul(spec, &end, 0);
    if (end == spec)
      return;
    spec = end;
    uint64_t at = offset + (uint64_t)word * sizeof(uint32_t);
    if (at + sizeof(uint32_t) <= allocBytes) {
      uint32_t written = (uint32_t)value;
      memcpy(base + at, &written, sizeof(written));
    }
    if (*spec == ',')
      ++spec;
    else if (*spec != '\0')
      return;
  }
}

static void fake_witness_words(uint32_t launchIndex, uint8_t *base,
                               size_t allocBytes, uint64_t offset) {
  fake_apply_witness(getenv("NPU_FAKE_TRANSPORT_WITNESS_ALL"), base, allocBytes,
                     offset);
  const char *at = getenv("NPU_FAKE_TRANSPORT_WITNESS_AT");
  if (at == NULL)
    return;
  char *end = NULL;
  unsigned long want = strtoul(at, &end, 0);
  if (end == at || *end != ':' || want != launchIndex)
    return;
  fake_apply_witness(end + 1, base, allocBytes, offset);
}

/* One line per invocation naming how many launches it carried, so a test can
   tell a batch of four from four dispatches without reading a clock. */
static void fake_count_invoke(uint32_t launches) {
  const char *path = getenv("NPU_FAKE_TRANSPORT_COUNT");
  if (path == NULL || *path == '\0')
    return;
  FILE *out = fopen(path, "a");
  if (out == NULL)
    return;
  fprintf(out, "%u\n", launches);
  fclose(out);
}

static uint32_t fake_read_u32(const uint8_t *at) {
  uint32_t value = 0;
  memcpy(&value, at, sizeof(value));
  return value;
}

/* The batch block, decoded the way the device trampoline decodes it. */
static int fake_run_batch(const uint8_t *block, size_t blockBytes,
                          remote_buf *pra, int *reportSlot) {
  *reportSlot = kBatchNoSlot;
  if (blockBytes < (size_t)kBatchHeaderBytes ||
      fake_read_u32(block) != (uint32_t)kBatchMagic)
    return (int)kBadParm;
  uint32_t count = fake_read_u32(block + 4);
  uint32_t slot = fake_read_u32(block + 8);
  if (count == 0 ||
      blockBytes < (size_t)kBatchHeaderBytes + count * kBatchRecordBytes)
    return (int)kBadParm;
  if (slot < kBufferSlots)
    *reportSlot = (int)slot;

  for (uint32_t i = 0; i < count; ++i) {
    const uint8_t *record = block + kBatchHeaderBytes + i * kBatchRecordBytes;
    uint32_t selector = fake_read_u32(record);
    uint32_t argsOffset = fake_read_u32(record + 8);
    uint32_t slotMap = fake_read_u32(record + 12);
    uint32_t arity = fake_read_u32(record + 16);
    uint32_t buffers = arity & 0xFFFFu;
    uint32_t scalars = arity >> 16;
    if (buffers > kBufferSlots ||
        argsOffset + (uint64_t)(buffers + scalars) * 8u > blockBytes)
      return (int)kBadParm;

    const uint8_t *args = block + argsOffset;
    uint8_t scalarHash = fake_scalar_hash(args, scalars * 8u);
    for (uint32_t j = 0; j < buffers; ++j) {
      uint32_t at = (slotMap >> (4 * j)) & 0xFu;
      if (at >= kBufferSlots)
        return (int)kBadParm;
      uint64_t offset = 0;
      memcpy(&offset, args + (scalars + j) * 8u, sizeof(offset));
      if (!fake_write_operand((uint8_t *)pra[2 + at].pv, pra[2 + at].nLen,
                              offset, selector, j, scalarHash))
        return (int)kBadParm;
      if (j + 1 == buffers)
        fake_witness_words(i, (uint8_t *)pra[2 + at].pv, pra[2 + at].nLen,
                           offset);
    }
  }
  fake_count_invoke(count);
  return 0;
}

/* The session table register-mode calls fill and run records name. Per
   process, like the device's own .bss table is per protection domain. */
static struct {
  uint8_t *base;
  size_t bytes;
} fake_megastep_table[kMegastepTableEntries];

static uint32_t fake_read_u16(const uint8_t *at) {
  uint16_t value = 0;
  memcpy(&value, at, sizeof(value));
  return value;
}

/* The megastep block, decoded the way the device walker decodes it: register
   entries store this call's slots into the session table; run records resolve
   each operand reference -- call slot or table entry -- and leave the same
   bytes a per-call dispatch of the same launch leaves, which is what the
   differential test compares. */
static int fake_run_megastep(const uint8_t *block, size_t blockBytes,
                             remote_buf *pra) {
  if (blockBytes < (size_t)kMegastepHeaderBytes ||
      fake_read_u32(block) != (uint32_t)kMegastepMagic)
    return (int)kBadParm;
  uint32_t count = fake_read_u32(block + 4);
  uint32_t mode = fake_read_u32(block + 12);

  if (mode == (uint32_t)kMegastepModeRegister) {
    if (count == 0 || count > kBufferSlots ||
        blockBytes < (size_t)kMegastepHeaderBytes + count * kMegastepEntryBytes)
      return (int)kBadParm;
    for (uint32_t i = 0; i < count; ++i) {
      const uint8_t *entry =
          block + kMegastepHeaderBytes + i * kMegastepEntryBytes;
      uint32_t index = fake_read_u32(entry);
      uint32_t slot = fake_read_u32(entry + 4);
      if (index >= kMegastepTableEntries || slot >= kBufferSlots ||
          pra[2 + slot].pv == NULL)
        return (int)kBadParm;
      fake_megastep_table[index].base = (uint8_t *)pra[2 + slot].pv;
      fake_megastep_table[index].bytes = pra[2 + slot].nLen;
    }
    /* An invocation that ran no launches, so a count line of 0: the test's
       ledger separates the session's registrations from the step's one run. */
    fake_count_invoke(0);
    return 0;
  }

  if (mode != (uint32_t)kMegastepModeRun)
    return (int)kBadParm;
  /* A device that accepts registrations but refuses every run, for the
     poison test: the refusal has to reach a caller that cannot raise it (a
     free from a destructor) and still stop the render at the next launch. */
  if (fake_env_flag("NPU_FAKE_TRANSPORT_REFUSE_MEGASTEP_RUN"))
    return (int)kBadParm;
  if (count == 0 || count > kMegastepMaxLaunches ||
      blockBytes < (size_t)kMegastepHeaderBytes + count * kMegastepRecordBytes)
    return (int)kBadParm;

  for (uint32_t i = 0; i < count; ++i) {
    const uint8_t *record =
        block + kMegastepHeaderBytes + i * kMegastepRecordBytes;
    uint32_t selector = fake_read_u32(record);
    uint32_t argsOffset = fake_read_u32(record + 8);
    uint32_t arity = fake_read_u32(record + 12);
    uint32_t buffers = arity & 0xFFFFu;
    uint32_t scalars = arity >> 16;
    if (buffers > kBufferSlots ||
        argsOffset + (uint64_t)(buffers + scalars) * 8u > blockBytes)
      return (int)kBadParm;

    const uint8_t *args = block + argsOffset;
    uint8_t scalarHash = fake_scalar_hash(args, scalars * 8u);
    for (uint32_t j = 0; j < buffers; ++j) {
      uint32_t ref = fake_read_u16(record + 16 + 2 * j);
      uint8_t *base;
      size_t bytes;
      if ((ref & kMegastepRefTable) != 0) {
        uint32_t at = ref & (kMegastepRefTable - 1);
        if (at >= kMegastepTableEntries || fake_megastep_table[at].base == NULL)
          return (int)kBadParm;
        base = fake_megastep_table[at].base;
        bytes = fake_megastep_table[at].bytes;
      } else {
        if (ref >= kBufferSlots || pra[2 + ref].pv == NULL)
          return (int)kBadParm;
        base = (uint8_t *)pra[2 + ref].pv;
        bytes = pra[2 + ref].nLen;
      }
      uint64_t offset = 0;
      memcpy(&offset, args + (scalars + j) * 8u, sizeof(offset));
      if (!fake_write_operand(base, bytes, offset, selector, j, scalarHash))
        return (int)kBadParm;
    }
  }
  fake_count_invoke(count);
  return 0;
}

__attribute__((visibility("default"))) int
remote_handle64_invoke(uint64_t handle, uint32_t dsc, remote_buf *pra) {
  uint32_t method = (dsc >> 24) & 0x1F;
  if (method != kDispatchMethod)
    return 0;

  const uint32_t *primIn = (const uint32_t *)pra[0].pv;
  const uint8_t *scalars = (const uint8_t *)pra[1].pv;
  if (primIn == NULL || pra[0].nLen < 3 * sizeof(uint32_t) || scalars == NULL)
    return (int)kBadParm;

  uint32_t selector = primIn[0];
  uint32_t launchWord = primIn[1];
  uint32_t scalarBytes = primIn[2];
  if ((size_t)scalarBytes > pra[1].nLen)
    return (int)kBadParm;

  /* A library signed before an APPENDED row has no case for that selector and
     refuses at its argument check, ahead of the fan-out and of any write
     (mojo_npu_imp.c's npu_run_launch). NPU_FAKE_TRANSPORT_NO_SELECTOR names
     one such row, which is the only way to present a host whose own manifest
     carries a kernel to a device that does not -- the asymmetry `loadFunction`
     cannot see, since it resolves against the host's kNPUKernels. */
  {
    const char *absent = getenv("NPU_FAKE_TRANSPORT_NO_SELECTOR");
    if (absent != NULL && (uint32_t)strtoul(absent, NULL, 0) == selector)
      return (int)kBadParm;
  }

  /* The OTHER way an invoke fails: the row is there and the transport is not.
     A caller that treats every invoke failure as "the row is missing" would
     disable a working kernel for the rest of the process on a transient, so
     the two have to be presentable separately. */
  {
    const char *faulted = getenv("NPU_FAKE_TRANSPORT_FAULT_SELECTOR");
    if (faulted != NULL && (uint32_t)strtoul(faulted, NULL, 0) == selector)
      return (int)kConnRefused;
  }

  /* A library predating batching has no manifest row for the batch selector,
     so it refuses. Emulated rather than assumed, because "an older skel
     refuses instead of mis-running" is the forward-compatibility claim the
     block's design rests on. */
  if (getenv("NPU_FAKE_TRANSPORT_NO_BATCH") != NULL &&
      selector == kBatchSelector)
    return (int)kBadParm;

  /* The power vote, answered as the device body answers it rather than with
     the generic operand pattern, so the host's latched outcome -- vote
     liveness and the skel's HMX capability -- is testable at this seam. */
  if (selector == kPowerVoteSelector) {
    if (getenv("NPU_FAKE_TRANSPORT_NO_POWER_VOTE") != NULL)
      return (int)kBadParm;
    uint32_t *report = (uint32_t *)pra[2].pv;
    size_t words = pra[2].nLen / sizeof(uint32_t);
    if (report != NULL) {
      memset(report, 0, words * sizeof(uint32_t));
      if (words > 0)
        report[0] =
            getenv("NPU_FAKE_TRANSPORT_POWER_NOT_RESIDENT") == NULL ? 1 : 0;
      if (words > 5 && getenv("NPU_FAKE_TRANSPORT_HMX_BUILD") != NULL)
        report[5] = (uint32_t)kPowerVoteHmxMarker;
    }
    fake_count_invoke(1);
    return 0;
  }

  /* A library signed before the megastep row refuses its selector by name;
     emulated for the same reason the batch's is. */
  if (getenv("NPU_FAKE_TRANSPORT_NO_MEGASTEP") != NULL &&
      selector == (uint32_t)kMegastepSelector)
    return (int)kBadParm;

  if (selector == (uint32_t)kMegastepSelector) {
    fake_sleep_us(fake_env_us("NPU_FAKE_TRANSPORT_INVOKE_US"));
    return fake_run_megastep(scalars, scalarBytes, pra);
  }

  if ((launchWord & kLaunchBatch) != 0) {
    if (selector != kBatchSelector)
      return (int)kBadParm;
    long batchBlockUs = fake_env_us("NPU_FAKE_TRANSPORT_INVOKE_US");
    fake_sleep_us(batchBlockUs);
    int reportSlot = kBatchNoSlot;
    ++fake_batches;
    if (scalarBytes >= kBatchHeaderBytes)
      fake_batched_launches += fake_read_u32(scalars + 4);
    int status = fake_run_batch(scalars, scalarBytes, pra, &reportSlot);
    (void)reportSlot;
    return status;
  }

  ++fake_single_launches;

  // The packer fills the leading slots and leaves the rest empty.
  uint32_t bufferCount = 0;
  while (bufferCount < kBufferSlots && pra[2 + bufferCount].pv != NULL)
    ++bufferCount;

  // A launch that asked to be timed sends one slot past the kernel's own, and
  // that slot is not an operand: it carries no offset in the scalar block and
  // no kernel writes it. Taking it off the count here is what keeps the offsets
  // decoding from the block's tail, as they do without it.
  uint8_t *timingSlot = NULL;
  size_t timingBytes = 0;
  if ((launchWord & kLaunchTime) != 0 && bufferCount > 0) {
    --bufferCount;
    timingSlot = (uint8_t *)pra[2 + bufferCount].pv;
    timingBytes = pra[2 + bufferCount].nLen;
  }
  if (scalarBytes < bufferCount * 8u)
    return (int)kBadParm;

  long blockUs = fake_env_us("NPU_FAKE_TRANSPORT_INVOKE_US");
  long deviceUs = fake_env_us("NPU_FAKE_TRANSPORT_DEVICE_US");
  fake_sleep_us(blockUs);
  if (timingSlot != NULL && timingBytes >= 6 * sizeof(uint64_t)) {
    /* `npu_launch_timing_t`: four pcycle stamps then two qtimer stamps, as raw
       counter reads. Only the qtimer pair is synthesised against a rate, since
       that is the pair a host converts to nanoseconds. */
    uint64_t stamps[6];
    uint64_t ticks = (uint64_t)deviceUs * kFakeQtimerHz / 1000000u;
    stamps[0] = 1000;
    stamps[1] = 1000 + selector;
    stamps[2] = 1000 + selector + 10;
    stamps[3] = 1000 + selector + 20;
    stamps[4] = 500000;
    stamps[5] = 500000 + ticks;
    memcpy(timingSlot, stamps, sizeof(stamps));

    /* The session sync block (`npu_sync_report.h`, 80 bytes): after the phase
       record when the slot holds that whole, directly after the stamps when
       not -- the size-decides-layout rule `qshimPhaseCopy` writes by,
       restated. The phase region, when present, stays zeroed: a probe-less
       skel's phases read back as absent while the sync block still travels,
       which is the PERF_DEVICE shape the counter leg reads. */
    {
      size_t syncAt = sizeof(stamps);
      if (timingBytes - sizeof(stamps) >= 2432u)
        syncAt += 2432u;
      if (timingBytes >= syncAt + 80u) {
        uint32_t head[2] = {0x434E5953u /* "SYNC" */, 1u};
        uint64_t counts[9] = {9u,  1u,      5u,     2u,     1u,
                              17u, 123456u, 96000u, 32000u};
        memcpy(timingSlot + syncAt, head, sizeof(head));
        memcpy(timingSlot + syncAt + sizeof(head), counts, sizeof(counts));
      }
    }
  }

  uint8_t scalarHash =
      fake_scalar_hash(scalars, scalarBytes - (size_t)bufferCount * 8);
  for (uint32_t i = 0; i < bufferCount; ++i) {
    uint64_t offset;
    memcpy(&offset, scalars + scalarBytes - (size_t)(bufferCount - i) * 8, 8);
    if (!fake_write_operand((uint8_t *)pra[2 + i].pv, pra[2 + i].nLen, offset,
                            selector, i, scalarHash))
      return (int)kBadParm;
    if (i + 1 == bufferCount)
      fake_witness_words(0, (uint8_t *)pra[2 + i].pv, pra[2 + i].nLen, offset);
  }
  fake_count_invoke(1);
  return 0;
}
