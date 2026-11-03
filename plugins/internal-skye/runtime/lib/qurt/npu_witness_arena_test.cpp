//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
// The witness arena's two claims, on the host, with no device.
//
//   1. A LAUNCH READS ITS OWN SLOT. One slot per in-flight launch, indexed out
//      of one process-lifetime allocation, so a batch of N carries N distinct
//      witnesses. The mutation that must fail is the tempting simplification:
//      point every launch at one slot and watch the refusal land on the wrong
//      op -- or vanish, when the last writer overwrites it.
//
//   2. A GUARD IS EVALUATED AFTER THE LAUNCH RAN, AND ITS CLASS DECIDES WHAT
//      THAT COSTS. A correctness guard records a pending error naming the
//      originating op; an attribution guard on the same slot records loudly and
//      the flush still succeeds. The pending error is reported by the next
//      raising caller, never dropped, and shouted about at teardown if nobody
//      ever asks.
//
// The arena is written by the Mojo graph stubs (`dispatch.mojo`,
// `registrations.mojo`), which this test cannot run: it builds the same layout
// itself, which is also what keeps the layout honest -- three writers now, all
// of which have to agree.
//
//===----------------------------------------------------------------------===//

#include "QuRTDeviceContext.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <string>
#include <unistd.h>
#include <vector>

using namespace M;
using namespace M::Driver;

namespace {

int failures = 0;

void expect(bool condition, const std::string &what) {
  printf("%s: %s\n", condition ? "ok" : "FAIL", what.c_str());
  if (!condition)
    ++failures;
}

/// The arena layout, restated from `dispatch.mojo` for the reason the wire
/// layout in `npu_fake_transport.c` is restated: a change that stops agreeing
/// has to break something.
constexpr uint32_t kArenaMagic = 0x5749544Eu;
constexpr uint32_t kGuardMagic = 0x47554152u;
constexpr uint32_t kWitnessBytes = 128;
constexpr uint32_t kGuardBytes = 128;
constexpr uint32_t kSlotBytes = kWitnessBytes + kGuardBytes;
constexpr uint32_t kHeaderBytes = 128;
constexpr uint32_t kSlots = 64;
constexpr uint32_t kArenaBytes = kHeaderBytes + kSlots * kSlotBytes;
constexpr uint32_t kGuardHeaderWords = 8;
constexpr uint32_t kGuardWords = 12;
constexpr uint32_t kCondNone = 0xFFFFFFFFu;
constexpr uint32_t kClassCorrectness = 1;
constexpr uint32_t kClassAttribution = 2;

/// `MATMUL_PATH_HMX_RAN` and a refusal code, as the signed kernel spells them.
constexpr uint32_t kPathUnset = 0;
constexpr uint32_t kPathHmxRan = 0x484D5831u;
constexpr uint32_t kPathNoGrant = 3;
constexpr uint32_t kArmFromANewerSkel = 99;

constexpr size_t kOperandBytes = 4096;

/// Everything loud on this path goes to stderr, so a test that asks whether a
/// guard was RECORDED has to read it back. Without this, gate 3 would pass on
/// an attribution guard that did nothing at all -- "it did not abort" is
/// satisfied by silence.
struct CapturedStderr {
  int saved = -1;
  std::string path;

  void begin(const char *tag) {
    path = std::string("/tmp/npu_arena_stderr_") + tag + ".txt";
    fflush(stderr);
    saved = dup(STDERR_FILENO);
    int fd = open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0600);
    if (fd >= 0) {
      dup2(fd, STDERR_FILENO);
      close(fd);
    }
  }

  std::string end() {
    fflush(stderr);
    if (saved >= 0) {
      dup2(saved, STDERR_FILENO);
      close(saved);
      saved = -1;
    }
    std::string text;
    FILE *in = fopen(path.c_str(), "r");
    if (in != nullptr) {
      char buffer[512];
      while (fgets(buffer, sizeof(buffer), in) != nullptr)
        text += buffer;
      fclose(in);
    }
    remove(path.c_str());
    return text;
  }
};

uint32_t *arenaWords(void *base) { return static_cast<uint32_t *>(base); }

void initArena(void *base, uint32_t slots = kSlots) {
  uint32_t *w = arenaWords(base);
  memset(base, 0, kArenaBytes);
  w[1] = slots;
  w[2] = kSlotBytes;
  w[3] = kWitnessBytes;
  w[4] = 0;
  w[0] = kArenaMagic;
}

uint32_t *slotAt(void *base, uint32_t slot) {
  return arenaWords(base) + kHeaderBytes / 4 + slot * (kSlotBytes / 4);
}

uint32_t *guardRecord(uint32_t *slot) { return slot + kWitnessBytes / 4; }

/// One armed guard, as `_arm_guard` writes it.
void armGuard(uint32_t *slot, uint32_t guardClass, uint32_t word,
              uint32_t condWord, uint32_t condValue,
              const std::vector<uint32_t> &accepts, const char *op) {
  uint32_t *record = guardRecord(slot);
  uint32_t index = record[1];
  record[1] = index + 1;
  record[2] = static_cast<uint32_t>(strlen(op));
  auto address = reinterpret_cast<uint64_t>(op);
  record[4] = static_cast<uint32_t>(address & 0xFFFFFFFFu);
  record[5] = static_cast<uint32_t>(address >> 32);
  uint32_t *entry = record + kGuardHeaderWords + index * kGuardWords;
  entry[0] = guardClass;
  entry[1] = word;
  entry[2] = condWord;
  entry[3] = condValue;
  entry[4] = static_cast<uint32_t>(accepts.size());
  for (size_t i = 0; i < accepts.size(); ++i)
    entry[5 + i] = accepts[i];
  record[0] = kGuardMagic;
}

/// `_arm_matmul_ran`: CORRECTNESS on slot 7, unconditional.
void armMatmulRan(uint32_t *slot, const char *op) {
  armGuard(slot, kClassCorrectness, 7, kCondNone, 0, {kPathUnset, kPathHmxRan},
           op);
}

/// `_arm_unpack_arm_known`: ATTRIBUTION on slot 22, only on the HMX arm.
void armUnpackArm(uint32_t *slot, const char *op) {
  armGuard(slot, kClassAttribution, 22, 7, kPathHmxRan, {0, 1, 2, 3, 4}, op);
}

struct Harness {
  DeviceRef device;
  DeviceContextRef context;
  /// MODULAR_NPU_BATCH as it was before this arm, restored by `closeHarness`.
  std::string savedBatch;
  bool hadBatch = false;
  DeviceBufferRef arena;
  DeviceBufferRef operands;
  DeviceFunctionRef matmul;

  DeviceContext &ctx() { return *context; }
  void *arenaBase() { return arena->ptr(); }
};

/// A context of its own per arm: `batching_` latches on a context's FIRST
/// launch (`QuRTDeviceContext.cpp`), while `npuBatchRequested()` re-reads the
/// environment every call -- so a test that flips MODULAR_NPU_BATCH and reuses
/// one context silently runs both arms on one transport and reads as a clean
/// A/B (#442's footgun). The environment is restored after, so an arm cannot
/// leak its setting into the next.
bool openHarness(Harness &out, Device &device, bool batch) {
  const char *before = getenv("MODULAR_NPU_BATCH");
  out.hadBatch = before != nullptr;
  out.savedBatch = before == nullptr ? "" : before;
  setenv("MODULAR_NPU_BATCH", batch ? "1" : "0", 1);

  bool ok = false;
  ErrorOr<DeviceContextRef> context = device.createContext();
  if (!context.isError()) {
    out.context = context.takeValue();
    ErrorOr<DeviceBufferRef> arena = out.ctx().enqueueCreateBuffer(kArenaBytes);
    ErrorOr<DeviceBufferRef> operands =
        out.ctx().enqueueCreateBuffer(kOperandBytes * 4);
    ErrorOr<DeviceFunctionRef> matmul =
        out.ctx().loadFunction("", "matmul", "");
    if (!arena.isError() && !operands.isError() && !matmul.isError()) {
      out.arena = arena.takeValue();
      out.operands = operands.takeValue();
      out.matmul = matmul.takeValue();
      initArena(out.arenaBase());
      ok = true;
    }
  }

  // NOT restored here: `batching_` latches on the context's first LAUNCH, not
  // on its creation, so an arm that restored the environment before dispatching
  // would run on the other transport and read as a clean A/B.
  return ok;
}

void closeHarness(Harness &h) {
  if (h.hadBatch)
    setenv("MODULAR_NPU_BATCH", h.savedBatch.c_str(), 1);
  else
    unsetenv("MODULAR_NPU_BATCH");
}

/// One `matmul` dispatch whose witness is `slot` of the arena. Four buffers,
/// three scalars, exactly as the manifest row declares and the graph stub
/// packs them.
ErrorOrSuccess launch(Harness &h, uint32_t *witness) {
  auto *bytes = static_cast<uint8_t *>(h.operands->ptr());
  void *c = bytes;
  void *a = bytes + kOperandBytes;
  void *b = bytes + 2 * kOperandBytes;
  void *w = witness;
  int32_t m = 1, n = 64, k = 64;
  void *args[7] = {&c, &a, &b, &w, &m, &n, &k};
  size_t sizes[7] = {sizeof(void *), sizeof(void *),  sizeof(void *),
                     sizeof(void *), sizeof(int32_t), sizeof(int32_t),
                     sizeof(int32_t)};
  return h.ctx().enqueueFunctionExecDirect(*h.matmul, 1, 1, 1, 0x1F, 1, 1, 0,
                                           nullptr, 0, args, 7, sizes);
}

/// One `matmul` dispatch whose OUTPUT operand is `out` and whose input is
/// `in`. The stand-in stamps a deterministic pattern into every operand it is
/// handed, so `out` carries a mark that means "this launch has run" -- which
/// is all the gates below need to tell a produced buffer from an unproduced
/// one, without restating the stand-in's formula.
ErrorOrSuccess launchInto(Harness &h, void *out, void *in, uint32_t *witness) {
  auto *bytes = static_cast<uint8_t *>(h.operands->ptr());
  void *b = bytes + 2 * kOperandBytes;
  void *w = witness;
  int32_t m = 1, n = 64, k = 64;
  void *args[7] = {&out, &in, &b, &w, &m, &n, &k};
  size_t sizes[7] = {sizeof(void *), sizeof(void *),  sizeof(void *),
                     sizeof(void *), sizeof(int32_t), sizeof(int32_t),
                     sizeof(int32_t)};
  return h.ctx().enqueueFunctionExecDirect(*h.matmul, 1, 1, 1, 0x1F, 1, 1, 0,
                                           nullptr, 0, args, 7, sizes);
}

/// THE MUTATION SEAM. `sharedSlot` is the "share one slot to save arena bytes"
/// simplification the charter forbids: every launch of the batch points at
/// slot 0. It must break gate 1.
uint32_t *slotFor(Harness &h, uint32_t launchIndex, bool sharedSlot) {
  return slotAt(h.arenaBase(), sharedSlot ? 0 : launchIndex);
}

/// GATE 1 -- a launch reads its own slot under batching. Four launches, the
/// third one refused; the error must name the third one's op and no other.
void gateOwnSlot(Device &device, bool sharedSlot) {
  Harness h;
  if (!openHarness(h, device, /*batch=*/true)) {
    expect(false, "gate 1: harness");
    return;
  }
  std::string ran = "7=" + std::to_string(kPathHmxRan);
  setenv("NPU_FAKE_TRANSPORT_WITNESS_ALL", ran.c_str(), 1);
  setenv("NPU_FAKE_TRANSPORT_WITNESS_AT",
         ("2:7=" + std::to_string(kPathNoGrant)).c_str(), 1);

  static const char *ops[4] = {"mo.matmul", "mo.batch_matmul", "mo.conv",
                               "mo.matmul"};
  for (uint32_t i = 0; i < 4; ++i) {
    uint32_t *slot = slotFor(h, i, sharedSlot);
    memset(slot, 0, kSlotBytes);
    armMatmulRan(slot, ops[i]);
    ErrorOrSuccess sent = launch(h, slot);
    expect(!sent.isError(),
           "gate 1: launch " + std::to_string(i) + " enqueued");
  }
  ErrorOrSuccess flushed = h.ctx().synchronize();
  unsetenv("NPU_FAKE_TRANSPORT_WITNESS_ALL");
  unsetenv("NPU_FAKE_TRANSPORT_WITNESS_AT");

  bool raised = flushed.isError();
  std::string message = raised ? std::string(flushed.getError()) : "";
  printf("   gate 1 (%s): %s\n", sharedSlot ? "shared slot" : "own slot",
         raised ? message.c_str() : "no error");
  bool namesConv = message.find("mo.conv") != std::string::npos;
  bool namesTheThird = message.find("launch 3 of 4") != std::string::npos;
  if (sharedSlot) {
    // The mutant: one slot means the refusal is attributed to whichever launch
    // happened to be armed last, so the op name is wrong -- and with the
    // accepted code written over it by a later launch, it can be missed
    // entirely. Either way this must not read as a pass.
    expect(!(raised && namesConv && namesTheThird),
           "MUTANT (one slot for the batch) does NOT name the refusing op");
  } else {
    expect(raised, "gate 1: a mid-batch refusal is reported at the flush");
    expect(namesConv, "gate 1: the error names the ORIGINATING op, mo.conv -- "
                      "not the kernel and not the batch");
    expect(namesTheThird, "gate 1: the error names which launch of the flush");
  }
  closeHarness(h);
}

/// GATE 2 -- late, not lost, and reported by the next raising caller.
void gateDeferredReport(Device &device, bool dropTheRecord) {
  Harness h;
  if (!openHarness(h, device, /*batch=*/true)) {
    expect(false, "gate 2: harness");
    return;
  }
  setenv("NPU_FAKE_TRANSPORT_WITNESS_ALL", "7=3", 1);

  uint32_t *slot = slotAt(h.arenaBase(), 0);
  memset(slot, 0, kSlotBytes);
  if (!dropTheRecord)
    armMatmulRan(slot, "mo.matmul");
  ErrorOrSuccess sent = launch(h, slot);
  expect(!sent.isError(), "gate 2: the enqueue itself does not fail");
  expect(slot[7] != kPathNoGrant,
         "gate 2: held, so the witness is still unwritten -- late, not lost");

  // The flush happens inside this second enqueue's pending-error check, so this
  // is the "next raising caller" the charter names.
  uint32_t *second = slotAt(h.arenaBase(), 1);
  memset(second, 0, kSlotBytes);
  ErrorOrSuccess next = launch(h, second);
  ErrorOrSuccess drained = h.ctx().synchronize();
  unsetenv("NPU_FAKE_TRANSPORT_WITNESS_ALL");

  bool reported = next.isError() || drained.isError();
  std::string message = next.isError()      ? std::string(next.getError())
                        : drained.isError() ? std::string(drained.getError())
                                            : "";
  printf("   gate 2 (%s): %s\n", dropTheRecord ? "record dropped" : "armed",
         reported ? message.c_str() : "no error");
  expect(slot[7] == kPathNoGrant,
         "gate 2: after the flush the slot carries what the kernel wrote");
  if (dropTheRecord) {
    // The mutant: arm nothing, and the same refusal passes silently. This is
    // what the guard buys, stated as a thing that can be taken away.
    expect(!reported, "MUTANT (no guard armed) lets the refusal through -- so "
                      "the armed run's error came from the guard");
  } else {
    expect(reported, "gate 2: the refusal is reported by a later raising call");
    expect(message.find("mo.matmul") != std::string::npos,
           "gate 2: and it names the op");
  }
  closeHarness(h);
}

/// GATE 3 -- an attribution guard fires and the render continues.
void gateAttribution(Device &device, bool promoteToCorrectness) {
  Harness h;
  if (!openHarness(h, device, /*batch=*/true)) {
    expect(false, "gate 3: harness");
    return;
  }
  // The array ran, so slot 7 is accepted and the conditional guard on 22 is
  // live -- and 22 carries a code no host knows.
  std::string words = "7=" + std::to_string(kPathHmxRan) + "," +
                      "22=" + std::to_string(kArmFromANewerSkel);
  setenv("NPU_FAKE_TRANSPORT_WITNESS_ALL", words.c_str(), 1);

  uint32_t *slot = slotAt(h.arenaBase(), 0);
  memset(slot, 0, kSlotBytes);
  armMatmulRan(slot, "mo.matmul");
  if (promoteToCorrectness)
    armGuard(slot, kClassCorrectness, 22, 7, kPathHmxRan, {0, 1, 2, 3, 4},
             "mo.matmul");
  else
    armUnpackArm(slot, "mo.matmul");

  CapturedStderr captured;
  captured.begin("gate3");
  ErrorOrSuccess sent = launch(h, slot);
  ErrorOrSuccess flushed = h.ctx().synchronize();
  ErrorOrSuccess after = h.ctx().synchronize();
  std::string loud = captured.end();
  unsetenv("NPU_FAKE_TRANSPORT_WITNESS_ALL");

  bool aborted = sent.isError() || flushed.isError() || after.isError();
  printf("   gate 3 (%s): %s\n",
         promoteToCorrectness ? "promoted to correctness" : "attribution",
         aborted ? "aborted" : "continued");
  if (promoteToCorrectness) {
    // The mutant: relabel the instrument as a correctness guard and it takes
    // the render down. That is the regression-gate-becomes-correctness-gate
    // trap, and this is the control that proves gate 3 measures the class and
    // not the absence of a firing guard.
    expect(aborted,
           "MUTANT (attribution promoted to correctness) DOES abort -- "
           "so the passing arm is the class, not a silent guard");
  } else {
    expect(!aborted, "gate 3: an attribution guard does not abort the flush");
    // AND it is not silent. Without this the gate passes on a guard that was
    // never evaluated, which is the same reading as one that behaved.
    expect(loud.find("npu witness:") != std::string::npos &&
               loud.find("word 22 = 99") != std::string::npos,
           "gate 3: it is recorded loudly, naming the word and the code");
  }
  closeHarness(h);
}

/// GATE 6 -- a pending error nobody ever checks is shouted at teardown rather
/// than dropped. The one real gap in the inherited mechanism: the flush
/// records, and if the process tears the context down before any raising
/// caller runs, the refusal would otherwise vanish.
void gateTeardownShout(Device &device, bool checkAtTeardown) {
  std::string loud;
  {
    Harness h;
    if (!openHarness(h, device, /*batch=*/true)) {
      expect(false, "gate 6: harness");
      return;
    }
    setenv("NPU_FAKE_TRANSPORT_WITNESS_ALL", "7=3", 1);
    uint32_t *slot = slotAt(h.arenaBase(), 0);
    memset(slot, 0, kSlotBytes);
    armMatmulRan(slot, "mo.matmul");
    ErrorOrSuccess sent = launch(h, slot);
    expect(!sent.isError(), "gate 6: the launch is held, not run");

    CapturedStderr captured;
    captured.begin("gate6");
    if (!checkAtTeardown) {
      // The mutant: drain the slot before the destructor sees it, which is
      // what the code did before teardown checked. The launch still runs and
      // the guard still fires -- and nothing says so.
      ErrorOrSuccess taken = h.ctx().checkPendingError();
      ErrorOrSuccess drained = h.ctx().synchronize();
      (void)drained;
      (void)h.ctx().checkPendingError();
      (void)taken;
    }
    closeHarness(h);
    // `h` dies here: the destructor flushes the held launch, the guard fires,
    // and nothing downstream will ever ask.
    h.arena = DeviceBufferRef();
    h.operands = DeviceBufferRef();
    h.matmul = DeviceFunctionRef();
    h.context = DeviceContextRef();
    loud = captured.end();
  }
  unsetenv("NPU_FAKE_TRANSPORT_WITNESS_ALL");
  bool shouted =
      loud.find("still pending at context teardown") != std::string::npos;
  printf("   gate 6 (%s): %s\n",
         checkAtTeardown ? "teardown checks" : "drained first",
         shouted ? "shouted" : "silent");
  if (checkAtTeardown)
    expect(shouted, "gate 6: a dropped pending error is loud at teardown");
  else
    expect(!shouted, "MUTANT (drained before teardown) is silent -- so the "
                     "shout comes from the teardown check, not from the flush");
}

/// GATE 4 -- the arena is process-lifetime: no allocation between launches, so
/// the batch is free to hold. #441's finding, re-run against the real layout.
void gateNoAllocInLoop(Device &device, void (*statsReset)(),
                       void (*stats)(uint64_t *, int)) {
  Harness h;
  if (!openHarness(h, device, /*batch=*/true)) {
    expect(false, "gate 4: harness");
    return;
  }
  // One launch and a flush first, then reset: a context's FIRST launch opens
  // its skel session, which costs one allocation once. Measuring from there
  // makes the reading the PER-LAUNCH cost, which is the claim.
  uint32_t *warm = slotAt(h.arenaBase(), 0);
  memset(warm, 0, kSlotBytes);
  ErrorOrSuccess warmed = launch(h, warm);
  ErrorOrSuccess opened = h.ctx().synchronize();
  if (warmed.isError() || opened.isError()) {
    expect(false, "gate 4: warm-up launch");
    closeHarness(h);
    return;
  }

  statsReset();
  for (uint32_t i = 0; i < 8; ++i) {
    uint32_t *slot = slotAt(h.arenaBase(), i);
    memset(slot, 0, kSlotBytes);
    ErrorOrSuccess sent = launch(h, slot);
    if (sent.isError()) {
      expect(false, "gate 4: launch");
      closeHarness(h);
      return;
    }
  }
  uint64_t raw[6] = {};
  stats(raw, 6);
  uint64_t loopAllocs = raw[0], loopFrees = raw[1];
  ErrorOrSuccess flushed = h.ctx().synchronize();
  (void)flushed;
  stats(raw, 6);
  raw[0] = loopAllocs;
  raw[1] = loopFrees;
  double width = raw[3] == 0 ? 0.0 : double(raw[4]) / double(raw[3]);
  printf("   gate 4: %llu allocs, %llu frees inside the launch loop; %llu "
         "batches over %llu launches (mean width %.2f)\n",
         (unsigned long long)raw[0], (unsigned long long)raw[1],
         (unsigned long long)raw[3], (unsigned long long)raw[4], width);
  expect(raw[0] == 0 && raw[1] == 0,
         "gate 4: taking a slot reaches neither memAlloc nor memFree");
  expect(width >= 8.0, "gate 4: and the eight launches travel as one batch");
  closeHarness(h);
}

/// GATE 5 -- an arena too small for the batch cap is refused, not silently
/// aliased. The cross-language constant check, which is the only thing keeping
/// `NPU_ARENA_SLOTS` and `kNPUBatchMaxLaunches` in step.
void gateSkewedArena(Device &device) {
  Harness h;
  if (!openHarness(h, device, /*batch=*/false)) {
    expect(false, "gate 5: harness");
    return;
  }
  initArena(h.arenaBase(), /*slots=*/8);
  uint32_t *slot = slotAt(h.arenaBase(), 0);
  memset(slot, 0, kSlotBytes);
  armMatmulRan(slot, "mo.matmul");
  ErrorOrSuccess sent = launch(h, slot);
  ErrorOrSuccess drained = h.ctx().synchronize();
  bool refused = sent.isError() || drained.isError();
  std::string message = sent.isError()      ? std::string(sent.getError())
                        : drained.isError() ? std::string(drained.getError())
                                            : "";
  printf("   gate 5: %s\n", refused ? message.c_str() : "accepted");
  expect(refused && message.find("share a slot") != std::string::npos,
         "gate 5: an arena shorter than the batch cap is refused by name");
  closeHarness(h);
}

/// What a host-executing registration does between two launches: reads a
/// device buffer and writes another, through raw pointers, asking this class
/// nothing. `Concat`, `Split`, `Cast`, `Gather`, the elementwise host walks and
/// `npu_view_copy`'s materializer are all exactly this shape.
enum class HostReadArm { Synchronous, Batched, BatchedAfterFlush };

/// GATE 7 -- THE 186A MECHANISM, ON THE HOST AND WITH NO DEVICE.
///
/// Run 186A batched the graph path for the first time and produced five
/// deterministically wrong stage hashes at rc=0 with every witness guard
/// passing. This is that, reduced to its three ops.
///
/// A registration that runs ON THE HOST touches device memory only through
/// `unsafe_ptr()`, so nothing flushes and nothing orders it against a launch
/// still held in `batch_`. It reads the producer's PRE-LAUNCH bytes, computes
/// a plausible-wrong result and hands it downstream. Nothing fails: rc stays
/// 0, and the witness guards stay silent because the kernels that ran DID run
/// correctly -- on wrong inputs, which a family-2 guard cannot see by
/// construction. `QuRTDeviceContext.h`'s `batching()` comment predicted this
/// caller class exactly ("a kernel that runs on the host while claiming a
/// device is exactly such a caller") and scoped batching to launch sequences
/// known to contain none; the graph path is not one, and the census in
/// `.agentwork/flux-npu/WITNESS_GUARD_CLASSIFICATION.md` names the sites.
///
/// Produce into X, copy X to Y on the host with no context call in between,
/// consume Y. `seen` is what the host copy read -- the only thing that
/// matters, since that is what a real host op would have computed from.
void gateHostReadOrdering(Device &device, HostReadArm arm) {
  constexpr uint32_t kSentinel = 0x5A5A5A5Au;
  Harness h;
  if (!openHarness(h, device, /*batch=*/arm != HostReadArm::Synchronous)) {
    expect(false, "gate 7: harness");
    return;
  }
  auto *bytes = static_cast<uint8_t *>(h.operands->ptr());
  auto *x = bytes;
  auto *y = bytes + kOperandBytes;
  auto *spare = bytes + 3 * kOperandBytes;

  // X says "unproduced" until a launch stamps it.
  memcpy(x, &kSentinel, sizeof(kSentinel));

  uint32_t *first = slotAt(h.arenaBase(), 0);
  memset(first, 0, kSlotBytes);
  ErrorOrSuccess produced = launchInto(h, x, spare, first);
  expect(!produced.isError(), "gate 7: the producer is enqueued");

  // THE CONTROL, and the only difference between this arm and the one above
  // it. A discriminator that cannot be flipped is measuring something else.
  if (arm == HostReadArm::BatchedAfterFlush) {
    ErrorOrSuccess flushed = h.ctx().synchronize();
    expect(!flushed.isError(), "gate 7: the control's flush succeeds");
  }

  // The host op. No context call -- that is the entire point.
  memcpy(y, x, 64);
  uint32_t seen = 0;
  memcpy(&seen, y, sizeof(seen));

  uint32_t *second = slotAt(h.arenaBase(), 1);
  memset(second, 0, kSlotBytes);
  ErrorOrSuccess consumed = launchInto(h, spare, y, second);
  ErrorOrSuccess drained = h.ctx().synchronize();
  expect(!consumed.isError() && !drained.isError(),
         "gate 7: the consumer runs and the batch drains");

  const char *name = arm == HostReadArm::Synchronous ? "synchronous"
                     : arm == HostReadArm::Batched   ? "batched"
                                                     : "batched, flushed first";
  bool stale = seen == kSentinel;
  printf("   gate 7 (%-22s): the host copy read 0x%08x -- %s\n", name, seen,
         stale ? "THE PRODUCER HAD NOT RUN" : "the producer's output");

  if (arm == HostReadArm::Batched)
    expect(stale, "gate 7: UNDER BATCHING a host op reads the producer's "
                  "pre-launch bytes -- rc 0, guards silent, answer wrong");
  else
    expect(!stale,
           arm == HostReadArm::Synchronous
               ? "gate 7: on the synchronous transport the host op reads the "
                 "producer's output, which is what every host-executing "
                 "registration assumes today"
               : "gate 7 CONTROL: a flush before the host op restores the "
                 "ordering under batching -- so the batched arm measures "
                 "ordering and not something else");
  closeHarness(h);
}

/// GATE 8 -- and it compounds. Eight producer/host-copy pairs in one flush,
/// which is the width run 186A measured (8.500 over 160 flushes). Every copy
/// after the first reads a buffer whose producer is also still held, so the
/// error does not stay local to one op: a whole stage computes from bytes that
/// predate it.
void gateHostReadDepth(Device &device, bool batch) {
  constexpr uint32_t kSentinel = 0x5A5A5A5Au;
  constexpr uint32_t kDepth = 8;
  constexpr size_t kStride = 512;
  Harness h;
  if (!openHarness(h, device, batch)) {
    expect(false, "gate 8: harness");
    return;
  }
  auto *bytes = static_cast<uint8_t *>(h.operands->ptr());
  auto *spare = bytes + 3 * kOperandBytes;

  uint32_t stale = 0;
  for (uint32_t i = 0; i < kDepth; ++i) {
    auto *x = bytes + i * kStride;
    auto *y = bytes + i * kStride + 256;
    memcpy(x, &kSentinel, sizeof(kSentinel));

    uint32_t *slot = slotAt(h.arenaBase(), i);
    memset(slot, 0, kSlotBytes);
    ErrorOrSuccess produced = launchInto(h, x, spare, slot);
    if (produced.isError()) {
      expect(false, "gate 8: producer " + std::to_string(i));
      closeHarness(h);
      return;
    }
    memcpy(y, x, 64);
    uint32_t seen = 0;
    memcpy(&seen, y, sizeof(seen));
    if (seen == kSentinel)
      ++stale;
  }
  ErrorOrSuccess drained = h.ctx().synchronize();
  expect(!drained.isError(), "gate 8: the batch drains");

  printf("   gate 8 (%s): %u of %u host copies read a producer that had not "
         "run\n",
         batch ? "batched" : "synchronous", stale, kDepth);
  if (batch)
    expect(stale == kDepth,
           "gate 8: every host copy in a held batch reads stale bytes -- the "
           "error is a whole stage's, not one op's");
  else
    expect(stale == 0,
           "gate 8: with no batching every host copy reads its producer");
  closeHarness(h);
}

} // namespace

int main() {
  const char *fakeLib = getenv("NPU_FAKE_TRANSPORT");
  if (fakeLib == nullptr || *fakeLib == '\0') {
    printf("FAIL: NPU_FAKE_TRANSPORT is not set\n");
    return 1;
  }
  setenv("MODULAR_NPU_TRANSPORT_LIB", fakeLib, 1);
  setenv("MODULAR_NPU_POWER", "0", 1);

  ErrorOr<DeviceRef> device = QuRTDevice::create(0);
  if (device.isError()) {
    printf("FAIL: device: %s\n", device.getError());
    return 1;
  }

  void *handle = dlopen(fakeLib, RTLD_NOW);
  if (handle == nullptr) {
    printf("FAIL: dlopen %s: %s\n", fakeLib, dlerror());
    return 1;
  }
  auto stats = reinterpret_cast<void (*)(uint64_t *, int)>(
      dlsym(handle, "npu_fake_transport_stats"));
  auto statsReset = reinterpret_cast<void (*)()>(
      dlsym(handle, "npu_fake_transport_stats_reset"));
  if (stats == nullptr || statsReset == nullptr) {
    printf("FAIL: the stand-in exports no counters\n");
    return 1;
  }

  printf("gate 1 -- one slot per in-flight launch\n");
  gateOwnSlot(**device, /*sharedSlot=*/false);
  gateOwnSlot(**device, /*sharedSlot=*/true);

  printf("gate 2 -- the flush records, the next raising caller reports\n");
  gateDeferredReport(**device, /*dropTheRecord=*/false);
  gateDeferredReport(**device, /*dropTheRecord=*/true);

  printf("gate 3 -- an attribution guard never aborts a render\n");
  gateAttribution(**device, /*promoteToCorrectness=*/false);
  gateAttribution(**device, /*promoteToCorrectness=*/true);

  printf("gate 4 -- the arena costs no allocation per launch\n");
  gateNoAllocInLoop(**device, statsReset, stats);

  printf("gate 5 -- a skewed arena is refused\n");
  gateSkewedArena(**device);

  printf("gate 6 -- a pending error nobody checks is loud at teardown\n");
  gateTeardownShout(**device, /*checkAtTeardown=*/true);
  gateTeardownShout(**device, /*checkAtTeardown=*/false);

  printf("gate 7 -- a host op between two launches reads the held one's "
         "operand (run 186A)\n");
  gateHostReadOrdering(**device, HostReadArm::Synchronous);
  gateHostReadOrdering(**device, HostReadArm::Batched);
  gateHostReadOrdering(**device, HostReadArm::BatchedAfterFlush);

  printf("gate 8 -- and it compounds across a flush\n");
  gateHostReadDepth(**device, /*batch=*/false);
  gateHostReadDepth(**device, /*batch=*/true);

  if (failures != 0) {
    printf("%d check(s) failed\n", failures);
    return 1;
  }
  printf("all checks passed\n");
  return 0;
}
