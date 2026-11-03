//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Host-side proof of the per-dispatch IO recorder, end to end through the real
// launch path: `QuRTDevice::create` binds the stand-in transport through the
// MODULAR_NPU_TRANSPORT_LIB seam, and the real `enqueueFunctionExecDirect`
// packs and dispatches a chain of real manifest kernels with interior-offset
// operands out of one arena -- the model pool's shape. The recorder must
// capture every dispatch in order, with the selector each launch named, the
// allocation/offset tuple each operand crossed as, and content hashes that are
// stable within a process and across processes.
//
// The control that could fail: the same chain runs with recording off, and its
// buffer bytes must be identical to the recorded run's. A recorder that
// mutates what it observes fails here, which is the property that makes its
// records usable as evidence.
//
// The orchestrating process spawns itself three times (recorded twice, off
// once) because the recorder latches its environment once per process, which
// is also how it is used on a device run.
//
//===----------------------------------------------------------------------===//

#include "NPUDispatchRecorder.h"
#include "QuRTDeviceContext.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <optional>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

using namespace M;
using namespace M::Driver;

namespace {

int failures = 0;

void expect(bool condition, const std::string &what) {
  if (!condition) {
    printf("FAIL: %s\n", what.c_str());
    ++failures;
  }
}

constexpr size_t kArenaBytes = 1 << 20;
constexpr size_t kSmallBytes = 64 << 10;
constexpr uint64_t kWindow = 512;

// Interior operand offsets, echoing npu_launch_offsets_test's arena case.
constexpr uint64_t kVaddOffsets[3] = {0, 98304, 196608};
constexpr uint64_t kVscaleArenaOffset = 4096;
constexpr uint64_t kVscaleSmallOffset = 512;
constexpr uint64_t kSoftmaxOffsets[3] = {0, 16384, 262144};

/// The recorder's window fold, restated so a silent change to the recorded
/// hash function fails a test instead of shifting every future dump.
uint64_t windowFold(const uint8_t *bytes, size_t count) {
  uint64_t hash = 14695981039346656037ull;
  size_t whole = count & ~size_t{7};
  for (size_t i = 0; i < whole; i += 8) {
    uint64_t word;
    memcpy(&word, bytes + i, sizeof(word));
    hash = (hash ^ word) * 1099511628211ull;
  }
  for (size_t i = whole; i < count; ++i)
    hash = (hash ^ bytes[i]) * 1099511628211ull;
  return hash;
}

//===----------------------------------------------------------------------===//
// Child: run the dispatch chain
//===----------------------------------------------------------------------===//

int childFail(const char *stage, const char *error) {
  printf("child %s: %s\n", stage, error);
  return 1;
}

int runChild(const char *resultPath) {
  ErrorOr<DeviceRef> device = QuRTDevice::create(0);
  if (device.isError())
    return childFail("device", device.getError());
  ErrorOr<DeviceContextRef> context = (*device)->createContext();
  if (context.isError())
    return childFail("context", context.getError());
  DeviceContext &ctx = **context;

  ErrorOr<DeviceBufferRef> arena = ctx.enqueueCreateBuffer(kArenaBytes);
  if (arena.isError())
    return childFail("arena", arena.getError());
  ErrorOr<DeviceBufferRef> small = ctx.enqueueCreateBuffer(kSmallBytes);
  if (small.isError())
    return childFail("small", small.getError());
  auto *arenaBytes = static_cast<uint8_t *>((*arena)->ptr());
  auto *smallBytes = static_cast<uint8_t *>((*small)->ptr());

  ErrorOr<DeviceFunctionRef> vadd = ctx.loadFunction("", "vadd", "");
  ErrorOr<DeviceFunctionRef> vscale = ctx.loadFunction("", "vscale_offset", "");
  ErrorOr<DeviceFunctionRef> softmax = ctx.loadFunction("", "softmax", "");
  if (vadd.isError() || vscale.isError() || softmax.isError())
    return childFail("loadFunction", "a manifest kernel failed to resolve");

  std::vector<uint8_t> arenaPattern(kArenaBytes);
  for (size_t i = 0; i < kArenaBytes; ++i)
    arenaPattern[i] = static_cast<uint8_t>(i * 131 + 17);
  std::vector<uint8_t> smallZeros(kSmallBytes, 0);

  auto launch = [&](DeviceFunctionRef &function, void *buffers[],
                    uint32_t bufferCount, uint32_t scalars[],
                    uint32_t scalarCount) -> const char * {
    void *args[8];
    size_t argSizes[8];
    for (uint32_t i = 0; i < bufferCount; ++i) {
      args[i] = &buffers[i];
      argSizes[i] = sizeof(void *);
    }
    for (uint32_t i = 0; i < scalarCount; ++i) {
      args[bufferCount + i] = &scalars[i];
      argSizes[bufferCount + i] = sizeof(uint32_t);
    }
    ErrorOrSuccess result = ctx.enqueueFunctionExecDirect(
        *function, /*gridX=*/1, /*gridY=*/1, /*gridZ=*/1, /*blockX=*/1,
        /*blockY=*/1, /*blockZ=*/1, /*sharedMemBytes=*/0, /*attrs=*/nullptr,
        /*numAttrs=*/0, args, bufferCount + scalarCount, argSizes);
    return result.isError() ? result.getError() : nullptr;
  };

  // Two identical rounds. Each restages both buffers, so every launch record
  // of round two must hash identically to its round-one twin.
  for (int round = 0; round < 2; ++round) {
    if (ErrorOrSuccess err = ctx.enqueueCopyToDevice(
            arenaBytes, arenaPattern.data(), kArenaBytes))
      return childFail("stage arena", err.getError());
    if (ErrorOrSuccess err =
            ctx.enqueueCopyToDevice(smallBytes, smallZeros.data(), kSmallBytes))
      return childFail("stage small", err.getError());

    void *vaddBufs[3] = {arenaBytes + kVaddOffsets[0],
                         arenaBytes + kVaddOffsets[1],
                         arenaBytes + kVaddOffsets[2]};
    uint32_t vaddScalars[1] = {1024};
    if (const char *err = launch(*vadd, vaddBufs, 3, vaddScalars, 1))
      return childFail("vadd", err);

    void *vscaleBufs[2] = {arenaBytes + kVscaleArenaOffset,
                           smallBytes + kVscaleSmallOffset};
    uint32_t vscaleScalars[3] = {512, 16, 3};
    if (const char *err = launch(*vscale, vscaleBufs, 2, vscaleScalars, 3))
      return childFail("vscale_offset", err);

    void *softmaxBufs[3] = {smallBytes + kSoftmaxOffsets[0],
                            smallBytes + kSoftmaxOffsets[1],
                            arenaBytes + kSoftmaxOffsets[2]};
    uint32_t softmaxScalars[2] = {128, 64};
    if (const char *err = launch(*softmax, softmaxBufs, 3, softmaxScalars, 2))
      return childFail("softmax", err);
  }

  FILE *result = fopen(resultPath, "wb");
  if (result == nullptr)
    return childFail("result", "cannot open the result file");
  fwrite(arenaBytes, 1, kArenaBytes, result);
  fwrite(smallBytes, 1, kSmallBytes, result);
  fclose(result);
  return 0;
}

//===----------------------------------------------------------------------===//
// Orchestrator: spawn, then judge the dumps
//===----------------------------------------------------------------------===//

struct Spawn {
  std::optional<std::string> dumpDir;
  bool dumpBytes = false;
  std::string resultPath;
};

int spawnChild(const std::string &self, const std::string &fakeLib,
               const Spawn &mode) {
  pid_t pid = fork();
  if (pid < 0)
    return -1;
  if (pid == 0) {
    setenv("MODULAR_NPU_TRANSPORT_LIB", fakeLib.c_str(), 1);
    if (mode.dumpDir) {
      setenv("MODULAR_NPU_DISPATCH_DUMP", mode.dumpDir->c_str(), 1);
      setenv("MODULAR_NPU_DISPATCH_DUMP_WINDOW",
             std::to_string(kWindow).c_str(), 1);
      if (mode.dumpBytes) {
        setenv("MODULAR_NPU_DISPATCH_DUMP_BYTES", "1", 1);
        setenv("MODULAR_NPU_DISPATCH_DUMP_BYTES_CAP", "4096", 1);
      }
    }
    execl(self.c_str(), self.c_str(), "--child", mode.resultPath.c_str(),
          static_cast<char *>(nullptr));
    _exit(127);
  }
  int status = 0;
  if (waitpid(pid, &status, 0) != pid)
    return -1;
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

std::vector<std::string> readLines(const std::string &path) {
  std::vector<std::string> lines;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line))
    if (!line.empty())
      lines.push_back(line);
  return lines;
}

std::string jsonField(const std::string &line, const std::string &key) {
  std::string marker = "\"" + key + "\":";
  size_t at = line.find(marker);
  if (at == std::string::npos)
    return "";
  at += marker.size();
  size_t end = at;
  if (line[at] == '"') {
    end = line.find('"', at + 1);
    return line.substr(at + 1, end - at - 1);
  }
  while (end < line.size() && line[end] != ',' && line[end] != '}' &&
         line[end] != ']')
    ++end;
  return line.substr(at, end - at);
}

/// Occurrences of `key` (for per-slot fields inside the bufs array), in order.
std::vector<std::string> jsonFields(const std::string &line,
                                    const std::string &key) {
  std::vector<std::string> values;
  std::string marker = "\"" + key + "\":";
  size_t from = 0;
  while (true) {
    size_t at = line.find(marker, from);
    if (at == std::string::npos)
      return values;
    values.push_back(jsonField(line.substr(at), key));
    from = at + marker.size();
  }
}

/// Drops what legitimately differs between two processes: raw addresses and
/// the meta/bytecap bookkeeping.
std::string normalized(const std::vector<std::string> &lines) {
  std::ostringstream out;
  for (const std::string &line : lines) {
    std::string kind = jsonField(line, "k");
    if (kind == "meta" || kind == "bytecap")
      continue;
    std::string scrubbed = line;
    for (const char *key : {"\"base\":\"", "\"dst\":\"", "\"src\":\""}) {
      size_t from = 0;
      while (true) {
        size_t at = scrubbed.find(key, from);
        if (at == std::string::npos)
          break;
        size_t open = at + strlen(key);
        size_t close = scrubbed.find('"', open);
        scrubbed.replace(open, close - open, "X");
        from = open;
      }
    }
    out << scrubbed << "\n";
  }
  return out.str();
}

std::vector<uint8_t> readBytes(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)),
                              std::istreambuf_iterator<char>());
}

std::string singleDumpFile(const std::string &directory) {
  std::error_code errorCode;
  for (const auto &entry :
       std::filesystem::directory_iterator(directory, errorCode)) {
    std::string name = entry.path().filename().string();
    if (name.rfind("npu_iodump_", 0) == 0)
      return entry.path().string();
  }
  return "";
}

int runOrchestrator(const char *self) {
  const char *fake = getenv("NPU_FAKE_TRANSPORT");
  const char *tmp = getenv("TEST_TMPDIR");
  if (fake == nullptr || tmp == nullptr) {
    printf("FAIL: NPU_FAKE_TRANSPORT and TEST_TMPDIR must be set\n");
    return 1;
  }
  char resolved[4096];
  std::string fakeLib =
      realpath(fake, resolved) != nullptr ? resolved : std::string(fake);
  std::string base(tmp);

  Spawn runA{base + "/dumpA", true, base + "/resultA.bin"};
  Spawn runB{base + "/dumpB", false, base + "/resultB.bin"};
  Spawn runOff{std::nullopt, false, base + "/resultOFF.bin"};
  expect(spawnChild(self, fakeLib, runA) == 0, "recorded run A exited 0");
  expect(spawnChild(self, fakeLib, runB) == 0, "recorded run B exited 0");
  expect(spawnChild(self, fakeLib, runOff) == 0, "unrecorded run exited 0");
  if (failures)
    return 1;

  // The control that could fail: recording must not change a byte of what the
  // dispatches computed.
  std::vector<uint8_t> resultA = readBytes(runA.resultPath);
  std::vector<uint8_t> resultOff = readBytes(runOff.resultPath);
  expect(resultA.size() == kArenaBytes + kSmallBytes,
         "recorded run wrote a full result");
  expect(resultA == resultOff,
         "recording changed the dispatched bytes: the recorder wrote to an "
         "operand");

  // Off means off: no dump file appears anywhere a latch could have put one.
  expect(singleDumpFile(base).empty() &&
             singleDumpFile(base + "/dumpOFF").empty(),
         "an unrecorded run left a dump file");

  std::string dumpPathA = singleDumpFile(*runA.dumpDir);
  expect(!dumpPathA.empty(), "recorded run A produced a dump file");
  if (dumpPathA.empty())
    return 1;
  std::vector<std::string> linesA = readLines(dumpPathA);

  // The chain, in order, twice: two staging copies then three launches per
  // round, every launch accepted by the stand-in.
  std::vector<std::string> launches;
  std::vector<std::string> copies;
  for (const std::string &line : linesA) {
    std::string kind = jsonField(line, "k");
    if (kind == "launch")
      launches.push_back(line);
    else if (kind == "copy_to")
      copies.push_back(line);
  }
  expect(copies.size() == 4, "four staging copies were recorded, got " +
                                 std::to_string(copies.size()));
  expect(launches.size() == 6,
         "six launches were recorded, got " + std::to_string(launches.size()));
  if (launches.size() != 6)
    return 1;

  const char *expectedNames[3] = {"vadd", "vscale_offset", "softmax"};
  const char *expectedSelectors[3] = {"0", "1", "7"};
  for (size_t i = 0; i < 6; ++i) {
    const std::string &line = launches[i];
    expect(jsonField(line, "name") == expectedNames[i % 3],
           "launch " + std::to_string(i) + " name");
    expect(jsonField(line, "sel") == expectedSelectors[i % 3],
           "launch " + std::to_string(i) + " selector");
    expect(jsonField(line, "mask") == "1",
           "launch " + std::to_string(i) + " thread mask");
    expect(jsonField(line, "status") == "0",
           "launch " + std::to_string(i) + " status");
  }

  // Sequence numbers are the file's own order: strictly increasing across
  // every counted record kind, so "which dispatch came first" is never
  // ambiguous. A bytecap annotation names the launch it interrupted rather
  // than counting, so it is exempt. P5 M4.2 order 088 adds a second record per
  // launch -- the `issue`, written before the call crosses -- so a launch seq
  // is counted once at issue and repeated once at completion.
  long long previousSeq = -1;
  bool ordered = true;
  for (const std::string &line : linesA) {
    std::string kind = jsonField(line, "k");
    if (kind == "meta" || kind == "bytecap")
      continue;
    long long value = atoll(jsonField(line, "seq").c_str());
    if (kind == "launch")
      ordered = ordered && value == previousSeq;
    else
      ordered = ordered && value == previousSeq + 1;
    previousSeq = value;
  }
  expect(ordered, "sequence numbers count every record exactly once");

  // THE 088 PROPERTY. Order 088's stall could not say whether launch 290 was
  // never issued or was issued and never returned, because a record existed
  // only once a call came back. Every issued launch now announces itself first,
  // so the two are different shapes on the wire: an `issue` with no `launch` is
  // a device holding a dispatch, and no `issue` at all is a host that stopped.
  std::vector<std::string> issues;
  for (const std::string &line : linesA)
    if (jsonField(line, "k") == "issue")
      issues.push_back(line);
  expect(issues.size() == launches.size(),
         "every completed launch announced itself first, got " +
             std::to_string(issues.size()) + " issues for " +
             std::to_string(launches.size()) + " launches");
  bool issuedFirst = true;
  for (size_t i = 0; i < issues.size() && i < launches.size(); ++i)
    issuedFirst = issuedFirst &&
                  jsonField(issues[i], "seq") == jsonField(launches[i], "seq");
  expect(issuedFirst, "issue and launch records pair by sequence number");

  // THE 091 PROPERTY, and this bug is deliberately the example. The witness is
  // the last buffer slot, and order 091 found the graph path allocating it at
  // 32 BYTES while the kernels write 32 WORDS -- so a size-keyed readback
  // matched nothing on 1612 of 1612 launches, and would have read 18 words out
  // of an 8-word buffer had it matched. Two halves are pinned here:
  //
  //   1. `wit` is never longer than the slot's own bytes allow. That is the
  //      mutation: shrink the witness allocation and this bound is what names
  //      it, instead of the recorder walking off the end.
  //   2. `witBytes` is reported, so an undersized witness is visible in the
  //      dump as a number rather than as a missing field.
  bool bounded = true;
  bool sized = true;
  for (const std::string &line : launches) {
    std::string bytes = jsonField(line, "witBytes");
    if (bytes.empty())
      continue;
    long long witnessBytes = atoll(bytes.c_str());
    size_t open = line.find("\"wit\":[");
    size_t close = line.find(']', open);
    if (open == std::string::npos || close == std::string::npos)
      continue;
    std::string body = line.substr(open + 7, close - open - 7);
    size_t words = body.empty() ? 0 : 1;
    for (char c : body)
      if (c == ',')
        ++words;
    bounded = bounded && static_cast<long long>(words) * 4 <= witnessBytes;
    // 32 words is the declared contract (`NPU_WITNESS_WORDS = 3 + 5 + 4*6`),
    // so a conforming witness slot carries at least 128 bytes. Stated in BYTES
    // on purpose: the units are what order 091 was about.
    sized = sized && witnessBytes >= 128;
  }
  expect(bounded, "wit never reports more words than the witness slot holds");

  expect(sized, "the witness slot carries the declared 128 bytes (32 words)");

  // ORDER 091, AS THE EXAMPLE. The graph path allocated the witness at 32
  // BYTES while the kernels write 32 WORDS. The rule that decides how much of
  // a witness may be read is asserted directly, in bytes, because the fake
  // harness's slots are all comfortably large and an end-to-end check would
  // pass vacuously:
  //
  //   32 bytes  -> 8 words   (091's bug: reported, not overread)
  //   128 bytes -> 32 words  (the declared contract, whole)
  //   192 bytes -> 32 words  (the report width still caps an over-allocation)
  //
  expect(M::Driver::npuWitnessWordsToReport(32, 0) == 8,
         "a 32-BYTE witness reports 8 words, not 32 -- order 091's bug");
  expect(M::Driver::npuWitnessWordsToReport(128, 0) == 32,
         "a 128-byte witness reports all 32 declared words");
  expect(M::Driver::npuWitnessWordsToReport(88, 0) == 22,
         "a short slot reports what it holds -- the clamp is the allocation's, "
         "not the report width's");
  expect(M::Driver::npuWitnessWordsToReport(512, 0) == 32,
         "the graph path's over-allocated slot still reports 32, not 128");
  expect(M::Driver::npuWitnessWordsToReport(0, 0) == 0,
         "an empty witness reports nothing");
  expect(M::Driver::npuWitnessWordsToReport(128, 200) == 0,
         "an offset past the allocation reports nothing rather than reading");

  // The operand tuples: offsets as packed for the wire, allocations by size.
  auto expectSlots = [&](const std::string &line, const char *what,
                         std::initializer_list<uint64_t> offsets,
                         std::initializer_list<uint64_t> allocSizes) {
    std::vector<std::string> off = jsonFields(line, "off");
    std::vector<std::string> asz = jsonFields(line, "asz");
    expect(off.size() == offsets.size(),
           std::string(what) + " records every operand");
    if (off.size() != offsets.size())
      return;
    size_t i = 0;
    for (uint64_t expected : offsets) {
      expect(off[i] == std::to_string(expected),
             std::string(what) + " slot " + std::to_string(i) + " offset");
      ++i;
    }
    i = 0;
    for (uint64_t expected : allocSizes) {
      expect(asz[i] == std::to_string(expected), std::string(what) + " slot " +
                                                     std::to_string(i) +
                                                     " allocation size");
      ++i;
    }
  };
  expectSlots(launches[0], "vadd",
              {kVaddOffsets[0], kVaddOffsets[1], kVaddOffsets[2]},
              {kArenaBytes, kArenaBytes, kArenaBytes});
  expectSlots(launches[1], "vscale_offset",
              {kVscaleArenaOffset, kVscaleSmallOffset},
              {kArenaBytes, kSmallBytes});
  expectSlots(launches[2], "softmax",
              {kSoftmaxOffsets[0], kSoftmaxOffsets[1], kSoftmaxOffsets[2]},
              {kSmallBytes, kSmallBytes, kArenaBytes});

  // Scalar values cross as packed.
  expect(launches[0].find("\"scalars\":[\"0x400\"]") != std::string::npos,
         "vadd scalar block");
  expect(launches[1].find("\"scalars\":[\"0x200\",\"0x10\",\"0x3\"]") !=
             std::string::npos,
         "vscale_offset scalar block");

  // The all-zero flag: the small buffer is staged as zeros, so operands read
  // from it arrive all-zero and leave patterned once the stand-in writes.
  std::vector<std::string> vscalePreZ = jsonFields(launches[1], "preZ");
  std::vector<std::string> vscalePostZ = jsonFields(launches[1], "postZ");
  std::vector<std::string> vaddPreZ = jsonFields(launches[0], "preZ");
  expect(vscalePreZ.size() == 2 && vscalePreZ[1] == "1",
         "a zero-staged operand window flags preZ");
  expect(vscalePostZ.size() == 2 && vscalePostZ[1] == "0",
         "a written operand window clears postZ");
  expect(!vaddPreZ.empty() && vaddPreZ[0] == "0",
         "a patterned operand window does not flag preZ");

  // Hash stability within a process: round two restaged identical bytes, so
  // each launch's hashes must equal its round-one twin's.
  for (size_t i = 0; i < 3; ++i) {
    expect(jsonFields(launches[i], "pre") == jsonFields(launches[i + 3], "pre"),
           "round-stable pre hashes for " + std::string(expectedNames[i]));
    expect(jsonFields(launches[i], "post") ==
               jsonFields(launches[i + 3], "post"),
           "round-stable post hashes for " + std::string(expectedNames[i]));
  }

  // Hash correctness against an independent fold of the bytes the run staged.
  std::vector<uint8_t> arenaPattern(kWindow);
  for (size_t i = 0; i < kWindow; ++i)
    arenaPattern[i] = static_cast<uint8_t>(i * 131 + 17);
  char expectedHash[32];
  snprintf(expectedHash, sizeof(expectedHash), "0x%llx",
           static_cast<unsigned long long>(
               windowFold(arenaPattern.data(), kWindow)));
  expect(jsonFields(launches[0], "pre")[0] == expectedHash,
         "vadd slot 0 pre-hash matches an independent fold of the staged "
         "pattern");
  expect(jsonField(copies[0], "dstPost") == expectedHash,
         "the arena staging copy's post-hash matches the same fold");

  // Cross-process stability: run B differs only in raw addresses and byte
  // dumping, so the normalized records must be identical.
  std::string dumpPathB = singleDumpFile(*runB.dumpDir);
  expect(!dumpPathB.empty(), "recorded run B produced a dump file");
  if (!dumpPathB.empty())
    expect(normalized(linesA) == normalized(readLines(dumpPathB)),
           "two processes' normalized dumps are identical");

  // Byte dumping respects its cap: the window files that fit, then one
  // bytecap record, and not a byte more. The first launch is seq 2, after the
  // two staging copies.
  std::string bytesDir = *runA.dumpDir + "/bytes";
  std::vector<uint8_t> firstWindow = readBytes(bytesDir + "/d2_s0_pre.bin");
  expect(firstWindow.size() == kWindow,
         "the first launch's first window was dumped whole");
  expect(windowFold(firstWindow.data(), firstWindow.size()) ==
             windowFold(arenaPattern.data(), kWindow),
         "the dumped window holds the staged bytes");
  bool sawByteCap = false;
  for (const std::string &line : linesA)
    sawByteCap = sawByteCap || jsonField(line, "k") == "bytecap";
  expect(sawByteCap, "the byte cap was reported once reached");

  if (failures) {
    printf("\n%d check(s) failed\n", failures);
    return 1;
  }
  printf("PASS -- the recorder captures every dispatch and changes none\n");
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  if (argc == 3 && strcmp(argv[1], "--child") == 0)
    return runChild(argv[2]);
  return runOrchestrator(argv[0]);
}
