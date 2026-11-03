//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// The provenance predicate of the weights-NOFLUSH classification, held to its
// contract through the same C seam the backend queries: memory is constant
// because the loader declared it, never because of its size, its name, or
// anything guessed. Pure interval arithmetic -- no FastRPC, no device.
//
// The properties that could fail silently on the device are the ones pinned
// here: nothing declared covers nothing (the safe default); a query one byte
// past a declared end, straddling a gap between two declarations, or starting
// before a declared base must all refuse -- each of those, misclassified,
// marks a live buffer as never-flushed and corrupts without a diagnostic.
//
//===----------------------------------------------------------------------===//

#include "npu_constant_ranges.h"

#include <cstdint>
#include <cstdio>

namespace {

int failures = 0;

void expect(bool condition, const char *what) {
  if (!condition) {
    std::printf("FAIL: %s\n", what);
    ++failures;
  }
}

bool covers(const uint8_t *base, size_t byteSize) {
  return MODULAR_NPUConstantHostRangeCovers(base, byteSize) != 0;
}

// Backing storage so every queried address is real memory; the registry only
// ever compares addresses.
uint8_t arena[4096];

} // namespace

int main() {
  MODULAR_NPUForgetConstantHostRanges();

  // The safe default: nothing declared, nothing covered.
  expect(!covers(arena, 1), "undeclared memory reads as covered");
  expect(!covers(arena, sizeof(arena)), "undeclared span reads as covered");

  // One declared blob: [256, 512).
  MODULAR_NPUDeclareConstantHostRange(arena + 256, 256);
  expect(covers(arena + 256, 256), "the exact declared range is not covered");
  expect(covers(arena + 300, 100), "an interior sub-range is not covered");
  expect(covers(arena + 256, 1), "the first declared byte is not covered");
  expect(covers(arena + 511, 1), "the last declared byte is not covered");
  expect(!covers(arena + 512, 1), "one past the declared end reads as covered");
  expect(!covers(arena + 255, 2), "a range starting before the declared base "
                                  "reads as covered");
  expect(!covers(arena + 400, 200), "a range straddling the declared end "
                                    "reads as covered");
  expect(!covers(arena + 256, 257), "one byte more than the declaration "
                                    "reads as covered");

  // Degenerate queries classify nothing.
  expect(!covers(arena + 256, 0), "a zero-byte query reads as covered");
  expect(MODULAR_NPUConstantHostRangeCovers(nullptr, 16) == 0,
         "a null query reads as covered");

  // A second, disjoint blob accumulates without touching the first, and the
  // gap between them stays uncovered -- a copy spanning two separate blobs is
  // not wholly constant-sourced.
  MODULAR_NPUDeclareConstantHostRange(arena + 1024, 256);
  expect(covers(arena + 1024, 256), "the second declared range is not covered");
  expect(covers(arena + 300, 100), "declaring a second range dropped the "
                                   "first");
  expect(!covers(arena + 512, 512), "the gap between two declarations reads "
                                    "as covered");
  expect(!covers(arena + 400, 800), "a span across the gap reads as covered");

  // Blobs staged back to back merge: the loader declared one constant region
  // in pieces, and a copy spanning the seam is still wholly constant-sourced.
  MODULAR_NPUDeclareConstantHostRange(arena + 512, 512);
  expect(covers(arena + 256, 1024), "touching declarations did not merge");
  expect(covers(arena + 400, 800), "a span across a merged seam is not "
                                   "covered");
  expect(!covers(arena + 256, 1025), "a merged range covers past its end");

  // Re-declaring and overlapping declarations are harmless.
  MODULAR_NPUDeclareConstantHostRange(arena + 256, 256);
  MODULAR_NPUDeclareConstantHostRange(arena + 300, 400);
  expect(covers(arena + 256, 1024), "re-declaration disturbed the registry");
  expect(!covers(arena + 255, 2), "overlap handling moved the declared base");

  // Degenerate declarations register nothing.
  MODULAR_NPUDeclareConstantHostRange(arena + 2048, 0);
  MODULAR_NPUDeclareConstantHostRange(nullptr, 64);
  expect(!covers(arena + 2048, 1), "a zero-byte declaration registered");

  // Forgetting restores the safe default.
  MODULAR_NPUForgetConstantHostRanges();
  expect(!covers(arena + 300, 100), "a forgotten range reads as covered");

  if (failures == 0)
    std::printf("PASS: npu_constant_ranges_test\n");
  return failures == 0 ? 0 : 1;
}
