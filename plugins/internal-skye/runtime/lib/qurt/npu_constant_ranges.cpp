//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// The registry behind npu_constant_ranges.h: intervals of host memory the
// loader declared as init-time model constants.
//
// Kept as a base-to-end map coalesced on insert, so the query is one
// containment lookup -- the same shape as QuRTDevice's allocation map. Ranges
// that touch are merged because a loader staging blobs back to back has
// declared one constant region in pieces, and a copy spanning the seam is
// still wholly constant-sourced.
//
//===----------------------------------------------------------------------===//

#include "npu_constant_ranges.h"

#include <cstdint>
#include <map>
#include <mutex>

namespace {

std::mutex rangesMu;

/// Declared constant memory, base to exclusive end. A function-local static so
/// a declaration arriving before any other initializer still lands somewhere
/// constructed.
std::map<uintptr_t, uintptr_t> &ranges() {
  static std::map<uintptr_t, uintptr_t> map;
  return map;
}

} // namespace

extern "C" void MODULAR_NPUDeclareConstantHostRange(const void *base,
                                                    size_t byteSize) {
  if (base == nullptr || byteSize == 0)
    return;
  auto start = reinterpret_cast<uintptr_t>(base);
  uintptr_t end = start + byteSize;

  std::lock_guard<std::mutex> lock(rangesMu);
  auto &map = ranges();
  // Absorb every declared range that touches [start, end), then keep the
  // union. The range before the insertion point can still reach into the new
  // one, so the walk starts one earlier when it does.
  auto it = map.upper_bound(start);
  if (it != map.begin() && std::prev(it)->second >= start)
    --it;
  while (it != map.end() && it->first <= end) {
    start = std::min(start, it->first);
    end = std::max(end, it->second);
    it = map.erase(it);
  }
  map[start] = end;
}

extern "C" void MODULAR_NPUForgetConstantHostRanges(void) {
  std::lock_guard<std::mutex> lock(rangesMu);
  ranges().clear();
}

extern "C" int MODULAR_NPUConstantHostRangeCovers(const void *base,
                                                  size_t byteSize) {
  if (base == nullptr || byteSize == 0)
    return 0;
  auto start = reinterpret_cast<uintptr_t>(base);

  std::lock_guard<std::mutex> lock(rangesMu);
  auto &map = ranges();
  auto after = map.upper_bound(start);
  if (after == map.begin())
    return 0;
  auto [rangeStart, rangeEnd] = *std::prev(after);
  // Subtraction rather than `start + byteSize <= rangeEnd`, which could wrap.
  return start < rangeEnd && byteSize <= rangeEnd - start ? 1 : 0;
}
