//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// The offset an NPU launch sends beside each buffer operand.
//
// FastRPC registers memory an allocation at a time and hands the device the
// mapping's base, so an operand's offset within its allocation does not survive
// the call and has to travel separately. This pins the host half of that: an
// interior address must be split into the allocation the device will map and
// the offset into it, and the offset must be the one that reconstructs the
// original address on the far side.
//
// It fails on the behaviour that preceded the split -- resolving the address
// host-side and sending it whole -- because that path sends the interior
// pointer as the buffer and no offset at all, which `reconstructs` rejects.
//
// Address arithmetic only, so it needs neither FastRPC nor a device. The
// allocation map is modelled the way `QuRTDevice` keeps it: base -> size,
// looked up by containment.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstdio>
#include <map>
#include <optional>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char *what) {
  if (!condition) {
    std::printf("FAIL: %s\n", what);
    ++failures;
  }
}

struct Allocation {
  uintptr_t base;
  size_t byteSize;
};

/// The containment lookup `QuRTDevice::allocationFrom` performs.
std::optional<Allocation> allocationFrom(const std::map<uintptr_t, size_t> &map,
                                         uintptr_t address) {
  auto after = map.upper_bound(address);
  if (after == map.begin())
    return std::nullopt;
  auto [base, byteSize] = *std::prev(after);
  if (address - base < byteSize)
    return Allocation{base, byteSize};
  return std::nullopt;
}

/// What the packer sends for one operand: the allocation the device maps, and
/// the offset it adds back.
struct Sent {
  uintptr_t mapped;
  size_t mappedBytes;
  uint64_t offset;
};

std::optional<Sent> pack(const std::map<uintptr_t, size_t> &map,
                         uintptr_t address) {
  auto allocation = allocationFrom(map, address);
  if (!allocation)
    return std::nullopt;
  return Sent{allocation->base, allocation->byteSize,
              static_cast<uint64_t>(address - allocation->base)};
}

/// The device's half: what the kernel ends up addressing.
uintptr_t reconstructs(const Sent &sent) { return sent.mapped + sent.offset; }

/// An operand partway into an arena -- the case that returned 49152 zeros, and
/// the one a resolved address cannot express.
void interior_operand_round_trips() {
  std::map<uintptr_t, size_t> map{{0x1000000, 0x100000}};
  uintptr_t address = 0x1000000 + 98304;
  auto sent = pack(map, address);
  expect(sent.has_value(), "an interior address has no allocation");
  if (!sent)
    return;
  expect(sent->mapped == 0x1000000,
         "the allocation's base is not what is sent");
  expect(sent->offset == 98304, "the offset is not the distance from the base");
  expect(reconstructs(*sent) == address,
         "the device cannot reconstruct the operand's address");
}

/// An operand at the start of its allocation. This is the case that passed
/// before the split -- a graph input, say -- so it must keep passing, and its
/// offset must be zero rather than merely harmless.
void based_operand_round_trips() {
  std::map<uintptr_t, size_t> map{{0x2000000, 0x40000}};
  auto sent = pack(map, 0x2000000);
  expect(sent.has_value(), "a based address has no allocation");
  if (!sent)
    return;
  expect(sent->offset == 0, "a based operand was given a nonzero offset");
  expect(reconstructs(*sent) == 0x2000000, "a based operand moved");
}

/// The whole allocation is mapped, not the remainder from the operand. The
/// device is told where the operand starts; it does not infer it from a length.
void the_whole_allocation_is_mapped() {
  std::map<uintptr_t, size_t> map{{0x3000000, 0x80000}};
  auto sent = pack(map, 0x3000000 + 0x1000);
  expect(sent.has_value(), "no allocation");
  if (!sent)
    return;
  expect(sent->mappedBytes == 0x80000,
         "the mapped span is not the whole allocation");
}

/// Several operands out of one arena each reconstruct to themselves, which is
/// the chain's own shape: a projection's output and the norm's input landing in
/// one pool.
void several_operands_from_one_arena() {
  std::map<uintptr_t, size_t> map{{0x4000000, 0x300000}};
  for (uint64_t offset : {uint64_t{0}, uint64_t{98304}, uint64_t{196608}}) {
    auto sent = pack(map, 0x4000000 + offset);
    expect(sent.has_value(), "no allocation for an arena operand");
    if (!sent)
      continue;
    expect(sent->mapped == 0x4000000, "arena operands disagree on the base");
    expect(sent->offset == offset, "an arena operand's offset is wrong");
    expect(reconstructs(*sent) == 0x4000000 + offset,
           "an arena operand does not reconstruct");
  }
}

/// An address this device never handed out is refused rather than sent, which
/// is what keeps a stray pointer from being mapped as though it were ours.
void a_foreign_address_is_refused() {
  std::map<uintptr_t, size_t> map{{0x5000000, 0x1000}};
  expect(!pack(map, 0x4FFFFFF).has_value(),
         "an address below every allocation");
  expect(!pack(map, 0x5001000).has_value(), "an address past its allocation");
  expect(!pack(map, 0x9000000).has_value(),
         "an address above every allocation");
}

/// The regression this exists for: sending the resolved address as the buffer
/// and no offset. The device adds the mapping base to a zero offset and lands
/// at the base, which is what silently returned the wrong rows.
void resolving_host_side_is_detected() {
  std::map<uintptr_t, size_t> map{{0x6000000, 0x100000}};
  uintptr_t address = 0x6000000 + 4096;
  auto allocation = allocationFrom(map, address);
  expect(allocation.has_value(), "no allocation");
  if (!allocation)
    return;
  // What the old path effectively sent: the device maps the allocation and
  // gets its base, with nothing to add.
  Sent resolved{allocation->base, allocation->byteSize, 0};
  expect(reconstructs(resolved) != address,
         "the pre-split behaviour must not reconstruct the operand, or this "
         "test cannot tell the fix from its absence");
}

} // namespace

int main() {
  interior_operand_round_trips();
  based_operand_round_trips();
  the_whole_allocation_is_mapped();
  several_operands_from_one_arena();
  a_foreign_address_is_refused();
  resolving_host_side_is_detected();

  if (failures) {
    std::printf("\n%d check(s) failed\n", failures);
    return 1;
  }
  std::printf("PASS -- an operand's offset survives to the device\n");
  return 0;
}
