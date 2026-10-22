// Copyright (c) the JPEG XL Project Authors.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "lib/jxl/enc_lz77_bucket.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/enc_lz77_bucket.cc"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

void RelaxCosts(float* costs, uint32_t* lengths, uint32_t* distances,
                uint32_t* contexts, size_t count, uint32_t first_length,
                uint32_t distance, uint32_t context, float cost) {
  const hwy::HWY_NAMESPACE::CappedTag<float, 8> d;
  const hwy::HWY_NAMESPACE::Rebind<uint32_t, decltype(d)> du;
  const auto offered = Set(d, cost);
  size_t i = 0;
  for (; i + Lanes(d) <= count; i += Lanes(d)) {
    const auto old = LoadU(d, costs + i);
    const auto better = Lt(offered, old);
    if (AllFalse(d, better)) continue;
    const auto mask = RebindMask(du, better);
    StoreU(IfThenElse(better, offered, old), d, costs + i);
    StoreU(IfThenElse(mask, Iota(du, first_length + i), LoadU(du, lengths + i)),
           du, lengths + i);
    StoreU(IfThenElse(mask, Set(du, distance), LoadU(du, distances + i)),
           du, distances + i);
    StoreU(IfThenElse(mask, Set(du, context), LoadU(du, contexts + i)),
           du, contexts + i);
  }
  for (; i < count; ++i) {
    if (cost < costs[i]) {
      costs[i] = cost;
      lengths[i] = first_length + i;
      distances[i] = distance;
      contexts[i] = context;
    }
  }
}

uint64_t FingerprintMask(const uint8_t* tags, size_t count, uint8_t tag) {
  const hwy::HWY_NAMESPACE::CappedTag<uint8_t, 32> d;
  const auto wanted = Set(d, tag);
  uint64_t result = 0;
  size_t i = 0;
  for (; i + Lanes(d) <= count; i += Lanes(d)) {
    uint8_t bits[8] = {};
    StoreMaskBits(d, Eq(LoadU(d, tags + i), wanted), bits);
    uint64_t mask = 0;
    for (size_t b = 0; b < (Lanes(d) + 7) / 8; ++b) {
      mask |= uint64_t{bits[b]} << (8 * b);
    }
    result |= mask << i;
  }
  for (; i < count; ++i) {
    if (tags[i] == tag) result |= uint64_t{1} << i;
  }
  return result;
}

void SpecialPrefixMasks(const uint8_t* tags, size_t pos, size_t safe_distance,
                        uint8_t tag, const LZ77SpecialPrefixGroup* groups,
                        size_t count, uint16_t* masks) {
  const hwy::HWY_NAMESPACE::CappedTag<uint8_t, 16> d;
  const auto wanted = Set(d, tag);
  for (size_t g = 0; g < count; ++g) {
    const auto& group = groups[g];
    uint32_t mask = 0;
    if (group.max_distance > safe_distance) {
      const size_t lanes = std::min<size_t>(16, group.max_distance - safe_distance);
      const uint32_t valid = group.valid & ((uint32_t{1} << lanes) - 1);
      if (pos >= group.max_distance) {
        const uint8_t* start = tags + pos - group.max_distance;
        for (size_t i = 0; i < 16; i += Lanes(d)) {
          uint8_t bits[8] = {};
          StoreMaskBits(d, Eq(LoadU(d, start + i), wanted), bits);
          mask |= (uint32_t{bits[0]} | (uint32_t{bits[1]} << 8)) << i;
        }
        mask &= valid;
      } else {
        // The first rows may not have enough history for a whole vector.
        for (size_t i = 0; i < 16; ++i) {
          if (!(valid & (uint32_t{1} << i))) continue;
          const size_t distance = group.max_distance - i;
          if (distance <= pos && tags[pos - distance] == tag) mask |= uint32_t{1} << i;
        }
      }
    }
    masks[g] = static_cast<uint16_t>(mask);
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {
HWY_EXPORT(FingerprintMask);
HWY_EXPORT(SpecialPrefixMasks);
HWY_EXPORT(RelaxCosts);
LZ77RelaxFunction GetLZ77RelaxFunction() {
  HWY_DYNAMIC_DISPATCH(RelaxCosts)(nullptr, nullptr, nullptr, nullptr,
                                  0, 0, 0, 0, 0);
  return HWY_DYNAMIC_POINTER(RelaxCosts);
}
LZ77FingerprintFunction GetLZ77FingerprintFunction() {
  // Resolve the CPU target once before returning a per-stream function pointer.
  uint8_t dummy[1] = {};
  (void)HWY_DYNAMIC_DISPATCH(FingerprintMask)(dummy, 1, 0);
  return HWY_DYNAMIC_POINTER(FingerprintMask);
}
LZ77SpecialPrefixFunction GetLZ77SpecialPrefixFunction() {
  HWY_DYNAMIC_DISPATCH(SpecialPrefixMasks)(nullptr, 0, 0, 0, nullptr, 0, nullptr);
  return HWY_DYNAMIC_POINTER(SpecialPrefixMasks);
}
}  // namespace jxl
#endif
