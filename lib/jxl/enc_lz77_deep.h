// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef LIB_JXL_ENC_LZ77_DEEP_H_
#define LIB_JXL_ENC_LZ77_DEEP_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace jxl {

// Longer prefix chains supplement short-match history. Hashes propose earlier
// positions only; the caller must still verify symbols, window and bit cost.
class LZ77DeepPrefixChain {
 public:
  void Build(const std::vector<uint32_t>& values, size_t prefix) {
    previous_.assign(values.size(), UINT32_MAX);
    if (values.size() < prefix || prefix == 0) return;
    constexpr uint64_t kBase = 0x9e3779b185ebca87ULL;
    constexpr size_t kBuckets = 1u << 16;
    std::vector<uint32_t> heads(kBuckets, UINT32_MAX);
    uint64_t hash = 0, power = 1;
    for (size_t j = 0; j < prefix; ++j) {
      hash = hash * kBase + values[j] + 1ULL;
      if (j + 1 < prefix) power *= kBase;
    }
    for (size_t pos = 0; pos + prefix <= values.size(); ++pos) {
      const uint64_t mixed = (hash ^ (hash >> 32)) * 0xd6e8feb86659fd93ULL;
      const size_t bucket = (mixed >> 48) & (kBuckets - 1);
      previous_[pos] = heads[bucket];
      heads[bucket] = static_cast<uint32_t>(pos);
      if (pos + prefix < values.size()) {
        hash = (hash - (values[pos] + 1ULL) * power) * kBase +
               values[pos + prefix] + 1ULL;
      }
    }
  }

  template <typename Visit>
  void Find(size_t pos, size_t window, size_t count, const Visit& visit) const {
    if (previous_.empty()) return;
    uint32_t previous = previous_[pos];
    for (size_t i = 0; i < count && previous != UINT32_MAX; ++i) {
      if (pos - previous >= window) break;
      visit(previous);
      previous = previous_[previous];
    }
  }

 private:
  std::vector<uint32_t> previous_;
};

}  // namespace jxl

#endif  // LIB_JXL_ENC_LZ77_DEEP_H_
