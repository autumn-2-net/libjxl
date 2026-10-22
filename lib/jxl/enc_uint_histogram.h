// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_UINT_HISTOGRAM_H_
#define LIB_JXL_ENC_UINT_HISTOGRAM_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "lib/jxl/dec_ans.h"
#include "lib/jxl/enc_ans_params.h"

namespace jxl {

// A refining uint config preserves every token and extra-bit count used by the
// supported candidates, without preserving the extra-bit payload itself.
// Original values are still used when writing the bitstream.
class WeightedUintHistogram {
 public:
  static constexpr size_t kBins = (1u << 12) + (32 - 12) * (1u << 7);

  static bool Supports(const HybridUintConfig& cfg) {
    return cfg.split_exponent <= 12 && cfg.msb_in_token <= 2 &&
           cfg.lsb_in_token <= 5 &&
           cfg.msb_in_token + cfg.lsb_in_token <= cfg.split_exponent;
  }

  bool Build(const uint32_t* data, size_t size) {
    values_.clear();
    if (size < 4096 ||
        size > static_cast<size_t>(std::numeric_limits<ANSHistBin>::max())) {
      return false;
    }
    std::vector<uint32_t> slots(kBins);
    values_.reserve(std::min(size, kBins));
    const HybridUintConfig refining(12, 2, 5);
    for (size_t i = 0; i < size; ++i) {
      uint32_t bucket, nbits, bits;
      refining.Encode(data[i], &bucket, &nbits, &bits);
      uint32_t& slot = slots[bucket];
      if (slot == 0) {
        values_.push_back({data[i], 1});
        slot = static_cast<uint32_t>(values_.size());
      } else {
        ++values_[slot - 1].count;
      }
    }
    return true;
  }

  // The caller supplies a zeroed, sufficiently large histogram, as it does for
  // FastAdd. Keep uint32 arithmetic for extra bits, matching EstimateTokenCost.
  uint32_t AddTo(const HybridUintConfig& cfg, Histogram* histogram) const {
    JXL_DASSERT(Supports(cfg));
    uint32_t extra_bits = 0;
    for (const auto& entry : values_) {
      uint32_t token, nbits, bits;
      cfg.Encode(entry.value, &token, &nbits, &bits);
      JXL_DASSERT(token < histogram->counts.size());
      histogram->counts[token] += entry.count;
      extra_bits += nbits * entry.count;
    }
    return extra_bits;
  }

  size_t size() const { return values_.size(); }

 private:
  struct ValueCount {
    uint32_t value;
    uint32_t count;
  };
  std::vector<ValueCount> values_;
};

}  // namespace jxl

#endif  // LIB_JXL_ENC_UINT_HISTOGRAM_H_
