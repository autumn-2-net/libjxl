// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef LIB_JXL_ENC_LZ77_LCP_H_
#define LIB_JXL_ENC_LZ77_LCP_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "lib/jxl/base/bits.h"

namespace jxl {

// Kasai LCP construction followed by a blocked static range-minimum index.
// Only block minima have a sparse table; partial blocks use prefix/suffix
// minima, or a short scan when both ranks are in the same block.
class LZ77SuffixLcp {
 public:
  void Build(const std::vector<uint32_t>& values, const int32_t* suffixes,
              const std::vector<uint32_t>& ranks) {
    const size_t size = values.size();
    lcp_.assign(size, 0);
    size_t common = 0;
    for (size_t pos = 0; pos < size; ++pos) {
      const size_t rank = ranks[pos];
      if (rank == 0) { common = 0; continue; }
      const size_t previous = suffixes[rank - 1];
      while (pos + common < size && previous + common < size &&
             values[pos + common] == values[previous + common]) ++common;
      lcp_[rank] = static_cast<uint32_t>(common);
      if (common) --common;
    }
    prefix_.resize(size);
    suffix_.resize(size);
    const size_t blocks = (size + kBlock - 1) / kBlock;
    minima_.clear();
    if (blocks == 0) return;
    minima_.emplace_back(blocks, UINT32_MAX);
    for (size_t block = 0; block < blocks; ++block) {
      const size_t begin = block * kBlock, end = std::min(size, begin + kBlock);
      uint32_t low = UINT32_MAX;
      for (size_t i = begin; i < end; ++i) prefix_[i] = low = std::min(low, lcp_[i]);
      minima_[0][block] = low;
      low = UINT32_MAX;
      for (size_t i = end; i > begin; --i) suffix_[i - 1] = low = std::min(low, lcp_[i - 1]);
    }
    for (size_t step = 1; step * 2 <= blocks; step *= 2) {
      const auto& previous = minima_.back();
      std::vector<uint32_t> next(blocks - step * 2 + 1);
      for (size_t i = 0; i < next.size(); ++i)
        next[i] = std::min(previous[i], previous[i + step]);
      minima_.push_back(std::move(next));
    }
  }

  uint32_t Between(size_t a, size_t b) const {
    if (a == b) return UINT32_MAX;
    if (a > b) std::swap(a, b);
    ++a;
    if (a / kBlock == b / kBlock) {
      uint32_t low = UINT32_MAX;
      for (size_t i = a; i <= b; ++i) low = std::min(low, lcp_[i]);
      return low;
    }
    uint32_t low = std::min(suffix_[a], prefix_[b]);
    const size_t begin = a / kBlock + 1, end = b / kBlock;
    if (end > begin) {
      const size_t level = FloorLog2Nonzero(end - begin);
      const size_t width = size_t{1} << level;
      low = std::min(low, std::min(minima_[level][begin], minima_[level][end - width]));
    }
    return low;
  }

 private:
  static constexpr size_t kBlock = 32;
  std::vector<uint32_t> lcp_, prefix_, suffix_;
  std::vector<std::vector<uint32_t>> minima_;
};

}  // namespace jxl

#endif  // LIB_JXL_ENC_LZ77_LCP_H_
