// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef LIB_JXL_ENC_LZ77_COST_H_
#define LIB_JXL_ENC_LZ77_COST_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "lib/jxl/base/bits.h"

namespace jxl {

struct LZ77DistanceBound {
  float cost = -std::numeric_limits<float>::infinity();
  uint32_t symbol = 0;
};

// Every symbol in a HybridUint(4,2,0) class has the same model cost.
// Include all classes in the window, even distances absent at this position.
template <typename Price>
LZ77DistanceBound MinimumLZ77DistancePrice(uint32_t maximum,
                                        const Price& price) {
  LZ77DistanceBound result{std::numeric_limits<float>::infinity(), 0};
  for (uint64_t first = 0; first <= maximum;) {
    const float bits = price(static_cast<uint32_t>(first));
    if (bits < result.cost) result = {bits, static_cast<uint32_t>(first)};
    first += first < 16 ? 1 : uint64_t{1} << (FloorLog2Nonzero(first) - 2);
  }
  return result;
}

// HybridUint(4,2,0) has only 128 distance price classes, even with a uint32
// dictionary. Cache prices by class instead of a window-sized float array.
class LZ77DistancePrices {
 public:
  template <typename Price>
  explicit LZ77DistancePrices(const Price& price) {
    for (uint32_t i = 0; i < 16; ++i) prices_[i] = price(i);
    for (uint32_t bit = 4; bit < 32; ++bit) {
      for (uint32_t top = 0; top < 4; ++top) {
        prices_[16 + 4 * (bit - 4) + top] = price((4u + top) << (bit - 2));
      }
    }
  }
  float operator()(uint32_t distance) const {
    if (distance < 16) return prices_[distance];
    const uint32_t bit = FloorLog2Nonzero(distance);
    return prices_[16 + 4 * (bit - 4) + ((distance >> (bit - 2)) & 3)];
  }
 private:
  std::array<float, 128> prices_;
};

// One nondominated distance is best throughout [begin, end). Length-code
// boundaries split this into constant-cost runs. The optional boundary parser
// retains both ends of long runs instead of relaxing every interior endpoint.
template <typename Cost, typename LengthCost, typename Visit>
void VisitLZ77CostIntervals(size_t minimum, size_t begin, size_t end,
                           uint32_t distance, float distance_cost, Cost prefix_cost,
                           const LengthCost& length_cost, const Visit& visit) {
  if (begin >= end) return;
  const size_t limit = end - minimum;
  size_t extent = 1;
  while (minimum + extent <= begin) {
    extent = extent > limit / 2 ? limit : extent * 2;
  }
  while (begin < end) {
    const size_t stop = minimum + extent;
    const Cost cost = prefix_cost + (length_cost(begin - minimum) + distance_cost);
    visit(begin, stop, distance, cost);
    begin = stop;
    if (extent == limit) break;
    extent = extent > limit / 2 ? limit : extent * 2;
  }
}

template <typename Cost, typename LengthCost, typename Visit>
void VisitLZ77CostRange(size_t minimum, size_t begin, size_t end,
                       uint32_t distance, float distance_cost, Cost prefix_cost,
                       const LengthCost& length_cost, const Visit& visit,
                       bool boundary_only) {
  VisitLZ77CostIntervals(minimum, begin, end, distance, distance_cost, prefix_cost,
      length_cost, [&](size_t first, size_t stop, uint32_t symbol, Cost cost) {
        if (!boundary_only || stop - first <= 16) {
          for (size_t j = first; j < stop; ++j) visit(j, symbol, cost);
        } else {
          for (size_t j = first; j < first + 8; ++j) visit(j, symbol, cost);
          for (size_t j = stop - 8; j < stop; ++j) visit(j, symbol, cost);
        }
      });
}

// With HybridUintConfig(0, 0, 0), lengths 0, 1, [2,3], [4,7], ...
// have constant token and extra-bit costs. Reuse those costs and each run's
// distance cost, but still visit every possible match endpoint in order.
template <typename Cost, typename LengthCost, typename DistanceCost, typename Visit>
void VisitLZ77CostSpans(size_t min_length,
                       const std::vector<uint32_t>& distances,
                       Cost prefix_cost, const LengthCost& length_cost,
                       const DistanceCost& distance_cost, const Visit& visit) {
  static_assert(std::is_floating_point<Cost>::value, "floating-point costs required");
  if (distances.size() <= min_length) return;
  const size_t limit = distances.size() - min_length;
  size_t extent = 1;
  size_t j = min_length;
  for (;;) {
    const size_t end = min_length + extent;
    const float len_cost = length_cost(j - min_length);
    do {
      const uint32_t distance = distances[j];
      const Cost cost = prefix_cost + (len_cost + distance_cost(distance));
      do {
        visit(j, distance, cost);
        ++j;
      } while (j < end && distances[j] == distance);
    } while (j < end);
    if (extent == limit) break;
    // Saturate before doubling to also work on 32-bit size_t.
    extent = extent > limit / 2 ? limit : extent * 2;
  }
}

}  // namespace jxl

#endif  // LIB_JXL_ENC_LZ77_COST_H_
