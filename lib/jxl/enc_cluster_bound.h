// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef LIB_JXL_ENC_CLUSTER_BOUND_H_
#define LIB_JXL_ENC_CLUSTER_BOUND_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace jxl {

using ClusterEntropySketch = std::array<double, 16>;

inline ClusterEntropySketch MakeClusterEntropySketch(const std::vector<int32_t>& counts) {
  ClusterEntropySketch sketch{};
  for (size_t i = 0; i < counts.size(); ++i) {
    size_t bucket = i;
    if (i >= 8) {
      bucket = 8;
      for (size_t value = i >> 4; value && bucket < 15; value >>= 1) ++bucket;
    }
    sketch[bucket] += counts[i];
  }
  return sketch;
}

// Grouping symbols only decreases Shannon entropy. Subtract a rounding margin
// before comparing with the existing float/fixed-point ANS population costs.
inline double ClusterMergeLowerBound(const ClusterEntropySketch& a,
                                     const ClusterEntropySketch& b) {
  double total = 0, entropy = 0;
  for (size_t i = 0; i < a.size(); ++i) total += a[i] + b[i];
  if (total == 0) return 0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double count = a[i] + b[i];
    if (count) entropy += count * std::log2(total / count);
  }
  return std::max(0.0, entropy - 1.0 - total * 0.0001);
}

}  // namespace jxl

#endif  // LIB_JXL_ENC_CLUSTER_BOUND_H_
