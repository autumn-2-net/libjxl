// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef LIB_JXL_MODULAR_ENCODING_ENC_MA_ENTROPY_H_
#define LIB_JXL_MODULAR_ENCODING_ENC_MA_ENTROPY_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/jxl/ans_params.h"

namespace jxl {

inline constexpr uint32_t kMaLogTableLimit = 65536;

inline const float* MaCountLogTable() {
  static const std::vector<float> table = [] {
    std::vector<float> result(kMaLogTableLimit + 1);
    for (size_t i = 1; i < result.size(); ++i)
      result[i] = static_cast<float>(std::log2(static_cast<double>(i)));
    return result;
  }();
  return table.data();
}

// Incremental entropy with the same probability floor as MA's histogram
// model. A frequency-of-frequencies Fenwick tree tracks the clipped tail;
// updating a split touches changed counts rather than taking every log again.
class MaClippedEntropy {
 public:
  MaClippedEntropy(const int32_t* counts, size_t symbols, uint32_t maximum_total,
                   const double* table, size_t table_max, bool sliding = false)
      : table_(table), table_max_(table_max), sliding_(sliding),
        rare_(maximum_total > ANS_TAB_SIZE
                  ? (maximum_total - 1) / ANS_TAB_SIZE + 1 : 0) {
    for (size_t i = 0; i < symbols; ++i) Replace(0, counts[i]);
  }

  void Replace(uint32_t before, uint32_t after) {
    if (before == after) return;
    const double old_entropy = CountEntropy(before);
    const double new_entropy = CountEntropy(after);
    sum_ += new_entropy - old_entropy;
    Update(before, -int64_t{before}, -old_entropy);
    Update(after, after, new_entropy);
  }

  float Bits(uint32_t total) {
    if (total == 0) return 0;
    double clipped_sum = 0;
    int64_t clipped_mass = 0;
    if (!rare_.empty()) {
      size_t index = std::min<size_t>((total - 1) / ANS_TAB_SIZE, rare_.size() - 1);
      if (sliding_) {
        // Totals move monotonically during a split sweep. Each rare-count
        // bucket enters or leaves the clipped tail at most once per sweep.
        while (cursor_ < index) {
          ++cursor_;
          tail_mass_ += rare_[cursor_].mass;
          tail_sum_ += rare_[cursor_].entropy;
        }
        while (cursor_ > index) {
          tail_mass_ -= rare_[cursor_].mass;
          tail_sum_ -= rare_[cursor_].entropy;
          --cursor_;
        }
        clipped_mass = tail_mass_;
        clipped_sum = tail_sum_;
      } else {
        for (; index; index &= index - 1) {
          clipped_mass += rare_[index].mass;
          clipped_sum += rare_[index].entropy;
        }
      }
    }
    const double count_entropy = CountEntropy(total);
    const double log_total = count_entropy / total;
    const double bits = count_entropy - sum_ + clipped_sum +
        clipped_mass * (ANS_LOG_TAB_SIZE - log_total);
    return static_cast<float>(std::max(0.0, bits));
  }

 private:
  double CountEntropy(uint32_t count) const {
    if (count == 0) return 0;
    return count <= table_max_ ? table_[count]
        : count * std::log2(static_cast<double>(count));
  }
  void Update(size_t count, int64_t mass, double entropy) {
    if (count == 0) return;
    if (sliding_) {
      if (count >= rare_.size()) return;
      rare_[count].mass += mass;
      rare_[count].entropy += entropy;
      if (count <= cursor_) {
        tail_mass_ += mass;
        tail_sum_ += entropy;
      }
      return;
    }
    for (size_t i = count; i < rare_.size(); i += i & (~i + 1)) {
      rare_[i].mass += mass;
      rare_[i].entropy += entropy;
    }
  }
  struct RareCount {
    int64_t mass = 0;
    double entropy = 0;
  };
  const double* table_;
  size_t table_max_;
  bool sliding_ = false;
  size_t cursor_ = 0;
  int64_t tail_mass_ = 0;
  double tail_sum_ = 0;
  double sum_ = 0;
  std::vector<RareCount> rare_;
};

}  // namespace jxl

#endif  // LIB_JXL_MODULAR_ENCODING_ENC_MA_ENTROPY_H_
