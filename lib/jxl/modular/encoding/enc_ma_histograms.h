// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef LIB_JXL_MODULAR_ENCODING_ENC_MA_HISTOGRAMS_H_
#define LIB_JXL_MODULAR_ENCODING_ENC_MA_HISTOGRAMS_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "lib/jxl/modular/encoding/enc_ma.h"

namespace jxl {

inline size_t MaHistogramSymbolCount(const int32_t* counts, size_t size) {
  while (size != 0 && counts[size - 1] == 0) --size;
  return size;
}

// Remove parent-zero bins while keeping each nonzero in the same SIMD lane
// and in the same per-lane order. Child distributions are subsets of these
// parents, so no split can introduce a removed symbol. The dense experiment
// removes lane holes too: counts stay exact, floating reduction order may change.
inline bool CompactMaHistograms(size_t predictors, size_t lanes,
                                 std::vector<int32_t>* counts, size_t* stride,
                                 std::array<uint8_t, 256>* mapping,
                                 bool dense = false) {
  if (predictors == 0 || lanes == 0 || *stride > mapping->size() ||
      *stride % lanes != 0 || counts->size() != predictors * *stride) return false;
  std::array<uint8_t, 256> used{};
  for (size_t p = 0; p < predictors; ++p) {
    for (size_t s = 0; s < *stride; ++s) {
      used[s] |= (*counts)[p * *stride + s] != 0;
    }
  }
  std::array<size_t, 256> next{};
  size_t compact_stride = 0;
  size_t dense_symbols = 0;
  mapping->fill(0);
  for (size_t s = 0; s < *stride; ++s) {
    if (!used[s]) continue;
    const size_t lane = s % lanes;
    const size_t destination = dense ? dense_symbols++ : next[lane]++ * lanes + lane;
    (*mapping)[s] = static_cast<uint8_t>(destination);
    compact_stride = dense ? (dense_symbols + lanes - 1) / lanes * lanes
                           : std::max(compact_stride, next[lane] * lanes);
  }
  if (compact_stride == 0 || compact_stride * 4 > *stride * 3) return false;
  std::vector<int32_t> compact(predictors * compact_stride);
  for (size_t p = 0; p < predictors; ++p) {
    for (size_t s = 0; s < *stride; ++s) {
      if (used[s]) compact[p * compact_stride + (*mapping)[s]] =
          (*counts)[p * *stride + s];
    }
  }
  counts->swap(compact);
  *stride = compact_stride;
  return true;
}

struct MaParentHistogram {
  std::array<int32_t, 256> counts{};
  uint32_t weight = 0;
  uint32_t extra_bits = 0;
  size_t symbols = 0;
};

inline void CollectMaParentHistogram(const TreeSamples& samples, size_t pred,
                                     size_t begin, size_t end,
                                     MaParentHistogram* histogram) {
  histogram->counts.fill(0);
  uint32_t weight = 0;
  size_t extra = 0;
  const auto& tokens = samples.RTokens(pred);
  for (size_t i = begin; i < end; ++i) {
    const auto token = tokens[i];
    const size_t count = samples.Count(i);
    histogram->counts[token.tok] += count;
    weight += count;
    extra += token.nbits * count;
  }
  histogram->weight = weight;
  histogram->extra_bits = static_cast<uint32_t>(extra);
  histogram->symbols = MaHistogramSymbolCount(histogram->counts.data(), 256);
}

inline MaParentHistogram SubtractMaParentHistogram(
    const MaParentHistogram& parent, const MaParentHistogram& child) {
  MaParentHistogram result;
  for (size_t i = 0; i < result.counts.size(); ++i) {
    JXL_DASSERT(parent.counts[i] >= child.counts[i]);
    result.counts[i] = parent.counts[i] - child.counts[i];
  }
  result.weight = parent.weight - child.weight;
  result.extra_bits = parent.extra_bits - child.extra_bits;
  result.symbols = MaHistogramSymbolCount(result.counts.data(), result.counts.size());
  return result;
}

inline std::vector<size_t> MaActiveHistogramBlocks(const int32_t* counts,
                                                   size_t size, size_t lanes) {
  std::vector<size_t> blocks;
  if (lanes == 0) return blocks;
  for (size_t base = 0; base < size; base += lanes) {
    for (size_t i = base; i < std::min(size, base + lanes); ++i) {
      if (counts[i] != 0) {
        blocks.push_back(base);
        break;
      }
    }
  }
  return blocks;
}

struct MaHistogramBudget {
  static constexpr size_t kLimit = 32 * 1024 * 1024;
  size_t used = 0;
  size_t peak = 0;
};

struct MaHistogramShards {
  std::vector<int32_t> counts;
  std::vector<size_t> extra;
};

template <bool kStatic>
bool MaSamePropertyPartition(const TreeSamples& samples, size_t a, size_t b,
                             size_t begin, size_t end) {
  return samples.SameProperty<kStatic>(a, b, begin, end);
}

// Adjacent predictors share property values and weights. Keep separate exact
// histograms while loading those inputs only once per sample.
template <bool kStatic>
void CollectPairedMaHistogram(const TreeSamples& samples, size_t prop_index,
                             size_t begin, size_t end, size_t symbols,
                             int32_t* first, int32_t* second,
                             size_t* extra_first, size_t* extra_second,
                             size_t first_predictor = 0) {
  JXL_DASSERT(first_predictor + 1 < samples.NumPredictors());
  const auto& first_tokens = samples.RTokens(first_predictor);
  const auto& second_tokens = samples.RTokens(first_predictor + 1);
  for (size_t i = begin; i < end; ++i) {
    const size_t value = samples.Property<kStatic>(prop_index, i);
    const size_t offset = value * symbols;
    const size_t count = samples.Count(i);
    const auto a = first_tokens[i], b = second_tokens[i];
    first[offset + a.tok] += count;
    second[offset + b.tok] += count;
    extra_first[value] += a.nbits * count;
    extra_second[value] += b.nbits * count;
  }
}

// Four independent accumulators avoid a long load/add/store dependency when
// neighboring samples repeatedly hit the same property/residual bucket.
template <bool kStatic>
void CollectShardedMaHistogram(const TreeSamples& samples, size_t pred,
                               size_t prop_index, size_t begin, size_t end,
                               size_t symbols, size_t values, int32_t* counts,
                               size_t* extra, MaHistogramShards* scratch) {
  const size_t cells = values * symbols;
  scratch->counts.assign(3 * cells, 0);
  scratch->extra.assign(3 * values, 0);
  int32_t* count_shards[] = {counts, scratch->counts.data(),
      scratch->counts.data() + cells, scratch->counts.data() + 2 * cells};
  size_t* extra_shards[] = {extra, scratch->extra.data(),
      scratch->extra.data() + values, scratch->extra.data() + 2 * values};
  const auto& tokens = samples.RTokens(pred);
  const auto add = [&](size_t i, size_t shard) {
    const auto token = tokens[i];
    const size_t count = samples.Count(i);
    const size_t value = samples.Property<kStatic>(prop_index, i);
    count_shards[shard][value * symbols + token.tok] += count;
    extra_shards[shard][value] += token.nbits * count;
  };
  size_t i = begin;
  for (; end - i >= 4; i += 4) {
    add(i, 0);
    add(i + 1, 1);
    add(i + 2, 2);
    add(i + 3, 3);
  }
  for (; i < end; ++i) add(i, 0);
  for (size_t cell = 0; cell < cells; ++cell) {
    counts[cell] += count_shards[1][cell] + count_shards[2][cell] + count_shards[3][cell];
  }
  for (size_t value = 0; value < values; ++value) {
    extra[value] += extra_shards[1][value] + extra_shards[2][value] + extra_shards[3][value];
  }
}

// Only the larger child inherits this cache. Its parent's histograms become
// the child's by subtracting the sibling's samples. Quantization and residual
// tokens are fixed for the lifetime of one ComputeBestTree invocation.
class MaHistogramCache {
 public:
  static std::unique_ptr<MaHistogramCache> Create(
      const TreeSamples& samples, size_t symbols, MaHistogramBudget* budget,
      bool property_counts = false) {
    if (symbols == 0 || symbols > 256 || samples.NumPredictors() == 0 ||
        budget->used > MaHistogramBudget::kLimit) return nullptr;
    std::vector<size_t> offsets(1, 0);
    for (size_t p = 0; p < samples.NumProperties(); ++p) {
      const size_t values = samples.NumPropertyValues(p);
      if (values > MaHistogramBudget::kLimit - offsets.back()) return nullptr;
      offsets.push_back(offsets.back() + values);
    }
    const size_t bytes_per_cell = symbols * sizeof(int32_t) + sizeof(size_t);
    const size_t overhead = offsets.size() * sizeof(size_t) +
                            samples.NumProperties() + sizeof(MaHistogramCache) +
                            (property_counts ? offsets.back() *
                                 (sizeof(int) + sizeof(uint32_t)) : 0);
    if (overhead > MaHistogramBudget::kLimit - budget->used ||
        offsets.back() > (MaHistogramBudget::kLimit - budget->used - overhead) /
                            bytes_per_cell / samples.NumPredictors()) {
      return nullptr;
    }
    const size_t cells = offsets.back() * samples.NumPredictors();
    return std::unique_ptr<MaHistogramCache>(new MaHistogramCache(
        std::move(offsets), samples.NumPredictors(), symbols, budget,
        overhead + cells * bytes_per_cell, property_counts));
  }

  ~MaHistogramCache() { budget_->used -= bytes_; }
  MaHistogramCache(const MaHistogramCache&) = delete;
  MaHistogramCache& operator=(const MaHistogramCache&) = delete;

  bool Has(size_t prop) const { return valid_[prop] != 0; }

  // A skipped property must not pass an unsubtracted parent histogram to a child.
  void Invalidate(size_t prop) { valid_[prop] = 0; }

  void CaptureProperty(size_t prop, const int* used, const uint32_t* weight) {
    const size_t size = offsets_[prop + 1] - offsets_[prop];
    std::copy_n(used, size, used_.data() + offsets_[prop]);
    std::copy_n(weight, size, weight_.data() + offsets_[prop]);
  }

  template <bool kStatic>
  void SubtractProperty(const TreeSamples& samples, size_t prop,
                        size_t removed_begin, size_t removed_end,
                        int* used, uint32_t* weight) {
    const size_t prop_index = kStatic ? prop : prop - samples.NumStaticProps();
    for (size_t i = removed_begin; i < removed_end; ++i) {
      const size_t cell = offsets_[prop] + samples.Property<kStatic>(prop_index, i);
      JXL_DASSERT(used_[cell] > 0 && weight_[cell] >= samples.Count(i));
      --used_[cell];
      weight_[cell] -= samples.Count(i);
    }
    const size_t size = offsets_[prop + 1] - offsets_[prop];
    std::copy_n(used_.data() + offsets_[prop], size, used);
    std::copy_n(weight_.data() + offsets_[prop], size, weight);
  }

  void Capture(size_t prop, size_t pred, const int32_t* counts,
               const size_t* extra, size_t symbols) {
    JXL_DASSERT(symbols <= symbols_);
    const size_t base = Base(prop, pred);
    for (size_t v = 0; v < offsets_[prop + 1] - offsets_[prop]; ++v) {
      std::copy_n(counts + v * symbols, symbols,
                  counts_.data() + (base + v) * symbols_);
      extra_[base + v] = extra[v];
    }
    valid_[prop] = 1;
  }

  template <bool kStatic>
  void SubtractAndCopy(const TreeSamples& samples, size_t prop, size_t pred,
                       size_t removed_begin, size_t removed_end,
                       int32_t* counts, size_t* extra, size_t symbols) {
    JXL_DASSERT(symbols <= symbols_);
    const size_t base = Base(prop, pred);
    const size_t prop_index = kStatic ? prop : prop - samples.NumStaticProps();
    const auto& tokens = samples.RTokens(pred);
    for (size_t i = removed_begin; i < removed_end; ++i) {
      const size_t v = samples.Property<kStatic>(prop_index, i);
      const auto token = tokens[i];
      const size_t count = samples.Count(i);
      auto& frequency = counts_[(base + v) * symbols_ + token.tok];
      JXL_DASSERT(frequency >= static_cast<int32_t>(count));
      frequency -= static_cast<int32_t>(count);
      JXL_DASSERT(extra_[base + v] >= token.nbits * count);
      extra_[base + v] -= token.nbits * count;
    }
    Copy(prop, pred, counts, extra, symbols);
  }

  template <bool kStatic>
  void SubtractPairedAndCopy(
      const TreeSamples& samples, size_t prop, size_t first_pred,
      size_t removed_begin, size_t removed_end, int32_t* first, int32_t* second,
      size_t* first_extra, size_t* second_extra, size_t symbols) {
    JXL_DASSERT(first_pred + 1 < predictors_ && symbols <= symbols_);
    const size_t first_base = Base(prop, first_pred);
    const size_t second_base = Base(prop, first_pred + 1);
    const size_t prop_index = kStatic ? prop : prop - samples.NumStaticProps();
    const auto& a = samples.RTokens(first_pred);
    const auto& b = samples.RTokens(first_pred + 1);
    for (size_t i = removed_begin; i < removed_end; ++i) {
      const size_t value = samples.Property<kStatic>(prop_index, i);
      const size_t weight = samples.Count(i);
      auto& first_count = counts_[(first_base + value) * symbols_ + a[i].tok];
      auto& second_count = counts_[(second_base + value) * symbols_ + b[i].tok];
      JXL_DASSERT(first_count >= static_cast<int32_t>(weight));
      JXL_DASSERT(second_count >= static_cast<int32_t>(weight));
      first_count -= static_cast<int32_t>(weight);
      second_count -= static_cast<int32_t>(weight);
      JXL_DASSERT(extra_[first_base + value] >= a[i].nbits * weight);
      JXL_DASSERT(extra_[second_base + value] >= b[i].nbits * weight);
      extra_[first_base + value] -= a[i].nbits * weight;
      extra_[second_base + value] -= b[i].nbits * weight;
    }
    Copy(prop, first_pred, first, first_extra, symbols);
    Copy(prop, first_pred + 1, second, second_extra, symbols);
  }

  void CopyPredictor(size_t prop, size_t source, size_t destination) {
    const size_t values = offsets_[prop + 1] - offsets_[prop];
    const size_t from = Base(prop, source), to = Base(prop, destination);
    std::copy_n(counts_.data() + from * symbols_, values * symbols_,
                counts_.data() + to * symbols_);
    std::copy_n(extra_.data() + from, values, extra_.data() + to);
  }

 private:
  void Copy(size_t prop, size_t pred, int32_t* counts, size_t* extra,
            size_t symbols) const {
    const size_t base = Base(prop, pred);
    const size_t values = offsets_[prop + 1] - offsets_[prop];
    if (symbols == symbols_) {
      std::copy_n(counts_.data() + base * symbols_, values * symbols, counts);
    } else {
      for (size_t v = 0; v < values; ++v) {
        std::copy_n(counts_.data() + (base + v) * symbols_, symbols,
                    counts + v * symbols);
      }
    }
    std::copy_n(extra_.data() + base, values, extra);
  }

  MaHistogramCache(std::vector<size_t> offsets, size_t predictors,
                   size_t symbols, MaHistogramBudget* budget, size_t bytes,
                   bool property_counts)
      : offsets_(std::move(offsets)),
        predictors_(predictors), symbols_(symbols), budget_(budget),
        bytes_(bytes), counts_(offsets_.back() * predictors * symbols),
        extra_(offsets_.back() * predictors), valid_(offsets_.size() - 1),
        used_(property_counts ? offsets_.back() : 0),
        weight_(property_counts ? offsets_.back() : 0) {
    budget_->used += bytes_;
    budget_->peak = std::max(budget_->peak, budget_->used);
  }
  size_t Base(size_t prop, size_t pred) const {
    return offsets_[prop] * predictors_ +
           pred * (offsets_[prop + 1] - offsets_[prop]);
  }
  std::vector<size_t> offsets_;
  size_t predictors_;
  size_t symbols_;
  MaHistogramBudget* budget_;
  size_t bytes_;
  std::vector<int32_t> counts_;
  std::vector<size_t> extra_;
  std::vector<uint8_t> valid_;
  std::vector<int> used_;
  std::vector<uint32_t> weight_;
};

}  // namespace jxl

#endif  // LIB_JXL_MODULAR_ENCODING_ENC_MA_HISTOGRAMS_H_
