// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/modular/encoding/enc_ma.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <queue>
#include <type_traits>
#include <vector>

#include "lib/jxl/ans_params.h"
#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/enc_lossless_profile.h"
#include "lib/jxl/enc_lossless_probe_pool.h"
#include "lib/jxl/modular/encoding/dec_ma.h"
#include "lib/jxl/modular/encoding/enc_ma_histograms.h"
#include "lib/jxl/modular/encoding/enc_ma_entropy.h"
#include "lib/jxl/modular/encoding/enc_ma_racing.h"
#include "lib/jxl/enc_lossless_workers.h"
#include "lib/jxl/modular/encoding/ma_common.h"
#include "lib/jxl/modular/modular_image.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/modular/encoding/enc_ma.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/base/fast_math-inl.h"
#include "lib/jxl/base/random.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/modular/encoding/context_predict.h"
#include "lib/jxl/modular/options.h"
#include "lib/jxl/pack_signed.h"
HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Eq;
using hwy::HWY_NAMESPACE::IfThenElse;
using hwy::HWY_NAMESPACE::Lt;
using hwy::HWY_NAMESPACE::Max;

const HWY_FULL(float) df;
const HWY_FULL(int32_t) di;
// Pad array size for use in EstimateBits.
size_t Padded(size_t x) {
  JXL_DASSERT(Lanes(df) == Lanes(di));
  return RoundUpTo(x, Lanes(df));
}

// Compute entropy of the histogram, taking into account the minimum probability
// for symbols with non-zero counts.
float EstimateBits(const int32_t *counts, size_t num_symbols) {
  JXL_DASSERT(num_symbols == Padded(num_symbols));
  auto total_v = Zero(di);
  for (size_t i = 0; i < num_symbols; i += Lanes(di)) {
    const auto counts_iv = LoadU(di, &counts[i]);
    total_v = Add(total_v, counts_iv);
  }
  total_v = SumOfLanes(di, total_v);

  const auto minprob = Set(df, 1.0f / ANS_TAB_SIZE);
  const auto inv_total = Set(df, 1.0f / GetLane(total_v));
  auto bits_lanes = Zero(df);
  for (size_t i = 0; i < num_symbols; i += Lanes(df)) {
    const auto counts_iv = LoadU(di, &counts[i]);
    const auto counts_fv = ConvertTo(df, counts_iv);
    const auto probs = Mul(counts_fv, inv_total);
    const auto mprobs = Max(probs, minprob);
    const auto nbps = IfThenZeroElse(Eq(counts_iv, total_v),
                                     BitCast(di, FastLog2f(df, mprobs)));
    bits_lanes = Sub(bits_lanes, Mul(counts_fv, BitCast(df, nbps)));
  }
  return GetLane(SumOfLanes(df, bits_lanes));
}

constexpr uint32_t kEntropyTableMaxCount = 256;

// Most tree nodes have few samples. Reuse the exact SIMD expression rather
// than approximating entropy or changing its floating-point reduction order.
const float* SmallHistogramEntropy(uint32_t total) {
  static const uint32_t limit = LosslessWorkerSetting(
      std::getenv("JXL_MA_ENTROPY_TABLE_LIMIT"), kEntropyTableMaxCount, 4096);
  if (total > limit) return nullptr;
  static const std::vector<float> table = [] {
    std::vector<float> values(
        (limit + 1) * (limit + 2) / 2 + Lanes(df));
    for (uint32_t n = 1; n <= limit; ++n) {
      const auto total_v = Set(di, static_cast<int32_t>(n));
      const auto inv_total = Set(df, 1.0f / n);
      const auto minprob = Set(df, 1.0f / ANS_TAB_SIZE);
      float* row = values.data() + n * (n + 1) / 2;
      for (uint32_t i = 0; i <= n; i += Lanes(df)) {
        const auto counts = Iota(di, static_cast<int32_t>(i));
        const auto cf = ConvertTo(df, counts);
        const auto nbps = IfThenZeroElse(
            Eq(counts, total_v),
            BitCast(di, FastLog2f(df, Max(Mul(cf, inv_total), minprob))));
        // Tail lanes overlap the next row, which is initialized afterwards.
        StoreU(Mul(cf, BitCast(df, nbps)), df, row + i);
      }
    }
    return values;
  }();
  return table.data() + total * (total + 1) / 2;
}

template <bool kLookup>
HWY_INLINE auto EntropyContribution(decltype(Zero(di)) counts,
                                    decltype(Zero(di)) total,
                                    decltype(Zero(df)) inverse,
                                    const float* table) {
  if (kLookup) return GatherIndex(df, table, counts);
  const auto cf = ConvertTo(df, counts);
  const auto nbps = IfThenZeroElse(
      Eq(counts, total),
      BitCast(di, FastLog2f(df, Max(Mul(cf, inverse),
                                  Set(df, 1.0f / ANS_TAB_SIZE)))));
  return Mul(cf, BitCast(df, nbps));
}

// The weighted sample totals depend only on the property split, not on the
// predictor. Update both histograms and evaluate them in one pass, preserving
// EstimateBits' floating-point operations and per-side lane accumulation order.
template <bool kLookupAbove, bool kLookupBelow, bool kSparse>
void MoveCountsAndEstimateBitsImpl(
    int32_t* above, int32_t* below, int32_t* increase, size_t num_symbols,
    uint32_t total_above, uint32_t total_below, const float* table_above,
    const float* table_below, const size_t* blocks, size_t num_blocks,
    float* bits_above, float* bits_below) {
  JXL_DASSERT(num_symbols == Padded(num_symbols));
  const auto total_r = Set(di, static_cast<int32_t>(total_above));
  const auto total_l = Set(di, static_cast<int32_t>(total_below));
  const auto inv_r = Set(df, 1.0f / GetLane(total_r));
  const auto inv_l = Set(df, 1.0f / GetLane(total_l));
  auto bits_r = Zero(df);
  auto bits_l = Zero(df);
  const size_t iterations = kSparse ? num_blocks : num_symbols / Lanes(di);
  for (size_t block = 0; block < iterations; ++block) {
    const size_t i = kSparse ? blocks[block] : block * Lanes(di);
    const auto delta = LoadU(di, increase + i);
    const auto r = Sub(LoadU(di, above + i), delta);
    const auto l = Add(LoadU(di, below + i), delta);
    StoreU(r, di, above + i);
    StoreU(l, di, below + i);
    StoreU(Zero(di), di, increase + i);
    bits_r = Sub(bits_r, EntropyContribution<kLookupAbove>(
                             r, total_r, inv_r, table_above));
    bits_l = Sub(bits_l, EntropyContribution<kLookupBelow>(
                             l, total_l, inv_l, table_below));
  }
  *bits_above = GetLane(SumOfLanes(df, bits_r));
  *bits_below = GetLane(SumOfLanes(df, bits_l));
}

template <bool kSparse>
void MoveCountsAndEstimateBits(int32_t* above, int32_t* below, int32_t* increase,
                               size_t num_symbols, uint32_t total_above,
                               uint32_t total_below, const size_t* blocks,
                               size_t num_blocks, float* bits_above,
                               float* bits_below) {
  const float* table_above = SmallHistogramEntropy(total_above);
  const float* table_below = SmallHistogramEntropy(total_below);
  if (table_above && table_below) {
    MoveCountsAndEstimateBitsImpl<true, true, kSparse>(
        above, below, increase, num_symbols, total_above, total_below,
        table_above, table_below, blocks, num_blocks, bits_above, bits_below);
  } else if (table_above) {
    MoveCountsAndEstimateBitsImpl<true, false, kSparse>(
        above, below, increase, num_symbols, total_above, total_below,
        table_above, table_below, blocks, num_blocks, bits_above, bits_below);
  } else if (table_below) {
    MoveCountsAndEstimateBitsImpl<false, true, kSparse>(
        above, below, increase, num_symbols, total_above, total_below,
        table_above, table_below, blocks, num_blocks, bits_above, bits_below);
  } else {
    MoveCountsAndEstimateBitsImpl<false, false, kSparse>(
        above, below, increase, num_symbols, total_above, total_below,
        nullptr, nullptr, blocks, num_blocks, bits_above, bits_below);
  }
}

constexpr uint32_t kCountEntropyTableSize = 65536;

// Cache log(count), not log(count / total), so larger histograms can reuse a
// small one-dimensional table. The probability floor is unchanged; floating
// point rounding may select a different valid tree.
void MoveCountsAndLogBits(int32_t* above, int32_t* below, int32_t* increase,
                          size_t symbols, uint32_t total_above,
                          uint32_t total_below, const float* logs,
                          float* bits_above, float* bits_below) {
  const auto lr = Set(df, logs[total_above]);
  const auto ll = Set(df, logs[total_below]);
  const auto cap = Set(df, float{ANS_LOG_TAB_SIZE});
  auto br = Zero(df), bl = Zero(df);
  for (size_t s = 0; s < symbols; s += Lanes(di)) {
    const auto delta = LoadU(di, increase + s);
    const auto r = Sub(LoadU(di, above + s), delta);
    const auto l = Add(LoadU(di, below + s), delta);
    StoreU(r, di, above + s);
    StoreU(l, di, below + s);
    StoreU(Zero(di), di, increase + s);
    br = Add(br, Mul(ConvertTo(df, r), Min(cap, Sub(lr, GatherIndex(df, logs, r)))));
    bl = Add(bl, Mul(ConvertTo(df, l), Min(cap, Sub(ll, GatherIndex(df, logs, l)))));
  }
  *bits_above = GetLane(SumOfLanes(df, br));
  *bits_below = GetLane(SumOfLanes(df, bl));
}

const double* CountEntropyTable() {
  static const std::vector<double> table = [] {
    std::vector<double> values(kCountEntropyTableSize + 1);
    for (uint32_t n = 1; n <= kCountEntropyTableSize; ++n) {
      values[n] = n * std::log2(static_cast<double>(n));
    }
    return values;
  }();
  return table.data();
}

HWY_INLINE double CountEntropy(uint32_t count, const double* table) {
  return count <= kCountEntropyTableSize
             ? table[count]
             : count * std::log2(static_cast<double>(count));
}

double HistogramCountEntropy(const int32_t* counts, size_t num_symbols,
                              const double* table) {
  double sum = 0;
  for (size_t i = 0; i < num_symbols; ++i) sum += CountEntropy(counts[i], table);
  return sum;
}

// H = N log2(N) - sum(c log2(c)). Moving one property value only changes bins
// with a nonzero increment. Double accumulators limit cancellation drift.
// This is a scoring heuristic, not the old min-probability-clipped estimate;
// final candidate selection still compares complete encoded byte counts.
void MoveCountsAndShannonBits(int32_t* above, int32_t* below, int32_t* increase,
                              size_t num_symbols, uint32_t total_above,
                              uint32_t total_below, const double* table,
                              double* sum_above, double* sum_below,
                              float* bits_above, float* bits_below) {
  for (size_t i = 0; i < num_symbols; ++i) {
    const int32_t delta = increase[i];
    if (delta == 0) continue;
    const int32_t r = above[i], l = below[i];
    *sum_above += CountEntropy(r - delta, table) - CountEntropy(r, table);
    *sum_below += CountEntropy(l + delta, table) - CountEntropy(l, table);
    above[i] = r - delta;
    below[i] = l + delta;
    increase[i] = 0;
  }
  *bits_above = std::max(0.0, CountEntropy(total_above, table) - *sum_above);
  *bits_below = std::max(0.0, CountEntropy(total_below, table) - *sum_below);
}

void MakeSplitNode(size_t pos, int property, int splitval, Predictor lpred,
                   int64_t loff, Predictor rpred, int64_t roff, Tree *tree) {
  // Note that the tree splits on *strictly greater*.
  (*tree)[pos].lchild = tree->size();
  (*tree)[pos].rchild = tree->size() + 1;
  (*tree)[pos].splitval = splitval;
  (*tree)[pos].property = property;
  tree->emplace_back();
  tree->back().property = -1;
  tree->back().predictor = rpred;
  tree->back().predictor_offset = roff;
  tree->back().multiplier = 1;
  tree->emplace_back();
  tree->back().property = -1;
  tree->back().predictor = lpred;
  tree->back().predictor_offset = loff;
  tree->back().multiplier = 1;
}

enum class IntersectionType { kNone, kPartial, kInside };
IntersectionType BoxIntersects(StaticPropRange needle, StaticPropRange haystack,
                               uint32_t &partial_axis, uint32_t &partial_val) {
  bool partial = false;
  for (size_t i = 0; i < kNumStaticProperties; i++) {
    if (haystack[i][0] >= needle[i][1]) {
      return IntersectionType::kNone;
    }
    if (haystack[i][1] <= needle[i][0]) {
      return IntersectionType::kNone;
    }
    if (haystack[i][0] <= needle[i][0] && haystack[i][1] >= needle[i][1]) {
      continue;
    }
    partial = true;
    partial_axis = i;
    if (haystack[i][0] > needle[i][0] && haystack[i][0] < needle[i][1]) {
      partial_val = haystack[i][0] - 1;
    } else {
      JXL_DASSERT(haystack[i][1] > needle[i][0] &&
                  haystack[i][1] < needle[i][1]);
      partial_val = haystack[i][1] - 1;
    }
  }
  return partial ? IntersectionType::kPartial : IntersectionType::kInside;
}

template<bool S, bool kBatch = false>
void SplitTreeSamples(TreeSamples &tree_samples, size_t begin, size_t pos,
                      size_t end, size_t prop, uint32_t val) {
  std::array<size_t, 128> left, right;
  size_t buffered = 0;
  size_t begin_pos = begin;
  size_t end_pos = pos;
  do {
    while (begin_pos < pos &&
           tree_samples.Property<S>(prop, begin_pos) <= val) {
      ++begin_pos;
    }
    while (end_pos < end && tree_samples.Property<S>(prop, end_pos) > val) {
      ++end_pos;
    }
    if (begin_pos < pos && end_pos < end) {
      if constexpr (kBatch) {
        // Scanned positions are disjoint, so postponing these swaps cannot
        // change any later partition predicate or the resulting permutation.
        left[buffered] = begin_pos;
        right[buffered] = end_pos;
        if (++buffered == left.size()) {
          tree_samples.SwapBatch(left.data(), right.data(), buffered);
          buffered = 0;
        }
      } else {
        tree_samples.Swap(begin_pos, end_pos);
      }
    }
    ++begin_pos;
    ++end_pos;
  } while (begin_pos < pos && end_pos < end);
  if constexpr (kBatch) {
    tree_samples.SwapBatch(left.data(), right.data(), buffered);
  }
}

template <bool S, bool kCompact = false, bool kPropertyBins = false>
void CollectExtraBitsIncrease(TreeSamples &tree_samples,
                              const std::vector<ResidualToken> &rtokens,
                              std::vector<int> &count_increase,
                              std::vector<size_t> &extra_bits_increase,
                              size_t begin, size_t end, size_t prop_idx,
                              size_t max_symbols, const uint8_t* mapping = nullptr,
                              const uint32_t* property_mapping = nullptr) {
  for (size_t i2 = begin; i2 < end; i2++) {
    const ResidualToken &rt = rtokens[i2];
    size_t cnt = tree_samples.Count(i2);
    size_t p = tree_samples.Property<S>(prop_idx, i2);
    if constexpr (kPropertyBins) p = property_mapping[p];
    size_t sym = kCompact ? mapping[rt.tok] : rt.tok;
    size_t ebi = rt.nbits * cnt;
    count_increase[p * max_symbols + sym] += cnt;
    extra_bits_increase[p] += ebi;
  }
}

Status FindBestSplitImpl(TreeSamples &tree_samples, float threshold,
                   const std::vector<ModularMultiplierInfo> &mul_info,
                   StaticPropRange initial_static_prop_range,
                   float fast_decode_multiplier, Tree *tree, ThreadPool *pool,
                   size_t sample_begin, size_t sample_end,
                   std::vector<uint8_t> initial_redundant, bool subtree,
                   uint64_t profile_context) {
  using Clock = std::chrono::steady_clock;
  const char* trace_value = std::getenv("JXL_LOSSLESS_MA_TRACE");
  const bool trace = !subtree && tree_samples.LosslessSpeedups() &&
                     tree_samples.NumDistinctSamples() >= 65536 &&
                     trace_value && std::strcmp(trace_value, "1") == 0;
  LosslessProfileScope profile(profile_context,
                              subtree ? "ma.subtree" : "ma.find_best_split");
  const char* speculate_env = std::getenv("JXL_MA_SPECULATION_TRACE");
  const bool speculation_trace = profile.Enabled() && !subtree &&
      tree_samples.LosslessSpeedups() && speculate_env &&
      std::strcmp(speculate_env, "1") == 0;
  struct SpeculationStats {
    uint64_t attempts = 0, hits = 0, lead_ns = 0, hit_lead_ns = 0;
    uint64_t max_hit_lead_ns = 0;
  };
  std::array<SpeculationStats, 3> speculation;
  size_t speculation_nodes = 0;
  profile.Set("lossless_speedups", tree_samples.LosslessSpeedups());
  profile.Set("fast_entropy", tree_samples.FastEntropy());
  const char* mdl_env = std::getenv("JXL_MA_DESCRIPTION_COST");
  const float mdl_weight = tree_samples.LosslessSpeedups() && mdl_env &&
      mdl_env[0] >= '1' && mdl_env[0] <= '3' && mdl_env[1] == '\0'
      ? (mdl_env[0] == '1' ? 0.25f : mdl_env[0] == '2' ? 0.5f : 1.0f) : 0;
  profile.Set("description_weight_milli", static_cast<uint64_t>(mdl_weight * 1000));
  const auto description_cost = [&](const int32_t* counts, size_t symbols,
                                     uint32_t count) {
    if (!mdl_weight) return 0.0f;
    size_t active = 0;
    for (size_t i = 0; i < symbols; ++i) active += counts[i] != 0;
    // Finite-sample model-complexity penalty. It influences encoder learning
    // only; the resulting tree and entropy models use the standard syntax.
    return active > 1 ? mdl_weight * 0.5f * (active - 1) *
                           FastLog2f(static_cast<float>(count) + 1) : 0.0f;
  };
  const char* partition_env = std::getenv("JXL_MA_BATCH_PARTITION");
  const bool batch_partition = tree_samples.LosslessSpeedups() &&
      (!partition_env || std::strcmp(partition_env, "1") == 0);
  profile.Set("batch_partition", batch_partition);
  uint64_t batch_partition_nodes = 0;
  const char* fused_env = std::getenv("JXL_MA_FUSED_HISTOGRAM");
  const char* multi_env = std::getenv("JXL_MA_MULTI_FUSED_HISTOGRAM");
  const bool multi_fused = !multi_env || std::strcmp(multi_env, "1") == 0;
  const bool fused_histogram =
      tree_samples.LosslessSpeedups() &&
      (tree_samples.NumPredictors() == 2 ||
       (multi_fused && tree_samples.NumPredictors() > 2)) &&
      (!fused_env || std::strcmp(fused_env, "1") == 0);
  profile.Set("fused_histogram", fused_histogram);
  const char* pred_dedup_env = std::getenv("JXL_MA_PREDICTOR_DEDUP");
  const bool predictor_dedup = tree_samples.LosslessSpeedups() &&
      tree_samples.NumPredictors() > 2 && pred_dedup_env &&
      std::strcmp(pred_dedup_env, "1") == 0;
  uint64_t equivalent_predictors = 0;
  const char* padding_env = std::getenv("JXL_MA_PADDED_HISTOGRAM");
  const bool padded_histogram = tree_samples.LosslessSpeedups() && padding_env &&
      std::strcmp(padding_env, "1") == 0;
  uint64_t padded_nodes = 0;
  const auto histogram_stride = [&](size_t symbols) {
    const size_t stride = Padded(symbols);
    // Avoid mapping common low residuals from every property bucket to a small
    // subset of cache sets. Keep the stride monotonic and SIMD-aligned.
    if (padded_histogram && stride >= 32 && stride < 256 &&
        (stride & (stride - 1)) == 0) {
      ++padded_nodes;
      return stride + Lanes(di);
    }
    return stride;
  };
  uint64_t fused_pair_samples = 0;
  const char* trim_env = std::getenv("JXL_MA_TRIM_ENTROPY");
  const bool trim_entropy = tree_samples.LosslessSpeedups() &&
      !tree_samples.FastEntropy() && (!trim_env || std::strcmp(trim_env, "1") == 0);
  profile.Set("trim_entropy", trim_entropy);
  uint64_t entropy_tail_lanes_saved = 0;
  const char* sparse_env = std::getenv("JXL_MA_SPARSE_ENTROPY");
  const bool sparse_entropy = tree_samples.LosslessSpeedups() &&
      !tree_samples.FastEntropy() && sparse_env && std::strcmp(sparse_env, "1") == 0;
  profile.Set("sparse_entropy", sparse_entropy);
  const char* shards_env = std::getenv("JXL_MA_HISTOGRAM_SHARDS");
  const bool use_shards = tree_samples.LosslessSpeedups() && shards_env &&
                         std::strcmp(shards_env, "4") == 0;
  profile.Set("histogram_shards", use_shards ? 4 : 1);
  const bool measure = trace || profile.Enabled();
  const char* log_table_env = std::getenv("JXL_MA_COUNT_LOG_TABLE");
  const float* count_logs = tree_samples.LosslessSpeedups() && log_table_env &&
      std::strcmp(log_table_env, "1") == 0 ? MaCountLogTable() : nullptr;
  const auto start = measure ? Clock::now() : Clock::time_point{};
  double histogram_seconds = 0, entropy_seconds = 0;
  double partition_seconds = 0;
  const char* parent_env = std::getenv("JXL_MA_PARENT_HISTOGRAM");
  const char* reuse_parent_env = std::getenv("JXL_MA_PARENT_REUSE");
  const bool reuse_parent = tree_samples.LosslessSpeedups() && reuse_parent_env &&
      std::strcmp(reuse_parent_env, "1") == 0;
  const bool fused_parent = tree_samples.LosslessSpeedups() && (reuse_parent ||
                            (parent_env && std::strcmp(parent_env, "1") == 0));
  profile.Set("fused_parent_histogram", fused_parent);
  profile.Set("parent_reuse", reuse_parent);
  uint64_t parent_reused_nodes = 0, parent_subtracted_samples = 0;
  double leaf_histogram_seconds = 0, parent_histogram_seconds = 0;
  uint64_t parent_parallel_nodes = 0, parent_sample_visits = 0;
  uint64_t nodes_visited = 0, histogram_passes = 0, histogram_samples = 0;
  const char* compact_env = std::getenv("JXL_MA_COMPACT_HISTOGRAM");
  const bool compact_histograms = tree_samples.LosslessSpeedups() &&
      !tree_samples.FastEntropy() &&
      (!compact_env || std::strcmp(compact_env, "1") == 0);
  const char* dense_env = std::getenv("JXL_MA_DENSE_HISTOGRAM");
  const bool dense_histograms = compact_histograms && dense_env &&
      (std::strcmp(dense_env, "1") == 0 || std::strcmp(dense_env, "2") == 0);
  const size_t compact_limit = dense_histograms && dense_env[0] == '2' ? 4096 : 256;
  profile.Set("dense_histograms", dense_histograms);
  const char* bins_env = std::getenv("JXL_MA_PROPERTY_BINS");
  const bool sparse_property_bins = bins_env && std::strcmp(bins_env, "2") == 0;
  const bool compact_property_bins = tree_samples.LosslessSpeedups() && bins_env &&
      (std::strcmp(bins_env, "1") == 0 || sparse_property_bins);
  uint64_t compact_nodes = 0, compact_slots_saved = 0;
  uint64_t threshold_evaluations = 0;
  uint64_t entropy_blocks_saved = 0;
  uint64_t sharded_samples = 0;
  uint64_t cache_hits = 0, cache_saved_samples = 0;
  const char* cache_env = std::getenv("JXL_MA_HISTOGRAM_CACHE");
  const bool use_cache = tree_samples.LosslessSpeedups() &&
                         (!cache_env || std::strcmp(cache_env, "1") == 0);
  const char* paired_cache_env = std::getenv("JXL_MA_PAIRED_CACHE");
  const bool paired_cache = fused_histogram && use_cache && paired_cache_env &&
      std::strcmp(paired_cache_env, "1") == 0;
  const char* property_env = std::getenv("JXL_MA_PROPERTY_CACHE");
  const bool property_cache = use_cache &&
      (!property_env || std::strcmp(property_env, "1") == 0);
  uint64_t property_cache_hits = 0, property_saved_samples = 0;
  const char* dedup_property_env = std::getenv("JXL_MA_PROPERTY_DEDUP");
  const bool dedup_properties = tree_samples.LosslessSpeedups() &&
      dedup_property_env && std::strcmp(dedup_property_env, "1") == 0;
  profile.Set("property_dedup", dedup_properties);
  uint64_t duplicate_properties = 0, duplicate_histogram_samples = 0;
  const char* race_env = std::getenv("JXL_MA_STATISTICAL_RACE");
  const bool statistical_race = tree_samples.LosslessSpeedups() && race_env &&
      std::strcmp(race_env, "1") == 0;
  uint64_t race_nodes = 0, race_draws = 0, race_pruned = 0;
  double race_seconds = 0;
  uint64_t constant_properties = 0, inherited_properties = 0;
  MaHistogramBudget cache_budget;
  const double* entropy_table = tree_samples.FastEntropy() ? CountEntropyTable()
                                                          : nullptr;
  const char* clipped_env = std::getenv("JXL_MA_CLIPPED_ENTROPY");
  const bool clipped_entropy = tree_samples.LosslessSpeedups() &&
      !tree_samples.FastEntropy() && clipped_env && clipped_env[0] >= '1' &&
      clipped_env[0] <= '5' && clipped_env[1] == '\0';
  const double* clipped_table = clipped_entropy ? CountEntropyTable() : nullptr;
  profile.Set("clipped_entropy", clipped_entropy);
  const char* refinement_env = std::getenv("JXL_MA_REFINE_QUANTILES");
  const bool refine_quantiles = tree_samples.LosslessSpeedups() &&
      !tree_samples.FastEntropy() && !clipped_entropy && refinement_env &&
      std::strcmp(refinement_env, "1") == 0;
  const char* quantile_env = std::getenv("JXL_MA_QUANTILE_SPLITS");
  const size_t quantile_splits = refine_quantiles ? 32 : tree_samples.LosslessSpeedups() &&
      !tree_samples.FastEntropy() && !clipped_entropy && quantile_env &&
      (quantile_env[0] == '1' || quantile_env[0] == '2') && quantile_env[1] == '\0'
      ? (quantile_env[0] == '1' ? 64 : 32) : 0;
  profile.Set("quantile_splits", quantile_splits);
  profile.Set("refine_quantiles", refine_quantiles);
  struct NodeInfo {
    size_t pos;
    size_t begin;
    size_t end;
    StaticPropRange static_prop_range;
    std::unique_ptr<MaHistogramCache> histograms;
    size_t removed_begin = 0;
    size_t removed_end = 0;
    std::vector<uint8_t> redundant_properties;
    std::vector<MaParentHistogram> parents;
  };
  std::vector<NodeInfo> nodes;
  nodes.push_back(NodeInfo{0, sample_begin, sample_end,
                           initial_static_prop_range});
  nodes.back().redundant_properties = std::move(initial_redundant);
  const char* subtrees_env = std::getenv("JXL_MA_SUBTREE_PARALLEL");
  const char* small_subtrees_env = std::getenv("JXL_MA_SMALL_SUBTREES");
  const size_t subtree_minimum = small_subtrees_env &&
      std::strcmp(small_subtrees_env, "1") == 0 ? 8192 : 65536;
  const bool parallel_subtrees = !subtree && pool &&
      tree_samples.LosslessSpeedups() && sample_end - sample_begin >= subtree_minimum &&
      subtrees_env && std::strcmp(subtrees_env, "1") == 0;
  const char* balanced_env = std::getenv("JXL_MA_SUBTREE_BALANCED");
  const bool balanced_subtrees = parallel_subtrees && balanced_env &&
      std::strcmp(balanced_env, "1") == 0;
  struct SubtreeJob {
    size_t destination, begin, end;
    StaticPropRange range;
    std::vector<uint8_t> redundant;
    Tree tree;
  };
  std::vector<SubtreeJob> subtrees;

  size_t num_predictors = tree_samples.NumPredictors();
  size_t num_properties = tree_samples.NumProperties();
  std::vector<int32_t> parent_counts(tree_samples.LosslessSpeedups() ? Padded(256)
                                                                   : 0);
  std::vector<MaParentHistogram> parent_histograms(fused_parent ? num_predictors
                                                               : 0);

  struct SplitInfo {
    size_t prop = 0;
    uint32_t val = 0;
    size_t pos = 0;
    float lcost = std::numeric_limits<float>::max();
    float rcost = std::numeric_limits<float>::max();
    Predictor lpred = Predictor::Zero;
    Predictor rpred = Predictor::Zero;
    float Cost() const { return lcost + rcost; }
  };
  struct CostInfo {
    float cost = std::numeric_limits<float>::max();
    float extra_cost = 0;
    Predictor pred = Predictor::Zero;
    float Cost() const { return cost + extra_cost; }
  };
  struct PropertyWorkspace {
    std::vector<int> used_count, count_increase, paired_counts;
    std::vector<uint32_t> weight;
    std::vector<uint32_t> property_bins;
    std::vector<size_t> extra_bits_increase, paired_extra;
    std::vector<CostInfo> costs_l, costs_r;
    std::vector<int32_t> counts_above, counts_below;
    MaHistogramShards histogram_shards;
    double histogram_seconds = 0, entropy_seconds = 0;
    uint64_t histogram_passes = 0, histogram_samples = 0;
    uint64_t threshold_evaluations = 0, entropy_blocks_saved = 0;
    uint64_t sharded_samples = 0, cache_hits = 0, cache_saved_samples = 0;
    uint64_t fused_pair_samples = 0, entropy_tail_lanes_saved = 0;
    uint64_t duplicate_properties = 0, duplicate_histogram_samples = 0;
    uint64_t constant_properties = 0, inherited_properties = 0;
    uint64_t property_cache_hits = 0, property_saved_samples = 0;
    uint64_t paired_cache_samples = 0;
    uint64_t property_bins_compacted = 0, property_slots_saved = 0;
  };
  // Scratch is owned by a worker, results by a property. Cache storage is
  // allocated before dispatch and each task only writes its property's slice.
  std::vector<PropertyWorkspace> workspaces(1);
  std::vector<std::array<SplitInfo, 4>> property_splits(num_properties);
  size_t worker_slots = 1;
  bool pool_initialized = false;
  uint64_t parallel_nodes = 0, parallel_properties = 0;
  double parallel_seconds = 0;
  const auto init_workers = [&](size_t threads) -> Status {
    worker_slots = threads;
    // A custom runner may report fewer workers for the predictor task range.
    // Keep counters belonging to workers used by earlier Property dispatches.
    if (workspaces.size() < threads) workspaces.resize(threads);
    pool_initialized = true;
    return true;
  };

  while (!nodes.empty()) {
    if (profile.Enabled()) ++nodes_visited;
    size_t pos = nodes.back().pos;
    size_t begin = nodes.back().begin;
    size_t end = nodes.back().end;

    StaticPropRange static_prop_range = nodes.back().static_prop_range;
    auto cache = std::move(nodes.back().histograms);
    const size_t removed_begin = nodes.back().removed_begin;
    const size_t removed_end = nodes.back().removed_end;
    auto redundant_properties = std::move(nodes.back().redundant_properties);
    auto known_histograms = std::move(nodes.back().parents);
    nodes.pop_back();
    if (begin == end) continue;
    const size_t subtree_grain = balanced_subtrees && pool_initialized
        ? std::max<size_t>(1024, std::min<size_t>(65536,
            (sample_end - sample_begin) / (4 * worker_slots))) : 65536;
    if (parallel_subtrees && pool_initialized && worker_slots > 1 &&
        end - begin <= subtree_grain) {
      // A parent cache can still reference samples in the sibling's range.
      // Discard it before either range is mutated by independent workers.
      subtrees.push_back(SubtreeJob{pos, begin, end, static_prop_range,
                                    std::move(redundant_properties),
                                    Tree{(*tree)[pos]}});
      continue;
    }
    const bool known_parent = !known_histograms.empty();
    if (known_parent) {
      parent_histograms.swap(known_histograms);
      ++parent_reused_nodes;
    }

    SplitInfo best_split_static_constant;
    SplitInfo best_split_static;
    SplitInfo best_split_nonstatic;
    SplitInfo best_split_nowp;
    std::vector<uint64_t> property_finished;

    JXL_DASSERT(begin <= end);
    JXL_DASSERT(end <= tree_samples.NumDistinctSamples());

    float base_bits = 0;
    const size_t leaf_pred = tree_samples.PredictorIndex((*tree)[pos].predictor);
    if (tree_samples.LosslessSpeedups()) {
      const auto before = measure ? Clock::now() : Clock::time_point{};
      // Leaf termination only needs the current predictor. Do not build all
      // predictor histograms for a node that cannot be split anyway.
      size_t parent_symbols = 0;
      size_t parent_extra_bits = 0;
      uint32_t parent_count = 0;
      const int32_t* leaf_counts = parent_counts.data();
      if (fused_parent) {
        auto& h = parent_histograms[leaf_pred];
        if (!known_parent) CollectMaParentHistogram(tree_samples, leaf_pred, begin, end, &h);
        leaf_counts = h.counts.data();
        parent_symbols = h.symbols;
        parent_extra_bits = h.extra_bits;
        parent_count = h.weight;
      } else {
        std::fill(parent_counts.begin(), parent_counts.end(), 0);
        const auto& tokens = tree_samples.RTokens(leaf_pred);
        for (size_t i = begin; i < end; ++i) {
          const auto& token = tokens[i];
          const size_t count = tree_samples.Count(i);
          parent_counts[token.tok] += count;
          parent_count += count;
          parent_symbols = std::max(parent_symbols, size_t{token.tok} + 1);
          parent_extra_bits += token.nbits * count;
        }
      }
      base_bits = entropy_table
                      ? std::max(0.0, CountEntropy(parent_count, entropy_table) -
                                          HistogramCountEntropy(
                                              leaf_counts, parent_symbols,
                                              entropy_table))
                      : EstimateBits(leaf_counts, Padded(parent_symbols));
      base_bits += static_cast<uint32_t>(parent_extra_bits);
      base_bits += description_cost(leaf_counts, parent_symbols, parent_count);
      if (measure) leaf_histogram_seconds +=
          std::chrono::duration<double>(Clock::now() - before).count();
      if (base_bits <= threshold && mul_info.empty()) continue;
    }

    const auto before_parent = measure ? Clock::now() : Clock::time_point{};
    size_t max_symbols = 0;
    std::vector<int32_t> counts;
    std::vector<uint32_t> tot_extra_bits(num_predictors);
    uint32_t total_count = 0;
    if (fused_parent) {
      // Tokens are bytes. Build fixed-width histograms in one pass, reuse the
      // leaf predictor, then compact once to the existing shared cache stride.
      const auto collect_parent = [&](uint32_t pred, size_t) -> Status {
        if (!known_parent && pred != leaf_pred) {
          CollectMaParentHistogram(tree_samples, pred, begin, end,
                                    &parent_histograms[pred]);
        }
        return true;
      };
      const bool parallel = !known_parent && pool && num_predictors > 2 && end - begin >= 8192 &&
          static_cast<uint64_t>(end - begin) * (num_predictors - 1) >= 65536 &&
          (!pool_initialized || worker_slots > 1);
      if (parallel) {
        JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, num_predictors, init_workers,
                                      collect_parent, "MaParentHistograms"));
        if (worker_slots > 1) ++parent_parallel_nodes;
      } else {
        for (size_t pred = 0; pred < num_predictors; ++pred) {
          JXL_RETURN_IF_ERROR(collect_parent(pred, 0));
        }
      }
      for (const auto& h : parent_histograms) {
        max_symbols = std::max(max_symbols, h.symbols);
      }
      max_symbols = histogram_stride(max_symbols);
      counts.resize(num_predictors * max_symbols);
      for (size_t pred = 0; pred < num_predictors; ++pred) {
        const auto& h = parent_histograms[pred];
        std::copy_n(h.counts.data(), h.symbols,
                    counts.data() + pred * max_symbols);
        tot_extra_bits[pred] = h.extra_bits;
      }
      total_count = parent_histograms[leaf_pred].weight;
      if (!known_parent) parent_sample_visits +=
          static_cast<uint64_t>(end - begin) * (num_predictors - 1);
    } else {
      // Compute the maximum token in the range.
      for (size_t pred = 0; pred < num_predictors; pred++) {
        for (size_t i = begin; i < end; i++) {
          uint32_t tok = tree_samples.Token(pred, i);
          max_symbols = max_symbols > tok + 1 ? max_symbols : tok + 1;
        }
      }
      max_symbols = histogram_stride(max_symbols);
      counts.resize(max_symbols * num_predictors);
      for (size_t pred = 0; pred < num_predictors; pred++) {
        size_t extra_bits = 0;
        const std::vector<ResidualToken>& rtokens = tree_samples.RTokens(pred);
        for (size_t i = begin; i < end; i++) {
          const ResidualToken& rt = rtokens[i];
          size_t count = tree_samples.Count(i);
          if (pred == 0 && tree_samples.LosslessSpeedups()) total_count += count;
          size_t eb = rt.nbits * count;
          counts[pred * max_symbols + rt.tok] += count;
          extra_bits += eb;
        }
        tot_extra_bits[pred] = extra_bits;
      }
      parent_sample_visits +=
          static_cast<uint64_t>(end - begin) * num_predictors * 2;
    }
    if (measure) parent_histogram_seconds +=
        std::chrono::duration<double>(Clock::now() - before_parent).count();
    const bool incremental_node = clipped_entropy &&
        (clipped_env[0] == '1' || clipped_env[0] == '4' ||
         ((clipped_env[0] == '2' || clipped_env[0] == '5') && end - begin >= 2048) ||
         (clipped_env[0] == '3' && end - begin >= 16384));
    std::array<uint8_t, 256> symbol_mapping;
    const size_t original_symbols = max_symbols;
    const bool compact = compact_histograms && !cache && end - begin <= compact_limit &&
        max_symbols >= 32 && CompactMaHistograms(num_predictors, Lanes(di),
            &counts, &max_symbols, &symbol_mapping, dense_histograms);
    if (compact) {
      ++compact_nodes;
      compact_slots_saved += (original_symbols - max_symbols) * num_predictors;
    }
    std::vector<size_t> predictor_symbols;
    if (trim_entropy) {
      predictor_symbols.resize(num_predictors);
      for (size_t pred = 0; pred < num_predictors; ++pred) {
        // Every child histogram is a subset of this parent. Its zero tail
        // cannot contribute, but the shared storage/cache stride stays intact.
        predictor_symbols[pred] = Padded(MaHistogramSymbolCount(
            counts.data() + pred * max_symbols, max_symbols));
      }
    }
    std::vector<double> count_entropies;
    std::vector<std::vector<size_t>> active_blocks(sparse_entropy ? num_predictors : 0);
    if (sparse_entropy) {
      for (size_t pred = 0; pred < num_predictors; ++pred) {
        active_blocks[pred] = MaActiveHistogramBlocks(
            counts.data() + pred * max_symbols, max_symbols, Lanes(di));
      }
    }
    if (entropy_table) {
      count_entropies.resize(num_predictors);
      for (size_t pred = 0; pred < num_predictors; ++pred) {
        count_entropies[pred] = HistogramCountEntropy(
            counts.data() + pred * max_symbols, max_symbols, entropy_table);
      }
    }

    if (!tree_samples.LosslessSpeedups()) {
      size_t pred = tree_samples.PredictorIndex((*tree)[pos].predictor);
      base_bits =
          EstimateBits(counts.data() + pred * max_symbols, max_symbols) +
          tot_extra_bits[pred];
    }

    SplitInfo *best = &best_split_nonstatic;

    SplitInfo forced_split;
    // The multiplier ranges cut halfway through the current ranges of static
    // properties. We do this even if the current node is not a leaf, to
    // minimize the number of nodes in the resulting tree.
    for (const auto &mmi : mul_info) {
      uint32_t axis;
      uint32_t val;
      IntersectionType t =
          BoxIntersects(static_prop_range, mmi.range, axis, val);
      if (t == IntersectionType::kNone) continue;
      if (t == IntersectionType::kInside) {
        (*tree)[pos].multiplier = mmi.multiplier;
        break;
      }
      if (t == IntersectionType::kPartial) {
        JXL_DASSERT(axis < kNumStaticProperties);
        forced_split.val = tree_samples.QuantizeStaticProperty(axis, val);
        forced_split.prop = axis;
        forced_split.lcost = forced_split.rcost = base_bits / 2 - threshold;
        forced_split.lpred = forced_split.rpred = (*tree)[pos].predictor;
        best = &forced_split;
        best->pos = begin;
        JXL_DASSERT(best->prop == tree_samples.PropertyFromIndex(best->prop));
        if (best->prop < tree_samples.NumStaticProps()) {
        for (size_t x = begin; x < end; x++) {
          if (tree_samples.Property<true>(best->prop, x) <= best->val) {
            best->pos++;
          }
        }
      } else {
        size_t prop = best->prop - tree_samples.NumStaticProps();
        for (size_t x = begin; x < end; x++) {
          if (tree_samples.Property<false>(prop, x) <= best->val) {
            best->pos++;
          }
        }
      }
        break;
      }
    }

    if (best != &forced_split) {
      // Small nodes cannot amortize a dense histogram copy. Pending sibling
      // caches share a bounded per-tree budget, independent of tree depth.
      if (use_cache && !cache && end - begin >= 8192) {
        cache = MaHistogramCache::Create(tree_samples, max_symbols, &cache_budget,
                                         property_cache);
      }
      // For each property, compute which of its values are used, and what
      // tokens correspond to those usages. Then, iterate through the values,
      // and compute the entropy of each side of the split (of the form `prop >
      // threshold`). Finally, find the split that minimizes the cost.
      // The lower the threshold, the higher the expected noisiness of the
      // estimate. Thus, discourage changing predictors.
      float change_pred_penalty = 800.0f / (100.0f + threshold);
      std::vector<size_t> representative(num_predictors);
      std::iota(representative.begin(), representative.end(), 0);
      std::vector<float> predictor_penalty(num_predictors);
      for (size_t pred = 0; pred < num_predictors; ++pred) {
        const Predictor predictor = tree_samples.PredictorFromIndex(pred);
        float penalty = 0;
        if (predictor != (*tree)[pos].predictor &&
            (*tree)[pos].predictor != Predictor::Weighted) {
          penalty = change_pred_penalty;
        }
        if (predictor == Predictor::Weighted) penalty += 1e-8;
        if (predictor == Predictor::Zero) penalty -= 1e-8;
        predictor_penalty[pred] = penalty;
        if (!predictor_dedup) continue;
        for (size_t previous = 0; previous < pred; ++previous) {
          if (representative[previous] != previous ||
              predictor_penalty[previous] != penalty ||
              tot_extra_bits[previous] != tot_extra_bits[pred] ||
              std::memcmp(counts.data() + previous * max_symbols,
                          counts.data() + pred * max_symbols,
                          max_symbols * sizeof(counts[0])) != 0) continue;
          const auto* a = tree_samples.RTokens(previous).data() + begin;
          const auto* b = tree_samples.RTokens(pred).data() + begin;
          if (std::memcmp(a, b, (end - begin) * sizeof(*a)) != 0) continue;
          representative[pred] = previous;
          ++equivalent_predictors;
          break;
        }
      }
      std::vector<uint8_t> race_keep;
      bool inherited_histograms = false;
      if (statistical_race && cache) {
        for (size_t p = 0; p < num_properties; ++p) {
          inherited_histograms |= cache->Has(p);
        }
      }
      if (statistical_race && !inherited_histograms &&
          base_bits > threshold && end - begin >= 262144 &&
          num_properties > tree_samples.NumStaticProps() + 4 &&
          MaPropertyRace::Fits(tree_samples, max_symbols)) {
        race_keep.assign(num_properties, 1);
        const auto before = measure ? Clock::now() : Clock::time_point{};
        ++race_nodes;
        MaPropertyRace race(tree_samples, begin, end, max_symbols);
        size_t previous_draws = 0;
        for (size_t draws : {size_t{64}, size_t{256}}) {
          std::vector<size_t> properties;
          for (size_t p = tree_samples.NumStaticProps(); p < num_properties; ++p) {
            if (race_keep[p]) properties.push_back(p);
          }
          if (properties.size() <= 4) break;
          const auto run = [&](uint32_t i, size_t) -> Status {
            race.Extend(properties[i], draws);
            return true;
          };
          JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, properties.size(), init_workers,
                                        run, "MaPropertyRace"));
          race_draws += properties.size() * MaPropertyRace::kPanels *
                        (draws - previous_draws);
          previous_draws = draws;
          const auto next_keep = race.Survivors(race_keep);
          // No useful separation at the first budget: spend remaining work on
          // the original complete split search, not another proposal round.
          if (next_keep == race_keep) break;
          race_keep = next_keep;
        }
        for (size_t p = 0; p < num_properties; ++p) {
          if (!race_keep[p]) {
            ++race_pruned;
            // A skipped cached property still describes the parent range.
            // Never pass that stale table down as the current node's cache.
            if (cache) cache->Invalidate(p);
          }
        }
        if (measure) race_seconds +=
            std::chrono::duration<double>(Clock::now() - before).count();
      }
      struct SeenProperty {
        uint64_t signature;
        size_t property;
      };
      std::vector<SeenProperty> seen_properties;
      const bool parallel = pool && tree_samples.LosslessSpeedups() &&
          end - begin >= 1024 &&
          (end - begin) * num_predictors * num_properties >= 65536 &&
          (!pool_initialized || worker_slots > 1);
      if (speculation_trace && parallel && end - begin >= 65536) {
        property_finished.resize(num_properties);
      }
      if (dedup_properties && redundant_properties.empty()) {
        redundant_properties.resize(num_properties);
      }
      std::fill(property_splits.begin(), property_splits.end(),
                std::array<SplitInfo, 4>{});
      const auto evaluate_property = [&](uint32_t prop, size_t thread) -> Status {
        if (!race_keep.empty() && !race_keep[prop]) return true;
        struct CompletionStamp {
          uint64_t* destination;
          ~CompletionStamp() {
            if (destination) {
              *destination = std::chrono::duration_cast<std::chrono::nanoseconds>(
                  Clock::now().time_since_epoch()).count();
            }
          }
        } stamp{property_finished.empty() ? nullptr : &property_finished[prop]};
        auto& w = workspaces[thread];
        auto& prop_value_used_count = w.used_count;
        auto& prop_value_weight = w.weight;
        auto& count_increase = w.count_increase;
        auto& extra_bits_increase = w.extra_bits_increase;
        auto& paired_counts = w.paired_counts;
        auto& paired_extra = w.paired_extra;
        auto& costs_l = w.costs_l;
        auto& costs_r = w.costs_r;
        auto& counts_above = w.counts_above;
        auto& counts_below = w.counts_below;
        counts_above.resize(max_symbols);
        counts_below.resize(max_symbols);
        if (!redundant_properties.empty() && redundant_properties[prop]) {
          if (profile.Enabled()) ++w.inherited_properties;
          return true;
        }
        costs_l.clear();
        costs_r.clear();
        size_t prop_size = tree_samples.NumPropertyValues(prop);
        // A workspace outlives the node; sibling nodes can have different
        // symbol strides, so reset all bins rather than reuse a stale tail.
        // Clear prop_value_used_count (which cannot be cleared "on the go")
        prop_value_used_count.clear();
        prop_value_used_count.resize(prop_size);
        if (tree_samples.LosslessSpeedups()) prop_value_weight.assign(prop_size, 0);

        size_t first_used = prop_size;
        size_t last_used = 0;

        const bool cached_property = cache && cache->Has(prop);
        // Both the distinct-sample count (partition boundary) and the weighted
        // count (entropy) must follow the same sibling subtraction as tokens.
        if (property_cache && cached_property) {
          if (prop < tree_samples.NumStaticProps()) {
            cache->SubtractProperty<true>(tree_samples, prop, removed_begin,
                removed_end, prop_value_used_count.data(), prop_value_weight.data());
          } else {
            cache->SubtractProperty<false>(tree_samples, prop, removed_begin,
                removed_end, prop_value_used_count.data(), prop_value_weight.data());
          }
          for (size_t p = 0; p < prop_size; ++p) {
            if (prop_value_used_count[p] != 0) {
              first_used = std::min(first_used, p);
              last_used = p;
            }
          }
          if (profile.Enabled()) {
            ++w.property_cache_hits;
            w.property_saved_samples += end - begin - (removed_end - removed_begin);
          }
        } else if (prop < tree_samples.NumStaticProps()) {
          for (size_t i = begin; i < end; i++) {
            size_t p = tree_samples.Property<true>(prop, i);
            prop_value_used_count[p]++;
            if (tree_samples.LosslessSpeedups()) {
              prop_value_weight[p] += tree_samples.Count(i);
            }
            last_used = std::max(last_used, p);
            first_used = std::min(first_used, p);
          }
        } else {
          size_t prop_idx = prop - tree_samples.NumStaticProps();
          for (size_t i = begin; i < end; i++) {
            size_t p = tree_samples.Property<false>(prop_idx, i);
            prop_value_used_count[p]++;
            if (tree_samples.LosslessSpeedups()) {
              prop_value_weight[p] += tree_samples.Count(i);
            }
            last_used = std::max(last_used, p);
            first_used = std::min(first_used, p);
          }
        }
        // A constant property cannot split this node. Its histograms would
        // only be populated and cleared, once per predictor, with no result.
        if (tree_samples.LosslessSpeedups() && first_used == last_used) {
          if (dedup_properties) {
            redundant_properties[prop] = 1;
            if (profile.Enabled()) ++w.constant_properties;
          }
          return true;
        }
        // Cross-property dedup requires preceding results. Parallel tasks
        // evaluate those equivalent candidates too, then reduce in prop order.
        if (dedup_properties && !parallel && end - begin >= 64) {
          const bool is_static = prop < tree_samples.NumStaticProps();
          const bool is_wp = tree_samples.PropertyFromIndex(prop) == kWPProp;
          uint64_t signature = 14695981039346656037ULL;
          for (size_t value = first_used; value <= last_used; ++value) {
            if (prop_value_used_count[value] == 0) continue;
            signature = (signature ^ value) * 1099511628211ULL;
            signature = (signature ^ prop_value_used_count[value]) * 1099511628211ULL;
          }
          bool duplicate = false;
          for (const auto& seen : seen_properties) {
            if (seen.signature != signature ||
                (seen.property < tree_samples.NumStaticProps()) != is_static ||
                (tree_samples.PropertyFromIndex(seen.property) == kWPProp) != is_wp) {
              continue;
            }
            // Marginals only filter comparisons. Verify every sample, since
            // identical counts alone say nothing about the residual partition.
            duplicate = is_static
                ? MaSamePropertyPartition<true>(tree_samples, prop, seen.property,
                                                 begin, end)
                : MaSamePropertyPartition<false>(tree_samples,
                      prop - tree_samples.NumStaticProps(),
                      seen.property - tree_samples.NumStaticProps(), begin, end);
            if (duplicate) break;
          }
          if (duplicate) {
            // All predictor/threshold costs were already evaluated. Strict
            // improvement and matching static/WP classes preserve tie choices.
            if (cache) cache->Invalidate(prop);
            redundant_properties[prop] = 1;
            if (profile.Enabled()) {
              ++w.duplicate_properties;
              w.duplicate_histogram_samples += num_predictors *
                  (cached_property ? removed_end - removed_begin : end - begin);
            }
            return true;
          }
          seen_properties.push_back({signature, prop});
        }
        if (property_cache && cache && !cached_property) {
          cache->CaptureProperty(prop, prop_value_used_count.data(),
                                  prop_value_weight.data());
        }
        const size_t used_bins = compact_property_bins && !cache &&
                (sparse_property_bins || end - begin < 256 || compact)
            ? std::count_if(prop_value_used_count.begin(), prop_value_used_count.end(),
                              [](int count) { return count != 0; })
            : prop_size;
        const bool packed_bins = compact_property_bins && !cache &&
            (sparse_property_bins || end - begin < 256 || compact) &&
            2 * used_bins < prop_size;
        if (packed_bins) {
          w.property_bins.resize(prop_size);
          size_t bin = 0;
          for (size_t i = 0; i < prop_size; ++i)
            if (prop_value_used_count[i]) w.property_bins[i] = bin++;
          ++w.property_bins_compacted;
          w.property_slots_saved += (prop_size - used_bins) * max_symbols * num_predictors;
        }
        count_increase.assign((packed_bins ? used_bins : prop_size) * max_symbols, 0);
        extra_bits_increase.assign(packed_bins ? used_bins : prop_size, 0);
        costs_l.resize(last_used - first_used);
        costs_r.resize(last_used - first_used);
        std::vector<uint8_t> score_split;
        if (quantile_splits && end - begin >= 512 &&
            prop >= tree_samples.NumStaticProps() &&
            last_used - first_used > quantile_splits) {
          // Weighted quantile candidates, retaining both edge thresholds and
          // every large jump. Histograms still include all training samples.
          // This changes the learned tree, not the lossless reconstruction.
          score_split.assign(last_used - first_used, 0);
          uint64_t pending_weight = 0;
          const uint64_t quantum = std::max<uint64_t>(1, total_count / quantile_splits);
          size_t last_nonempty = first_used;
          for (size_t i = first_used; i < last_used; ++i) {
            if (!prop_value_used_count[i]) continue;
            last_nonempty = i;
            pending_weight += prop_value_weight[i];
            if (i == first_used || pending_weight >= quantum) {
              score_split[i - first_used] = 1;
              pending_weight = 0;
            }
          }
          score_split[last_nonempty - first_used] = 1;
        }
        const bool refine_split = refine_quantiles && !score_split.empty() &&
            uint64_t{count_increase.size()} * num_predictors * sizeof(int) <= (8u << 20);
        if (refine_quantiles && !refine_split) score_split.clear();
        std::vector<std::vector<int>> saved_counts(refine_split ? num_predictors : 0);
        std::vector<std::vector<size_t>> saved_extra(refine_split ? num_predictors : 0);
        for (int pass = 0; pass < (refine_split ? 2 : 1); ++pass) {
          if (pass) {
            std::vector<float> coarse_costs(score_split.size());
            for (size_t i = 0; i < coarse_costs.size(); ++i)
              coarse_costs[i] = costs_l[i].Cost() + costs_r[i].Cost();
            score_split = RefineMaQuantileMask(score_split, coarse_costs);
          }
          // For all predictors, compute the right and left costs of each split.
          for (size_t pred = 0; pred < num_predictors; pred++) {
            if (representative[pred] != pred) {
              if (cache) cache->CopyPredictor(prop, representative[pred], pred);
              continue;
            }
            const size_t pair_start = pred & ~size_t{1};
            const bool pair_available =
                pair_start + 1 < num_predictors &&
                representative[pair_start] == pair_start &&
                representative[pair_start + 1] == pair_start + 1;
            const size_t scoring_symbols =
                trim_entropy ? predictor_symbols[pred] : max_symbols;
            // Compute cost and histogram increments for each property value.
            const auto before_histogram =
                measure ? Clock::now() : Clock::time_point{};
            if (profile.Enabled() && !pass) {
              ++w.histogram_passes;
              w.histogram_samples +=
                  cached_property ? removed_end - removed_begin : end - begin;
              if (cached_property) {
                ++w.cache_hits;
                w.cache_saved_samples +=
                    end - begin - (removed_end - removed_begin);
              }
            }
            const std::vector<ResidualToken>& rtokens =
                tree_samples.RTokens(pred);
            if (pass) {
              count_increase = std::move(saved_counts[pred]);
              extra_bits_increase = std::move(saved_extra[pred]);
            } else if (packed_bins) {
              const auto collect = [&](auto is_static, auto is_compact) {
                CollectExtraBitsIncrease<decltype(is_static)::value,
                                         decltype(is_compact)::value, true>(
                    tree_samples, rtokens, count_increase, extra_bits_increase,
                    begin, end,
                    is_static ? prop : prop - tree_samples.NumStaticProps(),
                    max_symbols, symbol_mapping.data(), w.property_bins.data());
              };
              if (prop < tree_samples.NumStaticProps()) {
                if (compact)
                  collect(std::true_type{}, std::true_type{});
                else
                  collect(std::true_type{}, std::false_type{});
              } else {
                if (compact)
                  collect(std::false_type{}, std::true_type{});
                else
                  collect(std::false_type{}, std::false_type{});
              }
            } else if (compact && prop < tree_samples.NumStaticProps()) {
              CollectExtraBitsIncrease<true, true>(
                  tree_samples, rtokens, count_increase, extra_bits_increase,
                  begin, end, prop, max_symbols, symbol_mapping.data());
            } else if (compact) {
              CollectExtraBitsIncrease<false, true>(
                  tree_samples, rtokens, count_increase, extra_bits_increase,
                  begin, end, prop - tree_samples.NumStaticProps(), max_symbols,
                  symbol_mapping.data());
            } else if (cached_property && paired_cache && end - begin >= 256 &&
                       pair_available) {
              if (pred % 2 == 0) {
                paired_counts.resize(prop_size * max_symbols);
                paired_extra.resize(prop_size);
                if (prop < tree_samples.NumStaticProps()) {
                  cache->SubtractPairedAndCopy<true>(
                      tree_samples, prop, pred, removed_begin, removed_end,
                      count_increase.data(), paired_counts.data(),
                      extra_bits_increase.data(), paired_extra.data(),
                      max_symbols);
                } else {
                  cache->SubtractPairedAndCopy<false>(
                      tree_samples, prop, pred, removed_begin, removed_end,
                      count_increase.data(), paired_counts.data(),
                      extra_bits_increase.data(), paired_extra.data(),
                      max_symbols);
                }
                if (profile.Enabled()) {
                  w.paired_cache_samples += removed_end - removed_begin;
                }
              } else {
                count_increase.swap(paired_counts);
                extra_bits_increase.swap(paired_extra);
              }
            } else if (cached_property &&
                       prop < tree_samples.NumStaticProps()) {
              cache->SubtractAndCopy<true>(
                  tree_samples, prop, pred, removed_begin, removed_end,
                  count_increase.data(), extra_bits_increase.data(),
                  max_symbols);
            } else if (cached_property) {
              cache->SubtractAndCopy<false>(
                  tree_samples, prop, pred, removed_begin, removed_end,
                  count_increase.data(), extra_bits_increase.data(),
                  max_symbols);
            } else if (fused_histogram && end - begin >= 256 &&
                       pair_available) {
              if (pred % 2 == 0) {
                paired_counts.assign(prop_size * max_symbols, 0);
                paired_extra.assign(prop_size, 0);
                if (prop < tree_samples.NumStaticProps()) {
                  CollectPairedMaHistogram<true>(
                      tree_samples, prop, begin, end, max_symbols,
                      count_increase.data(), paired_counts.data(),
                      extra_bits_increase.data(), paired_extra.data(), pred);
                } else {
                  CollectPairedMaHistogram<false>(
                      tree_samples, prop - tree_samples.NumStaticProps(), begin,
                      end, max_symbols, count_increase.data(),
                      paired_counts.data(), extra_bits_increase.data(),
                      paired_extra.data(), pred);
                }
                if (profile.Enabled()) w.fused_pair_samples += end - begin;
              } else {
                count_increase.swap(paired_counts);
                extra_bits_increase.swap(paired_extra);
              }
            } else if (use_shards && end - begin >= 4096 &&
                       end - begin >= 4 * prop_size * max_symbols) {
              if (prop < tree_samples.NumStaticProps()) {
                CollectShardedMaHistogram<true>(
                    tree_samples, pred, prop, begin, end, max_symbols,
                    prop_size, count_increase.data(),
                    extra_bits_increase.data(), &w.histogram_shards);
              } else {
                CollectShardedMaHistogram<false>(
                    tree_samples, pred, prop - tree_samples.NumStaticProps(),
                    begin, end, max_symbols, prop_size, count_increase.data(),
                    extra_bits_increase.data(), &w.histogram_shards);
              }
              if (profile.Enabled()) w.sharded_samples += end - begin;
            } else if (prop < tree_samples.NumStaticProps()) {
              CollectExtraBitsIncrease<true>(
                  tree_samples, rtokens, count_increase, extra_bits_increase,
                  begin, end, prop, max_symbols);
            } else {
              CollectExtraBitsIncrease<false>(
                  tree_samples, rtokens, count_increase, extra_bits_increase,
                  begin, end, prop - tree_samples.NumStaticProps(),
                  max_symbols);
            }
            if (!pass && cache && !cached_property) {
              cache->Capture(prop, pred, count_increase.data(),
                             extra_bits_increase.data(), max_symbols);
            }
            if (refine_split && !pass) {
              saved_counts[pred] = count_increase;
              saved_extra[pred] = extra_bits_increase;
            }
            const auto before_entropy =
                measure ? Clock::now() : Clock::time_point{};
            if (measure) {
              w.histogram_seconds += std::chrono::duration<double>(
                                         before_entropy - before_histogram)
                                         .count();
            }
            memcpy(counts_above.data(), counts.data() + pred * max_symbols,
                   max_symbols * sizeof counts_above[0]);
            memset(counts_below.data(), 0,
                   max_symbols * sizeof counts_below[0]);
            size_t extra_bits_below = 0;
            uint32_t count_below = 0;
            double sum_above = entropy_table ? count_entropies[pred] : 0;
            double sum_below = 0;
            MaClippedEntropy clipped_above(
                counts_above.data(), incremental_node ? max_symbols : 0,
                incremental_node ? total_count : 0, clipped_table,
                kCountEntropyTableSize,
                incremental_node && clipped_env[0] >= '4');
            MaClippedEntropy clipped_below(
                nullptr, 0, incremental_node ? total_count : 0, clipped_table,
                kCountEntropyTableSize,
                incremental_node && clipped_env[0] >= '4');
            // Exclude last used: this ensures neither counts_above nor
            // counts_below is empty.
            for (size_t i = first_used; i < last_used; i++) {
              if (!prop_value_used_count[i]) continue;
              const size_t bin = packed_bins ? w.property_bins[i] : i;
              if (profile.Enabled()) ++w.threshold_evaluations;
              if (profile.Enabled()) {
                w.entropy_tail_lanes_saved += max_symbols - scoring_symbols;
              }
              extra_bits_below += extra_bits_increase[bin];
              // The increase for this property value has been used, and will
              // not be used again: clear it. Also below.
              extra_bits_increase[bin] = 0;
              float rbits, lbits;
              if (tree_samples.LosslessSpeedups()) {
                count_below += prop_value_weight[i];
                if (!score_split.empty() && !score_split[i - first_used]) {
                  int32_t* increase = count_increase.data() + bin * max_symbols;
                  for (size_t s = 0; s < max_symbols; s += Lanes(di)) {
                    const auto delta = LoadU(di, increase + s);
                    StoreU(Sub(LoadU(di, counts_above.data() + s), delta), di,
                           counts_above.data() + s);
                    StoreU(Add(LoadU(di, counts_below.data() + s), delta), di,
                           counts_below.data() + s);
                    StoreU(Zero(di), di, increase + s);
                  }
                  continue;
                }
                if (incremental_node) {
                  int32_t* increase = count_increase.data() + bin * max_symbols;
                  for (size_t symbol = 0; symbol < max_symbols; ++symbol) {
                    const int32_t delta = increase[symbol];
                    if (!delta) continue;
                    const int32_t r = counts_above[symbol],
                                  l = counts_below[symbol];
                    clipped_above.Replace(r, r - delta);
                    clipped_below.Replace(l, l + delta);
                    counts_above[symbol] = r - delta;
                    counts_below[symbol] = l + delta;
                    increase[symbol] = 0;
                  }
                  rbits = clipped_above.Bits(total_count - count_below);
                  lbits = clipped_below.Bits(count_below);
                } else if (entropy_table) {
                  MoveCountsAndShannonBits(
                      counts_above.data(), counts_below.data(),
                      count_increase.data() + bin * max_symbols, max_symbols,
                      total_count - count_below, count_below, entropy_table,
                      &sum_above, &sum_below, &rbits, &lbits);
                } else if (count_logs && count_below <= kMaLogTableLimit &&
                           total_count - count_below <= kMaLogTableLimit &&
                           total_count > kEntropyTableMaxCount) {
                  MoveCountsAndLogBits(
                      counts_above.data(), counts_below.data(),
                      count_increase.data() + bin * max_symbols,
                      scoring_symbols, total_count - count_below, count_below,
                      count_logs, &rbits, &lbits);
                } else if (sparse_entropy && active_blocks[pred].size() <
                                                 max_symbols / Lanes(di)) {
                  // Parent-zero symbols stay zero for every split. Preserve
                  // lane indices/reduction order while skipping empty blocks.
                  MoveCountsAndEstimateBits<true>(
                      counts_above.data(), counts_below.data(),
                      count_increase.data() + bin * max_symbols,
                      scoring_symbols, total_count - count_below, count_below,
                      active_blocks[pred].data(), active_blocks[pred].size(),
                      &rbits, &lbits);
                  if (profile.Enabled()) {
                    w.entropy_blocks_saved +=
                        max_symbols / Lanes(di) - active_blocks[pred].size();
                  }
                } else {
                  MoveCountsAndEstimateBits<false>(
                      counts_above.data(), counts_below.data(),
                      count_increase.data() + bin * max_symbols,
                      scoring_symbols, total_count - count_below, count_below,
                      nullptr, 0, &rbits, &lbits);
                }
              } else {
                for (size_t sym = 0; sym < max_symbols; sym++) {
                  counts_above[sym] -= count_increase[bin * max_symbols + sym];
                  counts_below[sym] += count_increase[bin * max_symbols + sym];
                  count_increase[bin * max_symbols + sym] = 0;
                }
                rbits = EstimateBits(counts_above.data(), max_symbols);
                lbits = EstimateBits(counts_below.data(), max_symbols);
              }
              float rcost = rbits + tot_extra_bits[pred] - extra_bits_below;
              float lcost = lbits + extra_bits_below;
              rcost += description_cost(counts_above.data(), scoring_symbols,
                                        total_count - count_below);
              lcost += description_cost(counts_below.data(), scoring_symbols,
                                        count_below);
              JXL_DASSERT(extra_bits_below <= tot_extra_bits[pred]);
              const float penalty = predictor_penalty[pred];
              if (rcost + penalty < costs_r[i - first_used].Cost()) {
                costs_r[i - first_used].cost = rcost;
                costs_r[i - first_used].extra_cost = penalty;
                costs_r[i - first_used].pred =
                    tree_samples.PredictorFromIndex(pred);
              }
              if (lcost + penalty < costs_l[i - first_used].Cost()) {
                costs_l[i - first_used].cost = lcost;
                costs_l[i - first_used].extra_cost = penalty;
                costs_l[i - first_used].pred =
                    tree_samples.PredictorFromIndex(pred);
              }
            }
            if (measure) {
              w.entropy_seconds +=
                  std::chrono::duration<double>(Clock::now() - before_entropy)
                      .count();
            }
            if (cache) {
              // The old scratch path never reads the last bin. Cached tables
              // must retain a complete, predictor-specific histogram instead.
              extra_bits_increase[last_used] = 0;
              std::fill_n(count_increase.data() + last_used * max_symbols,
                          max_symbols, 0);
            }
          }
        }
        // Iterate through the possible splits and find the one with minimum sum
        // of costs of the two sides.
        size_t split = begin;
        for (size_t i = first_used; i < last_used; i++) {
          if (!prop_value_used_count[i]) continue;
          split += prop_value_used_count[i];
          if (!score_split.empty() && !score_split[i - first_used]) continue;
          float rcost = costs_r[i - first_used].cost;
          float lcost = costs_l[i - first_used].cost;

          bool uses_wp = tree_samples.PropertyFromIndex(prop) == kWPProp ||
                         costs_l[i - first_used].pred == Predictor::Weighted ||
                         costs_r[i - first_used].pred == Predictor::Weighted;
          bool zero_entropy_side = rcost == 0 || lcost == 0;

          SplitInfo &best_ref =
              tree_samples.PropertyFromIndex(prop) < kNumStaticProperties
                  ? property_splits[prop][zero_entropy_side ? 0 : 1]
                  : property_splits[prop][uses_wp ? 2 : 3];
          if (lcost + rcost < best_ref.Cost()) {
            best_ref.prop = prop;
            best_ref.val = i;
            best_ref.pos = split;
            best_ref.lcost = lcost;
            best_ref.lpred = costs_l[i - first_used].pred;
            best_ref.rcost = rcost;
            best_ref.rpred = costs_r[i - first_used].pred;
          }
        }
        // Clear extra_bits_increase and cost_increase for last_used.
        const size_t last_bin = packed_bins ? w.property_bins[last_used] : last_used;
        extra_bits_increase[last_bin] = 0;
        for (size_t sym = 0; sym < max_symbols; sym++) {
          count_increase[last_bin * max_symbols + sym] = 0;
        }
        return true;
      };
      if (base_bits > threshold) {
        if (parallel) {
          const auto before = measure ? Clock::now() : Clock::time_point{};
          JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, num_properties, init_workers,
                                        evaluate_property, "MaSplitProperties"));
          if (worker_slots > 1) {
            ++parallel_nodes;
            parallel_properties += num_properties;
          }
          if (measure) parallel_seconds +=
              std::chrono::duration<double>(Clock::now() - before).count();
        } else {
          for (size_t prop = 0; prop < num_properties; ++prop) {
            JXL_RETURN_IF_ERROR(evaluate_property(prop, 0));
          }
        }
      }
      // Preserve strict improvement and stable ties independently of task
      // completion order. Partitioning starts only after every task has joined.
      SplitInfo* winners[] = {&best_split_static_constant, &best_split_static,
                             &best_split_nonstatic, &best_split_nowp};
      for (const auto& candidates : property_splits) {
        for (size_t i = 0; i < candidates.size(); ++i) {
          if (candidates[i].Cost() < winners[i]->Cost()) {
            *winners[i] = candidates[i];
          }
        }
      }

      // Try to avoid introducing WP.
      if (best_split_nowp.Cost() + threshold < base_bits &&
          best_split_nowp.Cost() <= fast_decode_multiplier * best->Cost()) {
        best = &best_split_nowp;
      }
      // Split along static props if possible and not significantly more
      // expensive.
      if (best_split_static.Cost() + threshold < base_bits &&
          best_split_static.Cost() <= fast_decode_multiplier * best->Cost()) {
        best = &best_split_static;
      }
      // Split along static props to create constant nodes if possible.
      if (best_split_static_constant.Cost() + threshold < base_bits) {
        best = &best_split_static_constant;
      }
      if (!property_finished.empty() && base_bits > threshold) {
        ++speculation_nodes;
        auto completion = property_finished;
        std::sort(completion.begin(), completion.end());
        const auto choose = [&](const std::array<SplitInfo, 4>& candidates) {
          size_t winner = 2;
          if (candidates[3].Cost() + threshold < base_bits &&
              candidates[3].Cost() <=
                  fast_decode_multiplier * candidates[winner].Cost()) {
            winner = 3;
          }
          if (candidates[1].Cost() + threshold < base_bits &&
              candidates[1].Cost() <=
                  fast_decode_multiplier * candidates[winner].Cost()) {
            winner = 1;
          }
          if (candidates[0].Cost() + threshold < base_bits) winner = 0;
          return candidates[winner];
        };
        // Replay only results already completed at each launch point. This
        // measures opportunity; it does not launch or assume free CPU for a
        // guessed subtree, and cannot change the real split decision.
        for (size_t quarter = 1; quarter <= 3; ++quarter) {
          const uint64_t cutoff =
              completion[std::max(size_t{1}, num_properties * quarter / 4) - 1];
          std::array<SplitInfo, 4> candidates;
          for (size_t prop = 0; prop < num_properties; ++prop) {
            if (property_finished[prop] > cutoff) continue;
            for (size_t c = 0; c < candidates.size(); ++c) {
              if (property_splits[prop][c].Cost() < candidates[c].Cost()) {
                candidates[c] = property_splits[prop][c];
              }
            }
          }
          const auto predicted = choose(candidates);
          if (!(predicted.Cost() + threshold < base_bits)) continue;
          auto& stats = speculation[quarter - 1];
          ++stats.attempts;
          const uint64_t lead = completion.back() - cutoff;
          stats.lead_ns += lead;
          if (best->Cost() + threshold < base_bits &&
              predicted.prop == best->prop && predicted.val == best->val &&
              predicted.lpred == best->lpred && predicted.rpred == best->rpred) {
            ++stats.hits;
            stats.hit_lead_ns += lead;
            stats.max_hit_lead_ns = std::max(stats.max_hit_lead_ns, lead);
          }
        }
      }
    }

    if (best->Cost() + threshold < base_bits) {
      uint32_t p = tree_samples.PropertyFromIndex(best->prop);
      pixel_type dequant =
          tree_samples.UnquantizeProperty(best->prop, best->val);
      // Split node and try to split children.
      MakeSplitNode(pos, p, dequant, best->lpred, 0, best->rpred, 0, tree);
      // "Sort" according to winning property
      const auto before_partition = measure ? Clock::now() : Clock::time_point{};
      if (batch_partition && end - begin >= 256) {
        ++batch_partition_nodes;
        if (best->prop < tree_samples.NumStaticProps()) {
          SplitTreeSamples<true, true>(tree_samples, begin, best->pos, end,
                                       best->prop, best->val);
        } else {
          SplitTreeSamples<false, true>(tree_samples, begin, best->pos, end,
              best->prop - tree_samples.NumStaticProps(), best->val);
        }
      } else if (best->prop < tree_samples.NumStaticProps()) {
        SplitTreeSamples<true>(tree_samples, begin, best->pos, end, best->prop,
                               best->val);
      } else {
        SplitTreeSamples<false>(tree_samples, begin, best->pos, end,
                                best->prop - tree_samples.NumStaticProps(),
                                best->val);
      }
      auto new_sp_range = static_prop_range;
      if (measure) {
        partition_seconds += std::chrono::duration<double>(
            Clock::now() - before_partition).count();
      }
      if (p < kNumStaticProperties) {
        JXL_DASSERT(static_cast<uint32_t>(dequant + 1) <= new_sp_range[p][1]);
        new_sp_range[p][1] = dequant + 1;
        JXL_DASSERT(new_sp_range[p][0] < new_sp_range[p][1]);
      }
      nodes.push_back(
          NodeInfo{(*tree)[pos].rchild, begin, best->pos, new_sp_range});
      nodes.back().redundant_properties = redundant_properties;
      new_sp_range = static_prop_range;
      if (p < kNumStaticProperties) {
        JXL_DASSERT(new_sp_range[p][0] <= static_cast<uint32_t>(dequant + 1));
        new_sp_range[p][0] = dequant + 1;
        JXL_DASSERT(new_sp_range[p][0] < new_sp_range[p][1]);
      }
      nodes.push_back(
          NodeInfo{(*tree)[pos].lchild, best->pos, end, new_sp_range});
      nodes.back().redundant_properties = std::move(redundant_properties);
      // Reuse only for sufficiently unbalanced splits. A balanced sibling
      // would cost as much to subtract as directly scanning the child.
      const size_t low = best->pos - begin, high = end - best->pos;
      if (reuse_parent && std::min(low, high) <= (end - begin - 1) / 3 &&
          std::max(low, high) >= 512 &&
          !(parallel_subtrees && pool_initialized && worker_slots > 1 &&
            std::max(low, high) <= subtree_grain)) {
        // Only scan the smaller child. Integer histogram subtraction recovers
        // all predictors of its larger sibling without visiting those samples.
        auto& smaller = nodes[nodes.size() - (low <= high ? 2 : 1)];
        auto& larger = nodes[nodes.size() - (low > high ? 2 : 1)];
        smaller.parents.resize(num_predictors);
        larger.parents.resize(num_predictors);
        for (size_t pred = 0; pred < num_predictors; ++pred) {
          CollectMaParentHistogram(tree_samples, pred, smaller.begin, smaller.end,
                                    &smaller.parents[pred]);
          larger.parents[pred] = SubtractMaParentHistogram(
              parent_histograms[pred], smaller.parents[pred]);
        }
        parent_sample_visits += num_predictors * (smaller.end - smaller.begin);
        parent_subtracted_samples += num_predictors * (larger.end - larger.begin);
      }
      if (cache && best != &forced_split && std::min(low, high) <= (end - begin - 1) / 3 &&
          std::max(low, high) >= 4096) {
        NodeInfo& larger = nodes[nodes.size() - (low > high ? 2 : 1)];
        larger.histograms = std::move(cache);
        larger.removed_begin = low > high ? best->pos : begin;
        larger.removed_end = low > high ? end : best->pos;
      }
    }
  }
  if (!subtrees.empty()) {
    LosslessProfileScope dispatch(profile.Id(), "ma.subtrees");
    std::vector<size_t> order(subtrees.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      return subtrees[a].end - subtrees[a].begin >
             subtrees[b].end - subtrees[b].begin;
    });
    const size_t tasks = std::min(worker_slots,
        LosslessWorkerLimits::FromEnvironment().Tasks(
            subtrees.size(), MaHistogramBudget::kLimit));
    std::atomic<size_t> next{0};
    const auto init_subtrees = [&](size_t threads) -> Status {
      dispatch.Set("worker_slots", threads);
      return true;
    };
    const auto learn_subtrees = [&](uint32_t, size_t) -> Status {
      for (;;) {
        const size_t index = next.fetch_add(1, std::memory_order_relaxed);
        if (index >= order.size()) break;
        auto& job = subtrees[order[index]];
        // Quantization is shared and immutable. Swaps touch only this job's
        // disjoint range; tree storage and histogram budgets are task-local.
        JXL_RETURN_IF_ERROR(FindBestSplitImpl(
            tree_samples, threshold, mul_info, job.range,
            fast_decode_multiplier, &job.tree, nullptr, job.begin, job.end,
            std::move(job.redundant), true, dispatch.Id()));
      }
      return true;
    };
    dispatch.Set("jobs", subtrees.size());
    dispatch.Set("worker_tasks", tasks);
    dispatch.Set("max_samples", subtrees[order.front()].end -
                                    subtrees[order.front()].begin);
    dispatch.Set("balanced", balanced_subtrees);
    dispatch.Set("cache_budget_bytes", tasks * MaHistogramBudget::kLimit);
    JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, tasks, init_subtrees, learn_subtrees,
                                  "MaSubtrees"));
    size_t added_nodes = 0;
    for (const auto& job : subtrees) {
      // Child indices are local to the job. The root replaces its original
      // leaf, while all other nodes are appended after the pool has joined.
      JXL_ENSURE(!job.tree.empty());
      JXL_ENSURE(tree->size() <= kMaxTreeSize);
      JXL_ENSURE(job.tree.size() - 1 <= kMaxTreeSize - tree->size());
      const size_t offset = tree->size() - 1;
      for (size_t i = 0; i < job.tree.size(); ++i) {
        auto node = job.tree[i];
        if (node.property >= 0) {
          node.lchild += offset;
          node.rchild += offset;
        }
        if (i == 0) {
          (*tree)[job.destination] = node;
        } else {
          tree->push_back(node);
        }
      }
      added_nodes += job.tree.size() - 1;
    }
    dispatch.Set("added_nodes", added_nodes);
  }
  if (paired_cache && profile.Enabled()) {
    LosslessProfileScope measured(profile.Id(), "ma.paired_cache_stats");
    uint64_t samples = 0;
    for (const auto& w : workspaces) samples += w.paired_cache_samples;
    measured.Set("paired_samples", samples);
  }
  if (profile.Enabled()) {
    LosslessProfileScope measured(profile.Id(), "ma.compact_histogram_stats");
    measured.Set("nodes", compact_nodes);
    measured.Set("parent_slots_saved", compact_slots_saved);
    uint64_t property_bins = 0, property_slots = 0;
    for (const auto& w : workspaces) {
      property_bins += w.property_bins_compacted;
      property_slots += w.property_slots_saved;
    }
    measured.Set("property_bins_compacted", property_bins);
    measured.Set("property_slots_saved", property_slots);
  }
  if (speculation_trace) {
    for (size_t i = 0; i < speculation.size(); ++i) {
      LosslessProfileScope measured(profile.Id(), "ma.speculation_stats");
      const auto& stats = speculation[i];
      measured.Set("completed_percent", (i + 1) * 25);
      measured.Set("eligible_nodes", speculation_nodes);
      measured.Set("attempts", stats.attempts);
      measured.Set("hits", stats.hits);
      measured.Set("lead_ns", stats.lead_ns);
      measured.Set("hit_lead_ns", stats.hit_lead_ns);
      measured.Set("max_hit_lead_ns", stats.max_hit_lead_ns);
    }
  }
  for (const auto& w : workspaces) {
    histogram_seconds += w.histogram_seconds;
    entropy_seconds += w.entropy_seconds;
    histogram_passes += w.histogram_passes;
    histogram_samples += w.histogram_samples;
    threshold_evaluations += w.threshold_evaluations;
    entropy_blocks_saved += w.entropy_blocks_saved;
    sharded_samples += w.sharded_samples;
    cache_hits += w.cache_hits;
    cache_saved_samples += w.cache_saved_samples;
    fused_pair_samples += w.fused_pair_samples;
    entropy_tail_lanes_saved += w.entropy_tail_lanes_saved;
    duplicate_properties += w.duplicate_properties;
    duplicate_histogram_samples += w.duplicate_histogram_samples;
    constant_properties += w.constant_properties;
    inherited_properties += w.inherited_properties;
    property_cache_hits += w.property_cache_hits;
    property_saved_samples += w.property_saved_samples;
  }
  {
    LosslessProfileScope equivalence(profile.Id(), "ma.predictor_equivalence");
    equivalence.Set("enabled", predictor_dedup);
    equivalence.Set("equivalent_predictors", equivalent_predictors);
    equivalence.Set("padded_histogram", padded_histogram);
    equivalence.Set("padded_nodes", padded_nodes);
  }
  profile.Set("split_worker_slots", worker_slots);
  profile.Set("parallel_nodes", parallel_nodes);
  profile.Set("parallel_properties", parallel_properties);
  profile.Set("parallel_wall_ns", static_cast<uint64_t>(parallel_seconds * 1e9));
  profile.Set("parent_parallel_nodes", parent_parallel_nodes);
  profile.Set("parent_sample_visits", parent_sample_visits);
  profile.Set("parent_reused_nodes", parent_reused_nodes);
  profile.Set("parent_subtracted_samples", parent_subtracted_samples);
  profile.Set("leaf_histogram_ns",
              static_cast<uint64_t>(leaf_histogram_seconds * 1e9));
  profile.Set("parent_histogram_ns",
              static_cast<uint64_t>(parent_histogram_seconds * 1e9));
  profile.Set("samples", sample_end - sample_begin);
  profile.Set("nodes", tree->size());
  profile.Set("nodes_visited", nodes_visited);
  profile.Set("predictors", num_predictors);
  profile.Set("properties", num_properties);
  profile.Set("histogram_passes", histogram_passes);
  profile.Set("race_nodes", race_nodes);
  profile.Set("race_draws", race_draws);
  profile.Set("race_pruned_properties", race_pruned);
  profile.Set("race_ns", static_cast<uint64_t>(race_seconds * 1e9));
  profile.Set("histogram_samples", histogram_samples);
  profile.Set("threshold_evaluations", threshold_evaluations);
  profile.Set("entropy_blocks_saved", entropy_blocks_saved);
  profile.Set("sharded_samples", sharded_samples);
  profile.Set("histogram_ns", static_cast<uint64_t>(histogram_seconds * 1e9));
  profile.Set("split_entropy_ns", static_cast<uint64_t>(entropy_seconds * 1e9));
  profile.Set("partition_ns", static_cast<uint64_t>(partition_seconds * 1e9));
  profile.Set("batch_partition_nodes", batch_partition_nodes);
  profile.Set("fused_pair_samples", fused_pair_samples);
  profile.Set("entropy_tail_lanes_saved", entropy_tail_lanes_saved);
  profile.Set("duplicate_properties", duplicate_properties);
  profile.Set("duplicate_histogram_samples", duplicate_histogram_samples);
  profile.Set("constant_properties", constant_properties);
  profile.Set("inherited_properties", inherited_properties);
  profile.Set("histogram_cache_hits", cache_hits);
  profile.Set("histogram_saved_samples", cache_saved_samples);
  profile.Set("histogram_cache_peak_bytes", cache_budget.peak);
  profile.Set("property_cache", property_cache);
  profile.Set("property_cache_hits", property_cache_hits);
  profile.Set("property_saved_samples", property_saved_samples);
  if (trace) {
    const double total = std::chrono::duration<double>(Clock::now() - start).count();
    std::fprintf(stderr,
                 "lossless-ma samples=%" PRIuS " nodes=%" PRIuS
                 " histogram_work_s=%.6f entropy_work_s=%.6f wall_s=%.6f\n",
                 tree_samples.NumDistinctSamples(), tree->size(),
                 histogram_seconds, entropy_seconds,
                 total);
    std::fflush(stderr);
  }
  return true;
}

Status FindBestSplit(TreeSamples &tree_samples, float threshold,
                    const std::vector<ModularMultiplierInfo> &mul_info,
                    StaticPropRange initial_static_prop_range,
                    float fast_decode_multiplier, Tree *tree, ThreadPool *pool) {
  const char* channels_env = std::getenv("JXL_MA_CHANNEL_TREE");
  if (tree_samples.ChannelTreeEligible() && mul_info.empty() &&
      tree_samples.NumDistinctSamples() >= 8192 && tree_samples.NumStaticProps() &&
      tree_samples.PropertyFromIndex(0) == 0 && tree_samples.NumPropertyValues(0) <= 32 &&
      channels_env && channels_env[0] >= '1' && channels_env[0] <= '3' && channels_env[1] == '\0') {
    const size_t size = tree_samples.NumDistinctSamples();
    std::vector<size_t> counts(tree_samples.NumPropertyValues(0));
    for (size_t i = 0; i < size; ++i) ++counts[tree_samples.Property<true>(0, i)];
    std::vector<size_t> occupied;
    for (size_t i = 0; i < counts.size(); ++i) if (counts[i]) occupied.push_back(i);
    if (occupied.size() > 1) {
      LosslessProfileScope profile(LosslessProfileCurrent(), "ma.channel_trees");
      struct Part {
        size_t node, begin, end, lo, hi;
        StaticPropRange range;
        Tree learned;
        size_t depth = 0;
      };
      std::vector<Part> pending{{0, 0, size, 0, occupied.size(), initial_static_prop_range, {}}};
      std::vector<Part> jobs;
      const Predictor pred = tree_samples.PredictorFromIndex(0);
      size_t spatial = tree_samples.NumProperties();
      for (size_t i = tree_samples.NumStaticProps(); i < tree_samples.NumProperties(); ++i)
        if (tree_samples.PropertyFromIndex(i) == 2) spatial = i;
      while (!pending.empty()) {
        Part p = std::move(pending.back());
        pending.pop_back();
        if (p.hi - p.lo == 1) {
          if (spatial != tree_samples.NumProperties() &&
              p.depth < size_t(channels_env[0] - '1') && p.end - p.begin >= 16384) {
            std::vector<size_t> bins(tree_samples.NumPropertyValues(spatial));
            const size_t property = spatial - tree_samples.NumStaticProps();
            for (size_t i = p.begin; i < p.end; ++i)
              ++bins[tree_samples.Property<false>(property, i)];
            size_t lower = 0, quant = 0;
            for (; quant + 1 < bins.size(); ++quant) {
              lower += bins[quant];
              if (2 * lower >= p.end - p.begin) break;
            }
            if (quant + 1 < bins.size() && lower >= 4096 && p.end - p.begin - lower >= 4096) {
              const size_t cut = p.begin + lower;
              MakeSplitNode(p.node, 2, tree_samples.UnquantizeProperty(spatial, quant),
                            pred, 0, pred, 0, tree);
              SplitTreeSamples<false, true>(tree_samples, p.begin, cut, p.end, property, quant);
              pending.push_back({(*tree)[p.node].rchild, p.begin, cut, p.lo, p.hi, p.range, {}, p.depth + 1});
              pending.push_back({(*tree)[p.node].lchild, cut, p.end, p.lo, p.hi, p.range, {}, p.depth + 1});
              continue;
            }
          }
          p.learned.push_back((*tree)[p.node]);
          jobs.push_back(std::move(p));
          continue;
        }
        const size_t mid = p.lo + (p.hi - p.lo) / 2;
        size_t cut = p.begin;
        for (size_t i = p.lo; i < mid; ++i) cut += counts[occupied[i]];
        const size_t quant = occupied[mid - 1];
        const int split = tree_samples.UnquantizeProperty(0, quant);
        JXL_ENSURE(split >= 0 && uint64_t{static_cast<uint32_t>(split)} + 1 < p.range[0][1]);
        MakeSplitNode(p.node, 0, split, pred, 0, pred, 0, tree);
        SplitTreeSamples<true, true>(tree_samples, p.begin, cut, p.end, 0, quant);
        auto low = p.range, high = p.range;
        low[0][1] = split + 1;
        high[0][0] = split + 1;
        pending.push_back({(*tree)[p.node].rchild, p.begin, cut, p.lo, mid, low, {}});
        pending.push_back({(*tree)[p.node].lchild, cut, p.end, mid, p.hi, high, {}});
      }
      // A channel split is normal static-property syntax. Quantization is
      // shared; every learner mutates only its own already-partitioned samples.
      // The shared arena allows nested property/subtree work without new pools.
      const auto limits = LosslessWorkerLimits::FromEnvironment();
      LosslessProbePool arena;
      JXL_RETURN_IF_ERROR(arena.Run(jobs.size(), pool, limits.tasks,
          limits.Tasks(jobs.size(), MaHistogramBudget::kLimit),
          [&](size_t i, ThreadPool* nested) -> Status {
            auto& p = jobs[i];
            return FindBestSplitImpl(tree_samples, threshold, mul_info, p.range,
                fast_decode_multiplier, &p.learned, nested, p.begin, p.end,
                {}, false, profile.Id());
          }));
      for (const auto& p : jobs) {
        JXL_ENSURE(!p.learned.empty() && tree->size() <= kMaxTreeSize);
        JXL_ENSURE(p.learned.size() - 1 <= kMaxTreeSize - tree->size());
        const size_t offset = tree->size() - 1;
        for (size_t i = 0; i < p.learned.size(); ++i) {
          auto node = p.learned[i];
          if (node.property >= 0) {
            node.lchild += offset;
            node.rchild += offset;
          }
          if (i == 0) (*tree)[p.node] = node;
          else tree->push_back(node);
        }
      }
      profile.Set("channels", occupied.size());
      profile.Set("jobs", jobs.size());
      profile.Set("samples", size);
      return true;
    }
  }
  return FindBestSplitImpl(
      tree_samples, threshold, mul_info, initial_static_prop_range,
      fast_decode_multiplier, tree, pool, 0, tree_samples.NumDistinctSamples(),
      {}, false, LosslessProfileCurrent());
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {

HWY_EXPORT(FindBestSplit);  // Local function.

Status ComputeBestTree(TreeSamples &tree_samples, float threshold,
                       const std::vector<ModularMultiplierInfo> &mul_info,
                       StaticPropRange static_prop_range,
                       float fast_decode_multiplier, Tree *tree,
                       ThreadPool *pool) {
  // TODO(veluca): take into account that different contexts can have different
  // uint configs.
  //
  // Initialize tree.
  tree->emplace_back();
  tree->back().property = -1;
  tree->back().predictor = tree_samples.PredictorFromIndex(0);
  tree->back().predictor_offset = 0;
  tree->back().multiplier = 1;
  JXL_ENSURE(tree_samples.NumProperties() < 64);

  JXL_ENSURE(tree_samples.NumDistinctSamples() <=
             std::numeric_limits<uint32_t>::max());
  return HWY_DYNAMIC_DISPATCH(FindBestSplit)
  (tree_samples, threshold, mul_info, static_prop_range, fast_decode_multiplier,
   tree, pool);
}

#if JXL_CXX_LANG < JXL_CXX_17
constexpr int32_t TreeSamples::kPropertyRange;
constexpr uint32_t TreeSamples::kDedupEntryUnused;
#endif

bool TreeSamples::FastDedupRequested() {
  const char* value = std::getenv("JXL_MA_FAST_DEDUP");
  return !value || std::strcmp(value, "1") == 0;
}

Status TreeSamples::SetPredictor(Predictor predictor,
                                 ModularOptions::TreeMode wp_tree_mode) {
  if (wp_tree_mode == ModularOptions::TreeMode::kWPOnly) {
    predictors = {Predictor::Weighted};
    residuals.resize(1);
    return true;
  }
  if (wp_tree_mode == ModularOptions::TreeMode::kNoWP &&
      predictor == Predictor::Weighted) {
    return JXL_FAILURE("Invalid predictor settings");
  }
  if (predictor == Predictor::Variable) {
    for (size_t i = 0; i < kNumModularPredictors; i++) {
      predictors.push_back(static_cast<Predictor>(i));
    }
    std::swap(predictors[0], predictors[static_cast<int>(Predictor::Weighted)]);
    std::swap(predictors[1], predictors[static_cast<int>(Predictor::Gradient)]);
  } else if (predictor == Predictor::Best) {
    predictors = {Predictor::Weighted, Predictor::Gradient};
  } else {
    predictors = {predictor};
  }
  if (wp_tree_mode == ModularOptions::TreeMode::kNoWP) {
    predictors.erase(
        std::remove(predictors.begin(), predictors.end(), Predictor::Weighted),
        predictors.end());
  }
  residuals.resize(predictors.size());
  return true;
}

Status TreeSamples::SetProperties(const std::vector<uint32_t> &properties,
                                  ModularOptions::TreeMode wp_tree_mode) {
  props_to_use = properties;
  if (wp_tree_mode == ModularOptions::TreeMode::kWPOnly) {
    props_to_use = {static_cast<uint32_t>(kWPProp)};
  }
  if (wp_tree_mode == ModularOptions::TreeMode::kGradientOnly) {
    props_to_use = {static_cast<uint32_t>(kGradientProp)};
  }
  if (wp_tree_mode == ModularOptions::TreeMode::kNoWP) {
    props_to_use.erase(
        std::remove(props_to_use.begin(), props_to_use.end(), kWPProp),
        props_to_use.end());
  }
  if (props_to_use.empty()) {
    return JXL_FAILURE("Invalid property set configuration");
  }
  num_static_props = 0;
  // Check that if static properties present, then those are at the beginning.
  for (size_t i = 0; i < props_to_use.size(); ++i) {
    uint32_t prop = props_to_use[i];
    if (prop < kNumStaticProperties) {
      JXL_DASSERT(i == prop);
      num_static_props++;
    }
  }
  props.resize(props_to_use.size() - num_static_props);
  return true;
}

void TreeSamples::InitTable(size_t log_size) {
  size_t size = 1ULL << log_size;
  if (dedup_table_.size() == size) return;
  if (fast_dedup_) {
    // Rehash into empty slots. Retaining old entries can register a sample
    // twice, leaving a stale alias after its 16-bit count saturates.
    dedup_table_.assign(size, kDedupEntryUnused);
  } else {
    dedup_table_.resize(size, kDedupEntryUnused);
  }
  for (size_t i = 0; i < NumDistinctSamples(); i++) {
    if (sample_counts[i] != std::numeric_limits<uint16_t>::max()) {
      AddToTable(i);
    }
  }
}

bool TreeSamples::AddToTableAndMerge(size_t a) {
  size_t pos1 = Hash1(a);
  size_t pos2 = fast_dedup_ ? 0 : Hash2(a);
  if (dedup_table_[pos1] != kDedupEntryUnused &&
      IsSameSample(a, dedup_table_[pos1])) {
    JXL_DASSERT(sample_counts[a] == 1);
    sample_counts[dedup_table_[pos1]]++;
    // Remove from hash table samples that are saturated.
    if (sample_counts[dedup_table_[pos1]] ==
        std::numeric_limits<uint16_t>::max()) {
      dedup_table_[pos1] = kDedupEntryUnused;
    }
    return true;
  }
  if (fast_dedup_) pos2 = Hash2(a);
  if (dedup_table_[pos2] != kDedupEntryUnused &&
      IsSameSample(a, dedup_table_[pos2])) {
    JXL_DASSERT(sample_counts[a] == 1);
    sample_counts[dedup_table_[pos2]]++;
    // Remove from hash table samples that are saturated.
    if (sample_counts[dedup_table_[pos2]] ==
        std::numeric_limits<uint16_t>::max()) {
      dedup_table_[pos2] = kDedupEntryUnused;
    }
    return true;
  }
  if (fast_dedup_) {
    // Reuse the probe positions; the sample and table have not changed.
    AddToTable(a, pos1, pos2);
  } else {
    AddToTable(a);
  }
  return false;
}

void TreeSamples::AddToTable(size_t a) {
  AddToTable(a, Hash1(a), Hash2(a));
}

bool TreeSamples::AddWeightedToTableAndMerge(size_t a) {
  return AddWeightedToTableAndMerge(a, Hash1(a), Hash2(a));
}

bool TreeSamples::AddWeightedToTableAndMerge(size_t a, size_t pos1, size_t pos2) {
  const size_t positions[] = {pos1, pos2};
  for (size_t pos : positions) {
    const uint32_t previous = dedup_table_[pos];
    if (previous == kDedupEntryUnused || !IsSameSample(a, previous)) continue;
    const uint32_t take = std::min<uint32_t>(
        sample_counts[a],
        std::numeric_limits<uint16_t>::max() - sample_counts[previous]);
    sample_counts[previous] += take;
    sample_counts[a] -= take;
    if (sample_counts[previous] == std::numeric_limits<uint16_t>::max()) {
      dedup_table_[pos] = kDedupEntryUnused;
    }
    if (sample_counts[a] == 0) return true;
  }
  if (sample_counts[a] != std::numeric_limits<uint16_t>::max()) {
    AddToTable(a, positions[0], positions[1]);
  }
  return false;
}

void TreeSamples::HashBatch(size_t begin, size_t count, uint64_t* first,
                           uint64_t* second) const {
  constexpr uint64_t c1 = 0x1e35a7bd, c2 = 0x1e35a7bd1e35a7bd;
  std::fill_n(first, count, c1);
  std::fill_n(second, count, c2);
  // Traverse each contiguous field once, with independent hash chains per row.
  // Preserve both original hash functions, including their field order.
  for (const auto& r : residuals) {
    for (size_t i = 0; i < count; ++i) {
      first[i] = (first[i] * c1 + r[begin + i].tok) * c1 + r[begin + i].nbits;
    }
  }
  for (size_t p = 0; p < num_static_props; ++p) {
    for (size_t i = 0; i < count; ++i) {
      first[i] = first[i] * c1 + static_props[p][begin + i];
      second[i] = second[i] * c2 ^ static_props[p][begin + i];
    }
  }
  for (const auto& p : props) {
    for (size_t i = 0; i < count; ++i) {
      first[i] = first[i] * c1 + p[begin + i];
      second[i] = second[i] * c2 ^ p[begin + i];
    }
  }
  for (const auto& r : residuals) {
    for (size_t i = 0; i < count; ++i) {
      second[i] = (second[i] * c2 ^ r[begin + i].tok) * c2 ^ r[begin + i].nbits;
    }
  }
  for (size_t i = 0; i < count; ++i) {
    first[i] = (first[i] >> 16) & (dedup_table_.size() - 1);
    second[i] = (second[i] >> 16) & (dedup_table_.size() - 1);
  }
}

void TreeSamples::MergeSampleBatch(const TreeSamples& batch, size_t begin,
                                  size_t end) {
  constexpr size_t kBatch = 256;
  std::array<uint64_t, kBatch> first, second;
  std::array<size_t, kBatch> retained;
  for (size_t start = begin; start < end;) {
    const size_t count = std::min(kBatch, end - start);
    const size_t base = sample_counts.size();
    const auto append = [&](auto& target, const auto& source) {
      target.insert(target.end(), source.begin() + start,
                    source.begin() + start + count);
    };
    for (size_t p = 0; p < residuals.size(); ++p) {
      append(residuals[p], batch.residuals[p]);
    }
    for (size_t p = 0; p < num_static_props; ++p) {
      append(static_props[p], batch.static_props[p]);
    }
    for (size_t p = 0; p < props.size(); ++p) append(props[p], batch.props[p]);
    append(sample_counts, batch.sample_counts);
    HashBatch(base, count, first.data(), second.data());
    size_t kept = 0;
    for (size_t i = 0; i < count; ++i) {
      if (!AddWeightedToTableAndMerge(base + i, first[i], second[i])) {
        retained[kept++] = i;
      }
    }
    start += count;
    if (kept == count) continue;
    // Hash entries still refer to un-compacted rows. Remap in increasing order
    // before moving fields, retaining the saturation rules of scalar merging.
    for (size_t i = 0; i < kept; ++i) {
      const size_t old = base + retained[i], next = base + i;
      if (dedup_table_[first[retained[i]]] == old) {
        dedup_table_[first[retained[i]]] = next;
      }
      if (dedup_table_[second[retained[i]]] == old) {
        dedup_table_[second[retained[i]]] = next;
      }
    }
    const auto compact = [&](auto& field) {
      for (size_t i = 0; i < kept; ++i) field[base + i] = field[base + retained[i]];
      field.resize(base + kept);
    };
    for (auto& r : residuals) compact(r);
    for (size_t p = 0; p < num_static_props; ++p) compact(static_props[p]);
    for (auto& p : props) compact(p);
    compact(sample_counts);
  }
}

Status TreeSamples::MergeSamples(std::vector<TreeSamples>& batches) {
  JXL_ENSURE(NumSamples() == 0 && !HasSamples());
  return AppendSamples(batches);
}

Status TreeSamples::AppendSamples(std::vector<TreeSamples>& batches) {
  FlushSampleBatch();
  for (auto& batch : batches) batch.FlushSampleBatch();
  JXL_ENSURE(!HasSamples() || !dedup_table_.empty());
  JXL_ENSURE(!predictors.empty() && !props_to_use.empty());
  const size_t previous_distinct = NumDistinctSamples();
  size_t total_distinct = previous_distinct, total_samples = NumSamples();
  for (const auto& batch : batches) {
    JXL_ENSURE(predictors == batch.predictors &&
               props_to_use == batch.props_to_use &&
               compact_properties == batch.compact_properties &&
               num_static_props == batch.num_static_props);
    JXL_ENSURE(batch.NumDistinctSamples() <=
               std::numeric_limits<uint32_t>::max() - total_distinct);
    JXL_ENSURE(batch.NumSamples() <=
               std::numeric_limits<size_t>::max() - total_samples);
    total_distinct += batch.NumDistinctSamples();
    total_samples += batch.NumSamples();
  }
  if (total_distinct == previous_distinct) return true;
  const char* batch_env = std::getenv("JXL_MA_BATCH_MERGE");
  const bool batch_merge = lossless_speedups_ && batch_env &&
                           std::strcmp(batch_env, "1") == 0;
  LosslessProfileScope profile(LosslessProfileCurrent(), "ma.merge_dedup");
  profile.Set("batch_merge", batch_merge);
  profile.Set("input_entries", total_distinct);
  profile.Set("weighted_samples", total_samples);
  // The legacy table's resize retains old slot aliases. A new wave must
  // rehash from empty slots so saturated weighted entries cannot be duplicated.
  if (!fast_dedup_) dedup_table_.clear();
  PrepareForSamples(total_distinct - previous_distinct);
  for (auto& batch : batches) {
    if (batch_merge) {
      MergeSampleBatch(batch, 0, batch.NumDistinctSamples());
      batch = TreeSamples(lossless_speedups_, fast_entropy_);
      continue;
    }
    for (size_t a = 0; a < batch.NumDistinctSamples(); ++a) {
      for (size_t p = 0; p < residuals.size(); ++p) {
        residuals[p].push_back(batch.residuals[p][a]);
      }
      for (size_t p = 0; p < num_static_props; ++p) {
        static_props[p].push_back(batch.static_props[p][a]);
      }
      for (size_t p = 0; p < props.size(); ++p) {
        props[p].push_back(batch.props[p][a]);
      }
      sample_counts.push_back(batch.sample_counts[a]);
      if (AddWeightedToTableAndMerge(sample_counts.size() - 1)) {
        for (auto& r : residuals) r.pop_back();
        for (size_t p = 0; p < num_static_props; ++p) static_props[p].pop_back();
        for (auto& p : props) p.pop_back();
        sample_counts.pop_back();
      }
    }
    // Release each input as it is consumed, including its deduplication table.
    batch = TreeSamples(lossless_speedups_, fast_entropy_);
  }
  num_samples = total_samples;
  profile.Set("output_entries", NumDistinctSamples());
  return true;
}

void TreeSamples::AddToTable(size_t a, size_t pos1, size_t pos2) {
  if (dedup_table_[pos1] == kDedupEntryUnused) {
    dedup_table_[pos1] = a;
  } else if (dedup_table_[pos2] == kDedupEntryUnused) {
    dedup_table_[pos2] = a;
  }
}

void TreeSamples::PrepareForSamples(size_t extra_num_samples) {
  FlushSampleBatch();
  const auto reserve = [&](auto& values) {
    size_t capacity = values.size() + extra_num_samples;
    if (lossless_speedups_ && capacity > values.capacity()) {
      capacity = std::max(capacity, values.capacity() +
          std::min(values.capacity() / 2, values.max_size() - values.capacity()));
    }
    values.reserve(capacity);
  };
  for (auto &res : residuals) {
    reserve(res);
  }
  for (size_t i = 0; i < num_static_props; ++i) {
    reserve(static_props[i]);
  }
  for (auto &p : props) {
    reserve(p);
  }
  if (lossless_speedups_) reserve(sample_counts);
  size_t total_num_samples = extra_num_samples + sample_counts.size();
  size_t next_size = CeilLog2Nonzero(total_num_samples * 3 / 2);
  InitTable(next_size);
}

size_t TreeSamples::Hash1(size_t a) const {
  constexpr uint64_t constant = 0x1e35a7bd;
  uint64_t h = constant;
  for (const auto &r : residuals) {
    h = h * constant + r[a].tok;
    h = h * constant + r[a].nbits;
  }
  for (size_t i = 0; i < num_static_props; ++i) {
    h = h * constant + static_props[i][a];
  }
  for (const auto &p : props) {
    h = h * constant + p[a];
  }
  return (h >> 16) & (dedup_table_.size() - 1);
}
size_t TreeSamples::Hash2(size_t a) const {
  constexpr uint64_t constant = 0x1e35a7bd1e35a7bd;
  uint64_t h = constant;
  for (size_t i = 0; i < num_static_props; ++i) {
    h = h * constant ^ static_props[i][a];
  }
  for (const auto &p : props) {
    h = h * constant ^ p[a];
  }
  for (const auto &r : residuals) {
    h = h * constant ^ r[a].tok;
    h = h * constant ^ r[a].nbits;
  }
  return (h >> 16) & (dedup_table_.size() - 1);
}

bool TreeSamples::IsSameSample(size_t a, size_t b) const {
  if (fast_dedup_) {
    for (const auto& r : residuals) {
      if (r[a].tok != r[b].tok || r[a].nbits != r[b].nbits) return false;
    }
    for (size_t i = 0; i < num_static_props; ++i) {
      if (static_props[i][a] != static_props[i][b]) return false;
    }
    for (const auto& p : props) {
      if (p[a] != p[b]) return false;
    }
    return true;
  }
  bool ret = true;
  for (const auto &r : residuals) {
    if (r[a].tok != r[b].tok) {
      ret = false;
    }
    if (r[a].nbits != r[b].nbits) {
      ret = false;
    }
  }
  for (size_t i = 0; i < num_static_props; ++i) {
    if (static_props[i][a] != static_props[i][b]) {
      ret = false;
    }
  }
  for (const auto &p : props) {
    if (p[a] != p[b]) {
      ret = false;
    }
  }
  return ret;
}

void TreeSamples::AddSample(pixel_type_w pixel, const Properties &properties,
                            const pixel_type_w *predictions) {
  for (size_t i = 0; i < predictors.size(); i++) {
    pixel_type v = pixel - predictions[static_cast<int>(predictors[i])];
    uint32_t tok, nbits, bits;
    HybridUintConfig(4, 1, 2).Encode(PackSigned(v), &tok, &nbits, &bits);
    JXL_DASSERT(tok < 256);
    JXL_DASSERT(nbits < 256);
    ResidualToken token = {static_cast<uint8_t>(tok),
                           static_cast<uint8_t>(nbits)};
    residuals[i].push_back(token);
  }
  for (size_t i = 0; i < num_static_props; ++i) {
    static_props[i].push_back(QuantizeStaticProperty(i, properties[i]));
  }
  for (size_t i = num_static_props; i < props_to_use.size(); i++) {
    props[i - num_static_props].push_back(QuantizeProperty(i, properties[props_to_use[i]]));
  }
  sample_counts.push_back(1);
  num_samples++;
  if (sample_batching_) {
    if (++pending_samples_ == 256) FlushSampleBatch();
    return;
  }
  if (AddToTableAndMerge(sample_counts.size() - 1)) {
    for (auto &r : residuals) r.pop_back();
    for (size_t i = 0; i < num_static_props; ++i) static_props[i].pop_back();
    for (auto &p : props) p.pop_back();
    sample_counts.pop_back();
  }
}

void TreeSamples::FlushSampleBatch() {
  if (pending_samples_ == 0) return;
  JXL_DASSERT(pending_samples_ <= 256);
  const size_t begin = sample_counts.size() - pending_samples_;
  std::array<uint64_t, 256> first, second;
  HashBatch(begin, pending_samples_, first.data(), second.data());
  size_t destination = begin;
  // Hash the contiguous columns together, then merge in original sample order.
  // Compaction only overwrites earlier rows; future rows and hashes stay valid.
  for (size_t i = 0; i < pending_samples_; ++i) {
    const size_t source = begin + i;
    if (source != destination) {
      for (auto& r : residuals) r[destination] = r[source];
      for (size_t p = 0; p < num_static_props; ++p) {
        static_props[p][destination] = static_props[p][source];
      }
      for (auto& p : props) p[destination] = p[source];
      sample_counts[destination] = sample_counts[source];
    }
    if (!AddWeightedToTableAndMerge(destination, first[i], second[i])) {
      ++destination;
    }
  }
  for (auto& r : residuals) r.resize(destination);
  for (size_t p = 0; p < num_static_props; ++p) {
    static_props[p].resize(destination);
  }
  for (auto& p : props) p.resize(destination);
  sample_counts.resize(destination);
  pending_samples_ = 0;
}

void TreeSamples::Swap(size_t a, size_t b) {
  if (a == b) return;
  for (auto &r : residuals) {
    std::swap(r[a], r[b]);
  }
  for (size_t i = 0; i < num_static_props; ++i) {
    std::swap(static_props[i][a], static_props[i][b]);
  }
  for (auto &p : props) {
    std::swap(p[a], p[b]);
  }
  std::swap(sample_counts[a], sample_counts[b]);
}

void TreeSamples::SwapBatch(const size_t* a, const size_t* b, size_t count) {
  const auto swap_field = [&](auto& field) {
    for (size_t i = 0; i < count; ++i) std::swap(field[a[i]], field[b[i]]);
  };
  for (auto& r : residuals) swap_field(r);
  for (size_t i = 0; i < num_static_props; ++i) swap_field(static_props[i]);
  for (auto& p : props) swap_field(p);
  swap_field(sample_counts);
}

namespace {
std::vector<int32_t> QuantizeHistogram(const std::vector<uint32_t> &histogram,
                                       size_t num_chunks) {
  if (histogram.empty() || num_chunks == 0) return {};
  uint64_t sum = std::accumulate(histogram.begin(), histogram.end(), 0LU);
  if (sum == 0) return {};
  // TODO(veluca): selecting distinct quantiles is likely not the best
  // way to go about this.
  std::vector<int32_t> thresholds;
  uint64_t cumsum = 0;
  uint64_t threshold = 1;
  for (size_t i = 0; i < histogram.size(); i++) {
    cumsum += histogram[i];
    if (cumsum * num_chunks >= threshold * sum) {
      thresholds.push_back(i);
      while (cumsum * num_chunks >= threshold * sum) threshold++;
    }
  }
  JXL_DASSERT(thresholds.size() <= num_chunks);
  // last value collects all histogram and is not really a threshold
  thresholds.pop_back();
  return thresholds;
}

std::vector<int32_t> QuantizeSamples(const std::vector<int32_t> &samples,
                                     size_t num_chunks) {
  if (samples.empty()) return {};
  int min = *std::min_element(samples.begin(), samples.end());
  constexpr int kRange = 512;
  min = jxl::Clamp1(min, -kRange, kRange);
  std::vector<uint32_t> counts(2 * kRange + 1);
  for (int s : samples) {
    uint32_t sample_offset = jxl::Clamp1(s, -kRange, kRange) - min;
    counts[sample_offset]++;
  }
  std::vector<int32_t> thresholds = QuantizeHistogram(counts, num_chunks);
  for (auto &v : thresholds) v += min;
  return thresholds;
}

// `to[i]` is assigned value `v` conforming `from[v] <= i && from[v-1] > i`.
// This is because the decision node in the tree splits on (property) > i,
// hence everything that is not > of a threshold should be clustered
// together.
template <typename T>
void QuantMap(const std::vector<int32_t> &from, std::vector<T> &to,
              size_t num_pegs, int bias) {
  to.resize(num_pegs);
  size_t mapped = 0;
  for (size_t i = 0; i < num_pegs; i++) {
    while (mapped < from.size() && static_cast<int>(i) - bias > from[mapped]) {
      mapped++;
    }
    JXL_DASSERT(static_cast<T>(mapped) == mapped);
    to[i] = mapped;
  }
}
}  // namespace

void TreeSamples::PreQuantizeProperties(
    const StaticPropRange &range,
    const std::vector<ModularMultiplierInfo> &multiplier_info,
    const std::vector<uint32_t> &group_pixel_count,
    const std::vector<uint32_t> &channel_pixel_count,
    std::vector<pixel_type> &pixel_samples,
    std::vector<pixel_type> &diff_samples, size_t max_property_values) {
  // If we have forced splits because of multipliers, choose channel and group
  // thresholds accordingly.
  std::vector<int32_t> group_multiplier_thresholds;
  std::vector<int32_t> channel_multiplier_thresholds;
  for (const auto &v : multiplier_info) {
    if (v.range[0][0] != range[0][0]) {
      channel_multiplier_thresholds.push_back(v.range[0][0] - 1);
    }
    if (v.range[0][1] != range[0][1]) {
      channel_multiplier_thresholds.push_back(v.range[0][1] - 1);
    }
    if (v.range[1][0] != range[1][0]) {
      group_multiplier_thresholds.push_back(v.range[1][0] - 1);
    }
    if (v.range[1][1] != range[1][1]) {
      group_multiplier_thresholds.push_back(v.range[1][1] - 1);
    }
  }
  std::sort(channel_multiplier_thresholds.begin(),
            channel_multiplier_thresholds.end());
  channel_multiplier_thresholds.resize(
      std::unique(channel_multiplier_thresholds.begin(),
                  channel_multiplier_thresholds.end()) -
      channel_multiplier_thresholds.begin());
  std::sort(group_multiplier_thresholds.begin(),
            group_multiplier_thresholds.end());
  group_multiplier_thresholds.resize(
      std::unique(group_multiplier_thresholds.begin(),
                  group_multiplier_thresholds.end()) -
      group_multiplier_thresholds.begin());

  compact_properties.resize(props_to_use.size());
  auto quantize_channel = [&]() {
    if (!channel_multiplier_thresholds.empty()) {
      return channel_multiplier_thresholds;
    }
    return QuantizeHistogram(channel_pixel_count, max_property_values);
  };
  auto quantize_group_id = [&]() {
    if (!group_multiplier_thresholds.empty()) {
      return group_multiplier_thresholds;
    }
    return QuantizeHistogram(group_pixel_count, max_property_values);
  };
  auto quantize_coordinate = [&]() {
    std::vector<int32_t> quantized;
    quantized.reserve(max_property_values - 1);
    for (size_t i = 0; i + 1 < max_property_values; i++) {
      quantized.push_back((i + 1) * 256 / max_property_values - 1);
    }
    return quantized;
  };
  std::vector<int32_t> abs_pixel_thresholds;
  std::vector<int32_t> pixel_thresholds;
  auto quantize_pixel_property = [&]() {
    if (pixel_thresholds.empty()) {
      pixel_thresholds = QuantizeSamples(pixel_samples, max_property_values);
    }
    return pixel_thresholds;
  };
  auto quantize_abs_pixel_property = [&]() {
    if (abs_pixel_thresholds.empty()) {
      quantize_pixel_property();  // Compute the non-abs thresholds.
      for (auto &v : pixel_samples) v = std::abs(v);
      abs_pixel_thresholds =
          QuantizeSamples(pixel_samples, max_property_values);
    }
    return abs_pixel_thresholds;
  };
  std::vector<int32_t> abs_diff_thresholds;
  std::vector<int32_t> diff_thresholds;
  auto quantize_diff_property = [&]() {
    if (diff_thresholds.empty()) {
      diff_thresholds = QuantizeSamples(diff_samples, max_property_values);
    }
    return diff_thresholds;
  };
  auto quantize_abs_diff_property = [&]() {
    if (abs_diff_thresholds.empty()) {
      quantize_diff_property();  // Compute the non-abs thresholds.
      for (auto &v : diff_samples) v = std::abs(v);
      abs_diff_thresholds = QuantizeSamples(diff_samples, max_property_values);
    }
    return abs_diff_thresholds;
  };
  auto quantize_wp = [&]() {
    if (max_property_values < 32) {
      return std::vector<int32_t>{-127, -63, -31, -15, -7, -3, -1, 0,
                                  1,    3,   7,   15,  31, 63, 127};
    }
    if (max_property_values < 64) {
      return std::vector<int32_t>{-255, -191, -127, -95, -63, -47, -31, -23,
                                  -15,  -11,  -7,   -5,  -3,  -1,  0,   1,
                                  3,    5,    7,    11,  15,  23,  31,  47,
                                  63,   95,   127,  191, 255};
    }
    return std::vector<int32_t>{
        -255, -223, -191, -159, -127, -111, -95, -79, -63, -55, -47,
        -39,  -31,  -27,  -23,  -19,  -15,  -13, -11, -9,  -7,  -6,
        -5,   -4,   -3,   -2,   -1,   0,    1,   2,   3,   4,   5,
        6,    7,    9,    11,   13,   15,   19,  23,  27,  31,  39,
        47,   55,   63,   79,   95,   111,  127, 159, 191, 223, 255};
  };

  property_mapping.resize(props_to_use.size() - num_static_props);
  for (size_t i = 0; i < props_to_use.size(); i++) {
    if (props_to_use[i] == 0) {
      compact_properties[i] = quantize_channel();
    } else if (props_to_use[i] == 1) {
      compact_properties[i] = quantize_group_id();
    } else if (props_to_use[i] == 2 || props_to_use[i] == 3) {
      compact_properties[i] = quantize_coordinate();
    } else if (props_to_use[i] == 6 || props_to_use[i] == 7 ||
               props_to_use[i] == 8 ||
               (props_to_use[i] >= kNumNonrefProperties &&
                (props_to_use[i] - kNumNonrefProperties) % 4 == 1)) {
      compact_properties[i] = quantize_pixel_property();
    } else if (props_to_use[i] == 4 || props_to_use[i] == 5 ||
               (props_to_use[i] >= kNumNonrefProperties &&
                (props_to_use[i] - kNumNonrefProperties) % 4 == 0)) {
      compact_properties[i] = quantize_abs_pixel_property();
    } else if (props_to_use[i] >= kNumNonrefProperties &&
               (props_to_use[i] - kNumNonrefProperties) % 4 == 2) {
      compact_properties[i] = quantize_abs_diff_property();
    } else if (props_to_use[i] == kWPProp) {
      compact_properties[i] = quantize_wp();
    } else {
      compact_properties[i] = quantize_diff_property();
    }
    if (i < num_static_props) {
      QuantMap(compact_properties[i], static_property_mapping[i],
               kPropertyRange * 2 + 1, kPropertyRange);
    } else {
      QuantMap(compact_properties[i], property_mapping[i - num_static_props],
               kPropertyRange * 2 + 1, kPropertyRange);
    }
  }
}

void CollectPixelSamples(const Image &image, const ModularOptions &options,
                         uint32_t group_id,
                         std::vector<uint32_t> &group_pixel_count,
                         std::vector<uint32_t> &channel_pixel_count,
                         std::vector<pixel_type> &pixel_samples,
                         std::vector<pixel_type> &diff_samples) {
  if (options.nb_repeats == 0) return;
  if (group_pixel_count.size() <= group_id) {
    group_pixel_count.resize(group_id + 1);
  }
  if (channel_pixel_count.size() < image.channel.size()) {
    channel_pixel_count.resize(image.channel.size());
  }
  Rng rng(group_id);
  // Sample 10% of the final number of samples for property quantization.
  float fraction = std::min(options.nb_repeats * 0.1, 0.99);
  Rng::GeometricDistribution dist = Rng::MakeGeometric(fraction);
  size_t total_pixels = 0;
  std::vector<size_t> channel_ids;
  for (size_t i = 0; i < image.channel.size(); i++) {
    if (i >= image.nb_meta_channels &&
        (image.channel[i].w > options.max_chan_size ||
         image.channel[i].h > options.max_chan_size)) {
      break;
    }
    if (image.channel[i].w <= 1 || image.channel[i].h == 0) {
      continue;  // skip empty or width-1 channels.
    }
    channel_ids.push_back(i);
    group_pixel_count[group_id] += image.channel[i].w * image.channel[i].h;
    channel_pixel_count[i] += image.channel[i].w * image.channel[i].h;
    total_pixels += image.channel[i].w * image.channel[i].h;
  }
  if (channel_ids.empty()) return;
  pixel_samples.reserve(pixel_samples.size() + fraction * total_pixels);
  diff_samples.reserve(diff_samples.size() + fraction * total_pixels);
  size_t i = 0;
  size_t y = 0;
  size_t x = 0;
  auto advance = [&](size_t amount) {
    x += amount;
    // Detect row overflow (rare).
    while (x >= image.channel[channel_ids[i]].w) {
      x -= image.channel[channel_ids[i]].w;
      y++;
      // Detect end-of-channel (even rarer).
      if (y == image.channel[channel_ids[i]].h) {
        i++;
        y = 0;
        if (i >= channel_ids.size()) {
          return;
        }
      }
    }
  };
  advance(rng.Geometric(dist));
  for (; i < channel_ids.size(); advance(rng.Geometric(dist) + 1)) {
    const pixel_type *row = image.channel[channel_ids[i]].Row(y);
    pixel_samples.push_back(row[x]);
    size_t xp = x == 0 ? 1 : x - 1;
    diff_samples.push_back(static_cast<int64_t>(row[x]) - row[xp]);
  }
}

// TODO(veluca): very simple encoding scheme. This should be improved.
Status TokenizeTree(const Tree &tree, std::vector<Token> *tokens,
                    Tree *decoder_tree) {
  JXL_ENSURE(tree.size() <= kMaxTreeSize);
  std::queue<int> q;
  q.push(0);
  size_t leaf_id = 0;
  decoder_tree->clear();
  while (!q.empty()) {
    int cur = q.front();
    q.pop();
    JXL_ENSURE(tree[cur].property >= -1);
    tokens->emplace_back(kPropertyContext, tree[cur].property + 1);
    if (tree[cur].property == -1) {
      tokens->emplace_back(kPredictorContext,
                           static_cast<int>(tree[cur].predictor));
      tokens->emplace_back(kOffsetContext,
                           PackSigned(tree[cur].predictor_offset));
      uint32_t mul_log = Num0BitsBelowLS1Bit_Nonzero(tree[cur].multiplier);
      uint32_t mul_bits = (tree[cur].multiplier >> mul_log) - 1;
      tokens->emplace_back(kMultiplierLogContext, mul_log);
      tokens->emplace_back(kMultiplierBitsContext, mul_bits);
      JXL_ENSURE(tree[cur].predictor < Predictor::Best);
      decoder_tree->emplace_back(
          -1, 0, static_cast<int>(leaf_id), 0, tree[cur].predictor,
          tree[cur].predictor_offset, tree[cur].multiplier);
      leaf_id++;
      continue;
    }
    decoder_tree->emplace_back(
        tree[cur].property, tree[cur].splitval,
        static_cast<int>(decoder_tree->size() + q.size() + 1),
        static_cast<int>(decoder_tree->size() + q.size() + 2), Predictor::Zero,
        0, 1);
    q.push(tree[cur].lchild);
    q.push(tree[cur].rchild);
    tokens->emplace_back(kSplitValContext, PackSigned(tree[cur].splitval));
  }
  return true;
}

}  // namespace jxl
#endif  // HWY_ONCE
