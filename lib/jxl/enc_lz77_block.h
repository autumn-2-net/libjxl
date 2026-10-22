// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef LIB_JXL_ENC_LZ77_BLOCK_H_
#define LIB_JXL_ENC_LZ77_BLOCK_H_

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/enc_ans_params.h"
#include "lib/jxl/enc_lossless_profile.h"
#include "lib/jxl/enc_lossless_workers.h"
#include "lib/jxl/enc_lz77_cost.h"
#include "lib/jxl/enc_lz77_index.h"

namespace jxl {

struct LZ77SeedSpan {
  size_t begin, end, distance;

  bool Search(size_t pos) const {
    // Preserve both copy boundaries and regularly spaced repair origins.
    return end - begin < 64 || pos - begin < 8 || end - pos <= 8 ||
           (pos - begin) % 32 == 0;
  }
};

inline std::vector<LZ77SeedSpan> LZ77SeedSpans(
    const std::vector<Token>& seed, size_t minimum, size_t width) {
  std::vector<LZ77SeedSpan> spans;
  size_t pos = 0;
  for (size_t i = 0; i < seed.size(); ++i) {
    if (!seed[i].is_lz77_length) { ++pos; continue; }
    const size_t length = seed[i].value + minimum;
    JXL_DASSERT(i + 1 < seed.size());
    const size_t symbol = seed[++i].value;
    const size_t distance = width && symbol < kNumSpecialDistances
        ? SpecialDistance(symbol, width)
        : symbol + 1 - (width ? kNumSpecialDistances : 0);
    spans.push_back({pos, pos + length, distance});
    pos += length;
  }
  return spans;
}

// Split only at first-pass token boundaries. The dictionary is not reset:
// references may cross these encoder-only boundaries in the final bitstream.
inline std::vector<size_t> LZ77ParseCuts(const std::vector<Token>& seed,
                                        size_t size, size_t parts,
                                        size_t minimum) {
  std::vector<size_t> cuts{0};
  size_t position = 0, target = 1;
  for (size_t i = 0; i < seed.size(); ++i) {
    const bool match = seed[i].is_lz77_length;
    position += match ? seed[i].value + minimum : 1;
    if (match) ++i;
    if (target < parts && position >= size * target / parts && position < size) {
      cuts.push_back(position);
      do { ++target; } while (target < parts && position >= size * target / parts);
    }
  }
  JXL_DASSERT(position == size);
  cuts.push_back(size);
  return cuts;
}

inline std::vector<size_t> LZ77WorkCuts(const std::vector<Token>& input,
                                      const std::vector<Token>& seed,
                                      size_t parts, size_t minimum) {
  const auto visit = [&](const auto& callback) {
    size_t pos = 0;
    uint64_t work = 0;
    for (size_t i = 0; i < seed.size(); ++i) {
      if (seed[i].is_lz77_length) {
        const size_t length = seed[i].value + minimum;
        work += 16 * std::min<size_t>(64, length) + length / 16 + 1;
        pos += length;
        ++i;
      } else {
        work += pos && input[pos].value == input[pos - 1].value ? 1 : 16;
        ++pos;
      }
      callback(pos, work);
    }
    JXL_DASSERT(pos == input.size());
    return work;
  };
  const uint64_t total = visit([](size_t, uint64_t) {});
  std::vector<size_t> cuts{0};
  size_t target = 1;
  visit([&](size_t pos, uint64_t work) {
    if (target < parts && work >= total * target / parts && pos < input.size()) {
      cuts.push_back(pos);
      do { ++target; } while (target < parts && work >= total * target / parts);
    }
  });
  cuts.push_back(input.size());
  return cuts;
}

template <bool kIntervals, typename Index, typename Estimator>
Status ParseLZ77Blocks(const std::vector<Token>& input,
                       const std::vector<Token>& seed, const Index& index,
                       const Estimator& costs, const HistogramParams& params,
                       const LZ77Params& lz77, size_t width, size_t window,
                       std::vector<Token>* output) {
  LosslessProfileScope profile(params.lossless_profile, "lz77.block_parse");
  const char* cache_env = std::getenv("JXL_LOSSLESS_LZ77_INDEX_CACHE");
  const bool cache = cache_env &&
      (std::strcmp(cache_env, "1") == 0 || std::strcmp(cache_env, "2") == 0);
  const bool sparse_cache = cache && cache_env[0] == '2';
  const char* frontier_env = std::getenv("JXL_LOSSLESS_LZ77_SHARED_FRONTIER");
  const bool shared_frontier = frontier_env && std::strcmp(frontier_env, "1") == 0;
  const char* carry_env = std::getenv("JXL_LOSSLESS_LZ77_FRONTIER_CARRY");
  const bool carry_frontier = carry_env && std::strcmp(carry_env, "1") == 0;
  const char* bound_value = std::getenv("JXL_LOSSLESS_LZ77_DISTANCE_BOUND");
  const bool distance_bound = bound_value && std::strcmp(bound_value, "1") == 0;
  const auto price_bound = distance_bound ? MinimumLZ77DistancePrice(
      static_cast<uint32_t>(window + (width ? kNumSpecialDistances : 0) - 1),
      [&](uint32_t d) { return costs.DistCost(d, lz77); }) : LZ77DistanceBound{};
  const uint64_t cache_bytes = cache ? (sparse_cache ? 32768 : 4 * window) : 0;
  const auto limits = LosslessWorkerLimits::FromEnvironment();
  const size_t workers = params.lossless_lz77_pool ? limits.Tasks(
      std::max<size_t>(1, std::min<size_t>(32, input.size() / 65536)),
      (8u << 20) + cache_bytes) : 1;
  const size_t granularity = LosslessWorkerSetting(
      std::getenv("JXL_LOSSLESS_LZ77_BLOCK_GRAIN"), 1, 8);
  const size_t parts = std::max(workers,
      std::min(workers * granularity, input.size() / 16384));
  const char* work_env = std::getenv("JXL_LOSSLESS_LZ77_WORK_CUTS");
  const bool work_cuts = work_env && std::strcmp(work_env, "1") == 0;
  const auto cuts = work_cuts
      ? LZ77WorkCuts(input, seed, parts, lz77.min_length)
      : LZ77ParseCuts(seed, input.size(), parts, lz77.min_length);
  const size_t tasks = cuts.size() - 1;
  const size_t minimum = lz77.min_length;
  const auto config = params.UintConfig();
  const char* table_env = std::getenv("JXL_LOSSLESS_LZ77_DISTANCE_TABLE");
  const char* class_env = std::getenv("JXL_LOSSLESS_LZ77_DISTANCE_CLASSES");
  const bool use_classes = class_env && std::strcmp(class_env, "1") == 0;
  std::unique_ptr<LZ77DistancePrices> distance_classes;
  if (use_classes) distance_classes = std::make_unique<LZ77DistancePrices>(
      [&](uint32_t distance) { return costs.DistCost(distance, lz77); });
  std::vector<float> distance_table;
  if (table_env && std::strcmp(table_env, "1") == 0) {
    distance_table.resize(window + (width ? kNumSpecialDistances : 0));
    size_t begin = 0;
    while (begin < distance_table.size()) {
      const size_t span = begin < 16 ? 1 : size_t{1} << (FloorLog2Nonzero(begin) - 2);
      const size_t end = std::min(distance_table.size(), begin + span);
      std::fill(distance_table.begin() + begin, distance_table.begin() + end,
                 costs.DistCost(begin, lz77));
      begin = end;
    }
  }
  const char* calendar_env = std::getenv("JXL_LOSSLESS_LZ77_CALENDAR");
  const bool calendar = kIntervals && calendar_env && std::strcmp(calendar_env, "1") == 0;
  const char* zero_env = std::getenv("JXL_LOSSLESS_LZ77_ZERO_TAIL");
  const bool zero_tail = zero_env && std::strcmp(zero_env, "1") == 0;
  const char* bound_env = std::getenv("JXL_LOSSLESS_LZ77_LITERAL_BOUND");
  const bool literal_bound = bound_env && std::strcmp(bound_env, "1") == 0;
  const char* vector_env = std::getenv("JXL_LOSSLESS_LZ77_VECTOR_DP");
  const bool vector_costs = vector_env && std::strcmp(vector_env, "1") == 0;
  const char* dominance_env = std::getenv("JXL_LOSSLESS_LZ77_RANGE_DOMINANCE");
  const bool range_dominance = dominance_env && std::strcmp(dominance_env, "1") == 0;
  const char* jump_env = std::getenv("JXL_LOSSLESS_LZ77_DP_JUMP");
  const bool jump_runs = kIntervals && jump_env && std::strcmp(jump_env, "1") == 0;
  const char* dyadic_env = std::getenv("JXL_LOSSLESS_LZ77_DYADIC_DP");
  const unsigned dyadic = kIntervals && dyadic_env && dyadic_env[0] >= '1' &&
      dyadic_env[0] <= '3' && dyadic_env[1] == '\0' ? dyadic_env[0] - '0' : 0;
  const char* envelope_env = std::getenv("JXL_LOSSLESS_LZ77_BLOCK_ENVELOPE");
  const unsigned envelope_shift = kIntervals && envelope_env &&
      envelope_env[0] >= '1' && envelope_env[0] <= '3' && envelope_env[1] == '\0'
      ? (envelope_env[0] == '1' ? 6 : envelope_env[0] == '2' ? 4 : 8) : 0;
  const char* lazy_env = std::getenv("JXL_LOSSLESS_LZ77_LAZY");
  const bool lazy = lazy_env && std::strcmp(lazy_env, "1") == 0;
  const char* guide_env = std::getenv("JXL_LOSSLESS_LZ77_SEED_GUIDED");
  const bool guided = guide_env &&
      (std::strcmp(guide_env, "1") == 0 || std::strcmp(guide_env, "2") == 0);
  const bool sparse_origins = guided && guide_env[0] == '2';
  const auto spans = guided ? LZ77SeedSpans(seed, minimum, width)
                            : std::vector<LZ77SeedSpan>{};
  std::vector<std::vector<Token>> outputs(tasks);
  profile.Set("tasks", tasks);
  profile.Set("work_cuts", work_cuts);
  profile.Set("tokens", input.size());
  profile.Set("cache_bytes_per_task", cache_bytes);
  profile.Set("intervals", kIntervals);
  profile.Set("shared_frontier", shared_frontier);
  profile.Set("frontier_carry", carry_frontier);
  profile.Set("distance_bound", distance_bound);
  profile.Set("seed_guided", guided ? (sparse_origins ? 2 : 1) : 0);
  const auto work = [&](uint32_t task, size_t) -> Status {
    LosslessProfileScope block(profile.Id(), "lz77.block");
    const size_t begin = cuts[task], end = cuts[task + 1];
    const size_t count = end - begin;
    size_t search_end = end;
    if (zero_tail) {
      // With nonnegative model costs, a zero-cost literal suffix cannot be
      // improved by starting another copy. Earlier copies may still end here.
      while (search_end > begin) {
        const auto& t = input[search_end - 1];
        uint32_t token, nbits, bits;
        config.Encode(t.value, &token, &nbits, &bits);
        if (nbits || costs.Bits(t.context, token) != 0) break;
        --search_end;
      }
    }
    std::unique_ptr<LZ77MatchCache> verified;
    if (cache) verified = std::make_unique<LZ77MatchCache>(window, sparse_cache ? 4096 : 0);
    if (lazy) {
      std::vector<double> prefix(count + 1);
      for (size_t i = 0; i < count; ++i) {
        const auto& t = input[begin + i];
        uint32_t token, nbits, bits;
        config.Encode(t.value, &token, &nbits, &bits);
        prefix[i + 1] = prefix[i] + nbits + costs.Bits(t.context, token);
      }
      struct Choice { size_t length = 0; uint32_t distance = 0; double gain = 0; };
      size_t queries = 0;
      const auto choose = [&](size_t local) {
        Choice best;
        if (local + minimum > count || prefix.back() == prefix[local]) return best;
        ++queries;
        const size_t pos = begin + local;
        const auto distance_cost = [&](uint32_t d) {
          return distance_table.empty() ? costs.DistCost(d, lz77) : distance_table[d];
        };
        index.FindMatches(pos, distance_cost, [&](size_t length, uint32_t distance) {
          VisitLZ77CostIntervals(minimum, minimum, length + 1, distance,
              distance_cost(distance), 0,
              [&](size_t residual) { return costs.LenCost(input[pos].context, residual, lz77); },
              [&](size_t, size_t last, uint32_t d, float bits) {
                const size_t n = last - 1;
                const double gain = prefix[local + n] - prefix[local] - bits;
                if (gain > best.gain || (gain == best.gain && gain > 0 && n > best.length))
                  best = {n, d, gain};
              });
        }, verified.get(), count - local);
        return best;
      };
      auto& out = outputs[task];
      out.reserve(count);
      size_t local = 0;
      Choice current;
      bool pending = false;
      while (local < count) {
        if (!pending) current = choose(local);
        pending = false;
        if (current.length >= minimum && local + 1 < count) {
          const auto next = choose(local + 1);
          if (next.gain > current.gain) {
            out.push_back(input[begin + local++]);
            current = next;
            pending = true;
            continue;
          }
        }
        if (current.length < minimum) {
          out.push_back(input[begin + local++]);
        } else {
          out.emplace_back(input[begin + local].context, current.length - minimum);
          out.back().is_lz77_length = true;
          out.emplace_back(lz77.nonserialized_distance_context, current.distance);
          local += current.length;
        }
      }
      block.Set("tokens", count);
      block.Set("queries", queries);
      block.Set("lazy", 1);
      return true;
    }
    LZ77RangeParser parser(count, kIntervals ? 1u << 16 : 0, calendar,
                           vector_costs, envelope_shift, dyadic, range_dominance);
    std::vector<double> literal_prefix(literal_bound ? count + 1 : 0);
    if (literal_bound) {
      for (size_t i = 0; i < count; ++i) {
        const auto& token = input[begin + i];
        uint32_t symbol, nbits, bits;
        config.Encode(token.value, &symbol, &nbits, &bits);
        literal_prefix[i + 1] = literal_prefix[i] +
            static_cast<double>(costs.Bits(token.context, symbol)) + nbits;
      }
    }
    size_t pruned_intervals = 0, pruned_endpoints = 0;
    size_t skip = 0, run = 0, previous_end = 0;
    uint32_t previous_distance = 0;
    size_t span = std::lower_bound(spans.begin(), spans.end(), begin,
        [](const auto& copy, size_t pos) { return copy.end <= pos; }) - spans.begin();
    size_t queries = 0, hinted = 0, jumped = 0;
    LZ77FrontierCarry carried;
    for (size_t local = 0; local < count; ++local) {
      const size_t pos = begin + local;
      if constexpr (kIntervals) parser.Advance(local);
      if (jump_runs && skip) {
        const size_t target = parser.JumpLiteralOnly(local, std::min(count, local + skip));
        if (target > local) {
          jumped += target - local;
          skip -= target - local;
          local = target - 1;
          continue;
        }
      }
      uint32_t token, nbits, bits;
      config.Encode(input[pos].value, &token, &nbits, &bits);
      const float literal = parser.At(local).cost + costs.Bits(input[pos].context, token) + nbits;
      parser.Literal(local, input[pos].context, literal);
      if (pos >= search_end) continue;
      if (skip) { --skip; continue; }
      while (span < spans.size() && spans[span].end <= pos) ++span;
      const bool in_copy = span < spans.size() && spans[span].begin <= pos;
      if (sparse_origins && in_copy && !spans[span].Search(pos)) continue;
      const size_t hint = in_copy ? spans[span].distance : 0;
      ++queries;
      hinted += hint != 0;
      LZ77MatchDominance frontier;
      const auto distance_cost = [&](uint32_t distance) {
        if (use_classes) return (*distance_classes)(distance);
        return distance_table.empty() ? costs.DistCost(distance, lz77) : distance_table[distance];
      };
      if (shared_frontier) {
        index.FindMatches(pos, distance_cost, [](size_t, uint32_t) {},
                          verified.get(), count - local, hint, &frontier, price_bound,
                          carry_frontier ? &carried : nullptr);
      } else {
        index.FindMatches(pos, distance_cost, [&](size_t length, uint32_t distance) {
          frontier.Add(static_cast<uint32_t>(length), distance, distance_cost(distance));
        }, verified.get(), count - local, hint, nullptr, price_bound,
           carry_frontier ? &carried : nullptr);
      }
      size_t longest = 0;
      uint32_t longest_distance = 0;
      const auto relax = [&](size_t first, size_t last, uint32_t distance) {
        VisitLZ77CostIntervals(minimum, first, last + 1, distance,
            distance_cost(distance), parser.At(local).cost,
            [&](size_t value) { return costs.LenCost(input[pos].context, value, lz77); },
            [&](size_t first_length, size_t end_length, uint32_t symbol, float cost) {
              if (literal_bound) {
                // Nonnegative literal costs bound every endpoint in this
                // constant-cost interval. Include float accumulation error;
                // near ties remain in the normal shortest-path search.
                const double upper = parser.At(local).cost +
                    literal_prefix[local + end_length - 1] - literal_prefix[local];
                const double margin = (end_length + 2) *
                    std::numeric_limits<float>::epsilon() * std::max(1.0, upper);
                if (cost > upper + margin) {
                  ++pruned_intervals;
                  pruned_endpoints += end_length - first_length;
                  return;
                }
              }
              parser.Match(local, first_length, end_length, symbol, input[pos].context, cost);
            });
        if (last >= longest) { longest = last; longest_distance = distance; }
      };
      if (frontier.Overflowed()) {
        index.FindMatches(pos, distance_cost, [&](size_t length, uint32_t distance) {
          relax(minimum, length, distance);
        }, verified.get(), count - local, hint, nullptr, {},
           carry_frontier ? &carried : nullptr);
      } else {
        size_t first = minimum;
        frontier.ForEach([&](size_t length, uint32_t distance) {
          relax(first, length, distance);
          first = length + 1;
        });
      }
      if (carry_frontier) carried.Remember(pos, frontier);
      if (longest < minimum) { run = 0; continue; }
      const size_t match_end = pos + longest;
      if (longest_distance == 0 || (longest_distance == 1 && width != 0) ||
          (longest >= 256 && match_end == previous_end &&
           longest_distance == previous_distance)) ++run;
      else run = 0;
      previous_end = match_end;
      previous_distance = longest_distance;
      if (run >= 8 && longest >= 9) { skip = longest - 9; run = 0; }
    }
    auto& out = outputs[task];
    if constexpr (kIntervals) parser.Advance(count);
    out.reserve(count);
    for (size_t pos = count; pos;) {
      const auto& node = parser.At(pos);
      JXL_ENSURE(node.length != 0 && node.length <= pos);
      if (node.distance) {
        out.emplace_back(static_cast<uint32_t>(lz77.nonserialized_distance_context),
                         node.distance - 1);
      }
      out.emplace_back(input[begin + pos - node.length].context,
                       node.distance ? node.length - minimum : input[begin + pos - 1].value);
      out.back().is_lz77_length = node.distance != 0;
      pos -= node.length;
    }
    std::reverse(out.begin(), out.end());
    block.Set("tokens", count);
    block.Set("queries", queries);
    block.Set("hinted", hinted);
    block.Set("jumped_tokens", jumped);
    block.Set("envelope_shift", envelope_shift);
    block.Set("dyadic", dyadic);
    block.Set("calendar", calendar);
    block.Set("vector_costs", vector_costs);
    block.Set("intervals", parser.Intervals());
    block.Set("scalar_endpoints", parser.ScalarEndpoints());
    block.Set("dominated_intervals", parser.DominatedIntervals());
    block.Set("peak_queued", parser.PeakQueued());
    block.Set("zero_tail", end - search_end);
    block.Set("pruned_intervals", pruned_intervals);
    block.Set("pruned_endpoints", pruned_endpoints);
    return true;
  };
  std::atomic<size_t> next{0};
  const auto dynamic_work = [&](uint32_t, size_t thread) -> Status {
    for (;;) {
      const size_t task = next.fetch_add(1, std::memory_order_relaxed);
      if (task >= tasks) return true;
      JXL_RETURN_IF_ERROR(work(task, thread));
    }
  };
  JXL_RETURN_IF_ERROR(RunOnPool(params.lossless_lz77_pool, 0, std::min(workers, tasks),
      [&](size_t workers) -> Status { profile.Set("worker_slots", workers); return true; },
      dynamic_work, "LosslessLZ77Blocks"));
  size_t count = 0;
  for (const auto& out : outputs) count += out.size();
  output->reserve(count);
  for (const auto& out : outputs) output->insert(output->end(), out.begin(), out.end());
  return true;
}

}  // namespace jxl

#endif  // LIB_JXL_ENC_LZ77_BLOCK_H_
