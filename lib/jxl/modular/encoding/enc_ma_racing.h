// Copyright (c) the JPEG XL Project Authors.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef LIB_JXL_MODULAR_ENCODING_ENC_MA_RACING_H_
#define LIB_JXL_MODULAR_ENCODING_ENC_MA_RACING_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "lib/jxl/enc_lossless_racing.h"
#include "lib/jxl/modular/encoding/context_predict.h"
#include "lib/jxl/modular/encoding/enc_ma.h"

namespace jxl {

// Refine the two best separated coarse valleys, keeping every coarse split.
// Entropy need not be unimodal, so this is an opt-in tree-learning heuristic.
inline std::vector<uint8_t> RefineMaQuantileMask(
    const std::vector<uint8_t>& coarse, const std::vector<float>& costs) {
  auto refined = coarse;
  if (coarse.size() != costs.size()) return refined;
  std::vector<size_t> points, order;
  for (size_t i = 0; i < coarse.size(); ++i)
    if (coarse[i] && std::isfinite(costs[i])) points.push_back(i);
  for (size_t i = 0; i < points.size(); ++i) order.push_back(i);
  std::stable_sort(order.begin(), order.end(),
      [&](size_t a, size_t b) { return costs[points[a]] < costs[points[b]]; });
  std::vector<uint8_t> expanded(coarse.size());
  size_t valleys = 0;
  for (size_t index : order) {
    if (expanded[points[index]]) continue;
    const size_t begin = index ? points[index - 1] : 0;
    const size_t end = index + 1 < points.size() ? points[index + 1] + 1 : coarse.size();
    std::fill(refined.begin() + begin, refined.begin() + end, 1);
    std::fill(expanded.begin() + begin, expanded.begin() + end, 1);
    if (++valleys == 2) break;
  }
  return refined;
}

// Node-local proposal statistics. Survivors still use the complete original
// histogram/threshold search; this never supplies a split or partition index.
class MaPropertyRace {
 public:
  static constexpr size_t kPanels = 4;
  MaPropertyRace(const TreeSamples& samples, size_t begin, size_t end,
                 size_t symbols)
      : samples_(samples), begin_(begin), end_(end), symbols_(symbols),
        states_(samples.NumProperties()) {}

  static bool Fits(const TreeSamples& samples, size_t symbols) {
    size_t values = 0;
    for (size_t p = samples.NumStaticProps(); p < samples.NumProperties(); ++p) {
      if (samples.NumPropertyValues(p) > 256) return false;
      values += samples.NumPropertyValues(p);
    }
    const uint64_t cells = uint64_t{values} * symbols *
                            samples.NumPredictors() * kPanels;
    return cells * sizeof(uint32_t) <= (32u << 20);
  }

  void Extend(size_t property, size_t target) {
    auto& state = states_[property];
    const size_t values = samples_.NumPropertyValues(property);
    const size_t predictors = samples_.NumPredictors();
    if (state.hist.empty()) {
      state.hist.resize(kPanels * predictors * values * symbols_);
      state.extra.resize(kPanels * predictors * values);
      state.weight.resize(kPanels * values);
    }
    for (size_t panel = 0; panel < kPanels; ++panel) {
      for (size_t k = state.draws; k < target; ++k) {
        const size_t index = SampleIndex(panel, k);
        const size_t value = samples_.Property<false>(
            property - samples_.NumStaticProps(), index);
        const uint32_t weight = samples_.Count(index);
        state.weight[panel * values + value] += weight;
        state.mass[panel] += weight;
        state.mass2[panel] += double(weight) * weight;
        for (size_t pred = 0; pred < predictors; ++pred) {
          const auto token = samples_.RTokens(pred)[index];
          const size_t row = (panel * predictors + pred) * values + value;
          state.hist[row * symbols_ + token.tok] += weight;
          state.extra[row] += uint64_t{weight} * token.nbits;
        }
      }
      state.costs[panel] = Score(state, panel, values);
    }
    state.draws = target;
  }

  std::vector<uint8_t> Survivors(const std::vector<uint8_t>& active) const {
    auto keep = active;
    std::vector<size_t> order;
    for (size_t p = samples_.NumStaticProps(); p < states_.size(); ++p) {
      if (active[p]) order.push_back(p);
    }
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      return Mean(states_[a]) < Mean(states_[b]);
    });
    if (order.empty() || !Reliable(states_[order[0]])) return keep;
    const auto& leader = states_[order[0]];
    std::vector<double> reference;
    for (double cost : leader.costs) reference.push_back(cost + 1.0);
    // Preserve several competing partitions and the WP-specific property.
    for (size_t i = 4; i < order.size(); ++i) {
      const size_t p = order[i];
      if (samples_.PropertyFromIndex(p) == kWPProp || !Reliable(states_[p])) continue;
      std::vector<double> costs;
      for (double cost : states_[p].costs) costs.push_back(cost + 1.0);
      const auto evidence = PairedLosslessEvidence(costs, reference, 0.02);
      if (evidence.ClearlyWorse(0.01)) keep[p] = 0;
    }
    return keep;
  }
  const std::array<double, kPanels>& Costs(size_t property) const {
    return states_[property].costs;
  }

 private:
  struct State {
    std::vector<uint32_t> hist, weight;
    std::vector<uint64_t> extra;
    std::array<double, kPanels> mass{}, mass2{}, costs{};
    size_t draws = 0;
  };
  size_t SampleIndex(size_t panel, size_t k) const {
    uint64_t x = uint64_t{k} + panel * 0x9e3779b97f4a7c15ULL +
                 begin_ * 0xbf58476d1ce4e5b9ULL + end_;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return begin_ + x % (end_ - begin_);
  }
  static double CountCost(double count) {
    return count > 1 ? count * std::log2(count) : 0;
  }
  static bool Reliable(const State& state) {
    for (size_t i = 0; i < kPanels; ++i) {
      if (state.mass2[i] == 0 ||
          state.mass[i] * state.mass[i] / state.mass2[i] < 32 ||
          !std::isfinite(state.costs[i])) return false;
    }
    return true;
  }
  static double Mean(const State& state) {
    double sum = 0;
    for (double cost : state.costs) sum += cost;
    return sum / kPanels;
  }
  double Score(const State& state, size_t panel, size_t values) const {
    if (values < 2 || state.mass[panel] == 0) {
      return std::numeric_limits<double>::infinity();
    }
    std::vector<double> left(values, std::numeric_limits<double>::infinity());
    auto right = left;
    double parent_best = std::numeric_limits<double>::infinity();
    for (size_t pred = 0; pred < samples_.NumPredictors(); ++pred) {
      const size_t base = (panel * samples_.NumPredictors() + pred) * values;
      std::vector<uint32_t> above(symbols_), below(symbols_);
      double extra_total = 0, entropy_above = 0, entropy_below = 0;
      for (size_t v = 0; v < values; ++v) {
        extra_total += state.extra[base + v];
        for (size_t s = 0; s < symbols_; ++s) {
          above[s] += state.hist[(base + v) * symbols_ + s];
        }
      }
      for (uint32_t count : above) entropy_above += CountCost(count);
      parent_best = std::min(parent_best,
          CountCost(state.mass[panel]) - entropy_above + extra_total);
      double mass_left = 0, extra_left = 0;
      for (size_t v = 0; v + 1 < values; ++v) {
        mass_left += state.weight[panel * values + v];
        extra_left += state.extra[base + v];
        for (size_t s = 0; s < symbols_; ++s) {
          const uint32_t delta = state.hist[(base + v) * symbols_ + s];
          if (delta == 0) continue;
          entropy_above += CountCost(above[s] - delta) - CountCost(above[s]);
          entropy_below += CountCost(below[s] + delta) - CountCost(below[s]);
          above[s] -= delta;
          below[s] += delta;
        }
        if (mass_left == 0 || mass_left == state.mass[panel]) continue;
        left[v] = std::min(left[v],
            CountCost(mass_left) - entropy_below + extra_left);
        right[v] = std::min(right[v], CountCost(state.mass[panel] - mass_left) -
                           entropy_above + extra_total - extra_left);
      }
    }
    double best = parent_best;
    for (size_t v = 0; v + 1 < values; ++v) {
      best = std::min(best, left[v] + right[v]);
    }
    return std::max(0.0, best) / state.mass[panel];
  }
  const TreeSamples& samples_;
  size_t begin_, end_, symbols_;
  std::vector<State> states_;
};

}  // namespace jxl
#endif
