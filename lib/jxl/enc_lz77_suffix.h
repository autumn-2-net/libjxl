// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef LIB_JXL_ENC_LZ77_SUFFIX_H_
#define LIB_JXL_ENC_LZ77_SUFFIX_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "lib/jxl/base/bits.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_lz77_cost.h"
#include "lib/jxl/enc_lz77_parse.h"
#include "lib/jxl/enc_lz77_lcp.h"

namespace jxl {

bool LosslessSuffixArrayAvailable();
int BuildLosslessSuffixArray(int32_t* values, int32_t* suffixes, int32_t size,
                            int32_t alphabet, int32_t scratch);

// Active suffix ranks in a sliding window. Each upper level marks nonempty
// words below it, giving predecessor/successor queries without tree allocation.
class LZ77ActiveRanks {
 public:
  explicit LZ77ActiveRanks(size_t size) {
    do {
      size = (size + 63) / 64;
      levels_.emplace_back(std::max<size_t>(1, size), 0);
    } while (size > 1);
  }

  void Set(size_t rank, bool active) {
    for (auto& level : levels_) {
      const size_t word = rank / 64;
      const uint64_t mask = uint64_t{1} << (rank % 64);
      const bool was_empty = level[word] == 0;
      if (active) level[word] |= mask;
      else level[word] &= ~mask;
      if (was_empty == (level[word] == 0)) break;
      rank = word;
    }
  }

  size_t Before(size_t rank) const { return BeforeAt(0, rank); }
  size_t After(size_t rank) const { return AtOrAfter(0, rank + 1); }

 private:
  size_t BeforeAt(size_t level, size_t rank) const {
    if (!rank || level == levels_.size()) return SIZE_MAX;
    const size_t word = (rank - 1) / 64;
    const size_t bit = (rank - 1) % 64;
    const uint64_t mask = UINT64_MAX >> (63 - bit);
    const uint64_t present = levels_[level][word] & mask;
    if (present) return word * 64 + FloorLog2Nonzero(present);
    const size_t previous = BeforeAt(level + 1, word);
    return previous == SIZE_MAX ? SIZE_MAX
        : previous * 64 + FloorLog2Nonzero(levels_[level][previous]);
  }

  size_t AtOrAfter(size_t level, size_t rank) const {
    if (level == levels_.size()) return SIZE_MAX;
    const size_t word = rank / 64;
    if (word >= levels_[level].size()) return SIZE_MAX;
    const uint64_t present = levels_[level][word] & (UINT64_MAX << (rank % 64));
    if (present) return word * 64 + Num0BitsBelowLS1Bit_Nonzero(present);
    const size_t next = AtOrAfter(level + 1, word + 1);
    return next == SIZE_MAX ? SIZE_MAX
        : next * 64 + Num0BitsBelowLS1Bit_Nonzero(levels_[level][next]);
  }

  std::vector<std::vector<uint64_t>> levels_;
};

// Lexicographically adjacent earlier suffixes nominate long matches. Recent
// short-prefix and image-stride references retain inexpensive shorter copies.
// This is a different match search, not a claim of optimal parsing in a window.
class LZ77SuffixIndex {
 public:
  bool Build(const Token* input, size_t size, size_t window, size_t minimum,
             size_t width, bool sliding = false, bool precompute_lcp = false) {
    if (!LosslessSuffixArrayAvailable() || !input || !size || !minimum ||
        size > (1u << 24) || window == 0 || window > UINT32_MAX) return false;
    uint32_t largest = 0;
    for (size_t i = 0; i < size; ++i) largest = std::max(largest, input[i].value);
    if (largest >= (1u << 20)) return false;
    window_ = window;
    minimum_ = minimum;
    special_count_ = 0;
    special_.clear();
    symbols_.clear();
    anchors_.clear();
    values_.resize(size);
    zeros_.resize(size);
    before_.assign(size, UINT32_MAX);
    after_.assign(size, UINT32_MAX);
    recent_.assign(size, UINT32_MAX);
    before_length_.clear();
    after_length_.clear();
    for (size_t i = 0; i < size; ++i) values_[i] = input[i].value;
    const size_t scratch = 6 * (largest + 1);
    std::vector<int32_t> suffixes(size + scratch);
    if (BuildLosslessSuffixArray(reinterpret_cast<int32_t*>(values_.data()),
          suffixes.data(), size, largest + 1, scratch) != 0) return false;
    std::vector<uint32_t> ranks;
    if (sliding || precompute_lcp) {
      ranks.resize(size);
      for (size_t rank = 0; rank < size; ++rank) ranks[suffixes[rank]] = rank;
    }
    if (sliding) {
      LZ77ActiveRanks active(size);
      for (size_t pos = 0; pos < size; ++pos) {
        if (pos) active.Set(ranks[pos - 1], true);
        if (pos >= window) active.Set(ranks[pos - window], false);
        const size_t before = active.Before(ranks[pos]);
        const size_t after = active.After(ranks[pos]);
        if (before != SIZE_MAX) before_[pos] = suffixes[before];
        if (after != SIZE_MAX) after_[pos] = suffixes[after];
      }
    } else {
      std::vector<uint32_t> stack;
      stack.reserve(size);
      for (size_t rank = 0; rank < size; ++rank) {
        const uint32_t pos = suffixes[rank];
        while (!stack.empty() && stack.back() > pos) {
          after_[stack.back()] = pos;
          stack.pop_back();
        }
        if (!stack.empty()) before_[pos] = stack.back();
        stack.push_back(pos);
      }
    }
    if (precompute_lcp) {
      LZ77SuffixLcp lcp;
      lcp.Build(values_, suffixes.data(), ranks);
      before_length_.resize(size);
      after_length_.resize(size);
      for (size_t pos = 0; pos < size; ++pos) {
        if (before_[pos] != UINT32_MAX)
          before_length_[pos] = lcp.Between(ranks[pos], ranks[before_[pos]]);
        if (after_[pos] != UINT32_MAX)
          after_length_[pos] = lcp.Between(ranks[pos], ranks[after_[pos]]);
      }
    }
    std::vector<uint32_t> heads(1u << 16, UINT32_MAX);
    for (size_t i = 0; i + 3 <= size; ++i) {
      uint32_t key = values_[i] * 0x85ebca6bu;
      key ^= values_[i + 1] * 0xc2b2ae35u;
      key ^= values_[i + 2] * 0x9e3779b9u;
      key = (key ^ (key >> 16)) & 65535;
      recent_[i] = heads[key];
      heads[key] = i;
    }
    uint32_t zeros = 0;
    for (size_t i = size; i; --i) {
      zeros = values_[i - 1] ? 0 : std::min<size_t>(window, zeros + 1);
      zeros_[i - 1] = zeros;
    }
    if (width) {
      special_count_ = kNumSpecialDistances;
      int largest_distance = 0;
      for (int i = kNumSpecialDistances - 1; i >= 0; --i) {
        const int distance = SpecialDistance(i, width);
        special_[distance] = i;
        largest_distance = std::max(largest_distance, distance);
      }
      if (largest_distance > 0 && largest_distance < 65536) {
        symbols_.assign(largest_distance + 1, -1);
        for (const auto& entry : special_) {
          if (entry.first > 0) symbols_[entry.first] = entry.second;
        }
      }
      for (size_t i = 0; i < 24; ++i) {
        const int distance = SpecialDistance(i, width);
        if (distance > 0 && static_cast<size_t>(distance) < window)
          anchors_.push_back(distance);
      }
    }
    return true;
  }

  template <typename Cost, typename Visit>
  void FindMatches(size_t pos, const Cost& cost, const Visit& visit,
                   LZ77MatchCache* cache = nullptr, size_t maximum = SIZE_MAX,
                   size_t hint_distance = 0,
                   LZ77MatchDominance* result = nullptr,
                   LZ77DistanceBound bound = {},
                   const LZ77FrontierCarry* carry = nullptr) const {
    (void)bound;
    if (result) result->Reset();
    if (pos >= values_.size()) return;
    const size_t limit = std::min(maximum, values_.size() - pos);
    if (limit < minimum_) return;
    if (cache) cache->Advance(pos);
    LZ77MatchDominance local_frontier;
    auto& frontier = result ? *result : local_frontier;
    if (carry) {
      carry->Visit(pos, minimum_, limit, [&](size_t length, uint32_t symbol) {
        visit(length, symbol);
        frontier.Add(static_cast<uint32_t>(length), symbol, cost(symbol));
      });
    }
    const auto candidate = [&](uint32_t previous, size_t known = SIZE_MAX) {
      if (previous >= pos || pos - previous >= window_ ||
          values_[previous] != values_[pos] ||
          values_[previous + minimum_ - 1] != values_[pos + minimum_ - 1]) return;
      const size_t distance = pos - previous;
      uint32_t symbol = special_count_ + distance - 1;
      if (!symbols_.empty()) {
        if (distance < symbols_.size() && symbols_[distance] >= 0) symbol = symbols_[distance];
      } else {
        const auto found = special_.find(distance);
        if (found != special_.end()) symbol = found->second;
      }
      const float bits = cost(symbol);
      const size_t required = frontier.RequiredLength(bits, symbol, minimum_);
      if (required > limit || values_[pos + required - 1] != values_[previous + required - 1]) return;
      size_t length;
      if (known != SIZE_MAX) {
        length = std::min(limit, known);
      } else {
        length = std::min<size_t>(limit, std::min(zeros_[pos], zeros_[previous]));
        if (cache) length = std::max(length, cache->Prefix(distance, limit));
        length += LZ77MatchLength(values_.data() + pos + length,
                                  values_.data() + previous + length, limit - length);
      }
      if (cache) cache->Remember(distance, length);
      if (length < required) return;
      visit(length, symbol);
      frontier.Add(length, symbol, bits);
    };
    if (hint_distance && hint_distance <= pos) candidate(pos - hint_distance);
    candidate(before_[pos], before_length_.empty() ? SIZE_MAX : before_length_[pos]);
    candidate(after_[pos], after_length_.empty() ? SIZE_MAX : after_length_[pos]);
    uint32_t previous = recent_[pos];
    for (size_t i = 0; i < 8 && previous < pos && pos - previous < window_; ++i) {
      candidate(previous);
      previous = recent_[previous];
    }
    for (size_t distance : anchors_) {
      if (distance <= pos) candidate(pos - distance);
    }
  }

 private:
  size_t window_ = 0, minimum_ = 3, special_count_ = 0;
  std::vector<uint32_t> values_, zeros_, before_, after_, recent_, anchors_;
  std::vector<uint32_t> before_length_, after_length_;
  std::vector<int16_t> symbols_;
  std::unordered_map<int, uint32_t> special_;
};

}  // namespace jxl
#endif
