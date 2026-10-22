// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef LIB_JXL_ENC_LZ77_INDEX_H_
#define LIB_JXL_ENC_LZ77_INDEX_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_lossless_workers.h"
#include "lib/jxl/enc_lz77_bucket.h"
#include "lib/jxl/enc_lz77_cost.h"
#include "lib/jxl/enc_lz77_deep.h"
#include "lib/jxl/enc_lz77_parse.h"

namespace jxl {

// A stable counting-sort index reproduces the recent-position bucket windows
// without mutable insertion state. Different positions can then be matched
// concurrently; every match still refers strictly to earlier symbols.
template <size_t kShort = 32, bool kFullPrefix = true, size_t kLong = 32,
          typename Value = uint32_t>
class LZ77StaticIndex {
 public:
  LZ77StaticIndex(const Token* input, size_t size, size_t window,
                  size_t minimum, size_t width, bool tags = false,
                  bool simd_tags = false, unsigned run_keys = 0,
                  bool coordinates = false, bool anchors_first = false,
                  unsigned deep_prefix = 0, bool defer_build = false,
                  bool word_match = false)
      : values_(size),
        zeros_(size),
        window_(window),
        minimum_(minimum),
        run_keys_(run_keys),
        deep_prefix_(deep_prefix),
        anchors_first_(anchors_first),
        word_match_(word_match),
        short_slots_(size),
        long_slots_(size),
        short_positions_(size),
        long_positions_(size),
        short_offsets_(kHashes + 1),
        long_offsets_(kHashes + 1),
        short_tags_(tags ? size : 0),
        long_tags_(tags ? size : 0),
        counts_(coordinates ? size : 0) {
    static_assert(kShort <= 255 && kLong <= 255, "bucket count fits in a byte");
    JXL_DASSERT(size <= UINT32_MAX);
    if (tags && simd_tags) filter_ = GetLZ77FingerprintFunction();
    if (!defer_build) {
      for (size_t i = 0; i < size; ++i) values_[i] = input[i].value;
      if (deep_prefix_) {
        if constexpr (std::is_same_v<Value, uint32_t>) {
          deep16_.Build(values_, 16);
          deep64_.Build(values_, 64);
        } else {
          const std::vector<uint32_t> wide(values_.begin(), values_.end());
          deep16_.Build(wide, 16);
          deep64_.Build(wide, 64);
        }
      }
      uint32_t zeros = 0;
      for (size_t i = size; i != 0; --i) {
        zeros = values_[i - 1] == 0 ? std::min<size_t>(window_, zeros + 1) : 0;
        zeros_[i - 1] = zeros;
      }
      for (size_t i = 0; i < size; ++i) {
        const uint32_t hash = ShortHash(i);
        ++short_offsets_[(hash & kMask) + 1];
        short_slots_[i] = hash;
        if (i + 8 <= size) {
          const uint32_t key = LongHash(i, hash);
          if (!SharedRunSlot(i)) ++long_offsets_[(key & kMask) + 1];
          long_slots_[i] = key;
        }
      }
      for (size_t i = 1; i <= kHashes; ++i) {
        short_offsets_[i] += short_offsets_[i - 1];
        long_offsets_[i] += long_offsets_[i - 1];
      }
      auto short_next = short_offsets_;
      auto long_next = long_offsets_;
      for (size_t i = 0; i < size; ++i) {
        const uint32_t hash = short_slots_[i];
        const uint32_t slot = short_next[hash & kMask]++;
        short_slots_[i] = slot;
        short_positions_[slot] = static_cast<uint32_t>(i);
        if (tags) short_tags_[slot] = static_cast<uint8_t>(hash >> 24);
        if (coordinates)
          counts_[i] =
              std::min<size_t>(kShort, slot + 1 - short_offsets_[hash & kMask]);
        if (i + 8 <= size) {
          if (SharedRunSlot(i)) {
            long_slots_[i] = long_slots_[i - 1];
            if (coordinates) counts_[i] |= counts_[i - 1] & 0xff00;
            continue;
          }
          const uint32_t key = long_slots_[i];
          const uint32_t long_slot = long_next[key & kMask]++;
          long_slots_[i] = long_slot;
          long_positions_[long_slot] = static_cast<uint32_t>(i);
          if (tags) long_tags_[long_slot] = static_cast<uint8_t>(hash >> 24);
          if (coordinates)
            counts_[i] |= std::min<size_t>(
                              kLong, long_slot + 1 - long_offsets_[key & kMask])
                          << 8;
        }
      }
    }
    if (width) {
      special_count_ = kNumSpecialDistances;
      int largest = 0;
      for (int i = kNumSpecialDistances - 1; i >= 0; --i) {
        const int distance = SpecialDistance(i, width);
        special_[distance] = i;
        largest = std::max(largest, distance);
      }
      if (largest > 0 && largest < 65536) {
        distance_symbols_.assign(static_cast<size_t>(largest) + 1, -1);
        for (const auto& pair : special_) {
          if (pair.first > 0) distance_symbols_[pair.first] = pair.second;
        }
      }
      for (size_t i = 0; i < 24; ++i) {
        const int distance = SpecialDistance(i, width);
        if (distance > 0 && static_cast<size_t>(distance) < window_) {
          anchors_.push_back(distance);
        }
      }
    }
  }

  Status BuildParallel(const Token* input, ThreadPool* pool) {
    JXL_ENSURE(run_keys_ != 2 && deep_prefix_ == 0);
    const size_t size = values_.size();
    const size_t parts = std::max<size_t>(
        1,
        LosslessWorkerLimits::FromEnvironment().Tasks(
            std::max<size_t>(1, size / 16384), 2 * kHashes * sizeof(uint32_t)));
    std::vector<std::vector<uint32_t>> short_counts(
        parts, std::vector<uint32_t>(kHashes));
    std::vector<std::vector<uint32_t>> long_counts(
        parts, std::vector<uint32_t>(kHashes));
    std::vector<size_t> tails(parts), carry(parts);
    JXL_RETURN_IF_ERROR(RunOnPool(
        pool, 0, parts, ThreadPool::NoInit,
        [&](uint32_t part, size_t) -> Status {
          const size_t begin = size * part / parts,
                       end = size * (part + 1) / parts;
          uint32_t zeros = 0;
          for (size_t i = end; i > begin; --i) {
            values_[i - 1] = input[i - 1].value;
            zeros =
                values_[i - 1] == 0 ? std::min<size_t>(window_, zeros + 1) : 0;
            zeros_[i - 1] = zeros;
          }
          size_t tail = end;
          while (tail > begin && values_[tail - 1] == 0) --tail;
          tails[part] = end - tail;
          return true;
        },
        "LosslessIndexValues"));
    size_t trailing = 0;
    for (size_t part = parts; part != 0; --part) {
      carry[part - 1] = trailing;
      const size_t count = size * part / parts - size * (part - 1) / parts;
      const size_t prefix = count ? zeros_[size * (part - 1) / parts] : 0;
      trailing = std::min(window_, prefix == count ? trailing + count : prefix);
    }
    JXL_RETURN_IF_ERROR(RunOnPool(
        pool, 0, parts, ThreadPool::NoInit,
        [&](uint32_t part, size_t) -> Status {
          const size_t begin = size * part / parts,
                       end = size * (part + 1) / parts;
          for (size_t i = end - tails[part]; i < end; ++i)
            zeros_[i] = std::min(window_, size_t{zeros_[i]} + carry[part]);
          for (size_t i = begin; i < end; ++i) {
            const uint32_t hash = ShortHash(i);
            short_slots_[i] = hash;
            ++short_counts[part][hash & kMask];
            if (i + 8 <= size) {
              const uint32_t key = LongHash(i, hash);
              long_slots_[i] = key;
              ++long_counts[part][key & kMask];
            }
          }
          return true;
        },
        "LosslessIndexCounts"));
    // Stable counting sort: prefix each bucket across source-order chunks.
    // Chunk workers own disjoint scatter ranges, including equal hash keys.
    std::fill(short_offsets_.begin(), short_offsets_.end(), 0);
    std::fill(long_offsets_.begin(), long_offsets_.end(), 0);
    for (size_t part = 0; part < parts; ++part) {
      for (size_t hash = 0; hash < kHashes; ++hash) {
        short_offsets_[hash + 1] += short_counts[part][hash];
        long_offsets_[hash + 1] += long_counts[part][hash];
      }
    }
    for (size_t hash = 1; hash <= kHashes; ++hash) {
      short_offsets_[hash] += short_offsets_[hash - 1];
      long_offsets_[hash] += long_offsets_[hash - 1];
    }
    auto short_next = short_offsets_, long_next = long_offsets_;
    for (size_t part = 0; part < parts; ++part) {
      for (size_t hash = 0; hash < kHashes; ++hash) {
        const uint32_t a = short_counts[part][hash],
                       b = long_counts[part][hash];
        short_counts[part][hash] = short_next[hash];
        long_counts[part][hash] = long_next[hash];
        short_next[hash] += a;
        long_next[hash] += b;
      }
    }
    return RunOnPool(
        pool, 0, parts, ThreadPool::NoInit,
        [&](uint32_t part, size_t) -> Status {
          const size_t begin = size * part / parts,
                       end = size * (part + 1) / parts;
          for (size_t i = begin; i < end; ++i) {
            const uint32_t hash = short_slots_[i];
            const uint32_t slot = short_counts[part][hash & kMask]++;
            short_slots_[i] = slot;
            short_positions_[slot] = i;
            if (!short_tags_.empty()) short_tags_[slot] = hash >> 24;
            if (!counts_.empty())
              counts_[i] = std::min<size_t>(
                  kShort, slot + 1 - short_offsets_[hash & kMask]);
            if (i + 8 <= size) {
              const uint32_t key = long_slots_[i];
              const uint32_t long_slot = long_counts[part][key & kMask]++;
              long_slots_[i] = long_slot;
              long_positions_[long_slot] = i;
              if (!long_tags_.empty()) long_tags_[long_slot] = hash >> 24;
              if (!counts_.empty())
                counts_[i] |=
                    std::min<size_t>(kLong,
                                     long_slot + 1 - long_offsets_[key & kMask])
                    << 8;
            }
          }
          return true;
        },
        "LosslessIndexScatter");
  }

  template <typename Cost, typename Visit>
  void FindMatches(size_t pos, const Cost& cost, const Visit& visit,
                   LZ77MatchCache* cache = nullptr, size_t maximum = SIZE_MAX,
                   size_t hint_distance = 0,
                   LZ77MatchDominance* result = nullptr,
                   LZ77DistanceBound bound = {},
                   const LZ77FrontierCarry* carry = nullptr) const {
    if (result) result->Reset();
    if (pos >= values_.size()) return;
    const size_t limit = std::min(maximum, values_.size() - pos);
    if (limit < minimum_) return;
    if (cache) cache->Advance(pos);
    LZ77MatchDominance local_frontier;
    auto& frontier = result ? *result : local_frontier;
    bool complete = false;
    if (carry) {
      carry->Visit(pos, minimum_, limit, [&](size_t length, uint32_t symbol) {
        const float bits = cost(symbol);
        visit(length, symbol);
        frontier.Add(static_cast<uint32_t>(length), symbol, bits);
        complete |= length == limit &&
            (bits < bound.cost || (bits == bound.cost && symbol <= bound.symbol));
      });
      if (complete) return;
    }
    const auto candidate = [&](uint32_t previous) {
      if (previous >= pos || pos - previous >= window_ ||
          values_[pos] != values_[previous] ||
          values_[pos + minimum_ - 1] != values_[previous + minimum_ - 1])
        return;
      const size_t distance = pos - previous;
      uint32_t symbol = static_cast<uint32_t>(special_count_ + distance - 1);
      if (!distance_symbols_.empty()) {
        if (distance < distance_symbols_.size() &&
            distance_symbols_[distance] >= 0) {
          symbol = distance_symbols_[distance];
        }
      } else {
        const auto found = special_.find(static_cast<int>(distance));
        if (found != special_.end()) symbol = found->second;
      }
      const float bits = cost(symbol);
      const size_t required = frontier.RequiredLength(bits, symbol, minimum_);
      if (required > limit ||
          values_[pos + required - 1] != values_[previous + required - 1])
        return;
      size_t length = std::min<size_t>(limit, std::min(zeros_[pos], zeros_[previous]));
      if (cache) length = std::max(length, cache->Prefix(distance, limit));
      if (word_match_) {
        length += LZ77WordMatchLength(values_.data() + pos + length,
            values_.data() + previous + length, limit - length);
      } else {
        length += LZ77MatchLength(values_.data() + pos + length,
            values_.data() + previous + length, limit - length);
      }
      if (cache) cache->Remember(distance, length);
      if (length < required) return;
      visit(length, symbol);
      frontier.Add(static_cast<uint32_t>(length), symbol, bits);
      complete = length == limit &&
          (bits < bound.cost || (bits == bound.cost && symbol <= bound.symbol));
    };
    // A first-pass reference is an extra candidate, not an assumed match.
    // Validate it through the same bounds and symbol comparisons as the index.
    if (hint_distance && hint_distance <= pos) {
      candidate(static_cast<uint32_t>(pos - hint_distance));
      if (complete) return;
    }
    const uint32_t hash = counts_.empty() ? ShortHash(pos) : 0;
    const uint32_t slot = short_slots_[pos];
    const uint8_t tag = short_tags_.empty() ? 0 : short_tags_[slot];
    const size_t count = counts_.empty()
        ? std::min<size_t>(kShort, slot + 1 - short_offsets_[hash & kMask])
        : counts_[pos] & 255;
    const uint32_t oldest = count == kShort ? short_positions_[slot + 1 - count] : 0;
    const auto long_candidate = [&](uint32_t previous) {
      if (previous >= oldest) return;
      if (run_keys_ == 2 && ZeroSuffix(pos) && ZeroSuffix(previous) &&
          zeros_[previous] >= zeros_[pos]) {
        // Align run ends so a different-length zero prefix can reuse the same
        // following nonzero sequence. candidate() still verifies every symbol.
        previous += zeros_[previous] - zeros_[pos];
      }
      candidate(previous);
    };
    const auto anchors = [&]() {
      for (uint32_t distance : anchors_) {
        if (distance <= pos && pos - distance < oldest)
          candidate(static_cast<uint32_t>(pos - distance));
        if (complete) break;
      }
    };
    if (anchors_first_) anchors();
    if (complete) return;
    if (deep_prefix_) {
      const size_t visits = deep_prefix_ == 1 ? 4 : 8;
      deep64_.Find(pos, window_, visits, candidate);
      deep16_.Find(pos, window_, visits, candidate);
    }
    if (filter_) {
      const size_t start = slot + 1 - count;
      // Include the current suffix for a full SIMD lane, then mask it out.
      uint64_t mask = filter_(short_tags_.data() + start, count, tag);
      mask &= ~(uint64_t{1} << (count - 1));
      while (mask) {
        const size_t bit = FloorLog2Nonzero(mask);
        mask ^= uint64_t{1} << bit;
        candidate(short_positions_[start + bit]);
        if (complete) return;
      }
    } else {
      for (size_t i = 1; i < count; ++i) {
        if (short_tags_.empty() || short_tags_[slot - i] == tag)
          candidate(short_positions_[slot - i]);
        if (complete) return;
      }
    }
    if (pos + 8 <= values_.size()) {
      const uint32_t long_slot = long_slots_[pos];
      const size_t available = counts_.empty() ? std::min<size_t>(
          kLong, long_slot + 1 - long_offsets_[LongHash(pos, hash) & kMask])
          : counts_[pos] >> 8;
      const size_t long_count = deep_prefix_ == 2 ? std::min<size_t>(9, available)
                                                 : available;
      if (filter_) {
        const size_t start = long_slot + 1 - long_count;
        uint64_t mask = filter_(long_tags_.data() + start, long_count, tag);
        mask &= ~(uint64_t{1} << (long_count - 1));
        while (mask) {
          const size_t bit = FloorLog2Nonzero(mask);
          mask ^= uint64_t{1} << bit;
          const uint32_t previous = long_positions_[start + bit];
          long_candidate(previous);
          if (complete) return;
        }
      } else {
        for (size_t i = 1; i < long_count; ++i) {
          if (!long_tags_.empty() && long_tags_[long_slot - i] != tag) continue;
          const uint32_t previous = long_positions_[long_slot - i];
          long_candidate(previous);
          if (complete) return;
        }
      }
    }
    if (!anchors_first_) anchors();
  }

 private:
  bool ZeroSuffix(size_t pos) const {
    return zeros_[pos] >= 8 && zeros_[pos] < window_ &&
           pos + zeros_[pos] + 3 <= values_.size();
  }
  bool SharedRunSlot(size_t pos) const {
    return run_keys_ == 2 && pos != 0 && ZeroSuffix(pos) &&
           ZeroSuffix(pos - 1) && zeros_[pos - 1] == zeros_[pos] + 1;
  }
  uint32_t ShortHash(size_t pos) const {
    if (pos + 3 > values_.size()) return 0;
    uint32_t hash = 0;
    for (size_t i = 0; i < 3; ++i) {
      hash ^= values_[pos + i] + 0x9e3779b9u + (hash << 6) + (hash >> 2);
    }
    hash ^= hash >> 16;
    hash *= 0x85ebca6bu;
    hash ^= hash >> 13;
    hash *= 0xc2b2ae35u;
    return hash ^ (hash >> 16);
  }
  uint32_t LongHash(size_t pos, uint32_t hash) const {
    if ((run_keys_ == 1 && zeros_[pos] >= 8) ||
        (run_keys_ == 2 && ZeroSuffix(pos))) {
      // Index the nonzero suffix rather than placing every interior position
      // of a long zero run into one bucket. Short-prefix history remains intact.
      if (run_keys_ == 1) hash ^= zeros_[pos] * 0x9e3779b9u;
      const size_t next = pos + zeros_[pos];
      for (size_t i = 0; i < 3 && next + i < values_.size(); ++i) {
        hash ^= values_[next + i] + 0x9e3779b9u + (hash << 6) + (hash >> 2);
      }
      hash ^= hash >> 16;
      hash *= 0x85ebca6bu;
      return hash ^ (hash >> 13);
    }
    hash ^= values_[pos + 3] * 0x85ebca6bu;
    hash ^= values_[pos + 7] * 0xc2b2ae35u;
    if constexpr (kFullPrefix) {
      hash ^= values_[pos + 4] * 0x9e3779b9u;
      hash ^= values_[pos + 5] * 0x27d4eb2du;
      hash ^= values_[pos + 6] * 0x165667b1u;
    }
    return hash ^ (hash >> 16);
  }
  static constexpr size_t kHashes = 1u << 15;
  static constexpr uint32_t kMask = kHashes - 1;
  LZ77FingerprintFunction filter_ = nullptr;
  std::vector<Value> values_;
  std::vector<uint32_t> zeros_;
  size_t window_, minimum_, special_count_ = 0;
  unsigned run_keys_, deep_prefix_;
  LZ77DeepPrefixChain deep16_, deep64_;
  bool anchors_first_;
  bool word_match_;
  std::vector<uint32_t> short_slots_, long_slots_, short_positions_, long_positions_;
  std::vector<uint32_t> short_offsets_, long_offsets_;
  std::vector<uint8_t> short_tags_, long_tags_;
  std::vector<uint16_t> counts_;
  std::vector<uint32_t> anchors_;
  std::vector<int16_t> distance_symbols_;
  std::unordered_map<int, uint32_t> special_;
};

}  // namespace jxl

#endif  // LIB_JXL_ENC_LZ77_INDEX_H_
