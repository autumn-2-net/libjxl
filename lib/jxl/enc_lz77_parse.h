// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef LIB_JXL_ENC_LZ77_PARSE_H_
#define LIB_JXL_ENC_LZ77_PARSE_H_

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <system_error>
#include <vector>

#include "lib/jxl/base/status.h"
#include "lib/jxl/base/byte_order.h"
#include "lib/jxl/enc_lz77_bucket.h"

namespace jxl {

inline uint32_t LosslessLZ77ChainLimit(const char* text, uint32_t maximum) {
  if (!text || !*text) return maximum;
  uint32_t requested = 0;
  const char* end = text + std::strlen(text);
  const auto parsed = std::from_chars(text, end, requested);
  if (parsed.ec != std::errc{} || parsed.ptr != end ||
      requested == 0 || requested > 256) return maximum;
  return std::min(maximum, requested);
}

inline bool PreferLZ77RangeParsing(uint64_t tokens, uint64_t matches,
                                  uint64_t covered, uint64_t long_covered) {
  return tokens >= 65536 && matches != 0 && covered / matches >= 32 &&
         covered >= tokens - tokens / 2 && long_covered >= tokens / 4;
}

inline bool PreferLZ77MatchCaching(size_t tokens, size_t seed_tokens) {
  // The first pass already paid to measure repetition. Sparse/noisy streams
  // should not pay for a window-sized cache they are unlikely to reuse.
  return tokens >= 4096 && seed_tokens != 0 && seed_tokens <= tokens / 16;
}

inline Status TrimLZ77SeedPrefix(const std::vector<Token>& source, size_t prefix,
                                 const std::vector<Token>& seed, size_t minimum,
                                 std::vector<Token>* output) {
  JXL_ENSURE(prefix <= source.size());
  size_t position = 0;
  for (size_t i = 0; i < seed.size(); ++i) {
    const auto& token = seed[i];
    if (!token.is_lz77_length) {
      JXL_ENSURE(position < source.size());
      if (position >= prefix) output->push_back(source[position]);
      ++position;
      continue;
    }
    const size_t length = size_t{token.value} + minimum;
    JXL_ENSURE(i + 1 < seed.size() && length <= source.size() - position);
    const auto& distance = seed[++i];
    const size_t first = std::max(position, prefix), end = position + length;
    if (first < end) {
      if (end - first >= minimum) {
        output->emplace_back(source[first].context, end - first - minimum);
        output->back().is_lz77_length = true;
        output->push_back(distance);
      } else {
        output->insert(output->end(), source.begin() + first, source.begin() + end);
      }
    }
    position = end;
  }
  return position == source.size();
}

template <typename Value>
inline size_t LZ77MatchLength(const Value* a, const Value* b, size_t count) {
  size_t length = 0;
  while (count - length >= 8 &&
         std::memcmp(a + length, b + length, 8 * sizeof(Value)) == 0) {
    length += 8;
  }
  while (length < count && a[length] == b[length]) ++length;
  return length;
}

template <typename Value>
inline size_t LZ77WordMatchLength(const Value* a, const Value* b, size_t count) {
  static_assert(sizeof(Value) <= 8 && 8 % sizeof(Value) == 0, "whole integer samples");
  constexpr size_t kValues = 8 / sizeof(Value);
  size_t length = 0;
  // LoadLE64 is unaligned-safe. Locating the first differing byte avoids
  // rescanning a failed word; division maps that byte back to a whole symbol.
  while (count - length >= kValues) {
    const uint64_t diff = LoadLE64(reinterpret_cast<const uint8_t*>(a + length)) ^
                          LoadLE64(reinterpret_cast<const uint8_t*>(b + length));
    if (diff) return length + Num0BitsBelowLS1Bit_Nonzero(diff) / (8 * sizeof(Value));
    length += kValues;
  }
  while (length < count && a[length] == b[length]) ++length;
  return length;
}

// Queries advance monotonically. A verified match at distance d remains an
// exact prefix at later positions within that interval, including overlaps.
// This covers arbitrary periodic sequences, not only runs of one value.
class LZ77MatchCache {
 public:
  explicit LZ77MatchCache(size_t window, size_t entries = 0)
      : ends_(entries ? std::min(entries, window) : window),
        distances_(entries ? ends_.size() : 0), window_(window) {}

  void Advance(size_t pos) {
    JXL_DASSERT(pos >= position_);
    position_ = pos;
  }

  size_t Prefix(size_t distance, size_t limit) const {
    if (distance == 0 || distance >= window_) return 0;
    const size_t slot = distances_.empty() ? distance : distance & (ends_.size() - 1);
    if ((!distances_.empty() && distances_[slot] != distance) ||
        ends_[slot] <= position_) return 0;
    return std::min<size_t>(limit, ends_[slot] - position_);
  }

  void Remember(size_t distance, size_t length) {
    if (distance == 0 || distance >= window_) return;
    const size_t slot = distances_.empty() ? distance : distance & (ends_.size() - 1);
    if (!distances_.empty()) distances_[slot] = static_cast<uint32_t>(distance);
    // Saturation can only shorten a cached prefix, never invent a match.
    const size_t max = std::numeric_limits<uint32_t>::max();
    ends_[slot] = static_cast<uint32_t>(
        position_ >= max ? max : position_ + std::min(length, max - position_));
  }

 private:
  std::vector<uint32_t> ends_, distances_;
  size_t window_;
  size_t position_ = 0;
};

struct LZ77Match {
  uint32_t length;
  uint32_t distance;
};

// Previously verified matches can rule out equal-or-shorter, more costly
// candidates. A full cache loses pruning opportunities, never legal matches.
class LZ77MatchDominance {
 public:
  void Reset() { size_ = 0; overflowed_ = false; }
  bool Overflowed() const { return overflowed_; }
  template <typename Visit>
  void ForEach(const Visit& visit) const {
    for (size_t i = 0; i < size_; ++i) visit(entries_[i].length, entries_[i].distance);
  }
  size_t RequiredLength(float cost, uint32_t distance, size_t minimum) const {
    for (size_t i = size_; i != 0; --i) {
      const Entry& entry = entries_[i - 1];
      if (Cheaper(entry.cost, entry.distance, cost, distance)) {
        return std::max(minimum, size_t{entry.length} + 1);
      }
    }
    return minimum;
  }

  void Add(uint32_t length, uint32_t distance, float cost) {
    size_t begin = 0;
    while (begin < size_ &&
           (entries_[begin].cost < cost ||
            (entries_[begin].cost == cost && entries_[begin].distance < distance))) {
      if (entries_[begin].length >= length) return;
      ++begin;
    }
    if (begin < size_ && entries_[begin].cost == cost &&
        entries_[begin].distance == distance && entries_[begin].length >= length) return;
    size_t end = begin;
    while (end < size_ && entries_[end].length <= length) ++end;
    const size_t remaining = size_ - (end - begin);
    if (remaining == entries_.size()) { overflowed_ = true; return; }
    if (end != begin) {
      std::move(entries_.begin() + end, entries_.begin() + size_,
                entries_.begin() + begin);
    }
    std::move_backward(entries_.begin() + begin, entries_.begin() + remaining,
                       entries_.begin() + remaining + 1);
    entries_[begin] = {length, distance, cost};
    size_ = remaining + 1;
  }

 private:
  struct Entry {
    uint32_t length;
    uint32_t distance;
    float cost;
  };
  static bool Cheaper(float a, uint32_t ad, float b, uint32_t bd) {
    return a < b || (a == b && ad <= bd);
  }
  std::array<Entry, 16> entries_;
  size_t size_ = 0;
  bool overflowed_ = false;
};

struct LZ77MatchRange {
  size_t begin;
  size_t end;
  uint32_t distance;
  float distance_cost;
};

// A verified copy remains valid at later origins up to its original end.
// Carry only Pareto endpoints; evicting entries loses opportunities, not data.
class LZ77FrontierCarry {
 public:
  void Remember(size_t position, const LZ77MatchDominance& frontier) {
    position_ = position;
    size_ = 0;
    frontier.ForEach([&](size_t length, uint32_t symbol) {
      if (length <= std::numeric_limits<size_t>::max() - position &&
          size_ < entries_.size()) {
        entries_[size_++] = {position + length, symbol};
      }
    });
  }

  template <typename Callback>
  void Visit(size_t position, size_t minimum, size_t maximum,
             const Callback& visit) const {
    if (position < position_) return;
    for (size_t i = 0; i < size_; ++i) {
      if (entries_[i].end <= position) continue;
      const size_t length = std::min(maximum, entries_[i].end - position);
      if (length >= minimum) visit(length, entries_[i].symbol);
    }
  }

 private:
  struct Entry { size_t end; uint32_t symbol; };
  std::array<Entry, 16> entries_;
  size_t position_ = 0;
  size_t size_ = 0;
};

// The cheapest distance changes only at a match endpoint, not at every length.
template <typename Cost>
void BuildLZ77MatchRanges(std::vector<LZ77Match>* matches, size_t min_length,
                         const Cost& cost,
                         std::vector<LZ77MatchRange>* ranges) {
  ranges->clear();
  const auto better = [](const LZ77MatchRange& a, const LZ77MatchRange& b) {
    return a.distance_cost < b.distance_cost ||
        (a.distance_cost == b.distance_cost && a.distance <= b.distance);
  };
  for (const auto& match : *matches) {
    if (match.length < min_length) continue;
    const LZ77MatchRange candidate{0, size_t{match.length} + 1,
                                   match.distance, cost(match.distance)};
    size_t i = 0;
    while (i < ranges->size() && (*ranges)[i].end > candidate.end) ++i;
    if (i != 0 && better((*ranges)[i - 1], candidate)) continue;
    if (i < ranges->size() && (*ranges)[i].end == candidate.end &&
        better((*ranges)[i], candidate)) continue;
    size_t end = i;
    while (end < ranges->size() && better(candidate, (*ranges)[end])) ++end;
    ranges->erase(ranges->begin() + i, ranges->begin() + end);
    ranges->insert(ranges->begin() + i, candidate);
  }
  for (size_t i = 0; i < ranges->size(); ++i) {
    (*ranges)[i].begin = i + 1 == ranges->size() ? min_length : (*ranges)[i + 1].end;
  }
  std::reverse(ranges->begin(), ranges->end());
}

// Forward shortest-path sweep. Equal-cost match-length intervals become
// delayed offers instead of repeatedly writing every destination position.
class LZ77RangeParser {
 public:
  struct Step {
    uint32_t length = 0;
    uint32_t distance = 0;  // Zero denotes a literal, otherwise code + 1.
    uint32_t context = 0;
    float cost = std::numeric_limits<float>::infinity();
  };

  explicit LZ77RangeParser(size_t size, size_t queue_limit = 1u << 18,
                           bool calendar = false, bool vector_costs = false,
                           unsigned block_shift = 0, unsigned dyadic = 0,
                           bool range_dominance = false)
      : steps_(vector_costs ? 0 : size + 1),
        costs_(vector_costs ? size + 1 : 0, std::numeric_limits<float>::infinity()),
        lengths_(vector_costs ? size + 1 : 0),
        distances_(vector_costs ? size + 1 : 0),
        contexts_(vector_costs ? size + 1 : 0), queue_limit_(queue_limit),
        heads_(calendar ? size + 1 : 0, UINT32_MAX),
        block_shift_(block_shift),
        block_best_(block_shift ? (size >> block_shift) + 1 : 0),
        cover_costs_(range_dominance ? (size >> 4) + 1 : 0,
                     std::numeric_limits<float>::infinity()) {
    if (vector_costs) {
      relax_ = GetLZ77RelaxFunction();
      costs_[0] = 0;
    } else {
      steps_[0].cost = 0;
    }
    if (dyadic) {
      dyadic_shift_ = dyadic == 1 ? 0 : dyadic == 2 ? 6 : 4;
      while (dyadic_base_ <= (size >> dyadic_shift_)) {
        dyadic_base_ *= 2;
        ++dyadic_depth_;
      }
      dyadic_.resize(2 * dyadic_base_);
    }
  }

  Step At(size_t pos) const {
    return relax_ ? Step{lengths_[pos], distances_[pos], contexts_[pos], costs_[pos]}
                  : steps_[pos];
  }

  void Advance(size_t pos) {
    if (!dyadic_.empty()) {
      // Future-only updates cannot change an ancestor shared with the last
      // queried leaf. Recompute just the changed suffix of its root path.
      unsigned first = 0;
      const size_t block = pos >> dyadic_shift_;
      if (dyadic_previous_ != SIZE_MAX) {
        JXL_DASSERT(block >= dyadic_previous_);
        first = block == dyadic_previous_ ? dyadic_depth_ + 1
            : dyadic_depth_ - FloorLog2Nonzero(block ^ dyadic_previous_);
      }
      DyadicOffer best = first ? dyadic_path_[first - 1] : DyadicOffer{};
      for (unsigned depth = first; depth <= dyadic_depth_; ++depth) {
        const auto& item = dyadic_[(dyadic_base_ + block) >> (dyadic_depth_ - depth)];
        if (item.cost < best.cost) best = item;
        dyadic_path_[depth] = best;
      }
      dyadic_previous_ = block;
      if (best.cost != std::numeric_limits<float>::infinity())
        Relax(pos, best.origin, best.distance, best.context, best.cost);
      return;
    }
    if (!block_best_.empty()) {
      const auto& offer = block_best_[pos >> block_shift_];
      if (offer.cost != std::numeric_limits<float>::infinity())
        Relax(pos, offer.origin, offer.distance, offer.context, offer.cost);
      return;
    }
    if (!heads_.empty()) {
      while (next_time_ <= pos) {
        uint32_t id = heads_[next_time_++];
        while (id != UINT32_MAX) {
          const auto node = events_[id];
          events_[id].next = free_event_;
          free_event_ = id;
          --pending_events_;
          if (node.offer.end > pos) {
            active_.push_back(node.offer);
            std::push_heap(active_.begin(), active_.end(), HigherCost);
          }
          id = node.next;
        }
      }
    }
    while (!pending_.empty() && pending_.front().begin <= pos) {
      std::pop_heap(pending_.begin(), pending_.end(), LaterBegin);
      const Offer offer = pending_.back();
      pending_.pop_back();
      if (offer.end > pos) {
        active_.push_back(offer);
        std::push_heap(active_.begin(), active_.end(), HigherCost);
      }
    }
    while (!active_.empty() && active_.front().end <= pos) {
      std::pop_heap(active_.begin(), active_.end(), HigherCost);
      active_.pop_back();
    }
    if (active_.size() >= compact_at_) {
      active_.erase(std::remove_if(active_.begin(), active_.end(),
                                  [pos](const Offer& o) { return o.end <= pos; }),
                    active_.end());
      std::make_heap(active_.begin(), active_.end(), HigherCost);
      compact_at_ = 2 * std::max<size_t>(4096, active_.size());
    }
    if (!active_.empty()) {
      const auto& offer = active_.front();
      Relax(pos, offer.origin, offer.distance, offer.context, offer.cost);
    }
  }

  void Literal(size_t origin, uint32_t context, float cost) {
    Relax(origin + 1, origin, 0, context, cost);
  }

  // The caller has already excluded match origins in [pos, limit) and literal
  // costs are nonnegative. Once explicit endpoint writes are behind us, an
  // active constant-cost offer no dearer than the current path dominates the
  // intervening literals until the next offer event. Only the landing state is
  // needed: no later backtrack can enter the skipped states except via it.
  // Equal-cost paths may choose a different copy, with identical model cost.
  size_t JumpLiteralOnly(size_t pos, size_t limit) {
    if (relax_ || !dyadic_.empty() || !heads_.empty() || active_.empty() ||
        pos < last_explicit_ || limit <= pos) return pos;
    const auto& offer = active_.front();
    if (offer.end <= pos + 1 || offer.cost > steps_[pos].cost) return pos;
    size_t target = std::min(limit, offer.end - 1);
    if (!pending_.empty()) {
      if (pending_.front().begin <= pos + 1) return pos;
      target = std::min(target, pending_.front().begin - 1);
    }
    if (target > pos) Relax(target, offer.origin, offer.distance, offer.context, offer.cost);
    return target;
  }

  void Match(size_t origin, size_t begin, size_t end, uint32_t distance,
             uint32_t context, float cost) {
    JXL_DASSERT(begin > 0 && begin < end &&
        origin + end <= (relax_ ? costs_.size() : steps_.size()));
    ++intervals_;
    if (!cover_costs_.empty()) {
      const size_t first = origin + begin, last = origin + end;
      const size_t first_block = first >> 4, last_block = (last - 1) >> 4;
      if (last_block - first_block < 4) {
        bool covered = true;
        for (size_t b = first_block; b <= last_block; ++b) {
          if (cover_costs_[b] > cost) { covered = false; break; }
        }
        if (covered) { ++dominated_intervals_; return; }
      }
      // Only a wholly covered block is a valid certificate. Limiting updates
      // to 32 blocks bounds overhead for long copies; omitted blocks merely
      // miss pruning opportunities. The original offer is still queued/written.
      const size_t first_full = (first + 15) >> 4;
      const size_t end_full = std::min(last >> 4, first_full + 32);
      for (size_t b = first_full; b < end_full; ++b)
        cover_costs_[b] = std::min(cover_costs_[b], cost);
    }
    if (!dyadic_.empty()) {
      const DyadicOffer offer{cost, static_cast<uint32_t>(origin), distance + 1, context};
      const size_t width = size_t{1} << dyadic_shift_;
      size_t first = origin + begin, last = origin + end;
      const size_t aligned = std::min(last, (first + width - 1) & ~(width - 1));
      for (; first < aligned; ++first) {
        Relax(first, origin, distance + 1, context, cost);
        ++scalar_endpoints_;
      }
      const size_t full_end = std::max(first, last & ~(width - 1));
      for (size_t pos = full_end; pos < last; ++pos) {
        Relax(pos, origin, distance + 1, context, cost);
        ++scalar_endpoints_;
      }
      size_t left = dyadic_base_ + (first >> dyadic_shift_);
      size_t right = dyadic_base_ + (full_end >> dyadic_shift_);
      JXL_DASSERT(left == right || dyadic_previous_ == SIZE_MAX ||
                  (first >> dyadic_shift_) > dyadic_previous_);
      const auto update = [&](size_t node) {
        if (cost < dyadic_[node].cost) dyadic_[node] = offer;
      };
      while (left < right) {
        if (left & 1) update(left++);
        if (right & 1) update(--right);
        left >>= 1;
        right >>= 1;
      }
      return;
    }
    if (!block_best_.empty()) {
      // Each fully covered block needs only its cheapest constant-cost offer.
      // Partial boundaries remain explicit. No event heap or match is dropped.
      const size_t width = size_t{1} << block_shift_;
      size_t first = origin + begin, last = origin + end;
      const size_t aligned = std::min(last, (first + width - 1) & ~(width - 1));
      for (; first < aligned; ++first) {
        Relax(first, origin, distance + 1, context, cost);
        ++scalar_endpoints_;
      }
      const Offer offer{origin, origin + begin, origin + end, distance + 1, context, cost};
      for (; last - first >= width; first += width) {
        auto& best = block_best_[first >> block_shift_];
        if (cost < best.cost) best = offer;
      }
      for (; first < last; ++first) {
        Relax(first, origin, distance + 1, context, cost);
        ++scalar_endpoints_;
      }
      return;
    }
    if (end - begin <= 16 || Queued() >= queue_limit_) {
      // Short intervals cost less to update directly. The same operation also
      // bounds queue memory without dropping matches or using timing vetoes.
      scalar_endpoints_ += end - begin;
      last_explicit_ = std::max(last_explicit_, origin + end - 1);
      if (relax_) {
        const size_t pos = origin + begin;
        relax_(costs_.data() + pos, lengths_.data() + pos,
                distances_.data() + pos, contexts_.data() + pos,
                end - begin, begin, distance + 1, context, cost);
        return;
      }
      for (size_t length = begin; length < end; ++length) {
        Relax(origin + length, origin, distance + 1, context, cost);
      }
      return;
    }
    const Offer offer{origin, origin + begin, origin + end,
                       distance + 1, context, cost};
    if (heads_.empty()) {
      pending_.push_back(offer);
      std::push_heap(pending_.begin(), pending_.end(), LaterBegin);
    } else {
      uint32_t id = free_event_;
      if (id == UINT32_MAX) {
        id = static_cast<uint32_t>(events_.size());
        events_.push_back({offer, heads_[offer.begin]});
      } else {
        free_event_ = events_[id].next;
        events_[id] = {offer, heads_[offer.begin]};
      }
      heads_[offer.begin] = id;
      ++pending_events_;
    }
    peak_queued_ = std::max(peak_queued_, Queued());
  }

  size_t PeakQueued() const { return peak_queued_; }
  uint64_t Intervals() const { return intervals_; }
  uint64_t ScalarEndpoints() const { return scalar_endpoints_; }
  uint64_t DominatedIntervals() const { return dominated_intervals_; }

 private:
  struct DyadicOffer {
    float cost = std::numeric_limits<float>::infinity();
    uint32_t origin = 0, distance = 0, context = 0;
  };
  struct Offer {
    size_t origin = 0;
    size_t begin = 0;
    size_t end = 0;
    uint32_t distance = 0;
    uint32_t context = 0;
    float cost = std::numeric_limits<float>::infinity();
  };
  struct Event { Offer offer; uint32_t next; };
  size_t Queued() const {
    return pending_.size() + pending_events_ + active_.size();
  }
  static bool LaterBegin(const Offer& a, const Offer& b) {
    return a.begin > b.begin;
  }
  static bool HigherCost(const Offer& a, const Offer& b) {
    return a.cost > b.cost || (a.cost == b.cost && a.origin > b.origin);
  }
  void Relax(size_t pos, size_t origin, uint32_t distance, uint32_t context,
             float cost) {
    if (relax_) {
      if (cost < costs_[pos]) {
        costs_[pos] = cost;
        lengths_[pos] = static_cast<uint32_t>(pos - origin);
        distances_[pos] = distance;
        contexts_[pos] = context;
      }
      return;
    }
    if (cost < steps_[pos].cost) {
      steps_[pos] = {static_cast<uint32_t>(pos - origin), distance, context, cost};
    }
  }
  std::vector<Step> steps_;
  std::vector<float> costs_;
  std::vector<uint32_t> lengths_, distances_, contexts_;
  LZ77RelaxFunction relax_ = nullptr;
  std::vector<Offer> pending_;
  std::vector<Offer> active_;
  size_t queue_limit_;
  // Integer destinations permit O(1) scheduling instead of a begin-time heap.
  // Slots are recycled after activation; expired offers still leave the cost
  // heap using the existing exact sweep, without approximating match costs.
  std::vector<uint32_t> heads_;
  std::vector<Event> events_;
  size_t next_time_ = 0, pending_events_ = 0;
  uint32_t free_event_ = UINT32_MAX;
  size_t compact_at_ = 4096;
  size_t peak_queued_ = 0;
  uint64_t intervals_ = 0;
  uint64_t scalar_endpoints_ = 0;
  size_t last_explicit_ = 0;
  unsigned block_shift_ = 0;
  std::vector<Offer> block_best_;
  size_t dyadic_base_ = 1, dyadic_previous_ = SIZE_MAX;
  unsigned dyadic_depth_ = 0;
  unsigned dyadic_shift_ = 0;
  std::vector<DyadicOffer> dyadic_;
  std::array<DyadicOffer, sizeof(size_t) * 8> dyadic_path_{};
  std::vector<float> cover_costs_;
  uint64_t dominated_intervals_ = 0;
};

}  // namespace jxl

#endif  // LIB_JXL_ENC_LZ77_PARSE_H_
