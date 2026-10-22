// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef LIB_JXL_ENC_LZ77_BUCKET_H_
#define LIB_JXL_ENC_LZ77_BUCKET_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "hwy/base.h"
#include "lib/jxl/base/bits.h"

namespace jxl {

using LZ77FingerprintFunction =
    uint64_t (*)(const uint8_t*, size_t, uint8_t);
LZ77FingerprintFunction GetLZ77FingerprintFunction();
struct LZ77SpecialPrefixGroup {
  uint32_t max_distance;
  uint16_t valid;
};
// Tags must have 16 readable padding bytes after the token stream.
using LZ77SpecialPrefixFunction = void (*)(
    const uint8_t*, size_t, size_t, uint8_t,
    const LZ77SpecialPrefixGroup*, size_t, uint16_t*);
LZ77SpecialPrefixFunction GetLZ77SpecialPrefixFunction();
using LZ77RelaxFunction = void (*)(float*, uint32_t*, uint32_t*, uint32_t*,
                                 size_t, uint32_t, uint32_t, uint32_t, float);
LZ77RelaxFunction GetLZ77RelaxFunction();

template <size_t N>
void LZ77InsertRepeatedBucket(std::array<uint32_t, N>* positions,
                              uint16_t* next, uint16_t* size,
                              size_t begin, size_t end) {
  const size_t count = end - begin;
  const size_t retained = std::min(N, count);
  const size_t first = (*next + count - retained) % N;
  for (size_t i = 0; i < retained; ++i)
    (*positions)[(first + i) % N] = static_cast<uint32_t>(end - retained + i);
  *next = static_cast<uint16_t>((*next + count) % N);
  *size = static_cast<uint16_t>(std::min<size_t>(N, *size + count));
}

// Contiguous recent-position queues, independently bounded per prefix.
template <size_t kCapacity>
class LZ77RecentBuckets {
 public:
  static_assert(kCapacity != 0 && (kCapacity & (kCapacity - 1)) == 0, "power of two");
  LZ77RecentBuckets() : buckets_(1u << 15) {}
  // Enable before the first Insert; positions alone cannot reconstruct tags.
  void EnableFingerprints(bool measure) {
    if constexpr (kCapacity <= 64) {
      fingerprints_.resize(buckets_.size());
      filter_ = GetLZ77FingerprintFunction();
      measure_ = measure;
    }
  }
  void Insert(uint32_t key, uint32_t pos) {
    Insert(key, pos, key);
  }
  void Insert(uint32_t key, uint32_t pos, uint32_t prefix) {
    auto& bucket = buckets_[key & ((1u << 15) - 1)];
    if (filter_) {
      auto& tags = fingerprints_[key & ((1u << 15) - 1)];
      const uint8_t tag = prefix >> 24;
      if (bucket.size == 0) tags.common = tag;
      if (tag != tags.common) tags.mixed = true;
      tags.values[bucket.next] = tag;
    }
    bucket.positions[bucket.next] = pos;
    bucket.next = (bucket.next + 1) & (kCapacity - 1);
    if (bucket.size < kCapacity) ++bucket.size;
  }
  template <typename Visit>
  void ForEach(uint32_t key, const Visit& visit) const {
    ForEach(key, visit, key);
  }
  template <typename Visit>
  void ForEach(uint32_t key, const Visit& visit, uint32_t prefix) const {
    const auto& bucket = buckets_[key & ((1u << 15) - 1)];
    if (filter_ && bucket.size != 0) {
      auto& tags = fingerprints_[key & ((1u << 15) - 1)];
      const uint64_t valid = ~uint64_t{0} >> (64 - bucket.size);
      const uint8_t tag = prefix >> 24;
      const uint64_t mask = tags.mixed
          ? filter_(tags.values.data(), kCapacity, tag) & valid
          : (tag == tags.common ? valid : 0);
      if (measure_) {
        scanned_ += bucket.size;
        retained_ += hwy::PopCount(mask);
        if (tags.mixed) ++simd_queries_;
        else ++uniform_queries_;
      }
      if (mask == valid) {
        tags.common = tag;
        tags.mixed = false;
      }
      if (mask != valid) {
        const auto walk = [&](uint64_t bits) {
          while (bits) {
            const size_t index = FloorLog2Nonzero(bits);
            visit(bucket.positions[index]);
            bits &= ~(uint64_t{1} << index);
          }
        };
        // Preserve newest-first order across the ring wrap.
        const uint64_t before = (uint64_t{1} << bucket.next) - 1;
        walk(mask & before);
        walk(mask & ~before);
        return;
      }
    }
    for (size_t i = 0; i < bucket.size; ++i) {
      visit(bucket.positions[(bucket.next - 1 - i) & (kCapacity - 1)]);
    }
  }
  // A full bucket may have evicted special-distance candidates older than this.
  uint32_t OldestIfFull(uint32_t key) const {
    const auto& bucket = buckets_[key & ((1u << 15) - 1)];
    return bucket.size == kCapacity ? bucket.positions[bucket.next] : 0;
  }
  uint32_t EvictedByNextInsert(uint32_t key) const {
    const auto& bucket = buckets_[key & ((1u << 15) - 1)];
    return bucket.size == kCapacity ? bucket.positions[bucket.next] : UINT32_MAX;
  }
  uint64_t Scanned() const { return scanned_; }
  uint64_t Retained() const { return retained_; }
  uint64_t SIMDQueries() const { return simd_queries_; }
  uint64_t UniformQueries() const { return uniform_queries_; }

 private:
  struct Bucket {
    std::array<uint32_t, kCapacity> positions{};
    uint32_t next = 0;
    uint32_t size = 0;
  };
  struct Fingerprints {
    std::array<uint8_t, kCapacity> values{};
    uint8_t common = 0;
    bool mixed = false;
  };
  std::vector<Bucket> buckets_;
  mutable std::vector<Fingerprints> fingerprints_;
  LZ77FingerprintFunction filter_ = nullptr;
  bool measure_ = false;
  mutable uint64_t scanned_ = 0, retained_ = 0;
  mutable uint64_t simd_queries_ = 0, uniform_queries_ = 0;
};

// Four representatives at each of three ages: every 4th, 16th, and 64th
// eviction from a long-prefix bucket. Queries visit at most twelve positions.
class LZ77BucketHistory {
 public:
  explicit LZ77BucketHistory(bool enabled) : buckets_(enabled ? 1u << 15 : 0) {}
  void Insert(uint32_t key, uint32_t pos) {
    auto& bucket = buckets_[key & ((1u << 15) - 1)];
    ++bucket.count;
    for (size_t level = 0; level < 3; ++level) {
      const size_t shift = 2 + 2 * level;
      if ((bucket.count & ((1u << shift) - 1)) == 0) {
        bucket.positions[level * 4 + ((bucket.count >> shift) & 3)] = pos;
      }
    }
  }
  template <typename Visit>
  void ForEach(uint32_t key, const Visit& visit) const {
    const auto& bucket = buckets_[key & ((1u << 15) - 1)];
    std::array<uint32_t, 12> seen{};
    size_t visited = 0;
    for (size_t level = 0; level < 3; ++level) {
      const uint32_t count = bucket.count >> (2 + 2 * level);
      for (uint32_t i = 0; i < 4 && i < count; ++i) {
        const uint32_t pos = bucket.positions[level * 4 + ((count - i) & 3)];
        bool duplicate = false;
        for (size_t j = 0; j < visited; ++j) duplicate |= seen[j] == pos;
        if (!duplicate) { seen[visited++] = pos; visit(pos); }
      }
    }
  }
 private:
  struct Bucket {
    std::array<uint32_t, 12> positions{};
    uint32_t count = 0;
  };
  std::vector<Bucket> buckets_;
};

}  // namespace jxl

#endif  // LIB_JXL_ENC_LZ77_BUCKET_H_
