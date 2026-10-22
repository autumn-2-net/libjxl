// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef LIB_JXL_MODULAR_TRANSFORM_ENC_PALETTE_MAP_H_
#define LIB_JXL_MODULAR_TRANSFORM_ENC_PALETTE_MAP_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/jxl/base/status.h"

namespace jxl {

// Exact keys, including hidden RGB and signed transformed samples. Hashes only
// choose a slot; equality always compares every channel. Entries retain source
// insertion order, independently of table layout and rehashing.
class LosslessPaletteMap {
 public:
  struct Entry {
    std::array<int32_t, 4> key{};
    size_t count = 0;
    size_t index = 0;
    bool implicit = false;
    bool seen = false;
  };

  explicit LosslessPaletteMap(size_t channels) : channels_(channels), slots_(1024) {
    JXL_DASSERT(channels >= 1 && channels <= 4);
  }

  Entry& Get(const std::vector<int32_t>& color) {
    JXL_DASSERT(color.size() == channels_);
    std::array<int32_t, 4> key{};
    for (size_t c = 0; c < channels_; ++c) key[c] = color[c];
    if ((entries_.size() + 1) * 2 > slots_.size()) Rehash();
    size_t slot = Hash(key) & (slots_.size() - 1);
    while (slots_[slot]) {
      auto& entry = entries_[slots_[slot] - 1];
      if (entry.key == key) return entry;
      slot = (slot + 1) & (slots_.size() - 1);
    }
    entries_.push_back(Entry{key});
    slots_[slot] = entries_.size();
    return entries_.back();
  }

  size_t Size() const { return entries_.size(); }

 private:
  uint64_t Hash(const std::array<int32_t, 4>& key) const {
    uint64_t hash = 0x9e3779b97f4a7c15ULL;
    for (size_t c = 0; c < channels_; ++c)
      hash = (hash ^ static_cast<uint32_t>(key[c])) * 0xbf58476d1ce4e5b9ULL;
    return hash ^ (hash >> 32);
  }
  void Rehash() {
    slots_.assign(slots_.size() * 2, 0);
    for (size_t i = 0; i < entries_.size(); ++i) {
      size_t slot = Hash(entries_[i].key) & (slots_.size() - 1);
      while (slots_[slot]) slot = (slot + 1) & (slots_.size() - 1);
      slots_[slot] = i + 1;
    }
  }
  size_t channels_;
  std::vector<Entry> entries_;
  std::vector<uint32_t> slots_;
};

}  // namespace jxl
#endif
