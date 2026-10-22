// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_MA_TREE_GROUPS_H_
#define LIB_JXL_ENC_MA_TREE_GROUPS_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace jxl {

// Partition ordered work items without splitting an item or leaving empty
// shards. The tree-sharding caller uses stream IDs and the static group property;
// sample gathering uses channel IDs without introducing any tree boundaries.
inline std::vector<size_t> MaTreeGroupBoundaries(
    const std::vector<uint64_t>& work, size_t requested,
    uint64_t min_work = 1 << 20, size_t limit = 8) {
  uint64_t total = 0;
  for (uint64_t count : work) total += count;
  min_work = std::max<uint64_t>(min_work, 1);
  const size_t shards = std::min<size_t>(
      std::min(requested, limit), total / min_work);
  std::vector<size_t> boundaries{0};
  if (shards > 1) {
    uint64_t prefix = 0, previous = 0;
    for (size_t i = 0; i + 1 < work.size(); ++i) {
      prefix += work[i];
      const uint64_t target = (total / shards) * boundaries.size() +
          ((total % shards) * boundaries.size() + shards - 1) / shards;
      if (boundaries.size() < shards && prefix >= target &&
          prefix - previous >= min_work && total - prefix >= min_work) {
        boundaries.push_back(i + 1);
        previous = prefix;
      }
    }
  }
  boundaries.push_back(work.size());
  return boundaries;
}

}  // namespace jxl

#endif  // LIB_JXL_ENC_MA_TREE_GROUPS_H_
