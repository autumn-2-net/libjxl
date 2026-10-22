// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_LOSSLESS_SCAN_H_
#define LIB_JXL_ENC_LOSSLESS_SCAN_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/status.h"

namespace jxl {

struct CodecMetadata;
class JxlEncoderChunkedFrameAdapter;

struct LosslessScanTile {
  size_t x = 0, y = 0, width = 0, height = 0;
  size_t anchor_x = 0, anchor_y = 0;
  uint64_t horizontal = 0, vertical = 0, observations = 0;
  size_t peak_row = 0;
  double Activity() const {
    return observations ? double(horizontal + vertical) / observations : 0;
  }
};

struct LosslessScanRegion {
  size_t x, y, width, height;
};

// Frame-local, read-only coverage map. It contains no encoded-size prediction.
// Byte comparisons include hidden RGB, every sample byte and extra channels.
struct LosslessScan {
  size_t width = 0, height = 0;
  uint64_t pixels = 0;
  bool literal_candidate = true;
  size_t sampled_colors = 0, sampled_flat = 0, sampled_pixels = 0;
  // Distinct RGB tuples seen before an optional bounded scan stops. Excludes
  // alpha: this is also a lower bound for an RGBA palette, not an upper bound.
  size_t palette_color_lower_bound = 0;
  std::vector<LosslessScanTile> tiles;

  bool Available() const { return !tiles.empty(); }
  bool LikelyContinuousTone() const {
    return sampled_pixels != 0 && sampled_colors > sampled_pixels / 2 &&
           sampled_flat < sampled_pixels / 2;
  }
  bool LikelySparseContinuousTone() const {
    if (sampled_flat >= sampled_pixels) return false;
    const size_t active = sampled_pixels - sampled_flat;
    return active >= 64 && sampled_colors > active - active / 4;
  }
  bool PreferCompactBandAtlas() const {
    // Distributed detail benefits from shared training across short row bands.
    // Flat-heavy compact screenshots keep the larger contextual witnesses.
    return pixels >= (1u << 19) && pixels < (2u << 20) &&
           sampled_pixels && sampled_flat < sampled_pixels / 2;
  }
  bool LikelyDenseTexture() const {
    return sampled_pixels && sampled_flat < sampled_pixels / 3 &&
           sampled_colors > sampled_pixels / 8;
  }
  std::vector<LosslessScanRegion> Regions() const;
};

StatusOr<LosslessScan> ScanLosslessInput(const CodecMetadata& metadata,
                                         JxlEncoderChunkedFrameAdapter& input,
                                         ThreadPool* pool);

}  // namespace jxl
#endif  // LIB_JXL_ENC_LOSSLESS_SCAN_H_
