// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_lossless_scan.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <numeric>
#include <unordered_set>

#include "lib/jxl/base/common.h"
#include "lib/jxl/enc_lossless_profile.h"
#include "lib/jxl/encode_internal.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/modular/transform/palette.h"

namespace jxl {
namespace {

template <size_t kBytes>
bool Different(const uint8_t* a, const uint8_t* b, size_t bytes) {
  return std::memcmp(a, b, kBytes == 0 ? bytes : kBytes) != 0;
}

template <size_t kBytes>
Status ScanPlane(const uint8_t* data, size_t stride, size_t bytes,
                 size_t columns, size_t rows,
                 std::vector<LosslessScanTile>* tiles, ThreadPool* pool) {
  // Workers own whole tile rows. No atomic update or image-sized scratch.
  return RunOnPool(
      pool, 0, rows, ThreadPool::NoInit,
      [&](uint32_t ty, size_t) -> Status {
        for (size_t tx = 0; tx < columns; ++tx) {
          auto& tile = (*tiles)[ty * columns + tx];
          for (size_t y = tile.y; y < tile.y + tile.height; ++y) {
            const uint8_t* row = data + y * stride;
            size_t row_activity = 0, first_x = tile.x;
            for (size_t x = tile.x; x < tile.x + tile.width; ++x) {
              const uint8_t* pixel = row + x * bytes;
              const bool h =
                  x && Different<kBytes>(pixel, pixel - bytes, bytes);
              const bool v =
                  y && Different<kBytes>(pixel, pixel - stride, bytes);
              if ((h || v) && row_activity == 0) first_x = x;
              row_activity += h + v;
              tile.horizontal += h;
              tile.vertical += v;
            }
            if (row_activity > tile.peak_row) {
              tile.peak_row = row_activity;
              tile.anchor_x = first_x;
              tile.anchor_y = y;
            }
          }
          tile.observations += 2 * uint64_t{tile.width} * tile.height;
        }
        return true;
      },
      "LosslessCoverageScan");
}

bool LiteralEvidence(const uint8_t* data, size_t stride, size_t bytes,
                     size_t width, size_t height, LosslessScan* scan) {
  std::array<uint64_t, 1024> colors;
  size_t flat = 0;
  for (size_t y = 0; y < 32; ++y) {
    const size_t sy = uint64_t{y} * (height - 1) / 31;
    for (size_t x = 0; x < 32; ++x) {
      const size_t sx = uint64_t{x} * (width - 1) / 31;
      const uint8_t* p = data + sy * stride + sx * bytes;
      uint64_t hash = 14695981039346656037ULL;
      for (size_t c = 0; c < bytes; ++c)
        hash = (hash ^ p[c]) * 1099511628211ULL;
      colors[32 * y + x] = hash;
      flat += !std::memcmp(p, p + (sx + 1 < width ? bytes : 0), bytes) &&
              !std::memcmp(p, p + (sy + 1 < height ? stride : 0), bytes);
    }
  }
  std::sort(colors.begin(), colors.end());
  scan->sampled_colors = std::unique(colors.begin(), colors.end()) - colors.begin();
  scan->sampled_flat = flat;
  scan->sampled_pixels = colors.size();
  return flat >= colors.size() / 8 ||
         scan->sampled_colors <= colors.size() / 4;
}

}  // namespace

StatusOr<LosslessScan> ScanLosslessInput(const CodecMetadata& metadata,
                                         JxlEncoderChunkedFrameAdapter& input,
                                         ThreadPool* pool) {
  LosslessScan result;
  if (input.IsJPEG() || input.StreamingInput() || !input.xsize ||
      !input.ysize || metadata.m.num_extra_channels > 8)
    return result;
  for (const auto& ec : metadata.m.extra_channel_info) {
    if (ec.dim_shift != 0) return result;
  }
  LosslessProfileScope profile(LosslessProfileCurrent(), "lossless.scan");
  result.width = input.xsize;
  result.height = input.ysize;
  result.pixels = uint64_t{input.xsize} * input.ysize;
  const size_t tw = std::max<size_t>(64, DivCeil(input.xsize, size_t{64}));
  const size_t th = std::max<size_t>(64, DivCeil(input.ysize, size_t{64}));
  const size_t columns = DivCeil(input.xsize, tw),
               rows = DivCeil(input.ysize, th);
  result.tiles.reserve(columns * rows);
  for (size_t y = 0; y < input.ysize; y += th) {
    for (size_t x = 0; x < input.xsize; x += tw) {
      LosslessScanTile tile;
      tile.x = x;
      tile.y = y;
      tile.width = std::min(tw, input.xsize - x);
      tile.height = std::min(th, input.ysize - y);
      tile.anchor_x = x + tile.width / 2;
      tile.anchor_y = y + tile.height / 2;
      result.tiles.push_back(tile);
    }
  }
  auto source = input.GetInputSource();
  JxlPixelFormat color{};
  source.get_color_channels_pixel_format(source.opaque, &color);
  const bool interleaved_alpha =
      color.num_channels == 2 || color.num_channels == 4;
  for (size_t plane = 0; plane <= metadata.m.num_extra_channels; ++plane) {
    if (plane == 1 && interleaved_alpha) continue;
    JxlPixelFormat format = color;
    if (plane != 0) {
      source.get_extra_channel_pixel_format(source.opaque, plane - 1, &format);
      format.num_channels = 1;
    }
    const size_t bytes = BytesPerPixel(format);
    size_t stride = 0;
    auto buffer =
        plane == 0
            ? GetColorBuffer(source, 0, 0, input.xsize, input.ysize, &stride)
            : GetExtraChannelBuffer(source, plane - 1, 0, 0, input.xsize,
                                    input.ysize, &stride);
    if (!buffer) return JXL_FAILURE("Missing lossless scan input channel");
    JXL_ENSURE(bytes && stride / bytes >= input.xsize);
    const auto* data = static_cast<const uint8_t*>(buffer.get());
    if (plane == 0) {
      result.literal_candidate =
          LiteralEvidence(data, stride, bytes, input.xsize, input.ysize, &result);
      const char* scope = std::getenv("JXL_LOSSLESS_PALETTE_SCOPE");
      if (scope && std::strcmp(scope, "1") == 0 &&
          !metadata.m.color_encoding.IsGray() &&
          (color.num_channels == 3 || color.num_channels == 4) &&
          (color.data_type == JXL_TYPE_UINT8 || color.data_type == JXL_TYPE_UINT16) &&
          BitsPerChannel(color.data_type) == metadata.m.bit_depth.bits_per_sample) {
        const size_t rgb_bytes = 3 * BitsPerChannel(color.data_type) / 8;
        const size_t limit = 70000 + palette_internal::kImplicitPaletteSize;
        std::unordered_set<uint64_t> colors;
        colors.reserve(limit + 1);
        uint64_t previous = UINT64_MAX;
        for (size_t y = 0; y < input.ysize && colors.size() <= limit; ++y) {
          const auto* row = data + y * stride;
          for (size_t x = 0; x < input.xsize; ++x) {
            uint64_t key = 0;
            std::memcpy(&key, row + x * bytes, rgb_bytes);
            if (key == previous) continue;
            previous = key;
            colors.insert(key);
            if (colors.size() > limit) break;
          }
        }
        result.palette_color_lower_bound = colors.size();
      }
    }
    switch (bytes) {
#define JXL_SCAN_BYTES(N)                                                \
  case N:                                                                \
    JXL_RETURN_IF_ERROR(ScanPlane<N>(data, stride, bytes, columns, rows, \
                                     &result.tiles, pool));              \
    break
      JXL_SCAN_BYTES(1);
      JXL_SCAN_BYTES(2);
      JXL_SCAN_BYTES(3);
      JXL_SCAN_BYTES(4);
      JXL_SCAN_BYTES(6);
      JXL_SCAN_BYTES(8);
      JXL_SCAN_BYTES(12);
      JXL_SCAN_BYTES(16);
#undef JXL_SCAN_BYTES
      default:
        JXL_RETURN_IF_ERROR(ScanPlane<0>(data, stride, bytes, columns, rows,
                                         &result.tiles, pool));
    }
  }
  profile.Set("pixels", result.pixels);
  profile.Set("tiles", result.tiles.size());
  profile.Set("palette_color_lower_bound", result.palette_color_lower_bound);
  return result;
}

std::vector<LosslessScanRegion> LosslessScan::Regions() const {
  std::vector<LosslessScanRegion> result;
  if (!Available()) return result;
  std::vector<size_t> active;
  for (size_t i = 0; i < tiles.size(); ++i) {
    if (tiles[i].horizontal || tiles[i].vertical) active.push_back(i);
  }
  if (active.empty()) active.push_back(tiles.size() / 2);
  std::stable_sort(active.begin(), active.end(), [&](size_t a, size_t b) {
    return tiles[a].Activity() < tiles[b].Activity();
  });
  // High, medium and low activity, spatially distributed within each stratum.
  // Original pixels are copied later; scanning never downsamples the probes.
  for (size_t i = 0; i < 3; ++i) {
    const size_t lo = (2 - i) * active.size() / 3;
    const size_t hi = std::max(lo + 1, (3 - i) * active.size() / 3);
    size_t chosen = active[lo + (hi - lo) / 2];
    double farthest = -1;
    for (size_t j = lo; j < hi; ++j) {
      const auto& tile = tiles[active[j]];
      double distance = 2;
      for (const auto& r : result) {
        const double dx = (double(tile.anchor_x) - (r.x + r.width / 2)) / width;
        const double dy =
            (double(tile.anchor_y) - (r.y + r.height / 2)) / height;
        distance = std::min(distance, dx * dx + dy * dy);
      }
      if (distance > farthest) {
        chosen = active[j];
        farthest = distance;
      }
    }
    const auto& tile = tiles[chosen];
    const size_t rw = std::min(width, i == 2 ? size_t{256} : size_t{1024});
    const size_t rh = std::min(height, i == 2                 ? size_t{256}
                                       : pixels >= (2u << 20) ? size_t{64}
                                                              : size_t{32});
    const size_t x = std::min(
        width - rw, tile.anchor_x > rw / 2 ? tile.anchor_x - rw / 2 : 0);
    const size_t y = std::min(
        height - rh, tile.anchor_y > rh / 2 ? tile.anchor_y - rh / 2 : 0);
    if (std::none_of(result.begin(), result.end(), [&](const auto& r) {
          return r.x == x && r.y == y && r.width == rw && r.height == rh;
        }))
      result.push_back({x, y, rw, rh});
  }
  return result;
}

}  // namespace jxl
