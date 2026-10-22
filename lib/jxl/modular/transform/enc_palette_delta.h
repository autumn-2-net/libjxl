// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef LIB_JXL_MODULAR_TRANSFORM_ENC_PALETTE_DELTA_H_
#define LIB_JXL_MODULAR_TRANSFORM_ENC_PALETTE_DELTA_H_

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

#include "lib/jxl/base/status.h"
#include "lib/jxl/modular/encoding/context_predict.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/transform/enc_palette_map.h"
#include "lib/jxl/modular/transform/transform.h"

namespace jxl {

inline std::vector<uint8_t> ExactZeroDeltaMask(
    const Image& input, uint32_t begin_c, uint32_t end_c, Predictor predictor) {
  if (end_c >= input.channel.size() || begin_c > end_c ||
      end_c - begin_c < 2 || end_c - begin_c > 3 || input.bitdepth > 16 ||
      predictor == Predictor::Weighted || predictor >= Predictor::Best) return {};
  const auto& first = input.channel[begin_c];
  for (size_t c = begin_c; c <= end_c; ++c) {
    if (input.channel[c].w != first.w || input.channel[c].h != first.h) return {};
  }
  std::vector<uint8_t> mask(first.w * first.h, 1);
  for (size_t y = 0; y < first.h; ++y) {
    for (size_t x = 0; x < first.w; ++x) {
      for (size_t c = begin_c; c <= end_c; ++c) {
        const auto& channel = input.channel[c];
        const auto* pixel = channel.Row(y) + x;
        const auto predicted = PredictNoTreeNoWP(first.w, pixel,
            channel.plane.PixelsPerRow(), x, y, predictor);
        if (*pixel != predicted.guess) {
          mask[y * first.w + x] = 0;
          break;
        }
      }
    }
  }
  return mask;
}

inline Status AddExactZeroDelta(Image& input, uint32_t begin_c,
    uint32_t* deltas, Predictor* predictor, Predictor residual_predictor,
    const std::vector<uint8_t>& mask) {
  JXL_ENSURE(*deltas == 0 && !input.channel.empty());
  auto& old_table = input.channel[0];
  auto& indices = input.channel[begin_c + 1];
  JXL_ENSURE(mask.size() == indices.w * indices.h);
  JXL_ASSIGN_OR_RETURN(auto table,
      Channel::Create(input.memory_manager(), old_table.w + 1, old_table.h));
  table.hshift = table.vshift = -1;
  for (size_t c = 0; c < table.h; ++c) {
    table.Row(c)[0] = 0;
    std::copy_n(old_table.Row(c), old_table.w, table.Row(c) + 1);
  }
  for (size_t y = 0; y < indices.h; ++y) {
    auto* row = indices.Row(y);
    for (size_t x = 0; x < indices.w; ++x) {
      JXL_ENSURE(row[x] >= 0 && row[x] < INT32_MAX);
      row[x] = mask[y * indices.w + x] ? 0 : row[x] + 1;
    }
  }
  old_table = std::move(table);
  *deltas = 1;
  *predictor = residual_predictor;
  return true;
}

// A palette of exact residual tuples, not quantized colors. Input and transform
// parameters remain untouched when the requested palette cannot represent it.
inline StatusOr<bool> FwdExactDeltaPalette(
    Image& input, uint32_t begin_c, uint32_t end_c, uint32_t* colors,
    uint32_t* deltas, Predictor* predictor, Predictor residual_predictor) {
  if (end_c >= input.channel.size() || begin_c < input.nb_meta_channels || end_c < begin_c ||
      end_c - begin_c < 2 || end_c - begin_c > 3 ||
      input.bitdepth < 1 || input.bitdepth > 16 || *colors == 0 ||
      *colors > 65536 || residual_predictor == Predictor::Weighted ||
      residual_predictor >= Predictor::Best) return false;
  JXL_RETURN_IF_ERROR(CheckEqualChannels(input, begin_c, end_c));
  const size_t channels = end_c - begin_c + 1;
  const auto& first = input.channel[begin_c];
  const size_t width = first.w, height = first.h;
  if (!width || !height) return false;
  JXL_ASSIGN_OR_RETURN(auto indices, Channel::Create(input.memory_manager(),
      width, height, first.hshift, first.vshift));
  LosslessPaletteMap palette(channels);
  std::vector<std::vector<pixel_type>> entries;
  std::vector<pixel_type> residual(channels);
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      for (size_t c = 0; c < channels; ++c) {
        const auto& channel = input.channel[begin_c + c];
        const auto* pixel = channel.Row(y) + x;
        const auto prediction = PredictNoTreeNoWP(width, pixel,
            channel.plane.PixelsPerRow(), x, y, residual_predictor);
        const int64_t difference = int64_t(*pixel) - prediction.guess;
        if (difference < INT32_MIN || difference > INT32_MAX) return false;
        residual[c] = static_cast<pixel_type>(difference);
      }
      auto& entry = palette.Get(residual);
      if (!entry.seen) {
        if (entries.size() == *colors) return false;
        entry.seen = true;
        entry.index = entries.size();
        entries.push_back(residual);
      }
      ++entry.count;
      indices.Row(y)[x] = entry.index;
    }
  }
  std::vector<size_t> order(entries.size()), remap(entries.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    const size_t ac = palette.Get(entries[a]).count;
    const size_t bc = palette.Get(entries[b]).count;
    return ac == bc ? entries[a] < entries[b] : ac > bc;
  });
  JXL_ASSIGN_OR_RETURN(auto table,
      Channel::Create(input.memory_manager(), entries.size(), channels));
  table.hshift = table.vshift = -1;
  for (size_t i = 0; i < order.size(); ++i) {
    remap[order[i]] = i;
    for (size_t c = 0; c < channels; ++c) table.Row(c)[i] = entries[order[i]][c];
  }
  for (size_t y = 0; y < height; ++y) {
    auto* row = indices.Row(y);
    for (size_t x = 0; x < width; ++x) row[x] = remap[row[x]];
  }
  input.channel[begin_c] = std::move(indices);
  input.channel.erase(input.channel.begin() + begin_c + 1,
                      input.channel.begin() + end_c + 1);
  input.channel.insert(input.channel.begin(), std::move(table));
  ++input.nb_meta_channels;
  *colors = 0;
  *deltas = entries.size();
  *predictor = residual_predictor;
  return true;
}

}  // namespace jxl
#endif
