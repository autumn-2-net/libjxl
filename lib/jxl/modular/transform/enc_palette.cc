// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/modular/transform/enc_palette.h"
#include "lib/jxl/modular/transform/enc_palette_order.h"

#include <jxl/memory_manager.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/modular/encoding/context_predict.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/options.h"
#include "lib/jxl/modular/transform/enc_transform.h"
#include "lib/jxl/modular/transform/enc_palette_map.h"
#include "lib/jxl/modular/transform/enc_palette_delta.h"
#include "lib/jxl/modular/transform/palette.h"
#include "lib/jxl/modular/transform/transform.h"

namespace jxl {

namespace palette_internal {

static constexpr bool kEncodeToHighQualityImplicitPalette = true;

// Inclusive.
static constexpr int kMinImplicitPaletteIndex = -(2 * 72 - 1);

float ColorDistance(const std::vector<float> &JXL_RESTRICT a,
                    const std::vector<pixel_type> &JXL_RESTRICT b) {
  JXL_DASSERT(a.size() == b.size());
  float distance = 0;
  float ave3 = 0;
  if (a.size() >= 3) {
    ave3 = (a[0] + b[0] + a[1] + b[1] + a[2] + b[2]) * (1.21f / 3.0f);
  }
  float sum_a = 0;
  float sum_b = 0;
  for (size_t c = 0; c < a.size(); ++c) {
    const float difference =
        static_cast<float>(a[c]) - static_cast<float>(b[c]);
    float weight = c == 0 ? 3 : c == 1 ? 5 : 2;
    if (c < 3 && (a[c] + b[c] >= ave3)) {
      const float add_w[3] = {
          1.15,
          1.15,
          1.12,
      };
      weight += add_w[c];
      if (c == 2 && ((a[2] + b[2]) < 1.22 * ave3)) {
        weight -= 0.5;
      }
    }
    distance += difference * difference * weight * weight;
    const int sum_weight = c == 0 ? 3 : c == 1 ? 5 : 1;
    sum_a += a[c] * sum_weight;
    sum_b += b[c] * sum_weight;
  }
  distance *= 4;
  float sum_difference = sum_a - sum_b;
  distance += sum_difference * sum_difference;
  return distance;
}

static int QuantizeColorToImplicitPaletteIndex(
    const std::vector<pixel_type> &color, const int palette_size,
    const int bit_depth, bool high_quality) {
  int index = 0;
  int quant = (1 << bit_depth) - 1;
  // When bit_depth == 1, half would be equal quant without this fuse.
  int half = (bit_depth > 1) ? (1 << (bit_depth - 1)) : 0;
  if (high_quality) {
    int multiplier = 1;
    for (int value : color) {
      int quantized = ((kLargeCube - 1) * value + half) / quant;
      JXL_DASSERT((quantized % kLargeCube) == quantized);
      index += quantized * multiplier;
      multiplier *= kLargeCube;
    }
    return index + palette_size + kLargeCubeOffset;
  } else {
    int multiplier = 1;
    for (int value : color) {
      value -= 1 << (std::max(0, bit_depth - 3));
      value = std::max(0, value);
      int quantized = ((kLargeCube - 1) * value + half) / quant;
      JXL_DASSERT((quantized % kLargeCube) == quantized);
      if (quantized > kSmallCube - 1) {
        quantized = kSmallCube - 1;
      }
      index += quantized * multiplier;
      multiplier *= kSmallCube;
    }
    return index + palette_size;
  }
}

}  // namespace palette_internal

int RoundInt(int value, int div) {  // symmetric rounding around 0
  if (value < 0) return -RoundInt(-value, div);
  return (value + div / 2) / div;
}

struct PaletteIterationData {
  static constexpr int kMaxDeltas = 128;
  bool final_run = false;
  std::vector<pixel_type> deltas[3];
  std::vector<double> delta_distances;
  std::vector<pixel_type> frequent_deltas[3];

  // Populates `frequent_deltas` with items from `deltas` based on frequencies
  // and color distances.
  void FindFrequentColorDeltas(int num_pixels, int bitdepth) {
    using pixel_type_3d = std::array<pixel_type, 3>;
    std::map<pixel_type_3d, double> delta_frequency_map;
    pixel_type bucket_size = 3 << std::max(0, bitdepth - 8);
    // Store frequency weighted by delta distance from quantized value.
    for (size_t i = 0; i < deltas[0].size(); ++i) {
      pixel_type_3d delta = {
          {RoundInt(deltas[0][i], bucket_size),
           RoundInt(deltas[1][i], bucket_size),
           RoundInt(deltas[2][i], bucket_size)}};  // a basic form of clustering
      if (delta[0] == 0 && delta[1] == 0 && delta[2] == 0) continue;
      delta_frequency_map[delta] += sqrt(sqrt(delta_distances[i]));
    }

    const float delta_distance_multiplier = 1.0f / num_pixels;

    // Weigh frequencies by magnitude and normalize.
    for (auto &delta_frequency : delta_frequency_map) {
      std::vector<pixel_type> current_delta = {delta_frequency.first[0],
                                               delta_frequency.first[1],
                                               delta_frequency.first[2]};
      float delta_distance =
          std::sqrt(palette_internal::ColorDistance({0, 0, 0}, current_delta)) +
          1;
      delta_frequency.second *=
          static_cast<double>(delta_distance) * delta_distance_multiplier;
    }

    // Sort by weighted frequency.
    using pixel_type_3d_frequency = std::pair<pixel_type_3d, double>;
    std::vector<pixel_type_3d_frequency> sorted_delta_frequency_map(
        delta_frequency_map.begin(), delta_frequency_map.end());
    std::sort(
        sorted_delta_frequency_map.begin(), sorted_delta_frequency_map.end(),
        [](const pixel_type_3d_frequency &a, const pixel_type_3d_frequency &b) {
          return a.second > b.second;
        });

    // Store the top deltas.
    for (auto &delta_frequency : sorted_delta_frequency_map) {
      if (frequent_deltas[0].size() >= kMaxDeltas) break;
      // Number obtained by optimizing on jyrki31 corpus:
      if (delta_frequency.second < 17) break;
      for (int c = 0; c < 3; ++c) {
        frequent_deltas[c].push_back(delta_frequency.first[c] * bucket_size);
      }
    }
  }
};

Status FwdPaletteIteration(Image &input, uint32_t begin_c, uint32_t end_c,
                           uint32_t &nb_colors, uint32_t &nb_deltas,
                           bool ordered, bool lossy, Predictor &predictor,
                           const weighted::Header &wp_header,
                           PaletteIterationData &palette_iteration_data,
                           bool lossless_encoder_tools) {
  JXL_QUIET_RETURN_IF_ERROR(CheckEqualChannels(input, begin_c, end_c));
  JXL_ENSURE(begin_c >= input.nb_meta_channels);
  JxlMemoryManager *memory_manager = input.memory_manager();
  uint32_t nb = end_c - begin_c + 1;

  size_t w = input.channel[begin_c].w;
  size_t h = input.channel[begin_c].h;
  if (input.bitdepth >= 32) return false;
  if (!lossy && nb_colors < 2) return false;

  if (!lossy && nb == 1) {
    // Channel palette special case
    if (nb_colors == 0) return false;
    std::vector<pixel_type> lookup;
    pixel_type minval;
    pixel_type maxval;
    compute_minmax(input.channel[begin_c], &minval, &maxval);
    size_t lookup_table_size =
        static_cast<int64_t>(maxval) - static_cast<int64_t>(minval) + 1;
    if (lookup_table_size > palette_internal::kMaxPaletteLookupTableSize) {
      // a lookup table would use too much memory, instead use a slower approach
      // with std::set
      std::set<pixel_type> chpalette;
      pixel_type idx = 0;
      for (size_t y = 0; y < h; y++) {
        const pixel_type *p = input.channel[begin_c].Row(y);
        for (size_t x = 0; x < w; x++) {
          const bool new_color = chpalette.insert(p[x]).second;
          if (new_color) {
            idx++;
            if (idx > static_cast<int>(nb_colors)) return false;
          }
        }
      }
      JXL_DEBUG_V(6, "Channel %i uses only %i colors.", begin_c, idx);
      JXL_ASSIGN_OR_RETURN(Channel pch,
                           Channel::Create(memory_manager, idx, 1));
      pch.hshift = -1;
      pch.vshift = -1;
      nb_colors = idx;
      idx = 0;
      pixel_type *JXL_RESTRICT p_palette = pch.Row(0);
      for (pixel_type p : chpalette) {
        p_palette[idx++] = p;
      }
      for (size_t y = 0; y < h; y++) {
        pixel_type *p = input.channel[begin_c].Row(y);
        for (size_t x = 0; x < w; x++) {
          for (idx = 0;
               p[x] != p_palette[idx] && idx < static_cast<int>(nb_colors);
               idx++) {
            // no-op
          }
          JXL_DASSERT(idx < static_cast<int>(nb_colors));
          p[x] = idx;
        }
      }
      predictor = Predictor::Zero;
      input.nb_meta_channels++;
      input.channel.insert(input.channel.begin(), std::move(pch));

      return true;
    }
    lookup.resize(lookup_table_size, 0);
    pixel_type idx = 0;
    for (size_t y = 0; y < h; y++) {
      const pixel_type *p = input.channel[begin_c].Row(y);
      for (size_t x = 0; x < w; x++) {
        if (lookup[p[x] - minval] == 0) {
          lookup[p[x] - minval] = 1;
          idx++;
          if (idx > static_cast<int>(nb_colors)) return false;
        }
      }
    }
    JXL_DEBUG_V(6, "Channel %i uses only %i colors.", begin_c, idx);
    JXL_ASSIGN_OR_RETURN(Channel pch, Channel::Create(memory_manager, idx, 1));
    pch.hshift = -1;
    pch.vshift = -1;
    nb_colors = idx;
    idx = 0;
    pixel_type *JXL_RESTRICT p_palette = pch.Row(0);
    for (size_t i = 0; i < lookup_table_size; i++) {
      if (lookup[i]) {
        p_palette[idx] = i + minval;
        lookup[i] = idx;
        idx++;
      }
    }
    for (size_t y = 0; y < h; y++) {
      pixel_type *p = input.channel[begin_c].Row(y);
      for (size_t x = 0; x < w; x++) p[x] = lookup[p[x] - minval];
    }
    predictor = Predictor::Zero;
    input.nb_meta_channels++;
    input.channel.insert(input.channel.begin(), std::move(pch));
    return true;
  }

  Image quantized_input(memory_manager);
  if (lossy) {
    JXL_ASSIGN_OR_RETURN(quantized_input, Image::Create(memory_manager, w, h,
                                                        input.bitdepth, nb));
    for (size_t c = 0; c < nb; c++) {
      JXL_RETURN_IF_ERROR(CopyImageTo(input.channel[begin_c + c].plane,
                                      &quantized_input.channel[c].plane));
    }
  }

  JXL_DEBUG_V(
      7, "Trying to represent channels %i-%i using at most a %i-color palette.",
      begin_c, end_c, nb_colors);
  nb_deltas = 0;
  bool delta_used = false;
  std::set<std::vector<pixel_type>> candidate_palette;
  std::vector<std::vector<pixel_type>> candidate_palette_imageorder;
  std::vector<pixel_type> color(nb);
  std::vector<float> color_with_error(nb);
  std::vector<const pixel_type *> p_in(nb);
  std::map<std::vector<pixel_type>, size_t> inv_palette;
  const char* map_env = std::getenv("JXL_LOSSLESS_PALETTE_MAP");
  const bool flat_map = lossless_encoder_tools && !lossy && nb <= 4 && map_env &&
      std::strcmp(map_env, "1") == 0;
  LosslessPaletteMap colors(std::min<uint32_t>(nb, 4));

  if (lossy) {
    palette_iteration_data.FindFrequentColorDeltas(w * h, input.bitdepth);
    nb_deltas = palette_iteration_data.frequent_deltas[0].size();

    // Count color frequency for colors that make a cross.
    std::map<std::vector<pixel_type>, size_t> color_freq_map;
    for (size_t y = 1; y + 1 < h; y++) {
      for (uint32_t c = 0; c < nb; c++) {
        p_in[c] = input.channel[begin_c + c].Row(y);
      }
      for (size_t x = 1; x + 1 < w; x++) {
        for (uint32_t c = 0; c < nb; c++) {
          color[c] = p_in[c][x];
        }
        int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        bool makes_cross = true;
        for (int i = 0; i < 4 && makes_cross; ++i) {
          int dx = offsets[i][0];
          int dy = offsets[i][1];
          for (uint32_t c = 0; c < nb && makes_cross; c++) {
            if (input.channel[begin_c + c].Row(y + dy)[x + dx] != color[c]) {
              makes_cross = false;
            }
          }
        }
        if (makes_cross) color_freq_map[color] += 1;
      }
    }
    // Add colors satisfying frequency condition to the palette.
    constexpr float kImageFraction = 0.01f;
    size_t color_frequency_lower_bound = 5 + input.h * input.w * kImageFraction;
    for (const auto &color_freq : color_freq_map) {
      if (color_freq.second > color_frequency_lower_bound) {
        candidate_palette.insert(color_freq.first);
        candidate_palette_imageorder.push_back(color_freq.first);
      }
    }
  }

  std::map<std::vector<pixel_type>, bool> implicit_color;
  std::vector<std::vector<pixel_type>> implicit_colors;
  implicit_colors.reserve(palette_internal::kImplicitPaletteSize);
  for (size_t k = 0; k < palette_internal::kImplicitPaletteSize; k++) {
    for (size_t i = 0; i < nb; i++) {
      color[i] = palette_internal::GetPaletteValue(nullptr, k, i, 0, 0,
                                                   input.bitdepth);
    }
    if (flat_map) colors.Get(color).implicit = true;
    else implicit_color[color] = true;
    implicit_colors.push_back(color);
  }

  std::map<std::vector<pixel_type>, size_t> color_freq_map;
  uint32_t implicit_colors_used = 0;
  std::vector<pixel_type> last_color;
  LosslessPaletteMap::Entry* last_entry = nullptr;
  for (size_t y = 0; y < h; y++) {
    for (uint32_t c = 0; c < nb; c++) {
      p_in[c] = input.channel[begin_c + c].Row(y);
    }
    for (size_t x = 0; x < w; x++) {
      if (lossy && candidate_palette.size() >= nb_colors) break;
      for (uint32_t c = 0; c < nb; c++) {
        color[c] = p_in[c][x];
      }
      LosslessPaletteMap::Entry* entry = nullptr;
      if (flat_map) {
        if (!last_entry || color != last_color) {
          last_color = color;
          last_entry = &colors.Get(color);
        }
        entry = last_entry;
      }
      const bool new_color = flat_map ? !entry->seen : candidate_palette.insert(color).second;
      if (flat_map) entry->seen = true;
      if (new_color) {
        if (flat_map ? entry->implicit : implicit_color[color]) {
          implicit_colors_used++;
        } else {
          candidate_palette_imageorder.push_back(color);
          if (candidate_palette_imageorder.size() > nb_colors) {
            return false;  // too many colors
          }
        }
      }
      if (flat_map) ++entry->count;
      else color_freq_map[color] += 1;
    }
  }

  nb_colors = nb_deltas + candidate_palette_imageorder.size();
  const auto frequency = [&](const std::vector<pixel_type>& key) {
    return flat_map ? colors.Get(key).count : color_freq_map[key];
  };

  // not useful to make a single-color palette
  if (!lossy && nb_colors + implicit_colors_used == 1) return false;
  // TODO(jon): if this happens (e.g. solid white group), special-case it for
  // faster encode

  for (size_t k = 0; k < palette_internal::kImplicitPaletteSize; k++) {
    color = implicit_colors[k];
    // still add the color to the explicit palette if it is frequent enough
    if (frequency(color) > 10) {
      nb_colors++;
      candidate_palette_imageorder.push_back(color);
    }
  }
  for (size_t k = 0; k < palette_internal::kImplicitPaletteSize; k++) {
    color = implicit_colors[k];
    if (flat_map) colors.Get(color).index = nb_colors + k;
    else inv_palette[color] = nb_colors + k;
  }

  JXL_DEBUG_V(6, "Channels %i-%i can be represented using a %i-color palette.",
              begin_c, end_c, nb_colors);

  JXL_ASSIGN_OR_RETURN(Channel pch,
                       Channel::Create(memory_manager, nb_colors, nb));
  pch.hshift = -1;
  pch.vshift = -1;
  pixel_type *JXL_RESTRICT p_palette = pch.Row(0);
  ptrdiff_t onerow = pch.plane.PixelsPerRow();
  ptrdiff_t onerow_image = input.channel[begin_c].plane.PixelsPerRow();
  const int bit_depth = std::min(input.bitdepth, 24);

  if (lossy) {
    for (uint32_t i = 0; i < nb_deltas; i++) {
      for (size_t c = 0; c < 3; c++) {
        p_palette[c * onerow + i] =
            palette_iteration_data.frequent_deltas[c][i];
      }
    }
  }
  // Separate the palette in two buckets, first the common colors, then the
  // rare colors.
  // Within each bucket, the colors are sorted on luma (times alpha).
  float freq_threshold = 4;  // arbitrary threshold
  int clr = 0;
  const char* order_env = std::getenv("JXL_LOSSLESS_PALETTE_ORDER");
  const int frequency_order = lossless_encoder_tools && !lossy && order_env &&
      order_env[0] >= '1' && order_env[0] <= '9' && order_env[1] == '\0'
      ? order_env[0] - '0' : 0;
  const char* geometry_env = std::getenv("JXL_LOSSLESS_PALETTE_GEOMETRY");
  const bool principal = lossless_encoder_tools && !lossy && geometry_env &&
      (std::strcmp(geometry_env, "1") == 0 || std::strcmp(geometry_env, "2") == 0);
  if (ordered && nb >= 3 && nb <= 4 && principal) {
    std::vector<std::array<int32_t, 4>> tuples(candidate_palette_imageorder.size());
    std::vector<size_t> frequencies(tuples.size());
    for (size_t i = 0; i < tuples.size(); ++i) {
      const auto& pcol = candidate_palette_imageorder[i];
      std::copy(pcol.begin(), pcol.end(), tuples[i].begin());
      frequencies[i] = frequency(pcol);
    }
    const auto order = PalettePrincipalOrder(tuples, frequencies, nb, geometry_env[0] == '2');
    std::vector<std::vector<pixel_type>> reordered;
    reordered.reserve(order.size());
    for (size_t i : order) reordered.push_back(std::move(candidate_palette_imageorder[i]));
    candidate_palette_imageorder = std::move(reordered);
  } else if (ordered && nb >= 3 && nb <= 4 && frequency_order >= 8) {
    const size_t count = candidate_palette_imageorder.size();
    std::vector<size_t> frequencies(count), bands(count);
    for (size_t i = 0; i < count; ++i) {
      const auto& pcol = candidate_palette_imageorder[i];
      frequencies[i] = frequency(pcol);
      bands[i] = FloorLog2Nonzero(std::max<size_t>(1, frequencies[i])) / 2;
      if (flat_map) colors.Get(pcol).index = i;
      else inv_palette[pcol] = i;
    }
    const size_t step = std::max<size_t>(1, (w * h + 262143) / 262144);
    std::vector<uint64_t> pairs;
    const auto index = [&](size_t x, size_t y) -> size_t {
      for (size_t c = 0; c < nb; ++c) color[c] = input.channel[begin_c + c].Row(y)[x];
      return flat_map ? colors.Get(color).index : inv_palette[color];
    };
    const auto add = [&](size_t a, size_t b) {
      if (a >= count || b >= count || a == b) return;
      if (frequency_order == 9 && bands[a] != bands[b]) return;
      if (a > b) std::swap(a, b);
      pairs.push_back((uint64_t{a} << 32) | b);
    };
    for (size_t pos = 0; pos < w * h; pos += step) {
      const size_t x = pos % w, y = pos / w, current = index(x, y);
      if (x + 1 < w) add(current, index(x + 1, y));
      if (y + 1 < h) add(current, index(x, y + 1));
    }
    const auto order = PaletteAdjacencyOrder(frequencies, std::move(pairs));
    JXL_ENSURE(order.size() == count);
    std::vector<std::vector<pixel_type>> ordered_colors;
    ordered_colors.reserve(count);
    for (size_t i : order) ordered_colors.push_back(std::move(candidate_palette_imageorder[i]));
    candidate_palette_imageorder = std::move(ordered_colors);
  } else if (ordered && nb >= 3 && nb <= 4 && frequency_order >= 6) {
    struct PaletteKey {
      std::array<uint32_t, 4> curve;
      size_t band, index;
    };
    std::vector<PaletteKey> keys;
    keys.reserve(candidate_palette_imageorder.size());
    for (size_t i = 0; i < candidate_palette_imageorder.size(); ++i) {
      const auto& pcol = candidate_palette_imageorder[i];
      std::array<uint32_t, 4> axes{};
      for (size_t c = 0; c < nb; ++c) axes[c] = uint32_t(pcol[c]) ^ 0x80000000u;
      keys.push_back({PaletteHilbertKey(axes, nb, 32),
          frequency_order == 6 ? FloorLog2Nonzero(std::max<size_t>(1, frequency(pcol))) / 2 : 0,
          i});
    }
    std::sort(keys.begin(), keys.end(), [nb](const auto& a, const auto& b) {
      if (a.band != b.band) return a.band > b.band;
      return PaletteInterleavedLess(a.curve, b.curve, nb);
    });
    std::vector<std::vector<pixel_type>> ordered_colors;
    ordered_colors.reserve(keys.size());
    for (const auto& key : keys)
      ordered_colors.push_back(std::move(candidate_palette_imageorder[key.index]));
    candidate_palette_imageorder = std::move(ordered_colors);
  } else if (ordered && nb >= 3) {
    JXL_DEBUG_V(7, "Palette of %i colors, using luma order", nb_colors);
    // sort on luma (multiplied by alpha if available)
    std::sort(candidate_palette_imageorder.begin(),
              candidate_palette_imageorder.end(),
              [&](const std::vector<pixel_type>& ap, const std::vector<pixel_type>& bp) {
                if (frequency_order) {
                  const size_t af = frequency(ap), bf = frequency(bp);
                  if (frequency_order == 1 && af != bf) return af > bf;
                  if (frequency_order == 2 || frequency_order == 3) {
                    const auto ab = FloorLog2Nonzero(std::max<size_t>(1, af)) / 2;
                    const auto bb = FloorLog2Nonzero(std::max<size_t>(1, bf)) / 2;
                    if (ab != bb) return ab > bb;
                  }
                  if (frequency_order == 4 && (af > 4) != (bf > 4))
                    return af > 4;
                  if (frequency_order >= 3) {
                    // Compare an interleaved signed-color Morton key without
                    // truncating channels or materializing a 128-bit key.
                    int highest_bit = -1;
                    size_t channel = 0;
                    for (size_t c = 0; c < ap.size(); ++c) {
                      const uint32_t delta = uint32_t(ap[c]) ^ uint32_t(bp[c]);
                      if (delta != 0) {
                        const int bit = FloorLog2Nonzero(delta);
                        if (bit > highest_bit) {
                          highest_bit = bit;
                          channel = c;
                        }
                      }
                    }
                    return highest_bit >= 0 && ap[channel] < bp[channel];
                  }
                }
                float ay;
                float by;
                ay = (0.299f * ap[0] + 0.587f * ap[1] + 0.114f * ap[2] + 0.1f);
                if (ap.size() > 3) ay *= 1.f + ap[3];
                by = (0.299f * bp[0] + 0.587f * bp[1] + 0.114f * bp[2] + 0.1f);
                if (bp.size() > 3) by *= 1.f + bp[3];
                // put common colors first, transparent dark to opaque bright,
                // then rare colors, bright to dark
                ay = frequency(ap) > freq_threshold ? -ay : ay;
                by = frequency(bp) > freq_threshold ? -by : by;
                return ay < by;
              });
  } else {
    JXL_DEBUG_V(7, "Palette of %i colors, using image order", nb_colors);
  }

  for (auto pcol : candidate_palette_imageorder) {
    JXL_DEBUG_V(9, "  Color %i :  ", clr);
    for (size_t i = 0; i < nb; i++) {
      p_palette[nb_deltas + i * onerow + clr] = pcol[i];
      JXL_DEBUG_V(9, "%i ", pcol[i]);
    }
    if (flat_map) colors.Get(pcol).index = clr;
    else inv_palette[pcol] = clr;
    clr++;
  }
  std::vector<weighted::State> wp_states;
  for (size_t c = 0; c < nb; c++) {
    wp_states.emplace_back(wp_header, w, h);
  }
  std::vector<pixel_type *> p_quant(nb);
  // Three rows of error for dithering: y to y + 2.
  // Each row has two pixels of padding in the ends, which is
  // beneficial for both precision and encoding speed.
  std::vector<std::vector<float>> error_row[3];
  if (lossy) {
    for (auto &row : error_row) {
      row.resize(nb);
      for (size_t c = 0; c < nb; ++c) {
        row[c].resize(w + 4);
      }
    }
  }
  last_color.clear();
  size_t last_index = 0;
  for (size_t y = 0; y < h; y++) {
    for (size_t c = 0; c < nb; c++) {
      p_in[c] = input.channel[begin_c + c].Row(y);
      if (lossy) p_quant[c] = quantized_input.channel[c].Row(y);
    }
    pixel_type *JXL_RESTRICT p = input.channel[begin_c].Row(y);
    for (size_t x = 0; x < w; x++) {
      int index;
      if (!lossy) {
        for (size_t c = 0; c < nb; c++) color[c] = p_in[c][x];
        if (flat_map) {
          if (color != last_color) {
            last_index = colors.Get(color).index;
            last_color = color;
          }
          index = last_index;
        } else index = inv_palette[color];
      } else {
        int best_index = 0;
        bool best_is_delta = false;
        float best_distance = std::numeric_limits<float>::infinity();
        std::vector<pixel_type> best_val(nb, 0);
        std::vector<pixel_type> ideal_residual(nb, 0);
        std::vector<pixel_type> quantized_val(nb);
        std::vector<pixel_type> predictions(nb);
        for (double diffusion_multiplier : {0.55, 0.75}) {
          for (size_t c = 0; c < nb; c++) {
            color_with_error[c] =
                p_in[c][x] + (palette_iteration_data.final_run ? 1 : 0) *
                                 diffusion_multiplier * error_row[0][c][x + 2];
            color[c] = Clamp1(lround(color_with_error[c]), 0l,
                              (1l << input.bitdepth) - 1);
          }

          for (size_t c = 0; c < nb; ++c) {
            predictions[c] = PredictNoTreeWP(w, p_quant[c] + x, onerow_image, x,
                                             y, predictor, &wp_states[c])
                                 .guess;
          }
          const auto TryIndex = [&](const int index) {
            for (size_t c = 0; c < nb; c++) {
              quantized_val[c] = palette_internal::GetPaletteValue(
                  p_palette, index, /*c=*/c,
                  /*palette_size=*/nb_colors,
                  /*onerow=*/onerow, /*bit_depth=*/bit_depth);
              if (index < static_cast<int>(nb_deltas)) {
                quantized_val[c] += predictions[c];
              }
            }
            const float color_distance =
                32.0 / (1LL << std::max(0, 2 * (bit_depth - 8))) *
                palette_internal::ColorDistance(color_with_error,
                                                quantized_val);
            float index_penalty = 0;
            if (index == -1) {
              index_penalty = -124;
            } else if (index < 0) {
              index_penalty = -2 * index;
            } else if (index < static_cast<int>(nb_deltas)) {
              index_penalty = 250;
            } else if (index < static_cast<int>(nb_colors)) {
              index_penalty = 150;
            } else if (index < static_cast<int>(nb_colors) +
                                   palette_internal::kLargeCubeOffset) {
              index_penalty = 70;
            } else {
              index_penalty = 256;
            }
            const float distance = color_distance + index_penalty;
            if (distance < best_distance) {
              best_distance = distance;
              best_index = index;
              best_is_delta = index < static_cast<int>(nb_deltas);
              best_val.swap(quantized_val);
              for (size_t c = 0; c < nb; ++c) {
                ideal_residual[c] = color_with_error[c] - predictions[c];
              }
            }
          };
          for (index = palette_internal::kMinImplicitPaletteIndex;
               index < static_cast<int32_t>(nb_colors); index++) {
            TryIndex(index);
          }
          TryIndex(palette_internal::QuantizeColorToImplicitPaletteIndex(
              color, nb_colors, bit_depth,
              /*high_quality=*/false));
          if (palette_internal::kEncodeToHighQualityImplicitPalette) {
            TryIndex(palette_internal::QuantizeColorToImplicitPaletteIndex(
                color, nb_colors, bit_depth,
                /*high_quality=*/true));
          }
        }
        index = best_index;
        delta_used |= best_is_delta;
        if (!palette_iteration_data.final_run) {
          for (size_t c = 0; c < 3; ++c) {
            palette_iteration_data.deltas[c].push_back(ideal_residual[c]);
          }
          palette_iteration_data.delta_distances.push_back(best_distance);
        }

        for (size_t c = 0; c < nb; ++c) {
          wp_states[c].UpdateErrors(best_val[c], x, y, w);
          p_quant[c][x] = best_val[c];
        }
        float len_error = 0;
        for (size_t c = 0; c < nb; ++c) {
          float local_error = color_with_error[c] - best_val[c];
          len_error += local_error * local_error;
        }
        len_error = std::sqrt(len_error);
        float modulate = 1.0;
        int len_limit = 38 << std::max(0, bit_depth - 8);
        if (len_error > len_limit) {
          modulate *= len_limit / len_error;
        }
        for (size_t c = 0; c < nb; ++c) {
          float total_error = (color_with_error[c] - best_val[c]);

          // If the neighboring pixels have some error in the opposite
          // direction of total_error, cancel some or all of it out before
          // spreading among them.
          constexpr int offsets[12][2] = {{1, 2}, {0, 3}, {0, 4}, {1, 1},
                                          {1, 3}, {2, 2}, {1, 0}, {1, 4},
                                          {2, 1}, {2, 3}, {2, 0}, {2, 4}};
          float total_available = 0;
          for (int i = 0; i < 11; ++i) {
            const int row = offsets[i][0];
            const int col = offsets[i][1];
            if (std::signbit(error_row[row][c][x + col]) !=
                std::signbit(total_error)) {
              total_available += error_row[row][c][x + col];
            }
          }
          float weight =
              std::abs(total_error) / (std::abs(total_available) + 1e-3);
          weight = std::min(weight, 1.0f);
          for (int i = 0; i < 11; ++i) {
            const int row = offsets[i][0];
            const int col = offsets[i][1];
            if (std::signbit(error_row[row][c][x + col]) !=
                std::signbit(total_error)) {
              total_error += weight * error_row[row][c][x + col];
              error_row[row][c][x + col] *= (1 - weight);
            }
          }
          total_error *= modulate;
          const float remaining_error = (1.0f / 14.) * total_error;
          error_row[0][c][x + 3] += 2 * remaining_error;
          error_row[0][c][x + 4] += remaining_error;
          error_row[1][c][x + 0] += remaining_error;
          for (int i = 0; i < 5; ++i) {
            error_row[1][c][x + i] += remaining_error;
            error_row[2][c][x + i] += remaining_error;
          }
        }
      }
      if (palette_iteration_data.final_run) p[x] = index;
    }
    if (lossy) {
      for (size_t c = 0; c < nb; ++c) {
        error_row[0][c].swap(error_row[1][c]);
        error_row[1][c].swap(error_row[2][c]);
        std::fill(error_row[2][c].begin(), error_row[2][c].end(), 0.f);
      }
    }
  }
  if (!delta_used) {
    predictor = Predictor::Zero;
  }
  if (palette_iteration_data.final_run) {
    input.nb_meta_channels++;
    input.channel.erase(input.channel.begin() + begin_c + 1,
                        input.channel.begin() + end_c + 1);
    input.channel.insert(input.channel.begin(), std::move(pch));
  }
  nb_colors -= nb_deltas;
  return true;
}

Status FwdPalette(Image &input, uint32_t begin_c, uint32_t end_c,
                  uint32_t &nb_colors, uint32_t &nb_deltas, bool ordered,
                  bool lossy, Predictor &predictor,
                  const weighted::Header &wp_header,
                  bool lossless_encoder_tools) {
  const char* delta_env = std::getenv("JXL_LOSSLESS_DELTA_PALETTE");
  Predictor zero_predictor = Predictor::Zero;
  std::vector<uint8_t> zero_mask;
  if (lossless_encoder_tools && !lossy && delta_env &&
      delta_env[0] >= '4' && delta_env[0] <= '6' &&
      delta_env[1] == '\0') {
    zero_predictor = delta_env[0] == '4' ? Predictor::Gradient
        : delta_env[0] == '5' ? Predictor::Left : Predictor::Top;
    zero_mask = ExactZeroDeltaMask(input, begin_c, end_c, zero_predictor);
  }
  if (lossless_encoder_tools && !lossy && delta_env &&
      delta_env[0] >= '1' && delta_env[0] <= '3' &&
      delta_env[1] == '\0') {
    const Predictor p = delta_env[0] == '1' ? Predictor::Gradient
        : delta_env[0] == '2' ? Predictor::Left : Predictor::Top;
    JXL_ASSIGN_OR_RETURN(bool applied, FwdExactDeltaPalette(input, begin_c,
        end_c, &nb_colors, &nb_deltas, &predictor, p));
    if (applied) return true;
  }
  PaletteIterationData palette_iteration_data;
  uint32_t nb_colors_orig = nb_colors;
  uint32_t nb_deltas_orig = nb_deltas;
  // preprocessing pass in case of lossy palette
  if (lossy && input.bitdepth >= 8) {
    JXL_RETURN_IF_ERROR(FwdPaletteIteration(
        input, begin_c, end_c, nb_colors_orig, nb_deltas_orig, ordered, lossy,
        predictor, wp_header, palette_iteration_data, lossless_encoder_tools));
  }
  palette_iteration_data.final_run = true;
  JXL_RETURN_IF_ERROR(FwdPaletteIteration(input, begin_c, end_c, nb_colors, nb_deltas,
                             ordered, lossy, predictor, wp_header,
                             palette_iteration_data, lossless_encoder_tools));
  if (!zero_mask.empty()) {
    JXL_RETURN_IF_ERROR(AddExactZeroDelta(input, begin_c, &nb_deltas,
                                           &predictor, zero_predictor, zero_mask));
  }
  return true;
}

}  // namespace jxl
