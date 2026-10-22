// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_modular.h"
#include "lib/jxl/enc_lossless_profile.h"
#include "lib/jxl/enc_lossless_probe_pool.h"
#include "lib/jxl/enc_lossless_workers.h"
#include "lib/jxl/modular/encoding/enc_predict.h"
#include "lib/jxl/enc_ma_search.h"
#include "lib/jxl/enc_ma_tree_groups.h"
#include "lib/jxl/enc_neural_plan.h"
#include "lib/jxl/enc_final_plan.h"

#include <jxl/cms_interface.h>
#include <jxl/memory_manager.h>
#include <jxl/types.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/chroma_from_luma.h"
#include "lib/jxl/common.h"
#include "lib/jxl/compressed_dc.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/dec_modular.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_ans_params.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/enc_cache.h"
#include "lib/jxl/enc_fields.h"
#include "lib/jxl/enc_gaborish.h"
#include "lib/jxl/enc_modular_simd.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/enc_patch_dictionary.h"
#include "lib/jxl/enc_quant_weights.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/modular/encoding/context_predict.h"
#include "lib/jxl/modular/encoding/dec_ma.h"
#include "lib/jxl/modular/encoding/enc_encoding.h"
#include "lib/jxl/modular/encoding/enc_ma.h"
#include "lib/jxl/modular/encoding/enc_ma_histograms.h"
#include "lib/jxl/modular/encoding/encoding.h"
#include "lib/jxl/modular/encoding/ma_common.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/options.h"
#include "lib/jxl/modular/transform/enc_rct.h"
#include "lib/jxl/modular/transform/enc_transform.h"
#include "lib/jxl/modular/transform/squeeze.h"
#include "lib/jxl/modular/transform/squeeze_params.h"
#include "lib/jxl/modular/transform/transform.h"
#include "lib/jxl/pack_signed.h"
#include "lib/jxl/passes_state.h"
#include "lib/jxl/quant_weights.h"
#include "modular/options.h"

namespace jxl {

namespace {
// constexpr bool kPrintTree = false;

class LosslessStageTimer {
 public:
  LosslessStageTimer(const CompressParams& params, const FrameDimensions& dim,
                     const char* stage) : stage_(stage) {
    const char* trace = std::getenv("JXL_LOSSLESS_SEARCH_TRACE");
    enabled_ = params.IsLossless() && params.modular_mode &&
               uint64_t{dim.xsize} * dim.ysize > 262144 && trace &&
               std::strcmp(trace, "1") == 0;
    if (enabled_) {
      start_ = std::chrono::steady_clock::now();
      std::fprintf(stderr, "lossless-modular event=start step=%s\n", stage_);
      std::fflush(stderr);
    }
  }
  ~LosslessStageTimer() {
    if (enabled_) {
      std::fprintf(stderr, "lossless-modular event=end step=%s seconds=%.6f\n",
                   stage_, std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - start_).count());
      std::fflush(stderr);
    }
  }
 private:
  const char* stage_;
  bool enabled_;
  std::chrono::steady_clock::time_point start_;
};

// Squeeze default quantization factors
// these quantization factors are for -Q 50  (other qualities simply scale the
// factors; things are rounded down and obviously cannot get below 1)
const float squeeze_quality_factor =
    0.35;  // for easy tweaking of the quality range (decrease this number for
           // higher quality)
const float squeeze_luma_factor =
    1.1;  // for easy tweaking of the balance between luma (or anything
          // non-chroma) and chroma (decrease this number for higher quality
          // luma)
const float squeeze_quality_factor_xyb = 4.0f;
const float squeeze_quality_factor_y = 1.5f;

const float squeeze_xyb_qtable[3][16] = {
    {163.84, 81.92, 40.96, 20.48, 10.24, 5.12, 2.56, 1.28, 0.64, 0.32, 0.16,
     0.08, 0.04, 0.02, 0.01, 0.005},  // Y
    {1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1, 0.5, 0.5, 0.5, 0.5,
     0.5},  // X
    {2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1, 0.5, 0.5, 0.5,
     0.5},  // B-Y
};

const float squeeze_luma_qtable[16] = {163.84, 81.92, 40.96, 20.48, 10.24, 5.12,
                                       2.56,   1.28,  0.64,  0.32,  0.16,  0.08,
                                       0.04,   0.02,  0.01,  0.005};
// for 8-bit input, the range of YCoCg chroma is -255..255 so basically this
// does 4:2:0 subsampling (two most fine grained layers get quantized away)
const float squeeze_chroma_qtable[16] = {
    1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1, 0.5, 0.5, 0.5, 0.5, 0.5};

// Merges the trees in `trees` using nodes that decide on stream_id, as defined
// by `tree_splits`.
Status MergeTrees(const std::vector<Tree>& trees,
                  const std::vector<size_t>& tree_splits, size_t begin,
                  size_t end, Tree* tree) {
  JXL_ENSURE(trees.size() + 1 == tree_splits.size());
  JXL_ENSURE(end > begin);
  JXL_ENSURE(end <= trees.size());
  if (end == begin + 1) {
    // Insert the tree, adding the opportune offset to all child nodes.
    // This will make the leaf IDs wrong, but subsequent roundtripping will fix
    // them.
    size_t sz = tree->size();
    tree->insert(tree->end(), trees[begin].begin(), trees[begin].end());
    for (size_t i = sz; i < tree->size(); i++) {
      (*tree)[i].lchild += sz;
      (*tree)[i].rchild += sz;
    }
    return true;
  }
  size_t mid = (begin + end) / 2;
  size_t splitval = tree_splits[mid] - 1;
  size_t cur = tree->size();
  tree->emplace_back(1 /*stream_id*/, static_cast<int>(splitval), 0, 0,
                     Predictor::Zero, 0, 1);
  (*tree)[cur].lchild = tree->size();
  JXL_RETURN_IF_ERROR(MergeTrees(trees, tree_splits, mid, end, tree));
  (*tree)[cur].rchild = tree->size();
  JXL_RETURN_IF_ERROR(MergeTrees(trees, tree_splits, begin, mid, tree));
  return true;
}

void QuantizeChannel(Channel& ch, const int q) {
  if (q == 1) return;
  for (size_t y = 0; y < ch.plane.ysize(); y++) {
    pixel_type* row = ch.plane.Row(y);
    for (size_t x = 0; x < ch.plane.xsize(); x++) {
      if (row[x] < 0) {
        row[x] = -((-row[x] + q / 2) / q) * q;
      } else {
        row[x] = ((row[x] + q / 2) / q) * q;
      }
    }
  }
}

// convert binary32 float that corresponds to custom [bits]-bit float (with
// [exp_bits] exponent bits) to a [bits]-bit integer representation that should
// fit in pixel_type
Status float_to_int(const float* const row_in, pixel_type* const row_out,
                    size_t xsize, unsigned int bits, unsigned int exp_bits,
                    bool fp, double dfactor) {
  JXL_ENSURE(sizeof(pixel_type) * 8 >= bits);
  if (!fp) {
    if (bits > 22) {
      for (size_t x = 0; x < xsize; ++x) {
        row_out[x] = row_in[x] * dfactor + (row_in[x] < 0 ? -0.5 : 0.5);
      }
    } else {
      float factor = dfactor;
      for (size_t x = 0; x < xsize; ++x) {
        row_out[x] = row_in[x] * factor + (row_in[x] < 0 ? -0.5f : 0.5f);
      }
    }
    return true;
  }
  if (bits == 32 && fp) {
    JXL_ENSURE(exp_bits == 8);
    memcpy(static_cast<void*>(row_out), static_cast<const void*>(row_in),
           4 * xsize);
    return true;
  }

  JXL_ENSURE(bits > 0);
  int exp_bias = (1 << (exp_bits - 1)) - 1;
  int max_exp = (1 << exp_bits) - 1;
  uint32_t sign = (1u << (bits - 1));
  int mant_bits = bits - exp_bits - 1;
  int mant_shift = 23 - mant_bits;
  for (size_t x = 0; x < xsize; ++x) {
    uint32_t f;
    memcpy(&f, &row_in[x], 4);
    int signbit = (f >> 31);
    f &= 0x7fffffff;
    if (f == 0) {
      row_out[x] = (signbit ? sign : 0);
      continue;
    }
    int exp = (f >> 23) - 127;
    int mantissa = (f & 0x007fffff);
    // broke up the binary32 into its parts, now reassemble into
    // arbitrary float
    if (exp == 128) {
      // NaN or infinity
      f = (signbit ? sign : 0);
      f |= ((1 << exp_bits) - 1) << mant_bits;
      f |= mantissa >> mant_shift;
      row_out[x] = static_cast<pixel_type>(f);
      continue;
    }
    exp += exp_bias;
    if (exp <= 0) {  // will become a subnormal number
      // add implicit leading 1 to mantissa
      mantissa |= 0x00800000;
      if (exp < -mant_bits) {
        return JXL_FAILURE(
            "Invalid float number: %g cannot be represented with %i "
            "exp_bits and %i mant_bits (exp %i)",
            row_in[x], exp_bits, mant_bits, exp);
      }
      mantissa >>= 1 - exp;
      exp = 0;
    }
    // exp should be representable in exp_bits, otherwise input was
    // invalid; max_exp is NaN or infinity
    if (exp >= max_exp) return JXL_FAILURE("Invalid float exponent");
    if (mantissa & ((1 << mant_shift) - 1)) {
      return JXL_FAILURE("%g is losing precision (mant: %x)", row_in[x],
                         mantissa);
    }
    mantissa >>= mant_shift;
    f = (signbit ? sign : 0);
    f |= (exp << mant_bits);
    f |= mantissa;
    row_out[x] = static_cast<pixel_type>(f);
  }
  return true;
}

struct WPCostWork {
  size_t pixels = 0;
  size_t pruned_modes = 0;
  size_t checkpoints = 0;
};

float EstimateWPCost(const Image& img, size_t i, float cost_limit,
                     WPCostWork* work, bool lossless,
                     size_t selected_channel = SIZE_MAX) {
  const char* flat_env = std::getenv("JXL_LOSSLESS_WP_FLAT");
  const bool flat = lossless && flat_env &&
      (std::strcmp(flat_env, "1") == 0 || std::strcmp(flat_env, "3") == 0);
  size_t extra_bits = 0;
  float histo_cost = 0;
  size_t total_pixels = 0;
  for (const auto& ch : img.channel) total_pixels += ch.w * ch.h;
  // Prefix entropy is monotone in exact arithmetic. Leave a generous margin
  // for the float entropy estimator; never prune on a near tie.
  const double bound_margin = 16.0 + total_pixels * 0.001;
  const bool bounded = std::isfinite(cost_limit) &&
      cost_limit < std::numeric_limits<float>::max() && total_pixels >= 65536;
  HybridUintConfig config;
  int32_t cutoffs[] = {-500, -392, -255, -191, -127, -95, -63, -47, -31,
                       -23,  -15,  -11,  -7,   -4,   -3,  -1,  0,   1,
                       3,    5,    7,    11,   15,   23,  31,  47,  63,
                       95,   127,  191,  255,  392,  500};
  constexpr size_t nc = sizeof(cutoffs) / sizeof(*cutoffs) + 1;
  Histogram histo[nc] = {};
  weighted::Header wp_header;
  PredictorMode(i, &wp_header);
  for (size_t c = 0; c < img.channel.size(); ++c) {
    if (selected_channel != SIZE_MAX && c != selected_channel) continue;
    const Channel& ch = img.channel[c];
    const ptrdiff_t onerow = ch.plane.PixelsPerRow();
    weighted::State wp_state(wp_header, ch.w, ch.h);
    Properties properties(1);
    bool unhealthy = false;
    for (size_t y = 0; y < ch.h; y++) {
      const pixel_type* JXL_RESTRICT r = ch.Row(y);
      for (size_t x = 0; x < ch.w; x++) {
        size_t offset = 0;
        pixel_type_w left = (x ? r[x - 1] : y ? *(r + x - onerow) : 0);
        pixel_type_w top = (y ? *(r + x - onerow) : left);
        pixel_type_w topleft = (x && y ? *(r + x - 1 - onerow) : left);
        pixel_type_w topright =
            (x + 1 < ch.w && y ? *(r + x + 1 - onerow) : top);
        pixel_type_w toptop = (y > 1 ? *(r + x - onerow - onerow) : top);
        pixel_type guess = flat
            ? PredictWeightedFlat<true>(&wp_state, x, y, ch.w, top, left,
                                         topright, topleft, toptop, &properties, offset)
            : wp_state.Predict<true>(x, y, ch.w, top, left, topright, topleft,
                                     toptop, &properties, offset);
        size_t ctx = 0;
        for (int c : cutoffs) {
          ctx += (c >= properties[0]) ? 1 : 0;
        }
        pixel_type res;
        unhealthy |= SubOverflow(r[x], guess, res);
        uint32_t token;
        uint32_t nbits;
        uint32_t bits;
        config.Encode(PackSigned(res), &token, &nbits, &bits);
        histo[ctx].Add(token);
        extra_bits += nbits;
        wp_state.UpdateErrors(r[x], x, y, ch.w);
      }
      work->pixels += ch.w;
      if (bounded && (y & 63) == 63) {
        double lower_bound = histo_cost + static_cast<double>(extra_bits);
        for (const auto& h : histo) lower_bound += h.ShannonEntropy();
        ++work->checkpoints;
        if (lower_bound > cost_limit + bound_margin) {
          ++work->pruned_modes;
          return std::numeric_limits<float>::max();
        }
      }
    }
    if (unhealthy) {
      // Force this predictor option to be rejected by the cost selector.
      return std::numeric_limits<float>::max();
    }
    for (auto& h : histo) {
      histo_cost += h.ShannonEntropy();
      h.Clear();
    }
  }
  return histo_cost + extra_bits;
}

bool do_transform(Image& image, const Transform& tr,
                  const weighted::Header& wp_header,
                  jxl::ThreadPool* pool = nullptr, bool force_jxlart = false) {
  Transform t = tr;
  bool did_it = true;
  if (force_jxlart) {
    if (!t.MetaApply(image)) return false;
  } else {
    did_it = TransformForward(t, image, wp_header, pool);
  }
  if (did_it) image.transform.push_back(t);
  return did_it;
}

Status FinalTransforms(Image& image, size_t start) {
  if (!CurrentFinalPlan()) return true;
  std::vector<int64_t> values;
  if (!ReplayingFinalPlan()) {
    for (size_t i = start; i < image.transform.size(); ++i) {
      const auto& t = image.transform[i];
      if (t.id == TransformId::kSqueeze || t.lossy_palette) {
        CurrentFinalPlan()->unsupported = "squeeze or lossy palette";
        return true;
      }
      values.insert(values.end(), {static_cast<int>(t.id), t.begin_c,
          t.id == TransformId::kRCT ? t.rct_type : 0,
          t.id == TransformId::kPalette ? t.num_c : 0,
          t.id == TransformId::kPalette ? t.nb_colors : 0,
          t.id == TransformId::kPalette ? static_cast<int>(t.predictor) : 0,
          t.ordered_palette});
    }
  }
  JXL_RETURN_IF_ERROR(FinalPlanExchange("transforms", &values));
  if (!ReplayingFinalPlan()) return true;
  JXL_ENSURE(values.size() % 7 == 0 && values.size() <= 7 * 512);
  for (size_t i = 0; i < values.size(); i += 7) {
    JXL_ENSURE(values[i] >= 0 && values[i] <= 1 && values[i + 1] >= 0 &&
        static_cast<size_t>(values[i + 1]) < image.channel.size() &&
        values[i + 2] >= 0 && values[i + 2] < 42 && values[i + 3] >= 0 &&
        static_cast<size_t>(values[i + 3]) <= image.channel.size() &&
        values[i + 4] >= 0 && values[i + 4] <= 65536 &&
        values[i + 5] >= 0 && values[i + 5] < 14 &&
        values[i + 6] >= 0 && values[i + 6] <= 1);
    Transform t(static_cast<TransformId>(values[i]));
    t.begin_c = values[i + 1];
    t.rct_type = values[i + 2];
    t.num_c = values[i + 3];
    t.nb_colors = values[i + 4];
    t.predictor = static_cast<Predictor>(values[i + 5]);
    t.ordered_palette = values[i + 6];
    t.lossy_palette = false;
    JXL_ENSURE(t.begin_c + (t.id == TransformId::kRCT ? 3 : t.num_c) <=
               image.channel.size());
    JXL_ENSURE(t.id != TransformId::kPalette || t.num_c > 0);
    JXL_ENSURE(do_transform(image, t, weighted::Header()));
  }
  return true;
}

Status FinalTree(Tree* tree) {
  if (!CurrentFinalPlan()) return true;
  std::vector<int64_t> values;
  if (!ReplayingFinalPlan() && !tree->empty()) {
    std::vector<size_t> queue{0};
    for (size_t i = 0; i < queue.size(); ++i) {
      JXL_ENSURE(queue.size() <= tree->size() && queue[i] < tree->size());
      const auto& n = (*tree)[queue[i]];
      values.insert(values.end(), {n.property,
          n.property < 0 ? n.predictor_offset : n.splitval,
          n.property < 0 ? static_cast<int>(n.predictor) : 0,
          n.property < 0 ? n.multiplier : 1});
      if (n.property >= 0) { queue.push_back(n.lchild); queue.push_back(n.rchild); }
    }
  }
  JXL_RETURN_IF_ERROR(FinalPlanExchange("tree", &values));
  if (!ReplayingFinalPlan()) return true;
  const size_t count = values.size() / 4;
  JXL_ENSURE(values.size() % 4 == 0 && count > 0 && count % 2 == 1 &&
             count <= CurrentFinalPlan()->tree_limit);
  tree->clear();
  size_t next = 1;
  for (size_t i = 0; i < count; ++i) {
    const auto* v = values.data() + 4 * i;
    JXL_ENSURE(i < next && v[0] >= -1 && v[0] < 256 &&
               v[1] >= INT32_MIN && v[1] <= INT32_MAX);
    if (v[0] == -1) {
      JXL_ENSURE(v[2] >= 0 && v[2] < 14 && v[3] > 0 && v[3] < (1LL << 31));
      tree->push_back(PropertyDecisionNode::Leaf(static_cast<Predictor>(v[2]), v[1], v[3]));
    } else {
      JXL_ENSURE(v[2] == 0 && v[3] == 1 && next + 2 <= count);
      tree->push_back(PropertyDecisionNode::Split(v[0], v[1], next));
      next += 2;
    }
  }
  JXL_ENSURE(next == count);
  struct Visit { size_t index; int stage; int32_t lo, hi; };
  std::array<std::pair<int32_t, int32_t>, 256> ranges;
  for (auto& range : ranges) range = {INT32_MIN, INT32_MAX};
  std::vector<Visit> stack{{0, 0, 0, 0}};
  while (!stack.empty()) {
    JXL_ENSURE(stack.size() < 2048);
    auto& item = stack.back();
    const auto& n = (*tree)[item.index];
    if (n.property < 0) { stack.pop_back(); continue; }
    auto& range = ranges[n.property];
    if (item.stage == 0) {
      JXL_ENSURE(range.first <= n.splitval && n.splitval < range.second);
      item.lo = range.first; item.hi = range.second; item.stage = 1;
      range.first = n.splitval + 1;
      stack.push_back({n.lchild, 0, 0, 0});
    } else if (item.stage == 1) {
      range = {item.lo, n.splitval}; item.stage = 2;
      stack.push_back({n.rchild, 0, 0, 0});
    } else {
      range = {item.lo, item.hi}; stack.pop_back();
    }
  }
  return true;
}

StatusOr<bool> maybe_do_transform(Image& image, const Transform& tr,
                                  const CompressParams& cparams,
                                  const weighted::Header& wp_header,
                                  float cost_before,
                                  jxl::ThreadPool* pool = nullptr,
                                  bool force_jxlart = false) {
  if (force_jxlart || cparams.speed_tier >= SpeedTier::kSquirrel) {
    return do_transform(image, tr, wp_header, pool, force_jxlart);
  }
  bool did_it = do_transform(image, tr, wp_header, pool);
  if (did_it) {
    JXL_ASSIGN_OR_RETURN(float cost_after, EstimateCost(image));
    JXL_DEBUG_V(7, "Cost before: %f  cost after: %f", cost_before, cost_after);
    if (cost_after > cost_before) {
      Transform t = image.transform.back();
      if (!t.Inverse(image, wp_header, pool)) {
        return false;
      }
      image.transform.pop_back();
      did_it = false;
    }
  }
  return did_it;
}

Status try_palettes(Image& gi, int& max_bitdepth, int& maxval,
                    const CompressParams& cparams_,
                    float channel_colors_percent,
                    jxl::ThreadPool* pool = nullptr) {
  if (auto* audit = cparams_.options.histogram_params.direct_audit) {
    if (cparams_.palette_colors > 0 && !gi.channel.empty()) {
      ++audit->palette_attempts;
      Transform palette(TransformId::kPalette);
      palette.begin_c = gi.nb_meta_channels;
      palette.num_c = gi.channel.size() - gi.nb_meta_channels;
      palette.nb_colors = cparams_.palette_colors;
      palette.ordered_palette = true;
      palette.lossy_palette = false;
      palette.lossless_encoder_tools = cparams_.IsLossless() && cparams_.modular_mode;
      palette.predictor = Predictor::Zero;
      if (palette.num_c > 0 && do_transform(gi, palette, weighted::Header(), pool)) {
        ++audit->palette_applied;
      }
    }
    return true;
  }
  float cost_before = 0.f;
  size_t did_palette = 0;
  float nb_pixels = gi.channel[0].w * gi.channel[0].h;
  int nb_chans = gi.channel.size() - gi.nb_meta_channels;
  // arbitrary estimate: 4.8 bpp for 8-bit RGB
  float arbitrary_bpp_estimate = 0.2f * gi.bitdepth * nb_chans;
  const char* palette_force_env = std::getenv("JXL_LOSSLESS_PALETTE_FORCE");
  const bool force_palette = cparams_.IsLossless() && cparams_.modular_mode &&
      !cparams_.lossy_palette && cparams_.palette_colors > 0 &&
      cparams_.palette_colors <= 65536 && palette_force_env &&
      std::strcmp(palette_force_env, "1") == 0;

  if (cparams_.palette_colors != 0 || cparams_.lossy_palette) {
    // when not estimating, assume some arbitrary bpp
    if (cparams_.speed_tier <= SpeedTier::kSquirrel) {
      JXL_ASSIGN_OR_RETURN(cost_before, EstimateCost(gi));
    } else {
      cost_before = nb_pixels * arbitrary_bpp_estimate;
    }
    // all-channel palette (e.g. RGBA)
    if (nb_chans > 1) {
      Transform maybe_palette(TransformId::kPalette);
      maybe_palette.begin_c = gi.nb_meta_channels;
      maybe_palette.num_c = nb_chans;
      // Heuristic choice of max colors for a palette:
      // max_colors = nb_pixels * estimated_bpp_without_palette * 0.0005 +
      //              + nb_pixels / 128 + 128
      //       (estimated_bpp_without_palette = cost_before / nb_pixels)
      // Rationale: small image with large palette is not effective;
      // also if the entropy (estimated bpp) is low (e.g. mostly solid/gradient
      // areas), palette is less useful and may even be counterproductive.
      maybe_palette.nb_colors = std::min(
          static_cast<int>(cost_before * 0.0005f + nb_pixels / 128 + 128),
          std::abs(cparams_.palette_colors));
      if (force_palette) maybe_palette.nb_colors = cparams_.palette_colors;
      maybe_palette.ordered_palette = cparams_.palette_colors >= 0;
      maybe_palette.lossless_encoder_tools = cparams_.IsLossless() && cparams_.modular_mode;
      maybe_palette.lossy_palette =
          (cparams_.lossy_palette && maybe_palette.num_c == 3);
      if (maybe_palette.lossy_palette) {
        maybe_palette.predictor = Predictor::Average4;
      }
      // TODO(veluca): use a custom weighted header if using the weighted
      // predictor.
      if (force_palette) {
        did_palette = do_transform(gi, maybe_palette, weighted::Header(), pool);
      } else {
        JXL_ASSIGN_OR_RETURN(
            did_palette,
            maybe_do_transform(gi, maybe_palette, cparams_, weighted::Header(),
                               cost_before, pool, cparams_.options.zero_tokens));
      }
    }
    // all-minus-one-channel palette (RGB with separate alpha, or CMY with
    // separate K)
    if (!did_palette && nb_chans > 3) {
      Transform maybe_palette_3(TransformId::kPalette);
      maybe_palette_3.begin_c = gi.nb_meta_channels;
      maybe_palette_3.num_c = nb_chans - 1;
      maybe_palette_3.nb_colors = std::min(
          static_cast<int>(cost_before * 0.0005f + nb_pixels / 128 + 128),
          std::abs(cparams_.palette_colors));
      if (force_palette) maybe_palette_3.nb_colors = cparams_.palette_colors;
      maybe_palette_3.ordered_palette = cparams_.palette_colors >= 0;
      maybe_palette_3.lossless_encoder_tools = cparams_.IsLossless() && cparams_.modular_mode;
      maybe_palette_3.lossy_palette = cparams_.lossy_palette;
      if (maybe_palette_3.lossy_palette) {
        maybe_palette_3.predictor = Predictor::Average4;
      }
      if (force_palette) {
        did_palette = do_transform(gi, maybe_palette_3, weighted::Header(), pool);
      } else {
        JXL_ASSIGN_OR_RETURN(
            did_palette,
            maybe_do_transform(gi, maybe_palette_3, cparams_, weighted::Header(),
                               cost_before, pool, cparams_.options.zero_tokens));
      }
    }
  }

  if (channel_colors_percent > 0) {
    // single channel palette (like FLIF's ChannelCompact)
    size_t nb_channels = gi.channel.size() - gi.nb_meta_channels - did_palette;
    int orig_bitdepth = max_bitdepth;
    max_bitdepth = 0;
    if (nb_channels > 0 && (did_palette || cost_before == 0)) {
      if (cparams_.speed_tier < SpeedTier::kSquirrel) {
        JXL_ASSIGN_OR_RETURN(cost_before, EstimateCost(gi));
      } else {
        cost_before = 0;
      }
    }
    for (size_t i = did_palette; i < nb_channels + did_palette; i++) {
      int32_t min;
      int32_t max;
      compute_minmax(gi.channel[gi.nb_meta_channels + i], &min, &max);
      int64_t colors = static_cast<int64_t>(max) - min + 1;
      JXL_DEBUG_V(10, "Channel %" PRIuS ": range=%i..%i", i, min, max);
      Transform maybe_palette_1(TransformId::kPalette);
      maybe_palette_1.begin_c = i + gi.nb_meta_channels;
      maybe_palette_1.num_c = 1;
      // simple heuristic: if less than X percent of the values in the range
      // actually occur, it is probably worth it to do a compaction
      // (but only if the channel palette is less than 6% the size of the
      // image itself)
      maybe_palette_1.nb_colors =
          std::min(static_cast<int>(nb_pixels / 16),
                   static_cast<int>(channel_colors_percent / 100. * colors));
      JXL_ASSIGN_OR_RETURN(
          bool did_ch_palette,
          maybe_do_transform(gi, maybe_palette_1, cparams_, weighted::Header(),
                             cost_before, pool));
      if (did_ch_palette) {
        // effective bit depth is lower, adjust quantization accordingly
        compute_minmax(gi.channel[gi.nb_meta_channels + i], &min, &max);
        if (max < maxval) maxval = max;
        int ch_bitdepth =
            (max > 0 ? CeilLog2Nonzero(static_cast<uint32_t>(max)) : 0);
        if (ch_bitdepth > max_bitdepth) max_bitdepth = ch_bitdepth;
      } else {
        max_bitdepth = orig_bitdepth;
      }
    }
  }
  return true;
}

}  // namespace

StatusOr<std::unique_ptr<ModularFrameEncoder>> ModularFrameEncoder::Create(
    JxlMemoryManager* memory_manager, const FrameHeader& frame_header,
    const CompressParams& cparams_orig, bool streaming_mode,
    bool jpeg_transcode) {
  auto self = std::unique_ptr<ModularFrameEncoder>(
      new ModularFrameEncoder(memory_manager));
  JXL_RETURN_IF_ERROR(
      self->Init(frame_header, cparams_orig, streaming_mode, jpeg_transcode));
  return self;
}

ModularFrameEncoder::ModularFrameEncoder(JxlMemoryManager* memory_manager)
    : memory_manager_(memory_manager) {}

Status ModularFrameEncoder::Init(const FrameHeader& frame_header,
                                 const CompressParams& cparams_orig,
                                 bool streaming_mode, bool jpeg_transcode) {
  frame_dim_ = frame_header.ToFrameDimensions();
  cparams_ = cparams_orig;
  const bool lossless_pixels = cparams_.modular_mode && cparams_.IsLossless();
  const char* jpeg_ma = std::getenv("JXL_LOSSLESS_JPEG_MA_SPEEDUPS");
  // JPEG's integer auxiliary streams can use exact MA cost/count reuse too.
  // Do not infer JPEG input from VarDCT mode, which also covers lossy images.
  cparams_.options.lossless_tree_speedups = lossless_pixels ||
      (jpeg_transcode && (!jpeg_ma || std::strcmp(jpeg_ma, "1") == 0));
  const auto* metadata = frame_header.nonserialized_metadata;
  cparams_.options.lossless_channel_trees = lossless_pixels && !jpeg_transcode && metadata &&
      metadata->m.bit_depth.bits_per_sample <= 8 &&
      metadata->m.bit_depth.exponent_bits_per_sample == 0 &&
      std::all_of(metadata->m.extra_channel_info.begin(), metadata->m.extra_channel_info.end(),
          [](const ExtraChannelInfo& channel) {
            return channel.bit_depth.bits_per_sample <= 8 &&
                   channel.bit_depth.exponent_bits_per_sample == 0;
          });
  const char* final_channel = std::getenv("JXL_MA_CHANNEL_FINAL_ONLY");
  if (final_channel && std::strcmp(final_channel, "1") == 0 &&
      (cparams_.lossless_disposable_probe || cparams_.speed_tier > SpeedTier::kTortoise)) {
    cparams_.options.lossless_channel_trees = false;
  }
  const char* fast_entropy = std::getenv("JXL_LOSSLESS_FAST_ENTROPY");
  cparams_.options.lossless_fast_entropy =
      lossless_pixels &&
      fast_entropy && std::strcmp(fast_entropy, "1") == 0;

  size_t num_streams =
      ModularStreamId::Num(frame_dim_, frame_header.passes.num_passes);

  // Progressive lossless only benefits from levels 2 and higher
  // Lower levels of faster decoding can outperform higher tiers
  // depending on the PC
  if (cparams_.responsive == 1 && cparams_.IsLossless() &&
      cparams_.decoding_speed_tier == 1) {
    cparams_.decoding_speed_tier = 2;
  }
  if (cparams_.responsive == 1 && cparams_.IsLossless()) {
    // RCT selection seems bugged with Squeeze, YCoCg works well.
    if (cparams_.colorspace < 0) {
      cparams_.colorspace = 6;
    }
  }

  if (cparams_.ModularPartIsLossless()) {
    const auto disable_wp = [this] () {
        cparams_.options.wp_tree_mode = ModularOptions::TreeMode::kNoWP;
        if (cparams_.options.predictor == Predictor::Weighted) {
          // Predictor::Best turns to Predictor::Gradient anyways.
          cparams_.options.predictor = Predictor::Gradient;
        }
    };
    switch (cparams_.decoding_speed_tier) {
      case 0:
        cparams_.options.fast_decode_multiplier = 1.001f;
        break;
      case 1:  // No Weighted predictor
        cparams_.options.fast_decode_multiplier = 1.005f;
        disable_wp();
        break;
      case 2: {  // No Weighted predictor and Group size 0 defined in
                 // enc_frame.cc
        cparams_.options.fast_decode_multiplier = 1.015f;
        disable_wp();
        break;
      }
      case 3: {  // Gradient only, Group size 0, and Fast MA tree
        cparams_.options.wp_tree_mode = ModularOptions::TreeMode::kGradientOnly;
        cparams_.options.predictor = Predictor::Gradient;
        break;
      }
      default: {  // Gradient only, Group size 0, and No MA tree
        cparams_.options.wp_tree_mode = ModularOptions::TreeMode::kGradientOnly;
        cparams_.options.predictor = Predictor::Gradient;
        cparams_.options.nb_repeats = 0;
        // Disabling MA Trees sometimes doesn't increase decode speed
        // depending on PC
        break;
      }
    }
  }

  for (size_t i = 0; i < num_streams; ++i) {
    stream_images_.emplace_back(memory_manager_);
  }

  // use a sensible default if nothing explicit is specified:
  // Squeeze for lossy, no squeeze for lossless
  if (cparams_.responsive < 0) {
    if (cparams_.ModularPartIsLossless()) {
      cparams_.responsive = 0;
    } else {
      cparams_.responsive = 1;
    }
  }

  cparams_.options.splitting_heuristics_node_threshold =
      75 + 14 * static_cast<int>(cparams_.speed_tier) +
      10 * cparams_.decoding_speed_tier;

  {
    // Set properties.
    std::vector<uint32_t> prop_order;
    if (cparams_.responsive) {
      // Properties in order of their likelihood of being useful for Squeeze
      // residuals.
      prop_order = {0, 1, 4, 5, 6, 7, 8, 15, 9, 10, 11, 12, 13, 14, 2, 3};
    } else {
      // Same, but for the non-Squeeze case.
      prop_order = {0, 1, 15, 9, 10, 11, 12, 13, 14, 2, 3, 4, 5, 6, 7, 8};
      // if few groups, don't use group as a property
      if (num_streams < 30 && cparams_.speed_tier > SpeedTier::kTortoise &&
          cparams_orig.ModularPartIsLossless()) {
        prop_order.erase(prop_order.begin() + 1);
      }
    }
    int max_properties = std::min<int>(
        cparams_.options.max_properties,
        static_cast<int>(
            frame_header.nonserialized_metadata->m.num_extra_channels) +
            (frame_header.nonserialized_metadata->m.color_encoding.IsGray() ? 0 : 2));
    switch (cparams_.speed_tier) {
      case SpeedTier::kHare:
        cparams_.options.splitting_heuristics_properties.assign(
            prop_order.begin(), prop_order.begin() + 4);
        cparams_.options.max_property_values = 48;
        cparams_.options.nb_repeats *= 0.5f;
        break;
      case SpeedTier::kWombat:
        cparams_.options.splitting_heuristics_properties.assign(
            prop_order.begin(), prop_order.begin() + 5);
        cparams_.options.max_property_values = 64;
        cparams_.options.nb_repeats *= 0.7f;
        break;
      case SpeedTier::kSquirrel:
        cparams_.options.splitting_heuristics_properties.assign(
            prop_order.begin(), prop_order.begin() + 7);
        cparams_.options.max_property_values = 96;
        break;
      case SpeedTier::kKitten:
        cparams_.options.splitting_heuristics_properties.assign(
            prop_order.begin(), prop_order.begin() + 10);
        cparams_.options.max_property_values = 128;
        cparams_.options.nb_repeats *= 1.1f;
        break;
      case SpeedTier::kGlacier:
      case SpeedTier::kTortoise:
        cparams_.options.splitting_heuristics_properties = prop_order;
        cparams_.options.max_property_values = 256;
        cparams_.options.nb_repeats *= 1.3f;
        break;
      default:
        cparams_.options.splitting_heuristics_properties.assign(
            prop_order.begin(), prop_order.begin() + 3);
        cparams_.options.max_property_values = 32;
        cparams_.options.nb_repeats *= 0.3f;
        break;
    }
    if (cparams_.speed_tier > SpeedTier::kTortoise) {
      // Gradient in previous channels.
      for (int i = 0; i < max_properties; i++) {
        cparams_.options.splitting_heuristics_properties.push_back(
            kNumNonrefProperties + i * 4 + 3);
      }
    } else {
      // All the extra properties in Tortoise mode.
      for (int i = 0; i < max_properties * 4; i++) {
        cparams_.options.splitting_heuristics_properties.push_back(
            kNumNonrefProperties + i);
      }
    }
  }
  cparams_.options.nb_repeats = std::min(1.0f, cparams_.options.nb_repeats);

  if ((cparams_.options.predictor == Predictor::Average0 ||
       cparams_.options.predictor == Predictor::Average1 ||
       cparams_.options.predictor == Predictor::Average2 ||
       cparams_.options.predictor == Predictor::Average3 ||
       cparams_.options.predictor == Predictor::Average4 ||
       cparams_.options.predictor == Predictor::Weighted) &&
      !cparams_.ModularPartIsLossless()) {
    // Lossy + Average/Weighted predictors does not work, so switch to default
    // predictors.
    cparams_.options.predictor = kUndefinedPredictor;
  }

  if (cparams_.options.predictor == kUndefinedPredictor) {
    // no explicit predictor(s) given, set a good default
    if ((cparams_.speed_tier <= SpeedTier::kGlacier ||
         cparams_.modular_mode == false) &&
        cparams_.IsLossless() && cparams_.responsive == JXL_FALSE) {
      // TODO(veluca): allow all predictors that don't break residual
      // multipliers in lossy mode.
      cparams_.options.predictor = Predictor::Variable;
    } else if (cparams_.responsive || cparams_.lossy_palette) {
      // zero predictor for Squeeze residues and lossy palette indices
      // TODO: Try adding 'Squeezed' predictor set, with the most
      // common predictors used by Variable in squeezed images, including none.
      cparams_.options.predictor = Predictor::Zero;
    } else if (!cparams_.IsLossless()) {
      // If not responsive and lossy. TODO(veluca): use near_lossless instead?
      cparams_.options.predictor = Predictor::Gradient;
    } else if (cparams_.speed_tier < SpeedTier::kFalcon) {
      // try median and weighted predictor for anything else
      cparams_.options.predictor = Predictor::Best;
    } else if (cparams_.speed_tier == SpeedTier::kFalcon) {
      // just weighted predictor in falcon mode
      cparams_.options.predictor = Predictor::Weighted;
    } else if (cparams_.speed_tier > SpeedTier::kFalcon) {
      // just gradient predictor in thunder mode
      cparams_.options.predictor = Predictor::Gradient;
    }
  } else {
    if (cparams_.lossy_palette) cparams_.options.predictor = Predictor::Zero;
  }
  if (!cparams_.ModularPartIsLossless()) {
    if (cparams_.options.predictor == Predictor::Weighted ||
        cparams_.options.predictor == Predictor::Variable ||
        cparams_.options.predictor == Predictor::Best)
      cparams_.options.predictor = Predictor::Zero;
  }
  tree_splits_.push_back(0);
  if (cparams_.modular_mode == false) {
    JXL_ASSIGN_OR_RETURN(ModularStreamId qt0, ModularStreamId::QuantTable(0));
    cparams_.options.fast_decode_multiplier = 1.0f;
    tree_splits_.push_back(ModularStreamId::VarDCTDC(0).ID(frame_dim_));
    tree_splits_.push_back(ModularStreamId::ModularDC(0).ID(frame_dim_));
    tree_splits_.push_back(ModularStreamId::ACMetadata(0).ID(frame_dim_));
    tree_splits_.push_back(qt0.ID(frame_dim_));
    tree_splits_.push_back(ModularStreamId::ModularAC(0, 0).ID(frame_dim_));
    ac_metadata_size.resize(frame_dim_.num_dc_groups);
    extra_dc_precision.resize(frame_dim_.num_dc_groups);
  }
  tree_splits_.push_back(num_streams);
  cparams_.options.max_chan_size = frame_dim_.group_dim;
  cparams_.options.group_dim = frame_dim_.group_dim;

  ApplyMaSearchSettings(&cparams_.options);

  // TODO(veluca): figure out how to use different predictor sets per channel.
  stream_options_.resize(num_streams, cparams_.options);

  stream_options_[0] = cparams_.options;
  if (cparams_.speed_tier == SpeedTier::kFalcon) {
    stream_options_[0].tree_kind = ModularOptions::TreeKind::kWPFixedDC;
  } else if (cparams_.speed_tier == SpeedTier::kThunder) {
    stream_options_[0].tree_kind = ModularOptions::TreeKind::kGradientFixedDC;
  }
  stream_options_[0].histogram_params =
      HistogramParams::ForModular(cparams_, {}, streaming_mode);
  return true;
}

Status ModularFrameEncoder::ComputeEncodingData(
    const FrameHeader& frame_header, const ImageMetadata& metadata,
    Image3F* JXL_RESTRICT color, const std::vector<ImageF>& extra_channels,
    const Rect& group_rect, const FrameDimensions& patch_dim,
    const Rect& frame_area_rect, PassesEncoderState* JXL_RESTRICT enc_state,
    const JxlCmsInterface& cms, ThreadPool* pool, AuxOut* aux_out,
    bool do_color) {
  LosslessStageTimer timer(cparams_, frame_dim_, "transforms");
  LosslessProfileScope profile(cparams_.options.histogram_params.lossless_profile,
                               "modular.transforms");
  JxlMemoryManager* memory_manager = enc_state->memory_manager();
  JXL_DEBUG_V(6, "Computing modular encoding data for frame %s",
              frame_header.DebugString().c_str());

  bool groupwise = enc_state->streaming_mode;

  if (do_color && frame_header.loop_filter.gab && !groupwise) {
    float w = 0.9908511000000001f;
    float weights[3] = {w, w, w};
    JXL_RETURN_IF_ERROR(GaborishInverse(color, Rect(*color), weights, pool));
  }

  if (do_color && metadata.bit_depth.bits_per_sample <= 16 &&
      cparams_.speed_tier < SpeedTier::kCheetah &&
      cparams_.decoding_speed_tier < 2 && !groupwise) {
    JXL_RETURN_IF_ERROR(FindBestPatchDictionary(
        *color, enc_state, cms, nullptr, aux_out,
        cparams_.color_transform == ColorTransform::kXYB));
    JXL_RETURN_IF_ERROR(PatchDictionaryEncoder::SubtractFrom(
        enc_state->shared.image_features.patches, color));
  }

  if (cparams_.custom_splines.HasAny()) {
    PassesSharedState& shared = enc_state->shared;
    ImageFeatures& image_features = shared.image_features;
    image_features.splines.SetData(cparams_.custom_splines);
  }

  // Convert ImageBundle to modular Image object
  const size_t xsize = patch_dim.xsize;
  const size_t ysize = patch_dim.ysize;

  int nb_chans = 3;
  if (metadata.color_encoding.IsGray() &&
      cparams_.color_transform == ColorTransform::kNone) {
    nb_chans = 1;
  }
  if (!do_color) nb_chans = 0;

  nb_chans += extra_channels.size();

  bool fp = metadata.bit_depth.floating_point_sample &&
            cparams_.color_transform != ColorTransform::kXYB;

  // bits_per_sample is just metadata for XYB images.
  if (metadata.bit_depth.bits_per_sample >= 32 && do_color &&
      cparams_.color_transform != ColorTransform::kXYB) {
    if (metadata.bit_depth.bits_per_sample == 32 && fp == false) {
      return JXL_FAILURE("uint32_t not supported in enc_modular");
    } else if (metadata.bit_depth.bits_per_sample > 32) {
      return JXL_FAILURE("bits_per_sample > 32 not supported");
    }
  }

  // in the non-float case, there is an implicit 0 sign bit
  int max_bitdepth =
      do_color ? metadata.bit_depth.bits_per_sample + (fp ? 0 : 1) : 0;
  Image& gi = stream_images_[0];
  JXL_ASSIGN_OR_RETURN(
      gi, Image::Create(memory_manager, xsize, ysize,
                        metadata.bit_depth.bits_per_sample, nb_chans));
  int c = 0;
  if (cparams_.color_transform == ColorTransform::kXYB &&
      cparams_.modular_mode == true) {
    float enc_factors[3] = {65536.0f, 4096.0f, 4096.0f};
    if (cparams_.butteraugli_distance > 0 && !cparams_.responsive) {
      // quantize XYB here and then treat it as a lossless image
      enc_factors[0] *= 1.f / (1.f + 23.f * cparams_.butteraugli_distance);
      enc_factors[1] *= 1.f / (1.f + 14.f * cparams_.butteraugli_distance);
      enc_factors[2] *= 1.f / (1.f + 14.f * cparams_.butteraugli_distance);
      cparams_.butteraugli_distance = 0;
    }
    if (cparams_.manual_xyb_factors.size() == 3) {
      JXL_RETURN_IF_ERROR(DequantMatricesSetCustomDC(
          memory_manager, &enc_state->shared.matrices,
          cparams_.manual_xyb_factors.data()));
      // TODO(jon): update max_bitdepth in this case
    } else {
      JXL_RETURN_IF_ERROR(DequantMatricesSetCustomDC(
          memory_manager, &enc_state->shared.matrices, enc_factors));
      max_bitdepth = 12;
    }
  }
  pixel_type maxval = gi.bitdepth < 32 ? (1u << gi.bitdepth) - 1 : 0;
  if (do_color) {
    for (; c < 3; c++) {
      if (metadata.color_encoding.IsGray() &&
          cparams_.color_transform == ColorTransform::kNone &&
          c != (cparams_.color_transform == ColorTransform::kXYB ? 1 : 0))
        continue;
      int c_out = c;
      // XYB is encoded as YX(B-Y)
      if (cparams_.color_transform == ColorTransform::kXYB && c < 2)
        c_out = 1 - c_out;
      double factor = maxval;
      if (cparams_.color_transform == ColorTransform::kXYB)
        factor = enc_state->shared.matrices.InvDCQuant(c);
      if (c == 2 && cparams_.color_transform == ColorTransform::kXYB) {
        JXL_ENSURE(!fp);
        for (size_t y = 0; y < ysize; ++y) {
          const float* const JXL_RESTRICT row_in = color->PlaneRow(c, y);
          pixel_type* const JXL_RESTRICT row_out = gi.channel[c_out].Row(y);
          pixel_type* const JXL_RESTRICT row_Y = gi.channel[0].Row(y);
          for (size_t x = 0; x < xsize; ++x) {
            // TODO(eustas): check if std::roundf is appropriate
            row_out[x] = row_in[x] * factor + 0.5f;
            row_out[x] -= row_Y[x];
          }
        }
      } else {
        int bits = metadata.bit_depth.bits_per_sample;
        int exp_bits = metadata.bit_depth.exponent_bits_per_sample;
        gi.channel[c_out].hshift = frame_header.chroma_subsampling.HShift(c);
        gi.channel[c_out].vshift = frame_header.chroma_subsampling.VShift(c);
        size_t xsize_shifted = DivCeil(xsize, 1 << gi.channel[c_out].hshift);
        size_t ysize_shifted = DivCeil(ysize, 1 << gi.channel[c_out].vshift);
        JXL_RETURN_IF_ERROR(
            gi.channel[c_out].shrink(xsize_shifted, ysize_shifted));
        const auto process_row = [&](const int task,
                                     const int thread) -> Status {
          const size_t y = task;
          const float* const JXL_RESTRICT row_in =
              color->PlaneRow(c, y + group_rect.y0()) + group_rect.x0();
          pixel_type* const JXL_RESTRICT row_out = gi.channel[c_out].Row(y);
          JXL_RETURN_IF_ERROR(float_to_int(row_in, row_out, xsize_shifted, bits,
                                           exp_bits, fp, factor));
          return true;
        };
        JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, ysize_shifted,
                                      ThreadPool::NoInit, process_row,
                                      "float2int"));
      }
    }
    if (metadata.color_encoding.IsGray() &&
        cparams_.color_transform == ColorTransform::kNone)
      c = 1;
  }

  for (size_t ec = 0; ec < extra_channels.size(); ec++, c++) {
    const ExtraChannelInfo& eci = metadata.extra_channel_info[ec];
    size_t ecups = frame_header.extra_channel_upsampling[ec];
    JXL_RETURN_IF_ERROR(
        gi.channel[c].shrink(DivCeil(patch_dim.xsize_upsampled, ecups),
                             DivCeil(patch_dim.ysize_upsampled, ecups)));
    gi.channel[c].hshift = gi.channel[c].vshift =
        CeilLog2Nonzero(ecups) - CeilLog2Nonzero(frame_header.upsampling);

    int bits = eci.bit_depth.bits_per_sample;
    int exp_bits = eci.bit_depth.exponent_bits_per_sample;
    bool ec_fp = eci.bit_depth.floating_point_sample;
    double factor = (ec_fp ? 1 : ((1u << eci.bit_depth.bits_per_sample) - 1));
    if (bits + (ec_fp ? 0 : 1) > max_bitdepth) {
      max_bitdepth = bits + (ec_fp ? 0 : 1);
    }
    const auto process_row = [&](const int task, const int thread) -> Status {
      const size_t y = task;
      const float* const JXL_RESTRICT row_in =
          extra_channels[ec].Row(y + group_rect.y0()) + group_rect.x0();
      pixel_type* const JXL_RESTRICT row_out = gi.channel[c].Row(y);
      JXL_RETURN_IF_ERROR(float_to_int(row_in, row_out,
                                       gi.channel[c].plane.xsize(), bits,
                                       exp_bits, ec_fp, factor));
      return true;
    };
    JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, gi.channel[c].plane.ysize(),
                                  ThreadPool::NoInit, process_row,
                                  "float2int"));
  }
  JXL_ENSURE(c == nb_chans);

  int level_max_bitdepth = (cparams_.level == 5 ? 16 : 32);
  if (max_bitdepth > level_max_bitdepth) {
    return JXL_FAILURE(
        "Bitdepth too high for level %i (need %i bits, have only %i in this "
        "level)",
        cparams_.level, max_bitdepth, level_max_bitdepth);
  }

  // Set options and apply transformations
  if (!cparams_.ModularPartIsLossless()) {
    if (cparams_.palette_colors != 0) {
      JXL_DEBUG_V(3, "Lossy encode, not doing palette transforms");
    }
    if (cparams_.color_transform == ColorTransform::kXYB) {
      cparams_.channel_colors_pre_transform_percent = 0;
    }
    cparams_.channel_colors_percent = 0;
    cparams_.palette_colors = 0;
    cparams_.lossy_palette = false;
  }

  if (!ReplayingFinalPlan()) {
  // Speculative transform costs are not final decisions.
  FinalPlanScope suspend(nullptr);
  // Global palette transforms
  float channel_colors_percent = 0;
  if (!cparams_.lossy_palette &&
      (cparams_.speed_tier <= SpeedTier::kThunder ||
       (do_color && metadata.bit_depth.bits_per_sample > 8))) {
    channel_colors_percent = cparams_.channel_colors_pre_transform_percent;
  }
    // Global palette causes bad progressive loading due to interpolation
    // but lossy palette is still required for JXL art.
  if (!groupwise && (cparams_.lossy_palette ||
      !(cparams_.responsive && cparams_.ModularPartIsLossless()))) {
    JXL_RETURN_IF_ERROR(try_palettes(gi, max_bitdepth, maxval, cparams_,
                                     channel_colors_percent, pool));
  }

  // don't do an RCT if we're short on bits
  if (cparams_.color_transform == ColorTransform::kNone && do_color &&
      gi.channel.size() - gi.nb_meta_channels >= 3 &&
      max_bitdepth + 1 < level_max_bitdepth) {
    if (cparams_.colorspace < 0 && (!cparams_.ModularPartIsLossless() ||
                                    cparams_.speed_tier > SpeedTier::kHare)) {
      Transform ycocg{TransformId::kRCT};
      ycocg.rct_type = 6;
      ycocg.begin_c = gi.nb_meta_channels;
      do_transform(gi, ycocg, weighted::Header(), pool);
      max_bitdepth++;
    } else if (cparams_.colorspace > 0) {
      Transform sg(TransformId::kRCT);
      sg.begin_c = gi.nb_meta_channels;
      sg.rct_type = cparams_.colorspace;
      do_transform(gi, sg, weighted::Header(), pool);
      max_bitdepth++;
    }
  }

  if (cparams_.move_to_front_from_channel > 0) {
    for (size_t tgt = 0;
         tgt + cparams_.move_to_front_from_channel < gi.channel.size(); tgt++) {
      size_t pos = cparams_.move_to_front_from_channel;
      while (pos > 0) {
        Transform move(TransformId::kRCT);
        if (pos == 1) {
          move.begin_c = tgt;
          move.rct_type = 28;  // RGB -> GRB
          pos -= 1;
        } else {
          move.begin_c = tgt + pos - 2;
          move.rct_type = 14;  // RGB -> BRG
          pos -= 2;
        }
        do_transform(gi, move, weighted::Header(), pool);
      }
    }
  }

  // don't do squeeze if we don't have some spare bits
  if (!groupwise && cparams_.responsive && !gi.channel.empty() &&
      max_bitdepth + 2 < level_max_bitdepth) {
    Transform t(TransformId::kSqueeze);
    // Check if default squeeze parameters are ok.
    std::vector<SqueezeParams> params;
    DefaultSqueezeParameters(&params, gi);
    // If image is smaller than group_dim, then default squeeze parameters
    // are not going too far. Else, channel size don't turn zero. Thus we only
    // check if tile does not go to zero-dim.
    size_t shift_cap = 7 + frame_header.group_size_shift;
    size_t hshift = 0;
    size_t vshift = 0;
    for (size_t i = 0; i < params.size(); ++i) {
      if (params[i].horizontal) {
        hshift++;
      } else {
        vshift++;
      }
      size_t dc_boost = (std::min(hshift, vshift) >= 3) ? 3 : 0;
      // In case we squeeze too much, truncate squeeze script.
      if (std::max(hshift, vshift) > shift_cap + dc_boost) {
        params.resize(i - 1);
        t.squeezes = params;
        break;
      }
    }
    do_transform(gi, t, weighted::Header(), pool);
    max_bitdepth += 2;
  }

  if (max_bitdepth + 1 > level_max_bitdepth) {
    // force no group RCTs if we don't have a spare bit
    cparams_.colorspace = 0;
  }
  JXL_ENSURE(max_bitdepth <= level_max_bitdepth);
  }
  if (CurrentFinalPlan()) {
    JXL_RETURN_IF_ERROR(FinalTransforms(gi, 0));
    std::vector<int64_t> state{max_bitdepth, maxval, cparams_.colorspace};
    JXL_RETURN_IF_ERROR(FinalPlanExchange("global_state", &state));
    JXL_ENSURE(state.size() == 3 && state[0] >= 0 && state[0] <= 32 &&
               state[1] >= INT32_MIN && state[1] <= INT32_MAX &&
               state[2] >= -1 && state[2] < 42);
    max_bitdepth = state[0]; maxval = state[1]; cparams_.colorspace = state[2];
  }

  if (!cparams_.ModularPartIsLossless()) {
    quants_.resize(gi.channel.size(), 1);
    float quantizer = 0.25f;
    if (!cparams_.responsive) {
      JXL_DEBUG_V(1,
                  "Warning: lossy compression without Squeeze "
                  "transform is just color quantization.");
      quantizer *= 0.1f;
    }
    float bitdepth_correction = 1.f;
    if (cparams_.color_transform != ColorTransform::kXYB) {
      bitdepth_correction = maxval / 255.f;
    }
    std::vector<float> quantizers;
    for (size_t i = 0; i < 3; i++) {
      float dist = cparams_.butteraugli_distance;
      quantizers.push_back(quantizer * powf(dist, 1.2) * bitdepth_correction);
    }
    for (size_t i = 0; i < extra_channels.size(); i++) {
      int ec_bitdepth =
          metadata.extra_channel_info[i].bit_depth.bits_per_sample;
      pixel_type ec_maxval = ec_bitdepth < 32 ? (1u << ec_bitdepth) - 1 : 0;
      bitdepth_correction = ec_maxval / 255.f;
      float dist = 0;
      if (i < cparams_.ec_distance.size()) dist = cparams_.ec_distance[i];
      if (dist < 0) dist = cparams_.butteraugli_distance;
      quantizers.push_back(quantizer * dist * bitdepth_correction);
    }
    if (cparams_.options.nb_repeats == 0) {
      return JXL_FAILURE("nb_repeats = 0 not supported with modular lossy!");
    }
    for (uint32_t i = gi.nb_meta_channels; i < gi.channel.size(); i++) {
      Channel& ch = gi.channel[i];
      int shift = ch.hshift + ch.vshift;  // number of pixel halvings
      if (shift > 16) shift = 16;
      if (shift > 0) shift--;
      int component = (do_color ? 0 : 3) + ch.component;
      int q;
      if (cparams_.color_transform == ColorTransform::kXYB && component < 3) {
        q = quantizers[component] * squeeze_quality_factor_xyb *
            squeeze_xyb_qtable[component][shift];
        if (component == 0) q *= squeeze_quality_factor_y;
      } else {
        if (cparams_.colorspace != 0 && component > 0 && component < 3) {
          q = quantizers[component] * squeeze_quality_factor *
              squeeze_chroma_qtable[shift];
        } else {
          q = quantizers[component] * squeeze_quality_factor *
              squeeze_luma_factor * squeeze_luma_qtable[shift];
        }
      }
      if (q < 1) q = 1;
      QuantizeChannel(gi.channel[i], q);
      quants_[i] = q;
    }
  }

  // Fill other groups.
  // DC
  for (size_t group_id = 0; group_id < patch_dim.num_dc_groups; group_id++) {
    const size_t rgx = group_id % patch_dim.xsize_dc_groups;
    const size_t rgy = group_id / patch_dim.xsize_dc_groups;
    const Rect rect(rgx * patch_dim.dc_group_dim, rgy * patch_dim.dc_group_dim,
                    patch_dim.dc_group_dim, patch_dim.dc_group_dim);
    size_t gx = rgx + frame_area_rect.x0() / 2048;
    size_t gy = rgy + frame_area_rect.y0() / 2048;
    size_t real_group_id = gy * frame_dim_.xsize_dc_groups + gx;
    // minShift==3 because (frame_dim.dc_group_dim >> 3) == frame_dim.group_dim
    // maxShift==1000 is infinity
    stream_params_.push_back(
        GroupParams{rect, 3, 1000, ModularStreamId::ModularDC(real_group_id)});
  }
  // AC global -> nothing.
  // AC
  for (size_t group_id = 0; group_id < patch_dim.num_groups; group_id++) {
    const size_t rgx = group_id % patch_dim.xsize_groups;
    const size_t rgy = group_id / patch_dim.xsize_groups;
    const Rect mrect(rgx * patch_dim.group_dim, rgy * patch_dim.group_dim,
                     patch_dim.group_dim, patch_dim.group_dim);
    size_t gx = rgx + frame_area_rect.x0() / (frame_dim_.group_dim);
    size_t gy = rgy + frame_area_rect.y0() / (frame_dim_.group_dim);
    size_t real_group_id = gy * frame_dim_.xsize_groups + gx;
    for (size_t i = 0; i < enc_state->progressive_splitter.GetNumPasses();
         i++) {
      int maxShift;
      int minShift;
      frame_header.passes.GetDownsamplingBracket(i, minShift, maxShift);
      stream_params_.push_back(
          GroupParams{mrect, minShift, maxShift,
                      ModularStreamId::ModularAC(real_group_id, i)});
    }
  }
  // if there's only one group, everything ends up in GlobalModular
  // in that case, also try RCTs/WP params for the one group
  if (stream_params_.size() == 2) {
    stream_params_.push_back(GroupParams{Rect(0, 0, xsize, ysize), 0, 1000,
                                         ModularStreamId::Global()});
  }
  gi_channel_.resize(stream_images_.size());

  const char* wp_parallel_env = std::getenv("JXL_LOSSLESS_WP_PARALLEL");
  const bool parallel_wp = pool && cparams_.IsLossless() &&
      cparams_.modular_mode && frame_dim_.num_groups == 1 && !CurrentFinalPlan() &&
      wp_parallel_env && std::strcmp(wp_parallel_env, "1") == 0;
  const bool wave_wp = pool && cparams_.IsLossless() &&
      cparams_.modular_mode && !CurrentFinalPlan() &&
      wp_parallel_env && (std::strcmp(wp_parallel_env, "2") == 0 ||
                          std::strcmp(wp_parallel_env, "3") == 0);
  std::vector<size_t> wp_modes(wave_wp ? stream_images_.size() : 0);

  const auto process_row = [&](const uint32_t i,
                               size_t /* thread */) -> Status {
    size_t stream = stream_params_[i].id.ID(frame_dim_);
    if (stream != 0) {
      stream_options_[stream] = stream_options_[0];
    }
    JXL_RETURN_IF_ERROR(PrepareStreamParams(
        stream_params_[i].rect, cparams_, stream_params_[i].minShift,
        stream_params_[i].maxShift, stream_params_[i].id, do_color, groupwise,
        parallel_wp ? pool : nullptr, wave_wp ? &wp_modes[stream] : nullptr));
    return true;
  };
  if (parallel_wp) {
    // A single image group cannot use group parallelism. Its predictor modes
    // are independent, so lend the pool to their search without nesting it.
    for (size_t i = 0; i < stream_params_.size(); ++i) {
      JXL_RETURN_IF_ERROR(process_row(static_cast<uint32_t>(i), 0));
    }
  } else {
    JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, stream_params_.size(),
                                  ThreadPool::NoInit, process_row,
                                  "ChooseParams"));
  }
  if (wave_wp) JXL_RETURN_IF_ERROR(PrepareWPModes(wp_modes, pool));
  {
    // Clear out channels that have been copied to groups.
    Image& full_image = stream_images_[0];
    size_t ch = full_image.nb_meta_channels;
    for (; ch < full_image.channel.size(); ch++) {
      Channel& fc = full_image.channel[ch];
      if (fc.w > frame_dim_.group_dim || fc.h > frame_dim_.group_dim) break;
    }
    for (; ch < full_image.channel.size(); ch++) {
      // TODO(eustas): shrink / assign channel to keep size consistency
      full_image.channel[ch].plane = ImageI();
    }
  }

  JXL_RETURN_IF_ERROR(ValidateChannelDimensions(gi, stream_options_[0]));
  return true;
}

Status ModularFrameEncoder::ComputeTree(ThreadPool* pool) {
  if (auto* audit = cparams_.options.histogram_params.direct_audit) {
    ++audit->trees;
    if (cparams_.custom_fixed_tree.empty() && !audit->allow_ma_learning) {
      ++audit->forbidden_searches;
      return JXL_FAILURE("Direct plan requires an explicit MA tree");
    }
  }
  LosslessStageTimer timer(cparams_, frame_dim_, "tree");
  LosslessProfileScope profile(cparams_.options.histogram_params.lossless_profile,
                               "modular.tree");
  std::vector<ModularMultiplierInfo> multiplier_info;
  if (!quants_.empty()) {
    for (uint32_t stream_id = 0; stream_id < stream_images_.size();
         stream_id++) {
      // skip non-modular stream_ids
      if (stream_id > 0 && gi_channel_[stream_id].empty()) continue;
      const Image& image = stream_images_[stream_id];
      const ModularOptions& options = stream_options_[stream_id];
      for (uint32_t i = image.nb_meta_channels; i < image.channel.size(); i++) {
        if (image.channel[i].w > options.max_chan_size ||
            image.channel[i].h > options.max_chan_size) {
          continue;
        }
        if (stream_id > 0 && gi_channel_[stream_id].empty()) continue;
        size_t ch_id = stream_id == 0
                           ? i
                           : gi_channel_[stream_id][i - image.nb_meta_channels];
        uint32_t q = quants_[ch_id];
        // Inform the tree splitting heuristics that each channel in each group
        // used this quantization factor. This will produce a tree with the
        // given multipliers.
        if (multiplier_info.empty() ||
            multiplier_info.back().range[1][0] != stream_id ||
            multiplier_info.back().multiplier != q) {
          StaticPropRange range;
          range[0] = {{i, i + 1}};
          range[1] = {{stream_id, stream_id + 1}};
          multiplier_info.push_back({range, q});
        } else {
          // Previous channel in the same group had the same quantization
          // factor. Don't provide two different ranges, as that creates
          // unnecessary nodes.
          multiplier_info.back().range[0][1] = i + 1;
        }
      }
    }
    // Merge group+channel settings that have the same channels and quantization
    // factors, to avoid unnecessary nodes.
    std::sort(multiplier_info.begin(), multiplier_info.end(),
              [](ModularMultiplierInfo a, ModularMultiplierInfo b) {
                return std::make_tuple(a.range, a.multiplier) <
                       std::make_tuple(b.range, b.multiplier);
              });
    size_t new_num = 1;
    for (size_t i = 1; i < multiplier_info.size(); i++) {
      ModularMultiplierInfo& prev = multiplier_info[new_num - 1];
      ModularMultiplierInfo& cur = multiplier_info[i];
      if (prev.range[0] == cur.range[0] && prev.multiplier == cur.multiplier &&
          prev.range[1][1] == cur.range[1][0]) {
        prev.range[1][1] = cur.range[1][1];
      } else {
        multiplier_info[new_num++] = multiplier_info[i];
      }
    }
    multiplier_info.resize(new_num);
  }

  if (cparams_.options.histogram_params.direct_audit) {
    const char* samples = std::getenv("JXL_DIRECT_SAMPLES");
    const char* policy = std::getenv("JXL_DIRECT_POLICY");
    if (samples && *samples && policy && std::strcmp(policy, "neural") == 0) {
      JXL_RETURN_IF_ERROR(ExportNeuralTreeSamples(
          stream_images_.data(), stream_options_.data(), stream_images_.size(), samples));
    }
  }
  if (ReplayingFinalPlan()) {
    JXL_RETURN_IF_ERROR(FinalTree(&tree_));
  } else if (!cparams_.custom_fixed_tree.empty()) {
    tree_ = cparams_.custom_fixed_tree;
  } else if (cparams_.speed_tier < SpeedTier::kFalcon ||
             !cparams_.modular_mode) {
    // Avoid creating a tree with leaves that don't correspond to any pixels.
    std::vector<size_t> useful_splits;
    useful_splits.reserve(tree_splits_.size());
    for (size_t chunk = 0; chunk < tree_splits_.size() - 1; chunk++) {
      bool has_pixels = false;
      size_t start = tree_splits_[chunk];
      size_t stop = tree_splits_[chunk + 1];
      for (size_t i = start; i < stop; i++) {
        if (!stream_images_[i].empty()) has_pixels = true;
      }
      if (has_pixels) {
        useful_splits.push_back(tree_splits_[chunk]);
      }
    }
    // Don't do anything if modular mode does not have any pixels in this image
    if (useful_splits.empty()) return true;
    useful_splits.push_back(tree_splits_.back());

    const char* shard_env = std::getenv("JXL_MA_TREE_SHARDS");
    const size_t requested_shards = shard_env && shard_env[0] && shard_env[1] == '\0' &&
        (shard_env[0] == '2' || shard_env[0] == '4' || shard_env[0] == '8')
        ? static_cast<size_t>(shard_env[0] - '0') : 1;
    // Experimental independent learners, not a faster replay of the same tree.
    // Keep JPEG, lossy and model/audit paths on their original partitioning.
    if (requested_shards > 1 && useful_splits.size() == 2 &&
        cparams_.modular_mode && cparams_.IsLossless() &&
        !cparams_.options.ma_search &&
        !cparams_.options.histogram_params.direct_audit &&
        stream_options_[useful_splits.front()].tree_kind ==
            ModularOptions::TreeKind::kLearn) {
      const size_t first = useful_splits.front();
      std::vector<uint64_t> work(useful_splits.back() - first);
      for (size_t i = first; i < useful_splits.back(); ++i) {
        const Image& image = stream_images_[i];
        for (size_t c = 0; c < image.channel.size(); ++c) {
          const Channel& channel = image.channel[c];
          if (c >= image.nb_meta_channels &&
              (channel.w > stream_options_[i].max_chan_size ||
               channel.h > stream_options_[i].max_chan_size)) break;
          work[i - first] += static_cast<uint64_t>(channel.w) * channel.h;
        }
      }
      useful_splits = MaTreeGroupBoundaries(work, requested_shards);
      for (size_t& boundary : useful_splits) boundary += first;
    }
    profile.Set("requested_tree_shards", requested_shards);
    profile.Set("tree_chunks", useful_splits.size() - 1);

    const char* split_env = std::getenv("JXL_MA_SPLIT_PARALLEL");
    const char* gather_env = std::getenv("JXL_MA_GATHER_PARALLEL");
    const bool single_parallel_learner =
        pool && useful_splits.size() == 2 && cparams_.modular_mode &&
        cparams_.IsLossless() && !cparams_.options.ma_search &&
        !cparams_.options.histogram_params.direct_audit;
    const bool parallel_split = single_parallel_learner &&
        split_env && std::strcmp(split_env, "1") == 0;
    const bool parallel_gather = single_parallel_learner &&
        gather_env && std::strcmp(gather_env, "1") == 0;
    const char* chunks_env = std::getenv("JXL_MA_CHUNK_PARALLEL");
    // Init enables these speedups for VarDCT only with actual JPEG input.
    const bool cooperative_chunks = pool && useful_splits.size() > 2 &&
        !cparams_.modular_mode &&
        cparams_.options.lossless_tree_speedups && !cparams_.options.ma_search &&
        !cparams_.options.histogram_params.direct_audit && !CurrentFinalPlan() &&
        split_env && std::strcmp(split_env, "1") == 0 &&
        (!chunks_env || std::strcmp(chunks_env, "1") == 0);
    profile.Set("parallel_split", parallel_split);
    profile.Set("parallel_gather", parallel_gather);
    profile.Set("cooperative_chunks", cooperative_chunks);

    std::vector<Tree> trees(useful_splits.size() - 1);
    const auto process_chunk = [&](size_t chunk, ThreadPool* split_pool) -> Status {
      LosslessProfileScope chunk_profile(profile.Id(), "modular.tree_chunk");
      chunk_profile.Set("chunk", chunk);
      uint32_t start = useful_splits[chunk];
      uint32_t stop = useful_splits[chunk + 1];
      while (start < stop && stream_images_[start].empty()) ++start;
      while (start < stop && stream_images_[stop - 1].empty()) --stop;

      if (stream_options_[start].tree_kind ==
          ModularOptions::TreeKind::kLearn) {
        JXL_ASSIGN_OR_RETURN(
            trees[chunk],
            LearnTree(stream_images_.data(), stream_options_.data(), start,
                      stop, multiplier_info, split_pool,
                      parallel_gather ? pool : nullptr));
      } else {
        size_t total_pixels = 0;
        for (size_t i = start; i < stop; i++) {
          for (const Channel& ch : stream_images_[i].channel) {
            total_pixels += ch.w * ch.h;
          }
        }
        total_pixels = std::max<size_t>(total_pixels, 1);

        trees[chunk] = PredefinedTree(stream_options_[start].tree_kind,
                                      total_pixels, 8, 0);
      }
      return true;
    };
    if (cooperative_chunks) {
      // Multiple learners share the caller's workers without reentering its
      // runner. Preserve chunk boundaries, sample collection and multipliers.
      const auto limits = LosslessWorkerLimits::FromEnvironment();
      LosslessProbePool arena;
      JXL_RETURN_IF_ERROR(arena.Run(
          trees.size(), pool, limits.tasks,
          limits.Tasks(trees.size(), MaHistogramBudget::kLimit), process_chunk));
    } else if (parallel_split || parallel_gather) {
      // The single learner owns the pool for its property searches. Never
      // invoke it inside a pool task: JxlParallelRunner is not reentrant.
      JXL_RETURN_IF_ERROR(process_chunk(0, parallel_split ? pool : nullptr));
    } else {
      JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, useful_splits.size() - 1,
                                    ThreadPool::NoInit,
                                    [&](uint32_t chunk, size_t) -> Status {
                                      return process_chunk(chunk, nullptr);
                                    },
                                    "LearnTrees"));
    }
    tree_.clear();
    JXL_RETURN_IF_ERROR(
        MergeTrees(trees, useful_splits, 0, useful_splits.size() - 1, &tree_));
  } else {
    // Fixed tree.
    size_t total_pixels = 0;
    int max_bitdepth = 0;
    for (const Image& img : stream_images_) {
      max_bitdepth = std::max(max_bitdepth, img.bitdepth);
      for (const Channel& ch : img.channel) {
        total_pixels += ch.w * ch.h;
      }
    }
    if (cparams_.speed_tier <= SpeedTier::kFalcon) {
      tree_ = PredefinedTree(ModularOptions::TreeKind::kWPFixedDC, total_pixels,
                             max_bitdepth, stream_options_[0].max_properties);
    } else if (cparams_.speed_tier <= SpeedTier::kThunder) {
      tree_ = PredefinedTree(ModularOptions::TreeKind::kGradientFixedDC,
                             total_pixels, max_bitdepth,
                             stream_options_[0].max_properties);
    } else {
      tree_ = {PropertyDecisionNode::Leaf(Predictor::Gradient)};
    }
  }
  tree_tokens_.resize(1);
  if (!ReplayingFinalPlan()) JXL_RETURN_IF_ERROR(FinalTree(&tree_));
  tree_tokens_[0].clear();
  Tree decoded_tree;
  JXL_RETURN_IF_ERROR(TokenizeTree(tree_, tree_tokens_.data(), &decoded_tree));
  JXL_ENSURE(tree_.size() == decoded_tree.size());
  tree_ = std::move(decoded_tree);
  if (auto* audit = cparams_.options.histogram_params.direct_audit) {
    audit->tree_nodes = tree_.size();
    const char* path = std::getenv("JXL_DIRECT_TREE_OUT");
    if (path && *path) JXL_RETURN_IF_ERROR(WriteNeuralTreeFile(path, tree_));
  }

  /* TODO(szabadka) Add text output callback to cparams
  if (kPrintTree && WantDebugOutput(aux_out)) {
    if (frame_header.dc_level > 0) {
      PrintTree(tree_, aux_out->debug_prefix + "/dc_frame_level" +
                           std::to_string(frame_header.dc_level) + "_tree");
    } else {
      PrintTree(tree_, aux_out->debug_prefix + "/global_tree");
    }
  } */
  return true;
}

Status ModularFrameEncoder::ComputeTokens(ThreadPool* pool) {
  LosslessStageTimer timer(cparams_, frame_dim_, "tokens");
  LosslessProfileScope profile(cparams_.options.histogram_params.lossless_profile,
                               "modular.tokens");
  size_t num_streams = stream_images_.size();
  stream_headers_.resize(num_streams);
  tokens_.resize(num_streams);
  image_widths_.resize(num_streams);
  const char* channel_env = std::getenv("JXL_LOSSLESS_TOKEN_PARALLEL");
  bool parallel_channels = pool && cparams_.IsLossless() && cparams_.modular_mode &&
      !CurrentFinalPlan() && channel_env && std::strcmp(channel_env, "1") == 0;
  if (parallel_channels) {
    size_t total = 0, largest = 0;
    for (size_t stream = 0; stream < num_streams; ++stream) {
      size_t count = 0;
      const auto& image = stream_images_[stream];
      for (size_t c = 0; c < image.channel.size(); ++c) {
        const auto& ch = image.channel[c];
        if (c >= image.nb_meta_channels &&
            (ch.w > stream_options_[stream].max_chan_size ||
             ch.h > stream_options_[stream].max_chan_size)) break;
        count += ch.w * ch.h;
      }
      total += count;
      largest = std::max(largest, count);
    }
    parallel_channels = largest >= 65536 && largest >= total - total / 4;
  }
  const auto process_stream = [&](const uint32_t stream_id,
                                  size_t /* thread */) -> Status {
    LosslessProfileScope stream_profile(profile.Id(), "modular.token_stream");
    stream_profile.Set("stream", stream_id);
    tokens_[stream_id].clear();
    JXL_RETURN_IF_ERROR(
        ModularCompress(stream_images_[stream_id], stream_options_[stream_id],
                        stream_id, tree_, stream_headers_[stream_id],
                        tokens_[stream_id], &image_widths_[stream_id],
                        parallel_channels ? pool : nullptr));
    return true;
  };
  if (parallel_channels) {
    for (size_t stream = 0; stream < num_streams; ++stream) {
      JXL_RETURN_IF_ERROR(process_stream(stream, 0));
    }
  } else {
    JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, num_streams, ThreadPool::NoInit,
                                  process_stream, "ComputeTokens"));
  }
  return true;
}

Status ModularFrameEncoder::EncodeGlobalInfo(bool streaming_mode,
                                             BitWriter* writer,
                                             AuxOut* aux_out, ThreadPool* pool) {
  LosslessStageTimer timer(cparams_, frame_dim_, "entropy");
  LosslessProfileScope profile(cparams_.options.histogram_params.lossless_profile,
                               "modular.global_entropy");
  JxlMemoryManager* memory_manager = writer->memory_manager();
  bool skip_rest = false;
  JXL_RETURN_IF_ERROR(
      writer->WithMaxBits(1, LayerType::ModularTree, aux_out, [&] {
        // If we are using brotli, or not using modular mode.
        if (tree_tokens_.empty() || tree_tokens_[0].empty()) {
          writer->Write(1, 0);
          skip_rest = true;
        } else {
          writer->Write(1, 1);
        }
        return true;
      }));
  if (skip_rest) return true;

  // Write tree
  HistogramParams params =
      HistogramParams::ForModular(cparams_, extra_dc_precision, streaming_mode);
  {
    EntropyEncodingData tree_code;
    JXL_ASSIGN_OR_RETURN(
        size_t cost, BuildAndEncodeHistograms(
                         memory_manager, params, kNumTreeContexts, tree_tokens_,
                         &tree_code, writer, LayerType::ModularTree, aux_out));
    (void)cost;
    JXL_RETURN_IF_ERROR(WriteTokens(tree_tokens_[0], tree_code, 0, writer,
                                    LayerType::ModularTree, aux_out));
  }
  params.streaming_mode = streaming_mode;
  params.add_missing_symbols = streaming_mode;
  params.image_widths = image_widths_;
  params.lossless_bucket_matcher = cparams_.IsLossless() &&
      cparams_.lossless_bucket_matcher && !CurrentFinalPlan();
  const char* parallel_lz77 = std::getenv("JXL_LOSSLESS_LZ77_PARALLEL");
  if (!streaming_mode && cparams_.modular_mode && cparams_.IsLossless() &&
      !CurrentFinalPlan() && parallel_lz77 &&
      std::strcmp(parallel_lz77, "1") == 0) {
    params.lossless_lz77_pool = cparams_.lossless_probe_runner ? nullptr : pool;
  }
  // Write histograms.
  JXL_ASSIGN_OR_RETURN(
      size_t cost, BuildAndEncodeHistograms(
                       memory_manager, params, (tree_.size() + 1) / 2, tokens_,
                       &code_, writer, LayerType::ModularGlobal, aux_out));
  (void)cost;
  return true;
}

Status ModularFrameEncoder::EncodeStream(BitWriter* writer, AuxOut* aux_out,
                                         LayerType layer,
                                         const ModularStreamId& stream) {
  size_t stream_id = stream.ID(frame_dim_);
  LosslessProfileScope profile(cparams_.options.histogram_params.lossless_profile,
                               "modular.stream");
  profile.Set("stream", stream_id);
  if (stream_images_[stream_id].channel.empty()) {
    JXL_DEBUG_V(10, "Modular stream %" PRIuS " is empty.", stream_id);
    return true;  // Image with no channels, header never gets decoded.
  }
  if (tokens_.empty()) {
    JXL_RETURN_IF_ERROR(ModularGenericCompress(
        stream_images_[stream_id], stream_options_[stream_id], *writer, aux_out,
        layer, stream_id));
  } else {
    JXL_RETURN_IF_ERROR(
        Bundle::Write(stream_headers_[stream_id], writer, layer, aux_out));
    JXL_RETURN_IF_ERROR(
        WriteTokens(tokens_[stream_id], code_, 0, writer, layer, aux_out));
  }
  return true;
}

void ModularFrameEncoder::ClearStreamData(const ModularStreamId& stream) {
  size_t stream_id = stream.ID(frame_dim_);
  Image empty_image(stream_images_[stream_id].memory_manager());
  std::swap(stream_images_[stream_id], empty_image);
}

void ModularFrameEncoder::ClearModularStreamData() {
  for (const auto& group : stream_params_) {
    ClearStreamData(group.id);
  }
  stream_params_.clear();
}

size_t ModularFrameEncoder::ComputeStreamingAbsoluteAcGroupId(
    size_t dc_group_id, size_t ac_group_id,
    const FrameDimensions& patch_dim) const {
  size_t dc_group_x = dc_group_id % frame_dim_.xsize_dc_groups;
  size_t dc_group_y = dc_group_id / frame_dim_.xsize_dc_groups;
  size_t ac_group_x = ac_group_id % patch_dim.xsize_groups;
  size_t ac_group_y = ac_group_id / patch_dim.xsize_groups;
  return (dc_group_x * 8 + ac_group_x) +
         (dc_group_y * 8 + ac_group_y) * frame_dim_.xsize_groups;
}

Status ModularFrameEncoder::PrepareStreamParams(const Rect& rect,
                                                const CompressParams& cparams,
                                                int minShift, int maxShift,
                                                const ModularStreamId& stream,
                                                bool do_color, bool groupwise,
                                                ThreadPool* wp_pool,
                                                size_t* deferred_wp_modes) {
  size_t stream_id = stream.ID(frame_dim_);
  if (stream_id == 0 && frame_dim_.num_groups != 1) {
    // If we have multiple groups, then the stream with ID 0 holds the full
    // image and we do not want to apply transforms or in general change the
    // pixel values.
    return true;
  }
  Image& full_image = stream_images_[0];
  JxlMemoryManager* memory_manager = full_image.memory_manager();
  const size_t xsize = rect.xsize();
  const size_t ysize = rect.ysize();
  Image& gi = stream_images_[stream_id];
  const size_t transform_start = stream_id == 0 ? gi.transform.size() : 0;
  if (stream_id > 0) {
    JXL_ASSIGN_OR_RETURN(gi, Image::Create(memory_manager, xsize, ysize,
                                           full_image.bitdepth, 0));
    // start at the first bigger-than-frame_dim.group_dim non-metachannel
    size_t c = full_image.nb_meta_channels;
    if (!groupwise) {
      for (; c < full_image.channel.size(); c++) {
        Channel& fc = full_image.channel[c];
        if (fc.w > frame_dim_.group_dim || fc.h > frame_dim_.group_dim) break;
      }
    }
    for (; c < full_image.channel.size(); c++) {
      Channel& fc = full_image.channel[c];
      int shift = std::min(fc.hshift, fc.vshift);
      if (shift > maxShift) continue;
      if (shift < minShift) continue;
      Rect r(rect.x0() >> fc.hshift, rect.y0() >> fc.vshift,
             rect.xsize() >> fc.hshift, rect.ysize() >> fc.vshift, fc.w, fc.h);
      if (r.xsize() == 0 || r.ysize() == 0) continue;
      gi_channel_[stream_id].push_back(c);
      JXL_ASSIGN_OR_RETURN(
          Channel gc, Channel::Create(memory_manager, r.xsize(), r.ysize()));
      gc.hshift = fc.hshift;
      gc.vshift = fc.vshift;
      for (size_t y = 0; y < r.ysize(); ++y) {
        memcpy(gc.Row(y), r.ConstRow(fc.plane, y),
               r.xsize() * sizeof(pixel_type));
      }
      gi.channel.emplace_back(std::move(gc));
    }

    if (gi.channel.empty()) return true;
    // Do some per-group transforms

    // Local palette transforms
    // TODO(veluca): make this work with quantize-after-prediction in lossy
    // mode.
    if (!ReplayingFinalPlan() && cparams_.ModularPartIsLossless() && !cparams.responsive &&
        !cparams.lossy_palette && cparams.speed_tier < SpeedTier::kCheetah) {
      int max_bitdepth = 0, maxval = 0;  // don't care about that here
      float channel_color_percent = 0;
        channel_color_percent = cparams.channel_colors_percent;
      JXL_RETURN_IF_ERROR(try_palettes(gi, max_bitdepth, maxval, cparams,
                                       channel_color_percent));
    }
  }

  // lossless and no specific color transform specified: try Nothing, YCoCg,
  // and 17 RCTs
  if (!ReplayingFinalPlan() && cparams.color_transform == ColorTransform::kNone &&
      cparams.IsLossless() && cparams.colorspace < 0 &&
      gi.channel.size() - gi.nb_meta_channels >= 3 &&
      cparams.responsive == JXL_FALSE && do_color &&
      cparams.speed_tier <= SpeedTier::kHare) {
    if (auto* audit = cparams.options.histogram_params.direct_audit) {
      ++audit->forbidden_searches;
      return JXL_FAILURE("Direct plan reached RCT search");
    }
    size_t nb_rcts_to_try = 0;
    switch (cparams.speed_tier) {
      case SpeedTier::kLightning:
      case SpeedTier::kThunder:
      case SpeedTier::kFalcon:
      case SpeedTier::kCheetah:
        nb_rcts_to_try = 0;  // Just do global YCoCg
        break;
      case SpeedTier::kHare:
        nb_rcts_to_try = 4;
        break;
      case SpeedTier::kWombat:
        nb_rcts_to_try = 5;
        break;
      case SpeedTier::kSquirrel:
        nb_rcts_to_try = 7;
        break;
      case SpeedTier::kKitten:
        nb_rcts_to_try = 9;
        break;
      case SpeedTier::kTectonicPlate:
      case SpeedTier::kGlacier:
      case SpeedTier::kTortoise:
        nb_rcts_to_try = 19;
        break;
    }
    float best_cost = std::numeric_limits<float>::max();
    size_t best_rct = 0;
    bool need_to_restore = (nb_rcts_to_try > 1);
    std::vector<Channel> orig;
    orig.reserve(3);
    // These should be 19 actually different transforms; the remaining ones
    // are equivalent to one of these (note that the first two are do-nothing
    // and YCoCg) modulo channel reordering (which only matters in the case of
    // MA-with-prev-channels-properties) and/or sign (e.g. RmG vs GmR)
    for (int rct_type : {0 * 7 + 0, 0 * 7 + 6, 0 * 7 + 5, 1 * 7 + 3, 3 * 7 + 5,
                         5 * 7 + 5, 1 * 7 + 5, 2 * 7 + 5, 1 * 7 + 1, 0 * 7 + 4,
                         1 * 7 + 2, 2 * 7 + 1, 2 * 7 + 2, 2 * 7 + 3, 4 * 7 + 4,
                         4 * 7 + 5, 0 * 7 + 2, 0 * 7 + 1, 0 * 7 + 3}) {
      if (nb_rcts_to_try == 0) break;
      nb_rcts_to_try--;
      // no-op rct_type; use as baseline cost
      if (rct_type == 0) {
        JXL_ASSIGN_OR_RETURN(best_cost, EstimateCost(gi));
        for (size_t c = 0; c < 3; ++c) {
          Channel& genuine = gi.channel[gi.nb_meta_channels + c];
          JXL_ASSIGN_OR_RETURN(
              Channel ch,
              Channel::Create(genuine.memory_manager(), genuine.w, genuine.h,
                              genuine.hshift, genuine.vshift));
          orig.emplace_back(std::move(ch));
          genuine.plane.Swap(orig[c].plane);
        }
      } else {
        std::array<const Channel*, 3> in = {&orig[0], &orig[1], &orig[2]};
        std::array<Channel*, 3> out = {&gi.channel[gi.nb_meta_channels + 0],
                                       &gi.channel[gi.nb_meta_channels + 1],
                                       &gi.channel[gi.nb_meta_channels + 2]};
        JXL_RETURN_IF_ERROR(FwdRct(in, out, rct_type, /* pool */ nullptr));
        JXL_ASSIGN_OR_RETURN(float cost, EstimateCost(gi));
        if (cost < best_cost) {
          best_rct = rct_type;
          best_cost = cost;
        }
      }
    }
    if (need_to_restore) {
      for (size_t c = 0; c < 3; ++c) {
        gi.channel[gi.nb_meta_channels + c].plane.Swap(orig[c].plane);
      }
    }
    // Apply the best RCT to the image for future encoding.
    if (best_rct != 0) {
      Transform sg(TransformId::kRCT);
      sg.begin_c = gi.nb_meta_channels;
      sg.rct_type = best_rct;
      do_transform(gi, sg, weighted::Header());
    }
  } else {
    // No need to try anything, just use the default options.
  }
  JXL_RETURN_IF_ERROR(FinalTransforms(gi, transform_start));
  size_t nb_wp_modes = 1;
  if (cparams.direct_plan >= 0) {
    stream_options_[stream_id].wp_mode = cparams.options.wp_mode;
  } else if (cparams.speed_tier <= SpeedTier::kTortoise) {
    nb_wp_modes = 5;
  } else if (cparams.speed_tier <= SpeedTier::kKitten) {
    nb_wp_modes = 2;
  }
  if (!ReplayingFinalPlan() && nb_wp_modes > 1 &&
      PredictorHasWeighted(stream_options_[stream_id].predictor)) {
    if (auto* audit = cparams.options.histogram_params.direct_audit) {
      ++audit->forbidden_searches;
      return JXL_FAILURE("Direct plan reached WP search");
    }
    if (deferred_wp_modes) {
      *deferred_wp_modes = nb_wp_modes;
      return true;
    }
    LosslessProfileScope wp_profile(
        cparams.options.histogram_params.lossless_profile, "modular.wp_search");
    wp_profile.Set("modes", nb_wp_modes);
    wp_profile.Set("stream", stream_id);
    float best_cost = std::numeric_limits<float>::max();
    stream_options_[stream_id].wp_mode = 0;
    const char* bound_value = std::getenv("JXL_LOSSLESS_WP_BOUND");
    const bool bound_wp = cparams.options.lossless_tree_speedups &&
        bound_value && std::strcmp(bound_value, "1") == 0;
    WPCostWork work;
    size_t pixels = 0;
    for (const auto& ch : gi.channel) pixels += ch.w * ch.h;
    const bool parallel_modes = wp_pool && pixels >= 65536;
    std::array<float, 5> costs;
    std::array<WPCostWork, 5> works;
    if (parallel_modes) {
      const auto evaluate = [&](uint32_t mode, size_t) -> Status {
        costs[mode] = EstimateWPCost(
            gi, mode, std::numeric_limits<float>::infinity(), &works[mode],
            cparams.ModularPartIsLossless());
        return true;
      };
      JXL_RETURN_IF_ERROR(RunOnPool(wp_pool, 0, nb_wp_modes, ThreadPool::NoInit,
                                    evaluate, "LosslessWPModes"));
      for (size_t i = 0; i < nb_wp_modes; ++i) {
        work.pixels += works[i].pixels;
        work.pruned_modes += works[i].pruned_modes;
        work.checkpoints += works[i].checkpoints;
      }
    }
    for (size_t i = 0; i < nb_wp_modes; i++) {
      float cost = parallel_modes ? costs[i] : EstimateWPCost(
          gi, i, bound_wp ? best_cost : std::numeric_limits<float>::infinity(), &work,
          cparams.ModularPartIsLossless());
      if (cost < best_cost) {
        best_cost = cost;
        stream_options_[stream_id].wp_mode = i;
      }
    }
    wp_profile.Set("bounded", bound_wp && !parallel_modes);
    wp_profile.Set("parallel_modes", parallel_modes ? nb_wp_modes : 1);
    wp_profile.Set("scanned_pixels", work.pixels);
    wp_profile.Set("saved_pixels", pixels * nb_wp_modes - work.pixels);
    wp_profile.Set("pruned_modes", work.pruned_modes);
    wp_profile.Set("checkpoints", work.checkpoints);
  }
  if (CurrentFinalPlan()) {
    std::vector<int64_t> mode{static_cast<int64_t>(stream_id), stream_options_[stream_id].wp_mode};
    JXL_RETURN_IF_ERROR(FinalPlanExchange("wp", &mode));
    JXL_ENSURE(mode.size() == 2 && mode[0] == stream_id && mode[1] >= 0 && mode[1] <= 4);
    stream_options_[stream_id].wp_mode = mode[1];
  }
  return true;
}

Status ModularFrameEncoder::PrepareWPModes(const std::vector<size_t>& modes,
                                          ThreadPool* pool) {
  LosslessProfileScope profile(cparams_.options.histogram_params.lossless_profile,
                               "modular.wp_wave");
  const char* wp_parallel_env = std::getenv("JXL_LOSSLESS_WP_PARALLEL");
  const bool split_channels = wp_parallel_env && std::strcmp(wp_parallel_env, "3") == 0;
  struct Job { size_t stream, mode, pixels, channel; float cost = 0; };
  std::vector<Job> jobs;
  std::vector<std::array<float, 5>> costs(modes.size());
  size_t max_width = 0;
  for (size_t stream = 0; stream < modes.size(); ++stream) {
    if (!modes[stream]) continue;
    size_t pixels = 0;
    for (const auto& ch : stream_images_[stream].channel) {
      pixels += ch.w * ch.h;
      max_width = std::max(max_width, ch.w);
    }
    for (size_t mode = 0; mode < modes[stream]; ++mode) {
      if (split_channels) {
        for (size_t c = 0; c < stream_images_[stream].channel.size(); ++c) {
          const auto& ch = stream_images_[stream].channel[c];
          jobs.push_back({stream, mode, ch.w * ch.h, c});
        }
      } else {
        jobs.push_back({stream, mode, pixels, SIZE_MAX});
      }
    }
  }
  if (jobs.empty()) return true;
  std::stable_sort(jobs.begin(), jobs.end(), [](const auto& a, const auto& b) {
    return a.pixels > b.pixels;
  });
  const size_t tasks = LosslessWorkerLimits::FromEnvironment().Tasks(
      jobs.size(), (1u << 20) + uint64_t{max_width} * 256);
  std::atomic<size_t> next{0};
  const auto evaluate = [&](uint32_t, size_t) -> Status {
    for (;;) {
      const size_t i = next.fetch_add(1, std::memory_order_relaxed);
      if (i >= jobs.size()) return true;
      auto& job = jobs[i];
      LosslessProfileScope mode(profile.Id(), "modular.wp_mode");
      WPCostWork work;
      job.cost = EstimateWPCost(stream_images_[job.stream],
          job.mode, std::numeric_limits<float>::infinity(), &work, true, job.channel);
      mode.Set("stream", job.stream);
      mode.Set("mode", job.mode);
      mode.Set("channel", job.channel == SIZE_MAX ? 0 : job.channel + 1);
      mode.Set("scanned_pixels", work.pixels);
    }
  };
  profile.Set("jobs", jobs.size());
  profile.Set("worker_tasks", tasks);
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, tasks, ThreadPool::NoInit, evaluate,
                                "LosslessWPWave"));
  // Channels start with independent WP state. Reduce after joining, never
  // atomically add floating costs in worker-completion order.
  std::sort(jobs.begin(), jobs.end(), [](const Job& a, const Job& b) {
    return std::tie(a.stream, a.mode, a.channel) < std::tie(b.stream, b.mode, b.channel);
  });
  for (const auto& job : jobs) costs[job.stream][job.mode] += job.cost;
  // All group/mode evaluations are independent; retain the original tie order.
  for (size_t stream = 0; stream < modes.size(); ++stream) {
    if (!modes[stream]) continue;
    size_t best = 0;
    for (size_t mode = 1; mode < modes[stream]; ++mode) {
      if (costs[stream][mode] < costs[stream][best]) best = mode;
    }
    stream_options_[stream].wp_mode = best;
  }
  return true;
}

constexpr float q_deadzone = 0.62f;
int QuantizeWP(const int32_t* qrow, size_t onerow, size_t c, size_t x, size_t y,
               size_t w, weighted::State* wp_state, float value,
               float inv_factor, bool* has_outliers) {
  float svalue = value * inv_factor;
  PredictionResult pred =
      PredictNoTreeWP(w, qrow + x, onerow, x, y, Predictor::Weighted, wp_state);
  svalue -= pred.guess;
  if (svalue > -q_deadzone && svalue < q_deadzone) svalue = 0;
  int residual = 0;
  if (std::isnan(svalue) ||
      svalue > static_cast<float>(std::numeric_limits<int>::max()) ||
      svalue < static_cast<float>(std::numeric_limits<int>::min())) {
    *has_outliers = true;
  } else {
    residual = std::round(svalue);
  }
  if (residual > 2 || residual < -2) residual = std::round(svalue * 0.5f) * 2;
  return residual + pred.guess;
}

int QuantizeGradient(const int32_t* qrow, size_t onerow, size_t c, size_t x,
                     size_t y, size_t w, float value, float inv_factor) {
  float svalue = value * inv_factor;
  PredictionResult pred =
      PredictNoTreeNoWP(w, qrow + x, onerow, x, y, Predictor::Gradient);
  svalue -= pred.guess;
  if (svalue > -q_deadzone && svalue < q_deadzone) svalue = 0;
  int residual = std::round(svalue);
  if (residual > 2 || residual < -2) residual = std::round(svalue * 0.5f) * 2;
  return residual + pred.guess;
}

Status ModularFrameEncoder::AddVarDCTDC(const FrameHeader& frame_header,
                                        const Image3F& dc, const Rect& r,
                                        size_t group_index, bool nl_dc,
                                        PassesEncoderState* enc_state,
                                        bool jpeg_transcode) {
  JxlMemoryManager* memory_manager = dc.memory_manager();
  extra_dc_precision[group_index] = nl_dc ? 1 : 0;
  float mul = 1 << extra_dc_precision[group_index];
  bool has_outliers = false;

  size_t stream_id = ModularStreamId::VarDCTDC(group_index).ID(frame_dim_);
  stream_options_[stream_id].max_chan_size = 0xFFFFFF;
  stream_options_[stream_id].predictor = Predictor::Weighted;
  stream_options_[stream_id].wp_tree_mode = ModularOptions::TreeMode::kWPOnly;
  if (cparams_.speed_tier >= SpeedTier::kSquirrel) {
    stream_options_[stream_id].tree_kind = ModularOptions::TreeKind::kWPFixedDC;
  }
  if (cparams_.speed_tier < SpeedTier::kSquirrel && !nl_dc) {
    stream_options_[stream_id].predictor =
        (cparams_.speed_tier < SpeedTier::kKitten ? Predictor::Variable
                                                  : Predictor::Best);
    stream_options_[stream_id].wp_tree_mode =
        ModularOptions::TreeMode::kDefault;
    stream_options_[stream_id].tree_kind = ModularOptions::TreeKind::kLearn;
  }
  if (cparams_.decoding_speed_tier >= 1) {
    stream_options_[stream_id].tree_kind =
        ModularOptions::TreeKind::kGradientFixedDC;
  }
  stream_options_[stream_id].histogram_params =
      stream_options_[0].histogram_params;

  JXL_ASSIGN_OR_RETURN(
      stream_images_[stream_id],
      Image::Create(memory_manager, r.xsize(), r.ysize(), 8, 3));
  const ColorCorrelation& color_correlation = enc_state->shared.cmap.base();
  if (nl_dc && stream_options_[stream_id].tree_kind ==
                   ModularOptions::TreeKind::kGradientFixedDC) {
    JXL_ENSURE(frame_header.chroma_subsampling.Is444());
    for (size_t c : {1, 0, 2}) {
      float inv_factor = enc_state->shared.quantizer.GetInvDcStep(c) * mul;
      float y_factor = enc_state->shared.quantizer.GetDcStep(1) / mul;
      float cfl_factor = color_correlation.DCFactors()[c];
      for (size_t y = 0; y < r.ysize(); y++) {
        int32_t* quant_row =
            stream_images_[stream_id].channel[c < 2 ? c ^ 1 : c].plane.Row(y);
        size_t stride = stream_images_[stream_id]
                            .channel[c < 2 ? c ^ 1 : c]
                            .plane.PixelsPerRow();
        const float* row = r.ConstPlaneRow(dc, c, y);
        if (c == 1) {
          for (size_t x = 0; x < r.xsize(); x++) {
            quant_row[x] = QuantizeGradient(quant_row, stride, c, x, y,
                                            r.xsize(), row[x], inv_factor);
          }
        } else {
          int32_t* quant_row_y =
              stream_images_[stream_id].channel[0].plane.Row(y);
          for (size_t x = 0; x < r.xsize(); x++) {
            quant_row[x] = QuantizeGradient(
                quant_row, stride, c, x, y, r.xsize(),
                row[x] - quant_row_y[x] * (y_factor * cfl_factor), inv_factor);
          }
        }
      }
    }
  } else if (nl_dc) {
    JXL_ENSURE(frame_header.chroma_subsampling.Is444());
    for (size_t c : {1, 0, 2}) {
      float inv_factor = enc_state->shared.quantizer.GetInvDcStep(c) * mul;
      float y_factor = enc_state->shared.quantizer.GetDcStep(1) / mul;
      float cfl_factor = color_correlation.DCFactors()[c];
      weighted::Header header;
      weighted::State wp_state(header, r.xsize(), r.ysize());
      for (size_t y = 0; y < r.ysize(); y++) {
        int32_t* quant_row =
            stream_images_[stream_id].channel[c < 2 ? c ^ 1 : c].plane.Row(y);
        size_t stride = stream_images_[stream_id]
                            .channel[c < 2 ? c ^ 1 : c]
                            .plane.PixelsPerRow();
        const float* row = r.ConstPlaneRow(dc, c, y);
        if (c == 1) {
          for (size_t x = 0; x < r.xsize(); x++) {
            quant_row[x] =
                QuantizeWP(quant_row, stride, c, x, y, r.xsize(), &wp_state,
                           row[x], inv_factor, &has_outliers);
            wp_state.UpdateErrors(quant_row[x], x, y, r.xsize());
          }
        } else {
          int32_t* quant_row_y =
              stream_images_[stream_id].channel[0].plane.Row(y);
          for (size_t x = 0; x < r.xsize(); x++) {
            quant_row[x] =
                QuantizeWP(quant_row, stride, c, x, y, r.xsize(), &wp_state,
                           row[x] - quant_row_y[x] * (y_factor * cfl_factor),
                           inv_factor, &has_outliers);
            wp_state.UpdateErrors(quant_row[x], x, y, r.xsize());
          }
        }
      }
    }
  } else if (frame_header.chroma_subsampling.Is444()) {
    for (size_t c : {1, 0, 2}) {
      float inv_factor = enc_state->shared.quantizer.GetInvDcStep(c) * mul;
      float y_factor = enc_state->shared.quantizer.GetDcStep(1) / mul;
      float cfl_factor = color_correlation.DCFactors()[c];
      for (size_t y = 0; y < r.ysize(); y++) {
        int32_t* quant_row =
            stream_images_[stream_id].channel[c < 2 ? c ^ 1 : c].plane.Row(y);
        const float* row = r.ConstPlaneRow(dc, c, y);
        if (c == 1) {
          for (size_t x = 0; x < r.xsize(); x++) {
            quant_row[x] = std::round(row[x] * inv_factor);
          }
        } else {
          int32_t* quant_row_y =
              stream_images_[stream_id].channel[0].plane.Row(y);
          for (size_t x = 0; x < r.xsize(); x++) {
            quant_row[x] =
                std::round((row[x] - quant_row_y[x] * (y_factor * cfl_factor)) *
                           inv_factor);
          }
        }
      }
    }
  } else {
    for (size_t c : {1, 0, 2}) {
      Rect rect(r.x0() >> frame_header.chroma_subsampling.HShift(c),
                r.y0() >> frame_header.chroma_subsampling.VShift(c),
                r.xsize() >> frame_header.chroma_subsampling.HShift(c),
                r.ysize() >> frame_header.chroma_subsampling.VShift(c));
      float inv_factor = enc_state->shared.quantizer.GetInvDcStep(c) * mul;
      size_t ys = rect.ysize();
      size_t xs = rect.xsize();
      Channel& ch = stream_images_[stream_id].channel[c < 2 ? c ^ 1 : c];
      ch.w = xs;
      ch.h = ys;
      JXL_RETURN_IF_ERROR(ch.shrink());
      for (size_t y = 0; y < ys; y++) {
        int32_t* quant_row = ch.plane.Row(y);
        const float* row = rect.ConstPlaneRow(dc, c, y);
        for (size_t x = 0; x < xs; x++) {
          quant_row[x] = std::round(row[x] * inv_factor);
        }
      }
    }
  }

  if (has_outliers) {
    return JXL_FAILURE("Unsupported range of DC values");
  }

  DequantDC(r, &enc_state->shared.dc_storage, &enc_state->shared.quant_dc,
            stream_images_[stream_id], enc_state->shared.quantizer.MulDC(),
            1.0 / mul, color_correlation.DCFactors(),
            frame_header.chroma_subsampling, enc_state->shared.block_ctx_map);
  return true;
}

Status ModularFrameEncoder::AddACMetadata(const Rect& r, size_t group_index,
                                          bool jpeg_transcode,
                                          PassesEncoderState* enc_state) {
  JxlMemoryManager* memory_manager = enc_state->memory_manager();
  size_t stream_id = ModularStreamId::ACMetadata(group_index).ID(frame_dim_);
  stream_options_[stream_id].max_chan_size = 0xFFFFFF;
  if (stream_options_[stream_id].predictor != Predictor::Weighted) {
    stream_options_[stream_id].wp_tree_mode = ModularOptions::TreeMode::kNoWP;
  }
  if (jpeg_transcode) {
    stream_options_[stream_id].tree_kind =
        ModularOptions::TreeKind::kJpegTranscodeACMeta;
  } else if (cparams_.speed_tier >= SpeedTier::kFalcon) {
    stream_options_[stream_id].tree_kind =
        ModularOptions::TreeKind::kFalconACMeta;
  } else if (cparams_.speed_tier > SpeedTier::kKitten) {
    stream_options_[stream_id].tree_kind = ModularOptions::TreeKind::kACMeta;
  }
  // If we are using a non-constant CfL field, and are in a slow enough mode,
  // re-enable tree computation for it.
  if (cparams_.speed_tier < SpeedTier::kSquirrel &&
      cparams_.force_cfl_jpeg_recompression) {
    stream_options_[stream_id].tree_kind = ModularOptions::TreeKind::kLearn;
  }
  stream_options_[stream_id].histogram_params =
      stream_options_[0].histogram_params;
  // YToX, YToB, ACS + QF, EPF
  Image& image = stream_images_[stream_id];
  JXL_ASSIGN_OR_RETURN(
      image, Image::Create(memory_manager, r.xsize(), r.ysize(), 8, 4));
  static_assert(kColorTileDimInBlocks == 8, "Color tile size changed");
  Rect cr(r.x0() >> 3, r.y0() >> 3, (r.xsize() + 7) >> 3, (r.ysize() + 7) >> 3);
  JXL_ASSIGN_OR_RETURN(
      image.channel[0],
      Channel::Create(memory_manager, cr.xsize(), cr.ysize(), 3, 3));
  JXL_ASSIGN_OR_RETURN(
      image.channel[1],
      Channel::Create(memory_manager, cr.xsize(), cr.ysize(), 3, 3));
  JXL_ASSIGN_OR_RETURN(
      image.channel[2],
      Channel::Create(memory_manager, r.xsize() * r.ysize(), 2, 0, 0));
  JXL_RETURN_IF_ERROR(ConvertPlaneAndClamp(cr, enc_state->shared.cmap.ytox_map,
                                           Rect(image.channel[0].plane),
                                           &image.channel[0].plane));
  JXL_RETURN_IF_ERROR(ConvertPlaneAndClamp(cr, enc_state->shared.cmap.ytob_map,
                                           Rect(image.channel[1].plane),
                                           &image.channel[1].plane));
  size_t num = 0;
  for (size_t y = 0; y < r.ysize(); y++) {
    AcStrategyRow row_acs = enc_state->shared.ac_strategy.ConstRow(r, y);
    const int32_t* row_qf = r.ConstRow(enc_state->shared.raw_quant_field, y);
    const uint8_t* row_epf = r.ConstRow(enc_state->shared.epf_sharpness, y);
    int32_t* out_acs = image.channel[2].plane.Row(0);
    int32_t* out_qf = image.channel[2].plane.Row(1);
    int32_t* row_out_epf = image.channel[3].plane.Row(y);
    for (size_t x = 0; x < r.xsize(); x++) {
      row_out_epf[x] = row_epf[x];
      if (!row_acs[x].IsFirstBlock()) continue;
      out_acs[num] = row_acs[x].RawStrategy();
      out_qf[num] = row_qf[x] - 1;
      num++;
    }
  }
  image.channel[2].w = num;
  ac_metadata_size[group_index] = num;
  return true;
}

Status ModularFrameEncoder::EncodeQuantTable(
    JxlMemoryManager* memory_manager, size_t size_x, size_t size_y,
    BitWriter* writer, const QuantEncoding& encoding, size_t idx,
    ModularFrameEncoder* modular_frame_encoder) {
  JXL_ENSURE(encoding.qraw.qtable);
  JXL_ENSURE(size_x * size_y * 3 == encoding.qraw.qtable->size());
  JXL_ENSURE(idx < kNumQuantTables);
  int* qtable = encoding.qraw.qtable->data();
  JXL_RETURN_IF_ERROR(F16Coder::Write(encoding.qraw.qtable_den, writer));
  if (modular_frame_encoder) {
    JXL_ASSIGN_OR_RETURN(ModularStreamId qt, ModularStreamId::QuantTable(idx));
    JXL_RETURN_IF_ERROR(modular_frame_encoder->EncodeStream(
        writer, nullptr, LayerType::Header, qt));
    return true;
  }
  JXL_ASSIGN_OR_RETURN(Image image,
                       Image::Create(memory_manager, size_x, size_y, 8, 3));
  for (size_t c = 0; c < 3; c++) {
    for (size_t y = 0; y < size_y; y++) {
      int32_t* JXL_RESTRICT row = image.channel[c].Row(y);
      for (size_t x = 0; x < size_x; x++) {
        row[x] = qtable[c * size_x * size_y + y * size_x + x];
      }
    }
  }
  ModularOptions cfopts;
  JXL_RETURN_IF_ERROR(ModularGenericCompress(image, cfopts, *writer));
  return true;
}

Status ModularFrameEncoder::AddQuantTable(size_t size_x, size_t size_y,
                                          const QuantEncoding& encoding,
                                          size_t idx) {
  JXL_ENSURE(idx < kNumQuantTables);
  JXL_ASSIGN_OR_RETURN(ModularStreamId qt, ModularStreamId::QuantTable(idx));
  size_t stream_id = qt.ID(frame_dim_);
  JXL_ENSURE(encoding.qraw.qtable);
  JXL_ENSURE(size_x * size_y * 3 == encoding.qraw.qtable->size());
  int* qtable = encoding.qraw.qtable->data();
  Image& image = stream_images_[stream_id];
  JxlMemoryManager* memory_manager = image.memory_manager();
  JXL_ASSIGN_OR_RETURN(image,
                       Image::Create(memory_manager, size_x, size_y, 8, 3));
  for (size_t c = 0; c < 3; c++) {
    for (size_t y = 0; y < size_y; y++) {
      int32_t* JXL_RESTRICT row = image.channel[c].Row(y);
      for (size_t x = 0; x < size_x; x++) {
        row[x] = qtable[c * size_x * size_y + y * size_x + x];
      }
    }
  }
  return true;
}
}  // namespace jxl
