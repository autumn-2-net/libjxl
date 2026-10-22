// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/memory_manager.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <locale>
#include <queue>
#include <utility>
#include <vector>

#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_ans_params.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/enc_fields.h"
#include "lib/jxl/enc_lossless_profile.h"
#include "lib/jxl/enc_lossless_workers.h"
#include "lib/jxl/enc_ma_search.h"
#include "lib/jxl/enc_ma_tree_groups.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/modular/encoding/context_predict.h"
#include "lib/jxl/modular/encoding/dec_ma.h"
#include "lib/jxl/modular/encoding/enc_ma.h"
#include "lib/jxl/modular/encoding/enc_predict.h"
#include "lib/jxl/modular/encoding/encoding.h"
#include "lib/jxl/modular/encoding/ma_common.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/options.h"
#include "lib/jxl/pack_signed.h"

namespace jxl {

namespace {
// `cutoffs` must be sorted.
Tree MakeFixedTree(int property, const std::vector<int32_t> &cutoffs,
                   Predictor pred, size_t num_pixels, int bitdepth) {
  size_t log_px = CeilLog2Nonzero(num_pixels);
  size_t min_gap = 0;
  // Reduce fixed tree height when encoding small images.
  if (log_px < 14) {
    min_gap = 8 * (14 - log_px);
  }
  const int shift = bitdepth > 11 ? std::min(4, bitdepth - 11) : 0;
  const int mul = 1 << shift;
  Tree tree;
  struct NodeInfo {
    size_t begin, end, pos;
  };
  std::queue<NodeInfo> q;
  // Leaf IDs will be set by roundtrip decoding the tree.
  tree.push_back(PropertyDecisionNode::Leaf(pred));
  q.push(NodeInfo{0, cutoffs.size(), 0});
  while (!q.empty()) {
    NodeInfo info = q.front();
    q.pop();
    if (info.begin + min_gap >= info.end) continue;
    uint32_t split = (info.begin + info.end) / 2;
    int32_t cutoff = cutoffs[split] * mul;
    tree[info.pos] = PropertyDecisionNode::Split(property, cutoff, tree.size());
    q.push(NodeInfo{split + 1, info.end, tree.size()});
    tree.push_back(PropertyDecisionNode::Leaf(pred));
    q.push(NodeInfo{info.begin, split, tree.size()});
    tree.push_back(PropertyDecisionNode::Leaf(pred));
  }
  return tree;
}

double TreeSampleFraction(const Channel& channel, const ModularOptions& options) {
  double pixel_fraction = std::min(1.0f, options.nb_repeats);
  // a fraction of 0 is used to disable learning entirely.
  if (pixel_fraction > 0) {
    pixel_fraction = std::max(pixel_fraction,
                              std::min(1.0, 1024.0 / (channel.w * channel.h)));
  }
  return pixel_fraction;
}

template <bool kPredictorPair>
Status GatherTreeData(const Image &image, pixel_type chan, size_t group_id,
                      const weighted::Header &wp_header,
                      const ModularOptions &options, TreeSamples &tree_samples,
                      size_t *total_pixels, bool sparse_gather,
                      size_t row_begin = 0, size_t row_end = SIZE_MAX) {
  const Channel &channel = image.channel[chan];
  row_end = std::min(row_end, channel.h);
  JXL_ENSURE(row_begin <= row_end);
  if (sparse_gather && options.nb_repeats == 0) {
    *total_pixels += channel.w * (row_end - row_begin);
    return true;
  }
  JxlMemoryManager *memory_manager = channel.memory_manager();

  JXL_DEBUG_V(7, "Learning %" PRIuS "x%" PRIuS " channel %d", channel.w,
              channel.h, chan);

  std::array<pixel_type, kNumStaticProperties> static_props = {
      {chan, static_cast<int>(group_id)}};
  Properties properties(kNumNonrefProperties +
                        kExtraPropsPerChannel * options.max_properties);
  const double pixel_fraction = TreeSampleFraction(channel, options);
  const uint64_t threshold =
      (std::numeric_limits<uint64_t>::max() >> 32) * pixel_fraction;
  uint64_t s[2] = {static_cast<uint64_t>(0x94D049BB133111EBull),
                   static_cast<uint64_t>(0xBF58476D1CE4E5B9ull)};
  // Xorshift128+ adapted from xorshift128+-inl.h
  auto use_sample = [&]() {
    auto s1 = s[0];
    const auto s0 = s[1];
    const auto bits = s1 + s0;  // b, c
    s[0] = s0;
    s1 ^= s1 << 23;
    s1 ^= s0 ^ (s1 >> 18) ^ (s0 >> 5);
    s[1] = s1;
    return (bits >> 32) <= threshold;
  };
  const size_t warm_begin = row_begin > 16 ? row_begin - 16 : 0;
  // Preserve the original random sample locations. Only the learned weighted
  // predictor state is approximated by the warm-up, never actual pixel coding.
  for (size_t y = 0; y < warm_begin; ++y)
    for (size_t x = 0; x < channel.w; ++x) (void)use_sample();

  const ptrdiff_t onerow = channel.plane.PixelsPerRow();
  JXL_ASSIGN_OR_RETURN(
      Channel references,
      Channel::Create(memory_manager, properties.size() - kNumNonrefProperties,
                      channel.w));
  weighted::State wp_state(wp_header, channel.w, channel.h);
  const char* flat_env = std::getenv("JXL_LOSSLESS_WP_FLAT");
  const bool flat = options.lossless_tree_speedups && flat_env &&
      (std::strcmp(flat_env, "2") == 0 || std::strcmp(flat_env, "3") == 0);
  const bool sparse_references =
      sparse_gather && references.w && pixel_fraction <= 0.25;
  const SparseLearnReferences reference_channels(
      image, chan, sparse_references ? references.w : 0);
  tree_samples.PrepareForSamples(pixel_fraction * (row_end - row_begin) * channel.w + 64);
  const bool multiple_predictors = tree_samples.NumPredictors() != 1;
  auto compute_sample = [&](const pixel_type *p, size_t x, size_t y) {
    pixel_type_w pred[kNumModularPredictors];
    const bool sampled = use_sample() && y >= row_begin;
    if (sampled && sparse_references) {
      reference_channels.Fill(x, y, &references);
    }
    if (sparse_gather && !sampled) {
      AdvanceLearnState<false>(&properties, channel.w, p + x, onerow, x, y,
                               &wp_state, flat);
    } else if (kPredictorPair && sampled) {
      PredictLearnBest<false>(&properties, channel.w, p + x, onerow, x, y,
                               references, &wp_state, pred);
    } else if (multiple_predictors && (sampled || !options.lossless_tree_speedups)) {
      PredictLearnAll(&properties, channel.w, p + x, onerow, x, y, references,
                      &wp_state, pred);
    } else if (multiple_predictors) {
      // Keep WP and the previous-gradient property in exactly the same state,
      // but do not compute fourteen predictions for a discarded sample.
      PredictLearn(&properties, channel.w, p + x, onerow, x, y,
                   Predictor::Zero, references, &wp_state);
    } else {
      pred[static_cast<int>(tree_samples.PredictorFromIndex(0))] =
          PredictLearn(&properties, channel.w, p + x, onerow, x, y,
                       tree_samples.PredictorFromIndex(0), references,
                       &wp_state)
              .guess;
    }
    if (sampled) {
      tree_samples.AddSample(p[x], properties, pred);
    }
    wp_state.UpdateErrors(p[x], x, y, channel.w);
  };

  *total_pixels += channel.w * (row_end - row_begin);
  for (size_t y = warm_begin; y < row_end; y++) {
    const pixel_type *JXL_RESTRICT p = channel.Row(y);
    if (!sparse_references) {
      PrecomputeReferences(channel, y, image, chan, &references);
    }
    InitPropsRow(&properties, static_props, y);

    // TODO(veluca): avoid computing WP if we don't use its property or
    // predictions.
    if (y > 1 && channel.w > 8 && references.w == 0) {
      for (size_t x = 0; x < 2; x++) {
        compute_sample(p, x, y);
      }
      for (size_t x = 2; x < channel.w - 2; x++) {
        pixel_type_w pred[kNumModularPredictors];
        const bool sampled = use_sample() && y >= row_begin;
        if (sparse_gather && !sampled) {
          AdvanceLearnState<true>(&properties, channel.w, p + x, onerow, x, y,
                                  &wp_state, flat);
        } else if (kPredictorPair && sampled) {
          PredictLearnBest<true>(&properties, channel.w, p + x, onerow, x, y,
                                  references, &wp_state, pred);
        } else if (multiple_predictors && (sampled || !options.lossless_tree_speedups)) {
          PredictLearnAllNEC(&properties, channel.w, p + x, onerow, x, y,
                             references, &wp_state, pred);
        } else if (multiple_predictors) {
          PredictLearnNEC(&properties, channel.w, p + x, onerow, x, y,
                          Predictor::Zero, references, &wp_state);
        } else {
          pred[static_cast<int>(tree_samples.PredictorFromIndex(0))] =
              PredictLearnNEC(&properties, channel.w, p + x, onerow, x, y,
                              tree_samples.PredictorFromIndex(0), references,
                              &wp_state)
                  .guess;
        }
        if (sampled) {
          tree_samples.AddSample(p[x], properties, pred);
        }
        wp_state.UpdateErrors(p[x], x, y, channel.w);
      }
      for (size_t x = channel.w - 2; x < channel.w; x++) {
        compute_sample(p, x, y);
      }
    } else {
      for (size_t x = 0; x < channel.w; x++) {
        compute_sample(p, x, y);
      }
    }
  }
  tree_samples.FlushSampleBatch();
  return true;
}

StatusOr<Tree> LearnTree(
    TreeSamples &&tree_samples, size_t total_pixels,
    const ModularOptions &options,
    const std::vector<ModularMultiplierInfo> &multiplier_info = {},
    StaticPropRange static_prop_range = {}, ThreadPool *pool = nullptr) {
  Tree tree;
  for (size_t i = 0; i < kNumStaticProperties; i++) {
    if (static_prop_range[i][1] == 0) {
      static_prop_range[i][1] = std::numeric_limits<uint32_t>::max();
    }
  }
  if (!tree_samples.HasSamples()) {
    tree.emplace_back();
    tree.back().predictor = tree_samples.PredictorFromIndex(0);
    tree.back().property = -1;
    tree.back().predictor_offset = 0;
    tree.back().multiplier = 1;
    return tree;
  }
  float pixel_fraction = tree_samples.NumSamples() * 1.0f / total_pixels;
  float required_cost = pixel_fraction * 0.9 + 0.1;
  tree_samples.AllSamplesDone();
  JXL_RETURN_IF_ERROR(ComputeBestTree(
      tree_samples, options.splitting_heuristics_node_threshold * required_cost,
      multiplier_info, static_prop_range, options.fast_decode_multiplier,
      &tree, pool));
  return tree;
}

Status EncodeModularChannelMAANS(const Image &image, pixel_type chan,
                                 const weighted::Header &wp_header,
                                 const Tree &global_tree, Token **tokenpp,
                                 size_t group_id, bool skip_encoder_fast_path) {
  const Channel &channel = image.channel[chan];
  JxlMemoryManager *memory_manager = channel.memory_manager();
  Token *tokenp = *tokenpp;
  JXL_ENSURE(channel.w != 0 && channel.h != 0);

  JXL_DEBUG_V(6,
              "Encoding %" PRIuS "x%" PRIuS
              " channel %d, "
              "(shift=%i,%i)",
              channel.w, channel.h, chan, channel.hshift, channel.vshift);

  std::array<pixel_type, kNumStaticProperties> static_props = {
      {chan, static_cast<int>(group_id)}};
  bool use_wp;
  bool is_wp_only;
  bool is_gradient_only;
  size_t num_props;
  FlatTree tree = FilterTree(global_tree, static_props, &num_props, &use_wp,
                             &is_wp_only, &is_gradient_only);
  MATreeLookup tree_lookup(tree);
  JXL_DEBUG_V(3, "Encoding using a MA tree with %" PRIuS " nodes", tree.size());

  // Check if this tree is a WP-only tree with a small enough property value
  // range.
  // Initialized to avoid clang-tidy complaining.
  auto tree_lut = jxl::make_unique<TreeLut<uint16_t, false, false>>();
  if (is_wp_only) {
    is_wp_only = TreeToLookupTable(tree, *tree_lut);
  }
  if (is_gradient_only) {
    is_gradient_only = TreeToLookupTable(tree, *tree_lut);
  }

  if (is_wp_only && !skip_encoder_fast_path) {
    const ptrdiff_t onerow = channel.plane.PixelsPerRow();
    weighted::State wp_state(wp_header, channel.w, channel.h);
    Properties properties(1);
    bool unhealthy = false;
    for (size_t y = 0; y < channel.h; y++) {
      const pixel_type *JXL_RESTRICT r = channel.Row(y);
      for (size_t x = 0; x < channel.w; x++) {
        size_t offset = 0;
        pixel_type_w left = (x ? r[x - 1] : y ? *(r + x - onerow) : 0);
        pixel_type_w top = (y ? *(r + x - onerow) : left);
        pixel_type_w topleft = (x && y ? *(r + x - 1 - onerow) : left);
        pixel_type_w topright =
            (x + 1 < channel.w && y ? *(r + x + 1 - onerow) : top);
        pixel_type_w toptop = (y > 1 ? *(r + x - onerow - onerow) : top);
        int32_t guess = wp_state.Predict</*compute_properties=*/true>(
            x, y, channel.w, top, left, topright, topleft, toptop, &properties,
            offset);
        uint32_t pos =
            kPropRangeFast +
            jxl::Clamp1(properties[0], -kPropRangeFast, kPropRangeFast - 1);
        uint32_t ctx_id = tree_lut->context_lookup[pos];
        int32_t residual;
        unhealthy |= SubOverflow(r[x], guess, residual);
        *tokenp++ = Token(ctx_id, PackSigned(residual));
        wp_state.UpdateErrors(r[x], x, y, channel.w);
      }
    }
    if (unhealthy) {
      return JXL_FAILURE("Residual overflow");
    }
  } else if (tree.size() == 1 && tree[0].predictor == Predictor::Gradient &&
             tree[0].multiplier == 1 && tree[0].predictor_offset == 0 &&
             !skip_encoder_fast_path) {
    const ptrdiff_t onerow = channel.plane.PixelsPerRow();
    bool unhealthy = false;
    for (size_t y = 0; y < channel.h; y++) {
      const pixel_type *JXL_RESTRICT r = channel.Row(y);
      for (size_t x = 0; x < channel.w; x++) {
        pixel_type_w left = (x ? r[x - 1] : y ? *(r + x - onerow) : 0);
        pixel_type_w top = (y ? *(r + x - onerow) : left);
        pixel_type_w topleft = (x && y ? *(r + x - 1 - onerow) : left);
        int32_t guess = ClampedGradient(top, left, topleft);
        int32_t residual;
        unhealthy |= SubOverflow(r[x], guess, residual);
        *tokenp++ = Token(tree[0].childID, PackSigned(residual));
      }
    }
    if (unhealthy) {
      return JXL_FAILURE("Residual overflow");
    }
  } else if (is_gradient_only && !skip_encoder_fast_path) {
    const ptrdiff_t onerow = channel.plane.PixelsPerRow();
    bool unhealthy = false;
    for (size_t y = 0; y < channel.h; y++) {
      const pixel_type *JXL_RESTRICT r = channel.Row(y);
      for (size_t x = 0; x < channel.w; x++) {
        pixel_type_w left = (x ? r[x - 1] : y ? *(r + x - onerow) : 0);
        pixel_type_w top = (y ? *(r + x - onerow) : left);
        pixel_type_w topleft = (x && y ? *(r + x - 1 - onerow) : left);
        int32_t guess = ClampedGradient(top, left, topleft);
        uint32_t pos =
            kPropRangeFast +
            std::min<pixel_type_w>(
                std::max<pixel_type_w>(-kPropRangeFast, top + left - topleft),
                kPropRangeFast - 1);
        uint32_t ctx_id = tree_lut->context_lookup[pos];
        int32_t residual;
        unhealthy |= SubOverflow(r[x], guess, residual);
        *tokenp++ = Token(ctx_id, PackSigned(residual));
      }
    }
    if (unhealthy) {
      return JXL_FAILURE("Residual overflow");
    }
  } else if (tree.size() == 1 && tree[0].predictor == Predictor::Zero &&
             tree[0].multiplier == 1 && tree[0].predictor_offset == 0 &&
             !skip_encoder_fast_path) {
    for (size_t y = 0; y < channel.h; y++) {
      const pixel_type *JXL_RESTRICT p = channel.Row(y);
      for (size_t x = 0; x < channel.w; x++) {
        *tokenp++ = Token(tree[0].childID, PackSigned(p[x]));
      }
    }
  } else if (tree.size() == 1 && tree[0].predictor != Predictor::Weighted &&
             (tree[0].multiplier & (tree[0].multiplier - 1)) == 0 &&
             tree[0].predictor_offset == 0 && !skip_encoder_fast_path) {
    // multiplier is a power of 2.
    uint32_t mul_shift =
        FloorLog2Nonzero(static_cast<uint32_t>(tree[0].multiplier));
    const ptrdiff_t onerow = channel.plane.PixelsPerRow();
    for (size_t y = 0; y < channel.h; y++) {
      const pixel_type *JXL_RESTRICT r = channel.Row(y);
      for (size_t x = 0; x < channel.w; x++) {
        PredictionResult pred = PredictNoTreeNoWP(channel.w, r + x, onerow, x,
                                                  y, tree[0].predictor);
        pixel_type_w residual = r[x] - pred.guess;
        JXL_DASSERT((residual >> mul_shift) * tree[0].multiplier == residual);
        *tokenp++ = Token(tree[0].childID, PackSigned(residual >> mul_shift));
      }
    }

  } else if (!use_wp && !skip_encoder_fast_path) {
    const ptrdiff_t onerow = channel.plane.PixelsPerRow();
    Properties properties(num_props);
    JXL_ASSIGN_OR_RETURN(
        Channel references,
        Channel::Create(memory_manager,
                        properties.size() - kNumNonrefProperties, channel.w));
    for (size_t y = 0; y < channel.h; y++) {
      const pixel_type *JXL_RESTRICT p = channel.Row(y);
      PrecomputeReferences(channel, y, image, chan, &references);
      InitPropsRow(&properties, static_props, y);
      for (size_t x = 0; x < channel.w; x++) {
        PredictionResult res =
            PredictTreeNoWP(&properties, channel.w, p + x, onerow, x, y,
                            tree_lookup, references);
        pixel_type_w residual = p[x] - res.guess;
        JXL_DASSERT(residual % res.multiplier == 0);
        *tokenp++ = Token(res.context, PackSigned(residual / res.multiplier));
      }
    }
  } else {
    const ptrdiff_t onerow = channel.plane.PixelsPerRow();
    Properties properties(num_props);
    JXL_ASSIGN_OR_RETURN(
        Channel references,
        Channel::Create(memory_manager,
                        properties.size() - kNumNonrefProperties, channel.w));
    weighted::State wp_state(wp_header, channel.w, channel.h);
    for (size_t y = 0; y < channel.h; y++) {
      const pixel_type *JXL_RESTRICT p = channel.Row(y);
      PrecomputeReferences(channel, y, image, chan, &references);
      InitPropsRow(&properties, static_props, y);
      for (size_t x = 0; x < channel.w; x++) {
        PredictionResult res =
            PredictTreeWP(&properties, channel.w, p + x, onerow, x, y,
                          tree_lookup, references, &wp_state);
        pixel_type_w residual = p[x] - res.guess;
        JXL_DASSERT(residual % res.multiplier == 0);
        *tokenp++ = Token(res.context, PackSigned(residual / res.multiplier));
        wp_state.UpdateErrors(p[x], x, y, channel.w);
      }
    }
  }
  *tokenpp = tokenp;
  return true;
}

}  // namespace

Tree PredefinedTree(ModularOptions::TreeKind tree_kind, size_t total_pixels,
                    int bitdepth, int prevprop) {
  switch (tree_kind) {
    case ModularOptions::TreeKind::kJpegTranscodeACMeta:
      // All the data is 0, so no need for a fancy tree.
      return {PropertyDecisionNode::Leaf(Predictor::Zero)};
    case ModularOptions::TreeKind::kTrivialTreeNoPredictor:
      // All the data is 0, so no need for a fancy tree.
      return {PropertyDecisionNode::Leaf(Predictor::Zero)};
    case ModularOptions::TreeKind::kFalconACMeta:
      // All the data is 0 except the quant field. TODO(veluca): make that 0
      // too.
      return {PropertyDecisionNode::Leaf(Predictor::Left)};
    case ModularOptions::TreeKind::kACMeta: {
      // Small image.
      if (total_pixels < 1024) {
        return {PropertyDecisionNode::Leaf(Predictor::Left)};
      }
      Tree tree;
      // 0: c > 1
      tree.push_back(PropertyDecisionNode::Split(0, 1, 1));
      // 1: c > 2
      tree.push_back(PropertyDecisionNode::Split(0, 2, 3));
      // 2: c > 0
      tree.push_back(PropertyDecisionNode::Split(0, 0, 5));
      // 3: EPF control field (all 0 or 4), top > 3
      tree.push_back(PropertyDecisionNode::Split(6, 3, 21));
      // 4: ACS+QF, y > 0
      tree.push_back(PropertyDecisionNode::Split(2, 0, 7));
      // 5: CfL x
      tree.push_back(PropertyDecisionNode::Leaf(Predictor::Gradient));
      // 6: CfL b
      tree.push_back(PropertyDecisionNode::Leaf(Predictor::Gradient));
      // 7: QF: split according to the left quant value.
      tree.push_back(PropertyDecisionNode::Split(7, 5, 9));
      // 8: ACS: split in 4 segments (8x8 from 0 to 3, large square 4-5, large
      // rectangular 6-11, 8x8 12+), according to previous ACS value.
      tree.push_back(PropertyDecisionNode::Split(7, 5, 15));
      // QF
      tree.push_back(PropertyDecisionNode::Split(7, 11, 11));
      tree.push_back(PropertyDecisionNode::Split(7, 3, 13));
      tree.push_back(PropertyDecisionNode::Leaf(Predictor::Left));
      tree.push_back(PropertyDecisionNode::Leaf(Predictor::Left));
      tree.push_back(PropertyDecisionNode::Leaf(Predictor::Left));
      tree.push_back(PropertyDecisionNode::Leaf(Predictor::Left));
      // ACS
      tree.push_back(PropertyDecisionNode::Split(7, 11, 17));
      tree.push_back(PropertyDecisionNode::Split(7, 3, 19));
      tree.push_back(PropertyDecisionNode::Leaf(Predictor::Zero));
      tree.push_back(PropertyDecisionNode::Leaf(Predictor::Zero));
      tree.push_back(PropertyDecisionNode::Leaf(Predictor::Zero));
      tree.push_back(PropertyDecisionNode::Leaf(Predictor::Zero));
      // EPF, left > 3
      tree.push_back(PropertyDecisionNode::Split(7, 3, 23));
      tree.push_back(PropertyDecisionNode::Split(7, 3, 25));
      tree.push_back(PropertyDecisionNode::Leaf(Predictor::Zero));
      tree.push_back(PropertyDecisionNode::Leaf(Predictor::Zero));
      tree.push_back(PropertyDecisionNode::Leaf(Predictor::Zero));
      tree.push_back(PropertyDecisionNode::Leaf(Predictor::Zero));
      return tree;
    }
    case ModularOptions::TreeKind::kWPFixedDC: {
      std::vector<int32_t> cutoffs = {
          -500, -392, -255, -191, -127, -95, -63, -47, -31, -23, -15,
          -11,  -7,   -4,   -3,   -1,   0,   1,   3,   5,   7,   11,
          15,   23,   31,   47,   63,   95,  127, 191, 255, 392, 500};
      return MakeFixedTree(kWPProp, cutoffs, Predictor::Weighted, total_pixels,
                           bitdepth);
    }
    case ModularOptions::TreeKind::kGradientFixedDC: {
      std::vector<int32_t> cutoffs = {
          -500, -392, -255, -191, -127, -95, -63, -47, -31, -23, -15,
          -11,  -7,   -4,   -3,   -1,   0,   1,   3,   5,   7,   11,
          15,   23,   31,   47,   63,   95,  127, 191, 255, 392, 500};
      return MakeFixedTree(
          prevprop > 0 ? kNumNonrefProperties + 2 : kGradientProp, cutoffs,
          Predictor::Gradient, total_pixels, bitdepth);
    }
    case ModularOptions::TreeKind::kLearn: {
      JXL_DEBUG_ABORT("internal: kLearn is not predefined tree");
      return {};
    }
  }
  JXL_DEBUG_ABORT("internal: unexpected TreeKind: %d",
                  static_cast<int>(tree_kind));
  return {};
}

StatusOr<Tree> LearnTree(
    const Image *images, const ModularOptions *options, const uint32_t start,
    const uint32_t stop,
    const std::vector<ModularMultiplierInfo> &multiplier_info = {},
    ThreadPool *pool = nullptr, ThreadPool *gather_pool = nullptr) {
  if (auto* audit = options[start].histogram_params.direct_audit) {
    if (!audit->allow_ma_learning) {
      ++audit->forbidden_searches;
      return JXL_FAILURE("Direct plan reached MA tree learning");
    }
    ++audit->ma_learning_calls;
  }
  LosslessProfileScope profile(options[start].histogram_params.lossless_profile,
                               "ma.learn");
  profile.Set("stream_start", start);
  profile.Set("stream_stop", stop);
  profile.Set("predictor", static_cast<uint32_t>(options[start].predictor));
  LosslessProfileScope prepare(profile.Id(), "ma.prepare");
  const char* trace_env = std::getenv("JXL_LOSSLESS_SEARCH_TRACE");
  const bool trace = options[start].lossless_tree_speedups &&
                     images[start].w * images[start].h >= 65536 && trace_env &&
                     std::strcmp(trace_env, "1") == 0;
  using Clock = std::chrono::steady_clock;
  const auto before_prepare = trace ? Clock::now() : Clock::time_point{};
  TreeSamples tree_samples(options[start].lossless_tree_speedups,
                            options[start].lossless_fast_entropy,
                            options[start].lossless_channel_trees);
  JXL_RETURN_IF_ERROR(tree_samples.SetPredictor(options[start].predictor,
                                                options[start].wp_tree_mode));
  const char* pair_env = std::getenv("JXL_MA_PREDICTOR_PAIR");
  const bool predictor_pair = tree_samples.LosslessSpeedups() &&
      (!pair_env || std::strcmp(pair_env, "1") == 0) &&
      tree_samples.NumPredictors() == 2 &&
      tree_samples.PredictorFromIndex(0) == Predictor::Weighted &&
      tree_samples.PredictorFromIndex(1) == Predictor::Gradient;
  const char* sparse_env = std::getenv("JXL_MA_SPARSE_GATHER");
  const bool sparse_gather = tree_samples.LosslessSpeedups() &&
      sparse_env && std::strcmp(sparse_env, "1") == 0;
  const char* sample_batch_env = std::getenv("JXL_MA_SAMPLE_BATCH");
  const bool sample_batch = tree_samples.LosslessSpeedups() && sample_batch_env &&
      std::strcmp(sample_batch_env, "1") == 0;
  JXL_RETURN_IF_ERROR(
      tree_samples.SetProperties(options[start].splitting_heuristics_properties,
                                 options[start].wp_tree_mode));
  uint32_t max_c = 0;
  std::vector<pixel_type> pixel_samples;
  std::vector<pixel_type> diff_samples;
  std::vector<uint32_t> group_pixel_count;
  std::vector<uint32_t> channel_pixel_count;
  for (uint32_t i = start; i < stop; i++) {
    max_c = std::max<uint32_t>(images[i].channel.size(), max_c);
    CollectPixelSamples(images[i], options[i], i, group_pixel_count,
                        channel_pixel_count, pixel_samples, diff_samples);
  }
  size_t variable_channels = max_c;
  std::vector<pixel_type> constants(max_c);
  const char* constant_env = std::getenv("JXL_MA_CONSTANT_CHANNELS");
  if (options[start].lossless_tree_speedups && !options[start].ma_search &&
      !options[start].histogram_params.direct_audit && constant_env &&
      std::strcmp(constant_env, "1") == 0) {
    // Only a globally constant trailing channel can bypass training. Check
    // every present group, including metadata-channel collisions, exactly.
    while (variable_channels) {
      const size_t c = variable_channels - 1;
      bool seen = false, constant = true;
      for (size_t i = start; i < stop && constant; ++i) {
        if (c >= images[i].channel.size()) continue;
        const auto& ch = images[i].channel[c];
        if (!ch.w || !ch.h) continue;
        if (c < images[i].nb_meta_channels || ch.w > options[i].max_chan_size ||
            ch.h > options[i].max_chan_size) { constant = false; break; }
        if (!seen) constants[c] = ch.Row(0)[0];
        seen = true;
        for (size_t y = 0; y < ch.h && constant; ++y) {
          const auto* row = ch.Row(y);
          constant = std::all_of(row, row + ch.w,
              [&](pixel_type value) { return value == constants[c]; });
        }
      }
      if (!seen || !constant) break;
      --variable_channels;
    }
  }
  profile.Set("constant_channels", max_c - variable_channels);
  StaticPropRange range;
  range[0] = {{0, max_c}};
  range[1] = {{start, stop}};

  tree_samples.PreQuantizeProperties(
      range, multiplier_info, group_pixel_count, channel_pixel_count,
      pixel_samples, diff_samples, options[start].max_property_values);
  const auto before_gather = trace ? Clock::now() : Clock::time_point{};
  prepare.Finish();
  LosslessProfileScope gather(profile.Id(), "ma.gather");

  size_t total_pixels = 0;
  for (size_t i = 0; i < images[start].channel.size(); i++) {
    if (i >= images[start].nb_meta_channels &&
        (images[start].channel[i].w > options[start].max_chan_size ||
         images[start].channel[i].h > options[start].max_chan_size)) {
      break;
    }
    if (i < variable_channels)
      total_pixels += images[start].channel[i].w * images[start].channel[i].h;
  }
  total_pixels = std::max<size_t>(total_pixels, 1);

  struct GatherTask {
    size_t image, channel, first_row, last_row;
  };
  std::vector<GatherTask> tasks;
  std::vector<uint64_t> work;
  std::vector<double> task_samples;
  double expected_samples = 0;
  const size_t stripe_count = options[start].lossless_channel_trees &&
      !options[start].ma_search && !options[start].histogram_params.direct_audit
      ? LosslessWorkerSetting(std::getenv("JXL_MA_GATHER_STRIPES"), 1, 16) : 1;
  for (size_t i = start; i < stop; i++) {
    size_t nb_channels = images[i].channel.size();

    if (images[i].w == 0 || images[i].h == 0 || nb_channels < 1)
      continue;  // is there any use for a zero-channel image?
    if (images[i].error) return JXL_FAILURE("Invalid image");
    JXL_ENSURE(options[i].tree_kind == ModularOptions::TreeKind::kLearn);

    JXL_DEBUG_V(
        2, "Encoding %" PRIuS "-channel, %i-bit, %" PRIuS "x%" PRIuS " image.",
        nb_channels, images[i].bitdepth, images[i].w, images[i].h);

    for (size_t c = 0; c < nb_channels; c++) {
      if (c >= images[i].nb_meta_channels &&
          (images[i].channel[c].w > options[i].max_chan_size ||
           images[i].channel[c].h > options[i].max_chan_size)) {
        break;
      }
      if (!images[i].channel[c].w || !images[i].channel[c].h) {
        continue;  // skip empty channels
      }
      if (c >= variable_channels) continue;
      const Channel& channel = images[i].channel[c];
      const size_t parts = channel.h >= 512 && uint64_t{channel.w} * channel.h >= (1u << 18)
          ? std::min(stripe_count, channel.h / 128) : 1;
      for (size_t part = 0; part < parts; ++part) {
        const size_t first = channel.h * part / parts, last = channel.h * (part + 1) / parts;
        tasks.push_back({i, c, first, last});
        work.push_back(uint64_t{channel.w} * (last - first));
        task_samples.push_back(TreeSampleFraction(channel, options[i]) * work.back() + 64);
        expected_samples += task_samples.back();
      }
    }
  }
  const auto collect = [&](size_t first, size_t last, TreeSamples& samples,
                           size_t* pixels) -> Status {
    samples.SetSampleBatching(sample_batch);
    for (size_t task = first; task < last; ++task) {
      const size_t i = tasks[task].image;
      const size_t c = tasks[task].channel;
      weighted::Header wp_header;
      Bundle::Init(&wp_header);
      if (PredictorHasWeighted(options[i].predictor)) {
        weighted::PredictorMode(options[i].wp_mode, &wp_header);
      }
      if (predictor_pair) {
        JXL_RETURN_IF_ERROR(GatherTreeData<true>(
            images[i], c, i, wp_header, options[i], samples, pixels,
            sparse_gather, tasks[task].first_row, tasks[task].last_row));
      } else {
        JXL_RETURN_IF_ERROR(GatherTreeData<false>(
            images[i], c, i, wp_header, options[i], samples, pixels,
            sparse_gather, tasks[task].first_row, tasks[task].last_row));
      }
    }
    return true;
  };

  // All batches use the global quantizer. Only collection is partitioned;
  // the merged weighted samples still train one global tree.
  constexpr uint64_t kGatherBudget = uint64_t{1} << 30;
  const size_t gather_limit = tree_samples.LosslessSpeedups()
      ? LosslessWorkerLimits::FromEnvironment().tasks : 8;
  const size_t sample_bytes = 2 * tree_samples.NumPredictors() +
      sizeof(uint32_t) * tree_samples.NumStaticProps() +
      tree_samples.NumProperties() - tree_samples.NumStaticProps() + sizeof(uint16_t);
  const double estimated_bytes = expected_samples * (2 * sample_bytes + 16);
  bool parallel_gather = false;
  size_t gather_workers = 1, gather_chunks = 1;
  size_t gather_waves = 0;
  bool wave_collected = false;
  const char* wave_env = std::getenv("JXL_MA_GATHER_WAVES");
  const bool waves = wave_env && std::strcmp(wave_env, "1") == 0;
  if (waves && gather_pool && tree_samples.LosslessSpeedups() &&
      expected_samples >= 8192 && estimated_bytes > kGatherBudget &&
      !options[start].ma_search && !options[start].histogram_params.direct_audit) {
    const TreeSamples prepared = tree_samples;
    gather_chunks = 0;
    for (size_t first = 0; first < tasks.size();) {
      size_t last = first;
      double wave_bytes = 0;
      while (last < tasks.size()) {
        const double bytes = task_samples[last] * (2 * sample_bytes + 16);
        if (last > first && wave_bytes + bytes > kGatherBudget) break;
        wave_bytes += bytes;
        ++last;
      }
      ++gather_waves;
      LosslessProfileScope wave(gather.Id(), "ma.gather_wave");
      wave.Set("first_channel", first);
      wave.Set("channels", last - first);
      std::vector<uint64_t> wave_work(work.begin() + first, work.begin() + last);
      const auto bounds = MaTreeGroupBoundaries(
          wave_work, gather_limit, 65536, gather_limit);
      bool wave_parallel = false;
      if (wave_bytes <= kGatherBudget && bounds.size() > 2) {
        std::vector<TreeSamples> batches;
        std::vector<size_t> pixels(bounds.size() - 1);
        const auto init = [&](size_t threads) -> Status {
          gather_workers = std::max(gather_workers, threads);
          wave_parallel = threads > 1;
          if (wave_parallel) batches.assign(bounds.size() - 1, prepared);
          return true;
        };
        const auto gather_batch = [&](uint32_t batch, size_t) -> Status {
          if (!wave_parallel) return true;
          LosslessProfileScope chunk(wave.Id(), "ma.gather_chunk");
          JXL_RETURN_IF_ERROR(collect(first + bounds[batch],
                                      first + bounds[batch + 1],
                                      batches[batch], &pixels[batch]));
          chunk.Set("pixels", pixels[batch]);
          chunk.Set("samples", batches[batch].NumDistinctSamples());
          chunk.Set("weighted_samples", batches[batch].NumSamples());
          return true;
        };
        JXL_RETURN_IF_ERROR(RunOnPool(gather_pool, 0, bounds.size() - 1, init,
                                      gather_batch, "MaGatherWave"));
        if (wave_parallel) {
          parallel_gather = true;
          gather_chunks += batches.size();
          LosslessProfileScope merge(wave.Id(), "ma.merge_samples");
          JXL_RETURN_IF_ERROR(tree_samples.AppendSamples(batches));
          for (size_t count : pixels) total_pixels += count;
        }
      }
      if (!wave_parallel) {
        ++gather_chunks;
        JXL_RETURN_IF_ERROR(collect(first, last, tree_samples, &total_pixels));
      }
      wave.Set("parallel", wave_parallel);
      first = last;
    }
    wave_collected = true;
  } else if (gather_pool && tree_samples.LosslessSpeedups() && expected_samples >= 8192 &&
      estimated_bytes <= kGatherBudget && !options[start].ma_search &&
      !options[start].histogram_params.direct_audit) {
    const auto bounds = MaTreeGroupBoundaries(work, gather_limit, 65536, gather_limit);
    if (bounds.size() > 2) {
      std::vector<TreeSamples> batches;
      std::vector<size_t> pixels(bounds.size() - 1);
      const auto init = [&](size_t threads) -> Status {
        gather_workers = threads;
        parallel_gather = threads > 1;
        if (parallel_gather) batches.assign(bounds.size() - 1, tree_samples);
        return true;
      };
      const auto gather_batch = [&](uint32_t batch, size_t /*thread*/) -> Status {
        if (!parallel_gather) return true;
        LosslessProfileScope chunk(gather.Id(), "ma.gather_chunk");
        JXL_RETURN_IF_ERROR(collect(bounds[batch], bounds[batch + 1],
                                     batches[batch], &pixels[batch]));
        chunk.Set("pixels", pixels[batch]);
        chunk.Set("samples", batches[batch].NumDistinctSamples());
        chunk.Set("weighted_samples", batches[batch].NumSamples());
        chunk.Set("channels", bounds[batch + 1] - bounds[batch]);
        return true;
      };
      JXL_RETURN_IF_ERROR(RunOnPool(gather_pool, 0, bounds.size() - 1, init,
                                    gather_batch, "MaGatherChannels"));
      if (parallel_gather) {
        gather_chunks = batches.size();
        LosslessProfileScope merge(gather.Id(), "ma.merge_samples");
        JXL_RETURN_IF_ERROR(tree_samples.MergeSamples(batches));
        for (size_t count : pixels) total_pixels += count;
      }
    }
  }
  if (!parallel_gather && !wave_collected) {
    JXL_RETURN_IF_ERROR(collect(0, tasks.size(), tree_samples, &total_pixels));
  }

  const auto before_split = trace ? Clock::now() : Clock::time_point{};
  const size_t samples = tree_samples.NumDistinctSamples();
  gather.Set("samples", samples);
  gather.Set("pixels", total_pixels);
  gather.Set("weighted_samples", tree_samples.NumSamples());
  gather.Set("fast_dedup", tree_samples.FastDedup());
  gather.Set("predictor_pair", predictor_pair);
  gather.Set("sparse_gather", sparse_gather);
  gather.Set("sample_batch", sample_batch);
  gather.Set("parallel_gather", parallel_gather);
  gather.Set("gather_worker_slots", gather_workers);
  gather.Set("gather_chunks", gather_chunks);
  gather.Set("gather_waves", gather_waves);
  gather.Set("gather_channels", tasks.size());
  gather.Set("stripe_limit", stripe_count);
  gather.Set("estimated_batch_bytes", static_cast<uint64_t>(
      std::min<double>(estimated_bytes, kGatherBudget + 1)));
  gather.Finish();
  LosslessProfileScope split(profile.Id(), "ma.split");
  if (options[start].ma_search) {
    options[start].ma_search->Gather(tree_samples, options[start]);
  }
  JXL_ASSIGN_OR_RETURN(Tree tree,
                       LearnTree(std::move(tree_samples), total_pixels,
                                 options[start], multiplier_info, range, pool));
  if (variable_channels == 0 && max_c) {
    tree = {PropertyDecisionNode::Leaf(Predictor::Zero, constants[0])};
  }
  for (size_t c = std::max<size_t>(1, variable_channels); c < max_c; ++c) {
    // Static property zero is the channel number; the left child is > split.
    for (auto& node : tree) {
      if (node.property >= 0) { node.lchild += 2; node.rchild += 2; }
    }
    tree.insert(tree.begin(), PropertyDecisionNode::Leaf(Predictor::Zero, constants[c]));
    tree.insert(tree.begin(), PropertyDecisionNode::Split(0, c - 1, 1, 2));
  }
  split.Set("samples", samples);
  split.Set("nodes", tree.size());
  if (options[start].ma_search) {
    options[start].ma_search->FinishTree(tree.size());
  }
  split.Finish();
  if (trace) {
    std::fprintf(stderr,
                 "lossless-learn samples=%" PRIuS " nodes=%" PRIuS
                 " prepare_s=%.6f gather_s=%.6f split_s=%.6f\n",
                 samples, tree.size(),
                 std::chrono::duration<double>(before_gather - before_prepare).count(),
                 std::chrono::duration<double>(before_split - before_gather).count(),
                 std::chrono::duration<double>(Clock::now() - before_split).count());
    std::fflush(stderr);
  }
  return tree;
}

Status ExportNeuralTreeSamples(const Image *images, const ModularOptions *opts,
                              size_t count, const char *path) {
  static_assert(kNumNonrefProperties == 16, "sample protocol properties");
  static_assert(kNumModularPredictors == 14, "sample protocol predictors");
  size_t total = 0;
  size_t num_properties = kNumNonrefProperties;
  for (size_t i = 0; i < count; ++i) {
    num_properties = std::max(num_properties,
        kNumNonrefProperties + kExtraPropsPerChannel * size_t(opts[i].max_properties));
    if (images[i].empty()) continue;
    for (size_t c = 0; c < images[i].channel.size(); ++c) {
      const Channel &ch = images[i].channel[c];
      if (c >= images[i].nb_meta_channels &&
          (ch.w > opts[i].max_chan_size || ch.h > opts[i].max_chan_size)) break;
      total += ch.w * ch.h;
    }
  }
  std::ofstream out(path, std::ios::out | std::ios::trunc);
  out.imbue(std::locale::classic());
  JXL_ENSURE(out.good());
  JXL_ENSURE(num_properties <= 256);
  out << "JXL_MA_SAMPLES_V1 " << num_properties << " 14 " << total << '\n';
  // One reproducible, jittered sample per equal-sized population stratum.
  // The final stratum has its actual weight, not the full stride.
  const size_t stride = std::max<size_t>(1, (total + 8191) / 8192);
  auto jitter = [](uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
  };
  size_t ordinal = 0, bin = 0, weight = std::min(stride, total);
  size_t next = weight ? jitter(0) % weight : 0;
  for (size_t i = 0; i < count; ++i) {
    const Image &image = images[i];
    if (image.empty()) continue;
    weighted::Header header;
    if (PredictorHasWeighted(opts[i].predictor)) {
      weighted::PredictorMode(opts[i].wp_mode, &header);
    }
    for (size_t c = 0; c < image.channel.size(); ++c) {
      const Channel &ch = image.channel[c];
      if (c >= image.nb_meta_channels &&
          (ch.w > opts[i].max_chan_size || ch.h > opts[i].max_chan_size)) break;
      if (!ch.w || !ch.h) continue;
      Properties properties(num_properties);
      const std::array<pixel_type, kNumStaticProperties> static_props = {
          {static_cast<pixel_type>(c), static_cast<pixel_type>(i)}};
      JXL_ASSIGN_OR_RETURN(Channel references,
                           Channel::Create(ch.memory_manager(),
                               num_properties - kNumNonrefProperties, ch.w));
      weighted::State state(header, ch.w, ch.h);
      const ptrdiff_t onerow = ch.plane.PixelsPerRow();
      for (size_t y = 0; y < ch.h; ++y) {
        const pixel_type *p = ch.Row(y);
        PrecomputeReferences(ch, y, image, c, &references);
        InitPropsRow(&properties, static_props, y);
        for (size_t x = 0; x < ch.w; ++x, ++ordinal) {
          if (ordinal == next) {
            pixel_type_w predictions[kNumModularPredictors];
            PredictLearnAll(&properties, ch.w, p + x, onerow, x, y,
                            references, &state, predictions);
            out << weight;
            for (auto value : properties) out << ' ' << value;
            for (auto prediction : predictions) {
              out << ' ' << (static_cast<int64_t>(p[x]) - prediction);
            }
            out << '\n';
            bin += weight;
            weight = std::min(stride, total - bin);
            next = weight ? bin + jitter(bin) % weight : total;
          } else {
            PredictLearn(&properties, ch.w, p + x, onerow, x, y,
                         Predictor::Zero, references, &state);
          }
          state.UpdateErrors(p[x], x, y, ch.w);
        }
      }
    }
  }
  JXL_ENSURE(ordinal == total && bin == total);
  out.close();
  JXL_ENSURE(!out.fail());
  return true;
}

Status ModularCompress(const Image &image, const ModularOptions &options,
                       size_t group_id, const Tree &tree, GroupHeader &header,
                       std::vector<Token> &tokens, size_t *width,
                       ThreadPool* channel_pool) {
  LosslessProfileScope profile(options.histogram_params.lossless_profile,
                               "tokens.generate");
  profile.Set("group", group_id);
  profile.Set("width", image.w);
  profile.Set("height", image.h);
  profile.Set("channels", image.channel.size());
  size_t nb_channels = image.channel.size();

  if (image.w == 0 || image.h == 0 || nb_channels < 1)
    return true;  // is there any use for a zero-channel image?
  if (image.error) return JXL_FAILURE("Invalid image");

  JXL_DEBUG_V(
      2, "Encoding %" PRIuS "-channel, %i-bit, %" PRIuS "x%" PRIuS " image.",
      nb_channels, image.bitdepth, image.w, image.h);

  // encode transforms
  Bundle::Init(&header);
  if (PredictorHasWeighted(options.predictor)) {
    weighted::PredictorMode(options.wp_mode, &header.wp_header);
  }
  header.transforms = image.transform;
  header.use_global_tree = true;

  size_t image_width = 0;
  size_t total_tokens = 0;
  for (size_t i = 0; i < nb_channels; i++) {
    if (i >= image.nb_meta_channels &&
        (image.channel[i].w > options.max_chan_size ||
         image.channel[i].h > options.max_chan_size)) {
      break;
    }
    if (image.channel[i].w > image_width) image_width = image.channel[i].w;
    total_tokens += image.channel[i].w * image.channel[i].h;
  }
  if (options.zero_tokens) {
    tokens.resize(tokens.size() + total_tokens, {0, 0});
  } else {
    // Do one big allocation for all the tokens we'll need,
    // to avoid reallocs that might require copying.
    size_t pos = tokens.size();
    tokens.resize(pos + total_tokens);
    Token *tokenp = tokens.data() + pos;
    std::vector<size_t> channels, offsets;
    for (size_t i = 0; i < nb_channels; i++) {
      if (i >= image.nb_meta_channels &&
          (image.channel[i].w > options.max_chan_size ||
           image.channel[i].h > options.max_chan_size)) {
        break;
      }
      if (!image.channel[i].w || !image.channel[i].h) {
        continue;  // skip empty channels
      }
      channels.push_back(i);
      offsets.push_back(tokenp - tokens.data());
      tokenp += image.channel[i].w * image.channel[i].h;
    }
    const auto encode_channel = [&](uint32_t task, size_t) -> Status {
      const size_t i = channels[task];
      Token* destination = tokens.data() + offsets[task];
      JXL_RETURN_IF_ERROR(
          EncodeModularChannelMAANS(image, i, header.wp_header, tree, &destination,
                                    group_id, options.skip_encoder_fast_path));
      JXL_ENSURE(destination == tokens.data() + offsets[task] +
                                 image.channel[i].w * image.channel[i].h);
      return true;
    };
    if (channel_pool && channels.size() > 1 && total_tokens >= 65536) {
      profile.Set("parallel_channels", channels.size());
      JXL_RETURN_IF_ERROR(RunOnPool(channel_pool, 0, channels.size(), ThreadPool::NoInit,
                                    encode_channel, "ModularTokenChannels"));
    } else {
      for (size_t task = 0; task < channels.size(); ++task) {
        JXL_RETURN_IF_ERROR(encode_channel(task, 0));
      }
    }
    // Make sure we actually wrote all tokens
    JXL_ENSURE(tokenp == tokens.data() + tokens.size());
  }

  *width = image_width;

  return true;
}

Status ModularGenericCompress(const Image &image, const ModularOptions &opts,
                              BitWriter &writer, AuxOut *aux_out,
                              LayerType layer, size_t group_id) {
  size_t nb_channels = image.channel.size();

  if (image.w == 0 || image.h == 0 || nb_channels < 1)
    return true;  // is there any use for a zero-channel image?
  if (image.error) return JXL_FAILURE("Invalid image");

  const char* trace_value = std::getenv("JXL_LOSSLESS_SEARCH_TRACE");
  const bool trace = opts.lossless_tree_speedups &&
                     uint64_t{image.w} * image.h >= 65536 && trace_value &&
                     std::strcmp(trace_value, "1") == 0;
  using Clock = std::chrono::steady_clock;
  const auto start = trace ? Clock::now() : Clock::time_point{};

  ModularOptions options = opts;  // Make a copy to modify it.
  if (options.predictor == kUndefinedPredictor) {
    options.predictor = Predictor::Gradient;
  }

  size_t bits = writer.BitsWritten();

  JxlMemoryManager *memory_manager = image.memory_manager();
  JXL_DEBUG_V(
      2, "Encoding %" PRIuS "-channel, %i-bit, %" PRIuS "x%" PRIuS " image.",
      nb_channels, image.bitdepth, image.w, image.h);

  // encode transforms
  GroupHeader header;
  Bundle::Init(&header);
  if (PredictorHasWeighted(options.predictor)) {
    weighted::PredictorMode(options.wp_mode, &header.wp_header);
  }
  header.transforms = image.transform;

  JXL_RETURN_IF_ERROR(Bundle::Write(header, &writer, layer, aux_out));

  // Compute tree.
  Tree tree;
  if (options.tree_kind == ModularOptions::TreeKind::kLearn) {
    JXL_ASSIGN_OR_RETURN(tree, LearnTree(&image, &options, 0, 1));
  } else {
    size_t total_pixels = 0;
    for (size_t i = 0; i < nb_channels; i++) {
      if (i >= image.nb_meta_channels &&
          (image.channel[i].w > options.max_chan_size ||
           image.channel[i].h > options.max_chan_size)) {
        break;
      }
      total_pixels += image.channel[i].w * image.channel[i].h;
    }
    total_pixels = std::max<size_t>(total_pixels, 1);

    tree = PredefinedTree(options.tree_kind, total_pixels, image.bitdepth,
                          options.max_properties);
  }

  const auto after_tree = trace ? Clock::now() : Clock::time_point{};
  Tree decoded_tree;
  std::vector<std::vector<Token>> tree_tokens(1);
  JXL_RETURN_IF_ERROR(TokenizeTree(tree, tree_tokens.data(), &decoded_tree));
  JXL_ENSURE(tree.size() == decoded_tree.size());
  tree = std::move(decoded_tree);

  // Write tree
  EntropyEncodingData code;
  JXL_ASSIGN_OR_RETURN(
      size_t cost,
      BuildAndEncodeHistograms(memory_manager, options.histogram_params,
                               kNumTreeContexts, tree_tokens, &code, &writer,
                               LayerType::ModularTree, aux_out));
  JXL_RETURN_IF_ERROR(WriteTokens(tree_tokens[0], code, 0, &writer,
                                  LayerType::ModularTree, aux_out));

  size_t image_width = 0;
  std::vector<std::vector<Token>> tokens(1);
  // it puts `use_global_tree = true` in the header, but this is not used
  // further
  JXL_RETURN_IF_ERROR(ModularCompress(image, options, group_id, tree, header,
                                      tokens[0], &image_width, nullptr));
  const auto after_tokens = trace ? Clock::now() : Clock::time_point{};

  // Write data
  code = {};
  HistogramParams histo_params = options.histogram_params;
  histo_params.image_widths.push_back(image_width);
  JXL_ASSIGN_OR_RETURN(
      cost, BuildAndEncodeHistograms(memory_manager, histo_params,
                                     (tree.size() + 1) / 2, tokens, &code,
                                     &writer, layer, aux_out));
  (void)cost;
  JXL_RETURN_IF_ERROR(WriteTokens(tokens[0], code, 0, &writer, layer, aux_out));

  if (trace) {
    std::fprintf(stderr,
                 "lossless-local group=%" PRIuS " width=%" PRIuS
                 " height=%" PRIuS " tree_s=%.6f tokens_s=%.6f entropy_s=%.6f\n",
                 group_id, image.w, image.h,
                 std::chrono::duration<double>(after_tree - start).count(),
                 std::chrono::duration<double>(after_tokens - after_tree).count(),
                 std::chrono::duration<double>(Clock::now() - after_tokens).count());
    std::fflush(stderr);
  }

  bits = writer.BitsWritten() - bits;
  JXL_DEBUG_V(4,
              "Modular-encoded a %" PRIuS "x%" PRIuS
              " bitdepth=%i nbchans=%" PRIuS " image in %" PRIuS " bytes",
              image.w, image.h, image.bitdepth, image.channel.size(), bits / 8);
  (void)bits;

  return true;
}

}  // namespace jxl
