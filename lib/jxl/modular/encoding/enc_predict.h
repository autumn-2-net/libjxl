// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef LIB_JXL_MODULAR_ENCODING_ENC_PREDICT_H_
#define LIB_JXL_MODULAR_ENCODING_ENC_PREDICT_H_

#include <algorithm>
#include <vector>

#include "lib/jxl/modular/encoding/context_predict.h"

namespace jxl {

template <bool kProperties>
JXL_INLINE pixel_type_w PredictWeightedFlat(
    weighted::State* state, size_t x, size_t y, size_t width,
    pixel_type_w n, pixel_type_w w, pixel_type_w ne, pixel_type_w nw,
    pixel_type_w nn, Properties* properties, size_t offset) {
  if (n == w && n == ne && n == nw && n == nn) {
    const size_t current = (y & 1) ? 0 : width + 2;
    const size_t previous = (y & 1) ? width + 2 : 0;
    const size_t north = previous + x;
    const size_t northeast = x + 1 < width ? north + 1 : north;
    const size_t northwest = x ? north - 1 : north;
    if ((!x || state->error[current + x - 1] == 0) &&
        state->error[north] == 0 && state->error[northeast] == 0 &&
        state->error[northwest] == 0) {
      // Every sub-predictor equals N. Zero signed errors force clamping to
      // [N, N], regardless of the accumulated unsigned predictor weights.
      state->pred = weighted::State::AddBits(n);
      for (auto& prediction : state->prediction) prediction = state->pred;
      if constexpr (kProperties) (*properties)[offset] = 0;
      return n;
    }
  }
  return state->Predict<kProperties>(x, y, width, n, w, ne, nw, nn, properties, offset);
}

// PrecomputeReferences materializes an entire row. Learning at low density
// only needs reference properties at selected x positions.
class SparseLearnReferences {
 public:
  SparseLearnReferences(const Image& image, size_t channel, size_t properties) {
    const auto& ch = image.channel[channel];
    for (size_t j = channel; j > 0 &&
         channels_.size() * kExtraPropsPerChannel < properties;) {
      const auto& ref = image.channel[--j];
      if (ref.w == ch.w && ref.h == ch.h && ref.hshift == ch.hshift &&
          ref.vshift == ch.vshift) {
        channels_.push_back(&ref);
      }
    }
  }

  JXL_INLINE void Fill(size_t x, size_t y, Channel* references) const {
    JXL_DASSERT(references->w % kExtraPropsPerChannel == 0);
    pixel_type* out = references->Row(x);
    size_t offset = 0;
    for (const Channel* channel : channels_) {
      const pixel_type* row = channel->Row(y);
      const pixel_type* above = channel->Row(y ? y - 1 : 0);
      const pixel_type_w value = row[x];
      const pixel_type_w left = x ? row[x - 1] : 0;
      const pixel_type_w top = y ? above[x] : left;
      const pixel_type_w top_left = x && y ? above[x - 1] : left;
      const pixel_type_w residual = value - ClampedGradient(left, top, top_left);
      out[offset++] = std::abs(value);
      out[offset++] = value;
      out[offset++] = std::abs(residual);
      out[offset++] = residual;
    }
    std::fill(out + offset, out + references->w, 0);
  }

 private:
  std::vector<const Channel*> channels_;
};

// An unselected learning pixel still advances causal WP errors and the
// previous-gradient property. Its other properties are never consumed.
template <bool kNoEdgeCases>
JXL_INLINE void AdvanceLearnState(
    Properties* properties, size_t width, const pixel_type* JXL_RESTRICT pixel,
    ptrdiff_t stride, size_t x, size_t y, weighted::State* state,
    bool flat = false) {
  const pixel_type_w left =
      kNoEdgeCases || x ? pixel[-1] : (y ? pixel[-stride] : 0);
  const pixel_type_w top = kNoEdgeCases || y ? pixel[-stride] : left;
  const pixel_type_w top_left =
      kNoEdgeCases || (x && y) ? pixel[-1 - stride] : left;
  (*properties)[9] = left + top - top_left;
  if (flat) {
    const pixel_type_w top_right =
        kNoEdgeCases || (x + 1 < width && y) ? pixel[1 - stride] : top;
    const pixel_type_w top_top =
        kNoEdgeCases || y > 1 ? pixel[-stride - stride] : top;
    PredictWeightedFlat<false>(state, x, y, width, top, left, top_right,
                                top_left, top_top, nullptr, 0);
    return;
  }
  constexpr int mode =
      detail::kUseWP | (kNoEdgeCases ? detail::kNoEdgeCases : 0);
  detail::Predict<mode>(nullptr, width, pixel, stride, x, y, Predictor::Zero,
                        nullptr, nullptr, state, nullptr);
}

// Encoder learning with exactly Weighted and Gradient retained. Compute the
// shared properties and WP state once; other predictions are never consumed.
template <bool kNoEdgeCases>
JXL_INLINE void PredictLearnBest(
    Properties* properties, size_t width, const pixel_type* JXL_RESTRICT pixel,
    ptrdiff_t stride, size_t x, size_t y, const Channel& references,
    weighted::State* state, pixel_type_w* predictions) {
  constexpr int mode = detail::kForceComputeProperties | detail::kUseWP |
                       (kNoEdgeCases ? detail::kNoEdgeCases : 0);
  predictions[static_cast<size_t>(Predictor::Weighted)] =
      detail::Predict<mode>(properties, width, pixel, stride, x, y,
                            Predictor::Weighted, nullptr, &references, state,
                            nullptr).guess;
  const pixel_type left = (*properties)[7];
  const pixel_type top = (*properties)[6];
  const pixel_type top_left =
      kNoEdgeCases || (x && y) ? pixel[-1 - stride] : left;
  predictions[static_cast<size_t>(Predictor::Gradient)] =
      ClampedGradient(left, top, top_left);
}

}  // namespace jxl

#endif  // LIB_JXL_MODULAR_ENCODING_ENC_PREDICT_H_
