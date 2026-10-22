// Copyright (c) the JPEG XL Project Authors.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef LIB_JXL_ENC_LOSSLESS_RACING_H_
#define LIB_JXL_ENC_LOSSLESS_RACING_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace jxl {

struct LosslessPairedEvidence {
  size_t samples = 0;
  double mean = 0, radius = std::numeric_limits<double>::infinity();
  double minimum = std::numeric_limits<double>::infinity();

  bool ClearlyWorse(double tolerance) const {
    return samples >= 3 && minimum > 0 && mean - radius > tolerance;
  }
};

// A conservative screening heuristic, NOT a confidence guarantee: neighboring
// image regions and learned entropy estimates are not independent observations.
inline LosslessPairedEvidence PairedLosslessEvidence(
    const std::vector<double>& candidate, const std::vector<double>& reference,
    double bias_margin) {
  LosslessPairedEvidence result;
  if (candidate.size() != reference.size()) return result;
  double m2 = 0;
  for (size_t i = 0; i < candidate.size(); ++i) {
    if (!std::isfinite(candidate[i]) || !std::isfinite(reference[i]) ||
        candidate[i] < 0 || reference[i] <= 0) {
      return {};
    }
    const double delta = candidate[i] / reference[i] - 1.0;
    if (!std::isfinite(delta)) return {};
    ++result.samples;
    const double old = result.mean;
    result.mean += (delta - old) / result.samples;
    m2 += (delta - old) * (delta - result.mean);
    result.minimum = std::min(result.minimum, delta);
  }
  if (result.samples >= 3) {
    result.radius = std::max(0.0, bias_margin) +
        2.5 * std::sqrt(std::max(0.0, m2) /
                       (result.samples * (result.samples - 1)));
  }
  return result;
}

// Optimistic whole-frame projection from a completed peer and paired regions.
// This is not a byte lower bound; correlated crops can miss global structure.
inline double ConservativeLosslessProjection(
    size_t reference_bytes, const std::vector<size_t>& candidate,
    const std::vector<size_t>& reference) {
  const auto evidence = PairedLosslessEvidence(
      std::vector<double>(candidate.begin(), candidate.end()),
      std::vector<double>(reference.begin(), reference.end()), 0.10);
  if (evidence.samples < 3 || !std::isfinite(evidence.radius)) return 0;
  const double ratio = std::max(
      0.0, 1.0 + std::min(evidence.minimum, evidence.mean - evidence.radius));
  return reference_bytes * ratio;
}

}  // namespace jxl
#endif
