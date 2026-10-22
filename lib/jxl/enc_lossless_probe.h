// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_LOSSLESS_PROBE_H_
#define LIB_JXL_ENC_LOSSLESS_PROBE_H_

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/enc_params.h"

namespace jxl {

struct CodecMetadata;
class JxlEncoderChunkedFrameAdapter;
struct LosslessScan;

// Encodes an independent sample frame without recursively invoking the search.
// Batched probes must not use the caller's runner. A serial sample may borrow
// the explicit params.lossless_probe_runner; the outer runner is then idle.
using LosslessProbeEncoder =
    std::function<StatusOr<size_t>(const CompressParams&, const CodecMetadata&,
                                   JxlEncoderChunkedFrameAdapter&)>;

struct LosslessProbeCandidate {
  const char* name;
  CompressParams params;
  size_t bytes = 0;
  size_t small_bytes = 0;
  bool exploratory = false;
  size_t context_bytes = 0;
};

struct LosslessProbeResult {
  // Stable ascending order by sample bytes; controls are not returned.
  std::vector<LosslessProbeCandidate> ranked;
  // Unvalidated cross-channel alternatives for ambiguous whole-frame feedback.
  // Their byte scores cover only the initial screening tiles.
  std::vector<LosslessProbeCandidate> exploratory;
  size_t samples = 0;
  size_t encodes = 0;
  size_t control_bytes = 0;
  size_t small_control_bytes = 0;
  size_t candidates = 0;
  size_t shortlisted = 0;
  size_t screen_samples = 0;
  size_t screen_bounded = 0;
  size_t screen_saved_encodes = 0;
  size_t bounded = 0;
  size_t validation_encodes = 0;
  size_t sample_pixels = 0;
};

struct LosslessProbeConfirmation {
  size_t samples = 0;
  size_t candidate_bytes = 0;
  size_t incumbent_bytes = 0;
  bool accepted = true;
};

// Parameters varied by the scheduler, excluding per-call diagnostic context
// and output budgets. Cached scores are local to one unchanged input frame.
bool SameLosslessCandidate(const CompressParams& a, const CompressParams& b);

struct LosslessCalibrationFeedback {
  CompressParams params;
  std::vector<size_t> deep, cheap;
  size_t full_bytes = 0;

  // A heuristic estimate, not a byte bound. Zero means insufficient evidence.
  double SiblingEstimate(const CompressParams& candidate,
                         const std::vector<size_t>& scores) const;
};

// Shrink noisy regional deep/cheap ratios toward the same-family median.
// This is a ranking estimate, never an encoded-size bound.
std::vector<double> ShrinkLosslessCalibration(
    const std::vector<CompressParams>& plans,
    const std::vector<size_t>& full_bytes,
    const std::vector<double>& projected,
    const std::vector<size_t>& independent_regions);

bool LosslessCalibrationAgreement(const std::vector<double>& projected,
                                   const std::vector<size_t>& full_bytes,
                                   size_t leader);

class LosslessCandidateScreen {
 public:
  LosslessCandidateScreen(const CodecMetadata& metadata,
                          JxlEncoderChunkedFrameAdapter& input,
                          ThreadPool* pool, const LosslessProbeEncoder& encode,
                          const LosslessScan* scan = nullptr,
                          size_t context_side = 0, bool full_width_strips = false);
  ~LosslessCandidateScreen();
  LosslessCandidateScreen(const LosslessCandidateScreen&) = delete;
  LosslessCandidateScreen& operator=(const LosslessCandidateScreen&) = delete;

  bool Available() const;
  // JXL_LOSSLESS_PROBE_TRAINING optionally caps only sampled MA learning.
  // Scores then describe that proxy, not the unchanged full-frame plan.
  // JXL_LOSSLESS_PROBE_ATLAS=1 concatenates full-width source bands so sampled
  // regions share a model. No resampling; vertical seams remain an estimate.
  StatusOr<std::vector<size_t>> Score(const CompressParams& params);
  // One unchanged region is a work quantum. Batch required prefix regions,
  // then shortlist before assigning the rest; partial scores never rank as totals.
  // Fast ranking uses disposable proxy parameters, then scores every surviving
  // candidate on all regions before returning its total (subject to the
  // explicit sample-training proxy above). Partial totals are never returned.
  StatusOr<std::vector<LosslessProbeCandidate>> Rank(
      std::vector<LosslessProbeCandidate> candidates,
      const CompressParams& incumbent, bool fast_first_round = false);
  // Check group geometry for close raw-family finalists before full encoding.
  // These remain crop estimates; callers must calibrate whole-frame geometry.
  StatusOr<std::vector<LosslessProbeCandidate>> SmallerGroups(
      const std::vector<LosslessProbeCandidate>& ranked);
  size_t Encodes() const;
  size_t SamplePixels() const;
  size_t Batches() const;
  StatusOr<LosslessProbeResult> Analyze(const CompressParams& params,
                                       const CompressParams& incumbent);

  // Preserve candidates with a meaningful local win when aggregate estimates
  // disagree; crops are evidence, not a proof of whole-frame dominance.
  static bool Promising(const std::vector<size_t>& candidate,
                        const std::vector<size_t>& incumbent);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

std::vector<LosslessProbeCandidate> LosslessProbePlan(const CompressParams& params);

// Visit distinct representations before near-duplicate training budgets.
// This changes a finite search budget's priorities, not an encoded-size bound.
std::vector<LosslessProbeCandidate> DiverseLosslessFrontier(
    std::vector<LosslessProbeCandidate> ranked);

// Uses unchanged, spatially distributed source rectangles, never a thumbnail.
// Screen a broad plan on representative tiles, then validate the shortlist on
// every sampled position. Estimates cannot replace a completed full encode.
// The optional incumbent calibrates scores to the actual whole-frame winner.
StatusOr<LosslessProbeResult> ProbeLosslessCandidates(
    const CompressParams& params, const CodecMetadata& metadata,
    JxlEncoderChunkedFrameAdapter& input, ThreadPool* pool,
    const LosslessProbeEncoder& encode,
    const CompressParams* incumbent = nullptr,
    bool feedback_bounds = false);

// Larger, independent source rectangles check a shortlisted candidate against
// the actual incumbent's parameters before granting it a full-frame encode.
StatusOr<LosslessProbeConfirmation> ConfirmLosslessCandidate(
    const CompressParams& candidate, const CompressParams& incumbent,
    const CodecMetadata& metadata, JxlEncoderChunkedFrameAdapter& input,
    ThreadPool* pool, const LosslessProbeEncoder& encode,
    size_t known_incumbent_bytes = 0);

}  // namespace jxl

#endif  // LIB_JXL_ENC_LOSSLESS_PROBE_H_
