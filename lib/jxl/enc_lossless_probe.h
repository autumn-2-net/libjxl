// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_LOSSLESS_PROBE_H_
#define LIB_JXL_ENC_LOSSLESS_PROBE_H_

#include <cstddef>
#include <functional>
#include <vector>

#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/enc_params.h"

namespace jxl {

struct CodecMetadata;
class JxlEncoderChunkedFrameAdapter;

// Encodes an independent sample frame without recursively invoking the search.
// The callback must not use the caller's runner: probes already share it.
using LosslessProbeEncoder =
    std::function<StatusOr<size_t>(const CompressParams&, const CodecMetadata&,
                                   JxlEncoderChunkedFrameAdapter&)>;

struct LosslessProbeCandidate {
  const char* name;
  CompressParams params;
  size_t bytes = 0;
};

struct LosslessProbeResult {
  // Stable ascending order by sample bytes; controls are not returned.
  std::vector<LosslessProbeCandidate> ranked;
  size_t samples = 0;
  size_t encodes = 0;
  size_t control_bytes = 0;
};

struct LosslessProbeConfirmation {
  size_t samples = 0;
  size_t candidate_bytes = 0;
  size_t incumbent_bytes = 0;
  bool accepted = true;
};

// Uses unchanged, spatially distributed source rectangles, never a thumbnail.
// Estimates only rank additional candidates; they cannot replace a full encode.
StatusOr<LosslessProbeResult> ProbeLosslessCandidates(
    const CompressParams& params, const CodecMetadata& metadata,
    JxlEncoderChunkedFrameAdapter& input, ThreadPool* pool,
    const LosslessProbeEncoder& encode);

// Larger, independent source rectangles check a shortlisted candidate against
// the actual incumbent's parameters before granting it a full-frame encode.
StatusOr<LosslessProbeConfirmation> ConfirmLosslessCandidate(
    const CompressParams& candidate, const CompressParams& incumbent,
    const CodecMetadata& metadata, JxlEncoderChunkedFrameAdapter& input,
    ThreadPool* pool, const LosslessProbeEncoder& encode,
    size_t known_incumbent_bytes = 0);

}  // namespace jxl

#endif  // LIB_JXL_ENC_LOSSLESS_PROBE_H_
