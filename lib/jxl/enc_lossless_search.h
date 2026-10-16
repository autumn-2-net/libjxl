// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_LOSSLESS_SEARCH_H_
#define LIB_JXL_ENC_LOSSLESS_SEARCH_H_

#include <jxl/memory_manager.h>

#include <atomic>
#include <cstddef>
#include <functional>

#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/enc_lossless_probe.h"
#include "lib/jxl/enc_params.h"

class JxlEncoderOutputProcessorWrapper;

namespace jxl {

struct AuxOut;
struct CodecMetadata;
struct FrameInfo;
class JxlEncoderChunkedFrameAdapter;

// Frame-local lower bound from finalized, disjoint group payloads. Saturating
// arithmetic also makes updates safe when several group encoders finish at once.
class LosslessTrialBudget {
 public:
  explicit LosslessTrialBudget(size_t limit) : limit_(limit) {}
  bool Reached() const {
    return bytes_.load(std::memory_order_relaxed) >= limit_;
  }
  void AddBytes(size_t bytes) {
    size_t old = bytes_.load(std::memory_order_relaxed);
    while (old < limit_) {
      const size_t next = bytes >= limit_ - old ? limit_ : old + bytes;
      if (bytes_.compare_exchange_weak(old, next, std::memory_order_relaxed))
        break;
    }
  }

 private:
  const size_t limit_;
  std::atomic<size_t> bytes_{0};
};

// The fork's cumulative effort levels, independent of backend speed tiers.
int LosslessSearchLevel(const CompressParams& params);
bool HasLosslessSearchOptions(const CompressParams& params);

using LosslessCandidateEncoder =
    std::function<Status(const CompressParams&, ThreadPool*,
                         JxlEncoderOutputProcessorWrapper*, AuxOut*)>;

// Returns false without writing when the input/options require the legacy path.
StatusOr<bool> EncodeLosslessSearch(
    JxlMemoryManager* memory_manager, const CompressParams& params,
    const FrameInfo& frame_info, const CodecMetadata& metadata,
    JxlEncoderChunkedFrameAdapter& frame_data, ThreadPool* pool,
    JxlEncoderOutputProcessorWrapper* output, AuxOut* aux_out,
    const LosslessCandidateEncoder& encode,
    const LosslessProbeEncoder& probe = {});

}  // namespace jxl

#endif  // LIB_JXL_ENC_LOSSLESS_SEARCH_H_
