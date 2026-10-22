// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license.
#ifndef LIB_JXL_ENC_MA_SEARCH_H_
#define LIB_JXL_ENC_MA_SEARCH_H_

#include <array>
#include <mutex>

#include "lib/jxl/enc_lossless_search.h"

namespace jxl {
struct TreeSamples;

// Encoder-only experiment. An action tunes learning, never supplies a tree.
constexpr size_t kMaSearchActions = 91;
struct MaSearchState {
  explicit MaSearchState(size_t id) : action(id) {}
  const size_t action;
  std::mutex mutex;
  size_t trees = 0, nodes = 0, samples = 0, distinct = 0;
  size_t predictors = 0, properties = 0;
  double fraction_sum = 0, threshold_sum = 0, bins_sum = 0;
  std::array<double, 14> costs{}, zero_counts{}, predictor_samples{};
  void Gather(const TreeSamples& data, const ModularOptions& options);
  void FinishTree(size_t size);
  void Describe(std::array<double, 39>* features) const;
};

void ApplyMaSearchSettings(ModularOptions* options);
bool MaSearchRequested();
Status EncodeMaSearch(JxlMemoryManager* memory_manager,
                      const CompressParams& params, const FrameInfo& frame_info,
                      const CodecMetadata& metadata,
                      JxlEncoderChunkedFrameAdapter& input, ThreadPool* pool,
                      JxlEncoderOutputProcessorWrapper* output, AuxOut* aux,
                      const LosslessCandidateEncoder& encode);
}  // namespace jxl
#endif  // LIB_JXL_ENC_MA_SEARCH_H_
