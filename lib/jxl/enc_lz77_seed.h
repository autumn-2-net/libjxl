// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef LIB_JXL_ENC_LZ77_SEED_H_
#define LIB_JXL_ENC_LZ77_SEED_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/enc_ans_params.h"
#include "lib/jxl/enc_lossless_profile.h"
#include "lib/jxl/enc_lossless_workers.h"
#include "lib/jxl/enc_lz77_parse.h"

namespace jxl {

// Encoder-only block boundaries restrict the seed's destinations, not its
// dictionary. The immutable index is reused by the subsequent optimal parser.
template <typename Index, typename LiteralCost, typename CopyCost,
          typename DistanceCost>
Status LZ77IndexedSeed(const std::vector<Token>& input, const Index& index,
                       const HistogramParams& params, const LZ77Params& lz77,
                       const LiteralCost& literal_cost,
                       const CopyCost& copy_cost,
                       const DistanceCost& distance_cost,
                       std::vector<Token>* output, float* decrease,
                       float* raw_cost) {
  LosslessProfileScope profile(params.lossless_profile, "lz77.indexed_seed");
  const size_t tasks = LosslessWorkerLimits::FromEnvironment().Tasks(
      std::max<size_t>(1, input.size() / 65536), 4u << 20);
  struct Part {
    std::vector<Token> tokens;
    double saving = 0, raw = 0;
  };
  std::vector<Part> parts(tasks);
  JXL_RETURN_IF_ERROR(RunOnPool(
      params.lossless_lz77_pool, 0, tasks, ThreadPool::NoInit,
      [&](uint32_t task, size_t) -> Status {
        const size_t begin = input.size() * task / tasks;
        const size_t end = input.size() * (task + 1) / tasks;
        auto& part = parts[task];
        std::vector<double> prefix(end - begin + 1);
        for (size_t i = begin; i < end; ++i)
          prefix[i - begin + 1] = prefix[i - begin] + literal_cost(input[i]);
        part.raw = prefix.back();
        part.tokens.reserve(end - begin);
        for (size_t pos = begin; pos < end;) {
          size_t best_length = 0;
          uint32_t best_distance = 0;
          double best_gain = 0;
          index.FindMatches(pos, distance_cost,
              [&](size_t length, uint32_t distance) {
                const double gain = prefix[pos - begin + length] -
                    prefix[pos - begin] - copy_cost(input[pos], length, distance);
                if (gain > best_gain ||
                    (gain == best_gain && gain > 0 && length > best_length)) {
                  best_length = length;
                  best_distance = distance;
                  best_gain = gain;
                }
              }, nullptr, end - pos);
          if (best_length < lz77.min_length) {
            part.tokens.push_back(input[pos++]);
            continue;
          }
          part.tokens.emplace_back(input[pos].context,
                                   best_length - lz77.min_length);
          part.tokens.back().is_lz77_length = true;
          part.tokens.emplace_back(lz77.nonserialized_distance_context,
                                   best_distance);
          part.saving += best_gain;
          pos += best_length;
        }
        return true;
      }, "LosslessIndexedSeed"));
  size_t size = 0;
  double saving = 0, raw = 0;
  for (const auto& part : parts) {
    size += part.tokens.size();
    saving += part.saving;
    raw += part.raw;
  }
  output->clear();
  output->reserve(size);
  for (const auto& part : parts)
    output->insert(output->end(), part.tokens.begin(), part.tokens.end());
  *decrease = static_cast<float>(saving);
  *raw_cost = static_cast<float>(raw);
  profile.Set("tasks", tasks);
  profile.Set("tokens", input.size());
  profile.Set("output_tokens", output->size());
  return true;
}

}  // namespace jxl

#endif  // LIB_JXL_ENC_LZ77_SEED_H_
