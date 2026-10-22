// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
#ifndef LIB_JXL_ENC_TOOL_POLICY_H_
#define LIB_JXL_ENC_TOOL_POLICY_H_

#include <array>
#include <functional>
#include <istream>
#include <vector>

#include "lib/jxl/enc_ans.h"

namespace jxl {

// Versioned, encoder-only experimental entropy decision point. No CLI presets
// or effort values are actions. Python trains; this code makes the decision.
constexpr size_t kToolStateFeatures = 12;
constexpr size_t kToolFeatures = 39;
using ToolFeatures = std::array<double, kToolFeatures>;

struct EntropyTool {
  int lz77;
  int clustering;
  int uint_method;
  int histogram;
  void Apply(HistogramParams* params) const;
};

const std::vector<EntropyTool>& EntropyTools();
ToolFeatures DescribeEntropyState(const HistogramParams& params, size_t contexts,
                                 const std::vector<std::vector<Token>>& tokens,
                                 LayerType layer);
void DescribeEntropyTool(const EntropyTool& tool, ToolFeatures* features);

// A small, validated forest format, not executable Python/pickle. Both targets
// are learned separately: log(bits/token), log(seconds). No baked-in reward.
class EntropyToolModel {
 public:
  bool Read(std::istream& input, const char* magic = "JXL_ENTROPY_TOOLS_V1",
            size_t action_limit = 189);
  std::array<double, 2> Predict(const ToolFeatures& features) const;
  int domain() const { return domain_; }
  const std::vector<size_t>& actions() const { return actions_; }

 private:
  struct Node {
    int feature, left, right;
    double threshold, value;
  };
  using Tree = std::vector<Node>;
  std::array<std::vector<Tree>, 2> forests_;
  std::vector<size_t> actions_;
  int domain_ = 0;
};

bool LosslessToolPolicyEnabled();

using EntropyKernel = std::function<StatusOr<size_t>(
    const HistogramParams&, std::vector<std::vector<Token>>&, EntropyEncodingData*,
    BitWriter*, AuxOut*)>;

StatusOr<size_t> EncodeWithEntropyTools(
    JxlMemoryManager* memory_manager, const HistogramParams& params,
    size_t contexts, std::vector<std::vector<Token>>& tokens,
    EntropyEncodingData* codes, BitWriter* writer, LayerType layer, AuxOut* aux,
    const EntropyKernel& kernel);

}  // namespace jxl
#endif  // LIB_JXL_ENC_TOOL_POLICY_H_
