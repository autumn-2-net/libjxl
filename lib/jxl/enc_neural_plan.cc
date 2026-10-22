// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license.
#include "lib/jxl/enc_neural_plan.h"

#include <array>
#include <fstream>
#include <limits>
#include <locale>
#include <sstream>
#include <vector>

namespace jxl {
Status ReadNeuralTreeFile(const char* path, Tree* tree) {
  std::ifstream in(path);
  in.imbue(std::locale::classic());
  std::string magic;
  size_t count;
  JXL_ENSURE(in >> magic >> count);
  JXL_ENSURE(magic == "JXL_MA_TREE_V1" && count > 0 && count <= 65535 && count % 2);
  using Ranges = std::array<std::array<int64_t, 2>, 256>;
  Ranges initial;
  for (auto& r : initial) r = {{INT32_MIN, INT32_MAX}};
  std::vector<Ranges> ranges(1, initial);
  tree->clear();
  for (size_t i = 0; i < count; ++i) {
    JXL_ENSURE(i < ranges.size());
    int property, predictor;
    int64_t value;
    uint64_t multiplier;
    JXL_ENSURE(in >> property >> value >> predictor >> multiplier);
    if (property == -1) {
      JXL_ENSURE(predictor >= 0 && predictor < 14 &&
                 value >= INT32_MIN && value <= INT32_MAX &&
                 multiplier > 0 && multiplier <= UINT32_MAX);
      tree->push_back(PropertyDecisionNode::Leaf(
          static_cast<Predictor>(predictor), value, multiplier));
    } else {
      JXL_ENSURE(property >= 0 && property < 256 && predictor == 0 && multiplier == 1 &&
                 ranges.size() + 2 <= count);
      auto left = ranges[i], right = ranges[i];
      JXL_ENSURE(left[property][0] <= value && value < left[property][1]);
      tree->push_back(PropertyDecisionNode::Split(property, value, ranges.size()));
      left[property][0] = value + 1;
      right[property][1] = value;
      ranges.push_back(left);
      ranges.push_back(right);
    }
  }
  JXL_ENSURE(ranges.size() == count);
  in >> std::ws;
  JXL_ENSURE(in.eof());
  return true;
}

Status WriteNeuralTreeFile(const char* path, const Tree& tree) {
  JXL_ENSURE(!tree.empty() && tree.size() <= 65535);
  std::ofstream out(path, std::ios::trunc);
  out.imbue(std::locale::classic());
  JXL_ENSURE(out.good());
  out << "JXL_MA_TREE_V1 " << tree.size() << '\n';
  std::vector<size_t> queue{0};
  std::vector<bool> seen(tree.size());
  for (size_t i = 0; i < queue.size(); ++i) {
    const size_t index = queue[i];
    JXL_ENSURE(index < tree.size() && !seen[index]);
    seen[index] = true;
    const auto& node = tree[index];
    if (node.property == -1) {
      out << -1 << ' ' << node.predictor_offset << ' '
          << static_cast<int>(node.predictor) << ' ' << node.multiplier << '\n';
    } else {
      out << node.property << ' ' << node.splitval << " 0 1\n";
      queue.push_back(node.lchild);
      queue.push_back(node.rchild);
    }
  }
  JXL_ENSURE(queue.size() == tree.size());
  out.close();
  JXL_ENSURE(!out.fail());
  return true;
}

Status ReadNeuralPlan(const std::string& text, bool jpeg, CompressParams* p) {
  JXL_ENSURE(text.size() <= 32768);
  std::istringstream in(text);
  in.imbue(std::locale::classic());
  std::string magic;
  int domain, rct, palette, group, wp, lz, split, msb, lsb, precision;
  int clusters, prefix, nodes;
  JXL_ENSURE(in >> magic >> domain >> rct >> palette >> group >> wp >> lz >>
             split >> msb >> lsb >> precision >> clusters >> prefix >> nodes);
  JXL_ENSURE(magic == "JXL_NEURAL_PLAN_V1" && domain == (jpeg ? 2 : 1));
  JXL_ENSURE(rct >= 0 && rct < 42 && palette >= 0 && palette <= 65536 &&
             group >= 0 && group <= 3 && wp >= 0 && wp <= 4 &&
             lz >= 0 && lz <= 15 && split >= 0 && split <= 7 &&
             msb >= 0 && lsb >= 0 && msb + lsb <= split &&
             (precision == 0 || precision == 6) &&
             clusters >= 1 && clusters <= 256 && prefix >= 0 && prefix <= 1 &&
             nodes >= 1 && nodes <= 511 && nodes % 2 == 1);
  // These are pixel transforms, not JPEG transcode decisions.
  JXL_ENSURE(!jpeg || (rct == 0 && palette == 0 && group == 1));
  const uint32_t max_literal = (1u << split) +
      ((32u - split) << (msb + lsb)) - 1;
  const uint32_t literal_limit = lz != 0 ? (prefix ? 512 : 224) :
                                          (prefix ? 32768 : 256);
  JXL_ENSURE(max_literal < literal_limit);

  using Ranges = std::array<std::array<int64_t, 2>, 16>;
  Ranges initial;
  for (auto& range : initial) {
    range = {{std::numeric_limits<int32_t>::min(),
              std::numeric_limits<int32_t>::max()}};
  }
  std::vector<Ranges> ranges(1, initial);
  std::vector<size_t> depth(1, 0);
  Tree tree;
  tree.reserve(nodes);
  // Breadth-first child allocation makes cycles, shared children and detached
  // nodes impossible. Check split ranges exactly as the decoder does.
  for (int i = 0; i < nodes; ++i) {
    JXL_ENSURE(static_cast<size_t>(i) < ranges.size());
    int property, predictor;
    int64_t value;
    JXL_ENSURE(in >> property >> value >> predictor);
    if (property == -1) {
      JXL_ENSURE(predictor >= 0 && predictor < 14 &&
                 value >= -1048576 && value <= 1048576);
      tree.push_back(PropertyDecisionNode::Leaf(
          static_cast<Predictor>(predictor), value));
    } else {
      JXL_ENSURE(property >= 0 && property < 16 && predictor == 0 &&
                 depth[i] < 8 && ranges.size() + 2 <= static_cast<size_t>(nodes));
      const auto parent = ranges[i];
      JXL_ENSURE(parent[property][0] <= value && value < parent[property][1]);
      tree.push_back(PropertyDecisionNode::Split(property, static_cast<int>(value),
                                                 ranges.size()));
      auto left = parent;
      auto right = parent;
      left[property][0] = value + 1;
      right[property][1] = value;
      ranges.push_back(left);
      ranges.push_back(right);
      const size_t child_depth = depth[i] + 1;
      depth.push_back(child_depth);
      depth.push_back(child_depth);
    }
  }
  JXL_ENSURE(ranges.size() == static_cast<size_t>(nodes));
  in >> std::ws;
  JXL_ENSURE(in.eof());
  p->custom_fixed_tree = std::move(tree);
  p->colorspace = rct;
  p->palette_colors = palette;
  p->modular_group_size_shift = group;
  p->options.wp_mode = wp;
  auto& h = p->options.histogram_params;
  h.lz77_method = static_cast<HistogramParams::LZ77Method>(lz);
  h.direct_explicit_uint = true;
  h.direct_uint_config = HybridUintConfig(split, msb, lsb);
  h.direct_clusters = clusters;
  h.force_huffman = prefix != 0;
  h.lossless_lz77_cost_spans = true;
  h.ans_histogram_strategy = precision == 0 ?
      HistogramParams::ANSHistogramStrategy::kDirect0 :
      HistogramParams::ANSHistogramStrategy::kDirect6;
  return true;
}
}  // namespace jxl
