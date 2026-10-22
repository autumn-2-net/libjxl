// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
#include "lib/jxl/enc_tool_policy.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <utility>

#include "lib/jxl/enc_aux_out.h"

namespace jxl {

void EntropyTool::Apply(HistogramParams* p) const {
  p->lz77_method = static_cast<HistogramParams::LZ77Method>(lz77);
  p->clustering = static_cast<HistogramParams::ClusteringType>(clustering);
  p->uint_method = static_cast<HistogramParams::HybridUintMethod>(uint_method);
  p->ans_histogram_strategy =
      static_cast<HistogramParams::ANSHistogramStrategy>(histogram);
  p->lz77_careful_first_pass = false;
}

const std::vector<EntropyTool>& EntropyTools() {
  static const auto tools = [] {
    std::vector<EntropyTool> result;
    // None, RLE, hash-1, hash-7, optimal-1, optimal-8, optimal-256.
    for (int lz : {0, 1, 2, 4, 12, 14, 15}) {
      for (int cluster : {0, 1, 2}) {
        for (int uint_method : {0, 2, 4}) {
          for (int histogram : {0, 1, 2}) {
            result.push_back({lz, cluster, uint_method, histogram});
          }
        }
      }
    }
    return result;
  }();
  return tools;
}

ToolFeatures DescribeEntropyState(const HistogramParams& params, size_t contexts,
                                 const std::vector<std::vector<Token>>& tokens,
                                 LayerType layer) {
  ToolFeatures f{};
  size_t total = 0, longest = 0, width = 0;
  for (const auto& stream : tokens) {
    total += stream.size();
    longest = std::max(longest, stream.size());
  }
  for (size_t w : params.image_widths) width = std::max(width, w);
  // Stratified, deterministic jitter avoids aliasing periodic token patterns.
  // Compare real neighbours, not neighbours in the downsampled sequence.
  const size_t stride = std::max<size_t>(1, (total + 16383) / 16384);
  const auto probe = [stride](size_t block) {
    uint64_t hash = static_cast<uint64_t>(block) + 0x9e3779b97f4a7c15ULL;
    hash = (hash ^ (hash >> 30)) * 0xbf58476d1ce4e5b9ULL;
    hash = (hash ^ (hash >> 27)) * 0x94d049bb133111ebULL;
    hash ^= hash >> 31;
    return block * stride + static_cast<size_t>(hash % stride);
  };
  size_t sampled = 0, zero = 0, same = 0, small = 0, neighbours = 0;
  double sum_log = 0;
  std::array<size_t, 33> bins{};
  size_t offset = 0;
  size_t block = 0, next = probe(0);
  for (const auto& stream : tokens) {
    for (; next < offset + stream.size(); next = probe(++block)) {
      const size_t i = next - offset;
      const auto value = stream[i].value;
      const double lv = std::log2(1.0 + value);
      ++sampled;
      zero += value == 0;
      small += value <= 1;
      sum_log += lv;
      ++bins[std::min<size_t>(32, static_cast<size_t>(lv))];
      if (i != 0) {
        ++neighbours;
        same += value == stream[i - 1].value;
      }
    }
    offset += stream.size();
  }
  const double n = std::max<size_t>(1, sampled);
  double entropy = 0;
  for (size_t bin : bins) {
    if (bin != 0) {
      const double p = bin / n;
      entropy -= p * std::log2(p);
    }
  }
  f[0] = params.tool_policy_domain;
  f[1] = static_cast<int>(layer);
  f[2] = std::log1p(static_cast<double>(total));
  f[3] = std::log1p(static_cast<double>(contexts));
  f[4] = std::log1p(static_cast<double>(tokens.size()));
  f[5] = zero / n;
  f[6] = same / static_cast<double>(std::max<size_t>(1, neighbours));
  f[7] = small / n;
  f[8] = sum_log / n;
  f[9] = entropy;
  f[10] = std::log1p(static_cast<double>(width));
  f[11] = longest / static_cast<double>(std::max<size_t>(1, total));
  return f;
}

void DescribeEntropyTool(const EntropyTool& tool, ToolFeatures* f) {
  std::fill(f->begin() + kToolStateFeatures, f->end(), 0.0);
  (*f)[12 + tool.lz77] = 1;
  (*f)[28 + tool.clustering] = 1;
  (*f)[31 + tool.uint_method] = 1;
  (*f)[36 + tool.histogram] = 1;
}

bool EntropyToolModel::Read(std::istream& input, const char* expected_magic,
                            size_t action_limit) {
  input.imbue(std::locale::classic());
  std::string magic;
  size_t features, actions;
  int domain;
  if (!(input >> magic >> features >> domain >> actions) ||
      magic != expected_magic || features != kToolFeatures ||
      (domain != 1 && domain != 2) || actions == 0 ||
      actions > action_limit) return false;
  EntropyToolModel parsed;
  parsed.domain_ = domain;
  for (size_t i = 0; i < actions; ++i) {
    size_t action;
    if (!(input >> action) || action >= action_limit ||
        std::find(parsed.actions_.begin(), parsed.actions_.end(), action) !=
            parsed.actions_.end()) return false;
    parsed.actions_.push_back(action);
  }
  size_t total_nodes = 0;
  for (auto& forest : parsed.forests_) {
    size_t trees;
    if (!(input >> trees) || trees == 0 || trees > 256) return false;
    for (size_t t = 0; t < trees; ++t) {
      size_t count;
      if (!(input >> count) || count == 0 || count > 65535 ||
          (total_nodes += count) > 1000000) return false;
      Tree tree(count);
      for (size_t i = 0; i < count; ++i) {
        auto& n = tree[i];
        if (!(input >> n.feature >> n.threshold >> n.left >> n.right >> n.value) ||
            !std::isfinite(n.threshold) || !std::isfinite(n.value)) return false;
        if (n.feature == -1) {
          if (n.left != -1 || n.right != -1) return false;
        } else if (n.feature < 0 || n.feature >= static_cast<int>(features) ||
                   n.left <= static_cast<int>(i) || n.right <= static_cast<int>(i) ||
                   n.left >= static_cast<int>(count) ||
                   n.right >= static_cast<int>(count)) {
          return false;
        }
      }
      forest.push_back(std::move(tree));
    }
  }
  std::string trailing;
  if (input >> trailing) return false;
  *this = std::move(parsed);
  return true;
}

std::array<double, 2> EntropyToolModel::Predict(const ToolFeatures& f) const {
  std::array<double, 2> result{};
  for (size_t h = 0; h < 2; ++h) {
    for (const auto& tree : forests_[h]) {
      size_t i = 0;
      while (tree[i].feature != -1) {
        const auto& n = tree[i];
        // sklearn tree inference uses float32 inputs.
        i = static_cast<float>(f[n.feature]) <= n.threshold ? n.left : n.right;
      }
      result[h] += tree[i].value;
    }
    if (!forests_[h].empty()) result[h] /= forests_[h].size();
  }
  return result;
}

namespace {
using Clock = std::chrono::steady_clock;

double Seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

struct Settings {
  std::string mode, log, model;
  size_t trials = 8, seed = 1, min_tokens = 4096, max_tokens = 4000000;
  double budget = 0.25, time_weight = 0;
  bool valid = true;
};

const Settings& Config() {
  static const auto config = [] {
    Settings s;
    const auto value = [](const char* name) {
      const char* p = std::getenv(name);
      return p == nullptr ? std::string() : std::string(p);
    };
    s.mode = value("JXL_TOOL_POLICY");
    s.log = value("JXL_TOOL_LOG");
    s.model = value("JXL_TOOL_MODEL");
    const auto number = [&](const char* name, double fallback) {
      const auto text = value(name);
      if (text.empty()) return fallback;
      std::istringstream in(text);
      in.imbue(std::locale::classic());
      double n;
      std::string rest;
      if (!(in >> n) || !std::isfinite(n) || (in >> rest)) {
        s.valid = false;
        return fallback;
      }
      return n;
    };
    const auto integer = [&](const char* name, size_t fallback, size_t maximum) {
      const double n = number(name, static_cast<double>(fallback));
      if (n < 0 || n > maximum || std::floor(n) != n) {
        s.valid = false;
        return fallback;
      }
      return static_cast<size_t>(n);
    };
    s.trials = integer("JXL_TOOL_TRIALS", s.trials, EntropyTools().size());
    s.seed = integer("JXL_TOOL_SEED", s.seed, 0xFFFFFFFFu);
    s.min_tokens = integer("JXL_TOOL_MIN_TOKENS", s.min_tokens, 1000000000);
    s.max_tokens = integer("JXL_TOOL_MAX_TOKENS", s.max_tokens, 1000000000);
    s.budget = number("JXL_TOOL_BUDGET", s.budget);
    s.time_weight = number("JXL_TOOL_TIME_WEIGHT", s.time_weight);
    s.valid &= s.budget > 0 && s.time_weight >= 0 && s.trials > 0 &&
               s.max_tokens >= s.min_tokens;
    s.valid &= s.mode.empty() || s.mode == "off" || s.mode == "baseline" ||
               s.mode == "collect" || s.mode == "random" || s.mode == "model";
    return s;
  }();
  return config;
}

// One file per encoder process; the Python collector merges files afterwards.
bool Record(const std::string& row) {
  if (Config().log.empty()) return true;
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);
  std::ofstream file(Config().log, std::ios::app);
  file.imbue(std::locale::classic());
  file << row << '\n';
  file.flush();
  return static_cast<bool>(file);
}

double Score(double bits, double seconds) {
  // Size-first by default. A caller may apply a soft stage-time penalty; raw
  // measurements remain unchanged and can be rescored without re-encoding.
  const double over = std::max(0.0, seconds / Config().budget - 1.0);
  // log(bits/ref) differs only by a constant for the same decision. This makes
  // the size term relative without using raw bytes as an arbitrary denominator.
  return std::log(std::max(1.0, bits)) + Config().time_weight * over * over;
}

EntropyTool FromParams(const HistogramParams& p) {
  return {static_cast<int>(p.lz77_method), static_cast<int>(p.clustering),
          static_cast<int>(p.uint_method), static_cast<int>(p.ans_histogram_strategy)};
}

}  // namespace

bool LosslessToolPolicyEnabled() {
  return !Config().mode.empty() && Config().mode != "off";
}

StatusOr<size_t> EncodeWithEntropyTools(
    JxlMemoryManager* memory, const HistogramParams& params, size_t contexts,
    std::vector<std::vector<Token>>& tokens, EntropyEncodingData* codes,
    BitWriter* writer, LayerType layer, AuxOut* aux, const EntropyKernel& kernel) {
  if (!Config().valid) return JXL_FAILURE("invalid JXL_TOOL_* configuration");
  size_t count = 0;
  for (const auto& stream : tokens) count += stream.size();
  // Streaming code reuses entropy state and is deliberately outside v1.
  if (params.streaming_mode || !params.initialize_global_state ||
      params.add_fixed_histograms || params.add_missing_symbols || !writer ||
      !codes->encoding_info.empty() || count < Config().min_tokens ||
      (Config().mode == "collect" && count > Config().max_tokens)) {
    return kernel(params, tokens, codes, writer, aux);
  }
  static std::atomic<size_t> next_id{0};
  const size_t id = next_id.fetch_add(1);
  const auto selection_start = Clock::now();
  auto features = DescribeEntropyState(params, contexts, tokens, layer);
  const auto& tools = EntropyTools();
  std::vector<int> selected;
  std::mt19937 random(static_cast<uint32_t>(Config().seed + id));
  const EntropyToolModel* model = nullptr;
  if (Config().mode == "model" ||
      (Config().mode == "random" && !Config().model.empty())) {
    // Immutable model: shared safely by all encoder threads.
    struct Loaded {
      EntropyToolModel model;
      bool valid = false;
      Loaded() {
        std::ifstream file(Config().model);
        valid = file && model.Read(file);
      }
    };
    static const Loaded loaded;
    if (!loaded.valid || loaded.model.domain() != params.tool_policy_domain) {
      return JXL_FAILURE("missing, invalid, or wrong-domain entropy tool model");
    }
    model = &loaded.model;
  }
  if (Config().mode == "model") {
    double best = std::numeric_limits<double>::infinity();
    int best_action = -1;
    // Normalization is constant across actions for this same token buffer.
    for (size_t action : model->actions()) {
      DescribeEntropyTool(tools[action], &features);
      const auto p = model->Predict(features);
      const double bits = std::exp(std::clamp(p[0], -30.0, 30.0)) * count;
      const double seconds = std::exp(std::clamp(p[1], -30.0, 30.0));
      const double score = Score(bits, seconds);
      if (score < best) { best = score; best_action = static_cast<int>(action); }
    }
    selected.push_back(best_action);
  } else if (Config().mode == "random") {
    const size_t count = model ? model->actions().size() : tools.size();
    std::uniform_int_distribution<size_t> draw(0, count - 1);
    const size_t index = draw(random);
    selected.push_back(static_cast<int>(model ? model->actions()[index] : index));
  } else if (Config().mode == "baseline") {
    selected.push_back(-1);
  } else {
    // Traditional warm start: several different kernels, plus seeded random
    // combinations. Every candidate is evaluated on the SAME original tokens.
    selected.push_back(-1);
    std::vector<int> pool{0, 62, 116, 188};
    std::vector<int> rest(tools.size());
    std::iota(rest.begin(), rest.end(), 0);
    std::shuffle(rest.begin(), rest.end(), random);
    pool.insert(pool.end(), rest.begin(), rest.end());
    for (int a : pool) {
      if (selected.size() >= Config().trials + 1) break;
      if (std::find(selected.begin(), selected.end(), a) == selected.end()) {
        selected.push_back(a);
      }
    }
    // Avoid systematically timing the same tool after the same predecessor.
    std::shuffle(selected.begin(), selected.end(), random);
  }
  const double selection_seconds = Seconds(selection_start);
  // Normal inference runs exactly one kernel, without copying or dry-writing
  // token buffers. Collection deliberately incurs those costs to obtain labels.
  if (Config().mode != "collect") {
    auto chosen = params;
    if (selected[0] >= 0) tools[selected[0]].Apply(&chosen);
    std::ostringstream row;
    row.imbue(std::locale::classic());
    row << std::setprecision(17) << "C1\t" << id << '\t' << selected[0]
        << '\t' << selection_seconds << '\t' << static_cast<int>(params.tool_policy_domain);
    if (!Record(row.str())) return JXL_FAILURE("cannot write entropy tool log");
    return kernel(chosen, tokens, codes, writer, aux);
  }

  size_t best_cost = 0;
  double best_score = std::numeric_limits<double>::infinity();
  int best_action = -1;
  EntropyEncodingData best_codes;
  BitWriter best_header(memory);
  std::vector<std::vector<Token>> best_tokens;
  AuxOut best_aux;
  for (int action : selected) {
    auto p = params;
    const auto tool = action < 0 ? FromParams(params) : tools[action];
    if (action >= 0) tool.Apply(&p);
    auto trial_tokens = tokens;
    const auto start = Clock::now();
    EntropyEncodingData trial_codes;
    BitWriter header(memory);
    AuxOut trial_aux;
    JXL_ASSIGN_OR_RETURN(size_t cost, kernel(p, trial_tokens, &trial_codes,
                                          &header, &trial_aux));
    size_t bits = header.BitsWritten();
    for (const auto& stream : trial_tokens) {
      if (stream.empty()) continue;
      BitWriter body(memory);
      JXL_RETURN_IF_ERROR(WriteTokens(stream, trial_codes, 0, &body, layer, nullptr));
      bits += body.BitsWritten();
    }
    const double seconds = Seconds(start);
    DescribeEntropyTool(tool, &features);
    std::ostringstream row;
    row.imbue(std::locale::classic());
    row << std::setprecision(17) << "T1\t" << id << '\t' << action << '\t'
        << bits << '\t' << seconds;
    for (double feature : features) row << '\t' << feature;
    if (!Record(row.str())) return JXL_FAILURE("cannot write entropy tool log");
    const double score = Score(static_cast<double>(bits), seconds);
    if (score < best_score) {
      best_score = score;
      best_action = action;
      best_cost = cost;
      best_tokens = std::move(trial_tokens);
      best_codes = std::move(trial_codes);
      best_header = std::move(header);
      best_aux = trial_aux;
    }
  }
  JXL_RETURN_IF_ERROR(writer->AppendUnaligned(best_header));
  tokens = std::move(best_tokens);
  *codes = std::move(best_codes);
  if (aux != nullptr) aux->Assimilate(best_aux);
  std::ostringstream row;
  row.imbue(std::locale::classic());
  row << std::setprecision(17) << "C1\t" << id << '\t' << best_action << '\t'
      << selection_seconds << '\t' << static_cast<int>(params.tool_policy_domain);
  if (!Record(row.str())) return JXL_FAILURE("cannot write entropy tool log");
  return best_cost;
}

}  // namespace jxl
