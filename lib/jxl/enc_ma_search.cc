// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license.
#include "lib/jxl/enc_ma_search.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>

#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_direct_plan.h"
#include "lib/jxl/enc_frame.h"
#include "lib/jxl/enc_lossless_profile.h"
#include "lib/jxl/enc_tool_policy.h"
#include "lib/jxl/encode_internal.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/modular/encoding/enc_ma.h"

namespace jxl {
namespace {
using Clock = std::chrono::steady_clock;
std::string Env(const char* name) {
  const char* value = std::getenv(name);
  return value ? value : "";
}
bool Number(const char* name, size_t fallback, size_t maximum, size_t* value) {
  const auto text = Env(name);
  if (text.empty()) { *value = fallback; return true; }
  if (text.find_first_not_of("0123456789") != std::string::npos) return false;
  std::istringstream in(text);
  return bool(in >> *value) && *value <= maximum;
}
struct Settings {
  float fraction, threshold;
  size_t bins;
  bool expand_predictors;
};
Settings Decode(size_t action) {
  const size_t n = (action - 1) % 45;
  return {std::array<float, 3>{0.25f, 0.5f, 1.0f}[n / 15],
          std::array<float, 5>{0.25f, 0.5f, 1.0f, 2.0f, 4.0f}[(n / 3) % 5],
          std::array<size_t, 3>{64, 128, 256}[n % 3], action > 45};
}
void DescribeAction(size_t action, ToolFeatures* f) {
  (*f)[34] = action == 0;
  (*f)[35] = (*f)[36] = (*f)[37] = (*f)[38] = 0;
  if (action) {
    const auto s = Decode(action);
    (*f)[35] = s.fraction;
    (*f)[36] = std::log2(s.threshold);
    (*f)[37] = std::log2(double(s.bins));
    (*f)[38] = s.expand_predictors;
  }
}
}  // namespace

void ApplyMaSearchSettings(ModularOptions* options) {
  const auto* state = options->ma_search;
  if (!state || state->action == 0) return;
  const auto s = Decode(state->action);
  options->nb_repeats = s.fraction;
  options->splitting_heuristics_node_threshold *= s.threshold;
  options->max_property_values = s.bins;
  if (s.expand_predictors && options->predictor == Predictor::Best &&
      options->wp_tree_mode == ModularOptions::TreeMode::kDefault) {
    options->predictor = Predictor::Variable;
  }
  // Size-first candidate, without deleting WP/predictor/property choices.
  options->fast_decode_multiplier = 1.0f;
}

void MaSearchState::Gather(const TreeSamples& data,
                            const ModularOptions& options) {
  decltype(costs) local_costs{}, local_zeros{}, local_samples{};
  // Only the incumbent supplies model features. Candidate statistics must not
  // leak into inference features or incur another full statistics pass.
  for (size_t p = 0; action == 0 && p < data.NumPredictors(); ++p) {
    const size_t predictor = static_cast<size_t>(data.PredictorFromIndex(p));
    std::array<double, 256> histogram{};
    double count = 0, extra_bits = 0;
    for (size_t i = 0; i < data.NumDistinctSamples(); ++i) {
      const double weight = data.Count(i);
      const auto token = data.RToken(p, i);
      histogram[token.tok] += weight;
      extra_bits += weight * token.nbits;
      count += weight;
    }
    if (count == 0) continue;
    double bits = extra_bits;
    for (double n : histogram) if (n) bits += n * std::log2(count / n);
    local_costs[predictor] += bits;
    local_zeros[predictor] += histogram[0];
    local_samples[predictor] += count;
  }
  // Tree workers scan independently; only the small reduction is serialized.
  std::lock_guard<std::mutex> lock(mutex);
  ++trees;
  samples += data.NumSamples();
  distinct += data.NumDistinctSamples();
  predictors += data.NumPredictors();
  properties += data.NumProperties();
  fraction_sum += options.nb_repeats;
  threshold_sum += options.splitting_heuristics_node_threshold;
  bins_sum += options.max_property_values;
  for (size_t p = 0; p < costs.size(); ++p) {
    costs[p] += local_costs[p];
    zero_counts[p] += local_zeros[p];
    predictor_samples[p] += local_samples[p];
  }
}

void MaSearchState::FinishTree(size_t size) {
  std::lock_guard<std::mutex> lock(mutex);
  nodes += size;
}

void MaSearchState::Describe(ToolFeatures* f) const {
  // Called after the synchronous candidate encode has joined all workers.
  (*f)[7] = std::log1p(double(samples));
  (*f)[8] = double(distinct) / std::max<size_t>(1, samples);
  (*f)[9] = double(properties) / std::max<size_t>(1, trees);
  (*f)[10] = double(predictors) / std::max<size_t>(1, trees);
  (*f)[11] = std::log1p(double(nodes));
  for (size_t p = 0; p < costs.size(); ++p) {
    (*f)[12 + p] = costs[p] / std::max(1.0, predictor_samples[p]);
  }
  for (size_t p = 0; p < 7; ++p) {
    (*f)[26 + p] = zero_counts[p] / std::max(1.0, predictor_samples[p]);
  }
  (*f)[33] = std::log1p(double(trees));
}

bool MaSearchRequested() {
  const auto mode = Env("JXL_MA_SEARCH");
  return !mode.empty() && mode != "off";
}

Status EncodeMaSearch(JxlMemoryManager* memory_manager,
                      const CompressParams& params, const FrameInfo& frame_info,
                      const CodecMetadata& metadata,
                      JxlEncoderChunkedFrameAdapter& input, ThreadPool* pool,
                      JxlEncoderOutputProcessorWrapper* output, AuxOut* aux,
                      const LosslessCandidateEncoder& encode) {
  const auto mode = Env("JXL_MA_SEARCH");
  LosslessProfileScope search_profile(LosslessProfileCurrent(), "ma.search",
                                      LosslessProfileRequested());
  JXL_ENSURE(mode == "baseline" || mode == "collect" || mode == "action" || mode == "pair" ||
             mode == "model" || mode == "random");
  JXL_ENSURE(!DirectPlanRequested() && !LosslessToolPolicyEnabled());
  // Version one is a still-image, nonstreaming effort-9 experiment. Reject
  // incompatible requests instead of silently changing unrelated tools.
  if (metadata.m.have_animation || frame_info.is_preview ||
      !frame_info.is_last || frame_info.blend || frame_info.dc_level != 0 ||
      frame_info.frame_type != FrameType::kRegularFrame) {
    std::fprintf(stderr, "MA search requires a standalone still frame "
                         "(animation=%d, preview=%d, last=%d, blend=%d, reference=%d, dc=%d, type=%d)\n",
                 metadata.m.have_animation, frame_info.is_preview, frame_info.is_last,
                 frame_info.blend, static_cast<int>(frame_info.save_as_reference),
                 static_cast<int>(frame_info.dc_level), static_cast<int>(frame_info.frame_type));
    return JXL_FAILURE("Unsupported MA search frame");
  }
  if (!(params.speed_tier == SpeedTier::kTortoise &&
             params.decoding_speed_tier == 0 && params.buffering == 0 &&
             params.output_mode == 0 &&
             params.progressive_dc <= 0 && params.custom_fixed_tree.empty())) {
    std::fprintf(stderr, "MA search requires effort=9, decoding_speed=0, buffering=0, "
                         "output_mode=0, no progressive DC/custom tree "
                         "(tier=%d, buffering=%d, output=%d, dc=%d)\n",
                 static_cast<int>(params.speed_tier), params.buffering,
                 params.output_mode, params.progressive_dc);
    return JXL_FAILURE("Unsupported MA search settings");
  }
  size_t trials, seed, action;
  JXL_ENSURE(Number("JXL_MA_TRIALS", 12, kMaSearchActions - 1, &trials));
  JXL_ENSURE(Number("JXL_MA_SEED", 17, 1000000000, &seed));
  JXL_ENSURE(Number("JXL_MA_ACTION", 0, kMaSearchActions - 1, &action));
  JXL_ENSURE(!input.IsJPEG() || action <= 45);
  ToolFeatures features{};
  features[0] = input.IsJPEG() ? 2 : 1;
  features[1] = std::log1p(double(input.xsize) * input.ysize);
  features[2] = std::log1p(double(input.xsize));
  features[3] = std::log1p(double(input.ysize));
  features[4] = metadata.m.bit_depth.bits_per_sample;
  features[5] = metadata.m.num_extra_channels;
  EntropyToolModel model;
  if (mode == "model" || (mode == "random" && !Env("JXL_MA_MODEL").empty())) {
    std::ifstream file(Env("JXL_MA_MODEL"));
    JXL_ENSURE(file && model.Read(file, "JXL_MA_SEARCH_V1", kMaSearchActions));
    JXL_ENSURE(model.domain() == (input.IsJPEG() ? 2 : 1));
    for (size_t a : model.actions()) JXL_ENSURE(!input.IsJPEG() || a <= 45);
  }
  std::vector<size_t> actions{mode == "action" ? action : 0};
  if (mode == "pair" && action != 0) actions.push_back(action);
  if (mode == "collect" || mode == "random") {
    std::vector<size_t> candidates(input.IsJPEG() ? 45 : kMaSearchActions - 1);
    std::iota(candidates.begin(), candidates.end(), 1);
    if (mode == "random" && !model.actions().empty()) {
      candidates = model.actions();
      candidates.erase(std::remove(candidates.begin(), candidates.end(), 0), candidates.end());
    }
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::shuffle(candidates.begin(), candidates.end(), rng);
    // Shared warm starts provide comparable labels, not fixed MA trees.
    if (mode == "collect") {
      const std::vector<size_t> warm = input.IsJPEG()
          ? std::vector<size_t>{21, 36, 26, 13}
          : std::vector<size_t>{21, 36, 26, 13, 66, 81, 71, 58};
      for (size_t a : warm) {
        candidates.erase(std::find(candidates.begin(), candidates.end(), a));
      }
      candidates.insert(candidates.begin(), warm.begin(), warm.end());
    }
    actions.insert(actions.end(), candidates.begin(), candidates.begin() + std::min(trials, candidates.size()));
  }
  std::ofstream log;
  if (!Env("JXL_MA_LOG").empty()) {
    log.open(Env("JXL_MA_LOG"), std::ios::app);
    JXL_ENSURE(log.good());
    log.imbue(std::locale::classic());
    log << std::setprecision(17);
  }
  size_t best_bytes = std::numeric_limits<size_t>::max();
  std::unique_ptr<JxlEncoderOutputProcessorWrapper> winner;
  AuxOut winner_aux;
  for (size_t index = 0; index < actions.size(); ++index) {
    const size_t candidate = actions[index];
    MaSearchState state(candidate);
    auto p = params;
    p.options.ma_search = &state;
    LosslessProfileScope candidate_profile(search_profile.Id(), "ma.candidate");
    candidate_profile.Set("action", candidate);
    if (candidate_profile.Enabled()) {
      p.options.histogram_params.lossless_profile = candidate_profile.Id();
    }
    AuxOut stats;
    auto sink = std::make_unique<JxlEncoderOutputProcessorWrapper>(memory_manager);
    const auto start = Clock::now();
    JXL_RETURN_IF_ERROR(encode(p, pool, sink.get(), aux ? &stats : nullptr));
    const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
    const size_t bytes = sink->CurrentPosition();
    candidate_profile.Set("bytes", bytes);
    JXL_ENSURE(bytes > 0);
    if (candidate == 0) {
      features[6] = std::log1p(double(bytes));
      state.Describe(&features);
      if (mode == "model") {
        std::vector<std::pair<double, size_t>> ranked;
        for (size_t a : model.actions()) if (a != 0) {
          DescribeAction(a, &features);
          ranked.emplace_back(model.Predict(features)[0], a);
        }
        std::sort(ranked.begin(), ranked.end());
        for (size_t i = 0; i < std::min(trials, ranked.size()); ++i) {
          actions.push_back(ranked[i].second);
        }
      }
    }
    DescribeAction(candidate, &features);
    if (log.is_open()) {
      log << "M1\t" << candidate << '\t' << bytes << '\t' << seconds;
      for (double f : features) log << '\t' << f;
      log << '\t' << state.trees << '\t' << state.nodes << '\t' << state.samples
          << '\t' << state.predictors << '\t' << state.properties
          << '\t' << state.fraction_sum << '\t' << state.threshold_sum
          << '\t' << state.bins_sum << '\n';
      log.flush();
      JXL_ENSURE(log.good());
    }
    if (bytes < best_bytes) {
      best_bytes = bytes;
      winner = std::move(sink);
      winner_aux = std::move(stats);
    }
  }
  JXL_ENSURE(winner != nullptr);
  JXL_RETURN_IF_ERROR(winner->SetFinalizedPosition());
  std::vector<uint8_t> bytes;
  JXL_RETURN_IF_ERROR(winner->CopyOutput(bytes));
  JXL_RETURN_IF_ERROR(AppendData(*output, bytes));
  if (aux) aux->Assimilate(winner_aux);
  return true;
}
}  // namespace jxl
