// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license.
#include "lib/jxl/enc_direct_plan.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <string>

#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_neural_plan.h"
#include "lib/jxl/encode_internal.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/jpeg/jpeg_data.h"
#include "lib/jxl/modular/encoding/enc_encoding.h"

namespace jxl {
namespace {
using Clock = std::chrono::steady_clock;
constexpr std::array<size_t, 8> kRadix = {4, 7, 3, 2, 3, 3, 2, 3};

std::string Env(const char* key) {
  const char* value = std::getenv(key);
  return value ? value : "";
}

bool Number(const char* key, size_t fallback, size_t limit, size_t* result) {
  const auto s = Env(key);
  if (s.empty()) { *result = fallback; return true; }
  if (s.find_first_not_of("0123456789") != std::string::npos) return false;
  std::istringstream in(s);
  return bool(in >> *result) && *result <= limit;
}

std::array<size_t, 8> Axes(size_t action) {
  std::array<size_t, 8> axes{};
  for (size_t i = axes.size(); i-- > 0;) {
    axes[i] = action % kRadix[i];
    action /= kRadix[i];
  }
  return axes;
}

void DescribePlan(size_t action, ToolFeatures* f) {
  std::fill(f->begin() + 12, f->end(), 0);
  auto axes = Axes(action);
  size_t offset = 12;
  for (size_t i = 0; i < axes.size(); ++i) {
    (*f)[offset + axes[i]] = 1;
    offset += kRadix[i];
  }
}

Status DescribeInput(JxlEncoderChunkedFrameAdapter& input,
                     const CodecMetadata& metadata, ToolFeatures* f) {
  (*f)[0] = input.IsJPEG() ? 2 : 1;
  (*f)[1] = std::log1p(double(input.xsize) * input.ysize);
  (*f)[2] = std::log1p(double(input.xsize));
  (*f)[3] = std::log1p(double(input.ysize));
  (*f)[4] = metadata.m.bit_depth.bits_per_sample;
  (*f)[5] = metadata.m.num_extra_channels;
  size_t n = 0, zeros = 0, same = 0, small = 0, neighbours = 0;
  double sum = 0, delta = 0;
  std::array<size_t, 33> bins{};
  const auto sample = [&](int64_t value, int64_t left, bool has_left) {
    const double magnitude = std::abs(double(value));
    const double lv = std::log2(1 + magnitude);
    ++n;
    zeros += value == 0;
    small += magnitude <= 1;
    sum += lv;
    ++bins[std::min<size_t>(32, size_t(lv))];
    if (has_left) {
      ++neighbours;
      same += value == left;
      delta += std::log2(1 + std::abs(double(value) - left));
    }
  };
  if (input.IsJPEG()) {
    for (const auto& component : input.GetJPEGData()->components) {
      const auto& c = component.coeffs;
      const size_t stride = std::max<size_t>(1, (c.size() + 4095) / 4096);
      for (size_t block = 0; block * stride < c.size(); ++block) {
        const size_t i = std::min(c.size() - 1,
            block * stride + (block * 2654435761ULL + 17) % stride);
        sample(c[i], i ? c[i - 1] : 0, i != 0);
      }
    }
  } else {
    auto source = input.GetInputSource();
    JxlPixelFormat format{};
    source.get_color_channels_pixel_format(source.opaque, &format);
    JXL_ENSURE(format.data_type == JXL_TYPE_UINT8 ||
               format.data_type == JXL_TYPE_UINT16);
    const size_t bytes = format.data_type == JXL_TYPE_UINT8 ? 1 : 2;
    const uint16_t endian_test = 1;
    const bool little = format.endianness == JXL_LITTLE_ENDIAN ||
        (format.endianness == JXL_NATIVE_ENDIAN &&
         *reinterpret_cast<const uint8_t*>(&endian_test) == 1);
    const size_t w = std::min<size_t>(16, input.xsize);
    const size_t h = std::min<size_t>(16, input.ysize);
    // Sixteen stratified blocks, not a single thumbnail or periodic pixel grid.
    for (size_t by = 0; by < 4; ++by) for (size_t bx = 0; bx < 4; ++bx) {
      size_t stride = 0;
      auto buffer = GetColorBuffer(source, bx * (input.xsize - w) / 3,
          by * (input.ysize - h) / 3, w, h, &stride);
      JXL_ENSURE(buffer && stride >= w * format.num_channels * bytes);
      const auto* data = static_cast<const uint8_t*>(buffer.get());
      for (size_t y = 0; y < h; ++y) {
        for (size_t c = 0; c < format.num_channels; ++c) {
          int64_t prev = 0;
          for (size_t x = 0; x < w; ++x) {
            const auto* p = data + y * stride + (x * format.num_channels + c) * bytes;
            int64_t v = bytes == 1 ? p[0] :
                (little ? p[0] + 256 * p[1] : 256 * p[0] + p[1]);
            sample(v, prev, x != 0);
            prev = v;
          }
        }
      }
    }
  }
  const double count = std::max<size_t>(1, n);
  (*f)[6] = zeros / count;
  (*f)[7] = same / double(std::max<size_t>(1, neighbours));
  (*f)[8] = small / count;
  (*f)[9] = sum / count;
  (*f)[10] = delta / std::max<size_t>(1, neighbours);
  for (auto bin : bins) if (bin) {
    const double p = bin / count;
    (*f)[11] -= p * std::log2(p);
  }
  return true;
}

void ApplyPlan(size_t action, const CodecMetadata& metadata,
               size_t pixels, CompressParams* p, DirectAudit* audit) {
  const auto a = Axes(action);
  p->direct_plan = static_cast<int>(action);
  p->lossless_tool_domain = 0;
  p->speed_tier = SpeedTier::kSquirrel;  // Not searched; guarded branches below.
  p->decoding_speed_tier = 0;
  p->buffering = 0;
  p->output_mode = 0;
  p->responsive = 0;
  p->progressive_dc = 0;
  p->patches = Override::kOff;
  p->palette_colors = std::array<int, 3>{0, 256, 65536}[a[2]];
  p->channel_colors_percent = 0;
  p->channel_colors_pre_transform_percent = 0;
  p->lossy_palette = false;
  p->colorspace = std::array<int, 4>{0, 6, 9, 18}[a[0]];
  p->modular_group_size_shift = a[3] == 0 ? 1 : 3;
  p->force_cfl_jpeg_recompression = false;
  p->force_lfs_jpeg_recompression = 0;
  p->options.nb_repeats = 0;
  p->options.wp_mode = 0;
  p->options.predictor = std::array<Predictor, 7>{
      Predictor::Zero, Predictor::Left, Predictor::Top, Predictor::Gradient,
      Predictor::Weighted, Predictor::Gradient, Predictor::Weighted}[a[1]];
  if (a[1] < 5) {
    p->custom_fixed_tree = {PropertyDecisionNode::Leaf(p->options.predictor)};
  } else {
    p->custom_fixed_tree = PredefinedTree(
        a[1] == 5 ? ModularOptions::TreeKind::kGradientFixedDC :
                    ModularOptions::TreeKind::kWPFixedDC,
        pixels, metadata.m.bit_depth.bits_per_sample, 0);
  }
  auto& h = p->options.histogram_params;
  h.direct_audit = audit;
  h.direct_clusters = std::array<size_t, 3>{1, 4, 8}[a[7]];
  h.tool_policy_domain = 0;
  h.lz77_method = static_cast<HistogramParams::LZ77Method>(a[4]);
  h.uint_method = std::array<HistogramParams::HybridUintMethod, 3>{
      HistogramParams::HybridUintMethod::kNone,
      HistogramParams::HybridUintMethod::k000,
      HistogramParams::HybridUintMethod::kContextMap}[a[5]];
  h.ans_histogram_strategy = a[6] == 0 ?
      HistogramParams::ANSHistogramStrategy::kDirect0 :
      HistogramParams::ANSHistogramStrategy::kDirect6;
}
}  // namespace

bool DirectPlanRequested() {
  const auto mode = Env("JXL_DIRECT_POLICY");
  return !mode.empty() && mode != "off";
}

Status EncodeDirectPlan(JxlMemoryManager* memory_manager,
                        const CompressParams& params,
                        const CodecMetadata& metadata,
                        JxlEncoderChunkedFrameAdapter& input, ThreadPool* pool,
                        JxlEncoderOutputProcessorWrapper* output, AuxOut* aux,
                        const LosslessCandidateEncoder& encode) {
  const auto start = Clock::now();
  const auto mode = Env("JXL_DIRECT_POLICY");
  JXL_ENSURE(mode == "collect" || mode == "model" || mode == "fixed" ||
             mode == "random" || mode == "neural");
  JXL_ENSURE(!LosslessToolPolicyEnabled());
  ToolFeatures features{};
  JXL_RETURN_IF_ERROR(DescribeInput(input, metadata, &features));
  size_t action, seed, trials;
  JXL_ENSURE(Number("JXL_DIRECT_ACTION", 0, kDirectPlanCount - 1, &action));
  JXL_ENSURE(Number("JXL_DIRECT_SEED", 1, 1000000000, &seed));
  JXL_ENSURE(Number("JXL_DIRECT_TRIALS", 24, 256, &trials) && trials > 0);
  std::vector<size_t> actions;
  if (mode == "model" || mode == "random") {
    EntropyToolModel model;
    std::ifstream file(Env("JXL_DIRECT_MODEL"));
    JXL_ENSURE(file && model.Read(file, "JXL_DIRECT_PLAN_V1", kDirectPlanCount));
    JXL_ENSURE(model.domain() == (input.IsJPEG() ? 2 : 1));
    if (mode == "random") {
      std::mt19937 rng(static_cast<uint32_t>(seed));
      action = model.actions()[rng() % model.actions().size()];
    } else {
      double best = std::numeric_limits<double>::infinity();
      for (size_t candidate : model.actions()) {
        DescribePlan(candidate, &features);
        const auto prediction = model.Predict(features);
        if (prediction[0] < best) { best = prediction[0]; action = candidate; }
      }
    }
    actions.push_back(action);
  } else if (mode == "fixed" || mode == "neural") {
    actions.push_back(action);
  } else {
    actions.resize(kDirectPlanCount);
    std::iota(actions.begin(), actions.end(), 0);
    if (input.IsJPEG()) {
      actions.erase(std::remove_if(actions.begin(), actions.end(), [](size_t id) {
        const auto a = Axes(id);
        return a[0] != 0 || a[2] != 0 || a[3] != 0;
      }), actions.end());
    }
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::shuffle(actions.begin(), actions.end(), rng);
    actions.resize(std::min(trials, actions.size()));
  }
  const double selection_seconds =
      std::chrono::duration<double>(Clock::now() - start).count();
  std::ofstream log;
  if (!Env("JXL_DIRECT_LOG").empty()) {
    log.open(Env("JXL_DIRECT_LOG"), std::ios::app);
    JXL_ENSURE(log.good());
    log.imbue(std::locale::classic());
    log << std::setprecision(17);
  }
  size_t best_bytes = std::numeric_limits<size_t>::max();
  std::unique_ptr<JxlEncoderOutputProcessorWrapper> winner;
  AuxOut winner_aux;
  for (size_t candidate : actions) {
    if (mode == "collect") std::fprintf(stderr, "Direct trial %zu\n", candidate);
    DirectAudit audit;
    auto p = params;
    ApplyPlan(candidate, metadata, input.xsize * input.ysize, &p, &audit);
    if (mode == "neural") {
      JXL_RETURN_IF_ERROR(ReadNeuralPlan(Env("JXL_DIRECT_PLAN"), input.IsJPEG(), &p));
      if (Env("JXL_DIRECT_TREE_PROFILE") == "1") {
        // Matched teacher/student settings: only the teacher may learn a tree.
        p.speed_tier = SpeedTier::kTortoise;
        p.options.predictor = Predictor::Variable;
        p.options.max_properties = 2;
        p.options.nb_repeats = 0.5f;
        if (Env("JXL_DIRECT_TEACHER") == "1") {
          JXL_ENSURE(Env("JXL_DIRECT_TREE_IN").empty());
          audit.allow_ma_learning = true;
          p.custom_fixed_tree.clear();
        } else if (!Env("JXL_DIRECT_TREE_IN").empty()) {
          JXL_RETURN_IF_ERROR(ReadNeuralTreeFile(
              Env("JXL_DIRECT_TREE_IN").c_str(), &p.custom_fixed_tree));
        }
      } else {
        JXL_ENSURE(Env("JXL_DIRECT_TEACHER").empty() &&
                   Env("JXL_DIRECT_TREE_IN").empty());
      }
    }
    DescribePlan(candidate, &features);
    AuxOut candidate_aux;
    auto sink = std::make_unique<JxlEncoderOutputProcessorWrapper>(memory_manager);
    const auto encode_start = Clock::now();
    JXL_RETURN_IF_ERROR(encode(p, pool, sink.get(), aux ? &candidate_aux : nullptr));
    const double seconds =
        std::chrono::duration<double>(Clock::now() - encode_start).count();
    JXL_ENSURE(audit.frames == 1 && audit.trees == 1 &&
               audit.entropy_calls > 0 && audit.forbidden_searches == 0);
    const size_t bytes = sink->CurrentPosition();
    if (log) {
      const bool tree_trial = mode == "neural" && Env("JXL_DIRECT_TREE_PROFILE") == "1";
      log << (tree_trial ? "T1\t" : mode == "neural" ? "N1\t" : "D1\t")
          << (mode == "neural" ? audit.tree_nodes.load() : candidate)
          << '\t' << bytes << '\t' << seconds
          << '\t' << selection_seconds;
      const size_t feature_count = mode == "neural" ? 12 : features.size();
      for (size_t i = 0; i < feature_count; ++i) log << '\t' << features[i];
      log << '\t' << audit.frames << '\t' << audit.trees << '\t' << audit.entropy_calls
          << '\t' << audit.maps << '\t' << audit.palette_attempts
          << '\t' << audit.palette_applied << '\t' << audit.histogram_calls
          << '\t' << audit.precision_trials << '\t' << audit.forbidden_searches;
      if (tree_trial) log << '\t' << audit.ma_learning_calls;
      log << '\n';
      log.flush();
      JXL_ENSURE(log.good());
    }
    if (bytes < best_bytes) {
      best_bytes = bytes;
      winner = std::move(sink);
      winner_aux = std::move(candidate_aux);
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
