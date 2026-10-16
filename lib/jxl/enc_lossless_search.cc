// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_lossless_search.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_frame.h"
#include "lib/jxl/encode_internal.h"

namespace jxl {

int LosslessSearchLevel(const CompressParams& params) {
  return 10 - static_cast<int>(params.speed_tier);
}

bool HasLosslessSearchOptions(const CompressParams& p) {
  const int level = LosslessSearchLevel(p);
  // Before queueing, -1 is still auto. With buffering <= 0 it resolves to 0.
  return level >= 1 && level <= 10 && p.output_mode <= 0 && p.buffering <= 0 &&
         p.use_full_image_heuristics && p.decoding_speed_tier == 0 &&
         p.responsive <= 0 && p.progressive_mode != Override::kOn &&
         p.qprogressive_mode != Override::kOn && p.progressive_dc <= 0 &&
         !p.custom_progressive_mode && p.resampling <= 1 &&
         p.ec_resampling <= 1 && !p.lossy_palette && !p.max_error_mode &&
         p.modular_group_size_shift == -1 && p.colorspace == -1 &&
         p.palette_colors == 1024 && p.palette_colors_is_default &&
         p.channel_colors_percent == 80.f &&
         p.channel_colors_percent_is_default &&
         p.channel_colors_pre_transform_percent == 95.f &&
         p.patches == Override::kDefault &&
         p.options.predictor == kUndefinedPredictor &&
         p.options.nb_repeats == 0.5f &&
         p.ma_tree_learning_percent_is_default &&
         p.options.max_properties == 0 &&
         p.options.wp_tree_mode == ModularOptions::TreeMode::kDefault &&
         !p.options.zero_tokens && p.custom_fixed_tree.empty() &&
         !p.custom_splines.HasAny() && p.debug_image == nullptr &&
         p.noise != Override::kOn && p.dots != Override::kOn &&
         p.photon_noise_iso == 0 && p.manual_noise.empty() &&
         p.manual_xyb_factors.empty();
}

namespace {

constexpr size_t kUnknownSize = std::numeric_limits<size_t>::max();

struct Trial {
  int level;
  CompressParams params;
  size_t size = kUnknownSize;
  bool baseline = true;
};

// Statistics only decide whether to run a cheap probe. Actual encoded sizes
// decide whether its family merits refinement; neither can evict the winner.
StatusOr<bool> TryLiteralProbe(JxlEncoderChunkedFrameAdapter& input) {
  if (uint64_t{input.xsize} * input.ysize <= 65536) return true;
  auto source = input.GetInputSource();
  JxlPixelFormat format{};
  source.get_color_channels_pixel_format(source.opaque, &format);
  const size_t pixel_bytes =
      format.num_channels * (BitsPerChannel(format.data_type) / kBitsPerByte);
  size_t stride;
  auto buffer = GetColorBuffer(source, 0, 0, input.xsize, input.ysize, &stride);
  if (!buffer) return JXL_FAILURE("Missing lossless search input");
  const auto* data = static_cast<const uint8_t*>(buffer.get());
  constexpr size_t kSide = 32;
  std::array<uint64_t, kSide * kSide> colors;
  size_t flat = 0;
  for (size_t y = 0; y < kSide; ++y) {
    const size_t sy = uint64_t{y} * (input.ysize - 1) / (kSide - 1);
    for (size_t x = 0; x < kSide; ++x) {
      const size_t sx = uint64_t{x} * (input.xsize - 1) / (kSide - 1);
      const uint8_t* p = data + sy * stride + sx * pixel_bytes;
      uint64_t hash = 14695981039346656037ULL;
      for (size_t c = 0; c < pixel_bytes; ++c) {
        hash = (hash ^ p[c]) * 1099511628211ULL;
      }
      colors[y * kSide + x] = hash;
      const size_t dx = sx + 1 < input.xsize ? pixel_bytes : 0;
      const size_t dy = sy + 1 < input.ysize ? stride : 0;
      flat += std::memcmp(p, p + dx, pixel_bytes) == 0 &&
              std::memcmp(p, p + dy, pixel_bytes) == 0;
    }
  }
  std::sort(colors.begin(), colors.end());
  const size_t unique =
      std::unique(colors.begin(), colors.end()) - colors.begin();
  return flat >= colors.size() / 8 || unique <= colors.size() / 4;
}

CompressParams Backend(const CompressParams& input, int effort) {
  CompressParams p = input;
  p.speed_tier = static_cast<SpeedTier>(10 - effort);
  return p;
}

CompressParams Literal(const CompressParams& input, int effort,
                       float fraction) {
  CompressParams p = Backend(input, effort);
  p.modular_group_size_shift = 3;
  p.options.predictor = Predictor::Zero;
  p.options.nb_repeats = fraction;
  p.palette_colors = 10000;
  p.patches = Override::kOff;
  p.channel_colors_percent = 0;
  return p;
}

class Search {
 public:
  Search(JxlMemoryManager* memory_manager, ThreadPool* pool, bool stats,
         bool trace, const LosslessCandidateEncoder& encode)
      : memory_manager_(memory_manager),
        pool_(pool),
        stats_(stats),
        trace_(trace),
        encode_(encode) {}

  void Add(int level, CompressParams params, bool baseline = true) {
    trials_.push_back({level, std::move(params), kUnknownSize, baseline});
  }

  size_t BestBefore(int level, bool baseline_only = false) const {
    size_t size = kUnknownSize;
    for (const auto& trial : trials_) {
      if (trial.level < level && (!baseline_only || trial.baseline)) {
        size = std::min(size, trial.size);
      }
    }
    return size;
  }

  bool Competitive(size_t size, int level, double tolerance) const {
    // New probes must not suppress a previously retained baseline candidate.
    return size != kUnknownSize &&
           static_cast<double>(size) <= BestBefore(level, true) * tolerance;
  }

  size_t BestIndexBefore(int level) const {
    size_t best = kUnknownSize;
    for (size_t i = 0; i < trials_.size(); ++i) {
      if (trials_[i].level < level &&
          (best == kUnknownSize || trials_[i].size < trials_[best].size)) {
        best = i;
      }
    }
    JXL_DASSERT(best != kUnknownSize && trials_[best].size != kUnknownSize);
    return best;
  }

  CompressParams BestParamsBefore(int level) const {
    return trials_[BestIndexBefore(level)].params;
  }

  // The caller chooses either candidate parallelism or data parallelism. The
  // supplied runner is never entered recursively or from overlapping calls.
  Status RunPending(size_t concurrency) {
    while (next_ < trials_.size()) {
      const size_t count = std::min(concurrency, trials_.size() - next_);
      std::vector<std::unique_ptr<JxlEncoderOutputProcessorWrapper>> outputs(
          count);
      std::vector<AuxOut> stats(count);
      std::vector<uint8_t> discarded(count, 0);
      const auto run = [&](uint32_t index, size_t) -> Status {
        const auto start = std::chrono::steady_clock::now();
        const auto& trial = trials_[next_ + index];
        if (trace_) TraceTrial(trial, "start", 0, 0);
        outputs[index] =
            jxl::make_unique<JxlEncoderOutputProcessorWrapper>(memory_manager_);
        // Baseline sizes affect later family eligibility and must stay exact.
        // Additional candidates only matter if they strictly beat the winner.
        const size_t limit = !trial.baseline && trial.params.IsLossless() && best_
                                 ? best_->CurrentPosition()
                                 : kUnknownSize;
        LosslessTrialBudget budget(limit);
        auto params = trial.params;
        if (limit != kUnknownSize) params.lossless_trial_budget = &budget;
        JXL_RETURN_IF_ERROR(
            encode_(params, count == 1 ? pool_ : nullptr,
                    outputs[index].get(), stats_ ? &stats[index] : nullptr));
        discarded[index] = limit != kUnknownSize && budget.Reached();
        if (trace_) {
          TraceTrial(trial, discarded[index] ? "bounded" : "end",
                     discarded[index] ? limit : outputs[index]->CurrentPosition(),
                     std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - start)
                         .count());
        }
        return true;
      };
      if (count == 1) {
        JXL_RETURN_IF_ERROR(run(0, 0));
      } else {
        JXL_RETURN_IF_ERROR(RunOnPool(pool_, 0, count, ThreadPool::NoInit, run,
                                      "LosslessCandidates"));
      }
      // Stable plan order, not completion order, breaks equal-size ties.
      for (size_t i = 0; i < count; ++i) {
        if (discarded[i]) continue;
        trials_[next_ + i].size = outputs[i]->CurrentPosition();
        if (!best_ ||
            outputs[i]->CurrentPosition() < best_->CurrentPosition() ||
            (outputs[i]->CurrentPosition() == best_->CurrentPosition() &&
             trials_[next_ + i].level < best_level_)) {
          best_ = std::move(outputs[i]);
          best_stats_ = stats[i];
          best_level_ = trials_[next_ + i].level;
        }
      }
      next_ += count;
    }
    return true;
  }

  StatusOr<size_t> Refine(int level, CompressParams params,
                          bool baseline = true) {
    Add(level, std::move(params), baseline);
    JXL_RETURN_IF_ERROR(RunPending(1));
    return size_t{trials_.back().size};
  }

  size_t Size(size_t index) const { return trials_[index].size; }
  size_t Count() const { return trials_.size(); }

  static void TraceTrial(const Trial& trial, const char* event, size_t bytes,
                         double seconds) {
    const auto& p = trial.params;
    std::fprintf(stderr,
                 "lossless-trial event=%s stage=%d baseline=%d backend=%d g=%d "
                 "rct=%d predictor=%u ma=%.3f palette=%d prev=%d careful=%d "
                 "bytes=%" PRIuS " seconds=%.6f\n",
                 event, trial.level, trial.baseline, LosslessSearchLevel(p),
                 p.modular_group_size_shift, p.colorspace,
                 static_cast<unsigned>(p.options.predictor),
                 p.options.nb_repeats, p.palette_colors,
                 p.options.max_properties, p.lz77_careful_first_pass, bytes,
                 seconds);
    std::fflush(stderr);
  }

  Status Finish(JxlEncoderOutputProcessorWrapper* output, AuxOut* aux_out) {
    JXL_ENSURE(best_);
    JXL_RETURN_IF_ERROR(best_->SetFinalizedPosition());
    std::vector<uint8_t> bytes;
    JXL_RETURN_IF_ERROR(best_->CopyOutput(bytes));
    JXL_RETURN_IF_ERROR(AppendData(*output, bytes));
    if (aux_out) aux_out->Assimilate(best_stats_);
    return true;
  }

 private:
  JxlMemoryManager* memory_manager_;
  ThreadPool* pool_;
  bool stats_;
  bool trace_;
  const LosslessCandidateEncoder& encode_;
  std::vector<Trial> trials_;
  size_t next_ = 0;
  int best_level_ = 0;
  std::unique_ptr<JxlEncoderOutputProcessorWrapper> best_;
  AuxOut best_stats_;
};

}  // namespace

StatusOr<bool> EncodeLosslessSearch(
    JxlMemoryManager* memory_manager, const CompressParams& params,
    const FrameInfo& frame_info, const CodecMetadata& metadata,
    JxlEncoderChunkedFrameAdapter& frame_data, ThreadPool* pool,
    JxlEncoderOutputProcessorWrapper* output, AuxOut* aux_out,
    const LosslessCandidateEncoder& encode, const LosslessProbeEncoder& probe) {
  const bool jpeg = frame_data.IsJPEG();
  if ((!jpeg && (!params.IsLossless() ||
                 params.color_transform != ColorTransform::kNone)) ||
      !HasLosslessSearchOptions(params) || frame_data.StreamingInput() ||
      metadata.m.bit_depth.floating_point_sample ||
      metadata.m.bit_depth.bits_per_sample > 16 || frame_info.dc_level != 0 ||
      frame_info.is_preview) {
    return false;
  }
  for (const auto& ec : metadata.m.extra_channel_info) {
    if (ec.bit_depth.floating_point_sample || ec.bit_depth.bits_per_sample > 16)
      return false;
  }
  const int level = LosslessSearchLevel(params);
  const char* trace_value = std::getenv("JXL_LOSSLESS_SEARCH_TRACE");
  const bool trace = trace_value && std::strcmp(trace_value, "1") == 0;
  // Crop estimates can reject a whole-frame winner. Keep this cost/size tradeoff
  // experimental; the default preserves the unrestricted two-pass candidates.
  const char* confirm_value = std::getenv("JXL_LOSSLESS_SEARCH_CONFIRM");
  const bool use_confirmation =
      confirm_value && std::strcmp(confirm_value, "1") == 0;
  const uint64_t pixels = uint64_t{frame_data.xsize} * frame_data.ysize;
  Search search(memory_manager, pool, aux_out != nullptr, trace, encode);
  // Always retain a cheap valid result, then let competing families report
  // their actual sizes before granting deeper searches.
  search.Add(1, Backend(params, 3));
  size_t literal_index = kUnknownSize;
  if (!jpeg && level >= 2) {
    JXL_ASSIGN_OR_RETURN(bool probe, TryLiteralProbe(frame_data));
    if (probe) {
      literal_index = search.Count();
      search.Add(2, Literal(params, 6, 0));
    }
    if (pixels <= 4096) {
      for (Predictor predictor :
           {Predictor::Zero, Predictor::Left, Predictor::Top, Predictor::Select,
            Predictor::Gradient, Predictor::Weighted}) {
        auto p = Backend(params, 9);
        p.options.predictor = predictor;
        p.options.nb_repeats = 0;
        search.Add(2, std::move(p));
      }
    }
  }
  // A conservative concurrency heuristic, not a hard allocation limit. Large
  // frames retain data parallelism; small probes share the caller's runner.
  const uint64_t scratch_estimate =
      pixels * (5 + uint64_t{metadata.m.num_extra_channels}) * 32;
  const size_t concurrency =
      pixels <= 262144 && scratch_estimate > 0
          ? static_cast<size_t>(std::max<uint64_t>(
                1, std::min<uint64_t>(4, (256ULL << 20) / scratch_estimate)))
          : 1;
  JXL_RETURN_IF_ERROR(search.RunPending(concurrency));

  LosslessProbeResult analysis;
  if (!jpeg && level >= 8) {
    const auto start = std::chrono::steady_clock::now();
    if (trace) {
      std::fprintf(stderr, "lossless-probes event=start\n");
      std::fflush(stderr);
    }
    JXL_ASSIGN_OR_RETURN(
        analysis,
        ProbeLosslessCandidates(params, metadata, frame_data, pool, probe));
    if (trace) {
      std::fprintf(stderr,
                   "lossless-probes event=end samples=%" PRIuS
                   " encodes=%" PRIuS " control_bytes=%" PRIuS
                   " seconds=%.6f\n",
                   analysis.samples, analysis.encodes, analysis.control_bytes,
                   std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - start)
                       .count());
      for (const auto& candidate : analysis.ranked) {
        std::fprintf(stderr, "lossless-probe name=%s bytes=%" PRIuS "\n",
                     candidate.name, candidate.bytes);
      }
    }
  }
  size_t regular_size = search.Size(0);
  size_t literal_size =
      literal_index == kUnknownSize ? kUnknownSize : search.Size(literal_index);
  size_t learned_size = kUnknownSize;
  size_t refined_size = kUnknownSize;
  size_t confirmed_incumbent = kUnknownSize;
  size_t confirmed_bytes = 0;
  const auto confirm = [&](int stage,
                           const CompressParams& candidate) -> StatusOr<bool> {
    if (!use_confirmation) return true;
    const size_t incumbent = search.BestIndexBefore(stage);
    const auto start = std::chrono::steady_clock::now();
    if (trace) {
      std::fprintf(stderr, "lossless-confirm event=start stage=%d\n", stage);
      std::fflush(stderr);
    }
    JXL_ASSIGN_OR_RETURN(
        auto result,
        ConfirmLosslessCandidate(
            candidate, search.BestParamsBefore(stage), metadata, frame_data,
            pool, probe,
            incumbent == confirmed_incumbent ? confirmed_bytes : 0));
    confirmed_incumbent = incumbent;
    confirmed_bytes = result.incumbent_bytes;
    if (trace) {
      std::fprintf(stderr,
                   "lossless-confirm event=end stage=%d samples=%" PRIuS
                   " candidate_bytes=%" PRIuS " incumbent_bytes=%" PRIuS
                   " accepted=%d seconds=%.6f\n",
                   stage, result.samples, result.candidate_bytes,
                   result.incumbent_bytes, result.accepted,
                   std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - start)
                       .count());
      std::fflush(stderr);
    }
    return bool{result.accepted};
  };
  for (int stage = 3; stage <= level; ++stage) {
    // Every decision only sees lower-level results. A higher effort extends
    // the same policy instead of changing the smaller effort's decisions.
    int backend = 0;
    if (stage == 3) backend = jpeg ? 5 : 7;
    if (stage == 5) backend = jpeg ? 7 : 8;
    if (stage == 7) backend = jpeg ? 8 : 9;
    if (stage == 9 && jpeg) backend = 9;
    if (stage == 10) backend = 10;
    // A fixed-tree probe cannot predict the benefit of learning a tree. Always
    // sample that family once before deciding whether deeper training pays.
    if (backend &&
        (stage == 3 || search.Competitive(regular_size, stage, 1.35))) {
      JXL_ASSIGN_OR_RETURN(size_t size,
                           search.Refine(stage, Backend(params, backend)));
      regular_size = std::min(regular_size, size);
    }
    if (stage == 4 && search.Competitive(literal_size, stage, 1.35)) {
      JXL_ASSIGN_OR_RETURN(size_t size,
                           search.Refine(stage, Literal(params, 8, 0)));
      literal_size = std::min(literal_size, size);
    }
    if (stage == 6 && search.Competitive(literal_size, stage, 1.35)) {
      JXL_ASSIGN_OR_RETURN(learned_size,
                           search.Refine(stage, Literal(params, 9, 0.01f)));
    }
    if (stage == 8 && search.Competitive(learned_size, stage, 1.02)) {
      JXL_ASSIGN_OR_RETURN(refined_size,
                           search.Refine(stage, Literal(params, 9, 0.1f)));
    }
    if (stage == 9 && refined_size != kUnknownSize &&
        static_cast<double>(refined_size) < learned_size * 0.99) {
      // Full training needs an observed gain from raising the sample rate.
      JXL_ASSIGN_OR_RETURN(size_t full,
                           search.Refine(stage, Literal(params, 9, 1.0f)));
      (void)full;
    }
    if (stage >= 8 && static_cast<size_t>(stage - 8) < analysis.ranked.size()) {
      // A higher effort promotes the next candidate from the same fixed probe
      // ranking. The measured full-frame bytes, not the probe score, can win.
      const auto& candidate = analysis.ranked[stage - 8].params;
      JXL_ASSIGN_OR_RETURN(bool accepted, confirm(stage, candidate));
      if (accepted) {
        JXL_ASSIGN_OR_RETURN(size_t size,
                             search.Refine(stage, candidate, false));
        (void)size;
      }
    }
    if (stage == 10 && analysis.samples != 0) {
      // Refine the actual e9 winner, not whichever family won the sample test.
      auto deeper = search.BestParamsBefore(stage);
      deeper.speed_tier = SpeedTier::kGlacier;
      deeper.options.nb_repeats = std::max(0.5f, deeper.options.nb_repeats);
      deeper.options.max_properties =
          std::min<int>(4, metadata.m.num_extra_channels +
                               (metadata.m.color_encoding.IsGray() ? 0 : 2));
      deeper.modular_group_size_shift = 3;
      deeper.lz77_careful_first_pass = true;
      JXL_ASSIGN_OR_RETURN(bool accepted, confirm(stage, deeper));
      if (accepted) {
        JXL_ASSIGN_OR_RETURN(size_t size,
                             search.Refine(stage, std::move(deeper), false));
        (void)size;
      }
    }
  }
  JXL_RETURN_IF_ERROR(search.Finish(output, aux_out));
  return true;
}

}  // namespace jxl
