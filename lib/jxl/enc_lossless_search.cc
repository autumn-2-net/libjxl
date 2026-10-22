// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_lossless_search.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_frame.h"
#include "lib/jxl/enc_final_plan.h"
#include "lib/jxl/enc_lossless_profile.h"
#include "lib/jxl/enc_lossless_probe_pool.h"
#include "lib/jxl/enc_lossless_racing.h"
#include "lib/jxl/enc_lossless_scan.h"
#include "lib/jxl/enc_lossless_workers.h"
#include "lib/jxl/encode_internal.h"
#include "lib/jxl/modular/transform/palette.h"

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
         !p.lossless_precise_entropy && !p.lossless_probe_lz77 &&
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
  size_t lower_bound = kUnknownSize;
  bool calibration = false;
};

struct DeferredTrial {
  CompressParams params;
  bool screen;
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

CompressParams WinnerNeighborhood(const CompressParams& winner,
                                  const CodecMetadata& metadata) {
  auto p = Backend(winner, 9);
  if (p.options.predictor == Predictor::Zero) {
    // Smaller groups can make sparse palette-index LZ77 much more expensive.
    p.options.nb_repeats = 0.03f;
  } else {
    p.modular_group_size_shift = 2;
    p.options.nb_repeats = 0.1f;
    p.options.max_properties =
        std::min<int>(2, metadata.m.num_extra_channels +
                             (metadata.m.color_encoding.IsGray() ? 0 : 2));
  }
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
        encode_(encode) {
    if (trace_) {
      static std::atomic<size_t> next_id{0};
      id_ = next_id.fetch_add(1, std::memory_order_relaxed);
      std::fprintf(stderr, "lossless-search event=start schema=1 search=%" PRIuS
                           "\n", id_);
      std::fflush(stderr);
    }
  }

  size_t Id() const { return id_; }

  void Add(int level, CompressParams params, bool baseline = true,
           bool calibration = false) {
    trials_.push_back({level, std::move(params), kUnknownSize, baseline,
                       kUnknownSize, calibration});
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

  // Cooperative finalists share one invocation of the supplied runner. The
  // original candidate-only mode gives each independent encoder no runner.
  Status RunPending(size_t concurrency, bool cooperative = false) {
    const auto batch_start = std::chrono::steady_clock::now();
    const bool capture = FinalPlanOutput() && !CurrentFinalPlan();
    if (capture) concurrency = 1;
    while (next_ < trials_.size()) {
      const size_t count = std::min(concurrency, trials_.size() - next_);
      std::vector<std::unique_ptr<JxlEncoderOutputProcessorWrapper>> outputs(
          count);
      std::vector<AuxOut> stats(count);
      std::vector<uint8_t> discarded(count, 0);
      std::vector<size_t> lower_bounds(count, kUnknownSize);
      std::vector<double> seconds(count, 0);
      std::vector<std::unique_ptr<FinalPlan>> plans(count);
      const auto run = [&](size_t index, ThreadPool* runner) -> Status {
        const auto start = std::chrono::steady_clock::now();
        const auto& trial = trials_[next_ + index];
        if (trace_) TraceTrial(next_ + index, "start", 0, 0);
        outputs[index] =
            jxl::make_unique<JxlEncoderOutputProcessorWrapper>(memory_manager_);
        // Baseline sizes affect later family eligibility and must stay exact.
        // Additional candidates only matter if they strictly beat the winner.
        const size_t limit = !trial.baseline && !trial.calibration &&
                                    trial.params.IsLossless() && best_
                                 ? best_->CurrentPosition()
                                 : kUnknownSize;
        LosslessTrialBudget budget(limit);
        auto params = trial.params;
        LosslessProfileScope profile(params.options.histogram_params.lossless_profile,
                                     "trial");
        profile.Set("search", id_);
        profile.Set("trial", next_ + index);
        profile.Set("stage", trial.level);
        profile.Set("backend", LosslessSearchLevel(params));
        profile.Set("calibration", trial.calibration);
        profile.Set("predictor", static_cast<uint32_t>(params.options.predictor));
        profile.Set("rct_plus_one", static_cast<uint64_t>(params.colorspace + 1));
        profile.Set("group_plus_one", static_cast<uint64_t>(params.modular_group_size_shift + 1));
        if (params.options.nb_repeats >= 0 && params.options.nb_repeats < 10000) {
          profile.Set("ma_ppm", static_cast<uint64_t>(params.options.nb_repeats * 1000000.0));
        }
        profile.Set("previous_plus_one", params.options.max_properties + 1);
        if (profile.Enabled()) params.options.histogram_params.lossless_profile = profile.Id();
        if (limit != kUnknownSize) params.lossless_trial_budget = &budget;
        if (capture) plans[index] = std::make_unique<FinalPlan>();
        FinalPlanScope final_scope(plans[index].get());
        if (capture) JXL_RETURN_IF_ERROR(FinalPlanParameters(&params));
        JXL_RETURN_IF_ERROR(
            encode_(params, capture ? nullptr : runner,
                    outputs[index].get(), stats_ ? &stats[index] : nullptr));
        discarded[index] = limit != kUnknownSize && budget.Reached();
        if (discarded[index]) lower_bounds[index] = limit;
        profile.Set("bounded", discarded[index]);
        profile.Set("bytes", outputs[index]->CurrentPosition());
        if (trace_) {
          seconds[index] = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - start)
                               .count();
          TraceTrial(next_ + index, discarded[index] ? "bounded" : "end",
                     discarded[index] ? limit : outputs[index]->CurrentPosition(),
                     seconds[index]);
        }
        return true;
      };
      if (count == 1) {
        JXL_RETURN_IF_ERROR(run(0, pool_));
      } else if (cooperative && pool_ && !capture) {
        LosslessProbePool arena;
        const auto limits = LosslessWorkerLimits::FromEnvironment();
        JXL_RETURN_IF_ERROR(arena.Run(count, pool_, limits.tasks, count, run));
      } else {
        JXL_RETURN_IF_ERROR(RunOnPool(pool_, 0, count, ThreadPool::NoInit,
                                      [&](uint32_t i, size_t) { return run(i, nullptr); },
                                      "LosslessCandidates"));
      }
      // Stable plan order, not completion order, breaks equal-size ties.
      for (size_t i = 0; i < count; ++i) {
        bounded_ += discarded[i];
        full_work_seconds_ += seconds[i];
        trials_[next_ + i].lower_bound = lower_bounds[i];
        if (discarded[i]) continue;
        trials_[next_ + i].size = outputs[i]->CurrentPosition();
        if (!best_ ||
            outputs[i]->CurrentPosition() < best_->CurrentPosition() ||
            (outputs[i]->CurrentPosition() == best_->CurrentPosition() &&
             trials_[next_ + i].level < best_level_)) {
          best_ = std::move(outputs[i]);
          best_stats_ = stats[i];
          best_level_ = trials_[next_ + i].level;
          best_index_ = next_ + i;
          best_plan_ = std::move(plans[i]);
        }
      }
      next_ += count;
    }
    if (trace_) {
      full_wall_seconds_ += std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - batch_start)
                                .count();
    }
    return true;
  }

  StatusOr<size_t> Refine(int level, CompressParams params,
                          bool baseline = true, bool calibration = false) {
    if (!baseline) {
      const size_t previous = Find(params);
      if (previous != kUnknownSize) return size_t{trials_[previous].size};
    }
    Add(level, std::move(params), baseline, calibration);
    JXL_RETURN_IF_ERROR(RunPending(1));
    return size_t{trials_.back().size};
  }

  size_t Find(const CompressParams& params) const {
    for (size_t i = 0; i < next_; ++i) {
      // The incumbent only shrinks, so a bounded trial cannot win on retry.
      if (SameLosslessCandidate(trials_[i].params, params)) return i;
    }
    return kUnknownSize;
  }

  size_t Related(const CompressParams& params) const {
    size_t best = kUnknownSize;
    for (size_t i = 0; i < next_; ++i) {
      // Cheap global probes have deliberately different backend heuristics;
      // use their paired regional correction, not the normal family estimator.
      if (trials_[i].calibration) continue;
      const auto& p = trials_[i].params;
      if (p.options.predictor != params.options.predictor ||
          (p.palette_colors == 0) != (params.palette_colors == 0) ||
          (p.options.nb_repeats == 0) != (params.options.nb_repeats == 0))
        continue;
      if (best == kUnknownSize || OutcomeSize(i) < OutcomeSize(best)) best = i;
    }
    return best;
  }

  size_t LowerBackend(const CompressParams& params) const {
    size_t best = kUnknownSize;
    for (size_t i = 0; i < next_; ++i) {
      const auto& trial = trials_[i];
      if (trial.calibration || trial.size == kUnknownSize ||
          LosslessSearchLevel(trial.params) >= LosslessSearchLevel(params))
        continue;
      auto comparable = trial.params;
      comparable.speed_tier = params.speed_tier;
      if (!SameLosslessCandidate(comparable, params)) continue;
      if (best == kUnknownSize || LosslessSearchLevel(trial.params) >
                                     LosslessSearchLevel(trials_[best].params))
        best = i;
    }
    return best;
  }

  size_t PaletteCalibrationPeer(const CompressParams& params) const {
    if (params.palette_colors <= 0) return kUnknownSize;
    for (size_t i = next_; i-- > 0;) {
      const auto& trial = trials_[i];
      if (!trial.calibration || trial.size == kUnknownSize ||
          trial.params.palette_colors <= 0 ||
          trial.params.palette_colors == params.palette_colors) continue;
      auto comparable = trial.params;
      comparable.palette_colors = params.palette_colors;
      if (SameLosslessCandidate(comparable, params)) return i;
    }
    return kUnknownSize;
  }

  size_t OutcomeSize(size_t index) const {
    return std::min(trials_[index].size, trials_[index].lower_bound);
  }
  const CompressParams& Params(size_t index) const { return trials_[index].params; }

  size_t Size(size_t index) const { return trials_[index].size; }
  size_t Count() const { return trials_.size(); }

  void TraceTrial(size_t index, const char* event, size_t bytes,
                  double seconds) const {
    const auto& trial = trials_[index];
    const auto& p = trial.params;
    std::fprintf(stderr,
                 "lossless-trial event=%s search=%" PRIuS " id=%" PRIuS
                 " stage=%d baseline=%d calibration=%d backend=%d g=%d "
                 "rct=%d predictor=%u ma=%.3f palette=%d prev=%d careful=%d precise_entropy=%d "
                 "bytes=%" PRIuS " seconds=%.6f\n",
                 event, id_, index, trial.level, trial.baseline, trial.calibration,
                 LosslessSearchLevel(p),
                 p.modular_group_size_shift, p.colorspace,
                 static_cast<unsigned>(p.options.predictor),
                 p.options.nb_repeats, p.palette_colors,
                 p.options.max_properties, p.lz77_careful_first_pass, p.lossless_precise_entropy, bytes,
                 seconds);
    std::fflush(stderr);
  }

  Status Finish(JxlEncoderOutputProcessorWrapper* output, AuxOut* aux_out) {
    JXL_ENSURE(best_);
    if (FinalPlanOutput() && !CurrentFinalPlan()) {
      // Keep the transcript of this exact winning encode, not a second encode
      // with the winner's parameters. Discard all losing/probe transcripts.
      JXL_ENSURE(best_plan_);
      if (!best_plan_->unsupported.empty()) {
        std::fprintf(stderr, "final-plan unsupported=%s\n", best_plan_->unsupported.c_str());
        return JXL_FAILURE("Winning final plan is not fully supported");
      }
      JXL_RETURN_IF_ERROR(best_plan_->Write(FinalPlanOutput()));
      std::fprintf(stderr,
          "final-plan mode=capture winner_bytes=%zu capture_bytes=%zu records=%zu\n",
          best_->CurrentPosition(), best_->CurrentPosition(), best_plan_->records.size());
    }
    JXL_RETURN_IF_ERROR(best_->SetFinalizedPosition());
    std::vector<uint8_t> bytes;
    JXL_RETURN_IF_ERROR(best_->CopyOutput(bytes));
    JXL_RETURN_IF_ERROR(AppendData(*output, bytes));
    if (aux_out) aux_out->Assimilate(best_stats_);
    if (trace_) {
      TraceTrial(best_index_, "winner", bytes.size(), 0);
      std::fprintf(stderr,
                   "lossless-search event=end schema=1 search=%" PRIuS
                   " attempts=%" PRIuS " completed=%" PRIuS
                   " bounded=%" PRIuS " winner=%" PRIuS " bytes=%" PRIuS
                   " full_work_seconds=%.6f full_wall_seconds=%.6f"
                   " seconds=%.6f\n",
                   id_, next_, next_ - bounded_, bounded_, best_index_,
                   bytes.size(), full_work_seconds_, full_wall_seconds_,
                   std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - start_).count());
      std::fflush(stderr);
    }
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
  size_t id_ = 0;
  size_t best_index_ = 0;
  size_t bounded_ = 0;
  double full_work_seconds_ = 0;
  double full_wall_seconds_ = 0;
  const std::chrono::steady_clock::time_point start_ =
      std::chrono::steady_clock::now();
  int best_level_ = 0;
  std::unique_ptr<JxlEncoderOutputProcessorWrapper> best_;
  std::unique_ptr<FinalPlan> best_plan_;
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
  // Retain the older optional confirmation experiment separately from the
  // shared preflight screen; its estimates can reject a whole-frame winner.
  const char* confirm_value = std::getenv("JXL_LOSSLESS_SEARCH_CONFIRM");
  const bool use_confirmation =
      confirm_value && std::strcmp(confirm_value, "1") == 0;
  const char* fast_preflight_value = std::getenv("JXL_LOSSLESS_FAST_PREFLIGHT");
  const bool fast_preflight =
      fast_preflight_value && std::strcmp(fast_preflight_value, "1") == 0;
  const char* feedback_value = std::getenv("JXL_LOSSLESS_PROBE_FEEDBACK");
  const bool probe_feedback =
      feedback_value && std::strcmp(feedback_value, "1") == 0;
  const char* family_value = std::getenv("JXL_LOSSLESS_FAST_FAMILY");
  const char* palette_race_value = std::getenv("JXL_LOSSLESS_PALETTE_RACE");
  const bool palette_preselect = palette_race_value &&
      std::strcmp(palette_race_value, "2") == 0;
  const bool palette_race = palette_race_value &&
      (std::strcmp(palette_race_value, "1") == 0 || palette_preselect);
  const bool fast_family = palette_race ||
      (family_value && std::strcmp(family_value, "1") == 0);
  const char* calibration_value = std::getenv("JXL_LOSSLESS_CALIBRATE");
  const bool calibrate =
      calibration_value && std::strcmp(calibration_value, "1") == 0;
  const char* fast_calibration_value = std::getenv("JXL_LOSSLESS_CALIBRATE_FAST");
  const bool fast_calibration =
      fast_calibration_value && std::strcmp(fast_calibration_value, "1") == 0;
  const char* refinement_calibration_value =
      std::getenv("JXL_LOSSLESS_CALIBRATE_REFINEMENTS");
  const bool refinement_calibration = refinement_calibration_value &&
      std::strcmp(refinement_calibration_value, "1") == 0;
  const char* route_calibration_value =
      std::getenv("JXL_LOSSLESS_CALIBRATE_ROUTES");
  const bool route_calibration = route_calibration_value &&
      std::strcmp(route_calibration_value, "1") == 0;
  const char* yield_value = std::getenv("JXL_LOSSLESS_PREFLIGHT_YIELD");
  const bool preflight_yield = !jpeg && yield_value &&
      std::strcmp(yield_value, "1") == 0;
  const char* anchor_value = std::getenv("JXL_LOSSLESS_ANCHOR_FIRST");
  const bool anchor_first = preflight_yield && anchor_value &&
      std::strcmp(anchor_value, "1") == 0;
  const uint64_t pixels = uint64_t{frame_data.xsize} * frame_data.ysize;
  const char* scope_routing_value = std::getenv("JXL_LOSSLESS_SCOPE_ROUTING");
  const bool broad_scope = scope_routing_value && std::strcmp(scope_routing_value, "2") == 0;
  const char* multiscale_value = std::getenv("JXL_LOSSLESS_MULTISCALE_RANK");
  const bool multiscale_rank = broad_scope && multiscale_value &&
      std::strcmp(multiscale_value, "1") == 0;
  const char* reuse_value = std::getenv("JXL_LOSSLESS_CALIBRATION_REUSE");
  const bool calibration_reuse = preflight_yield &&
      pixels >= (broad_scope ? (1u << 19) : (2u << 20)) && reuse_value &&
      std::strcmp(reuse_value, "1") == 0;
  const char* early_family_value = std::getenv("JXL_LOSSLESS_EARLY_FAMILY");
  const bool early_family = calibration_reuse && calibrate && fast_calibration &&
      route_calibration && early_family_value &&
      std::strcmp(early_family_value, "1") == 0;
  const char* order_value = std::getenv("JXL_LOSSLESS_CALIBRATION_ORDER");
  const bool calibration_order = calibration_reuse && order_value &&
      std::strcmp(order_value, "1") == 0;
  const char* plan_first_value = std::getenv("JXL_LOSSLESS_PLAN_FIRST");
  const bool plan_first = early_family && plan_first_value &&
      std::strcmp(plan_first_value, "1") == 0;
  const char* plan_buckets_value = std::getenv("JXL_LOSSLESS_PLAN_BUCKETS");
  const bool plan_buckets = plan_first && plan_buckets_value &&
      std::strcmp(plan_buckets_value, "1") == 0;
  const char* geometry_value = std::getenv("JXL_LOSSLESS_GEOMETRY_FIRST");
  const bool geometry_first = plan_first &&
      (!geometry_value || std::strcmp(geometry_value, "0") != 0);
  const char* automatic_yield_value = std::getenv("JXL_LOSSLESS_AUTOMATIC_YIELD");
  const bool automatic_yield = plan_first &&
      (!automatic_yield_value || std::strcmp(automatic_yield_value, "0") != 0);
  bool refine_winner =
      !jpeg && pixels > 4096 && metadata.m.num_extra_channels <= 8;
  for (const auto& ec : metadata.m.extra_channel_info) {
    if (ec.dim_shift != 0) refine_winner = false;
  }
  Search search(memory_manager, pool, aux_out != nullptr, trace, encode);
  LosslessScan direct_coverage;
  const char* frontier_value = std::getenv("JXL_LOSSLESS_DIRECT_FRONTIER");
  const char* compact_value = std::getenv("JXL_LOSSLESS_COMPACT_FRONTIER");
  const bool compact_frontier = pixels >= (1u << 19) && pixels < (2u << 20) &&
      compact_value && std::strcmp(compact_value, "1") == 0;
  const bool dense_frontier = frontier_value && std::strcmp(frontier_value, "5") == 0;
  const bool shrink_frontier = dense_frontier ||
      (frontier_value && std::strcmp(frontier_value, "4") == 0);
  const bool extended_frontier = shrink_frontier ||
      (frontier_value && std::strcmp(frontier_value, "3") == 0);
  if (!jpeg && level == 9 && refine_winner && (pixels >= (2u << 20) || compact_frontier) &&
      frontier_value && (std::strcmp(frontier_value, "1") == 0 ||
                         std::strcmp(frontier_value, "2") == 0 ||
                         extended_frontier) && !FinalPlanOutput()) {
    JXL_ASSIGN_OR_RETURN(direct_coverage, ScanLosslessInput(metadata, frame_data, pool));
    const auto& coverage = direct_coverage;
    LosslessCandidateScreen scout(metadata, frame_data, pool, probe, &coverage);
    const bool sparse_tone = extended_frontier && !coverage.LikelyContinuousTone() &&
        coverage.LikelySparseContinuousTone() && coverage.palette_color_lower_bound >
            70000u + palette_internal::kImplicitPaletteSize;
    const bool dense_tone = dense_frontier && coverage.LikelyDenseTexture() &&
        coverage.palette_color_lower_bound > 10000u + palette_internal::kImplicitPaletteSize;
    if (trace) std::fprintf(stderr, "lossless-frontier-gate search=%" PRIuS
        " continuous=%d sparse=%d dense=%d colors=%" PRIuS " flat=%" PRIuS " samples=%" PRIuS "\n",
        search.Id(), coverage.LikelyContinuousTone(), sparse_tone, dense_tone, coverage.sampled_colors,
        coverage.sampled_flat, coverage.sampled_pixels);
    if ((coverage.LikelyContinuousTone() || sparse_tone || dense_tone || compact_frontier) && scout.Available()) {
      auto plans = LosslessProbePlan(params);
      plans.push_back({"automatic-e9", Backend(params, 9)});
      if (extended_frontier)
        plans.push_back({"automatic-neighborhood", WinnerNeighborhood(params, metadata)});
      plans.push_back({"literal-i1", Literal(params, 9, 0.01f)});
      plans.push_back({"literal-i10", Literal(params, 9, 0.1f)});
      JXL_ASSIGN_OR_RETURN(auto ranked,
          scout.Rank(std::move(plans), Backend(params, 9), true));
      JXL_ENSURE(!ranked.empty());
      size_t finalists = 1;
      // This is a sampled ranking, not a guaranteed bound. Close plans retain
      // a second full trial. Lower-effort prefix dominance is not established
      // for this opt-in route; it must not become the normal effort-9 policy.
      while (finalists < std::min<size_t>(compact_frontier ? 2 : extended_frontier ? 4 : 2, ranked.size()) &&
             ranked[finalists].context_bytes <= ranked[0].context_bytes *
                 (compact_frontier ? 1.03 : sparse_tone ? 1.025 : 1.01))
        ++finalists;
      if (trace) {
        std::fprintf(stderr, "lossless-direct-frontier search=%" PRIuS
            " finalists=%" PRIuS " encodes=%" PRIuS " sample_pixels=%" PRIuS " compact=%d\n",
            search.Id(), finalists, scout.Encodes(), scout.SamplePixels(), compact_frontier);
        for (const auto& p : ranked) {
          std::fprintf(stderr, "lossless-frontier-plan search=%" PRIuS
              " name=%s bytes=%" PRIuS " rct=%d predictor=%u palette=%d ma=%.3f\n",
              search.Id(), p.name, p.context_bytes, p.params.colorspace,
              static_cast<unsigned>(p.params.options.predictor), p.params.palette_colors,
              p.params.options.nb_repeats);
        }
      }
      std::vector<size_t> selected(finalists);
      std::iota(selected.begin(), selected.end(), 0);
      const char* parallel_value = std::getenv("JXL_LOSSLESS_FINALISTS_PARALLEL");
      const bool parallel_finalists = parallel_value && std::strcmp(parallel_value, "1") == 0;
      const auto limits = LosslessWorkerLimits::FromEnvironment();
      const uint64_t scratch = pixels * (3 + uint64_t{metadata.m.num_extra_channels}) * 32 +
          (32u << 20);
      // For smaller frames the actual finalists can cost less than additional
      // full-image calibrations. Compare complete outputs using the shared pool.
      if (!compact_frontier && (frontier_value[0] == '2' || extended_frontier) && finalists > 1) {
        const char* fidelity_value = std::getenv("JXL_LOSSLESS_FRONTIER_CALIBRATION");
        const bool medium_calibration = !sparse_tone && fidelity_value &&
            std::strcmp(fidelity_value, "7") == 0;
        std::vector<double> projected(finalists);
        std::vector<size_t> full_bytes(finalists), independent_regions(finalists);
        std::vector<CompressParams> cheap_plans;
        const size_t cheap_begin = search.Count();
        for (size_t i = 0; i < finalists; ++i) {
          auto cheap = Backend(ranked[i].params, medium_calibration ? 7 : 4);
          if (!medium_calibration)
            cheap.options.nb_repeats = std::min(cheap.options.nb_repeats, 0.01f);
          cheap_plans.push_back(cheap);
          search.Add(8, cheap, true, true);
        }
        JXL_RETURN_IF_ERROR(search.RunPending(
            parallel_finalists ? limits.Tasks(finalists, scratch) : 1, parallel_finalists));
        for (size_t i = 0; i < finalists; ++i) {
          const size_t bytes = search.Size(cheap_begin + i);
          full_bytes[i] = bytes;
          JXL_ASSIGN_OR_RETURN(auto scores, scout.Score(cheap_plans[i]));
          independent_regions[i] = scores.size();
          const size_t denominator = std::accumulate(scores.begin(), scores.end(), size_t{0});
          projected[i] = denominator ? double(bytes) * ranked[i].context_bytes / denominator : 0;
          if (trace) std::fprintf(stderr, "lossless-frontier-calibration search=%" PRIuS
              " index=%" PRIuS " backend=%d bytes=%" PRIuS " predicted=%.3f\n",
              search.Id(), i, medium_calibration ? 7 : 4, bytes, projected[i]);
        }
        if (shrink_frontier) {
          projected = ShrinkLosslessCalibration(cheap_plans, full_bytes, projected,
                                                independent_regions);
          if (trace) for (size_t i = 0; i < finalists; ++i)
            std::fprintf(stderr, "lossless-frontier-shrink search=%" PRIuS
                " index=%" PRIuS " predicted=%.3f\n", search.Id(), i, projected[i]);
        }
        if (std::all_of(projected.begin(), projected.end(),
                        [](double score) { return score > 0; })) {
          std::stable_sort(selected.begin(), selected.end(),
              [&](size_t a, size_t b) { return projected[a] < projected[b]; });
          // Sparse foregrounds distort crop rankings. Keep two calibrated
          // alternatives, including RCT siblings just outside the initial tie.
          if (sparse_tone) {
            selected.resize(std::min<size_t>(2, selected.size()));
          } else if (projected[selected[0]] < projected[selected[1]] * 0.995 ||
                     (dense_frontier && LosslessCalibrationAgreement(projected, full_bytes, selected[0]))) {
            selected.resize(1);
          }
          if (extended_frontier && !sparse_tone && selected.size() == 1 &&
              ranked[selected[0]].params.options.predictor == Predictor::Variable) {
            const size_t leader = selected[0];
            for (size_t i = 0; i < finalists; ++i) {
              if (ranked[i].params.options.predictor != kUndefinedPredictor ||
                  ranked[i].params.modular_group_size_shift != 2) continue;
              // Crop boundaries can reverse the group-geometry ranking. A
              // competitive complete cheap witness keeps that geometry alive.
              if (search.Size(cheap_begin + i) <= search.Size(cheap_begin + leader) * 1.01 &&
                  projected[i] <= projected[leader] * 1.05) {
                selected.insert(selected.begin(), i);
                break;
              }
            }
          }
        }
      }
      for (size_t i : selected) search.Add(9, ranked[i].params, false);
      JXL_RETURN_IF_ERROR(search.RunPending(
          parallel_finalists ? limits.Tasks(selected.size(), scratch) : 1, parallel_finalists));
      JXL_RETURN_IF_ERROR(search.Finish(output, aux_out));
      return true;
    }
  }
  const char* global_value = std::getenv("JXL_LOSSLESS_GLOBAL_TOURNAMENT");
  if (!jpeg && level == 9 && refine_winner && global_value &&
      std::strcmp(global_value, "1") == 0 && !FinalPlanOutput()) {
    std::vector<CompressParams> plans{Backend(params, 9),
        Literal(params, 9, 0.01f), Literal(params, 9, 0.1f)};
    for (int rct : {0, 6, 10, 17, 26}) {
      auto p = Backend(params, 9);
      p.modular_group_size_shift = 3;
      p.palette_colors = 0;
      p.channel_colors_percent = 0;
      p.channel_colors_pre_transform_percent = 0;
      p.patches = Override::kOff;
      p.options.predictor = Predictor::Variable;
      p.options.max_properties = 2;
      p.options.nb_repeats = 0.1f;
      p.colorspace = rct;
      plans.push_back(p);
    }
    for (const auto& p : plans) search.Add(3, Backend(p, 7), false, true);
    const uint64_t scratch = pixels * (3 + uint64_t{metadata.m.num_extra_channels}) * 32 +
                               (32u << 20);
    JXL_RETURN_IF_ERROR(search.RunPending(
        std::max<size_t>(1, LosslessWorkerLimits::FromEnvironment().Tasks(plans.size(), scratch))));
    size_t winner = 0;
    for (size_t i = 1; i < plans.size(); ++i)
      if (search.Size(i) < search.Size(winner)) winner = i;
    JXL_ASSIGN_OR_RETURN(size_t ignored, search.Refine(9, plans[winner], false));
    (void)ignored;
    JXL_RETURN_IF_ERROR(search.Finish(output, aux_out));
    return true;
  }
  const char* strips_value = std::getenv("JXL_LOSSLESS_STRIP_TOURNAMENT");
  if (!jpeg && level == 9 && refine_winner && strips_value &&
      std::strcmp(strips_value, "1") == 0 && !FinalPlanOutput()) {
    JXL_ASSIGN_OR_RETURN(auto coverage, ScanLosslessInput(metadata, frame_data, pool));
    LosslessCandidateScreen strips(metadata, frame_data, pool, probe, &coverage, 0, true);
    if (strips.Available()) {
      auto plans = LosslessProbePlan(params);
      plans.push_back({"automatic", Backend(params, 9)});
      plans.push_back({"literal-i1", Literal(params, 9, 0.01f)});
      plans.push_back({"literal-i10", Literal(params, 9, 0.1f)});
      // Add reversible transforms that the old fixed two-transform plan missed.
      for (int rct : {10, 17, 26}) {
        auto p = Backend(params, 9);
        p.modular_group_size_shift = 3;
        p.palette_colors = 0;
        p.channel_colors_percent = 0;
        p.channel_colors_pre_transform_percent = 0;
        p.patches = Override::kOff;
        p.options.predictor = Predictor::Variable;
        p.options.max_properties = 2;
        p.options.nb_repeats = 0.1f;
        p.colorspace = rct;
        plans.push_back({"cross-rct", p});
      }
      JXL_ASSIGN_OR_RETURN(auto ranked,
          strips.Rank(std::move(plans), Backend(params, 9), true));
      JXL_ENSURE(!ranked.empty());
      if (trace) {
        std::fprintf(stderr, "lossless-strips search=%" PRIuS
            " candidates=%" PRIuS " encodes=%" PRIuS " sample_pixels=%" PRIuS "\n",
            search.Id(), ranked.size(), strips.Encodes(), strips.SamplePixels());
        for (const auto& p : ranked) {
          std::fprintf(stderr, "lossless-strip-plan search=%" PRIuS
              " name=%s bytes=%" PRIuS " rct=%d predictor=%u palette=%d ma=%.3f\n",
              search.Id(), p.name, p.context_bytes, p.params.colorspace,
              static_cast<unsigned>(p.params.options.predictor), p.params.palette_colors,
              p.params.options.nb_repeats);
        }
      }
      // An experimental shortlist, not an encoded-size bound. Full encodes
      // choose the output; corpus tests must catch screening mistakes.
      size_t finalists = 1;
      while (finalists < std::min<size_t>(2, ranked.size()) &&
             ranked[finalists].context_bytes <= ranked[0].context_bytes * 1.01)
        ++finalists;
      for (size_t i = 0; i < finalists; ++i) {
        JXL_ASSIGN_OR_RETURN(size_t ignored, search.Refine(9, ranked[i].params, false));
        (void)ignored;
      }
      JXL_RETURN_IF_ERROR(search.Finish(output, aux_out));
      return true;
    }
  }
  LosslessScan scan = std::move(direct_coverage);
  const char* scan_value = std::getenv("JXL_LOSSLESS_SCAN");
  if (scan_value && std::strcmp(scan_value, "1") == 0 && !jpeg && level >= 2 &&
      pixels >= (1u << 19)) {
    const auto start = std::chrono::steady_clock::now();
    if (!scan.Available()) {
      JXL_ASSIGN_OR_RETURN(scan, ScanLosslessInput(metadata, frame_data, pool));
    }
    if (trace) {
      const size_t active = std::count_if(scan.tiles.begin(), scan.tiles.end(),
          [](const auto& tile) { return tile.horizontal || tile.vertical; });
      std::fprintf(stderr, "lossless-scan search=%" PRIuS
          " pixels=%llu tiles=%" PRIuS " active=%" PRIuS
          " literal=%d palette_color_lower_bound=%" PRIuS " seconds=%.6f\n", search.Id(),
          static_cast<unsigned long long>(scan.pixels), scan.tiles.size(), active,
          scan.literal_candidate, scan.palette_color_lower_bound, std::chrono::duration<double>(
              std::chrono::steady_clock::now() - start).count());
      for (const auto& r : scan.Regions()) {
        std::fprintf(stderr, "lossless-scan-region search=%" PRIuS
            " x=%" PRIuS " y=%" PRIuS " width=%" PRIuS " height=%" PRIuS "\n",
            search.Id(), r.x, r.y, r.width, r.height);
      }
      std::fflush(stderr);
    }
  }
  // Always retain a cheap valid result, then let competing families report
  // their actual sizes before granting deeper searches.
  search.Add(1, Backend(params, 3));
  size_t literal_index = kUnknownSize;
  if (!jpeg && level >= 2) {
    bool literal_probe = scan.literal_candidate;
    if (!scan.Available()) {
      JXL_ASSIGN_OR_RETURN(literal_probe, TryLiteralProbe(frame_data));
    }
    if (literal_probe) {
      literal_index = search.Count();
      search.Add(2, Literal(params, 6, 0));
    }
    if (pixels <= 4096) {
      // Fixed predictors cannot stand in for a learned whole-frame model,
      // even on tiny images. Preserve its measured output in the same batch.
      search.Add(2, Backend(params, 9));
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

  LosslessCandidateScreen screen(metadata, frame_data, pool, probe,
                                  scan.Available() ? &scan : nullptr);
  LosslessCandidateScreen context_screen(metadata, frame_data, pool, probe,
                                         nullptr, 128);
  const char* context_value = std::getenv("JXL_LOSSLESS_CONTEXT_FEEDBACK");
  const bool context_feedback = preflight_yield &&
      (!context_value || std::strcmp(context_value, "0") != 0);
  const char* statistical_value = std::getenv("JXL_LOSSLESS_STATISTICAL_RACE");
  const bool statistical_feedback =
      statistical_value && std::strcmp(statistical_value, "1") == 0;
  std::vector<DeferredTrial> deferred_refinements;
  const char* transfer_value = std::getenv("JXL_LOSSLESS_SIBLING_TRANSFER");
  const bool sibling_transfer = transfer_value && std::strcmp(transfer_value, "1") == 0;
  std::vector<LosslessCalibrationFeedback> calibration_feedback;
  const auto score_total = [](const std::vector<size_t>& sizes) {
    return std::accumulate(sizes.begin(), sizes.end(), size_t{0});
  };
  bool early_racing = false;
  const char* sibling_value = std::getenv("JXL_LOSSLESS_SIBLING_RACE");
  const bool sibling_race = sibling_value && std::strcmp(sibling_value, "1") == 0;
  const bool scope_routing = broad_scope ||
      (scope_routing_value && std::strcmp(scope_routing_value, "1") == 0);
  const bool capacity_mismatch = scope_routing &&
      scan.palette_color_lower_bound > 10000u + palette_internal::kImplicitPaletteSize;
  const auto refine = [&](int stage, const CompressParams& candidate,
                          bool baseline = true,
                          bool exploration = false,
                          bool neighborhood = false,
                          bool early_promotion = false) -> StatusOr<size_t> {
    const size_t previous = search.Find(candidate);
    if (previous != kUnknownSize) return search.Size(previous);
    if (sibling_race && stage < 10 && stage >= 4) {
      const auto incumbent = search.BestParamsBefore(stage + 1);
      auto sibling = candidate;
      sibling.colorspace = incumbent.colorspace;
      sibling.modular_group_size_shift = incumbent.modular_group_size_shift;
      if (LosslessSearchLevel(incumbent) >= 9 &&
          incumbent.options.predictor == Predictor::Variable &&
          incumbent.palette_colors == 0 && SameLosslessCandidate(sibling, incumbent)) {
        auto control = Backend(incumbent, 4);
        auto proposal = Backend(candidate, 4);
        control.options.nb_repeats = std::min(0.01f, control.options.nb_repeats);
        proposal.options.nb_repeats = std::min(0.01f, proposal.options.nb_repeats);
        JXL_ASSIGN_OR_RETURN(size_t control_bytes, search.Refine(stage, control, false, true));
        JXL_ASSIGN_OR_RETURN(size_t proposal_bytes, search.Refine(stage, proposal, false, true));
        // A same-family whole-frame proxy preserves repetition and geometry.
        // This is still a heuristic: e10 retries every deferred exact plan.
        const bool defer = control_bytes && proposal_bytes > control_bytes * 1.0025;
        if (trace) {
          std::fprintf(stderr, "lossless-route search=%" PRIuS
              " stage=%d sibling_race=1 control_bytes=%" PRIuS
              " proposal_bytes=%" PRIuS " deferred=%d\n",
              search.Id(), stage, control_bytes, proposal_bytes, defer);
        }
        if (defer) {
          deferred_refinements.push_back({candidate, false});
          return size_t{kUnknownSize};
        }
      }
    }
    // Newly screened small frames must retain a complete conventional witness:
    // short strips underestimate the gain from learning its whole-image tree.
    if (preflight_yield && pixels < (1u << 19) &&
        SameLosslessCandidate(candidate, Backend(params, 9))) {
      return search.Refine(stage, candidate);
    }
    // e3 is the first learned-tree anchor. All subsequent pixel routes share
    // this gate, including conventional, literal and winner refinements.
    if ((stage > 3 || early_promotion) && screen.Available()) {
      const auto start = std::chrono::steady_clock::now();
      const size_t encodes_before = screen.Encodes();
      const size_t pixels_before = screen.SamplePixels();
      const size_t batches_before = screen.Batches();
      if (trace) {
        std::fprintf(stderr, "lossless-screen event=start search=%" PRIuS
                             " stage=%d\n", search.Id(), stage);
        std::fflush(stderr);
      }
      const auto incumbent_params = search.BestParamsBefore(stage + 1);
      JXL_ASSIGN_OR_RETURN(auto incumbent, screen.Score(incumbent_params));
      JXL_ASSIGN_OR_RETURN(auto scores, screen.Score(candidate));
      bool promising = LosslessCandidateScreen::Promising(scores, incumbent);
      auto same_rct_family = candidate;
      same_rct_family.colorspace = incumbent_params.colorspace;
      const bool paired_rct = (capacity_mismatch || broad_scope) && stage < 10 &&
          LosslessSearchLevel(incumbent_params) >= 9 &&
          candidate.palette_colors == 0 &&
          candidate.options.predictor == Predictor::Variable &&
          SameLosslessCandidate(same_rct_family, incumbent_params);
      bool paired_rct_keep = !paired_rct;
      if (paired_rct && (!broad_scope || promising || exploration)) {
        // A close RCT proposal is judged within its representation at a second
        // spatial scale, not against a palette that only fits a cropped image.
        JXL_ASSIGN_OR_RETURN(auto control_context, context_screen.Score(incumbent_params));
        JXL_ASSIGN_OR_RETURN(auto proposed_context, context_screen.Score(candidate));
        paired_rct_keep = score_total(proposed_context) < score_total(control_context) * 0.998 ||
            score_total(scores) < score_total(incumbent) * 0.99;
        if (multiscale_rank && score_total(control_context) && score_total(incumbent)) {
          const double relative = double(score_total(proposed_context)) /
              score_total(control_context) * double(score_total(scores)) / score_total(incumbent);
          paired_rct_keep = relative < 0.995;
        }
        if (broad_scope) promising &= paired_rct_keep;
        else promising = paired_rct_keep;
        if (trace) std::fprintf(stderr, "lossless-route search=%" PRIuS
            " stage=%d paired_rct=1 rct=%d context_bytes=%" PRIuS
            " control_bytes=%" PRIuS " accepted=%d\n", search.Id(), stage,
            candidate.colorspace, score_total(proposed_context),
            score_total(control_context), paired_rct_keep);
      }
      // Independent crop trees cannot reliably discriminate whole-frame group
      // geometry. Keep a close winner refinement rather than treating a small
      // sampling deficit as proof that the geometry loses.
      if (neighborhood && score_total(scores) <= score_total(incumbent) * 1.05)
        promising = true;
      double predicted_bytes = 0;
      const size_t related = search.Related(candidate);
      if (related != kUnknownSize) {
        JXL_ASSIGN_OR_RETURN(auto reference, screen.Score(search.Params(related)));
        if (score_total(reference) != 0) {
          predicted_bytes = static_cast<double>(search.OutcomeSize(related)) *
                            score_total(scores) / score_total(reference);
          const size_t best_size = search.BestBefore(stage + 1);
          // Calibrate each representation with real full-frame feedback. A
          // crop-trained raw tree and a palette/LZ77 route have different bias.
          // A bounded full trial supplies only a lower bound, never a winner.
          promising |= score_total(scores) <= score_total(incumbent) * 1.005 &&
                       predicted_bytes <= best_size * 1.005;
          if (predicted_bytes > best_size * 1.05) promising = false;
        }
      }
      const bool accepted = (promising && (!paired_rct || paired_rct_keep)) || exploration;
      // Whole-frame repetition can beat independently encoded crops. Their
      // ratio is not reliable evidence for pruning such a compact winner.
      const bool calibration_context =
          search.BestBefore(stage + 1) > score_total(incumbent);
      if (trace) {
        std::fprintf(stderr,
                     "lossless-screen event=end search=%" PRIuS
                     " stage=%d encodes=%" PRIuS " sample_pixels=%" PRIuS
                     " batches=%" PRIuS
                     " candidate_bytes=%" PRIuS " incumbent_bytes=%" PRIuS
                     " accepted=%d exploration=%d neighborhood=%d calibration_context=%d"
                     " backend=%d predictor=%u"
                     " g=%d rct=%d ma=%.3f predicted_bytes=%.1f seconds=%.6f\n",
                     search.Id(), stage, screen.Encodes() - encodes_before,
                     screen.SamplePixels() - pixels_before,
                     screen.Batches() - batches_before, score_total(scores),
                     score_total(incumbent), accepted, exploration, neighborhood,
                     calibration_context,
                     LosslessSearchLevel(candidate),
                     static_cast<unsigned>(candidate.options.predictor),
                     candidate.modular_group_size_shift, candidate.colorspace,
                     candidate.options.nb_repeats, predicted_bytes,
                     std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - start).count());
        std::fflush(stderr);
      }
      if (!accepted) {
        if (paired_rct) deferred_refinements.push_back({candidate, false});
        return size_t{kUnknownSize};
      }

      if (sibling_transfer && stage < 10 && !exploration && !neighborhood &&
          LosslessSearchLevel(incumbent_params) >= 9 &&
          incumbent_params.palette_colors > 0) {
        for (const auto& feedback : calibration_feedback) {
          const double predicted = feedback.SiblingEstimate(candidate, scores);
          const size_t best = search.BestBefore(stage + 1);
          if (predicted <= best * 1.20) continue;
          if (trace) std::fprintf(stderr, "lossless-route search=%" PRIuS
              " stage=%d sibling_transfer=1 rct=%d reference_rct=%d"
              " predicted_bytes=%.1f best_bytes=%" PRIuS " deferred=1\n",
              search.Id(), stage, candidate.colorspace, feedback.params.colorspace,
              predicted, best);
          deferred_refinements.push_back({candidate, false});
          return size_t{kUnknownSize};
        }
      }

      // Short strips favor fixed predictors over models that need a 2D
      // training context. Validate the same e9 plans on larger original-pixel
      // blocks, without paying for another low-effort whole-frame encode.
      const bool context_validated = context_feedback && early_racing && stage < 10 &&
          pixels < (1u << 19) && calibration_context &&
          candidate.palette_colors == 0 &&
          candidate.options.predictor != kUndefinedPredictor &&
          !neighborhood &&
          SameLosslessCandidate(incumbent_params, Backend(params, 9));
      if (context_validated) {
        const auto context_start = std::chrono::steady_clock::now();
        const size_t before_encodes = context_screen.Encodes();
        const size_t before_pixels = context_screen.SamplePixels();
        const size_t before_batches = context_screen.Batches();
        if (trace) {
          std::fprintf(stderr, "lossless-screen event=start search=%" PRIuS
              " stage=%d context_validation=1\n", search.Id(), stage);
        }
        JXL_ASSIGN_OR_RETURN(auto control, context_screen.Score(incumbent_params));
        JXL_ASSIGN_OR_RETURN(auto proposed, context_screen.Score(candidate));
        const double tiny_ratio = score_total(incumbent) == 0 ? 0 :
            double(score_total(scores)) / score_total(incumbent);
        const double context_ratio = score_total(control) == 0 ? 0 :
            double(score_total(proposed)) / score_total(control);
        bool local_win = false;
        for (size_t i = 0; i < control.size(); ++i)
          local_win |= proposed[i] < control[i] * 0.97;
        // A scale-dependent gain that mostly disappears is not evidence for
        // another full-frame trial. This is a heuristic, not a byte bound;
        // higher effort retains the exact candidate.
        const bool disappearing_gain = context_ratio >= 0.97 &&
            context_ratio - tiny_ratio > 3 * std::max(0.0, 1 - context_ratio);
        const bool weak_at_both_scales = tiny_ratio >= 0.98 && context_ratio >= 0.98;
        const bool defer = !local_win &&
            (disappearing_gain || weak_at_both_scales || context_ratio >= 1.01);
        if (trace) {
          std::fprintf(stderr, "lossless-screen event=end search=%" PRIuS
              " stage=%d context_validation=1 encodes=%" PRIuS
              " sample_pixels=%" PRIuS " batches=%" PRIuS
              " tiny_ratio=%.6f context_ratio=%.6f deferred=%d seconds=%.6f\n",
              search.Id(), stage, context_screen.Encodes() - before_encodes,
              context_screen.SamplePixels() - before_pixels,
              context_screen.Batches() - before_batches, tiny_ratio, context_ratio,
              defer, std::chrono::duration<double>(
                  std::chrono::steady_clock::now() - context_start).count());
        }
        if (defer) {
          deferred_refinements.push_back({candidate, false});
          return size_t{kUnknownSize};
        }
      }

      // The compact route has already paid for a complete automatic e9 anchor.
      // Do not follow it with another deep search for a weak regional gain.
      // Newly covered small images retain their separate, more cautious policy.
      if (preflight_yield && stage < 10 && neighborhood &&
          pixels >= (1u << 19) && pixels < (2u << 20) &&
          SameLosslessCandidate(incumbent_params, Backend(params, 9)) &&
          score_total(scores) >= score_total(incumbent) * 0.98) {
        if (trace) {
          std::fprintf(stderr, "lossless-route search=%" PRIuS
              " stage=%d calibration=1 compact_neighborhood=1"
              " accepted=0 deferred=1\n", search.Id(), stage);
        }
        deferred_refinements.push_back({candidate, true});
        return size_t{kUnknownSize};
      }

      const auto check_medium = [&]() -> StatusOr<bool> {
        auto medium = Backend(candidate, 7);
        JXL_ASSIGN_OR_RETURN(size_t medium_bytes,
                             search.Refine(stage, medium, false, true));
        const auto medium_start = std::chrono::steady_clock::now();
        const size_t medium_encodes = screen.Encodes();
        const size_t medium_pixels = screen.SamplePixels();
        const size_t medium_batches = screen.Batches();
        if (trace) {
          std::fprintf(stderr, "lossless-screen event=start search=%" PRIuS
                               " stage=%d calibration=1\n", search.Id(), stage);
        }
        JXL_ASSIGN_OR_RETURN(auto medium_scores, screen.Score(medium));
        const size_t medium_total = score_total(medium_scores);
        if (sibling_transfer) calibration_feedback.push_back(
            {candidate, scores, medium_scores, medium_bytes});
        const double prediction = medium_total == 0 ? 0 :
            static_cast<double>(medium_bytes) * score_total(scores) / medium_total;
        const bool keep = medium_total == 0 ||
                          prediction <= search.BestBefore(stage + 1) * 1.01;
        if (trace) {
          std::fprintf(stderr, "lossless-screen event=end search=%" PRIuS
              " stage=%d calibration=1 encodes=%" PRIuS " sample_pixels=%" PRIuS
              " batches=%" PRIuS " seconds=%.6f\n", search.Id(), stage,
              screen.Encodes() - medium_encodes,
              screen.SamplePixels() - medium_pixels,
              screen.Batches() - medium_batches,
              std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             medium_start).count());
          std::fprintf(stderr, "lossless-route search=%" PRIuS
              " stage=%d calibration=1 medium_calibration=1 accepted=%d"
              " cheap_bytes=%" PRIuS " predicted_bytes=%.1f best_bytes=%" PRIuS
              " deferred=%d\n", search.Id(), stage, keep, medium_bytes,
              prediction, search.BestBefore(stage + 1), !keep);
        }
        return bool{keep};
      };

      // The same plan at a lower backend is a closer anchor than the smallest
      // member of a family with different groups, predictors or training data.
      const size_t lower_backend = search.LowerBackend(candidate);
      if (!context_validated && calibrate && route_calibration && stage < 10 &&
          LosslessSearchLevel(candidate) >= 9 &&
          lower_backend != kUnknownSize) {
        JXL_ASSIGN_OR_RETURN(auto reference,
                             screen.Score(search.Params(lower_backend)));
        const size_t reference_total = score_total(reference);
        if (reference_total != 0) {
          const double predicted = static_cast<double>(search.Size(lower_backend)) *
                                   score_total(scores) / reference_total;
          const size_t best_size = search.BestBefore(stage + 1);
          // A lower-backend proxy can miss the very gain being estimated.
          // Paired samples from the requested backend and the already encoded
          // same-family incumbent are stronger evidence when they disagree.
          const bool comparable_peer = preflight_yield &&
              LosslessSearchLevel(incumbent_params) == LosslessSearchLevel(candidate) &&
              candidate.options.predictor == kUndefinedPredictor &&
              incumbent_params.options.predictor == kUndefinedPredictor &&
              candidate.colorspace == incumbent_params.colorspace &&
              candidate.palette_colors == incumbent_params.palette_colors;
          const bool deep_peer_gain = comparable_peer &&
              score_total(scores) < score_total(incumbent) * 0.995;
          // A completed deep rival can already beat the automatic upgrade's
          // calibrated projection. Confirm that result at a second scale,
          // against the SAME completed automatic parent, instead of paying
          // for a full e9 encode solely because raw crop sizes are close.
          // This is a heuristic, not a byte lower bound; e10 replays it.
          if (automatic_yield && pixels >= (2u << 20) &&
              LosslessSearchLevel(incumbent_params) >= 9 &&
              SameLosslessCandidate(candidate, Backend(params, 9)) &&
              predicted > best_size * 1.0025) {
            const auto context_start = std::chrono::steady_clock::now();
            const size_t before_encodes = context_screen.Encodes();
            const size_t before_pixels = context_screen.SamplePixels();
            if (trace) std::fprintf(stderr, "lossless-screen event=start search=%" PRIuS
                " stage=%d automatic_yield=1\n", search.Id(), stage);
            JXL_ASSIGN_OR_RETURN(auto parent_context,
                context_screen.Score(search.Params(lower_backend)));
            JXL_ASSIGN_OR_RETURN(auto proposed_context,
                context_screen.Score(candidate));
            const size_t parent_total = score_total(parent_context);
            const double context_prediction = parent_total == 0 ? 0 :
                static_cast<double>(search.Size(lower_backend)) *
                    score_total(proposed_context) / parent_total;
            const bool defer = parent_context.size() >= 3 &&
                proposed_context.size() == parent_context.size() &&
                context_prediction > best_size * 1.0025;
            if (trace) std::fprintf(stderr, "lossless-screen event=end search=%" PRIuS
                " stage=%d automatic_yield=1 encodes=%" PRIuS " sample_pixels=%" PRIuS
                " predicted_bytes=%.1f context_prediction=%.1f best_bytes=%" PRIuS
                " deferred=%d seconds=%.6f\n", search.Id(), stage,
                context_screen.Encodes() - before_encodes,
                context_screen.SamplePixels() - before_pixels,
                predicted, context_prediction, best_size, defer,
                std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             context_start).count());
            if (defer) {
              deferred_refinements.push_back({candidate, false});
              return size_t{kUnknownSize};
            }
          }
          const bool peer_loss_proposal = context_feedback && comparable_peer &&
              pixels >= (2u << 20) &&
              SameLosslessCandidate(candidate, Backend(params, 9)) &&
              score_total(scores) > score_total(incumbent) * 1.01;
          bool deep_peer_loss = false;
          if (peer_loss_proposal) {
            const auto context_start = std::chrono::steady_clock::now();
            const size_t before_encodes = context_screen.Encodes();
            const size_t before_pixels = context_screen.SamplePixels();
            if (trace) std::fprintf(stderr, "lossless-screen event=start search=%" PRIuS
                " stage=%d peer_context=1\n", search.Id(), stage);
            JXL_ASSIGN_OR_RETURN(auto control, context_screen.Score(incumbent_params));
            JXL_ASSIGN_OR_RETURN(auto proposed, context_screen.Score(candidate));
            deep_peer_loss = score_total(proposed) > score_total(control) * 1.01 &&
                std::equal(proposed.begin(), proposed.end(), control.begin(),
                    [](size_t a, size_t b) { return a >= b * 0.995; });
            if (trace) std::fprintf(stderr, "lossless-screen event=end search=%" PRIuS
                " stage=%d peer_context=1 encodes=%" PRIuS " sample_pixels=%" PRIuS
                " candidate_bytes=%" PRIuS " incumbent_bytes=%" PRIuS
                " deferred=%d seconds=%.6f\n", search.Id(), stage,
                context_screen.Encodes() - before_encodes,
                context_screen.SamplePixels() - before_pixels,
                score_total(proposed), score_total(control), deep_peer_loss,
                std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             context_start).count());
          }
          const bool keep = (predicted <= best_size * 1.02 || deep_peer_gain) &&
                            !deep_peer_loss;
          if (trace) {
            std::fprintf(stderr, "lossless-route search=%" PRIuS
                " stage=%d calibration=1 backend_feedback=1 accepted=%d reference=%" PRIuS
                " predicted_bytes=%.1f best_bytes=%" PRIuS
                " deep_peer_gain=%d deep_peer_loss=%d deferred=%d\n",
                search.Id(), stage, keep, lower_backend, predicted, best_size,
                deep_peer_gain, deep_peer_loss, !keep);
          }
          if (!keep) {
            deferred_refinements.push_back({candidate, !deep_peer_loss});
            return size_t{kUnknownSize};
          }
        }
      }

      // Independent crops do not preserve the number or boundaries of groups.
      // Compare cheap full-frame twins for geometry/training refinements of the
      // same representation, including those promoted by the early tournament.
      auto same_geometry = candidate;
      same_geometry.modular_group_size_shift =
          incumbent_params.modular_group_size_shift;
      auto same_neighborhood = same_geometry;
      same_neighborhood.speed_tier = incumbent_params.speed_tier;
      same_neighborhood.options.nb_repeats = incumbent_params.options.nb_repeats;
      same_neighborhood.options.max_properties = incumbent_params.options.max_properties;
      const bool paired_neighborhood = preflight_yield &&
          SameLosslessCandidate(same_neighborhood, incumbent_params);
      // A complete deep winner already exists. With no measured regional
      // improvement, postpone geometry-only exploration instead of encoding
      // two more full-frame calibration witnesses. This is a heuristic, not a
      // size bound; the higher budget still replays the original candidate.
      if (plan_first && stage < 10 && neighborhood &&
          LosslessSearchLevel(incumbent_params) >= 9 &&
          SameLosslessCandidate(same_geometry, incumbent_params) &&
          score_total(scores) >= score_total(incumbent)) {
        bool deferred = false;
        for (const auto& trial : deferred_refinements) {
          deferred |= SameLosslessCandidate(trial.params, candidate);
        }
        if (!deferred) deferred_refinements.push_back({candidate, false});
        if (trace) {
          std::fprintf(stderr, "lossless-route search=%" PRIuS
              " stage=%d plan_first=1 marginal_geometry=1 deferred=1\n",
              search.Id(), stage);
        }
        return size_t{kUnknownSize};
      }
      const bool raw_calibration = related == kUnknownSize &&
          LosslessSearchLevel(candidate) >= 9 && candidate.palette_colors == 0 &&
          candidate.options.predictor != kUndefinedPredictor &&
          candidate.options.nb_repeats > 0;
      const bool palette_calibration = route_calibration && stage < 10 &&
          related == kUnknownSize && LosslessSearchLevel(candidate) >= 9 &&
          LosslessSearchLevel(incumbent_params) >= 9 &&
          (incumbent_params.palette_colors == 0 || stage >= 8) &&
          candidate.palette_colors > 0 &&
          candidate.options.predictor != kUndefinedPredictor &&
          candidate.options.nb_repeats > 0;
      auto cheap_control = Backend(incumbent_params, 4);
      cheap_control.options.nb_repeats =
          std::min(0.01f, incumbent_params.options.nb_repeats);
      const bool reuse_cheap_control = calibration_reuse && paired_neighborhood &&
          search.Find(cheap_control) != kUnknownSize;
      const bool cheap_palette = palette_calibration &&
          candidate.options.predictor == Predictor::Zero;
      // At the final ordinary-effort refinement, a near-tie involving only
      // group geometry often survives e4 but loses the closer e7 witness.
      // Reverse those two gates, retaining the e4 test if e7 accepts.
      const bool closer_first = calibration_order && stage == 9 &&
          neighborhood && paired_neighborhood && reuse_cheap_control &&
          SameLosslessCandidate(same_geometry, incumbent_params) &&
          score_total(scores) <= score_total(incumbent) * 1.01;
      // Weak evidence normally needs an e7 witness. Use e4 first only for a
      // literal palette or a geometry pair whose control is already paid.
      const bool medium_first = !context_validated && preflight_yield && calibrate && route_calibration &&
          (closer_first ||
           !(calibration_reuse && (cheap_palette || reuse_cheap_control))) &&
          calibration_context && stage < 10 &&
          LosslessSearchLevel(candidate) >= 9 &&
          LosslessSearchLevel(incumbent_params) >= 9 &&
          score_total(scores) >= score_total(incumbent) * 0.98 &&
          ((neighborhood && paired_neighborhood) || raw_calibration || palette_calibration);
      if (medium_first) {
        if (trace && closer_first) {
          std::fprintf(stderr, "lossless-route search=%" PRIuS
              " stage=%d calibration_order=1 medium_first=1\n", search.Id(), stage);
        }
        JXL_ASSIGN_OR_RETURN(bool keep, check_medium());
        if (!keep) {
          deferred_refinements.push_back({candidate, true});
          return size_t{kUnknownSize};
        }
      }
      if (!context_validated && calibrate && route_calibration && calibration_context &&
          (!medium_first || closer_first) && neighborhood &&
          stage < 10 && LosslessSearchLevel(candidate) >= 9 &&
          (paired_neighborhood || (candidate.modular_group_size_shift !=
              incumbent_params.modular_group_size_shift &&
          SameLosslessCandidate(same_geometry, incumbent_params)))) {
        auto cheap_candidate = Backend(candidate, 4);
        cheap_candidate.options.nb_repeats = std::min(0.01f, candidate.options.nb_repeats);
        JXL_ASSIGN_OR_RETURN(size_t control_bytes,
                             search.Refine(stage, cheap_control, false, true));
        JXL_ASSIGN_OR_RETURN(size_t candidate_bytes,
                             search.Refine(stage, cheap_candidate, false, true));
        // A large regional win still merits exploration. Otherwise defer a
        // clear global loss; this is a heuristic, not an encoded-byte bound.
        const bool strong_gain =
            score_total(scores) < score_total(incumbent) * 0.95;
        bool keep = candidate_bytes <= control_bytes * 1.01 || strong_gain;
        double relative_prediction = 0;
        if (keep && !strong_gain) {
          const auto paired_start = std::chrono::steady_clock::now();
          const size_t paired_encodes = screen.Encodes();
          const size_t paired_pixels = screen.SamplePixels();
          const size_t paired_batches = screen.Batches();
          if (trace) {
            std::fprintf(stderr, "lossless-screen event=start search=%" PRIuS
                                 " stage=%d calibration=1\n", search.Id(), stage);
          }
          JXL_ASSIGN_OR_RETURN(auto control_scores, screen.Score(cheap_control));
          JXL_ASSIGN_OR_RETURN(auto candidate_scores, screen.Score(cheap_candidate));
          const size_t control_total = score_total(control_scores);
          const size_t candidate_total = score_total(candidate_scores);
          const size_t incumbent_total = score_total(incumbent);
          // Cancel the proxy's crop bias as well as its full-frame bias. This
          // matters when small groups help crops much more than the real frame.
          if (control_bytes && candidate_total && incumbent_total && control_total) {
            relative_prediction =
                (static_cast<double>(candidate_bytes) / control_bytes) *
                (static_cast<double>(score_total(scores)) / incumbent_total) *
                (static_cast<double>(control_total) / candidate_total);
            // A weak crop reversal is not enough to veto a measured global
            // geometry win; independent crops change the group boundaries.
            keep = relative_prediction <= 1.01 ||
                (calibration_reuse && candidate_bytes < control_bytes &&
                 relative_prediction <= 1.05);
          }
          if (trace) {
            std::fprintf(stderr, "lossless-screen event=end search=%" PRIuS
                " stage=%d calibration=1 encodes=%" PRIuS " sample_pixels=%" PRIuS
                " batches=%" PRIuS " seconds=%.6f\n", search.Id(), stage,
                screen.Encodes() - paired_encodes, screen.SamplePixels() - paired_pixels,
                screen.Batches() - paired_batches,
                std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                               paired_start).count());
          }
        }
        if (trace) {
          std::fprintf(stderr,
              "lossless-route search=%" PRIuS
              " stage=%d calibration=1 geometry_calibration=1 accepted=%d"
              " cheap_bytes=%" PRIuS " control_bytes=%" PRIuS
              " relative_prediction=%.6f deferred=%d\n", search.Id(), stage,
              keep, candidate_bytes, control_bytes, relative_prediction, !keep);
        }
        if (!keep) {
          deferred_refinements.push_back({candidate, true});
          return size_t{kUnknownSize};
        }
        // e4 can favor new contexts which a deep automatic tree has already
        // learned. Weak crop evidence needs a closer full-frame witness.
        if (!medium_first && paired_neighborhood && !strong_gain &&
            LosslessSearchLevel(incumbent_params) >= 9) {
          JXL_ASSIGN_OR_RETURN(bool medium_keep, check_medium());
          if (!medium_keep) {
            deferred_refinements.push_back({candidate, true});
            return size_t{kUnknownSize};
          }
        }
      }

      // Once a deep automatic model has already won, a much larger training
      // fraction deserves its own global calibration, not just crop optimism.
      // Preserve vetoed refinements for e10 instead of deleting that route.
      const bool calibrate_refinement = refinement_calibration && stage < 10 &&
          !(preflight_yield && lower_backend != kUnknownSize) &&
          LosslessSearchLevel(candidate) >= 9 &&
          LosslessSearchLevel(incumbent_params) >= 9 &&
          candidate.options.predictor == kUndefinedPredictor &&
          incumbent_params.options.predictor == kUndefinedPredictor &&
          candidate.palette_colors == incumbent_params.palette_colors &&
          candidate.colorspace == incumbent_params.colorspace &&
          incumbent_params.options.nb_repeats > 0 &&
          candidate.options.nb_repeats >= 3 * incumbent_params.options.nb_repeats;
      // For newly covered small images, completing a strong challenger is
      // preferable to rejecting it with a poorly calibrated e4 extrapolation.
      const bool strong_small_candidate = preflight_yield && pixels < (1u << 19) &&
          score_total(scores) < score_total(incumbent) * 0.95;
      const bool committed_early = plan_first && stage == 3 && early_promotion;
      if (!context_validated && calibrate && calibration_context && !medium_first && !strong_small_candidate &&
          !committed_early &&
          (raw_calibration || calibrate_refinement || palette_calibration)) {
        // Correct crop optimism using a cheap whole-frame member of the same
        // explicit representation. This is a heuristic, not a byte lower bound.
        // e4 still learns a tree, but uses fewer properties/bins than e6. The
        // paired samples must use the same proxy backend as the global probe.
        auto cheap = Backend(candidate, fast_calibration ? 4 : 6);
        cheap.options.nb_repeats = std::min(0.01f, candidate.options.nb_repeats);
        if (statistical_feedback && palette_calibration) {
          const size_t peer = search.PaletteCalibrationPeer(cheap);
          if (peer != kUnknownSize) {
            JXL_ASSIGN_OR_RETURN(auto peer_scores,
                                 screen.Score(search.Params(peer)));
            const double predicted = ConservativeLosslessProjection(
                search.Size(peer), scores, peer_scores);
            const size_t best_size = search.BestBefore(stage + 1);
            const bool deferred = predicted > best_size * 1.15;
            if (trace) {
              std::fprintf(stderr, "lossless-route search=%" PRIuS
                  " stage=%d statistical_peer=1 reference=%" PRIuS
                  " predicted_bytes=%.1f best_bytes=%" PRIuS " deferred=%d\n",
                  search.Id(), stage, peer, predicted, best_size, deferred);
            }
            if (deferred) {
              deferred_refinements.push_back({candidate, true});
              return size_t{kUnknownSize};
            }
          }
        }
        JXL_ASSIGN_OR_RETURN(size_t cheap_bytes,
                             search.Refine(stage, cheap, false, true));
        const auto sample_start = std::chrono::steady_clock::now();
        const size_t before_encodes = screen.Encodes();
        const size_t before_pixels = screen.SamplePixels();
        const size_t before_batches = screen.Batches();
        if (trace) {
          std::fprintf(stderr, "lossless-screen event=start search=%" PRIuS
                               " stage=%d calibration=1\n", search.Id(), stage);
        }
        JXL_ASSIGN_OR_RETURN(auto cheap_scores, screen.Score(cheap));
        double ratio = 1.0;
        const size_t cheap_total = score_total(cheap_scores);
        if (cheap_total != 0) {
          ratio = static_cast<double>(score_total(scores)) / cheap_total;
          // Preserve candidates with a strong local improvement. Tiny blank
          // regions do not determine the extrapolation for the entire frame.
          for (size_t i = 0; i < scores.size(); ++i) {
            if (cheap_scores[i] != 0 && cheap_scores[i] >= cheap_total / 16) {
              ratio = std::min(ratio, static_cast<double>(scores[i]) /
                                          cheap_scores[i]);
            }
          }
        }
        const double calibrated_bytes = cheap_bytes * ratio;
        const size_t best_size = search.BestBefore(stage + 1);
        // An early family switch must show a substantial calibrated gain.
        // Failure here only delays it to the ordinary late-stage tournament.
        bool keep = cheap_total == 0 ||
                    calibrated_bytes <= best_size * (early_promotion ? 0.97 : 1.15);
        if (trace) {
          std::fprintf(stderr,
              "lossless-screen event=end search=%" PRIuS
              " stage=%d calibration=1 encodes=%" PRIuS
              " sample_pixels=%" PRIuS " batches=%" PRIuS " seconds=%.6f\n",
              search.Id(), stage, screen.Encodes() - before_encodes,
              screen.SamplePixels() - before_pixels, screen.Batches() - before_batches,
              std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             sample_start).count());
          std::fprintf(stderr,
              "lossless-route search=%" PRIuS " stage=%d calibration=1 accepted=%d"
              " cheap_bytes=%" PRIuS " predicted_bytes=%.1f best_bytes=%" PRIuS
              " refinement_calibration=%d palette_calibration=%d early_promotion=%d deferred=%d"
              " ratio=%.6f backend=%d predictor=%u rct=%d g=%d ma=%.3f\n",
              search.Id(), stage, keep, cheap_bytes, calibrated_bytes, best_size,
              calibrate_refinement, palette_calibration, early_promotion,
              (calibrate_refinement || palette_calibration) && !keep,
              ratio, LosslessSearchLevel(candidate),
              static_cast<unsigned>(candidate.options.predictor), candidate.colorspace,
              candidate.modular_group_size_shift, candidate.options.nb_repeats);
          std::fflush(stderr);
        }
        if (!keep) {
          if (calibrate_refinement || palette_calibration)
            deferred_refinements.push_back({candidate, palette_calibration});
          return size_t{kUnknownSize};
        }
        // A weak apparent loss is too ambiguous for the e4 proxy. Escalate one
        // rung using the original plan/training fraction, not another deep e9
        // encode. Its complete output is also eligible to win.
        if (route_calibration && stage < 10 &&
            LosslessSearchLevel(incumbent_params) >= 9 &&
            (calibrated_bytes > best_size * 1.01 ||
             (preflight_yield &&
              score_total(scores) >= score_total(incumbent) * 0.98)) &&
            (raw_calibration || palette_calibration)) {
          JXL_ASSIGN_OR_RETURN(keep, check_medium());
          if (!keep) {
            deferred_refinements.push_back({candidate, true});
            return size_t{kUnknownSize};
          }
        }
      }
    }
    // A fixed-tree result does not calibrate a learned tree. Finish the first
    // admitted trial of a representation so later refinements get an exact
    // full-frame measurement, rather than extrapolating an abandoned prefix.
    const bool anchor = screen.Available() && search.Related(candidate) == kUnknownSize;
    return search.Refine(stage, candidate, baseline || anchor);
  };

  LosslessProbeResult analysis;
  const auto analyze = [&](const CompressParams& incumbent,
                            bool ambiguous_families) -> Status {
    const auto start = std::chrono::steady_clock::now();
    if (trace) {
      std::fprintf(stderr, "lossless-probes event=start search=%" PRIuS "\n",
                   search.Id());
      std::fflush(stderr);
    }
    if ((scan.Available() || preflight_yield) && screen.Available()) {
      JXL_ASSIGN_OR_RETURN(analysis, screen.Analyze(params, incumbent));
    } else {
      JXL_ASSIGN_OR_RETURN(
          analysis,
          ProbeLosslessCandidates(params, metadata, frame_data, pool, probe,
                                   &incumbent, probe_feedback));
    }
    // Crop-trained trees can miss interactions that only pay off on the whole
    // frame. When both established families are competitive but all samples
    // reject alternatives, reserve exploration rather than trusting a veto.
    const bool explore = analysis.ranked.empty() && ambiguous_families &&
                         !analysis.exploratory.empty();
    if (explore) analysis.ranked = std::move(analysis.exploratory);
    if (multiscale_rank && !analysis.ranked.empty()) {
      const auto& leader = analysis.ranked.front();
      if (leader.params.palette_colors == 0 &&
          leader.params.options.predictor == Predictor::Variable) {
        struct Ranked { size_t index; double score; };
        std::vector<Ranked> siblings;
        for (size_t i = 0; i < analysis.ranked.size(); ++i) {
          const auto& candidate = analysis.ranked[i];
          auto family = candidate.params;
          family.colorspace = leader.params.colorspace;
          if (!SameLosslessCandidate(family, leader.params) ||
              candidate.context_bytes > leader.context_bytes * 1.06) continue;
          JXL_ASSIGN_OR_RETURN(auto context, context_screen.Score(candidate.params));
          const double score = double(candidate.context_bytes) * score_total(context);
          siblings.push_back({i, score});
          if (trace) std::fprintf(stderr, "lossless-route search=%" PRIuS
              " multiscale_rank=1 rct=%d main_bytes=%" PRIuS " context_bytes=%" PRIuS "\n",
              search.Id(), candidate.params.colorspace, candidate.context_bytes, score_total(context));
        }
        auto positions = siblings;
        std::stable_sort(siblings.begin(), siblings.end(), [](const auto& a, const auto& b) {
          return a.score < b.score;
        });
        auto ranked = analysis.ranked;
        for (size_t i = 0; i < siblings.size(); ++i)
          analysis.ranked[positions[i].index] = ranked[siblings[i].index];
      }
    }
    if (trace) {
      std::fprintf(stderr,
                   "lossless-probes event=end search=%" PRIuS " samples=%" PRIuS
                   " encodes=%" PRIuS " control_bytes=%" PRIuS
                   " candidates=%" PRIuS " shortlisted=%" PRIuS
                   " screen_samples=%" PRIuS " bounded=%" PRIuS
                   " screen_bounded=%" PRIuS " screen_saved_encodes=%" PRIuS
                   " validation_encodes=%" PRIuS " sample_pixels=%" PRIuS
                   " small_control_bytes=%" PRIuS
                   " exploration=%d"
                   " seconds=%.6f\n",
                   search.Id(), analysis.samples, analysis.encodes,
                   analysis.control_bytes, analysis.candidates,
                   analysis.shortlisted, analysis.screen_samples,
                   analysis.bounded, analysis.screen_bounded,
                   analysis.screen_saved_encodes, analysis.validation_encodes,
                   analysis.sample_pixels, analysis.small_control_bytes,
                   explore,
                   std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - start)
                       .count());
      for (const auto& candidate : analysis.ranked) {
        std::fprintf(stderr, "lossless-probe search=%" PRIuS
                             " name=%s bytes=%" PRIuS " small_bytes=%" PRIuS
                             " exploratory=%d"
                             "\n",
                     search.Id(), candidate.name, candidate.bytes,
                     candidate.small_bytes, candidate.exploratory);
      }
    }
    return true;
  };
  std::vector<LosslessProbeCandidate> frontier;
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
      std::fprintf(stderr, "lossless-confirm event=start search=%" PRIuS
                           " stage=%d\n", search.Id(), stage);
      std::fflush(stderr);
    }
    const bool cached = incumbent == confirmed_incumbent && confirmed_bytes != 0;
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
                   "lossless-confirm event=end search=%" PRIuS
                   " stage=%d samples=%" PRIuS " encodes=%" PRIuS
                   " candidate_bytes=%" PRIuS " incumbent_bytes=%" PRIuS
                   " accepted=%d seconds=%.6f\n",
                   search.Id(), stage, result.samples,
                   result.samples * (cached ? 1 : 2), result.candidate_bytes,
                   result.incumbent_bytes, result.accepted,
                   std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - start)
                       .count());
      std::fflush(stderr);
    }
    return bool{result.accepted};
  };
  bool fast_literal = false;
  bool pending_automatic = false;
  bool early_family_winner = false;
  for (int stage = 3; stage <= level; ++stage) {
    // Each stage follows the same ordered decisions at every requested effort.
    // No decision sees a future stage or changes a lower effort's work.
    if (stage == 10) {
      for (const auto& candidate : deferred_refinements) {
        if (search.Find(candidate.params) != kUnknownSize) continue;
        if (trace) {
          std::fprintf(stderr, "lossless-route search=%" PRIuS
              " stage=10 deferred_replay=1 backend=%d screened=%d\n",
              search.Id(), LosslessSearchLevel(candidate.params), candidate.screen);
          std::fflush(stderr);
        }
        JXL_ASSIGN_OR_RETURN(size_t size,
            candidate.screen ? refine(stage, candidate.params, false, false, true)
                             : search.Refine(stage, candidate.params, false));
        (void)size;
      }
    }
    if (stage == 3 && plan_first && screen.Available() &&
        literal_size != kUnknownSize &&
        (literal_size >= static_cast<double>(regular_size) * 0.8 ||
         capacity_mismatch || broad_scope) &&
        literal_size <= static_cast<double>(regular_size) * 1.35) {
      // Let the shared screen nominate a strongly separated family before the
      // automatic e7 full-frame anchor when cheap full-frame families compete.
      // Commit to one complete e9 encode,
      // instead of first paying for a complete e4 witness of the same plan.
      // The actual output must beat the cheap anchors by a substantial margin.
      JXL_RETURN_IF_ERROR(analyze(search.BestParamsBefore(stage), false));
      if (!analysis.ranked.empty()) {
        auto& candidate = analysis.ranked.front().params;
        if (candidate.palette_colors == 0 &&
            candidate.options.predictor == Predictor::Variable) {
          JXL_ASSIGN_OR_RETURN(auto proposed, screen.Score(candidate));
          JXL_ASSIGN_OR_RETURN(auto automatic, screen.Score(Backend(params, 7)));
          const size_t before = search.BestBefore(stage);
          if (score_total(automatic) != 0 &&
              score_total(proposed) < score_total(automatic) * (broad_scope ? 0.9 : 0.8)) {
            candidate.lossless_bucket_matcher = plan_buckets && !scope_routing;
            JXL_ASSIGN_OR_RETURN(size_t size,
                refine(stage, candidate, false, false, false, true));
            early_family_winner = size != kUnknownSize && size < before * 0.8 &&
                SameLosslessCandidate(search.BestParamsBefore(stage + 1), candidate);
            auto cheap = Backend(candidate, 4);
            cheap.options.nb_repeats = std::min(0.01f, candidate.options.nb_repeats);
            if (early_family_winner) {
              deferred_refinements.push_back({cheap, false});
              deferred_refinements.push_back({Backend(params, 7), false});
              deferred_refinements.push_back({Backend(params, 9), false});
              if (candidate.lossless_bucket_matcher) {
                auto chain_candidate = candidate;
                chain_candidate.lossless_bucket_matcher = false;
                deferred_refinements.push_back({chain_candidate, false});
              }
            } else {
              JXL_ASSIGN_OR_RETURN(size_t cheap_bytes,
                                   search.Refine(stage, cheap, false, true));
              (void)cheap_bytes;
            }
            if (trace) {
              std::fprintf(stderr, "lossless-route search=%" PRIuS
                  " stage=3 plan_first=1 buckets=%d promoted=%d previous_bytes=%" PRIuS
                  " best_bytes=%" PRIuS "\n", search.Id(), plan_buckets,
                  early_family_winner, before, search.BestBefore(stage + 1));
            }
          }
        }
      }
    }
    if (stage == 4) {
      // Experimental routing from full-frame feedback, not a safe lower bound.
      // A clearly winning fixed palette can justify direct learned-palette
      // trials instead of reopening all sampled predictor families.
      fast_literal = fast_family && !early_family_winner && !jpeg && screen.Available() &&
          literal_size != kUnknownSize && regular_size != kUnknownSize &&
          // Header-dominated/repetitive frames can favor a tiny predictor tree
          // despite a very strong palette anchor (e.g. gray periodic stripes).
          literal_size >= 1024 &&
          literal_size < static_cast<double>(regular_size) * 0.8;
      if (trace && fast_family) {
        std::fprintf(stderr, "lossless-route search=%" PRIuS
            " stage=4 fast_literal=%d literal_bytes=%" PRIuS
            " regular_bytes=%" PRIuS "\n", search.Id(), fast_literal,
            literal_size, regular_size);
        std::fflush(stderr);
      }
      // Do not reopen every representation when cheap full-frame evidence has
      // already separated the families. Such images still use the shared gate
      // on their conventional and later crop-proposed routes.
      early_racing = screen.Available() &&
                     search.Competitive(literal_size, stage, 1.35);
    }
    if (fast_literal) {
      // Fixed stage prefixes retain effort monotonicity within this route.
      // These are full lossless encodes; no sample score becomes the output.
      if (stage == 4 || stage == 6 || stage == 9) {
        const float fraction = stage == 4 ? 0.01f : stage == 6 ? 0.03f : 1.0f;
        auto candidate = Literal(params, 9, fraction);
        if (palette_preselect && stage == 4) {
          size_t best_score = kUnknownSize;
          for (float sample_fraction : {0.01f, 0.03f, 1.0f}) {
            const auto choice = Literal(params, 9, sample_fraction);
            JXL_ASSIGN_OR_RETURN(auto scored, screen.Score(choice));
            const size_t bytes = score_total(scored);
            if (bytes != 0 && bytes < best_score) {
              best_score = bytes;
              candidate = choice;
            }
          }
          for (float sample_fraction : {0.01f, 0.03f, 1.0f}) {
            auto choice = Literal(params, 9, sample_fraction);
            if (!SameLosslessCandidate(candidate, choice))
              deferred_refinements.push_back({std::move(choice), false});
          }
          if (trace) {
            std::fprintf(stderr, "lossless-route search=%" PRIuS
                " stage=4 palette_preselect=1 ma=%.3f sample_bytes=%" PRIuS "\n",
                search.Id(), candidate.options.nb_repeats, best_score);
          }
        }
        if (palette_race && stage > 4) {
          // Paired regional evidence only ranks this already separated family.
          // It is not a bound on the unencoded whole frame. Replay at level 10.
          JXL_ASSIGN_OR_RETURN(auto proposed, screen.Score(candidate));
          JXL_ASSIGN_OR_RETURN(auto current,
              screen.Score(search.BestParamsBefore(stage)));
          const size_t proposed_bytes = score_total(proposed);
          const size_t current_bytes = score_total(current);
          const bool accepted = current_bytes == 0 || proposed_bytes == 0 ||
              proposed_bytes < static_cast<double>(current_bytes) * 0.999;
          if (trace) {
            std::fprintf(stderr, "lossless-route search=%" PRIuS
                " stage=%d palette_race=1 candidate_bytes=%" PRIuS
                " incumbent_bytes=%" PRIuS " accepted=%d\n",
                search.Id(), stage, proposed_bytes, current_bytes, accepted);
          }
          if (!accepted) {
            deferred_refinements.push_back({candidate, false});
            continue;
          }
        }
        JXL_ASSIGN_OR_RETURN(size_t size,
            search.Refine(stage, candidate, false));
        (void)size;
      }
      if (stage == 10) {
        auto deeper = search.BestParamsBefore(stage);
        deeper.speed_tier = SpeedTier::kGlacier;
        deeper.options.nb_repeats = std::max(0.5f, deeper.options.nb_repeats);
        deeper.lz77_careful_first_pass = true;
        JXL_ASSIGN_OR_RETURN(size_t size, search.Refine(stage, deeper, false));
        (void)size;
      }
      continue;
    }
    if (early_racing && stage >= 4) {
      if (stage == 4 && broad_scope && early_family_winner) {
        // Reuse the completed tournament. A new incumbent does not invalidate
        // its measured crop scores or justify a second set of cheap proxies.
        frontier = analysis.ranked;
        frontier.erase(std::remove_if(frontier.begin(), frontier.end(),
            [&](const auto& candidate) {
              return search.Find(candidate.params) != kUnknownSize;
            }), frontier.end());
        if (trace) std::fprintf(stderr, "lossless-route search=%" PRIuS
            " stage=4 reused_frontier=1 candidates=%" PRIuS "\n",
            search.Id(), frontier.size());
      }
      if (stage == 4 && !(broad_scope && early_family_winner)) {
        bool geometry_winner = false;
        if (geometry_first && regular_size <= literal_size) {
          const auto start = std::chrono::steady_clock::now();
          const size_t encodes_before = screen.Encodes();
          const size_t pixels_before = screen.SamplePixels();
          const size_t batches_before = screen.Batches();
          if (trace) std::fprintf(stderr, "lossless-screen event=start search=%" PRIuS
              " stage=4 geometry_first=1\n", search.Id());
          JXL_ASSIGN_OR_RETURN(auto ranked,
              screen.Rank(LosslessProbePlan(params), Backend(params, 7), true));
          JXL_ASSIGN_OR_RETURN(auto smaller, screen.SmallerGroups(ranked));
          JXL_ASSIGN_OR_RETURN(auto control, screen.Score(Backend(params, 7)));
          if (trace) std::fprintf(stderr, "lossless-screen event=end search=%" PRIuS
              " stage=4 geometry_first=1 encodes=%" PRIuS " sample_pixels=%" PRIuS
              " batches=%" PRIuS " proposals=%" PRIuS " seconds=%.6f\n",
              search.Id(), screen.Encodes() - encodes_before,
              screen.SamplePixels() - pixels_before, screen.Batches() - batches_before,
              smaller.size(), std::chrono::duration<double>(
                  std::chrono::steady_clock::now() - start).count());
          if (!smaller.empty()) {
            const auto& candidate = smaller.front();
            if (candidate.context_bytes < score_total(control) * 0.95) {
              // A crop cannot establish whole-frame group geometry. Calibrate
              // this exact representation before committing to its deep search.
              const auto medium = Backend(candidate.params, 7);
              JXL_ASSIGN_OR_RETURN(size_t medium_bytes,
                  search.Refine(stage, medium, false, true));
              const auto sample_start = std::chrono::steady_clock::now();
              const size_t sample_encodes = screen.Encodes();
              const size_t sample_pixels = screen.SamplePixels();
              const size_t sample_batches = screen.Batches();
              if (trace) std::fprintf(stderr, "lossless-screen event=start search=%" PRIuS
                  " stage=4 geometry_calibration=1\n", search.Id());
              JXL_ASSIGN_OR_RETURN(auto medium_scores, screen.Score(medium));
              if (trace) std::fprintf(stderr, "lossless-screen event=end search=%" PRIuS
                  " stage=4 geometry_calibration=1 encodes=%" PRIuS
                  " sample_pixels=%" PRIuS " batches=%" PRIuS " seconds=%.6f\n",
                  search.Id(), screen.Encodes() - sample_encodes,
                  screen.SamplePixels() - sample_pixels, screen.Batches() - sample_batches,
                  std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - sample_start).count());
              const size_t medium_total = score_total(medium_scores);
              const double prediction = medium_total == 0 ? 0 :
                  double(medium_bytes) * candidate.context_bytes / medium_total;
              if (medium_total && prediction < regular_size * 0.85) {
                JXL_ASSIGN_OR_RETURN(size_t size,
                    search.Refine(stage, candidate.params, false));
                geometry_winner = size < regular_size * 0.85 &&
                    SameLosslessCandidate(search.BestParamsBefore(stage + 1),
                                          candidate.params);
              }
              if (trace) std::fprintf(stderr, "lossless-route search=%" PRIuS
                  " stage=4 geometry_first=1 predictor=%u g=2"
                  " predicted_bytes=%.1f promoted=%d seconds=%.6f\n",
                  search.Id(), static_cast<unsigned>(candidate.params.options.predictor),
                  prediction, geometry_winner, std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - start).count());
            }
          }
          // Keep the automatic upgrade in the normal calibrated frontier. Do
          // not pay for it unconditionally before an already completed rival.
          if (geometry_winner) {
            pending_automatic = true;
            deferred_refinements.push_back({Backend(params, 8), false});
          }
        }
        // Independent crops can rank the conventional route below the shortlist
        // even when it is already the best full-frame family. Measure its next
        // inexpensive rung first; promote a real gain before spending on rivals.
        if (preflight_yield && regular_size <= literal_size && !geometry_winner) {
          JXL_ASSIGN_OR_RETURN(size_t automatic,
                               search.Refine(stage, Backend(params, 8)));
          // Small crop trees understate the whole-frame automatic model. Do
          // not let a flat e7/e8 result bury e9 beyond the frontier budget.
          const bool promote = pixels < (1u << 19) ||
              automatic < static_cast<double>(regular_size) * 0.995;
          if (promote && calibration_reuse && pixels >= (2u << 20)) {
            JXL_ASSIGN_OR_RETURN(auto auto_scores, screen.Score(Backend(params, 9)));
            const size_t auto_total = score_total(auto_scores);
            // Reuse the staged batch screen: scoring every exact plan here
            // bypasses its cheap shortlist and can cost more than the upgrade.
            JXL_ASSIGN_OR_RETURN(auto rivals,
                screen.Rank(LosslessProbePlan(params), Backend(params, 9), true));
            for (const auto& candidate : rivals) {
              if (auto_total && candidate.context_bytes < auto_total * 0.85) {
                pending_automatic = true;
                break;
              }
            }
          }
          if (trace) {
            std::fprintf(stderr, "lossless-route search=%" PRIuS
                " stage=%d automatic_feedback=1 previous_bytes=%" PRIuS
                " cheap_bytes=%" PRIuS " promote=%d pending=%d\n",
                search.Id(), stage, regular_size, automatic,
                promote && !pending_automatic, pending_automatic);
          }
          if (promote && !pending_automatic) {
            JXL_ASSIGN_OR_RETURN(size_t deep,
                                 search.Refine(stage, Backend(params, 9)));
            (void)deep;
          }
        }
        const auto incumbent = search.BestParamsBefore(stage + 1);
        // Scan-guided ranking already tests every family. Do not run a second,
        // independent small-crop tournament before the cached context screen.
        if (scan.Available()) {
          analysis.ranked = LosslessProbePlan(params);
        } else {
          JXL_RETURN_IF_ERROR(analyze(incumbent, false));
        }
        const auto add = [&](const char* name, const CompressParams& p) {
          if (search.Find(p) != kUnknownSize) return;
          for (const auto& candidate : frontier) {
            if (SameLosslessCandidate(candidate.params, p)) return;
          }
          frontier.push_back({name, p});
        };
        add("regular-e8", Backend(params, 8));
        add("regular-e9", Backend(params, 9));
        add("anchor-neighborhood", WinnerNeighborhood(incumbent, metadata));
        if (literal_index != kUnknownSize) {
          add("literal-fixed", Literal(params, 8, 0));
          add("literal-i1", Literal(params, 9, 0.01f));
          add("literal-i10", Literal(params, 9, 0.1f));
          add("literal-i100", Literal(params, 9, 1.0f));
        }
        for (const auto& candidate : analysis.ranked)
          add(candidate.name, candidate.params);
        for (const auto& candidate : analysis.exploratory)
          add(candidate.name, candidate.params);
        const auto start = std::chrono::steady_clock::now();
        const size_t encodes_before = screen.Encodes();
        const size_t pixels_before = screen.SamplePixels();
        const size_t batches_before = screen.Batches();
        const size_t candidates = frontier.size();
        if (trace) {
          std::fprintf(stderr, "lossless-screen event=start search=%" PRIuS
                               " stage=4 ranking=1\n", search.Id());
          std::fflush(stderr);
        }
        JXL_ASSIGN_OR_RETURN(
            frontier, screen.Rank(std::move(frontier), incumbent,
                                   fast_preflight || scan.Available()));
        const char* beam_value = std::getenv("JXL_LOSSLESS_FAMILY_BEAM");
        if (beam_value && std::strcmp(beam_value, "1") == 0)
          frontier = DiverseLosslessFrontier(std::move(frontier));
        if (trace) {
          std::fprintf(stderr,
                       "lossless-screen event=end search=%" PRIuS
                       " stage=4 ranking=1 fast=%d candidates=%" PRIuS
                       " survivors=%" PRIuS " encodes=%" PRIuS
                       " sample_pixels=%" PRIuS " batches=%" PRIuS " seconds=%.6f\n",
                       search.Id(), fast_preflight || scan.Available(), candidates, frontier.size(),
                       screen.Encodes() - encodes_before,
                       screen.SamplePixels() - pixels_before,
                       screen.Batches() - batches_before,
                       std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - start).count());
          for (const auto& candidate : frontier) {
            std::fprintf(stderr,
                         "lossless-probe search=%" PRIuS
                         " name=%s context_bytes=%" PRIuS "\n",
                         search.Id(), candidate.name, candidate.context_bytes);
          }
          std::fflush(stderr);
        }
      }
      if (stage == 5 && pending_automatic) {
        // Judge the pending upgrade against a completed rival, using its
        // measured e8 parent. Preserve rejected upgrades for the e10 budget.
        pending_automatic = false;
        JXL_ASSIGN_OR_RETURN(size_t size,
                             refine(stage, Backend(params, 9), false));
        if (size == kUnknownSize) {
          deferred_refinements.push_back({Backend(params, 9), false});
        }
      }
      // Effort grants more promotions from one fixed, early ranking. Each
      // decision uses the current complete winner, never a future effort.
      const size_t index = stage - 4;
      if (index < frontier.size()) {
        JXL_ASSIGN_OR_RETURN(bool accepted, confirm(stage, frontier[index].params));
        if (accepted) {
          const bool neighborhood = preflight_yield && SameLosslessCandidate(
              frontier[index].params,
              WinnerNeighborhood(search.BestParamsBefore(stage + 1), metadata));
          JXL_ASSIGN_OR_RETURN(size_t size, refine(
              stage, frontier[index].params, false,
              stage == 4 && !early_family_winner, neighborhood));
          (void)size;
        }
      }
      if (stage == 9) {
        auto geometry = WinnerNeighborhood(search.BestParamsBefore(stage + 1), metadata);
        if (broad_scope && early_family_winner) {
          deferred_refinements.push_back({geometry, false});
        } else {
          JXL_ASSIGN_OR_RETURN(size_t size,
              refine(stage, geometry, false, false, true));
          (void)size;
        }
      }
      if (stage == 10) {
        JXL_ASSIGN_OR_RETURN(size_t size, refine(stage, Backend(params, 10), false));
        (void)size;
        auto deeper = search.BestParamsBefore(stage);
        deeper.speed_tier = SpeedTier::kGlacier;
        deeper.options.nb_repeats = std::max(0.5f, deeper.options.nb_repeats);
        deeper.options.max_properties =
            std::min<int>(4, metadata.m.num_extra_channels +
                                 (metadata.m.color_encoding.IsGray() ? 0 : 2));
        deeper.modular_group_size_shift = 3;
        deeper.lz77_careful_first_pass = true;
        JXL_ASSIGN_OR_RETURN(size_t deep_size, refine(stage, deeper, false));
        (void)deep_size;
      }
      continue;
    }
    if (stage == 8 && !jpeg && !early_family_winner) {
      const size_t best_literal = std::min(literal_size, learned_size);
      const bool ambiguous_families =
          regular_size != kUnknownSize && best_literal != kUnknownSize &&
          std::max(regular_size, best_literal) <=
              1.35 * std::min(regular_size, best_literal);
      JXL_RETURN_IF_ERROR(
          analyze(search.BestParamsBefore(stage), ambiguous_families));
    }
    int backend = 0;
    if (stage == 3) backend = jpeg ? 5 : 7;
    if (stage == 5) backend = jpeg ? 7 : (refine_winner ? 0 : 8);
    if (stage == 7) backend = jpeg ? 8 : 9;
    if (stage == 9 && jpeg) backend = 9;
    if (stage == 10) backend = 10;
    // A fixed-tree probe cannot predict the benefit of learning a tree. Always
    // sample that family once before deciding whether deeper training pays.
    if (backend && !(stage == 3 && early_family_winner) &&
        (stage == 3 || search.Competitive(regular_size, stage, 1.35))) {
      JXL_ASSIGN_OR_RETURN(size_t size,
                           refine(stage, Backend(params, backend)));
      regular_size = std::min(regular_size, size);
    }
    if (stage == 4 && search.Competitive(literal_size, stage, 1.35)) {
      JXL_ASSIGN_OR_RETURN(size_t size,
                           refine(stage, Literal(params, 8, 0)));
      literal_size = std::min(literal_size, size);
    }
    if (stage == 5 && refine_winner) {
      if (early_family && screen.Available() &&
          SameLosslessCandidate(search.BestParamsBefore(stage), Backend(params, 7))) {
        // Reuse the batched family shortlist before paying for a deep automatic
        // geometry. Only a completed challenger can become the new anchor.
        JXL_RETURN_IF_ERROR(analyze(search.BestParamsBefore(stage), false));
        if (!analysis.ranked.empty()) {
          const auto& candidate = analysis.ranked.front().params;
          if (candidate.palette_colors == 0 &&
              candidate.options.predictor == Predictor::Variable) {
            const size_t before = search.BestBefore(stage + 1);
            JXL_ASSIGN_OR_RETURN(size_t size,
                refine(stage, candidate, false, false, false, true));
            early_family_winner = size < before && SameLosslessCandidate(
                search.BestParamsBefore(stage + 1), candidate);
            if (trace) {
              std::fprintf(stderr, "lossless-route search=%" PRIuS
                  " stage=5 early_family=1 promoted=%d previous_bytes=%" PRIuS
                  " best_bytes=%" PRIuS "\n", search.Id(), early_family_winner,
                  before, search.BestBefore(stage + 1));
            }
          }
        }
      }
      if (!early_family_winner && anchor_first &&
          pixels >= (2u << 20) && screen.Available() &&
          SameLosslessCandidate(search.BestParamsBefore(stage), Backend(params, 7))) {
        const auto start = std::chrono::steady_clock::now();
        const size_t encodes_before = screen.Encodes();
        const size_t pixels_before = screen.SamplePixels();
        const size_t batches_before = screen.Batches();
        if (trace) {
          std::fprintf(stderr, "lossless-screen event=start search=%" PRIuS
                               " stage=5 anchor_order=1\n", search.Id());
        }
        JXL_ASSIGN_OR_RETURN(auto automatic, screen.Score(Backend(params, 9)));
        JXL_ASSIGN_OR_RETURN(auto neighbor, screen.Score(
            WinnerNeighborhood(search.BestParamsBefore(stage), metadata)));
        const bool regional_first = score_total(automatic) < score_total(neighbor) * 0.95;
        if (trace) {
          std::fprintf(stderr, "lossless-screen event=end search=%" PRIuS
              " stage=5 anchor_order=1 encodes=%" PRIuS " sample_pixels=%" PRIuS
              " batches=%" PRIuS " seconds=%.6f\n", search.Id(),
              screen.Encodes() - encodes_before,
              screen.SamplePixels() - pixels_before,
              screen.Batches() - batches_before,
              std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
        }
        bool automatic_first = regional_first;
        size_t control_bytes = 0, neighbor_bytes = 0;
        if (regional_first) {
          // Changing the anchor also changes later family screening. Require
          // whole-frame agreement before changing that history; cropped trees
          // can reverse the global group/context preference. These witnesses
          // are shared with the existing paired neighborhood calibration.
          auto cheap_control = Backend(params, 4);
          cheap_control.options.nb_repeats = 0.01f;
          auto cheap_neighbor = Backend(WinnerNeighborhood(params, metadata), 4);
          cheap_neighbor.options.nb_repeats = 0.01f;
          JXL_ASSIGN_OR_RETURN(control_bytes,
                               search.Refine(stage, cheap_control, false, true));
          JXL_ASSIGN_OR_RETURN(neighbor_bytes,
                               search.Refine(stage, cheap_neighbor, false, true));
          automatic_first = control_bytes <= neighbor_bytes;
        }
        if (trace) {
          std::fprintf(stderr, "lossless-route search=%" PRIuS
              " stage=5 anchor_order=1 automatic_first=%d regional_first=%d"
              " automatic_bytes=%" PRIuS " neighborhood_bytes=%" PRIuS
              " global_control_bytes=%" PRIuS " global_neighbor_bytes=%" PRIuS "\n",
              search.Id(), automatic_first, regional_first, score_total(automatic),
              score_total(neighbor), control_bytes, neighbor_bytes);
        }
        // Reuse these exact scores in the normal gate. Stage 9 still refines its
        // current winner; a rejected automatic anchor keeps the old order.
        if (automatic_first) {
          JXL_ASSIGN_OR_RETURN(size_t size, refine(stage, Backend(params, 9)));
          regular_size = std::min(regular_size, size);
          if (size != kUnknownSize) continue;
        }
      }
      if (preflight_yield && pixels < (2u << 20) &&
          SameLosslessCandidate(search.BestParamsBefore(stage), Backend(params, 7))) {
        // For compact automatic-family winners, complete the ordinary e9
        // anchor first. Its measured result can screen the neighborhood at e9
        // instead of paying for that deep challenger before the anchor exists.
        JXL_ASSIGN_OR_RETURN(size_t size,
                             search.Refine(stage, Backend(params, 9)));
        regular_size = std::min(regular_size, size);
        continue;
      }
      // Spend the middle stage on a different geometry/context model, not a
      // third consecutive effort upgrade of the same default route.
      JXL_ASSIGN_OR_RETURN(
          size_t size,
          refine(
              stage,
              WinnerNeighborhood(search.BestParamsBefore(stage), metadata),
              false, false, true));
      (void)size;
    }
    if (stage == 6 && search.Competitive(literal_size, stage, 1.35)) {
      JXL_ASSIGN_OR_RETURN(learned_size,
                           refine(stage, Literal(params, 9, 0.01f)));
    }
    if (stage == 8 && search.Competitive(learned_size, stage, 1.02)) {
      JXL_ASSIGN_OR_RETURN(refined_size,
                           refine(stage, Literal(params, 9, 0.1f)));
    }
    if (stage == 9 && refined_size != kUnknownSize &&
        static_cast<double>(refined_size) < learned_size * 0.99) {
      // The small-frame policy requires a measured gain from more training.
      JXL_ASSIGN_OR_RETURN(size_t full,
                           refine(stage, Literal(params, 9, 1.0f)));
      (void)full;
    }
    if (stage >= 8 && static_cast<size_t>(stage - 8) < analysis.ranked.size()) {
      // A higher effort promotes the next candidate from the same fixed probe
      // ranking. The measured full-frame bytes, not the probe score, can win.
      const auto& candidate = analysis.ranked[stage - 8].params;
      JXL_ASSIGN_OR_RETURN(bool accepted, confirm(stage, candidate));
      if (accepted) {
        const bool explore = stage == 8 && analysis.ranked[0].exploratory;
        JXL_ASSIGN_OR_RETURN(size_t size,
                             refine(stage, candidate, false, explore));
        (void)size;
      }
    }
    if (stage == 9 && refine_winner) {
      // Tiny crops cannot rank group geometry or cross-channel tree learning
      // on a whole frame. Explore this neighborhood of the actual winner,
      // including this stage's completed trials, rather than the crop winner.
      // The retained output and a real encoded-byte lower bound protect
      // against a misleading refinement, not a predicted compression ratio.
      JXL_ASSIGN_OR_RETURN(
          size_t size,
          refine(
              stage,
              WinnerNeighborhood(search.BestParamsBefore(stage + 1), metadata),
              false, false, true));
      (void)size;
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
                             refine(stage, deeper, false));
        (void)size;
      }
    }
  }
  JXL_RETURN_IF_ERROR(search.Finish(output, aux_out));
  return true;
}

}  // namespace jxl
