// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_lossless_probe.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

#include "lib/jxl/enc_lossless_profile.h"
#include "lib/jxl/enc_lossless_probe_pool.h"
#include "lib/jxl/enc_lossless_racing.h"
#include "lib/jxl/enc_lossless_workers.h"
#include "lib/jxl/enc_lossless_scan.h"
#include "lib/jxl/encode_internal.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/modular/transform/palette.h"

namespace jxl {
namespace {

std::vector<LosslessProbeCandidate> MakePlan(const CompressParams& input) {
  std::vector<LosslessProbeCandidate> plan;
  auto p = input;
  p.speed_tier = SpeedTier::kTortoise;
  p.options.nb_repeats = 0.1f;
  p.patches = Override::kOff;
  p.modular_group_size_shift = 3;
  p.options.predictor = Predictor::Zero;
  p.channel_colors_percent = 0;
  p.palette_colors = 70000;
  plan.push_back({"large-palette-zero", p});
  p.options.predictor = Predictor::Gradient;
  plan.push_back({"large-palette-gradient", p});

  p.palette_colors = 0;
  p.channel_colors_pre_transform_percent = 0;
  for (int rct : {0, 6}) {
    p.colorspace = rct;
    p.options.predictor = Predictor::Zero;
    plan.push_back({rct == 0 ? "raw-zero" : "ycocg-zero", p});
    p.options.predictor = Predictor::Gradient;
    plan.push_back({rct == 0 ? "raw-gradient" : "ycocg-gradient", p});
    p.options.predictor = Predictor::Weighted;
    plan.push_back({rct == 0 ? "raw-weighted" : "ycocg-weighted", p});
  }
  p.colorspace = 0;
  p.options.predictor = Predictor::Variable;
  p.options.max_properties = 2;
  plan.push_back({"raw-cross-channel", p});
  p.colorspace = 6;
  plan.push_back({"ycocg-cross-channel", p});
  const char* transforms = std::getenv("JXL_LOSSLESS_RCT_SCREEN");
  if (transforms && std::strcmp(transforms, "1") == 0) {
    for (const auto& rct : std::array<std::pair<int, const char*>, 3>{{
             {10, "rct10-cross-channel"}, {17, "rct17-cross-channel"},
             {26, "rct26-cross-channel"}}}) {
      p.colorspace = rct.first;
      plan.push_back({rct.second, p});
    }
  }
  p.options.max_properties = 0;
  for (int rct : {0, 6}) {
    p.colorspace = rct;
    p.options.predictor = Predictor::Select;
    plan.push_back({rct == 0 ? "raw-select" : "ycocg-select", p});
    p.options.predictor = Predictor::Left;
    plan.push_back({rct == 0 ? "raw-left" : "ycocg-left", p});
  }
  p.colorspace = 0;
  p.options.predictor = Predictor::Top;
  plan.push_back({"raw-top", p});
  return plan;
}

template <typename Task>
Status RunProbeTasks(size_t count, ThreadPool* pool, const Task& task) {
  // Share the caller's runner without nested runners. Dynamic assignment keeps
  // long palette/tree probes from leaving the other workers idle.
  const char* limit = std::getenv("JXL_LOSSLESS_PROBE_WORKERS");
  // The runner still limits actual threads; cap independent encoder states.
  const size_t max_workers = LosslessWorkerSetting(
      limit, LosslessWorkerLimits::FromEnvironment().tasks, 64);
  const size_t workers = std::min(max_workers, count);
  LosslessProfileScope profile(LosslessProfileCurrent(), "lossless.probe_tasks");
  profile.Set("cells", count);
  profile.Set("worker_slots", workers);
  std::atomic<size_t> next{0};
  const auto run = [&](uint32_t, size_t) -> Status {
    for (;;) {
      const size_t index = next.fetch_add(1, std::memory_order_relaxed);
      if (index >= count) return true;
      JXL_RETURN_IF_ERROR(task(index));
    }
  };
  const auto init = [&](size_t threads) -> Status {
    profile.Set("runner_threads", threads);
    return true;
  };
  return RunOnPool(pool, 0, workers, init, run, "LosslessFirstPass");
}

struct Sample {
  std::vector<uint8_t> color;
  std::unique_ptr<JxlEncoderChunkedFrameAdapter> input;
};

Status CopyRectangle(JxlChunkedFrameInputSource& source, size_t channel,
                     size_t x, size_t y, size_t width, size_t height,
                     JxlPixelFormat format, std::vector<uint8_t>* pixels) {
  size_t stride = 0;
  auto buffer = channel == 0
                    ? GetColorBuffer(source, x, y, width, height, &stride)
                    : GetExtraChannelBuffer(source, channel - 1, x, y, width,
                                            height, &stride);
  if (!buffer) return JXL_FAILURE("Missing lossless probe input channel");
  const size_t row_bytes = width * BytesPerPixel(format);
  JXL_ENSURE(row_bytes != 0 && stride >= row_bytes);
  pixels->resize(row_bytes * height);
  for (size_t row = 0; row < height; ++row) {
    std::memcpy(pixels->data() + row * row_bytes,
                static_cast<const uint8_t*>(buffer.get()) + row * stride,
                row_bytes);
  }
  return true;
}

StatusOr<Sample> CopySample(JxlChunkedFrameInputSource& source,
                            JxlPixelFormat color_format, size_t extra_channels,
                            size_t x, size_t y, size_t width, size_t height) {
  Sample sample;
  sample.input = std::make_unique<JxlEncoderChunkedFrameAdapter>(
      width, height, extra_channels);
  JXL_RETURN_IF_ERROR(CopyRectangle(source, 0, x, y, width, height,
                                    color_format, &sample.color));
  JXL_ENSURE(sample.input->SetFromBuffer(0, sample.color.data(),
                                         sample.color.size(), color_format));
  const bool interleaved_alpha =
      color_format.num_channels == 2 || color_format.num_channels == 4;
  for (size_t ec = 0; ec < extra_channels; ++ec) {
    if (ec == 0 && interleaved_alpha) continue;
    JxlPixelFormat format{};
    source.get_extra_channel_pixel_format(source.opaque, ec, &format);
    format.align = 0;
    format.num_channels = 1;
    std::vector<uint8_t> pixels;
    JXL_RETURN_IF_ERROR(
        CopyRectangle(source, ec + 1, x, y, width, height, format, &pixels));
    JXL_ENSURE(sample.input->SetFromBuffer(ec + 1, pixels.data(), pixels.size(),
                                           format));
  }
  return sample;
}

StatusOr<Sample> CopyBandAtlas(JxlChunkedFrameInputSource& source,
                              JxlPixelFormat color_format, size_t extra_channels,
                              size_t width, size_t band_height,
                              const std::vector<size_t>& rows) {
  Sample sample;
  sample.input = std::make_unique<JxlEncoderChunkedFrameAdapter>(
      width, band_height * rows.size(), extra_channels);
  const bool interleaved_alpha = color_format.num_channels == 2 ||
                                 color_format.num_channels == 4;
  for (size_t channel = 0; channel <= extra_channels; ++channel) {
    if (channel == 1 && interleaved_alpha) continue;
    auto format = color_format;
    if (channel) {
      source.get_extra_channel_pixel_format(source.opaque, channel - 1, &format);
      format.num_channels = 1;
      format.align = 0;
    }
    std::vector<uint8_t> pixels, band;
    for (size_t row : rows) {
      JXL_RETURN_IF_ERROR(CopyRectangle(source, channel, 0, row, width,
                                        band_height, format, &band));
      pixels.insert(pixels.end(), band.begin(), band.end());
    }
    if (channel == 0) {
      sample.color = std::move(pixels);
      JXL_ENSURE(sample.input->SetFromBuffer(0, sample.color.data(),
                                             sample.color.size(), format));
    } else {
      JXL_ENSURE(sample.input->SetFromBuffer(channel, pixels.data(),
                                             pixels.size(), format));
    }
  }
  return sample;
}

}  // namespace

std::vector<LosslessProbeCandidate> LosslessProbePlan(const CompressParams& params) {
  return MakePlan(params);
}

bool LosslessCalibrationAgreement(const std::vector<double>& projected,
                                   const std::vector<size_t>& full_bytes,
                                   size_t leader) {
  if (projected.size() < 2 || projected.size() != full_bytes.size() ||
      leader >= projected.size()) return false;
  for (size_t i = 0; i < projected.size(); ++i) {
    if (!(projected[i] > 0) || !std::isfinite(projected[i]) || !full_bytes[i]) return false;
    if (i != leader && !(projected[leader] < projected[i] * 0.998 &&
                        full_bytes[leader] < full_bytes[i] * 0.99)) return false;
  }
  return true;
}

std::vector<LosslessProbeCandidate> DiverseLosslessFrontier(
    std::vector<LosslessProbeCandidate> ranked) {
  const auto family = [](const CompressParams& p) {
    return std::make_tuple(p.modular_group_size_shift, p.colorspace,
        p.palette_colors > 0, p.channel_colors_percent,
        p.channel_colors_pre_transform_percent, p.options.predictor,
        p.options.max_properties, p.patches);
  };
  std::vector<size_t> representatives;
  for (size_t i = 0; i < ranked.size(); ++i) {
    auto found = std::find_if(representatives.begin(), representatives.end(),
        [&](size_t j) { return family(ranked[i].params) == family(ranked[j].params); });
    if (found == representatives.end()) {
      representatives.push_back(i);
    } else {
      const auto& previous = ranked[*found];
      // Prefer a cheaper training budget only for almost tied region scores.
      // The complete output still has to win on actual bytes.
      if (ranked[i].context_bytes <= previous.context_bytes * 1.0025 &&
          ranked[i].params.options.nb_repeats < previous.params.options.nb_repeats)
        *found = i;
    }
  }
  std::vector<uint8_t> selected(ranked.size());
  std::vector<LosslessProbeCandidate> result;
  result.reserve(ranked.size());
  for (size_t i : representatives) {
    selected[i] = 1;
    result.push_back(std::move(ranked[i]));
  }
  for (size_t i = 0; i < ranked.size(); ++i) {
    if (!selected[i]) result.push_back(std::move(ranked[i]));
  }
  return result;
}

bool SameLosslessCandidate(const CompressParams& a, const CompressParams& b) {
  const auto key = [](const CompressParams& p) {
    return std::make_tuple(p.speed_tier, p.modular_group_size_shift, p.colorspace,
                           p.palette_colors, p.channel_colors_percent,
                           p.channel_colors_pre_transform_percent, p.patches,
                           p.options.predictor, p.options.nb_repeats,
                           p.options.max_properties, p.lz77_careful_first_pass,
                           p.lossless_bucket_matcher, p.lossless_precise_entropy,
                           p.lossless_probe_lz77);
  };
  return key(a) == key(b);
}

std::vector<double> ShrinkLosslessCalibration(
    const std::vector<CompressParams>& plans,
    const std::vector<size_t>& full_bytes,
    const std::vector<double>& projected,
    const std::vector<size_t>& independent_regions) {
  auto result = projected;
  if (plans.size() != full_bytes.size() || plans.size() != projected.size() ||
      plans.size() != independent_regions.size()) return result;
  for (size_t i = 0; i < plans.size(); ++i) {
    if (!full_bytes[i] || projected[i] <= 0 || plans[i].palette_colors != 0 ||
        plans[i].colorspace < 0 || plans[i].options.predictor != Predictor::Variable)
      continue;
    std::vector<double> ratios;
    for (size_t j = 0; j < plans.size(); ++j) {
      auto sibling = plans[j];
      sibling.colorspace = plans[i].colorspace;
      if (plans[j].colorspace < 0 || !full_bytes[j] || projected[j] <= 0 ||
          !SameLosslessCandidate(plans[i], sibling)) continue;
      ratios.push_back(projected[j] / full_bytes[j]);
    }
    if (ratios.size() < 2) continue;
    std::sort(ratios.begin(), ratios.end());
    const double median = (ratios[(ratios.size() - 1) / 2] +
                           ratios[ratios.size() / 2]) * 0.5;
    const double confidence = double(independent_regions[i]) /
        (double(independent_regions[i]) + 4.0);
    result[i] = full_bytes[i] * (median + confidence *
        (projected[i] / full_bytes[i] - median));
  }
  return result;
}

double LosslessCalibrationFeedback::SiblingEstimate(
    const CompressParams& candidate, const std::vector<size_t>& scores) const {
  if (!full_bytes || deep.size() < 3 || cheap.size() != deep.size() ||
      scores.size() != deep.size() || candidate.palette_colors != 0 ||
      candidate.colorspace < 0 || params.colorspace < 0 ||
      candidate.options.predictor != Predictor::Variable) return 0;
  auto sibling = candidate;
  sibling.colorspace = params.colorspace;
  if (!SameLosslessCandidate(sibling, params)) return 0;
  // Do not transfer a rejection across a meaningful local gain or across a
  // change of predictor, group geometry, training budget or palette semantics.
  for (size_t i = 0; i < scores.size(); ++i) {
    if (scores[i] < deep[i] * 0.95) return 0;
  }
  const size_t denominator = std::accumulate(cheap.begin(), cheap.end(), size_t{0});
  return denominator ? double(full_bytes) *
      std::accumulate(scores.begin(), scores.end(), size_t{0}) / denominator : 0;
}

struct LosslessCandidateScreen::Impl {
  const CodecMetadata& metadata;
  JxlEncoderChunkedFrameAdapter& input;
  ThreadPool* pool;
  LosslessProbeEncoder encode;
  const LosslessScan* scan;
  struct Cached {
    CompressParams params;
    std::vector<size_t> bytes;
  };
  std::vector<Sample> samples;
  std::vector<Cached> cache;
  size_t encodes = 0;
  size_t sample_pixels = 0;
  size_t batches = 0;
  size_t context_side = 0;
  bool full_width_strips = false;
  const bool data_parallel = [] {
    const char* value = std::getenv("JXL_LOSSLESS_PROBE_DATA_PARALLEL");
    return value && std::strcmp(value, "1") == 0;
  }();
  const bool cooperative = [] {
    const char* value = std::getenv("JXL_LOSSLESS_PROBE_COOPERATIVE");
    return value && std::strcmp(value, "1") == 0;
  }();
  const float training_cap = [] {
    const char* value = std::getenv("JXL_LOSSLESS_PROBE_TRAINING");
    if (value && std::strcmp(value, "1") == 0) return 0.01f;
    if (value && std::strcmp(value, "3") == 0) return 0.03f;
    if (value && std::strcmp(value, "10") == 0) return 0.1f;
    return 0.0f;
  }();
  const bool adaptive_coverage = [] {
    const char* value = std::getenv("JXL_LOSSLESS_PREFLIGHT_YIELD");
    return value && std::strcmp(value, "1") == 0;
  }();
  const bool batch_regions = [] {
    const char* value = std::getenv("JXL_LOSSLESS_BATCH_PROBES");
    return !value || std::strcmp(value, "1") == 0;
  }();

  bool Available() const {
    if (!encode || input.IsJPEG() || input.StreamingInput() ||
        uint64_t{input.xsize} * input.ysize <
            (adaptive_coverage ? (1u << 17) : (1u << 19)) ||
        metadata.m.num_extra_channels > 8) return false;
    for (const auto& ec : metadata.m.extra_channel_info) {
      if (ec.dim_shift != 0) return false;
    }
    return true;
  }

  Status Init() {
    if (!samples.empty()) return true;
    auto source = input.GetInputSource();
    JxlPixelFormat format{};
    source.get_color_channels_pixel_format(source.opaque, &format);
    format.align = 0;
    const bool large = uint64_t{input.xsize} * input.ysize >= (2u << 20);
    const bool small = uint64_t{input.xsize} * input.ysize < (1u << 19);
    const bool compact = uint64_t{input.xsize} * input.ysize < (1u << 18);
    const size_t width = std::min<size_t>(compact ? 128 : small ? 256 : 1024, input.xsize);
    const size_t height = std::min<size_t>(compact ? 8 : small ? 16 : large ? 64 : 32, input.ysize);
    std::vector<std::tuple<size_t, size_t, size_t, size_t>> regions;
    if (full_width_strips) {
      // Preserve row distances and horizontal group boundaries. These are
      // original-resolution bands, not a resized image or interleaved rows.
      const size_t ch = std::min<size_t>(input.ysize,
          std::max<size_t>(8, std::min<size_t>(32, 65536 / input.xsize)));
      if (scan && scan->Available()) {
        std::vector<size_t> active;
        for (size_t i = 0; i < scan->tiles.size(); ++i)
          if (scan->tiles[i].horizontal || scan->tiles[i].vertical) active.push_back(i);
        if (active.empty()) active.push_back(scan->tiles.size() / 2);
        std::stable_sort(active.begin(), active.end(), [&](size_t a, size_t b) {
          return scan->tiles[a].Activity() > scan->tiles[b].Activity();
        });
        for (size_t i = 0; i < 3; ++i) {
          const size_t lo = i * active.size() / 3;
          const size_t hi = std::max(lo + 1, (i + 1) * active.size() / 3);
          size_t best = active[lo], separation = 0;
          for (size_t j = lo; j < hi; ++j) {
            const size_t y = scan->tiles[active[j]].anchor_y;
            size_t distance = input.ysize;
            for (const auto& r : regions) {
              const size_t center = std::get<1>(r) + ch / 2;
              distance = std::min(distance, y > center ? y - center : center - y);
            }
            if (distance > separation) { best = active[j]; separation = distance; }
          }
          const size_t y = scan->tiles[best].anchor_y;
          regions.emplace_back(0, std::min(input.ysize - ch, y > ch / 2 ? y - ch / 2 : 0),
                               input.xsize, ch);
        }
      } else {
        for (size_t i = 0; i < 3; ++i)
          regions.emplace_back(0, i * (input.ysize - ch) / 2, input.xsize, ch);
      }
    } else if (context_side) {
      const size_t cw = std::min(context_side, input.xsize);
      const size_t ch = std::min(context_side, input.ysize);
      for (size_t i = 0; i < 3; ++i) {
        regions.emplace_back(i * (input.xsize - cw) / 2,
                             i * (input.ysize - ch) / 2, cw, ch);
      }
    } else if (scan && scan->Available()) {
      const size_t context_limit = large ? 256 : LosslessWorkerSetting(
          std::getenv("JXL_LOSSLESS_COMPACT_CONTEXT_SIDE"), 256, 256);
      for (const auto& r : scan->Regions()) {
        if (r.width == r.height && r.width > context_limit) {
          // Keep the original center and pixels, shrinking only the low-
          // activity context square. Long horizontal probe bands are unchanged.
          regions.emplace_back(r.x + (r.width - context_limit) / 2,
              r.y + (r.height - context_limit) / 2, context_limit, context_limit);
        } else {
          regions.emplace_back(r.x, r.y, r.width, r.height);
        }
      }
    } else {
      for (size_t i = 0; i < 3; ++i) {
        regions.emplace_back(i * (input.xsize - width) / 2,
                             i * (input.ysize - height) / 2, width, height);
      }
      const size_t side_x = std::min<size_t>(compact ? 32 : small ? 64 : 256, input.xsize);
      const size_t side_y = std::min<size_t>(compact ? 32 : small ? 64 : 256, input.ysize);
      regions.emplace_back((input.xsize - side_x) / 2,
                           (input.ysize - side_y) / 2, side_x, side_y);
    }
    std::sort(regions.begin(), regions.end());
    regions.erase(std::unique(regions.begin(), regions.end()), regions.end());
    const char* atlas = std::getenv("JXL_LOSSLESS_PROBE_ATLAS");
    const bool uniform_atlas = atlas && std::strcmp(atlas, "3") == 0;
    const bool routed_atlas = atlas && std::strcmp(atlas, "4") == 0;
    const bool eight_bit = metadata.m.bit_depth.bits_per_sample <= 8 &&
        metadata.m.bit_depth.exponent_bits_per_sample == 0 &&
        std::all_of(metadata.m.extra_channel_info.begin(), metadata.m.extra_channel_info.end(),
            [](const ExtraChannelInfo& ec) {
              return ec.bit_depth.bits_per_sample <= 8 && ec.bit_depth.exponent_bits_per_sample == 0;
            });
    const bool compact_bands = eight_bit && scan && scan->PreferCompactBandAtlas();
    const bool atlas_enabled = atlas && (uniform_atlas || std::strcmp(atlas, "1") == 0 ||
        (routed_atlas && compact_bands) ||
        ((std::strcmp(atlas, "2") == 0 || routed_atlas) && scan &&
         scan->palette_color_lower_bound > 10000u + palette_internal::kImplicitPaletteSize));
    if (!context_side && !full_width_strips && atlas_enabled) {
      // Shared training and dictionary across representative original-pixel
      // bands. Horizontal distances stay intact; vertical seams are artificial,
      // so these remain proxy scores, never a bound on the complete image.
      const size_t band_count = uniform_atlas ? std::min<size_t>(8, input.ysize) : 1;
      const char* band_value = std::getenv("JXL_LOSSLESS_PROBE_BAND_PIXELS");
      const bool automatic_bands = eight_bit &&
          ((!band_value && routed_atlas) || (band_value && std::strcmp(band_value, "auto") == 0));
      const bool tone = scan && (scan->LikelyContinuousTone() ||
          (large && scan->LikelySparseContinuousTone() &&
           scan->palette_color_lower_bound > 70000u + palette_internal::kImplicitPaletteSize));
      const size_t band_pixels = automatic_bands
          ? (compact_bands ? 8192 : (large && tone ? 32768 : 65536))
          : LosslessWorkerSetting(band_value, 65536, 262144);
      const size_t bh = std::min<size_t>(input.ysize,
          uniform_atlas
              ? std::max<size_t>(1, std::min<size_t>(16, 131072 / input.xsize / band_count))
              : std::max<size_t>(8, std::min<size_t>(32, band_pixels / input.xsize)));
      std::vector<size_t> rows;
      if (uniform_atlas) {
        // Equal-area strata avoid giving one dense center crop most of the
        // vote. Copy contiguous rows, preserving each band's horizontal runs.
        for (size_t i = 0; i < band_count; ++i) {
          const size_t center = (2 * i + 1) * input.ysize / (2 * band_count);
          rows.push_back(std::min(input.ysize - bh, center > bh / 2 ? center - bh / 2 : 0));
        }
      } else {
        for (const auto& region : regions) {
          const size_t center = std::get<1>(region) + std::get<3>(region) / 2;
          rows.push_back(std::min(input.ysize - bh, center > bh / 2 ? center - bh / 2 : 0));
        }
      }
      std::sort(rows.begin(), rows.end());
      rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
      JXL_ASSIGN_OR_RETURN(auto sample, CopyBandAtlas(source, format,
          metadata.m.num_extra_channels, input.xsize, bh, rows));
      samples.push_back(std::move(sample));
      return true;
    }
    for (const auto& region : regions) {
      JXL_ASSIGN_OR_RETURN(
          Sample sample,
          CopySample(source, format, metadata.m.num_extra_channels,
                     std::get<0>(region), std::get<1>(region),
                     std::get<2>(region), std::get<3>(region)));
      samples.push_back(std::move(sample));
    }
    return true;
  }

  size_t Index(const CompressParams& params) {
    auto candidate = params;
    // Cropping can make an impossible whole-frame palette fit. Match that
    // global eligibility in the proxy. Group-local palettes can still differ,
    // so this does not prove that a complete candidate loses.
    if (scan && candidate.IsLossless() && candidate.palette_colors > 0 &&
        scan->palette_color_lower_bound > size_t(candidate.palette_colors) +
            palette_internal::kImplicitPaletteSize) {
      candidate.palette_colors = 0;
    }
    // This changes the sample model, never the full-frame plan. Keep fixed
    // predictor palette budgets separate: their training can select contexts.
    if (training_cap && candidate.IsLossless() &&
        (candidate.options.predictor == Predictor::Variable ||
         candidate.options.predictor == kUndefinedPredictor)) {
      candidate.options.nb_repeats =
          std::min(candidate.options.nb_repeats, training_cap);
    }
    for (size_t i = 0; i < cache.size(); ++i) {
      if (SameLosslessCandidate(candidate, cache[i].params)) return i;
    }
    candidate.lossless_trial_budget = nullptr;
    candidate.lossless_probe_runner = nullptr;
    cache.push_back({candidate, std::vector<size_t>(
                                    samples.size(),
                                    std::numeric_limits<size_t>::max())});
    return cache.size() - 1;
  }

  Status Evaluate(size_t candidate, size_t tile, ThreadPool* runner = nullptr) {
    auto& cached = cache[candidate];
    auto sample_metadata = metadata;
    auto& sample = *samples[tile].input;
    JXL_RETURN_IF_ERROR(sample_metadata.size.Set(sample.xsize, sample.ysize));
    sample_metadata.m.have_animation = false;
    sample_metadata.m.have_preview = false;
    auto params = cached.params;
    params.lossless_probe_runner = runner;
    JXL_ASSIGN_OR_RETURN(cached.bytes[tile],
                         encode(params, sample_metadata, sample));
    return true;
  }

  Status Round(const std::vector<size_t>& candidates, size_t tile) {
    std::vector<size_t> pending;
    for (size_t index : candidates) {
      if (cache[index].bytes[tile] == std::numeric_limits<size_t>::max())
        pending.push_back(index);
    }
    if (pending.empty()) return true;
    std::sort(pending.begin(), pending.end());
    pending.erase(std::unique(pending.begin(), pending.end()), pending.end());
    ++batches;
    JXL_RETURN_IF_ERROR(RunProbeTasks(
        pending.size(), pool, [&](size_t i) { return Evaluate(pending[i], tile); }));
    encodes += pending.size();
    sample_pixels += pending.size() * samples[tile].input->xsize *
                     samples[tile].input->ysize;
    return true;
  }

  Status Rounds(const std::vector<size_t>& candidates,
                const std::vector<size_t>& tiles, size_t begin, size_t end) {
    if (!batch_regions) {
      for (size_t i = begin; i < end; ++i)
        JXL_RETURN_IF_ERROR(Round(candidates, tiles[i]));
      return true;
    }
    // There is no eligibility decision within this batch. Queue all required
    // regions so a slow candidate cannot hold the other workers at a barrier.
    // Each cache cell is written by exactly one task, and cache never grows
    // while workers run. Selection only consumes completed, comparable totals.
    std::vector<std::pair<size_t, size_t>> pending;
    for (size_t i = begin; i < end; ++i) {
      for (size_t candidate : candidates) {
        if (cache[candidate].bytes[tiles[i]] ==
            std::numeric_limits<size_t>::max()) {
          pending.emplace_back(candidate, tiles[i]);
        }
      }
    }
    if (pending.empty()) return true;
    std::sort(pending.begin(), pending.end());
    pending.erase(std::unique(pending.begin(), pending.end()), pending.end());
    LosslessProfileScope batch_profile(LosslessProfileCurrent(),
                                       "lossless.screen_batch");
    const char* order_value = std::getenv("JXL_LOSSLESS_PROBE_LPT");
    const bool longest_first = order_value && std::strcmp(order_value, "1") == 0;
    if (longest_first) {
      const auto work = [&](const auto& task) {
        const auto& p = cache[task.first].params;
        const auto& sample = *samples[task.second].input;
        const double predictors = p.options.predictor == Predictor::Variable ? 8 :
            p.options.predictor == kUndefinedPredictor || p.options.predictor == Predictor::Best ? 2 : 1;
        return double(sample.xsize) * sample.ysize *
            (1.0 + predictors * p.options.nb_repeats * 10.0);
      };
      std::stable_sort(pending.begin(), pending.end(),
          [&](const auto& a, const auto& b) { return work(a) > work(b); });
    }
    batch_profile.Set("cells", pending.size());
    batch_profile.Set("longest_first", longest_first);
    ++batches;
    if (cooperative && pool) {
      LosslessProbePool arena;
      const auto limits = LosslessWorkerLimits::FromEnvironment();
      const size_t roots = limits.Tasks(pending.size(), 32u << 20);
      JXL_RETURN_IF_ERROR(arena.Run(pending.size(), pool, limits.tasks, roots,
          [&](size_t i, ThreadPool* nested) {
            return Evaluate(pending[i].first, pending[i].second, nested);
          }));
    } else {
      JXL_RETURN_IF_ERROR(RunProbeTasks(
          pending.size(), pool, [&](size_t i) {
            return Evaluate(pending[i].first, pending[i].second);
          }));
    }
    encodes += pending.size();
    for (const auto& task : pending) {
      const auto& sample = *samples[task.second].input;
      sample_pixels += sample.xsize * sample.ysize;
    }
    return true;
  }
};

LosslessCandidateScreen::LosslessCandidateScreen(
    const CodecMetadata& metadata, JxlEncoderChunkedFrameAdapter& input,
    ThreadPool* pool, const LosslessProbeEncoder& encode, const LosslessScan* scan,
    size_t context_side, bool full_width_strips)
    : impl_(new Impl{metadata, input, pool, encode, scan, {}, {}}) {
  impl_->context_side = context_side;
  impl_->full_width_strips = full_width_strips;
}

LosslessCandidateScreen::~LosslessCandidateScreen() = default;

bool LosslessCandidateScreen::Available() const { return impl_->Available(); }
size_t LosslessCandidateScreen::Encodes() const { return impl_->encodes; }
size_t LosslessCandidateScreen::SamplePixels() const {
  return impl_->sample_pixels;
}
size_t LosslessCandidateScreen::Batches() const { return impl_->batches; }

StatusOr<std::vector<size_t>> LosslessCandidateScreen::Score(
    const CompressParams& params) {
  if (!Available()) return std::vector<size_t>{};
  JXL_RETURN_IF_ERROR(impl_->Init());
  const size_t candidate = impl_->Index(params);
  std::vector<size_t> pending;
  for (size_t i = 0; i < impl_->samples.size(); ++i) {
    if (impl_->cache[candidate].bytes[i] == std::numeric_limits<size_t>::max())
      pending.push_back(i);
  }
  if (!pending.empty()) {
    ++impl_->batches;
    bool large = false;
    for (size_t i : pending) {
      const auto& sample = *impl_->samples[i].input;
      large |= uint64_t{sample.xsize} * sample.ysize >= 65536;
    }
    if (impl_->cooperative && impl_->pool) {
      LosslessProbePool arena;
      const auto limits = LosslessWorkerLimits::FromEnvironment();
      JXL_RETURN_IF_ERROR(arena.Run(pending.size(), impl_->pool, limits.tasks,
          limits.Tasks(pending.size(), 32u << 20), [&](size_t i, ThreadPool* nested) {
            return impl_->Evaluate(candidate, pending[i], nested);
          }));
    } else if (impl_->data_parallel && impl_->pool && large && pending.size() <= 4) {
      // The outer runner is idle here. Samples execute sequentially and borrow
      // that same runner, so neither overlapping calls nor nested pools occur.
      for (size_t tile : pending)
        JXL_RETURN_IF_ERROR(impl_->Evaluate(candidate, tile, impl_->pool));
    } else {
      JXL_RETURN_IF_ERROR(RunProbeTasks(
          pending.size(), impl_->pool,
          [&](size_t i) { return impl_->Evaluate(candidate, pending[i]); }));
    }
  }
  impl_->encodes += pending.size();
  for (size_t i : pending) {
    const auto& sample = impl_->samples[i];
    impl_->sample_pixels += sample.input->xsize * sample.input->ysize;
  }
  return std::vector<size_t>(impl_->cache[candidate].bytes);
}

StatusOr<std::vector<LosslessProbeCandidate>> LosslessCandidateScreen::Rank(
    std::vector<LosslessProbeCandidate> candidates,
    const CompressParams& incumbent, bool fast_first_round) {
  if (!Available() || candidates.empty()) return candidates;
  std::vector<LosslessProbeCandidate> unique;
  for (auto& candidate : candidates) {
    const bool duplicate = std::any_of(
        unique.begin(), unique.end(), [&](const auto& previous) {
          return SameLosslessCandidate(candidate.params, previous.params);
        });
    if (!duplicate) unique.push_back(std::move(candidate));
  }
  candidates = std::move(unique);
  auto ordering_control = incumbent;
  const char* control_env = std::getenv("JXL_LOSSLESS_CHEAP_PROBE_CONTROL");
  const uint64_t pixels = uint64_t{impl_->input.xsize} * impl_->input.ysize;
  const bool compact_control = pixels >= (1u << 19) && pixels < (2u << 20);
  if (fast_first_round && incumbent.IsLossless() && control_env &&
      (std::strcmp(control_env, "1") == 0 ||
       (std::strcmp(control_env, "2") == 0 && compact_control))) {
    // This result orders regions by complexity; it is not an incumbent size
    // bound. Expensive candidate scores are still measured at their own tier.
    ordering_control.speed_tier = std::max(ordering_control.speed_tier,
        impl_->scan ? static_cast<SpeedTier>(6) : SpeedTier::kSquirrel);
    ordering_control.options.nb_repeats =
        std::min(ordering_control.options.nb_repeats, 0.01f);
  }
  JXL_ASSIGN_OR_RETURN(auto control, Score(ordering_control));
  std::vector<size_t> tiles(control.size());
  std::iota(tiles.begin(), tiles.end(), 0);
  std::stable_sort(tiles.begin(), tiles.end(), [&](size_t a, size_t b) {
    const auto& sa = *impl_->samples[a].input;
    const auto& sb = *impl_->samples[b].input;
    return static_cast<double>(control[a]) / (sa.xsize * sa.ysize) >
           static_cast<double>(control[b]) / (sb.xsize * sb.ysize);
  });
  // Hard and median-complexity regions first, rather than two blank corners.
  if (tiles.size() > 2) std::swap(tiles[1], tiles[tiles.size() / 2]);
  std::vector<size_t> indices;
  for (const auto& candidate : candidates) {
    auto screening = candidate.params;
    if (fast_first_round) {
      // Keep explicit representation options, but avoid optimal LZ77 parsing
      // and deep tree learning. Auto heuristics may differ at this speed tier;
      // these scores only select a shortlist, never replace exact evaluation.
      screening.speed_tier = std::max(screening.speed_tier, SpeedTier::kSquirrel);
      if (impl_->scan) {
        screening.speed_tier = std::max(screening.speed_tier,
                                         static_cast<SpeedTier>(6));
      }
      screening.options.nb_repeats =
          std::min(screening.options.nb_repeats, 0.01f);
    }
    indices.push_back(impl_->Index(screening));
  }
  // Different training depths may collapse to the same cheap model. Score that
  // model once per region; never let concurrent jobs write the same cache slot.
  auto unique_indices = indices;
  std::sort(unique_indices.begin(), unique_indices.end());
  unique_indices.erase(std::unique(unique_indices.begin(), unique_indices.end()),
                       unique_indices.end());
  const char* race_env = std::getenv("JXL_LOSSLESS_STATISTICAL_RACE");
  const bool statistical = incumbent.IsLossless() && race_env &&
      std::strcmp(race_env, "1") == 0 && tiles.size() >= 3;
  const size_t initial = std::min<size_t>(2, tiles.size());
  JXL_RETURN_IF_ERROR(impl_->Rounds(unique_indices, tiles, 0, initial));
  std::vector<size_t> order(candidates.size());
  std::iota(order.begin(), order.end(), 0);
  const auto partial = [&](size_t i) {
    size_t total = 0;
    for (size_t j = 0; j < initial; ++j)
      total += impl_->cache[indices[i]].bytes[tiles[j]];
    return total;
  };
  std::stable_sort(order.begin(), order.end(),
                   [&](size_t a, size_t b) { return partial(a) < partial(b); });
  std::vector<size_t> survivors;
  bool conventional = false, literal = false;
  const size_t shortlist = fast_first_round ? 4 : 6;
  const size_t cutoff = partial(order[std::min(shortlist, order.size()) - 1]);
  for (size_t i : order) {
    const auto& p = candidates[i].params;
    const bool regular = p.options.predictor == kUndefinedPredictor;
    const bool palette = p.options.predictor == Predictor::Zero &&
                         p.palette_colors > 0;
    const bool cross_channel = p.options.predictor == Predictor::Variable;
    // Preserve distinct representations when prefix evidence is ambiguous.
    const bool close = fast_first_round && partial(i) <= cutoff * 1.03;
    if (survivors.size() < shortlist || close || (regular && !conventional) ||
        (palette && !literal) || cross_channel) {
      survivors.push_back(i);
      conventional |= regular;
      literal |= palette;
    }
  }
  if (statistical) {
    LosslessProfileScope race(LosslessProfileCurrent(), "lossless.statistical_race");
    const size_t before = survivors.size(), encodes_before = Encodes();
    const auto scores = [&](size_t i, size_t count) {
      std::vector<double> result;
      for (size_t t = 0; t < count; ++t) {
        result.push_back(impl_->cache[indices[i]].bytes[tiles[t]]);
      }
      return result;
    };
    bool worthwhile = false, regular_seen = false, palette_seen = false;
    const auto prefix_reference = scores(survivors.front(), initial);
    for (size_t rank = 0; rank < survivors.size(); ++rank) {
      const size_t i = survivors[rank];
      const auto& p = candidates[i].params;
      const bool regular = p.options.predictor == kUndefinedPredictor;
      const bool palette = p.options.predictor == Predictor::Zero &&
                           p.palette_colors > 0;
      const bool protected_candidate = rank < 2 ||
          (regular && !regular_seen) || (palette && !palette_seen) ||
          p.options.predictor == Predictor::Variable;
      regular_seen |= regular;
      palette_seen |= palette;
      const auto evidence = PairedLosslessEvidence(
          scores(i, initial), prefix_reference, 0);
      // This only skips a pruning attempt, never a candidate. Ambiguous
      // prefixes go directly to the existing full-region evaluation.
      worthwhile |= !protected_candidate && evidence.samples >= 2 &&
          evidence.minimum > (fast_first_round ? 0.10 : 0.05);
    }
    // Respect the existing frontier budget. Statistical uncertainty is not a
    // reason to promote every previously deferred representation to a full trial.
    for (size_t evaluated = initial; worthwhile && evaluated < tiles.size();) {
      std::vector<size_t> pending;
      for (size_t i : survivors) pending.push_back(indices[i]);
      std::sort(pending.begin(), pending.end());
      pending.erase(std::unique(pending.begin(), pending.end()), pending.end());
      JXL_RETURN_IF_ERROR(impl_->Rounds(pending, tiles, evaluated, evaluated + 1));
      ++evaluated;
      const auto sum = [&](size_t i) {
        double result = 0;
        for (double value : scores(i, evaluated)) result += value;
        return result;
      };
      std::stable_sort(survivors.begin(), survivors.end(),
                       [&](size_t a, size_t b) { return sum(a) < sum(b); });
      const auto reference = scores(survivors.front(), evaluated);
      std::vector<size_t> keep;
      bool regular_kept = false, palette_kept = false;
      for (size_t i : survivors) {
        const auto& p = candidates[i].params;
        const bool regular = p.options.predictor == kUndefinedPredictor;
        const bool palette = p.options.predictor == Predictor::Zero &&
                             p.palette_colors > 0;
        const bool cross = p.options.predictor == Predictor::Variable;
        const auto evidence = PairedLosslessEvidence(
            scores(i, evaluated), reference, fast_first_round ? 0.08 : 0.03);
        if (keep.size() < 2 || !evidence.ClearlyWorse(0.02) ||
            (regular && !regular_kept) || (palette && !palette_kept) || cross) {
          keep.push_back(i);
          regular_kept |= regular;
          palette_kept |= palette;
        }
      }
      const bool progress = keep.size() < survivors.size();
      survivors = std::move(keep);
      if (!progress) break;
    }
    race.Set("candidates", before);
    race.Set("survivors", survivors.size());
    race.Set("pruned", before - survivors.size());
    race.Set("encodes", Encodes() - encodes_before);
    race.Set("screening_skipped", !worthwhile);
  }
  std::vector<size_t> remaining;
  for (size_t i : survivors) {
    // Exact scores have a different cache key from their cheap proxies. Never
    // combine a cheap prefix with exact suffixes or expose it as encoded size.
    indices[i] = impl_->Index(candidates[i].params);
    remaining.push_back(indices[i]);
  }
  JXL_RETURN_IF_ERROR(impl_->Rounds(remaining, tiles,
                                   fast_first_round ? 0 : initial, tiles.size()));
  std::vector<LosslessProbeCandidate> ranked;
  for (size_t i : survivors) {
    const auto& scores = impl_->cache[indices[i]].bytes;
    candidates[i].context_bytes =
        std::accumulate(scores.begin(), scores.end(), size_t{0});
    ranked.push_back(std::move(candidates[i]));
  }
  std::stable_sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
    return a.context_bytes < b.context_bytes;
  });
  return ranked;
}

StatusOr<std::vector<LosslessProbeCandidate>>
LosslessCandidateScreen::SmallerGroups(
    const std::vector<LosslessProbeCandidate>& ranked) {
  std::vector<LosslessProbeCandidate> result;
  if (!Available() || ranked.empty() || ranked.front().context_bytes == 0)
    return result;
  size_t tested = 0;
  for (const auto& candidate : ranked) {
    if (candidate.context_bytes > ranked.front().context_bytes * 1.02) break;
    if (candidate.params.palette_colors != 0 ||
        candidate.params.options.predictor == kUndefinedPredictor ||
        candidate.params.modular_group_size_shift != 3) continue;
    if (tested++ == 2) break;
    auto smaller = candidate;
    smaller.params.modular_group_size_shift = 2;
    JXL_ASSIGN_OR_RETURN(auto scores, Score(smaller.params));
    smaller.context_bytes =
        std::accumulate(scores.begin(), scores.end(), size_t{0});
    if (smaller.context_bytes < candidate.context_bytes * 0.995) {
      result.push_back(std::move(smaller));
    }
  }
  std::stable_sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
    return a.context_bytes < b.context_bytes;
  });
  return result;
}

bool LosslessCandidateScreen::Promising(
    const std::vector<size_t>& candidate, const std::vector<size_t>& incumbent) {
  if (candidate.empty() || candidate.size() != incumbent.size()) return true;
  const double total = std::accumulate(candidate.begin(), candidate.end(), 0.0);
  const double control = std::accumulate(incumbent.begin(), incumbent.end(), 0.0);
  if (total < control * 0.998) return true;
  for (size_t i = 0; i < candidate.size(); ++i) {
    if (candidate[i] < incumbent[i] * 0.98) return true;
  }
  return false;
}

StatusOr<LosslessProbeResult> LosslessCandidateScreen::Analyze(
    const CompressParams& params, const CompressParams& incumbent) {
  LosslessProbeResult result;
  const size_t before = Encodes(), pixels_before = SamplePixels();
  auto plan = MakePlan(params);
  result.candidates = plan.size();
  for (const auto& candidate : plan) {
    if (candidate.params.options.predictor == Predictor::Variable) {
      result.exploratory.push_back(candidate);
      result.exploratory.back().exploratory = true;
    }
  }
  JXL_ASSIGN_OR_RETURN(auto ranked, Rank(std::move(plan), incumbent, true));
  JXL_ASSIGN_OR_RETURN(auto control, Score(incumbent));
  result.samples = control.size();
  result.control_bytes = std::accumulate(control.begin(), control.end(), size_t{0});
  result.small_control_bytes = result.control_bytes;
  result.shortlisted = ranked.size();
  for (auto& candidate : ranked) {
    JXL_ASSIGN_OR_RETURN(auto scores, Score(candidate.params));
    if (Promising(scores, control) ||
        candidate.context_bytes <= result.control_bytes * 1.01) {
      candidate.bytes = candidate.small_bytes = candidate.context_bytes;
      result.ranked.push_back(std::move(candidate));
    }
  }
  result.encodes = Encodes() - before;
  result.sample_pixels = SamplePixels() - pixels_before;
  return result;
}

StatusOr<LosslessProbeResult> ProbeLosslessCandidates(
    const CompressParams& params, const CodecMetadata& metadata,
    JxlEncoderChunkedFrameAdapter& input, ThreadPool* pool,
    const LosslessProbeEncoder& encode, const CompressParams* incumbent,
    bool feedback_bounds) {
  LosslessProbeResult result;
  // Small frames already have cheap full-frame predictor trials. Avoid
  // multiplying work or allocating many planes for unusual extra-channel data.
  if (!encode || input.IsJPEG() || input.StreamingInput() ||
      uint64_t{input.xsize} * input.ysize <= 4096 ||
      metadata.m.num_extra_channels > 8) {
    return result;
  }
  for (const auto& ec : metadata.m.extra_channel_info) {
    if (ec.dim_shift != 0) return result;
  }
  auto source = input.GetInputSource();
  JxlPixelFormat color_format{};
  source.get_color_channels_pixel_format(source.opaque, &color_format);
  color_format.align = 0;
  const size_t width = std::min<size_t>(64, input.xsize);
  const size_t height = std::min<size_t>(64, input.ysize);
  CodecMetadata sample_metadata = metadata;
  JXL_RETURN_IF_ERROR(sample_metadata.size.Set(width, height));
  sample_metadata.m.have_animation = false;
  sample_metadata.m.have_preview = false;

  std::vector<Sample> samples;
  std::vector<std::pair<size_t, size_t>> positions;
  for (size_t gy = 0; gy < 3; ++gy) {
    for (size_t gx = 0; gx < 3; ++gx) {
      const size_t x = uint64_t{2 * gx + 1} * (input.xsize - width) / 6;
      const size_t y = uint64_t{2 * gy + 1} * (input.ysize - height) / 6;
      const auto position = std::make_pair(x, y);
      if (std::find(positions.begin(), positions.end(), position) !=
          positions.end())
        continue;
      positions.push_back(position);
      JXL_ASSIGN_OR_RETURN(
          Sample sample,
          CopySample(source, color_format, metadata.m.num_extra_channels, x, y,
                     width, height));
      samples.push_back(std::move(sample));
    }
  }
  auto plan = MakePlan(params);
  result.samples = samples.size();
  result.candidates = plan.size();
  auto control = incumbent ? *incumbent : params;
  if (!incumbent) {
    control.speed_tier = SpeedTier::kTortoise;
    control.options.nb_repeats = 0.1f;
    control.patches = Override::kOff;
  }
  std::vector<size_t> control_sizes(samples.size());
  JXL_RETURN_IF_ERROR(
      RunProbeTasks(samples.size(), pool, [&](size_t tile) -> Status {
        JXL_ASSIGN_OR_RETURN(
            control_sizes[tile],
            encode(control, sample_metadata, *samples[tile].input));
        return true;
      }));
  result.control_bytes =
      std::accumulate(control_sizes.begin(), control_sizes.end(), size_t{0});
  result.small_control_bytes = result.control_bytes;

  // Stratify by actual compressibility rather than only looking at a fixed
  // diagonal. The difficult, intermediate and easy regions all get a vote.
  std::vector<size_t> tile_order(samples.size());
  std::iota(tile_order.begin(), tile_order.end(), 0);
  std::stable_sort(
      tile_order.begin(), tile_order.end(),
      [&](size_t a, size_t b) { return control_sizes[a] < control_sizes[b]; });
  result.screen_samples = std::min<size_t>(3, samples.size());
  std::vector<uint8_t> screened(samples.size(), 0);
  for (size_t i = 0; i < result.screen_samples; ++i) {
    screened[tile_order[(2 * i + 1) * samples.size() /
                        (2 * result.screen_samples)]] = 1;
  }
  LosslessProfileScope screen_profile(LosslessProfileCurrent(),
                                      "lossless.screen_tiles");
  screen_profile.Set("cells", plan.size() * result.screen_samples);
  screen_profile.Set("feedback_bounds", feedback_bounds);
  constexpr size_t kShortlist = 6;
  constexpr size_t kUnmeasured = std::numeric_limits<size_t>::max();
  std::array<size_t, kShortlist> best_complete;
  best_complete.fill(kUnmeasured);
  std::atomic<size_t> cutoff{kUnmeasured};
  std::mutex completed_mutex;
  std::vector<size_t> screen_encodes(plan.size(), 0);
  std::vector<uint8_t> screen_bounded(plan.size(), 0);
  std::vector<size_t> screen_order;
  for (size_t i = 0; i < samples.size(); ++i) {
    const size_t tile = feedback_bounds ? tile_order[samples.size() - 1 - i] : i;
    if (screened[tile]) screen_order.push_back(tile);
  }
  JXL_RETURN_IF_ERROR(
      RunProbeTasks(plan.size(), pool, [&](size_t candidate) -> Status {
        auto& entry = plan[candidate];
        // Cross-channel alternatives also feed the separate exploration list;
        // their complete prefix score is required even outside the shortlist.
        const bool can_bound = feedback_bounds &&
                               entry.params.options.predictor != Predictor::Variable;
        for (size_t tile : screen_order) {
          // Only completed totals tighten the cutoff. Nonnegative remaining
          // sizes cannot rescue a prefix strictly above the current sixth best.
          // Equality is retained so stable plan-order ties are unchanged.
          if (can_bound && entry.bytes > cutoff.load(std::memory_order_relaxed)) {
            screen_bounded[candidate] = 1;
            entry.bytes = kUnmeasured;
            break;
          }
          JXL_ASSIGN_OR_RETURN(size_t bytes,
                               encode(entry.params, sample_metadata,
                                      *samples[tile].input));
          entry.bytes += bytes;
          ++screen_encodes[candidate];
        }
        if (feedback_bounds && !screen_bounded[candidate]) {
          std::lock_guard<std::mutex> lock(completed_mutex);
          const auto pos = std::lower_bound(best_complete.begin(),
                                             best_complete.end(), entry.bytes);
          if (pos != best_complete.end()) {
            std::move_backward(pos, best_complete.end() - 1, best_complete.end());
            *pos = entry.bytes;
            cutoff.store(best_complete.back(), std::memory_order_relaxed);
          }
        }
        return true;
      }));
  const size_t completed_encodes =
      std::accumulate(screen_encodes.begin(), screen_encodes.end(), size_t{0});
  result.screen_bounded =
      std::accumulate(screen_bounded.begin(), screen_bounded.end(), size_t{0});
  result.screen_saved_encodes = plan.size() * result.screen_samples - completed_encodes;
  screen_profile.Set("encoded_cells", completed_encodes);
  screen_profile.Set("bound_rejections", result.screen_bounded);
  screen_profile.Set("saved_encodes", result.screen_saved_encodes);
  screen_profile.Finish();
  result.encodes = samples.size() + completed_encodes;
  for (const auto& candidate : plan) {
    if (candidate.params.options.predictor == Predictor::Variable) {
      result.exploratory.push_back(candidate);
      result.exploratory.back().exploratory = true;
    }
  }
  std::vector<size_t> order(plan.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return plan[a].bytes < plan[b].bytes;
  });
  // At most 90 sample encodes for 15 families, versus the old 99 for nine.
  // A shortlist is a heuristic, not a proof that discarded families lose on
  // the whole frame. Real full-frame winners are always retained separately.
  result.shortlisted = std::min(kShortlist, order.size());
  std::vector<size_t> extra_encodes(result.shortlisted, 0);
  std::vector<uint8_t> bounded(result.shortlisted, 0);
  const double limit = result.control_bytes * 1.10;
  JXL_RETURN_IF_ERROR(
      RunProbeTasks(result.shortlisted, pool, [&](size_t index) -> Status {
        auto& candidate = plan[order[index]];
        // Expensive regions first make the nonnegative partial-size bound
        // useful.
        for (auto tile = tile_order.rbegin(); tile != tile_order.rend();
             ++tile) {
          if (screened[*tile]) continue;
          if (candidate.bytes > limit) {
            bounded[index] = 1;
            break;
          }
          JXL_ASSIGN_OR_RETURN(
              size_t bytes,
              encode(candidate.params, sample_metadata, *samples[*tile].input));
          candidate.bytes += bytes;
          ++extra_encodes[index];
        }
        return true;
      }));
  for (size_t i = 0; i < result.shortlisted; ++i) {
    result.encodes += extra_encodes[i];
    result.bounded += bounded[i];
    // Keep a margin for small-frame header overhead and sampling error.
    if (!bounded[i] && plan[order[i]].bytes <= limit) {
      result.ranked.push_back(std::move(plan[order[i]]));
    }
  }
  std::stable_sort(
      result.ranked.begin(), result.ranked.end(),
      [](const auto& a, const auto& b) { return a.bytes < b.bytes; });
  for (auto& candidate : result.ranked) candidate.small_bytes = candidate.bytes;
  result.sample_pixels = result.encodes * width * height;

  // Small independent trees can exaggerate a context model's whole-frame
  // benefit. Spend the saved small-tile work on a larger unchanged region,
  // reusing one incumbent encode for up to three shortlisted alternatives.
  constexpr size_t kValidationSide = 384;
  if (!result.ranked.empty() && uint64_t{input.xsize} * input.ysize >=
                                    4 * kValidationSide * kValidationSide) {
    const size_t validation_width =
        std::min<size_t>(kValidationSide, input.xsize);
    const size_t validation_height =
        std::min<size_t>(kValidationSide, input.ysize);
    const auto anchor = positions[tile_order[2 * tile_order.size() / 3]];
    const size_t x =
        std::min(input.xsize - validation_width,
                 anchor.first + width / 2 > validation_width / 2
                     ? anchor.first + width / 2 - validation_width / 2
                     : 0);
    const size_t y =
        std::min(input.ysize - validation_height,
                 anchor.second + height / 2 > validation_height / 2
                     ? anchor.second + height / 2 - validation_height / 2
                     : 0);
    JXL_ASSIGN_OR_RETURN(
        Sample validation,
        CopySample(source, color_format, metadata.m.num_extra_channels, x, y,
                   validation_width, validation_height));
    CodecMetadata validation_metadata = sample_metadata;
    JXL_RETURN_IF_ERROR(
        validation_metadata.size.Set(validation_width, validation_height));
    result.ranked.resize(std::min<size_t>(3, result.ranked.size()));
    std::vector<size_t> bytes(result.ranked.size() + 1);
    JXL_RETURN_IF_ERROR(
        RunProbeTasks(bytes.size(), pool, [&](size_t i) -> Status {
          JXL_ASSIGN_OR_RETURN(
              bytes[i], encode(i == 0 ? control : result.ranked[i - 1].params,
                               validation_metadata, *validation.input));
          return true;
        }));
    result.validation_encodes = bytes.size();
    result.encodes += bytes.size();
    result.sample_pixels += bytes.size() * validation_width * validation_height;
    result.control_bytes = bytes[0];
    for (size_t i = 0; i < result.ranked.size(); ++i) {
      result.ranked[i].bytes = bytes[i + 1];
    }
    result.ranked.erase(
        std::remove_if(result.ranked.begin(), result.ranked.end(),
                       [&](const auto& candidate) {
                         // A disagreement between scales can be useful. Losing
                         // at both is no evidence for another full-frame trial.
                         return candidate.bytes > result.control_bytes * 1.05 ||
                                (candidate.bytes >= result.control_bytes &&
                                 candidate.small_bytes >=
                                     result.small_control_bytes);
                       }),
        result.ranked.end());
    std::stable_sort(
        result.ranked.begin(), result.ranked.end(),
        [](const auto& a, const auto& b) { return a.bytes < b.bytes; });
  }
  return result;
}

StatusOr<LosslessProbeConfirmation> ConfirmLosslessCandidate(
    const CompressParams& candidate, const CompressParams& incumbent,
    const CodecMetadata& metadata, JxlEncoderChunkedFrameAdapter& input,
    ThreadPool* pool, const LosslessProbeEncoder& encode,
    size_t known_incumbent_bytes) {
  LosslessProbeConfirmation result;
  // Do not pay for near-full-frame confirmation on already cheap inputs.
  if (!encode || uint64_t{input.xsize} * input.ysize <= 262144) return result;
  auto source = input.GetInputSource();
  JxlPixelFormat format{};
  source.get_color_channels_pixel_format(source.opaque, &format);
  format.align = 0;
  const size_t width = std::min<size_t>(512, input.xsize);
  const size_t height = std::min<size_t>(512, input.ysize);
  CodecMetadata sample_metadata = metadata;
  JXL_RETURN_IF_ERROR(sample_metadata.size.Set(width, height));
  sample_metadata.m.have_animation = false;
  sample_metadata.m.have_preview = false;
  std::vector<Sample> samples;
  std::vector<std::pair<size_t, size_t>> positions;
  for (size_t i = 0; i < 3; ++i) {
    const size_t x = i * (input.xsize - width) / 2;
    const size_t y = i * (input.ysize - height) / 2;
    const auto position = std::make_pair(x, y);
    if (std::find(positions.begin(), positions.end(), position) !=
        positions.end())
      continue;
    positions.push_back(position);
    JXL_ASSIGN_OR_RETURN(
        Sample sample, CopySample(source, format, metadata.m.num_extra_channels,
                                  x, y, width, height));
    samples.push_back(std::move(sample));
  }
  result.samples = samples.size();
  const bool cached = known_incumbent_bytes != 0;
  std::vector<size_t> sizes((cached ? 1 : 2) * samples.size());
  const size_t workers = std::min<size_t>(4, sizes.size());
  const auto run = [&](uint32_t worker, size_t) -> Status {
    for (size_t task = worker; task < sizes.size(); task += workers) {
      JXL_ASSIGN_OR_RETURN(
          sizes[task],
          encode(!cached && task < samples.size() ? incumbent : candidate,
                 sample_metadata, *samples[task % samples.size()].input));
    }
    return true;
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, workers, ThreadPool::NoInit, run,
                                "LosslessConfirmation"));
  result.incumbent_bytes = known_incumbent_bytes;
  for (size_t i = 0; i < samples.size(); ++i) {
    if (!cached) result.incumbent_bytes += sizes[i];
    result.candidate_bytes += sizes[(cached ? 0 : samples.size()) + i];
  }
  result.accepted = result.candidate_bytes < result.incumbent_bytes;
  return result;
}

}  // namespace jxl
