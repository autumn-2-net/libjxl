// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_lossless_probe.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "lib/jxl/encode_internal.h"
#include "lib/jxl/image_metadata.h"

namespace jxl {
namespace {

std::vector<LosslessProbeCandidate> MakePlan(const CompressParams& input) {
  std::vector<LosslessProbeCandidate> plan;
  auto p = input;
  p.speed_tier = SpeedTier::kTortoise;
  p.options.nb_repeats = 0.1f;
  p.patches = Override::kOff;
  plan.push_back({"regular-control", p});
  p.modular_group_size_shift = 3;
  p.options.predictor = Predictor::Zero;
  p.options.nb_repeats = 0.01f;
  p.palette_colors = 10000;
  p.channel_colors_percent = 0;
  plan.push_back({"literal-control", p});

  p.options.nb_repeats = 0.1f;
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
  return plan;
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

}  // namespace

StatusOr<LosslessProbeResult> ProbeLosslessCandidates(
    const CompressParams& params, const CodecMetadata& metadata,
    JxlEncoderChunkedFrameAdapter& input, ThreadPool* pool,
    const LosslessProbeEncoder& encode) {
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
  const size_t width = std::min<size_t>(96, input.xsize);
  const size_t height = std::min<size_t>(96, input.ysize);
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
  result.encodes = samples.size() * plan.size();
  std::vector<size_t> sizes(result.encodes);
  // At most four concurrent sample encoders, with no nested runner calls.
  const size_t workers = std::min<size_t>(4, result.encodes);
  const auto run = [&](uint32_t worker, size_t) -> Status {
    for (size_t task = worker; task < sizes.size(); task += workers) {
      const size_t candidate = task / samples.size();
      auto& sample = samples[task % samples.size()];
      JXL_ASSIGN_OR_RETURN(sizes[task], encode(plan[candidate].params,
                                               sample_metadata, *sample.input));
    }
    return true;
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, workers, ThreadPool::NoInit, run,
                                "LosslessFirstPass"));
  for (size_t candidate = 0; candidate < plan.size(); ++candidate) {
    for (size_t tile = 0; tile < samples.size(); ++tile) {
      plan[candidate].bytes += sizes[candidate * samples.size() + tile];
    }
  }
  result.control_bytes = std::min(plan[0].bytes, plan[1].bytes);
  for (size_t i = 2; i < plan.size(); ++i) {
    // Keep a margin for small-frame header overhead and sampling error.
    if (static_cast<double>(plan[i].bytes) <= result.control_bytes * 1.05) {
      result.ranked.push_back(std::move(plan[i]));
    }
  }
  std::stable_sort(
      result.ranked.begin(), result.ranked.end(),
      [](const auto& a, const auto& b) { return a.bytes < b.bytes; });
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
