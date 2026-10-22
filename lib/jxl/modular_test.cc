// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/cms.h>
#include <jxl/encode.h>
#include <jxl/memory_manager.h>
#include <jxl/types.h>

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "lib/extras/codec.h"
#include "lib/extras/codec_in_out.h"
#include "lib/extras/dec/color_hints.h"
#include "lib/extras/dec/decode.h"
#include "lib/extras/dec/jxl.h"
#include "lib/extras/enc/jxl.h"
#include "lib/extras/metrics.h"
#include "lib/extras/packed_image.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/random.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/butteraugli/butteraugli.h"
#include "lib/jxl/cms/color_encoding_cms.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/common.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/enc_fields.h"
#include "lib/jxl/enc_lossless_workers.h"
#include "lib/jxl/enc_ma_tree_groups.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/enc_toc.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/headers.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/image_test_utils.h"
#include "lib/jxl/modular/encoding/enc_encoding.h"
#include "lib/jxl/modular/encoding/enc_ma_histograms.h"
#include "lib/jxl/modular/encoding/enc_ma_entropy.h"
#include "lib/jxl/modular/encoding/enc_ma_racing.h"
#include "lib/jxl/modular/encoding/enc_predict.h"
#include "lib/jxl/modular/encoding/encoding.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/options.h"
#include "lib/jxl/modular/transform/squeeze_params.h"
#include "lib/jxl/modular/transform/enc_palette_map.h"
#include "lib/jxl/modular/transform/enc_palette_order.h"
#include "lib/jxl/modular/transform/enc_palette.h"
#include "lib/jxl/modular/transform/enc_palette_delta.h"
#include "lib/jxl/modular/transform/palette.h"
#include "lib/jxl/modular/transform/transform.h"
#include "lib/jxl/padded_bytes.h"
#include "lib/jxl/pack_signed.h"
#include "lib/jxl/test_image.h"
#include "lib/jxl/test_memory_manager.h"
#include "lib/jxl/test_utils.h"
#include "lib/jxl/testing.h"

namespace jxl {
namespace {

using ::jxl::test::ButteraugliDistance;
using ::jxl::test::ReadTestData;
using ::jxl::test::Roundtrip;
using ::jxl::test::TestImage;

TEST(ModularTest, LosslessPaletteMapExactKeys) {
  for (size_t channels : {1u, 2u, 3u, 4u}) {
    LosslessPaletteMap map(channels);
    std::map<std::vector<int32_t>, size_t> reference;
    for (size_t pass = 0; pass < 2; ++pass) {
      for (size_t i = 0; i < 12000; ++i) {
        std::vector<int32_t> key(channels);
        for (size_t c = 0; c < channels; ++c)
          key[c] = static_cast<int32_t>((i * 137 + c * 65537) % 131071) - 65535;
        if (channels == 4) key[3] = 0;
        auto& entry = map.Get(key);
        ++entry.count;
        entry.index = i;
        ++reference[key];
      }
    }
    ASSERT_EQ(map.Size(), reference.size());
    for (const auto& pair : reference) EXPECT_EQ(map.Get(pair.first).count, pair.second);
    std::vector<int32_t> zero(channels), negative(channels, INT32_MIN);
    map.Get(zero).index = 17;
    map.Get(negative).index = 91;
    EXPECT_EQ(map.Get(zero).index, 17u);
    EXPECT_EQ(map.Get(negative).index, 91u);
  }
}

TEST(ModularTest, LosslessParallelRGBA) {
  for (uint32_t bits : {8u, 16u}) {
    for (int pattern : {0, 1, 2, 3, 4}) {
    TestImage t;
    ASSERT_TRUE(t.SetDimensions(513, 259));
    ASSERT_TRUE(t.SetChannels(4));
    const auto type = bits == 8 ? JXL_TYPE_UINT8 : JXL_TYPE_UINT16;
    t.SetAllBitDepths(bits).SetDataType(type).SetEndianness(JXL_LITTLE_ENDIAN);
    JXL_TEST_ASSIGN_OR_DIE(auto frame, t.AddFrame());
    const uint32_t maximum = (1u << bits) - 1;
    for (size_t y = 0; y < 259; ++y) {
      for (size_t x = 0; x < 513; ++x) {
        for (size_t c = 0; c < 4; ++c) {
          uint32_t value = c == 3 ? ((x / 17 + y / 19) % 3) * (maximum / 2)
              : (x * 13 + y * 7 + c * 173 + (x / 7) * (y / 3)) & maximum;
          if (pattern == 1 && c == 3) value = maximum;
          if (pattern == 2 && c == 3) value = 0;
          if (pattern == 3) value = (c + 1) * 17;
          if (pattern == 4 && c == 3)
            value = maximum - (x == 512 && y == 258 ? 1 : 0);
          ASSERT_TRUE(frame.SetValue(y, x, c, float(value) / maximum));
        }
      }
    }
    for (int threads : {0, 4}) {
      test::ThreadPoolForTests pool(threads);
      extras::JXLCompressParams cparams;
      cparams.distance = 0;
      cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 9);
      cparams.AddOption(JXL_ENC_FRAME_SETTING_MODULAR_GROUP_SIZE, 0);
      cparams.AddOption(JXL_ENC_FRAME_SETTING_KEEP_INVISIBLE, 1);
      extras::JXLDecompressParams dparams;
      dparams.accepted_formats = {{4, type, JXL_LITTLE_ENDIAN, 0}};
      extras::PackedPixelFile decoded;
      ASSERT_GT(Roundtrip(t.ppf(), cparams, dparams, pool.get(), &decoded), 0u);
      ASSERT_EQ(decoded.frames.size(), 1u);
      const auto& a = t.ppf().frames[0].color;
      const auto& b = decoded.frames[0].color;
      ASSERT_EQ(a.xsize, b.xsize);
      ASSERT_EQ(a.ysize, b.ysize);
      for (size_t y = 0; y < a.ysize; ++y) {
        EXPECT_EQ(std::memcmp(a.const_pixels(y, 0, 0), b.const_pixels(y, 0, 0),
                              a.xsize * 4 * (bits / 8)), 0);
      }
    }
  }
  }
}

TEST(ModularTest, LosslessPaletteReorderingRGBA) {
  for (uint32_t bits : {8u, 16u}) {
    for (uint32_t colors : {17u, 2048u}) {
      for (uint32_t limit : {16u, 10000u}) {
        SCOPED_TRACE(testing::Message() << bits << ":" << colors << ":" << limit);
        TestImage t;
        ASSERT_TRUE(t.SetDimensions(129, 73));
        ASSERT_TRUE(t.SetChannels(4));
        const auto type = bits == 8 ? JXL_TYPE_UINT8 : JXL_TYPE_UINT16;
        t.SetAllBitDepths(bits).SetDataType(type).SetEndianness(JXL_LITTLE_ENDIAN);
        JXL_TEST_ASSIGN_OR_DIE(auto frame, t.AddFrame());
        const uint32_t maximum = (1u << bits) - 1;
        for (size_t y = 0; y < 73; ++y) {
          for (size_t x = 0; x < 129; ++x) {
            const uint32_t index = (x + 129 * y) % colors;
            for (size_t c = 0; c < 4; ++c) {
              const uint32_t value = c == 3 ? (index % 3) * (maximum / 2)
                  : c == 0 ? index & maximum
                  : c == 1 ? (index * 13 + (index >> 8)) & maximum
                           : (index * 173 + 71) & maximum;
              ASSERT_TRUE(frame.SetValue(y, x, c, float(value) / maximum));
            }
          }
        }
        test::ThreadPoolForTests pool(4);
        extras::JXLCompressParams cparams;
        cparams.distance = 0;
        cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 9);
        cparams.AddOption(JXL_ENC_FRAME_SETTING_MODULAR_GROUP_SIZE, 3);
        cparams.AddOption(JXL_ENC_FRAME_SETTING_MODULAR_PREDICTOR, 0);
        cparams.AddOption(JXL_ENC_FRAME_SETTING_PALETTE_COLORS, limit);
        cparams.AddFloatOption(JXL_ENC_FRAME_SETTING_MODULAR_MA_TREE_LEARNING_PERCENT, 1);
        cparams.AddFloatOption(JXL_ENC_FRAME_SETTING_CHANNEL_COLORS_GLOBAL_PERCENT, 0);
        cparams.AddOption(JXL_ENC_FRAME_SETTING_KEEP_INVISIBLE, 1);
        extras::JXLDecompressParams dparams;
        dparams.accepted_formats = {{4, type, JXL_LITTLE_ENDIAN, 0}};
        extras::PackedPixelFile decoded;
        ASSERT_GT(Roundtrip(t.ppf(), cparams, dparams, pool.get(), &decoded), 0u);
        ASSERT_EQ(decoded.frames.size(), 1u);
        const auto& a = t.ppf().frames[0].color;
        const auto& b = decoded.frames[0].color;
        ASSERT_EQ(a.xsize, b.xsize);
        ASSERT_EQ(a.ysize, b.ysize);
        for (size_t y = 0; y < a.ysize; ++y) {
          EXPECT_EQ(std::memcmp(a.const_pixels(y, 0, 0), b.const_pixels(y, 0, 0),
                                a.xsize * 4 * (bits / 8)), 0);
        }
      }
    }
  }
}

TEST(ModularTest, LosslessStripedLearningRGBA) {
  for (uint32_t bits : {8u, 16u}) {
    constexpr size_t width = 520, height = 521;
    TestImage image;
    ASSERT_TRUE(image.SetDimensions(width, height));
    ASSERT_TRUE(image.SetChannels(4));
    const auto type = bits == 8 ? JXL_TYPE_UINT8 : JXL_TYPE_UINT16;
    image.SetAllBitDepths(bits).SetDataType(type).SetEndianness(JXL_LITTLE_ENDIAN);
    JXL_TEST_ASSIGN_OR_DIE(auto frame, image.AddFrame());
    const uint32_t maximum = (1u << bits) - 1;
    for (size_t y = 0; y < height; ++y) {
      for (size_t x = 0; x < width; ++x) {
        for (size_t c = 0; c < 4; ++c) {
          const uint32_t value = c == 3 ? ((x / 19 + y / 31) % 3) * (maximum / 2)
              : (x * 131 + y * 37 + c * 197 + (x / 3) * (y / 5)) & maximum;
          ASSERT_TRUE(frame.SetValue(y, x, c, float(value) / maximum));
        }
      }
    }
    for (int threads : {0, 4}) {
      test::ThreadPoolForTests pool(threads);
      extras::JXLCompressParams params;
      params.distance = 0;
      params.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 9);
      params.AddOption(JXL_ENC_FRAME_SETTING_MODULAR_GROUP_SIZE, 3);
      params.AddOption(JXL_ENC_FRAME_SETTING_MODULAR_PREDICTOR, 15);
      params.AddFloatOption(JXL_ENC_FRAME_SETTING_MODULAR_MA_TREE_LEARNING_PERCENT, 10);
      params.AddOption(JXL_ENC_FRAME_SETTING_PALETTE_COLORS, 0);
      params.AddFloatOption(JXL_ENC_FRAME_SETTING_CHANNEL_COLORS_GLOBAL_PERCENT, 0);
      params.AddFloatOption(JXL_ENC_FRAME_SETTING_CHANNEL_COLORS_GROUP_PERCENT, 0);
      params.AddOption(JXL_ENC_FRAME_SETTING_KEEP_INVISIBLE, 1);
      extras::JXLDecompressParams output;
      output.accepted_formats = {{4, type, JXL_LITTLE_ENDIAN, 0}};
      extras::PackedPixelFile decoded;
      ASSERT_GT(Roundtrip(image.ppf(), params, output, pool.get(), &decoded), 0u);
      ASSERT_EQ(decoded.frames.size(), 1u);
      const auto& a = image.ppf().frames[0].color;
      const auto& b = decoded.frames[0].color;
      ASSERT_EQ(a.xsize, b.xsize);
      ASSERT_EQ(a.ysize, b.ysize);
      for (size_t y = 0; y < height; ++y)
        EXPECT_EQ(std::memcmp(a.const_pixels(y, 0, 0), b.const_pixels(y, 0, 0),
                              width * 4 * (bits / 8)), 0);
    }
  }
}

TEST(ModularTest, ExactDeltaPaletteTransform) {
  for (size_t width : {1u, 13u, 65u}) {
    for (int bits : {8, 16}) {
      for (Predictor p : {Predictor::Left, Predictor::Top, Predictor::Gradient}) {
        SCOPED_TRACE(testing::Message() << width << ":" << bits << ":" << int(p));
        JXL_TEST_ASSIGN_OR_DIE(auto input,
            Image::Create(test::MemoryManager(), width, 19, bits, 4));
        std::vector<int32_t> original;
        for (size_t c = 0; c < 4; ++c) {
          for (size_t y = 0; y < 19; ++y) {
            for (size_t x = 0; x < width; ++x) {
              const int32_t value = c == 3 ? ((x + y) % 3) * (((1 << bits) - 1) / 2)
                  : ((x * 37 + y * 157 + c * 73) * (bits == 16 ? 257 : 1)) & ((1 << bits) - 1);
              input.channel[c].Row(y)[x] = value;
              original.push_back(value);
            }
          }
        }
        uint32_t colors = 1, deltas = 17;
        Predictor predictor = Predictor::Zero;
        JXL_TEST_ASSIGN_OR_DIE(bool insufficient, FwdExactDeltaPalette(input, 0,
            3, &colors, &deltas, &predictor, p));
        ASSERT_FALSE(insufficient);
        ASSERT_EQ(input.channel.size(), 4u);
        ASSERT_EQ(input.nb_meta_channels, 0u);
        ASSERT_EQ(colors, 1u);
        ASSERT_EQ(deltas, 17u);
        ASSERT_EQ(predictor, Predictor::Zero);
        size_t i = 0;
        for (const auto& channel : input.channel)
          for (size_t y = 0; y < channel.h; ++y)
            for (size_t x = 0; x < channel.w; ++x)
              ASSERT_EQ(channel.Row(y)[x], original[i++]);
        colors = 65536;
        JXL_TEST_ASSIGN_OR_DIE(bool applied, FwdExactDeltaPalette(input, 0,
            3, &colors, &deltas, &predictor, p));
        ASSERT_TRUE(applied);
        ASSERT_EQ(colors, 0u);
        ASSERT_GT(deltas, 0u);
        ASSERT_TRUE(InvPalette(input, 0, colors, deltas, predictor,
                                 weighted::Header(), nullptr));
        ASSERT_EQ(input.channel.size(), 4u);
        ASSERT_EQ(input.nb_meta_channels, 0u);
        i = 0;
        for (const auto& channel : input.channel)
          for (size_t y = 0; y < channel.h; ++y)
            for (size_t x = 0; x < channel.w; ++x)
              ASSERT_EQ(channel.Row(y)[x], original[i++]);
      }
    }
  }
}

TEST(ModularTest, PaletteToolsRequireLosslessFrameEligibility) {
  const char* mode = std::getenv("JXL_LOSSLESS_DELTA_PALETTE");
  if (!mode || mode[0] < '1' || mode[0] > '6' || mode[1] != '\0') GTEST_SKIP();
  for (bool eligible : {false, true}) {
    JXL_TEST_ASSIGN_OR_DIE(auto input,
        Image::Create(test::MemoryManager(), 31, 17, 8, 3));
    for (size_t c = 0; c < 3; ++c)
      for (size_t y = 0; y < 17; ++y)
        for (size_t x = 0; x < 31; ++x)
          input.channel[c].Row(y)[x] = ((x / 5 + y / 3) * (13 + c * 3) + 17) % 256;
    uint32_t colors = 10000, deltas = 0;
    Predictor predictor = Predictor::Zero;
    ASSERT_TRUE(FwdPalette(input, 0, 2, colors, deltas, true, false,
                             predictor, weighted::Header(), eligible));
    if (eligible) {
      EXPECT_GT(deltas, 0u);
    } else {
      EXPECT_EQ(deltas, 0u);
      EXPECT_EQ(predictor, Predictor::Zero);
    }
  }
}

TEST(ModularTest, PaletteAdjacencyKeepsEveryColorOnce) {
  const std::vector<size_t> frequencies{1000, 80, 10, 40, 20, 1};
  std::vector<uint64_t> pairs;
  const auto edge = [&](uint32_t a, uint32_t b, size_t count) {
    pairs.insert(pairs.end(), count, (uint64_t{a} << 32) | b);
  };
  edge(0, 3, 100);
  edge(1, 3, 90);
  edge(1, 4, 80);
  edge(0, 4, 70);  // Would close a cycle.
  edge(2, 5, 10);
  edge(2, 2, 50);  // Self edge is ignored.
  edge(2, 999, 60);  // Invalid endpoint is ignored.
  const auto order = PaletteAdjacencyOrder(frequencies, pairs);
  EXPECT_EQ(order, (std::vector<size_t>{0, 3, 1, 4, 2, 5}));
  EXPECT_TRUE(PaletteAdjacencyOrder({}, {}).empty());
  EXPECT_EQ(PaletteAdjacencyOrder({3}, {}), std::vector<size_t>{0});
  EXPECT_EQ(PaletteAdjacencyOrder({1, 3, 2}, {}), (std::vector<size_t>{1, 2, 0}));
  for (size_t n : {2u, 3u, 9u, 64u, 257u}) {
    std::vector<size_t> counts(n, 1);
    pairs.clear();
    for (uint32_t a = 0; a < n; ++a) {
      for (uint32_t b = a + 1; b < n; ++b) edge(a, b, 1 + (a * 7 + b * 3) % 5);
    }
    auto permutation = PaletteAdjacencyOrder(counts, pairs);
    ASSERT_EQ(permutation.size(), n);
    std::sort(permutation.begin(), permutation.end());
    for (size_t i = 0; i < n; ++i) EXPECT_EQ(permutation[i], i);
  }
}

TEST(ModularTest, CountLogTableClippedEntropy) {
  const float* logs = MaCountLogTable();
  EXPECT_EQ(logs[0], 0);
  for (uint32_t total : {1u, 255u, 4095u, 4096u, 4097u, 65536u}) {
    for (uint32_t count = 0; count <= total; count += std::max(1u, total / 997)) {
      const double expected = count ? count * std::min(12.0,
          std::log2(double(total) / count)) : 0;
      const float actual = count * std::min(12.0f, logs[total] - logs[count]);
      EXPECT_NEAR(actual, expected, std::max(0.00001, double(count) * 2e-6));
    }
  }
}

TEST(ModularTest, PalettePrincipalOrderPermutation) {
  for (size_t channels : {3u, 4u}) {
    for (bool banded : {false, true}) {
      std::vector<std::array<int32_t, 4>> colors;
      std::vector<size_t> frequencies;
      for (int32_t i = 0; i < 257; ++i) {
        colors.push_back({i * 65537 - 10000000, -i * 257, (i % 17) * 999,
                          i % 3 == 0 ? 0 : 65535});
        frequencies.push_back(1 + (i * 13) % 255);
      }
      auto order = PalettePrincipalOrder(colors, frequencies, channels, banded);
      auto sorted = order;
      std::sort(sorted.begin(), sorted.end());
      ASSERT_EQ(sorted.size(), colors.size());
      for (size_t i = 0; i < sorted.size(); ++i) EXPECT_EQ(sorted[i], i);
      EXPECT_EQ(order, PalettePrincipalOrder(colors, frequencies, channels, banded));
    }
  }
  EXPECT_TRUE(PalettePrincipalOrder({}, {}, 3, false).empty());
  EXPECT_EQ(PalettePrincipalOrder({{{0, 0, 0, 0}}}, {1}, 4, true),
            (std::vector<size_t>{0}));
}

TEST(ModularTest, PaletteHilbertOrderContinuity) {
  struct Point {
    std::array<uint32_t, 4> axes, key;
  };
  for (size_t dimensions : {2u, 3u, 4u}) {
    for (unsigned bits : {1u, 2u, 3u}) {
      std::vector<Point> points;
      const size_t total = size_t{1} << (dimensions * bits);
      for (size_t index = 0; index < total; ++index) {
        Point point{};
        for (size_t c = 0; c < dimensions; ++c)
          point.axes[c] = (index >> (c * bits)) & ((1u << bits) - 1);
        point.key = PaletteHilbertKey(point.axes, dimensions, bits);
        points.push_back(point);
      }
      std::sort(points.begin(), points.end(), [&](const auto& a, const auto& b) {
        return PaletteInterleavedLess(a.key, b.key, dimensions);
      });
      for (size_t i = 1; i < points.size(); ++i) {
        EXPECT_TRUE(PaletteInterleavedLess(points[i - 1].key, points[i].key, dimensions));
        uint32_t distance = 0;
        for (size_t c = 0; c < dimensions; ++c)
          distance += std::abs(int32_t(points[i - 1].axes[c]) - int32_t(points[i].axes[c]));
        EXPECT_EQ(distance, 1u);
      }
    }
  }
}

TEST(ModularTest, ClippedEntropyIncrementalHistogram) {
  std::vector<double> table(257);
  for (size_t i = 1; i < table.size(); ++i) table[i] = i * std::log2(double(i));
  for (bool sliding : {false, true}) {
  for (uint32_t total : {0u, 1u, 4095u, 4096u, 4097u, 65536u, 1000000u}) {
    SCOPED_TRACE(testing::Message() << "sliding=" << sliding << " total=" << total);
    std::array<int32_t, 256> right{}, left{};
    uint32_t remaining = total;
    for (size_t i = 1; i < right.size(); ++i) {
      right[i] = std::min<uint32_t>(remaining, i % 3 + 1);
      remaining -= right[i];
    }
    right[0] = remaining;
    MaClippedEntropy r(right.data(), right.size(), total, table.data(), 256, sliding);
    MaClippedEntropy l(left.data(), left.size(), total, table.data(), 256, sliding);
    uint32_t nr = total, nl = 0;
    const auto check = [&](const auto& counts, auto& entropy, uint32_t n) {
      double expected = 0;
      for (int32_t c : counts) {
        if (c) expected -= c * std::log2(std::max(double(c) / n, 1.0 / ANS_TAB_SIZE));
      }
      EXPECT_NEAR(entropy.Bits(n), expected, 1e-4 + 1e-7 * expected);
    };
    check(right, r, nr);
    check(left, l, nl);
    for (size_t step = 0; step < 2048; ++step) {
      const size_t i = (step * 137) % right.size();
      const bool reverse = step % 7 == 6;
      const int32_t amount = std::min<int32_t>(reverse ? left[i] : right[i],
          i == 0 ? total / 8 + 1 : step % 17 + 1);
      const int32_t delta = reverse ? -amount : amount;
      r.Replace(right[i], right[i] - delta);
      l.Replace(left[i], left[i] + delta);
      right[i] -= delta;
      left[i] += delta;
      nr -= delta;
      nl += delta;
      check(right, r, nr);
      check(left, l, nl);
    }
  }
  }
}

TEST(ModularTest, MaTreeGroupBoundaries) {
  EXPECT_EQ(MaTreeGroupBoundaries({}, 8, 1), (std::vector<size_t>{0, 0}));
  EXPECT_EQ(MaTreeGroupBoundaries({0, 0, 0}, 8, 1),
            (std::vector<size_t>{0, 3}));
  EXPECT_EQ(MaTreeGroupBoundaries({10, 10, 10, 10}, 1, 1),
            (std::vector<size_t>{0, 4}));
  EXPECT_EQ(MaTreeGroupBoundaries({10, 10, 10, 10}, 4, 1),
            (std::vector<size_t>{0, 1, 2, 3, 4}));
  EXPECT_EQ(MaTreeGroupBoundaries({0, 10, 0, 10, 0}, 8, 1),
            (std::vector<size_t>{0, 2, 5}));
  EXPECT_EQ(MaTreeGroupBoundaries({10, 10, 10, 10}, 8, 20),
            (std::vector<size_t>{0, 2, 4}));
  EXPECT_EQ(MaTreeGroupBoundaries({100, 1, 0, 1}, 8, 10),
            (std::vector<size_t>{0, 4}));
  for (size_t requested : {0u, 1u, 2u, 4u, 8u, 100u}) {
    std::vector<uint64_t> work(113);
    for (size_t i = 0; i < work.size(); ++i) work[i] = (i * 17) % 31;
    const auto bounds = MaTreeGroupBoundaries(work, requested, 30);
    ASSERT_GE(bounds.size(), 2u);
    EXPECT_EQ(bounds.front(), 0u);
    EXPECT_EQ(bounds.back(), work.size());
    EXPECT_LE(bounds.size(), 9u);
    for (size_t i = 1; i < bounds.size(); ++i) {
      ASSERT_GT(bounds[i], bounds[i - 1]);
      uint64_t count = 0;
      for (size_t j = bounds[i - 1]; j < bounds[i]; ++j) count += work[j];
      EXPECT_GE(count, 30u);
    }
  }
}

TEST(ModularTest, MaHistogramSymbolCount) {
  EXPECT_EQ(MaHistogramSymbolCount(nullptr, 0), 0u);
  std::vector<int32_t> counts(256);
  EXPECT_EQ(MaHistogramSymbolCount(counts.data(), counts.size()), 0u);
  for (size_t tail : {0u, 1u, 7u, 8u, 15u, 16u, 31u, 127u, 255u}) {
    counts[tail] = 13;
    EXPECT_EQ(MaHistogramSymbolCount(counts.data(), counts.size()), tail + 1);
    EXPECT_EQ(MaHistogramSymbolCount(counts.data(), tail + 1), tail + 1);
    counts[tail] = 0;
  }
}

TEST(ModularTest, DenseMaHistogramKeepsCounts) {
  for (size_t lanes : {1u, 4u, 8u, 16u}) {
    for (size_t predictors : {1u, 2u, 14u}) {
      const size_t original_stride = 256;
      std::vector<int32_t> counts(predictors * original_stride);
      for (size_t p = 0; p < predictors; ++p) {
        for (size_t s : {0u, 16u, 32u, 64u, 128u, 240u})
          counts[p * original_stride + s] = (p + s) % 11 + 1;
      }
      auto original = counts;
      size_t stride = original_stride;
      std::array<uint8_t, 256> mapping;
      ASSERT_TRUE(CompactMaHistograms(predictors, lanes, &counts, &stride,
                                      &mapping, true));
      ASSERT_EQ(stride, (6 + lanes - 1) / lanes * lanes);
      for (size_t p = 0; p < predictors; ++p) {
        int64_t sum_before = 0, sum_after = 0;
        for (size_t s = 0; s < original_stride; ++s) {
          const int32_t count = original[p * original_stride + s];
          sum_before += count;
          if (count) EXPECT_EQ(counts[p * stride + mapping[s]], count);
        }
        for (size_t s = 0; s < stride; ++s) sum_after += counts[p * stride + s];
        EXPECT_EQ(sum_before, sum_after);
      }
    }
  }
}

TEST(ModularTest, CompactMaHistogramsPreserveLanesAndCounts) {
  for (size_t lanes : {1u, 4u, 8u, 16u, 32u, 64u}) {
    for (size_t predictors : {1u, 2u, 14u}) {
      SCOPED_TRACE(lanes);
      SCOPED_TRACE(predictors);
      size_t stride = 256;
      std::vector<int32_t> counts(stride * predictors);
      for (size_t p = 0; p < predictors; ++p) {
        for (size_t s : {0u, 7u, 15u, 31u, 127u, 255u}) {
          if ((p + s) % 3) counts[p * stride + s] = 1 + p * 257 + s;
        }
      }
      const auto original = counts;
      std::array<uint8_t, 256> mapping;
      const bool compact = CompactMaHistograms(predictors, lanes, &counts,
                                               &stride, &mapping);
      if (!compact) {
        EXPECT_EQ(counts, original);
        EXPECT_EQ(stride, 256u);
        continue;
      }
      ASSERT_EQ(stride % lanes, 0u);
      ASSERT_LE(stride * 4, 256u * 3);
      for (size_t p = 0; p < predictors; ++p) {
        for (size_t lane = 0; lane < lanes; ++lane) {
          std::vector<int32_t> before, after;
          for (size_t s = lane; s < 256; s += lanes) {
            const int32_t count = original[p * 256 + s];
            if (count) {
              before.push_back(count);
              EXPECT_EQ(mapping[s] % lanes, lane);
              ASSERT_LT(mapping[s], stride);
              EXPECT_EQ(counts[p * stride + mapping[s]], count);
            }
          }
          for (size_t s = lane; s < stride; s += lanes) {
            if (counts[p * stride + s]) after.push_back(counts[p * stride + s]);
          }
          EXPECT_EQ(before, after);
        }
      }
    }
  }
  size_t stride = 256;
  std::array<uint8_t, 256> mapping;
  std::vector<int32_t> dense(256, 1);
  EXPECT_FALSE(CompactMaHistograms(1, 8, &dense, &stride, &mapping));
  EXPECT_EQ(dense, std::vector<int32_t>(256, 1));
  EXPECT_EQ(stride, 256u);
  std::vector<int32_t> empty(256);
  EXPECT_FALSE(CompactMaHistograms(1, 8, &empty, &stride, &mapping));
  EXPECT_FALSE(CompactMaHistograms(1, 0, &empty, &stride, &mapping));
}

TEST(ModularTest, MaParentHistogram) {
  for (Predictor predictor :
       {Predictor::Zero, Predictor::Best, Predictor::Variable}) {
    TreeSamples samples(true);
    ASSERT_TRUE(samples.SetPredictor(predictor, ModularOptions::TreeMode::kDefault));
    ASSERT_TRUE(samples.SetProperties({0, 1, 2, 3, 6, 9},
                                      ModularOptions::TreeMode::kDefault));
    StaticPropRange range{};
    for (auto& r : range) r = {{0, 256}};
    std::vector<pixel_type> pixels{-511, -255, -1, 0, 1, 255, 511};
    auto differences = pixels;
    samples.PreQuantizeProperties(range, {}, {1024}, {1024}, pixels, differences, 32);
    samples.PrepareForSamples(4096);
    for (size_t i = 0; i < 70000; ++i) {
      Properties properties(16);
      const size_t value = i < 66000 ? 0 : i - 66000;
      properties[2] = value % 127;
      properties[6] = value % 511;
      pixel_type_w predictions[kNumModularPredictors] = {};
      for (size_t p = 0; p < kNumModularPredictors; ++p) {
        predictions[p] = static_cast<int>(value * p * 17) - 10000;
      }
      const pixel_type_w pixel = value % 2 ? -(pixel_type_w{1} << 28)
                                            : static_cast<pixel_type_w>(value);
      samples.AddSample(pixel, properties, predictions);
    }
    MaParentHistogram actual;
    for (size_t pred = 0; pred < samples.NumPredictors(); ++pred) {
      const size_t end = samples.NumDistinctSamples();
      for (const auto& interval : {std::pair<size_t, size_t>{0, end},
                                   {end / 3, end}, {1, 2}, {end, end}}) {
        std::array<int32_t, 256> expected{};
        size_t extra = 0;
        uint32_t weight = 0;
        size_t symbols = 0;
        for (size_t i = interval.first; i < interval.second; ++i) {
          const auto token = samples.RToken(pred, i);
          const size_t count = samples.Count(i);
          expected[token.tok] += count;
          extra += token.nbits * count;
          weight += count;
          symbols = std::max(symbols, size_t{token.tok} + 1);
        }
        CollectMaParentHistogram(samples, pred, interval.first, interval.second,
                                  &actual);
        EXPECT_EQ(actual.counts, expected);
        EXPECT_EQ(actual.extra_bits, static_cast<uint32_t>(extra));
        EXPECT_EQ(actual.weight, weight);
        EXPECT_EQ(actual.symbols, symbols);
      }
      MaParentHistogram parent;
      CollectMaParentHistogram(samples, pred, 0, end, &parent);
      for (size_t split : {size_t{0}, size_t{1}, end / 3, end - 1, end}) {
        MaParentHistogram smaller, expected;
        CollectMaParentHistogram(samples, pred, 0, split, &smaller);
        CollectMaParentHistogram(samples, pred, split, end, &expected);
        const auto subtracted = SubtractMaParentHistogram(parent, smaller);
        EXPECT_EQ(subtracted.counts, expected.counts);
        EXPECT_EQ(subtracted.weight, expected.weight);
        EXPECT_EQ(subtracted.extra_bits, expected.extra_bits);
        EXPECT_EQ(subtracted.symbols, expected.symbols);
      }
    }
  }
}

TEST(ModularTest, LosslessWorkerLimits) {
  EXPECT_EQ(8, LosslessWorkerSetting(nullptr, 8, 64));
  EXPECT_EQ(32, LosslessWorkerSetting("32", 8, 64));
  for (const char* value : {"", "0", "-1", "+16", " 16", "16x", "65",
                            "999999999999999999999999999999999999"}) {
    EXPECT_EQ(8, LosslessWorkerSetting(value, 8, 64));
  }
  LosslessWorkerLimits limits{32, uint64_t{2} << 30};
  EXPECT_EQ(0, limits.Tasks(0, 123));
  EXPECT_EQ(32, limits.Tasks(130, uint64_t{32} << 20));
  EXPECT_EQ(16, limits.Tasks(42, uint64_t{128} << 20));
  EXPECT_EQ(3, limits.Tasks(3, uint64_t{128} << 20));
  EXPECT_EQ(1, limits.Tasks(42, uint64_t{4} << 30));
  EXPECT_EQ(32, limits.Tasks(42, 0));
  const std::vector<uint64_t> channels(64, 65536);
  EXPECT_EQ(9, MaTreeGroupBoundaries(channels, 32, 65536).size());
  EXPECT_EQ(33, MaTreeGroupBoundaries(channels, 32, 65536, 32).size());
  EXPECT_EQ(65, MaTreeGroupBoundaries(channels, 64, 65536, 64).size());
}

TEST(ModularTest, MaQuantileRefinementRetainsCoarseCandidates) {
  std::vector<uint8_t> coarse(65);
  std::vector<float> costs(65, std::numeric_limits<float>::infinity());
  for (size_t i = 0; i < 65; i += 8) {
    coarse[i] = 1;
    costs[i] = 100;
  }
  costs[16] = 1;
  costs[48] = 2;
  const auto fine = RefineMaQuantileMask(coarse, costs);
  for (size_t i = 0; i < 65; ++i) {
    if (coarse[i] || (i >= 8 && i <= 24) || (i >= 40 && i <= 56))
      EXPECT_EQ(fine[i], 1);
    else EXPECT_EQ(fine[i], 0);
  }
  EXPECT_EQ(RefineMaQuantileMask({}, {}), (std::vector<uint8_t>{}));
  EXPECT_EQ(RefineMaQuantileMask({1, 0, 1}, {2, 0, 3}),
            (std::vector<uint8_t>{1, 1, 1}));
}

TEST(ModularTest, MaRacePairedEvidence) {
  EXPECT_TRUE(PairedLosslessEvidence({130, 130, 130}, {100, 100, 100}, 0.08)
                  .ClearlyWorse(0.02));
  EXPECT_FALSE(PairedLosslessEvidence({130, 130}, {100, 100}, 0.08)
                   .ClearlyWorse(0.02));
  EXPECT_FALSE(PairedLosslessEvidence({90, 150, 150}, {100, 100, 100}, 0.03)
                   .ClearlyWorse(0.02));
  EXPECT_FALSE(PairedLosslessEvidence({103, 103, 103}, {100, 100, 100}, 0.03)
                   .ClearlyWorse(0.02));
  EXPECT_FALSE(PairedLosslessEvidence({1, 1, 1}, {0, 0, 0}, 0).ClearlyWorse(0));
  EXPECT_FALSE(PairedLosslessEvidence({1}, {1, 1}, 0).ClearlyWorse(0));
  EXPECT_FALSE(PairedLosslessEvidence(
      {INFINITY, 1, 1}, {1, 1, 1}, 0).ClearlyWorse(0));
  EXPECT_DOUBLE_EQ(1100, ConservativeLosslessProjection(
      1000, {120, 120, 120}, {100, 100, 100}));
  EXPECT_EQ(0, ConservativeLosslessProjection(1000, {120, 120}, {100, 100}));
  EXPECT_EQ(0, ConservativeLosslessProjection(1000, {120, 120, 120}, {0, 0, 0}));
  EXPECT_LT(ConservativeLosslessProjection(
      1000, {30, 120, 120}, {100, 100, 100}), 400);
}

TEST(ModularTest, MaRaceIncrementalHistograms) {
  TreeSamples samples(true);
  ASSERT_TRUE(samples.SetPredictor(Predictor::Zero, ModularOptions::TreeMode::kDefault));
  ASSERT_TRUE(samples.SetProperties({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                                    ModularOptions::TreeMode::kDefault));
  StaticPropRange range{};
  for (auto& r : range) r = {{0, 256}};
  std::vector<pixel_type> pixels{-511, -255, -1, 0, 1, 255, 511};
  auto differences = pixels;
  samples.PreQuantizeProperties(range, {}, {65536}, {65536}, pixels, differences, 32);
  samples.PrepareForSamples(4096);
  uint32_t rng = 0x1234567;
  for (size_t i = 0; i < 4096; ++i) {
    Properties properties(16);
    for (size_t p = 2; p < properties.size(); ++p) {
      rng = rng * 1664525u + 1013904223u;
      properties[p] = (rng >> 16) & 511;
    }
    const pixel_type_w pixel = i & 1 ? 255 : 0;
    properties[2] = pixel;
    pixel_type_w predictions[kNumModularPredictors] = {};
    samples.AddSample(pixel, properties, predictions);
  }
  samples.AllSamplesDone();
  ASSERT_TRUE(MaPropertyRace::Fits(samples, 256));
  MaPropertyRace incremental(samples, 0, samples.NumDistinctSamples(), 256);
  MaPropertyRace direct(samples, 0, samples.NumDistinctSamples(), 256);
  for (size_t p = samples.NumStaticProps(); p < samples.NumProperties(); ++p) {
    incremental.Extend(p, 64);
    incremental.Extend(p, 256);
    direct.Extend(p, 256);
    for (size_t panel = 0; panel < MaPropertyRace::kPanels; ++panel) {
      EXPECT_DOUBLE_EQ(incremental.Costs(p)[panel], direct.Costs(p)[panel]);
      EXPECT_GE(incremental.Costs(p)[panel], 0);
    }
  }
  const std::vector<uint8_t> all(samples.NumProperties(), 1);
  const auto keep = incremental.Survivors(all);
  EXPECT_EQ(keep, direct.Survivors(all));
  EXPECT_TRUE(keep[samples.PropertyIndex(2)]);
  EXPECT_GE(std::count(keep.begin(), keep.end(), uint8_t{1}), 4);
  EXPECT_LT(std::count(keep.begin(), keep.end(), uint8_t{1}), keep.size());
  for (size_t p = 0; p < samples.NumStaticProps(); ++p) EXPECT_TRUE(keep[p]);
}

TEST(ModularTest, MaSplitParallelPoolAndErrors) {
  JXL_TEST_ASSIGN_OR_DIE(
      Image image, Image::Create(test::MemoryManager(), 256, 256, 16, 3));
  uint32_t random = 0x243f6a88;
  for (size_t c = 0; c < image.channel.size(); ++c) {
    for (size_t y = 0; y < image.h; ++y) {
      for (size_t x = 0; x < image.w; ++x) {
        random ^= random << 13;
        random ^= random >> 17;
        random ^= random << 5;
        image.channel[c].Row(y)[x] =
            x < 64 ? (x + y + c * 13) : random & 65535;
      }
    }
  }
  for (int threads : {0, 4}) {
    test::ThreadPoolForTests workers(threads);
    test::ThreadPoolForTests fewer_workers(2);
    struct RunnerState {
      ThreadPool* delegate;
      ThreadPool* smaller;
      size_t calls = 0;
      bool fail = false;
      bool alternate = false;
    } state{workers.get(), fewer_workers.get()};
    ThreadPool pool(
        [](void* opaque, void* context, JxlParallelRunInit init,
           JxlParallelRunFunction run, uint32_t begin,
           uint32_t end) -> JxlParallelRetCode {
          auto* state = static_cast<RunnerState*>(opaque);
          ++state->calls;
          if (state->fail) return JXL_PARALLEL_RET_RUNNER_ERROR;
          ThreadPool* delegate = state->alternate && state->calls % 2 == 0
                                     ? state->smaller
                                     : state->delegate;
          return delegate->runner()(delegate->runner_opaque(), context, init,
                                     run, begin, end);
        },
        &state);
    ModularOptions options;
    options.lossless_tree_speedups = true;
    options.nb_repeats = 1.0;
    for (Predictor predictor : {Predictor::Best, Predictor::Variable}) {
      options.predictor = predictor;
      state.calls = 0;
      JXL_TEST_ASSIGN_OR_DIE(Tree tree,
                            LearnTree(&image, &options, 0, 1, {}, &pool));
      EXPECT_GT(state.calls, 0u);
      if (threads == 0) EXPECT_EQ(state.calls, 1u);
      ASSERT_FALSE(tree.empty());
      for (const auto& node : tree) {
        if (node.property >= 0) {
          EXPECT_LT(node.lchild, tree.size());
          EXPECT_LT(node.rchild, tree.size());
        }
      }
      std::vector<Token> tokens;
      Tree decoded_tree;
      EXPECT_TRUE(TokenizeTree(tree, &tokens, &decoded_tree));
    }
    state.calls = 0;
    EXPECT_TRUE(LearnTree(&image, &options, 0, 1, {}, nullptr, &pool).ok());
    EXPECT_EQ(state.calls, 1u);
    state.fail = true;
    EXPECT_FALSE(LearnTree(&image, &options, 0, 1, {}, &pool).ok());
    EXPECT_FALSE(LearnTree(&image, &options, 0, 1, {}, nullptr, &pool).ok());
    // A supplied pool must not accidentally enable this path for lossy work.
    options.lossless_tree_speedups = false;
    state.calls = 0;
    EXPECT_TRUE(LearnTree(&image, &options, 0, 1, {}, &pool, &pool).ok());
    EXPECT_EQ(state.calls, 0u);
    if (threads > 0) {
      options.lossless_tree_speedups = true;
      state.fail = false;
      state.alternate = true;
      EXPECT_TRUE(LearnTree(&image, &options, 0, 1, {}, &pool).ok());
      EXPECT_GT(state.calls, 2u);
    }
  }
}

TEST(ModularTest, MaSubtreesPreserveSamplesAndRunnerErrors) {
  const char* enabled = std::getenv("JXL_MA_SUBTREE_PARALLEL");
  if (!enabled || std::strcmp(enabled, "1") != 0) GTEST_SKIP();
  TreeSamples original(true);
  ASSERT_TRUE(original.SetPredictor(Predictor::Variable,
                                    ModularOptions::TreeMode::kDefault));
  ASSERT_TRUE(original.SetProperties({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
                                     ModularOptions::TreeMode::kDefault));
  StaticPropRange range{};
  for (auto& r : range) r = {{0, 256}};
  std::vector<pixel_type> pixels{-65535, -511, -127, -1, 0, 1, 127, 511, 65535};
  auto differences = pixels;
  original.PreQuantizeProperties(range, {}, {1024}, {1024}, pixels,
                                  differences, 32);
  original.PrepareForSamples(100000);
  uint32_t random = 0x243f6a88;
  const auto next = [&]() {
    random ^= random << 13;
    random ^= random >> 17;
    random ^= random << 5;
    return random;
  };
  Properties properties(16);
  pixel_type_w predictions[kNumModularPredictors] = {};
  pixel_type_w pixel = 0;
  for (size_t i = 0; i < 98304; ++i) {
    properties[0] = i % 4;
    properties[1] = (i / 4) % 8;
    for (size_t p = 2; p < 12; ++p) {
      properties[p] = static_cast<int>(next() % 511) - 255;
    }
    pixel = next() & 65535;
    for (size_t p = 0; p < kNumModularPredictors; ++p) {
      predictions[p] = next() & 262143;
    }
    predictions[i % 4] = pixel - (next() % 31);
    original.AddSample(pixel, properties, predictions);
  }
  // A saturated duplicate count must survive independent subtree swaps too.
  for (size_t i = 0; i < 66000; ++i) {
    original.AddSample(pixel, properties, predictions);
  }
  original.AllSamplesDone();
  ASSERT_GT(original.NumDistinctSamples(), 65536u);
  const auto snapshot = [](const TreeSamples& samples) {
    std::vector<std::vector<uint32_t>> entries;
    for (size_t i = 0; i < samples.NumDistinctSamples(); ++i) {
      std::vector<uint32_t> entry{static_cast<uint32_t>(samples.Count(i))};
      for (size_t p = 0; p < samples.NumPredictors(); ++p) {
        entry.push_back(samples.RToken(p, i).tok);
        entry.push_back(samples.RToken(p, i).nbits);
      }
      for (size_t p = 0; p < samples.NumProperties(); ++p) {
        entry.push_back(p < samples.NumStaticProps()
                            ? samples.Property<true>(p, i)
                            : samples.Property<false>(
                                  p - samples.NumStaticProps(), i));
      }
      entries.push_back(std::move(entry));
    }
    std::sort(entries.begin(), entries.end());
    return entries;
  };
  const auto expected = snapshot(original);
  test::ThreadPoolForTests workers(4);
  test::ThreadPoolForTests single(0);
  struct RunnerState {
    ThreadPool* workers;
    ThreadPool* single;
    size_t subtree_calls = 0;
    std::atomic<bool> inside{false};
    std::atomic<bool> nested{false};
    bool fewer = false;
    bool fail = false;
  } state{workers.get(), single.get()};
  ThreadPool pool(
      [](void* opaque, void* context, JxlParallelRunInit init,
         JxlParallelRunFunction run, uint32_t begin,
         uint32_t end) -> JxlParallelRetCode {
        auto& s = *static_cast<RunnerState*>(opaque);
        if (s.inside) {
          s.nested = true;
          return JXL_PARALLEL_RET_RUNNER_ERROR;
        }
        // This fixture has 12 property tasks and 14 predictor tasks.
        const bool subtrees = end - begin <= 8;
        if (subtrees) ++s.subtree_calls;
        if (subtrees && s.fail) return JXL_PARALLEL_RET_RUNNER_ERROR;
        ThreadPool* runner = subtrees && s.fewer ? s.single : s.workers;
        s.inside = true;
        const auto result = runner->runner()(
            runner->runner_opaque(), context, init, run, begin, end);
        s.inside = false;
        return result;
      },
      &state);
  for (bool fewer : {false, true}) {
    state.fewer = fewer;
    state.subtree_calls = 0;
    TreeSamples samples = original;
    Tree tree;
    ASSERT_TRUE(ComputeBestTree(samples, 100, {}, range, 1.0f, &tree, &pool));
    EXPECT_EQ(state.subtree_calls, 1u);
    EXPECT_FALSE(state.nested);
    EXPECT_EQ(samples.NumSamples(), original.NumSamples());
    EXPECT_EQ(snapshot(samples), expected);
    std::vector<uint8_t> visited(tree.size());
    std::vector<size_t> pending{0};
    while (!pending.empty()) {
      const size_t index = pending.back();
      pending.pop_back();
      ASSERT_LT(index, tree.size());
      ASSERT_EQ(visited[index], 0);
      visited[index] = 1;
      const auto& node = tree[index];
      if (node.property >= 0) {
        pending.push_back(node.lchild);
        pending.push_back(node.rchild);
      }
    }
    EXPECT_EQ(std::count(visited.begin(), visited.end(), 1), tree.size());
    std::vector<Token> tokens;
    Tree decoded;
    EXPECT_TRUE(TokenizeTree(tree, &tokens, &decoded));
  }
  state.fail = true;
  TreeSamples samples = original;
  Tree tree;
  EXPECT_FALSE(ComputeBestTree(samples, 100, {}, range, 1.0f, &tree, &pool));
  EXPECT_FALSE(state.nested);
}

TEST(ModularTest, MaChannelTreesPreserveSamplesAndRunnerErrors) {
  const char* mode = std::getenv("JXL_MA_CHANNEL_TREE");
  if (!mode || mode[0] < '1' || mode[0] > '3' || mode[1] != '\0') GTEST_SKIP();
  EXPECT_FALSE(TreeSamples(false, false, true).ChannelTreeEligible());
  EXPECT_FALSE(TreeSamples(true).ChannelTreeEligible());
  TreeSamples original(true, false, true);
  ASSERT_TRUE(original.SetPredictor(Predictor::Variable, ModularOptions::TreeMode::kDefault));
  ASSERT_TRUE(original.SetProperties({0, 1, 2, 3, 4, 5, 6, 9}, ModularOptions::TreeMode::kDefault));
  StaticPropRange range{};
  for (auto& r : range) r = {{0, 256}};
  std::vector<pixel_type> pixels{-65535, -511, -1, 0, 1, 511, 65535};
  auto differences = pixels;
  original.PreQuantizeProperties(range, {}, {65536}, {65536, 65536, 65536, 65536},
                                  pixels, differences, 32);
  original.PrepareForSamples(131072);
  uint32_t random = 0x12345678;
  Properties properties(16);
  pixel_type_w predictions[kNumModularPredictors] = {};
  pixel_type_w pixel = 0;
  for (size_t i = 0; i < 131072; ++i) {
    properties[0] = i % 4;
    properties[1] = (i / 4) % 2;
    properties[2] = (i / 256) % 256;
    properties[3] = i % 256;
    for (size_t p = 0; p < kNumModularPredictors; ++p) {
      random ^= random << 13;
      random ^= random >> 17;
      random ^= random << 5;
      predictions[p] = random & 65535;
    }
    pixel = predictions[i % 4] + i % 31;
    original.AddSample(pixel, properties, predictions);
  }
  for (size_t i = 0; i < 66000; ++i) original.AddSample(pixel, properties, predictions);
  original.AllSamplesDone();
  const auto snapshot = [](const TreeSamples& samples) {
    std::vector<std::vector<uint32_t>> result;
    for (size_t i = 0; i < samples.NumDistinctSamples(); ++i) {
      std::vector<uint32_t> row{static_cast<uint32_t>(samples.Count(i))};
      for (size_t p = 0; p < samples.NumPredictors(); ++p) {
        row.push_back(samples.RToken(p, i).tok);
        row.push_back(samples.RToken(p, i).nbits);
      }
      for (size_t p = 0; p < samples.NumProperties(); ++p)
        row.push_back(p < samples.NumStaticProps() ? samples.Property<true>(p, i)
            : samples.Property<false>(p - samples.NumStaticProps(), i));
      result.push_back(std::move(row));
    }
    std::sort(result.begin(), result.end());
    return result;
  };
  const auto expected = snapshot(original);
  for (int threads : {0, 4}) {
    test::ThreadPoolForTests delegate(threads);
    bool fail = false;
    struct State {
      ThreadPool* delegate;
      bool* fail;
      std::atomic<bool> inside{false};
      std::atomic<size_t> calls{0};
    };
    State state{delegate.get(), &fail};
    ThreadPool pool([](void* opaque, void* context, JxlParallelRunInit init,
                      JxlParallelRunFunction fn, uint32_t begin, uint32_t end) {
      auto& s = *static_cast<State*>(opaque);
      if (s.inside || *s.fail) return JXL_PARALLEL_RET_RUNNER_ERROR;
      ++s.calls;
      s.inside = true;
      const auto result = s.delegate->runner()(s.delegate->runner_opaque(), context, init, fn, begin, end);
      s.inside = false;
      return result;
    }, &state);
    TreeSamples samples = original;
    Tree tree;
    ASSERT_TRUE(ComputeBestTree(samples, 100, {}, range, 1.0f, &tree, &pool));
    ASSERT_FALSE(tree.empty());
    EXPECT_EQ(tree[0].property, 0);
    EXPECT_EQ(state.calls.load(), 1u);
    EXPECT_EQ(snapshot(samples), expected);
    std::vector<uint8_t> visited(tree.size());
    std::vector<size_t> pending{0};
    while (!pending.empty()) {
      const size_t i = pending.back();
      pending.pop_back();
      ASSERT_LT(i, tree.size());
      ASSERT_EQ(visited[i], 0);
      visited[i] = 1;
      if (tree[i].property >= 0) {
        pending.push_back(tree[i].lchild);
        pending.push_back(tree[i].rchild);
      }
    }
    EXPECT_EQ(std::count(visited.begin(), visited.end(), 1), tree.size());
    std::vector<Token> tokens;
    Tree decoded;
    ASSERT_TRUE(TokenizeTree(tree, &tokens, &decoded));
    samples = original;
    tree.clear();
    fail = true;
    EXPECT_FALSE(ComputeBestTree(samples, 100, {}, range, 1.0f, &tree, &pool));
  }
}

TEST(ModularTest, MaWeightedSampleMerge) {
  using Counts = std::map<std::vector<uint32_t>, uint64_t>;
  const auto accumulate = [](const TreeSamples& samples, Counts* counts) {
    for (size_t a = 0; a < samples.NumDistinctSamples(); ++a) {
      std::vector<uint32_t> key;
      for (size_t p = 0; p < samples.NumPredictors(); ++p) {
        key.push_back(samples.RToken(p, a).tok);
        key.push_back(samples.RToken(p, a).nbits);
      }
      for (size_t p = 0; p < samples.NumProperties(); ++p) {
        key.push_back(p < samples.NumStaticProps()
                          ? samples.Property<true>(p, a)
                          : samples.Property<false>(p - samples.NumStaticProps(), a));
      }
      (*counts)[key] += samples.Count(a);
    }
  };
  for (bool speedups : {false, true}) {
    for (Predictor predictor : {Predictor::Zero, Predictor::Best, Predictor::Variable}) {
      TreeSamples prototype(speedups);
      ASSERT_TRUE(prototype.SetPredictor(predictor, ModularOptions::TreeMode::kDefault));
      ASSERT_TRUE(prototype.SetProperties({0, 1, 2, 3, 6, 9},
                                         ModularOptions::TreeMode::kDefault));
      StaticPropRange range{};
      for (auto& r : range) r = {{0, 256}};
      std::vector<pixel_type> pixels{-511, -255, -64, -1, 0, 1, 64, 255, 511};
      auto differences = pixels;
      prototype.PreQuantizeProperties(range, {}, {4096}, {4096}, pixels, differences, 32);
      std::vector<TreeSamples> batches(4, prototype);
      const size_t repeats[] = {40000, 60000, 65536, 131071};
      const auto add = [](TreeSamples& samples, size_t value) {
        Properties properties(16);
        properties[2] = value % 127;
        properties[3] = value % 251;
        properties[6] = value % 511;
        properties[9] = value % 31;
        pixel_type_w predictions[kNumModularPredictors] = {};
        for (size_t p = 0; p < kNumModularPredictors; ++p) {
          predictions[p] = static_cast<int>(p) * (value % 17);
        }
        samples.AddSample(value % 4096, properties, predictions);
      };
      Counts expected;
      size_t total = 0;
      for (size_t b = 0; b < batches.size(); ++b) {
        batches[b].PrepareForSamples(2048);
        for (size_t i = 0; i < repeats[b]; ++i) add(batches[b], 1);
        for (size_t i = 0; i < 2048; ++i) add(batches[b], i + b * 13);
        total += batches[b].NumSamples();
        accumulate(batches[b], &expected);
      }
      TreeSamples waved = prototype;
      for (const auto& batch : batches) {
        std::vector<TreeSamples> wave{batch};
        ASSERT_TRUE(waved.AppendSamples(wave));
        EXPECT_EQ(wave[0].NumSamples(), 0u);
      }
      Counts wave_actual;
      accumulate(waved, &wave_actual);
      EXPECT_EQ(wave_actual, expected);
      EXPECT_EQ(waved.NumSamples(), total);
      waved.AllSamplesDone();
      std::vector<TreeSamples> late{batches[0]};
      EXPECT_FALSE(waved.AppendSamples(late));
      TreeSamples merged = prototype;
      ASSERT_TRUE(merged.MergeSamples(batches));
      EXPECT_EQ(merged.NumSamples(), total);
      for (const auto& batch : batches) EXPECT_EQ(batch.NumSamples(), 0u);
      Counts actual;
      accumulate(merged, &actual);
      EXPECT_EQ(actual, expected);
      // Saturated counts must not remain in the dedup table for later appends.
      TreeSamples one = prototype;
      one.PrepareForSamples(1);
      add(one, 1);
      accumulate(one, &expected);
      add(merged, 1);
      actual.clear();
      accumulate(merged, &actual);
      EXPECT_EQ(actual, expected);
      EXPECT_EQ(merged.NumSamples(), total + 1);
      TreeSamples empty = prototype;
      std::vector<TreeSamples> wrong(1, prototype);
      ASSERT_TRUE(wrong[0].SetPredictor(Predictor::Gradient,
                                      ModularOptions::TreeMode::kDefault));
      EXPECT_FALSE(empty.MergeSamples(wrong));
      std::vector<TreeSamples> none;
      EXPECT_TRUE(empty.MergeSamples(none));
    }
  }
}

TEST(ModularTest, MaActiveHistogramBlocks) {
  std::vector<int32_t> counts(257);
  for (size_t i : {0u, 3u, 17u, 64u, 255u, 256u}) counts[i] = i + 1;
  for (size_t lanes : {1u, 4u, 8u, 16u, 64u}) {
    const auto blocks = MaActiveHistogramBlocks(counts.data(), counts.size(), lanes);
    std::vector<bool> visited(counts.size());
    size_t last = 0;
    for (size_t i = 0; i < blocks.size(); ++i) {
      EXPECT_EQ(blocks[i] % lanes, 0u);
      if (i != 0) EXPECT_GT(blocks[i], last);
      last = blocks[i];
      bool active = false;
      for (size_t j = blocks[i]; j < std::min(counts.size(), blocks[i] + lanes); ++j) {
        active |= counts[j] != 0;
        visited[j] = true;
      }
      EXPECT_TRUE(active);
    }
    for (size_t i = 0; i < counts.size(); ++i) {
      if (counts[i] != 0) EXPECT_TRUE(visited[i]);
    }
  }
  std::fill(counts.begin(), counts.end(), 0);
  EXPECT_TRUE(MaActiveHistogramBlocks(counts.data(), counts.size(), 8).empty());
  EXPECT_TRUE(MaActiveHistogramBlocks(nullptr, 0, 8).empty());
}

TEST(ModularTest, WeightedFlatPreservesState) {
  JxlMemoryManager* memory = test::MemoryManager();
  for (size_t width : {1u, 2u, 9u, 127u}) {
    constexpr size_t height = 80;
    for (int bits : {8, 16, 28}) {
      JXL_TEST_ASSIGN_OR_DIE(auto channel,
                             Channel::Create(memory, width, height));
      uint32_t random = 0x243f6a88;
      for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
          random ^= random << 13;
          random ^= random >> 17;
          random ^= random << 5;
          const pixel_type level = y < 16 ? 0 : y < 32 ? 117 :
              y < 48 ? -191 : (1 << (bits - 1)) - 1;
          channel.Row(y)[x] = (y >= 64 || (y % 16 == 15 && x % 7 == 0))
              ? static_cast<int32_t>(random & ((1u << bits) - 1)) - (1 << (bits - 1))
              : level;
        }
      }
      for (int mode = 0; mode < 5; ++mode) {
        SCOPED_TRACE(testing::Message() << width << ":" << bits << ":" << mode);
        weighted::Header header;
        weighted::PredictorMode(mode, &header);
        weighted::State reference(header, width, height), fast(header, width, height);
        Properties expected(1), actual(1);
        for (size_t y = 0; y < height; ++y) {
          for (size_t x = 0; x < width; ++x) {
            const pixel_type_w w = x ? channel.Row(y)[x - 1] :
                y ? channel.Row(y - 1)[x] : 0;
            const pixel_type_w n = y ? channel.Row(y - 1)[x] : w;
            const pixel_type_w nw = x && y ? channel.Row(y - 1)[x - 1] : w;
            const pixel_type_w ne = y && x + 1 < width ? channel.Row(y - 1)[x + 1] : n;
            const pixel_type_w nn = y > 1 ? channel.Row(y - 2)[x] : n;
            const auto prediction = reference.Predict<true>(
                x, y, width, n, w, ne, nw, nn, &expected, 0);
            EXPECT_EQ(prediction, PredictWeightedFlat<true>(
                &fast, x, y, width, n, w, ne, nw, nn, &actual, 0));
            EXPECT_EQ(expected, actual);
            EXPECT_EQ(reference.pred, fast.pred);
            for (size_t i = 0; i < weighted::kNumPredictors; ++i) {
              EXPECT_EQ(reference.prediction[i], fast.prediction[i]);
            }
            reference.UpdateErrors(channel.Row(y)[x], x, y, width);
            fast.UpdateErrors(channel.Row(y)[x], x, y, width);
          }
          EXPECT_EQ(reference.error, fast.error);
          for (size_t i = 0; i < weighted::kNumPredictors; ++i) {
            EXPECT_EQ(reference.pred_errors[i], fast.pred_errors[i]);
          }
        }
      }
    }
  }
}

TEST(ModularTest, MaPredictorPairPreservesPropertiesAndState) {
  JxlMemoryManager* memory = test::MemoryManager();
  for (size_t width : {1u, 2u, 9u, 37u}) {
    constexpr size_t height = 19;
    for (int bits : {8, 16, 28}) {
      for (size_t reference_props : {0u, 8u}) {
        for (int mode = 0; mode < 4; ++mode) {
          SCOPED_TRACE(testing::Message() << width << ":" << bits << ":"
                                         << reference_props << ":" << mode);
          JXL_TEST_ASSIGN_OR_DIE(auto channel, Channel::Create(memory, width, height));
          JXL_TEST_ASSIGN_OR_DIE(auto references,
                                 Channel::Create(memory, reference_props, width));
          uint32_t random = 0x243f6a88;
          for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
              random ^= random << 13;
              random ^= random >> 17;
              random ^= random << 5;
              const int32_t value = random & ((1u << bits) - 1);
              channel.Row(y)[x] = y < 3 ? 0 : value - (1 << (bits - 1));
            }
          }
          for (size_t x = 0; x < width; ++x) {
            for (size_t p = 0; p < reference_props; ++p) {
              references.Row(x)[p] = static_cast<int>(x * 29 + p * 13) - 255;
            }
          }
          weighted::Header header;
          weighted::PredictorMode(mode, &header);
          weighted::State full(header, width, height), pair(header, width, height),
              sparse(header, width, height);
          Properties expected(kNumNonrefProperties + reference_props);
          Properties actual(expected.size());
          Properties selected(expected.size());
          const ptrdiff_t stride = channel.plane.PixelsPerRow();
          for (size_t y = 0; y < height; ++y) {
            InitPropsRow(&expected, {{1, 7}}, y);
            InitPropsRow(&actual, {{1, 7}}, y);
            InitPropsRow(&selected, {{1, 7}}, y);
            for (size_t x = 0; x < width; ++x) {
              const pixel_type* pixel = channel.Row(y) + x;
              pixel_type_w all[kNumModularPredictors] = {};
              pixel_type_w best[kNumModularPredictors] = {};
              pixel_type_w sampled[kNumModularPredictors] = {};
              const bool keep = (x + 7 * y) % 13 == 0 || x + 1 == width;
              // Exercise border and no-edge paths and their shared WP history.
              if (y > 1 && x > 1 && x + 2 < width && reference_props == 0) {
                PredictLearnAllNEC(&expected, width, pixel, stride, x, y,
                                    references, &full, all);
                PredictLearnBest<true>(&actual, width, pixel, stride, x, y,
                                        references, &pair, best);
                if (keep) {
                  PredictLearnAllNEC(&selected, width, pixel, stride, x, y,
                                      references, &sparse, sampled);
                } else {
                  AdvanceLearnState<true>(&selected, width, pixel, stride, x, y,
                                          &sparse, true);
                }
              } else {
                PredictLearnAll(&expected, width, pixel, stride, x, y,
                                 references, &full, all);
                PredictLearnBest<false>(&actual, width, pixel, stride, x, y,
                                         references, &pair, best);
                if (keep) {
                  PredictLearnAll(&selected, width, pixel, stride, x, y,
                                   references, &sparse, sampled);
                } else {
                  AdvanceLearnState<false>(&selected, width, pixel, stride, x, y,
                                           &sparse, true);
                }
              }
              EXPECT_EQ(expected, actual);
              EXPECT_EQ(expected[9], selected[9]);
              if (keep) {
                EXPECT_EQ(expected, selected);
                for (size_t p = 0; p < kNumModularPredictors; ++p) {
                  EXPECT_EQ(all[p], sampled[p]);
                }
              }
              for (Predictor p : {Predictor::Weighted, Predictor::Gradient}) {
                EXPECT_EQ(all[static_cast<size_t>(p)], best[static_cast<size_t>(p)]);
              }
              full.UpdateErrors(*pixel, x, y, width);
              pair.UpdateErrors(*pixel, x, y, width);
              sparse.UpdateErrors(*pixel, x, y, width);
            }
          }
        }
      }
    }
  }
}

TEST(ModularTest, MaSparseReferences) {
  JxlMemoryManager* memory = test::MemoryManager();
  for (size_t width : {1u, 2u, 9u, 37u}) {
    constexpr size_t height = 7;
    JXL_TEST_ASSIGN_OR_DIE(auto image, Image::Create(memory, width, height, 16, 5));
    uint32_t random = 0x243f6a88;
    for (auto& channel : image.channel) {
      for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
          random ^= random << 13;
          random ^= random >> 17;
          random ^= random << 5;
          channel.Row(y)[x] = static_cast<int32_t>(random & 65535) - 32768;
        }
      }
    }
    // Exercise skipped incompatible channels, missing references and boundaries.
    image.channel[1].hshift = 1;
    image.channel[3].vshift = 1;
    for (size_t c = 0; c < image.channel.size(); ++c) {
      for (size_t props : {4u, 8u, 24u}) {
        JXL_TEST_ASSIGN_OR_DIE(auto expected, Channel::Create(memory, props, width));
        JXL_TEST_ASSIGN_OR_DIE(auto actual, Channel::Create(memory, props, width));
        SparseLearnReferences sparse(image, c, props);
        for (size_t y = 0; y < height; ++y) {
          PrecomputeReferences(image.channel[c], y, image, c, &expected);
          for (size_t x = 0; x < width; ++x) {
            std::fill_n(actual.Row(x), props, -12345);
            sparse.Fill(x, y, &actual);
            for (size_t p = 0; p < props; ++p) {
              EXPECT_EQ(expected.Row(x)[p], actual.Row(x)[p]);
            }
          }
        }
      }
    }
  }
}

TEST(ModularTest, MaDedupPreservesWeightedSamples) {
  EXPECT_FALSE(TreeSamples().FastDedup());
  for (Predictor predictor :
       {Predictor::Zero, Predictor::Best, Predictor::Variable}) {
    for (bool accelerated : {false, true}) {
      SCOPED_TRACE(testing::Message() << static_cast<int>(predictor)
                                     << ":" << accelerated);
      TreeSamples samples(accelerated);
      ASSERT_TRUE(samples.SetPredictor(predictor,
                                      ModularOptions::TreeMode::kDefault));
      ASSERT_TRUE(samples.SetProperties({0, 1, 2, 3, 6, 9},
                                       ModularOptions::TreeMode::kDefault));
      std::vector<pixel_type> pixels{-511, -255, -64, -1, 0, 1, 64, 255, 511};
      auto differences = pixels;
      StaticPropRange range{};
      for (auto& r : range) r = {{0, 256}};
      samples.PreQuantizeProperties(range, {}, {4096}, {4096}, pixels,
                                    differences, 32);
      samples.PrepareForSamples(8);
      using SampleCounts = std::map<std::vector<uint32_t>, uint64_t>;
      SampleCounts expected;
      const auto add = [&](int i) {
        Properties properties(16);
        properties[0] = i % 7;
        properties[1] = i % 13;
        properties[2] = i % 257;
        properties[3] = (i * 7) % 1024 - 511;
        properties[6] = (i * 31) % 1024 - 511;
        properties[9] = (i * 67) % 1024 - 511;
        pixel_type_w predictions[static_cast<size_t>(Predictor::Best)] = {};
        for (size_t p = 0; p < static_cast<size_t>(Predictor::Best); ++p) {
          predictions[p] = (i % 17) * static_cast<int>(p) - 1024;
        }
        const pixel_type pixel = (i % 23) * 139 - 2048;
        std::vector<uint32_t> key;
        for (size_t p = 0; p < samples.NumPredictors(); ++p) {
          const pixel_type residual =
              pixel - predictions[static_cast<size_t>(samples.PredictorFromIndex(p))];
          uint32_t tok, nbits, bits;
          HybridUintConfig(4, 1, 2).Encode(PackSigned(residual), &tok, &nbits, &bits);
          key.push_back(tok);
          key.push_back(nbits);
        }
        for (size_t p = 0; p < samples.NumProperties(); ++p) {
          const pixel_type value = properties[samples.PropertyFromIndex(p)];
          key.push_back(p < samples.NumStaticProps()
                            ? samples.QuantizeStaticProperty(p, value)
                            : samples.QuantizeProperty(p, value));
        }
        ++expected[key];
        samples.AddSample(pixel, properties, predictions);
      };
      // Saturate two 16-bit counts, then exercise collisions and table growth.
      for (int i = 0; i < 131077; ++i) add(0);
      for (int i = 1; i <= 128; ++i) add(i);
      for (int batch = 0; batch < 3; ++batch) {
        samples.PrepareForSamples(1000 * (batch + 1));
        for (int i = 0; i < 1000; ++i) add((i * 373 + batch * 997) % 4096);
      }
      ASSERT_EQ(samples.NumSamples(), 134205u);
      samples.AllSamplesDone();
      SampleCounts actual;
      for (size_t i = 0; i < samples.NumDistinctSamples(); ++i) {
        std::vector<uint32_t> key;
        for (size_t p = 0; p < samples.NumPredictors(); ++p) {
          key.push_back(samples.RToken(p, i).tok);
          key.push_back(samples.RToken(p, i).nbits);
        }
        for (size_t p = 0; p < samples.NumProperties(); ++p) {
          key.push_back(p < samples.NumStaticProps()
                            ? samples.Property<true>(p, i)
                            : samples.Property<false>(p - samples.NumStaticProps(), i));
        }
        actual[key] += samples.Count(i);
      }
      EXPECT_EQ(actual, expected);
    }
  }
}

TEST(ModularTest, MaDedupGrowthBeforeCountSaturation) {
  TreeSamples samples(true);
  if (!samples.FastDedup()) GTEST_SKIP() << "Fast deduplication disabled";
  ASSERT_TRUE(samples.SetPredictor(Predictor::Zero,
                                  ModularOptions::TreeMode::kDefault));
  ASSERT_TRUE(samples.SetProperties({0}, ModularOptions::TreeMode::kDefault));
  std::vector<pixel_type> pixels{0, 1};
  auto differences = pixels;
  StaticPropRange range{};
  for (auto& r : range) r = {{0, 256}};
  samples.PreQuantizeProperties(range, {}, {1}, {1}, pixels, differences, 32);
  samples.PrepareForSamples(8);
  Properties properties(16);
  pixel_type_w predictions[static_cast<size_t>(Predictor::Best)] = {};
  // Token 2 hashes to the same primary bucket before and after this growth.
  for (size_t i = 0; i < 65530; ++i) samples.AddSample(1, properties, predictions);
  samples.PrepareForSamples(16);
  for (size_t i = 0; i < 100; ++i) samples.AddSample(1, properties, predictions);
  ASSERT_EQ(samples.NumSamples(), 65630u);
  size_t count = 0;
  for (size_t i = 0; i < samples.NumDistinctSamples(); ++i) {
    EXPECT_EQ(samples.RToken(0, i).tok, 2);
    EXPECT_EQ(samples.RToken(0, i).nbits, 0);
    count += samples.Count(i);
  }
  EXPECT_EQ(count, 65630u);
}

TEST(ModularTest, MaPropertyPartitionVerification) {
  for (bool reverse : {false, true}) {
    TreeSamples samples(true);
    ASSERT_TRUE(samples.SetPredictor(Predictor::Best,
                                    ModularOptions::TreeMode::kDefault));
    ASSERT_TRUE(samples.SetProperties({0, 1, 2, 3},
                                     ModularOptions::TreeMode::kDefault));
    std::vector<pixel_type> pixels{-255, -1, 0, 1, 255};
    auto differences = pixels;
    StaticPropRange range{};
    for (auto& r : range) r = {{0, 256}};
    samples.PreQuantizeProperties(range, {}, {128}, {128}, pixels, differences, 32);
    samples.PrepareForSamples(512);
    pixel_type_w predictions[static_cast<size_t>(Predictor::Best)] = {};
    for (int i = 0; i < 128; ++i) {
      Properties properties(16);
      properties[0] = properties[1] = i % 3;
      properties[2] = i;
      properties[3] = reverse ? 127 - i : i;
      samples.AddSample(i, properties, predictions);
    }
    samples.AllSamplesDone();
    const size_t count = samples.NumDistinctSamples();
    ASSERT_GT(count, 1u);
    EXPECT_TRUE(MaSamePropertyPartition<true>(samples, 0, 1, 0, count));
    EXPECT_EQ(MaSamePropertyPartition<false>(samples, 0, 1, 0, count), !reverse);
    EXPECT_TRUE(MaSamePropertyPartition<false>(samples, 0, 1, count, count));
    std::map<uint32_t, size_t> first, second;
    for (size_t i = 0; i < count; ++i) {
      first[samples.Property<false>(0, i)] += samples.Count(i);
      second[samples.Property<false>(1, i)] += samples.Count(i);
    }
    // Equal marginals do not prove equal membership of the two partitions.
    EXPECT_EQ(first, second);
  }
}

TEST(ModularTest, MaMultiPredictorHistograms) {
  for (auto mode : {ModularOptions::TreeMode::kDefault,
                    ModularOptions::TreeMode::kNoWP}) {
    TreeSamples samples(true);
    ASSERT_TRUE(samples.SetPredictor(Predictor::Variable, mode));
    ASSERT_TRUE(samples.SetProperties({0, 1, 2, 3, 6, 9}, mode));
    std::vector<pixel_type> pixels{-65535, -255, -1, 0, 1, 255, 65535};
    auto differences = pixels;
    StaticPropRange range{};
    for (auto& r : range) r = {{0, 256}};
    samples.PreQuantizeProperties(range, {}, {128}, {128}, pixels, differences, 32);
    samples.PrepareForSamples(2048);
    pixel_type_w predictions[static_cast<size_t>(Predictor::Best)] = {};
    for (int i = 0; i < 513; ++i) {
      Properties properties(16);
      properties[0] = i % 3;
      properties[1] = i % 5;
      properties[2] = i;
      properties[3] = 512 - i;
      properties[6] = i - 256;
      properties[9] = (i % 7) * 300 - 900;
      for (size_t p = 0; p < static_cast<size_t>(Predictor::Best); ++p) {
        predictions[p] = static_cast<pixel_type_w>(
            (i * (p + 1) * 7919) % 65536) - 32768;
      }
      for (int repeat = 0; repeat <= i % 3; ++repeat) {
        samples.AddSample(i * 73 - 16000, properties, predictions);
      }
    }
    samples.AllSamplesDone();
    ASSERT_GT(samples.NumPredictors(), 2u);
    ASSERT_GT(samples.NumSamples(), samples.NumDistinctSamples());
    EXPECT_EQ(samples.NumPredictors() % 2,
              mode == ModularOptions::TreeMode::kNoWP ? 1u : 0u);
    for (size_t pred = 0; pred + 1 < samples.NumPredictors(); pred += 2) {
      for (size_t prop = 0; prop < samples.NumProperties(); ++prop) {
        const size_t values = samples.NumPropertyValues(prop);
        std::array<std::vector<int32_t>, 2> counts, expected;
        std::array<std::vector<size_t>, 2> extra, expected_extra;
        for (size_t p = 0; p < 2; ++p) {
          counts[p].assign(values * 256, 3);
          expected[p] = counts[p];
          extra[p].assign(values, 5);
          expected_extra[p] = extra[p];
          for (size_t i = 0; i < samples.NumDistinctSamples(); ++i) {
            const size_t value = prop < samples.NumStaticProps()
                ? samples.Property<true>(prop, i)
                : samples.Property<false>(prop - samples.NumStaticProps(), i);
            const auto token = samples.RToken(pred + p, i);
            expected[p][value * 256 + token.tok] += samples.Count(i);
            expected_extra[p][value] += samples.Count(i) * token.nbits;
          }
        }
        const size_t n = samples.NumDistinctSamples();
        for (const auto part : {std::pair<size_t, size_t>{0, 0}, {0, 1},
                               {1, n - 1}, {n - 1, n}}) {
          if (prop < samples.NumStaticProps()) {
            CollectPairedMaHistogram<true>(samples, prop, part.first, part.second,
                256, counts[0].data(), counts[1].data(), extra[0].data(), extra[1].data(), pred);
          } else {
            CollectPairedMaHistogram<false>(samples, prop - samples.NumStaticProps(),
                part.first, part.second, 256, counts[0].data(), counts[1].data(),
                extra[0].data(), extra[1].data(), pred);
          }
        }
        EXPECT_EQ(counts, expected) << pred << ':' << prop;
        EXPECT_EQ(extra, expected_extra) << pred << ':' << prop;
      }
    }
  }
}

TEST(ModularTest, MaSampleBatches) {
  TreeSamples original(true);
  ASSERT_TRUE(original.SetPredictor(Predictor::Best,
                                   ModularOptions::TreeMode::kDefault));
  ASSERT_TRUE(original.SetProperties({0, 1, 2, 3, 6, 9},
                                    ModularOptions::TreeMode::kDefault));
  std::vector<pixel_type> pixels{-255, -64, -1, 0, 1, 64, 255};
  auto differences = pixels;
  StaticPropRange range{};
  for (auto& r : range) r = {{0, 256}};
  original.PreQuantizeProperties(range, {}, {128}, {128}, pixels, differences, 32);
  original.PrepareForSamples(1024);
  pixel_type_w predictions[static_cast<size_t>(Predictor::Best)] = {};
  for (int i = 0; i < 256; ++i) {
    Properties properties(16);
    properties[0] = i % 3;
    properties[1] = i % 5;
    properties[2] = i;
    properties[3] = 255 - i;
    properties[6] = i - 128;
    properties[9] = (i % 7) * 30 - 90;
    for (size_t p = 0; p < static_cast<size_t>(Predictor::Best); ++p) {
      predictions[p] = (i * (p + 1)) % 71;
    }
    for (int repeat = 0; repeat <= i % 3; ++repeat) {
      original.AddSample(i * 3 - 100, properties, predictions);
    }
  }
  original.AllSamplesDone();
  const size_t samples = original.NumDistinctSamples();
  ASSERT_GT(samples, 128u);
  ASSERT_GT(original.NumSamples(), samples);
  ASSERT_EQ(original.NumPredictors(), 2u);
  for (size_t prop = 0; prop < original.NumProperties(); ++prop) {
    const size_t values = original.NumPropertyValues(prop);
    std::array<std::vector<int32_t>, 2> counts, expected;
    std::array<std::vector<size_t>, 2> extra, expected_extra;
    for (size_t pred = 0; pred < 2; ++pred) {
      counts[pred].resize(values * 256);
      expected[pred].resize(values * 256);
      extra[pred].resize(values);
      expected_extra[pred].resize(values);
      for (size_t i = 0; i < samples; ++i) {
        const size_t value = prop < original.NumStaticProps()
            ? original.Property<true>(prop, i)
            : original.Property<false>(prop - original.NumStaticProps(), i);
        const auto token = original.RToken(pred, i);
        expected[pred][value * 256 + token.tok] += original.Count(i);
        expected_extra[pred][value] += original.Count(i) * token.nbits;
      }
    }
    for (const auto range : {std::pair<size_t, size_t>{0, 0}, {0, 1},
                             {1, samples - 1}, {samples - 1, samples}}) {
      if (prop < original.NumStaticProps()) {
        CollectPairedMaHistogram<true>(
            original, prop, range.first, range.second, 256, counts[0].data(),
            counts[1].data(), extra[0].data(), extra[1].data());
      } else {
        CollectPairedMaHistogram<false>(
            original, prop - original.NumStaticProps(), range.first, range.second,
            256, counts[0].data(), counts[1].data(), extra[0].data(), extra[1].data());
      }
    }
    EXPECT_EQ(counts, expected);
    EXPECT_EQ(extra, expected_extra);
  }
  for (size_t count : {0, 1, 63, 128, 129, 255}) {
    SCOPED_TRACE(count);
    auto expected = original, actual = original;
    std::vector<size_t> left(count), right(count);
    for (size_t i = 0; i < count; ++i) {
      left[i] = (i * 13) % samples;
      right[i] = i % 17 == 0 ? left[i] : (i * 31 + 1) % samples;
      expected.Swap(left[i], right[i]);
    }
    actual.SwapBatch(left.data(), right.data(), count);
    EXPECT_EQ(actual.NumSamples(), expected.NumSamples());
    for (size_t i = 0; i < samples; ++i) {
      EXPECT_EQ(actual.Count(i), expected.Count(i));
      for (size_t pred = 0; pred < original.NumPredictors(); ++pred) {
        EXPECT_EQ(actual.RToken(pred, i).tok, expected.RToken(pred, i).tok);
        EXPECT_EQ(actual.RToken(pred, i).nbits, expected.RToken(pred, i).nbits);
      }
      for (size_t prop = 0; prop < original.NumProperties(); ++prop) {
        if (prop < original.NumStaticProps()) {
          EXPECT_EQ(actual.Property<true>(prop, i), expected.Property<true>(prop, i));
        } else {
          const size_t index = prop - original.NumStaticProps();
          EXPECT_EQ(actual.Property<false>(index, i), expected.Property<false>(index, i));
        }
      }
    }
  }
}

TEST(ModularTest, MaSampleBatchPreservesWeightedSamples) {
  for (Predictor predictor : {Predictor::Best, Predictor::Variable}) {
    TreeSamples scalar(true, false);
    ASSERT_TRUE(scalar.SetPredictor(predictor, ModularOptions::TreeMode::kDefault));
    ASSERT_TRUE(scalar.SetProperties({0, 1, 2, 3, 6, 9},
                                     ModularOptions::TreeMode::kDefault));
    std::vector<pixel_type> pixels{-255, -64, -1, 0, 1, 64, 255};
    auto differences = pixels;
    StaticPropRange range{};
    for (auto& r : range) r = {{0, 256}};
    scalar.PreQuantizeProperties(range, {}, {128}, {128}, pixels, differences, 32);
    auto batched = scalar;
    batched.SetSampleBatching(true);
    scalar.PrepareForSamples(4096);
    batched.PrepareForSamples(4096);
    pixel_type_w predictions[kNumModularPredictors] = {};
    for (size_t i = 0; i < 71037; ++i) {
      // Saturation, repeated samples, distinct samples, a partial batch and a
      // rehash while a partial batch is pending all exercise different cases.
      const int v = i < 70000 ? 0 : static_cast<int>((i * 31) % 257);
      Properties properties(16);
      properties[0] = v % 3;
      properties[1] = v % 5;
      properties[2] = v;
      properties[3] = 255 - v;
      properties[6] = v - 128;
      properties[9] = v % 7 * 30 - 90;
      for (size_t p = 0; p < kNumModularPredictors; ++p) {
        predictions[p] = (v * (p + 1)) % 71;
      }
      scalar.AddSample(v * 3 - 100, properties, predictions);
      batched.AddSample(v * 3 - 100, properties, predictions);
      if (i == 70119) {
        scalar.PrepareForSamples(8192);
        batched.PrepareForSamples(8192);
      }
    }
    scalar.AllSamplesDone();
    batched.AllSamplesDone();
    ASSERT_EQ(scalar.NumSamples(), batched.NumSamples());
    ASSERT_EQ(scalar.NumDistinctSamples(), batched.NumDistinctSamples());
    for (size_t i = 0; i < scalar.NumDistinctSamples(); ++i) {
      ASSERT_EQ(scalar.Count(i), batched.Count(i));
      for (size_t p = 0; p < scalar.NumPredictors(); ++p) {
        ASSERT_EQ(scalar.RToken(p, i).tok, batched.RToken(p, i).tok);
        ASSERT_EQ(scalar.RToken(p, i).nbits, batched.RToken(p, i).nbits);
      }
      for (size_t p = 0; p < scalar.NumProperties(); ++p) {
        if (p < scalar.NumStaticProps()) {
          ASSERT_EQ(scalar.Property<true>(p, i), batched.Property<true>(p, i));
        } else {
          const size_t index = p - scalar.NumStaticProps();
          ASSERT_EQ(scalar.Property<false>(index, i),
                    batched.Property<false>(index, i));
        }
      }
    }
  }
}

TEST(ModularTest, MaHistogramPredictorCopy) {
  TreeSamples samples(true, false);
  ASSERT_TRUE(samples.SetPredictor(Predictor::Variable,
                                  ModularOptions::TreeMode::kDefault));
  ASSERT_TRUE(samples.SetProperties({0, 1, 2, 3, 6, 9},
                                   ModularOptions::TreeMode::kDefault));
  std::vector<pixel_type> pixels{-255, -64, -1, 0, 1, 64, 255};
  auto differences = pixels;
  StaticPropRange range{};
  for (auto& r : range) r = {{0, 256}};
  samples.PreQuantizeProperties(range, {}, {128}, {128}, pixels, differences, 32);
  MaHistogramBudget budget;
  auto cache = MaHistogramCache::Create(samples, 72, &budget);
  ASSERT_TRUE(cache);
  for (size_t prop = 0; prop < samples.NumProperties(); ++prop) {
    const size_t values = samples.NumPropertyValues(prop);
    std::vector<int32_t> counts(values * 72);
    std::vector<size_t> extra(values);
    for (size_t i = 0; i < counts.size(); ++i) counts[i] = i * 3 + prop;
    for (size_t i = 0; i < extra.size(); ++i) extra[i] = i * 7 + prop;
    cache->Capture(prop, 0, counts.data(), extra.data(), 72);
    cache->CopyPredictor(prop, 0, 3);
    for (size_t stride : {size_t{72}, size_t{40}}) {
      std::vector<int32_t> actual(values * stride);
      std::vector<size_t> actual_extra(values);
      if (prop < samples.NumStaticProps()) {
        cache->SubtractAndCopy<true>(samples, prop, 3, 0, 0,
            actual.data(), actual_extra.data(), stride);
      } else {
        cache->SubtractAndCopy<false>(samples, prop, 3, 0, 0,
            actual.data(), actual_extra.data(), stride);
      }
      EXPECT_EQ(actual_extra, extra);
      for (size_t value = 0; value < values; ++value) {
        for (size_t symbol = 0; symbol < stride; ++symbol) {
          ASSERT_EQ(actual[value * stride + symbol], counts[value * 72 + symbol]);
        }
      }
    }
  }
}

TEST(ModularTest, MaHistogramSubtraction) {
  TreeSamples samples;
  ASSERT_TRUE(samples.SetPredictor(Predictor::Variable,
                                  ModularOptions::TreeMode::kDefault));
  ASSERT_TRUE(samples.SetProperties({0, 1, 2, 3, 6, 9},
                                   ModularOptions::TreeMode::kDefault));
  std::vector<pixel_type> pixels{-255, -64, -1, 0, 1, 64, 255};
  auto differences = pixels;
  StaticPropRange range{};
  for (auto& r : range) r = {{0, 256}};
  samples.PreQuantizeProperties(range, {}, {128}, {128}, pixels, differences, 32);
  samples.PrepareForSamples(512);
  pixel_type_w predictions[static_cast<size_t>(Predictor::Best)] = {};
  for (int i = 0; i < 128; ++i) {
    Properties properties(16);
    properties[2] = i;
    properties[3] = 127 - i;
    properties[6] = i - 64;
    properties[9] = (i % 7) * 30 - 90;
    for (size_t p = 0; p < static_cast<size_t>(Predictor::Best); ++p) {
      predictions[p] = (i * (p + 1)) % 71;
    }
    for (int repeat = 0; repeat < 3; ++repeat) {
      samples.AddSample(i * 3 - 100, properties, predictions);
    }
  }
  samples.AllSamplesDone();
  ASSERT_GT(samples.NumDistinctSamples(), 64u);
  ASSERT_GT(samples.NumSamples(), samples.NumDistinctSamples());
  MaHistogramBudget budget;
  auto cache = MaHistogramCache::Create(samples, 256, &budget, true);
  ASSERT_NE(cache, nullptr);
  ASSERT_GT(budget.used, 0u);
  for (size_t prop = 0; prop < samples.NumProperties(); ++prop) {
    const size_t values = samples.NumPropertyValues(prop);
    for (size_t pred = 0; pred < samples.NumPredictors(); ++pred) {
      std::vector<int32_t> counts(values * 256);
      std::vector<size_t> extra(values);
      std::vector<int> used(values);
      std::vector<uint32_t> weight(values);
      auto accumulate = [&](size_t first) {
        std::fill(counts.begin(), counts.end(), 0);
        std::fill(extra.begin(), extra.end(), 0);
        std::fill(used.begin(), used.end(), 0);
        std::fill(weight.begin(), weight.end(), 0);
        for (size_t i = first; i < samples.NumDistinctSamples(); ++i) {
          const size_t v = prop < samples.NumStaticProps()
              ? samples.Property<true>(prop, i)
              : samples.Property<false>(prop - samples.NumStaticProps(), i);
          const auto token = samples.RToken(pred, i);
          counts[v * 256 + token.tok] += samples.Count(i);
          extra[v] += token.nbits * samples.Count(i);
          ++used[v];
          weight[v] += samples.Count(i);
        }
      };
      accumulate(0);
      MaHistogramShards scratch;
      std::vector<int32_t> sharded_counts(values * 256);
      std::vector<size_t> sharded_extra(values);
      const size_t n = samples.NumDistinctSamples();
      for (const auto range : {std::pair<size_t, size_t>{0, 1}, {1, n - 3}, {n - 3, n}}) {
        if (prop < samples.NumStaticProps()) {
          CollectShardedMaHistogram<true>(samples, pred, prop, range.first, range.second,
              256, values, sharded_counts.data(), sharded_extra.data(), &scratch);
        } else {
          CollectShardedMaHistogram<false>(samples, pred, prop - samples.NumStaticProps(),
              range.first, range.second, 256, values, sharded_counts.data(),
              sharded_extra.data(), &scratch);
        }
      }
      EXPECT_EQ(sharded_counts, counts);
      EXPECT_EQ(sharded_extra, extra);
      cache->Capture(prop, pred, counts.data(), extra.data(), 256);
      cache->CaptureProperty(prop, used.data(), weight.data());
      // Consecutive subtractions cover weighted duplicates, every predictor,
      // static/dynamic properties, the last bin and empty resulting bins.
      for (size_t first = 0; first < 32; first += 16) {
        std::vector<int32_t> got_counts(values * 256);
        std::vector<size_t> got_extra(values);
        std::vector<int> got_used(values);
        std::vector<uint32_t> got_weight(values);
        if (prop < samples.NumStaticProps()) {
          cache->SubtractAndCopy<true>(samples, prop, pred, first, first + 16,
                                      got_counts.data(), got_extra.data(), 256);
          cache->SubtractProperty<true>(samples, prop, first, first + 16,
                                        got_used.data(), got_weight.data());
        } else {
          cache->SubtractAndCopy<false>(samples, prop, pred, first, first + 16,
                                       got_counts.data(), got_extra.data(), 256);
          cache->SubtractProperty<false>(samples, prop, first, first + 16,
                                         got_used.data(), got_weight.data());
        }
        accumulate(first + 16);
        EXPECT_EQ(got_counts, counts);
        EXPECT_EQ(got_extra, extra);
        EXPECT_EQ(got_used, used);
        EXPECT_EQ(got_weight, weight);
      }
      std::vector<int32_t> narrow(values * 64);
      std::vector<size_t> narrow_extra(values);
      if (prop < samples.NumStaticProps()) {
        cache->SubtractAndCopy<true>(samples, prop, pred, 32, 32,
                                    narrow.data(), narrow_extra.data(), 64);
      } else {
        cache->SubtractAndCopy<false>(samples, prop, pred, 32, 32,
                                     narrow.data(), narrow_extra.data(), 64);
      }
      for (size_t v = 0; v < values; ++v) {
        for (size_t symbol = 0; symbol < 64; ++symbol) {
          EXPECT_EQ(narrow[v * 64 + symbol], counts[v * 256 + symbol]);
        }
      }
      EXPECT_EQ(narrow_extra, extra);
      if (prop < samples.NumStaticProps()) {
        cache->SubtractProperty<true>(samples, prop, 32, samples.NumDistinctSamples(),
                                      used.data(), weight.data());
      } else {
        cache->SubtractProperty<false>(samples, prop, 32, samples.NumDistinctSamples(),
                                       used.data(), weight.data());
      }
      EXPECT_EQ(used, std::vector<int>(values, 0));
      EXPECT_EQ(weight, std::vector<uint32_t>(values, 0));
    }
    EXPECT_TRUE(cache->Has(prop));
    cache->Invalidate(prop);
    EXPECT_FALSE(cache->Has(prop));
  }
  // Paired subtraction must retain both independent predictor histograms,
  // including weighted samples, changing output strides and an empty child.
  for (size_t prop = 0; prop < samples.NumProperties(); ++prop) {
    const size_t values = samples.NumPropertyValues(prop);
    const size_t n = samples.NumDistinctSamples();
    for (size_t pred = 0; pred + 1 < samples.NumPredictors(); pred += 2) {
      std::array<std::vector<int32_t>, 2> expected, got;
      std::array<std::vector<size_t>, 2> expected_extra, got_extra;
      const auto accumulate = [&](size_t begin) {
        for (size_t p = 0; p < 2; ++p) {
          expected[p].assign(values * 256, 0);
          expected_extra[p].assign(values, 0);
          for (size_t i = begin; i < n; ++i) {
            const size_t v = prop < samples.NumStaticProps()
                ? samples.Property<true>(prop, i)
                : samples.Property<false>(prop - samples.NumStaticProps(), i);
            const auto token = samples.RToken(pred + p, i);
            expected[p][v * 256 + token.tok] += samples.Count(i);
            expected_extra[p][v] += token.nbits * samples.Count(i);
          }
        }
      };
      accumulate(0);
      for (size_t p = 0; p < 2; ++p) {
        cache->Capture(prop, pred + p, expected[p].data(),
                        expected_extra[p].data(), 256);
      }
      for (const auto interval : {std::pair<size_t, size_t>{0, 0}, {0, 1},
                                   {1, n / 2}, {n / 2, n - 1}, {n - 1, n}}) {
        const size_t symbols = interval.first % 2 ? 64 : 256;
        for (size_t p = 0; p < 2; ++p) {
          got[p].assign(values * symbols, -1);
          got_extra[p].assign(values, size_t(-1));
        }
        if (prop < samples.NumStaticProps()) {
          cache->SubtractPairedAndCopy<true>(
              samples, prop, pred, interval.first, interval.second, got[0].data(),
              got[1].data(), got_extra[0].data(), got_extra[1].data(), symbols);
        } else {
          cache->SubtractPairedAndCopy<false>(
              samples, prop, pred, interval.first, interval.second, got[0].data(),
              got[1].data(), got_extra[0].data(), got_extra[1].data(), symbols);
        }
        accumulate(interval.second);
        for (size_t p = 0; p < 2; ++p) {
          EXPECT_EQ(got_extra[p], expected_extra[p]);
          for (size_t v = 0; v < values; ++v) {
            for (size_t s = 0; s < symbols; ++s) {
              EXPECT_EQ(got[p][v * symbols + s], expected[p][v * 256 + s]);
            }
          }
        }
      }
    }
  }
  const size_t with_properties = budget.used;
  cache.reset();
  EXPECT_EQ(budget.used, 0u);
  auto token_only = MaHistogramCache::Create(samples, 256, &budget);
  ASSERT_NE(token_only, nullptr);
  size_t property_values = 0;
  for (size_t p = 0; p < samples.NumProperties(); ++p) {
    property_values += samples.NumPropertyValues(p);
  }
  const size_t property_bytes = property_values * (sizeof(int) + sizeof(uint32_t));
  EXPECT_EQ(with_properties - budget.used, property_bytes);
  token_only.reset();
  budget.used = MaHistogramBudget::kLimit - with_properties + 1;
  EXPECT_EQ(MaHistogramCache::Create(samples, 256, &budget, true), nullptr);
  EXPECT_NE(MaHistogramCache::Create(samples, 256, &budget), nullptr);
  budget.used = 0;
  EXPECT_EQ(MaHistogramCache::Create(samples, 0, &budget), nullptr);
  EXPECT_EQ(MaHistogramCache::Create(samples, 257, &budget), nullptr);
  budget.used = MaHistogramBudget::kLimit;
  EXPECT_EQ(MaHistogramCache::Create(samples, 256, &budget), nullptr);
  EXPECT_EQ(budget.used, MaHistogramBudget::kLimit);
}

void TestLosslessGroups(size_t group_size_shift) {
  const std::vector<uint8_t> orig = ReadTestData("jxl/flower/flower.png");
  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));
  t.ClearMetadata();
  ASSERT_TRUE(t.SetDimensions(t.ppf().xsize() / 4, t.ppf().ysize() / 4));

  extras::JXLCompressParams cparams;
  cparams.distance = 0.0f;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_MODULAR_GROUP_SIZE, group_size_shift);
  extras::JXLDecompressParams dparams;
  dparams.accepted_formats = {{3, JXL_TYPE_UINT8, JXL_LITTLE_ENDIAN, 0}};

  extras::PackedPixelFile ppf_out;
  size_t compressed_size =
      Roundtrip(t.ppf(), cparams, dparams, nullptr, &ppf_out);
  EXPECT_LE(compressed_size, 280000u);
  float disntance2 = test::ComputeDistance2(t.ppf(), ppf_out);
  EXPECT_EQ(0.0f, disntance2);
}

TEST(ModularTest, RoundtripLosslessGroups128) { TestLosslessGroups(0); }

JXL_TSAN_SLOW_TEST(ModularTest, RoundtripLosslessGroups512) {
  TestLosslessGroups(2);
}

JXL_TSAN_SLOW_TEST(ModularTest, RoundtripLosslessGroups1024) {
  TestLosslessGroups(3);
}

void TestLarge(size_t dim, size_t co_dim, size_t group_size_shift) {
  for (bool wide : {true, false}) {
    size_t w = dim;
    size_t h = co_dim;
    if (!wide) std::swap(w, h);
    TestImage t;
    ASSERT_TRUE(t.SetDimensions(w, h));
    JXL_TEST_ASSIGN_OR_DIE(auto frame, t.AddFrame());
    frame.ZeroFill();
    extras::JXLCompressParams cparams;
    cparams.AddOption(JXL_ENC_FRAME_SETTING_MODULAR_GROUP_SIZE, group_size_shift);
    cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 1);
    cparams.AddOption(JXL_ENC_FRAME_SETTING_MODULAR, 1);
    extras::JXLDecompressParams dparams;
    extras::PackedPixelFile ppf_out;
    size_t compressed_size =
        Roundtrip(t.ppf(), cparams, dparams, nullptr, &ppf_out);
    EXPECT_LE(compressed_size, 16384u);
  }
}

TEST(ModularTest, LargeGss0) {
  TestLarge(514 * 1024, 1, 0);
}

TEST(ModularTest, LargeGss1) {
  TestLarge(514 * 1024, 1, 1);
}

TEST(ModularTest, LargeGss2) {
  TestLarge(514 * 1024, 1, 2);
}

TEST(ModularTest, LargeGss3) {
  TestLarge(514 * 1024, 1, 3);
}

TEST(ModularTest, LargeDcGss0) {
  TestLarge(129 * 1024, 64, 0);
}

/* DISABLED: uses 10+GiB memory */
/*
TEST(ModularTest, LargeDcGss1) {
  TestLarge(514 * 1024, 64, 1);
}

TEST(ModularTest, LargeDcGss2) {
  TestLarge(2051 * 1024, 64, 2);
}
*/

TEST(ModularTest, RoundtripLosslessCustomWpPermuteRCT) {
  const std::vector<uint8_t> orig =
      ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));
  t.ClearMetadata();
  ASSERT_TRUE(t.SetDimensions(100, 100));

  extras::JXLCompressParams cparams;
  cparams.distance = 0.0f;
  // 9 = permute to GBR, to test the special case of permutation-only
  cparams.AddOption(JXL_ENC_FRAME_SETTING_MODULAR_COLOR_SPACE, 9);
  cparams.AddOption(JXL_ENC_FRAME_SETTING_MODULAR_PREDICTOR,
                    static_cast<int64_t>(Predictor::Weighted));
  // slowest speed so different WP modes are tried
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 9);
  extras::JXLDecompressParams dparams;
  dparams.accepted_formats = {{3, JXL_TYPE_UINT8, JXL_LITTLE_ENDIAN, 0}};

  extras::PackedPixelFile ppf_out;
  size_t compressed_size =
      Roundtrip(t.ppf(), cparams, dparams, nullptr, &ppf_out);
  EXPECT_LE(compressed_size, 10169u);
  EXPECT_EQ(0.0f, test::ComputeDistance2(t.ppf(), ppf_out));
}

TEST(ModularTest, RoundtripLossyDeltaPalette) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  const std::vector<uint8_t> orig =
      ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CompressParams cparams;
  cparams.modular_mode = true;
  cparams.color_transform = jxl::ColorTransform::kNone;
  cparams.lossy_palette = true;
  cparams.palette_colors = 0;
  extras::JXLDecompressParams dparams;

  auto io = jxl::make_unique<CodecInOut>(memory_manager);

  auto io_out = jxl::make_unique<CodecInOut>(memory_manager);
  ASSERT_TRUE(SetFromBytes(Bytes(orig), io.get()));
  ASSERT_TRUE(io->ShrinkTo(300, 100));

  size_t compressed_size;
  JXL_EXPECT_OK(
      Roundtrip(io.get(), cparams, dparams, io_out.get(), _, &compressed_size));
  EXPECT_LE(compressed_size, 6800u);
  EXPECT_SLIGHTLY_BELOW(
      ButteraugliDistance(io->frames, io_out->frames, ButteraugliParams(),
                          *JxlGetDefaultCms(),
                          /*distmap=*/nullptr),
      1.5);
}
TEST(ModularTest, RoundtripLossyDeltaPaletteWP) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  const std::vector<uint8_t> orig =
      ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CompressParams cparams;
  cparams.SetLossless();
  cparams.lossy_palette = true;
  cparams.palette_colors = 0;
  // TODO(jon): this is currently ignored, and Avg4 is always used instead
  cparams.options.predictor = jxl::Predictor::Weighted;
  extras::JXLDecompressParams dparams;

  auto io_out = jxl::make_unique<CodecInOut>(memory_manager);

  auto io = jxl::make_unique<CodecInOut>(memory_manager);
  ASSERT_TRUE(SetFromBytes(Bytes(orig), io.get()));
  ASSERT_TRUE(io->ShrinkTo(300, 100));

  size_t compressed_size;
  JXL_EXPECT_OK(
      Roundtrip(io.get(), cparams, dparams, io_out.get(), _, &compressed_size));
  EXPECT_LE(compressed_size, 6500u);
  EXPECT_SLIGHTLY_BELOW(
      ButteraugliDistance(io->frames, io_out->frames, ButteraugliParams(),
                          *JxlGetDefaultCms(),
                          /*distmap=*/nullptr),
      1.5);
}

TEST(ModularTest, RoundtripLossy) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  const std::vector<uint8_t> orig =
      ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CompressParams cparams;
  cparams.modular_mode = true;
  cparams.butteraugli_distance = 2.f;
  cparams.SetCms(*JxlGetDefaultCms());
  extras::JXLDecompressParams dparams;

  auto io_out = jxl::make_unique<CodecInOut>(memory_manager);

  auto io = jxl::make_unique<CodecInOut>(memory_manager);
  ASSERT_TRUE(SetFromBytes(Bytes(orig), io.get()));

  size_t compressed_size;
  JXL_EXPECT_OK(
      Roundtrip(io.get(), cparams, dparams, io_out.get(), _, &compressed_size));
  EXPECT_LE(compressed_size, 23000u);
  EXPECT_SLIGHTLY_BELOW(
      ButteraugliDistance(io->frames, io_out->frames, ButteraugliParams(),
                          *JxlGetDefaultCms(),
                          /*distmap=*/nullptr),
      3.0);
}

TEST(ModularTest, RoundtripLossy16) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  const std::vector<uint8_t> orig =
      ReadTestData("external/raw.pixls/DJI-FC6310-16bit_709_v4_krita.png");
  CompressParams cparams;
  cparams.modular_mode = true;
  cparams.butteraugli_distance = 2.f;
  extras::JXLDecompressParams dparams;

  auto io_out = jxl::make_unique<CodecInOut>(memory_manager);

  auto io = jxl::make_unique<CodecInOut>(memory_manager);
  ASSERT_TRUE(SetFromBytes(Bytes(orig), io.get()));
  ASSERT_TRUE(!io->metadata.m.have_preview);
  ASSERT_TRUE(io->frames.size() == 1);
  ASSERT_TRUE(
      io->frames[0].TransformTo(ColorEncoding::SRGB(), *JxlGetDefaultCms()));
  io->metadata.m.color_encoding = ColorEncoding::SRGB();

  size_t compressed_size;
  JXL_EXPECT_OK(
      Roundtrip(io.get(), cparams, dparams, io_out.get(), _, &compressed_size));
  EXPECT_LE(compressed_size, 230u);
  EXPECT_SLIGHTLY_BELOW(
      ButteraugliDistance(io->frames, io_out->frames, ButteraugliParams(),
                          *JxlGetDefaultCms(),
                          /*distmap=*/nullptr),
      1.8);
}

TEST(ModularTest, RoundtripPredictorTokenPaths) {
  auto* memory_manager = test::MemoryManager();
  const std::array<std::pair<size_t, size_t>, 4> sizes = {
      {{1, 1}, {1, 17}, {23, 1}, {31, 19}}};
  for (auto size : sizes) {
    const size_t width = size.first, height = size.second;
    JXL_TEST_ASSIGN_OR_DIE(Image image,
                          Image::Create(memory_manager, width, height, 16, 3));
    for (size_t c = 0; c < 3; ++c) {
      for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
          image.channel[c].plane.Row(y)[x] =
              (x * 1997 + y * 997 + (x / 7) * (y / 3) * (c + 1) * 31) % 65536;
        }
      }
    }
    for (size_t predictor = 0; predictor < kNumModularPredictors; ++predictor) {
      for (bool skip_fast : {false, true}) {
        SCOPED_TRACE(testing::Message() << width << "x" << height
                                       << " predictor=" << predictor
                                       << " skip_fast=" << skip_fast);
        ModularOptions options;
        options.predictor = static_cast<Predictor>(predictor);
        options.nb_repeats = 0;
        options.skip_encoder_fast_path = skip_fast;
        BitWriter writer{memory_manager};
        ASSERT_TRUE(ModularGenericCompress(image, options, writer));
        writer.ZeroPadToByte();
        JXL_TEST_ASSIGN_OR_DIE(
            Image decoded, Image::Create(memory_manager, width, height, 16, 3));
        Status status = true;
        {
          BitReader reader(writer.GetSpan());
          BitReaderScopedCloser closer(reader, status);
          ASSERT_TRUE(
              ModularGenericDecompress(&reader, decoded, nullptr, 0, &options));
        }
        ASSERT_TRUE(status);
        for (size_t c = 0; c < 3; ++c) {
          for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
              EXPECT_EQ(image.channel[c].plane.Row(y)[x],
                        decoded.channel[c].plane.Row(y)[x]);
            }
          }
        }
      }
    }
  }
}

TEST(ModularTest, RoundtripMultiPredictorHistograms) {
  auto* memory_manager = test::MemoryManager();
  for (auto mode : {ModularOptions::TreeMode::kDefault,
                    ModularOptions::TreeMode::kNoWP}) {
    for (bool speedups : {false, true}) {
      constexpr size_t width = 131, height = 67;
      JXL_TEST_ASSIGN_OR_DIE(Image image,
                            Image::Create(memory_manager, width, height, 16, 3));
      for (size_t c = 0; c < 3; ++c) {
        for (size_t y = 0; y < height; ++y) {
          for (size_t x = 0; x < width; ++x) {
            image.channel[c].plane.Row(y)[x] =
                (x * 1997 + y * 997 + (x / 7) * (y / 3) * (c + 1) * 31) % 65536;
          }
        }
      }
      ModularOptions options;
      options.predictor = Predictor::Variable;
      options.wp_tree_mode = mode;
      options.lossless_tree_speedups = speedups;
      options.nb_repeats = 1;
      options.max_properties = 2;
      BitWriter writer{memory_manager};
      ASSERT_TRUE(ModularGenericCompress(image, options, writer));
      writer.ZeroPadToByte();
      JXL_TEST_ASSIGN_OR_DIE(Image decoded,
                            Image::Create(memory_manager, width, height, 16, 3));
      Status status = true;
      {
        BitReader reader(writer.GetSpan());
        BitReaderScopedCloser closer(reader, status);
        ASSERT_TRUE(ModularGenericDecompress(&reader, decoded, nullptr, 0, &options));
      }
      ASSERT_TRUE(status);
      for (size_t c = 0; c < 3; ++c) {
        for (size_t y = 0; y < height; ++y) {
          for (size_t x = 0; x < width; ++x) {
            EXPECT_EQ(image.channel[c].plane.Row(y)[x], decoded.channel[c].plane.Row(y)[x]);
          }
        }
      }
    }
  }
}

TEST(ModularTest, RoundtripExtraProperties) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  constexpr size_t kSize = 250;
  JXL_TEST_ASSIGN_OR_DIE(Image image,
                         Image::Create(memory_manager, kSize, kSize,
                                       /*bitdepth=*/8, 3));
  ModularOptions options;
  options.max_properties = 4;
  options.predictor = Predictor::Zero;
  Rng rng(0);
  for (size_t y = 0; y < kSize; y++) {
    for (size_t x = 0; x < kSize; x++) {
      image.channel[0].plane.Row(y)[x] = image.channel[2].plane.Row(y)[x] =
          rng.UniformU(0, 9);
    }
  }
  ZeroFillImage(&image.channel[1].plane);
  BitWriter writer{memory_manager};
  ASSERT_TRUE(ModularGenericCompress(image, options, writer));
  writer.ZeroPadToByte();
  JXL_TEST_ASSIGN_OR_DIE(Image decoded,
                         Image::Create(memory_manager, kSize, kSize,
                                       /*bitdepth=*/8, image.channel.size()));
  for (size_t i = 0; i < image.channel.size(); i++) {
    const Channel& ch = image.channel[i];
    JXL_TEST_ASSIGN_OR_DIE(
        decoded.channel[i],
        Channel::Create(memory_manager, ch.w, ch.h, ch.hshift, ch.vshift));
  }
  Status status = true;
  {
    BitReader reader(writer.GetSpan());
    BitReaderScopedCloser closer(reader, status);
    ASSERT_TRUE(ModularGenericDecompress(&reader, decoded, /*header=*/nullptr,
                                         /*group_id=*/0, &options));
  }
  ASSERT_TRUE(status);
  ASSERT_EQ(image.channel.size(), decoded.channel.size());
  for (size_t c = 0; c < image.channel.size(); c++) {
    for (size_t y = 0; y < image.channel[c].plane.ysize(); y++) {
      for (size_t x = 0; x < image.channel[c].plane.xsize(); x++) {
        EXPECT_EQ(image.channel[c].plane.Row(y)[x],
                  decoded.channel[c].plane.Row(y)[x])
            << "c = " << c << ", x = " << x << ",  y = " << y;
      }
    }
  }
}

struct RoundtripLosslessConfig {
  int bitdepth;
  int responsive;
};
class ModularTestParam
    : public ::testing::TestWithParam<RoundtripLosslessConfig> {};

std::vector<RoundtripLosslessConfig> GenerateLosslessTests() {
  std::vector<RoundtripLosslessConfig> all;
  for (int responsive = 0; responsive <= 1; responsive++) {
    for (int bitdepth = 1; bitdepth < 32; bitdepth++) {
      if (responsive && bitdepth > 30) continue;
      all.push_back({bitdepth, responsive});
    }
  }
  return all;
}
std::string LosslessTestDescription(
    const testing::TestParamInfo<ModularTestParam::ParamType>& info) {
  std::stringstream name;
  name << info.param.bitdepth << "bit";
  if (info.param.responsive) name << "Squeeze";
  return name.str();
}

JXL_GTEST_INSTANTIATE_TEST_SUITE_P(RoundtripLossless, ModularTestParam,
                                   testing::ValuesIn(GenerateLosslessTests()),
                                   LosslessTestDescription);

TEST_P(ModularTestParam, RoundtripLossless) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  RoundtripLosslessConfig config = GetParam();
  int bitdepth = config.bitdepth;
  int responsive = config.responsive;

  Rng generator(123);
  const std::vector<uint8_t> orig =
      ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  extras::PackedPixelFile ppf1;
  ASSERT_TRUE(DecodeBytes(Bytes(orig), extras::ColorHints(), &ppf1));

  // vary the dimensions a bit, in case of bugs related to
  // even vs odd width or height.
  size_t xsize = 423 + bitdepth;
  size_t ysize = 467 + bitdepth;

  auto io = jxl::make_unique<jxl::CodecInOut>(memory_manager);
  ASSERT_TRUE(io->SetSize(xsize, ysize));
  io->metadata.m.color_encoding = jxl::ColorEncoding::SRGB(false);
  io->metadata.m.SetUintSamples(bitdepth);

  double factor = ((1lu << bitdepth) - 1lu);
  double ifactor = 1.0 / factor;
  JXL_TEST_ASSIGN_OR_DIE(Image3F noise_added,
                         Image3F::Create(memory_manager, xsize, ysize));

  for (size_t c = 0; c < 3; c++) {
    for (size_t y = 0; y < ysize; y++) {
      float* out = noise_added.PlaneRow(c, y);
      for (size_t x = 0; x < xsize; x++) {
        // make the least significant bits random
        float f = *ppf1.frames[0].color.const_pixels(y, x, c) +
                  generator.UniformF(0.0f, 1.f / 255.f);
        if (f > 1.f) f = 1.f;
        // quantize to the bitdepth we're testing
        unsigned int u = static_cast<unsigned int>(std::lround(f * factor));
        out[x] = u * ifactor;
      }
    }
  }
  ASSERT_TRUE(io->SetFromImage(std::move(noise_added),
                               jxl::ColorEncoding::SRGB(false)));

  CompressParams cparams;
  cparams.modular_mode = true;
  cparams.color_transform = jxl::ColorTransform::kNone;
  cparams.butteraugli_distance = 0.f;
  cparams.options.predictor = {Predictor::Zero};
  cparams.speed_tier = SpeedTier::kThunder;
  cparams.responsive = responsive;
  extras::JXLDecompressParams dparams;
  auto io2 = jxl::make_unique<jxl::CodecInOut>(memory_manager);
  size_t compressed_size;
  JXL_EXPECT_OK(
      Roundtrip(io.get(), cparams, dparams, io2.get(), _, &compressed_size));
  EXPECT_LE(compressed_size, bitdepth * xsize * ysize / 3.0 * 1.1);
  EXPECT_LE(0, ComputeDistance2(io->Main(), io2->Main(), *JxlGetDefaultCms()));
  size_t different = 0;
  for (size_t c = 0; c < 3; c++) {
    for (size_t y = 0; y < ysize; y++) {
      const float* in = io->Main().color()->PlaneRow(c, y);
      const float* out = io2->Main().color()->PlaneRow(c, y);
      for (size_t x = 0; x < xsize; x++) {
        uint32_t uin = std::lround(in[x] * factor);
        uint32_t uout = std::lround(out[x] * factor);
        // check that the integer values are identical
        if (uin != uout) different++;
      }
    }
  }
  EXPECT_EQ(different, 0u);
}

TEST(ModularTest, RoundtripLosslessCustomFloat) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  auto io = jxl::make_unique<jxl::CodecInOut>(memory_manager);
  size_t xsize = 100;
  size_t ysize = 300;
  ASSERT_TRUE(io->SetSize(xsize, ysize));
  io->metadata.m.bit_depth.bits_per_sample = 18;
  io->metadata.m.bit_depth.exponent_bits_per_sample = 6;
  io->metadata.m.bit_depth.floating_point_sample = true;
  io->metadata.m.modular_16_bit_buffer_sufficient = false;
  ColorEncoding color_encoding;
  color_encoding.Tf().SetTransferFunction(TransferFunction::kLinear);
  color_encoding.SetColorSpace(ColorSpace::kRGB);
  JXL_TEST_ASSIGN_OR_DIE(Image3F testimage,
                         Image3F::Create(memory_manager, xsize, ysize));
  float factor = 1.f / (1 << 14);
  for (size_t c = 0; c < 3; c++) {
    for (size_t y = 0; y < ysize; y++) {
      float* const JXL_RESTRICT row = testimage.PlaneRow(c, y);
      for (size_t x = 0; x < xsize; x++) {
        row[x] = factor * (x ^ y);
      }
    }
  }
  ASSERT_TRUE(io->SetFromImage(std::move(testimage), color_encoding));
  io->metadata.m.color_encoding = color_encoding;
  io->metadata.m.SetIntensityTarget(255);

  CompressParams cparams;
  cparams.modular_mode = true;
  cparams.color_transform = jxl::ColorTransform::kNone;
  cparams.butteraugli_distance = 0.f;
  cparams.options.predictor = {Predictor::Zero};
  cparams.speed_tier = SpeedTier::kThunder;
  cparams.decoding_speed_tier = 2;
  extras::JXLDecompressParams dparams;

  auto io2 = jxl::make_unique<jxl::CodecInOut>(memory_manager);
  size_t compressed_size;
  JXL_EXPECT_OK(
      Roundtrip(io.get(), cparams, dparams, io2.get(), _, &compressed_size));
  EXPECT_LE(compressed_size, 23000u);
  JXL_EXPECT_OK(SamePixels(*io->Main().color(), *io2->Main().color(), _));
}

void WriteHeaders(BitWriter* writer, size_t xsize, size_t ysize) {
  ASSERT_TRUE(writer->WithMaxBits(16, LayerType::Header, nullptr, [&] {
    writer->Write(8, 0xFF);
    writer->Write(8, kCodestreamMarker);
    return true;
  }));
  auto metadata = jxl::make_unique<CodecMetadata>();
  EXPECT_TRUE(metadata->size.Set(xsize, ysize));
  EXPECT_TRUE(
      WriteSizeHeader(metadata->size, writer, LayerType::Header, nullptr));
  metadata->m.color_encoding = ColorEncoding::LinearSRGB(/*is_gray=*/true);
  metadata->m.xyb_encoded = false;
  metadata->m.SetUintSamples(31);
  EXPECT_TRUE(
      WriteImageMetadata(metadata->m, writer, LayerType::Header, nullptr));
  metadata->transform_data.nonserialized_xyb_encoded = metadata->m.xyb_encoded;
  EXPECT_TRUE(Bundle::Write(metadata->transform_data, writer, LayerType::Header,
                            nullptr));
  writer->ZeroPadToByte();
  FrameHeader frame_header(metadata.get());
  frame_header.encoding = FrameEncoding::kModular;
  frame_header.loop_filter.gab = false;
  frame_header.loop_filter.epf_iters = 0;
  EXPECT_TRUE(WriteFrameHeader(frame_header, writer, nullptr));
}

// Tree with single node, zero predictor, offset is 1 and multiplier is 1,
// entropy code is prefix tree with alphabet size 256 and all bits lengths 8.
void WriteHistograms(BitWriter* writer) {
  writer->Write(1, 1);  // default DC quant
  writer->Write(1, 1);  // has_tree
  // tree histograms
  writer->Write(1, 0);         // LZ77 disabled
  writer->Write(3, 1);         // simple context map
  writer->Write(1, 1);         // prefix code
  writer->Write(7, 0x63);      // UnintConfig(3, 2, 1)
  writer->Write(12, 0xfef);    // alphabet_size = 256
  writer->Write(32, 0x10003);  // all bit lengths 8
  // tree tokens
  writer->Write(8, 0);   // tree leaf
  writer->Write(8, 0);   // zero predictor
  writer->Write(8, 64);  // offset = UnpackSigned(ReverseBits(64)) = 1
  writer->Write(16, 0);  // multiplier = 1
  // histograms
  writer->Write(1, 0);         // LZ77 disabled
  writer->Write(1, 1);         // prefix code
  writer->Write(7, 0x63);      // UnintConfig(3, 2, 1)
  writer->Write(12, 0xfef);    // alphabet_size = 256
  writer->Write(32, 0x10003);  // all bit lengths 8
}

TEST(ModularTest, PredictorIntegerOverflow) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  const size_t xsize = 1;
  const size_t ysize = 1;
  BitWriter writer{memory_manager};
  WriteHeaders(&writer, xsize, ysize);
  std::vector<std::unique_ptr<BitWriter>> group_codes;
  group_codes.emplace_back(jxl::make_unique<BitWriter>(memory_manager));
  {
    std::unique_ptr<BitWriter>& bw = group_codes[0];
    ASSERT_TRUE(bw->WithMaxBits(1 << 20, LayerType::Header, nullptr, [&] {
      WriteHistograms(bw.get());
      GroupHeader header;
      header.use_global_tree = true;
      EXPECT_TRUE(Bundle::Write(header, bw.get(), LayerType::Header, nullptr));
      // After UnpackSigned this becomes (1 << 31) - 1, the largest pixel_type,
      // and after adding the offset we get -(1 << 31).
      bw->Write(8, 119);
      bw->Write(28, 0xfffffff);
      bw->ZeroPadToByte();
      return true;
    }));
  }
  std::vector<size_t> sizes_0;
  for (const auto& bw : group_codes) {
    sizes_0.push_back(bw->BitsWritten() / kBitsPerByte);
  }
  EXPECT_TRUE(WriteTocPermutation({}, &writer, nullptr));
  EXPECT_TRUE(WriteTocSizes(sizes_0, &writer, nullptr));
  ASSERT_TRUE(writer.AppendByteAligned(group_codes));

  PaddedBytes compressed = std::move(writer).TakeBytes();
  extras::PackedPixelFile ppf;
  extras::JXLDecompressParams params;
  params.accepted_formats.push_back({1, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0});
  EXPECT_TRUE(DecodeImageJXL(compressed.data(), compressed.size(), params,
                             nullptr, &ppf));
  ASSERT_EQ(1u, ppf.frames.size());
  const auto& img = ppf.frames[0].color;
  const auto* pixels = reinterpret_cast<const float*>(img.pixels());
  EXPECT_EQ(-1.0f, pixels[0]);
}

TEST(ModularTest, UnsqueezeIntegerOverflow) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  // Image width is 9 so we can test both the SIMD and non-vector code paths.
  const size_t xsize = 9;
  const size_t ysize = 2;
  BitWriter writer{memory_manager};
  WriteHeaders(&writer, xsize, ysize);
  std::vector<std::unique_ptr<BitWriter>> group_codes;
  group_codes.emplace_back(jxl::make_unique<BitWriter>(memory_manager));
  {
    std::unique_ptr<BitWriter>& bw = group_codes[0];
    ASSERT_TRUE(bw->WithMaxBits(1 << 20, LayerType::Header, nullptr, [&] {
      WriteHistograms(bw.get());
      GroupHeader header;
      header.use_global_tree = true;
      header.transforms.emplace_back();
      header.transforms[0].id = TransformId::kSqueeze;
      SqueezeParams params;
      params.horizontal = false;
      params.in_place = true;
      params.begin_c = 0;
      params.num_c = 1;
      header.transforms[0].squeezes.emplace_back(params);
      EXPECT_TRUE(Bundle::Write(header, bw.get(), LayerType::Header, nullptr));
      for (size_t i = 0; i < xsize * ysize; ++i) {
        // After UnpackSigned and adding offset, this becomes (1 << 31) - 1,
        // both in the image and in the residual channels, and unsqueeze makes
        // them ~(3 << 30) and (1 << 30) (in pixel_type_w) and the first wraps
        // around to about -(1 << 30).
        bw->Write(8, 119);
        bw->Write(28, 0xffffffe);
      }
      bw->ZeroPadToByte();
      return true;
    }));
  }
  std::vector<size_t> sizes_1;
  for (const auto& bw : group_codes) {
    sizes_1.push_back(bw->BitsWritten() / kBitsPerByte);
  }
  EXPECT_TRUE(WriteTocPermutation({}, &writer, nullptr));
  EXPECT_TRUE(WriteTocSizes(sizes_1, &writer, nullptr));
  ASSERT_TRUE(writer.AppendByteAligned(group_codes));

  PaddedBytes compressed = std::move(writer).TakeBytes();
  extras::PackedPixelFile ppf;
  extras::JXLDecompressParams params;
  params.accepted_formats.push_back({1, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0});
  EXPECT_TRUE(DecodeImageJXL(compressed.data(), compressed.size(), params,
                             nullptr, &ppf));
  ASSERT_EQ(1u, ppf.frames.size());
  const auto& img = ppf.frames[0].color;
  const float* pixels = reinterpret_cast<const float*>(img.pixels());
  for (size_t x = 0; x < xsize; ++x) {
    EXPECT_NEAR(-0.5f, pixels[x], 1e-10);
    EXPECT_NEAR(0.5f, pixels[xsize + x], 1e-10);
  }
}

}  // namespace
}  // namespace jxl
