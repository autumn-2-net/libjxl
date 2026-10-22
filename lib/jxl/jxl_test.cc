// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/extras/dec/jxl.h"

#include <jxl/cms.h>
#include <jxl/color_encoding.h>
#include <jxl/encode.h>
#include <jxl/memory_manager.h>
#include <jxl/types.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

#include "hwy/targets.h"
#include "lib/extras/codec.h"
#include "lib/extras/codec_in_out.h"
#include "lib/extras/dec/decode.h"
#include "lib/extras/enc/encode.h"
#include "lib/extras/enc/jxl.h"
#include "lib/extras/packed_image.h"
#include "lib/jxl/alpha.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/butteraugli/butteraugli.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/common.h"  // JXL_HIGH_PRECISION
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/enc_frame.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/enc_lossless_probe.h"
#include "lib/jxl/enc_lossless_probe_pool.h"
#include "lib/jxl/enc_lossless_profile.h"
#include "lib/jxl/enc_lossless_scan.h"
#include "lib/jxl/enc_lossless_search.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/fake_parallel_runner_testonly.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/modular/encoding/enc_encoding.h"
#include "lib/jxl/modular/encoding/encoding.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/transform/transform.h"
#include "lib/jxl/test_image.h"
#include "lib/jxl/test_memory_manager.h"
#include "lib/jxl/test_utils.h"
#include "lib/jxl/testing.h"

namespace jxl {

struct AuxOut;

namespace {
using ::jxl::extras::JXLCompressParams;
using ::jxl::extras::JXLDecompressParams;
using ::jxl::extras::PackedPixelFile;
using ::jxl::test::ButteraugliDistance;
using ::jxl::test::ComputeDistance2;
using ::jxl::test::ReadTestData;
using ::jxl::test::Roundtrip;
using ::jxl::test::TestImage;
using ::jxl::test::ThreadPoolForTests;

#define JXL_TEST_NL 0  // Disabled in code

TEST(JxlTest, RoundtripSinglePixel) {
  TestImage t;
  ASSERT_TRUE(t.SetDimensions(1, 1));
  JXL_TEST_ASSIGN_OR_DIE(auto frame, t.AddFrame());
  frame.ZeroFill();
  PackedPixelFile ppf_out;
  EXPECT_NEAR(Roundtrip(t.ppf(), {}, {}, nullptr, &ppf_out), 31, 10);
}

TEST(JxlTest, RoundtripSinglePixelWithAlpha) {
  TestImage t;
  ASSERT_TRUE(t.SetDimensions(1, 1));
  ASSERT_TRUE(t.SetChannels(4));
  JXL_TEST_ASSIGN_OR_DIE(auto frame, t.AddFrame());
  frame.ZeroFill();
  PackedPixelFile ppf_out;
  EXPECT_NEAR(Roundtrip(t.ppf(), {}, {}, nullptr, &ppf_out), 34, 10);
}

// Changing serialized signature causes Decode to fail.
TEST(JxlTest, RoundtripMarker) {
  if (JXL_CRASH_ON_ERROR) {
    GTEST_SKIP() << "Skipping due to JXL_CRASH_ON_ERROR";
  }
  TestImage t;
  ASSERT_TRUE(t.SetDimensions(1, 1));
  JXL_TEST_ASSIGN_OR_DIE(auto frame, t.AddFrame());
  frame.ZeroFill();
  for (size_t i = 0; i < 2; ++i) {
    std::vector<uint8_t> compressed;
    EXPECT_TRUE(extras::EncodeImageJXL({}, t.ppf(), /*jpeg_bytes=*/nullptr,
                                       &compressed));
    compressed[i] ^= 0xFF;
    PackedPixelFile ppf_out;
    EXPECT_FALSE(extras::DecodeImageJXL(compressed.data(), compressed.size(),
                                        {}, /* decoded_bytes */ nullptr,
                                        &ppf_out));
  }
}

TEST(JxlTest, RoundtripTinyFast) {
  ThreadPool* pool = nullptr;
  const std::vector<uint8_t> orig =
      ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));
  t.ClearMetadata();
  ASSERT_TRUE(t.SetDimensions(32, 32));

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 7);
  cparams.distance = 4.0f;
  extras::JXLDecompressParams dparams;

  PackedPixelFile ppf_out;
  EXPECT_NEAR(Roundtrip(t.ppf(), cparams, dparams, pool, &ppf_out), 181, 15);
}

TEST(JxlTest, RoundtripSmallD1) {
  ThreadPool* pool = nullptr;
  const std::vector<uint8_t> orig =
      ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));
  t.ClearMetadata();
  size_t xsize = t.ppf().info.xsize / 8;
  size_t ysize = t.ppf().info.ysize / 8;
  ASSERT_TRUE(t.SetDimensions(xsize, ysize));

  {
    PackedPixelFile ppf_out;
    EXPECT_NEAR(Roundtrip(t.ppf(), {}, {}, pool, &ppf_out), 916, 40);
    EXPECT_SLIGHTLY_BELOW(ButteraugliDistance(t.ppf(), ppf_out), 0.92);
  }

  // With a lower intensity target than the default, the bitrate should be
  // smaller.
  t.ppf().info.intensity_target = 100.0f;

  {
    PackedPixelFile ppf_out;
    EXPECT_NEAR(Roundtrip(t.ppf(), {}, {}, pool, &ppf_out), 724, 20);
    EXPECT_SLIGHTLY_BELOW(ButteraugliDistance(t.ppf(), ppf_out), 1.3);
    EXPECT_EQ(ppf_out.info.intensity_target, t.ppf().info.intensity_target);
  }
}
TEST(JxlTest, RoundtripResample2) {
  ThreadPool* pool = nullptr;
  const std::vector<uint8_t> orig =
      ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));
  t.ClearMetadata();

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_RESAMPLING, 2);
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 3);  // kFalcon
  extras::JXLDecompressParams dparams;

  PackedPixelFile ppf_out;
  EXPECT_NEAR(Roundtrip(t.ppf(), cparams, dparams, pool, &ppf_out), 17300, 500);
  EXPECT_SLIGHTLY_BELOW(ComputeDistance2(t.ppf(), ppf_out), 90);
}

TEST(JxlTest, RoundtripResample2Slow) {
  ThreadPool* pool = nullptr;
  const std::vector<uint8_t> orig =
      ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));
  t.ClearMetadata();

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_RESAMPLING, 2);
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 9);  // kTortoise
  cparams.distance = 10.0;
  extras::JXLDecompressParams dparams;

  PackedPixelFile ppf_out;
  EXPECT_NEAR(Roundtrip(t.ppf(), cparams, dparams, pool, &ppf_out), 3988, 300);
  EXPECT_SLIGHTLY_BELOW(ComputeDistance2(t.ppf(), ppf_out), 270);
}

TEST(JxlTest, RoundtripResample2MT) {
  ThreadPoolForTests pool(4);
  const std::vector<uint8_t> orig = ReadTestData("jxl/flower/flower.png");
  // image has to be large enough to have multiple groups after downsampling
  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));
  t.ClearMetadata();

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_RESAMPLING, 2);
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 3);  // kFalcon
  extras::JXLDecompressParams dparams;

  PackedPixelFile ppf_out;
  EXPECT_NEAR(Roundtrip(t.ppf(), cparams, dparams, pool.get(), &ppf_out),
              206917, 2000);
  EXPECT_SLIGHTLY_BELOW(ComputeDistance2(t.ppf(), ppf_out), 340);
}

// Roundtrip the image using a parallel runner that executes single-threaded but
// in random order.
TEST(JxlTest, RoundtripOutOfOrderProcessing) {
  FakeParallelRunner fake_pool(/*order_seed=*/123, /*num_threads=*/8);
  ThreadPool pool(&JxlFakeParallelRunner, &fake_pool);
  const std::vector<uint8_t> orig = ReadTestData("jxl/flower/flower.png");
  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));
  t.ClearMetadata();
  // Image size is selected so that the block border needed is larger than the
  // amount of pixels available on the next block.
  ASSERT_TRUE(t.SetDimensions(513, 515));

  JXLCompressParams cparams;
  // Force epf so we end up needing a lot of border.
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EPF, 3);
  extras::JXLDecompressParams dparams;

  PackedPixelFile ppf_out;
  EXPECT_NEAR(Roundtrip(t.ppf(), cparams, dparams, &pool, &ppf_out), 23666,
              400);
  EXPECT_LE(ButteraugliDistance(t.ppf(), ppf_out), 1.35);
}

TEST(JxlTest, RoundtripOutOfOrderProcessingBorder) {
  FakeParallelRunner fake_pool(/*order_seed=*/47, /*num_threads=*/8);
  ThreadPool pool(&JxlFakeParallelRunner, &fake_pool);
  const std::vector<uint8_t> orig = ReadTestData("jxl/flower/flower.png");
  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));
  t.ClearMetadata();
  // Image size is selected so that the block border needed is larger than the
  // amount of pixels available on the next block.
  ASSERT_TRUE(t.SetDimensions(513, 515));

  JXLCompressParams cparams;
  // Force epf so we end up needing a lot of border.
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EPF, 3);
  cparams.AddOption(JXL_ENC_FRAME_SETTING_RESAMPLING, 2);
  extras::JXLDecompressParams dparams;

  PackedPixelFile ppf_out;
  EXPECT_NEAR(Roundtrip(t.ppf(), cparams, dparams, &pool, &ppf_out), 9479, 200);
  EXPECT_LE(ButteraugliDistance(t.ppf(), ppf_out), 2.9);
}

TEST(JxlTest, RoundtripResample4) {
  ThreadPool* pool = nullptr;
  const std::vector<uint8_t> orig =
      ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));
  t.ClearMetadata();

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_RESAMPLING, 4);
  extras::JXLDecompressParams dparams;

  PackedPixelFile ppf_out;
  EXPECT_NEAR(Roundtrip(t.ppf(), cparams, dparams, pool, &ppf_out), 5888, 100);
  EXPECT_SLIGHTLY_BELOW(ButteraugliDistance(t.ppf(), ppf_out), 22);
}

TEST(JxlTest, RoundtripResample8) {
  ThreadPool* pool = nullptr;
  const std::vector<uint8_t> orig =
      ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));
  t.ClearMetadata();

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_RESAMPLING, 8);
  extras::JXLDecompressParams dparams;

  PackedPixelFile ppf_out;
  EXPECT_NEAR(Roundtrip(t.ppf(), cparams, dparams, pool, &ppf_out), 2036, 50);
  EXPECT_SLIGHTLY_BELOW(ButteraugliDistance(t.ppf(), ppf_out), 50);
}

TEST(JxlTest, RoundtripUnalignedD2) {
  ThreadPool* pool = nullptr;
  const std::vector<uint8_t> orig =
      ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));
  t.ClearMetadata();
  size_t xsize = t.ppf().info.xsize / 12;
  size_t ysize = t.ppf().info.ysize / 7;
  ASSERT_TRUE(t.SetDimensions(xsize, ysize));

  JXLCompressParams cparams;
  cparams.distance = 2.0;
  extras::JXLDecompressParams dparams;

  PackedPixelFile ppf_out;
  EXPECT_NEAR(Roundtrip(t.ppf(), cparams, dparams, pool, &ppf_out), 540, 37);
  EXPECT_SLIGHTLY_BELOW(ButteraugliDistance(t.ppf(), ppf_out), 1.72);
}

TEST(JxlTest, RoundtripMultiGroup) {
  const std::vector<uint8_t> orig = ReadTestData("jxl/flower/flower.png");
  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));
  t.ClearMetadata();
  ASSERT_TRUE(t.SetDimensions(600, 1024));

  auto test = [&](jxl::SpeedTier speed_tier, float target_distance,
                  size_t expected_size, float expected_distance) {
    ThreadPoolForTests pool(4);
    JXLCompressParams cparams;
    int64_t effort = 10 - static_cast<int>(speed_tier);
    cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, effort);
    cparams.distance = target_distance;
    extras::JXLDecompressParams dparams;

    PackedPixelFile ppf_out;
    EXPECT_NEAR(Roundtrip(t.ppf(), cparams, dparams, pool.get(), &ppf_out),
                expected_size, 814);
    EXPECT_SLIGHTLY_BELOW(ComputeDistance2(t.ppf(), ppf_out),
                          expected_distance);
  };

  auto run_kitten = std::async(std::launch::async, test, SpeedTier::kKitten,
                               1.0f, 58737u, 9.1);
  auto run_wombat = std::async(std::launch::async, test, SpeedTier::kWombat,
                               2.0f, 35561u, 16.5);
}

TEST(JxlTest, RoundtripRGBToGrayscale) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  ThreadPoolForTests pool(4);
  const std::vector<uint8_t> orig = ReadTestData("jxl/flower/flower.png");
  auto io = jxl::make_unique<jxl::CodecInOut>(memory_manager);
  ASSERT_TRUE(SetFromBytes(Bytes(orig), io.get(), pool.get()));
  ASSERT_TRUE(io->ShrinkTo(600, 1024));

  CompressParams cparams;
  cparams.butteraugli_distance = 1.0f;
  cparams.speed_tier = SpeedTier::kFalcon;

  JXLDecompressParams dparams;
  dparams.color_space = "Gra_D65_Rel_SRG";

  auto io2 = jxl::make_unique<jxl::CodecInOut>(memory_manager);
  EXPECT_FALSE(io->Main().IsGray());
  size_t compressed_size;
  JXL_EXPECT_OK(Roundtrip(io.get(), cparams, dparams, io2.get(), _,
                          &compressed_size, pool.get()));
  EXPECT_LE(compressed_size, 65000u);
  EXPECT_TRUE(io2->Main().IsGray());

  // Convert original to grayscale here, because TransformTo refuses to
  // convert between grayscale and RGB.
  ColorEncoding srgb_lin = ColorEncoding::LinearSRGB(/*is_gray=*/false);
  ASSERT_TRUE(io->frames[0].TransformTo(srgb_lin, *JxlGetDefaultCms()));
  Image3F* color = io->Main().color();
  for (size_t y = 0; y < color->ysize(); ++y) {
    float* row_r = color->PlaneRow(0, y);
    float* row_g = color->PlaneRow(1, y);
    float* row_b = color->PlaneRow(2, y);
    for (size_t x = 0; x < color->xsize(); ++x) {
      float luma = 0.2126 * row_r[x] + 0.7152 * row_g[x] + 0.0722 * row_b[x];
      row_r[x] = row_g[x] = row_b[x] = luma;
    }
  }
  ColorEncoding srgb_gamma = ColorEncoding::SRGB(/*is_gray=*/false);
  ASSERT_TRUE(io->frames[0].TransformTo(srgb_gamma, *JxlGetDefaultCms()));
  io->metadata.m.color_encoding = io2->Main().c_current();
  io->Main().OverrideProfile(io2->Main().c_current());
  EXPECT_SLIGHTLY_BELOW(
      ButteraugliDistance(io->frames, io2->frames, ButteraugliParams(),
                          *JxlGetDefaultCms(),
                          /*distmap=*/nullptr, pool.get()),
      1.4);
}

TEST(JxlTest, RoundtripLargeFast) {
  ThreadPoolForTests pool(8);
  const std::vector<uint8_t> orig = ReadTestData("jxl/flower/flower.png");
  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));
  t.ClearMetadata();

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 7);  // kSquirrel
  extras::JXLDecompressParams dparams;

  PackedPixelFile ppf_out;
  EXPECT_NEAR(Roundtrip(t.ppf(), cparams, dparams, pool.get(), &ppf_out),
              488000, 12000);
  EXPECT_SLIGHTLY_BELOW(ComputeDistance2(t.ppf(), ppf_out), 78);
}

TEST(JxlTest, RoundtripLargeEmptyModular) {
  ThreadPoolForTests pool(8);
  TestImage t;
  // On 32-bit machines memory could be limited to 2GiB. Current use factor
  // is about 20x, that means hat 6MPx image is about the max available size.
  constexpr bool kSkinny = (sizeof(size_t) == 4);
  constexpr size_t kDim = kSkinny ? 2400 : 4096;
  const size_t kLim = std::min<size_t>(kDim, 1024);
  ASSERT_TRUE(t.SetDimensions(kDim, kDim));
  t.SetDataType(JXL_TYPE_UINT8);
  ASSERT_TRUE(t.SetChannels(4));
  JXL_TEST_ASSIGN_OR_DIE(auto frame, t.AddFrame());
  frame.ZeroFill();
  for (size_t c = 0; c < 4; ++c) {
    for (size_t y = 0; y < kLim; y += (c + 1)) {
      for (size_t x = 0; x < kLim; x += ((y % 4) + 3)) {
        ASSERT_TRUE(frame.SetValue(y, x, c, 0.88));
      }
    }
  }

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 1);
  cparams.AddOption(JXL_ENC_FRAME_SETTING_MODULAR, 1);
  cparams.AddOption(JXL_ENC_FRAME_SETTING_RESPONSIVE, 1);
  cparams.AddOption(JXL_ENC_FRAME_SETTING_DECODING_SPEED, 2);
  extras::JXLDecompressParams dparams;

  PackedPixelFile ppf_out;
  constexpr size_t kTargetSize = kSkinny ? 539100 : 620200;
  EXPECT_NEAR(Roundtrip(t.ppf(), cparams, dparams, pool.get(), &ppf_out),
              kTargetSize, 10000);
  EXPECT_SLIGHTLY_BELOW(ButteraugliDistance(t.ppf(), ppf_out), 0.21);
}

TEST(JxlTest, RoundtripOutputColorSpace) {
  ThreadPoolForTests pool(8);
  const std::vector<uint8_t> orig = ReadTestData("jxl/flower/flower.png");
  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));
  t.ClearMetadata();

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 7);  // kSquirrel

  JXLDecompressParams dparams;
  dparams.color_space = "RGB_D65_DCI_Rel_709";
  PackedPixelFile ppf_out;
  EXPECT_NEAR(Roundtrip(t.ppf(), cparams, dparams, pool.get(), &ppf_out),
              487000, 12000);
  EXPECT_SLIGHTLY_BELOW(ComputeDistance2(t.ppf(), ppf_out), 78);
}

TEST(JxlTest, RoundtripDotsForceEpf) {
  ThreadPoolForTests pool(8);
  const std::vector<uint8_t> orig =
      ReadTestData("external/wesaturate/500px/cvo9xd_keong_macan_srgb8.png");
  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));
  t.ClearMetadata();

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 7);  // kSquirrel
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EPF, 2);
  cparams.AddOption(JXL_ENC_FRAME_SETTING_DOTS, 1);
  extras::JXLDecompressParams dparams;

  PackedPixelFile ppf_out;
  EXPECT_NEAR(Roundtrip(t.ppf(), cparams, dparams, pool.get(), &ppf_out), 41777,
              700);
  EXPECT_SLIGHTLY_BELOW(ComputeDistance2(t.ppf(), ppf_out), 18);
}

// Checks for differing size/distance in two consecutive runs of distance 2,
// which involves additional processing including adaptive reconstruction.
// Failing this may be a sign of race conditions or invalid memory accesses.
TEST(JxlTest, RoundtripD2Consistent) {
  ThreadPoolForTests pool(8);
  const std::vector<uint8_t> orig = ReadTestData("jxl/flower/flower.png");
  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));
  t.ClearMetadata();

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 7);  // kSquirrel
  cparams.distance = 2.0;
  extras::JXLDecompressParams dparams;

  // Try each xsize mod kBlockDim to verify right border handling.
  for (size_t xsize = 48; xsize > 40; --xsize) {
    ASSERT_TRUE(t.SetDimensions(xsize, 15));

    PackedPixelFile ppf2;
    const size_t size2 =
        Roundtrip(t.ppf(), cparams, dparams, pool.get(), &ppf2);

    PackedPixelFile ppf3;
    const size_t size3 =
        Roundtrip(t.ppf(), cparams, dparams, pool.get(), &ppf3);

    // Exact same compressed size.
    EXPECT_EQ(size2, size3);

    // Exact same distance.
    const float dist2 = ComputeDistance2(t.ppf(), ppf2);
    const float dist3 = ComputeDistance2(t.ppf(), ppf3);
    EXPECT_EQ(dist2, dist3);
  }
}

// Same as above, but for full image, testing multiple groups.
TEST(JxlTest, RoundtripLargeConsistent) {
  const std::vector<uint8_t> orig = ReadTestData("jxl/flower/flower.png");
  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));
  t.ClearMetadata();

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 7);  // kSquirrel
  cparams.distance = 2.0;
  extras::JXLDecompressParams dparams;

  auto roundtrip_and_compare = [&]() {
    ThreadPoolForTests pool(8);
    PackedPixelFile ppf2;
    size_t size = Roundtrip(t.ppf(), cparams, dparams, pool.get(), &ppf2);
    double dist = ComputeDistance2(t.ppf(), ppf2);
    return std::tuple<size_t, double>(size, dist);
  };

  // Try each xsize mod kBlockDim to verify right border handling.
  auto future2 = std::async(std::launch::async, roundtrip_and_compare);
  auto future3 = std::async(std::launch::async, roundtrip_and_compare);

  const auto result2 = future2.get();
  const auto result3 = future3.get();

  // Exact same compressed size.
  EXPECT_EQ(std::get<0>(result2), std::get<0>(result3));

  // Exact same distance.
  EXPECT_EQ(std::get<1>(result2), std::get<1>(result3));
}

TEST(JxlTest, RoundtripSmallNL) {
  ThreadPool* pool = nullptr;
  const std::vector<uint8_t> orig =
      ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));
  t.ClearMetadata();
  size_t xsize = t.ppf().info.xsize / 8;
  size_t ysize = t.ppf().info.ysize / 8;
  ASSERT_TRUE(t.SetDimensions(xsize, ysize));

  PackedPixelFile ppf_out;
  EXPECT_NEAR(Roundtrip(t.ppf(), {}, {}, pool, &ppf_out), 916, 45);
  EXPECT_SLIGHTLY_BELOW(ButteraugliDistance(t.ppf(), ppf_out), 0.92);
}

TEST(JxlTest, RoundtripNoGaborishNoAR) {
  ThreadPool* pool = nullptr;
  const std::vector<uint8_t> orig =
      ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));
  t.ClearMetadata();

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EPF, 0);
  cparams.AddOption(JXL_ENC_FRAME_SETTING_GABORISH, 0);
  extras::JXLDecompressParams dparams;

  PackedPixelFile ppf_out;
  EXPECT_NEAR(Roundtrip(t.ppf(), cparams, dparams, pool, &ppf_out), 44555, 400);
  EXPECT_SLIGHTLY_BELOW(ButteraugliDistance(t.ppf(), ppf_out), 1.55);
}

TEST(JxlTest, RoundtripSmallNoGaborish) {
  ThreadPool* pool = nullptr;
  const std::vector<uint8_t> orig =
      ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));
  t.ClearMetadata();
  size_t xsize = t.ppf().info.xsize / 8;
  size_t ysize = t.ppf().info.ysize / 8;
  ASSERT_TRUE(t.SetDimensions(xsize, ysize));

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_GABORISH, 0);
  extras::JXLDecompressParams dparams;

  PackedPixelFile ppf_out;
  EXPECT_NEAR(Roundtrip(t.ppf(), cparams, dparams, pool, &ppf_out), 1066, 20);
  EXPECT_SLIGHTLY_BELOW(ButteraugliDistance(t.ppf(), ppf_out), 1.0);
}

TEST(JxlTest, RoundtripSmallPatchesAlpha) {
  ThreadPool* pool = nullptr;
  TestImage t;
  ASSERT_TRUE(t.SetDimensions(256, 256));
  ASSERT_TRUE(t.SetChannels(4));
  ASSERT_TRUE(t.SetColorEncoding("RGB_D65_SRG_Rel_Lin"));
  JXL_TEST_ASSIGN_OR_DIE(auto frame, t.AddFrame());
  frame.ZeroFill();
  // This pattern should be picked up by the patch detection heuristics.
  for (size_t y = 0; y < t.ppf().info.ysize; ++y) {
    for (size_t x = 0; x < t.ppf().info.xsize; ++x) {
      if (x % 4 == 0 && (y / 32) % 4 == 0) {
        ASSERT_TRUE(frame.SetValue(y, x, 1, 127.0f / 255.0f));
      }
      ASSERT_TRUE(frame.SetValue(y, x, 3, 1.0f));
    }
  }

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 7);  // kSquirrel
  cparams.distance = 0.1f;
  extras::JXLDecompressParams dparams;

  PackedPixelFile ppf_out;
  EXPECT_NEAR(Roundtrip(t.ppf(), cparams, dparams, pool, &ppf_out), 462, 100);
  EXPECT_SLIGHTLY_BELOW(ButteraugliDistance(t.ppf(), ppf_out), 0.016f);
}

TEST(JxlTest, RoundtripSmallPatches) {
  ThreadPool* pool = nullptr;
  TestImage t;
  ASSERT_TRUE(t.SetDimensions(256, 256));
  ASSERT_TRUE(t.SetColorEncoding("RGB_D65_SRG_Rel_Lin"));
  JXL_TEST_ASSIGN_OR_DIE(auto frame, t.AddFrame());
  frame.ZeroFill();
  // This pattern should be picked up by the patch detection heuristics.
  for (size_t y = 0; y < t.ppf().info.ysize; ++y) {
    for (size_t x = 0; x < t.ppf().info.xsize; ++x) {
      if (x % 4 == 0 && (y / 32) % 4 == 0) {
        ASSERT_TRUE(frame.SetValue(y, x, 1, 127.0f / 255.0f));
      }
    }
  }

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 7);  // kSquirrel
  cparams.distance = 0.1f;
  extras::JXLDecompressParams dparams;

  PackedPixelFile ppf_out;
  EXPECT_NEAR(Roundtrip(t.ppf(), cparams, dparams, pool, &ppf_out), 363, 100);
  EXPECT_SLIGHTLY_BELOW(ButteraugliDistance(t.ppf(), ppf_out), 0.015f);
}

// TODO(szabadka) Add encoder and decoder API functions that accept frame
// buffers in arbitrary unsigned and floating point formats, and then roundtrip
// test the lossless codepath to make sure the exact binary representations
// are preserved.
#if JXL_FALSE
TEST(JxlTest, RoundtripImageBundleOriginalBits) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  // Image does not matter, only io.metadata.m and io2.metadata.m are tested.
  JXL_TEST_ASSIGN_OR_DIE(Image3F image, Image3F::Create(memory_manager, 1, 1));
  ZeroFillImage(&image);
  auto io = jxl::make_unique<CodecInOut>(memory_manager);
  io->metadata.m.color_encoding = ColorEncoding::LinearSRGB();
  JXL_EXPECT_OK(
      io->SetFromImage(std::move(image), ColorEncoding::LinearSRGB()));

  CompressParams cparams;
  extras::JXLDecompressParams dparams;

  // Test unsigned integers from 1 to 32 bits
  for (uint32_t bit_depth = 1; bit_depth <= 32; bit_depth++) {
    if (bit_depth == 32) {
      // TODO(lode): allow testing 32, however the code below ends up in
      // enc_modular which does not support 32. We only want to test the header
      // encoding though, so try without modular.
      break;
    }

    io->metadata.m.SetUintSamples(bit_depth);
    auto io2 = jxl::make_unique<CodecInOut>(memory_manager);
    JXL_EXPECT_OK(Roundtrip(io.get(), cparams, dparams, io2.get(), _));

    EXPECT_EQ(bit_depth, io2->metadata.m.bit_depth.bits_per_sample);
    EXPECT_FALSE(io2->metadata.m.bit_depth.floating_point_sample);
    EXPECT_EQ(0u, io2->metadata.m.bit_depth.exponent_bits_per_sample);
    EXPECT_EQ(0u, io2->metadata.m.GetAlphaBits());
  }

  // Test various existing and non-existing floating point formats
  for (uint32_t bit_depth = 8; bit_depth <= 32; bit_depth++) {
    if (bit_depth != 32) {
      // TODO(user): test other float types once they work
      break;
    }

    uint32_t exponent_bit_depth;
    if (bit_depth < 10) {
      exponent_bit_depth = 2;
    } else if (bit_depth < 12) {
      exponent_bit_depth = 3;
    } else if (bit_depth < 16) {
      exponent_bit_depth = 4;
    } else if (bit_depth < 20) {
      exponent_bit_depth = 5;
    } else if (bit_depth < 24) {
      exponent_bit_depth = 6;
    } else if (bit_depth < 28) {
      exponent_bit_depth = 7;
    } else {
      exponent_bit_depth = 8;
    }

    io->metadata.m.bit_depth.bits_per_sample = bit_depth;
    io->metadata.m.bit_depth.floating_point_sample = true;
    io->metadata.m.bit_depth.exponent_bits_per_sample = exponent_bit_depth;

    auto io2 = jxl::make_unique<CodecInOut>(memory_manager);
    JXL_EXPECT_OK(Roundtrip(io.get(), cparams, dparams, io2.get()));

    EXPECT_EQ(bit_depth, io2->metadata.m.bit_depth.bits_per_sample);
    EXPECT_TRUE(io2->metadata.m.bit_depth.floating_point_sample);
    EXPECT_EQ(exponent_bit_depth,
              io2->metadata.m.bit_depth.exponent_bits_per_sample);
    EXPECT_EQ(0u, io2->metadata.m.GetAlphaBits());
  }
}
#endif

TEST(JxlTest, RoundtripGrayscale) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  const std::vector<uint8_t> orig = ReadTestData(
      "external/wesaturate/500px/cvo9xd_keong_macan_grayscale.png");
  auto io = jxl::make_unique<CodecInOut>(memory_manager);
  ASSERT_TRUE(SetFromBytes(Bytes(orig), io.get()));
  ASSERT_NE(io->xsize(), 0u);
  ASSERT_TRUE(io->ShrinkTo(128, 128));
  EXPECT_TRUE(io->Main().IsGray());
  EXPECT_EQ(8u, io->metadata.m.bit_depth.bits_per_sample);
  EXPECT_FALSE(io->metadata.m.bit_depth.floating_point_sample);
  EXPECT_EQ(0u, io->metadata.m.bit_depth.exponent_bits_per_sample);
  EXPECT_TRUE(io->metadata.m.color_encoding.Tf().IsSRGB());

  {
    CompressParams cparams;
    cparams.butteraugli_distance = 1.0;

    std::vector<uint8_t> compressed;
    EXPECT_TRUE(test::EncodeFile(cparams, io.get(), &compressed));
    auto io2 = jxl::make_unique<CodecInOut>(memory_manager);
    EXPECT_TRUE(test::DecodeFile({}, Bytes(compressed), io2.get()));
    EXPECT_TRUE(io2->Main().IsGray());

    EXPECT_LE(compressed.size(), 7000u);
    EXPECT_SLIGHTLY_BELOW(
        ButteraugliDistance(io->frames, io2->frames, ButteraugliParams(),
                            *JxlGetDefaultCms(),
                            /*distmap=*/nullptr),
        1.6);
  }

  // Test with larger butteraugli distance and other settings enabled so
  // different jxl codepaths trigger.
  {
    CompressParams cparams;
    cparams.butteraugli_distance = 8.0;

    std::vector<uint8_t> compressed;
    EXPECT_TRUE(test::EncodeFile(cparams, io.get(), &compressed));
    auto io2 = jxl::make_unique<CodecInOut>(memory_manager);
    EXPECT_TRUE(test::DecodeFile({}, Bytes(compressed), io2.get()));
    EXPECT_TRUE(io2->Main().IsGray());

    EXPECT_LE(compressed.size(), 1300u);
    EXPECT_SLIGHTLY_BELOW(
        ButteraugliDistance(io->frames, io2->frames, ButteraugliParams(),
                            *JxlGetDefaultCms(),
                            /*distmap=*/nullptr),
        6.7);
  }

  {
    CompressParams cparams;
    cparams.butteraugli_distance = 1.0;

    std::vector<uint8_t> compressed;
    EXPECT_TRUE(test::EncodeFile(cparams, io.get(), &compressed));

    auto io2 = jxl::make_unique<CodecInOut>(memory_manager);
    JXLDecompressParams dparams;
    dparams.color_space = "RGB_D65_SRG_Rel_SRG";
    EXPECT_TRUE(test::DecodeFile(dparams, Bytes(compressed), io2.get()));
    EXPECT_FALSE(io2->Main().IsGray());

    EXPECT_LE(compressed.size(), 7000u);
    EXPECT_SLIGHTLY_BELOW(
        ButteraugliDistance(io->frames, io2->frames, ButteraugliParams(),
                            *JxlGetDefaultCms(),
                            /*distmap=*/nullptr),
        1.6);
  }
}

TEST(JxlTest, RoundtripAlpha) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  const std::vector<uint8_t> orig =
      ReadTestData("external/wesaturate/500px/tmshre_riaphotographs_alpha.png");
  auto io = jxl::make_unique<CodecInOut>(memory_manager);
  ASSERT_TRUE(SetFromBytes(Bytes(orig), io.get()));

  ASSERT_NE(io->xsize(), 0u);
  ASSERT_TRUE(io->metadata.m.HasAlpha());
  ASSERT_TRUE(io->Main().HasAlpha());
  ASSERT_TRUE(io->ShrinkTo(300, 300));

  CompressParams cparams;
  cparams.butteraugli_distance = 1.0;

  EXPECT_EQ(8u, io->metadata.m.bit_depth.bits_per_sample);
  EXPECT_FALSE(io->metadata.m.bit_depth.floating_point_sample);
  EXPECT_EQ(0u, io->metadata.m.bit_depth.exponent_bits_per_sample);
  EXPECT_TRUE(io->metadata.m.color_encoding.Tf().IsSRGB());
  std::vector<uint8_t> compressed;
  EXPECT_TRUE(test::EncodeFile(cparams, io.get(), &compressed));

  EXPECT_LE(compressed.size(), 20000u);

  for (bool use_image_callback : {false, true}) {
    for (bool unpremul_alpha : {false, true}) {
      auto io2 = jxl::make_unique<CodecInOut>(memory_manager);
      JXLDecompressParams dparams;
      dparams.use_image_callback = use_image_callback;
      dparams.unpremultiply_alpha = unpremul_alpha;
      EXPECT_TRUE(test::DecodeFile(dparams, Bytes(compressed), io2.get()));
      EXPECT_SLIGHTLY_BELOW(
          ButteraugliDistance(io->frames, io2->frames, ButteraugliParams(),
                              *JxlGetDefaultCms(),
                              /*distmap=*/nullptr),
          1.05);
    }
  }
}

namespace {
// Performs "PremultiplyAlpha" for each ImageBundle (preview/frames).
Status PremultiplyAlpha(CodecInOut& io) {
  const auto doPremultiplyAlpha = [](ImageBundle& bundle) -> Status {
    if (!bundle.HasAlpha()) return true;
    if (!bundle.HasColor()) return true;
    auto* color = bundle.color();
    const auto* alpha = bundle.alpha();
    JXL_ENSURE(color->ysize() == alpha->ysize());
    JXL_ENSURE(color->xsize() == alpha->xsize());
    for (size_t y = 0; y < color->ysize(); y++) {
      ::jxl::PremultiplyAlpha(color->PlaneRow(0, y), color->PlaneRow(1, y),
                              color->PlaneRow(2, y), alpha->Row(y),
                              color->xsize());
    }
    return true;
  };
  ExtraChannelInfo* eci = io.metadata.m.Find(ExtraChannel::kAlpha);
  JXL_ENSURE(eci != nullptr && !eci->alpha_associated);
  if (io.metadata.m.have_preview) {
    JXL_RETURN_IF_ERROR(doPremultiplyAlpha(io.preview_frame));
  }
  for (ImageBundle& ib : io.frames) {
    JXL_RETURN_IF_ERROR(doPremultiplyAlpha(ib));
  }
  eci->alpha_associated = true;
  return true;
}

Status UnpremultiplyAlpha(CodecInOut& io) {
  const auto doUnpremultiplyAlpha = [](ImageBundle& bundle) -> Status {
    if (!bundle.HasAlpha()) return true;
    if (!bundle.HasColor()) return true;
    auto* color = bundle.color();
    const auto* alpha = bundle.alpha();
    JXL_ENSURE(color->ysize() == alpha->ysize());
    JXL_ENSURE(color->xsize() == alpha->xsize());
    for (size_t y = 0; y < color->ysize(); y++) {
      ::jxl::UnpremultiplyAlpha(color->PlaneRow(0, y), color->PlaneRow(1, y),
                                color->PlaneRow(2, y), alpha->Row(y),
                                color->xsize());
    }
    return true;
  };
  ExtraChannelInfo* eci = io.metadata.m.Find(ExtraChannel::kAlpha);
  JXL_ENSURE(eci != nullptr && eci->alpha_associated);
  if (io.metadata.m.have_preview) {
    JXL_RETURN_IF_ERROR(doUnpremultiplyAlpha(io.preview_frame));
  }
  for (ImageBundle& ib : io.frames) {
    JXL_RETURN_IF_ERROR(doUnpremultiplyAlpha(ib));
  }
  eci->alpha_associated = false;
  return true;
}
}  // namespace

TEST(JxlTest, RoundtripAlphaPremultiplied) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  const std::vector<uint8_t> orig =
      ReadTestData("external/wesaturate/500px/tmshre_riaphotographs_alpha.png");
  auto io = jxl::make_unique<CodecInOut>(memory_manager);
  auto io_nopremul = jxl::make_unique<CodecInOut>(memory_manager);
  ASSERT_TRUE(SetFromBytes(Bytes(orig), io.get()));
  ASSERT_TRUE(SetFromBytes(Bytes(orig), io_nopremul.get()));

  ASSERT_NE(io->xsize(), 0u);
  ASSERT_TRUE(io->metadata.m.HasAlpha());
  ASSERT_TRUE(io->Main().HasAlpha());
  ASSERT_TRUE(io->ShrinkTo(300, 300));
  ASSERT_TRUE(io_nopremul->ShrinkTo(300, 300));

  CompressParams cparams;
  cparams.butteraugli_distance = 1.0;
  cparams.SetCms(*JxlGetDefaultCms());

  EXPECT_FALSE(io->Main().AlphaIsPremultiplied());
  EXPECT_TRUE(PremultiplyAlpha(*io));
  EXPECT_TRUE(io->Main().AlphaIsPremultiplied());

  EXPECT_FALSE(io_nopremul->Main().AlphaIsPremultiplied());

  std::vector<uint8_t> compressed;
  EXPECT_TRUE(test::EncodeFile(cparams, io.get(), &compressed));
  EXPECT_LE(compressed.size(), 18613u);

  for (bool use_image_callback : {false, true}) {
    for (bool unpremul_alpha : {false, true}) {
      for (bool use_uint8 : {false, true}) {
        printf(
            "Testing premultiplied alpha using %s %s requesting "
            "%spremultiplied output.\n",
            use_uint8 ? "uint8" : "float",
            use_image_callback ? "image callback" : "image_buffer",
            unpremul_alpha ? "un" : "");
        auto io2 = jxl::make_unique<CodecInOut>(memory_manager);
        JXLDecompressParams dparams;
        dparams.use_image_callback = use_image_callback;
        dparams.unpremultiply_alpha = unpremul_alpha;
        if (use_uint8) {
          dparams.accepted_formats.push_back(
              {4, JXL_TYPE_UINT8, JXL_LITTLE_ENDIAN, 0});
        }
        EXPECT_TRUE(test::DecodeFile(dparams, Bytes(compressed), io2.get()));

        EXPECT_EQ(unpremul_alpha, !io2->Main().AlphaIsPremultiplied());
        if (!unpremul_alpha) {
          EXPECT_SLIGHTLY_BELOW(
              ButteraugliDistance(io->frames, io2->frames, ButteraugliParams(),
                                  *JxlGetDefaultCms(),
                                  /*distmap=*/nullptr),
              1.111);
          EXPECT_TRUE(UnpremultiplyAlpha(*io2));
          EXPECT_FALSE(io2->Main().AlphaIsPremultiplied());
        }
        EXPECT_SLIGHTLY_BELOW(
            ButteraugliDistance(io_nopremul->frames, io2->frames,
                                ButteraugliParams(), *JxlGetDefaultCms(),
                                /*distmap=*/nullptr),
            1.1);
      }
    }
  }
}

TEST(JxlTest, RoundtripAlphaResampling) {
  ThreadPool* pool = nullptr;
  const std::vector<uint8_t> orig =
      ReadTestData("external/wesaturate/500px/tmshre_riaphotographs_alpha.png");
  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));
  t.ClearMetadata();
  ASSERT_NE(t.ppf().info.xsize, 0u);
  ASSERT_TRUE(t.ppf().info.alpha_bits > 0);

  JXLCompressParams cparams;
  cparams.alpha_distance = 1.0;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 5);  // kHare
  cparams.AddOption(JXL_ENC_FRAME_SETTING_RESAMPLING, 2);
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EXTRA_CHANNEL_RESAMPLING, 2);
  extras::JXLDecompressParams dparams;

  PackedPixelFile ppf_out;
  EXPECT_NEAR(Roundtrip(t.ppf(), cparams, dparams, pool, &ppf_out), 13700, 130);
  EXPECT_SLIGHTLY_BELOW(ButteraugliDistance(t.ppf(), ppf_out), 4.85);
}

TEST(JxlTest, RoundtripAlphaResamplingOnlyAlpha) {
  ThreadPool* pool = nullptr;
  const std::vector<uint8_t> orig =
      ReadTestData("external/wesaturate/500px/tmshre_riaphotographs_alpha.png");
  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));
  t.ClearMetadata();
  ASSERT_NE(t.ppf().info.xsize, 0u);
  ASSERT_TRUE(t.ppf().info.alpha_bits > 0);

  JXLCompressParams cparams;
  cparams.alpha_distance = 1.0;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 3);  // kFalcon
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EXTRA_CHANNEL_RESAMPLING, 2);
  extras::JXLDecompressParams dparams;

  PackedPixelFile ppf_out;
  EXPECT_NEAR(Roundtrip(t.ppf(), cparams, dparams, pool, &ppf_out), 33136,
              1000);
  EXPECT_SLIGHTLY_BELOW(ButteraugliDistance(t.ppf(), ppf_out), 1.52);
}

TEST(JxlTest, RoundtripAlphaNonMultipleOf8) {
  ThreadPool* pool = nullptr;
  const std::vector<uint8_t> orig =
      ReadTestData("external/wesaturate/500px/tmshre_riaphotographs_alpha.png");
  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));
  t.ClearMetadata();
  ASSERT_TRUE(t.SetDimensions(12, 12));
  ASSERT_NE(t.ppf().info.xsize, 0u);
  ASSERT_TRUE(t.ppf().info.alpha_bits > 0);
  EXPECT_EQ(t.ppf().frames[0].color.format.data_type, JXL_TYPE_UINT8);

  PackedPixelFile ppf_out;
  EXPECT_NEAR(Roundtrip(t.ppf(), {}, {}, pool, &ppf_out), 78, 10);
  EXPECT_SLIGHTLY_BELOW(ButteraugliDistance(t.ppf(), ppf_out), 0.007);
}

TEST(JxlTest, RoundtripAlpha16) {
  ThreadPoolForTests pool(4);
  // The image is wider than 512 pixels to ensure multiple groups are tested.
  size_t xsize = 1200;
  size_t ysize = 160;
  TestImage t;
  ASSERT_TRUE(t.SetDimensions(xsize, ysize));
  ASSERT_TRUE(t.SetChannels(4));
  t.SetAllBitDepths(16);
  JXL_TEST_ASSIGN_OR_DIE(auto frame, t.AddFrame());
  // Generate 16-bit pattern that uses various colors and alpha values.
  const float mul = 1.0f / 65535;
  for (size_t y = 0; y < ysize; y++) {
    for (size_t x = 0; x < xsize; x++) {
      uint16_t r = y * 65535 / ysize;
      uint16_t g = x * 65535 / xsize;
      uint16_t b = (y + x) * 65535 / (xsize + ysize);
      ASSERT_TRUE(frame.SetValue(y, x, 0, r * mul));
      ASSERT_TRUE(frame.SetValue(y, x, 1, g * mul));
      ASSERT_TRUE(frame.SetValue(y, x, 2, b * mul));
      ASSERT_TRUE(frame.SetValue(y, x, 3, g * mul));
    }
  }

  ASSERT_NE(t.ppf().info.xsize, 0u);
  ASSERT_EQ(t.ppf().info.alpha_bits, 16u);

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 6);  // kWombat
  cparams.distance = 0.5;
  cparams.alpha_distance = 0.5;
  extras::JXLDecompressParams dparams;

  PackedPixelFile ppf_out;
  // TODO(szabadka) Investigate big size difference on i686
  // This still keeps happening (2023-04-18).
  EXPECT_NEAR(Roundtrip(t.ppf(), cparams, dparams, pool.get(), &ppf_out), 3810,
              120);
  EXPECT_SLIGHTLY_BELOW(ButteraugliDistance(t.ppf(), ppf_out), 0.6);
}

TEST(JxlTest, LosslessProfileScopesRestoreThreadContext) {
  EXPECT_EQ(LosslessProfileCurrent(), 0u);
  {
    LosslessProfileScope root(0, "test.root", true);
    const uint64_t root_id = root.Id();
    ASSERT_NE(root_id, 0u);
    LosslessProfileScope disabled(0, "test.disabled");
    EXPECT_EQ(disabled.Id(), 0u);
    EXPECT_EQ(LosslessProfileCurrent(), root_id);
    ThreadPoolForTests pool(4);
    std::vector<uint64_t> ids(16);
    ASSERT_TRUE(RunOnPool(
        pool.get(), 0, ids.size(), ThreadPool::NoInit,
        [&](uint32_t i, size_t) -> Status {
          const uint64_t before = LosslessProfileCurrent();
          LosslessProfileScope child(root_id, "test.child");
          ids[i] = child.Id();
          EXPECT_EQ(LosslessProfileCurrent(), child.Id());
          child.Finish();
          child.Finish();
          EXPECT_EQ(LosslessProfileCurrent(), before);
          return true;
        }, "ProfileContextTest"));
    std::sort(ids.begin(), ids.end());
    EXPECT_NE(ids.front(), 0u);
    EXPECT_EQ(std::adjacent_find(ids.begin(), ids.end()), ids.end());
    EXPECT_EQ(LosslessProfileCurrent(), root_id);
  }
  EXPECT_EQ(LosslessProfileCurrent(), 0u);
}

TEST(JxlTest, LosslessSearchAutoOutputMode) {
  CompressParams params;
  params.SetLossless();
  params.speed_tier = SpeedTier::kLightning;
  params.output_mode = -1;
  EXPECT_TRUE(HasLosslessSearchOptions(params));
  params.output_mode = 0;
  EXPECT_TRUE(HasLosslessSearchOptions(params));
  params.output_mode = 1;
  EXPECT_FALSE(HasLosslessSearchOptions(params));
  params.output_mode = 2;
  EXPECT_FALSE(HasLosslessSearchOptions(params));
  params.output_mode = -1;
  params.buffering = 3;
  EXPECT_FALSE(HasLosslessSearchOptions(params));
}

TEST(JxlTest, LosslessDiverseFrontier) {
  CompressParams palette;
  palette.palette_colors = 10000;
  palette.options.predictor = Predictor::Zero;
  palette.options.nb_repeats = 1;
  auto cheap = palette;
  cheap.options.nb_repeats = .01f;
  auto raw = palette;
  raw.palette_colors = 0;
  raw.options.predictor = Predictor::Variable;
  std::vector<LosslessProbeCandidate> plan{{"deep", palette}, {"cheap", cheap}, {"raw", raw}};
  plan[0].context_bytes = 10000;
  plan[1].context_bytes = 10020;
  plan[2].context_bytes = 11000;
  auto ranked = DiverseLosslessFrontier(plan);
  ASSERT_EQ(ranked.size(), 3u);
  EXPECT_STREQ(ranked[0].name, "cheap");
  EXPECT_STREQ(ranked[1].name, "raw");
  EXPECT_STREQ(ranked[2].name, "deep");
  plan[1].context_bytes = 10026;
  ranked = DiverseLosslessFrontier(plan);
  EXPECT_STREQ(ranked[0].name, "deep");
  EXPECT_STREQ(ranked[1].name, "raw");
  EXPECT_STREQ(ranked[2].name, "cheap");
  EXPECT_TRUE(DiverseLosslessFrontier({}).empty());
}

TEST(JxlTest, LosslessFrameSearchRetainsSmallerCandidates) {
  ThreadPoolForTests pool(4);
  for (size_t bits : {8, 16}) {
    TestImage t;
    ASSERT_TRUE(t.SetDimensions(137, 96));
    ASSERT_TRUE(t.SetChannels(4));
    t.SetAllBitDepths(bits);
    JXL_TEST_ASSIGN_OR_DIE(auto frame, t.AddFrame());
    const uint32_t maxval = (1u << bits) - 1;
    for (size_t y = 0; y < 96; ++y) {
      for (size_t x = 0; x < 137; ++x) {
        for (size_t c = 0; c < 4; ++c) {
          const uint32_t value =
              c == 3 ? (x % 3) * (maxval / 2)
                     : ((x % 13) * 103 + (y % 17) * 47 + c) % (maxval + 1);
          ASSERT_TRUE(frame.SetValue(y, x, c, value * (1.0f / maxval)));
        }
      }
    }
    JXLCompressParams params;
    params.distance = 0.0;
    params.AddOption(JXL_ENC_FRAME_SETTING_KEEP_INVISIBLE, 1);
    JXLDecompressParams dparams;
    dparams.accepted_formats.push_back(t.ppf().frames[0].color.format);
    PackedPixelFile decoded;
    size_t previous_size = 0;
    for (int effort = 1; effort <= 10; ++effort) {
      auto candidate = params;
      candidate.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, effort);
      const size_t size =
          Roundtrip(t.ppf(), candidate, dparams, pool.get(), &decoded);
      ASSERT_GT(size, 0u);
      EXPECT_EQ(ComputeDistance2(t.ppf(), decoded), 0.0);
      if (previous_size != 0) EXPECT_LE(size, previous_size);
      previous_size = size;
    }
  }
}

TEST(JxlTest, LosslessFrameSearchSmallFixedPredictors) {
  TestImage t;
  ASSERT_TRUE(t.SetDimensions(32, 32));
  ASSERT_TRUE(t.SetChannels(1));
  t.SetAllBitDepths(16);
  JXL_TEST_ASSIGN_OR_DIE(auto frame, t.AddFrame());
  for (size_t y = 0; y < 32; ++y) {
    for (size_t x = 0; x < 32; ++x) {
      ASSERT_TRUE(frame.SetValue(y, x, 0, (x * 2000 + y % 4) / 65535.f));
    }
  }
  JXLCompressParams params;
  params.distance = 0;
  params.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 9);
  JXLDecompressParams dparams;
  dparams.accepted_formats.push_back(t.ppf().frames[0].color.format);
  PackedPixelFile decoded;
  const size_t searched =
      Roundtrip(t.ppf(), params, dparams, nullptr, &decoded);
  ASSERT_GT(searched, 0u);
  EXPECT_EQ(ComputeDistance2(t.ppf(), decoded), 0.0);
  params.AddOption(JXL_ENC_FRAME_SETTING_MODULAR_PREDICTOR, 2);
  params.AddFloatOption(JXL_ENC_FRAME_SETTING_MODULAR_MA_TREE_LEARNING_PERCENT,
                        0.f);
  const size_t fixed_top =
      Roundtrip(t.ppf(), params, dparams, nullptr, &decoded);
  ASSERT_GT(fixed_top, 0u);
  EXPECT_EQ(ComputeDistance2(t.ppf(), decoded), 0.0);
  EXPECT_LE(searched, fixed_top);
}

TEST(JxlTest, LosslessSearchFeedbackAndRunnerOrder) {
  // Mock byte costs exercise the policy without relying on a photographic
  // training sample continuing to favor the same predictor after updates.
  JxlMemoryManager* memory = test::MemoryManager();
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(257, 257));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  std::vector<uint8_t> pixels(257 * 257 * 3, 0);
  JxlEncoderChunkedFrameAdapter input(257, 257, 0);
  ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                  {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}));
  FrameInfo frame_info;
  for (int scenario : {0, 1, 2}) {
    const bool literal_wins = scenario == 1;
    std::vector<uint8_t> serial_result;
    for (int order : {0, 17, 91}) {
      FakeParallelRunner runner(order, 4);
      ThreadPool pool(&JxlFakeParallelRunner, &runner);
      size_t previous = 0;
      for (int effort = 1; effort <= 10; ++effort) {
        CompressParams params;
        params.SetLossless();
        params.speed_tier = static_cast<SpeedTier>(10 - effort);
        JxlEncoderOutputProcessorWrapper output(memory);
        int ordinary_calls = 0;
        int learned_calls = 0;
        const auto encode = [&](const CompressParams& p, ThreadPool* inner,
                                JxlEncoderOutputProcessorWrapper* out,
                                AuxOut*) -> Status {
          const bool literal = p.options.predictor == Predictor::Zero;
          if (literal && p.options.nb_repeats > 0) ++learned_calls;
          if (!literal) ++ordinary_calls;
          // Scenario 2 models the map regression: a poor fixed-tree probe,
          // followed by a learned tree that beats the initially winning
          // literal.
          const size_t bytes =
              literal
                  ? (scenario != 0 ? 50 : 500)
                  : (scenario == 2 && p.speed_tier != SpeedTier::kFalcon ? 25
                                                                         : 200);
          if (effort >= 2 && p.speed_tier == SpeedTier::kFalcon)
            EXPECT_EQ(inner, nullptr);
          return AppendData(*out, std::vector<uint8_t>(bytes, literal ? 1 : 2));
        };
        JXL_TEST_ASSIGN_OR_DIE(
            bool searched,
            EncodeLosslessSearch(memory, params, frame_info, metadata, input,
                                 order == 0 ? nullptr : &pool, &output, nullptr,
                                 encode));
        ASSERT_TRUE(searched);
        ASSERT_TRUE(output.SetFinalizedPosition());
        std::vector<uint8_t> bytes;
        ASSERT_TRUE(output.CopyOutput(bytes));
        if (previous) EXPECT_LE(bytes.size(), previous);
        previous = bytes.size();
        if (literal_wins && effort >= 2)
          EXPECT_EQ(ordinary_calls, effort >= 3 ? 2 : 1);
        if (scenario == 0 || scenario == 2) EXPECT_EQ(learned_calls, 0);
        if (scenario == 2 && effort >= 3) EXPECT_EQ(bytes.size(), 25);
        if (effort == 10) {
          if (order == 0)
            serial_result = bytes;
          else
            EXPECT_EQ(bytes, serial_result);
        }
      }
    }
  }
}

TEST(JxlTest, LosslessFastFamilyFeedbackAndEffortPrefix) {
  const char* flag = std::getenv("JXL_LOSSLESS_FAST_FAMILY");
  if (!flag || std::strcmp(flag, "1") != 0) {
    GTEST_SKIP() << "Opt-in fast-family experiment is disabled";
  }
  auto* memory = test::MemoryManager();
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(1025, 513));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  std::vector<uint8_t> pixels(1025 * 513 * 3, 17);
  JxlEncoderChunkedFrameAdapter input(1025, 513, 0);
  ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                  {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}));
  for (int scenario : {0, 1, 2}) {
    const bool dominant = scenario != 1;
    for (int order : {0, 17}) {
      FakeParallelRunner runner(order, 4);
      ThreadPool pool(&JxlFakeParallelRunner, &runner);
      size_t previous = std::numeric_limits<size_t>::max();
      for (int effort = 1; effort <= 10; ++effort) {
        CompressParams params;
        params.SetLossless();
        params.speed_tier = static_cast<SpeedTier>(10 - effort);
        std::atomic<size_t> probes{0};
        const auto probe = [&](const CompressParams&, const CodecMetadata&,
                               JxlEncoderChunkedFrameAdapter&) -> StatusOr<size_t> {
          ++probes;
          return size_t{100};
        };
        const auto encode = [&](const CompressParams& p, ThreadPool*,
                                JxlEncoderOutputProcessorWrapper* out,
                                AuxOut*) -> Status {
          size_t bytes = p.speed_tier == SpeedTier::kFalcon ? 1000 : 900;
          if (p.options.predictor == Predictor::Zero) {
            bytes = dominant ? 400 : 950;
            if (p.options.nb_repeats == 0.01f) bytes = 230;
            if (p.options.nb_repeats == 0.03f) bytes = 220;
            if (p.options.nb_repeats == 1.0f) bytes = 210;
            if (p.speed_tier == SpeedTier::kGlacier) bytes = 200;
          }
          return AppendData(*out, std::vector<uint8_t>(bytes * (scenario == 2 ? 1 : 16), 0));
        };
        JxlEncoderOutputProcessorWrapper output(memory);
        JXL_TEST_ASSIGN_OR_DIE(bool searched, EncodeLosslessSearch(
            memory, params, FrameInfo{}, metadata, input,
            order == 0 ? nullptr : &pool, &output, nullptr, encode, probe));
        ASSERT_TRUE(searched);
        ASSERT_TRUE(output.SetFinalizedPosition());
        std::vector<uint8_t> bytes;
        ASSERT_TRUE(output.CopyOutput(bytes));
        EXPECT_LE(bytes.size(), previous);
        previous = bytes.size();
        if (scenario == 0) {
          const size_t expected[] = {1000, 400, 400, 230, 230, 220, 220, 220, 210, 200};
          EXPECT_EQ(bytes.size(), expected[effort - 1] * 16);
          EXPECT_EQ(probes.load(), 0u);
        } else if (effort >= 4) {
          EXPECT_GT(probes.load(), 0u);
        }
      }
    }
  }
}

TEST(JxlTest, LosslessProbePreservesSourceSamples) {
  constexpr size_t width = 257;
  constexpr size_t height = 131;
  constexpr size_t stride = 2304;  // RGBA16 padded to 256-byte alignment.
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(width, height));
  metadata.m.SetUintSamples(16);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  metadata.m.SetAlphaBits(16);
  metadata.m.extra_channel_info.emplace_back();
  metadata.m.num_extra_channels = 2;
  metadata.m.extra_channel_info[1].type = ExtraChannel::kSelectionMask;
  metadata.m.extra_channel_info[1].bit_depth.bits_per_sample = 16;
  metadata.m.have_animation = true;
  std::vector<uint8_t> pixels(stride * height, 0xa5);
  std::vector<uint8_t> mask(width * height * 2);
  const auto store = [](uint8_t* p, size_t value) {
    p[0] = value >> 8;
    p[1] = value & 255;
  };
  const auto load = [](const uint8_t* p) -> size_t {
    return (size_t{p[0]} << 8) | p[1];
  };
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      uint8_t* p = pixels.data() + y * stride + 8 * x;
      store(p, x);
      store(p + 2, y);
      store(p + 4, x + 3 * y);
      store(p + 6, x < 64 ? 0 : 12345);
      store(mask.data() + 2 * (y * width + x), 29 * x + y);
    }
  }
  const auto original = pixels;
  JxlEncoderChunkedFrameAdapter input(width, height, 2);
  ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                  {4, JXL_TYPE_UINT16, JXL_BIG_ENDIAN, 256}));
  ASSERT_TRUE(input.SetFromBuffer(2, mask.data(), mask.size(),
                                  {1, JXL_TYPE_UINT16, JXL_BIG_ENDIAN, 0}));
  CompressParams params;
  params.SetLossless();
  std::atomic<size_t> calls{0};
  const auto encode = [&](const CompressParams& p, const CodecMetadata& m,
                          JxlEncoderChunkedFrameAdapter& sample)
      -> StatusOr<size_t> {
    ++calls;
    EXPECT_EQ(sample.xsize, 64u);
    EXPECT_EQ(sample.ysize, 64u);
    EXPECT_EQ(m.xsize(), sample.xsize);
    EXPECT_FALSE(m.m.have_animation);
    auto source = sample.GetInputSource();
    JxlPixelFormat format{};
    source.get_color_channels_pixel_format(source.opaque, &format);
    EXPECT_EQ(format.endianness, JXL_BIG_ENDIAN);
    EXPECT_EQ(format.align, 0u);
    size_t row_stride;
    auto color = GetColorBuffer(source, 0, 0, 64, 64, &row_stride);
    size_t mask_stride;
    auto extra = GetExtraChannelBuffer(source, 1, 0, 0, 64, 64, &mask_stride);
    const auto* data = static_cast<const uint8_t*>(color.get());
    const auto* selection = static_cast<const uint8_t*>(extra.get());
    const size_t x0 = load(data);
    const size_t y0 = load(data + 2);
    for (size_t y : {0, 31, 63}) {
      for (size_t x : {0, 31, 63}) {
        const uint8_t* pixel = data + y * row_stride + x * 8;
        EXPECT_EQ(load(pixel), x0 + x);
        EXPECT_EQ(load(pixel + 2), y0 + y);
        EXPECT_EQ(load(pixel + 4), x0 + x + 3 * (y0 + y));
        EXPECT_EQ(load(pixel + 6), x0 + x < 64 ? 0u : 12345u);
        EXPECT_EQ(load(selection + y * mask_stride + 2 * x),
                  29 * (x0 + x) + y0 + y);
      }
    }
    return size_t{p.colorspace == 0 && p.options.predictor == Predictor::Gradient
                      ? 100u : 1000u};
  };
  ThreadPoolForTests pool(4);
  JXL_TEST_ASSIGN_OR_DIE(auto result, ProbeLosslessCandidates(
      params, metadata, input, pool.get(), encode));
  EXPECT_EQ(result.samples, 9u);
  EXPECT_EQ(result.encodes, calls.load());
  ASSERT_FALSE(result.ranked.empty());
  EXPECT_STREQ(result.ranked[0].name, "raw-gradient");
  EXPECT_EQ(result.ranked[0].bytes, 900u);
  EXPECT_EQ(pixels, original);
}

TEST(JxlTest, LosslessScanCoversOffCenterContentAndReusesScores) {
  constexpr size_t width = 1537, height = 2049;
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(width, height));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  std::vector<uint8_t> pixels(width * height * 3, 255);
  for (size_t y = 107; y < 151; ++y) {
    for (size_t x = 119; x < 291; ++x) {
      pixels[(y * width + x) * 3] = (x + y) % 7;
    }
  }
  const auto original = pixels;
  JxlEncoderChunkedFrameAdapter input(width, height, 0);
  ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                  {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}));
  ThreadPoolForTests pool(4);
  JXL_TEST_ASSIGN_OR_DIE(auto serial, ScanLosslessInput(metadata, input, nullptr));
  JXL_TEST_ASSIGN_OR_DIE(auto scan, ScanLosslessInput(metadata, input, pool.get()));
  ASSERT_TRUE(scan.Available());
  EXPECT_EQ(scan.pixels, width * height);
  ASSERT_EQ(scan.tiles.size(), serial.tiles.size());
  for (size_t i = 0; i < scan.tiles.size(); ++i) {
    EXPECT_EQ(scan.tiles[i].horizontal, serial.tiles[i].horizontal);
    EXPECT_EQ(scan.tiles[i].vertical, serial.tiles[i].vertical);
    EXPECT_EQ(scan.tiles[i].anchor_x, serial.tiles[i].anchor_x);
    EXPECT_EQ(scan.tiles[i].anchor_y, serial.tiles[i].anchor_y);
  }
  const auto regions = scan.Regions();
  ASSERT_FALSE(regions.empty());
  for (const auto& r : regions) {
    EXPECT_LE(r.x + r.width, width);
    EXPECT_LE(r.y + r.height, height);
    bool content = false;
    for (size_t y = r.y; y < r.y + r.height; ++y) {
      for (size_t x = r.x; x < r.x + r.width; ++x) {
        content |= pixels[(y * width + x) * 3] != 255;
      }
    }
    EXPECT_TRUE(content);
  }
  std::atomic<size_t> calls{0};
  const auto encode = [&](const CompressParams&, const CodecMetadata&,
                          JxlEncoderChunkedFrameAdapter&) -> StatusOr<size_t> {
    ++calls;
    return size_t{100};
  };
  LosslessCandidateScreen screen(metadata, input, pool.get(), encode, &scan);
  CompressParams params;
  params.SetLossless();
  JXL_TEST_ASSIGN_OR_DIE(auto first, screen.Score(params));
  JXL_TEST_ASSIGN_OR_DIE(auto again, screen.Score(params));
  EXPECT_EQ(first, again);
  EXPECT_EQ(calls.load(), regions.size());
  JXL_TEST_ASSIGN_OR_DIE(auto analysis, screen.Analyze(params, params));
  const size_t after = calls.load();
  for (const auto& candidate : analysis.ranked) {
    JXL_TEST_ASSIGN_OR_DIE(auto scores, screen.Score(candidate.params));
    EXPECT_EQ(std::accumulate(scores.begin(), scores.end(), size_t{0}),
              candidate.bytes);
  }
  EXPECT_EQ(calls.load(), after);
  EXPECT_EQ(pixels, original);
}

TEST(JxlTest, LosslessScanIncludesHiddenRgbAndSeparateUint16Channel) {
  constexpr size_t width = 1025, height = 513;
  constexpr size_t stride = (width * 8 + 255) / 256 * 256;
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(width, height));
  metadata.m.SetUintSamples(16);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  metadata.m.SetAlphaBits(16);
  metadata.m.extra_channel_info.emplace_back();
  metadata.m.num_extra_channels = 2;
  metadata.m.extra_channel_info[1].type = ExtraChannel::kSelectionMask;
  metadata.m.extra_channel_info[1].bit_depth.bits_per_sample = 16;
  std::vector<uint8_t> pixels(stride * height, 0);
  std::vector<uint8_t> mask(width * height * 2, 0);
  // Only a low sample byte changes; alpha stays zero throughout.
  pixels[97 * stride + 89 * 8 + 1] = 1;
  mask[2 * (383 * width + 913) + 1] = 1;
  JxlEncoderChunkedFrameAdapter input(width, height, 2);
  ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                  {4, JXL_TYPE_UINT16, JXL_BIG_ENDIAN, 256}));
  ASSERT_TRUE(input.SetFromBuffer(2, mask.data(), mask.size(),
                                  {1, JXL_TYPE_UINT16, JXL_BIG_ENDIAN, 0}));
  JXL_TEST_ASSIGN_OR_DIE(auto scan, ScanLosslessInput(metadata, input, nullptr));
  uint64_t horizontal = 0, vertical = 0;
  for (const auto& tile : scan.tiles) {
    horizontal += tile.horizontal;
    vertical += tile.vertical;
  }
  EXPECT_EQ(horizontal, 4u);
  EXPECT_EQ(vertical, 4u);
  EXPECT_EQ(scan.pixels, width * height);
}

TEST(JxlTest, LosslessScanUniformAndNarrowImages) {
  for (const auto& shape : {std::make_pair(1u, 1u), std::make_pair(1u, 8193u),
                             std::make_pair(8193u, 1u)}) {
    const size_t width = shape.first, height = shape.second;
    CodecMetadata metadata;
    ASSERT_TRUE(metadata.size.Set(width, height));
    metadata.m.SetUintSamples(8);
    metadata.m.color_encoding = ColorEncoding::SRGB(true);
    std::vector<uint8_t> pixels(width * height, 17);
    JxlEncoderChunkedFrameAdapter input(width, height, 0);
    ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                    {1, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}));
    JXL_TEST_ASSIGN_OR_DIE(auto scan, ScanLosslessInput(metadata, input, nullptr));
    EXPECT_TRUE(scan.Available());
    EXPECT_TRUE(scan.literal_candidate);
    EXPECT_LE(scan.tiles.size(), 4096u);
    for (const auto& r : scan.Regions()) {
      EXPECT_LE(r.x + r.width, width);
      EXPECT_LE(r.y + r.height, height);
      EXPECT_GT(r.width * r.height, 0u);
    }
  }
}

TEST(JxlTest, LosslessCandidateScreenPreservesSamplesAndCachesScores) {
  constexpr size_t width = 2053, height = 1025;
  constexpr size_t stride = (width * 8 + 255) / 256 * 256;
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(width, height));
  metadata.m.SetUintSamples(16);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  metadata.m.SetAlphaBits(16);
  metadata.m.extra_channel_info.emplace_back();
  metadata.m.num_extra_channels = 2;
  metadata.m.extra_channel_info[1].type = ExtraChannel::kSelectionMask;
  metadata.m.extra_channel_info[1].bit_depth.bits_per_sample = 16;
  metadata.m.have_animation = true;
  std::vector<uint8_t> pixels(stride * height, 0xa5);
  std::vector<uint8_t> mask(width * height * 2);
  const auto store = [](uint8_t* p, size_t value) {
    p[0] = value >> 8;
    p[1] = value & 255;
  };
  const auto load = [](const uint8_t* p) -> size_t {
    return (size_t{p[0]} << 8) | p[1];
  };
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      uint8_t* p = pixels.data() + y * stride + 8 * x;
      store(p, x);
      store(p + 2, y);
      store(p + 4, x + 3 * y);
      store(p + 6, x < width / 2 ? 0 : 12345);
      store(mask.data() + 2 * (y * width + x), 29 * x + y);
    }
  }
  const auto original = pixels;
  JxlEncoderChunkedFrameAdapter input(width, height, 2);
  ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                  {4, JXL_TYPE_UINT16, JXL_BIG_ENDIAN, 256}));
  ASSERT_TRUE(input.SetFromBuffer(2, mask.data(), mask.size(),
                                  {1, JXL_TYPE_UINT16, JXL_BIG_ENDIAN, 0}));
  CompressParams params;
  params.SetLossless();
  std::atomic<size_t> calls{0};
  const auto encode = [&](const CompressParams& p, const CodecMetadata& m,
                          JxlEncoderChunkedFrameAdapter& sample)
      -> StatusOr<size_t> {
    ++calls;
    EXPECT_TRUE((sample.xsize == 1024 && sample.ysize == 64) ||
                (sample.xsize == 256 && sample.ysize == 256));
    EXPECT_EQ(m.xsize(), sample.xsize);
    EXPECT_EQ(m.ysize(), sample.ysize);
    EXPECT_FALSE(m.m.have_animation);
    EXPECT_EQ(p.lossless_trial_budget, nullptr);
    auto source = sample.GetInputSource();
    JxlPixelFormat format{};
    source.get_color_channels_pixel_format(source.opaque, &format);
    EXPECT_EQ(format.endianness, JXL_BIG_ENDIAN);
    EXPECT_EQ(format.align, 0u);
    size_t row_stride, mask_stride;
    auto color = GetColorBuffer(source, 0, 0, sample.xsize, sample.ysize,
                                &row_stride);
    auto extra = GetExtraChannelBuffer(source, 1, 0, 0, sample.xsize,
                                       sample.ysize, &mask_stride);
    const auto* data = static_cast<const uint8_t*>(color.get());
    const auto* selection = static_cast<const uint8_t*>(extra.get());
    const size_t x0 = load(data), y0 = load(data + 2);
    for (size_t y : {size_t{0}, sample.ysize / 2, sample.ysize - 1}) {
      for (size_t x : {size_t{0}, sample.xsize / 2, sample.xsize - 1}) {
        const uint8_t* pixel = data + y * row_stride + x * 8;
        EXPECT_EQ(load(pixel), x0 + x);
        EXPECT_EQ(load(pixel + 2), y0 + y);
        EXPECT_EQ(load(pixel + 4), x0 + x + 3 * (y0 + y));
        EXPECT_EQ(load(pixel + 6), x0 + x < width / 2 ? 0u : 12345u);
        EXPECT_EQ(load(selection + y * mask_stride + 2 * x),
                  29 * (x0 + x) + y0 + y);
      }
    }
    return size_t{sample.xsize + sample.ysize +
                  static_cast<size_t>(p.modular_group_size_shift + 1)};
  };
  std::vector<size_t> serial;
  ThreadPoolForTests pool(4);
  for (ThreadPool* runner : {static_cast<ThreadPool*>(nullptr), pool.get()}) {
    calls = 0;
    LosslessCandidateScreen screen(metadata, input, runner, encode);
    ASSERT_TRUE(screen.Available());
    JXL_TEST_ASSIGN_OR_DIE(auto scores, screen.Score(params));
    EXPECT_EQ(calls.load(), 4u);
    EXPECT_EQ(screen.SamplePixels(), 3u * 1024 * 64 + 256 * 256);
    auto diagnostic = params;
    diagnostic.options.histogram_params.lossless_profile = 1234;
    LosslessTrialBudget budget(1);
    diagnostic.lossless_trial_budget = &budget;
    JXL_TEST_ASSIGN_OR_DIE(auto cached, screen.Score(diagnostic));
    EXPECT_EQ(scores, cached);
    EXPECT_EQ(calls.load(), 4u);
    diagnostic.modular_group_size_shift = 2;
    JXL_TEST_ASSIGN_OR_DIE(auto changed, screen.Score(diagnostic));
    EXPECT_NE(scores, changed);
    EXPECT_EQ(calls.load(), 8u);
    EXPECT_EQ(screen.Encodes(), 8u);
    if (!runner) serial = scores;
    else EXPECT_EQ(scores, serial);
    auto gradient = params;
    gradient.options.predictor = Predictor::Gradient;
    auto literal = params;
    literal.options.predictor = Predictor::Zero;
    JXL_TEST_ASSIGN_OR_DIE(
        auto ranked,
        screen.Rank({{"gradient", gradient}, {"literal", literal}}, params, true));
    ASSERT_EQ(ranked.size(), 2u);
    for (const auto& candidate : ranked) {
      EXPECT_EQ(candidate.context_bytes,
                std::accumulate(scores.begin(), scores.end(), size_t{0}));
    }
  }
  EXPECT_EQ(pixels, original);
  EXPECT_FALSE(LosslessCandidateScreen::Promising({110, 110}, {100, 100}));
  EXPECT_TRUE(LosslessCandidateScreen::Promising({90, 120}, {100, 100}));
  EXPECT_FALSE(LosslessCandidateScreen::Promising({100, 100}, {100, 100}));
  EXPECT_TRUE(LosslessCandidateScreen::Promising({}, {100, 100}));
}

TEST(JxlTest, LosslessCandidateScreenRacesEqualWorkAndResumesCachedSamples) {
  constexpr size_t width = 1025, height = 513;
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(width, height));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  std::vector<uint8_t> pixels(width * height * 3, 23);
  JxlEncoderChunkedFrameAdapter input(width, height, 0);
  ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                  {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}));
  CompressParams params;
  params.SetLossless();
  std::mutex mutex;
  std::vector<int> calls;
  const auto encode = [&](const CompressParams& p, const CodecMetadata&,
                          JxlEncoderChunkedFrameAdapter&) -> StatusOr<size_t> {
    std::lock_guard<std::mutex> lock(mutex);
    calls.push_back(p.colorspace);
    return size_t{p.colorspace < 0 ? 1000u : 100u + 10u * p.colorspace};
  };
  std::vector<LosslessProbeCandidate> plan;
  for (int i = 0; i < 12; ++i) {
    auto candidate = params;
    candidate.colorspace = i;
    candidate.options.predictor = Predictor::Gradient;
    plan.push_back({"test", candidate});
  }
  plan.push_back(plan[0]);  // Duplicate jobs must not race on the same cache slot.
  ThreadPoolForTests pool(4);
  LosslessCandidateScreen screen(metadata, input, pool.get(), encode);
  JXL_TEST_ASSIGN_OR_DIE(auto ranked, screen.Rank(plan, params));
  ASSERT_EQ(ranked.size(), 6u);
  EXPECT_EQ(screen.Encodes(), 4u + 12 * 2 + 6 * 2);
  for (size_t i = 0; i < ranked.size(); ++i) {
    EXPECT_EQ(ranked[i].params.colorspace, i);
    EXPECT_EQ(ranked[i].context_bytes, 4 * (100 + 10 * i));
  }
  // All candidates get the same two-region prefix before the shortlist. The
  // independent regions may share a queue, without a barrier between them.
  std::vector<int> prefix(calls.begin() + 4, calls.begin() + 4 + 24);
  std::sort(prefix.begin(), prefix.end());
  for (int i = 0; i < 24; ++i) EXPECT_EQ(prefix[i], i / 2);
  const size_t batches = screen.Batches();
  const char* batching = std::getenv("JXL_LOSSLESS_BATCH_PROBES");
  EXPECT_EQ(batches, (!batching || std::strcmp(batching, "1") == 0) ? 3u : 5u);
  JXL_TEST_ASSIGN_OR_DIE(auto winner, screen.Score(plan[0].params));
  EXPECT_EQ(screen.Encodes(), 40u);
  EXPECT_EQ(screen.Batches(), batches);
  JXL_TEST_ASSIGN_OR_DIE(auto resumed, screen.Score(plan[11].params));
  EXPECT_EQ(screen.Encodes(), 42u);
  EXPECT_EQ(screen.Batches(), batches + 1);
  EXPECT_EQ(resumed, std::vector<size_t>(4, 210));
  JXL_TEST_ASSIGN_OR_DIE(auto cached, screen.Score(plan[11].params));
  EXPECT_EQ(screen.Encodes(), 42u);
  EXPECT_EQ(cached, resumed);
}

TEST(JxlTest, LosslessPaletteScopeMatchesGlobalCapacity) {
  const char* mode = std::getenv("JXL_LOSSLESS_PALETTE_SCOPE");
  if (!mode || std::strcmp(mode, "1")) GTEST_SKIP();
  constexpr size_t width = 1025, height = 513;
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(width, height));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  metadata.m.SetAlphaBits(8);
  for (bool many : {false, true}) {
    std::vector<uint8_t> pixels(width * height * 4);
    for (size_t i = 0; i < width * height; ++i) {
      const size_t value = many ? i : 17;
      pixels[4 * i] = value & 255;
      pixels[4 * i + 1] = (value >> 8) & 255;
      pixels[4 * i + 2] = (value >> 16) & 255;
      pixels[4 * i + 3] = many ? 0 : i & 255;
    }
    JxlEncoderChunkedFrameAdapter input(width, height, 1);
    ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                    {4, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}));
    JXL_TEST_ASSIGN_OR_DIE(auto scan, ScanLosslessInput(metadata, input, nullptr));
    if (many) EXPECT_GT(scan.palette_color_lower_bound, 70000u);
    else EXPECT_EQ(scan.palette_color_lower_bound, 1u);
    CompressParams params;
    params.SetLossless();
    params.palette_colors = 10000;
    const auto encode = [&](const CompressParams& p, const CodecMetadata&,
                            JxlEncoderChunkedFrameAdapter&) -> StatusOr<size_t> {
      EXPECT_EQ(p.palette_colors, many ? 0 : 10000);
      return size_t{100};
    };
    LosslessCandidateScreen screen(metadata, input, nullptr, encode, &scan);
    JXL_TEST_ASSIGN_OR_DIE(auto scores, screen.Score(params));
    EXPECT_FALSE(scores.empty());
    EXPECT_EQ(params.palette_colors, 10000);
    JXL_TEST_ASSIGN_OR_DIE(auto ranked, screen.Rank({{"palette", params}}, params));
    ASSERT_EQ(ranked.size(), 1u);
    EXPECT_EQ(ranked[0].params.palette_colors, 10000);
  }
}

TEST(JxlTest, LosslessProbeAtlasKeepsPixelsAndExtraChannels) {
  const char* mode = std::getenv("JXL_LOSSLESS_PROBE_ATLAS");
  if (!mode || (std::strcmp(mode, "1") && std::strcmp(mode, "2") && std::strcmp(mode, "3"))) GTEST_SKIP();
  constexpr size_t width = 1025, height = 769;
  const bool uniform = std::strcmp(mode, "3") == 0;
  const size_t expected_height = uniform ? 8 * std::min<size_t>(16, 131072 / width / 8) : 96;
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(width, height));
  metadata.m.SetUintSamples(16);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  metadata.m.SetAlphaBits(16);
  metadata.m.extra_channel_info.emplace_back();
  metadata.m.num_extra_channels = 2;
  metadata.m.extra_channel_info[1].type = ExtraChannel::kSelectionMask;
  metadata.m.extra_channel_info[1].bit_depth.bits_per_sample = 16;
  std::vector<uint16_t> pixels(width * height * 4), mask(width * height);
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      const size_t i = y * width + x;
      pixels[4 * i] = x;
      pixels[4 * i + 1] = y;
      pixels[4 * i + 2] = 3 * x + y;
      pixels[4 * i + 3] = x < width / 2 ? 0 : 23456;
      mask[i] = x + 7 * y;
    }
  }
  JxlEncoderChunkedFrameAdapter input(width, height, 2);
  ASSERT_TRUE(input.SetFromBuffer(0, reinterpret_cast<const uint8_t*>(pixels.data()), pixels.size() * 2,
                                  {4, JXL_TYPE_UINT16, JXL_NATIVE_ENDIAN, 0}));
  ASSERT_TRUE(input.SetFromBuffer(2, reinterpret_cast<const uint8_t*>(mask.data()), mask.size() * 2,
                                  {1, JXL_TYPE_UINT16, JXL_NATIVE_ENDIAN, 0}));
  std::atomic<size_t> calls{0};
  const auto encode = [&](const CompressParams&, const CodecMetadata& m,
                          JxlEncoderChunkedFrameAdapter& sample) -> StatusOr<size_t> {
    ++calls;
    EXPECT_EQ(sample.xsize, width);
    EXPECT_EQ(sample.ysize, expected_height);
    EXPECT_EQ(m.ysize(), sample.ysize);
    auto source = sample.GetInputSource();
    size_t stride, ec_stride;
    auto color = GetColorBuffer(source, 0, 0, sample.xsize, sample.ysize, &stride);
    auto extra = GetExtraChannelBuffer(source, 1, 0, 0, sample.xsize, sample.ysize, &ec_stride);
    EXPECT_EQ(stride, width * 8);
    EXPECT_EQ(ec_stride, width * 2);
    const auto* data = static_cast<const uint16_t*>(color.get());
    const auto* ec = static_cast<const uint16_t*>(extra.get());
    size_t seams = 0, previous_y = 0;
    for (size_t y = 0; y < sample.ysize; ++y) {
      const size_t original_y = data[y * width * 4 + 1];
      EXPECT_LT(original_y, height);
      if (y && original_y != previous_y + 1) ++seams;
      previous_y = original_y;
      for (size_t x = 0; x < width; ++x) {
        const size_t i = y * width + x;
        EXPECT_EQ(data[4 * i], x);
        EXPECT_EQ(data[4 * i + 1], original_y);
        EXPECT_EQ(data[4 * i + 2], 3 * x + original_y);
        EXPECT_EQ(data[4 * i + 3], x < width / 2 ? 0u : 23456u);
        EXPECT_EQ(ec[i], x + 7 * original_y);
      }
    }
    EXPECT_EQ(seams, uniform ? 7u : 2u);
    return size_t{100};
  };
  CompressParams params;
  params.SetLossless();
  ThreadPoolForTests pool(4);
  LosslessScan scan;
  scan.palette_color_lower_bound = 20000;
  LosslessCandidateScreen screen(metadata, input, pool.get(), encode, &scan);
  JXL_TEST_ASSIGN_OR_DIE(auto first, screen.Score(params));
  JXL_TEST_ASSIGN_OR_DIE(auto cached, screen.Score(params));
  EXPECT_EQ(first, cached);
  EXPECT_EQ(first, std::vector<size_t>{100});
  EXPECT_EQ(calls.load(), 1u);
  EXPECT_EQ(screen.SamplePixels(), width * expected_height);
  if (std::strcmp(mode, "2") == 0) {
    scan.palette_color_lower_bound = 100;
    const auto separate = [](const CompressParams&, const CodecMetadata&,
                              JxlEncoderChunkedFrameAdapter&) -> StatusOr<size_t> {
      return size_t{100};
    };
    LosslessCandidateScreen low_capacity(metadata, input, pool.get(), separate, &scan);
    JXL_TEST_ASSIGN_OR_DIE(auto separate_scores, low_capacity.Score(params));
    EXPECT_GT(separate_scores.size(), 1u);
  }
}

TEST(JxlTest, LosslessProbeDataParallelDoesNotNestRunner) {
  const char* value = std::getenv("JXL_LOSSLESS_PROBE_DATA_PARALLEL");
  if (!value || std::strcmp(value, "1")) GTEST_SKIP();
  ThreadPoolForTests workers(4);
  struct Guard { ThreadPool* inner; std::atomic<bool> active{false}; } guard{workers.get()};
  ThreadPool runner([](void* opaque, void* codec, JxlParallelRunInit init,
                        JxlParallelRunFunction run, uint32_t begin, uint32_t end) {
    auto& state = *static_cast<Guard*>(opaque);
    if (state.active.exchange(true)) return JXL_PARALLEL_RET_RUNNER_ERROR;
    const auto result = state.inner->runner()(state.inner->runner_opaque(), codec, init, run, begin, end);
    state.active = false;
    return result;
  }, &guard);
  constexpr size_t width = 1025, height = 769;
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(width, height));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  std::vector<uint8_t> pixels(width * height * 3, 23);
  JxlEncoderChunkedFrameAdapter input(width, height, 0);
  ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                  {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}));
  CompressParams params;
  params.SetLossless();
  std::atomic<size_t> borrowed{0}, tasks{0}, batch{0};
  const auto encode = [&](const CompressParams& p, const CodecMetadata&,
                          JxlEncoderChunkedFrameAdapter&) -> StatusOr<size_t> {
    if (p.lossless_probe_runner) {
      EXPECT_EQ(p.lossless_probe_runner, &runner);
      ++borrowed;
      JXL_RETURN_IF_ERROR(RunOnPool(p.lossless_probe_runner, 0, 17,
          ThreadPool::NoInit, [&](uint32_t, size_t) -> Status { ++tasks; return true; }, "ProbeBorrowTest"));
    } else {
      ++batch;
    }
    return size_t{100};
  };
  LosslessCandidateScreen screen(metadata, input, &runner, encode);
  JXL_TEST_ASSIGN_OR_DIE(auto score, screen.Score(params));
  EXPECT_EQ(borrowed.load(), score.size());
  EXPECT_EQ(tasks.load(), 17 * score.size());
  auto other = params;
  other.colorspace = 6;
  JXL_TEST_ASSIGN_OR_DIE(auto ranked, screen.Rank({{"other", other}}, params));
  EXPECT_EQ(batch.load(), score.size());
  EXPECT_EQ(params.lossless_probe_runner, nullptr);
  EXPECT_EQ(ranked[0].params.lossless_probe_runner, nullptr);
}

TEST(JxlTest, LosslessSampleTrainingProxyDoesNotChangePlans) {
  const char* value = std::getenv("JXL_LOSSLESS_PROBE_TRAINING");
  if (!value || (std::strcmp(value, "1") && std::strcmp(value, "3"))) GTEST_SKIP();
  const float cap = value[0] == '1' ? 0.01f : 0.03f;
  constexpr size_t width = 1025, height = 513;
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(width, height));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  std::vector<uint8_t> pixels(width * height * 3, 23);
  JxlEncoderChunkedFrameAdapter input(width, height, 0);
  ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                  {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}));
  CompressParams p;
  p.SetLossless();
  p.options.predictor = Predictor::Variable;
  p.options.nb_repeats = 0.1f;
  std::atomic<size_t> calls{0};
  const auto encode = [&](const CompressParams& sample, const CodecMetadata&,
                          JxlEncoderChunkedFrameAdapter&) -> StatusOr<size_t> {
    ++calls;
    EXPECT_FLOAT_EQ(sample.options.nb_repeats,
                    sample.options.predictor == Predictor::Zero ? 0.1f : cap);
    return size_t{100};
  };
  ThreadPoolForTests pool(4);
  LosslessCandidateScreen screen(metadata, input, pool.get(), encode);
  auto other = p;
  other.options.nb_repeats = 0.5f;
  JXL_TEST_ASSIGN_OR_DIE(auto ranked, screen.Rank({{"a", p}, {"b", other}}, p));
  EXPECT_EQ(ranked.size(), 2u);
  EXPECT_EQ(calls.load(), 4u);
  EXPECT_FLOAT_EQ(ranked[0].params.options.nb_repeats, 0.1f);
  EXPECT_FLOAT_EQ(ranked[1].params.options.nb_repeats, 0.5f);
  JXL_TEST_ASSIGN_OR_DIE(auto reused, screen.Score(other));
  EXPECT_EQ(calls.load(), 4u);
  EXPECT_EQ(reused, std::vector<size_t>(4, 100));
  p.options.predictor = Predictor::Zero;
  JXL_TEST_ASSIGN_OR_DIE(auto fixed, screen.Score(p));
  EXPECT_EQ(calls.load(), 8u);
}

TEST(JxlTest, LosslessCheapRankingSeparatesScoresAndRetainsAmbiguousFamilies) {
  constexpr size_t width = 1025, height = 513;
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(width, height));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  std::vector<uint8_t> pixels(width * height * 3, 23);
  JxlEncoderChunkedFrameAdapter input(width, height, 0);
  ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                  {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}));
  CompressParams params;
  params.SetLossless();
  params.speed_tier = SpeedTier::kTortoise;
  params.options.nb_repeats = 0.5f;
  std::atomic<size_t> cheap_calls{0}, exact_calls{0};
  const auto exact_size = [](const CompressParams& p) -> size_t {
    return 1000 + 7 * p.colorspace - (p.options.nb_repeats == 1.0f ? 100 : 0);
  };
  const auto encode = [&](const CompressParams& p, const CodecMetadata&,
                          JxlEncoderChunkedFrameAdapter&) -> StatusOr<size_t> {
    EXPECT_EQ(p.lossless_trial_budget, nullptr);
    if (p.speed_tier == SpeedTier::kSquirrel) {
      ++cheap_calls;
      EXPECT_FLOAT_EQ(p.options.nb_repeats, 0.01f);
      return static_cast<size_t>(100 + 10 * p.colorspace);
    }
    ++exact_calls;
    EXPECT_EQ(p.speed_tier, SpeedTier::kTortoise);
    return exact_size(p);
  };
  std::vector<LosslessProbeCandidate> plan;
  for (int i = 0; i < 12; ++i) {
    auto p = params;
    p.colorspace = i;
    p.options.predictor = Predictor::Gradient;
    if (i == 9) p.options.predictor = kUndefinedPredictor;
    if (i == 10) p.options.predictor = Predictor::Zero;
    if (i == 11) p.options.predictor = Predictor::Variable;
    plan.push_back({"test", p});
  }
  auto alias = plan[1];
  alias.params.options.nb_repeats = 1.0f;
  plan.push_back(alias);
  plan.push_back(plan[0]);
  ThreadPoolForTests pool(4);
  LosslessCandidateScreen screen(metadata, input, pool.get(), encode);
  JXL_TEST_ASSIGN_OR_DIE(auto ranked, screen.Rank(plan, params, true));
  ASSERT_EQ(ranked.size(), 7u);
  const char* control_env = std::getenv("JXL_LOSSLESS_CHEAP_PROBE_CONTROL");
  const bool cheap_control = control_env && (std::strcmp(control_env, "1") == 0 ||
      std::strcmp(control_env, "2") == 0);
  EXPECT_EQ(cheap_calls.load(), cheap_control ? 28u : 24u);
  EXPECT_EQ(exact_calls.load(), (cheap_control ? 0u : 4u) + ranked.size() * 4);
  EXPECT_EQ(ranked[0].params.colorspace, 1);
  EXPECT_FLOAT_EQ(ranked[0].params.options.nb_repeats, 1.0f);
  for (const auto& candidate : ranked) {
    EXPECT_EQ(candidate.context_bytes, 4 * exact_size(candidate.params));
    EXPECT_EQ(candidate.params.speed_tier, SpeedTier::kTortoise);
    EXPECT_GE(candidate.params.options.nb_repeats, 0.5f);
  }
  const size_t calls = screen.Encodes();
  JXL_TEST_ASSIGN_OR_DIE(auto repeated, screen.Rank(plan, params, true));
  EXPECT_EQ(screen.Encodes(), calls);
  ASSERT_EQ(repeated.size(), ranked.size());
  for (size_t i = 0; i < ranked.size(); ++i) {
    EXPECT_TRUE(SameLosslessCandidate(repeated[i].params, ranked[i].params));
    EXPECT_EQ(repeated[i].context_bytes, ranked[i].context_bytes);
  }
  JXL_TEST_ASSIGN_OR_DIE(auto resumed, screen.Score(plan[8].params));
  EXPECT_EQ(screen.Encodes(), calls + 4);
  EXPECT_EQ(resumed, std::vector<size_t>(4, exact_size(plan[8].params)));
}

TEST(JxlTest, LosslessStatisticalRaceCachesOnlyCompletedScores) {
  const char* enabled = std::getenv("JXL_LOSSLESS_STATISTICAL_RACE");
  if (!enabled || std::strcmp(enabled, "1") != 0) {
    GTEST_SKIP() << "Run with JXL_LOSSLESS_STATISTICAL_RACE=1";
  }
  constexpr size_t width = 1025, height = 513;
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(width, height));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  std::vector<uint8_t> pixels(width * height * 3, 23);
  JxlEncoderChunkedFrameAdapter input(width, height, 0);
  ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                  {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}));
  CompressParams params;
  params.SetLossless();
  params.speed_tier = SpeedTier::kTortoise;
  params.options.nb_repeats = 0.5f;
  std::atomic<size_t> cheap{0}, exact{0};
  const auto encode = [&](const CompressParams& p, const CodecMetadata&,
                          JxlEncoderChunkedFrameAdapter&) -> StatusOr<size_t> {
    if (p.speed_tier == SpeedTier::kSquirrel) {
      ++cheap;
      return static_cast<size_t>(100 + 10 * p.colorspace);
    }
    ++exact;
    return static_cast<size_t>(1000 + p.colorspace);
  };
  std::vector<LosslessProbeCandidate> plan;
  for (int i = 0; i < 12; ++i) {
    auto p = params;
    p.colorspace = i;
    p.options.predictor = Predictor::Gradient;
    plan.push_back({"statistical-test", p});
  }
  plan.push_back(plan[0]);
  ThreadPoolForTests pool(4);
  LosslessCandidateScreen screen(metadata, input, pool.get(), encode);
  JXL_TEST_ASSIGN_OR_DIE(auto ranked, screen.Rank(plan, params, true));
  ASSERT_EQ(ranked.size(), 2u);
  EXPECT_EQ(cheap.load(), 30u);
  EXPECT_EQ(exact.load(), 12u);
  for (size_t i = 0; i < ranked.size(); ++i) {
    EXPECT_EQ(ranked[i].context_bytes, 4 * (1000 + i));
  }
  const size_t calls = screen.Encodes();
  JXL_TEST_ASSIGN_OR_DIE(auto repeated, screen.Rank(plan, params, true));
  EXPECT_EQ(screen.Encodes(), calls);
  ASSERT_EQ(repeated.size(), ranked.size());
  JXL_TEST_ASSIGN_OR_DIE(auto resumed, screen.Score(plan[3].params));
  EXPECT_EQ(screen.Encodes(), calls + 4);
  EXPECT_EQ(resumed, std::vector<size_t>(4, 1003));
  std::atomic<size_t> ambiguous_cheap{0}, ambiguous_exact{0};
  const auto ambiguous_encode = [&](const CompressParams& p, const CodecMetadata&,
      JxlEncoderChunkedFrameAdapter&) -> StatusOr<size_t> {
    if (p.speed_tier == SpeedTier::kSquirrel) {
      ++ambiguous_cheap;
      return size_t{100};
    }
    ++ambiguous_exact;
    return static_cast<size_t>(1000 + p.colorspace);
  };
  LosslessCandidateScreen ambiguous_screen(metadata, input, pool.get(),
                                           ambiguous_encode);
  JXL_TEST_ASSIGN_OR_DIE(auto ambiguous,
                        ambiguous_screen.Rank(plan, params, true));
  ASSERT_EQ(ambiguous.size(), 12u);
  EXPECT_EQ(ambiguous_cheap.load(), 24u);
  EXPECT_EQ(ambiguous_exact.load(), 52u);
  for (size_t i = 0; i < ambiguous.size(); ++i) {
    EXPECT_EQ(ambiguous[i].context_bytes, 4 * (1000 + i));
  }
}

TEST(JxlTest, LosslessBatchedProbeFailureAndRetry) {
  constexpr size_t width = 1025, height = 513;
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(width, height));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  std::vector<uint8_t> pixels(width * height * 3, 23);
  JxlEncoderChunkedFrameAdapter input(width, height, 0);
  ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                  {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}));
  CompressParams params;
  params.SetLossless();
  auto candidate = params;
  candidate.options.predictor = Predictor::Gradient;
  candidate.colorspace = 1;
  std::vector<LosslessProbeCandidate> plan{{"one", candidate}};
  candidate.colorspace = 2;
  plan.push_back({"two", candidate});
  plan.push_back(plan.front());
  std::atomic<bool> fail{true};
  std::mutex mutex;
  std::map<std::pair<int, uintptr_t>, size_t> completed;
  const auto encode = [&](const CompressParams& p, const CodecMetadata&,
                          JxlEncoderChunkedFrameAdapter& sample) -> StatusOr<size_t> {
    if (p.colorspace == 1 && fail.exchange(false)) {
      return JXL_FAILURE("Intentional probe failure");
    }
    std::lock_guard<std::mutex> lock(mutex);
    ++completed[{p.colorspace, reinterpret_cast<uintptr_t>(&sample)}];
    return size_t{100};
  };
  ThreadPoolForTests pool(4);
  LosslessCandidateScreen screen(metadata, input, pool.get(), encode);
  EXPECT_FALSE(screen.Rank(plan, params).status());
  JXL_TEST_ASSIGN_OR_DIE(auto ranked, screen.Rank(plan, params));
  ASSERT_EQ(ranked.size(), 2u);
  for (const auto& result : ranked) EXPECT_EQ(result.context_bytes, 400u);
  ASSERT_EQ(completed.size(), 12u);
  for (const auto& cell : completed) EXPECT_EQ(cell.second, 1u);
}

TEST(JxlTest, LosslessSmallScreenFailureAndRetry) {
  constexpr size_t width = 513, height = 257;
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(width, height));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  std::vector<uint8_t> pixels(width * height * 3, 17);
  JxlEncoderChunkedFrameAdapter input(width, height, 0);
  ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                  {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}));
  CompressParams params;
  params.SetLossless();
  auto incumbent = params;
  incumbent.palette_colors = 123;
  std::atomic<bool> fail{true};
  std::mutex mutex;
  std::map<std::tuple<int, int, int, uintptr_t>, size_t> completed;
  const auto encode = [&](const CompressParams& p, const CodecMetadata&,
                          JxlEncoderChunkedFrameAdapter& sample) -> StatusOr<size_t> {
    if (p.palette_colors != 123 && fail.exchange(false)) {
      return JXL_FAILURE("Intentional small screen failure");
    }
    std::lock_guard<std::mutex> lock(mutex);
    ++completed[{static_cast<int>(p.options.predictor), p.colorspace,
                 p.palette_colors, reinterpret_cast<uintptr_t>(&sample)}];
    return size_t{p.palette_colors == 123 ? 200u : 100u};
  };
  ThreadPoolForTests pool(8);
  EXPECT_FALSE(ProbeLosslessCandidates(params, metadata, input, pool.get(),
                                        encode, &incumbent).status());
  completed.clear();
  JXL_TEST_ASSIGN_OR_DIE(auto result, ProbeLosslessCandidates(
      params, metadata, input, pool.get(), encode, &incumbent));
  EXPECT_EQ(result.encodes, 90u);
  EXPECT_EQ(result.screen_samples, 3u);
  EXPECT_EQ(result.control_bytes, 1800u);
  ASSERT_EQ(result.ranked.size(), 6u);
  for (const auto& candidate : result.ranked) EXPECT_EQ(candidate.bytes, 900u);
  ASSERT_EQ(completed.size(), 90u);
  for (const auto& cell : completed) EXPECT_EQ(cell.second, 1u);
}

TEST(JxlTest, LosslessProbeFeedbackPreservesRanking) {
  for (const size_t width : {513u, 1025u}) {
    const size_t height = width == 513 ? 513 : 769;
    CodecMetadata metadata;
    ASSERT_TRUE(metadata.size.Set(width, height));
    metadata.m.SetUintSamples(16);
    metadata.m.color_encoding = ColorEncoding::SRGB();
    std::vector<uint16_t> pixels(width * height * 3);
    for (size_t y = 0; y < height; ++y) {
      for (size_t x = 0; x < width; ++x) {
        pixels[3 * (y * width + x)] = x;
        pixels[3 * (y * width + x) + 1] = y;
      }
    }
    JxlEncoderChunkedFrameAdapter input(width, height, 0);
    ASSERT_TRUE(input.SetFromBuffer(
        0, reinterpret_cast<const uint8_t*>(pixels.data()), pixels.size() * 2,
        {3, JXL_TYPE_UINT16, JXL_NATIVE_ENDIAN, 0}));
    CompressParams params;
    params.SetLossless();
    auto incumbent = params;
    incumbent.palette_colors = 123;
    ThreadPoolForTests threaded(8);
    FakeParallelRunner runner(71, 4);
    ThreadPool shuffled(&JxlFakeParallelRunner, &runner);
    for (size_t scenario = 0; scenario < 18; ++scenario) {
      std::atomic<size_t> calls{0};
      const auto encode = [&](const CompressParams& p, const CodecMetadata&,
                              JxlEncoderChunkedFrameAdapter& sample)
          -> StatusOr<size_t> {
        ++calls;
        auto source = sample.GetInputSource();
        size_t stride;
        auto buffer = GetColorBuffer(source, 0, 0, sample.xsize, sample.ysize,
                                     &stride);
        const auto* first = static_cast<const uint16_t*>(buffer.get());
        if (p.palette_colors == 123) return size_t{1000} + first[0] + first[1];
        if (scenario == 0) {
          // Six early completed candidates establish a bound; a late winner
          // must survive, and cross-channel exploratory scores stay complete.
          if (p.options.predictor == Predictor::Top) return size_t{50};
          if (p.palette_colors || p.options.predictor == Predictor::Zero ||
              p.options.predictor == Predictor::Gradient) return size_t{100};
          return size_t{10000};
        }
        if (scenario == 1) {
          // Exact ties, including zero-size suffixes, cannot be pruned at >=.
          return size_t{first[0] > width / 2 && first[1] > height / 2 ? 300u : 0u};
        }
        // Varied deterministic costs, independent of execution order.
        uint32_t hash = scenario * 104729u + first[0] * 73856093u +
                        first[1] * 19349663u + p.palette_colors * 83492791u +
                        static_cast<uint32_t>(p.options.predictor) * 2654435761u +
                        p.colorspace * 123457u + sample.xsize;
        hash ^= hash >> 16;
        return size_t{hash % 3001};
      };
      JXL_TEST_ASSIGN_OR_DIE(auto baseline, ProbeLosslessCandidates(
          params, metadata, input, nullptr, encode, &incumbent, false));
      const auto compare = [](const auto& expected, const auto& actual) {
        ASSERT_EQ(expected.size(), actual.size());
        for (size_t i = 0; i < expected.size(); ++i) {
          EXPECT_STREQ(expected[i].name, actual[i].name);
          EXPECT_TRUE(SameLosslessCandidate(expected[i].params, actual[i].params));
          EXPECT_EQ(expected[i].bytes, actual[i].bytes);
          EXPECT_EQ(expected[i].small_bytes, actual[i].small_bytes);
          EXPECT_EQ(expected[i].exploratory, actual[i].exploratory);
        }
      };
      for (ThreadPool* pool : {static_cast<ThreadPool*>(nullptr), &shuffled,
                               threaded.get()}) {
        SCOPED_TRACE(testing::Message() << "width=" << width
                                       << " scenario=" << scenario);
        calls = 0;
        JXL_TEST_ASSIGN_OR_DIE(auto result, ProbeLosslessCandidates(
            params, metadata, input, pool, encode, &incumbent, true));
        compare(baseline.ranked, result.ranked);
        compare(baseline.exploratory, result.exploratory);
        EXPECT_EQ(result.control_bytes, baseline.control_bytes);
        EXPECT_EQ(result.validation_encodes, baseline.validation_encodes);
        EXPECT_EQ(result.encodes, calls.load());
        EXPECT_EQ(result.encodes + result.screen_saved_encodes, baseline.encodes);
        EXPECT_EQ(result.sample_pixels + result.screen_saved_encodes * 64 * 64,
                  baseline.sample_pixels);
        if (scenario == 0 && !pool) EXPECT_GT(result.screen_saved_encodes, 0u);
        if (scenario == 1) EXPECT_EQ(result.screen_bounded, 0u);
      }
    }
  }
}

TEST(JxlTest, LosslessProbeRacingUsesIncumbentAndValidatesRegions) {
  constexpr size_t width = 513, height = 513;
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(width, height));
  metadata.m.SetUintSamples(16);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  std::vector<uint16_t> pixels(width * height * 3);
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      pixels[3 * (y * width + x)] = x;
      pixels[3 * (y * width + x) + 1] = y;
    }
  }
  JxlEncoderChunkedFrameAdapter input(width, height, 0);
  ASSERT_TRUE(input.SetFromBuffer(
      0, reinterpret_cast<const uint8_t*>(pixels.data()), pixels.size() * 2,
      {3, JXL_TYPE_UINT16, JXL_NATIVE_ENDIAN, 0}));
  CompressParams params;
  params.SetLossless();
  auto incumbent = params;
  incumbent.palette_colors = 123;
  for (int order : {0, 19, 71}) {
    FakeParallelRunner runner(order, 4);
    ThreadPool pool(&JxlFakeParallelRunner, &runner);
    std::atomic<size_t> calls{0}, control_calls{0}, misleading_calls{0};
    const auto probe = [&](const CompressParams& p, const CodecMetadata&,
                           JxlEncoderChunkedFrameAdapter& sample)
        -> StatusOr<size_t> {
      ++calls;
      auto source = sample.GetInputSource();
      size_t stride;
      auto buffer = GetColorBuffer(source, 0, 0, sample.xsize, sample.ysize,
                                   &stride);
      const auto* first = static_cast<const uint16_t*>(buffer.get());
      const size_t tile = 3 * ((first[1] - 74) / 150) + (first[0] - 74) / 150;
      if (p.palette_colors == 123) {
        ++control_calls;
        return size_t{100 * (tile + 1)};
      }
      if (p.options.predictor == Predictor::Variable) {
        if (p.colorspace == 6) return size_t{200};
        ++misleading_calls;
        // Wins the representative tiles, but loses on the remaining regions.
        return size_t{tile == 1 || tile == 4 || tile == 7 ? 50u : 2000u};
      }
      if (p.colorspace == 0 && p.options.predictor == Predictor::Gradient)
        return size_t{250};
      return size_t{1000};
    };
    JXL_TEST_ASSIGN_OR_DIE(
        auto result,
        ProbeLosslessCandidates(params, metadata, input,
                                 order == 0 ? nullptr : &pool, probe, &incumbent));
    EXPECT_EQ(control_calls.load(), 9u);
    EXPECT_EQ(result.samples, 9u);
    EXPECT_EQ(result.screen_samples, 3u);
    EXPECT_EQ(result.candidates, 15u);
    EXPECT_EQ(result.shortlisted, 6u);
    EXPECT_EQ(result.control_bytes, 4500u);
    EXPECT_EQ(result.encodes, calls.load());
    EXPECT_LT(result.encodes, 90u);
    EXPECT_GT(result.bounded, 0u);
    EXPECT_LT(misleading_calls.load(), 9u);
    ASSERT_FALSE(result.ranked.empty());
    EXPECT_STREQ(result.ranked.front().name, "ycocg-cross-channel");
    EXPECT_EQ(result.ranked.front().bytes, 1800u);
    for (const auto& candidate : result.ranked) {
      EXPECT_STRNE(candidate.name, "raw-cross-channel");
    }
  }
}

TEST(JxlTest, LosslessProbeValidatesLargerContext) {
  constexpr size_t width = 1025, height = 769;
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(width, height));
  metadata.m.SetUintSamples(16);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  metadata.m.SetAlphaBits(16);
  std::vector<uint16_t> pixels(width * height * 4);
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      auto* p = pixels.data() + 4 * (y * width + x);
      p[0] = x;
      p[1] = y;
      p[2] = x + y;
      p[3] = x < 700 ? 0 : 12345;
    }
  }
  JxlEncoderChunkedFrameAdapter input(width, height, 1);
  ASSERT_TRUE(input.SetFromBuffer(
      0, reinterpret_cast<const uint8_t*>(pixels.data()), pixels.size() * 2,
      {4, JXL_TYPE_UINT16, JXL_NATIVE_ENDIAN, 0}));
  CompressParams params;
  params.SetLossless();
  auto incumbent = params;
  incumbent.palette_colors = 123;
  std::atomic<size_t> calls{0}, large_calls{0};
  const auto probe = [&](const CompressParams& p, const CodecMetadata& m,
                         JxlEncoderChunkedFrameAdapter& sample)
      -> StatusOr<size_t> {
    ++calls;
    const bool large = sample.xsize == 384;
    EXPECT_TRUE(large || sample.xsize == 64);
    EXPECT_EQ(m.xsize(), sample.xsize);
    EXPECT_EQ(m.m.bit_depth.bits_per_sample, 16u);
    if (large) ++large_calls;
    auto source = sample.GetInputSource();
    size_t stride;
    auto buffer = GetColorBuffer(source, 0, 0, sample.xsize, sample.ysize,
                                 &stride);
    const auto* first = static_cast<const uint16_t*>(buffer.get());
    EXPECT_EQ(first[2], first[0] + first[1]);
    EXPECT_EQ(first[3], first[0] < 700 ? 0 : 12345);
    if (p.palette_colors == 123) return size_t{large ? 8000u : 1000u};
    if (p.options.predictor == Predictor::Gradient && p.colorspace == 0)
      // Both scales lose, even though both are within the screening margin.
      return size_t{large ? 8040u : 1010u};
    if (p.options.predictor == Predictor::Variable) {
      if (p.colorspace == 6) return size_t{large ? 7000u : 200u};
      // Preserve a plausible alternative when the two scales disagree.
      return size_t{large ? 8100u : 300u};
    }
    return size_t{2000};
  };
  ThreadPoolForTests pool(4);
  JXL_TEST_ASSIGN_OR_DIE(
      auto result, ProbeLosslessCandidates(params, metadata, input, pool.get(),
                                            probe, &incumbent));
  EXPECT_EQ(result.encodes, calls.load());
  EXPECT_EQ(result.validation_encodes, 4u);
  EXPECT_EQ(large_calls.load(), 4u);
  EXPECT_LE(result.encodes, 94u);
  EXPECT_EQ(result.sample_pixels,
            (result.encodes - 4) * 64 * 64 + 4 * 384 * 384);
  EXPECT_EQ(result.control_bytes, 8000u);
  ASSERT_EQ(result.ranked.size(), 2u);
  EXPECT_STREQ(result.ranked[0].name, "ycocg-cross-channel");
  EXPECT_EQ(result.ranked[0].bytes, 7000u);
  EXPECT_STREQ(result.ranked[1].name, "raw-cross-channel");

}

TEST(JxlTest, LosslessCalibrationRejectsCropOptimismAndRetainsLocalWins) {
  const char* value = std::getenv("JXL_LOSSLESS_CALIBRATE");
  if (!value || std::strcmp(value, "1") != 0) GTEST_SKIP();
  const char* fast = std::getenv("JXL_LOSSLESS_CALIBRATE_FAST");
  const int proxy_effort = fast && std::strcmp(fast, "1") == 0 ? 4 : 6;
  auto* memory = test::MemoryManager();
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(1025, 769));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  std::vector<uint8_t> pixels(1025 * 769 * 3, 17);
  JxlEncoderChunkedFrameAdapter input(1025, 769, 0);
  ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                  {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}));
  ThreadPoolForTests pool(4);
  for (int scenario = 0; scenario < 6; ++scenario) {
    for (ThreadPool* runner : {static_cast<ThreadPool*>(nullptr), pool.get()}) {
      size_t previous = std::numeric_limits<size_t>::max();
      for (int effort = 3; effort <= 10; ++effort) {
        SCOPED_TRACE(testing::Message() << "scenario=" << scenario
                                       << " effort=" << effort);
        CompressParams params;
        params.SetLossless();
        params.speed_tier = static_cast<SpeedTier>(10 - effort);
        std::atomic<size_t> cheap_calls{0}, deep_calls{0};
        const auto probe = [&](const CompressParams& p, const CodecMetadata&,
                               JxlEncoderChunkedFrameAdapter& sample) -> StatusOr<size_t> {
          if (p.options.predictor == Predictor::Variable && p.palette_colors == 0) {
            // The last case rejects the deep model but keeps the cheap winner,
            // including when the regional correction is greater than one.
            if (LosslessSearchLevel(p) <= 6) return size_t{scenario == 5 ? 10u : 100u};
            if (scenario == 5) return size_t{20};
            if (scenario == 1 || (scenario == 2 && sample.xsize == 256)) return size_t{20};
            return size_t{90};
          }
          // A globally repetitive frame can be smaller than the crop sum.
          const size_t scale = scenario == 4 ? 4 : 1;
          return scale * size_t{p.palette_colors == 10000 ? 100u : 1000u};
        };
        const auto encode = [&](const CompressParams& p, ThreadPool*,
                                JxlEncoderOutputProcessorWrapper* out,
                                AuxOut*) -> Status {
          size_t bytes = p.options.predictor == Predictor::Zero ? 800 : 900;
          if (p.options.predictor == Predictor::Variable && p.palette_colors == 0) {
            if (LosslessSearchLevel(p) <= 6) {
              ++cheap_calls;
              EXPECT_EQ(LosslessSearchLevel(p), proxy_effort);
              EXPECT_EQ(p.lossless_trial_budget, nullptr);
              bytes = scenario == 3 || scenario == 5 ? 700 : 1600;
            } else {
              ++deep_calls;
              bytes = scenario == 0 ? 2000 : scenario == 3 ? 900 : 400;
            }
          }
          return AppendData(*out, std::vector<uint8_t>(bytes, 0));
        };
        JxlEncoderOutputProcessorWrapper output(memory);
        JXL_TEST_ASSIGN_OR_DIE(bool searched, EncodeLosslessSearch(
            memory, params, FrameInfo{}, metadata, input, runner, &output,
            nullptr, encode, probe));
        ASSERT_TRUE(searched);
        ASSERT_TRUE(output.SetFinalizedPosition());
        std::vector<uint8_t> bytes;
        ASSERT_TRUE(output.CopyOutput(bytes));
        EXPECT_LE(bytes.size(), previous);
        previous = bytes.size();
        if (effort >= 4) {
          if (scenario == 4) {
            EXPECT_EQ(cheap_calls.load(), 0u);
          } else {
            EXPECT_GT(cheap_calls.load(), 0u);
          }
          EXPECT_LE(cheap_calls.load(), 2u);
          if (scenario == 0 || scenario == 5) {
            EXPECT_EQ(deep_calls.load(), 0u);
            EXPECT_EQ(bytes.size(), scenario == 5 ? 700u : 800u);
          } else {
            EXPECT_GT(deep_calls.load(), 0u);
            EXPECT_EQ(bytes.size(), scenario == 3 ? 700u : 400u);
          }
        }
      }
    }
  }
}

TEST(JxlTest, LosslessCalibrationDefersExpensiveRefinementsWithoutLosingE10) {
  for (const char* key : {"JXL_LOSSLESS_CALIBRATE",
                          "JXL_LOSSLESS_CALIBRATE_REFINEMENTS"}) {
    const char* value = std::getenv(key);
    if (!value || std::strcmp(value, "1") != 0) GTEST_SKIP();
  }
  auto* memory = test::MemoryManager();
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(1025, 769));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  std::vector<uint8_t> pixels(1025 * 769 * 3, 17);
  JxlEncoderChunkedFrameAdapter input(1025, 769, 0);
  ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                  {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}));
  ThreadPoolForTests pool(4);
  // A losing refinement, an underestimated winner recovered at e10, and a
  // strong calibrated winner that must still run at ordinary effort.
  for (int scenario = 0; scenario < 3; ++scenario) {
    for (ThreadPool* runner : {static_cast<ThreadPool*>(nullptr), pool.get()}) {
      size_t previous = std::numeric_limits<size_t>::max();
      for (int effort : {3, 5, 7, 9, 10}) {
        SCOPED_TRACE(testing::Message() << "scenario=" << scenario
                                       << " effort=" << effort);
        CompressParams params;
        params.SetLossless();
        params.speed_tier = static_cast<SpeedTier>(10 - effort);
        size_t cheap_calls = 0, expensive_calls = 0;
        const auto probe = [&](const CompressParams& p, const CodecMetadata&,
                               JxlEncoderChunkedFrameAdapter&) -> StatusOr<size_t> {
          if (p.options.predictor != kUndefinedPredictor) return size_t{500};
          const int backend = LosslessSearchLevel(p);
          if (backend <= 6) return size_t{scenario == 2 ? 200u : 100u};
          if (backend == 9 && p.options.nb_repeats >= 0.5f) {
            return size_t{scenario == 2 ? 10u : 90u};
          }
          return size_t{100};
        };
        const auto encode = [&](const CompressParams& p, ThreadPool*,
                                JxlEncoderOutputProcessorWrapper* out,
                                AuxOut*) -> Status {
          size_t size = 3000;
          if (p.options.predictor == kUndefinedPredictor) {
            const int backend = LosslessSearchLevel(p);
            size = backend == 3 ? 1000 : 900;
            if (backend <= 6 && p.options.nb_repeats <= 0.01f) {
              ++cheap_calls;
              EXPECT_EQ(p.lossless_trial_budget, nullptr);
              size = scenario == 2 ? 1000 : 1300;
            }
            if (backend == 9) {
              size = 700;
              if (p.options.nb_repeats >= 0.5f) {
                ++expensive_calls;
                size = scenario == 0 ? 800 : 600;
              }
            }
          }
          return AppendData(*out, std::vector<uint8_t>(size, 0));
        };
        JxlEncoderOutputProcessorWrapper output(memory);
        JXL_TEST_ASSIGN_OR_DIE(bool searched, EncodeLosslessSearch(
            memory, params, FrameInfo{}, metadata, input, runner, &output,
            nullptr, encode, probe));
        ASSERT_TRUE(searched);
        ASSERT_TRUE(output.SetFinalizedPosition());
        std::vector<uint8_t> bytes;
        ASSERT_TRUE(output.CopyOutput(bytes));
        EXPECT_LE(bytes.size(), previous);
        previous = bytes.size();
        const bool run_expensive = effort == 10 || (scenario == 2 && effort >= 7);
        EXPECT_EQ(expensive_calls, run_expensive ? 1u : 0u);
        EXPECT_EQ(cheap_calls, effort >= 7 ? 1u : 0u);
        const size_t expected = effort == 3 ? 900 :
            (run_expensive && scenario != 0 ? 600 : 700);
        EXPECT_EQ(bytes.size(), expected);
      }
    }
  }
}

TEST(JxlTest, LosslessRouteCalibrationDefersAndRecoversCandidates) {
  for (const char* key : {"JXL_LOSSLESS_CALIBRATE",
                          "JXL_LOSSLESS_CALIBRATE_ROUTES"}) {
    const char* value = std::getenv(key);
    if (!value || std::strcmp(value, "1") != 0) GTEST_SKIP();
  }
  auto* memory = test::MemoryManager();
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(1025, 769));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  std::vector<uint8_t> pixels(1025 * 769 * 3, 17);
  // Keep all three scan strata active; a uniform frame uses one fallback crop.
  for (size_t y = 0; y < 769; ++y) {
    for (size_t x = 0; x < 1025; ++x) {
      for (size_t c = 0; c < 3; ++c)
        pixels[3 * (y * 1025 + x) + c] = (x / 8 + y / 8) % 5;
    }
  }
  JxlEncoderChunkedFrameAdapter input(1025, 769, 0);
  ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                  {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}));
  ThreadPoolForTests pool(4);
  for (bool geometry : {false, true}) {
    // Losing proxy, false negative recovered at e10, and a strong local win.
    for (int scenario = 0; scenario < (geometry ? 4 : 3); ++scenario) {
      for (ThreadPool* runner : {static_cast<ThreadPool*>(nullptr), pool.get()}) {
        size_t previous = std::numeric_limits<size_t>::max();
        for (int effort : {3, 9, 10}) {
          SCOPED_TRACE(testing::Message() << "geometry=" << geometry
              << " scenario=" << scenario << " effort=" << effort);
          CompressParams params;
          params.SetLossless();
          params.speed_tier = static_cast<SpeedTier>(10 - effort);
          size_t cheap_calls = 0, deep_calls = 0;
          const auto probe = [&](const CompressParams& p, const CodecMetadata&,
                                 JxlEncoderChunkedFrameAdapter& sample) -> StatusOr<size_t> {
            const bool raw = p.options.predictor == Predictor::Variable &&
                             p.palette_colors == 0;
            const bool palette = p.options.predictor == Predictor::Zero &&
                                 p.palette_colors == 10000;
            if (raw) {
              if (LosslessSearchLevel(p) <= 6) {
                return size_t{geometry && scenario == 3 &&
                    p.modular_group_size_shift == 2 ? 90u : 100u};
              }
              if (geometry && p.modular_group_size_shift == 2)
                return size_t{scenario == 2 ? 50u : 79u};
              return size_t{p.colorspace == 6 ? 80u : 90u};
            }
            if (palette) {
              if (geometry || p.options.nb_repeats == 0 ||
                  LosslessSearchLevel(p) <= 6) return size_t{100};
              return size_t{scenario == 2 ? 10u :
                           (sample.xsize == 256 ? 70u : 90u)};
            }
            return size_t{1000};
          };
          const auto encode = [&](const CompressParams& p, ThreadPool*,
                                  JxlEncoderOutputProcessorWrapper* out,
                                  AuxOut*) -> Status {
            size_t bytes = 900;
            const bool raw = p.options.predictor == Predictor::Variable &&
                             p.palette_colors == 0;
            const bool palette = p.options.predictor == Predictor::Zero &&
                                 p.palette_colors == 10000;
            if (palette && p.options.nb_repeats == 0) bytes = 800;
            if (!geometry && palette && p.options.nb_repeats > 0) {
              if (LosslessSearchLevel(p) <= 6) {
                ++cheap_calls;
                EXPECT_EQ(p.lossless_trial_budget, nullptr);
                bytes = 1300;
              } else {
                ++deep_calls;
                bytes = scenario == 0 ? 550 : 400;
              }
            }
            if (raw) {
              if (LosslessSearchLevel(p) <= 6) {
                EXPECT_EQ(p.lossless_trial_budget, nullptr);
                bytes = p.modular_group_size_shift == 2
                            ? (scenario == 3 ? 995 : 1100) : 1000;
                if (geometry && p.modular_group_size_shift == 2) ++cheap_calls;
              } else {
                bytes = p.colorspace == 6 ? 500 : 600;
                if (geometry && p.colorspace == 6 && p.modular_group_size_shift == 2) {
                  ++deep_calls;
                  bytes = scenario == 0 ? 550 : 400;
                }
              }
            }
            return AppendData(*out, std::vector<uint8_t>(bytes, 0));
          };
          JxlEncoderOutputProcessorWrapper output(memory);
          JXL_TEST_ASSIGN_OR_DIE(bool searched, EncodeLosslessSearch(
              memory, params, FrameInfo{}, metadata, input, runner, &output,
              nullptr, encode, probe));
          ASSERT_TRUE(searched);
          ASSERT_TRUE(output.SetFinalizedPosition());
          std::vector<uint8_t> bytes;
          ASSERT_TRUE(output.CopyOutput(bytes));
          EXPECT_LE(bytes.size(), previous);
          previous = bytes.size();
          if (effort >= 9) {
            if (geometry || scenario != 2) EXPECT_GT(cheap_calls, 0u);
            if (effort == 9 && scenario != 2) {
              EXPECT_EQ(deep_calls, 0u);
            } else {
              EXPECT_GT(deep_calls, 0u);
            }
            const bool improvement = scenario != 0 &&
                                     (effort == 10 || scenario == 2);
            EXPECT_EQ(bytes.size(), improvement ? 400u : 500u);
          }
        }
      }
    }
  }
}

TEST(JxlTest, LosslessBackendAndMediumFeedbackUseMeasuredAnchors) {
  for (const char* key : {"JXL_LOSSLESS_CALIBRATE",
                          "JXL_LOSSLESS_CALIBRATE_ROUTES"}) {
    const char* value = std::getenv(key);
    if (!value || std::strcmp(value, "1") != 0) GTEST_SKIP();
  }
  auto* memory = test::MemoryManager();
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(1025, 769));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  std::vector<uint8_t> pixels(1025 * 769 * 3, 17);
  JxlEncoderChunkedFrameAdapter input(1025, 769, 0);
  ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                  {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}));
  for (bool medium : {false, true}) {
    for (int scenario = 0; scenario < 3; ++scenario) {
      size_t previous = std::numeric_limits<size_t>::max();
      for (int effort : {3, 9, 10}) {
        SCOPED_TRACE(testing::Message() << "medium=" << medium
            << " scenario=" << scenario << " effort=" << effort);
        CompressParams params;
        params.SetLossless();
        params.speed_tier = static_cast<SpeedTier>(10 - effort);
        size_t deep_calls = 0, medium_calls = 0;
        const auto probe = [&](const CompressParams& p, const CodecMetadata&,
                               JxlEncoderChunkedFrameAdapter&) -> StatusOr<size_t> {
          const int backend = LosslessSearchLevel(p);
          if (p.options.predictor == kUndefinedPredictor) {
            if (p.modular_group_size_shift == 2) return size_t{103};
            if (backend == 9) return size_t{medium ? 120u :
                                                   scenario == 2 ? 50u : 90u};
            return size_t{100};
          }
          if (medium && p.options.predictor == Predictor::Variable &&
              p.palette_colors == 0) return size_t{backend <= 7 ? 100u : 90u};
          return size_t{1000};
        };
        const auto encode = [&](const CompressParams& p, ThreadPool*,
                                JxlEncoderOutputProcessorWrapper* out,
                                AuxOut*) -> Status {
          const int backend = LosslessSearchLevel(p);
          size_t bytes = 2000;
          if (p.options.predictor == kUndefinedPredictor) {
            bytes = backend == 3 ? 1500 : 900;
            if (backend == 9 && p.modular_group_size_shift == 2) bytes = 700;
            if (!medium && backend == 9 && p.options.nb_repeats == 0.5f) {
              ++deep_calls;
              bytes = scenario == 0 ? 800 : 600;
            }
          }
          if (medium && p.options.predictor == Predictor::Variable &&
              p.palette_colors == 0) {
            bytes = 850;
            if (backend == 7) {
              ++medium_calls;
              EXPECT_EQ(p.lossless_trial_budget, nullptr);
              bytes = scenario == 2 ? 650 : 900;
            } else if (backend >= 9) {
              ++deep_calls;
              bytes = scenario == 0 ? 800 : 600;
            }
          }
          return AppendData(*out, std::vector<uint8_t>(bytes, 0));
        };
        JxlEncoderOutputProcessorWrapper output(memory);
        JXL_TEST_ASSIGN_OR_DIE(bool searched, EncodeLosslessSearch(
            memory, params, FrameInfo{}, metadata, input, nullptr, &output,
            nullptr, encode, probe));
        ASSERT_TRUE(searched);
        ASSERT_TRUE(output.SetFinalizedPosition());
        std::vector<uint8_t> bytes;
        ASSERT_TRUE(output.CopyOutput(bytes));
        EXPECT_LE(bytes.size(), previous);
        previous = bytes.size();
        if (effort >= 9) {
          if (medium) EXPECT_GT(medium_calls, 0u);
          if (effort == 9 && scenario != 2) EXPECT_EQ(deep_calls, 0u);
          else EXPECT_GT(deep_calls, 0u);
          const bool improvement = scenario != 0 &&
                                   (scenario == 2 || effort == 10);
          EXPECT_EQ(bytes.size(), improvement ? 600u : 700u);
        }
      }
    }
  }
}

TEST(JxlTest, LosslessCalibrationReuseDecisions) {
  for (const char* key : {"JXL_LOSSLESS_CALIBRATE",
                          "JXL_LOSSLESS_CALIBRATE_ROUTES",
                          "JXL_LOSSLESS_PREFLIGHT_YIELD",
                          "JXL_LOSSLESS_CALIBRATION_REUSE"}) {
    const char* value = std::getenv(key);
    if (!value || std::strcmp(value, "1") != 0) GTEST_SKIP();
  }
  auto* memory = test::MemoryManager();
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(2049, 1025));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  std::vector<uint8_t> pixels(2049 * 1025 * 3, 17);
  JxlEncoderChunkedFrameAdapter input(2049, 1025, 0);
  ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                  {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}));
  for (int scenario : {0, 1, 2}) {
    size_t previous = std::numeric_limits<size_t>::max();
    for (int effort : {3, 4, 5, 7, 9, 10}) {
      SCOPED_TRACE(testing::Message() << scenario << " effort=" << effort);
      CompressParams params;
      params.SetLossless();
      params.speed_tier = static_cast<SpeedTier>(10 - effort);
      size_t automatic_deep = 0;
      size_t rejected_exact_probes = 0;
      const auto probe = [&](const CompressParams& p, const CodecMetadata&,
                             JxlEncoderChunkedFrameAdapter&) -> StatusOr<size_t> {
        const int backend = LosslessSearchLevel(p);
        if (p.options.predictor == Predictor::Left ||
            p.options.predictor == Predictor::Top) {
          if (backend >= 9) ++rejected_exact_probes;
          return size_t{100000};
        }
        if (scenario == 0) {
          if (p.options.predictor != kUndefinedPredictor) return size_t{1000};
          return size_t{backend >= 9 && p.modular_group_size_shift == 2 ? 104u : 100u};
        }
        if (p.options.predictor == kUndefinedPredictor) return size_t{500};
        if (p.options.predictor == Predictor::Variable && p.palette_colors == 0)
          return size_t{backend >= 9 ? 100u : 140u};
        return size_t{1000};
      };
      const auto encode = [&](const CompressParams& p, ThreadPool*,
                              JxlEncoderOutputProcessorWrapper* out,
                              AuxOut*) -> Status {
        const int backend = LosslessSearchLevel(p);
        size_t bytes = scenario == 0 ? 30000 : 11500;
        if (p.options.predictor == kUndefinedPredictor) {
          if (scenario == 0) {
            bytes = backend <= 3 ? 15000 : backend == 4 ? 12000 :
                    backend < 9 ? 10000 : 9000;
            if (p.modular_group_size_shift == 2)
              bytes = backend == 4 ? 11880 : backend >= 9 ? 7500 : 10000;
          } else {
            bytes = backend <= 3 ? 15000 : backend <= 7 ? 10000 :
                    backend == 8 ? 9500 : 7000;
          }
          if (backend == 9 && p.modular_group_size_shift == -1 &&
              p.options.nb_repeats == 0.5f) ++automatic_deep;
        } else if (scenario != 0 && p.options.predictor == Predictor::Variable &&
                   p.palette_colors == 0) {
          bytes = backend < 9 ? 10000 : scenario == 1 ? 6000 : 11000;
        }
        return AppendData(*out, std::vector<uint8_t>(bytes, 0));
      };
      JxlEncoderOutputProcessorWrapper output(memory);
      JXL_TEST_ASSIGN_OR_DIE(bool searched, EncodeLosslessSearch(
          memory, params, FrameInfo{}, metadata, input, nullptr, &output,
          nullptr, encode, probe));
      ASSERT_TRUE(searched);
      ASSERT_TRUE(output.SetFinalizedPosition());
      std::vector<uint8_t> bytes;
      ASSERT_TRUE(output.CopyOutput(bytes));
      EXPECT_LE(bytes.size(), previous);
      previous = bytes.size();
      EXPECT_EQ(rejected_exact_probes, 0u);
      if (effort >= 9) {
        EXPECT_EQ(bytes.size(), scenario == 0 ? 7500u : scenario == 1 ? 6000u : 7000u);
        if (scenario == 1 && effort == 9) EXPECT_EQ(automatic_deep, 0u);
        if (scenario == 1 && effort == 10) EXPECT_GT(automatic_deep, 0u);
        if (scenario == 2) EXPECT_GT(automatic_deep, 0u);
      }
    }
  }
}

TEST(JxlTest, LosslessEarlyFamilyUsesMeasuredWinnerAndKeepsFallback) {
  for (const char* key : {"JXL_LOSSLESS_CALIBRATE",
                          "JXL_LOSSLESS_CALIBRATE_FAST",
                          "JXL_LOSSLESS_CALIBRATE_ROUTES",
                          "JXL_LOSSLESS_PREFLIGHT_YIELD",
                          "JXL_LOSSLESS_CALIBRATION_REUSE",
                          "JXL_LOSSLESS_EARLY_FAMILY"}) {
    const char* value = std::getenv(key);
    if (!value || std::strcmp(value, "1") != 0) GTEST_SKIP();
  }
  auto* memory = test::MemoryManager();
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(2049, 1025));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  std::vector<uint8_t> pixels(2049 * 1025 * 3, 17);
  JxlEncoderChunkedFrameAdapter input(2049, 1025, 0);
  ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                  {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}));
  // A real early win, a misleading proxy, and a weak proxy deferred until e8.
  for (int scenario : {0, 1, 2}) {
    size_t previous = std::numeric_limits<size_t>::max();
    for (int effort : {3, 4, 5, 7, 8, 9, 10}) {
      SCOPED_TRACE(testing::Message() << scenario << " effort=" << effort);
      CompressParams params;
      params.SetLossless();
      params.speed_tier = static_cast<SpeedTier>(10 - effort);
      size_t deep_raw = 0, deep_geometry = 0;
      const auto probe = [&](const CompressParams& p, const CodecMetadata&,
                             JxlEncoderChunkedFrameAdapter&) -> StatusOr<size_t> {
        if (p.options.predictor == kUndefinedPredictor) return size_t{100};
        if (p.palette_colors == 0 && p.options.predictor == Predictor::Variable)
          return size_t{LosslessSearchLevel(p) >= 9 ? (scenario == 2 ? 90u : 70u) : 100u};
        return size_t{1000};
      };
      const auto encode = [&](const CompressParams& p, ThreadPool*,
                              JxlEncoderOutputProcessorWrapper* out,
                              AuxOut*) -> Status {
        const int backend = LosslessSearchLevel(p);
        size_t bytes = 30000;
        if (p.options.predictor == kUndefinedPredictor) {
          bytes = backend <= 3 ? 15000 : 10000;
          if (p.modular_group_size_shift == 2 && backend >= 9) {
            ++deep_geometry;
            bytes = scenario == 2 ? 9900 : 8000;
          }
        } else if (p.palette_colors == 0 &&
                   p.options.predictor == Predictor::Variable) {
          bytes = scenario == 2 ? 11000 : 13000;
          if (backend >= 9) {
            ++deep_raw;
            bytes = scenario == 1 ? 12000 : 6000;
          }
        }
        return AppendData(*out, std::vector<uint8_t>(bytes, 0));
      };
      JxlEncoderOutputProcessorWrapper output(memory);
      JXL_TEST_ASSIGN_OR_DIE(bool searched, EncodeLosslessSearch(
          memory, params, FrameInfo{}, metadata, input, nullptr, &output,
          nullptr, encode, probe));
      ASSERT_TRUE(searched);
      ASSERT_TRUE(output.SetFinalizedPosition());
      std::vector<uint8_t> bytes;
      ASSERT_TRUE(output.CopyOutput(bytes));
      EXPECT_LE(bytes.size(), previous);
      previous = bytes.size();
      if (effort == 5) {
        EXPECT_EQ(bytes.size(), scenario == 0 ? 6000u : scenario == 1 ? 8000u : 9900u);
        if (scenario == 2) EXPECT_EQ(deep_raw, 0u);
        else EXPECT_GT(deep_raw, 0u);
        if (scenario == 0) EXPECT_EQ(deep_geometry, 0u);
        else EXPECT_GT(deep_geometry, 0u);
      }
      if (effort >= 8) {
        EXPECT_EQ(bytes.size(), scenario == 1 ? 8000u : 6000u);
        EXPECT_GT(deep_raw, 0u);
      }
    }
  }
}

TEST(JxlTest, CooperativeProbePoolNestedScratchAndFailures) {
  for (int threads : {0, 1, 4, 8}) {
    test::ThreadPoolForTests caller(threads);
    for (size_t probes : {0u, 1u, 7u, 31u}) {
      SCOPED_TRACE(testing::Message() << threads << ":" << probes);
      LosslessProbePool arena;
      std::vector<size_t> sums(probes);
      ASSERT_TRUE(arena.Run(probes, caller.get(), 8, 4,
          [&](size_t probe, ThreadPool* inner) -> Status {
            std::vector<size_t> scratch;
            std::atomic<size_t> sum{0};
            JXL_RETURN_IF_ERROR(RunOnPool(inner, 0, 19,
                [&](size_t workers) -> Status { scratch.resize(workers); return true; },
                [&](uint32_t item, size_t thread) -> Status {
                  scratch[thread] = item + 1;
                  JXL_RETURN_IF_ERROR(RunOnPool(inner, 0, 13, ThreadPool::NoInit,
                      [&](uint32_t child, size_t) -> Status {
                        sum.fetch_add((probe + 1) * (item + 1) + child,
                                      std::memory_order_relaxed);
                        return true;
                      }, "CooperativeNested"));
                  JXL_ENSURE(scratch[thread] == item + 1);
                  return true;
                }, "CooperativeParent"));
            sums[probe] = sum.load();
            return true;
          }));
      for (size_t i = 0; i < probes; ++i)
        EXPECT_EQ(sums[i], (i + 1) * 190 * 13 + 19 * 78);
    }
    for (bool fail_init : {false, true}) {
      LosslessProbePool arena;
      EXPECT_FALSE(arena.Run(9, caller.get(), 8, 3,
          [&](size_t probe, ThreadPool* inner) -> Status {
            return RunOnPool(inner, 0, 17,
                [&](size_t) -> Status { return !(fail_init && probe == 2); },
                [&](uint32_t item, size_t) -> Status {
                  return fail_init || probe != 2 || item != 4;
                }, "CooperativeFailure");
          }));
      // The same arena is reusable after all failed work has joined.
      EXPECT_TRUE(arena.Run(3, caller.get(), 8, 3,
          [](size_t, ThreadPool*) -> Status { return true; }));
    }
  }
  ThreadPool rejected([](void*, void*, JxlParallelRunInit,
                         JxlParallelRunFunction, uint32_t, uint32_t) {
    return JXL_PARALLEL_RET_RUNNER_ERROR;
  }, nullptr);
  LosslessProbePool arena;
  EXPECT_FALSE(arena.Run(1, &rejected, 1, 1,
      [](size_t, ThreadPool*) -> Status { return true; }));
}

TEST(JxlTest, CooperativeProbePoolSingletonChildren) {
  for (int threads : {0, 1, 4}) {
    test::ThreadPoolForTests caller(threads);
    LosslessProbePool arena;
    std::atomic<size_t> calls{0};
    ASSERT_TRUE(arena.Run(11, caller.get(), 4, 3,
        [&](size_t, ThreadPool* inner) -> Status {
          size_t slots = 0;
          return RunOnPool(inner, 0, 1,
              [&](size_t workers) -> Status { slots = workers; return true; },
              [&](uint32_t task, size_t thread) -> Status {
                JXL_ENSURE(task == 0 && thread < slots);
                calls.fetch_add(1, std::memory_order_relaxed);
                return RunOnPool(inner, 0, 1, ThreadPool::NoInit,
                    [](uint32_t, size_t) -> Status { return true; }, "SingletonChild");
              }, "SingletonParent");
        }));
    EXPECT_EQ(calls.load(), 11u);
    EXPECT_FALSE(arena.Run(3, caller.get(), 4, 2,
        [](size_t, ThreadPool* inner) -> Status {
          return RunOnPool(inner, 0, 1, ThreadPool::NoInit,
              [](uint32_t, size_t) -> Status { return false; }, "SingletonFailure");
        }));
  }
}

TEST(JxlTest, ContinuousToneGateRejectsFlatOrLowColorSamples) {
  LosslessScan scan;
  EXPECT_FALSE(scan.LikelyContinuousTone());
  scan.sampled_pixels = 1024;
  scan.sampled_colors = 1024;
  EXPECT_TRUE(scan.LikelyContinuousTone());
  scan.sampled_colors = 512;
  EXPECT_FALSE(scan.LikelyContinuousTone());
  scan.sampled_colors = 513;
  scan.sampled_flat = 512;
  EXPECT_FALSE(scan.LikelyContinuousTone());
  scan.sampled_flat = 511;
  EXPECT_TRUE(scan.LikelyContinuousTone());
  scan.sampled_flat = 896;
  scan.sampled_colors = 96;
  EXPECT_FALSE(scan.LikelySparseContinuousTone());
  scan.sampled_colors = 97;
  EXPECT_TRUE(scan.LikelySparseContinuousTone());
  scan.sampled_flat = 961;
  EXPECT_FALSE(scan.LikelySparseContinuousTone());
  scan.sampled_flat = 1025;
  EXPECT_FALSE(scan.LikelySparseContinuousTone());
}

TEST(JxlTest, CompactBandAtlasKeepsFlatHeavyContext) {
  LosslessScan scan;
  scan.pixels = 640 * 960;
  EXPECT_FALSE(scan.PreferCompactBandAtlas());
  scan.sampled_pixels = 1024;
  scan.sampled_flat = 400;
  EXPECT_TRUE(scan.PreferCompactBandAtlas());
  scan.sampled_flat = 512;
  EXPECT_FALSE(scan.PreferCompactBandAtlas());
  scan.sampled_flat = 0;
  scan.pixels = 1u << 18;
  EXPECT_FALSE(scan.PreferCompactBandAtlas());
  scan.pixels = 2u << 20;
  EXPECT_FALSE(scan.PreferCompactBandAtlas());
}

TEST(JxlTest, DenseTextureRequiresDistributedVariation) {
  LosslessScan scan;
  EXPECT_FALSE(scan.LikelyDenseTexture());
  scan.sampled_pixels = 1024;
  scan.sampled_flat = 204;
  scan.sampled_colors = 248;
  EXPECT_TRUE(scan.LikelyDenseTexture());
  scan.sampled_colors = 128;
  EXPECT_FALSE(scan.LikelyDenseTexture());
  scan.sampled_colors = 248;
  scan.sampled_flat = 342;
  EXPECT_FALSE(scan.LikelyDenseTexture());
}

TEST(JxlTest, CalibrationAgreementNeedsBothCompleteSignals) {
  EXPECT_TRUE(LosslessCalibrationAgreement({7480004.8, 7514732.5, 7641875.3},
                                            {7884611, 8036352, 7986653}, 0));
  EXPECT_FALSE(LosslessCalibrationAgreement({7480004.8, 7514732.5}, {7884611, 7900000}, 0));
  EXPECT_FALSE(LosslessCalibrationAgreement({100, 100.1}, {100, 105}, 0));
  EXPECT_FALSE(LosslessCalibrationAgreement({100, 99}, {100, 105}, 0));
  EXPECT_FALSE(LosslessCalibrationAgreement({100, 105}, {100, 99}, 0));
  EXPECT_FALSE(LosslessCalibrationAgreement({100}, {100}, 0));
  EXPECT_FALSE(LosslessCalibrationAgreement({100, 105}, {100}, 0));
  EXPECT_FALSE(LosslessCalibrationAgreement({100, 105}, {100, 106}, 2));
  EXPECT_FALSE(LosslessCalibrationAgreement({100, 105}, {100, 0}, 0));
  EXPECT_FALSE(LosslessCalibrationAgreement({100, std::numeric_limits<double>::infinity()}, {100, 106}, 0));
}

TEST(JxlTest, CalibrationShrinkSharesOnlyMatchedFamilies) {
  CompressParams raw;
  raw.SetLossless();
  raw.palette_colors = 0;
  raw.colorspace = 6;
  raw.options.predictor = Predictor::Variable;
  auto sibling = raw;
  sibling.colorspace = 10;
  auto unrelated = sibling;
  unrelated.modular_group_size_shift = 2;
  const std::vector<CompressParams> plans{raw, sibling, unrelated};
  const auto result = ShrinkLosslessCalibration(plans, {1000, 970, 900},
                                               {800, 873, 700}, {1, 1, 1});
  EXPECT_NEAR(result[0], 840, 1e-9);
  EXPECT_NEAR(result[1], 834.2, 1e-9);
  EXPECT_EQ(result[2], 700);
  const auto more = ShrinkLosslessCalibration(plans, {1000, 970, 900},
                                             {800, 873, 700}, {16, 16, 16});
  EXPECT_LT(more[0], result[0]);
  EXPECT_GT(more[1], result[1]);
  EXPECT_EQ(ShrinkLosslessCalibration(plans, {0, 970, 900},
                                      {800, 873, 700}, {1, 1, 1}),
            (std::vector<double>{800, 873, 700}));
}

TEST(JxlTest, SiblingCalibrationRequiresMatchedEvidence) {
  LosslessCalibrationFeedback feedback;
  feedback.params.SetLossless();
  feedback.params.options.predictor = Predictor::Variable;
  feedback.params.palette_colors = 0;
  feedback.params.colorspace = 6;
  feedback.deep = {100, 200, 300};
  feedback.cheap = {200, 400, 600};
  feedback.full_bytes = 12000;
  auto candidate = feedback.params;
  candidate.colorspace = 17;
  EXPECT_DOUBLE_EQ(feedback.SiblingEstimate(candidate, {100, 200, 300}), 6000);
  EXPECT_DOUBLE_EQ(feedback.SiblingEstimate(candidate, {94, 400, 600}), 0);
  EXPECT_DOUBLE_EQ(feedback.SiblingEstimate(candidate, {100, 200}), 0);
  candidate.options.nb_repeats += 0.01f;
  EXPECT_DOUBLE_EQ(feedback.SiblingEstimate(candidate, {100, 200, 300}), 0);
  candidate = feedback.params;
  ++candidate.modular_group_size_shift;
  EXPECT_DOUBLE_EQ(feedback.SiblingEstimate(candidate, {100, 200, 300}), 0);
  candidate = feedback.params;
  candidate.palette_colors = 10000;
  EXPECT_DOUBLE_EQ(feedback.SiblingEstimate(candidate, {100, 200, 300}), 0);
  candidate = feedback.params;
  feedback.cheap = {0, 0, 0};
  EXPECT_DOUBLE_EQ(feedback.SiblingEstimate(candidate, {100, 200, 300}), 0);
  feedback.deep = {100};
  feedback.cheap = {200};
  EXPECT_DOUBLE_EQ(feedback.SiblingEstimate(candidate, {100}), 0);
}

TEST(JxlTest, DirectFrontierUsesRealFinalistSizesAndPropagatesFailure) {
  const char* enabled = std::getenv("JXL_LOSSLESS_DIRECT_FRONTIER");
  if (!enabled || (std::strcmp(enabled, "1") && std::strcmp(enabled, "2") &&
                  std::strcmp(enabled, "3") && std::strcmp(enabled, "4") &&
                  std::strcmp(enabled, "5"))) GTEST_SKIP();
  const char* compact_value = std::getenv("JXL_LOSSLESS_COMPACT_FRONTIER");
  const bool compact_frontier = compact_value && std::strcmp(compact_value, "1") == 0;
  const bool calibrated = enabled[0] != '1' && !compact_frontier;
  const char* fidelity_value = std::getenv("JXL_LOSSLESS_FRONTIER_CALIBRATION");
  const int calibration_level = fidelity_value && std::strcmp(fidelity_value, "7") == 0 ? 7 : 4;
  const size_t width = compact_frontier ? 769 : 2049, height = compact_frontier ? 769 : 1025;
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(width, height));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  std::vector<uint8_t> pixels(width * height * 3, 17);
  for (size_t i = 0; i < width * height; ++i) {
    uint32_t value = static_cast<uint32_t>(i + 1) * 0x9e3779b9u;
    value ^= value >> 16;
    for (size_t c = 0; c < 3; ++c) pixels[3 * i + c] = value >> (8 * c);
  }
  JxlEncoderChunkedFrameAdapter input(width, height, 0);
  ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                  {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}));
  CompressParams params;
  params.SetLossless();
  params.speed_tier = SpeedTier::kTortoise;
  const auto candidate = [](const CompressParams& p) {
    return p.options.predictor == Predictor::Variable && p.palette_colors == 0 &&
           (p.colorspace == 0 || p.colorspace == 6);
  };
  for (bool fail : {false, true}) {
    std::atomic<size_t> encodes{0}, cheap_encodes{0};
    const auto encode = [&](const CompressParams& p, ThreadPool* runner,
                            JxlEncoderOutputProcessorWrapper* output, AuxOut*) -> Status {
      ++encodes;
      EXPECT_TRUE(candidate(p));
      std::vector<uint32_t> scratch(32);
      JXL_RETURN_IF_ERROR(RunOnPool(runner, 0, scratch.size(), ThreadPool::NoInit,
          [&](uint32_t i, size_t) -> Status { scratch[i] = i + 1; return true; },
          "FinalistTestWork"));
      for (size_t i = 0; i < scratch.size(); ++i) EXPECT_EQ(scratch[i], i + 1);
      if (LosslessSearchLevel(p) == calibration_level) {
        ++cheap_encodes;
        EXPECT_TRUE(calibrated);
      } else {
        EXPECT_EQ(LosslessSearchLevel(p), 9);
      }
      return AppendData(*output, std::vector<uint8_t>(p.colorspace == 0 ? 9000 : 8000));
    };
    const auto probe = [&](const CompressParams& p, const CodecMetadata&,
                           JxlEncoderChunkedFrameAdapter&) -> StatusOr<size_t> {
      if (fail) return JXL_FAILURE("Intentional scout failure");
      if (candidate(p)) return size_t{p.colorspace == 0 ? 100u : 101u};
      return size_t{1000};
    };
    auto* memory = test::MemoryManager();
    JxlEncoderOutputProcessorWrapper output(memory);
    test::ThreadPoolForTests pool(4);
    auto result = EncodeLosslessSearch(memory, params, FrameInfo{}, metadata,
        input, pool.get(), &output, nullptr, encode, probe);
    if (fail) {
      EXPECT_FALSE(result.ok());
      EXPECT_EQ(encodes.load(), 0u);
    } else {
      ASSERT_TRUE(result.ok());
      ASSERT_TRUE(std::move(result).value_());
      ASSERT_TRUE(output.SetFinalizedPosition());
      std::vector<uint8_t> bytes;
      ASSERT_TRUE(output.CopyOutput(bytes));
      EXPECT_EQ(encodes.load(), calibrated ? 3u : 2u);
      EXPECT_EQ(cheap_encodes.load(), calibrated ? 2u : 0u);
      EXPECT_EQ(bytes.size(), 8000u);
    }
  }
}

TEST(JxlTest, SparseFrontierCalibratesCloseTransformSiblings) {
  const char* mode = std::getenv("JXL_LOSSLESS_DIRECT_FRONTIER");
  if (!mode || (std::strcmp(mode, "3") && std::strcmp(mode, "4") &&
                std::strcmp(mode, "5"))) GTEST_SKIP();
  for (const auto& setting : {std::pair<const char*, const char*>{
           "JXL_LOSSLESS_PALETTE_SCOPE", "1"}, {"JXL_LOSSLESS_RCT_SCREEN", "1"}}) {
    const char* value = std::getenv(setting.first);
    if (!value || std::strcmp(value, setting.second)) GTEST_SKIP();
  }
  constexpr size_t width = 2049, height = 1025;
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(width, height));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  std::vector<uint8_t> pixels(width * height * 3);
  for (size_t i = 0; i < width * height / 4; ++i) {
    uint32_t value = static_cast<uint32_t>(i + 1) * 0x9e3779b9u;
    value ^= value >> 16;
    for (size_t c = 0; c < 3; ++c) pixels[3 * i + c] = value >> (8 * c);
  }
  JxlEncoderChunkedFrameAdapter input(width, height, 0);
  ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                  {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}));
  CompressParams params;
  params.SetLossless();
  params.speed_tier = SpeedTier::kTortoise;
  const auto transform = [](const CompressParams& p) {
    return p.options.predictor == Predictor::Variable && p.palette_colors == 0 &&
        (p.colorspace == 6 || p.colorspace == 10 || p.colorspace == 26);
  };
  std::vector<int> cheap, deep;
  const auto encode = [&](const CompressParams& p, ThreadPool*,
                          JxlEncoderOutputProcessorWrapper* output, AuxOut*) -> Status {
    EXPECT_TRUE(transform(p));
    const bool calibration = LosslessSearchLevel(p) == 4;
    (calibration ? cheap : deep).push_back(p.colorspace);
    const size_t bytes = p.colorspace == 10 ? 8000 : p.colorspace == 6 ? 9000 : 10000;
    return AppendData(*output, std::vector<uint8_t>(bytes + (calibration ? 2000 : 0)));
  };
  const auto probe = [&](const CompressParams& p, const CodecMetadata&,
                         JxlEncoderChunkedFrameAdapter&) -> StatusOr<size_t> {
    if (!transform(p)) return size_t{10000};
    if (LosslessSearchLevel(p) == 4) return size_t{1100};
    return size_t{p.colorspace == 6 ? 1000u : p.colorspace == 26 ? 1010u : 1020u};
  };
  auto* memory = test::MemoryManager();
  JxlEncoderOutputProcessorWrapper output(memory);
  JXL_TEST_ASSIGN_OR_DIE(bool searched, EncodeLosslessSearch(memory, params,
      FrameInfo{}, metadata, input, nullptr, &output, nullptr, encode, probe));
  ASSERT_TRUE(searched);
  EXPECT_EQ(cheap.size(), 3u);
  ASSERT_EQ(deep.size(), 2u);
  EXPECT_EQ(deep[0], 10);
  EXPECT_EQ(deep[1], 6);
  ASSERT_TRUE(output.SetFinalizedPosition());
  std::vector<uint8_t> bytes;
  ASSERT_TRUE(output.CopyOutput(bytes));
  EXPECT_EQ(bytes.size(), 8000u);
}

TEST(JxlTest, LosslessScopeWinnerDoesNotReopenLiteralBranch) {
  for (const char* key : {"JXL_LOSSLESS_CALIBRATE", "JXL_LOSSLESS_CALIBRATE_FAST",
          "JXL_LOSSLESS_CALIBRATE_ROUTES", "JXL_LOSSLESS_PREFLIGHT_YIELD",
          "JXL_LOSSLESS_CALIBRATION_REUSE", "JXL_LOSSLESS_EARLY_FAMILY",
          "JXL_LOSSLESS_PLAN_FIRST", "JXL_LOSSLESS_SCAN",
          "JXL_LOSSLESS_FAST_FAMILY",
          "JXL_LOSSLESS_PALETTE_SCOPE", "JXL_LOSSLESS_SCOPE_ROUTING"}) {
    const char* value = std::getenv(key);
    if (!value || std::strcmp(value, "1")) GTEST_SKIP();
  }
  constexpr size_t width = 2049, height = 1025;
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(width, height));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  std::vector<uint8_t> pixels(width * height * 3);
  for (size_t i = 0; i < width * height; ++i) {
    const size_t value = i < 80000 ? i : 17;
    pixels[3 * i] = value & 255;
    pixels[3 * i + 1] = (value >> 8) & 255;
    pixels[3 * i + 2] = (value >> 16) & 255;
  }
  JxlEncoderChunkedFrameAdapter input(width, height, 0);
  ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                  {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}));
  CompressParams params;
  params.SetLossless();
  params.speed_tier = SpeedTier::kTortoise;
  size_t raw_deep = 0, literal_deep = 0, automatic_before_raw = 0;
  const auto encode = [&](const CompressParams& p, ThreadPool*,
                          JxlEncoderOutputProcessorWrapper* output, AuxOut*) -> Status {
    size_t bytes = 30000;
    if (p.options.predictor == kUndefinedPredictor) {
      bytes = 15000;
      if (!raw_deep && LosslessSearchLevel(p) >= 7) ++automatic_before_raw;
    }
    if (p.options.predictor == Predictor::Zero && p.palette_colors > 0) {
      bytes = 10000;
      if (LosslessSearchLevel(p) >= 9) ++literal_deep;
    }
    if (p.options.predictor == Predictor::Variable && p.palette_colors == 0 &&
        LosslessSearchLevel(p) >= 9) {
      ++raw_deep;
      bytes = 4000;
    }
    return AppendData(*output, std::vector<uint8_t>(bytes));
  };
  const auto probe = [](const CompressParams& p, const CodecMetadata&,
                         JxlEncoderChunkedFrameAdapter&) -> StatusOr<size_t> {
    if (p.options.predictor == Predictor::Variable && p.palette_colors == 0) return size_t{70};
    if (p.options.predictor == kUndefinedPredictor) return size_t{100};
    return size_t{1000};
  };
  auto* memory = test::MemoryManager();
  JxlEncoderOutputProcessorWrapper output(memory);
  JXL_TEST_ASSIGN_OR_DIE(bool searched, EncodeLosslessSearch(memory, params,
      FrameInfo{}, metadata, input, nullptr, &output, nullptr, encode, probe));
  ASSERT_TRUE(searched);
  ASSERT_TRUE(output.SetFinalizedPosition());
  std::vector<uint8_t> bytes;
  ASSERT_TRUE(output.CopyOutput(bytes));
  EXPECT_EQ(bytes.size(), 4000u);
  EXPECT_GT(raw_deep, 0u);
  EXPECT_EQ(literal_deep, 0u);
  EXPECT_EQ(automatic_before_raw, 0u);
}

TEST(JxlTest, LosslessCalibrationOrderPreservesBothGatesAndE10) {
  for (const char* key : {"JXL_LOSSLESS_CALIBRATE",
                          "JXL_LOSSLESS_CALIBRATE_FAST",
                          "JXL_LOSSLESS_CALIBRATE_ROUTES",
                          "JXL_LOSSLESS_PREFLIGHT_YIELD",
                          "JXL_LOSSLESS_CALIBRATION_REUSE",
                          "JXL_LOSSLESS_EARLY_FAMILY",
                          "JXL_LOSSLESS_CALIBRATION_ORDER"}) {
    const char* value = std::getenv(key);
    if (!value || std::strcmp(value, "1") != 0) GTEST_SKIP();
  }
  auto* memory = test::MemoryManager();
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(2049, 1025));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  std::vector<uint8_t> pixels(2049 * 1025 * 3, 17);
  JxlEncoderChunkedFrameAdapter input(2049, 1025, 0);
  ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                  {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}));
  // e7 rejects; e7 accepts but e4 rejects; both accept; clear crop loss;
  // strong crop win. Deferred plans remain reachable at e10.
  for (int scenario : {0, 1, 2, 3, 4}) {
    for (int effort : {9, 10}) {
      SCOPED_TRACE(testing::Message() << scenario << " effort=" << effort);
      CompressParams params;
      params.SetLossless();
      params.speed_tier = static_cast<SpeedTier>(10 - effort);
      std::vector<int> geometry_witnesses;
      const auto raw = [](const CompressParams& p) {
        return p.palette_colors == 0 && p.colorspace == 0 &&
               p.options.predictor == Predictor::Variable;
      };
      const auto probe = [&](const CompressParams& p, const CodecMetadata&,
                             JxlEncoderChunkedFrameAdapter&) -> StatusOr<size_t> {
        if (p.options.predictor == kUndefinedPredictor) return size_t{100};
        if (raw(p)) {
          if (LosslessSearchLevel(p) < 9) return size_t{100};
          if (p.modular_group_size_shift == 2) {
            if (scenario == 3) return size_t{72};
            if (scenario == 4) return size_t{60};
          }
          return size_t{70};
        }
        return size_t{1000};
      };
      const auto encode = [&](const CompressParams& p, ThreadPool*,
                              JxlEncoderOutputProcessorWrapper* out,
                              AuxOut*) -> Status {
        const int backend = LosslessSearchLevel(p);
        size_t bytes = 30000;
        if (p.options.predictor == kUndefinedPredictor) {
          bytes = backend <= 3 ? 15000 : 10000;
        } else if (raw(p)) {
          bytes = backend >= 9 ? 6000 : 13000;
          if (p.modular_group_size_shift == 2) {
            if (backend == 4 || backend == 7) geometry_witnesses.push_back(backend);
            bytes = backend >= 9 ? 5500 :
                backend == 7 ? (scenario == 0 ? 10000 : 8000) :
                scenario == 1 || scenario == 3 ? 15000 : 13000;
          }
        }
        return AppendData(*out, std::vector<uint8_t>(bytes, 0));
      };
      JxlEncoderOutputProcessorWrapper output(memory);
      JXL_TEST_ASSIGN_OR_DIE(bool searched, EncodeLosslessSearch(
          memory, params, FrameInfo{}, metadata, input, nullptr, &output,
          nullptr, encode, probe));
      ASSERT_TRUE(searched);
      ASSERT_TRUE(output.SetFinalizedPosition());
      std::vector<uint8_t> bytes;
      ASSERT_TRUE(output.CopyOutput(bytes));
      EXPECT_EQ(geometry_witnesses,
                scenario >= 3 ? std::vector<int>({4}) :
                scenario == 0 ? std::vector<int>({7}) : std::vector<int>({7, 4}));
      EXPECT_EQ(bytes.size(),
                effort == 10 || scenario == 2 || scenario == 4 ? 5500u : 6000u);
    }
  }
}

TEST(JxlTest, LosslessPlanFirstFallbackAndDeferredRecovery) {
  for (const char* key : {"JXL_LOSSLESS_CALIBRATE",
                          "JXL_LOSSLESS_CALIBRATE_FAST",
                          "JXL_LOSSLESS_CALIBRATE_ROUTES",
                          "JXL_LOSSLESS_PREFLIGHT_YIELD",
                          "JXL_LOSSLESS_CALIBRATION_REUSE",
                          "JXL_LOSSLESS_EARLY_FAMILY",
                          "JXL_LOSSLESS_PLAN_FIRST"}) {
    const char* value = std::getenv(key);
    if (!value || std::strcmp(value, "1") != 0) GTEST_SKIP();
  }
  auto* memory = test::MemoryManager();
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(2049, 1025));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  std::vector<uint8_t> pixels(2049 * 1025 * 3, 17);
  JxlEncoderChunkedFrameAdapter input(2049, 1025, 0);
  ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                  {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}));
  const char* bucket_value = std::getenv("JXL_LOSSLESS_PLAN_BUCKETS");
  const bool routed_buckets = bucket_value && std::strcmp(bucket_value, "1") == 0;
  // Strong win; misleading full result; weak regional evidence; hidden
  // automatic winner; hidden geometry winner; better original chain matcher;
  // better e4 witness; noncompeting cheap families. Scenarios 3-6 expose
  // the heuristic's limit and check that the e10 recovery is not screened out.
  for (int scenario : {0, 1, 2, 3, 4, 5, 6, 7}) {
    if (scenario == 5 && !routed_buckets) continue;
    size_t previous = std::numeric_limits<size_t>::max();
    for (int effort : {3, 4, 5, 7, 8, 9, 10}) {
      SCOPED_TRACE(testing::Message() << scenario << " effort=" << effort);
      CompressParams params;
      params.SetLossless();
      params.speed_tier = static_cast<SpeedTier>(10 - effort);
      size_t automatic_calls = 0, geometry_calls = 0, cheap_calls = 0;
      const auto raw = [](const CompressParams& p) {
        return p.palette_colors == 0 && p.colorspace == 0 &&
               p.options.predictor == Predictor::Variable;
      };
      const auto probe = [&](const CompressParams& p, const CodecMetadata&,
                             JxlEncoderChunkedFrameAdapter&) -> StatusOr<size_t> {
        if (p.options.predictor == kUndefinedPredictor) return size_t{100};
        if (raw(p)) return size_t{
            LosslessSearchLevel(p) >= 9 ? (scenario == 2 ? 90u : 70u) : 100u};
        return size_t{1000};
      };
      const auto encode = [&](const CompressParams& p, ThreadPool*,
                              JxlEncoderOutputProcessorWrapper* out,
                              AuxOut*) -> Status {
        const int backend = LosslessSearchLevel(p);
        size_t bytes = p.options.predictor == Predictor::Zero &&
                               p.palette_colors == 10000 && scenario != 7
                           ? 14000 : 30000;
        if (p.options.predictor == kUndefinedPredictor) {
          bytes = backend <= 3 ? 15000 : (scenario == 3 ? 5000 : 10000);
          if (backend >= 7) ++automatic_calls;
        } else if (raw(p)) {
          bytes = backend >= 9 ? (scenario == 1 ? 14000 : 6000) : 13000;
          if (backend == 4) {
            ++cheap_calls;
            if (scenario == 6) bytes = 5000;
          }
          if (scenario == 5 && backend >= 9 && !p.lossless_bucket_matcher)
            bytes = 5500;
          if (p.modular_group_size_shift == 2) {
            ++geometry_calls;
            bytes = backend >= 9 && scenario == 4 ? 5500 : 16000;
          }
        }
        return AppendData(*out, std::vector<uint8_t>(bytes, 0));
      };
      JxlEncoderOutputProcessorWrapper output(memory);
      JXL_TEST_ASSIGN_OR_DIE(bool searched, EncodeLosslessSearch(
          memory, params, FrameInfo{}, metadata, input, nullptr, &output,
          nullptr, encode, probe));
      ASSERT_TRUE(searched);
      ASSERT_TRUE(output.SetFinalizedPosition());
      std::vector<uint8_t> bytes;
      ASSERT_TRUE(output.CopyOutput(bytes));
      EXPECT_LE(bytes.size(), previous);
      previous = bytes.size();
      if (scenario == 7) {
        EXPECT_GT(automatic_calls, 0u);
        EXPECT_EQ(bytes.size(), effort < 5 ? 10000u : 6000u);
      } else if (scenario == 0 || scenario >= 3) {
        if (effort < 10) {
          EXPECT_EQ(automatic_calls, 0u);
          EXPECT_EQ(geometry_calls, 0u);
          EXPECT_EQ(cheap_calls, 0u);
          EXPECT_EQ(bytes.size(), 6000u);
        } else {
          EXPECT_GT(automatic_calls, 0u);
          EXPECT_GT(cheap_calls, 0u);
          EXPECT_EQ(bytes.size(), scenario == 3 || scenario == 6 ? 5000u :
                                  scenario >= 4 ? 5500u : 6000u);
        }
      } else {
        EXPECT_GT(automatic_calls, 0u);
        EXPECT_LE(bytes.size(), 10000u);
      }
    }
  }
}

TEST(JxlTest, LosslessGeometryFirstCalibratesAndRecovers) {
  for (const char* key : {"JXL_LOSSLESS_CALIBRATE",
                          "JXL_LOSSLESS_CALIBRATE_FAST",
                          "JXL_LOSSLESS_CALIBRATE_ROUTES",
                          "JXL_LOSSLESS_PREFLIGHT_YIELD",
                          "JXL_LOSSLESS_CALIBRATION_REUSE",
                          "JXL_LOSSLESS_EARLY_FAMILY",
                          "JXL_LOSSLESS_PLAN_FIRST"}) {
    const char* value = std::getenv(key);
    if (!value || std::strcmp(value, "1") != 0) GTEST_SKIP();
  }
  const char* geometry_value = std::getenv("JXL_LOSSLESS_GEOMETRY_FIRST");
  if (geometry_value && std::strcmp(geometry_value, "0") == 0) GTEST_SKIP();
  auto* memory = test::MemoryManager();
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(2049, 1025));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  std::vector<uint8_t> pixels(2049 * 1025 * 3, 17);
  JxlEncoderChunkedFrameAdapter input(2049, 1025, 0);
  ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                  {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}));
  // Measured win, misleading calibration, failed full candidate, and a hidden
  // automatic e8 winner. All efforts share a prefix; e10 reopens deferred work.
  for (int scenario : {0, 1, 2, 3}) {
    size_t previous = std::numeric_limits<size_t>::max();
    for (int effort : {3, 4, 5, 7, 8, 9, 10}) {
      SCOPED_TRACE(testing::Message() << scenario << " effort=" << effort);
      CompressParams params;
      params.SetLossless();
      params.speed_tier = static_cast<SpeedTier>(10 - effort);
      size_t automatic_calls = 0, medium_calls = 0, deep_calls = 0;
      const auto raw = [](const CompressParams& p) {
        return p.palette_colors == 0 && p.colorspace == 0 &&
               p.options.predictor == Predictor::Variable;
      };
      const auto probe = [&](const CompressParams& p, const CodecMetadata&,
                             JxlEncoderChunkedFrameAdapter&) -> StatusOr<size_t> {
        if (p.options.predictor == kUndefinedPredictor)
          return size_t{LosslessSearchLevel(p) >= 9 ? 110u : 150u};
        if (raw(p)) return size_t{LosslessSearchLevel(p) < 9 ? 120u :
            p.modular_group_size_shift == 2 ? 90u : 96u};
        return size_t{1000};
      };
      const auto encode = [&](const CompressParams& p, ThreadPool*,
                              JxlEncoderOutputProcessorWrapper* out,
                              AuxOut*) -> Status {
        const int backend = LosslessSearchLevel(p);
        size_t bytes = p.options.predictor == Predictor::Zero &&
                               p.palette_colors == 10000 ? 5000 : 30000;
        if (p.options.predictor == kUndefinedPredictor) {
          bytes = backend <= 3 ? 15000 : backend == 7 ? 4500 : 3500;
          if (backend >= 8) ++automatic_calls;
          if (scenario == 3 && backend == 8) bytes = 2500;
        } else if (raw(p)) {
          bytes = backend >= 9 ? 3500 : 6000;
          if (p.modular_group_size_shift == 2) {
            if (backend == 7) ++medium_calls;
            if (backend >= 9) ++deep_calls;
            bytes = backend >= 9 ? (scenario == 2 ? 4500 : 3000) :
                    scenario == 1 ? 9000 : 5000;
          }
        }
        return AppendData(*out, std::vector<uint8_t>(bytes, 0));
      };
      JxlEncoderOutputProcessorWrapper output(memory);
      JXL_TEST_ASSIGN_OR_DIE(bool searched, EncodeLosslessSearch(
          memory, params, FrameInfo{}, metadata, input, nullptr, &output,
          nullptr, encode, probe));
      ASSERT_TRUE(searched);
      ASSERT_TRUE(output.SetFinalizedPosition());
      std::vector<uint8_t> bytes;
      ASSERT_TRUE(output.CopyOutput(bytes));
      EXPECT_LE(bytes.size(), previous);
      previous = bytes.size();
      if (effort == 3) {
        EXPECT_EQ(bytes.size(), 4500u);
        EXPECT_EQ(medium_calls, 0u);
      } else {
        EXPECT_GT(medium_calls, 0u);
        if (scenario == 0 || scenario == 3) {
          EXPECT_GT(deep_calls, 0u);
          EXPECT_EQ(bytes.size(), effort == 10 && scenario == 3 ? 2500u : 3000u);
          if (effort < 10) EXPECT_EQ(automatic_calls, 0u);
        } else {
          EXPECT_GT(automatic_calls, 0u);
          EXPECT_LE(bytes.size(), 3500u);
          if (scenario == 1 && effort < 10) EXPECT_EQ(deep_calls, 0u);
        }
      }
    }
  }
}

TEST(JxlTest, LosslessAutomaticYieldConfirmsBothScalesAndReplaysE10) {
  for (const char* key : {"JXL_LOSSLESS_CALIBRATE",
                          "JXL_LOSSLESS_CALIBRATE_FAST",
                          "JXL_LOSSLESS_CALIBRATE_ROUTES",
                          "JXL_LOSSLESS_PREFLIGHT_YIELD",
                          "JXL_LOSSLESS_CALIBRATION_REUSE",
                          "JXL_LOSSLESS_EARLY_FAMILY",
                          "JXL_LOSSLESS_PLAN_FIRST"}) {
    const char* value = std::getenv(key);
    if (!value || std::strcmp(value, "1") != 0) GTEST_SKIP();
  }
  auto* memory = test::MemoryManager();
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(2049, 1025));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  std::vector<uint8_t> pixels(2049 * 1025 * 3, 17);
  JxlEncoderChunkedFrameAdapter input(2049, 1025, 0);
  ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                  {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}));
  const char* yield_value = std::getenv("JXL_LOSSLESS_AUTOMATIC_YIELD");
  if (yield_value && std::strcmp(yield_value, "0") == 0) GTEST_SKIP();
  // Both scales lose; second scale wins; hidden global win recovered at e10;
  // first scale wins; unusable second-scale parent. A strong first scale
  // requires no extra probes, while missing evidence keeps the full trial.
  for (int scenario : {0, 1, 2, 3, 4}) {
    size_t previous = std::numeric_limits<size_t>::max();
    for (int effort : {3, 5, 7, 8, 9, 10}) {
      SCOPED_TRACE(testing::Message() << scenario << " effort=" << effort);
      CompressParams params;
      params.SetLossless();
      params.speed_tier = static_cast<SpeedTier>(10 - effort);
      size_t automatic_deep = 0, context_probes = 0;
      const auto automatic = [](const CompressParams& p) {
        return p.options.predictor == kUndefinedPredictor &&
               p.modular_group_size_shift == -1;
      };
      const auto raw = [](const CompressParams& p) {
        return p.palette_colors == 0 && p.colorspace == 0 &&
               p.options.predictor == Predictor::Variable;
      };
      const auto probe = [&](const CompressParams& p, const CodecMetadata&,
                             JxlEncoderChunkedFrameAdapter& frame) -> StatusOr<size_t> {
        if (automatic(p)) {
          if (LosslessSearchLevel(p) < 9) {
            return size_t{scenario == 4 && frame.xsize == 128 ? 0u : 10000u};
          }
          if (frame.xsize == 128) ++context_probes;
          return size_t{scenario == 3 || (scenario == 1 && frame.xsize == 128)
                            ? 8285u : 8615u};
        }
        if (raw(p)) return size_t{LosslessSearchLevel(p) >= 9 ? 8650u : 10000u};
        return size_t{30000};
      };
      const auto encode = [&](const CompressParams& p, ThreadPool*,
                              JxlEncoderOutputProcessorWrapper* out,
                              AuxOut*) -> Status {
        const int backend = LosslessSearchLevel(p);
        size_t bytes = 30000;
        if (automatic(p)) {
          bytes = backend <= 3 ? 15000 : backend < 9 ? 7000 :
                  scenario == 0 ? 6050 : 5800;
          if (backend >= 9) ++automatic_deep;
        } else if (raw(p)) {
          bytes = backend >= 9 ? 6000 : 7000;
        }
        return AppendData(*out, std::vector<uint8_t>(bytes, 0));
      };
      JxlEncoderOutputProcessorWrapper output(memory);
      JXL_TEST_ASSIGN_OR_DIE(bool searched, EncodeLosslessSearch(
          memory, params, FrameInfo{}, metadata, input, nullptr, &output,
          nullptr, encode, probe));
      ASSERT_TRUE(searched);
      ASSERT_TRUE(output.SetFinalizedPosition());
      std::vector<uint8_t> bytes;
      ASSERT_TRUE(output.CopyOutput(bytes));
      EXPECT_LE(bytes.size(), previous);
      previous = bytes.size();
      if (effort < 7) continue;
      const bool defer = effort < 10 && (scenario == 0 || scenario == 2);
      EXPECT_EQ(automatic_deep == 0, defer);
      EXPECT_EQ(bytes.size(), scenario == 0 || defer ? 6000u : 5800u);
      if (effort < 10) EXPECT_EQ(context_probes == 0, scenario == 3);
    }
  }
}

TEST(JxlTest, LosslessSmallerGroupsUsesCloseFinalistsAndCachedExactScores) {
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(1025, 1025));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  std::vector<uint8_t> pixels(1025 * 1025 * 3, 17);
  JxlEncoderChunkedFrameAdapter input(1025, 1025, 0);
  ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                  {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}));
  size_t calls = 0;
  const auto probe = [&](const CompressParams& p, const CodecMetadata&,
                         JxlEncoderChunkedFrameAdapter&) -> StatusOr<size_t> {
    ++calls;
    EXPECT_EQ(LosslessSearchLevel(p), 9);
    if (p.modular_group_size_shift == 3) return size_t{1000};
    EXPECT_EQ(p.modular_group_size_shift, 2);
    EXPECT_NE(p.options.predictor, Predictor::Gradient);
    return size_t{p.options.predictor == Predictor::Variable ? 950u : 990u};
  };
  LosslessCandidateScreen screen(metadata, input, nullptr, probe);
  CompressParams p;
  p.SetLossless();
  p.speed_tier = SpeedTier::kTortoise;
  p.palette_colors = 0;
  p.modular_group_size_shift = 3;
  std::vector<LosslessProbeCandidate> ranked;
  for (const auto predictor : {Predictor::Select, Predictor::Variable,
                               Predictor::Gradient}) {
    p.options.predictor = predictor;
    JXL_TEST_ASSIGN_OR_DIE(auto scores, screen.Score(p));
    LosslessProbeCandidate candidate{"test", p};
    for (size_t bytes : scores) candidate.context_bytes += bytes;
    ranked.push_back(candidate);
  }
  JXL_TEST_ASSIGN_OR_DIE(auto result, screen.SmallerGroups(ranked));
  ASSERT_EQ(result.size(), 2u);
  EXPECT_EQ(result.front().params.options.predictor, Predictor::Variable);
  EXPECT_LT(result.front().context_bytes, result.back().context_bytes);
  const size_t before = calls;
  JXL_TEST_ASSIGN_OR_DIE(auto cached, screen.SmallerGroups(ranked));
  EXPECT_EQ(calls, before);
  EXPECT_EQ(cached.front().context_bytes, result.front().context_bytes);
  // Do not spend probes on a distant family, even when there is spare capacity.
  ranked.resize(2);
  ranked[1].context_bytes = ranked[0].context_bytes * 2;
  JXL_TEST_ASSIGN_OR_DIE(auto separated, screen.SmallerGroups(ranked));
  ASSERT_EQ(separated.size(), 1u);
  EXPECT_EQ(separated.front().params.options.predictor, Predictor::Select);
  ranked[0].params.palette_colors = 10000;
  JXL_TEST_ASSIGN_OR_DIE(auto palette, screen.SmallerGroups(ranked));
  EXPECT_TRUE(palette.empty());
  JXL_TEST_ASSIGN_OR_DIE(auto empty, screen.SmallerGroups({}));
  EXPECT_TRUE(empty.empty());
}

TEST(JxlTest, LosslessCandidateMatcherHasIndependentProbeCache) {
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(1025, 1025));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  std::vector<uint8_t> pixels(1025 * 1025 * 3, 17);
  JxlEncoderChunkedFrameAdapter input(1025, 1025, 0);
  ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                  {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}));
  size_t calls = 0;
  const auto probe = [&](const CompressParams& p, const CodecMetadata&,
                         JxlEncoderChunkedFrameAdapter&) -> StatusOr<size_t> {
    ++calls;
    return size_t{p.lossless_bucket_matcher ? 99u : 100u};
  };
  LosslessCandidateScreen screen(metadata, input, nullptr, probe);
  CompressParams chain;
  chain.SetLossless();
  auto buckets = chain;
  buckets.lossless_bucket_matcher = true;
  EXPECT_FALSE(SameLosslessCandidate(chain, buckets));
  JXL_TEST_ASSIGN_OR_DIE(auto reference, screen.Score(chain));
  const size_t first_calls = calls;
  EXPECT_GT(first_calls, 0u);
  JXL_TEST_ASSIGN_OR_DIE(auto candidate, screen.Score(buckets));
  EXPECT_EQ(calls, 2 * first_calls);
  EXPECT_NE(reference, candidate);
  JXL_TEST_ASSIGN_OR_DIE(auto cached, screen.Score(buckets));
  EXPECT_EQ(candidate, cached);
  EXPECT_EQ(calls, 2 * first_calls);
}

TEST(JxlTest, LosslessYieldSmallScreenCoverageAndCache) {
  const char* value = std::getenv("JXL_LOSSLESS_PREFLIGHT_YIELD");
  if (!value || std::strcmp(value, "1") != 0) GTEST_SKIP();
  for (size_t width : {255u, 256u, 512u, 768u}) {
    constexpr size_t height = 512;
    CodecMetadata metadata;
    ASSERT_TRUE(metadata.size.Set(width, height));
    metadata.m.SetUintSamples(8);
    metadata.m.SetAlphaBits(8);
    metadata.m.color_encoding = ColorEncoding::SRGB();
    std::vector<uint8_t> pixels(width * height * 4, 17);
    for (size_t i = 3; i < pixels.size(); i += 4) pixels[i] = 0;
    const auto original = pixels;
    JxlEncoderChunkedFrameAdapter input(width, height, 1);
    ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                    {4, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}));
    size_t calls = 0;
    const auto probe = [&](const CompressParams&, const CodecMetadata&,
                           JxlEncoderChunkedFrameAdapter& sample) -> StatusOr<size_t> {
      ++calls;
      EXPECT_EQ(sample.xsize * sample.ysize, width < 512 ? 1024u : 4096u);
      auto source = sample.GetInputSource();
      size_t stride;
      auto buffer = GetColorBuffer(source, 0, 0, sample.xsize, sample.ysize, &stride);
      const auto* data = static_cast<const uint8_t*>(buffer.get());
      for (size_t y = 0; y < sample.ysize; ++y) {
        for (size_t x = 0; x < sample.xsize; ++x) {
          for (size_t c = 0; c < 3; ++c) EXPECT_EQ(data[y * stride + 4 * x + c], 17);
          EXPECT_EQ(data[y * stride + 4 * x + 3], 0);
        }
      }
      return size_t{100};
    };
    LosslessCandidateScreen screen(metadata, input, nullptr, probe);
    EXPECT_EQ(screen.Available(), width >= 256);
    if (!screen.Available()) continue;
    CompressParams params;
    params.SetLossless();
    JXL_TEST_ASSIGN_OR_DIE(auto first, screen.Score(params));
    JXL_TEST_ASSIGN_OR_DIE(auto cached, screen.Score(params));
    EXPECT_EQ(first, cached);
    EXPECT_EQ(calls, 4u);
    EXPECT_EQ(screen.SamplePixels(), width < 512 ? 4096u : 16384u);
    EXPECT_EQ(pixels, original);
  }
}

TEST(JxlTest, LosslessYieldPromotesMeasuredAutomaticRouteDespiteBadCrops) {
  const char* value = std::getenv("JXL_LOSSLESS_PREFLIGHT_YIELD");
  if (!value || std::strcmp(value, "1") != 0) GTEST_SKIP();
  auto* memory = test::MemoryManager();
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(1025, 769));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  std::vector<uint8_t> pixels(1025 * 769 * 3, 17);
  JxlEncoderChunkedFrameAdapter input(1025, 769, 0);
  ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                  {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}));
  size_t previous = std::numeric_limits<size_t>::max();
  for (int effort : {3, 4, 7, 9, 10}) {
    SCOPED_TRACE(effort);
    CompressParams params;
    params.SetLossless();
    params.speed_tier = static_cast<SpeedTier>(10 - effort);
    size_t automatic_deep_calls = 0;
    const auto probe = [](const CompressParams& p, const CodecMetadata&,
                          JxlEncoderChunkedFrameAdapter&) -> StatusOr<size_t> {
      // Crops wrongly prefer every explicit alternative to automatic e9.
      return size_t{p.options.predictor == kUndefinedPredictor ? 500u : 100u};
    };
    const auto encode = [&](const CompressParams& p, ThreadPool*,
                            JxlEncoderOutputProcessorWrapper* out, AuxOut*) -> Status {
      size_t bytes = 1150;
      if (p.options.predictor == kUndefinedPredictor &&
          p.modular_group_size_shift == -1 && p.options.nb_repeats == 0.5f) {
        const int backend = LosslessSearchLevel(p);
        bytes = backend <= 3 ? 1500 : backend <= 7 ? 1000 : backend == 8 ? 950 : 700;
        if (backend == 9) ++automatic_deep_calls;
      }
      return AppendData(*out, std::vector<uint8_t>(bytes, 0));
    };
    JxlEncoderOutputProcessorWrapper output(memory);
    JXL_TEST_ASSIGN_OR_DIE(bool searched, EncodeLosslessSearch(
        memory, params, FrameInfo{}, metadata, input, nullptr, &output,
        nullptr, encode, probe));
    ASSERT_TRUE(searched);
    ASSERT_TRUE(output.SetFinalizedPosition());
    std::vector<uint8_t> bytes;
    ASSERT_TRUE(output.CopyOutput(bytes));
    EXPECT_LE(bytes.size(), previous);
    previous = bytes.size();
    EXPECT_EQ(bytes.size(), effort >= 4 ? 700u : 1000u);
    EXPECT_EQ(automatic_deep_calls, effort >= 4 ? 1u : 0u);
  }
}

TEST(JxlTest, LosslessCompactFlatBackendStillMeasuresE9) {
  const char* value = std::getenv("JXL_LOSSLESS_PREFLIGHT_YIELD");
  if (!value || std::strcmp(value, "1") != 0) GTEST_SKIP();
  auto* memory = test::MemoryManager();
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(512, 256));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  std::vector<uint8_t> pixels(512 * 256 * 3, 17);
  JxlEncoderChunkedFrameAdapter input(512, 256, 0);
  ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                  {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}));
  for (int effort : {3, 4, 9, 10}) {
    SCOPED_TRACE(effort);
    CompressParams params;
    params.SetLossless();
    params.speed_tier = static_cast<SpeedTier>(10 - effort);
    size_t anchors = 0;
    const auto probe = [](const CompressParams& p, const CodecMetadata&,
                          JxlEncoderChunkedFrameAdapter&) -> StatusOr<size_t> {
      return size_t{p.options.predictor == kUndefinedPredictor ? 500u : 100u};
    };
    const auto encode = [&](const CompressParams& p, ThreadPool*,
                            JxlEncoderOutputProcessorWrapper* out, AuxOut*) -> Status {
      size_t bytes = 1150;
      if (p.options.predictor == kUndefinedPredictor &&
          p.modular_group_size_shift == -1 && p.options.nb_repeats == 0.5f) {
        const int backend = LosslessSearchLevel(p);
        bytes = backend <= 3 ? 1500 : backend <= 8 ? 1000 : 700;
        if (backend == 9) ++anchors;
      }
      return AppendData(*out, std::vector<uint8_t>(bytes, 0));
    };
    JxlEncoderOutputProcessorWrapper output(memory);
    JXL_TEST_ASSIGN_OR_DIE(bool searched, EncodeLosslessSearch(
        memory, params, FrameInfo{}, metadata, input, nullptr, &output,
        nullptr, encode, probe));
    ASSERT_TRUE(searched);
    EXPECT_EQ(output.CurrentPosition(), effort >= 4 ? 700u : 1000u);
    EXPECT_EQ(anchors, effort >= 4 ? 1u : 0u);
  }
}

TEST(JxlTest, LosslessContextScreenCachesUnchangedUint16Regions) {
  const char* value = std::getenv("JXL_LOSSLESS_PREFLIGHT_YIELD");
  if (!value || std::strcmp(value, "1") != 0) GTEST_SKIP();
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(512, 256));
  metadata.m.SetUintSamples(16);
  metadata.m.SetAlphaBits(16);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  std::vector<uint16_t> pixels(512 * 256 * 4);
  for (size_t y = 0; y < 256; ++y) {
    for (size_t x = 0; x < 512; ++x) {
      pixels[(y * 512 + x) * 4] = x;
      pixels[(y * 512 + x) * 4 + 1] = y;
      pixels[(y * 512 + x) * 4 + 2] = 65535;
    }
  }
  const auto original = pixels;
  JxlEncoderChunkedFrameAdapter input(512, 256, 1);
  ASSERT_TRUE(input.SetFromBuffer(0, reinterpret_cast<const uint8_t*>(pixels.data()),
                                  pixels.size() * sizeof(uint16_t),
                                  {4, JXL_TYPE_UINT16, JXL_NATIVE_ENDIAN, 0}));
  std::vector<std::pair<size_t, size_t>> starts;
  const auto probe = [&](const CompressParams&, const CodecMetadata& m,
                          JxlEncoderChunkedFrameAdapter& sample) -> StatusOr<size_t> {
    EXPECT_EQ(sample.xsize, 128u);
    EXPECT_EQ(sample.ysize, 128u);
    EXPECT_EQ(m.m.bit_depth.bits_per_sample, 16u);
    auto source = sample.GetInputSource();
    size_t stride;
    auto buffer = GetColorBuffer(source, 0, 0, 128, 128, &stride);
    const auto* data = static_cast<const uint16_t*>(buffer.get());
    const size_t x0 = data[0], y0 = data[1];
    starts.emplace_back(x0, y0);
    for (size_t y = 0; y < 128; ++y) {
      const auto* row = reinterpret_cast<const uint16_t*>(
          static_cast<const uint8_t*>(buffer.get()) + y * stride);
      for (size_t x = 0; x < 128; ++x) {
        EXPECT_EQ(row[4 * x], x0 + x);
        EXPECT_EQ(row[4 * x + 1], y0 + y);
        EXPECT_EQ(row[4 * x + 2], 65535);
        EXPECT_EQ(row[4 * x + 3], 0);
      }
    }
    return size_t{100};
  };
  CompressParams params;
  params.SetLossless();
  LosslessCandidateScreen screen(metadata, input, nullptr, probe, nullptr, 128);
  ASSERT_TRUE(screen.Available());
  JXL_TEST_ASSIGN_OR_DIE(auto first, screen.Score(params));
  JXL_TEST_ASSIGN_OR_DIE(auto cached, screen.Score(params));
  EXPECT_EQ(first, cached);
  EXPECT_EQ(screen.Encodes(), 3u);
  EXPECT_EQ(screen.SamplePixels(), 3u * 128 * 128);
  EXPECT_EQ(starts, (std::vector<std::pair<size_t, size_t>>{{0, 0}, {192, 64}, {384, 128}}));
  EXPECT_EQ(pixels, original);
}

TEST(JxlTest, LosslessContextFeedbackKeepsLocalGainsAndReplaysDeferrals) {
  const char* value = std::getenv("JXL_LOSSLESS_PREFLIGHT_YIELD");
  if (!value || std::strcmp(value, "1") != 0) GTEST_SKIP();
  auto* memory = test::MemoryManager();
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(512, 256));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  std::vector<uint8_t> pixels(512 * 256 * 3, 17);
  JxlEncoderChunkedFrameAdapter input(512, 256, 0);
  ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                  {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}));
  for (bool local_gain : {false, true}) {
    for (int effort : {9, 10}) {
      SCOPED_TRACE(testing::Message() << "local_gain=" << local_gain
                                     << " effort=" << effort);
      CompressParams params;
      params.SetLossless();
      params.speed_tier = static_cast<SpeedTier>(10 - effort);
      size_t raw_calls = 0;
      const auto target = [](const CompressParams& p) {
        return p.palette_colors == 0 && p.options.predictor == Predictor::Variable;
      };
      const auto probe = [&](const CompressParams& p, const CodecMetadata&,
                              JxlEncoderChunkedFrameAdapter& sample) -> StatusOr<size_t> {
        if (p.options.predictor == kUndefinedPredictor) return size_t{100};
        if (!target(p)) return size_t{150};
        return size_t{sample.ysize == 128 ? (local_gain ? 90u : 100u) : 70u};
      };
      const auto encode = [&](const CompressParams& p, ThreadPool*,
                              JxlEncoderOutputProcessorWrapper* out, AuxOut*) -> Status {
        size_t bytes = 1100;
        if (p.options.predictor == kUndefinedPredictor) {
          bytes = LosslessSearchLevel(p) <= 3 ? 1500 :
                  LosslessSearchLevel(p) < 9 ? 1000 : 900;
        } else if (target(p) && LosslessSearchLevel(p) >= 9) {
          ++raw_calls;
          bytes = 800;
        }
        return AppendData(*out, std::vector<uint8_t>(bytes, 0));
      };
      JxlEncoderOutputProcessorWrapper output(memory);
      JXL_TEST_ASSIGN_OR_DIE(bool searched, EncodeLosslessSearch(
          memory, params, FrameInfo{}, metadata, input, nullptr, &output,
          nullptr, encode, probe));
      ASSERT_TRUE(searched);
      EXPECT_EQ(output.CurrentPosition(), local_gain || effort == 10 ? 800u : 900u);
      if (local_gain || effort == 10) EXPECT_GT(raw_calls, 0u);
      else EXPECT_EQ(raw_calls, 0u);
    }
  }
}

TEST(JxlTest, LosslessPeerContextDefersConsistentLossAndRecoversAtE10) {
  const char* value = std::getenv("JXL_LOSSLESS_PREFLIGHT_YIELD");
  if (!value || std::strcmp(value, "1") != 0) GTEST_SKIP();
  auto* memory = test::MemoryManager();
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(2049, 1025));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  std::vector<uint8_t> pixels(2049 * 1025 * 3, 17);
  JxlEncoderChunkedFrameAdapter input(2049, 1025, 0);
  ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                  {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}));
  for (bool context_loses : {false, true}) {
    for (int effort : {9, 10}) {
      SCOPED_TRACE(testing::Message() << context_loses << " " << effort);
      CompressParams params;
      params.SetLossless();
      params.speed_tier = static_cast<SpeedTier>(10 - effort);
      size_t automatic_calls = 0;
      const auto probe = [&](const CompressParams& p, const CodecMetadata&,
                              JxlEncoderChunkedFrameAdapter& sample) -> StatusOr<size_t> {
        if (p.options.predictor != kUndefinedPredictor) return size_t{300};
        if (LosslessSearchLevel(p) < 9) return size_t{150};
        if (sample.ysize == 128 && sample.xsize == 128) {
          return size_t{p.modular_group_size_shift == -1 ?
                           (context_loses ? 105u : 90u) : 100u};
        }
        // One favorable strip admits the candidate; the independent square
        // context then decides whether its overall deficit is consistent.
        return size_t{p.modular_group_size_shift == -1 ?
                         (sample.ysize == 64 ? 90u : 115u) : 99u};
      };
      const auto encode = [&](const CompressParams& p, ThreadPool*,
                              JxlEncoderOutputProcessorWrapper* out, AuxOut*) -> Status {
        size_t bytes = 3000;
        if (p.options.predictor == kUndefinedPredictor) {
          const int backend = LosslessSearchLevel(p);
          bytes = backend <= 3 ? 1500 : backend < 9 ? 1000 : 800;
          if (p.modular_group_size_shift == -1 && backend >= 9) {
            ++automatic_calls;
            bytes = 700;
          }
        }
        return AppendData(*out, std::vector<uint8_t>(bytes, 0));
      };
      JxlEncoderOutputProcessorWrapper output(memory);
      JXL_TEST_ASSIGN_OR_DIE(bool searched, EncodeLosslessSearch(
          memory, params, FrameInfo{}, metadata, input, nullptr, &output,
          nullptr, encode, probe));
      ASSERT_TRUE(searched);
      EXPECT_EQ(output.CurrentPosition(), context_loses && effort == 9 ? 800u : 700u);
      if (context_loses && effort == 9) EXPECT_EQ(automatic_calls, 0u);
      else EXPECT_GT(automatic_calls, 0u);
    }
  }
}

TEST(JxlTest, LosslessYieldSmallAnchorAndStrongChallengerAreNotVetoed) {
  const char* value = std::getenv("JXL_LOSSLESS_PREFLIGHT_YIELD");
  if (!value || std::strcmp(value, "1") != 0) GTEST_SKIP();
  auto* memory = test::MemoryManager();
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(500, 500));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  std::vector<uint8_t> pixels(500 * 500 * 3, 17);
  JxlEncoderChunkedFrameAdapter input(500, 500, 0);
  ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                  {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}));
  size_t previous = std::numeric_limits<size_t>::max();
  for (int effort : {3, 4, 5, 7, 8, 9, 10}) {
    SCOPED_TRACE(effort);
    CompressParams params;
    params.SetLossless();
    params.speed_tier = static_cast<SpeedTier>(10 - effort);
    size_t anchors = 0, cheap_raw = 0, deep_raw = 0;
    const auto probe = [](const CompressParams& p, const CodecMetadata&,
                          JxlEncoderChunkedFrameAdapter&) -> StatusOr<size_t> {
      if (p.options.predictor == kUndefinedPredictor)
        return size_t{LosslessSearchLevel(p) >= 9 ? 125u : 100u};
      if (p.options.predictor == Predictor::Variable && p.palette_colors == 0)
        return size_t{LosslessSearchLevel(p) <= 6 ? 90u : 80u};
      return size_t{500};
    };
    const auto encode = [&](const CompressParams& p, ThreadPool*,
                            JxlEncoderOutputProcessorWrapper* out, AuxOut*) -> Status {
      const int backend = LosslessSearchLevel(p);
      size_t bytes = 3000;
      if (p.options.predictor == kUndefinedPredictor) {
        bytes = backend == 3 ? 1200 : 1000;
        if (backend == 9 && p.modular_group_size_shift == -1 &&
            p.options.nb_repeats == 0.5f) {
          ++anchors;
          bytes = 700;
        }
      }
      if (p.options.predictor == Predictor::Variable && p.palette_colors == 0) {
        if (backend <= 6) ++cheap_raw;
        if (backend >= 9) ++deep_raw;
        bytes = backend >= 9 ? 600 : 2000;
      }
      return AppendData(*out, std::vector<uint8_t>(bytes, 0));
    };
    JxlEncoderOutputProcessorWrapper output(memory);
    JXL_TEST_ASSIGN_OR_DIE(bool searched, EncodeLosslessSearch(
        memory, params, FrameInfo{}, metadata, input, nullptr, &output,
        nullptr, encode, probe));
    ASSERT_TRUE(searched);
    ASSERT_TRUE(output.SetFinalizedPosition());
    std::vector<uint8_t> bytes;
    ASSERT_TRUE(output.CopyOutput(bytes));
    EXPECT_LE(bytes.size(), previous);
    previous = bytes.size();
    EXPECT_EQ(bytes.size(), effort >= 8 ? 600u : effort >= 5 ? 700u : 1000u);
    EXPECT_EQ(anchors, effort >= 5 ? 1u : 0u);
    EXPECT_EQ(cheap_raw, 0u);
    if (effort >= 8) EXPECT_GT(deep_raw, 0u);
  }
}

TEST(JxlTest, LosslessAnchorOrderKeepsEffortPrefixesAndRejectFallback) {
  for (const char* key : {"JXL_LOSSLESS_PREFLIGHT_YIELD", "JXL_LOSSLESS_ANCHOR_FIRST"}) {
    const char* value = std::getenv(key);
    if (!value || std::strcmp(value, "1") != 0) GTEST_SKIP();
  }
  auto* memory = test::MemoryManager();
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(1537, 1537));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  std::vector<uint8_t> pixels(1537 * 1537 * 3, 17);
  JxlEncoderChunkedFrameAdapter input(1537, 1537, 0);
  ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                  {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}));
  for (bool global_neighbor : {false, true}) {
    for (bool reject : {false, true}) {
      size_t previous = std::numeric_limits<size_t>::max();
      for (int effort : {3, 4, 5, 7, 9, 10}) {
        SCOPED_TRACE(testing::Message()
                     << "reject=" << reject << " effort=" << effort);
        CompressParams params;
        params.SetLossless();
        params.speed_tier = static_cast<SpeedTier>(10 - effort);
        std::vector<CompressParams> calls;
        const auto probe =
            [&](const CompressParams& p, const CodecMetadata&,
                JxlEncoderChunkedFrameAdapter&) -> StatusOr<size_t> {
          if (p.options.predictor != kUndefinedPredictor) return size_t{1000};
          if (p.modular_group_size_shift == 2) return size_t{100};
          if (LosslessSearchLevel(p) < 9) return size_t{reject ? 50u : 100u};
          return size_t{90};
        };
        const auto encode = [&](const CompressParams& p, ThreadPool*,
                                JxlEncoderOutputProcessorWrapper* out,
                                AuxOut*) -> Status {
          calls.push_back(p);
          size_t bytes = 3000;
          if (p.options.predictor == kUndefinedPredictor) {
            bytes = LosslessSearchLevel(p) <= 3 ? 1200 : 1000;
            if (LosslessSearchLevel(p) == 4)
              bytes = global_neighbor && p.modular_group_size_shift == 2 ? 1400
                                                                         : 1500;
            if (LosslessSearchLevel(p) >= 9)
              bytes = p.modular_group_size_shift == 2 ? 850 : 800;
          }
          return AppendData(*out, std::vector<uint8_t>(bytes, 0));
        };
        JxlEncoderOutputProcessorWrapper output(memory);
        JXL_TEST_ASSIGN_OR_DIE(
            bool searched,
            EncodeLosslessSearch(memory, params, FrameInfo{}, metadata, input,
                                 nullptr, &output, nullptr, encode, probe));
        ASSERT_TRUE(searched);
        ASSERT_TRUE(output.SetFinalizedPosition());
        std::vector<uint8_t> bytes;
        ASSERT_TRUE(output.CopyOutput(bytes));
        EXPECT_LE(bytes.size(), previous);
        previous = bytes.size();
        if (!reject && effort >= 5) {
          EXPECT_EQ(bytes.size(), global_neighbor && effort < 10 ? 850u : 800u);
          auto first = std::find_if(
              calls.begin(), calls.end(),
              [](const auto& p) { return LosslessSearchLevel(p) >= 9; });
          ASSERT_NE(first, calls.end());
          EXPECT_EQ(first->modular_group_size_shift, global_neighbor ? 2 : -1);
        }
        if (reject && effort < 10) {
          EXPECT_EQ(bytes.size(), 1000u);
          EXPECT_EQ(std::count_if(calls.begin(), calls.end(),
                                  [](const auto& p) {
                                    return LosslessSearchLevel(p) >= 9;
                                  }),
                    0);
        }
      }
    }
  }
}

TEST(JxlTest, LosslessAnchorDoesNotAddCompactTrials) {
  for (const char* key : {"JXL_LOSSLESS_PREFLIGHT_YIELD", "JXL_LOSSLESS_ANCHOR_FIRST"}) {
    const char* value = std::getenv(key);
    if (!value || std::strcmp(value, "1") != 0) GTEST_SKIP();
  }
  auto* memory = test::MemoryManager();
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(1025, 769));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  std::vector<uint8_t> pixels(1025 * 769 * 3, 17);
  JxlEncoderChunkedFrameAdapter input(1025, 769, 0);
  ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                  {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}));
  for (size_t witness : {1050u, 999u, 998u, 997u, 950u}) {
    CompressParams params;
    params.SetLossless();
    params.speed_tier = static_cast<SpeedTier>(1);
    size_t deep = 0, medium = 0;
    const auto probe = [](const CompressParams& p, const CodecMetadata&,
                          JxlEncoderChunkedFrameAdapter&) -> StatusOr<size_t> {
      if (p.options.predictor != kUndefinedPredictor) return size_t{1000};
      return size_t{p.modular_group_size_shift == 2 ? 99u : 100u};
    };
    const auto encode = [&](const CompressParams& p, ThreadPool*,
                            JxlEncoderOutputProcessorWrapper* out,
                            AuxOut*) -> Status {
      size_t bytes = 3000;
      const int backend = LosslessSearchLevel(p);
      if (p.options.predictor == kUndefinedPredictor) {
        bytes = backend <= 3 ? 1200 : backend >= 9 ? 700 : 1000;
        if (p.modular_group_size_shift == 2) {
          if (backend == 7) {
            ++medium;
            bytes = witness;
          }
          if (backend == 9) {
            ++deep;
            bytes = 650;
          }
        }
      }
      return AppendData(*out, std::vector<uint8_t>(bytes, 0));
    };
    JxlEncoderOutputProcessorWrapper output(memory);
    JXL_TEST_ASSIGN_OR_DIE(
        bool searched,
        EncodeLosslessSearch(memory, params, FrameInfo{}, metadata, input,
                             nullptr, &output, nullptr, encode, probe));
    ASSERT_TRUE(searched);
    ASSERT_TRUE(output.SetFinalizedPosition());
    std::vector<uint8_t> bytes;
    ASSERT_TRUE(output.CopyOutput(bytes));
    EXPECT_EQ(bytes.size(), 700u);
    EXPECT_EQ(medium, 0u);
    EXPECT_EQ(deep, 0u);
  }
}

TEST(JxlTest, LosslessPalettePreselectionDefersUntriedPlans) {
  const char* mode = std::getenv("JXL_LOSSLESS_PALETTE_RACE");
  if (!mode || std::strcmp(mode, "2")) GTEST_SKIP();
  auto* memory = test::MemoryManager();
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(1025, 769));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  std::vector<uint8_t> pixels(1025 * 769 * 3, 17);
  JxlEncoderChunkedFrameAdapter input(1025, 769, 0);
  ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                  {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}));
  for (int effort : {9, 10}) {
    CompressParams params;
    params.SetLossless();
    params.speed_tier = static_cast<SpeedTier>(10 - effort);
    std::array<size_t, 3> full{};
    const auto plan = [](const CompressParams& p) {
      return p.options.nb_repeats < 0.02f ? 0u
          : p.options.nb_repeats < 0.5f ? 1u : 2u;
    };
    const auto probe = [&](const CompressParams& p, const CodecMetadata&,
                            JxlEncoderChunkedFrameAdapter&) -> StatusOr<size_t> {
      if (p.options.predictor != Predictor::Zero) return size_t{1000};
      return size_t{std::array<size_t, 3>{{100, 90, 110}}[plan(p)]};
    };
    const auto encode = [&](const CompressParams& p, ThreadPool*,
                            JxlEncoderOutputProcessorWrapper* out,
                            AuxOut*) -> Status {
      const int backend = LosslessSearchLevel(p);
      size_t bytes = backend <= 3 ? 7000 : 5000;
      if (p.options.predictor == Predictor::Zero && p.palette_colors == 10000) {
        bytes = backend < 9 ? 2000
            : std::array<size_t, 3>{{1600, 1500, 1400}}[plan(p)];
        if (backend == 9) ++full[plan(p)];
      }
      return AppendData(*out, std::vector<uint8_t>(bytes, 0));
    };
    JxlEncoderOutputProcessorWrapper output(memory);
    JXL_TEST_ASSIGN_OR_DIE(bool searched, EncodeLosslessSearch(memory, params,
        FrameInfo{}, metadata, input, nullptr, &output, nullptr, encode, probe));
    ASSERT_TRUE(searched);
    ASSERT_TRUE(output.SetFinalizedPosition());
    std::vector<uint8_t> bytes;
    ASSERT_TRUE(output.CopyOutput(bytes));
    EXPECT_EQ(bytes.size(), effort == 9 ? 1500u : 1400u);
    EXPECT_EQ(full[0], effort == 9 ? 0u : 1u);
    EXPECT_EQ(full[1], 1u);
    EXPECT_EQ(full[2], effort == 9 ? 0u : 1u);
  }
}

TEST(JxlTest, LosslessYieldCompactWeakNeighborhoodWaitsForE10) {
  const char* value = std::getenv("JXL_LOSSLESS_PREFLIGHT_YIELD");
  if (!value || std::strcmp(value, "1") != 0) GTEST_SKIP();
  auto* memory = test::MemoryManager();
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(1025, 769));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  std::vector<uint8_t> pixels(1025 * 769 * 3, 17);
  JxlEncoderChunkedFrameAdapter input(1025, 769, 0);
  ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                  {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}));
  for (int effort : {9, 10}) {
    CompressParams params;
    params.SetLossless();
    params.speed_tier = static_cast<SpeedTier>(10 - effort);
    size_t refinements = 0;
    const auto probe = [](const CompressParams& p, const CodecMetadata&,
                          JxlEncoderChunkedFrameAdapter&) -> StatusOr<size_t> {
      if (p.options.predictor == kUndefinedPredictor)
        return size_t{p.modular_group_size_shift == 2 ? 99u : 100u};
      return size_t{500};
    };
    const auto encode = [&](const CompressParams& p, ThreadPool*,
                            JxlEncoderOutputProcessorWrapper* out, AuxOut*) -> Status {
      const int backend = LosslessSearchLevel(p);
      size_t bytes = 3000;
      if (p.options.predictor == kUndefinedPredictor) {
        bytes = backend <= 3 ? 1200 : backend >= 9 ? 700 : 1000;
        if (backend == 9 && p.modular_group_size_shift == 2) {
          ++refinements;
          bytes = 650;
        }
      }
      return AppendData(*out, std::vector<uint8_t>(bytes, 0));
    };
    JxlEncoderOutputProcessorWrapper output(memory);
    JXL_TEST_ASSIGN_OR_DIE(bool searched, EncodeLosslessSearch(
        memory, params, FrameInfo{}, metadata, input, nullptr, &output,
        nullptr, encode, probe));
    ASSERT_TRUE(searched);
    ASSERT_TRUE(output.SetFinalizedPosition());
    std::vector<uint8_t> bytes;
    ASSERT_TRUE(output.CopyOutput(bytes));
    EXPECT_EQ(bytes.size(), effort == 9 ? 700u : 650u);
    EXPECT_EQ(refinements, effort == 9 ? 0u : 1u);
  }
}

TEST(JxlTest, LosslessYieldChecksLocalOptimismAndRecoversAtE10) {
  for (const char* key : {"JXL_LOSSLESS_PREFLIGHT_YIELD", "JXL_LOSSLESS_SCAN",
                         "JXL_LOSSLESS_CALIBRATE", "JXL_LOSSLESS_CALIBRATE_ROUTES"}) {
    const char* value = std::getenv(key);
    if (!value || std::strcmp(value, "1") != 0) GTEST_SKIP();
  }
  auto* memory = test::MemoryManager();
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(1025, 769));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  std::vector<uint8_t> pixels(1025 * 769 * 3);
  for (size_t y = 0; y < 769; ++y) {
    for (size_t x = 0; x < 1025; ++x) {
      for (size_t c = 0; c < 3; ++c) pixels[3 * (y * 1025 + x) + c] = (x / 8 + y / 8) % 5;
    }
  }
  JxlEncoderChunkedFrameAdapter input(1025, 769, 0);
  ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                  {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}));
  for (bool recovered_winner : {false, true}) {
    size_t previous = std::numeric_limits<size_t>::max();
    for (int effort : {3, 9, 10}) {
      SCOPED_TRACE(testing::Message() << "recovered=" << recovered_winner << " e=" << effort);
      CompressParams params;
      params.SetLossless();
      params.speed_tier = static_cast<SpeedTier>(10 - effort);
      size_t deep_calls = 0, medium_calls = 0;
      const auto probe = [](const CompressParams& p, const CodecMetadata&,
                            JxlEncoderChunkedFrameAdapter& sample) -> StatusOr<size_t> {
        const int backend = LosslessSearchLevel(p);
        if (p.options.predictor == kUndefinedPredictor) {
          if (p.modular_group_size_shift == 2) return size_t{120};
          return size_t{backend >= 9 ? 95u : 100u};
        }
        if (p.options.predictor == Predictor::Variable && p.palette_colors == 0) {
          if (backend <= 6) return size_t{400};
          if (backend == 7) return size_t{120};
          return size_t{sample.xsize == 256 ? 80u : 120u};
        }
        return size_t{1000};
      };
      const auto encode = [&](const CompressParams& p, ThreadPool*,
                              JxlEncoderOutputProcessorWrapper* out, AuxOut*) -> Status {
        const int backend = LosslessSearchLevel(p);
        size_t bytes = 3000;
        if (p.options.predictor == kUndefinedPredictor) {
          bytes = backend <= 3 ? 1500 : backend >= 9 ? 900 : 1000;
          if (p.modular_group_size_shift == 2) bytes = 1200;
        }
        if (p.options.predictor == Predictor::Variable && p.palette_colors == 0) {
          bytes = 1600;
          if (backend == 7) {
            ++medium_calls;
            bytes = 1400;
          } else if (backend >= 9) {
            ++deep_calls;
            bytes = recovered_winner ? 700 : 950;
          }
        }
        return AppendData(*out, std::vector<uint8_t>(bytes, 0));
      };
      JxlEncoderOutputProcessorWrapper output(memory);
      JXL_TEST_ASSIGN_OR_DIE(bool searched, EncodeLosslessSearch(
          memory, params, FrameInfo{}, metadata, input, nullptr, &output,
          nullptr, encode, probe));
      ASSERT_TRUE(searched);
      ASSERT_TRUE(output.SetFinalizedPosition());
      std::vector<uint8_t> bytes;
      ASSERT_TRUE(output.CopyOutput(bytes));
      EXPECT_LE(bytes.size(), previous);
      previous = bytes.size();
      if (effort >= 9) EXPECT_GT(medium_calls, 0u);
      if (effort == 9) EXPECT_EQ(deep_calls, 0u);
      if (effort == 10) EXPECT_GT(deep_calls, 0u);
      EXPECT_EQ(bytes.size(), effort == 3 ? 1000u :
                effort == 10 && recovered_winner ? 700u : 900u);
    }
  }
}

TEST(JxlTest, LosslessSearchExploresWhenSamplesVetoMixedFamilies) {
  auto* memory = test::MemoryManager();
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(513, 513));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  std::vector<uint8_t> pixels(513 * 513 * 3, 17);
  JxlEncoderChunkedFrameAdapter input(513, 513, 0);
  ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                  {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}));
  for (bool ambiguous : {false, true}) {
    for (int effort : {7, 8, 9}) {
      CompressParams params;
      params.SetLossless();
      params.speed_tier = static_cast<SpeedTier>(10 - effort);
      size_t exploratory_calls = 0;
      const auto encode = [&](const CompressParams& p, ThreadPool*,
                              JxlEncoderOutputProcessorWrapper* output,
                              AuxOut*) -> Status {
        size_t bytes = 500;
        if (p.options.predictor == Predictor::Zero && !ambiguous) bytes = 1500;
        if (p.options.predictor == Predictor::Variable) {
          ++exploratory_calls;
          bytes = p.colorspace == 0 ? 200 : 180;
        }
        return AppendData(*output, std::vector<uint8_t>(bytes, 7));
      };
      const auto probe = [](const CompressParams& p, const CodecMetadata&,
                             JxlEncoderChunkedFrameAdapter&) -> StatusOr<size_t> {
        // Every crop rejects cross-channel models, despite their whole-frame
        // benefit in the mock above. Only ambiguous full-frame feedback gets
        // reserved exploratory trials; one decisively better family does not.
        return size_t{p.palette_colors == 1024 ? 100u : 1000u};
      };
      JxlEncoderOutputProcessorWrapper output(memory);
      JXL_TEST_ASSIGN_OR_DIE(
          bool searched,
          EncodeLosslessSearch(memory, params, FrameInfo{}, metadata, input,
                               nullptr, &output, nullptr, encode, probe));
      ASSERT_TRUE(searched);
      ASSERT_TRUE(output.SetFinalizedPosition());
      std::vector<uint8_t> bytes;
      ASSERT_TRUE(output.CopyOutput(bytes));
      const size_t expected = !ambiguous || effort == 7 ? 500 :
                              effort == 8 ? 200 : 180;
      EXPECT_EQ(bytes.size(), expected);
      EXPECT_EQ(exploratory_calls == 0, !ambiguous || effort == 7);
    }
  }
}

TEST(JxlTest, LosslessTrialBudgetSaturates) {
  LosslessTrialBudget zero(0);
  EXPECT_TRUE(zero.Reached());
  const size_t maximum = std::numeric_limits<size_t>::max();
  LosslessTrialBudget large(maximum);
  large.AddBytes(maximum - 1);
  EXPECT_FALSE(large.Reached());
  large.AddBytes(2);
  EXPECT_TRUE(large.Reached());
  large.AddBytes(maximum);
  EXPECT_TRUE(large.Reached());
  LosslessTrialBudget parallel(1025);
  ThreadPoolForTests pool(4);
  ASSERT_TRUE(RunOnPool(pool.get(), 0, 1024, ThreadPool::NoInit,
                        [&](uint32_t, size_t) -> Status {
                          parallel.AddBytes(1);
                          return true;
                        }, "LosslessBudgetTest"));
  EXPECT_FALSE(parallel.Reached());
  parallel.AddBytes(1);
  EXPECT_TRUE(parallel.Reached());
}

TEST(JxlTest, LosslessTwoPassPreservesCumulativeCandidates) {
  JxlMemoryManager* memory = test::MemoryManager();
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(513, 513));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  std::vector<uint8_t> pixels(513 * 513 * 3, 17);
  JxlEncoderChunkedFrameAdapter input(513, 513, 0);
  ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                  {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}));
  FrameInfo frame_info;
  // A good sample estimate can still lose on the full image. Higher efforts
  // retain their lower levels' candidates, regardless of misleading probes.
  for (bool misleading : {false, true}) {
    std::vector<uint8_t> serial_output;
    for (int order : {0, 19, 71}) {
      FakeParallelRunner runner(order, 4);
      ThreadPool pool(&JxlFakeParallelRunner, &runner);
      std::vector<std::tuple<int, int, int, int, bool>> previous_calls;
      size_t previous_size = 0;
      for (int effort = 7; effort <= 10; ++effort) {
        CompressParams params;
        params.SetLossless();
        params.speed_tier = static_cast<SpeedTier>(10 - effort);
        std::mutex mutex;
        std::vector<std::tuple<int, int, int, int, bool>> calls;
        bool legacy_e10_seen = false;
        const auto encode = [&](const CompressParams& p, ThreadPool*,
                                JxlEncoderOutputProcessorWrapper* output,
                                AuxOut*) -> Status {
          {
            std::lock_guard<std::mutex> lock(mutex);
            calls.emplace_back(static_cast<int>(p.speed_tier),
                               static_cast<int>(p.options.predictor),
                               p.palette_colors, p.colorspace,
                               p.lz77_careful_first_pass);
            legacy_e10_seen |= p.speed_tier == SpeedTier::kGlacier &&
                              !p.lz77_careful_first_pass;
          }
          size_t size = 500;
          if (p.colorspace == 0 && p.options.predictor == Predictor::Gradient)
            size = misleading ? 900 : 300;
          if (p.colorspace == 6 && p.options.predictor == Predictor::Gradient)
            size = misleading ? 800 : 250;
          if (p.colorspace == 0 && p.options.predictor == Predictor::Weighted)
            size = misleading ? 700 : 280;
          if (p.lz77_careful_first_pass) size = misleading ? 600 : 200;
          if (p.lossless_trial_budget) {
            p.lossless_trial_budget->AddBytes(size);
            if (p.lossless_trial_budget->Reached()) {
              // Deliberately leave a tiny, invalid partial output. It must not
              // displace the real winner or change later candidate selection.
              return AppendData(*output, std::vector<uint8_t>(1, 99));
            }
          }
          return AppendData(*output, std::vector<uint8_t>(size, 7));
        };
        const auto probe = [](const CompressParams& p, const CodecMetadata&,
                               JxlEncoderChunkedFrameAdapter&) -> StatusOr<size_t> {
          if (p.lz77_careful_first_pass) return size_t{50};
          if (p.colorspace == 0 && p.options.predictor == Predictor::Gradient)
            return size_t{100};
          if (p.colorspace == 6 && p.options.predictor == Predictor::Gradient)
            return size_t{110};
          if (p.colorspace == 0 && p.options.predictor == Predictor::Weighted)
            return size_t{120};
          return size_t{1000};
        };
        JxlEncoderOutputProcessorWrapper output(memory);
        JXL_TEST_ASSIGN_OR_DIE(bool searched, EncodeLosslessSearch(
            memory, params, frame_info, metadata, input,
            order == 0 ? nullptr : &pool, &output, nullptr, encode, probe));
        ASSERT_TRUE(searched);
        ASSERT_TRUE(output.SetFinalizedPosition());
        std::vector<uint8_t> bytes;
        ASSERT_TRUE(output.CopyOutput(bytes));
        if (previous_size) EXPECT_LE(bytes.size(), previous_size);
        previous_size = bytes.size();
        std::sort(calls.begin(), calls.end());
        EXPECT_TRUE(std::includes(calls.begin(), calls.end(),
                                   previous_calls.begin(), previous_calls.end()));
        previous_calls = calls;
        if (misleading) EXPECT_EQ(bytes.size(), 500u);
        if (effort == 10) {
          EXPECT_TRUE(legacy_e10_seen);
          EXPECT_EQ(bytes.size(), misleading ? 500u : 200u);
          if (order == 0) serial_output = bytes;
          else EXPECT_EQ(bytes, serial_output);
        }
      }
    }
  }
}

TEST(JxlTest, LosslessSharedScreenGatesRoutesAndRetainsCompleteWinners) {
  constexpr size_t width = 1025, height = 513;
  auto* memory = jxl::test::MemoryManager();
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(width, height));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  std::vector<uint8_t> pixels(width * height * 3, 42);
  JxlEncoderChunkedFrameAdapter input(width, height, 0);
  ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                  {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}));
  FrameInfo frame_info;
  const auto promising = [](const CompressParams& p) {
    return p.colorspace == 0 && p.options.predictor == Predictor::Variable &&
           p.modular_group_size_shift == 3 &&
           p.speed_tier == SpeedTier::kTortoise &&
           p.options.nb_repeats == 0.1f;
  };
  for (bool misleading : {false, true}) {
    for (int order : {0, 19}) {
      FakeParallelRunner runner(order, 4);
      ThreadPool pool(&JxlFakeParallelRunner, &runner);
      std::vector<CompressParams> previous_calls;
      size_t previous_size = std::numeric_limits<size_t>::max();
      for (int effort = 1; effort <= 10; ++effort) {
        CompressParams params;
        params.SetLossless();
        params.speed_tier = static_cast<SpeedTier>(10 - effort);
        std::vector<CompressParams> calls;
        const auto encode = [&](const CompressParams& p, ThreadPool*,
                                JxlEncoderOutputProcessorWrapper* output,
                                AuxOut*) -> Status {
          calls.push_back(p);
          size_t size = 0;
          if (LosslessSearchLevel(p) == 3) size = 120;
          if (LosslessSearchLevel(p) == 6 &&
              p.options.predictor == Predictor::Zero) size = 110;
          if (LosslessSearchLevel(p) == 7 &&
              p.modular_group_size_shift == -1) size = 100;
          if (promising(p)) size = misleading ? 150 : 80;
          EXPECT_NE(size, 0u) << "A dominated expensive route reached full encode";
          if (p.lossless_trial_budget) {
            p.lossless_trial_budget->AddBytes(size);
            if (p.lossless_trial_budget->Reached()) {
              return AppendData(*output, std::vector<uint8_t>(1, 99));
            }
          }
          return AppendData(*output, std::vector<uint8_t>(size, 7));
        };
        const auto probe = [&](const CompressParams& p, const CodecMetadata&,
                               JxlEncoderChunkedFrameAdapter&) -> StatusOr<size_t> {
          if (promising(p)) return size_t{90};
          if (LosslessSearchLevel(p) == 7 &&
              p.modular_group_size_shift == -1) return size_t{100};
          return size_t{200};
        };
        JxlEncoderOutputProcessorWrapper output(memory);
        JXL_TEST_ASSIGN_OR_DIE(bool searched, EncodeLosslessSearch(
            memory, params, frame_info, metadata, input,
            order == 0 ? nullptr : &pool, &output, nullptr, encode, probe));
        ASSERT_TRUE(searched);
        ASSERT_TRUE(output.SetFinalizedPosition());
        std::vector<uint8_t> bytes;
        ASSERT_TRUE(output.CopyOutput(bytes));
        EXPECT_LE(bytes.size(), previous_size);
        previous_size = bytes.size();
        ASSERT_GE(calls.size(), previous_calls.size());
        for (size_t i = 0; i < previous_calls.size(); ++i) {
          EXPECT_TRUE(SameLosslessCandidate(calls[i], previous_calls[i]));
        }
        previous_calls = calls;
        if (effort >= 8) {
          EXPECT_EQ(calls.size(), 4u);
          EXPECT_EQ(bytes.size(), misleading ? 100u : 80u);
        }
      }
    }
  }
}

TEST(JxlTest, LosslessConfirmationRejectsMisleadingSmallSamples) {
  constexpr size_t width = 1025, height = 513;
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(width, height));
  metadata.m.SetUintSamples(16);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  metadata.m.SetAlphaBits(16);
  std::vector<uint16_t> pixels(width * height * 4);
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      auto* p = pixels.data() + 4 * (y * width + x);
      p[0] = x;
      p[1] = y;
      p[2] = x + y;
      p[3] = x < 700 ? 0 : 12345;
    }
  }
  JxlEncoderChunkedFrameAdapter input(width, height, 1);
  ASSERT_TRUE(input.SetFromBuffer(0, reinterpret_cast<const uint8_t*>(pixels.data()), pixels.size() * 2,
                                  {4, JXL_TYPE_UINT16, JXL_NATIVE_ENDIAN, 0}));
  CompressParams params;
  params.SetLossless();
  params.speed_tier = SpeedTier::kTortoise;
  auto incumbent = params;
  incumbent.colorspace = 6;
  incumbent.options.nb_repeats = 0.5f;
  const auto encode = [&](const CompressParams& p, const CodecMetadata& m,
                          JxlEncoderChunkedFrameAdapter& sample)
      -> StatusOr<size_t> {
    EXPECT_EQ(m.m.bit_depth.bits_per_sample, 16u);
    auto source = sample.GetInputSource();
    size_t stride;
    auto buffer = GetColorBuffer(source, 0, 0, sample.xsize, sample.ysize, &stride);
    const auto* row = static_cast<const uint16_t*>(buffer.get());
    EXPECT_EQ(row[2], row[0] + row[1]);
    EXPECT_EQ(row[3], row[0] < 700 ? 0 : 12345);
    const bool candidate = p.colorspace == 0 && p.options.predictor == Predictor::Gradient;
    if (sample.xsize == 64) return size_t{candidate ? 100u : 1000u};
    EXPECT_EQ(sample.xsize, 512u);
    EXPECT_EQ(sample.ysize, 512u);
    EXPECT_TRUE(candidate || p.colorspace == 6);
    return size_t{candidate ? 2000u : 1000u};
  };
  for (size_t threads : {0, 4}) {
    ThreadPoolForTests pool(threads);
    JXL_TEST_ASSIGN_OR_DIE(auto ranked, ProbeLosslessCandidates(
        params, metadata, input, pool.get(), encode));
    ASSERT_FALSE(ranked.ranked.empty());
    EXPECT_STREQ(ranked.ranked[0].name, "raw-gradient");
    JXL_TEST_ASSIGN_OR_DIE(auto result, ConfirmLosslessCandidate(
        ranked.ranked[0].params, incumbent, metadata, input, pool.get(), encode));
    EXPECT_EQ(result.samples, 3u);
    EXPECT_EQ(result.candidate_bytes, 6000u);
    EXPECT_EQ(result.incumbent_bytes, 3000u);
    EXPECT_FALSE(result.accepted);
    std::atomic<size_t> cached_calls{0};
    const auto cached_encode = [&](const CompressParams& p, const CodecMetadata& m,
                                    JxlEncoderChunkedFrameAdapter& sample)
        -> StatusOr<size_t> {
      ++cached_calls;
      EXPECT_EQ(p.colorspace, 0);
      return encode(p, m, sample);
    };
    JXL_TEST_ASSIGN_OR_DIE(auto cached, ConfirmLosslessCandidate(
        ranked.ranked[0].params, incumbent, metadata, input, pool.get(),
        cached_encode, result.incumbent_bytes));
    EXPECT_EQ(cached_calls.load(), 3u);
    EXPECT_EQ(cached.candidate_bytes, result.candidate_bytes);
    EXPECT_EQ(cached.incumbent_bytes, result.incumbent_bytes);
    EXPECT_EQ(cached.accepted, result.accepted);
  }
}

TEST(JxlTest, LosslessWinnerNeighborhoodUsesWholeFrameFeedback) {
  JxlMemoryManager* memory = test::MemoryManager();
  CodecMetadata metadata;
  ASSERT_TRUE(metadata.size.Set(513, 513));
  metadata.m.SetUintSamples(8);
  metadata.m.color_encoding = ColorEncoding::SRGB();
  std::vector<uint8_t> pixels(513 * 513 * 3, 17);
  JxlEncoderChunkedFrameAdapter input(513, 513, 0);
  ASSERT_TRUE(input.SetFromBuffer(0, pixels.data(), pixels.size(),
                                  {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0}));
  for (int scenario = 0; scenario < 4; ++scenario) {
    const bool refinement_wins = (scenario & 1) != 0;
    const bool literal_wins = (scenario & 2) != 0;
    for (int effort : {8, 9, 10}) {
      CompressParams params;
      params.SetLossless();
      params.speed_tier = static_cast<SpeedTier>(10 - effort);
      int neighbors = 0;
      const auto encode = [&](const CompressParams& p, ThreadPool*,
                              JxlEncoderOutputProcessorWrapper* output,
                              AuxOut*) -> Status {
        size_t bytes = 500;
        if (p.options.predictor == Predictor::Gradient) {
          if (p.colorspace == 0) bytes = 300;
          if (p.colorspace == 6) bytes = literal_wins ? 400 : 250;
        }
        if (literal_wins && p.options.predictor == Predictor::Zero) bytes = 250;
        const bool is_neighbor =
            literal_wins ? p.options.nb_repeats == 0.03f
                         : (p.modular_group_size_shift == 2 &&
                            p.options.max_properties == 2 &&
                            p.options.predictor == Predictor::Gradient);
        if (is_neighbor) {
          ++neighbors;
          if (literal_wins) {
            EXPECT_EQ(p.modular_group_size_shift, 3);
            EXPECT_EQ(p.options.predictor, Predictor::Zero);
            EXPECT_EQ(p.palette_colors, 10000);
            EXPECT_EQ(p.options.max_properties, 0);
          } else {
            // Best crop: raw-gradient. Whole-frame winner: YCoCg-gradient.
            EXPECT_EQ(p.colorspace, 6);
            EXPECT_EQ(p.options.predictor, Predictor::Gradient);
            EXPECT_EQ(p.palette_colors, 0);
            EXPECT_EQ(p.channel_colors_pre_transform_percent, 0.f);
            EXPECT_EQ(p.options.nb_repeats, 0.1f);
          }
          EXPECT_EQ(p.speed_tier, SpeedTier::kTortoise);
          EXPECT_NE(p.lossless_trial_budget, nullptr);
          bytes = refinement_wins ? 200 : 600;
        }
        if (p.lossless_trial_budget) {
          p.lossless_trial_budget->AddBytes(bytes);
          if (p.lossless_trial_budget->Reached()) {
            return AppendData(*output, std::vector<uint8_t>(1, 99));
          }
        }
        return AppendData(*output, std::vector<uint8_t>(bytes, 7));
      };
      const auto probe =
          [](const CompressParams& p, const CodecMetadata&,
             JxlEncoderChunkedFrameAdapter&) -> StatusOr<size_t> {
        if (p.options.predictor == Predictor::Gradient) {
          if (p.colorspace == 0) return size_t{100};
          if (p.colorspace == 6) return size_t{110};
        }
        return size_t{1000};
      };
      JxlEncoderOutputProcessorWrapper output(memory);
      JXL_TEST_ASSIGN_OR_DIE(
          bool searched,
          EncodeLosslessSearch(memory, params, FrameInfo{}, metadata, input,
                               nullptr, &output, nullptr, encode, probe));
      ASSERT_TRUE(searched);
      ASSERT_TRUE(output.SetFinalizedPosition());
      std::vector<uint8_t> bytes;
      ASSERT_TRUE(output.CopyOutput(bytes));
      // e5 already tries the literal refinement; neither a completed nor a
      // bounded copy may be encoded again at e9.
      EXPECT_EQ(neighbors, literal_wins || effort >= 9 ? 1 : 0);
      const size_t expected =
          effort == 8 && !literal_wins ? 300u : (refinement_wins ? 200u : 250u);
      EXPECT_EQ(bytes.size(), expected);
    }
  }
}

TEST(JxlTest, LosslessTreeScoringRoundtrip) {
  auto* memory = test::MemoryManager();
  struct RestoreTargets {
    ~RestoreTargets() { hwy::SetSupportedTargetsForTest(0); }
  } restore_targets;
  constexpr size_t width = 257, height = 131;
  for (int64_t target : hwy::SupportedAndGeneratedTargets()) {
    hwy::SetSupportedTargetsForTest(target);
    SCOPED_TRACE(hwy::TargetName(target));
    for (int pattern = 0; pattern < 5; ++pattern) {
      JXL_TEST_ASSIGN_OR_DIE(Image image,
                             Image::Create(memory, width, height, 16, 4));
      for (size_t c = 0; c < 4; ++c) {
        for (size_t y = 0; y < height; ++y) {
          for (size_t x = 0; x < width; ++x) {
            image.channel[c].Row(y)[x] =
                pattern == 0   ? 17
                : pattern == 1 ? (x / 8 + y / 8 + c) % 2 * 65535
                : pattern == 2
                    ? (x * 257 + y * 31 + c * 1000) % 65536
                    : (x * 19391 + y * 311 + x * y * 1777 + c) % 65536;
            if (pattern == 4) {
              image.channel[c].Row(y)[x] =
                  static_cast<int>((x * 373 + y * 929 + c * 17) % 65536) -
                  32768;
            }
          }
        }
      }
      for (Predictor predictor :
           {Predictor::Zero, Predictor::Best, Predictor::Variable}) {
        for (float fraction : {0.0f, 0.01f, 0.5f, 1.0f}) {
          SCOPED_TRACE(testing::Message()
                       << pattern << ":" << static_cast<int>(predictor) << ":"
                       << fraction);
          ModularOptions options;
          options.predictor = predictor;
          options.nb_repeats = fraction;
          for (int variant = 0; variant < 3; ++variant) {
            SCOPED_TRACE(testing::Message() << "variant=" << variant);
            options.lossless_tree_speedups = variant >= 1;
            options.lossless_fast_entropy = variant == 2;
            BitWriter writer(memory);
            ASSERT_TRUE(ModularGenericCompress(image, options, writer));
            writer.ZeroPadToByte();
            JXL_TEST_ASSIGN_OR_DIE(Image decoded,
                                   Image::Create(memory, width, height, 16, 4));
            Status status = true;
            {
              BitReader reader(writer.GetSpan());
              BitReaderScopedCloser closer(reader, status);
              ASSERT_TRUE(ModularGenericDecompress(&reader, decoded, nullptr, 0,
                                                   &options));
            }
            ASSERT_TRUE(status);
            ASSERT_EQ(decoded.channel.size(), image.channel.size());
            for (size_t c = 0; c < 4; ++c) {
              for (size_t y = 0; y < height; ++y) {
                EXPECT_TRUE(std::equal(image.channel[c].Row(y),
                                       image.channel[c].Row(y) + width,
                                       decoded.channel[c].Row(y)));
              }
            }
          }
        }
      }
    }
  }
}

// Compare stored samples, including invisible RGB and separate extra planes.
// Compressed sizes and representations are optimization results, not goldens.
void ExpectExactLosslessSamples(const PackedPixelFile& input,
                                const PackedPixelFile& output) {
  ASSERT_EQ(input.info.xsize, output.info.xsize);
  ASSERT_EQ(input.info.ysize, output.info.ysize);
  ASSERT_EQ(input.info.bits_per_sample, output.info.bits_per_sample);
  ASSERT_EQ(input.info.exponent_bits_per_sample,
            output.info.exponent_bits_per_sample);
  ASSERT_EQ(input.info.num_color_channels, output.info.num_color_channels);
  ASSERT_EQ(input.info.alpha_bits, output.info.alpha_bits);
  ASSERT_EQ(input.info.alpha_exponent_bits, output.info.alpha_exponent_bits);
  ASSERT_EQ(input.info.alpha_premultiplied, output.info.alpha_premultiplied);
  ASSERT_EQ(input.frames.size(), output.frames.size());
  const auto compare = [](const extras::PackedImage& a,
                          const extras::PackedImage& b) {
    ASSERT_EQ(a.xsize, b.xsize);
    ASSERT_EQ(a.ysize, b.ysize);
    ASSERT_EQ(a.format.num_channels, b.format.num_channels);
    ASSERT_EQ(a.format.data_type, b.format.data_type);
    ASSERT_EQ(a.format.endianness, b.format.endianness);
    const size_t row_bytes = a.xsize * a.pixel_stride();
    ASSERT_LE(row_bytes, a.stride);
    ASSERT_LE(row_bytes, b.stride);
    for (size_t y = 0; y < a.ysize; ++y) {
      EXPECT_EQ(0, memcmp(a.const_pixels(y, 0, 0), b.const_pixels(y, 0, 0),
                          row_bytes))
          << "row=" << y;
    }
  };
  for (size_t i = 0; i < input.frames.size(); ++i) {
    SCOPED_TRACE(testing::Message() << "frame=" << i);
    compare(input.frames[i].color, output.frames[i].color);
    ASSERT_EQ(input.frames[i].extra_channels.size(),
              output.frames[i].extra_channels.size());
    for (size_t c = 0; c < input.frames[i].extra_channels.size(); ++c) {
      SCOPED_TRACE(testing::Message() << "extra_channel=" << c);
      compare(input.frames[i].extra_channels[c],
              output.frames[i].extra_channels[c]);
    }
  }
}

TEST(JxlTest, LosslessPreservesStoragePrecisionDespiteSignificanceHints) {
  for (uint32_t bits : {8u, 16u}) {
    TestImage t;
    ASSERT_TRUE(t.SetDimensions(31, 17));
    ASSERT_TRUE(t.SetChannels(4));
    t.SetAllBitDepths(bits);
    JXL_TEST_ASSIGN_OR_DIE(auto frame, t.AddFrame());
    frame.RandomFill();
    for (size_t x = 0; x < 31; ++x) ASSERT_TRUE(frame.SetValue(0, x, 3, 0));
    t.ppf().lossless_color_bits = bits;
    t.ppf().lossless_alpha_bits = bits;
    t.ppf().info.bits_per_sample = 5;
    t.ppf().info.alpha_bits = 3;
    JXLDecompressParams dparams;
    dparams.accepted_formats.push_back(t.ppf().frames[0].color.format);
    auto params = test::CompressParamsForLossless();
    params.AddOption(JXL_ENC_FRAME_SETTING_KEEP_INVISIBLE, 1);
    PackedPixelFile decoded;
    ASSERT_GT(Roundtrip(t.ppf(), params, dparams, nullptr, &decoded), 0u);
    // Compare raw stored samples, not values quantized to the significance hint.
    t.ppf().info.bits_per_sample = bits;
    t.ppf().info.alpha_bits = bits;
    ExpectExactLosslessSamples(t.ppf(), decoded);
  }
}

TEST(JxlTest, LosslessPngSignificanceRequiresExactStoredSamples) {
  auto png_encoder = extras::Encoder::FromExtension(".png");
  if (!png_encoder) GTEST_SKIP() << "PNG support disabled";
  // RGBA sBIT=4, including its CRC, inserted immediately after IHDR.
  const std::vector<uint8_t> sbit = {
      0, 0, 0, 4, 115, 66, 73, 84, 4, 4, 4, 4, 154, 199, 33, 115};
  for (uint32_t bits : {8u, 16u}) {
    for (int changed_channel : {-1, 0, 3}) {
      SCOPED_TRACE(testing::Message() << bits << " " << changed_channel);
      TestImage t;
      ASSERT_TRUE(t.SetDimensions(16, 3));
      ASSERT_TRUE(t.SetChannels(4));
      t.SetAllBitDepths(bits);
      JXL_TEST_ASSIGN_OR_DIE(auto frame, t.AddFrame());
      for (size_t y = 0; y < 3; ++y) {
        for (size_t x = 0; x < 16; ++x) {
          for (size_t c = 0; c < 4; ++c) {
            t.ppf().frames[0].color.SetPixelValue(y, x, c, ((x + y + c) % 16) / 15.f);
          }
        }
      }
      // Hidden RGB must be validated too, not discarded by an alpha test.
      t.ppf().frames[0].color.SetPixelValue(0, 0, 3, 0);
      if (changed_channel >= 0) {
        t.ppf().frames[0].color.SetPixelValue(
            0, 0, changed_channel, 1.f / ((1u << bits) - 1));
      }
      extras::EncodedImage encoded;
      ASSERT_TRUE(png_encoder->Encode(t.ppf(), &encoded, nullptr));
      ASSERT_EQ(encoded.bitstreams.size(), 1u);
      auto& png = encoded.bitstreams[0];
      ASSERT_GE(png.size(), 33u);
      png.insert(png.begin() + 33, sbit.begin(), sbit.end());
      TestImage parsed;
      ASSERT_TRUE(parsed.DecodeFromBytes(png));
      EXPECT_EQ(parsed.ppf().lossless_color_bits, changed_channel == 0 ? bits : 4u);
      EXPECT_EQ(parsed.ppf().lossless_alpha_bits, changed_channel == 3 ? bits : 4u);
      JXLDecompressParams dparams;
      dparams.accepted_formats.push_back(t.ppf().frames[0].color.format);
      auto params = test::CompressParamsForLossless();
      params.AddOption(JXL_ENC_FRAME_SETTING_KEEP_INVISIBLE, 1);
      PackedPixelFile decoded;
      ASSERT_GT(Roundtrip(parsed.ppf(), params, dparams, nullptr, &decoded), 0u);
      ExpectExactLosslessSamples(t.ppf(), decoded);
    }
  }
}

TEST(JxlTest, LosslessPreservesSamplesWithGammaOutsideCompactRange) {
  for (size_t channels : {1u, 3u, 4u}) {
    TestImage t;
    ASSERT_TRUE(t.SetDimensions(31, 17));
    ASSERT_TRUE(t.SetChannels(channels));
    JXL_TEST_ASSIGN_OR_DIE(auto frame, t.AddFrame());
    frame.RandomFill();
    t.ppf().color_encoding.transfer_function = JXL_TRANSFER_FUNCTION_GAMMA;
    t.ppf().color_encoding.gamma = 2.5;
    JXLDecompressParams dparams;
    dparams.accepted_formats.push_back(t.ppf().frames[0].color.format);
    auto params = test::CompressParamsForLossless();
    params.AddOption(JXL_ENC_FRAME_SETTING_KEEP_INVISIBLE, 1);
    PackedPixelFile decoded;
    ASSERT_GT(Roundtrip(t.ppf(), params, dparams, nullptr, &decoded), 0u);
    ExpectExactLosslessSamples(t.ppf(), decoded);
    EXPECT_EQ(decoded.primary_color_representation, PackedPixelFile::kIccIsPrimary);
    EXPECT_FALSE(decoded.icc.empty());
  }
}

TEST(JxlTest, LosslessScanRoundtripHiddenRgbAndExtraChannels) {
  const char* enabled = std::getenv("JXL_LOSSLESS_SCAN");
  if (!enabled || std::strcmp(enabled, "1") != 0) GTEST_SKIP();
  ThreadPoolForTests pool(4);
  for (uint32_t bits : {8u, 16u}) {
    TestImage t;
    ASSERT_TRUE(t.SetDimensions(1025, 513));
    ASSERT_TRUE(t.SetChannels(5));
    t.SetAllBitDepths(bits);
    t.ppf().extra_channels_info[0].ec_info.type = JXL_CHANNEL_SELECTION_MASK;
    JXL_TEST_ASSIGN_OR_DIE(auto frame, t.AddFrame());
    frame.ZeroFill();
    const uint32_t max = (1u << bits) - 1;
    for (size_t y = 43; y < 131; ++y) {
      for (size_t x = 73; x < 329; ++x) {
        for (size_t c = 0; c < 3; ++c) {
          ASSERT_TRUE(frame.SetValue(y, x, c, float((x * 13 + y * 7 + c) % max) / max));
        }
        ASSERT_TRUE(frame.SetValue(y, x, 3, x < 151 ? 0.0f : 0.5f));
      }
    }
    for (size_t y = 401; y < 413; ++y) {
      for (size_t x = 751; x < 829; ++x) {
        t.ppf().frames[0].extra_channels[0].SetPixelValue(
            y, x, 0, float((x + y) % max) / max);
      }
    }
    JXLDecompressParams dparams;
    dparams.accepted_formats.push_back(t.ppf().frames[0].color.format);
    size_t previous = std::numeric_limits<size_t>::max();
    for (int effort : {3, 9, 10}) {
      auto cparams = test::CompressParamsForLossless();
      cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, effort);
      cparams.AddOption(JXL_ENC_FRAME_SETTING_KEEP_INVISIBLE, 1);
      cparams.AddOption(JXL_ENC_FRAME_SETTING_BUFFERING, 0);
      PackedPixelFile decoded;
      const size_t size = Roundtrip(t.ppf(), cparams, dparams, pool.get(), &decoded);
      ASSERT_GT(size, 0u);
      EXPECT_LE(size, previous);
      previous = size;
      ExpectExactLosslessSamples(t.ppf(), decoded);
    }
  }
}

JXL_SLOW_TEST(JxlTest, RoundtripLossless8) {
  ThreadPoolForTests pool(8);
  const std::vector<uint8_t> orig =
      ReadTestData("external/wesaturate/500px/tmshre_riaphotographs_srgb8.png");
  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));
  t.ClearMetadata();

  JXLCompressParams cparams = test::CompressParamsForLossless();
  JXLDecompressParams dparams;
  dparams.accepted_formats.push_back(t.ppf().frames[0].color.format);

  PackedPixelFile ppf_out;
  size_t compressed_size =
      Roundtrip(t.ppf(), cparams, dparams, pool.get(), &ppf_out);
  ASSERT_GT(compressed_size, 0u);
  ExpectExactLosslessSamples(t.ppf(), ppf_out);
  EXPECT_EQ(ComputeDistance2(t.ppf(), ppf_out), 0.0);
}

JXL_SLOW_TEST(JxlTest, RoundtripLossless8ThunderGradient) {
  ThreadPoolForTests pool(8);
  const std::vector<uint8_t> orig =
      ReadTestData("external/wesaturate/500px/tmshre_riaphotographs_srgb8.png");
  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));
  t.ClearMetadata();

  JXLCompressParams cparams = test::CompressParamsForLossless();
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 2);             // kThunder
  cparams.AddOption(JXL_ENC_FRAME_SETTING_MODULAR_PREDICTOR, 5);  // Gradient
  JXLDecompressParams dparams;
  dparams.accepted_formats.push_back(t.ppf().frames[0].color.format);

  PackedPixelFile ppf_out;
  ASSERT_GT(Roundtrip(t.ppf(), cparams, dparams, pool.get(), &ppf_out), 0u);
  ExpectExactLosslessSamples(t.ppf(), ppf_out);
  EXPECT_EQ(ComputeDistance2(t.ppf(), ppf_out), 0.0);
}

JXL_SLOW_TEST(JxlTest, RoundtripLossless8LightningGradient) {
  ThreadPoolForTests pool(8);
  const std::vector<uint8_t> orig =
      ReadTestData("external/wesaturate/500px/tmshre_riaphotographs_srgb8.png");
  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));
  t.ClearMetadata();

  JXLCompressParams cparams = test::CompressParamsForLossless();
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 1);  // kLightning
  JXLDecompressParams dparams;
  dparams.accepted_formats.push_back(t.ppf().frames[0].color.format);

  PackedPixelFile ppf_out;
  ASSERT_GT(Roundtrip(t.ppf(), cparams, dparams, pool.get(), &ppf_out), 0u);
  ExpectExactLosslessSamples(t.ppf(), ppf_out);
  EXPECT_EQ(ComputeDistance2(t.ppf(), ppf_out), 0.0);
}

JXL_SLOW_TEST(JxlTest, RoundtripLossless8Falcon) {
  ThreadPoolForTests pool(8);
  const std::vector<uint8_t> orig =
      ReadTestData("external/wesaturate/500px/tmshre_riaphotographs_srgb8.png");
  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));
  t.ClearMetadata();

  JXLCompressParams cparams = test::CompressParamsForLossless();
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 3);  // kFalcon
  JXLDecompressParams dparams;
  dparams.accepted_formats.push_back(t.ppf().frames[0].color.format);

  PackedPixelFile ppf_out;
  ASSERT_GT(Roundtrip(t.ppf(), cparams, dparams, pool.get(), &ppf_out), 0u);
  ExpectExactLosslessSamples(t.ppf(), ppf_out);
  EXPECT_EQ(ComputeDistance2(t.ppf(), ppf_out), 0.0);
}

TEST(JxlTest, RoundtripLossless8Alpha) {
  ThreadPool* pool = nullptr;
  const std::vector<uint8_t> orig =
      ReadTestData("external/wesaturate/500px/tmshre_riaphotographs_alpha.png");
  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));
  t.ClearMetadata();
  ASSERT_EQ(t.ppf().info.alpha_bits, 8u);
  EXPECT_EQ(t.ppf().frames[0].color.format.data_type, JXL_TYPE_UINT8);

  JXLCompressParams cparams = test::CompressParamsForLossless();

  JXLDecompressParams dparams;
  dparams.accepted_formats.push_back(t.ppf().frames[0].color.format);

  PackedPixelFile ppf_out;
  size_t compressed_size = Roundtrip(t.ppf(), cparams, dparams, pool, &ppf_out);
  ASSERT_GT(compressed_size, 0u);
  ExpectExactLosslessSamples(t.ppf(), ppf_out);
  EXPECT_EQ(ComputeDistance2(t.ppf(), ppf_out), 0.0);
  EXPECT_EQ(ppf_out.info.alpha_bits, 8u);
  EXPECT_TRUE(test::SameAlpha(t.ppf(), ppf_out));
}

TEST(JxlTest, RoundtripLossless16Alpha) {
  ThreadPool* pool = nullptr;
  size_t xsize = 1200;
  size_t ysize = 160;
  TestImage t;
  ASSERT_TRUE(t.SetDimensions(xsize, ysize));
  ASSERT_TRUE(t.SetChannels(4));
  t.SetAllBitDepths(16);
  JXL_TEST_ASSIGN_OR_DIE(auto frame, t.AddFrame());
  frame.ZeroFill();
  // Generate 16-bit pattern that uses various colors and alpha values.
  const float mul = 1.0f / 65535;
  for (size_t y = 0; y < ysize; y++) {
    for (size_t x = 0; x < xsize; x++) {
      uint16_t r = y * 65535 / ysize;
      uint16_t g = x * 65535 / xsize + 37;
      uint16_t b = (y + x) * 65535 / (xsize + ysize);
      ASSERT_TRUE(frame.SetValue(y, x, 0, r * mul));
      ASSERT_TRUE(frame.SetValue(y, x, 1, g * mul));
      ASSERT_TRUE(frame.SetValue(y, x, 2, b * mul));
      ASSERT_TRUE(frame.SetValue(y, x, 3, g * mul));
    }
  }
  ASSERT_EQ(t.ppf().info.bits_per_sample, 16u);
  ASSERT_EQ(t.ppf().info.alpha_bits, 16u);

  JXLCompressParams cparams = test::CompressParamsForLossless();

  JXLDecompressParams dparams;
  dparams.accepted_formats.push_back(t.ppf().frames[0].color.format);

  PackedPixelFile ppf_out;
  size_t compressed_size = Roundtrip(t.ppf(), cparams, dparams, pool, &ppf_out);
  ASSERT_GT(compressed_size, 0u);
  ExpectExactLosslessSamples(t.ppf(), ppf_out);
  EXPECT_EQ(ComputeDistance2(t.ppf(), ppf_out), 0.0);
  EXPECT_EQ(ppf_out.info.alpha_bits, 16u);
  EXPECT_TRUE(test::SameAlpha(t.ppf(), ppf_out));
}

TEST(JxlTest, RoundtripLossless16AlphaNotMisdetectedAs8Bit) {
  ThreadPool* pool = nullptr;
  size_t xsize = 128;
  size_t ysize = 128;
  TestImage t;
  ASSERT_TRUE(t.SetDimensions(xsize, ysize));
  ASSERT_TRUE(t.SetChannels(4));
  t.SetAllBitDepths(16);
  JXL_TEST_ASSIGN_OR_DIE(auto frame, t.AddFrame());
  // All 16-bit values, both color and alpha, of this image are below 64.
  // This allows testing if a code path wrongly concludes it's an 8-bit instead
  // of 16-bit image (or even 6-bit).
  const float mul = 1.0f / 65535;
  for (size_t y = 0; y < ysize; y++) {
    for (size_t x = 0; x < xsize; x++) {
      uint16_t r = y * 64 / ysize;
      uint16_t g = x * 64 / xsize + 37;
      uint16_t b = (y + x) * 64 / (xsize + ysize);
      ASSERT_TRUE(frame.SetValue(y, x, 0, r * mul));
      ASSERT_TRUE(frame.SetValue(y, x, 1, g * mul));
      ASSERT_TRUE(frame.SetValue(y, x, 2, b * mul));
      ASSERT_TRUE(frame.SetValue(y, x, 3, g * mul));
    }
  }
  ASSERT_EQ(t.ppf().info.bits_per_sample, 16u);
  ASSERT_EQ(t.ppf().info.alpha_bits, 16u);

  JXLCompressParams cparams = test::CompressParamsForLossless();

  JXLDecompressParams dparams;
  dparams.accepted_formats.push_back(t.ppf().frames[0].color.format);

  PackedPixelFile ppf_out;
  size_t compressed_size = Roundtrip(t.ppf(), cparams, dparams, pool, &ppf_out);
  ASSERT_GT(compressed_size, 0u);
  ExpectExactLosslessSamples(t.ppf(), ppf_out);
  EXPECT_EQ(ComputeDistance2(t.ppf(), ppf_out), 0.0);
  EXPECT_EQ(ppf_out.info.bits_per_sample, 16u);
  EXPECT_EQ(ppf_out.info.alpha_bits, 16u);
  EXPECT_TRUE(test::SameAlpha(t.ppf(), ppf_out));
}

TEST(JxlTest, RoundtripDots) {
  ThreadPool* pool = nullptr;
  const std::vector<uint8_t> orig =
      ReadTestData("external/wesaturate/500px/cvo9xd_keong_macan_srgb8.png");
  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));
  t.ClearMetadata();
  ASSERT_NE(t.ppf().info.xsize, 0u);
  EXPECT_EQ(t.ppf().info.bits_per_sample, 8u);
  EXPECT_EQ(t.ppf().color_encoding.transfer_function,
            JXL_TRANSFER_FUNCTION_SRGB);

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 7);  // kSquirrel
  cparams.AddOption(JXL_ENC_FRAME_SETTING_DOTS, 1);
  cparams.distance = 0.05;
  extras::JXLDecompressParams dparams;

  PackedPixelFile ppf_out;
  EXPECT_NEAR(Roundtrip(t.ppf(), cparams, dparams, pool, &ppf_out), 236173u,
              3000u);
  EXPECT_SLIGHTLY_BELOW(ButteraugliDistance(t.ppf(), ppf_out), 0.165);
}

TEST(JxlTest, RoundtripDisablePerceptual) {
  ThreadPool* pool = nullptr;
  const std::vector<uint8_t> orig = ReadTestData("jxl/flower/flower.png");
  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));
  t.ClearMetadata();
  ASSERT_NE(t.ppf().info.xsize, 0u);
  EXPECT_EQ(t.ppf().info.bits_per_sample, 8u);
  EXPECT_EQ(t.ppf().color_encoding.transfer_function,
            JXL_TRANSFER_FUNCTION_SRGB);

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 7);  // kSquirrel
  cparams.AddOption(JXL_ENC_FRAME_SETTING_DISABLE_PERCEPTUAL_HEURISTICS, 1);
  cparams.distance = 1.0;
  extras::JXLDecompressParams dparams;

  PackedPixelFile ppf_out;

  size_t expected_size = 487778;
  EXPECT_NEAR(Roundtrip(t.ppf(), cparams, dparams, pool, &ppf_out),
              expected_size, 4000);
  // TODO(veluca): figure out why we can't get below this value.
  EXPECT_SLIGHTLY_BELOW(ButteraugliDistance(t.ppf(), ppf_out), 11.0);
}

TEST(JxlTest, RoundtripNoise) {
  ThreadPool* pool = nullptr;
  const std::vector<uint8_t> orig =
      ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));
  t.ClearMetadata();
  ASSERT_NE(t.ppf().info.xsize, 0u);
  EXPECT_EQ(t.ppf().info.bits_per_sample, 8u);
  EXPECT_EQ(t.ppf().color_encoding.transfer_function,
            JXL_TRANSFER_FUNCTION_SRGB);

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 7);  // kSquirrel
  cparams.AddOption(JXL_ENC_FRAME_SETTING_NOISE, 1);
  extras::JXLDecompressParams dparams;

  PackedPixelFile ppf_out;
  EXPECT_NEAR(Roundtrip(t.ppf(), cparams, dparams, pool, &ppf_out), 41009, 750);
  EXPECT_SLIGHTLY_BELOW(ButteraugliDistance(t.ppf(), ppf_out), 1.42);
}

TEST(JxlTest, RoundtripLossless8Gray) {
  ThreadPool* pool = nullptr;
  const std::vector<uint8_t> orig = ReadTestData(
      "external/wesaturate/500px/cvo9xd_keong_macan_grayscale.png");
  TestImage t;
  ASSERT_TRUE(t.SetColorEncoding("Gra_D65_Rel_SRG"));
  ASSERT_TRUE(t.DecodeFromBytes(orig));
  t.ClearMetadata();
  EXPECT_EQ(t.ppf().color_encoding.color_space, JXL_COLOR_SPACE_GRAY);
  EXPECT_EQ(t.ppf().info.bits_per_sample, 8u);

  JXLCompressParams cparams = test::CompressParamsForLossless();

  JXLDecompressParams dparams;
  dparams.accepted_formats.push_back(t.ppf().frames[0].color.format);

  PackedPixelFile ppf_out;
  ASSERT_GT(Roundtrip(t.ppf(), cparams, dparams, pool, &ppf_out), 0u);
  ExpectExactLosslessSamples(t.ppf(), ppf_out);
  EXPECT_EQ(ComputeDistance2(t.ppf(), ppf_out), 0.0);
  EXPECT_EQ(ppf_out.color_encoding.color_space, JXL_COLOR_SPACE_GRAY);
  EXPECT_EQ(ppf_out.info.bits_per_sample, 8u);
}

TEST(JxlTest, RoundtripAnimation) {
  if (!jxl::extras::CanDecode(jxl::extras::Codec::kGIF)) {
    fprintf(stderr, "Skipping test because of missing GIF decoder.\n");
    return;
  }
  ThreadPool* pool = nullptr;
  const std::vector<uint8_t> orig = ReadTestData("jxl/traffic_light.gif");
  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));
  t.ClearMetadata();
  EXPECT_EQ(4u, t.ppf().frames.size());

  JXLDecompressParams dparams;
  dparams.accepted_formats.push_back(t.ppf().frames[0].color.format);

  PackedPixelFile ppf_out;
  EXPECT_SLIGHTLY_BELOW(Roundtrip(t.ppf(), {}, dparams, pool, &ppf_out), 2666);

  ASSERT_TRUE(t.CoalesceGIFAnimationWithAlpha());
  ASSERT_EQ(ppf_out.frames.size(), t.ppf().frames.size());
  static constexpr double kMaxButteraugli =
#if JXL_HIGH_PRECISION
      1.55;
#else
      1.75;
#endif
  EXPECT_LE(ButteraugliDistance(t.ppf(), ppf_out), kMaxButteraugli);
}

TEST(JxlTest, RoundtripLosslessAnimation) {
  if (!jxl::extras::CanDecode(jxl::extras::Codec::kGIF)) {
    GTEST_SKIP() << "Missing GIF decoder";
  }
  ThreadPool* pool = nullptr;
  const std::vector<uint8_t> orig = ReadTestData("jxl/traffic_light.gif");
  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));
  t.ClearMetadata();
  EXPECT_EQ(4u, t.ppf().frames.size());

  JXLCompressParams cparams = test::CompressParamsForLossless();

  JXLDecompressParams dparams;
  dparams.accepted_formats.push_back(t.ppf().frames[0].color.format);

  PackedPixelFile ppf_out;
  ASSERT_GT(Roundtrip(t.ppf(), cparams, dparams, pool, &ppf_out), 0u);

  ASSERT_TRUE(t.CoalesceGIFAnimationWithAlpha());
  ASSERT_EQ(ppf_out.frames.size(), t.ppf().frames.size());
  ExpectExactLosslessSamples(t.ppf(), ppf_out);
  EXPECT_LE(ButteraugliDistance(t.ppf(), ppf_out), 5e-4);
}

TEST(JxlTest, RoundtripAnimationPatches) {
  if (!jxl::extras::CanDecode(jxl::extras::Codec::kGIF)) {
    fprintf(stderr, "Skipping test because of missing GIF decoder.\n");
    return;
  }
  ThreadPool* pool = nullptr;
  const std::vector<uint8_t> orig = ReadTestData("jxl/animation_patches.gif");

  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));
  t.ClearMetadata();
  ASSERT_EQ(2u, t.ppf().frames.size());

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_PATCHES, 1);

  JXLDecompressParams dparams;
  dparams.accepted_formats.push_back(t.ppf().frames[0].color.format);

  PackedPixelFile ppf_out;
  // 40k with no patches, 27k with patch frames encoded multiple times.
  EXPECT_SLIGHTLY_BELOW(Roundtrip(t.ppf(), cparams, dparams, pool, &ppf_out),
                        21378u);
  EXPECT_EQ(ppf_out.frames.size(), t.ppf().frames.size());
  // >10 with broken patches; not all patches are detected on borders.
  EXPECT_SLIGHTLY_BELOW(ButteraugliDistance(t.ppf(), ppf_out), 1.88);
}

size_t RoundtripJpeg(const std::vector<uint8_t>& jpeg_in, ThreadPool* pool) {
  std::vector<uint8_t> compressed;
  JXLCompressParams params;
  EXPECT_TRUE(extras::EncodeImageJXL(params, extras::PackedPixelFile(), &jpeg_in,
                                     &compressed));

  jxl::JXLDecompressParams dparams;
  test::SetThreadParallelRunner(dparams, pool);
  {
    std::vector<uint8_t> out;
    jxl::PackedPixelFile ppf;
    EXPECT_FALSE(DecodeImageJXL(compressed.data(), compressed.size() - 1,
                                dparams, nullptr, &ppf, &out));
  }
  std::vector<uint8_t> out;
  jxl::PackedPixelFile ppf;
  EXPECT_TRUE(DecodeImageJXL(compressed.data(), compressed.size(), dparams,
                             nullptr, &ppf, &out));
  EXPECT_EQ(out.size(), jpeg_in.size());
  size_t failures = 0;
  for (size_t i = 0; i < std::min(out.size(), jpeg_in.size()); i++) {
    if (out[i] != jpeg_in[i]) {
      EXPECT_EQ(out[i], jpeg_in[i])
          << "byte mismatch " << i << " " << out[i] << " != " << jpeg_in[i];
      if (++failures > 4) {
        return compressed.size();
      }
    }
  }
  return compressed.size();
}

void RoundtripJpegToPixels(const std::vector<uint8_t>& jpeg_in,
                           JXLDecompressParams dparams, ThreadPool* pool,
                           PackedPixelFile* ppf_out) {
  std::vector<uint8_t> jpeg_bytes(jpeg_in.data(),
                                  jpeg_in.data() + jpeg_in.size());
  std::vector<uint8_t> compressed;
  EXPECT_TRUE(extras::EncodeImageJXL({}, extras::PackedPixelFile(), &jpeg_bytes,
                                     &compressed));

  test::DefaultAcceptedFormats(dparams);
  test::SetThreadParallelRunner(dparams, pool);
  EXPECT_TRUE(DecodeImageJXL(compressed.data(), compressed.size(), dparams,
                             nullptr, ppf_out, nullptr));
}

JXL_TRANSCODE_JPEG_TEST(JxlTest, RoundtripJpegRecompression444) {
  ThreadPoolForTests pool(8);
  const std::vector<uint8_t> orig =
      ReadTestData("jxl/flower/flower.png.im_q85_444.jpg");
  // JPEG size is 696,659 bytes.
  EXPECT_GT(RoundtripJpeg(orig, pool.get()), 0u);
}

JXL_TRANSCODE_JPEG_TEST(JxlTest, RoundtripJpegRecompressionToPixels) {
  TEST_LIBJPEG_SUPPORT();
  ThreadPoolForTests pool(8);
  const std::vector<uint8_t> orig =
      ReadTestData("jxl/flower/flower.png.im_q85_444.jpg");
  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));

  PackedPixelFile ppf_out;
  RoundtripJpegToPixels(orig, {}, pool.get(), &ppf_out);
  EXPECT_SLIGHTLY_BELOW(ComputeDistance2(t.ppf(), ppf_out), 12);
}

JXL_TRANSCODE_JPEG_TEST(JxlTest, RoundtripJpegRecompressionToPixels420) {
  TEST_LIBJPEG_SUPPORT();
  ThreadPoolForTests pool(8);
  const std::vector<uint8_t> orig =
      ReadTestData("jxl/flower/flower.png.im_q85_420.jpg");
  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));

  PackedPixelFile ppf_out;
  RoundtripJpegToPixels(orig, {}, pool.get(), &ppf_out);
  EXPECT_SLIGHTLY_BELOW(ComputeDistance2(t.ppf(), ppf_out), 11);
}

JXL_TRANSCODE_JPEG_TEST(JxlTest,
                        RoundtripJpegRecompressionToPixels420EarlyFlush) {
  TEST_LIBJPEG_SUPPORT();
  ThreadPoolForTests pool(8);
  const std::vector<uint8_t> orig =
      ReadTestData("jxl/flower/flower.png.im_q85_420.jpg");
  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));

  JXLDecompressParams dparams;
  dparams.max_downsampling = 8u;

  PackedPixelFile ppf_out;
  RoundtripJpegToPixels(orig, dparams, pool.get(), &ppf_out);
  EXPECT_SLIGHTLY_BELOW(ComputeDistance2(t.ppf(), ppf_out), 4410);
}

JXL_TRANSCODE_JPEG_TEST(JxlTest, RoundtripJpegRecompressionToPixels420Mul16) {
  TEST_LIBJPEG_SUPPORT();
  ThreadPoolForTests pool(8);
  const std::vector<uint8_t> orig =
      ReadTestData("jxl/flower/flower_cropped.jpg");
  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));

  PackedPixelFile ppf_out;
  RoundtripJpegToPixels(orig, {}, pool.get(), &ppf_out);
  EXPECT_SLIGHTLY_BELOW(ComputeDistance2(t.ppf(), ppf_out), 4);
}

JXL_TRANSCODE_JPEG_TEST(JxlTest, RoundtripJpegRecompressionToPixelsAsymmetric) {
  TEST_LIBJPEG_SUPPORT();
  ThreadPoolForTests pool(8);
  const std::vector<uint8_t> orig =
      ReadTestData("jxl/flower/flower.png.im_q85_asymmetric.jpg");
  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));

  PackedPixelFile ppf_out;
  RoundtripJpegToPixels(orig, {}, pool.get(), &ppf_out);
  EXPECT_SLIGHTLY_BELOW(ComputeDistance2(t.ppf(), ppf_out), 10);
}

JXL_TRANSCODE_JPEG_TEST(JxlTest, RoundtripJpegRecompressionGray) {
  ThreadPoolForTests pool(8);
  const std::vector<uint8_t> orig =
      ReadTestData("jxl/flower/flower.png.im_q85_gray.jpg");
  // JPEG size is 456,528 bytes.
  EXPECT_GT(RoundtripJpeg(orig, pool.get()), 0u);
}

JXL_TRANSCODE_JPEG_TEST(JxlTest, RoundtripJpegRecompression420) {
  ThreadPoolForTests pool(8);
  const std::vector<uint8_t> orig =
      ReadTestData("jxl/flower/flower.png.im_q85_420.jpg");
  // JPEG size is 546,797 bytes.
  EXPECT_GT(RoundtripJpeg(orig, pool.get()), 0u);
}

JXL_TRANSCODE_JPEG_TEST(JxlTest, RoundtripJpegRecompressionLumaSubsample) {
  ThreadPoolForTests pool(8);
  const std::vector<uint8_t> orig =
      ReadTestData("jxl/flower/flower.png.im_q85_luma_subsample.jpg");
  // JPEG size is 400,724 bytes.
  EXPECT_GT(RoundtripJpeg(orig, pool.get()), 0u);
}

JXL_TRANSCODE_JPEG_TEST(JxlTest, RoundtripJpegRecompression444wh12) {
  // 444 JPEG that has an interesting sampling-factor (1x2, 1x2, 1x2).
  ThreadPoolForTests pool(8);
  const std::vector<uint8_t> orig =
      ReadTestData("jxl/flower/flower.png.im_q85_444_1x2.jpg");
  // JPEG size is 703,874 bytes.
  EXPECT_GT(RoundtripJpeg(orig, pool.get()), 0u);
}

JXL_TRANSCODE_JPEG_TEST(JxlTest, RoundtripJpegRecompression422) {
  ThreadPoolForTests pool(8);
  const std::vector<uint8_t> orig =
      ReadTestData("jxl/flower/flower.png.im_q85_422.jpg");
  // JPEG size is 522,057 bytes.
  EXPECT_GT(RoundtripJpeg(orig, pool.get()), 0u);
}

JXL_TRANSCODE_JPEG_TEST(JxlTest, RoundtripJpegRecompression440) {
  ThreadPoolForTests pool(8);
  const std::vector<uint8_t> orig =
      ReadTestData("jxl/flower/flower.png.im_q85_440.jpg");
  // JPEG size is 603,623 bytes.
  EXPECT_GT(RoundtripJpeg(orig, pool.get()), 0u);
}

JXL_TRANSCODE_JPEG_TEST(JxlTest, RoundtripJpegRecompressionAsymmetric) {
  // 2x vertical downsample of one chroma channel, 2x horizontal downsample of
  // the other.
  ThreadPoolForTests pool(8);
  const std::vector<uint8_t> orig =
      ReadTestData("jxl/flower/flower.png.im_q85_asymmetric.jpg");
  // JPEG size is 604,601 bytes.
  EXPECT_GT(RoundtripJpeg(orig, pool.get()), 0u);
}

JXL_TRANSCODE_JPEG_TEST(JxlTest, RoundtripJpegRecompression420Progr) {
  ThreadPoolForTests pool(8);
  const std::vector<uint8_t> orig =
      ReadTestData("jxl/flower/flower.png.im_q85_420_progr.jpg");
  // JPEG size is 522,057 bytes.
  EXPECT_GT(RoundtripJpeg(orig, pool.get()), 0u);
}

JXL_TRANSCODE_JPEG_TEST(JxlTest, RoundtripJpegRecompressionMetadata) {
  ThreadPoolForTests pool(8);
  const std::vector<uint8_t> orig =
      ReadTestData("jxl/jpeg_reconstruction/1x1_exif_xmp.jpg");
  // JPEG size is 4290 bytes
  EXPECT_GT(RoundtripJpeg(orig, pool.get()), 0u);
}

JXL_TRANSCODE_JPEG_TEST(JxlTest, JpegLfSmoothingRequiresOptIn) {
  const auto original =
      ReadTestData("jxl/jpeg_reconstruction/sideways_bench.jpg");
  // A same-version roundtrip alone misses this bug: libjxl 0.12 smooths the
  // reconstructed DC coefficients unless the encoder sets the skip flag.
  for (bool legacy_backend : {false, true}) {
    for (int effort : {1, 9}) {
      SCOPED_TRACE(testing::Message() << "legacy=" << legacy_backend
                                     << " effort=" << effort);
      JXLCompressParams params;
      params.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, effort);
      if (legacy_backend) {
        params.AddFloatOption(
            JXL_ENC_FRAME_SETTING_MODULAR_MA_TREE_LEARNING_PERCENT, 50.f);
      }
      std::vector<uint8_t> default_stream;
      ASSERT_TRUE(extras::EncodeImageJXL(params, PackedPixelFile(), &original,
                                         &default_stream));
      std::vector<uint8_t> default_restored;
      PackedPixelFile default_decoded;
      ASSERT_TRUE(extras::DecodeImageJXL(default_stream.data(), default_stream.size(),
                                         {}, nullptr, &default_decoded, &default_restored));
      EXPECT_EQ(default_restored, original);
      for (int smoothing : {-1, 0, 1}) {
        SCOPED_TRACE(testing::Message() << "smoothing=" << smoothing);
        auto explicit_params = params;
        explicit_params.AddOption(JXL_ENC_FRAME_SETTING_JPEG_RECON_LFS,
                                   smoothing);
        std::vector<uint8_t> compressed;
        ASSERT_TRUE(extras::EncodeImageJXL(explicit_params, PackedPixelFile(),
                                           &original, &compressed));
        std::vector<uint8_t> restored;
        PackedPixelFile decoded;
        ASSERT_TRUE(extras::DecodeImageJXL(compressed.data(), compressed.size(),
                                           {}, nullptr, &decoded, &restored));
        EXPECT_EQ(restored, original);
      }
    }
  }
}

JXL_TRANSCODE_JPEG_TEST(JxlTest, LosslessSearchJpegCumulative) {
  ThreadPoolForTests pool(4);
  for (const char* name : {"jxl/jpeg_reconstruction/sideways_bench.jpg",
                           "jxl/jpeg_reconstruction/1x1_exif_xmp.jpg",
                           "jxl/flower/flower.png.im_q85_420_progr.jpg"}) {
    const auto original = ReadTestData(name);
    size_t previous_size = 0;
    for (int effort = 1; effort <= 10; ++effort) {
      JXLCompressParams params;
      params.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, effort);
      test::SetThreadParallelRunner(params, pool.get());
      std::vector<uint8_t> compressed;
      ASSERT_TRUE(extras::EncodeImageJXL(params, PackedPixelFile(), &original,
                                         &compressed));
      if (previous_size != 0) EXPECT_LE(compressed.size(), previous_size);
      previous_size = compressed.size();
      std::vector<uint8_t> restored;
      PackedPixelFile decoded;
      ASSERT_TRUE(extras::DecodeImageJXL(compressed.data(), compressed.size(),
                                         {}, nullptr, &decoded, &restored));
      EXPECT_EQ(restored, original);
    }
  }
}

JXL_TRANSCODE_JPEG_TEST(JxlTest, RoundtripJpegRecompressionEmptyExif) {
  ThreadPoolForTests pool(8);
  const std::vector<uint8_t> orig = {
      /* clang-format off */
      // SOI
      0xff, 0xd8,  //
      // APP1
      0xff, 0xe1, 0x00, 0x08, 0x45, 0x78, 0x69, 0x66, 0x00, 0x00,  //
      // DQT
      0xff, 0xdb, 0x00, 0x43, 0x00, 0x03, 0x02, 0x02, 0x03, 0x02,  //
      0x02, 0x03, 0x03, 0x03, 0x03, 0x04, 0x03, 0x03, 0x04, 0x05,  //
      0x08, 0x05, 0x05, 0x04, 0x04, 0x05, 0x0a, 0x07, 0x07, 0x06,  //
      0x08, 0x0c, 0x0a, 0x0c, 0x0c, 0x0b, 0x0a, 0x0b, 0x0b, 0x0d,  //
      0x0e, 0x12, 0x10, 0x0d, 0x0e, 0x11, 0x0e, 0x0b, 0x0b, 0x10,  //
      0x16, 0x10, 0x11, 0x13, 0x14, 0x15, 0x15, 0x15, 0x0c, 0x0f,  //
      0x17, 0x18, 0x16, 0x14, 0x18, 0x12, 0x14, 0x15, 0x14,        //
      // SOF
      0xff, 0xc0, 0x00, 0x0b, 0x08, 0x00, 0x01, 0x00, 0x01, 0x01,  //
      0x01, 0x11, 0x00,                                            //
      // DHT
      0xff, 0xc4, 0x00, 0xd2, 0x00, 0x00, 0x01, 0x05, 0x01, 0x01,  //
      0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  //
      0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,  //
      0x09, 0x0a, 0x0b, 0x10, 0x00, 0x02, 0x01, 0x03, 0x03, 0x02,  //
      0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7d,  //
      0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31,  //
      0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32,  //
      0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52,  //
      0xd1, 0xf0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16,  //
      0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a,  //
      0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45,  //
      0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57,  //
      0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,  //
      0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83,  //
      0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94,  //
      0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,  //
      0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,  //
      0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7,  //
      0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8,  //
      0xd9, 0xda, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8,  //
      0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,  //
      0xf9, 0xfa,                                                  //
      // SOS
      0xff, 0xda, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3f, 0x00,  //
      // entropy coded data
      0xfc, 0xaa, 0xaf,  //
      // EOI
      0xff, 0xd9,  //
      /* clang-format on */
  };
  EXPECT_GT(RoundtripJpeg(orig, pool.get()), 0u);
}

JXL_TRANSCODE_JPEG_TEST(JxlTest, RoundtripJpegRecompressionRestarts) {
  ThreadPoolForTests pool(8);
  const std::vector<uint8_t> orig =
      ReadTestData("jxl/jpeg_reconstruction/bicycles_restarts.jpg");
  // JPEG size is 87478 bytes
  EXPECT_GT(RoundtripJpeg(orig, pool.get()), 0u);
}

JXL_TRANSCODE_JPEG_TEST(JxlTest, RoundtripJpegRecompressionOrientationICC) {
  ThreadPoolForTests pool(8);
  const std::vector<uint8_t> orig =
      ReadTestData("jxl/jpeg_reconstruction/sideways_bench.jpg");
  // JPEG size is 15252 bytes
  EXPECT_GT(RoundtripJpeg(orig, pool.get()), 0u);
}

TEST(JxlTest, RoundtripProgressive) {
  ThreadPoolForTests pool(4);
  const std::vector<uint8_t> orig = ReadTestData("jxl/flower/flower.png");
  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));
  t.ClearMetadata();
  ASSERT_TRUE(t.SetDimensions(600, 1024));

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_PROGRESSIVE_DC, 1);
  cparams.AddOption(JXL_ENC_FRAME_SETTING_PROGRESSIVE_AC, 1);
  cparams.AddOption(JXL_ENC_FRAME_SETTING_RESPONSIVE, 1);
  extras::JXLDecompressParams dparams;

  PackedPixelFile ppf_out;
  EXPECT_NEAR(Roundtrip(t.ppf(), cparams, dparams, pool.get(), &ppf_out),
              68606u, 500u);
  EXPECT_SLIGHTLY_BELOW(ButteraugliDistance(t.ppf(), ppf_out), 1.2);
}

TEST(JxlTest, RoundtripProgressiveLevel2Slow) {
  ThreadPoolForTests pool(8);
  const std::vector<uint8_t> orig = ReadTestData("jxl/flower/flower.png");
  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));
  t.ClearMetadata();
  ASSERT_TRUE(t.SetDimensions(600u, 1024u));

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 9);  // kTortoise
  cparams.AddOption(JXL_ENC_FRAME_SETTING_PROGRESSIVE_DC, 2);
  cparams.AddOption(JXL_ENC_FRAME_SETTING_PROGRESSIVE_AC, 1);
  cparams.AddOption(JXL_ENC_FRAME_SETTING_RESPONSIVE, 1);
  extras::JXLDecompressParams dparams;

  PackedPixelFile ppf_out;
  EXPECT_NEAR(Roundtrip(t.ppf(), cparams, dparams, pool.get(), &ppf_out), 71969,
              1000);
  EXPECT_SLIGHTLY_BELOW(ButteraugliDistance(t.ppf(), ppf_out), 1.20);
}

TEST(JxlTest, RoundtripUnsignedCustomBitdepthLossless) {
  ThreadPool* pool = nullptr;
  for (uint32_t num_channels = 1; num_channels < 6; ++num_channels) {
    for (JxlEndianness endianness : {JXL_LITTLE_ENDIAN, JXL_BIG_ENDIAN}) {
      for (uint32_t bitdepth = 3; bitdepth <= 16; ++bitdepth) {
        if (bitdepth <= 8 && endianness == JXL_BIG_ENDIAN) continue;
        printf("Testing %u channel unsigned %u bit %s endian lossless.\n",
               num_channels, bitdepth,
               endianness == JXL_LITTLE_ENDIAN ? "little" : "big");
        TestImage t;
        ASSERT_TRUE(t.SetDimensions(256u, 256u));
        ASSERT_TRUE(t.SetChannels(num_channels));
        t.SetAllBitDepths(bitdepth).SetEndianness(endianness);
        JXL_TEST_ASSIGN_OR_DIE(auto frame, t.AddFrame());
        frame.RandomFill();
        t.ppf().input_bitdepth.type = JXL_BIT_DEPTH_FROM_CODESTREAM;

        JXLCompressParams cparams = test::CompressParamsForLossless();

        JXLDecompressParams dparams;
        dparams.accepted_formats.push_back(t.ppf().frames[0].color.format);
        dparams.output_bitdepth.type = JXL_BIT_DEPTH_FROM_CODESTREAM;

        PackedPixelFile ppf_out;
        Roundtrip(t.ppf(), cparams, dparams, pool, &ppf_out);

        ASSERT_TRUE(test::SamePixels(t.ppf(), ppf_out));
      }
    }
  }
}

TEST(JxlTest, LosslessPNMRoundtrip) {
  static const char* kChannels[] = {"", "g", "ga", "rgb", "rgba"};
  static const char* kExtension[] = {"", ".pgm", ".pam", ".ppm", ".pam"};
  for (size_t bit_depth = 1; bit_depth <= 16; ++bit_depth) {
    for (size_t channels = 1; channels <= 4; ++channels) {
      if (bit_depth == 1 && (channels == 2 || channels == 4)) continue;
      std::string extension(kExtension[channels]);
      std::string filename = "jxl/flower/flower_small." +
                             std::string(kChannels[channels]) + ".depth" +
                             std::to_string(bit_depth) + extension;
      const std::vector<uint8_t> orig = ReadTestData(filename);
      test::TestImage t;
      if (channels < 3) {
        ASSERT_TRUE(t.SetColorEncoding("Gra_D65_Rel_SRG"));
      }
      ASSERT_TRUE(t.DecodeFromBytes(orig));

      JXLCompressParams cparams = test::CompressParamsForLossless();
      cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 1);  // kLightning

      JXLDecompressParams dparams;
      dparams.accepted_formats.push_back(t.ppf().frames[0].color.format);
      dparams.output_bitdepth.type = JXL_BIT_DEPTH_FROM_CODESTREAM;

      PackedPixelFile ppf_out;
      Roundtrip(t.ppf(), cparams, dparams, nullptr, &ppf_out);

      extras::EncodedImage encoded;
      auto encoder = extras::Encoder::FromExtension(extension);
      ASSERT_TRUE(encoder.get());
      ASSERT_TRUE(encoder->Encode(ppf_out, &encoded, nullptr));
      ASSERT_EQ(encoded.bitstreams.size(), 1u);
      ASSERT_FALSE(encoded.bitstreams[0].empty());
      TestImage restored;
      if (channels < 3) {
        ASSERT_TRUE(restored.SetColorEncoding("Gra_D65_Rel_SRG"));
      }
      ASSERT_TRUE(restored.DecodeFromBytes(encoded.bitstreams[0]));
      ExpectExactLosslessSamples(t.ppf(), restored.ppf());
    }
  }
}

class JxlTest : public ::testing::TestWithParam<const char*> {};

TEST_P(JxlTest, LosslessSmallFewColors) {
  ThreadPoolForTests pool(8);
  const std::vector<uint8_t> orig = ReadTestData(GetParam());
  TestImage t;
  ASSERT_TRUE(t.DecodeFromBytes(orig));
  t.ClearMetadata();

  JXLCompressParams cparams;
  cparams.distance = 0;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 1);
  JXLDecompressParams dparams;
  dparams.accepted_formats.push_back(t.ppf().frames[0].color.format);

  PackedPixelFile ppf_out;
  Roundtrip(t.ppf(), cparams, dparams, pool.get(), &ppf_out);
  EXPECT_EQ(ComputeDistance2(t.ppf(), ppf_out), 0.0);
}

JXL_GTEST_INSTANTIATE_TEST_SUITE_P(
    ImageTests, JxlTest,
    ::testing::Values("jxl/blending/cropped_traffic_light_frame-0.png",
                      "palette/358colors.png"));

struct StreamingTestParam {
  size_t xsize;
  size_t ysize;
  bool is_grey;
  bool has_alpha;
  int effort;
  bool progressive;

  size_t num_channels() const {
    return (is_grey ? 1 : 3) + (has_alpha ? 1 : 0);
  }

  float max_psnr() const { return is_grey ? 90 : 40; }

  static std::vector<StreamingTestParam> All() {
    std::vector<StreamingTestParam> params;
    for (int e : {1, 3, 4, 7}) {
      for (bool g : {false, true}) {
        params.push_back(StreamingTestParam{357, 517, g, false, e, false});
        params.push_back(StreamingTestParam{2247, 2357, g, false, e, false});
      }
    }
    params.push_back(StreamingTestParam{2247, 2357, false, false, 1, true});
    params.push_back(StreamingTestParam{2247, 2157, false, false, 5, false});
    params.push_back(StreamingTestParam{2247, 2157, false, true, 5, false});
    return params;
  }
};

std::ostream& operator<<(std::ostream& out, StreamingTestParam p) {
  out << (p.is_grey ? "Grey" : "RGB");
  out << p.xsize << "x" << p.ysize;
  out << "e" << p.effort;
  if (p.progressive) {
    out << "Progressive";
  }
  return out;
}

class JxlStreamingTest : public ::testing::TestWithParam<StreamingTestParam> {};

TEST_P(JxlStreamingTest, Roundtrip) {
  const StreamingTestParam& p = GetParam();

  jxl::test::TestImage image;
  ASSERT_TRUE(image.SetDimensions(p.xsize, p.ysize));
  image.SetDataType(JXL_TYPE_UINT8);
  ASSERT_TRUE(image.SetChannels(p.num_channels()));
  image.SetAllBitDepths(8);
  JXL_TEST_ASSIGN_OR_DIE(auto frame, image.AddFrame());
  frame.RandomFill();
  JXLCompressParams cparams;
  cparams.distance = 0.1;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, p.effort);
  cparams.AddOption(JXL_ENC_FRAME_SETTING_BUFFERING, 3);
  if (p.progressive) {
    cparams.AddOption(JXL_ENC_FRAME_SETTING_PROGRESSIVE_AC, 1);
  }
  extras::JXLDecompressParams dparams;

  ThreadPoolForTests pool(8);
  PackedPixelFile ppf_out;
  Roundtrip(image.ppf(), cparams, dparams, pool.get(), &ppf_out);
  EXPECT_GT(jxl::test::ComputePSNR(image.ppf(), ppf_out), p.max_psnr());
}

JXL_GTEST_INSTANTIATE_TEST_SUITE_P(
    JxlStreamingTest, JxlStreamingTest,
    testing::ValuesIn(StreamingTestParam::All()));

struct StreamingEncodingTestParam {
  std::string file;
  int effort;
  float distance;
  int group_size;
  float palette_percent;

  static std::vector<StreamingEncodingTestParam> All() {
    std::vector<StreamingEncodingTestParam> params;
    for (const auto* file :
         {"jxl/flower/flower.png", "jxl/flower/flower_alpha.png"}) {
      for (int effort : {1, 3, 5, 6}) {
        if (effort != 1) {
          params.push_back(
              StreamingEncodingTestParam{file, effort, 1.0, 1, -1});
          params.push_back(
              StreamingEncodingTestParam{file, effort, 4.0, 1, -1});
        }
        for (auto group_size : {-1, 0}) {
          for (float palette_percent : {-1, 50, 100}) {
            params.push_back(StreamingEncodingTestParam{
                file, effort, 0.0, group_size, palette_percent});
          }
        }
      }
    }
    return params;
  }
};

std::ostream& operator<<(std::ostream& out,
                         const StreamingEncodingTestParam& p) {
  out << p.file << "-";
  out << "e" << p.effort;
  if (p.distance == 0) {
    out << "Lossless";
    out << "G" << p.group_size << "P" << p.palette_percent;
  } else {
    out << "D" << p.distance;
  }
  return out;
}

class JxlStreamingEncodingTest
    : public ::testing::TestWithParam<StreamingEncodingTestParam> {};

TEST_P(JxlStreamingEncodingTest, StreamingSamePixels) {
  const auto param = GetParam();

  const std::vector<uint8_t> orig = ReadTestData(param.file);
  jxl::test::TestImage image;
  ASSERT_TRUE(image.DecodeFromBytes(orig));

  JXLCompressParams cparams;
  cparams.distance = param.distance;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, param.effort);
  cparams.AddOption(JXL_ENC_FRAME_SETTING_MODULAR_GROUP_SIZE, param.group_size);
  cparams.AddFloatOption(JXL_ENC_FRAME_SETTING_CHANNEL_COLORS_GROUP_PERCENT,
                         param.palette_percent);
  cparams.AddOption(JXL_ENC_FRAME_SETTING_USE_FULL_IMAGE_HEURISTICS, 0);
  extras::JXLDecompressParams dparams;

  ThreadPoolForTests pool(8);
  PackedPixelFile ppf_out;
  Roundtrip(image.ppf(), cparams, dparams, pool.get(), &ppf_out);

  cparams.AddOption(JXL_ENC_FRAME_SETTING_BUFFERING, 3);
  PackedPixelFile ppf_out_streaming;
  Roundtrip(image.ppf(), cparams, dparams, pool.get(), &ppf_out_streaming);

  EXPECT_TRUE(jxl::test::SamePixels(ppf_out, ppf_out_streaming));
}

JXL_GTEST_INSTANTIATE_TEST_SUITE_P(
    JxlStreamingTest, JxlStreamingEncodingTest,
    testing::ValuesIn(StreamingEncodingTestParam::All()));

}  // namespace
}  // namespace jxl
