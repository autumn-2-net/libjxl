// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/memory_manager.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "hwy/targets.h"
#include "lib/jxl/ans_params.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/random.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_ans_params.h"
#include "lib/jxl/enc_ans_simd.h"
#include "lib/jxl/enc_uint_histogram.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/enc_cluster.h"
#include "lib/jxl/enc_cluster_bound.h"
#include "lib/jxl/enc_lz77_cost.h"
#include "lib/jxl/enc_lz77.h"
#include "lib/jxl/enc_lz77_bucket.h"
#include "lib/jxl/enc_lz77_parse.h"
#include "lib/jxl/enc_lz77_index.h"
#include "lib/jxl/enc_lz77_block.h"
#include "lib/jxl/enc_lz77_suffix.h"
#include "lib/jxl/enc_lz77_seed.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/enc_tool_policy.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/test_memory_manager.h"
#include "lib/jxl/test_utils.h"
#include "lib/jxl/testing.h"

namespace jxl {
namespace {

void RoundtripTestcase(int n_histograms, int alphabet_size,
                       const std::vector<Token>& input_values,
                       const HistogramParams& params = HistogramParams(),
                       bool require_lz77 = false) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  constexpr uint16_t kMagic1 = 0x9e33;
  constexpr uint16_t kMagic2 = 0x8b04;

  BitWriter writer{memory_manager};
  // Space for magic bytes.
  ASSERT_TRUE(writer.WithMaxBits(16, LayerType::Header, nullptr, [&] {
    writer.Write(16, kMagic1);
    return true;
  }));

  EntropyEncodingData codes;
  std::vector<std::vector<Token>> input_values_vec;
  input_values_vec.push_back(input_values);

  JXL_TEST_ASSIGN_OR_DIE(
      size_t cost,
      BuildAndEncodeHistograms(memory_manager, params, n_histograms,
                               input_values_vec, &codes, &writer,
                               LayerType::Header, nullptr));
  (void)cost;
  if (require_lz77) ASSERT_TRUE(codes.lz77.enabled);
  ASSERT_TRUE(WriteTokens(input_values_vec[0], codes, 0, &writer,
                          LayerType::Header, nullptr));

  // Magic bytes + padding
  ASSERT_TRUE(writer.WithMaxBits(24, LayerType::Header, nullptr, [&] {
    writer.Write(16, kMagic2);
    writer.ZeroPadToByte();
    return true;
  }));

  // We do not truncate the output. Reading past the end reads out zeroes
  // anyway.
  BitReader br(writer.GetSpan());

  ASSERT_EQ(br.ReadBits(16), kMagic1);

  std::vector<uint8_t> dec_context_map;
  ANSCode decoded_codes;
  ASSERT_TRUE(DecodeHistograms(memory_manager, &br, n_histograms,
                               &decoded_codes, &dec_context_map));
  ASSERT_EQ(dec_context_map, codes.context_map);
  JXL_TEST_ASSIGN_OR_DIE(ANSSymbolReader reader,
                         ANSSymbolReader::Create(&decoded_codes, &br,
                             params.image_widths.empty() ? 0 : params.image_widths[0]));

  for (size_t i = 0; i < input_values.size(); ++i) {
    const Token& symbol = input_values[i];
    uint32_t read_symbol =
        reader.ReadHybridUint(symbol.context, &br, dec_context_map);
    ASSERT_EQ(read_symbol, symbol.value) << "token " << i;
  }
  ASSERT_TRUE(reader.CheckANSFinalState());

  ASSERT_EQ(br.ReadBits(16), kMagic2);
  EXPECT_TRUE(br.Close());
}

TEST(ANSTest, EmptyRoundtrip) {
  RoundtripTestcase(2, ANS_MAX_ALPHABET_SIZE, std::vector<Token>());
}

TEST(ANSTest, ParallelUintSearchRoundtripAndRunnerFailure) {
  const char* enabled = std::getenv("JXL_LOSSLESS_UINT_PARALLEL");
  const char* weighted = std::getenv("JXL_LOSSLESS_UINT_HISTOGRAM");
  if (!enabled || std::strcmp(enabled, "1") || !weighted || std::strcmp(weighted, "1"))
    GTEST_SKIP();
  std::vector<Token> values;
  for (size_t c = 0; c < 16; ++c) {
    const size_t count = c < 8 ? 10000 : 1000;
    for (size_t i = 0; i < count; ++i) {
      const uint32_t v = c < 4 ? (i % (c + 1)) :
          c < 8 ? ((i * 31) % 1031) << (c - 4) :
                  static_cast<uint32_t>(i * 2654435761u);
      values.emplace_back(c, v);
    }
  }
  for (int threads : {0, 4}) {
    test::ThreadPoolForTests delegate(threads);
    struct State { ThreadPool* pool; size_t calls = 0; bool fail = false; }
        state{delegate.get()};
    ThreadPool pool([](void* opaque, void* context, JxlParallelRunInit init,
                        JxlParallelRunFunction run, uint32_t begin, uint32_t end) {
      auto* s = static_cast<State*>(opaque);
      ++s->calls;
      if (s->fail) return JXL_PARALLEL_RET_RUNNER_ERROR;
      return s->pool->runner()(s->pool->runner_opaque(), context, init, run, begin, end);
    }, &state);
    for (bool prefix : {false, true}) {
      HistogramParams params;
      params.lossless_lz77_cost_spans = true;
      params.lossless_lz77_pool = &pool;
      params.uint_method = HistogramParams::HybridUintMethod::kBest;
      params.clustering = HistogramParams::ClusteringType::kFastest;
      params.lz77_method = HistogramParams::LZ77Method::kNone;
      params.force_huffman = prefix;
      state.calls = 0;
      RoundtripTestcase(16, ANS_MAX_ALPHABET_SIZE, values, params);
      EXPECT_GT(state.calls, 0u);
      state.fail = true;
      std::vector<std::vector<Token>> streams{values};
      EntropyEncodingData codes;
      auto* memory = test::MemoryManager();
      BitWriter writer(memory);
      EXPECT_FALSE(BuildAndEncodeHistograms(memory, params, 16, streams, &codes,
          &writer, LayerType::Header, nullptr).ok());
      state.fail = false;
    }
  }
}

TEST(ANSTest, LosslessHybridSeparatesTreeAndEntropyEffort) {
  CompressParams input;
  input.SetLossless();
  input.speed_tier = SpeedTier::kSquirrel;
  input.lossless_precise_entropy = true;
  const auto hybrid = HistogramParams::ForModular(input, {}, false);
  EXPECT_EQ(input.speed_tier, SpeedTier::kSquirrel);
  EXPECT_EQ(hybrid.lz77_method, HistogramParams::LZ77Method::kOptc256);
  EXPECT_EQ(hybrid.clustering, HistogramParams::ClusteringType::kBest);
  EXPECT_EQ(hybrid.uint_method, HistogramParams::HybridUintMethod::kBest);
  EXPECT_EQ(hybrid.ans_histogram_strategy, HistogramParams::ANSHistogramStrategy::kPrecise);
  input.modular_mode = false;
  const auto lossy = HistogramParams::ForModular(input, {}, false);
  input.lossless_precise_entropy = false;
  const auto control = HistogramParams::ForModular(input, {}, false);
  EXPECT_EQ(lossy.lz77_method, control.lz77_method);
  EXPECT_EQ(lossy.clustering, control.clustering);
  EXPECT_EQ(lossy.uint_method, control.uint_method);
}

TEST(ANSTest, DisposableProbeChangesOnlyLosslessLz77) {
  CompressParams input;
  input.SetLossless();
  input.speed_tier = SpeedTier::kTortoise;
  const auto full = HistogramParams::ForModular(input, {}, false);
  for (uint8_t mode : {1, 2}) {
    input.lossless_probe_lz77 = mode;
    const auto probe = HistogramParams::ForModular(input, {}, false);
    EXPECT_EQ(input.speed_tier, SpeedTier::kTortoise);
    EXPECT_EQ(probe.lz77_method, mode == 1
        ? HistogramParams::LZ77Method::kLZ77b3w3t
        : HistogramParams::LZ77Method::kOptc8);
    EXPECT_EQ(probe.clustering, full.clustering);
    EXPECT_EQ(probe.uint_method, full.uint_method);
    EXPECT_EQ(probe.ans_histogram_strategy, full.ans_histogram_strategy);
    input.modular_mode = false;
    const auto lossy = HistogramParams::ForModular(input, {}, false);
    input.lossless_probe_lz77 = 0;
    EXPECT_EQ(lossy.lz77_method, HistogramParams::ForModular(input, {}, false).lz77_method);
    input.SetLossless();
  }
}

TEST(ANSTest, WordMatchLengthHandlesTailsAndOverlaps) {
  const auto check = [](auto tag) {
    using Value = decltype(tag);
    std::vector<Value> a(300), b(300);
    for (size_t offset = 0; offset < 8; ++offset) {
      for (size_t count = 0; count <= 257; ++count) {
        for (size_t mismatch : {size_t{0}, count / 2, count ? count - 1 : 0, count}) {
          for (size_t i = 0; i < a.size(); ++i) a[i] = b[i] = static_cast<Value>(i * 0x9137u);
          if (mismatch < count) b[offset + mismatch] ^= Value{1};
          EXPECT_EQ(LZ77WordMatchLength(a.data() + offset, b.data() + offset, count), mismatch);
          EXPECT_EQ(LZ77WordMatchLength(a.data() + offset, a.data() + offset, count), count);
        }
      }
    }
    for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<Value>((i % 7) * 257);
    for (size_t distance = 1; distance <= 17; ++distance) {
      for (size_t count = 0; count < 257; ++count) {
        size_t expected = 0;
        while (expected < count && a[expected] == a[expected + distance]) ++expected;
        EXPECT_EQ(LZ77WordMatchLength(a.data(), a.data() + distance, count), expected);
      }
    }
  };
  check(uint8_t{});
  check(uint16_t{});
  check(uint32_t{});
}

TEST(ANSTest, LosslessClusterParallelPreservesHistogramMass) {
  std::vector<Histogram> input(8192);
  std::vector<int64_t> expected(256);
  for (size_t i = 0; i < input.size(); ++i) {
    if (i % 17 == 0) continue;
    // Empty, repeated, single-symbol and different-length distributions.
    for (size_t k = 0; k < 1 + i % 37; ++k) {
      const size_t symbol = (i % 31 + k * k) % (i % 3 ? 32 : 256);
      input[i].Add(symbol);
      ++expected[symbol];
    }
  }
  for (int threads : {0, 4}) {
    test::ThreadPoolForTests delegate(threads);
    struct State {
      ThreadPool* pool;
      size_t calls = 0;
      bool fail = false;
    } state{delegate.get()};
    ThreadPool pool(
        [](void* opaque, void* context, JxlParallelRunInit init,
           JxlParallelRunFunction run, uint32_t begin,
           uint32_t end) -> JxlParallelRetCode {
          auto* s = static_cast<State*>(opaque);
          ++s->calls;
          if (s->fail) return JXL_PARALLEL_RET_RUNNER_ERROR;
          return s->pool->runner()(s->pool->runner_opaque(), context, init,
                                  run, begin, end);
        }, &state);
    for (bool prefix : {false, true}) {
      for (auto mode : {HistogramParams::ClusteringType::kFastest,
                        HistogramParams::ClusteringType::kFast,
                        HistogramParams::ClusteringType::kBest}) {
        SCOPED_TRACE(threads);
        SCOPED_TRACE(prefix);
        SCOPED_TRACE(static_cast<int>(mode));
        HistogramParams params;
        params.lossless_lz77_cost_spans = true;
        params.lossless_lz77_pool = &pool;
        params.force_huffman = prefix;
        params.clustering = mode;
        std::vector<Histogram> clustered;
        std::vector<uint32_t> symbols;
        state.calls = 0;
        ASSERT_TRUE(ClusterHistograms(params, input, 32, &clustered, &symbols));
        ASSERT_GT(state.calls, 0u);
        ASSERT_EQ(symbols.size(), input.size());
        std::vector<int64_t> actual(expected.size());
        for (const auto& h : clustered) {
          for (size_t k = 0; k < h.counts.size(); ++k) actual[k] += h.counts[k];
        }
        EXPECT_EQ(actual, expected);
        for (uint32_t symbol : symbols) EXPECT_LT(symbol, clustered.size());

        state.fail = true;
        clustered.clear();
        EXPECT_FALSE(ClusterHistograms(params, input, 32, &clustered, &symbols));
        state.fail = false;

        params.lossless_lz77_cost_spans = false;
        clustered.clear();
        state.calls = 0;
        ASSERT_TRUE(ClusterHistograms(params, input, 32, &clustered, &symbols));
        EXPECT_EQ(state.calls, 0u);
      }
    }
  }
}

TEST(ANSTest, LosslessClusterManyContextsRoundtrip) {
  constexpr size_t kContexts = 8192;
  std::vector<Token> tokens;
  for (size_t c = 0; c < kContexts; ++c) {
    for (size_t i = 0; i < 1 + c % 29; ++i) {
      tokens.emplace_back(c, (c % 71 + i * 19) % 257);
    }
  }
  for (int threads : {0, 4}) {
    test::ThreadPoolForTests pool(threads);
    for (bool prefix : {false, true}) {
      HistogramParams params;
      params.lossless_lz77_cost_spans = true;
      params.lossless_lz77_pool = pool.get();
      params.lz77_method = HistogramParams::LZ77Method::kNone;
      params.force_huffman = prefix;
      RoundtripTestcase(kContexts, 257, tokens, params);
    }
  }
}

TEST(ANSTest, WeightedUintHistogram) {
  std::vector<uint32_t> data;
  for (uint32_t i = 0; i < 65536; ++i) data.push_back(i);
  uint32_t random = 1729;
  for (size_t i = 0; i < 32768; ++i) {
    random = random * 1664525u + 1013904223u;
    data.push_back(random);
  }
  for (uint32_t bits = 0; bits < 32; ++bits) {
    const uint32_t value = uint32_t{1} << bits;
    for (size_t i = 0; i < 31; ++i) {
      data.push_back(value - 1);
      data.push_back(value);
      data.push_back(value + 1);
      data.push_back(~uint32_t{0});
    }
  }
  WeightedUintHistogram weighted;
  ASSERT_TRUE(weighted.Build(data.data(), data.size()));
  EXPECT_LE(weighted.size(), WeightedUintHistogram::kBins);
  for (uint32_t split = 0; split <= 12; ++split) {
    for (uint32_t msb = 0; msb <= 2 && msb <= split; ++msb) {
      for (uint32_t lsb = 0; lsb <= 5 && msb + lsb <= split; ++lsb) {
        const HybridUintConfig cfg(split, msb, lsb);
        SCOPED_TRACE(testing::Message() << split << "," << msb << "," << lsb);
        ASSERT_TRUE(WeightedUintHistogram::Supports(cfg));
        Histogram expected, actual;
        uint32_t expected_bits = 0;
        for (uint32_t value : data) {
          uint32_t token, nbits, bits;
          cfg.Encode(value, &token, &nbits, &bits);
          expected.Add(token);
          expected_bits += nbits;
        }
        actual.EnsureCapacity(expected.counts.size());
        EXPECT_EQ(weighted.AddTo(cfg, &actual), expected_bits);
        actual.Condition();
        EXPECT_EQ(actual.counts, expected.counts);
        EXPECT_EQ(actual.total_count, expected.total_count);
      }
    }
  }
  EXPECT_FALSE(WeightedUintHistogram::Supports(HybridUintConfig(13, 0, 0)));
  EXPECT_FALSE(WeightedUintHistogram::Supports(HybridUintConfig(4, 3, 0)));
  EXPECT_FALSE(WeightedUintHistogram::Supports(HybridUintConfig(6, 0, 6)));
  EXPECT_FALSE(weighted.Build(data.data(), 4095));
  EXPECT_EQ(weighted.size(), 0u);
  EXPECT_FALSE(weighted.Build(nullptr, 0));
  EXPECT_FALSE(weighted.Build(nullptr, static_cast<size_t>(INT32_MAX) + 1));
}

TEST(ANSTest, WeightedUintRoundtrip) {
  std::vector<Token> values;
  uint32_t random = 314159;
  for (size_t i = 0; i < 40000; ++i) {
    random = random * 1664525u + 1013904223u;
    values.emplace_back(i % 3, i % 5 == 0 ? random : (i * 3) % 511);
  }
  for (bool huffman : {false, true}) {
    for (bool lossless : {false, true}) {
      for (auto method : {HistogramParams::HybridUintMethod::kFast,
                          HistogramParams::HybridUintMethod::kBest}) {
        HistogramParams params;
        params.lossless_lz77_cost_spans = lossless;
        params.force_huffman = huffman;
        params.uint_method = method;
        params.lz77_method = HistogramParams::LZ77Method::kNone;
        RoundtripTestcase(3, ANS_MAX_ALPHABET_SIZE, values, params);
      }
    }
  }
}

TEST(ANSTest, ParallelLZ77StreamsRoundtrip) {
  const char* fast_env = std::getenv("JXL_LOSSLESS_LZ77_FAST_PARALLEL");
  const bool fast = fast_env && std::strcmp(fast_env, "1") == 0;
  const char* arena_env = std::getenv("JXL_LOSSLESS_LZ77_STREAM_ARENA");
  const bool indexed_arena = arena_env && std::strcmp(arena_env, "1") == 0;
  using Method = HistogramParams::LZ77Method;
  std::vector<Method> methods{Method::kOptc256};
  if (fast) {
    methods.push_back(Method::kLZ77b7w3t);
    methods.push_back(Method::kLZ77b3w3f);
  }
  std::vector<std::vector<Token>> source(8);
  for (size_t s = 0; s < source.size() - 1; ++s) {
    const size_t size = s == 0 ? 3 : 16384 + s * 8192;
    for (size_t i = 0; i < size; ++i) {
      source[s].emplace_back((i + s) % 3, ((i % 1024) * 97 + s * 31) % 257);
    }
  }
  for (int threads : {0, 4}) {
    test::ThreadPoolForTests delegate(threads);
    struct State {
      ThreadPool* pool;
      size_t calls = 0;
      bool fail = false;
      size_t fail_on_call = 0;
    };
    State state{delegate.get()};
    ThreadPool pool(
        [](void* opaque, void* context, JxlParallelRunInit init,
           JxlParallelRunFunction run, uint32_t begin,
           uint32_t end) -> JxlParallelRetCode {
          auto* state = static_cast<State*>(opaque);
          ++state->calls;
          if (state->fail || state->calls == state->fail_on_call) {
            return JXL_PARALLEL_RET_RUNNER_ERROR;
          }
          return state->pool->runner()(state->pool->runner_opaque(), context,
                                       init, run, begin, end);
        },
        &state);
    for (Method method : methods) {
      for (bool huffman : {false, true}) {
        SCOPED_TRACE(threads);
        SCOPED_TRACE(huffman);
        HistogramParams params;
        params.lossless_lz77_cost_spans = true;
        params.lossless_lz77_pool = &pool;
        params.lossless_bucket_matcher = indexed_arena;
        params.lz77_method = method;
        params.force_huffman = huffman;
        params.image_widths = {0, 1, 17, 257, 0, 31, 128, 0};
        auto streams = source;
        BitWriter header(test::MemoryManager());
        EntropyEncodingData codes;
        state.calls = 0;
        JXL_TEST_ASSIGN_OR_DIE(
            size_t cost, BuildAndEncodeHistograms(test::MemoryManager(), params,
                                                  3, streams, &codes, &header,
                                                  LayerType::Header, nullptr));
        (void)cost;
        ASSERT_GT(state.calls, 0u);
        ASSERT_TRUE(codes.lz77.enabled);
        header.ZeroPadToByte();
        BitReader header_reader(header.GetSpan());
        ANSCode decoded_codes;
        std::vector<uint8_t> contexts;
        ASSERT_TRUE(DecodeHistograms(test::MemoryManager(), &header_reader, 3,
                                     &decoded_codes, &contexts));
        ASSERT_TRUE(header_reader.Close());
        for (size_t s = 0; s < streams.size(); ++s) {
          BitWriter body(test::MemoryManager());
          ASSERT_TRUE(WriteTokens(streams[s], codes, 0, &body,
                                  LayerType::Header, nullptr));
          body.ZeroPadToByte();
          BitReader bits(body.GetSpan());
          JXL_TEST_ASSIGN_OR_DIE(
              ANSSymbolReader reader,
              ANSSymbolReader::Create(&decoded_codes, &bits,
                                      params.image_widths[s]));
          for (const auto& symbol : source[s]) {
            ASSERT_EQ(reader.ReadHybridUint(symbol.context, &bits, contexts),
                      symbol.value);
          }
          ASSERT_TRUE(reader.CheckANSFinalState());
          ASSERT_TRUE(bits.Close());
        }
        state.fail = true;
        streams = source;
        BitWriter failed_header(test::MemoryManager());
        EXPECT_FALSE(BuildAndEncodeHistograms(test::MemoryManager(), params, 3,
                                              streams, &codes, &failed_header,
                                              LayerType::Header, nullptr)
                         .ok());
        state.fail = false;
        if (fast && method == Method::kOptc256) {
          // The first-pass pool succeeds; failure in the subsequent optimal
          // parser must propagate as well.
          state.calls = 0;
          state.fail_on_call = 2;
          streams = source;
          BitWriter failed_optimal(test::MemoryManager());
          EXPECT_FALSE(BuildAndEncodeHistograms(
                           test::MemoryManager(), params, 3, streams, &codes,
                           &failed_optimal, LayerType::Header, nullptr)
                           .ok());
          EXPECT_EQ(state.calls, 2u);
          state.fail_on_call = 0;
        }
      }
    }
    HistogramParams excluded;
    excluded.lossless_lz77_pool = &pool;
    excluded.lz77_method = Method::kLZ77b7w3t;
    state.calls = 0;
    state.fail = true;
    auto streams = source;
    BitWriter header(test::MemoryManager());
    EntropyEncodingData codes;
    EXPECT_TRUE(BuildAndEncodeHistograms(test::MemoryManager(), excluded, 3,
                                         streams, &codes, &header,
                                         LayerType::Header, nullptr)
                    .ok());
    EXPECT_EQ(state.calls, 0u);
  }
}

TEST(ANSTest, LZ77CompactContextsRoundtrip) {
  const char* compact = std::getenv("JXL_LOSSLESS_LZ77_COMPACT");
  if (!compact || std::strcmp(compact, "1") != 0) GTEST_SKIP();
  // Match starts, copied source positions and endpoints have different
  // contexts. Backtracking must recover the start, not the source or end.
  for (size_t tail : {0u, 1u, 7u, 15u}) {
    std::vector<Token> source;
    for (size_t i = 0; i < 65536 + tail; ++i) {
      source.emplace_back((i * 7 + i / 31) % 19,
                          i % 257 < 233 ? (i % 257) * 3 : (i * 17) % 511);
    }
    for (size_t width : {0u, 1u, 113u}) {
      for (bool huffman : {false, true}) {
        SCOPED_TRACE(testing::Message() << tail << " width=" << width
                                       << " huffman=" << huffman);
        HistogramParams params;
        params.lossless_lz77_cost_spans = true;
        params.lz77_method = HistogramParams::LZ77Method::kOptc256;
        params.image_widths = {width};
        params.force_huffman = huffman;
        RoundtripTestcase(19, ANS_MAX_ALPHABET_SIZE, source, params, true);
      }
    }
  }
}

TEST(ANSTest, EntropyToolRegistryRoundtrip) {
  std::vector<Token> values;
  for (size_t i = 0; i < 4096; ++i) {
    values.emplace_back(i % 3, i % 97 < 70 ? 0 : (i % 17) * 65537);
  }
  ASSERT_EQ(EntropyTools().size(), 189);
  for (size_t i = 0; i < EntropyTools().size(); ++i) {
    SCOPED_TRACE(i);
    HistogramParams params;
    EntropyTools()[i].Apply(&params);
    RoundtripTestcase(3, ANS_MAX_ALPHABET_SIZE, values, params);
  }
}

TEST(ANSTest, DirectEntropyRoundtrip) {
  std::vector<Token> values;
  for (size_t i = 0; i < 2048; ++i) {
    values.emplace_back(i % 3, i % 97 < 80 ? 0 : (i % 17) * 65537);
  }
  for (int lz : {0, 1, 2}) {
    for (auto uint_method : {HistogramParams::HybridUintMethod::kNone,
                            HistogramParams::HybridUintMethod::k000,
                            HistogramParams::HybridUintMethod::kContextMap}) {
      for (auto strategy : {HistogramParams::ANSHistogramStrategy::kDirect0,
                            HistogramParams::ANSHistogramStrategy::kDirect6}) {
        for (size_t clusters : {1, 4, 8}) {
          DirectAudit audit;
          HistogramParams params;
          params.direct_audit = &audit;
          params.direct_clusters = clusters;
          params.lz77_method = static_cast<HistogramParams::LZ77Method>(lz);
          params.uint_method = uint_method;
          params.ans_histogram_strategy = strategy;
          // Many unused contexts and even a fully empty stream must not produce
          // empty ANS symbol tables or lose context labels during compaction.
          RoundtripTestcase(20, ANS_MAX_ALPHABET_SIZE, values, params);
          RoundtripTestcase(20, ANS_MAX_ALPHABET_SIZE, {}, params);
          RoundtripTestcase(1, ANS_MAX_ALPHABET_SIZE, {}, params);
          EXPECT_EQ(audit.forbidden_searches, 0);
          EXPECT_LE(audit.precision_trials.load(), audit.histogram_calls.load());
        }
      }
    }
  }
}

TEST(ANSTest, DirectModelFormatIsolation) {
  std::string valid = "JXL_DIRECT_PLAN_V1 39 2 1 9071 "
                      "1 1 -1 0 -1 -1 2 1 1 -1 0 -1 -1 -3";
  EntropyToolModel model;
  std::istringstream old(valid);
  EXPECT_FALSE(model.Read(old));
  std::istringstream input(valid);
  ASSERT_TRUE(model.Read(input, "JXL_DIRECT_PLAN_V1", 9072));
  EXPECT_EQ(model.actions()[0], 9071);
  EXPECT_EQ(model.domain(), 2);
}

TEST(ANSTest, EntropyToolFeaturesUseRealNeighbours) {
  std::vector<std::vector<Token>> tokens(1);
  for (size_t i = 0; i < 65536; ++i) tokens[0].emplace_back(0, i % 2);
  HistogramParams params;
  params.tool_policy_domain = 2;
  auto features = DescribeEntropyState(params, 1, tokens, LayerType::Ac);
  EXPECT_EQ(features[0], 2);
  EXPECT_EQ(features[6], 0);
  EXPECT_NEAR(features[5], 0.5, 0.02);
  for (const auto& tool : EntropyTools()) {
    DescribeEntropyTool(tool, &features);
    double sum = 0;
    for (size_t i = kToolStateFeatures; i < kToolFeatures; ++i) sum += features[i];
    EXPECT_EQ(sum, 4);
  }
}

TEST(ANSTest, EntropyToolModelValidation) {
  const std::string valid =
      "JXL_ENTROPY_TOOLS_V1 39 1 2 0 188 "
      "1 3 2 1.5 1 2 0 -1 0 -1 -1 2 -1 0 -1 -1 4 "
      "1 1 -1 0 -1 -1 -3";
  EntropyToolModel model;
  std::istringstream input(valid);
  ASSERT_TRUE(model.Read(input));
  ToolFeatures features{};
  EXPECT_EQ(model.Predict(features)[0], 2);
  features[2] = 2;
  EXPECT_EQ(model.Predict(features)[0], 4);
  EXPECT_EQ(model.Predict(features)[1], -3);
  for (const std::string& bad : std::vector<std::string>{
           "JXL_ENTROPY_TOOLS_V2 39 1 1 0",
           "JXL_ENTROPY_TOOLS_V1 39 1 2 0 0",
           "JXL_ENTROPY_TOOLS_V1 39 1 1 189",
           "JXL_ENTROPY_TOOLS_V1 39 1 1 0 1 1 0 1 0 0 2",
           valid + " trailing"}) {
    std::istringstream stream(bad);
    EXPECT_FALSE(model.Read(stream));
  }
  // Failed loads cannot partially replace a previously validated model.
  EXPECT_EQ(model.Predict(features)[0], 4);
}

TEST(ANSTest, SingleSymbolRoundtrip) {
  for (uint32_t i = 0; i < ANS_MAX_ALPHABET_SIZE; i++) {
    RoundtripTestcase(2, ANS_MAX_ALPHABET_SIZE, {{0, i}});
  }
  for (uint32_t i = 0; i < ANS_MAX_ALPHABET_SIZE; i++) {
    RoundtripTestcase(2, ANS_MAX_ALPHABET_SIZE,
                      std::vector<Token>(1024, {0, i}));
  }
}

TEST(ANSTest, ClusterEntropyLowerBound) {
  uint32_t rng = 0x243f6a88;
  for (size_t trial = 0; trial < 1000; ++trial) {
    Histogram a, b;
    const size_t count = 1 + trial % 256;
    a.EnsureCapacity(count);
    b.EnsureCapacity(count);
    for (size_t i = 0; i < count; ++i) {
      rng = rng * 1664525u + 1013904223u;
      a.counts[i] = (rng & 7) ? 0 : (rng >> 8) % 100000;
      rng = rng * 1664525u + 1013904223u;
      b.counts[i] = (rng & 3) ? 0 : (rng >> 8) % 100000;
    }
    a.Condition();
    b.Condition();
    const double bound = ClusterMergeLowerBound(
        MakeClusterEntropySketch(a.counts), MakeClusterEntropySketch(b.counts));
    Histogram merged = a;
    merged.AddHistogram(b);
    JXL_TEST_ASSIGN_OR_DIE(float cost, merged.ANSPopulationCost());
    EXPECT_LE(bound, cost);
  }
}

TEST(ANSTest, LZ77DistancePriceClasses) {
  const auto price = [](uint32_t value) {
    uint32_t token, bits, unused;
    HybridUintConfig().Encode(value, &token, &bits, &unused);
    return float(token * token + bits) / 8;
  };
  const LZ77DistancePrices prices(price);
  for (uint32_t value = 0; value < (1u << 20); ++value)
    ASSERT_EQ(prices(value), price(value));
  for (uint64_t bit = 4; bit < 32; ++bit) {
    for (uint64_t top = 4; top < 8; ++top) {
      const uint64_t begin = top << (bit - 2), end = (top + 1) << (bit - 2);
      EXPECT_EQ(prices(begin), price(begin));
      EXPECT_EQ(prices(end - 1), price(end - 1));
    }
  }
}

TEST(ANSTest, LZ77DistanceBoundPrunesOnlyDominatedMatches) {
  for (uint32_t model : {0u, 1u, 2u}) {
    const auto price = [=](uint32_t value) {
      uint32_t token, nbits, unused;
      HybridUintConfig().Encode(value, &token, &nbits, &unused);
      return float(nbits) + float(model == 0 ? token : (token * 47 + model) % 31) / 8;
    };
    for (uint32_t maximum : {0u, 15u, 16u, 31u, 32u, 1055u, 65539u}) {
      const auto bound = MinimumLZ77DistancePrice(maximum, price);
      float best = std::numeric_limits<float>::infinity();
      uint32_t first = 0;
      for (uint32_t d = 0; d <= maximum; ++d) {
        if (price(d) < best) { best = price(d); first = d; }
      }
      EXPECT_EQ(bound.cost, best);
      EXPECT_EQ(bound.symbol, first);
    }
    const auto wide = MinimumLZ77DistancePrice(UINT32_MAX, price);
    EXPECT_LE(wide.cost, price(UINT32_MAX));
    for (size_t width : {0u, 1u, 17u}) {
      std::vector<Token> input;
      for (size_t i = 0; i < 2048; ++i)
        input.emplace_back(i % 3, i % 211 < 170 ? 0 : (i * 71) % 37);
      LZ77StaticIndex<> index(input.data(), input.size(), 1024, 3, width, true);
      const auto bound = MinimumLZ77DistancePrice(1024 + kNumSpecialDistances, price);
      size_t ordinary_prices = 0, bounded_prices = 0;
      for (size_t pos = 0; pos <= input.size(); pos += 5) {
        for (size_t limit : {0u, 2u, 7u, 257u}) {
          std::vector<float> expected(limit + 1, std::numeric_limits<float>::infinity());
          auto actual = expected;
          const auto collect = [&](std::vector<float>& scores) {
            return [&](size_t length, uint32_t d) {
              for (size_t n = 3; n <= length; ++n)
                scores[n] = std::min(scores[n], price(d));
            };
          };
          index.FindMatches(pos, [&](uint32_t d) { ++ordinary_prices; return price(d); },
                            collect(expected), nullptr, limit);
          index.FindMatches(pos, [&](uint32_t d) { ++bounded_prices; return price(d); },
                            collect(actual), nullptr, limit, 0, nullptr, bound);
          ASSERT_EQ(actual, expected) << "model=" << model << " width=" << width
                                      << " pos=" << pos << " limit=" << limit;
        }
      }
      EXPECT_LE(bounded_prices, ordinary_prices);
      if (model == 0) EXPECT_LT(bounded_prices, ordinary_prices);
    }
  }
}

TEST(ANSTest, LZ77CostSpans) {
  const auto len_cost = [](size_t value) {
    uint32_t token, bits, nbits;
    HybridUintConfig(0, 0, 0).Encode(value, &token, &nbits, &bits);
    return nbits + (token % 7) * 0.725f;
  };
  const auto dist_cost = [](uint32_t value) {
    uint32_t token, bits, nbits;
    HybridUintConfig().Encode(value, &token, &nbits, &bits);
    return nbits + (token % 11) * 0.375f;
  };
  for (size_t min_length : {3, 5, 16}) {
    for (size_t count : {0, 1, 2, 3, 4, 7, 8, 9, 255, 256, 257, 65539}) {
      for (int pattern = 0; pattern < 3; ++pattern) {
        SCOPED_TRACE(testing::Message() << min_length << ':' << count << ':'
                                       << pattern);
        std::vector<uint32_t> distances(min_length + count);
        for (size_t j = min_length; j < distances.size(); ++j) {
          distances[j] = pattern == 0 ? 17 : pattern == 1 ? j / 31 : j % 13;
        }
        size_t visits = 0, length_calls = 0, distance_calls = 0;
        VisitLZ77CostSpans(
            min_length, distances, 1234.375f,
            [&](size_t len) { ++length_calls; return len_cost(len); },
            [&](uint32_t dist) { ++distance_calls; return dist_cost(dist); },
            [&](size_t j, uint32_t dist, float cost) {
              EXPECT_EQ(j, min_length + visits++);
              EXPECT_EQ(dist, distances[j]);
              EXPECT_EQ(cost, 1234.375f + (len_cost(j - min_length) +
                                         dist_cost(distances[j])));
            });
        EXPECT_EQ(visits, count);
        EXPECT_LE(length_calls, 18);
        EXPECT_LE(distance_calls, count);
        if (pattern == 0) EXPECT_EQ(length_calls, distance_calls);
      }
    }
  }
}

TEST(ANSTest, LZ77CostRange) {
  const auto length_cost = [](size_t value) {
    uint32_t token, bits, nbits;
    HybridUintConfig(0, 0, 0).Encode(value, &token, &nbits, &bits);
    return nbits + (token % 7) * 0.725f;
  };
  for (size_t minimum : {3u, 7u}) {
    for (size_t count : {0u, 1u, 2u, 17u, 257u, 4097u}) {
      for (size_t offset : {0u, 1u, 7u, 30u, 127u}) {
        for (bool boundary_only : {false, true}) {
          const size_t begin = minimum + offset;
          const size_t end = begin + count;
          std::vector<bool> visited(count);
          size_t visits = 0;
          VisitLZ77CostRange(
              minimum, begin, end, 17, 2.375f, 16777216.0, length_cost,
              [&](size_t length, uint32_t distance, double cost) {
                ASSERT_GE(length, begin);
                ASSERT_LT(length, end);
                EXPECT_FALSE(visited[length - begin]);
                visited[length - begin] = true;
                ++visits;
                EXPECT_EQ(distance, 17u);
                EXPECT_EQ(cost, 16777216.0 + (length_cost(length - minimum) + 2.375f));
              }, boundary_only);
          if (!boundary_only) EXPECT_EQ(visits, count);
          if (count) {
            EXPECT_TRUE(visited.front());
            EXPECT_TRUE(visited.back());
          }
          if (boundary_only && count > 256) EXPECT_LT(visits, count);
        }
      }
    }
  }
}

TEST(ANSTest, LZ77CostSpansLargePrefix) {
  // A fractional-bit improvement must survive a large accumulated cost when
  // the caller explicitly selects the double-precision experimental parser.
  const double prefix = 16777216.0;
  const std::vector<uint32_t> distances{0, 0, 0, 1, 1, 2};
  size_t visits = 0;
  VisitLZ77CostSpans(3, distances, prefix,
      [](size_t length) { return length == 0 ? 0.125f : 0.375f; },
      [](uint32_t distance) { return distance == 1 ? 0.25f : 0.5f; },
      [&](size_t length, uint32_t distance, double cost) {
        ++visits;
        EXPECT_EQ(distance, distances[length]);
        EXPECT_EQ(cost - prefix, length == 3 ? 0.375 : length == 4 ? 0.625 : 0.875);
      });
  EXPECT_EQ(visits, 3u);
}

TEST(ANSTest, LZ77CostSpansRoundtrip) {
  std::vector<Token> values;
  for (size_t i = 0; i < 12000; ++i) {
    values.emplace_back(i % 3, i % 97 < 70 ? 0 : (i % 17) * 65537);
  }
  for (auto method : {HistogramParams::LZ77Method::kOptc1,
                      HistogramParams::LZ77Method::kOptc3,
                      HistogramParams::LZ77Method::kOptc8,
                      HistogramParams::LZ77Method::kOptc256}) {
    for (bool spans : {false, true}) {
      for (bool huffman : {false, true}) {
        HistogramParams params;
        params.lz77_method = method;
        params.lossless_lz77_cost_spans = spans;
        params.force_huffman = huffman;
        RoundtripTestcase(3, ANS_MAX_ALPHABET_SIZE, values, params, true);
      }
    }
  }
}

TEST(ANSTest, LZ77MatchDominance) {
  LZ77MatchDominance ties;
  EXPECT_EQ(ties.RequiredLength(1.0f, 7, 3), 3u);
  ties.Add(10, 7, 1.0f);
  EXPECT_EQ(ties.RequiredLength(1.0f, 6, 3), 3u);
  EXPECT_EQ(ties.RequiredLength(1.0f, 7, 3), 11u);
  EXPECT_EQ(ties.RequiredLength(2.0f, 0, 3), 11u);
  EXPECT_EQ(ties.RequiredLength(0.5f, 100, 3), 3u);
  ties.Add(100, 4, 0.5f);
  EXPECT_EQ(ties.RequiredLength(1.0f, 7, 3), 101u);
  EXPECT_EQ(ties.RequiredLength(0.5f, 3, 3), 3u);
  for (uint32_t seed = 0; seed < 32; ++seed) {
    LZ77MatchDominance frontier;
    std::vector<LZ77Match> seen;
    const auto cost = [&](uint32_t distance) {
      return float((distance * 7 + seed) % 47) * 0.25f;
    };
    for (uint32_t count = 0; count < 128; ++count) {
      for (uint32_t distance = 0; distance < 64; ++distance) {
        size_t exact = 3;
        for (const auto& previous : seen) {
          if (cost(previous.distance) < cost(distance) ||
              (cost(previous.distance) == cost(distance) &&
               previous.distance <= distance)) {
            exact = std::max(exact, size_t{previous.length} + 1);
          }
        }
        const size_t required = frontier.RequiredLength(cost(distance), distance, 3);
        EXPECT_LE(required, exact);
        EXPECT_GE(required, 3u);
        if (count <= 16) EXPECT_EQ(required, exact);
      }
      const LZ77Match next{3 + (count * 23 + seed) % 137, (count * 17) % 64};
      frontier.Add(next.length, next.distance, cost(next.distance));
      seen.push_back(next);
    }
  }
}

TEST(ANSTest, LZ77StaticIndexBounds) {
  std::vector<Token> values;
  for (size_t i = 0; i < 4096; ++i) {
    const uint32_t value = i % 257 < 170 ? 0 :
        static_cast<uint32_t>((i % 31) * 65537u);
    values.emplace_back(i % 3, value);
  }
  for (size_t window : {32u, 128u, 1024u}) {
    for (size_t width : {0u, 1u, 17u}) {
      LZ77StaticIndex<> index(values.data(), values.size(), window, 3, width);
      size_t matches = 0;
      for (size_t pos = 0; pos < values.size(); ++pos) {
        index.FindMatches(pos, [](uint32_t distance) { return float(distance % 19); },
                          [&](size_t length, uint32_t symbol) {
          const size_t distance = width && symbol < kNumSpecialDistances
              ? SpecialDistance(symbol, width)
              : symbol + 1 - (width ? kNumSpecialDistances : 0);
          ASSERT_GT(distance, 0u);
          ASSERT_LE(distance, pos);
          ASSERT_LT(distance, window);
          ASSERT_GE(length, 3u);
          ASSERT_LE(pos + length, values.size());
          for (size_t k = 0; k < length; ++k) {
            ASSERT_EQ(values[pos + k].value, values[pos - distance + k].value);
          }
          ++matches;
        });
      }
      EXPECT_GT(matches, 0u);
    }
  }
}

TEST(ANSTest, LZ77RunKeysSparseRepetition) {
  std::vector<Token> input;
  for (size_t row = 0; row <= 128; ++row) {
    input.insert(input.end(), 97, Token(0, 0));
    for (size_t x = 0; x < 8; ++x)
      input.emplace_back(0, 1 + x + (row == 128 ? 0 : row * 11));
  }
  LZ77StaticIndex<> index(input.data(), input.size(), 16384, 3, 0,
                          true, false, true);
  for (size_t pos = 128 * 105; pos < input.size(); ++pos) {
    size_t longest = 0;
    index.FindMatches(pos, [](uint32_t d) { return float(d % 23); },
        [&](size_t length, uint32_t symbol) {
          const size_t distance = size_t{symbol} + 1;
          ASSERT_GT(distance, 0u);
          ASSERT_LE(distance, pos);
          ASSERT_LT(distance, 16384u);
          ASSERT_LE(pos + length, input.size());
          for (size_t i = 0; i < length; ++i)
            ASSERT_EQ(input[pos + i].value, input[pos - distance + i].value);
          longest = std::max(longest, length);
        });
    if (pos == 128 * 105) EXPECT_EQ(longest, 105u);
  }
}

TEST(ANSTest, LZ77RepeatedBucketUpdates) {
  for (uint16_t initial = 0; initial < 7; ++initial) {
    for (size_t count : {0u, 1u, 3u, 7u, 8u, 101u, 10000u}) {
      std::array<uint32_t, 7> expected{{1, 2, 3, 4, 5, 6, 7}};
      auto actual = expected;
      uint16_t next = initial, size = initial;
      uint16_t next_fast = next, size_fast = size;
      for (size_t pos = 19; pos < 19 + count; ++pos) {
        expected[next] = pos;
        next = (next + 1) % 7;
        size = std::min<uint16_t>(7, size + 1);
      }
      LZ77InsertRepeatedBucket(&actual, &next_fast, &size_fast, 19, 19 + count);
      EXPECT_EQ(expected, actual);
      EXPECT_EQ(next, next_fast);
      EXPECT_EQ(size, size_fast);
    }
  }
}

TEST(ANSTest, LZ77WorkCutsKeepSeedBoundaries) {
  std::vector<Token> input;
  for (size_t i = 0; i < 4096; ++i) input.emplace_back(0, i < 3072 ? i % 17 : 0);
  for (size_t parts : {1u, 4u, 16u, 64u}) {
    const auto cuts = LZ77WorkCuts(input, input, parts, 3);
    EXPECT_EQ(cuts.front(), 0u);
    EXPECT_EQ(cuts.back(), input.size());
    EXPECT_LE(cuts.size(), parts + 1);
    for (size_t i = 1; i < cuts.size(); ++i) EXPECT_GT(cuts[i], cuts[i - 1]);
    if (parts > 1) EXPECT_LT(cuts[1], input.size() / parts);
  }
  std::vector<Token> seed(input.begin(), input.begin() + 17);
  seed.emplace_back(0, 3072 - 17 - 3);
  seed.back().is_lz77_length = true;
  seed.emplace_back(1, 16);
  seed.insert(seed.end(), input.begin() + 3072, input.end());
  const auto cuts = LZ77WorkCuts(input, seed, 16, 3);
  for (size_t cut : cuts) EXPECT_TRUE(cut <= 17 || cut >= 3072);
}

TEST(ANSTest, LZ77StaticIndexSIMDFilter) {
  std::vector<Token> input;
  for (size_t i = 0; i < 8192; ++i)
    input.emplace_back(i % 5, i % 127 < 80 ? i % 11 : (i * 71) % 4093);
  for (size_t width : {0u, 1u, 17u}) {
    LZ77StaticIndex<> scalar(input.data(), input.size(), 1024, 3, width, true);
    LZ77StaticIndex<> vector(input.data(), input.size(), 1024, 3, width, true, true);
    LZ77StaticIndex<> coordinates(input.data(), input.size(), 1024, 3, width,
                                  true, true, false, true);
    for (size_t pos = 0; pos < input.size(); pos += 7) {
      std::vector<std::pair<size_t, uint32_t>> a, b;
      const auto cost = [](uint32_t d) { return float(d % 23); };
      scalar.FindMatches(pos, cost, [&](size_t n, uint32_t d) { a.emplace_back(n, d); });
      vector.FindMatches(pos, cost, [&](size_t n, uint32_t d) { b.emplace_back(n, d); });
      EXPECT_EQ(a, b);
      b.clear();
      coordinates.FindMatches(pos, cost, [&](size_t n, uint32_t d) { b.emplace_back(n, d); });
      EXPECT_EQ(a, b);
    }
  }
}

TEST(ANSTest, LZ77ParallelIndexPreservesCandidateCosts) {
  for (size_t count : {size_t{0}, size_t{7}, size_t{65539}, size_t{262147}}) {
    std::vector<Token> input;
    for (size_t i = 0; i < count; ++i)
      input.emplace_back(i % 3, i % 60000 < 44000 ? 0 : (i * 17) % 127);
    for (size_t width : {0u, 17u}) {
      for (bool coordinates : {false, true}) {
        test::ThreadPoolForTests workers(4);
        LZ77StaticIndex<> control(input.data(), count, 32768, 3, width,
                                   true, true, 0, coordinates, true);
        LZ77StaticIndex<> parallel(input.data(), count, 32768, 3, width,
                                    true, true, 0, coordinates, true, 0, true);
        ASSERT_TRUE(parallel.BuildParallel(input.data(), workers.get()));
        LZ77StaticIndex<32, true, 32, uint16_t> narrow(input.data(), count, 32768, 3, width,
                                    true, true, 0, coordinates, true, 0, true);
        ASSERT_TRUE(narrow.BuildParallel(input.data(), workers.get()));
        for (size_t pos = 0; pos < count; pos += 127) {
          std::vector<float> a(513, std::numeric_limits<float>::infinity()), b = a;
          const auto cost = [](uint32_t d) { return 1 + float(d % 31) / 8; };
          const auto collect = [&](auto& best) {
            return [&](size_t length, uint32_t distance) {
              const size_t actual_distance = width && distance < kNumSpecialDistances
                  ? SpecialDistance(distance, width)
                  : distance + 1 - (width ? kNumSpecialDistances : 0);
              ASSERT_GT(actual_distance, 0u);
              ASSERT_LE(actual_distance, pos);
              ASSERT_LT(actual_distance, 32768u);
              ASSERT_LE(length, count - pos);
              for (size_t i = 0; i < length; ++i)
                ASSERT_EQ(input[pos + i].value, input[pos - actual_distance + i].value);
              for (size_t n = 3; n <= std::min<size_t>(length, 512); ++n)
                best[n] = std::min(best[n], cost(distance));
            };
          };
          control.FindMatches(pos, cost, collect(a));
          parallel.FindMatches(pos, cost, collect(b));
          ASSERT_EQ(a, b) << "count=" << count << " pos=" << pos;
          std::fill(b.begin(), b.end(), std::numeric_limits<float>::infinity());
          narrow.FindMatches(pos, cost, collect(b));
          ASSERT_EQ(a, b) << "narrow count=" << count << " pos=" << pos;
        }
      }
    }
  }
  std::vector<Token> input(65536, Token(0, 0));
  LZ77StaticIndex<> index(input.data(), input.size(), 32768, 3, 0,
                          false, false, 0, false, false, 0, true);
  ThreadPool fail([](void*, void*, JxlParallelRunInit, JxlParallelRunFunction,
                       uint32_t, uint32_t) { return JXL_PARALLEL_RET_RUNNER_ERROR; }, nullptr);
  EXPECT_FALSE(index.BuildParallel(input.data(), &fail));
}

TEST(ANSTest, LZ77SharedFrontierCarriesVerifiedCosts) {
  std::vector<Token> input;
  for (size_t i = 0; i < 8192; ++i) {
    const uint32_t value = i % 211 < 170 ? i % 11 : (i * 71) % 1021;
    input.emplace_back(i % 5, value);
  }
  for (size_t width : {0u, 1u, 17u}) {
    LZ77StaticIndex<> index(input.data(), input.size(), 1024, 3, width, true);
    for (size_t pos = 0; pos <= input.size(); pos += 8) {
      for (size_t limit : {0u, 2u, 7u, 513u}) {
        LZ77MatchDominance carried;
        carried.Add(900, 123, 1.0f);  // The callee must reset reused scratch.
        std::vector<float> expected(limit + 1, std::numeric_limits<float>::infinity());
        const auto price = [](uint32_t d) { return float(d % 23); };
        index.FindMatches(pos, price, [&](size_t length, uint32_t d) {
          for (size_t n = 3; n <= length; ++n) expected[n] = std::min(expected[n], price(d));
        }, nullptr, limit);
        index.FindMatches(pos, price, [](size_t, uint32_t) {}, nullptr, limit, 0, &carried);
        ASSERT_FALSE(carried.Overflowed());
        auto actual = std::vector<float>(limit + 1, std::numeric_limits<float>::infinity());
        carried.ForEach([&](size_t length, uint32_t d) {
          const size_t distance = width && d < kNumSpecialDistances ? SpecialDistance(d, width)
              : d + 1 - (width ? kNumSpecialDistances : 0);
          ASSERT_GT(distance, 0u);
          ASSERT_LE(distance, pos);
          ASSERT_LE(length, limit);
          ASSERT_LE(length, input.size() - pos);
          for (size_t n = 0; n < length; ++n)
            ASSERT_EQ(input[pos + n].value, input[pos - distance + n].value);
          for (size_t n = 3; n <= length; ++n) actual[n] = std::min(actual[n], price(d));
        });
        EXPECT_EQ(actual, expected) << "position=" << pos << " limit=" << limit;
      }
    }
  }
}

TEST(ANSTest, LZ77FrontierInheritanceKeepsVerifiedSubstrings) {
  std::vector<Token> input;
  for (size_t i = 0; i < 8192; ++i) {
    input.emplace_back(i % 5, i % 211 < 170 ? i % 11 : (i * 71) % 1021);
  }
  for (size_t width : {0u, 1u, 17u}) {
    LZ77StaticIndex<> index(input.data(), input.size(), 1024, 3, width, true);
    LZ77FrontierCarry carry;
    for (size_t pos = 0; pos <= input.size(); pos += pos % 7 + 1) {
      const size_t limit = std::min<size_t>(513, input.size() - pos);
      const auto price = [](uint32_t d) { return float(d % 23); };
      std::vector<float> expected(limit + 1, std::numeric_limits<float>::infinity());
      index.FindMatches(pos, price, [&](size_t length, uint32_t d) {
        for (size_t n = 3; n <= length; ++n) expected[n] = std::min(expected[n], price(d));
      }, nullptr, limit);
      LZ77MatchDominance frontier;
      index.FindMatches(pos, price, [&](size_t length, uint32_t symbol) {
        const size_t distance = width && symbol < kNumSpecialDistances
            ? SpecialDistance(symbol, width) : symbol + 1 - (width ? kNumSpecialDistances : 0);
        ASSERT_GT(distance, 0u);
        ASSERT_LE(distance, pos);
        ASSERT_LT(distance, 1024u);
        ASSERT_LE(length, limit);
        for (size_t n = 0; n < length; ++n)
          ASSERT_EQ(input[pos + n].value, input[pos - distance + n].value);
      }, nullptr, limit, 0, &frontier, {}, &carry);
      ASSERT_FALSE(frontier.Overflowed());
      std::vector<float> actual(limit + 1, std::numeric_limits<float>::infinity());
      frontier.ForEach([&](size_t length, uint32_t d) {
        for (size_t n = 3; n <= length; ++n) actual[n] = std::min(actual[n], price(d));
      });
      for (size_t n = 3; n <= limit; ++n) EXPECT_LE(actual[n], expected[n]);
      carry.Remember(pos, frontier);
    }
  }
  LZ77MatchDominance frontier;
  frontier.Add(20, 1, 1.0f);
  LZ77FrontierCarry carry;
  carry.Remember(100, frontier);
  for (size_t pos : {0u, 99u, 118u, 120u, 130u}) {
    carry.Visit(pos, 3, 100, [&](size_t, uint32_t) { ADD_FAILURE(); });
  }
  carry.Visit(103, 3, 8, [&](size_t length, uint32_t symbol) {
    EXPECT_EQ(length, 8u);
    EXPECT_EQ(symbol, 1u);
  });
}

TEST(ANSTest, LZ77WarmSeedBoundaryPreservesTokensAndContexts) {
  for (size_t minimum : {3u, 5u}) {
    std::vector<Token> source;
    for (size_t i = 0; i < 520; ++i) source.emplace_back(i % 5, (i % 7) * 65537u);
    std::vector<Token> seed(source.begin(), source.begin() + 8);
    seed.emplace_back(static_cast<uint32_t>(source[8].context), 512 - minimum);
    seed.back().is_lz77_length = true;
    seed.emplace_back(5, 6);  // Linear distance 7, including overlapping copies.
    for (size_t prefix = 0; prefix <= source.size(); ++prefix) {
      std::vector<Token> trimmed;
      ASSERT_TRUE(TrimLZ77SeedPrefix(source, prefix, seed, minimum, &trimmed));
      std::vector<uint32_t> decoded;
      for (size_t i = 0; i < prefix; ++i) decoded.push_back(source[i].value);
      for (size_t i = 0; i < trimmed.size(); ++i) {
        const auto& token = trimmed[i];
        ASSERT_LT(decoded.size(), source.size());
        EXPECT_EQ(token.context, source[decoded.size()].context);
        if (!token.is_lz77_length) {
          decoded.push_back(token.value);
          continue;
        }
        ASSERT_LT(i + 1, trimmed.size());
        const auto& distance = trimmed[++i];
        EXPECT_EQ(distance.context, 5u);
        ASSERT_LT(distance.value, decoded.size());
        const size_t length = token.value + minimum;
        ASSERT_LE(length, source.size() - decoded.size());
        for (size_t j = 0; j < length; ++j)
          decoded.push_back(decoded[decoded.size() - distance.value - 1]);
      }
      ASSERT_EQ(decoded.size(), source.size());
      for (size_t i = 0; i < source.size(); ++i) EXPECT_EQ(decoded[i], source[i].value);
    }
    std::vector<Token> ignored;
    EXPECT_FALSE(TrimLZ77SeedPrefix(source, source.size() + 1, seed, minimum, &ignored));
  }
}

TEST(ANSTest, IndexedSeedGlobalDictionary) {
  for (int threads : {0, 4}) {
    test::ThreadPoolForTests pool(threads);
    for (size_t size : {0u, 1u, 3u, 257u, 131071u, 262144u}) {
      SCOPED_TRACE(testing::Message() << size << ":" << threads);
      std::vector<Token> source;
      for (size_t i = 0; i < size; ++i) {
        source.emplace_back(i % 3, i % 1024 < 700
            ? static_cast<uint32_t>((i % 97) * 65537) : 0);
      }
      for (size_t width : {0u, 17u}) {
        LZ77StaticIndex<> index(source.data(), size, 32768, 3, width, true);
        HistogramParams params;
        params.lossless_lz77_pool = pool.get();
        LZ77Params lz77;
        lz77.min_length = 3;
        lz77.nonserialized_distance_context = 3;
        std::vector<Token> result;
        float saving = 0, raw = 0;
        ASSERT_TRUE(LZ77IndexedSeed(source, index, params, lz77,
            [](const Token&) { return 16.0f; },
            [](const Token&, size_t, uint32_t) { return 24.0f; },
            [](uint32_t) { return 8.0f; }, &result, &saving, &raw));
        std::vector<uint32_t> decoded;
        for (size_t i = 0; i < result.size(); ++i) {
          if (!result[i].is_lz77_length) {
            decoded.push_back(result[i].value);
            continue;
          }
          const size_t length = result[i].value + lz77.min_length;
          ASSERT_LT(i + 1, result.size());
          const uint32_t symbol = result[++i].value;
          const size_t distance = width && symbol < kNumSpecialDistances
              ? SpecialDistance(symbol, width)
              : symbol + 1 - (width ? kNumSpecialDistances : 0);
          ASSERT_GT(distance, 0u);
          ASSERT_LE(distance, decoded.size());
          ASSERT_LE(decoded.size() + length, size);
          for (size_t j = 0; j < length; ++j)
            decoded.push_back(decoded[decoded.size() - distance]);
        }
        ASSERT_EQ(decoded.size(), size);
        for (size_t i = 0; i < size; ++i) ASSERT_EQ(decoded[i], source[i].value);
        EXPECT_GE(saving, 0);
        EXPECT_FLOAT_EQ(raw, 16.0f * size);
      }
    }
  }
}

TEST(ANSTest, LZ77IndexedParallelRoundtrip) {
  const char* enabled = std::getenv("JXL_LOSSLESS_LZ77_INDEXED");
  const char* cost = std::getenv("JXL_LOSSLESS_LZ77_MATCH_COST");
  if (!enabled || (std::strcmp(enabled, "1") && std::strcmp(enabled, "2")) ||
      !cost || std::strcmp(cost, "2")) {
    GTEST_SKIP();
  }
  std::vector<Token> source;
  for (size_t i = 0; i < 262144; ++i) {
    source.emplace_back((i / 7) % 3,
        i % 1024 < 700 ? static_cast<uint32_t>((i % 37) * 257u) : 0);
  }
  for (int threads : {0, 4}) {
    test::ThreadPoolForTests delegate(threads);
    struct State {
      ThreadPool* pool;
      size_t calls = 0, tasks = 0;
      bool fail = false;
    };
    State state{delegate.get()};
    ThreadPool pool(
        [](void* opaque, void* context, JxlParallelRunInit init,
           JxlParallelRunFunction run, uint32_t begin, uint32_t end) -> JxlParallelRetCode {
          auto* state = static_cast<State*>(opaque);
          ++state->calls;
          state->tasks += end - begin;
          if (state->fail) return JXL_PARALLEL_RET_RUNNER_ERROR;
          return state->pool->runner()(state->pool->runner_opaque(), context,
                                      init, run, begin, end);
        }, &state);
    for (size_t width : {0u, 1u, 17u}) {
      for (bool huffman : {false, true}) {
        HistogramParams params;
        params.lossless_lz77_cost_spans = true;
        params.lossless_lz77_pool = &pool;
        params.lossless_bucket_matcher = true;
        params.lz77_method = HistogramParams::LZ77Method::kOptc256;
        params.force_huffman = huffman;
        params.image_widths = {width};
        state.calls = 0;
        state.tasks = 0;
        RoundtripTestcase(3, ANS_MAX_ALPHABET_SIZE, source, params, true);
        EXPECT_GT(state.calls, 0u);
        EXPECT_GT(state.tasks, 1u);
        state.fail = true;
        std::vector<std::vector<Token>> streams{source};
        BitWriter writer(test::MemoryManager());
        EntropyEncodingData codes;
        EXPECT_FALSE(BuildAndEncodeHistograms(test::MemoryManager(), params, 3,
                                              streams, &codes, &writer,
                                              LayerType::Header, nullptr).ok());
        state.fail = false;
      }
    }
  }
}

TEST(ANSTest, LZ77SmallIndexWithoutPoolRoundtrip) {
  for (size_t size : {8191u, 8192u, 65535u, 65536u}) {
    for (uint32_t scale : {1u, 65537u}) {
      std::vector<Token> source;
      for (size_t i = 0; i < size; ++i) {
        source.emplace_back(
            i % 3,
            i % 997 < 700 ? static_cast<uint32_t>((i % 43) * 17) * scale : 0);
      }
      for (size_t width : {0u, 17u}) {
        for (bool huffman : {false, true}) {
          SCOPED_TRACE(testing::Message() << size << ":" << width << ":"
                                          << huffman << ":" << scale);
          HistogramParams params;
          params.lossless_lz77_cost_spans = true;
          params.lossless_bucket_matcher = true;
          params.lz77_method = HistogramParams::LZ77Method::kOptc256;
          params.force_huffman = huffman;
          params.image_widths = {width};
          ASSERT_EQ(params.lossless_lz77_pool, nullptr);
          RoundtripTestcase(3, ANS_MAX_ALPHABET_SIZE, source, params, true);
        }
      }
    }
  }
}

TEST(ANSTest, LZ77SeedSpans) {
  std::vector<Token> seed;
  seed.emplace_back(0, 99);
  seed.emplace_back(0, 97);
  seed.back().is_lz77_length = true;
  seed.emplace_back(1, 0);
  seed.emplace_back(0, 7);
  const auto spans = LZ77SeedSpans(seed, 3, 0);
  ASSERT_EQ(spans.size(), 1u);
  EXPECT_EQ(spans[0].begin, 1u);
  EXPECT_EQ(spans[0].end, 101u);
  EXPECT_EQ(spans[0].distance, 1u);
  EXPECT_TRUE(spans[0].Search(1));
  EXPECT_TRUE(spans[0].Search(8));
  EXPECT_FALSE(spans[0].Search(9));
  EXPECT_TRUE(spans[0].Search(33));
  EXPECT_TRUE(spans[0].Search(93));
  EXPECT_TRUE(spans[0].Search(100));
  EXPECT_TRUE(LZ77SeedSpans({}, 3, 0).empty());
  const auto cuts = LZ77ParseCuts(seed, 102, 4, 3);
  EXPECT_EQ(cuts, (std::vector<size_t>{0, 101, 102}));
}

TEST(ANSTest, LZ77SuffixIndexBoundsAndLongestMatch) {
  if (!LosslessSuffixArrayAvailable()) GTEST_SKIP();
  std::vector<Token> input;
  for (size_t i = 0; i < 1024; ++i)
    input.emplace_back(i % 3, i % 91 < 60 ? i % 7 : (i * 17) % 257);
  for (size_t window : {32u, 2048u}) {
    LZ77SuffixIndex index;
    ASSERT_TRUE(index.Build(input.data(), input.size(), window, 3, 17));
    LZ77MatchCache cache(window, 16);
    for (size_t pos = 0; pos < input.size(); ++pos) {
      size_t longest = 0;
      index.FindMatches(pos, [](uint32_t d) { return float(d % 19); },
          [&](size_t length, uint32_t symbol) {
        const size_t distance = symbol < kNumSpecialDistances
            ? SpecialDistance(symbol, 17) : symbol + 1 - kNumSpecialDistances;
        ASSERT_GT(distance, 0u);
        ASSERT_LT(distance, window);
        ASSERT_LE(distance, pos);
        ASSERT_LE(pos + length, input.size());
        for (size_t k = 0; k < length; ++k)
          ASSERT_EQ(input[pos + k].value, input[pos - distance + k].value);
        longest = std::max(longest, length);
      }, &cache);
      if (window < input.size()) continue;
      size_t exact = 0;
      for (size_t before = 0; before < pos; ++before) {
        size_t length = 0;
        while (pos + length < input.size() &&
               input[before + length].value == input[pos + length].value) ++length;
        if (length >= 3) exact = std::max(exact, length);
      }
      EXPECT_EQ(longest, exact);
    }
  }
  LZ77SuffixIndex unsupported;
  EXPECT_FALSE(unsupported.Build(nullptr, 0, 2048, 3, 0));
  EXPECT_FALSE(unsupported.Build(input.data(), input.size(), 2048, 0, 0));
  ASSERT_TRUE(unsupported.Build(input.data(), input.size(), 2048, 3, 1));
  ASSERT_TRUE(unsupported.Build(input.data(), input.size(), 2048, 3, 0));
  input[0].value = UINT32_MAX;
  EXPECT_FALSE(unsupported.Build(input.data(), input.size(), 2048, 3, 0));
}

TEST(ANSTest, LZ77ActiveSuffixRanks) {
  for (size_t size : {1u, 63u, 64u, 65u, 4096u, 4097u, 262145u}) {
    LZ77ActiveRanks ranks(size);
    std::set<size_t> reference;
    uint32_t random = 0x12345678;
    for (size_t i = 0; i < 20000; ++i) {
      random ^= random << 13;
      random ^= random >> 17;
      random ^= random << 5;
      const size_t rank = random % size;
      const bool insert = (random & 16) != 0;
      ranks.Set(rank, insert);
      if (insert) reference.insert(rank);
      else reference.erase(rank);
      for (size_t query : {size_t{0}, rank, size - 1}) {
        auto before = reference.lower_bound(query);
        auto after = reference.upper_bound(query);
        EXPECT_EQ(ranks.Before(query),
                  before == reference.begin() ? SIZE_MAX : *--before);
        EXPECT_EQ(ranks.After(query),
                  after == reference.end() ? SIZE_MAX : *after);
      }
    }
  }
}

TEST(ANSTest, LZ77SlidingSuffixFindsLongestWindowMatch) {
  if (!LosslessSuffixArrayAvailable()) GTEST_SKIP();
  for (size_t pattern : {0u, 1u, 2u}) {
    std::vector<Token> input;
    uint32_t random = 0x87654321;
    for (size_t i = 0; i < 2048; ++i) {
      random ^= random << 13;
      random ^= random >> 17;
      random ^= random << 5;
      const uint32_t value = pattern == 0 ? random % 7
          : pattern == 1 ? (i % 73 < 50 ? i % 5 : random % 17)
                         : (i % 127 < 117 ? 0 : random % 11);
      input.emplace_back(i % 3, value);
    }
    for (size_t window : {1u, 4u, 32u, 65u, 256u, 4096u}) {
      SCOPED_TRACE(testing::Message() << pattern << ":" << window);
      LZ77SuffixIndex index;
      ASSERT_TRUE(index.Build(input.data(), input.size(), window, 3, 0, true));
      for (size_t pos = 0; pos < input.size(); ++pos) {
        size_t longest = 0;
        index.FindMatches(pos, [](uint32_t) { return 1.0f; },
            [&](size_t length, uint32_t symbol) {
          const size_t distance = symbol + 1;
          ASSERT_LT(distance, window);
          ASSERT_LE(distance, pos);
          ASSERT_LE(pos + length, input.size());
          for (size_t k = 0; k < length; ++k)
            ASSERT_EQ(input[pos + k].value, input[pos - distance + k].value);
          longest = std::max(longest, length);
        });
        size_t exact = 0;
        for (size_t distance = 1; distance < window && distance <= pos; ++distance) {
          size_t length = 0;
          while (pos + length < input.size() &&
                 input[pos + length].value == input[pos - distance + length].value)
            ++length;
          if (length >= 3) exact = std::max(exact, length);
        }
        ASSERT_EQ(longest, exact) << "position=" << pos;
      }
    }
  }
}

TEST(ANSTest, LZ77IndexValidatesHints) {
  std::vector<Token> input;
  for (size_t i = 0; i < 1024; ++i) input.emplace_back(0, (i / 53 + i % 7) % 19);
  LZ77StaticIndex<> index(input.data(), input.size(), 256, 3, 17, true);
  for (size_t pos = 0; pos < input.size(); ++pos) {
    for (size_t hint : {0u, 1u, 7u, 17u, 256u, 1024u}) {
      index.FindMatches(pos, [](uint32_t code) { return float(code % 17); },
          [&](size_t length, uint32_t code) {
        const size_t distance = code < kNumSpecialDistances
            ? SpecialDistance(code, 17) : code + 1 - kNumSpecialDistances;
        ASSERT_GT(distance, 0u);
        ASSERT_LE(distance, pos);
        ASSERT_LT(distance, 256u);
        ASSERT_LE(length, std::min<size_t>(91, input.size() - pos));
        for (size_t i = 0; i < length; ++i)
          ASSERT_EQ(input[pos + i].value, input[pos - distance + i].value);
      }, nullptr, 91, hint);
    }
  }
}

TEST(ANSTest, LZ77AnchorOrderingPreservesCostFrontier) {
  std::vector<Token> input;
  for (size_t i = 0; i < 8192; ++i) {
    const size_t x = i % 73, y = i / 73;
    input.emplace_back(i % 3, x < 40 ? 0 : (x * 7 + y % 13) % 31);
  }
  LZ77StaticIndex<> before(input.data(), input.size(), 2048, 3, 73, true);
  LZ77StaticIndex<> after(input.data(), input.size(), 2048, 3, 73, true,
                           false, false, false, true);
  for (size_t pos = 0; pos < input.size(); pos += 7) {
    for (unsigned model : {0u, 1u, 2u}) {
      const auto cost = [&](uint32_t symbol) {
        return model == 0 ? 1.0f : model == 1 ? float(symbol % 17)
                                              : float((symbol * 71) % 29);
      };
      const size_t limit = std::min<size_t>(127, input.size() - pos);
      std::vector<float> expected(limit + 1, 1e20f), actual = expected;
      const auto collect = [&](const auto& index, std::vector<float>& out) {
        index.FindMatches(pos, cost, [&](size_t length, uint32_t symbol) {
          for (size_t n = 3; n <= length; ++n) out[n] = std::min(out[n], cost(symbol));
        }, nullptr, limit);
      };
      collect(before, expected);
      collect(after, actual);
      EXPECT_EQ(expected, actual) << "position=" << pos << " model=" << model;
    }
  }
}

TEST(ANSTest, LZ77AlignedZeroRuns) {
  std::vector<Token> input;
  const auto zeros = [&](size_t n) {
    for (size_t i = 0; i < n; ++i) input.emplace_back(i % 3, 0);
  };
  const auto tail = [&]() {
    for (size_t i = 0; i < 41; ++i) input.emplace_back(i % 5, i * 137 + 11);
  };
  zeros(400);
  tail();
  for (size_t i = 0; i < 100; ++i) {
    zeros(11 + i % 17);
    for (size_t j = 0; j < 5; ++j) input.emplace_back(j, 5000 + i * 19 + j);
  }
  const size_t target = input.size();
  zeros(200);
  tail();
  input.emplace_back(0, 98765);
  for (bool coordinates : {false, true}) {
    LZ77StaticIndex<> index(input.data(), input.size(), 8192, 3, 0, true,
                            false, 2, coordinates);
    size_t longest = 0;
    index.FindMatches(target, [](uint32_t) { return 1.0f; },
        [&](size_t length, uint32_t) { longest = std::max(longest, length); });
    ASSERT_GE(longest, 241u);
    LZ77MatchCache cache(8192, 128);
    for (size_t pos = 0; pos < input.size(); ++pos) {
      index.FindMatches(pos, [](uint32_t d) { return float(d % 17); },
          [&](size_t length, uint32_t symbol) {
        const size_t distance = symbol + 1;
        ASSERT_LE(distance, pos);
        ASSERT_LT(distance, 8192u);
        ASSERT_LE(length, std::min<size_t>(251, input.size() - pos));
        for (size_t i = 0; i < length; ++i)
          ASSERT_EQ(input[pos + i].value, input[pos - distance + i].value);
      }, &cache, 251);
    }
  }
}

TEST(ANSTest, LZ77DeepPrefixRecallAndBounds) {
  std::vector<Token> input;
  const auto block = [&](uint32_t id) {
    for (uint32_t i = 0; i < 192; ++i)
      input.emplace_back(i % 3, i < 8 ? i * 17 : 1000 + id * 193 + i);
  };
  block(0);
  for (uint32_t i = 1; i < 100; ++i) block(i);
  const size_t target = input.size();
  block(0);
  for (unsigned mode : {1u, 2u}) {
    for (bool coordinates : {false, true}) {
      LZ77StaticIndex<> index(input.data(), input.size(), 32768, 3, 0,
                              true, false, 0, coordinates, true, mode);
      size_t longest = 0;
      index.FindMatches(target, [](uint32_t) { return 1.0f; },
          [&](size_t length, uint32_t) { longest = std::max(longest, length); });
      EXPECT_EQ(longest, 192u);
      LZ77MatchCache cache(32768, 128);
      for (size_t pos = 0; pos < input.size(); pos += 7) {
        index.FindMatches(pos, [](uint32_t d) { return float(d % 19); },
            [&](size_t length, uint32_t symbol) {
          const size_t distance = symbol + 1;
          ASSERT_LE(distance, pos);
          ASSERT_LT(distance, 32768u);
          ASSERT_LE(length, std::min<size_t>(129, input.size() - pos));
          for (size_t i = 0; i < length; ++i)
            ASSERT_EQ(input[pos + i].value, input[pos - distance + i].value);
        }, &cache, 129);
      }
    }
  }
  for (size_t size : {0u, 1u, 2u, 3u, 7u, 15u, 16u, 63u, 64u, 65u}) {
    std::vector<Token> small(size, Token(0, UINT32_MAX));
    LZ77StaticIndex<> index(small.data(), size, 32, 3, 0,
                            true, false, 0, false, true, 2);
    for (size_t pos = 0; pos < size; ++pos) {
      index.FindMatches(pos, [](uint32_t d) { return float(d % 7); },
          [&](size_t length, uint32_t symbol) {
        EXPECT_LE(symbol + 1, pos);
        EXPECT_LT(symbol + 1, 32u);
        EXPECT_LE(length, size - pos);
      });
    }
  }
}

TEST(ANSTest, LZ77SuffixLcpRangeQueries) {
  for (size_t size : {0u, 1u, 2u, 31u, 32u, 33u, 65u, 513u, 1025u}) {
    for (unsigned pattern : {0u, 1u, 2u}) {
      std::vector<uint32_t> values(size), ranks(size);
      std::vector<int32_t> suffixes(size);
      for (size_t i = 0; i < size; ++i) {
        values[i] = pattern == 0 ? 0 : pattern == 1 ? i % 17
            : i % 91 < 60 ? 0 : uint32_t(i * 1664525u);
        suffixes[i] = static_cast<int32_t>(i);
      }
      std::sort(suffixes.begin(), suffixes.end(), [&](size_t a, size_t b) {
        return std::lexicographical_compare(values.begin() + a, values.end(),
                                             values.begin() + b, values.end());
      });
      for (size_t i = 0; i < size; ++i) ranks[suffixes[i]] = i;
      LZ77SuffixLcp lcp;
      lcp.Build(values, suffixes.data(), ranks);
      for (size_t q = 0; size && q < 4096; ++q) {
        const size_t a = q * 131 % size, b = (q * 71 + q / 7) % size;
        if (a == b) { EXPECT_EQ(lcp.Between(ranks[a], ranks[b]), UINT32_MAX); continue; }
        size_t length = 0;
        while (a + length < size && b + length < size && values[a + length] == values[b + length]) ++length;
        EXPECT_EQ(lcp.Between(ranks[a], ranks[b]), length);
      }
    }
  }
}

TEST(ANSTest, LZ77SuffixLcpPreservesMatchCosts) {
  if (!LosslessSuffixArrayAvailable()) GTEST_SKIP();
  std::vector<Token> input;
  for (size_t i = 0; i < 4096; ++i)
    input.emplace_back(i % 3, i % 97 < 70 ? (i % 43) * 17 : 0);
  for (bool sliding : {false, true}) {
    for (size_t window : {64u, 4096u}) {
      LZ77SuffixIndex before, after;
      ASSERT_TRUE(before.Build(input.data(), input.size(), window, 3, 0, sliding));
      ASSERT_TRUE(after.Build(input.data(), input.size(), window, 3, 0, sliding, true));
      for (size_t pos = 0; pos < input.size(); pos += 7) {
        const size_t limit = std::min<size_t>(129, input.size() - pos);
        std::vector<float> expected(limit + 1, 1e20f), actual = expected;
        const auto cost = [](uint32_t d) { return float(d % 19); };
        const auto collect = [&](const auto& index, auto& result) {
          index.FindMatches(pos, cost, [&](size_t length, uint32_t symbol) {
            for (size_t i = 3; i <= length; ++i) result[i] = std::min(result[i], cost(symbol));
          }, nullptr, limit);
        };
        collect(before, expected);
        collect(after, actual);
        EXPECT_EQ(expected, actual);
      }
    }
  }
}

TEST(ANSTest, LZ77MatchRangeFrontier) {
  const auto cost = [](uint32_t distance) { return float((distance * 7) % 19); };
  for (size_t count : {0, 1, 2, 7, 63}) {
    std::vector<LZ77Match> matches;
    for (size_t i = 0; i < count; ++i) {
      matches.push_back({static_cast<uint32_t>(3 + (i * 17) % 67),
                          static_cast<uint32_t>((i * 31) % 101)});
    }
    std::vector<LZ77MatchRange> ranges;
    BuildLZ77MatchRanges(&matches, 3, cost, &ranges);
    size_t end = 3;
    for (const auto& range : ranges) {
      EXPECT_EQ(range.begin, end);
      EXPECT_LT(range.begin, range.end);
      end = range.end;
      for (size_t len = range.begin; len < range.end; ++len) {
        float best_cost = std::numeric_limits<float>::infinity();
        uint32_t best_distance = 0;
        for (const auto& match : matches) {
          if (match.length >= len && (cost(match.distance) < best_cost ||
              (cost(match.distance) == best_cost && match.distance < best_distance))) {
            best_cost = cost(match.distance);
            best_distance = match.distance;
          }
        }
        EXPECT_EQ(range.distance_cost, best_cost);
        EXPECT_EQ(range.distance, best_distance);
      }
    }
    size_t longest = 2;
    for (const auto& match : matches) longest = std::max(longest, size_t{match.length});
    EXPECT_EQ(end, longest + 1);
  }
}

TEST(ANSTest, LZ77SpecialDistanceRoundtrip) {
  std::vector<Token> values;
  for (size_t i = 0; i < 12000; ++i) {
    values.emplace_back(i % 3, i % 97 < 70 ? 0 : (i % 17) * 65537);
  }
  for (size_t width : {1u, 2u, 7u, 16u, 128u, 1024u}) {
    for (bool huffman : {false, true}) {
      HistogramParams params;
      params.lz77_method = HistogramParams::LZ77Method::kOptc256;
      params.image_widths = {width};
      params.lossless_lz77_cost_spans = true;
      params.force_huffman = huffman;
      RoundtripTestcase(3, ANS_MAX_ALPHABET_SIZE, values, params, true);
    }
  }
}

TEST(ANSTest, FastLZ77DistanceRangesRoundtrip) {
  std::vector<Token> values;
  for (size_t i = 0; i < 12347; ++i) {
    const uint32_t value = i % 97 < 70 ? 0 : (i % 17) * 65537;
    values.emplace_back(i % 3, value);
  }
  using Method = HistogramParams::LZ77Method;
  for (auto method : {Method::kLZ77b1w3f, Method::kLZ77b3w3f,
                      Method::kLZ77b7w3f, Method::kLZ77b15w3f,
                      Method::kLZ77b31w3f, Method::kLZ77b1w3t,
                      Method::kLZ77b3w3t, Method::kLZ77b7w3t,
                      Method::kLZ77b15w3t, Method::kLZ77b31w3t}) {
    // Tiny widths create duplicate special distances. Width zero has none;
    // a large width leaves some distances ahead of the current position.
    for (size_t width : {0u, 1u, 2u, 7u, 128u, 4096u}) {
      for (bool huffman : {false, true}) {
        SCOPED_TRACE(static_cast<int>(method));
        SCOPED_TRACE(width);
        SCOPED_TRACE(huffman);
        HistogramParams params;
        params.lz77_method = method;
        params.lossless_lz77_cost_spans = true;
        params.image_widths = {width};
        params.force_huffman = huffman;
        RoundtripTestcase(3, ANS_MAX_ALPHABET_SIZE, values, params, true);
      }
    }
  }
}

TEST(ANSTest, LZ77ZeroChainRoundtrip) {
  // Equal zero runs with distinct nonzero tails, longer matching records,
  // and repetitions crossing the sliding-window boundary.
  std::vector<Token> values;
  for (size_t i = 0; i < kWindowSize + 8197; ++i) {
    const size_t column = i % 32;
    const uint32_t value = column < 8 ? 0 :
        column < 16 ? static_cast<uint32_t>((i / 32) % 13 + column * 17) :
                      static_cast<uint32_t>(column * 257);
    values.emplace_back(i % 3, value);
  }
  for (bool huffman : {false, true}) {
    HistogramParams params;
    params.lossless_lz77_cost_spans = true;
    params.lz77_method = HistogramParams::LZ77Method::kOptc256;
    params.force_huffman = huffman;
    params.image_widths = {32};
    RoundtripTestcase(3, ANS_MAX_ALPHABET_SIZE, values, params, true);
  }
}

TEST(ANSTest, LZ77RecentBuckets) {
  LZ77RecentBuckets<4> buckets;
  std::vector<uint32_t> seen;
  const auto collect = [&](uint32_t pos) { seen.push_back(pos); };
  buckets.ForEach(7, collect);
  EXPECT_TRUE(seen.empty());
  EXPECT_EQ(buckets.OldestIfFull(7), 0);
  EXPECT_EQ(buckets.EvictedByNextInsert(7), UINT32_MAX);
  for (uint32_t i = 10; i < 13; ++i) buckets.Insert(7, i);
  EXPECT_EQ(buckets.OldestIfFull(7), 0);
  EXPECT_EQ(buckets.EvictedByNextInsert(7), UINT32_MAX);
  buckets.Insert(7 + (1u << 15), 13);
  EXPECT_EQ(buckets.OldestIfFull(7), 10);
  EXPECT_EQ(buckets.EvictedByNextInsert(7), 10);
  for (uint32_t i = 14; i < 20; ++i) buckets.Insert(7, i);
  buckets.Insert(8, 100);
  buckets.ForEach(7, collect);
  EXPECT_EQ(seen, (std::vector<uint32_t>{19, 18, 17, 16}));
  EXPECT_EQ(buckets.OldestIfFull(7), 16);
  EXPECT_EQ(buckets.EvictedByNextInsert(7), 16);
  seen.clear();
  buckets.ForEach(8, collect);
  EXPECT_EQ(seen, (std::vector<uint32_t>{100}));
}

TEST(ANSTest, LZ77BucketHistory) {
  LZ77BucketHistory history(true);
  for (uint32_t count = 0; count <= 1025; ++count) {
    SCOPED_TRACE(count);
    if (count != 0) history.Insert(7 + ((count & 1) << 15), 1000 + count);
    std::vector<uint32_t> expected, seen;
    for (uint32_t stride : {4u, 16u, 64u}) {
      const uint32_t samples = count / stride;
      for (uint32_t i = 0; i < std::min(4u, samples); ++i) {
        const uint32_t pos = 1000 + (samples - i) * stride;
        if (std::find(expected.begin(), expected.end(), pos) == expected.end()) {
          expected.push_back(pos);
        }
      }
    }
    history.ForEach(7, [&](uint32_t pos) { seen.push_back(pos); });
    EXPECT_EQ(seen, expected);
    history.ForEach(8, [](uint32_t) { ADD_FAILURE() << "unrelated bucket"; });
  }
}

TEST(ANSTest, LZ77FingerprintMasks) {
  const auto mask = GetLZ77FingerprintFunction();
  std::array<uint8_t, 66> tags{};
  for (size_t i = 0; i < tags.size(); ++i) tags[i] = i % 5;
  for (size_t offset : {0u, 1u}) {
    for (size_t length = 0; length <= 64; ++length) {
      for (uint8_t tag : {uint8_t{0}, uint8_t{3}, uint8_t{255}}) {
        uint64_t expected = 0;
        for (size_t i = 0; i < length; ++i) {
          if (tags[offset + i] == tag) expected |= uint64_t{1} << i;
        }
        EXPECT_EQ(mask(tags.data() + offset, length, tag), expected);
      }
    }
  }
}

TEST(ANSTest, LZ77SpecialPrefixMasks) {
  struct RestoreTargets {
    ~RestoreTargets() { hwy::SetSupportedTargetsForTest(0); }
  } restore_targets;
  std::vector<uint8_t> tags(273);
  for (size_t i = 0; i < tags.size(); ++i) tags[i] = (i * 13 + i / 7) % 5;
  constexpr std::array<LZ77SpecialPrefixGroup, 5> groups{{
      {16, 0xffff}, {32, 0x5a5a}, {57, 0x8001}, {256, 0xffff}, {512, 0x001f}}};
  std::string targets;
  for (int64_t target : hwy::SupportedAndGeneratedTargets()) {
    hwy::SetSupportedTargetsForTest(target);
    SCOPED_TRACE(hwy::TargetName(target));
    targets += std::string(hwy::TargetName(target)) + " ";
    const auto filter = GetLZ77SpecialPrefixFunction();
    filter(nullptr, 0, 0, 0, nullptr, 0, nullptr);
    for (size_t pos = 0; pos <= 256; ++pos) {
      for (size_t safe : {0u, 1u, 15u, 16u, 31u, 48u, 255u, 512u}) {
        for (uint8_t tag : {uint8_t{0}, uint8_t{3}, uint8_t{255}}) {
          std::array<uint16_t, groups.size()> actual{};
          filter(tags.data(), pos, safe, tag, groups.data(), groups.size(),
                 actual.data());
          for (size_t g = 0; g < groups.size(); ++g) {
            uint16_t expected = 0;
            for (size_t bit = 0; bit < 16; ++bit) {
              const size_t distance = groups[g].max_distance - bit;
              if ((groups[g].valid & (uint32_t{1} << bit)) &&
                  distance > safe && distance <= pos &&
                  tags[pos - distance] == tag) {
                expected |= uint16_t{1} << bit;
              }
            }
            EXPECT_EQ(actual[g], expected)
                << "pos=" << pos << " safe=" << safe << " group=" << g;
          }
        }
      }
    }
  }
  RecordProperty("tested_targets", targets);
}

TEST(ANSTest, LZ77PrefixFilterRoundtrip) {
  for (size_t size : {0u, 1u, 2u, 3u, 17u, 4099u, 65539u}) {
    std::vector<Token> values;
    for (size_t i = 0; i < size; ++i) {
      const size_t phase = i % 1024;
      const uint32_t value = phase < 700 ? static_cast<uint32_t>((phase % 43) * 65537u)
                                        : (phase < 900 ? 0 : static_cast<uint32_t>(phase));
      values.emplace_back(i % 3, value);
    }
    for (size_t width : {0u, 1u, 2u, 7u, 16u, 257u, 1048592u}) {
      for (auto method : {HistogramParams::LZ77Method::kLZ77b3w3f,
                          HistogramParams::LZ77Method::kLZ77b7w3t}) {
        SCOPED_TRACE(testing::Message() << size << ":" << width << ":" << static_cast<int>(method));
        HistogramParams params;
        params.lossless_lz77_cost_spans = true;
        params.lz77_method = method;
        params.image_widths = {width};
        RoundtripTestcase(3, ANS_MAX_ALPHABET_SIZE, values, params, size >= 4099);
      }
    }
  }
}

TEST(ANSTest, LZ77FingerprintBucketOrder) {
  LZ77RecentBuckets<32> plain, filtered;
  filtered.EnableFingerprints(true);
  for (uint32_t i = 0; i < 257; ++i) {
    const uint32_t key = 17 + ((i % 5) << 24);
    plain.Insert(key, i);
    filtered.Insert(key, i);
    for (uint32_t wanted = 0; wanted < 6; ++wanted) {
      std::vector<uint32_t> expected, actual;
      plain.ForEach(17, [&](uint32_t pos) {
        if (pos % 5 == wanted) expected.push_back(pos);
      });
      filtered.ForEach(17 + (wanted << 24),
                        [&](uint32_t pos) { actual.push_back(pos); });
      EXPECT_EQ(actual, expected);
    }
  }
  EXPECT_GT(filtered.Scanned(), filtered.Retained());
  // Long keys may differ beyond the mandatory short prefix.
  LZ77RecentBuckets<8> long_keys;
  long_keys.EnableFingerprints(false);
  long_keys.Insert(0x01000009, 23, 0x78000001);
  std::vector<uint32_t> seen;
  long_keys.ForEach(0x92000009, [&](uint32_t pos) { seen.push_back(pos); },
                    0x78000001);
  EXPECT_EQ(seen, (std::vector<uint32_t>{23}));
  for (uint32_t i = 100; i < 132; ++i) filtered.Insert(0x09000011, i);
  seen.clear();
  filtered.ForEach(0x09000011, [&](uint32_t pos) { seen.push_back(pos); });
  EXPECT_EQ(seen.size(), 32u);
  filtered.Insert(0x02000011, 132);
  seen.clear();
  filtered.ForEach(0x09000011, [&](uint32_t pos) { seen.push_back(pos); });
  EXPECT_EQ(seen.size(), 31u);
  EXPECT_EQ(seen.front(), 131u);
  EXPECT_GT(filtered.UniformQueries(), 0u);
}

TEST(ANSTest, LZ77BucketHistoryRoundtrip) {
  std::vector<Token> values;
  // More than 32 records share a short prefix. Vary the middle of the longer
  // key and revisit evicted records; the final truncated record tests tails.
  for (size_t record = 0; record < 2048; ++record) {
    const uint32_t variant = (record * 37) % 257;
    for (size_t column = 0; column < 19; ++column) {
      const uint32_t value = column < 4 ? static_cast<uint32_t>(column + 1) :
          column == 7 ? 9u : (variant * 65537u) ^ (column * 31337u);
      values.emplace_back((record + column) % 3, value);
    }
  }
  values.resize(values.size() - 5);
  for (bool nominated : {false, true}) {
    for (size_t width : {0u, 1u, 19u, 16384u}) {
      for (bool huffman : {false, true}) {
        HistogramParams params;
        params.lz77_method = HistogramParams::LZ77Method::kOptc256;
        params.lossless_lz77_cost_spans = true;
        params.lossless_bucket_matcher = nominated;
        params.force_huffman = huffman;
        params.image_widths = {width};
        RoundtripTestcase(3, ANS_MAX_ALPHABET_SIZE, values, params, true);
      }
    }
  }
}

TEST(ANSTest, LZ77BlockMatchLength) {
  std::vector<uint32_t> a(137), b(137);
  for (size_t i = 0; i < a.size(); ++i) a[i] = i * 65537;
  for (size_t offset = 0; offset < 8; ++offset) {
    for (size_t size = 0; size <= 129; ++size) {
      b = a;
      EXPECT_EQ(LZ77MatchLength(a.data() + offset, b.data() + offset, size), size);
      for (size_t mismatch = 0; mismatch < size; ++mismatch) {
        b[offset + mismatch] ^= 0x80000000u;
        EXPECT_EQ(LZ77MatchLength(a.data() + offset, b.data() + offset, size), mismatch);
        b[offset + mismatch] = a[offset + mismatch];
      }
    }
  }
  std::fill(a.begin(), a.end(), 7);
  for (size_t distance = 1; distance < 16; ++distance) {
    EXPECT_EQ(LZ77MatchLength(a.data() + distance, a.data(), a.size() - distance),
              a.size() - distance);
  }
}

TEST(ANSTest, LZ77MatchCacheRouting) {
  EXPECT_TRUE(PreferLZ77MatchCaching(4096, 256));
  EXPECT_FALSE(PreferLZ77MatchCaching(4095, 1));
  EXPECT_FALSE(PreferLZ77MatchCaching(4096, 257));
  EXPECT_FALSE(PreferLZ77MatchCaching(4096, 0));
  EXPECT_TRUE(PreferLZ77MatchCaching(518400, 9));
  EXPECT_FALSE(PreferLZ77MatchCaching(518400, 500000));
}

TEST(ANSTest, LosslessLZ77ChainLimit) {
  const char* invalid[] = {nullptr, "", "0", "-8", "+8", "8x", " 8", "8 ",
                          "257", "999999999999999999999999"};
  for (const char* value : invalid) {
    EXPECT_EQ(LosslessLZ77ChainLimit(value, 256), 256u);
  }
  EXPECT_EQ(LosslessLZ77ChainLimit("8", 256), 8u);
  EXPECT_EQ(LosslessLZ77ChainLimit("256", 256), 256u);
  EXPECT_EQ(LosslessLZ77ChainLimit("8", 3), 3u);
  EXPECT_EQ(LosslessLZ77ChainLimit("1", 8), 1u);
}

TEST(ANSTest, LZ77MatchCachePrefixes) {
  for (size_t period : {1, 2, 7, 31}) {
    std::vector<uint32_t> values(3072);
    for (size_t i = 0; i < values.size(); ++i) {
      values[i] = (i % period) * 65537u;
      if (i % 157 == 0) values[i] ^= 0xffffffffu;
    }
    for (size_t window : {1, 2, 8, 32, 128}) {
      LZ77MatchCache cache(window);
      for (size_t pos = 0; pos < values.size(); ++pos) {
        cache.Advance(pos);
        for (size_t distance = 1; distance <= std::min(pos, window); ++distance) {
          const size_t limit = std::min(values.size() - pos, size_t{1} + pos % 217);
          const size_t prefix = cache.Prefix(distance, limit);
          size_t expected = 0;
          while (expected < limit && values[pos + expected] ==
                                     values[pos - distance + expected]) ++expected;
          ASSERT_LE(prefix, expected);
          const size_t actual = prefix + LZ77MatchLength(
              values.data() + pos + prefix, values.data() + pos - distance + prefix,
              limit - prefix);
          ASSERT_EQ(actual, expected);
          cache.Remember(distance, actual);
        }
      }
    }
  }
  LZ77MatchCache cache(8);
  cache.Advance(100);
  cache.Remember(3, 500);
  cache.Advance(200);
  EXPECT_EQ(cache.Prefix(3, 500), 400u);
  EXPECT_EQ(cache.Prefix(3, 2), 2u);
  EXPECT_EQ(cache.Prefix(8, 500), 0u);
  cache.Advance(600);
  EXPECT_EQ(cache.Prefix(3, 500), 0u);
  const size_t max = std::numeric_limits<uint32_t>::max();
  cache.Advance(max - 2);
  cache.Remember(3, 100);
  EXPECT_EQ(cache.Prefix(3, 100), 2u);
}

TEST(ANSTest, LZ77SparseMatchCacheCollisions) {
  LZ77MatchCache cache(1024, 8);
  cache.Advance(100);
  cache.Remember(3, 100);
  EXPECT_EQ(cache.Prefix(3, 100), 100u);
  EXPECT_EQ(cache.Prefix(11, 100), 0u);
  cache.Remember(11, 25);
  EXPECT_EQ(cache.Prefix(3, 100), 0u);
  cache.Advance(110);
  EXPECT_EQ(cache.Prefix(11, 100), 15u);
  cache.Remember(19, 0);
  EXPECT_EQ(cache.Prefix(11, 100), 0u);
  EXPECT_EQ(cache.Prefix(19, 100), 0u);
  cache.Remember(1024, 100);
  EXPECT_EQ(cache.Prefix(1024, 100), 0u);
}

TEST(ANSTest, LZ77NonzeroRunsRoundtrip) {
  for (size_t run : {size_t{4095}, size_t{8193}, kWindowSize + 1}) {
    std::vector<Token> values;
    for (uint32_t value : {1u, 65535u, 7u}) {
      for (size_t i = 0; i < run; ++i) values.emplace_back(i % 3, value);
      values.emplace_back(0, value + 1);
    }
    for (size_t width : {0, 1, 480}) {
      HistogramParams params;
      params.lz77_method = HistogramParams::LZ77Method::kOptc256;
      params.lossless_lz77_cost_spans = true;
      params.image_widths = {width};
      for (bool huffman : {false, true}) {
        params.force_huffman = huffman;
        RoundtripTestcase(3, ANS_MAX_ALPHABET_SIZE, values, params, true);
      }
    }
  }
}

TEST(ANSTest, LZ77RangeRouting) {
  EXPECT_TRUE(PreferLZ77RangeParsing(65536, 1024, 32768, 16384));
  EXPECT_FALSE(PreferLZ77RangeParsing(65535, 1024, 32768, 16384));
  EXPECT_FALSE(PreferLZ77RangeParsing(65536, 0, 32768, 16384));
  EXPECT_FALSE(PreferLZ77RangeParsing(65536, 1025, 32768, 16384));
  EXPECT_FALSE(PreferLZ77RangeParsing(65536, 512, 32767, 16384));
  EXPECT_FALSE(PreferLZ77RangeParsing(65536, 1024, 32768, 16383));
}

TEST(ANSTest, LZ77LiteralOnlyJumpsPreserveShortestPathCost) {
  constexpr size_t count = 4096;
  for (size_t queue_limit : {size_t{0}, size_t{4}, size_t{65536}}) {
    std::vector<float> reference(count + 1, std::numeric_limits<float>::infinity());
    reference[0] = 0;
    const auto literal = [](size_t pos) { return float(pos % 5); };
    const auto offers = [&](size_t pos, const auto& add) {
      if (pos % 64) return;
      for (size_t k = 0; k < 4; ++k) {
        const size_t begin = 3 + k * 11;
        const size_t end = std::min(count - pos + 1, 200 + k * 173);
        if (begin < end) add(begin, end, uint32_t(4 + k), float(6 + k));
      }
      if (pos + 15 <= count) add(3, 16, uint32_t{19}, 0.5f);
    };
    for (size_t pos = 0; pos < count; ++pos) {
      reference[pos + 1] = std::min(reference[pos + 1], reference[pos] + literal(pos));
      offers(pos, [&](size_t begin, size_t end, uint32_t, float bits) {
        for (size_t n = begin; n < end; ++n)
          reference[pos + n] = std::min(reference[pos + n], reference[pos] + bits);
      });
    }
    LZ77RangeParser parser(count, queue_limit);
    size_t jumped = 0;
    for (size_t pos = 0; pos < count;) {
      parser.Advance(pos);
      EXPECT_FLOAT_EQ(parser.At(pos).cost, reference[pos]);
      if (pos % 64) {
        const size_t next = parser.JumpLiteralOnly(pos, std::min(count, (pos / 64 + 1) * 64));
        if (next > pos) { jumped += next - pos; pos = next; continue; }
      }
      parser.Literal(pos, 0, parser.At(pos).cost + literal(pos));
      offers(pos, [&](size_t begin, size_t end, uint32_t distance, float bits) {
        parser.Match(pos, begin, end, distance, 0, parser.At(pos).cost + bits);
      });
      ++pos;
    }
    parser.Advance(count);
    EXPECT_FLOAT_EQ(parser.At(count).cost, reference[count]);
    float path_cost = 0;
    for (size_t pos = count; pos;) {
      const auto node = parser.At(pos);
      ASSERT_GT(node.length, 0u);
      ASSERT_LE(node.length, pos);
      const size_t origin = pos - node.length;
      if (node.distance == 0) {
        ASSERT_EQ(node.length, 1u);
        path_cost += literal(origin);
      } else {
        bool found = false;
        offers(origin, [&](size_t begin, size_t end, uint32_t distance, float bits) {
          if (distance + 1 == node.distance && node.length >= begin && node.length < end) {
            found = true;
            path_cost += bits;
          }
        });
        ASSERT_TRUE(found);
      }
      pos = origin;
    }
    EXPECT_FLOAT_EQ(path_cost, reference[count]);
    if (queue_limit == 65536) EXPECT_GT(jumped, count / 4);
    if (queue_limit == 0) EXPECT_EQ(jumped, 0u);
  }
}

TEST(ANSTest, LZ77RangeParserMatchesScalarCosts) {
  constexpr size_t kSize = 8192;
  for (bool dominance : {false, true}) {
  for (size_t mode = 0; mode < 16; ++mode) {
  for (size_t queue_limit : {0, 16, 32768}) {
    SCOPED_TRACE(testing::Message() << "mode=" << mode << " queue=" << queue_limit);
    LZ77RangeParser parser(kSize, queue_limit, mode < 4 && (mode & 1),
                           mode < 4 ? (mode & 2) : (mode & 1),
                           mode < 4 || mode >= 10 ? 0 : 4 + 2 * ((mode - 4) / 2),
                           mode >= 10 ? 1 + (mode - 10) / 2 : 0, dominance);
    std::vector<float> reference(kSize + 1, std::numeric_limits<float>::infinity());
    reference[0] = 0;
    for (size_t i = 0; i < kSize; ++i) {
      parser.Advance(i);
      ASSERT_EQ(parser.At(i).cost, reference[i]) << i;
      const float literal = reference[i] + 1 + (i % 7) * 0.125f;
      parser.Literal(i, 0, literal);
      reference[i + 1] = std::min(reference[i + 1], literal);
      for (size_t j = 0; j < 3; ++j) {
        const size_t begin = 3 + j * 11;
        const size_t end = std::min(kSize - i + 1,
            begin + 1 + (i * 17 + j * 3) % (i % 97 == 0 ? 1024 : 71));
        if (begin >= end) continue;
        const float cost = reference[i] + 1 + ((i * 11 + j) % 13) * 0.25f;
        parser.Match(i, begin, end, j + 7, 0, cost);
        for (size_t length = begin; length < end; ++length) {
          reference[i + length] = std::min(reference[i + length], cost);
        }
      }
    }
    parser.Advance(kSize);
    EXPECT_EQ(parser.At(kSize).cost, reference[kSize]);
    EXPECT_LE(parser.PeakQueued(), queue_limit);
    if (dominance) EXPECT_GT(parser.DominatedIntervals(), 0u);
    if (queue_limit == 0 && mode < 10) EXPECT_GT(parser.ScalarEndpoints(), 0u);
    for (size_t pos = kSize; pos != 0;) {
      ASSERT_GT(parser.At(pos).length, 0u);
      ASSERT_LE(parser.At(pos).length, pos);
      pos -= parser.At(pos).length;
    }
  }
  }
  }
}

TEST(ANSTest, LZ77CostSpansLosslessOnly) {
  CompressParams params;
  EXPECT_FALSE(HistogramParams::ForModular(params, {}, false)
                   .lossless_lz77_cost_spans);
  params.SetLossless();
  EXPECT_TRUE(HistogramParams::ForModular(params, {}, false)
                  .lossless_lz77_cost_spans);
  params.butteraugli_distance = 1;
  EXPECT_FALSE(HistogramParams::ForModular(params, {}, false)
                   .lossless_lz77_cost_spans);
  params.SetLossless();
  params.ec_distance = {0.5f};
  EXPECT_FALSE(HistogramParams::ForModular(params, {}, false)
                   .lossless_lz77_cost_spans);
}

TEST(ANSTest, DisposableEntropyDoesNotChangeFinalOrLossyPolicy) {
  CompressParams params;
  params.SetLossless();
  params.speed_tier = SpeedTier::kTortoise;
  const auto final = HistogramParams::ForModular(params, {}, false);
  EXPECT_EQ(final.uint_method, HistogramParams::HybridUintMethod::kBest);
  params.lossless_disposable_probe = true;
  const auto probe = HistogramParams::ForModular(params, {}, false);
  const char* mode = std::getenv("JXL_LOSSLESS_PROBE_ENTROPY");
  const bool enabled = mode && (!std::strcmp(mode, "1") || !std::strcmp(mode, "2"));
  EXPECT_EQ(probe.uint_method, enabled ? HistogramParams::HybridUintMethod::kFast
                                      : final.uint_method);
  EXPECT_EQ(probe.lz77_method, final.lz77_method);
  if (enabled && mode[0] == '2') {
    EXPECT_EQ(probe.clustering, HistogramParams::ClusteringType::kFast);
    EXPECT_EQ(probe.ans_histogram_strategy, HistogramParams::ANSHistogramStrategy::kApproximate);
  }
  params.butteraugli_distance = 1;
  const auto lossy_probe = HistogramParams::ForModular(params, {}, false);
  params.lossless_disposable_probe = false;
  const auto lossy_final = HistogramParams::ForModular(params, {}, false);
  EXPECT_EQ(lossy_probe.uint_method, lossy_final.uint_method);
  EXPECT_EQ(lossy_probe.clustering, lossy_final.clustering);
  EXPECT_EQ(lossy_probe.ans_histogram_strategy, lossy_final.ans_histogram_strategy);
}

TEST(ANSTest, LZ77ZeroRunWindowBoundary) {
  // A window-minus-one zero run must not be mistaken for a full window:
  // shifting it retains a nonzero token even if the new final token is zero.
  for (size_t run : {kWindowSize - 1, kWindowSize, kWindowSize + 1}) {
    for (bool careful : {false, true}) {
      for (bool huffman : {false, true}) {
        SCOPED_TRACE(testing::Message() << "run=" << run
                                       << " careful=" << careful
                                       << " huffman=" << huffman);
        std::vector<Token> values;
        const size_t prefix = kWindowSize / 8;
        values.reserve(prefix + 2 * kWindowSize + 32);
        for (size_t i = 0; i < prefix; ++i) {
          values.emplace_back(0, 1 + i % 251);
        }
        values.insert(values.end(), run, Token(0, 0));
        values.emplace_back(0, 510);
        values.insert(values.end(), kWindowSize + 16, Token(0, 0));
        values.emplace_back(0, 65535);
        HistogramParams params;
        params.lz77_method = HistogramParams::LZ77Method::kOptc1;
        params.lz77_careful_first_pass = careful;
        params.lossless_lz77_cost_spans = true;
        params.force_huffman = huffman;
        RoundtripTestcase(1, ANS_MAX_ALPHABET_SIZE, values, params, true);
        // Also exercises the opt-in bucket matcher when selected externally.
        params.lz77_method = HistogramParams::LZ77Method::kOptc256;
        RoundtripTestcase(1, ANS_MAX_ALPHABET_SIZE, values, params, true);
      }
    }
  }
}

#if defined(ADDRESS_SANITIZER) || defined(MEMORY_SANITIZER) || \
    defined(THREAD_SANITIZER)
constexpr size_t kReps = 3;
#else
constexpr size_t kReps = 10;
#endif

void RoundtripRandomStream(int alphabet_size, size_t reps = kReps,
                           size_t num = 1 << 18) {
  constexpr int kNumHistograms = 3;
  Rng rng(0);
  for (size_t i = 0; i < reps; i++) {
    std::vector<Token> symbols;
    for (size_t j = 0; j < num; j++) {
      int context = rng.UniformI(0, kNumHistograms);
      int value = rng.UniformU(0, alphabet_size);
      symbols.emplace_back(context, value);
    }
    RoundtripTestcase(kNumHistograms, alphabet_size, symbols);
  }
}

void RoundtripRandomUnbalancedStream(int alphabet_size) {
  constexpr int kNumHistograms = 3;
  constexpr int kPrecision = 1 << 10;
  Rng rng(0);
  for (size_t i = 0; i < kReps; i++) {
    std::vector<int> distributions[kNumHistograms] = {};
    for (auto& distr : distributions) {
      distr.resize(kPrecision);
      int symbol = 0;
      int remaining = 1;
      for (int k = 0; k < kPrecision; k++) {
        if (remaining == 0) {
          if (symbol < alphabet_size - 1) symbol++;
          // There is no meaning behind this distribution: it's anything that
          // will create a nonuniform distribution and won't have too few
          // symbols usually. Also we want different distributions we get to be
          // sufficiently dissimilar.
          remaining = rng.UniformU(0, kPrecision - k + 1);
        }
        distr[k] = symbol;
        remaining--;
      }
    }
    std::vector<Token> symbols;
    for (int j = 0; j < 1 << 18; j++) {
      int context = rng.UniformI(0, kNumHistograms);
      int value = rng.UniformU(0, kPrecision);
      symbols.emplace_back(context, value);
    }
    RoundtripTestcase(kNumHistograms + 1, alphabet_size, symbols);
  }
}

TEST(ANSTest, RandomStreamRoundtrip3Small) { RoundtripRandomStream(3, 1, 16); }

TEST(ANSTest, RandomStreamRoundtrip3) { RoundtripRandomStream(3); }

TEST(ANSTest, RandomStreamRoundtripBig) {
  RoundtripRandomStream(ANS_MAX_ALPHABET_SIZE);
}

TEST(ANSTest, RandomUnbalancedStreamRoundtrip3) {
  RoundtripRandomUnbalancedStream(3);
}

TEST(ANSTest, RandomUnbalancedStreamRoundtripBig) {
  RoundtripRandomUnbalancedStream(ANS_MAX_ALPHABET_SIZE);
}

TEST(ANSTest, UintConfigRoundtrip) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  for (size_t log_alpha_size = 5; log_alpha_size <= 8; log_alpha_size++) {
    std::vector<HybridUintConfig> uint_config;
    std::vector<HybridUintConfig> uint_config_dec;
    for (size_t i = 0; i < log_alpha_size; i++) {
      for (size_t j = 0; j <= i; j++) {
        for (size_t k = 0; k <= i - j; k++) {
          uint_config.emplace_back(i, j, k);
        }
      }
    }
    uint_config.emplace_back(log_alpha_size, 0, 0);
    uint_config_dec.resize(uint_config.size());
    BitWriter writer{memory_manager};
    ASSERT_TRUE(writer.WithMaxBits(
        10 * uint_config.size(), LayerType::Header, nullptr, [&] {
          EncodeUintConfigs(uint_config, &writer, log_alpha_size);
          return true;
        }));
    writer.ZeroPadToByte();
    BitReader br(writer.GetSpan());
    EXPECT_TRUE(DecodeUintConfigs(log_alpha_size, &uint_config_dec, &br));
    EXPECT_TRUE(br.Close());
    for (size_t i = 0; i < uint_config.size(); i++) {
      EXPECT_EQ(uint_config[i].split_token, uint_config_dec[i].split_token);
      EXPECT_EQ(uint_config[i].msb_in_token, uint_config_dec[i].msb_in_token);
      EXPECT_EQ(uint_config[i].lsb_in_token, uint_config_dec[i].lsb_in_token);
    }
  }
}

void TestCheckpointing(bool ans, bool lz77) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  std::vector<std::vector<Token>> input_values(1);
  for (size_t i = 0; i < 1024; i++) {
    input_values[0].emplace_back(0, i % 4);
  }
  // up to lz77 window size.
  for (size_t i = 0; i < (1 << 20) - 1022; i++) {
    input_values[0].emplace_back(0, (i % 5) + 4);
  }
  // Ensure that when the window wraps around, new values are different.
  input_values[0].emplace_back(0, 0);
  for (size_t i = 0; i < 1024; i++) {
    input_values[0].emplace_back(0, i % 4);
  }

  EntropyEncodingData codes;
  HistogramParams params;
  params.lz77_method = lz77 ? HistogramParams::LZ77Method::kLZ77b3w3f
                            : HistogramParams::LZ77Method::kNone;
  params.force_huffman = !ans;

  BitWriter writer{memory_manager};
  {
    auto input_values_copy = input_values;
    JXL_TEST_ASSIGN_OR_DIE(
        size_t cost,
        BuildAndEncodeHistograms(memory_manager, params, 1, input_values_copy,
                                 &codes, &writer, LayerType::Header, nullptr));
    (void)cost;
    ASSERT_TRUE(WriteTokens(input_values_copy[0], codes, 0, &writer,
                            LayerType::Header, nullptr));
    writer.ZeroPadToByte();
  }

  // We do not truncate the output. Reading past the end reads out zeroes
  // anyway.
  BitReader br(writer.GetSpan());
  Status status = true;
  {
    BitReaderScopedCloser bc(br, status);

    std::vector<uint8_t> dec_context_map;
    ANSCode decoded_codes;
    ASSERT_TRUE(DecodeHistograms(memory_manager, &br, 1, &decoded_codes,
                                 &dec_context_map));
    ASSERT_EQ(dec_context_map, codes.context_map);
    JXL_TEST_ASSIGN_OR_DIE(ANSSymbolReader reader,
                           ANSSymbolReader::Create(&decoded_codes, &br));

    auto checkpoint = jxl::make_unique<ANSSymbolReader::Checkpoint>();
    size_t br_pos = 0;
    constexpr size_t kInterval = ANSSymbolReader::kMaxCheckpointInterval - 2;
    for (size_t i = 0; i < input_values[0].size(); i++) {
      if (i % kInterval == 0 && i > 0) {
        reader.Restore(*checkpoint);
        ASSERT_TRUE(br.Close());
        br = BitReader(writer.GetSpan());
        br.SkipBits(br_pos);
        for (size_t j = i - kInterval; j < i; j++) {
          Token symbol = input_values[0][j];
          uint32_t read_symbol =
              reader.ReadHybridUint(symbol.context, &br, dec_context_map);
          ASSERT_EQ(read_symbol, symbol.value) << "j = " << j;
        }
      }
      if (i % kInterval == 0) {
        reader.Save(checkpoint.get());
        br_pos = br.TotalBitsConsumed();
      }
      Token symbol = input_values[0][i];
      uint32_t read_symbol =
          reader.ReadHybridUint(symbol.context, &br, dec_context_map);
      ASSERT_EQ(read_symbol, symbol.value) << "i = " << i;
    }
    ASSERT_TRUE(reader.CheckANSFinalState());
  }
  EXPECT_TRUE(status);
}

TEST(ANSTest, TestCheckpointingANS) {
  TestCheckpointing(/*ans=*/true, /*lz77=*/false);
}

TEST(ANSTest, TestCheckpointingPrefix) {
  TestCheckpointing(/*ans=*/false, /*lz77=*/false);
}

TEST(ANSTest, TestCheckpointingANSLZ77) {
  TestCheckpointing(/*ans=*/true, /*lz77=*/true);
}

TEST(ANSTest, TestCheckpointingPrefixLZ77) {
  TestCheckpointing(/*ans=*/false, /*lz77=*/true);
}

std::string AnsSimdTestDescription(
    const testing::TestParamInfo<HybridUintConfig>& info) {
  std::stringstream name;
  const HybridUintConfig& cfg = info.param;
  name << cfg.split_exponent << "_" << cfg.msb_in_token << "_" << cfg.lsb_in_token;
  return name.str();
}

std::vector<HybridUintConfig> GenerateAnsSimdTests() {
  return {HybridUintConfig{0, 0, 0},  HybridUintConfig{2, 0, 1},
          HybridUintConfig{3, 1, 0},  HybridUintConfig{3, 1, 2},
          HybridUintConfig{3, 2, 0},  HybridUintConfig{3, 2, 1},
          HybridUintConfig{4, 1, 0},  HybridUintConfig{4, 1, 2},
          HybridUintConfig{4, 1, 3},  HybridUintConfig{4, 2, 0},
          HybridUintConfig{4, 2, 1},  HybridUintConfig{4, 2, 2},
          HybridUintConfig{5, 1, 0},  HybridUintConfig{5, 1, 2},
          HybridUintConfig{5, 1, 4},  HybridUintConfig{5, 2, 0},
          HybridUintConfig{5, 2, 1},  HybridUintConfig{5, 2, 2},
          HybridUintConfig{5, 2, 3},  HybridUintConfig{6, 0, 0},
          HybridUintConfig{6, 2, 4},  HybridUintConfig{7, 0, 0},
          HybridUintConfig{8, 0, 0},  HybridUintConfig{9, 0, 0},
          HybridUintConfig{10, 0, 0}, HybridUintConfig{11, 0, 0},
          HybridUintConfig{12, 0, 0}};
}

class AnsSimdTest : public ::testing::TestWithParam<HybridUintConfig> {};

JXL_GTEST_INSTANTIATE_TEST_SUITE_P(AnsSimdTestAll,
                                   AnsSimdTest,
                                   testing::ValuesIn(GenerateAnsSimdTests()),
                                   AnsSimdTestDescription);

TEST_P(AnsSimdTest, EstimateTokenCost) {
  const HybridUintConfig& cfg = GetParam();
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  constexpr size_t kNumItems = 8 * 1024 * 1024;
  std::vector<uint32_t> in;
  std::vector<uint32_t> expected_out;
  constexpr size_t kTail = 1024;
  in.reserve(kNumItems + 311306 + kTail);
  uint32_t expected_eb = 0;
  for (size_t i = 0; i < kNumItems; ++i) {
    in.push_back(i);
  }
  for (size_t b = 23; b <= 31; ++b) {
    size_t base = 1 << b;
    for (size_t k = base - 16384; k <= base + 16384; ++k) {
      in.push_back(k);
    }
  }
  for (size_t n = 0; n <= 16384; ++n) {
    in.push_back(static_cast<uint32_t>(~0u) - n);
  }
  expected_out.reserve(in.size() + 1024);
  for (size_t i = 0; i < in.size(); ++i) {
    uint32_t v = in[i];
    uint32_t tok, nbits, bits;
    cfg.Encode(v, &tok, &nbits, &bits);
    expected_out.push_back(tok);
    expected_eb += nbits;
  }
  size_t data_size = in.size();
  in.resize(data_size + kTail);

  JXL_ASSIGN_OR_QUIT(
      AlignedMemory out,
      AlignedMemory::Create(memory_manager, data_size * sizeof(uint32_t)),
      "Failed to rallocate memory.");
  uint32_t eb = EstimateTokenCost(in.data(), data_size, cfg, out);
  ASSERT_EQ(eb, expected_eb);
  uint32_t* actual_out = out.address<uint32_t>();
  for (size_t i = 0; i < data_size; ++i) {
    ASSERT_EQ(actual_out[i], expected_out[i]) << i;
  }
}
}  // namespace
}  // namespace jxl
