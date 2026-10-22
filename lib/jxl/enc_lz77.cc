// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_lz77.h"
#include "lib/jxl/enc_lossless_profile.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "lib/jxl/ans_params.h"
#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/fast_math-inl.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_ans_params.h"
#include "lib/jxl/enc_lz77_cost.h"
#include "lib/jxl/enc_lz77_bucket.h"
#include "lib/jxl/enc_lossless_workers.h"
#include "lib/jxl/enc_lossless_probe_pool.h"
#include "lib/jxl/enc_lz77_parse.h"
#include "lib/jxl/enc_lz77_index.h"
#include "lib/jxl/enc_lz77_block.h"
#include "lib/jxl/enc_lz77_suffix.h"
#include "lib/jxl/enc_lz77_seed.h"

namespace jxl {

namespace {

class SymbolCostEstimator {
 public:
  SymbolCostEstimator(size_t num_contexts, bool force_huffman,
                      const std::vector<std::vector<Token>>& tokens,
                      const LZ77Params& lz77,
                      const HybridUintConfig& uint_config = HybridUintConfig())
      : num_contexts_(num_contexts) {
    LosslessProfileScope model(LosslessProfileCurrent(), "lz77.symbol_costs");
    model.Set("contexts", num_contexts);
    LosslessProfileScope histogram(model.Id(), "lz77.cost_histogram");
    std::vector<Histogram> builder(num_contexts);
    // Build histograms for estimating lz77 savings.
    for (const auto& stream : tokens) {
      for (const auto& token : stream) {
        uint32_t tok, nbits, bits;
        (token.is_lz77_length ? lz77.length_uint_config : uint_config)
            .Encode(token.value, &tok, &nbits, &bits);
        tok += token.is_lz77_length ? lz77.min_symbol : 0;
        JXL_DASSERT(token.context < num_contexts);
        builder[token.context].Add(tok);
      }
    }
    histogram.Finish();
    LosslessProfileScope tables(model.Id(), "lz77.cost_tables");
    max_alphabet_size_ = 0;
    for (size_t i = 0; i < num_contexts; i++) {
      max_alphabet_size_ =
          std::max(max_alphabet_size_, builder[i].counts.size());
    }
    // Initialize all unseen symbol costs to ANS_LOG_TAB_SIZE.
    bits_.assign(num_contexts * max_alphabet_size_, ANS_LOG_TAB_SIZE);
    // TODO(veluca): SIMD?
    add_symbol_cost_.resize(num_contexts);
    for (size_t i = 0; i < num_contexts; i++) {
      float inv_total = 1.0f / (builder[i].total_count + 1e-8f);
      float total_cost = 0;
      for (size_t j = 0; j < builder[i].counts.size(); j++) {
        size_t cnt = builder[i].counts[j];
        float cost = 0;
        if (cnt != 0 && cnt != builder[i].total_count) {
          cost = -FastLog2f(cnt * inv_total);
          if (force_huffman) cost = std::ceil(cost);
        } else if (cnt == 0) {
          cost = ANS_LOG_TAB_SIZE;  // Highest possible cost.
        }
        bits_[i * max_alphabet_size_ + j] = cost;
        total_cost += cost * builder[i].counts[j];
      }
      // Penalty for adding a lz77 symbol to this contest (only used for static
      // cost model). Higher penalty for contexts that have a very low
      // per-symbol entropy.
      add_symbol_cost_[i] = std::max(0.0f, 6.0f - total_cost * inv_total);
    }
    model.Set("alphabet", max_alphabet_size_);
  }
  float Bits(size_t ctx, size_t sym) const {
    if (ctx >= num_contexts_ || sym >= max_alphabet_size_) {
      return ANS_LOG_TAB_SIZE;
    }
    return bits_[ctx * max_alphabet_size_ + sym];
  }
  float LenCost(size_t ctx, size_t len, const LZ77Params& lz77) const {
    uint32_t nbits, bits, tok;
    lz77.length_uint_config.Encode(len, &tok, &nbits, &bits);
    tok += lz77.min_symbol;
    return nbits + Bits(ctx, tok);
  }
  float DistCost(size_t len, const LZ77Params& lz77) const {
    uint32_t nbits, bits, tok;
    HybridUintConfig().Encode(len, &tok, &nbits, &bits);
    if (lz77.nonserialized_distance_context >= num_contexts_) {
      return nbits + ANS_LOG_TAB_SIZE;
    }
    return nbits + Bits(lz77.nonserialized_distance_context, tok);
  }
  float AddSymbolCost(size_t idx) const { return add_symbol_cost_[idx]; }

 private:
  size_t num_contexts_;
  size_t max_alphabet_size_;
  std::vector<float> bits_;
  std::vector<float> add_symbol_cost_;
};

std::vector<std::vector<Token>> ApplyLZ77_RLE(
    const HistogramParams& params, size_t num_contexts,
    const std::vector<std::vector<Token>>& tokens, const LZ77Params& lz77) {
  std::vector<std::vector<Token>> tokens_lz77(tokens.size());
  // TODO(veluca): tune heuristics here.
  SymbolCostEstimator sce(num_contexts, params.force_huffman, tokens, lz77);
  float bit_decrease = 0;
  size_t total_symbols = 0;
  std::vector<float> sym_cost;
  HybridUintConfig uint_config;
  for (size_t stream = 0; stream < tokens.size(); stream++) {
    size_t distance_multiplier =
        params.image_widths.size() > stream ? params.image_widths[stream] : 0;
    const auto& in = tokens[stream];
    auto& out = tokens_lz77[stream];
    total_symbols += in.size();
    // Cumulative sum of bit costs.
    sym_cost.resize(in.size() + 1);
    for (size_t i = 0; i < in.size(); i++) {
      uint32_t tok, nbits, unused_bits;
      uint_config.Encode(in[i].value, &tok, &nbits, &unused_bits);
      sym_cost[i + 1] = sce.Bits(in[i].context, tok) + nbits + sym_cost[i];
    }
    out.reserve(in.size());
    for (size_t i = 0; i < in.size(); i++) {
      size_t num_to_copy = 0;
      size_t distance_symbol = 0;  // 1 for RLE.
      if (distance_multiplier != 0) {
        distance_symbol = 1;  // Special distance 1 if enabled.
        JXL_DASSERT(kSpecialDistances[1][0] == 1);
        JXL_DASSERT(kSpecialDistances[1][1] == 0);
      }
      if (i > 0) {
        for (; i + num_to_copy < in.size(); num_to_copy++) {
          if (in[i + num_to_copy].value != in[i - 1].value) {
            break;
          }
        }
      }
      if (num_to_copy == 0) {
        out.push_back(in[i]);
        continue;
      }
      float cost = sym_cost[i + num_to_copy] - sym_cost[i];
      // This subtraction might overflow, but that's OK.
      size_t lz77_len = num_to_copy - lz77.min_length;
      float lz77_cost = num_to_copy >= lz77.min_length
                            ? CeilLog2Nonzero(lz77_len + 1) + 1
                            : 0;
      if (num_to_copy < lz77.min_length || cost <= lz77_cost) {
        for (size_t j = 0; j < num_to_copy; j++) {
          out.push_back(in[i + j]);
        }
        i += num_to_copy - 1;
        continue;
      }
      // Output the LZ77 length
      out.emplace_back(in[i].context, static_cast<uint32_t>(lz77_len));
      out.back().is_lz77_length = true;
      i += num_to_copy - 1;
      bit_decrease += cost - lz77_cost;
      // Output the LZ77 copy distance.
      out.emplace_back(
          static_cast<uint32_t>(lz77.nonserialized_distance_context),
          static_cast<uint32_t>(distance_symbol));
    }
  }

  if (params.direct_audit || bit_decrease > total_symbols * 0.2 + 16) {
    return tokens_lz77;
  }
  return {};
}


// Computes a Murmur-style mix hash over HashSize elements.
template <int kHashSize=3>
uint32_t GetHash(size_t pos, const std::vector<uint32_t>& data_) {
  uint32_t h = 0;
  if (pos + kHashSize <= data_.size()) {
    for (size_t i = 0; i < kHashSize; i++) {
      h ^= data_[pos+i] + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;

  } else {
    return 0;
  }
  return h;
}

// Hash chain for LZ77 matching
template <uint32_t kMaxChainLength, bool kBlockCompare = false>
struct HashChain {
  size_t size_;
  std::vector<uint32_t> data_;

  unsigned hash_num_values_ = 32768;
  unsigned hash_mask_ = hash_num_values_ - 1;
  unsigned hash_shift_ = 5;

  std::vector<int> head;
  std::vector<uint32_t> chain;
  std::vector<int> val;

  // Speed up repetitions of zero
  std::vector<int> headz;
  std::vector<uint32_t> chainz;
  std::vector<uint32_t> zeros;
  uint32_t numzeros = 0;
  std::unique_ptr<LZ77MatchCache> match_cache_;

  size_t window_size_;
  size_t window_mask_;
  size_t min_length_;
  size_t max_length_;

  // Map of special distance codes.
  std::unordered_map<int, int> special_dist_table_;
  size_t num_special_distances_ = 0;

  uint32_t maxchainlength = kMaxChainLength;  // window_size_ to allow all

  HashChain(const Token* data, size_t size, size_t window_size,
            size_t min_length, size_t max_length, size_t distance_multiplier)
      : size_(size),
        window_size_(window_size),
        window_mask_(window_size - 1),
        min_length_(min_length),
        max_length_(max_length) {
    data_.resize(size);
    for (size_t i = 0; i < size; i++) {
      data_[i] = data[i].value;
    }

    head.resize(hash_num_values_, -1);
    val.resize(window_size_, -1);
    chain.resize(window_size_);
    for (uint32_t i = 0; i < window_size_; ++i) {
      chain[i] = i;  // same value as index indicates uninitialized
    }

    zeros.resize(window_size_);
    headz.resize(window_size_ + 1, -1);
    chainz.resize(window_size_);
    for (uint32_t i = 0; i < window_size_; ++i) {
      chainz[i] = i;
    }
    // Translate distance to special distance code.
    if (distance_multiplier) {
      // Count down, so if due to small distance multiplier multiple distances
      // map to the same code, the smallest code will be used in the end.
      for (int i = kNumSpecialDistances - 1; i >= 0; --i) {
        special_dist_table_[SpecialDistance(i, distance_multiplier)] = i;
      }
      num_special_distances_ = kNumSpecialDistances;
    }
  }


  void EnableMatchCache() {
    match_cache_ = std::make_unique<LZ77MatchCache>(window_size_);
  }

  uint32_t CountZeros(size_t pos, uint32_t prevzeros) const {
    size_t end = pos + window_size_;
    if (end > size_) end = size_;
    if (prevzeros > 0) {
      // Only a full previous window can extend by checking its new last value.
      // A run of window_size - 1 ended at a nonzero token still in this window.
      if (prevzeros >= window_size_ && data_[end - 1] == 0 &&
          end == pos + window_size_) {
        return prevzeros;
      } else {
        return prevzeros - 1;
      }
    }
    uint32_t num = 0;
    while (pos + num < end && data_[pos + num] == 0) num++;
    return num;
  }

  void Update(size_t pos) {
    if (match_cache_) match_cache_->Advance(pos);
    uint32_t hashval = GetHash(pos, data_) & hash_mask_;
    uint32_t wpos = pos & window_mask_;

    val[wpos] = static_cast<int>(hashval);
    if (head[hashval] != -1) chain[wpos] = head[hashval];
    head[hashval] = wpos;

    if (pos > 0 && data_[pos] != data_[pos - 1]) numzeros = 0;
    numzeros = CountZeros(pos, numzeros);

    zeros[wpos] = numzeros;
    if (headz[numzeros] != -1) chainz[wpos] = headz[numzeros];
    headz[numzeros] = wpos;
  }

  void Update(size_t pos, size_t len) {
    for (size_t i = 0; i < len; i++) {
      Update(pos + i);
    }
  }

  template <bool kKeepShortMatches = false, typename CB>
  void FindMatches(size_t pos, int max_dist, const CB& found_match) const {
    uint32_t wpos = pos & window_mask_;
    uint32_t hashval = val[wpos];
    uint32_t hashpos = chain[wpos];

    int prev_dist = 0;
    int end = std::min<int>(pos + max_length_, size_);
    uint32_t chainlength = 0;
    uint32_t best_len = 0;
    for (;;) {
      int dist = (hashpos <= wpos) ? (wpos - hashpos)
                                   : (wpos - hashpos + window_mask_ + 1);
      if (dist < prev_dist) break;
      if (dist > max_dist) break;
      prev_dist = dist;
      uint32_t len = 0;
      if (dist > 0) {
        int i = pos;
        int j = pos - dist;
        const size_t prefix = match_cache_ ? match_cache_->Prefix(dist, end - i) : 0;
        if (prefix != 0) {
          i += prefix;
          j += prefix;
        } else if (numzeros > 3) {
          int r = std::min<int>(numzeros - 1, zeros[hashpos]);
          if (i + r >= end) r = end - i - 1;
          i += r;
          j += r;
        }
        if constexpr (kBlockCompare) {
          i += LZ77MatchLength(data_.data() + i, data_.data() + j, end - i);
        } else {
          if (match_cache_) {
            i += LZ77MatchLength(data_.data() + i, data_.data() + j, end - i);
          } else {
            while (i < end && data_[i] == data_[j]) {
              i++;
              j++;
            }
          }
        }
        len = i - pos;
        if (match_cache_) match_cache_->Remember(dist, len);
        // This can trigger even if the new length is slightly smaller than the
        // best length, because it is possible for a slightly cheaper distance
        // symbol to occur.
        if (len >= min_length_ && (kKeepShortMatches || len + 2 >= best_len)) {
          auto it = special_dist_table_.find(dist);
          int dist_symbol = (it == special_dist_table_.end())
                                ? (num_special_distances_ + dist - 1)
                                : it->second;
          found_match(len, dist_symbol);
          if (len > best_len) best_len = len;
        }
      }

      chainlength++;
      if (chainlength >= maxchainlength) break;

      if (numzeros >= 3 && len > numzeros) {
        if (hashpos == chainz[hashpos]) break;
        hashpos = chainz[hashpos];
        if (zeros[hashpos] != numzeros) break;
      } else {
        if (hashpos == chain[hashpos]) break;
        hashpos = chain[hashpos];
        if (val[hashpos] != static_cast<int>(hashval)) {
          // outdated hash value
          break;
        }
      }
    }
  }
  void FindMatch(size_t pos, int max_dist, size_t* result_dist_symbol,
                 size_t* result_len) const {
    *result_dist_symbol = 0;
    *result_len = 1;
    FindMatches(pos, max_dist, [&](size_t len, size_t dist_symbol) {
      if (len > *result_len ||
          (len == *result_len && *result_dist_symbol > dist_symbol)) {
        *result_len = len;
        *result_dist_symbol = dist_symbol;
      }
    });
  }
};

float LenCost(size_t len) {
  uint32_t nbits, bits, tok;
  HybridUintConfig(1, 0, 0).Encode(len, &tok, &nbits, &bits);
  constexpr float kCostTable[] = {
      2.797667318563126,  3.213177690381199,  2.5706009246743737,
      2.408392498667534,  2.829649191872326,  3.3923087753324577,
      4.029267451554331,  4.415576699706408,  4.509357574741465,
      9.21481543803004,   10.020590190114898, 11.858671627804766,
      12.45853300490526,  11.713105831990857, 12.561996324849314,
      13.775477692278367, 13.174027068768641,
  };
  size_t table_size = sizeof kCostTable / sizeof *kCostTable;
  if (tok >= table_size) tok = table_size - 1;
  return kCostTable[tok] + nbits;
}

// TODO(veluca): this does not take into account usage or non-usage of distance
// multipliers.
float DistCost(size_t dist) {
  uint32_t nbits, bits, tok;
  HybridUintConfig(7, 0, 0).Encode(dist, &tok, &nbits, &bits);
  constexpr float kCostTable[] = {
      6.368282626312716,  5.680793277090298,  8.347404197105247,
      7.641619201599141,  6.914328374119438,  7.959808291537444,
      8.70023120759855,   8.71378518934703,   9.379132523982769,
      9.110472749092708,  9.159029569270908,  9.430936766731973,
      7.278284055315169,  7.8278514904267755, 10.026641158289236,
      9.976049229827066,  9.64351607048908,   9.563403863480442,
      10.171474111762747, 10.45950155077234,  9.994813912104219,
      10.322524683741156, 8.465808729388186,  8.756254166066853,
      10.160930174662234, 10.247329273413435, 10.04090403724809,
      10.129398517544082, 9.342311691539546,  9.07608009102374,
      10.104799540677513, 10.378079384990906, 10.165828974075072,
      10.337595322341553, 7.940557464567944,  10.575665823319431,
      11.023344321751955, 10.736144698831827, 11.118277044595054,
      7.468468230648442,  10.738305230932939, 10.906980780216568,
      10.163468216353817, 10.17805759656433,  11.167283670483565,
      11.147050200274544, 10.517921919244333, 10.651764778156886,
      10.17074446448919,  11.217636876224745, 11.261630721139484,
      11.403140815247259, 10.892472096873417, 11.1859607804481,
      8.017346947551262,  7.895143720278828,  11.036577113822025,
      11.170562110315794, 10.326988722591086, 10.40872184751056,
      11.213498225466386, 11.30580635516863,  10.672272515665442,
      10.768069466228063, 11.145257364153565, 11.64668307145549,
      10.593156194627339, 11.207499484844943, 10.767517766396908,
      10.826629811407042, 10.737764794499988, 10.6200448518045,
      10.191315385198092, 8.468384171390085,  11.731295299170432,
      11.824619886654398, 10.41518844301179,  10.16310536548649,
      10.539423685097576, 10.495136599328031, 10.469112847728267,
      11.72057686174922,  10.910326337834674, 11.378921834673758,
      11.847759036098536, 11.92071647623854,  10.810628276345282,
      11.008601085273893, 11.910326337834674, 11.949212023423133,
      11.298614839104337, 11.611603659010392, 10.472930394619985,
      11.835564720850282, 11.523267392285337, 12.01055816679611,
      8.413029688994023,  11.895784139536406, 11.984679534970505,
      11.220654278717394, 11.716311684833672, 10.61036646226114,
      10.89849965960364,  10.203762898863669, 10.997560826267238,
      11.484217379438984, 11.792836176993665, 12.24310468755171,
      11.464858097919262, 12.212747017409377, 11.425595666074955,
      11.572048533398757, 12.742093965163013, 11.381874288645637,
      12.191870445817015, 11.683156920035426, 11.152442115262197,
      11.90303691580457,  11.653292787169159, 11.938615382266098,
      16.970641701570223, 16.853602280380002, 17.26240782594733,
      16.644655390108507, 17.14310889757499,  16.910935455445955,
      17.505678976959697, 17.213498225466388, 2.4162310293553024,
      3.494587244462329,  3.5258600986408344, 3.4959806589517095,
      3.098390886949687,  3.343454654302911,  3.588847442290287,
      4.14614790111827,   5.152948641990529,  7.433696808092598,
      9.716311684833672,
  };
  size_t table_size = sizeof kCostTable / sizeof *kCostTable;
  if (tok >= table_size) tok = table_size - 1;
  return kCostTable[tok] + nbits;
}


// Alternative optimal-parser match finder. The parser and entropy model stay
// intact; recent short/long-prefix buckets and spatial anchors propose matches.
template <size_t kCapacity, bool kHistory = false, bool kFullPrefix = false,
          size_t kLongCapacity = 8>
class LZ77BucketMatcher {
 public:
  LZ77BucketMatcher(const Token* input, size_t size, size_t window,
                   size_t min_length, size_t max_length, size_t width)
      : data_(size), zeros_(window), window_(window),
        min_length_(min_length), max_length_(max_length) {
    for (size_t i = 0; i < size; ++i) data_[i] = input[i].value;
    if (width) {
      special_count_ = kNumSpecialDistances;
      for (int i = kNumSpecialDistances - 1; i >= 0; --i) {
        special_[SpecialDistance(i, width)] = i;
      }
      const char* dense = std::getenv("JXL_LOSSLESS_BUCKET_DISTANCES");
      if (dense && std::strcmp(dense, "1") == 0) {
        int largest = 0;
        for (const auto& entry : special_) largest = std::max(largest, entry.first);
        // Keep unusually wide channels on the fixed-size sparse map.
        if (largest > 0 && largest < 65536) {
          special_lookup_.assign(static_cast<size_t>(largest) + 1, -1);
          for (const auto& entry : special_) {
            if (entry.first > 0) special_lookup_[entry.first] = entry.second;
          }
        }
      }
      for (size_t i = 0; i < 24; ++i) {
        const int distance = SpecialDistance(i, width);
        if (distance > 0 && static_cast<size_t>(distance) < window) {
          anchors_.push_back(distance);
        }
      }
    }
  }
  void Update(size_t pos) {
    hash_ = GetHash(pos, data_);
    short_.Insert(hash_, static_cast<uint32_t>(pos));
    long_key_ = hash_;
    if (pos + 8 <= data_.size()) {
      long_key_ ^= data_[pos + 3] * 0x85ebca6bu;
      long_key_ ^= data_[pos + 7] * 0xc2b2ae35u;
      if constexpr (kFullPrefix) {
        long_key_ ^= data_[pos + 4] * 0x9e3779b9u;
        long_key_ ^= data_[pos + 5] * 0x27d4eb2du;
        long_key_ ^= data_[pos + 6] * 0x165667b1u;
      }
      long_key_ ^= long_key_ >> 16;
      if constexpr (kHistory) {
        const uint32_t evicted = long_.EvictedByNextInsert(long_key_);
        if (evicted != UINT32_MAX) history_.Insert(long_key_, evicted);
      }
      // The long bucket also yields short matches. Its fingerprint must use
      // only the mandatory three-token prefix, not the optional eight tokens.
      long_.Insert(long_key_, static_cast<uint32_t>(pos), hash_);
    }
    const size_t end = std::min(pos + window_, data_.size());
    if (zero_run_) {
      if (zero_run_ < window_ || end != pos + window_ || data_[end - 1] != 0) {
        --zero_run_;
      }
    } else {
      while (pos + zero_run_ < end && data_[pos + zero_run_] == 0) ++zero_run_;
    }
    zeros_[pos & (window_ - 1)] = zero_run_;
  }
  template <bool kKeepShortMatches = false, typename Visit>
  void FindMatches(size_t pos, int max_distance, const Visit& visit) const {
    size_t best = 0;
    const size_t limit = std::min(max_length_, data_.size() - pos);
    const auto candidate = [&](uint32_t previous) {
      if (previous >= pos || pos - previous > static_cast<size_t>(max_distance) ||
          pos - previous >= window_) return;
      const size_t required = kKeepShortMatches ? min_length_ :
          std::max(min_length_, best > 2 ? best - 2 : 0);
      if (required > limit || data_[pos] != data_[previous] ||
          data_[pos + required - 1] != data_[previous + required - 1]) return;
      size_t length = std::min<size_t>(
          limit, std::min(zero_run_, zeros_[previous & (window_ - 1)]));
      length += LZ77MatchLength(data_.data() + pos + length,
                              data_.data() + previous + length, limit - length);
      if (length < required) return;
      const size_t distance = pos - previous;
      uint32_t symbol = static_cast<uint32_t>(special_count_ + distance - 1);
      if (!special_lookup_.empty()) {
        if (distance < special_lookup_.size() && special_lookup_[distance] >= 0) {
          symbol = special_lookup_[distance];
        }
      } else {
        const auto found = special_.find(static_cast<int>(distance));
        if (found != special_.end()) symbol = found->second;
      }
      visit(length, symbol);
      best = std::max(best, length);
    };
    short_.ForEach(hash_, candidate);
    const uint32_t oldest = short_.OldestIfFull(hash_);
    if (pos + 8 <= data_.size()) {
      long_.ForEach(long_key_, [&](uint32_t previous) {
        if (previous < oldest) candidate(previous);
      }, hash_);
      if constexpr (kHistory) {
        history_.ForEach(long_key_, [&](uint32_t previous) {
          if (previous < oldest) candidate(previous);
        });
      }
    }
    for (uint32_t distance : anchors_) {
      if (distance <= pos && pos - distance < oldest) {
        candidate(static_cast<uint32_t>(pos - distance));
      }
    }
  }

  template <typename Cost, typename Visit>
  void FindCostMatches(size_t pos, int max_distance, const Cost& cost,
                       const Visit& visit) const {
    LZ77MatchDominance frontier;
    const size_t limit = std::min(max_length_, data_.size() - pos);
    if (limit < min_length_) return;
    const auto candidate = [&](uint32_t previous) {
      if (previous >= pos || pos - previous > static_cast<size_t>(max_distance) ||
          pos - previous >= window_ || data_[pos] != data_[previous] ||
          data_[pos + min_length_ - 1] != data_[previous + min_length_ - 1]) return;
      const size_t distance = pos - previous;
      uint32_t symbol = static_cast<uint32_t>(special_count_ + distance - 1);
      if (!special_lookup_.empty()) {
        if (distance < special_lookup_.size() && special_lookup_[distance] >= 0) {
          symbol = special_lookup_[distance];
        }
      } else {
        const auto found = special_.find(static_cast<int>(distance));
        if (found != special_.end()) symbol = found->second;
      }
      const float bits = cost(symbol);
      const size_t required = frontier.RequiredLength(bits, symbol, min_length_);
      if (required > limit ||
          data_[pos + required - 1] != data_[previous + required - 1]) return;
      size_t length = std::min<size_t>(
          limit, std::min(zero_run_, zeros_[previous & (window_ - 1)]));
      length += LZ77MatchLength(data_.data() + pos + length,
                              data_.data() + previous + length, limit - length);
      if (length < required) return;
      visit(length, symbol);
      frontier.Add(static_cast<uint32_t>(length), symbol, bits);
    };
    short_.ForEach(hash_, candidate);
    const uint32_t oldest = short_.OldestIfFull(hash_);
    if (pos + 8 <= data_.size()) {
      long_.ForEach(long_key_, [&](uint32_t previous) {
        if (previous < oldest) candidate(previous);
      }, hash_);
      if constexpr (kHistory) {
        history_.ForEach(long_key_, [&](uint32_t previous) {
          if (previous < oldest) candidate(previous);
        });
      }
    }
    for (uint32_t distance : anchors_) {
      if (distance <= pos && pos - distance < oldest) {
        candidate(static_cast<uint32_t>(pos - distance));
      }
    }
  }

  bool Retains(size_t pos, uint32_t previous) const {
    bool retained = false;
    short_.ForEach(hash_, [&](uint32_t p) { retained |= p == previous; });
    const uint32_t oldest = short_.OldestIfFull(hash_);
    if (previous < oldest) {
      if (pos + 8 <= data_.size()) {
        long_.ForEach(long_key_, [&](uint32_t p) { retained |= p == previous; },
                       hash_);
        if constexpr (kHistory) {
          history_.ForEach(long_key_, [&](uint32_t p) { retained |= p == previous; });
        }
      }
      for (uint32_t distance : anchors_) {
        retained |= distance <= pos && pos - distance == previous;
      }
    }
    return retained;
  }
  bool DenseDistances() const { return !special_lookup_.empty(); }
  void EnableFingerprints(bool measure) {
    if (min_length_ < 3) return;
    short_.EnableFingerprints(measure);
    long_.EnableFingerprints(measure);
  }
  uint64_t FingerprintScanned() const { return short_.Scanned() + long_.Scanned(); }
  uint64_t FingerprintRetained() const { return short_.Retained() + long_.Retained(); }
  uint64_t FingerprintSIMDQueries() const {
    return short_.SIMDQueries() + long_.SIMDQueries();
  }
  uint64_t FingerprintUniformQueries() const {
    return short_.UniformQueries() + long_.UniformQueries();
  }

 private:
  std::vector<uint32_t> data_, zeros_;
  size_t window_, min_length_, max_length_, special_count_ = 0;
  uint32_t hash_ = 0, long_key_ = 0, zero_run_ = 0;
  LZ77RecentBuckets<kCapacity> short_;
  LZ77RecentBuckets<kLongCapacity> long_;
  LZ77BucketHistory history_{kHistory};
  std::unordered_map<int, uint32_t> special_;
  std::vector<int16_t> special_lookup_;
  std::vector<uint32_t> anchors_;
};

// Fast hash map for LZ77 compression using a fixed-size ring buffer per bucket.
// Overwrites the oldest candidates when a bucket reaches max capacity.
// For optimal performance kBucketSize should be a power of 2 minus 1
template <uint16_t kBucketSize, int kHashSize,
          size_t kSpecialDistancesToCheck = (kBucketSize <= 3 ? 24 : 120)>
class LZ77HashMap {
public:
    struct MatchStats {
        uint64_t queries = 0, special_candidates = 0, bucket_candidates = 0;
        uint64_t special_extends = 0, bucket_extends = 0;
        uint64_t special_prefix_tokens = 0, bucket_prefix_tokens = 0;
        uint64_t special_short_prefix = 0, bucket_short_prefix = 0;
    };
    const MatchStats& Stats() const { return match_stats_; }
    size_t PrefixBytes() const { return prefix_tags_.size(); }
    // Initializes the hash map mask, table size, and distance lookup table.
    explicit LZ77HashMap(const std::vector<uint32_t>& data,
                        size_t distance_multiplier, bool range_search,
                        uint32_t hash_bits = 14, unsigned fast_ranges = 0,
                        unsigned prefix_filter = 0)
        : data_(data),
          range_search_(range_search),
          fast_ranges_(fast_ranges),
          hash_mask_((uint32_t{1} << hash_bits) - 1),
          hash_table_(static_cast<size_t>(hash_mask_) + 1) {
        num_special_distances_ = 0;
        if (distance_multiplier) {
            num_special_distances_ = kNumSpecialDistances;
            special_dist_table_.assign(distance_multiplier * 8 + 8, -1);
            for (int i = static_cast<int>(kNumSpecialDistances) - 1; i >= 0; --i) {
                int dist = SpecialDistance(i, distance_multiplier);
                if (dist >= 0 && static_cast<size_t>(dist) < special_dist_table_.size()) {
                    special_dist_table_[dist] = i;
                }
            }
            if (kSpecialDistancesToCheck > 0) {
                size_t n = std::min(kSpecialDistancesToCheck, kNumSpecialDistances);
                special_dists_.reserve(n);
                for (size_t i = 0; i < n; ++i) {
                    int dist = SpecialDistance(i, distance_multiplier);
                    if (dist > 0 && static_cast<size_t>(dist) <= kWindowSize) {
                        special_dists_.push_back({static_cast<uint32_t>(dist), static_cast<uint8_t>(i)});
                    }
                }
                if (range_search_) {
                    std::sort(special_dists_.begin(), special_dists_.end(),
                              [](const auto& a, const auto& b) { return a.dist < b.dist; });
                }
                if (prefix_filter && !special_dists_.empty()) {
                    if (range_search_) pending_prefix_ = prefix_filter;
                    else BuildPrefixFilter(prefix_filter);
                }
            }
        }
    }

    // Searches for the longest matching sequence in the bucket for 'pos'.
    // Updates length and dist_symbol with the best match found, then inserts 'pos'.
    template <bool kMeasure = false>
    void FindMatch(size_t pos, size_t& len, size_t& dist_symbol, size_t min_len) {
        if constexpr (kMeasure) ++match_stats_.queries;
        // Long runs may need very few lookups. Build the O(n) cache only
        // after enough real candidate work, with at least half the input left.
        if (pending_prefix_ && prefix_candidates_ >= std::max<size_t>(4096, data_.size() / 2)) {
            const unsigned mode = pending_prefix_;
            pending_prefix_ = 0;
            if (pos < data_.size() / 2) BuildPrefixFilter(mode);
        }
        const uint32_t h = GetHash<kHashSize>(pos, data_) & hash_mask_;
        const bool filter = !prefix_tags_.empty() && min_len >= kHashSize;
        const uint8_t tag = filter ? prefix_tags_[pos] : 0;
        len = 0;
        dist_symbol = 0;

        // If the bucket is full, entries older than oldest_pos were evicted.
        // Candidates with dist <= max_safe_dist are guaranteed to still be present in
        // the bucket (if they match the 3-token hash), so we only need to test special
        // distances where dist > max_safe_dist.
        // If the bucket is not full, no entry has ever been evicted, so we can skip entirely.
        if (hash_table_[h].size == kBucketSize) {
            const uint32_t oldest_pos = hash_table_[h].data[hash_table_[h].idx];
            const size_t max_safe_dist = pos - oldest_pos;
            auto first = special_dists_.begin();
            auto last = special_dists_.end();
            if (range_search_) {
                // All smaller matching distances are still present in the
                // hash bucket. Skip that whole range, not one entry at a time.
                if (first == last || special_dists_.back().dist <= max_safe_dist) {
                    first = last;
                } else if (!filter || !prefix_masks_ || kMeasure) {
                    const auto less = [](size_t value, const auto& entry) {
                        return value < entry.dist;
                    };
                    first = std::upper_bound(first, last, max_safe_dist, less);
                    if (special_dists_.back().dist > pos) {
                        last = std::upper_bound(first, last, pos, less);
                    }
                }
            }
            if (pending_prefix_) prefix_candidates_ += last - first;
            const auto try_special = [&](const PrecomputedSpecialDist& sd) {
                const size_t req_len =
                    len == 0 ? min_len : (sd.symbol < dist_symbol ? len : len + 1);
                if (pos + req_len > data_.size()) return;
                const size_t candidate = pos - sd.dist;
                if (data_[candidate + req_len - 1] != data_[pos + req_len - 1] ||
                    data_[candidate] != data_[pos]) {
                    return;
                }
                size_t cur_length = MatchLength(candidate, pos);
                if constexpr (kMeasure) {
                    ++match_stats_.special_extends;
                    match_stats_.special_prefix_tokens += cur_length;
                    match_stats_.special_short_prefix += cur_length < kHashSize;
                }
                if (cur_length < req_len) {
                    return;
                }
                len = cur_length;
                dist_symbol = sd.symbol;
            };
            if (filter && prefix_masks_ && first != last) {
                std::array<uint16_t, kSpecialDistancesToCheck> masks;
                prefix_masks_(prefix_tags_.data(), pos, max_safe_dist, tag,
                              prefix_groups_.data(), prefix_groups_.size(), masks.data());
                if constexpr (kMeasure) match_stats_.special_candidates += last - first;
                for (size_t g = 0; g < prefix_groups_.size(); ++g) {
                    uint32_t mask = masks[g];
                    while (mask) {
                        // Highest bit is the nearest distance, preserving the
                        // ascending-distance traversal used by range search.
                        const size_t bit = FloorLog2Nonzero(mask);
                        const uint32_t distance = prefix_groups_[g].max_distance - bit;
                        try_special({distance, static_cast<uint8_t>(special_dist_table_[distance])});
                        mask ^= uint32_t{1} << bit;
                    }
                }
            } else {
                for (auto it = first; it != last; ++it) {
                    const auto& sd = *it;
                    if (sd.dist <= max_safe_dist || sd.dist > pos) continue;
                    if constexpr (kMeasure) ++match_stats_.special_candidates;
                    // Equal prefixes necessarily have equal tags. Collisions
                    // still go through the exact token comparison.
                    if (filter && prefix_tags_[pos - sd.dist] != tag) continue;
                    try_special(sd);
                }
            }
        }

        // Scan all existing entries in the bucket
        for (uint16_t i = 0; i < hash_table_[h].size; i++) {
            const uint32_t candidate = hash_table_[h].data[i];
            size_t dist = pos - candidate;
            if (dist > kWindowSize) {
                continue;
            }
            if constexpr (kMeasure) ++match_stats_.bucket_candidates;

            // Map distance to a symbol encoding (prefers special distance codes if available)
            size_t cur_dist_symbol = (num_special_distances_ + dist - 1);
            if (dist < special_dist_table_.size()) {
                int lookup = special_dist_table_[dist];
                if (lookup >= 0) {
                    cur_dist_symbol = lookup;
                }
            }

            // Reject if candidate cannot beat current best (len, dist_symbol)
            const size_t req_len =
                len == 0 ? min_len : (cur_dist_symbol < dist_symbol ? len : len + 1);
            if (pos + req_len > data_.size() ||
                data_[candidate + req_len - 1] != data_[pos + req_len - 1]) {
                continue;
            }

            size_t cur_length = MatchLength(candidate, pos);
            if constexpr (kMeasure) {
                ++match_stats_.bucket_extends;
                match_stats_.bucket_prefix_tokens += cur_length;
                match_stats_.bucket_short_prefix += cur_length < kHashSize;
            }
            if (cur_length < req_len) {
                continue;
            }

            len = cur_length;
            dist_symbol = cur_dist_symbol;
        }

        // Insert 'pos' into the bucket ring buffer
        hash_table_[h].size = std::min(static_cast<uint16_t>(hash_table_[h].size + 1), kBucketSize);
        hash_table_[h].data[hash_table_[h].idx] = static_cast<uint32_t>(pos);
        hash_table_[h].idx = hash_table_[h].idx + 1 - static_cast<uint16_t>(hash_table_[h].idx + 1 >= kBucketSize) * kBucketSize;
    }

    // Inserts 'pos' into its bucket without performing a match search.
    void Update(size_t pos) {
        const uint32_t h = GetHash<kHashSize>(pos, data_) & hash_mask_;
        hash_table_[h].size = std::min(static_cast<uint16_t>(hash_table_[h].size + 1), kBucketSize);
        hash_table_[h].data[hash_table_[h].idx] = static_cast<uint32_t>(pos);
        hash_table_[h].idx = hash_table_[h].idx + 1 - static_cast<uint16_t>(hash_table_[h].idx + 1 >= kBucketSize) * kBucketSize;
    }

    void UpdateRange(size_t begin, size_t end) {
        if (!(fast_ranges_ & 1)) {
            for (size_t pos = begin; pos < end; ++pos) Update(pos);
            return;
        }
        while (begin < end) {
            size_t run_end = begin + 1;
            const size_t limit = end + std::min<size_t>(data_.size() - end,
                                                        kHashSize - 1);
            while (run_end < limit && data_[run_end] == data_[begin]) ++run_end;
            const size_t count = run_end - begin >= kHashSize
                ? run_end - begin - kHashSize + 1 : 0;
            if (count < kBucketSize) {
                Update(begin++);
                continue;
            }
            const uint32_t hash = GetHash<kHashSize>(begin, data_) & hash_mask_;
            auto& bucket = hash_table_[hash];
            LZ77InsertRepeatedBucket(&bucket.data, &bucket.idx, &bucket.size,
                                      begin, begin + count);
            begin += count;
        }
    }

private:
    void BuildPrefixFilter(unsigned mode) {
        LosslessProfileScope build(LosslessProfileCurrent(), "lz77.prefix_cache");
        build.Set("tokens", data_.size());
        prefix_tags_.resize(data_.size() + 16);
        for (size_t i = 0; i + kHashSize <= data_.size(); ++i) {
            prefix_tags_[i] = GetHash<kHashSize>(i, data_) >> 24;
        }
        if (mode >= 2 && range_search_) {
            prefix_masks_ = GetLZ77SpecialPrefixFunction();
            for (const auto& sd : special_dists_) {
                if (prefix_groups_.empty() || sd.dist > prefix_groups_.back().max_distance) {
                    prefix_groups_.push_back({sd.dist + 15, 0});
                }
                auto& group = prefix_groups_.back();
                group.valid |= uint16_t{1} << (group.max_distance - sd.dist);
            }
        }
    }

    // Fixed-capacity bucket stored as a ring buffer.
    struct HashBucket {
        std::array<uint32_t, kBucketSize> data;
        uint16_t idx = 0;   // Insertion index
        uint16_t size = 0;  // Current entry count (<= kBucketSize)
    };

    // Measures matching prefix length between indices 'a' and 'b'.
    size_t MatchLength(size_t a, size_t b) const {
        JXL_DASSERT(a < b);
        if (fast_ranges_ & 2) return LZ77MatchLength(
            data_.data() + a, data_.data() + b, data_.size() - b);
        size_t len = 0;
        while (b + len < data_.size() && data_[a + len] == data_[b + len]) {
            ++len;
        }
        return len;
    }

    struct PrecomputedSpecialDist {
        uint32_t dist;
        uint8_t symbol;
    };

    const std::vector<uint32_t>& data_;
    const bool range_search_;
    const unsigned fast_ranges_;
    uint32_t hash_mask_;
    std::vector<HashBucket> hash_table_;
    std::vector<int8_t> special_dist_table_;
    std::vector<uint8_t> prefix_tags_;
    std::vector<LZ77SpecialPrefixGroup> prefix_groups_;
    LZ77SpecialPrefixFunction prefix_masks_ = nullptr;
    unsigned pending_prefix_ = 0;
    uint64_t prefix_candidates_ = 0;
    std::vector<PrecomputedSpecialDist> special_dists_;
    size_t num_special_distances_;
    MatchStats match_stats_;
};


// Fast LZ77 compression on streams of tokens using cost-based matching decision.
// Returns the compressed token streams if savings exceed the threshold.
using IndexedSeedFunction = std::function<Status(
    size_t, const SymbolCostEstimator&, std::vector<Token>*, float*, float*)>;
template<int kBucketSize = 7, int kHashSize = 3, bool kRuntimeCostComparison = true>
StatusOr<std::vector<std::vector<Token>>> ApplyLZ77_LZ77(
    const HistogramParams& params, size_t num_contexts,
    const std::vector<std::vector<Token>>& tokens, const LZ77Params& lz77,
    size_t indexed_stream = SIZE_MAX, const IndexedSeedFunction& indexed_seed = {}
) {
  LosslessProfileScope fast(params.lossless_profile, "lz77.fast");
  fast.Set("bucket_size", kBucketSize);
  fast.Set("hash_size", kHashSize);
  const char* ranges_env = std::getenv("JXL_LOSSLESS_FAST_LZ77_DISTANCES");
  const bool range_search = params.lossless_lz77_cost_spans &&
                           (!ranges_env || std::strcmp(ranges_env, "1") == 0);
  fast.Set("distance_ranges", range_search);
  const char* fast_env = std::getenv("JXL_LOSSLESS_LZ77_FAST_RANGES");
  const unsigned fast_ranges = params.lossless_lz77_cost_spans && fast_env &&
      fast_env[0] >= '1' && fast_env[0] <= '3' && fast_env[1] == '\0'
      ? fast_env[0] - '0' : 0;
  fast.Set("fast_ranges", fast_ranges);
  const char* prefix_env = std::getenv("JXL_LOSSLESS_LZ77_PREFIX_FILTER");
  // Reuse the existing lossless fast-matcher gate. Explicit 0 disables the
  // filter; 1 keeps its scalar implementation for controlled comparisons.
  const unsigned prefix_filter = !params.lossless_lz77_cost_spans || params.direct_audit
      ? 0 : prefix_env
          ? (prefix_env[0] >= '0' && prefix_env[0] <= '2' && prefix_env[1] == '\0'
                 ? prefix_env[0] - '0' : 0)
          : ((fast_ranges & 2) ? 2 : 0);
  fast.Set("prefix_filter", prefix_filter);
  std::vector<std::vector<Token>> tokens_lz77(tokens.size());
  SymbolCostEstimator sce(num_contexts, params.force_huffman, tokens, lz77,
                          params.UintConfig());
  float bit_decrease = 0;
  float total_raw_cost = 0;
  size_t in_symbols = 0;
  size_t total_symbols = 0;
  HybridUintConfig uint_config = params.UintConfig();
  struct StreamStats {
    float decrease = 0, raw_cost = 0;
  };
  std::vector<StreamStats> stats(tokens.size());
  std::vector<size_t> order;
  size_t largest = 0, max_width = 0;
  for (size_t stream = 0; stream < tokens.size(); ++stream) {
    in_symbols += tokens[stream].size();
    largest = std::max(largest, tokens[stream].size());
    if (!tokens[stream].empty()) order.push_back(stream);
    if (stream < params.image_widths.size()) {
      max_width = std::max(max_width, params.image_widths[stream]);
    }
  }
  const uint64_t scratch = (uint64_t{largest} + 1) * (prefix_filter ? 9 : 8) +
      (uint64_t{kBucketSize + 1} * 4 << 14) + uint64_t{max_width} * 8 +
      (2u << 20);
  const auto worker_limits = LosslessWorkerLimits::FromEnvironment();
  const size_t tasks = worker_limits.Tasks(order.size(), scratch);
  const char* parallel_env = std::getenv("JXL_LOSSLESS_LZ77_FAST_PARALLEL");
  const bool parallel = params.lossless_lz77_cost_spans &&
      params.lossless_lz77_pool && !params.direct_audit && !indexed_seed && tasks > 1 &&
      in_symbols >= 65536 && parallel_env && std::strcmp(parallel_env, "1") == 0;
  std::vector<std::vector<float>> worker_costs(1);

  const auto encode_stream = [&](size_t stream, size_t thread) -> Status {
    if (indexed_seed && stream == indexed_stream) {
      float saving = 0;
      JXL_RETURN_IF_ERROR(indexed_seed(stream, sce, &tokens_lz77[stream],
                                       &saving, &stats[stream].raw_cost));
      bit_decrease += saving;
      return true;
    }
    auto& sym_cost = worker_costs[thread];
    float stream_decrease = 0;
    float& decrease = parallel ? stream_decrease : bit_decrease;
    LosslessProfileScope setup(fast.Id(), "lz77.fast_setup");
    setup.Set("stream", stream);
    setup.Set("tokens", tokens[stream].size());
    size_t distance_multiplier =
        params.image_widths.size() > stream ? params.image_widths[stream] : 0;
    const auto& in = tokens[stream];
    auto& out = tokens_lz77[stream];
    std::vector<uint32_t> data;
    data.resize(in.size());

    // Cumulative sum of bit costs for fast range estimation
    sym_cost.resize(in.size() + 1);
    for (size_t pos = 0; pos < in.size(); pos++) {
      uint32_t tok, nbits, unused_bits;
      uint_config.Encode(in[pos].value, &tok, &nbits, &unused_bits);
      sym_cost[pos + 1] = sce.Bits(in[pos].context, tok) + nbits + sym_cost[pos];
      data[pos] = in[pos].value;
    }
    stats[stream].raw_cost = sym_cost[in.size()];

    out.reserve(in.size());
    size_t min_length = lz77.min_length;
    JXL_DASSERT(min_length >= 3);

    LZ77HashMap<kBucketSize, kHashSize> hash_map =
        LZ77HashMap<kBucketSize, kHashSize>(data, distance_multiplier, range_search,
                                          14, fast_ranges, prefix_filter);

    setup.Finish();
    LosslessProfileScope matching(fast.Id(), "lz77.fast_match");
    matching.Set("stream", stream);
    matching.Set("tokens", in.size());
    uint64_t match_calls = 0, accepted = 0, matched_tokens = 0, rejected = 0;

    for (size_t pos = 0; pos < in.size(); pos++) {
      out.push_back(in[pos]);

      // Guard against reading past input bounds
      if (pos + kHashSize > in.size()) {
        continue;
      }

      size_t len;
      size_t dist_symbol;

      // Find longest sequence match in current bucket
      if (matching.Enabled()) ++match_calls;
      // Sample by query count, not pixel coordinates (which alias image rows).
      if (matching.Enabled() && match_calls % 1021 == 1) {
        hash_map.template FindMatch<true>(pos, len, dist_symbol, min_length);
      } else {
        hash_map.FindMatch(pos, len, dist_symbol, min_length);
      }
      if (len < min_length) continue;

      // Bit cost comparison: raw tokens vs LZ77 pair
      float cost = sym_cost[pos + len] - sym_cost[pos];
      size_t lz77_len = len - lz77.min_length;
      float lz77_cost = LenCost(lz77_len) + DistCost(dist_symbol) +
                        sce.AddSymbolCost(out.back().context);

      if (kRuntimeCostComparison && lz77_cost > cost) {
        if (matching.Enabled()) ++rejected;
        if (cost < len && len > kHashSize) {
          out.insert(out.end(), in.begin() + pos + 1,
                       in.begin() + pos + len - kHashSize + 1);
          hash_map.UpdateRange(pos + 1, pos + len - kHashSize + 1);
          pos += len - kHashSize;
        }
        continue;
      }

      decrease += cost - lz77_cost;
      if (matching.Enabled()) {
        ++accepted;
        matched_tokens += len;
      }

      // Emit LZ77 length and distance tokens
      out.back().value = len - min_length;
      out.back().is_lz77_length = true;
      out.emplace_back(
          static_cast<uint32_t>(lz77.nonserialized_distance_context),
          static_cast<uint32_t>(dist_symbol));

      // Update hash map for skipped positions inside the match
      hash_map.UpdateRange(pos + 1, pos + len);
      pos += len - 1;
    }
    matching.Set("match_calls", match_calls);
    matching.Set("accepted_matches", accepted);
    matching.Set("matched_tokens", matched_tokens);
    matching.Set("rejected_matches", rejected);
    matching.Set("output_tokens", out.size());
    matching.Set("prefix_bytes", hash_map.PrefixBytes());
    const auto& match_stats = hash_map.Stats();
    matching.Set("sampled_queries", match_stats.queries);
    matching.Set("sampled_special_candidates", match_stats.special_candidates);
    matching.Set("sampled_bucket_candidates", match_stats.bucket_candidates);
    matching.Set("sampled_special_extends", match_stats.special_extends);
    matching.Set("sampled_bucket_extends", match_stats.bucket_extends);
    matching.Set("sampled_special_prefix_tokens", match_stats.special_prefix_tokens);
    matching.Set("sampled_bucket_prefix_tokens", match_stats.bucket_prefix_tokens);
    matching.Set("sampled_special_short_prefix", match_stats.special_short_prefix);
    matching.Set("sampled_bucket_short_prefix", match_stats.bucket_short_prefix);
    stats[stream].decrease = stream_decrease;
    return true;
  };
  fast.Set("worker_tasks", parallel ? tasks : 1);
  fast.Set("scratch_per_worker", scratch);
  if (parallel) {
    // Streams share only the read-only symbol cost model. Scratch belongs to
    // the runner worker; reductions happen in input order after joining.
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      return tokens[a].size() > tokens[b].size();
    });
    std::atomic<size_t> next{0};
    const auto init = [&](size_t threads) -> Status {
      worker_costs.resize(threads);
      fast.Set("worker_slots", threads);
      return true;
    };
    const auto work = [&](uint32_t, size_t thread) -> Status {
      for (;;) {
        const size_t index = next.fetch_add(1, std::memory_order_relaxed);
        if (index >= order.size()) break;
        JXL_RETURN_IF_ERROR(encode_stream(order[index], thread));
      }
      return true;
    };
    JXL_RETURN_IF_ERROR(RunOnPool(params.lossless_lz77_pool, 0, tasks, init,
                                  work, "Lz77FastStreams"));
  } else {
    fast.Set("worker_slots", 1);
    for (size_t stream = 0; stream < tokens.size(); ++stream) {
      JXL_RETURN_IF_ERROR(encode_stream(stream, 0));
    }
  }
  for (size_t stream = 0; stream < tokens.size(); ++stream) {
    total_raw_cost += stats[stream].raw_cost;
    total_symbols += tokens_lz77[stream].size();
    if (parallel) bit_decrease += stats[stream].decrease;
  }

  // Return LZ77 streams only if net bit savings justify overhead
  float min_bit_decrease = total_symbols * 0.2f + 16.0f;
  if (kRuntimeCostComparison && total_raw_cost <= in_symbols * 2.5f) {
    min_bit_decrease =
        std::min(min_bit_decrease, total_raw_cost * 0.01f + 256.0f);
  }
  fast.Set("tokens", in_symbols);
  fast.Set("output_tokens", total_symbols);
  fast.Set("kept", bit_decrease > min_bit_decrease);
  if (params.direct_audit || bit_decrease > min_bit_decrease) {
    return tokens_lz77;
  }
  return std::vector<std::vector<Token>>{};
}

// Greedy LZ77 with lazy matching, a hash chain of up to 256 candidates and a
// per-match cost check (the greedy method before #4928). Used as the first pass
// of ApplyLZ77_Optimal for small inputs. Unlike ApplyLZ77_LZ77, it only takes
// matches that are cheaper than the literals they replace. On small token
// streams, taking every match can make LZ77 look useless, and then optimal
// matching is skipped altogether.
constexpr uint32_t kLazyGreedyChainLength = 256;
// Inputs with fewer tokens than this use ApplyLZ77_LazyGreedy as the first pass
// of ApplyLZ77_Optimal (and all inputs with lz77_careful_first_pass).
constexpr size_t kLazyGreedyMaxTokens = size_t{1} << 16;

std::vector<std::vector<Token>> ApplyLZ77_LazyGreedy(
    const HistogramParams& params, size_t num_contexts,
    const std::vector<std::vector<Token>>& tokens, const LZ77Params& lz77) {
  LosslessProfileScope lazy(params.lossless_profile, "lz77.lazy");
  std::vector<std::vector<Token>> tokens_lz77(tokens.size());
  SymbolCostEstimator sce(num_contexts, params.force_huffman, tokens, lz77,
                          params.UintConfig());
  float bit_decrease = 0;
  size_t total_symbols = 0;
  HybridUintConfig uint_config = params.UintConfig();
  std::vector<float> sym_cost;
  for (size_t stream = 0; stream < tokens.size(); stream++) {
    LosslessProfileScope setup(lazy.Id(), "lz77.lazy_setup");
    setup.Set("stream", stream);
    setup.Set("tokens", tokens[stream].size());
    size_t distance_multiplier =
        params.image_widths.size() > stream ? params.image_widths[stream] : 0;
    const auto& in = tokens[stream];
    auto& out = tokens_lz77[stream];
    total_symbols += in.size();
    // Cumulative sum of bit costs.
    sym_cost.resize(in.size() + 1);
    for (size_t i = 0; i < in.size(); i++) {
      uint32_t tok, nbits, unused_bits;
      uint_config.Encode(in[i].value, &tok, &nbits, &unused_bits);
      sym_cost[i + 1] = sce.Bits(in[i].context, tok) + nbits + sym_cost[i];
    }

    out.reserve(in.size());
    size_t max_distance = std::min(in.size(), kWindowSize);
    size_t min_length = lz77.min_length;
    JXL_DASSERT(min_length >= 3);
    size_t max_length = in.size();

    // Use next power of two as window size.
    size_t window_size = 1;
    while (window_size < max_distance && window_size < kWindowSize) {
      window_size <<= 1;
    }

    HashChain<kLazyGreedyChainLength> chain(in.data(), in.size(), window_size,
                                           min_length, max_length,
                                           distance_multiplier);
    setup.Finish();
    LosslessProfileScope matching(lazy.Id(), "lz77.lazy_match");
    matching.Set("stream", stream);
    matching.Set("tokens", in.size());
    size_t len;
    size_t dist_symbol;

    const size_t max_lazy_match_len = 256;  // 0 to disable lazy matching

    // Whether the next symbol was already updated (to test lazy matching)
    bool already_updated = false;
    for (size_t i = 0; i < in.size(); i++) {
      out.push_back(in[i]);
      if (!already_updated) chain.Update(i);
      already_updated = false;
      chain.FindMatch(i, max_distance, &dist_symbol, &len);
      if (len >= min_length) {
        if (len < max_lazy_match_len && i + 1 < in.size()) {
          // Try length at next symbol lazy matching
          chain.Update(i + 1);
          already_updated = true;
          size_t len2, dist_symbol2;
          chain.FindMatch(i + 1, max_distance, &dist_symbol2, &len2);
          if (len2 > len) {
            // Use the lazy match. Add literal, and use the next length starting
            // from the next byte.
            ++i;
            already_updated = false;
            len = len2;
            dist_symbol = dist_symbol2;
            out.push_back(in[i]);
          }
        }

        float cost = sym_cost[i + len] - sym_cost[i];
        size_t lz77_len = len - lz77.min_length;
        float lz77_cost = LenCost(lz77_len) + DistCost(dist_symbol) +
                          sce.AddSymbolCost(out.back().context);

        if (lz77_cost <= cost) {
          out.back().value = len - min_length;
          out.back().is_lz77_length = true;
          out.emplace_back(
              static_cast<uint32_t>(lz77.nonserialized_distance_context),
              static_cast<uint32_t>(dist_symbol));
          bit_decrease += cost - lz77_cost;
        } else {
          // LZ77 match ignored, and symbol already pushed. Push all other
          // symbols and skip.
          for (size_t j = 1; j < len; j++) {
            out.push_back(in[i + j]);
          }
        }

        if (already_updated) {
          chain.Update(i + 2, len - 2);
          already_updated = false;
        } else {
          chain.Update(i + 1, len - 1);
        }
        i += len - 1;
      } else {
        // Literal, already pushed
      }
    }
  }

  lazy.Set("tokens", total_symbols);
  lazy.Set("kept", bit_decrease > total_symbols * 0.2 + 16);
  if (bit_decrease > total_symbols * 0.2 + 16) {
    return tokens_lz77;
  }
  return {};
}

struct LZ77CachedMatches {
  struct Position {
    size_t position;
    uint32_t offset;
    uint32_t count;
  };
  std::vector<Position> positions;
  std::vector<LZ77Match> matches;
  size_t replay_limit = std::numeric_limits<size_t>::max();
};

template <uint32_t kMaxChainLength>
std::vector<std::vector<Token>> ApplyLZ77_RangeParsed(
    const HistogramParams& params, size_t num_contexts,
    const std::vector<std::vector<Token>>& tokens, const LZ77Params& lz77,
    std::vector<std::vector<Token>> seed, int mode) {
  const bool refit = mode != 5;
  const bool cache_matches = mode == 4;
  const bool length_spans = lz77.length_uint_config.split_exponent == 0 &&
                           lz77.length_uint_config.msb_in_token == 0 &&
                           lz77.length_uint_config.lsb_in_token == 0;
  const HybridUintConfig uint_config = params.UintConfig();
  constexpr size_t kCacheLimit = 32u << 20;
  size_t cache_bytes = 0;
  std::vector<LZ77CachedMatches> caches(cache_matches ? tokens.size() : 0);
  for (int pass = 0; pass < (refit ? 2 : 1); ++pass) {
    const SymbolCostEstimator sce(num_contexts + 1, params.force_huffman,
                                   seed, lz77, uint_config);
    std::vector<std::vector<Token>> parsed(tokens.size());
    for (size_t stream = 0; stream < tokens.size(); ++stream) {
      const auto& in = tokens[stream];
      if (in.empty()) continue;
      auto& out = parsed[stream];
      auto* cache = cache_matches ? &caches[stream] : nullptr;
      const bool full_replay = pass != 0 && cache &&
          cache->replay_limit == std::numeric_limits<size_t>::max();
      LosslessProfileScope profile(params.lossless_profile, "lz77.range_parse");
      profile.Set("pass", pass);
      profile.Set("tokens", in.size());
      profile.Set("replay", full_replay);
      const size_t width = stream < params.image_widths.size()
                               ? params.image_widths[stream] : 0;
      const size_t max_distance = std::min(in.size(), kWindowSize);
      size_t window = 1;
      while (window < max_distance && window < kWindowSize) window <<= 1;
      std::unique_ptr<HashChain<kMaxChainLength, true>> chain;
      if (!full_replay) {
        chain = std::make_unique<HashChain<kMaxChainLength, true>>(
            in.data(), in.size(), window, lz77.min_length, in.size(), width);
      }
      LZ77RangeParser parser(in.size());
      std::vector<LZ77Match> matches;
      std::vector<LZ77MatchRange> ranges;
      matches.reserve(kMaxChainLength);
      ranges.reserve(kMaxChainLength);
      // Default HybridUint(4,2,0) has 128 classes for uint32 distances.
      // Each class has a fixed token and extra-bit count under both models.
      std::array<size_t, 128> seen{};
      std::array<size_t, 128> slots{};
      size_t replay_index = 0, skip = 0, repeats = 0;
      size_t previous_end = 0, previous_distance = 0;
      uint64_t match_calls = 0, replay_positions = 0, endpoints = 0;
      for (size_t i = 0; i < in.size(); ++i) {
        parser.Advance(i);
        uint32_t tok, nbits, bits;
        uint_config.Encode(in[i].value, &tok, &nbits, &bits);
        parser.Literal(i, in[i].context,
                       parser.At(i).cost + (sce.Bits(in[i].context, tok) + nbits));
        matches.clear();
        if (chain) chain->Update(i);
        const bool replay = pass != 0 && cache && i < cache->replay_limit;
        if (replay) {
          if (replay_index == cache->positions.size() ||
              cache->positions[replay_index].position != i) continue;
          const auto& saved = cache->positions[replay_index++];
          matches.assign(cache->matches.begin() + saved.offset,
                          cache->matches.begin() + saved.offset + saved.count);
          ++replay_positions;
        } else {
          if (skip != 0) { --skip; continue; }
          ++match_calls;
          chain->template FindMatches<true>(
              i, max_distance, [&](size_t len, uint32_t distance) {
                uint32_t cls, extra_bits, unused;
                HybridUintConfig().Encode(distance, &cls, &extra_bits, &unused);
                if (cls >= seen.size() || seen[cls] != i + 1) {
                  if (cls < seen.size()) {
                    seen[cls] = i + 1;
                    slots[cls] = matches.size();
                  }
                  matches.push_back({static_cast<uint32_t>(len), distance});
                } else {
                  auto& kept = matches[slots[cls]];
                  if (len > kept.length || (len == kept.length && distance < kept.distance)) {
                    kept = {static_cast<uint32_t>(len), distance};
                  }
                }
              });
          if (matches.empty()) { repeats = 0; continue; }
          if (pass == 0 && cache && i < cache->replay_limit) {
            const size_t bytes = sizeof(LZ77CachedMatches::Position) +
                                 matches.size() * sizeof(LZ77Match);
            if (bytes > kCacheLimit - cache_bytes) {
              cache->replay_limit = i;
            } else {
              cache->positions.push_back({i, static_cast<uint32_t>(cache->matches.size()),
                                          static_cast<uint32_t>(matches.size())});
              cache->matches.insert(cache->matches.end(), matches.begin(), matches.end());
              cache_bytes += bytes;
            }
          }
        }
        BuildLZ77MatchRanges(&matches, lz77.min_length,
                            [&](uint32_t distance) { return sce.DistCost(distance, lz77); },
                            &ranges);
        for (const auto& range : ranges) {
          size_t begin = range.begin;
          endpoints += range.end - begin;
          while (begin < range.end) {
            const size_t residual = begin - lz77.min_length;
            const size_t limit = range.end - lz77.min_length;
            size_t extent = residual + 1;
            if (length_spans) {
              extent = 1;
              while (extent <= residual) {
                extent = extent > limit / 2 ? limit : extent * 2;
              }
            }
            const size_t end = std::min(range.end, lz77.min_length + extent);
            const float cost = parser.At(i).cost +
                (sce.LenCost(in[i].context, residual, lz77) + range.distance_cost);
            parser.Match(i, begin, end, range.distance, in[i].context, cost);
            begin = end;
          }
        }
        if (!replay) {
          const size_t length = ranges.back().end - 1;
          const size_t distance = ranges.back().distance;
          const size_t end = i + length;
          if (distance == 0 || (distance == 1 && width != 0) ||
              (length > 255 && end == previous_end && distance == previous_distance)) {
            ++repeats;
          } else {
            repeats = 0;
          }
          previous_end = end;
          previous_distance = distance;
          if (repeats >= 8 && length > 8) {
            skip = length - 9;
            repeats = 0;
          }
        }
      }
      parser.Advance(in.size());
      for (size_t pos = in.size(); pos != 0;) {
        const auto& step = parser.At(pos);
        JXL_DASSERT(step.length != 0 && step.length <= pos);
        if (step.length == 0 || step.length > pos) return {};
        const bool is_match = step.distance != 0;
        if (is_match) out.emplace_back(lz77.nonserialized_distance_context, step.distance - 1);
        out.emplace_back(step.context, is_match ? step.length - lz77.min_length
                                               : in[pos - 1].value);
        out.back().is_lz77_length = is_match;
        pos -= step.length;
      }
      std::reverse(out.begin(), out.end());
      profile.Set("match_calls", match_calls);
      profile.Set("replay_positions", replay_positions);
      profile.Set("length_candidates", endpoints);
      profile.Set("intervals", parser.Intervals());
      profile.Set("peak_queued", parser.PeakQueued());
      profile.Set("scalar_endpoints", parser.ScalarEndpoints());
      profile.Set("cache_bytes", cache_bytes);
      profile.Set("partial_replay", pass != 0 && cache && !full_replay);
    }
    seed = std::move(parsed);
  }
  return seed;
}

template <uint32_t kMaxChainLength, size_t kBucketCapacity = 0, bool kHistory = false,
          bool kFullPrefix = false, size_t kLongCapacity = 8, typename Cost = float,
          bool kCompact = false>
StatusOr<std::vector<std::vector<Token>>> ApplyLZ77_Optimal(
    const HistogramParams& params, size_t num_contexts,
    const std::vector<std::vector<Token>>& tokens, const LZ77Params& lz77,
    const std::vector<std::vector<Token>>* cost_seed = nullptr) {
  // First pass: decides whether optimal matching runs, and gives the symbol
  // cost estimate. Small inputs (ICC profiles, MA trees, small images) use the
  // slower lazy greedy method: the fast one can reject LZ77 for them although
  // it pays off. For large inputs the fast method is good enough.
  size_t total_tokens = 0;
  for (const auto& stream : tokens) total_tokens += stream.size();
  size_t indexed_seed_stream = SIZE_MAX;
  std::function<Status(const std::vector<Token>&, const SymbolCostEstimator&,
                       std::vector<Token>*)> reuse_index;
  auto first_pass = [&]() -> StatusOr<std::vector<std::vector<Token>>> {
    if (cost_seed) return std::vector<std::vector<Token>>(*cost_seed);
    LosslessProfileScope first(params.lossless_profile, "lz77.first_pass");
    first.Set("tokens", total_tokens);
    const bool careful = params.lz77_careful_first_pass ||
                         total_tokens < kLazyGreedyMaxTokens;
    first.Set("careful", careful);
    if (careful)
      return ApplyLZ77_LazyGreedy(params, num_contexts, tokens, lz77);
    if constexpr (kBucketCapacity == 32 && !kHistory && kCompact) {
      const char* reuse_env = std::getenv("JXL_LOSSLESS_LZ77_SHARED_INDEX");
      const char* indexed_env = std::getenv("JXL_LOSSLESS_LZ77_INDEXED");
      const char* block_dp_env = std::getenv("JXL_LOSSLESS_LZ77_BLOCK_DP");
      if (params.lossless_lz77_cost_spans && !params.direct_audit &&
          reuse_env && std::strcmp(reuse_env, "1") == 0 && indexed_env &&
          std::strcmp(indexed_env, "2") == 0 && block_dp_env &&
          std::strcmp(block_dp_env, "2") == 0 && !tokens.empty()) {
        size_t largest = 0;
        for (size_t i = 1; i < tokens.size(); ++i)
          if (tokens[i].size() > tokens[largest].size()) largest = i;
        if (tokens[largest].size() >= 65536 &&
            tokens[largest].size() <= (1u << 24)) {
          indexed_seed_stream = largest;
          const size_t width = largest < params.image_widths.size()
                                   ? params.image_widths[largest]
                                   : 0;
          size_t window = 1;
          while (window < tokens[largest].size() && window < kWindowSize)
            window <<= 1;
          const auto run =
              [&](auto tag) -> StatusOr<std::vector<std::vector<Token>>> {
            using Index =
                LZ77StaticIndex<32, kFullPrefix, kLongCapacity, decltype(tag)>;
            LosslessProfileScope build(params.lossless_profile,
                                       "lz77.shared_index_build");
            auto index = std::make_shared<Index>(
                tokens[largest].data(), tokens[largest].size(), window,
                lz77.min_length, width, true, false, 0, false, true, 0, true);
            JXL_RETURN_IF_ERROR(index->BuildParallel(
                tokens[largest].data(), params.lossless_lz77_pool));
            build.Set("tokens", tokens[largest].size());
            build.Set("value_bytes", sizeof(tag));
            build.Finish();
            reuse_index = [&, index, largest, width, window](
                              const std::vector<Token>& seed,
                              const SymbolCostEstimator& costs,
                              std::vector<Token>* out) -> Status {
              return ParseLZ77Blocks<true>(tokens[largest], seed, *index, costs,
                                           params, lz77, width, window, out);
            };
            IndexedSeedFunction seed =
                [&, index](size_t stream, const SymbolCostEstimator& costs,
                           std::vector<Token>* out, float* saving,
                           float* raw) -> Status {
              const auto literal_cost = [&](const Token& token) {
                uint32_t symbol, nbits, bits;
                params.UintConfig().Encode(token.value, &symbol, &nbits, &bits);
                return nbits + costs.Bits(token.context, symbol);
              };
              const auto copy_cost = [&](const Token& token, size_t length,
                                         uint32_t distance) {
                return LenCost(length - lz77.min_length) + DistCost(distance) +
                       costs.AddSymbolCost(token.context);
              };
              return LZ77IndexedSeed(
                  tokens[stream], *index, params, lz77, literal_cost, copy_cost,
                  [](uint32_t d) { return DistCost(d); }, out, saving, raw);
            };
            return ApplyLZ77_LZ77<7, 3, true>(params, num_contexts, tokens,
                                              lz77, largest, seed);
          };
          if (std::all_of(
                  tokens[largest].begin(), tokens[largest].end(),
                  [](const Token& t) { return t.value <= UINT16_MAX; })) {
            return run(uint16_t{0});
          }
          return run(uint32_t{0});
        }
      }
    }
    const char* seed_blocks = std::getenv("JXL_LOSSLESS_LZ77_SEED_BLOCKS");
    if (params.lossless_lz77_cost_spans && params.lossless_lz77_pool &&
        !params.direct_audit && seed_blocks &&
        (std::strcmp(seed_blocks, "1") == 0 || std::strcmp(seed_blocks, "2") == 0)) {
      const bool warm = seed_blocks[0] == '2';
      const size_t seed_history = LosslessWorkerSetting(
          std::getenv("JXL_LOSSLESS_LZ77_SEED_HISTORY"), 32768, 1u << 20);
      size_t biggest = 0;
      for (size_t i = 1; i < tokens.size(); ++i) {
        if (tokens[i].size() > tokens[biggest].size()) biggest = i;
      }
      if (!tokens.empty() && tokens[biggest].size() >= 262144 &&
          tokens[biggest].size() >= total_tokens - total_tokens / 4) {
        const size_t parts = LosslessWorkerLimits::FromEnvironment().Tasks(
            std::min<size_t>(16, tokens[biggest].size() / 65536), 8u << 20);
        auto block_params = params;
        block_params.image_widths.clear();
        std::vector<std::vector<Token>> blocks;
        std::vector<size_t> owners, prefixes;
        for (size_t i = 0; i < tokens.size(); ++i) {
          const size_t count = i == biggest ? parts : 1;
          for (size_t part = 0; part < count; ++part) {
            const size_t begin = tokens[i].size() * part / count;
            const size_t prefix = warm ? std::min(begin, seed_history) : 0;
            blocks.emplace_back(tokens[i].begin() + begin - prefix,
                                tokens[i].begin() + tokens[i].size() * (part + 1) / count);
            owners.push_back(i);
            prefixes.push_back(prefix);
            block_params.image_widths.push_back(i < params.image_widths.size() ? params.image_widths[i] : 0);
          }
        }
        JXL_ASSIGN_OR_RETURN(auto seeds,
            (ApplyLZ77_LZ77<7, 3, true>(block_params, num_contexts, blocks, lz77)));
        if (seeds.empty()) return seeds;
        std::vector<std::vector<Token>> result(tokens.size());
        std::vector<size_t> sizes(tokens.size());
        for (size_t i = 0; i < seeds.size(); ++i) sizes[owners[i]] += seeds[i].size();
        for (size_t i = 0; i < result.size(); ++i) result[i].reserve(sizes[i]);
        for (size_t i = 0; i < seeds.size(); ++i) {
          auto& dest = result[owners[i]];
          if (prefixes[i]) {
            JXL_RETURN_IF_ERROR(TrimLZ77SeedPrefix(blocks[i], prefixes[i], seeds[i],
                                                   lz77.min_length, &dest));
          } else {
            dest.insert(dest.end(), seeds[i].begin(), seeds[i].end());
          }
        }
        first.Set("seed_blocks", parts);
        first.Set("warm_prefix_tokens", std::accumulate(prefixes.begin(), prefixes.end(), size_t{0}));
        return result;
      }
    }
    return ApplyLZ77_LZ77<7, 3, true>(params, num_contexts, tokens, lz77);
  };
  JXL_ASSIGN_OR_RETURN(std::vector<std::vector<Token>> tokens_for_cost_estimate,
                       first_pass());
  // If greedy-LZ77 does not give better compression than no-lz77, no reason to
  // run the optimal matching.
  if (tokens_for_cost_estimate.empty()) return std::vector<std::vector<Token>>{};
  const char* match_cost_env = std::getenv("JXL_LOSSLESS_LZ77_MATCH_COST");
  int match_cost_mode = params.lossless_lz77_cost_spans && match_cost_env &&
      match_cost_env[0] >= '1' && match_cost_env[0] <= '7' && match_cost_env[1] == '\0'
      ? match_cost_env[0] - '0' : 0;
  {
    LosslessProfileScope choice(params.lossless_profile, "lz77.parser_choice");
    if (choice.Enabled() || match_cost_mode == 7) {
      uint64_t matches = 0, covered = 0, long_covered = 0;
      for (const auto& stream : tokens_for_cost_estimate) {
        for (const auto& token : stream) {
          if (!token.is_lz77_length) continue;
          const size_t length = token.value + lz77.min_length;
          ++matches;
          covered += length;
          if (length >= 64) long_covered += length;
        }
      }
      choice.Set("tokens", total_tokens);
      choice.Set("matches", matches);
      choice.Set("covered", covered);
      choice.Set("long_covered", long_covered);
      choice.Set("requested", match_cost_mode);
      if (match_cost_mode == 7) {
        match_cost_mode = PreferLZ77RangeParsing(total_tokens, matches, covered, long_covered)
                              ? 5 : 0;
      }
      choice.Set("mode", match_cost_mode);
    }
  }
  if (match_cost_mode >= 4) {
    return ApplyLZ77_RangeParsed<kMaxChainLength>(
        params, num_contexts, tokens, lz77, std::move(tokens_for_cost_estimate),
        match_cost_mode);
  }
  SymbolCostEstimator sce = [&] {
    LosslessProfileScope costs(params.lossless_profile, "lz77.cost_model");
    costs.Set("contexts", num_contexts);
    return SymbolCostEstimator(num_contexts + 1, params.force_huffman,
                               tokens_for_cost_estimate, lz77, params.UintConfig());
  }();
  std::vector<std::vector<Token>> tokens_lz77(tokens.size());
  HybridUintConfig uint_config = params.UintConfig();
  const char* cost_spans_env = std::getenv("JXL_LOSSLESS_LZ77_COST_SPANS");
  const char* detail_env = std::getenv("JXL_LOSSLESS_LZ77_DETAIL");
  const bool detailed = detail_env && std::strcmp(detail_env, "1") == 0;
  const bool cost_spans = params.lossless_lz77_cost_spans &&
                         lz77.length_uint_config.split_exponent == 0 &&
                         lz77.length_uint_config.msb_in_token == 0 &&
                         lz77.length_uint_config.lsb_in_token == 0 &&
                         (!cost_spans_env || cost_spans_env[0] != '0');
  const char* dominance_env = std::getenv("JXL_LOSSLESS_LZ77_DOMINANCE");
  const bool dominance = params.lossless_lz77_cost_spans && !params.direct_audit &&
      match_cost_mode >= 2 && dominance_env && std::strcmp(dominance_env, "1") == 0;
  const char* interval_env = std::getenv("JXL_LOSSLESS_LZ77_INTERVAL_DP");
  const int interval_dp = params.lossless_lz77_cost_spans && !params.direct_audit &&
      cost_spans && interval_env &&
      (interval_env[0] == '1' || interval_env[0] == '2') &&
      interval_env[1] == '\0' ? interval_env[0] - '0' : 0;
  const char* match_cache_env = std::getenv("JXL_LOSSLESS_LZ77_MATCH_CACHE");
  const bool allow_match_cache = params.lossless_lz77_cost_spans && kBucketCapacity == 0 &&
      (!match_cache_env || std::strcmp(match_cache_env, "0") != 0);
  const bool force_match_cache = match_cache_env &&
      std::strcmp(match_cache_env, "1") == 0;
  // Experimental search-depth override; keep the parser, first pass, effort
  // and bucket matchers unchanged. Never affect lossy coding or plan audits.
  const uint32_t chain_limit = params.lossless_lz77_cost_spans && !params.direct_audit
      ? LosslessLZ77ChainLimit(std::getenv("JXL_LOSSLESS_LZ77_CHAIN_LIMIT"), kMaxChainLength)
      : kMaxChainLength;
  struct StoredContextStep {
    uint32_t len;
    uint32_t dist_symbol;
    uint32_t ctx;
    Cost total_cost = std::numeric_limits<Cost>::max();
  };
  struct CompactStep {
    uint32_t len;
    uint32_t dist_symbol;
    Cost total_cost = std::numeric_limits<Cost>::max();
  };
  using MatchInfo = std::conditional_t<kCompact, CompactStep, StoredContextStep>;
  LosslessProfileScope dispatch(params.lossless_profile, "lz77.stream_dispatch");
  std::vector<size_t> order;
  std::vector<uint8_t> match_cache_streams(tokens.size());
  bool any_match_cache = false;
  size_t largest = 0;
  for (size_t stream = 0; stream < tokens.size(); ++stream) {
    if (tokens[stream].empty()) continue;
    order.push_back(stream);
    largest = std::max(largest, tokens[stream].size());
    match_cache_streams[stream] = allow_match_cache &&
        (force_match_cache || PreferLZ77MatchCaching(
            tokens[stream].size(), tokens_for_cost_estimate[stream].size()));
    any_match_cache |= match_cache_streams[stream] != 0;
  }
  const char* index_env = std::getenv("JXL_LOSSLESS_LZ77_INDEXED");
  const char* small_env = std::getenv("JXL_LOSSLESS_LZ77_SMALL_INDEX");
  const bool small_index = small_env && std::strcmp(small_env, "1") == 0;
  const size_t index_minimum = small_index ? 8192 : 65536;
  const char* bucket_audit_env = std::getenv("JXL_LOSSLESS_LZ77_BUCKET_AUDIT");
  const bool indexed = kBucketCapacity == 32 && !kHistory && kCompact &&
      params.lossless_lz77_cost_spans && !params.direct_audit &&
      (params.lossless_lz77_pool || small_index) && match_cost_mode == 2 &&
      largest >= index_minimum && largest <= (1u << 24) && index_env &&
      ((std::strcmp(index_env, "1") == 0 && largest >= total_tokens - total_tokens / 4) ||
       std::strcmp(index_env, "2") == 0) &&
      !(bucket_audit_env && std::strcmp(bucket_audit_env, "1") == 0);
  // Bound extra parser scratch, excluding the input and retained output streams.
  // Hash windows are capped at 1 Mi tokens; cost/distance arrays are per stream.
  // Compact nodes recover the context from the match's original start token.
  // Account for nodes, cumulative costs, copied values, and preallocated match
  // distances (plus distance costs only when requested). Reserving the known
  // bound avoids a second allocation and growth-capacity slack for long runs.
  // The compact 32-entry bucket modes fit below the chain's 20 MiB allowance
  // (including the wide long-prefix queue). Older bucket modes remain serial.
  const uint64_t per_token = kCompact
      ? sizeof(MatchInfo) + sizeof(Cost) + sizeof(uint32_t) +
            sizeof(uint32_t) + (match_cost_mode != 0 ? sizeof(float) : 0)
      : (sizeof(Cost) == 8 ? 64 : 48);
  const uint64_t scratch = (uint64_t{largest} + (kCompact ? 1 : 0)) * per_token +
      uint64_t{20 + (any_match_cache ? 4u : 0u)} * kWindowSize + (1u << 20);
  size_t workers = 1;
  if (params.lossless_lz77_pool && params.lossless_lz77_cost_spans &&
      !params.direct_audit && (kBucketCapacity == 0 ||
                              (kCompact && kBucketCapacity == 32)) &&
      total_tokens >= 65536) {
    workers = std::max<size_t>(1,
        LosslessWorkerLimits::FromEnvironment().Tasks(order.size(), scratch));
  }
  // The indexed matcher uses the same pool inside a stream; never nest pools.
  if (indexed || reuse_index) workers = 1;
  dispatch.Set("worker_tasks", workers);
  dispatch.Set("streams", order.size());
  dispatch.Set("scratch_per_worker", scratch);
  dispatch.Set("compact_steps", kCompact);
  dispatch.Set("bucket_capacity", kBucketCapacity);
  dispatch.Set("step_bytes", sizeof(MatchInfo));
  dispatch.Set("indexed", indexed);
  const auto process_stream = [&](size_t stream, const HistogramParams& params) -> Status {
    if (reuse_index && stream == indexed_seed_stream) {
      JXL_RETURN_IF_ERROR(reuse_index(tokens_for_cost_estimate[stream], sce,
                                       &tokens_lz77[stream]));
      reuse_index = {};
      return true;
    }
    std::vector<Cost> sym_cost;
    std::vector<uint32_t> dist_symbols;
    std::vector<float> dist_costs;
    size_t distance_multiplier =
        params.image_widths.size() > stream ? params.image_widths[stream] : 0;
    const auto& in = tokens[stream];
    auto& out = tokens_lz77[stream];
    const bool match_cache = match_cache_streams[stream] != 0;
    LosslessProfileScope stream_profile(dispatch.Id(), "lz77.optimal_stream");
    stream_profile.Set("stream", stream);
    stream_profile.Set("tokens", in.size());
    stream_profile.Set("chain_limit", kBucketCapacity ? kMaxChainLength : chain_limit);
    const char* block_env = std::getenv("JXL_LOSSLESS_LZ77_BLOCK_DP");
    const char* tag_env = std::getenv("JXL_LOSSLESS_LZ77_INDEX_TAGS");
    const bool index_tags = tag_env &&
        (std::strcmp(tag_env, "1") == 0 || std::strcmp(tag_env, "2") == 0);
    const char* run_env = std::getenv("JXL_LOSSLESS_LZ77_RUN_KEYS");
    const unsigned run_keys = run_env && run_env[0] >= '1' && run_env[0] <= '2' &&
        run_env[1] == '\0' ? run_env[0] - '0' : 0;
    const char* coordinate_env = std::getenv("JXL_LOSSLESS_LZ77_INDEX_COORDINATES");
    const bool coordinates = coordinate_env && std::strcmp(coordinate_env, "1") == 0;
    const char* anchor_env = std::getenv("JXL_LOSSLESS_LZ77_ANCHORS_FIRST");
    const bool anchors_first = anchor_env && std::strcmp(anchor_env, "1") == 0;
    const char* deep_env = std::getenv("JXL_LOSSLESS_LZ77_DEEP_PREFIX");
    const unsigned deep_prefix = deep_env && deep_env[0] >= '1' &&
        deep_env[0] <= '2' && deep_env[1] == '\0' ? deep_env[0] - '0' : 0;
    if (indexed && cost_spans && in.size() >= index_minimum && block_env &&
        (std::strcmp(block_env, "1") == 0 || std::strcmp(block_env, "2") == 0)) {
      size_t window = 1;
      while (window < in.size() && window < kWindowSize) window <<= 1;
      LosslessProfileScope build(params.lossless_profile, "lz77.index_build");
      const char* suffix_env = std::getenv("JXL_LOSSLESS_LZ77_SUFFIX");
      if (suffix_env && suffix_env[0] >= '1' && suffix_env[0] <= '4' &&
          suffix_env[1] == '\0') {
        LZ77SuffixIndex suffix;
        if (suffix.Build(in.data(), in.size(), window, lz77.min_length,
                          distance_multiplier, suffix_env[0] == '2' || suffix_env[0] == '3',
                          suffix_env[0] >= '3')) {
          build.Set("tokens", in.size());
          build.Set("suffix", 1);
          build.Finish();
          return ParseLZ77Blocks<true>(in, tokens_for_cost_estimate[stream], suffix, sce,
                                       params, lz77, distance_multiplier, window, &out);
        }
      }
      const char* build_env = std::getenv("JXL_LOSSLESS_LZ77_PARALLEL_INDEX");
      const bool parallel_build = params.lossless_lz77_pool && build_env &&
          std::strcmp(build_env, "1") == 0 && run_keys != 2 && deep_prefix == 0;
      const char* word_env = std::getenv("JXL_LOSSLESS_LZ77_WORD_MATCH");
      const bool word_match = word_env && std::strcmp(word_env, "1") == 0;
      const auto parse_indexed = [&](auto tag) -> Status {
      LZ77StaticIndex<32, kFullPrefix, kLongCapacity, decltype(tag)> index(
          in.data(), in.size(), window, lz77.min_length, distance_multiplier,
          index_tags, index_tags && tag_env[0] == '2', run_keys, coordinates,
          anchors_first, deep_prefix, parallel_build, word_match);
      if (parallel_build) JXL_RETURN_IF_ERROR(index.BuildParallel(in.data(), params.lossless_lz77_pool));
      build.Set("tokens", in.size());
      build.Set("run_keys", run_keys);
      build.Set("coordinates", coordinates);
      build.Set("anchors_first", anchors_first);
      build.Set("deep_prefix", deep_prefix);
      build.Set("parallel_build", parallel_build);
      build.Set("value_bytes", sizeof(tag));
      build.Set("word_match", word_match);
      build.Finish();
      if (block_env[0] == '2') {
        return ParseLZ77Blocks<true>(in, tokens_for_cost_estimate[stream], index, sce,
                                    params, lz77, distance_multiplier, window, &out);
      }
      return ParseLZ77Blocks<false>(in, tokens_for_cost_estimate[stream], index, sce,
                                   params, lz77, distance_multiplier, window, &out);
      };
      const char* narrow = std::getenv("JXL_LOSSLESS_LZ77_COMPACT_VALUES");
      if (narrow && std::strcmp(narrow, "1") == 0 &&
          std::all_of(in.begin(), in.end(), [](const Token& token) { return token.value <= UINT16_MAX; })) {
        return parse_indexed(uint16_t{0});
      }
      return parse_indexed(uint32_t{0});
    }
    LosslessProfileScope setup(params.lossless_profile, "lz77.setup");
    if constexpr (kCompact) {
      dist_symbols.reserve(in.size() + 1);
      if (match_cost_mode != 0) dist_costs.reserve(in.size() + 1);
    }
    // Cumulative sum of bit costs.
    sym_cost.resize(in.size() + 1);
    for (size_t i = 0; i < in.size(); i++) {
      uint32_t tok, nbits, unused_bits;
      uint_config.Encode(in[i].value, &tok, &nbits, &unused_bits);
      sym_cost[i + 1] = sce.Bits(in[i].context, tok) + nbits + sym_cost[i];
    }

    out.reserve(in.size());
    size_t max_distance = std::min(in.size(), kWindowSize);
    size_t min_length = lz77.min_length;
    JXL_DASSERT(min_length >= 3);
    size_t max_length = in.size();

    // Use next power of two as window size.
    size_t window_size = 1;
    while (window_size < max_distance && window_size < kWindowSize) {
      window_size <<= 1;
    }

    using Matcher = std::conditional_t<kBucketCapacity == 0,
        HashChain<kMaxChainLength>,
        LZ77BucketMatcher<kBucketCapacity, kHistory, kFullPrefix, kLongCapacity>>;
    Matcher chain(
        in.data(), in.size(), window_size, min_length, max_length, distance_multiplier);
    using StaticIndex = LZ77StaticIndex<32, kFullPrefix, kLongCapacity>;
    std::unique_ptr<StaticIndex> static_index;
    constexpr size_t kMatchBatch = 16384;
    std::vector<LZ77MatchDominance> indexed_matches;
    std::vector<std::unique_ptr<LZ77MatchCache>> indexed_caches;
    size_t indexed_tasks = 1;
    size_t indexed_begin = 0, indexed_end = 0;
    uint64_t indexed_positions = 0, indexed_replays = 0, indexed_overflows = 0;
    if (indexed && in.size() >= index_minimum) {
      LosslessProfileScope build(params.lossless_profile, "lz77.index_build");
      static_index = std::make_unique<StaticIndex>(
          in.data(), in.size(), window_size, min_length, distance_multiplier,
          index_tags, false, run_keys, coordinates, anchors_first, deep_prefix);
      indexed_matches.resize(kMatchBatch);
      const char* cache_env = std::getenv("JXL_LOSSLESS_LZ77_INDEX_CACHE");
      const bool cache = cache_env && std::strcmp(cache_env, "1") == 0;
      indexed_tasks = LosslessWorkerLimits::FromEnvironment().Tasks(
          kMatchBatch / 1024, cache ? 4 * window_size : 1u << 20);
      if (cache) {
        for (size_t task = 0; task < indexed_tasks; ++task) {
          indexed_caches.emplace_back(std::make_unique<LZ77MatchCache>(window_size));
        }
      }
      build.Set("tokens", in.size());
      build.Set("tasks", indexed_tasks);
      build.Set("cache_bytes", indexed_caches.size() * 4 * window_size);
    }
    if constexpr (kBucketCapacity == 0) {
      chain.maxchainlength = chain_limit;
      if (match_cache) chain.EnableMatchCache();
    }
    std::unique_ptr<HashChain<kMaxChainLength>> audit_chain;
    if constexpr (kBucketCapacity != 0) {
      const char* fingerprints = std::getenv("JXL_LOSSLESS_LZ77_FINGERPRINT");
      if (params.lossless_lz77_cost_spans && fingerprints &&
          std::strcmp(fingerprints, "1") == 0) {
        chain.EnableFingerprints(params.lossless_profile != 0);
      }
      const char* audit_env = std::getenv("JXL_LOSSLESS_LZ77_BUCKET_AUDIT");
      if (params.lossless_profile && audit_env && std::strcmp(audit_env, "1") == 0) {
        audit_chain = std::make_unique<HashChain<kMaxChainLength>>(
            in.data(), in.size(), window_size, min_length, max_length, distance_multiplier);
      }
    }

    // Total cost to encode the first N symbols.
    std::vector<MatchInfo> prefix_costs(in.size() + 1);
    prefix_costs[0].total_cost = 0;
    setup.Set("window", window_size);
    setup.Finish();
    LosslessProfileScope parse(params.lossless_profile, "lz77.match_parse");
    parse.Set("cost_spans", cost_spans);
    parse.Set("match_cache", match_cache);
    parse.Set("seed_tokens", tokens_for_cost_estimate[stream].size());
    parse.Set("match_cost_mode", match_cost_mode);
    parse.Set("refit", cost_seed != nullptr);
    parse.Set("bucket_capacity", kBucketCapacity);
    if constexpr (kBucketCapacity != 0) {
      parse.Set("dense_distances", chain.DenseDistances());
    }
    parse.Set("bucket_history", kHistory);
    parse.Set("full_prefix", kFullPrefix);
    parse.Set("long_capacity", kBucketCapacity ? kLongCapacity : 0);
    parse.Set("cost_bits", sizeof(Cost) * 8);
    parse.Set("compact_steps", kCompact);
    parse.Set("dominance", dominance && kBucketCapacity != 0);
    parse.Set("interval_dp", static_index ? interval_dp : 0);
    uint64_t match_calls = 0, length_candidates = 0, skipped = 0;
    LosslessProfileScope detail(detailed ? parse.Id() : 0, "lz77.parse_sample");
    LosslessProfileScope audit(audit_chain ? parse.Id() : 0, "lz77.bucket_audit");
    uint64_t audit_samples = 0, old_better = 0, new_better = 0;
    uint64_t evicted = 0, filtered = 0, missing_special = 0;
    uint64_t missing_short = 0, missing_medium = 0, missing_long = 0;
    uint32_t audit_rng = 0x85a308d3u + static_cast<uint32_t>(stream);
    size_t audit_calls = 0, audit_next = audit_rng & 127;
    std::vector<LZ77Match> audit_old, audit_new;
    using SampleClock = std::chrono::steady_clock;
    std::array<uint64_t, 5> sample_ns{}, sample_count{};
    uint32_t sample_rng = 0x243f6a88u + static_cast<uint32_t>(stream);
    size_t next_sample = sample_rng & 1023;
    SampleClock::time_point sample_start;
    const auto sample_stage = [&](bool sampled, size_t stage) {
      if (!sampled) return;
      const auto now = SampleClock::now();
      sample_ns[stage] += std::chrono::duration_cast<std::chrono::nanoseconds>(
                             now - sample_start).count();
      ++sample_count[stage];
      sample_start = now;
    };

    size_t rle_length = 0;
    size_t skip_lz77 = 0;
    size_t prev_match_end = 0;
    size_t prev_match_dist = 0;
    const auto update_skip = [&](size_t position, size_t length, uint32_t distance) {
      const size_t end = position + length;
      if (distance == 0 || (distance == 1 && distance_multiplier != 0) ||
          (length >= 256 && end == prev_match_end && distance == prev_match_dist)) {
        ++rle_length;
      } else {
        rle_length = 0;
      }
      prev_match_end = end;
      prev_match_dist = distance;
      if (rle_length >= 8 && length >= 9) {
        skip_lz77 = length - 9;
        rle_length = 0;
      }
    };
    for (size_t i = 0; i < in.size(); i++) {
      const bool sampled = detail.Enabled() && i == next_sample;
      if (sampled) {
        sample_rng = sample_rng * 1664525u + 1013904223u;
        next_sample = i + 512 + ((sample_rng >> 16) & 1023);
        sample_start = SampleClock::now();
      }
      if (!static_index) chain.Update(i);
      if constexpr (kBucketCapacity != 0) {
        if (audit_chain) audit_chain->Update(i);
      }
      sample_stage(sampled, 0);
      const Cost lit_cost =
          prefix_costs[i].total_cost + sym_cost[i + 1] - sym_cost[i];
      if (prefix_costs[i + 1].total_cost > lit_cost) {
        prefix_costs[i + 1].dist_symbol = 0;
        prefix_costs[i + 1].len = 1;
        if constexpr (!kCompact) prefix_costs[i + 1].ctx = in[i].context;
        prefix_costs[i + 1].total_cost = lit_cost;
      }
      sample_stage(sampled, 1);
      if (skip_lz77 > 0) {
        if (parse.Enabled()) ++skipped;
        skip_lz77--;
        continue;
      }
      if constexpr (kBucketCapacity != 0) {
        if (audit_chain && audit_calls++ == audit_next) {
          audit_rng = audit_rng * 1664525u + 1013904223u;
          audit_next += 64 + ((audit_rng >> 16) & 127);
          ++audit_samples;
          audit_old.clear();
          audit_new.clear();
          audit_chain->FindMatches(i, max_distance, [&](size_t len, uint32_t dist) {
            audit_old.push_back({static_cast<uint32_t>(len), dist});
          });
          chain.FindMatches(i, max_distance, [&](size_t len, uint32_t dist) {
            audit_new.push_back({static_cast<uint32_t>(len), dist});
          });
          const auto score = [&](const LZ77Match& match) {
            return sym_cost[i + match.length] - sym_cost[i] -
                (sce.LenCost(in[i].context, match.length - min_length, lz77) +
                 sce.DistCost(match.distance, lz77));
          };
          Cost old_score = 0, new_score = 0;
          LZ77Match best{};
          for (const auto& m : audit_old) {
            if (score(m) > old_score) { old_score = score(m); best = m; }
          }
          for (const auto& m : audit_new) new_score = std::max(new_score, score(m));
          if (new_score > old_score + 0.25f) ++new_better;
          if (old_score > new_score + 0.25f) {
            ++old_better;
            const uint32_t distance = distance_multiplier && best.distance < kNumSpecialDistances
                ? SpecialDistance(best.distance, distance_multiplier)
                : best.distance + 1 - (distance_multiplier ? kNumSpecialDistances : 0);
            const bool kept = chain.Retains(i, static_cast<uint32_t>(i - distance));
            if (kept) ++filtered;
            else ++evicted;
            missing_special += distance_multiplier && best.distance >= 24 &&
                               best.distance < kNumSpecialDistances;
            if (best.length < 8) ++missing_short;
            else if (best.length < 32) ++missing_medium;
            else ++missing_long;
          }
        }
      }
      dist_symbols.clear();
      const auto update_cost = [&](size_t j, uint32_t distance, Cost cost) {
        if (prefix_costs[i + j].total_cost > cost) {
          prefix_costs[i + j].len = j;
          prefix_costs[i + j].dist_symbol = distance + 1;
          if constexpr (!kCompact) prefix_costs[i + j].ctx = in[i].context;
          prefix_costs[i + j].total_cost = cost;
        }
      };
      bool used_intervals = false;
      size_t interval_longest = 0;
      uint32_t interval_distance = 0;
      if (parse.Enabled()) ++match_calls;
      if (match_cost_mode != 0) {
        dist_costs.clear();
        const auto match = [&](size_t len, uint32_t dist_symbol) {
          const float cost = sce.DistCost(dist_symbol, lz77);
          if (dist_symbols.size() <= len) {
            dist_symbols.resize(len + 1, dist_symbol);
            dist_costs.resize(len + 1, cost);
          }
          if (cost < dist_costs[len] ||
              (cost == dist_costs[len] && dist_symbol < dist_symbols[len])) {
            dist_symbols[len] = dist_symbol;
            dist_costs[len] = cost;
          }
        };
        if (match_cost_mode >= 2) {
          if (static_index) {
            if (i >= indexed_end) {
              indexed_begin = i;
              indexed_end = std::min(in.size(), i + kMatchBatch);
              const size_t count = indexed_end - indexed_begin;
              const size_t tasks = std::min(indexed_tasks, (count + 1023) / 1024);
              LosslessProfileScope batch(params.lossless_profile, "lz77.index_batch");
              const auto init = [&](size_t threads) -> Status {
                batch.Set("worker_slots", threads);
                return true;
              };
              const auto work = [&](uint32_t task, size_t) -> Status {
                const size_t begin = count * task / tasks;
                const size_t end = count * (task + 1) / tasks;
                for (size_t offset = begin; offset < end; ++offset) {
                  auto& frontier = indexed_matches[offset];
                  frontier.Reset();
                  static_index->FindMatches(indexed_begin + offset,
                      [&](uint32_t distance) { return sce.DistCost(distance, lz77); },
                      [&](size_t length, uint32_t distance) {
                        frontier.Add(static_cast<uint32_t>(length), distance,
                                     sce.DistCost(distance, lz77));
                      }, indexed_caches.empty() ? nullptr : indexed_caches[task].get());
                }
                return true;
              };
              JXL_RETURN_IF_ERROR(RunOnPool(params.lossless_lz77_pool, 0, tasks,
                                            init, work, "LosslessLZ77Indexed"));
              indexed_positions += count;
              batch.Set("positions", count);
              batch.Set("tasks", tasks);
            }
            ++indexed_replays;
            const auto& frontier = indexed_matches[i - indexed_begin];
            if (frontier.Overflowed()) {
              ++indexed_overflows;
              static_index->FindMatches(i,
                  [&](uint32_t distance) { return sce.DistCost(distance, lz77); }, match);
            } else if (interval_dp) {
              used_intervals = true;
              size_t begin = min_length;
              frontier.ForEach([&](size_t length, uint32_t distance) {
                VisitLZ77CostRange(min_length, begin, length + 1, distance,
                    sce.DistCost(distance, lz77), prefix_costs[i].total_cost,
                    [&](size_t value) { return sce.LenCost(in[i].context, value, lz77); },
                    update_cost, interval_dp == 2);
                begin = length + 1;
                interval_longest = length;
                interval_distance = distance;
              });
            } else {
              frontier.ForEach(match);
            }
          } else if constexpr (kBucketCapacity != 0) {
            if (dominance) {
              chain.FindCostMatches(i, max_distance,
                  [&](uint32_t symbol) { return sce.DistCost(symbol, lz77); }, match);
            } else {
              chain.template FindMatches<true>(i, max_distance, match);
            }
          } else {
            chain.template FindMatches<true>(i, max_distance, match);
          }
        } else {
          chain.FindMatches(i, max_distance, match);
        }
      } else {
        chain.FindMatches(i, max_distance,
                        [&dist_symbols](size_t len, size_t dist_symbol) {
                          if (dist_symbols.size() <= len) {
                            dist_symbols.resize(len + 1, dist_symbol);
                          }
                          if (dist_symbol < dist_symbols[len]) {
                            dist_symbols[len] = dist_symbol;
                          }
                        });
      }
      sample_stage(sampled, 2);
      if (used_intervals) {
        if (interval_longest < min_length) {
          rle_length = 0;
        } else {
          if (parse.Enabled()) length_candidates += interval_longest - min_length + 1;
          update_skip(i, interval_longest, interval_distance);
        }
        continue;
      }
      if (dist_symbols.size() <= min_length) {
        rle_length = 0;
        continue;
      }
      if (match_cost_mode != 0) {
        // A longer match also covers each shorter endpoint. Keep the cheapest
        // distance under the current entropy model, not the lowest code ID.
        uint32_t best_distance = dist_symbols.back();
        float best_cost = dist_costs.back();
        for (size_t j = dist_symbols.size() - 1; j >= min_length; --j) {
          if (dist_costs[j] < best_cost ||
              (dist_costs[j] == best_cost && dist_symbols[j] < best_distance)) {
            best_distance = dist_symbols[j];
            best_cost = dist_costs[j];
          }
          dist_symbols[j] = best_distance;
        }
      } else {
        size_t best_cost = dist_symbols.back();
        for (size_t j = dist_symbols.size() - 1; j >= min_length; j--) {
          if (dist_symbols[j] < best_cost) {
            best_cost = dist_symbols[j];
          }
          dist_symbols[j] = best_cost;
        }
      }
      sample_stage(sampled, 3);
      if (cost_spans) {
        VisitLZ77CostSpans(
            min_length, dist_symbols, prefix_costs[i].total_cost,
            [&](size_t len) { return sce.LenCost(in[i].context, len, lz77); },
            [&](uint32_t dist) { return sce.DistCost(dist, lz77); }, update_cost);
      } else {
        for (size_t j = min_length; j < dist_symbols.size(); j++) {
          // Cost model that uses results from lazy LZ77.
          float lz77_cost = sce.LenCost(in[i].context, j - min_length, lz77) +
                            sce.DistCost(dist_symbols[j], lz77);
          update_cost(j, dist_symbols[j], prefix_costs[i].total_cost + lz77_cost);
        }
      }
      sample_stage(sampled, 4);
      if (parse.Enabled()) length_candidates += dist_symbols.size() - min_length;
      // We are in a RLE or long repeating sequence: skip all the symbols except
      // the first 8 and the last 8. This avoids quadratic costs for sequences
      // with long runs of repeating patterns.
      update_skip(i, dist_symbols.size() - 1, dist_symbols.back());
    }
    parse.Set("match_calls", match_calls);
    parse.Set("length_candidates", length_candidates);
    parse.Set("skipped_positions", skipped);
    parse.Set("indexed_positions", indexed_positions);
    parse.Set("indexed_replays", indexed_replays);
    parse.Set("indexed_overflows", indexed_overflows);
    if constexpr (kBucketCapacity != 0) {
      parse.Set("fingerprint_scanned", chain.FingerprintScanned());
      parse.Set("fingerprint_retained", chain.FingerprintRetained());
      parse.Set("fingerprint_simd_queries", chain.FingerprintSIMDQueries());
      parse.Set("fingerprint_uniform_queries", chain.FingerprintUniformQueries());
    }
    audit.Set("samples", audit_samples);
    audit.Set("old_better", old_better);
    audit.Set("new_better", new_better);
    audit.Set("evicted", evicted);
    audit.Set("filtered", filtered);
    audit.Set("special_24_119", missing_special);
    audit.Set("length_lt8", missing_short);
    audit.Set("length_8_31", missing_medium);
    audit.Set("length_ge32", missing_long);
    audit.Finish();
    detail.Set("hash_count", sample_count[0]);
    detail.Set("hash_ns", sample_ns[0]);
    detail.Set("literal_count", sample_count[1]);
    detail.Set("literal_ns", sample_ns[1]);
    detail.Set("find_count", sample_count[2]);
    detail.Set("find_ns", sample_ns[2]);
    detail.Set("prepare_count", sample_count[3]);
    detail.Set("prepare_ns", sample_ns[3]);
    detail.Set("dp_count", sample_count[4]);
    detail.Set("dp_ns", sample_ns[4]);
    detail.Set("mean_stride", 1024);
    detail.Finish();
    parse.Finish();
    LosslessProfileScope emit(params.lossless_profile, "lz77.backtrack");
    size_t pos = in.size();
    while (pos > 0) {
      bool is_lz77_length = prefix_costs[pos].dist_symbol != 0;
      if (is_lz77_length) {
        size_t dist_symbol = prefix_costs[pos].dist_symbol - 1;
        out.emplace_back(
            static_cast<uint32_t>(lz77.nonserialized_distance_context),
            static_cast<uint32_t>(dist_symbol));
      }
      uint32_t val =
          is_lz77_length
              ? (prefix_costs[pos].len - static_cast<uint32_t>(min_length))
              : in[pos - 1].value;
      uint32_t context;
      if constexpr (kCompact) {
        context = in[pos - prefix_costs[pos].len].context;
      } else {
        context = prefix_costs[pos].ctx;
      }
      out.emplace_back(context, val);
      out.back().is_lz77_length = is_lz77_length;
      pos -= prefix_costs[pos].len;
    }
    if (!out.empty()) std::reverse(out.begin(), out.end());
    emit.Set("tokens", out.size());
    return true;
  };
  const char* arena_env = std::getenv("JXL_LOSSLESS_LZ77_STREAM_ARENA");
  const bool cooperative_streams = indexed && !reuse_index &&
      params.lossless_lz77_pool && order.size() > 1 && arena_env &&
      std::strcmp(arena_env, "1") == 0;
  if (cooperative_streams) {
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      return tokens[a].size() > tokens[b].size();
    });
    LosslessProbePool arena;
    const auto limits = LosslessWorkerLimits::FromEnvironment();
    const size_t roots = limits.Tasks(order.size(), scratch);
    dispatch.Set("cooperative_streams", roots);
    JXL_RETURN_IF_ERROR(arena.Run(order.size(), params.lossless_lz77_pool,
        limits.tasks, roots, [&](size_t i, ThreadPool* nested) -> Status {
          auto child_params = params;
          child_params.lossless_lz77_pool = nested;
          return process_stream(order[i], child_params);
        }));
  } else if (workers > 1) {
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      return tokens[a].size() > tokens[b].size();
    });
    std::atomic<size_t> next{0};
    const auto init = [&](size_t threads) -> Status {
      dispatch.Set("worker_slots", threads);
      return true;
    };
    const auto work = [&](uint32_t, size_t) -> Status {
      for (;;) {
        const size_t index = next.fetch_add(1, std::memory_order_relaxed);
        if (index >= order.size()) return true;
        JXL_RETURN_IF_ERROR(process_stream(order[index], params));
      }
    };
    JXL_RETURN_IF_ERROR(RunOnPool(params.lossless_lz77_pool, 0, workers, init,
                                  work, "LosslessLZ77Streams"));
  } else {
    dispatch.Set("worker_slots", 1);
    for (size_t stream : order) JXL_RETURN_IF_ERROR(process_stream(stream, params));
  }
  dispatch.Finish();
  if (match_cost_mode == 3 && !cost_seed) {
    // One bounded model refit, never an unbounded fixed-point search. This is
    // an opt-in size experiment; its second parse may still produce more bits.
    return ApplyLZ77_Optimal<kMaxChainLength, kBucketCapacity, kHistory, kFullPrefix, kLongCapacity, Cost, kCompact>(
        params, num_contexts, tokens, lz77, &tokens_lz77);
  }
  return tokens_lz77;
}

}  // namespace

StatusOr<std::vector<std::vector<Token>>> ApplyLZ77(
    const HistogramParams& params, size_t num_contexts,
    const std::vector<std::vector<Token>>& tokens, const LZ77Params& lz77) {
  LosslessProfileScope profile(params.lossless_profile, "lz77.total");
  profile.Set("method", static_cast<uint64_t>(params.lz77_method));
  profile.Set("contexts", num_contexts);
  if (profile.Enabled()) {
    size_t count = 0;
    for (const auto& stream : tokens) count += stream.size();
    profile.Set("tokens", count);
  }
  switch (params.lz77_method) {
    case HistogramParams::LZ77Method::kRLE:
      return ApplyLZ77_RLE(params, num_contexts, tokens, lz77);
    case HistogramParams::LZ77Method::kLZ77b1w3f:
      return ApplyLZ77_LZ77<1, 3, false>(params, num_contexts, tokens, lz77);
    case HistogramParams::LZ77Method::kLZ77b3w3f:
      return ApplyLZ77_LZ77<3, 3, false>(params, num_contexts, tokens, lz77);
    case HistogramParams::LZ77Method::kLZ77b7w3f:
      return ApplyLZ77_LZ77<7, 3, false>(params, num_contexts, tokens, lz77);
    case HistogramParams::LZ77Method::kLZ77b15w3f:
      return ApplyLZ77_LZ77<15, 3, false>(params, num_contexts, tokens, lz77);
    case HistogramParams::LZ77Method::kLZ77b31w3f:
      return ApplyLZ77_LZ77<31, 3, false>(params, num_contexts, tokens, lz77);
    case HistogramParams::LZ77Method::kLZ77b1w3t:
      return ApplyLZ77_LZ77<1, 3, true>(params, num_contexts, tokens, lz77);
    case HistogramParams::LZ77Method::kLZ77b3w3t:
      return ApplyLZ77_LZ77<3, 3, true>(params, num_contexts, tokens, lz77);
    case HistogramParams::LZ77Method::kLZ77b7w3t:
      return ApplyLZ77_LZ77<7, 3, true>(params, num_contexts, tokens, lz77);
    case HistogramParams::LZ77Method::kLZ77b15w3t:
      return ApplyLZ77_LZ77<15, 3, true>(params, num_contexts, tokens, lz77);
    case HistogramParams::LZ77Method::kLZ77b31w3t:
      return ApplyLZ77_LZ77<31, 3, true>(params, num_contexts, tokens, lz77);
    case HistogramParams::LZ77Method::kOptc1:
      return ApplyLZ77_Optimal<1>(params, num_contexts, tokens, lz77);
    case HistogramParams::LZ77Method::kOptc3:
      return ApplyLZ77_Optimal<3>(params, num_contexts, tokens, lz77);
    case HistogramParams::LZ77Method::kOptc8:
      return ApplyLZ77_Optimal<8>(params, num_contexts, tokens, lz77);
    case HistogramParams::LZ77Method::kOptc256: {
      const char* mode = std::getenv("JXL_LOSSLESS_LZ77_BUCKETS");
      if (params.lossless_lz77_cost_spans && params.lossless_bucket_matcher &&
          !params.direct_audit) {
        return ApplyLZ77_Optimal<256, 32, false, false, 8, float, true>(
            params, num_contexts, tokens, lz77);
      }
      if (params.lossless_lz77_cost_spans && mode) {
        if (!params.direct_audit && std::strcmp(mode, "parallel32") == 0) {
          return ApplyLZ77_Optimal<256, 32, false, false, 8, float, true>(
              params, num_contexts, tokens, lz77);
        }
        if (!params.direct_audit && std::strcmp(mode, "parallel-wide") == 0) {
          return ApplyLZ77_Optimal<256, 32, false, true, 32, float, true>(
              params, num_contexts, tokens, lz77);
        }
        if (std::strcmp(mode, "precise-ring") == 0) {
          return ApplyLZ77_Optimal<256, 32, false, true, 32, double>(
              params, num_contexts, tokens, lz77);
        }
        if (std::strcmp(mode, "wide-ring") == 0) {
          return ApplyLZ77_Optimal<256, 32, false, true, 32>(params, num_contexts, tokens, lz77);
        }
        if (std::strcmp(mode, "wide-key") == 0) {
          return ApplyLZ77_Optimal<256, 32, true, true>(params, num_contexts, tokens, lz77);
        }
        if (std::strcmp(mode, "history") == 0) {
          return ApplyLZ77_Optimal<256, 32, true>(params, num_contexts, tokens, lz77);
        }
        if (std::strcmp(mode, "32") == 0) {
          return ApplyLZ77_Optimal<256, 32>(params, num_contexts, tokens, lz77);
        }
        if (std::strcmp(mode, "128") == 0) {
          return ApplyLZ77_Optimal<256, 128>(params, num_contexts, tokens, lz77);
        }
      }
      const char* compact = std::getenv("JXL_LOSSLESS_LZ77_COMPACT");
      const bool use_compact = params.lossless_lz77_cost_spans && compact &&
          std::strcmp(compact, "1") == 0;
      if (use_compact) {
        return ApplyLZ77_Optimal<256, 0, false, false, 8, float, true>(
            params, num_contexts, tokens, lz77);
      }
      return ApplyLZ77_Optimal<256>(params, num_contexts, tokens, lz77);
    }
    default:
      return std::vector<std::vector<Token>>{};
  }
}

}  // namespace jxl
