// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef LIB_JXL_MODULAR_TRANSFORM_ENC_PALETTE_ORDER_H_
#define LIB_JXL_MODULAR_TRANSFORM_ENC_PALETTE_ORDER_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <numeric>
#include <vector>

namespace jxl {

// Project complete color tuples on their principal variance direction. The
// projection chooses only a permutation; original channel values stay intact.
inline std::vector<size_t> PalettePrincipalOrder(
    const std::vector<std::array<int32_t, 4>>& colors,
    const std::vector<size_t>& frequencies, size_t channels, bool banded) {
  const size_t n = colors.size();
  std::vector<size_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  if (n < 2 || channels == 0 || channels > 4 || frequencies.size() != n)
    return order;
  std::array<double, 4> mean{}, axis{};
  std::array<std::array<double, 4>, 4> covariance{};
  for (const auto& color : colors)
    for (size_t c = 0; c < channels; ++c) mean[c] += color[c];
  for (size_t c = 0; c < channels; ++c) mean[c] /= n;
  for (const auto& color : colors)
    for (size_t c = 0; c < channels; ++c)
      for (size_t d = 0; d < channels; ++d)
        covariance[c][d] += (color[c] - mean[c]) * (color[d] - mean[d]);
  size_t widest = 0;
  for (size_t c = 1; c < channels; ++c)
    if (covariance[c][c] > covariance[widest][widest]) widest = c;
  axis[widest] = 1;
  for (size_t iteration = 0; iteration < 12; ++iteration) {
    std::array<double, 4> next{};
    double norm = 0;
    for (size_t c = 0; c < channels; ++c) {
      for (size_t d = 0; d < channels; ++d)
        next[c] += covariance[c][d] * axis[d];
      norm += next[c] * next[c];
    }
    if (norm == 0) break;
    const double scale = 1 / std::sqrt(norm);
    for (size_t c = 0; c < channels; ++c) axis[c] = next[c] * scale;
  }
  std::vector<double> projection(n);
  std::vector<size_t> bands(n);
  for (size_t i = 0; i < n; ++i) {
    for (size_t c = 0; c < channels; ++c)
      projection[i] += (colors[i][c] - mean[c]) * axis[c];
    if (banded)
      for (size_t frequency = frequencies[i]; frequency >= 4; frequency >>= 2)
        ++bands[i];
  }
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    if (bands[a] != bands[b]) return bands[a] > bands[b];
    if (projection[a] != projection[b]) return projection[a] < projection[b];
    if (colors[a] != colors[b]) return colors[a] < colors[b];
    return a < b;
  });
  return order;
}

// Maximum-weight degree-two forest: frequently adjacent colors become short
// chains without introducing cycles. A chain is emitted from its more frequent
// endpoint. This is a palette permutation, not a new bitstream transform.
inline std::vector<size_t> PaletteAdjacencyOrder(
    const std::vector<size_t>& frequencies, std::vector<uint64_t> pairs) {
  const size_t n = frequencies.size();
  struct Edge { uint32_t a, b; size_t weight; };
  std::sort(pairs.begin(), pairs.end());
  std::vector<Edge> edges;
  for (size_t begin = 0; begin < pairs.size();) {
    size_t end = begin + 1;
    while (end < pairs.size() && pairs[end] == pairs[begin]) ++end;
    const uint32_t a = pairs[begin] >> 32, b = uint32_t(pairs[begin]);
    if (a < n && b < n && a != b) edges.push_back({a, b, end - begin});
    begin = end;
  }
  std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
    if (a.weight != b.weight) return a.weight > b.weight;
    if (a.a != b.a) return a.a < b.a;
    return a.b < b.b;
  });
  std::vector<size_t> parent(n), sizes(n, 1);
  std::iota(parent.begin(), parent.end(), 0);
  const auto root = [&](size_t node) {
    while (parent[node] != node) {
      parent[node] = parent[parent[node]];
      node = parent[node];
    }
    return node;
  };
  std::vector<std::array<size_t, 2>> neighbors(n, {n, n});
  std::vector<uint8_t> degree(n, 0), visited(n, 0);
  for (const auto& edge : edges) {
    if (degree[edge.a] == 2 || degree[edge.b] == 2) continue;
    size_t a = root(edge.a), b = root(edge.b);
    if (a == b) continue;
    neighbors[edge.a][degree[edge.a]++] = edge.b;
    neighbors[edge.b][degree[edge.b]++] = edge.a;
    if (sizes[a] < sizes[b]) std::swap(a, b);
    parent[b] = a;
    sizes[a] += sizes[b];
  }
  std::vector<size_t> endpoints;
  for (size_t i = 0; i < n; ++i) if (degree[i] < 2) endpoints.push_back(i);
  std::stable_sort(endpoints.begin(), endpoints.end(), [&](size_t a, size_t b) {
    return frequencies[a] > frequencies[b];
  });
  std::vector<size_t> order;
  order.reserve(n);
  for (size_t endpoint : endpoints) {
    size_t node = endpoint;
    while (node < n && !visited[node]) {
      order.push_back(node);
      visited[node] = 1;
      const size_t next = neighbors[node][0];
      node = next < n && !visited[next] ? next : neighbors[node][1];
    }
  }
  return order;
}

// Skilling's axes-to-transpose construction (public-domain reference code):
// https://doi.org/10.1063/1.1751381 . Keys retain every coordinate bit.
inline std::array<uint32_t, 4> PaletteHilbertKey(
    std::array<uint32_t, 4> axes, size_t dimensions, unsigned bits) {
  if (dimensions < 2 || dimensions > axes.size() || bits == 0 || bits > 32)
    return axes;
  const uint32_t highest = uint32_t{1} << (bits - 1);
  for (uint32_t mask = highest; mask > 1; mask >>= 1) {
    const uint32_t low = mask - 1;
    for (size_t c = 0; c < dimensions; ++c) {
      if (axes[c] & mask) {
        axes[0] ^= low;
      } else {
        const uint32_t exchange = (axes[0] ^ axes[c]) & low;
        axes[0] ^= exchange;
        axes[c] ^= exchange;
      }
    }
  }
  for (size_t c = 1; c < dimensions; ++c) axes[c] ^= axes[c - 1];
  uint32_t correction = 0;
  for (uint32_t mask = highest; mask > 1; mask >>= 1)
    if (axes[dimensions - 1] & mask) correction ^= mask - 1;
  for (size_t c = 0; c < dimensions; ++c) axes[c] ^= correction;
  return axes;
}

inline bool PaletteInterleavedLess(const std::array<uint32_t, 4>& a,
                                    const std::array<uint32_t, 4>& b,
                                    size_t dimensions) {
  uint32_t highest = 0;
  size_t channel = 0;
  for (size_t c = 0; c < dimensions; ++c) {
    const uint32_t difference = a[c] ^ b[c];
    // A strictly higher differing bit determines the interleaved order.
    if (difference > highest && (difference ^ highest) > highest) {
      highest = difference;
      channel = c;
    }
  }
  return highest != 0 && a[channel] < b[channel];
}

}  // namespace jxl

#endif  // LIB_JXL_MODULAR_TRANSFORM_ENC_PALETTE_ORDER_H_
