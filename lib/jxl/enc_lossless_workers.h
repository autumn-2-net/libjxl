// Copyright (c) the JPEG XL Project Authors.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef LIB_JXL_ENC_LOSSLESS_WORKERS_H_
#define LIB_JXL_ENC_LOSSLESS_WORKERS_H_

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <system_error>

namespace jxl {

inline size_t LosslessWorkerSetting(const char* text, size_t fallback,
                                    size_t maximum) {
  if (!text || !*text) return fallback;
  size_t value = 0;
  const char* end = text + std::strlen(text);
  const auto parsed = std::from_chars(text, end, value);
  return parsed.ec == std::errc{} && parsed.ptr == end && value > 0 &&
                 value <= maximum
             ? value : fallback;
}

// Limits independent scratch owners, not the runner's thread count. The
// supplied runner remains the only executor and is never entered recursively.
struct LosslessWorkerLimits {
  size_t tasks = 8;
  uint64_t scratch_bytes = uint64_t{1} << 30;

  static LosslessWorkerLimits FromEnvironment() {
    return {LosslessWorkerSetting(
                std::getenv("JXL_LOSSLESS_WORKER_LIMIT"), 8, 64),
            uint64_t{LosslessWorkerSetting(
                std::getenv("JXL_LOSSLESS_WORKER_MEMORY_MIB"), 1024, 8192)}
                << 20};
  }

  size_t Tasks(size_t jobs, uint64_t scratch_per_task) const {
    // A single oversized job still has to run; this only bounds concurrency.
    const uint64_t memory_slots = scratch_per_task
        ? std::max<uint64_t>(1, scratch_bytes / scratch_per_task) : tasks;
    return static_cast<size_t>(
        std::min({uint64_t{jobs}, uint64_t{tasks}, memory_slots}));
  }
};

}  // namespace jxl
#endif
