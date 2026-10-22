// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_LOSSLESS_PROFILE_H_
#define LIB_JXL_ENC_LOSSLESS_PROFILE_H_

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace jxl {

inline uint64_t LosslessProfileNextId() {
  static std::atomic<uint64_t> next{1};
  return next.fetch_add(1, std::memory_order_relaxed);
}

struct LosslessProfileThread {
  uint64_t current = 0;
  uint64_t id = 0;
};

inline LosslessProfileThread& LosslessProfileThreadState() {
  static thread_local LosslessProfileThread state;
  return state;
}

inline uint64_t LosslessProfileCurrent() {
  return LosslessProfileThreadState().current;
}

inline bool LosslessProfileRequested() {
  const char* value = std::getenv("JXL_LOSSLESS_PROFILE");
  return value && std::strcmp(value, "1") == 0;
}

// Opt-in, encoder-only spans. Carried context IDs cross runner boundaries;
// thread-local nesting associates synchronous children with their parent.
// Durations are wall intervals, NOT CPU time. Overlapping spans must not be
// added as wall time, and parent durations already include their children.
class LosslessProfileScope {
 public:
  LosslessProfileScope(uint64_t context, const char* phase, bool root = false)
      : phase_(phase) {
    if (context == 0 && !root) return;
    auto& thread = LosslessProfileThreadState();
    if (thread.id == 0) thread.id = LosslessProfileNextId();
    thread_ = thread.id;
    previous_ = thread.current;
    parent_ = previous_ != 0 ? previous_ : context;
    id_ = LosslessProfileNextId();
    thread.current = id_;
    start_ = Now();
    active_ = true;
    Emit("begin", start_);
  }

  LosslessProfileScope(const LosslessProfileScope&) = delete;
  LosslessProfileScope& operator=(const LosslessProfileScope&) = delete;
  ~LosslessProfileScope() { Finish(); }

  uint64_t Id() const { return id_; }
  bool Enabled() const { return id_ != 0; }

  void Set(const char* key, uint64_t value) {
    if (!active_) return;
    for (size_t i = 0; i < num_metrics_; ++i) {
      if (std::strcmp(metrics_[i].key, key) == 0) {
        metrics_[i].value = value;
        return;
      }
    }
    if (num_metrics_ < kMaxMetrics) metrics_[num_metrics_++] = {key, value};
  }

  void Finish() {
    if (!active_) return;
    const uint64_t end = Now();
    LosslessProfileThreadState().current = previous_;
    active_ = false;
    Emit("end", end);
  }

 private:
  static uint64_t Now() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
  }

  void Emit(const char* event, uint64_t at) const {
    char line[2048];
    const int first = std::snprintf(
        line, sizeof(line),
        "lossless-profile schema=1 event=%s id=%" PRIu64 " parent=%" PRIu64
        " thread=%" PRIu64 " phase=%s start_ns=%" PRIu64 " at_ns=%" PRIu64
        " elapsed_ns=%" PRIu64,
        event, id_, parent_, thread_, phase_, start_, at, at - start_);
    if (first < 0 || static_cast<size_t>(first) >= sizeof(line) - 1) return;
    size_t used = static_cast<size_t>(first);
    for (size_t i = 0; i < num_metrics_; ++i) {
      const int n =
          std::snprintf(line + used, sizeof(line) - used, " %s=%" PRIu64,
                        metrics_[i].key, metrics_[i].value);
      if (n < 0 || static_cast<size_t>(n) >= sizeof(line) - used - 1) break;
      used += static_cast<size_t>(n);
    }
    line[used++] = '\n';
    // One stdio write per record prevents partial-line interleaving.
    std::fwrite(line, 1, used, stderr);
  }

  struct Metric {
    const char* key;
    uint64_t value;
  };
  static constexpr size_t kMaxMetrics = 48;
  Metric metrics_[kMaxMetrics];
  size_t num_metrics_ = 0;
  const char* phase_;
  uint64_t id_ = 0, parent_ = 0, previous_ = 0, thread_ = 0, start_ = 0;
  bool active_ = false;
};

}  // namespace jxl

#endif  // LIB_JXL_ENC_LOSSLESS_PROFILE_H_
