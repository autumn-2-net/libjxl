// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license.
#ifndef LIB_JXL_ENC_DIRECT_AUDIT_H_
#define LIB_JXL_ENC_DIRECT_AUDIT_H_

#include <atomic>
#include <cstddef>

namespace jxl {
// One frame owns this state. Worker tasks share counters, never policy globals.
struct DirectAudit {
  std::atomic<size_t> frames{0}, trees{0}, entropy_calls{0}, maps{0};
  std::atomic<size_t> palette_attempts{0}, palette_applied{0};
  std::atomic<size_t> histogram_calls{0}, precision_trials{0};
  std::atomic<size_t> forbidden_searches{0};
  bool allow_ma_learning = false;
  std::atomic<size_t> ma_learning_calls{0}, tree_nodes{0};
};
}  // namespace jxl
#endif
