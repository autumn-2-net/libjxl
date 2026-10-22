// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license.
#ifndef LIB_JXL_ENC_FINAL_PLAN_H_
#define LIB_JXL_ENC_FINAL_PLAN_H_

#include <cstdint>
#include <string>
#include <vector>

#include "lib/jxl/base/status.h"

namespace jxl {
struct CompressParams;

// Experimental, encoder-only decision transcript, never part of the codestream.
// Capture/replay runs synchronously with a null ThreadPool. Ordinary encodes
// never install a transcript. Records contain decisions, not source pixels.
struct FinalPlanRecord {
  std::string name;
  std::vector<int64_t> values;
};

class FinalPlan {
 public:
  bool replay = false;
  size_t cursor = 0;
  size_t tree_limit = 0;
  std::string unsupported;
  std::vector<FinalPlanRecord> records;
  Status Exchange(const char* name, std::vector<int64_t>* values);
  Status Read(const char* path);
  Status Write(const char* path) const;
  Status Finished() const;
};

FinalPlan* CurrentFinalPlan();
inline bool ReplayingFinalPlan() {
  return CurrentFinalPlan() && CurrentFinalPlan()->replay;
}
class FinalPlanScope {
 public:
  explicit FinalPlanScope(FinalPlan* plan);
  ~FinalPlanScope();
  FinalPlanScope(const FinalPlanScope&) = delete;
  FinalPlanScope& operator=(const FinalPlanScope&) = delete;

 private:
  FinalPlan* previous_;
};

Status FinalPlanExchange(const char* name, std::vector<int64_t>* values);
Status FinalPlanParameters(CompressParams* params);
const char* FinalPlanOutput();
const char* FinalPlanInput();
}  // namespace jxl
#endif
