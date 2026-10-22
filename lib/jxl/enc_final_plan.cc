// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license.
#include "lib/jxl/enc_final_plan.h"

#include <cstdlib>
#include <fstream>
#include <locale>

#include "lib/jxl/enc_params.h"

namespace jxl {
namespace {
thread_local FinalPlan* current = nullptr;
constexpr size_t kMaxValues = 16000000;
const char* Path(const char* key) {
  const char* value = std::getenv(key);
  return value && *value ? value : nullptr;
}
}  // namespace

const char* FinalPlanOutput() { return Path("JXL_FINAL_PLAN_OUT"); }
const char* FinalPlanInput() { return Path("JXL_FINAL_PLAN_IN"); }
FinalPlan* CurrentFinalPlan() { return current; }
FinalPlanScope::FinalPlanScope(FinalPlan* plan) : previous_(current) {
  current = plan;
}
FinalPlanScope::~FinalPlanScope() { current = previous_; }

Status FinalPlan::Exchange(const char* name, std::vector<int64_t>* values) {
  if (!replay) {
    records.push_back({name, *values});
    return true;
  }
  if (cursor >= records.size() || records[cursor].name != name) {
    return JXL_FAILURE("Final plan expected %s at record %zu", name, cursor);
  }
  *values = records[cursor++].values;
  return true;
}
Status FinalPlanExchange(const char* name, std::vector<int64_t>* values) {
  return current ? current->Exchange(name, values) : Status(true);
}
Status FinalPlan::Finished() const {
  JXL_ENSURE(!replay || cursor == records.size());
  return true;
}
Status FinalPlan::Read(const char* path) {
  std::ifstream in(path);
  in.imbue(std::locale::classic());
  std::string magic;
  size_t count;
  JXL_ENSURE(in >> magic >> count);
  JXL_ENSURE(magic == "JXL_FINAL_PLAN_V1" && count > 0 && count <= 1000000);
  records.clear();
  size_t total = 0;
  for (size_t i = 0; i < count; ++i) {
    FinalPlanRecord record;
    size_t n;
    JXL_ENSURE(in >> record.name >> n);
    JXL_ENSURE(record.name.size() <= 64 && n <= kMaxValues - total);
    total += n;
    record.values.resize(n);
    for (auto& v : record.values) JXL_ENSURE(in >> v);
    records.push_back(std::move(record));
  }
  in >> std::ws;
  JXL_ENSURE(in.eof());
  cursor = 0;
  replay = true;
  return true;
}
Status FinalPlan::Write(const char* path) const {
  std::ofstream out(path, std::ios::trunc);
  out.imbue(std::locale::classic());
  JXL_ENSURE(out.good());
  out << "JXL_FINAL_PLAN_V1 " << records.size() << '\n';
  for (const auto& r : records) {
    out << r.name << ' ' << r.values.size();
    for (auto v : r.values) out << ' ' << v;
    out << '\n';
  }
  out.close();
  JXL_ENSURE(!out.fail());
  return true;
}

Status FinalPlanParameters(CompressParams* p) {
  // These retain the winner's framing/scaffolding. Search outcomes are separate
  // records below this header; replay does not infer them from an effort level.
  std::vector<int64_t> v = {static_cast<int>(p->speed_tier),
                            p->modular_group_size_shift,
                            p->colorspace,
                            p->responsive,
                            static_cast<int>(p->patches),
                            p->palette_colors,
                            static_cast<int>(p->options.predictor),
                            p->options.max_properties,
                            p->force_cfl_jpeg_recompression,
                            p->force_lfs_jpeg_recompression,
                            p->lz77_careful_first_pass,
                            static_cast<int>(p->keep_invisible)};
  JXL_RETURN_IF_ERROR(FinalPlanExchange("frame", &v));
  JXL_ENSURE(v.size() == 12 && v[0] >= 0 && v[0] <= 9 && v[1] >= -1 &&
             v[1] <= 3 && v[2] >= -1 && v[2] < 42 && v[3] >= -1 && v[3] <= 1 &&
             v[4] >= -1 && v[4] <= 1 && v[5] >= -65536 && v[5] <= 70000 &&
             v[6] >= -1 && v[6] <= 15 && v[7] >= -1 && v[7] <= 7 && v[8] >= 0 &&
             v[8] <= 1 && v[9] >= -1 && v[9] <= 0 && v[10] >= 0 && v[10] <= 1 &&
             v[11] == static_cast<int>(p->keep_invisible));
  if (ReplayingFinalPlan()) {
    p->speed_tier = static_cast<SpeedTier>(v[0]);
    p->modular_group_size_shift = v[1];
    p->colorspace = v[2];
    p->responsive = v[3];
    p->patches = static_cast<Override>(v[4]);
    p->palette_colors = v[5];
    p->options.predictor = static_cast<Predictor>(v[6]);
    p->options.max_properties = v[7];
    p->force_cfl_jpeg_recompression = v[8];
    p->force_lfs_jpeg_recompression = v[9];
    p->lz77_careful_first_pass = v[10];
    p->keep_invisible = static_cast<Override>(v[11]);
  }
  return true;
}
}  // namespace jxl
