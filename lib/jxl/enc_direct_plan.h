// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license.
#ifndef LIB_JXL_ENC_DIRECT_PLAN_H_
#define LIB_JXL_ENC_DIRECT_PLAN_H_

#include "lib/jxl/enc_lossless_search.h"
#include "lib/jxl/enc_tool_policy.h"

namespace jxl {
constexpr size_t kDirectPlanCount = 9072;
bool DirectPlanRequested();
// Model inference and collection live inside the encoder, before transforms.
Status EncodeDirectPlan(JxlMemoryManager* memory_manager,
                        const CompressParams& params,
                        const CodecMetadata& metadata,
                        JxlEncoderChunkedFrameAdapter& input, ThreadPool* pool,
                        JxlEncoderOutputProcessorWrapper* output, AuxOut* aux,
                        const LosslessCandidateEncoder& encode);
}  // namespace jxl
#endif
