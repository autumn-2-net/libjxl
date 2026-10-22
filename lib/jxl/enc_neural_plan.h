// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license.
#ifndef LIB_JXL_ENC_NEURAL_PLAN_H_
#define LIB_JXL_ENC_NEURAL_PLAN_H_

#include <string>

#include "lib/jxl/enc_params.h"

namespace jxl {
// Experimental encoder-only protocol. This is not a codestream extension.
Status ReadNeuralPlan(const std::string& text, bool jpeg, CompressParams* params);
Status ReadNeuralTreeFile(const char* path, Tree* tree);
Status WriteNeuralTreeFile(const char* path, const Tree& tree);
}  // namespace jxl
#endif
