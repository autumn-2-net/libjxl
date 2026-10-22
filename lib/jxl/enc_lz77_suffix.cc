// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "lib/jxl/enc_lz77_suffix.h"

#ifdef JXL_EXPERIMENTAL_LIBSAIS
#include "libsais.h"
#endif

namespace jxl {
bool LosslessSuffixArrayAvailable() {
#ifdef JXL_EXPERIMENTAL_LIBSAIS
  return true;
#else
  return false;
#endif
}
int BuildLosslessSuffixArray(int32_t* values, int32_t* suffixes, int32_t size,
                            int32_t alphabet, int32_t scratch) {
#ifdef JXL_EXPERIMENTAL_LIBSAIS
  return libsais_int(values, suffixes, size, alphabet, scratch);
#else
  (void)values; (void)suffixes; (void)size; (void)alphabet; (void)scratch;
  return -1;
#endif
}
}  // namespace jxl
