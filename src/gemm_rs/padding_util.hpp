//===- padding_util.hpp --------------------------------------- C++ ---===//
//
// Copyright 2025 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#pragma once
#include <assert.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "flux/flux.h"

namespace bytedance {
namespace flux {

// pad m-dim of the input tensor to be multiple of TPxtile_size
void pad_m_to_TPxTile(
    void const *input,
    void const *scale,  // shape: (m_size, 1)
    void *output,
    void *padded_scale,
    int m_size,
    int n_size,
    int tp_size,
    int tile_size,
    DataTypeEnum input_dtype,
    DataTypeEnum scale_dtype,
    cudaStream_t stream);
}  // namespace flux
}  // namespace bytedance
