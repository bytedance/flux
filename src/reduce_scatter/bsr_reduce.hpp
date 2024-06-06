//===- bsr_reduce.hpp --------------------------------------------- C++ ---===//
//
// Copyright 2023 ByteDance Ltd. and/or its affiliates. All rights reserved.
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
#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include <vector>
#include <assert.h>
#include "flux/flux.h"
#include <stdio.h>
#define FETCH_128bit(pointer) (reinterpret_cast<float4 *>(pointer))[0]

namespace bytedance {
namespace flux {

void bsr2dense_reduce(
    void *input,
    void *output,
    std::vector<int> shape,
    int block_h,
    int block_w,
    DataTypeEnum dtype,
    cudaStream_t stream);

}
}  // namespace bytedance
