//===- topk_gather_rs.hpp --------------------------------------------- C++ ---===//
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
#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include <vector>
#include <assert.h>
#include "flux/flux.h"
#include <stdio.h>
#include "flux/args/moe_gather_rs.h"

namespace bytedance {
namespace flux {

void topk_gather_rs(
    TopKReduceGatherRSArguments const &args, DataTypeEnum dtype, cudaStream_t stream);

void topk_gather_rs_v2(
    TopKReduceGatherRSV2Arguments const &args, DataTypeEnum dtype, cudaStream_t stream);

void ep_topk_gather_rs(
    TopKReduceGatherRSArguments const &args,
    DataTypeEnum dtype,
    int32_t ep_m_start,
    int32_t ep_m_end,
    cudaStream_t stream);

void ep_topk_gather_rs_v2(
    TopKReduceGatherRSV2Arguments const &args,
    DataTypeEnum dtype,
    int32_t ep_m_start,
    int32_t ep_m_end,
    cudaStream_t stream);
}  // namespace flux
}  // namespace bytedance
