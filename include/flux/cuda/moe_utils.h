//===- moe_utils.h ----------------------------------------------- C++ ---===//
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

#include <cuda_runtime_api.h>

namespace bytedance::flux {
/**
 * @brief a none-deterministic way to calculate scatter_index from choosed_experts.
 *
 * @param[in] choosed_experts : of topk * ntokens
 * @param[in] count : count of per experts.
 * @param[out] scatter_index : of topk * ntokens
 * @param[in] total_num : topk * ntokens
 * @param[in] expert_num
 * @param[in] stream
 */
void calc_scatter_index(
    const int *choosed_experts,  // of total_num
    const int *count,            // of expert_num
    int *scatter_index,          // of total_num
    const int total_num,         // topk * ntokens
    int expert_num,
    cudaStream_t stream);

}  // namespace bytedance::flux
