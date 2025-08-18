//===- triton_util.h ------------------------------------------- C++ ------===//
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
#include <cstdint>
#include <cuda_runtime_api.h>

namespace bytedance::flux {
void get_moe_ag_scatter_args(
    const int32_t *splits_gpu_ptr,
    const int32_t *cumsum_per_rank_gpu_ptr,
    void *problem_schedule_ptr,
    int num_scheds,
    int32_t *gather_index_ptr,
    int32_t *scatter_index_ptr,
    int ep_start,
    int ep_experts,
    int world_size,
    int M_this_ep,
    int tile_size_m,
    int32_t *m_pad_ptr,
    int32_t *gather_a_ptr,
    int32_t *scatter_d_ptr,
    int32_t *expert_idx_ptr,
    int32_t *rank_start_ptr,
    int32_t *rank_end_ptr,
    cudaStream_t stream);

}
