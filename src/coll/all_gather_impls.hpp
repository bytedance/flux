//===- all_gather_impls.hpp --------------------------------------- C++ ---===//
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
#include "flux/flux.h"

namespace bytedance {
namespace flux {

struct AllGatherParams {
  void *input_ptrs[kMaxWorldSize];  // input_ptrs[rank]: (m * world_size, n)
  void *scale_ptrs[kMaxWorldSize];
  int32_t *ag_barriers[kMaxWorldSize];  // ag signal
  int32_t *counter;                     // sync between block to write signal

  int32_t world_size;
  int32_t rank;
  int sub_world_size;
  int32_t m;  // input.size(0), actually m_per_rank
  int32_t n;  // input.size(1)
  bool has_scale;
  int32_t *ag_signal;  // the signal to ensure that GEMM is launched after AllGather
};

void ag_a2a_mode(
    const AllGatherParams &params,
    DataTypeEnum input_dtype,
    DataTypeEnum scale_dtype,
    cudaStream_t stream);

void ag_ring_with_scale(
    const AllGatherParams &params,
    int input_elem_size,
    int scale_elem_size,
    int num_grids,
    bool use_2d_mode,
    cudaStream_t stream);

}  // namespace flux
}  // namespace bytedance
