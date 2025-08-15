//===- pre_attn_a2a_transpose_impls.hpp ------------------------ C++ ------===//
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
struct PreAttnAll2AllTransposeParam {
  void **input_ptrs;
  void *output_ptr;
  void *barrier_ptrs[kMaxWorldSize];
  int32_t bs;
  int32_t local_nheads;
  int32_t local_seq_len;
  int32_t head_dim;
  int32_t rank;
  int32_t world_size;
  int32_t TILE_M;  // along the seq dim of the output, ensure that local_seq_len % TILE_M == 0;
  int32_t TILE_N;  // along the head_dim and local_nheads dim of output, ensure that
                   // TILE_N % head_dim == 0
  int32_t m_per_barrier;
  int32_t n_per_barrier;
  int32_t NUM_COMM_SM;
};

void pre_attn_all2all_transpose_impl(
    const PreAttnAll2AllTransposeParam param, DataTypeEnum input_dtype, cudaStream_t stream);

}  // namespace flux
}  // namespace bytedance
