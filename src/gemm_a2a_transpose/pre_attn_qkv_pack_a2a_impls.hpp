//===- pre_attn_qkv_pack_a2a_impls.hpp ------------------------- C++ ------===//
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
struct PreAttnQKVPackA2AParams {
  void **input_ptrs;
  void *q_ptr;
  void *k_ptr;
  void *v_ptr;
  void *barrier_ptrs[kMaxWorldSize];
  int32_t bs;
  int32_t local_q_nheads;
  int32_t local_k_nheads;
  int32_t local_v_nheads;
  int32_t head_dim;
  int32_t rank;
  int32_t world_size;
  int32_t TILE_S;   // tile size of output(qkv pack) seq dim
  int32_t TILE_NH;  // tile size of output(qkv pack) nheads dim
  int32_t m_per_barrier;
  int32_t n_per_barrier;
  int32_t num_comm_sm;
  int32_t cusum_seq_lens[kMaxWorldSize + 1];
  bool skip_barrier = false;
};

void pre_attn_qkv_pack_a2a_impl(
    const PreAttnQKVPackA2AParams params, DataTypeEnum input_dtype, cudaStream_t stream);

}  // namespace flux
}  // namespace bytedance
