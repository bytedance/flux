//===- all2all_single_2d_impl.hpp --------------------------------- C++ ---===//
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
#include <nvshmemx.h>
#include <nvshmem.h>
namespace bytedance {
namespace flux {

struct All2AllSingleParams {
  void *input_comm_ptr;   // symm buf
  void *output_comm_ptr;  // symm buf
  void *output_ptr;       // normal buf
  uint64_t *barrier_ptr;
  int32_t *input_splits;
  int32_t *output_splits;
  int64_t n_dim;
  int64_t max_split;

  int32_t rank;
  int32_t local_rank;
  int32_t local_world_size;
  int32_t world_size;
  int32_t nvshmem_team;
};

void a2a_single_impl(
    const All2AllSingleParams params,
    DataTypeEnum input_dtype,
    int32_t num_comm_sm,
    cudaStream_t stream);
}  // namespace flux
}  // namespace bytedance
