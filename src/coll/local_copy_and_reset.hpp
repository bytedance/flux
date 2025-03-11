//===- local_copy_and_reset.hpp ----------------------------------- C++ ---===//
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

void local_copy_and_reset_impl(
    void *input_src,
    void *input_dst,
    void *scale_src,
    void *scale_dst,
    int32_t *counter,
    int32_t *ag_barrier,
    int32_t world_size,
    int32_t rank,
    int32_t m,
    int32_t n,
    int32_t **sync_barriers,  // a list of barrier pointer to sync between device, if set to
                              // nullptr, the sync is not perform.
    DataTypeEnum input_dtype,
    DataTypeEnum scale_dtype,
    bool sync_ring_mode,  // sync in ring_mode or not.
    cudaStream_t stream);

size_t get_local_copy_max_block_num(size_t num_input, int32_t pack_size = 1);
}  // namespace flux
}  // namespace bytedance
