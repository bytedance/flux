//===- helper_kernels.h ------------------------------------------- C++ ---===//
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
#include <assert.h>
#include "flux/flux.h"
#include <stdio.h>

namespace bytedance::flux {

bool bitwise_check(DataTypeEnum dtype, void *ptr_A, void *ptr_B, size_t capacity);

void uniform_initialize(
    DataTypeEnum dtype,
    void *ptr,
    size_t capacity,
    uint64_t seed,
    double min = 0.0,
    double max = 1.0,
    void *stream = nullptr);

void cudaipc_barrier_all_on_stream_impl(
    cudaStream_t stream, int32_t **sync_buffer_ptr, int rank, int world_size);
}  // namespace bytedance::flux
