//===- all2all_impl.hpp --------------------------------------- C++ ---===//
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
#include <cuda.h>

namespace bytedance {
namespace flux {

struct All2allParams {
  void *input_ptr;
  void *output_ptr;
  int32_t *splits_input_buffer;
  int32_t *splits_output_buffer;
  uint64_t *signal_buffer;
  int32_t *input_splits_cumsum;
  float *scale_input_buffer;
  float *scale_output_buffer;
  int32_t rank;
  int32_t world_size;
  int32_t ndim;
  int32_t element_size;
  int32_t max_token;
  int32_t expert_per_rank;
  uint64_t signal_to_wait;
  bool with_scale;
};

void all2all_impl(const All2allParams &params, cudaStream_t stream);

}  // namespace flux
}  // namespace bytedance
