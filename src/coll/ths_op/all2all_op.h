//===- all2all_op.h -------------------------------------------- C++ ---===//
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
#include <vector>
#include <cuda_runtime_api.h>
#include <c10/core/ScalarType.h>
#include <torch/all.h>
namespace bytedance::flux::ths_op {
class All2AllInference {
 public:
  All2AllInference(
      int64_t max_m,
      int64_t n_dim,
      int64_t rank,
      int64_t total_num_experts,
      int64_t world_size,
      int64_t local_world_size,
      int64_t max_element_size);

  ~All2AllInference();
  std::vector<torch::Tensor> get_input_buffer(
      std::vector<int32_t> input_shape, int64_t element_size, bool with_scale);
  std::vector<torch::Tensor> forward(
      std::vector<int32_t> input_size,
      torch::Tensor input_split_cumsum,
      int64_t element_size,
      bool with_scale);
  std::vector<torch::Tensor> forward_with_stream(
      std::vector<int32_t> input_size,
      torch::Tensor input_split_cumsum,
      int64_t element_size,
      bool with_scale,
      cudaStream_t stream);

 private:
  class All2AllInferenceOpImpl;
  All2AllInferenceOpImpl *impl_;
};

}  // namespace bytedance::flux::ths_op
