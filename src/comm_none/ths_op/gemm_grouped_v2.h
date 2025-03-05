//===- gemm_grouped_v2.h ----------------------------------------- C++ ---===//
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

#include <torch/all.h>

namespace bytedance::flux::ths_op {
class GemmGroupedV2 {
 public:
  GemmGroupedV2(
      torch::Tensor weight, int64_t num_experts, at::ScalarType in_type, at::ScalarType out_type);
  ~GemmGroupedV2();
  torch::Tensor forward(
      torch::Tensor input,
      torch::Tensor splits_cpu,
      c10::optional<std::vector<torch::Tensor>> input_scale,
      c10::optional<std::vector<torch::Tensor>> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum,
      int64_t sm_margin);

 private:
  class GemmGroupedV2Impl;
  GemmGroupedV2Impl *impl_ = nullptr;
};
}  // namespace bytedance::flux::ths_op
