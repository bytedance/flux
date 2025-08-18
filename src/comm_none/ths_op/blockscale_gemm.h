//===- blockscale_gemm.h ------------------------------------------ C++ ---===//
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
#include "flux/ths_op/ths_op.h"

namespace bytedance::flux::ths_op {
class BlockScaleGemm {
 public:
  BlockScaleGemm(c10::ScalarType input_dtype, c10::ScalarType output_dtype, int32_t num_streams);
  ~BlockScaleGemm();

  torch::Tensor forward(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale);

  torch::Tensor forward_multistream(
      torch::Tensor input,
      torch::Tensor input_splits,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale);

  torch::Tensor reference(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale);

  torch::Tensor profiling(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::intrusive_ptr<ProfilingContext> opt_ctx);

 private:
  class BlockScaleGemmImpl;
  BlockScaleGemmImpl *impl_ = nullptr;
};

}  // namespace bytedance::flux::ths_op
