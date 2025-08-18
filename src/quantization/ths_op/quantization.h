//===- quantization.h --------------------------------------------- C++ ---===//
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
#include <c10/core/ScalarType.h>
#include <torch/all.h>

namespace bytedance::flux::ths_op {
class Quantization {
 public:
  Quantization(c10::ScalarType input_dtype, c10::ScalarType output_dtype, int32_t num_streams);
  ~Quantization();

  std::tuple<
      torch::Tensor,
      torch::Tensor,
      c10::optional<torch::Tensor>,
      c10::optional<torch::Tensor>>
  quantize_vector_blockwise(torch::Tensor input, bool return_transpose, float eps = 0.0f);

  std::tuple<
      torch::Tensor,
      torch::Tensor,
      c10::optional<torch::Tensor>,
      c10::optional<torch::Tensor>>
  quantize_square_blockwise(torch::Tensor input, bool return_transpose, float eps = 0.0f);

  std::tuple<
      torch::Tensor,
      torch::Tensor,
      c10::optional<torch::Tensor>,
      c10::optional<torch::Tensor>>
  batch_quantize_square_blockwise(torch::Tensor input, bool return_transpose, float eps = 0.0f);

 private:
  class QuantizationImpl;
  QuantizationImpl *impl = nullptr;
};

}  // namespace bytedance::flux::ths_op
