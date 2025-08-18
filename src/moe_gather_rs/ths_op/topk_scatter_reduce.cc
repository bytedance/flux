//===- topk_scatter_reduce.cc ------------------------------------- C++ ---===//
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

#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <cuda_bf16.h>
#include "flux/ths_op/ths_op.h"
#include "moe_gather_rs/moe_utils.h"

namespace bytedance::flux::ths_op {
using torch::Tensor;
torch::Tensor
topk_scatter_reduce(std::vector<torch::Tensor> inputs, torch::Tensor scatter_idx, int64_t TOPK) {
  FLUX_CHECK(!inputs.empty());
  int32_t topk = TOPK;
  int32_t M = inputs[0].size(0);
  int32_t N = inputs[0].size(1);
  int32_t new_M = M / topk;
  FLUX_CHECK_DIV(M, topk);
  FLUX_CHECK_EQ(scatter_idx.numel(), M);
  torch::Tensor output = torch::empty({new_M, inputs[0].size(1)}, inputs[0].options());
  std::vector<void *> ptrs;
  for (int i = 0; i < inputs.size(); i++) {
    ptrs.push_back(inputs[i].data_ptr());
  }
  auto data_type_enum = from_torch_dtype(output.scalar_type());
  topk_reduce_scatter_impl(
      ptrs.data(),
      ptrs.size(),
      data_type_enum,
      scatter_idx.data_ptr<int32_t>(),
      topk,
      output.data_ptr(),
      new_M,
      N);
  return output;
}

}  // namespace bytedance::flux::ths_op
