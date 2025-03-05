//===- topk_reduce_gather_rs.cc ---------------------------------------------- C++ ---===//
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
#include "flux/args/moe_gather_rs.h"
#include "moe_gather_rs/ths_op/topk_reduce_gather_rs.h"
#include "moe_gather_rs/topk_gather_rs.hpp"
namespace bytedance::flux::ths_op {
using torch::Tensor;

void
topk_reduce_gather_rs(TopKReduceGatherRSArguments const &args, torch::Tensor output) {
  topk_gather_rs(args, from_torch_dtype(output.scalar_type()), c10::cuda::getCurrentCUDAStream());
}

void
ep_topk_reduce_gather_rs(
    TopKReduceGatherRSArguments const &args,
    torch::Tensor output,
    int32_t ep_m_start,
    int ep_m_end) {
  ep_topk_gather_rs(
      args,
      from_torch_dtype(output.scalar_type()),
      ep_m_start,
      ep_m_end,
      c10::cuda::getCurrentCUDAStream());
}

}  // namespace bytedance::flux::ths_op
