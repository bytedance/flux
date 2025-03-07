//===- gemm_grouped_v2_ag_scatter.h ------------------------------- C++ ---===//
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
#include "coll/ths_op/all_gather_types.h"
#include "flux/ths_op/ths_op.h"

namespace bytedance::flux::ths_op {
std::tuple<
    int,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
prepare_moe_ag_scatter_args(
    torch::Tensor splits_gpu,
    torch::Tensor scatter_index,
    int ntokens,
    int topk,
    int num_weights_group,
    int ep_start,
    int ep_nexperts,
    int rank,
    int world_size,
    int tile_size_m,
    intptr_t stream_);

class GemmGroupedV2AGScatterOp {
 public:
  GemmGroupedV2AGScatterOp(
      std::shared_ptr<Group> tp_group,
      int ep_size,
      int max_ntokens,
      int ffn_hidden,  // before TP shard
      int hidden,
      int num_experts,
      int topk,
      at::ScalarType input_dtype,
      at::ScalarType output_dtype);
  ~GemmGroupedV2AGScatterOp();
  void clear_buffers();
  torch::Tensor forward(
      torch::Tensor inputs_shard,
      torch::Tensor weights,
      torch::Tensor splits_gpu,
      torch::Tensor scatter_index,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      c10::optional<torch::Tensor> outputs_buf,
      c10::optional<torch::Tensor> allgather_output,
      bool fast_accum,
      int sm_margin,
      AllGatherOptionWithOptional ag_option);
  torch::Tensor forward_triton_aot(
      torch::Tensor inputs_shard,
      torch::Tensor weights,
      torch::Tensor splits_gpu,
      torch::Tensor scatter_index,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      c10::optional<torch::Tensor> outputs_buf,
      c10::optional<torch::Tensor> allgather_output,
      bool fast_accum,
      int sm_margin,
      AllGatherOptionWithOptional ag_option);
  std::vector<torch::Tensor> forward_multiple_weights(
      torch::Tensor inputs_shard,
      std::vector<torch::Tensor> weights,
      torch::Tensor splits_gpu,
      torch::Tensor scatter_index,
      c10::optional<std::vector<torch::Tensor>> bias,
      c10::optional<std::vector<torch::Tensor>> input_scale,
      c10::optional<std::vector<torch::Tensor>> weight_scale,
      c10::optional<std::vector<torch::Tensor>> output_scale,
      c10::optional<std::vector<torch::Tensor>> outputs_buf,
      c10::optional<torch::Tensor> allgather_output,
      bool fast_accum,
      int sm_margin,
      AllGatherOptionWithOptional ag_option);
  std::vector<torch::Tensor> profiling(
      torch::Tensor inputs_shard,
      std::vector<torch::Tensor> weights,
      torch::Tensor splits_gpu,
      torch::Tensor scatter_index,
      c10::optional<std::vector<torch::Tensor>> input_scale,
      c10::optional<std::vector<torch::Tensor>> weight_scale,
      c10::optional<std::vector<torch::Tensor>> output_scale,
      c10::optional<std::vector<torch::Tensor>> outputs_buf,
      c10::optional<torch::Tensor> allgather_output,
      bool fast_accum,
      int sm_margin,
      AllGatherOptionWithOptional ag_option,
      c10::intrusive_ptr<ProfilingContext> opt_ctx);

 private:
  class GemmGroupedV2AGScatterOpImpl;
  GemmGroupedV2AGScatterOpImpl *impl_ = nullptr;
};

}  // namespace bytedance::flux::ths_op
