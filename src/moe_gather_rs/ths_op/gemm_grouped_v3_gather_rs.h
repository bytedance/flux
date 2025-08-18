//===- gemm_grouped_v3_gather_rs.h -------------------------------- C++ ---===//
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
#include "flux/ths_op/ths_op.h"

namespace bytedance::flux::ths_op {

class GemmGroupedV3GatherRS {
 public:
  GemmGroupedV3GatherRS(
      int64_t total_num_experts,
      int64_t max_m,
      int64_t n_dim,
      int64_t topk,
      int64_t rank,
      int64_t world_size,
      int64_t tp_world_size,
      int64_t ep_world_size,
      int64_t max_input_groups);
  ~GemmGroupedV3GatherRS();
  torch::Tensor forward_gather_rs(
      torch::Tensor input,
      torch::Tensor weight,
      torch::Tensor splits_cpu,
      torch::Tensor routing_idx,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_vec_scale,
      bool fast_accum,
      int sm_margin,
      bool with_stream_sync);
  torch::Tensor forward_gather_rs_multiple(
      std::vector<torch::Tensor> inputs,
      std::vector<torch::Tensor> weights,
      torch::Tensor splits_cpu,
      torch::Tensor routing_idx,
      c10::optional<std::vector<torch::Tensor>> input_scale,
      c10::optional<std::vector<torch::Tensor>> weight_scale,
      c10::optional<std::vector<torch::Tensor>> output_vec_scale,
      bool fast_accum,
      int sm_margin,
      bool with_stream_sync);
  torch::Tensor profiling(
      torch::Tensor input,
      torch::Tensor weight,
      torch::Tensor splits_cpu,
      torch::Tensor routing_idx,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_vec_scale,
      bool fast_accum,
      int sm_margin,
      bool with_stream_sync,
      c10::intrusive_ptr<ProfilingContext> opt_ctx);
  torch::Tensor forward_gather_rs_no_zerobuffer(
      torch::Tensor input,
      torch::Tensor weight,
      torch::Tensor splits_cpu,
      torch::Tensor routing_idx,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_vec_scale,
      bool fast_accum,
      int sm_margin,
      bool with_stream_sync);
  void ep_zero_buffer();

 private:
  class GemmGroupedV3GatherRSOpImpl;
  GemmGroupedV3GatherRSOpImpl *impl_;
};

}  // namespace bytedance::flux::ths_op
