//===- gemm_reduce_scatter.h -------------------------------------- C++ ---===//
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
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include "flux/ths_op/ths_op.h"
#include "coll/ths_op/reduce_scatter_op.h"

namespace bytedance::flux::ths_op {
class GemmRS {
 public:
  GemmRS(
      std::shared_ptr<Group> tp_group,
      int32_t nnodes,
      int32_t max_m,
      int32_t n_dim,
      c10::ScalarType input_dtype,
      c10::ScalarType output_dtype,
      bool transpose_weight,
      bool fuse_reduction,
      bool ring_reduction);
  ~GemmRS();
  void zero_buffers();
  torch::Tensor forward(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum,
      const ReduceScatterOptionWithOptional &reduce_scatter_option);
  torch::Tensor profiling(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum,
      c10::intrusive_ptr<ProfilingContext> opt_ctx,
      const ReduceScatterOptionWithOptional &reduce_scatter_option);
  void forward_barrier(
      torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias);
  torch::Tensor forward_reduce_scatter(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale);

 private:
  class GemmRSImpl;
  GemmRSImpl *impl_ = nullptr;
};
}  // namespace bytedance::flux::ths_op
