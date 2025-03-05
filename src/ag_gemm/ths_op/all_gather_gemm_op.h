//===- all_gather_gemm_op.h --------------------------------------- C++ ---===//
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
#include <c10/util/intrusive_ptr.h>
#include <c10/util/Optional.h>
#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include "flux/ths_op/ths_op.h"
#include "coll/ths_op/all_gather_types.h"

namespace bytedance::flux::ths_op {

class AllGatherGemmOp {
 public:
  AllGatherGemmOp(
      std::shared_ptr<Group> tp_group,
      int32_t nnodes,
      int32_t full_m,
      int32_t n_dim,
      int32_t k_dim,
      c10::ScalarType input_dtype,
      c10::ScalarType output_dtype,
      bool use_pdl);

  ~AllGatherGemmOp();

  torch::Tensor forward(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum,
      bool transpose_weight,
      AllGatherOptionWithOptional opt,
      c10::optional<torch::Tensor> gathered_input);

  torch::Tensor gemm_only(
      torch::Tensor input,  // this should be the full input
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      c10::optional<torch::Tensor> input_scale,  // this should be the full scale
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum,
      bool transpose_weight);

  torch::Tensor profiling(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum,
      bool transpose_weight,
      AllGatherOptionWithOptional option_,
      c10::optional<torch::Tensor> gathered_input,
      c10::intrusive_ptr<ProfilingContext> opt_ctx);

 private:
  class AllGatherGemmOpImpl;
  AllGatherGemmOpImpl *impl_ = nullptr;
};
}  // namespace bytedance::flux::ths_op
