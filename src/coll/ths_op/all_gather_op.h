//===- all_gather_op.h -------------------------------------------- C++ ---===//
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

#include "all_gather_types.h"
#include <ATen/core/ivalue.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>
#include <stdlib.h>
#include <torch/all.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <cuda_runtime_api.h>

namespace bytedance::flux::ths_op {
/** AllGather is not as common as it seems:
 *  1. it's actually with business logic, such as input_buffer and input_scale_buffer, which is for
 * int8 GEMM all-gather
 */

class AllGatherOp {
 public:
  AllGatherOp(
      std::shared_ptr<Group> tp_group,
      int nnodes,
      size_t max_m,
      size_t k,
      at::ScalarType input_dtype);

  ~AllGatherOp();

  void run_with_optional_options(
      torch::Tensor input,
      c10::optional<torch::Tensor> input_scale,
      const AllGatherOptionWithOptional &opt,
      cudaStream_t stream);

  void run(
      const torch::Tensor &input,
      c10::optional<torch::Tensor> input_scale,
      const AllGatherOption &opt,
      cudaStream_t stream);

  // only provide local tensor
  torch::Tensor local_input_buffer();
  torch::Tensor local_input_scale_buffer();
  torch::Tensor local_barrier_buffer();

  int32_t *ag_signal_ptr() const;

  cudaEvent_t &get_local_prepare_event();

 private:
  class AllGatherOpImpl;
  AllGatherOpImpl *impl_;
};
}  // namespace bytedance::flux::ths_op
