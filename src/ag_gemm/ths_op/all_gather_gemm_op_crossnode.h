//===- all_gather_gemm_op_crossnode.h ----------------------------- C++ ---===//
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
#include "coll/ths_op/all_gather_types.h"

namespace bytedance::flux::ths_op {

class AllGatherGemmOpCrossNode {
 public:
  AllGatherGemmOpCrossNode(
      std::shared_ptr<Group> tp_group,
      std::shared_ptr<Group> intra_node_group,
      int32_t nnodes,
      torch::Tensor output_buffer,
      int32_t full_m,
      int32_t n_dim,
      int32_t k_dim,
      c10::ScalarType input_dtype,
      bool transpose_weight = true,
      bool local_copy = false,
      c10::optional<AGRingMode> ring_mode_ = c10::nullopt);
  ~AllGatherGemmOpCrossNode();
  void reset_signals();
  void copy_local(torch::Tensor input);
  torch::Tensor gemm_only(torch::Tensor input, torch::Tensor full_input, torch::Tensor weight);
  torch::Tensor forward(torch::Tensor input, torch::Tensor weight);

 private:
  class AllGatherGemmOpCrossNodeImpl;
  AllGatherGemmOpCrossNodeImpl *impl_ = nullptr;
};
}  // namespace bytedance::flux::ths_op
