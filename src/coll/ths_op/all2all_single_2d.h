//===- all2all_single_2d.h ---------------------------------------- C++ ---===//
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
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include "flux/ths_op/flux_shm.h"
namespace bytedance::flux::ths_op {
class All2AllSingle {
 public:
  All2AllSingle(
      std::shared_ptr<Group> pg,
      int64_t max_split,
      int64_t n_dim,
      int64_t local_world_size,
      at::ScalarType input_dtype,
      int64_t ep_team);

  ~All2AllSingle();

  torch::Tensor forward(
      torch::Tensor input,
      torch::Tensor output,
      torch::Tensor input_splits,
      torch::Tensor output_splits,
      int32_t num_comm_sm);

 private:
  class All2AllSingleImpl;
  All2AllSingleImpl *impl_;
};

}  // namespace bytedance::flux::ths_op
