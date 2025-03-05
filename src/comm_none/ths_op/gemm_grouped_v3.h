//===- gemm_grouped_v3.h ----------------------------------------- C++ ---===//
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
#include "flux/ths_op/ths_op.h"

namespace bytedance::flux::ths_op {
class GemmGroupedV3 {
 public:
  GemmGroupedV3(torch::Tensor weight, int64_t num_experts);
  ~GemmGroupedV3();
  torch::Tensor forward(torch::Tensor input, torch::Tensor splits_cpu);
  torch::Tensor profiling(
      torch::Tensor input, torch::Tensor splits_cpu, c10::intrusive_ptr<ProfilingContext> opt_ctx);

 private:
  class GemmGroupedV3Impl;
  GemmGroupedV3Impl *impl_ = nullptr;
};
}  // namespace bytedance::flux::ths_op
