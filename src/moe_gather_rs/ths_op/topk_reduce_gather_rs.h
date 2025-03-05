//===- topk_reduce_gather_rs.h ------------------------------------------- C++ ---===//
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
#include "flux/args/moe_gather_rs.h"
namespace bytedance::flux::ths_op {
void topk_reduce_gather_rs(TopKReduceGatherRSArguments const &args, torch::Tensor output);
void ep_topk_reduce_gather_rs(
    TopKReduceGatherRSArguments const &args, torch::Tensor output, int ep_m_start, int ep_m_end);

}  // namespace bytedance::flux::ths_op
