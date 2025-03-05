//===- helper_ops.h ----------------------------------------------- C++ ---===//
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

namespace bytedance::flux::ths_op {
void bsr_reduce(torch::Tensor input, torch::Tensor output, int block_h, int block_w);
std::pair<torch::Tensor, c10::optional<torch::Tensor>> pad_m_to_TPxTile(
    torch::Tensor input, c10::optional<torch::Tensor> input_scale, int tp_size, int tile_size);
void ring_reduce(torch::Tensor input, torch::Tensor output, int32_t dim, int rank);
}  // namespace bytedance::flux::ths_op
