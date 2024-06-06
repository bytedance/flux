//===- nvshmem_utils.h -------------------------------------------- C++ ---===//
//
// Copyright 2023 ByteDance Ltd. and/or its affiliates. All rights reserved.
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
#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <vector>
namespace bytedance::flux {
torch::Tensor nvshmem_create_tensor(const std::vector<int64_t> &shape, c10::ScalarType dtype);

std::vector<torch::Tensor> nvshmem_create_tensor_list(
    const std::vector<int64_t> &shape, c10::ScalarType dtype);

std::vector<torch::Tensor> create_ipc_tensors(
    c10d::ProcessGroup &pg, const std::vector<int64_t> &shape, c10::ScalarType dtype);
}  // namespace bytedance::flux
