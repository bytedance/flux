//===- flux_shm.h ------------------------------------------------------ C++ ---===//
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
#include <cuda_runtime_api.h>
#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <vector>
#include "c10/util/intrusive_ptr.h"
namespace bytedance::flux {

void init_flux_shm(c10::intrusive_ptr<c10d::ProcessGroup> pg);
torch::Tensor flux_create_tensor(
    const std::vector<int64_t> &shape,
    c10::ScalarType dtype,
    c10::intrusive_ptr<c10d::ProcessGroup> pg = nullptr);
std::vector<torch::Tensor> flux_create_tensor_list(
    const std::vector<int64_t> &shape,
    c10::ScalarType dtype,
    c10::intrusive_ptr<c10d::ProcessGroup> pg = nullptr);
void flux_barrier_all_on_stream(
    cudaStream_t stream,
    c10::optional<std::vector<torch::Tensor>> barrier_tensors = c10::nullopt,
    c10::optional<int> rank = c10::nullopt);
void pyflux_barrier_all_on_stream(
    intptr_t stream,
    c10::optional<std::vector<torch::Tensor>> barrier_tensors = c10::nullopt,
    c10::optional<int> rank = c10::nullopt);

// suggest use the functions above if possible
std::vector<torch::Tensor> cudaipc_create_tensor_list(
    c10::intrusive_ptr<c10d::ProcessGroup> pg,
    const std::vector<int64_t> &shape,
    c10::ScalarType dtype);

#ifdef FLUX_SHM_USE_NVSHMEM
std::vector<torch::Tensor> nvshmem_create_tensor_list(
    const std::vector<int64_t> &shape, c10::ScalarType dtype);
torch::Tensor nvshmem_create_tensor(const std::vector<int64_t> &shape, c10::ScalarType dtype);
#endif

}  // namespace bytedance::flux
