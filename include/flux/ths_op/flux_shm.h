//===- flux_shm.h ------------------------------------------------------ C++ ---===//
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
#include <cuda_runtime_api.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/torch.h>
#include <vector>
namespace bytedance::flux {

class Group {
 public:
  virtual int get_rank() = 0;
  virtual int get_size() = 0;
  //   virtual std::string get_group_name() const = 0;
  virtual void all_gather_cpu(const void *src, void *dst, int64_t count) = 0;
  virtual void broadcast_cpu(void *src, int64_t nbytes, int root_rank) = 0;
  virtual void sync() = 0;
};

// never mind the efficiency: just a simple helper
float all_reduce_max_float(Group *group, const float src);

class C10dProcessGroup : public Group {
 public:
  C10dProcessGroup(const std::string &name, c10::intrusive_ptr<c10d::ProcessGroup> pg);
  ~C10dProcessGroup();

  int get_rank() override;
  int get_size() override;
#if 0
    std::string get_group_name() const override;
#endif

  void sync() override;
  void all_gather_cpu(const void *src, void *dst, int64_t nbytes) override;
  void broadcast_cpu(void *src, int64_t nbytes, int root_rank = 0) override;

 private:
  class Impl;
  Impl *impl_ = nullptr;
};

torch::Tensor flux_create_tensor(
    const std::vector<int64_t> &shape,
    c10::ScalarType dtype,
    c10::intrusive_ptr<c10d::ProcessGroup> pg = nullptr);
std::vector<torch::Tensor> flux_create_tensor_list(
    const std::vector<int64_t> &shape,
    c10::ScalarType dtype,
    c10::intrusive_ptr<c10d::ProcessGroup> pg = nullptr,
    bool ring_mode = false);
void flux_barrier_all_on_stream(
    cudaStream_t stream,
    c10::optional<std::vector<torch::Tensor>> barrier_tensors = c10::nullopt,
    c10::optional<int> rank = c10::nullopt,
    bool ring_mode = false);

void init_flux_shm(Group *pg);
torch::Tensor flux_create_tensor(
    const std::vector<int64_t> &shape, c10::ScalarType dtype, Group *group = nullptr);
std::vector<torch::Tensor> flux_create_tensor_list(
    const std::vector<int64_t> &shape,
    c10::ScalarType dtype,
    Group *group = nullptr,
    bool ring_mode = false);

class GroupBarrier {
 public:
  GroupBarrier(std::shared_ptr<Group> group, bool ring_mode = false);
  ~GroupBarrier();
  void barrier_all(cudaStream_t stream);

 private:
  class BarrierImpl;
  BarrierImpl *impl_ = nullptr;
};

// suggest use the functions above if possible
std::vector<torch::Tensor> cudaipc_create_tensor_list(
    Group *group,
    const std::vector<int64_t> &shape,
    c10::ScalarType dtype,
    bool ring_mode = false);

#ifdef FLUX_SHM_USE_NVSHMEM
std::vector<torch::Tensor> nvshmem_create_tensor_list(
    const std::vector<int64_t> &shape, c10::ScalarType dtype);

torch::Tensor nvshmem_create_tensor(const std::vector<int64_t> &shape, c10::ScalarType dtype);
#endif

// NOTE: this should be a symetric operation
std::vector<torch::Tensor> flux_create_shm_tensor_list(
    const std::vector<int64_t> &shape, c10::ScalarType dtype, Group *group);

}  // namespace bytedance::flux
