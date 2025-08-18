//===- all2all_single_2d.cc --------------------------------------- C++ ---===//
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
#include "coll/ths_op/all2all_single_2d.h"

#include <ATen/core/List.h>
#include <ATen/core/jit_type.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/ops/empty.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/intrusive_ptr.h>
#include <cuda_runtime_api.h>
#include <cutlass/gemm/gemm.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <torch/all.h>

#include <cstdlib>
#include <iostream>
#include <mutex>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <vector>

#include "coll/all2all_single_2d_impl.hpp"
#include "flux/args/moe_gather_rs.h"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/cuda_stub.h"
#include "flux/flux.h"
#include "flux/gemm_meta.h"
#include "flux/op_registry.h"
#include "flux/ths_op/ths_op.h"
#include "flux/ths_op/util.h"

#define MOD_VALUE 1000000
namespace bytedance {
namespace flux {
namespace ths_op {

using torch::Tensor;

class All2AllSingle::All2AllSingleImpl {
 private:
  std::shared_ptr<Group> pg_;
  int64_t max_split_;
  int64_t n_dim_;
  int32_t local_world_size_;
  int32_t rank_;
  int32_t local_rank_;
  int32_t world_size_;
  int32_t nnodes_;
  int32_t node_id_;
  at::ScalarType input_dtype_;

  torch::Tensor input_comm_buf_;
  torch::Tensor output_comm_buf_;
  torch::Tensor barrier_;
  nvshmem_team_t nvshmem_team;

 public:
  All2AllSingleImpl(
      std::shared_ptr<Group> pg,
      int64_t max_split,
      int64_t n_dim,
      int64_t local_world_size,
      at::ScalarType input_dtype,
      int64_t nvshmem_team_)
      : pg_(pg),
        max_split_(max_split),
        n_dim_(n_dim),
        local_world_size_(local_world_size),
        rank_(pg->get_rank()),
        world_size_(pg->get_size()),
        input_dtype_(input_dtype),
        nvshmem_team((nvshmem_team_t)nvshmem_team_) {
    FLUX_CHECK(world_size_ % local_world_size_ == 0);
    this->nnodes_ = world_size_ / local_world_size_;
    this->node_id_ = this->rank_ / this->local_world_size_;
    this->local_rank_ = this->rank_ % this->local_world_size_;
    int64_t max_m = max_split * world_size_;
    this->input_comm_buf_ = nvshmem_create_tensor({max_m, n_dim}, input_dtype_, false);
    this->output_comm_buf_ = nvshmem_create_tensor({max_m, n_dim}, input_dtype_, false);
    this->barrier_ = nvshmem_create_tensor({world_size_}, c10::ScalarType::Long, true);
  }

  void
  check_io(
      torch::Tensor input,
      torch::Tensor output,
      torch::Tensor input_splits,
      torch::Tensor output_splits) {
    CHECK_NDIM(input, 2);
    CHECK_NDIM(output, 2);
    CHECK_NDIM(input_splits, 1);
    CHECK_NDIM(output_splits, 1);

    FLUX_CHECK(input.dtype() == this->input_dtype_);
    FLUX_CHECK(output.dtype() == this->input_dtype_);
    FLUX_CHECK(input_splits.dtype() == at::ScalarType::Int);
    FLUX_CHECK(output_splits.dtype() == at::ScalarType::Int);

    FLUX_CHECK(input.is_contiguous());
    FLUX_CHECK(output.is_contiguous());
    FLUX_CHECK(input_splits.is_contiguous());
    FLUX_CHECK(output_splits.is_contiguous());

    FLUX_CHECK(input.device().is_cuda());
    FLUX_CHECK(output.device().is_cuda());
    FLUX_CHECK(input_splits.device().is_cuda());
    FLUX_CHECK(output_splits.device().is_cuda());

    FLUX_CHECK(input_splits.size(0) == this->world_size_);
    FLUX_CHECK(output_splits.size(0) == this->world_size_);
    FLUX_CHECK(input.size(1) == output.size(1));
    FLUX_CHECK(input.size(1) <= this->n_dim_);
    FLUX_CHECK(input.size(0) <= input_comm_buf_.size(0));
    FLUX_CHECK(output.size(0) <= output_comm_buf_.size(0));
  }

  void
  a2a_single(
      torch::Tensor input,
      torch::Tensor output,
      torch::Tensor input_splits,
      torch::Tensor output_splits,
      int32_t num_comm_sm,
      cudaStream_t stream) {
    All2AllSingleParams args{
        .input_comm_ptr = this->input_comm_buf_.data_ptr(),
        .output_comm_ptr = this->output_comm_buf_.data_ptr(),
        .output_ptr = output.data_ptr(),
        .barrier_ptr = reinterpret_cast<uint64_t *>(this->barrier_.data_ptr()),
        .input_splits = input_splits.data_ptr<int32_t>(),
        .output_splits = output_splits.data_ptr<int32_t>(),
        .n_dim = input.size(1),
        .max_split = this->max_split_,
        .rank = this->rank_,
        .local_rank = this->local_rank_,
        .local_world_size = this->local_world_size_,
        .world_size = this->world_size_,
        .nvshmem_team = this->nvshmem_team};
    a2a_single_impl(args, ths_op::from_torch_dtype(this->input_dtype_), num_comm_sm, stream);
  }

  torch::Tensor
  forward_impl(
      torch::Tensor input,
      torch::Tensor output,
      torch::Tensor input_splits,
      torch::Tensor output_splits,
      int32_t num_comm_sm,
      cudaStream_t stream) {
    check_io(input, output, input_splits, output_splits);
    // copy to symm buf
    CUDA_CHECK(cudaMemcpyAsync(
        input_comm_buf_.data_ptr(),
        input.data_ptr(),
        input.nbytes(),
        cudaMemcpyDeviceToDevice,
        stream));

    // perform a2a single
    a2a_single(input, output, input_splits, output_splits, num_comm_sm, stream);
    return output;
  }

  torch::Tensor
  forward(
      torch::Tensor input,
      torch::Tensor output,
      torch::Tensor input_splits,
      torch::Tensor output_splits,
      int32_t num_comm_sm) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    return forward_impl(input, output, input_splits, output_splits, num_comm_sm, stream);
  }
};

All2AllSingle::All2AllSingle(
    std::shared_ptr<Group> pg,
    int64_t max_split,
    int64_t n_dim,
    int64_t local_world_size,
    at::ScalarType input_dtype,
    int64_t ep_team)
    : impl_(new All2AllSingleImpl(pg, max_split, n_dim, local_world_size, input_dtype, ep_team)) {}

All2AllSingle::~All2AllSingle() { delete impl_; }

torch::Tensor
All2AllSingle::forward(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor input_splits,
    torch::Tensor output_splits,
    int32_t num_comm_sm) {
  FLUX_CHECK(impl_ != nullptr) << "All2AllSingle is not initialized!";
  return impl_->forward(input, output, input_splits, output_splits, num_comm_sm);
}

}  // namespace ths_op
}  // namespace flux
}  // namespace bytedance
