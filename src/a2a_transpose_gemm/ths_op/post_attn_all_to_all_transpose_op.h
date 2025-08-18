//===- post_attn_all_to_all_transpose_op.h ------------------------ C++ ---===//
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
#include <c10/util/Optional.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <torch/all.h>

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include "a2a_transpose_gemm/post_attn_a2a_transpose_impls.hpp"
#include "all_to_all_types.h"

namespace bytedance::flux {

class PostAttnAllToAllTransposeOp {
 public:
 public:
  PostAttnAllToAllTransposeOp(
      std::shared_ptr<Group> pg_world,
      int nnodes,
      int sp_size,
      size_t bs,
      size_t num_heads,
      size_t seq,
      size_t head_dim,
      int32_t max_num_comm_buf,
      at::ScalarType input_dtype,
      bool a2a_only);

  torch::Tensor run(
      torch::Tensor input,
      const AllToAllOption &opt,
      int32_t comm_buf_idx,
      int32_t num_comm_sm,
      cudaStream_t stream);

  torch::Tensor run(
      torch::Tensor input,
      torch::Tensor seq_lens_cpu,
      const AllToAllOption &opt,
      int32_t comm_buf_idx,
      int32_t num_comm_sm,
      cudaStream_t stream);

  // only provide local tensor
  torch::Tensor
  local_comm_output_buffer() {
    return comm_output_buffer_;
  }

  torch::Tensor get_comm_result(
      int32_t comm_buf_idx,
      const AllToAllOption &opt,
      int32_t bs,
      int32_t local_seq_len,
      int32_t nheads,
      int32_t head_dim,
      cudaStream_t stream);

  torch::Tensor
  local_barrier_buffer() {
    return barrier_buffer_;
  }

  int32_t *a2a_signal_ptr() const;

  int32_t
  m_per_barrier() {
    return this->copy_param_.TILE_M;
  }

  void reset_signals(const AllToAllOption &opt, cudaStream_t stream);

  void sp_group_barrier_async(cudaStream_t stream);

 private:
  void create_symetric_buffers();
  void create_sync_buffers();

  bool is_p2p_atomic_supported();

  void copy_all_to_all_with_transpose(
      torch::Tensor input, const AllToAllOption &opt, cudaStream_t stream);

  void copy_all_to_all(
      torch::Tensor input,
      const AllToAllOption &opt,
      cudaStream_t stream,
      bool is_dyn_seq = false);

  void
  _ensure_topo_initialized() {
    if (!topo_utils::is_topo_initialized()) {
      topo_utils::initialize_topo(this->pg_world_.get());
    }
  }

  void init_all2all_copy_param(
      torch::Tensor input,
      int32_t comm_buf_idx,
      int32_t num_comm_sm,
      bool use_read,
      bool skip_barrier);

  void init_all2all_copy_param(
      torch::Tensor input,
      torch::Tensor seq_lens_cpu,
      int32_t comm_buf_idx,
      int32_t num_comm_sm,
      bool use_read,
      bool skip_barrier);

 private:
  std::shared_ptr<Group> pg_world_;

  int nnodes_;
  int world_size_;
  int rank_;
  int local_world_size_;
  int local_rank_;
  int sp_size_;
  int sp_rank_;

  int bs_, num_heads_, seq_, head_dim_;
  at::ScalarType input_dtype_;

  // used for the cuda-ipc-barrier
  std::vector<torch::Tensor> sync_buffers_;
  std::vector<torch::Tensor> comm_output_buffers_;  // all2all transpose output buffers
  std::vector<torch::Tensor> barrier_buffers_;
  torch::Tensor comm_output_buffer_;

  // torch::Tensor output_buffer;
  torch::Tensor barrier_buffer_;
  torch::Tensor sync_ptrs_buffer_;  // symetric memory for barrier all. only with atomic supported
  torch::Tensor barrier_ptrs_buffer_;
  std::vector<torch::Tensor> comm_output_ptrs_buffer_list_;

  std::vector<void *> comm_output_ptrs_;  // all2all transpose output ptrs
  std::vector<int32_t *> barrier_ptrs_;
  std::vector<int32_t *> sync_ptrs_;

  int32_t max_num_comm_buf_;
  bool a2a_only_ = false;
  bool p2p_atomic_supported_ = false;

  PostAttnAll2AllParams copy_param_;
};
}  // namespace bytedance::flux
