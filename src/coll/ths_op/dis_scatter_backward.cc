//===- dis_scatter_backward.cc ------------------------------------------ C++ ---===//
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
#include <ATen/core/List.h>
#include <ATen/core/jit_type.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/ops/empty.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/intrusive_ptr.h>
#include <coll/ths_op/dis_scatter_backward.h>
#include <cuda_runtime_api.h>
#include <cutlass/gemm/gemm.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <torch/all.h>

#include <coll/dis_scatter_backward_impl.hpp>
#include <cstdlib>
#include <cutlass/util/packed_stride.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <vector>

#include "coll/dis_scatter_forward_impl.hpp"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/helper_kernels.h"
#include "flux/flux.h"
#include "flux/ths_op/ths_op.h"
#include "flux/ths_op/util.h"
#include "flux/ths_op/flux_shm.h"

namespace bytedance {
namespace flux {
namespace ths_op {

using torch::Tensor;

/// This class only runs the basic grouped_gemm, it is mainly used for testing
class DisScatterBackward::DisScatterBackwardOpImpl {
 private:
  at::ScalarType _st;
  int32_t total_num_experts;
  int32_t max_token;  // max token equals to bs * max_seq_len  TODO: optimize the memory usage
  int32_t n_dim;
  int32_t topk;
  int32_t rank;
  int32_t tp_world_size;
  int32_t ep_world_size;
  int32_t dp_world_size;
  int32_t local_world_size;
  int32_t n_nodes;
  int32_t node_id;
  int32_t dp_rank;
  torch::Tensor output_buffer;
  torch::Tensor internal_buffer;
  torch::Tensor reduction_buffer;
  // used to perform the barrier all within the node
  torch::Tensor local_barrier_buffer;
  torch::Tensor all2all_index_tensor;
  torch::Tensor token_counts_send;
  // for forward_gpu
  torch::Tensor ep_token_cum_sum_gpu;
  torch::Tensor splits_cum_sum_gpu;
  torch::Tensor splits_cum_sum_gpu_per_rank;
  // for forward
  std::vector<int32_t> ep_token_cum_sum;
  std::vector<int32_t> splits_cum_sum;
  std::vector<int32_t *> local_barrier_ptrs;
  bool buffer_initialized;
  // for forward_gpu
  float scale_factor;
  nvshmem_team_t nvshmem_team;

  void
  init_buffer(torch::Tensor input, int inter_buffer_m) {
    if (this->n_nodes > 1) {
      printf("Init internal buffer for across node with M:%d\n", inter_buffer_m);
      auto data_type = input.scalar_type();
      if (!this->buffer_initialized) {
        FLUX_CHECK(!this->output_buffer.defined());
        FLUX_CHECK(!this->internal_buffer.defined());
        FLUX_CHECK(!this->reduction_buffer.defined());
        // the internal buffer should be dyanmic scaled according to the token distribution
        // banlance
        this->internal_buffer = nvshmem_create_tensor({inter_buffer_m, this->n_dim}, data_type);
        // the output buffer should be initilized only once
        this->output_buffer =
            nvshmem_create_tensor({this->max_token * this->n_nodes, this->n_dim}, data_type);
        this->reduction_buffer =
            nvshmem_create_tensor({this->max_token * this->n_nodes, this->n_dim}, data_type);
        // local_barrier is used to performance the barrier_all within the node
        this->local_barrier_buffer =
            nvshmem_create_tensor({this->local_world_size}, c10::ScalarType::Int);
        this->local_barrier_buffer.zero_();
        for (int i = 0; i < this->local_world_size; i++) {
          int target_rank_in_team = this->node_id * this->local_world_size + i;
          int target_rank_global = nvshmem_team_translate_pe(
              this->nvshmem_team, target_rank_in_team, NVSHMEM_TEAM_WORLD);
          if (target_rank_in_team != this->rank) {
            this->local_barrier_ptrs.push_back(
                reinterpret_cast<int32_t *>(
                    nvshmem_ptr(this->local_barrier_buffer.data_ptr(), target_rank_global)));
          } else {
            this->local_barrier_ptrs.push_back(this->local_barrier_buffer.data_ptr<int32_t>());
          }
        }

        this->token_counts_send =
            torch::zeros({this->n_nodes}, input.options().dtype(at::ScalarType::Int));
        int index_size = this->n_nodes * (this->max_token * 4 + this->max_token * this->topk * 2) +
                         (this->max_token * (this->topk + 1));
        this->all2all_index_tensor =
            torch::empty({index_size}, input.options().dtype(at::ScalarType::Int));

      } else {
        // only scale the output_buffer
        FLUX_CHECK(this->output_buffer.defined());
        FLUX_CHECK(this->reduction_buffer.defined());
        FLUX_CHECK(this->internal_buffer.defined());
        FLUX_CHECK(this->local_barrier_buffer.defined());
        // scale the output_buffer if necessary
        if (inter_buffer_m > this->internal_buffer.size(0)) {
          this->internal_buffer = nvshmem_create_tensor({inter_buffer_m, this->n_dim}, data_type);
        }
      }
    } else {
      // there is no need to scale the output buffer when there is only one node
      printf("Init the output buffer for dis-scatte-backward with max_m:%d\n", this->max_token);
      if (!this->buffer_initialized) {
        auto data_type = input.scalar_type();
        this->token_counts_send =
            torch::zeros({this->local_world_size}, input.options().dtype(at::ScalarType::Int));
        int index_size = this->local_world_size * (this->max_token * (this->topk + 2)) +
                         (this->max_token * (this->topk + 1));
        this->all2all_index_tensor =
            torch::empty({index_size}, input.options().dtype(at::ScalarType::Int));

        FLUX_CHECK(!this->output_buffer.defined());
        this->output_buffer = nvshmem_create_tensor(
            {this->max_token * this->local_world_size, this->n_dim}, data_type);
      }
    }
    this->buffer_initialized = true;
  }

 public:
  DisScatterBackwardOpImpl(
      int64_t total_num_experts,
      int64_t max_token,
      int64_t n_dim,
      int64_t topk,
      int64_t rank,
      int64_t tp_world_size,
      int64_t ep_world_size,
      int64_t local_world_size,
      float moe_capacity_ratio,
      int nvshmem_team_)
      : total_num_experts(total_num_experts),
        max_token(max_token),
        n_dim(n_dim),
        topk(topk),
        rank(rank),
        buffer_initialized(false),
        tp_world_size(tp_world_size),
        ep_world_size(ep_world_size),
        local_world_size(local_world_size),
        scale_factor(moe_capacity_ratio),
        nvshmem_team((nvshmem_team_t)nvshmem_team_) {
    // ep world size is the global world size
    // tp_world_size: the tp/sp world size of the attn
    FLUX_CHECK(total_num_experts % ep_world_size == 0);
    FLUX_CHECK(ep_world_size % tp_world_size == 0);
    FLUX_CHECK(ep_world_size % local_world_size == 0);
    FLUX_CHECK(local_world_size % tp_world_size == 0);
    this->n_nodes = ep_world_size / local_world_size;
    FLUX_CHECK(this->n_nodes < kMaxNodes);
    FLUX_CHECK(this->total_num_experts % this->n_nodes == 0);
    this->node_id = rank / local_world_size;
    this->dp_world_size = ep_world_size / tp_world_size;
    FLUX_CHECK(rank < ep_world_size);
    this->dp_rank = rank / tp_world_size;
    this->ep_token_cum_sum.resize(this->ep_world_size + 1, 0);
    this->ep_token_cum_sum[0] = 0;
    this->splits_cum_sum.resize(this->total_num_experts + 1, 0);
    this->splits_cum_sum[0] = 0;
    auto gpu_tensor_option = at::TensorOptions(at::ScalarType::Int)
                                 .device(torch::kCUDA)
                                 .device_index(at::cuda::current_device());
    this->ep_token_cum_sum_gpu = torch::zeros({this->ep_world_size + 1}, gpu_tensor_option);
    this->splits_cum_sum_gpu = torch::zeros({this->total_num_experts + 1}, gpu_tensor_option);
    this->splits_cum_sum_gpu_per_rank = torch::zeros({this->ep_world_size + 1}, gpu_tensor_option);
  }

  DisScatterBackwardBuildIndexParams
  preprocess_index_multinodes(
      torch::Tensor input,
      torch::Tensor ag_exp_indices,
      torch::Tensor ag_scatter_idx,
      torch::Tensor token_counts_send,
      torch::Tensor index_tensor,
      bool splits_on_gpu) {
    int32_t n_tokens_total = ag_exp_indices.numel() / this->topk;
    int32_t *ag_exp_indices_ptr = ag_exp_indices.data_ptr<int32_t>();
    int32_t *ag_scatter_idx_ptr = ag_scatter_idx.data_ptr<int32_t>();
    int32_t *block_count_send = token_counts_send.data_ptr<int32_t>();
    int32_t *block_idx_send = index_tensor.data_ptr<int32_t>();
    int32_t *block_n_tokens = block_idx_send + this->n_nodes * this->max_token;
    int32_t *token_idx_send = block_n_tokens + this->n_nodes * this->max_token;
    int32_t *token_topk_count = token_idx_send + this->n_nodes * this->max_token;
    int32_t *token_scatterd_pos = token_topk_count + this->n_nodes * this->max_token;
    int32_t *token_scatterd_local_rank =
        token_scatterd_pos + this->n_nodes * this->max_token * this->topk;
    int32_t *reduce_token_index =
        token_scatterd_local_rank + this->n_nodes * this->max_token * this->topk;
    DisScatterBackwardBuildIndexParams args{
        ag_exp_indices_ptr,
        ag_scatter_idx_ptr,
        this->total_num_experts,
        this->rank,
        this->ep_world_size,
        this->local_world_size,
        this->n_nodes,
        this->max_token,
        this->topk,
        this->node_id,
        {0},
        {0},
        {0},
        /*ep_cum_sum_gpu_ptr=*/nullptr,
        /*ep_token_cum_sum_gpu_ptr=*/nullptr,
        block_count_send,
        block_idx_send,
        block_n_tokens,
        token_idx_send,
        token_topk_count,
        token_scatterd_pos,
        token_scatterd_local_rank,
        reduce_token_index};
    int experts_per_rank = this->total_num_experts / this->ep_world_size;
    int local_rank = this->rank % this->local_world_size;
    FLUX_CHECK(kMaxNodes > this->n_nodes);
    if (splits_on_gpu) {
      args.ep_cum_sum_gpu_ptr = this->splits_cum_sum_gpu_per_rank.data_ptr<int32_t>();
      args.ep_token_cum_sum_gpu_ptr = this->ep_token_cum_sum_gpu.data_ptr<int32_t>();
    } else {
      FLUX_CHECK(args.ep_cum_sum_gpu_ptr == nullptr);
      FLUX_CHECK(args.ep_token_cum_sum_gpu_ptr == nullptr);
      for (int n = 0; n < this->n_nodes; n++) {
        int tmp_rank = local_rank + n * this->local_world_size;
        FLUX_CHECK(tmp_rank < this->ep_world_size)
            << "rank:" << tmp_rank << " larger than worldsize: " << this->ep_world_size << "\n";
        args.global_token_start[n] = this->ep_token_cum_sum[tmp_rank];
        args.global_token_end[n] = this->ep_token_cum_sum[tmp_rank + 1];
      }
      for (int i = 0; i < this->local_world_size; i++) {
        int target_rank = i + this->node_id * this->local_world_size;
        args.ep_cum_sum[i] = this->splits_cum_sum[target_rank * experts_per_rank];
      }
    }
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    dis_scatter_backward_build_index_impl(args, stream);
    return args;
  }

  DisScatterBackwardBuildIndexSingleNodeParams
  preprocess_index_singlenode(
      torch::Tensor input,
      torch::Tensor ag_exp_indices,
      torch::Tensor ag_scatter_idx,
      torch::Tensor token_counts_send,
      torch::Tensor index_tensor,
      bool splits_on_gpu) {
    int32_t n_tokens_total = ag_exp_indices.numel() / this->topk;
    int32_t *ag_exp_indices_ptr = ag_exp_indices.data_ptr<int32_t>();
    int32_t *ag_scatter_idx_ptr = ag_scatter_idx.data_ptr<int32_t>();
    int32_t *token_count_send_ptr = token_counts_send.data_ptr<int32_t>();
    int32_t *token_idx_send_ptr = index_tensor.data_ptr<int32_t>();
    int32_t *token_local_reduce_pos_ptr =
        token_idx_send_ptr + this->local_world_size * this->max_token;
    int32_t *reduce_token_idx_ptr =
        token_local_reduce_pos_ptr + this->local_world_size * this->max_token * (this->topk + 1);
    DisScatterBackwardBuildIndexSingleNodeParams args{
        ag_exp_indices_ptr,
        ag_scatter_idx_ptr,
        this->total_num_experts,
        this->rank,
        this->ep_world_size,
        this->local_world_size,
        this->max_token,
        this->topk,
        {0},
        {0},
        {0},
        /*ep_cum_sum_gpu_ptr=*/nullptr,
        /*ep_token_cum_sum_gpu_ptr=*/nullptr,
        token_count_send_ptr,
        token_idx_send_ptr,
        token_local_reduce_pos_ptr,
        reduce_token_idx_ptr};
    int experts_per_rank = this->total_num_experts / this->ep_world_size;
    FLUX_CHECK(this->ep_world_size == this->local_world_size);
    if (splits_on_gpu) {
      args.ep_cum_sum_gpu_ptr = this->splits_cum_sum_gpu_per_rank.data_ptr<int32_t>();
      args.ep_token_cum_sum_gpu_ptr = this->ep_token_cum_sum_gpu.data_ptr<int32_t>();
    } else {
      FLUX_CHECK(args.ep_cum_sum_gpu_ptr == nullptr);
      FLUX_CHECK(args.ep_token_cum_sum_gpu_ptr == nullptr);
      for (int rank = 0; rank < this->ep_world_size; rank++) {
        args.global_token_start[rank] = this->ep_token_cum_sum[rank];
        args.global_token_end[rank] = this->ep_token_cum_sum[rank + 1];
        args.ep_cum_sum[rank] = this->splits_cum_sum[rank * experts_per_rank];
      }
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    dis_scatter_backward_build_index_single_node_impl(args, stream);
    return args;
  }

  void
  local_barrier(cudaStream_t stream) {
    int local_rank = this->rank % this->local_world_size;
    cudaipc_barrier_all_on_stream_impl(
        stream, this->local_barrier_ptrs.data(), local_rank, this->local_world_size, false);
  }

  torch::Tensor
  all_to_all_combine_multinodes(
      torch::Tensor input,
      c10::optional<torch::Tensor> optional_result,
      DisScatterBackwardBuildIndexParams index_args,
      int sm_margin,
      bool insert_extra_barrier,
      bool splits_on_gpu,
      bool copy_input_to_comm_buffer) {
    int element_size = torch::elementSize(input.scalar_type());
    int local_rank = this->rank % this->local_world_size;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    uint8_t *shared_input_ptr = reinterpret_cast<uint8_t *>(this->internal_buffer.data_ptr());
    torch::Tensor result;
    int32_t m_token_cur_rank;
    if (optional_result.has_value()) {
      FLUX_CHECK(splits_on_gpu);
      FLUX_CHECK(optional_result->is_contiguous());
      result = optional_result.value();
      CHECK_NDIM(result, 2);
      FLUX_CHECK(result.size(1) == input.size(1));
      FLUX_CHECK(result.device().is_cuda());
      m_token_cur_rank = result.size(0);
    } else {
      FLUX_CHECK(!splits_on_gpu);
      m_token_cur_rank =
          this->ep_token_cum_sum[this->rank + 1] - this->ep_token_cum_sum[this->rank];
      result = torch::empty({m_token_cur_rank, input.size(1)}, input.options());
    }
    if (copy_input_to_comm_buffer) {
      // if already called at the python level, no need to call it again
      // copy to the nvshmem buffer
      CUDA_CHECK(cudaMemcpyAsync(
          shared_input_ptr,
          input.data_ptr(),
          input.numel() * element_size,
          cudaMemcpyDeviceToDevice,
          stream));
      // perform the local barrier within the node
      local_barrier(stream);
    }
    int global_rank_offset = nvshmem_team_translate_pe(this->nvshmem_team, 0, NVSHMEM_TEAM_WORLD);
    // perform the all_to_all_combine here
    DisScatterBackwardParams args{
        {nullptr},
        {nullptr},
        this->reduction_buffer.data_ptr(),
        result.data_ptr(),
        input.size(1),
        sm_margin,
        global_rank_offset,
        index_args};
    // TODO: should be perform only once
    for (int i = 0; i < this->local_world_size; i++) {
      int target_rank = i + this->node_id * this->local_world_size;
      if (i == local_rank) {
        args.internal_ptrs[i] = this->internal_buffer.data_ptr();
        args.output_ptrs[i] = this->output_buffer.data_ptr();
      } else {
        int peer_rank_in_team = i + this->rank / this->local_world_size * this->local_world_size;
        int peer_rank_global =
            nvshmem_team_translate_pe(this->nvshmem_team, peer_rank_in_team, NVSHMEM_TEAM_WORLD);
        args.internal_ptrs[i] = nvshmem_ptr(this->internal_buffer.data_ptr(), peer_rank_global);
        args.output_ptrs[i] = nvshmem_ptr(this->output_buffer.data_ptr(), peer_rank_global);
      }
    }
    dis_scatter_backward_impl(args, stream);
    nvshmemx_barrier_on_stream(this->nvshmem_team, stream);
    topk_reduce_impl(
        this->output_buffer.data_ptr(),
        result.data_ptr(),
        index_args.reduce_token_idx,
        m_token_cur_rank,
        input.size(1),
        this->topk,
        stream);
    if (insert_extra_barrier) {
      nvshmemx_barrier_on_stream(this->nvshmem_team, stream);
    }
    return result;
    // perform the customized reduction here
  }

  torch::Tensor
  all_to_all_combine_singlenode(
      torch::Tensor input,
      c10::optional<torch::Tensor> optional_result,
      DisScatterBackwardBuildIndexSingleNodeParams index_args,
      int sm_margin,
      bool insert_extra_barrier,
      bool splits_on_gpu) {
    int element_size = torch::elementSize(input.scalar_type());
    int local_rank = this->rank % this->local_world_size;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    FLUX_CHECK(this->output_buffer.defined());
    torch::Tensor result;
    int32_t m_token_cur_rank;
    if (optional_result.has_value()) {
      FLUX_CHECK(splits_on_gpu);
      FLUX_CHECK(optional_result->is_contiguous());
      result = optional_result.value();
      CHECK_NDIM(result, 2);
      FLUX_CHECK(result.size(1) == input.size(1));
      FLUX_CHECK(result.device().is_cuda());
      m_token_cur_rank = result.size(0);
    } else {
      FLUX_CHECK(!splits_on_gpu);
      m_token_cur_rank =
          this->ep_token_cum_sum[this->rank + 1] - this->ep_token_cum_sum[this->rank];
      result = torch::empty({m_token_cur_rank, input.size(1)}, input.options());
    }
    int global_rank_offset = nvshmem_team_translate_pe(this->nvshmem_team, 0, NVSHMEM_TEAM_WORLD);
    // perform the all_to_all_combine here
    DisScatterBackwardSingleNodeParams args{
        input.data_ptr(),
        {nullptr},
        result.data_ptr(),
        input.size(1),
        sm_margin,
        global_rank_offset,
        index_args};
    for (int target_rank = 0; target_rank < this->ep_world_size; target_rank++) {
      if (target_rank == this->rank) {
        args.output_ptrs[target_rank] = this->output_buffer.data_ptr();
      } else {
        int global_target_rank =
            nvshmem_team_translate_pe(this->nvshmem_team, target_rank, NVSHMEM_TEAM_WORLD);
        args.output_ptrs[target_rank] =
            nvshmem_ptr(this->output_buffer.data_ptr(), global_target_rank);
        FLUX_CHECK(args.output_ptrs[target_rank] != nullptr);
      }
    }
    dis_scatter_backward_single_node_impl(args, stream);
    nvshmemx_barrier_on_stream(this->nvshmem_team, stream);
    topk_reduce_impl(
        this->output_buffer.data_ptr(),
        result.data_ptr(),
        index_args.reduce_token_idx,
        m_token_cur_rank,
        input.size(1),
        this->topk,
        stream);
    if (insert_extra_barrier) {
      nvshmemx_barrier_on_stream(this->nvshmem_team, stream);
    }
    return result;
    // perform the customized reduction here
  }

  void
  check_inputs(
      torch::Tensor input,
      torch::Tensor ag_exp_indices,
      torch::Tensor ag_scatter_idx,
      torch::Tensor splits,
      torch::Tensor ep_token_counts,
      bool spilts_on_gpu) {
    // is_contiguous assert
    FLUX_CHECK(input.is_contiguous());
    FLUX_CHECK(ag_exp_indices.is_contiguous());
    FLUX_CHECK(ag_scatter_idx.is_contiguous());
    FLUX_CHECK(splits.is_contiguous());
    FLUX_CHECK(ep_token_counts.is_contiguous());
    // shape && dim assert
    FLUX_CHECK(input.dim() == 2);
    FLUX_CHECK(ag_exp_indices.dim() == 1);
    FLUX_CHECK(ag_scatter_idx.dim() == 1);
    FLUX_CHECK(splits.dim() == 1);
    FLUX_CHECK(ep_token_counts.dim() == 1);
    FLUX_CHECK(ag_scatter_idx.size(0) == ag_exp_indices.size(0));
    FLUX_CHECK(ag_scatter_idx.size(0) % this->topk == 0);
    FLUX_CHECK(
        splits.size(0) == this->total_num_experts ||
        splits.size(0) == this->total_num_experts + 1);
    FLUX_CHECK(ep_token_counts.size(0) == this->ep_world_size);
    // device assert
    FLUX_CHECK(input.device().is_cuda());
    FLUX_CHECK(ag_exp_indices.device().is_cuda());
    FLUX_CHECK(ag_scatter_idx.device().is_cuda());
    if (spilts_on_gpu) {
      FLUX_CHECK(splits.device().is_cuda());
      FLUX_CHECK(ep_token_counts.device().is_cuda());
    } else {
      FLUX_CHECK(splits.device().is_cpu());
      FLUX_CHECK(ep_token_counts.device().is_cpu());
    }
  }

  torch::Tensor
  forward_impl(
      torch::Tensor input,
      torch::Tensor ag_exp_indices,   //  all ep ranks has the same ag_exp_indices
      torch::Tensor ag_scatter_idx,   //  all ep ranks has difference ag_scatter_idx(due to atomic)
      torch::Tensor splits_cpu,       // all ep ranks has the same splits_cpu
      torch::Tensor ep_token_counts,  // the number of the input tokens for each ep rank
      int sm_margin,
      bool insert_extra_barrier,
      bool copy_input_to_comm_buffer,
      c10::optional<UnifiedGemmHParams> const &hparams) {
    check_inputs(input, ag_exp_indices, ag_scatter_idx, splits_cpu, ep_token_counts, false);
    int n_token_total = ag_exp_indices.numel() / this->topk;
    for (int ep_rank = 0; ep_rank < this->ep_world_size; ep_rank++) {
      this->ep_token_cum_sum[ep_rank + 1] =
          reinterpret_cast<int32_t *>(ep_token_counts.data_ptr())[ep_rank] +
          this->ep_token_cum_sum[ep_rank];
    }
    for (int expert = 0; expert < this->total_num_experts; expert++) {
      this->splits_cum_sum[expert + 1] =
          reinterpret_cast<int32_t *>(splits_cpu.data_ptr())[expert] +
          this->splits_cum_sum[expert];
    }
    // initialize the nvshmem comm buffer here
    if (!this->buffer_initialized) {
      // calculate the avg token per ep rank:
      // avg_token = this->max_token * dp_world_size * topk / this->ep_world_size
      // recommend to set the initial size of output_buffer to be two times of the avg_token

      int avg_token_per_rank_expected =
          this->max_token * this->dp_world_size * this->topk / this->ep_world_size;
      // size alignd to 1024
      int inter_buffer_m =
          (int(avg_token_per_rank_expected * this->scale_factor) + 1024 - 1) / 1024 * 1024;
      this->init_buffer(input, inter_buffer_m);
    }
    int need_scale_buffer_flag = 0;
    int new_internal_buffer_m = 0;
    int experts_per_rank = this->total_num_experts / this->ep_world_size;
    for (int ep_rank = 0; ep_rank < this->ep_world_size; ep_rank++) {
      int expert_id_start = ep_rank * experts_per_rank;
      int expert_id_end = expert_id_start + experts_per_rank;
      int internal_m_size =
          this->splits_cum_sum[expert_id_end] - this->splits_cum_sum[expert_id_start];
      if (internal_m_size > this->internal_buffer.size(0)) {
        need_scale_buffer_flag = 1;
        new_internal_buffer_m = std::max(new_internal_buffer_m, internal_m_size);
      }
      if (ep_rank == this->rank) {
        // check whether the splits info is correct
        FLUX_CHECK(internal_m_size == input.size(0));
      }
    }
    if (need_scale_buffer_flag && this->n_nodes > 1) {
      // rescale the internal nvshmem buffer here when there are multiple nodes
      int aligned_m_size = (new_internal_buffer_m + 1024 - 1) / 1024 * 1024;
      this->init_buffer(input, aligned_m_size);
      cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
      nvshmemx_barrier_on_stream(this->nvshmem_team, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    this->token_counts_send.zero_();
    torch::Tensor result;
    if (this->n_nodes > 1) {
      auto index_args = preprocess_index_multinodes(
          input,
          ag_exp_indices,
          ag_scatter_idx,
          this->token_counts_send,
          this->all2all_index_tensor,
          false);
      result = all_to_all_combine_multinodes(
          input,
          c10::nullopt,
          index_args,
          sm_margin,
          insert_extra_barrier,
          false,
          copy_input_to_comm_buffer);
    } else {
      auto index_args = preprocess_index_singlenode(
          input,
          ag_exp_indices,
          ag_scatter_idx,
          this->token_counts_send,
          this->all2all_index_tensor,
          false);
      result = all_to_all_combine_singlenode(
          input, c10::nullopt, index_args, sm_margin, insert_extra_barrier, false);
    }
    auto cur_stream = c10::cuda::getCurrentCUDAStream();
    auto default_stream = c10::cuda::getDefaultCUDAStream();

    result.record_stream(cur_stream);
    input.record_stream(cur_stream);
    ag_exp_indices.record_stream(cur_stream);
    ag_scatter_idx.record_stream(cur_stream);

    result.record_stream(default_stream);
    input.record_stream(default_stream);
    ag_exp_indices.record_stream(default_stream);
    ag_scatter_idx.record_stream(default_stream);
    return result;
    // perform local reduction here
    // return this->output_buffer;
  }

  torch::Tensor
  forward_gpu_impl(
      torch::Tensor input,
      torch::Tensor ag_exp_indices,   //  all ep ranks has the same ag_exp_indices
      torch::Tensor ag_scatter_idx,   //  all ep ranks has difference ag_scatter_idx(due to atomic)
      torch::Tensor splits,           // all ep ranks has the same splits
      torch::Tensor ep_token_counts,  // the number of the input tokens for each ep rank
      torch::Tensor output,
      int sm_margin,
      bool insert_extra_barrier,
      bool copy_input_to_comm_buffer,
      c10::optional<UnifiedGemmHParams> const &hparams) {
    check_inputs(input, ag_exp_indices, ag_scatter_idx, splits, ep_token_counts, true);
    auto cur_stream = c10::cuda::getCurrentCUDAStream();
    shape_info_cum_sum_impl(
        reinterpret_cast<int32_t *>(splits.data_ptr()),
        reinterpret_cast<int32_t *>(ep_token_counts.data_ptr()),
        reinterpret_cast<int32_t *>(this->splits_cum_sum_gpu.data_ptr()),
        reinterpret_cast<int32_t *>(this->splits_cum_sum_gpu_per_rank.data_ptr()),
        reinterpret_cast<int32_t *>(this->ep_token_cum_sum_gpu.data_ptr()),
        this->total_num_experts,
        this->rank,
        this->ep_world_size,
        this->local_world_size,
        cur_stream);

    // initialize the nvshmem comm buffer here
    if (!this->buffer_initialized) {
      int avg_token_per_rank_expected =
          this->max_token * this->dp_world_size * this->topk / this->ep_world_size;
      // size alignd to 1024
      int inter_buffer_m =
          (int(avg_token_per_rank_expected * this->scale_factor) + 1024 - 1) / 1024 * 1024;
      this->init_buffer(input, inter_buffer_m);
    }

    this->token_counts_send.zero_();
    if (this->n_nodes > 1) {
      auto index_args = preprocess_index_multinodes(
          input,
          ag_exp_indices,
          ag_scatter_idx,
          this->token_counts_send,
          this->all2all_index_tensor,
          true);
      output = all_to_all_combine_multinodes(
          input,
          output,
          index_args,
          sm_margin,
          insert_extra_barrier,
          true,
          copy_input_to_comm_buffer);
    } else {
      auto index_args = preprocess_index_singlenode(
          input,
          ag_exp_indices,
          ag_scatter_idx,
          this->token_counts_send,
          this->all2all_index_tensor,
          true);
      output = all_to_all_combine_singlenode(
          input, output, index_args, sm_margin, insert_extra_barrier, true);
    }
    auto default_stream = c10::cuda::getDefaultCUDAStream();

    output.record_stream(cur_stream);
    input.record_stream(cur_stream);
    ag_exp_indices.record_stream(cur_stream);
    ag_scatter_idx.record_stream(cur_stream);
    splits.record_stream(cur_stream);
    ep_token_counts.record_stream(cur_stream);

    output.record_stream(default_stream);
    input.record_stream(default_stream);
    ag_exp_indices.record_stream(default_stream);
    ag_scatter_idx.record_stream(default_stream);
    splits.record_stream(default_stream);
    ep_token_counts.record_stream(default_stream);

    return output;
    // perform local reduction here
    // return this->output_buffer;
  }

  void
  copy_to_input_comm_buffer(torch::Tensor input) {
    if (!this->buffer_initialized) {
      int avg_token_per_rank_expected =
          this->max_token * this->dp_world_size * this->topk / this->ep_world_size;
      // size alignd to 1024
      int inter_buffer_m =
          (int(avg_token_per_rank_expected * this->scale_factor) + 1024 - 1) / 1024 * 1024;
      this->init_buffer(input, inter_buffer_m);
    }
    if (this->n_nodes > 1) {
      cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
      // only enabled when multi-nodes implementation is activated
      uint8_t *shared_input_ptr = reinterpret_cast<uint8_t *>(this->internal_buffer.data_ptr());
      FLUX_CHECK(input.nbytes() <= this->internal_buffer.nbytes());
      int element_size = torch::elementSize(input.scalar_type());
      CUDA_CHECK(cudaMemcpyAsync(
          shared_input_ptr,
          input.data_ptr(),
          input.numel() * element_size,
          cudaMemcpyDeviceToDevice,
          stream));
      // perform the local barrier within the node
      // local_barrier(stream);
    }
  }

  void
  ep_barrier_all() {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    nvshmemx_barrier_on_stream(this->nvshmem_team, stream);
  }
  void
  local_barrier_all() {
    FLUX_CHECK(this->buffer_initialized) << "The internal buffer not initialized!, please call "
                                            "forward/copy_to_input_comm_buffer first";
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    if (this->n_nodes > 1) {
      local_barrier(stream);
    } else {
      // same as nvshmemx_barrier
      nvshmemx_barrier_on_stream(this->nvshmem_team, stream);
    }
  }

  torch::Tensor
  get_input_comm_buffer(std::vector<int64_t> shape, at::ScalarType dtype) {
    if (!this->buffer_initialized) {
      auto current_device = c10::cuda::current_device();
      auto option_gpu =
          at::TensorOptions().dtype(dtype).device(at::kCUDA).device_index(current_device);
      torch::Tensor example_tensor = torch::empty(shape, option_gpu);
      int avg_token_per_rank_expected =
          this->max_token * this->dp_world_size * this->topk / this->ep_world_size;
      // size alignd to 1024
      int inter_buffer_m =
          (int(avg_token_per_rank_expected * this->scale_factor) + 1024 - 1) / 1024 * 1024;
      this->init_buffer(example_tensor, inter_buffer_m);
    }
    FLUX_CHECK(shape[0] <= this->internal_buffer.size(0));
    FLUX_CHECK(shape[1] == this->internal_buffer.size(1));
    return this->internal_buffer.slice(0, 0, shape[0]);
  }

  torch::Tensor
  forward(
      torch::Tensor input,
      torch::Tensor ag_exp_indices,
      torch::Tensor ag_scatter_idx,
      torch::Tensor splits_cpu,
      torch::Tensor ep_token_counts,
      int sm_margin,
      bool insert_extra_barrier) {
    return this->forward_impl(
        input,
        ag_exp_indices,
        ag_scatter_idx,
        splits_cpu,
        ep_token_counts,
        sm_margin,
        insert_extra_barrier,
        true,
        c10::nullopt);
  }

  torch::Tensor
  forward_gpu(
      torch::Tensor input,
      torch::Tensor ag_exp_indices,
      torch::Tensor ag_scatter_idx,
      torch::Tensor splits,
      torch::Tensor ep_token_counts,
      torch::Tensor output,
      int sm_margin,
      bool insert_extra_barrier) {
    return this->forward_gpu_impl(
        input,
        ag_exp_indices,
        ag_scatter_idx,
        splits,
        ep_token_counts,
        output,
        sm_margin,
        insert_extra_barrier,
        true,
        c10::nullopt);
  }

  torch::Tensor
  forward_no_cpy(
      torch::Tensor input,
      torch::Tensor ag_exp_indices,
      torch::Tensor ag_scatter_idx,
      torch::Tensor splits_cpu,
      torch::Tensor ep_token_counts,
      int sm_margin,
      bool insert_extra_barrier) {
    return this->forward_impl(
        input,
        ag_exp_indices,
        ag_scatter_idx,
        splits_cpu,
        ep_token_counts,
        sm_margin,
        insert_extra_barrier,
        false,
        c10::nullopt);
  }

  torch::Tensor
  forward_gpu_no_cpy(
      torch::Tensor input,
      torch::Tensor ag_exp_indices,
      torch::Tensor ag_scatter_idx,
      torch::Tensor splits,
      torch::Tensor ep_token_counts,
      torch::Tensor output,
      int sm_margin,
      bool insert_extra_barrier) {
    return this->forward_gpu_impl(
        input,
        ag_exp_indices,
        ag_scatter_idx,
        splits,
        ep_token_counts,
        output,
        sm_margin,
        insert_extra_barrier,
        false,
        c10::nullopt);
  }

  torch::Tensor
  profiling(
      torch::Tensor input,
      torch::Tensor ag_exp_indices,
      torch::Tensor ag_scatter_idx,
      torch::Tensor splits_cpu,
      torch::Tensor ep_token_counts,
      int sm_margin,
      c10::intrusive_ptr<ProfilingContext> opt_ctx) {}

  std::tuple<int64_t, int64_t, int64_t>
  get_pickle_info() const {
    return std::make_tuple(this->max_token, this->n_dim, this->total_num_experts);
  }
};

DisScatterBackward::DisScatterBackward(
    int64_t total_num_experts,
    int64_t max_token,
    int64_t n_dim,
    int64_t topk,
    int64_t rank,
    int64_t tp_world_size,
    int64_t ep_world_size,
    int64_t local_world_size,
    float moe_capacity_ratio,
    int team)
    : impl_(new DisScatterBackwardOpImpl(
          total_num_experts,
          max_token,
          n_dim,
          topk,
          rank,
          tp_world_size,
          ep_world_size,
          local_world_size,
          moe_capacity_ratio,
          team)) {}
DisScatterBackward::~DisScatterBackward() { delete impl_; }
torch::Tensor
DisScatterBackward::forward(
    torch::Tensor input,
    torch::Tensor ag_exp_indices,
    torch::Tensor ag_scatter_idx,
    torch::Tensor splits_cpu,
    torch::Tensor ep_token_counts,
    int64_t sm_margin,
    bool insert_extra_barrier) {
  FLUX_CHECK(impl_ != nullptr) << "DisScatterBackward is not initialized";
  return impl_->forward(
      input,
      ag_exp_indices,
      ag_scatter_idx,
      splits_cpu,
      ep_token_counts,
      sm_margin,
      insert_extra_barrier);
}

torch::Tensor
DisScatterBackward::forward_gpu(
    torch::Tensor input,
    torch::Tensor ag_exp_indices,
    torch::Tensor ag_scatter_idx,
    torch::Tensor splits,
    torch::Tensor ep_token_counts,
    torch::Tensor output,
    int64_t sm_margin,
    bool insert_extra_barrier) {
  FLUX_CHECK(impl_ != nullptr) << "DisScatterBackward is not initialized";
  return impl_->forward_gpu(
      input,
      ag_exp_indices,
      ag_scatter_idx,
      splits,
      ep_token_counts,
      output,
      sm_margin,
      insert_extra_barrier);
}

torch::Tensor
DisScatterBackward::forward_no_cpy(
    torch::Tensor input,
    torch::Tensor ag_exp_indices,
    torch::Tensor ag_scatter_idx,
    torch::Tensor splits_cpu,
    torch::Tensor ep_token_counts,
    int64_t sm_margin,
    bool insert_extra_barrier) {
  FLUX_CHECK(impl_ != nullptr) << "DisScatterBackward is not initialized";
  return impl_->forward_no_cpy(
      input,
      ag_exp_indices,
      ag_scatter_idx,
      splits_cpu,
      ep_token_counts,
      sm_margin,
      insert_extra_barrier);
}

torch::Tensor
DisScatterBackward::forward_gpu_no_cpy(
    torch::Tensor input,
    torch::Tensor ag_exp_indices,
    torch::Tensor ag_scatter_idx,
    torch::Tensor splits,
    torch::Tensor ep_token_counts,
    torch::Tensor output,
    int64_t sm_margin,
    bool insert_extra_barrier) {
  FLUX_CHECK(impl_ != nullptr) << "DisScatterBackward is not initialized";
  return impl_->forward_gpu_no_cpy(
      input,
      ag_exp_indices,
      ag_scatter_idx,
      splits,
      ep_token_counts,
      output,
      sm_margin,
      insert_extra_barrier);
}
void
DisScatterBackward::copy_to_input_comm_buffer(torch::Tensor input) {
  FLUX_CHECK(impl_ != nullptr) << "DisScatterBackward is not initialized";
  impl_->copy_to_input_comm_buffer(input);
}
void
DisScatterBackward::ep_barrier_all() {
  FLUX_CHECK(impl_ != nullptr) << "DisScatterBackward is not initialized";
  impl_->ep_barrier_all();
}
void
DisScatterBackward::local_barrier_all() {
  FLUX_CHECK(impl_ != nullptr) << "DisScatterBackward is not initialized";
  impl_->local_barrier_all();
}
torch::Tensor
DisScatterBackward::get_input_comm_buffer(std::vector<int64_t> shape, at::ScalarType data_type) {
  FLUX_CHECK(impl_ != nullptr) << "DisScatterBackward is not initialized";
  return impl_->get_input_comm_buffer(shape, data_type);
}

torch::Tensor
DisScatterBackward::profiling(
    torch::Tensor input,
    torch::Tensor ag_exp_indices,
    torch::Tensor ag_scatter_idx,
    torch::Tensor splits_cpu,
    torch::Tensor ep_token_counts,
    int64_t sm_margin,
    c10::intrusive_ptr<ProfilingContext> opt_ctx) {
  FLUX_CHECK(impl_ != nullptr) << "DisScatterBackward is not initialized";
  return impl_->profiling(
      input, ag_exp_indices, ag_scatter_idx, splits_cpu, ep_token_counts, sm_margin, opt_ctx);
}

}  // namespace ths_op
}  // namespace flux
}  // namespace bytedance
