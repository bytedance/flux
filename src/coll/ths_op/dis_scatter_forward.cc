//===- dis_scatter_forward.cc ------------------------------------------ C++ ---===//
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
#include <coll/ths_op/dis_scatter_forward.h>
#include <cuda_runtime_api.h>
#include <cutlass/gemm/gemm.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <torch/all.h>

#include <coll/dis_scatter_forward_impl.hpp>
#include <cstdlib>
#include <cutlass/util/packed_stride.hpp>
#include <iostream>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <vector>

#include "flux/cuda/cuda_common.h"
#include "flux/flux.h"
#include "flux/ths_op/ths_op.h"
#include "flux/ths_op/flux_shm.h"
#include "flux/ths_op/util.h"
namespace bytedance {
namespace flux {
namespace ths_op {

using torch::Tensor;

/// This class only runs the basic grouped_gemm, it is mainly used for testing
class DisScatterForward::DisScatterForwardOpImpl {
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
  std::vector<torch::Tensor> output_buffer;
  std::vector<torch::Tensor> internal_buffer;
  torch::Tensor barrier_buffer;
  torch::Tensor local_barrier;
  torch::Tensor pre_comm_topk_idx;
  torch::Tensor pre_comm_topk_val;
  torch::Tensor token_counts;
  torch::Tensor all2all_dispatch_index;
  torch::Tensor tmp_local_barrier;
  // for the forward gpu only
  torch::Tensor splits_cum_sum_gpu;
  // the cum sum per at the rank-wise
  torch::Tensor splits_cum_sum_gpu_per_rank;
  torch::Tensor ep_token_cum_sum_gpu;
  std::vector<int32_t> ep_token_cum_sum;
  std::vector<int32_t> splits_cum_sum;
  bool buffer_initialized;
  bool pre_comm_buffer_initialized;
  float scale_fator;
  int duplicate_comm_buffer;
  nvshmem_team_t nvshmem_team;

  void
  init_buffer_pre_comm(torch::Tensor cur_topk_indices) {
    if (!this->pre_comm_buffer_initialized) {
      // local buffer should persists as a memeber in the class or record the stream,
      this->tmp_local_barrier =
          torch::zeros({3}, cur_topk_indices.options().dtype(c10::ScalarType::Int));
    }
    this->pre_comm_buffer_initialized = true;
  }

  void
  init_buffer(torch::Tensor input, int output_buffer_m) {
    // this is very dangerous: should call this in a symmetric way
    LOG(INFO) << "Init output buffer with M:" << output_buffer_m;
    auto data_type = input.scalar_type();
    if (!this->buffer_initialized) {
      FLUX_CHECK(this->output_buffer.empty());
      FLUX_CHECK(this->internal_buffer.empty());
      // calculate the avg token per ep rank:
      // avg_token = this->max_token * dp_world_size * topk / this->ep_world_size
      // recommend to set the initial size of output_buffer to be two times of the avg_token
      for (int comm_buffer_id = 0; comm_buffer_id < duplicate_comm_buffer; comm_buffer_id++) {
        this->output_buffer.push_back(
            nvshmem_create_tensor({output_buffer_m, this->n_dim}, data_type));
        // the internal buffer should be initilized only once
        this->internal_buffer.push_back(
            nvshmem_create_tensor({this->max_token * this->n_nodes, this->n_dim}, data_type));
      }
      this->barrier_buffer = nvshmem_create_tensor(
          {2 * this->n_nodes * ALL2ALL_DISPATCH_SPLITS}, c10::ScalarType::Long);
      this->barrier_buffer.zero_();
      // local buffer should persists as a memeber in the class or record the stream,
      this->local_barrier = torch::zeros(
          {this->n_nodes * ALL2ALL_DISPATCH_SPLITS}, input.options().dtype(at::ScalarType::Int));
      this->token_counts = torch::zeros(
          {2 * this->n_nodes * ALL2ALL_DISPATCH_SPLITS},
          input.options().dtype(at::ScalarType::Int));
      int index_tensor_size = 5 * this->n_nodes * this->max_token * ALL2ALL_DISPATCH_SPLITS;
      this->all2all_dispatch_index =
          torch::empty({index_tensor_size}, input.options().dtype(at::ScalarType::Int));
      this->ep_token_cum_sum_gpu =
          torch::zeros({this->ep_world_size + 1}, input.options().dtype(at::ScalarType::Int));
      this->splits_cum_sum_gpu =
          torch::zeros({this->total_num_experts + 2}, input.options().dtype(at::ScalarType::Int));
      this->splits_cum_sum_gpu_per_rank =
          torch::zeros({this->ep_world_size + 1}, input.options().dtype(at::ScalarType::Int));
    } else {
      // only scale the output_buffer
      FLUX_CHECK(!this->output_buffer.empty());
      FLUX_CHECK(!this->internal_buffer.empty());
      FLUX_CHECK(this->barrier_buffer.defined());
      // scale the output_buffer if necessary
      if (output_buffer_m > this->output_buffer[0].size(0)) {
        this->output_buffer.clear();
        for (int i = 0; i < this->duplicate_comm_buffer; i++) {
          this->output_buffer.push_back(
              nvshmem_create_tensor({output_buffer_m, this->n_dim}, data_type));
        }
      }
    }
    this->buffer_initialized = true;
  }

 public:
  DisScatterForwardOpImpl(
      int64_t total_num_experts,
      int64_t max_token,
      int64_t n_dim,
      int64_t topk,
      int64_t rank,
      int64_t tp_world_size,
      int64_t ep_world_size,
      int64_t local_world_size,
      float moe_capacity_ratio,
      int duplicate_comm_buffer,
      int nvshmem_team_)
      : total_num_experts(total_num_experts),
        max_token(max_token),
        n_dim(n_dim),
        topk(topk),
        rank(rank),
        buffer_initialized(false),
        pre_comm_buffer_initialized(false),
        tp_world_size(tp_world_size),
        ep_world_size(ep_world_size),
        local_world_size(local_world_size),
        scale_fator(moe_capacity_ratio),
        duplicate_comm_buffer(duplicate_comm_buffer),
        nvshmem_team((nvshmem_team_t)nvshmem_team_) {
    // ep world size is the global world size
    // tp_world_size: the tp/sp world size of the attn
    FLUX_CHECK(total_num_experts % ep_world_size == 0);
    FLUX_CHECK(ep_world_size % tp_world_size == 0);
    FLUX_CHECK(ep_world_size % local_world_size == 0);
    FLUX_CHECK(local_world_size % tp_world_size == 0);
    this->n_nodes = ep_world_size / local_world_size;
    this->node_id = rank / local_world_size;
    this->dp_world_size = ep_world_size / tp_world_size;
    FLUX_CHECK(rank < ep_world_size);
    this->dp_rank = rank / tp_world_size;
    this->ep_token_cum_sum.resize(this->ep_world_size + 1, 0);
    this->ep_token_cum_sum[0] = 0;
    this->splits_cum_sum.resize(this->total_num_experts + 1, 0);
    this->splits_cum_sum[0] = 0;
  }
  DisScatterForwardBuildIndexParams
  preprocess_index(
      torch::Tensor input,
      torch::Tensor ag_exp_indices,
      torch::Tensor ag_scatter_idx,
      torch::Tensor token_counts,
      torch::Tensor index_tensor,
      bool splits_on_gpu) {
    int32_t n_tokens_cur_ep = input.size(0);
    int32_t n_tokens_total = ag_exp_indices.numel() / this->topk;
    int32_t *token_counts_send_ptr = token_counts.data_ptr<int32_t>();
    int32_t *token_counts_recv_ptr =
        token_counts_send_ptr + this->n_nodes * ALL2ALL_DISPATCH_SPLITS;
    int32_t *block_idx_to_send_ptr = index_tensor.data_ptr<int32_t>();
    int32_t *token_count_in_block_ptr =
        block_idx_to_send_ptr + this->n_nodes * this->max_token * ALL2ALL_DISPATCH_SPLITS;
    int32_t *sorted_token_idx_send_ptr =
        token_count_in_block_ptr + this->n_nodes * this->max_token * ALL2ALL_DISPATCH_SPLITS;
    int32_t *sorted_token_idx_recv_ptr =
        sorted_token_idx_send_ptr + this->n_nodes * this->max_token * ALL2ALL_DISPATCH_SPLITS;
    int32_t *ag_exp_indices_ptr = ag_exp_indices.data_ptr<int32_t>();
    int32_t *ag_scatter_idx_ptr = ag_scatter_idx.data_ptr<int32_t>();
    DisScatterForwardBuildIndexParams args{
        ag_exp_indices_ptr,
        ag_scatter_idx_ptr,
        this->total_num_experts,
        this->n_nodes,
        n_tokens_cur_ep,
        this->max_token,
        this->topk,
        this->node_id,
        {0},  // global_token_start, to be initialized later
        {0},  // global_token_end, to be initialized later
              /*ep_token_cum_sum_gpu_ptr=*/
        nullptr,
        this->rank % this->local_world_size,
        this->local_world_size,
        // following are output params
        token_counts_send_ptr,
        block_idx_to_send_ptr,
        token_count_in_block_ptr,
        sorted_token_idx_send_ptr,
        token_counts_recv_ptr,
        sorted_token_idx_recv_ptr};
    // initialize the global token start and end
    int local_rank = this->rank % this->local_world_size;
    FLUX_CHECK(kMaxNodes > this->n_nodes);
    if (splits_on_gpu) {
      args.ep_token_cum_sum_gpu_ptr = this->ep_token_cum_sum_gpu.data_ptr<int32_t>();
      FLUX_CHECK(args.ep_token_cum_sum_gpu_ptr != nullptr);
    } else {
      FLUX_CHECK(args.ep_token_cum_sum_gpu_ptr == nullptr);
      for (int n = 0; n < this->n_nodes; n++) {
        int tmp_rank = local_rank + n * this->local_world_size;
        FLUX_CHECK(tmp_rank < this->ep_world_size);
        args.global_token_start[n] = this->ep_token_cum_sum[tmp_rank];
        args.global_token_end[n] = this->ep_token_cum_sum[tmp_rank + 1];
      }
    }
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    // dis_scatter_forward_build_index_impl(args, stream);
    dis_scatter_forward_build_index_flatten_impl(args, stream);
    return args;
  }

  void
  all_to_all_dispatch(
      torch::Tensor input,
      DisScatterForwardBuildIndexParams index_args,
      int sm_margin,
      bool splits_on_gpu,
      bool with_cp_flag,
      int comm_buffer_id) {
    int32_t *splits_cum_sum_gpu_per_rank_ptr = nullptr;
    int32_t *ep_token_cum_sum_gpu_ptr = nullptr;
    if (splits_on_gpu) {
      FLUX_CHECK(this->splits_cum_sum_gpu.defined());
      FLUX_CHECK(this->ep_token_cum_sum_gpu.defined());
      splits_cum_sum_gpu_per_rank_ptr =
          reinterpret_cast<int32_t *>(this->splits_cum_sum_gpu_per_rank.data_ptr());
      ep_token_cum_sum_gpu_ptr =
          reinterpret_cast<int32_t *>(this->ep_token_cum_sum_gpu.data_ptr());
    }
    DisScatterForwardParams args{
        {nullptr},
        {nullptr},
        {nullptr},
        {0},
        reinterpret_cast<int32_t *>(this->local_barrier.data_ptr()),
        splits_cum_sum_gpu_per_rank_ptr,
        this->ep_world_size,
        this->rank,
        this->local_world_size,
        input.size(0),
        input.size(1),
        sm_margin,
        index_args,
        this->nvshmem_team};
    int element_size = torch::elementSize(input.scalar_type());
    int local_rank = this->rank % this->local_world_size;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    if (with_cp_flag) {
      uint8_t *shared_input_ptr =
          reinterpret_cast<uint8_t *>(this->internal_buffer[comm_buffer_id].data_ptr()) +
          this->n_dim * this->node_id * this->max_token * element_size;
      // copy the input to the Symmetric address space
      CUDA_CHECK(cudaMemcpyAsync(
          shared_input_ptr, input.data_ptr(), input.nbytes(), cudaMemcpyDeviceToDevice, stream));
    }
    int experts_per_rank = this->total_num_experts / this->ep_world_size;
    for (int i = 0; i < this->local_world_size; i++) {
      int target_rank = i + this->node_id * this->local_world_size;
      if (!splits_on_gpu) {
        args.ep_cum_sum[i] = this->splits_cum_sum[target_rank * experts_per_rank];
      }
      if (i == local_rank) {
        args.internal_ptrs[i] = this->internal_buffer[comm_buffer_id].data_ptr();
        args.output_ptrs[i] = this->output_buffer[comm_buffer_id].data_ptr();
        args.barrier_ptrs[i] = this->barrier_buffer.data_ptr();
      } else {
        int peer_rank_this_team = i + this->rank / this->local_world_size * this->local_world_size;
        int peer_rank =
            nvshmem_team_translate_pe(nvshmem_team, peer_rank_this_team, NVSHMEM_TEAM_WORLD);
        args.internal_ptrs[i] =
            nvshmem_ptr(this->internal_buffer[comm_buffer_id].data_ptr(), peer_rank);
        args.output_ptrs[i] =
            nvshmem_ptr(this->output_buffer[comm_buffer_id].data_ptr(), peer_rank);
        args.barrier_ptrs[i] = nvshmem_ptr(this->barrier_buffer.data_ptr(), peer_rank);
      }
    }
    // dis_scatter_forward_impl(args, stream);
    dis_scatter_forward_flatten_impl(args, stream);
    // TODO: can we eliminate the barrier?
    // clean the barrier buffer per each forward iteration
    this->barrier_buffer.zero_();
    this->local_barrier.zero_();
    // TODO can be replace with a cuda ipc barrier within the single node
    if (with_cp_flag)
      // when no cpy flag is set, then the data copy in/out from the comm buffer
      // is exposed to the big op and will be handled manually
      nvshmemx_barrier_on_stream(this->nvshmem_team, stream);
  }
  torch::Tensor
  forward_impl(
      torch::Tensor input,
      torch::Tensor ag_exp_indices,   //  all ep ranks has the same ag_exp_indices
      torch::Tensor ag_scatter_idx,   //  all ep ranks has difference ag_scatter_idx(due to atomic)
      torch::Tensor splits_cpu,       // all ep ranks has the same splits_cpu
      torch::Tensor ep_token_counts,  // the number of the input tokens for each ep rank
      int sm_margin,
      bool copy_to_local_tensor,
      bool with_cpy_flag,
      int comm_buffer_id,
      c10::optional<UnifiedGemmHParams> const &hparams) {
    // Suppose we have 128 experts here, the drop_token is disabled and
    // the max sequence length is 1024, topk=6.
    // EP in DP = 32, SP = 2,
    // then the shape of each input tensor are:
    // per rank: splits_cpu (128)
    // per rank: exp_indices (1024/SP, 6), ag_exp_indices(1024/SP*dp_wolrd_size, 6)
    // per rank: scatter_idx (1024/SP, 6), ag_scatter_idx(1024/SP*dp_wolrd_size, 6)
    FLUX_CHECK(input.dim() == 2);
    FLUX_CHECK(input.size(0) <= this->max_token)
        << "input.size(0)" << input.size(0) << " max_token: " << this->max_token;
    FLUX_CHECK(ag_exp_indices.dim() == 1);
    FLUX_CHECK(ag_scatter_idx.dim() == 1);
    FLUX_CHECK(input.is_contiguous());
    FLUX_CHECK(ag_exp_indices.is_contiguous());
    FLUX_CHECK(ag_scatter_idx.is_contiguous());
    FLUX_CHECK(ag_scatter_idx.size(0) % this->topk == 0);
    FLUX_CHECK(ag_exp_indices.size(0) % this->topk == 0);
    FLUX_CHECK(
        splits_cpu.size(0) == this->total_num_experts ||
        splits_cpu.size(0) == this->total_num_experts + 1);
    FLUX_CHECK(ep_token_counts.size(0) == this->ep_world_size);
    // create the filter token index for different nodes
    int n_token_input = input.size(0);
    FLUX_CHECK(ag_exp_indices.numel() % this->topk == 0);
    FLUX_CHECK(ag_scatter_idx.numel() == ag_exp_indices.numel());
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
    if (!this->buffer_initialized) {
      int avg_token_per_rank_expected =
          this->max_token * this->dp_world_size * this->topk / this->ep_world_size;
      // size alignd to 1024
      int output_buffer_m =
          (int(avg_token_per_rank_expected * this->scale_fator) + 1024 - 1) / 1024 * 1024;
      this->init_buffer(input, output_buffer_m);
    }
    // dynamic scale the output buffer here according to the splits_cpu
    int need_scale_buffer_flag = 0;
    int new_output_buffer_m = 0;
    int output_buffer_m_cur_rank = -1;
    int experts_per_rank = this->total_num_experts / this->ep_world_size;
    for (int ep_rank = 0; ep_rank < this->ep_world_size; ep_rank++) {
      int expert_id_start = ep_rank * experts_per_rank;
      int expert_id_end = expert_id_start + experts_per_rank;
      int output_m_size =
          this->splits_cum_sum[expert_id_end] - this->splits_cum_sum[expert_id_start];
      if (output_m_size > this->output_buffer[comm_buffer_id].size(0)) {
        need_scale_buffer_flag = 1;
        new_output_buffer_m = std::max(new_output_buffer_m, output_m_size);
      }
      if (ep_rank == this->rank) {
        output_buffer_m_cur_rank = output_m_size;
      }
    }
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    if (need_scale_buffer_flag == 1) {
      int aligned_output_m = (new_output_buffer_m + 1023) / 1024 * 1024;
      this->init_buffer(input, aligned_output_m);
      nvshmemx_barrier_on_stream(this->nvshmem_team, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    FLUX_CHECK(this->barrier_buffer.defined());
    FLUX_CHECK(this->local_barrier.defined());

    this->token_counts.zero_();
    auto index_args = preprocess_index(
        input,
        ag_exp_indices,
        ag_scatter_idx,
        this->token_counts,
        this->all2all_dispatch_index,
        false);

    // run the all to all dispatch here
    all_to_all_dispatch(input, index_args, sm_margin, false, with_cpy_flag, comm_buffer_id);
    if (!copy_to_local_tensor) {
      return this->output_buffer[comm_buffer_id].slice(0, 0, output_buffer_m_cur_rank);
    }
    auto comm_out = this->output_buffer[comm_buffer_id].slice(0, 0, output_buffer_m_cur_rank);
    auto re = torch::empty({output_buffer_m_cur_rank, this->n_dim}, comm_out.options());
    re.copy_(comm_out);
    // TODO can be replace with a cuda ipc barrier within the single node
    nvshmemx_barrier_on_stream(this->nvshmem_team, stream);
    auto cur_stream = c10::cuda::getCurrentCUDAStream();
    auto default_stream = c10::cuda::getDefaultCUDAStream();
    input.record_stream(cur_stream);
    ag_exp_indices.record_stream(cur_stream);
    ag_scatter_idx.record_stream(cur_stream);
    re.record_stream(cur_stream);
    input.record_stream(default_stream);
    ag_exp_indices.record_stream(default_stream);
    ag_scatter_idx.record_stream(default_stream);
    re.record_stream(default_stream);
    return re;
  }

  torch::Tensor
  forward_gpu_impl(
      torch::Tensor input,
      torch::Tensor ag_exp_indices,   //  all ep ranks has the same ag_exp_indices
      torch::Tensor ag_scatter_idx,   //  all ep ranks has difference ag_scatter_idx(due to atomic)
      torch::Tensor splits,           //  all ep ranks has the same splits
      torch::Tensor ep_token_counts,  //  the number of the input tokens for each ep rank
      torch::Tensor output_tensor,
      int sm_margin,
      bool copy_to_local_tensor,
      bool with_cpy_flag,
      int comm_buffer_id,
      c10::optional<UnifiedGemmHParams> const &hparams) {
    // Suppose we have 128 experts here, the drop_token is disabled and
    // the max sequence length is 1024, topk=6.
    // EP in DP = 32, SP = 2,
    // then the shape of each input tensor are:
    // per rank: splits (128)
    // per rank: exp_indices (1024/SP, 6), ag_exp_indices(1024/SP*dp_wolrd_size, 6)
    // per rank: scatter_idx (1024/SP, 6), ag_scatter_idx(1024/SP*dp_wolrd_size, 6)
    FLUX_CHECK(splits.is_cuda());
    FLUX_CHECK(ep_token_counts.is_cuda());
    FLUX_CHECK(input.dim() == 2);
    FLUX_CHECK(input.size(0) <= this->max_token)
        << "input.size(0)" << input.size(0) << " max_token: " << this->max_token;
    FLUX_CHECK(ag_exp_indices.dim() == 1);
    FLUX_CHECK(ag_scatter_idx.dim() == 1);
    FLUX_CHECK(input.is_contiguous());
    FLUX_CHECK(ag_exp_indices.is_contiguous());
    FLUX_CHECK(ag_scatter_idx.is_contiguous());
    FLUX_CHECK(ag_scatter_idx.size(0) % this->topk == 0);
    FLUX_CHECK(ag_exp_indices.size(0) % this->topk == 0);
    FLUX_CHECK(
        splits.size(0) == this->total_num_experts ||
        splits.size(0) == this->total_num_experts + 1);
    FLUX_CHECK(ep_token_counts.size(0) == this->ep_world_size);
    // create the filter token index for different nodes
    int n_token_input = input.size(0);
    FLUX_CHECK(ag_exp_indices.numel() % this->topk == 0);
    FLUX_CHECK(ag_scatter_idx.numel() == ag_exp_indices.numel());
    int n_token_total = ag_exp_indices.numel() / this->topk;

    // init buffer before launch shape_info_cum_sum_impl
    if (!this->buffer_initialized) {
      int avg_token_per_rank_expected =
          this->max_token * this->dp_world_size * this->topk / this->ep_world_size;
      // size alignd to 1024
      int output_buffer_m =
          (int(avg_token_per_rank_expected * this->scale_fator) + 1024 - 1) / 1024 * 1024;
      this->init_buffer(input, output_buffer_m);
    }
    FLUX_CHECK(this->output_buffer[comm_buffer_id].size(0) >= output_tensor.size(0));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
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
        stream);

    // forward_gpu cannot scale the comm buffer accordingly, need to specify the
    // moe_capacity ratio at the initialization

    FLUX_CHECK(this->barrier_buffer.defined());
    FLUX_CHECK(this->local_barrier.defined());

    this->token_counts.zero_();
    auto index_args = preprocess_index(
        input,
        ag_exp_indices,
        ag_scatter_idx,
        this->token_counts,
        this->all2all_dispatch_index,
        true);

    // run the all to all dispatch here
    all_to_all_dispatch(input, index_args, sm_margin, true, with_cpy_flag, comm_buffer_id);
    // directly write the output data to the output tensor

    if (!copy_to_local_tensor) {
      return this->output_buffer[comm_buffer_id].slice(0, 0, output_tensor.size(0));
    }
    FLUX_CHECK(output_tensor.defined());
    FLUX_CHECK(output_tensor.is_contiguous());
    CHECK_NDIM(output_tensor, 2);
    size_t min_m = std::min(output_tensor.size(0), this->output_buffer[comm_buffer_id].size(0));
    auto comm_out = this->output_buffer[comm_buffer_id].slice(0, 0, min_m);
    torch::Tensor out_buf = output_tensor.slice(0, 0, min_m);
    out_buf.copy_(comm_out);
    // TODO can be replace with a cuda ipc barrier within the single node
    nvshmemx_barrier_on_stream(this->nvshmem_team, stream);
    auto cur_stream = c10::cuda::getCurrentCUDAStream();
    auto default_stream = c10::cuda::getDefaultCUDAStream();
    input.record_stream(cur_stream);
    ag_exp_indices.record_stream(cur_stream);
    ag_scatter_idx.record_stream(cur_stream);
    output_tensor.record_stream(cur_stream);
    input.record_stream(default_stream);
    ag_exp_indices.record_stream(default_stream);
    ag_scatter_idx.record_stream(default_stream);
    output_tensor.record_stream(default_stream);
    return output_tensor;
  }

  torch::Tensor
  forward_gpu(
      torch::Tensor input,
      torch::Tensor ag_exp_indices,
      torch::Tensor ag_scatter_idx,
      torch::Tensor splits,
      torch::Tensor ep_token_counts,
      torch::Tensor output_tensor,
      int sm_margin,
      bool copy_to_local_tensor) {
    return this->forward_gpu_impl(
        input,
        ag_exp_indices,
        ag_scatter_idx,
        splits,
        ep_token_counts,
        output_tensor,
        sm_margin,
        copy_to_local_tensor,
        true,
        0,
        c10::nullopt);
  }

  torch::Tensor
  forward(
      torch::Tensor input,
      torch::Tensor ag_exp_indices,
      torch::Tensor ag_scatter_idx,
      torch::Tensor splits_cpu,
      torch::Tensor ep_token_counts,
      int sm_margin,
      bool copy_to_local_tensor) {
    return this->forward_impl(
        input,
        ag_exp_indices,
        ag_scatter_idx,
        splits_cpu,
        ep_token_counts,
        sm_margin,
        copy_to_local_tensor,
        true,
        0,
        c10::nullopt);
  }
  std::vector<torch::Tensor>
  pre_comm_index(
      torch::Tensor ep_token_counts_cpu,
      torch::Tensor cur_topk_indices,
      torch::Tensor cur_topk_values,
      int sm_margin) {
    FLUX_CHECK(ep_token_counts_cpu.is_contiguous());
    FLUX_CHECK(cur_topk_indices.is_contiguous());
    FLUX_CHECK(cur_topk_values.is_contiguous());
    FLUX_CHECK(ep_token_counts_cpu.is_cpu());
    FLUX_CHECK(cur_topk_indices.is_cuda());
    FLUX_CHECK(cur_topk_values.is_cuda());
    int indice_element_size = torch::elementSize(cur_topk_indices.scalar_type());
    int values_element_size = torch::elementSize(cur_topk_values.scalar_type());
    int indice_n_dim = cur_topk_indices.size(1);
    int values_n_dim = cur_topk_values.size(1);
    FLUX_CHECK(indice_element_size <= 8);
    FLUX_CHECK(values_element_size <= 8);
    FLUX_CHECK(ep_token_counts_cpu.dtype() == at::ScalarType::Int);
    // not limit the topk_indices and values
    // FLUX_CHECK(cur_topk_indices.dtype() == at::ScalarType::Int);
    // FLUX_CHECK(cur_topk_values.dtype() == c10::ScalarType::Float);
    FLUX_CHECK(ep_token_counts_cpu.size(0) == this->ep_world_size);
    // init the comm buffer with the max datatype
    if (!this->pre_comm_topk_idx.defined()) {
      this->pre_comm_topk_idx = nvshmem_create_tensor(
          {this->ep_world_size * this->max_token * this->total_num_experts},
          c10::ScalarType::Long);
      this->pre_comm_topk_val = nvshmem_create_tensor(
          {this->ep_world_size * this->max_token * this->total_num_experts},
          c10::ScalarType::Long);
      assert(torch::elementSize(c10::ScalarType::Long) == 8);
      // printf("Init index comm buffer with Long(%dB)\n",
      // torch::elementSize(c10::ScalarType::Long));
    }
    // shape check
    FLUX_CHECK(cur_topk_indices.dim() == 2);
    FLUX_CHECK(cur_topk_values.dim() == 2);
    FLUX_CHECK(cur_topk_indices.size(1) <= this->total_num_experts);
    FLUX_CHECK(cur_topk_values.size(1) <= this->total_num_experts);
    FLUX_CHECK(cur_topk_indices.size(0) <= this->ep_world_size * this->max_token);
    FLUX_CHECK(cur_topk_values.size(0) <= this->ep_world_size * this->max_token);

    FLUX_CHECK(this->pre_comm_topk_idx.defined());
    FLUX_CHECK(this->pre_comm_topk_val.defined());
    for (int ep_rank = 0; ep_rank < this->ep_world_size; ep_rank++) {
      this->ep_token_cum_sum[ep_rank + 1] =
          reinterpret_cast<int32_t *>(ep_token_counts_cpu.data_ptr())[ep_rank] +
          this->ep_token_cum_sum[ep_rank];
    }
    this->init_buffer_pre_comm(cur_topk_indices);

    int total_tokens = this->ep_token_cum_sum[this->ep_world_size];
    torch::Tensor ag_topk_indices =
        torch::empty({total_tokens, cur_topk_indices.size(1)}, cur_topk_indices.options());
    torch::Tensor ag_topk_values =
        torch::empty({total_tokens, cur_topk_values.size(1)}, cur_topk_values.options());
    FLUX_CHECK(kMaxWorldSize >= this->ep_world_size);
    this->tmp_local_barrier.zero_();
    DisScatterPreCommIndexParams args{
        {0},
        nullptr,
        this->topk,
        this->rank,
        this->ep_world_size,
        this->total_num_experts,
        sm_margin,
        indice_element_size,
        values_element_size,
        indice_n_dim,
        values_n_dim,
        reinterpret_cast<int *>(this->tmp_local_barrier.data_ptr()),
        cur_topk_indices.data_ptr(),
        cur_topk_values.data_ptr(),
        this->pre_comm_topk_idx.data_ptr(),
        this->pre_comm_topk_val.data_ptr(),
        ag_topk_indices.data_ptr(),
        ag_topk_values.data_ptr(),
        this->nvshmem_team,
    };
    for (int i = 0; i <= this->ep_world_size; i++) {
      args.ep_cum_sum[i] = this->ep_token_cum_sum[i];
    }
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    dis_scatter_pre_comm_index_impl(args, stream);
    nvshmemx_barrier_on_stream(this->nvshmem_team, stream);
    std::vector<torch::Tensor> re({ag_topk_indices, ag_topk_values});
    auto cur_stream = c10::cuda::getCurrentCUDAStream();
    auto default_stream = c10::cuda::getDefaultCUDAStream();
    cur_topk_indices.record_stream(cur_stream);
    cur_topk_values.record_stream(cur_stream);
    ag_topk_indices.record_stream(cur_stream);
    ag_topk_values.record_stream(cur_stream);
    cur_topk_indices.record_stream(default_stream);
    cur_topk_values.record_stream(default_stream);
    ag_topk_indices.record_stream(default_stream);
    ag_topk_values.record_stream(default_stream);
    return re;
  }

  std::vector<torch::Tensor>
  pre_comm_index_gpu(
      torch::Tensor ep_token_counts_gpu,
      torch::Tensor cur_topk_indices,
      torch::Tensor cur_topk_values,
      torch::Tensor ag_topk_indices,
      torch::Tensor ag_topk_values,
      int sm_margin) {
    FLUX_CHECK(ep_token_counts_gpu.is_contiguous());
    FLUX_CHECK(cur_topk_indices.is_contiguous());
    FLUX_CHECK(cur_topk_values.is_contiguous());
    FLUX_CHECK(ep_token_counts_gpu.is_cuda());
    FLUX_CHECK(cur_topk_indices.is_cuda());
    FLUX_CHECK(cur_topk_values.is_cuda());
    int indice_element_size = torch::elementSize(cur_topk_indices.scalar_type());
    int values_element_size = torch::elementSize(cur_topk_values.scalar_type());
    int indice_n_dim = cur_topk_indices.size(1);
    int values_n_dim = cur_topk_values.size(1);
    FLUX_CHECK(indice_element_size <= 8);
    FLUX_CHECK(values_element_size <= 8);
    FLUX_CHECK(ep_token_counts_gpu.dtype() == at::ScalarType::Int);

    FLUX_CHECK(ep_token_counts_gpu.size(0) == this->ep_world_size);

    // init the comm buffer with the max datatype
    if (!this->pre_comm_topk_idx.defined()) {
      this->pre_comm_topk_idx = nvshmem_create_tensor(
          {this->ep_world_size * this->max_token * this->total_num_experts},
          c10::ScalarType::Long);
      this->pre_comm_topk_val = nvshmem_create_tensor(
          {this->ep_world_size * this->max_token * this->total_num_experts},
          c10::ScalarType::Long);
      FLUX_CHECK(torch::elementSize(c10::ScalarType::Long) == 8);
      // printf("Init index comm buffer with Long(%dB)\n",
      // torch::elementSize(c10::ScalarType::Long));
    }
    // shape check
    FLUX_CHECK(cur_topk_indices.dim() == 2);
    FLUX_CHECK(cur_topk_values.dim() == 2);
    FLUX_CHECK(cur_topk_indices.size(1) <= this->total_num_experts);
    FLUX_CHECK(cur_topk_values.size(1) <= this->total_num_experts);
    FLUX_CHECK(cur_topk_indices.size(0) <= this->ep_world_size * this->max_token);
    FLUX_CHECK(cur_topk_values.size(0) <= this->ep_world_size * this->max_token);
    FLUX_CHECK(ag_topk_indices.size(1) == cur_topk_indices.size(1));
    FLUX_CHECK(ag_topk_values.size(1) == cur_topk_values.size(1));
    if (!this->ep_token_cum_sum_gpu.defined()) {
      this->ep_token_cum_sum_gpu = torch::zeros(
          {this->ep_world_size + 1}, ep_token_counts_gpu.options().dtype(at::ScalarType::Int));
    }
    FLUX_CHECK(this->pre_comm_topk_idx.defined());
    FLUX_CHECK(this->pre_comm_topk_val.defined());
    FLUX_CHECK(this->ep_token_cum_sum_gpu.defined());
    auto cur_stream = c10::cuda::getCurrentCUDAStream();
    shape_info_cum_sum_impl(
        nullptr,
        reinterpret_cast<int32_t *>(ep_token_counts_gpu.data_ptr()),
        nullptr,
        nullptr,
        reinterpret_cast<int32_t *>(this->ep_token_cum_sum_gpu.data_ptr()),
        this->total_num_experts,
        this->rank,
        this->ep_world_size,
        this->local_world_size,
        cur_stream);
    // for (int ep_rank = 0; ep_rank < this->ep_world_size; ep_rank++) {
    //   this->ep_token_cum_sum[ep_rank + 1] =
    //       reinterpret_cast<int32_t *>(ep_token_counts_cpu.data_ptr())[ep_rank] +
    //       this->ep_token_cum_sum[ep_rank];
    // }
    this->init_buffer_pre_comm(cur_topk_indices);

    // int total_tokens = this->ep_token_cum_sum[this->ep_world_size];
    // torch::Tensor ag_topk_indices =
    //     torch::empty({total_tokens, this->topk}, cur_topk_indices.options());
    // torch::Tensor ag_topk_values =
    //     torch::empty({total_tokens, this->topk}, cur_topk_values.options());
    FLUX_CHECK(kMaxWorldSize >= this->ep_world_size);
    this->tmp_local_barrier.zero_();
    DisScatterPreCommIndexParams args{
        {0},
        reinterpret_cast<int32_t *>(this->ep_token_cum_sum_gpu.data_ptr()),
        this->topk,
        this->rank,
        this->ep_world_size,
        this->total_num_experts,
        sm_margin,
        indice_element_size,
        values_element_size,
        indice_n_dim,
        values_n_dim,
        reinterpret_cast<int *>(this->tmp_local_barrier.data_ptr()),
        cur_topk_indices.data_ptr(),
        cur_topk_values.data_ptr(),
        this->pre_comm_topk_idx.data_ptr(),
        this->pre_comm_topk_val.data_ptr(),
        ag_topk_indices.data_ptr(),
        ag_topk_values.data_ptr(),
    };
    // for (int i = 0; i <= this->ep_world_size; i++) {
    //   args.ep_cum_sum[i] = this->ep_token_cum_sum[i];
    // }
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    dis_scatter_pre_comm_index_impl(args, stream);
    nvshmemx_barrier_on_stream(this->nvshmem_team, stream);
    std::vector<torch::Tensor> re({ag_topk_indices, ag_topk_values});
    auto default_stream = c10::cuda::getDefaultCUDAStream();
    cur_topk_indices.record_stream(cur_stream);
    cur_topk_values.record_stream(cur_stream);
    ag_topk_indices.record_stream(cur_stream);
    ag_topk_values.record_stream(cur_stream);
    cur_topk_indices.record_stream(default_stream);
    cur_topk_values.record_stream(default_stream);
    ag_topk_indices.record_stream(default_stream);
    ag_topk_values.record_stream(default_stream);
    return re;
  }

  void
  copy_to_input_comm_buffer(torch::Tensor input, int comm_buffer_id) {
    if (!this->buffer_initialized) {
      int avg_token_per_rank_expected =
          this->max_token * this->dp_world_size * this->topk / this->ep_world_size;
      // size alignd to 1024
      int output_buffer_m =
          (int(avg_token_per_rank_expected * this->scale_fator) + 1024 - 1) / 1024 * 1024;
      this->init_buffer(input, output_buffer_m);
    }
    FLUX_CHECK(comm_buffer_id < duplicate_comm_buffer);
    FLUX_CHECK(input.nbytes() <= this->internal_buffer[comm_buffer_id].nbytes());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    int element_size = torch::elementSize(input.scalar_type());
    void *shared_input_ptr =
        reinterpret_cast<uint8_t *>(this->internal_buffer[comm_buffer_id].data_ptr()) +
        this->n_dim * this->node_id * this->max_token * element_size;
    CUDA_CHECK(cudaMemcpyAsync(
        shared_input_ptr, input.data_ptr(), input.nbytes(), cudaMemcpyDeviceToDevice, stream));
  }
  torch::Tensor
  copy_from_output_comm_buffer(torch::Tensor output, int comm_buffer_id) {
    FLUX_CHECK(comm_buffer_id < duplicate_comm_buffer);
    if (!this->buffer_initialized) {
      int avg_token_per_rank_expected =
          this->max_token * this->dp_world_size * this->topk / this->ep_world_size;
      // size alignd to 1024
      int output_buffer_m =
          (int(avg_token_per_rank_expected * this->scale_fator) + 1024 - 1) / 1024 * 1024;
      this->init_buffer(output, output_buffer_m);
    }
    FLUX_CHECK(output.nbytes() <= this->output_buffer[comm_buffer_id].nbytes());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    void *output_comm_buffer_ptr = this->output_buffer[comm_buffer_id].data_ptr();
    CUDA_CHECK(cudaMemcpyAsync(
        output.data_ptr(),
        output_comm_buffer_ptr,
        output.nbytes(),
        cudaMemcpyDeviceToDevice,
        stream));
    // TODO: can be removed?
    // nvshmemx_barrier_all_on_stream(stream);
    return output;
  }
  torch::Tensor
  get_input_comm_buffer(std::vector<int64_t> shape, at::ScalarType dtype, int comm_buffer_id) {
    FLUX_CHECK(comm_buffer_id < duplicate_comm_buffer);
    if (!this->buffer_initialized) {
      auto current_device = c10::cuda::current_device();
      auto option_gpu =
          at::TensorOptions().dtype(dtype).device(at::kCUDA).device_index(current_device);
      torch::Tensor example_tensor = torch::empty(shape, option_gpu);
      int avg_token_per_rank_expected =
          this->max_token * this->dp_world_size * this->topk / this->ep_world_size;
      // size alignd to 1024
      int output_buffer_m =
          (int(avg_token_per_rank_expected * this->scale_fator) + 1024 - 1) / 1024 * 1024;
      this->init_buffer(example_tensor, output_buffer_m);
    }
    FLUX_CHECK(shape[0] <= this->internal_buffer[comm_buffer_id].size(0));
    FLUX_CHECK(shape[1] == this->internal_buffer[comm_buffer_id].size(1));
    return this->internal_buffer[comm_buffer_id].slice(0, 0, shape[0]);
  }
  void
  ep_barrier_all() {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    // nvshmemx_barrier_all_on_stream(stream);
    nvshmemx_barrier_on_stream(this->nvshmem_team, stream);
  }

  torch::Tensor
  forward_gpu_no_cpy(
      torch::Tensor input,
      torch::Tensor ag_exp_indices,
      torch::Tensor ag_scatter_idx,
      torch::Tensor splits,
      torch::Tensor ep_token_counts,
      torch::Tensor output_tensor,
      int sm_margin,
      int comm_buffer_id) {
    return this->forward_gpu_impl(
        input,
        ag_exp_indices,
        ag_scatter_idx,
        splits,
        ep_token_counts,
        output_tensor,
        sm_margin,
        false,
        false,
        comm_buffer_id,
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
      int comm_buffer_id) {
    return this->forward_impl(
        input,
        ag_exp_indices,
        ag_scatter_idx,
        splits_cpu,
        ep_token_counts,
        sm_margin,
        false,
        false,
        comm_buffer_id,
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

DisScatterForward::DisScatterForward(
    int64_t total_num_experts,
    int64_t max_token,
    int64_t n_dim,
    int64_t topk,
    int64_t rank,
    int64_t tp_world_size,
    int64_t ep_world_size,
    int64_t local_world_size,
    float moe_capacity_ratio,
    int64_t duplicate_comm_buffer,
    int team)
    : impl_(new DisScatterForwardOpImpl(
          total_num_experts,
          max_token,
          n_dim,
          topk,
          rank,
          tp_world_size,
          ep_world_size,
          local_world_size,
          moe_capacity_ratio,
          duplicate_comm_buffer,
          team)) {}
DisScatterForward::~DisScatterForward() { delete impl_; }
torch::Tensor
DisScatterForward::forward(
    torch::Tensor input,
    torch::Tensor ag_exp_indices,
    torch::Tensor ag_scatter_idx,
    torch::Tensor splits_cpu,
    torch::Tensor ep_token_counts,
    int64_t sm_margin,
    bool copy_to_local_tensor) {
  FLUX_CHECK(impl_ != nullptr) << "DisScatterForward is not initialized";
  return impl_->forward(
      input,
      ag_exp_indices,
      ag_scatter_idx,
      splits_cpu,
      ep_token_counts,
      sm_margin,
      copy_to_local_tensor);
}
torch::Tensor
DisScatterForward::forward_gpu(
    torch::Tensor input,
    torch::Tensor ag_exp_indices,
    torch::Tensor ag_scatter_idx,
    torch::Tensor splits,
    torch::Tensor ep_token_counts,
    torch::Tensor output_tensor,
    int64_t sm_margin,
    bool copy_to_local_tensor) {
  FLUX_CHECK(impl_ != nullptr) << "DisScatterForward is not initialized";
  return impl_->forward_gpu(
      input,
      ag_exp_indices,
      ag_scatter_idx,
      splits,
      ep_token_counts,
      output_tensor,
      sm_margin,
      copy_to_local_tensor);
}

torch::Tensor
DisScatterForward::forward_no_cpy(
    torch::Tensor input,
    torch::Tensor ag_exp_indices,
    torch::Tensor ag_scatter_idx,
    torch::Tensor splits_cpu,
    torch::Tensor ep_token_counts,
    int64_t sm_margin,
    int64_t comm_buffer_id) {
  FLUX_CHECK(impl_ != nullptr) << "DisScatterForward is not initialized";
  return impl_->forward_no_cpy(
      input,
      ag_exp_indices,
      ag_scatter_idx,
      splits_cpu,
      ep_token_counts,
      sm_margin,
      comm_buffer_id);
}

torch::Tensor
DisScatterForward::forward_gpu_no_cpy(
    torch::Tensor input,
    torch::Tensor ag_exp_indices,
    torch::Tensor ag_scatter_idx,
    torch::Tensor splits,
    torch::Tensor ep_token_counts,
    torch::Tensor output_tensor,
    int64_t sm_margin,
    int64_t comm_buffer_id) {
  FLUX_CHECK(impl_ != nullptr) << "DisScatterForward is not initialized";
  return impl_->forward_gpu_no_cpy(
      input,
      ag_exp_indices,
      ag_scatter_idx,
      splits,
      ep_token_counts,
      output_tensor,
      sm_margin,
      comm_buffer_id);
}
torch::Tensor
DisScatterForward::copy_from_output_comm_buffer(torch::Tensor output, int64_t comm_buffer_id) {
  FLUX_CHECK(impl_ != nullptr) << "DisScatterForward is not initialized";
  return impl_->copy_from_output_comm_buffer(output, comm_buffer_id);
}
void
DisScatterForward::copy_to_input_comm_buffer(torch::Tensor input, int64_t comm_buffer_id) {
  FLUX_CHECK(impl_ != nullptr) << "DisScatterForward is not initialized";
  return impl_->copy_to_input_comm_buffer(input, comm_buffer_id);
}
torch::Tensor
DisScatterForward::get_input_comm_buffer(
    std::vector<int64_t> shape, at::ScalarType data_type, int64_t comm_buffer_id) {
  FLUX_CHECK(impl_ != nullptr) << "DisScatterForward is not initialized";
  return impl_->get_input_comm_buffer(shape, data_type, comm_buffer_id);
}
void
DisScatterForward::ep_barrier_all() {
  FLUX_CHECK(impl_ != nullptr) << "DisScatterForward is not initialized";
  return impl_->ep_barrier_all();
}

std::vector<torch::Tensor>
DisScatterForward::pre_comm_index(
    torch::Tensor ep_token_counts_cpu,
    torch::Tensor cur_topk_indices,
    torch::Tensor cur_topk_values,
    int64_t sm_margin) {
  FLUX_CHECK(impl_ != nullptr) << "DisScatterForward is not initialized";
  return impl_->pre_comm_index(ep_token_counts_cpu, cur_topk_indices, cur_topk_values, sm_margin);
}

std::vector<torch::Tensor>
DisScatterForward::pre_comm_index_gpu(
    torch::Tensor ep_token_counts_cpu,
    torch::Tensor cur_topk_indices,
    torch::Tensor cur_topk_values,
    torch::Tensor ag_topk_indices,
    torch::Tensor ag_topk_values,
    int64_t sm_margin) {
  FLUX_CHECK(impl_ != nullptr) << "DisScatterForward is not initialized";
  return impl_->pre_comm_index_gpu(
      ep_token_counts_cpu,
      cur_topk_indices,
      cur_topk_values,
      ag_topk_indices,
      ag_topk_values,
      sm_margin);
}

torch::Tensor
DisScatterForward::profiling(
    torch::Tensor input,
    torch::Tensor ag_exp_indices,
    torch::Tensor ag_scatter_idx,
    torch::Tensor splits_cpu,
    torch::Tensor ep_token_counts,
    int64_t sm_margin,
    c10::intrusive_ptr<ProfilingContext> opt_ctx) {
  FLUX_CHECK(impl_ != nullptr) << "DisScatterForward is not initialized";
  return impl_->profiling(
      input, ag_exp_indices, ag_scatter_idx, splits_cpu, ep_token_counts, sm_margin, opt_ctx);
}

}  // namespace ths_op
}  // namespace flux
}  // namespace bytedance
