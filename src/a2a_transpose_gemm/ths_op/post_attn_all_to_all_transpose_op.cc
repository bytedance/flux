//===- post_attn_all_to_all_transpose_op.cc ----------------------- C++ ---===//
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

#include "a2a_transpose_gemm/ths_op/post_attn_all_to_all_transpose_op.h"
#include "a2a_transpose_gemm/ths_op/all_to_all_types.h"
#include <c10/cuda/CUDAFunctions.h>
#include "flux/ths_op/ths_op.h"
#include "flux/cuda/cuda_common.h"
#include "flux/utils.h"
#include "flux/ths_op/util.h"
#include <ATen/core/TensorBase.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Optional.h>
#include <cassert>

namespace bytedance::flux {

using ths_op::from_torch_dtype;
using ths_op::is_s8_torch_dtype;

namespace {
constexpr int kNumSignals = 64;  // TODO(houqi.1993) set this global
constexpr int kTileM = 256;
}  // namespace

PostAttnAllToAllTransposeOp::PostAttnAllToAllTransposeOp(
    std::shared_ptr<Group> pg_world,
    int nnodes,
    int sp_size,
    size_t bs,
    size_t num_heads,
    size_t seq,
    size_t head_dim,
    int32_t max_num_comm_buf,
    at::ScalarType input_dtype,
    bool a2a_only)
    : pg_world_(pg_world),
      nnodes_(nnodes),
      world_size_(pg_world->get_size()),
      rank_(pg_world->get_rank()),
      local_world_size_(world_size_ / nnodes),
      local_rank_(rank_ % local_world_size_),
      sp_size_(sp_size),
      sp_rank_(rank_ % sp_size),
      bs_(bs),
      num_heads_(num_heads),
      seq_(seq),
      head_dim_(head_dim),
      max_num_comm_buf_(max_num_comm_buf),
      input_dtype_(input_dtype),
      comm_output_ptrs_(sp_size, nullptr),
      barrier_ptrs_(sp_size, nullptr),
      sync_ptrs_(sp_size, nullptr),
      a2a_only_(a2a_only),
      p2p_atomic_supported_(is_p2p_atomic_supported()) {
  FLUX_CHECK_DIV(world_size_, nnodes)
      << "invalid nnodes: world_size[" << world_size_ << "] %% nnodes[" << nnodes_ << "] != 0";
  FLUX_CHECK_DIV(local_world_size_, sp_size_);
  FLUX_CHECK(max_num_comm_buf_ >= 1)
      << "max_num_comm_buf need greater than or equal to 1, got " << max_num_comm_buf_;
  create_symetric_buffers();
}

int32_t *
PostAttnAllToAllTransposeOp::a2a_signal_ptr() const {
  return this->copy_param_.a2a_signal;
}

void
PostAttnAllToAllTransposeOp::init_all2all_copy_param(
    torch::Tensor input,
    int32_t comm_buf_idx,
    int32_t num_comm_sm,
    bool use_read,
    bool skip_barrier) {
  CHECK_NDIM(input, 4);
  FLUX_CHECK(input.is_contiguous()) << "input should be contiguous.";
  FLUX_CHECK(input.size(0) <= this->bs_);
  if (!this->a2a_only_) {
    FLUX_CHECK(input.size(1) <= this->num_heads_ / this->sp_size_) << "nheads mismatch.";
    FLUX_CHECK(input.size(2) <= this->seq_) << "seq len exceeds limit.";
  } else {
    FLUX_CHECK(input.size(1) <= this->seq_) << "seq len exceeds limit.";
    FLUX_CHECK(input.size(2) <= this->num_heads_ / this->sp_size_) << "nheads mismatch.";
  }
  FLUX_CHECK(input.size(3) == this->head_dim_) << "head dim mismatch.";

  // for a2a transpose, input is [bs, local_nheads, seq_len, head_dim]
  // for a2a only, input shape is [bs, seq_len, local_nheads, head_dim]
  int32_t bs = input.size(0);
  int32_t local_nheads = this->a2a_only_ ? input.size(2) : input.size(1);
  int32_t seq_len = this->a2a_only_ ? input.size(1) : input.size(2);
  int32_t nheads = local_nheads * sp_size_;
  int32_t head_dim = input.size(3);

  FLUX_CHECK(comm_buf_idx < this->comm_output_ptrs_buffer_list_.size())
      << "comm_buf_idx out of range.";
  // init params for d2d kernel
  if (!use_read) {
    this->copy_param_.input_ptr = input.data_ptr();
    this->copy_param_.output_ptrs =
        (void **)this->comm_output_ptrs_buffer_list_[comm_buf_idx].data_ptr();
  } else {
    FLUX_CHECK(!use_read) << "all2all kernel only support write mode.";
  }

  this->copy_param_.barrier_ptrs = (void **)this->barrier_ptrs_buffer_.data_ptr();
  this->copy_param_.sync_barriers = (int32_t **)this->sync_ptrs_buffer_.data_ptr();
  this->copy_param_.bs = bs;
  this->copy_param_.nheads = nheads;
  this->copy_param_.seq_len = seq_len;
  this->copy_param_.head_dim = head_dim;
  this->copy_param_.world_size = this->sp_size_;
  this->copy_param_.rank = this->sp_rank_;
  this->copy_param_.skip_barrier = skip_barrier;
  // the last element in barrier buffer as a2a signal.
  this->copy_param_.a2a_signal =
      (int32_t *)this->barrier_ptrs_[this->sp_rank_] + this->barrier_buffer_.numel() - 1;
  this->copy_param_.TILE_M = kTileM;
  if (num_comm_sm >= sp_size_)
    this->copy_param_.num_comm_sm = num_comm_sm / sp_size_ * sp_size_;
  else
    this->copy_param_.num_comm_sm = num_comm_sm;
  FLUX_CHECK(this->copy_param_.num_comm_sm > 0);
}

void
PostAttnAllToAllTransposeOp::init_all2all_copy_param(
    torch::Tensor input,
    torch::Tensor seq_lens_cpu,
    int32_t comm_buf_idx,
    int32_t num_comm_sm,
    bool use_read,
    bool skip_barrier) {
  this->init_all2all_copy_param(input, comm_buf_idx, num_comm_sm, use_read, skip_barrier);
  FLUX_CHECK(seq_lens_cpu.device().type() == torch::kCPU) << "seq_lens is not on cpu\n";
  FLUX_CHECK(seq_lens_cpu.size(0) == this->sp_size_);
  FLUX_CHECK(seq_lens_cpu.dtype() == c10::ScalarType::Int);

  int32_t *seq_lens_ptr = seq_lens_cpu.data_ptr<int32_t>();
  this->copy_param_.cusum_seq_lens[0] = 0;
  int32_t sum = 0;
  for (int32_t i = 0; i < this->sp_size_; ++i) {
    sum += *(seq_lens_ptr + i);
    this->copy_param_.cusum_seq_lens[i + 1] = sum;
  }
}

void
PostAttnAllToAllTransposeOp::create_symetric_buffers() {
  // input buffer: [bs, s/n, nh*hd]
  int32_t local_seq = this->seq_ / this->sp_size_;
  this->comm_output_buffers_ = flux_create_tensor_list(
      {this->max_num_comm_buf_, this->bs_, local_seq, this->num_heads_ * this->head_dim_},
      this->input_dtype_,
      this->pg_world_.get(),
      /*ring_mode=*/false,
      /*init_zero=*/true);
  // flux_create_tensor_list return the symm bufs within node
  this->comm_output_buffer_ = this->comm_output_buffers_[this->local_rank_];
  int32_t sp_group_offset = this->local_rank_ / this->sp_size_ * this->sp_size_;
  for (int i = 0; i < sp_size_; ++i) {
    this->comm_output_ptrs_[i] = this->comm_output_buffers_[i + sp_group_offset].data_ptr();
  }

  size_t ptrs_buffer_size = sizeof(void *) * (this->sp_size_);
  auto ptr_buffer_options = at::TensorOptions(at::kCUDA)
                                .device_index(at::cuda::current_device())
                                .dtype(c10::ScalarType::Byte);

  // init copy comm buf ptrs to device tensor
  for (int32_t i = 0; i < this->max_num_comm_buf_; ++i) {
    torch::Tensor comm_buf_pts =
        torch::empty({ptrs_buffer_size}, at::TensorOptions(at::kCUDA).dtype(at::ScalarType::Byte));
    std::vector<void *> comm_buf_ptrs(this->sp_size_, nullptr);
    for (int32_t j = 0; j < this->sp_size_; ++j) {
      auto comm_buf = this->comm_output_buffers_[j + sp_group_offset].slice(0, i, i + 1);
      comm_buf_ptrs[j] = comm_buf.data_ptr();
      FLUX_CHECK(comm_buf_ptrs[j] != nullptr) << "ptr of symm buf is invalid.";
    }
    CUDA_CHECK(cudaMemcpy(
        comm_buf_pts.data_ptr(), comm_buf_ptrs.data(), ptrs_buffer_size, cudaMemcpyHostToDevice));
    comm_output_ptrs_buffer_list_.push_back(comm_buf_pts);
  }

  // barrier buffer
  // the last element in barrier buffer as the a2a signal
  this->barrier_buffers_ = flux_create_tensor_list(
      {this->bs_ * ((local_seq + kTileM - 1) / kTileM) + 1},
      c10::ScalarType::Int,
      this->pg_world_.get(),
      /*ring_mode=*/false,
      /*init_zero=*/true);
  this->barrier_buffer_ = this->barrier_buffers_[this->local_rank_];
  for (int i = 0; i < sp_size_; ++i) {
    this->barrier_ptrs_[i] = (int32_t *)this->barrier_buffers_[i + sp_group_offset].data_ptr();
  }

  size_t barrier_ptrs_buffer_size = sizeof(void *) * (this->sp_size_);
  this->barrier_ptrs_buffer_ =
      torch::empty({static_cast<long>(barrier_ptrs_buffer_size)}, ptr_buffer_options);

  CUDA_CHECK(cudaMemcpy(
      this->barrier_ptrs_buffer_.data_ptr(),
      this->barrier_ptrs_.data(),
      barrier_ptrs_buffer_size,
      cudaMemcpyHostToDevice));

  create_sync_buffers();
}

void
PostAttnAllToAllTransposeOp::create_sync_buffers() {
  // Each block of local copy kernel needs to be synchronized
  size_t comm_block_num = get_post_attn_all2all_transpose_block_num(this->bs_, this->seq_, kTileM);
  auto intra_node_sync_buffers = flux_create_tensor_list(
      {static_cast<long>(this->sp_size_ * comm_block_num)},
      c10::ScalarType::Int,
      this->pg_world_.get(),
      /*ring_mode=*/false,
      /*init_zero=*/true);
  int32_t sp_group_offset = this->local_rank_ / this->sp_size_ * this->sp_size_;

  size_t sync_ptrs_buffer_size = sizeof(void *) * (this->sp_size_);
  auto options = at::TensorOptions(at::kCUDA)
                     .device_index(at::cuda::current_device())
                     .dtype(c10::ScalarType::Byte);
  this->sync_ptrs_buffer_ = torch::empty({static_cast<long>(sync_ptrs_buffer_size)}, options);
  for (int i = 0; i < sp_size_; ++i) {
    this->sync_ptrs_[i] =
        reinterpret_cast<int32_t *>(intra_node_sync_buffers[i + sp_group_offset].data_ptr());
  }
  CUDA_CHECK(cudaMemcpy(
      this->sync_ptrs_buffer_.data_ptr(),
      this->sync_ptrs_.data(),
      sync_ptrs_buffer_size,
      cudaMemcpyHostToDevice));

  this->sync_buffers_.clear();
  for (int32_t i = 0; i < this->sp_size_; ++i) {
    this->sync_buffers_.push_back(intra_node_sync_buffers[i + sp_group_offset]);
  }
}

bool
PostAttnAllToAllTransposeOp::is_p2p_atomic_supported() {
  int current_device = at::cuda::current_device();
  int next = (this->local_rank_ + 1) % this->local_world_size_;
  int p2p_atomic_supported = false;
  if (current_device != next) {
    CUDA_CHECK(cudaDeviceGetP2PAttribute(
        &p2p_atomic_supported, cudaDevP2PAttrNativeAtomicSupported, current_device, next));
  }
  return p2p_atomic_supported;
}

torch::Tensor
PostAttnAllToAllTransposeOp::get_comm_result(
    int32_t comm_buf_idx,
    const AllToAllOption &opt,
    int32_t bs,
    int32_t local_seq_len,
    int32_t nheads,
    int32_t head_dim,
    cudaStream_t stream) {
  size_t nelems = (size_t)bs * local_seq_len * nheads * head_dim;
  FLUX_CHECK(comm_buf_idx < this->max_num_comm_buf_) << "comm_buf_idx out of range.";
  torch::Tensor out_buffer = local_comm_output_buffer()
                                 .reshape({this->max_num_comm_buf_, -1})
                                 .slice(0, comm_buf_idx, comm_buf_idx + 1)
                                 .reshape(-1)
                                 .slice(0, 0, nelems)
                                 .reshape({bs, local_seq_len, nheads, head_dim});
  if (opt.return_comm_buf) {
    return out_buffer;
  } else {
    torch::Tensor user_buf = ths_op::empty_with_uninitialized_data(
        std::vector<int64_t>{bs, local_seq_len, nheads, head_dim}, out_buffer.options());
    sp_group_barrier_async(stream);
    CUDA_CHECK(cudaMemcpyAsync(
        user_buf.data_ptr(),  // symm buf
        out_buffer.data_ptr(),
        user_buf.nbytes(),
        cudaMemcpyDeviceToDevice,
        stream));
    return user_buf;
  }
  return out_buffer;
}

torch::Tensor
PostAttnAllToAllTransposeOp::run(
    torch::Tensor input,
    const AllToAllOption &opt,
    int32_t comm_buf_idx,
    int32_t num_comm_sm,
    cudaStream_t stream) {
  FLUX_CHECK(opt.use_cuda_core) << "All2All only support using cuda core";
  this->init_all2all_copy_param(input, comm_buf_idx, num_comm_sm, opt.use_read, opt.skip_barrier);
  if (opt.mode == A2ARingMode::All2All) {
    if (!this->a2a_only_)
      copy_all_to_all_with_transpose(input, opt, stream);
    else
      copy_all_to_all(input, opt, stream);
  } else {
    FLUX_CHECK(false) << "opt mode not supported";
  }

  int32_t bs = input.size(0);
  int32_t local_nheads = this->a2a_only_ ? input.size(2) : input.size(1);
  int32_t seq_len = this->a2a_only_ ? input.size(1) : input.size(2);
  int32_t nheads = local_nheads * sp_size_;
  int32_t head_dim = input.size(3);
  int32_t local_seq_len = seq_len / this->sp_size_;
  return get_comm_result(comm_buf_idx, opt, bs, local_seq_len, nheads, head_dim, stream);
}

torch::Tensor
PostAttnAllToAllTransposeOp::run(
    torch::Tensor input,
    torch::Tensor seq_lens_cpu,
    const AllToAllOption &opt,
    int32_t comm_buf_idx,
    int32_t num_comm_sm,
    cudaStream_t stream) {
  FLUX_CHECK(opt.use_cuda_core) << "All2All only support using cuda core";
  this->init_all2all_copy_param(
      input, seq_lens_cpu, comm_buf_idx, num_comm_sm, opt.use_read, opt.skip_barrier);
  if (opt.mode == A2ARingMode::All2All) {
    if (this->a2a_only_)
      copy_all_to_all(input, opt, stream, /*is_dyn*/ true);
    else
      FLUX_CHECK(false) << "opt mode not supported";
  } else {
    FLUX_CHECK(false) << "opt mode not supported";
  }
  int32_t bs = input.size(0);
  int32_t local_nheads = this->a2a_only_ ? input.size(2) : input.size(1);
  int32_t nheads = local_nheads * sp_size_;
  int32_t head_dim = input.size(3);
  int32_t local_seq_len = *(seq_lens_cpu.data_ptr<int32_t>() + this->sp_rank_);
  return get_comm_result(comm_buf_idx, opt, bs, local_seq_len, nheads, head_dim, stream);
}

void
PostAttnAllToAllTransposeOp::sp_group_barrier_async(cudaStream_t stream) {
  flux_barrier_all_on_stream(
      stream,
      this->sync_buffers_,
      this->sp_rank_,
      /*ring_mode=*/false,
      /*force_flux_impl*/ true);
}

void
PostAttnAllToAllTransposeOp::copy_all_to_all_with_transpose(
    torch::Tensor input, const AllToAllOption &opt, cudaStream_t stream) {
  if (!opt.use_read) {
    if (!opt.fuse_sync)
      sp_group_barrier_async(stream);
    post_attn_a2a_transpose_impl(
        this->copy_param_,
        ths_op::from_torch_dtype(this->input_dtype_),
        opt.fuse_sync ? SyncMethod::SyncAtomic : SyncMethod::SyncNone,
        stream);
  } else {
    FLUX_CHECK(false) << "all to all does not support read mode.";
  }
}

void
PostAttnAllToAllTransposeOp::copy_all_to_all(
    torch::Tensor input, const AllToAllOption &opt, cudaStream_t stream, bool is_dyn_seq) {
  if (!opt.use_read) {
    if (!opt.fuse_sync)
      sp_group_barrier_async(stream);
    if (!is_dyn_seq)
      post_attn_a2a_impl(
          this->copy_param_,
          ths_op::from_torch_dtype(this->input_dtype_),
          opt.fuse_sync ? SyncMethod::SyncAtomic : SyncMethod::SyncNone,
          stream);
    else
      post_attn_a2a_dyn_impl(
          this->copy_param_,
          ths_op::from_torch_dtype(this->input_dtype_),
          opt.fuse_sync ? SyncMethod::SyncAtomic : SyncMethod::SyncNone,
          stream);
  } else {
    FLUX_CHECK(false) << "all to all does not support read mode.";
  }
}

void
PostAttnAllToAllTransposeOp::reset_signals(const AllToAllOption &opt, cudaStream_t stream) {
  this->barrier_buffer_.zero_();
}

}  // namespace bytedance::flux
