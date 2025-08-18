//===- gemm_all2all_transpose.cc ---------------------------------- C++ ---===//
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

#include "gemm_a2a_transpose/ths_op/gemm_all2all_transpose.h"
#include "flux/args/gemm_a2a_transpose.h"
#include "flux/cuda/cuda_stub.h"
#include "flux/cuda/cuda_common.h"
#include "flux/flux.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_meta.h"
#include "flux/op_registry.h"
#include "flux/runtime_config.h"
#include "flux/ths_op/flux_shm.h"
#include "flux/ths_op/ths_op.h"
#include "flux/ths_op/topo_utils.h"
#include "flux/ths_op/util.h"
#include "flux/utils.h"
#include <ATen/core/jit_type.h>
#include <ATen/core/List.h>
#include <ATen/cuda/CachingHostAllocator.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/ops/empty.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/logging_is_not_google_glog.h>
#include <c10/util/Optional.h>
#include <cstdio>
#include <cuda_runtime_api.h>

#include "gemm_a2a_transpose/pre_attn_a2a_transpose_impls.hpp"
#include "gemm_a2a_transpose/pre_attn_qkv_pack_a2a_impls.hpp"

namespace bytedance::flux::ths_op {
using torch::Tensor;

namespace {
// TODO(zxg): move to utils
auto
get_gemm_meta(
    PreAttnAllToAllCommOp a2a_comm_op,
    at::ScalarType input_torch_dtype,
    at::ScalarType output_torch_dtype,
    bool transpose_weight,
    bool has_bias,
    bool fast_accum = false) {
  ArchEnum arch = get_arch();
  SMCoreEnum sm_core = get_sm_core();
  auto input_dtype = from_torch_dtype(input_torch_dtype);
  auto output_dtype = from_torch_dtype(output_torch_dtype);
  bool is_fp8_gemm = is_fp8_torch_dtype(input_torch_dtype);
  bool is_s8_gemm = is_s8_torch_dtype(input_torch_dtype);
  DataTypeEnum accum_type = is_s8_gemm ? _S32{}() : _FP32{}();
  auto dtype_config = make_gemm_dtype_config(
      input_dtype, input_dtype, has_bias ? output_dtype : _Void{}(), output_dtype, accum_type);

  auto gemm_layout = transpose_weight ? _RRR{}() : _RCR{}();
  UnifiedImplMeta impl_spec = None{};

  bool use_fast_accum = fast_accum && is_fp8_gemm;
  auto impl = ((int)arch < (int)_Sm90{}()) ? _GemmV2{}() : _GemmV3{}();
  if (impl == _GemmV2{}) {
    impl_spec = make_gemm_v2_meta(use_fast_accum);
  } else if (impl == _GemmV3{}) {
    impl_spec = make_gemm_v3_meta(use_fast_accum, /*block_scale=*/false);
  }
  auto comm_op = a2a_comm_op == PreAttnAllToAllCommOp::A2ATranspose ? _PreAttnAllToAllTranspose{}()
                                                                    : _PreAttnQKVPackAllToAll{}();
  auto meta = make_gemm_meta(dtype_config, arch, sm_core, comm_op, gemm_layout, impl, impl_spec);
  return meta;
}

RuntimeConfig
get_rt_config(int world_size, int nnodes, int m, int n, int k) {
  return make_runtime_config(
      m, n, k, make_all_to_all_transpose_runtime_config(world_size, nnodes));
}
}  // namespace
class GemmAllToAllTransposeOp::GemmAllToAllTransposeOpImpl {
 private:
  std::shared_ptr<Group> pg_world_;
  const int32_t nnodes_;
  const int32_t sp_size_;
  const int32_t bs_;
  const int32_t seq_;
  const int32_t hidden_dim_;  // nheads_input = hidden_dim_ / head_dim_
  const int32_t head_dim_;
  const int32_t n_dim_;
  const int32_t nheads_;  // for gemm output, nheads_ = n_dim_ / head_dim_
  const c10::ScalarType input_dtype_;
  const c10::ScalarType output_dtype_;
  const bool transpose_weight_;
  const int32_t gqa_;
  const PreAttnAllToAllCommOp comm_op_;
  const int32_t max_num_comm_buf_;  // for no_cpy api

 private:
  const int32_t rank_;
  const int32_t world_size_;
  const int32_t local_world_size_;
  const int32_t local_rank_;
  const int32_t sp_rank_;
  const int32_t node_idx_;

 private:
  // Symmetrically distributed tensor
  std::vector<torch::Tensor> gemm_output_buffers_;
  std::vector<torch::Tensor> barrier_buffers_;
  std::vector<torch::Tensor> sp_group_sync_buffers_;

  torch::Tensor barrier_buffer_;
  torch::Tensor gemm_output_buffer_;
  torch::Tensor gemm_workspace_buffer_;
  torch::Tensor gemm_output_ptrs_buffer_;
  std::vector<torch::Tensor> comm_buf_ptrs_buffer_list_;

  std::vector<void *> gemm_output_ptrs_;
  std::vector<void *> barrier_ptrs_;
  std::vector<int32_t> cusum_seq_lens_;

  cudaStream_t comm_stream_;
  cudaEvent_t comm_event_;
  cudaEvent_t ready_event_;

  void
  init_output_buffer() {
    int32_t local_seq = this->seq_ / this->sp_size_;
    int32_t gemm_m = this->bs_ * local_seq;
    int32_t gemm_n = this->n_dim_;
    this->gemm_output_buffers_ = flux_create_tensor_list(
        {this->max_num_comm_buf_, gemm_m, gemm_n}, this->output_dtype_, this->pg_world_.get());
    this->gemm_output_buffer_ = this->gemm_output_buffers_[this->local_rank_];

    FLUX_CHECK(this->gemm_output_ptrs_.size() == this->sp_size_);
    FLUX_CHECK(this->gemm_output_buffers_.size() == this->local_world_size_);
    int32_t sp_group_offset = this->local_rank_ / this->sp_size_ * this->sp_size_;
    for (int32_t i = 0; i < this->sp_size_; ++i) {
      this->gemm_output_ptrs_[i] = this->gemm_output_buffers_[i + sp_group_offset].data_ptr();
      FLUX_CHECK(this->gemm_output_ptrs_[i] != nullptr) << "ptr of symm buf is invalid.";
    }

    const int ptrs_buffer_bytes = sizeof(void *) * (this->sp_size_);
    this->gemm_output_ptrs_buffer_ = torch::empty(
        {ptrs_buffer_bytes}, at::TensorOptions(at::kCUDA).dtype(at::ScalarType::Byte));
    CUDA_CHECK(cudaMemcpy(
        this->gemm_output_ptrs_buffer_.data_ptr(),
        this->gemm_output_ptrs_.data(),
        ptrs_buffer_bytes,
        cudaMemcpyHostToDevice));

    // init copy comm buf ptrs to device tensor
    for (int32_t i = 0; i < this->max_num_comm_buf_; ++i) {
      torch::Tensor comm_buf_pts = torch::empty(
          {ptrs_buffer_bytes}, at::TensorOptions(at::kCUDA).dtype(at::ScalarType::Byte));
      std::vector<void *> comm_buf_ptrs(this->sp_size_, nullptr);
      for (int32_t j = 0; j < this->sp_size_; ++j) {
        auto comm_buf = this->gemm_output_buffers_[j + sp_group_offset].slice(0, i, i + 1);
        comm_buf_ptrs[j] = comm_buf.data_ptr();
        FLUX_CHECK(comm_buf_ptrs[j] != nullptr) << "ptr of symm buf is invalid.";
      }
      CUDA_CHECK(cudaMemcpy(
          comm_buf_pts.data_ptr(),
          comm_buf_ptrs.data(),
          ptrs_buffer_bytes,
          cudaMemcpyHostToDevice));
      comm_buf_ptrs_buffer_list_.push_back(comm_buf_pts);
    }
  }

  void
  init_barrier_buffer(int64_t buffer_size) {
    this->barrier_buffers_ =
        flux_create_tensor_list({buffer_size}, c10::ScalarType::Byte, this->pg_world_.get());
    this->barrier_buffer_ = this->barrier_buffers_[this->local_rank_];

    FLUX_CHECK(this->barrier_ptrs_.size() == this->sp_size_);
    FLUX_CHECK(this->barrier_buffers_.size() == this->local_world_size_);
    int32_t sp_group_offset = this->local_rank_ / this->sp_size_ * this->sp_size_;
    for (int32_t i = 0; i < this->sp_size_; ++i) {
      this->barrier_ptrs_[i] = this->barrier_buffers_[i + sp_group_offset].data_ptr();
      FLUX_CHECK(this->barrier_ptrs_[i] != nullptr) << "ptr of symm buf is invalid.";
    }
  }

  void
  init_group_sync_buffer() {
    auto intra_node_sync_buffers = flux_create_tensor_list(
        {static_cast<long>(this->sp_size_)},
        c10::ScalarType::Int,
        this->pg_world_.get(),
        /*ring_mode=*/false,
        /*init_zero=*/true);
    int32_t sp_group_offset = this->local_rank_ / this->sp_size_ * this->sp_size_;
    sp_group_sync_buffers_.clear();
    for (int32_t i = 0; i < this->sp_size_; ++i) {
      sp_group_sync_buffers_.push_back(intra_node_sync_buffers[i + sp_group_offset]);
    }
  }

  bool
  has_nvlink() {
    _ensure_topo_initialized();
    static int has_nvlink_env = get_int_from_env("FLUX_FORCE_NVLINK", -1);
    if (has_nvlink_env == -1) {
      if (topo_utils::has_nvswitch()) {
        return true;
      } else {
        return false;
      }
    }
    return has_nvlink_env;
  }

  void
  lazy_init_gemm_workspace_buffer(torch::Tensor input, int64_t buffer_size) {
    if (buffer_size <= 0) {
      return;
    }
    buffer_size = (buffer_size + 127) / 128 * 128;
    if (!this->gemm_workspace_buffer_.defined() ||
        buffer_size > this->gemm_workspace_buffer_.numel()) {
      auto options = input.options().dtype(c10::ScalarType::Byte);
      this->gemm_workspace_buffer_ = torch::empty({buffer_size}, options);
    }
  }

  // FIXME(zhengxuegui.0): Currently users need to ensure that the parameters initialized in
  // different sp groups are same.
 public:
  GemmAllToAllTransposeOpImpl(
      std::shared_ptr<Group> pg_world,
      int32_t nnodes,
      int32_t sp_size,
      int32_t bs,
      int32_t seq,
      int32_t hidden_dim,
      int32_t head_dim,
      int32_t n_dim,
      c10::ScalarType input_dtype,
      c10::ScalarType output_dtype,
      bool transpose_weight,
      int32_t gqa,
      PreAttnAllToAllCommOp comm_op,
      int32_t max_num_comm_buf)
      : pg_world_(pg_world),
        nnodes_(nnodes),
        sp_size_(sp_size),
        bs_(bs),
        seq_(seq),
        hidden_dim_(hidden_dim),
        head_dim_(head_dim),
        n_dim_(n_dim),
        nheads_(n_dim / head_dim),
        input_dtype_(input_dtype),
        output_dtype_(output_dtype),
        transpose_weight_(transpose_weight),
        gqa_(gqa),
        comm_op_(comm_op),
        max_num_comm_buf_(max_num_comm_buf),
        rank_(pg_world->get_rank()),
        world_size_(pg_world->get_size()),
        local_world_size_(world_size_ / nnodes),
        local_rank_(rank_ % local_world_size_),
        sp_rank_(local_rank_ % sp_size),
        node_idx_(rank_ / local_world_size_),
        gemm_output_ptrs_(sp_size, nullptr),
        barrier_ptrs_(sp_size, nullptr) {
    TORCH_CHECK(
        rank_ >= 0 && rank_ < world_size_,
        "invalid rank: " + std::to_string(rank_) +
            " and world_size: " + std::to_string(world_size_));
    TORCH_CHECK(
        world_size_ % nnodes_ == 0,
        "invalid nnodes: world_size[" + std::to_string(world_size_) + "] % nnodes[" +
            std::to_string(nnodes_) + "] != 0");

    this->init_output_buffer();
    this->init_group_sync_buffer();
    int32_t local_seq = this->seq_ / this->local_world_size_;
    int32_t gemm_m = this->bs_ * local_seq;
    int32_t gemm_n = this->n_dim_;
    int32_t min_tile_n = 32;
    int32_t min_tile_m = 32;
    int32_t num_tile_m = (gemm_m + min_tile_m - 1) / min_tile_m;
    int32_t num_tile_n = (gemm_n + min_tile_n - 1) / min_tile_n;
    int64_t max_barrier_workspace_size =
        (num_tile_m * num_tile_n * sizeof(int32_t) + 127) / 128 * 128;
    // the seq_len of different ranks may vary, so directly allocate the maximum barrier buffer.
    // won't cost a lot of memory
    this->init_barrier_buffer(max_barrier_workspace_size);
    CUDA_CHECK(cudaEventCreate(&comm_event_));
    CUDA_CHECK(cudaEventCreate(&ready_event_));
    CUDA_CHECK(cudaStreamCreateWithPriority(
        &this->comm_stream_, cudaStreamNonBlocking, get_highest_cuda_stream_priority()));

    // FLUX_CHECK(has_nvlink()) << "GemmAllToAllTranspose is only supported on nvlink.";
    FLUX_CHECK(input_dtype_ == c10::ScalarType::BFloat16)
        << "gemm + all2all + transpose only accept BF16 input";
    FLUX_CHECK(output_dtype_ == c10::ScalarType::BFloat16)
        << "gemm + all2all + transpose only accept BF16 output";
    FLUX_CHECK(hidden_dim_ % head_dim_ == 0 && n_dim_ % head_dim_ == 0);
    FLUX_CHECK(nheads_ % sp_size_ == 0);
    FLUX_CHECK(seq_ % sp_size_ == 0);
    FLUX_CHECK(sp_size > 0 && local_world_size_ % sp_size_ == 0) << "sp size is invalid.";
    FLUX_CHECK(max_num_comm_buf_ >= 1) << "max_num_comm_buf need greater than or equal to 1";
    // FLUX_CHECK(nnodes_ == 1) << "gemm a2a is only supported on single node.";
    if (comm_op_ == PreAttnAllToAllCommOp::A2ATranspose)
      FLUX_CHECK(gqa_ == 0) << "gqa must be equal to 0 for A2ATranspose";
    if (comm_op_ == PreAttnAllToAllCommOp::QKVPackA2A)
      FLUX_CHECK(gqa_ > 0) << "gqa must be greater than 0 for QKVPackA2A";

    cusum_seq_lens_.resize(sp_size_ + 1, 0);
  }

  ~GemmAllToAllTransposeOpImpl() {
    CUDA_CHECK(cudaEventDestroy(comm_event_));
    CUDA_CHECK(cudaEventDestroy(ready_event_));
    CUDA_CHECK(cudaStreamDestroy(comm_stream_));
  }

  torch::Tensor
  a2a_transpose_impl(
      torch::Tensor input_ptrs,  // [bs, local_seq_len, nheads * head_dim]
      torch::Tensor output,      // [bs, local_nheads, seq_len, head_dim]
      int32_t m_per_barrier,     // tile_m of gemm
      int32_t n_per_barrier,     // tile_n of gemm
      int32_t num_comm_sm,
      cudaStream_t stream) {
    CHECK_NDIM(output, 4);
    int32_t bs = output.size(0);
    int32_t seq_len = output.size(2);
    int32_t local_nheads = output.size(1);
    int32_t head_dim = output.size(3);

    const int32_t TILE_M = (128 + head_dim - 1) / head_dim * head_dim;
    const int32_t TILE_N = (512 + head_dim - 1) / head_dim * head_dim;
    PreAttnAll2AllTransposeParam param = {
        .input_ptrs = (void **)input_ptrs.data_ptr(),
        .output_ptr = output.data_ptr(),
        .bs = bs,
        .local_nheads = local_nheads,
        .local_seq_len = seq_len / this->sp_size_,
        .head_dim = head_dim,
        .rank = this->sp_rank_,
        .world_size = this->sp_size_,
        .TILE_M = TILE_M,
        .TILE_N = TILE_N,
        .m_per_barrier = m_per_barrier,
        .n_per_barrier = n_per_barrier,
        .NUM_COMM_SM = num_comm_sm};
    for (int32_t i = 0; i < this->sp_size_; i++) {
      param.barrier_ptrs[i] = this->barrier_ptrs_[i];
    }
    pre_attn_all2all_transpose_impl(param, ths_op::from_torch_dtype(this->output_dtype_), stream);
    return output;
  }

  void
  check_device_tensor(torch::Tensor tensor, std::vector<int64_t> shapes, c10::ScalarType dtype) {
    FLUX_CHECK(tensor.dim() == shapes.size()) << "tensor dim mismatch.";
    CHECK_CUDA(tensor);
    FLUX_CHECK(tensor.is_contiguous()) << "tensor should be contiguous.";
    for (int i = 0; i < shapes.size(); ++i)
      FLUX_CHECK(tensor.size(i) == shapes[i]) << "shape mismatch.";
    CHECK_TYPE(tensor, dtype);
  }

  std::vector<torch::Tensor>
  qkv_pack_a2a_impl(
      torch::Tensor input_ptrs,  // [bs, local_seq_len, nheads * head_dim]
      int32_t bs,
      int32_t seq_len,
      int32_t head_dim,
      int32_t local_q_nheads,
      int32_t local_k_nheads,
      int32_t local_v_nheads,
      int32_t m_per_barrier,  // tile_m of gemm
      int32_t n_per_barrier,  // tile_n of gemm
      int32_t num_comm_sm,
      bool skip_barrier,
      c10::optional<std::vector<torch::Tensor>> external_outputs,
      cudaStream_t stream) {
    torch::Tensor out_q, out_k, out_v;  // [bs, seq_len, local_heads, head_dim]

    if (!external_outputs.has_value()) {
      out_q = empty_with_uninitialized_data(
          std::vector<int64_t>{bs, seq_len, local_q_nheads, head_dim},
          at::TensorOptions(this->output_dtype_).device(at::kCUDA));
      out_k = empty_with_uninitialized_data(
          std::vector<int64_t>{bs, seq_len, local_k_nheads, head_dim},
          at::TensorOptions(this->output_dtype_).device(at::kCUDA));
      out_v = empty_with_uninitialized_data(
          std::vector<int64_t>{bs, seq_len, local_v_nheads, head_dim},
          at::TensorOptions(this->output_dtype_).device(at::kCUDA));
    } else {
      FLUX_CHECK(external_outputs.value().size() == 3);
      out_q = external_outputs.value()[0];
      out_k = external_outputs.value()[1];
      out_v = external_outputs.value()[2];
      check_device_tensor(
          out_q, std::vector<int64_t>{bs, seq_len, local_q_nheads, head_dim}, this->output_dtype_);
      check_device_tensor(
          out_k, std::vector<int64_t>{bs, seq_len, local_k_nheads, head_dim}, this->output_dtype_);
      check_device_tensor(
          out_v, std::vector<int64_t>{bs, seq_len, local_v_nheads, head_dim}, this->output_dtype_);
    }

    constexpr int32_t TILE_S = 256;
    constexpr int32_t TILE_NH = 4;
    PreAttnQKVPackA2AParams params = {
        .input_ptrs = (void **)input_ptrs.data_ptr(),
        .q_ptr = (void *)out_q.data_ptr(),
        .k_ptr = (void *)out_k.data_ptr(),
        .v_ptr = (void *)out_v.data_ptr(),
        .bs = bs,
        .local_q_nheads = local_q_nheads,
        .local_k_nheads = local_k_nheads,
        .local_v_nheads = local_v_nheads,
        .head_dim = head_dim,
        .rank = this->sp_rank_,
        .world_size = this->sp_size_,
        .TILE_S = TILE_S,
        .TILE_NH = TILE_NH,
        .m_per_barrier = m_per_barrier,
        .n_per_barrier = n_per_barrier,
        .num_comm_sm = num_comm_sm,
        .skip_barrier = skip_barrier};
    for (int32_t i = 0; i < this->sp_size_; i++) {
      params.barrier_ptrs[i] = this->barrier_ptrs_[i];
    }
    for (int32_t i = 0; i < this->sp_size_ + 1; ++i) {
      params.cusum_seq_lens[i] = this->cusum_seq_lens_[i];
    }
    pre_attn_qkv_pack_a2a_impl(params, ths_op::from_torch_dtype(this->output_dtype_), stream);
    return {out_q, out_k, out_v};
  }

  void
  reset_cusum_seq_lens(int32_t local_seq_len, c10::optional<torch::Tensor> seq_lens_cpu) {
    this->cusum_seq_lens_.resize(this->sp_size_ + 1);
    this->cusum_seq_lens_[0] = 0;
    if (seq_lens_cpu.has_value()) {
      FLUX_CHECK(comm_op_ == PreAttnAllToAllCommOp::QKVPackA2A) << "only QKVPackA2A support dp.";
      CHECK_NDIM(seq_lens_cpu.value(), 1);
      FLUX_CHECK(seq_lens_cpu.value().device().type() == torch::kCPU)
          << "seq_lens is not on cpu\n";
      FLUX_CHECK(seq_lens_cpu.value().size(0) == this->sp_size_);
      FLUX_CHECK(seq_lens_cpu.value().dtype() == c10::ScalarType::Int);
      int32_t *seq_lens_ptr = seq_lens_cpu->data_ptr<int32_t>();
      for (int32_t i = 0; i < this->sp_size_; ++i) {
        int cur_seq_len = *(seq_lens_ptr + i);
        this->cusum_seq_lens_[i + 1] = this->cusum_seq_lens_[i] + cur_seq_len;
        if (i == this->sp_rank_) {
          FLUX_CHECK(local_seq_len == cur_seq_len);
        }
      }
    } else {
      for (int32_t i = 0; i < this->sp_size_; ++i) {
        this->cusum_seq_lens_[i + 1] = this->cusum_seq_lens_[i] + local_seq_len;
      }
    }
  }

  std::vector<torch::Tensor>
  forward_impl(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> seq_lens_cpu,
      c10::optional<torch::Tensor> bias,
      c10::optional<std::vector<torch::Tensor>> external_outputs,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum,
      int32_t num_comm_sm,
      int32_t sm_margin,
      c10::optional<UnifiedGemmHParams> const &hparams_or_none,
      cudaStream_t stream) {
    FLUX_CHECK(!input_scale.has_value() && !weight_scale.has_value() && !output_scale.has_value())
        << "GemmAllToAllTranspose does not support scale.";

    CHECK_NDIM(input, 3);  // [bs, local_seq_len, nh * hd]
    CHECK_NDIM(weight, 2);
    FLUX_CHECK(input.is_contiguous()) << "input should be contiguous.";
    FLUX_CHECK(weight.is_contiguous()) << "weight should be contiguous.";
    FLUX_CHECK(input.size(0) <= this->bs_) << "batch size exceeds limit.";
    FLUX_CHECK(input.size(2) == this->hidden_dim_) << "hidden dim mismatch.";
    FLUX_CHECK(input.size(1) <= this->seq_ / this->sp_size_) << "local seq exceeds limit.";

    reset_cusum_seq_lens(input.size(1), seq_lens_cpu);
    if (num_comm_sm == -1) {
      num_comm_sm = this->world_size_;
    }
    FLUX_CHECK(num_comm_sm > 0);

    int32_t k = input.size(2);
    int32_t m = input.numel() / k;
    int32_t n = this->n_dim_;
    if (this->transpose_weight_) {
      FLUX_CHECK_EQ(weight.size(0), k);
      FLUX_CHECK_EQ(weight.size(1), n);
    } else {
      FLUX_CHECK_EQ(weight.size(0), n);
      FLUX_CHECK_EQ(weight.size(1), k);
    }

    auto meta = get_gemm_meta(
        this->comm_op_,
        this->input_dtype_,
        this->output_dtype_,
        this->transpose_weight_,
        /*has_bias=*/bias.has_value(),
        /*fast_accum=*/fast_accum);

    auto rt_config = get_rt_config(this->sp_size_, this->nnodes_, m, n, k);

    UnifiedGemmHParams hparams = hparams_or_none.has_value()
                                     ? hparams_or_none.value()
                                     : OpRegistry::instance().get_hparams(meta, rt_config);
    std::unique_ptr<GemmOperatorBase> cutlass_op = OpRegistry::instance().get_op(meta, hparams);
    auto [tile_m, tile_n, _] = hparams.tile_shape();

    std::any gemm_args;
    auto data_ptr_or = [](auto &&t, void *other) -> void * {
      return t.has_value() ? t->data_ptr() : other;
    };

    gemm_args = GemmAllToAllTransposeArguments{
        .m = m,
        .n = n,
        .k = k,
        .nnodes = this->nnodes_,
        .rank = this->sp_rank_,
        .world_size = this->sp_size_,
        .alpha = 1.0f,
        .beta = bias.has_value() ? 1.0f : 0.0f,
        .input = input.data_ptr(),
        .weight = weight.data_ptr(),
        .bias = data_ptr_or(bias, nullptr),
        .gemm_output = this->gemm_output_buffer_.data_ptr(),
        .barrier_ptrs = this->barrier_ptrs_.data(),
        .sm_margin = sm_margin + num_comm_sm};

    sp_group_barrier_all(stream);

    int64_t barrier_workspace_size = cutlass_op->get_barrier_workspace_size(gemm_args);

    torch::Tensor out_q_tensor;

    int32_t bs = input.size(0);
    int32_t seq_len = this->cusum_seq_lens_[this->sp_size_];
    int32_t local_nheads = this->n_dim_ / this->head_dim_ / this->sp_size_;
    int32_t local_q_nheads, local_k_nheads, local_v_nheads;

    if (gqa_ == 0) {
      local_q_nheads = local_nheads;
      local_k_nheads = 0;
      local_v_nheads = 0;
      if (!external_outputs.has_value()) {
        out_q_tensor = empty_with_uninitialized_data(
            std::vector<int64_t>{bs, local_q_nheads, seq_len, this->head_dim_},
            at::TensorOptions(this->output_dtype_)
                .device(at::kCUDA)
                .device_index(c10::cuda::current_device()));
      } else {
        FLUX_CHECK(external_outputs.value().size() == 1);
        out_q_tensor = external_outputs.value()[0];
        check_device_tensor(
            out_q_tensor,
            std::vector<int64_t>{bs, local_q_nheads, seq_len, this->head_dim_},
            this->output_dtype_);
      }
    } else {
      FLUX_CHECK(local_nheads % (gqa_ + 2) == 0);
      local_q_nheads = local_nheads / (gqa_ + 2) * gqa_;
      local_k_nheads = local_nheads / (gqa_ + 2);
      local_v_nheads = local_k_nheads;
    }

    FLUX_CHECK(this->barrier_buffer_.nbytes() >= barrier_workspace_size);
    this->reset_barrier(stream);
    // this group barrier ensure that all signals are reset before launch communication kernel
    // under pull mode.
    sp_group_barrier_all(stream);

    int64_t workspace_size = cutlass_op->get_workspace_size(gemm_args);
    this->lazy_init_gemm_workspace_buffer(input, workspace_size);

    CUDA_CHECK(cudaEventRecord(this->ready_event_, stream));

    cutlass_op->run(
        gemm_args, workspace_size ? this->gemm_workspace_buffer_.data_ptr() : nullptr, stream);

    CUDA_CHECK(cudaStreamWaitEvent(this->comm_stream_, this->ready_event_));
    if (get_int_from_env("CUDA_DEVICE_MAX_CONNECTIONS", -1) != 1) {
      CU_CHECK(CUStreamWaitValue(
          this->comm_stream_,
          (CUdeviceptr)(this->barrier_buffer_.data_ptr()),
          1,
          CU_STREAM_WAIT_VALUE_EQ));
    }

    std::vector<torch::Tensor> outs;

    // dispatch communication kernel
    if (comm_op_ == PreAttnAllToAllCommOp::A2ATranspose) {
      out_q_tensor = a2a_transpose_impl(
          this->gemm_output_ptrs_buffer_,
          out_q_tensor,
          tile_m,
          tile_n,
          num_comm_sm,
          this->comm_stream_);
      outs = {out_q_tensor};
    } else if (comm_op_ == PreAttnAllToAllCommOp::QKVPackA2A) {
      outs = qkv_pack_a2a_impl(
          this->gemm_output_ptrs_buffer_,
          bs,
          seq_len,
          this->head_dim_,
          local_q_nheads,
          local_k_nheads,
          local_v_nheads,
          tile_m,
          tile_n,
          num_comm_sm,
          /*skip_barrier=*/false,
          external_outputs,
          this->comm_stream_);
    } else {
      FLUX_CHECK(false) << "Unsupported mode.";
    }
    CUDA_CHECK(cudaEventRecord(this->comm_event_, this->comm_stream_));
    CUDA_CHECK(cudaStreamWaitEvent(stream, this->comm_event_));

    // only `out_q_tensor` is valid for A2ATranspose
    return outs;
  }

  std::vector<torch::Tensor>
  forward(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> seq_lens_cpu,
      c10::optional<torch::Tensor> bias,
      c10::optional<std::vector<torch::Tensor>> external_outputs,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum,
      int32_t num_comm_sm,
      int32_t sm_margin) {
    auto stream = at::cuda::getCurrentCUDAStream();
    auto outs = forward_impl(
        input,
        weight,
        seq_lens_cpu,
        bias,
        external_outputs,
        input_scale,
        weight_scale,
        output_scale,
        fast_accum,
        num_comm_sm,
        sm_margin,
        c10::nullopt,
        stream);

    // user need to insert the necessary wait_stream/record_stream to ensure that the inputs is
    // not deallocated.
    // auto record_stream_for_optional_tensor = [&](c10::optional<torch::Tensor>
    // t) -> void {
    //   if (t.has_value())
    //     t.value().record_stream(stream);
    // };

    // input.record_stream(stream);
    // weight.record_stream(stream);
    // record_stream_for_optional_tensor(bias);
    // record_stream_for_optional_tensor(input_scale);
    // record_stream_for_optional_tensor(weight_scale);
    // record_stream_for_optional_tensor(output_scale);
    // if (external_outputs.has_value()) {
    //   for (auto t : external_outputs.value()) {
    //     t.record_stream(stream);
    //   }
    // }
    return outs;
  }

  std::vector<torch::Tensor>
  qkv_pack_a2a(
      torch::Tensor qkv,
      c10::optional<torch::Tensor> seq_lens_cpu,
      int32_t num_comm_sm,
      bool is_input_in_comm_buf,
      int32_t comm_buf_idx) {
    auto stream = at::cuda::getCurrentCUDAStream();
    CHECK_NDIM(qkv, 4);  // [bs, local_seq_len, nh, hd]
    FLUX_CHECK(qkv.is_contiguous()) << "qkv should be contiguous.";
    FLUX_CHECK(qkv.size(0) <= this->bs_) << "batch size exceeds limit.";
    FLUX_CHECK(qkv.size(3) == this->head_dim_) << "hidden dim mismatch.";
    FLUX_CHECK(qkv.size(2) * qkv.size(3) == this->n_dim_) << "hidden dim mismatch.";
    FLUX_CHECK(qkv.size(1) <= this->seq_ / this->sp_size_) << "local seq exceeds limit.";
    FLUX_CHECK(comm_op_ == PreAttnAllToAllCommOp::QKVPackA2A);

    reset_cusum_seq_lens(qkv.size(1), seq_lens_cpu);
    if (num_comm_sm == -1) {
      num_comm_sm = this->world_size_;
    }
    FLUX_CHECK(num_comm_sm > 0);

    // copy input to symm buf
    // if is_input_in_comm_buf is true, barrier and d2d are called by user.
    if (!is_input_in_comm_buf) {
      FLUX_CHECK(comm_buf_idx == 0);
      sp_group_barrier_all(stream);
      CUDA_CHECK(cudaMemcpyAsync(
          this->gemm_output_buffer_.data_ptr(),  // symm buf
          qkv.data_ptr(),
          qkv.nbytes(),
          cudaMemcpyDeviceToDevice,
          stream));
      sp_group_barrier_all(stream);
    }

    int32_t n = qkv.size(2) * qkv.size(3);
    int32_t m = qkv.numel() / n;
    int32_t bs = qkv.size(0);
    int32_t seq_len = this->cusum_seq_lens_[this->sp_size_];
    int32_t local_nheads = n / this->head_dim_ / this->sp_size_;
    int32_t local_q_nheads, local_k_nheads, local_v_nheads;
    FLUX_CHECK(local_nheads % (gqa_ + 2) == 0);
    local_q_nheads = local_nheads / (gqa_ + 2) * gqa_;
    local_k_nheads = local_nheads / (gqa_ + 2);
    local_v_nheads = local_k_nheads;

    FLUX_CHECK(comm_buf_idx < this->comm_buf_ptrs_buffer_list_.size())
        << "comm_buf_idx out of range.";

    // a2a
    auto outs = qkv_pack_a2a_impl(
        this->comm_buf_ptrs_buffer_list_[comm_buf_idx],
        bs,
        seq_len,
        this->head_dim_,
        local_q_nheads,
        local_k_nheads,
        local_v_nheads,
        m,
        n,
        num_comm_sm,
        /*skip_barrier=*/true,
        c10::nullopt,
        stream);
    // qkv.record_stream(stream);
    return outs;
  }

  void
  check_input(torch::Tensor input) {
    CHECK_NDIM(input, 4);  // [bs, local_seq_len, nh, hd]
    FLUX_CHECK(input.is_contiguous()) << "qkv should be contiguous.";
    FLUX_CHECK(input.size(0) <= this->bs_) << "batch size exceeds limit.";
    FLUX_CHECK(input.size(3) == this->head_dim_) << "hidden dim mismatch.";
    FLUX_CHECK(input.size(2) * input.size(3) <= this->n_dim_) << "hidden dim exceeds limit.";
    FLUX_CHECK(input.size(1) <= this->seq_ / this->sp_size_) << "local seq exceeds limit.";
    CHECK_TYPE(input, this->input_dtype_);
  }

  torch::Tensor
  get_comm_buf(int32_t comm_buf_idx) {
    FLUX_CHECK(comm_buf_idx < this->max_num_comm_buf_) << "comm buf idx out of range";
    size_t local_seq = this->seq_ / this->sp_size_;
    size_t buf_size = local_seq * (size_t)this->bs_ * (size_t)this->n_dim_;
    torch::Tensor comm_buffer =
        this->gemm_output_buffer_.reshape({this->max_num_comm_buf_, buf_size})
            .slice(0, comm_buf_idx, comm_buf_idx + 1);
    return comm_buffer;
  }

  torch::Tensor
  get_input_comm_buf(torch::Tensor input, int32_t comm_buf_idx) {
    check_input(input);
    size_t nelems = input.numel();
    torch::Tensor full_comm_buf = get_comm_buf(comm_buf_idx);
    torch::Tensor comm_buffer =
        full_comm_buf.reshape(-1).slice(0, 0, nelems).reshape(input.sizes());
    return comm_buffer;
  }

  torch::Tensor
  pre_attn_a2a(
      torch::Tensor input,
      c10::optional<torch::Tensor> seq_lens_cpu,
      int32_t num_comm_sm,
      bool is_input_in_comm_buf,
      int32_t comm_buf_idx) {
    auto stream = at::cuda::getCurrentCUDAStream();
    check_input(input);
    FLUX_CHECK(comm_op_ == PreAttnAllToAllCommOp::QKVPackA2A);

    reset_cusum_seq_lens(input.size(1), seq_lens_cpu);
    if (num_comm_sm == -1) {
      num_comm_sm = this->world_size_;
    }
    FLUX_CHECK(num_comm_sm > 0);

    // copy input to symm buf
    if (!is_input_in_comm_buf) {
      FLUX_CHECK(comm_buf_idx == 0);
      sp_group_barrier_all(stream);
      CUDA_CHECK(cudaMemcpyAsync(
          this->gemm_output_buffer_.data_ptr(),  // symm buf
          input.data_ptr(),
          input.nbytes(),
          cudaMemcpyDeviceToDevice,
          stream));
      sp_group_barrier_all(stream);
    }

    int32_t n = input.size(2) * input.size(3);
    int32_t m = input.numel() / n;
    int32_t bs = input.size(0);
    int32_t seq_len = this->cusum_seq_lens_[this->sp_size_];
    int32_t local_nheads = n / this->head_dim_ / this->sp_size_;

    FLUX_CHECK(comm_buf_idx < this->comm_buf_ptrs_buffer_list_.size())
        << "comm_buf_idx out of range.";
    // a2a
    auto outs = qkv_pack_a2a_impl(
        this->comm_buf_ptrs_buffer_list_[comm_buf_idx],
        bs,
        seq_len,
        this->head_dim_,
        /*local_q_nheads=*/local_nheads,
        /*local_k_nheads=*/0,
        /*local_v_nheads=*/0,
        m,
        n,
        num_comm_sm,
        /*skip_barrier=*/true,
        c10::nullopt,
        stream);
    // input.record_stream(stream);
    return outs[0];
  }

  void
  sp_group_barrier_all(cudaStream_t stream) {
    flux_barrier_all_on_stream(
        stream,
        this->sp_group_sync_buffers_,
        this->sp_rank_,
        /*ring_mode=*/false,
        /*force_flux_impl*/ true);
  }

  void
  reset_barrier(cudaStream_t stream) {
    if (this->barrier_buffer_.defined()) {
      CUDA_CHECK(cudaMemsetAsync(
          this->barrier_buffer_.data_ptr(), 0, this->barrier_buffer_.nbytes(), stream));
    }
  }

  std::vector<torch::Tensor>
  profiling(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum,
      int32_t num_comm_sm,
      int32_t sm_margin,
      c10::intrusive_ptr<ProfilingContext> opt_ctx) {
    auto stream = c10::cuda::getCurrentCUDAStream();
    int32_t k = input.size(2);
    int32_t m = input.numel() / k;
    int32_t n = this->n_dim_;

    auto meta = unify_type(get_gemm_meta(
        this->comm_op_,
        this->input_dtype_,
        this->output_dtype_,
        this->transpose_weight_,
        /*has_bias=*/bias.has_value(),
        /*fast_accum=*/fast_accum));

    auto rt_config = get_rt_config(this->sp_size_, this->nnodes_, m, n, k);
    ProfilingContext tmp_ctx("__tmp__");
    ProfilingContext *ctx = opt_ctx == nullptr ? &tmp_ctx : opt_ctx.get();

    auto filter_hparams = [&](UnifiedGemmHParams const &hparams) { return true; };

    auto elapsed_tensor = torch::empty({}, weight.options().dtype(c10::ScalarType::Float));
    auto reduced_elapsed_tensor = elapsed_tensor.clone();

    OpRegistry::instance().visit_hparams(
        [&](UnifiedGemmHParams const &hparams) {
          if (not filter_hparams(hparams)) {
            return;
          }
          // filter non-consistent hparams
          constexpr int warm_iters = 10;
          constexpr int iters = 100;
          float total_elapsed = 0;

          auto stream = c10::cuda::getCurrentCUDAStream();
          sp_group_barrier_all(stream);
          c10::cuda::stream_synchronize(stream);
          // warm up
          for (int iter = 0; iter < warm_iters; ++iter) {
            auto _ [[maybe_unused]] = this->forward_impl(
                input,
                weight,
                c10::nullopt,
                bias,
                c10::nullopt,
                input_scale,
                weight_scale,
                output_scale,
                fast_accum,
                num_comm_sm,
                sm_margin,
                hparams,
                stream);
          }

          sp_group_barrier_all(stream);
          c10::cuda::stream_synchronize(stream);
          GpuTimer timer;
          timer.start(stream);
          for (int iter = 0; iter < iters; ++iter) {
            auto _ [[maybe_unused]] = this->forward_impl(
                input,
                weight,
                c10::nullopt,
                bias,
                c10::nullopt,
                input_scale,
                weight_scale,
                output_scale,
                fast_accum,
                num_comm_sm,
                sm_margin,
                hparams,
                stream);
          }
          timer.stop();
          total_elapsed += timer.elapsed_millis();

          // Avoid GPU frequency adjustment
          flux_barrier_all_on_stream(stream);
          c10::cuda::stream_synchronize(stream);
          sleep(1);

          float avg_elapsed = int(total_elapsed / iters * 1000) / 1000.0;
          float reduce_elapsed = all_reduce_max_float(this->pg_world_.get(), avg_elapsed);
          ctx->add(meta, rt_config, hparams, reduce_elapsed);
        },
        meta);

    auto best_hparams = ctx->record_best(meta, rt_config);
    return this->forward_impl(
        std::move(input),
        std::move(weight),
        c10::nullopt,
        std::move(bias),
        c10::nullopt,
        std::move(input_scale),
        std::move(weight_scale),
        std::move(output_scale),
        fast_accum,
        num_comm_sm,
        sm_margin,
        std::move(best_hparams),
        stream);
  }

  void
  _ensure_topo_initialized() {
    if (!topo_utils::is_topo_initialized()) {
      topo_utils::initialize_topo(this->pg_world_.get());
    }
  }
};

GemmAllToAllTransposeOp::GemmAllToAllTransposeOp(
    std::shared_ptr<Group> pg_world,
    int32_t nnodes,
    int32_t sp_size,
    int32_t bs,
    int32_t seq,
    int32_t hidden_dim,
    int32_t head_dim,
    int32_t n_dim,
    c10::ScalarType input_dtype,
    c10::ScalarType output_dtype,
    bool transpose_weight,
    int32_t gqa,
    PreAttnAllToAllCommOp comm_op,
    int32_t max_num_comm_buf)
    : impl_(new GemmAllToAllTransposeOpImpl(
          pg_world,
          nnodes,
          sp_size,
          bs,
          seq,
          hidden_dim,
          head_dim,
          n_dim,
          input_dtype,
          output_dtype,
          transpose_weight,
          gqa,
          comm_op,
          max_num_comm_buf)) {}

GemmAllToAllTransposeOp::~GemmAllToAllTransposeOp() { delete impl_; }

void
GemmAllToAllTransposeOp::sp_group_barrier_all() {
  FLUX_CHECK(this->impl_ != nullptr) << "GemmAllToAllTransposeOp is not initialized";
  auto stream = c10::cuda::getCurrentCUDAStream();
  this->impl_->sp_group_barrier_all(stream);
}

torch::Tensor
GemmAllToAllTransposeOp::get_input_comm_buf(torch::Tensor input, int32_t comm_buf_idx) {
  FLUX_CHECK(this->impl_ != nullptr) << "GemmAllToAllTransposeOp is not initialized";
  return this->impl_->get_input_comm_buf(std::move(input), comm_buf_idx);
}

std::vector<torch::Tensor>
GemmAllToAllTransposeOp::pre_attn_qkv_pack_a2a_no_cpy(
    torch::Tensor qkv,
    c10::optional<torch::Tensor> seq_lens_cpu,
    int32_t num_comm_sm,
    int32_t comm_buf_idx) {
  FLUX_CHECK(this->impl_ != nullptr) << "GemmAllToAllTransposeOp is not initialized";
  return this->impl_->qkv_pack_a2a(
      std::move(qkv),
      std::move(seq_lens_cpu),
      num_comm_sm,
      /*is_input_in_comm_buf*/ true,
      comm_buf_idx);
}

torch::Tensor
GemmAllToAllTransposeOp::pre_attn_a2a_no_cpy(
    torch::Tensor input,
    c10::optional<torch::Tensor> seq_lens_cpu,
    int32_t num_comm_sm,
    int32_t comm_buf_idx) {
  FLUX_CHECK(this->impl_ != nullptr) << "GemmAllToAllTransposeOp is not initialized";
  return this->impl_->pre_attn_a2a(
      std::move(input),
      std::move(seq_lens_cpu),
      num_comm_sm,
      /*is_input_in_comm_buf*/ true,
      comm_buf_idx);
}

std::vector<torch::Tensor>
GemmAllToAllTransposeOp::pre_attn_qkv_pack_a2a(
    torch::Tensor qkv, c10::optional<torch::Tensor> seq_lens_cpu, int32_t num_comm_sm) {
  FLUX_CHECK(this->impl_ != nullptr) << "GemmAllToAllTransposeOp is not initialized";
  return this->impl_->qkv_pack_a2a(
      std::move(qkv), std::move(seq_lens_cpu), num_comm_sm, /*is_input_in_comm_buf*/ false, 0);
}

torch::Tensor
GemmAllToAllTransposeOp::pre_attn_a2a(
    torch::Tensor input, c10::optional<torch::Tensor> seq_lens_cpu, int32_t num_comm_sm) {
  FLUX_CHECK(this->impl_ != nullptr) << "GemmAllToAllTransposeOp is not initialized";
  return this->impl_->pre_attn_a2a(
      std::move(input), std::move(seq_lens_cpu), num_comm_sm, /*is_input_in_comm_buf*/ false, 0);
}

std::vector<torch::Tensor>
GemmAllToAllTransposeOp::forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> seq_lens_cpu,
    c10::optional<torch::Tensor> bias,
    c10::optional<std::vector<torch::Tensor>> outputs,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale,
    c10::optional<torch::Tensor> output_scale,
    bool fast_accum,
    int32_t num_comm_sm,
    int32_t sm_margin) {
  FLUX_CHECK(this->impl_ != nullptr) << "GemmAllToAllTransposeOp is not initialized";
  return this->impl_->forward(
      std::move(input),
      std::move(weight),
      std::move(seq_lens_cpu),
      std::move(bias),
      std::move(outputs),
      std::move(input_scale),
      std::move(weight_scale),
      std::move(output_scale),
      fast_accum,
      num_comm_sm,
      sm_margin);
}

std::vector<torch::Tensor>
GemmAllToAllTransposeOp::profiling(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale,
    c10::optional<torch::Tensor> output_scale,
    bool fast_accum,
    int32_t num_comm_sm,
    int32_t sm_margin,
    c10::intrusive_ptr<ProfilingContext> opt_ctx) {
  FLUX_CHECK(this->impl_ != nullptr) << "GemmAllToAllTransposeOp is not initialized";
  return this->impl_->profiling(
      std::move(input),
      std::move(weight),
      std::move(bias),
      std::move(input_scale),
      std::move(weight_scale),
      std::move(output_scale),
      fast_accum,
      num_comm_sm,
      sm_margin,
      opt_ctx);
}

}  // namespace bytedance::flux::ths_op
