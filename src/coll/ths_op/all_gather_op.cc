//===- all_gather_op.cc ------------------------------------------ C++ ---===//
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

#include "coll/all_gather_impls.hpp"
#include "coll/local_copy_and_reset.hpp"
#include "coll/ths_op/all_gather_op.h"
#include "coll/ths_op/all_gather_types.h"
#include "flux/flux.h"
#include "flux/ths_op/ths_op.h"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/cuda_stub.h"
#include "flux/utils.h"
#include <ATen/core/TensorBase.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/util/Optional.h>

#define SPLIT 1

namespace bytedance::flux::ths_op {

using ths_op::is_s8_torch_dtype;

namespace {
constexpr int kNumSignals = 64;  // TODO(houqi.1993) set this global
}  // namespace

class AllGatherOp::AllGatherOpImpl {
 public:
  AllGatherOpImpl(
      std::shared_ptr<Group> tp_group,
      int nnodes,
      size_t max_m,
      size_t k,
      at::ScalarType input_dtype);

  ~AllGatherOpImpl();

  void run_with_optional_options(
      torch::Tensor input,
      c10::optional<torch::Tensor> input_scale,
      const AllGatherOptionWithOptional &opt,
      cudaStream_t stream);

  void run(
      const torch::Tensor &input,
      c10::optional<torch::Tensor> input_scale,
      const AllGatherOption &opt,
      cudaStream_t stream);

  // only provide local tensor
  torch::Tensor
  local_input_buffer() {
    return input_buffer_;
  }
  torch::Tensor
  local_input_scale_buffer() {
    return input_scale_buffer_;
  }
  torch::Tensor
  local_barrier_buffer() {
    return barrier_buffer;
  }

  int32_t *ag_signal_ptr() const;

  cudaEvent_t &
  get_local_prepare_event() {
    return this->local_prepare_event;
  }

 private:
  void create_symetric_buffers();
  void create_symetric_buffers_with_input_scale();
  void create_sync_buffers();

  void barrier_async(cudaStream_t stream);
  bool is_p2p_atomic_supported();

  void copy_local_and_sync_with_cudaMemcpyAsync(
      const torch::Tensor &input,
      const c10::optional<torch::Tensor> &input_scale,
      bool input_buffer_copied,
      bool use_cuda_core_ag,
      cudaStream_t stream);

  void copy_local_and_sync_with_kernel(
      const torch::Tensor &input,
      const c10::optional<torch::Tensor> &input_scale,
      bool input_buffer_copied,
      bool fuse_sync,
      cudaStream_t stream);

  void set_ready(int rank_, int segment, int split_index, cudaStream_t stream);
  void wait_ready(int rank_, int segment, int split_index, cudaStream_t stream);

  void copy_all_to_all(
      torch::Tensor input,
      c10::optional<torch::Tensor> input_scale,
      bool use_cuda_core,
      cudaStream_t stream);

  void copy_ring_pull(
      torch::Tensor input, c10::optional<torch::Tensor> input_scale, cudaStream_t stream);

  void copy_ring_push_1d(
      torch::Tensor input, c10::optional<torch::Tensor> input_scale, cudaStream_t stream);

  void copy_ring_push_2d_pcie(
      torch::Tensor input, c10::optional<torch::Tensor> input_scale, cudaStream_t stream);

  void copy_ring_push_by_kernel(
      torch::Tensor input,
      c10::optional<torch::Tensor> input_scale,
      bool use_2d_mode,
      cudaStream_t stream);

  template <int src_rank, int dst_rank>
  void ring_all_gather_run(int chunk_size, cudaStream_t stream);

  template <int src_rank, int dst_rank, typename Transfer, typename... Transfers>
  void ring_all_gather_run(int chunk_size, cudaStream_t stream);

  template <int reordered, typename Transfer, typename... Transfers>
  void ring_all_gather_reorder(int chunk_size, cudaStream_t stream);

  void ring_all_gather(int chunk_size, cudaStream_t stream);

  void
  _ensure_topo_initialized() {
    if (!topo_utils::is_topo_initialized()) {
      topo_utils::initialize_topo(this->tp_group.get());
    }
  }

 private:
  std::shared_ptr<Group> tp_group;

  int nnodes;
  int world_size;
  int rank;
  int local_world_size;
  int local_rank;

  int max_m_dim, k_dim;
  at::ScalarType input_dtype;
  bool with_input_scale = false;

  // used for the cuda-ipc-barrier
  std::vector<torch::Tensor> sync_buffers;
  std::vector<torch::Tensor> input_buffers;
  std::vector<torch::Tensor> input_scale_buffers;
  std::vector<torch::Tensor> barrier_buffers;
  torch::Tensor input_buffer_;
  torch::Tensor input_scale_buffer_;
  torch::Tensor output_buffer;
  torch::Tensor barrier_buffer;
  torch::Tensor counter_buffer;
  torch::Tensor sync_ptrs_buffer;  // symetric memory for barrier all. only with atomic supported
  torch::Tensor ag_signal;

  std::vector<void *> input_ptrs;
  std::vector<void *> input_scale_ptrs;
  std::vector<int32_t *> barrier_ptrs;
  std::vector<int32_t *> sync_ptrs;

  bool sync_in_ring_mode = false;

  // copy param for s8 gemm
  AllGatherParams copy_param;

  cudaEvent_t local_prepare_event;
};

AllGatherOp::AllGatherOpImpl::AllGatherOpImpl(
    std::shared_ptr<Group> tp_group_,
    int nnodes_,
    size_t max_m_dim,
    size_t k_dim,
    at::ScalarType input_dtype)
    : tp_group(tp_group_),
      nnodes(nnodes_),
      world_size(tp_group_->get_size()),
      rank(tp_group_->get_rank()),
      local_world_size(world_size / nnodes),
      local_rank(rank % local_world_size),
      max_m_dim(max_m_dim),
      k_dim(k_dim),
      input_dtype(input_dtype),
      with_input_scale(is_s8_torch_dtype(input_dtype)),
      input_ptrs(world_size, nullptr),
      input_scale_ptrs(world_size, nullptr),
      barrier_ptrs(world_size, nullptr),
      sync_ptrs(world_size, nullptr),
      sync_in_ring_mode(
          !is_p2p_atomic_supported() || this->local_world_size > kMaxLocalWorldSize) {
  _ensure_topo_initialized();
  FLUX_CHECK_DIV(world_size, nnodes)
      << "invalid nnodes: world_size[" << world_size << "] %% nnodes[" << nnodes << "] != 0";
  create_symetric_buffers();
  CUDA_CHECK(cudaEventCreateWithFlags(&this->local_prepare_event, cudaEventDisableTiming));
}

int32_t *
AllGatherOp::AllGatherOpImpl::ag_signal_ptr() const {
  return copy_param.ag_signal;
}

void
AllGatherOp::AllGatherOpImpl::create_symetric_buffers() {
  // input buffer
  this->input_buffers = flux_create_tensor_list(
      {max_m_dim, k_dim},
      this->input_dtype,
      this->tp_group.get(),
      this->local_world_size > kMaxLocalWorldSize);
  this->input_buffer_ = this->input_buffers[this->local_rank];
  for (int i = 0; i < world_size; ++i) {
    if (i / this->local_world_size == rank / this->local_world_size) {
      this->input_ptrs[i] = this->input_buffers[i].data_ptr();
    } else {
      this->input_ptrs[i] = nullptr;
    }
  }

  // barrier buffer
  this->barrier_buffers = flux_create_tensor_list(
      {kNumSignals},
      c10::ScalarType::Int,
      this->tp_group.get(),
      this->local_world_size > kMaxLocalWorldSize);
  this->barrier_buffer = this->barrier_buffers[this->local_rank];
  for (int i = 0; i < world_size; ++i) {
    if (i / this->local_world_size == rank / this->local_world_size) {
      this->barrier_ptrs[i] = (int32_t *)this->barrier_buffers[i].data_ptr();
    } else {
      this->barrier_ptrs[i] = (int32_t *)nullptr;
    }
  }

  // counter buffer
  auto options =
      torch::TensorOptions().device(torch::Device(torch::kCUDA)).dtype(c10::ScalarType::Int);
  this->counter_buffer = torch::empty({kNumSignals + 1}, options);
  this->counter_buffer.zero_();

  if (this->with_input_scale) {
    create_symetric_buffers_with_input_scale();
  }
  create_sync_buffers();

  // init params for d2d kernel
  for (int i = 0; i < world_size; ++i) {
    this->copy_param.input_ptrs[i] = this->input_ptrs[i];
    this->copy_param.scale_ptrs[i] = this->input_scale_ptrs[i];
    this->copy_param.ag_barriers[i] = this->barrier_ptrs[i];
  }

  this->copy_param.has_scale = this->with_input_scale;
  this->copy_param.counter = (int32_t *)this->counter_buffer.data_ptr();
  this->copy_param.world_size = this->world_size;
  this->copy_param.sub_world_size = topo_utils::topo_numa_local_world_size();
  this->copy_param.rank = this->rank;
  this->copy_param.ag_signal = (int32_t *)(this->copy_param.counter + this->world_size);
}

// TODO(houqi.1993) create tensor list bad. implement symmtric_tensor later
void
AllGatherOp::AllGatherOpImpl::create_symetric_buffers_with_input_scale() {
  // input scale buffer
  this->input_scale_buffers = flux_create_tensor_list(
      {max_m_dim, 1},
      c10::ScalarType::Float,
      this->tp_group.get(),
      this->local_world_size > kMaxLocalWorldSize);
  this->input_scale_buffer_ = this->input_scale_buffers[this->local_rank];
  for (int i = 0; i < world_size; ++i) {
    if (i / this->local_world_size == rank / this->local_world_size) {
      this->input_scale_ptrs[i] = this->input_scale_buffers[i].data_ptr();
    } else {
      this->input_scale_ptrs[i] = nullptr;
    }
  }
}

void
AllGatherOp::AllGatherOpImpl::create_sync_buffers() {
  // Each block of local copy kernel needs to be synchronized
  size_t local_copy_block_num = get_local_copy_max_block_num(max_m_dim / this->world_size * k_dim);
  this->sync_buffers = flux_create_tensor_list(
      {static_cast<long>(this->world_size * local_copy_block_num)},
      c10::ScalarType::Int,
      this->tp_group.get(),
      this->local_world_size > kMaxLocalWorldSize);
  this->sync_buffers[this->rank].zero_();  // zeros the sync buffer for cuda ipc at the start

  size_t sync_ptrs_buffer_size = sizeof(void *) * (this->world_size);
  auto options = at::TensorOptions(at::kCUDA)
                     .device_index(at::cuda::current_device())
                     .dtype(c10::ScalarType::Byte);
  this->sync_ptrs_buffer = torch::empty({static_cast<long>(sync_ptrs_buffer_size)}, options);
  for (int i = 0; i < world_size; ++i) {
    if (i / this->local_world_size == rank / this->local_world_size) {
      this->sync_ptrs[i] = reinterpret_cast<int32_t *>(this->sync_buffers[i].data_ptr());
    } else {
      this->sync_ptrs[i] = nullptr;
    }
  }
  CUDA_CHECK(cudaMemcpy(
      this->sync_ptrs_buffer.data_ptr(),
      this->sync_ptrs.data(),
      sync_ptrs_buffer_size,
      cudaMemcpyHostToDevice));
}

bool
AllGatherOp::AllGatherOpImpl::is_p2p_atomic_supported() {
  int current_device = at::cuda::current_device();
  int next = (this->local_rank + 1) % this->local_world_size;
  int p2p_atomic_supported = false;
  if (current_device != next) {
    CUDA_CHECK(cudaDeviceGetP2PAttribute(
        &p2p_atomic_supported, cudaDevP2PAttrNativeAtomicSupported, current_device, next));
  }
  return p2p_atomic_supported;
}

void
AllGatherOp::AllGatherOpImpl::run_with_optional_options(
    torch::Tensor input,
    torch::optional<torch::Tensor> input_scale,
    const AllGatherOptionWithOptional &opt,
    cudaStream_t stream) {
  _ensure_topo_initialized();
  bool with_input_scale = input_scale.has_value() && this->with_input_scale;
  run(input,
      this->with_input_scale ? input_scale : c10::nullopt,
      AllGatherOption{
          .input_buffer_copied = opt.input_buffer_copied.value_or(false),
          .use_cuda_core_local = opt.use_cuda_core_local.value_or(with_input_scale),
          .use_cuda_core_ag = opt.use_cuda_core_ag.value_or(with_input_scale),
          .fuse_sync = opt.fuse_sync.value_or(with_input_scale),
          .use_read = opt.use_read.value_or(false),
          .mode = opt.mode.value_or(get_default_ag_ring_mode()),
      },
      stream);
}

void
AllGatherOp::AllGatherOpImpl::run(
    const torch::Tensor &input,
    torch::optional<torch::Tensor> input_scale,
    const AllGatherOption &opt,
    cudaStream_t stream) {
  if (!this->with_input_scale) {
    FLUX_CHECK(!input_scale.has_value())
        << "input_scale is provided but not supported for this AllGatherOp";
  }
  if (opt.use_cuda_core_local) {
    // local_copy flag check within below function
    copy_local_and_sync_with_kernel(
        input, input_scale, opt.input_buffer_copied, opt.fuse_sync, stream);
  } else {
    copy_local_and_sync_with_cudaMemcpyAsync(
        input, input_scale, opt.input_buffer_copied, opt.use_cuda_core_ag, stream);
  }

  CUDA_CHECK(cudaEventRecord(this->local_prepare_event, stream));

  if (opt.mode == AGRingMode::All2All) {
    /// All2All algorithm
    copy_all_to_all(input, input_scale, opt.use_cuda_core_ag, stream);
  } else if (opt.mode == AGRingMode::Ring1D) {
    if (opt.use_cuda_core_ag) {
      if (opt.use_read) {
        FLUX_LOG_FIRST_N(INFO, 1)
            << "WARNING: Ring1D+CUDA core does not support read mode. use write instead\n";
      }
      copy_ring_push_by_kernel(input, input_scale, false, stream);
    } else {
      // always the  0 <- 1 <- 2 <- 3 <- 0 order
      if (!opt.use_read) {
        copy_ring_push_1d(input, input_scale, stream);
      } else {
        copy_ring_pull(input, input_scale, stream);
      }
    }
  } else if (opt.mode == AGRingMode::Ring2D) {
    if (opt.use_read) {
      FLUX_LOG_FIRST_N(INFO, 1) << "warning: Ring2D only supports write. use write mode instead\n";
    }
    if (opt.use_cuda_core_ag) {
      LOG(INFO) << "use Ring2D+CUDA core mode\n";
      copy_ring_push_by_kernel(input, input_scale, true, stream);
    } else {
      copy_ring_push_2d_pcie(input, input_scale, stream);
    }
  } else {
    FLUX_CHECK(false) << "opt mode not supported";
  }
}

void
AllGatherOp::AllGatherOpImpl::copy_local_and_sync_with_cudaMemcpyAsync(
    const torch::Tensor &input,
    const c10::optional<torch::Tensor> &input_scale,
    bool is_input_buffer_copied,
    bool use_cuda_core_ag,
    cudaStream_t stream) {
  size_t chunk_size = input.nbytes();
  void *input_ptr = input.data_ptr();
  void *input_buffer_ptr = this->input_buffer_.data_ptr();

  barrier_async(stream);
  if (!is_input_buffer_copied) {
    CUDA_CHECK(cudaMemcpyAsync(
        ptr_offset(input_buffer_ptr, this->rank * chunk_size),
        input_ptr,
        chunk_size,
        cudaMemcpyDefault,
        stream));

    // only input scale of s8 gemm requires all gather.
    // TODO(houqi.1993) only S8 has input scale?
    if (input_scale.has_value()) {
      void *input_scale_ptr = input_scale->data_ptr();
      void *input_scale_buffer_ptr = this->input_scale_buffer_.data_ptr();
      size_t scale_size = input_scale->numel() * input_scale->element_size();

      CUDA_CHECK(cudaMemcpyAsync(
          ptr_offset(input_scale_buffer_ptr, this->rank * scale_size),
          input_scale_ptr,
          scale_size,
          cudaMemcpyDefault,
          stream));
    }
  }

  // reset counter && ag signal
  if (use_cuda_core_ag) {
    CUDA_CHECK(cudaMemsetAsync(
        this->counter_buffer.data_ptr(), 0, this->counter_buffer.nbytes(), stream));
  }
  CUDA_CHECK(
      cudaMemsetAsync(this->barrier_buffer.data_ptr(), 0, this->barrier_buffer.nbytes(), stream));
  for (int j = 0; j < SPLIT; ++j) {
    set_ready(this->rank, this->rank, j, stream);
  }
  barrier_async(stream);
}

void
AllGatherOp::AllGatherOpImpl::barrier_async(cudaStream_t stream) {
  flux_barrier_all_on_stream(
      stream, this->sync_buffers, this->rank, this->local_world_size > kMaxLocalWorldSize);
}

void
AllGatherOp::AllGatherOpImpl::copy_local_and_sync_with_kernel(
    const torch::Tensor &input,
    const c10::optional<torch::Tensor> &input_scale,
    bool input_buffer_copied,
    bool fuse_sync,
    cudaStream_t stream) {
  // if local_copy is True, `copy_local` has been called by user.
  // Just reset ag_barrier and counter, and set m to 0 to avoid redudant copy.
  int32_t m = input_buffer_copied ? 0 : input.size(0);
  int32_t n = input.size(1);
  bool has_input_scale = this->with_input_scale && input_scale.has_value();
  void *scale_src = has_input_scale ? input_scale->data_ptr() : nullptr;

  size_t chunk_size = input.nbytes();
  size_t scale_size = has_input_scale ? input_scale->nbytes() : 0;

  void *input_dst = ptr_offset((void *)this->input_buffer_.data_ptr(), this->rank * chunk_size);
  void *scale_dst =
      has_input_scale
          ? ptr_offset((void *)this->input_scale_buffer_.data_ptr(), this->rank * scale_size)
          : nullptr;
  c10::ScalarType torch_scale_dtype =
      has_input_scale ? input_scale->scalar_type() : c10::ScalarType::Float;
  if (!fuse_sync) {
    barrier_async(stream);
  }
  local_copy_and_reset_impl(
      input.data_ptr(),
      input_dst,
      scale_src,
      scale_dst,
      (int32_t *)this->counter_buffer.data_ptr(),
      this->barrier_ptrs[this->rank],
      this->world_size,
      this->rank,
      m,
      n,
      fuse_sync ? (int **)this->sync_ptrs_buffer.data_ptr() : nullptr,
      ths_op::from_torch_dtype(this->input_dtype),
      ths_op::from_torch_dtype(torch_scale_dtype),
      this->sync_in_ring_mode,
      stream);
  if (!fuse_sync) {
    barrier_async(stream);
  }
}

void
AllGatherOp::AllGatherOpImpl::set_ready(
    int rank_, int segment, int split_index, cudaStream_t stream) {
  CU_CHECK(CUStreamWriteValue(
      stream,
      (CUdeviceptr)(barrier_ptrs[rank_] + (segment * SPLIT + split_index)),
      1,
      CU_STREAM_WRITE_VALUE_DEFAULT));
}

void
AllGatherOp::AllGatherOpImpl::wait_ready(
    int rank_, int segment, int split_index, cudaStream_t stream) {
  CU_CHECK(CUStreamWaitValue(
      stream,
      (CUdeviceptr)(barrier_ptrs[rank_] + (segment * SPLIT + split_index)),
      1,
      CU_STREAM_WRITE_VALUE_DEFAULT));
}

void
AllGatherOp::AllGatherOpImpl::copy_all_to_all(
    torch::Tensor input,
    c10::optional<torch::Tensor> input_scale,
    bool use_cuda_core,
    cudaStream_t stream) {
  size_t chunk_size = input.nbytes();
  size_t split_chunk_size = chunk_size / SPLIT;
  bool has_input_scale = this->with_input_scale && input_scale.has_value();
  if (use_cuda_core) {
    // use d2d copy kernel for s8 gemm
    c10::ScalarType torch_scale_dtype =
        has_input_scale ? input_scale->scalar_type() : c10::ScalarType::Float;
    this->copy_param.m = input.size(0);
    this->copy_param.n = input.size(1);
    ag_a2a_mode(
        this->copy_param,
        ths_op::from_torch_dtype(this->input_dtype),
        ths_op::from_torch_dtype(torch_scale_dtype),
        stream);
  } else {
    for (int i = rank + 1; i < (world_size + rank); ++i) {
      auto id = i % this->world_size;
      for (int j = 0; j < SPLIT; ++j) {
        auto split_offset = j * split_chunk_size;
        CUDA_CHECK(cudaMemcpyAsync(
            ptr_offset(this->input_ptrs[rank], id * chunk_size + split_offset),
            ptr_offset(this->input_ptrs[id], id * chunk_size + split_offset),
            split_chunk_size,
            cudaMemcpyDefault,
            stream));
        if (has_input_scale) {
          size_t chunk_size = input_scale->nbytes();
          size_t split_chunk_size = chunk_size / SPLIT;
          CUDA_CHECK(cudaMemcpyAsync(
              ptr_offset(this->input_scale_ptrs[this->rank], id * chunk_size + split_offset),
              ptr_offset(this->input_scale_ptrs[id], id * chunk_size + split_offset),
              split_chunk_size,
              cudaMemcpyDefault,
              stream));
        }
        set_ready(this->rank, id, j, stream);
      }
    }
  }
}

void
AllGatherOp::AllGatherOpImpl::copy_ring_pull(
    torch::Tensor input, c10::optional<torch::Tensor> input_scale, cudaStream_t stream) {
  // barrier_ptrs[rank, segment, split] means rank data is ready
  size_t chunk_size = input.nbytes();
  size_t split_chunk_size = chunk_size / SPLIT;
  // always the  0 <- 1 <- 2 <- 3 <- 0 order
  int from_rank = (this->rank + 1) % this->world_size;  // always copy to rank next
  for (int i = 0; i < this->world_size - 1; i++) {
    int recv_segment = (this->rank + i + 1) % this->world_size;  // copy from self
    for (int j = 0; j < SPLIT; ++j) {
      auto split_offset = j * split_chunk_size;
      if (i != 0) {
        // previous rank recv done
        wait_ready(from_rank, recv_segment, j, stream);
      }
      CUDA_CHECK(cudaMemcpyAsync(
          ptr_offset(this->input_ptrs[this->rank], recv_segment * chunk_size + split_offset),
          ptr_offset(this->input_ptrs[from_rank], recv_segment * chunk_size + split_offset),
          split_chunk_size,
          cudaMemcpyDeviceToDevice,
          stream));

      if (input_scale.has_value()) {
        size_t chunk_size = input_scale->nbytes();
        size_t split_chunk_size = chunk_size / SPLIT;
        CUDA_CHECK(cudaMemcpyAsync(
            ptr_offset(
                this->input_scale_ptrs[this->rank], recv_segment * chunk_size + split_offset),
            ptr_offset(
                this->input_scale_ptrs[from_rank], recv_segment * chunk_size + split_offset),
            split_chunk_size,
            cudaMemcpyDeviceToDevice,
            stream));
      }

      set_ready(this->rank, recv_segment, j, stream);
    }
  }
}

void
AllGatherOp::AllGatherOpImpl::copy_ring_push_1d(
    torch::Tensor input, c10::optional<torch::Tensor> input_scale, cudaStream_t stream) {
  size_t chunk_size = input.nbytes();
  size_t split_chunk_size = chunk_size / SPLIT;
  // always the  0 <- 1 <- 2 <- 3 <- 0 order
  int to_rank =
      (this->rank - 1 + this->world_size) % this->world_size;  // always recv data from rank prev
  for (int i = 0; i < this->world_size - 1; i++) {
    int send_segment = (this->rank + i) % this->world_size;
    for (int j = 0; j < SPLIT; ++j) {
      auto split_offset = j * split_chunk_size;
      if (i != 0) {  // previous rank recv done. i == 0 it is always ready
        wait_ready(this->rank, send_segment, j, stream);
      }
      CUDA_CHECK(cudaMemcpyAsync(
          ptr_offset(this->input_ptrs[to_rank], send_segment * chunk_size + split_offset),
          ptr_offset(this->input_ptrs[this->rank], send_segment * chunk_size + split_offset),
          split_chunk_size,
          cudaMemcpyDeviceToDevice,
          stream));

      if (input_scale.has_value()) {
        size_t chunk_size = input_scale->nbytes();
        size_t split_chunk_size = chunk_size / SPLIT;
        CUDA_CHECK(cudaMemcpyAsync(
            ptr_offset(this->input_scale_ptrs[to_rank], send_segment * chunk_size + split_offset),
            ptr_offset(
                this->input_scale_ptrs[this->rank], send_segment * chunk_size + split_offset),
            split_chunk_size,
            cudaMemcpyDeviceToDevice,
            stream));
      }
      set_ready(to_rank, send_segment, j, stream);
    }
  }
}

void
AllGatherOp::AllGatherOpImpl::copy_ring_push_2d_pcie(
    torch::Tensor input, c10::optional<torch::Tensor> input_scale, cudaStream_t stream) {
  size_t chunk_size = input.nbytes();
  size_t split_chunk_size = chunk_size / SPLIT;
  // [0, numa_world_size) stages:  0 <- 1 <- 2 <- 3 <- 4 <- 5 <- 6 <- 7 <- 0
  // [numa_world_size, world_size) stages: 0 <- 1 <- 2 <-3 <- 0 && 4 <- 5 <- 6 <- 7 <- 4
  int to_rank = (rank - 1 + world_size) % world_size;  // always recv data from rank prev
  int numa_world_size = topo_utils::topo_numa_local_world_size();
  FLUX_CHECK_DIV(this->local_world_size, numa_world_size);
  int numa_nodes = this->local_world_size / numa_world_size;
  FLUX_CHECK_EQ(numa_nodes, 2) << " world_size " << this->local_world_size
                               << " with numa_world_size " << numa_world_size;
  int nnode = rank / numa_world_size;
  for (int i = 0; i < world_size - 1; i++) {  // with inner and intra numa node
    int send_segment = (rank + i) % world_size;
    bool is_2d_step = i >= numa_world_size && rank % numa_world_size == 0;
    if (is_2d_step) {
      send_segment = (send_segment + numa_world_size) % world_size;
      to_rank = (rank - 1 + numa_world_size) % numa_world_size + nnode * numa_world_size;
    }
    for (int j = 0; j < SPLIT; ++j) {
      auto split_offset = j * split_chunk_size;
      if (i != 0 && !is_2d_step) {  // for i == 0 it is always ready
        // previous rank recv done
        wait_ready(rank, send_segment, j, stream);
      }
      CUDA_CHECK(cudaMemcpyAsync(
          ptr_offset(this->input_ptrs[to_rank], send_segment * chunk_size + split_offset),
          ptr_offset(this->input_ptrs[rank], send_segment * chunk_size + split_offset),
          split_chunk_size,
          cudaMemcpyDeviceToDevice,
          stream));

      if (input_scale.has_value()) {
        size_t chunk_size = input_scale->nbytes();
        size_t split_chunk_size = chunk_size / SPLIT;
        CUDA_CHECK(cudaMemcpyAsync(
            ptr_offset(this->input_scale_ptrs[to_rank], send_segment * chunk_size + split_offset),
            ptr_offset(
                this->input_scale_ptrs[this->rank], send_segment * chunk_size + split_offset),
            split_chunk_size,
            cudaMemcpyDeviceToDevice,
            stream));
      }

      set_ready(to_rank, send_segment, j, stream);
    }
  }
}

void
AllGatherOp::AllGatherOpImpl::copy_ring_push_by_kernel(
    torch::Tensor input,
    c10::optional<torch::Tensor> input_scale,
    bool use_2d_mode,
    cudaStream_t stream) {
  bool has_input_scale = input_scale.has_value();
  // use d2d copy kernel for s8 gemm
  c10::ScalarType torch_scale_dtype =
      has_input_scale ? input_scale->scalar_type() : c10::ScalarType::Float;
  this->copy_param.m = input.size(0);
  this->copy_param.n = input.size(1);
  this->copy_param.has_scale = has_input_scale;
  ag_ring_with_scale(
      this->copy_param,
      c10::elementSize(this->input_dtype),
      c10::elementSize(torch_scale_dtype),
      4,  // TODO(houqi.1993) for PCI-e 4 is always enough
      use_2d_mode,
      stream);
}

AllGatherOp::AllGatherOpImpl::~AllGatherOpImpl() { cudaEventDestroy(local_prepare_event); }

AllGatherOp::AllGatherOp(
    std::shared_ptr<Group> tp_group,
    int nnodes,
    size_t max_m,
    size_t k,
    at::ScalarType input_dtype)
    : impl_(new AllGatherOpImpl(tp_group, nnodes, max_m, k, input_dtype)) {}

AllGatherOp::~AllGatherOp() { delete impl_; }

void
AllGatherOp::run_with_optional_options(
    torch::Tensor input,
    c10::optional<torch::Tensor> input_scale,
    const AllGatherOptionWithOptional &opt,
    cudaStream_t stream) {
  FLUX_CHECK(impl_ != nullptr) << "AllGatherOp is not initialized";
  impl_->run_with_optional_options(input, input_scale, opt, stream);
}

void
AllGatherOp::run(
    const torch::Tensor &input,
    c10::optional<torch::Tensor> input_scale,
    const AllGatherOption &opt,
    cudaStream_t stream) {
  FLUX_CHECK(impl_ != nullptr) << "AllGatherOp is not initialized";
  impl_->run(input, input_scale, opt, stream);
}

// only provide local tensor
torch::Tensor
AllGatherOp::local_input_buffer() {
  FLUX_CHECK(impl_ != nullptr) << "AllGatherOp is not initialized";
  return impl_->local_input_buffer();
}
torch::Tensor
AllGatherOp::local_input_scale_buffer() {
  FLUX_CHECK(impl_ != nullptr) << "AllGatherOp is not initialized";
  return impl_->local_input_scale_buffer();
}
torch::Tensor
AllGatherOp::local_barrier_buffer() {
  FLUX_CHECK(impl_ != nullptr) << "AllGatherOp is not initialized";
  return impl_->local_barrier_buffer();
}

int32_t *
AllGatherOp::ag_signal_ptr() const {
  FLUX_CHECK(impl_ != nullptr) << "AllGatherOp is not initialized";
  return impl_->ag_signal_ptr();
}

cudaEvent_t &
AllGatherOp::get_local_prepare_event() {
  FLUX_CHECK(impl_ != nullptr) << "AllGatherOp is not initialized";
  return impl_->get_local_prepare_event();
}

}  // namespace bytedance::flux::ths_op
