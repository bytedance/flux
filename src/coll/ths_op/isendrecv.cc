//===- isendrecv.cc ------------------------------------------ C++ ---===//
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
#include "coll/ths_op/isendrecv.h"

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
#include <cutlass/util/packed_stride.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <vector>

#include "flux/flux.h"
#include "flux/ths_op/flux_shm.h"
#define MOD_VALUE 1000000
namespace bytedance {
namespace flux {
namespace ths_op {

using torch::Tensor;

class AsyncSendRecv::AsyncSendRecvOpImpl {
 private:
  int32_t max_m;
  int32_t n_dim;
  int32_t rank;
  int32_t world_size;
  at::ScalarType dtype;
  int32_t duplicate;
  std::vector<torch::Tensor> data_buffers;
  std::vector<torch::Tensor> signal_buffers;
  void
  init_buffer() {
    FLUX_CHECK(data_buffers.empty());
    FLUX_CHECK(signal_buffers.empty());
    auto current_device = c10::cuda::current_device();
    auto option_gpu =
        at::TensorOptions().dtype(this->dtype).device(at::kCUDA).device_index(current_device);
    for (int i = 0; i < duplicate; i++) {
      data_buffers.push_back(nvshmem_create_tensor({max_m, n_dim}, this->dtype));
    }
    for (int i = 0; i < duplicate; i++) {
      // may lead to memory segmentation?
      signal_buffers.push_back(nvshmem_create_tensor({1}, c10::ScalarType::Long));
    }
  }

 public:
  AsyncSendRecvOpImpl(
      int64_t max_m,
      int64_t n_dim,
      int64_t rank,
      int64_t world_size,
      at::ScalarType input_dtype,
      int64_t duplicate)
      : max_m(max_m),
        n_dim(n_dim),
        rank(rank),
        world_size(world_size),
        dtype(input_dtype),
        duplicate(duplicate) {
    init_buffer();
  }

  torch::Tensor
  get_comm_buffer(int64_t comm_buffer_id) {
    return data_buffers[comm_buffer_id];
  }
  torch::Tensor
  read_comm_buffer(int64_t tgt_rank, int64_t src_comm_buffer_id, int64_t tgt_comm_buffer_id) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    nvshmemx_getmem_on_stream(
        data_buffers[tgt_comm_buffer_id].data_ptr(),
        data_buffers[src_comm_buffer_id].data_ptr(),
        data_buffers[src_comm_buffer_id].nbytes(),
        tgt_rank,
        stream);
    return data_buffers[tgt_comm_buffer_id];
  }

  void
  write_comm_buffer(int64_t tgt_rank, int64_t src_comm_buffer_id, int64_t tgt_comm_buffer_id) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    nvshmemx_putmem_on_stream(
        data_buffers[tgt_comm_buffer_id].data_ptr(),
        data_buffers[src_comm_buffer_id].data_ptr(),
        data_buffers[src_comm_buffer_id].nbytes(),
        tgt_rank,
        stream);
    // TODO whether need to call nvshmem_quiet()?
    // nvshmemx_quiet_on_stream(stream);
  }

  void
  set_signal(int64_t tgt_rank, int64_t comm_buffer_id, int64_t value) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    nvshmemx_signal_op_on_stream(
        reinterpret_cast<uint64_t *>(signal_buffers[comm_buffer_id].data_ptr()),
        value,
        NVSHMEM_SIGNAL_SET,
        tgt_rank,
        stream);
  }
  void
  reset_signal(int64_t comm_buffer_id) {
    this->signal_buffers[comm_buffer_id].zero_();
  }
  void
  wait_signal_eq(int64_t comm_buffer_id, int64_t value) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    nvshmemx_signal_wait_until_on_stream(
        reinterpret_cast<uint64_t *>(signal_buffers[comm_buffer_id].data_ptr()),
        NVSHMEM_CMP_EQ,
        value,
        stream);
  }
};

AsyncSendRecv::AsyncSendRecv(
    int64_t max_m,
    int64_t n_dim,
    int64_t rank,        // rank in pp
    int64_t world_size,  // world_size of pp
    at::ScalarType input_dtype,
    int64_t duplicate)
    : impl_(new AsyncSendRecvOpImpl(max_m, n_dim, rank, world_size, input_dtype, duplicate)) {}

AsyncSendRecv::~AsyncSendRecv() { delete impl_; }

torch::Tensor
AsyncSendRecv::get_comm_buffer(int64_t comm_buff_id) {
  FLUX_CHECK(impl_ != nullptr) << "AsyncSendRecv is not initialized!";
  return impl_->get_comm_buffer(comm_buff_id);
}
torch::Tensor
AsyncSendRecv::read_comm_buffer(
    int64_t tgt_rank, int64_t src_comm_buffer_id, int64_t tgt_comm_buffer_id) {
  FLUX_CHECK(impl_ != nullptr) << "AsyncSendRecv is not initialized!";
  return impl_->read_comm_buffer(tgt_rank, src_comm_buffer_id, tgt_comm_buffer_id);
}
void
AsyncSendRecv::write_comm_buffer(
    int64_t tgt_rank, int64_t src_comm_buffer_id, int64_t tgt_comm_buffer_id) {
  FLUX_CHECK(impl_ != nullptr) << "AsyncSendRecv is not initialized!";
  return impl_->write_comm_buffer(tgt_rank, src_comm_buffer_id, tgt_comm_buffer_id);
}
void
AsyncSendRecv::set_signal(int64_t tgt_rank, int64_t comm_buffer_id, int64_t value) {
  FLUX_CHECK(impl_ != nullptr) << "AsyncSendRecv is not initialized!";
  return impl_->set_signal(tgt_rank, comm_buffer_id, value);
}
void
AsyncSendRecv::wait_signal_eq(int64_t comm_buffer_id, int64_t value) {
  FLUX_CHECK(impl_ != nullptr) << "AsyncSendRecv is not initialized!";
  return impl_->wait_signal_eq(comm_buffer_id, value);
}

void
AsyncSendRecv::reset_signal(int64_t comm_buffer_id) {
  FLUX_CHECK(impl_ != nullptr) << "AsyncSendRecv is not initialized!";
  return impl_->reset_signal(comm_buffer_id);
}

}  // namespace ths_op
}  // namespace flux
}  // namespace bytedance
