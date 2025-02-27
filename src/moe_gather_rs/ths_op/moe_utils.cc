//===- moe_utils.cc ----------------------------------------------- C++ ---===//
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

#include "flux/flux.h"
#include "flux/ths_op/ths_op.h"
#include "flux/ths_op/util.h"
#include "torch/all.h"
#include "torch/library.h"
#include <c10/cuda/CUDAStream.h>
#include "flux/cuda/cuda_common.h"
#include <nvshmem.h>

#include "moe_gather_rs/moe_utils.h"
#include "moe_gather_rs/ths_op/moe_utils.h"

using torch::Tensor;

namespace bytedance {
namespace flux {
namespace ths_op {

Tensor
sort(Tensor input) {
  TORCH_CHECK(input.scalar_type() == c10::ScalarType::Long);
  Tensor output = torch::empty_like(input);

  Tensor index = torch::arange(0, input.numel(), input.options());
  Tensor sorted_index = torch::empty_like(index);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  sort_impl(
      input.numel(),
      static_cast<uint64_t *>(input.data_ptr()),
      static_cast<uint64_t *>(output.data_ptr()),
      static_cast<uint64_t *>(index.data_ptr()),
      static_cast<uint64_t *>(sorted_index.data_ptr()),
      stream);
  return sorted_index;
}

Tensor
setup_shared_memory(
    int64_t rank, int64_t world_size, Tensor local_data, std::vector<void *> *host_ptrs) {
  TORCH_CHECK(
      host_ptrs->size() == (size_t)world_size,
      "host_ptrs.size():" + std::to_string(host_ptrs->size()) +
          " != world_size:" + std::to_string(world_size));

  for (int64_t i = 0; i < world_size; ++i) {
    if (i == rank) {
      host_ptrs->at(i) = local_data.data_ptr();
    } else {
      host_ptrs->at(i) = nvshmem_ptr(local_data.data_ptr(), i);
    }
  }

  // copy host_ptrs to device memory
  int64_t nbytes = world_size * sizeof(void *);
  Tensor dev_ptrs = torch::empty({nbytes}, local_data.options().dtype(at::ScalarType::Byte));
  cudaMemcpy(dev_ptrs.data_ptr(), host_ptrs->data(), nbytes, cudaMemcpyHostToDevice);
  return dev_ptrs;
}

void
calculate_prepared_nums(
    int64_t num_experts,
    int64_t num_tokens,
    Tensor splits_gpu,
    Tensor splits_offsets,
    Tensor last_src,
    Tensor prepared_nums) {
  TORCH_CHECK(splits_gpu.scalar_type() == c10::ScalarType::Int);
  TORCH_CHECK(splits_offsets.scalar_type() == c10::ScalarType::Int);
  TORCH_CHECK(last_src.scalar_type() == c10::ScalarType::Long);
  TORCH_CHECK(prepared_nums.scalar_type() == c10::ScalarType::Int);
  TORCH_CHECK(splits_gpu.size(0) == num_experts);
  TORCH_CHECK(splits_offsets.size(0) == num_experts);
  TORCH_CHECK(last_src.size(0) == num_tokens);
  TORCH_CHECK(prepared_nums.size(0) == num_experts);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  calculate_prepared_nums_impl(
      num_experts,
      num_tokens,
      static_cast<int32_t *>(splits_gpu.data_ptr()),
      static_cast<int32_t *>(splits_offsets.data_ptr()),
      static_cast<index_t *>(last_src.data_ptr()),
      static_cast<int32_t *>(prepared_nums.data_ptr()),
      stream);
}

void
calculate_transport_info(
    int64_t num_experts,
    int64_t world_size,
    Tensor prepared_order,
    Tensor prepared_offsets,
    Tensor transport_nums) {
  TORCH_CHECK(prepared_order.scalar_type() == c10::ScalarType::Long);
  TORCH_CHECK(prepared_offsets.scalar_type() == c10::ScalarType::Long);
  TORCH_CHECK(transport_nums.scalar_type() == c10::ScalarType::Long);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  calculate_transport_info_impl(
      num_experts,
      world_size,
      static_cast<index_t *>(prepared_order.data_ptr()),
      static_cast<index_t *>(prepared_offsets.data_ptr()),
      static_cast<index_t *>(transport_nums.data_ptr()),
      stream);
}

void
scatter_add(
    int64_t world_size,
    int64_t tokens_per_rank,
    int64_t hidden_size,
    Tensor index,
    Tensor input,
    Tensor output) {
  TORCH_CHECK(index.scalar_type() == c10::ScalarType::Long);
  TORCH_CHECK(input.scalar_type() == c10::ScalarType::Half);
  TORCH_CHECK(output.scalar_type() == c10::ScalarType::Half);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  scatter_add_impl(
      world_size,
      tokens_per_rank,
      hidden_size,
      static_cast<index_t *>(index.data_ptr()),
      static_cast<void *>(input.data_ptr()),
      static_cast<void *>(output.data_ptr()),
      stream);
}

void
index_put(int64_t num_tokens, int64_t topk, Tensor index, Tensor data, bool value) {
  TORCH_CHECK(index.scalar_type() == c10::ScalarType::Long);
  TORCH_CHECK(data.scalar_type() == c10::ScalarType::Bool);
  TORCH_CHECK(index.size(0) == num_tokens);
  TORCH_CHECK(data.size(0) == num_tokens * topk);

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  index_put_impl(
      num_tokens,
      topk,
      static_cast<index_t *>(index.data_ptr()),
      static_cast<bool *>(data.data_ptr()),
      value,
      stream);
}

TORCH_LIBRARY(flux, m) {
  m.def("calculate_prepared_nums", calculate_prepared_nums);
  m.def("calculate_transport_info", calculate_transport_info);
  m.def("scatter_add", scatter_add);
  m.def("index_put", index_put);
  m.def("sort", sort);
}

class TransportOp::TransportOpImpl {
 public:
  TransportOpImpl(int64_t rank, int64_t world_size, torch::Tensor recv_buffer);

  void copy_by_sm(torch::Tensor send_buffer, Tensor transport_offsets, Tensor transport_nbytes);

  void copy_by_ce(torch::Tensor send_buffer, Tensor transport_offsets, Tensor transport_nbytes);

 private:
  const int64_t rank;
  const int64_t world_size;
  int64_t num_tokens;
  int64_t hidden_size;

  torch::Tensor recv_buffer;
  std::vector<void *> host_recv_buffer_ptrs;
  torch::Tensor dev_recv_buffer_ptrs;
};

TransportOp::TransportOpImpl::TransportOpImpl(
    int64_t rank, int64_t world_size, torch::Tensor recv_buffer)
    : rank(rank),
      world_size(world_size),
      recv_buffer(recv_buffer),
      host_recv_buffer_ptrs(world_size, nullptr) {
  // setup shared recv buffer
  this->dev_recv_buffer_ptrs =
      setup_shared_memory(rank, world_size, recv_buffer, &host_recv_buffer_ptrs);
  num_tokens = recv_buffer.size(0);
  hidden_size = recv_buffer.size(1);
}

void
TransportOp::TransportOpImpl::copy_by_sm(
    Tensor send_buffer, Tensor transport_offsets, Tensor transport_nbytes) {
  TORCH_CHECK(transport_offsets.size(0) == this->world_size);
  TORCH_CHECK(transport_offsets.scalar_type() == c10::ScalarType::Long);

  TORCH_CHECK(transport_nbytes.size(0) == this->world_size);
  TORCH_CHECK(transport_nbytes.scalar_type() == c10::ScalarType::Long);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  int64_t num_tokens = send_buffer.size(0);
  int64_t hidden_size = send_buffer.size(1);
  int64_t tokens_per_rank = num_tokens / this->world_size;

  transport_impl(
      this->rank,
      this->world_size,
      tokens_per_rank,
      hidden_size,
      send_buffer.data_ptr(),
      static_cast<index_t *>(transport_offsets.data_ptr()),
      static_cast<index_t *>(transport_nbytes.data_ptr()),
      reinterpret_cast<void **>(dev_recv_buffer_ptrs.data_ptr()),
      stream);
}

void
TransportOp::TransportOpImpl::copy_by_ce(
    Tensor send_buffer, Tensor transport_offsets, Tensor transport_nbytes) {
  TORCH_CHECK(transport_offsets.size(0) == this->world_size);
  TORCH_CHECK(transport_offsets.scalar_type() == c10::ScalarType::Long);

  TORCH_CHECK(transport_nbytes.size(0) == this->world_size);
  TORCH_CHECK(transport_nbytes.scalar_type() == c10::ScalarType::Long);

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  int64_t size_of_elem = 2;
  int64_t bytes_per_row = hidden_size * size_of_elem;
  int64_t tokens_per_rank = num_tokens / world_size;
  for (int64_t dst_rank = 0; dst_rank < this->world_size; ++dst_rank) {
    int64_t transport_offset =
        *(reinterpret_cast<int64_t *>(transport_offsets.data_ptr()) + dst_rank);

    int64_t send_offset = dst_rank * tokens_per_rank + transport_offset;
    void *src_ptr = reinterpret_cast<char *>(send_buffer.data_ptr()) + send_offset * bytes_per_row;

    int64_t recv_offset = this->rank * tokens_per_rank + transport_offset;
    void *dst_ptr = reinterpret_cast<char *>(this->host_recv_buffer_ptrs[dst_rank]) +
                    recv_offset * bytes_per_row;

    int64_t transport_nbyte =
        *(reinterpret_cast<int64_t *>(transport_nbytes.data_ptr()) + dst_rank);
    CUDA_CHECK(
        cudaMemcpyAsync(dst_ptr, src_ptr, transport_nbyte, cudaMemcpyDeviceToDevice, stream));
  }
}

TransportOp::TransportOp(int64_t rank, int64_t world_size, torch::Tensor recv_buffer)
    : impl_(new TransportOpImpl(rank, world_size, recv_buffer)) {}

void
TransportOp::copy_by_sm(
    torch::Tensor send_buffer, torch::Tensor transport_offsets, torch::Tensor transport_nbytes) {
  FLUX_CHECK(impl_ != nullptr) << "TransportOpImpl is not initialized";
  impl_->copy_by_sm(send_buffer, transport_offsets, transport_nbytes);
}

void
TransportOp::copy_by_ce(
    torch::Tensor send_buffer, torch::Tensor transport_offsets, torch::Tensor transport_nbytes) {
  FLUX_CHECK(impl_ != nullptr) << "TransportOpImpl is not initialized";
  impl_->copy_by_ce(send_buffer, transport_offsets, transport_nbytes);
}

class All2AllOp::All2AllOpImpl {
 public:
  All2AllOpImpl(int64_t rank, int64_t world_size, torch::Tensor recv_buffer);
  ~All2AllOpImpl() = default;

  void forward(c10::List<Tensor> send_buffer);

 private:
  const int32_t rank;
  const int32_t world_size;
  torch::Tensor recv_buffer;
  std::vector<void *> host_recv_buffer_ptrs;
};

All2AllOp::All2AllOpImpl::All2AllOpImpl(int64_t rank, int64_t world_size, Tensor recv_buffer)
    : rank(rank),
      world_size(world_size),
      recv_buffer(recv_buffer),
      host_recv_buffer_ptrs(world_size, nullptr) {
  TORCH_CHECK(recv_buffer.size(0) == this->world_size);
  TORCH_CHECK(recv_buffer.scalar_type() == c10::ScalarType::Long);

  setup_shared_memory(rank, world_size, recv_buffer, &host_recv_buffer_ptrs);
}

void
All2AllOp::All2AllOpImpl::forward(c10::List<Tensor> send_buffer) {
  TORCH_CHECK(send_buffer.size() == (size_t)this->world_size);
  TORCH_CHECK(send_buffer.get(0).size(0) == this->recv_buffer.size(1));
  TORCH_CHECK(send_buffer.get(0).scalar_type() == c10::ScalarType::Long);

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  int64_t nbytes_per_block = this->recv_buffer.size(1) * sizeof(index_t);
  for (int64_t dst_rank = 0; dst_rank < this->world_size; ++dst_rank) {
    void *src_ptr = send_buffer.get(dst_rank).data_ptr();
    void *dst_ptr = reinterpret_cast<char *>(this->host_recv_buffer_ptrs[dst_rank]) +
                    this->rank * nbytes_per_block;
    CUDA_CHECK(
        cudaMemcpyAsync(dst_ptr, src_ptr, nbytes_per_block, cudaMemcpyDeviceToDevice, stream));
  }
}

All2AllOp::All2AllOp(int64_t rank, int64_t world_size, torch::Tensor recv_buffer)
    : impl_(new All2AllOpImpl(rank, world_size, recv_buffer)) {}
All2AllOp::~All2AllOp() { delete impl_; }

void
All2AllOp::forward(c10::List<torch::Tensor> send_buffer) {
  FLUX_CHECK(impl_ != nullptr) << "All2AllOp is not initialized";
  impl_->forward(send_buffer);
}

}  // namespace ths_op
}  // namespace flux
}  // namespace bytedance
