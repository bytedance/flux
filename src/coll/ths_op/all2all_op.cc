//===- all2all_op.cc ------------------------------------------ C++ ---===//
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
#include <coll/ths_op/all2all_op.h>
#include <cuda_runtime_api.h>
#include <cutlass/gemm/gemm.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <torch/all.h>

#include <coll/all2all_impl.hpp>
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

class All2AllInference::All2AllInferenceOpImpl {
 private:
  int32_t max_m;
  int32_t n_dim;
  int32_t rank;
  int32_t total_num_experts;
  int32_t expert_per_rank;
  int32_t world_size;
  int32_t local_world_size;
  int32_t max_element_size;
  int32_t call_count;
  torch::Tensor splits_input_buffer;
  torch::Tensor splits_output_buffer;
  torch::Tensor signal_buffer;
  torch::Tensor input_buffer;
  torch::Tensor output_buffer;
  torch::Tensor scale_input_buffer;
  torch::Tensor scale_output_buffer;

  at::ScalarType
  size_to_dtype(int element_size) {
    at::ScalarType _st;
    if (element_size == 1) {
#if TORCH_SUPPORT_FP8
      _st = c10::ScalarType::Float8_e4m3fn;
#else
      FLUX_CHECK(false);
#endif
    } else if (element_size == 2) {
      _st = c10::ScalarType::BFloat16;
    } else if (element_size == 4) {
      _st = c10::ScalarType::Int;
    } else if (element_size == 8) {
      _st = c10::ScalarType::Long;
    } else {
      FLUX_CHECK(false);
    }
    return _st;
  }

  void
  init_buffer() {
    // initialize the comm buffer only once
    FLUX_CHECK(!input_buffer.defined());
    FLUX_CHECK(!output_buffer.defined());
    auto _st = size_to_dtype(this->max_element_size);
    this->splits_input_buffer =
        nvshmem_create_tensor({this->total_num_experts}, c10::ScalarType::Int);
    this->splits_output_buffer =
        nvshmem_create_tensor({this->total_num_experts * 2}, c10::ScalarType::Int);
    this->signal_buffer = nvshmem_create_tensor({this->world_size * 4}, c10::ScalarType::Long);
    this->input_buffer = nvshmem_create_tensor({this->max_m, this->n_dim}, _st);
    this->output_buffer =
        nvshmem_create_tensor({this->world_size * this->max_m * 2, this->n_dim}, _st);
    this->scale_input_buffer = nvshmem_create_tensor({this->max_m}, c10::ScalarType::Float);
    this->scale_output_buffer =
        nvshmem_create_tensor({this->world_size * this->max_m * 2}, c10::ScalarType::Float);
  }

  torch::Tensor
  change_dtype(torch::Tensor input, int element_size) {
    auto _st = size_to_dtype(element_size);
    torch::Tensor re = at::from_blob(
        const_cast<void *>(input.data_ptr()), input.sizes(), input.options().dtype(_st));
    return re;
  }

 public:
  All2AllInferenceOpImpl(
      int64_t max_m,
      int64_t n_dim,
      int64_t rank,
      int64_t total_num_experts,
      int64_t world_size,
      int64_t local_world_size,
      int64_t max_element_size)
      : max_m(max_m),
        n_dim(n_dim),
        rank(rank),
        total_num_experts(total_num_experts),
        world_size(world_size),
        local_world_size(local_world_size),
        max_element_size(max_element_size) {
    FLUX_CHECK(rank < world_size);
    FLUX_CHECK(world_size % local_world_size == 0);
    init_buffer();
    FLUX_CHECK(this->total_num_experts % this->world_size == 0);
    this->expert_per_rank = this->total_num_experts / this->world_size;
    this->call_count = 1;  // start from 1, becase the initial values of signal buffer is 0
  }

  std::vector<torch::Tensor>
  get_input_buffer(std::vector<int32_t> input_shape, int64_t element_size, bool with_scale) {
    std::vector<torch::Tensor> out_vec;
    // last op before all2all should directly write the output into the nvshmem comm buffer
    FLUX_CHECK(element_size <= this->max_element_size);
    FLUX_CHECK(input_shape.size() == 2);
    FLUX_CHECK(input_shape[0] <= this->max_m);
    FLUX_CHECK(input_shape[1] == this->n_dim);
    // data buffer may has the wrong data type, but has sufficient space
    auto tmp_input = change_dtype(this->input_buffer, element_size);
    out_vec.push_back(tmp_input.slice(0, 0, input_shape[0]));
    if (with_scale) {
      out_vec.push_back(this->scale_input_buffer.slice(0, 0, input_shape[0]));
    }
    return out_vec;
  }

  std::vector<torch::Tensor>
  forward(
      std::vector<int32_t> input_shape,
      torch::Tensor input_split_cumsum,
      int64_t element_size,
      bool with_scale,
      cudaStream_t stream) {
    FLUX_CHECK(input_split_cumsum.device().is_cuda())
        << "input/output split size tensor should be placed on CPU";
    FLUX_CHECK(
        input_split_cumsum.dim() == 1 &&
        input_split_cumsum.size(0) == (this->total_num_experts + 1))
        << "Expect the input splits has the shape with (num_experts + 1)";
    FLUX_CHECK(input_split_cumsum.dtype() == at::ScalarType::Int);
    FLUX_CHECK(element_size <= this->max_element_size);
    // FLUX_CHECK(output_split_sizes.dtype() == at::ScalarType::Int);
    std::vector<torch::Tensor> output_vec;
    int act_pos = this->call_count % 2;
    int signal_buffer_start = act_pos * this->world_size;
    int signal_buffer_end = signal_buffer_start + this->world_size;
    int splits_buffer_start = act_pos * this->total_num_experts;
    int splits_buffer_end = splits_buffer_start + this->total_num_experts;
    int data_buffer_start = act_pos * this->world_size * this->max_m;
    int data_buffer_end = data_buffer_start + this->world_size * this->max_m;
    int *splits_input_ptr = reinterpret_cast<int *>(this->splits_input_buffer.data_ptr());
    int *splits_output_ptr =
        reinterpret_cast<int *>(this->splits_output_buffer.data_ptr()) + splits_buffer_start;

    uint64_t *signal_buffer_ptr =
        reinterpret_cast<uint64_t *>(this->signal_buffer.data_ptr()) + signal_buffer_start;
    char *output_buffer_ptr = reinterpret_cast<char *>(this->output_buffer.data_ptr()) +
                              data_buffer_start * this->n_dim * element_size;
    float *scale_input_buffer_ptr = reinterpret_cast<float *>(this->scale_input_buffer.data_ptr());
    float *scale_output_buffer_ptr =
        reinterpret_cast<float *>(this->scale_output_buffer.data_ptr()) + data_buffer_start;
    All2allParams args{
        this->input_buffer.data_ptr(),
        output_buffer_ptr,
        splits_input_ptr,
        splits_output_ptr,
        signal_buffer_ptr,
        reinterpret_cast<int32_t *>(input_split_cumsum.data_ptr()),
        scale_input_buffer_ptr,
        scale_output_buffer_ptr,
        this->rank,
        this->world_size,
        this->n_dim,
        element_size,
        this->max_m,
        this->expert_per_rank,
        this->call_count,
        with_scale};
    all2all_impl(args, stream);
    std::vector<torch::Tensor> out_vec;

    out_vec.push_back(this->splits_output_buffer.slice(0, splits_buffer_start, splits_buffer_end));
    torch::Tensor data_tensor = change_dtype(this->output_buffer, element_size);
    out_vec.push_back(data_tensor.slice(0, data_buffer_start, data_buffer_end));
    if (with_scale)
      out_vec.push_back(this->scale_output_buffer.slice(0, data_buffer_start, data_buffer_end));
    this->call_count = (this->call_count + 1) % MOD_VALUE;

    return out_vec;
  }
};

All2AllInference::All2AllInference(
    int64_t max_m,
    int64_t n_dim,
    int64_t rank,
    int64_t total_num_experts,
    int64_t world_size,
    int64_t local_world_size,
    int64_t max_element_size)
    : impl_(new All2AllInferenceOpImpl(
          max_m, n_dim, rank, total_num_experts, world_size, local_world_size, max_element_size)) {
}

All2AllInference::~All2AllInference() { delete impl_; }

std::vector<torch::Tensor>
All2AllInference::forward(
    std::vector<int32_t> input_shape,
    torch::Tensor input_split_cumsum,
    int64_t element_size,
    bool with_scale) {
  FLUX_CHECK(impl_ != nullptr) << "All2AllInference is not initialized!";
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  return this->forward_with_stream(
      input_shape, input_split_cumsum, element_size, with_scale, stream);
}

std::vector<torch::Tensor>
All2AllInference::forward_with_stream(
    std::vector<int32_t> input_shape,
    torch::Tensor input_split_cumsum,
    int64_t element_size,
    bool with_scale,
    cudaStream_t stream) {
  FLUX_CHECK(impl_ != nullptr) << "All2AllInference is not initialized!";
  return impl_->forward(input_shape, input_split_cumsum, element_size, with_scale, stream);
}

std::vector<torch::Tensor>
All2AllInference::get_input_buffer(
    std::vector<int32_t> input_shape, int64_t element_size, bool with_scale) {
  FLUX_CHECK(impl_ != nullptr) << "All2AllInference is not initialized!";
  return impl_->get_input_buffer(input_shape, element_size, with_scale);
}

}  // namespace ths_op
}  // namespace flux
}  // namespace bytedance
