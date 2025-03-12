//===- gemm_grouped_v3_gather_rs.cc -------------------------------------------- C++ ---===//
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
#include "moe_gather_rs/ths_op/gemm_grouped_v3_gather_rs.h"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/cuda_stub.h"
#include "flux/gemm_meta.h"
#include "flux/op_registry.h"
#include "flux/ths_op/ths_op.h"
#include "flux/cuda/cuda_common.h"
#include "flux/flux.h"
#include "flux/ths_op/util.h"
#include "flux/args/moe_gather_rs.h"
#include "torch/all.h"
#include "cutlass/gemm/gemm.h"
#include <ATen/core/List.h>
#include <ATen/ops/empty.h>
#include <c10/core/ScalarType.h>
#include <c10/util/intrusive_ptr.h>
#include <cuda_runtime_api.h>
#include <ATen/core/jit_type.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAEvent.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <cstdlib>
#include <mutex>
#include <iostream>
#include <vector>
#include "cutlass/util/packed_stride.hpp"
#include "moe_gather_rs/ths_op/topk_reduce_gather_rs.h"
#include <nvshmem.h>
#include <nvshmemx.h>
#include "moe_gather_rs/moe_utils.h"

namespace bytedance {
namespace flux {
namespace ths_op {

using torch::Tensor;

/// This class only runs the basic grouped_gemm, it is mainly used for testing
class GemmGroupedV3GatherRS::GemmGroupedV3GatherRSOpImpl {
  using UnderlyingProblemShape = cute::Shape<int, int, int>;
  using StrideA = cute::tuple<int64_t, cute::C<1>, cute::C<0>>;
  // Stride for the weight tensor when it's layout is ColMajor
  using StrideB_Col = cute::tuple<int64_t, cute::C<1>, cute::C<0>>;
  // Stride for the weight tensor when it's layout is RowMajor
  using StrideB_Row = cute::tuple<cute::C<1>, int64_t, cute::C<0>>;
  using StrideC_Row = cute::tuple<int64_t, cute::C<1>, cute::C<0>>;
  using StrideC_Col = cute::tuple<cute::C<1>, int64_t, cute::C<0>>;
  using StrideC = StrideC_Col;
  using StrideD = StrideC;

  using ElementAccumulator = float;

 private:
  int SPLITS;
  at::ScalarType _st;
  torch::Tensor splits_gpu;
  int32_t num_experts;
  const int32_t total_num_experts;
  // Save for arguments used for the group gemm
  torch::Tensor host_buffer;
  torch::Tensor device_buffer;
  torch::Tensor workspace;
  size_t problem_sizes_offset;
  size_t ptr_a_offset;
  size_t ptr_b_offset;
  size_t ptr_c_offset;
  size_t ptr_d_offset;
  size_t lda_offset;
  size_t ldb_offset;
  size_t ldc_offset;
  size_t ldd_offset;
  size_t weight_scale_offset;
  size_t input_scale_offset;
  size_t output_vec_scale_offset;
  size_t interDs_offset;
  size_t total_bytes;
  int32_t rank;
  int32_t world_size;  // the total world size
  // Note: Flux gather-rs support multiple groups of input/weight tensors
  // the output results of different group gemms should have the same shape
  // and will be reduced first before the following reduce-scatter operator.
  // This is mainly used in the backward operation of the Swiglu.
  int32_t max_m;
  int32_t max_input_groups;
  int32_t n_dim;
  int32_t topk;
  int32_t tp_world_size;  // the world size of tensor parallel
  int32_t ep_world_size;  // the world size of expert parallel
  torch::Tensor output_buffer;
  torch::Tensor ptr_buffer;
  std::vector<torch::Tensor> inter_Ds;
  torch::Tensor barrier;
  torch::Tensor ep_indexes;
  torch::Tensor ep_indexes_count;
  cudaEvent_t ready_event;
  cudaEvent_t gather_rs_start_event;
  std::vector<at::cuda::CUDAStream> gather_rs_stream;
  std::vector<int32_t> splits_cum_sum;
  bool buffer_initialized;
  bool drop_token;  // whether to enable the token drop feature

  size_t
  args_offset_and_buffer_size(size_t problem_count, size_t input_groups) {
    // problem count here refers to the number of the experts
    // the size of two kinds of Stride object for B tensor should be the same
    static_assert(sizeof(StrideB_Col) == sizeof(StrideB_Row));
    static_assert(sizeof(StrideC_Col) == sizeof(StrideC_Row));
    this->problem_sizes_offset = 0;
    this->ptr_a_offset =
        CUDA_MEM_ALIGN(problem_sizes_offset + sizeof(UnderlyingProblemShape) * problem_count);
    this->ptr_b_offset = CUDA_MEM_ALIGN(ptr_a_offset + sizeof(void *) * problem_count);
    this->ptr_c_offset = CUDA_MEM_ALIGN(ptr_b_offset + sizeof(void *) * problem_count);
    this->ptr_d_offset = CUDA_MEM_ALIGN(ptr_c_offset + sizeof(void *) * problem_count);
    this->lda_offset = CUDA_MEM_ALIGN(ptr_d_offset + sizeof(void *) * problem_count);
    this->ldb_offset = CUDA_MEM_ALIGN(lda_offset + sizeof(StrideA) * problem_count);
    this->ldc_offset = CUDA_MEM_ALIGN(ldb_offset + sizeof(StrideB_Col) * problem_count);
    this->ldd_offset = CUDA_MEM_ALIGN(ldc_offset + sizeof(StrideC) * problem_count);
    this->weight_scale_offset = CUDA_MEM_ALIGN(ldd_offset + sizeof(StrideD) * problem_count);
    this->input_scale_offset =
        CUDA_MEM_ALIGN(weight_scale_offset + sizeof(float *) * problem_count);
    this->output_vec_scale_offset =
        CUDA_MEM_ALIGN(input_scale_offset + sizeof(float *) * input_groups);
    this->interDs_offset =
        CUDA_MEM_ALIGN(output_vec_scale_offset + sizeof(float *) * input_groups);
    this->total_bytes = CUDA_MEM_ALIGN(interDs_offset + sizeof(void *) * input_groups);
    return this->total_bytes;
  }

  void
  init_buffer(torch::Tensor weight, size_t buffer_size, int max_m, int n_dim) {
    std::vector<void *> output_scatter_ptrs(world_size, nullptr);
    int64_t bsize = static_cast<int64_t>(buffer_size);
    auto output_type = weight.scalar_type();
    if (weight.scalar_type() == at::ScalarType::Float8_e5m2 ||
        weight.scalar_type() == at::ScalarType::Float8_e4m3fn) {
      // hard code: fp8 gemm will use bf16 to save the outputs
      output_type = at::ScalarType::BFloat16;
    }
    CHECK(!this->host_buffer.defined());
    this->host_buffer = torch::empty(
        {bsize}, weight.options().dtype(at::ScalarType::Byte).device(torch::DeviceType::CPU));

    CHECK(!this->device_buffer.defined());
    this->device_buffer = torch::empty({bsize}, weight.options().dtype(at::ScalarType::Byte));
    CHECK(!this->splits_gpu.defined());
    this->splits_gpu =
        torch::empty({this->num_experts + 1}, weight.options().dtype(at::ScalarType::Int));
    CHECK(!this->output_buffer.defined());
    int output_buffer_m = max_m / this->topk;
    /*
    // shut down the fuse-reduction. may lead to the precison problem
    // leave the comments here if want to enable fuse-reduction in the future
    if (this->ep_world_size > 1) {
      // expert parallel will use gloab_red to perform the reduction
      // which need to zero the buffer at the begin, therefore, when expert
      // parallel is enabled, we allocate a buffer as small as possible
      output_buffer_m = output_buffer_m / this->world_size;
    }
    */
    this->output_buffer = nvshmem_create_tensor({output_buffer_m, n_dim}, output_type);
    this->output_buffer.zero_();
    CHECK(this->inter_Ds.empty());
    for (int i = 0; i < this->max_input_groups; i++) {
      this->inter_Ds.emplace_back(
          torch::zeros({max_m, n_dim}, weight.options().dtype(output_type)));
    }

    for (int i = 0; i < this->world_size; ++i) {
      if (i == this->rank) {
        output_scatter_ptrs[i] = output_buffer.data_ptr();
      } else {
        output_scatter_ptrs[i] = nvshmem_ptr(output_buffer.data_ptr(), i);
      }
    }
    const int output_ptr_buffer_bytes = sizeof(void *) * (this->world_size);
    CHECK(!ptr_buffer.defined());
    this->ptr_buffer =
        torch::empty({output_ptr_buffer_bytes}, weight.options().dtype(at::ScalarType::Byte));
    CUDA_CHECK(cudaMemcpy(
        this->ptr_buffer.data_ptr(),
        output_scatter_ptrs.data(),
        output_ptr_buffer_bytes,
        cudaMemcpyHostToDevice));
    CHECK(!this->barrier.defined());
    this->barrier = torch::zeros({SPLITS + 1}, weight.options().dtype(at::ScalarType::Int));
    if (this->ep_world_size > 1) {
      CHECK(!this->ep_indexes.defined());
      this->ep_indexes = torch::zeros(
          {(this->topk + 2) * (max_m / this->topk)}, weight.options().dtype(at::ScalarType::Int));
      CHECK(!this->ep_indexes_count.defined());
      this->ep_indexes_count = torch::zeros({1}, weight.options().dtype(at::ScalarType::Int));
    }
  }

  void
  init_workspace(torch::Tensor weight, int64_t workspace_size) {
    if (workspace_size <= 0)
      return;
    workspace_size = (workspace_size + 127) / 128 * 128;
    if (!this->workspace.defined() || workspace_size > this->workspace.numel()) {
      this->workspace =
          torch::empty({workspace_size}, weight.options().dtype(at::ScalarType::Byte));
    }
  }

  GemmGroupedV3GatherRSArguments
  prepare_args(
      std::vector<torch::Tensor> inputs,
      std::vector<torch::Tensor> weights,
      torch::Tensor splits_cpu,
      torch::Tensor routing_idx,
      c10::optional<std::vector<torch::Tensor>> input_scale,
      c10::optional<std::vector<torch::Tensor>> weight_scale,
      c10::optional<std::vector<torch::Tensor>> output_vec_scale,
      int sm_margin,
      bool with_stream_sync) {
    CHECK(splits_cpu.device() == torch::DeviceType::CPU);
    CHECK(splits_cpu.scalar_type() == at::ScalarType::Int);
    int64_t M = inputs[0].size(0);
    int64_t K = inputs[0].size(1);
    int64_t E = weights[0].size(0);
    int64_t N = weights[0].size(1);
    int64_t num_input_sets = inputs.size();
    CHECK(num_input_sets <= this->max_input_groups) << "input_groups:" << num_input_sets;
    int64_t problem_count = this->num_experts * SPLITS * num_input_sets;
    int64_t tp_rank = this->rank % this->tp_world_size;
    int64_t ep_rank = this->rank / this->tp_world_size;
    const int eid_start = E * ep_rank;
    const int eid_end = eid_start + E;
    int64_t globalM = routing_idx.size(0);
    std::vector<UnderlyingProblemShape> problem_sizes_vec(problem_count);
    std::vector<const void *> ptr_A_vec(problem_count, nullptr);
    std::vector<const void *> ptr_B_vec(problem_count, nullptr);
    std::vector<const void *> ptr_C_vec(problem_count, nullptr);
    std::vector<void *> ptr_D_vec(problem_count, nullptr);  // useless
    std::vector<StrideA> lda_vec(problem_count);
    std::vector<StrideB_Col> ldb_vec(problem_count);
    std::vector<StrideC> ldc_vec(problem_count);
    std::vector<StrideD> ldd_vec(problem_count);
    std::vector<float *> weight_scale_vec(problem_count);
    std::vector<float *> input_scale_vec(num_input_sets);
    std::vector<float *> output_vec_scale_vec(num_input_sets);
    std::vector<void *> interDs_ptr_vec(num_input_sets);

    // set the output type here accordlingly
    auto output_type = weights[0].scalar_type();
    if (weights[0].scalar_type() == at::ScalarType::Float8_e5m2 ||
        weights[0].scalar_type() == at::ScalarType::Float8_e4m3fn) {
      output_type = at::ScalarType::BFloat16;
    }
    auto element_size = c10::elementSize(inputs[0].scalar_type());
    auto output_element_size = c10::elementSize(output_type);
    for (int i = 0; i < this->total_num_experts; i++) {
      this->splits_cum_sum[i + 1] =
          this->splits_cum_sum[i] + reinterpret_cast<int32_t *>(splits_cpu.data_ptr())[i];
    }
    if (!this->drop_token) {
      CHECK(this->splits_cum_sum[this->total_num_experts] == routing_idx.size(0))
          << "The sum of split cpu is not aligned with routing_idx.size(0) \n";
    } else {
      int dropped_token_n =
          reinterpret_cast<int32_t *>(splits_cpu.data_ptr())[this->total_num_experts];
      CHECK(this->splits_cum_sum[this->total_num_experts] + dropped_token_n == routing_idx.size(0))
          << "The sum of split cpu is not aligned with routing_idx.size(0) \n";
    }
    const int64_t new_N = N / SPLITS;
    CHECK(N % SPLITS == 0);
    for (int sid = 0; sid < SPLITS; sid++) {
      for (int gid = 0; gid < num_input_sets; gid++) {
        int64_t M_acc = 0;
        for (int eid = 0; eid < E; eid++) {
          int64_t Mi;
          Mi = reinterpret_cast<int32_t *>(splits_cpu.data_ptr())[eid + eid_start];
          int problem_idx = sid * E * num_input_sets + gid * E + eid;
          problem_sizes_vec[problem_idx] = UnderlyingProblemShape({new_N, Mi, K});
          const void *ptr_Ai =
              reinterpret_cast<uint8_t *>(inputs[gid].data_ptr()) + M_acc * K * element_size;
          ptr_A_vec[problem_idx] = ptr_Ai;
          const void *ptr_Bi = reinterpret_cast<uint8_t *>(weights[gid].data_ptr()) +
                               (eid * N + sid * new_N) * K * element_size;
          ptr_B_vec[problem_idx] = ptr_Bi;
          ptr_C_vec[problem_idx] = nullptr;  // no bias here
          void *ptr_Di = reinterpret_cast<uint8_t *>(this->inter_Ds[gid].data_ptr()) +
                         M_acc * N * output_element_size + sid * new_N * output_element_size;
          ptr_D_vec[problem_idx] = ptr_Di;
          // swap A B here
          // {K, cute::Int<1>{}, 0};
          lda_vec[problem_idx] = cutlass::make_cute_packed_stride(
              StrideB_Col{}, cute::make_shape(static_cast<int>(Mi), static_cast<int>(K), 1));
          ldb_vec[problem_idx] = cutlass::make_cute_packed_stride(
              StrideA{}, cute::make_shape(static_cast<int>(new_N), static_cast<int>(K), 1));
          ldc_vec[problem_idx] = {cute::Int<1>{}, N, cute::Int<0>{}};
          ldd_vec[problem_idx] = {cute::Int<1>{}, N, cute::Int<0>{}};
          if (weight_scale.has_value()) {
            torch::Tensor cur_weight_scale = weight_scale.value()[gid];
            CHECK_1D(cur_weight_scale, E);
            CHECK(cur_weight_scale.scalar_type() == at::ScalarType::Float);
            weight_scale_vec[problem_idx] =
                reinterpret_cast<float *>(cur_weight_scale.data_ptr()) + eid;
          } else {
            weight_scale_vec[problem_idx] = nullptr;
          }
          M_acc += Mi;
        }
        CHECK(M_acc == M) << "M_acc" << M_acc
                          << "!= input.size(0):" << M;  // check whether the splits_cpu is legal
      }
    }

    uint8_t *host_buffer_ptr = reinterpret_cast<uint8_t *>(host_buffer.data_ptr());
    std::memcpy(
        host_buffer_ptr + problem_sizes_offset,
        problem_sizes_vec.data(),
        sizeof(UnderlyingProblemShape) * problem_count);
    std::memcpy(host_buffer_ptr + ptr_a_offset, ptr_A_vec.data(), sizeof(void *) * problem_count);
    std::memcpy(host_buffer_ptr + ptr_b_offset, ptr_B_vec.data(), sizeof(void *) * problem_count);
    std::memcpy(host_buffer_ptr + ptr_c_offset, ptr_C_vec.data(), sizeof(void *) * problem_count);
    std::memcpy(host_buffer_ptr + ptr_d_offset, ptr_D_vec.data(), sizeof(void *) * problem_count);
    std::memcpy(host_buffer_ptr + lda_offset, lda_vec.data(), sizeof(StrideA) * problem_count);
    std::memcpy(host_buffer_ptr + ldb_offset, ldb_vec.data(), sizeof(StrideB_Col) * problem_count);
    std::memcpy(host_buffer_ptr + ldc_offset, ldc_vec.data(), sizeof(StrideC) * problem_count);
    std::memcpy(host_buffer_ptr + ldd_offset, ldd_vec.data(), sizeof(StrideD) * problem_count);
    std::memcpy(
        host_buffer_ptr + weight_scale_offset,
        weight_scale_vec.data(),
        sizeof(float *) * problem_count);

    if (input_scale.has_value()) {
      auto input_scale_vals = input_scale.value();
      CHECK(num_input_sets == input_scale_vals.size());
      for (int gid = 0; gid < input_scale_vals.size(); gid++) {
        CHECK(input_scale_vals[gid].scalar_type() == at::ScalarType::Float);
        input_scale_vec[gid] = reinterpret_cast<float *>(input_scale_vals[gid].data_ptr());
      }
      std::memcpy(
          host_buffer_ptr + input_scale_offset,
          input_scale_vec.data(),
          sizeof(float *) * input_scale_vals.size());
    }
    if (output_vec_scale.has_value()) {
      auto output_vec_scale_val = output_vec_scale.value();
      CHECK(num_input_sets == output_vec_scale_val.size());
      for (int gid = 0; gid < output_vec_scale_val.size(); gid++) {
        CHECK_1D(output_vec_scale_val[gid], M);
        CHECK(output_vec_scale_val[gid].scalar_type() == at::ScalarType::Float);
        output_vec_scale_vec[gid] =
            reinterpret_cast<float *>(output_vec_scale_val[gid].data_ptr());
      }
      std::memcpy(
          host_buffer_ptr + output_vec_scale_offset,
          output_vec_scale_vec.data(),
          sizeof(float *) * output_vec_scale_val.size());
    }
    for (int gid = 0; gid < num_input_sets; gid++) {
      interDs_ptr_vec[gid] = this->inter_Ds[gid].data_ptr();
    }
    std::memcpy(
        host_buffer_ptr + interDs_offset,
        interDs_ptr_vec.data(),
        sizeof(void *) * interDs_ptr_vec.size());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    CUDA_CHECK(cudaMemcpyAsync(
        device_buffer.data_ptr(),
        host_buffer.data_ptr(),
        this->total_bytes,
        cudaMemcpyHostToDevice,
        stream));
    CUDA_CHECK(cudaMemcpyAsync(
        splits_gpu.data_ptr(),
        splits_cpu.data_ptr(),
        sizeof(int32_t) * this->num_experts,
        cudaMemcpyHostToDevice,
        stream));
    uint8_t *device_buffer_ptr = reinterpret_cast<uint8_t *>(device_buffer.data_ptr());
    void *problem_sizes_device = device_buffer_ptr + problem_sizes_offset;
    void **ptr_A_device = reinterpret_cast<void **>(device_buffer_ptr + ptr_a_offset);
    void **ptr_B_device = reinterpret_cast<void **>(device_buffer_ptr + ptr_b_offset);
    void **ptr_C_device = reinterpret_cast<void **>(device_buffer_ptr + ptr_c_offset);
    void **ptr_D_device = reinterpret_cast<void **>(device_buffer_ptr + ptr_d_offset);
    void **ptr_output_scatter = reinterpret_cast<void **>(this->ptr_buffer.data_ptr());
    void *lda_device = reinterpret_cast<void *>(device_buffer_ptr + lda_offset);
    void *ldb_device = reinterpret_cast<void *>(device_buffer_ptr + ldb_offset);
    void *ldc_device = reinterpret_cast<void *>(device_buffer_ptr + ldc_offset);
    void *ldd_device = reinterpret_cast<void *>(device_buffer_ptr + ldd_offset);
    void **interDs_device = reinterpret_cast<void **>(device_buffer_ptr + interDs_offset);
    float **weight_scale_device = nullptr;
    if (weight_scale.has_value()) {
      weight_scale_device = reinterpret_cast<float **>(device_buffer_ptr + weight_scale_offset);
    }
    float **input_scale_device = nullptr;
    if (input_scale.has_value()) {
      input_scale_device = reinterpret_cast<float **>(device_buffer_ptr + input_scale_offset);
    }
    float **output_vec_scale_device = nullptr;
    if (output_vec_scale.has_value()) {
      output_vec_scale_device =
          reinterpret_cast<float **>(device_buffer_ptr + output_vec_scale_offset);
    }

    float alpha = 1.0, beta = 0.0;
    void *problem_sizes_host = reinterpret_cast<void *>(host_buffer_ptr + problem_sizes_offset);
    int32_t *ptr_routing_idx = reinterpret_cast<int32_t *>(routing_idx.data_ptr());
    const int ep_offset_start = this->splits_cum_sum[eid_start];
    const int ep_offset_end = this->splits_cum_sum[eid_end];
    int *pos_filtered = nullptr;
    int *token_idx_filtered = nullptr;  // length = (globalM/topk)
    int *total_token_acc = nullptr;
    if (this->ep_world_size > 1) {
      // calculate the indexes that routed to current expert rank
      this->ep_indexes_count.zero_();
      int *index_start = reinterpret_cast<int32_t *>(this->ep_indexes.data_ptr());
      pos_filtered = index_start + globalM / this->topk;  // length = (globalM/topk) * (topk+1)
      token_idx_filtered = index_start;                   // length = (globalM/topk)
      total_token_acc = reinterpret_cast<int32_t *>(this->ep_indexes_count.data_ptr());
      ep_index_filter_impl(
          reinterpret_cast<int32_t *>(routing_idx.data_ptr()),
          pos_filtered,
          token_idx_filtered,
          total_token_acc,
          globalM,
          this->topk,
          ep_offset_start,
          ep_offset_end,
          stream);
    }
    if (with_stream_sync) {
      // the host buffer is shared across all layers in the model.
      // if the gather-rs is called without ag-scatter, please set
      // with stream sync to true.
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    GemmGroupedV3GatherRSArguments args{
        problem_sizes_device,
        problem_count,
        alpha,
        beta,
        const_cast<const void **>(ptr_B_device),
        const_cast<const void **>(ptr_A_device),
        const_cast<const void **>(ptr_C_device),
        ptr_D_device,
        ldb_device,
        lda_device,
        ldc_device,
        ldd_device,
        problem_sizes_host,
        rank,
        world_size,
        ptr_output_scatter,
        interDs_device,
        this->topk,
        reinterpret_cast<int32_t *>(this->barrier.data_ptr()),
        ptr_routing_idx,
        SPLITS,
        M,
        this->n_dim,
        this->tp_world_size,
        this->ep_world_size,
        globalM,
        ep_offset_start,
        ep_offset_end,
        input_scale_device,
        weight_scale_device,
        output_vec_scale_device,
        sm_margin,
        num_input_sets,
        pos_filtered,
        token_idx_filtered,
        total_token_acc};

    return args;
  }

 public:
  GemmGroupedV3GatherRSOpImpl(
      int64_t total_num_experts,
      int64_t max_m,
      int64_t n_dim,
      int64_t topk,
      int64_t rank,
      int64_t world_size,
      int64_t tp_world_size,
      int64_t ep_world_size,
      int64_t max_input_groups)
      : total_num_experts(total_num_experts),
        max_m(max_m),
        n_dim(n_dim),
        topk(topk),
        rank(rank),
        world_size(world_size),
        buffer_initialized(false),
        tp_world_size(tp_world_size),
        ep_world_size(ep_world_size),
        max_input_groups(max_input_groups) {
    CHECK(this->tp_world_size * this->ep_world_size == this->world_size)
        << "Tp world size x Ep world size != World size";
    CHECK(this->total_num_experts % this->ep_world_size == 0)
        << "The number of experts is not divisible by the EP world size";
    // TODO(ZSL): the SPLITS should be tuned.
    this->SPLITS = 8;
    this->num_experts = this->total_num_experts / this->ep_world_size;
    this->gather_rs_stream.push_back(at::cuda::getStreamFromPool());
    this->splits_cum_sum.resize(total_num_experts + 1, 0);
    this->splits_cum_sum[0] = 0;
    CUDA_CHECK(cudaEventCreateWithFlags(&this->ready_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&this->gather_rs_start_event, cudaEventDisableTiming));
  }

  void
  ep_zero_buffer() {
    assert(this->ep_world_size > 1);
    if (this->output_buffer.defined())
      this->output_buffer.zero_();
  }

  torch::Tensor
  forward_gather_rs_impl(
      std::vector<torch::Tensor> inputs,
      std::vector<torch::Tensor> weights,
      torch::Tensor splits_cpu,
      torch::Tensor routing_idx,
      c10::optional<std::vector<torch::Tensor>> input_scale,
      c10::optional<std::vector<torch::Tensor>> weight_scale,
      c10::optional<std::vector<torch::Tensor>> output_vec_scale,
      bool fast_accum,
      int sm_margin,
      bool with_stream_sync,
      bool with_zero_buffer,
      c10::optional<UnifiedGemmHParams> const &hparams) {
    /*
      Note: When expert parallel is enabled, the inputs/weights tensor should be
      the partial the current expert parallel rank. But the splits_cpu and routing
      idx should be global no matter whether expert parallel is enabled, which means the
      splits_cpu/routing_idx should contains all the experts / tokens no matter whether expert
      parallel is enabled.
    */
    CHECK(inputs.size() == weights.size());
    CHECK(inputs.size() <= this->max_input_groups);
    if (!this->buffer_initialized) {
      this->_st = weights[0].scalar_type();
      size_t args_buffer_size = args_offset_and_buffer_size(
          num_experts * SPLITS * this->max_input_groups, this->max_input_groups);
      this->init_buffer(
          weights[0],
          args_buffer_size,
          this->max_m,
          this->n_dim);  // allocate host and device buffers
      this->buffer_initialized = true;
    }
    int globalM = routing_idx.size(0);
    int M = inputs[0].size(0);
    int K = inputs[0].size(1);
    int E = weights[0].size(0);
    int N = weights[0].size(1);
    for (int i = 0; i < inputs.size(); i++) {
      CHECK_3D(weights[i], this->num_experts, this->n_dim, K);
      CHECK_INPUT(weights[i], this->_st);
      CHECK_INPUT(inputs[i], this->_st);
      CHECK_NDIM(inputs[i], 2);
      CHECK_2D(inputs[i], M, K);
    }
    CHECK(splits_cpu.scalar_type() == at::ScalarType::Int);
    CHECK(routing_idx.scalar_type() == at::ScalarType::Int);
    CHECK_NDIM(splits_cpu, 1);

    if (this->ep_world_size == 1) {
      CHECK(M == globalM);
    }
    CHECK(globalM % this->world_size == 0);
    CHECK(globalM % this->topk == 0);
    CHECK(globalM <= this->max_m) << "routing_idx.size(0) " << globalM << " larger than max_m"
                                  << this->max_m
                                  << ", Please set env JANUS_FLUX_M_MAX appropriately\n";
    CHECK(M <= globalM) << "input.size(0) " << M << " larger than routing_idx.size(0)\n";
    CHECK(N == this->n_dim) << "weight.size(1) != n_dim";
    this->drop_token = splits_cpu.size(0) == this->total_num_experts + 1;
    if (!this->drop_token) {
      CHECK(splits_cpu.size(0) == this->total_num_experts)
          << "splits 0-dim mismatch: " << splits_cpu.size(0) << " != " << this->total_num_experts;
    } else {
      assert(this->ep_world_size > 1);  // currently only ep support drop token
    }
    CHECK(N >= 8) << "N must be greater than or equal 8 for cutlass grouped gemm.";
    CHECK(K >= 8) << "K must be greater than or equal 8 for cutlass grouped gemm.";
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    if (this->ep_world_size > 1) {
      if (with_zero_buffer) {
        this->ep_zero_buffer();
      }
      nvshmemx_barrier_all_on_stream(stream);
    }
    if (M > 0) {
      GemmGroupedV3GatherRSArguments args = prepare_args(
          inputs,
          weights,
          splits_cpu,
          routing_idx,
          input_scale,
          weight_scale,
          output_vec_scale,
          sm_margin,
          with_stream_sync);
      ArchEnum arch = get_arch();

      auto dtype = from_torch_dtype(this->_st);
      DataTypeEnum output_type = dtype;
      int gather_rs_ctas = 28;  // can be tuned in the future
      bool is_fp8 = (dtype == _E4M3{}) || (dtype == _E5M2{});
      if (is_fp8) {
        // if the dtype of input is fp8, use bfloat16 as the output dtype
        output_type = _BF16{};
        gather_rs_ctas = 32;
      }
      auto dt_conf =
          to_gemm_dtype_config(make_gemm_dtype_config(dtype, dtype, output_type, output_type));
      auto impl_spec = make_gemm_v3_meta(fast_accum and dt_conf.is_input_fp8());
      auto comm_spec = make_gather_rs_meta(this->topk);
      auto meta = make_gemm_meta(
          dt_conf, arch, _GatherRS{}, _RCC{}, _GemmGroupedV3{}(), impl_spec, comm_spec);
      auto rt_conf = make_runtime_config(N, cute::ceil_div(globalM, this->num_experts), K);

      auto hparams_filter = [&](UnifiedGemmHParams const &hparams) {
        auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));
        if (dt_conf.is_input_fp8()) {
          auto comm_params = std::get<unified_type_t<GatherRSHParams>>(hparams.comm_spec());
          bool flag = comm_params.gather_rs_ctas() == gather_rs_ctas &&
                      comm_params.n_dim() == N;
          return flag;
        } else {
          // bf16 filter based on the gather_rs_ctas && topk
          // TODO(ZSL): cleanup the code 
          auto comm_params = std::get<unified_type_t<GatherRSHParams>>(hparams.comm_spec());
          return comm_params.n_dim() == N;
        }
      };

      OpRegistry::OpPtr gemm_op;
      if (hparams.has_value()) {
        gemm_op = OpRegistry::instance().get_op(meta, hparams.value());
        auto& comm_params = std::get<unified_type_t<GatherRSHParams>>(hparams->comm_spec());
        gather_rs_ctas = comm_params.gather_rs_ctas();
      } else {
        gemm_op = OpRegistry::instance().get_op(meta, rt_conf, hparams_filter);
      }
      int64_t workspace_size = gemm_op->get_workspace_size(args);

      this->init_workspace(weights[0], workspace_size);
      void *workspace_ptr = this->workspace.defined() ? this->workspace.data_ptr() : nullptr;
#ifdef MOE_GATHER_RS_SEPARATE_IMPL
      CUDA_CHECK(cudaEventRecord(this->gather_rs_start_event, stream));
#endif
      int tp_rank = this->rank % this->tp_world_size;
      int ep_rank = this->rank / this->tp_world_size;
      gemm_op->run(args, workspace_ptr, stream);
#ifdef MOE_GATHER_RS_SEPARATE_IMPL
      {
        TopKReduceGatherRSArguments topk_gather_rs_args{
            args.rank,
            args.world_size,
            args.output_scatter_ptrs,
            args.inter_Ds,
            args.topk,
            args.barrier,
            args.routing_idx,
            args.SPLITS,
            args.totalM,
            args.n_dim,
            gather_rs_ctas,
            this->tp_world_size,
            this->ep_world_size,
            globalM,
            input_scale_ptr,
            output_vec_scale_ptr};
        // TODO the number of ctas for gather-rs should be put into the
        // hparams
        int32_t *gather_rs_wait_barrier = &args.barrier[args.SPLITS];
        // perform the gather_rs on another stream;
        at::cuda::CUDAStreamGuard guard(this->gather_rs_stream[0]);
        cudaStream_t rs_stream = this->gather_rs_stream[0];
        CUDA_CHECK(cudaStreamWaitEvent(rs_stream, this->gather_rs_start_event));
        CU_CHECK(CUStreamWaitValue(
            rs_stream, (CUdeviceptr)(gather_rs_wait_barrier), 1, CU_STREAM_WAIT_VALUE_EQ));
        if (this->ep_world_size == 1) {
          // pure tp parallel
          topk_reduce_gather_rs(topk_gather_rs_args, this->inter_D);
        } else {
          const int eid_start = ep_rank * this->num_experts;
          const int eid_end = (ep_rank + 1) * this->num_experts;
          const int ep_offset_start = this->splits_cum_sum[eid_start];
          const int ep_offset_end = this->splits_cum_sum[eid_end];

          ep_topk_reduce_gather_rs(
              topk_gather_rs_args, this->inter_D, ep_offset_start, ep_offset_end);
        }
        CUDA_CHECK(cudaEventRecord(this->ready_event, rs_stream));
      }
      // wait the data to be ready
      CUDA_CHECK(cudaStreamWaitEvent(stream, this->ready_event));
#endif
    }
    nvshmemx_barrier_all_on_stream(stream);
    this->barrier.zero_();
    if (this->ep_world_size == 1) {
      return this->output_buffer.slice(0, 0, M / this->topk)
          .view({this->world_size, M / this->world_size / this->topk, N})
          .sum(torch::IntArrayRef({0}));
    } else {
      // expert parallel size is not 1
      // the reduction between different experts are performed
      // by global_red, therefore no need to perform the reduction
      // any more, in constrast, it will also introduce a zero_ operation
      // into the critical path.
      return this->output_buffer.slice(0, 0, globalM / this->topk)
          .view({this->world_size, globalM / this->world_size / this->topk, N})
          .sum(torch::IntArrayRef({0}));
    }
  }

  torch::Tensor
  forward_gather_rs(
      torch::Tensor input,
      torch::Tensor weight,
      torch::Tensor splits_cpu,
      torch::Tensor routing_idx,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_vec_scale,
      bool fast_accum,
      int sm_margin,
      bool with_stream_sync) {
    c10::optional<std::vector<torch::Tensor>> input_scale_wrap;
    c10::optional<std::vector<torch::Tensor>> weight_scale_wrap;
    c10::optional<std::vector<torch::Tensor>> output_vec_scale_wrap;
    if (input_scale.has_value()) {
      input_scale_wrap = {input_scale.value()};
    }
    if (weight_scale.has_value()) {
      weight_scale_wrap = {weight_scale.value()};
    }
    if (output_vec_scale.has_value()) {
      output_vec_scale_wrap = {output_vec_scale.value()};
    }
    return forward_gather_rs_impl(
        {input},
        {weight},
        std::move(splits_cpu),
        std::move(routing_idx),
        std::move(input_scale_wrap),
        std::move(weight_scale_wrap),
        std::move(output_vec_scale_wrap),
        fast_accum,
        sm_margin,
        with_stream_sync,
        true,
        c10::nullopt);
  }

  torch::Tensor
  forward_gather_rs_no_zerobuffer(
      torch::Tensor input,
      torch::Tensor weight,
      torch::Tensor splits_cpu,
      torch::Tensor routing_idx,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_vec_scale,
      bool fast_accum,
      int sm_margin,
      bool with_stream_sync) {
    c10::optional<std::vector<torch::Tensor>> input_scale_wrap;
    c10::optional<std::vector<torch::Tensor>> weight_scale_wrap;
    c10::optional<std::vector<torch::Tensor>> output_vec_scale_wrap;
    if (input_scale.has_value()) {
      input_scale_wrap = {input_scale.value()};
    }
    if (weight_scale.has_value()) {
      weight_scale_wrap = {weight_scale.value()};
    }
    if (output_vec_scale.has_value()) {
      output_vec_scale_wrap = {output_vec_scale.value()};
    }
    return forward_gather_rs_impl(
        {input},
        {weight},
        std::move(splits_cpu),
        std::move(routing_idx),
        std::move(input_scale_wrap),
        std::move(weight_scale_wrap),
        std::move(output_vec_scale_wrap),
        fast_accum,
        sm_margin,
        with_stream_sync,
        false,
        c10::nullopt);
  }
  torch::Tensor
  profiling(
      torch::Tensor input,
      torch::Tensor weight,
      torch::Tensor splits_cpu,
      torch::Tensor routing_idx,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_vec_scale,
      bool fast_accum,
      int sm_margin,
      bool with_stream_sync,
      c10::intrusive_ptr<ProfilingContext> opt_ctx) {
    c10::optional<std::vector<torch::Tensor>> input_scale_wrap;
    c10::optional<std::vector<torch::Tensor>> weight_scale_wrap;
    c10::optional<std::vector<torch::Tensor>> output_vec_scale_wrap;
    if (input_scale.has_value()) {
      input_scale_wrap = {input_scale.value()};
    }
    if (weight_scale.has_value()) {
      weight_scale_wrap = {weight_scale.value()};
    }
    if (output_vec_scale.has_value()) {
      output_vec_scale_wrap = {output_vec_scale.value()};
    }

    int globalM = routing_idx.size(0);
    int K = input.size(1);
    int E = weight.size(0);
    int N = weight.size(1);
    ArchEnum arch = get_arch();
    auto weight_dtype = weight.scalar_type();
    auto dtype = from_torch_dtype(weight_dtype);
    DataTypeEnum output_type = dtype;
    bool is_fp8 = (dtype == _E4M3{}) || (dtype == _E5M2{});
    if (is_fp8) {
      // if the dtype of input is fp8, use bfloat16 as the output dtype
      output_type = _BF16{};
    }
    auto dt_conf =
        to_gemm_dtype_config(make_gemm_dtype_config(dtype, dtype, output_type, output_type));
    auto impl_spec = make_gemm_v3_meta(fast_accum and dt_conf.is_input_fp8());
    auto comm_spec = make_gather_rs_meta(this->topk);
    auto meta = unify_type(make_gemm_meta(
        dt_conf, arch, _GatherRS{}, _RCC{}, _GemmGroupedV3{}(), impl_spec, comm_spec));
    auto rt_conf = make_runtime_config(N, cute::ceil_div(globalM, this->num_experts), K);
    ProfilingContext tmp_ctx("__tmp__");
    ProfilingContext *ctx = opt_ctx == nullptr ? &tmp_ctx : opt_ctx.get();

    OpRegistry::instance().visit_hparams(
        [&](UnifiedGemmHParams const &hparams) {
          constexpr int warm_iters = 5;
          constexpr int iters = 10;
          float total_elapsed = 0;
          auto cp_hparams = hparams;
          auto comm_params = std::get<unified_type_t<GatherRSHParams>>(cp_hparams.comm_spec());
          if (comm_params.n_dim() != N) {
            return;
          }
          auto stream = c10::cuda::getCurrentCUDAStream();
          for (int iter = 0; iter < warm_iters + iters; ++iter) {
            GpuTimer timer;
            timer.start(stream);
            auto output [[maybe_unused]] = this->forward_gather_rs_impl(
                {input},
                {weight},
                splits_cpu,
                routing_idx,
                input_scale_wrap,
                weight_scale_wrap,
                output_vec_scale_wrap,
                fast_accum,
                sm_margin,
                false,  // whether with stream sync
                true,   // whether with zero buffer
                cp_hparams);
            timer.stop();
            if (iter >= warm_iters) {
              total_elapsed += timer.elapsed_millis();
            }
          }

          float avg_elapsed = int(total_elapsed / iters * 1000) / 1000.0;
          ctx->add(meta, rt_conf, hparams, avg_elapsed);
        },
        meta);

    auto best_hparams = ctx->record_best(meta, rt_conf);

    return this->forward_gather_rs_impl(
        {input},
        {weight},
        splits_cpu,
        routing_idx,
        input_scale_wrap,
        weight_scale_wrap,
        output_vec_scale_wrap,
        fast_accum,
        sm_margin,
        false,  // whether with stream sync
        true,
        std::move(best_hparams));
  }

  torch::Tensor
  forward_gather_rs_multiple(
      std::vector<torch::Tensor> inputs,
      std::vector<torch::Tensor> weights,
      torch::Tensor splits_cpu,
      torch::Tensor routing_idx,
      c10::optional<std::vector<torch::Tensor>> input_scale,
      c10::optional<std::vector<torch::Tensor>> weight_scale,
      c10::optional<std::vector<torch::Tensor>> output_vec_scale,
      bool fast_accum,
      int sm_margin,
      bool with_stream_sync) {
    // all the inputs && weights share the same splits_cpu and routing_idx
    CHECK(inputs.size() == weights.size());
    return forward_gather_rs_impl(
        std::move(inputs),
        std::move(weights),
        std::move(splits_cpu),
        std::move(routing_idx),
        std::move(input_scale),
        std::move(weight_scale),
        std::move(output_vec_scale),
        fast_accum,
        sm_margin,
        with_stream_sync,
        true,
        c10::nullopt);
  }
  std::tuple<int64_t, int64_t, int64_t>
  get_pickle_info() const {
    return std::make_tuple(this->max_m, this->n_dim, this->num_experts);
  }
};

GemmGroupedV3GatherRS::GemmGroupedV3GatherRS(
    int64_t total_num_experts,
    int64_t max_m,
    int64_t n_dim,
    int64_t topk,
    int64_t rank,
    int64_t world_size,
    int64_t tp_world_size,
    int64_t ep_world_size,
    int64_t max_input_groups)
    : impl_(new GemmGroupedV3GatherRSOpImpl(
          total_num_experts,
          max_m,
          n_dim,
          topk,
          rank,
          world_size,
          tp_world_size,
          ep_world_size,
          max_input_groups)) {}
GemmGroupedV3GatherRS::~GemmGroupedV3GatherRS() { delete impl_; }
torch::Tensor
GemmGroupedV3GatherRS::forward_gather_rs(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor splits_cpu,
    torch::Tensor routing_idx,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale,
    c10::optional<torch::Tensor> output_vec_scale,
    bool fast_accum,
    int sm_margin,
    bool with_stream_sync) {
  FLUX_CHECK(impl_ != nullptr) << "GemmGroupedV3GatherRS is not initialized";
  return impl_->forward_gather_rs(
      input,
      weight,
      splits_cpu,
      routing_idx,
      input_scale,
      weight_scale,
      output_vec_scale,
      fast_accum,
      sm_margin,
      with_stream_sync);
}
torch::Tensor
GemmGroupedV3GatherRS::forward_gather_rs_no_zerobuffer(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor splits_cpu,
    torch::Tensor routing_idx,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale,
    c10::optional<torch::Tensor> output_vec_scale,
    bool fast_accum,
    int sm_margin,
    bool with_stream_sync) {
  FLUX_CHECK(impl_ != nullptr) << "GemmGroupedV3GatherRS is not initialized";
  return impl_->forward_gather_rs_no_zerobuffer(
      input,
      weight,
      splits_cpu,
      routing_idx,
      input_scale,
      weight_scale,
      output_vec_scale,
      fast_accum,
      sm_margin,
      with_stream_sync);
}
void
GemmGroupedV3GatherRS::ep_zero_buffer() {
  FLUX_CHECK(impl_ != nullptr) << "GemmGroupedV3GatherRS is not initialized";
  return impl_->ep_zero_buffer();
}

torch::Tensor
GemmGroupedV3GatherRS::forward_gather_rs_multiple(
    std::vector<torch::Tensor> inputs,
    std::vector<torch::Tensor> weights,
    torch::Tensor splits_cpu,
    torch::Tensor routing_idx,
    c10::optional<std::vector<torch::Tensor>> input_scale,
    c10::optional<std::vector<torch::Tensor>> weight_scale,
    c10::optional<std::vector<torch::Tensor>> output_vec_scale,
    bool fast_accum,
    int sm_margin,
    bool with_stream_sync) {
  FLUX_CHECK(impl_ != nullptr) << "GemmGroupedV3GatherRS is not initialized";
  return impl_->forward_gather_rs_multiple(
      inputs,
      weights,
      splits_cpu,
      routing_idx,
      input_scale,
      weight_scale,
      output_vec_scale,
      fast_accum,
      sm_margin,
      with_stream_sync);
}
torch::Tensor
GemmGroupedV3GatherRS::profiling(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor splits_cpu,
    torch::Tensor routing_idx,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale,
    c10::optional<torch::Tensor> output_vec_scale,
    bool fast_accum,
    int sm_margin,
    bool with_stream_sync,
    c10::intrusive_ptr<ProfilingContext> opt_ctx) {
  FLUX_CHECK(impl_ != nullptr) << "GemmGroupedV3GatherRS is not initialized";
  return impl_->profiling(
      input,
      weight,
      splits_cpu,
      routing_idx,
      input_scale,
      weight_scale,
      output_vec_scale,
      fast_accum,
      sm_margin,
      with_stream_sync,
      opt_ctx);
}

}  // namespace ths_op
}  // namespace flux
}  // namespace bytedance
