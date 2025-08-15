//===- gemm_grouped_v2_gather_rs.cc -------------------------------------------- C++ ---===//
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

#include "moe_gather_rs/ths_op/gemm_grouped_v2_gather_rs.h"

#include <ATen/core/List.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/ivalue.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CachingHostAllocator.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include <cuda_runtime_api.h>
#include <cutlass/fast_math.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/layout/matrix.h>
#include <torch/all.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cutlass/util/packed_stride.hpp>
#include <iostream>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <utility>
#include <vector>

#include "flux/args/moe_gather_rs.h"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/cuda_stub.h"
#include "flux/flux.h"
#include "flux/gemm_meta.h"
#include "flux/op_registry.h"
#include "flux/ths_op/flux_shm.h"
#include "flux/ths_op/ths_op.h"
#include "flux/ths_op/util.h"
#include "flux/utils.h"
#include "moe_gather_rs/topk_gather_rs.hpp"
#include "moe_gather_rs/workspace_helper.h"
#if defined(FLUX_WITH_TRITON_AOT)
#include "moe_utils.h"
#include "triton_aot_generated/flux_triton_aot.h"
#endif

namespace {
// the copy tile size for TopkReduceScatterOp. has nothing to do with the GEMM tile_size_m
static constexpr int kTileSizeM = 128, kTileSizeN = 1024;
long
get_args_workspace_size(int problem_count) {
  using bytedance::flux::pad_to;
  constexpr int kAlignment = 128;
  // the workspace size
  int bytes =
      pad_to(sizeof(cutlass::gemm::GemmCoord) * problem_count, kAlignment) * 1  // problem_sizes
      + pad_to(sizeof(void *) * problem_count, kAlignment) * 4   // ptr_A/ptr_B/ptr_C/ptr_D
      + pad_to(sizeof(int64_t) * problem_count, kAlignment) * 5  // lda/ldb/ldc/ldd/ldr
      + pad_to(sizeof(float *) * problem_count, kAlignment) * 2  // scale_A/scale_B
      + pad_to(sizeof(int) * 1, kAlignment) * 1;                 // non_empty_problem_count
  return bytes;
}
c10::optional<std::vector<torch::Tensor>>
as_optional_vec(c10::optional<torch::Tensor> &t) {
  if (t.has_value()) {
    return c10::optional<std::vector<torch::Tensor>>{{t.value()}};
  }
  return {};
}

void *
data_ptr_or(c10::optional<torch::Tensor> &t, void *other) {
  return t.has_value() ? t->data_ptr() : other;
}
int
get_rs_threadblock_count() {
  static int rs_num_blocks = bytedance::flux::get_int_from_env("FLUX_RS_BLOCKS", 3);
  return rs_num_blocks;
}
}  // namespace

namespace bytedance::flux::ths_op {

using torch::Tensor;

class TopkReduceScatterOp::TopkReduceScatterOpImpl {
 private:
  std::shared_ptr<Group> tp_group;
  int32_t rank;
  int32_t world_size;  // the total world size
  int32_t max_m;
  int32_t n_dim;
  int32_t topk;
  at::ScalarType output_dtype;
  const int ep_nexperts;
  const int ep_world_size;  // the world size of expert parallel
  const bool do_all_reduce;
  const bool use_read_mode;
  const int n_split;

  torch::Tensor reduce_buffer;
  std::vector<torch::Tensor> reduce_buffers;
  torch::Tensor reduce_buffer_dptrs;
  torch::Tensor tile_barrier;
  std::vector<torch::Tensor> tile_barriers;
  torch::Tensor tile_barrier_dptrs;
  torch::Tensor barrier;
  std::vector<torch::Tensor> barriers;
  int **barrier_dev_ptrs = nullptr;

  bool buffer_initialized = false;

 private:
  void
  init_buffer_once(at::ScalarType dtype) {
    if (this->buffer_initialized)
      return;
    std::vector<void *> hptrs(this->world_size, nullptr);
    const int ptr_bytes = sizeof(void *) * this->world_size;
    // initialize the output buffer
    this->reduce_buffers = flux_create_tensor_list(
        {this->max_m / this->topk, this->n_dim}, dtype, this->tp_group.get());
    this->reduce_buffer = this->reduce_buffers[this->rank];
    for (int i = 0; i < this->world_size; ++i) {
      hptrs[i] = reduce_buffers[i].data_ptr();
    }
    CHECK(!reduce_buffer_dptrs.defined());
    this->reduce_buffer_dptrs =
        torch::empty({ptr_bytes}, at::TensorOptions(at::kCUDA).dtype(at::ScalarType::Byte));
    CUDA_CHECK(cudaMemcpy(
        this->reduce_buffer_dptrs.data_ptr(), hptrs.data(), ptr_bytes, cudaMemcpyHostToDevice));
    torch::cuda::synchronize();
    this->buffer_initialized = true;
  }
  int
  get_tile_barrier_size(int num_tiles) const {
    return num_tiles;
  }

  void
  create_rs_barrier() {
    int m_tiles_at_most = (this->max_m + kTileSizeM - 1) / kTileSizeM + this->ep_nexperts;
    int n_tiles = (this->n_dim + kTileSizeN - 1) / kTileSizeN;
    int num_tiles = m_tiles_at_most * n_tiles;

    int tile_barrier_size = get_tile_barrier_size(num_tiles);
    if (!this->tile_barrier.defined() || this->tile_barrier.numel() < tile_barrier_size) {
      // initialize the tile_barrier
      this->tile_barriers =
          flux_create_tensor_list({tile_barrier_size}, at::ScalarType::Int, this->tp_group.get());
      this->tile_barrier = this->tile_barriers[this->rank];
      std::vector<int *> hptrs(world_size, nullptr);
      const int ptr_bytes = sizeof(int *) * this->world_size;
      for (int i = 0; i < this->world_size; ++i) {
        hptrs[i] = (int *)this->tile_barriers[i].data_ptr();
      }
      CHECK(!tile_barrier_dptrs.defined());
      this->tile_barrier_dptrs =
          torch::empty({ptr_bytes}, at::TensorOptions(at::kCUDA).dtype(at::ScalarType::Byte));
      CUDA_CHECK(cudaMemcpy(
          this->tile_barrier_dptrs.data_ptr(), hptrs.data(), ptr_bytes, cudaMemcpyHostToDevice));
    }
  }

 public:
  TopkReduceScatterOpImpl(
      std::shared_ptr<Group> tp_group_,
      int max_m,
      int n_dim,
      int topk,
      at::ScalarType output_dtype,
      int ep_nexperts,
      int ep_world_size,
      const std::vector<torch::Tensor> &barriers,
      int n_split_,
      bool do_all_reduce_ = false,
      bool use_read_mode_ = false)
      : tp_group(tp_group_),
        rank(tp_group_->get_rank()),
        world_size(tp_group_->get_size()),
        max_m(max_m),
        n_dim(n_dim),
        topk(topk),
        output_dtype(output_dtype),
        ep_nexperts(ep_nexperts),
        ep_world_size(ep_world_size),
        do_all_reduce(do_all_reduce_),
        use_read_mode(use_read_mode_),
        n_split(n_split_),
        barriers(barriers) {
    this->init_buffer_once(output_dtype);
    this->create_rs_barrier();

    std::vector<void *> barrier_ptrs(world_size, nullptr);
    for (int i = 0; i < this->tp_group->get_size(); i++) {
      barrier_ptrs[i] = this->barriers[i].data_ptr();
    }
    CUDA_CHECK(cudaMalloc(&this->barrier_dev_ptrs, world_size * sizeof(void *)));
    CUDA_CHECK(cudaMemcpy(
        this->barrier_dev_ptrs,
        barrier_ptrs.data(),
        world_size * sizeof(void *),
        cudaMemcpyHostToDevice));
    torch::cuda::synchronize();  // we don't assume create/run on the same stream so sync is safe
    this->barrier = this->barriers[this->rank];
  }

  torch::Tensor
  run(std::vector<torch::Tensor> gemm_outs,  // of group_size
      c10::optional<torch::Tensor> output_,
      int ep_start,
      int ep_nexperts,
      torch::Tensor splits,
      torch::Tensor routing_idx,
      c10::optional<std::vector<torch::Tensor>> output_vec_scales,
      int num_thread_blocks,
      intptr_t cp_stream) {
    at::cuda::CUDAStream stream =
        at::cuda::getStreamFromExternal((cudaStream_t)cp_stream, at::cuda::current_device());
    at::cuda::CUDAStreamGuard _(stream);
    CHECK_INPUT(routing_idx, at::ScalarType::Int);
    CHECK_INPUT(splits, at::ScalarType::Int);
    int N = this->n_dim;
    int m_full = routing_idx.size(0);
    int ntokens = m_full / this->topk;
    int ntokens_per_rank = ntokens / this->world_size;
    int ntokens_out = this->do_all_reduce ? ntokens : ntokens_per_rank;
    FLUX_CHECK_GE(gemm_outs.size(), 1);
    FLUX_CHECK_LE(gemm_outs.size(), kMaxNumGroups);
    auto dtype = gemm_outs[0].scalar_type();

    auto output = output_.value_or(empty_with_uninitialized_data(
        std::vector<int64_t>{ntokens_out, N}, gemm_outs[0].options()));
    CHECK_TYPE(output, dtype);
    CHECK_2D(output, ntokens_out, N);

    TopKReduceGatherRSV2Arguments args{
        .output_ptr = (void *)output.data_ptr(),
        .splits = splits.data_ptr<int>(),
        .routing_idx = routing_idx.data_ptr<int>(),
        .m_full = m_full,
        .n = N,
        .nexperts = ep_nexperts * this->ep_world_size,
        .topk = this->topk,
        .input_groups = (int)gemm_outs.size(),
        .do_all_reduce = this->do_all_reduce,
        .use_read_mode = this->use_read_mode,
        .threadblock_count = num_thread_blocks,
        .tile_size_m = kTileSizeM,
        .tile_size_n = kTileSizeN,
        .rank = this->rank,
        .world_size = this->world_size,
        .n_split = this->n_split,
        .barrier = this->barrier_dev_ptrs,
        .reduce_ptrs = (void **)this->reduce_buffer_dptrs.data_ptr(),
        .tile_barrier_ptrs = (int **)this->tile_barrier_dptrs.data_ptr(),
    };
    for (int i = 0; i < gemm_outs.size(); i++) {
      args.input_ptrs[i] = (void *)gemm_outs[i].data_ptr();
      args.output_vec_scale_ptrs[i] =
          output_vec_scales.has_value() ? (float *)output_vec_scales->at(i).data_ptr() : nullptr;
    }
    auto output_dtype = from_torch_dtype(dtype);
    if (this->ep_world_size == 1) {
      topk_gather_rs_v2(args, output_dtype, (cudaStream_t)cp_stream);
    } else {
      ep_topk_gather_rs_v2(args, output_dtype, ep_start, ep_nexperts, (cudaStream_t)cp_stream);
    }
    if (this->do_all_reduce) {
      cudaMemcpyAsync(
          output.data_ptr(),
          this->reduce_buffer.data_ptr(),
          ntokens * this->n_dim * output.element_size(),
          cudaMemcpyDeviceToDevice,
          (cudaStream_t)cp_stream);
    }
    return output;
  }

  void
  reset_buffer() {
    if (this->tile_barrier.defined()) {
      this->tile_barrier.zero_();
    }
  }
};

/// This class only runs the basic grouped_gemm, it is mainly used for testing
class GemmGroupedV2GatherRSOp::GemmGroupedV2GatherRSOpImpl {
 private:
  std::shared_ptr<Group> tp_group;
  int32_t ep_nexperts;
  int32_t ep_start;
  const int32_t total_num_experts;
  int32_t max_m;
  int32_t n_dim;
  int32_t topk;
  at::ScalarType output_dtype;
  int32_t max_input_groups;
  int32_t rank;
  int32_t world_size;     // the total world size
  int32_t tp_world_size;  // the world size of tensor parallel
  int32_t ep_world_size;  // the world size of expert parallel
  int n_split;
  bool do_all_reduce;
  torch::Tensor barrier;
  std::vector<torch::Tensor> barriers;
  std::unique_ptr<TopkReduceScatterOp> topk_reduce_scatter_op = nullptr;

  torch::Tensor workspace;
  cudaEvent_t gemm_start_event;
  cudaEvent_t gather_rs_done_event;
  cudaStream_t gather_rs_stream;
  GroupBarrier group_barrier;

  int
  get_barrier_size(int problem_count) const {
    return pad_to(this->n_split, 128) * 2  // 1st: ready flag per tile, 2nd: counter per split
           + pad_to(problem_count, 128);   // counter for each problem gemm done tiles
  }

  void
  create_barriers() {
    const int problem_count = this->n_split * this->ep_nexperts * this->max_input_groups;
    const int barrier_size = get_barrier_size(problem_count);
    if (this->barriers.empty()) {
      this->barriers = flux_create_tensor_list(
          std::vector<int64_t>{barrier_size}, at::ScalarType::Int, this->tp_group.get(), true);
      this->barrier = this->barriers[this->rank];
    }
  }

  void
  create_workspace_or_expand(int64_t workspace_size) {
    if (workspace_size <= 0)
      return;
    workspace_size = pad_to(workspace_size, 128);
    if (!this->workspace.defined() || workspace_size > this->workspace.numel()) {
      this->workspace = torch::empty(
          {workspace_size}, at::TensorOptions().dtype(at::ScalarType::Byte).device(at::kCUDA));
    }
  }

  c10::cuda::CUDAStream
  CreateReduceScatterStream() {
    at::cuda::CUDAGuard guard(at::cuda::current_device());
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithPriority(
        &stream, cudaStreamNonBlocking, get_highest_cuda_stream_priority()));
    return at::cuda::getStreamFromExternal(stream, at::cuda::current_device());
  }

  int
  n_split_fixed(int n_split, int n_dim) {
    if (n_dim / n_split % kTileSizeN != 0) {
      FLUX_CHECK_DIV(n_dim, kTileSizeN);
      n_split = n_dim / kTileSizeN;
    }
    return n_split;
  }

 public:
  GemmGroupedV2GatherRSOpImpl(
      std::shared_ptr<Group> tp_group_,
      int64_t total_num_experts,
      int64_t max_m,
      int64_t n_dim,
      int64_t topk,
      at::ScalarType output_dtype,
      int64_t tp_world_size,
      int64_t ep_world_size,
      int64_t max_input_groups,
      int64_t n_split_,
      bool do_all_reduce_ = false,
      bool use_read_mode = false)
      : tp_group(tp_group_),
        total_num_experts(total_num_experts),
        max_m(max_m),
        n_dim(n_dim),
        topk(topk),
        output_dtype(output_dtype),
        max_input_groups(max_input_groups),
        rank(tp_group_->get_rank()),
        world_size(tp_group_->get_size()),
        tp_world_size(tp_world_size),
        ep_world_size(ep_world_size),
        n_split(n_split_fixed(n_split_, n_dim)),
        do_all_reduce(do_all_reduce_),
        group_barrier(tp_group_, false) {
    if (this->n_split != n_split_) {
      FLUX_LOG_FIRST_N(WARN, 1) << "warning: (n / split_n) % " << kTileSizeN
                                << " != 0, set split_n=" << this->n_split << "\n";
    }
    FLUX_CHECK_EQ(this->tp_world_size * this->ep_world_size, this->world_size);
    FLUX_CHECK_DIV(this->total_num_experts, this->ep_world_size);
    FLUX_CHECK_LE(max_input_groups, kMaxNumGroups);
    this->ep_nexperts = this->total_num_experts / this->ep_world_size;
    int ep_rank = this->rank / this->tp_world_size;
    this->ep_start = this->ep_nexperts * ep_rank;
    this->gather_rs_stream = CreateReduceScatterStream();
    CUDA_CHECK(cudaEventCreateWithFlags(&this->gemm_start_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&this->gather_rs_done_event, cudaEventDisableTiming));
    create_barriers();
    topk_reduce_scatter_op = std::make_unique<TopkReduceScatterOp>(
        tp_group_,
        max_m,
        n_dim,
        topk,
        output_dtype,
        total_num_experts / ep_world_size,
        ep_world_size,
        this->barriers,
        this->n_split,
        do_all_reduce_,
        use_read_mode);
  }

  torch::Tensor
  forward_gather_rs_impl(
      std::vector<torch::Tensor> inputs,
      std::vector<torch::Tensor> weights,
      torch::Tensor splits,
      torch::Tensor routing_idx,
      c10::optional<std::vector<torch::Tensor>> bias,
      c10::optional<std::vector<torch::Tensor>> input_scales,
      c10::optional<std::vector<torch::Tensor>> weight_scales,
      c10::optional<std::vector<torch::Tensor>> output_vec_scales,
      bool fast_accum,
      int sm_margin,
      bool with_stream_sync,
      c10::optional<UnifiedGemmHParams> const &hparams) {
    /*
      Note: When expert parallel is enabled, the inputs/weights tensor should be
      the partial the current expert parallel rank. But the splits_cpu and routing
      idx should be global no matter whether expert parallel is enabled, which means the
      splits_cpu/routing_idx should contains all the experts / tokens no matter whether expert
      parallel is enabled.
    */
    FLUX_CHECK(!bias.has_value());
    FLUX_CHECK_LE(inputs.size(), this->max_input_groups);
    int num_groups = inputs.size();
    FLUX_CHECK_LE(num_groups, this->max_input_groups);
    FLUX_CHECK_EQ(num_groups, weights.size());

    at::ScalarType input_torch_type = weights[0].scalar_type();
    FLUX_CHECK(input_torch_type != at::ScalarType::Char)
        << "Moe AG+Scatter INT8 not supported yet";
    bool is_fp8 = is_fp8_torch_dtype(input_torch_type);
    // if the dtype of input is fp8, use bfloat16 as the output dtype
    at::ScalarType output_torch_type = is_fp8 ? at::ScalarType::BFloat16 : input_torch_type;
    DataTypeEnum output_type = from_torch_dtype(output_torch_type);
    int m_full = routing_idx.size(0);
    int ntokens = m_full / this->topk;
    int n_tokens_per_rank = ntokens / this->world_size;
    int M_this_ep = inputs[0].size(0);
    int K = inputs[0].size(1);
    int E = weights[0].size(0);
    int N = weights[0].size(1);
    // check input/weight
    for (int i = 0; i < num_groups; i++) {
      CHECK_3D(weights[i], this->ep_nexperts, N, K);  // only RCR layout supported
      CHECK_INPUT(weights[i], input_torch_type);
      CHECK_2D(inputs[i], M_this_ep, K);
      CHECK_INPUT(inputs[i], input_torch_type);
    }
    // check input_scale/weight_scale/output_vec_scale
    if (input_scales.has_value()) {
      FLUX_CHECK_EQ(input_scales->size(), num_groups);
      for (auto &input_scale : input_scales.value()) {
        CHECK_1D(input_scale, 1);
        CHECK_INPUT(input_scale, at::ScalarType::Float);
      }
    }
    if (weight_scales.has_value()) {
      FLUX_CHECK_EQ(weight_scales->size(), num_groups);
      for (auto &weight_scale : weight_scales.value()) {
        CHECK_1D(weight_scale, E);
        CHECK_INPUT(weight_scale, at::ScalarType::Float);
      }
    }
    if (output_vec_scales.has_value()) {
      FLUX_CHECK_EQ(output_vec_scales->size(), num_groups);
      for (auto &output_vec_scale : output_vec_scales.value()) {
        CHECK_1D(output_vec_scale, M_this_ep);
        CHECK_INPUT(output_vec_scale, at::ScalarType::Float);
      }
    }

    CHECK_INPUT(routing_idx, at::ScalarType::Int);
    if (this->ep_world_size == 1) {
      FLUX_CHECK_EQ(M_this_ep, m_full);
    } else {
      FLUX_CHECK_LE(M_this_ep, m_full) << "input.size(0) larger than routing_idx.size(0)";
    }
    FLUX_CHECK_DIV(m_full, this->world_size * this->topk);
    FLUX_CHECK_LE(m_full, this->max_m) << "input.size(0) " << M_this_ep << " larger than max_m\n";
    FLUX_CHECK_EQ(N, this->n_dim);

    FLUX_CHECK_GE(N, 8) << "N must be greater than or equal 8 for cutlass grouped gemm.";
    FLUX_CHECK_GE(K, 8) << "K must be greater than or equal 8 for cutlass grouped gemm.";
    torch::Tensor splits_gpu;
    if (!splits.is_cuda()) {
      splits_gpu = empty_with_uninitialized_data(
          splits.sizes(), at::TensorOptions(c10::kCUDA).dtype(at::ScalarType::Int));
      splits_gpu.copy_(splits, true);
    } else {
      splits_gpu = splits;
    }
    CHECK_INPUT(splits_gpu, at::ScalarType::Int);
    CHECK_1D(splits_gpu, this->total_num_experts);

    auto stream = c10::cuda::getCurrentCUDAStream();

    ArchEnum arch = get_arch();
    SMCoreEnum sm_core = get_sm_core();
    auto input_type = from_torch_dtype(input_torch_type);
    auto dt_conf = to_gemm_dtype_config(
        make_gemm_dtype_config(input_type, input_type, output_type, output_type));
    auto impl_spec = make_gemm_v2_meta(fast_accum and dt_conf.is_input_fp8());
    // always use topk=1 impl: to save some compile time
    auto comm_spec = make_gather_rs_meta(1);
    auto meta = make_gemm_meta(
        dt_conf, arch, sm_core, _GatherRS{}, _RCR{}, _GemmGroupedV2{}(), impl_spec, comm_spec);
    auto rt_conf = make_runtime_config(N, cute::ceil_div(m_full, this->ep_nexperts), K);
    OpRegistry::OpPtr gemm_op;
    if (hparams.has_value()) {
      gemm_op = OpRegistry::instance().get_op(meta, hparams.value());
    } else {
      gemm_op = OpRegistry::instance().get_op(meta, rt_conf);
    }

    std::vector<torch::Tensor> gemm_outs;
    for (int i = 0; i < num_groups; i++) {
      gemm_outs.push_back(empty_with_uninitialized_data(
          std::vector<int64_t>{M_this_ep, N},
          at::TensorOptions(at::kCUDA).dtype(output_torch_type)));
    }
    torch::Tensor output = empty_with_uninitialized_data(
        std::vector<int64_t>{this->do_all_reduce ? ntokens : n_tokens_per_rank, N},
        at::TensorOptions(at::kCUDA).dtype(output_torch_type));

    MoeGatherRSWorkspaceArgs ws_args{
        .num_groups = num_groups,
        .N_split = this->n_split,
        .ep_start = this->ep_start,
        .ep_nexperts = this->ep_nexperts,
        .N = N,
        .K = K,
        .splits_gpu = splits_gpu.data_ptr<int>()};
    for (int i = 0; i < num_groups; i++) {
      ws_args.input[i] = inputs[i].data_ptr();
      ws_args.weights[i] = weights[i].data_ptr();
      ws_args.output[i] = gemm_outs[i].data_ptr();
      ws_args.input_scales[i] =
          input_scales.has_value() ? input_scales->at(i).data_ptr<float>() : nullptr;
      ws_args.weight_scales[i] =
          weight_scales.has_value() ? weight_scales->at(i).data_ptr<float>() : nullptr;
    }

    int problem_count = ws_args.N_split * ws_args.num_groups * ws_args.ep_nexperts;
    torch::Tensor workspace_gpu = empty_with_uninitialized_data(
        std::vector<int64_t>{get_args_workspace_size(problem_count)},
        at::TensorOptions(at::kCUDA).dtype(at::ScalarType::Char));
    void *workspace = workspace_gpu.data_ptr();
    make_workspace(
        ws_args,
        GemmLayoutEnum::RCR,
        c10::elementSize(input_torch_type),
        c10::elementSize(output_torch_type),
        workspace,
        stream);

    constexpr int kAlignment = 128;

    // the offsets
    int offset_problem_sizes = 0;
    int offset_ptr_A = pad_to(
        offset_problem_sizes + problem_count * sizeof(cutlass::gemm::GemmCoord), kAlignment);
    int offset_ptr_B = pad_to(offset_ptr_A + problem_count * sizeof(void *), kAlignment);
    int offset_ptr_C = pad_to(offset_ptr_B + problem_count * sizeof(void *), kAlignment);
    int offset_ptr_D = pad_to(offset_ptr_C + problem_count * sizeof(void *), kAlignment);
    int offset_lda = pad_to(offset_ptr_D + problem_count * sizeof(void *), kAlignment);
    int offset_ldb = pad_to(offset_lda + problem_count * sizeof(int64_t), kAlignment);
    int offset_ldc = pad_to(offset_ldb + problem_count * sizeof(int64_t), kAlignment);
    int offset_ldd = pad_to(offset_ldc + problem_count * sizeof(int64_t), kAlignment);
    int offset_ldr = pad_to(offset_ldd + problem_count * sizeof(int64_t), kAlignment);
    int offset_scale_A = pad_to(offset_ldr + problem_count * sizeof(int64_t), kAlignment);
    int offset_scale_B = pad_to(offset_scale_A + problem_count * sizeof(float *), kAlignment);
    int offset_non_empty_problem_count =
        pad_to(offset_scale_B + problem_count * sizeof(float *), kAlignment);
    // the ptrs
    cutlass::gemm::GemmCoord *problem_sizes =
        (cutlass::gemm::GemmCoord *)((char *)workspace + offset_problem_sizes);
    void **ptr_A = (void **)((char *)workspace + offset_ptr_A);
    void **ptr_B = (void **)((char *)workspace + offset_ptr_B);
    void **ptr_C = (void **)((char *)workspace + offset_ptr_C);
    void **ptr_D = (void **)((char *)workspace + offset_ptr_D);
    int64_t *lda = (int64_t *)((char *)workspace + offset_lda);
    int64_t *ldb = (int64_t *)((char *)workspace + offset_ldb);
    int64_t *ldc = (int64_t *)((char *)workspace + offset_ldc);
    int64_t *ldd = (int64_t *)((char *)workspace + offset_ldd);
    int64_t *ldr = (int64_t *)((char *)workspace + offset_ldr);
    float **scale_A = (float **)((char *)workspace + offset_scale_A);
    float **scale_B = (float **)((char *)workspace + offset_scale_B);
    int *non_empty_problem_count = (int *)((char *)workspace + offset_non_empty_problem_count);

    float alpha = 1.0, beta = 0.0;

    GemmGroupedV2GatherRSArguments args{
        .problem_sizes = problem_sizes,
        .problem_count = problem_count,
        .non_empty_problem_count = non_empty_problem_count,
        .alpha = alpha,
        .beta = beta,
        .ptr_A = ptr_A,
        .ptr_B = ptr_B,
        .ptr_C = ptr_C,
        .ptr_D = ptr_D,
        .lda = lda,
        .ldb = ldb,
        .ldc = ldc,
        .ldd = ldd,
        .ldr = ldr,
        .scaleA = (float const **)scale_A,
        .scaleB = (float const **)scale_B,
        .topk = this->topk,
        .barrier = this->barrier.data_ptr<int>(),
        .routing_idx = routing_idx.data_ptr<int32_t>(),
        .n_split = n_split,
        .sm_margin = sm_margin + get_rs_threadblock_count()};

    int64_t workspace_size = gemm_op->get_workspace_size(args);
    this->create_workspace_or_expand(workspace_size);

    group_barrier.barrier_all(stream);

    // ensure barrier initialized correctly
    CUDA_CHECK(cudaEventRecord(this->gemm_start_event, stream));
    CUDA_CHECK(cudaStreamWaitEvent(gather_rs_stream, this->gemm_start_event));

    if (M_this_ep > 0) {
      gemm_op->run(args, this->workspace.defined() ? this->workspace.data_ptr() : nullptr, stream);
    } else {
      this->barrier.fill_(1);
    }
    output = topk_reduce_scatter_op->run(
        gemm_outs,
        output,
        this->ep_start,
        this->ep_nexperts,
        splits_gpu,
        routing_idx,
        output_vec_scales,
        get_rs_threadblock_count(),
        (intptr_t)gather_rs_stream);
    CUDA_CHECK(cudaEventRecord(this->gather_rs_done_event, gather_rs_stream));
    CUDA_CHECK(cudaStreamWaitEvent(stream, this->gather_rs_done_event));

    group_barrier.barrier_all(stream);
    this->barrier.zero_();
    this->topk_reduce_scatter_op->reset_buffer();
    return output;
  }

  torch::Tensor
  forward_gather_rs(
      torch::Tensor input,
      torch::Tensor weight,
      torch::Tensor splits_cpu,
      torch::Tensor routing_idx,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_vec_scale,
      bool fast_accum,
      int sm_margin,
      bool with_stream_sync) {
    if (input.scalar_type() == torch::kInt8) {
      return forward_gather_rs_triton_aot(
          input,
          weight,
          splits_cpu,
          routing_idx,
          c10::nullopt,
          input_scale,
          weight_scale,
          output_vec_scale,
          fast_accum,
          sm_margin,
          with_stream_sync);
    }
    return forward_gather_rs_impl(
        {std::move(input)},
        {std::move(weight)},
        std::move(splits_cpu),
        std::move(routing_idx),
        as_optional_vec(bias),
        as_optional_vec(input_scale),
        as_optional_vec(weight_scale),
        as_optional_vec(output_vec_scale),
        fast_accum,
        sm_margin,
        with_stream_sync,
        c10::nullopt);
  }

  torch::Tensor
  forward_gather_rs_triton_aot(
      torch::Tensor input,
      torch::Tensor weight,
      torch::Tensor splits,
      torch::Tensor routing_idx,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_vec_scale,
      bool fast_accum,
      int sm_margin,
      bool with_stream_sync) {
#if defined(FLUX_WITH_TRITON_AOT)
    int M_this_ep = input.size(0);
    int K = input.size(1);
    int E = weight.size(0);
    int N = weight.size(1);
    int m_full = routing_idx.size(0);
    int ntokens = m_full / this->topk;
    int n_tokens_per_rank = ntokens / this->world_size;
    at::ScalarType input_dtype = weight.scalar_type();
    bool is_fp8 = is_fp8_torch_dtype(input_dtype);
    bool is_s8_gemm = input_dtype == at::ScalarType::Char;
    CHECK_INPUT(input, input_dtype);
    CHECK_INPUT(weight, input_dtype);

    // check input_scale/weight_scale/output_vec_scale
    if (input_scale.has_value()) {
      if (is_s8_gemm) {
        FLUX_CHECK_EQ(input_scale->numel(), M_this_ep);
      } else {
        CHECK_1D(input_scale.value(), 1);
      }
      CHECK_INPUT(input_scale.value(), at::ScalarType::Float);
    }
    FLUX_CHECK(weight_scale.has_value());
    if (weight_scale.has_value()) {
      if (is_s8_gemm) {
        CHECK_2D(weight_scale.value(), E, N);
      } else {
        CHECK_1D(weight_scale.value(), E);
      }
      CHECK_INPUT(weight_scale.value(), at::ScalarType::Float);
    }
    FLUX_CHECK(output_vec_scale.has_value());
    if (output_vec_scale.has_value()) {
      FLUX_CHECK_EQ(output_vec_scale->numel(), M_this_ep);
      CHECK_INPUT(output_vec_scale.value(), at::ScalarType::Float);
    }

    CHECK_INPUT(routing_idx, at::ScalarType::Int);
    if (this->ep_world_size == 1) {
      FLUX_CHECK_EQ(M_this_ep, m_full);
    } else {
      FLUX_CHECK_LE(M_this_ep, m_full) << "input.size(0) larger than routing_idx.size(0)";
    }
    FLUX_CHECK_DIV(m_full, this->world_size * this->topk);
    FLUX_CHECK_LE(m_full, this->max_m) << "input.size(0) " << M_this_ep << " larger than max_m\n";
    FLUX_CHECK_EQ(N, this->n_dim);
    FLUX_CHECK_DIV(N, 16) << "N % 16 == 0 expected for triton grouped gemm.";
    FLUX_CHECK_DIV(K, 16) << "K % 16 == 0 expected for triton grouped gemm.";

    torch::Tensor splits_cpu, splits_gpu;
    auto option_cpu = at::TensorOptions(at::kCPU).pinned_memory(true).dtype(at::ScalarType::Int);
    auto option_gpu = at::TensorOptions(at::kCUDA).dtype(at::ScalarType::Int);
    if (splits.is_cuda()) {
      splits_gpu = splits;
      splits_cpu = empty_with_uninitialized_data(splits.sizes(), option_cpu);
      splits_cpu.copy_(splits, false);  // non-blocking copy
    } else {
      splits_cpu = splits;
      splits_gpu = empty_with_uninitialized_data(splits.sizes(), option_gpu);
      auto splits_pin = empty_with_uninitialized_data(splits.sizes(), option_cpu);
      splits_pin.copy_(splits, true);      // async copy
      splits_gpu.copy_(splits_pin, true);  // async copy
    }
    at::ScalarType output_dtype = is_fp8 || is_s8_gemm ? at::ScalarType::BFloat16 : input_dtype;

    using FuncType = decltype(moe_gather_rs_grouped_gemm_s8_ex);
    FuncType *grouped_gemm_func = nullptr;
    moe_gather_rs_grouped_gemm_kernel__triton_algo_info_t algo_info;
    if (is_s8_gemm) {
      grouped_gemm_func = moe_gather_rs_grouped_gemm_s8_ex;
      algo_info = moe_gather_rs_grouped_gemm_kernel__triton_algo_info_t{
          .N_SPLIT = n_split,
          .BLOCK_SIZE_M = 64,
          .BLOCK_SIZE_N = 128,
          .BLOCK_SIZE_K = 64,
          .num_warps = 4,
          .num_stages = 4};
    } else if (input_dtype == torch::kHalf) {
      grouped_gemm_func = moe_gather_rs_grouped_gemm_fp16_ex;
      algo_info = moe_gather_rs_grouped_gemm_kernel__triton_algo_info_t{
          .N_SPLIT = n_split,
          .BLOCK_SIZE_M = 128,
          .BLOCK_SIZE_N = 128,
          .BLOCK_SIZE_K = 64,
          .num_warps = 4,
          .num_stages = 4};
    } else if (input_dtype == torch::kBFloat16) {
      grouped_gemm_func = moe_gather_rs_grouped_gemm_bf16_ex;
      algo_info = moe_gather_rs_grouped_gemm_kernel__triton_algo_info_t{
          .N_SPLIT = n_split,
          .BLOCK_SIZE_M = 128,
          .BLOCK_SIZE_N = 128,
          .BLOCK_SIZE_K = 64,
          .num_warps = 4,
          .num_stages = 4};
    } else {
      FLUX_CHECK(false) << "unsupported dtype " << input_dtype;
    }

    int *splits_ptr = splits_cpu.data_ptr<int>() + this->ep_start;
    int blocked_m_tiles = 0;
    int tile_size_m = algo_info.BLOCK_SIZE_M;
    for (int i = 0; i < this->ep_nexperts; i++) {
      blocked_m_tiles += (splits_ptr[i] + tile_size_m - 1) / tile_size_m;
    }
    torch::Tensor gather_a_index = empty_with_uninitialized_data(
        std::vector<int64_t>{tile_size_m * blocked_m_tiles}, option_gpu);
    torch::Tensor expert_index =
        empty_with_uninitialized_data(std::vector<int64_t>{blocked_m_tiles}, option_gpu);
    auto stream = at::cuda::getCurrentCUDAStream();
    calc_moe_triton_blocked_gather_a(
        splits_gpu.data_ptr<int>(),
        this->ep_start,
        this->ep_nexperts,
        tile_size_m,
        gather_a_index.data_ptr<int>(),
        expert_index.data_ptr<int>(),
        ep_nexperts,
        1024,
        stream);
    torch::Tensor gemm_out = empty_with_uninitialized_data(
        std::vector<int64_t>{M_this_ep, N}, option_gpu.dtype(output_dtype));

    torch::Tensor output = empty_with_uninitialized_data(
        std::vector<int64_t>{this->do_all_reduce ? ntokens : n_tokens_per_rank, N},
        at::TensorOptions(at::kCUDA).dtype(output_dtype));

    group_barrier.barrier_all(stream);

    // ensure barrier initialized correctly
    CUDA_CHECK(cudaEventRecord(this->gemm_start_event, stream));

    if (M_this_ep == 0) {
      this->barrier.fill_(1);
    } else {
      auto rtn = grouped_gemm_func(
          (CUstream)stream,
          (CUdeviceptr)input.data_ptr(),
          (CUdeviceptr)weight.data_ptr(),
          (CUdeviceptr)gemm_out.data_ptr(),
          (CUdeviceptr)data_ptr_or(input_scale, nullptr),       // input_scale
          (CUdeviceptr)data_ptr_or(weight_scale, nullptr),      // weight_scale
          (CUdeviceptr)data_ptr_or(output_vec_scale, nullptr),  // output_scale
          (CUdeviceptr)gather_a_index.data_ptr(),
          (CUdeviceptr)expert_index.data_ptr(),
          blocked_m_tiles * tile_size_m,
          N,
          K,
          ep_nexperts,
          M_this_ep,
          input.stride(0),
          input.stride(1),
          weight.stride(0),
          weight.stride(2),
          weight.stride(1),  // transpose_weight
          gemm_out.stride(0),
          gemm_out.stride(1),
          (CUdeviceptr)barrier.data_ptr(),
          algo_info);
      CU_CHECK(rtn);
    }

    // ensure barrier initialized correctly
    CUDA_CHECK(cudaStreamWaitEvent(gather_rs_stream, this->gemm_start_event));
    output = this->topk_reduce_scatter_op->run(
        {gemm_out},
        output,
        this->ep_start,
        this->ep_nexperts,
        splits_gpu,
        routing_idx,
        c10::nullopt,
        get_rs_threadblock_count(),
        (intptr_t)gather_rs_stream);
    CUDA_CHECK(cudaEventRecord(this->gather_rs_done_event, gather_rs_stream));
    CUDA_CHECK(cudaStreamWaitEvent(stream, this->gather_rs_done_event));

    group_barrier.barrier_all(stream);
    this->barrier.zero_();
    this->topk_reduce_scatter_op->reset_buffer();
    return output;
#else
    FLUX_CHECK(false) << "please compile with --triton-aot option.";
#endif
  }

  torch::Tensor
  profiling(
      torch::Tensor input,
      torch::Tensor weight,
      torch::Tensor splits_cpu,
      torch::Tensor routing_idx,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_vec_scale,
      bool fast_accum,
      int sm_margin,
      bool with_stream_sync,
      c10::intrusive_ptr<ProfilingContext> opt_ctx) {
    int full_m = routing_idx.size(0);
    int K = input.size(1);
    int E = weight.size(0);
    int N = weight.size(1);
    ArchEnum arch = get_arch();
    SMCoreEnum sm_core = get_sm_core();
    auto weight_dtype = weight.scalar_type();
    auto dtype = from_torch_dtype(weight_dtype);
    bool is_fp8 = (dtype == _E4M3{}) || (dtype == _E5M2{});
    // if the dtype of input is fp8, use bfloat16 as the output dtype
    DataTypeEnum output_type = is_fp8 ? dtype : _BF16{};
    auto dt_conf =
        to_gemm_dtype_config(make_gemm_dtype_config(dtype, dtype, output_type, output_type));
    auto impl_spec = make_gemm_v2_meta(fast_accum and dt_conf.is_input_fp8());
    // always use topk=1 impl: to save some compile time
    auto comm_spec = make_gather_rs_meta(1);
    auto meta = unify_type(make_gemm_meta(
        dt_conf, arch, sm_core, _GatherRS{}, _RCR{}, _GemmGroupedV2{}(), impl_spec, comm_spec));
    auto rt_conf = make_runtime_config(cute::ceil_div(full_m, this->ep_nexperts), N, K);
    ProfilingContext tmp_ctx("__tmp__");
    ProfilingContext *ctx = opt_ctx == nullptr ? &tmp_ctx : opt_ctx.get();

    OpRegistry::instance().visit_hparams(
        [&](UnifiedGemmHParams const &hparams) {
          constexpr int warm_iters = 5;
          constexpr int iters = 10;
          float total_elapsed = 0;
          auto cp_hparams = hparams;
          auto comm_params = std::get<unified_type_t<GatherRSHParams>>(cp_hparams.comm_spec());
          if (comm_params.n_dim_per_split() != N / this->n_split) {
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
                as_optional_vec(bias),
                as_optional_vec(input_scale),
                as_optional_vec(weight_scale),
                as_optional_vec(output_vec_scale),
                fast_accum,
                sm_margin,
                false,  // whether with stream sync
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
        as_optional_vec(bias),
        as_optional_vec(input_scale),
        as_optional_vec(weight_scale),
        as_optional_vec(output_vec_scale),
        fast_accum,
        sm_margin,
        false,  // whether with stream sync
        std::move(best_hparams));
  }

  torch::Tensor
  forward_gather_rs_multiple(
      std::vector<torch::Tensor> inputs,
      std::vector<torch::Tensor> weights,
      torch::Tensor splits_cpu,
      torch::Tensor routing_idx,
      c10::optional<std::vector<torch::Tensor>> bias,
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
        std::move(bias),
        std::move(input_scale),
        std::move(weight_scale),
        std::move(output_vec_scale),
        fast_accum,
        sm_margin,
        with_stream_sync,
        c10::nullopt);
  }
  std::tuple<int64_t, int64_t, int64_t>
  get_pickle_info() const {
    return std::make_tuple(this->max_m, this->n_dim, this->ep_nexperts);
  }
};

TopkReduceScatterOp::TopkReduceScatterOp(
    std::shared_ptr<Group> tp_group,
    int max_m,
    int n_dim,
    int topk,
    at::ScalarType output_dtype,
    int ep_nexperts,
    int ep_world_size,
    std::vector<torch::Tensor> barriers,
    int n_split,
    bool do_all_reduce,
    bool use_read_mode)
    : impl_(new TopkReduceScatterOpImpl(
          tp_group,
          max_m,
          n_dim,
          topk,
          output_dtype,
          ep_nexperts,
          ep_world_size,
          barriers,
          n_split,
          do_all_reduce,
          use_read_mode)) {}
TopkReduceScatterOp::~TopkReduceScatterOp() { delete impl_; }
void
TopkReduceScatterOp::reset_buffer() {
  FLUX_CHECK(impl_ != nullptr) << "TopkReduceScatterOp not initialized";
  impl_->reset_buffer();
}
torch::Tensor
TopkReduceScatterOp::run(
    std::vector<torch::Tensor> gemm_outs,  // of group_size
    c10::optional<torch::Tensor> output,
    int ep_start,
    int ep_nexperts,
    torch::Tensor splits,
    torch::Tensor routing_idx,
    c10::optional<std::vector<torch::Tensor>> output_vec_scales,
    int num_thread_blocks,
    intptr_t cp_stream) {
  FLUX_CHECK(impl_ != nullptr) << "TopkReduceScatterOp not initialized";
  return impl_->run(
      std::move(gemm_outs),
      std::move(output),
      ep_start,
      ep_nexperts,
      std::move(splits),
      std::move(routing_idx),
      std::move(output_vec_scales),
      num_thread_blocks,
      cp_stream);
}

GemmGroupedV2GatherRSOp::GemmGroupedV2GatherRSOp(
    std::shared_ptr<Group> tp_group_,
    int64_t total_num_experts,
    int64_t max_m,
    int64_t n_dim,
    int64_t topk,
    at::ScalarType output_dtype,
    int64_t tp_world_size,
    int64_t ep_world_size,
    int64_t max_input_groups,
    int64_t n_split_,
    bool do_all_reduce,
    bool use_read_mode)
    : impl_(new GemmGroupedV2GatherRSOpImpl(
          tp_group_,
          total_num_experts,
          max_m,
          n_dim,
          topk,
          output_dtype,
          tp_world_size,
          ep_world_size,
          max_input_groups,
          n_split_,
          do_all_reduce,
          use_read_mode)) {}

GemmGroupedV2GatherRSOp::~GemmGroupedV2GatherRSOp() { delete impl_; }
torch::Tensor
GemmGroupedV2GatherRSOp::forward_gather_rs(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor splits_cpu,
    torch::Tensor routing_idx,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale,
    c10::optional<torch::Tensor> output_vec_scale,
    bool fast_accum,
    int sm_margin,
    bool with_stream_sync) {
  FLUX_CHECK(impl_ != nullptr) << "GemmGroupedV2GatherRSOp not initialized";
  return impl_->forward_gather_rs(
      std::move(input),
      std::move(weight),
      std::move(splits_cpu),
      std::move(routing_idx),
      std::move(bias),
      std::move(input_scale),
      std::move(weight_scale),
      std::move(output_vec_scale),
      fast_accum,
      sm_margin,
      with_stream_sync);
}
torch::Tensor
GemmGroupedV2GatherRSOp::forward_gather_rs_triton_aot(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor splits,
    torch::Tensor routing_idx,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale,
    c10::optional<torch::Tensor> output_vec_scale,
    bool fast_accum,
    int sm_margin,
    bool with_stream_sync) {
  FLUX_CHECK(impl_ != nullptr) << "GemmGroupedV2GatherRSOp not initialized";
  return impl_->forward_gather_rs_triton_aot(
      std::move(input),
      std::move(weight),
      std::move(splits),
      std::move(routing_idx),
      std::move(bias),
      std::move(input_scale),
      std::move(weight_scale),
      std::move(output_vec_scale),
      fast_accum,
      sm_margin,
      with_stream_sync);
}
torch::Tensor
GemmGroupedV2GatherRSOp::profiling(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor splits_cpu,
    torch::Tensor routing_idx,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale,
    c10::optional<torch::Tensor> output_vec_scale,
    bool fast_accum,
    int sm_margin,
    bool with_stream_sync,
    c10::intrusive_ptr<ProfilingContext> opt_ctx) {
  FLUX_CHECK(impl_ != nullptr) << "GemmGroupedV2GatherRSOp not initialized";
  return impl_->profiling(
      std::move(input),
      std::move(weight),
      std::move(splits_cpu),
      std::move(routing_idx),
      std::move(bias),
      std::move(input_scale),
      std::move(weight_scale),
      std::move(output_vec_scale),
      fast_accum,
      sm_margin,
      with_stream_sync,
      std::move(opt_ctx));
}
torch::Tensor
GemmGroupedV2GatherRSOp::forward_gather_rs_multiple(
    std::vector<torch::Tensor> inputs,
    std::vector<torch::Tensor> weights,
    torch::Tensor splits_cpu,
    torch::Tensor routing_idx,
    c10::optional<std::vector<torch::Tensor>> bias,
    c10::optional<std::vector<torch::Tensor>> input_scale,
    c10::optional<std::vector<torch::Tensor>> weight_scale,
    c10::optional<std::vector<torch::Tensor>> output_vec_scale,
    bool fast_accum,
    int sm_margin,
    bool with_stream_sync) {
  FLUX_CHECK(impl_ != nullptr) << "GemmGroupedV2GatherRSOp not initialized";
  return impl_->forward_gather_rs_multiple(
      std::move(inputs),
      std::move(weights),
      std::move(splits_cpu),
      std::move(routing_idx),
      std::move(bias),
      std::move(input_scale),
      std::move(weight_scale),
      std::move(output_vec_scale),
      fast_accum,
      sm_margin,
      with_stream_sync);
}

}  // namespace bytedance::flux::ths_op
