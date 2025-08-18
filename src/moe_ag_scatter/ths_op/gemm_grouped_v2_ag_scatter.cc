//===- gemm_grouped_v2_ag_scatter.cc ------------------------------ C++ ---===//
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
#include "moe_ag_scatter/ths_op/gemm_grouped_v2_ag_scatter.h"
#include "coll/ths_op/all_gather_op.h"
#include "coll/ths_op/all_gather_types.h"
#include "flux/args/moe_ag_scatter.h"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/cuda_stub.h"
#include "flux/flux.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_meta.h"
#include "flux/op_registry.h"
#include "flux/ths_op/flux_shm.h"
#include "flux/ths_op/ths_op.h"
#include "flux/ths_op/topo_utils.h"
#include "flux/ths_op/util.h"
#include "moe_ag_scatter/sort_util.h"
#include "moe_ag_scatter/triton_util.h"
#include "moe_ag_scatter/workspace_util.h"
#include <ATen/core/jit_type.h>
#include <ATen/core/List.h>
#include <ATen/core/TensorBody.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/ops/empty.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/Logging.h>
#include <c10/util/Optional.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <utility>
#include <torch/cuda.h>
#include <torch/types.h>
#if defined(FLUX_WITH_TRITON_AOT)
#include "triton_aot_generated/flux_triton_aot.h"
#endif

namespace {
c10::optional<std::vector<torch::Tensor>>
as_optional_vec(c10::optional<torch::Tensor> &t) {
  if (t.has_value()) {
    return c10::optional<std::vector<torch::Tensor>>{{t.value()}};
  }
  return {};
}
}  // namespace

namespace bytedance::flux::ths_op {

/**
 * @return M_this_ep, M_this_ep_pad, gather_A_index, scatter_D_index, expert_idx, rank_start_idx,
 * rank_end_idx
 */
std::tuple<
    int,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
prepare_moe_ag_scatter_args(
    torch::Tensor splits_gpu,
    torch::Tensor scatter_index,
    int ntokens,
    int topk,
    int num_weights_group,
    int ep_start,
    int ep_nexperts,
    int rank,
    int world_size,
    int tile_size_m,
    intptr_t stream_) {
  cudaStream_t stream = (cudaStream_t)stream_;
  int nexperts = splits_gpu.numel();  // TODO(houqi.1993) no drop tokens?

  // should be M_this_ep, but never mind gather_index takes little memory
  torch::Tensor gather_index = empty_with_uninitialized_data(
      std::vector<int64_t>{ntokens * topk},
      torch::TensorOptions(torch::kCUDA).dtype(at::ScalarType::Int));
  torch::Tensor sorted_gather_index = empty_with_uninitialized_data(
      std::vector<int64_t>{ntokens * topk},
      torch::TensorOptions(torch::kCUDA).dtype(at::ScalarType::Int));
  torch::Tensor sorted_scatter_index = empty_with_uninitialized_data(
      std::vector<int64_t>{ntokens * topk},
      torch::TensorOptions(torch::kCUDA).dtype(at::ScalarType::Int));
  torch::Tensor M_this_ep_holder = empty_with_uninitialized_data(
      std::vector<int64_t>{1},
      torch::TensorOptions(torch::kCPU).dtype(at::ScalarType::Int).pinned_memory(true));
  torch::Tensor sorted_splits = empty_with_uninitialized_data(
      std::vector<int64_t>{ep_nexperts * world_size},
      torch::TensorOptions(torch::kCUDA).dtype(at::ScalarType::Int));
  torch::Tensor sorted_splits_cumsum = empty_with_uninitialized_data(
      std::vector<int64_t>{ep_nexperts * world_size},
      torch::TensorOptions(torch::kCUDA).dtype(at::ScalarType::Int));
  calc_gather_index_impl(
      nexperts,
      ntokens,
      topk,
      ep_start,
      ep_start + ep_nexperts,
      splits_gpu.const_data_ptr<int32_t>(),
      scatter_index.const_data_ptr<int32_t>(),
      gather_index.data_ptr<int32_t>(),
      M_this_ep_holder.data_ptr<int>(),
      stream);

  AGScatterSortOpArgumentsV2 args = {
      rank,
      world_size,
      ntokens,
      ep_nexperts,
      splits_gpu.const_data_ptr<int32_t>() + ep_start,
      gather_index.const_data_ptr<int32_t>(),
      sorted_splits.data_ptr<int32_t>(),
      sorted_splits_cumsum.data_ptr<int32_t>(),
      sorted_scatter_index.data_ptr<int32_t>(),
      sorted_gather_index.data_ptr<int32_t>(),
  };
  ag_scatter_sort_impl_v2(args, stream);

  int M_this_ep = scatter_index.numel();  // for EP=1, M_this_ep is always M_full
  if (ep_nexperts != nexperts) {
    CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)stream));
    M_this_ep = *M_this_ep_holder.const_data_ptr<int32_t>();
  }

  int max_problem_schedule_size = world_size * ep_nexperts * num_weights_group;
  torch::Tensor problem_schedules_gpu = empty_with_uninitialized_data(
      std::vector<int64_t>{(int64_t)(max_problem_schedule_size * sizeof(ProblemSchedV2))},
      torch::TensorOptions(torch::kByte).device(at::kCUDA));

  get_sorted_problem_schedule_cuda_v2(
      splits_gpu.data_ptr<int32_t>(),
      rank,
      world_size,
      sorted_splits_cumsum.data_ptr<int32_t>(),
      ep_start,
      ep_nexperts,
      tile_size_m,
      num_weights_group,
      (ProblemSchedV2 *)problem_schedules_gpu.data_ptr(),
      stream);

  // maybe larger than needed, but never mind the waste, just too little
  int m_pad = pad_to(M_this_ep, tile_size_m) + ep_nexperts * tile_size_m;
  int num_tiles_pad = m_pad / tile_size_m;

  auto option = torch::TensorOptions(torch::kInt32).device(torch::kCUDA);
  torch::Tensor m_pad_holder = empty_with_uninitialized_data(std::vector<int64_t>{1}, option);
  torch::Tensor gather_a_index =
      empty_with_uninitialized_data(std::vector<int64_t>{m_pad}, option);
  torch::Tensor scatter_d_index =
      empty_with_uninitialized_data(std::vector<int64_t>{m_pad}, option);
  torch::Tensor expert_index =
      empty_with_uninitialized_data(std::vector<int64_t>{num_tiles_pad}, option);
  torch::Tensor rank_start_index =
      empty_with_uninitialized_data(std::vector<int64_t>{num_tiles_pad}, option);
  torch::Tensor rank_end_index =
      empty_with_uninitialized_data(std::vector<int64_t>{num_tiles_pad}, option);

  get_moe_ag_scatter_args(
      splits_gpu.data_ptr<int>(),
      sorted_splits_cumsum.data_ptr<int>(),
      problem_schedules_gpu.data_ptr(),
      max_problem_schedule_size,
      sorted_gather_index.data_ptr<int>(),
      sorted_scatter_index.data_ptr<int>(),
      ep_start,
      ep_nexperts,
      world_size,
      M_this_ep,
      tile_size_m,
      m_pad_holder.data_ptr<int>(),
      gather_a_index.data_ptr<int32_t>(),
      scatter_d_index.data_ptr<int32_t>(),
      expert_index.data_ptr<int32_t>(),
      rank_start_index.data_ptr<int32_t>(),
      rank_end_index.data_ptr<int32_t>(),
      stream);
  return std::tuple(
      M_this_ep,
      m_pad_holder,
      gather_a_index,
      scatter_d_index,
      expert_index,
      rank_start_index,
      rank_end_index);
}

class GemmGroupedV2AGScatterOp::GemmGroupedV2AGScatterOpImpl {
 private:
  std::shared_ptr<Group> tp_group;
  const int rank;
  const int world_size;
  const int ep_size;
  const int ffn_tp_size;
  const int ep_rank;
  const int ffn_tp_rank;
  const int max_ntokens;
  const int N;
  const int hidden;
  const int nexperts;
  const int topk;
  at::ScalarType input_dtype;
  at::ScalarType output_dtype;
  const int32_t ep_nexperts;
  const int32_t ep_start;

  torch::Tensor workspace_buffer;

  c10::cuda::CUDAStream cp_stream;
  cudaEvent_t ready_event;
  cudaEvent_t all_gather_event;

  AllGatherOp ag_op;
  GroupBarrier group_barrier;

 private:
  c10::cuda::CUDAStream
  create_cp_stream() const {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, CU_STREAM_NON_BLOCKING));
    return at::cuda::getStreamFromExternal(stream, at::cuda::current_device());
  }

  void
  _ensure_topo_initialized() {
    if (!topo_utils::is_topo_initialized()) {
      topo_utils::initialize_topo(this->tp_group.get());
    }
  }

  AllGatherOption
  materialize(const AllGatherOptionWithOptional opt, bool with_input_scale) {
    return AllGatherOption{
        .input_buffer_copied = opt.input_buffer_copied.value_or(false),
        .use_cuda_core_local = opt.use_cuda_core_local.value_or(with_input_scale),
        .use_cuda_core_ag = opt.use_cuda_core_ag.value_or(with_input_scale),
        .fuse_sync = opt.fuse_sync.value_or(with_input_scale),
        .use_read = opt.use_read.value_or(false),
        .mode = opt.mode.value_or(get_default_ag_ring_mode()),
    };
  }

 public:
  GemmGroupedV2AGScatterOpImpl(
      std::shared_ptr<Group> tp_group,
      int ep_size,
      int max_ntokens,
      int ffn_hidden,  // before TP shard
      int hidden,
      int nexperts,
      int topk,
      at::ScalarType input_dtype,
      at::ScalarType output_dtype)
      : tp_group(tp_group),
        world_size(tp_group->get_size()),
        ep_size(ep_size),
        ffn_tp_size(world_size / ep_size),
        rank(tp_group->get_rank()),
        ffn_tp_rank(rank % ffn_tp_size),
        ep_rank(rank / ffn_tp_size),
        max_ntokens(max_ntokens),
        N(ffn_hidden / ffn_tp_size),
        hidden(hidden),
        nexperts(nexperts),
        topk(topk),
        input_dtype(input_dtype),
        output_dtype(output_dtype),
        ep_nexperts(nexperts / ep_size),
        ep_start(this->ep_nexperts * ep_rank),
        cp_stream(create_cp_stream()),
        ag_op(
            this->tp_group,
            1,  // TODO(houqi.1993) only support 1 nodes
            max_ntokens,
            hidden,
            input_dtype),
        group_barrier(this->tp_group, this->tp_group->get_size() > 8) {
    _ensure_topo_initialized();
    CHECK_DIV(nexperts, ep_size);
    CHECK_DIV(ffn_hidden, ffn_tp_size);
    CUDA_CHECK(cudaEventCreateWithFlags(&this->ready_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&this->all_gather_event, cudaEventDisableTiming));
  }

  ~GemmGroupedV2AGScatterOpImpl() {
    CUDA_CHECK(cudaEventDestroy(this->all_gather_event));
    CUDA_CHECK(cudaEventDestroy(this->ready_event));
    CUDA_CHECK(cudaStreamDestroy(this->cp_stream));
  }

 protected:
  auto
  get_gemm_meta(bool fast_accum) const {
    auto arch = get_arch();
    auto gemm_layout = _RCR{};  // TODO(houqi.1993) only RCR supported
    auto input_dtype = from_torch_dtype(this->input_dtype);
    auto output_dtype = from_torch_dtype(this->output_dtype);
    auto dt_conf = make_gemm_dtype_config(input_dtype, input_dtype, output_dtype, output_dtype);
    auto v2_meta = make_gemm_v2_meta(fast_accum && dt_conf.is_input_fp8());
    auto meta =
        make_gemm_meta(dt_conf, arch, _AGScatter{}, gemm_layout, _GemmGroupedV2{}, v2_meta);
    return meta;
  }

  auto
  get_rt_conf() const {
    return make_runtime_config(512, this->N, this->hidden);
  }

  std::vector<torch::Tensor>
  forward_impl(
      torch::Tensor inputs_shard,
      std::vector<torch::Tensor> weights,
      torch::Tensor splits_gpu,
      torch::Tensor scatter_index,
      c10::optional<std::vector<torch::Tensor>> input_scales,
      c10::optional<std::vector<torch::Tensor>> weight_scales,
      c10::optional<std::vector<torch::Tensor>> output_scales,
      c10::optional<std::vector<torch::Tensor>> outputs_buf,
      c10::optional<torch::Tensor> allgather_output,
      bool fast_accum,
      int sm_margin,
      const AllGatherOption &opt,
      c10::optional<UnifiedGemmHParams> const &hparams) {
    FLUX_CHECK(
        inputs_shard.scalar_type() == at::ScalarType::BFloat16 ||
        inputs_shard.scalar_type() == at::ScalarType::Half ||
        inputs_shard.scalar_type() == at::ScalarType::Float8_e4m3fn ||
        inputs_shard.scalar_type() == at::ScalarType::Float8_e5m2)
        << inputs_shard.scalar_type();
    // Step 0. do some shape checks
    int const N = this->N;
    int const K = hidden;
    // doing shape CHECK
    CHECK_INPUT(inputs_shard, this->input_dtype);
    CHECK_NDIM(inputs_shard, 2);
    const int tokens_per_rank = inputs_shard.size(0);
    CHECK_2D(inputs_shard, tokens_per_rank, K);

    const int ntokens = tokens_per_rank * world_size;

    const std::size_t num_weights_group = weights.size();
    for (std::size_t i = 0; i < num_weights_group; ++i) {
      CHECK_INPUT(weights[i], this->input_dtype);
      CHECK_3D(weights[i], this->ep_nexperts, N, K);
    }

    CHECK_INPUT(splits_gpu, torch::kInt32);
    CHECK_NDIM(splits_gpu, 1);
    FLUX_CHECK_LE(this->nexperts, splits_gpu.size(0));

    CHECK_INPUT(scatter_index, torch::kInt32);
    CHECK_2D(scatter_index, ntokens, this->topk);

    FLUX_CHECK(!input_scales.has_value());
    FLUX_CHECK(!weight_scales.has_value());
    if (output_scales.has_value()) {
      TORCH_CHECK_EQ(output_scales->size(), num_weights_group);
      for (std::size_t i = 0; i < num_weights_group; ++i) {
        CHECK_INPUT(output_scales->at(i), torch::kFloat32);
        CHECK_1D(output_scales->at(i), this->ep_nexperts);
      }
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // Step 1: get op. and prepare op buffers
    auto meta = this->get_gemm_meta(fast_accum);
    auto rt_conf = this->get_rt_conf();
    OpRegistry::OpPtr op;
    if (hparams.has_value()) {
      op = OpRegistry::instance().get_op(meta, hparams.value());
    } else {
      op = OpRegistry::instance().get_op(meta, rt_conf);
    }
    const auto tile_shape = op->get_runtime_gemm_hparams().tile_shape();
    const int tile_M = cute::get<0>(tile_shape);
    const int tile_N = cute::get<1>(tile_shape);

    // Step 2: Launch AG comm as early as possible
    bool is_s8_gemm = is_s8_torch_dtype(inputs_shard.scalar_type());
    FLUX_CHECK(!is_s8_gemm) << "not support INT8 MOE AG+Scatter yet";

    CUDA_CHECK(cudaEventRecord(this->ready_event, stream));
    CUDA_CHECK(cudaStreamWaitEvent(this->cp_stream, this->ready_event));
    ag_op.run(inputs_shard, c10::nullopt, opt, this->cp_stream);

    // Step 3: helper kernels. for preparing gather_index & sort tokens & outputs
    int topk = this->topk;
    int ep_nexperts = this->ep_nexperts;
    int nexperts = this->nexperts;
    int ep_start = this->ep_start;
    // should be M_this_ep, but never mind gather_index takes little memory
    auto opt_i32d = torch::TensorOptions(torch::kCUDA)
                        .dtype(at::ScalarType::Int)
                        .device_index(at::cuda::current_device());
    auto opt_i32h =
        torch::TensorOptions(torch::kCPU).dtype(at::ScalarType::Int).pinned_memory(true);
    torch::Tensor gather_index =
        empty_with_uninitialized_data(std::vector<int64_t>{ntokens * topk}, opt_i32d);
    torch::Tensor sorted_gather_index =
        empty_with_uninitialized_data(std::vector<int64_t>{ntokens * topk}, opt_i32d);
    torch::Tensor sorted_scatter_index =
        empty_with_uninitialized_data(std::vector<int64_t>{ntokens * topk}, opt_i32d);
    torch::Tensor M_this_ep_holder =
        empty_with_uninitialized_data(std::vector<int64_t>{1}, opt_i32h);
    torch::Tensor sorted_splits =
        empty_with_uninitialized_data(std::vector<int64_t>{ep_nexperts * world_size}, opt_i32d);
    torch::Tensor sorted_splits_cumsum =
        empty_with_uninitialized_data(std::vector<int64_t>{ep_nexperts * world_size}, opt_i32d);
    calc_gather_index_impl(
        nexperts,
        ntokens,
        topk,
        ep_start,
        ep_start + ep_nexperts,
        splits_gpu.const_data_ptr<int32_t>(),
        scatter_index.const_data_ptr<int32_t>(),
        gather_index.data_ptr<int32_t>(),
        M_this_ep_holder.data_ptr<int>(),
        stream);

    AGScatterSortOpArgumentsV2 moe_sort_args = {
        rank,
        world_size,
        ntokens,
        ep_nexperts,
        splits_gpu.const_data_ptr<int32_t>() + ep_start,
        gather_index.const_data_ptr<int32_t>(),
        sorted_splits.data_ptr<int32_t>(),
        sorted_splits_cumsum.data_ptr<int32_t>(),
        sorted_scatter_index.data_ptr<int32_t>(),
        sorted_gather_index.data_ptr<int32_t>(),
    };
    ag_scatter_sort_impl_v2(moe_sort_args, stream);

    sort_scatter_index_to_per_expert(
        sorted_scatter_index.data_ptr<int>(),
        splits_gpu.data_ptr<int>(),
        ep_start,
        ep_nexperts,
        stream);

    int M_this_ep = scatter_index.numel();  // for EP=1, M_this_ep is always M_full
    if (ep_nexperts != nexperts) {
      CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)stream));
      M_this_ep = *M_this_ep_holder.const_data_ptr<int32_t>();
    }

    int num_problem_schedules = ep_nexperts * world_size * num_weights_group;
    torch::Tensor problem_schedules_gpu = empty_with_uninitialized_data(
        std::vector<int64_t>{num_problem_schedules * (int64_t)sizeof(ProblemSchedule)},
        torch::TensorOptions(torch::kInt8).device(torch::kCUDA));
    // Step 4: prepare GEMM args
    torch::Tensor barrier = ag_op.local_barrier_buffer();
    torch::Tensor input_buffer = ag_op.local_input_buffer().slice(0, 0, ntokens);

    // shapes check
    std::vector<torch::Tensor> outputs = outputs_buf.value_or([&]() {
      std::vector<torch::Tensor> outputs;
      for (std::size_t i = 0; i < num_weights_group; ++i) {
        outputs.emplace_back(empty_with_uninitialized_data(
            std::vector<int64_t>{M_this_ep, N}, inputs_shard.options()));
      };
      return outputs;
    }());

    TORCH_CHECK_EQ(outputs.size(), num_weights_group);
    for (std::size_t i = 0; i < num_weights_group; ++i) {
      CHECK_INPUT(outputs[i], this->output_dtype);
      CHECK_2D(outputs[i], M_this_ep, N);
    }

    // set the output type here accordlingly
    auto args = GemmGroupedV2AGScatterArguments{
        .rank = rank,
        .world_size = world_size,
        .sm_margin = sm_margin,
        .num_groups = (int)num_weights_group,
        .ep_start = ep_start,
        .ep_nexperts = ep_nexperts,
        .input = input_buffer.data_ptr(),
        .M_this_ep = M_this_ep,
        .N = N,
        .K = K,
        .splits = splits_gpu.data_ptr<int>(),
        .gather_A = sorted_gather_index.data_ptr<int32_t>(),
        .scatter_D = sorted_scatter_index.data_ptr<int32_t>(),
        .problem_schedules = problem_schedules_gpu.data_ptr(),
        .num_problem_schedules = num_problem_schedules,
        .accum_per_rank_ptr = sorted_splits_cumsum.data_ptr<int32_t>(),
        .tile_size_m = tile_M,
        .tile_size_n = tile_N,
        .barrier_ptr = barrier.data_ptr<int32_t>()};
    for (int gid = 0; gid < num_weights_group; gid++) {
      args.weight[gid] = weights[gid].data_ptr();
      args.output[gid] = outputs[gid].data_ptr();
      args.scaleD[gid] =
          output_scales.has_value() ? output_scales->at(gid).data_ptr<float>() : nullptr;
    }

    CUDA_CHECK(cudaStreamWaitEvent(stream, ag_op.get_local_prepare_event()));
    if (M_this_ep > 0) {
      int64_t workspace_size = op->get_workspace_size(args);
      lazy_init_buffer_tensor(&this->workspace_buffer, workspace_size);

      // Step 5: launch GEMM
      op->run(args, workspace_size ? this->workspace_buffer.data_ptr() : nullptr, stream);
    }
    CUDA_CHECK(cudaStreamWaitEvent(stream, this->all_gather_event));

    if (allgather_output.has_value()) {
      CHECK_INPUT(allgather_output.value(), this->input_dtype);
      CHECK_2D(allgather_output.value(), ntokens, K);
      CUDA_CHECK(cudaMemcpyAsync(
          allgather_output->data_ptr(),
          input_buffer.const_data_ptr(),
          allgather_output->nbytes(),
          cudaMemcpyDeviceToDevice,
          stream));
    }

    return outputs;
  }

#if defined(FLUX_WITH_TRITON_AOT)
  using FuncType = decltype(moe_ag_scatter_grouped_gemm_s8_ex);
  moe_ag_scatter_grouped_gemm_kernel__triton_algo_info_t
  get_default_triton_algo_info(at::ScalarType input_dtype, bool has_bias) {
    moe_ag_scatter_grouped_gemm_kernel__triton_algo_info_t algo_info;
    bool is_s8_gemm = is_s8_torch_dtype(input_dtype);
    if (is_s8_gemm) {
      algo_info = moe_ag_scatter_grouped_gemm_kernel__triton_algo_info_t{
          .WITH_BIAS = has_bias,
          .BLOCK_SIZE_M = 64,
          .BLOCK_SIZE_N = 128,
          .BLOCK_SIZE_K = 64,
          .GROUP_SIZE_M = 4,
          .num_warps = 4,
          .num_stages = 4};
    } else if (input_dtype == torch::kHalf) {
      algo_info = moe_ag_scatter_grouped_gemm_kernel__triton_algo_info_t{
          .WITH_BIAS = has_bias,
          .BLOCK_SIZE_M = 128,
          .BLOCK_SIZE_N = 128,
          .BLOCK_SIZE_K = 32,
          .GROUP_SIZE_M = 8,
          .num_warps = 4,
          .num_stages = 3};
    } else if (input_dtype == torch::kBFloat16) {
      algo_info = moe_ag_scatter_grouped_gemm_kernel__triton_algo_info_t{
          .WITH_BIAS = has_bias,
          .BLOCK_SIZE_M = 128,
          .BLOCK_SIZE_N = 128,
          .BLOCK_SIZE_K = 32,
          .GROUP_SIZE_M = 8,
          .num_warps = 4,
          .num_stages = 3};
    } else {
      FLUX_CHECK(false) << "unsupported dtype " << input_dtype;
    }
    return algo_info;
  }
  FuncType *
  get_triton_aot_func(at::ScalarType input_dtype) {
    if (input_dtype == torch::kInt8) {
      return moe_ag_scatter_grouped_gemm_s8_ex;
    } else if (input_dtype == torch::kHalf) {
      return moe_ag_scatter_grouped_gemm_fp16_ex;
    } else if (input_dtype == torch::kBFloat16) {
      return moe_ag_scatter_grouped_gemm_bf16_ex;
    } else {
      FLUX_CHECK(false) << "unsupported dtype " << input_dtype;
      return nullptr;
    }
  }
#endif
  std::vector<torch::Tensor>
  forward_triton_aot_impl(
      torch::Tensor inputs_shard,
      std::vector<torch::Tensor> weights,
      torch::Tensor splits_gpu,
      torch::Tensor scatter_index,
      c10::optional<std::vector<torch::Tensor>> biases,
      c10::optional<std::vector<torch::Tensor>> input_scales,
      c10::optional<std::vector<torch::Tensor>> weight_scales,
      c10::optional<std::vector<torch::Tensor>> output_scales,
      c10::optional<std::vector<torch::Tensor>> outputs_bufs,
      c10::optional<torch::Tensor> allgather_output,
      bool fast_accum,
      int sm_margin,
      const AllGatherOption &opt) {
#if defined(FLUX_WITH_TRITON_AOT)
    FLUX_CHECK(weights.size() == 1);
    bool is_fp8_gemm = c10::isFloat8Type(inputs_shard.scalar_type());
    bool is_s8_gemm = is_s8_torch_dtype(inputs_shard.scalar_type());
    FLUX_CHECK(!is_fp8_gemm) << "not support INT8 MOE AG+Scatter yet";
    // Step 0. do some shape checks
    int const N = this->N;
    int const K = this->hidden;
    // doing shape CHECK
    CHECK_INPUT(inputs_shard, this->input_dtype);
    CHECK_NDIM(inputs_shard, 2);
    const int tokens_per_rank = inputs_shard.size(0);
    CHECK_2D(inputs_shard, tokens_per_rank, K);

    const int ntokens = tokens_per_rank * world_size;

    const std::size_t num_weights_group = weights.size();
    for (std::size_t i = 0; i < num_weights_group; ++i) {
      CHECK_INPUT(weights[i], this->input_dtype);
      CHECK_3D(weights[i], this->ep_nexperts, N, K);  // RCR layout
    }

    CHECK_INPUT(splits_gpu, torch::kInt32);
    CHECK_NDIM(splits_gpu, 1);
    FLUX_CHECK_LE(this->nexperts, splits_gpu.size(0));

    CHECK_INPUT(scatter_index, torch::kInt32);
    CHECK_2D(scatter_index, ntokens, this->topk);

    if (is_s8_gemm) {
      FLUX_CHECK(biases.has_value());
      FLUX_CHECK(input_scales.has_value());
      FLUX_CHECK(weight_scales.has_value());
    } else {
      FLUX_CHECK(!biases.has_value());
      FLUX_CHECK(!input_scales.has_value());
      FLUX_CHECK(!weight_scales.has_value());
    }
    if (biases.has_value()) {
      FLUX_CHECK_EQ(biases->size(), num_weights_group);
      for (int i = 0; i < num_weights_group; i++) {
        CHECK_INPUT(biases->at(i), torch::kFloat32);
        CHECK_3D(biases->at(i), this->ep_nexperts, 1, N);
      }
    }
    if (input_scales.has_value()) {
      FLUX_CHECK_EQ(input_scales->size(), num_weights_group);
      for (int i = 0; i < num_weights_group; i++) {
        CHECK_INPUT(input_scales->at(i), torch::kFloat32);
        CHECK_1D(input_scales->at(i), tokens_per_rank);
      }
    }
    if (weight_scales.has_value()) {
      for (int i = 0; i < num_weights_group; i++) {
        CHECK_INPUT(weight_scales->at(i), torch::kFloat32);
        CHECK_3D(weight_scales->at(i), this->ep_nexperts, 1, N);
      }
    }
    if (output_scales.has_value()) {
      TORCH_CHECK_EQ(output_scales->size(), num_weights_group);
      for (std::size_t i = 0; i < num_weights_group; ++i) {
        CHECK_INPUT(output_scales->at(i), torch::kFloat32);
        CHECK_1D(output_scales->at(i), this->ep_nexperts);
      }
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // Step 2: Launch AG comm as early as possible
    bool allgather_input_scale = input_scales.has_value() && is_s8_gemm;

    CUDA_CHECK(cudaEventRecord(this->ready_event, stream));
    CUDA_CHECK(cudaStreamWaitEvent(this->cp_stream, this->ready_event));
    ag_op.run(
        inputs_shard,
        allgather_input_scale ? c10::optional<torch::Tensor>{input_scales->at(0)} : c10::nullopt,
        opt,
        this->cp_stream);

    // Step 3: helper kernels. for preparing gather_index & sort tokens & outputs
    // should be M_this_ep, but never mind gather_index takes little memory
    int M_this_ep;
    torch::Tensor m_pad_holder;
    torch::Tensor gather_a_index;
    torch::Tensor scatter_d_index;
    torch::Tensor expert_index;
    torch::Tensor rank_start_index;
    torch::Tensor rank_end_index;

    FuncType *moe_ag_scatter_grouped_gemm = get_triton_aot_func(inputs_shard.scalar_type());
    auto algo_info = get_default_triton_algo_info(inputs_shard.scalar_type(), biases.has_value());
    std::tie(
        M_this_ep,
        m_pad_holder,
        gather_a_index,
        scatter_d_index,
        expert_index,
        rank_start_index,
        rank_end_index) =
        prepare_moe_ag_scatter_args(
            splits_gpu,
            scatter_index,
            ntokens,
            topk,
            1,
            ep_start,
            ep_nexperts,
            rank,
            world_size,
            algo_info.BLOCK_SIZE_M,
            (intptr_t)stream);

    // Step 4: prepare GEMM args
    torch::Tensor barrier = ag_op.local_barrier_buffer();
    torch::Tensor input_buffer = ag_op.local_input_buffer().slice(0, 0, ntokens);
    c10::optional<torch::Tensor> input_scale_tensor =
        allgather_input_scale
            ? c10::optional<torch::Tensor>{ag_op.local_input_scale_buffer().slice(0, 0, ntokens)}
            : c10::nullopt;

    // shapes check
    std::vector<torch::Tensor> outputs = outputs_bufs.value_or([&]() {
      std::vector<torch::Tensor> outputs;
      auto option = at::TensorOptions(this->output_dtype).device(torch::kCUDA);
      for (std::size_t i = 0; i < num_weights_group; ++i) {
        outputs.emplace_back(
            empty_with_uninitialized_data(std::vector<int64_t>{M_this_ep, N}, option));
      };
      return outputs;
    }());

    TORCH_CHECK_EQ(outputs.size(), num_weights_group);
    for (std::size_t i = 0; i < num_weights_group; ++i) {
      CHECK_INPUT(outputs[i], this->output_dtype);
      CHECK_2D(outputs[i], M_this_ep, N);
    }

    FLUX_CHECK(input_scales.has_value());
    FLUX_CHECK(weight_scales.has_value());

    if (M_this_ep > 0) {
      CUDA_CHECK(cudaStreamWaitEvent(stream, ag_op.get_local_prepare_event()));
      auto rtn = moe_ag_scatter_grouped_gemm(
          (CUstream)stream,
          (CUdeviceptr)input_buffer.data_ptr(),
          (CUdeviceptr)weights[0].data_ptr(),
          (CUdeviceptr)outputs[0].data_ptr(),
          (CUdeviceptr)(biases.has_value() ? biases->at(0).data_ptr() : nullptr),  // bias
          (CUdeviceptr)input_scale_tensor->data_ptr(),                             // input_scale
          (CUdeviceptr)(weight_scales.has_value() ? weight_scales->at(0).data_ptr()
                                                  : nullptr),  // weight_scale
          (CUdeviceptr)(output_scales.has_value() ? output_scales->at(0).data_ptr()
                                                  : nullptr),  // output_scale
          (CUdeviceptr)gather_a_index.data_ptr(),
          (CUdeviceptr)scatter_d_index.data_ptr(),
          (CUdeviceptr)expert_index.data_ptr(),
          (CUdeviceptr)rank_start_index.data_ptr(),
          (CUdeviceptr)rank_end_index.data_ptr(),
          (CUdeviceptr)m_pad_holder.data_ptr(),
          N,
          K,
          ep_nexperts,
          M_this_ep,
          input_buffer.stride(0),
          input_buffer.stride(1),
          weights[0].stride(0),
          weights[0].stride(2),
          weights[0].stride(1),  // transpose_weight
          outputs[0].stride(0),
          outputs[0].stride(1),
          (CUdeviceptr)barrier.data_ptr(),
          algo_info);
      CU_CHECK(rtn);
    }

    CUDA_CHECK(cudaStreamWaitEvent(stream, this->all_gather_event));

    if (allgather_output.has_value()) {
      CHECK_INPUT(allgather_output.value(), this->input_dtype);
      CHECK_2D(allgather_output.value(), ntokens, K);
      CUDA_CHECK(cudaMemcpyAsync(
          allgather_output->data_ptr(),
          input_buffer.const_data_ptr(),
          allgather_output->nbytes(),
          cudaMemcpyDeviceToDevice,
          stream));
    }

    return outputs;
#else
    FLUX_CHECK(false) << "please compile with --triton-aot option.";
#endif
  }

 public:
  std::vector<torch::Tensor>
  forward_multiple_weights(
      torch::Tensor inputs_shard,
      std::vector<torch::Tensor> weights,
      torch::Tensor splits_gpu,
      torch::Tensor scatter_index,
      c10::optional<std::vector<torch::Tensor>> bias,
      c10::optional<std::vector<torch::Tensor>> input_scale,
      c10::optional<std::vector<torch::Tensor>> weight_scale,
      c10::optional<std::vector<torch::Tensor>> output_scale,
      c10::optional<std::vector<torch::Tensor>> outputs_buf,
      c10::optional<torch::Tensor> allgather_output,
      bool fast_accum,
      int sm_margin,
      AllGatherOptionWithOptional ag_option) {
    bool is_s8_gemm = inputs_shard.scalar_type() == torch::kInt8;
    AllGatherOption option = materialize(ag_option, is_s8_gemm && input_scale.has_value());
    return forward_impl(
        std::move(inputs_shard),
        std::move(weights),
        std::move(splits_gpu),
        std::move(scatter_index),
        std::move(input_scale),
        std::move(weight_scale),
        std::move(output_scale),
        std::move(outputs_buf),
        std::move(allgather_output),
        fast_accum,
        sm_margin,
        option,
        c10::nullopt);
  }

  void
  clear_buffers() {
    // do nothing. just keep pace with v3 code
  }

  torch::Tensor
  forward(
      torch::Tensor inputs_shard,
      torch::Tensor weights,
      torch::Tensor splits_gpu,
      torch::Tensor scatter_index,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      c10::optional<torch::Tensor> outputs_buf,
      c10::optional<torch::Tensor> allgather_output,
      bool fast_accum,
      int sm_margin,
      AllGatherOptionWithOptional ag_option) {
    if (inputs_shard.scalar_type() == torch::kInt8) {
      return forward_triton_aot(
          inputs_shard,
          weights,
          splits_gpu,
          scatter_index,
          bias,
          input_scale,
          weight_scale,
          output_scale,
          outputs_buf,
          allgather_output,
          fast_accum,
          sm_margin,
          ag_option);
    }
    FLUX_CHECK(!bias.has_value());
    bool is_s8_gemm = inputs_shard.scalar_type() == torch::kInt8;
    AllGatherOption option = materialize(ag_option, is_s8_gemm && input_scale.has_value());
    auto outputs = forward_impl(
        std::move(inputs_shard),
        {weights},
        std::move(splits_gpu),
        std::move(scatter_index),
        as_optional_vec(input_scale),
        as_optional_vec(weight_scale),
        as_optional_vec(output_scale),
        as_optional_vec(outputs_buf),
        std::move(allgather_output),
        fast_accum,
        sm_margin,
        option,
        c10::nullopt);
    return outputs[0];
  }

  torch::Tensor
  forward_triton_aot(
      torch::Tensor inputs_shard,
      torch::Tensor weights,
      torch::Tensor splits_gpu,
      torch::Tensor scatter_index,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      c10::optional<torch::Tensor> outputs_buf,
      c10::optional<torch::Tensor> allgather_output,
      bool fast_accum,
      int sm_margin,
      AllGatherOptionWithOptional ag_option) {
    bool is_s8_gemm = inputs_shard.scalar_type() == torch::kInt8;
    AllGatherOption option = materialize(ag_option, is_s8_gemm && input_scale.has_value());
    auto outputs = forward_triton_aot_impl(
        std::move(inputs_shard),
        {weights},
        std::move(splits_gpu),
        std::move(scatter_index),
        as_optional_vec(bias),
        as_optional_vec(input_scale),
        as_optional_vec(weight_scale),
        as_optional_vec(output_scale),
        as_optional_vec(outputs_buf),
        std::move(allgather_output),
        fast_accum,
        sm_margin,
        option);
    return outputs[0];
  }

  std::vector<torch::Tensor>
  profiling(
      torch::Tensor inputs_shard,
      std::vector<torch::Tensor> weights,
      torch::Tensor splits_gpu,
      torch::Tensor scatter_index,
      c10::optional<std::vector<torch::Tensor>> input_scale,
      c10::optional<std::vector<torch::Tensor>> weight_scale,
      c10::optional<std::vector<torch::Tensor>> output_scale,
      c10::optional<std::vector<torch::Tensor>> outputs_buf,
      c10::optional<torch::Tensor> allgather_output,
      bool fast_accum,
      int sm_margin,
      AllGatherOptionWithOptional ag_option,
      c10::intrusive_ptr<ProfilingContext> opt_ctx) {
    bool is_s8_gemm = inputs_shard.scalar_type() == torch::kInt8;
    AllGatherOption option = materialize(ag_option, is_s8_gemm && input_scale.has_value());
    auto meta = unify_type(this->get_gemm_meta(fast_accum));
    auto rt_conf = this->get_rt_conf();
    ProfilingContext tmp_ctx("__tmp__");
    ProfilingContext *ctx = opt_ctx == nullptr ? &tmp_ctx : opt_ctx.get();

    auto elapsed_tensor = torch::empty({}, inputs_shard.options().dtype(c10::ScalarType::Float));
    auto reduced_elapsed_tensor = elapsed_tensor.clone();

    OpRegistry::instance().visit_hparams(
        [&](UnifiedGemmHParams const &hparams) {
          // filter non-consistent hparams
          constexpr int warm_iters = 5;
          constexpr int iters = 10;
          float total_elapsed = 0;

          auto stream = c10::cuda::getCurrentCUDAStream();
          group_barrier.barrier_all(stream);
          c10::cuda::stream_synchronize(stream);
          auto cp_hparams = hparams;
          for (int iter = 0; iter < warm_iters + iters; ++iter) {
            GpuTimer timer;
            timer.start(stream);
            auto output [[maybe_unused]] = this->forward_impl(
                inputs_shard,
                weights,
                splits_gpu,
                scatter_index,
                input_scale,
                weight_scale,
                output_scale,
                outputs_buf,
                allgather_output,
                fast_accum,
                sm_margin,
                option,
                cp_hparams);
            timer.stop();
            if (iter >= warm_iters) {
              total_elapsed += timer.elapsed_millis();
            }
          }

          // Avoid GPU frequency adjustment
          group_barrier.barrier_all(stream);
          c10::cuda::stream_synchronize(stream);
          sleep(1);
          float avg_elapsed = int(total_elapsed / iters * 1000) / 1000.0;
          float reduce_elapsed = all_reduce_max_float(this->tp_group.get(), avg_elapsed);
          ctx->add(meta, rt_conf, hparams, reduce_elapsed);
        },
        meta);

    auto best_hparams = ctx->record_best(meta, rt_conf);
    return this->forward_impl(
        std::move(inputs_shard),
        std::move(weights),
        std::move(splits_gpu),
        std::move(scatter_index),
        std::move(input_scale),
        std::move(weight_scale),
        std::move(output_scale),
        std::move(outputs_buf),
        std::move(allgather_output),
        fast_accum,
        sm_margin,
        option,
        std::move(best_hparams));
  }
};

GemmGroupedV2AGScatterOp::GemmGroupedV2AGScatterOp(
    std::shared_ptr<Group> tp_group,
    int ep_size,
    int max_ntokens,
    int ffn_hidden,  // before TP shard
    int hidden,
    int num_experts,
    int topk,
    at::ScalarType input_dtype,
    at::ScalarType output_dtype)
    : impl_(new GemmGroupedV2AGScatterOpImpl(
          tp_group,
          ep_size,
          max_ntokens,
          ffn_hidden,  // before TP shard
          hidden,
          num_experts,
          topk,
          input_dtype,
          output_dtype)) {}
GemmGroupedV2AGScatterOp::~GemmGroupedV2AGScatterOp() { delete impl_; }

void
GemmGroupedV2AGScatterOp::clear_buffers() {
  FLUX_CHECK(impl_ != nullptr) << "GemmGroupedV2AGScatterOp is not initialized";
  impl_->clear_buffers();
}
torch::Tensor
GemmGroupedV2AGScatterOp::forward(
    torch::Tensor inputs_shard,
    torch::Tensor weights,
    torch::Tensor splits_gpu,
    torch::Tensor scatter_index,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale,
    c10::optional<torch::Tensor> output_scale,
    c10::optional<torch::Tensor> outputs_buf,
    c10::optional<torch::Tensor> allgather_output,
    bool fast_accum,
    int sm_margin,
    AllGatherOptionWithOptional ag_option) {
  FLUX_CHECK(impl_ != nullptr) << "GemmGroupedV2AGScatterOp is not initialized";
  return impl_->forward(
      std::move(inputs_shard),
      std::move(weights),
      std::move(splits_gpu),
      std::move(scatter_index),
      std::move(bias),
      std::move(input_scale),
      std::move(weight_scale),
      std::move(output_scale),
      std::move(outputs_buf),
      std::move(allgather_output),
      fast_accum,
      sm_margin,
      ag_option);
}
torch::Tensor
GemmGroupedV2AGScatterOp::forward_triton_aot(
    torch::Tensor inputs_shard,
    torch::Tensor weights,
    torch::Tensor splits_gpu,
    torch::Tensor scatter_index,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale,
    c10::optional<torch::Tensor> output_scale,
    c10::optional<torch::Tensor> outputs_buf,
    c10::optional<torch::Tensor> allgather_output,
    bool fast_accum,
    int sm_margin,
    AllGatherOptionWithOptional ag_option) {
  FLUX_CHECK(impl_ != nullptr) << "GemmGroupedV2AGScatterOp is not initialized";
  return impl_->forward_triton_aot(
      std::move(inputs_shard),
      std::move(weights),
      std::move(splits_gpu),
      std::move(scatter_index),
      std::move(bias),
      std::move(input_scale),
      std::move(weight_scale),
      std::move(output_scale),
      std::move(outputs_buf),
      std::move(allgather_output),
      fast_accum,
      sm_margin,
      ag_option);
}
std::vector<torch::Tensor>
GemmGroupedV2AGScatterOp::forward_multiple_weights(
    torch::Tensor inputs_shard,
    std::vector<torch::Tensor> weights,
    torch::Tensor splits_gpu,
    torch::Tensor scatter_index,
    c10::optional<std::vector<torch::Tensor>> bias,
    c10::optional<std::vector<torch::Tensor>> input_scale,
    c10::optional<std::vector<torch::Tensor>> weight_scale,
    c10::optional<std::vector<torch::Tensor>> output_scale,
    c10::optional<std::vector<torch::Tensor>> outputs_buf,
    c10::optional<torch::Tensor> allgather_output,
    bool fast_accum,
    int sm_margin,
    AllGatherOptionWithOptional ag_option) {
  FLUX_CHECK(impl_ != nullptr) << "GemmGroupedV2AGScatterOp is not initialized";
  return impl_->forward_multiple_weights(
      std::move(inputs_shard),
      std::move(weights),
      std::move(splits_gpu),
      std::move(scatter_index),
      std::move(bias),
      std::move(input_scale),
      std::move(weight_scale),
      std::move(output_scale),
      std::move(outputs_buf),
      std::move(allgather_output),
      fast_accum,
      sm_margin,
      ag_option);
}
std::vector<torch::Tensor>
GemmGroupedV2AGScatterOp::profiling(
    torch::Tensor inputs_shard,
    std::vector<torch::Tensor> weights,
    torch::Tensor splits_gpu,
    torch::Tensor scatter_index,
    c10::optional<std::vector<torch::Tensor>> input_scale,
    c10::optional<std::vector<torch::Tensor>> weight_scale,
    c10::optional<std::vector<torch::Tensor>> output_scale,
    c10::optional<std::vector<torch::Tensor>> outputs_buf,
    c10::optional<torch::Tensor> allgather_output,
    bool fast_accum,
    int sm_margin,
    AllGatherOptionWithOptional ag_option,
    c10::intrusive_ptr<ProfilingContext> opt_ctx) {
  FLUX_CHECK(impl_ != nullptr) << "GemmGroupedV2AGScatterOp is not initialized";
  return impl_->profiling(
      std::move(inputs_shard),
      std::move(weights),
      std::move(splits_gpu),
      std::move(scatter_index),
      std::move(input_scale),
      std::move(weight_scale),
      std::move(output_scale),
      std::move(outputs_buf),
      std::move(allgather_output),
      fast_accum,
      sm_margin,
      ag_option,
      std::move(opt_ctx));
}

}  // namespace bytedance::flux::ths_op
