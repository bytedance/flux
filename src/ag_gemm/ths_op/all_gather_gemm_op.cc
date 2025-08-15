//===- all_gather_gemm_op.cc -------------------------------------- C++ ---===//
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

#include "ag_gemm/ths_op/all_gather_gemm_op.h"

#include "coll/ths_op/all_gather_types.h"
#include "coll/ths_op/all_gather_op.h"
#include "flux/args/ag_gemm.h"
#include "ag_gemm/ths_op/gemm_with_barrier.h"
#include "flux/cuda/cuda_common.h"
#include "flux/flux.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_meta.h"
#include "flux/op_registry.h"
#include "flux/runtime_config.h"
#include "flux/ths_op/flux_shm.h"
#include "flux/ths_op/ths_op.h"
#include "flux/ths_op/util.h"
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/List.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/tensor.h>
#include <ATen/ops/zeros.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/Optional.h>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <torch/all.h>

namespace bytedance {
namespace flux {
namespace ths_op {
using torch::Tensor;

namespace {
inline void *
ptr_offset(void *ptr, ptrdiff_t offset) {
  return static_cast<char *>(ptr) + offset;
}
}  // namespace

/// All Gather GEMM Kernel OP
class AllGatherGemmOp::AllGatherGemmOpImpl {
 private:
  using FlagType = int32_t;

 private:
  std::shared_ptr<Group> tp_group;
  int world_size;
  int nnodes;
  cudaStream_t cp_stream;
  cudaEvent_t cp_event;
  cudaEvent_t ready_event;
  cudaEvent_t all_gather_event;

  GemmWithBarirer gemm_op;
  AllGatherOp ag_op;

  bool use_pdl;  // sm90 feature

 private:
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
  AllGatherGemmOpImpl(
      std::shared_ptr<Group> tp_group,
      int32_t nnodes,
      int32_t full_m,
      int32_t n_dim,
      int32_t k_dim,
      c10::ScalarType input_dtype,
      c10::ScalarType output_dtype,
      bool use_pdl)
      : tp_group(tp_group),
        world_size(tp_group->get_size()),
        nnodes(nnodes),
        gemm_op(tp_group->get_rank(), tp_group->get_size(), nnodes),
        ag_op(tp_group, nnodes, full_m, k_dim, input_dtype),
        use_pdl(use_pdl) {
    // copy stream
    CUDA_CHECK(cudaStreamCreateWithPriority(
        &this->cp_stream, cudaStreamNonBlocking, get_highest_cuda_stream_priority()));
    // create events
    CUDA_CHECK(cudaEventCreateWithFlags(&this->cp_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&this->ready_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&this->all_gather_event, cudaEventDisableTiming));

    if (this->use_pdl) {
      FLUX_CHECK(get_arch() == ArchEnum::Sm90);
    }
  }  // AllGatherGemmOpImpl

  ~AllGatherGemmOpImpl() {
    cudaStreamDestroy(cp_stream);
    cudaEventDestroy(cp_event);
    cudaEventDestroy(ready_event);
    cudaEventDestroy(all_gather_event);
  }

  torch::Tensor
  forward(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum,
      bool transpose_weight,
      AllGatherOptionWithOptional opt,
      c10::optional<torch::Tensor> gathered_input) {
    auto stream = at::cuda::getCurrentCUDAStream();

    bool is_s8_gemm = is_s8_torch_dtype(input.scalar_type());
    bool with_input_scale = is_s8_gemm && input_scale.has_value();
    return forward_impl(
        input,
        weight,
        bias,
        output,
        input_scale,
        weight_scale,
        output_scale,
        fast_accum,
        transpose_weight,
        materialize(opt, with_input_scale),
        gathered_input,
        c10::nullopt,
        stream);
  }

  // never mind the result
  torch::Tensor
  gemm_only(
      torch::Tensor input,  // this should be the full input
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      c10::optional<torch::Tensor> input_scale,  // this should be the full scale
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum,
      bool transpose_weight) {
    torch::Tensor barrier = torch::ones(
        {this->world_size},
        at::TensorOptions(at::ScalarType::Int)
            .device(torch::kCUDA)
            .device_index(at::cuda::current_device()));
    return this->gemm_op.forward(
        input,  // never mind the result
        weight,
        bias,
        output,
        input_scale,
        weight_scale,
        output_scale,
        barrier,
        fast_accum,
        nullptr,
        transpose_weight);
  }

  torch::Tensor
  forward_impl(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum,
      bool transpose_weight,
      const AllGatherOption &opt,
      c10::optional<torch::Tensor> gathered_input,
      c10::optional<UnifiedGemmHParams> const &hparams,
      cudaStream_t stream) {
    if (use_pdl && opt.use_cuda_core_ag) {
      return forward_with_pdl_impl(
          input,
          weight,
          bias,
          output,
          input_scale,
          weight_scale,
          output_scale,
          fast_accum,
          transpose_weight,
          opt,
          gathered_input,
          hparams,
          stream);
    } else {
      return forward_default_impl(
          input,
          weight,
          bias,
          output,
          input_scale,
          weight_scale,
          output_scale,
          fast_accum,
          transpose_weight,
          opt,
          gathered_input,
          hparams,
          stream);
    }
  }

  torch::Tensor
  forward_default_impl(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum,
      bool transpose_weight,
      const AllGatherOption &opt,
      c10::optional<torch::Tensor> gathered_input,
      c10::optional<UnifiedGemmHParams> const &hparams,
      cudaStream_t stream) {
    torch::Tensor barrier = ag_op.local_barrier_buffer();
    int M = input.size(0) * this->world_size;
    torch::Tensor input_buffer = ag_op.local_input_buffer().slice(0, 0, M);
    bool is_s8_gemm = is_s8_torch_dtype(input.scalar_type());
    at::optional<torch::Tensor> input_scale_tensor =
        is_s8_gemm
            ? (input_scale.has_value()
                   ? at::optional<torch::Tensor>{ag_op.local_input_scale_buffer().slice(0, 0, M)}
                   : c10::nullopt)
            : input_scale;

    // TODO(houqi.1993)
    CUDA_CHECK(cudaEventRecord(this->ready_event, stream));
    CUDA_CHECK(cudaStreamWaitEvent(this->cp_stream, this->ready_event));

    ag_op.run(input, is_s8_gemm ? input_scale : c10::nullopt, opt, this->cp_stream);

    CUDA_CHECK(cudaStreamWaitEvent(stream, ag_op.get_local_prepare_event()));

    auto result = this->gemm_op.forward(
        input_buffer,
        std::move(weight),
        std::move(bias),
        std::move(output),
        input_scale_tensor,
        std::move(weight_scale),
        std::move(output_scale),
        barrier,
        fast_accum,
        transpose_weight,
        hparams,
        opt.use_cuda_core_ag ? this->ag_op.ag_signal_ptr() : nullptr,
        stream);
    if (gathered_input.has_value()) {
      CHECK_INPUT(gathered_input.value(), input.scalar_type());
      CHECK_2D(gathered_input.value(), input.size(0) * this->world_size, input.size(1));
      CUDA_CHECK(cudaMemcpyAsync(
          gathered_input->data_ptr(),
          input_buffer.data_ptr(),
          gathered_input->nbytes(),
          cudaMemcpyDeviceToDevice,
          stream));
    }
    return result;
  }

  torch::Tensor
  forward_with_pdl_impl(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum,
      bool transpose_weight,
      const AllGatherOption &opt,
      c10::optional<torch::Tensor> gathered_input,
      c10::optional<UnifiedGemmHParams> const &hparams,
      cudaStream_t stream) {
    torch::Tensor barrier = ag_op.local_barrier_buffer();
    int M = input.size(0) * this->world_size;
    torch::Tensor input_buffer = ag_op.local_input_buffer().slice(0, 0, M);
    bool is_s8_gemm = is_s8_torch_dtype(input.scalar_type());
    at::optional<torch::Tensor> input_scale_tensor =
        is_s8_gemm
            ? (input_scale.has_value()
                   ? at::optional<torch::Tensor>{ag_op.local_input_scale_buffer().slice(0, 0, M)}
                   : c10::nullopt)
            : input_scale;

    auto output_buf = this->gemm_op.initialize(
        input_buffer,
        std::move(weight),
        std::move(bias),
        std::move(output),
        input_scale_tensor,
        std::move(weight_scale),
        std::move(output_scale),
        barrier,
        fast_accum,
        transpose_weight,
        hparams,
        stream);

    ag_op.run(input, is_s8_gemm ? input_scale : c10::nullopt, opt, stream);

    this->gemm_op.run(stream, /*launch_with_pdl=*/true);
    if (gathered_input.has_value()) {
      CHECK_INPUT(gathered_input.value(), input.scalar_type());
      CHECK_2D(gathered_input.value(), input.size(0) * this->world_size, input.size(1));
      CUDA_CHECK(cudaMemcpyAsync(
          gathered_input->data_ptr(),
          input_buffer.data_ptr(),
          gathered_input->nbytes(),
          cudaMemcpyDeviceToDevice,
          stream));
    }
    return output_buf;
  }

  torch::Tensor
  profiling(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum,
      bool transpose_weight,
      AllGatherOptionWithOptional option_,
      c10::optional<torch::Tensor> gathered_input,
      c10::intrusive_ptr<ProfilingContext> opt_ctx) {
    at::ScalarType input_dtype = input.scalar_type();
    bool is_fp8_gemm = is_fp8_torch_dtype(input_dtype);
    bool is_s8_gemm = is_s8_torch_dtype(input_dtype);
    at::ScalarType output_dtype =
        is_fp8_gemm || is_s8_gemm ? at::ScalarType::BFloat16 : input_dtype;

    if (is_fp8_gemm || is_s8_gemm) {
      FLUX_CHECK(!transpose_weight) << "FP8/INT8 GEMM does not support transpose_weight";
    }

    AllGatherOption option = materialize(option_, is_s8_gemm && input_scale.has_value());

    // NOTE: input shape is [m, k], where m = M / TP and M is the size for GEMM.
    int M = input.size(0) * this->world_size;
    int n = transpose_weight ? weight.size(1) : weight.size(0);
    int k = transpose_weight ? weight.size(0) : weight.size(1);

    auto stream = at::cuda::getCurrentCUDAStream();
    auto meta = unify_type(get_gemm_meta(
        input_dtype,
        output_dtype,
        transpose_weight,
        /*has_bias=*/bias.has_value(),
        /*fast_accum=*/fast_accum));
    auto rt_config = get_rt_config(
        this->world_size,
        this->nnodes,
        M,  // this should be full M for GEMM
        n,
        k,
        AGRingMode::All2All);  // TODO(houqi.1993) set this later

    ProfilingContext tmp_ctx("__tmp__");
    ProfilingContext *ctx = !opt_ctx ? &tmp_ctx : opt_ctx.get();

    // TODO: Add filter
    auto filter_hparams = [&](UnifiedGemmHParams const &hparams) { return true; };
    auto elapsed_tensor = torch::empty({}, weight.options().dtype(c10::ScalarType::Float));
    auto reduced_elapsed_tensor = elapsed_tensor.clone();

    OpRegistry::instance().visit_hparams(
        [&](UnifiedGemmHParams const &hparams) {
          if (not filter_hparams(hparams)) {
            return;
          }
          constexpr int warmup = 5;
          constexpr int iters = 10;
          float total_elapsed = 0;

          auto stream = c10::cuda::getCurrentCUDAStream();
          this->tp_group->sync();
          for (int iter = 0; iter < warmup + iters; ++iter) {
            GpuTimer timer;
            timer.start(stream);
            auto _ [[maybe_unused]] = this->forward_impl(
                input,
                weight,
                bias,
                output,
                input_scale,
                weight_scale,
                output_scale,
                fast_accum,
                transpose_weight,
                option,  // profile with input buffer
                gathered_input,
                hparams,
                stream);
            timer.stop();
            if (iter >= warmup) {
              total_elapsed += timer.elapsed_millis();
            }
          }

          // Avoid GPU frequency adjustment
          this->tp_group->sync();
          sleep(1);

          float avg_elapsed = int(total_elapsed / iters * 1000) / 1000.0;
          float reduce_elapsed = all_reduce_max_float(this->tp_group.get(), avg_elapsed);
          ctx->add(meta, rt_config, hparams, reduce_elapsed);
        },
        meta);

    auto best_hparams = ctx->record_best(meta, rt_config);
    return this->forward_impl(
        std::move(input),
        std::move(weight),
        std::move(bias),
        std::move(output),
        std::move(input_scale),
        std::move(weight_scale),
        std::move(output_scale),
        fast_accum,
        transpose_weight,
        option,
        std::move(gathered_input),
        std::move(best_hparams),
        stream);
  }
};

AllGatherGemmOp::AllGatherGemmOp(
    std::shared_ptr<Group> tp_group,
    int32_t nnodes,
    int32_t full_m,
    int32_t n_dim,
    int32_t k_dim,
    c10::ScalarType input_dtype,
    c10::ScalarType output_dtype,
    bool use_pdl)
    : impl_(new AllGatherGemmOpImpl(
          tp_group, nnodes, full_m, n_dim, k_dim, input_dtype, output_dtype, use_pdl)) {}

AllGatherGemmOp::~AllGatherGemmOp() { delete impl_; }

torch::Tensor
AllGatherGemmOp::forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> output,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale,
    c10::optional<torch::Tensor> output_scale,
    bool fast_accum,
    bool transpose_weight,
    AllGatherOptionWithOptional opt,
    c10::optional<torch::Tensor> gathered_input) {
  FLUX_CHECK(this->impl_ != nullptr) << "AllGatherGemmOp is not initialized";
  return this->impl_->forward(
      std::move(input),
      std::move(weight),
      std::move(bias),
      std::move(output),
      std::move(input_scale),
      std::move(weight_scale),
      std::move(output_scale),
      fast_accum,
      transpose_weight,
      opt,
      std::move(gathered_input));
}

torch::Tensor
AllGatherGemmOp::gemm_only(
    torch::Tensor input,  // this should be the full input
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> output,
    c10::optional<torch::Tensor> input_scale,  // this should be the full scale
    c10::optional<torch::Tensor> weight_scale,
    c10::optional<torch::Tensor> output_scale,
    bool fast_accum,
    bool transpose_weight) {
  FLUX_CHECK(this->impl_ != nullptr) << "AllGatherGemmOp is not initialized";
  return this->impl_->gemm_only(
      std::move(input),
      std::move(weight),
      std::move(bias),
      std::move(output),
      std::move(input_scale),
      std::move(weight_scale),
      std::move(output_scale),
      fast_accum,
      transpose_weight);
}

torch::Tensor
AllGatherGemmOp::profiling(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> output,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale,
    c10::optional<torch::Tensor> output_scale,
    bool fast_accum,
    bool transpose_weight,
    AllGatherOptionWithOptional option_,
    c10::optional<torch::Tensor> gathered_input,
    c10::intrusive_ptr<ProfilingContext> opt_ctx) {
  FLUX_CHECK(this->impl_ != nullptr) << "AllGatherGemmOp is not initialized";
  return this->impl_->profiling(
      std::move(input),
      std::move(weight),
      std::move(bias),
      std::move(output),
      std::move(input_scale),
      std::move(weight_scale),
      std::move(output_scale),
      fast_accum,
      transpose_weight,
      option_,
      std::move(gathered_input),
      opt_ctx);
}

}  // namespace ths_op
}  // namespace flux
}  // namespace bytedance
