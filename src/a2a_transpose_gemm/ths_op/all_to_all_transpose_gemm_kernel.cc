//===- all_to_all_transpose_gemm_kernel.cc ------------------------ C++ ---===//
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

#include "a2a_transpose_gemm/ths_op/all_to_all_transpose_gemm_kernel.h"

#include "a2a_transpose_gemm/ths_op/all_to_all_types.h"
#include "post_attn_all_to_all_transpose_op.h"
#include "flux/args/a2a_transpose_gemm.h"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/cuda_stub.h"
#include "flux/flux.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_meta.h"
#include "flux/op_registry.h"
#include "flux/runtime_config.h"
#include "flux/ths_op/ths_op.h"
#include "flux/ths_op/topo_utils.h"
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
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#define SPLIT 1
#define MAX_NUM_SIGNAL 64

namespace bytedance {
namespace flux {
namespace ths_op {
using torch::Tensor;

namespace {
inline void *
ptr_offset(void *ptr, ptrdiff_t offset) {
  return static_cast<char *>(ptr) + offset;
}

auto
get_gemm_meta(
    bool a2a_only,
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

  auto comm_op = a2a_only ? _PostAttnAllToAllOnly{}() : _PostAttnAllToAllTranspose{}();
  auto meta = make_gemm_meta(dtype_config, arch, sm_core, comm_op, gemm_layout, impl, impl_spec);
  return meta;
}

RuntimeConfig
get_rt_config(
    int world_size,
    int nnodes,
    int m,
    int n,
    int k,
    A2ARingMode ring_mode = A2ARingMode::All2All) {
  return make_runtime_config(
      m, n, k, make_all_to_all_transpose_runtime_config(world_size, nnodes, (int)ring_mode));
}

// TODO(chhuang): reuse `src/ag_gemm/ths_op/gemm_with_barrier.h`
class GemmWithBarirer : public torch::CustomClassHolder {
 private:
  void
  lazy_init_gemm_buffer(torch::Tensor input, int64_t buffer_size) {
    if (buffer_size <= 0)
      return;
    buffer_size = (buffer_size + 127) / 128 * 128;
    if (!this->gemm_buffer.defined() || buffer_size > this->gemm_buffer.numel()) {
      auto options = input.options().dtype(c10::ScalarType::Byte);
      this->gemm_buffer =
          empty_with_uninitialized_data(std::vector<int64_t>{buffer_size}, options);
    }
  }

 public:
  GemmWithBarirer(int rank, int world_size, int32_t nnodes, int32_t sp_size, bool a2a_only)
      : nnodes(nnodes),
        world_size(world_size),
        rank(rank),
        sp_size(sp_size),
        sp_rank(rank % sp_size),
        a2a_only(a2a_only) {}

  torch::Tensor
  forward(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      torch::Tensor barrier,
      bool fast_accum,
      bool transpose_weight,
      c10::optional<UnifiedGemmHParams> const &hparams,
      int32_t *a2a_signal,
      int32_t m_per_barrier,
      int32_t sm_margin,
      cudaStream_t stream) {
    return forward_impl(
        input,
        weight,
        bias,
        output,
        input_scale,
        weight_scale,
        output_scale,
        barrier,
        fast_accum,
        transpose_weight,
        hparams,
        a2a_signal,
        m_per_barrier,
        sm_margin,
        stream);
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
      torch::Tensor barrier,
      bool fast_accum,
      bool transpose_weight,
      c10::optional<UnifiedGemmHParams> const &hparams,
      int32_t *a2a_signal,
      int32_t m_per_barrier,
      int32_t sm_margin,
      cudaStream_t stream) {
    at::ScalarType input_dtype = input.scalar_type();
    at::ScalarType output_dtype = input_dtype;

    // verify all kinds of shapes
    FLUX_CHECK((!transpose_weight && input_dtype == at::ScalarType::BFloat16))
        << "GEMM only support BF16 without transpose weight";

    CHECK_NDIM(input, 2);
    CHECK_CUDA(input);
    CHECK_NDIM(weight, 2);
    CHECK_CUDA(weight);

    int m = input.size(0);
    int n = transpose_weight ? weight.size(1) : weight.size(0);
    int k = transpose_weight ? weight.size(0) : weight.size(1);
    CHECK_2D(input, m, k);
    CHECK_TYPE(weight, input_dtype);

    if (bias.has_value()) {
      CHECK_TYPE(bias.value(), output_dtype);
      CHECK_CUDA(bias.value());
    }

    auto meta = get_gemm_meta(
        this->a2a_only,
        input_dtype,
        output_dtype,
        transpose_weight,
        /*has_bias=*/bias.has_value(),
        /*fast_accum=*/fast_accum);
    std::unique_ptr<GemmOperatorBase> cutlass_op;
    auto rt_config = get_rt_config(this->sp_size, this->nnodes, m, n, k, A2ARingMode::All2All);
    if (hparams.has_value()) {
      cutlass_op = OpRegistry::instance().get_op(meta, hparams.value());
    } else {
      auto params = OpRegistry::instance().get_hparams(meta, rt_config);
      cutlass_op = OpRegistry::instance().get_op(meta, params);
    }
    torch::Tensor output_tensor;

    if (output.has_value()) {
      output_tensor = output.value();
      CHECK_CUDA(output_tensor);
      FLUX_CHECK(output_tensor.is_contiguous()) << "output_tensor should be contiguous.";
      CHECK_2D(output_tensor, m, n);
      CHECK_TYPE(output_tensor, output_dtype);
    } else {
      output_tensor = empty_with_uninitialized_data(
          std::vector<int64_t>{m, n},
          at::TensorOptions(output_dtype)
              .device(at::kCUDA)
              .device_index(c10::cuda::current_device()));
    }
    std::any gemm_args;
    auto data_ptr_or = [](auto &&t, void *other) -> void * {
      return t.has_value() ? t->data_ptr() : other;
    };

    // A2A GEMM Arguments
    gemm_args = A2ATransposeGemmKernelArguments{
        .m = m,
        .n = n,
        .k = k,
        .m_per_barrier = m_per_barrier,
        .sm_margin = sm_margin,
        .rank = sp_rank,
        .world_size = sp_size,
        .nnodes = nnodes,
        .alpha = 1.0f,
        .beta = bias.has_value() ? 1.0f : 0.0f,
        .input = input.data_ptr(),
        .weight = weight.data_ptr(),
        .bias = data_ptr_or(bias, nullptr),
        .output = output_tensor.data_ptr(),
        .barrier_buffer = barrier.data_ptr(),
        .a2a_signal = a2a_signal};

    // A2A Gemm Workspace
    int64_t workspace_size = cutlass_op->get_workspace_size(gemm_args);
    this->lazy_init_gemm_buffer(input, workspace_size);

    /// GEMM
    if (a2a_signal != nullptr) {
      CU_CHECK(CUStreamWaitValue(stream, (CUdeviceptr)(a2a_signal), 1, CU_STREAM_WAIT_VALUE_EQ));
    }
    cutlass_op->run(gemm_args, workspace_size ? this->gemm_buffer.data_ptr() : nullptr, stream);
    return output_tensor;
  }

 private:
  // used to ThreadblockSwizzle logic.
  int nnodes;
  int world_size;
  int rank;
  int sp_size;
  int sp_rank;
  bool a2a_only;

  // cutlass gemm workspace buffer
  torch::Tensor gemm_buffer;
};

}  // namespace

/// Alltoall+transpose + GEMM Kernel OP
class AllToAllTransposeGemmOp::AllToAllTransposeGemmOpImpl {
 private:
  using FlagType = int32_t;

 private:
  std::shared_ptr<Group> pg_world;
  int rank;
  int world_size;
  int local_world_size;
  int nnodes;
  int sp_size;
  int sp_rank;
  bool a2a_only;
  cudaStream_t compute_stream;
  cudaEvent_t cp_event;
  cudaEvent_t ready_event;
  cudaEvent_t compute_event;

  GemmWithBarirer gemm_op;
  PostAttnAllToAllTransposeOp a2a_op;

 private:
  AllToAllOption
  materialize(const AllToAllOptionWithOptional opt, bool return_comm_buf, bool skip_barrier) {
    return AllToAllOption{
        .input_buffer_copied = opt.input_buffer_copied.value_or(false),
        .use_cuda_core = opt.use_cuda_core.value_or(true),
        .fuse_sync = opt.fuse_sync.value_or(true),
        .use_read = opt.use_read.value_or(false),
        .skip_barrier = opt.skip_barrier.value_or(skip_barrier),
        .return_comm_buf = opt.return_comm_buf.value_or(return_comm_buf),
        .mode = opt.mode.value_or(get_default_a2a_ring_mode())};
  }

 public:
  AllToAllTransposeGemmOpImpl(
      std::shared_ptr<Group> pg_world,
      int32_t nnodes,
      int32_t sp_size,
      int32_t bs,
      int32_t num_head,
      int32_t seq,
      int32_t head_dim,
      int32_t max_num_comm_buf,
      c10::ScalarType input_dtype,
      c10::ScalarType output_dtype,
      bool a2a_only)
      : pg_world(pg_world),
        rank(pg_world->get_rank()),
        world_size(pg_world->get_size()),
        local_world_size(pg_world->get_size() / nnodes),
        nnodes(nnodes),
        sp_size(sp_size),
        sp_rank(pg_world->get_rank() % sp_size),
        a2a_only(a2a_only),
        gemm_op(pg_world->get_rank(), pg_world->get_size(), nnodes, sp_size, a2a_only),
        a2a_op(
            pg_world,
            nnodes,
            sp_size,
            bs,
            num_head,
            seq,
            head_dim,
            max_num_comm_buf,
            input_dtype,
            a2a_only) {
    FLUX_CHECK(input_dtype == c10::ScalarType::BFloat16) << "A2A Gemm Only support BF16 input";
    FLUX_CHECK(output_dtype == c10::ScalarType::BFloat16) << "A2A Gemm Only support BF16 output";
    FLUX_CHECK(local_world_size % sp_size == 0) << "local_world_size must be divisible by sp_size";
    // copy stream
    CUDA_CHECK(cudaStreamCreateWithPriority(
        &this->compute_stream, cudaStreamNonBlocking, get_highest_cuda_stream_priority()));
    // create events
    CUDA_CHECK(cudaEventCreateWithFlags(&this->cp_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&this->ready_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&this->compute_event, cudaEventDisableTiming));

  }  // AllToAllTransposeGemmOpImpl

  ~AllToAllTransposeGemmOpImpl() {
    cudaStreamDestroy(compute_stream);
    cudaEventDestroy(cp_event);
    cudaEventDestroy(ready_event);
    cudaEventDestroy(compute_event);
  }

  torch::Tensor
  forward(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> seq_lens_cpu,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum,
      bool transpose_weight,
      AllToAllOptionWithOptional opt,
      c10::optional<torch::Tensor> a2a_transpose_output,
      int32_t num_comm_sm,
      int32_t sm_margin) {
    auto stream = at::cuda::getCurrentCUDAStream();
    auto out = forward_impl(
        input,
        weight,
        seq_lens_cpu,
        bias,
        output,
        input_scale,
        weight_scale,
        output_scale,
        fast_accum,
        transpose_weight,
        materialize(opt, true, false),
        a2a_transpose_output,
        num_comm_sm,
        sm_margin,
        c10::nullopt,  // use default hparams
        stream);

    // user need to insert the necessary wait_stream/record_stream to ensure that the inputs is
    // not deallocated.
    // input.record_stream(stream);
    // weight.record_stream(stream);
    // auto record_stream_for_optional_tensor = [&](c10::optional<torch::Tensor> t) -> void {
    //   if (t.has_value())
    //     t.value().record_stream(stream);
    // };
    // record_stream_for_optional_tensor(bias);
    // record_stream_for_optional_tensor(output);
    // record_stream_for_optional_tensor(input_scale);
    // record_stream_for_optional_tensor(weight_scale);
    // record_stream_for_optional_tensor(output_scale);
    return out;
  }

  torch::Tensor
  post_attn_a2a(
      torch::Tensor input,
      c10::optional<torch::Tensor> seq_lens_cpu,
      AllToAllOptionWithOptional opt,
      int32_t num_comm_sm,
      bool return_comm_buf,
      int32_t comm_buf_idx) {
    auto stream = at::cuda::getCurrentCUDAStream();
    auto out = post_attn_a2a_impl(
        input,
        seq_lens_cpu,
        materialize(opt, return_comm_buf, true),
        comm_buf_idx,
        num_comm_sm,
        stream);
    // input.record_stream(stream);
    return out;
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
      bool transpose_weight,
      int32_t num_comm_sm,
      int32_t sm_margin) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    torch::Tensor barrier = torch::ones(
        {this->sp_size},
        at::TensorOptions(at::ScalarType::Int)
            .device(torch::kCUDA)
            .device_index(at::cuda::current_device()));
    int K = transpose_weight ? weight.size(0) : weight.size(1);
    int M = input.numel() / K;
    if (num_comm_sm == -1) {
      num_comm_sm = this->world_size;
    }
    FLUX_CHECK(num_comm_sm > 0);
    int32_t m_per_barrier = (M + this->sp_size - 1) / this->sp_size;
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
        transpose_weight,
        c10::nullopt,
        nullptr,
        m_per_barrier,
        sm_margin + num_comm_sm,
        stream);
  }

  torch::Tensor
  forward_impl(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> seq_lens_cpu,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum,
      bool transpose_weight,
      AllToAllOption opt,
      c10::optional<torch::Tensor> a2a_transpose_output,
      int32_t num_comm_sm,
      int32_t sm_margin,
      c10::optional<UnifiedGemmHParams> const &hparams,
      cudaStream_t stream) {
    if (num_comm_sm == -1) {
      num_comm_sm = this->world_size;
    }
    FLUX_CHECK(num_comm_sm > 0);
    FLUX_CHECK(!opt.skip_barrier);
    CHECK_NDIM(weight, 2);
    CHECK_NDIM(input, 4);
    FLUX_CHECK(input.is_contiguous()) << "input should be contiguous.";
    FLUX_CHECK(weight.is_contiguous()) << "weight should be contiguous.";

    int N = transpose_weight ? weight.size(1) : weight.size(0);
    int K = transpose_weight ? weight.size(0) : weight.size(1);
    int M;
    if (seq_lens_cpu.has_value()) {
      // dp mode, seq len of each rank is different.
      auto seq_lens_cpu_tensor = seq_lens_cpu.value();
      FLUX_CHECK(seq_lens_cpu_tensor.device().type() == torch::kCPU) << "seq_lens is not on cpu";
      FLUX_CHECK(seq_lens_cpu_tensor.dtype() == c10::ScalarType::Int);
      FLUX_CHECK(seq_lens_cpu_tensor.is_contiguous()) << "seq_lens must be contiguous.";

      int32_t total_seq_len = 0;
      int32_t *seq_lens_ptr = seq_lens_cpu_tensor.data_ptr<int32_t>();
      for (int32_t i = 0; i < this->sp_size; ++i) {
        total_seq_len += *(seq_lens_ptr + i);
      }
      int32_t cur_local_seq_len = *(seq_lens_ptr + this->sp_rank);
      int bs = input.numel() / (K / this->sp_size) / total_seq_len;
      M = cur_local_seq_len * bs;
    } else {
      M = input.numel() / K;
    }
    torch::Tensor barrier = a2a_op.local_barrier_buffer();
    torch::Tensor gemm_input_buffer =
        a2a_op.local_comm_output_buffer().reshape(-1).slice(0, 0, (size_t)M * K).reshape({M, K});
    at::optional<torch::Tensor> input_scale_tensor = c10::nullopt;

    FLUX_CHECK(!transpose_weight);
    FLUX_CHECK(!input_scale.has_value() && !weight_scale.has_value() && !output_scale.has_value())
        << "A2A Gemm does not support scale.";

    CUDA_CHECK(cudaEventRecord(this->ready_event, stream));
    CUDA_CHECK(cudaStreamWaitEvent(this->compute_stream, this->ready_event));

    if (seq_lens_cpu.has_value())
      a2a_op.run(input, seq_lens_cpu.value(), opt, /*comm_buf_idx*/ 0, num_comm_sm, stream);
    else
      a2a_op.run(input, opt, /*comm_buf_idx*/ 0, num_comm_sm, stream);

    auto result = this->gemm_op.forward(
        gemm_input_buffer,
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
        opt.use_cuda_core ? this->a2a_op.a2a_signal_ptr() : nullptr,
        a2a_op.m_per_barrier(),
        sm_margin + num_comm_sm,
        this->compute_stream);
    if (a2a_transpose_output.has_value()) {
      CHECK_INPUT(a2a_transpose_output.value(), input.scalar_type());
      FLUX_CHECK(a2a_transpose_output->numel() == gemm_input_buffer.numel());
      CUDA_CHECK(cudaMemcpyAsync(
          a2a_transpose_output->data_ptr(),
          gemm_input_buffer.data_ptr(),
          a2a_transpose_output->nbytes(),
          cudaMemcpyDeviceToDevice,
          this->compute_stream));
    }
    CUDA_CHECK(cudaEventRecord(this->compute_event, this->compute_stream));
    CUDA_CHECK(cudaStreamWaitEvent(stream, this->compute_event));
    a2a_op.reset_signals(opt, stream);
    return result;
  }

  torch::Tensor
  post_attn_a2a_impl(
      torch::Tensor input,
      c10::optional<torch::Tensor> seq_lens_cpu,
      AllToAllOption opt,
      int32_t comm_buf_idx,
      int32_t num_comm_sm,
      cudaStream_t stream) {
    if (num_comm_sm == -1) {
      num_comm_sm = this->world_size;
    }
    FLUX_CHECK(num_comm_sm > 0);
    CHECK_NDIM(input, 4);
    FLUX_CHECK(input.is_contiguous()) << "input should be contiguous.";
    if (seq_lens_cpu.has_value())
      return a2a_op.run(input, seq_lens_cpu.value(), opt, comm_buf_idx, num_comm_sm, stream);
    else
      return a2a_op.run(input, opt, comm_buf_idx, num_comm_sm, stream);
  }

  void
  sp_group_barrier_all(cudaStream_t stream) {
    a2a_op.sp_group_barrier_async(stream);
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
      AllToAllOptionWithOptional option_opt,
      c10::optional<torch::Tensor> a2a_transpose_output,
      int32_t num_comm_sm,
      int32_t sm_margin,
      c10::intrusive_ptr<ProfilingContext> opt_ctx) {
    AllToAllOption option = materialize(option_opt, true, false);

    at::ScalarType input_dtype = input.scalar_type();
    bool is_fp8_gemm = is_fp8_torch_dtype(input_dtype);
    bool is_s8_gemm = is_s8_torch_dtype(input_dtype);
    at::ScalarType output_dtype =
        is_fp8_gemm || is_s8_gemm ? at::ScalarType::BFloat16 : input_dtype;

    FLUX_CHECK(!(transpose_weight == true && is_fp8_gemm == true))
        << "FP8 GEMM does not support transpose weight";

    // NOTE: input shape is [bs, nh/n, s, hd]
    //       weight shape is [nh * hd, nh * hd]
    int n = transpose_weight ? weight.size(1) : weight.size(0);
    int k = transpose_weight ? weight.size(0) : weight.size(1);
    int m = input.numel() / k;

    auto stream = at::cuda::getCurrentCUDAStream();
    auto meta = unify_type(get_gemm_meta(
        this->a2a_only,
        input_dtype,
        output_dtype,
        transpose_weight,
        /*has_bias=*/bias.has_value(),
        /*fast_accum=*/fast_accum));
    auto rt_config = get_rt_config(
        this->sp_size,
        this->nnodes,
        m,
        n,
        k,
        A2ARingMode::All2All);  // TODO: set this later

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
          constexpr int warmup = 20;
          constexpr int iters = 100;
          float total_elapsed = 0;

          auto stream = c10::cuda::getCurrentCUDAStream();
          this->pg_world->sync();
          for (int iter = 0; iter < warmup + iters; ++iter) {
            GpuTimer timer;
            timer.start(stream);
            auto _ [[maybe_unused]] = this->forward_impl(
                input,
                weight,
                /*seq_len_cpu = */ c10::nullopt,
                bias,
                output,
                input_scale,
                weight_scale,
                output_scale,
                fast_accum,
                transpose_weight,
                option,  // profile with input buffer
                a2a_transpose_output,
                num_comm_sm,
                sm_margin,
                hparams,
                stream);
            timer.stop();
            if (iter >= warmup) {
              total_elapsed += timer.elapsed_millis();
            }
          }

          // Avoid GPU frequency adjustment
          this->pg_world->sync();
          sleep(1);

          float avg_elapsed = int(total_elapsed / iters * 1000) / 1000.0;
          float reduce_elapsed = all_reduce_max_float(this->pg_world.get(), avg_elapsed);
          ctx->add(meta, rt_config, hparams, reduce_elapsed);
        },
        meta);

    auto best_hparams = ctx->record_best(meta, rt_config);
    return this->forward_impl(
        std::move(input),
        std::move(weight),
        /*seq_len_cpu = */ c10::nullopt,
        std::move(bias),
        std::move(output),
        std::move(input_scale),
        std::move(weight_scale),
        std::move(output_scale),
        fast_accum,
        transpose_weight,
        option,
        std::move(a2a_transpose_output),
        num_comm_sm,
        sm_margin,
        std::move(best_hparams),
        stream);
  }
};

AllToAllTransposeGemmOp::AllToAllTransposeGemmOp(
    std::shared_ptr<Group> pg_world,
    int32_t nnodes,
    int32_t sp_size,
    int32_t bs,
    int32_t num_head,
    int32_t seq,
    int32_t head_dim,
    int32_t max_num_comm_buf,
    c10::ScalarType input_dtype,
    c10::ScalarType output_dtype,
    bool a2a_only)
    : impl_(new AllToAllTransposeGemmOpImpl(
          pg_world,
          nnodes,
          sp_size,
          bs,
          num_head,
          seq,
          head_dim,
          max_num_comm_buf,
          input_dtype,
          output_dtype,
          a2a_only)) {}

AllToAllTransposeGemmOp::~AllToAllTransposeGemmOp() { delete impl_; }

torch::Tensor
AllToAllTransposeGemmOp::forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> seq_lens_cpu,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> output,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale,
    c10::optional<torch::Tensor> output_scale,
    bool fast_accum,
    bool transpose_weight,
    AllToAllOptionWithOptional option,
    c10::optional<torch::Tensor> a2a_transpose_output,
    int32_t num_comm_sm,
    int32_t sm_margin) {
  FLUX_CHECK(this->impl_ != nullptr) << "AllToAllTransposeGemmOp is not initialized";
  return this->impl_->forward(
      std::move(input),
      std::move(weight),
      std::move(seq_lens_cpu),
      std::move(bias),
      std::move(output),
      std::move(input_scale),
      std::move(weight_scale),
      std::move(output_scale),
      fast_accum,
      transpose_weight,
      option,
      std::move(a2a_transpose_output),
      num_comm_sm,
      sm_margin);
}

torch::Tensor
AllToAllTransposeGemmOp::post_attn_a2a(
    torch::Tensor input,
    c10::optional<torch::Tensor> seq_lens_cpu,
    AllToAllOptionWithOptional opt,
    int32_t num_comm_sm) {
  FLUX_CHECK(this->impl_ != nullptr) << "AllToAllTransposeGemmOp is not initialized";
  return this->impl_->post_attn_a2a(
      std::move(input),
      std::move(seq_lens_cpu),
      opt,
      num_comm_sm,
      /*return_comm_buf*/ false,
      /*comm_buf_idx*/ 0);
}

torch::Tensor
AllToAllTransposeGemmOp::post_attn_a2a_no_cpy(
    torch::Tensor input,
    c10::optional<torch::Tensor> seq_lens_cpu,
    AllToAllOptionWithOptional opt,
    int32_t num_comm_sm,
    int32_t comm_buf_idx) {
  FLUX_CHECK(this->impl_ != nullptr) << "AllToAllTransposeGemmOp is not initialized";
  return this->impl_->post_attn_a2a(
      std::move(input),
      std::move(seq_lens_cpu),
      opt,
      num_comm_sm,
      /*return_comm_buf*/ true,
      comm_buf_idx);
}

void
AllToAllTransposeGemmOp::sp_group_barrier_all() {
  FLUX_CHECK(this->impl_ != nullptr) << "AllToAllTransposeGemmOp is not initialized";
  auto stream = at::cuda::getCurrentCUDAStream();
  return this->impl_->sp_group_barrier_all(stream);
}

torch::Tensor
AllToAllTransposeGemmOp::gemm_only(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> output,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale,
    c10::optional<torch::Tensor> output_scale,
    bool fast_accum,
    bool transpose_weight,
    int32_t num_comm_sm,
    int32_t sm_margin) {
  FLUX_CHECK(this->impl_ != nullptr) << "AllToAllTransposeGemmOp is not initialized";
  return this->impl_->gemm_only(
      std::move(input),
      std::move(weight),
      std::move(bias),
      std::move(output),
      std::move(input_scale),
      std::move(weight_scale),
      std::move(output_scale),
      fast_accum,
      transpose_weight,
      num_comm_sm,
      sm_margin);
}

torch::Tensor
AllToAllTransposeGemmOp::profiling(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> output,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale,
    c10::optional<torch::Tensor> output_scale,
    bool fast_accum,
    bool transpose_weight,
    AllToAllOptionWithOptional option,
    c10::optional<torch::Tensor> a2a_transpose_output,
    int32_t num_comm_sm,
    int32_t sm_margin,
    c10::intrusive_ptr<ProfilingContext> opt_ctx) {
  FLUX_CHECK(this->impl_ != nullptr) << "AllToAllTransposeGemmOp is not initialized";
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
      option,
      std::move(a2a_transpose_output),
      num_comm_sm,
      sm_margin,
      opt_ctx);
}

}  // namespace ths_op
}  // namespace flux
}  // namespace bytedance
