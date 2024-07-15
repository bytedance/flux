//===- all_gather_gemm_kernel.cc ---------------------------------- C++ ---===//
//
// Copyright 2023 ByteDance Ltd. and/or its affiliates. All rights reserved.
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

#include "all_gather_ring_order.h"

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
#include "flux/args/all_gather.h"
#include "flux/utils.h"
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/List.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/tensor.h>
#include <ATen/ops/zeros.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/intrusive_ptr.h>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime_api.h>
#ifdef FLUX_SHM_USE_NVSHMEM
#include <nvshmem.h>
#include <nvshmemx.h>
#endif
#include <pybind11/cast.h>
#include <pybind11/pytypes.h>
#include <torch/all.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <variant>
#include <torch/python.h>
#include "all_gather_types.h"
#include "pybind11/pybind11.h"

#define CUStreamWaitValue(...) cuda_stub().cuStreamWaitValue32_v2(__VA_ARGS__)
#define CUStreamWriteValue(...) cuda_stub().cuStreamWriteValue32_v2(__VA_ARGS__)

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
}  // namespace

/// All Gather GEMM Kernel OP
class AGKernel : public torch::CustomClassHolder {
 private:
  using FlagType = int32_t;

 private:
  c10::intrusive_ptr<c10d::ProcessGroup> tp_group;
  int32_t nnodes;
  int32_t full_m;
  int32_t n_dim;
  int32_t k_dim;
  c10::ScalarType input_dtype;
  c10::ScalarType output_dtype;
  const bool transpose_weight;
  const bool local_copy;
  const bool is_fp8_gemm;

 private:
  int32_t rank;
  int32_t world_size;
  int32_t local_world_size;
  int32_t local_rank;
#ifndef FLUX_SHM_USE_NVSHMEM
  // used for the cuda-ipc-barrier
  std::vector<torch::Tensor> sync_buffers;
#endif
  std::vector<torch::Tensor> input_buffers;
  std::vector<torch::Tensor> output_buffers;
  std::vector<torch::Tensor> barrier_buffers;
  torch::Tensor input_buffer;
  torch::Tensor output_buffer;
  torch::Tensor barrier_buffer;

  std::vector<void *> input_ptrs;
  std::vector<FlagType *> barrier_ptrs;
  AGRingMode ring_mode;

  // cutlass gemm
  std::any gemm_args;  // AGKernelArguments/AGFP8KernelArguments
  torch::Tensor gemm_buffer;
  void *workspace;
  std::unique_ptr<GemmOperatorBase> cutlass_op;

  size_t chunk_size;
  size_t split_chunk_size;

  int num_cp_streams;
  std::vector<at::cuda::CUDAStream> cp_streams;
  std::vector<at::cuda::CUDAStream> reset_stream;

  cudaEvent_t cp_event;
  cudaEvent_t ready_event;
  cudaEvent_t all_gather_event;

 public:
  AGKernel(
      c10::intrusive_ptr<c10d::ProcessGroup> tp_group_,
      int32_t nnodes,
      int32_t full_m,
      int32_t n_dim,
      int32_t k_dim,
      c10::ScalarType input_dtype,
      c10::ScalarType output_dtype,
      bool transpose_weight = true,
      bool local_copy = false,
      AGRingMode ring_mode_ = AGRingMode::Auto)
      : tp_group(tp_group_),
        nnodes(nnodes),
        full_m(full_m),
        n_dim(n_dim),
        k_dim(k_dim),
        input_dtype(input_dtype),
        output_dtype(output_dtype),
        transpose_weight(transpose_weight),
        local_copy(local_copy),
        is_fp8_gemm(is_fp8_torch_dtype(input_dtype)),
        rank(tp_group->getRank()),
        world_size(tp_group->getSize()),
        local_world_size(world_size / nnodes),
        local_rank(rank % local_world_size),
        input_ptrs(world_size, nullptr),
        barrier_ptrs(world_size, nullptr) {
    FLUX_CHECK(rank >= 0 && rank < world_size)
        << "invalid rank: " << rank << " and world_size: " << world_size;
    FLUX_CHECK(world_size % nnodes == 0)
        << "invalid nnodes: world_size[" << world_size << "] %% nnodes[" << nnodes << "] != 0";
    FLUX_CHECK(!(transpose_weight == true && is_fp8_gemm == true))
        << "FP8 GEMM does not support transpose weight";
    this->ring_mode = get_ring_mode(ring_mode_);

    // input buffer
    this->input_buffers = flux_create_tensor_list({full_m, k_dim}, input_dtype, this->tp_group);
    this->input_buffer = this->input_buffers[this->local_rank];
    for (int i = 0; i < world_size; ++i) {
      if (i / this->local_world_size == rank / this->local_world_size) {
        // on the same node
        this->input_ptrs[i] = this->input_buffers[i].data_ptr();
      } else {
        this->input_ptrs[i] = nullptr;
      }
    }

    // output buffer
    this->output_buffer = torch::empty(
        {full_m, n_dim},
        at::TensorOptions()
            .dtype(output_dtype)
            .device(at::kCUDA)
            .device_index(c10::cuda::current_device()));

    // barrier buffer
    int num_signals = MAX_NUM_SIGNAL;
    this->barrier_buffers =
        flux_create_tensor_list({num_signals}, c10::ScalarType::Int, this->tp_group);
    this->barrier_buffer = this->barrier_buffers[this->local_rank];
    for (int i = 0; i < world_size; ++i) {
      if (i / this->local_world_size == rank / this->local_world_size) {
        // on the same node
        this->barrier_ptrs[i] = (FlagType *)this->barrier_buffers[i].data_ptr();
      } else {
        this->barrier_ptrs[i] = (FlagType *)nullptr;
      }
    }

    // for (int i = 0; i < world_size; ++i) {
    //   this->barrier_ptrs[i] = (FlagType *)nvshmem_ptr(this->barrier_buffer.data_ptr(), i);
    // }

    // copy stream
    this->num_cp_streams = 1;
    for (int i = 0; i < this->num_cp_streams; ++i) {
      this->cp_streams.push_back(at::cuda::getStreamFromPool());
    }
    // create events
    CUDA_CHECK(cudaEventCreateWithFlags(&this->cp_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&this->ready_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&this->all_gather_event, cudaEventDisableTiming));
    // reset stream
    for (int i = 0; i < 1; ++i) {
      this->reset_stream.push_back(at::cuda::getStreamFromPool());
    }
#ifndef FLUX_SHM_USE_NVSHMEM
    this->sync_buffers =
        flux_create_tensor_list({this->world_size}, c10::ScalarType::Int, this->tp_group);
    this->sync_buffers[this->rank].zero_();  // zeros the sync buffer for cuda ipc at the start
#endif
  }

  void
  lazy_init_gemm_buffer(torch::Tensor input, int64_t buffer_size) {
    if (buffer_size <= 0)
      return;
    buffer_size = (buffer_size + 127) / 128 * 128;
    if (!this->gemm_buffer.defined() || buffer_size > this->gemm_buffer.numel()) {
      auto options = input.options().dtype(c10::ScalarType::Byte);
      this->gemm_buffer = torch::empty({buffer_size}, options);
    }
  }

  torch::Tensor
  get_optional_scale_tensor(c10::optional<torch::Tensor> optional_scale_tensor) {
    torch::Tensor scale_tensor;
    if (optional_scale_tensor.has_value()) {
      scale_tensor = optional_scale_tensor.value();
    } else {
      scale_tensor = torch::empty(
          {1},
          at::TensorOptions()
              .dtype(c10::ScalarType::Float)
              .device(at::kCUDA)
              .device_index(c10::cuda::current_device()));
    }
    return scale_tensor;
  }

  ~AGKernel() {
    cudaEventDestroy(cp_event);
    cudaEventDestroy(ready_event);
    cudaEventDestroy(all_gather_event);
  }

  auto
  get_gemm_meta(bool has_bias, bool fast_accum = false) {
    ArchEnum arch = get_arch();
    auto input_dtype = from_torch_dtype(this->input_dtype);
    auto output_dtype = from_torch_dtype(this->output_dtype);
    auto dtype_config = make_gemm_dtype_config(
        input_dtype, input_dtype, has_bias ? output_dtype : _Void{}(), output_dtype);

    auto gemm_layout = this->transpose_weight ? _RRR{}() : _RCR{}();
    UnifiedImplMeta impl_spec = None{};

    bool use_fast_accum = fast_accum and dtype_config.is_input_fp8();
    auto impl = ((int)arch < (int)_Sm90{}()) ? _GemmV2{}() : _GemmV3{}();
    if (impl == _GemmV2{}) {
      impl_spec = make_gemm_v2_meta(use_fast_accum);
    } else if (impl == _GemmV3{}) {
      impl_spec = make_gemm_v3_meta(use_fast_accum);
    }

    auto meta = make_gemm_meta(dtype_config, arch, _AGKernel{}, gemm_layout, impl, impl_spec);
    return meta;
  }

  RuntimeConfig
  get_rt_config(
      const torch::Tensor &input, const torch::Tensor &weight, c10::optional<torch::Tensor> bias) {
    CHECK_INPUT(input, this->input_dtype);
    CHECK_INPUT(weight, this->input_dtype);

    FLUX_CHECK_EQ(input.dim(), 2);
    FLUX_CHECK_EQ(weight.dim(), 2);

    if (bias.has_value()) {
      CHECK_INPUT(bias.value(), this->output_dtype);
      FLUX_CHECK_EQ(bias->dim(), 2);
      if (this->is_fp8_gemm) {
        FLUX_CHECK_EQ(1, bias->size(0));
      } else {
        FLUX_CHECK_EQ(input.size(0) * this->world_size, bias->size(0));
      }
      FLUX_CHECK_EQ(n_dim, bias->size(1));
    }

    int n = this->transpose_weight ? weight.size(1) : weight.size(0);
    int k = this->transpose_weight ? weight.size(0) : weight.size(1);

    FLUX_CHECK(n == this->n_dim) << "n-dim != expected n_dim: " << n << " vs " << this->n_dim;
    FLUX_CHECK(k == this->k_dim) << "weight k-dim mismatch: " << k << " != " << this->k_dim;

    return make_runtime_config(
        input.size(0) * this->world_size,
        n_dim,
        k_dim,
        make_all_gather_runtime_config(world_size, nnodes, (int)ring_mode));
  }

  torch::Tensor
  forward_impl(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum,
      c10::optional<UnifiedGemmHParams> const &hparams) {
    auto meta = get_gemm_meta(/*has_bias=*/bias.has_value(), /*fast_accum=*/fast_accum);
    auto rt_config = get_rt_config(input, weight, bias);

    if (hparams.has_value()) {
      this->cutlass_op = OpRegistry::instance().get_op(meta, hparams.value());
    } else {
      auto params = OpRegistry::instance().get_hparams(meta, rt_config);
      this->cutlass_op = OpRegistry::instance().get_op(meta, params);
    }

    if (this->is_fp8_gemm) {
      torch::Tensor in_scale = get_optional_scale_tensor(input_scale);
      torch::Tensor w_scale = get_optional_scale_tensor(weight_scale);
      torch::Tensor out_scale = get_optional_scale_tensor(output_scale);

      this->gemm_args = AGFP8KernelArguments{
          .m = rt_config.m(),
          .n = rt_config.n(),
          .k = rt_config.k(),
          .rank = static_cast<int>(rank),
          .world_size = static_cast<int>(world_size),
          .nnodes = static_cast<int>(nnodes),
          .alpha = 1.0f,
          .beta = 0.0f,
          .A = input.data_ptr(),
          .agA = this->input_buffer.data_ptr(),
          .B = weight.data_ptr(),
          .C = nullptr,
          .Aux = nullptr,
          .D = this->output_buffer.data_ptr(),
          .barrier_buffer = this->barrier_buffer.data_ptr(),
          .Vector = bias.has_value() ? bias->data_ptr() : nullptr,
          .abs_max_Aux = nullptr,
          .abs_max_D = nullptr,
          .scaleA = (float *)in_scale.data_ptr(),
          .scaleB = (float *)w_scale.data_ptr(),
          .scaleC = nullptr,
          .scaleD = (float *)out_scale.data_ptr(),
          .scaleAux = nullptr};
    } else {
      // AG GEMM Arguments
      this->gemm_args = AGKernelArguments{
          .m = rt_config.m(),
          .n = rt_config.n(),
          .k = rt_config.k(),
          .rank = static_cast<int>(rank),
          .world_size = static_cast<int>(world_size),
          .nnodes = static_cast<int>(nnodes),
          .alpha = 1.0f,
          .beta = bias.has_value() ? 1.0f : 0.0f,
          .input = input.data_ptr(),
          .input_buffer = this->input_buffer.data_ptr(),
          .weight = weight.data_ptr(),
          .bias = bias.has_value() ? bias->data_ptr() : nullptr,
          .output = this->output_buffer.data_ptr(),
          .barrier_buffer = this->barrier_buffer.data_ptr()};
    }

    // AG Gemm Workspace
    int64_t workspace_size = cutlass_op->get_workspace_size(this->gemm_args);
    this->lazy_init_gemm_buffer(input, workspace_size);
    this->workspace = this->gemm_buffer.defined() ? this->gemm_buffer.data_ptr() : nullptr;

    /// AG GEMM
    cudaStream_t current_stream = c10::cuda::getCurrentCUDAStream();
    if (!this->local_copy) {
      copy_local(input);
    }
    CUDA_CHECK(cudaEventRecord(this->ready_event, current_stream));

#if defined(FLUX_DEBUG)
    static bool kDebugRunGemm = get_bool_from_env("FLUX_AG_RUN_GEMM", true);
    static bool kPushMode = get_bool_from_env("FLUX_AG_CPY_PUSH", true);
#else
    static bool kDebugRunGemm = true;
    static bool kPushMode = true;
#endif

    if (ring_mode == AGRingMode::All2All) {
      /// All2All algorithm
      copy_all_to_all(input, this->cp_streams[0]);
    } else if (ring_mode == AGRingMode::RingCustom) {
      auto stream = cp_streams[0];
      at::cuda::CUDAStreamGuard guard(stream);
      CUDA_CHECK(cudaStreamWaitEvent(stream, this->ready_event));
      ring_all_gather(stream);
    } else {  // copy in ring mode. PCI-e path
      auto stream = cp_streams[0];
      at::cuda::CUDAStreamGuard guard(stream);
      CUDA_CHECK(cudaStreamWaitEvent(stream, this->ready_event));
      // always the  0 <- 1 <- 2 <- 3 <- 0 order
      if (kPushMode) {
        if (ring_mode == AGRingMode::Ring1D) {
          copy_ring_push_1d(input, stream);
        } else if (ring_mode == AGRingMode::Ring2D) {
          // TODO(houqi.1993) check performance if NUMA nodes != 2
          copy_ring_push_2d_pcie(input, stream);
        } else {
          FLUX_CHECK(false) << "only support world_size 4 and 8";
        }
      } else {
        copy_ring_pull(input, stream);
      }
    }
    CUDA_CHECK(cudaEventRecord(this->cp_event, this->cp_streams[0]));

    /// GEMM
    if (kDebugRunGemm) {
      cutlass_op->run(this->gemm_args, this->workspace, current_stream);
    } else {
      CUDA_CHECK(cudaStreamWaitEvent(current_stream, ready_event));
    }

    return this->output_buffer.slice(0, 0, input.size(0) * this->world_size);
  }

  torch::Tensor
  forward_gemm_impl(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum,
      c10::optional<UnifiedGemmHParams> const &hparams) {
    auto meta = get_gemm_meta(/*has_bias=*/bias.has_value(), /*fast_accum=*/fast_accum);
    auto rt_config = get_rt_config(input, weight, bias);

    if (hparams.has_value()) {
      this->cutlass_op = OpRegistry::instance().get_op(meta, hparams.value());
    } else {
      auto params = OpRegistry::instance().get_hparams(meta, rt_config);
      this->cutlass_op = OpRegistry::instance().get_op(meta, params);
    }

    int m = rt_config.m();
    int n = rt_config.n();
    int k = rt_config.k();

    if (this->is_fp8_gemm) {
      torch::Tensor in_scale = get_optional_scale_tensor(input_scale);
      torch::Tensor w_scale = get_optional_scale_tensor(weight_scale);
      torch::Tensor out_scale = get_optional_scale_tensor(output_scale);

      this->gemm_args = AGFP8KernelArguments{
          .m = m,
          .n = n,
          .k = k,
          .rank = static_cast<int>(rank),
          .world_size = static_cast<int>(world_size),
          .nnodes = static_cast<int>(nnodes),
          .alpha = 1.0f,
          .beta = 0.0f,
          .A = input.data_ptr(),
          .agA = this->input_buffer.data_ptr(),
          .B = weight.data_ptr(),
          .C = nullptr,
          .Aux = nullptr,
          .D = this->output_buffer.data_ptr(),
          .barrier_buffer = this->barrier_buffer.data_ptr(),
          .Vector = bias.has_value() ? bias->data_ptr() : nullptr,
          .abs_max_Aux = nullptr,
          .abs_max_D = nullptr,
          .scaleA = (float *)in_scale.data_ptr(),
          .scaleB = (float *)w_scale.data_ptr(),
          .scaleC = nullptr,
          .scaleD = (float *)out_scale.data_ptr(),
          .scaleAux = nullptr};
    } else {
      // AG GEMM Arguments
      this->gemm_args = AGKernelArguments{
          .m = m,
          .n = n,
          .k = k,
          .rank = static_cast<int>(rank),
          .world_size = static_cast<int>(world_size),
          .nnodes = static_cast<int>(nnodes),
          .alpha = 1.0f,
          .beta = bias.has_value() ? 1.0f : 0.0f,
          .input = input.data_ptr(),
          .input_buffer = this->input_buffer.data_ptr(),
          .weight = weight.data_ptr(),
          .bias = bias.has_value() ? bias->data_ptr() : nullptr,
          .output = this->output_buffer.data_ptr(),
          .barrier_buffer = this->barrier_buffer.data_ptr()};
    }

    // AG GEMM Workspace
    int64_t workspace_size = cutlass_op->get_workspace_size(this->gemm_args);
    this->lazy_init_gemm_buffer(input, workspace_size);
    this->workspace = this->gemm_buffer.defined() ? this->gemm_buffer.data_ptr() : nullptr;

    cudaStream_t current_stream = c10::cuda::getCurrentCUDAStream();
    CUDA_CHECK(cudaStreamWaitEvent(current_stream, this->all_gather_event));

    // AG GEMM
    cutlass_op->run(this->gemm_args, this->workspace, current_stream);

    // reset signal buffer
    this->reset_signals();

    return this->output_buffer.slice(0, 0, input.size(0) * this->world_size);
  }

  void
  forward_allgather(torch::Tensor input) {
    this->chunk_size = input.numel() * input.element_size();
    this->split_chunk_size = this->chunk_size / SPLIT;

    CHECK_INPUT(input, this->input_dtype);
    FLUX_CHECK_EQ(input.dim(), 2);
    FLUX_CHECK_LE(input.size(0) * this->local_world_size, this->full_m);
    FLUX_CHECK_EQ(input.size(1), this->k_dim);

    void *input_ptr = input.data_ptr();
    void *input_buffer_ptr = this->input_buffer.data_ptr();

    cudaStream_t current_stream = c10::cuda::getCurrentCUDAStream();
    CUDA_CHECK(cudaEventRecord(this->ready_event, current_stream));

    {
      at::cuda::CUDAStreamGuard guard(this->cp_streams[0]);
      auto all_gather_stream = this->cp_streams[0];
      CUDA_CHECK(cudaStreamWaitEvent(all_gather_stream, this->ready_event));

      if (!this->local_copy) {
        CUDA_CHECK(cudaMemcpyAsync(
            ptr_offset(input_buffer_ptr, this->rank * this->chunk_size),
            input_ptr,
            this->chunk_size,
            cudaMemcpyDefault,
            all_gather_stream));

        for (int j = 0; j < SPLIT; ++j) {
          set_ready(this->rank, this->rank, j, all_gather_stream);
        }
#ifdef FLUX_SHM_USE_NVSHMEM
        flux_barrier_all_on_stream(all_gather_stream);
#else
        flux_barrier_all_on_stream(all_gather_stream, this->sync_buffers, this->rank);
#endif
        CUDA_CHECK(cudaEventRecord(this->all_gather_event, all_gather_stream));
      }

      /// All2All algorithm
      {
        at::cuda::CUDAStreamGuard guard(this->cp_streams[0]);
        auto stream = cp_streams[0];
        (cudaStreamWaitEvent(stream, this->ready_event));

        for (int i = rank + 1; i < (world_size + rank); ++i) {
          auto id = i % this->world_size;
          for (int j = 0; j < SPLIT; ++j) {
            auto split_offset = j * this->split_chunk_size;
            CUDA_CHECK(cudaMemcpyAsync(
                ptr_offset(this->input_ptrs[this->rank], id * this->chunk_size + split_offset),
                ptr_offset(this->input_ptrs[id], id * this->chunk_size + split_offset),
                this->split_chunk_size,
                cudaMemcpyDefault,
                all_gather_stream));
            set_ready(this->rank, id, j, all_gather_stream);
          }
        }
      }
    }
  }

  torch::Tensor
  forward_gemm(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum) {
    this->chunk_size = input.numel() * input.element_size();
    this->split_chunk_size = this->chunk_size / SPLIT;
    return forward_gemm_impl(
        std::move(input),
        std::move(weight),
        std::move(bias),
        std::move(input_scale),
        std::move(weight_scale),
        std::move(output_scale),
        fast_accum,
        c10::nullopt);
  }

  torch::Tensor
  gemm_only(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum) {
    this->chunk_size = input.numel() * input.element_size();
    this->split_chunk_size = this->chunk_size / SPLIT;

    // set signals to 1
    this->barrier_buffer.fill_(1);
    return forward_gemm_impl(
        std::move(input),
        std::move(weight),
        std::move(bias),
        std::move(input_scale),
        std::move(weight_scale),
        std::move(output_scale),
        fast_accum,
        c10::nullopt);
  }

  torch::Tensor
  forward(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum) {
    if (this->local_copy) {
      FLUX_CHECK_EQ(input.numel() * input.element_size(), this->chunk_size);
    }

    this->chunk_size = input.numel() * input.element_size();
    this->split_chunk_size = this->chunk_size / SPLIT;
    auto result = forward_impl(
        std::move(input),
        std::move(weight),
        std::move(bias),
        std::move(input_scale),
        std::move(weight_scale),
        std::move(output_scale),
        fast_accum,
        c10::nullopt);
    this->reset_signals(); // clear the signals at the end of the forward
    return result;
  }

  torch::Tensor
  gather() {
    FLUX_CHECK_DIV(this->chunk_size, (this->n_dim * c10::elementSize(this->input_dtype)));
    return this->input_buffer.slice(
        0, 0, this->chunk_size / (this->n_dim * c10::elementSize(this->input_dtype)));
  }

  void
  reset_signals() {
    auto current_stream = c10::cuda::getCurrentCUDAStream();
    cudaStreamWaitEvent(current_stream, this->cp_event);
#ifdef FLUX_SHM_USE_NVSHMEM
    flux_barrier_all_on_stream(current_stream);
#else
    flux_barrier_all_on_stream(current_stream, this->sync_buffers, this->rank);
#endif

    this->barrier_buffer.zero_();
  }

  void
  copy_local(const torch::Tensor &input) {
    this->chunk_size = input.numel() * input.element_size();
    this->split_chunk_size = this->chunk_size / SPLIT;
    cudaStream_t current_stream = c10::cuda::getCurrentCUDAStream();
    void *input_ptr = input.data_ptr();
    void *input_buffer_ptr = this->input_buffer.data_ptr();

    CUDA_CHECK(cudaMemcpyAsync(
        ptr_offset(input_buffer_ptr, this->rank * chunk_size),
        input_ptr,
        this->chunk_size,
        cudaMemcpyDefault,
        current_stream));

    for (int j = 0; j < SPLIT; ++j) {
      set_ready(this->rank, this->rank, j, current_stream);
    }
#ifdef FLUX_SHM_USE_NVSHMEM
    flux_barrier_all_on_stream(current_stream);
#else
    flux_barrier_all_on_stream(current_stream, this->sync_buffers, this->rank);
#endif
  }

  torch::Tensor
  profiling(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum,
      c10::intrusive_ptr<ProfilingContext> opt_ctx) {
#ifdef FLUX_SHM_USE_NVSHMEM
    auto meta = unify_type(this->get_gemm_meta(bias.has_value(), fast_accum));
    auto rt_config = get_rt_config(input, weight, bias);

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
          nvshmemx_barrier_all_on_stream(stream);
          c10::cuda::stream_synchronize(stream);

          for (int iter = 0; iter < warmup + iters; ++iter) {
            this->reset_signals();
            this->copy_local(input);
            GpuTimer timer;
            timer.start(stream);
            auto output [[maybe_unused]] = this->forward_impl(
                input, weight, bias, input_scale, weight_scale, output_scale, fast_accum, hparams);
            timer.stop();
            if (iter >= warmup) {
              total_elapsed += timer.elapsed_millis();
            }
          }

          // Avoid GPU frequency adjustment
          nvshmemx_barrier_all_on_stream(stream);
          c10::cuda::stream_synchronize(stream);
          sleep(1);

          float avg_elapsed = int(total_elapsed / iters * 1000) / 1000.0;
          elapsed_tensor.copy_(torch::full({}, avg_elapsed));

          nvshmemx_float_max_reduce_on_stream(
              NVSHMEM_TEAM_WORLD,
              static_cast<float *>(reduced_elapsed_tensor.data_ptr()),
              static_cast<float const *>(elapsed_tensor.data_ptr()),
              1,
              stream);

          float reduce_elapsed = reduced_elapsed_tensor.item().toFloat();
          ctx->add(meta, rt_config, hparams, reduce_elapsed);
        },
        meta);

    auto best_hparams = ctx->record_best(meta, rt_config);
    return this->forward_impl(
        std::move(input),
        std::move(weight),
        std::move(bias),
        std::move(input_scale),
        std::move(weight_scale),
        std::move(output_scale),
        fast_accum,
        std::move(best_hparams));
#else
    // only support profiling when nvshmem is enabled
    assert(false);
    return torch::Tensor();
#endif
  }

 private:
  void
  set_ready(int rank_, int segment, int split_index, cudaStream_t stream) {
    CU_CHECK(CUStreamWriteValue(
        stream,
        (CUdeviceptr)(barrier_ptrs[rank_] + (segment * SPLIT + split_index)),
        1,
        CU_STREAM_WRITE_VALUE_DEFAULT));
  }

  void
  wait_ready(int rank_, int segment, int split_index, cudaStream_t stream) {
    CU_CHECK(CUStreamWaitValue(
        stream,
        (CUdeviceptr)(barrier_ptrs[rank_] + (segment * SPLIT + split_index)),
        1,
        CU_STREAM_WRITE_VALUE_DEFAULT));
  }


  void
  copy_all_to_all(torch::Tensor input, at::cuda::CUDAStream stream) {
    at::cuda::CUDAStreamGuard guard(stream);
    CUDA_CHECK(cudaStreamWaitEvent(stream, this->ready_event));

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
        set_ready(this->rank, id, j, stream);
      }
    }
  }

  void
  copy_ring_pull(torch::Tensor input, at::cuda::CUDAStream stream) {
    // barrier_ptrs[rank, segment, split] means rank data is ready
    at::cuda::CUDAStreamGuard guard(stream);
    CUDA_CHECK(cudaStreamWaitEvent(stream, this->ready_event));
    // always the  0 <- 1 <- 2 <- 3 <- 0 order
    int from_rank = (rank + 1) % this->world_size;  // always copy to rank next
    for (int i = 0; i < world_size - 1; i++) {
      int recv_segment = (rank + i + 1) % world_size;  // copy from self
      for (int j = 0; j < SPLIT; ++j) {
        auto split_offset = j * split_chunk_size;
        if (i != 0) {
          // previous rank recv done
          wait_ready(from_rank, recv_segment, j, stream);
        }
        CUDA_CHECK(cudaMemcpyAsync(
            ptr_offset(this->input_ptrs[rank], recv_segment * chunk_size + split_offset),
            ptr_offset(this->input_ptrs[from_rank], recv_segment * chunk_size + split_offset),
            split_chunk_size,
            cudaMemcpyDeviceToDevice,
            stream));
        set_ready(this->rank, recv_segment, j, stream);
      }
    }
  }

  void
  copy_ring_push_1d(torch::Tensor input, at::cuda::CUDAStream stream) {
    // always the  0 <- 1 <- 2 <- 3 <- 0 order
    int to_rank = (rank - 1 + world_size) % world_size;  // always recv data from rank prev
    for (int i = 0; i < world_size - 1; i++) {
      int send_segment = (rank + i) % world_size;
      for (int j = 0; j < SPLIT; ++j) {
        auto split_offset = j * split_chunk_size;
        if (i != 0) {  // previous rank recv done. i == 0 it is always ready
          wait_ready(this->rank, send_segment, j, stream);
        }
        CUDA_CHECK(cudaMemcpyAsync(
            ptr_offset(this->input_ptrs[to_rank], send_segment * chunk_size + split_offset),
            ptr_offset(this->input_ptrs[rank], send_segment * chunk_size + split_offset),
            this->split_chunk_size,
            cudaMemcpyDeviceToDevice,
            stream));
        set_ready(to_rank, send_segment, j, stream);
      }
    }
  }

  void
  copy_ring_push_2d_pcie(torch::Tensor input, at::cuda::CUDAStream stream) {
  }

  void
  ring_all_gather(at::cuda::CUDAStream stream) {
    ring_all_gather_reorder<0, FLUX_AG_RING_ORDER>(stream);
  }

  template <int reordered, typename Transfer, typename... Transfers>
  void
  ring_all_gather_reorder(at::cuda::CUDAStream stream) {
    if (Transfer::exec_rank == this->rank) {
      ring_all_gather_run<Transfer::src_idx, Transfer::dst_idx, Transfer, Transfers...>(stream);
    } else {
      if constexpr (reordered + 1 <= sizeof...(Transfers)) {
        ring_all_gather_reorder<reordered + 1, Transfers..., Transfer>(stream);
      }
    }
  }

  template <int src_rank, int dst_rank, typename Transfer, typename... Transfers>
  void
  ring_all_gather_run(at::cuda::CUDAStream stream) {
    Transfer::run(
        input_ptrs, barrier_ptrs, src_rank, dst_rank, chunk_size, split_chunk_size, stream);
    ring_all_gather_run<src_rank, dst_rank, Transfers...>(stream);
  }

  template <int src_rank, int dst_rank>
  void
  ring_all_gather_run(at::cuda::CUDAStream stream) {}
};

void
init_ag_kernel_ops(py::module &m) {}
static int _register_ag_kernel_ops [[maybe_unused]] = []() {
  ThsOpsInitRegistry::instance().register_one("all_gather_gemm_kernel", [](py::module &m) {
    py::enum_<AGRingMode>(m, "AGRingMode", py::arithmetic())
        .value("Auto", AGRingMode::Auto)
        .value("All2All", AGRingMode::All2All)
        .value("Ring1D", AGRingMode::Ring1D)
        .value("Ring2D", AGRingMode::Ring2D);
    py::class_<AGKernel>(m, "AGKernel")
        .def(
            py::init([](c10::intrusive_ptr<c10d::ProcessGroup> tp_group,
                        int32_t nnodes,
                        int32_t full_m,
                        int32_t n_dim,
                        int32_t k_dim,
                        py::object py_input_dtype,
                        py::object py_output_dtype,
                        bool transpose_weight,
                        bool local_copy,
                        AGRingMode ring_mode) {
              auto input_dtype = torch::python::detail::py_object_to_dtype(py_input_dtype);
              auto output_dtype = py_output_dtype.is(py::none())
                                      ? input_dtype
                                      : torch::python::detail::py_object_to_dtype(py_output_dtype);
              return new AGKernel(
                  tp_group,
                  nnodes,
                  full_m,
                  n_dim,
                  k_dim,
                  input_dtype,
                  output_dtype,
                  transpose_weight,
                  local_copy,
                  ring_mode);
            }),
            py::arg("tp_group"),
            py::arg("nnodes"),
            py::arg("full_m"),
            py::arg("n_dim"),
            py::arg("k_dim"),
            py::arg("input_dtype"),
            py::arg("output_dtype") = py::none(),
            py::arg("transpose_weight") = true,
            py::arg("local_copy") = false,
            py::arg("ring_mode") = AGRingMode::Auto)
        .def("forward_allgather", &AGKernel::forward_allgather, py::arg("input"))
        .def(
            "forward_gemm",
            &AGKernel::forward_gemm,
            py::arg("input"),
            py::arg("weight"),
            py::arg("bias") = py::none(),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none(),
            py::arg("output_scale") = py::none(),
            py::arg("fast_accum") = false)
        .def(
            "gemm_only",
            &AGKernel::gemm_only,
            py::arg("input"),
            py::arg("weight"),
            py::arg("bias") = py::none(),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none(),
            py::arg("output_scale") = py::none(),
            py::arg("fast_accum") = false)
        .def(
            "forward",
            &AGKernel::forward,
            py::arg("input"),
            py::arg("weight"),
            py::arg("bias") = py::none(),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none(),
            py::arg("output_scale") = py::none(),
            py::arg("fast_accum") = false)
        .def("gather", &AGKernel::gather)
        .def("reset_signals", &AGKernel::reset_signals)
        .def("copy_local", &AGKernel::copy_local, py::arg("input"))
        .def(
            "profiling",
            &AGKernel::profiling,
            py::arg("input"),
            py::arg("weight"),
            py::arg("bias") = py::none(),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none(),
            py::arg("output_scale") = py::none(),
            py::arg("fast_accum") = false,
            py::arg("prof_ctx") = nullptr);
  });
  return 0;
}();

}  // namespace ths_op
}  // namespace flux
}  // namespace bytedance
