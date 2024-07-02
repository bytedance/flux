//===- gemm_reduce_scatter.cc ------------------------------------- C++ ---===//
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

#include "c10/cuda/CUDAGuard.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_meta.h"
#include "flux/runtime_config.h"
#include "flux/ths_op/ths_op.h"
#include "flux/cuda/cuda_common.h"
#include "flux/flux.h"
#include "flux/op_registry.h"
#include "flux/ths_op/util.h"
#include "flux/args/reduce_scatter.h"
#include <ATen/core/List.h>
#include <ATen/ops/empty.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/util/Optional.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/logging_is_not_google_glog.h>
#include <cuda_runtime_api.h>
#include <ATen/core/jit_type.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAEvent.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/python.h>
#include "flux/utils.h"
#include "reduce_scatter/ths_op/helper_ops.h"
#include "reduce_scatter/reduce_scatter_barrier_struct.hpp"
#include "flux/ths_op/topo_utils.h"
#include "torch/serialize.h"
#ifdef FLUX_REDUCE_SCATTERT_WITH_NCCL
#include "nccl.h"
#endif

#ifdef FLUX_SHM_USE_NVSHMEM
#include "nvshmemx.h"
#endif
namespace bytedance::flux::ths_op {
using torch::Tensor;

class GemmRS : public torch::CustomClassHolder {
 private:
  c10::intrusive_ptr<c10d::ProcessGroup> tp_group;
  const int32_t nnodes;
  const int32_t max_m;
  const int32_t n_dim;
  const c10::ScalarType input_dtype;
  const c10::ScalarType output_dtype;
  const bool transpose_weight;
  const bool fuse_reduction;

 private:
  const int32_t rank;
  const int32_t world_size;
  const int32_t local_world_size;
  const int32_t local_rank;
  const int32_t node_idx;

 private:
  // Symmetrically distributed tensor
  std::vector<torch::Tensor> output_buffers;
  std::vector<torch::Tensor> reduce_buffers;
  std::vector<torch::Tensor> barrier_buffers;
#ifndef FLUX_SHM_USE_NVSHMEM
  // used for the cuda-ipc-barrier
  std::vector<torch::Tensor> sync_buffers;
#endif
  torch::Tensor output_buffer;
  torch::Tensor reduce_buffer;
  torch::Tensor barrier_buffer;
  torch::Tensor gemm_buffer;
  std::vector<void *> output_scatter_ptrs;
  std::vector<void *> barrier_ptrs;
  bool no_nvlink;
  int sub_world_size;
  c10::cuda::CUDAStream rs_stream_;
  cudaEvent_t event_;
  bool use_1d_ring;
  bool use_p2p_read;
  const bool is_fp8_gemm;

#ifdef FLUX_REDUCE_SCATTERT_WITH_NCCL
  ncclComm_t nccl_comm;
#endif
  void
  init_output_buffer() {
    // update max_m and allocate buffer
    if (get_arch() == _Sm90{} || no_nvlink || (get_arch() == _Sm80{} && nnodes > 1)) {
      int reduce_m_dim = (get_arch() == _Sm90{})
                             ? (max_m + world_size - 1) / world_size * nnodes * nnodes
                             : max_m;
      this->reduce_buffers =
          flux_create_tensor_list({reduce_m_dim, n_dim}, output_dtype, this->tp_group);
      this->reduce_buffer = this->reduce_buffers[this->local_rank];
    }
    if (get_arch() == _Sm80{} && nnodes > 1 && from_torch_dtype(this->input_dtype) == _BF16{}) {
      // SM80 does not support the fuse reduction for the bfloat16 data type
      // we have to use the float32 global_red instruction when SM80 && nnodes>1 && input_type=bf16
      // Therefore, in this case, here double the size of the output_buffer.
      this->output_buffers =
          flux_create_tensor_list({max_m * 2, n_dim}, output_dtype, this->tp_group);
    } else {
      this->output_buffers = flux_create_tensor_list({max_m, n_dim}, output_dtype, this->tp_group);
    }
    this->output_buffer = this->output_buffers[this->local_rank];
    for (int i = 0; i < world_size; ++i) {
      if (i / this->local_world_size == rank / this->local_world_size) {
        output_scatter_ptrs[i] = this->output_buffers[i % this->local_world_size].data_ptr();
        // only check for ranks on the same node
        TORCH_CHECK(
            output_scatter_ptrs[i] != nullptr, "nullptr buffr of rank " + std::to_string(i));
      } else {
        output_scatter_ptrs[i] = nullptr;
      }
    }
#ifndef FLUX_SHM_USE_NVSHMEM
    this->sync_buffers =
        flux_create_tensor_list({this->world_size}, c10::ScalarType::Int, this->tp_group);
    this->sync_buffers[this->rank].zero_();  // zeros the sync buffer for cuda ipc at the start
#endif
  }

  void
  lazy_init_barrier_buffer(int64_t buffer_size) {
    if ((buffer_size == 0) ||
        (barrier_buffer.defined() && buffer_size <= barrier_buffer.numel())) {
      return;
    }
    this->barrier_buffers =
        flux_create_tensor_list({buffer_size}, c10::ScalarType::Byte, this->tp_group);
    this->barrier_buffer = this->barrier_buffers[this->local_rank];
    for (int i = 0; i < world_size; ++i) {
      if (i / this->local_world_size == rank / this->local_world_size) {
        barrier_ptrs[i] = this->barrier_buffers[i % this->local_world_size].data_ptr();
        // only check for ranks on the same node
        TORCH_CHECK(barrier_ptrs[i] != nullptr, "nullptr buffr of rank " + std::to_string(i));
      } else {
        barrier_ptrs[i] = nullptr;
      }
    }
  }

  bool
  has_nvlink() {
    return true;
  }

  bool
  use_1d_ring_or_not() {
    ensure_nvml_init();
    int devid = at::cuda::current_device();
    std::string devname(get_gpu_device_name(devid));
    if (devname != "NVIDIA L20" && world_size == 8) {
      return false;
    }
    return true;
  }

  bool
  use_p2p_read_or_not() {
    ensure_nvml_init();
    int devid = at::cuda::current_device();
    std::string devname(get_gpu_device_name(devid));
    if (devname != "NVIDIA L20") {
      return true;
    }
    return false;
  }

  void
  lazy_init_gemm_buffer(torch::Tensor input, int64_t buffer_size) {
    if (buffer_size <= 0) {
      return;
    }
    buffer_size = (buffer_size + 127) / 128 * 128;
    if (!this->gemm_buffer.defined() || buffer_size > this->gemm_buffer.numel()) {
      auto options = input.options().dtype(c10::ScalarType::Byte);
      this->gemm_buffer = torch::empty({buffer_size}, options);
    }
  }

  c10::cuda::CUDAStream
  CreateReduceScatterStream() {
    at::cuda::CUDAGuard guard(at::cuda::current_device());
    cudaStream_t rs_stream = nullptr;
    int least_priority, greatest_priority;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
    CUDA_CHECK(cudaStreamCreateWithPriority(&rs_stream, cudaStreamNonBlocking, greatest_priority));
    return at::cuda::getStreamFromExternal(rs_stream, at::cuda::current_device());
  }

 public:
  GemmRS(
      c10::intrusive_ptr<c10d::ProcessGroup> tp_group_,
      int32_t nnodes,
      int32_t max_m,
      int32_t n_dim,
      c10::ScalarType input_dtype,
      c10::ScalarType output_dtype,
      bool transpose_weight,
      bool fuse_reduction)
      : tp_group(tp_group_),
        nnodes(nnodes),
        max_m(max_m),
        n_dim(n_dim),
        input_dtype(input_dtype),
        output_dtype(output_dtype),
        transpose_weight(transpose_weight),
        fuse_reduction(fuse_reduction),
        rank(tp_group->getRank()),
        world_size(tp_group->getSize()),
        local_world_size(world_size / nnodes),
        local_rank(rank % local_world_size),
        node_idx(rank / local_world_size),
        output_scatter_ptrs(world_size, nullptr),
        barrier_ptrs(world_size, nullptr),
        no_nvlink(!has_nvlink()),
        rs_stream_(CreateReduceScatterStream()),  // private stream. never dup with gemm stream
        use_1d_ring(use_1d_ring_or_not()),
        use_p2p_read(use_p2p_read_or_not()),
        is_fp8_gemm(is_fp8_torch_dtype(input_dtype)) {
    TORCH_CHECK(
        rank >= 0 && rank < world_size,
        "invalid rank: " + std::to_string(rank) +
            " and world_size: " + std::to_string(world_size));
    TORCH_CHECK(
        world_size % nnodes == 0,
        "invalid nnodes: world_size[" + std::to_string(world_size) + "] % nnodes[" +
            std::to_string(nnodes) + "] != 0");
    TORCH_CHECK(
        !fuse_reduction || input_dtype == at::ScalarType::Half,
        "Fuse reduction only support float16 type on SM80 due to instruction limitation.");
    this->init_output_buffer();
    CUDA_CHECK(cudaEventCreate(&event_));
#if defined(FLUX_DEBUG)
    if (no_nvlink) {
      LOG(WARNING) << "NvLink is not supported, seems running on a PCI-e machine.";
      ensure_nvml_init();
      int devid = at::cuda::current_device();
      std::string devname(get_gpu_device_name(devid));
      if (devname != "NVIDIA A100 80GB PCIe" && devname != "NVIDIA A800 80GB PCIe") {
        LOG(WARNING) << "Only NVIDIA A100/A800 80GB PCIe is tuned for. got " << devname;
      }
      if (world_size > 4 && world_size != 8) {
        LOG(WARNING) << "Only TensorParallel = 4 or 8 is tuned for. got " << world_size;
      }
      unsigned int gen = get_pcie_gen(devid);
      if (gen != 4) {
        LOG(WARNING) << "only PCI-e 4 version is tuned for. got PCI-e " << gen;
      }
    }
#endif
#ifdef FLUX_REDUCE_SCATTERT_WITH_NCCL
    if (nnodes > 1 && no_nvlink) {
      nccl_comm = topo_utils::create_nccl_comm_with_processgroup(tp_group);
    } else {
      nccl_comm = nullptr;
    }
#endif
  }

  ~GemmRS() {
    CUDA_CHECK(cudaEventDestroy(event_));
    CUDA_CHECK(cudaStreamDestroy(rs_stream_));
#ifdef FLUX_REDUCE_SCATTERT_WITH_NCCL
    if (nccl_comm) {
      NCCL_CHECK(ncclCommDestroy(nccl_comm));
    }
#endif
  }

  auto
  get_gemm_meta(bool has_bias, bool fast_accum = false) {
    ArchEnum arch = get_arch();
    auto gemm_layout = transpose_weight ? _RRR{}() : _RCR{}();
    auto input_dtype = from_torch_dtype(this->input_dtype);
    auto output_dtype = from_torch_dtype(this->output_dtype);
    auto dt_conf = make_gemm_dtype_config(
        input_dtype, input_dtype, has_bias ? output_dtype : _Void{}(), output_dtype);

    fast_accum = fast_accum & dt_conf.is_input_fp8();
    bool is_gemm_v2 = ((int)arch < (int)_Sm90{}());
    auto meta = make_gemm_meta(
        dt_conf,
        arch,
        _ReduceScatter{},
        gemm_layout,
        is_gemm_v2 ? _GemmV2{}() : _GemmV3{}(),
        is_gemm_v2 ? UnifiedImplMeta(make_gemm_v2_meta(fast_accum))
                   : UnifiedImplMeta(make_gemm_v3_meta(fast_accum)),
        make_reduce_scatter_meta(
            this->fuse_reduction,
            nnodes > 1        ? _AcrossNode{}()
            : this->no_nvlink ? _IntraNodePcie{}()
                              : _IntraNode{}()));
    return meta;
  }

  RuntimeConfig
  get_rt_conf(torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias) {
    CHECK_INPUT(input, this->input_dtype);
    CHECK_INPUT(weight, this->input_dtype);
    TORCH_CHECK(input.dim() == 2, "input dim is not 2");
    TORCH_CHECK(weight.dim() == 2, "weight dim is not 2");
    int32_t m = input.size(0);
    int32_t k = input.size(1);
    int32_t n = transpose_weight ? weight.size(1) : weight.size(0);

    if (bias.has_value()) {
      CHECK_INPUT(bias.value(), this->output_dtype);
      TORCH_CHECK(bias->dim() == 2, "bias dim is not 2");
      TORCH_CHECK(
          m == bias->size(0),
          "bias dim0 != m: " + std::to_string(bias->size(0)) + " vs " + std::to_string(m));
      TORCH_CHECK(
          n == bias->size(1),
          "bias dim1 != n: " + std::to_string(bias->size(1)) + " vs " + std::to_string(n));
    }

    // row major for streamk, todo: make weight layout an option
    int32_t wk = transpose_weight ? weight.size(0) : weight.size(1);
    FLUX_CHECK_LE(m, this->max_m) << "m-dim greater than maximum possible value";
    FLUX_CHECK_EQ(n, this->n_dim) << "n-dim != expected n_dim";
    FLUX_CHECK_EQ(wk, k) << "weight k-dim mismatch";
    return make_runtime_config(m, n, k, make_reduce_scatter_runtime_config(world_size, nnodes));
  }

  void
  forward_gemm_impl(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum,
      c10::optional<UnifiedGemmHParams> const &hparams) {
    auto meta = get_gemm_meta(/*has_bias=*/bias.has_value(), fast_accum);
    auto rt_conf = get_rt_conf(input, weight, bias);
    // get cutlass op
    OpRegistry::OpPtr cutlass_op;
    if (hparams.has_value()) {
      cutlass_op = OpRegistry::instance().get_op(meta, hparams.value());
    } else {
      cutlass_op = OpRegistry::instance().get_op(meta, rt_conf);
    }

    // TODO(houqi.1993) using args instead of envs
    static int num_blocks = get_int_from_env("FLUX_RS_BLOCKS", 12);
    static bool use_barrier_queue = get_bool_from_env("FLUX_RS_USE_BARRIER_QUEUE", false);
    static bool use_gemmk = get_bool_from_env("FLUX_RS_USE_GEMMK", no_nvlink);
    static bool use_cudaMemcpyAsync = get_bool_from_env("FLUX_RS_USE_CUDA_MEMCPY_ASYNC", false);
    static int n_split = get_int_from_env("FLUX_RS_N_SPLIT", 1);
    static bool per_tile_flags = get_bool_from_env("FLUX_RS_PER_TILE_FLAGS", no_nvlink);
    ReduceScatterArguments reduce_scatter_args{
        .reduce_scatter_num_blocks = num_blocks,
        .rs_stream = rs_stream_,
        .event = event_,
        .use_barrier_queue = use_barrier_queue,
        .use_gemmk = use_gemmk,
        .per_tile_flags = per_tile_flags,
        .use_cudaMemcpyAsync = use_cudaMemcpyAsync,
        .n_split = n_split,
        .sub_world_size = this->sub_world_size,
#ifdef FLUX_REDUCE_SCATTERT_WITH_NCCL
        .opaque = nccl_comm,
#else
        .opaque = nullptr,
#endif
        .use_1d_ring = use_1d_ring,
        .use_p2p_read = use_p2p_read,
    };
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    if (!is_fp8_gemm) {
      FLUX_CHECK(!input_scale.has_value());
      FLUX_CHECK(!weight_scale.has_value());
      FLUX_CHECK(!output_scale.has_value());
      const GemmReduceScatterArguments args{
          .m = rt_conf.m(),
          .n = rt_conf.n(),
          .k = rt_conf.k(),
          .rank = static_cast<int>(this->rank),
          .world_size = static_cast<int>(this->world_size),
          .nnodes = static_cast<int>(this->nnodes),
          .alpha = 1.0f,
          .beta = bias.has_value() ? 1.0f : 0.0f,
          .input = input.data_ptr(),
          .weight = weight.data_ptr(),
          .bias = bias.has_value() ? bias->data_ptr() : nullptr,
          .output_scatter_ptrs = this->output_scatter_ptrs.data(),
          .local_reduce_buffer =
              this->reduce_buffer.defined() ? this->reduce_buffer.data_ptr() : nullptr,
          .barrier_ptrs = this->barrier_ptrs.data(),
          .avail_sms = no_nvlink ? 1 : -1,
          .reduce_scatter_args = reduce_scatter_args};

      // initialize workspace
      int64_t workspace_size = cutlass_op->get_workspace_size(args);
      this->lazy_init_gemm_buffer(input, workspace_size);
      void *workspace = this->gemm_buffer.defined() ? this->gemm_buffer.data_ptr() : nullptr;

      // initialize barrier workspace
      int64_t barrier_workspace_size = cutlass_op->get_barrier_workspace_size(args);
      // * 8 is for corner case reduce_scatter tiles. never mind this won't be a large memory
      barrier_workspace_size = barrier_workspace_size / sizeof(int) * sizeof(PerTileFlags) * 8;
      this->lazy_init_barrier_buffer(barrier_workspace_size);

      if ((fuse_reduction && !(meta.arch() == _Sm90{})) || this->no_nvlink) {
        // need to zero buffers;
        zero_buffers();
      }
      cutlass_op->run(args, workspace, stream);

    } else {
      GemmReduceScatterFp8Arguments fp8_args{
          .m = rt_conf.m(),
          .n = rt_conf.n(),
          .k = rt_conf.k(),
          .rank = static_cast<int>(this->rank),
          .world_size = static_cast<int>(this->world_size),
          .nnodes = static_cast<int>(this->nnodes),
          .alpha = 1.0f,
          .beta = bias.has_value() ? 1.0f : 0.0f,
          .input = input.data_ptr(),
          .weight = weight.data_ptr(),
          .bias = bias.has_value() ? bias->data_ptr() : nullptr,
          .output_scatter_ptrs = this->output_scatter_ptrs.data(),
          .local_reduce_buffer =
              this->reduce_buffer.defined() ? this->reduce_buffer.data_ptr() : nullptr,
          .barrier_ptrs = this->barrier_ptrs.data(),
          .avail_sms = no_nvlink ? 1 : -1,
          .reduce_scatter_args = reduce_scatter_args,
          .Aux = nullptr,
          .Vector = bias.has_value() ? bias->data_ptr() : nullptr,
          .abs_max_Aux = nullptr,
          .abs_max_D = nullptr,
          .scaleA = (float *)(input_scale.has_value() ? input_scale->data_ptr() : nullptr),
          .scaleB = (float *)(weight_scale.has_value() ? weight_scale->data_ptr() : nullptr),
          .scaleC = nullptr,
          .scaleD = (float *)(output_scale.has_value() ? output_scale->data_ptr() : nullptr),
          .scaleAux = nullptr};

      // initialize workspace
      int64_t workspace_size = cutlass_op->get_workspace_size(fp8_args);
      this->lazy_init_gemm_buffer(input, workspace_size);
      void *workspace = this->gemm_buffer.defined() ? this->gemm_buffer.data_ptr() : nullptr;

      // initialize barrier workspace
      int64_t barrier_workspace_size = cutlass_op->get_barrier_workspace_size(fp8_args);
      // * 8 is for corner case reduce_scatter tiles. never mind this won't be a large memory
      barrier_workspace_size = barrier_workspace_size / sizeof(int) * sizeof(PerTileFlags) * 8;
      this->lazy_init_barrier_buffer(barrier_workspace_size);

      if ((fuse_reduction && !(meta.arch() == _Sm90{})) || this->no_nvlink) {
        // need to zero buffers;
        zero_buffers();
      }
      cutlass_op->run(fp8_args, workspace, stream);
    }

  }  // namespace ths_op

  torch::Tensor
  forward_reduce_scatter_impl(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<UnifiedGemmHParams> hparams) {
    auto meta = get_gemm_meta(/*has_bias=*/bias.has_value());  // fast_accum doesn't matter
    auto rt_conf = get_rt_conf(input, weight, bias);

    // get cutlass op
    OpRegistry::OpPtr cutlass_op;
    if (hparams.has_value()) {
      cutlass_op = OpRegistry::instance().get_op(meta, hparams.value());
    } else {
      cutlass_op = OpRegistry::instance().get_op(meta, rt_conf);
    }

    int m = rt_conf.m();
    int n = rt_conf.n();

    if (((int)get_arch() < (int)_Sm90{}())) {
      auto full_output = this->output_buffer.slice(0, 0, m);
      if (nnodes > 1 && !no_nvlink) {
        // printf("fuse_reduction:%d \n\n", fuse_reduction);
        auto unified_hparams = cutlass_op->get_runtime_gemm_hparams();
        auto tile_shape = unified_hparams.tile_shape();
        auto [tile_M, tile_N, tile_K] = tile_shape;
        int m_rank = m / world_size;
        auto result = torch::empty({m_rank, n}, this->output_buffer.options());
        auto output_to_reduce =
            this->reduce_buffer.slice(0, 0, nnodes * m_rank).view({nnodes, m_rank, this->n_dim});
        bsr_reduce(output_to_reduce, result, tile_M, tile_N);
        return result;
        // return full_output;
      } else if (no_nvlink) {
        int m_per_rank = m / this->world_size;
        auto output_2d =
            output_buffer.slice(0, m_per_rank * this->rank, m_per_rank * (this->rank + 1));
        constexpr int kNumaWorldSize = 4;
        constexpr int kNumaNodes = 2;
        int local_world_size = world_size / nnodes;
        int local_rank = rank % local_world_size;
        int node_id = rank / local_world_size;
        int numa_id = local_rank / kNumaWorldSize;
        int rank_numa_local = local_rank % kNumaWorldSize;
        int rank_prev = (rank_numa_local - 1 + kNumaWorldSize) % kNumaWorldSize;
        rank_prev += numa_id * kNumaWorldSize + node_id * local_world_size;
        int rank_next = (rank_numa_local + 1) % kNumaWorldSize;
        rank_next += numa_id * kNumaWorldSize + node_id * local_world_size;
        int rank_from = numa_id == 0 ? rank_next : rank_prev;
        for (int i = 1; i < nnodes; i++) {
          int reduce_unused_segment = (rank_from + kNumaNodes + i * local_world_size) % world_size;
          auto segment_other_node = reduce_buffer.slice(
              0, m_per_rank * reduce_unused_segment, m_per_rank * (reduce_unused_segment + 1));
          output_2d.add_(segment_other_node);
        }
        return output_2d;
      } else {
        int local_world_size = world_size / nnodes;
        if (fuse_reduction) {
          auto length = m / world_size;
          // return this->output_buffer.slice(0, rank * length, (rank + 1) * length).unsqueeze(0);
          return this->output_buffer.slice(0, 0, length).unsqueeze(0);
        } else {
          auto output_4d = full_output.view({nnodes, local_world_size, m / world_size, n});
          auto output = output_4d.sum(1);  // (nnodes,m_rank,n)
          return output;
        }
      }
    } else if (meta.arch() == _Sm90{}) {
      int reduce_m_dim = m / world_size * nnodes * nnodes;
      auto full_output = this->reduce_buffer.slice(0, 0, reduce_m_dim);
      auto output_4d = full_output.view({nnodes, nnodes, m / world_size, n});
      if (nnodes == 1) {
        auto output = output_4d[node_idx].sum(0);  // (m_rank,n)
        return output;
      } else {
        int m_rank = m / world_size;
        auto output = torch::empty({m_rank, n}, output_buffer.options());
        auto unified_hparams = cutlass_op->get_runtime_gemm_hparams();
        auto tile_shape = unified_hparams.tile_shape();
        auto [tile_M, tile_N, tile_K] = tile_shape;
        bsr_reduce(output_4d[node_idx], output, tile_M, tile_N);
        return output;
      }
    } else {
      TORCH_CHECK(false, "unsupported arch:" + std::string(enum_to_string(meta.arch())));
    }
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
    forward_gemm_impl(
        input, weight, bias, input_scale, weight_scale, output_scale, fast_accum, hparams);
    forward_barrier(input, weight, bias);
    return forward_reduce_scatter_impl(input, weight, bias, hparams);
  }

  void
  forward_gemm(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum) {
    return forward_gemm_impl(
        std::move(input),
        std::move(weight),
        std::move(bias),
        input_scale,
        weight_scale,
        output_scale,
        fast_accum,
        c10::nullopt);
  }

  void
  forward_barrier(torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    if (get_arch() == _Sm90{} and nnodes == 1) {
      // only local reduce, skip nvshmem barrier
    } else {
#ifdef FLUX_SHM_USE_NVSHMEM
      flux_barrier_all_on_stream(stream);
#else
      flux_barrier_all_on_stream(stream, this->sync_buffers, this->rank);
#endif
    }
  }

  torch::Tensor
  forward_reduce_scatter(
      torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias) {
    return forward_reduce_scatter_impl(
        std::move(input), std::move(weight), std::move(bias), c10::nullopt);
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
    return forward_impl(
        std::move(input),
        std::move(weight),
        std::move(bias),
        std::move(input_scale),
        std::move(weight_scale),
        std::move(output_scale),
        fast_accum,
        c10::nullopt);
  }

  void
  zero_buffers() {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    if (this->output_buffer.defined()) {
      this->output_buffer.zero_();
    }
    if (this->barrier_buffer.defined()) {
      this->barrier_buffer.zero_();
    }
    if (this->reduce_buffer.defined()) {
      this->reduce_buffer.zero_();
    }
#ifdef FLUX_SHM_USE_NVSHMEM
    flux_barrier_all_on_stream(stream);
#else
    flux_barrier_all_on_stream(stream, this->sync_buffers, this->rank);
#endif
    if (!no_nvlink) {
      c10::cuda::stream_synchronize(stream);
    }
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
    auto meta = unify_type(this->get_gemm_meta(/*has_bias=*/bias.has_value(), fast_accum));
    auto rt_conf = this->get_rt_conf(input, weight, bias);
    ProfilingContext tmp_ctx("__tmp__");
    ProfilingContext *ctx = opt_ctx == nullptr ? &tmp_ctx : opt_ctx.get();

    auto filter_hparams = [&](UnifiedGemmHParams const &hparams) { return true; };

    auto elapsed_tensor = torch::empty({}, weight.options().dtype(c10::ScalarType::Float));
    auto reduced_elapsed_tensor = elapsed_tensor.clone();

    OpRegistry::instance().visit_hparams(
        [&](UnifiedGemmHParams const &hparams) {
          if (not filter_hparams(hparams)) {
            return;
          }
          // filter non-consistent hparams
          constexpr int warm_iters = 5;
          constexpr int iters = 10;
          float total_elapsed = 0;

          auto stream = c10::cuda::getCurrentCUDAStream();
          flux_barrier_all_on_stream(stream);
          c10::cuda::stream_synchronize(stream);
          for (int iter = 0; iter < warm_iters + iters; ++iter) {
            GpuTimer timer;
            timer.start(stream);
            auto output [[maybe_unused]] = this->forward_impl(
                input, weight, bias, input_scale, weight_scale, output_scale, fast_accum, hparams);
            timer.stop();
            if (iter >= warm_iters) {
              total_elapsed += timer.elapsed_millis();
            }
          }

          // Avoid GPU frequency adjustment
          flux_barrier_all_on_stream(stream);
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
          ctx->add(meta, rt_conf, hparams, reduce_elapsed);
        },
        meta);

    auto best_hparams = ctx->record_best(meta, rt_conf);
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
    FLUX_CHECK(false) << "only support profiling when nvshmem is enabled";
    return torch::Tensor();
#endif
  }

  void
  _ensure_topo_initialized() {
    if (!topo_utils::is_topo_initialized()) {
      topo_utils::initialize_topo(this->tp_group);
    }
  }
};  // namespace flux

namespace py = pybind11;

static int _register_gemm_rs_ops [[maybe_unused]] = []() {
  ThsOpsInitRegistry::instance().register_one("gemm_reduce_scatter", [](py::module &m) {
    py::class_<GemmRS>(m, "GemmRS")
        .def(
            py::init([](c10::intrusive_ptr<c10d::ProcessGroup> tp_group,
                        int32_t nnodes,
                        int32_t max_m,
                        int32_t n_dim,
                        py::object py_input_dtype,
                        py::object py_output_dtype,
                        bool transpose_weight,
                        bool fuse_reduction) {
              auto input_dtype = torch::python::detail::py_object_to_dtype(py_input_dtype);
              auto output_dtype = py_output_dtype.is(py::none())
                                      ? input_dtype
                                      : torch::python::detail::py_object_to_dtype(py_output_dtype);

              return new GemmRS(
                  tp_group,
                  nnodes,
                  max_m,
                  n_dim,
                  input_dtype,
                  output_dtype,
                  transpose_weight,
                  fuse_reduction);
            }),
            py::arg("tp_group"),
            py::arg("nnodes"),
            py::arg("max_m"),
            py::arg("n_dim"),
            py::arg("input_dtype"),
            py::arg("output_dtype") = py::none(),
            py::arg("transpose_weight") = false,
            py::arg("fuse_reduction") = false)
        .def("zero_buffers", &GemmRS::zero_buffers)
        .def(
            "forward",
            &GemmRS::forward,
            py::arg("input"),
            py::arg("weight"),
            py::arg("bias") = py::none(),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none(),
            py::arg("output_scale") = py::none(),
            py::arg("fast_accum") = false)
        .def(
            "forward_gemm",
            &GemmRS::forward_gemm,
            py::arg("input"),
            py::arg("weight"),
            py::arg("bias") = py::none(),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none(),
            py::arg("output_scale") = py::none(),
            py::arg("fast_accum") = false)
        .def(
            "forward_barrier",
            &GemmRS::forward_barrier,
            py::arg("input"),
            py::arg("weight"),
            py::arg("bias") = py::none())
        .def(
            "forward_reduce_scatter",
            &GemmRS::forward_reduce_scatter,
            py::arg("input"),
            py::arg("weight"),
            py::arg("bias") = py::none())
        .def(
            "profiling",
            &GemmRS::profiling,
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

}  // namespace bytedance::flux::ths_op
