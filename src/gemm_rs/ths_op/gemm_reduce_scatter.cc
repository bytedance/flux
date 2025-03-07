//===- gemm_reduce_scatter.cc ------------------------------------- C++ ---===//
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
#include "gemm_rs/ths_op/gemm_reduce_scatter.h"
#include "flux/args/gemm_rs.h"
#include "flux/cuda/cuda_common.h"
#include "flux/flux.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_meta.h"
#include "flux/op_registry.h"
#include "flux/runtime_config.h"
#include "flux/ths_op/flux_shm.h"
#include "flux/ths_op/ths_op.h"
#include "flux/ths_op/topo_utils.h"
#include "flux/ths_op/util.h"
#include "flux/utils.h"
#include "gemm_rs/reduce_scatter_barrier_struct.hpp"
#include "gemm_rs/tile_scheduler/threadblock_swizzle_segment_util.hpp"
#include "gemm_rs/ths_op/helper_ops.h"
#include <ATen/core/jit_type.h>
#include <ATen/core/List.h>
#include <ATen/cuda/CachingHostAllocator.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/ops/empty.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/logging_is_not_google_glog.h>
#include <c10/util/Optional.h>
#include <cstdio>
#include <optional>
#include <cuda_runtime_api.h>
#include "coll/ths_op/reduce_scatter_op.h"
#ifdef FLUX_REDUCE_SCATTER_WITH_NCCL
#include "nccl.h"
#endif

namespace bytedance::flux::ths_op {
using torch::Tensor;

std::string
to_string(const std::optional<bool> &value) {
  if (value) {
    return value.value() ? "true" : "false";
  } else {
    return "nullopt";
  }
}
std::string
to_string(const std::optional<int> &vlaue) {
  if (vlaue) {
    return std::to_string(vlaue.value());
  } else {
    return "nullopt";
  }
}
std::string
to_string(const ReduceScatterOptionWithOptional &opt) {
  std::stringstream ss;
  ss << "ReduceScatterOptionWithOptional{";
  ss << "use_barrier_queue: " << to_string(opt.use_barrier_queue) << ", ";
  ss << "use_1d_ring: " << to_string(opt.use_1d_ring) << ", ";
  ss << "use_p2p_read: " << to_string(opt.use_p2p_read) << ", ";
  ss << "use_cudaMemcpyAsync: " << to_string(opt.use_cudaMemcpyAsync) << ", ";
  ss << "use_gemmk: " << to_string(opt.use_gemmk) << ", ";
  ss << "per_tile_flags: " << to_string(opt.per_tile_flags) << ", ";
  ss << "n_split: " << to_string(opt.n_split) << ", ";
  ss << "num_blocks: " << to_string(opt.num_blocks) << ", ";
  ss << "ring_mode: ";
  if (opt.ring_mode) {
    switch (*opt.ring_mode) {
      case RingMode::All2All: ss << "All2All"; break;
      case RingMode::Ring1D: ss << "Ring1D"; break;
      case RingMode::Ring2D: ss << "Ring2D"; break;
      default: ss << "Unknown"; break;
    }
  } else {
    ss << "nullopt";
  }
  ss << "}";
  return ss.str();
}

class GemmRS::GemmRSImpl {
 private:
  std::shared_ptr<Group> group_;
  const int32_t nnodes;
  const int32_t max_m;
  const int32_t n_dim;
  const c10::ScalarType input_dtype;
  const c10::ScalarType output_dtype;
  const bool transpose_weight;
  const bool fuse_reduction;
  const bool ring_reduction;

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
  std::vector<torch::Tensor> reduce_buffers_pin;

  GroupBarrier group_barrier;
  torch::Tensor output_buffer;
  torch::Tensor reduce_buffer;
  torch::Tensor barrier_buffer;
  torch::Tensor gemm_buffer;
  std::vector<void *> output_scatter_ptrs;
  std::vector<void *> barrier_ptrs;
  std::vector<void *> reduce_buffer_ptrs;
  bool no_nvlink;
  int sub_world_size;
  c10::cuda::CUDAStream rs_stream_;
  cudaEvent_t event_;
  bool is_l20;
  const bool is_fp8_gemm;
  const bool is_s8_gemm;

#ifdef FLUX_REDUCE_SCATTER_WITH_NCCL
  ncclComm_t nccl_comm;
#endif
  void
  init_output_buffer() {
    // update max_m and allocate buffer
    if (get_arch() == _Sm90{} || no_nvlink || (get_arch() == _Sm80{} && nnodes > 1)) {
      int reduce_m_dim = (get_arch() == _Sm90{} && fuse_reduction)
                             ? (max_m + world_size - 1) / world_size * nnodes * nnodes
                             : max_m;
      this->reduce_buffers =
          flux_create_tensor_list({reduce_m_dim, n_dim}, output_dtype, this->group_.get());
      static bool use_shm = get_bool_from_env("FLUX_RS_USE_SHM", false);
      if (use_shm) {
        this->reduce_buffers_pin =
            flux_create_shm_tensor_list({reduce_m_dim, n_dim}, output_dtype, this->group_.get());
      }
      this->reduce_buffer = this->reduce_buffers[this->local_rank];
      for (int i = 0; i < world_size; i++) {
        if (i / this->local_world_size == rank / this->local_world_size) {
          if (use_shm && this->world_size != this->sub_world_size &&
              (i + 1) % this->sub_world_size == 0) {
            reduce_buffer_ptrs[i] =
                this->reduce_buffers_pin[i % this->local_world_size].data_ptr();
          } else {
            reduce_buffer_ptrs[i] = this->reduce_buffers[i % this->local_world_size].data_ptr();
          }
          // only check for ranks on the same node
          FLUX_CHECK(reduce_buffer_ptrs[i] != nullptr) << "nullptr buffer of rank " << i;
        } else {
          reduce_buffer_ptrs[i] = nullptr;
        }
      }
    }
    if (get_arch() == _Sm80{} && nnodes > 1 && from_torch_dtype(this->input_dtype) == _BF16{}) {
      // SM80 does not support the fuse reduction for the bfloat16 data type
      // we have to use the float32 global_red instruction when SM80 && nnodes>1 && input_type=bf16
      // Therefore, in this case, here double the size of the output_buffer.
      this->output_buffers =
          flux_create_tensor_list({max_m * 2, n_dim}, output_dtype, this->group_.get());
    } else {
      this->output_buffers =
          flux_create_tensor_list({max_m, n_dim}, output_dtype, this->group_.get());
    }
    this->output_buffer = this->output_buffers[this->local_rank];
    for (int i = 0; i < world_size; ++i) {
      if (i / this->local_world_size == rank / this->local_world_size) {
        output_scatter_ptrs[i] = this->output_buffers[i % this->local_world_size].data_ptr();
        // only check for ranks on the same node
        TORCH_CHECK(
            output_scatter_ptrs[i] != nullptr, "nullptr buffer of rank " + std::to_string(i));
      } else {
        output_scatter_ptrs[i] = nullptr;
      }
    }
  }

  void
  lazy_init_barrier_buffer(int64_t buffer_size) {
    if ((buffer_size == 0) ||
        (barrier_buffer.defined() && buffer_size <= barrier_buffer.numel())) {
      return;
    }
    if (!this->barrier_buffers.empty()) {
      auto stream = c10::cuda::getCurrentCUDAStream();
      group_barrier.barrier_all(stream);
      c10::cuda::stream_synchronize(stream);
    }
    this->barrier_buffers =
        flux_create_tensor_list({buffer_size}, c10::ScalarType::Byte, this->group_.get());
    this->barrier_buffer = this->barrier_buffers[this->local_rank];
    for (int i = 0; i < world_size; ++i) {
      if (i / this->local_world_size == rank / this->local_world_size) {
        barrier_ptrs[i] = this->barrier_buffers[i % this->local_world_size].data_ptr();
        // only check for ranks on the same node
        TORCH_CHECK(barrier_ptrs[i] != nullptr, "nullptr buffer of rank " + std::to_string(i));
      } else {
        barrier_ptrs[i] = nullptr;
      }
    }
  }

  bool
  has_nvlink() {
    _ensure_topo_initialized();
    this->sub_world_size = topo_utils::topo_numa_local_world_size();
    static int has_nvlink_env = get_int_from_env("FLUX_FORCE_NVLINK", -1);
    if (has_nvlink_env == -1) {
      if (topo_utils::has_nvswitch()) {
        return true;
      } else {
        if (topo_utils::has_heterogeneous_nvlink()) {
          this->sub_world_size = topo_utils::topo_nvlink_local_world_size();
        }
        return false;
      }
    }
    return has_nvlink_env;
  }

  bool
  is_l20_or_not() {
    ensure_nvml_init();
    int devid = at::cuda::current_device();
    std::string devname(get_gpu_device_name(devid));
    return devname == "NVIDIA L20";
  }

  bool
  use_p2p_read_or_not() {
    return false;
  }

  // checks whether need to pad input m_dim to be multiple of TPxtile_size
  // if the kernel itself has constraint on the m_dim
  bool
  need_pad_m_to_TPxTile() const {
    return get_arch() == _Sm90{};
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
    CUDA_CHECK(cudaStreamCreateWithPriority(
        &rs_stream, cudaStreamNonBlocking, get_highest_cuda_stream_priority()));
    return at::cuda::getStreamFromExternal(rs_stream, at::cuda::current_device());
  }

  ReduceScatterOption
  materialize(const ReduceScatterOptionWithOptional &opt) {
    // some options are contradictory: per_tile_flags/use_gemm_k is only valid for CUDA CORE
    // implementations, which is not supported by cudaMemcpyAsync.
    bool opt_use_cuda_core = (opt.per_tile_flags.has_value() && opt.per_tile_flags.value()) ||
                             (opt.use_gemmk.has_value() && opt.use_gemmk.value());
    bool opt_use_cudaMemcpyAsync =
        opt.use_cudaMemcpyAsync.has_value() && opt.use_cudaMemcpyAsync.value();
    if (opt_use_cudaMemcpyAsync && opt_use_cuda_core) {
      CALL_ONCE({
        std::cerr << "WARNING: per_tile_flags/use_gemmk is set to true. force set "
                     "use_cudaMemcpyAsync to false\n";
      });
      opt_use_cudaMemcpyAsync = false;
    }

    bool default_per_tile_flags = this->no_nvlink;
    bool default_use_gemmk = this->no_nvlink;
    bool default_use_cudaMemcpyAsync = false;
    bool default_use_barrier_queue = false;
    bool default_use_p2p_read = true;
    int default_num_blocks = 6;
    int default_n_split = 1;
    bool default_use_1d_ring = this->world_size == this->sub_world_size;
    if (this->is_l20 && !opt_use_cuda_core) {
      default_per_tile_flags = false;
      default_use_gemmk = false;
      default_use_cudaMemcpyAsync = true;
      // for add_continous kernel that reduce local GPU buffers. use more grids
      default_num_blocks = 16;
      default_use_1d_ring = true;
      default_use_p2p_read = false;
    }
    if (opt.n_split.has_value() and opt.n_split.value() != 1) {
      CALL_ONCE({ std::cerr << "warning: n_split option is not implemented yet." << std::endl; });
    }

    return ReduceScatterOption{
        .use_barrier_queue = opt.use_barrier_queue.value_or(default_use_barrier_queue),
        .use_1d_ring = opt.use_1d_ring.value_or(default_use_1d_ring),
        .use_p2p_read = opt.use_p2p_read.value_or(default_use_p2p_read),
        .use_cudaMemcpyAsync = opt.use_cudaMemcpyAsync.value_or(default_use_cudaMemcpyAsync),
        .use_gemmk = opt.use_gemmk.value_or(default_use_gemmk),
        .per_tile_flags = opt.per_tile_flags.value_or(default_per_tile_flags),
        .n_split = opt.n_split.value_or(default_n_split),
        .num_blocks = opt.num_blocks.value_or(default_num_blocks),
        .ring_mode = opt.ring_mode.value_or(get_default_rs_ring_mode()),
    };
  }

 public:
  GemmRSImpl(
      std::shared_ptr<Group> group_,
      int32_t nnodes,
      int32_t max_m,
      int32_t n_dim,
      c10::ScalarType input_dtype,
      c10::ScalarType output_dtype,
      bool transpose_weight,
      bool fuse_reduction,
      bool ring_reduction)
      : group_(group_),
        nnodes(nnodes),
        max_m(max_m),
        n_dim(n_dim),
        input_dtype(input_dtype),
        output_dtype(output_dtype),
        transpose_weight(transpose_weight),
        fuse_reduction(fuse_reduction),
        ring_reduction(ring_reduction),
        rank(group_->get_rank()),
        world_size(group_->get_size()),
        local_world_size(world_size / nnodes),
        local_rank(rank % local_world_size),
        node_idx(rank / local_world_size),
        group_barrier(this->group_, false),
        output_scatter_ptrs(world_size, nullptr),
        barrier_ptrs(world_size, nullptr),
        reduce_buffer_ptrs(world_size, nullptr),
        no_nvlink(!has_nvlink()),
        rs_stream_(CreateReduceScatterStream()),  // private stream. never dup with gemm stream
        is_l20(is_l20_or_not()),
        is_fp8_gemm(c10::isFloat8Type(input_dtype)),
        is_s8_gemm(is_s8_dtype(from_torch_dtype(input_dtype))) {
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
    if (no_nvlink && rank == 0) {
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
#ifdef FLUX_REDUCE_SCATTER_WITH_NCCL
    if (nnodes > 1 && no_nvlink) {
      nccl_comm = topo_utils::create_nccl_comm_with_processgroup(tp_group);
    } else {
      nccl_comm = nullptr;
    }
#endif
  }

  ~GemmRSImpl() {
    CUDA_CHECK(cudaEventDestroy(event_));
    CUDA_CHECK(cudaStreamDestroy(rs_stream_));
#ifdef FLUX_REDUCE_SCATTER_WITH_NCCL
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
    DataTypeEnum accum_type = is_s8_gemm ? _S32{}() : _FP32{}();
    auto dt_conf = make_gemm_dtype_config(
        input_dtype, input_dtype, has_bias ? output_dtype : _Void{}(), output_dtype, accum_type);

    fast_accum = fast_accum & dt_conf.is_input_fp8();
    bool is_gemm_v2 = ((int)arch < (int)_Sm90{}());
    auto meta = make_gemm_meta(
        dt_conf,
        arch,
        _ReduceScatter{},
        gemm_layout,
        is_gemm_v2 ? _GemmV2{}() : _GemmV3{}(),
        is_gemm_v2 ? UnifiedImplMeta(make_gemm_v2_meta(fast_accum))
                   : UnifiedImplMeta(make_gemm_v3_meta(fast_accum, /*block_scale=*/false)),
        make_reduce_scatter_meta(
            this->fuse_reduction,
            nnodes > 1        ? _AcrossNode{}()
            : this->no_nvlink ? _IntraNodePcie{}()
                              : _IntraNode{}()));
    return meta;
  }

  RuntimeConfig
  get_rt_conf(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale) {
    CHECK_INPUT(input, this->input_dtype);
    CHECK_INPUT(weight, this->input_dtype);
    FLUX_CHECK_EQ(input.dim(), 2);
    FLUX_CHECK_EQ(weight.dim(), 2);
    int32_t m = input.size(0);
    int32_t k = input.size(1);
    int32_t n = transpose_weight ? weight.size(1) : weight.size(0);

    if (bias.has_value()) {
      CHECK_INPUT(bias.value(), this->output_dtype);
      if (is_fp8_gemm || is_s8_gemm) {
        if (bias->dim() == 1) {
          FLUX_CHECK_EQ(n, bias->size(0)) << "bias n-dim mismatch with weight n-dim";
        } else {
          FLUX_CHECK_EQ(bias->dim(), 2);
          FLUX_CHECK_EQ(1, bias->size(0)) << "FP8/S8 only support bias with shape (1, N)";
          FLUX_CHECK_EQ(n, bias->size(1)) << "bias n-dim mismatch with weight n-dim";
        }
      } else {
        FLUX_CHECK_EQ(bias->dim(), 2);
        FLUX_CHECK_EQ(m, bias->size(0)) << "bias m-dim mismatch with input m-dim";
        FLUX_CHECK_EQ(n, bias->size(1));
      }
    }

    if (is_s8_gemm && input_scale.has_value()) {
      FLUX_CHECK(input_scale->dim() == 1 || input_scale->dim() == 2);
      FLUX_CHECK_EQ(m, input_scale->size(0));
      if (input_scale->dim() == 2) {
        FLUX_CHECK_EQ(1, input_scale->size(1));
      }
    }
    if (is_s8_gemm && weight_scale.has_value()) {
      FLUX_CHECK(weight_scale->dim() == 1 || weight_scale->dim() == 2);
      if (weight_scale->dim() == 2) {
        FLUX_CHECK_EQ(1, weight_scale->size(0));
        FLUX_CHECK_EQ(n, weight_scale->size(1));
      } else {
        FLUX_CHECK_EQ(n, weight_scale->size(0));
      }
    }

    // row major for streamk, todo: make weight layout an option
    int32_t wk = transpose_weight ? weight.size(0) : weight.size(1);
    FLUX_CHECK_LE(m, this->max_m) << "m-dim greater than maximum possible value";
    FLUX_CHECK_EQ(n, this->n_dim) << "n-dim != expected n_dim";
    FLUX_CHECK_EQ(wk, k) << "weight k-dim mismatch";
    return make_runtime_config(m, n, k, make_reduce_scatter_runtime_config(world_size, nnodes));
  }

  // maybe we should put this into gemm_v2_reduce_scatter, but NO!!! I don't want to include torch
  // as flux_cuda dependency
  void
  initialize_args_workspace(
      const UnifiedGemmHParams &hparams,
      GemmReduceScatterArguments const &op_args,
      void *args_workspace,
      c10::cuda::CUDAStream stream) const {
    const ReduceScatterArguments &rs_args = op_args.reduce_scatter_args;
    // noop
    if (no_nvlink) {
      int tiled_size_m = cute::get<0>(hparams.tile_shape());
      int tiled_size_n = cute::get<1>(hparams.tile_shape());
      int tiled_size_k = cute::get<2>(hparams.tile_shape());
      int m_per_rank = op_args.m / op_args.world_size;
      int m_per_rank_fixed = (m_per_rank + tiled_size_m - 1) / tiled_size_m * tiled_size_m;
      bytedance::flux::ThreadBlockSwizzleSegmentUtils calculator(
          {rs_args.use_gemmk ? m_per_rank_fixed * op_args.world_size : op_args.m,
           op_args.n,
           op_args.k},
          {tiled_size_m, tiled_size_n, tiled_size_k},
          op_args.rank,
          op_args.world_size,
          rs_args.use_1d_ring ? op_args.world_size : rs_args.sub_world_size,
          op_args.nnodes,
          !rs_args.use_1d_ring,
          rs_args.per_tile_flags);
      auto pinned_tensor = at::empty(
          {static_cast<long>(sizeof(SegmentInfo) * this->world_size)},
          at::TensorOptions().dtype(at::kByte).pinned_memory(true));

      calculator.calc_segments_info((SegmentInfo *)pinned_tensor.data_ptr());

      CUDA_CHECK(cudaMemcpyAsync(
          args_workspace,
          pinned_tensor.data_ptr(),
          op_args.world_size * sizeof(bytedance::flux::SegmentInfo),
          cudaMemcpyHostToDevice,
          stream));
      FLUX_CHECK(at::cuda::CachingHostAllocator_recordEvent(
          pinned_tensor.data_ptr(), pinned_tensor.storage().data_ptr().get_context(), stream));
    }
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
      c10::optional<UnifiedGemmHParams> const &hparams,
      const ReduceScatterOption &opt) {
    auto meta = get_gemm_meta(/*has_bias=*/bias.has_value(), fast_accum);
    auto rt_conf = get_rt_conf(input, weight, bias, input_scale, weight_scale);
    // get cutlass op
    UnifiedGemmHParams hparams_ =
        hparams.has_value() ? hparams.value() : OpRegistry::instance().get_hparams(meta, rt_conf);
    OpRegistry::OpPtr cutlass_op = OpRegistry::instance().get_op(meta, hparams_);
    ReduceScatterArguments reduce_scatter_args{
        .reduce_scatter_num_blocks = opt.num_blocks,
        .rs_stream = rs_stream_,
        .event = event_,
        .use_barrier_queue = opt.use_barrier_queue,
        .use_gemmk = opt.use_gemmk,
        .per_tile_flags = opt.per_tile_flags,
        .use_cudaMemcpyAsync = opt.use_cudaMemcpyAsync,
        .n_split = opt.n_split,
        .sub_world_size = this->sub_world_size,
#ifdef FLUX_REDUCE_SCATTER_WITH_NCCL
        .opaque = nccl_comm,
#else
        .opaque = nullptr,
#endif
        .use_1d_ring = opt.use_1d_ring,
        .use_p2p_read = opt.use_p2p_read,
    };
    auto stream = c10::cuda::getCurrentCUDAStream();

    if (!is_fp8_gemm && !is_s8_gemm) {
      FLUX_CHECK(!input_scale.has_value());
      FLUX_CHECK(!weight_scale.has_value());
      FLUX_CHECK(!output_scale.has_value());
    }

    // only apply bias in rank 0
    float beta = bias.has_value() ? 1.0f : 0.0f;
    if (is_s8_gemm && static_cast<int>(this->rank) != 0) {
      beta = 0.f;
    }

    if (is_fp8_gemm) {
      FLUX_CHECK(!bias.has_value()) << "FP8 does not support bias";
    }

    GemmReduceScatterArguments args{
        .m = rt_conf.m(),
        .n = rt_conf.n(),
        .k = rt_conf.k(),
        .rank = static_cast<int>(this->rank),
        .world_size = static_cast<int>(this->world_size),
        .nnodes = static_cast<int>(this->nnodes),
        .alpha = 1.0f,
        .beta = beta,
        .input = input.data_ptr(),
        .weight = weight.data_ptr(),
        .bias = bias.has_value() ? bias->data_ptr() : nullptr,
        .output_scatter_ptrs = this->output_scatter_ptrs.data(),
        .reduce_buffer_ptrs = this->reduce_buffer_ptrs.data(),
        .barrier_ptrs = this->barrier_ptrs.data(),
        .avail_sms = no_nvlink ? 1 : -1,
        .Aux = nullptr,
        .Vector = bias.has_value() ? bias->data_ptr() : nullptr,
        .abs_max_Aux = nullptr,
        .abs_max_D = nullptr,
        .scaleA = (float *)(input_scale.has_value() ? input_scale->data_ptr() : nullptr),
        .scaleB = (float *)(weight_scale.has_value() ? weight_scale->data_ptr() : nullptr),
        .scaleC = nullptr,
        .scaleD = (float *)(output_scale.has_value() ? output_scale->data_ptr() : nullptr),
        .scaleAux = nullptr,
        .reduce_scatter_args = reduce_scatter_args,
    };

    if (no_nvlink) {
      int priority = 0;
      CUDA_CHECK(cudaStreamGetPriority(stream, &priority));
      if (priority == get_highest_cuda_stream_priority()) {
        FLUX_LOG_FIRST_N(INFO, 1) << "FLUX GemmRs on PCI-e expects GEMM runs on a low priority "
                                     "cudaStream. Otherwise the performance suffers\n";
      }
    }

    // initialize workspace
    int64_t workspace_size = cutlass_op->get_workspace_size(args);
    this->lazy_init_gemm_buffer(input, workspace_size);
    void *workspace = this->gemm_buffer.defined() ? this->gemm_buffer.data_ptr() : nullptr;
    initialize_args_workspace(hparams_, args, workspace, stream);

    // initialize barrier workspace
    int64_t barrier_workspace_size = cutlass_op->get_barrier_workspace_size(args);
    // * 8 is for corner case reduce_scatter tiles. never mind this won't be a large memory
    barrier_workspace_size = barrier_workspace_size * (sizeof(PerTileFlags) / sizeof(int)) * 8;
    this->lazy_init_barrier_buffer(barrier_workspace_size);

    if ((fuse_reduction && !(meta.arch() == _Sm90{})) || this->no_nvlink) {
      // need to zero buffers;
      zero_buffers();
    }
    cutlass_op->run(args, workspace, stream);

  }  // namespace ths_op

  torch::Tensor
  local_reduction(torch::Tensor buffer, int32_t dim, int32_t rank, bool ring_reduction) {
    // This function is used to accumulate paritial outputs from all rank(`fuse_reduction` is
    // disable).
    // set `ring_reduction` to false:
    //      accumulate partial output alone dim, the accumulation order may different from
    //      reduce_scatter primitive, leads to floating-point errors.
    // set `ring_reduction` to true:
    //      accumulate partial output in ring order to keep same order with reduce_scatter.
    //      this is useful to reduce floating-point errors.
    if (!ring_reduction) {
      return buffer.sum(dim);
    } else {
      int32_t dim_size = buffer.size(dim);
      torch::TensorOptions options =
          torch::TensorOptions().dtype(buffer.dtype()).device(buffer.device());
      std::vector<int64_t> output_shape;
      int32_t input_dims = buffer.dim();
      for (int32_t i = 0; i < input_dims; ++i) {
        if (i != dim)
          output_shape.push_back(buffer.size(i));
      }
      auto output = empty_with_uninitialized_data(output_shape, options);
      ring_reduce(buffer, output, dim, rank);
      return output;
    }
  }

  torch::Tensor
  forward_reduce_scatter_impl(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<UnifiedGemmHParams> hparams) {
    auto meta = get_gemm_meta(/*has_bias=*/bias.has_value());  // fast_accum doesn't matter
    auto rt_conf = get_rt_conf(input, weight, bias, input_scale, weight_scale);

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
          auto output_3d =
              full_output.view({nnodes, local_world_size, m / world_size, n})[node_idx];
          auto output = local_reduction(output_3d, 0, this->rank, this->ring_reduction);
          return output;
        }
      }
    } else if (meta.arch() == _Sm90{}) {
      if (fuse_reduction) {
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
        auto full_output = this->reduce_buffer.slice(0, 0, m);
        auto output_3d = full_output.view({world_size, m / world_size, n});
        auto output = local_reduction(output_3d, 0, this->rank, this->ring_reduction);
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
      c10::optional<UnifiedGemmHParams> const &hparams,
      const ReduceScatterOption &opt) {
    if (need_pad_m_to_TPxTile()) {
      // if need pad, decide hparams with the unpadded m before passing downwards to prevent
      // hparams of the padded m is different from the tile_size that is used for padding
      auto meta = get_gemm_meta(/*has_bias=*/bias.has_value(), fast_accum);
      auto rt_conf = get_rt_conf(input, weight, bias, input_scale, weight_scale);
      auto true_hparams = hparams.has_value() ? hparams.value()
                                              : OpRegistry::instance().get_hparams(meta, rt_conf);
      int tile_m = cute::get<0>(true_hparams.tile_shape());
      int origin_m_per_rank = input.size(0) / world_size;
      std::tie(input, input_scale) = pad_m_to_TPxTile(input, input_scale, world_size, tile_m);
      forward_gemm_impl(
          input,
          weight,
          bias,
          input_scale,
          weight_scale,
          output_scale,
          fast_accum,
          true_hparams,
          opt);
      forward_barrier(input, weight, bias);
      auto output = forward_reduce_scatter_impl(
          input, weight, bias, input_scale, weight_scale, true_hparams);
      return output.slice(0, 0, origin_m_per_rank);
    } else {
      forward_gemm_impl(
          input, weight, bias, input_scale, weight_scale, output_scale, fast_accum, hparams, opt);
      forward_barrier(input, weight, bias);
      return forward_reduce_scatter_impl(input, weight, bias, input_scale, weight_scale, hparams);
    }
  }

  void
  forward_barrier(torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    if (get_arch() == _Sm90{} and nnodes == 1) {
      // only local reduce, skip nvshmem barrier
    } else {
      group_barrier.barrier_all(stream);
    }
  }

  torch::Tensor
  forward_reduce_scatter(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale) {
    return forward_reduce_scatter_impl(
        std::move(input),
        std::move(weight),
        std::move(bias),
        input_scale,
        weight_scale,
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
      bool fast_accum,
      const ReduceScatterOptionWithOptional &opt) {
    return forward_impl(
        std::move(input),
        std::move(weight),
        std::move(bias),
        std::move(input_scale),
        std::move(weight_scale),
        std::move(output_scale),
        fast_accum,
        c10::nullopt,
        materialize(opt));
  }

  void
  zero_buffers() {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    if (!no_nvlink) {
      if (this->output_buffer.defined()) {
        this->output_buffer.zero_();
      }
      if (this->reduce_buffer.defined()) {
        this->reduce_buffer.zero_();
      }
    }
    if (this->barrier_buffer.defined()) {
      this->barrier_buffer.zero_();
    }
    group_barrier.barrier_all(stream);
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
      c10::intrusive_ptr<ProfilingContext> opt_ctx,
      const ReduceScatterOptionWithOptional &reduce_scatter_option) {
    auto meta = unify_type(this->get_gemm_meta(/*has_bias=*/bias.has_value(), fast_accum));
    auto rt_conf = this->get_rt_conf(input, weight, bias, input_scale, weight_scale);
    ProfilingContext tmp_ctx("__tmp__");
    ProfilingContext *ctx = opt_ctx == nullptr ? &tmp_ctx : opt_ctx.get();

    auto filter_hparams = [&](UnifiedGemmHParams const &hparams) { return true; };

    auto elapsed_tensor = torch::empty({}, weight.options().dtype(c10::ScalarType::Float));
    auto reduced_elapsed_tensor = elapsed_tensor.clone();
    auto opt = materialize(reduce_scatter_option);

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
                input,
                weight,
                bias,
                input_scale,
                weight_scale,
                output_scale,
                fast_accum,
                hparams,
                opt);
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
          float reduce_elapsed = all_reduce_max_float(this->group_.get(), avg_elapsed);
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
        std::move(best_hparams),
        opt);
  }

  void
  _ensure_topo_initialized() {
    if (!topo_utils::is_topo_initialized()) {
      topo_utils::initialize_topo(this->group_.get());
    }
  }
};  // namespace flux

GemmRS::GemmRS(
    std::shared_ptr<Group> group,
    int32_t nnodes,
    int32_t max_m,
    int32_t n_dim,
    c10::ScalarType input_dtype,
    c10::ScalarType output_dtype,
    bool transpose_weight,
    bool fuse_reduction,
    bool ring_reduction)
    : impl_(new GemmRSImpl(
          group,
          nnodes,
          max_m,
          n_dim,
          input_dtype,
          output_dtype,
          transpose_weight,
          fuse_reduction,
          ring_reduction)) {}
GemmRS::~GemmRS() { delete impl_; }
void
GemmRS::zero_buffers() {
  FLUX_CHECK(impl_ != nullptr) << "GemmRS is not initialized";
  impl_->zero_buffers();
}
torch::Tensor
GemmRS::forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale,
    c10::optional<torch::Tensor> output_scale,
    bool fast_accum,
    const ReduceScatterOptionWithOptional &reduce_scatter_option) {
  FLUX_CHECK(impl_ != nullptr) << "GemmRS is not initialized";
  return impl_->forward(
      std::move(input),
      std::move(weight),
      std::move(bias),
      std::move(input_scale),
      std::move(weight_scale),
      std::move(output_scale),
      fast_accum,
      reduce_scatter_option);
}
torch::Tensor
GemmRS::profiling(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale,
    c10::optional<torch::Tensor> output_scale,
    bool fast_accum,
    c10::intrusive_ptr<ProfilingContext> opt_ctx,
    const ReduceScatterOptionWithOptional &reduce_scatter_option) {
  FLUX_CHECK(impl_ != nullptr) << "GemmRS is not initialized";
  return impl_->profiling(
      std::move(input),
      std::move(weight),
      std::move(bias),
      std::move(input_scale),
      std::move(weight_scale),
      std::move(output_scale),
      fast_accum,
      opt_ctx,
      reduce_scatter_option);
}
void
GemmRS::forward_barrier(
    torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias) {
  FLUX_CHECK(impl_ != nullptr) << "GemmRS is not initialized";
  impl_->forward_barrier(std::move(input), std::move(weight), std::move(bias));
}
torch::Tensor
GemmRS::forward_reduce_scatter(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale) {
  FLUX_CHECK(impl_ != nullptr) << "GemmRS is not initialized";
  return impl_->forward_reduce_scatter(
      std::move(input),
      std::move(weight),
      std::move(bias),
      std::move(input_scale),
      std::move(weight_scale));
}

}  // namespace bytedance::flux::ths_op
