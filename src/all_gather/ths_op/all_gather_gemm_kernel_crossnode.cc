//===- all_gather_gemm_kernel_crossnode.cc ------------------------ C++ ---===//
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
#include "flux/gemm_meta.h"
#include "flux/op_registry.h"
#include "flux/runtime_config.h"
#include "flux/ths_op/ths_op.h"
#include "flux/ths_op/ths_op.h"
#include "flux/ths_op/util.h"
#include "flux/args/all_gather.h"
#if defined(FLUX_DEBUG)
#include "flux/utils.h"
#endif
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/List.h>
#include <ATen/core/TensorBody.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/from_blob.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/intrusive_ptr.h>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <nvshmemx.h>
#include "nccl.h"
#include "flux/ths_op/topo_utils.h"
#include "all_gather_types.h"

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
bool
get_bool_from_env_debug(const char *env, bool default_value) {
#if defined(FLUX_DEBUG)
  return get_bool_from_env(env, default_value);
#else
  return default_value;
#endif
}

bool
wait_value() {
  static bool value = get_bool_from_env_debug("FLUX_AG_WAIT_VALUE", true);
  return value;
}
bool
write_value() {
  static bool value = get_bool_from_env_debug("FLUX_AG_WRITE_VALUE", true);
  return value;
}
bool
run_gemm() {
  static bool value = get_bool_from_env_debug("FLUX_AG_RUN_GEMM", true);
  return value;
}
bool
run_nccl() {
  static bool value = get_bool_from_env_debug("FLUX_AG_RUN_NCCL", true);
  return value;
}
bool
run_memcpy() {
  static bool value = get_bool_from_env_debug("FLUX_AG_RUN_MEMCPY", true);
  return value;
}
}  // namespace

/// All Gather GEMM Kernel OP
class AGKernelCrossNode : public torch::CustomClassHolder {
 private:
  c10::intrusive_ptr<c10d::ProcessGroup> tp_group;
  c10::intrusive_ptr<c10d::ProcessGroup> intra_node_group;
  int32_t nnodes;
  int32_t rank;
  int32_t world_size;
  int32_t local_world_size;
  int32_t local_rank;
  int32_t node_id;

  torch::Tensor input_buffer;
  std::vector<torch::Tensor> input_buffers;
  const at::ScalarType _st;
  torch::Tensor output_buffer;

  int32_t full_m;
  int32_t n_dim;
  int32_t k_dim;
  const bool transpose_weight;
  const bool local_copy;

  torch::Tensor barrier_buffer;
  std::vector<torch::Tensor> barrier_buffers;
  std::vector<void *> input_ptrs;
  std::vector<void *> barrier_ptrs;

  AGRingMode ring_mode;

  AGKernelArguments gemm_args;
  std::unique_ptr<GemmOperatorBase> cutlass_op;
  torch::Tensor gemm_buffer;
  void *workspace;

  int num_cp_streams = 1;
  cudaEvent_t cp_event;
  std::vector<at::cuda::CUDAStream> cp_streams;
  size_t chunk_size;
  size_t split_chunk_size;
  size_t signal_size;
  size_t split_signal_size;

  cudaEvent_t ready_event;
  cudaEvent_t gemm_event;

  ncclComm_t nccl_comm;
  cudaEvent_t nccl_event;
  int num_nccl_streams = 1;
  std::vector<at::cuda::CUDAStream> nccl_streams;

 public:
  auto
  get_gemm_meta(bool has_bias) {
    ArchEnum arch = get_arch();
    auto gemm_layout = transpose_weight ? _RRR{}() : _RCR{}();
    auto dtype = from_torch_dtype(this->_st);
    auto dt_conf = make_gemm_dtype_config(dtype, dtype, has_bias ? dtype : _Void{}(), dtype);
    auto meta = make_gemm_meta(
        dt_conf,
        arch,
        _AGKernel{},
        gemm_layout,
        ((int)arch < (int)_Sm90{}()) ? _GemmV2{}() : _GemmV3{}());

    return meta;
  }

  AGKernelCrossNode(
      c10::intrusive_ptr<c10d::ProcessGroup> &tp_group_,
      c10::intrusive_ptr<c10d::ProcessGroup> &intra_node_group_,
      int32_t nnodes,
      torch::Tensor output_buffer,
      int32_t full_m,
      int32_t n_dim,
      int32_t k_dim,
      c10::ScalarType input_dtype,
      bool transpose_weight = true,
      bool local_copy = false,
      AGRingMode ring_mode_ = AGRingMode::Auto)
      : tp_group(tp_group_),
        intra_node_group(intra_node_group_),
        nnodes(nnodes),
        rank(tp_group->getRank()),
        world_size(tp_group->getSize()),
        local_world_size(world_size / nnodes),
        local_rank(rank % local_world_size),
        node_id(rank / local_world_size),
        _st(input_dtype),
        output_buffer(output_buffer),
        full_m(full_m),
        n_dim(n_dim),
        k_dim(k_dim),
        transpose_weight(transpose_weight),
        local_copy(local_copy),
        input_ptrs(local_world_size, nullptr),
        barrier_ptrs(local_world_size, nullptr) {
    FLUX_CHECK(rank >= 0 && rank < world_size)
        << "invalid rank: " << rank << " and world_size: " << world_size;
    FLUX_CHECK(world_size % nnodes == 0)
        << "invalid nnodes: world_size[" << world_size << "] %% nnodes[" << nnodes << "] != 0";

    CHECK_INPUT(output_buffer, _st);
    FLUX_CHECK(output_buffer.dim() == 2) << " Output buffer rank is not 2!";

    nccl_comm = topo_utils::create_nccl_comm_with_processgroup(tp_group);
    FLUX_CHECK(nccl_comm != nullptr);

    // Due to CUDA driver BUGS: use nvshmem_malloc only on PCI-e machines.  CUDA driver BUGG Won't
    // be fixed until CUDA 12.4
    // TODO(houqi.1993) don't know if NVLink invokes the BUG or NVSwitch. should be tested on PCI-e
    // machines with NVLink P2P connected
    _ensure_topo_initialized();
    this->ring_mode = get_ring_mode(ring_mode_);
    bool use_nvshmem_malloc = topo_utils::has_nvlink();
    // input buffer
    if (use_nvshmem_malloc) {
      this->input_buffers = nvshmem_create_tensor_list({this->full_m, this->k_dim}, input_dtype);
    } else {
      this->input_buffers =
          cudaipc_create_tensor_list(intra_node_group, {this->full_m, this->k_dim}, input_dtype);
    }
    FLUX_CHECK(this->input_buffers.size() == this->local_world_size)
        << " input_buffers.size() != local_world_size " << this->input_buffers.size() << " vs "
        << this->local_world_size;
    this->input_buffer = this->input_buffers[local_rank];
    for (int i = 0; i < this->local_world_size; ++i) {
      this->input_ptrs[i] = this->input_buffers[i].data_ptr();
      FLUX_CHECK(input_ptrs[i] != nullptr) << " input ptr null for local_rank: " << i;
    }
    // output buffer
    // this->output_buffer = torch::zeros({m, n}, weight.options());
    FLUX_CHECK(this->local_world_size <= MAX_NUM_SIGNAL)
        << "local_world_size <= " << MAX_NUM_SIGNAL
        << " expected, got : " << this->local_world_size;

    if (use_nvshmem_malloc) {
      this->barrier_buffers =
          nvshmem_create_tensor_list({MAX_NUM_SIGNAL * this->nnodes}, c10::ScalarType::Int);
    } else {
      this->barrier_buffers = cudaipc_create_tensor_list(
          intra_node_group, {MAX_NUM_SIGNAL * this->nnodes}, c10::ScalarType::Int);
    }
    FLUX_CHECK(this->barrier_buffers.size() == this->local_world_size)
        << "barrier_buffers.size() != local_world_size)" << this->barrier_buffers.size() << " vs "
        << this->local_world_size;
    this->barrier_buffer = this->barrier_buffers[local_rank];
    for (int i = 0; i < this->local_world_size; ++i) {
      this->barrier_ptrs[i] = this->barrier_buffers[i].data_ptr();
      FLUX_CHECK(barrier_ptrs[i] != nullptr) << " barrier ptr null for local_rank: " << i;
    }

    // Copy event/streams for communication
    CUDA_CHECK(cudaEventCreateWithFlags(&this->cp_event, cudaEventDisableTiming));
    this->num_cp_streams = 1;
    for (int i = 0; i < this->num_cp_streams; ++i) {
      this->cp_streams.push_back(at::cuda::getStreamFromPool());
    }

    // Data chunk size is based on world size not local one
    this->chunk_size =
        (this->full_m / this->world_size) * this->k_dim * this->input_buffer.element_size();
    this->split_chunk_size = chunk_size / SPLIT;
    // Signal size is also based on world size
    this->signal_size = SPLIT * sizeof(int32_t);
    this->split_signal_size = signal_size / SPLIT;

    CUDA_CHECK(cudaEventCreateWithFlags(&this->ready_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&this->gemm_event, cudaEventDisableTiming));

    // nccl event/stream
    CUDA_CHECK(cudaEventCreateWithFlags(&this->nccl_event, cudaEventDisableTiming));
    for (int i = 0; i < this->num_nccl_streams; ++i) {
      this->nccl_streams.push_back(at::cuda::getStreamFromPool());
    }
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

  ~AGKernelCrossNode() {
    cudaEventDestroy(cp_event);
    cudaEventDestroy(nccl_event);
    cudaEventDestroy(gemm_event);
    ncclCommFinalize(nccl_comm);
  }

  torch::Tensor
  forward_impl(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<UnifiedGemmHParams> const &hparams) {
    CHECK_INPUT(input, this->_st);
    CHECK_INPUT(weight, this->_st);
    TORCH_CHECK(input.dim() == 2, "input shape is not 2");
    TORCH_CHECK(weight.dim() == 2, "GEMM weight rank is not 2!");

    auto rt_config = get_rt_config(input, weight);
    auto meta = get_gemm_meta(/*has_bias=*/false);
    if (hparams.has_value()) {
      this->cutlass_op = OpRegistry::instance().get_op(meta, hparams.value());
    } else {
      this->cutlass_op = OpRegistry::instance().get_op(meta, rt_config);
    }

    // AGKernel Arguments
    this->gemm_args = {
        .m = rt_config.m(),
        .n = rt_config.n(),
        .k = rt_config.k(),
        .rank = static_cast<int>(rank),
        .world_size = static_cast<int>(world_size),
        .nnodes = static_cast<int>(nnodes),
        .alpha = 1.0,
        .beta = 0.0,
        .input = input.data_ptr(),
        .input_buffer = this->input_buffer.data_ptr(),
        .weight = weight.data_ptr(),
        .bias = nullptr,
        .output = this->output_buffer.data_ptr(),
        .barrier_buffer = this->barrier_buffer.data_ptr()};

    // AG Gemm Workspace
    int64_t workspace_size = cutlass_op->get_workspace_size(this->gemm_args);
    this->lazy_init_gemm_buffer(input, workspace_size);
    this->workspace = this->gemm_buffer.defined() ? this->gemm_buffer.data_ptr() : nullptr;

    /// AG GEMM
    cudaStream_t current_stream = c10::cuda::getCurrentCUDAStream();
    if (!local_copy) {
      copy_local(input);
    }
    CUDA_CHECK(cudaEventRecord(this->ready_event, current_stream));

    if (ring_mode == AGRingMode::All2All) {
      copy_all2all(input);
    } else if (ring_mode == AGRingMode::Ring2D) {
      FLUX_CHECK(this->local_world_size == 8);
      copy_2d_ring(input);
    } else if (ring_mode == AGRingMode::Ring1D) {
      copy_ring_1d_push();
    } else if (ring_mode == AGRingMode::RingCustom) {
      CUDA_CHECK(cudaStreamWaitEvent(cp_streams[0], this->ready_event));
      ring_all_gather(cp_streams[0]);
    } else {
      FLUX_CHECK(false) << "Unknown ring mode " << (int)ring_mode;
    }
    CUDA_CHECK(cudaEventRecord(this->ready_event, cp_streams[0]));

    /// GEMM
    if (run_gemm()) {
      // to make sure, launch gemm after ncclSendRecv. but it takes too long.
      if (ring_mode == AGRingMode::Ring1D ||
          ring_mode == AGRingMode::Ring2D) {  // PCI-e wait for nccl comm done
        CUDA_CHECK(cudaStreamWaitEvent(current_stream, this->nccl_event));
      }
      cutlass_op->run(this->gemm_args, this->workspace, current_stream);
    } else {
      CUDA_CHECK(cudaStreamWaitEvent(current_stream, this->ready_event));
    }

    return this->output_buffer;
  }

  torch::Tensor
  forward(torch::Tensor input, torch::Tensor weight) {
    return forward_impl(std::move(input), std::move(weight), c10::nullopt);
  }

  void
  reset_signals() {
    auto current_stream = c10::cuda::getCurrentCUDAStream();
    cudaStreamWaitEvent(current_stream, this->cp_event);
    nvshmemx_barrier_all_on_stream(current_stream);
    if (ring_mode == AGRingMode::All2All) {  // TODO(wenlei.bao) check if this is needed
      c10::cuda::stream_synchronize(current_stream);
    }

    this->barrier_buffer.zero_();
  }

  void
  copy_local(torch::Tensor input) {
    cudaStream_t current_stream = c10::cuda::getCurrentCUDAStream();
    void *input_ptr = input.data_ptr();
    void *input_buffer_ptr = this->input_buffer.data_ptr();
    void *barrier_ptr = this->barrier_buffer.data_ptr();

    CUDA_CHECK(cudaMemcpyAsync(
        ptr_offset(input_buffer_ptr, this->rank * chunk_size),
        input_ptr,
        this->chunk_size,
        cudaMemcpyDefault,
        current_stream));

    for (int j = 0; j < SPLIT; ++j) {
      auto split_signal = j * this->split_signal_size;
      CU_CHECK(CUStreamWriteValue(
          current_stream,
          (CUdeviceptr)(ptr_offset(barrier_ptr, this->rank * signal_size + split_signal)),
          1,
          CU_STREAM_WRITE_VALUE_DEFAULT));
    }
    nvshmemx_barrier_all_on_stream(current_stream);
    if (ring_mode == AGRingMode::All2All) {  // TODO(wenlei.bao) check if this is needed
      c10::cuda::stream_synchronize(current_stream);
    }
  }

 private:
  void
  copy_all2all(torch::Tensor input) {
    /// All2All algorithm
    /// Each node perform intra node communication: pull data from other intra node ranks
    {
      at::cuda::CUDAStreamGuard guard(this->cp_streams[0]);
      auto stream = cp_streams[0];
      CUDA_CHECK(cudaStreamWaitEvent(stream, this->ready_event));

      for (int i = local_rank + 1; i < (local_world_size + local_rank); ++i) {
        auto id = i % this->local_world_size;
        auto data_offset = (id + node_id * local_world_size) * chunk_size;
        auto signal_offset = (id + node_id * local_world_size) * signal_size;
        for (int j = 0; j < SPLIT; ++j) {
          auto split_offset = j * this->split_chunk_size;
          auto split_signal = j * this->split_signal_size;

          CUDA_CHECK(cudaMemcpyAsync(
              ptr_offset(this->input_ptrs[this->local_rank], data_offset + split_offset),
              ptr_offset(this->input_ptrs[id], data_offset + split_offset),
              this->split_chunk_size,
              cudaMemcpyDefault,
              stream));
          CU_CHECK(CUStreamWriteValue(
              stream,
              (CUdeviceptr)(ptr_offset(
                  barrier_ptrs[this->local_rank], signal_offset + split_signal)),
              1,
              CU_STREAM_WRITE_VALUE_DEFAULT));
        }
      }
    }

    /// Multi-node communications
    if (this->nnodes > 1) {
      // (nnodes - 1) phases
      for (int phase_id = 1; phase_id < this->nnodes; ++phase_id) {
        int pid = phase_id - 1;  // start from 0

        // NCCL across node communication
        {
          at::cuda::CUDAStreamGuard nccl_guard(this->nccl_streams[0]);
          if (pid == 0) {
            CUDA_CHECK(cudaStreamWaitEvent(this->nccl_streams[0], this->ready_event));
          }

          int send_rank = (this->rank + phase_id * this->local_world_size) % this->world_size;
          int recv_rank = (this->rank - phase_id * this->local_world_size + this->world_size) %
                          this->world_size;

          void *send_ptr = input.data_ptr();
          void *recv_ptr = ptr_offset(this->input_buffer.data_ptr(), recv_rank * this->chunk_size);

          NCCL_CHECK(ncclGroupStart());
          NCCL_CHECK(ncclSend(
              send_ptr, this->chunk_size, ncclInt8, send_rank, nccl_comm, this->nccl_streams[0]));
          NCCL_CHECK(ncclRecv(
              recv_ptr, this->chunk_size, ncclInt8, recv_rank, nccl_comm, this->nccl_streams[0]));
          NCCL_CHECK(ncclGroupEnd());

          cudaEventRecord(this->nccl_event, this->nccl_streams[0]);
        }
        // Intra-node communicaiton
        {
          at::cuda::CUDAStreamGuard guard(this->cp_streams[0]);
          auto stream = cp_streams[0];  // reuse one cp_stream
          CUDA_CHECK(cudaStreamWaitEvent(stream, this->nccl_event));

          int recv_id = (this->rank - phase_id * this->local_world_size + this->world_size) %
                        this->world_size;
          auto recv_signal_offset = recv_id * signal_size;

          CU_CHECK(CUStreamWriteValue(
              stream,
              (CUdeviceptr)(ptr_offset(barrier_ptrs[this->local_rank], recv_signal_offset)),
              1,
              CU_STREAM_WRITE_VALUE_DEFAULT));

          for (int i = local_rank + 1; i < (local_world_size + local_rank); ++i) {
            auto id = i % this->local_world_size;

            auto data_offset = (id + (recv_id / local_world_size * local_world_size)) * chunk_size;
            auto signal_offset =
                (id + (recv_id / local_world_size * local_world_size)) * signal_size;

            for (int j = 0; j < SPLIT; ++j) {
              auto split_offset = j * this->split_chunk_size;
              auto split_signal = j * this->split_signal_size;

              /// wait
              CU_CHECK(CUStreamWaitValue(
                  stream,
                  (CUdeviceptr)(ptr_offset(barrier_ptrs[id], signal_offset + split_signal)),
                  1,
                  CU_STREAM_WAIT_VALUE_EQ));

              CUDA_CHECK(cudaMemcpyAsync(
                  ptr_offset(this->input_ptrs[this->local_rank], data_offset + split_offset),
                  ptr_offset(this->input_ptrs[id], data_offset + split_offset),
                  this->split_chunk_size,
                  cudaMemcpyDefault,
                  stream));

              CU_CHECK(CUStreamWriteValue(
                  stream,
                  (CUdeviceptr)(ptr_offset(
                      barrier_ptrs[this->local_rank], signal_offset + split_signal)),
                  1,
                  CU_STREAM_WRITE_VALUE_DEFAULT));
            }
          }
        }
      }
    }
    CUDA_CHECK(cudaEventRecord(this->cp_event, this->cp_streams[0]));
  }

  void
  copy_to_next_node(int phase_id) {
    // NCCL across node communication
    auto nccl_stream = nccl_streams[0];
    int send_rank = (this->rank + phase_id * this->local_world_size) % this->world_size;
    int recv_rank =
        (this->rank - phase_id * this->local_world_size + this->world_size) % this->world_size;

    void *send_ptr = ptr_offset(this->input_buffer.data_ptr(), rank * this->chunk_size);
    void *recv_ptr = ptr_offset(this->input_buffer.data_ptr(), recv_rank * this->chunk_size);

    NCCL_CHECK(ncclGroupStart());
    NCCL_CHECK(ncclSend(send_ptr, this->chunk_size, ncclInt8, send_rank, nccl_comm, nccl_stream));
    NCCL_CHECK(ncclRecv(recv_ptr, this->chunk_size, ncclInt8, recv_rank, nccl_comm, nccl_stream));
    NCCL_CHECK(ncclGroupEnd());

    CUDA_CHECK(cudaEventRecord(this->nccl_event, nccl_stream));

    auto recv_signal_offset = recv_rank * signal_size;
    CU_CHECK(CUStreamWriteValue(
        nccl_stream,
        (CUdeviceptr)(ptr_offset(barrier_ptrs[this->local_rank], recv_signal_offset)),
        1,
        CU_STREAM_WRITE_VALUE_DEFAULT));
  }

  void
  copy_2d_ring(torch::Tensor input) {
    /// All2All algorithm
    /// Each node perform intra node communication: pull data from other intra node ranks
    auto cp_stream = cp_streams[0];
    auto nccl_stream = nccl_streams[0];
    CUDA_CHECK(cudaStreamWaitEvent(cp_stream, this->ready_event));
    CUDA_CHECK(cudaStreamWaitEvent(nccl_stream, this->ready_event));
    copy_local_ring(0, cp_stream);

    /// Multi-node communications
    // (nnodes - 1) phases
    for (int phase_id = 1; phase_id < this->nnodes; ++phase_id) {
      // Intra-node communicaiton
      CUDA_CHECK(cudaStreamWaitEvent(cp_stream, this->nccl_event));
      copy_local_ring(phase_id, cp_stream);
    }
    CUDA_CHECK(cudaEventRecord(this->cp_event, cp_stream));
  }

 public:
  torch::Tensor
  gemm_only_impl(
      torch::Tensor input,
      torch::Tensor full_input,
      torch::Tensor weight,
      c10::optional<UnifiedGemmHParams> const &hparams) {
    // input is gathered input
    CHECK_INPUT(input, this->_st);
    CHECK_INPUT(weight, this->_st);
    TORCH_CHECK(input.dim() == 2, "input shape is not 2");
    TORCH_CHECK(weight.dim() == 2, "GEMM weight rank is not 2!");

    auto rt_config = get_rt_config(input, weight);
    auto meta = get_gemm_meta(/*has_bias=*/false);
    if (hparams.has_value()) {
      this->cutlass_op = OpRegistry::instance().get_op(meta, hparams.value());
    } else {
      this->cutlass_op = OpRegistry::instance().get_op(meta, rt_config);
    }

    // AGKernel Arguments
    this->gemm_args = {
        .m = rt_config.m(),
        .n = rt_config.n(),
        .k = rt_config.k(),
        .rank = static_cast<int>(rank),
        .world_size = static_cast<int>(world_size),
        .nnodes = static_cast<int>(nnodes),
        .alpha = 1.0,
        .beta = 0.0,
        .input = input.data_ptr(),
        .input_buffer = full_input.data_ptr(),
        .weight = weight.data_ptr(),
        .bias = nullptr,
        .output = this->output_buffer.data_ptr(),
        .barrier_buffer = this->barrier_buffer.data_ptr()};

    // AG Gemm Workspace
    int64_t workspace_size = cutlass_op->get_workspace_size(this->gemm_args);
    this->lazy_init_gemm_buffer(input, workspace_size);
    this->workspace = this->gemm_buffer.defined() ? this->gemm_buffer.data_ptr() : nullptr;

    /// AG GEMM
    cudaStream_t current_stream = c10::cuda::getCurrentCUDAStream();

    /// GEMM
    cutlass_op->run(this->gemm_args, this->workspace, current_stream);

    return this->output_buffer;
  }

  torch::Tensor
  gemm_only(torch::Tensor input, torch::Tensor full_input, torch::Tensor weight) {
    return gemm_only_impl(
        std::move(input), std::move(full_input), std::move(weight), c10::nullopt);
  }

 private:
  void
  wait_cross_numa_done(int phase_id, cudaStream_t stream) {
    int segment_offset = local_world_size * phase_id;
    if (wait_value()) {
      CUstreamBatchMemOpParams params[2 * SPLIT];
      int counter = 0;
      for (int rank_x_numa : std::array<int, 2>{0, intra_numa_world_size}) {
        int local_recv_rank = (rank_x_numa - 1 + local_world_size) % local_world_size;
        int send_segment = (segment_offset + intra_numa_world_size - 1 + rank_x_numa +
                            this->node_id * this->local_world_size) %
                           this->world_size;
        for (int j = 0; j < SPLIT; j++) {
          void *ptr = ptr_offset(
              this->barrier_ptrs[local_recv_rank],
              send_segment * signal_size + j * split_signal_size);
          auto &param = params[counter];
          param.operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
          param.waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
          param.waitValue.address = (CUdeviceptr)ptr;
          param.waitValue.flags = CU_STREAM_WAIT_VALUE_EQ;
          param.waitValue.value = 1;
          counter++;
        }
        cuda_stub().cuStreamBatchMemOp_v2(stream, 2 * SPLIT, params, 0);
      }
    }
  }

  void
  copy_local_ring(int phase_id, at::cuda::CUDAStream stream) {
    int segment_offset = local_world_size * phase_id;
    // segment_offset % local_world_size == 0
    // [0, local_world_size) stages:  0 <- 1 <- 2 <- 3 <- 4 <- 5 <- 6 <- 7 <- 0
    // [local_world_size, world_size) stages: 0 <- 1 <- 2 <-3 <- 0 && 4 <- 5 <- 6 <- 7 <- 4
    int to_rank =
        (local_rank - 1 + local_world_size) % local_world_size;  // always recv data from rank prev
    bool has_cross_numa_send = rank % intra_numa_world_size == 0;
    bool has_cross_numa_recv = (rank + 1) % intra_numa_world_size == 0;
    bool has_cross_numa = has_cross_numa_recv && has_cross_numa_send;
    int numa_node = local_rank / intra_numa_world_size;
    for (int i = 0; i < local_world_size - 1; i++) {  // with inner and intra numa node
      int send_segment = (local_rank + i) % local_world_size;
      if (i >= intra_numa_world_size && local_rank % intra_numa_world_size == 0) {
        send_segment = (send_segment + intra_numa_world_size) % local_world_size;
        to_rank = (local_rank - 1 + intra_numa_world_size) % intra_numa_world_size +
                  numa_node * intra_numa_world_size;
      }
      bool run_nccl_comm =
          (phase_id < this->nnodes - 1) &&
          ((has_cross_numa && i == intra_numa_world_size) || (!has_cross_numa && i == 0));
      // run_nccl_comm = (phase_id < this->nnodes - 1) && (i == 0);
      run_nccl_comm = (phase_id < this->nnodes - 1) && (i == intra_numa_world_size);
      if (run_nccl_comm) {
        auto nccl_stream = nccl_streams[0];
        CUDA_CHECK(cudaEventRecord(this->cp_event, stream));
        CUDA_CHECK(cudaStreamWaitEvent(nccl_stream, this->cp_event));
        wait_cross_numa_done(phase_id, nccl_stream);
        // wait for cross numa stages copy done
        copy_to_next_node(phase_id + 1);
      }
      send_segment = (segment_offset + send_segment + this->node_id * this->local_world_size) %
                     this->world_size;  // segment_offset % local_world_size == 0

      for (int j = 0; j < SPLIT; ++j) {
        auto split_offset = j * this->split_chunk_size;
        auto split_signal = j * this->split_signal_size;
        auto data_offset = send_segment * chunk_size + split_offset;
        auto signal_offset = send_segment * signal_size + split_signal;
        void *send_ptr = ptr_offset(this->input_ptrs[local_rank], data_offset);
        void *recv_ptr = ptr_offset(this->input_ptrs[to_rank], data_offset);
        void *signal_ptr = ptr_offset(barrier_ptrs[to_rank], signal_offset);
        void *local_signal_ptr = ptr_offset(barrier_ptrs[local_rank], signal_offset);
        // make sure i == 0
        if (i != 0 && !(i >= intra_numa_world_size && local_rank % intra_numa_world_size == 0)) {
          // previous local_rank recv done
          if (wait_value()) {
            CU_CHECK(CUStreamWaitValue(
                stream, (CUdeviceptr)(local_signal_ptr), 1, CU_STREAM_WAIT_VALUE_EQ));
          }
        }
        CUDA_CHECK(cudaMemcpyAsync(
            recv_ptr, send_ptr, this->split_chunk_size, cudaMemcpyDeviceToDevice, stream));
        if (write_value()) {
          CU_CHECK(CUStreamWriteValue(
              stream, (CUdeviceptr)(signal_ptr), 1, CU_STREAM_WRITE_VALUE_DEFAULT));
        }
      }
    }
  }

  void
  copy_ring_1d_push() {  // use 1d ring
    auto cp_stream = this->cp_streams[0];
    auto nccl_stream = this->nccl_streams[0];
    CUDA_CHECK(cudaStreamWaitEvent(cp_stream, this->ready_event));
    CUDA_CHECK(cudaStreamWaitEvent(nccl_stream, this->ready_event));
    // always the  0 <- 1 <- 2 <- 3 <- xxx order
    int from_rank = (this->rank + 1) % this->world_size;  // always copy from rank next
    int to_rank =
        (this->rank - 1 + this->world_size) % this->world_size;  // always copy to rank prev
    int local_to_rank = to_rank % this->local_world_size;
    for (int i = 0; i < this->world_size - 1; i++) {
      int send_segment = (this->rank + i) % this->world_size;
      int recv_segment = (this->rank + i + 1) % this->world_size;
      for (int j = 0; j < SPLIT; ++j) {
        auto data_offset = send_segment * this->chunk_size + j * this->split_chunk_size;
        auto recv_data_offset = recv_segment * this->chunk_size + j * this->split_chunk_size;
        auto signal_offset = send_segment * this->signal_size + j * this->split_signal_size;
        void *send_ptr = ptr_offset(this->input_ptrs[this->local_rank], data_offset);
        void *signal_ptr = ptr_offset(barrier_ptrs[local_to_rank], signal_offset);
        void *local_signal_ptr = ptr_offset(barrier_ptrs[this->local_rank], signal_offset);
        if (i != 0 && wait_value()) {  // previous rank recv done. i == 0 it is always ready
          CU_CHECK(CUStreamWaitValue(
              cp_stream, (CUdeviceptr)(local_signal_ptr), 1, CU_STREAM_WAIT_VALUE_EQ));
        }
        bool inter_node_recv = (this->rank + 1) % local_world_size == 0;
        bool inter_node_send = this->rank % local_world_size == 0;  // 2->1->0->world_size-1->xxx
        if (inter_node_recv) {
          void *recv_ptr = ptr_offset(this->input_ptrs[this->local_rank], recv_data_offset);
          // NOTE: recv stream also send. use no cp_stream for recv
          CUDA_CHECK(cudaEventRecord(nccl_event, cp_stream));
          CUDA_CHECK(cudaStreamWaitEvent(nccl_stream, nccl_event));
          if (run_nccl()) {
            NCCL_CHECK(ncclGroupStart());
            NCCL_CHECK(ncclRecv(
                recv_ptr, this->split_chunk_size, ncclInt8, from_rank, nccl_comm, nccl_stream));
            NCCL_CHECK(ncclGroupEnd());
          }
          if (write_value()) {
            auto recv_signal_offset =
                recv_segment * this->signal_size + j * this->split_signal_size;
            void *recv_signal_ptr = ptr_offset(barrier_ptrs[this->local_rank], recv_signal_offset);
            CU_CHECK(CUStreamWriteValue(
                nccl_stream, (CUdeviceptr)(recv_signal_ptr), 1, CU_STREAM_WRITE_VALUE_DEFAULT));
          }
          CUDA_CHECK(cudaEventRecord(nccl_event, nccl_stream));
        }
        if (inter_node_send && run_nccl()) {
          // copy data and set peer signal
          NCCL_CHECK(ncclGroupStart());
          NCCL_CHECK(ncclSend(
              send_ptr, this->split_chunk_size, ncclInt8, to_rank, nccl_comm, nccl_stream));
          NCCL_CHECK(ncclGroupEnd());
          CUDA_CHECK(cudaEventRecord(nccl_event, nccl_stream));
        } else if (!inter_node_send) {
          void *recv_ptr = ptr_offset(this->input_ptrs[local_to_rank], data_offset);
          if (run_memcpy()) {
            CUDA_CHECK(cudaMemcpyAsync(
                recv_ptr, send_ptr, this->split_chunk_size, cudaMemcpyDeviceToDevice, cp_stream));
          }
          if (write_value()) {
            CU_CHECK(CUStreamWriteValue(
                cp_stream, (CUdeviceptr)(signal_ptr), 1, CU_STREAM_WRITE_VALUE_DEFAULT));
          }
        }
        if (inter_node_recv) {
          // do we really need this wait? or next iter wait local_signal_ptr is enough
          CUDA_CHECK(cudaStreamWaitEvent(cp_stream, nccl_event));
        }
        CUDA_CHECK(cudaStreamWaitEvent(cp_stream, nccl_event));
      }
    }
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

  RuntimeConfig
  get_rt_config(torch::Tensor input, torch::Tensor weight) {
    int n = transpose_weight ? weight.size(1) : weight.size(0);
    int k = transpose_weight ? weight.size(0) : weight.size(1);

    TORCH_CHECK(
        n == this->n_dim,
        "n-dim != expected n_dim: " + std::to_string(n) + " vs " + std::to_string(this->n_dim));
    TORCH_CHECK(
        k == this->k_dim,
        "weight k-dim mismatch: " + std::to_string(k) + " != " + std::to_string(this->k_dim));

    return make_runtime_config(
        full_m, n_dim, k_dim, make_all_gather_runtime_config(world_size, nnodes, (int)ring_mode));
  }

  void
  _ensure_topo_initialized() {
    if (!topo_utils::is_topo_initialized()) {
      topo_utils::initialize_topo(this->tp_group);
    }
  }
};

namespace py = pybind11;

static int _register_ag_kernel_crossnode_ops [[maybe_unused]] = []() {
  ThsOpsInitRegistry::instance().register_one(
      "all_gather_gemm_kernel_crossnode", [](py::module &m) {
        py::class_<AGKernelCrossNode>(m, "AGKernelCrossNode")
            .def(
                py::init([](c10::intrusive_ptr<c10d::ProcessGroup> tp_group,
                            c10::intrusive_ptr<c10d::ProcessGroup> intra_node_group,
                            int32_t nnodes,
                            torch::Tensor output_buffer,
                            int32_t full_m,
                            int32_t n_dim,
                            int32_t k_dim,
                            py::object py_input_dtype,
                            bool transpose_weight = true,
                            bool local_copy = false,
                            AGRingMode ring_mode = AGRingMode::Auto) {
                  auto input_dtype = torch::python::detail::py_object_to_dtype(py_input_dtype);
                  return new AGKernelCrossNode(
                      tp_group,
                      intra_node_group,
                      nnodes,
                      output_buffer,
                      full_m,
                      n_dim,
                      k_dim,
                      input_dtype,
                      transpose_weight,
                      local_copy,
                      ring_mode);
                }),
                py::arg("tp_group"),
                py::arg("intra_node_group"),
                py::arg("nnodes"),
                py::arg("output_buffer"),
                py::arg("full_m"),
                py::arg("n_dim"),
                py::arg("k_dim"),
                py::arg("input_dtype"),
                py::arg("transpose_weight") = true,
                py::arg("local_copy") = false,
                py::arg("ring_mode") = AGRingMode::Auto)
            .def("reset_signals", &AGKernelCrossNode::reset_signals)
            .def("copy_local", &AGKernelCrossNode::copy_local)
            .def("gemm_only", &AGKernelCrossNode::gemm_only)
            .def("forward", &AGKernelCrossNode::forward);
      });
  return 0;
}();

}  // namespace ths_op
}  // namespace flux
}  // namespace bytedance
