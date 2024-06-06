//===- all_gather_gemm.cc ----------------------------------------- C++ ---===//
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

#include "flux/cuda/cuda_common.h"
#include "flux/cuda/cuda_stub.h"
#include "flux/flux.h"
#include "flux/gemm_meta.h"
#include "flux/op_registry.h"
#include "flux/ths_op/ths_op.h"
#include "flux/ths_op/util.h"
#include "flux/args/all_gather.h"
#include <ATen/core/jit_type.h>
#include <ATen/core/List.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/ops/empty.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/intrusive_ptr.h>
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <nvshmem.h>
#include <nvshmemx.h>

#define CUStreamWaitValue(...) cuda_stub().cuStreamWaitValue64_v2(__VA_ARGS__)
#define CUStreamWriteValue(...) cuda_stub().cuStreamWriteValue64_v2(__VA_ARGS__)

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

class AllGatherGemm : public torch::CustomClassHolder {
 private:
  const int64_t rank;
  const int64_t world_size;
  const int64_t stages;
  const at::ScalarType _st;

  torch::Tensor input_buffer;
  torch::Tensor weight;
  torch::Tensor output_buffer;

  torch::Tensor gemm_buffer[8];  // gemm workspace
  // std::vector<torch::Tensor> gemm_buffers;

  std::vector<torch::Tensor> cuda_shm_handles;
  std::vector<void *> base_ptrs;
  std::vector<void *> cpu_input_ag_ptrs;

  // cuda signal
  torch::Tensor signal;
  std::vector<void *> signal_base_ptrs;
  std::vector<void *> signal_ptrs;
  std::vector<std::pair<unsigned long long int, unsigned long long int>>
      stage_signal_ptrs;  // pair{wait, write}

  const bool transpose_weight;
  const bool gather_output;

  /// utils to shift overhead to initilization
  int full_m;
  size_t input_offset;
  size_t output_offset;
  int next_rank;

  std::vector<int> comm_stream_ids;
  std::vector<int> gemm_stream_ids;
  std::vector<void *> stage_src_ptrs;
  std::vector<void *> stage_tgt_ptrs;
  std::vector<int> stage_comp_off;
  std::vector<at::cuda::CUDAStream> cp_streams;
  std::vector<at::cuda::CUDAStream> gemm_streams;
  std::vector<AllGatherGemmArguments> args;
  std::vector<void *> workspace;
  std::vector<std::unique_ptr<GemmOperatorBase>> gemm_ops;

  std::vector<at::cuda::CUDAStream> reset_stream;
  cudaEvent_t ready_event;
  cudaEvent_t complete_event;
  cudaEvent_t comm_records[8];
  cudaEvent_t comp_records[8];

 private:
  void
  lazy_init_gemm_buffer(size_t id, int64_t buffer_size) {
    if (buffer_size <= 0)
      return;
    buffer_size = (buffer_size + 127) / 128 * 128;

    if (!this->gemm_buffer[id].defined() || buffer_size > this->gemm_buffer[id].numel()) {
      this->gemm_buffer[id] =
          torch::empty({buffer_size}, this->weight.options().dtype(at::ScalarType::Byte));
    }
  }

 public:
  /// All Gather GEMM OP
  AllGatherGemm(
      int64_t rank,
      int64_t world_size,
      torch::Tensor input_buffer,
      torch::Tensor weight,
      torch::Tensor output_buffer,
      std::vector<torch::Tensor> cuda_shm_handles,
      std::vector<int64_t> cuda_shm_offsets,
      torch::Tensor signal,
      std::vector<torch::Tensor> signal_handles,
      std::vector<int64_t> signal_offsets,
      bool transpose_weight = true,
      bool gather_output = false)
      : rank(rank),
        world_size(world_size),
        stages(world_size),
        _st(weight.scalar_type()),
        input_buffer(input_buffer),
        weight(weight),
        output_buffer(output_buffer),
        cuda_shm_handles(cuda_shm_handles),
        base_ptrs(world_size, nullptr),
        cpu_input_ag_ptrs(world_size, nullptr),
        signal(signal),
        signal_base_ptrs(world_size, nullptr),
        signal_ptrs(world_size, nullptr),
        transpose_weight(transpose_weight),
        gather_output(gather_output) {
    TORCH_CHECK(
        rank >= 0 && rank < world_size,
        "invalid rank: " + std::to_string(rank) +
            " and world_size: " + std::to_string(world_size));
    CHECK_INPUT(input_buffer, _st);
    CHECK_INPUT(weight, _st);
    CHECK_INPUT(output_buffer, _st);
    TORCH_CHECK(input_buffer.dim() == 2, "Input buffer rank is not 2!")
    TORCH_CHECK(weight.dim() == 2, "GEMM weight rank is not 2!");
    TORCH_CHECK(output_buffer.dim() == 2, "Output buffer rank is not 2!");

    CUDA_CHECK(cudaSetDevice(this->rank));
    // Enable peer to peer access
    for (int64_t tgt = 0; tgt < this->world_size; ++tgt) {
      if (tgt == this->rank)
        continue;
      int can_access;
      CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, this->rank, tgt));
      if (can_access != 1) {
        CUDA_CHECK(cudaDeviceEnablePeerAccess(tgt, 0));
      }
    }

    // Open shm handles as device pointer
    TORCH_CHECK(
        this->cuda_shm_handles.size() == (size_t)this->world_size,
        "cuda_shm_handles.size():" + std::to_string(cuda_shm_handles.size()) +
            " != world_size:" + std::to_string(this->world_size));

    for (int64_t i = 0; i < this->world_size; ++i) {
      if (i == this->rank) {
        this->base_ptrs[i] = this->input_buffer.data_ptr();
      } else {
        auto handle_ts = this->cuda_shm_handles[i].tensor_data();
        TORCH_CHECK(
            handle_ts.dtype() == at::ScalarType::Byte,
            "the " + std::to_string(i) + "-th cuda shm handle is not a byte tensor");
        cudaIpcMemHandle_t handle = *reinterpret_cast<cudaIpcMemHandle_t *>(handle_ts.data_ptr());
        CUDA_CHECK(
            cudaIpcOpenMemHandle(&this->base_ptrs[i], handle, cudaIpcMemLazyEnablePeerAccess));
      }
    }

    for (int64_t i = 0; i < this->world_size; ++i) {
      if (i == this->rank) {
        this->cpu_input_ag_ptrs[i] = this->base_ptrs[i];
      } else {
        this->cpu_input_ag_ptrs[i] = ptr_offset(this->base_ptrs[i], cuda_shm_offsets[i]);
      }
    }

    // Get inter-process cuda signals
    for (int64_t i = 0; i < this->world_size; ++i) {
      if (i == this->rank) {
        this->signal_base_ptrs[i] = this->signal.data_ptr();
      } else {
        auto handle_ts = signal_handles[i].tensor_data();
        TORCH_CHECK(
            handle_ts.dtype() == at::ScalarType::Byte,
            "the " + std::to_string(i) + "-th cuda signal handle is not a byte tensor");
        cudaIpcMemHandle_t handle = *reinterpret_cast<cudaIpcMemHandle_t *>(handle_ts.data_ptr());
        CUDA_CHECK(cudaIpcOpenMemHandle(
            &this->signal_base_ptrs[i], handle, cudaIpcMemLazyEnablePeerAccess));
      }
    }

    for (int64_t i = 0; i < this->world_size; ++i) {
      if (i == this->rank) {
        this->signal_ptrs[i] = this->signal_base_ptrs[i];
      } else {
        this->signal_ptrs[i] = ptr_offset(this->signal_base_ptrs[i], signal_offsets[i]);
      }
    }

    // Streams 0/1 reserved for communication
    // Rest streams for gemm computations
    int num_stream = 2;
    for (int i = 0; i < num_stream; ++i) {
      this->cp_streams.push_back(at::cuda::getStreamFromPool());
      this->gemm_streams.push_back(at::cuda::getStreamFromPool());
    }

    // gemm ops per device
    ArchEnum arch = get_arch();
    auto gemm_layout = transpose_weight ? _RRR{}() : _RCR{}();
    auto dtype = from_torch_dtype(this->_st);
    auto meta = make_gemm_meta(
        make_gemm_dtype_config(dtype, dtype, _Void{}(), dtype),
        arch,
        _AllGather{},
        gemm_layout,
        ((int)arch < (int)_Sm90{}()) ? _GemmV2{}() : _GemmV3{}());

    // TODO: calculate m,n,k for each subop
    auto rt_conf = make_runtime_config();

    for (int i = 0; i < this->world_size; ++i) {
      this->gemm_ops.push_back(OpRegistry::instance().get_op(meta, rt_conf));
    }

    // gemm ops initilization
    this->full_m = input_buffer.size(0);
    int m = this->full_m / this->world_size;
    int k = input_buffer.size(1);
    int n = transpose_weight ? weight.size(1) : weight.size(0);
    int wk = transpose_weight ? weight.size(0) : weight.size(1);
    TORCH_CHECK(
        wk == k, "weight k-dim mismatch: " + std::to_string(wk) + " != " + std::to_string(k));

    // gemm ops args
    this->input_offset = m * k * sizeof(input_buffer.dtype());
    this->output_offset = m * n * sizeof(input_buffer.dtype());

    for (int i = 0; i < this->stages; ++i) {
      AllGatherGemmArguments arg{
          m,
          n,
          k,
          static_cast<int>(this->rank),
          static_cast<int>(this->world_size),
          1.0,
          0.0,
          ptr_offset(this->input_buffer.data_ptr(), i * this->input_offset),
          this->weight.data_ptr(),
          nullptr,
          ptr_offset(this->output_buffer.data_ptr(), i * this->output_offset),
      };
      args.push_back(arg);
    }
    int64_t workspace_sizes[this->stages];
    for (int i = 0; i < this->stages; ++i) {
      workspace_sizes[i] = gemm_ops[i]->get_workspace_size(args[i]);
    }
    for (int i = 0; i < this->stages; ++i) {
      this->lazy_init_gemm_buffer(i, workspace_sizes[i]);
      workspace.push_back(
          this->gemm_buffer[i].defined() ? this->gemm_buffer[i].data_ptr() : nullptr);
    }

    // index and pointer pre-computation
    this->next_rank = (this->rank + 1) % this->world_size;

    for (int i = 0; i < this->stages; ++i) {
      this->comm_stream_ids.push_back(i % num_stream);
      this->gemm_stream_ids.push_back(i % num_stream);
      auto off = (this->rank - i + this->world_size) % this->world_size;
      auto off_size = off * this->input_offset;
      this->stage_tgt_ptrs.push_back(
          ptr_offset(this->cpu_input_ag_ptrs[this->next_rank], off_size));
      this->stage_src_ptrs.push_back(ptr_offset(this->input_buffer.data_ptr(), off_size));
      this->stage_comp_off.push_back(off);

      auto dptr_offset = sizeof(unsigned long long int);
      auto wait_dptr =
          (unsigned long long int)(ptr_offset(this->signal_ptrs[this->rank], i * dptr_offset));
      auto write_dptr = (unsigned long long int)(ptr_offset(
          this->signal_ptrs[this->next_rank], i * dptr_offset));
      this->stage_signal_ptrs.push_back(std::make_pair(wait_dptr, write_dptr));
    }

    // reset stream
    for (int i = 0; i < 1; ++i) {
      this->reset_stream.push_back(at::cuda::getStreamFromPool());
    }
    cudaEventCreateWithFlags(&this->ready_event, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&this->complete_event, cudaEventDisableTiming);
    for (int i = 0; i < 8; ++i) {
      cudaEventCreateWithFlags(&this->comm_records[i], cudaEventDisableTiming);
      cudaEventCreateWithFlags(&this->comp_records[i], cudaEventDisableTiming);
    }
  }

  ~AllGatherGemm() {
    for (int i = 0; i < this->world_size; ++i) {
      if (i != this->rank) {
        CUDA_CHECK(cudaIpcCloseMemHandle(this->base_ptrs[i]));
        CUDA_CHECK(cudaIpcCloseMemHandle(this->signal_base_ptrs[i]));
      }
    }
  }

  torch::Tensor
  forward(torch::Tensor input) {
    CUDA_CHECK(cudaSetDevice(this->rank));
    CHECK_INPUT(input, this->_st);
    TORCH_CHECK(input.dim() == 2, "input shape is not 2");
    int m = input.size(0);
    int k = input.size(1);
    int n = transpose_weight ? weight.size(1) : weight.size(0);  // StreamK default for Ampere

    void const *input_ptr = input.data_ptr();
    void const *weight_ptr = this->weight.data_ptr();
    void const *bias_ptr = nullptr;  // currently no bias
    void *output_ptr = this->output_buffer.data_ptr();

    cudaStream_t current_stream = c10::cuda::getCurrentCUDAStream();
    CUDA_CHECK(cudaEventRecord(this->ready_event, current_stream));

    /// Update current rank gemm argument
    AllGatherGemmArguments arg0{
        m,
        n,
        k,
        static_cast<int>(this->rank),
        static_cast<int>(this->world_size),
        1.0,
        0.0,
        input_ptr,
        weight_ptr,
        bias_ptr,
        ptr_offset(output_ptr, this->rank * output_offset)};

    /// Ring algorithm
    /// stage 0
    {
      int s = 0;
      {  /// communicate
        at::cuda::CUDAStreamGuard s0_comm(cp_streams[comm_stream_ids[s]]);
        auto stream = cp_streams[comm_stream_ids[s]].stream();
        // wait
        cudaStreamWaitEvent(stream, this->ready_event);
        CU_CHECK(CUStreamWaitValue(
            stream, this->stage_signal_ptrs[s].second, 0, CU_STREAM_WAIT_VALUE_EQ));
        // send
        cudaMemcpyAsync(
            this->stage_tgt_ptrs[s], input_ptr, input_offset, cudaMemcpyDefault, stream);
        CU_CHECK(CUStreamWriteValue(
            stream, this->stage_signal_ptrs[s].second, 1, CU_STREAM_WRITE_VALUE_DEFAULT));
      }
      {  /// compute
        at::cuda::CUDAStreamGuard s0_gemm(gemm_streams[gemm_stream_ids[s]]);
        auto stream = gemm_streams[gemm_stream_ids[s]].stream();
        auto comp_offset = stage_comp_off[s];
        // wait
        cudaStreamWaitEvent(stream, this->ready_event);
        // gemm
        gemm_ops[comp_offset]->run(arg0, workspace[comp_offset], stream);
        // record
        cudaEventRecord(this->comp_records[s], stream);
        cudaStreamWaitEvent(current_stream, this->comp_records[s]);
      }
    }

    for (int s = 1; s < stages - 1; ++s) {
      {  /// communicate
        at::cuda::CUDAStreamGuard s_comm(cp_streams[comm_stream_ids[s]]);
        auto stream = cp_streams[comm_stream_ids[s]].stream();
        // wait
        cudaStreamWaitEvent(stream, this->ready_event);
        CU_CHECK(CUStreamWaitValue(
            stream, this->stage_signal_ptrs[s].second, 0, CU_STREAM_WAIT_VALUE_EQ));
        CU_CHECK(CUStreamWaitValue(
            stream, this->stage_signal_ptrs[s - 1].first, 1, CU_STREAM_WAIT_VALUE_EQ));
        // send
        cudaMemcpyAsync(
            this->stage_tgt_ptrs[s],
            this->stage_src_ptrs[s],
            input_offset,
            cudaMemcpyDefault,
            stream);
        // record
        cudaEventRecord(this->comm_records[s], stream);
        CU_CHECK(CUStreamWriteValue(
            stream, this->stage_signal_ptrs[s].second, 1, CU_STREAM_WRITE_VALUE_DEFAULT));
      }
      {  /// compute
        at::cuda::CUDAStreamGuard s_gemm(gemm_streams[gemm_stream_ids[s]]);
        auto stream = gemm_streams[gemm_stream_ids[s]].stream();
        auto comp_offset = stage_comp_off[s];
        // wait
        cudaStreamWaitEvent(stream, this->ready_event);
        CU_CHECK(CUStreamWaitValue(
            stream, this->stage_signal_ptrs[s - 1].first, 1, CU_STREAM_WAIT_VALUE_EQ));
        // gemm
        gemm_ops[comp_offset]->run(args[comp_offset], workspace[comp_offset], stream);
        // record
        cudaEventRecord(this->comp_records[s], stream);
        cudaStreamWaitEvent(current_stream, this->comp_records[s]);
      }
      {  // reset signal
        at::cuda::CUDAStreamGuard reset_guard(this->reset_stream[0]);
        auto rstream = this->reset_stream[0].stream();
        cudaStreamWaitEvent(rstream, this->comm_records[s]);
        cudaStreamWaitEvent(rstream, this->comp_records[s]);
        CU_CHECK(CUStreamWriteValue(
            rstream, this->stage_signal_ptrs[s - 1].first, 0, CU_STREAM_WRITE_VALUE_DEFAULT));
      }
    }

    /// last stage
    {
      int s = stages - 1;
      if (gather_output) {
        // copy current
        at::cuda::CUDAStreamGuard s_comm(cp_streams[comm_stream_ids[s]]);
        auto stream = cp_streams[comm_stream_ids[s]].stream();
        cudaStreamWaitEvent(stream, this->ready_event);
        cudaMemcpyAsync(
            ptr_offset(this->input_buffer.data_ptr(), this->rank * this->input_offset),
            input_ptr,
            input_offset,
            cudaMemcpyDefault,
            stream);
        cudaEventRecord(this->comm_records[s], stream);
        cudaStreamWaitEvent(current_stream, this->comm_records[s]);
      }
      {  /// compute
        at::cuda::CUDAStreamGuard s7_gemm_comp(gemm_streams[gemm_stream_ids[s]]);
        auto stream = gemm_streams[gemm_stream_ids[s]].stream();
        auto comp_offset = stage_comp_off[s];
        // wait
        cudaStreamWaitEvent(stream, this->ready_event);
        CUStreamWaitValue(
            stream, this->stage_signal_ptrs[s - 1].first, 1, CU_STREAM_WAIT_VALUE_EQ);
        // gemm
        gemm_ops[comp_offset]->run(args[comp_offset], workspace[comp_offset], stream);
        // record
        cudaEventRecord(this->comp_records[s], stream);
        cudaStreamWaitEvent(current_stream, this->comp_records[s]);
      }
      {  // reset signal
        at::cuda::CUDAStreamGuard reset_guard(this->reset_stream[0]);
        auto rstream = this->reset_stream[0].stream();
        cudaStreamWaitEvent(rstream, this->comp_records[s]);
        CU_CHECK(CUStreamWriteValue(
            rstream, this->stage_signal_ptrs[s - 1].first, 0, CU_STREAM_WRITE_VALUE_DEFAULT));
        cudaEventRecord(this->complete_event, rstream);
        cudaStreamWaitEvent(current_stream, this->complete_event);
      }
    }

    return this->output_buffer;
  }
};

static int _register_ag_ops [[maybe_unused]] = []() {
  ThsOpsInitRegistry::instance().register_one("all_gather_gemm", [](py::module &m) {
    py::class_<AllGatherGemm>(m, "AllGatherGemm")
        .def(py::init<
             int64_t,
             int64_t,
             torch::Tensor,
             torch::Tensor,
             torch::Tensor,
             std::vector<torch::Tensor>,
             std::vector<int64_t>,
             torch::Tensor,
             std::vector<torch::Tensor>,
             std::vector<int64_t>,
             bool,
             bool>())
        .def("forward", &AllGatherGemm::forward);
  });
  return 0;
}();
}  // namespace ths_op
}  // namespace flux
}  // namespace bytedance
