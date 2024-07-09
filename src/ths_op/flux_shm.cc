//===- flux_shm.cc ---------------------------------------------- C++ ---===//
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
#include <vector>
#include <ATen/ops/from_blob.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAFunctions.h>
#include <torch/cuda.h>
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/helper_kernels.h"
#include "flux/flux.h"
#include "flux/ths_op/flux_shm.h"

#ifdef FLUX_SHM_USE_NVSHMEM
#include <nvshmemx.h>
#endif

namespace bytedance {
namespace flux {

#ifdef FLUX_SHM_USE_NVSHMEM
namespace {
std::array<const char *, 5> kNvshmemInitStatus = {
    "NVSHMEM_STATUS_NOT_INITIALIZED",
    "NVSHMEM_STATUS_IS_BOOTSTRAPPED",
    "NVSHMEM_STATUS_IS_INITIALIZED",
    "NVSHMEM_STATUS_LIMITED_MPG",
    "NVSHMEM_STATUS_FULL_MPG"};
void
check_nvshmem_init() {
  FLUX_CHECK(nvshmemx_init_status() >= NVSHMEM_STATUS_IS_INITIALIZED)
      << "nvshmem not initialized: status " << kNvshmemInitStatus[nvshmemx_init_status()];
}
}  // namespace
torch::Tensor
nvshmem_create_tensor(const std::vector<int64_t> &shape, c10::ScalarType dtype) {
  check_nvshmem_init();
  auto option_gpu =
      at::TensorOptions().dtype(dtype).device(at::kCUDA).device_index(c10::cuda::current_device());
  auto size = torch::elementSize(dtype) *
              std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
  FLUX_CHECK(size != 0);
  void *ptr = nvshmem_malloc(size);
  FLUX_CHECK(ptr != nullptr);
  CUDA_CHECK(cudaMemset(ptr, 0, size)); // memset the allocated buffer
  return at::from_blob(
      ptr, shape, [](void *ptr) { nvshmem_free(ptr); }, option_gpu);
}

std::vector<torch::Tensor>
nvshmem_create_tensor_list(const std::vector<int64_t> &shape, c10::ScalarType dtype) {
  check_nvshmem_init();
  auto option_gpu =
      at::TensorOptions().dtype(dtype).device(at::kCUDA).device_index(c10::cuda::current_device());
  auto size = torch::elementSize(dtype) *
              std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
  FLUX_CHECK(size != 0);
  int local_world_size = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
  int rank = nvshmem_my_pe();
  int local_rank = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  std::vector<torch::Tensor> tensors;
  tensors.reserve(local_world_size);
  void *ptr = nvshmem_malloc(size);
  CUDA_CHECK(cudaMemset(ptr, 0, size)); // memset the allocated buffer
  FLUX_CHECK(ptr != nullptr);
  int rank_offset = rank - local_rank;
  for (int i = 0; i < local_world_size; i++) {
    // runs this call nvshmem failure, don't know why
    //  nvshmem_team_translate_pe(NVSHMEMX_TEAM_NODE, local_rank, NVSHMEM_TEAM_WORLD)
    int rank_global = i + rank_offset;
    if (rank == rank_global) {
      tensors.emplace_back(
          at::from_blob(ptr, shape, [](void *ptr) { nvshmem_free(ptr); }, option_gpu));
    } else {
      void *rptr = nvshmem_ptr(ptr, rank_global);
      FLUX_CHECK(rptr != nullptr) << "rank " << rank;
      tensors.emplace_back(at::from_blob(rptr, shape, option_gpu));
    }
  }

  return tensors;
}

#endif

std::vector<torch::Tensor>
cudaipc_create_tensor_list(
    c10::intrusive_ptr<c10d::ProcessGroup> pg,
    const std::vector<int64_t> &shape,
    c10::ScalarType dtype) {
  auto option_gpu =
      at::TensorOptions().dtype(dtype).device(at::kCUDA).device_index(c10::cuda::current_device());

  FLUX_CHECK(pg->getSize() <= torch::cuda::device_count())
      << "create_ipc_tensors should only be used intra node";
  auto size = torch::elementSize(dtype) *
              std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
  FLUX_CHECK(size != 0);
  void *ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&ptr, size));
  CUDA_CHECK(cudaMemset(ptr, 0, size)); // memset the allocated buffer
  cudaIpcMemHandle_t handle;
  CUDA_CHECK(cudaIpcGetMemHandle(&handle, ptr));

  auto option_local = at::TensorOptions()
                          .dtype(torch::kUInt8)
                          .device(at::kCUDA)
                          .device_index(c10::cuda::current_device());
  auto handle_d = torch::empty({sizeof(cudaIpcMemHandle_t)}, option_local);
  CUDA_CHECK(cudaMemcpy(
      handle_d.data_ptr(), &handle, sizeof(cudaIpcMemHandle_t), cudaMemcpyHostToDevice));
  auto handles_d = torch::empty({sizeof(cudaIpcMemHandle_t) * pg->getSize()}, option_local);
  pg->_allgather_base(handles_d, handle_d)->wait();

  std::vector<cudaIpcMemHandle_t> handles_h(pg->getSize());
  CUDA_CHECK(cudaMemcpy(
      handles_h.data(),
      handles_d.data_ptr(),
      sizeof(cudaIpcMemHandle_t) * pg->getSize(),
      cudaMemcpyDeviceToHost));

  std::vector<void *> ptrs(pg->getSize());
  for (int i = 0; i < pg->getSize(); ++i) {
    if (i != pg->getRank()) {
      CUDA_CHECK(cudaIpcOpenMemHandle(&ptrs[i], handles_h[i], cudaIpcMemLazyEnablePeerAccess));
    } else {
      ptrs[i] = ptr;
    }
  }

  std::vector<torch::Tensor> tensors;
  for (int i = 0; i < pg->getSize(); ++i) {
    torch::Tensor tensor;
    if (i == pg->getRank()) {
      tensor = at::from_blob(ptr, shape, [](void *ptr) { cudaFree(ptr); }, option_gpu);
    } else {
      tensor =
          at::from_blob(ptrs[i], shape, [](void *ptr) { cudaIpcCloseMemHandle(ptr); }, option_gpu);
    }
    tensors.emplace_back(tensor);
  }
  return tensors;
}

void
init_flux_shm(c10::intrusive_ptr<c10d::ProcessGroup> c10_pg) {
#ifdef FLUX_SHM_USE_NVSHMEM
  nvshmemx_init_attr_t init_attr;
  init_attr.mpi_comm = (void *)c10_pg.get();  // bad! pretend I'm the MPIComm
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &init_attr);
  int mype = nvshmem_my_pe();
  CHECK(c10_pg->getRank() == mype)
      << "NVShmem init: rank does not match PE!" << c10_pg->getRank() << " vs " << mype;
#endif
}

torch::Tensor
flux_create_tensor(
    const std::vector<int64_t> &shape,
    c10::ScalarType dtype,
    c10::intrusive_ptr<c10d::ProcessGroup> pg) {
#ifdef FLUX_SHM_USE_NVSHMEM
  return nvshmem_create_tensor(shape, dtype);
#else
  FLUX_CHECK(false && "This line should never be reached");
  return torch::Tensor();
#endif
}
std::vector<torch::Tensor>
flux_create_tensor_list(
    const std::vector<int64_t> &shape,
    c10::ScalarType dtype,
    c10::intrusive_ptr<c10d::ProcessGroup> pg) {
#ifdef FLUX_SHM_USE_NVSHMEM
  return nvshmem_create_tensor_list(shape, dtype);
#else
  FLUX_CHECK(pg != nullptr);
  return cudaipc_create_tensor_list(pg, shape, dtype);
#endif
}

void
flux_barrier_all_on_stream(
    cudaStream_t stream,
    c10::optional<std::vector<torch::Tensor>> sync_buffers,
    c10::optional<int> rank) {
#ifdef FLUX_SHM_USE_NVSHMEM
  nvshmemx_barrier_all_on_stream(stream);
#else
  FLUX_CHECK(sync_buffers.has_value());
  FLUX_CHECK(rank.has_value());
  std::vector<int32_t *> sync_buffer_ptrs;
  auto sync_buffers_val = sync_buffers.value();
  FLUX_CHECK(sync_buffers_val[rank.value()].defined());
  int world_size = sync_buffers_val.size();
  for (int i = 0; i < sync_buffers_val.size(); i++) {
    sync_buffer_ptrs.push_back(reinterpret_cast<int32_t *>(sync_buffers_val[i].data_ptr()));
  }
  cudaipc_barrier_all_on_stream_impl(stream, sync_buffer_ptrs.data(), rank.value(), world_size);
#endif
}

void
pyflux_barrier_all_on_stream(
    intptr_t stream,
    c10::optional<std::vector<torch::Tensor>> sync_buffers,
    c10::optional<int> rank) {
  flux_barrier_all_on_stream((cudaStream_t)stream, sync_buffers, rank);
}

}  // namespace flux
}  // namespace bytedance
