//===- module.cpp ------------------------------------------------- C++ ---===//
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
#include <array>
#include <c10/cuda/CUDAStream.h>
#include <cstddef>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <torch/all.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <vector>

inline void
init_with_c10d_pg(c10::intrusive_ptr<c10d::ProcessGroup> c10_pg) {
  nvshmemx_init_attr_t init_attr;
  init_attr.mpi_comm = (void *)c10_pg.get();  // bad! pretend I'm the MPIComm
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &init_attr);
  int mype = nvshmem_my_pe();
  CHECK(c10_pg->getRank() == mype)
      << "NVShmem init: rank does not match PE!" << c10_pg->getRank() << " vs " << mype;
}

namespace {
std::array<const char *, 5> kNvshmemInitStatus = {
    "NVSHMEM_STATUS_NOT_INITIALIZED",
    "NVSHMEM_STATUS_IS_BOOTSTRAPPED",
    "NVSHMEM_STATUS_IS_INITIALIZED",
    "NVSHMEM_STATUS_LIMITED_MPG",
    "NVSHMEM_STATUS_FULL_MPG"};
void
check_nvshmem_init() {
  CHECK(nvshmemx_init_status() >= NVSHMEM_STATUS_IS_INITIALIZED)
      << "nvshmem not initialized: status " << kNvshmemInitStatus[nvshmemx_init_status()];
}
}  // namespace

inline torch::Tensor
create_tensor(const std::vector<int64_t> &shape, c10::ScalarType dtype) {
  check_nvshmem_init();
  auto option_gpu =
      at::TensorOptions().dtype(dtype).device(at::kCUDA).device_index(c10::cuda::current_device());
  auto size = torch::elementSize(dtype) *
              std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
  return at::from_blob(
      nvshmem_malloc(size), shape, [](void *ptr) { nvshmem_free(ptr); }, option_gpu);
}

inline void
quiet() {
  check_nvshmem_init();

  nvshmem_quiet();
}

inline void
barrier_all() {
  check_nvshmem_init();
  nvshmem_barrier_all();
}

inline void
barrier_on_stream(void *stream) {
  check_nvshmem_init();
  nvshmemx_barrier_all_on_stream((cudaStream_t)stream);
}

void
putmem_on_stream(void *dest, const void *source, size_t bytes, int pe, void *stream) {
  check_nvshmem_init();
  nvshmemx_putmem_on_stream(dest, source, bytes, pe, (cudaStream_t)stream);
}

#define NVSHMEMI_TYPENAME_P_IMPL_PYBIND(TYPENAME, TYPE)    \
  void TYPENAME##_p(ptrdiff_t ptr, TYPE value, int peer) { \
    check_nvshmem_init();                                  \
    nvshmem_##TYPENAME##_p((TYPE *)ptr, value, peer);      \
  }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_TYPENAME_P_IMPL_PYBIND)
#undef NVSHMEMI_TYPENAME_P_IMPL_PYBIND

#define NVSHMEMI_TYPENAME_P_ON_STREAM_IMPL_PYBIND(TYPENAME, TYPE)                      \
  void TYPENAME##_p_on_stream(ptrdiff_t ptr, TYPE value, int peer, void *stream) {     \
    check_nvshmem_init();                                                              \
    nvshmemx_##TYPENAME##_p_on_stream((TYPE *)ptr, value, peer, (cudaStream_t)stream); \
  }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_TYPENAME_P_ON_STREAM_IMPL_PYBIND)
#undef NVSHMEMI_TYPENAME_P_ON_STREAM_IMPL_PYBIND
