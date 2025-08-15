//===- flux_shm.cc ---------------------------------------------- C++ ---===//
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
#include "flux/ths_op/flux_shm.h"

#include <ATen/ops/from_blob.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/intrusive_ptr.h>
#include <errno.h>     // for errno
#include <sys/mman.h>  // for mmap, shm_open, munmap, shm_unlink
#include <sys/stat.h>  // for fstat, stat
#include <torch/cuda.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <memory>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <vector>

#include "flux/cuda/cuda_common.h"
#include "flux/cuda/helper_kernels.h"
#include "flux/flux.h"
#include "flux/utils.h"
#ifdef FLUX_SHM_USE_NVSHMEM
#include <nvshmemx.h>
#endif

namespace bytedance::flux {

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
nvshmem_create_tensor(const std::vector<int64_t> &shape, c10::ScalarType dtype, bool init_zero) {
  check_nvshmem_init();
  auto current_device = c10::cuda::current_device();
  auto option_gpu =
      at::TensorOptions().dtype(dtype).device(at::kCUDA).device_index(current_device);
  int64_t element_size = torch::elementSize(dtype);
  int64_t count = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<>());
  int64_t size = element_size * count;

  FLUX_CHECK(size != 0);
  at::cuda::device_synchronize();
  void *ptr;
  if (init_zero) {
    // https://docs.nvidia.com/nvshmem/api/gen/api/memory.html#c.nvshmem_calloc
    ptr = nvshmem_calloc(count, element_size);
  } else {
    ptr = nvshmem_malloc(size);
  }
  FLUX_CHECK(ptr != nullptr) << "NVSHMEM_MALLOC failed, please set larger "
                                "NVSHMEM_SYMMETRIC_SIZE(1000000000 by default)\n";

  return at::from_blob(
      ptr,
      shape,
      [=](void *ptr) {
        at::cuda::CUDAGuard guard(current_device);
        at::cuda::device_synchronize();
        nvshmem_free(ptr);
      },
      option_gpu);
}

std::vector<torch::Tensor>
nvshmem_create_tensor_list(
    const std::vector<int64_t> &shape, c10::ScalarType dtype, bool init_zero) {
  check_nvshmem_init();
  auto current_device = c10::cuda::current_device();
  auto option_gpu = at::TensorOptions(at::kCUDA).dtype(dtype).device_index(current_device);
  auto element_size = torch::elementSize(dtype);
  auto count = std::accumulate(shape.begin(), shape.end(), (size_t)1, std::multiplies<>());
  auto size = element_size * count;
  FLUX_CHECK_NE(size, 0);
  int local_world_size = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
  int rank = nvshmem_my_pe();
  int local_rank = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  std::vector<torch::Tensor> tensors;
  tensors.reserve(local_world_size);

  at::cuda::device_synchronize();
  void *ptr;
  if (init_zero) {
    ptr = nvshmem_calloc(count, element_size);
  } else {
    ptr = nvshmem_malloc(size);
  }
  FLUX_CHECK(ptr != nullptr);

  int rank_offset = rank - local_rank;
  for (int i = 0; i < local_world_size; i++) {
    // runs this call nvshmem failure, don't know why
    //  nvshmem_team_translate_pe(NVSHMEMX_TEAM_NODE, local_rank, NVSHMEM_TEAM_WORLD)
    int rank_global = i + rank_offset;
    if (rank == rank_global) {
      tensors.emplace_back(
          at::from_blob(
              ptr,
              shape,
              [=](void *ptr) {
                // std::cerr << "enter nvshmem_free " << ptr << "\n";
                at::cuda::CUDAGuard guard(current_device);
                at::cuda::device_synchronize();
                // std::cerr << "do nvshmem_free " << ptr << "\n";
                nvshmem_free(ptr);
                at::cuda::device_synchronize();
                // std::cerr << "exit nvshmem_free " << ptr << "\n";
              },
              option_gpu));
    } else {
      void *rptr = nvshmem_ptr(ptr, rank_global);
      FLUX_CHECK(rptr != nullptr) << "rank " << rank;
      tensors.emplace_back(at::from_blob(rptr, shape, option_gpu));
    }
  }

  return tensors;
}

#endif

class C10dProcessGroup::Impl {
 public:
  Impl(const std::string &name, c10::intrusive_ptr<c10d::ProcessGroup> pg)
      : name_(name), pg_(pg) {}

  // std::string
  // get_group_name() const override {
  //   return name_;
  // }

  void
  all_gather_cpu(const void *src, void *dst, int64_t nbytes) {
    if (pg_->getBackendType() == c10d::ProcessGroup::NCCL) {
      all_gather_cpu_by_gpu(src, dst, nbytes);
    } else {
      all_gather_cpu_direct(src, dst, nbytes);
    }
  }

  void
  broadcast_cpu(void *ptr, int64_t nbytes, int root_rank) {
    if (pg_->getBackendType() == c10d::ProcessGroup::NCCL) {
      broadcast_cpu_by_gpu(ptr, nbytes, root_rank);
    } else {
      broadcast_cpu_direct(ptr, nbytes, root_rank);
    }
  }

  int
  get_rank() {
    return pg_->getRank();
  }
  int
  get_size() {
    return pg_->getSize();
  }

  void
  sync() {
    pg_->barrier()->wait();
  }

 private:
  void
  all_gather_cpu_by_gpu(const void *src, void *dst, int64_t nbytes) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto option = at::TensorOptions(torch::kUInt8)
                      .device(at::kCUDA)
                      .device_index(c10::cuda::current_device());
    auto gpu_buffer = torch::empty(nbytes, option);
    auto gpu_buffer_full = torch::empty({static_cast<long>(nbytes * pg_->getSize())}, option);
    CUDA_CHECK(
        cudaMemcpyAsync(gpu_buffer.data_ptr(), src, nbytes, cudaMemcpyHostToDevice, stream));
    pg_->_allgather_base(gpu_buffer_full, gpu_buffer)->wait();
    CUDA_CHECK(cudaMemcpyAsync(
        dst, gpu_buffer_full.data_ptr(), nbytes * pg_->getSize(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void
  all_gather_cpu_direct(const void *src, void *dst, int64_t nbytes) {
    auto option = at::TensorOptions(torch::kUInt8).device(at::kCPU);
    auto dst_tensor = at::from_blob(dst, {nbytes * pg_->getSize()}, option);
    auto src_tensor = at::from_blob(const_cast<void *>(src), {nbytes}, option);
    pg_->_allgather_base(dst_tensor, src_tensor)->wait();
  }

  void
  broadcast_cpu_by_gpu(void *ptr, int64_t nbytes, int root_rank) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto tensors = std::vector<torch::Tensor>{
        at::empty({nbytes}, at::TensorOptions(torch::kUInt8).device(at::kCUDA))};
    auto opt = c10d::BroadcastOptions();
    opt.rootRank = root_rank;
    CUDA_CHECK(
        cudaMemcpyAsync(tensors[0].data_ptr(), ptr, nbytes, cudaMemcpyHostToDevice, stream));
    pg_->broadcast(tensors, opt)->wait();
    CUDA_CHECK(
        cudaMemcpyAsync(ptr, tensors[0].data_ptr(), nbytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void
  broadcast_cpu_direct(void *ptr, int64_t nbytes, int root_rank) {
    auto tensors = std::vector<torch::Tensor>{
        at::from_blob(ptr, {nbytes}, at::TensorOptions(torch::kUInt8).device(at::kCPU))};
    auto opt = c10d::BroadcastOptions();
    opt.rootRank = root_rank;
    pg_->broadcast(tensors, opt)->wait();
  }

 private:
  std::string name_;
  c10::intrusive_ptr<c10d::ProcessGroup> pg_;
};

C10dProcessGroup::C10dProcessGroup(
    const std::string &name, c10::intrusive_ptr<c10d::ProcessGroup> pg)
    : impl_(new Impl(name, pg)) {}
C10dProcessGroup::~C10dProcessGroup() { delete impl_; }

int
C10dProcessGroup::get_rank() {
  return impl_->get_rank();
}
int
C10dProcessGroup::get_size() {
  return impl_->get_size();
}

void
C10dProcessGroup::sync() {
  impl_->sync();
}
void
C10dProcessGroup::all_gather_cpu(const void *src, void *dst, int64_t nbytes) {
  impl_->all_gather_cpu(src, dst, nbytes);
}

void
C10dProcessGroup::broadcast_cpu(void *ptr, int64_t nbytes, int root_rank) {
  impl_->broadcast_cpu(ptr, nbytes, root_rank);
}

std::vector<torch::Tensor>
cudaipc_create_tensor_list(
    Group *group,
    const std::vector<int64_t> &shape,
    c10::ScalarType dtype,
    bool ring_mode,
    bool init_zero) {
  FLUX_CHECK(group != nullptr);
  int cur_rank = group->get_rank();
  int world_size = group->get_size();

  // If ring mode is disabled (ring_mode == false), the function will return tensors from all
  // peers.
  // This means that the tensors collected from all available ranks in the communication group will
  // be included in the return value.
  //
  // If ring mode is enabled (ring_mode == true), the function will also returns a list of tensors:
  // where only the adjacent tensors has the correct address and the others are set with the start
  // address of current rank

  // This ring mode behavior is typically used in scenarios where the p2p protocol is not worked
  // any more such as the number of peers exceeds 8.
  auto option_gpu =
      at::TensorOptions(dtype).device(at::kCUDA).device_index(c10::cuda::current_device());

  FLUX_CHECK_LE(world_size, torch::cuda::device_count())
      << "create_ipc_tensors should only be used intra node";
  auto size = torch::elementSize(dtype) *
              std::accumulate(shape.begin(), shape.end(), (size_t)1, std::multiplies<>());
  FLUX_CHECK(size != 0);
  void *ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&ptr, size));
  if (init_zero) {
    CUDA_CHECK(cudaMemset(ptr, 0, size));  // memset the allocated buffer
  }
  cudaIpcMemHandle_t handle;
  CUDA_CHECK(cudaIpcGetMemHandle(&handle, ptr));
  std::vector<cudaIpcMemHandle_t> handles(world_size);
  group->all_gather_cpu(&handle, handles.data(), sizeof(cudaIpcMemHandle_t));

  int prev_peer = (cur_rank - 1 + world_size) % world_size;
  int next_peer = (cur_rank + 1) % world_size;
  std::vector<torch::Tensor> tensors;
  std::vector<void *> ptrs(world_size);
  std::set<int> neighbors({prev_peer, cur_rank, next_peer});
  if (ring_mode == false) {
    for (int i = 0; i < world_size; ++i) {
      if (i != cur_rank) {
        CUDA_CHECK(cudaIpcOpenMemHandle(&ptrs[i], handles[i], cudaIpcMemLazyEnablePeerAccess));
      } else {
        ptrs[i] = ptr;
      }
    }

    for (int i = 0; i < world_size; ++i) {
      torch::Tensor tensor;
      if (i == cur_rank) {
        tensor = at::from_blob(ptr, shape, [](void *ptr) { cudaFree(ptr); }, option_gpu);
      } else {
        tensor = at::from_blob(
            ptrs[i], shape, [](void *ptr) { cudaIpcCloseMemHandle(ptr); }, option_gpu);
      }
      tensors.emplace_back(tensor);
    }
  } else {
    for (int i = 0; i < world_size; ++i) {
      if (i == cur_rank) {
        ptrs[i] = ptr;
      } else if (neighbors.count(i) == 1) {
        CUDA_CHECK(cudaIpcOpenMemHandle(&ptrs[i], handles[i], cudaIpcMemLazyEnablePeerAccess));
      } else {
        ptrs[i] = nullptr;
      }
    }
    for (int i = 0; i < world_size; ++i) {
      torch::Tensor tensor;
      if (i == cur_rank) {
        tensor = at::from_blob(ptr, shape, [](void *ptr) { cudaFree(ptr); }, option_gpu);
      } else if (neighbors.count(i)) {
        tensor = at::from_blob(
            ptrs[i], shape, [](void *ptr) { cudaIpcCloseMemHandle(ptr); }, option_gpu);
      } else {
        // For non-adjacent ranks, use the current rank’s sync buffer to fill them.
        // Undefined(nullptr) tensors are treated as None during pybind transmission, resulting in
        // an invalid type.
        tensor = at::from_blob(ptr, shape, [](void *ptr) {}, option_gpu);
      }
      tensors.emplace_back(tensor);
    }
  }
  return tensors;
}

std::vector<torch::Tensor>
cudaipc_create_tensor_list(
    c10::intrusive_ptr<c10d::ProcessGroup> pg,
    const std::vector<int64_t> &shape,
    c10::ScalarType dtype,
    bool ring_mode,
    bool init_zero) {
  auto group = std::make_unique<C10dProcessGroup>("", pg);
  return cudaipc_create_tensor_list(group.get(), shape, dtype, ring_mode, init_zero);
}

void
init_flux_shm(Group *group) {
  // Guarantee that cudaCtx has been created. If cudaCtx is not created on some ranks, which may
  // cause nvshmemx_init_attr to hang. Empirically, we can try to create a cudaCtx on the device by
  // calling cudaFree(0), although the offical doc describes that no operation is performed.
  CUDA_CHECK(cudaFree(0));

  int rank = group->get_rank();
  int nranks = group->get_size();
#ifdef FLUX_SHM_USE_NVSHMEM
  // if (group->get_size() > kMaxLocalWorldSize) {
  //   fprintf(
  //       stderr,
  //       "Current node has %d devices, will use the ring-implementation in the
  //       flux_shm_barrier\n", group->get_size());
  //   return;
  // }
  nvshmemx_uniqueid_t id;
  if (rank == 0) {
    nvshmemx_get_uniqueid(&id);
  }
  group->broadcast_cpu(&id, sizeof(id), 0);

  nvshmemx_init_attr_t init_attr;
  nvshmemx_set_attr_uniqueid_args(rank, nranks, &id, &init_attr);
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &init_attr);
  int mype = nvshmem_my_pe();
  FLUX_CHECK_EQ(rank, mype) << "NVShmem init: rank does not match PE!";
#endif
}

void
init_flux_shm(c10::intrusive_ptr<c10d::ProcessGroup> c10_pg) {
  auto group = std::make_unique<C10dProcessGroup>("", c10_pg);
  init_flux_shm(group.get());
}

torch::Tensor
flux_create_tensor(
    const std::vector<int64_t> &shape,
    c10::ScalarType dtype,
    c10::intrusive_ptr<c10d::ProcessGroup> pg) {
#ifdef FLUX_SHM_USE_NVSHMEM
  if (torch::cuda::device_count() <= 8) {
    return nvshmem_create_tensor(shape, dtype);
  } else {
    FLUX_CHECK(false && "This line should never be reached");
    return torch::Tensor();
  }
#else
  FLUX_CHECK(false && "This line should never be reached");
  return torch::Tensor();
#endif
}

std::vector<torch::Tensor>
flux_create_tensor_list(
    const std::vector<int64_t> &shape,
    c10::ScalarType dtype,
    Group *group,
    bool ring_mode,
    bool init_zero) {
#ifdef FLUX_SHM_USE_NVSHMEM
  static bool use_nvshmem = get_bool_from_env("FLUX_USE_NVSHMEM", true);
  if (torch::cuda::device_count() <= kMaxLocalWorldSize && use_nvshmem) {
    return nvshmem_create_tensor_list(shape, dtype, init_zero);
  } else {
    return cudaipc_create_tensor_list(group, shape, dtype, ring_mode, init_zero);
  }
#else
  FLUX_CHECK(group != nullptr);
  return cudaipc_create_tensor_list(group, shape, dtype, ring_mode, init_zero);
#endif
}

std::vector<torch::Tensor>
flux_create_tensor_list(
    const std::vector<int64_t> &shape,
    c10::ScalarType dtype,
    c10::intrusive_ptr<c10d::ProcessGroup> pg,
    bool ring_mode,
    bool init_zero) {
  auto group = std::make_unique<C10dProcessGroup>("", pg);
  return flux_create_tensor_list(shape, dtype, group.get(), ring_mode, init_zero);
}

void
cudaipc_barrier_all_on_stream(
    cudaStream_t stream,
    c10::optional<std::vector<torch::Tensor>> sync_buffers,
    c10::optional<int> rank,
    bool ring_mode) {
  FLUX_CHECK(sync_buffers.has_value());
  FLUX_CHECK(rank.has_value());
  std::vector<int32_t *> sync_buffer_ptrs;
  auto sync_buffers_val = sync_buffers.value();

  FLUX_CHECK(sync_buffers_val[rank.value()].defined());

  int world_size = sync_buffers_val.size();
  for (int i = 0; i < sync_buffers_val.size(); i++) {
    sync_buffer_ptrs.push_back(reinterpret_cast<int32_t *>(sync_buffers_val[i].data_ptr()));
  }
  cudaipc_barrier_all_on_stream_impl(
      stream, sync_buffer_ptrs.data(), rank.value(), world_size, ring_mode);
}

void
flux_barrier_all_on_stream(
    cudaStream_t stream,
    c10::optional<std::vector<torch::Tensor>> sync_buffers,
    c10::optional<int> rank,
    bool ring_mode,
    bool force_flux_impl) {
#ifdef FLUX_SHM_USE_NVSHMEM
  if (!ring_mode && !force_flux_impl)
    nvshmemx_barrier_all_on_stream(stream);
  else
    cudaipc_barrier_all_on_stream(stream, sync_buffers, rank, ring_mode);
#else
  cudaipc_barrier_all_on_stream(stream, sync_buffers, rank, ring_mode);
#endif
}

#define CHECK_SYS(rtn)

namespace {
struct ShmHandle {  // like NCCL does
  int shm_fd = 0;
  char *shm_path = nullptr;
  void *ptr = nullptr;
  void *dptr = nullptr;
  size_t size = 0;
  size_t real_size = 0;
  int *refcount = nullptr;
};

}  // namespace

#define FLUX_SHM_PATH_MAXLEN 64

void
shared_memory_open_or_die(
    char *shm_path, size_t shm_size, int refcount_allowed, ShmHandle *handle) {
  handle->size = shm_size;
  handle->real_size = shm_size + sizeof(int);
  bool create = refcount_allowed > 0;
  if (create) {
    char path[FLUX_SHM_PATH_MAXLEN] = {0};
    if (shm_path == nullptr) {
      sprintf(path, "/dev/shm/flux-XXXXXX");
      shm_path = path;
      handle->shm_fd = mkstemp(shm_path);
      FLUX_CHECK(handle->shm_fd > 0) << " mkstemp " << shm_path << " failed : " << strerror(errno);
    } else {
      FLUX_CHECK(open(shm_path, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR))
          << " open " << shm_path << " failed: " << strerror(errno);
    }

    FLUX_CHECK_EQ(fallocate(handle->shm_fd, 0, 0, handle->real_size), 0)
        << "ftruncate failed: " << strerror(errno);
  } else {
    FLUX_CHECK(shm_path != nullptr) << "no shm path to open";
    handle->shm_fd = open(shm_path, O_RDWR, S_IRUSR | S_IWUSR);
    FLUX_CHECK(handle->shm_fd > 0) << " open " << shm_path << " failed " << strerror(errno);
  }

  handle->ptr = mmap(0, handle->real_size, PROT_READ | PROT_WRITE, MAP_SHARED, handle->shm_fd, 0);
  FLUX_CHECK(handle->ptr != MAP_FAILED) << " mmap failed " << strerror(errno);

  handle->refcount = (int *)((char *)handle->ptr + handle->real_size);
  std::atomic<int> refcount(*handle->refcount);
  if (create) {
    refcount.store(refcount_allowed);
  } else {
    // each time 1 peer attach, refcount decrease by 1. if all attached, unlink
    if (refcount.fetch_sub(1) == 1) {
      if (unlink(shm_path) != 0) {
        std::cerr << "unlink shared memory " << shm_path << " error " << strerror(errno);
      }
    }
  }

  CUDA_CHECK(cudaHostRegister(handle->ptr, handle->real_size, cudaHostRegisterMapped));
  CUDA_CHECK(cudaHostGetDevicePointer(&handle->dptr, handle->ptr, 0));
  if (create) {
    memset((char *)handle->ptr, 0, handle->size);
    int slen = strlen(shm_path);
    handle->shm_path = new char[slen + 1];
    memcpy(handle->shm_path, shm_path, slen + 1);
  } else {
    handle->shm_path = nullptr;
  }
}

void
shared_memory_close_or_die(ShmHandle handle) {
  if (handle.shm_fd >= 0) {
    if (int rtn = close(handle.shm_fd)) {
      fprintf(stderr, "close fd by shared memory failed: %s", strerror(errno));
    }
    if (handle.shm_path != nullptr && handle.refcount != nullptr && *handle.refcount > 0) {
      if (unlink(handle.shm_path) != 0) {
        fprintf(
            stderr,
            "[WARN] unlink shared memory %s failed, error: %s",
            handle.shm_path,
            strerror(errno));
      }
    }
    delete[] handle.shm_path;
  }

  if (handle.ptr) {
    if (handle.dptr) {
      CUDA_CHECK(cudaHostUnregister(handle.ptr));
    }
    if (munmap(handle.ptr, handle.real_size) != 0) {
      fprintf(
          stderr,
          "[WARN] munmap of shared memory %p size %ld failed, error: %s",
          handle.ptr,
          handle.real_size,
          strerror(errno));
    }
  }
}

std::vector<torch::Tensor>
flux_create_shm_tensor_list(
    const std::vector<int64_t> &shape, c10::ScalarType dtype, Group *group) {
  FLUX_CHECK(group != nullptr);
  int rank = group->get_rank(), world_size = group->get_size();
  ShmHandle handle;
  size_t heap_size = torch::elementSize(dtype) *
                     std::accumulate(shape.begin(), shape.end(), (size_t)1, std::multiplies<>());

  // each create
  shared_memory_open_or_die(nullptr, heap_size, world_size - 1, &handle);

  std::vector<char> shm_names(world_size * FLUX_SHM_PATH_MAXLEN, '\0');
  group->all_gather_cpu(handle.shm_path, shm_names.data(), FLUX_SHM_PATH_MAXLEN);

  // atexit(close_sysmem_shm_file); // TODO(houiq.1993)
  /* Do first touch, for NUMA awareness */
  auto option = at::TensorOptions().dtype(dtype).device(at::kCPU).pinned_memory(true);
  std::vector<torch::Tensor> tensors;
  for (int i = 0; i < world_size; i++) {
    if (i == rank) {
      tensors.emplace_back(
          at::from_blob(
              handle.dptr, shape, [=](void *ptr) { shared_memory_close_or_die(handle); }, option));
    } else {
      char *shm_path = shm_names.data() + FLUX_SHM_PATH_MAXLEN * i;
      ShmHandle tmp_handle;
      shared_memory_open_or_die(shm_path, heap_size, -1, &tmp_handle);
      tensors.emplace_back(
          at::from_blob(
              tmp_handle.dptr,
              shape,
              [=](void *ptr) { shared_memory_close_or_die(tmp_handle); },
              option));
    }
  }
  group->sync();
  return tensors;
}

std::vector<torch::Tensor>
flux_create_shm_tensor_list(
    const std::vector<int64_t> &shape,
    c10::ScalarType dtype,
    c10::intrusive_ptr<c10d::ProcessGroup> pg) {
  auto group = std::make_unique<C10dProcessGroup>("", pg);
  return flux_create_shm_tensor_list(shape, dtype, group.get());
}

class BarrierInterface {
 public:
  virtual void barrier_all(cudaStream_t stream) = 0;
};

#if defined(FLUX_SHM_USE_NVSHMEM)
class NvshmemGroupBarrier : public BarrierInterface {
 public:
  NvshmemGroupBarrier(bool ring_mode = false) {
    FLUX_CHECK(ring_mode == false) << "nvshmem does not support ring_mode barrier";
  }

  void
  barrier_all(cudaStream_t stream) override {
    nvshmemx_barrier_all_on_stream(stream);
  }
};
#endif

class FluxGroupBarrier : public BarrierInterface {
 public:
  FluxGroupBarrier(std::shared_ptr<Group> pg, bool ring_mode = false)
      : pg_(pg), ring_mode_(ring_mode) {
    sync_buffers_ = cudaipc_create_tensor_list(
        this->pg_.get(), {pg_->get_size()}, c10::ScalarType::Int, ring_mode_);
    this->sync_buffers_[pg_->get_rank()].zero_();
  }

  void
  barrier_all(cudaStream_t stream) override {
    cudaipc_barrier_all_on_stream(
        stream, this->sync_buffers_, this->pg_->get_rank(), this->ring_mode_);
  }

 private:
  std::shared_ptr<Group> pg_;
  bool ring_mode_ = false;
  std::vector<torch::Tensor> sync_buffers_;
};

class GroupBarrier::BarrierImpl {
 public:
  BarrierImpl(std::shared_ptr<Group> group, bool ring_mode, bool force_flux_impl) {
    bool force_ring_mode = torch::cuda::device_count() > kMaxLocalWorldSize;
    if (force_ring_mode) {
      FLUX_LOG_FIRST_N(INFO, 1) << "too many devices. use ring_mode for barrier instead";
      ring_mode = true;
    }
    static bool use_nvshmem = get_bool_from_env("FLUX_USE_NVSHMEM", true);
    if (ring_mode || !use_nvshmem || force_flux_impl) {  // always use flux barrier for ring mode
      impl_ = std::make_unique<FluxGroupBarrier>(group, ring_mode);
    } else {
#if defined(FLUX_SHM_USE_NVSHMEM)
      impl_ = std::make_unique<NvshmemGroupBarrier>(ring_mode);
#else
      impl_ = std::make_unique<FluxGroupBarrier>(group, ring_mode);
#endif
    }
  }
  void
  barrier_all(cudaStream_t stream) {
    impl_->barrier_all(stream);
  }

 private:
  std::unique_ptr<BarrierInterface> impl_;
};

GroupBarrier::GroupBarrier(std::shared_ptr<Group> pg, bool ring_mode, bool force_flux_impl) {
  impl_ = new BarrierImpl(std::move(pg), ring_mode, force_flux_impl);
}

void
GroupBarrier::barrier_all(cudaStream_t stream) {
  impl_->barrier_all(stream);
}

GroupBarrier::~GroupBarrier() { delete impl_; }

float
all_reduce_max_float(Group *group, const float src) {
  FLUX_CHECK(group != nullptr);
  std::vector<float> vec(group->get_size(), 0);
  group->all_gather_cpu(&src, vec.data(), sizeof(float));
  return *std::max_element(vec.begin(), vec.end());
}

}  // namespace bytedance::flux
