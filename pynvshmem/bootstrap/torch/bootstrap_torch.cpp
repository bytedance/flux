//===- boostrap_torch.cpp ----------------------------------------- C++ ---===//
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

#include "host/nvshmemx_error.h"
#include "modules/bootstrap/bootstrap_util.h"
#include "modules/bootstrap/nvshmemi_bootstrap.h"

#include <ATen/core/TensorBase.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/util/Logging.h>
#include <cassert>
#include <stdint.h>
#include <stdlib.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

// don't own this. make use this lives longer than using this
static c10d::ProcessGroup *pg = nullptr;
static int nvshmem_initialized_torch = 0;

static int
bootstrap_c10_pg_barrier(struct bootstrap_handle *handle) {
  LOG(INFO) << "bootstrap_c10_pg_barrier start with c10d::ProcessGroup " << pg;
  assert(pg != nullptr);
  auto work = pg->barrier();
  if (!work->wait()) {
    LOG(ERROR) << "bootstrap_c10_pg_barrier hangs";
    return NVSHMEMX_ERROR_INTERNAL;
  }
  LOG(INFO) << "bootstrap_c10_pg_barrier done";
  return NVSHMEMX_SUCCESS;
}

static int
bootstrap_c10_pg_allgather(
    const void *sendbuf, void *recvbuf, int length, struct bootstrap_handle *handle) {
  LOG(INFO) << "bootstrap_c10_pg_allgather start with size " << length << " c10d::ProcessGroup "
            << pg << " pg_size " << handle->pg_size;

  assert(pg != nullptr);
  // cpu -> gpu -> gpu_gather -> cpu_gather
  auto option_gpu = at::TensorOptions()
                        .dtype(at::ScalarType::Byte)
                        .device(at::kCUDA)
                        .device_index(c10::cuda::current_device());
  auto option_cpu = at::TensorOptions().dtype(at::ScalarType::Byte).device(at::kCPU);

  auto src_cpu = at::from_blob(const_cast<void *>(sendbuf), length, option_cpu);
  auto dst_cpu = at::from_blob(const_cast<void *>(recvbuf), length * handle->pg_size, option_cpu);
  auto src_gpu = at::empty({length}, option_gpu);
  auto dst_gpu = at::empty({length * handle->pg_size}, option_gpu);
  src_gpu.copy_(src_cpu);
  auto work = pg->_allgather_base(dst_gpu, src_gpu);
  if (!work->wait()) {
    LOG(ERROR) << "bootstrap_c10_pg_allgather hangs";
    return NVSHMEMX_ERROR_INTERNAL;
  }
  dst_cpu.copy_(dst_gpu);
  LOG(INFO) << "bootstrap_c10_pg_allgather done";
  return NVSHMEMX_SUCCESS;
}

static int
bootstrap_c10_pg_alltoall(
    const void *sendbuf, void *recvbuf, int length, struct bootstrap_handle *handle) {
  LOG(INFO) << "bootstrap_c10_pg_alltoall start with size " << length << " c10d::ProcessGroup "
            << pg;
  assert(pg != nullptr);
  // cpu -> gpu -> gpu_gather -> cpu_gather
  auto option_gpu = at::TensorOptions()
                        .dtype(at::ScalarType::Byte)
                        .device(at::kCUDA)
                        .device_index(c10::cuda::current_device());
  auto option_cpu = at::TensorOptions().dtype(at::ScalarType::Byte).device(at::kCPU);

  length = length * handle->pg_size;
  auto src_cpu = at::from_blob(const_cast<void *>(sendbuf), length, option_cpu);
  auto dst_cpu = at::from_blob(const_cast<void *>(recvbuf), length, option_cpu);
  auto src_gpu = at::empty({length}, option_gpu);
  auto dst_gpu = at::empty({length}, option_gpu);
  src_gpu.copy_(src_cpu);
  std::vector<int64_t> outputSplitSizes, inputSplitSizes;
  auto work = pg->alltoall_base(dst_gpu, src_gpu, outputSplitSizes, inputSplitSizes);
  if (!work->wait()) {
    LOG(ERROR) << "bootstrap_c10_pg_alltoall hangs";
    return NVSHMEMX_ERROR_INTERNAL;
  }
  dst_cpu.copy_(dst_gpu);
  LOG(INFO) << "bootstrap_c10_pg_alltoall done";
  return NVSHMEMX_SUCCESS;
}

static void
bootstrap_c10_pg_global_exit(int status) {
  // leave it to c10d::ProcessGroup owner: do nothing
  LOG(INFO) << "bootstrap_c10_pg_global_exit: do nothing";
}

static int
bootstrap_c10_pg_finalize(bootstrap_handle_t *handle) {
  // leave it to c10d::ProcessGroup owner: do nothing
  LOG(INFO) << "bootstrap_c10_pg_finalize: do nothing";
out:
  return NVSHMEMX_SUCCESS;
}

int
nvshmemi_bootstrap_plugin_init(void *attr, bootstrap_handle_t *handle, const int abi_version) {
  if (attr == nullptr) {
    LOG(ERROR) << "nvshmemi_bootstrap_plugin_init with attr == nullptr ";
    return NVSHMEMX_ERROR_INVALID_VALUE;
  }
  int bootstrap_version = NVSHMEMI_BOOTSTRAP_ABI_VERSION;
  if (!nvshmemi_is_bootstrap_compatible(bootstrap_version, abi_version)) {
    BOOTSTRAP_ERROR_PRINT(
        "Torch bootstrap version (%d) is not compatible with "
        "NVSHMEM version (%d)",
        bootstrap_version,
        abi_version);
    exit(-1);
  }

  if (!nvshmem_initialized_torch) {
    pg = (c10d::ProcessGroup *)attr;
    CHECK(pg != nullptr);
    handle->pg_rank = pg->getRank();
    handle->pg_size = pg->getSize();

    handle->allgather = bootstrap_c10_pg_allgather;
    handle->alltoall = bootstrap_c10_pg_alltoall;
    handle->barrier = bootstrap_c10_pg_barrier;
    handle->global_exit = bootstrap_c10_pg_global_exit;
    handle->finalize = bootstrap_c10_pg_finalize;
    nvshmem_initialized_torch = 1;
  } else {
    LOG(WARNING) << "nvshmemi_bootstrap_plugin_init already initialized!";
  }
  return NVSHMEMX_SUCCESS;
}
