//===- topo_utils.cc -------------------------------------------- C++ ---===//
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

#include "flux/ths_op/topo_utils.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <mutex>
#include <numeric>
#include <ostream>
#include <vector>
#include <set>
#include <iostream>
#include <ATen/ops/empty.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/cuda_stub.h"
#include "flux/cuda/nvml_stub.h"
#include "flux/flux.h"
#include "flux/ths_op/util.h"
#include "flux/utils.h"
#include "torch/csrc/distributed/c10d/Utils.hpp"
#include "flux/cuda/cuda_common.h"

#define MAX_BUSID_SIZE 16
#define MAXPATHSIZE 1024

namespace bytedance::flux::topo_utils {

bool
has_nvlink_support(int devid) {
  ensure_nvml_init();
  nvmlDevice_t device;
  NVML_CHECK(nvml_stub().nvmlDeviceGetHandleByIndex(devid, &device));
  nvmlEnableState_t enabled;
  nvmlReturn_t status = nvml_stub().nvmlDeviceGetNvLinkState(device, 0, &enabled);
  if (status == NVML_ERROR_NOT_SUPPORTED) {
    return false;
  }
  NVML_CHECK(status);
  return enabled == NVML_FEATURE_ENABLED;
}

struct TopoInfo {
  int world_size;
  int numa_world_size;    // if no NUMA detected, using world_size as default;
  int nvlink_world_size;  // 1 for no nvlink support. nvlink_world_size may not equal world_size
                          // without NvSwitch
};

static void
allgather_cpu(
    c10::intrusive_ptr<c10d::ProcessGroup> pg, const void *sendbuf, void *recvbuf, int length) {
  // cpu -> gpu -> gpu_gather -> cpu_gather
  auto option_gpu = at::TensorOptions()
                        .dtype(at::ScalarType::Byte)
                        .device(at::kCUDA)
                        .device_index(c10::cuda::current_device());
  auto option_cpu = at::TensorOptions().dtype(at::ScalarType::Byte).device(at::kCPU);

  auto src_cpu = at::from_blob(const_cast<void *>(sendbuf), length, option_cpu);
  auto dst_cpu = at::from_blob(const_cast<void *>(recvbuf), length * pg->getSize(), option_cpu);
  auto src_gpu = at::empty({length}, option_gpu);
  auto dst_gpu = at::empty({length * pg->getSize()}, option_gpu);
  src_gpu.copy_(src_cpu);
  auto work = pg->_allgather_base(dst_gpu, src_gpu);
  FLUX_CHECK(work->wait()) << "bootstrap_c10_pg_allgather hangs";
  dst_cpu.copy_(dst_gpu);
  LOG(INFO) << "bootstrap_c10_pg_allgather done";
}

std::array<char, MAX_BUSID_SIZE>
get_gpu_bus_id(CUdevice gpu_device_id) {
  std::array<char, MAX_BUSID_SIZE> bus_id;
  CUDA_CHECK(cudaDeviceGetPCIBusId(bus_id.data(), MAX_BUSID_SIZE, gpu_device_id));
  return bus_id;
}

static std::string
get_device_path(CUdevice gpu_device_id) {
  auto bus_id = get_gpu_bus_id(gpu_device_id);
  char pathname[MAXPATHSIZE + 1];
  char bus_path[] = "/sys/class/pci_bus/0000:00/device";

  for (int i = 0; i < MAX_BUSID_SIZE; i++) {
    bus_id[i] = tolower(bus_id[i]);
  }
  memcpy(bus_path + sizeof("/sys/class/pci_bus/") - 1, bus_id.data(), sizeof("0000:00") - 1);

  char *cuda_rpath = realpath(bus_path, NULL);
  CHECK(cuda_rpath != nullptr);

  strncpy(pathname, cuda_rpath, MAXPATHSIZE);
  strncpy(pathname + strlen(pathname), "/", MAXPATHSIZE - strlen(pathname));
  strncpy(pathname + strlen(pathname), bus_id.data(), MAXPATHSIZE - strlen(pathname));
  free(cuda_rpath);

  char *path = realpath(pathname, NULL);
  CHECK(path != nullptr);
  return std::string(path);
}

static int
get_numa_id(const std::string &path) {
  char npath[PATH_MAX];
  snprintf(npath, PATH_MAX, "%s/numa_node", path.c_str());
  npath[PATH_MAX - 1] = '\0';

  int numaId = -1;
  FILE *file = fopen(npath, "r");
  if (file == NULL)
    return -1;
  if (fscanf(file, "%d", &numaId) == EOF) {
    fclose(file);
    return -1;
  }
  fclose(file);

  return numaId;
}

static bool
init_topo(const std::vector<CUdevice> &gpu_device_ids, TopoInfo &topo_info) {
  int world_size = gpu_device_ids.size();
  CHECK(world_size > 0);
  topo_info.world_size = world_size;

  std::vector<int> numa_ids;
  std::vector<std::array<char, MAX_BUSID_SIZE>> bus_ids;
  std::vector<std::string> device_paths;
  for (int i = 0; i < world_size; i++) {
    bus_ids.emplace_back(get_gpu_bus_id(gpu_device_ids[i]));
    device_paths.emplace_back(get_device_path(gpu_device_ids[i]));
    numa_ids.emplace_back(get_numa_id(device_paths.back()));
  }

  auto is_all_values_the_same = [](const auto &v) {
    auto value = v[0];
    for (size_t i = 1; i < v.size(); i++) {
      if (v[i] != value) {
        return false;
      }
    }
    return true;
  };

  bool has_numa_support = numa_ids[0] >= 0;
  if (has_numa_support) {
    std::set<int> numa_id_set{numa_ids.begin(), numa_ids.end()};
    std::vector<int> numa_world_sizes;
    for (int numa_id : numa_id_set) {
      numa_world_sizes.push_back(std::count_if(
          numa_ids.begin(), numa_ids.end(), [numa_id](int x) { return x == numa_id; }));
    }
    CHECK(numa_world_sizes.size() > 0);
    bool is_even_distributed = is_all_values_the_same(numa_world_sizes);
    if (!is_even_distributed) {
      LOG(WARNING) << "nodes is not even distributed across NUMA cores";
      topo_info.numa_world_size = world_size;
    } else {
      topo_info.numa_world_size = numa_world_sizes[0];
    }
  } else {
    topo_info.numa_world_size = world_size;
  }

  std::vector<bool> has_nvlinks;
  for (int i = 0; i < world_size; i++) {
    has_nvlinks.push_back(has_nvlink_support(gpu_device_ids[i]));
  }
  bool has_nvlink = has_nvlinks[0];
  bool all_nvlink_the_same = is_all_values_the_same(has_nvlinks);
  if (!all_nvlink_the_same) {
    LOG(WARNING) << "nodes has different nvlink support";
    topo_info.nvlink_world_size = 1;
  } else if (has_nvlink) {
    std::vector<nvmlIntNvLinkDeviceType_t> nvlink_device_types;
    for (int i = 0; i < world_size; i++) {
      int gpu_device_id = gpu_device_ids[i];
      nvmlDevice_t device;
      ensure_nvml_init();
      NVML_CHECK(nvml_stub().nvmlDeviceGetHandleByIndex(gpu_device_id, &device));
      nvmlIntNvLinkDeviceType_t device_type;
#if HAS_NVMLDEVICEGETNVLINKREMOTEDEVICETYPE  // don't pollute the log
      try {
        NVML_CHECK(nvml_stub().nvmlDeviceGetNvLinkRemoteDeviceType(device, 0, &device_type));
      } catch (...) {
#endif
        // maybe old driver has no nvmlDeviceGetNvLinkRemoteDeviceType
        nvmlPciInfo_t pcie_info;
        NVML_CHECK(nvml_stub().nvmlDeviceGetNvLinkRemotePciInfo(device, 0, &pcie_info));
        auto to_char_array = [](char bytes[MAX_BUSID_SIZE]) {
          std::array<char, MAX_BUSID_SIZE> arr;
          std::memcpy(arr.data(), bytes, MAX_BUSID_SIZE);
          return arr;
        };
        if (std::find(bus_ids.begin(), bus_ids.end(), to_char_array(pcie_info.busIdLegacy)) ==
            bus_ids.end()) {
          LOG(INFO) << "nvlink device type is NVML_NVLINK_DEVICE_TYPE_SWITCH";
          device_type = NVML_NVLINK_DEVICE_TYPE_SWITCH;
        } else {
          device_type = NVML_NVLINK_DEVICE_TYPE_GPU;
        }
#if HAS_NVMLDEVICEGETNVLINKREMOTEDEVICETYPE
      }
#endif
      nvlink_device_types.push_back(device_type);
    }
    bool is_nvlink_homogeneous = is_all_values_the_same(nvlink_device_types);
    if (!is_nvlink_homogeneous || nvlink_device_types[0] != NVML_NVLINK_DEVICE_TYPE_SWITCH) {
      topo_info.nvlink_world_size = 2;  // treat as P2P NvLink
    } else {
      topo_info.nvlink_world_size = world_size;  // treat as all-to-all connection
    }
  } else {
    topo_info.nvlink_world_size = 1;
  }
  return true;
}

std::vector<CUdevice>
get_processs_group_devices(c10::intrusive_ptr<c10d::ProcessGroup> pg) {
  int world_size = pg->getSize();
  CHECK(world_size > 0);

  CUdevice gpu_device_id;
  std::vector<CUdevice> gpu_device_ids(world_size, -1);
  CU_CHECK(cuda_stub().cuCtxGetDevice(&gpu_device_id));
  // allgather gpu_device_id
  allgather_cpu(pg, &gpu_device_id, gpu_device_ids.data(), sizeof(int));
  std::set<CUdevice> gpu_device_set{gpu_device_ids.begin(), gpu_device_ids.end()};
  FLUX_CHECK_EQ(gpu_device_ids.size(), gpu_device_set.size());
  return gpu_device_ids;
}

bool
init_topo(c10::intrusive_ptr<c10d::ProcessGroup> pg, TopoInfo &topo_info) {
  return init_topo(get_processs_group_devices(pg), topo_info);
}

namespace {
static TopoInfo topo_info;
static bool initialized = false;
static std::mutex mutex;
}  // namespace

ncclComm_t
create_nccl_comm_with_processgroup(c10::intrusive_ptr<c10d::ProcessGroup> pg) {
  //-- NCCL--
  ncclComm_t nccl_comm;
  int rank = pg->getRank();
  auto stream = c10::cuda::getCurrentCUDAStream();
  void *hptr_id = nullptr;
  CUDA_CHECK(cudaMallocHost(&hptr_id, sizeof(ncclUniqueId)));

  ncclUniqueId &id = *static_cast<ncclUniqueId *>(hptr_id);
  if (rank == 0) {
    ncclGetUniqueId(&id);
  }

  std::vector<at::Tensor> src_gpus{at::empty(
      {sizeof(ncclUniqueId)},
      at::TensorOptions()
          .dtype(at::ScalarType::Byte)
          .device(at::kCUDA)
          .device_index(c10::cuda::current_device()))};
  c10::cuda::memcpy_and_sync(
      src_gpus[0].data_ptr(), hptr_id, sizeof(ncclUniqueId), cudaMemcpyHostToDevice, stream);

  auto work = pg->broadcast(src_gpus);
  work->wait();
  c10::cuda::memcpy_and_sync(
      hptr_id, src_gpus[0].data_ptr(), sizeof(ncclUniqueId), cudaMemcpyDeviceToHost, stream);
  NCCL_CHECK(ncclCommInitRank(&nccl_comm, pg->getSize(), id, rank));
  CUDA_CHECK(cudaFreeHost(hptr_id));
  return nccl_comm;
}

bool
is_topo_initialized() {
  std::lock_guard<std::mutex> _(mutex);
  return initialized;
}

void
initialize_topo(const std::vector<int> &gpu_device_ids) {
  std::lock_guard<std::mutex> _(mutex);
  if (initialized) {
    FLUX_LOG_FIRST_N(INFO, 1) << "topology is already initialized";
    return;
  }
  init_topo(gpu_device_ids, topo_info);
  initialized = true;
  // create a ncclComm_t only to get the topo
  LOG(INFO) << "topo_info:" << topo_info.world_size << " numa size " << topo_info.numa_world_size
            << " nvlink size " << topo_info.nvlink_world_size;
}

void
initialize_topo(c10::intrusive_ptr<c10d::ProcessGroup> group) {
  initialize_topo(get_processs_group_devices(group));
}

bool
has_nvswitch() {
  std::lock_guard<std::mutex> _(mutex);
  FLUX_CHECK(initialized);
  return topo_info.nvlink_world_size == topo_info.world_size;
}

bool
has_heterogeneous_nvlink() {  // has NVLink but no NVSwitch
  std::lock_guard<std::mutex> _(mutex);
  FLUX_CHECK(initialized);
  return topo_info.nvlink_world_size > 1 && topo_info.world_size != topo_info.nvlink_world_size;
}

int
topo_nvlink_local_world_size() {
  std::lock_guard<std::mutex> _(mutex);
  FLUX_CHECK(initialized);
  return topo_info.nvlink_world_size;
}

bool
has_nvlink() {
  std::lock_guard<std::mutex> _(mutex);
  FLUX_CHECK(initialized);
  return topo_info.nvlink_world_size > 1;
}

bool
has_heterogeneous_pcie() {
  std::lock_guard<std::mutex> _(mutex);
  FLUX_CHECK(initialized);
  return topo_info.numa_world_size != topo_info.world_size;
}

int
topo_numa_local_world_size() {
  std::lock_guard<std::mutex> _(mutex);
  FLUX_CHECK(initialized);
  return topo_info.numa_world_size;
}

}  // namespace bytedance::flux::topo_utils
