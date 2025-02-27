//===- test_topo_utils.cc ----------------------------------------- C++ ---===//
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

#include <stdio.h>
#include <stdlib.h>
#include <numeric>
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/nvml_stub.h"
#include "flux/ths_op/topo_utils.h"
using bytedance::flux::nvml_stub;

int
main() {
  NVML_CHECK(nvmlInit());
  nvmlDevice_t device;
  NVML_CHECK(nvml_stub().nvmlDeviceGetHandleByIndex_v2(0, &device));
  unsigned int cap_out = 0;
  for (int cap = 0; cap < NVML_NVLINK_CAP_COUNT; cap++) {
    NVML_CHECK(nvml_stub().nvmlDeviceGetNvLinkCapability(
        device, 0, (nvmlNvLinkCapability_t)cap, &cap_out));
    printf("cap: %d, cap_out: %d\n", cap, cap_out);
  }

  unsigned int version;
  NVML_CHECK(nvml_stub().nvmlDeviceGetNvLinkVersion(device, 0, &version));
  printf("version: %d\n", version);  // got 5 on pci-e machine

  nvmlEnableState_t enabled;
  NVML_CHECK(nvml_stub().nvmlDeviceGetNvLinkState(device, 0, &enabled));
  printf("nvlink enabled: %d\n", enabled);

  printf("name: %s\n", bytedance::flux::get_gpu_device_name(0));

  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  printf("device_count: %d\n", device_count);

  std::vector<int> device_ids(device_count);
  std::iota(device_ids.begin(), device_ids.end(), 0);
  bytedance::flux::topo_utils::initialize_topo(device_ids);

  bool is_initialized = bytedance::flux::topo_utils::is_topo_initialized();
  printf("is_initialized: %d\n", is_initialized);
  bool has_nvlink = bytedance::flux::topo_utils::has_nvlink();
  printf("has_nvlink: %d\n", has_nvlink);
  bool has_nvswitch = bytedance::flux::topo_utils::has_nvswitch();
  printf("has_nvswitch: %d\n", has_nvswitch);
  bool has_heterogeneous_nvlink = bytedance::flux::topo_utils::has_heterogeneous_nvlink();
  printf("has_heterogeneous_nvlink: %d\n", has_heterogeneous_nvlink);
  int topo_nvlink_local_world_size = bytedance::flux::topo_utils::topo_nvlink_local_world_size();
  printf("topo_nvlink_local_world_size: %d\n", topo_nvlink_local_world_size);

  int topo_numa_local_world_size = bytedance::flux::topo_utils::topo_numa_local_world_size();
  printf("topo_numa_local_world_size: %d\n", topo_numa_local_world_size);
  bool has_heterogeneous_pcie = bytedance::flux::topo_utils::has_heterogeneous_pcie();
  printf("has_heterogeneous_pcie: %d\n", has_heterogeneous_pcie);
  return 0;
}
