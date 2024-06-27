//===- cuda_common.cc --------------------------------------------- C++ ---===//
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
#include "flux/cuda/nvml_stub.h"

namespace bytedance::flux {

void
ensure_nvml_init() {
  static bool inited = []() -> bool {
    NVML_CHECK(nvml_stub().nvmlInit());  // can be initialized many times.
    return true;
  }();
}

// why not std::string? flux/th_op is compiled with -D_GLIBCXX_USE_CXX11_ABI=0 but flux/cuda is not
const char *
get_gpu_device_name(int devid) {
  ensure_nvml_init();
  constexpr int kMaxDevices = 32;
  static std::array<char[NVML_DEVICE_NAME_V2_BUFFER_SIZE], kMaxDevices> kDeviceNames = []() {
    std::array<char[NVML_DEVICE_NAME_V2_BUFFER_SIZE], kMaxDevices> device_names;
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    for (int i = 0; i < count; i++) {
      nvmlDevice_t device;
      NVML_CHECK(nvml_stub().nvmlDeviceGetHandleByIndex(i, &device));
      NVML_CHECK(
          nvml_stub().nvmlDeviceGetName(device, device_names[i], NVML_DEVICE_NAME_V2_BUFFER_SIZE));
    }
    return device_names;
  }();
  return kDeviceNames[devid];
}

unsigned
get_pcie_gen(int devid) {
  nvmlDevice_t device;
  NVML_CHECK(nvml_stub().nvmlDeviceGetHandleByIndex(devid, &device));
  unsigned int gen = 0;
  NVML_CHECK(nvml_stub().nvmlDeviceGetMaxPcieLinkGeneration(device, &gen));
  return gen;
}

int
get_sm_count() {
  static int sms = []() {
    int num_sms = -1;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    return num_sms;
  }();
  return sms;
}
}  // namespace bytedance::flux
