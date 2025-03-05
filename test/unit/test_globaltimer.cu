//===- test_globaltimer.cu ---------------------------------------- C++ ---===//
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

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include "flux/cuda/cuda_common_device.hpp"
#include "cuda_utils.hpp"
#include <cuda/std/chrono>

__global__ void
dump_nano_sleep_ts(uint64_t *ptr, int duration_ns) {
  if (threadIdx.x == 0) {
    // ptr[blockIdx.x] = bytedance::flux::__nano();
    ptr[blockIdx.x] = cuda::std::chrono::high_resolution_clock::now().time_since_epoch().count();
  }
  __nanosleep(duration_ns);
  ptr += gridDim.x;
  if (threadIdx.x == 0) {
    // ptr[blockIdx.x] = bytedance::flux::__nano();
    ptr[blockIdx.x] = cuda::std::chrono::high_resolution_clock::now().time_since_epoch().count();
  }
}

struct GlobalTimerRunner {
  GlobalTimerRunner(int device_id_, int num_blocks_)
      : stream(device_id_),
        timer(device_id_),
        vector(device_id_, num_blocks_ * 2, 0),
        num_blocks(num_blocks_),
        device_id(device_id_) {}
  CudaStream stream;
  CudaEventTimer timer;
  DeviceVector<uint64_t> vector;
  int num_blocks;
  int device_id;
  void
  Run() {
    ScopedDevice _(device_id);
    timer.Start(stream);
    dump_nano_sleep_ts<<<num_blocks, 1, 0, stream>>>(vector.ptr(), 1);
    CUDA_CHECK(cudaGetLastError());
    timer.Stop();
  }

  float
  Sync() {
    ScopedDevice _(device_id);
    stream.sync();
    return timer.GetEclapsedTime();
  }
};

template <typename T>
T
Max(const std::vector<T> &arr) {
  T value = arr[0];
  for (int i = 1; i < arr.size(); i++) {
    value = max(value, arr[i]);
  }
  return value;
}

template <typename T>
T
Min(const std::vector<T> &arr) {
  T value = arr[0];
  for (int i = 1; i < arr.size(); i++) {
    value = min(value, arr[i]);
  }
  return value;
}

template <typename T>
void
Normlize(std::vector<T> &arr) {
  T value = arr[0];
  for (int i = 0; i < arr.size(); i++) {
    arr[i] -= value;
  }
}

int
main() {
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaDeviceEnablePeerAccess(1, 0));
  CUDA_CHECK(cudaSetDevice(1));
  CUDA_CHECK(cudaDeviceEnablePeerAccess(0, 0));

  int num_blocks = 8;
  constexpr int kNumDevices = 8;
  std::array<GlobalTimerRunner *, kNumDevices> runners;
  std::array<float, kNumDevices> durations;
  std::array<std::vector<uint64_t>, kNumDevices> clock_datas;
  for (int i = 0; i < kNumDevices; i++) {
    runners[i] = new GlobalTimerRunner(i, num_blocks);
  }

  for (int i = 0; i < kNumDevices; i++) {
    auto &t = *runners[i];
    t.Run();
  }
  for (int i = 0; i < kNumDevices; i++) {
    auto &t = *runners[i];
    durations[i] = t.Sync();
  }

  uint64_t start_cpu = std::chrono::system_clock::now().time_since_epoch().count();
  std::cout << "std::clock: " << start_cpu << "\n";
  for (int i = 0; i < kNumDevices; i++) {
    auto &t = *runners[i];
    t.Run();
  }
  for (int i = 0; i < kNumDevices; i++) {
    auto &t = *runners[i];
    durations[i] = t.Sync();
  }
  uint64_t stop_cpu = std::chrono::system_clock::now().time_since_epoch().count();
  std::cout << "cpu duration: " << (stop_cpu - start_cpu) << std::endl;
  for (int i = 0; i < kNumDevices; i++) {
    auto &t = *runners[i];
    clock_datas[i] = t.vector.cpu();
  }
  uint64_t base = clock_datas[0][0];
  std::cout << "cuda::std::clock: " << base << "\n";
  for (const auto &clock_data : clock_datas) {
    for (const auto &item : clock_data) {
      base = std::min(base, item);
    }
  }
  for (auto &clock_data : clock_datas) {
    for (auto &item : clock_data) {
      item -= base;
    }
  }
  // for (int i = 0; i < kNumDevices; i++) {
  //   printf(">>> device %d\n", i);
  //   const auto &clock_data = clock_datas[i];
  //   for (int j = 0; j < num_blocks; j++) {
  //     printf("%ld\t", clock_data[j]);
  //   }
  //   printf("\n");
  //   for (int j = 0; j < num_blocks; j++) {
  //     printf("%ld\t", clock_data[j + num_blocks]);
  //   }
  //   printf("\n");
  // }

  for (int i = 0; i < kNumDevices; i++) {
    printf("%lu, ", clock_datas[i][0]);
  }
  printf("\n");

  return 0;
}
