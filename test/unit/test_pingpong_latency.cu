//===- test_pingpong_latency.cu ----------------------------------- C++ ---===//
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

#include <cuda_runtime_api.h>
#include "cutlass/barrier.h"
#include "flux/cuda/cuda_common_device.hpp"
#include "flux/cuda/cuda_common.h"
#include "cuda_utils.hpp"
#include "flux/cuda/system_barrier.hpp"
#include <stdio.h>
#include <array>
#include "utils.hpp"

DEFINE_int32(warmup_iters, 5, "");
DEFINE_int32(iters, 10, "");
DEFINE_int32(devid, 0, "run from which device. data flow is always 0->1");
DEFINE_int64(num_flags, 1024, "");
DEFINE_int32(num_threads, 1024, "");
DEFINE_int32(data_at_device, 1, "");
DEFINE_bool(run_copy, false, "");
DEFINE_bool(use_cudaMemcpy, false, "use cudaMemcpy to use PCI-e bandwidth");
DEFINE_bool(bidirectional, true, "");

template <typename BarrierType = cutlass::detail::SystemBarrier>
__global__ void
ping_pong(int index, int64_t n, int *barrier_ptr) {
  // if index == 0
  //  set remote i to 1
  //  wait for remote i+1 to 1
  // else if index == 1
  //  wait current i to 1
  //  set current i+1 to 1
  print_per_block("kernel ping_pong from %d with n=%d, barrier: %p\n", index, n, barrier_ptr);
  if (index == 0) {
    for (int i = 0; i < n; i += 2) {
      BarrierType::arrive_inc(barrier_ptr, threadIdx.x, i, (int)1);
      BarrierType::wait_eq(barrier_ptr, threadIdx.x, i + 1, (int)1);
    }
  } else {
    for (int i = 0; i < n; i += 2) {
      BarrierType::wait_eq(barrier_ptr, threadIdx.x, i, (int)1);
      BarrierType::arrive_inc(barrier_ptr, threadIdx.x, i + 1, (int)1);
    }
  }
}

template <typename T, typename VecType = uint4>
__global__ void
run_copy(const T *src, T *dst, size_t elems) {
  // copy as int4
  VecType *isrc = (VecType *)src;
  VecType *idst = (VecType *)dst;
  elems = elems / sizeof(VecType) * sizeof(T);
  int stride = gridDim.x * blockDim.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
// copy as VecType
#pragma unroll(8)
  for (size_t i = 0; i < elems / stride; i++) {
    idst[idx] = isrc[idx];
    idx += stride;
  }
}

int
main(int argc, char **argv) {
  init_flags(&argc, &argv, true);

  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaDeviceEnablePeerAccess(1, 0));
  CUDA_CHECK(cudaSetDevice(1));
  CUDA_CHECK(cudaDeviceEnablePeerAccess(0, 0));

  struct PerfInstance {
    PerfInstance(int id, int64_t bytes)
        : pingpong_stream(id),
          copy_stream(id),
          timer(id),
          src(id == 0 ? 0 : 1, bytes, 0),
          dst(id == 0 ? 1 : 0, bytes, 0) {}
    CudaStream pingpong_stream;
    CudaStream copy_stream;
    CudaEventTimer timer;
    DeviceVector<char> src;
    DeviceVector<char> dst;
  };

  constexpr int kCopyBytes = 1024 * 1024 * 1024;
  std::array<PerfInstance, 2> instances{PerfInstance(0, kCopyBytes), PerfInstance(1, kCopyBytes)};
  DeviceVector<int> dptr(FLAGS_data_at_device, FLAGS_num_flags, 0);

  auto RunCopyFrom = [&](int id) {
    CUDA_CHECK(cudaSetDevice(id));
    PerfInstance &instance = instances[id];
    if (FLAGS_run_copy) {
      if (FLAGS_use_cudaMemcpy) {
        CUDA_CHECK(cudaMemcpyAsync(
            instance.dst,
            instance.src,
            kCopyBytes,
            cudaMemcpyDeviceToDevice,
            instance.copy_stream));
      } else {
        run_copy<<<8, 1024, 0, instance.copy_stream>>>(
            (char *)instance.dst, (char *)instance.src, kCopyBytes);
        CUDA_CHECK(cudaGetLastError());
      }
    }
  };
  auto RunCopy = [&]() {
    RunCopyFrom(0);
    if (FLAGS_bidirectional)
      RunCopyFrom(1);
  };

  for (int devid : {0, 1}) {
    CUDA_CHECK(cudaSetDevice(devid));
    PerfInstance &instance = instances[devid];
    fprintf(stderr, "run from %d\n", devid);
    ping_pong<<<1, FLAGS_num_threads, 0, instance.pingpong_stream>>>(
        devid, FLAGS_num_flags, (int *)dptr);
    CUDA_CHECK(cudaGetLastError());
  }
  RunCopy();
  for (int devid : {0, 1}) {
    CUDA_CHECK(cudaSetDevice(devid));
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  CUDA_CHECK(cudaMemset(dptr, 0, sizeof(int) * FLAGS_num_flags));

  fprintf(stderr, "warmup done\n");
  // perf start
  std::array<float, 2> duration_ms_total = {0, 0};
  for (int i = 0; i < FLAGS_iters; i++) {
    RunCopy();
    for (int devid : {0, 1}) {
      PerfInstance &instance = instances[devid];
      ScopedDevice _(devid);
      instance.timer.Start(instance.pingpong_stream);
      ping_pong<<<1, FLAGS_num_threads, 0, instance.pingpong_stream>>>(
          devid, FLAGS_num_flags, (int *)dptr);
      CUDA_CHECK(cudaGetLastError());
      instance.timer.Stop();
    }
    for (int devid : {0, 1}) {
      PerfInstance &instance = instances[devid];
      CUDA_CHECK(cudaSetDevice(devid));
      CUDA_CHECK(cudaDeviceSynchronize());
      duration_ms_total[devid] += instance.timer.GetEclapsedTime();
      if (devid == FLAGS_data_at_device) {
        CUDA_CHECK(cudaMemset(dptr, 0, sizeof(int) * FLAGS_num_flags));
      }
    }
  }
  printf(
      "dev 0 avg: %0.2f us/ping-pong\n",
      duration_ms_total[0] * 1000 / FLAGS_iters / FLAGS_num_flags * 2);
  printf(
      "dev 1 avg: %0.2f us/ping-pong\n",
      duration_ms_total[1] * 1000 / FLAGS_iters / FLAGS_num_flags * 2);

  return 0;
}
