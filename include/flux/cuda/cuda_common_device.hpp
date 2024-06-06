//===- cuda_common_device.hpp ------------------------------------- C++ ---===//
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

#pragma once

#include "cute/util/debug.hpp"
#include "cute/util/print.hpp"
#include <cstring>
#include <cuda/std/atomic>
#include <cuda/std/chrono>

template <typename T = int>
using atomic_ref_sys = cuda::atomic_ref<T, cuda::thread_scope_system>;
template <typename T = int>
using atomic_ref_dev = cuda::atomic_ref<T, cuda::thread_scope_device>;

namespace bytedance::flux {
__device__ __inline__ uint64_t
global_timer() {
  uint64_t mclk;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(mclk));
  return mclk;
}

template <typename... T>
CUTLASS_DEVICE void
dprint_t0(const T &...args) {
#ifdef FLUX_DEBUG
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    printf(args...);
  }
#endif
}

#define DPRINT_T0(fmt, ...)                                                     \
  bytedance::flux::dprint_t0(                                                   \
      "[%d] " fmt,                                                              \
      blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y, \
      ##__VA_ARGS__)

// usage: dprint_cute0(layout, "layout: i = %d", i);
template <typename CuteType, typename... T>
CUTLASS_DEVICE void
dprint_cute0(const CuteType &value, const T &...args) {
#if defined(FLUX_DEBUG)
  if (cute::thread0()) {
    printf(args...);
    printf(" : ");
    cute::print(value);
    printf("\n");
  }
#endif
}

CUTLASS_DEVICE void
atomic_store_sys(int *ptr, int value) {
  atomic_ref_sys<int> ref(*ptr);
  ref.store(value, cuda::memory_order_relaxed);
}

CUTLASS_DEVICE void
atomic_store_release_sys(int *ptr, int value) {
  atomic_ref_sys<int> ref(*ptr);
  ref.store(value, cuda::memory_order_release);
}

CUTLASS_DEVICE int
atomic_load_sys(int *ptr) {
  return atomic_ref_sys<int>(*ptr).load(cuda::memory_order_relaxed);
}

CUTLASS_DEVICE int
atomic_load_acquire_sys(int *ptr) {
  return atomic_ref_sys<int>(*ptr).load(cuda::memory_order_acquire);
}

CUTLASS_DEVICE void
atomic_store_dev(int *ptr, int value) {
  atomic_ref_dev<int> ref(*ptr);
  ref.store(value, cuda::memory_order_relaxed);
}

CUTLASS_DEVICE void
atomic_store_release_dev(int *ptr, int value) {
  atomic_ref_dev<int> ref(*ptr);
  ref.store(value, cuda::memory_order_release);
}

CUTLASS_DEVICE int
atomic_load_dev(int *ptr) {
  return atomic_ref_dev<int>(*ptr).load(cuda::memory_order_relaxed);
}

CUTLASS_DEVICE int
atomic_load_acquire_dev(int *ptr) {
  return atomic_ref_dev<int>(*ptr).load(cuda::memory_order_acquire);
}

CUTLASS_DEVICE int
atomic_add_sys(int *ptr, int value) {
  atomic_ref_sys<int> ref(*ptr);
  return ref.fetch_add(value, cuda::memory_order_relaxed);
}

CUTLASS_DEVICE int
atomic_add_release_sys(int *ptr, int value) {
  atomic_ref_sys<int> ref(*ptr);
  return ref.fetch_add(value, cuda::memory_order_release);
}

CUTLASS_DEVICE int
atomic_add_dev(int *ptr, int value) {
  atomic_ref_dev<int> ref(*ptr);
  return ref.fetch_add(value, cuda::memory_order_relaxed);
}

CUTLASS_DEVICE int
atomic_add_release_dev(int *ptr, int value) {
  atomic_ref_dev<int> ref(*ptr);
  return ref.fetch_add(value, cuda::memory_order_release);
}

CUTLASS_DEVICE void
write_t0(uint64_t *ptr, uint64_t value) {
  if constexpr (1) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
      *ptr = value;
    }
  }
}

CUTLASS_DEVICE void
write_clock_t0(uint64_t *ptr) {
  if constexpr (1) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
      *ptr = global_timer();
    }
  }
}

CUTLASS_DEVICE void
nanosleep(unsigned int nanoseconds) {
  if (nanoseconds == 0)
    return;
  // __nanosleep is an approximately method, which has no guaratee.
  clock_t start = global_timer();
  while (global_timer() - start < nanoseconds) {
  }
}

}  // namespace bytedance::flux
