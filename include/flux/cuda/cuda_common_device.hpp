//===- cuda_common_device.hpp ------------------------------------- C++ ---===//
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

#pragma once

#include "cute/util/debug.hpp"
#include "cute/util/print.hpp"
#include "cutlass/detail/helper_macros.hpp"
#include <cstddef>
#include <cstring>

#include <cuda/atomic>
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
print_if(bool condition, T &&...args) {
#ifdef FLUX_DEBUG
  if (condition)
    printf(args...);
#endif
}

CUTLASS_DEVICE
int
blockid() {
  return blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
}

CUTLASS_DEVICE
int
threadid() {
  return threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
}

template <typename... T>
CUTLASS_DEVICE void
print_per_block_(T &&...args) {
  print_if(threadid() == 0, args...);
}

#define print_per_block(fmt, ...)                                               \
  bytedance::flux::print_per_block_(                                            \
      "L%d@[%d] " fmt,                                                          \
      __LINE__,                                                                 \
      blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y, \
      ##__VA_ARGS__)

template <typename... T>
CUTLASS_DEVICE void
print_per_kernel(const T &...args) {
  print_if(threadid() == 0 && blockid() == 0, args...);
}

// usage: print_cute_per_kernel(layout, "layout: i = %d", i);
template <typename CuteType, typename... T>
CUTLASS_DEVICE void
print_cute_per_kernel(const CuteType &value, const T &...args) {
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

CUTLASS_DEVICE int
ld_acquire_device(int *ptr) {
  int state = 0;
  // Acquire pattern using acquire modifier
  asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(ptr));
  return state;
}

CUTLASS_DEVICE int
ld_acquire_sys(int *ptr) {
  int state = 0;
  // Acquire pattern using acquire modifier
  asm volatile("ld.global.acquire.sys.b32 %0, [%1];\n" : "=r"(state) : "l"(ptr));
  return state;
}

CUTLASS_DEVICE void
st_release_device(int *ptr, int value) {
  asm volatile("st.release.gpu.b32 [%0], %1;\n" : : "l"(ptr), "r"(value));
}

CUTLASS_DEVICE void
st_release_system(int *ptr, int value) {
  asm volatile("st.release.sys.b32 [%0], %1;\n" : : "l"(ptr), "r"(value));
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

CUTLASS_DEVICE
void
sub_barrier_sync(uint32_t barrier_id, uint32_t num_threads) {
  asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "r"(num_threads));
}

/**
 * @brief make sure aligned from outside
 *  dst % sizeof(PackType) == 0
 *  and src % sizeof(PackType) == 0
 *  and nbytes % sizeof(PackType) == 0
 */
template <typename PackType>
CUTLASS_GLOBAL void
copy_continous_aligned_kernel(
    void *__restrict__ dst, const void *__restrict__ src, size_t nbytes) {
  size_t npacks = nbytes / sizeof(PackType);
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const auto step = blockDim.x * gridDim.x;
  PackType *src_packed = (PackType *)__builtin_assume_aligned(src, sizeof(PackType));
  PackType *dst_packed = (PackType *)__builtin_assume_aligned(dst, sizeof(PackType));
  src_packed += index, dst_packed += index;
  for (int i = index; i < npacks; i += step, src_packed += step, dst_packed += step) {
    *dst_packed = *src_packed;
  }
}

/**
 * @brief a CUDA version std::upper_bound like this:
 * https://cplusplus.com/reference/algorithm/lower_bound/
 *
 * @tparam T
 * @param first
 * @param last
 * @param value
 * @return __device__ const*
 */
template <class T>
__inline__ __device__ const T *
lower_bound_kernel(const T *first, const T *last, const T &value) {
  const T *it;
  size_t count, step;
  count = std::distance(first, last);

  while (count > 0) {
    it = first;
    step = count / 2;
    it += step;

    if (*it < value) {
      first = ++it;
      count -= step + 1;
    } else
      count = step;
  }

  return first;
}

/**
 * @brief a CUDA version std::upper_bound like this:
 * https://cplusplus.com/reference/algorithm/upper_bound/
 *
 * @tparam T
 * @param first
 * @param last
 * @param value
 * @return __device__ const*
 */
template <class T>
__inline__ __device__ const T *
upper_bound_kernel(const T *first, const T *last, const T &value) {
  const T *it;
  size_t count, step;
  count = std::distance(first, last);

  while (count > 0) {
    it = first;
    step = count / 2;
    it += step;

    if (*it <= value) {
      first = ++it;
      count -= step + 1;
    } else
      count = step;
  }

  return first;
}

}  // namespace bytedance::flux
