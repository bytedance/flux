//===- reduce_scatter_barrier_struct.hpp -------------------------- C++ ---===//
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

#include <stdint.h>
#include <type_traits>
#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
#define FLUX_HOST_DEVICE __forceinline__ __device__ __host__
#define FLUX_DEVICE __forceinline__ __device__
#elif defined(__CUDACC_RTC__)
#define FLUX_HOST_DEVICE __forceinline__ __device__
#define FLUX_DEVICE __forceinline__ __device__
#else
#define FLUX_HOST_DEVICE inline
#define FLUX_DEVICE inline
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Waddress-of-packed-member"

namespace bytedance::flux {

struct __attribute__((packed)) PerTileFlags {
  // per tile structs
  int epilogue;
  int padding_epilogue[8];  // don't delete. for m < kM
  int reduce;
  int padding_reduce[8];  // don't delete. for m < kM
  int epilogue_queue;
  int reduce_queue;
  uint64_t epilogue_start;
  uint64_t epilogue_stop;
  uint64_t wait_queue_start;
  uint64_t wait_start;
  uint64_t copy_start;
  uint64_t copy_stop;
  uint64_t arrive_inc_stop;
  // non per-tile structs. maybe per-sm or per-segment. use with caution
  int extra;
};

struct __attribute__((packed)) PerRankFlags {
  int gemm_done;
  int buffer_ready;
  int copy_done;
  int counter;
  int remote_copy_done;
  uint64_t gemm_done_ts;
  uint64_t buffer_ready_ts;
  uint64_t copy_done_ts;
  uint64_t counter_ts;
  uint64_t remote_copy_done_ts;
};

struct __attribute__((packed)) BarrierWorkQeueuFlags {
  int epilogue_done;  // 8 for 8 gpus
  int reduce_done;
};

#define END_OF_STRCTURE \
  void *end_of_structure() { return (void *)((char *)ptr_ + size_ * sizeof(BaseType)); }

#define DEFINE_SOA_MEMBER(member)                                                     \
  using member##_type = decltype(std::declval<BaseType>().member);                    \
  FLUX_HOST_DEVICE member##_type *member##_ptr(int index) {                           \
    return reinterpret_cast<member##_type *>(                                         \
               reinterpret_cast<char *>(ptr_) + size_ * offsetof(BaseType, member)) + \
           index;                                                                     \
  }                                                                                   \
  FLUX_HOST_DEVICE member##_type member(int index) { return *member##_ptr(index); }

#define DEFINE_AOS_MEMBER(member)                                                   \
  using member##_type = decltype(std::declval<BaseType>().member);                  \
  FLUX_HOST_DEVICE member##_type member(int index) { return *member##_ptr(index); } \
  FLUX_HOST_DEVICE member##_type *member##_ptr(int index) { return &(ptr_[index].member); }

//// Per-Tile-Flags ////
#define PerTileFlagsDefines(ADD_MEMBER_FN) \
  ADD_MEMBER_FN(epilogue)                  \
  ADD_MEMBER_FN(reduce)                    \
  ADD_MEMBER_FN(epilogue_queue)            \
  ADD_MEMBER_FN(reduce_queue)              \
  ADD_MEMBER_FN(epilogue_start)            \
  ADD_MEMBER_FN(epilogue_stop)             \
  ADD_MEMBER_FN(wait_queue_start)          \
  ADD_MEMBER_FN(wait_start)                \
  ADD_MEMBER_FN(copy_start)                \
  ADD_MEMBER_FN(copy_stop)                 \
  ADD_MEMBER_FN(arrive_inc_stop)           \
  ADD_MEMBER_FN(extra)                     \
  END_OF_STRCTURE

class PerTileFlagsSoAWrapper {
 private:
  void *ptr_;
  int size_;

 public:
  using BaseType = PerTileFlags;
  FLUX_HOST_DEVICE
  PerTileFlagsSoAWrapper(void *ptr, int size) : ptr_(ptr), size_(size) {}
  PerTileFlagsDefines(DEFINE_SOA_MEMBER)
};

class PerTileFlagsAoSWrapper {
 private:
  PerTileFlags *ptr_;
  int size_;

 public:
  using BaseType = PerTileFlags;
  FLUX_HOST_DEVICE
  PerTileFlagsAoSWrapper(void *ptr, int size) : ptr_((PerTileFlags *)ptr), size_(size) {}

  PerTileFlagsDefines(DEFINE_AOS_MEMBER)
};
#undef PerTileFlagsDefines

using PerTileFlagsWrapper = PerTileFlagsSoAWrapper;

//// Per-Rank-Flags ////

#define PerRankFlagsDefines(ADD_MEMBER_FN) \
  ADD_MEMBER_FN(gemm_done)                 \
  ADD_MEMBER_FN(buffer_ready)              \
  ADD_MEMBER_FN(copy_done)                 \
  ADD_MEMBER_FN(counter)                   \
  ADD_MEMBER_FN(remote_copy_done)          \
  ADD_MEMBER_FN(gemm_done_ts)              \
  ADD_MEMBER_FN(buffer_ready_ts)           \
  ADD_MEMBER_FN(copy_done_ts)              \
  ADD_MEMBER_FN(counter_ts)                \
  ADD_MEMBER_FN(remote_copy_done_ts)       \
  END_OF_STRCTURE

class PerRankFlagsSoAWrapper {
 private:
  void *ptr_;
  int size_;

 public:
  using BaseType = PerRankFlags;
  FLUX_HOST_DEVICE
  PerRankFlagsSoAWrapper(void *ptr, int size) : ptr_(ptr), size_(size) {}
  PerRankFlagsDefines(DEFINE_SOA_MEMBER)
};

class PerRankFlagsAoSWrapper {
 private:
  PerRankFlags *ptr_;
  int size_;

 public:
  using BaseType = PerRankFlags;
  FLUX_HOST_DEVICE
  PerRankFlagsAoSWrapper(void *ptr, int size) : ptr_((PerRankFlags *)ptr), size_(size) {}

  PerRankFlagsDefines(DEFINE_AOS_MEMBER)
};

#undef PerRankFlagsDefines

using PerRankFlagsWrapper = PerRankFlagsAoSWrapper;
//// WorkQueue ///
#define BarrierWorkQeueuFlagsDefines(ADD_MEMBER_FN) \
  ADD_MEMBER_FN(epilogue_done)                      \
  ADD_MEMBER_FN(reduce_done)                        \
  END_OF_STRCTURE

class BarrierWorkQeueuFlagsSoAWrapper {
 private:
  void *ptr_;
  int size_;

 public:
  using BaseType = BarrierWorkQeueuFlags;
  FLUX_HOST_DEVICE
  BarrierWorkQeueuFlagsSoAWrapper(void *ptr, int size) : ptr_(ptr), size_(size) {}
  BarrierWorkQeueuFlagsDefines(DEFINE_SOA_MEMBER)
};

class BarrierWorkQeueuFlagsAoSWrapper {
 private:
  BarrierWorkQeueuFlags *ptr_;
  int size_;

 public:
  using BaseType = BarrierWorkQeueuFlags;
  FLUX_HOST_DEVICE
  BarrierWorkQeueuFlagsAoSWrapper(void *ptr, int size)
      : ptr_((BarrierWorkQeueuFlags *)ptr), size_(size) {}

  BarrierWorkQeueuFlagsDefines(DEFINE_AOS_MEMBER)
};

#undef BarrierWorkQeueuFlagsDefines

using BarrierWorkQeueuFlagsWrapper = BarrierWorkQeueuFlagsAoSWrapper;

#undef DEFINE_AOS_MEMBER
#undef DEFINE_SOA_MEMBER

}  // namespace bytedance::flux

#pragma GCC diagnostic pop
