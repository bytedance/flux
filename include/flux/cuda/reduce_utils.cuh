//===- reduce_utils.cuh ----------------------------------------------- C++ ---===//
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
#include <cuda_runtime.h>

namespace bytedance::flux {

constexpr int kWarpSize = 32;

// Input: each thread in a warp call this function
//        with its `id` (lane_id) and its `count` to be summed.
// Output: each thread get a presum of all threads' `count`
//        that have `id` less than or equal to its own `id`
template <class T>
__inline__ __device__ T
warp_prefix_sum(int id, T count) {
  for (int i = 1; i < kWarpSize; i <<= 1) {
    T val = __shfl_up_sync(0xffffffff, count, i);
    if (id >= i)
      count += val;
  }
  return count;
}

template <class T>
__inline__ __device__ void
aligned_block_prefix_sum_and_sync(const T *data_in, T *data_out, int count, int align) {
  int warp_idx = threadIdx.x / kWarpSize;
  int lane_idx = threadIdx.x % kWarpSize;
  if (warp_idx == 0) {
    int cur_offset = 0;
    int count_pad = (count + kWarpSize - 1) / kWarpSize * kWarpSize;
    for (int i = lane_idx; i < count_pad; i += kWarpSize) {
      int len = i < count ? data_in[i] : 0;
      len = (len + align - 1) / align * align;
      int temp_offset = warp_prefix_sum(threadIdx.x, len);
      if (i < count) {
        data_out[i] = cur_offset + temp_offset;
      }
      cur_offset += __shfl_sync(0xffffffff, temp_offset, kWarpSize - 1);
    }
  }
  __syncthreads();
}

template <class T>
__inline__ __device__ void
block_prefix_sum_and_sync(const T *data_in, T *data_out, int count) {
  aligned_block_prefix_sum_and_sync(data_in, data_out, count, 1);
}

}  // namespace bytedance::flux
