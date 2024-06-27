//===- reduce_utils.cuh ----------------------------------------------- C++ ---===//
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
#include <cuda_runtime.h>

namespace bytedance::flux {

// Input: each thread in a warp call this function
//        with its `id` (lane_id) and its `count` to be summed.
// Output: each thread get a presum of all threads' `count`
//        that have `id` less than or equal to its own `id`
template <class T>
__inline__ __device__ T
warp_prefix_sum(int id, T count) {
  for (int i = 1; i < 32; i <<= 1) {
    T val = __shfl_up_sync(0xffffffff, count, i);
    if (id >= i)
      count += val;
  }
  return count;
}

}  // namespace bytedance::flux
