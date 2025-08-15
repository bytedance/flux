//===- reduce_kernel.cu ----------------------------------------- C++ ---===//
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

#include <stdint.h>
#include <cuda_fp16.h>

__device__ __forceinline__ void
global_red(half2 const &D, void *ptr) {
  uint32_t const &data = reinterpret_cast<uint32_t const &>(D);
  asm volatile(
      "{\n"
      "  red.global.sys.add.noftz.f16x2 [%0], %1;\n"
      "}\n"
      :
      : "l"(ptr), "r"(data));
}

extern "C" __global__ void
reduce_kernel(void *dst, const void *src, size_t nbytes) {
  constexpr int kElemsPerVec = sizeof(half2);
  size_t elems = nbytes / kElemsPerVec;
  half2 *src_vec = (half2 *)src;
  half2 *dst_vec = (half2 *)dst;

#pragma unroll(8)
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < elems; i += gridDim.x * blockDim.x) {
    global_red(src_vec[i], dst_vec + i);
  }
}
