//===- copy_kernel.cu ------------------------------------------- C++ ---===//
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

extern "C" __global__ void
copy_kernel(void *__restrict__ to_ptr, void *__restrict__ from_ptr, size_t nbytes) {
  size_t elems = nbytes / sizeof(uint4);
  uint4 *dst_ptr = (uint4 *)to_ptr;
  uint4 *src_ptr = (uint4 *)from_ptr;
#pragma unroll(4)
  for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < elems;
       tid += blockDim.x * gridDim.x) {
    dst_ptr[tid] = src_ptr[tid];
  }
}
