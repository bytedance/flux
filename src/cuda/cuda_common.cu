//===- cuda_common.cu ------------------------------------------- C++ ---===//
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
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/cuda_common_device.hpp"

namespace bytedance::flux {
void
copy_continous_aligned(
    void *dst,
    const void *src,
    size_t nbytes,
    int threadblock_count,
    int thread_count,
    cudaStream_t stream) {
  dim3 grid(threadblock_count);
  dim3 block(thread_count);
  {  // copy by uint4
    using PackT = uint4;
    constexpr int kPackSize = sizeof(PackT);
    if (intptr_t(dst) % sizeof(PackT) == 0 && intptr_t(src) % sizeof(PackT) == 0 &&
        nbytes % kPackSize == 0) {
      copy_continous_aligned_kernel<PackT><<<grid, block, 0, stream>>>(dst, src, nbytes);
      CUTE_CHECK_ERROR(cudaGetLastError());
      return;
    }
  }
  {  // copy by uint2
    using PackT = uint2;
    constexpr int kPackSize = sizeof(PackT);
    if (intptr_t(dst) % sizeof(PackT) == 0 && intptr_t(src) % sizeof(PackT) == 0 &&
        nbytes % kPackSize == 0) {
      copy_continous_aligned_kernel<PackT><<<grid, block, 0, stream>>>(dst, src, nbytes);
      CUTE_CHECK_ERROR(cudaGetLastError());
      return;
    }
  }
  {  // copy by uint
    using PackT = uint;
    constexpr int kPackSize = sizeof(PackT);
    if (intptr_t(dst) % sizeof(PackT) == 0 && intptr_t(src) % sizeof(PackT) == 0 &&
        nbytes % kPackSize == 0) {
      copy_continous_aligned_kernel<PackT><<<grid, block, 0, stream>>>(dst, src, nbytes);
      CUTE_CHECK_ERROR(cudaGetLastError());
      return;
    }
  }
  {  // copy by int16_t
    using PackT = int16_t;
    constexpr int kPackSize = sizeof(PackT);
    if (intptr_t(dst) % sizeof(PackT) == 0 && intptr_t(src) % sizeof(PackT) == 0 &&
        nbytes % kPackSize == 0) {
      copy_continous_aligned_kernel<PackT><<<grid, block, 0, stream>>>(dst, src, nbytes);
      CUTE_CHECK_ERROR(cudaGetLastError());
      return;
    }
  }
  {  // copy by int8_t
    using PackT = int8_t;
    copy_continous_aligned_kernel<PackT><<<grid, block, 0, stream>>>(dst, src, nbytes);
    CUTE_CHECK_ERROR(cudaGetLastError());
    return;
  }
}

}  // namespace bytedance::flux
