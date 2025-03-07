//===- test_cuda_common.cu ---------------------------------------- C++ ---===//
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
#include "cutlass/util/device_memory.h"
#include "flux/flux.h"
#include "flux/cuda/cuda_common.h"

namespace bytedance::flux {

void
test_copy_continous_aligned(int elems, bool alignment = true) {
  std::vector<int32_t> src(elems), dst(elems);
  int32_t *src_ptr = src.data(), *dst_ptr = dst.data();

  constexpr int kAlignment = sizeof(uint4);

  cutlass::DeviceAllocation<int32_t> src_d(elems + (alignment ? 0 : 1)), dst_d(elems);
  int32_t *src_dptr = src_d.get(), *dst_dptr = dst_d.get();
  // suppose allocation is aligned
  FLUX_CHECK_DIV(intptr_t(src_dptr), kAlignment);
  FLUX_CHECK_DIV(intptr_t(dst_dptr), kAlignment);
  if (!alignment) {
    src_dptr += 1;  // force not aligned
  }

  cudaStream_t stream = nullptr;
  for (int i = 0; i < elems; i++) {
    src_ptr[i] = i + 1;
  }
  CUDA_CHECK(
      cudaMemcpyAsync(src_dptr, src_ptr, elems * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

  copy_continous_aligned(dst_dptr, src_dptr, elems * sizeof(int32_t), 1, 512, stream);

  CUDA_CHECK(
      cudaMemcpyAsync(dst_ptr, dst_dptr, elems * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));

  for (int i = 0; i < elems; i++) {
    FLUX_CHECK_EQ(dst_ptr[i], src_ptr[i]) << " at index " << i;
  }
  std::cerr << "check passed\n";
}
}  // namespace bytedance::flux

int
main() {
  cudaFree(0);
  bytedance::flux::test_copy_continous_aligned(128, true);      // copy by uint4
  bytedance::flux::test_copy_continous_aligned(128 + 2, true);  // copy by uint2
  bytedance::flux::test_copy_continous_aligned(128 + 1, true);  // copy by int
  bytedance::flux::test_copy_continous_aligned(1024, false);
  bytedance::flux::test_copy_continous_aligned(1024 + 2, false);
  return 0;
}
