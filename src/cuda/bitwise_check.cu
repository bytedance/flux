//===- bitwise_check.cu ------------------------------------------- C++ ---===//
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

#include "flux/cuda/helper_kernels.h"
#include "cute/container/tuple.hpp"
#include "flux/cuda/cuda_common.h"
#include "flux/flux.h"

namespace bytedance {
namespace flux {

template <typename Element>
__global__ void
BlockCompareEqualKernel(int *equal, Element *ptr_A, Element *ptr_B, size_t capacity) {
  size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
#pragma unroll
  for (; idx < capacity; idx += gridDim.x * blockDim.x) {
    Element a = ptr_A[idx];
    Element b = ptr_B[idx];

    if (a != b) {
      *equal = 0;

      return;
    }
  }
}

template <typename Element>
bool
BlockCompareEqual(
    Element *ptr_A, Element *ptr_B, size_t capacity, int grid_size = 0, int block_size = 0) {
  int equal_flag = 1;
  int *device_equal_flag = nullptr;

  if (cudaMalloc((void **)&device_equal_flag, sizeof(int)) != cudaSuccess) {
    throw std::runtime_error("Failed to allocate device flag.");
  }

  if (cudaMemcpy(device_equal_flag, &equal_flag, sizeof(int), cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    throw std::runtime_error("Failed to copy equality flag to device.");
  }

  if (!grid_size || !block_size) {
    // if grid_size or block_size are zero, query occupancy using the CUDA Occupancy API
    // cudaError_t result = cudaOccupancyMaxPotentialBlockSize(
    //     &grid_size, &block_size, reinterpret_cast<void const *>(BlockCompareEqual<Element>));

    // if (result != cudaSuccess) {
    //   throw std::runtime_error("Failed to query occupancy.");
    // }

    // Limit block size. This has the effect of increasing the number of items processed by a
    // single thread and reduces the impact of initialization overhead.
    // block_size = (block_size < 128 ? block_size : 128);
    constexpr int THREADS = 256;
    block_size = THREADS;
    grid_size = (capacity + block_size - 1) / block_size;
  }

  dim3 grid(grid_size, 1, 1);
  dim3 block(block_size, 1, 1);

  BlockCompareEqualKernel<Element><<<grid, block>>>(device_equal_flag, ptr_A, ptr_B, capacity);

  if (cudaMemcpy(&equal_flag, device_equal_flag, sizeof(int), cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    cudaFree(device_equal_flag);

    throw std::runtime_error("Failed to copy equality flag from device.");
  }

  cudaFree(device_equal_flag);

  return equal_flag;
}

bool
bitwise_check(DataTypeEnum dtype, void *ptr_A, void *ptr_B, size_t capacity) {
  return tuple_return_if(
      cute::make_tuple(_FP16{}, _BF16{}),
      [dtype](auto c_dtype) { return dtype == c_dtype; },

      [&](auto c_dtype) {
        using Element = decltype(to_cutlass_element(c_dtype));
        return BlockCompareEqual<Element>(
            static_cast<Element *>(ptr_A), static_cast<Element *>(ptr_B), capacity);
      },
      [dtype]() {
        FLUX_CHECK(false) << "unsupported dtype: " << dtype;
        return false;
      });
}

}  // namespace flux
}  // namespace bytedance
