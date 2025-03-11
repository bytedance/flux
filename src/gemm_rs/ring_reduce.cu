
//===- ring_reduce.cu --------------------------------------------- C++ ---===//
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
#include "flux/flux.h"
#include "gemm_rs/ring_reduce.hpp"

namespace bytedance {
namespace flux {

constexpr int32_t kMaxPackBytes = 128 / 8;

template <typename T, int32_t PACK_SIZE>
struct alignas(sizeof(T) * PACK_SIZE) PackedData {
  __device__
  PackedData() {}

  __device__ void
  zero_() {
    for (int32_t i = 0; i < PACK_SIZE; ++i)
      elem[i] = static_cast<T>(0);
  }

  __device__ void
  add_(const PackedData<T, PACK_SIZE> &other) {
    for (int32_t i = 0; i < PACK_SIZE; ++i)
      elem[i] = elem[i] + other.elem[i];
  }

  T elem[PACK_SIZE];
};

template <typename T, int32_t PACK_SIZE>
bool
isAligned(void *ptr) {
  return reinterpret_cast<std::uintptr_t>(ptr) % sizeof(PackedData<T, PACK_SIZE>) == 0;
}

template <typename PackedDataType, const int32_t BLOCK_SIZE>
__global__ void
packed_ring_reduction(
    PackedDataType *input,
    PackedDataType *output,
    int32_t rank,
    int32_t node_num,
    int32_t world_size,
    int32_t chunk_items) {
  const int32_t tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  const int32_t node_idx = tid / chunk_items;
  const int32_t chunk_idx = tid - node_idx * chunk_items;
  const int32_t output_offset = node_idx * chunk_items + chunk_idx;
  if (tid < node_num * chunk_items) {
    PackedDataType *base_ptr = input + chunk_idx + node_idx * world_size * chunk_items;

    PackedDataType accum;
    accum.zero_();
    const int32_t end = world_size + rank + 1;
    for (int32_t i = rank + 1; i < end; i += 1) {
      int32_t idx = i < world_size ? i : i - world_size;
      PackedDataType elem = *(base_ptr + idx * chunk_items);
      accum.add_(elem);
    }
    output[output_offset] = accum;
  }
}

void
ring_reduce(
    void *input,
    void *output,
    int32_t rank,
    int32_t node_num,
    int32_t world_size,
    int32_t chunk_size,
    DataTypeEnum dtype,
    cudaStream_t stream) {
  constexpr int32_t kThreadPerBlock = 256;
  int32_t num_output = node_num * chunk_size;
  dim3 block_size(kThreadPerBlock);

  tuple_return_if(
      cute::make_tuple(_FP16{}, _BF16{}),
      [&](auto cdtype) { return cdtype == dtype; },
      [&](auto cdtype) {
        using Element = decltype(to_cutlass_element(cdtype));
        constexpr int32_t kMaxPackSize = kMaxPackBytes / sizeof(Element);
        if (isAligned<Element, kMaxPackSize>(input) && isAligned<Element, kMaxPackSize>(output) &&
            chunk_size % kMaxPackSize == 0) {
          dim3 grid_size((num_output / kMaxPackSize + kThreadPerBlock - 1) / kThreadPerBlock);
          packed_ring_reduction<PackedData<Element, kMaxPackSize>, kThreadPerBlock>
              <<<grid_size, block_size, 0, stream>>>(
                  reinterpret_cast<PackedData<Element, kMaxPackSize> *>(input),
                  reinterpret_cast<PackedData<Element, kMaxPackSize> *>(output),
                  rank,
                  node_num,
                  world_size,
                  chunk_size / kMaxPackSize);
        } else {
          dim3 grid_size((num_output + kThreadPerBlock - 1) / kThreadPerBlock);
          packed_ring_reduction<PackedData<Element, 1>, kThreadPerBlock>
              <<<grid_size, block_size, 0, stream>>>(
                  reinterpret_cast<PackedData<Element, 1> *>(input),
                  reinterpret_cast<PackedData<Element, 1> *>(output),
                  rank,
                  node_num,
                  world_size,
                  chunk_size);
        }
      },
      [&]() { FLUX_CHECK(false) << "unsupported for dtype: " << dtype; });
}

}  // namespace flux
}  // namespace bytedance
