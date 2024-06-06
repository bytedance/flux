//===- bsr_reduce.cu ---------------------------------------------- C++ ---===//
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

#include "flux/cuda/cuda_common.h"
#include "flux/flux.h"
#include "reduce_scatter/bsr_reduce.hpp"

namespace bytedance {
namespace flux {

template <typename Element>
CUTLASS_HOST_DEVICE float
element_to_float(Element x) {
  if constexpr (std::is_same_v<Element, __half>) {
    return __half2float(x);
  } else if constexpr (std::is_same_v<Element, __nv_bfloat16>) {
    return __bfloat162float(x);
  } else {
    static_assert(cutlass::detail::dependent_false<Element>, "unsupported Element");
  }
}

template <typename Element>
CUTLASS_HOST_DEVICE Element
float_to_element(float x) {
  if constexpr (std::is_same_v<Element, __half>) {
    return __float2half(x);
  } else if constexpr (std::is_same_v<Element, __nv_bfloat16>) {
    return __float2bfloat16(x);
  } else {
    static_assert(cutlass::detail::dependent_false<Element>, "unsupported Element");
  }
}

template <typename Element, const int BLOCK_M, const int BLOCK_N, const int TOTAL_THREADS>
__global__ void
bsr2dense_reduce_kernel(
    Element *input, Element *output, const int M, const int N, const int red_dim_size) {
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  const int tid = threadIdx.x;

  constexpr int ELEMENT_SIZE = sizeof(Element);
  constexpr int N_PER_ITERATION = TOTAL_THREADS * 16 / ELEMENT_SIZE;  // here 16 represents float4
  static_assert((BLOCK_M * BLOCK_N) % N_PER_ITERATION == 0);
  constexpr int UNROLL_ITER_N = BLOCK_M * BLOCK_N / N_PER_ITERATION;
  constexpr int N_REG = 16 / ELEMENT_SIZE;
  assert(blockDim.x == TOTAL_THREADS);
#pragma unroll
  for (int iter = 0; iter < UNROLL_ITER_N; iter++) {
    float acc[N_REG];
    Element reg128[N_REG];
    // perform reduce
#pragma unroll
    for (int i = 0; i < N_REG; i++) {
      acc[i] = 0;
    }
    for (int red_pos = 0; red_pos < red_dim_size; red_pos++) {
      const int block_idx = by * (N / BLOCK_N) + bx;
      int pos =
          red_pos * M * N + block_idx * BLOCK_M * BLOCK_N + iter * N_PER_ITERATION + tid * N_REG;
      FETCH_128bit(&reg128[0]) = FETCH_128bit(&input[pos]);
#pragma unroll
      for (int i = 0; i < N_REG; i++) {
        acc[i] += element_to_float(reg128[i]);
      }
    }
#pragma unroll
    for (int i = 0; i < N_REG; i++) {
      reg128[i] = float_to_element<Element>(acc[i]);
    }
    // write to the output
    const int offset_within_block = iter * N_PER_ITERATION + tid * N_REG;
    const int m_within_block = offset_within_block / BLOCK_N;
    const int n_within_block = offset_within_block % BLOCK_N;
    const int output_pos_m = by * BLOCK_M + m_within_block;
    const int output_pos_n = bx * BLOCK_N + n_within_block;
    const int out_pos = output_pos_m * N + output_pos_n;
    FETCH_128bit(&output[out_pos]) = FETCH_128bit(&reg128[0]);
  }
}

void
bsr2dense_reduce(
    void *input,
    void *output,
    std::vector<int> shape,
    int block_h,
    int block_w,
    DataTypeEnum dtype,
    cudaStream_t stream) {
  // input shape BxMxN
  // output shape MxN
  assert(shape.size() == 3);
  const int M = shape[shape.size() - 2];
  const int N = shape[shape.size() - 1];
  const int reduce_dim_size = shape[0];
  constexpr int TOTAL_THREADS = 256;

  dim3 block_dim(TOTAL_THREADS);
  dim3 grid_dim(N / block_w, M / block_h);

  tuple_return_if(
      tuple_cartesian_product(
          cute::make_tuple(_FP16{}, _BF16{}),
          cute::make_tuple(
              cute::make_tuple(cute::_128{}, cute::_128{}),
              cute::make_tuple(cute::_128{}, cute::_256{}),
              cute::make_tuple(cute::_256{}, cute::_128{}))),
      [&](auto tup) {
        auto [cdtype, cshape] = tup;
        auto [tile_M, tile_N] = cshape;
        return cdtype == dtype and tile_M == block_h and tile_N == block_w;
      },
      [&](auto tup) {
        auto [cdtype, cshape] = tup;
        auto [c_tile_M, c_tile_N] = cshape;
        using Element = decltype(to_cuda_dtype(cdtype));
        constexpr int tile_M = decltype(c_tile_M){};
        constexpr int tile_N = decltype(c_tile_N){};

        bsr2dense_reduce_kernel<Element, tile_M, tile_N, TOTAL_THREADS>
            <<<grid_dim, block_dim, 0, stream>>>(
                static_cast<Element *>(input),
                static_cast<Element *>(output),
                M,
                N,
                reduce_dim_size);
      },
      [&]() {
        FLUX_CHECK(false) << "unsupported for dtype=" << dtype << ",block_h=" << block_h
                          << ",block_w=" << block_w;
      });
}

}  // namespace flux
}  // namespace bytedance
