//===- inplace_cast.cu ------------------------------------------- C++ ---===//
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

/*! \file
    \brief Example of in-place cast from float32 to float16.

    This example demonstrates an implementation of in-place cast from float32 to float16.
    The casted results are written back to the original piece of memory in a contiguous way.
    The major challenge is that blocks working on the latter regions may overwrite to the previous
   regions.
    To solve this problem, each block will signal after it finishes reading its own data to
   registers, and blocks overwriting previous blocks must wait for the corresponding signal from
   previous blocks before writing.
    For <num_blocks>, it should have at least one per SM. On L20, two per SM is the best
   (92*2=184). Too many may cause contention on atomic and reduce performance.
    For <block_size>, on L20, best is 256. If too large, it may reduce occupancy because each block
   will take too many registers. Also, this number should be adjusted together with TS (see the
   comments above the definition of TS)
*/

#include "inplace_cast/inplace_cast.hpp"

#define SIGNAL_VAL 2  // Two chunks will decrement the signal of the same chunk

namespace bytedance {
namespace flux {

__device__ void
signal(unsigned *flags, unsigned chunk_id) {
  // Increment flag only after all threads of the block finishes
  __syncthreads();

  volatile unsigned *vflags = (volatile unsigned *)flags;

  if (threadIdx.x == 0) {
    vflags[chunk_id] = SIGNAL_VAL;
  }
}

__device__ void
wait(unsigned *flags, unsigned chunk_id) {
  volatile unsigned *vflags = (volatile unsigned *)flags;

  if (threadIdx.x == 0 && chunk_id != 0) {
    int target_cid = chunk_id / 2;
    while (vflags[target_cid] == 0)
      ;
    atomicDec(flags + target_cid, SIGNAL_VAL);
  }

  // All threads of the block continues only after the flag is in correct value
  __syncthreads();
}

__device__ unsigned
get_chunk_id(size_t data_size, unsigned *chunk_counter) {
  __shared__ unsigned chunk_id;
  if (threadIdx.x == 0) {
    chunk_id = atomicAdd(chunk_counter, 1);
  }
  __syncthreads();
  return chunk_id;
}

template <typename FromT, typename ToT>
__global__ void
inplace_cast_fp32_to_bf16_kernel(
    FromT *ptr, unsigned *flags, size_t data_size, unsigned *chunk_counter) {
  FromT regs[INPLACE_CAST_TS];
  ToT *regs_casted = (ToT *)regs;
  unsigned chunk_id = get_chunk_id(data_size, chunk_counter);

  while (chunk_id < (data_size + blockDim.x * INPLACE_CAST_TS - 1) /
                        (blockDim.x * INPLACE_CAST_TS)) {  // need an extra chunk for the tail
#pragma unroll
    for (size_t i = 0; i < INPLACE_CAST_TS / 4; ++i) {
      unsigned int offset =
          chunk_id * blockDim.x * INPLACE_CAST_TS / 4 + i * blockDim.x + threadIdx.x;
      if (offset < (data_size + 4 - 1) / 4) {  // deal with vectorized load/store (vector_length=4)
        reinterpret_cast<float4 *>(regs)[i] = __ldcs(reinterpret_cast<float4 *>(ptr) + offset);
      }
    }

    signal(flags, chunk_id);

#pragma unroll
    for (size_t i = 0; i < INPLACE_CAST_TS; ++i) {
      // TODO: maybe handle other types, like 64->32?
      regs_casted[i] = __float2bfloat16_rn(regs[i]);
    }

    wait(flags, chunk_id);

#pragma unroll
    for (size_t i = 0; i < INPLACE_CAST_TS / 4; ++i) {
      unsigned int offset =
          chunk_id * blockDim.x * INPLACE_CAST_TS / 4 + i * blockDim.x + threadIdx.x;
      if (offset < (data_size + 4 - 1) / 4) {  // deal with vectorized load/store (vector_length=4)
        __stcg(reinterpret_cast<float2 *>(ptr) + offset, reinterpret_cast<float2 *>(regs)[i]);
      }
    }

    chunk_id = get_chunk_id(data_size, chunk_counter);
  }
}

void
inplace_cast_fp32_to_bf16_impl(
    void *data,
    size_t size,
    unsigned *flags,
    unsigned *counter,
    cudaStream_t stream,
    int num_blocks,
    int block_size) {
  inplace_cast_fp32_to_bf16_kernel<float, __nv_bfloat16>
      <<<num_blocks, block_size, 0, stream>>>(static_cast<float *>(data), flags, size, counter);
}

}  // namespace flux
}  // namespace bytedance
