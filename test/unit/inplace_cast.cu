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

    To run this example:

      $ ./inplace_cast <data_size> <num_iters> <num_blocks> <block_size>

    The above example command will run the example program with <data_size> number of float32 as
   source input. It will run the in-place cast kernel <num_iters> times and take the average
   execution time. The kernel is run with <num_blocks> and each block has <block_size> number of
   threads.
    For <num_blocks>, it should have at least one per SM. On L20, two per SM is the best
   (92*2=184). Too many may cause contention on atomic and reduce performance.
    For <block_size>, on L20, best is 256. If too large, it may reduce occupancy because each block
   will take too many registers. Also, this number should be adjusted together with TS (see the
   comments above the definition of TS)
*/

#include <cuda_bf16.h>
#include <chrono>
#include <string>
#include "cuda_utils.hpp"

// TS controls how many registers are used to hold data temporarily.
// If too large, each block will use too many registers. This will reduce occupancy and so
// significantly reduce performance.
// If too small, each thread gets too few works to do and may increase scheduling overheads.
// On L20, empirically, it was found that TS=8 and block_size=256 gives the best performance. TS=16
// and block_size=256 will give slightly worse performance although both will have occupancy of
// 100%. If occupancy is not 100%, the performance will be very bad.
// To calculate occupancy, first, figure out the register usage per thread with
// `cuobjdump -res-usage inplace_cast`. Then, input the register per thread and threads per block
// (block size) into the occupancy calculator in nsight-compute. Please see
// https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#occupancy-calculator for
// detailed usage.
#define TS 8
#define DUMMY_COUNT 4294967296  // Size for the dummy memory used for flushing L2 cache
#define SIGNAL_VAL 2            // Two chunks will decrement the signal of the same chunk

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
kernel(FromT *ptr, unsigned *flags, size_t data_size, unsigned *chunk_counter) {
  FromT regs[TS];
  ToT *regs_casted = (ToT *)regs;
  unsigned chunk_id = get_chunk_id(data_size, chunk_counter);

  while (chunk_id < (data_size + blockDim.x * TS - 1) /
                        (blockDim.x * TS)) {  // need an extra chunk for the tail
#pragma unroll
    for (size_t i = 0; i < TS / 4; ++i) {
      unsigned int offset = chunk_id * blockDim.x * TS / 4 + i * blockDim.x + threadIdx.x;
      if (offset < (data_size + 4 - 1) / 4) {  // deal with vectorized load/store (vector_length=4)
        reinterpret_cast<float4 *>(regs)[i] = __ldcs(reinterpret_cast<float4 *>(ptr) + offset);
      }
    }

    signal(flags, chunk_id);

#pragma unroll
    for (size_t i = 0; i < TS; ++i) {
      // TODO: maybe handle other types, like 64->32?
      regs_casted[i] = __float2bfloat16_rn(regs[i]);
    }

    wait(flags, chunk_id);

#pragma unroll
    for (size_t i = 0; i < TS / 4; ++i) {
      unsigned int offset = chunk_id * blockDim.x * TS / 4 + i * blockDim.x + threadIdx.x;
      if (offset < (data_size + 4 - 1) / 4) {  // deal with vectorized load/store (vector_length=4)
        __stcg(reinterpret_cast<float2 *>(ptr) + offset, reinterpret_cast<float2 *>(regs)[i]);
      }
    }

    chunk_id = get_chunk_id(data_size, chunk_counter);
  }
}

__global__ void
fill_data(float *data, size_t data_size) {
  size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  while (id < data_size) {
    data[id] = id;
    id += gridDim.x * blockDim.x;
  }
}

int
main(int argc, char **argv) {
  // In terms of number of floats
  size_t data_size = std::stoul(argv[1]);

  // Take average of `num_iters` times for time measurement. Normally 1 is good enough.
  int num_iters = atoi(argv[2]);

  // Typically should have at least one per SM. On L20, two per SM is the best (92*2=184). Too many
  // may cause contention on atomic and reduce performance.
  int num_blocks = atoi(argv[3]);

  // On L20, best is 256. If too large, it may reduce occupancy because each block will take too
  // many registers.
  int block_size = atoi(argv[4]);

  float *data;
  CUDA_CHECK(cudaMalloc(&data, data_size * sizeof(float)));
  fill_data<<<1, 256>>>(data, data_size);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // dummy is for flushing L2 cache
  float *dummy;
  cudaMalloc(&dummy, DUMMY_COUNT * sizeof(float));

  size_t num_chunks = data_size / (block_size * TS);

  unsigned *flags;
  CUDA_CHECK(cudaMalloc(&flags, num_chunks * sizeof(unsigned)));
  CUDA_CHECK(cudaMemset(flags, 0, num_chunks * sizeof(unsigned)));

  unsigned *chunk_counter;
  CUDA_CHECK(cudaMalloc(&chunk_counter, sizeof(unsigned)));
  CUDA_CHECK(cudaMemset(chunk_counter, 0, sizeof(unsigned)));

  printf("gridDim: %d, blockDim: %d\n", num_blocks, block_size);

  // Correctness check

  kernel<float, __nv_bfloat16><<<num_blocks, block_size>>>(data, flags, data_size, chunk_counter);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemset(chunk_counter, 0, sizeof(unsigned)));
  CUDA_CHECK(cudaDeviceSynchronize());

  float *data_host = (float *)malloc(data_size * sizeof(float));
  CUDA_CHECK(cudaMemcpy(data_host, data, data_size * sizeof(float), cudaMemcpyDeviceToHost));

  __nv_bfloat16 *data_casted_host = (__nv_bfloat16 *)data_host;

#if (CUDART_VERSION >= 12020)
  for (size_t i = 0; i < data_size; ++i) {
    if (data_casted_host[i] != __float2bfloat16_rn(i)) {
      printf(
          "Unmatch Actual: %f, Expected: %f, i: %lu\n",
          __bfloat162float(data_casted_host[i]),
          __bfloat162float(__float2bfloat16_rn(i)),
          i);
    }
  }
  printf("Correctness check completes\n");
#endif

  // Warm up

  for (int i = 0; i < 5; ++i) {
    kernel<float, __nv_bfloat16>
        <<<num_blocks, block_size>>>(data, flags, data_size, chunk_counter);
    CUDA_CHECK(cudaMemset(chunk_counter, 0, sizeof(unsigned)));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // Flush L2 cache by writing something into dummy
  fill_data<<<1, 256>>>(dummy, DUMMY_COUNT);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Latency measurement

  bytedance::flux::GpuTimer timer;
  timer.start();

  for (int i = 0; i < num_iters; ++i) {
    kernel<float, __nv_bfloat16>
        <<<num_blocks, block_size>>>(data, flags, data_size, chunk_counter);
    CUDA_CHECK(cudaMemset(chunk_counter, 0, sizeof(unsigned)));
  }

  timer.stop();
  CUDA_CHECK(cudaDeviceSynchronize());

  float latency_ms = timer.elapsed_millis() / num_iters;

  printf("Avg latency: %f ms\n", latency_ms);
  printf(
      "Avg throughput: %lf GB/s\n",
      (double)data_size * sizeof(float) * 1.5 / 1024 / 1024 / 1024 / latency_ms * 1000);
}
