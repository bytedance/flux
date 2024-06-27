//===- cudaipc_barrier_all.cu ------------------------------------------- C++ ---===//
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

#include "cute/container/tuple.hpp"
#include "flux/cuda/cuda_common.h"
#include "flux/flux.h"
#include "flux/cuda/system_barrier.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/barrier.h"
namespace bytedance {
namespace flux {

struct CudaIpcBarrierAllArgs {
  int32_t *sync_buffers[kMaxWorldSize];
  int rank;
  int world_size;
};

__global__ void
CudaIpcBarrierAllKernel(CudaIpcBarrierAllArgs args) {
  int **sync_buffers = args.sync_buffers;
  int world_size = args.world_size;
  int cur_rank = args.rank;
  if (threadIdx.x < world_size) {
    // set achieved flag for others
    int *sync_buffer_dst = sync_buffers[threadIdx.x] + cur_rank;
    uint32_t const data = 1;
#pragma unroll 1
    while (atomicCAS_system(sync_buffer_dst, 0, 1) != 0) {
    }
    int *wait_ptr = sync_buffers[cur_rank] + threadIdx.x;
#pragma unroll 1
    while (atomicCAS_system(wait_ptr, 1, 0) != 1) {
    }
  }
}

void
cudaipc_barrier_all_on_stream_impl(
    cudaStream_t stream, int32_t **sync_buffer_ptr, int rank, int world_size) {
  dim3 grid_dim(1);
  dim3 block_dim(kMaxWorldSize);
  CudaIpcBarrierAllArgs args;
  args.world_size = world_size;
  args.rank = rank;
  assert(world_size < kMaxWorldSize);
  for (int i = 0; i < world_size; i++) {
    args.sync_buffers[i] = sync_buffer_ptr[i];
  }
  CudaIpcBarrierAllKernel<<<grid_dim, block_dim, 0, stream>>>(args);
}

}  // namespace flux
}  // namespace bytedance
