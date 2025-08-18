//===- cudaipc_barrier_all.cu ------------------------------------------- C++ ---===//
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

#include "flux/cuda/cuda_common_device.hpp"
#include "flux/flux.h"
namespace bytedance {
namespace flux {

struct CudaIpcBarrierAllArgs {
  int32_t *sync_buffers[kMaxWorldSize];
  int rank;
  int world_size;
};

__device__ int
ld_acquire_sys(volatile int *ptr) {
  int state = 0;
  // Acquire pattern using acquire modifier
  asm volatile("ld.global.acquire.sys.b32 %0, [%1];\n" : "=r"(state) : "l"(ptr));
  return state;
}

__device__ int
store_release_sys(volatile int *ptr, int value) {
  int state = 0;
  // Acquire pattern using acquire modifier
  asm volatile("st.release.sys.b32 [%0], %1;\n" : : "l"(ptr), "r"(value));
  return state;
}

__global__ void
CudaIpcBarrierAllKernel(CudaIpcBarrierAllArgs args) {
  int **sync_buffers = args.sync_buffers;
  int world_size = args.world_size;
  int cur_rank = args.rank;
  if (threadIdx.x < world_size) {
    // set achieved flag for others
    int *sync_buffer_dst = sync_buffers[threadIdx.x] + cur_rank;
#pragma unroll 1
    while (atomicCAS_system(sync_buffer_dst, 0, 1) != 0) {
    }
    int *wait_ptr = sync_buffers[cur_rank] + threadIdx.x;
#pragma unroll 1
    while (atomicCAS_system(wait_ptr, 1, 0) != 1) {
    }
  }
}

__global__ void
CudaIpcBarrierAllRingModeKernel(CudaIpcBarrierAllArgs args) {
  // no atomic guarantee, dangerous to call to continuous barrier all
  int **sync_buffers = args.sync_buffers;
  int world_size = args.world_size;
  int cur_rank = args.rank;
  int next_peer = (cur_rank + 1) % world_size;
  int prev_peer = (cur_rank - 1 + world_size) % world_size;
  volatile int *ptr_next_peer = sync_buffers[next_peer];
  volatile int *ptr_cur_rank = sync_buffers[cur_rank];
  if (threadIdx.x != 0)
    return;
  if (cur_rank == 0) {
    // PCIE do not support automic
    store_release_sys(ptr_next_peer, 1);
  } else {
#pragma unroll 1
    while (ld_acquire_sys(ptr_cur_rank) != 1) {
    }
    store_release_sys(ptr_next_peer, 1);
  }
  __threadfence_system();
  // quit in a reversed order
  if (cur_rank != world_size - 1) {
#pragma unroll 1
    while (ld_acquire_sys(ptr_next_peer) != 0) {
    }
  }
  ptr_cur_rank[0] = 0;
}

void
cudaipc_barrier_all_on_stream_impl(
    cudaStream_t stream, int32_t **sync_buffer_ptr, int rank, int world_size, bool ring_mode) {
  dim3 grid_dim(1);
  dim3 block_dim(kMaxWorldSize);
  CudaIpcBarrierAllArgs args;
  args.world_size = world_size;
  args.rank = rank;
  assert(world_size < kMaxWorldSize);
  for (int i = 0; i < world_size; i++) {
    args.sync_buffers[i] = sync_buffer_ptr[i];
  }
  if (!ring_mode) {
    CudaIpcBarrierAllKernel<<<grid_dim, block_dim, 0, stream>>>(args);
  } else {
    CudaIpcBarrierAllRingModeKernel<<<grid_dim, block_dim, 0, stream>>>(args);
  }
}

}  // namespace flux
}  // namespace bytedance
