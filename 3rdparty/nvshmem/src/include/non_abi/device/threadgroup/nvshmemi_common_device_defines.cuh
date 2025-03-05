/*
 * Copyright (c) 2016-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */
#ifndef _NVSHMEM_COMMON_DEVICE_DEFINES_CUH_
#define _NVSHMEM_COMMON_DEVICE_DEFINES_CUH_
#include <cuda_runtime.h>
#include "device_host/nvshmem_common.cuh"

#if defined(__CUDACC_RDC__)
#define EXTERN_CONSTANT extern __constant__
#else
#define EXTERN_CONSTANT static __constant__
#endif
EXTERN_CONSTANT nvshmemi_device_host_state_t nvshmemi_device_state_d;
#undef EXTERN_CONSTANT

typedef enum {
    nvshmemi_threadgroup_thread = 0,
    NVSHMEMI_THREADGROUP_THREAD = 0,
    nvshmemi_threadgroup_warp = 1,
    NVSHMEMI_THREADGROUP_WARP = 1,
    nvshmemi_threadgroup_block = 2,
    NVSHMEMI_THREADGROUP_BLOCK = 2
} threadgroup_t;

#ifdef __CUDA_ARCH__

template <threadgroup_t scope>
__device__ __forceinline__ int nvshmemi_thread_id_in_threadgroup() {
    switch (scope) {
        case NVSHMEMI_THREADGROUP_THREAD:
            return 0;
        case NVSHMEMI_THREADGROUP_WARP:
            int myIdx;
            asm volatile("mov.u32  %0, %laneid;" : "=r"(myIdx));
            return myIdx;
        case NVSHMEMI_THREADGROUP_BLOCK:
            return (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y);
        default:
            printf("unrecognized threadscope passed\n");
            assert(0);
            return -1;
    }
}

template <threadgroup_t scope>
__device__ inline int nvshmemi_threadgroup_size() {
    switch (scope) {
        case NVSHMEMI_THREADGROUP_THREAD:
            return 1;
        case NVSHMEMI_THREADGROUP_WARP:
            return ((blockDim.x * blockDim.y * blockDim.z) < warpSize)
                       ? (blockDim.x * blockDim.y * blockDim.z)
                       : warpSize;
        case NVSHMEMI_THREADGROUP_BLOCK:
            return (blockDim.x * blockDim.y * blockDim.z);
        default:
            printf("unrecognized threadscope passed\n");
            assert(0);
            return -1;
    }
}

template <threadgroup_t scope>
__device__ inline void nvshmemi_threadgroup_sync() {
    switch (scope) {
        case NVSHMEMI_THREADGROUP_THREAD:
            return;
        case NVSHMEMI_THREADGROUP_WARP:
            __syncwarp();
            break;
        case NVSHMEMI_THREADGROUP_BLOCK:
            __syncthreads();
            break;
        default:
            printf("unrecognized threadscope passed\n");
            assert(0);
            break;
    }
}
#endif

#endif
