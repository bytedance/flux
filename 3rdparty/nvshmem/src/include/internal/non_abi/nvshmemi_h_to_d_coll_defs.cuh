/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See COPYRIGHT.txt for license information
 */

/*
 * This file strictly forward declares APIs defined in device headers which are called
 * internally by the host library. These API calls are not part of the ABI since they are
 * statically compiled into the host code and unused from the application.
 */

#ifndef _NVSHMEMI_H_TO_D_COLL_DEFS_H_
#define _NVSHMEMI_H_TO_D_COLL_DEFS_H_

#include <cuda_runtime.h>

#include "non_abi/nvshmem_build_options.h"  // IWYU pragma: keep
#ifdef NVSHMEM_ENABLE_ALL_DEVICE_INLINING
#include "non_abi/device/pt-to-pt/transfer_device.cuh"
#else
#include "non_abi/device/pt-to-pt/nvshmemi_transfer_api.cuh"
#endif

#include "non_abi/device/coll/defines.cuh"

/* Collectives start */
__global__ void barrier_on_stream_kernel(int start, int stride, int size, long *pSync,
                                         long *counter);
__global__ void barrier_on_stream_kernel_warp(int start, int stride, int size, long *pSync,
                                              long *counter);
__global__ void barrier_on_stream_kernel_block(int start, int stride, int size, long *pSync,
                                               long *counter);

__global__ void sync_on_stream_kernel(int start, int stride, int size, long *pSync, long *counter);
__global__ void sync_on_stream_kernel_warp(int start, int stride, int size, long *pSync,
                                           long *counter);
__global__ void sync_on_stream_kernel_block(int start, int stride, int size, long *pSync,
                                            long *counter);
__global__ void sync_all_on_stream_kernel();
__global__ void sync_all_on_stream_kernel_warp();
__global__ void sync_all_on_stream_kernel_block();

template <typename TYPE>
__global__ void alltoall_on_stream_kernel(nvshmem_team_t team, TYPE *dest, const TYPE *source,
                                          size_t nelems) {
#ifdef __CUDA_ARCH__
    if (!blockIdx.x)
        nvshmemi_alltoall_threadgroup<TYPE, NVSHMEMI_THREADGROUP_BLOCK>(team, dest, source, nelems);
#endif
}

template <threadgroup_t SCOPE>
__global__ void barrier_on_stream_kernel_threadgroup(nvshmem_team_t team, int in_cuda_graph) {
#ifdef __CUDA_ARCH__
    int myidx = nvshmemi_thread_id_in_threadgroup<SCOPE>();

    if (nvshmemi_device_state_d.job_connectivity >= NVSHMEMI_JOB_GPU_LDST_REMOTE_ATOMICS) {
        nvshmemi_transfer_quiet<SCOPE>(false);
    }
    if (in_cuda_graph) {
        nvshmemi_threadgroup_sync<SCOPE>();
        if (!myidx) __threadfence_system();
        nvshmemi_threadgroup_sync<SCOPE>();
    }

    nvshmemi_sync_algo_threadgroup<SCOPE>(team);

    if (!myidx) {
        if (nvshmemi_device_state_d.job_connectivity > NVSHMEMI_JOB_GPU_PROXY)
            nvshmemi_transfer_enforce_consistency_at_target(false);
    }
#endif
}

template <typename T>
__global__ void broadcast_on_stream_kernel(nvshmem_team_t team, T *dest, const T *source,
                                           size_t nelems, int PE_root) {
#ifdef __CUDA_ARCH__
    if (!blockIdx.x)
        nvshmemi_broadcast_threadgroup<T, NVSHMEMI_THREADGROUP_BLOCK>(team, dest, source, nelems,
                                                                      PE_root);
#endif
}

template <typename TYPE>
__global__ void fcollect_on_stream_kernel(nvshmem_team_t team, TYPE *dest, const TYPE *source,
                                          size_t nelems) {
#ifdef __CUDA_ARCH__
    nvshmem_team_t myteam = nvshmemi_device_state_d.team_pool[team]->team_dups[blockIdx.x];
    // Divide nelems by # of blocks per grid for cases: nelems_per_block == 0 and nelems_remain !=
    // 0, != 0 and nelems_remain != 0, != 0 and nelems_remain == 0
    int nelems_remain = (nelems % gridDim.x);
    int nelems_per_block = (nelems / gridDim.x);
    int my_nelems = nelems_per_block;
    if (blockIdx.x == gridDim.x - 1) {
        my_nelems = nelems_per_block + nelems_remain;
    }

    if (my_nelems > 0)
        nvshmemi_fcollect_threadgroup<TYPE, NVSHMEMI_THREADGROUP_BLOCK>(
            myteam, dest, source + nelems_per_block * blockIdx.x,
            nelems_per_block * blockIdx.x + nvshmemi_team_my_pe(myteam) * nelems, my_nelems);
#endif
}

template <typename TYPE, rdxn_ops_t OP>
__global__ void rdxn_on_stream_kernel(nvshmem_team_t team, TYPE *dest, const TYPE *source,
                                      size_t nreduce) {
#ifdef __CUDA_ARCH__
    if (!blockIdx.x)
        nvshmemi_reduce_threadgroup<TYPE, OP, NVSHMEMI_THREADGROUP_BLOCK>(team, dest, source,
                                                                          nreduce);
#endif
}

template <typename TYPE, rdxn_ops_t OP>
__global__ void reducescatter_on_stream_kernel(nvshmem_team_t team, TYPE *dest, const TYPE *source,
                                               size_t nreduce) {
#ifdef __CUDA_ARCH__
    if (!blockIdx.x)
        nvshmemi_reducescatter_threadgroup<TYPE, OP, NVSHMEMI_THREADGROUP_BLOCK>(team, dest, source,
                                                                                 nreduce);
#endif
}

template <threadgroup_t SCOPE>
__global__ void sync_on_stream_kernel_threadgroup(nvshmem_team_t team) {
#ifdef __CUDA_ARCH__
    nvshmemi_sync_algo_threadgroup<SCOPE>(team);
#endif
}
/* collectives end */
#endif
