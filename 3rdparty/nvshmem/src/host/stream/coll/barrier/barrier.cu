/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include "internal/host/util.h"
#include "internal/host/debug.h"
#include "internal/non_abi/nvshmemi_h_to_d_coll_defs.cuh"

int nvshmemi_call_barrier_on_stream_kernel(nvshmem_team_t team, cudaStream_t stream) {
    int num_blocks = 1;
    int num_threads_per_block;
    int in_cuda_graph = 0;

    if (nvshmemi_job_connectivity <= NVSHMEMI_JOB_GPU_LDST_REMOTE_ATOMICS) {
        int size = nvshmemi_team_pool[team]->size;
        num_threads_per_block = size - 1;  // Have enough threads for alltoall algo
    } else {
        num_threads_per_block = nvshmemi_options.BARRIER_TG_DISSEM_KVAL;
    }

    cudaStreamCaptureStatus status;
    CUDA_RUNTIME_CHECK(cudaStreamIsCapturing(stream, &status));
    if (status == cudaStreamCaptureStatusActive) in_cuda_graph = 1;

    if (num_threads_per_block <= 32) {
        barrier_on_stream_kernel_threadgroup<NVSHMEMI_THREADGROUP_WARP>
            <<<num_blocks, 32, 0, stream>>>(team, in_cuda_graph);
    } else {
        barrier_on_stream_kernel_threadgroup<NVSHMEMI_THREADGROUP_BLOCK>
            <<<num_blocks, num_threads_per_block, 0, stream>>>(team, in_cuda_graph);
    }
    CUDA_RUNTIME_CHECK(cudaGetLastError());
    return 0;
}

int nvshmemi_call_sync_on_stream_kernel(nvshmem_team_t team, cudaStream_t stream) {
    int num_blocks = 1;
    int num_threads_per_block;
    if (nvshmemi_job_connectivity <= NVSHMEMI_JOB_GPU_LDST_REMOTE_ATOMICS) {
        int size = nvshmemi_team_pool[team]->size;
        num_threads_per_block = size - 1;  // Have enough threads for alltoall algo
    } else {
        num_threads_per_block = nvshmemi_options.BARRIER_TG_DISSEM_KVAL;
    }

    if (num_threads_per_block <= 32) {
        sync_on_stream_kernel_threadgroup<NVSHMEMI_THREADGROUP_WARP>
            <<<num_blocks, 32, 0, stream>>>(team);
    } else {
        sync_on_stream_kernel_threadgroup<NVSHMEMI_THREADGROUP_BLOCK>
            <<<num_blocks, num_threads_per_block, 0, stream>>>(team);
    }
    CUDA_RUNTIME_CHECK(cudaGetLastError());
    return 0;
}
