/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef ALLTOALL_DEVICE_CUH
#define ALLTOALL_DEVICE_CUH
#include <cuda_runtime.h>
#include "non_abi/device/wait/nvshmemi_wait_until_apis.cuh"
#include "non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh"
#include "non_abi/device/common/nvshmemi_common_device.cuh"
#include "barrier.cuh"

#ifdef __CUDA_ARCH__

#define NVSHMEMI_ALLTOALL_SMALL_MSGSIZE 16
#define NVSHMEMI_ALLTOALL_MEDIUM_MSGSIZE 16384

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_alltoall_allpush_threadgroup(nvshmem_team_t team, T *dest,
                                                             const T *source, size_t nelems) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int PE_start = teami->start;
    int stride = teami->stride;
    int PE_size = teami->size;
    int next_rank, src_offset, dst_offset;
    const int mype = nvshmemi_device_state_d.mype;
    int my_idx_in_active_set = (mype - PE_start) / stride;
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    uint64_t *psync = (uint64_t *)nvshmemi_team_get_psync(teami, ALLTOALL);
    uint64_t *pwrk = &teami->alltoall_pwrk[teami->alltoall_count % 2];
    const size_t msgsize = nelems * sizeof(T);
    const int first_unused_warp = (PE_size + (warpSize - 1)) / warpSize;
    const int my_warp_idx = myIdx / warpSize;
    const int num_warps = groupSize / warpSize;

    dst_offset = nelems * my_idx_in_active_set;

    /* Do remote ops and local ops < 16 bytes from a single thread */
    /* TODO: Find a more optimal transfer point than 16 bytes */
    for (int i = myIdx; i < PE_size; i += groupSize) {
        next_rank = PE_start + ((my_idx_in_active_set + i) % PE_size) * stride;
        void *peer_base_addr = (void *)__ldg(
            (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + next_rank);
        src_offset = nelems * ((next_rank - PE_start) / stride);
        if (!peer_base_addr) {
            /* We are breaking rank with the rest of the group here so send the RMA with thread
             * scope. */
            nvshmemi_transfer_put_signal<NVSHMEMI_THREADGROUP_THREAD>(
                (void *)(dest + dst_offset), (void *)(source + src_offset), msgsize,
                (void *)(psync + mype), 1ULL, NVSHMEMI_AMO_SIGNAL_ADD, next_rank, true);
        } else if (msgsize <= NVSHMEMI_ALLTOALL_SMALL_MSGSIZE) {
            nvshmemi_put_nbi_threadgroup<T, NVSHMEMI_THREADGROUP_THREAD>(
                dest + dst_offset, source + src_offset, nelems, next_rank);
        }
    }

    if (SCOPE == NVSHMEMI_THREADGROUP_BLOCK && PE_size < groupSize &&
        num_warps > first_unused_warp && msgsize > NVSHMEMI_ALLTOALL_SMALL_MSGSIZE &&
        msgsize <= NVSHMEMI_ALLTOALL_MEDIUM_MSGSIZE) {
        if (my_warp_idx >= first_unused_warp) {
            for (int ii = my_warp_idx - first_unused_warp; ii < PE_size;
                 ii += (num_warps - first_unused_warp)) {
                next_rank = PE_start + ((my_idx_in_active_set + ii) % PE_size) * stride;
                src_offset = nelems * ((next_rank - PE_start) / stride);
                void *peer_base_addr = (void *)__ldg(
                    (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p +
                    next_rank);
                if (peer_base_addr) {
                    nvshmemi_put_nbi_threadgroup<T, NVSHMEMI_THREADGROUP_WARP>(
                        dest + dst_offset, source + src_offset, nelems, next_rank);
                }
            }
        }
    } else if (msgsize > NVSHMEMI_ALLTOALL_SMALL_MSGSIZE) {
        for (int ii = 0; ii < PE_size; ii++) {
            next_rank = PE_start + ((my_idx_in_active_set + ii) % PE_size) * stride;
            src_offset = nelems * ((next_rank - PE_start) / stride);
            void *peer_base_addr = (void *)__ldg(
                (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + next_rank);
            if (peer_base_addr) {
                nvshmemi_put_nbi_threadgroup<T, SCOPE>(dest + dst_offset, source + src_offset,
                                                       nelems, next_rank);
            }
        }
    }

    nvshmemi_threadgroup_sync<SCOPE>();
    /* A fence and signal is required - note that we can skip any size check here because it's
     * inherent in the boolean. */
    if (myIdx == 0) {
        atomicAdd((unsigned long long *)pwrk, 1ULL);
        __threadfence_system();
    }
    nvshmemi_threadgroup_sync<SCOPE>();
    for (int i = myIdx; i < PE_size; i += groupSize) {
        next_rank = PE_start + ((my_idx_in_active_set + i) % PE_size) * stride;
        void *peer_base_addr = (void *)__ldg(
            (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + next_rank);
        if (peer_base_addr) {
            nvshmemi_signal_op((psync + mype), 1ULL, NVSHMEMI_AMO_SIGNAL_ADD, next_rank);
        }
    }

    nvshmemi_threadgroup_sync<SCOPE>();
    for (int i = myIdx; i < PE_size; i += groupSize) {
        next_rank = PE_start + ((my_idx_in_active_set + i) % PE_size) * stride;
        nvshmemi_wait_until_greater_than_equals<uint64_t>((psync + next_rank), *pwrk,
                                                          NVSHMEMI_CALL_SITE_SIGNAL_WAIT_UNTIL_GE);
    }
    if (SCOPE == NVSHMEMI_THREADGROUP_BLOCK && PE_size < groupSize) {
        if (my_warp_idx == first_unused_warp)
            nvshmemi_transfer_quiet<NVSHMEMI_THREADGROUP_WARP>(false);
    } else
        nvshmemi_transfer_quiet<SCOPE>(false);
    nvshmemi_threadgroup_sync<SCOPE>();
    if (myIdx == 0) teami->alltoall_count++;
}

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_alltoall_p2p_allpush_threadgroup(nvshmem_team_t team, T *dest,
                                                                 const T *source, size_t nelems) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int PE_start = teami->start;
    int PE_stride = teami->stride;
    int stride = teami->stride;
    int PE_size = teami->size;
    int next_rank;
    int src_offset;
    int dst_offset;
    const int mype = nvshmemi_device_state_d.mype;
    int my_idx_in_active_set = (mype - PE_start) / PE_stride;
    T *dst_ptr;
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();

    for (int ii = 0; ii < PE_size; ii++) {
        next_rank = PE_start + ((my_idx_in_active_set + ii) % PE_size) * stride;
        src_offset = nelems * ((next_rank - PE_start) / stride);
        dst_offset = nelems * ((mype - PE_start) / stride);
        dst_ptr = (T *)nvshmemi_ptr((void *)(dest + dst_offset), next_rank);
        nvshmemi_memcpy_threadgroup<SCOPE>(dst_ptr, source + src_offset, nelems * sizeof(T));
    }
    nvshmemi_barrier_threadgroup<SCOPE>(team);
}

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_alltoall_threadgroup(nvshmem_team_t team, T *dest, const T *source,
                                                     size_t nelems) {
    if (nvshmemi_device_state_d.job_connectivity <= NVSHMEMI_JOB_GPU_LDST_REMOTE_ATOMICS)
        nvshmemi_alltoall_p2p_allpush_threadgroup<T, SCOPE>(team, dest, source, nelems);
    else
        nvshmemi_alltoall_allpush_threadgroup<T, SCOPE>(team, dest, source, nelems);
}

#endif /* __CUDA_ARCH__ */
#endif /* ALLTOALL_DEVICE_CUH */
