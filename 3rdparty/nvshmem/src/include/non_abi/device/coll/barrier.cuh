/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */
#ifndef BARRIER_DEVICE_CUH
#define BARRIER_DEVICE_CUH
#include <cuda_runtime.h>
#include "non_abi/device/team/nvshmemi_team_defines.cuh"
#include "non_abi/nvshmem_build_options.h"
#ifdef NVSHMEM_ENABLE_ALL_DEVICE_INLINING
#include "non_abi/device/pt-to-pt/transfer_device.cuh"
#else
#include "non_abi/device/pt-to-pt/nvshmemi_transfer_api.cuh"
#endif
#include "non_abi/device/wait/nvshmemi_wait_until_apis.cuh"
#include "non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh"
#include "non_abi/device/common/nvshmemi_common_device.cuh"
#include "utils.cuh"

#ifdef __CUDA_ARCH__

template <int k, int logk, threadgroup_t SCOPE>
__device__ static inline void sync_dissem_pow2_threadgroup(nvshmem_team_t team) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int start = teami->start;
    int stride = teami->stride;
    int size = teami->size;
    volatile long *pSync = (volatile long *)nvshmemi_team_get_psync(teami, SYNC);
    volatile long *sync_counter = (volatile long *)nvshmemi_team_get_sync_counter(teami);
    int my_idx_in_active_set = (nvshmemi_device_state_d.mype - start) / stride;
    volatile long *sync_arr = NULL;
    int shift;
    int to_nbr_idx, to_nbr;
    int from_nbr_idx, from_nbr;
    int temp = size - 1; /* used to calculate number of phases */
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    sync_arr = (volatile long *)pSync;
    int pow_k = 1;
    int phase_num = 0;
    volatile long *counter = sync_counter;
    while (temp) {
        /* notify neighbors */
        for (int j = myIdx + 1; j <= k - 1; j += groupSize) {
            shift = j << phase_num;
            if (shift >= size) break;

            to_nbr_idx = my_idx_in_active_set + shift;
            if (to_nbr_idx >= size) to_nbr_idx = to_nbr_idx - size;
            to_nbr = start + to_nbr_idx * stride;

            nvshmemi_signal_for_barrier<long>(((long *)sync_arr + nvshmemi_device_state_d.mype),
                                              counter[0], to_nbr);
        }

        /* wait for neighbors notification */
        for (int j = myIdx + 1; j <= k - 1; j += groupSize) {
            shift = j << phase_num;
            if (shift >= size) break;

            from_nbr_idx = my_idx_in_active_set - shift;
            if (from_nbr_idx < 0) from_nbr_idx = size + from_nbr_idx;
            from_nbr = start + from_nbr_idx * stride;

            nvshmemi_wait_until_greater_than_equals<volatile long>(sync_arr + from_nbr, counter[0],
                                                                   NVSHMEMI_CALL_SITE_BARRIER_WARP);
        }
        pow_k <<= logk;
        temp >>= logk;
        phase_num++;
        nvshmemi_threadgroup_sync<SCOPE>();
    }
    if (!myIdx) sync_counter[0] += 1;
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <threadgroup_t SCOPE>
__device__ static inline void sync_dissem_threadgroup_2(int start, int stride, int size,
                                                        volatile long *pSync,
                                                        volatile long *sync_counter) {
    int num_phases = 0;
    int k =
        min(nvshmemi_device_state_d.gpu_coll_env_params_var.barrier_tg_dissem_kval, size); /* radix
                                                for the dissemination algorithm */
    int my_idx_in_active_set = (nvshmemi_device_state_d.mype - start) / stride;
    volatile long *sync_arr = NULL;
    int shift;
    int to_nbr_idx, to_nbr;
    int from_nbr_idx, from_nbr;
    int temp = size - 1; /* used to calculate number of phases */
    while (temp) {
        num_phases++;
        temp /= k;
    }
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    sync_arr = (volatile long *)pSync;
    int pow_k = 1;
    volatile long *counter = sync_counter;
    for (int i = 0; i < num_phases; i++) {
        /* notify neighbors */
        for (int j = myIdx + 1; j <= k - 1; j += groupSize) {
            shift = j * pow_k;
            if (shift >= size) break;
            to_nbr_idx = (my_idx_in_active_set + shift) % size;
            to_nbr = start + to_nbr_idx * stride;
            nvshmemi_signal_for_barrier<long>(((long *)sync_arr + nvshmemi_device_state_d.mype),
                                              counter[0], to_nbr);
        }

        /* wait for neighbors notification */
        for (int j = myIdx + 1; j <= k - 1; j += groupSize) {
            shift = j * pow_k;
            if (shift >= size) break;

            from_nbr_idx = my_idx_in_active_set - shift;
            if (from_nbr_idx < 0) from_nbr_idx = size + from_nbr_idx;
            from_nbr = start + from_nbr_idx * stride;

            nvshmemi_wait_until_greater_than_equals<volatile long>(sync_arr + from_nbr, counter[0],
                                                                   NVSHMEMI_CALL_SITE_BARRIER_WARP);
        }
        pow_k *= k;
        nvshmemi_threadgroup_sync<SCOPE>();
    }
    if (!myIdx) sync_counter[0] += 1;
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <threadgroup_t SCOPE>
__device__ static inline void sync_dissem_threadgroup(nvshmem_team_t team) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int start = teami->start;
    int stride = teami->stride;
    int size = teami->size;
    volatile long *pSync = (volatile long *)nvshmemi_team_get_psync(teami, SYNC);
    volatile long *sync_counter = (volatile long *)nvshmemi_team_get_sync_counter(teami);

    sync_dissem_threadgroup_2<SCOPE>(start, stride, size, pSync, sync_counter);
}

template <threadgroup_t SCOPE>
__device__ static inline void nvshmemi_sync_algo_threadgroup(nvshmem_team_t team) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int size = teami->size;
    int k = min(nvshmemi_device_state_d.gpu_coll_env_params_var.barrier_tg_dissem_kval, size);
    k = max(k, 2);
    switch (k) {
        case 2:
            sync_dissem_pow2_threadgroup<2, 1, SCOPE>(team);
            break;
        case 4:
            sync_dissem_pow2_threadgroup<4, 2, SCOPE>(team);
            break;
        case 8:
            sync_dissem_pow2_threadgroup<8, 3, SCOPE>(team);
            break;
        case 16:
            sync_dissem_pow2_threadgroup<16, 4, SCOPE>(team);
            break;
        case 32:
            sync_dissem_pow2_threadgroup<32, 5, SCOPE>(team);
            break;
        default:
            sync_dissem_threadgroup<SCOPE>(team);
            break;
    }
}

template <threadgroup_t SCOPE>
__device__ inline void nvshmemi_sync_threadgroup(nvshmem_team_t team) {
    nvshmemi_threadgroup_sync<SCOPE>();
    nvshmemi_sync_algo_threadgroup<SCOPE>(team);
}

template <threadgroup_t SCOPE>
__device__ inline void nvshmemi_barrier_threadgroup(nvshmem_team_t team) {
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    nvshmemi_threadgroup_sync<SCOPE>();
    if ((nvshmemi_device_state_d.job_connectivity > NVSHMEMI_JOB_GPU_LDST)) {
        nvshmemi_transfer_quiet<SCOPE>(true);
    } else if (!myIdx) {
        __threadfence_system();
    }
    nvshmemi_threadgroup_sync<SCOPE>();

    nvshmemi_sync_algo_threadgroup<SCOPE>(team);

    if (!myIdx) {
        if (nvshmemi_device_state_d.job_connectivity > NVSHMEMI_JOB_GPU_PROXY)
            nvshmemi_transfer_enforce_consistency_at_target(false);
    }
    nvshmemi_threadgroup_sync<SCOPE>();
}

#endif /* __CUDA_ARCH__ */

#endif /* BARRIER_DEVICE_CUH */
