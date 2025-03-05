/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */
#ifndef FCOLLECT_DEVICE_CUH
#define FCOLLECT_DEVICE_CUH

#if not defined __CUDACC_RTC__
#include <stdint.h>
#include <limits.h>
#else
#include "cuda/std/cstdint"
#include <cuda/std/climits>
#endif

#include <cuda_runtime.h>
#include "device_host/nvshmem_common.cuh"
#include "non_abi/nvshmem_build_options.h"
#ifdef NVSHMEM_ENABLE_ALL_DEVICE_INLINING
#include "non_abi/device/pt-to-pt/transfer_device.cuh"
#else
#include "non_abi/device/pt-to-pt/nvshmemi_transfer_api.cuh"
#endif
#include "non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh"
#include "non_abi/device/common/nvshmemi_common_device.cuh"

#define _FCOLLECT_LL8_PSYNC_SCALE_FACTOR 2

#define _FCOLLECT_MAX(x, y) ((x) > (y) ? (x) : (y))
#define _FCOLLECT_MIN(x, y) ((x) < (y) ? (x) : (y))

typedef enum { LL8 = 0, LL128 } ll_version_t;

#ifdef __CUDA_ARCH__

template <typename T, threadgroup_t SCOPE>
__device__ __forceinline__ void nvshmemi_fcollect_nvls_ll_threadgroup(nvshmem_team_t team, T *dest,
                                                                      const T *source,
                                                                      size_t nelems) {
#if __CUDA_ARCH__ >= 900 && CUDART_VERSION >= 12010
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    const size_t fcollect_ll_threshold =
        nvshmemi_device_state_d.gpu_coll_env_params_var.fcollect_ll_threshold;
    const size_t fcollect_count = teami->fcollect_count;
    const uint32_t ll_flag = teami->fcollect_count;
    char *pWrk = (char *)nvshmemi_team_get_psync(teami, FCOLLECT) +
                 (2 * teami->size * fcollect_ll_threshold *
                  (fcollect_count % 2)); /* same for NVLS in terms of size */
    const size_t pack_offset = nvshmemi_team_my_pe(team) * nelems *
                               (sizeof(T) / sizeof(uint32_t)); /* offset in pSync space */
    /* Find the multicast ptr for pWrk + pack_offset and do a store to remote pSync */
    void *mcast_pWrk = nvshmemi_mc_ptr(teami, (void *)((uint64_t *)pWrk + pack_offset));
    nvshmemi_mcast_packLL<T, SCOPE>((uint64_t *)mcast_pWrk, source, nelems, ll_flag);
    for (int ii = 0; ii < teami->size; ii += 1) {
        size_t prev_offset = nelems * ii * (sizeof(T) / sizeof(uint32_t));
        nvshmemi_mcast_recvLL<T, SCOPE>(dest + (ii * nelems), (uint64_t *)pWrk + prev_offset,
                                        nelems, ll_flag);
    }

    nvshmemi_threadgroup_sync<SCOPE>();
#else
    assert(0 && "NVLink SHARP is not supported on this platform");
#endif
}

template <typename T, threadgroup_t SCOPE, ll_version_t LL_VERSION, bool NODE_SAFE>
__device__ __forceinline__ void nvshmemi_fcollect_allpush_ll_threadgroup(nvshmem_team_t team,
                                                                         T *dest, const T *source,
                                                                         size_t nelems) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    nvshmemi_team_t *teami_node = NULL;
    const size_t fcollect_count = teami->fcollect_count;
    const uint32_t ll_flag = teami->fcollect_count;
    const int my_pe_in_team = nvshmemi_team_my_pe(team);
    const int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    const int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    const int myWarpIdx = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
    size_t data_element_offset;
    size_t psync_element_offset;
    size_t psync_remote_write_elements;
    T *peer_addr;

    T *pWrk;
    size_t pack_offset;
    size_t max_data_elems_per_warp;
    size_t max_psync_elems_per_warp;
    int next_pe, start_pe;
    int num_pes_per_group, remaining_pes;
    int num_warp_groups, num_warps_per_group;
    int warp_id, warp_count, warp_group_id;

    if (NODE_SAFE) {
        /*
         * Really we mean team shared here, but that is not a team that exists today.
         * TODO: Replace team_node with team_shared.
         */
        if (teami->are_gpus_p2p_connected) {
            teami_node = teami;
        } else if (teami->team_node != NVSHMEM_TEAM_INVALID &&
                   nvshmemi_device_state_d.team_pool[NVSHMEM_TEAM_SHARED]->is_team_node) {
            teami_node = nvshmemi_device_state_d.team_pool[teami->team_node];
        }
    }

    warp_count = nvshmemi_threadgroup_size<SCOPE>() / 32;
    warp_id = myIdx / 32;

    if (LL_VERSION == LL8) {
        const size_t fcollect_ll_threshold =
            nvshmemi_device_state_d.gpu_coll_env_params_var.fcollect_ll_threshold / sizeof(T);
        pWrk = (T *)nvshmemi_team_get_psync(teami, FCOLLECT) +
               (_FCOLLECT_LL8_PSYNC_SCALE_FACTOR * teami->size * fcollect_ll_threshold *
                (fcollect_count % 2));
        /* round up to 16 bytes*/
        psync_remote_write_elements =
            NVSHMEMI_TEAM_ROUND_UP((nelems * _FCOLLECT_LL8_PSYNC_SCALE_FACTOR), 16 / sizeof(T));
        pack_offset = my_pe_in_team * psync_remote_write_elements;
        if (!NODE_SAFE || !teami->is_team_node) {
            nvshmemi_packLL<T, SCOPE, 1>((T *)(pWrk + pack_offset), source, nelems, ll_flag, teami,
                                         1, my_pe_in_team);
        }
    } else {
        const size_t fcollect_ll_threshold =
            nvshmemi_device_state_d.gpu_coll_env_params_var.fcollect_ll128_threshold / sizeof(T);
        pWrk = (T *)nvshmemi_team_get_psync(teami, FCOLLECT_128) +
               (teami->size * NVSHMEMI_FCOLLECT_LL128_CALC_PSYNC_SIZE(fcollect_ll_threshold, T) *
                (fcollect_count % 2));
        psync_remote_write_elements = NVSHMEMI_FCOLLECT_LL128_CALC_PSYNC_SIZE(nelems, T);
        pack_offset = my_pe_in_team * psync_remote_write_elements;
        if (!NODE_SAFE || !teami->is_team_node) {
            nvshmemi_packLL128<T, SCOPE, 1>(pWrk + pack_offset, source, nelems, ll_flag, teami, 1,
                                            my_pe_in_team, 0);
        }
    }

    /* send out non blocking puts for all remote PEs */
    if (teami_node != teami) {
        for (uint32_t ii = myIdx + 1; ii < teami->size; ii += groupSize) {
            next_pe = teami->start + ((my_pe_in_team + ii) % teami->size) * teami->stride;
            if (NODE_SAFE) {
                if (nvshmemi_ptr(pWrk, next_pe) == NULL) {
                    nvshmemi_put_nbi_threadgroup<T, NVSHMEMI_THREADGROUP_THREAD>(
                        pWrk + pack_offset, pWrk + pack_offset, psync_remote_write_elements,
                        next_pe);
                }
            } else {
                nvshmemi_put_nbi_threadgroup<T, NVSHMEMI_THREADGROUP_THREAD>(
                    pWrk + pack_offset, pWrk + pack_offset, psync_remote_write_elements, next_pe);
            }
        }
        nvshmemi_threadgroup_sync<NVSHMEMI_THREADGROUP_WARP>();
    }

    if (LL_VERSION == LL8) {
        max_data_elems_per_warp = _LL_8_DATA_BYTES_PER_WARP * _LL_MAX_UNROLL / sizeof(T);
        max_psync_elems_per_warp = _LL_8_PSYNC_BYTES_PER_WARP * _LL_MAX_UNROLL / sizeof(T);
        if (nelems < max_data_elems_per_warp) {
            if (teami_node) {
                for (uint32_t ii = warp_id; ii < teami_node->size; ii += warp_count) {
                    next_pe = teami_node->start +
                              ((my_pe_in_team + ii) % teami_node->size) * teami_node->stride;
                    peer_addr = (T *)nvshmemi_ptr(pWrk, next_pe) + pack_offset;
                    nvshmemi_packLL_naive<T, NVSHMEMI_THREADGROUP_WARP>((uint64_t *)peer_addr,
                                                                        source, nelems, ll_flag);
                }
            }

            for (uint32_t ii = warp_id; ii < teami->size; ii += warp_count) {
                next_pe = ii;
                pack_offset =
                    next_pe * NVSHMEMI_TEAM_ROUND_UP((nelems * _FCOLLECT_LL8_PSYNC_SCALE_FACTOR),
                                                     16 / sizeof(T));
                nvshmemi_recvLL<T, NVSHMEMI_THREADGROUP_WARP>(
                    dest + (next_pe * nelems), (uint64_t *)(pWrk + pack_offset), nelems, ll_flag);
            }
            goto out;
        }
    } else {
        max_data_elems_per_warp = _LL_128_DATA_BYTES_PER_WARP * _LL_MAX_UNROLL / sizeof(T);
        max_psync_elems_per_warp = _LL_128_PSYNC_BYTES_PER_WARP * _LL_MAX_UNROLL / sizeof(T);
    }

    num_warp_groups =
        _FCOLLECT_MAX(1, warp_count / _FCOLLECT_MAX(1, nelems / max_data_elems_per_warp));
    num_warps_per_group = warp_count / num_warp_groups;
    warp_group_id = warp_id / num_warps_per_group;

    if (teami_node != NULL) {
        /* first n ggroups take on an extra PE in the case of remainder */
        num_pes_per_group = teami_node->size / num_warp_groups;
        remaining_pes = teami_node->size % num_warp_groups;
        num_pes_per_group += warp_group_id < remaining_pes ? 1 : 0;

        start_pe = num_pes_per_group * warp_group_id;
        start_pe += warp_group_id >= remaining_pes ? remaining_pes : 0;

        data_element_offset = warp_id % num_warps_per_group * max_data_elems_per_warp;
        psync_element_offset = warp_id % num_warps_per_group * max_psync_elems_per_warp;
        /* All warps except final one per-pe should be full. */
        for (; data_element_offset + max_data_elems_per_warp < nelems;
             data_element_offset += num_warps_per_group * max_data_elems_per_warp,
             psync_element_offset += num_warps_per_group * max_psync_elems_per_warp) {
            if (LL_VERSION == LL8) {
                nvshmemi_packLL<T, NVSHMEMI_THREADGROUP_WARP, _LL_MAX_UNROLL>(
                    pWrk + pack_offset + psync_element_offset, source + data_element_offset,
                    max_data_elems_per_warp, ll_flag, teami_node, num_pes_per_group, start_pe);
            } else {
                nvshmemi_packLL128<T, NVSHMEMI_THREADGROUP_WARP, _LL_MAX_UNROLL>(
                    pWrk + pack_offset + psync_element_offset, source + data_element_offset,
                    max_data_elems_per_warp, ll_flag, teami_node, num_pes_per_group, start_pe,
                    warp_id % num_warps_per_group);
            }
        }

        if (nelems > data_element_offset) {
            if (LL_VERSION == LL8) {
                nvshmemi_packLL<T, NVSHMEMI_THREADGROUP_WARP, 1>(
                    pWrk + pack_offset + psync_element_offset, source + data_element_offset,
                    nelems - data_element_offset, ll_flag, teami_node, num_pes_per_group, start_pe);
            } else {
                nvshmemi_packLL128<T, NVSHMEMI_THREADGROUP_WARP, 1>(
                    pWrk + pack_offset + psync_element_offset, source + data_element_offset,
                    nelems - data_element_offset, ll_flag, teami_node, num_pes_per_group, start_pe,
                    warp_id % num_warps_per_group);
                ;
            }
        }
    }

    /* todo: also try unrolling in recvLL */
    num_pes_per_group = teami->size / num_warp_groups;
    remaining_pes = teami->size % num_warp_groups;
    /* first n ggroups take on an extra PE in the case of remainder */
    num_pes_per_group += warp_group_id < remaining_pes ? 1 : 0;

    start_pe = num_pes_per_group * warp_group_id;
    start_pe += warp_group_id >= remaining_pes ? remaining_pes : 0;
    data_element_offset = warp_id % num_warps_per_group * max_data_elems_per_warp;
    psync_element_offset = warp_id % num_warps_per_group * max_psync_elems_per_warp;
    for (; data_element_offset + max_data_elems_per_warp < nelems;
         data_element_offset += num_warps_per_group * max_data_elems_per_warp,
         psync_element_offset += num_warps_per_group * max_psync_elems_per_warp) {
        for (next_pe = start_pe; next_pe < start_pe + num_pes_per_group; next_pe++) {
            if (LL_VERSION == LL8) {
                pack_offset =
                    next_pe * NVSHMEMI_TEAM_ROUND_UP((nelems * _FCOLLECT_LL8_PSYNC_SCALE_FACTOR),
                                                     16 / sizeof(T)) +
                    psync_element_offset;
                nvshmemi_recvLL<T, NVSHMEMI_THREADGROUP_WARP>(
                    dest + (next_pe * nelems) + data_element_offset,
                    (uint64_t *)(pWrk + pack_offset), max_data_elems_per_warp, ll_flag);
            } else {
                pack_offset = next_pe * NVSHMEMI_FCOLLECT_LL128_CALC_PSYNC_SIZE(nelems, T) +
                              psync_element_offset;
                nvshmemi_recvLL128<T, _LL_MAX_UNROLL>(
                    dest + (next_pe * nelems) + data_element_offset, pWrk + pack_offset,
                    max_data_elems_per_warp, ll_flag);
            }
        }
    }

    if (nelems > data_element_offset) {
        for (next_pe = start_pe; next_pe < start_pe + num_pes_per_group; next_pe++) {
            if (LL_VERSION == LL8) {
                pack_offset =
                    next_pe * NVSHMEMI_TEAM_ROUND_UP((nelems * _FCOLLECT_LL8_PSYNC_SCALE_FACTOR),
                                                     16 / sizeof(T)) +
                    psync_element_offset;
                nvshmemi_recvLL<T, NVSHMEMI_THREADGROUP_WARP>(
                    dest + (next_pe * nelems) + data_element_offset,
                    (uint64_t *)(pWrk + pack_offset), nelems - data_element_offset, ll_flag);
            } else {
                pack_offset = next_pe * NVSHMEMI_FCOLLECT_LL128_CALC_PSYNC_SIZE(nelems, T) +
                              psync_element_offset;
                nvshmemi_recvLL128<T, 1>(dest + (next_pe * nelems) + data_element_offset,
                                         pWrk + pack_offset, nelems - data_element_offset, ll_flag);
            }
        }
    }

out:

    nvshmemi_threadgroup_sync<SCOPE>();
}

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_fcollect_allpush_threadgroup(nvshmem_team_t team, T *dest,
                                                             const T *source, int dest_offset,
                                                             size_t nelems) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int PE_start = teami->start;
    int PE_stride = teami->stride;
    int PE_size = teami->size;
    int stride = PE_stride;
    int next_rank;
    const int mype = nvshmemi_device_state_d.mype;
    int my_idx_in_active_set = (mype - PE_start) / PE_stride;
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();

    // nvshmemi_threadgroup_sync<SCOPE>();
    for (int ii = 0; ii < PE_size; ii++) {
        next_rank = PE_start + ((my_idx_in_active_set + ii) % PE_size) * stride;
        nvshmemi_put_nbi_threadgroup<T, SCOPE>(dest + dest_offset, source, nelems, next_rank);
    }
    nvshmemi_barrier_threadgroup<SCOPE>(team);
}

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_fcollect_p2p_allpush_threadgroup(nvshmem_team_t team, T *dest,
                                                                 const T *source, int dest_offset,
                                                                 size_t nelems) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int PE_start = teami->start;
    int PE_stride = teami->stride;
    int PE_size = teami->size;
    int stride = PE_stride;
    int next_rank;
    const int mype = nvshmemi_device_state_d.mype;
    int my_idx_in_active_set = (mype - PE_start) / PE_stride;
    T *dst_ptr;
    nvshmemi_threadgroup_sync<SCOPE>();
    for (int ii = 0; ii < PE_size; ii++) {
        next_rank = PE_start + ((my_idx_in_active_set + ii) % PE_size) * stride;
        dst_ptr = (T *)nvshmemi_ptr((void *)(dest + dest_offset), next_rank);
        nvshmemi_memcpy_threadgroup<SCOPE>(dst_ptr, source, nelems * sizeof(T));
    }
    nvshmemi_barrier_threadgroup<SCOPE>(team);
}

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_fcollect_nvls_allpush_threadgroup(nvshmem_team_t team, T *dest,
                                                                  const T *source, int dest_offset,
                                                                  size_t nelems) {
#if __CUDA_ARCH__ >= 900 && CUDART_VERSION >= 12010
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    nvshmemi_threadgroup_sync<SCOPE>();
    T *dst_ptr = (T *)nvshmemi_mc_ptr(teami, (void *)(dest + dest_offset));
    nvshmemi_mcast_memcpy_threadgroup<T, SCOPE>(dst_ptr, source, nelems * sizeof(T));
    nvshmemi_barrier_threadgroup<SCOPE>(team);
#else
    assert(0 && "NVLS is not supported on this platform");
#endif
}

template <typename T, threadgroup_t SCOPE>
__device__ __forceinline__ void nvshmemi_fcollect_threadgroup(nvshmem_team_t team, T *dest,
                                                              const T *source, int dest_offset,
                                                              size_t nelems) {
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int nthreads = nvshmemi_threadgroup_size<SCOPE>();
    if (!myIdx) /* Only one thread should increment fcollect_count */
        nvshmemi_device_state_d.team_pool[team]->fcollect_count += 1;
    nvshmemi_threadgroup_sync<SCOPE>();
    int fcollect_algo = nvshmemi_device_state_d.gpu_coll_env_params_var.fcollect_algo;
    int p2p_direct =
        (nvshmemi_device_state_d.job_connectivity <= NVSHMEMI_JOB_GPU_LDST_REMOTE_ATOMICS);
    const size_t fcollect_ll_threshold =
        nvshmemi_device_state_d.gpu_coll_env_params_var.fcollect_ll_threshold;
    const size_t fcollect_ll128_threshold =
        nvshmemi_device_state_d.gpu_coll_env_params_var.fcollect_ll128_threshold;
    /* NVLS LL performs better with block scoped than thread/warp scoped operations
       due to better efficiency of distributing cvt/pack/unpack ops across threads across GPUs */
    const uint8_t prefer_nvls_ll = (SCOPE == NVSHMEMI_THREADGROUP_BLOCK);
    bool valid_ll_configuration = SCOPE != NVSHMEMI_THREADGROUP_THREAD && (nthreads % 32 == 0) &&
                                  sizeof(T) >= sizeof(uint32_t) && nelems % 2 == 0;
    /* DISABLE non NVLS LL for hybrid MNNVL configurations. */
    valid_ll_configuration &=
        (nvshmemi_device_state_d.team_pool[NVSHMEM_TEAM_WORLD_INDEX]->are_gpus_p2p_connected ||
         nvshmemi_device_state_d.team_pool[NVSHMEM_TEAM_SHARED]->is_team_node ||
         !nvshmemi_device_state_d.team_pool[NVSHMEM_TEAM_NODE_INDEX]->are_gpus_p2p_connected);
    /* This 2-level selection logic is implemented to reduce code duplication of calling leaf
     * functions on the device code */
    switch (fcollect_algo) {
        case 0: /* default selection */ {
            if (valid_ll_configuration && fcollect_ll_threshold >= (nelems * sizeof(T))) {
                fcollect_algo = FCOLLECT_LL8; /* LL algorithm */
            } else if (valid_ll_configuration && (nelems * sizeof(T)) < fcollect_ll128_threshold) {
                fcollect_algo = FCOLLECT_LL128;
            } else if (sizeof(T) >= sizeof(uint32_t) && (nelems % 2 == 0) &&
                       nvshmemi_device_state_d.gpu_coll_env_params_var.fcollect_ll_threshold >=
                           (nelems * sizeof(T)) &&
                       nvshmemi_device_state_d.team_pool[team]->nvls_rsc_base_ptr != NULL &&
                       prefer_nvls_ll) {
                fcollect_algo = FCOLLECT_NVLS_LL; /* NVLS LL algorithm */
            } else if (nvshmemi_device_state_d.team_pool[team]->nvls_rsc_base_ptr != NULL &&
                       (nelems * sizeof(T)) % 4 == 0) {
                fcollect_algo = FCOLLECT_NVLS; /* NVLS One shot algorithm */
            } else {
                fcollect_algo = FCOLLECT_ONESHOT; /* P2P One shot algorithm */
            }
        } break;
        case FCOLLECT_LL8: /* LL algorithm */
            break;
        case FCOLLECT_ONESHOT: /* One shot */
            break;
        case FCOLLECT_NVLS:
            if (nvshmemi_device_state_d.team_pool[team]->nvls_rsc_base_ptr != NULL &&
                (nelems * sizeof(T)) % 4 == 0) {
                /* NVLS simple */
                break;
            } else {
                fcollect_algo = FCOLLECT_ONESHOT; /* One shot */
                break;
            }
        case FCOLLECT_NVLS_LL:
            if (sizeof(T) >= sizeof(uint32_t) && (nelems % 2 == 0) &&
                nvshmemi_device_state_d.gpu_coll_env_params_var.fcollect_ll_threshold >=
                    (nelems * sizeof(T)) &&
                nvshmemi_device_state_d.team_pool[team]->nvls_rsc_base_ptr != NULL) {
                fcollect_algo = FCOLLECT_NVLS_LL; /* Use NVLS LL */
            } else if (nvshmemi_device_state_d.team_pool[team]->nvls_rsc_base_ptr != NULL &&
                       (nelems * sizeof(T)) % 4 == 0) {
                fcollect_algo = FCOLLECT_NVLS; /* Switch to NVLS simple */
            } else {
                fcollect_algo = FCOLLECT_ONESHOT; /* One shot */
            }
            break;
        case FCOLLECT_LL128: /* LL 128 */
            break;
        default:
            assert(0 && "Unsupported fcollect algo");
            break;
    }

    switch (fcollect_algo) {
        case FCOLLECT_LL8:
            if (nvshmemi_device_state_d.team_pool[NVSHMEM_TEAM_NODE_INDEX]
                    ->are_gpus_p2p_connected) {
                nvshmemi_fcollect_allpush_ll_threadgroup<T, SCOPE, LL8, true>(team, dest, source,
                                                                              nelems);
            } else {
                nvshmemi_fcollect_allpush_ll_threadgroup<T, SCOPE, LL8, false>(team, dest, source,
                                                                               nelems);
            }
            break;
        case FCOLLECT_ONESHOT:
            if (p2p_direct)
                nvshmemi_fcollect_p2p_allpush_threadgroup<T, SCOPE>(team, dest, source, dest_offset,
                                                                    nelems);
            else
                nvshmemi_fcollect_allpush_threadgroup<T, SCOPE>(team, dest, source, dest_offset,
                                                                nelems);
            break;
        case FCOLLECT_NVLS:
            nvshmemi_fcollect_nvls_allpush_threadgroup<T, SCOPE>(team, dest, source, dest_offset,
                                                                 nelems);
            break;
        case FCOLLECT_LL128:
            if (nvshmemi_device_state_d.team_pool[NVSHMEM_TEAM_NODE_INDEX]
                    ->are_gpus_p2p_connected) {
                nvshmemi_fcollect_allpush_ll_threadgroup<T, SCOPE, LL128, true>(team, dest, source,
                                                                                nelems);
            } else {
                nvshmemi_fcollect_allpush_ll_threadgroup<T, SCOPE, LL128, false>(team, dest, source,
                                                                                 nelems);
            }
            break;
        case FCOLLECT_NVLS_LL:
            nvshmemi_fcollect_nvls_ll_threadgroup<T, SCOPE>(team, dest, source, nelems);
            break;
        default:
            assert(0);
            break;
    }
}

#endif /* __CUDA_ARCH__ */
#endif /* FCOLLECT_DEVICE_CUH */
