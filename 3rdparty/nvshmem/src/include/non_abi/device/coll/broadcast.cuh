/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef BROADCAST_DEVICE_CUH
#define BROADCAST_DEVICE_CUH

#include <cuda_runtime.h>
#include "device_host_transport/nvshmem_common_transport.h"  // for NVSHMEMI_OP_PUT
#include "non_abi/nvshmem_build_options.h"
#ifdef NVSHMEM_ENABLE_ALL_DEVICE_INLINING
#include "non_abi/device/pt-to-pt/transfer_device.cuh"
#else
#include "non_abi/device/pt-to-pt/nvshmemi_transfer_api.cuh"
#endif
#include "non_abi/device/wait/nvshmemi_wait_until_apis.cuh"
#include "non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh"
#include "non_abi/device/common/nvshmemi_common_device.cuh"
#include "non_abi/device/team/nvshmemi_team_defines.cuh"

#ifdef __CUDA_ARCH__
template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_bcast_intranode_tree_threadgroup(nvshmem_team_t team, T *dest,
                                                                 const T *source, size_t nelems,
                                                                 int PE_root, int k) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    const int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    const int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    if (!myIdx) { /* Only one thread should increment */
        teami->ll_flag++;
        teami->bcast_count++;
        if (teami->bcast_sync_offset + nelems * sizeof(T) * 2 >
            sizeof(long) * NVSHMEMI_BCAST_SYNC_SIZE) {
            nvshmemi_barrier_threadgroup<NVSHMEMI_THREADGROUP_THREAD>(team);
            teami->bcast_sync_offset = 0;
        }
    }
    nvshmemi_threadgroup_sync<SCOPE>();
    const uint32_t ll_flag = teami->ll_flag;
    char *pWrk = (char *)nvshmemi_team_get_psync(teami, BCAST);
    size_t recv_offset = teami->bcast_sync_offset;
    const int my_pe_in_team = nvshmemi_team_my_pe(team);

    if (PE_root != my_pe_in_team) {
        nvshmemi_recvLL<T, SCOPE>(dest, (uint64_t *)(pWrk + recv_offset), nelems, ll_flag);
    } else {
        nvshmemi_packLL_naive<T, SCOPE>((uint64_t *)(pWrk + recv_offset), source, nelems, ll_flag);
    }

    /*if (SCOPE == NVSHMEMI_THREADGROUP_BLOCK) {
        nvshmemi_threadgroup_sync<SCOPE>(); // wait for block scoped recvLL/packLL to complete
        int warp_id = myIdx / 32;
        int num_warps = (groupSize + 31) / 32;
        size_t size_per_peer = nelems * sizeof(T) * 2;
        int num_peers = k;
        size_t total_size = size_per_peer * num_peers;
        int size_per_warp = min (nelems * sizeof(T) * 2, 512l);
        for (size_t start = warp_id * size_per_warp; start < total_size; start += num_warps *
    size_per_warp) { int peer_id = start / size_per_peer; size_t offset = start % size_per_peer; int
    child_in_team = ((my_pe_in_team + (teami->size - PE_root)) % teami->size) * k + peer_id + 1; if
    (child_in_team >= teami->size) break; child_in_team = (child_in_team + PE_root) % teami->size;
            int child =
                nvshmemi_team_translate_pe(teami, child_in_team,
    nvshmemi_device_state_d.team_pool[NVSHMEM_TEAM_WORLD_INDEX]);
            //printf("sending data %d to %d at offset %llu, peer_id: %d, size_per_peer: %llu,
    nelems: %llu\n", size_per_warp, child, offset, peer_id, size_per_peer, nelems);
            //printf("myIdx: %d, start: %lld, num_warps: %d\n", myIdx, start, num_warps);
            nvshmemi_put_nbi_threadgroup<char, NVSHMEMI_THREADGROUP_WARP>(pWrk + recv_offset +
    offset, pWrk + recv_offset + offset, size_per_warp, child);
        }
    } else  */
    {
        nvshmemi_threadgroup_sync<SCOPE>();
        for (int i = 0; i < k; i++) {
            int child_in_team =
                ((my_pe_in_team + (teami->size - PE_root)) % teami->size) * k + i + 1;
            if (child_in_team >= teami->size) break;
            child_in_team = (child_in_team + PE_root) % teami->size;
            int child = nvshmemi_team_translate_pe(team, child_in_team, NVSHMEM_TEAM_WORLD_INDEX);

            nvshmemii_put_nbi_threadgroup<uint64_t, SCOPE>(
                (uint64_t *)(pWrk + recv_offset), (uint64_t *)(pWrk + recv_offset),
                nelems * sizeof(T) / sizeof(uint32_t), child);
        }
    }
    if (PE_root == my_pe_in_team && dest != source)
        nvshmemi_memcpy_threadgroup<SCOPE>(dest, source, nelems * sizeof(T));
    if (!myIdx) { /* Only one thread should increment */
        teami->bcast_sync_offset += sizeof(T) * nelems * 2;
    }
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_bcast_internode_tree_threadgroup(nvshmem_team_t team, T *dest,
                                                                 const T *source, size_t nelems,
                                                                 int PE_root, int k) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    const int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    const int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    if (!myIdx) { /* Only one thread should increment */
        teami->ll_flag++;
        teami->bcast_count++;
        if ((teami->bcast_sync_offset + nelems * sizeof(T) * 2) >
            (sizeof(long) * NVSHMEMI_BCAST_SYNC_SIZE)) {
            nvshmemi_barrier_threadgroup<NVSHMEMI_THREADGROUP_THREAD>(team);
            teami->bcast_sync_offset = 0;
        }
    }
    nvshmemi_threadgroup_sync<SCOPE>();
    const uint32_t ll_flag = teami->ll_flag;
    char *pWrk = (char *)nvshmemi_team_get_psync(teami, BCAST);
    size_t recv_offset = teami->bcast_sync_offset;
    const int my_pe_in_team = nvshmemi_team_my_pe(team);

    if (PE_root != my_pe_in_team) {
        nvshmemi_recvLL<T, SCOPE>(dest, (uint64_t *)(pWrk + recv_offset), nelems, ll_flag);
    } else {
        nvshmemi_packLL_naive<T, SCOPE>((uint64_t *)(pWrk + recv_offset), source, nelems, ll_flag);
    }
    nvshmemi_threadgroup_sync<SCOPE>();
    for (int i = myIdx; i < k; i += groupSize) {
        int child_in_team = ((my_pe_in_team + (teami->size - PE_root)) % teami->size) * k + i + 1;
        if (child_in_team >= teami->size) break;
        child_in_team = (child_in_team + PE_root) % teami->size;
        int child = nvshmemi_team_translate_pe(team, child_in_team, NVSHMEM_TEAM_WORLD_INDEX);

        nvshmemi_put_nbi_threadgroup<uint64_t, NVSHMEMI_THREADGROUP_THREAD>(
            (uint64_t *)(pWrk + recv_offset), (uint64_t *)(pWrk + recv_offset),
            nelems * sizeof(T) / sizeof(uint32_t), child);
    }
    if (PE_root == my_pe_in_team && dest != source)
        nvshmemi_memcpy_threadgroup<SCOPE>(dest, source, nelems * sizeof(T));
    if (!myIdx) { /* Only one thread should increment */
        teami->bcast_sync_offset += sizeof(T) * nelems * 2;
    }
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_bcast_tree_threadgroup(nvshmem_team_t team, T *dest,
                                                       const T *source, size_t nelems, int PE_root,
                                                       int k) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    const int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    const int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    if (!myIdx) { /* Only one thread should increment */
        teami->ll_flag++;
        teami->bcast_count++;
        if ((teami->bcast_sync_offset + nelems * sizeof(T) * 2) >
            (sizeof(long) * NVSHMEMI_BCAST_SYNC_SIZE)) {
            nvshmemi_barrier_threadgroup<NVSHMEMI_THREADGROUP_THREAD>(team);
            teami->bcast_sync_offset = 0;
        }
    }
    nvshmemi_threadgroup_sync<SCOPE>();
    const uint32_t ll_flag = teami->ll_flag;
    char *pWrk = (char *)nvshmemi_team_get_psync(teami, BCAST);
    size_t recv_offset = teami->bcast_sync_offset;
    const int my_pe_in_team = nvshmemi_team_my_pe(team);

    if (PE_root != my_pe_in_team) {
        nvshmemi_recvLL<T, SCOPE>(dest, (uint64_t *)(pWrk + recv_offset), nelems, ll_flag);
    } else {
        nvshmemi_packLL_naive<T, SCOPE>((uint64_t *)(pWrk + recv_offset), source, nelems, ll_flag);
    }
    nvshmemi_threadgroup_sync<SCOPE>();
    /* Do remote transfers first */
    for (int i = myIdx; i < k; i += groupSize) {
        int child_in_team = ((my_pe_in_team + (teami->size - PE_root)) % teami->size) * k + i + 1;
        if (child_in_team >= teami->size) break;
        child_in_team = (child_in_team + PE_root) % teami->size;
        int child = nvshmemi_team_translate_pe(team, child_in_team, NVSHMEM_TEAM_WORLD_INDEX);
        bool is_remote = (nvshmemi_ptr(pWrk, child) == NULL) ? true : false;
        if (is_remote)
            nvshmemi_put_nbi_threadgroup<uint64_t, NVSHMEMI_THREADGROUP_THREAD>(
                (uint64_t *)(pWrk + recv_offset), (uint64_t *)(pWrk + recv_offset),
                nelems * sizeof(T) / sizeof(uint32_t), child);
    }

    /* Do P2P transfers */
    for (int i = 0; i < k; i++) {
        int child_in_team = ((my_pe_in_team + (teami->size - PE_root)) % teami->size) * k + i + 1;
        if (child_in_team >= teami->size) break;
        child_in_team = (child_in_team + PE_root) % teami->size;
        int child = nvshmemi_team_translate_pe(team, child_in_team, NVSHMEM_TEAM_WORLD_INDEX);

        bool is_remote = (nvshmemi_ptr(pWrk, child) == NULL) ? true : false;
        if (!is_remote)
            nvshmemii_put_nbi_threadgroup<uint64_t, SCOPE>(
                (uint64_t *)(pWrk + recv_offset), (uint64_t *)(pWrk + recv_offset),
                nelems * sizeof(T) / sizeof(uint32_t), child);
    }

    if (PE_root == my_pe_in_team && dest != source)
        nvshmemi_memcpy_threadgroup<SCOPE>(dest, source, nelems * sizeof(T));
    if (!myIdx) { /* Only one thread should increment */
        teami->bcast_sync_offset += sizeof(T) * nelems * 2;
    }
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_bcast_nonLL_tree_threadgroup(nvshmem_team_t team, T *dest,
                                                             const T *source, size_t nelems,
                                                             int PE_root, int k) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    const int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    const int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    if (!myIdx) { /* Only one thread should increment */
        teami->ll_flag++;
        teami->bcast_count++;
        if ((teami->bcast_sync_offset + sizeof(uint64_t)) >
            (sizeof(long) * NVSHMEMI_BCAST_SYNC_SIZE)) {
            nvshmemi_barrier_threadgroup<NVSHMEMI_THREADGROUP_THREAD>(team);
            teami->bcast_sync_offset = 0;
        }
    }
    nvshmemi_threadgroup_sync<SCOPE>();
    const uint32_t ll_flag = teami->ll_flag;
    char *pWrk = (char *)nvshmemi_team_get_psync(teami, BCAST);
    size_t recv_offset = teami->bcast_sync_offset;
    const int my_pe_in_team = nvshmemi_team_my_pe(team);

    if (PE_root != my_pe_in_team) {
        nvshmemi_wait_until<uint64_t>((uint64_t *)(pWrk + recv_offset), NVSHMEM_CMP_EQ, ll_flag);
    }
    nvshmemi_threadgroup_sync<SCOPE>();
    /* Do remote transfers first */
    for (int i = myIdx; i < k; i += groupSize) {
        int child_in_team = ((my_pe_in_team + (teami->size - PE_root)) % teami->size) * k + i + 1;
        if (child_in_team >= teami->size) break;
        child_in_team = (child_in_team + PE_root) % teami->size;
        int child = nvshmemi_team_translate_pe(team, child_in_team, NVSHMEM_TEAM_WORLD_INDEX);
        bool is_remote = (nvshmemi_ptr(pWrk, child) == NULL) ? true : false;
        if (is_remote)
            nvshmemi_put_signal_threadgroup<T, NVSHMEMI_THREADGROUP_THREAD>(
                dest, (PE_root == my_pe_in_team) ? source : dest, nelems,
                (uint64_t *)(pWrk + recv_offset), ll_flag, NVSHMEMI_AMO_SIGNAL_SET, child, 1);
    }

    /* Do P2P transfers */
    for (int i = 0; i < k; i++) {
        int child_in_team = ((my_pe_in_team + (teami->size - PE_root)) % teami->size) * k + i + 1;
        if (child_in_team >= teami->size) break;
        child_in_team = (child_in_team + PE_root) % teami->size;
        int child = nvshmemi_team_translate_pe(team, child_in_team, NVSHMEM_TEAM_WORLD_INDEX);

        bool is_remote = (nvshmemi_ptr(pWrk, child) == NULL) ? true : false;
        if (!is_remote)
            nvshmemii_put_signal_threadgroup<T, SCOPE>(
                dest, (PE_root == my_pe_in_team) ? source : dest, nelems,
                (uint64_t *)(pWrk + recv_offset), ll_flag, NVSHMEMI_AMO_SIGNAL_SET, child, 1);
    }

    if (PE_root == my_pe_in_team && dest != source)
        nvshmemi_memcpy_threadgroup<SCOPE>(dest, source, nelems * sizeof(T));
    if (!myIdx) { /* Only one thread should increment */
        nvshmemi_quiet<NVSHMEMI_THREADGROUP_THREAD>();
        teami->bcast_sync_offset +=
            2 * sizeof(uint64_t); /* incrementing minimally by 16 bytes because this buffer is used
                                     by packLL and packLL does 16byte writes */
    }
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_bcast_put2all_threadgroup(nvshmem_team_t team, T *dest,
                                                          const T *source, size_t nelems,
                                                          int PE_root) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int i;
    int PE_start = teami->start;
    int PE_stride = teami->stride;
    int PE_size = teami->size;
    int stride = PE_stride;
    int root = nvshmemi_team_translate_pe(team, PE_root, NVSHMEM_TEAM_WORLD_INDEX);
    int PE_end = PE_start + (stride * PE_size);
    if (root == nvshmemi_device_state_d.mype) {
        for (i = PE_start; i < PE_end; i += stride) {
            nvshmemi_put_nbi_threadgroup<T, SCOPE>(dest, source, nelems, i);
        }
    }
    nvshmemi_barrier_threadgroup<SCOPE>(team);
}

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_bcast_put2all_direct_threadgroup(nvshmem_team_t team, T *dest,
                                                                 const T *source, size_t nelems,
                                                                 int PE_root) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int i;
    int PE_start = teami->start;
    int PE_stride = teami->stride;
    int PE_size = teami->size;
    int stride = PE_stride;
    int root = nvshmemi_team_translate_pe(team, PE_root, NVSHMEM_TEAM_WORLD_INDEX);
    int PE_end = PE_start + (stride * PE_size);
    T *dst_ptr;
    if (root == nvshmemi_device_state_d.mype) {
        for (i = PE_start; i < PE_end; i += stride) {
            dst_ptr = (T *)nvshmemi_ptr(dest, i);
            nvshmemi_memcpy_threadgroup<SCOPE>(dst_ptr, source, nelems * sizeof(T));
        }
    }
    nvshmemi_barrier_threadgroup<SCOPE>(team);
}

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_bcast_hierarchical_threadgroup(nvshmem_team_t team, T *dest,
                                                               const T *source, size_t nelems,
                                                               int PE_root) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    if (teami->is_team_same_mype_node) {
        nvshmemi_bcast_internode_tree_threadgroup<T, SCOPE>(
            team, dest, source, nelems, PE_root,
            nvshmemi_device_state_d.gpu_coll_env_params_var.bcast_tree_kval);
    } else if (teami->is_team_node) {
        nvshmemi_bcast_intranode_tree_threadgroup<T, SCOPE>(
            team, dest, source, nelems, PE_root,
            nvshmemi_device_state_d.gpu_coll_env_params_var.bcast_tree_kval);
    } else {
        int team_npes_node = nvshmemi_team_n_pes(teami->team_node);
        int PE_root_idx_in_team_node = PE_root % team_npes_node;
        int my_idx_in_team_node = nvshmemi_team_my_pe(team) % team_npes_node;
        if (PE_root_idx_in_team_node == my_idx_in_team_node) {
            nvshmemi_bcast_internode_tree_threadgroup<T, SCOPE>(
                teami->team_same_mype_node, dest, source, nelems, PE_root / team_npes_node,
                nvshmemi_device_state_d.gpu_coll_env_params_var.bcast_tree_kval);
        }
        nvshmemi_bcast_intranode_tree_threadgroup<T, SCOPE>(
            teami->team_node, dest, dest, nelems, PE_root_idx_in_team_node,
            nvshmemi_device_state_d.gpu_coll_env_params_var.bcast_tree_kval);
    }
}

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_broadcast_threadgroup(nvshmem_team_t team, T *dest, const T *source,
                                                      size_t nelems, int PE_root) {
    int bcast_algo = nvshmemi_device_state_d.gpu_coll_env_params_var.bcast_algo;
    switch (bcast_algo) {
        case 0:
            if (NVSHMEMI_BCAST_SYNC_SIZE * sizeof(long) >= (nelems * sizeof(T) * 2) &&
                sizeof(T) >= sizeof(uint32_t) && nelems % 2 == 0 &&
                nelems * sizeof(T) <= 16384) { /* LL algos */
                if (nvshmemi_team_n_pes(team) > 32 &&
                    nvshmemi_device_state_d.pe_dist ==
                        NVSHMEMI_PE_DIST_BLOCK) { /* hierarchical topo-aware */
                    bcast_algo = 2;
                } else
                    bcast_algo = 3;
            } else /* non-LL algorithm */
                bcast_algo = 4;
            break;
        case 1: /* Brutefoce algorithm: send one to all followed by barrier */
            break;
        case 2: /* Topology aware - two level hierarchical algorithm with LL approach */
            if (NVSHMEMI_BCAST_SYNC_SIZE * sizeof(long) >= (nelems * sizeof(T) * 2) &&
                sizeof(T) >= sizeof(uint32_t) && nelems % 2 == 0 &&
                nvshmemi_device_state_d.pe_dist == NVSHMEMI_PE_DIST_BLOCK) {
            } else {
                /*printf("User selected algo: %d, but it is not supported with currect config, \
                        using default algo selection strategy..\n", \
                        nvshmemi_device_state_d.gpu_coll_env_params_var.bcast_algo);*/
                bcast_algo = 1;
            }
            break;
        case 3: /* Topology unaware tree algrithm with LL approach*/
            if (NVSHMEMI_BCAST_SYNC_SIZE * sizeof(long) >= (nelems * sizeof(T) * 2) &&
                sizeof(T) >= sizeof(uint32_t) && nelems % 2 == 0) {
            } else {
                /*printf("User selected algo: %d, but it is not supported with currect config, \
                        using default algo selection strategy..\n",
                        nvshmemi_device_state_d.gpu_coll_env_params_var.bcast_algo);*/
                bcast_algo = 1;
            }
            break;
        case 4: /* Topology unaware flat tree algrithm with LL approach*/
            if (NVSHMEMI_BCAST_SYNC_SIZE * sizeof(long) >= (nelems * sizeof(T) * 2) &&
                sizeof(T) >= sizeof(uint32_t) && nelems % 2 == 0) {
            } else {
                /* printf("User selected algo: %d, but it is not supported with currect config, \
                        using default algo selection strategy..\n",
                        nvshmemi_device_state_d.gpu_coll_env_params_var.bcast_algo);*/
                bcast_algo = 1;
            }
            break;
        default:
            printf("Specified bcast algo:%d not supported, aborting...\n", bcast_algo);
            assert(0);
            break;
    }

    switch (bcast_algo) {
        case 1: /* Brutefoce algorithm: send one to all followed by barrier */
            if (nvshmemi_device_state_d.job_connectivity <= NVSHMEMI_JOB_GPU_LDST_REMOTE_ATOMICS) {
                nvshmemi_bcast_put2all_direct_threadgroup<T, SCOPE>(team, dest, source, nelems,
                                                                    PE_root);
            } else {
                nvshmemi_bcast_put2all_threadgroup<T, SCOPE>(team, dest, source, nelems, PE_root);
            }
            break;
        case 2: /* Topology aware - two level hierarchical algorithm with LL approach */
            nvshmemi_bcast_hierarchical_threadgroup<T, SCOPE>(team, dest, source, nelems, PE_root);
            break;
        case 3: /* Topology unaware tree algrithm with LL approach*/
            nvshmemi_bcast_tree_threadgroup<T, SCOPE>(
                team, dest, source, nelems, PE_root,
                nvshmemi_device_state_d.gpu_coll_env_params_var.bcast_tree_kval);
            break;
        case 4: /* Topology unaware flat tree algrithm with LL approach*/
            nvshmemi_bcast_nonLL_tree_threadgroup<T, SCOPE>(team, dest, source, nelems, PE_root,
                                                            nvshmemi_team_n_pes(team) - 1);
            break;
        default:
            assert(0);
            break;
    }
}

#endif /* __CUDA_ARCH__ */

#endif
