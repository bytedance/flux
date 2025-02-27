/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef REDUCESCATTER_DEVICE_CUH
#define REDUCESCATTER_DEVICE_CUH

#include <cuda_runtime.h>
#include "non_abi/device/pt-to-pt/proxy_device.cuh"
#include "non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh"
#include "non_abi/device/common/nvshmemi_common_device.cuh"
#include "non_abi/device/team/nvshmemi_team_defines.cuh"
#include "non_abi/nvshmem_build_options.h"
#ifdef NVSHMEM_ENABLE_ALL_DEVICE_INLINING
#include "non_abi/device/pt-to-pt/transfer_device.cuh"
#else
#include "non_abi/device/pt-to-pt/nvshmemi_transfer_api.cuh"
#endif
#include "non_abi/device/wait/nvshmemi_wait_until_apis.cuh"
#include "utils.cuh"
#include "fcollect.cuh"
#include "broadcast.cuh"
#include "reduce.cuh"

#ifdef __CUDACC__
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;
#endif

#ifdef __CUDA_ARCH__

template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
__device__ inline void nvshmemi_reducescatter_allpush_threadgroup(nvshmem_team_t team, TYPE *dest,
                                                                  const TYPE *source,
                                                                  size_t nreduce) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    TYPE *pWrk = (TYPE *)nvshmemi_team_get_psync(teami, REDUCE);

    if (teami->size == 1) {
        nvshmemi_memcpy_threadgroup<SCOPE>((char *)dest, (char *)source, sizeof(TYPE) * nreduce);
        nvshmemi_threadgroup_sync<SCOPE>();
        return;
    }

    if (nreduce * sizeof(TYPE) * teami->size >
        nvshmemi_device_state_d.gpu_coll_env_params_var.reduce_scratch_size * 0.5) {
        assert(0 &&
               "Not enough space to perform reducescatter. Increase value of "
               "NVSHMEM_REDUCE_SCRATCH_SIZE");
    }

    /* pWrk: pe0 data array, pe1 data array, pe2 data array, .... */
    for (int i = 1; i < teami->size; i++) {  // don't send to self
        int peer_pe_idx = (teami->my_pe + i) % teami->size;
        int peer_pe = nvshmemi_team_translate_pe(team, peer_pe_idx, NVSHMEM_TEAM_WORLD);
        nvshmemii_put_nbi_threadgroup<TYPE, SCOPE>(
            (TYPE *)((char *)pWrk + teami->my_pe * nreduce * sizeof(TYPE)),
            source + peer_pe_idx * nreduce, nreduce, peer_pe);
    }
    nvshmemi_barrier_threadgroup<SCOPE>(team);

    TYPE *op1, *op2;
    TYPE *op3;
    for (int pe = 1; pe < teami->size; pe++) {  // Need to perform npes - 1 reductions
        op3 = dest;
        if (pe == 1) {  // this is the first reduction
            if (nvshmemi_team_my_pe(team) == 0) {
                op1 = (TYPE *)source;
                op2 = (TYPE *)((char *)pWrk + nreduce * sizeof(TYPE));
            } else if (nvshmemi_team_my_pe(team) == 1) {
                op1 = (TYPE *)(char *)pWrk;
                op2 = (TYPE *)source + nreduce;
            } else {
                op1 = (TYPE *)(char *)pWrk;
                op2 = (TYPE *)((char *)pWrk + nreduce * sizeof(TYPE));
            }
        } else {
            op1 = dest;
            if (pe == nvshmemi_team_my_pe(team))
                op2 = (TYPE *)source + pe * nreduce;
            else
                op2 = (TYPE *)((char *)pWrk + pe * nreduce * sizeof(TYPE));
        }
        gpu_linear_reduce_threadgroup<TYPE, OP, SCOPE>(op1, op2, op3, nreduce);
    }
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
__device__ inline void nvshmemi_reducescatter_nvls_allpush_threadgroup(nvshmem_team_t team,
                                                                       TYPE *dest,
                                                                       const TYPE *source,
                                                                       size_t nreduce) {
#if __CUDA_ARCH__ >= 900 && CUDART_VERSION >= 12010
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int my_idx_in_active_set = (nvshmemi_device_state_d.mype - teami->start) / (teami->stride);
    int offset = nreduce * my_idx_in_active_set;
    TYPE *src_ptr = (TYPE *)nvshmemi_mc_ptr(teami, (void *)(source + offset));
    nvshmemi_threadgroup_sync<SCOPE>();
    nvshmemi_local_reduce_mcast_threadgroup<TYPE, OP, SCOPE>(dest, src_ptr, nreduce);
    /* Since ld.red is done atomically on the NVSwitch, the value obtained into local dest
     * ref for a given PE would be ready, right away. We can still have a case that after returning
     * from this kernel, source buffer can be mutated on one PE, while another PE is still
     * performing ld.red, causing data correctness issue. We don't however need to add
     * threadfence_system for ordering since the subsequent load to source buffer will be ordered
     * already to prior ld.reduce (RAR) by HW.
     */
    nvshmemi_sync_threadgroup<SCOPE>(team);
#else
    assert(0 && "Unsupported NVLS algo on this platform");
#endif
}

template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
__device__ inline void nvshmemi_reducescatter_threadgroup(nvshmem_team_t team, TYPE *dest,
                                                          const TYPE *source, size_t nreduce) {
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();

    if (!myIdx) /* Only one thread should increment rdxn_count */
        nvshmemi_device_state_d.team_pool[team]->rdxn_count += 1;
    nvshmemi_threadgroup_sync<SCOPE>();

    constexpr bool is_float_v = is_float<TYPE>::value;
    constexpr bool is_double_v = is_double<TYPE>::value;
    /* For SUM/AND/XOR/OR, support is untyped */
    /* For MIN/MAX, support is type specific */
    constexpr bool is_mcast_red_op =
        NVSHMEMI_MCAST_RDXN_OP_IS_CAP_UNTYPED(OP) ||
        ((OP == RDXN_OPS_MIN || OP == RDXN_OPS_MAX) && (!is_float_v && !is_double_v));
    constexpr bool is_mcast_type = (sizeof(TYPE) >= sizeof(uint32_t));

    int reducescatter_algo = nvshmemi_device_state_d.gpu_coll_env_params_var.reducescatter_algo;
    /* This 2-level selection logic is implemented to reduce code duplication of calling leaf
     * functions on the device code */
    switch (reducescatter_algo) {
        case 0: /* default selection */
            if (nvshmemi_device_state_d.team_pool[team]->nvls_rsc_base_ptr != NULL &&
                ((nreduce * sizeof(TYPE)) % 4 == 0) && is_mcast_type && is_mcast_red_op) {
                reducescatter_algo = 3; /* NVLS One Shot */
            } else {
                reducescatter_algo = 2; /* P2P One Shot */
            }
            break;
        case 2: /* One Shot */
            break;
        case 3:
            if (nvshmemi_device_state_d.team_pool[team]->nvls_rsc_base_ptr != NULL &&
                ((nreduce * sizeof(TYPE)) % 4 == 0) && is_mcast_type && is_mcast_red_op) {
                /* NVLS one shot */
                break;
            } else {
                reducescatter_algo = 2; /* One shot */
                break;
            }
        default:
            assert(0 && "Specified reducescatter algo not supported, aborting...\n");
            break;
    }

    switch (reducescatter_algo) {
        case 2:
            nvshmemi_reducescatter_allpush_threadgroup<TYPE, OP, SCOPE>(team, dest, source,
                                                                        nreduce);
            break;
        case 3:
            nvshmemi_reducescatter_nvls_allpush_threadgroup<TYPE, OP, SCOPE>(team, dest, source,
                                                                             nreduce);
            break;
        default:
            assert(0);
            break;
    }
}

#endif /* __CUDA_ARCH__ */
#endif /* REDUCESCATTER_DEVICE_CUH */
