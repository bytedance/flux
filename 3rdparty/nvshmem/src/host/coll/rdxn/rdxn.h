/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef NVSHMEMI_RDXN_COMMON_CPU_H
#define NVSHMEMI_RDXN_COMMON_CPU_H
#include <driver_types.h>
#include <stddef.h>

#include "cpu_coll.h"
#include "non_abi/nvshmem_build_options.h"
#include "device_host/nvshmem_common.cuh"
#include "internal/host/nvshmem_internal.h"
#include "device_host/nvshmem_types.h"
#include "internal/host/util.h"
#ifdef NVSHMEM_USE_NCCL
#include "nccl.h"
#endif

template <typename TYPE, rdxn_ops_t OP>
void nvshmemi_call_rdxn_on_stream_kernel(nvshmem_team_t team, TYPE *dest, const TYPE *source,
                                         size_t nreduce, cudaStream_t stream);

template <typename TYPE, rdxn_ops_t OP>
int nvshmemi_reduce_on_stream(nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nreduce,
                              cudaStream_t stream) {
#ifdef NVSHMEM_USE_NCCL
    nvshmemi_team_t *teami = nvshmemi_team_pool[team];
    if (nvshmemi_use_nccl && nvshmemi_get_nccl_op<OP>() != ncclNumOps &&
        nvshmemi_get_nccl_dt<TYPE>() != ncclNumTypes) {
        NCCL_CHECK(nccl_ftable.AllReduce(source, dest, nreduce, nvshmemi_get_nccl_dt<TYPE>(),
                                         nvshmemi_get_nccl_op<OP>(), (ncclComm_t)teami->nccl_comm,
                                         stream));
    } else
#endif /* NVSHMEM_USE_NCCL */
    {
        nvshmemi_call_rdxn_on_stream_kernel<TYPE, OP>(team, dest, source, nreduce, stream);
    }
    return 0;
}

#endif /* NVSHMEMI_RDXN_COMMON_CPU_H */
