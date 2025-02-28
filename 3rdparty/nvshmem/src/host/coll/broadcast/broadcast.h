/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef NVSHMEMI_BCAST_COMMON_CPU_H
#define NVSHMEMI_BCAST_COMMON_CPU_H
#include <driver_types.h>                    // for cudaStream_t, CUstr...
#include <stddef.h>                          // for size_t
#include "device_host/nvshmem_types.h"       // for nvshmemi_team_t
#include "cpu_coll.h"                        // for nvshmemi_get_nccl_dt
#include "device_host/nvshmem_common.cuh"    // for nvshmemi_team_pool
#include "internal/host/nvshmem_internal.h"  // for nvshmemi_use_nccl
#include "internal/host/util.h"              // for NCCL_CHECK
#include "non_abi/nvshmem_build_options.h"   // for NVSHMEM_USE_NCCL
#ifdef NVSHMEM_USE_NCCL
#include "nccl.h"
#endif

template <typename T>
void nvshmemi_call_broadcast_on_stream_kernel(nvshmem_team_t team, T *dest, const T *source,
                                              size_t nelems, int PE_root, cudaStream_t stream);

template <typename TYPE>
int nvshmemi_broadcast_on_stream(nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nelems,
                                 int PE_root, cudaStream_t stream) {
#ifdef NVSHMEM_USE_NCCL
    nvshmemi_team_t *teami = nvshmemi_team_pool[team];
    if (nvshmemi_use_nccl && nvshmemi_get_nccl_dt<TYPE>() != ncclNumTypes) {
        NCCL_CHECK(nccl_ftable.Broadcast(source, dest, nelems, nvshmemi_get_nccl_dt<TYPE>(),
                                         PE_root, (ncclComm_t)teami->nccl_comm, stream));
    } else
#endif
    {
        nvshmemi_call_broadcast_on_stream_kernel<TYPE>(team, dest, source, nelems, PE_root, stream);
    }
    return 0;
}
#endif
