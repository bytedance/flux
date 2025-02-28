/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef NVSHMEMI_ALLTOALL_COMMON_CPU_H
#define NVSHMEMI_ALLTOALL_COMMON_CPU_H
#include <driver_types.h>                    // for cudaStream_t, CUstream_st
#include <stddef.h>                          // for size_t
#include "cpu_coll.h"                        // for nccl_ftable, nccl_func...
#include "device_host/nvshmem_common.cuh"    // for nvshmemi_team_pool
#include "device_host/nvshmem_types.h"       // for nvshmemi_team_t, nvshm...
#include "host/nvshmem_api.h"                // for nvshmem_team_n_pes
#include "internal/host/nvshmem_internal.h"  // for nccl_version, nvshmemi...
#include "internal/host/util.h"              // for NCCL_CHECK
#include "non_abi/nvshmem_build_options.h"   // for NVSHMEM_USE_NCCL
#ifdef NVSHMEM_USE_NCCL
#include "nccl.h"  // for ncclComm, ncclComm_t
#endif

template <typename TYPE>
void nvshmemi_call_alltoall_on_stream_kernel(nvshmem_team_t team, TYPE *dest, const TYPE *source,
                                             size_t nelems, cudaStream_t stream);

template <typename TYPE>
int nvshmemi_alltoall_on_stream(nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nelems,
                                cudaStream_t stream) {
#ifdef NVSHMEM_USE_NCCL
    nvshmemi_team_t *teami = nvshmemi_team_pool[team];
    int team_n_pes = nvshmem_team_n_pes(team);
    if (nvshmemi_use_nccl && nvshmemi_get_nccl_dt<TYPE>() != ncclNumTypes &&
        ((nccl_version >= 2700 && team_n_pes <= 4096 /* NCCL limit for Group API */) ||
         (nccl_version >= 2800 && team_n_pes <= 32768 /* NCCL limit for Group API */))) {
        size_t rank_offset = nelems * sizeof(TYPE);
        NCCL_CHECK(nccl_ftable.GroupStart());
        for (int pe = 0; pe < team_n_pes; pe++) {
            NCCL_CHECK(nccl_ftable.Send(((char *)source) + pe * rank_offset, nelems,
                                        nvshmemi_get_nccl_dt<TYPE>(), pe,
                                        (ncclComm_t)teami->nccl_comm, stream));
            NCCL_CHECK(nccl_ftable.Recv(((char *)dest) + pe * rank_offset, nelems,
                                        nvshmemi_get_nccl_dt<TYPE>(), pe,
                                        (ncclComm_t)teami->nccl_comm, stream));
        }
        NCCL_CHECK(nccl_ftable.GroupEnd());
    } else
#endif /* NVSHMEM_USE_NCCL */
    {
        nvshmemi_call_alltoall_on_stream_kernel<TYPE>(team, dest, source, nelems, stream);
    }
    return 0;
}

#endif /* NVSHMEMI_ALLTOALL_COMMON_CPU_H */
