/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See COPYRIGHT.txt for license information
 */

#ifndef _NVSHMEM_COLL_API_H_
#define _NVSHMEM_COLL_API_H_
#include "device_host/nvshmem_types.h"
#include "device_host/nvshmem_common.cuh"
#include "host/nvshmem_macros.h"

#ifdef __cplusplus
extern "C" {
#endif
//===============================
// standard nvshmem collective calls
//===============================

// alltoall(s) collectives
#define DECL_NVSHMEM_TYPENAME_ALLTOALL(TYPENAME, TYPE)                                            \
    NVSHMEMI_HOSTDEVICE_PREFIX int nvshmem_##TYPENAME##_alltoall(nvshmem_team_t team, TYPE *dest, \
                                                                 const TYPE *src, size_t nelems);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DECL_NVSHMEM_TYPENAME_ALLTOALL)
#undef DECL_NVSHMEM_TYPENAME_ALLTOALL

#define DECL_NVSHMEM_TYPENAME_ALLTOALLS(TYPENAME, TYPE)                                            \
    NVSHMEMI_HOSTDEVICE_PREFIX int nvshmem_##TYPENAME##_alltoalls(nvshmem_team_t team, TYPE *dest, \
                                                                  const TYPE *src, ptrdiff_t dst,  \
                                                                  ptrdiff_t sst, size_t nelems);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DECL_NVSHMEM_TYPENAME_ALLTOALLS)
#undef DECL_NVSHMEM_TYPENAME_ALLTOALLS

NVSHMEMI_HOSTDEVICE_PREFIX int nvshmem_alltoallmem(nvshmem_team_t team, void *dest, const void *src,
                                                   size_t nelems);

// barrier collectives
NVSHMEMI_HOSTDEVICE_PREFIX int nvshmem_barrier(nvshmem_team_t team);
NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_barrier_all();

// sync collectives
NVSHMEMI_HOSTDEVICE_PREFIX int nvshmem_team_sync(nvshmem_team_t team);
NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_sync_all();
#define nvshmem_sync nvshmem_team_sync

// broadcast collectives
#define DECL_NVSHMEM_TYPENAME_BROADCAST(TYPENAME, TYPE)            \
    NVSHMEMI_HOSTDEVICE_PREFIX int nvshmem_##TYPENAME##_broadcast( \
        nvshmem_team_t team, TYPE *dest, const TYPE *src, size_t nelem, int PE_root);
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DECL_NVSHMEM_TYPENAME_BROADCAST)
#undef DECL_NVSHMEM_TYPENAME_BROADCAST

NVSHMEMI_HOSTDEVICE_PREFIX int nvshmem_broadcastmem(nvshmem_team_t team, void *dest,
                                                    const void *src, size_t nelems, int PE_root);

// fcollect collective
#define DECL_NVSHMEM_TYPENAME_FCOLLECT(TYPENAME, TYPE)                                            \
    NVSHMEMI_HOSTDEVICE_PREFIX int nvshmem_##TYPENAME##_fcollect(nvshmem_team_t team, TYPE *dest, \
                                                                 const TYPE *src, size_t nelem);
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DECL_NVSHMEM_TYPENAME_FCOLLECT)
#undef DECL_NVSHMEM_TYPENAME_FCOLLECT

NVSHMEMI_HOSTDEVICE_PREFIX int nvshmem_fcollectmem(nvshmem_team_t team, void *dest, const void *src,
                                                   size_t nelems);
// reduction collectives
#define NVSHMEMI_DECL_TEAM_REDUCE(NAME, TYPE, OP)                  \
    NVSHMEMI_HOSTDEVICE_PREFIX int nvshmem_##NAME##_##OP##_reduce( \
        nvshmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce);

NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_DECL_TEAM_REDUCE, and)
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_DECL_TEAM_REDUCE, or)
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_DECL_TEAM_REDUCE, xor)

NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_DECL_TEAM_REDUCE, max)
NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_DECL_TEAM_REDUCE, min)

NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(NVSHMEMI_DECL_TEAM_REDUCE, sum)
NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(NVSHMEMI_DECL_TEAM_REDUCE, prod)

#undef NVSHMEMI_DECL_TEAM_REDUCE

// reducescatter collectives
#define NVSHMEMI_DECL_TEAM_REDUCESCATTER(NAME, TYPE, OP)                  \
    NVSHMEMI_HOSTDEVICE_PREFIX int nvshmem_##NAME##_##OP##_reducescatter( \
        nvshmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce);

NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_DECL_TEAM_REDUCESCATTER, and)
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_DECL_TEAM_REDUCESCATTER, or)
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_DECL_TEAM_REDUCESCATTER, xor)

NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_DECL_TEAM_REDUCESCATTER, max)
NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_DECL_TEAM_REDUCESCATTER, min)

NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(NVSHMEMI_DECL_TEAM_REDUCESCATTER, sum)
NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(NVSHMEMI_DECL_TEAM_REDUCESCATTER, prod)

#undef NVSHMEMI_DECL_TEAM_REDUCESCATTER

#ifdef __cplusplus
}
#endif

#endif /* NVSHMEM_COLL_H */
