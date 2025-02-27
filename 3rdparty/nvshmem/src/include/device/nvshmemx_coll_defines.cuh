#ifndef _NVSHMEMX_COLL_DEFINES_CUH_
#define _NVSHMEMX_COLL_DEFINES_CUH_

#include <cuda_runtime.h>

#include "device_host/nvshmem_common.cuh"
#include "device/nvshmem_coll_defines.cuh"
#include "device/nvshmem_device_macros.h"

#ifdef __CUDA_ARCH__
#define DEFN_NVSHMEMX_TYPENAME_ALLTOALL_THREADGROUP(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)      \
    static __device__ NVSHMEMI_DEVICE_INLINE int                                                   \
        nvshmem##SC_PREFIX##_##TYPENAME##_alltoall##SC_SUFFIX(nvshmem_team_t team, TYPE *dest,     \
                                                              const TYPE *source, size_t nelems) { \
        nvshmemi_alltoall_threadgroup<TYPE, nvshmemi_threadgroup_##SC>(team, dest, source,         \
                                                                       nelems);                    \
        return 0;                                                                                  \
    }

static __device__ NVSHMEMI_DEVICE_INLINE int nvshmemx_alltoallmem_warp(nvshmem_team_t team,
                                                                       void *dest,
                                                                       const void *source,
                                                                       size_t nelems) {
    nvshmemi_alltoall_threadgroup<char, nvshmemi_threadgroup_warp>(team, (char *)dest,
                                                                   (const char *)source, nelems);
    return 0;
}

static __device__ NVSHMEMI_DEVICE_INLINE int nvshmemx_alltoallmem_block(nvshmem_team_t team,
                                                                        void *dest,
                                                                        const void *source,
                                                                        size_t nelems) {
    nvshmemi_alltoall_threadgroup<char, nvshmemi_threadgroup_block>(team, (char *)dest,
                                                                    (const char *)source, nelems);
    return 0;
}

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(DEFN_NVSHMEMX_TYPENAME_ALLTOALL_THREADGROUP, warp,
                                                 _warp, x)
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(DEFN_NVSHMEMX_TYPENAME_ALLTOALL_THREADGROUP, block,
                                                 _block, x)

#define DEFN_NVSHMEMX_BARRIER_SCOPE(SC, SC_SUFFIX, SC_PREFIX)                             \
    static __device__ NVSHMEMI_DEVICE_INLINE int nvshmem##SC_PREFIX##_barrier##SC_SUFFIX( \
        nvshmem_team_t team) {                                                            \
        nvshmemi_barrier_threadgroup<nvshmemi_threadgroup_##SC>(team);                    \
        return 0;                                                                         \
    }

DEFN_NVSHMEMX_BARRIER_SCOPE(warp, _warp, x)
DEFN_NVSHMEMX_BARRIER_SCOPE(block, _block, x)
#undef DEFN_NVSHMEMX_BARRIER_SCOPE

#define DEFN_NVSHMEMX_BARRIER_ALL_SCOPE(SC, SC_SUFFIX, SC_PREFIX)                                 \
    static __device__ NVSHMEMI_DEVICE_INLINE void nvshmem##SC_PREFIX##_barrier_all##SC_SUFFIX() { \
        nvshmemi_barrier_threadgroup<nvshmemi_threadgroup_##SC>(NVSHMEM_TEAM_WORLD);              \
    }

DEFN_NVSHMEMX_BARRIER_ALL_SCOPE(warp, _warp, x)
DEFN_NVSHMEMX_BARRIER_ALL_SCOPE(block, _block, x)
#undef DEFN_NVSHMEMX_BARRIER_ALL_SCOPE

#define DEFN_NVSHMEMX_SYNC_SCOPE(SC, SC_SUFFIX, SC_PREFIX)                                  \
    static __device__ NVSHMEMI_DEVICE_INLINE int nvshmem##SC_PREFIX##_team_sync##SC_SUFFIX( \
        nvshmem_team_t team) {                                                              \
        nvshmemi_sync_threadgroup<nvshmemi_threadgroup_##SC>(team);                         \
        return 0;                                                                           \
    }

DEFN_NVSHMEMX_SYNC_SCOPE(warp, _warp, x)
DEFN_NVSHMEMX_SYNC_SCOPE(block, _block, x)
#undef DEFN_NVSHMEMX_SYNC_SCOPE

#define DEFN_NVSHMEMX_SYNC_ALL_SCOPE(SC, SC_SUFFIX, SC_PREFIX)                                 \
    static __device__ NVSHMEMI_DEVICE_INLINE void nvshmem##SC_PREFIX##_sync_all##SC_SUFFIX() { \
        nvshmemi_sync_threadgroup<nvshmemi_threadgroup_##SC>(NVSHMEM_TEAM_WORLD);              \
    }

DEFN_NVSHMEMX_SYNC_ALL_SCOPE(warp, _warp, x)
DEFN_NVSHMEMX_SYNC_ALL_SCOPE(block, _block, x)
#undef DEFN_NVSHMEMX_SYNC_ALL_SCOPE

#define DEFN_NVSHMEMX_TYPENAME_BROADCAST_THREADGROUP(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE) \
    static __device__ NVSHMEMI_DEVICE_INLINE int                                               \
        nvshmem##SC_PREFIX##_##TYPENAME##_broadcast##SC_SUFFIX(                                \
            nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nelems, int PE_root) { \
        nvshmemi_broadcast_threadgroup<TYPE, nvshmemi_threadgroup_##SC>(team, dest, source,    \
                                                                        nelems, PE_root);      \
        return 0;                                                                              \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(DEFN_NVSHMEMX_TYPENAME_BROADCAST_THREADGROUP, warp,
                                                 _warp, x)
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(DEFN_NVSHMEMX_TYPENAME_BROADCAST_THREADGROUP,
                                                 block, _block, x)
#undef DEFN_NVSHMEMX_TYPENAME_BROADCAST_THREADGROUP

#define DEFN_NVSHMEMX_TYPENAME_FCOLLECT_THREADGROUP(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)      \
    static __device__ NVSHMEMI_DEVICE_INLINE int                                                   \
        nvshmem##SC_PREFIX##_##TYPENAME##_fcollect##SC_SUFFIX(nvshmem_team_t team, TYPE *dest,     \
                                                              const TYPE *source, size_t nelems) { \
        nvshmemi_fcollect_threadgroup<TYPE, nvshmemi_threadgroup_##SC>(                            \
            team, dest, source, nelems * nvshmem_team_my_pe(team), nelems);                        \
        return 0;                                                                                  \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(DEFN_NVSHMEMX_TYPENAME_FCOLLECT_THREADGROUP, warp,
                                                 _warp, x)
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(DEFN_NVSHMEMX_TYPENAME_FCOLLECT_THREADGROUP, block,
                                                 _block, x)
#undef DEFN_NVSHMEMX_TYPENAME_FCOLLECT_THREADGROUP

#define DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, OP) \
    static __device__ NVSHMEMI_DEVICE_INLINE int                                                   \
        nvshmem##SC_PREFIX##_##TYPENAME##_##OP##_reduce##SC_SUFFIX(                                \
            nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nreduce) {                 \
        nvshmemi_reduce_threadgroup<TYPE, RDXN_OPS_##OP, nvshmemi_threadgroup_##SC>(               \
            team, dest, source, nreduce);                                                          \
        return 0;                                                                                  \
    }

#define DEFN_NVSHMEM_REDUCE_THREADGROUP(SC, SC_SUFFIX, SC_PREFIX)                                  \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE2(                                            \
        DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP, SC, SC_SUFFIX, SC_PREFIX, and)               \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE2(                                            \
        DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP, SC, SC_SUFFIX, SC_PREFIX, or)                \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE2(                                            \
        DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP, SC, SC_SUFFIX, SC_PREFIX, xor)               \
                                                                                                   \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE2(                                           \
        DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP, SC, SC_SUFFIX, SC_PREFIX, max)               \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE2(                                           \
        DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP, SC, SC_SUFFIX, SC_PREFIX, min)               \
                                                                                                   \
    NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES_WITH_SCOPE2(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP, \
                                                     SC, SC_SUFFIX, SC_PREFIX, sum)                \
    NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES_WITH_SCOPE2(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP, \
                                                     SC, SC_SUFFIX, SC_PREFIX, prod)

DEFN_NVSHMEM_REDUCE_THREADGROUP(warp, _warp, x);
DEFN_NVSHMEM_REDUCE_THREADGROUP(block, _block, x);
#undef DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP

#define DEFN_NVSHMEMX_TYPENAME_OP_REDUCESCATTER_THREADGROUP(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, \
                                                            TYPE, OP)                           \
    static __device__ NVSHMEMI_DEVICE_INLINE int                                                \
        nvshmem##SC_PREFIX##_##TYPENAME##_##OP##_reducescatter##SC_SUFFIX(                      \
            nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nreduce) {              \
        nvshmemi_reducescatter_threadgroup<TYPE, RDXN_OPS_##OP, nvshmemi_threadgroup_##SC>(     \
            team, dest, source, nreduce);                                                       \
        return 0;                                                                               \
    }

#define DEFN_NVSHMEM_REDUCESCATTER_THREADGROUP(SC, SC_SUFFIX, SC_PREFIX)                    \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE2(                                     \
        DEFN_NVSHMEMX_TYPENAME_OP_REDUCESCATTER_THREADGROUP, SC, SC_SUFFIX, SC_PREFIX, and) \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE2(                                     \
        DEFN_NVSHMEMX_TYPENAME_OP_REDUCESCATTER_THREADGROUP, SC, SC_SUFFIX, SC_PREFIX, or)  \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE2(                                     \
        DEFN_NVSHMEMX_TYPENAME_OP_REDUCESCATTER_THREADGROUP, SC, SC_SUFFIX, SC_PREFIX, xor) \
                                                                                            \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE2(                                    \
        DEFN_NVSHMEMX_TYPENAME_OP_REDUCESCATTER_THREADGROUP, SC, SC_SUFFIX, SC_PREFIX, max) \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE2(                                    \
        DEFN_NVSHMEMX_TYPENAME_OP_REDUCESCATTER_THREADGROUP, SC, SC_SUFFIX, SC_PREFIX, min) \
                                                                                            \
    NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES_WITH_SCOPE2(                                       \
        DEFN_NVSHMEMX_TYPENAME_OP_REDUCESCATTER_THREADGROUP, SC, SC_SUFFIX, SC_PREFIX, sum) \
    NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES_WITH_SCOPE2(                                       \
        DEFN_NVSHMEMX_TYPENAME_OP_REDUCESCATTER_THREADGROUP, SC, SC_SUFFIX, SC_PREFIX, prod)

DEFN_NVSHMEM_REDUCESCATTER_THREADGROUP(warp, _warp, x);
DEFN_NVSHMEM_REDUCESCATTER_THREADGROUP(block, _block, x);
#undef DEFN_NVSHMEMX_TYPENAME_OP_REDUCESCATTER_THREADGROUP
#undef DEFN_NVSHMEM_REDUCESCATTER_THREADGROUP
#endif /* __CUDA_ARCH__ */

#endif
