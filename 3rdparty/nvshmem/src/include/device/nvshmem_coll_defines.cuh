#ifndef _NVSHMEM_COLL_DEFINES_CUH_
#define _NVSHMEM_COLL_DEFINES_CUH_

#include <cuda_runtime.h>

#include "device_host/nvshmem_common.cuh"
#include "device_host/nvshmem_types.h"
#include "device/nvshmem_device_macros.h"
#include "non_abi/device/coll/defines.cuh"

#ifdef __CUDA_ARCH__

static __device__ NVSHMEMI_DEVICE_INLINE int nvshmem_alltoallmem(nvshmem_team_t team, void *dest,
                                                                 const void *source,
                                                                 size_t nelems) {
    nvshmemi_alltoall_threadgroup<char, nvshmemi_threadgroup_thread>(team, (char *)dest,
                                                                     (const char *)source, nelems);
    return 0;
}

#define DEFN_NVSHMEM_TYPENAME_ALLTOALL(TYPENAME, TYPE)                                       \
    static __device__ NVSHMEMI_DEVICE_INLINE int nvshmem_##TYPENAME##_alltoall(              \
        nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nelems) {                \
        nvshmemi_alltoall_threadgroup<TYPE, nvshmemi_threadgroup_thread>(team, dest, source, \
                                                                         nelems);            \
        return 0;                                                                            \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFN_NVSHMEM_TYPENAME_ALLTOALL)
#undef DEFN_NVSHMEM_TYPENAME_ALLTOALL

static __device__ NVSHMEMI_DEVICE_INLINE int nvshmem_barrier(nvshmem_team_t team) {
    nvshmemi_barrier_threadgroup<nvshmemi_threadgroup_thread>(team);
    return 0;
}

static __device__ NVSHMEMI_DEVICE_INLINE void nvshmem_barrier_all() {
    nvshmemi_barrier_threadgroup<nvshmemi_threadgroup_thread>(NVSHMEM_TEAM_WORLD);
}

static __device__ NVSHMEMI_DEVICE_INLINE int nvshmem_team_sync(nvshmem_team_t team) {
    nvshmemi_sync_threadgroup<nvshmemi_threadgroup_thread>(team);
    return 0;
}

static __device__ NVSHMEMI_DEVICE_INLINE void nvshmem_sync_all() {
    nvshmemi_sync_threadgroup<nvshmemi_threadgroup_thread>(NVSHMEM_TEAM_WORLD);
}

#define DEFN_NVSHMEM_TYPENAME_BROADCAST(TYPENAME, TYPE)                                       \
    static __device__ NVSHMEMI_DEVICE_INLINE int nvshmem_##TYPENAME##_broadcast(              \
        nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nelems, int PE_root) {    \
        nvshmemi_broadcast_threadgroup<TYPE, nvshmemi_threadgroup_thread>(team, dest, source, \
                                                                          nelems, PE_root);   \
        return 0;                                                                             \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFN_NVSHMEM_TYPENAME_BROADCAST)
#undef DEFN_NVSHMEM_TYPENAME_BROADCAST

#define DEFN_NVSHMEM_TYPENAME_FCOLLECT(TYPENAME, TYPE)                          \
    static __device__ NVSHMEMI_DEVICE_INLINE int nvshmem_##TYPENAME##_fcollect( \
        nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nelems) {   \
        nvshmemi_fcollect_threadgroup<TYPE, nvshmemi_threadgroup_thread>(       \
            team, dest, source, nelems * nvshmem_team_my_pe(team), nelems);     \
        return 0;                                                               \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFN_NVSHMEM_TYPENAME_FCOLLECT)
#undef DEFN_NVSHMEM_TYPENAME_FCOLLECT

#define DEFN_NVSHMEM_TYPENAME_OP_REDUCE(TYPENAME, TYPE, OP)                            \
    static __device__ NVSHMEMI_DEVICE_INLINE int nvshmem_##TYPENAME##_##OP##_reduce(   \
        nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nreduce) {         \
        nvshmemi_reduce_threadgroup<TYPE, RDXN_OPS_##OP, nvshmemi_threadgroup_thread>( \
            team, dest, source, nreduce);                                              \
        return 0;                                                                      \
    }

#define DEFN_NVSHMEM_REDUCE()                                                     \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCE, and)  \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCE, or)   \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCE, xor)  \
                                                                                  \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCE, max) \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCE, min) \
                                                                                  \
    NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCE, sum)    \
    NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCE, prod)

DEFN_NVSHMEM_REDUCE();
#undef DEFN_NVSHMEM_TYPENAME_OP_REDUCE
#undef DEFN_NVSHMEM_REDUCE

#define DEFN_NVSHMEM_TYPENAME_OP_REDUCESCATTER(TYPENAME, TYPE, OP)                            \
    static __device__ NVSHMEMI_DEVICE_INLINE int nvshmem_##TYPENAME##_##OP##_reducescatter(   \
        nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nreduce) {                \
        nvshmemi_reducescatter_threadgroup<TYPE, RDXN_OPS_##OP, nvshmemi_threadgroup_thread>( \
            team, dest, source, nreduce);                                                     \
        return 0;                                                                             \
    }

#define DEFN_NVSHMEM_REDUCESCATTER()                                                     \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCESCATTER, and)  \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCESCATTER, or)   \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCESCATTER, xor)  \
                                                                                         \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCESCATTER, max) \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCESCATTER, min) \
                                                                                         \
    NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCESCATTER, sum)    \
    NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCESCATTER, prod)

DEFN_NVSHMEM_REDUCESCATTER();
#undef DEFN_NVSHMEM_TYPENAME_OP_REDUCESCATTER
#undef DEFN_NVSHMEM_REDUCESCATTER
#endif /* __CUDA_ARCH__ */

#endif
