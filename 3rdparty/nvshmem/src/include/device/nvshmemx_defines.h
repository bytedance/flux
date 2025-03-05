/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See COPYRIGHT.txt for license information
 */

#ifndef _NVSHMEMX_DEFINES_H_
#define _NVSHMEMX_DEFINES_H_

#include <cuda_runtime.h>
#include "device_host/nvshmem_common.cuh"
#include "non_abi/device/common/nvshmemi_common_device.cuh"
#include "non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh"
#include "device/nvshmemx_collective_launch_apis.h"

#ifdef __CUDA_ARCH__
#ifdef __cplusplus
extern "C" {
#endif

__device__ inline void nvshmemx_vendor_get_version_info(int *major, int *minor, int *patch) {
    *major = NVSHMEM_VENDOR_MAJOR_VERSION;
    *minor = NVSHMEM_VENDOR_MINOR_VERSION;
    *patch = NVSHMEM_VENDOR_PATCH_VERSION;
}

__device__ inline void nvshmemx_signal_op(uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) {
    nvshmemi_signal_op(sig_addr, signal, sig_op, pe);
}

__device__ inline void *nvshmemx_mc_ptr(nvshmem_team_t team, const void *ptr) {
    return nvshmemi_mc_ptr(nvshmemi_device_state_d.team_pool[team], ptr);
}

#define NVSHMEM_TYPE_PUT_THREADGROUP(Name, Type, Group)                                         \
    __device__ inline void nvshmemx_##Name##_put_##Group(Type *dest, const Type *source,        \
                                                         size_t nelems, int pe) {               \
        nvshmemi_put_threadgroup<Type, nvshmemi_threadgroup_##Group>(dest, source, nelems, pe); \
    }

#define DEFINE_NVSHMEM_TYPE_PUT_THREADGROUP(Name, Type) \
    NVSHMEM_TYPE_PUT_THREADGROUP(Name, Type, warp)      \
    NVSHMEM_TYPE_PUT_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFINE_NVSHMEM_TYPE_PUT_THREADGROUP)
#undef DEFINE_NVSHMEM_TYPE_PUT_THREADGROUP
#ifdef __cplusplus
}
#endif

template <typename T>
__device__ inline void nvshmemi_signal(T *dest, const T value, int pe) {
    const void *peer_base_addr =
        (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);
    if (peer_base_addr != NULL) {
        volatile T *dest_actual =
            (volatile T *)((char *)(peer_base_addr) +
                           ((char *)dest - (char *)(nvshmemi_device_state_d.heap_base)));
        *dest_actual = value;
    } else {
        nvshmemi_transfer_amo_nonfetch<T>((void *)dest, value, pe, NVSHMEMI_AMO_SIGNAL);
    }
}

#define NVSHMEMI_TYPENAME_PUT_SIGNAL_SCOPE(SCOPE, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)        \
    __device__ inline void nvshmemi_##TYPENAME##_put_signal##SC_SUFFIX(                        \
        TYPE *dest, const TYPE *source, size_t nelems, uint64_t *sig_addr, uint64_t signal,    \
        int sig_op, int pe, bool is_nbi) {                                                     \
        NVSHMEMI_DECL_THREAD_IDX##SC_SUFFIX();                                                 \
        void *peer_base_addr = (void *)__ldg(                                                  \
            (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);      \
        if (peer_base_addr) {                                                                  \
            nvshmemx_##TYPENAME##_put##SC_SUFFIX(dest, source, nelems, pe);                    \
            if (myIdx == 0) {                                                                  \
                __threadfence_system();                                                        \
                nvshmemx_signal_op(sig_addr, signal, sig_op, pe);                              \
            }                                                                                  \
            NVSHMEMI_SYNC##SC_SUFFIX();                                                        \
        } else {                                                                               \
            NVSHMEMI_SYNC##SC_SUFFIX();                                                        \
            nvshmemi_transfer_put_signal<nvshmemi_threadgroup_##SCOPE>(                        \
                (void *)dest, (void *)source, nelems * sizeof(TYPE), (void *)sig_addr, signal, \
                (nvshmemi_amo_t)sig_op, pe, is_nbi);                                           \
            NVSHMEMI_SYNC##SC_SUFFIX();                                                        \
        }                                                                                      \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(NVSHMEMI_TYPENAME_PUT_SIGNAL_SCOPE, warp, _warp, x)
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(NVSHMEMI_TYPENAME_PUT_SIGNAL_SCOPE, block, _block,
                                                 x)
#undef NVSHMEMI_TYPENAME_PUT_SIGNAL_SCOPE
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __CUDA_ARCH__

/* __device__ nvshmem_<typename>_put_signal_scope */
#define NVSHMEMI_TYPENAME_PUT_SIGNAL_SCOPE_IMPL(SCOPE, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE) \
    __device__ inline void nvshmemx_##TYPENAME##_put_signal##SC_SUFFIX(                      \
        TYPE *dest, const TYPE *source, size_t nelems, uint64_t *sig_addr, uint64_t signal,  \
        int sig_op, int pe) {                                                                \
        nvshmemi_put_signal_threadgroup<TYPE, nvshmemi_threadgroup_##SCOPE>(                 \
            dest, source, nelems, sig_addr, signal, sig_op, pe, 0);                          \
    }
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(NVSHMEMI_TYPENAME_PUT_SIGNAL_SCOPE_IMPL, warp,
                                                 _warp, x)
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(NVSHMEMI_TYPENAME_PUT_SIGNAL_SCOPE_IMPL, block,
                                                 _block, x)
#undef NVSHMEMI_TYPENAME_PUT_SIGNAL_SCOPE_IMPL

/* __device__ nvshmem_putmem_signal_scope */
#define NVSHMEMI_PUTMEM_SIGNAL_SCOPE_IMPL(SCOPE, SC_SUFFIX, SC_PREFIX)                      \
    __device__ inline void nvshmemx_putmem_signal##SC_SUFFIX(                               \
        void *dest, const void *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, \
        int sig_op, int pe) {                                                               \
        nvshmemi_put_signal_threadgroup<char, nvshmemi_threadgroup_##SCOPE>(                \
            (char *)dest, (const char *)source, nelems, sig_addr, signal, sig_op, pe, 0);   \
    }

NVSHMEMI_PUTMEM_SIGNAL_SCOPE_IMPL(warp, _warp, x)
NVSHMEMI_PUTMEM_SIGNAL_SCOPE_IMPL(block, _block, x)
#undef NVSHMEMI_PUTMEM_SIGNAL_SCOPE_IMPL

/* __device__ nvshmem_putsize_signal_scope */
#define NVSHMEMI_PUTSIZE_SIGNAL_SCOPE_IMPL(SCOPE, SC_SUFFIX, SC_PREFIX, BITS)                 \
    __device__ inline void nvshmemx_put##BITS##_signal##SC_SUFFIX(                            \
        void *dest, const void *source, size_t nelems, uint64_t *sig_addr, uint64_t signal,   \
        int sig_op, int pe) {                                                                 \
        nvshmemx_putmem_signal##SC_SUFFIX(dest, source, nelems *(BITS / 8), sig_addr, signal, \
                                          sig_op, pe);                                        \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_SCOPE2(NVSHMEMI_PUTSIZE_SIGNAL_SCOPE_IMPL, warp, _warp, x)
NVSHMEMI_REPT_FOR_SIZES_WITH_SCOPE2(NVSHMEMI_PUTSIZE_SIGNAL_SCOPE_IMPL, block, _block, x)
#undef NVSHMEMI_REPT_PUTSIZE_SIGNAL_FOR_SCOPE

/* __device__ nvshmem_<typename>_put_signal_nbi_scope */
#define NVSHMEMI_TYPENAME_PUT_SIGNAL_NBI_SCOPE_IMPL(SCOPE, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE) \
    __device__ inline void nvshmemx_##TYPENAME##_put_signal_nbi##SC_SUFFIX(                      \
        TYPE *dest, const TYPE *source, size_t nelems, uint64_t *sig_addr, uint64_t signal,      \
        int sig_op, int pe) {                                                                    \
        nvshmemi_put_signal_threadgroup<TYPE, nvshmemi_threadgroup_##SCOPE>(                     \
            dest, source, nelems, sig_addr, signal, sig_op, pe, 1);                              \
    }
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(NVSHMEMI_TYPENAME_PUT_SIGNAL_NBI_SCOPE_IMPL, warp,
                                                 _warp, x)
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(NVSHMEMI_TYPENAME_PUT_SIGNAL_NBI_SCOPE_IMPL, block,
                                                 _block, x)
#undef NVSHMEMI_TYPENAME_PUT_SIGNAL_NBI_SCOPE_IMPL

/* __device__ nvshmem_putmem_signal_nbi_scope */
#define NVSHMEMI_PUTMEM_SIGNAL_NBI_SCOPE_IMPL(SCOPE, SC_SUFFIX, SC_PREFIX)                  \
    __device__ inline void nvshmemx_putmem_signal_nbi##SC_SUFFIX(                           \
        void *dest, const void *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, \
        int sig_op, int pe) {                                                               \
        nvshmemi_put_signal_threadgroup<char, nvshmemi_threadgroup_##SCOPE>(                \
            (char *)dest, (const char *)source, nelems, sig_addr, signal, sig_op, pe, 1);   \
    }

NVSHMEMI_PUTMEM_SIGNAL_NBI_SCOPE_IMPL(warp, _warp, x)
NVSHMEMI_PUTMEM_SIGNAL_NBI_SCOPE_IMPL(block, _block, x)
#undef NVSHMEMI_PUTMEM_SIGNAL_NBI_SCOPE_IMPL

#define NVSHMEMI_PUTSIZE_SIGNAL_NBI_SCOPE_IMPL(SCOPE, SC_SUFFIX, SC_PREFIX, BITS)             \
    __device__ inline void nvshmemx_put##BITS##_signal_nbi##SC_SUFFIX(                        \
        void *dest, const void *source, size_t nelems, uint64_t *sig_addr, uint64_t signal,   \
        int sig_op, int pe) {                                                                 \
        nvshmemx_putmem_signal##SC_SUFFIX(dest, source, nelems *(BITS / 8), sig_addr, signal, \
                                          sig_op, pe);                                        \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_SCOPE2(NVSHMEMI_PUTSIZE_SIGNAL_NBI_SCOPE_IMPL, warp, _warp, x)
NVSHMEMI_REPT_FOR_SIZES_WITH_SCOPE2(NVSHMEMI_PUTSIZE_SIGNAL_NBI_SCOPE_IMPL, block, _block, x)
#undef NVSHMEMI_REPT_PUTSIZE_SIGNAL_NBI_FOR_SCOPE

#define NVSHMEM_TYPE_GET_THREADGROUP(Name, Type, Group)                                         \
    __device__ inline void nvshmemx_##Name##_get_##Group(Type *dest, const Type *source,        \
                                                         size_t nelems, int pe) {               \
        nvshmemi_get_threadgroup<Type, nvshmemi_threadgroup_##Group>(dest, source, nelems, pe); \
    }

#define DEFINE_NVSHMEM_TYPE_GET(Name, Type)        \
    NVSHMEM_TYPE_GET_THREADGROUP(Name, Type, warp) \
    NVSHMEM_TYPE_GET_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFINE_NVSHMEM_TYPE_GET)
#undef DEFINE_NVSHMEM_TYPE_GET

#define NVSHMEM_PUTSIZE_THREADGROUP(Name, Type, Group)                                  \
    __device__ inline void nvshmemx_put##Name##_##Group(void *dest, const void *source, \
                                                        size_t nelems, int pe) {        \
        nvshmemi_put_threadgroup<Type, nvshmemi_threadgroup_##Group>(                   \
            (Type *)dest, (const Type *)source, nelems, pe);                            \
    }

#define DEFINE_NVSHMEM_PUTSIZE_THREADGROUP(Name, Type) \
    NVSHMEM_PUTSIZE_THREADGROUP(Name, Type, warp)      \
    NVSHMEM_PUTSIZE_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(DEFINE_NVSHMEM_PUTSIZE_THREADGROUP)
#undef DEFINE_NVSHMEM_PUTSIZE_THREADGROUP

#define NVSHMEM_GETSIZE_THREADGROUP(Name, Type, Group)                                  \
    __device__ inline void nvshmemx_get##Name##_##Group(void *dest, const void *source, \
                                                        size_t nelems, int pe) {        \
        nvshmemi_get_threadgroup<Type, nvshmemi_threadgroup_##Group>(                   \
            (Type *)dest, (const Type *)source, nelems, pe);                            \
    }

#define DEFINE_NVSHMEM_GETSIZE_THREADGROUP(Name, Type) \
    NVSHMEM_GETSIZE_THREADGROUP(Name, Type, warp)      \
    NVSHMEM_GETSIZE_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(DEFINE_NVSHMEM_GETSIZE_THREADGROUP)
#undef DEFINE_NVSHMEM_GETSIZE_THREADGROUP

#define DEFINE_NVSHMEM_PUTMEM_THREADGROUP(Group)                                                 \
    __device__ inline void nvshmemx_putmem_##Group(void *dest, const void *source, size_t bytes, \
                                                   int pe) {                                     \
        nvshmemi_put_threadgroup<char, nvshmemi_threadgroup_##Group>(                            \
            (char *)dest, (const char *)source, bytes, pe);                                      \
    }

DEFINE_NVSHMEM_PUTMEM_THREADGROUP(warp)
DEFINE_NVSHMEM_PUTMEM_THREADGROUP(block)

#define DEFINE_NVSHMEM_GETMEM_THREADGROUP(Group)                                                 \
    __device__ inline void nvshmemx_getmem_##Group(void *dest, const void *source, size_t bytes, \
                                                   int pe) {                                     \
        nvshmemi_get_threadgroup<char, nvshmemi_threadgroup_##Group>(                            \
            (char *)dest, (const char *)source, bytes, pe);                                      \
    }

DEFINE_NVSHMEM_GETMEM_THREADGROUP(warp)
DEFINE_NVSHMEM_GETMEM_THREADGROUP(block)

#define NVSHMEM_TYPE_PUT_NBI_THREADGROUP(Name, Type, Group)                                    \
    __device__ inline void nvshmemx_##Name##_put_nbi_##Group(Type *dest, const Type *source,   \
                                                             size_t nelems, int pe) {          \
        nvshmemi_put_nbi_threadgroup<Type, nvshmemi_threadgroup_##Group>(dest, source, nelems, \
                                                                         pe);                  \
    }

#define DEFINE_NVSHMEM_TYPE_PUT_NBI_THREADGROUP(Name, Type) \
    NVSHMEM_TYPE_PUT_NBI_THREADGROUP(Name, Type, warp)      \
    NVSHMEM_TYPE_PUT_NBI_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFINE_NVSHMEM_TYPE_PUT_NBI_THREADGROUP)
#undef DEFINE_NVSHMEM_TYPE_PUT_NBI_THREADGROUP

#define NVSHMEM_TYPE_GET_NBI_THREADGROUP(Name, Type, Group)                                    \
    __device__ inline void nvshmemx_##Name##_get_nbi_##Group(Type *dest, const Type *source,   \
                                                             size_t nelems, int pe) {          \
        nvshmemi_get_nbi_threadgroup<Type, nvshmemi_threadgroup_##Group>(dest, source, nelems, \
                                                                         pe);                  \
    }

#define DEFINE_NVSHMEM_TYPE_GET_NBI_THREADGROUP(Name, Type) \
    NVSHMEM_TYPE_GET_NBI_THREADGROUP(Name, Type, warp)      \
    NVSHMEM_TYPE_GET_NBI_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFINE_NVSHMEM_TYPE_GET_NBI_THREADGROUP)
#undef DEFINE_NVSHMEM_TYPE_GET_NBI_THREADGROUP

#define NVSHMEM_PUTSIZE_NBI_THREADGROUP(Name, Type, Group)                                  \
    __device__ inline void nvshmemx_put##Name##_nbi_##Group(void *dest, const void *source, \
                                                            size_t nelems, int pe) {        \
        nvshmemi_put_nbi_threadgroup<Type, nvshmemi_threadgroup_##Group>(                   \
            (Type *)dest, (const Type *)source, nelems, pe);                                \
    }

#define DEFINE_NVSHMEM_PUTSIZE_NBI_THREADGROUP(Name, Type) \
    NVSHMEM_PUTSIZE_NBI_THREADGROUP(Name, Type, warp)      \
    NVSHMEM_PUTSIZE_NBI_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(DEFINE_NVSHMEM_PUTSIZE_NBI_THREADGROUP)
#undef DEFINE_NVSHMEM_PUTSIZE_NBI_THREADGROUP

#define NVSHMEM_GETSIZE_NBI_THREADGROUP(Name, Type, Group)                                  \
    __device__ inline void nvshmemx_get##Name##_nbi_##Group(void *dest, const void *source, \
                                                            size_t nelems, int pe) {        \
        nvshmemi_get_nbi_threadgroup<Type, nvshmemi_threadgroup_##Group>(                   \
            (Type *)dest, (const Type *)source, nelems, pe);                                \
    }

#define DEFINE_NVSHMEM_GETSIZE_NBI_THREADGROUP(Name, Type) \
    NVSHMEM_GETSIZE_NBI_THREADGROUP(Name, Type, warp)      \
    NVSHMEM_GETSIZE_NBI_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(DEFINE_NVSHMEM_GETSIZE_NBI_THREADGROUP)
#undef DEFINE_NVSHMEM_GETSIZE_NBI_THREADGROUP

#define DEFINE_NVSHMEM_PUTMEM_NBI_THREADGROUP(Group)                                   \
    __device__ inline void nvshmemx_putmem_nbi_##Group(void *dest, const void *source, \
                                                       size_t bytes, int pe) {         \
        nvshmemi_put_nbi_threadgroup<char, nvshmemi_threadgroup_##Group>(              \
            (char *)dest, (const char *)source, bytes, pe);                            \
    }

DEFINE_NVSHMEM_PUTMEM_NBI_THREADGROUP(warp)
DEFINE_NVSHMEM_PUTMEM_NBI_THREADGROUP(block)

#define DEFINE_NVSHMEM_GETMEM_NBI_THREADGROUP(Group)                                   \
    __device__ inline void nvshmemx_getmem_nbi_##Group(void *dest, const void *source, \
                                                       size_t bytes, int pe) {         \
        nvshmemi_get_nbi_threadgroup<char, nvshmemi_threadgroup_##Group>(              \
            (char *)dest, (const char *)source, bytes, pe);                            \
    }

DEFINE_NVSHMEM_GETMEM_NBI_THREADGROUP(warp)
DEFINE_NVSHMEM_GETMEM_NBI_THREADGROUP(block)

#define NVSHMEM_TYPE_IPUT_THREADGROUP(Name, Type, Group)                                          \
    __device__ inline void nvshmemx_##Name##_iput_##Group(                                        \
        Type *dest, const Type *source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe) {    \
        NVSHMEMI_SYNC_##Group();                                                                  \
        void *peer_base_addr = (void *)__ldg(                                                     \
            (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);         \
        if (peer_base_addr) {                                                                     \
            NVSHMEMI_DECL_THREAD_IDX_##Group();                                                   \
            NVSHMEMI_DECL_THREADGROUP_SIZE_##Group();                                             \
            Type *dest_actual;                                                                    \
            dest_actual = (Type *)((char *)(peer_base_addr) +                                     \
                                   ((char *)dest - (char *)(nvshmemi_device_state_d.heap_base))); \
            int i;                                                                                \
            for (i = myIdx; i < nelems; i += groupSize) {                                         \
                *(dest_actual + i * dst) = *((volatile Type *)source + i * sst);                  \
            }                                                                                     \
            NVSHMEMI_SYNC_##Group();                                                              \
        } else {                                                                                  \
            printf("nvshmemx_" #Name "_iput_" #Group                                              \
                   " not implemented over remote network transports\n");                          \
            assert(0);                                                                            \
        }                                                                                         \
    }

#define DEFINE_NVSHMEM_TYPE_IPUT_THREADGROUP(Name, Type) \
    NVSHMEM_TYPE_IPUT_THREADGROUP(Name, Type, warp)      \
    NVSHMEM_TYPE_IPUT_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFINE_NVSHMEM_TYPE_IPUT_THREADGROUP)
#undef DEFINE_NVSHMEM_TYPE_IPUT_THREADGROUP

#define NVSHMEM_IPUTSIZE_THREADGROUP(Name, Type, Group)                                           \
    __device__ inline void nvshmemx_iput##Name##_##Group(                                         \
        void *dest, const void *source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe) {    \
        NVSHMEMI_SYNC_##Group();                                                                  \
        void *peer_base_addr = (void *)__ldg(                                                     \
            (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);         \
        if (peer_base_addr) {                                                                     \
            NVSHMEMI_DECL_THREAD_IDX_##Group();                                                   \
            NVSHMEMI_DECL_THREADGROUP_SIZE_##Group();                                             \
            Type *dest_actual;                                                                    \
            dest_actual = (Type *)((char *)(peer_base_addr) +                                     \
                                   ((char *)dest - (char *)(nvshmemi_device_state_d.heap_base))); \
            int i;                                                                                \
            for (i = myIdx; i < nelems; i += groupSize) {                                         \
                *((Type *)dest_actual + i * dst) = *((Type *)source + i * sst);                   \
            }                                                                                     \
            NVSHMEMI_SYNC_##Group();                                                              \
        } else {                                                                                  \
            printf("nvshmemx_iput" #Name "_" #Group                                               \
                   " not implemented over remote network transports\n");                          \
            assert(0);                                                                            \
        }                                                                                         \
    }

#define DEFINE_NVSHMEM_IPUTSIZE_THREADGROUP(Name, Type) \
    NVSHMEM_IPUTSIZE_THREADGROUP(Name, Type, warp)      \
    NVSHMEM_IPUTSIZE_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(DEFINE_NVSHMEM_IPUTSIZE_THREADGROUP)
#undef DEFINE_NVSHMEM_IPUTSIZE_THREADGROUP

#define NVSHMEM_TYPE_IGET_THREADGROUP(Name, Type, Group)                                       \
    __device__ inline void nvshmemx_##Name##_iget_##Group(                                     \
        Type *dest, const Type *source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe) { \
        NVSHMEMI_SYNC_##Group();                                                               \
        void *peer_base_addr = (void *)__ldg(                                                  \
            (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);      \
        if (peer_base_addr) {                                                                  \
            NVSHMEMI_DECL_THREAD_IDX_##Group();                                                \
            NVSHMEMI_DECL_THREADGROUP_SIZE_##Group();                                          \
            Type *source_actual;                                                               \
            source_actual =                                                                    \
                (Type *)((char *)(peer_base_addr) +                                            \
                         ((char *)source - (char *)(nvshmemi_device_state_d.heap_base)));      \
            int i;                                                                             \
            for (i = myIdx; i < nelems; i += groupSize) {                                      \
                *(dest + i * dst) = *(source_actual + i * sst);                                \
            }                                                                                  \
            NVSHMEMI_SYNC_##Group();                                                           \
        } else {                                                                               \
            printf("nvshmemx_" #Name "_iget_" #Group                                           \
                   " not implemented over remote network transports\n");                       \
            assert(0);                                                                         \
        }                                                                                      \
    }

#define DEFINE_NVSHMEM_TYPE_IGET_THREADGROUP(Name, Type) \
    NVSHMEM_TYPE_IGET_THREADGROUP(Name, Type, warp)      \
    NVSHMEM_TYPE_IGET_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFINE_NVSHMEM_TYPE_IGET_THREADGROUP)
#undef DEFINE_NVSHMEM_TYPE_IGET_THREADGROUP

#define NVSHMEM_IGETSIZE_THREADGROUP(Name, Type, Group)                                        \
    __device__ inline void nvshmemx_iget##Name##_##Group(                                      \
        void *dest, const void *source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe) { \
        NVSHMEMI_SYNC_##Group();                                                               \
        void *peer_base_addr = (void *)__ldg(                                                  \
            (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);      \
        if (peer_base_addr) {                                                                  \
            NVSHMEMI_DECL_THREAD_IDX_##Group();                                                \
            NVSHMEMI_DECL_THREADGROUP_SIZE_##Group();                                          \
            char *source_actual;                                                               \
            source_actual = ((char *)(peer_base_addr) +                                        \
                             ((char *)source - (char *)(nvshmemi_device_state_d.heap_base)));  \
            int i;                                                                             \
            for (i = myIdx; i < nelems; i += groupSize) {                                      \
                *((Type *)dest + i * dst) = *((Type *)source_actual + i * sst);                \
            }                                                                                  \
            NVSHMEMI_SYNC_##Group();                                                           \
        } else {                                                                               \
            printf("nvshmemx_iget" #Name "_" #Group                                            \
                   " not implemented over remote network transports\n");                       \
            assert(0);                                                                         \
        }                                                                                      \
    }

#define DEFINE_NVSHMEM_IGETSIZE_THREADGROUP(Name, Type) \
    NVSHMEM_IGETSIZE_THREADGROUP(Name, Type, warp)      \
    NVSHMEM_IGETSIZE_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(DEFINE_NVSHMEM_IGETSIZE_THREADGROUP)
#undef DEFINE_NVSHMEM_IGETSIZE_THREADGROUP

#endif /* __CUDA_ARCH__ */

#ifdef __cplusplus
}
#endif
#include "non_abi/device/coll/defines.cuh"

#endif
