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

#ifndef _NVSHMEM_DEFINES_H_
#define _NVSHMEM_DEFINES_H_

#include <cuda_runtime.h>
#if not defined __CUDACC_RTC__
#include <stdint.h>
#else
#include <cuda/std/cstdint>
#endif

#include "device_host/nvshmem_common.cuh"
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

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __CUDA_ARCH__

__device__ inline int nvshmem_team_my_pe(nvshmem_team_t team) { return nvshmemi_team_my_pe(team); }

__device__ inline int nvshmem_team_n_pes(nvshmem_team_t team) { return nvshmemi_team_n_pes(team); }

__device__ inline int nvshmem_team_translate_pe(nvshmem_team_t src_team, int src_pe,
                                                nvshmem_team_t dest_team) {
    return nvshmemi_team_translate_pe(src_team, src_pe, dest_team);
}

__device__ inline int nvshmem_my_pe(void) { return nvshmemi_device_state_d.mype; }

__device__ inline int nvshmem_n_pes(void) { return nvshmemi_device_state_d.npes; }

__device__ inline void nvshmem_info_get_name(char *name) {
    size_t i;
    const char *str = NVSHMEM_VENDOR_STRING;

    /* Copy up to NVSHMEM_MAX_NAME_LEN-1 chars, then add NULL terminator */
    for (i = 0; i < NVSHMEM_MAX_NAME_LEN - 1 && str[i] != '\0'; i++) name[i] = str[i];

    name[i] = '\0';
}

__device__ inline void nvshmem_info_get_version(int *major, int *minor) {
    *major = NVSHMEM_MAJOR_VERSION;
    *minor = NVSHMEM_MINOR_VERSION;
}

/*__device__ nvshmem_p*/
#define NVSHMEMI_TYPENAME_P_IMPL(TYPENAME, TYPE)                                          \
    __device__ inline void nvshmem_##TYPENAME##_p(TYPE *dest, const TYPE value, int pe) { \
        nvshmemi_p<TYPE>(dest, value, pe);                                                \
    }
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_TYPENAME_P_IMPL)
#undef NVSHMEMI_TYPENAME_P_IMPL

/*__device__ nvshmem_g*/
#define NVSHMEMI_TYPENAME_G_IMPL(TYPENAME, TYPE)                                \
    __device__ inline TYPE nvshmem_##TYPENAME##_g(const TYPE *source, int pe) { \
        return nvshmemi_g<TYPE>(source, pe);                                    \
    }
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_TYPENAME_G_IMPL)
#undef NVSHMEMI_TYPENAME_G_IMPL

/*__device__ nvshmem_<typename>_put*/
#define NVSHMEMI_TYPENAME_PUT_IMPL(TYPENAME, TYPE)                                                 \
    __device__ inline void nvshmem_##TYPENAME##_put(TYPE *dest, const TYPE *source, size_t nelems, \
                                                    int pe) {                                      \
        nvshmemi_put_threadgroup<TYPE, NVSHMEMI_THREADGROUP_THREAD>(dest, source, nelems, pe);     \
    }
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_TYPENAME_PUT_IMPL)
#undef NVSHMEMI_TYPENAME_PUT_IMPL

/*__device__ nvshmem_<typename>_put_signal*/
#define NVSHMEMI_TYPENAME_PUT_SIGNAL_IMPL(TYPENAME, TYPE)                                         \
    __device__ inline void nvshmem_##TYPENAME##_put_signal(TYPE *dest, const TYPE *source,        \
                                                           size_t nelems, uint64_t *sig_addr,     \
                                                           uint64_t signal, int sig_op, int pe) { \
        nvshmemi_put_signal_threadgroup<TYPE, NVSHMEMI_THREADGROUP_THREAD>(                       \
            dest, source, nelems, sig_addr, signal, sig_op, pe, 0);                               \
    }
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_TYPENAME_PUT_SIGNAL_IMPL)
#undef NVSHMEMI_TYPENAME_PUT_SIGNAL_IMPL

/*__device__ nvshmem_<typename>_get*/
#define NVSHMEMI_TYPENAME_GET_IMPL(TYPENAME, TYPE)                                                 \
    __device__ inline void nvshmem_##TYPENAME##_get(TYPE *dest, const TYPE *source, size_t nelems, \
                                                    int pe) {                                      \
        nvshmemi_get_threadgroup<TYPE, NVSHMEMI_THREADGROUP_THREAD>(dest, source, nelems, pe);     \
    }
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_TYPENAME_GET_IMPL)
#undef NVSHMEMI_TYPENAME_GET_IMPL

/*__device__ nvshmem_put<bits>*/
__device__ inline void nvshmem_put8(void *dest, const void *source, size_t nelems, int pe) {
    nvshmemi_put_threadgroup<int8_t, NVSHMEMI_THREADGROUP_THREAD>(
        (int8_t *)dest, (const int8_t *)source, nelems, pe);
}
__device__ inline void nvshmem_put16(void *dest, const void *source, size_t nelems, int pe) {
    nvshmemi_put_threadgroup<int16_t, NVSHMEMI_THREADGROUP_THREAD>(
        (int16_t *)dest, (const int16_t *)source, nelems, pe);
}
__device__ inline void nvshmem_put32(void *dest, const void *source, size_t nelems, int pe) {
    nvshmemi_put_threadgroup<int32_t, NVSHMEMI_THREADGROUP_THREAD>(
        (int32_t *)dest, (const int32_t *)source, nelems, pe);
}
__device__ inline void nvshmem_put64(void *dest, const void *source, size_t nelems, int pe) {
    nvshmemi_put_threadgroup<int64_t, NVSHMEMI_THREADGROUP_THREAD>(
        (int64_t *)dest, (const int64_t *)source, nelems, pe);
}
__device__ inline void nvshmem_put128(void *dest, const void *source, size_t nelems, int pe) {
    nvshmemi_put_threadgroup<int4, NVSHMEMI_THREADGROUP_THREAD>((int4 *)dest, (const int4 *)source,
                                                                nelems, pe);
}

/*__device__ nvshmem_putmem_signal*/
__device__ inline void nvshmem_putmem_signal(void *dest, const void *source, size_t bytes,
                                             uint64_t *sig_addr, uint64_t signal, int sig_op,
                                             int pe) {
    nvshmemi_put_signal_threadgroup<char, NVSHMEMI_THREADGROUP_THREAD>(
        (char *)dest, (const char *)source, bytes, sig_addr, signal, sig_op, pe, 0);
}

/*__device__ nvshmem_put<bits>_signal*/
#define NVSHMEMI_SIZE_PUT_SIGNAL_IMPL(BITS)                                                    \
    __device__ inline void nvshmem_put##BITS##_signal(void *dest, const void *source,          \
                                                      size_t nelems, uint64_t *sig_addr,       \
                                                      uint64_t signal, int sig_op, int pe) {   \
        nvshmem_putmem_signal(dest, source, nelems *(BITS / 8), sig_addr, signal, sig_op, pe); \
    }
NVSHMEMI_REPT_FOR_SIZES(NVSHMEMI_SIZE_PUT_SIGNAL_IMPL)

/*__device__ nvshmem_get<bits>*/
__device__ inline void nvshmem_get8(void *dest, const void *source, size_t nelems, int pe) {
    nvshmemi_get_threadgroup<int8_t, NVSHMEMI_THREADGROUP_THREAD>(
        (int8_t *)dest, (const int8_t *)source, nelems, pe);
}
__device__ inline void nvshmem_get16(void *dest, const void *source, size_t nelems, int pe) {
    nvshmemi_get_threadgroup<int16_t, NVSHMEMI_THREADGROUP_THREAD>(
        (int16_t *)dest, (const int16_t *)source, nelems, pe);
}
__device__ inline void nvshmem_get32(void *dest, const void *source, size_t nelems, int pe) {
    nvshmemi_get_threadgroup<int32_t, NVSHMEMI_THREADGROUP_THREAD>(
        (int32_t *)dest, (const int32_t *)source, nelems, pe);
}
__device__ inline void nvshmem_get64(void *dest, const void *source, size_t nelems, int pe) {
    nvshmemi_get_threadgroup<int64_t, NVSHMEMI_THREADGROUP_THREAD>(
        (int64_t *)dest, (const int64_t *)source, nelems, pe);
}
__device__ inline void nvshmem_get128(void *dest, const void *source, size_t nelems, int pe) {
    nvshmemi_get_threadgroup<int4, NVSHMEMI_THREADGROUP_THREAD>((int4 *)dest, (const int4 *)source,
                                                                nelems, pe);
}

/*__device__ nvshmem_putmem*/
__device__ inline void nvshmem_putmem(void *dest, const void *source, size_t bytes, int pe) {
    nvshmemi_put_threadgroup<char, NVSHMEMI_THREADGROUP_THREAD>((char *)dest, (const char *)source,
                                                                bytes, pe);
}

/*__device__ nvshmem_getmem*/
__device__ inline void nvshmem_getmem(void *dest, const void *source, size_t bytes, int pe) {
    nvshmemi_get_threadgroup<char, NVSHMEMI_THREADGROUP_THREAD>((char *)dest, (const char *)source,
                                                                bytes, pe);
}

/*__device__ nvshmem_<typename>_put_nbi*/
#define NVSHMEMI_TYPENAME_PUT_NBI_IMPL(TYPENAME, TYPE)                                             \
    __device__ inline void nvshmem_##TYPENAME##_put_nbi(TYPE *dest, const TYPE *source,            \
                                                        size_t nelems, int pe) {                   \
        nvshmemi_put_nbi_threadgroup<TYPE, NVSHMEMI_THREADGROUP_THREAD>(dest, source, nelems, pe); \
    }
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_TYPENAME_PUT_NBI_IMPL)
#undef NVSHMEMI_TYPENAME_PUT_NBI_IMPL

/*__device__ nvshmem_<typename>_put_signal_nbi*/
#define NVSHMEMI_TYPENAME_PUT_SIGNAL_NBI_IMPL(TYPENAME, TYPE)                               \
    __device__ inline void nvshmem_##TYPENAME##_put_signal_nbi(                             \
        TYPE *dest, const TYPE *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, \
        int sig_op, int pe) {                                                               \
        nvshmemi_put_signal_threadgroup<TYPE, NVSHMEMI_THREADGROUP_THREAD>(                 \
            dest, source, nelems, sig_addr, signal, sig_op, pe, 1);                         \
    }
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_TYPENAME_PUT_SIGNAL_NBI_IMPL)
#undef NVSHMEMI_TYPENAME_PUT_SIGNAL_NBI_IMPL

/*__device__ nvshmem_<typename>_get_nbi*/
#define NVSHMEMI_TYPENAME_GET_NBI_IMPL(TYPENAME, TYPE)                                             \
    __device__ inline void nvshmem_##TYPENAME##_get_nbi(TYPE *dest, const TYPE *source,            \
                                                        size_t nelems, int pe) {                   \
        nvshmemi_get_nbi_threadgroup<TYPE, NVSHMEMI_THREADGROUP_THREAD>(dest, source, nelems, pe); \
    }
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_TYPENAME_GET_NBI_IMPL)
#undef NVSHMEMI_TYPENAME_GET_NBI_IMPL

/*__device__ nvshmem_put<bits>_nbi*/
__device__ inline void nvshmem_put8_nbi(void *dest, const void *source, size_t nelems, int pe) {
    nvshmemi_put_nbi_threadgroup<int8_t, NVSHMEMI_THREADGROUP_THREAD>(
        (int8_t *)dest, (const int8_t *)source, nelems, pe);
}
__device__ inline void nvshmem_put16_nbi(void *dest, const void *source, size_t nelems, int pe) {
    nvshmemi_put_nbi_threadgroup<int16_t, NVSHMEMI_THREADGROUP_THREAD>(
        (int16_t *)dest, (const int16_t *)source, nelems, pe);
}
__device__ inline void nvshmem_put32_nbi(void *dest, const void *source, size_t nelems, int pe) {
    nvshmemi_put_nbi_threadgroup<int32_t, NVSHMEMI_THREADGROUP_THREAD>(
        (int32_t *)dest, (const int32_t *)source, nelems, pe);
}
__device__ inline void nvshmem_put64_nbi(void *dest, const void *source, size_t nelems, int pe) {
    nvshmemi_put_nbi_threadgroup<int64_t, NVSHMEMI_THREADGROUP_THREAD>(
        (int64_t *)dest, (const int64_t *)source, nelems, pe);
}
__device__ inline void nvshmem_put128_nbi(void *dest, const void *source, size_t nelems, int pe) {
    nvshmemi_put_nbi_threadgroup<int4, NVSHMEMI_THREADGROUP_THREAD>(
        (int4 *)dest, (const int4 *)source, nelems, pe);
}
/*__device__ nvshmem_get<bits>_nbi*/
__device__ inline void nvshmem_get8_nbi(void *dest, const void *source, size_t nelems, int pe) {
    nvshmemi_get_nbi_threadgroup<int8_t, NVSHMEMI_THREADGROUP_THREAD>(
        (int8_t *)dest, (const int8_t *)source, nelems, pe);
}
__device__ inline void nvshmem_get16_nbi(void *dest, const void *source, size_t nelems, int pe) {
    nvshmemi_get_nbi_threadgroup<int16_t, NVSHMEMI_THREADGROUP_THREAD>(
        (int16_t *)dest, (const int16_t *)source, nelems, pe);
}
__device__ inline void nvshmem_get32_nbi(void *dest, const void *source, size_t nelems, int pe) {
    nvshmemi_get_nbi_threadgroup<int32_t, NVSHMEMI_THREADGROUP_THREAD>(
        (int32_t *)dest, (const int32_t *)source, nelems, pe);
}
__device__ inline void nvshmem_get64_nbi(void *dest, const void *source, size_t nelems, int pe) {
    nvshmemi_get_nbi_threadgroup<int64_t, NVSHMEMI_THREADGROUP_THREAD>(
        (int64_t *)dest, (const int64_t *)source, nelems, pe);
}
__device__ inline void nvshmem_get128_nbi(void *dest, const void *source, size_t nelems, int pe) {
    nvshmemi_get_nbi_threadgroup<int4, NVSHMEMI_THREADGROUP_THREAD>(
        (int4 *)dest, (const int4 *)source, nelems, pe);
}
/*__device__ nvshmem_putmem_nbi*/
__device__ inline void nvshmem_putmem_nbi(void *dest, const void *source, size_t bytes, int pe) {
    nvshmemi_put_nbi_threadgroup<char, NVSHMEMI_THREADGROUP_THREAD>(
        (char *)dest, (const char *)source, bytes, pe);
}

/*__device__ nvshmem_putmem_signal_nbi*/
__device__ inline void nvshmem_putmem_signal_nbi(void *dest, const void *source, size_t bytes,
                                                 uint64_t *sig_addr, uint64_t signal, int sig_op,
                                                 int pe) {
    nvshmemi_put_signal_threadgroup<char, NVSHMEMI_THREADGROUP_THREAD>(
        (char *)dest, (const char *)source, bytes, sig_addr, signal, sig_op, pe, 1);
}

/*__device__ nvshmem_put<bits>_signal*/
#define NVSHMEMI_SIZE_PUT_SIGNAL_NBI_IMPL(BITS)                                                    \
    __device__ inline void nvshmem_put##BITS##_signal_nbi(void *dest, const void *source,          \
                                                          size_t nelems, uint64_t *sig_addr,       \
                                                          uint64_t signal, int sig_op, int pe) {   \
        nvshmem_putmem_signal_nbi(dest, source, nelems *(BITS / 8), sig_addr, signal, sig_op, pe); \
    }
NVSHMEMI_REPT_FOR_SIZES(NVSHMEMI_SIZE_PUT_SIGNAL_NBI_IMPL)

/*__device__ nvshmem_getmem_nbi*/
__device__ inline void nvshmem_getmem_nbi(void *dest, const void *source, size_t bytes, int pe) {
    nvshmemi_get_nbi_threadgroup<char, NVSHMEMI_THREADGROUP_THREAD>(
        (char *)dest, (const char *)source, bytes, pe);
}

#define NVSHMEM_TYPE_IPUT(NAME, TYPE)                                                             \
    __device__ inline void nvshmem_##NAME##_iput(TYPE *dest, const TYPE *source, ptrdiff_t dst,   \
                                                 ptrdiff_t sst, size_t nelems, int pe) {          \
        void *peer_base_addr = (void *)__ldg(                                                     \
            (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);         \
        if (peer_base_addr) {                                                                     \
            TYPE *dest_actual;                                                                    \
            dest_actual = (TYPE *)((char *)(peer_base_addr) +                                     \
                                   ((char *)dest - (char *)(nvshmemi_device_state_d.heap_base))); \
            int i;                                                                                \
            for (i = 0; i < nelems; i++) {                                                        \
                *(dest_actual + i * dst) = *(source + i * sst);                                   \
            }                                                                                     \
        } else {                                                                                  \
            printf("nvshmem_" #NAME "_iput not implemented over remote network transports\n");    \
            assert(0);                                                                            \
        }                                                                                         \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEM_TYPE_IPUT)
#undef NVSHMEM_TYPE_IPUT

#define NVSHMEM_IPUTSIZE(NAME, type)                                                          \
    __device__ inline void nvshmem_iput##NAME(void *dest, const void *source, ptrdiff_t dst,  \
                                              ptrdiff_t sst, size_t nelems, int pe) {         \
        void *peer_base_addr = (void *)__ldg(                                                 \
            (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);     \
        if (peer_base_addr) {                                                                 \
            char *dest_actual;                                                                \
            dest_actual = ((char *)(peer_base_addr) +                                         \
                           ((char *)dest - (char *)(nvshmemi_device_state_d.heap_base)));     \
            int i;                                                                            \
            for (i = 0; i < nelems; i++) {                                                    \
                *((type *)dest_actual + i * dst) = *((type *)source + i * sst);               \
            }                                                                                 \
        } else {                                                                              \
            printf("nvshmem_iput" #NAME " not implemented over remote network transports\n"); \
            assert(0);                                                                        \
        }                                                                                     \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEM_IPUTSIZE)
#undef NVSHMEM_IPUTSIZE

#define NVSHMEM_TYPE_IGET(Name, TYPE)                                                           \
    __device__ inline void nvshmem_##Name##_iget(TYPE *dest, const TYPE *source, ptrdiff_t dst, \
                                                 ptrdiff_t sst, size_t nelems, int pe) {        \
        void *peer_base_addr = (void *)__ldg(                                                   \
            (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);       \
        if (peer_base_addr) {                                                                   \
            TYPE *source_actual;                                                                \
            source_actual =                                                                     \
                (TYPE *)((char *)(peer_base_addr) +                                             \
                         ((char *)source - (char *)(nvshmemi_device_state_d.heap_base)));       \
            int i;                                                                              \
            for (i = 0; i < nelems; i++) {                                                      \
                *(dest + i * dst) = *(source_actual + i * sst);                                 \
            }                                                                                   \
        } else {                                                                                \
            printf("nvshmem_" #Name "_iget not implemented over remote network transports\n");  \
            assert(0);                                                                          \
        }                                                                                       \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEM_TYPE_IGET)
#undef NVSHMEM_TYPE_IGET

#define NVSHMEM_IGETSIZE(Name, TYPE)                                                          \
    __device__ inline void nvshmem_iget##Name(void *dest, const void *source, ptrdiff_t dst,  \
                                              ptrdiff_t sst, size_t nelems, int pe) {         \
        void *peer_base_addr = (void *)__ldg(                                                 \
            (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);     \
        if (peer_base_addr) {                                                                 \
            char *source_actual;                                                              \
            source_actual = ((char *)(peer_base_addr) +                                       \
                             ((char *)source - (char *)(nvshmemi_device_state_d.heap_base))); \
            int i;                                                                            \
            for (i = 0; i < nelems; i++) {                                                    \
                *((TYPE *)dest + i * dst) = *((TYPE *)source_actual + i * sst);               \
            }                                                                                 \
        } else {                                                                              \
            printf("nvshmem_iget" #Name " not implemented over remote network transports\n"); \
            assert(0);                                                                        \
        }                                                                                     \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEM_IGETSIZE)
#undef NVSHMEM_IGETSIZE

/**** TEST API ****/
#define NVSHMEM_TEST(Name, Type)                                                       \
    __device__ inline int nvshmem_##Name##_test(Type *ivar, int cmp, Type cmp_value) { \
        int return_value = nvshmemi_test<Type>(ivar, cmp, cmp_value);                  \
        if (return_value == 1) nvshmemi_transfer_syncapi_update_mem();                 \
        return return_value;                                                           \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEM_TEST)
#undef NVSHMEM_TEST

#define NVSHMEM_TEST_ALL(Name, Type)                                                               \
    __device__ inline int nvshmem_##Name##_test_all(Type *ivars, size_t nelems, const int *status, \
                                                    int cmp, Type cmp_value) {                     \
        bool test_set_is_empty = true;                                                             \
        for (size_t i = 0; i < nelems; i++) {                                                      \
            if (!status || status[i] == 0) {                                                       \
                if (nvshmemi_test<Type>(&ivars[i], cmp, cmp_value) == 0) return 0;                 \
                test_set_is_empty = false;                                                         \
            }                                                                                      \
        }                                                                                          \
        if (test_set_is_empty == false) nvshmemi_transfer_syncapi_update_mem();                    \
                                                                                                   \
        return 1;                                                                                  \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEM_TEST_ALL)
#undef NVSHMEM_TEST_ALL

#define NVSHMEM_TEST_ANY(Name, Type)                                              \
    __device__ inline size_t nvshmem_##Name##_test_any(                           \
        Type *ivars, size_t nelems, const int *status, int cmp, Type cmp_value) { \
        unsigned long long start_idx =                                            \
            atomicAdd(nvshmemi_device_state_d.test_wait_any_start_idx_ptr, 1);    \
        for (size_t i = 0; i < nelems; i++) {                                     \
            size_t idx = (i + (size_t)start_idx) % nelems;                        \
            if (!status || status[idx] == 0) {                                    \
                if (nvshmemi_test<Type>(&ivars[idx], cmp, cmp_value) == 1) {      \
                    nvshmemi_transfer_syncapi_update_mem();                       \
                    return idx;                                                   \
                }                                                                 \
            }                                                                     \
        }                                                                         \
                                                                                  \
        return SIZE_MAX;                                                          \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEM_TEST_ANY)
#undef NVSHMEM_TEST_ANY

#define NVSHMEM_TEST_SOME(Name, Type)                                                              \
    __device__ inline size_t nvshmem_##Name##_test_some(                                           \
        Type *ivars, size_t nelems, size_t *indices, const int *status, int cmp, Type cmp_value) { \
        size_t num_satisfied = 0;                                                                  \
        for (size_t i = 0; i < nelems; i++) {                                                      \
            if (!status || status[i] == 0) {                                                       \
                if (nvshmemi_test<Type>(&ivars[i], cmp, cmp_value) == 1) {                         \
                    indices[num_satisfied++] = i;                                                  \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        if (num_satisfied > 0) nvshmemi_transfer_syncapi_update_mem();                             \
                                                                                                   \
        return num_satisfied;                                                                      \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEM_TEST_SOME)
#undef NVSHMEM_TEST_SOME

#define NVSHMEM_TEST_ALL_VECTOR(Name, Type)                                           \
    __device__ inline int nvshmem_##Name##_test_all_vector(                           \
        Type *ivars, size_t nelems, const int *status, int cmp, Type *cmp_value) {    \
        bool test_set_is_empty = true;                                                \
        for (size_t i = 0; i < nelems; i++) {                                         \
            if (!status || status[i] == 0) {                                          \
                if (nvshmemi_test<Type>(&ivars[i], cmp, cmp_value[i]) == 0) return 0; \
                test_set_is_empty = false;                                            \
            }                                                                         \
        }                                                                             \
        if (test_set_is_empty == false) nvshmemi_transfer_syncapi_update_mem();       \
                                                                                      \
        return 1;                                                                     \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEM_TEST_ALL_VECTOR)
#undef NVSHMEM_TEST_ALL_VECTOR

#define NVSHMEM_TEST_ANY_VECTOR(Name, Type)                                        \
    __device__ inline size_t nvshmem_##Name##_test_any_vector(                     \
        Type *ivars, size_t nelems, const int *status, int cmp, Type *cmp_value) { \
        unsigned long long start_idx =                                             \
            atomicAdd(nvshmemi_device_state_d.test_wait_any_start_idx_ptr, 1);     \
        for (size_t i = 0; i < nelems; i++) {                                      \
            size_t idx = (i + (size_t)start_idx) % nelems;                         \
            if (!status || status[idx] == 0) {                                     \
                if (nvshmemi_test<Type>(&ivars[idx], cmp, cmp_value[idx]) == 1) {  \
                    nvshmemi_transfer_syncapi_update_mem();                        \
                    return idx;                                                    \
                }                                                                  \
            }                                                                      \
        }                                                                          \
                                                                                   \
        return SIZE_MAX;                                                           \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEM_TEST_ANY_VECTOR)
#undef NVSHMEM_TEST_ANY_VECTOR

#define NVSHMEM_TEST_SOME_VECTOR(Name, Type)                                                       \
    __device__ inline size_t nvshmem_##Name##_test_some_vector(Type *ivars, size_t nelems,         \
                                                               size_t *indices, const int *status, \
                                                               int cmp, Type *cmp_value) {         \
        size_t num_satisfied = 0;                                                                  \
        for (size_t i = 0; i < nelems; i++) {                                                      \
            if (!status || status[i] == 0) {                                                       \
                if (nvshmemi_test<Type>(&ivars[i], cmp, cmp_value[i]) == 1) {                      \
                    indices[num_satisfied++] = i;                                                  \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        if (num_satisfied > 0) nvshmemi_transfer_syncapi_update_mem();                             \
                                                                                                   \
        return num_satisfied;                                                                      \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEM_TEST_SOME_VECTOR)
#undef NVSHMEM_TEST_SOME_VECTOR

/**** WAIT API ****/
#define NVSHMEM_WAIT_UNTIL(Name, Type)                                                        \
    __device__ inline void nvshmem_##Name##_wait_until(Type *ivar, int cmp, Type cmp_value) { \
        nvshmemi_wait_until<Type>(ivar, cmp, cmp_value);                                      \
                                                                                              \
        nvshmemi_transfer_syncapi_update_mem();                                               \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEM_WAIT_UNTIL)
#undef NVSHMEM_WAIT_UNTIL

__device__ inline uint64_t nvshmem_signal_fetch(uint64_t *sig_addr) {
    return *((volatile uint64_t *)sig_addr);
}

__device__ inline uint64_t nvshmem_signal_wait_until(uint64_t *sig_addr, int cmp,
                                                     uint64_t cmp_val) {
    nvshmemi_wait_until<uint64_t>(sig_addr, cmp, cmp_val);
    return *sig_addr;
}

#define NVSHMEM_WAIT_UNTIL_ALL(Name, Type)                                        \
    __device__ inline void nvshmem_##Name##_wait_until_all(                       \
        Type *ivars, size_t nelems, const int *status, int cmp, Type cmp_value) { \
        bool waited = false;                                                      \
        for (size_t i = 0; i < nelems; i++) {                                     \
            if (!status || status[i] == 0) {                                      \
                waited = true;                                                    \
                nvshmemi_wait_until<Type>(&ivars[i], cmp, cmp_value);             \
            }                                                                     \
        }                                                                         \
                                                                                  \
        if (waited) nvshmemi_transfer_syncapi_update_mem();                       \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEM_WAIT_UNTIL_ALL)
#undef NVSHMEM_WAIT_UNTIL_ALL

#define NVSHMEM_WAIT_UNTIL_ANY(Name, Type)                                        \
    __device__ inline size_t nvshmem_##Name##_wait_until_any(                     \
        Type *ivars, size_t nelems, const int *status, int cmp, Type cmp_value) { \
        bool wait_set_is_empty = true;                                            \
        size_t idx;                                                               \
        if (nelems == 0) return SIZE_MAX;                                         \
        unsigned long long start_idx =                                            \
            atomicAdd(nvshmemi_device_state_d.test_wait_any_start_idx_ptr, 1);    \
                                                                                  \
        for (size_t i = 0;; i++) {                                                \
            idx = (i + (size_t)start_idx) % nelems;                               \
            if (!status || status[idx] == 0) {                                    \
                wait_set_is_empty = false;                                        \
                if (nvshmemi_test<Type>(&ivars[idx], cmp, cmp_value)) break;      \
            } else if (i >= nelems && wait_set_is_empty)                          \
                break;                                                            \
        }                                                                         \
                                                                                  \
        if (wait_set_is_empty == false) nvshmemi_transfer_syncapi_update_mem();   \
                                                                                  \
        return wait_set_is_empty ? SIZE_MAX : idx;                                \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEM_WAIT_UNTIL_ANY)
#undef NVSHMEM_WAIT_UNTIL_ANY

#define NVSHMEM_WAIT_UNTIL_SOME(Name, Type)                                                        \
    __device__ inline size_t nvshmem_##Name##_wait_until_some(                                     \
        Type *ivars, size_t nelems, size_t *indices, const int *status, int cmp, Type cmp_value) { \
        size_t i;                                                                                  \
        int num_satisfied = 0;                                                                     \
        bool wait_set_is_empty = true;                                                             \
        for (i = 0; i < nelems; i++) {                                                             \
            if (!status || status[i] == 0) {                                                       \
                wait_set_is_empty = false;                                                         \
                if (nvshmem_##Name##_test(&ivars[i], cmp, cmp_value) == 1)                         \
                    indices[num_satisfied++] = i;                                                  \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        if (wait_set_is_empty == false && num_satisfied == 0) { /* do wait_any*/                   \
            indices[num_satisfied++] =                                                             \
                nvshmem_##Name##_wait_until_any(ivars, nelems, status, cmp, cmp_value);            \
        }                                                                                          \
                                                                                                   \
        if (num_satisfied > 0) nvshmemi_transfer_syncapi_update_mem();                             \
                                                                                                   \
        return num_satisfied;                                                                      \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEM_WAIT_UNTIL_SOME)
#undef NVSHMEM_WAIT_UNTIL_SOME

#define NVSHMEM_WAIT_UNTIL_ALL_VECTOR(Name, Type)                                  \
    __device__ inline void nvshmem_##Name##_wait_until_all_vector(                 \
        Type *ivars, size_t nelems, const int *status, int cmp, Type *cmp_value) { \
        bool waited = false;                                                       \
        for (size_t i = 0; i < nelems; i++) {                                      \
            if (!status || status[i] == 0) {                                       \
                waited = true;                                                     \
                nvshmemi_wait_until<Type>(&ivars[i], cmp, cmp_value[i]);           \
            }                                                                      \
        }                                                                          \
                                                                                   \
        if (waited) nvshmemi_transfer_syncapi_update_mem();                        \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEM_WAIT_UNTIL_ALL_VECTOR)
#undef NVSHMEM_WAIT_UNTIL_ALL_VECTOR

#define NVSHMEM_WAIT_UNTIL_ANY_VECTOR(Name, Type)                                  \
    __device__ inline size_t nvshmem_##Name##_wait_until_any_vector(               \
        Type *ivars, size_t nelems, const int *status, int cmp, Type *cmp_value) { \
        bool wait_set_is_empty = true;                                             \
        size_t idx;                                                                \
        if (nelems == 0) return SIZE_MAX;                                          \
        unsigned long long start_idx =                                             \
            atomicAdd(nvshmemi_device_state_d.test_wait_any_start_idx_ptr, 1);     \
                                                                                   \
        for (size_t i = 0;; i++) {                                                 \
            idx = (i + (size_t)start_idx) % nelems;                                \
            if (!status || status[idx] == 0) {                                     \
                wait_set_is_empty = false;                                         \
                if (nvshmemi_test<Type>(&ivars[idx], cmp, cmp_value[idx])) break;  \
            } else if (i >= nelems && wait_set_is_empty)                           \
                break;                                                             \
        }                                                                          \
                                                                                   \
        if (wait_set_is_empty == false) nvshmemi_transfer_syncapi_update_mem();    \
                                                                                   \
        return wait_set_is_empty ? SIZE_MAX : idx;                                 \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEM_WAIT_UNTIL_ANY_VECTOR)
#undef NVSHMEM_WAIT_UNTIL_ANY_VECTOR

#define NVSHMEM_WAIT_UNTIL_SOME_VECTOR(Name, Type)                                             \
    __device__ inline size_t nvshmem_##Name##_wait_until_some_vector(                          \
        Type *ivars, size_t nelems, size_t *indices, const int *status, int cmp,               \
        Type *cmp_value) {                                                                     \
        size_t i;                                                                              \
        int num_satisfied = 0;                                                                 \
        bool wait_set_is_empty = true;                                                         \
        for (i = 0; i < nelems; i++) {                                                         \
            if (!status || status[i] == 0) {                                                   \
                wait_set_is_empty = false;                                                     \
                if (nvshmem_##Name##_test(&ivars[i], cmp, cmp_value[i]) == 1)                  \
                    indices[num_satisfied++] = i;                                              \
            }                                                                                  \
        }                                                                                      \
                                                                                               \
        if (wait_set_is_empty == false && num_satisfied == 0) { /* do wait_any*/               \
            indices[num_satisfied++] =                                                         \
                nvshmem_##Name##_wait_until_any_vector(ivars, nelems, status, cmp, cmp_value); \
        }                                                                                      \
                                                                                               \
        if (num_satisfied > 0) nvshmemi_transfer_syncapi_update_mem();                         \
                                                                                               \
        return num_satisfied;                                                                  \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEM_WAIT_UNTIL_SOME_VECTOR)
#undef NVSHMEM_WAIT_UNTIL_SOME_VECTOR

/* nvshmem_quiet and nvshmem_fence API */
__device__ inline void nvshmem_quiet() { nvshmemi_quiet<NVSHMEMI_THREADGROUP_THREAD>(); }

__device__ inline void nvshmem_fence() { nvshmemi_fence(); }

#define NVSHMEM_TYPE_ATOMIC_FETCH_ADD(Name, Type)                                                \
    __device__ inline Type nvshmem_##Name##_atomic_fetch_add(Type *target, Type value, int pe) { \
        void *peer_base_addr = (void *)__ldg(                                                    \
            (const unsigned long long *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);        \
        if (nvshmemi_device_state_d.job_connectivity <= NVSHMEMI_JOB_GPU_LDST_ATOMICS) {         \
            Type *target_actual =                                                                \
                (Type *)((char *)peer_base_addr +                                                \
                         ((char *)target - (char *)nvshmemi_device_state_d.heap_base));          \
                                                                                                 \
            return ((Type)atomicAdd_system(target_actual, value));                               \
        } else {                                                                                 \
            return nvshmemi_transfer_amo_fetch<Type>((void *)target, value, 0, pe,               \
                                                     NVSHMEMI_AMO_FETCH_ADD);                    \
        }                                                                                        \
    }

#define NVSHMEM_TYPE_ATOMIC_FETCH_ADD_CAST(Name, Type, subType)                                  \
    __device__ inline Type nvshmem_##Name##_atomic_fetch_add(Type *target, Type value, int pe) { \
        void *peer_base_addr = (void *)__ldg(                                                    \
            (const unsigned long long *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);        \
        if (nvshmemi_device_state_d.job_connectivity <= NVSHMEMI_JOB_GPU_LDST_ATOMICS) {         \
            Type *target_actual =                                                                \
                (Type *)((char *)peer_base_addr +                                                \
                         ((char *)target - (char *)nvshmemi_device_state_d.heap_base));          \
                                                                                                 \
            return (Type)atomicAdd_system((subType *)target_actual, *((subType *)&value));       \
        } else {                                                                                 \
            return nvshmemi_transfer_amo_fetch<Type>((void *)target, value, 0, pe,               \
                                                     NVSHMEMI_AMO_FETCH_ADD);                    \
        }                                                                                        \
    }

NVSHMEM_TYPE_ATOMIC_FETCH_ADD(int, int)
NVSHMEM_TYPE_ATOMIC_FETCH_ADD_CAST(long, long, unsigned long long int)
NVSHMEM_TYPE_ATOMIC_FETCH_ADD(uint, unsigned int)
NVSHMEM_TYPE_ATOMIC_FETCH_ADD_CAST(ulong, unsigned long, unsigned int)
NVSHMEM_TYPE_ATOMIC_FETCH_ADD(ulonglong, unsigned long long int)
NVSHMEM_TYPE_ATOMIC_FETCH_ADD_CAST(int32, int32_t, int)
NVSHMEM_TYPE_ATOMIC_FETCH_ADD_CAST(uint32, uint32_t, unsigned int)
NVSHMEM_TYPE_ATOMIC_FETCH_ADD_CAST(uint64, uint64_t, unsigned long long int)
NVSHMEM_TYPE_ATOMIC_FETCH_ADD_CAST(size, size_t, unsigned long long int)
/*the following types are not implemented for FADD becuase of lack of CUDA support
 * ptrdiff_t
 * longlong
 * int64_t
 */
#undef NVSHMEM_TYPE_ATOMIC_FETCH_ADD
#undef NVSHMEM_TYPE_ATOMIC_FETCH_ADD_CAST

#define NVSHMEM_TYPE_ATOMIC_ADD_EMULATE(Name, Type)                                        \
    __device__ inline void nvshmem_##Name##_atomic_add(Type *target, Type value, int pe) { \
        /*need a better check for case when to use only proxy-based atomics*/              \
        if (nvshmemi_device_state_d.job_connectivity <= NVSHMEMI_JOB_GPU_LDST_ATOMICS) {   \
            nvshmem_##Name##_atomic_fetch_add(target, value, pe);                          \
        } else {                                                                           \
            nvshmemi_transfer_amo_nonfetch<Type>(target, value, pe, NVSHMEMI_AMO_ADD);     \
        }                                                                                  \
    }

NVSHMEM_TYPE_ATOMIC_ADD_EMULATE(int, int)
NVSHMEM_TYPE_ATOMIC_ADD_EMULATE(long, long)
NVSHMEM_TYPE_ATOMIC_ADD_EMULATE(uint, unsigned int)
NVSHMEM_TYPE_ATOMIC_ADD_EMULATE(ulong, unsigned long)
NVSHMEM_TYPE_ATOMIC_ADD_EMULATE(ulonglong, unsigned long long)
NVSHMEM_TYPE_ATOMIC_ADD_EMULATE(int32, int32_t)
NVSHMEM_TYPE_ATOMIC_ADD_EMULATE(uint32, uint32_t)
NVSHMEM_TYPE_ATOMIC_ADD_EMULATE(uint64, uint64_t)
NVSHMEM_TYPE_ATOMIC_ADD_EMULATE(size, size_t)
/*the following types are not implemented for ADD becuase of lack of CUDA support
 * ptrdiff_t
 * longlong
 * int64_t
 */

#undef NVSHMEM_TYPE_ATOMIC_ADD_EMULATE

#define NVSHMEM_TYPE_ATOMIC_FETCH_INC(Name, Type)                                         \
    __device__ inline Type nvshmem_##Name##_atomic_fetch_inc(Type *target, int pe) {      \
        void *peer_base_addr = (void *)__ldg(                                             \
            (const unsigned long long *)nvshmemi_device_state_d.peer_heap_base_p2p + pe); \
        if (nvshmemi_device_state_d.job_connectivity <= NVSHMEMI_JOB_GPU_LDST_ATOMICS) {  \
            Type *target_actual =                                                         \
                (Type *)((char *)peer_base_addr +                                         \
                         ((char *)target - (char *)nvshmemi_device_state_d.heap_base));   \
                                                                                          \
            return atomicInc_system(target_actual, UINT_MAX);                             \
        } else {                                                                          \
            return nvshmemi_transfer_amo_fetch<Type>((void *)target, (Type)1, 0, pe,      \
                                                     NVSHMEMI_AMO_FETCH_INC);             \
        }                                                                                 \
    }

#define NVSHMEM_TYPE_ATOMIC_FETCH_INC_CAST(Name, Type, subType)                           \
    __device__ inline Type nvshmem_##Name##_atomic_fetch_inc(Type *target, int pe) {      \
        void *peer_base_addr = (void *)__ldg(                                             \
            (const unsigned long long *)nvshmemi_device_state_d.peer_heap_base_p2p + pe); \
        if (nvshmemi_device_state_d.job_connectivity <= NVSHMEMI_JOB_GPU_LDST_ATOMICS) {  \
            Type *target_actual =                                                         \
                (Type *)((char *)peer_base_addr +                                         \
                         ((char *)target - (char *)nvshmemi_device_state_d.heap_base));   \
                                                                                          \
            return (Type)atomicInc_system((subType *)target_actual, UINT_MAX);            \
        } else {                                                                          \
            return nvshmemi_transfer_amo_fetch<Type>((void *)target, (Type)1, 0, pe,      \
                                                     NVSHMEMI_AMO_FETCH_INC);             \
        }                                                                                 \
    }

#define NVSHMEM_TYPE_ATOMIC_FETCH_INC_EMULATE(Name, Type)                            \
    __device__ inline Type nvshmem_##Name##_atomic_fetch_inc(Type *target, int pe) { \
        return nvshmem_##Name##_atomic_fetch_add(target, (Type)1, pe);               \
    }

NVSHMEM_TYPE_ATOMIC_FETCH_INC_EMULATE(int, int)
NVSHMEM_TYPE_ATOMIC_FETCH_INC_EMULATE(long, long)
NVSHMEM_TYPE_ATOMIC_FETCH_INC(uint, unsigned int)
NVSHMEM_TYPE_ATOMIC_FETCH_INC_EMULATE(ulong, unsigned long)
NVSHMEM_TYPE_ATOMIC_FETCH_INC_EMULATE(ulonglong, unsigned long long)
NVSHMEM_TYPE_ATOMIC_FETCH_INC_EMULATE(int32, int32_t)
NVSHMEM_TYPE_ATOMIC_FETCH_INC_CAST(uint32, uint32_t, unsigned int)
NVSHMEM_TYPE_ATOMIC_FETCH_INC_EMULATE(uint64, uint64_t)
NVSHMEM_TYPE_ATOMIC_FETCH_INC_EMULATE(size, size_t)
/*the following types are not implemented for INC becuase of lack of CUDA support
 * ptrdiff_t
 * longlong
 * int64_t
 */

#undef NVSHMEM_TYPE_ATOMIC_FETCH_INC
#undef NVSHMEM_TYPE_ATOMIC_FETCH_INC_CAST
#undef NVSHMEM_TYPE_ATOMIC_FETCH_INC_EMULATE

#define NVSHMEM_TYPE_ATOMIC_INC_EMULATE(Name, Type)                                              \
    __device__ inline void nvshmem_##Name##_atomic_inc(Type *target, int pe) {                   \
        /*need a better check for case when to use only proxy-based atomcis*/                    \
        if (nvshmemi_device_state_d.job_connectivity <= NVSHMEMI_JOB_GPU_LDST_ATOMICS) {         \
            nvshmem_##Name##_atomic_fetch_inc(target, pe);                                       \
        } else {                                                                                 \
            nvshmemi_transfer_amo_nonfetch<Type>((void *)target, (Type)1, pe, NVSHMEMI_AMO_ADD); \
        }                                                                                        \
    }

NVSHMEM_TYPE_ATOMIC_INC_EMULATE(int, int)
NVSHMEM_TYPE_ATOMIC_INC_EMULATE(long, long)
NVSHMEM_TYPE_ATOMIC_INC_EMULATE(uint, unsigned int)
NVSHMEM_TYPE_ATOMIC_INC_EMULATE(ulong, unsigned long)
NVSHMEM_TYPE_ATOMIC_INC_EMULATE(ulonglong, unsigned long long)
NVSHMEM_TYPE_ATOMIC_INC_EMULATE(int32, int32_t)
NVSHMEM_TYPE_ATOMIC_INC_EMULATE(uint32, uint32_t)
NVSHMEM_TYPE_ATOMIC_INC_EMULATE(uint64, uint64_t)
NVSHMEM_TYPE_ATOMIC_INC_EMULATE(size, size_t)
/*the following types are not implemented for INC becuase of lack of CUDA support
 * ptrdiff_t
 * longlong
 * int64_t
 */
#undef NVSHMEM_TYPE_ATOMIC_INC_EMULATE

#define NVSHMEM_TYPE_COMPARE_SWAP(Name, Type)                                                  \
    __device__ inline Type nvshmem_##Name##_atomic_compare_swap(Type *target, Type compare,    \
                                                                Type value, int pe) {          \
        void *peer_base_addr = (void *)__ldg(                                                  \
            (const unsigned long long *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);      \
        if (nvshmemi_device_state_d.job_connectivity <= NVSHMEMI_JOB_GPU_LDST_ATOMICS) {       \
            Type *target_actual =                                                              \
                (Type *)((char *)peer_base_addr +                                              \
                         ((char *)target - (char *)nvshmemi_device_state_d.heap_base));        \
                                                                                               \
            return (Type)atomicCAS_system(target_actual, compare, value);                      \
        } else {                                                                               \
            return nvshmemi_transfer_amo_fetch<Type>((void *)target, (Type)value, compare, pe, \
                                                     NVSHMEMI_AMO_COMPARE_SWAP);               \
        }                                                                                      \
    }

#define NVSHMEM_TYPE_COMPARE_SWAP_CAST(Name, Type, subType)                                    \
    __device__ inline Type nvshmem_##Name##_atomic_compare_swap(Type *target, Type compare,    \
                                                                Type value, int pe) {          \
        void *peer_base_addr = (void *)__ldg(                                                  \
            (const unsigned long long *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);      \
        if (nvshmemi_device_state_d.job_connectivity <= NVSHMEMI_JOB_GPU_LDST_ATOMICS) {       \
            Type *target_actual =                                                              \
                (Type *)((char *)peer_base_addr +                                              \
                         ((char *)target - (char *)nvshmemi_device_state_d.heap_base));        \
                                                                                               \
            return (Type)atomicCAS_system((subType *)target_actual, *((subType *)&compare),    \
                                          *((subType *)&value));                               \
        } else {                                                                               \
            return nvshmemi_transfer_amo_fetch<Type>((void *)target, (Type)value, compare, pe, \
                                                     NVSHMEMI_AMO_COMPARE_SWAP);               \
        }                                                                                      \
    }

NVSHMEM_TYPE_COMPARE_SWAP(int, int)
NVSHMEM_TYPE_COMPARE_SWAP_CAST(long, long, unsigned long long int)
NVSHMEM_TYPE_COMPARE_SWAP_CAST(longlong, long long, unsigned long long int)
NVSHMEM_TYPE_COMPARE_SWAP(uint, unsigned int)
NVSHMEM_TYPE_COMPARE_SWAP_CAST(ulong, unsigned long, unsigned long long int)
NVSHMEM_TYPE_COMPARE_SWAP(ulonglong, unsigned long long int)
NVSHMEM_TYPE_COMPARE_SWAP_CAST(int32, int32_t, int)
NVSHMEM_TYPE_COMPARE_SWAP_CAST(int64, int64_t, unsigned long long int)
NVSHMEM_TYPE_COMPARE_SWAP_CAST(uint32, uint32_t, unsigned int)
NVSHMEM_TYPE_COMPARE_SWAP_CAST(uint64, uint64_t, unsigned long long int)
NVSHMEM_TYPE_COMPARE_SWAP_CAST(size, size_t, unsigned long long int)
NVSHMEM_TYPE_COMPARE_SWAP_CAST(ptrdiff, ptrdiff_t, unsigned long long int)

#define NVSHMEM_TYPE_FETCH_AND(Name, Type)                                                       \
    __device__ inline Type nvshmem_##Name##_atomic_fetch_and(Type *target, Type value, int pe) { \
        void *peer_base_addr = (void *)__ldg(                                                    \
            (const unsigned long long *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);        \
        if (nvshmemi_device_state_d.job_connectivity <= NVSHMEMI_JOB_GPU_LDST_ATOMICS) {         \
            Type *target_actual =                                                                \
                (Type *)((char *)peer_base_addr +                                                \
                         ((char *)target - (char *)nvshmemi_device_state_d.heap_base));          \
                                                                                                 \
            return atomicAnd_system(target_actual, value);                                       \
        } else {                                                                                 \
            return nvshmemi_transfer_amo_fetch<Type>((void *)target, (Type)value, 0, pe,         \
                                                     NVSHMEMI_AMO_FETCH_AND);                    \
        }                                                                                        \
    }

#define NVSHMEM_TYPE_FETCH_AND_CAST(Name, Type, subType)                                         \
    __device__ inline Type nvshmem_##Name##_atomic_fetch_and(Type *target, Type value, int pe) { \
        void *peer_base_addr = (void *)__ldg(                                                    \
            (const unsigned long long *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);        \
        if (nvshmemi_device_state_d.job_connectivity <= NVSHMEMI_JOB_GPU_LDST_ATOMICS) {         \
            Type *target_actual =                                                                \
                (Type *)((char *)peer_base_addr +                                                \
                         ((char *)target - (char *)nvshmemi_device_state_d.heap_base));          \
                                                                                                 \
            return atomicAnd_system((subType *)target_actual, *((subType *)&value));             \
        } else {                                                                                 \
            return nvshmemi_transfer_amo_fetch<Type>((void *)target, (Type)value, 0, pe,         \
                                                     NVSHMEMI_AMO_FETCH_AND);                    \
        }                                                                                        \
    }

NVSHMEM_TYPE_FETCH_AND(uint, unsigned int)
NVSHMEM_TYPE_FETCH_AND_CAST(ulong, unsigned long, unsigned long long int)
NVSHMEM_TYPE_FETCH_AND(ulonglong, unsigned long long)
NVSHMEM_TYPE_FETCH_AND_CAST(int32, int32_t, unsigned int)
NVSHMEM_TYPE_FETCH_AND_CAST(int64, int64_t, unsigned long long int)
NVSHMEM_TYPE_FETCH_AND_CAST(uint32, uint32_t, unsigned int)
NVSHMEM_TYPE_FETCH_AND_CAST(uint64, uint64_t, unsigned long long int)

#define NVSHMEM_TYPE_AND_EMULATE(Name, Type)                                                   \
    __device__ inline void nvshmem_##Name##_atomic_and(Type *target, Type value, int pe) {     \
        if (nvshmemi_device_state_d.job_connectivity <= NVSHMEMI_JOB_GPU_LDST_ATOMICS) {       \
            nvshmem_##Name##_atomic_fetch_and(target, (Type)value, pe);                        \
        } else {                                                                               \
            nvshmemi_transfer_amo_nonfetch<Type>((void *)target, value, pe, NVSHMEMI_AMO_AND); \
        }                                                                                      \
    }

NVSHMEM_TYPE_AND_EMULATE(uint, unsigned int)
NVSHMEM_TYPE_AND_EMULATE(ulong, unsigned long)
NVSHMEM_TYPE_AND_EMULATE(ulonglong, unsigned long long)
NVSHMEM_TYPE_AND_EMULATE(int32, int32_t)
NVSHMEM_TYPE_AND_EMULATE(int64, int64_t)
NVSHMEM_TYPE_AND_EMULATE(uint32, uint32_t)
NVSHMEM_TYPE_AND_EMULATE(uint64, uint64_t)

#define NVSHMEM_TYPE_FETCH_OR(Name, Type)                                                       \
    __device__ inline Type nvshmem_##Name##_atomic_fetch_or(Type *target, Type value, int pe) { \
        void *peer_base_addr = (void *)__ldg(                                                   \
            (const unsigned long long *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);       \
        if (nvshmemi_device_state_d.job_connectivity <= NVSHMEMI_JOB_GPU_LDST_ATOMICS) {        \
            Type *target_actual =                                                               \
                (Type *)((char *)peer_base_addr +                                               \
                         ((char *)target - (char *)nvshmemi_device_state_d.heap_base));         \
                                                                                                \
            return atomicOr_system(target_actual, value);                                       \
        } else {                                                                                \
            return nvshmemi_transfer_amo_fetch<Type>((void *)target, (Type)value, 0, pe,        \
                                                     NVSHMEMI_AMO_FETCH_OR);                    \
        }                                                                                       \
    }

#define NVSHMEM_TYPE_FETCH_OR_CAST(Name, Type, subType)                                         \
    __device__ inline Type nvshmem_##Name##_atomic_fetch_or(Type *target, Type value, int pe) { \
        void *peer_base_addr = (void *)__ldg(                                                   \
            (const unsigned long long *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);       \
        if (nvshmemi_device_state_d.job_connectivity <= NVSHMEMI_JOB_GPU_LDST_ATOMICS) {        \
            Type *target_actual =                                                               \
                (Type *)((char *)peer_base_addr +                                               \
                         ((char *)target - (char *)nvshmemi_device_state_d.heap_base));         \
                                                                                                \
            return atomicOr_system((subType *)target_actual, *((subType *)&value));             \
        } else {                                                                                \
            return nvshmemi_transfer_amo_fetch<Type>((void *)target, (Type)value, 0, pe,        \
                                                     NVSHMEMI_AMO_FETCH_OR);                    \
        }                                                                                       \
    }

NVSHMEM_TYPE_FETCH_OR(uint, unsigned int)
NVSHMEM_TYPE_FETCH_OR_CAST(ulong, unsigned long, unsigned long long int)
NVSHMEM_TYPE_FETCH_OR(ulonglong, unsigned long long)
NVSHMEM_TYPE_FETCH_OR_CAST(int32, int32_t, unsigned int)
NVSHMEM_TYPE_FETCH_OR_CAST(int64, int64_t, unsigned long long int)
NVSHMEM_TYPE_FETCH_OR_CAST(uint32, uint32_t, unsigned int)
NVSHMEM_TYPE_FETCH_OR_CAST(uint64, uint64_t, unsigned long long int)

#define NVSHMEM_TYPE_OR_EMULATE(Name, Type)                                                   \
    __device__ inline void nvshmem_##Name##_atomic_or(Type *target, Type value, int pe) {     \
        if (nvshmemi_device_state_d.job_connectivity <= NVSHMEMI_JOB_GPU_LDST_ATOMICS) {      \
            nvshmem_##Name##_atomic_fetch_or(target, (Type)value, pe);                        \
        } else {                                                                              \
            nvshmemi_transfer_amo_nonfetch<Type>((void *)target, value, pe, NVSHMEMI_AMO_OR); \
        }                                                                                     \
    }

NVSHMEM_TYPE_OR_EMULATE(uint, unsigned int)
NVSHMEM_TYPE_OR_EMULATE(ulong, unsigned long)
NVSHMEM_TYPE_OR_EMULATE(ulonglong, unsigned long long)
NVSHMEM_TYPE_OR_EMULATE(int32, int32_t)
NVSHMEM_TYPE_OR_EMULATE(int64, int64_t)
NVSHMEM_TYPE_OR_EMULATE(uint32, uint32_t)
NVSHMEM_TYPE_OR_EMULATE(uint64, uint64_t)

#define NVSHMEM_TYPE_FETCH_XOR(Name, Type)                                                       \
    __device__ inline Type nvshmem_##Name##_atomic_fetch_xor(Type *target, Type value, int pe) { \
        void *peer_base_addr = (void *)__ldg(                                                    \
            (const unsigned long long *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);        \
        if (nvshmemi_device_state_d.job_connectivity <= NVSHMEMI_JOB_GPU_LDST_ATOMICS) {         \
            Type *target_actual =                                                                \
                (Type *)((char *)peer_base_addr +                                                \
                         ((char *)target - (char *)nvshmemi_device_state_d.heap_base));          \
                                                                                                 \
            return atomicXor_system(target_actual, value);                                       \
        } else {                                                                                 \
            return nvshmemi_transfer_amo_fetch<Type>((void *)target, (Type)value, 0, pe,         \
                                                     NVSHMEMI_AMO_FETCH_XOR);                    \
        }                                                                                        \
    }

#define NVSHMEM_TYPE_FETCH_XOR_CAST(Name, Type, subType)                                         \
    __device__ inline Type nvshmem_##Name##_atomic_fetch_xor(Type *target, Type value, int pe) { \
        void *peer_base_addr = (void *)__ldg(                                                    \
            (const unsigned long long *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);        \
        if (nvshmemi_device_state_d.job_connectivity <= NVSHMEMI_JOB_GPU_LDST_ATOMICS) {         \
            Type *target_actual =                                                                \
                (Type *)((char *)peer_base_addr +                                                \
                         ((char *)target - (char *)nvshmemi_device_state_d.heap_base));          \
                                                                                                 \
            return atomicXor_system((subType *)target_actual, *((subType *)&value));             \
        } else {                                                                                 \
            return nvshmemi_transfer_amo_fetch<Type>((void *)target, (Type)value, 0, pe,         \
                                                     NVSHMEMI_AMO_FETCH_XOR);                    \
        }                                                                                        \
    }

NVSHMEM_TYPE_FETCH_XOR(uint, unsigned int)
NVSHMEM_TYPE_FETCH_XOR_CAST(ulong, unsigned long, unsigned long long int)
NVSHMEM_TYPE_FETCH_XOR(ulonglong, unsigned long long)
NVSHMEM_TYPE_FETCH_XOR_CAST(int32, int32_t, unsigned int)
NVSHMEM_TYPE_FETCH_XOR_CAST(int64, int64_t, unsigned long long int)
NVSHMEM_TYPE_FETCH_XOR_CAST(uint32, uint32_t, unsigned int)
NVSHMEM_TYPE_FETCH_XOR_CAST(uint64, uint64_t, unsigned long long int)

#define NVSHMEM_TYPE_XOR_EMULATE(Name, Type)                                                   \
    __device__ inline void nvshmem_##Name##_atomic_xor(Type *target, Type value, int pe) {     \
        if (nvshmemi_device_state_d.job_connectivity <= NVSHMEMI_JOB_GPU_LDST_ATOMICS) {       \
            nvshmem_##Name##_atomic_fetch_xor(target, (Type)value, pe);                        \
        } else {                                                                               \
            nvshmemi_transfer_amo_nonfetch<Type>((void *)target, value, pe, NVSHMEMI_AMO_XOR); \
        }                                                                                      \
    }

NVSHMEM_TYPE_XOR_EMULATE(uint, unsigned int)
NVSHMEM_TYPE_XOR_EMULATE(ulong, unsigned long)
NVSHMEM_TYPE_XOR_EMULATE(ulonglong, unsigned long long)
NVSHMEM_TYPE_XOR_EMULATE(int32, int32_t)
NVSHMEM_TYPE_XOR_EMULATE(int64, int64_t)
NVSHMEM_TYPE_XOR_EMULATE(uint32, uint32_t)
NVSHMEM_TYPE_XOR_EMULATE(uint64, uint64_t)

#define NVSHMEM_TYPE_SWAP(Name, Type)                                                       \
    __device__ inline Type nvshmem_##Name##_atomic_swap(Type *target, Type value, int pe) { \
        void *peer_base_addr = (void *)__ldg(                                               \
            (const unsigned long long *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);   \
        if (nvshmemi_device_state_d.job_connectivity <= NVSHMEMI_JOB_GPU_LDST_ATOMICS) {    \
            Type *target_actual =                                                           \
                (Type *)((char *)peer_base_addr +                                           \
                         ((char *)target - (char *)nvshmemi_device_state_d.heap_base));     \
                                                                                            \
            return (Type)atomicExch_system(target_actual, value);                           \
        } else {                                                                            \
            return nvshmemi_transfer_amo_fetch<Type>((void *)target, (Type)value, 0, pe,    \
                                                     NVSHMEMI_AMO_SWAP);                    \
        }                                                                                   \
    }

#define NVSHMEM_TYPE_SWAP_CAST(Name, Type, subType)                                         \
    __device__ inline Type nvshmem_##Name##_atomic_swap(Type *target, Type value, int pe) { \
        void *peer_base_addr = (void *)__ldg(                                               \
            (const unsigned long long *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);   \
        if (nvshmemi_device_state_d.job_connectivity <= NVSHMEMI_JOB_GPU_LDST_ATOMICS) {    \
            Type *target_actual =                                                           \
                (Type *)((char *)peer_base_addr +                                           \
                         ((char *)target - (char *)nvshmemi_device_state_d.heap_base));     \
                                                                                            \
            return (Type)atomicExch_system((subType *)target_actual, *((subType *)&value)); \
        } else {                                                                            \
            return nvshmemi_transfer_amo_fetch<Type>((void *)target, (Type)value, 0, pe,    \
                                                     NVSHMEMI_AMO_SWAP);                    \
        }                                                                                   \
    }

NVSHMEM_TYPE_SWAP(int, int)
NVSHMEM_TYPE_SWAP_CAST(long, long, int)
NVSHMEM_TYPE_SWAP_CAST(longlong, long long, unsigned long long int)
NVSHMEM_TYPE_SWAP(uint, unsigned int)
NVSHMEM_TYPE_SWAP_CAST(ulong, unsigned long, unsigned long long int)
NVSHMEM_TYPE_SWAP(ulonglong, unsigned long long)
NVSHMEM_TYPE_SWAP_CAST(int32, int32_t, unsigned int)
NVSHMEM_TYPE_SWAP_CAST(int64, int64_t, unsigned long long int)
NVSHMEM_TYPE_SWAP_CAST(uint32, uint32_t, unsigned int)
NVSHMEM_TYPE_SWAP_CAST(uint64, uint64_t, unsigned long long int)
NVSHMEM_TYPE_SWAP(float, float)
NVSHMEM_TYPE_SWAP_CAST(double, double, unsigned long long)
NVSHMEM_TYPE_SWAP_CAST(size, size_t, unsigned long long int)
NVSHMEM_TYPE_SWAP_CAST(ptrdiff, ptrdiff_t, unsigned long long int)

#define NVSHMEM_TYPE_FETCH_EMULATE(Name, Type)                                            \
    __device__ inline Type nvshmem_##Name##_atomic_fetch(const Type *target, int pe) {    \
        return nvshmem_##Name##_atomic_fetch_or(const_cast<Type *>(target), (Type)0, pe); \
    }

#define NVSHMEM_TYPE_FETCH_EMULATE_CAST(Name, Type, subName, subType)                  \
    __device__ inline Type nvshmem_##Name##_atomic_fetch(const Type *target, int pe) { \
        subType temp = nvshmem_##subName##_atomic_fetch_or(                            \
            const_cast<subType *>((const subType *)target), (subType)0, pe);           \
        return *((Type *)&temp);                                                       \
    }

NVSHMEM_TYPE_FETCH_EMULATE_CAST(int, int, uint, unsigned int)
NVSHMEM_TYPE_FETCH_EMULATE_CAST(long, long, ulonglong, unsigned long long int)
NVSHMEM_TYPE_FETCH_EMULATE_CAST(longlong, long long, ulonglong, unsigned long long int)
NVSHMEM_TYPE_FETCH_EMULATE(uint, unsigned int)
NVSHMEM_TYPE_FETCH_EMULATE(ulong, unsigned long)
NVSHMEM_TYPE_FETCH_EMULATE(ulonglong, unsigned long long)
NVSHMEM_TYPE_FETCH_EMULATE(int32, int32_t)
NVSHMEM_TYPE_FETCH_EMULATE(int64, int64_t)
NVSHMEM_TYPE_FETCH_EMULATE(uint32, uint32_t)
NVSHMEM_TYPE_FETCH_EMULATE(uint64, uint64_t)
NVSHMEM_TYPE_FETCH_EMULATE_CAST(float, float, uint, unsigned int)
NVSHMEM_TYPE_FETCH_EMULATE_CAST(double, double, ulonglong, unsigned long long int)
NVSHMEM_TYPE_FETCH_EMULATE_CAST(size, size_t, ulonglong, unsigned long long int)
NVSHMEM_TYPE_FETCH_EMULATE_CAST(ptrdiff, ptrdiff_t, ulonglong, unsigned long long int)

#define NVSHMEM_TYPE_SET_EMULATE(Name, Type)                                                   \
    __device__ inline void nvshmem_##Name##_atomic_set(Type *target, Type value, int pe) {     \
        if (nvshmemi_device_state_d.job_connectivity <= NVSHMEMI_JOB_GPU_LDST_ATOMICS) {       \
            nvshmem_##Name##_atomic_swap(target, value, pe);                                   \
        } else {                                                                               \
            nvshmemi_transfer_amo_nonfetch<Type>((void *)target, value, pe, NVSHMEMI_AMO_SET); \
        }                                                                                      \
    }

NVSHMEM_TYPE_SET_EMULATE(int, int)
NVSHMEM_TYPE_SET_EMULATE(long, long)
NVSHMEM_TYPE_SET_EMULATE(longlong, long long)
NVSHMEM_TYPE_SET_EMULATE(uint, unsigned int)
NVSHMEM_TYPE_SET_EMULATE(ulong, unsigned long)
NVSHMEM_TYPE_SET_EMULATE(ulonglong, unsigned long long)
NVSHMEM_TYPE_SET_EMULATE(int32, int32_t)
NVSHMEM_TYPE_SET_EMULATE(int64, int64_t)
NVSHMEM_TYPE_SET_EMULATE(uint32, uint32_t)
NVSHMEM_TYPE_SET_EMULATE(uint64, uint64_t)
NVSHMEM_TYPE_SET_EMULATE(float, float)
NVSHMEM_TYPE_SET_EMULATE(double, double)
NVSHMEM_TYPE_SET_EMULATE(size, size_t)
NVSHMEM_TYPE_SET_EMULATE(ptrdiff, ptrdiff_t)

__device__ inline void *nvshmem_ptr(const void *ptr, int pe) { return nvshmemi_ptr(ptr, pe); }

#endif /* __CUDA_ARCH__ */

#ifdef __cplusplus
}
#endif
#endif /* _NVSHMEM_DEFINES_H_ */
