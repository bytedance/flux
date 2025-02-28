/*
 * Copyright (c) 2016-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */
#ifndef _NVSHMEM_COMMON_DEVICE_CUH_
#define _NVSHMEM_COMMON_DEVICE_CUH_

#include <cuda_runtime.h>
#if not defined __CUDACC_RTC__
#include <stdint.h>
#include <stddef.h>
#include <type_traits>
#else
#include <cuda/std/cstdint>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>
#endif
#include "non_abi/nvshmem_build_options.h"
#include "device_host/nvshmem_common.cuh"
#include "device_host_transport/nvshmem_common_transport.h"
#include "non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh"
#ifdef NVSHMEM_ENABLE_ALL_DEVICE_INLINING
#include "non_abi/device/pt-to-pt/transfer_device.cuh"
#else
#include "non_abi/device/pt-to-pt/nvshmemi_transfer_api.cuh"
#endif
#include "non_abi/device/pt-to-pt/proxy_device.cuh"
#include "non_abi/device/team/nvshmemi_team_defines.cuh"

#define _LL_MAX_UNROLL 4

#define _LL_128_FLAG_THREAD 8
#define _LL_128_FLAG_SIZE 8
#define _LL_128_STORE_SIZE 16
#define _LL_128_PACKETS_PER_WARP 4
#define _LL_128_PACKET_DATA_SIZE 120
#define _LL_128_PACKET_PSYNC_SIZE 128
#define _LL_128_NUM_DATA_ELEMS_PER_PACKET(T) (_LL_128_PACKET_DATA_SIZE / sizeof(T))
#define _LL_128_NUM_PSYNC_ELEMS_PER_PACKET(T) (_LL_128_PACKET_PSYNC_SIZE / sizeof(T))
#define _LL_128_NUM_ELEMS_PER_STORE(T) (_LL_128_STORE_SIZE / sizeof(T))
#define _LL_128_NUM_ELEMS_PER_FLAG(T) (_LL_128_FLAG_SIZE / sizeof(T))
#define _LL_128_NUM_DATA_ELEMS_PER_WARP(UNROLL, T) \
    (UNROLL * _LL_128_PACKETS_PER_WARP * _LL_128_NUM_DATA_ELEMS_PER_PACKET(T))
#define _LL_128_NUM_PSYNC_ELEMS_PER_WARP(UNROLL, T) \
    (UNROLL * _LL_128_PACKETS_PER_WARP * _LL_128_NUM_PSYNC_ELEMS_PER_PACKET(T))

#define _LL_8_DATA_BYTES_PER_WARP 256
#define _LL_8_PSYNC_BYTES_PER_WARP 512
#define _LL_128_DATA_BYTES_PER_WARP 480
#define _LL_128_PSYNC_BYTES_PER_WARP 512

typedef enum { _LL_PSYNC_NON_VOLATILE = 0, _LL_PSYNC_VOLATILE } nvshmemi_psync_volatile_t;

#ifdef __CUDA_ARCH__
__device__ int nvshmemi_team_translate_pe(nvshmemi_team_t *src_team, int src_pe,
                                          nvshmemi_team_t *dest_team);
__device__ long *nvshmemi_team_get_psync(nvshmemi_team_t *team, nvshmemi_team_op_t op);
__device__ long *nvshmemi_team_get_sync_counter(nvshmemi_team_t *team);

template <threadgroup_t SCOPE>
__device__ inline void nvshmemi_quiet() {
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    if ((nvshmemi_device_state_d.job_connectivity > NVSHMEMI_JOB_GPU_LDST)) {
        nvshmemi_transfer_quiet<SCOPE>(true);
    } else {
        if (!myIdx)
            __threadfence_system(); /* Use __threadfence_system instead of __threadfence
                                     for data visibility in case of intra-node GPU transfers */
        nvshmemi_threadgroup_sync<SCOPE>();
    }
}

template __device__ void nvshmemi_quiet<NVSHMEMI_THREADGROUP_THREAD>();
template __device__ void nvshmemi_quiet<NVSHMEMI_THREADGROUP_WARP>();
template __device__ void nvshmemi_quiet<NVSHMEMI_THREADGROUP_BLOCK>();

__device__ inline void nvshmemi_fence() {
    if (nvshmemi_device_state_d.job_connectivity > NVSHMEMI_JOB_GPU_LDST) {
        nvshmemi_transfer_fence<NVSHMEMI_THREADGROUP_THREAD>();
    }
    __threadfence_system(); /* Use __threadfence_system instead of __threadfence
                               for data visibility in case of intra-node GPU transfers */
}

template <typename T>
__device__ inline int nvshmemi_test(volatile T *ivar, int cmp, T cmp_value) {
    int return_value = 0;
    if (NVSHMEM_CMP_GE == cmp) {
        if (*ivar >= cmp_value) return_value = 1;
    } else if (NVSHMEM_CMP_EQ == cmp) {
        if (*ivar == cmp_value) return_value = 1;
    } else if (NVSHMEM_CMP_NE == cmp) {
        if (*ivar != cmp_value) return_value = 1;
    } else if (NVSHMEM_CMP_GT == cmp) {
        if (*ivar > cmp_value) return_value = 1;
    } else if (NVSHMEM_CMP_LT == cmp) {
        if (*ivar < cmp_value) return_value = 1;
    } else if (NVSHMEM_CMP_LE == cmp) {
        if (*ivar <= cmp_value) return_value = 1;
    }
    return return_value;
}

#if not defined __CUDACC_RTC__
#define TYPE_IS_FLOAT(T) std::is_floating_point<T>::value
#else
#define TYPE_IS_FLOAT(T) cuda::std::is_floating_point<T>::value
#endif

// mcast store of 16B
template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_mcast16_store_threadgroup(int4 *dest, const int4 *source,
                                                          size_t len) {
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    for (size_t j = myIdx * 4; j < (len / sizeof(uint32_t)); j += groupSize * 4) {
        uint32_t u4[4];
        asm("ld.global.v4.b32 {%0, %1, %2, %3}, [%4]; "
            : "=r"(u4[0]), "=r"(u4[1]), "=r"(u4[2]), "=r"(u4[3])
            : "l"(source + j / 4));
        asm("multimem.st.global.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(dest + j / 4), "r"(u4[0]),
            "r"(u4[1]), "r"(u4[2]), "r"(u4[3])
            : "memory");
    }
}

// mcast store of 8B
template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_mcast8_store_threadgroup(uint64_t *dest, const uint64_t *source,
                                                         size_t len) {
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    for (size_t j = myIdx; j < (len / sizeof(uint64_t)); j += groupSize) {
        uint64_t val1;
        asm("ld.global.b64 %0, [%1];" : "=l"(val1) : "l"(source + j));
        asm("multimem.st.global.u64 [%0], %1;" ::"l"(dest + j), "l"(val1) : "memory");
    }
}

// mcast store of 4B
template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_mcast4_store_threadgroup(uint32_t *dest, const uint32_t *source,
                                                         size_t len) {
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    for (size_t j = myIdx; j < (len / sizeof(uint32_t)); j += groupSize) {
        uint32_t val1;
        asm("ld.global.b32 %0, [%1];" : "=r"(val1) : "l"(source + j));
        asm("multimem.st.global.u32 [%0], %1;" ::"l"(dest + j), "r"(val1) : "memory");
    }
}

/**
 * This function returns non-zero unaligned bytes that require a unicast store
 */
template <typename T, threadgroup_t SCOPE>
__device__ inline size_t nvshmemi_mcast_memcpy_threadgroup(T *__restrict__ dst,
                                                           const T *__restrict__ src, size_t len) {
    /*
     * If src and dst are 16B aligned copy as much as possible using 16B chunks
     */
    if ((uintptr_t)dst % 16 == 0 && (uintptr_t)src % 16 == 0 && len >= 16) {
        int4 *__restrict__ dst_p = (int4 *)dst;
        const int4 *__restrict__ src_p = (const int4 *)src;
        const size_t nelems = len / 16;
        nvshmemi_mcast16_store_threadgroup<T, SCOPE>(dst_p, src_p, len);
        len -= nelems * 16;

        if (0 == len) return 0;
        dst = (T *)(dst_p + nelems);
        src = (T *)(src_p + nelems);
    }

    /*
     * If src and dst are 8B aligned copy as much as possible using 8B chunks
     */
    if ((uintptr_t)dst % 8 == 0 && (uintptr_t)src % 8 == 0 && len >= 8) {
        const size_t nelems = len / 8;
        uint64_t *__restrict__ dst_p = (uint64_t *)dst;
        const uint64_t *__restrict__ src_p = (const uint64_t *)src;
        nvshmemi_mcast8_store_threadgroup<T, SCOPE>(dst_p, src_p, len);
        len -= nelems * 8;

        if (0 == len) return 0;
        dst = (T *)(dst_p + nelems);
        src = (T *)(src_p + nelems);
    }

    /*
     * If src and dst are 4B aligned copy as much as possible using 4B chunks
     */
    if ((uintptr_t)dst % 4 == 0 && (uintptr_t)src % 4 == 0 && len >= 4) {
        const size_t nelems = len / 4;
        uint32_t *__restrict__ dst_p = (uint32_t *)dst;
        const uint32_t *__restrict__ src_p = (const uint32_t *)src;
        nvshmemi_mcast4_store_threadgroup<T, SCOPE>(dst_p, src_p, len);
        len -= nelems * 4;

        if (0 == len) return 0;
        dst = (T *)(dst_p + nelems);
        src = (T *)(src_p + nelems);
    }

    /* if len is non-zero, caller will retry with unicast stores */
    return (len);
}

template <threadgroup_t SCOPE>
__device__ inline void nvshmemi_memcpy_threadgroup(void *__restrict__ dst,
                                                   const void *__restrict__ src, size_t len) {
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    /*
     * If src and dst are 16B aligned copy as much as possible using 16B chunks
     */
    if ((uintptr_t)dst % 16 == 0 && (uintptr_t)src % 16 == 0) {
        int4 *__restrict__ dst_p = (int4 *)dst;
        const int4 *__restrict__ src_p = (const int4 *)src;
        const size_t nelems = len / 16;

        for (size_t i = myIdx; i < nelems; i += groupSize) dst_p[i] = src_p[i];

        len -= nelems * 16;

        if (0 == len) return;

        dst = (void *)(dst_p + nelems);
        src = (void *)(src_p + nelems);
    }

    /*
     * If src and dst are 8B aligned copy as much as possible using 8B chunks
     */
    if ((uintptr_t)dst % 8 == 0 && (uintptr_t)src % 8 == 0) {
        uint64_t *__restrict__ dst_p = (uint64_t *)dst;
        const uint64_t *__restrict__ src_p = (const uint64_t *)src;
        const size_t nelems = len / 8;

        for (size_t i = myIdx; i < nelems; i += groupSize) dst_p[i] = src_p[i];

        len -= nelems * 8;

        if (0 == len) return;

        dst = (void *)(dst_p + nelems);
        src = (void *)(src_p + nelems);
    }

    /*
     * If src and dst are 4B aligned copy as much as possible using 4B chunks
     */
    if ((uintptr_t)dst % 4 == 0 && (uintptr_t)src % 4 == 0) {
        uint32_t *__restrict__ dst_p = (uint32_t *)dst;
        const uint32_t *__restrict__ src_p = (const uint32_t *)src;
        const size_t nelems = len / 4;

        for (size_t i = myIdx; i < nelems; i += groupSize) dst_p[i] = src_p[i];

        len -= nelems * 4;

        if (0 == len) return;

        dst = (void *)(dst_p + nelems);
        src = (void *)(src_p + nelems);
    }

    /*
     * If src and dst are 2B aligned copy as much as possible using 2B chunks
     */
    if ((uintptr_t)dst % 2 == 0 && (uintptr_t)src % 2 == 0) {
        uint16_t *__restrict__ dst_p = (uint16_t *)dst;
        const uint16_t *__restrict__ src_p = (const uint16_t *)src;
        const size_t nelems = len / 2;

        for (size_t i = myIdx; i < nelems; i += groupSize) dst_p[i] = src_p[i];

        len -= nelems * 2;

        if (0 == len) return;

        dst = (void *)(dst_p + nelems);
        src = (void *)(src_p + nelems);
    }

    unsigned char *__restrict__ dst_c = (unsigned char *)dst;
    const unsigned char *__restrict__ src_c = (const unsigned char *)src;

    for (size_t i = myIdx; i < len; i += groupSize) dst_c[i] = src_c[i];
}

template <typename T>
__device__ inline void nvshmemi_p(T *dest, const T value, int pe) {
    const void *peer_base_addr =
        (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);
    if (peer_base_addr) {
        T *dest_actual = (T *)((char *)(peer_base_addr) +
                               ((char *)dest - (char *)(nvshmemi_device_state_d.heap_base)));
        *dest_actual = value;
    } else {
        nvshmemi_transfer_rma_p<T>((void *)dest, value, pe);
    }
}

template <typename T>
__device__ inline T nvshmemi_g(const T *source, int pe) {
    const void *peer_base_addr =
        (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);
    if (peer_base_addr) {
        T *source_actual = (T *)((char *)(peer_base_addr) +
                                 ((char *)source - (char *)(nvshmemi_device_state_d.heap_base)));
        return *source_actual;
    } else {
        return nvshmemi_transfer_rma_g<T>((void *)source, pe);
    }
}

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_put_threadgroup(T *dest, const T *source, size_t nelems, int pe) {
    nvshmemi_threadgroup_sync<SCOPE>();
    void *peer_base_addr =
        (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);
    if (peer_base_addr) {
        char *dest_actual =
            (char *)(peer_base_addr) + ((char *)dest - (char *)(nvshmemi_device_state_d.heap_base));
        nvshmemi_memcpy_threadgroup<SCOPE>((void *)dest_actual, (const void *)source,
                                           nelems * sizeof(T));
    } else {
        nvshmemi_transfer_rma<SCOPE, NVSHMEMI_OP_PUT>((void *)dest, (void *)source,
                                                      nelems * sizeof(T), pe);
    }
    nvshmemi_threadgroup_sync<SCOPE>();
}

__device__ inline void nvshmemi_signal_op(uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) {
    const void *peer_base_addr =
        (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);
    if (sig_op == NVSHMEMI_AMO_SIGNAL_SET && peer_base_addr != NULL) {
        volatile uint64_t *dest_actual =
            (volatile uint64_t *)((char *)(peer_base_addr) +
                                  ((char *)sig_addr - (char *)(nvshmemi_device_state_d.heap_base)));
        *dest_actual = signal;
    } else if (nvshmemi_device_state_d.job_connectivity <= NVSHMEMI_JOB_GPU_LDST) {
        volatile uint64_t *dest_actual =
            (volatile uint64_t *)((char *)(peer_base_addr) +
                                  ((char *)sig_addr - (char *)(nvshmemi_device_state_d.heap_base)));
        /* sig_op == NVSHMEM_SIGNAL_ADD */
        atomicAdd_system((unsigned long long *)dest_actual, signal);
    } else {
        nvshmemi_transfer_amo_nonfetch<uint64_t>((void *)sig_addr, signal, pe,
                                                 (nvshmemi_amo_t)sig_op);
    }
}

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemii_put_signal_threadgroup(T *dest, const T *source, size_t nelems,
                                                        uint64_t *sig_addr, uint64_t signal,
                                                        int sig_op, int pe, bool is_nbi) {
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    void *peer_base_addr =
        (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);
    if (peer_base_addr) {
        char *dest_actual =
            (char *)(peer_base_addr) + ((char *)dest - (char *)(nvshmemi_device_state_d.heap_base));
        nvshmemi_memcpy_threadgroup<SCOPE>((void *)dest_actual, (const void *)source,
                                           nelems * sizeof(T));
        nvshmemi_threadgroup_sync<SCOPE>();
        if (!myIdx) {
            __threadfence_system();
            nvshmemi_signal_op(sig_addr, signal, sig_op, pe);
        }
    } else {
        nvshmemi_transfer_put_signal<SCOPE>((void *)dest, (void *)source, nelems * sizeof(T),
                                            (void *)sig_addr, signal, (nvshmemi_amo_t)sig_op, pe,
                                            is_nbi);
    }
}

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_put_signal_threadgroup(T *dest, const T *source, size_t nelems,
                                                       uint64_t *sig_addr, uint64_t signal,
                                                       int sig_op, int pe, bool is_nbi) {
    nvshmemi_threadgroup_sync<SCOPE>();
    nvshmemii_put_signal_threadgroup<T, SCOPE>(dest, source, nelems, sig_addr, signal, sig_op, pe,
                                               is_nbi);
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_get_threadgroup(T *dest, const T *source, size_t nelems, int pe) {
    nvshmemi_threadgroup_sync<SCOPE>();
    void *peer_base_addr =
        (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);
    if (peer_base_addr) {
        char *source_actual = (char *)(peer_base_addr) +
                              ((char *)source - (char *)(nvshmemi_device_state_d.heap_base));
        nvshmemi_memcpy_threadgroup<SCOPE>((void *)dest, (const void *)source_actual,
                                           nelems * sizeof(T));
    } else {
        nvshmemi_transfer_rma<SCOPE, NVSHMEMI_OP_GET>((void *)source, (void *)dest,
                                                      nelems * sizeof(T), pe);
    }
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemii_put_nbi_threadgroup(T *dest, const T *source, size_t nelems,
                                                     int pe) {
    void *peer_base_addr =
        (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);
    if (peer_base_addr) {
        char *dest_actual =
            (char *)(peer_base_addr) + ((char *)dest - (char *)(nvshmemi_device_state_d.heap_base));
        nvshmemi_memcpy_threadgroup<SCOPE>((void *)dest_actual, (const void *)source,
                                           nelems * sizeof(T));
    } else {
        nvshmemi_transfer_rma_nbi<SCOPE, NVSHMEMI_OP_PUT>((void *)dest, (void *)source,
                                                          nelems * sizeof(T), pe);
    }
}

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_put_nbi_threadgroup(T *dest, const T *source, size_t nelems,
                                                    int pe) {
    nvshmemi_threadgroup_sync<SCOPE>();
    nvshmemii_put_nbi_threadgroup<T, SCOPE>(dest, source, nelems, pe);
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_get_nbi_threadgroup(T *dest, const T *source, size_t nelems,
                                                    int pe) {
    nvshmemi_threadgroup_sync<SCOPE>();
    void *peer_base_addr =
        (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);
    if (peer_base_addr) {
        char *source_actual = (char *)(peer_base_addr) +
                              ((char *)source - (char *)(nvshmemi_device_state_d.heap_base));
        nvshmemi_memcpy_threadgroup<SCOPE>((void *)dest, (const void *)source_actual,
                                           nelems * sizeof(T));
    } else {
        nvshmemi_transfer_rma_nbi<SCOPE, NVSHMEMI_OP_GET>((void *)source, (void *)dest,
                                                          nelems * sizeof(T), pe);
    }
    nvshmemi_threadgroup_sync<SCOPE>();
}

__device__ inline void *nvshmemi_mc_ptr(nvshmemi_team_t *team, const void *ptr) {
    ptrdiff_t offset = (char *)ptr - (char *)nvshmemi_device_state_d.heap_base;
    if (ptr >= nvshmemi_device_state_d.heap_base && offset < nvshmemi_device_state_d.heap_size &&
        team->nvls_rsc_base_ptr != NULL) {
        void *mc_addr = (void *)__ldg((const long long unsigned *)team->nvls_rsc_base_ptr);
        if (mc_addr != NULL) mc_addr = (void *)((char *)mc_addr + offset);
        return mc_addr;
    } else
        return NULL;
}

__device__ inline void *nvshmemi_ptr(const void *ptr, int pe) {
    ptrdiff_t offset = (char *)ptr - (char *)nvshmemi_device_state_d.heap_base;

    if (ptr >= nvshmemi_device_state_d.heap_base && offset < nvshmemi_device_state_d.heap_size) {
        void *peer_addr = (void *)__ldg(
            (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);
        if (peer_addr != NULL) peer_addr = (void *)((char *)peer_addr + offset);
        return peer_addr;
    } else
        return NULL;
}

template <typename T, int UNROLL>
__device__ inline void nvshmemi_store_128b_register(uint64_t dest_regs[2 * UNROLL],
                                                    uint64_t ll_flag, uint64_t ll_flag_mask, T *src,
                                                    uint8_t unroll_stride) {
    union {
        uint64_t regs8[2];
        uint32_t regs4[4];
    };
    int myIdx = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();

    /* in both 16 and 8 byte cases, the flag thread needs to write two 8 byte chunks. */

#pragma unroll
    for (int unroll = 0; unroll < UNROLL; unroll++) {
        if (sizeof(T) >= 8) {
            uint64_t *src8 = (uint64_t *)(src + unroll * unroll_stride);
            asm("ld.b64 %0, [%1];" : "=l"(regs8[0]) : "l"(src8));
            asm("ld.b64 %0, [%1];" : "=l"(regs8[1]) : "l"(src8 + 1));
        } else if (sizeof(T) == 4) {
            uint32_t *src4 = (uint32_t *)(src + unroll * unroll_stride);
            asm("ld.b32 %0, [%1];" : "=r"(regs4[0]) : "l"(src4));
            asm("ld.b32 %0, [%1];" : "=r"(regs4[1]) : "l"(src4 + 1));
            asm("ld.b32 %0, [%1];" : "=r"(regs4[2]) : "l"(src4 + 2));
            asm("ld.b32 %0, [%1];" : "=r"(regs4[3]) : "l"(src4 + 3));
        } else {
            uint32_t lower, upper;
            /* funnelshift_r will use shift to select the
             * proper four bytes depending on the alignment.
             * 0x10 = 2, 0x11 = 3, 0x01 = 1
             */
            uint8_t shift = (uintptr_t)src % 4;
            /* We need to align the buffer to the nearest 4 bytes
             * Shift down 2 bytes in the case of 2B alignment down
             * 1B in the case of 1B alignment. Shifting down is always
             * appropriate due to 256 B alignment of cudaMalloc().
             */
            uint32_t *src4 =
                (uint32_t *)((uintptr_t)(src + unroll * unroll_stride) & -(uintptr_t)4);
            /* 2 Byte aligned - 2B in lower, 2B in upper */
            /* 1 Byte aligned (0x01) - 1B in upper, 3B in lower */
            /* 1 Byte aligned (0x11) - 3B in upper, 1B in lower */
            asm("ld.b32 %0, [%1];" : "=r"(lower) : "l"(src4));
            asm("ld.b32 %0, [%1];" : "=r"(upper) : "l"(src4 + 1));
            regs4[0] = __funnelshift_r(lower, upper, 8 * shift);
            asm("ld.b32 %0, [%1];" : "=r"(lower) : "l"(src4 + 1));
            asm("ld.b32 %0, [%1];" : "=r"(upper) : "l"(src4 + 2));
            regs4[1] = __funnelshift_r(lower, upper, 8 * shift);
            asm("ld.b32 %0, [%1];" : "=r"(lower) : "l"(src4 + 2));
            asm("ld.b32 %0, [%1];" : "=r"(upper) : "l"(src4 + 3));
            regs4[2] = __funnelshift_r(lower, upper, 8 * shift);
            asm("ld.b32 %0, [%1];" : "=r"(lower) : "l"(src4 + 4));
            asm("ld.b32 %0, [%1];" : "=r"(upper) : "l"(src4 + 5));
            regs4[3] = __funnelshift_r(lower, upper, 8 * shift);
        }
        dest_regs[unroll * 2] = regs8[0];
        dest_regs[unroll * 2 + 1] = (regs8[1] & ll_flag) | ll_flag_mask;
    }
}

template <typename T>
__device__ inline void nvshmemi_store_varlen_register(uint64_t dest_regs[2], uint64_t ll_flag,
                                                      uint64_t ll_flag_mask, void *src,
                                                      uint8_t nelems) {
    uint8_t i;
    union {
        uint64_t regs8[2];
        uint32_t regs4[4];
    };

    /* in both 16 and 8 byte cases, the flag thread needs to write two 8 byte chunks. */
    if (sizeof(T) >= 8) {
        uint64_t *src8 = (uint64_t *)src;
        for (i = 0; i < nelems; i++) {
            asm("ld.b64 %0, [%1];" : "=l"(regs8[i]) : "l"(src8 + i));
        }
    } else if (sizeof(T) == 4) {
        uint32_t *src4 = (uint32_t *)src;
        for (i = 0; i < nelems; i++) {
            asm("ld.b32 %0, [%1];" : "=r"(regs4[i]) : "l"(src4 + i));
        }
    } else {
        uint32_t lower, upper;
        /* funnelshift_r will use shift to select the
         * proper four bytes depending on the alignment.
         * 0x10 = 2, 0x11 = 3, 0x01 = 1
         */
        uint8_t shift = (uintptr_t)src % 4;
        /* We need to align the buffer to the nearest 4 bytes
         * Shift down 2 bytes in the case of 2B alignment down
         * 1B in the case of 1B alignment. Shifting down is always
         * appropriate due to 256 B alignment of cudaMalloc().
         */
        uint32_t *src4 = (uint32_t *)((uintptr_t)src & -(uintptr_t)4);
        /* 2 Byte aligned - 2B in lower, 2B in upper */
        /* 1 Byte aligned (0x01) - 1B in upper, 3B in lower */
        /* 1 Byte aligned (0x11) - 3B in upper, 1B in lower */
        for (i = 0; i < nelems; i++) {
            asm("ld.b32 %0, [%1];" : "=r"(lower) : "l"(src4 + i));
            asm("ld.b32 %0, [%1];" : "=r"(upper) : "l"(src4 + i + 1));
            regs4[i] = __funnelshift_r(lower, upper, 8 * shift);
        }
    }

    dest_regs[0] = regs8[0];
    dest_regs[1] = (regs8[1] & ll_flag) | ll_flag_mask;
}

template <typename T, threadgroup_t SCOPE, int UNROLL>
__device__ inline void nvshmemi_packLL128(T *psync, const T *source, size_t nelems,
                                          uint64_t ll_flag, nvshmemi_team_t *teami, int pe_count,
                                          int team_offset, int pe_group_offset) {
    const int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    const uint32_t warp_mask = 0x000000FF << (8 * (myIdx / 8));
    const int num_warps = nvshmemi_threadgroup_size<SCOPE>() / NVSHMEMI_WARP_SIZE;
    /* max 127 */
    const uint8_t remain = nelems % _LL_128_NUM_DATA_ELEMS_PER_PACKET(T);
    /* max ~ 2048 * 16 * 4*/
    const uint32_t source_offset =
        (myIdx / _LL_128_FLAG_THREAD) * UNROLL * _LL_128_NUM_DATA_ELEMS_PER_PACKET(T) +
        myIdx % _LL_128_FLAG_THREAD * _LL_128_NUM_ELEMS_PER_STORE(T);
    const uint32_t psync_offset =
        (myIdx / _LL_128_FLAG_THREAD) * UNROLL * _LL_128_NUM_PSYNC_ELEMS_PER_PACKET(T) +
        myIdx % _LL_128_FLAG_THREAD * _LL_128_NUM_ELEMS_PER_STORE(T);
    const uint32_t source_stride = num_warps * _LL_128_NUM_DATA_ELEMS_PER_WARP(UNROLL, T);
    const uint32_t psync_stride = num_warps * _LL_128_NUM_PSYNC_ELEMS_PER_WARP(UNROLL, T);
    int current_global_pe_index;
    const bool is_flag_thread = !((myIdx + 1) % _LL_128_FLAG_THREAD);

    /* We will always have a SCOPES worth of elements when doing unrolling */
    if (UNROLL > 1) {
        assert(remain == 0);
        assert(nelems % _LL_128_NUM_DATA_ELEMS_PER_WARP(UNROLL, T) == 0);
    }

    /* Both source_end and psync_end remove the last < 128 bytes.
     * These will be handled after the loop. */
    T *source_end = (T *)(source + nelems - remain);

    uint64_t regs[2 * UNROLL];
    T *source_ptr;
    T *psync_ptr;
    T *remote_psync_ptr;
    uint64_t ll_flag_mask;

    assert(nvshmemi_threadgroup_size<SCOPE>() % NVSHMEMI_WARP_SIZE == 0);
    assert((uintptr_t)psync % 16 == 0);
    assert(sizeof(T) <= 8);

    if (!is_flag_thread) {
        ll_flag = UINT64_MAX;
        ll_flag_mask = 0x0ULL;
    } else {
        ll_flag_mask = ll_flag;
    }
    __syncwarp();

    /* Initial offset by thread */
    source_ptr = (T *)source + source_offset;
    psync_ptr = (T *)psync + psync_offset;

    /* Only gate on source_end. psync_end is not important here. */
    for (; source_ptr < source_end; source_ptr += source_stride, psync_ptr += psync_stride) {
        nvshmemi_store_128b_register<T, UNROLL>(regs, ll_flag, ll_flag_mask, source_ptr,
                                                _LL_128_NUM_DATA_ELEMS_PER_PACKET(T));

        /* definitely possible we are not synchronized before loads to psync. Significant perf
         * overhead? */
        for (int pe = 0; pe < pe_count; pe++) {
            current_global_pe_index =
                teami->start + (team_offset + (pe_group_offset + pe) % pe_count) * teami->stride;
            /*             if (VOLATILE && current_global_pe_index == my_pe_idx) {
                            continue;
                        } */
            remote_psync_ptr = (T *)nvshmemi_ptr(psync_ptr, current_global_pe_index);
#pragma unroll
            for (int unroll = 0; unroll < UNROLL * 2; unroll += 2) {
                asm volatile(
                    "st.volatile.global.v2.b64 [%0], {%1,%2};" ::"l"((int4 *)remote_psync_ptr),
                    "l"(regs[unroll]), "l"(regs[unroll + 1]));
                remote_psync_ptr += _LL_128_NUM_PSYNC_ELEMS_PER_PACKET(T);
            }
        }
    }

    if (UNROLL == 1 && remain) {
        uint8_t num_elems_per_store;
        T *psync_end =
            psync + (nelems - remain) +
            (nelems / _LL_128_NUM_DATA_ELEMS_PER_PACKET(T)) * _LL_128_NUM_ELEMS_PER_FLAG(T);
        /* Last packet will contain < 128 bytes. the variable remain is usually > 0. */
        /* only need 1/4th of a warp */
        if (myIdx < 8) {
            source_ptr = source_end + source_offset;
            psync_ptr = psync_end + psync_offset;
            if (source_offset + _LL_128_NUM_ELEMS_PER_STORE(T) <= remain) {
                nvshmemi_store_128b_register<T, 1>(regs, UINT64_MAX, ll_flag_mask, source_ptr,
                                                   _LL_128_NUM_DATA_ELEMS_PER_PACKET(T));
            } else {
                num_elems_per_store = source_offset > remain ? 0 : remain - source_offset;
                nvshmemi_store_varlen_register<T>(regs, ll_flag, ll_flag_mask, source_ptr,
                                                  num_elems_per_store);
            }
            __syncwarp(warp_mask);
            for (int pe = 0; pe < pe_count; pe++) {
                current_global_pe_index =
                    teami->start +
                    (team_offset + (pe_group_offset + pe) % pe_count) * teami->stride;
                /*                 if (VOLATILE && current_global_pe_index == my_pe_idx) {
                                    continue;
                                } */
                remote_psync_ptr = (T *)nvshmemi_ptr(psync_ptr, current_global_pe_index);
                asm volatile(
                    "st.volatile.global.v2.b64 [%0], {%1,%2};" ::"l"((int4 *)remote_psync_ptr),
                    "l"(regs[0]), "l"(regs[1]));
            }
        }
    }
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <int UNROLL>
__device__ inline void nvshmemi_ll128_load_psync(uint64_t reg[UNROLL * 2], uint64_t *src,
                                                 uint64_t ll_flag, uint32_t warp_mask,
                                                 bool is_flag_thread) {
    uint64_t *offset_src;
    int i;
    bool flag_arrived;
    do {
        flag_arrived = true;
#pragma unroll
        for (i = 0; i < UNROLL * 2; i += 2) {
            offset_src = src + _LL_128_NUM_PSYNC_ELEMS_PER_PACKET(uint64_t) * (i / 2);
            asm volatile("ld.volatile.global.v2.b64 {%0,%1}, [%2];"
                         : "=l"(reg[i]), "=l"(reg[i + 1])
                         : "l"((int4 *)offset_src));
            flag_arrived &= (reg[i + 1] == ll_flag);
            flag_arrived |= !is_flag_thread;
        }
        flag_arrived = __all_sync(warp_mask, flag_arrived != false);
    } while (!flag_arrived);
}

template <typename T, int UNROLL>
__device__ inline void nvshmemi_recvLL128(T *dest, const T *psync, size_t nelems, uint64_t flag) {
    const int myIdx = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
    const uint32_t warp_mask = 0x000000FF << (8 * (myIdx / 8));
    /* max 127 */
    const uint8_t remain = nelems % _LL_128_NUM_DATA_ELEMS_PER_PACKET(T);
    /* max ~ 2048 * 16 * 4*/
    const uint32_t dest_offset =
        (myIdx / _LL_128_FLAG_THREAD) * UNROLL * _LL_128_NUM_DATA_ELEMS_PER_PACKET(T) +
        myIdx % _LL_128_FLAG_THREAD * _LL_128_NUM_ELEMS_PER_STORE(T);
    const uint32_t psync_offset =
        (myIdx / _LL_128_FLAG_THREAD) * UNROLL * _LL_128_NUM_PSYNC_ELEMS_PER_PACKET(T) +
        myIdx % _LL_128_FLAG_THREAD * _LL_128_NUM_ELEMS_PER_STORE(T);
    const bool is_flag_thread = !((myIdx + 1) % _LL_128_FLAG_THREAD);
    uint8_t source_copy_len = 16 >> is_flag_thread;

    /* We will always have a SCOPES worth of elements when doing unrolling */
    if (UNROLL > 1) {
        assert(remain == 0);
        assert(nelems % _LL_128_NUM_DATA_ELEMS_PER_WARP(UNROLL, T) == 0);
    }

    /* Both dest_end and psync_end remove the last < 128 bytes.
     * These will be handled after the loop. */
    T *dest_end = (T *)(dest + nelems - remain);

    uint64_t regs[2 * UNROLL];
    T *dest_ptr;
    T *cur_dest_ptr;
    T *psync_ptr;

    assert((uintptr_t)psync % 16 == 0);
    assert(sizeof(T) <= 8);

    /* Initial offset by thread */
    dest_ptr = (T *)dest + dest_offset;
    psync_ptr = (T *)psync + psync_offset;
    for (; dest_ptr < dest_end; dest_ptr += _LL_128_NUM_DATA_ELEMS_PER_WARP(UNROLL, T),
                                psync_ptr += _LL_128_NUM_PSYNC_ELEMS_PER_WARP(UNROLL, T)) {
        if (UNROLL > 1) {
            nvshmemi_ll128_load_psync<UNROLL>(regs, (uint64_t *)psync_ptr, flag, 0xFFFFFFFF,
                                              is_flag_thread);
        } else {
            nvshmemi_ll128_load_psync<UNROLL>(regs, (uint64_t *)psync_ptr, flag, warp_mask,
                                              is_flag_thread);
        }

        cur_dest_ptr = dest_ptr;
#pragma unroll
        for (int i = 0; i < UNROLL * 2; i += 2) {
            asm("st.global.b64 [%0], %1;" ::"l"((uint64_t *)cur_dest_ptr), "l"(regs[i]));
            if (!is_flag_thread) {
                asm("st.global.b64 [%0], %1;" ::"l"((uint64_t *)cur_dest_ptr + 1),
                    "l"(regs[i + 1]));
            }
            if (UNROLL > 1) {
                __syncwarp();
            } else {
                __syncwarp(warp_mask);
            }
            cur_dest_ptr += _LL_128_NUM_DATA_ELEMS_PER_PACKET(T);
        }
    }

    if (UNROLL == 1 && remain) {
        T *psync_end =
            (T *)psync + (nelems - remain) +
            (nelems / _LL_128_NUM_DATA_ELEMS_PER_PACKET(T)) * _LL_128_NUM_ELEMS_PER_FLAG(T);
        if (remain && myIdx < 8) {
            dest_ptr = dest_end + dest_offset;
            psync_ptr = psync_end + psync_offset;
            nvshmemi_ll128_load_psync<1>(regs, (uint64_t *)psync_ptr, flag, warp_mask,
                                         is_flag_thread);
            if (dest_offset < remain) {
                source_copy_len = dest_offset + _LL_128_NUM_ELEMS_PER_STORE(T) < remain
                                      ? _LL_128_STORE_SIZE
                                      : ((remain - dest_offset) * sizeof(T));
                nvshmemi_memcpy_threadgroup<NVSHMEMI_THREADGROUP_THREAD>(dest_ptr, regs,
                                                                         source_copy_len);
            }
        }
    }
    __syncwarp();
}

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_mcast_recvLL(T *dest, const uint64_t *src, size_t nelems,
                                             uint32_t flag) {
    // Assumptions: sizeof(T) >= 4 bytes, num_subelems is a multiple of 2
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    size_t num_subelems = nelems * (sizeof(T) / sizeof(uint32_t));
    if (TYPE_IS_FLOAT(T)) {
        for (int i = myIdx; i < num_subelems; i += groupSize) {
            float data1, flag1;
            volatile uint32_t flagu32;
            do {
                asm volatile("ld.volatile.global.v2.f32 {%0, %1}, [%2];"
                             : "=f"(data1), "=f"(flag1)
                             : "l"(&src[i]));
                asm("cvt.rni.u32.f32 %0, %1;" : "=r"(flagu32) : "f"(flag1));
            } while ((flagu32 != flag));
            *(float *)((char *)dest + i * sizeof(float)) = data1;
        }
    } else {
        for (int i = 2 * myIdx; i < num_subelems; i += 2 * groupSize) {
            uint32_t flag1, flag2, data1, data2;
            do {
                asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                             : "=r"(data1), "=r"(flag1), "=r"(data2), "=r"(flag2)
                             : "l"(&src[i]));
            } while ((flag1 != flag) || (flag2 != flag));
            *(uint32_t *)((char *)dest + i * sizeof(uint32_t)) = data1;
            *(uint32_t *)((char *)dest + (i + 1) * sizeof(uint32_t)) = data2;
        }
    }
}

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_mcast_packLL(uint64_t *dest, const T *source, size_t nelems,
                                             uint32_t ll_flag) {
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    const size_t num_subelems = nelems * (sizeof(T) / sizeof(float));
    float flagf32;
    if (TYPE_IS_FLOAT(T)) asm("cvt.rn.f32.u32 %0, %1;" : "=f"(flagf32) : "r"(ll_flag));
    for (int i = 2 * myIdx; i < num_subelems; i += 2 * groupSize) {
        if (TYPE_IS_FLOAT(T)) {
            float val1 = *(float *)((char *)source + i * sizeof(float));
            float val2 = *(float *)((char *)source + (i + 1) * sizeof(float));
            asm("multimem.st.global.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(&dest[i]), "f"(val1),
                "f"(flagf32), "f"(val2), "f"(flagf32)
                : "memory");
        } else {
            uint32_t val1 = *(uint32_t *)((char *)source + i * sizeof(uint32_t));
            uint32_t val2 = *(uint32_t *)((char *)source + (i + 1) * sizeof(uint32_t));
            asm("multimem.st.global.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(&dest[i]), "r"(val1),
                "r"(ll_flag), "r"(val2), "r"(ll_flag)
                : "memory");
        }
    }
}

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_recvLL(T *dest, const uint64_t *src, size_t nelems, uint32_t flag) {
    // Assumptions: sizeof(T) >= 4 bytes, num_subelems is a multiple of 2
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    size_t num_subelems = nelems * (sizeof(T) / sizeof(uint32_t));
    for (int i = 2 * myIdx; i < num_subelems; i += 2 * groupSize) {
        uint32_t flag1, flag2, data1, data2;
        do {
            asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(data1), "=r"(flag1), "=r"(data2), "=r"(flag2)
                         : "l"(&src[i]));

        } while ((flag1 != flag) || (flag2 != flag));
        *(uint32_t *)((char *)dest + i * sizeof(uint32_t)) = data1;
        *(uint32_t *)((char *)dest + (i + 1) * sizeof(uint32_t)) = data2;
    }
}

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_packLL_naive(uint64_t *dest, const T *source, size_t nelems,
                                             uint32_t ll_flag) {
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    size_t num_subelems = nelems * (sizeof(T) / sizeof(uint32_t));
    for (int i = myIdx * 2; i < num_subelems; i += groupSize * 2) {
        size_t dst_offset = 2 * i * sizeof(uint32_t);
        size_t src_offset = i * sizeof(uint32_t);
        asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(
                         (uint32_t *)((char *)dest + dst_offset)),
                     "r"(*(uint32_t *)((char *)source + src_offset)), "r"(ll_flag),
                     "r"(*(uint32_t *)((char *)source + src_offset + sizeof(uint32_t))),
                     "r"(ll_flag));
    }
}

template <typename T, threadgroup_t SCOPE, int UNROLL>
__device__ inline void nvshmemi_packLL(T *psync, const T *source, size_t nelems, uint32_t ll_flag,
                                       nvshmemi_team_t *teami, int pe_count, int team_offset) {
    const size_t num_subelems = nelems * (sizeof(T) / sizeof(uint32_t));
    const int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    const int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    /* each thread will write 2 subelements (8 bytes) */
    const size_t element_start = myIdx * UNROLL * 2;
    const size_t element_stride = groupSize * UNROLL * 2;
    size_t element_offset;
    uint32_t regs[2 * UNROLL];

    int current_global_pe_index;
    uint32_t *current_dest_address;

    /* We need to be sure to fill each unrolled loop */
    assert(num_subelems % UNROLL * 2 == 0);
    assert((uintptr_t)psync % 16 == 0);
    assert(sizeof(T) >= sizeof(uint32_t));

    for (element_offset = element_start; element_offset < num_subelems;
         element_offset += element_stride) {
#pragma unroll
        for (int unroll = 0; unroll < UNROLL * 2; unroll += 2) {
            if (sizeof(T) == sizeof(uint64_t)) {
                asm("ld.global.v2.u32 {%0,%1}, [%2];"
                    : "=r"(regs[unroll]), "=r"(regs[unroll + 1])
                    : "l"((uint32_t *)source + element_offset + unroll));
            } else {
                regs[unroll] = *((uint32_t *)source + element_offset + unroll);
                regs[unroll + 1] = *((uint32_t *)source + element_offset + unroll + 1);
            }
        }
        for (int pe = 0; pe < pe_count; pe++) {
            current_global_pe_index = teami->start + (team_offset + pe) * teami->stride;
            current_dest_address =
                (uint32_t *)nvshmemi_ptr(psync, current_global_pe_index) + element_offset * 2;

#pragma unroll
            for (int unroll = 0; unroll < UNROLL * 2; unroll += 2) {
                asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(
                                 (int4 *)current_dest_address),
                             "r"(regs[unroll]), "r"(ll_flag), "r"(regs[unroll + 1]), "r"(ll_flag));
                current_dest_address += sizeof(int4) / sizeof(uint32_t);
            }
        }
    }
}

#endif /* __CUDA__ARCH__ */
#endif /* _NVSHMEM_COMMON_DEVICE_CUH_ */
