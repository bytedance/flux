/*
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef TRANSFER_DEVICE_CUH
#define TRANSFER_DEVICE_CUH

#include <cuda_runtime.h>
#include "non_abi/nvshmem_build_options.h"
#include "non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh"
#include "device_host/nvshmem_common.cuh"
#include "device/nvshmem_device_macros.h"
#include "non_abi/device/pt-to-pt/proxy_device.cuh"

#ifdef NVSHMEM_IBGDA_SUPPORT
#include "non_abi/device/pt-to-pt/ibgda_device.cuh"
#endif

#define TRANSFER_REPT_FOR_STANDARD_RMA_TYPES(FN_TEMPLATE) \
    FN_TEMPLATE(char)                                     \
    FN_TEMPLATE(unsigned char)                            \
    FN_TEMPLATE(short)                                    \
    FN_TEMPLATE(unsigned short)                           \
    FN_TEMPLATE(int)                                      \
    FN_TEMPLATE(unsigned int)                             \
    FN_TEMPLATE(long)                                     \
    FN_TEMPLATE(unsigned long)                            \
    FN_TEMPLATE(long long)                                \
    FN_TEMPLATE(unsigned long long)                       \
    FN_TEMPLATE(float)                                    \
    FN_TEMPLATE(double)

#define TRANSFER_REPT_FOR_STANDARD_AMO_TYPES(FN_TEMPLATE) \
    FN_TEMPLATE(short)                                    \
    FN_TEMPLATE(unsigned short)                           \
    FN_TEMPLATE(int)                                      \
    FN_TEMPLATE(unsigned int)                             \
    FN_TEMPLATE(long)                                     \
    FN_TEMPLATE(unsigned long)                            \
    FN_TEMPLATE(long long)                                \
    FN_TEMPLATE(unsigned long long)

#define TRANSFER_REPT_FOR_EXTENDED_AMO_TYPES(FN_TEMPLATE) \
    FN_TEMPLATE(float)                                    \
    FN_TEMPLATE(double)

#define TRANSFER_REPT_FOR_ALL_SCOPES(FN_TEMPLATE) \
    FN_TEMPLATE(NVSHMEMI_THREADGROUP_THREAD)      \
    FN_TEMPLATE(NVSHMEMI_THREADGROUP_WARP)        \
    FN_TEMPLATE(NVSHMEMI_THREADGROUP_BLOCK)

#ifdef NVSHMEM_ENABLE_ALL_DEVICE_INLINING
#define NVSHMEMI_STATIC static
#else
#define NVSHMEMI_STATIC
#endif

#ifdef __CUDA_ARCH__
__device__ NVSHMEMI_DEVICE_INLINE void nvshmemi_transfer_syncapi_update_mem() {
    __threadfence(); /* 1. Ensures consitency op is not called before the prior test/wait condition
                        has been met
                        2. Needed to prevent reorder of instructions after sync api (when the
                        following if condition is false) */
    if (nvshmemi_device_state_d.job_connectivity > NVSHMEMI_JOB_GPU_PROXY) {
        nvshmemi_proxy_enforce_consistency_at_target(true);
    }
}

template <typename T>
NVSHMEMI_STATIC __device__ NVSHMEMI_DEVICE_INLINE void nvshmemi_transfer_rma_p(void *rptr,
                                                                               const T value,
                                                                               int pe) {
#ifdef NVSHMEM_IBGDA_SUPPORT
    if (nvshmemi_device_state_d.ibgda_is_initialized) {
        nvshmemi_ibgda_rma_p<T>(rptr, value, pe);
    } else
#endif
    {
        nvshmemi_proxy_rma_p<T>(rptr, value, pe);
    }
}

#define TRANSFER_DECL_RMA_P(Type) \
    template __device__ void nvshmemi_transfer_rma_p<Type>(void *rptr, const Type value, int pe);

TRANSFER_REPT_FOR_STANDARD_RMA_TYPES(TRANSFER_DECL_RMA_P)

template <typename T>
NVSHMEMI_STATIC __device__ NVSHMEMI_DEVICE_INLINE T nvshmemi_transfer_rma_g(void *rptr, int pe) {
#ifdef NVSHMEM_IBGDA_SUPPORT
    if (nvshmemi_device_state_d.ibgda_is_initialized) {
        return nvshmemi_ibgda_rma_g<T>(rptr, pe);
    } else
#endif
    {
        return nvshmemi_proxy_rma_g<T>(rptr, pe);
    }
}

#define TRANSFER_DECL_RMA_G(Type) \
    template __device__ Type nvshmemi_transfer_rma_g<Type>(void *rptr, int pe);

TRANSFER_REPT_FOR_STANDARD_RMA_TYPES(TRANSFER_DECL_RMA_G)

template <threadgroup_t SCOPE, nvshmemi_op_t channel_op>
NVSHMEMI_STATIC __device__ NVSHMEMI_DEVICE_INLINE void nvshmemi_transfer_rma(void *rptr, void *lptr,
                                                                             size_t bytes, int pe) {
#ifdef NVSHMEM_IBGDA_SUPPORT
    if (nvshmemi_device_state_d.ibgda_is_initialized) {
        nvshmemi_ibgda_rma<SCOPE, channel_op>(rptr, lptr, bytes, pe);
    } else
#endif
    {
        int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
        if (!myIdx) {
            nvshmemi_proxy_rma_nbi(rptr, lptr, bytes, pe, channel_op);
            nvshmemi_proxy_quiet(false);
            if (SCOPE == nvshmemi_threadgroup_thread)
                __threadfence_block(); /* to prevent reuse of src buffer before quiet completion;
                                    for warp/block scope, following sync op will accomplish that */
        }
    }
}

#define TRANSFER_DECL_RMA(SCOPE)                                                                   \
    template __device__ void nvshmemi_transfer_rma<SCOPE, NVSHMEMI_OP_PUT>(void *rptr, void *lptr, \
                                                                           size_t bytes, int pe);  \
    template __device__ void nvshmemi_transfer_rma<SCOPE, NVSHMEMI_OP_GET>(void *rptr, void *lptr, \
                                                                           size_t bytes, int pe);

TRANSFER_REPT_FOR_ALL_SCOPES(TRANSFER_DECL_RMA)

template <threadgroup_t SCOPE>
NVSHMEMI_STATIC __device__ NVSHMEMI_DEVICE_INLINE void nvshmemi_transfer_put_signal(
    void *rptr, void *lptr, size_t bytes, void *sig_addr, uint64_t signal, nvshmemi_amo_t sig_op,
    int pe, bool is_nbi) {
#ifdef NVSHMEM_IBGDA_SUPPORT
    if (nvshmemi_device_state_d.ibgda_is_initialized) {
        nvshmemi_ibgda_put_signal<SCOPE>(rptr, lptr, bytes, sig_addr, signal, sig_op, pe, is_nbi);
    } else
#endif
    {
        int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
        if (!myIdx) {
            nvshmemi_proxy_rma_nbi(rptr, lptr, bytes, pe, NVSHMEMI_OP_PUT);
            nvshmemi_proxy_fence();
            nvshmemi_proxy_amo_nonfetch<uint64_t>(sig_addr, signal, pe, sig_op);
            if (is_nbi == 0) {
                nvshmemi_proxy_quiet(false);
                if (SCOPE == nvshmemi_threadgroup_thread)
                    __threadfence_block(); /* to prevent reuse of src buffer before quiet completion
                                        for warp/block scope, following sync op will accomplish that
                                      */
            }
        }
    }
}

#define TRANSFER_DECL_PUT_SIGNAL(SCOPE)                                        \
    template __device__ void nvshmemi_transfer_put_signal<SCOPE>(              \
        void *rptr, void *lptr, size_t bytes, void *sig_rptr, uint64_t signal, \
        nvshmemi_amo_t sig_op, int pe, bool is_nbi);

TRANSFER_REPT_FOR_ALL_SCOPES(TRANSFER_DECL_PUT_SIGNAL)

template <threadgroup_t SCOPE, nvshmemi_op_t channel_op>
NVSHMEMI_STATIC __device__ NVSHMEMI_DEVICE_INLINE void nvshmemi_transfer_rma_nbi(void *rptr,
                                                                                 void *lptr,
                                                                                 size_t bytes,
                                                                                 int pe) {
#ifdef NVSHMEM_IBGDA_SUPPORT
    if (nvshmemi_device_state_d.ibgda_is_initialized) {
        nvshmemi_ibgda_rma_nbi<SCOPE, channel_op>(rptr, lptr, bytes, pe);
    } else
#endif
    {
        int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
        if (!myIdx) nvshmemi_proxy_rma_nbi(rptr, lptr, bytes, pe, channel_op);
    }
}

#define TRANSFER_DECL_RMA_NBI(SCOPE)                                            \
    template __device__ void nvshmemi_transfer_rma_nbi<SCOPE, NVSHMEMI_OP_PUT>( \
        void *rptr, void *lptr, size_t bytes, int pe);                          \
    template __device__ void nvshmemi_transfer_rma_nbi<SCOPE, NVSHMEMI_OP_GET>( \
        void *rptr, void *lptr, size_t bytes, int pe);

TRANSFER_REPT_FOR_ALL_SCOPES(TRANSFER_DECL_RMA_NBI)

template <typename T>
NVSHMEMI_STATIC __device__ NVSHMEMI_DEVICE_INLINE T nvshmemi_transfer_amo_fetch(void *rptr, T value,
                                                                                T compare, int pe,
                                                                                nvshmemi_amo_t op) {
#ifdef NVSHMEM_IBGDA_SUPPORT
    if (nvshmemi_device_state_d.ibgda_is_initialized) {
        return nvshmemi_ibgda_amo_fetch<T>(rptr, value, compare, pe, op);
    } else
#endif
    {
        T retval;
        nvshmemi_proxy_amo_fetch<T>(rptr, (void *)&retval, value, compare, pe, op);
        return retval;
    }
}

#define TRANSFER_DECL_AMO_FETCH(Type)                           \
    template __device__ Type nvshmemi_transfer_amo_fetch<Type>( \
        void *rptr, const Type value, const Type compare, int pe, nvshmemi_amo_t op);

TRANSFER_REPT_FOR_STANDARD_AMO_TYPES(TRANSFER_DECL_AMO_FETCH);
TRANSFER_REPT_FOR_EXTENDED_AMO_TYPES(TRANSFER_DECL_AMO_FETCH);

template <typename T>
NVSHMEMI_STATIC __device__ NVSHMEMI_DEVICE_INLINE void nvshmemi_transfer_amo_nonfetch(
    void *rptr, T value, int pe, nvshmemi_amo_t op) {
#ifdef NVSHMEM_IBGDA_SUPPORT
    if (nvshmemi_device_state_d.ibgda_is_initialized) {
        nvshmemi_ibgda_amo_nonfetch<T>(rptr, value, pe, op);
    } else
#endif
    {
        nvshmemi_proxy_amo_nonfetch<T>(rptr, value, pe, op);
    }
}

#define TRANSFER_DECL_AMO_NONFETCH(Type)                                                        \
    template __device__ void nvshmemi_transfer_amo_nonfetch<Type>(void *rptr, const Type value, \
                                                                  int pe, nvshmemi_amo_t op);

TRANSFER_REPT_FOR_STANDARD_AMO_TYPES(TRANSFER_DECL_AMO_NONFETCH);
TRANSFER_REPT_FOR_EXTENDED_AMO_TYPES(TRANSFER_DECL_AMO_NONFETCH);

template <threadgroup_t SCOPE>
NVSHMEMI_STATIC __device__ NVSHMEMI_DEVICE_INLINE void nvshmemi_transfer_quiet(bool use_membar) {
    // quiet_on_stream also shares this code path.
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
#ifdef NVSHMEM_IBGDA_SUPPORT
    if (nvshmemi_device_state_d.ibgda_is_initialized) {
        nvshmemi_ibgda_quiet<SCOPE>();
        if (use_membar) {
            if (!myIdx) {
                __threadfence_system();
            }
        }
    } else
#endif
        if (nvshmemi_device_state_d.proxy == NVSHMEMI_PROXY_FULL) {
        if (!myIdx) {
            nvshmemi_proxy_quiet(use_membar);
        }
    }
    nvshmemi_threadgroup_sync<SCOPE>();
}

#define TRANSFER_DECL_QUIET(SCOPE) \
    template __device__ void nvshmemi_transfer_quiet<SCOPE>(bool use_membar);

TRANSFER_REPT_FOR_ALL_SCOPES(TRANSFER_DECL_QUIET);
#undef TRANSFER_DECL_QUIET

template <threadgroup_t SCOPE>
NVSHMEMI_STATIC __device__ NVSHMEMI_DEVICE_INLINE void nvshmemi_transfer_fence() {
#ifdef NVSHMEM_IBGDA_SUPPORT
    if (nvshmemi_device_state_d.ibgda_is_initialized) {
        nvshmemi_ibgda_fence<SCOPE>();
    } else
#endif
        if ((nvshmemi_device_state_d.proxy == NVSHMEMI_PROXY_FULL) &&
            !nvshmemi_device_state_d.proxy_ops_are_ordered) {
        int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
        if (!myIdx) {
            nvshmemi_proxy_fence();
        }
    }
    nvshmemi_threadgroup_sync<SCOPE>();
}

#define TRANSFER_DECL_FENCE(SCOPE) template __device__ void nvshmemi_transfer_fence<SCOPE>();

TRANSFER_REPT_FOR_ALL_SCOPES(TRANSFER_DECL_FENCE);
#undef TRANSFER_DECL_FENCE

NVSHMEMI_STATIC __device__ NVSHMEMI_DEVICE_INLINE void
nvshmemi_transfer_enforce_consistency_at_target(bool use_membar) {
#ifdef NVSHMEM_IBGDA_SUPPORT
    if (nvshmemi_device_state_d.ibgda_is_initialized) {
        nvshmemi_ibgda_enforce_consistency_at_target(use_membar);
    } else
#endif
    {
        nvshmemi_proxy_enforce_consistency_at_target(use_membar);
    }
}
#endif /* __CUDA_ARCH__ */

#endif /* TRANSFER_DEVICE_CUH */
