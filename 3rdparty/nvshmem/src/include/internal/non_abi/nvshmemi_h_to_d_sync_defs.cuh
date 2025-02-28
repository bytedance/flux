/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See COPYRIGHT.txt for license information
 */

/*
 * This file strictly forward declares APIs defined in device headers which are called
 * internally by the host library. These API calls are not part of the ABI since they are
 * statically compiled into the host code and unused from the application.
 */

#ifndef _NVSHMEMI_H_TO_D_SYNC_DEFS_H_
#define _NVSHMEMI_H_TO_D_SYNC_DEFS_H_

#include <cuda_runtime.h>

#include "device/nvshmem_defines.h"
#include "non_abi/device/common/nvshmemi_common_device.cuh"
#include "non_abi/device/wait/nvshmemi_wait_until_apis.cuh"
#include "non_abi/device/common/nvshmemi_common_device.cuh"

/* sync start */
#ifdef __CUDA_ARCH__
#define NVSHMEMI_TYPENAME_WAIT_UNTIL_ON_STREAM_KERNEL(TYPENAME, TYPE)         \
    static __global__ void nvshmemi_##TYPENAME##_wait_until_on_stream_kernel( \
        volatile TYPE *ivar, int cmp, TYPE cmp_value) {                       \
        nvshmem_##TYPENAME##_wait_until((TYPE *)ivar, cmp, cmp_value);        \
    }
#else
#define NVSHMEMI_TYPENAME_WAIT_UNTIL_ON_STREAM_KERNEL(TYPENAME, TYPE)         \
    static __global__ void nvshmemi_##TYPENAME##_wait_until_on_stream_kernel( \
        volatile TYPE *ivar, int cmp, TYPE cmp_value) {}
#endif
NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMI_TYPENAME_WAIT_UNTIL_ON_STREAM_KERNEL)
#undef NVSHMEMI_TYPENAME_WAIT_UNTIL_ON_STREAM_KERNEL

#ifdef __CUDA_ARCH__
#define NVSHMEMI_TYPENAME_WAIT_UNTIL_ALL_ON_STREAM_KERNEL(TYPENAME, TYPE)                   \
    static __global__ void nvshmemi_##TYPENAME##_wait_until_all_on_stream_kernel(           \
        volatile TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value) {  \
        nvshmem_##TYPENAME##_wait_until_all((TYPE *)ivars, nelems, status, cmp, cmp_value); \
    }
#else
#define NVSHMEMI_TYPENAME_WAIT_UNTIL_ALL_ON_STREAM_KERNEL(TYPENAME, TYPE)         \
    static __global__ void nvshmemi_##TYPENAME##_wait_until_all_on_stream_kernel( \
        volatile TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value) {}
#endif
NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMI_TYPENAME_WAIT_UNTIL_ALL_ON_STREAM_KERNEL)
#undef NVSHMEMI_TYPENAME_WAIT_UNTIL_ALL_ON_STREAM_KERNEL

#ifdef __CUDA_ARCH__
#define NVSHMEMI_TYPENAME_WAIT_UNTIL_ALL_VECTOR_ON_STREAM_KERNEL(TYPENAME, TYPE)                   \
    static __global__ void nvshmemi_##TYPENAME##_wait_until_all_vector_on_stream_kernel(           \
        volatile TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE *cmp_value) {        \
        nvshmem_##TYPENAME##_wait_until_all_vector((TYPE *)ivars, nelems, status, cmp, cmp_value); \
    }
#else
#define NVSHMEMI_TYPENAME_WAIT_UNTIL_ALL_VECTOR_ON_STREAM_KERNEL(TYPENAME, TYPE)         \
    static __global__ void nvshmemi_##TYPENAME##_wait_until_all_vector_on_stream_kernel( \
        volatile TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE *cmp_value) {}
#endif
NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMI_TYPENAME_WAIT_UNTIL_ALL_VECTOR_ON_STREAM_KERNEL)
#undef NVSHMEMI_TYPENAME_WAIT_UNTIL_ALL_VECTOR_ON_STREAM_KERNEL

static __global__ void nvshmemi_signal_wait_until_on_stream_kernel(volatile uint64_t *sig_addr,
                                                                   int cmp, uint64_t cmp_value) {
#ifdef __CUDA_ARCH__
    nvshmemi_wait_until<uint64_t>(sig_addr, cmp, cmp_value);
#endif
}

static __global__ void nvshmemi_signal_op_kernel(uint64_t *sig_addr, uint64_t signal, int sig_op,
                                                 int pe) {
#ifdef __CUDA_ARCH__
    nvshmemi_signal_op(sig_addr, signal, sig_op, pe);
#endif
}

static __global__ void nvshmemi_proxy_quiet_entrypoint() {
#ifdef __CUDA_ARCH__
    nvshmemi_quiet<NVSHMEMI_THREADGROUP_BLOCK>();
#endif
}
/* sync end */
#endif
