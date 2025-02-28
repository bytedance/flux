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

#ifndef _NVSHMEMI_H_TO_D_RMA_DEFS_H_
#define _NVSHMEMI_H_TO_D_RMA_DEFS_H_

#include <cuda_runtime.h>

#include "non_abi/nvshmem_build_options.h"
#ifdef NVSHMEM_ENABLE_ALL_DEVICE_INLINING
#include "non_abi/device/pt-to-pt/transfer_device.cuh"
#else
#include "non_abi/device/pt-to-pt/nvshmemi_transfer_api.cuh"
#endif

/* rma start */
#ifdef __CUDA_ARCH__
__device__ void nvshmemi_transfer_rma_nbi_translator(void *rptr, void *lptr,
                                                     rma_bytesdesc_t bytesdesc, int pe,
                                                     const nvshmemi_op_t desc) {
    switch (desc) {
        case NVSHMEMI_OP_PUT:
            nvshmemi_transfer_rma_nbi<NVSHMEMI_THREADGROUP_THREAD, NVSHMEMI_OP_PUT>(
                (void *)rptr, (void *)lptr, (size_t)(bytesdesc.nelems * bytesdesc.elembytes), pe);
            break;
        case NVSHMEMI_OP_P:
            nvshmemi_transfer_rma_nbi<NVSHMEMI_THREADGROUP_THREAD, NVSHMEMI_OP_PUT>(
                (void *)rptr, (void *)lptr, (size_t)(bytesdesc.nelems * bytesdesc.elembytes), pe);
            break;
        case NVSHMEMI_OP_GET:
            nvshmemi_transfer_rma_nbi<NVSHMEMI_THREADGROUP_THREAD, NVSHMEMI_OP_GET>(
                (void *)rptr, (void *)lptr, (size_t)(bytesdesc.nelems * bytesdesc.elembytes), pe);
            break;
        case NVSHMEMI_OP_G:
            nvshmemi_transfer_rma_nbi<NVSHMEMI_THREADGROUP_THREAD, NVSHMEMI_OP_GET>(
                (void *)rptr, (void *)lptr, (size_t)(bytesdesc.nelems * bytesdesc.elembytes), pe);
            break;
        default:
            printf("Incorrect argument to on-stream\n");
    }
}
#endif

__global__ void nvshmemi_proxy_rma_entrypoint(void *rptr, void *lptr, rma_bytesdesc_t bytesdesc,
                                              int pe, const nvshmemi_op_t desc) {
#ifdef __CUDA_ARCH__
    nvshmemi_transfer_rma_nbi_translator((void *)rptr, (void *)lptr, bytesdesc, pe, desc);
#endif
}

__global__ void nvshmemi_proxy_rma_entrypoint_blocking(void *rptr, void *lptr,
                                                       rma_bytesdesc_t bytesdesc, int pe,
                                                       const nvshmemi_op_t desc) {
#ifdef __CUDA_ARCH__
    nvshmemi_transfer_rma_nbi_translator((void *)rptr, (void *)lptr, bytesdesc, pe, desc);
    nvshmemi_transfer_quiet<NVSHMEMI_THREADGROUP_THREAD>(true);
#endif
}

__global__ void nvshmemi_proxy_rma_signal_entrypoint(void *rptr, void *lptr,
                                                     rma_bytesdesc_t bytesdesc, uint64_t *sig_addr,
                                                     uint64_t signal, int sig_op, int pe,
                                                     const nvshmemi_op_t desc) {
#ifdef __CUDA_ARCH__
    nvshmemi_transfer_rma_nbi_translator((void *)rptr, (void *)lptr, bytesdesc, pe, desc);
    nvshmemi_transfer_amo_nonfetch((void *)sig_addr, signal, pe, (nvshmemi_amo_t)sig_op);
#endif
}

__global__ void nvshmemi_proxy_rma_signal_entrypoint_blocking(void *rptr, void *lptr,
                                                              rma_bytesdesc_t bytesdesc,
                                                              uint64_t *sig_addr, uint64_t signal,
                                                              int sig_op, int pe,
                                                              const nvshmemi_op_t desc) {
#ifdef __CUDA_ARCH__
    nvshmemi_transfer_put_signal<NVSHMEMI_THREADGROUP_THREAD>(
        (void *)rptr, (void *)lptr, (size_t)(bytesdesc.nelems * bytesdesc.elembytes),
        (void *)sig_addr, signal, (nvshmemi_amo_t)sig_op, pe, false);
    nvshmemi_transfer_quiet<NVSHMEMI_THREADGROUP_THREAD>(true);
#endif
}
#endif
