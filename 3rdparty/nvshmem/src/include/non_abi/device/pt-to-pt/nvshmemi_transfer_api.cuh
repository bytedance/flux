/*
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include <cuda_runtime.h>
#include "non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh"

#ifndef _NVSHMEMI_TRANSFER_H_
#define _NVSHMEMI_TRANSFER_H_
template <typename T>
__device__ void nvshmemi_transfer_rma_p(void *rptr, const T value, int pe);

template <typename T>
__device__ T nvshmemi_transfer_rma_g(void *rptr, int pe);

template <threadgroup_t SCOPE, nvshmemi_op_t channel_op>
__device__ void nvshmemi_transfer_rma(void *rptr, void *lptr, size_t bytes, int pe);

template <threadgroup_t SCOPE>
__device__ void nvshmemi_transfer_put_signal(void *rptr, void *lptr, size_t bytes, void *sig_addr,
                                             uint64_t signal, nvshmemi_amo_t sig_op, int pe,
                                             bool is_nbi);

template <threadgroup_t SCOPE, nvshmemi_op_t channel_op>
__device__ void nvshmemi_transfer_rma_nbi(void *rptr, void *lptr, size_t bytes, int pe);

template <typename T>
__device__ T nvshmemi_transfer_amo_fetch(void *rptr, T value, T compare, int pe, nvshmemi_amo_t op);

template <typename T>
__device__ void nvshmemi_transfer_amo_nonfetch(void *rptr, T value, int pe, nvshmemi_amo_t op);

template <threadgroup_t SCOPE>
__device__ void nvshmemi_transfer_quiet(bool use_membar);

template <threadgroup_t SCOPE>
__device__ void nvshmemi_transfer_fence();
__device__ void nvshmemi_transfer_enforce_consistency_at_target(bool use_membar);
__device__ void nvshmemi_transfer_syncapi_update_mem();
#endif /* _NVSHMEMI_TRANSFER_H_ */
