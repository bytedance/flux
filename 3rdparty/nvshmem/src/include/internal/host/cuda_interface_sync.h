/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _CUDA_INTERFACE_SYNC_H_
#define _CUDA_INTERFACE_SYNC_H_
#include "device_host/nvshmem_common.cuh"

#define DECL_CALL_NVSHMEMI_TYPENAME_WAIT_UNTIL_ON_STREAM_KERNEL(type, TYPE)               \
    void call_nvshmemi_##type##_wait_until_on_stream_kernel(volatile TYPE *ivar, int cmp, \
                                                            TYPE cmp_value, cudaStream_t cstream);
NVSHMEMI_REPT_FOR_WAIT_TYPES(DECL_CALL_NVSHMEMI_TYPENAME_WAIT_UNTIL_ON_STREAM_KERNEL)
#undef DECL_CALL_NVSHMEMI_TYPENAME_WAIT_UNTIL_ON_STREAM_KERNEL

#define DECL_CALL_NVSHMEMI_TYPENAME_WAIT_UNTIL_ALL_ON_STREAM_KERNEL(type, TYPE)          \
    void call_nvshmemi_##type##_wait_until_all_on_stream_kernel(                         \
        volatile TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value, \
        cudaStream_t cstream);
NVSHMEMI_REPT_FOR_WAIT_TYPES(DECL_CALL_NVSHMEMI_TYPENAME_WAIT_UNTIL_ALL_ON_STREAM_KERNEL)
#undef DECL_CALL_NVSHMEMI_TYPENAME_WAIT_UNTIL_ALL_ON_STREAM_KERNEL

#define DECL_CALL_NVSHMEMI_TYPENAME_WAIT_UNTIL_ALL_VECTOR_ON_STREAM_KERNEL(type, TYPE)    \
    void call_nvshmemi_##type##_wait_until_all_vector_on_stream_kernel(                   \
        volatile TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE *cmp_value, \
        cudaStream_t cstream);
NVSHMEMI_REPT_FOR_WAIT_TYPES(DECL_CALL_NVSHMEMI_TYPENAME_WAIT_UNTIL_ALL_VECTOR_ON_STREAM_KERNEL)
#undef DECL_CALL_NVSHMEMI_TYPENAME_WAIT_UNTIL_ALL_VECTOR_ON_STREAM_KERNEL

void call_nvshmemi_signal_wait_until_on_stream_kernel(volatile uint64_t *sig_addr, int cmp,
                                                      uint64_t cmp_value, cudaStream_t cstream);

void call_nvshmemi_signal_op_kernel(uint64_t *sig_addr, uint64_t signal, int sig_op, int pe,
                                    cudaStream_t cstrm);
#endif
