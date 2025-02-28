/*
 * Copyright (c) 2016-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include <cuda.h>
#include <cuda_runtime.h>

#include "device_host/nvshmem_common.cuh"
#include "team_internal.h"
#include "internal/host/util.h"

template <typename TYPE, rdxn_ops_t OP>
extern __global__ void nvshmemi_reduce_kernel(int start, int stride, int size, TYPE *dst,
                                              const TYPE *source, size_t nreduce, TYPE *pWrk,
                                              volatile long *pSync, volatile long *sync_counter);

template <typename T>
__global__ void nvshmemi_init_array_kernel(T *array, int len, T val) {
    for (int i = 0; i < len; i++) array[i] = val;
}

template <typename T>
void nvshmemi_call_init_array_kernel(T *array, int len, T val) {
    nvshmemi_init_array_kernel<T><<<1, 1>>>(array, len, val);
    CUDA_RUNTIME_CHECK(cudaGetLastError());
    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
}

template void nvshmemi_call_init_array_kernel<nvshmemi_team_t *>(nvshmemi_team_t **, int,
                                                                 nvshmemi_team_t *);

template void nvshmemi_call_init_array_kernel<long>(long *, int, long);

template <typename TYPE, rdxn_ops_t OP>
void nvshmemi_call_reduce_kernel(int start, int stride, int size, TYPE *dst, const TYPE *source,
                                 size_t nreduce, TYPE *pWrk, volatile long *pSync,
                                 volatile long *sync_counter) {
    nvshmemi_reduce_kernel<TYPE, OP>
        <<<1, 1>>>(start, stride, size, dst, source, nreduce, pWrk, pSync, sync_counter);
    CUDA_RUNTIME_CHECK(cudaGetLastError());
    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
}

template void nvshmemi_call_reduce_kernel<unsigned char, (rdxn_ops)0>(
    int, int, int, unsigned char *, unsigned char const *, unsigned long, unsigned char *,
    long volatile *, long volatile *);

template void nvshmemi_call_reduce_kernel<int, (rdxn_ops)4>(int, int, int, int *, int const *,
                                                            unsigned long, int *, long volatile *,
                                                            long volatile *);
