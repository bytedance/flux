/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "reduce_common.cuh"
#include "internal/non_abi/nvshmemi_h_to_d_coll_defs.cuh"

/* This is a special kernel that is launched only with
one thread and is used during team creation in nvshmemi_team_plit_strided fn */
template <typename TYPE, rdxn_ops_t OP>
__global__ void nvshmemi_reduce_kernel(int start, int stride, int size, TYPE *dst,
                                       const TYPE *source, size_t nreduce, TYPE *pWrk,
                                       volatile long *pSync, volatile long *sync_counter) {
#ifdef __CUDA_ARCH__
    gpu_rdxn_on_demand_2<TYPE, OP>(start, stride, size, dst, source, nreduce, pWrk, pSync,
                                   sync_counter);
#endif
}

template __global__ void nvshmemi_reduce_kernel<unsigned char, (rdxn_ops)0>(
    int, int, int, unsigned char *, unsigned char const *, unsigned long, unsigned char *,
    long volatile *, long volatile *);
template __global__ void nvshmemi_reduce_kernel<int, (rdxn_ops)4>(int, int, int, int *, int const *,
                                                                  unsigned long, int *,
                                                                  long volatile *, long volatile *);
