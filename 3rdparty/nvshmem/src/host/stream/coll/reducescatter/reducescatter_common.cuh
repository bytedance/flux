/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */
#ifndef __REDUCESCATTER_COMMON_CUH__
#define __REDUCESCATTER_COMMON_CUH__
#include <cuda.h>
#include <cuda_runtime.h>
#include <map>
#include <string>
#include <typeinfo>

#include "internal/host/util.h"
#include "internal/non_abi/nvshmemi_h_to_d_coll_defs.cuh"

using namespace std;

static map<pair<string, rdxn_ops_t>, size_t> nvshmemi_reducescatter_maxblocksize;

template <typename TYPE, rdxn_ops_t OP>
void nvshmemi_call_reducescatter_on_stream_kernel(nvshmem_team_t team, TYPE *dest,
                                                  const TYPE *source, size_t nreduce,
                                                  cudaStream_t stream) {
    int tmp;
    pair<string, rdxn_ops_t> map_pair(string(typeid(TYPE).name()), OP);
    if (nvshmemi_reducescatter_maxblocksize.find(map_pair) ==
        nvshmemi_reducescatter_maxblocksize.end()) {
        CUDA_RUNTIME_CHECK(cudaOccupancyMaxPotentialBlockSize(
            &tmp, (int *)&nvshmemi_reducescatter_maxblocksize[map_pair],
            reducescatter_on_stream_kernel<TYPE, OP>));
    }
    /* By default select occupancy */
    int num_threads_per_block = (nvshmemi_reducescatter_maxblocksize[map_pair] > nreduce)
                                    ? nreduce
                                    : nvshmemi_reducescatter_maxblocksize[map_pair];

    /* Use env to override the value */
    if (nvshmemi_options.REDUCESCATTER_NTHREADS_provided) {
        num_threads_per_block = nvshmemi_options.REDUCESCATTER_NTHREADS;
    }

    int num_blocks = 1;
    reducescatter_on_stream_kernel<TYPE, OP>
        <<<num_blocks, num_threads_per_block, 0, stream>>>(team, dest, source, nreduce);
    CUDA_RUNTIME_CHECK(cudaGetLastError());
}

#define INSTANTIATE_NVSHMEMI_CALL_REDUCESCATTER_ON_STREAM_KERNEL(TYPE, OP) \
    template void nvshmemi_call_reducescatter_on_stream_kernel<TYPE, OP>(  \
        nvshmem_team_t, TYPE *, const TYPE *, size_t, cudaStream_t);

#define REPT_FOR_BITWISE_TYPES(FN, OP) \
    FN(int8_t, RDXN_OPS_##OP)          \
    FN(uint8_t, RDXN_OPS_##OP)         \
    FN(uint16_t, RDXN_OPS_##OP)        \
    FN(int16_t, RDXN_OPS_##OP)         \
    FN(uint32_t, RDXN_OPS_##OP)        \
    FN(int32_t, RDXN_OPS_##OP)         \
    FN(uint64_t, RDXN_OPS_##OP)        \
    FN(int64_t, RDXN_OPS_##OP)         \
    FN(char, RDXN_OPS_##OP)            \
    FN(long long, RDXN_OPS_##OP)       \
    FN(unsigned long long, RDXN_OPS_##OP)

#define REPT_FOR_FLOATING_TYPES(FN, OP) \
    FN(half, RDXN_OPS_##OP)             \
    FN(__nv_bfloat16, RDXN_OPS_##OP)    \
    FN(float, RDXN_OPS_##OP)            \
    FN(double, RDXN_OPS_##OP)

#endif /* __REDUCESCATTER_COMMON_CUH__ */
