/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <map>
#include <typeinfo>

#include "internal/host/util.h"
#include "internal/non_abi/nvshmemi_h_to_d_coll_defs.cuh"

using namespace std;

map<string, size_t> nvshmemi_broadcast_maxblocksize;

template <typename T>
void nvshmemi_call_broadcast_on_stream_kernel(nvshmem_team_t team, T *dest, const T *source,
                                              size_t nelems, int PE_root, cudaStream_t stream) {
    int tmp;
    string type_str(typeid(T).name());
    if (nvshmemi_broadcast_maxblocksize.find(type_str) == nvshmemi_broadcast_maxblocksize.end()) {
        CUDA_RUNTIME_CHECK(cudaOccupancyMaxPotentialBlockSize(
            &tmp, (int *)&nvshmemi_broadcast_maxblocksize[type_str],
            broadcast_on_stream_kernel<T>));
    }
    int num_threads_per_block = (nvshmemi_broadcast_maxblocksize[type_str] > nelems)
                                    ? nelems
                                    : nvshmemi_broadcast_maxblocksize[type_str];
    int num_blocks = 1;
    broadcast_on_stream_kernel<T>
        <<<num_blocks, num_threads_per_block, 0, stream>>>(team, dest, source, nelems, PE_root);
    CUDA_RUNTIME_CHECK(cudaGetLastError());
}

#define INSTANTIATE_NVSHMEMI_CALL_BROADCAST_ON_STREAM_KERNEL(TYPE) \
    template void nvshmemi_call_broadcast_on_stream_kernel<TYPE>(  \
        nvshmem_team_t, TYPE *, const TYPE *, size_t, int, cudaStream_t);
INSTANTIATE_NVSHMEMI_CALL_BROADCAST_ON_STREAM_KERNEL(uint8_t)
INSTANTIATE_NVSHMEMI_CALL_BROADCAST_ON_STREAM_KERNEL(uint16_t)
INSTANTIATE_NVSHMEMI_CALL_BROADCAST_ON_STREAM_KERNEL(uint32_t)
INSTANTIATE_NVSHMEMI_CALL_BROADCAST_ON_STREAM_KERNEL(uint64_t)
INSTANTIATE_NVSHMEMI_CALL_BROADCAST_ON_STREAM_KERNEL(int8_t)
INSTANTIATE_NVSHMEMI_CALL_BROADCAST_ON_STREAM_KERNEL(int16_t)
INSTANTIATE_NVSHMEMI_CALL_BROADCAST_ON_STREAM_KERNEL(int32_t)
INSTANTIATE_NVSHMEMI_CALL_BROADCAST_ON_STREAM_KERNEL(int64_t)
INSTANTIATE_NVSHMEMI_CALL_BROADCAST_ON_STREAM_KERNEL(half)
INSTANTIATE_NVSHMEMI_CALL_BROADCAST_ON_STREAM_KERNEL(__nv_bfloat16)
INSTANTIATE_NVSHMEMI_CALL_BROADCAST_ON_STREAM_KERNEL(float)
INSTANTIATE_NVSHMEMI_CALL_BROADCAST_ON_STREAM_KERNEL(char)
INSTANTIATE_NVSHMEMI_CALL_BROADCAST_ON_STREAM_KERNEL(double)
INSTANTIATE_NVSHMEMI_CALL_BROADCAST_ON_STREAM_KERNEL(long long)
INSTANTIATE_NVSHMEMI_CALL_BROADCAST_ON_STREAM_KERNEL(unsigned long long)
