/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "broadcast.h"
#include <cuda_runtime.h>                    // for cudaStreamSynchronize
#include <stddef.h>                          // for size_t
#include "device_host/nvshmem_common.cuh"    // for NVSHMEMI_REPT_FOR_STAN...
#include "device_host/nvshmem_types.h"       // for nvshmem_team_t
#include "host/nvshmem_coll_api.h"           // for nvshmem_char_broadcast
#include "internal/host/nvshmem_internal.h"  // for nvshmemi_state, nvshme...
#include "internal/host/nvshmemi_types.h"    // for nvshmemi_state
#include "internal/host/nvshmem_nvtx.hpp"    // for nvtx_cond_range, NVTX_...
#include "internal/host/util.h"              // for CUDA_RUNTIME_CHECK

#define DEFN_NVSHMEM_TYPENAME_BROADCAST(TYPENAME, TYPE)                                     \
    int nvshmem_##TYPENAME##_broadcast(nvshmem_team_t team, TYPE *dest, const TYPE *source, \
                                       size_t nelems, int PE_root) {                        \
        NVTX_FUNC_RANGE_IN_GROUP(COLL);                                                     \
        NVSHMEMI_CHECK_INIT_STATUS();                                                       \
        NVSHMEM_API_NOT_SUPPORTED_WITH_LIMITED_MPG_RUNS();                                  \
        nvshmemi_broadcast_on_stream<TYPE>(team, dest, source, nelems, PE_root,             \
                                           nvshmemi_state->my_stream);                      \
        CUDA_RUNTIME_CHECK(cudaStreamSynchronize(nvshmemi_state->my_stream));               \
        return 0;                                                                           \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFN_NVSHMEM_TYPENAME_BROADCAST)
#undef DEFN_NVSHMEM_TYPENAME_BROADCAST

int nvshmem_broadcastmem(nvshmem_team_t team, void *dest, const void *source, size_t nelems,
                         int PE_root) {
    NVTX_FUNC_RANGE_IN_GROUP(COLL);
    NVSHMEMI_CHECK_INIT_STATUS();
    NVSHMEM_API_NOT_SUPPORTED_WITH_LIMITED_MPG_RUNS();
    nvshmemi_broadcast_on_stream<char>(team, (char *)dest, (const char *)source, nelems, PE_root,
                                       nvshmemi_state->my_stream);
    CUDA_RUNTIME_CHECK(cudaStreamSynchronize(nvshmemi_state->my_stream));
    return 0;
}
