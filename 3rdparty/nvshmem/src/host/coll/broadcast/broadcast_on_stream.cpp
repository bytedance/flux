/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include <driver_types.h>                    // for cudaStream_t
#include <stddef.h>                          // for size_t
#include "broadcast.h"                       // for nvshmemi_broadcast_on_...
#include "device_host/nvshmem_common.cuh"    // for NVSHMEMI_REPT_FOR_STAN...
#include "device_host/nvshmem_types.h"       // for nvshmem_team_t
#include "host/nvshmemx_coll_api.h"          // for nvshmemx_char_broadcas...
#include "internal/host/nvshmem_internal.h"  // for NVSHMEMI_CHECK_INIT_ST...
#include "internal/host/nvshmem_nvtx.hpp"    // for nvtx_cond_range, NVTX_...
#include "internal/host/util.h"              // for NVSHMEM_API_NOT_SUPPOR...

#define DEFN_NVSHMEMX_BROADCAST_ON_STREAM(TYPENAME, TYPE)                                         \
    int nvshmemx_##TYPENAME##_broadcast_on_stream(nvshmem_team_t team, TYPE *dest,                \
                                                  const TYPE *source, size_t nelems, int PE_root, \
                                                  cudaStream_t stream) {                          \
        NVTX_FUNC_RANGE_IN_GROUP(COLL);                                                           \
        NVSHMEMI_CHECK_INIT_STATUS();                                                             \
        NVSHMEM_API_NOT_SUPPORTED_WITH_LIMITED_MPG_RUNS();                                        \
        return nvshmemi_broadcast_on_stream<TYPE>(team, dest, source, nelems, PE_root, stream);   \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFN_NVSHMEMX_BROADCAST_ON_STREAM)
#undef DEFN_NVSHMEMX_BROADCAST_ON_STREAM

int nvshmemx_broadcastmem_on_stream(nvshmem_team_t team, void *dest, const void *source,
                                    size_t nelems, int PE_root, cudaStream_t stream) {
    return nvshmemx_char_broadcast_on_stream(team, (char *)dest, (const char *)source, nelems,
                                             PE_root, stream);
}
