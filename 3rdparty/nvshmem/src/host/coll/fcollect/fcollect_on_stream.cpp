/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include <driver_types.h>                    // for cudaStream_t
#include <stddef.h>                          // for size_t
#include "device_host/nvshmem_common.cuh"    // for NVSHMEMI_REPT_FOR_STAN...
#include "device_host/nvshmem_types.h"       // for nvshmem_team_t
#include "fcollect.h"                        // for nvshmemi_fcollect_on_s...
#include "host/nvshmemx_coll_api.h"          // for nvshmemx_char_fcollect...
#include "internal/host/nvshmem_internal.h"  // for NVSHMEMI_CHECK_INIT_ST...
#include "internal/host/nvshmem_nvtx.hpp"    // for nvtx_cond_range, NVTX_...
#include "internal/host/util.h"              // for NVSHMEM_API_NOT_SUPPOR...

#define DEFN_NVSHMEMX_TYPENAME_FCOLLECT_ON_STREAM(TYPENAME, TYPE)                                  \
    int nvshmemx_##TYPENAME##_fcollect_on_stream(                                                  \
        nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nelems, cudaStream_t stream) { \
        NVTX_FUNC_RANGE_IN_GROUP(COLL);                                                            \
        NVSHMEMI_CHECK_INIT_STATUS();                                                              \
        NVSHMEM_API_NOT_SUPPORTED_WITH_LIMITED_MPG_RUNS();                                         \
        return nvshmemi_fcollect_on_stream<TYPE>(team, dest, source, nelems, stream);              \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFN_NVSHMEMX_TYPENAME_FCOLLECT_ON_STREAM)

int nvshmemx_fcollectmem_on_stream(nvshmem_team_t team, void *dest, const void *source,
                                   size_t nelems, cudaStream_t stream) {
    return nvshmemx_char_fcollect_on_stream(team, (char *)dest, (const char *)source, nelems,
                                            stream);
}
