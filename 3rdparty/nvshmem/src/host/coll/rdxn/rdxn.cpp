/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "rdxn.h"
#include <cuda_runtime.h>                    // for cudaStreamSynchronize
#include <stddef.h>                          // for size_t
#include "device_host/nvshmem_common.cuh"    // for RDXN_OPS_max, RDXN_OPS...
#include "device_host/nvshmem_types.h"       // for nvshmem_team_t
#include "host/nvshmem_coll_api.h"           // for nvshmem_char_max_reduce
#include "internal/host/nvshmem_internal.h"  // for nvshmemi_state, nvshme...
#include "internal/host/nvshmemi_types.h"    // for nvshmemi_state
#include "internal/host/nvshmem_nvtx.hpp"    // for nvtx_cond_range, NVTX_...
#include "internal/host/util.h"              // for CUDA_RUNTIME_CHECK

#define DEFN_NVSHMEM_TYPENAME_OP_REDUCE(TYPENAME, TYPE, OP)                                     \
    int nvshmem_##TYPENAME##_##OP##_reduce(nvshmem_team_t team, TYPE *dest, const TYPE *source, \
                                           size_t nreduce) {                                    \
        NVTX_FUNC_RANGE_IN_GROUP(COLL);                                                         \
        NVSHMEMI_CHECK_INIT_STATUS();                                                           \
        NVSHMEM_API_NOT_SUPPORTED_WITH_LIMITED_MPG_RUNS();                                      \
        nvshmemi_reduce_on_stream<TYPE, RDXN_OPS_##OP>(team, dest, source, nreduce,             \
                                                       nvshmemi_state->my_stream);              \
        CUDA_RUNTIME_CHECK(cudaStreamSynchronize(nvshmemi_state->my_stream));                   \
        return 0;                                                                               \
    }

NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCE, and)
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCE, or)
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCE, xor)

NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCE, max)
NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCE, min)

NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCE, sum)
NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCE, prod)
