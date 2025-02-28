/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "reducescatter_common.cuh"
#include "internal/non_abi/nvshmemi_h_to_d_coll_defs.cuh"

REPT_FOR_BITWISE_TYPES(INSTANTIATE_NVSHMEMI_CALL_REDUCESCATTER_ON_STREAM_KERNEL, MAX)
REPT_FOR_FLOATING_TYPES(INSTANTIATE_NVSHMEMI_CALL_REDUCESCATTER_ON_STREAM_KERNEL, MAX)
