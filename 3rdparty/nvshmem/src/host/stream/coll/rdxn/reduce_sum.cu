/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "reduce_common.cuh"
#include "internal/non_abi/nvshmemi_h_to_d_coll_defs.cuh"

REPT_FOR_BITWISE_TYPES(INSTANTIATE_NVSHMEMI_CALL_RDXN_ON_STREAM_KERNEL, SUM)
REPT_FOR_FLOATING_TYPES(INSTANTIATE_NVSHMEMI_CALL_RDXN_ON_STREAM_KERNEL, SUM)
