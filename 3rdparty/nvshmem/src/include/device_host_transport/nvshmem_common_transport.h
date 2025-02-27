/*
 * Copyright (c) 2016-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef __NVSHMEM_TRANSPORT_COMMON_H
#define __NVSHMEM_TRANSPORT_COMMON_H

#if not defined __CUDACC_RTC__
#include <stdint.h>
#include <limits.h>
#else
#include <cuda/std/cstdint>
#include <cuda/std/climits>
#endif

typedef enum {
    NVSHMEMI_OP_PUT = 1,
    NVSHMEMI_OP_P,
    NVSHMEMI_OP_PUT_SIGNAL,
    NVSHMEMI_OP_GET,
    NVSHMEMI_OP_G,
    NVSHMEMI_OP_FENCE,
    NVSHMEMI_OP_AMO,
    NVSHMEMI_OP_QUIET,
    NVSHMEMI_OP_SENTINEL = INT_MAX,
} nvshmemi_op_t;

typedef enum {
    NVSHMEMI_AMO_ACK = 1,
    NVSHMEMI_AMO_INC,
    NVSHMEMI_AMO_SET,
    NVSHMEMI_AMO_ADD,
    NVSHMEMI_AMO_AND,
    NVSHMEMI_AMO_OR,
    NVSHMEMI_AMO_XOR,
    NVSHMEMI_AMO_SIGNAL,
    NVSHMEM_SIGNAL_SET,
    NVSHMEM_SIGNAL_ADD,
    NVSHMEMI_AMO_SIGNAL_SET = NVSHMEM_SIGNAL_SET,  // Note - NVSHMEM_SIGNAL_SET == 9
    NVSHMEMI_AMO_SIGNAL_ADD = NVSHMEM_SIGNAL_ADD,  // Note - NVSHMEM_SIGNAL_ADD == 10
    NVSHMEMI_AMO_END_OF_NONFETCH,                  // end of nonfetch atomics
    NVSHMEMI_AMO_FETCH,
    NVSHMEMI_AMO_FETCH_INC,
    NVSHMEMI_AMO_FETCH_ADD,
    NVSHMEMI_AMO_FETCH_AND,
    NVSHMEMI_AMO_FETCH_OR,
    NVSHMEMI_AMO_FETCH_XOR,
    NVSHMEMI_AMO_SWAP,
    NVSHMEMI_AMO_COMPARE_SWAP,
    NVSHMEMI_AMO_OP_SENTINEL = INT_MAX,
} nvshmemi_amo_t;

typedef struct {
    volatile uint64_t data;
    volatile uint64_t flag;
} g_elem_t;

#endif
