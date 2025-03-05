/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See COPYRIGHT.txt for license information
 */

#ifndef _NVSHMEM_CONSTANTS_H_
#define _NVSHMEM_CONSTANTS_H_

#if not defined __CUDACC_RTC__
#include <limits.h>
#else
#include <cuda/std/climits>
#endif
#include "non_abi/nvshmem_version.h"

#define CHANNEL_BUF_SIZE (1 << CHANNEL_BUF_SIZE_LOG)
#define CHANNEL_BUF_SIZE_LOG 22
#define CHANNEL_ENTRY_BYTES 8

/* This is not the NVSHMEM release version, it is the supported OpenSHMEM spec version. */
#define NVSHMEM_MAJOR_VERSION 1
#define NVSHMEM_MINOR_VERSION 3

#define NVSHMEM_VENDOR_VERSION                                                   \
    ((NVSHMEM_VENDOR_MAJOR_VERSION)*10000 + (NVSHMEM_VENDOR_MINOR_VERSION)*100 + \
     (NVSHMEM_VENDOR_PATCH_VERSION))

#define NVSHMEMI_SUBST_AND_STRINGIFY_HELPER(S) #S
#define NVSHMEMI_SUBST_AND_STRINGIFY(S) NVSHMEMI_SUBST_AND_STRINGIFY_HELPER(S)

#define NVSHMEM_VENDOR_STRING \
    "NVSHMEM v"                                       \
            NVSHMEMI_SUBST_AND_STRINGIFY(NVSHMEM_VENDOR_MAJOR_VERSION) "."      \
            NVSHMEMI_SUBST_AND_STRINGIFY(NVSHMEM_VENDOR_MINOR_VERSION) "."      \
            NVSHMEMI_SUBST_AND_STRINGIFY(NVSHMEM_VENDOR_PATCH_VERSION)

#define NVSHMEM_MAX_NAME_LEN 256

enum nvshmemi_cmp_type {
    NVSHMEM_CMP_EQ = 0,
    NVSHMEM_CMP_NE,
    NVSHMEM_CMP_GT,
    NVSHMEM_CMP_LE,
    NVSHMEM_CMP_LT,
    NVSHMEM_CMP_GE,
    NVSHMEM_CMP_SENTINEL = INT_MAX,
};

enum nvshmemi_thread_support {
    NVSHMEM_THREAD_SINGLE = 0,
    NVSHMEM_THREAD_FUNNELED,
    NVSHMEM_THREAD_SERIALIZED,
    NVSHMEM_THREAD_MULTIPLE,
    NVSHMEM_THREAD_TYPE_SENTINEL = INT_MAX,
};

enum {
    PROXY_GLOBAL_EXIT_NOT_REQUESTED = 0,
    PROXY_GLOBAL_EXIT_INIT,
    PROXY_GLOBAL_EXIT_REQUESTED,
    PROXY_GLOBAL_EXIT_FINISHED,
    PROXY_GLOBAL_EXIT_MAX_STATE = INT_MAX
};

#define PROXY_DMA_REQ_BYTES 32
#define PROXY_AMO_REQ_BYTES 40
#define PROXY_INLINE_REQ_BYTES 24

enum {
    NVSHMEM_STATUS_NOT_INITIALIZED = 0,
    NVSHMEM_STATUS_IS_BOOTSTRAPPED,
    NVSHMEM_STATUS_IS_INITIALIZED,
    NVSHMEM_STATUS_LIMITED_MPG,
    NVSHMEM_STATUS_FULL_MPG,
    NVSHMEM_STATUS_INVALID = INT_MAX,
};

#endif
