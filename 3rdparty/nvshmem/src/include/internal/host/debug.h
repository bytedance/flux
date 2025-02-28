/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See COPYRIGHT.txt for license information
 */

#ifndef _DEBUG_H_
#define _DEBUG_H_

#include "non_abi/nvshmem_build_options.h"  // IWYU pragma: keep
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <ctype.h>

extern int nvshmem_debug_level;
extern uint64_t nvshmem_debug_mask;
extern pthread_mutex_t nvshmem_debug_output_lock;
extern FILE *nvshmem_debug_file;

typedef enum {
    NVSHMEM_LOG_NONE = 0,
    NVSHMEM_LOG_VERSION = 1,
    NVSHMEM_LOG_WARN = 2,
    NVSHMEM_LOG_INFO = 3,
    NVSHMEM_LOG_ABORT = 4,
    NVSHMEM_LOG_TRACE = 5
} nvshmem_debug_log_level;
typedef enum {
    NVSHMEM_INIT = 1,
    NVSHMEM_COLL = 2,
    NVSHMEM_P2P = 4,
    NVSHMEM_PROXY = 8,
    NVSHMEM_TRANSPORT = 16,
    NVSHMEM_MEM = 32,
    NVSHMEM_BOOTSTRAP = 64,
    NVSHMEM_TOPO = 128,
    NVSHMEM_UTIL = 256,
    NVSHMEM_TEAM = 512,
    NVSHMEM_ALL = ~0
} nvshmem_debug_log_sub_sys;

extern "C" {
void nvshmem_debug_log(nvshmem_debug_log_level level, unsigned long flags, const char *filefunc,
                       int line, const char *fmt, ...);
}

#define WARN(...) nvshmem_debug_log(NVSHMEM_LOG_WARN, NVSHMEM_ALL, __FILE__, __LINE__, __VA_ARGS__)
#define INFO(FLAGS, ...) \
    nvshmem_debug_log(NVSHMEM_LOG_INFO, (FLAGS), __func__, __LINE__, __VA_ARGS__)

#ifdef NVSHMEM_TRACE
#include <chrono>  // IWYU pragma: keep
#define TRACE(FLAGS, ...) \
    nvshmem_debug_log(NVSHMEM_LOG_TRACE, (FLAGS), __func__, __LINE__, __VA_ARGS__)
extern std::chrono::high_resolution_clock::time_point nvshmem_epoch;
#else
#define TRACE(...)
#endif

static int strcmp_case_insensitive(const char *a, const char *b) {
    int ca, cb;
    do {
        ca = (unsigned char)*a++;
        cb = (unsigned char)*b++;
        ca = tolower(toupper(ca));
        cb = tolower(toupper(cb));
    } while (ca == cb && ca != '\0');
    return ca - cb;
}

#endif
