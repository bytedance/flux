/****
 * Copyright (c) 2016-2020, NVIDIA Corporation.  All rights reserved.
 *
 * Copyright 2011 Sandia Corporation. Under the terms of Contract
 * DE-AC04-94AL85000 with Sandia Corporation, the U.S.  Government
 * retains certain rights in this software.
 *
 * Copyright (c) 2017 Intel Corporation. All rights reserved.
 * This software is available to you under the BSD license.
 *
 * Portions of this file are derived from Sandia OpenSHMEM.
 *
 * See COPYRIGHT for license information
 ****/

#ifndef _UTIL_H
#define _UTIL_H

#include <stdio.h>
#include <stdint.h>
#include <cassert>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>
#include <errno.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdbool.h>
#include <stddef.h>
#include <sstream>
#include <tuple>
#include <vector>
#include <inttypes.h>
#include "device_host/nvshmem_types.h"
#include "non_abi/nvshmemx_error.h"
#include "internal/host/error_codes_internal.h"
#include "internal/host/debug.h"
#include "internal/host/nvshmemi_types.h"
#include "bootstrap_host_transport/env_defs_internal.h"
#include "non_abi/nvshmem_build_options.h"

#ifndef likely
#define likely(x) (__builtin_expect(!!(x), 1))
#endif

#ifndef unlikely
#define unlikely(x) (__builtin_expect(!!(x), 0))
#endif

#define NZ_DEBUG_JMP(status, err, label, ...)                                               \
    do {                                                                                    \
        if (unlikely(status != 0)) {                                                        \
            if (nvshmem_debug_level >= NVSHMEM_LOG_TRACE) {                                 \
                fprintf(stderr, "%s:%d: non-zero status: %d ", __FILE__, __LINE__, status); \
                fprintf(stderr, __VA_ARGS__);                                               \
            }                                                                               \
            status = err;                                                                   \
            goto label;                                                                     \
        }                                                                                   \
    } while (0)

#define CUDA_DRIVER_CHECK(cmd)                    \
    do {                                          \
        CUresult r = cmd;                         \
        cuGetErrorString(r, &p_err_str);          \
        if (unlikely(CUDA_SUCCESS != r)) {        \
            WARN("Cuda failure '%s'", p_err_str); \
            return NVSHMEMI_UNHANDLED_CUDA_ERROR; \
        }                                         \
    } while (false)

#define CUDA_CHECK(stmt)                                                                      \
    do {                                                                                      \
        CUresult result = (stmt);                                                             \
        cuGetErrorString(result, &p_err_str);                                                 \
        if (unlikely(CUDA_SUCCESS != result)) {                                               \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, p_err_str); \
            exit(-1);                                                                         \
        }                                                                                     \
        assert(CUDA_SUCCESS == result);                                                       \
    } while (0)

#define CUDA_RUNTIME_CHECK(stmt)                                                  \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (unlikely(cudaSuccess != result)) {                                    \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
            exit(-1);                                                             \
        }                                                                         \
        assert(cudaSuccess == result);                                            \
    } while (0)

#define CUDA_RUNTIME_CHECK_GOTO(stmt, res, label)                                 \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (unlikely(cudaSuccess != result)) {                                    \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
            res = NVSHMEMI_UNHANDLED_CUDA_ERROR;                                  \
            goto label;                                                           \
        }                                                                         \
    } while (0)

#define NCCL_CHECK(cmd)                                                   \
    do {                                                                  \
        ncclResult_t r = cmd;                                             \
        if (r != ncclSuccess) {                                           \
            printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, \
                   nccl_ftable.GetErrorString(r));                        \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

#define CUDA_RUNTIME_ERROR_STRING(result)                                         \
    do {                                                                          \
        if (unlikely(cudaSuccess != result)) {                                    \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
        }                                                                         \
    } while (0)

#define CUDA_DRIVER_ERROR_STRING(result)                                                      \
    do {                                                                                      \
        if (unlikely(CUDA_SUCCESS != result)) {                                               \
            cuGetErrorString(result, &p_err_str);                                             \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, p_err_str); \
        }                                                                                     \
    } while (0)

#define NVSHMEM_API_NOT_SUPPORTED_WITH_LIMITED_MPG_RUNS()                                        \
    if (nvshmemi_is_limited_mpg_run) {                                                           \
        fprintf(stderr,                                                                          \
                "[%s:%d] Called NVSHMEM API not supported with limited MPG (Multiple Processes " \
                "Per GPU) runs\n",                                                               \
                __FILE__, __LINE__);                                                             \
        exit(-1);                                                                                \
    }

#define NVSHMEMU_THREAD_CS_INIT nvshmemu_thread_cs_init
#define NVSHMEMU_THREAD_CS_ENTER nvshmemu_thread_cs_enter
#define NVSHMEMU_THREAD_CS_EXIT nvshmemu_thread_cs_exit
#define NVSHMEMU_THREAD_CS_FINALIZE nvshmemu_thread_cs_finalize

#define NVSHMEMU_MAPPED_PTR_TRANSLATE(toPtr, fromPtr, peer)                          \
    toPtr = (void *)((char *)(nvshmemi_state->heap_obj->get_local_pe_base()[peer]) + \
                     ((char *)fromPtr - (char *)(nvshmemi_device_state.heap_base)));

#define NVSHMEMU_UNMAPPED_PTR_PE_TRANSLATE(toPtr, fromPtr, peer)                                  \
    if (nvshmemi_device_state.enable_rail_opt) {                                                  \
        int proxy_pe = (peer / nvshmemi_state->npes_node) * nvshmemi_state->npes_node +           \
                       nvshmemi_state->mype_node;                                                 \
        toPtr = (void *)((char *)(nvshmemi_state->heap_obj->get_remote_pe_base()[proxy_pe]) +     \
                         +((int)(peer % nvshmemi_state->npes_node) - nvshmemi_state->mype_node) * \
                             nvshmemi_device_state.heap_size +                                    \
                         ((char *)fromPtr - (char *)(nvshmemi_device_state.heap_base)));          \
        peer = proxy_pe;                                                                          \
    } else {                                                                                      \
        toPtr = (void *)((char *)(nvshmemi_state->heap_obj->get_remote_pe_base()[peer]) +         \
                         ((char *)fromPtr - (char *)(nvshmemi_device_state.heap_base)));          \
    }

#define NVSHMEMU_UNMAPPED_PTR_TRANSLATE(toPtr, fromPtr, peer)                      \
    toPtr = (void *)((char *)(nvshmemi_device_state.peer_heap_base_remote[peer]) + \
                     ((char *)fromPtr - (char *)(nvshmemi_device_state.heap_base)));

void nvshmemu_thread_cs_init();
void nvshmemu_thread_cs_finalize();
void nvshmemu_thread_cs_enter();
void nvshmemu_thread_cs_exit();

int nvshmemu_get_num_gpus_per_node();

uint64_t getHostHash();
nvshmemResult_t nvshmemu_gethostname(char *hostname, int maxlen);
void setup_sig_handler();
char *nvshmemu_hexdump(void *ptr, size_t len);
void nvshmemu_debug_log_cpuset(int category, const char *thread_name);

#define NVSHMEMI_WRAPLEN 80
char *nvshmemu_wrap(const char *str, const size_t wraplen, const char *indent,
                    const int strip_backticks);

inline size_t nvshmemu_compute_log2(size_t x) {
    size_t tmp = x;
    size_t y = 0;
    while (tmp >> 1) {
        tmp >>= 1;
        y++;
    }

    return y;
}

extern const char *p_err_str;

extern struct nvshmemi_options_s nvshmemi_options;
extern int nvshmemi_job_connectivity;

enum { NVSHMEMI_OPTIONS_STYLE_INFO = 0, NVSHMEMI_OPTIONS_STYLE_RST };

int nvshmemi_options_init(void);
void nvshmemi_options_print(int style);
void nvshmemi_check_state_and_init();
void nvshmemi_ibgda_get_device_state(void **state);

#define NVSHMEMU_FOR_EACH(__index, count) \
    for (uint64_t __index = 0; __index < (uint64_t)(count); __index++)
#define NVSHMEMU_FOR_EACH_IF(x, count, condition, code) \
    NVSHMEMU_FOR_EACH(x, count) {                       \
        if ((condition)) {                              \
            (code);                                     \
        }                                               \
    }
#define NVSHMEMU_ROUND_UP(x, y) (((x) + (y)-1) / (y)) * (y)
#define NVSHMEMU_HOST_PTR_FREE(ptr) \
    do {                            \
        if ((ptr)) {                \
            free(ptr);              \
            ptr = NULL;             \
        }                           \
    } while (0)

#define NVSHMEMU_HOST_PTR_DELETE(ptr) \
    do {                              \
        if ((ptr) != nullptr) {       \
            delete (ptr);             \
            (ptr) = nullptr;          \
        }                             \
    } while (0)

/* Inspired from C++ bitset */
#define NVSHMEMU_IS_BIT_SET(bmp, bitpos) ((bmp) & (1 << (bitpos)))
#define NVSHMEMU_SET_BIT(bmp, bitpos) ((bmp) | (1 << (bitpos)))
#define NVSHMEMU_CLEAR_BIT(bmp, bitpos) ((bmp) & ~(1 << (bitpos)))

#endif
