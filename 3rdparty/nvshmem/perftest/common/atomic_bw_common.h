/*
 * Copyright (c) 2021, NVIDIA CORPORATION   All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto   Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See COPYRIGHT.txt for license information
 */

#ifndef _ATOMIC_BW_COMMON_H_
#define _ATOMIC_BW_COMMON_H_

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <getopt.h>
#include "utils.h"

#define MAX_ITERS 10
#define MAX_SKIP 10
#define THREADS 1024
#define BLOCKS 4
#define MAX_MSG_SIZE 64 * 1024

void atomic_op_parse(char *amo, nvshmemi_amo_t *atomic_op) {
    size_t string_length = strnlen(amo, 20);

    if (strncmp(amo, "inc", string_length) == 0) {
        *atomic_op = NVSHMEMI_AMO_INC;
    } else if (strncmp(amo, "fetch_inc", string_length) == 0) {
        *atomic_op = NVSHMEMI_AMO_FETCH_INC;
    } else if (strncmp(amo, "set", string_length) == 0) {
        *atomic_op = NVSHMEMI_AMO_SET;
    } else if (strncmp(amo, "add", string_length) == 0) {
        *atomic_op = NVSHMEMI_AMO_ADD;
    } else if (strncmp(amo, "fetch_add", string_length) == 0) {
        *atomic_op = NVSHMEMI_AMO_FETCH_ADD;
    } else if (strncmp(amo, "and", string_length) == 0) {
        *atomic_op = NVSHMEMI_AMO_AND;
    } else if (strncmp(amo, "fetch_and", string_length) == 0) {
        *atomic_op = NVSHMEMI_AMO_FETCH_AND;
    } else if (strncmp(amo, "or", string_length) == 0) {
        *atomic_op = NVSHMEMI_AMO_OR;
    } else if (strncmp(amo, "fetch_or", string_length) == 0) {
        *atomic_op = NVSHMEMI_AMO_FETCH_OR;
    } else if (strncmp(amo, "xor", string_length) == 0) {
        *atomic_op = NVSHMEMI_AMO_XOR;
    } else if (strncmp(amo, "fetch_xor", string_length) == 0) {
        *atomic_op = NVSHMEMI_AMO_FETCH_XOR;
    } else if (strncmp(amo, "swap", string_length) == 0) {
        *atomic_op = NVSHMEMI_AMO_SWAP;
    } else if (strncmp(amo, "compare_swap", string_length) == 0) {
        *atomic_op = NVSHMEMI_AMO_COMPARE_SWAP;
    } else {
        *atomic_op = NVSHMEMI_AMO_ACK;
    }
}

void atomic_usage(void) {
    fprintf(stderr, "Please supply an atomic operation to perform. Valid options are:   \n");
    fprintf(stderr, "inc\n");
    fprintf(stderr, "fetch_inc\n");
    fprintf(stderr, "set\n");
    fprintf(stderr, "add\n");
    fprintf(stderr, "fetch_add\n");
    fprintf(stderr, "and\n");
    fprintf(stderr, "fetch_and\n");
    fprintf(stderr, "or\n");
    fprintf(stderr, "fetch_or\n");
    fprintf(stderr, "xor\n");
    fprintf(stderr, "fetch_xor\n");
    fprintf(stderr, "swap\n");
    fprintf(stderr, "compare_swap\n");
}

#define DEFINE_ATOMIC_BW_FN_NO_ARG(AMO)                                                            \
    __global__ void atomic_##AMO##_bw(uint64_t *data_d, volatile unsigned int *counter_d, int len, \
                                      int pe, int iter) {                                          \
        int i, j, peer, tid, slice;                                                                \
        unsigned int counter;                                                                      \
        int threads = gridDim.x * blockDim.x;                                                      \
        tid = blockIdx.x * blockDim.x + threadIdx.x;                                               \
                                                                                                   \
        peer = !pe;                                                                                \
        slice = threads;                                                                           \
                                                                                                   \
        for (i = 0; i < iter; i++) {                                                               \
            for (j = 0; j < len - slice; j += slice) {                                             \
                int idx = j + tid;                                                                 \
                nvshmem_uint64_atomic_##AMO(data_d + idx, peer);                                   \
                __syncthreads();                                                                   \
            }                                                                                      \
                                                                                                   \
            int idx = j + tid;                                                                     \
            if (idx < len) nvshmem_uint64_atomic_##AMO(data_d + idx, peer);                        \
                                                                                                   \
            /* synchronizing across blocks */                                                      \
            __syncthreads();                                                                       \
                                                                                                   \
            if (!threadIdx.x) {                                                                    \
                __threadfence();                                                                   \
                counter = atomicInc((unsigned int *)counter_d, UINT_MAX);                          \
                if (counter == (gridDim.x * (i + 1) - 1)) {                                        \
                    *(counter_d + 1) += 1;                                                         \
                }                                                                                  \
                while (*(counter_d + 1) != i + 1)                                                  \
                    ;                                                                              \
            }                                                                                      \
                                                                                                   \
            __syncthreads();                                                                       \
        }                                                                                          \
                                                                                                   \
        /* synchronizing across blocks */                                                          \
        __syncthreads();                                                                           \
                                                                                                   \
        if (!threadIdx.x) {                                                                        \
            __threadfence();                                                                       \
            counter = atomicInc((unsigned int *)counter_d, UINT_MAX);                              \
            if (counter == (gridDim.x * (i + 1) - 1)) {                                            \
                nvshmem_quiet();                                                                   \
                *(counter_d + 1) += 1;                                                             \
            }                                                                                      \
            while (*(counter_d + 1) != i + 1)                                                      \
                ;                                                                                  \
        }                                                                                          \
    }

#define DEFINE_ATOMIC_BW_FN_ONE_ARG(AMO, SET_EXPR)                                                 \
    __global__ void atomic_##AMO##_bw(uint64_t *data_d, volatile unsigned int *counter_d, int len, \
                                      int pe, int iter) {                                          \
        int i, j, peer, tid, slice;                                                                \
        unsigned int counter;                                                                      \
        int threads = gridDim.x * blockDim.x;                                                      \
        tid = blockIdx.x * blockDim.x + threadIdx.x;                                               \
                                                                                                   \
        peer = !pe;                                                                                \
        slice = threads;                                                                           \
                                                                                                   \
        for (i = 0; i < iter; i++) {                                                               \
            for (j = 0; j < len - slice; j += slice) {                                             \
                int idx = j + tid;                                                                 \
                nvshmem_uint64_atomic_##AMO(data_d + idx, SET_EXPR, peer);                         \
                __syncthreads();                                                                   \
            }                                                                                      \
                                                                                                   \
            int idx = j + tid;                                                                     \
            if (idx < len) nvshmem_uint64_atomic_##AMO(data_d + idx, SET_EXPR, peer);              \
                                                                                                   \
            /* synchronizing across blocks */                                                      \
            __syncthreads();                                                                       \
                                                                                                   \
            if (!threadIdx.x) {                                                                    \
                __threadfence();                                                                   \
                counter = atomicInc((unsigned int *)counter_d, UINT_MAX);                          \
                if (counter == (gridDim.x * (i + 1) - 1)) {                                        \
                    *(counter_d + 1) += 1;                                                         \
                }                                                                                  \
                while (*(counter_d + 1) != i + 1)                                                  \
                    ;                                                                              \
            }                                                                                      \
                                                                                                   \
            __syncthreads();                                                                       \
        }                                                                                          \
                                                                                                   \
        /* synchronizing across blocks */                                                          \
        __syncthreads();                                                                           \
                                                                                                   \
        if (!threadIdx.x) {                                                                        \
            __threadfence();                                                                       \
            counter = atomicInc((unsigned int *)counter_d, UINT_MAX);                              \
            if (counter == (gridDim.x * (i + 1) - 1)) {                                            \
                nvshmem_quiet();                                                                   \
                *(counter_d + 1) += 1;                                                             \
            }                                                                                      \
            while (*(counter_d + 1) != i + 1)                                                      \
                ;                                                                                  \
        }                                                                                          \
                                                                                                   \
        __syncthreads();                                                                           \
    }

#define DEFINE_ATOMIC_BW_FN_TWO_ARG(AMO, COMPARE_EXPR, SET_EXPR)                                   \
    __global__ void atomic_##AMO##_bw(uint64_t *data_d, volatile unsigned int *counter_d, int len, \
                                      int pe, int iter) {                                          \
        int i, j, peer, tid, slice;                                                                \
        unsigned int counter;                                                                      \
        int threads = gridDim.x * blockDim.x;                                                      \
        tid = blockIdx.x * blockDim.x + threadIdx.x;                                               \
                                                                                                   \
        peer = !pe;                                                                                \
        slice = threads;                                                                           \
                                                                                                   \
        for (i = 0; i < iter; i++) {                                                               \
            for (j = 0; j < len - slice; j += slice) {                                             \
                int idx = j + tid;                                                                 \
                nvshmem_uint64_atomic_##AMO(data_d + idx, COMPARE_EXPR, SET_EXPR, peer);           \
                __syncthreads();                                                                   \
            }                                                                                      \
                                                                                                   \
            int idx = j + tid;                                                                     \
            if (idx < len) {                                                                       \
                nvshmem_uint64_atomic_##AMO(data_d + idx, COMPARE_EXPR, SET_EXPR, peer);           \
            }                                                                                      \
                                                                                                   \
            /* synchronizing across blocks */                                                      \
            __syncthreads();                                                                       \
                                                                                                   \
            if (!threadIdx.x) {                                                                    \
                __threadfence();                                                                   \
                counter = atomicInc((unsigned int *)counter_d, UINT_MAX);                          \
                if (counter == (gridDim.x * (i + 1) - 1)) {                                        \
                    *(counter_d + 1) += 1;                                                         \
                }                                                                                  \
                while (*(counter_d + 1) != i + 1)                                                  \
                    ;                                                                              \
            }                                                                                      \
                                                                                                   \
            __syncthreads();                                                                       \
        }                                                                                          \
                                                                                                   \
        /* synchronizing across blocks */                                                          \
        __syncthreads();                                                                           \
                                                                                                   \
        if (!threadIdx.x) {                                                                        \
            __threadfence();                                                                       \
            counter = atomicInc((unsigned int *)counter_d, UINT_MAX);                              \
            if (counter == (gridDim.x * (i + 1) - 1)) {                                            \
                nvshmem_quiet();                                                                   \
                *(counter_d + 1) += 1;                                                             \
            }                                                                                      \
            while (*(counter_d + 1) != i + 1)                                                      \
                ;                                                                                  \
        }                                                                                          \
    }

#endif /* _ATOMIC_BW_COMMON_H_ */
