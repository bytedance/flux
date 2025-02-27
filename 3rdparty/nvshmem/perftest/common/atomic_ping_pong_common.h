/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See COPYRIGHT.txt for license information
 */

#ifndef _ATOMIC_PING_PONG_COMMON_H_
#define _ATOMIC_PING_PONG_COMMON_H_

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include "utils.h"

void atomic_op_parse(char *v[], nvshmemi_amo_t *atomic_op) {
    char *amo = v[1];
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
    fprintf(stderr, "Please supply an atomic operation to perform. Valid options are: \n");
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

#define DEFINE_PING_PONG_TEST_FOR_AMO_NO_ARG(TYPE, TYPE_NAME, AMO, COMPARE_EXPR)               \
    __global__ void ping_pong_##TYPE_NAME##_##AMO(TYPE *flag_d, int pe, int iter) {            \
        int i, peer;                                                                           \
                                                                                               \
        assert(1 == blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y * gridDim.z); \
        peer = !pe;                                                                            \
                                                                                               \
        for (i = 0; i < iter; i++) {                                                           \
            if (pe) {                                                                          \
                nvshmem_##TYPE_NAME##_wait_until(flag_d, NVSHMEM_CMP_EQ, COMPARE_EXPR);        \
                nvshmem_##TYPE_NAME##_atomic_##AMO(flag_d, peer);                              \
            } else {                                                                           \
                nvshmem_##TYPE_NAME##_atomic_##AMO(flag_d, peer);                              \
                nvshmem_##TYPE_NAME##_wait_until(flag_d, NVSHMEM_CMP_EQ, COMPARE_EXPR);        \
            }                                                                                  \
        }                                                                                      \
        nvshmem_quiet();                                                                       \
    }

#define DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(TYPE, TYPE_NAME, AMO, COMPARE_EXPR, SET_EXPR)    \
    __global__ void ping_pong_##TYPE_NAME##_##AMO(TYPE *flag_d, int pe, int iter, TYPE value,  \
                                                  TYPE cmp) {                                  \
        int i, peer;                                                                           \
                                                                                               \
        assert(1 == blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y * gridDim.z); \
        peer = !pe;                                                                            \
                                                                                               \
        for (i = 0; i < iter; i++) {                                                           \
            if (pe) {                                                                          \
                nvshmem_##TYPE_NAME##_wait_until(flag_d, NVSHMEM_CMP_EQ, COMPARE_EXPR);        \
                nvshmem_##TYPE_NAME##_atomic_##AMO(flag_d, SET_EXPR, peer);                    \
            } else {                                                                           \
                nvshmem_##TYPE_NAME##_atomic_##AMO(flag_d, SET_EXPR, peer);                    \
                nvshmem_##TYPE_NAME##_wait_until(flag_d, NVSHMEM_CMP_EQ, COMPARE_EXPR);        \
            }                                                                                  \
        }                                                                                      \
        nvshmem_quiet();                                                                       \
    }

#define DEFINE_PING_PONG_TEST_FOR_AMO_TWO_ARG(TYPE, TYPE_NAME, AMO, COMPARE_EXPR, SET_EXPR)    \
    __global__ void ping_pong_##TYPE_NAME##_##AMO(TYPE *flag_d, int pe, int iter, TYPE value,  \
                                                  TYPE cmp) {                                  \
        int i, peer;                                                                           \
                                                                                               \
        assert(1 == blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y * gridDim.z); \
        peer = !pe;                                                                            \
                                                                                               \
        for (i = 0; i < iter; i++) {                                                           \
            if (pe) {                                                                          \
                nvshmem_##TYPE_NAME##_wait_until(flag_d, NVSHMEM_CMP_EQ, SET_EXPR);            \
                nvshmem_##TYPE_NAME##_atomic_##AMO(flag_d, COMPARE_EXPR, SET_EXPR, peer);      \
            } else {                                                                           \
                nvshmem_##TYPE_NAME##_atomic_##AMO(flag_d, COMPARE_EXPR, SET_EXPR, peer);      \
                nvshmem_##TYPE_NAME##_wait_until(flag_d, NVSHMEM_CMP_EQ, SET_EXPR);            \
            }                                                                                  \
        }                                                                                      \
        nvshmem_quiet();                                                                       \
    }

#define MAIN_SETUP(c, v, mype, npes, flag_d, stream, h_size_arr, h_tables, h_lat, atomic_op)    \
    do {                                                                                        \
        if (c == 1) {                                                                           \
            fprintf(stderr, "You must pass a valid atomic operation name to the exeutable.\n"); \
            atomic_usage();                                                                     \
        } else {                                                                                \
            *atomic_op = NVSHMEMI_AMO_ACK;                                                      \
            atomic_op_parse(v, atomic_op);                                                      \
            if (*atomic_op == NVSHMEMI_AMO_ACK) {                                               \
                fprintf(stderr, "Error, No valid atomic supplied, exiting.\n");                 \
            }                                                                                   \
        }                                                                                       \
        /* Ignore the initial atomic argument if they pass MPI args. */                         \
        c--;                                                                                    \
        v++;                                                                                    \
        init_wrapper(&c, &v);                                                                   \
                                                                                                \
        mype = nvshmem_my_pe();                                                                 \
        npes = nvshmem_n_pes();                                                                 \
                                                                                                \
        if (npes != 2) {                                                                        \
            fprintf(stderr, "This test requires exactly two processes \n");                     \
            finalize_wrapper();                                                                 \
            exit(-1);                                                                           \
        }                                                                                       \
                                                                                                \
        alloc_tables(&h_tables, 2, 1);                                                          \
        h_size_arr = (uint64_t *)h_tables[0];                                                   \
        h_lat = (double *)h_tables[1];                                                          \
                                                                                                \
        flag_d = nvshmem_malloc(sizeof(uint64_t));                                              \
        CUDA_CHECK(cudaMemset(flag_d, 0, sizeof(uint64_t)));                                    \
                                                                                                \
        CUDA_CHECK(cudaStreamCreate(&stream));                                                  \
                                                                                                \
        nvshmem_barrier_all();                                                                  \
                                                                                                \
        CUDA_CHECK(cudaDeviceSynchronize());                                                    \
                                                                                                \
        if (mype == 0) {                                                                        \
            printf("Note: This test measures full round-trip latency\n");                       \
        }                                                                                       \
    } while (0)

#define RUN_TEST_WITHOUT_ARG(TYPE, TYPE_NAME, AMO, flag_d, mype, iter, skip, h_lat, h_size_arr, \
                             flag_init)                                                         \
    do {                                                                                        \
        int size = sizeof(TYPE);                                                                \
                                                                                                \
        int status = 0;                                                                         \
        h_size_arr[0] = size;                                                                   \
        void *args_1[] = {&flag_d, &mype, &skip};                                               \
        void *args_2[] = {&flag_d, &mype, &iter};                                               \
                                                                                                \
        float milliseconds;                                                                     \
        cudaEvent_t start, stop;                                                                \
        cudaEventCreate(&start);                                                                \
        cudaEventCreate(&stop);                                                                 \
        TYPE flag_init_var = flag_init;                                                         \
                                                                                                \
        CUDA_CHECK(cudaDeviceSynchronize());                                                    \
        CUDA_CHECK(cudaMemcpy(flag_d, &flag_init_var, sizeof(TYPE), cudaMemcpyHostToDevice));   \
        nvshmem_barrier_all();                                                                  \
                                                                                                \
        cudaEventRecord(start, stream);                                                         \
        status = nvshmemx_collective_launch((const void *)ping_pong_##TYPE_NAME##_##AMO, 1, 1,  \
                                            args_1, 0, stream);                                 \
        if (status != NVSHMEMX_SUCCESS) {                                                       \
            fprintf(stderr, "shmemx_collective_launch failed %d \n", status);                   \
            exit(-1);                                                                           \
        }                                                                                       \
        cudaEventRecord(stop, stream);                                                          \
                                                                                                \
        cudaStreamSynchronize(stream);                                                          \
                                                                                                \
        nvshmem_barrier_all();                                                                  \
        CUDA_CHECK(cudaMemcpy(flag_d, &flag_init_var, sizeof(TYPE), cudaMemcpyHostToDevice));   \
        cudaEventRecord(start, stream);                                                         \
        status = nvshmemx_collective_launch((const void *)ping_pong_##TYPE_NAME##_##AMO, 1, 1,  \
                                            args_2, 0, stream);                                 \
        if (status != NVSHMEMX_SUCCESS) {                                                       \
            fprintf(stderr, "shmemx_collective_launch failed %d  \n", status);                  \
            exit(-1);                                                                           \
        }                                                                                       \
        cudaEventRecord(stop, stream);                                                          \
        CUDA_CHECK(cudaStreamSynchronize(stream));                                              \
        /* give latency in us */                                                                \
        cudaEventElapsedTime(&milliseconds, start, stop);                                       \
        h_lat[0] = (milliseconds * 1000) / iter;                                                \
                                                                                                \
        nvshmem_barrier_all();                                                                  \
                                                                                                \
        if (mype == 0) {                                                                        \
            print_table_v1("shmem_at_" #TYPE "_" #AMO "_ping_lat", "None", "size (Bytes)",      \
                           "latency", "us", '-', h_size_arr, h_lat, 1);                         \
        }                                                                                       \
                                                                                                \
        CUDA_CHECK(cudaDeviceSynchronize());                                                    \
                                                                                                \
    } while (0)

#define RUN_TEST_WITH_ARG(TYPE, TYPE_NAME, AMO, flag_d, mype, iter, skip, h_lat, h_size_arr, val, \
                          cmp, flag_init)                                                         \
    do {                                                                                          \
        int size = sizeof(TYPE);                                                                  \
        TYPE compare, value, flag_init_var;                                                       \
                                                                                                  \
        int status = 0;                                                                           \
        h_size_arr[0] = size;                                                                     \
        void *args_1[] = {&flag_d, &mype, &skip, &value, &compare};                               \
        void *args_2[] = {&flag_d, &mype, &iter, &value, &compare};                               \
                                                                                                  \
        float milliseconds;                                                                       \
        cudaEvent_t start, stop;                                                                  \
        cudaEventCreate(&start);                                                                  \
        cudaEventCreate(&stop);                                                                   \
                                                                                                  \
        compare = cmp;                                                                            \
        value = val;                                                                              \
        flag_init_var = flag_init;                                                                \
                                                                                                  \
        CUDA_CHECK(cudaDeviceSynchronize());                                                      \
        CUDA_CHECK(cudaMemcpy(flag_d, &flag_init_var, sizeof(TYPE), cudaMemcpyHostToDevice));     \
        nvshmem_barrier_all();                                                                    \
                                                                                                  \
        status = nvshmemx_collective_launch((const void *)ping_pong_##TYPE_NAME##_##AMO, 1, 1,    \
                                            args_1, 0, stream);                                   \
        if (status != NVSHMEMX_SUCCESS) {                                                         \
            fprintf(stderr, "shmemx_collective_launch failed %d \n", status);                     \
            exit(-1);                                                                             \
        }                                                                                         \
                                                                                                  \
        cudaStreamSynchronize(stream);                                                            \
                                                                                                  \
        nvshmem_barrier_all();                                                                    \
        CUDA_CHECK(cudaMemcpy(flag_d, &flag_init_var, sizeof(TYPE), cudaMemcpyHostToDevice));     \
        cudaEventRecord(start, stream);                                                           \
        status = nvshmemx_collective_launch((const void *)ping_pong_##TYPE_NAME##_##AMO, 1, 1,    \
                                            args_2, 0, stream);                                   \
        if (status != NVSHMEMX_SUCCESS) {                                                         \
            fprintf(stderr, "shmemx_collective_launch failed %d  \n", status);                    \
            exit(-1);                                                                             \
        }                                                                                         \
        cudaEventRecord(stop, stream);                                                            \
        cudaStreamSynchronize(stream);                                                            \
        /* give latency in us */                                                                  \
        cudaEventElapsedTime(&milliseconds, start, stop);                                         \
        h_lat[0] = (milliseconds * 1000) / iter;                                                  \
                                                                                                  \
        nvshmem_barrier_all();                                                                    \
                                                                                                  \
        if (mype == 0) {                                                                          \
            print_table_v1("shmem_at_" #TYPE "_" #AMO "_lat", "None", "size (Bytes)", "latency",  \
                           "us", '-', h_size_arr, h_lat, 1);                                      \
        }                                                                                         \
                                                                                                  \
        CUDA_CHECK(cudaDeviceSynchronize());                                                      \
                                                                                                  \
    } while (0)

#define MAIN_CLEANUP(flag_d, stream, h_tables, num_entries) \
    do {                                                    \
        if (flag_d) nvshmem_free(flag_d);                   \
        cudaStreamDestroy(stream);                          \
        free_tables(h_tables, 2);                           \
        finalize_wrapper();                                 \
    } while (0);

#endif /* _ATOMIC_PING_PONG_COMMON_H_ */
