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

#ifndef COLL_TEST_H
#define COLL_TEST_H
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#ifdef NVSHMEMTEST_MPI_SUPPORT
#include "mpi.h"
#endif
#ifdef NVSHMEMTEST_SHMEM_SUPPORT
#include "shmem.h"
#include "shmemx.h"
#endif
#include "utils.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <sys/time.h>

using namespace std;

#define MAX_ITERS 100
#define MAX_SKIP 10
#define BARRIER_MAX_ITERS 1000
#define BARRIER_MAX_SKIP 10
#define MAX_NPES 128
#define TEST_NUM_TPB_BLOCK 256

typedef struct run_opt {
    int run_thread;
    int run_warp;
    int run_block;
} run_opt_t;

#define cuda_check_error()                                                                   \
    {                                                                                        \
        cudaError_t e = cudaGetLastError();                                                  \
        if (cudaSuccess != e) {                                                              \
            printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(-1);                                                                        \
        }                                                                                    \
    }

#define PROCESS_OPTS(run_options)                          \
    do {                                                   \
        int opt;                                           \
        run_options.run_thread = 1;                        \
        run_options.run_warp = 1;                          \
        run_options.run_block = 1;                         \
                                                           \
        while ((opt = getopt(argc, argv, "twba")) != -1) { \
            switch (opt) {                                 \
                case 't':                                  \
                    run_options.run_thread = 1;            \
                    run_options.run_warp = 0;              \
                    run_options.run_block = 0;             \
                    break;                                 \
                case 'w':                                  \
                    run_options.run_warp = 1;              \
                    run_options.run_thread = 0;            \
                    run_options.run_block = 0;             \
                    break;                                 \
                case 'b':                                  \
                    run_options.run_block = 1;             \
                    run_options.run_thread = 0;            \
                    run_options.run_warp = 0;              \
                    break;                                 \
                case 'a':                                  \
                default:                                   \
                    run_options.run_thread = 1;            \
                    run_options.run_warp = 1;              \
                    run_options.run_block = 1;             \
                    break;                                 \
            }                                              \
        }                                                  \
    } while (0)

int page_size_roundoff(int value) {
    int page_sz = getpagesize();
    return ((value + page_sz - 1) / page_sz) * page_sz;
}

#endif /*COLL_TEST_H*/
