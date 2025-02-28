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

#ifndef UTILS
#define UTILS
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda.h>
#ifdef NVSHMEMTEST_MPI_SUPPORT
#include "mpi.h"
#endif
#ifdef NVSHMEMTEST_SHMEM_SUPPORT
#include "shmem.h"
#include "shmemx.h"
#endif
#include "nvshmem.h"
#include "nvshmemx.h"

using namespace std;

#undef CUDA_CHECK
#define CUDA_CHECK(stmt)                                                          \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (cudaSuccess != result) {                                              \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
            exit(-1);                                                             \
        }                                                                         \
        assert(cudaSuccess == result);                                            \
    } while (0)

#define CU_CHECK(stmt)                                                                     \
    do {                                                                                   \
        CUresult result = (stmt);                                                          \
        char str[1024];                                                                    \
        if (CUDA_SUCCESS != result) {                                                      \
            CUresult ret = cuGetErrorString(result, (const char **)&str);                  \
            fprintf(stderr, "[%s:%d] cuda failed with (%d) %s \n", __FILE__, __LINE__,     \
                    (int)result, (ret != CUDA_SUCCESS) ? "cuGetErrorString failed" : str); \
            exit(-1);                                                                      \
        }                                                                                  \
        assert(CUDA_SUCCESS == result);                                                    \
    } while (0)

#define ERROR_EXIT(...)                                                  \
    do {                                                                 \
        fprintf(stderr, "%s:%s:%d: ", __FILE__, __FUNCTION__, __LINE__); \
        fprintf(stderr, __VA_ARGS__);                                    \
        exit(-1);                                                        \
    } while (0)

#define ERROR_PRINT(...)                                                 \
    do {                                                                 \
        fprintf(stderr, "%s:%s:%d: ", __FILE__, __FUNCTION__, __LINE__); \
        fprintf(stderr, __VA_ARGS__);                                    \
    } while (0)

#undef WARN_PRINT
#define WARN_PRINT(...)                                                  \
    do {                                                                 \
        fprintf(stdout, "%s:%s:%d: ", __FILE__, __FUNCTION__, __LINE__); \
        fprintf(stdout, __VA_ARGS__);                                    \
    } while (0)

#ifdef _NVSHMEM_DEBUG
#define DEBUG_PRINT(...) fprintf(stderr, __VA_ARGS__);
#else
#define DEBUG_PRINT(...) \
    do {                 \
    } while (0)
#endif

#define SHMEM_WRK_VALUE 0
#define MAX_ELEMS (1 * 1024 * 1024)
#define MAX_ELEMS_LOG 20
#define MS_TO_S 1000
#define B_TO_GB (1 << 30)

extern __device__ int clockrate;

void init_wrapper(int *c, char ***v);
void finalize_wrapper();
void alloc_tables(void ***table_mem, int num_tables, int num_entries_per_table);
void free_tables(void **tables, int num_tables);
void print_table_basic(const char *job_name, const char *subjob_name, const char *var_name,
                       const char *output_var, const char *units, const char plus_minus,
                       uint64_t *size, double *value, int num_entries);
void print_table_v1(const char *job_name, const char *subjob_name, const char *var_name,
                    const char *output_var, const char *units, const char plus_minus,
                    uint64_t *size, double *value, int num_entries);
void print_table_v2(const char *job_name, const char *subjob_name, const char *var_name,
                    const char *output_var, const char *units, const char plus_minus,
                    uint64_t *size, double **value, int num_entries);
#endif
