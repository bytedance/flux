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

#include "coll_test.h"
#define LARGEST_DT uint64_t
int coll_max_iters = MAX_ITERS;

#define RUN_RDXN(TYPENAME, TYPE, OP, team, d_source, h_source, d_dest, h_dest, num_elems, array, \
                 stream)                                                                         \
    do {                                                                                         \
        int iters;                                                                               \
        int skip = MAX_SKIP;                                                                     \
        float ms = 0.0f;                                                                         \
        cudaEvent_t start_event, stop_event;                                                     \
        CUDA_CHECK(cudaEventCreate(&start_event));                                               \
        CUDA_CHECK(cudaEventCreate(&stop_event));                                                \
        for (iters = 0; iters < coll_max_iters + skip; iters++) {                                \
            nvshmemx_barrier_all_on_stream(stream);                                              \
                                                                                                 \
            if (iters >= skip) CUDA_CHECK(cudaEventRecord(start_event, stream));                 \
            nvshmemx_##TYPENAME##_##OP##_reduce_on_stream(                                       \
                team, (TYPE *)d_dest, (const TYPE *)d_source, num_elems, stream);                \
                                                                                                 \
            if (iters >= skip) {                                                                 \
                CUDA_CHECK(cudaEventRecord(stop_event, stream));                                 \
                CUDA_CHECK(cudaStreamSynchronize(stream));                                       \
                CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));                  \
                array[iters - skip] = ms * 1e+3;                                                 \
            }                                                                                    \
        }                                                                                        \
        CUDA_CHECK(cudaEventDestroy(start_event));                                               \
        CUDA_CHECK(cudaEventDestroy(stop_event));                                                \
    } while (0)

#define RUN_RDXN_ITERS(TYPENAME, TYPE, team, d_source, h_source, d_dest, h_dest, num_elems,  \
                       stream, mype, size, usec_sum, usec_prod, usec_and, usec_or, usec_xor, \
                       usec_min, usec_max)                                                   \
    do {                                                                                     \
        size = num_elems * sizeof(TYPE);                                                     \
        RUN_RDXN(TYPENAME, TYPE, sum, team, d_source, h_source, d_dest, h_dest, num_elems,   \
                 usec_sum, stream);                                                          \
    } while (0)

int main(int argc, char **argv) {
    int status = 0;
    int mype;
    int i = 0;
    size_t size = (MAX_ELEMS * 8) * sizeof(LARGEST_DT);
    size_t alloc_size;
    int num_elems;
    LARGEST_DT *h_buffer = NULL;
    LARGEST_DT *d_buffer = NULL;
    LARGEST_DT *d_source, *d_dest;
    LARGEST_DT *h_source, *h_dest;
    char size_string[100];
    uint64_t size_array[MAX_ELEMS_LOG];
    double **sum_latency_array = (double **)malloc(MAX_ELEMS_LOG * sizeof(double *));
    double **prod_latency_array = (double **)malloc(MAX_ELEMS_LOG * sizeof(double *));
    double **and_latency_array = (double **)malloc(MAX_ELEMS_LOG * sizeof(double *));
    double **or_latency_array = (double **)malloc(MAX_ELEMS_LOG * sizeof(double *));
    double **xor_latency_array = (double **)malloc(MAX_ELEMS_LOG * sizeof(double *));
    double **min_latency_array = (double **)malloc(MAX_ELEMS_LOG * sizeof(double *));
    double **max_latency_array = (double **)malloc(MAX_ELEMS_LOG * sizeof(double *));
    cudaStream_t stream;

    memset(size_array, 0, MAX_ELEMS_LOG * sizeof(uint64_t));
    for (int i = 0; i < MAX_ELEMS_LOG; i++) {
        sum_latency_array[i] = (double *)calloc(coll_max_iters, sizeof(double));
        prod_latency_array[i] = (double *)calloc(coll_max_iters, sizeof(double));
        and_latency_array[i] = (double *)calloc(coll_max_iters, sizeof(double));
        or_latency_array[i] = (double *)calloc(coll_max_iters, sizeof(double));
        xor_latency_array[i] = (double *)calloc(coll_max_iters, sizeof(double));
        min_latency_array[i] = (double *)calloc(coll_max_iters, sizeof(double));
        max_latency_array[i] = (double *)calloc(coll_max_iters, sizeof(double));
    }

    DEBUG_PRINT("symmetric size requested %lu\n", size);
    sprintf(size_string, "%lu", size);

    status = setenv("NVSHMEM_SYMMETRIC_SIZE", size_string, 1);
    if (status) {
        fprintf(stderr, "setenv failed \n");
        status = -1;
        goto out;
    }

    init_wrapper(&argc, &argv);

    mype = nvshmem_my_pe();
    CUDA_CHECK(cudaStreamCreate(&stream));

    num_elems = MAX_ELEMS / 2;
    alloc_size = (num_elems * 2) * sizeof(long);

    CUDA_CHECK(cudaHostAlloc(&h_buffer, alloc_size, cudaHostAllocDefault));
    h_source = (LARGEST_DT *)h_buffer;
    h_dest = (LARGEST_DT *)&h_source[num_elems];

    d_buffer = (LARGEST_DT *)nvshmem_malloc(alloc_size);
    if (!d_buffer) {
        fprintf(stderr, "nvshmem_malloc failed \n");
        status = -1;
        goto out;
    }

    /* Run NVLS only specific cases */
    d_source = (LARGEST_DT *)d_buffer;
    d_dest = (LARGEST_DT *)&d_source[num_elems];

    i = 0;
    for (num_elems = 1; num_elems < (MAX_ELEMS / 2); num_elems *= 2) {
        RUN_RDXN_ITERS(float, float, NVSHMEM_TEAM_WORLD, (float *)d_source, (float *)h_source,
                       (float *)d_dest, (float *)h_dest, num_elems, stream, mype, size_array[i],
                       sum_latency_array[i], prod_latency_array[i], and_latency_array[i],
                       or_latency_array[i], xor_latency_array[i], min_latency_array[i],
                       max_latency_array[i]);
        i++;
    }

    if (!mype) {
        print_table_v2("reduction_on_stream", "float-sum", "size (Bytes)", "latency", "us", '-',
                       size_array, sum_latency_array, i);
    }

    nvshmem_barrier_all();
    CUDA_CHECK(cudaFreeHost(h_buffer));
    nvshmem_free(d_buffer);

    CUDA_CHECK(cudaStreamDestroy(stream));

    finalize_wrapper();

out:
    return status;
}
