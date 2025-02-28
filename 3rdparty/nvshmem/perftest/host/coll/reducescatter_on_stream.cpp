/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#ifdef MAX_ELEMS
#undef MAX_ELEMS
#endif
#define MAX_ELEMS 1024
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
            nvshmemx_##TYPENAME##_##OP##_reducescatter_on_stream(                                \
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
        nvshmem_barrier_all();                                                                   \
    } while (0)

#define RUN_RDXN_ITERS(TYPENAME, TYPE, team, d_source, h_source, d_dest, h_dest, num_elems, \
                       stream, mype, size, usec_sum, usec_min, usec_max)                    \
    do {                                                                                    \
        size = num_elems * sizeof(TYPE);                                                    \
        RUN_RDXN(TYPENAME, TYPE, sum, team, d_source, h_source, d_dest, h_dest, num_elems,  \
                 usec_sum, stream);                                                         \
        RUN_RDXN(TYPENAME, TYPE, min, team, d_source, h_source, d_dest, h_dest, num_elems,  \
                 usec_min, stream);                                                         \
        RUN_RDXN(TYPENAME, TYPE, max, team, d_source, h_source, d_dest, h_dest, num_elems,  \
                 usec_max, stream);                                                         \
    } while (0)

/* Add only float based */
#define RUN_RDXN_ITERS_FLOAT(team, d_source, h_source, d_dest, h_dest, num_elems, stream, mype,    \
                             size, usec_sum, usec_min, usec_max)                                   \
    do {                                                                                           \
        size = num_elems * sizeof(float);                                                          \
        RUN_RDXN(float, float, sum, team, d_source, h_source, d_dest, h_dest, num_elems, usec_sum, \
                 stream);                                                                          \
    } while (0)

#define RUN_RDXN_ITERS_DOUBLE(team, d_source, h_source, d_dest, h_dest, num_elems, stream, mype, \
                              size, usec_sum, usec_min, usec_max)                                \
    do {                                                                                         \
        size = num_elems * sizeof(double);                                                       \
        RUN_RDXN(double, double, sum, team, d_source, h_source, d_dest, h_dest, num_elems,       \
                 usec_sum, stream);                                                              \
    } while (0)

int main(int argc, char **argv) {
    int status = 0;
    int mype;
    int i = 0;
    size_t size = (MAX_ELEMS * 128 /* For source bufer */ +
                   MAX_ELEMS * 7 /* For dest buffer and other things */) *
                  sizeof(LARGEST_DT);
    size_t alloc_size;
    int num_elems;
    LARGEST_DT *h_buffer = NULL;
    LARGEST_DT *d_buffer = NULL;
    LARGEST_DT *d_source, *d_dest;
    LARGEST_DT *h_source, *h_dest;
    char size_string[100];
    uint64_t size_array[MAX_ELEMS_LOG];
    double **sum_latency_array = (double **)malloc(MAX_ELEMS_LOG * sizeof(double *));
    double **min_latency_array = (double **)malloc(MAX_ELEMS_LOG * sizeof(double *));
    double **max_latency_array = (double **)malloc(MAX_ELEMS_LOG * sizeof(double *));
    cudaStream_t stream;

    memset(size_array, 0, MAX_ELEMS_LOG * sizeof(uint64_t));
    for (int i = 0; i < MAX_ELEMS_LOG; i++) {
        sum_latency_array[i] = (double *)calloc(coll_max_iters, sizeof(double));
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
    assert(nvshmem_n_pes() <= 128);  // For larger runs, size calculations above have to be adjusted

    mype = nvshmem_my_pe();
    CUDA_CHECK(cudaStreamCreate(&stream));

    num_elems = MAX_ELEMS / 2;
    alloc_size =
        (num_elems /* For dest */ + num_elems * nvshmem_n_pes() /* For source */) * sizeof(long);

    CUDA_CHECK(cudaHostAlloc(&h_buffer, alloc_size, cudaHostAllocDefault));
    h_source = (LARGEST_DT *)h_buffer;
    h_dest = (LARGEST_DT *)&h_source[num_elems * nvshmem_n_pes()];

    d_buffer = (LARGEST_DT *)nvshmem_malloc(alloc_size);
    if (!d_buffer) {
        fprintf(stderr, "nvshmem_malloc failed \n");
        status = -1;
        goto out;
    }

    d_source = (LARGEST_DT *)d_buffer;
    d_dest = (LARGEST_DT *)&d_source[num_elems * nvshmem_n_pes()];

    i = 0;
    for (num_elems = 1; num_elems < (MAX_ELEMS / 2); num_elems *= 2) {
        RUN_RDXN_ITERS(int32, int32_t, NVSHMEM_TEAM_WORLD, (int *)d_source, (int *)h_source,
                       (int *)d_dest, (int *)h_dest, num_elems, stream, mype, size_array[i],
                       sum_latency_array[i], min_latency_array[i], max_latency_array[i]);
        i++;
    }

    if (!mype) {
        print_table_v2("reducescatter_on_stream", "int-sum", "size (Bytes)", "latency", "us", '-',
                       size_array, sum_latency_array, i);
        print_table_v2("reducescatter_on_stream", "int-min", "size (Bytes)", "latency", "us", '-',
                       size_array, min_latency_array, i);
        print_table_v2("reducescatter_on_stream", "int-max", "size (Bytes)", "latency", "us", '-',
                       size_array, max_latency_array, i);
    }

    i = 0;
    for (num_elems = 1; num_elems < (MAX_ELEMS / 2); num_elems *= 2) {
        RUN_RDXN_ITERS(int64, int64_t, NVSHMEM_TEAM_WORLD, d_source, h_source, d_dest, h_dest,
                       num_elems, stream, mype, size_array[i], sum_latency_array[i],
                       min_latency_array[i], max_latency_array[i]);
        i++;
    }

    if (!mype) {
        print_table_v2("reducescatter_on_stream", "int64-sum", "size (Bytes)", "latency", "us", '-',
                       size_array, sum_latency_array, i);
        print_table_v2("reducescatter_on_stream", "int64-min", "size (Bytes)", "latency", "us", '-',
                       size_array, min_latency_array, i);
        print_table_v2("reducescatter_on_stream", "int64-max", "size (Bytes)", "latency", "us", '-',
                       size_array, max_latency_array, i);
    }

    /* Useful for NVLink collectives */
    i = 0;
    for (num_elems = 1; num_elems < (MAX_ELEMS / 2); num_elems *= 2) {
        RUN_RDXN_ITERS_FLOAT(NVSHMEM_TEAM_WORLD, (float *)d_source, (float *)h_source,
                             (float *)d_dest, (float *)h_dest, num_elems, stream, mype,
                             size_array[i], sum_latency_array[i], min_latency_array[i],
                             max_latency_array[i]);
        i++;
    }

    if (!mype) {
        print_table_v2("reducescatter_on_stream", "float-sum", "size (Bytes)", "latency", "us", '-',
                       size_array, sum_latency_array, i);
    }

    i = 0;
    for (num_elems = 1; num_elems < (MAX_ELEMS / 2); num_elems *= 2) {
        RUN_RDXN_ITERS_DOUBLE(NVSHMEM_TEAM_WORLD, (double *)d_source, (double *)h_source,
                              (double *)d_dest, (double *)h_dest, num_elems, stream, mype,
                              size_array[i], sum_latency_array[i], min_latency_array[i],
                              max_latency_array[i]);
        i++;
    }

    if (!mype) {
        print_table_v2("reducescatter_on_stream", "double-sum", "size (Bytes)", "latency", "us",
                       '-', size_array, sum_latency_array, i);
    }

    nvshmem_barrier_all();
    CUDA_CHECK(cudaFreeHost(h_buffer));
    nvshmem_free(d_buffer);

    CUDA_CHECK(cudaStreamDestroy(stream));

    finalize_wrapper();

out:
    return status;
}
