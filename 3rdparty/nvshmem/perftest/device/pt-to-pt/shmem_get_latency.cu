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

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include "utils.h"

#define MAX_MSG_SIZE 64 * 1024
#define THREADS_PER_WARP 32
#define THREADS_PER_BLOCK 1024

__global__ void latency(int *data_d, int len, int pe, int iter) {
    int i, peer;

    peer = !pe;

    for (i = 0; i < iter; i++) {
        nvshmem_int_get_nbi(data_d, data_d, len, peer);
        nvshmem_quiet();
    }
}

#define LATENCY_THREADGROUP(group)                                            \
    __global__ void latency_##group(int *data_d, int len, int pe, int iter) { \
        int i, tid, peer;                                                     \
                                                                              \
        peer = !pe;                                                           \
        tid = threadIdx.x;                                                    \
                                                                              \
        for (i = 0; i < iter; i++) {                                          \
            nvshmemx_int_get_nbi_##group(data_d, data_d, len, peer);          \
                                                                              \
            __syncthreads();                                                  \
            if (!tid) nvshmem_quiet();                                        \
            __syncthreads();                                                  \
        }                                                                     \
    }

LATENCY_THREADGROUP(warp)
LATENCY_THREADGROUP(block)

int main(int c, char *v[]) {
    int mype, npes, size;
    int *data_d = NULL;

    int iter = 200;
    int skip = 20;
    int max_msg_size = MAX_MSG_SIZE;

    int array_size, i;
    void **h_tables;
    uint64_t *h_size_arr;
    double *h_lat;

    float milliseconds;
    cudaEvent_t start, stop;

    init_wrapper(&c, &v);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    if (npes != 2) {
        fprintf(stderr, "This test requires exactly two processes \n");
        goto finalize;
    }

    data_d = (int *)nvshmem_malloc(max_msg_size);
    CUDA_CHECK(cudaMemset(data_d, 0, max_msg_size));

    array_size = floor(std::log2((float)max_msg_size)) + 1;
    alloc_tables(&h_tables, 2, array_size);
    h_size_arr = (uint64_t *)h_tables[0];
    h_lat = (double *)h_tables[1];

    nvshmem_barrier_all();

    CUDA_CHECK(cudaDeviceSynchronize());

    i = 0;
    for (size = sizeof(int); size <= max_msg_size; size *= 2) {
        if (!mype) {
            int nelems;
            h_size_arr[i] = size;
            nelems = size / sizeof(int);

            latency<<<1, 1>>>(data_d, nelems, mype, skip);
            cudaEventRecord(start);
            latency<<<1, 1>>>(data_d, nelems, mype, iter);
            cudaEventRecord(stop);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventSynchronize(stop));

            /* give latency in us */
            cudaEventElapsedTime(&milliseconds, start, stop);
            h_lat[i] = (milliseconds * 1000) / iter;
            i++;
        }

        nvshmem_barrier_all();
    }

    if (mype == 0) {
        print_table_v1("shmem_g_latency", "Thread", "size (Bytes)", "latency", "us", '-',
                       h_size_arr, h_lat, i);
    }

    i = 0;
    for (size = sizeof(int); size <= max_msg_size; size *= 2) {
        if (!mype) {
            int nelems;
            h_size_arr[i] = size;
            nelems = size / sizeof(int);

            latency_warp<<<1, THREADS_PER_WARP>>>(data_d, nelems, mype, skip);
            cudaEventRecord(start);
            latency_warp<<<1, THREADS_PER_WARP>>>(data_d, nelems, mype, iter);
            cudaEventRecord(stop);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventSynchronize(stop));

            /* give latency in us */
            cudaEventElapsedTime(&milliseconds, start, stop);
            h_lat[i] = (milliseconds * 1000) / iter;
            i++;
        }

        nvshmem_barrier_all();
    }

    if (mype == 0) {
        print_table_v1("shmem_get_latency", "Warp", "size (Bytes)", "latency", "us", '-',
                       h_size_arr, h_lat, i);
    }

    i = 0;
    for (size = sizeof(int); size <= max_msg_size; size *= 2) {
        if (!mype) {
            int nelems;
            h_size_arr[i] = size;
            nelems = size / sizeof(int);

            latency_block<<<1, THREADS_PER_BLOCK>>>(data_d, nelems, mype, skip);
            cudaEventRecord(start);
            latency_block<<<1, THREADS_PER_BLOCK>>>(data_d, nelems, mype, iter);
            cudaEventRecord(stop);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventSynchronize(stop));

            /* give latency in us */
            cudaEventElapsedTime(&milliseconds, start, stop);
            h_lat[i] = (milliseconds * 1000) / iter;
            i++;
        }

        nvshmem_barrier_all();
    }

    if (mype == 0) {
        print_table_v1("shmem_get_latency", "Block", "size (Bytes)", "latency", "us", '-',
                       h_size_arr, h_lat, i);
    }

finalize:

    if (data_d) nvshmem_free(data_d);
    free_tables(h_tables, 2);
    finalize_wrapper();

    return 0;
}
