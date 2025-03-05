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

#define THREADS 512
#define MAX_MSG_SIZE 16 * 1024
#define UNROLL 8

__global__ void ping_pong(int *data_d, uint64_t *flag_d, int len, int pe, int iter) {
    int i, j, tid, peer;

    peer = !pe;
    tid = threadIdx.x;

    for (i = 0; i < iter; i++) {
        if (pe) {
            if (!tid) {
                nvshmem_uint64_wait_until(flag_d, NVSHMEM_CMP_EQ, (i + 1));
            }
            __syncthreads();

            for (j = tid; j < len; j += THREADS) {
                nvshmem_int_p(data_d + j, *(data_d + j), peer);
            }
            __syncthreads();

            if (!tid) {
                nvshmem_fence();
                nvshmemx_signal_op(flag_d, (i + 1), NVSHMEM_SIGNAL_SET, peer);
            }
            __syncthreads();
        } else {
            for (j = tid; j < len; j += THREADS) {
                nvshmem_int_p(data_d + j, *(data_d + j), peer);
            }
            __syncthreads();

            if (!tid) {
                nvshmem_fence();
                nvshmemx_signal_op(flag_d, (i + 1), NVSHMEM_SIGNAL_SET, peer);
            }
            __syncthreads();

            if (!tid) {
                nvshmem_uint64_wait_until(flag_d, NVSHMEM_CMP_EQ, (i + 1));
            }
            __syncthreads();
        }
    }

    if (!tid) nvshmem_quiet();
}

int main(int c, char *v[]) {
    int mype, npes, size;
    uint64_t *flag_d = NULL;
    int *data_d = NULL;

    sleep(10);

    int iter = 500;
    int skip = 50;
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

    array_size = floor(std::log2((float)max_msg_size)) + 1;
    alloc_tables(&h_tables, 2, array_size);
    h_size_arr = (uint64_t *)h_tables[0];
    h_lat = (double *)h_tables[1];

    data_d = (int *)nvshmem_malloc(max_msg_size);
    flag_d = (uint64_t *)nvshmem_malloc(sizeof(uint64_t));
    CUDA_CHECK(cudaMemset(data_d, 0, max_msg_size));

    nvshmem_barrier_all();

    CUDA_CHECK(cudaDeviceSynchronize());

    if (mype == 0) {
        printf("Note: This test measures full round-trip latency\n");
    }

    i = 0;
    for (size = sizeof(int); size <= max_msg_size; size *= 2) {
        int nelems, status = 0;
        nelems = size / sizeof(int);
        h_size_arr[i] = size;
        void *args_1[5] = {&data_d, &flag_d, &nelems, &mype, &skip};
        void *args_2[5] = {&data_d, &flag_d, &nelems, &mype, &iter};

        CUDA_CHECK(cudaMemset(flag_d, 0, sizeof(uint64_t)));
        CUDA_CHECK(cudaDeviceSynchronize());
        nvshmem_barrier_all();
        status = nvshmemx_collective_launch((const void *)ping_pong, 1, THREADS, args_1, 0, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemset(flag_d, 0, sizeof(uint64_t)));

        cudaEventRecord(start);
        status = nvshmemx_collective_launch((const void *)ping_pong, 1, THREADS, args_2, 0, 0);
        if (status != NVSHMEMX_SUCCESS) {
            printf("shmemx_collective_launch failed %d \n", status);
            exit(-1);
        }
        cudaEventRecord(stop);

        /* give latency in us */
        CUDA_CHECK(cudaEventSynchronize(stop));
        cudaEventElapsedTime(&milliseconds, start, stop);
        h_lat[i] = (milliseconds * 1000) / iter;
        nvshmem_barrier_all();
        i++;
    }

    if (mype == 0) {
        print_table_v1("shmem_ping_pong_lat", "None", "size (Bytes)", "latency", "us", '-',
                       h_size_arr, h_lat, i);
    }

finalize:

    if (data_d) nvshmem_free(data_d);
    if (flag_d) nvshmem_free(flag_d);
    free_tables(h_tables, 2);
    finalize_wrapper();

    return 0;
}
