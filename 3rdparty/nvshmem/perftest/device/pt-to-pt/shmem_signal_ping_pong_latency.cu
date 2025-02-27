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

#define MAX_MSG_SIZE 1 * 1024 * 1024
#define UNROLL 8

__global__ void ping_pong(uint64_t *flag_d, int pe, int iter) {
    int i, peer;

    peer = !pe;

    for (i = 0; i < iter; i++) {
        if (pe) {
            nvshmem_uint64_wait_until(flag_d, NVSHMEM_CMP_EQ, (i + 1));

            nvshmemx_signal_op(flag_d, (i + 1), NVSHMEM_SIGNAL_SET, peer);
        } else {
            nvshmemx_signal_op(flag_d, (i + 1), NVSHMEM_SIGNAL_SET, peer);

            nvshmem_uint64_wait_until(flag_d, NVSHMEM_CMP_EQ, (i + 1));
        }
    }
    nvshmem_quiet();
}

int main(int c, char *v[]) {
    int mype, npes;
    uint64_t *flag_d = NULL;
    cudaStream_t stream;

    int iter = 500;
    int skip = 50;

    void **h_tables;
    double *h_lat;
    uint64_t size = sizeof(uint64_t);

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

    alloc_tables(&h_tables, 2, 1);
    h_lat = (double *)h_tables[1];

    flag_d = (uint64_t *)nvshmem_malloc(sizeof(uint64_t));
    CUDA_CHECK(cudaMemset(flag_d, 0, sizeof(uint64_t)));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    nvshmem_barrier_all();

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    if (mype == 0) {
        printf("Note: This test measures full round-trip latency\n");
    }

    {
        int status = 0;
        void *args_1[] = {&flag_d, &mype, &skip};
        void *args_2[] = {&flag_d, &mype, &iter};

        CUDA_CHECK(cudaMemset(flag_d, 0, sizeof(uint64_t)));
        CUDA_CHECK(cudaDeviceSynchronize());
        nvshmem_barrier_all();

        status = nvshmemx_collective_launch((const void *)ping_pong, 1, 1, args_1, 0, stream);
        if (status != NVSHMEMX_SUCCESS) {
            fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
            exit(-1);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemset(flag_d, 0, sizeof(uint64_t)));
        nvshmem_barrier_all();

        cudaEventRecord(start, stream);
        status = nvshmemx_collective_launch((const void *)ping_pong, 1, 1, args_2, 0, stream);
        if (status != NVSHMEMX_SUCCESS) {
            fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
            exit(-1);
        }
        cudaEventRecord(stop, stream);

        /* give latency in us */
        CUDA_CHECK(cudaEventSynchronize(stop));
        cudaEventElapsedTime(&milliseconds, start, stop);
        *h_lat = (milliseconds * 1000) / iter;
        nvshmem_barrier_all();
    }

    if (mype == 0) {
        print_table_v1("shmem_sig_ping_lat", "None", "size (Bytes)", "latency", "us", '-', &size,
                       h_lat, 1);
    }
finalize:

    if (flag_d) nvshmem_free(flag_d);
    free_tables(h_tables, 2);
    finalize_wrapper();

    return 0;
}
