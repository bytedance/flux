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
#include <getopt.h>
#include "utils.h"

#define MAX_ITERS 10
#define MAX_SKIP 10
#define THREADS 1024
#define BLOCKS 4
#define MAX_MSG_SIZE 32 * 1024 * 1024
#define UNROLL 2

__global__ void bw(double *data_d, double *remote_d, volatile unsigned int *counter_d, int len,
                   int pe, int iter) {
    int u, i, j, tid, slice;
    unsigned int counter;
    int threads = gridDim.x * blockDim.x;
    tid = blockIdx.x * blockDim.x + threadIdx.x;

    slice = UNROLL * threads;

    for (i = 0; i < iter; i++) {
        for (j = 0; j < len - slice; j += slice) {
            for (u = 0; u < UNROLL; ++u) {
                int idx = j + u * threads + tid;
                *(remote_d + idx) = *(data_d + idx);
            }
            __syncthreads();
        }

        for (u = 0; u < UNROLL; ++u) {
            int idx = j + u * threads + tid;
            if (idx < len) *(remote_d + idx) = *(data_d + idx);
        }

        // synchronizing across blocks
        __syncthreads();

        if (!threadIdx.x) {
            __threadfence();
            counter = atomicInc((unsigned int *)counter_d, UINT_MAX);
            if (counter == (gridDim.x * (i + 1) - 1)) {
                *(counter_d + 1) += 1;
            }
            while (*(counter_d + 1) != i + 1)
                ;
        }

        __syncthreads();
    }

    // synchronizing across blocks
    __syncthreads();

    if (!threadIdx.x) {
        __threadfence();
        counter = atomicInc((unsigned int *)counter_d, UINT_MAX);
        if (counter == (gridDim.x * (i + 1) - 1)) {
            nvshmem_quiet();
            *(counter_d + 1) += 1;
        }
        while (*(counter_d + 1) != i + 1)
            ;
    }
}

int main(int argc, char *argv[]) {
    int mype, npes;
    double *data_d = NULL, *remote_d;
    unsigned int *counter_d;
    int max_blocks = BLOCKS, max_threads = THREADS;

    int iter = MAX_ITERS;
    int skip = MAX_SKIP;
    int max_msg_size = MAX_MSG_SIZE;

    int array_size, i;
    void **h_tables;
    uint64_t *h_size_arr;
    double *h_bw;

    float milliseconds;
    cudaEvent_t start, stop;

    init_wrapper(&argc, &argv);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    if (npes != 2) {
        fprintf(stderr, "This test requires exactly two processes \n");
        goto finalize;
    }

    while (1) {
        int c;
        c = getopt(argc, argv, "c:t:h");
        if (c == -1) break;

        switch (c) {
            case 'c':
                max_blocks = strtol(optarg, NULL, 0);
                break;
            case 't':
                max_threads = strtol(optarg, NULL, 0);
                break;
            default:
            case 'h':
                printf("-c [CTAs] -t [THREADS] \n");
                goto finalize;
        }
    }

    array_size = floor(std::log2((float)max_msg_size)) + 1;
    alloc_tables(&h_tables, 2, array_size);
    h_size_arr = (uint64_t *)h_tables[0];
    h_bw = (double *)h_tables[1];

    data_d = (double *)nvshmem_malloc(max_msg_size);
    CUDA_CHECK(cudaMemset(data_d, 0, max_msg_size));

    remote_d = (double *)nvshmem_ptr((void *)data_d, !mype);
    if (remote_d == NULL) {
        fprintf(stderr, "peer memory not accessible for LD/ST \n");
        goto finalize;
    }

    CUDA_CHECK(cudaMalloc((void **)&counter_d, sizeof(unsigned int) * 2));
    CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));

    CUDA_CHECK(cudaDeviceSynchronize());

    if (mype == 0) {
        printf("Size(Bytes) \t\t BWGB/sec\n");
        fflush(stdout);
    }

    int size;
    i = 0;
    if (mype == 0) {
        for (size = 1024; size <= MAX_MSG_SIZE; size *= 2) {
            int blocks = max_blocks, threads = max_threads;
            h_size_arr[i] = size;

            CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));

            bw<<<blocks, threads>>>(data_d, remote_d, counter_d, size / sizeof(double), mype, skip);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));

            cudaEventRecord(start);
            bw<<<blocks, threads>>>(data_d, remote_d, counter_d, size / sizeof(double), mype, iter);
            cudaEventRecord(stop);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventSynchronize(stop));

            cudaEventElapsedTime(&milliseconds, start, stop);
            h_bw[i] = size / (milliseconds * (B_TO_GB / (iter * MS_TO_S)));
            nvshmem_barrier_all();
            i++;
        }
    } else {
        for (size = 1024; size <= MAX_MSG_SIZE; size *= 2) {
            nvshmem_barrier_all();
        }
    }

    if (mype == 0) {
        print_table_v1("shmem_st_bw", "None", "size (Bytes)", "BW", "GB/sec", '+', h_size_arr, h_bw,
                       i);
    }

finalize:

    if (data_d) nvshmem_free(data_d);
    free_tables(h_tables, 2);
    finalize_wrapper();

    return 0;
}
