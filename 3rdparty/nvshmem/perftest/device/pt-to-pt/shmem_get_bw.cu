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

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <getopt.h>
#include "utils.h"

#define MAX_MSG_SIZE (32 * 1024 * 1024)

#define MAX_ITERS 200
#define MAX_SKIP 20
#define BLOCKS 4
#define THREADS_PER_BLOCK 1024

__global__ void bw(double *data_d, volatile unsigned int *counter_d, int len, int pe, int iter) {
    int i, peer;
    unsigned int counter;
    int tid = (threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z + threadIdx.z);
    int bid = blockIdx.x;
    int nblocks = gridDim.x;

    peer = !pe;
    for (i = 0; i < iter; i++) {
        nvshmemx_double_get_nbi_block(data_d + (bid * (len / nblocks)),
                                      data_d + (bid * (len / nblocks)), len / nblocks, peer);
        // synchronizing across blocks
        __syncthreads();
        if (!tid) {
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
    if (!tid) {
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
    double *data_d = NULL;
    unsigned int *counter_d;
    int max_blocks = BLOCKS, max_threads = THREADS_PER_BLOCK;
    int array_size, i;
    void **h_tables;
    uint64_t *h_size_arr;
    double *h_bw = NULL, *h_bw_total = NULL;
    double *d_bw = NULL, *d_bw_sum = NULL;
    bool bidirectional = false;

    int iter = MAX_ITERS;
    int skip = MAX_SKIP;

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
        c = getopt(argc, argv, "c:t:hb");

        if (c == -1) break;

        switch (c) {
            case 'c':
                max_blocks = strtol(optarg, NULL, 0);
                break;
            case 't':
                max_threads = strtol(optarg, NULL, 0);
                break;
            case 'b':
                bidirectional = true;
                break;
            default:
            case 'h':
                printf("-c [CTAs] -t [THREADS] -b\n");
                goto finalize;
        }
    }

    data_d = (double *)nvshmem_malloc(MAX_MSG_SIZE);
    CUDA_CHECK(cudaMemset(data_d, 0, MAX_MSG_SIZE));

    array_size = floor(std::log2((float)MAX_MSG_SIZE)) + 1;
    alloc_tables(&h_tables, 2, array_size);
    h_size_arr = (uint64_t *)h_tables[0];
    h_bw = (double *)h_tables[1];

    if (bidirectional) {
        h_bw_total = (double *)malloc(sizeof(double) * array_size);

        if (!h_bw_total) {
            fprintf(stderr, "Error: Unable to malloc on the host.\n");
            exit(1);
        }

        memset(h_bw_total, 0, sizeof(double) * array_size);

        /* Allocate on GPU. */
        CUDA_CHECK(cudaMalloc((void **)&d_bw, sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&d_bw_sum, sizeof(double)));
    }

    CUDA_CHECK(cudaMalloc((void **)&counter_d, sizeof(unsigned int) * 2));
    CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));

    CUDA_CHECK(cudaDeviceSynchronize());

    if (bidirectional || mype == 0) {
        i = 0;
        for (int size = 1024; size <= MAX_MSG_SIZE; size *= 2) {
            h_size_arr[i] = size;
            CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));
            bw<<<max_blocks, max_threads>>>(data_d, counter_d, size / sizeof(double), mype, skip);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));

            cudaEventRecord(start);
            bw<<<max_blocks, max_threads>>>(data_d, counter_d, size / sizeof(double), mype, iter);
            cudaEventRecord(stop);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventSynchronize(stop));

            cudaEventElapsedTime(&milliseconds, start, stop);
            h_bw[i] = size / (milliseconds * (B_TO_GB / (iter * MS_TO_S)));
            nvshmem_barrier_all();

            /* Sum all h_bw of each PE for bidirectional mode. */
            if (bidirectional) {
                CUDA_CHECK(cudaMemcpy(d_bw, &h_bw[i], sizeof(double), cudaMemcpyHostToDevice));
                nvshmem_double_sum_reduce(NVSHMEM_TEAM_WORLD, d_bw_sum, d_bw, 1);
                CUDA_CHECK(
                    cudaMemcpy(&h_bw_total[i], d_bw_sum, sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(
                    cudaMemcpy(&h_bw_total[i], d_bw_sum, sizeof(double), cudaMemcpyDeviceToHost));
            }

            i++;
        }
    } else {
        for (int size = 1024; size <= MAX_MSG_SIZE; size *= 2) {
            nvshmem_barrier_all();
        }
    }

    if (mype == 0) {
        double *p_h_bw_tmp = bidirectional ? h_bw_total : h_bw;
        const char *const test_name = bidirectional ? "shmem_get_bw_bidi" : "shmem_get_bw_uni";
        print_table_basic(test_name, "None", "size (Bytes)", "BW", "GB/sec", '+', h_size_arr,
                          p_h_bw_tmp, i);
    }

finalize:

    if (data_d) nvshmem_free(data_d);
    if (h_bw_total) free(h_bw_total);
    if (d_bw) cudaFree(d_bw);
    if (d_bw_sum) cudaFree(d_bw_sum);
    free_tables(h_tables, 2);
    finalize_wrapper();

    return 0;
}
