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

#define MAX_ITERS 100
#define MAX_SKIP 10
#define THREADS 1024
#define BLOCKS 8
#define MAX_DATA_SIZE 64 * 1024
#define MIN_DATA_SIZE 1024
#define ELEMENT_SIZE 8
#define STRIDE 1
#define UNROLL 2

template <typename T>
__device__ inline T call_nvshmem_g(T *rptr, int peer) {
    switch (sizeof(T)) {
        case 1:
            return nvshmem_uint8_g((uint8_t *)rptr, peer);
            break;
        case 2:
            return nvshmem_uint16_g((uint16_t *)rptr, peer);
            break;
        case 4:
            return nvshmem_uint32_g((uint32_t *)rptr, peer);
            break;
        case 8:
            return nvshmem_double_g((double *)rptr, peer);
            break;
        default:
            assert(0);
    }
    return (T)0;
}

template <typename T>
__global__ void bw(T *data_d, volatile unsigned int *counter_d, int len, int pe, int iter,
                   int stride) {
    int u, i, j, peer, tid, slice;
    unsigned int counter;
    int threads = gridDim.x * blockDim.x;
    tid = blockIdx.x * blockDim.x + threadIdx.x;

    peer = !pe;
    slice = UNROLL * threads * stride;

    // When stride > 1, each iteration requests less than len elements.
    // We increase the number of iterations to make up for that.
    for (i = 0; i < iter * stride; i++) {
        for (j = 0; j < len - slice; j += slice) {
            for (u = 0; u < UNROLL; ++u) {
                int idx = j + u * threads + tid * stride;
                *(data_d + idx) = call_nvshmem_g<T>(data_d + idx, peer);
            }
            __syncthreads(); /* This is required for performance over PCIe. PCIe has a P2P mailbox
                                protocol that has a window of 64KB for device BAR addresses. Not
                                synchronizing
                                across threads will lead to jumping in and out of the 64K window */
        }

        for (u = 0; u < UNROLL; ++u) {
            int idx = j + u * threads + tid * stride;
            if (idx < len) *(data_d + idx) = call_nvshmem_g<T>(data_d + idx, peer);
        }

        // synchronizing across blocks
        __syncthreads();

        if (!threadIdx.x) {
            __threadfence(); /* To ensure that the data received through shmem_g is
                                visible across the gpu */
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
            *(counter_d + 1) += 1;
        }
        while (*(counter_d + 1) != i + 1)
            ;
    }
}

void call_bw(int blocks, int threads, void *data_d, unsigned int *counter_d, size_t size,
             int element_size, int mype, int iter, int stride) {
    switch (element_size) {
        case 1:
            bw<uint8_t><<<blocks, threads>>>((uint8_t *)data_d, counter_d, size / sizeof(uint8_t),
                                             mype, iter, stride);
            break;
        case 2:
            bw<uint16_t><<<blocks, threads>>>((uint16_t *)data_d, counter_d,
                                              size / sizeof(uint16_t), mype, iter, stride);
            break;
        case 4:
            bw<uint32_t><<<blocks, threads>>>((uint32_t *)data_d, counter_d,
                                              size / sizeof(uint32_t), mype, iter, stride);
            break;
        case 8:
            bw<double><<<blocks, threads>>>((double *)data_d, counter_d, size / sizeof(double),
                                            mype, iter, stride);
            break;
        default:
            fprintf(stderr, "element_size=%d is not supported \n", element_size);
            exit(-EINVAL);
    }
}

int main(int argc, char *argv[]) {
    int mype, npes;
    void *data_d = NULL;
    unsigned int *counter_d;
    int max_blocks = BLOCKS, max_threads = THREADS;
    int array_size, i;
    void **h_tables;
    uint64_t *h_size_arr;
    double *h_bw;
    double *h_msgrate;
    bool report_msgrate = false;

    int iter = MAX_ITERS;
    int skip = MAX_SKIP;
    int64_t max_data_size = MAX_DATA_SIZE;
    int64_t min_data_size = MIN_DATA_SIZE;
    int element_size = ELEMENT_SIZE;
    int stride = STRIDE;

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
        c = getopt(argc, argv, "c:t:m:M:e:s:Rh");
        if (c == -1) break;

        switch (c) {
            case 'c':
                max_blocks = strtol(optarg, NULL, 0);
                break;
            case 't':
                max_threads = strtol(optarg, NULL, 0);
                break;
            case 'm':
                min_data_size = strtol(optarg, NULL, 0);
                break;
            case 'M':
                max_data_size = strtol(optarg, NULL, 0);
                break;
            case 'e':
                element_size = strtol(optarg, NULL, 0);
                break;
            case 's':
                stride = strtol(optarg, NULL, 0);
                break;
            case 'R':
                report_msgrate = true;
                break;
            default:
            case 'h':
                printf(
                    "-c [CTAs] -t [THREADS] -m [MIN_DATA] -M [MAX_DATA] -e [ELEMENT_SIZE] -s "
                    "[STRIDE] -R(report_msgrate) \n");
                goto finalize;
        }
    }

    if (min_data_size <= 0) {
        fprintf(stderr, "MIN_DATA must be a positive integer \n");
        goto finalize;
    }

    if (max_data_size <= 0) {
        fprintf(stderr, "MAX_DATA must be a positive integer \n");
        goto finalize;
    }

    if (min_data_size > max_data_size) {
        fprintf(stderr, "MIN_DATA must be less than or equal to MAX_DATA \n");
        goto finalize;
    }

    if (stride < 1) {
        fprintf(stderr, "STRIDE must be at least 1 \n");
        goto finalize;
    }

    array_size = floor(std::log2((float)max_data_size)) + 1;
    alloc_tables(&h_tables, 3, array_size);
    h_size_arr = (uint64_t *)h_tables[0];
    h_bw = (double *)h_tables[1];
    h_msgrate = (double *)h_tables[2];

    data_d = (void *)nvshmem_malloc(max_data_size);
    CUDA_CHECK(cudaMemset(data_d, 0, max_data_size));

    CUDA_CHECK(cudaMalloc((void **)&counter_d, sizeof(unsigned int) * 2));
    CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));

    CUDA_CHECK(cudaDeviceSynchronize());

    size_t size;
    i = 0;
    if (mype == 0) {
        for (size = (size_t)min_data_size; size <= (size_t)max_data_size; size *= 2) {
            int blocks = max_blocks, threads = max_threads;
            h_size_arr[i] = size;
            CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));
            call_bw(blocks, threads, data_d, counter_d, size, element_size, mype, skip, stride);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));

            cudaEventRecord(start);
            call_bw(blocks, threads, data_d, counter_d, size, element_size, mype, iter, stride);
            cudaEventRecord(stop);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventSynchronize(stop));

            cudaEventElapsedTime(&milliseconds, start, stop);
            h_bw[i] = size / (milliseconds * (B_TO_GB / (iter * MS_TO_S)));
            h_msgrate[i] = (double)(size / element_size) * iter / (milliseconds * MS_TO_S);
            nvshmem_barrier_all();
            i++;
        }
    } else {
        for (size = (size_t)min_data_size; size <= (size_t)max_data_size; size *= 2) {
            nvshmem_barrier_all();
        }
    }

    if (mype == 0) {
        print_table_v1("shmem_g_bw", "None", "size (Bytes)", "BW", "GB/sec", '+', h_size_arr, h_bw,
                       i);
        if (report_msgrate)
            print_table_v1("shmem_g_bw", "None", "size (Bytes)", "msgrate", "MMPS", '+', h_size_arr,
                           h_msgrate, i);
    }

finalize:

    if (data_d) nvshmem_free(data_d);
    free_tables(h_tables, 3);
    finalize_wrapper();

    return 0;
}
