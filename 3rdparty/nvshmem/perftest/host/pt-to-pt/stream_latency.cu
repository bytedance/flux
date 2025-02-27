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
#include <string.h>
#include <getopt.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "utils.h"

#define DEFAULT_ITERS 10
#define DEFAULT_MIN_MSG_SIZE 1
#define DEFAULT_MAX_MSG_SIZE 128 * 1024 * 1024

typedef enum { PUSH = 0, PULL = 1 } dir_t;

__global__ void test_kernel(void *data_d_local, long long int ncycles) {
    long long int sclk = clock64();
    long long int cyc = 0;
    while (cyc < ncycles) {
        cyc = clock64() - sclk;
    }
    *(long long int *)data_d_local = cyc;
}

int lat(void *data_d, void *data_d_local, int sizeBytes, int pe, int iter, dir_t dir,
        cudaStream_t strm, cudaEvent_t sev, cudaEvent_t eev, float *ms1, float *ms2, int ng, int nb,
        long long int ncycles) {
    int status = 0;
    int peer = !pe;

    if (dir == PUSH) {
        CUDA_CHECK(cudaEventRecord(sev, strm));
        for (int i = 0; i < iter; i++) {
            test_kernel<<<ng, nb, 0, strm>>>(data_d_local, ncycles);
            nvshmemx_putmem_on_stream((void *)data_d, (void *)data_d_local, sizeBytes, peer, strm);
        }
        CUDA_CHECK(cudaEventRecord(eev, strm));
        CUDA_CHECK(cudaEventSynchronize(eev));
        CUDA_CHECK(cudaEventElapsedTime(ms1, sev, eev));

        CUDA_CHECK(cudaEventRecord(sev, strm));
        for (int i = 0; i < iter; i++) {
            test_kernel<<<ng, nb, 0, strm>>>(data_d_local, ncycles);
            CUDA_CHECK(cudaStreamSynchronize(strm));
            nvshmem_putmem((void *)data_d, (void *)data_d_local, sizeBytes, peer);
        }
        CUDA_CHECK(cudaEventRecord(eev, strm));
        CUDA_CHECK(cudaEventSynchronize(eev));
        CUDA_CHECK(cudaEventElapsedTime(ms2, sev, eev));
    } else {
        CUDA_CHECK(cudaEventRecord(sev, strm));
        for (int i = 0; i < iter; i++) {
            nvshmemx_getmem_on_stream((void *)data_d_local, (void *)data_d, sizeBytes, peer, strm);
            test_kernel<<<ng, nb, 0, strm>>>(data_d_local, ncycles);
        }
        CUDA_CHECK(cudaEventRecord(eev, strm));
        CUDA_CHECK(cudaEventSynchronize(eev));
        CUDA_CHECK(cudaEventElapsedTime(ms1, sev, eev));

        CUDA_CHECK(cudaEventRecord(sev, strm));
        for (int i = 0; i < iter; i++) {
            nvshmem_getmem((void *)data_d_local, (void *)data_d, sizeBytes,
                           peer);  // shmem_getmem is blocking, so nvshmem_quiet is not needed
            test_kernel<<<ng, nb, 0, strm>>>(data_d_local, ncycles);
        }
        CUDA_CHECK(cudaEventRecord(eev, strm));
        CUDA_CHECK(cudaEventSynchronize(eev));
        CUDA_CHECK(cudaEventElapsedTime(ms2, sev, eev));
    }

    return status;
}

int main(int argc, char *argv[]) {
    int status = 0;
    int mype, npes;
    char *data_d = NULL, *data_d_local = NULL;
    uint64_t *size_array = NULL;
    double *offs_latency_array = NULL;
    double *ons_latency_array = NULL;
    int num_entries;
    int i;

    dir_t dir = PUSH;
    int iter = DEFAULT_ITERS;
    int min_msg_size = DEFAULT_MIN_MSG_SIZE;
    int max_msg_size = DEFAULT_MAX_MSG_SIZE;

    int nb = 1, nt = 32;
    long long int ncycles = 1;

    init_wrapper(&argc, &argv);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    if (npes != 2) {
        fprintf(stderr, "This test requires exactly two processes \n");
        status = -1;
        goto finalize;
    }

    while (1) {
        int c;
        c = getopt(argc, argv, "s:S:n:i:d:b:t:c:h");
        if (c == -1) break;

        switch (c) {
            case 's':
                min_msg_size = strtol(optarg, NULL, 0);
                break;
            case 'S':
                max_msg_size = strtol(optarg, NULL, 0);
                break;
            case 'n':
                iter = strtol(optarg, NULL, 0);
                break;
            case 'd':
                dir = (dir_t)strtol(optarg, NULL, 0);
                break;
            case 'b':
                nb = strtol(optarg, NULL, 0);
                break;
            case 't':
                nt = strtol(optarg, NULL, 0);
                break;
            case 'c':
                ncycles = strtol(optarg, NULL, 0);
                break;
            default:
            case 'h':
                printf(
                    "-n [Iterations] -S [Max message size] -s [Min message size] -i [Put/Get issue type : ON_STREAM(0) otherwise 1] -d [Direction of copy : PUSH(0) or PULL(1)] -b [# blocks] \
                 -t [# threads] -c [# cycles to wait in the the kernel]\n");
                goto finalize;
        }
    }

    num_entries = floor(std::log2((float)max_msg_size)) - floor(std::log2((float)min_msg_size)) + 1;
    size_array = (uint64_t *)calloc(sizeof(uint64_t), num_entries);
    if (!size_array) {
        status = -1;
        goto finalize;
    }

    offs_latency_array = (double *)calloc(sizeof(double), num_entries);
    if (!offs_latency_array) {
        status = -1;
        goto finalize;
    }

    ons_latency_array = (double *)calloc(sizeof(double), num_entries);
    if (!ons_latency_array) {
        status = -1;
        goto finalize;
    }

    data_d = (char *)nvshmem_malloc(max_msg_size);
    CUDA_CHECK(cudaMemset(data_d, 0, max_msg_size));

    data_d_local = (char *)nvshmem_malloc(max_msg_size);
    CUDA_CHECK(cudaMemset(data_d, 0, max_msg_size));

    cudaStream_t strm;
    CUDA_CHECK(cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking));

    CUDA_CHECK(cudaDeviceSynchronize());

    if (mype == 0) {
        float ms1, ms2;
        cudaEvent_t sev, eev;
        CUDA_CHECK(cudaEventCreate(&sev));
        CUDA_CHECK(cudaEventCreate(&eev));
        i = 0;
        for (int size = min_msg_size; size <= max_msg_size; size *= 2) {
            size_array[i] = size;
            lat(data_d, data_d_local, size, mype, iter, dir, strm, sev, eev, &ms1, &ms2, nb, nt,
                ncycles);
            ons_latency_array[i] = ms1 / iter * 1000;
            offs_latency_array[i] = ms2 / iter * 1000;
            i++;
        }

        print_table_v1("Stream_Latency", "with _on_stream", "size (Bytes)", "latency", "us", '-',
                       size_array, ons_latency_array, i);
        print_table_v1("Stream_Latency", "without _on_stream", "size (Bytes)", "latency", "us", '-',
                       size_array, offs_latency_array, i);

        CUDA_CHECK(cudaEventDestroy(sev));
        CUDA_CHECK(cudaEventDestroy(eev));

        nvshmem_barrier_all();

    } else {
        nvshmem_barrier_all();
    }

finalize:
    CUDA_CHECK(cudaStreamDestroy(strm));

    if (data_d) nvshmem_free(data_d);
    if (size_array) free(size_array);
    if (ons_latency_array) free(ons_latency_array);
    if (offs_latency_array) free(offs_latency_array);

    if (data_d_local) nvshmem_free(data_d_local);

    finalize_wrapper();

    return status;
}
