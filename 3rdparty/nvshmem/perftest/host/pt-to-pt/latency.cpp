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
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "utils.h"

#define DEFAULT_SKIP 10
#define DEFAULT_ITERS 10
#define DEFAULT_MIN_MSG_SIZE 1
#define DEFAULT_MAX_MSG_SIZE 128 * 1024 * 1024

typedef enum { ON_STREAM = 0, N_ISSUE_TYPES = 1 } putget_issue_t;

typedef enum { PUSH = 0, PULL = 1 } dir_t;

int lat(void *data_d, void *data_d_local, int sizeBytes, int pe, int iter, int skip,
        putget_issue_t iss, dir_t dir, cudaStream_t strm, cudaEvent_t sev, cudaEvent_t eev,
        float *ms, float *us) {
    int status = 0;
    int peer = !pe;
    struct timeval start, stop;

    if (iss == ON_STREAM) {
        if (dir == PUSH) {
            for (int i = 0; i < (iter + skip); i++) {
                if (i == skip) CUDA_CHECK(cudaEventRecord(sev, strm));
                nvshmemx_putmem_on_stream((void *)data_d, (void *)data_d_local, sizeBytes, peer,
                                          strm);
            }
        } else {
            for (int i = 0; i < (iter + skip); i++) {
                if (i == skip) CUDA_CHECK(cudaEventRecord(sev, strm));
                nvshmemx_getmem_on_stream((void *)data_d_local, (void *)data_d, sizeBytes, peer,
                                          strm);
            }
        }
        CUDA_CHECK(cudaEventRecord(eev, strm));
        CUDA_CHECK(cudaEventSynchronize(eev));
        CUDA_CHECK(cudaEventElapsedTime(ms, sev, eev));
    } else {
        if (dir == PUSH) {
            for (int i = 0; i < (iter + skip); i++) {
                if (i == skip) gettimeofday(&start, NULL);
                nvshmem_putmem((void *)data_d, (void *)data_d_local, sizeBytes, peer);
            }
        } else {
            for (int i = 0; i < (iter + skip); i++) {
                if (i == skip) gettimeofday(&start, NULL);
                nvshmem_getmem((void *)data_d_local, (void *)data_d, sizeBytes, peer);
            }
        }
        nvshmem_quiet();
        gettimeofday(&stop, NULL);
        *us = (stop.tv_usec - start.tv_usec) + (stop.tv_sec - start.tv_sec) * 1000000;
    }

    return status;
}

int main(int argc, char *argv[]) {
    int status = 0;
    int mype, npes;
    char *data_d = NULL, *data_d_local = NULL;
    void *data_h_local = NULL;
    uint64_t *size_array = NULL;
    double *latency_array = NULL;
    int num_entries;
    int i;

    putget_issue_t iss = N_ISSUE_TYPES;
    dir_t dir = PUSH;
    int iter = DEFAULT_ITERS;
    int skip = DEFAULT_SKIP;
    int min_msg_size = DEFAULT_MIN_MSG_SIZE;
    int max_msg_size = DEFAULT_MAX_MSG_SIZE;

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
        c = getopt(argc, argv, "s:S:n:k:i:d:h");
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
            case 'k':
                skip = strtol(optarg, NULL, 0);
                break;
            case 'i':
                iss = (putget_issue_t)strtol(optarg, NULL, 0);
                break;
            case 'd':
                dir = (dir_t)strtol(optarg, NULL, 0);
                break;
            default:
            case 'h':
                printf(
                    "-n [Iterations] -k [Iterations to skip before benchmarking] -S [Max message "
                    "size] -s [Min message size] -i [Put/Get issue type : ON_STREAM(0) otherwise "
                    "1] -d [Direction of copy : PUSH(0) or PULL(1)]\n");
                goto finalize;
        }
    }

    num_entries = floor(log2((float)max_msg_size)) - floor(log2((float)min_msg_size)) + 1;
    size_array = (uint64_t *)calloc(sizeof(uint64_t), num_entries);
    if (!size_array) {
        status = -1;
        goto finalize;
    }

    latency_array = (double *)calloc(sizeof(double), num_entries);
    if (!latency_array) {
        status = -1;
        goto finalize;
    }

    data_d = (char *)nvshmem_malloc(max_msg_size);
    CUDA_CHECK(cudaMemset(data_d, 0, max_msg_size));

    data_h_local = (double *)malloc(sizeof(double));
    if (!data_h_local) {
        fprintf(stderr, "malloc failed \n");
        status = -1;
        goto finalize;
    }

    memset(data_h_local, 0, sizeof(double));

#ifdef _NVSHMEM_REGISTRATION_CACHE_ENABLED
    CUDA_CHECK(cudaMalloc((void **)&data_d_local, max_msg_size));
#else
    data_d_local = (char *)nvshmem_malloc(max_msg_size);
#endif
    CUDA_CHECK(cudaMemset(data_d_local, 0, max_msg_size));

    cudaStream_t strm;
    CUDA_CHECK(cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking));

    CUDA_CHECK(cudaDeviceSynchronize());

    if (mype == 0) {
        float ms, us;
        cudaEvent_t sev, eev;
        CUDA_CHECK(cudaEventCreate(&sev));
        CUDA_CHECK(cudaEventCreate(&eev));
        i = 0;
        for (int size = min_msg_size; size <= max_msg_size; size *= 2) {
            lat(data_d, data_d_local, size, mype, iter, skip, iss, dir, strm, sev, eev, &ms, &us);
            size_array[i] = size;
            if (iss == ON_STREAM) {
                latency_array[i] = ms * 1000 / iter;
            } else {
                latency_array[i] = us / iter;
            }
            i++;
        }

        print_table_v1("Latency", "None", "size (Bytes)", "latency", "us", '-', size_array,
                       latency_array, i);
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
    if (latency_array) free(latency_array);

#ifdef _NVSHMEM_REGISTRATION_CACHE_ENABLED
    if (data_d_local) cudaFree(data_d_local);
#else
    if (data_d_local) nvshmem_free(data_d_local);
#endif

    finalize_wrapper();

    return status;
}
