/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.  All rights reserved.
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
#include <stdlib.h>
#include <cstring>
#include <unistd.h>
#include "utils.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <sys/time.h>

int main(int argc, char *argv[]) {
    int status = 0;
    int mype;
    struct timeval t_start, t_stop;
    char size_string[100];
    size_t min_malloc_size = 1 << 30;
    size_t max_alloc_size;
    uint64_t *h_size_arr;
    double *h_time;
    int loop_size = 0;
    size_t malloc_size;
    size_t total_alloc_size = 0;
    CU_CHECK(cuInit(0));
    CUdevice device;
    CU_CHECK(cuDeviceGet(&device, 0));
    CU_CHECK(cuDeviceTotalMem(
        &max_alloc_size, device)); /* The test assumes that all devices have same total memory */
    max_alloc_size *= 0.4;         /* For allocacting more than half,
                                      we need the fix in CUDA driver,
                                      available only in r460 and later */
    DEBUG_PRINT("symmetric size requested %lu\n", max_alloc_size);
    sprintf(size_string, "%lu", max_alloc_size);
    status = setenv("NVSHMEM_SYMMETRIC_SIZE", size_string, 1);
    if (status) {
        fprintf(stderr, "setenv failed \n");
        status = -1;
        goto out;
    }

    init_wrapper(&argc, &argv);
    mype = nvshmem_my_pe();

    malloc_size = min_malloc_size;
    total_alloc_size = 0;
    while (true) {
        total_alloc_size += malloc_size;
        if (total_alloc_size > max_alloc_size) break;
        malloc_size *= 2;
        loop_size++;
    }

    h_size_arr = (uint64_t *)malloc(sizeof(uint64_t) * loop_size);
    h_time = (double *)malloc(sizeof(double) * loop_size);

    malloc_size = min_malloc_size;
    for (int i = 0; i < loop_size; i++) {
        gettimeofday(&t_start, NULL);
        nvshmem_malloc(malloc_size);
        gettimeofday(&t_stop, NULL);
        h_size_arr[i] = malloc_size;
        h_time[i] =
            ((t_stop.tv_usec - t_start.tv_usec) + (1e+6 * (t_stop.tv_sec - t_start.tv_sec)));
        malloc_size *= 2;
    }
    if (!mype) {
        print_table_v1("malloc", "None", "size (Bytes)", "time", "us", '-', h_size_arr, h_time,
                       loop_size);
    }

    finalize_wrapper();
out:
    return status;
}
