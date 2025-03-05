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

#include "coll_test.h"

#define BARRIER_KERNEL(TG_PRE, THREADGROUP, THREAD_COMP)                                   \
    __global__ void test_barrier_call_kernel##THREADGROUP(nvshmem_team_t team, int iter) { \
        int i;                                                                             \
        if (!blockIdx.x && (threadIdx.x < THREAD_COMP)) {                                  \
            for (i = 0; i < iter; i++) {                                                   \
                nvshmem##TG_PRE##_barrier##THREADGROUP(team);                              \
            }                                                                              \
        }                                                                                  \
    }

#define BARRIER_ALL_KERNEL(TG_PRE, THREADGROUP, THREAD_COMP)              \
    __global__ void test_barrier_all_call_kernel##THREADGROUP(int iter) { \
        int i;                                                            \
        if (!blockIdx.x && (threadIdx.x < THREAD_COMP)) {                 \
            for (i = 0; i < iter; i++) {                                  \
                nvshmem##TG_PRE##_barrier_all##THREADGROUP();             \
            }                                                             \
        }                                                                 \
    }

BARRIER_KERNEL(, , 1);
BARRIER_KERNEL(x, _warp, warpSize);
BARRIER_KERNEL(x, _block, INT_MAX);

BARRIER_ALL_KERNEL(, , 1);
BARRIER_ALL_KERNEL(x, _warp, warpSize);
BARRIER_ALL_KERNEL(x, _block, INT_MAX);

int barrier_calling_kernel(nvshmem_team_t team, cudaStream_t stream, int mype, void **h_tables) {
    int status = 0;
    int nvshm_test_num_tpb = TEST_NUM_TPB_BLOCK;
    int skip = MAX_SKIP;
    int iter = MAX_ITERS;
    int num_blocks = 1;
    double *h_thread_lat = (double *)h_tables[0];
    double *h_warp_lat = (double *)h_tables[1];
    double *h_block_lat = (double *)h_tables[2];
    uint64_t num_tpb = TEST_NUM_TPB_BLOCK;
    void *barrier_args_1[] = {&team, &skip};
    void *barrier_args_2[] = {&team, &iter};
    void *barrier_all_args_1[] = {&skip};
    void *barrier_all_args_2[] = {&iter};
    float milliseconds;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    nvshmem_barrier_all();
    status = nvshmemx_collective_launch((const void *)test_barrier_call_kernel, num_blocks,
                                        nvshm_test_num_tpb, barrier_args_1, 0, stream);
    if (status != NVSHMEMX_SUCCESS) {
        fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
        exit(-1);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    nvshmem_barrier_all();

    cudaEventRecord(start, stream);
    status = nvshmemx_collective_launch((const void *)test_barrier_call_kernel, num_blocks,
                                        nvshm_test_num_tpb, barrier_args_2, 0, stream);
    if (status != NVSHMEMX_SUCCESS) {
        fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
        exit(-1);
    }
    cudaEventRecord(stop, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (!mype) {
        cudaEventElapsedTime(&milliseconds, start, stop);
        h_thread_lat[0] = (milliseconds * 1000.0) / (float)iter;
    }

    nvshmem_barrier_all();
    status = nvshmemx_collective_launch((const void *)test_barrier_call_kernel_warp, num_blocks,
                                        nvshm_test_num_tpb, barrier_args_1, 0, stream);
    if (status != NVSHMEMX_SUCCESS) {
        fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
        exit(-1);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    nvshmem_barrier_all();

    cudaEventRecord(start, stream);
    status = nvshmemx_collective_launch((const void *)test_barrier_call_kernel_warp, num_blocks,
                                        nvshm_test_num_tpb, barrier_args_2, 0, stream);
    if (status != NVSHMEMX_SUCCESS) {
        fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
        exit(-1);
    }
    cudaEventRecord(stop, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (!mype) {
        cudaEventElapsedTime(&milliseconds, start, stop);
        h_warp_lat[0] = (milliseconds * 1000.0) / (float)iter;
    }

    nvshmem_barrier_all();
    status = nvshmemx_collective_launch((const void *)test_barrier_call_kernel_block, num_blocks,
                                        nvshm_test_num_tpb, barrier_args_1, 0, stream);
    if (status != NVSHMEMX_SUCCESS) {
        fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
        exit(-1);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    nvshmem_barrier_all();

    cudaEventRecord(start, stream);
    status = nvshmemx_collective_launch((const void *)test_barrier_call_kernel_block, num_blocks,
                                        nvshm_test_num_tpb, barrier_args_2, 0, stream);
    if (status != NVSHMEMX_SUCCESS) {
        fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
        exit(-1);
    }
    cudaEventRecord(stop, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (!mype) {
        cudaEventElapsedTime(&milliseconds, start, stop);
        h_block_lat[0] = (milliseconds * 1000.0) / (float)iter;
    }

    if (!mype) {
        print_table_basic("barrier_device", "thread", "threads per block", "latency", "us", '-',
                          &num_tpb, h_thread_lat, 1);
        print_table_basic("barrier_device", "warp", "threads per block", "latency", "us", '-',
                          &num_tpb, h_warp_lat, 1);
        print_table_basic("barrier_device", "block", "threads per block", "latency", "us", '-',
                          &num_tpb, h_block_lat, 1);
    }

    nvshmem_barrier_all();
    status = nvshmemx_collective_launch((const void *)test_barrier_all_call_kernel, num_blocks,
                                        nvshm_test_num_tpb, barrier_all_args_1, 0, stream);
    if (status != NVSHMEMX_SUCCESS) {
        fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
        exit(-1);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    nvshmem_barrier_all();

    cudaEventRecord(start, stream);
    status = nvshmemx_collective_launch((const void *)test_barrier_all_call_kernel, num_blocks,
                                        nvshm_test_num_tpb, barrier_all_args_2, 0, stream);
    if (status != NVSHMEMX_SUCCESS) {
        fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
        exit(-1);
    }
    cudaEventRecord(stop, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (!mype) {
        cudaEventElapsedTime(&milliseconds, start, stop);
        h_thread_lat[0] = (milliseconds * 1000.0) / (float)iter;
    }

    nvshmem_barrier_all();
    status = nvshmemx_collective_launch((const void *)test_barrier_all_call_kernel_warp, num_blocks,
                                        nvshm_test_num_tpb, barrier_all_args_1, 0, stream);
    if (status != NVSHMEMX_SUCCESS) {
        fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
        exit(-1);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    nvshmem_barrier_all();

    cudaEventRecord(start, stream);
    status = nvshmemx_collective_launch((const void *)test_barrier_all_call_kernel_warp, num_blocks,
                                        nvshm_test_num_tpb, barrier_all_args_2, 0, stream);
    if (status != NVSHMEMX_SUCCESS) {
        fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
        exit(-1);
    }
    cudaEventRecord(stop, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (!mype) {
        cudaEventElapsedTime(&milliseconds, start, stop);
        h_warp_lat[0] = (milliseconds * 1000.0) / (float)iter;
    }

    nvshmem_barrier_all();
    status =
        nvshmemx_collective_launch((const void *)test_barrier_all_call_kernel_block, num_blocks,
                                   nvshm_test_num_tpb, barrier_all_args_1, 0, stream);
    if (status != NVSHMEMX_SUCCESS) {
        fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
        exit(-1);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    nvshmem_barrier_all();

    cudaEventRecord(start, stream);
    status =
        nvshmemx_collective_launch((const void *)test_barrier_all_call_kernel_block, num_blocks,
                                   nvshm_test_num_tpb, barrier_all_args_2, 0, stream);
    if (status != NVSHMEMX_SUCCESS) {
        fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
        exit(-1);
    }
    cudaEventRecord(stop, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (!mype) {
        cudaEventElapsedTime(&milliseconds, start, stop);
        h_block_lat[0] = (milliseconds * 1000.0) / (float)iter;
    }

    if (!mype) {
        print_table_basic("barrier_all_device", "thread", "threads per block", "latency", "us", '-',
                          &num_tpb, h_thread_lat, 1);
        print_table_basic("barrier_all_device", "warp", "threads per block", "latency", "us", '-',
                          &num_tpb, h_warp_lat, 1);
        print_table_basic("barrier_all_device", "block", "threads per block", "latency", "us", '-',
                          &num_tpb, h_block_lat, 1);
    }

    return status;
}

int main(int argc, char **argv) {
    int mype;
    cudaStream_t cstrm;
    void **h_tables;

    init_wrapper(&argc, &argv);
    alloc_tables(&h_tables, 3, 1);

    mype = nvshmem_my_pe();
    CUDA_CHECK(cudaStreamCreateWithFlags(&cstrm, cudaStreamNonBlocking));

    barrier_calling_kernel(NVSHMEM_TEAM_WORLD, cstrm, mype, h_tables);

    nvshmem_barrier_all();

    CUDA_CHECK(cudaStreamDestroy(cstrm));
    free_tables(h_tables, 3);
    finalize_wrapper();

    return 0;
}
