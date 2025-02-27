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
#define DATATYPE int64_t

#define CALL_BCAST(TYPENAME, TYPE, TG_PRE, THREADGROUP, THREAD_COMP, ELEM_COMP)                   \
    __global__ void test_##TYPENAME##_bcast_call_kern##THREADGROUP(                               \
        nvshmem_team_t team, TYPE *dest, const TYPE *source, int nelems, int PE_root, int mype,   \
        int iter) {                                                                               \
        int i;                                                                                    \
                                                                                                  \
        if (!blockIdx.x && (threadIdx.x < THREAD_COMP) && (nelems < ELEM_COMP)) {                 \
            for (i = 0; i < iter; i++) {                                                          \
                nvshmem##TG_PRE##_##TYPENAME##_broadcast##THREADGROUP(team, dest, source, nelems, \
                                                                      PE_root);                   \
            }                                                                                     \
        }                                                                                         \
    }

CALL_BCAST(int32, int32_t, , , 1, 512);
CALL_BCAST(int64, int64_t, , , 1, 512);
CALL_BCAST(int32, int32_t, x, _warp, warpSize, 4096);
CALL_BCAST(int64, int64_t, x, _warp, warpSize, 4096);
CALL_BCAST(int32, int32_t, x, _block, INT_MAX, INT_MAX);
CALL_BCAST(int64, int64_t, x, _block, INT_MAX, INT_MAX);

int broadcast_calling_kernel(nvshmem_team_t team, void *dest, const void *source, int mype,
                             int PE_root, int max_elems, cudaStream_t stream, void **h_tables) {
    int status = 0;
    int nvshm_test_num_tpb = TEST_NUM_TPB_BLOCK;
    int num_blocks = 1;
    int num_elems = 1;
    int i;
    int skip = MAX_SKIP;
    int iter = MAX_ITERS;
    uint64_t *h_size_array = (uint64_t *)h_tables[0];
    double *h_thread_lat = (double *)h_tables[1];
    double *h_warp_lat = (double *)h_tables[2];
    double *h_block_lat = (double *)h_tables[3];
    float milliseconds;
    void *args_1[] = {&team, &dest, &source, &num_elems, &mype, &PE_root, &skip};
    void *args_2[] = {&team, &dest, &source, &num_elems, &mype, &PE_root, &iter};
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float *ms_d = (float *)nvshmem_malloc(sizeof(float));
    float *ms_sum_d = (float *)nvshmem_malloc(sizeof(float));

    nvshmem_barrier_all();
    i = 0;
    for (num_elems = 1; num_elems < 512; num_elems *= 2) {
        status = nvshmemx_collective_launch((const void *)test_int32_bcast_call_kern, num_blocks,
                                            nvshm_test_num_tpb, args_1, 0, stream);
        if (status != NVSHMEMX_SUCCESS) {
            fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
            exit(-1);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        nvshmem_barrier_all();

        cudaEventRecord(start, stream);
        status = nvshmemx_collective_launch((const void *)test_int32_bcast_call_kern, num_blocks,
                                            nvshm_test_num_tpb, args_2, 0, stream);
        if (status != NVSHMEMX_SUCCESS) {
            fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
            exit(-1);
        }
        cudaEventRecord(stop, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (!mype) {
            cudaEventElapsedTime(&milliseconds, start, stop);
            h_thread_lat[i] = (milliseconds * 1000.0) / (float)iter;
        }
        i++;
        nvshmem_barrier_all();
    }

    i = 0;
    for (num_elems = 1; num_elems < 4096; num_elems *= 2) {
        status = nvshmemx_collective_launch((const void *)test_int32_bcast_call_kern_warp,
                                            num_blocks, nvshm_test_num_tpb, args_1, 0, stream);
        if (status != NVSHMEMX_SUCCESS) {
            fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
            exit(-1);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        nvshmem_barrier_all();

        cudaEventRecord(start, stream);
        status = nvshmemx_collective_launch((const void *)test_int32_bcast_call_kern_warp,
                                            num_blocks, nvshm_test_num_tpb, args_2, 0, stream);
        if (status != NVSHMEMX_SUCCESS) {
            fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
            exit(-1);
        }
        cudaEventRecord(stop, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (!mype) {
            cudaEventElapsedTime(&milliseconds, start, stop);
            h_warp_lat[i] = (milliseconds * 1000.0) / (float)iter;
        }
        i++;
        nvshmem_barrier_all();
    }

    i = 0;
    for (num_elems = 1; num_elems < max_elems; num_elems *= 2) {
        h_size_array[i] = num_elems * 4;
        status = nvshmemx_collective_launch((const void *)test_int32_bcast_call_kern_block,
                                            num_blocks, nvshm_test_num_tpb, args_1, 0, stream);
        if (status != NVSHMEMX_SUCCESS) {
            fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
            exit(-1);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        nvshmem_barrier_all();

        cudaEventRecord(start, stream);
        status = nvshmemx_collective_launch((const void *)test_int32_bcast_call_kern_block,
                                            num_blocks, nvshm_test_num_tpb, args_2, 0, stream);
        if (status != NVSHMEMX_SUCCESS) {
            fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
            exit(-1);
        }
        cudaEventRecord(stop, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (!mype) {
            cudaEventElapsedTime(&milliseconds, start, stop);
            h_block_lat[i] = (milliseconds * 1000.0) / (float)iter;
        }
        i++;
        nvshmem_barrier_all();
    }

    if (!mype) {
        print_table_v1("bcast_device", "32-bit-thread", "size (Bytes)", "latency", "us", '-',
                       h_size_array, h_thread_lat, i);
        print_table_v1("bcast_device", "32-bit-warp", "size (Bytes)", "latency", "us", '-',
                       h_size_array, h_warp_lat, i);
        print_table_v1("bcast_device", "32-bit-block", "size (Bytes)", "latency", "us", '-',
                       h_size_array, h_block_lat, i);
    }

    i = 0;
    for (num_elems = 1; num_elems < 512; num_elems *= 2) {
        status = nvshmemx_collective_launch((const void *)test_int64_bcast_call_kern, num_blocks,
                                            nvshm_test_num_tpb, args_1, 0, stream);
        if (status != NVSHMEMX_SUCCESS) {
            fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
            exit(-1);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        nvshmem_barrier_all();

        cudaEventRecord(start, stream);
        status = nvshmemx_collective_launch((const void *)test_int64_bcast_call_kern, num_blocks,
                                            nvshm_test_num_tpb, args_2, 0, stream);
        if (status != NVSHMEMX_SUCCESS) {
            fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
            exit(-1);
        }
        cudaEventRecord(stop, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (!mype) {
            cudaEventElapsedTime(&milliseconds, start, stop);
            h_thread_lat[i] = (milliseconds * 1000.0) / (float)iter;
        }
        i++;
        nvshmem_barrier_all();
    }

    i = 0;
    for (num_elems = 1; num_elems < 4096; num_elems *= 2) {
        status = nvshmemx_collective_launch((const void *)test_int64_bcast_call_kern_warp,
                                            num_blocks, nvshm_test_num_tpb, args_1, 0, stream);
        if (status != NVSHMEMX_SUCCESS) {
            fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
            exit(-1);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        nvshmem_barrier_all();

        cudaEventRecord(start, stream);
        status = nvshmemx_collective_launch((const void *)test_int64_bcast_call_kern_warp,
                                            num_blocks, nvshm_test_num_tpb, args_2, 0, stream);
        if (status != NVSHMEMX_SUCCESS) {
            fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
            exit(-1);
        }
        cudaEventRecord(stop, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (!mype) {
            cudaEventElapsedTime(&milliseconds, start, stop);
            h_warp_lat[i] = (milliseconds * 1000.0) / (float)iter;
        }
        i++;
        nvshmem_barrier_all();
    }

    i = 0;
    for (num_elems = 1; num_elems < max_elems; num_elems *= 2) {
        h_size_array[i] = num_elems * 8;
        status = nvshmemx_collective_launch((const void *)test_int64_bcast_call_kern_block,
                                            num_blocks, nvshm_test_num_tpb, args_1, 0, stream);
        if (status != NVSHMEMX_SUCCESS) {
            fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
            exit(-1);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        nvshmem_barrier_all();

        cudaEventRecord(start, stream);
        status = nvshmemx_collective_launch((const void *)test_int64_bcast_call_kern_block,
                                            num_blocks, nvshm_test_num_tpb, args_2, 0, stream);
        if (status != NVSHMEMX_SUCCESS) {
            fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
            exit(-1);
        }
        cudaEventRecord(stop, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        cudaEventElapsedTime(&milliseconds, start, stop);
        cudaMemcpy(ms_d, &milliseconds, sizeof(float), cudaMemcpyHostToDevice);
        nvshmem_float_sum_reduce(NVSHMEM_TEAM_WORLD, ms_sum_d, ms_d, 1);
        cudaMemcpy(&milliseconds, ms_sum_d, sizeof(float), cudaMemcpyDeviceToHost);
        if (!mype) {
            h_block_lat[i] =
                (milliseconds * 1000.0) / ((float)iter * nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD));
        }
        i++;
        nvshmem_barrier_all();
    }

    if (!mype) {
        print_table_v1("bcast_device", "64-bit-thread", "size (Bytes)", "latency", "us", '-',
                       h_size_array, h_thread_lat, i);
        print_table_v1("bcast_device", "64-bit-warp", "size (Bytes)", "latency", "us", '-',
                       h_size_array, h_warp_lat, i);
        print_table_v1("bcast_device", "64-bit-block", "size (Bytes)", "latency", "us", '-',
                       h_size_array, h_block_lat, i);
    }

    return status;
}

int main(int argc, char **argv) {
    int status = 0;
    int mype, array_size, max_elems;
    char *value = NULL;
    size_t size = (MAX_ELEMS * 2) * sizeof(DATATYPE);
    size_t alloc_size;
    int num_elems;
    DATATYPE *buffer = NULL;
    DATATYPE *h_buffer = NULL;
    DATATYPE *d_source, *d_dest;
    DATATYPE *h_source, *h_dest;
    int root = 0;
    char size_string[100];
    cudaStream_t cstrm;
    void **h_tables;

    max_elems = (MAX_ELEMS / 2);

    if (NULL != value) {
        max_elems = atoi(value);
        if (0 == max_elems) {
            fprintf(stderr, "Warning: min max elem size = 1\n");
            max_elems = 1;
        }
    }

    array_size = floor(std::log2((float)max_elems)) + 1;

    DEBUG_PRINT("symmetric size %lu\n", size);
    sprintf(size_string, "%lu", size);

    status = setenv("NVSHMEM_SYMMETRIC_SIZE", size_string, 1);
    if (status) {
        fprintf(stderr, "setenv failed \n");
        status = -1;
        goto out;
    }

    init_wrapper(&argc, &argv);
    alloc_tables(&h_tables, 4, array_size);

    mype = nvshmem_my_pe();
    CUDA_CHECK(cudaStreamCreateWithFlags(&cstrm, cudaStreamNonBlocking));

    DEBUG_PRINT("SHMEM: [%d of %d] hello shmem world! \n", mype,
                nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD));

    num_elems = MAX_ELEMS / 2;
    alloc_size = (num_elems * 2) * sizeof(DATATYPE);

    CUDA_CHECK(cudaHostAlloc(&h_buffer, alloc_size, cudaHostAllocDefault));
    h_source = (DATATYPE *)h_buffer;
    h_dest = (DATATYPE *)&h_source[num_elems];

    buffer = (DATATYPE *)nvshmem_malloc(alloc_size);
    if (!buffer) {
        fprintf(stderr, "nvshmem_malloc failed \n");
        status = -1;
        goto out;
    }

    d_source = (DATATYPE *)buffer;
    d_dest = (DATATYPE *)&d_source[num_elems];

    for (int i = 0; i < num_elems; i++) {
        h_source[i] = i;
    }

    CUDA_CHECK(cudaMemcpyAsync(d_source, h_source, (sizeof(DATATYPE) * num_elems),
                               cudaMemcpyHostToDevice, cstrm));
    CUDA_CHECK(cudaMemcpyAsync(d_dest, h_dest, (sizeof(DATATYPE) * num_elems),
                               cudaMemcpyHostToDevice, cstrm));

    broadcast_calling_kernel(NVSHMEM_TEAM_WORLD, d_dest, d_source, mype, root, max_elems, cstrm,
                             h_tables);

    CUDA_CHECK(cudaMemcpyAsync(h_source, d_source, (sizeof(DATATYPE) * num_elems),
                               cudaMemcpyDeviceToHost, cstrm));
    CUDA_CHECK(cudaMemcpyAsync(h_dest, d_dest, (sizeof(DATATYPE) * num_elems),
                               cudaMemcpyDeviceToHost, cstrm));

    nvshmem_barrier_all();

    CUDA_CHECK(cudaFreeHost(h_buffer));
    nvshmem_free(buffer);

    CUDA_CHECK(cudaStreamDestroy(cstrm));
    free_tables(h_tables, 4);
    finalize_wrapper();

out:
    return 0;
}
