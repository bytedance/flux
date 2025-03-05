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
#define LARGEST_DT double2

#ifdef MAX_ITERS
#undef MAX_ITERS
#endif
#define MAX_ITERS 50

#define CALL_RDXN(TG_PRE, TG, TYPENAME, TYPE, OP, THREAD_COMP, ELEM_COMP)                     \
    __global__ void test_##TYPENAME##_##OP##_reduce_kern##TG(                                 \
        nvshmem_team_t team, TYPE *dest, const TYPE *source, int nelems, int iter) {          \
        int i;                                                                                \
                                                                                              \
        if (!blockIdx.x && (threadIdx.x < THREAD_COMP) && (nelems < ELEM_COMP)) {             \
            for (i = 0; i < iter; i++) {                                                      \
                nvshmem##TG_PRE##_##TYPENAME##_##OP##_reduce##TG(team, dest, source, nelems); \
            }                                                                                 \
        }                                                                                     \
    }

#define CALL_RDXN_OPS_ALL_TG(TYPENAME, TYPE) \
    CALL_RDXN(x, _block, TYPENAME, TYPE, maxloc, INT_MAX, 2)

CALL_RDXN_OPS_ALL_TG(double2, double2)

#define SET_SIZE_ARR(TYPE, ELEM_COMP)                                \
    do {                                                             \
        j = 0;                                                       \
        for (num_elems = 1; num_elems < max_elems; num_elems *= 2) { \
            if (num_elems < ELEM_COMP) {                             \
                size_arr[j] = num_elems * sizeof(TYPE);              \
            } else {                                                 \
                size_arr[j] = 0;                                     \
            }                                                        \
            j++;                                                     \
        }                                                            \
    } while (0)

#define RUN_ITERS_OP(TYPENAME, TYPE, GROUP, OP, ELEM_COMP)                             \
    do {                                                                               \
        void *skip_arg_list[] = {&team, &dest, &source, &num_elems, &skip};            \
        void *time_arg_list[] = {&team, &dest, &source, &num_elems, &iter};            \
        float milliseconds;                                                            \
        cudaEvent_t start, stop;                                                       \
        cudaEventCreate(&start);                                                       \
        cudaEventCreate(&stop);                                                        \
        SET_SIZE_ARR(TYPE, ELEM_COMP);                                                 \
                                                                                       \
        nvshmem_barrier_all();                                                         \
        j = 0;                                                                         \
        for (num_elems = 1; num_elems < ELEM_COMP; num_elems *= 2) {                   \
            status = nvshmemx_collective_launch(                                       \
                (const void *)test_##TYPENAME##_##OP##_reduce_kern##GROUP, num_blocks, \
                nvshm_test_num_tpb, skip_arg_list, 0, stream);                         \
            if (status != NVSHMEMX_SUCCESS) {                                          \
                fprintf(stderr, "shmemx_collective_launch failed %d \n", status);      \
                exit(-1);                                                              \
            }                                                                          \
            CUDA_CHECK(cudaStreamSynchronize(stream));                                 \
            nvshmem_barrier_all();                                                     \
                                                                                       \
            cudaEventRecord(start, stream);                                            \
            status = nvshmemx_collective_launch(                                       \
                (const void *)test_##TYPENAME##_##OP##_reduce_kern##GROUP, num_blocks, \
                nvshm_test_num_tpb, time_arg_list, 0, stream);                         \
            if (status != NVSHMEMX_SUCCESS) {                                          \
                fprintf(stderr, "shmemx_collective_launch failed %d \n", status);      \
                exit(-1);                                                              \
            }                                                                          \
            cudaEventRecord(stop, stream);                                             \
            CUDA_CHECK(cudaStreamSynchronize(stream));                                 \
                                                                                       \
            if (!mype) {                                                               \
                cudaEventElapsedTime(&milliseconds, start, stop);                      \
                h_##OP##_lat[j] = (milliseconds * 1000.0) / (float)iter;               \
            }                                                                          \
            nvshmem_barrier_all();                                                     \
            j++;                                                                       \
        }                                                                              \
    } while (0)

#define RUN_ITERS(TYPENAME, TYPE, GROUP, ELEM_COMP) \
    RUN_ITERS_OP(TYPENAME, TYPE, GROUP, maxloc, ELEM_COMP);

int rdxn_calling_kernel(nvshmem_team_t team, void *dest, const void *source, int mype,
                        int max_elems, cudaStream_t stream, run_opt_t run_options,
                        void **h_tables) {
    int status = 0;
    int nvshm_test_num_tpb = TEST_NUM_TPB_BLOCK;
    int num_blocks = 1;
    int num_elems = 1;
    int iter = MAX_ITERS;
    int skip = MAX_SKIP;
    int j;
    uint64_t *size_arr = (uint64_t *)h_tables[0];
    double *h_maxloc_lat = (double *)h_tables[1];

    if (run_options.run_block) {
        RUN_ITERS(double2, double2, _block, max_elems);
        if (!mype) {
            print_table_v1("device_reduction", "double2-maxloc-b", "size (Bytes)", "latency", "us",
                           '-', size_arr, h_maxloc_lat, j);
        }
    }

    return status;
}

int main(int argc, char **argv) {
    int status = 0;
    int mype, array_size;
    size_t size = 0;
    size_t alloc_size;
    int num_elems;
    char *value = NULL;
    int max_elems = 2;  //(MAX_ELEMS / 2);
    int *h_buffer = NULL;
    int *d_source, *d_dest;
    int *h_source, *h_dest;
    char size_string[100];
    cudaStream_t cstrm;
    run_opt_t run_options;
    void **h_tables;

    PROCESS_OPTS(run_options);

    size = page_size_roundoff((MAX_ELEMS) * sizeof(LARGEST_DT));   // send buf
    size += page_size_roundoff((MAX_ELEMS) * sizeof(LARGEST_DT));  // recv buf

    DEBUG_PRINT("symmetric size requested %lu\n", size);
    sprintf(size_string, "%lu", size);

    status = setenv("NVSHMEM_SYMMETRIC_SIZE", size_string, 1);
    if (status) {
        fprintf(stderr, "setenv failed \n");
        status = -1;
        goto out;
    }

    value = getenv("NVSHMEM_PERF_COLL_MAX_ELEMS");

    if (NULL != value) {
        max_elems = atoi(value);
        if (0 == max_elems) {
            fprintf(stderr, "Warning: min max elem size = 1\n");
            max_elems = 1;
        }
    }

    array_size = floor(std::log2((float)max_elems)) + 1;

    init_wrapper(&argc, &argv);
    alloc_tables(&h_tables, 8, array_size);

    mype = nvshmem_my_pe();

    CUDA_CHECK(cudaStreamCreateWithFlags(&cstrm, cudaStreamNonBlocking));

    num_elems = 1;  // MAX_ELEMS / 2;
    alloc_size = (num_elems * 2) * sizeof(LARGEST_DT);

    CUDA_CHECK(cudaHostAlloc(&h_buffer, alloc_size, cudaHostAllocDefault));
    h_source = (int32_t *)h_buffer;
    h_dest = (int32_t *)&h_source[num_elems];

    d_source = (int32_t *)nvshmem_align(getpagesize(), num_elems * sizeof(LARGEST_DT));
    d_dest = (int32_t *)nvshmem_align(getpagesize(), num_elems * sizeof(LARGEST_DT));

    CUDA_CHECK(cudaMemcpyAsync(d_source, h_source, (sizeof(LARGEST_DT) * num_elems),
                               cudaMemcpyHostToDevice, cstrm));
    CUDA_CHECK(cudaMemcpyAsync(d_dest, h_dest, (sizeof(LARGEST_DT) * num_elems),
                               cudaMemcpyHostToDevice, cstrm));

    rdxn_calling_kernel(NVSHMEM_TEAM_WORLD, d_dest, d_source, mype, max_elems, cstrm, run_options,
                        h_tables);

    DEBUG_PRINT("last error = %s\n", cudaGetErrorString(cudaGetLastError()));

    CUDA_CHECK(cudaMemcpyAsync(h_source, d_source, (sizeof(LARGEST_DT) * num_elems),
                               cudaMemcpyDeviceToHost, cstrm));
    CUDA_CHECK(cudaMemcpyAsync(h_dest, d_dest, (sizeof(LARGEST_DT) * num_elems),
                               cudaMemcpyDeviceToHost, cstrm));

    nvshmem_barrier_all();

    CUDA_CHECK(cudaFreeHost(h_buffer));
    nvshmem_free(d_source);
    nvshmem_free(d_dest);

    CUDA_CHECK(cudaStreamDestroy(cstrm));

    finalize_wrapper();

out:
    return 0;
}
