/* foo
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
int coll_max_iters = FCOLLECT_MAX_ITERS;

int main(int argc, char **argv) {
    int status = 0;
    int mype, npes;
    uint64_t size = MAX_ELEMS * (MAX_NPES + 1) * sizeof(DATATYPE);
    size_t alloc_size;
    int num_elems;
    char *user_iters = NULL;
    DATATYPE *h_buffer = NULL;
    DATATYPE *d_buffer = NULL;
    DATATYPE *d_source, *d_dest;
    DATATYPE *h_source, *h_dest;
    char size_string[100];
    uint64_t size_array[MAX_ELEMS_LOG + 1];
    double **latency_array = (double **)malloc((MAX_ELEMS_LOG + 1) * sizeof(double *));
    cudaStream_t stream;

    memset(size_array, 0, (MAX_ELEMS_LOG + 1) * sizeof(uint64_t));
    for (int i = 0; i < MAX_ELEMS_LOG + 1; i++) {
        latency_array[i] = (double *)calloc(coll_max_iters, sizeof(double));
    }

    DEBUG_PRINT("symmetric size requested %lu\n", size);
    sprintf(size_string, "%lu", size);

    status = setenv("NVSHMEM_SYMMETRIC_SIZE", size_string, 1);
    if (status) {
        fprintf(stderr, "setenv failed \n");
        status = -1;
        goto out;
    }

    user_iters = getenv("NVSHMEMTEST_FCOLLECT_MAX_ITERS");
    if (user_iters != NULL) {
        coll_max_iters = atoi(user_iters);
        free(user_iters);
    }

    init_wrapper(&argc, &argv);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    assert(npes <= MAX_NPES);
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    num_elems = MAX_ELEMS / 2;
    alloc_size = num_elems * (npes + 1) * sizeof(DATATYPE);

    CUDA_CHECK(cudaHostAlloc(&h_buffer, alloc_size, cudaHostAllocDefault));
    h_source = (DATATYPE *)h_buffer;
    h_dest = (DATATYPE *)&h_source[num_elems];

    d_buffer = (DATATYPE *)nvshmem_malloc(alloc_size);
    if (!d_buffer) {
        fprintf(stderr, "nvshmem_malloc failed \n");
        status = -1;
        goto out;
    }

    d_source = (DATATYPE *)d_buffer;
    d_dest = (DATATYPE *)&d_source[num_elems];

    RUN_COLL_ON_STREAM(fcollect, FCOLLECT, int32, int32_t, (int32_t *)d_source, (int32_t *)h_source,
                       (int32_t *)d_dest, (int32_t *)h_dest, npes, -1, stream, size_array,
                       latency_array);
    if (!mype) {
        print_table_v2("fcollect_on_stream", "32-bit", "size (bytes)", "latency", "us", '-',
                       size_array, latency_array, MAX_ELEMS_LOG + 1);
    }

    RUN_COLL_ON_STREAM(fcollect, FCOLLECT, int64, int64_t, d_source, h_source, d_dest, h_dest, npes,
                       -1, stream, size_array, latency_array);
    if (!mype) {
        print_table_v2("fcollect_on_stream", "64-bit", "size (bytes)", "latency", "us", '-',
                       size_array, latency_array, MAX_ELEMS_LOG + 1);
    }

    nvshmem_barrier_all();

    CUDA_CHECK(cudaFreeHost(h_buffer));
    nvshmem_free(d_buffer);

    CUDA_CHECK(cudaStreamDestroy(stream));

    finalize_wrapper();

out:
    return 0;
}
