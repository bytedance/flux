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

#include "coll_test.h"
int coll_max_iters = BARRIER_MAX_ITERS;

int main(int c, char *v[]) {
    int status = 0;
    int mype;
    size_t size = 1;
    int iters = BARRIER_MAX_ITERS;
    int skip = BARRIER_MAX_SKIP;
    float ms = 0;
    cudaEvent_t start_event, stop_event;
    cudaStream_t stream;
    double latency_value;

    init_wrapper(&c, &v);

    mype = nvshmem_my_pe();
#ifdef _NVSHMEM_DEBUG
    int npes = nvshmem_n_pes();
#endif
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    DEBUG_PRINT("SHMEM: [%d of %d] hello shmem world! \n", mype, npes);

    for (iters = 0; iters < skip; iters++) {
        nvshmemx_barrier_all_on_stream(stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    nvshmem_barrier_all();

    CUDA_CHECK(cudaEventRecord(start_event, stream));
    for (iters = 0; iters < coll_max_iters; iters++) {
        nvshmemx_barrier_all_on_stream(stream);
    }
    CUDA_CHECK(cudaEventRecord(stop_event, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));

    if (!mype) {
        latency_value = (ms / coll_max_iters) * 1000;
        print_table_basic("barrier_all_on_stream", "None", "size (Bytes)", "latency", "us", '-',
                          &size, &latency_value, 1);
    }

    nvshmem_barrier_all();

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));

    finalize_wrapper();

    return status;
}
