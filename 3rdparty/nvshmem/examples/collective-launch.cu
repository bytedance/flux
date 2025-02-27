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
#include "nvshmem.h"
#include "nvshmemx.h"

#ifdef NVSHMEMTEST_MPI_SUPPORT
#include "mpi.h"
#endif

#undef CUDA_CHECK
#define CUDA_CHECK(stmt)                                                          \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (cudaSuccess != result) {                                              \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
            exit(-1);                                                             \
        }                                                                         \
    } while (0)

#define NVSHMEM_CHECK(stmt)                                                                \
    do {                                                                                   \
        int result = (stmt);                                                               \
        if (NVSHMEMX_SUCCESS != result) {                                                  \
            fprintf(stderr, "[%s:%d] nvshmem failed with error %d \n", __FILE__, __LINE__, \
                    result);                                                               \
            exit(-1);                                                                      \
        }                                                                                  \
    } while (0)

__global__ void reduce_ring(int *target, int mype, int npes) {
    int peer = (mype + 1) % npes;
    int lvalue = mype;

    for (int i = 0; i < npes; i++) {
        nvshmem_int_p(target, lvalue, peer);
        nvshmem_barrier_all();
        lvalue = *target + mype;
        nvshmem_barrier_all();
    }
}

int main(int c, char *v[]) {
    int mype, npes, mype_node;

#ifdef NVSHMEMTEST_MPI_SUPPORT
    bool use_mpi = false;
    char *value = getenv("NVSHMEMTEST_USE_MPI_LAUNCHER");
    if (value) use_mpi = atoi(value);
#endif

#ifdef NVSHMEMTEST_MPI_SUPPORT
    if (use_mpi) {
        MPI_Init(&c, &v);
        int rank, nranks;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nranks);
        MPI_Comm mpi_comm = MPI_COMM_WORLD;

        nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
        attr.mpi_comm = &mpi_comm;
        nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    } else
        nvshmem_init();
#else
    nvshmem_init();
#endif

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

    // application picks the device each PE will use
    CUDA_CHECK(cudaSetDevice(mype_node));
    int *u = (int *)nvshmem_calloc(1, sizeof(int));
    int *h = (int *)calloc(1, sizeof(int));

    void *args[] = {&u, &mype, &npes};
    dim3 dimBlock(1);
    dim3 dimGrid(1);

    NVSHMEM_CHECK(
        nvshmemx_collective_launch((const void *)reduce_ring, dimGrid, dimBlock, args, 0, 0));
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaMemcpy(h, u, sizeof(int), cudaMemcpyDeviceToHost);
    printf("results on device [%d] is %d \n", mype, h[0]);

    nvshmem_free(u);
    free(h);
    nvshmem_finalize();

#ifdef NVSHMEMTEST_MPI_SUPPORT
    if (use_mpi) MPI_Finalize();
#endif

    return 0;
}
