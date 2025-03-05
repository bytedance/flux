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

#ifndef COLL_TEST_H
#define COLL_TEST_H

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <assert.h>
#include <unistd.h>
#include "utils.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <sys/time.h>

#define MAX_SKIP 16
#define MAX_ITERS 128
#define MAX_NPES 128
#define BARRIER_MAX_ITERS 1000
#define FCOLLECT_MAX_ITERS 1024
#define BARRIER_MAX_SKIP 10

extern int coll_max_iters;

#define alltoall_src_size(DATATYPE, num_elems, npes) (sizeof(DATATYPE) * num_elems * npes)

#define alltoall_dest_size(DATATYPE, num_elems, npes) (sizeof(DATATYPE) * num_elems * npes)

#define fcollect_src_size(DATATYPE, num_elems, npes) (sizeof(DATATYPE) * num_elems)

#define fcollect_dest_size(DATATYPE, num_elems, npes) (sizeof(DATATYPE) * num_elems * npes)

#define broadcast_src_size(DATATYPE, num_elems, npes) (sizeof(DATATYPE) * num_elems)

#define broadcast_dest_size(DATATYPE, num_elems, npes) (sizeof(DATATYPE) * num_elems)

#define call_shmem_broadcast(TYPENAME, TYPE, team, d_dest, d_source, num_elems, root) \
    do {                                                                              \
        nvshmem_##TYPENAME##_broadcast(team, d_dest, d_source, num_elems, root);      \
    } while (0)

#define call_shmem_fcollect(TYPENAME, TYPE, team, d_dest, d_source, num_elems, root) \
    do {                                                                             \
        nvshmem_##TYPENAME##_fcollect(team, d_dest, d_source, num_elems);            \
    } while (0)

#define call_shmem_alltoall(TYPENAME, TYPE, team, d_dest, d_source, num_elems, root) \
    do {                                                                             \
        nvshmem_##TYPENAME##_alltoall(team, d_dest, d_source, num_elems);            \
    } while (0)

#define call_shmem_broadcast_on_stream(TYPENAME, TYPE, team, d_dest, d_source, num_elems, root, \
                                       stream)                                                  \
    do {                                                                                        \
        nvshmemx_##TYPENAME##_broadcast_on_stream(team, d_dest, d_source, num_elems, root,      \
                                                  stream);                                      \
    } while (0)

#define call_shmem_fcollect_on_stream(TYPENAME, TYPE, team, d_dest, d_source, num_elems, root, \
                                      stream)                                                  \
    do {                                                                                       \
        nvshmemx_##TYPENAME##_fcollect_on_stream(team, d_dest, d_source, num_elems, stream);   \
    } while (0)

#define call_shmem_alltoall_on_stream(TYPENAME, TYPE, team, d_dest, d_source, num_elems, root, \
                                      stream)                                                  \
    do {                                                                                       \
        nvshmemx_##TYPENAME##_alltoall_on_stream(team, d_dest, d_source, num_elems, stream);   \
    } while (0)

#define RUN_COLL(coll, COLL, TYPENAME, TYPE, d_source, h_source, d_dest, h_dest, npes, root,       \
                 stream, size_array, latency_array)                                                \
    do {                                                                                           \
        int array_index = 0;                                                                       \
        for (num_elems = 1; num_elems < (MAX_ELEMS / 2); num_elems *= 2) {                         \
            int iters = 0;                                                                         \
            double latency;                                                                        \
            int skip = MAX_SKIP;                                                                   \
            struct timeval t_start, t_stop;                                                        \
            int latency_iters = 0;                                                                 \
            nvshmem_barrier_all();                                                                 \
            for (iters = 0; iters < skip; iters++) {                                               \
                call_shmem_##coll(TYPENAME, TYPE, NVSHMEM_TEAM_WORLD, d_dest, d_source, num_elems, \
                                  root);                                                           \
            }                                                                                      \
            nvshmem_barrier_all();                                                                 \
            auto lat_idx_array = latency_array[array_index];                                       \
            for (iters = 0; iters < coll_max_iters; iters++) {                                     \
                gettimeofday(&t_start, NULL);                                                      \
                call_shmem_##coll(TYPENAME, TYPE, NVSHMEM_TEAM_WORLD, d_dest, d_source, num_elems, \
                                  root);                                                           \
                gettimeofday(&t_stop, NULL);                                                       \
                latency = ((t_stop.tv_usec - t_start.tv_usec) +                                    \
                           (1e+6 * (t_stop.tv_sec - t_start.tv_sec)));                             \
                lat_idx_array[latency_iters] = latency;                                            \
                latency_iters++;                                                                   \
            }                                                                                      \
            nvshmem_barrier_all();                                                                 \
            size_array[array_index] = num_elems * sizeof(DATATYPE);                                \
            array_index++;                                                                         \
        }                                                                                          \
    } while (0)

#define RUN_COLL_ON_STREAM(coll, COLL, TYPENAME, TYPE, d_source, h_source, d_dest, h_dest, npes, \
                           root, stream, size_array, latency_array)                              \
    do {                                                                                         \
        int array_index = 0;                                                                     \
        for (num_elems = 1; num_elems < (MAX_ELEMS / 2); num_elems *= 2) {                       \
            float latency = 0;                                                                   \
            int skip = MAX_SKIP;                                                                 \
            cudaEvent_t t_start, t_stop;                                                         \
            CUDA_CHECK(cudaEventCreate(&t_start));                                               \
            CUDA_CHECK(cudaEventCreate(&t_stop));                                                \
            int latency_iters = 0;                                                               \
            auto lat_idx_array = latency_array[array_index];                                     \
            nvshmem_barrier_all();                                                               \
            for (int iters = 0; iters < skip; iters++) {                                         \
                call_shmem_##coll##_on_stream(TYPENAME, TYPE, NVSHMEM_TEAM_WORLD, d_dest,        \
                                              d_source, num_elems, root, stream);                \
            }                                                                                    \
            CUDA_CHECK(cudaStreamSynchronize(stream));                                           \
            nvshmem_barrier_all();                                                               \
            for (int iters = 0; iters < coll_max_iters; iters++) {                               \
                CUDA_CHECK(cudaEventRecord(t_start, stream));                                    \
                call_shmem_##coll##_on_stream(TYPENAME, TYPE, NVSHMEM_TEAM_WORLD, d_dest,        \
                                              d_source, num_elems, root, stream);                \
                CUDA_CHECK(cudaEventRecord(t_stop, stream));                                     \
                CUDA_CHECK(cudaStreamSynchronize(stream));                                       \
                CUDA_CHECK(cudaEventElapsedTime(&latency, t_start, t_stop));                     \
                lat_idx_array[latency_iters] = latency * 1e+3;                                   \
                latency_iters++;                                                                 \
            }                                                                                    \
            CUDA_CHECK(cudaEventDestroy(t_start));                                               \
            CUDA_CHECK(cudaEventDestroy(t_stop));                                                \
            nvshmem_barrier_all();                                                               \
            size_array[array_index] = num_elems * sizeof(TYPE);                                  \
            array_index++;                                                                       \
        }                                                                                        \
    } while (0)

#endif /*COLL_TEST_H*/
