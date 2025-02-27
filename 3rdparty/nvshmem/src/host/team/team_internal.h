/*
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _NVSHMEMI_TEAM_INTERNAL_H_
#define _NVSHMEMI_TEAM_INTERNAL_H_

#include <stddef.h>
#include "device_host/nvshmem_common.cuh"

template <typename T>
void nvshmemi_call_init_array_kernel(T *array, int len, T val);

template <typename TYPE, rdxn_ops_t OP>
void nvshmemi_call_reduce_kernel(int start, int stride, int size, TYPE *dst, const TYPE *source,
                                 size_t nreduce, TYPE *pWrk, volatile long *pSync,
                                 volatile long *sync_counter);

long *nvshmemi_team_get_sync_counter(nvshmemi_team_t *team);

#endif
