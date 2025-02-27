/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See COPYRIGHT.txt for license information
 */

#ifndef _NVSHMEMX_COLLECTIVE_LAUNCH_APIS_H_
#define _NVSHMEMX_COLLECTIVE_LAUNCH_APIS_H_

#include <cuda_runtime.h>

int nvshmemx_collective_launch(const void *func, dim3 gridDims, dim3 blockDims, void **args,
                               size_t sharedMem, cudaStream_t stream);
int nvshmemx_collective_launch_query_gridsize(const void *func, dim3 blockDims, void **args,
                                              size_t sharedMem, int *gridsize);

#endif