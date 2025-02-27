/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _NVSHMEMI_DEVICE_STATE_H_
#define _NVSHMEMI_DEVICE_STATE_H_

#include <cuda_runtime.h>
#include <stdio.h>

int nvshmemi_setup_collective_launch();
int nvshmemi_teardown_collective_launch();

void nvshmemi_check_state_and_init_d();
typedef struct {
    cudaStream_t stream;
    cudaEvent_t begin_event;
    cudaEvent_t end_event;
} collective_launch_params_t;

typedef struct {
    int multi_processor_count;
    int cooperative_launch;
} cuda_device_attributes_t;

typedef struct nvshmemi_device_state {
    bool is_initialized;
    int cuda_device_id;
    cuda_device_attributes_t cu_dev_attrib;
    collective_launch_params_t claunch_params;
} nvshmemi_device_state_t;

extern nvshmemi_device_state_t nvshmemi_device_only_state;

#endif
