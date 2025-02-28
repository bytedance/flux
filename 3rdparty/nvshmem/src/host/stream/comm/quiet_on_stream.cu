/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "internal/host/util.h"
#include "internal/non_abi/nvshmemi_h_to_d_sync_defs.cuh"

static int nvshmemi_quiet_maxblocksize = -1;

void nvshmemi_call_proxy_quiet_entrypoint(cudaStream_t cstrm) {
    if (nvshmemi_quiet_maxblocksize == -1) {
        int tmp;
        CUDA_RUNTIME_CHECK(cudaOccupancyMaxPotentialBlockSize(
            &tmp, (int *)&nvshmemi_quiet_maxblocksize, nvshmemi_proxy_quiet_entrypoint));
    }
    int status = cudaLaunchKernel((const void *)nvshmemi_proxy_quiet_entrypoint, 1,
                                  nvshmemi_quiet_maxblocksize, NULL, 0, cstrm);
    if (status) {
        NVSHMEMI_ERROR_PRINT("cudaLaunchKernel() failed in nvshmem_quiet_on_stream \n");
    }
}
