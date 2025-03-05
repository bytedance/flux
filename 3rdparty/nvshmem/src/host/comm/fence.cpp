/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "host/nvshmem_api.h"  // IWYU pragma: keep
#include <cuda_runtime.h>
#include <driver_types.h>

#include "internal/host/nvshmem_internal.h"
#include "internal/host/nvshmemi_types.h"
#include "internal/host/nvshmem_nvtx.hpp"
#include "non_abi/nvshmemx_error.h"
#include "internal/host_transport/transport.h"
#include "internal/host/util.h"

void nvshmem_fence(void) {
    NVTX_FUNC_RANGE_IN_GROUP(MEMORDER);
    NVSHMEMI_CHECK_INIT_STATUS();

    int status;
    int tbitmap = nvshmemi_state->transport_bitmap;
    for (int j = 0; j < nvshmemi_state->num_initialized_transports; j++) {
        if (tbitmap & 1) {
            struct nvshmem_transport *tcurr =
                ((nvshmem_transport_t *)nvshmemi_state->transports)[j];
            if ((tcurr->attr & NVSHMEM_TRANSPORT_ATTR_NO_ENDPOINTS)) {
                for (int s = 0; s < MAX_PEER_STREAMS; s++) {
                    cudaStream_t custrm = nvshmemi_state->custreams[s];
                    CUDA_RUNTIME_CHECK_GOTO(cudaStreamSynchronize(custrm), status, out);
                }
            } else if (tcurr->host_ops.fence) {
                for (int k = 0; k < nvshmemi_state->npes; k++) {
                    status = tcurr->host_ops.fence(tcurr, k, 0);
                    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                          "nvshmem_fence() failed \n");
                }
            }
        }
        tbitmap >>= 1;
    }
out:
    return;
}
