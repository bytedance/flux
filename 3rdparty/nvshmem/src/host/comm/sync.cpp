/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include <cuda_runtime.h>                  // for cudaMemcpy, cud...
#include <driver_types.h>                  // for CUstream_st
#include <stddef.h>                        // for size_t, NULL
#include <stdint.h>                        // for uint64_t
#include "device_host/nvshmem_common.cuh"  // for NVSHMEMI_REPT_F...
#include "device_host_transport/nvshmem_common_transport.h"
#include "device_host_transport/nvshmem_constants.h"  // for NVSHMEM_SIGNAL_SET
#include "host/nvshmem_api.h"                         // for nvshmem_signal_...
#include "host/nvshmemx_api.h"                        // for nvshmemx_int32_...
#include "non_abi/nvshmemx_error.h"                   // for NVSHMEMI_NZ_EXIT
#include "internal/host/nvshmem_internal.h"           // for nvshmemi_signal...
#include "internal/host/cuda_interface_sync.h"        // for call_nvshmemi_i...
#include "internal/host/nvshmem_nvtx.hpp"             // for nvtx_cond_range
#include "internal/host/nvshmemi_symmetric_heap.hpp"  // for nvshmemi_symmet...
#include "internal/host/nvshmemi_types.h"             // for nvshmemi_state
#include "internal/host/util.h"                       // for NVSHMEM_API_NOT...

#define NVSHMEMX_TYPE_WAIT_UNTIL_ON_STREAM(type, TYPE)                                     \
    void nvshmemx_##type##_wait_until_on_stream(TYPE *ivar, int cmp, TYPE cmp_value,       \
                                                cudaStream_t cstream) {                    \
        NVTX_FUNC_RANGE_IN_GROUP(WAIT_ON_STREAM);                                          \
        NVSHMEM_API_NOT_SUPPORTED_WITH_LIMITED_MPG_RUNS();                                 \
        call_nvshmemi_##type##_wait_until_on_stream_kernel(ivar, cmp, cmp_value, cstream); \
    }
NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMX_TYPE_WAIT_UNTIL_ON_STREAM)
#undef NVSHMEMX_TYPE_WAIT_UNTIL_ON_STREAM

#define NVSHMEMX_TYPE_WAIT_UNTIL_ALL_ON_STREAM(type, TYPE)                                         \
    void nvshmemx_##type##_wait_until_all_on_stream(TYPE *ivars, size_t nelems, const int *status, \
                                                    int cmp, TYPE cmp_value,                       \
                                                    cudaStream_t cstream) {                        \
        NVTX_FUNC_RANGE_IN_GROUP(WAIT_ON_STREAM);                                                  \
        NVSHMEM_API_NOT_SUPPORTED_WITH_LIMITED_MPG_RUNS();                                         \
        call_nvshmemi_##type##_wait_until_all_on_stream_kernel(ivars, nelems, status, cmp,         \
                                                               cmp_value, cstream);                \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMX_TYPE_WAIT_UNTIL_ALL_ON_STREAM)
#undef NVSHMEMX_TYPE_WAIT_UNTIL_ALL_ON_STREAM

#define NVSHMEMX_TYPE_WAIT_UNTIL_ALL_VECTOR_ON_STREAM(type, TYPE)                                 \
    void nvshmemx_##type##_wait_until_all_vector_on_stream(                                       \
        TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE *cmp_value,                  \
        cudaStream_t cstream) {                                                                   \
        NVTX_FUNC_RANGE_IN_GROUP(WAIT_ON_STREAM);                                                 \
        NVSHMEM_API_NOT_SUPPORTED_WITH_LIMITED_MPG_RUNS();                                        \
        call_nvshmemi_##type##_wait_until_all_vector_on_stream_kernel(ivars, nelems, status, cmp, \
                                                                      cmp_value, cstream);        \
    }

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMX_TYPE_WAIT_UNTIL_ALL_VECTOR_ON_STREAM)
#undef NVSHMEMX_TYPE_WAIT_UNTIL_ALL_VECTOR_ON_STREAM

void nvshmemx_signal_wait_until_on_stream(uint64_t *sig_addr, int cmp, uint64_t cmp_value,
                                          cudaStream_t cstream) {
    NVTX_FUNC_RANGE_IN_GROUP(WAIT_ON_STREAM);
    NVSHMEM_API_NOT_SUPPORTED_WITH_LIMITED_MPG_RUNS();
    int status = 0;
    if (((cmp == NVSHMEM_CMP_GE) || (cmp == NVSHMEM_CMP_EQ)) &&
        nvshmemi_can_use_cuda_64_bit_stream_memops &&
        (nvshmemi_can_flush_remote_writes || nvshmemi_options.BYPASS_FLUSH) &&
        CUPFN(nvshmemi_cuda_syms, cuStreamWaitValue64)) {
        if (cmp == NVSHMEM_CMP_GE)
            status =
                CUPFN(nvshmemi_cuda_syms,
                      cuStreamWaitValue64(cstream, (CUdeviceptr)sig_addr, cmp_value,
                                          CU_STREAM_WAIT_VALUE_GEQ | CU_STREAM_WAIT_VALUE_FLUSH));
        else {  // cmp == NVSHMEM_CMP_EQ
            status =
                CUPFN(nvshmemi_cuda_syms,
                      cuStreamWaitValue64(cstream, (CUdeviceptr)sig_addr, cmp_value,
                                          CU_STREAM_WAIT_VALUE_EQ | CU_STREAM_WAIT_VALUE_FLUSH));
        }
        NVSHMEMI_NZ_EXIT(status, "cuStreamWaitValue64() failed\n");
    } else {
        call_nvshmemi_signal_wait_until_on_stream_kernel(sig_addr, cmp, cmp_value, cstream);
    }
}

void nvshmemi_signal_op_on_stream(uint64_t *sig_addr, uint64_t signal, int sig_op, int pe,
                                  cudaStream_t cstrm) {
    int status = 0;
    if (sig_op == NVSHMEMI_AMO_SIGNAL_SET &&
        nvshmemi_state->heap_obj->get_local_pe_base()[pe] != NULL) {
        void *peer_addr;
        NVSHMEMU_MAPPED_PTR_TRANSLATE(peer_addr, sig_addr, pe)
        if (nvshmemi_can_use_cuda_64_bit_stream_memops &&
            nvshmemi_job_connectivity == NVSHMEMI_JOB_GPU_LDST_ATOMICS &&
            CUPFN(nvshmemi_cuda_syms, cuStreamWriteValue64)) {
            status = CUPFN(nvshmemi_cuda_syms,
                           cuStreamWriteValue64(cstrm, (CUdeviceptr)peer_addr, signal, 0));
            NVSHMEMI_NZ_EXIT(status, "cuStreamWriteValue64() failed\n");
        } else {
            status = cudaMemcpyAsync(peer_addr, (const void *)&signal, sizeof(uint64_t),
                                     cudaMemcpyHostToDevice, cstrm);
            NVSHMEMI_NZ_EXIT(status, "cudaMemcpyAsync() failed\n");
        }
    } else {
        call_nvshmemi_signal_op_kernel(sig_addr, signal, sig_op, pe, cstrm);
    }
}

void nvshmemx_signal_op_on_stream(uint64_t *sig_addr, uint64_t signal, int sig_op, int pe,
                                  cudaStream_t cstrm) {
    nvshmemi_signal_op_on_stream(sig_addr, signal, sig_op, pe, cstrm);
}

uint64_t nvshmem_signal_fetch(uint64_t *sig_addr) {
    uint64_t signal;
    CUDA_RUNTIME_CHECK(cudaMemcpy(&signal, sig_addr, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    return signal;
}
