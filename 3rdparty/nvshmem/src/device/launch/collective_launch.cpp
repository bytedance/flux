/****
 * Copyright (c) 2017-2024, NVIDIA Corporation.  All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include <cuda.h>                                    // for CUDA_SUCCESS
#include <cuda_runtime.h>                            // for cudaGetErrorString
#include <driver_types.h>                            // for cudaError_t, cud...
#include <limits.h>                                  // for INT_MAX, INT_MIN
#include <stdio.h>                                   // for fprintf, stderr
#include <vector_types.h>                            // for dim3
#include "device/nvshmemx_collective_launch_apis.h"  // for nvshmemx_collect...
#include "internal/device/nvshmemi_device.h"         // for nvshmemi_device_...
#include "non_abi/nvshmemx_error.h"                  // for NVSHMEMI_NE_ERRO...

#define CUDA_RUNTIME_CHECK_GOTO(stmt, res, label)                                 \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (cudaSuccess != result) {                                              \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
            res = result;                                                         \
            goto label;                                                           \
        }                                                                         \
    } while (0)

void nvshmemi_check_state_and_init_d();

inline int nvshmemi_minv(int *vec, int count) {
    int minval = INT_MAX;
    for (int i = 0; i < count; i++) {
        if (vec[i] < minval) {
            minval = vec[i];
        }
    }
    return minval;
}

inline int nvshmemi_maxv(int *vec, int count) {
    int maxval = INT_MIN;
    for (int i = 0; i < count; i++) {
        if (vec[i] > maxval) {
            maxval = vec[i];
        }
    }
    return maxval;
}

static int _nvshmemi_collective_launch_query_gridsize(const void *func, dim3 blockDims, void **args,
                                                      size_t sharedMem, int *gridsize) {
    int multiProcessorCount;
    int blockSize = blockDims.x * blockDims.y * blockDims.z;
    int maxBlocksSM;
    int status = 0;

    nvshmemi_check_state_and_init_d();
    multiProcessorCount = nvshmemi_device_only_state.cu_dev_attrib.multi_processor_count;
    // get min blocks per SM, error out if 0 for any GPU
    status =
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksSM, func, blockSize, sharedMem);
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "cudaOccupancyMaxActiveBlocksPerMultiprocessor failed \n");

    // XXX: Returns maximum supported grid (including 0) on associated GPU
    *gridsize = maxBlocksSM * multiProcessorCount;  // XXX:caller chooses dimension of grid

out:
    return status;
}

static int _nvshmemi_collective_launch(const void *func, dim3 gridDims, dim3 blockDims, void **args,
                                       size_t sharedMem, cudaStream_t stream) {
    int multiProcessorCount;
    int blockSize = blockDims.x * blockDims.y * blockDims.z;
    int maxBlocksSM;
    int gridSize = -1;
    int launchFailed = 1;
    int status = 0;

    nvshmemi_check_state_and_init_d();
    // XXX: Supports the user passing a non-zero grid but of differing size across ranks
    if (gridDims.x == 0 && gridDims.y == 0 && gridDims.z == 0) {
        gridSize = 0;
    } else if (gridDims.x != 0 && gridDims.y != 0 && gridDims.z != 0) {
        gridSize = gridDims.x * gridDims.y * gridDims.z;
    }  // else
       // some but not all grid dim being 0 is illegal
       // XXX: if some ranks pass an illegal grid, others error out

    // get min blocks per SM, error out if 0 for any GPU
    status =
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksSM, func, blockSize, sharedMem);
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "cudaOccupancyMaxActiveBlocksPerMultiprocessor failed aborting job\n");

    multiProcessorCount = nvshmemi_device_only_state.cu_dev_attrib.multi_processor_count;
    if (gridSize == 0) { /*XXX : auto sizing */
        // two alternatives - run the minimum supported grid (>0) on all GPUs (global communication
        // needed) or run the maximum supported grid on each GPU (local decision)
        // XXX: Launches maximum supported grid (>0) on associated GPU
        if (maxBlocksSM > 0) { /*Launch will work only if all GPUs can run at least one CTA*/
            launchFailed = 0;
        }
        gridDims.x = maxBlocksSM * multiProcessorCount;
        gridDims.y = 1;
        gridDims.z = 1;
    } else if (gridSize > 0) { /* XXX : legal grid is provided by user*/
        if ((maxBlocksSM > 0) && (gridSize <= maxBlocksSM * multiProcessorCount)) { /*Works*/
            launchFailed = 0;
        }
    }

    /* TODO: make it obvious we aren't going to complete this call from this thread. Possibly global
     * exit? */
    NVSHMEMI_NZ_ERROR_JMP(launchFailed, NVSHMEMX_ERROR_COLLECTIVE_LAUNCH_FAILED, out,
                          "One or more PEs cannot launch \n");

    CUDA_RUNTIME_CHECK_GOTO(
        cudaEventRecord(nvshmemi_device_only_state.claunch_params.begin_event, stream), status,
        out);
    CUDA_RUNTIME_CHECK_GOTO(
        cudaStreamWaitEvent(nvshmemi_device_only_state.claunch_params.stream,
                            nvshmemi_device_only_state.claunch_params.begin_event, 0),
        status, out);

    if (nvshmemi_device_only_state.cu_dev_attrib.cooperative_launch) {
        status = cudaLaunchCooperativeKernel(func, gridDims, blockDims, args, sharedMem,
                                             nvshmemi_device_only_state.claunch_params.stream);
        NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_COLLECTIVE_LAUNCH_FAILED, out,
                              "Cooperative kernel launch failed \n");
    } else {
        status = cudaLaunchKernel(func, gridDims, blockDims, args, sharedMem,
                                  nvshmemi_device_only_state.claunch_params.stream);
        NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_COLLECTIVE_LAUNCH_FAILED, out,
                              "Kernel launch failed \n");
    }

    CUDA_RUNTIME_CHECK_GOTO(cudaEventRecord(nvshmemi_device_only_state.claunch_params.end_event,
                                            nvshmemi_device_only_state.claunch_params.stream),
                            status, out);

    CUDA_RUNTIME_CHECK_GOTO(
        cudaStreamWaitEvent(stream, nvshmemi_device_only_state.claunch_params.end_event, 0), status,
        out);

out:
    return status;
}

int nvshmemi_setup_collective_launch() {
    int leastPriority, greatestPriority, status = 0;
    CUDA_RUNTIME_CHECK_GOTO(
        cudaDeviceGetAttribute(&(nvshmemi_device_only_state.cu_dev_attrib.multi_processor_count),
                               cudaDevAttrMultiProcessorCount,
                               nvshmemi_device_only_state.cuda_device_id),
        status, out);

    CUDA_RUNTIME_CHECK_GOTO(
        cudaDeviceGetAttribute(&(nvshmemi_device_only_state.cu_dev_attrib.cooperative_launch),
                               cudaDevAttrCooperativeLaunch,
                               nvshmemi_device_only_state.cuda_device_id),
        status, out);

    if (!nvshmemi_device_only_state.cu_dev_attrib.cooperative_launch) {
        NVSHMEMI_WARN_PRINT(
            "Cooperative launch not supported on at least one PE; GPU-side synchronize may cause "
            "hang\n");
    }

    CUDA_RUNTIME_CHECK_GOTO(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority),
                            status, out);
    CUDA_RUNTIME_CHECK_GOTO(
        cudaStreamCreateWithPriority(&nvshmemi_device_only_state.claunch_params.stream,
                                     cudaStreamNonBlocking, greatestPriority),
        status, out);
    CUDA_RUNTIME_CHECK_GOTO(cudaEventCreate(&nvshmemi_device_only_state.claunch_params.begin_event,
                                            cudaEventDisableTiming),
                            status, out);
    CUDA_RUNTIME_CHECK_GOTO(cudaEventCreate(&nvshmemi_device_only_state.claunch_params.end_event,
                                            cudaEventDisableTiming),
                            status, out);

out:
    return status;
}

int nvshmemi_teardown_collective_launch() {
    int status = 0;

    if (!nvshmemi_device_only_state.is_initialized) goto out;

    CUDA_RUNTIME_CHECK_GOTO(cudaStreamDestroy(nvshmemi_device_only_state.claunch_params.stream),
                            status, out);
    CUDA_RUNTIME_CHECK_GOTO(cudaEventDestroy(nvshmemi_device_only_state.claunch_params.begin_event),
                            status, out);
    CUDA_RUNTIME_CHECK_GOTO(cudaEventDestroy(nvshmemi_device_only_state.claunch_params.end_event),
                            status, out);

out:
    return status;
}

int nvshmemx_collective_launch_query_gridsize(const void *func, dim3 blockDims, void **args,
                                              size_t sharedMem, int *gridsize) {
    return _nvshmemi_collective_launch_query_gridsize(func, blockDims, args, sharedMem, gridsize);
}

int nvshmemx_collective_launch(const void *func, dim3 gridDims, dim3 blockDims, void **args,
                               size_t sharedMem, cudaStream_t stream) {
    return _nvshmemi_collective_launch(func, gridDims, blockDims, args, sharedMem, stream);
}
