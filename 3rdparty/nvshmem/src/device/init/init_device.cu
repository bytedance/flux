/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include <stdio.h>
#include <algorithm>
#include <cuda_runtime.h>

#include "non_abi/nvshmem_build_options.h"
#include "non_abi/nvshmem_version.h"
#include "non_abi/nvshmemx_error.h"
#include "internal/device/nvshmemi_device.h"
#include "non_abi/device/pt-to-pt/proxy_device.cuh"
#include "device_host/nvshmem_common.cuh"
#include "device_host/nvshmem_types.h"

#ifdef NVSHMEM_IBGDA_SUPPORT
#include "device_host_transport/nvshmem_common_ibgda.h"

__constant__ nvshmemi_ibgda_device_state_t nvshmemi_ibgda_device_state_d;
#endif

nvshmemi_device_state_t nvshmemi_device_only_state;
__constant__ nvshmemi_device_host_state_t nvshmemi_device_state_d;
const nvshmemi_version_t nvshmemi_device_lib_version = {
    NVSHMEM_INTERLIB_MAJOR_VERSION, NVSHMEM_INTERLIB_MINOR_VERSION, NVSHMEM_INTERLIB_PATCH_VERSION};
__constant__ nvshmemi_version_t nvshmemi_device_lib_version_d = {
    NVSHMEM_INTERLIB_MAJOR_VERSION, NVSHMEM_INTERLIB_MINOR_VERSION, NVSHMEM_INTERLIB_PATCH_VERSION};

#ifdef __CUDA_ARCH__
#ifdef __cplusplus
extern "C" {
#endif
__device__ void nvshmem_global_exit(int status);
#ifdef __cplusplus
}
#endif

__device__ void nvshmem_global_exit(int status) {
    if (nvshmemi_device_state_d.proxy > NVSHMEMI_PROXY_NONE) {
        nvshmemi_proxy_global_exit(status);
    } else {
        /* TODO: Add device side printing macros */
        printf(
            "Device side proxy was called, but is not supported under your configuration. "
            "Please unset NVSHMEM_DISABLE_LOCAL_ONLY_PROXY, or set it to false.\n");
        assert(0);
    }
}
#endif

#ifdef __cplusplus
extern "C" {
void nvshmemi_get_mem_handle(void **dev_state_ptr, void **transport_dev_state_ptr);
}
#endif

static int _nvshmemi_init_device_only_state() {
    int status = 0;
    status = nvshmemi_setup_collective_launch();
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "_nvshmemi_init_device_only_state failed\n");
    nvshmemi_device_only_state.is_initialized = true;

out:
    return status;
}

void nvshmemi_check_state_and_init_d() {
    int status;
    int ret;

    if (nvshmemid_init_status() == NVSHMEM_STATUS_NOT_INITIALIZED)
        NVSHMEMI_ERROR_EXIT("nvshmem API called before nvshmem_init \n");
    if (nvshmemid_init_status() == NVSHMEM_STATUS_IS_BOOTSTRAPPED) {
        /* The fact that we can pass NVSHMEM_THREAD_SERIALIZED
         * here is an implementation detail. It should be fixed
         * if/when NVSHMEM_THREAD_* becomes significant. */
        status = nvshmemid_hostlib_init_attr(NVSHMEM_THREAD_SERIALIZED, &ret, 0, NULL,
                                             nvshmemi_device_lib_version, NULL);
        if (status) {
            NVSHMEMI_ERROR_EXIT("nvshmem initialization failed, exiting \n");
        }

        status = cudaGetDevice(&nvshmemi_device_only_state.cuda_device_id);
        if (status) {
            NVSHMEMI_ERROR_EXIT("nvshmem cuda device query failed, exiting \n");
        }

        nvshmemid_hostlib_finalize(NULL, NULL);
    }

    if (!nvshmemi_device_only_state.is_initialized) {
        status = _nvshmemi_init_device_only_state();
        if (status) {
            NVSHMEMI_ERROR_EXIT("nvshmem device initialization failed, exiting \n");
        }
    }
}

void nvshmemi_get_mem_handle(void **dev_state_ptr, void **transport_dev_state_ptr) {
    int status = 0;
    status = cudaGetSymbolAddress(dev_state_ptr, nvshmemi_device_state_d);
    if (status) {
        NVSHMEMI_ERROR_PRINT("Unable to access device state. %d\n", status);
        *dev_state_ptr = NULL;
    }
#ifdef NVSHMEM_IBGDA_SUPPORT
    status = cudaGetSymbolAddress(transport_dev_state_ptr, nvshmemi_ibgda_device_state_d);
    if (status) {
        NVSHMEMI_ERROR_PRINT("Unable to access ibgda device state. %d\n", status);
        *transport_dev_state_ptr = NULL;
    }
#endif
}

int nvshmemi_init_thread(int requested_thread_support, int *provided_thread_support,
                         unsigned int bootstrap_flags, nvshmemx_init_attr_t *bootstrap_attr,
                         nvshmemi_version_t nvshmem_app_version) {
    int status = 0;

#ifdef _NVSHMEM_DEBUG
    printf("  %-28s %d\n", "DEVICE CUDA API", CUDART_VERSION);
#endif
    status = nvshmemid_hostlib_init_attr(requested_thread_support, provided_thread_support,
                                         bootstrap_flags, bootstrap_attr,
                                         nvshmemi_device_lib_version, &nvshmemi_get_mem_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "nvshmem_internal_init_thread failed \n");

    if (nvshmemid_init_status() > NVSHMEM_STATUS_IS_BOOTSTRAPPED) {
        status = _nvshmemi_init_device_only_state();
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "nvshmem_internal_init_thread failed at init_device_only_state.\n");

        status = cudaGetDevice(&nvshmemi_device_only_state.cuda_device_id);
        if (status) {
            NVSHMEMI_ERROR_EXIT("nvshmem cuda device query failed, exiting \n");
        }
    }

out:
    return status;
}

#ifdef __cplusplus
extern "C" {
#endif
void nvshmemi_finalize() {
    int status;
    void *dev_state_ptr, *transport_dev_state_ptr = NULL;

    status = cudaGetSymbolAddress(&dev_state_ptr, nvshmemi_device_state_d);
    if (status) {
        NVSHMEMI_ERROR_PRINT("Unable to properly unregister device state.\n");
        nvshmemid_hostlib_finalize(NULL, NULL);
        return;
    }
#ifdef NVSHMEM_IBGDA_SUPPORT
    status = cudaGetSymbolAddress(&transport_dev_state_ptr, nvshmemi_ibgda_device_state_d);
    if (status) {
        NVSHMEMI_ERROR_PRINT("Unable to properly unregister device state.\n");
        nvshmemid_hostlib_finalize(NULL, NULL);
        return;
    }
#endif
    nvshmemid_hostlib_finalize(dev_state_ptr, transport_dev_state_ptr);
}
#ifdef __cplusplus
}
#endif
