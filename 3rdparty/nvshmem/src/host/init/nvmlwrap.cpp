#include "internal/host/nvmlwrap.h"
#include <dlfcn.h>                                       // for dlsym, dlclose, dlopen
#include <stdio.h>                                       // for NULL, snprintf
#include <string.h>                                      // for memset
#include "non_abi/nvshmemx_error.h"                      // for NVSHMEMI_ERROR_PRINT
#include "internal/host/debug.h"                         // for INFO, NVSHMEM_INIT
#include "internal/host/util.h"                          // for nvshmemi_options
#include "bootstrap_host_transport/env_defs_internal.h"  // for nvshmemi_options_s

#define LOAD_SYM(handle, symbol, funcptr, optional, ret)        \
    do {                                                        \
        void **cast = (void **)&funcptr;                        \
        void *tmp = dlsym(handle, symbol);                      \
        *cast = tmp;                                            \
        if (*cast == NULL && !optional) {                       \
            NVSHMEMI_ERROR_PRINT("Retrieve %s failed", symbol); \
            ret = NVSHMEMX_ERROR_INTERNAL;                      \
        }                                                       \
    } while (0)

int nvshmemi_nvml_ftable_init(struct nvml_function_table *nvml_ftable, void **nvml_handle) {
    int status = NVSHMEMX_SUCCESS;
    char path[1024];

    if (!nvshmemi_options.CUDA_PATH_provided)
        snprintf(path, 1024, "%s", "libnvidia-ml.so.1");
    else
        snprintf(path, 1024, "%s/%s", nvshmemi_options.CUDA_PATH, "libnvidia-ml.so.1");

    *nvml_handle = dlopen(path, RTLD_NOW);
    if (!(*nvml_handle)) {
        INFO(NVSHMEM_INIT, "NVML library not found. %s", path);
        status = NVSHMEMX_ERROR_NOT_SUPPORTED;
    } else {
        INFO(NVSHMEM_INIT, "NVML library found. %s", path);
        LOAD_SYM(*nvml_handle, "nvmlInit", nvml_ftable->nvmlInit, 0, status);
        LOAD_SYM(*nvml_handle, "nvmlShutdown", nvml_ftable->nvmlShutdown, 0, status);
        LOAD_SYM(*nvml_handle, "nvmlDeviceGetHandleByPciBusId",
                 nvml_ftable->nvmlDeviceGetHandleByPciBusId, 0, status);
        LOAD_SYM(*nvml_handle, "nvmlDeviceGetP2PStatus", nvml_ftable->nvmlDeviceGetP2PStatus, 0,
                 status);
        LOAD_SYM(*nvml_handle, "nvmlDeviceGetGpuFabricInfoV",
                 nvml_ftable->nvmlDeviceGetGpuFabricInfoV, 1, status);
    }

    if (status != NVSHMEMX_SUCCESS) {
        nvshmemi_nvml_ftable_fini(nvml_ftable, nvml_handle);
    }
    return status;
}

void nvshmemi_nvml_ftable_fini(struct nvml_function_table *nvml_ftable, void **nvml_handle) {
    if (*nvml_handle) {
        dlclose(*nvml_handle);
        *nvml_handle = NULL;
        memset(nvml_ftable, 0, sizeof(*nvml_ftable));
    }
}