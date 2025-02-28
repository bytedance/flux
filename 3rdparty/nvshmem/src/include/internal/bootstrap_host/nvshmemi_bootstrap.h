/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */
#ifndef NVSHMEMI_BOOTSTRAP_H
#define NVSHMEMI_BOOTSTRAP_H

#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"
#include "non_abi/nvshmem_version.h"
/* Version = major * 10000 + minor * 100 + patch*/
/* ABI Introduced in NVSHMEM 2.8.0 */
#define NVSHMEMI_BOOTSTRAP_ABI_VERSION                \
    (NVSHMEM_BOOTSTRAP_PLUGIN_MAJOR_VERSION * 10000 + \
     NVSHMEM_BOOTSTRAP_PLUGIN_MINOR_VERSION * 100 + NVSHMEM_BOOTSTRAP_PLUGIN_PATCH_VERSION)

#define NVSHMEM_BOOTSTRAP_MAJOR_VERSION(ver) (ver / 10000)
#define NVSHMEM_BOOTSTRAP_MAJOR_MINOR_VERSION(ver) (ver / 100)

static bool nvshmemi_is_bootstrap_compatible(int bootstrap_version, int nvshmem_version,
                                             bool boot_backward_compatible) {
    if (NVSHMEM_BOOTSTRAP_MAJOR_VERSION(bootstrap_version) !=
        NVSHMEM_BOOTSTRAP_MAJOR_VERSION(nvshmem_version)) {
        return false;
    }

    if (NVSHMEM_BOOTSTRAP_MAJOR_MINOR_VERSION(nvshmem_version) <
        NVSHMEM_BOOTSTRAP_MAJOR_MINOR_VERSION(bootstrap_version)) {
        if (boot_backward_compatible) {
            return true;
        }
        return false;
    }
    return true;
}

#if __cplusplus
extern "C" {
#endif
int nvshmemi_bootstrap_plugin_init(void *mpi_comm, bootstrap_handle_t *handle,
                                   const int nvshmem_version);
int nvshmemi_bootstrap_plugin_pre_init(bootstrap_handle_t *handle, const int nvshmem_version);
#if __cplusplus
}
#endif

#endif
