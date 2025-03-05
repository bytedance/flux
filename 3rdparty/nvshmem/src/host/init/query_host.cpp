/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include <stddef.h>                                                        // for size_t
#include "device_host/nvshmem_types.h"                                     // for nvshmem_team_t
#include "device_host_transport/nvshmem_constants.h"                       // for NVSHMEM_MAJOR...
#include "host/nvshmem_api.h"                                              // for nvshmem_team_...
#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"  // for bootstrap_han...
#include "non_abi/nvshmem_version.h"                                       // for NVSHMEM_VENDO...
#include "internal/host/nvshmemi_types.h"

int nvshmem_my_pe(void) { return nvshmemi_boot_handle.pg_rank; }

int nvshmem_n_pes(void) { return nvshmemi_boot_handle.pg_size; }

void nvshmem_info_get_name(char *name) {
    size_t i;
    const char *str = NVSHMEM_VENDOR_STRING;

    /* Copy up to NVSHMEM_MAX_NAME_LEN-1 chars, then add NULL terminator */
    for (i = 0; i < NVSHMEM_MAX_NAME_LEN - 1 && str[i] != '\0'; i++) name[i] = str[i];

    name[i] = '\0';
}

void nvshmem_info_get_version(int *major, int *minor) {
    *major = NVSHMEM_MAJOR_VERSION;
    *minor = NVSHMEM_MINOR_VERSION;
}

void nvshmemx_vendor_get_version_info(int *major, int *minor, int *patch) {
    *major = NVSHMEM_VENDOR_MAJOR_VERSION;
    *minor = NVSHMEM_VENDOR_MINOR_VERSION;
    *patch = NVSHMEM_VENDOR_PATCH_VERSION;
}

int nvshmemx_my_pe(nvshmemx_team_t team) { return nvshmem_team_my_pe((nvshmem_team_t)team); }

int nvshmemx_n_pes(nvshmemx_team_t team) { return nvshmem_team_n_pes((nvshmem_team_t)team); }
