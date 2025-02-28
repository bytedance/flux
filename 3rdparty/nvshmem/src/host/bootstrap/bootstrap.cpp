/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include <assert.h>
#include <cstddef>
#include <stdlib.h>
#include <string>
#include <unordered_map>
#include "bootstrap_host_transport/env_defs_internal.h"
#include "device_host/nvshmem_types.h"
#include "host/nvshmemx_api.h"
#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"
#include "internal/host/debug.h"
#include "internal/host/nvshmem_internal.h"
#include "internal/host/nvshmemi_bootstrap_library.h"
#include "internal/host/nvshmemi_types.h"
#include "internal/host/util.h"
#include "non_abi/nvshmemx_error.h"

static std::unordered_map<int, std::string> bootstrap_modes = {{BOOTSTRAP_MPI, "MPI"},
                                                               {BOOTSTRAP_SHMEM, "SHMEM"},
                                                               {BOOTSTRAP_PMI, "PMI"},
                                                               {BOOTSTRAP_PLUGIN, "PLUGIN"},
                                                               {BOOTSTRAP_UID, "UID"}};

static int bootstrap_flag2mode(int flags) {
    if (flags & NVSHMEMX_INIT_WITH_MPI_COMM) {
        return BOOTSTRAP_MPI;
    } else if (flags & NVSHMEMX_INIT_WITH_SHMEM) {
        return BOOTSTRAP_SHMEM;
    } else if (flags & NVSHMEMX_INIT_WITH_UNIQUEID) {
        return BOOTSTRAP_UID;
    } else {
        if (strcmp_case_insensitive(nvshmemi_options.BOOTSTRAP, "PMI") == 0) {
            return BOOTSTRAP_PMI;
        } else if (strcmp_case_insensitive(nvshmemi_options.BOOTSTRAP, "MPI") == 0) {
            return BOOTSTRAP_MPI;
        } else if (strcmp_case_insensitive(nvshmemi_options.BOOTSTRAP, "SHMEM") == 0) {
            return BOOTSTRAP_SHMEM;
        } else if (strcmp_case_insensitive(nvshmemi_options.BOOTSTRAP, "plugin") == 0) {
            return BOOTSTRAP_PLUGIN;
        } else {
            if (!flags) {
                /* UID bootstrap only enabled via init flags. So retry with correct API and flags */
                NVSHMEMI_ERROR_PRINT(
                    "Missing init flags for bootstrap %s. Retry with nvshmemx_init_attr and "
                    "non-zero flags\n",
                    nvshmemi_options.BOOTSTRAP);
            } else {
                NVSHMEMI_ERROR_PRINT("Invalid bootstrap '%s'\n", nvshmemi_options.BOOTSTRAP);
            }
            return -1;
        }
    }

    return -1; /* Shouldn't reach here */
}

int bootstrap_set_bootattr(int flags, void *nvattr, bootstrap_attr_t *boot_attr) {
    int mode = bootstrap_flag2mode(flags);
    nvshmemx_init_attr_t *nvshmem_attr = (nvshmemx_init_attr_t *)(nvattr);
    nvshmemx_init_args_t *init_args = NULL;
    switch (mode) {
        case BOOTSTRAP_MPI:
            if (nvshmem_attr) {
                assert(boot_attr != NULL);
                (*boot_attr).mpi_comm = nvshmem_attr->mpi_comm;
            }

            break;
        case BOOTSTRAP_SHMEM:
            if (nvshmem_attr) {
                assert(boot_attr != NULL);
                (*boot_attr).initialize_shmem = 0;
            }

            break;
        case BOOTSTRAP_UID:
            if (!nvshmem_attr) {
                NVSHMEMI_ERROR_PRINT(
                    "Missing nvshmem_init_attr_t args for UID bootstrap. Please retry by "
                    "populating uid_args member\n");
                assert(0);
            }

            assert(boot_attr != NULL);
            init_args = (nvshmemx_init_args_t *)(&(nvshmem_attr->args));
            (*boot_attr).uid_args = &(init_args->uid_args);
            break;
        case BOOTSTRAP_PMI:
            /* NOOP for attribute */
            break;
        case BOOTSTRAP_PLUGIN:
            if (NULL != nvshmem_attr) {
                NVSHMEMI_ERROR_PRINT(
                    "Expected a NULL nvshmem_init_attr_t, found a non-NULL structure\n");
                assert(0);
            }

            break;
        default:
            NVSHMEMI_ERROR_PRINT("Invalid bootstrap mode selected\n");
            return (NVSHMEMX_ERROR_INTERNAL);
    }

    return (0);
}

int bootstrap_preinit(int flags, bootstrap_handle_t *handle) {
    int status = NVSHMEMX_SUCCESS;
    const char *plugin_name = NULL;
    int mode = bootstrap_flag2mode(flags);
    switch (mode) {
        case BOOTSTRAP_MPI:
        case BOOTSTRAP_SHMEM:
        case BOOTSTRAP_PMI:
        case BOOTSTRAP_PLUGIN:
            /* NOOP for other modalities */
            return (status);
        case BOOTSTRAP_UID:
            plugin_name = nvshmemi_options.BOOTSTRAP_UID_PLUGIN;
            status = bootstrap_loader_preinit(plugin_name, handle);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "bootstrap_loader_preinit returned error for mode %s\n",
                                  bootstrap_modes[mode].c_str());
            break;
        default:
            NVSHMEMI_ERROR_PRINT("Invalid bootstrap mode selected\n");
            status = 1;
    }

out:
    return status;
}

int bootstrap_init(int flags, bootstrap_attr_t *attr, bootstrap_handle_t *handle) {
    int status = NVSHMEMX_SUCCESS;
    const char *plugin_name = NULL;

    int mode = bootstrap_flag2mode(flags);
    switch (mode) {
        case BOOTSTRAP_MPI:
            plugin_name = nvshmemi_options.BOOTSTRAP_MPI_PLUGIN;

            status =
                bootstrap_loader_init(plugin_name, (attr != NULL) ? attr->mpi_comm : NULL, handle);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "bootstrap_loader_init returned error for mode %s\n",
                                  bootstrap_modes[mode].c_str());
            break;
        case BOOTSTRAP_SHMEM:
            plugin_name = nvshmemi_options.BOOTSTRAP_SHMEM_PLUGIN;

            status = bootstrap_loader_init(plugin_name,
                                           (attr != NULL) ? &attr->initialize_shmem : NULL, handle);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "bootstrap_loader_init returned error for mode %s\n",
                                  bootstrap_modes[mode].c_str());
            break;
        case BOOTSTRAP_PMI:
            if (strcmp_case_insensitive(nvshmemi_options.BOOTSTRAP_PMI, "PMIX") == 0) {
                plugin_name = nvshmemi_options.BOOTSTRAP_PMIX_PLUGIN;
                status = bootstrap_loader_init(plugin_name, NULL, handle);
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                      "bootstrap_loader_init returned error for mode %s\n",
                                      bootstrap_modes[mode].c_str());
            } else if (strcmp_case_insensitive(nvshmemi_options.BOOTSTRAP_PMI, "PMI-2") == 0 ||
                       strcmp_case_insensitive(nvshmemi_options.BOOTSTRAP_PMI, "PMI2") == 0) {
                plugin_name = nvshmemi_options.BOOTSTRAP_PMI2_PLUGIN;
                status = bootstrap_loader_init(plugin_name, NULL, handle);
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                      "bootstrap_pmi_init returned error for mode %s\n",
                                      bootstrap_modes[mode].c_str());
            } else if (strcmp_case_insensitive(nvshmemi_options.BOOTSTRAP_PMI, "PMI") == 0) {
                plugin_name = nvshmemi_options.BOOTSTRAP_PMI_PLUGIN;
                status = bootstrap_loader_init(plugin_name, NULL, handle);
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                      "bootstrap_pmi_init returned error for mode %s\n",
                                      bootstrap_modes[mode].c_str());
            } else {
                NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                   "bootstrap_pmi_init invalid PMI bootstrap '%s'\n",
                                   nvshmemi_options.BOOTSTRAP_PMI);
            }
            break;
        case BOOTSTRAP_PLUGIN:
            if (!nvshmemi_options.BOOTSTRAP_PLUGIN_provided) {
                NVSHMEMI_ERROR_PRINT(
                    "Plugin bootstrap requires NVSHMEM_BOOTSTRAP_PLUGIN to be set\n");
                status = 1;
                goto out;
            }

            plugin_name = nvshmemi_options.BOOTSTRAP_PLUGIN;

            status = bootstrap_loader_init(nvshmemi_options.BOOTSTRAP_PLUGIN, NULL, handle);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "bootstrap_loader_init returned error for mode %s\n",
                                  bootstrap_modes[mode].c_str());
            break;
        case BOOTSTRAP_UID:
            assert(attr != NULL);
            plugin_name = nvshmemi_options.BOOTSTRAP_UID_PLUGIN;

            status = bootstrap_loader_init(plugin_name, (attr->uid_args), handle);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "bootstrap_loader_init returned error for mode %s\n",
                                  bootstrap_modes[mode].c_str());
            break;

        default:
            NVSHMEMI_ERROR_PRINT("Invalid bootstrap mode selected\n");
    }

out:
    return status;
}

void bootstrap_finalize() {
    int status = NVSHMEMX_SUCCESS;

    if (nvshmemi_device_state.nvshmemi_is_nvshmem_bootstrapped) {
        status = bootstrap_loader_finalize(&nvshmemi_boot_handle);
        NVSHMEMI_NZ_EXIT(status, "bootstrap finalization returned error\n");
        // Finalize the nvshmemi_session
        if (nvshmemi_default_session) {
            free(nvshmemi_default_session);
            nvshmemi_default_session = nullptr;
        }
        NVSHMEMU_THREAD_CS_FINALIZE();
    }
}
