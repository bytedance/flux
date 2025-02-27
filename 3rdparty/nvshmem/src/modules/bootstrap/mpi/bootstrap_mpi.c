/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include <assert.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include "bootstrap_host_transport/env_defs_internal.h"
#include "bootstrap_util.h"
#include "internal/bootstrap_host/nvshmemi_bootstrap.h"
#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"
#include "non_abi/nvshmemx_error.h"

static MPI_Comm bootstrap_comm = MPI_COMM_NULL;
static int nvshmem_initialized_mpi = 0;
int bootstrap_debug_enable = 0;

struct nvshmemi_options_s env_attr;

/* Define common types */
#define BOOTPRI_string "\"%s\""

#define BOOTSTRAP_OPTIONS_PRINT_ENV(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC, DESIRED_CAT, STYLE) \
    if (CATEGORY == DESIRED_CAT) {                                                                 \
        switch (STYLE) {                                                                           \
            char *desc_wrapped;                                                                    \
            case BOOTSTRAP_OPTIONS_STYLE_INFO:                                                     \
                desc_wrapped = bootstrap_util_wrap_string(SHORT_DESC, 80, "\t", 1);                \
                printf("  NVSHMEM_%-20s " BOOTPRI_##KIND " (type: %s, default: " BOOTPRI_##KIND    \
                       ")\n\t%s\n",                                                                \
                       #NAME, NVSHFMT_##KIND(env_attr.NAME), #KIND, NVSHFMT_##KIND(DEFAULT),       \
                       desc_wrapped);                                                              \
                free(desc_wrapped);                                                                \
                break;                                                                             \
            case BOOTSTRAP_OPTIONS_STYLE_RST:                                                      \
                desc_wrapped = bootstrap_util_wrap_string(SHORT_DESC, 80, NULL, 0);                \
                printf(".. c:var:: NVSHMEM_%s\n", #NAME);                                          \
                printf("\n");                                                                      \
                printf("| *Type: %s*\n", #KIND);                                                   \
                printf("| *Default: " BOOTPRI_##KIND "*\n", NVSHFMT_##KIND(DEFAULT));              \
                printf("\n");                                                                      \
                printf("%s\n", desc_wrapped);                                                      \
                printf("\n");                                                                      \
                free(desc_wrapped);                                                                \
                break;                                                                             \
            default:                                                                               \
                assert(0);                                                                         \
        }                                                                                          \
    }

static int bootstrap_mpi_showinfo(struct bootstrap_handle *handle, int style) {
    bootstrap_util_print_header(style, "Bootstrap Options");

#define NVSHMEMI_ENV_DEF(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC)        \
    BOOTSTRAP_OPTIONS_PRINT_ENV(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC, \
                                NVSHMEMI_ENV_CAT_BOOTSTRAP, style)
#include "env_defs.h"
#undef NVSHMEMI_ENV_DEF
    printf("\n");
    return 0;
}

static int bootstrap_mpi_barrier(struct bootstrap_handle *handle) {
    int status = MPI_SUCCESS;

    status = MPI_Barrier(bootstrap_comm);
    BOOTSTRAP_NE_ERROR_JMP(status, MPI_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                           "MPI_Barrier failed\n");

out:
    return status;
}

static int bootstrap_mpi_allgather(const void *sendbuf, void *recvbuf, int length,
                                   struct bootstrap_handle *handle) {
    int status = MPI_SUCCESS;

    status = MPI_Allgather(sendbuf, length, MPI_BYTE, recvbuf, length, MPI_BYTE, bootstrap_comm);
    BOOTSTRAP_NE_ERROR_JMP(status, MPI_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                           "MPI_Allgather failed\n");

out:
    return status;
}

static int bootstrap_mpi_alltoall(const void *sendbuf, void *recvbuf, int length,
                                  struct bootstrap_handle *handle) {
    int status = MPI_SUCCESS;

    status = MPI_Alltoall(sendbuf, length, MPI_BYTE, recvbuf, length, MPI_BYTE, bootstrap_comm);
    BOOTSTRAP_NE_ERROR_JMP(status, MPI_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                           "MPI_Alltoall failed\n");

out:
    return status;
}

static void bootstrap_mpi_global_exit(int status) {
    int rc = MPI_SUCCESS;

    rc = MPI_Abort(bootstrap_comm, status);
    if (rc != MPI_SUCCESS) {
        BOOTSTRAP_ERROR_PRINT("MPI_Abort failed. Manually exiting this process.\n");
        exit(1);
    }
}

static int bootstrap_mpi_finalize(bootstrap_handle_t *handle) {
    int status = MPI_SUCCESS, finalized;

    /* Ensure user hasn't finalized MPI before finalizing NVSHMEM */
    status = MPI_Finalized(&finalized);
    BOOTSTRAP_NE_ERROR_JMP(status, MPI_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                           "MPI_Finalized failed\n");

    if (finalized) {
        if (nvshmem_initialized_mpi) {
            status = NVSHMEMX_ERROR_INTERNAL;
            BOOTSTRAP_ERROR_PRINT("MPI is finalized\n");
        } else {
            status = 0;
        }

        goto out;
    }

    if (!finalized && nvshmem_initialized_mpi) {
        status = MPI_Comm_free(&bootstrap_comm);
        BOOTSTRAP_NE_ERROR_JMP(status, MPI_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                               "Freeing bootstrap communicator failed\n");
    }

    if (nvshmem_initialized_mpi) MPI_Finalize();

out:
    return status;
}

int nvshmemi_bootstrap_plugin_init(void *mpi_comm, bootstrap_handle_t *handle,
                                   const int abi_version) {
    int status = MPI_SUCCESS, initialized = 0, finalized = 0;
    MPI_Comm src_comm;
    int bootstrap_version = NVSHMEMI_BOOTSTRAP_ABI_VERSION;
    if (!nvshmemi_is_bootstrap_compatible(bootstrap_version, abi_version, true)) {
        BOOTSTRAP_ERROR_PRINT(
            "MPI bootstrap version (%d) is not compatible with NVSHMEM version (%d)",
            bootstrap_version, abi_version);
        exit(-1);
    }

    if (NULL == mpi_comm)
        src_comm = MPI_COMM_WORLD;
    else
        src_comm = *((MPI_Comm *)mpi_comm);

    status = MPI_Initialized(&initialized);
    BOOTSTRAP_NE_ERROR_JMP(status, MPI_SUCCESS, NVSHMEMX_ERROR_INTERNAL, error,
                           "MPI_Initialized failed\n");

    status = MPI_Finalized(&finalized);
    BOOTSTRAP_NE_ERROR_JMP(status, MPI_SUCCESS, NVSHMEMX_ERROR_INTERNAL, error,
                           "MPI_Finalized failed\n");

    if (!initialized && !finalized) {
        MPI_Init(NULL, NULL);
        nvshmem_initialized_mpi = 1;

        // Because MPI was not initialized, the only communicators that would
        // have been valid to pass are the predefined communicators
        if (src_comm != MPI_COMM_WORLD && src_comm != MPI_COMM_SELF) {
            status = NVSHMEMX_ERROR_INTERNAL;
            BOOTSTRAP_ERROR_PRINT("Invalid communicator\n");
            goto error;
        }
    } else if (finalized) {
        status = NVSHMEMX_ERROR_INTERNAL;
        BOOTSTRAP_ERROR_PRINT("MPI is finalized\n");
        goto error;
    }

    status = MPI_Comm_dup(src_comm, &bootstrap_comm);
    BOOTSTRAP_NE_ERROR_JMP(status, MPI_SUCCESS, NVSHMEMX_ERROR_INTERNAL, error,
                           "Creating bootstrap communicator failed\n");

    status = MPI_Comm_rank(bootstrap_comm, &handle->pg_rank);
    BOOTSTRAP_NE_ERROR_JMP(status, MPI_SUCCESS, NVSHMEMX_ERROR_INTERNAL, error,
                           "MPI_Comm_rank failed\n");

    status = MPI_Comm_size(bootstrap_comm, &handle->pg_size);
    BOOTSTRAP_NE_ERROR_JMP(status, MPI_SUCCESS, NVSHMEMX_ERROR_INTERNAL, error,
                           "MPI_Comm_size failed\n");

    handle->version = NVSHMEM_BOOTSTRAP_MAJOR_MINOR_VERSION(abi_version) <
                              NVSHMEM_BOOTSTRAP_MAJOR_MINOR_VERSION(bootstrap_version)
                          ? abi_version
                          : bootstrap_version;
    handle->allgather = bootstrap_mpi_allgather;
    handle->alltoall = bootstrap_mpi_alltoall;
    handle->barrier = bootstrap_mpi_barrier;
    handle->global_exit = bootstrap_mpi_global_exit;
    handle->finalize = bootstrap_mpi_finalize;
    handle->pre_init_ops = NULL;
    handle->comm_state = &bootstrap_comm;

    goto out;

error:
    if (nvshmem_initialized_mpi) {
        MPI_Finalize();
        nvshmem_initialized_mpi = 0;
    }

out:
    return status;
}
