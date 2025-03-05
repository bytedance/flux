/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include <assert.h>
#include <shmem.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bootstrap_host_transport/env_defs_internal.h"
#include "bootstrap_util.h"
#include "env_defs.h"
#include "internal/bootstrap_host/nvshmemi_bootstrap.h"
#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"
#include "non_abi/nvshmemx_error.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))

static size_t scratch_size;
static long *scratch;
static int nvshmem_initialized_shmem = 0;
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

static int bootstrap_shmem_showinfo(struct bootstrap_handle *handle, int style) {
    bootstrap_util_print_header(style, "Bootstrap Options");

#define NVSHMEMI_ENV_DEF(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC)        \
    BOOTSTRAP_OPTIONS_PRINT_ENV(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC, \
                                NVSHMEMI_ENV_CAT_BOOTSTRAP, style)
#include "env_defs.h"
#undef NVSHMEMI_ENV_DEF
    printf("\n");
    return 0;
}

void bootstrap_shmem_global_exit(int status) { shmem_global_exit(status); }

static int bootstrap_shmem_barrier(struct bootstrap_handle *handle) {
    int status = 0;

    shmem_barrier_all();
out:
    return status;
}

static int bootstrap_shmem_allgather(const void *sendbuf, void *recvbuf, int length,
                                     struct bootstrap_handle *handle) {
    int status = 0;
    void *sendbuf_i = NULL, *recvbuf_i = NULL;

    sendbuf_i = shmem_malloc(length);
    BOOTSTRAP_NULL_ERROR_JMP(sendbuf_i, status, NVSHMEMX_ERROR_INTERNAL, out,
                             "shmem_malloc failed\n");
    recvbuf_i = shmem_malloc(length * handle->pg_size);
    BOOTSTRAP_NULL_ERROR_JMP(recvbuf_i, status, NVSHMEMX_ERROR_INTERNAL, out,
                             "shmem_malloc failed\n");
    shmem_barrier_all();

    memcpy(sendbuf_i, sendbuf, length);

    shmem_barrier_all();
    assert(scratch_size >= SHMEM_COLLECT_SYNC_SIZE * sizeof(long));
    shmem_collect32(recvbuf_i, sendbuf_i, length / 4, 0, 0, handle->pg_size, scratch);
    shmem_barrier_all();

    memcpy(recvbuf, recvbuf_i, length * handle->pg_size);

    shmem_barrier_all();
    shmem_free(sendbuf_i);
    shmem_free(recvbuf_i);
    shmem_barrier_all();
out:
    return status;
}

static int bootstrap_shmem_alltoall(const void *sendbuf, void *recvbuf, int length,
                                    struct bootstrap_handle *handle) {
    int status = 0;
    void *sendbuf_i = NULL, *recvbuf_i = NULL;

    sendbuf_i = shmem_malloc(length * handle->pg_size);
    BOOTSTRAP_NULL_ERROR_JMP(sendbuf_i, status, NVSHMEMX_ERROR_INTERNAL, out,
                             "shmem_malloc failed\n");
    recvbuf_i = shmem_malloc(length * handle->pg_size);
    BOOTSTRAP_NULL_ERROR_JMP(recvbuf_i, status, NVSHMEMX_ERROR_INTERNAL, out,
                             "shmem_malloc failed\n");
    shmem_barrier_all();

    memcpy(sendbuf_i, sendbuf, length * handle->pg_size);

    shmem_barrier_all();
    assert(scratch_size >= SHMEM_ALLTOALL_SYNC_SIZE * sizeof(long));
    shmem_alltoall32(recvbuf_i, sendbuf_i, length / 4, 0, 0, handle->pg_size, scratch);
    shmem_barrier_all();

    memcpy(recvbuf, recvbuf_i, length * handle->pg_size);

    shmem_barrier_all();
    shmem_free(sendbuf_i);
    shmem_free(recvbuf_i);
    shmem_barrier_all();
out:
    return status;
}

static int bootstrap_shmem_finalize(bootstrap_handle_t *handle) {
    int status = 0;

    if (nvshmem_initialized_shmem) {
        shmem_free(scratch);
        shmem_finalize();
    } else {
        // FIXME: OpenSHMEM currently doesn't provide a way to check if the
        // library has been finalized. It's proposed for OpenSHMEM 1.6. Once
        // this becomes available, the buffer below should be freed.

        // if (!finalized)
        //     shmem_free(scratch);
    }

out:
    return status;
}

int nvshmemi_bootstrap_plugin_init(void *arg, bootstrap_handle_t *handle,
                                   const int nvshmem_version) {
    int status = 0;
    int bootstrap_version = NVSHMEMI_BOOTSTRAP_ABI_VERSION;
    if (!nvshmemi_is_bootstrap_compatible(bootstrap_version, nvshmem_version, true)) {
        BOOTSTRAP_ERROR_PRINT(
            "SHMEM bootstrap version (%d) is not compatible with NVSHMEM version (%d)",
            bootstrap_version, nvshmem_version);
        exit(-1);
    }

    nvshmemi_env_options_init(&env_attr);
    if (arg == NULL || *(int *)arg) {
        shmem_init();
        nvshmem_initialized_shmem = 1;
    }

    handle->pg_rank = shmem_my_pe();
    handle->pg_size = shmem_n_pes();

    scratch_size = MAX(SHMEM_COLLECT_SYNC_SIZE, SHMEM_ALLTOALL_SYNC_SIZE) * sizeof(long);
    scratch = shmem_malloc(scratch_size);
    BOOTSTRAP_NULL_ERROR_JMP(scratch, status, NVSHMEMX_ERROR_INTERNAL, out,
                             "shmem_malloc failed\n");

    for (int i = 0; i < scratch_size / sizeof(long); ++i) {
        scratch[i] = SHMEM_SYNC_VALUE;
    }

    handle->version = NVSHMEM_BOOTSTRAP_MAJOR_MINOR_VERSION(nvshmem_version) <
                              NVSHMEM_BOOTSTRAP_MAJOR_MINOR_VERSION(bootstrap_version)
                          ? nvshmem_version
                          : bootstrap_version;
    handle->allgather = bootstrap_shmem_allgather;
    handle->alltoall = bootstrap_shmem_alltoall;
    handle->barrier = bootstrap_shmem_barrier;
    handle->global_exit = bootstrap_shmem_global_exit;
    handle->finalize = bootstrap_shmem_finalize;
    handle->show_info = bootstrap_shmem_showinfo;
    handle->comm_state = NULL;
    handle->pre_init_ops = NULL;

out:
    return status;
}
