/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <pmix.h>
#include <stdint.h>  // IWYU pragma: keep
// IWYU pragma: no_include <bits/stdint-uintn.h>
#include <stdbool.h>

#include "bootstrap_util.h"
#include "bootstrap_host_transport/env_defs_internal.h"  // IWYU pragma: keep
#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"
#include "internal/bootstrap_host/nvshmemi_bootstrap.h"
#include "non_abi/nvshmemx_error.h"
#include "pmix_common.h"

#define BOOTSTRAP_PMIX_KEYSIZE 64

static pmix_proc_t myproc;
int bootstrap_debug_enable = 0;
static struct nvshmemi_options_s env_attr;

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

static int bootstrap_pmix_showinfo(struct bootstrap_handle *handle, int style) {
    bootstrap_util_print_header(style, "Bootstrap Options");

#define NVSHMEMI_ENV_DEF(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC)        \
    BOOTSTRAP_OPTIONS_PRINT_ENV(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC, \
                                NVSHMEMI_ENV_CAT_BOOTSTRAP, style)
#include "env_defs.h"
#undef NVSHMEMI_ENV_DEF
    printf("\n");
    return 0;
}

static int bootstrap_pmix_barrier(bootstrap_handle_t *handle) {
    pmix_status_t status = PMIx_Fence(NULL, 0, NULL, 0);
    BOOTSTRAP_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "PMIx_Fence failed\n");

out:
    return status;
}

static pmix_status_t bootstrap_pmix_exchange(void) {
    pmix_status_t status;
    pmix_info_t info;
    bool flag = true;

    status = PMIx_Commit();
    BOOTSTRAP_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "PMIx_Commit failed\n");

    PMIX_INFO_CONSTRUCT(&info);
    PMIX_INFO_LOAD(&info, PMIX_COLLECT_DATA, &flag, PMIX_BOOL);

    status = PMIx_Fence(NULL, 0, &info, 1);
    BOOTSTRAP_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "PMIx_Fence failed\n");

    PMIX_INFO_DESTRUCT(&info);

error:
out:
    return status;
}

static pmix_status_t bootstrap_pmix_put(char *key, void *value, size_t valuelen) {
    pmix_value_t val;
    pmix_status_t rc;

    PMIX_VALUE_CONSTRUCT(&val);
    val.type = PMIX_BYTE_OBJECT;
    val.data.bo.bytes = (char *)value;
    val.data.bo.size = valuelen;

    rc = PMIx_Put(PMIX_GLOBAL, key, &val);
    val.data.bo.bytes = NULL;  // protect the data
    val.data.bo.size = 0;
    PMIX_VALUE_DESTRUCT(&val);

    return rc;
}

static pmix_status_t bootstrap_pmix_get(int pe, char *key, void *value, size_t valuelen) {
    pmix_proc_t proc;
    pmix_value_t *val;
    pmix_status_t rc;

    /* ensure the region is zero'd out */
    memset(value, 0, valuelen);

    /* setup the ID of the proc whose info we are getting */
    PMIX_LOAD_NSPACE(proc.nspace, myproc.nspace);

    proc.rank = (uint32_t)pe;

    rc = PMIx_Get(&proc, key, NULL, 0, &val);

    if (PMIX_SUCCESS == rc) {
        if (NULL != val) {
            /* see if the data fits into the given region */
            if (valuelen < val->data.bo.size) {
                PMIX_VALUE_RELEASE(val);
                return PMIX_ERROR;
            }
            /* copy the results across */
            memcpy(value, val->data.bo.bytes, val->data.bo.size);
            PMIX_VALUE_RELEASE(val);
        }
    }

    return rc;
}

static int bootstrap_pmix_allgather(const void *sendbuf, void *recvbuf, int length,
                                    bootstrap_handle_t *handle) {
    static int key_index = 1;

    pmix_status_t status = PMIX_SUCCESS;
    void *kvs_value;
    char kvs_key[BOOTSTRAP_PMIX_KEYSIZE];  // FIXME: assert( 64 < PMIX_MAX_KEYLEN);

    if (handle->pg_size == 1) {
        memcpy(recvbuf, sendbuf, length);
        return 0;
    }

    snprintf(kvs_key, BOOTSTRAP_PMIX_KEYSIZE, "BOOTSTRAP-%04x", key_index);

    status = bootstrap_pmix_put(kvs_key, (void *)sendbuf, length);
    BOOTSTRAP_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "bootstrap_pmix_put failed\n");

    status = bootstrap_pmix_exchange();
    BOOTSTRAP_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                           "bootstrap_pmix_exchange failed\n");

    for (int i = 0; i < handle->pg_size; i++) {
        snprintf(kvs_key, BOOTSTRAP_PMIX_KEYSIZE, "BOOTSTRAP-%04x", key_index);

        // assumes that same length is passed by all the processes
        status = bootstrap_pmix_get(i, kvs_key, (char *)recvbuf + length * i, length);
        BOOTSTRAP_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "SPMI_KVS_Get failed\n");
    }

out:
    key_index++;
    return status;
}

static int bootstrap_pmix_alltoall(const void *sendbuf, void *recvbuf, int length,
                                   bootstrap_handle_t *handle) {
    static int key_index = 1;

    pmix_status_t status = 0;
    void *kvs_value;
    char kvs_key[BOOTSTRAP_PMIX_KEYSIZE];

    if (handle->pg_size == 1) {
        memcpy(recvbuf, sendbuf, length);
        return 0;
    }

    for (int i = 0; i < handle->pg_size; i++) {
        snprintf(kvs_key, BOOTSTRAP_PMIX_KEYSIZE, "BOOTSTRAP-%04x-%08x", key_index, i);

        status = bootstrap_pmix_put(kvs_key, (char *)sendbuf + i * length, length);
        BOOTSTRAP_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "bootstrap_pmix_put failed\n");
    }

    status = bootstrap_pmix_exchange();
    BOOTSTRAP_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                           "bootstrap_pmix_exchange failed\n");

    for (int i = 0; i < handle->pg_size; i++) {
        snprintf(kvs_key, BOOTSTRAP_PMIX_KEYSIZE, "BOOTSTRAP-%04x-%08x", key_index,
                 handle->pg_rank);

        // assumes that same length is passed by all the processes
        status = bootstrap_pmix_get(i, kvs_key, (char *)recvbuf + length * i, length);
        BOOTSTRAP_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "bootstrap_pmix_get failed\n");
    }

out:
    key_index++;
    return status;
}

static void bootstrap_pmix_global_exit(int status) {
    pmix_status_t rc = PMIX_SUCCESS;

    rc = PMIx_Abort(status, "NVSHMEM Global Exit.\n", NULL, 0);
    if (rc != PMIX_SUCCESS) {
        BOOTSTRAP_ERROR_PRINT("PMIx_Abort failed. Manually exiting this process.\n");
        exit(1);
    }
}

static int bootstrap_pmix_finalize(bootstrap_handle_t *handle) {
    pmix_status_t status;

    status = PMIx_Finalize(NULL, 0);
    BOOTSTRAP_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, error, "PMIx_Finalize failed\n");

error:
out:
    return status;
}

int nvshmemi_bootstrap_plugin_init(void *attr, bootstrap_handle_t *handle, const int abi_version) {
    pmix_status_t status = PMIX_SUCCESS;
    pmix_proc_t proc;
    proc.rank = PMIX_RANK_WILDCARD;
    pmix_value_t *val;
    int bootstrap_version = NVSHMEMI_BOOTSTRAP_ABI_VERSION;
    if (!nvshmemi_is_bootstrap_compatible(bootstrap_version, abi_version, true)) {
        BOOTSTRAP_ERROR_PRINT(
            "PMIx bootstrap version (%d) is not compatible with NVSHMEM version (%d)",
            bootstrap_version, abi_version);
        exit(-1);
    }

    PMIX_PROC_CONSTRUCT(&myproc);

    status = PMIx_Init(&myproc, NULL, 0);
    BOOTSTRAP_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "PMIx_Init failed\n");

    PMIX_LOAD_NSPACE(proc.nspace, myproc.nspace);
    proc.rank = PMIX_RANK_WILDCARD;

    status = PMIx_Get(&proc, PMIX_JOB_SIZE, NULL, 0, &val);
    BOOTSTRAP_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                           "PMIx_Get(PMIX_JOB_SIZE) failed\n");

    handle->version = NVSHMEM_BOOTSTRAP_MAJOR_MINOR_VERSION(abi_version) <
                              NVSHMEM_BOOTSTRAP_MAJOR_MINOR_VERSION(bootstrap_version)
                          ? abi_version
                          : bootstrap_version;
    handle->pg_rank = myproc.rank;
    handle->pg_size = val->data.uint32;
    handle->allgather = bootstrap_pmix_allgather;
    handle->alltoall = bootstrap_pmix_alltoall;
    handle->barrier = bootstrap_pmix_barrier;
    handle->global_exit = bootstrap_pmix_global_exit;
    handle->finalize = bootstrap_pmix_finalize;
    handle->comm_state = (void *)(&myproc);
    handle->pre_init_ops = NULL;
    handle->show_info = bootstrap_pmix_showinfo;

    PMIX_VALUE_RELEASE(val);

out:
    return status != PMIX_SUCCESS;
}
