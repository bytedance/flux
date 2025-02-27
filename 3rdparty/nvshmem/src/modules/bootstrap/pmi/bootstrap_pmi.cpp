/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include <stdint.h>  // IWYU pragma: keep
// IWYU pragma: no_include <bits/stdint-uintn.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "bootstrap_host_transport/env_defs_internal.h"
#include "non_abi/nvshmemx_error.h"
#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"
#include "internal/bootstrap_host/nvshmemi_bootstrap.h"
#include "bootstrap_util.h"
#include <assert.h>

#ifdef NVSHMEM_BUILD_PMI2

#include <errno.h>
#include <pmi2.h>

/* Rename PMI init function to avoid symbol clash */
#define bootstrap_pmi_init bootstrap_pmi2_init

static int wrap_pmi_size, wrap_pmi_rank;

#define WRAP_PMI_Finalize PMI2_Finalize
#define WRAP_PMI_Barrier PMI2_KVS_Fence

static inline int WRAP_PMI_Init(int *spawned) {
    int appnum;
    return PMI2_Init(spawned, &wrap_pmi_size, &wrap_pmi_rank, &appnum);
}

static inline int WRAP_PMI_Get_rank(int *rank) {
    *rank = wrap_pmi_rank;
    return 0;
}

static inline int WRAP_PMI_Get_size(int *size) {
    *size = wrap_pmi_size;
    return 0;
}

static inline int WRAP_PMI_KVS_Get_key_length_max(int *length) {
    *length = PMI2_MAX_KEYLEN;
    return 0;
}

static inline int WRAP_PMI_KVS_Get_value_length_max(int *length) {
    *length = PMI2_MAX_VALLEN;
    return 0;
}

static inline int WRAP_PMI_KVS_Get_my_name(char kvsname[], int length) {
    kvsname[0] = '\0';
    return 0;
}

static inline int WRAP_PMI_KVS_Get_name_length_max(int *length) {
    *length = 1;
    return 0;
}

static inline int WRAP_PMI_KVS_Commit(void) { return 0; }

static inline int WRAP_PMI_KVS_Put(const char kvsname[], const char key[], const char value[]) {
    return PMI2_KVS_Put(key, value);
}

static inline int WRAP_PMI_KVS_Get(const char kvsname[], const char key[], char value[],
                                   int length) {
    int vallen, status;

    status = PMI2_KVS_Get(NULL, PMI2_ID_NULL, key, value, PMI2_MAX_VALLEN, &vallen);

    if (vallen < 0)
        return -1;
    else
        return status;
}

#define WRAP_PMI_Abort PMI2_Abort

#else /* !NVSHMEM_BUILD_PMI2 */

#include <pmi_internal.h>

#define WRAP_PMI_Init SPMI_Init
#define WRAP_PMI_Finalize SPMI_Finalize

#define WRAP_PMI_Get_rank SPMI_Get_rank
#define WRAP_PMI_Get_size SPMI_Get_size

#define WRAP_PMI_KVS_Get_key_length_max SPMI_KVS_Get_key_length_max
#define WRAP_PMI_KVS_Get_my_name SPMI_KVS_Get_my_name
#define WRAP_PMI_KVS_Get_name_length_max SPMI_KVS_Get_name_length_max
#define WRAP_PMI_KVS_Get_value_length_max SPMI_KVS_Get_value_length_max

#define WRAP_PMI_Barrier SPMI_Barrier
#define WRAP_PMI_KVS_Commit SPMI_KVS_Commit

#define WRAP_PMI_KVS_Get SPMI_KVS_Get
#define WRAP_PMI_KVS_Put SPMI_KVS_Put
#define WRAP_PMI_Abort SPMI_Abort

#endif /* NVSHMEM_BUILD_PMI2 */

typedef struct {
    int singleton;
    int max_key_length;
    int max_value_length;
    int max_value_input_length;
    char *kvs_name;
    char *kvs_key;
    char *kvs_value;
} pmi_info_t;

static pmi_info_t pmi_info;
int bootstrap_debug_enable = 0;
struct nvshmemi_options_s env_attr;

static char encoding_table[] = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                                'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                                'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                                'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/'};
static char *decoding_table = NULL;
static int mod_table[] = {0, 2, 1};

static void base64_build_decoding_table() {
    decoding_table = (char *)malloc(256);

    for (int i = 0; i < 64; i++) decoding_table[(unsigned char)encoding_table[i]] = i;
}

static void base64_cleanup() { free(decoding_table); }

static size_t base64_encode_length(size_t in_len) { return (4 * ((in_len + 2) / 3)); }

static size_t base64_decode_length(size_t in_len) { return (in_len / 4 * 3); }

static size_t base64_encode(char *out, const unsigned char *in, size_t in_len) {
    size_t len = base64_encode_length(in_len);

    for (size_t i = 0, j = 0; i < in_len;) {
        uint32_t a = i < in_len ? (unsigned char)in[i++] : 0;
        uint32_t b = i < in_len ? (unsigned char)in[i++] : 0;
        uint32_t c = i < in_len ? (unsigned char)in[i++] : 0;

        uint32_t fused = (a << 0x10) + (b << 0x08) + c;

        out[j++] = encoding_table[(fused >> 3 * 6) & 0x3F];
        out[j++] = encoding_table[(fused >> 2 * 6) & 0x3F];
        out[j++] = encoding_table[(fused >> 1 * 6) & 0x3F];
        out[j++] = encoding_table[(fused >> 0 * 6) & 0x3F];
    }

    for (int i = 0; i < mod_table[in_len % 3]; i++) out[len - 1 - i] = '=';

    return len;
}

static size_t base64_decode(char *out, const char *in, size_t in_len) {
    size_t len = base64_decode_length(in_len);

    if (in[in_len - 1] == '=') (len)--;
    if (in[in_len - 2] == '=') (len)--;

    for (size_t i = 0, j = 0; i < in_len;) {
        uint32_t a = in[i] == '=' ? 0 & i++ : decoding_table[(int)(in[i++])];
        uint32_t b = in[i] == '=' ? 0 & i++ : decoding_table[(int)(in[i++])];
        uint32_t c = in[i] == '=' ? 0 & i++ : decoding_table[(int)(in[i++])];
        uint32_t d = in[i] == '=' ? 0 & i++ : decoding_table[(int)(in[i++])];

        uint32_t fused = (a << 3 * 6) + (b << 2 * 6) + (c << 1 * 6) + (d << 0 * 6);

        if (j < len) out[j++] = (fused >> 2 * 8) & 0xFF;
        if (j < len) out[j++] = (fused >> 1 * 8) & 0xFF;
        if (j < len) out[j++] = (fused >> 0 * 8) & 0xFF;
    }

    return len;
}

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

static int bootstrap_pmi_showinfo(struct bootstrap_handle *handle, int style) {
    bootstrap_util_print_header(style, "Bootstrap Options");
#define NVSHMEMI_ENV_DEF(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC)        \
    BOOTSTRAP_OPTIONS_PRINT_ENV(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC, \
                                NVSHMEMI_ENV_CAT_BOOTSTRAP, style)
#include "env_defs.h"
#undef NVSHMEMI_ENV_DEF

    printf("\n");
    return 0;
}

static int bootstrap_pmi_barrier(bootstrap_handle_t *handle) {
    int status = 0;

    status = WRAP_PMI_Barrier();
    BOOTSTRAP_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "WRAP_PMI_Barrier failed \n");

out:
    return status;
}

static int bootstrap_pmi_allgather(const void *sendbuf, void *recvbuf, int length,
                                   bootstrap_handle_t *handle) {
    int status = 0, length64;
    static int key_index = 1;

    if (handle->pg_size == 1) {
        memcpy(recvbuf, sendbuf, length);
        return 0;
    }

    // TODO: this can be worked around by breaking down the transfer into multiple messages
    int max_length = pmi_info.max_value_input_length;
    // int num_transfers = ((length + (max_length - 1)) / max_length);

    // INFO(NVSHMEM_BOOTSTRAP, "PMI allgather: transfer length: %d max input length: %d, transfers:
    // %d", length,
    //     max_length, num_transfers);

    int processed = 0;
    int transfer = 0;
    while (processed < length) {
        int curr_length = ((length - processed) > max_length) ? max_length : (length - processed);

        snprintf(pmi_info.kvs_key, pmi_info.max_key_length, "BOOTSTRAP-%04x-%08x-%04x", key_index,
                 handle->pg_rank, transfer);

        length64 = base64_encode((char *)pmi_info.kvs_value,
                                 (const unsigned char *)sendbuf + processed, curr_length);
        pmi_info.kvs_value[length64] = '\0';

        status = WRAP_PMI_KVS_Put(pmi_info.kvs_name, pmi_info.kvs_key, pmi_info.kvs_value);
        BOOTSTRAP_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "WRAP_PMI_KVS_Put failed \n");

        status = WRAP_PMI_KVS_Commit();
        BOOTSTRAP_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                               "WRAP_PMI_KVS_Commit failed \n");

        status = WRAP_PMI_Barrier();
        BOOTSTRAP_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "WRAP_PMI_Barrier failed \n");

        for (int i = 0; i < handle->pg_size; i++) {
            snprintf(pmi_info.kvs_key, pmi_info.max_key_length, "BOOTSTRAP-%04x-%08x-%04x",
                     key_index, i, transfer);

            // assumes that same length is passed by all the processes
            status =
                WRAP_PMI_KVS_Get(pmi_info.kvs_name, pmi_info.kvs_key, pmi_info.kvs_value, length64);
            BOOTSTRAP_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                   "WRAP_PMI_KVS_Get failed \n");

            base64_decode((char *)recvbuf + length * i + processed, (char *)pmi_info.kvs_value,
                          length64);
        }

        processed += curr_length;
        transfer++;
    }
out:
    key_index++;
    return status;
}

static int bootstrap_pmi_alltoall(const void *sendbuf, void *recvbuf, int length,
                                  bootstrap_handle_t *handle) {
    int status = 0, length64 = 0;
    static int key_index = 1;

    if (handle->pg_size == 1) {
        memcpy(recvbuf, sendbuf, length);
        return 0;
    }

    // TODO: this can be worked around by breaking down the transfer into multiple messages
    int max_length = pmi_info.max_value_input_length;
    // int num_transfers = ((length + (max_length - 1)) / max_length);

    // INFO(NVSHMEM_BOOTSTRAP, "PMI alltoall: transfer length: %d max input length: %d, transfers:
    // %d", length,
    //     max_length, num_transfers);

    int processed = 0;
    int transfer = 0;
    while (processed < length) {
        int curr_length = ((length - processed) > max_length) ? max_length : (length - processed);

        for (int i = 0; i < handle->pg_size; i++) {
            snprintf(pmi_info.kvs_key, pmi_info.max_key_length, "BOOTSTRAP-%04x-%08x-%08x-%04x",
                     key_index, handle->pg_rank, i, transfer);

            length64 =
                base64_encode((char *)pmi_info.kvs_value,
                              (const unsigned char *)sendbuf + i * length + processed, curr_length);
            pmi_info.kvs_value[length64] = '\0';

            status = WRAP_PMI_KVS_Put(pmi_info.kvs_name, pmi_info.kvs_key, pmi_info.kvs_value);
            BOOTSTRAP_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                   "WRAP_PMI_KVS_Put failed \n");
        }

        status = WRAP_PMI_KVS_Commit();
        BOOTSTRAP_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                               "WRAP_PMI_KVS_Commit failed \n");

        status = WRAP_PMI_Barrier();
        BOOTSTRAP_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "WRAP_PMI_Barrier failed \n");

        for (int i = 0; i < handle->pg_size; i++) {
            snprintf(pmi_info.kvs_key, pmi_info.max_key_length, "BOOTSTRAP-%04x-%08x-%08x-%04x",
                     key_index, i, handle->pg_rank, transfer);

            // assumes that same length is passed by all the processes
            status =
                WRAP_PMI_KVS_Get(pmi_info.kvs_name, pmi_info.kvs_key, pmi_info.kvs_value, length64);
            BOOTSTRAP_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                   "WRAP_PMI_KVS_Get failed \n");

            base64_decode((char *)recvbuf + length * i + processed, (char *)pmi_info.kvs_value,
                          length64);
        }

        processed += curr_length;
        transfer++;
    }
out:
    key_index++;
    return status;
}

static void bootstrap_pmi_global_exit(int status) {
#ifdef NVSHMEM_BUILD_PMI2
    if (status == 0) {
        BOOTSTRAP_ERROR_PRINT(
            "PMI2 does not support global exit with 0 status. Exiting with ECANCELED.\n");
        WRAP_PMI_Abort(ECANCELED, "NVSHMEM Global Exit.\n");
    } else {
#endif
        WRAP_PMI_Abort(status, "NVSHMEM Global Exit.\n");
#ifdef NVSHMEM_BUILD_PMI2
    }
    BOOTSTRAP_ERROR_PRINT("PMI2_Abort failed. Manually exiting this process.\n");
#else
    BOOTSTRAP_ERROR_PRINT("PMI_Abort failed. Manually exiting this process.\n");
#endif

    /* Both the PMI and PMI2 versions of abort only return if the abort was unsuccessful. */
    exit(1);
}

static int bootstrap_pmi_finalize(bootstrap_handle_t *handle) {
    int status = 0;

    status = WRAP_PMI_Finalize();
    BOOTSTRAP_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, error, "WRAP_PMI_Finalize failed \n");

    base64_cleanup();

    if (pmi_info.kvs_name) free(pmi_info.kvs_name);

error:
    return status;
}

int nvshmemi_bootstrap_plugin_init(void *attr, bootstrap_handle_t *handle, const int abi_version) {
    int status = 0;
    int spawned = 0;
    int rank, size, key_length, value_length, name_length;
    int bootstrap_version = NVSHMEMI_BOOTSTRAP_ABI_VERSION;
    if (!nvshmemi_is_bootstrap_compatible(bootstrap_version, abi_version, true)) {
        BOOTSTRAP_ERROR_PRINT(
            "PMI bootstrap version (%d) is not compatible with NVSHMEM version (%d)",
            bootstrap_version, abi_version);
        exit(-1);
    }

    status = WRAP_PMI_Init(&spawned);
    BOOTSTRAP_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "WRAP_PMI_Init_failed failed \n");

    handle->version = NVSHMEM_BOOTSTRAP_MAJOR_MINOR_VERSION(abi_version) <
                              NVSHMEM_BOOTSTRAP_MAJOR_MINOR_VERSION(bootstrap_version)
                          ? abi_version
                          : bootstrap_version;
#ifndef NVSHMEM_BUILD_PMI2
    if (!spawned) {
        // INFO(NVSHMEM_BOOTSTRAP, "taking singleton path");

        // singleton launch
        handle->pg_rank = 0;
        handle->pg_size = 1;
        pmi_info.singleton = 1;
        handle->allgather = bootstrap_pmi_allgather;
        handle->alltoall = bootstrap_pmi_alltoall;
        handle->global_exit = bootstrap_pmi_global_exit;
        handle->barrier = bootstrap_pmi_barrier;
        handle->pre_init_ops = NULL;
        handle->comm_state = (void *)&pmi_info;
        handle->show_info = bootstrap_pmi_showinfo;
    } else {
#endif
        status = WRAP_PMI_Get_rank(&rank);
        BOOTSTRAP_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, error,
                               "WRAP_PMI_Get_rank failed \n");

        status = WRAP_PMI_Get_size(&size);
        BOOTSTRAP_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, error,
                               "WRAP_PMI_Get_size failed \n");

        handle->pg_rank = rank;
        handle->pg_size = size;
        handle->allgather = bootstrap_pmi_allgather;
        handle->alltoall = bootstrap_pmi_alltoall;
        handle->global_exit = bootstrap_pmi_global_exit;
        handle->barrier = bootstrap_pmi_barrier;
        handle->pre_init_ops = NULL;
        handle->comm_state = (void *)&pmi_info;
        handle->show_info = bootstrap_pmi_showinfo;

        status = WRAP_PMI_KVS_Get_name_length_max(&name_length);
        BOOTSTRAP_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, error,
                               "WRAP_PMI_KVS_Get_name_length_max failed \n");

        status = WRAP_PMI_KVS_Get_key_length_max(&key_length);
        BOOTSTRAP_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, error,
                               "WRAP_PMI_KVS_Get_key_length_max failed \n");

        status = WRAP_PMI_KVS_Get_value_length_max(&value_length);
        BOOTSTRAP_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, error,
                               "WRAP_PMI_KVS_Get_value_length_max failed \n");

        pmi_info.max_key_length = key_length;
        pmi_info.max_value_length = value_length;

        // hacky workaround to allow space for metadata in KVS_Put, needs investgation
        pmi_info.max_value_input_length = base64_decode_length(value_length / 2);
        // INFO(NVSHMEM_BOOTSTRAP, "PMI max key length: %d max value length %d", key_length,
        //     value_length);

        pmi_info.kvs_name = (char *)malloc(name_length);

        BOOTSTRAP_NULL_ERROR_JMP(pmi_info.kvs_name, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, error,
                                 "memory allocation for kvs_name failed \n");

        pmi_info.kvs_key = (char *)malloc(key_length);

        BOOTSTRAP_NULL_ERROR_JMP(pmi_info.kvs_key, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, error,
                                 "memory allocation for kvs_key failed \n");

        pmi_info.kvs_value = (char *)malloc(value_length);

        BOOTSTRAP_NULL_ERROR_JMP(pmi_info.kvs_value, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, error,
                                 "memory allocation for kvs_value failed \n");

        status = WRAP_PMI_KVS_Get_my_name(pmi_info.kvs_name, name_length);
        BOOTSTRAP_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, error,
                               "WRAP_PMI_KVS_Get_my_name failed \n");
#ifndef NVSHMEM_BUILD_PMI2
    }
#endif

    handle->finalize = bootstrap_pmi_finalize;

    base64_build_decoding_table();

error:
    if (status) {
        if (pmi_info.kvs_name) free(pmi_info.kvs_name);
        if (pmi_info.kvs_key) free(pmi_info.kvs_key);
        if (pmi_info.kvs_value) free(pmi_info.kvs_value);
    }
out:
    return status;
}
