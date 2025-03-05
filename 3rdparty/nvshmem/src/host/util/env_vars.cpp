/****
 * Copyright (c) 2016-2021, NVIDIA Corporation.  All rights reserved.
 *
 * Copyright 2011 Sandia Corporation. Under the terms of Contract
 * DE-AC04-94AL85000 with Sandia Corporation, the U.S.  Government
 * retains certain rights in this software.
 *
 * Copyright (c) 2017 Intel Corporation. All rights reserved.
 * This software is available to you under the BSD license.
 *
 * Portions of this file are derived from Sandia OpenSHMEM.
 *
 * See COPYRIGHT for license information
 ****/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "internal/host/nvshmem_nvtx.hpp"
#include "internal/host/util.h"
#include "bootstrap_host_transport/env_defs_internal.h"
#include "device_host/nvshmem_common.cuh"  // IWYU pragma: keep

#define NVSHPRI_float "%0.2f"
#define NVSHPRI_double "%0.2f"
#define NVSHPRI_char "%hhd"
#define NVSHPRI_schar "%hhd"
#define NVSHPRI_short "%hd"
#define NVSHPRI_int "%d"
#define NVSHPRI_long "%ld"
#define NVSHPRI_longlong "%lld"
#define NVSHPRI_uchar "%hhu"
#define NVSHPRI_ushort "%hu"
#define NVSHPRI_uint "%u"
#define NVSHPRI_ulong "%lu"
#define NVSHPRI_ulonglong "%llu"
#define NVSHPRI_int8 "%" PRIi8
#define NVSHPRI_int16 "%" PRIi16
#define NVSHPRI_int32 "%" PRIi32
#define NVSHPRI_int64 "%" PRIi64
#define NVSHPRI_uint8 "%" PRIu8
#define NVSHPRI_uint16 "%" PRIu16
#define NVSHPRI_uint32 "%" PRIu32
#define NVSHPRI_uint64 "%" PRIu64
#define NVSHPRI_size "%zu"
#define NVSHPRI_ptrdiff "%zu"
#define NVSHPRI_bool "%s"
#define NVSHPRI_string "\"%s\""

struct nvshmemi_options_s nvshmemi_options;
bool nvshmemi_options_inited = false;

int nvshmemi_options_init(void) {
    int ret = 0;
    if (!nvshmemi_options_inited) {
        ret = nvshmemi_env_options_init(&nvshmemi_options);
        nvshmemi_options_inited = true;
        return (ret);
    } else {
        /* Options already inited */
        return (ret);
    }
}

static void nvshmemi_options_print_heading(const char *h, int style) {
    switch (style) {
        case NVSHMEMI_OPTIONS_STYLE_INFO:
            printf("%s:\n", h);
            break;
        case NVSHMEMI_OPTIONS_STYLE_RST:
            printf("%s\n", h);
            for (const char *c = h; *c != '\0'; c++) putchar('~');
            printf("\n\n");
            break;
        default:
            assert(0);  // FIXME
    }
}

#define NVSHMEMI_OPTIONS_PRINT_ENV(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC, DESIRED_CAT, STYLE) \
    if (CATEGORY == DESIRED_CAT) {                                                                \
        switch (STYLE) {                                                                          \
            char *desc_wrapped;                                                                   \
            case NVSHMEMI_OPTIONS_STYLE_INFO:                                                     \
                desc_wrapped = nvshmemu_wrap(SHORT_DESC, NVSHMEMI_WRAPLEN, "\t", 1);              \
                printf("  NVSHMEM_%-20s " NVSHPRI_##KIND " (type: %s, default: " NVSHPRI_##KIND   \
                       ")\n\t%s\n",                                                               \
                       #NAME, NVSHFMT_##KIND(nvshmemi_options.NAME), #KIND,                       \
                       NVSHFMT_##KIND(DEFAULT), desc_wrapped);                                    \
                free(desc_wrapped);                                                               \
                break;                                                                            \
            case NVSHMEMI_OPTIONS_STYLE_RST:                                                      \
                desc_wrapped = nvshmemu_wrap(SHORT_DESC, NVSHMEMI_WRAPLEN, NULL, 0);              \
                printf(".. c:var:: NVSHMEM_%s\n", #NAME);                                         \
                printf("\n");                                                                     \
                printf("| *Type: %s*\n", #KIND);                                                  \
                printf("| *Default: " NVSHPRI_##KIND "*\n", NVSHFMT_##KIND(DEFAULT));             \
                printf("\n");                                                                     \
                printf("%s\n", desc_wrapped);                                                     \
                printf("\n");                                                                     \
                free(desc_wrapped);                                                               \
                break;                                                                            \
            default:                                                                              \
                assert(0); /* FIXME */                                                            \
        }                                                                                         \
    }

void nvshmemi_options_print(int style) {
    nvshmemi_options_print_heading("Standard options", style);
#define NVSHMEMI_ENV_DEF(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC)       \
    NVSHMEMI_OPTIONS_PRINT_ENV(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC, \
                               NVSHMEMI_ENV_CAT_OPENSHMEM, style)
#include "env_defs.h"
#undef NVSHMEMI_ENV_DEF
    printf("\n");

    nvshmemi_options_print_heading("Bootstrap options", style);
#define NVSHMEMI_ENV_DEF(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC)       \
    NVSHMEMI_OPTIONS_PRINT_ENV(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC, \
                               NVSHMEMI_ENV_CAT_BOOTSTRAP, style)
#include "env_defs.h"
#undef NVSHMEMI_ENV_DEF
    printf("\n");

    nvshmemi_options_print_heading("Additional options", style);
#define NVSHMEMI_ENV_DEF(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC)                               \
    NVSHMEMI_OPTIONS_PRINT_ENV(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC, NVSHMEMI_ENV_CAT_OTHER, \
                               style)
#include "env_defs.h"
#undef NVSHMEMI_ENV_DEF
    printf("\n");

    nvshmemi_options_print_heading("Collectives options", style);
#define NVSHMEMI_ENV_DEF(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC)       \
    NVSHMEMI_OPTIONS_PRINT_ENV(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC, \
                               NVSHMEMI_ENV_CAT_COLLECTIVES, style)
#include "env_defs.h"
#undef NVSHMEMI_ENV_DEF
    printf("\n");

    nvshmemi_options_print_heading("Transport options", style);
#define NVSHMEMI_ENV_DEF(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC)       \
    NVSHMEMI_OPTIONS_PRINT_ENV(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC, \
                               NVSHMEMI_ENV_CAT_TRANSPORT, style)
#include "env_defs.h"
#undef NVSHMEMI_ENV_DEF
    printf("\n");

    if (nvshmemi_options.INFO_HIDDEN) {
        nvshmemi_options_print_heading("Hidden options", style);
#define NVSHMEMI_ENV_DEF(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC)                                \
    NVSHMEMI_OPTIONS_PRINT_ENV(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC, NVSHMEMI_ENV_CAT_HIDDEN, \
                               style)
#include "env_defs.h"
#undef NVSHMEMI_ENV_DEF
        printf("\n");
    }

#ifndef NVTX_DISABLE
    if (nvshmemi_options.NVTX) {
        nvshmemi_options_print_heading("NVTX options", style);
#define NVSHMEMI_ENV_DEF(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC)                              \
    NVSHMEMI_OPTIONS_PRINT_ENV(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC, NVSHMEMI_ENV_CAT_NVTX, \
                               style)
#include "env_defs.h"

        if (style == NVSHMEMI_OPTIONS_STYLE_RST) printf(".. code-block:: none\n\n");

        nvshmem_nvtx_print_options();
#undef NVSHMEMI_ENV_DEF
    }
#endif /* !NVTX_DISABLE */

    printf("\n");
}
