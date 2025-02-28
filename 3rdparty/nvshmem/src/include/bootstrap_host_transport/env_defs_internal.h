/****
 * Copyright (c) 2016-2023, NVIDIA Corporation.  All rights reserved.
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

#ifndef NVSHMEM_ENV_DEFS_INTERNAL
#define NVSHMEM_ENV_DEFS_INTERNAL

#include <alloca.h>   // for alloca
#include <errno.h>    // for errno
#include <math.h>     // for ceil
#include <stdbool.h>  // IWYU pragma: keep
#include <stdio.h>    // for fprintf, NULL, snprintf, sscanf, stderr, size_t
#include <stdlib.h>   // for getenv, strtol
#include <string.h>   // for strlen

enum nvshmemi_env_categories {
    NVSHMEMI_ENV_CAT_OPENSHMEM,
    NVSHMEMI_ENV_CAT_OTHER,
    NVSHMEMI_ENV_CAT_COLLECTIVES,
    NVSHMEMI_ENV_CAT_TRANSPORT,
    NVSHMEMI_ENV_CAT_HIDDEN,
    NVSHMEMI_ENV_CAT_NVTX,
    NVSHMEMI_ENV_CAT_BOOTSTRAP
};

#define SYMMETRIC_SIZE_DEFAULT 1024 * 1024 * 1024

typedef int nvshmemi_env_int;
typedef long nvshmemi_env_long;
typedef size_t nvshmemi_env_size;
typedef bool nvshmemi_env_bool;
typedef const char *nvshmemi_env_string;

#define NVSHFMT_int(_v) _v
#define NVSHFMT_long(_v) _v
#define NVSHFMT_size(_v) _v
#define NVSHFMT_bool(_v) (_v) ? "true" : "false"
#define NVSHFMT_string(_v) _v

struct nvshmemi_options_s {
#define NVSHMEMI_ENV_DEF(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC) nvshmemi_env_##KIND NAME;
#include "env_defs.h"
#undef NVSHMEMI_ENV_DEF

#define NVSHMEMI_ENV_DEF(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC) bool NAME##_provided;
#include "env_defs.h"
#undef NVSHMEMI_ENV_DEF
};

/* atol() + optional scaled suffix recognition: 1K, 2M, 3G, 1T */
static inline int nvshmemi_atol_scaled(const char *str, nvshmemi_env_size *out) {
    int scale, n;
    double p = -1.0;
    char f;

    n = sscanf(str, "%lf%c", &p, &f);

    if (n == 2) {
        switch (f) {
            case 'k':
            case 'K':
                scale = 10;
                break;
            case 'm':
            case 'M':
                scale = 20;
                break;
            case 'g':
            case 'G':
                scale = 30;
                break;
            case 't':
            case 'T':
                scale = 40;
                break;
            default:
                return 1;
        }
    } else if (p < 0) {
        return 1;
    } else
        scale = 0;

    *out = (nvshmemi_env_size)ceil(p * (1lu << scale));
    return 0;
}

static inline long nvshmemi_errchk_atol(const char *s) {
    long val;
    char *e;
    errno = 0;

    val = strtol(s, &e, 0);
    if (errno != 0 || e == s) {
        fprintf(stderr, "Environment variable conversion failed (%s)\n", s);
    }

    return val;
}

static inline const char *nvshmemi_getenv_helper(const char *prefix, const char *name) {
    char *env_name;
    const char *env_value = NULL;
    size_t len;
    int ret;

    len = strlen(prefix) + 1 /* '_' */ + strlen(name) + 1 /* '\0' */;
    env_name = (char *)alloca(len);
    ret = snprintf(env_name, len, "%s_%s", prefix, name);
    if (ret < 0)
        fprintf(stderr, "WARNING: Error in sprintf: %s_%s\n", prefix, name);
    else
        env_value = (const char *)getenv(env_name);

    return env_value;
}

static inline const char *nvshmemi_getenv(const char *name) {
    const char *env_value;

    env_value = nvshmemi_getenv_helper("NVSHMEM", name);
    if (env_value != NULL) return env_value;

    return NULL;
}

static inline int nvshmemi_getenv_string(const char *name, nvshmemi_env_string default_val,
                                         nvshmemi_env_string *out, bool *provided) {
    const char *env = nvshmemi_getenv(name);
    *provided = (env != NULL);
    *out = (*provided) ? env : default_val;
    return 0;
}

static inline int nvshmemi_getenv_int(const char *name, nvshmemi_env_int default_val,
                                      nvshmemi_env_int *out, bool *provided) {
    const char *env = nvshmemi_getenv(name);
    *provided = (env != NULL);
    *out = (*provided) ? (int)nvshmemi_errchk_atol(env) : default_val;
    return 0;
}

static inline int nvshmemi_getenv_long(const char *name, nvshmemi_env_long default_val,
                                       nvshmemi_env_long *out, bool *provided) {
    const char *env = nvshmemi_getenv(name);
    *provided = (env != NULL);
    *out = (*provided) ? nvshmemi_errchk_atol(env) : default_val;
    return 0;
}

static inline int nvshmemi_getenv_size(const char *name, nvshmemi_env_size default_val,
                                       nvshmemi_env_size *out, bool *provided) {
    const char *env = nvshmemi_getenv(name);
    *provided = (env != NULL);
    if (*provided) {
        int ret = nvshmemi_atol_scaled(env, out);
        if (ret) {
            fprintf(stderr, "Invalid size in environment variable '%s' (%s)\n", name, env);
            return ret;
        }
    } else
        *out = default_val;
    return 0;
}

static inline int nvshmemi_getenv_bool(const char *name, nvshmemi_env_bool default_val,
                                       nvshmemi_env_bool *out, bool *provided) {
    const char *env = nvshmemi_getenv(name);
    *provided = (env != NULL);

    if (*provided &&
        (env[0] == '0' || env[0] == 'N' || env[0] == 'n' || env[0] == 'F' || env[0] == 'f')) {
        *out = false;
    } else if (*provided) {
        /* The default behavior specified by OpenSHMEM is to enable boolean
         * options whenever the environment variable is set */
        *out = true;
    } else {
        *out = default_val;
    }

    return 0;
}

static inline int nvshmemi_env_options_init(struct nvshmemi_options_s *options) {
    int ret;
#define NVSHMEMI_ENV_DEF(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC)                              \
    ret = nvshmemi_getenv_##KIND(#NAME, DEFAULT, &(options->NAME), &(options->NAME##_provided)); \
    if (ret) return ret;
#include "env_defs.h"
#undef NVSHMEMI_ENV_DEF
    return 0;
}
#endif
