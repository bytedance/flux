/*
 * Copyright (c) 2016-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef BOOTSTRAP_UTIL_H
#define BOOTSTRAP_UTIL_H

#include <errno.h>   // for errno, EAGAIN, EINTR, EWOULDBLOCK
#include <limits.h>  // for INT_MIN
#include <stdio.h>   // for stderr, stdout, NULL

#define BOOTPRI_float "%0.2f"
#define BOOTPRI_double "%0.2f"
#define BOOTPRI_char "%hhd"
#define BOOTPRI_schar "%hhd"
#define BOOTPRI_short "%hd"
#define BOOTPRI_int "%d"
#define BOOTPRI_long "%ld"
#define BOOTPRI_longlong "%lld"
#define BOOTPRI_uchar "%hhu"
#define BOOTPRI_ushort "%hu"
#define BOOTPRI_uint "%u"
#define BOOTPRI_ulong "%lu"
#define BOOTPRI_ulonglong "%llu"
#define BOOTPRI_int8 "%" PRIi8
#define BOOTPRI_int16 "%" PRIi16
#define BOOTPRI_int32 "%" PRIi32
#define BOOTPRI_int64 "%" PRIi64
#define BOOTPRI_uint8 "%" PRIu8
#define BOOTPRI_uint16 "%" PRIu16
#define BOOTPRI_uint32 "%" PRIu32
#define BOOTPRI_uint64 "%" PRIu64
#define BOOTPRI_size "%zu"
#define BOOTPRI_ptrdiff "%zu"
#define BOOTPRI_bool "%s"
#define BOOTPRI_string "\"%s\""

typedef enum bootstrap_result {
    BOOTSTRAP_SUCCESS = 0,
    BOOTSTRAP_UNHANDLED_CUDA_ERROR = -1,
    BOOTSTRAP_SYSTEM_ERROR = -2,
    BOOTSTRAP_INTERNAL_ERROR = -3,
    BOOTSTRAP_INVALID_ARGUMENT = -4,
    BOOTSTRAP_INVALID_USAGE = -5,
    BOOTSTRAP_REMOTE_ERROR = -6,
    BOOTSTRAP_INPROGRESS = -7,
    BOOTSTRAP_NUM_RESULTS = -8,
    /* Reserved for upto N for bootstrap to nccl mapping */
    BOOTSTRAP_ERROR_MAX = INT_MIN
} bootstrap_result_t;

extern int bootstrap_debug_enable;

#define BOOTSTRAP_ERROR_PRINT(...)                                       \
    do {                                                                 \
        fprintf(stderr, "%s:%s:%d: ", __FILE__, __FUNCTION__, __LINE__); \
        fprintf(stderr, __VA_ARGS__);                                    \
        fprintf(stderr, "\n");                                           \
    } while (0)

#define BOOTSTRAP_DEBUG_PRINT(...)                                           \
    do {                                                                     \
        if (bootstrap_debug_enable) {                                        \
            fprintf(stdout, "%s:%s:%d: ", __FILE__, __FUNCTION__, __LINE__); \
            fprintf(stdout, __VA_ARGS__);                                    \
            fprintf(stdout, "\n");                                           \
        }                                                                    \
    } while (0)

#define BOOTSTRAP_INFO_PRINT(...)     \
    do {                              \
        fprintf(stdout, __VA_ARGS__); \
        fprintf(stdout, "\n");        \
    } while (0)

#define BOOTSTRAP_CHECK(call)                                                        \
    do {                                                                             \
        bootstrap_result_t RES = (call);                                             \
        if (RES != BOOTSTRAP_SUCCESS) {                                              \
            /* Print the back trace*/                                                \
            fprintf(stderr, "%s:%d: non-zero status: %d ", __FILE__, __LINE__, RES); \
            return RES;                                                              \
        }                                                                            \
    } while (0)

#define BOOTSTRAP_INFO(call)                                                         \
    do {                                                                             \
        bootstrap_result_t RES = (call);                                             \
        if (RES != BOOTSTRAP_SUCCESS) {                                              \
            /* Print the back trace*/                                                \
            fprintf(stderr, "%s:%d: non-zero status: %d ", __FILE__, __LINE__, RES); \
        }                                                                            \
    } while (0)

#define BOOTSTRAP_EQCHECK(statement, value)                                                  \
    do {                                                                                     \
        if ((statement) == value) {                                                          \
            /* Print the back trace*/                                                        \
            fprintf(stderr, "%s:%s:%d: ", __FILE__, __FUNCTION__, __LINE__);                 \
            fprintf(stderr, "non-zero status: %d => Reason: (%s)\n", BOOTSTRAP_SYSTEM_ERROR, \
                    strerror(errno));                                                        \
            return BOOTSTRAP_SYSTEM_ERROR;                                                   \
        }                                                                                    \
    } while (0)

#define BOOTSTRAP_NEQCHECK(statement, value)                                                 \
    do {                                                                                     \
        if ((statement) != value) {                                                          \
            /* Print the back trace*/                                                        \
            fprintf(stderr, "%s:%s:%d: ", __FILE__, __FUNCTION__, __LINE__);                 \
            fprintf(stderr, "non-zero status: %d => Reason: (%s)\n", BOOTSTRAP_SYSTEM_ERROR, \
                    strerror(errno));                                                        \
            return BOOTSTRAP_SYSTEM_ERROR;                                                   \
        }                                                                                    \
    } while (0)

#define BOOTSTRAP_SYSCHECK(call, name)                                                     \
    do {                                                                                   \
        int retval = (call);                                                               \
        if (retval == -1 && (errno == EINTR || errno == EWOULDBLOCK || errno == EAGAIN)) { \
            fprintf(stderr, "%s:%s:%d: ", __FILE__, __FUNCTION__, __LINE__);               \
            fprintf(stderr, "Call to " name " failed : %s\n", strerror(errno));            \
            return BOOTSTRAP_SYSTEM_ERROR;                                                 \
        }                                                                                  \
    } while (0)

#define BOOTSTRAP_SYSCHECKGOTO(statement, RES, label)                                       \
    do {                                                                                    \
        if ((statement) == -1) {                                                            \
            /* Print the back trace*/                                                       \
            RES = BOOTSTRAP_SYSTEM_ERROR;                                                   \
            fprintf(stderr, "%s:%s:%d: ", __FILE__, __FUNCTION__, __LINE__);                \
            fprintf(stderr, "non-zero status: %d => Reason: (%s)\n", RES, strerror(errno)); \
            goto label;                                                                     \
        }                                                                                   \
    } while (0)

#define BOOTSTRAP_EQCHECKGOTO(statement, value, RES, label)                                 \
    do {                                                                                    \
        if ((statement) == value) {                                                         \
            /* Print the back trace*/                                                       \
            RES = BOOTSTRAP_SYSTEM_ERROR;                                                   \
            fprintf(stderr, "%s:%s:%d: ", __FILE__, __FUNCTION__, __LINE__);                \
            fprintf(stderr, "non-zero status: %d => Reason: (%s)\n", RES, strerror(errno)); \
            goto label;                                                                     \
        }                                                                                   \
    } while (0)

#define BOOTSTRAP_CHECKGOTO(call, RES, label)                                \
    do {                                                                     \
        RES = (call);                                                        \
        if (RES != BOOTSTRAP_SUCCESS && RES != BOOTSTRAP_INPROGRESS) {       \
            /* Print the back trace*/                                        \
            fprintf(stderr, "%s:%s:%d: ", __FILE__, __FUNCTION__, __LINE__); \
            fprintf(stderr, "non-zero status: %d\n", RES);                   \
            goto label;                                                      \
        }                                                                    \
    } while (0)

#define BOOTSTRAP_NE_ERROR_JMP(status, expected, err, label, ...)                       \
    do {                                                                                \
        if (status != expected) {                                                       \
            fprintf(stderr, "%s:%d: non-zero status: %d ", __FILE__, __LINE__, status); \
            fprintf(stderr, __VA_ARGS__);                                               \
            status = err;                                                               \
            goto label;                                                                 \
        }                                                                               \
    } while (0)

#define BOOTSTRAP_NZ_ERROR_JMP(status, err, label, ...)                                 \
    do {                                                                                \
        if (status != 0) {                                                              \
            fprintf(stderr, "%s:%d: non-zero status: %d ", __FILE__, __LINE__, status); \
            fprintf(stderr, __VA_ARGS__);                                               \
            status = err;                                                               \
            goto label;                                                                 \
        }                                                                               \
    } while (0)

#define BOOTSTRAP_NULL_ERROR_JMP(var, status, err, label, ...)         \
    do {                                                               \
        if (var == NULL) {                                             \
            fprintf(stderr, "%s:%d: NULL value ", __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                              \
            status = err;                                              \
            goto label;                                                \
        }                                                              \
    } while (0)

#define BOOTSTRAP_PTR_FREE(ptr) \
    do {                        \
        if ((ptr) != NULL) {    \
            free(ptr);          \
        }                       \
    } while (0)

#ifdef __cplusplus
extern "C" {
#endif

char *bootstrap_util_wrap_string(const char *str, const size_t wraplen, const char *indent,
                                 const int strip_backticks);

void bootstrap_util_print_header(int style, const char *h);

#ifdef __cplusplus
};
#endif

#endif /*! BOOTSTRAP_UTIL_H */
