/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See COPYRIGHT.txt for license information
 */

#ifndef _NVSHMEMX_ERROR_H_
#define _NVSHMEMX_ERROR_H_
#include <errno.h>  // for errno
#include <limits.h>
#include <stdio.h>   // for stderr, stdout, NULL
#include <string.h>  // IWYU pragma: keep for strerror
#include <stdlib.h>  // IWYU pragma: keep for exit

/* The !! idiom is used to convert non-boolean types to booleans.
 * Doing so in this case allows us to ensure that __builtin_expect
 * will be given a clean boolean value for the comparison. */
#define nvshmemxi_error_unlikely(x) __builtin_expect(!!(x), 0)

#ifdef __cplusplus
extern "C" {
#endif

enum nvshmemx_status {
    NVSHMEMX_SUCCESS = 0,
    NVSHMEMX_ERROR_INVALID_VALUE,
    NVSHMEMX_ERROR_OUT_OF_MEMORY,
    NVSHMEMX_ERROR_NOT_SUPPORTED,
    NVSHMEMX_ERROR_SYMMETRY,
    NVSHMEMX_ERROR_GPU_NOT_SELECTED,
    NVSHMEMX_ERROR_COLLECTIVE_LAUNCH_FAILED,
    NVSHMEMX_ERROR_INTERNAL,
    NVSHMEMX_ERROR_SENTINEL = INT_MAX
};

#define NVSHMEMI_ERROR_EXIT(...)                                         \
    do {                                                                 \
        fprintf(stderr, "%s:%s:%d: ", __FILE__, __FUNCTION__, __LINE__); \
        fprintf(stderr, __VA_ARGS__);                                    \
        fprintf(stderr, "\n");                                           \
        exit(-1);                                                        \
    } while (0)

#define NVSHMEMI_ERROR_PRINT(...)                                        \
    do {                                                                 \
        fprintf(stderr, "%s:%s:%d: ", __FILE__, __FUNCTION__, __LINE__); \
        fprintf(stderr, __VA_ARGS__);                                    \
        fprintf(stderr, "\n");                                           \
    } while (0)

#define NVSHMEMI_WARN_PRINT(...)      \
    do {                              \
        fprintf(stdout, "WARN: ");    \
        fprintf(stdout, __VA_ARGS__); \
        fprintf(stdout, "\n");        \
    } while (0)

#define NVSHMEMI_ERROR_JMP(status, err, label, ...)                              \
    do {                                                                         \
        fprintf(stderr, "%s:%d: non-zero status: %d ", __FILE__, __LINE__, err); \
        fprintf(stderr, __VA_ARGS__);                                            \
        fprintf(stderr, "\n");                                                   \
        status = err;                                                            \
        goto label;                                                              \
    } while (0)

#define NVSHMEMI_NULL_ERROR_JMP(var, status, err, label, ...)          \
    do {                                                               \
        if (nvshmemxi_error_unlikely(var == NULL)) {                   \
            fprintf(stderr, "%s:%d: NULL value ", __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                              \
            fprintf(stderr, "\n");                                     \
            status = err;                                              \
            goto label;                                                \
        }                                                              \
    } while (0)

#define NVSHMEMI_EQ_ERROR_JMP(status, expected, err, label, ...)                     \
    do {                                                                             \
        if (nvshmemxi_error_unlikely(status == expected)) {                          \
            fprintf(stderr, "%s:%d: error status: %d ", __FILE__, __LINE__, status); \
            fprintf(stderr, __VA_ARGS__);                                            \
            fprintf(stderr, "\n");                                                   \
            status = err;                                                            \
            goto label;                                                              \
        }                                                                            \
    } while (0)

#define NVSHMEMI_NE_ERROR_JMP(status, expected, err, label, ...)                        \
    do {                                                                                \
        if (nvshmemxi_error_unlikely(status != expected)) {                             \
            fprintf(stderr, "%s:%d: non-zero status: %d ", __FILE__, __LINE__, status); \
            fprintf(stderr, __VA_ARGS__);                                               \
            fprintf(stderr, "\n");                                                      \
            status = err;                                                               \
            goto label;                                                                 \
        }                                                                               \
    } while (0)

#define NVSHMEMI_NZ_ERROR_JMP(status, err, label, ...)                                  \
    do {                                                                                \
        if (nvshmemxi_error_unlikely(status != 0)) {                                    \
            fprintf(stderr, "%s:%d: non-zero status: %d ", __FILE__, __LINE__, status); \
            fprintf(stderr, __VA_ARGS__);                                               \
            fprintf(stderr, "\n");                                                      \
            status = err;                                                               \
            goto label;                                                                 \
        }                                                                               \
    } while (0)

#define NVSHMEMI_CHECK_ERROR_JMP(statement, status, err, label, ...) \
    do {                                                             \
        if (nvshmemxi_error_unlikely(statement)) {                   \
            fprintf(stderr, "%s:%d: ", __FILE__, __LINE__);          \
            fprintf(stderr, __VA_ARGS__);                            \
            fprintf(stderr, "\n");                                   \
            status = err;                                            \
            goto label;                                              \
        }                                                            \
    } while (0)

#define NVSHMEMI_NZ_EXIT(status, ...)                                                          \
    do {                                                                                       \
        if (nvshmemxi_error_unlikely(status != 0)) {                                           \
            fprintf(stderr, "%s:%d: non-zero status: %d: %s, exiting... ", __FILE__, __LINE__, \
                    status, strerror(errno));                                                  \
            fprintf(stderr, __VA_ARGS__);                                                      \
            fprintf(stderr, "\n");                                                             \
            exit(-1);                                                                          \
        }                                                                                      \
    } while (0)

#define NVSHMEMI_NZ_SYSCHECK_EXIT(sys_status, ...)                                             \
    do {                                                                                       \
        if (nvshmemxi_error_unlikely((sys_status) != 0)) {                                     \
            fprintf(stderr, "%s:%d: non-zero status: %d: %s, exiting... ", __FILE__, __LINE__, \
                    (sys_status), strerror(sys_status));                                       \
            fprintf(stderr, __VA_ARGS__);                                                      \
            fprintf(stderr, "\n");                                                             \
            exit(-1);                                                                          \
        }                                                                                      \
    } while (0)

#ifdef __cplusplus
}
#endif

#endif
