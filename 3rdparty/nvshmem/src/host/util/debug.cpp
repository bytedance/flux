/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See COPYRIGHT.txt for license information
 */

#include <cuda_runtime.h>
#include <pthread.h>
#include <stdlib.h>
#include <syscall.h>
#include <unistd.h>
#include <cstdarg>
#include <cstdio>
#include "non_abi/nvshmem_build_options.h"  // IWYU pragma: keep for NVSHMEM_TRACE
#ifdef NVSHMEM_TRACE
#include <chrono>
#endif

#include "internal/host/debug.h"
#include "internal/host/util.h"
#define nvshmemi_gettid() (pid_t) syscall(SYS_gettid)

extern "C" {
void nvshmem_debug_log(nvshmem_debug_log_level level, unsigned long flags, const char *filefunc,
                       int line, const char *fmt, ...) {
    if (nvshmem_debug_level <= NVSHMEM_LOG_NONE) {
        return;
    }

    char hostname[1024];
    nvshmemu_gethostname(hostname, 1024);
    int cudaDev = -1;
    CUDA_RUNTIME_CHECK(cudaGetDevice(&cudaDev));

    char buffer[1024];
    size_t len = 0;
    pthread_mutex_lock(&nvshmem_debug_output_lock);
    if (level == NVSHMEM_LOG_WARN && nvshmem_debug_level >= NVSHMEM_LOG_WARN)
        len = snprintf(buffer, sizeof(buffer), "\n%s:%d:%d [%d] %s:%d NVSHMEM WARN ", hostname,
                       getpid(), nvshmemi_gettid(), cudaDev, filefunc, line);
    else if (level == NVSHMEM_LOG_INFO && nvshmem_debug_level >= NVSHMEM_LOG_INFO &&
             (flags & nvshmem_debug_mask))
        len = snprintf(buffer, sizeof(buffer), "%s:%d:%d [%d] NVSHMEM INFO ", hostname, getpid(),
                       nvshmemi_gettid(), cudaDev);
#ifdef NVSHMEM_TRACE
    else if (level == NVSHMEM_LOG_TRACE && nvshmem_debug_level >= NVSHMEM_LOG_TRACE &&
             (flags & nvshmem_debug_mask)) {
        auto delta = std::chrono::high_resolution_clock::now() - nvshmem_epoch;
        double timestamp =
            std::chrono::duration_cast<std::chrono::duration<double>>(delta).count() * 1000;
        len = snprintf(buffer, sizeof(buffer), "%s:%d:%d [%d] %f %s:%d NVSHMEM TRACE ", hostname,
                       getpid(), nvshmemi_gettid(), cudaDev, timestamp, filefunc, line);
    }
#endif
    if (len) {
        va_list vargs;
        va_start(vargs, fmt);
        (void)vsnprintf(buffer + len, sizeof(buffer) - len, fmt, vargs);
        va_end(vargs);
        fprintf(nvshmem_debug_file, "%s\n", buffer);
        fflush(nvshmem_debug_file);
    }
    pthread_mutex_unlock(&nvshmem_debug_output_lock);

    // If nvshmem_debug_level == NVSHMEM_LOG_ABORT then WARN() will also call abort()
    if (level == NVSHMEM_LOG_WARN && nvshmem_debug_level == NVSHMEM_LOG_ABORT) {
        fprintf(stderr, "\n%s:%d:%d [%d] %s:%d NVSHMEM ABORT\n", hostname, getpid(),
                nvshmemi_gettid(), cudaDev, filefunc, line);
        abort();
    }
}
}
