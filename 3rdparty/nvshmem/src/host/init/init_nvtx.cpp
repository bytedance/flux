/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "internal/host/nvshmem_nvtx.hpp"  // for nvtxOpt_t, ALLOC_OPT
// IWYU pragma: no_include <nvtx3/nvtxDetail/nvtxImplCore.h>

#ifndef NVTX_DISABLE
#include <stdio.h>                                       // for snprintf, NULL, printf
#include <stdlib.h>                                      // for free
#include <string.h>                                      // for strtok, strdup
#include <syscall.h>                                     // for SYS_gettid
#include <unistd.h>                                      // for syscall
#include "internal/host/debug.h"                         // for strcmp_case_insensi...
#include "internal/host/util.h"                          // for nvshmemi_options
#include "bootstrap_host_transport/env_defs_internal.h"  // for nvshmemi_options_s

int nvshmem_nvtx_options = 0;  // NVTX instrumentation is disabled by default

// mapping of environment variable options to internal ID
static const struct nvtxOptMap {
    nvtxOpt_t opt;
    const char *name;
    const char *desc;
} options[] = {
    {INIT_OPT, "init", "library setup"},
    {ALLOC_OPT, "alloc", "memory management"},
    {LAUNCH_OPT, "launch", "kernel launch routines"},
    {COLL_OPT, "coll", "collective communications"},
    {WAIT_OPT, "wait", "blocking point-to-point synchronization"},
    {WAIT_ON_STREAM_OPT, "wait_on_stream", "point-to-point synchronization (on stream)"},
    {TEST_OPT, "test", "non-blocking point-to-point synchronization"},
    {MEMORDER_OPT, "memorder", "memory ordering (quiet, fence)"},
    {QUIET_ON_STREAM_OPT, "quiet_on_stream", "nvshmemx_quiet_on_stream"},
    {ATOMIC_FETCH_OPT, "atomic_fetch", "fetching atomic memory operations"},
    {ATOMIC_SET_OPT, "atomic_set", "non-fetchong atomic memory operations"},
    {RMA_BLOCKING_OPT, "rma_blocking", "blocking remote memory access operations"},
    {RMA_NONBLOCKING_OPT, "rma_nonblocking", "non-blocking remote memory access operations"},
    {PROXY_OPT, "proxy", "activity of the proxy thread"},
    {DEFAULT_OPT, "common", "init,alloc,launch,coll,memorder,wait,atomic_fetch,rma_blocking,proxy"},
    {ALL_OPT, "all", "all groups"},
    {(nvtxOpt_t)0, "off", "disable all NVTX instrumentation"}};
static const int num_options = sizeof(options) / sizeof(options[0]);
static const int num_options_enable = num_options - 1;

void nvshmem_nvtx_set_thread_name(int pe, const char *suffix) {
    char name[32];
    if (suffix) {
        snprintf(name, sizeof(name), "NVSHMEM PE %d %s", pe, suffix);
    } else {
        snprintf(name, sizeof(name), "NVSHMEM PE %d", pe);
    }
    nvtxNameOsThreadA(syscall(SYS_gettid), name);
}

static bool initialized = false;
void nvshmem_nvtx_init() {
    if (initialized) {
        return;
    }
    initialized = true;

    // evaluate what should be instrumented from environment variable
    if (nvshmemi_options.NVTX_provided) {
        char *groups = strdup(nvshmemi_options.NVTX);
        char seps[] = ",";

        // iterate over all options from the environment variable
        char *gname = strtok(groups, seps);
        while (gname != NULL) {
            // handle enable options
            for (int i = 0; i < num_options_enable; ++i) {
                if (strcmp_case_insensitive(gname, options[i].name) == 0) {
                    nvshmem_nvtx_options |= options[i].opt;
                }
            }

            // handle disable options ('0' or last entry of options)
            if (strcmp_case_insensitive(gname, "0") == 0 ||
                strcmp_case_insensitive(gname, options[num_options - 1].name) == 0) {
                nvshmem_nvtx_options = 0;
            }

            gname = strtok(NULL, seps);
        }
        if (groups != NULL) {
            free(groups);
        }
    }
}

void nvshmem_nvtx_print_options() {
    for (int i = 0; i < num_options; ++i) {
        printf("\t  %-20s: %s\n", options[i].name, options[i].desc);
    }
}

#else /* !NVTX_DISABLE */

#include "internal/host/util.h"
void nvshmem_nvtx_init() {
    if (nvshmemi_options.NVTX_provided &&
        strcmp_case_insensitive(nvshmemi_options.NVTX, "off") != 0 &&
        strcmp(nvshmemi_options.NVTX, "") != 0) {
        printf(
            "NVTX instrumentation is not available. Check your NVSHMEM "
            "build. NVTX support requires C++11 or newer.\n");
    }
}

#endif /* !NVTX_DISABLE */
