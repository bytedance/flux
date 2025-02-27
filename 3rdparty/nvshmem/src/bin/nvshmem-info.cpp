/****
 * Copyright (c) 2016-2021, NVIDIA Corporation.  All rights reserved.
 *
 * See COPYRIGHT for license information
 ****/

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>  // IWYU pragma: keep
// IWYU pragma: no_include <bits/getopt_core.h>
#include "device_host_transport/nvshmem_constants.h"
#include "non_abi/nvshmem_version.h"
#include "internal/host/util.h"

int opt_env = 0;
int opt_env_hidden = 0;
int opt_env_rst = 0;
int opt_version = 0;
int opt_build = 0;

int main(int argc, char **argv) {
    int opt, ret = 0;

    while ((opt = getopt(argc, argv, "hadenrb")) != -1) {
        switch (opt) {
            case 'h':
                printf("Print information about NVSHMEM\n\n");
                printf("Usage: %s [options]\n\n", basename(argv[0]));
                printf("Options:\n");
                printf("    -h  This help message\n");
                printf("    -a  Print all output\n");
                printf("    -n  Print version number\n");
                printf("    -b  Print build information\n");
                printf("    -e  Print environment variables\n");
                printf("    -d  Include hidden environment variables in output\n");
                printf("    -r  RST format environment variable output\n");
                exit(0);
                break;
            case 'a':
                opt_env = 1;
                opt_env_hidden = 1;
                opt_version = 1;
                opt_build = 1;
                break;
            case 'n':
                opt_version = 1;
                break;
            case 'b':
                opt_build = 1;
                break;
            case 'e':
                opt_env = 1;
                break;
            case 'd':
                opt_env_hidden = 1;
                break;
            case 'r':
                opt_env_rst = 1;
                break;
            default:
                exit(1);
                break;
        }
    }

    if (opt_version) {
        printf("%s\n", NVSHMEM_VENDOR_STRING);
        printf("\n");
    }

    if (opt_build) {
        int driverVersion;
        cudaError_t err;

        printf("Build Information:\n");

        printf("  %-28s %d\n", "CUDA API", CUDA_VERSION);

        err = cudaDriverGetVersion(&driverVersion);
        if (err != cudaSuccess) driverVersion = -1;

        printf("  %-28s %d\n", "CUDA Driver", driverVersion);

        printf("  %-28s %s %s\n", "Build Timestamp", __DATE__, __TIME__);

        char *build_vars;

        build_vars = nvshmemu_wrap(NVSHMEM_BUILD_VARS, NVSHMEMI_WRAPLEN, "\t", 0);

        printf("  %-28s\n\t%s\n", "Build Variables",
               build_vars ? build_vars : "Error wrapping build vars");
        printf("\n");

        free(build_vars);
    }

    if (opt_env) {
        if (opt_env_hidden) {
            ret = setenv("NVSHMEM_INFO_HIDDEN", "1", 0);
            if (ret) {
                perror("Error setting NVSHMEM_INFO_HIDDEN");
                abort();
            }
        }

        ret = nvshmemi_options_init();

        if (ret) {
            printf("Error parsing environment\n");
            goto out;
        }

        if (opt_env_rst)
            nvshmemi_options_print(NVSHMEMI_OPTIONS_STYLE_RST);
        else
            nvshmemi_options_print(NVSHMEMI_OPTIONS_STYLE_INFO);
    }

out:
    return ret;
}
