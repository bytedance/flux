/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT.txt for license information
 */

#include <stdio.h>
#include <unistd.h>
#include "nvshmem.h"

int main(int argc, char **argv) {
    char hostname[256];

    int ret = gethostname(hostname, 256);
    if (ret < 0) {
        printf("Failed to get hostname\n");
        return 1;
    }

    printf("[%s][%ld] Starting up...\n", hostname, (long)getpid());

    nvshmem_init();

    int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    cudaSetDevice(mype_node);
    void *ptr = nvshmem_malloc(1);  // initialize NVSHMEM after device is set

    printf("[%s][%ld] Hello from PE %d of %d\n", hostname, (long)getpid(), nvshmem_my_pe(),
           nvshmem_n_pes());

    nvshmem_finalize();
    return 0;
}
