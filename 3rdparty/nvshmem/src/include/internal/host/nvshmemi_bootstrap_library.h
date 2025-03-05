/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef NVSHMEMI_BOOTSTRAP_LIBRARY_H
#define NVSHMEMI_BOOTSTRAP_LIBRARY_H

#include <cstddef>
#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"

enum { BOOTSTRAP_MPI = 0, BOOTSTRAP_SHMEM, BOOTSTRAP_PMI, BOOTSTRAP_PLUGIN, BOOTSTRAP_UID };

typedef struct bootstrap_attr {
    bootstrap_attr() : initialize_shmem(0), mpi_comm(NULL), uid_args(NULL) {}
    int initialize_shmem;
    void *mpi_comm;
    void *meta_data;
    void *uid_args;
} bootstrap_attr_t;

int bootstrap_set_bootattr(int flags, void *nvshmem_attr, bootstrap_attr_t *boot_attr);
int bootstrap_preinit(int flags, bootstrap_handle_t *handle);
int bootstrap_init(int flags, bootstrap_attr_t *attr, bootstrap_handle_t *handle);
void bootstrap_finalize();

int bootstrap_loader_preinit(const char *plugin, bootstrap_handle_t *handle);
int bootstrap_loader_init(const char *plugin, void *arg, bootstrap_handle_t *handle);
int bootstrap_loader_finalize(bootstrap_handle_t *handle);

#endif
