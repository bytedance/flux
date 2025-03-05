/*
 * Copyright (c) 2016-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */
#ifndef NVSHMEMI_BOOTSTRAP_DEFINES_H
#define NVSHMEMI_BOOTSTRAP_DEFINES_H

#include <limits.h>

typedef struct bootstrap_init_ops {
    void *cookie;
    int (*get_unique_id)(void *cookie);
} bootstrap_init_ops_t;

enum {
    BOOTSTRAP_OPTIONS_STYLE_INFO = 0,
    BOOTSTRAP_OPTIONS_STYLE_RST = 1,
    BOOTSTRAP_OPTIONS_STYLE_MAX = INT_MAX
};

typedef struct bootstrap_handle {
    int version;
    int pg_rank;
    int pg_size;
    int mype_node;
    int npes_node;
    int (*allgather)(const void *sendbuf, void *recvbuf, int bytes,
                     struct bootstrap_handle *handle);
    int (*alltoall)(const void *sendbuf, void *recvbuf, int bytes, struct bootstrap_handle *handle);
    int (*barrier)(struct bootstrap_handle *handle);
    void (*global_exit)(int status);
    int (*finalize)(struct bootstrap_handle *handle);
    int (*show_info)(struct bootstrap_handle *handle, int style);
    bootstrap_init_ops_t *pre_init_ops;
    void *comm_state;
} bootstrap_handle_v1;

typedef bootstrap_handle_v1 bootstrap_handle_t;

#endif
