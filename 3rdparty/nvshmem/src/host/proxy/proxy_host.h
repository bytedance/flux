/*
 * Copyright (c) 2016-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef PROXY_HOST_H
#define PROXY_HOST_H

#include <pthread.h>
#include <stdint.h>
#include "internal/host/nvshmem_internal.h"
#include "internal/host/nvshmemi_types.h"
#include "device_host/nvshmem_types.h"

#define CHANNEL_COUNT 1
#define COUNTER_TO_FLAG(state, counter) ((uint8_t)(!((counter >> state->channel_bufsize_log) & 1)))
#define WRAPPED_CHANNEL_BUF(state, ch, counter) (ch->buf + (counter & (state->channel_bufsize - 1)))

enum {
    PROXY_QUIET_STATUS_CHANNELS_INACTIVE = 0,
    PROXY_QUIET_STATUS_CHANNELS_IN_PROGRESS,
    PROXY_QUIET_STATUS_CHANNELS_DONE
};

enum { PROXY_CST_STATUS_CHANNELS_INACTIVE = 0, PROXY_CST_STATUS_CHANNELS_ACTIVE };

typedef struct proxy_channel {
    char *buf;
    uint64_t *issue;
    uint64_t *complete;
    uint64_t *quiet_issue;
    uint64_t *quiet_ack;
    uint64_t last_quiet_issue;
    uint64_t *cst_issue;
    uint64_t *cst_ack;
    uint64_t last_cst_issue;
    uint64_t processed;
    uint64_t last_sync;
} proxy_channel_t;

typedef struct {
    struct proxy_state *state;
    int stop;
} proxy_progress_params_t;

typedef struct proxy_state {
    int *transport_id;
    int transport_bitmap;
    struct nvshmem_transport **transport;
    int quiet_in_progress;
    int cst_in_progress;
    int quiet_ack_count;
    uint64_t channel_bufsize_log;
    uint64_t channel_bufsize;
    int channel_count;
    proxy_channel_t *channels;
    proxy_channel_t *channels_device;
    uint64_t channel_g_bufsize;
    int channel_in_progress;
    pthread_t progress_thread;
    proxy_progress_params_t progress_params;
    nvshmemi_state_t *nvshmemi_state;
    int *quiet_incoming_in_progress_pe;
    cudaStream_t stream;
    cudaStream_t queue_stream_out;
    cudaStream_t queue_stream_in;
    cudaEvent_t cuev;
    int finalize_count;
    int issued_get;
    nvshmemi_timeout_t *nvshmemi_timeout;
    bool is_consistency_api_supported;
    int gdr_device_native_ordering;
    int *global_exit_request_state;
    int *global_exit_code;
} proxy_state_t;

#endif
