/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _UCX_H
#define _UCX_H

#include <stdint.h>  // IWYU pragma: keep
// IWYU pragma: no_include <bits/stdint-uintn.h>
#include <stddef.h>
#include <ucp/api/ucp_def.h>

#include "non_abi/nvshmem_build_options.h"
#include "device_host_transport/nvshmem_common_transport.h"

#ifdef NVSHMEM_USE_GDRCOPY
#include "gdrapi.h"
#endif

/* This value is arbitrary. UCX doesn't give a max length for packed rkeys. */
#define NVSHMEMT_UCP_RKEY_PACKED_MAX_LEN 256

#define NVSHMEMT_UCP_ADDR_MAX_LEN 1024

#define NVSHMEMT_UCX_ATOMIC_POOL_SIZE (1 << 14)

#define NVSHMEMT_UCX_ATOMIC_POOL_MASK (NVSHMEMT_UCX_ATOMIC_POOL_SIZE - 1)

#define NVSHMEMT_UCX_BOUNCE_BUFFER_POOL_SIZE (1 << 22)

#define NVSHMEMT_UCX_BOUNCE_BUFFER_POOL_MASK (NVSHMEMT_UCX_BOUNCE_BUFFER_POOL_SIZE - 1)

typedef enum {
    NVSHMEMT_UCX_ATOMIC_SEND,
    NVSHMEMT_UCX_ATOMIC_RESP,
} nvshmemt_ucx_am_op_t;

typedef struct {
    ucp_ep_h ep;
    size_t op_size;
    uint64_t value;
    uint64_t cmp;
    uint64_t retflag;
    void *addr;
    void *retptr;
    nvshmemi_amo_t op;
} nvshmemt_ucx_am_send_header_t;

typedef struct {
    void *retptr;
    uint64_t retval;
    uint64_t retflag;
} nvshmemt_ucx_am_resp_header_t;

typedef struct {
    union {
        nvshmemt_ucx_am_resp_header_t resp_h;
        nvshmemt_ucx_am_send_header_t send_h;
    } header;
    bool is_proxy;
    bool in_use;
    bool nvshmem_owned;
    bool dynamic_alloc;
} nvshmemt_ucx_am_header_t;

typedef struct {
    char addr[NVSHMEMT_UCP_ADDR_MAX_LEN];
    int addr_len;
} ucx_ep_handle_t;

typedef struct {
    char rkey_packed_buf[NVSHMEMT_UCP_RKEY_PACKED_MAX_LEN];
    ucp_mem_h mem_handle;
    ucp_rkey_h ep_rkey_host;
    ucp_rkey_h ep_rkey_proxy;
    size_t rkey_packed_buf_len;
    void *ptr;
    bool local_only;
} nvshmemt_ucx_mem_handle_t;

typedef struct {
    void *ptr;
    size_t size;
    nvshmemt_ucx_mem_handle_t *mem_handle;
#ifdef NVSHMEM_USE_GDRCOPY
    gdr_mh_t mh;
    void *cpu_ptr;
    void *cpu_ptr_base;
#endif
} nvshmemt_ucx_mem_handle_info_t;

typedef struct {
    uint64_t value;
    uint64_t retvalue;
    uint64_t amo_retflag;
    void *amo_device_retptr;
    void *ucx_state;
    void *transport;
    bool in_use;
    bool is_proxy;
    bool amo_has_retval;
} nvshmemt_ucx_bounce_buffer_t;

typedef struct transport_ucx_state_t {
    ucp_config_t *library_config;
    struct transport_mem_handle_info_cache *cache;
    ucp_context_h library_context;
    ucp_worker_h worker_context;
    ucp_ep_h *endpoints;
    ucp_rkey_h *ep_rkeys;
    ucp_mem_h bounce_buffer_mem_handle;
    int ep_count;
    int proxy_ep_idx;
    int num_headers_requested;
    int log_level;
    uint16_t num_bounce_buffers_requested;
    nvshmemt_ucx_am_header_t send_headers[NVSHMEMT_UCX_ATOMIC_POOL_SIZE];
    nvshmemt_ucx_am_header_t recv_headers[NVSHMEMT_UCX_ATOMIC_POOL_SIZE];
    nvshmemt_ucx_bounce_buffer_t bounce_buffers[NVSHMEMT_UCX_BOUNCE_BUFFER_POOL_SIZE];
} transport_ucx_state_t;

struct ibv_device **(*get_device_list)(int *num_devices);
void (*free_device_list)(struct ibv_device **device_list);

#endif /* _UCX_H */
