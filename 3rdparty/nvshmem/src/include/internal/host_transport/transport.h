/*
 * Copyright (c) 2016-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef __TRANSPORT_H
#define __TRANSPORT_H

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>

/* This header, along with the six below, comprise
 * the ABI for transport modules.
 */
#include "bootstrap_host_transport/env_defs_internal.h"
#include "non_abi/nvshmem_version.h"
#include "device_host_transport/nvshmem_common_transport.h"
#include "non_abi/nvshmemx_error.h"
#include "non_abi/nvshmem_build_options.h"
#include "internal/host_transport/nvshmemi_transport_defines.h"
#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"

/* patch_version + minor_version * 100 + major_version * 10000 */
#define NVSHMEM_TRANSPORT_INTERFACE_VERSION           \
    (NVSHMEM_TRANSPORT_PLUGIN_MAJOR_VERSION * 10000 + \
     NVSHMEM_TRANSPORT_PLUGIN_MINOR_VERSION * 100 + NVSHMEM_TRANSPORT_PLUGIN_PATCH_VERSION)

#define NVSHMEM_TRANSPORT_MAJOR_VERSION(ver) (ver / 10000)
#define NVSHMEM_TRANSPORT_MAJOR_MINOR_VERSION(ver) (ver / 100)

enum {
    NVSHMEM_TRANSPORT_CAP_MAP = 1,
    NVSHMEM_TRANSPORT_CAP_MAP_GPU_ST = 1 << 1,
    NVSHMEM_TRANSPORT_CAP_MAP_GPU_LD = 1 << 2,
    NVSHMEM_TRANSPORT_CAP_MAP_GPU_ATOMICS = 1 << 3,
    NVSHMEM_TRANSPORT_CAP_CPU_WRITE = 1 << 4,
    NVSHMEM_TRANSPORT_CAP_CPU_READ = 1 << 5,
    NVSHMEM_TRANSPORT_CAP_CPU_ATOMICS = 1 << 6,
    NVSHMEM_TRANSPORT_CAP_GPU_WRITE = 1 << 7,
    NVSHMEM_TRANSPORT_CAP_GPU_READ = 1 << 8,
    NVSHMEM_TRANSPORT_CAP_GPU_ATOMICS = 1 << 9,
    NVSHMEM_TRANSPORT_CAP_MAX = INT_MAX
};

enum {
    TRANSPORT_OPTIONS_STYLE_INFO = 0,
    TRANSPORT_OPTIONS_STYLE_RST = 1,
    TRANSPORT_OPTIONS_STYLE_MAX = INT_MAX
};

enum {
    NVSHMEM_TRANSPORT_ATTR_NO_ENDPOINTS = 1,
    NVSHMEM_TRANSPORT_ATTR_CONNECTED = 1 << 1,
    NVSHMEM_TRANSPORT_ATTR_MAX = INT_MAX,
};

typedef enum {
    NVSHMEM_TRANSPORT_LIB_CODE_NONE = 0,
    NVSHMEM_TRANSPORT_LIB_CODE_IBGDA = 1,
    NVSHMEM_TRANSPORT_LIB_CODE_MAX = INT_MAX,
} nvshmem_transport_inline_lib_code_type_t;

typedef struct nvshmem_transport_pe_info {
    pcie_id_t pcie_id;
    int pe;
    uint64_t hostHash;
    cudaUUID_t gpu_uuid;
} nvshmem_transport_pe_info_t;

typedef struct rma_verb {
    nvshmemi_op_t desc;
    int is_nbi;
    int is_stream;
    cudaStream_t cstrm;
} rma_verb_t;

typedef struct rma_memdesc {
    void *ptr;
    uint64_t offset;
    nvshmem_mem_handle_t *handle;
} rma_memdesc_t;

typedef struct rma_bytesdesc {
    size_t nelems;
    int elembytes;
    ptrdiff_t srcstride;
    ptrdiff_t deststride;
} rma_bytesdesc_t;

typedef struct amo_verb {
    nvshmemi_amo_t desc;
    int is_fetch;
    int is_val;
    int is_cmp;
} amo_verb_t;

typedef struct amo_memdesc {
    rma_memdesc_t remote_memdesc;
    uint64_t retflag;
    void *retptr;
    void *valptr;
    void *cmpptr;
    uint64_t val;
    uint64_t cmp;
    nvshmem_mem_handle_t *ret_handle;
} amo_memdesc_t;

typedef struct amo_bytesdesc {
    int name_type;
    int elembytes;
} amo_bytesdesc_t;

typedef int (*rma_handle)(struct nvshmem_transport *tcurr, int pe, rma_verb_t verb,
                          rma_memdesc_t *remote, rma_memdesc_t *local, rma_bytesdesc_t bytesdesc,
                          int is_proxy);
typedef int (*amo_handle)(struct nvshmem_transport *tcurr, int pe, void *curetptr, amo_verb_t verb,
                          amo_memdesc_t *target, amo_bytesdesc_t bytesdesc, int is_proxy);
typedef int (*fence_handle)(struct nvshmem_transport *tcurr, int pe, int is_proxy);
typedef int (*quiet_handle)(struct nvshmem_transport *tcurr, int pe, int is_proxy);

struct nvshmem_transport_host_ops {
    int (*can_reach_peer)(int *access, nvshmem_transport_pe_info_t *peer_info,
                          struct nvshmem_transport *transport);
    int (*connect_endpoints)(struct nvshmem_transport *tcurr, int *selected_dev_ids,
                             int num_selected_devs);
    int (*get_mem_handle)(nvshmem_mem_handle_t *mem_handle, nvshmem_mem_handle_t *mem_handle_in,
                          void *buf, size_t size, struct nvshmem_transport *transport,
                          bool local_only);
    int (*release_mem_handle)(nvshmem_mem_handle_t *mem_handle,
                              struct nvshmem_transport *transport);
    int (*finalize)(struct nvshmem_transport *transport);
    int (*show_info)(struct nvshmem_transport *transport, int style);
    int (*progress)(struct nvshmem_transport *transport);

    rma_handle rma;
    amo_handle amo;
    fence_handle fence;
    quiet_handle quiet;
    int (*enforce_cst)(struct nvshmem_transport *transport);
    int (*enforce_cst_at_target)(struct nvshmem_transport *transport);
    int (*add_device_remote_mem_handles)(struct nvshmem_transport *transport, int transport_stride,
                                         nvshmem_mem_handle_t *mem_handles, uint64_t heap_offset,
                                         size_t size);
};

typedef struct nvshmem_transport {
    /* lib identifiers */
    int api_version;
    nvshmem_transport_inline_lib_code_type_t type;
    int *cap;
    /* APIs */
    struct nvshmem_transport_host_ops host_ops;
    /* Handles to bootstrap and internal state */
    bootstrap_handle_t *boot_handle;
    void *state;
    void *type_specific_shared_state;
    void *cache_handle;
    /* transport shares to lib */
    char **device_pci_paths;
    int attr;
    int n_devices;
    bool atomics_complete_on_quiet;
    bool is_successfully_initialized;
    bool no_proxy;
    /* lib shares to transport */
    void *heap_base;
    size_t log2_cumem_granularity;
    uint64_t max_op_len;
    uint32_t atomic_host_endian_min_size;
    int index;
    int my_pe;
    int n_pes;
} nvshmem_transport_v1;

typedef nvshmem_transport_v1 *nvshmem_transport_t;

int nvshmemt_p2p_init(nvshmem_transport_t *transport);

typedef int (*nvshmemi_transport_init_fn)(nvshmem_transport_t *transport,
                                          struct nvshmemi_cuda_fn_table *table, int api_version);

#endif
