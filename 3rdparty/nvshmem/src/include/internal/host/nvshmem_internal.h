/*
 * Copyright (c) 2016-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _INTERNAL_H
#define _INTERNAL_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <atomic>
#include <pthread.h>
#include <vector>
#include <map>

#include "device_host/nvshmem_common.cuh"
#include "internal/host/custom_malloc.h"
#include "internal/host/nvshmemi_symmetric_heap.hpp"
#include "internal/host/nvshmemi_types.h"
#include "non_abi/nvshmemx_error.h"
#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"
#include "internal/host_transport/cudawrap.h"
#include "internal/host_transport/transport.h"
#include "non_abi/nvshmem_build_options.h"

#define NVSHMEMI_LOCAL_BUF_CACHE_DEFAULT_SIZE 64

/* This is a requirement imposed by DMA-BUF which only supports 32-bit registrations */
#define NVSHMEMI_DMA_BUF_MAX_LENGTH 0x100000000ULL
#define NVSHMEMI_MAX_HANDLE_LENGTH 2147483648ULL

#define MAX_PEER_STREAMS 3

#define MAX_TRANSPORT_EP_COUNT 1

#define NUM_G_BUF_ELEMENTS 1024 * 1024

#define G_COALESCING_BUF_SIZE NUM_G_BUF_ELEMENTS *NVSHMEMI_WARP_SIZE * sizeof(uint64_t)
#define NVSHMEMI_CHECK_INIT_STATUS()                                                 \
    do {                                                                             \
        if (nvshmemi_device_state.nvshmemi_is_nvshmem_initialized == false)          \
            NVSHMEMI_ERROR_EXIT(                                                     \
                "NVSHMEM API called before NVSHMEM initialization has completed\n"); \
    } while (0)

#define NVSHMEMI_IPC_CHECK(ipcFuncResult)               \
    do {                                                \
        if (ipcFuncResult == -1) {                      \
            NVSHMEMI_ERROR_EXIT("Fatal IPC Failure\n"); \
        }                                               \
    } while (0)

#define NVSHMEMI_TRANSPORT_IS_CAP(transport, cap_idx, flag) ((transport)->cap[(cap_idx)] & flag)
#define NVSHMEMI_TRANSPORT_OPS_IS_GET_MEM(transport) ((transport)->host_ops.get_mem_handle != NULL)
#define NVSHMEMI_TRANSPORT_OPS_IS_RELEASE_MEM(transport) \
    ((transport)->host_ops.release_mem_handle != NULL)
#define NVSHMEMI_TRANSPORT_OPS_IS_ADD_DEVICE_REMOTE_MEM(transport) \
    ((transport)->host_ops.add_device_remote_mem_handles != NULL)

enum { NVSHMEMI_HEAP_KIND_VIDMEM = 0, NVSHMEMI_HEAP_KIND_SYSMEM };

typedef struct nvshmem_local_buf_handle {
    void *ptr;
    size_t length;
    nvshmem_mem_handle_t *handle;
    bool registered_by_us;
    bool linked_with_prev;
    bool linked_with_next;
} nvshmem_local_buf_handle_t;

typedef struct nvshmem_local_buf_cache {
    size_t array_size;
    size_t array_used;
    nvshmem_local_buf_handle_t **buffers;
    pthread_rwlock_t buffer_lock;
} nvshmem_local_buf_cache_t;

void nvshmemi_transport_buffer_unregister_all(nvshmem_transport_t transport);
int nvshmemi_local_mem_cache_init(nvshmem_local_buf_cache_t **cache);
void nvshmemi_local_mem_cache_fini(nvshmem_local_buf_cache_t *cache);

typedef struct {
    int error_checks;
} nvshmem_options_t;

typedef struct nvshmemi_session {
    void *bootstrap;
    /* volatile uint32_t *abort_flag; This feature needs CUDA support in nvshmem */
} nvshmemi_session_t;

extern struct nvshmemi_cuda_fn_table *nvshmemi_cuda_syms;
extern int nvshmemi_can_use_cuda_64_bit_stream_memops;
extern int nvshmemi_can_flush_remote_writes;
extern uint64_t *nvshmemi_host_hashes;
extern nvshmem_options_t nvshmem_options;
extern int nvshmemi_cuda_driver_version;
extern int nvshmemi_use_nccl;
extern bool nvshmemi_is_mps_available;
extern int nccl_version;
extern long nvshmemi_max_teams;
extern nvshmemi_session_t *nvshmemi_default_session;
extern bool nvshmemi_is_mpg_run;
extern bool nvshmemi_is_limited_mpg_run;

int nvshmemi_proxy_level(nvshmemi_state_t *state);
int nvshmemi_common_init(nvshmemi_state_t *state);
int nvshmemi_init_g_buffer();
void nvshmemi_init_symmetric_heap(nvshmemi_state_t *state, bool is_vmm, int heap_kind);
void nvshmemi_fini_symmetric_heap(nvshmemi_state_t *state);
int nvshmemi_init_device_state(nvshmemi_state_t *state);
int nvshmemi_setup_connections(nvshmemi_state_t *state);
int nvshmemi_setup_mops_kernels(nvshmemi_state_t *state);
void nvshmemi_signal_op_on_stream(uint64_t *sig_addr, uint64_t signal, int sig_op, int pe,
                                  cudaStream_t cstrm);
extern "C" {
__device__ void nvshmemi_signal_op(uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
void nvshmemi_get_mem_handle(void **dev_state_ptr, void **transport_dev_state_ptr);
}

void nvshmemi_barrier_all();
int nvshmemi_proxy_init(nvshmemi_state_t *state, int proxy_level);
int nvshmemi_proxy_finalize(nvshmemi_state_t *state);

struct nvshmem_mem_handle *nvshmemi_get_registered_buffer_handle(nvshmem_transport_t transport,
                                                                 void *addr, size_t *len);

static inline void nvshmemi_get_local_mem_handle(nvshmem_mem_handle_t **handle, size_t *len,
                                                 void *addr, int transport_idx) {
    nvshmem_transport_t transport = nvshmemi_state->transports[transport_idx];
    size_t max_len = transport->max_op_len;

    *handle = nvshmemi_state->heap_obj->get_transport_mem_handle(addr, len, nvshmemi_state->mype,
                                                                 transport_idx);
    if (*handle == NULL) {
        /* registered buffer lookup code */
        *handle = nvshmemi_get_registered_buffer_handle(transport, addr, len);
    }

    if (len) *len = *len < max_len ? *len : max_len;
    assert(*handle != NULL);
}

static inline void nvshmemi_get_remote_mem_handle(rma_memdesc_t *handle, size_t *len, void *addr,
                                                  int pe, int transport_idx) {
    nvshmem_transport_t transport = nvshmemi_state->transports[transport_idx];
    size_t max_len = transport->max_op_len;

    handle->handle =
        nvshmemi_state->heap_obj->get_transport_mem_handle(addr, len, pe, transport_idx);
    handle->offset = nvshmemi_state->heap_obj->get_mem_handle_addr_offset(addr);
    if (len) *len = *len < max_len ? *len : max_len;
    assert(handle->handle != NULL);
}
/* rptr is symmetric address on the local pe
   lptr is local address - either symmetric or not */
static inline void nvshmemi_process_multisend_rma(struct nvshmem_transport *tcurr, int transport_id,
                                                  int pe, rma_verb_t verb, void *rptr, void *lptr,
                                                  size_t size, bool is_proxy) {
    rma_memdesc_t localdesc, remotedesc;
    rma_bytesdesc_t bytes;
    bytes.srcstride = 1;
    bytes.deststride = 1;
    bytes.elembytes = 1;
    size_t local_chunk_size, remote_chunk_size, size_remaining;
    size_t chunk_size;
    size_remaining = size;
    int status;

    while (size_remaining) {
        localdesc.ptr = lptr;
        NVSHMEMU_UNMAPPED_PTR_PE_TRANSLATE(remotedesc.ptr, rptr, pe);
        remotedesc.offset = (char *)rptr - (char *)nvshmemi_device_state.heap_base;
        local_chunk_size = size_remaining;
        remote_chunk_size = size_remaining;
        nvshmemi_get_local_mem_handle(&localdesc.handle, &local_chunk_size, lptr, transport_id);
        nvshmemi_get_remote_mem_handle(&remotedesc, &remote_chunk_size, rptr, pe, transport_id);
        chunk_size = std::min(local_chunk_size, std::min(remote_chunk_size, size_remaining));
        bytes.nelems = chunk_size;
        status = tcurr->host_ops.rma(tcurr, pe, verb, &remotedesc, &localdesc, bytes, is_proxy);
        if (unlikely(status)) {
            NVSHMEMI_ERROR_PRINT("aborting due to error in process_channel_dma\n");
            exit(-1);
        }
        size_remaining -= chunk_size;
        lptr = (char *)lptr + chunk_size;
        rptr = (char *)rptr + chunk_size;
    }
}

int nvshmemi_update_device_state();

int nvshmemi_cuda_library_init(struct nvshmemi_cuda_fn_table *table);

int nvshmemi_get_pcie_attrs(pcie_id_t *pcie_id, CUdevice cudev);

void nvshmemi_add_transport(int id, int (*init_op)(nvshmem_transport_t *));
int nvshmemi_transport_init(struct nvshmemi_state_dec *state);
int nvshmemi_transport_finalize(struct nvshmemi_state_dec *state);

#ifdef __cplusplus
extern "C" {
#endif
void *nvshmemi_malloc(size_t size);
void nvshmemi_free(void *ptr);
#ifdef __cplusplus
}
#endif

#endif
