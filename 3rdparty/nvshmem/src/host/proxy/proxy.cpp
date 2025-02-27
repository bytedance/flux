/*
 * Copyright (c) 2016-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */
#include <assert.h>                        // for assert
#include <cuda.h>                          // for CUDA_SUCCESS
#include <cuda_runtime.h>                  // for cudaFreeHost
#include <driver_types.h>                  // for cudaStreamNon...
#include <inttypes.h>                      // for PRIu64
#include <math.h>                          // for log2
#include <pthread.h>                       // for pthread_create
#include <stdint.h>                        // for uint64_t, uin...
#include <stdio.h>                         // for fprintf, NULL
#include <stdlib.h>                        // for exit, free
#include <string.h>                        // for memset, memcpy
#include <unistd.h>                        // IWYU pragma: keep for getpid in NVSHMEM_TRACE case
#include "device_host/nvshmem_types.h"     // for nvshmemi_devi...
#include "device_host/nvshmem_common.cuh"  // for nvshmemi_devi...
#include "device_host_transport/nvshmem_constants.h"                       // for CHANNEL_BUF_S...
#include "host/nvshmem_api.h"                                              // for nvshmem_globa...
#include "non_abi/nvshmemx_error.h"                                        // for NVSHMEMI_ERRO...
#include "non_abi/nvshmem_build_options.h"                                 // IWYU pragma: keep
#include "device_host_transport/nvshmem_common_transport.h"                // for g_elem_t, NVS...
#include "internal/host/debug.h"                                           // for TRACE, INFO
#include "internal/host/nvshmem_internal.h"                                // for nvshmemi_cuda...
#include "internal/host/nvshmemi_symmetric_heap.hpp"                       // for nvshmemi_symm...
#include "internal/host/nvshmemi_types.h"                                  // for nvshmemi_state_t
#include "internal/host/nvtx3.hpp"                                         // for message
#include "internal/host/util.h"                                            // for CUDA_RUNTIME_...
#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"  // for nvshmemi_boot...
#include "internal/host_transport/cudawrap.h"                              // for CUPFN, nvshme...
#include "bootstrap_host_transport/env_defs_internal.h"                    // for nvshmemi_opti...
#include "internal/host_transport/transport.h"                             // for nvshmem_trans...
#include "proxy_host.h"                                                    // for proxy_state_t
#include "device_host/nvshmem_proxy_channel.h"

// use a different NVTX domain ("NVSHMEM_PROXY") for proxy activities
#define NVSHMEM_NVTX_DOMAIN NVSHMEM_PROXY
#include "internal/host/nvshmem_nvtx.hpp"  // for nvshmem_nvtx_...
// IWYU pragma: no_include "nvtx3.hpp"

uint64_t proxy_channel_g_buf_size;     /* Total size of g_buf in bytes */
uint64_t proxy_channel_g_buf_log_size; /* Total size of g_buf in bytes */

char *proxy_channel_g_buf;
char *proxy_channel_g_coalescing_buf;

// progress channels
static base_request_t **channel_req;

void *nvshmemi_proxy_progress(void *in);
void *nvshmemi_proxy_progress_minimal(void *in);

int nvshmemi_proxy_prep_minimal_state(proxy_state_t *state) {
    int *temp_global_exit_request_state;
    int *temp_global_exit_code;
    nvshmemi_timeout_t *nvshmemi_timeout_dptr;

    nvshmemi_device_state.global_exit_request_state = state->global_exit_request_state;

    CUDA_RUNTIME_CHECK(cudaHostGetDevicePointer(&temp_global_exit_request_state,
                                                state->global_exit_request_state, 0));
    CUDA_RUNTIME_CHECK(
        cudaHostGetDevicePointer(&temp_global_exit_code, state->global_exit_code, 0));
    CUDA_RUNTIME_CHECK(
        cudaHostGetDevicePointer(&nvshmemi_timeout_dptr, state->nvshmemi_timeout, 0));

    nvshmemi_device_state.global_exit_request_state = temp_global_exit_request_state;
    nvshmemi_device_state.global_exit_code = temp_global_exit_code;
    nvshmemi_device_state.timeout = nvshmemi_timeout_dptr;

    return 0;
}

int nvshmemi_proxy_setup_device_channels(proxy_state_t *state) {
    int status = 0;

    nvshmemi_device_state.proxy_channel_buf_size = state->channel_bufsize;
    nvshmemi_device_state.proxy_channel_buf_logsize = state->channel_bufsize_log;
    CUDA_RUNTIME_CHECK(
        cudaMalloc(&state->channels_device, sizeof(proxy_channel_t) * state->channel_count));
    INFO(NVSHMEM_PROXY, "channel buf: %p complete: %p quiet_issue: %p quiet_ack: %p",
         state->channels[0].buf, state->channels[0].complete, state->channels[0].quiet_issue,
         state->channels[0].quiet_ack);

    uint64_t *temp_buf_dptr;
    uint64_t *temp_complete_dptr;
    uint64_t *temp_quiet_issue_dptr;
    uint64_t *temp_quiet_ack_dptr;
    uint64_t *temp_cst_issue_dptr;
    uint64_t *temp_cst_ack_dptr;

    CUDA_RUNTIME_CHECK(cudaHostGetDevicePointer(&temp_buf_dptr, state->channels[0].buf, 0));
    CUDA_RUNTIME_CHECK(
        cudaHostGetDevicePointer(&temp_complete_dptr, state->channels[0].complete, 0));
    CUDA_RUNTIME_CHECK(
        cudaHostGetDevicePointer(&temp_quiet_issue_dptr, state->channels[0].quiet_issue, 0));
    CUDA_RUNTIME_CHECK(
        cudaHostGetDevicePointer(&temp_quiet_ack_dptr, state->channels[0].quiet_ack, 0));
    CUDA_RUNTIME_CHECK(
        cudaHostGetDevicePointer(&temp_cst_issue_dptr, state->channels[0].cst_issue, 0));
    CUDA_RUNTIME_CHECK(cudaHostGetDevicePointer(&temp_cst_ack_dptr, state->channels[0].cst_ack, 0));

    INFO(NVSHMEM_PROXY,
         "channel device_ptr buf: %p issue: %p complete: %p quiet_issue: %p quiet_ack: %p \n",
         temp_buf_dptr, state->channels[0].issue, temp_complete_dptr, temp_quiet_issue_dptr,
         temp_quiet_ack_dptr);

    nvshmemi_device_state.proxy_channels_buf = temp_buf_dptr;
    nvshmemi_device_state.proxy_channels_issue = state->channels[0].issue;
    nvshmemi_device_state.proxy_channels_complete = temp_complete_dptr;
    nvshmemi_device_state.proxy_channels_quiet_issue = temp_quiet_issue_dptr;
    nvshmemi_device_state.proxy_channels_quiet_ack = temp_quiet_ack_dptr;
    nvshmemi_device_state.proxy_channels_cst_issue = temp_cst_issue_dptr;
    nvshmemi_device_state.proxy_channels_cst_ack = temp_cst_ack_dptr;

    proxy_channel_g_buf_size = NUM_G_BUF_ELEMENTS * sizeof(g_elem_t);
    proxy_channel_g_buf_log_size = (uint64_t)log2((double)proxy_channel_g_buf_size);
    uint64_t *proxy_channel_g_buf_head_ptr;
    CUDA_RUNTIME_CHECK(cudaMalloc((void **)&proxy_channel_g_buf_head_ptr, sizeof(uint64_t)));
    CUDA_RUNTIME_CHECK(cudaMemset((void *)proxy_channel_g_buf_head_ptr, 0, sizeof(uint64_t)));

    uint64_t *proxy_channels_complete_local_ptr;
    CUDA_RUNTIME_CHECK(cudaMalloc((void **)&proxy_channels_complete_local_ptr, sizeof(uint64_t)));
    CUDA_RUNTIME_CHECK(cudaMemset((void *)proxy_channels_complete_local_ptr, 0, sizeof(uint64_t)));

    proxy_channel_g_buf = (char *)nvshmemi_malloc(proxy_channel_g_buf_size);
    NVSHMEMI_NULL_ERROR_JMP(proxy_channel_g_buf, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "failed allocating proxy_channel_g_buf");
    proxy_channel_g_coalescing_buf = (char *)nvshmemi_malloc(G_COALESCING_BUF_SIZE);
    NVSHMEMI_NULL_ERROR_JMP(proxy_channel_g_coalescing_buf, status, NVSHMEMX_ERROR_OUT_OF_MEMORY,
                            out, "failed allocating proxy_channel_g_coalescing_buf");

    nvshmemi_device_state.proxy_channel_g_buf_size = proxy_channel_g_buf_size;
    nvshmemi_device_state.proxy_channel_g_buf_log_size = proxy_channel_g_buf_log_size;
    nvshmemi_device_state.proxy_channel_g_buf_head_ptr = proxy_channel_g_buf_head_ptr;
    nvshmemi_device_state.proxy_channels_complete_local_ptr = proxy_channels_complete_local_ptr;
    nvshmemi_device_state.proxy_channel_g_buf = proxy_channel_g_buf;
    nvshmemi_device_state.proxy_channel_g_coalescing_buf = proxy_channel_g_coalescing_buf;
    assert(proxy_channel_g_buf_size % sizeof(g_elem_t) == 0);

out:
    return status;
}

inline void proxy_update_processed(proxy_channel_t *ch, int bytes) {
    ch->processed += bytes;

    if ((ch->processed - ch->last_sync) >= 1024) {
        *ch->complete = ch->processed;
        ch->last_sync = ch->processed;
        TRACE(NVSHMEM_PROXY, "updated processed to device %llu", ch->processed);
    }
}

int nvshmemi_proxy_create_channels(proxy_state_t *proxy_state) {
    int status = 0;

    proxy_channel_t *channels =
        (proxy_channel_t *)malloc(sizeof(proxy_channel_t) * proxy_state->channel_count);
    NVSHMEMI_NULL_ERROR_JMP(channels, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "failed allocating channels");
    memset(channels, 0, sizeof(proxy_channel_t) * proxy_state->channel_count);

    for (int i = 0; i < proxy_state->channel_count; i++) {
        // for put/get
        CUDA_RUNTIME_CHECK(cudaMallocHost((void **)&channels[i].buf, proxy_state->channel_bufsize,
                                          0)); /* CPU reads, GPU writes */
        memset(channels[i].buf, 0, proxy_state->channel_bufsize);

        CUDA_RUNTIME_CHECK(cudaMalloc(
            &channels[i].issue, sizeof(uint64_t))); /* issue is not accessed through LD/ST by CPU
                                                       thread, therefore on device memory */
        CUDA_RUNTIME_CHECK(cudaMemset(channels[i].issue, 0, sizeof(uint64_t)));

        CUDA_RUNTIME_CHECK(cudaMallocHost((void **)&channels[i].complete, sizeof(uint64_t),
                                          0)); /* CPU writes, GPU reads */
        CUDA_RUNTIME_CHECK(cudaMallocHost((void **)&channels[i].quiet_issue, sizeof(uint64_t),
                                          0)); /* CPU reads, GPU writes */
        CUDA_RUNTIME_CHECK(cudaMallocHost((void **)&channels[i].quiet_ack, sizeof(uint64_t),
                                          0)); /* CPU writes, GPU reads */
        CUDA_RUNTIME_CHECK(cudaMallocHost((void **)&channels[i].cst_issue, sizeof(uint64_t),
                                          0)); /* CPU reads, GPU writes */
        CUDA_RUNTIME_CHECK(cudaMallocHost((void **)&channels[i].cst_ack, sizeof(uint64_t),
                                          0)); /* CPU writes, GPU reads */

        *channels[i].complete = 0;
        *channels[i].quiet_issue = 0;
        *channels[i].quiet_ack = 0;
        channels[i].last_quiet_issue = 0;
        *channels[i].cst_issue = 0;
        *channels[i].cst_ack = 0;
        channels[i].last_cst_issue = 0;
    }

    proxy_state->channels = channels;

out:
    return status;
}

int nvshmemi_proxy_setup_connections(proxy_state_t *proxy_state) {
    int status;
    nvshmemi_state_t *state = proxy_state->nvshmemi_state;
    struct nvshmem_transport **transport = NULL;
    int *transport_id;

    proxy_state->transport_bitmap = 0;
    transport = proxy_state->transport =
        (struct nvshmem_transport **)calloc(state->npes, sizeof(struct nvshmem_transport *));
    NVSHMEMI_NULL_ERROR_JMP(transport, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "failed allocating space for transports \n");

    transport_id = proxy_state->transport_id = (int *)calloc(state->npes, sizeof(int));
    NVSHMEMI_NULL_ERROR_JMP(transport_id, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "failed allocating space for transport id \n");

    for (int j = 0; j < state->npes; j++) {
        for (int i = 0; i < state->num_initialized_transports; i++) {
            int transport_bit = (1 << i);
            // assumes symmetry of transport list at all PEs
            if (!((state->transport_bitmap) & transport_bit)) continue;
            struct nvshmem_transport *tcurr = state->transports[i];

            // finding the first transport with CPU WRITE capability
            if (!(tcurr->cap[j] &
                  (NVSHMEM_TRANSPORT_CAP_CPU_WRITE | NVSHMEM_TRANSPORT_CAP_CPU_READ)))
                continue;

            // assuming the transport is connected - IB RC
            assert(tcurr->attr & NVSHMEM_TRANSPORT_ATTR_CONNECTED);

            transport[j] = tcurr;
            transport_id[j] = i;
            proxy_state->transport_bitmap |= transport_bit;

            break;
        }
    }

    status = nvshmemi_boot_handle.barrier(&nvshmemi_boot_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "barrier failed \n");

out:
    if (status) {
        if (transport) free(transport);
    }
    return status;
}

int nvshmemi_proxy_init(nvshmemi_state_t *state, int proxy_level) {
    int status = 0;
    CUdevice device;

    if (proxy_level == NVSHMEMI_PROXY_NONE) {
        INFO(NVSHMEM_INIT,
             "Proxy is disabled. Device side wait_until timeouts and global exit will not function."
             "If this is undesired behavior, Please unset NVSHMEM_DISABLE_LOCAL_ONLY_PROXY, or set "
             "it to false.\n");
        return 0;
    }

    INFO(NVSHMEM_PROXY, "[%d] in proxy_init", state->mype);
    nvshmemu_debug_log_cpuset(NVSHMEM_PROXY, "proxy");

    proxy_state_t *proxy_state = (proxy_state_t *)calloc(1, sizeof(proxy_state_t));

    proxy_state->nvshmemi_state = state;

    CUDA_RUNTIME_CHECK(
        cudaMallocHost((void **)&proxy_state->global_exit_request_state, sizeof(int), 0));
    CUDA_RUNTIME_CHECK(cudaMallocHost((void **)&proxy_state->global_exit_code, sizeof(int), 0));
    CUDA_RUNTIME_CHECK(cudaMallocHost((void **)&proxy_state->nvshmemi_timeout,
                                      sizeof(nvshmemi_timeout_t), 0)); /* GPU writes, CPU reads */
    (*proxy_state->nvshmemi_timeout) = NVSHMEMI_TIMEOUT_INITIALIZER;
    status = nvshmemi_proxy_prep_minimal_state(proxy_state);
    if (status) {
        fprintf(stderr, "global exit context creation failed. \n");
        exit(-1);
    }
    /* Set here in case we are in an NVLink only build and don't call
     * nvshmemi_proxy_setup_device_channels*/
    nvshmemi_update_device_state();

    // create a minimal proxy thread if we only need global_exit support.
    if (proxy_level == NVSHMEMI_PROXY_MINIMAL) {
        proxy_state->progress_params.state = proxy_state;
        proxy_state->progress_params.stop = 0;
        proxy_state->finalize_count = 0;
        proxy_state->quiet_in_progress = PROXY_QUIET_STATUS_CHANNELS_INACTIVE;
        proxy_state->cst_in_progress = PROXY_CST_STATUS_CHANNELS_INACTIVE;
        proxy_state->issued_get = 0;
        status =
            pthread_create(&proxy_state->progress_thread, NULL, nvshmemi_proxy_progress_minimal,
                           (void *)&proxy_state->progress_params);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "pthread creation failed \n");
        state->proxy = (void *)proxy_state;
        goto out;
    }

    proxy_state->channel_bufsize_log = CHANNEL_BUF_SIZE_LOG;
    proxy_state->channel_bufsize = (1 << CHANNEL_BUF_SIZE_LOG);
    proxy_state->channel_count = CHANNEL_COUNT;
    status = nvshmemi_proxy_create_channels(proxy_state);
    if (status) {
        fprintf(stderr, "channel creation failed \n");
        exit(-1);
    }

    status = nvshmemi_proxy_setup_device_channels(proxy_state);
    if (status) {
        fprintf(stderr, "channel creation failed \n");
        exit(-1);
    }
    nvshmemi_update_device_state();

    status = nvshmemi_proxy_setup_connections(proxy_state);
    if (status) {
        fprintf(stderr, "connection setup failed \n");
        exit(-1);
    }

    INFO(NVSHMEM_PROXY, "[%d] after setting up proxy channels on device", state->mype);

    CUDA_RUNTIME_CHECK(cudaStreamCreateWithFlags(&proxy_state->stream, cudaStreamNonBlocking));
    CUDA_RUNTIME_CHECK(
        cudaStreamCreateWithFlags(&proxy_state->queue_stream_out, cudaStreamNonBlocking));
    CUDA_RUNTIME_CHECK(
        cudaStreamCreateWithFlags(&proxy_state->queue_stream_in, cudaStreamNonBlocking));
    CUDA_RUNTIME_CHECK(cudaEventCreateWithFlags(&proxy_state->cuev, cudaEventDefault));

    proxy_state->progress_params.state = proxy_state;
    proxy_state->progress_params.stop = 0;
    proxy_state->finalize_count = 0;
    proxy_state->quiet_in_progress = PROXY_QUIET_STATUS_CHANNELS_INACTIVE;
    proxy_state->cst_in_progress = PROXY_CST_STATUS_CHANNELS_INACTIVE;
    proxy_state->issued_get = 0;

    status = CUPFN(nvshmemi_cuda_syms, cuCtxGetDevice)(&device);
    if (status != CUDA_SUCCESS) {
        fprintf(stderr, "cuCtxGetDevice failed \n");
        exit(-1);
    }

    int write_options;
    status = CUPFN(nvshmemi_cuda_syms, cuDeviceGetAttribute)(
        &write_options,
        (CUdevice_attribute)CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS, device);
    if (status != CUDA_SUCCESS) {
        proxy_state->is_consistency_api_supported = false;
        cudaGetLastError();
        goto post_cst_api_check;
    }
    if (write_options & (CUdevice_attribute)CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST)
        proxy_state->is_consistency_api_supported = true;
    status = CUPFN(nvshmemi_cuda_syms, cuDeviceGetAttribute)(
        &proxy_state->gdr_device_native_ordering,
        (CUdevice_attribute)CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING, device);
    if (status != CUDA_SUCCESS) {
        proxy_state->gdr_device_native_ordering = 0;
        cudaGetLastError();
    }

post_cst_api_check:
    INFO(NVSHMEM_PROXY, "[%d] creating proxy thread", state->mype);

    status = pthread_create(&proxy_state->progress_thread, NULL, nvshmemi_proxy_progress,
                            (void *)&proxy_state->progress_params);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "pthread creation failed \n");

    state->proxy = (void *)proxy_state;

out:
    if (status != 0) {
        exit(-1);
    }
    return status;
}

inline int process_channel_dma(proxy_state_t *state, proxy_channel_t *ch, int *is_processed) {
    int status = 0;
    base_request_t *base_req;
    put_dma_request_0_t *dma_req_0;
    put_dma_request_1_t *dma_req_1;
    put_dma_request_2_t *dma_req_2;
    int pe;
    size_t size;
    uint8_t flag;
    uint64_t roffset, laddr;

    base_req = (base_request_t *)WRAPPED_CHANNEL_BUF(state, ch, ch->processed);
    roffset = (uint64_t)(((uint64_t)(base_req->roffset_high) << 8) | (base_req->roffset_low));

    dma_req_0 = (put_dma_request_0_t *)WRAPPED_CHANNEL_BUF(state, ch, (ch->processed + 8));
    flag = COUNTER_TO_FLAG(state, (ch->processed + 8));
    while (*((volatile uint8_t *)&dma_req_0->flag) != flag)
        ;

    dma_req_1 = (put_dma_request_1_t *)(put_dma_request_0_t *)WRAPPED_CHANNEL_BUF(
        state, ch, (ch->processed + 16));
    flag = COUNTER_TO_FLAG(state, (ch->processed + 16));
    while (*((volatile uint8_t *)&dma_req_1->flag) != flag)
        ;

    dma_req_2 = (put_dma_request_2_t *)(put_dma_request_0_t *)WRAPPED_CHANNEL_BUF(
        state, ch, (ch->processed + 24));
    flag = COUNTER_TO_FLAG(state, (ch->processed + 24));
    while (*((volatile uint8_t *)&dma_req_2->flag) != flag)
        ;

#if defined(NVSHMEM_PPC64LE) || defined(NVSHMEM_AARCH64)
    __sync_synchronize();  // XXX : prevents load from buf_d reordered to before load from issue_d
                           // (breaks rma)
#elif defined(NVSHMEM_X86_64)
    asm volatile("" : : : "memory");
#endif
    laddr = (uint64_t)(((uint64_t)(dma_req_0->laddr_high) << 32) |
                       ((uint64_t)(dma_req_0->laddr_3) << 16) |
                       ((uint64_t)(dma_req_0->laddr_2) << 8) | dma_req_1->laddr_low);
    size = (size_t)(((size_t)(dma_req_1->size_high) << 16) | (dma_req_1->size_low));
    pe = dma_req_2->pe;
    TRACE(NVSHMEM_PROXY, "process_channel_dma laddr %p pe %d", laddr, pe);

    // issue transport DMA
    {
        rma_verb_t verb;
        verb.desc = (nvshmemi_op_t)base_req->op;
        verb.is_nbi = 1;
        verb.is_stream = 0;
        verb.cstrm = NULL;
        void *rptr = (void *)((char *)(nvshmemi_device_state.heap_base) + roffset);
        nvshmemi_process_multisend_rma(state->transport[pe], state->transport_id[pe], pe, verb,
                                       rptr, (void *)laddr, size, 1);
    }
#if defined(NVSHMEM_PPC64LE) || defined(NVSHMEM_AARCH64)
    __sync_synchronize();  // XXX: prevents complete_d store reordered to before return from
                           // ibv_post_send (breaks rma -> quiet)
#endif

    *is_processed = 1;

    proxy_update_processed(ch, PROXY_DMA_REQ_BYTES);
    TRACE(NVSHMEM_PROXY,
          "[%d] process_channel_put_dma/proxy_update_processed processed %ld complete %ld",
          state->nvshmemi_state->mype, ch->processed, *ch->complete);

    return status;
}

inline int process_channel_inline(proxy_state_t *state, proxy_channel_t *ch, int *is_processed) {
    int status = 0;
    base_request_t *base_req;
    put_inline_request_0_t *inline_req_0;
    put_inline_request_1_t *inline_req_1;
    uint8_t flag;
    uint64_t roffset;
    nvshmemi_state_t *nvshmemi_state = state->nvshmemi_state;

    base_req = (base_request_t *)WRAPPED_CHANNEL_BUF(state, ch, ch->processed);
    roffset = (uint64_t)(((uint64_t)(base_req->roffset_high) << 8) | (base_req->roffset_low));

    inline_req_0 = (put_inline_request_0_t *)WRAPPED_CHANNEL_BUF(state, ch, (ch->processed + 8));
    flag = COUNTER_TO_FLAG(state, (ch->processed + 8));
    while (*((volatile uint8_t *)&inline_req_0->flag) != flag)
        ;

    inline_req_1 = (put_inline_request_1_t *)WRAPPED_CHANNEL_BUF(state, ch, (ch->processed + 16));
    flag = COUNTER_TO_FLAG(state, (ch->processed + 16));
    while (*((volatile uint8_t *)&inline_req_1->flag) != flag)
        ;

#if defined(NVSHMEM_PPC64LE) || defined(NVSHMEM_AARCH64)
    __sync_synchronize();  // XXX : prevents load from buf_d reordered to before load from issue_d
                           // (was present in dma function, was missing in inline function, breaks
                           // rma)
#elif defined(NVSHMEM_X86_64)
    asm volatile("" : : : "memory");
#endif

    uint32_t pe = inline_req_0->pe;
    uint64_t size = inline_req_1->size;
    uint64_t lvalue;

    lvalue = inline_req_0->lvalue_low;
    if (size == 8) {
        lvalue = lvalue | ((uint64_t)(inline_req_1->lvalue_high) << 32);
    }

    // issue transport DMA
    {
        rma_memdesc_t localdesc, remotedesc;
        rma_bytesdesc_t bytes;
        rma_verb_t verb;
        void *remote = (void *)((char *)(nvshmemi_device_state.heap_base) + roffset);
        void *remote_actual;
        NVSHMEMU_UNMAPPED_PTR_PE_TRANSLATE(remote_actual, remote, pe);
        void *local = (void *)&lvalue;
        struct nvshmem_transport *tcurr = state->transport[pe];
        int t = state->transport_id[pe];

        verb.desc = NVSHMEMI_OP_P;
        verb.is_nbi = 0;

        localdesc.ptr = local;
        localdesc.handle = NULL;
        remotedesc.ptr = remote_actual;
        nvshmemi_get_remote_mem_handle(&remotedesc, NULL, remote, pe, t);

        bytes.nelems = 1;
        bytes.elembytes = size;

        status = tcurr->host_ops.rma(tcurr, pe, verb, &remotedesc, &localdesc, bytes, 1);
        if (unlikely(status)) {
            NVSHMEMI_ERROR_PRINT("aborting due to error in process_channel_dma\n");
            exit(-1);
        }
    }
#if defined(NVSHMEM_PPC64LE) || defined(NVSHMEM_AARCH64)
    __sync_synchronize();  // XXX: prevents complete_d store reordered to before return from
                           // ibv_post_cq (breaks rma -> quiet)
#endif

    *is_processed = 1;

    proxy_update_processed(ch, PROXY_INLINE_REQ_BYTES);
    TRACE(NVSHMEM_PROXY,
          "[%d] process_channel_put_dma/proxy_update_processed processed %ld complete %ld",
          state->nvshmemi_state->mype, ch->processed, *ch->complete);

    return status;
}

int process_channel_amo(proxy_state_t *state, proxy_channel_t *ch, int *is_processed) {
    int status = 0;
    base_request_t *base_req;
    amo_request_0_t *req_0;
    amo_request_1_t *req_1;
    amo_request_2_t *req_2;
    uint8_t flag;
    uint64_t roffset;

    base_req = (base_request_t *)WRAPPED_CHANNEL_BUF(state, ch, ch->processed);
    roffset = (uint64_t)(((uint64_t)(base_req->roffset_high) << 8) | (base_req->roffset_low));

    req_0 = (amo_request_0_t *)WRAPPED_CHANNEL_BUF(state, ch, (ch->processed + 8));
    flag = COUNTER_TO_FLAG(state, (ch->processed + 8));
    while (*((volatile uint8_t *)&req_0->flag) != flag)
        ;

    req_1 = (amo_request_1_t *)WRAPPED_CHANNEL_BUF(state, ch, (ch->processed + 16));
    flag = COUNTER_TO_FLAG(state, (ch->processed + 16));
    while (*((volatile uint8_t *)&req_1->flag) != flag)
        ;

    req_2 = (amo_request_2_t *)WRAPPED_CHANNEL_BUF(state, ch, (ch->processed + 24));
    flag = COUNTER_TO_FLAG(state, (ch->processed + 24));
    while (*((volatile uint8_t *)&req_2->flag) != flag)
        ;

    amo_request_3_t *req_3;
    req_3 = (amo_request_3_t *)WRAPPED_CHANNEL_BUF(state, ch, (ch->processed + 32));
    flag = COUNTER_TO_FLAG(state, (ch->processed + 32));
    while (*((volatile uint8_t *)&req_3->flag) != flag)
        ;

#if defined(NVSHMEM_PPC64LE) || defined(NVSHMEM_AARCH64)
    __sync_synchronize();  // XXX : prevents load from buf_d reordered to before load from issue_d
                           // (was present in dma function, was missing in inline function, breaks
                           // rma)
#elif defined(_NVSHMEM_X86_64)
    asm volatile("" : : : "memory");
#endif

    uint32_t pe = req_0->pe;
    uint64_t size = req_1->size;
    nvshmemi_amo_t amo_op = (nvshmemi_amo_t)req_0->amo;
    uint64_t lvalue, cvalue = 0;

    lvalue = req_0->swap_add_low;
    lvalue = lvalue | ((uint64_t)req_1->swap_add_high << 32);

    if (amo_op == NVSHMEMI_AMO_COMPARE_SWAP) {
        /* This is safe because they are both 64 bit values. */
        cvalue = ((*reinterpret_cast<uint64_t *>(req_2)) & 0xFFFFFFFFFFFFFF00u);
        cvalue |= req_1->compare_low;
    }

    // issue transport amo
    {
        amo_verb_t verb;
        amo_bytesdesc_t bytes;
        amo_memdesc_t memdesc;
        void *remote = (void *)((char *)(nvshmemi_device_state.heap_base) + roffset);
        void *remote_actual =
            (void *)((char *)(nvshmemi_state->heap_obj->get_remote_pe_base()[pe]) + roffset);
        int t = state->transport_id[pe];
        struct nvshmem_transport *tcurr = state->transport[pe];

        verb.desc = amo_op;

        memset(&memdesc, 0, sizeof(amo_memdesc_t));
        memdesc.remote_memdesc.ptr = remote_actual;
        memdesc.remote_memdesc.offset = roffset;
        memdesc.val = lvalue;
        memdesc.cmp = cvalue;
        nvshmemi_get_remote_mem_handle(&memdesc.remote_memdesc, NULL, remote, pe, t);
        // pick spot in g buffer for fetch value
        if ((amo_op > NVSHMEMI_AMO_END_OF_NONFETCH)) {
            uint64_t g_buf_counter = ((*reinterpret_cast<uint64_t *>(req_3)) & 0xFFFFFFFFFFFFFF00u);
            g_buf_counter >>= 8;
            uint64_t offset = ((g_buf_counter * sizeof(g_elem_t)) & (proxy_channel_g_buf_size - 1));
            memdesc.retptr = (void *)(proxy_channel_g_buf + offset);
            memdesc.retflag =
                ((g_buf_counter * sizeof(g_elem_t)) >> proxy_channel_g_buf_log_size) * 2 + 1;
            nvshmemi_get_local_mem_handle(&memdesc.ret_handle, NULL, memdesc.retptr, t);
        }
        bytes.elembytes = size;

        status = tcurr->host_ops.amo(tcurr, pe, NULL, verb, &memdesc, bytes, 1);
        if (unlikely(status)) {
            NVSHMEMI_ERROR_PRINT("aborting due to error in process_channel_dma\n");
            exit(-1);
        }
    }

#if defined(NVSHMEM_PPC64LE) || defined(NVSHMEM_AARCH64)
    __sync_synchronize();  // XXX: prevents complete_d store reordered to before return from
                           // ibv_post_cq (breaks rma -> quiet)
#endif

    *is_processed = 1;

    proxy_update_processed(ch, PROXY_AMO_REQ_BYTES);
    INFO(NVSHMEM_PROXY,
         "[%d] process_channel_put_dma/proxy_update_processed processed %ld complete %ld \n",
         state->nvshmemi_state->mype, ch->processed, *ch->complete);
    /* Fetching atomics that complete on quiet need a consistency op to confirm proper ordering on
     * the device side. */
    if (amo_op > NVSHMEMI_AMO_END_OF_NONFETCH && nvshmemi_device_state.atomics_complete_on_quiet) {
        state->issued_get = 1;
    }

    return status;
}

void enforce_cst(proxy_state_t *proxy_state) {
#if defined(NVSHMEM_X86_64)
    nvshmemi_state_t *state = proxy_state->nvshmemi_state;
#endif

    int status = 0;
    if (nvshmemi_options.BYPASS_FLUSH) return;

    if (proxy_state->is_consistency_api_supported) {
        if (CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER > proxy_state->gdr_device_native_ordering &&
            CUPFN(nvshmemi_cuda_syms, cuFlushGPUDirectRDMAWrites)) {
            status =
                CUPFN(nvshmemi_cuda_syms,
                      cuFlushGPUDirectRDMAWrites(CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX,
                                                 CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER));
            /** We would want to use cudaFlushGPUDirectRDMAWritesToAllDevices when we enable
                consistent access of data on any GPU (and not just self GPU) with
               wait_until, quiet, barrier, etc. **/
        }
        return;
    }
#if defined(NVSHMEM_PPC64LE)
    status = cudaEventRecord(proxy_state->cuev, proxy_state->stream);
    if (unlikely(status != CUDA_SUCCESS)) {
        NVSHMEMI_ERROR_EXIT("cuEventRecord() failed in the proxy thread \n");
    }
#elif defined(NVSHMEM_X86_64)
    for (int i = 0; i < state->num_initialized_transports; i++) {
        if (!((state->transport_bitmap) & (1 << i))) continue;
        struct nvshmem_transport *tcurr = state->transports[i];
        if (!tcurr->host_ops.enforce_cst) continue;

        // assuming the transport is connected - IB RC
        if (tcurr->attr & NVSHMEM_TRANSPORT_ATTR_CONNECTED) {
            status = tcurr->host_ops.enforce_cst(tcurr);
            if (status) {
                NVSHMEMI_ERROR_PRINT("aborting due to error in progress_cst \n");
                exit(-1);
            }
        }
    }
#endif
}

inline void quiet_ack_channels(proxy_state_t *proxy_state) {
    for (int i = 0; i < proxy_state->channel_count; i++) {
        proxy_channel_t *ch = (proxy_state->channels + i);
        *((volatile uint64_t *)ch->quiet_ack) = ch->last_quiet_issue;
        TRACE(NVSHMEM_PROXY, "[%d] quiet_ack_channels quiet_ack %ld",
              proxy_state->nvshmemi_state->mype, *ch->quiet_ack);
    }
}

inline int quiet_channels_check(proxy_state_t *proxy_state) {
    int start_quiet = 0;

    for (int i = 0; i < proxy_state->channel_count; i++) {
        proxy_channel_t *ch = (proxy_state->channels + i);
        if (*((volatile uint64_t *)ch->quiet_issue) > ch->last_quiet_issue) {
            ch->last_quiet_issue = *((volatile uint64_t *)ch->quiet_issue);
            start_quiet = 1;
            TRACE(NVSHMEM_PROXY, "[%d] host proxy: received quiet on channel %d from GPU", getpid(),
                  i);
        }
    }

    return start_quiet;
}

inline int quiet_channels_test(proxy_state_t *proxy_state) {
    int processed = 1;

    for (int i = 0; i < proxy_state->channel_count; i++) {
        proxy_channel_t *ch = (proxy_state->channels + i);
        if (ch->processed < ch->last_quiet_issue) {
            TRACE(NVSHMEM_PROXY, "[%d] quiet_channels_test last_quiet_issue %ld processed %ld",
                  proxy_state->nvshmemi_state->mype, ch->last_quiet_issue, ch->processed);
            processed = 0;
        } else {
            TRACE(NVSHMEM_PROXY,
                  "processing quiet for channel %d from GPU "
                  "ch->processed: %llu ch->last_quiet_issue: %llu",
                  i, ch->processed, ch->last_quiet_issue);
        }
    }

    return processed;
}

inline void progress_global_exit(proxy_state_t *proxy_state) {
    if (*(volatile int *)proxy_state->global_exit_request_state == PROXY_GLOBAL_EXIT_REQUESTED) {
        nvshmem_global_exit(*proxy_state->global_exit_code);
    }
}

inline void progress_quiet(proxy_state_t *proxy_state) {
    // quiet processing at source
    if (proxy_state->quiet_in_progress == PROXY_QUIET_STATUS_CHANNELS_INACTIVE) {
        if (quiet_channels_check(proxy_state)) {
            proxy_state->quiet_in_progress = PROXY_QUIET_STATUS_CHANNELS_IN_PROGRESS;
            TRACE(NVSHMEM_PROXY, "[%d] quiet_progress PROXY_QUIET_STATUS_CHANNELS_IN_PROGRESS",
                  proxy_state->nvshmemi_state->mype);
        }
    }

    if (proxy_state->quiet_in_progress == PROXY_QUIET_STATUS_CHANNELS_IN_PROGRESS) {
        if (quiet_channels_test(proxy_state)) {
            proxy_state->quiet_in_progress = PROXY_QUIET_STATUS_CHANNELS_DONE;
            TRACE(NVSHMEM_PROXY, "[%d] quiet_progress PROXY_QUIET_STATUS_CHANNELS_DONE",
                  proxy_state->nvshmemi_state->mype);
        }
    }

    if (proxy_state->quiet_in_progress == PROXY_QUIET_STATUS_CHANNELS_DONE) {
        nvshmemi_state_t *state = proxy_state->nvshmemi_state;
        NVTX_SCOPE_IN_GROUP(PROXY, nvshmem_proxy_quiet);

        // issue quiet on connections to all peers, we might want to make transport level quiet a
        // non-blocking call
        for (int i = 0; i < state->npes; i++) {
            struct nvshmem_transport *tcurr;
            int status = 0;

            tcurr = proxy_state->transport[i];
            if (tcurr == NULL) continue;
            status = tcurr->host_ops.quiet(tcurr, i, 1);
            if (unlikely(status)) {
                NVSHMEMI_ERROR_PRINT("aborting due to error in progress_quiet \n");
                exit(-1);
            }
        }
#if defined(NVSHMEM_PPC64LE) || defined(NVSHMEM_AARCH64)
        __sync_synchronize();  // XXX: prevents quiet_ack_d store reordered to before return from
                               // ibv_poll_cq
#endif

        if (proxy_state->issued_get) {
            enforce_cst(proxy_state);
            proxy_state->issued_get = 0;
        }

        quiet_ack_channels(proxy_state);
        proxy_state->quiet_in_progress = PROXY_QUIET_STATUS_CHANNELS_INACTIVE;
    }
}

inline void cst_ack_channels(proxy_state_t *proxy_state) {
    for (int i = 0; i < proxy_state->channel_count; i++) {
        proxy_channel_t *ch = (proxy_state->channels + i);
        *((volatile uint64_t *)ch->cst_ack) = ch->last_cst_issue;
        TRACE(NVSHMEM_PROXY, "[%d] cst_ack_channels cst_ack %ld", proxy_state->nvshmemi_state->mype,
              *ch->cst_ack);
    }
}

inline int cst_channels_check(proxy_state_t *proxy_state) {
    int start_cst = 0;

    for (int i = 0; i < proxy_state->channel_count; i++) {
        proxy_channel_t *ch = (proxy_state->channels + i);
        if (*((volatile uint64_t *)ch->cst_issue) > ch->last_cst_issue) {
            ch->last_cst_issue = *((volatile uint64_t *)ch->cst_issue);
            start_cst = 1;
            TRACE(NVSHMEM_PROXY, "[%d] host proxy: received cst on channel %d from GPU %ld",
                  proxy_state->nvshmemi_state->mype, i, *ch->cst_issue);
        }
    }

    return start_cst;
}

inline void progress_cst(proxy_state_t *proxy_state) {
    if (proxy_state->cst_in_progress == PROXY_CST_STATUS_CHANNELS_INACTIVE) {
        if (cst_channels_check(proxy_state)) {
            proxy_state->cst_in_progress = PROXY_CST_STATUS_CHANNELS_ACTIVE;
            TRACE(NVSHMEM_PROXY, "[%d] cst_progress PROXY_CST_STATUS_CHANNELS_IN_PROGRESS",
                  proxy_state->nvshmemi_state->mype);
        }
    }

    if (proxy_state->cst_in_progress == PROXY_CST_STATUS_CHANNELS_ACTIVE) {
        enforce_cst(proxy_state);
#if defined(NVSHMEM_PPC64LE) || defined(NVSHMEM_AARCH64)
        __sync_synchronize();  // XXX: prevents cst_ack_d store reordered to before return from
                               // cuEventRecord
#endif
        cst_ack_channels(proxy_state);
        proxy_state->cst_in_progress = PROXY_CST_STATUS_CHANNELS_INACTIVE;
    }
}

inline int process_channel_fence(proxy_state_t *proxy_state, proxy_channel_t *ch) {
    int status = 0;
    nvshmemi_state_t *state = proxy_state->nvshmemi_state;

    for (int i = 0; i < state->npes; i++) {
        struct nvshmem_transport *tcurr;

        if (i == state->mype) continue;

        tcurr = proxy_state->transport[i];

        if (tcurr->host_ops.fence) status = tcurr->host_ops.fence(tcurr, i, 1);
        if (unlikely(status)) {
            NVSHMEMI_ERROR_PRINT("aborting due to error in process_fence \n");
            exit(-1);
        }
    }

    proxy_update_processed(ch, CHANNEL_ENTRY_BYTES);
    return 0;
}

inline void copy_from_channel(proxy_state_t *state, proxy_channel_t *ch, void *dest) {
    channel_bounce_buffer_t bounce;
    int counter, num_bytes = 0;
    void *channel_ptr;
    volatile char *channel_ptr_char;
    volatile uint64_t *channel_ptr_uint64;
    char *dest_ptr;
    int flag;

    dest_ptr = (char *)dest;
    counter = ch->processed;

    do {
        flag = COUNTER_TO_FLAG(state, ch->processed);
        channel_ptr = (void *)WRAPPED_CHANNEL_BUF(state, ch, counter);
        channel_ptr_char = (volatile char *)channel_ptr;
        channel_ptr_uint64 = (volatile uint64_t *)channel_ptr;
        while ((channel_ptr_char[0] & 1) != flag)
            ;
        bounce.whole_buffer = *channel_ptr_uint64;
        memcpy(dest_ptr, &bounce.bytes[1], 7);

        dest_ptr += 7;

        /* each channel buffer is 8 bytes. */
        num_bytes += 8;
        counter += 8;
        /* Note, the second to last bit being set denotes a continuation of the same request from
         * the other side. */
    } while ((channel_ptr_char[0] & 0x10) == 0x10);

    proxy_update_processed(ch, num_bytes);
}

inline void progress_channels(proxy_state_t *proxy_state) {
    int status = 0;

    for (int i = 0; i < proxy_state->channel_count; i++) {
        proxy_channel_t *ch = (proxy_state->channels + i);

        uint64_t counter;
        int flag;
        uint8_t flag_value;

        for (int j = 0; j < nvshmemi_options.PROXY_REQUEST_BATCH_MAX; j++) {
            counter = ch->processed;

            if (likely(channel_req[i] == NULL)) {
                flag = COUNTER_TO_FLAG(proxy_state, counter);
                channel_req[i] = (base_request_t *)WRAPPED_CHANNEL_BUF(proxy_state, ch, counter);
                /* edit out continuation bit. */
                flag_value = (uint8_t)(*((volatile uint8_t *)&channel_req[i]->flag) & 1);
                if (flag_value != flag) {
                    channel_req[i] =
                        NULL;  // XXX:this store should prevent the next load to be reordered, so
                               // fence should not be needed; but fence below unhangs barrier test
                } else {
                    TRACE(NVSHMEM_PROXY,
                          "[%d] progress_channels found new channeL_req %p counter %ld",
                          proxy_state->nvshmemi_state->mype, channel_req[i], counter);
                }
            }

#if defined(NVSHMEM_PPC64LE) || defined(NVSHMEM_AARCH64)
            __sync_synchronize();  // XXX: this makes a difference for barrier but not for get_nbi;
                                   // this is Load/Load ordering point, fence could be needed for
                                   // x86_64 (if data dependency is not enough)
#endif
            // NOTE: all process function except process_channel_dma either processes
            // the complete request of does not process it at all
            if (channel_req[i]) {
                TRACE(NVSHMEM_PROXY,
                      "[%d] progress_channels new request channel_req %p counter %ld",
                      proxy_state->nvshmemi_state->mype, channel_req[i], counter);
                int is_processed = 1;
                switch (channel_req[i]->op) {
                    case NVSHMEMI_OP_PUT:
                        TRACE(NVSHMEM_PROXY, "host proxy: received PUT \n");
                        is_processed = 0;
                        status = process_channel_dma(proxy_state, ch, &is_processed);
                        NVSHMEMI_NZ_EXIT(status, "error in process_channel_dma<PUT>\n");
                        break;
                    case NVSHMEMI_OP_G:
                    case NVSHMEMI_OP_GET:
                        TRACE(NVSHMEM_PROXY, "host proxy: received GET \n");
                        is_processed = 0;
                        status = process_channel_dma(proxy_state, ch, &is_processed);
                        if (likely(is_processed)) proxy_state->issued_get = 1;
                        NVSHMEMI_NZ_EXIT(status, "error in process_channel_dma<GET>\n");
                        break;
                    case NVSHMEMI_OP_P:
                        TRACE(NVSHMEM_PROXY, "host proxy: received P_CHAR \n");
                        is_processed = 0;
                        status = process_channel_inline(proxy_state, ch, &is_processed);
                        NVSHMEMI_NZ_EXIT(status, "error in process_channel_inline<char>\n");
                        break;
                    case NVSHMEMI_OP_AMO:
                        is_processed = 0;
                        status = process_channel_amo(proxy_state, ch, &is_processed);
                        NVSHMEMI_NZ_EXIT(status, "error in process_channel_inline<char>\n");
                        break;
                    case NVSHMEMI_OP_FENCE:
                        TRACE(NVSHMEM_PROXY, "host proxy: received FENCE \n");
                        status = process_channel_fence(proxy_state, ch);
                        NVSHMEMI_NZ_EXIT(status, "error in process_channel_fence\n");
                        break;
                    default:
                        fprintf(stderr, "invalid op type encountered in proxy \n");
                        exit(-1);
                }

                if (likely(is_processed)) {
                    channel_req[i] = NULL;
                } else {
                    // request is only partially processed, use the same request in the next
                    // iteration
                }
            } /*if(channel_req[i])*/ else {
                break;
            }
        }
    }
}

void progress_timeout_polling(proxy_state_t *proxy_state) {
    nvshmemi_timeout_t *timeout = proxy_state->nvshmemi_timeout;

    if (timeout->signal) {
        const char *str = "";
        switch (timeout->caller) {
            case NVSHMEMI_CALL_SITE_BARRIER:
                str = "nvshmem_barrier";
                break;
            case NVSHMEMI_CALL_SITE_BARRIER_WARP:
                str = "nvshmem_barrier_warp";
                break;
            case NVSHMEMI_CALL_SITE_BARRIER_THREADBLOCK:
                str = "nvshmem_barrier_block";
                break;
            case NVSHMEMI_CALL_SITE_WAIT_UNTIL_GE:
                str = "nvshmem_wait_until_ge";
                break;
            case NVSHMEMI_CALL_SITE_WAIT_UNTIL_EQ:
                str = "nvshmem_wait_until_eq";
                break;
            case NVSHMEMI_CALL_SITE_WAIT_UNTIL_NE:
                str = "nvshmem_wait_until_ne";
                break;
            case NVSHMEMI_CALL_SITE_WAIT_UNTIL_GT:
                str = "nvshmem_wait_until_gt";
                break;
            case NVSHMEMI_CALL_SITE_WAIT_UNTIL_LT:
                str = "nvshmem_wait_until_lt";
                break;
            case NVSHMEMI_CALL_SITE_WAIT_UNTIL_LE:
                str = "nvshmem_wait_until_le";
                break;
            case NVSHMEMI_CALL_SITE_WAIT_NE:
                str = "nvshmem_wait_ne";
                break;
            case NVSHMEMI_CALL_SITE_PROXY_CHECK_CHANNEL_AVAILABILITY:
                str = "check_channel_availability";
                break;
            case NVSHMEMI_CALL_SITE_PROXY_QUIET:
                str = "nvshmemi_proxy_quiet";
                break;
            case NVSHMEMI_CALL_SITE_PROXY_ENFORCE_CONSISTENCY_AT_TARGET:
                str = "nvshmemi_proxy_enforce_consistency_at_target";
                break;
            case NVSHMEMI_CALL_SITE_PROXY_GLOBAL_EXIT:
                str = "nvshmemi_proxy_global_exit";
                break;
            case NVSHMEMI_CALL_SITE_AMO_FETCH_WAIT_FLAG:
                str = "nvshmemi_call_site_amo_fetch_wait_flag";
                break;
            case NVSHMEMI_CALL_SITE_AMO_FETCH_WAIT_DATA:
                str = "nvshmemi_call_site_amo_fetch_wait_data";
                break;
            case NVSHMEMI_CALL_SITE_G_WAIT_FLAG:
                str = "nvshmemi_call_site_g_wait_flag";
                break;
            default: { str = "unknown call site, exiting"; }
        }
        NVSHMEMI_ERROR_PRINT("received timeout signal from GPU thread(s) in %s\n", str);
        NVSHMEMI_ERROR_PRINT("signal addr %" PRIu64 " signal val found %" PRIu64
                             " signal val expected %" PRIu64 "\n",
                             timeout->signal_addr, timeout->signal_val_found,
                             timeout->signal_val_expected);
        exit(-1);
    }
}

void progress_transports(proxy_state_t *proxy_state) {
    int status = 0;
    nvshmemi_state_t *state = proxy_state->nvshmemi_state;

    for (int i = 0; i < state->num_initialized_transports; i++) {
        struct nvshmem_transport *tcurr = state->transports[i];

        if (!((proxy_state->transport_bitmap) & (1 << i)) &&
            (tcurr->type != NVSHMEM_TRANSPORT_LIB_CODE_IBGDA))
            continue;

        if (tcurr->host_ops.progress == NULL) continue;

        status = tcurr->host_ops.progress(tcurr);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "transport %d progress failed \n", i);
    }
out:
    NVSHMEMI_NZ_EXIT(status, "error in progress_transport \n");
}

// this has to be call before channels are torn down
void force_flush(proxy_state_t *proxy_state) {}

inline void progress(proxy_state_t *proxy_state) {
    // progress global exit request
    progress_global_exit(proxy_state);

    // progress quiet ops
    progress_quiet(proxy_state);

    // progress cst ops
    progress_cst(proxy_state);

#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
    progress_timeout_polling(proxy_state);
#endif

    // progress channels
    progress_channels(proxy_state);

    // progress transports
    progress_transports(proxy_state);
}

void *nvshmemi_proxy_progress(void *in) {
    proxy_progress_params_t *params = (proxy_progress_params_t *)in;
    proxy_state_t *proxy_state = params->state;

    nvshmem_nvtx_set_thread_name(proxy_state->nvshmemi_state->mype, "proxy");

    // set context on the current thread
    INFO(NVSHMEM_PROXY, "setting current CUDA context to saved context: %p",
         proxy_state->nvshmemi_state->cucontext);
    CUresult curesult = CUDA_SUCCESS;
    curesult = CUPFN(nvshmemi_cuda_syms, cuCtxSetCurrent(proxy_state->nvshmemi_state->cucontext));
    if (curesult != CUDA_SUCCESS) {
        NVSHMEMI_ERROR_EXIT("failed setting context on the proxy thread \n");
    }

    // setup progress channels
    channel_req = (base_request_t **)calloc(proxy_state->channel_count, sizeof(base_request_t *));

    // call progress until stop is signalled
    do {
        progress(proxy_state);
    } while (!*((volatile int *)&params->stop));

    free(channel_req);

    return NULL;
}

void *nvshmemi_proxy_progress_minimal(void *in) {
    proxy_progress_params_t *params = (proxy_progress_params_t *)in;
    proxy_state_t *proxy_state = params->state;

    nvshmem_nvtx_set_thread_name(proxy_state->nvshmemi_state->mype, "proxy");
    do {
        progress_global_exit(proxy_state);
#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
        progress_timeout_polling(proxy_state);
#endif
    } while (!*((volatile int *)&params->stop));

    return NULL;
}

int nvshmemi_proxy_finalize(nvshmemi_state_t *state) {
    INFO(NVSHMEM_INIT, "In nvshmemi_proxy_finalize");
    proxy_state_t *proxy_state = (proxy_state_t *)state->proxy;

    proxy_state->progress_params.stop = 1;

    pthread_join(proxy_state->progress_thread, NULL);
    /* Late proxy init state */
    if (proxy_state->stream) CUDA_RUNTIME_CHECK(cudaStreamDestroy(proxy_state->stream));
    if (proxy_state->queue_stream_in)
        CUDA_RUNTIME_CHECK(cudaStreamDestroy(proxy_state->queue_stream_in));
    if (proxy_state->queue_stream_out)
        CUDA_RUNTIME_CHECK(cudaStreamDestroy(proxy_state->queue_stream_out));

    /* setup connections state */
    free(proxy_state->transport_id);
    free(proxy_state->transport);

    /* If we are in a proxy-initiated global exit, the frees below create a cycle
     * preventing us from finishing the global exit. In this case, we have to rely
     * on the program being terminated successfully to release remaining resources.
     */
    if (proxy_state->global_exit_request_state &&
        *proxy_state->global_exit_request_state > PROXY_GLOBAL_EXIT_NOT_REQUESTED)
        return 0;
    /* setup device channels state */
    if (nvshmemi_device_state.proxy_channel_g_coalescing_buf)
        nvshmemi_free(nvshmemi_device_state.proxy_channel_g_coalescing_buf);
    if (nvshmemi_device_state.proxy_channel_g_buf)
        nvshmemi_free(nvshmemi_device_state.proxy_channel_g_buf);
    if (nvshmemi_device_state.proxy_channels_complete_local_ptr)
        CUDA_RUNTIME_CHECK(cudaFree(nvshmemi_device_state.proxy_channels_complete_local_ptr));
    if (nvshmemi_device_state.proxy_channel_g_buf_head_ptr)
        CUDA_RUNTIME_CHECK(cudaFree(nvshmemi_device_state.proxy_channel_g_buf_head_ptr));
    if (proxy_state->channels_device) CUDA_RUNTIME_CHECK(cudaFree(proxy_state->channels_device));

    /* create channels state */
    for (int i = 0; i < proxy_state->channel_count; i++) {
        CUDA_RUNTIME_CHECK(cudaFreeHost(proxy_state->channels[i].complete));
        CUDA_RUNTIME_CHECK(cudaFreeHost(proxy_state->channels[i].quiet_issue));
        CUDA_RUNTIME_CHECK(cudaFreeHost(proxy_state->channels[i].quiet_ack));
        CUDA_RUNTIME_CHECK(cudaFreeHost(proxy_state->channels[i].cst_issue));
        CUDA_RUNTIME_CHECK(cudaFreeHost(proxy_state->channels[i].cst_ack));
        CUDA_RUNTIME_CHECK(cudaFree(proxy_state->channels[i].issue));
        CUDA_RUNTIME_CHECK(cudaFreeHost(proxy_state->channels[i].buf));
    }

    free(proxy_state->channels);

    /* Early proxy init state */
    if (proxy_state->global_exit_request_state)
        CUDA_RUNTIME_CHECK(cudaFreeHost(proxy_state->global_exit_request_state));
    if (proxy_state->global_exit_code)
        CUDA_RUNTIME_CHECK(cudaFreeHost(proxy_state->global_exit_code));
    if (proxy_state->nvshmemi_timeout)
        CUDA_RUNTIME_CHECK(cudaFreeHost(proxy_state->nvshmemi_timeout));

    return 0;
}
