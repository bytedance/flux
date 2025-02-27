/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "libfabric.h"
#include <assert.h>
#include <sys/uio.h>  // IWYU pragma: keep
// IWYU pragma: no_include <bits/types/struct_iovec.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <mutex>
#include <errno.h>
#ifdef NVSHMEM_X86_64
#include <immintrin.h>  // IWYU pragma: keep
#endif
// IWYU pragma: no_include <xmmintrin.h>

#include "internal/host_transport/cudawrap.h"
#include "bootstrap_host_transport/env_defs_internal.h"
#include "device_host_transport/nvshmem_common_transport.h"
#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"
#include "internal/host_transport/nvshmemi_transport_defines.h"
#include "non_abi/nvshmemx_error.h"
#include "non_abi/nvshmem_build_options.h"  // IWYU pragma: keep
#include "non_abi/nvshmem_version.h"
#include "rdma/fabric.h"
#include "rdma/fi_atomic.h"
#include "rdma/fi_cm.h"
#include "rdma/fi_domain.h"
#include "rdma/fi_endpoint.h"
#include "rdma/fi_eq.h"
#include "rdma/fi_errno.h"
#include "rdma/fi_rma.h"
#include "internal/host_transport/transport.h"
#include "transport_common.h"

#ifdef NVSHMEM_USE_GDRCOPY
#include "transport_gdr_common.h"

static struct gdrcopy_function_table gdrcopy_ftable;
static void *gdrcopy_handle = NULL;
static gdr_t gdr_desc;
static bool use_gdrcopy = false;
#endif

/* Note - this is required to not break on Slingshot systems
 * where we compile with libfabric < 1.15.
 */
#ifndef FI_OPT_CUDA_API_PERMITTED
#define FI_OPT_CUDA_API_PERMITTED 10
#endif

static bool use_staged_atomics = false;
threadSafeOpQueue nvshmemtLibfabricOpQueue;
std::mutex gdrRecvMutex;

int nvshmemt_libfabric_gdr_process_amos(nvshmem_transport_t transport);

static int nvshmemt_libfabric_gdr_process_completion(nvshmemt_libfabric_endpoint_t *ep,
                                                     struct fi_cq_msg_entry *entry) {
    nvshmemt_libfabric_gdr_op_ctx_t *op;
    int status = 0;

    /* Puts, Gets, G ops don't have a context. */
    if (!entry->op_context) {
        goto out;
    }

    op = (nvshmemt_libfabric_gdr_op_ctx_t *)entry->op_context;

    if (entry->flags & FI_SEND) {
        nvshmemtLibfabricOpQueue.putToSend(op);
    } else if (entry->flags & FI_RMA) {
        /* inlined p ops or atomic responses */
        nvshmemtLibfabricOpQueue.putToSend(op);
    } else if (entry->flags & FI_RECV) {
        op->ep = ep;
        nvshmemtLibfabricOpQueue.putToRecv(op);
    } else {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "Found an invalid message type in an ep completion.\n");
    }

out:
    return status;
}

static int nvshmemt_libfabric_progress(nvshmem_transport_t transport) {
    nvshmemt_libfabric_state_t *libfabric_state = (nvshmemt_libfabric_state_t *)transport->state;
    int status;

    for (int i = 0; i < NVSHMEMT_LIBFABRIC_DEFAULT_NUM_EPS; i++) {
        uint64_t cnt = fi_cntr_readerr(libfabric_state->eps[i].counter);

        if (cnt > 0) {
            NVSHMEMI_WARN_PRINT("Nonzero error count progressing EP %d (%" PRIu64 ")\n", i, cnt);

            struct fi_cq_err_entry err;
            memset(&err, 0, sizeof(struct fi_cq_err_entry));
            ssize_t nerr = fi_cq_readerr(libfabric_state->eps[i].cq, &err, 0);

            if (nerr > 0) {
                char str[100] = "\0";
                const char *err_str = fi_cq_strerror(libfabric_state->eps[i].cq, err.prov_errno,
                                                     err.err_data, str, 100);
                NVSHMEMI_WARN_PRINT(
                    "CQ %d reported error (%d): %s\n\tProvider error: %s\n\tSupplemental error "
                    "info: %s\n",
                    i, err.err, fi_strerror(err.err), err_str ? err_str : "none",
                    strlen(str) ? str : "none");
            } else if (nerr == -FI_EAGAIN) {
                NVSHMEMI_WARN_PRINT("fi_cq_readerr returned -FI_EAGAIN\n");
            } else {
                NVSHMEMI_WARN_PRINT("fi_cq_readerr returned %zd: %s\n", nerr,
                                    fi_strerror(-1 * nerr));
            }
            return NVSHMEMX_ERROR_INTERNAL;
        }

        {
            char buf[8192];
            ssize_t qstatus;
            nvshmemt_libfabric_endpoint_t *ep = &libfabric_state->eps[i];
            do {
                qstatus = fi_cq_read(ep->cq, buf, 8192 / sizeof(struct fi_cq_msg_entry));
                /* Note - EFA provider does not support selective completions */
                if (qstatus > 0) {
                    if (libfabric_state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
                        struct fi_cq_msg_entry *entry = (struct fi_cq_msg_entry *)buf;
                        for (int i = 0; i < qstatus; i++, entry++) {
                            nvshmemt_libfabric_gdr_process_completion(ep, entry);
                        }
                    } else {
                        NVSHMEMI_WARN_PRINT("Got %zd unexpected events on EP %d\n", qstatus, i);
                    }
                }
            } while (qstatus > 0);
            if (qstatus < 0 && qstatus != -FI_EAGAIN) {
                NVSHMEMI_WARN_PRINT("Error progressing CQ (%zd): %s\n", qstatus,
                                    fi_strerror(qstatus * -1));
                return NVSHMEMX_ERROR_INTERNAL;
            }
        }
    }

    if (libfabric_state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
        if (gdrRecvMutex.try_lock()) {
            status = nvshmemt_libfabric_gdr_process_amos(transport);
            gdrRecvMutex.unlock();
            if (status) {
                return NVSHMEMX_ERROR_INTERNAL;
            }
        }
    }

    return 0;
}

static inline int try_again(nvshmem_transport_t transport, int *status, uint64_t *num_retries) {
    if (likely(*status == 0)) {
        return 0;
    }

    if (*status == -FI_EAGAIN) {
        if (*num_retries >= NVSHMEMT_LIBFABRIC_MAX_RETRIES) {
            *status = NVSHMEMX_ERROR_INTERNAL;
            return 0;
        }
        (*num_retries)++;
        *status = nvshmemt_libfabric_progress(transport);
    }

    if (*status != 0) {
        *status = NVSHMEMX_ERROR_INTERNAL;
        return 0;
    }

    return 1;
}

template <typename T>
int perform_gdrcopy_amo(nvshmem_transport_t transport, nvshmemt_libfabric_gdr_op_ctx_t *op) {
    T old_value, new_value;
    uint64_t num_retries = 0;
    nvshmemt_libfabric_state_t *libfabric_state = (nvshmemt_libfabric_state_t *)transport->state;
    nvshmemt_libfabric_gdr_send_amo_op_t *received_op;
    nvshmemt_libfabric_gdr_op_ctx_t *resp_op = NULL;
    nvshmemt_libfabric_memhandle_info_t *handle_info;
    volatile T *ptr;
    int status = 0;

    received_op = &(op->send_amo);

    handle_info = (nvshmemt_libfabric_memhandle_info_t *)nvshmemt_mem_handle_cache_get(
        transport, libfabric_state->cache, received_op->target_addr);
    NVSHMEMI_NULL_ERROR_JMP(handle_info, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "Unable to get mem handle for atomic.\n");
#ifdef NVSHMEM_USE_GDRCOPY
    if (use_gdrcopy) {
        ptr = (volatile T *)((char *)handle_info->cpu_ptr +
                             ((char *)received_op->target_addr - (char *)handle_info->ptr));
    } else
#endif
    {
        ptr = (volatile T *)received_op->target_addr;
    }
    old_value = *ptr;

    switch (received_op->op) {
        case NVSHMEMI_AMO_SIGNAL:
        case NVSHMEMI_AMO_SIGNAL_SET:
        case NVSHMEMI_AMO_SET:
        case NVSHMEMI_AMO_SWAP: {
            /* The static_cast is used to truncate the uint64_t value of swap_add back to its
             * original length */
            new_value = static_cast<T>(received_op->swap_add);
            break;
        }
        case NVSHMEMI_AMO_ADD:
        case NVSHMEMI_AMO_SIGNAL_ADD:
        case NVSHMEMI_AMO_FETCH_ADD: {
            new_value = old_value + static_cast<T>(received_op->swap_add);
            break;
        }
        case NVSHMEMI_AMO_OR:
        case NVSHMEMI_AMO_FETCH_OR: {
            new_value = old_value | static_cast<T>(received_op->swap_add);
            break;
        }
        case NVSHMEMI_AMO_AND:
        case NVSHMEMI_AMO_FETCH_AND: {
            new_value = old_value & static_cast<T>(received_op->swap_add);
            break;
        }
        case NVSHMEMI_AMO_XOR:
        case NVSHMEMI_AMO_FETCH_XOR: {
            new_value = old_value ^ static_cast<T>(received_op->swap_add);
            break;
        }
        case NVSHMEMI_AMO_COMPARE_SWAP: {
            new_value = (old_value == static_cast<T>(received_op->comp))
                            ? static_cast<T>(received_op->swap_add)
                            : old_value;
            break;
        }
        case NVSHMEMI_AMO_FETCH: {
            new_value = old_value;
            break;
        }
        default: {
            NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                               "RMA/AMO verb %d not implemented\n", received_op->op);
        }
    }

    *ptr = new_value;
    STORE_BARRIER();

    if (received_op->op > NVSHMEMI_AMO_END_OF_NONFETCH) {
        do {
            resp_op = (nvshmemt_libfabric_gdr_op_ctx_t *)nvshmemtLibfabricOpQueue.getNextSend();
            status = resp_op == NULL ? -EAGAIN : 0;
        } while (try_again(transport, &status, &num_retries));

        num_retries = 0;
        NVSHMEMI_NULL_ERROR_JMP(resp_op, status, NVSHMEMX_ERROR_INTERNAL, out,
                                "Unable to respond to atomic request.\n");
        resp_op->ret_amo.elem.data = old_value;
        resp_op->ret_amo.elem.flag = received_op->retflag;
        resp_op->ret_amo.ret_addr = received_op->ret_addr;
        resp_op->type = NVSHMEMT_LIBFABRIC_ACK;

        do {
            status =
                fi_send(op->ep->endpoint, (void *)resp_op, sizeof(nvshmemt_libfabric_gdr_op_ctx_t),
                        NULL, received_op->ret_ep, (void *)resp_op);
        } while (try_again(transport, &status, &num_retries));
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Unable to respond to atomic request.\n");
        op->ep->submitted_ops++;
    }

out:
    return status;
}

int nvshmemt_libfabric_gdr_process_amo(nvshmem_transport_t transport,
                                       nvshmemt_libfabric_gdr_op_ctx_t *op) {
    int status = 0;

    switch (op->send_amo.size) {
        case 2:
            status = perform_gdrcopy_amo<uint16_t>(transport, op);
            break;
        case 4:
            status = perform_gdrcopy_amo<uint32_t>(transport, op);
            break;
        case 8:
            status = perform_gdrcopy_amo<uint64_t>(transport, op);
            break;
        default:
            NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                               "invalid element size encountered %u\n", op->send_amo.size);
    }

out:
    return status;
}

int nvshmemt_libfabric_gdr_process_ack(nvshmem_transport_t transport,
                                       nvshmemt_libfabric_gdr_op_ctx_t *op) {
    nvshmemt_libfabric_gdr_ret_amo_op_t *ret = &op->ret_amo;
    nvshmemt_libfabric_state_t *libfabric_state = (nvshmemt_libfabric_state_t *)transport->state;
    nvshmemt_libfabric_memhandle_info_t *handle_info;
    g_elem_t *elem;
    void *valid_cpu_ptr;

    handle_info = (nvshmemt_libfabric_memhandle_info_t *)nvshmemt_mem_handle_cache_get(
        transport, libfabric_state->cache, ret->ret_addr);
    if (!handle_info) {
        NVSHMEMI_ERROR_PRINT("Unable to get handle info for atomic response.\n");
        return NVSHMEMX_ERROR_INTERNAL;
    }

    valid_cpu_ptr =
        (void *)((char *)handle_info->cpu_ptr + ((char *)ret->ret_addr - (char *)handle_info->ptr));
    assert(valid_cpu_ptr);
    elem = (g_elem_t *)valid_cpu_ptr;
    elem->data = ret->elem.data;
    elem->flag = ret->elem.flag;
    return 0;
}

int nvshmemt_libfabric_gdr_process_amos(nvshmem_transport_t transport) {
    nvshmemt_libfabric_gdr_op_ctx_t *op;
    int status = 0;

    op = (nvshmemt_libfabric_gdr_op_ctx_t *)nvshmemtLibfabricOpQueue.getNextRecv();
    while (op) {
        if (op->type == NVSHMEMT_LIBFABRIC_SEND) {
            status = nvshmemt_libfabric_gdr_process_amo(transport, op);
        } else {
            status = nvshmemt_libfabric_gdr_process_ack(transport, op);
        }

        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "Unable to process atomic.\n");
        status = fi_recv(op->ep->endpoint, (void *)op, sizeof(nvshmemt_libfabric_gdr_op_ctx_t),
                         NULL, FI_ADDR_UNSPEC, (void *)op);
        if (status) {
            NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "Unable to re-post recv.\n");
        }
        op = (nvshmemt_libfabric_gdr_op_ctx_t *)nvshmemtLibfabricOpQueue.getNextRecv();
    }
out:
    return status;
}

static int nvshmemt_libfabric_quiet(struct nvshmem_transport *tcurr, int pe, int is_proxy) {
    nvshmemt_libfabric_state_t *libfabric_state = (nvshmemt_libfabric_state_t *)tcurr->state;
    nvshmemt_libfabric_endpoint_t *ep;
    int status = 0;

    if (is_proxy) {
        ep = &libfabric_state->eps[NVSHMEMT_LIBFABRIC_PROXY_EP_IDX];
    } else {
        ep = &libfabric_state->eps[NVSHMEMT_LIBFABRIC_HOST_EP_IDX];
    }

    if (likely(libfabric_state->prov_info->domain_attr->control_progress == FI_PROGRESS_MANUAL) ||
        (libfabric_state->prov_info->domain_attr->data_progress == FI_PROGRESS_MANUAL) ||
        (use_staged_atomics == true)
#ifdef NVSHMEM_USE_GDRCOPY
        || (use_gdrcopy == true)
#endif
    ) {
        uint64_t submitted, completed;
        for (;;) {
            completed = fi_cntr_read(ep->counter);
            submitted = ep->submitted_ops;
            if (completed == submitted)
                break;
            else {
                if (nvshmemt_libfabric_progress(tcurr)) {
                    status = NVSHMEMX_ERROR_INTERNAL;
                    break;
                }
            }
        }
    } else {
        status = fi_cntr_wait(ep->counter, ep->submitted_ops, NVSHMEMT_LIBFABRIC_QUIET_TIMEOUT_MS);
        if (status) {
            /* note - Status is negative for this function in error cases but
             * fi_strerror only accepts positive values.
             */
            NVSHMEMI_ERROR_PRINT("Error in quiet operation (%d): %s.\n", status,
                                 fi_strerror(status * -1));
            status = NVSHMEMX_ERROR_INTERNAL;
        }
    }

    return status;
}

static int nvshmemt_libfabric_show_info(struct nvshmem_transport *transport, int style) {
    NVSHMEMI_ERROR_PRINT("libfabric show info not implemented");
    return 0;
}

static int nvshmemt_libfabric_rma(struct nvshmem_transport *tcurr, int pe, rma_verb_t verb,
                                  rma_memdesc_t *remote, rma_memdesc_t *local,
                                  rma_bytesdesc_t bytesdesc, int is_proxy) {
    nvshmemt_libfabric_mem_handle_ep_t *remote_handle, *local_handle;
    nvshmemt_libfabric_state_t *libfabric_state = (nvshmemt_libfabric_state_t *)tcurr->state;
    struct iovec p_op_l_iov;
    struct fi_msg_rma p_op_msg;
    struct fi_rma_iov p_op_r_iov;
    nvshmemt_libfabric_endpoint_t *ep;
    size_t op_size;
    uint64_t num_retries = 0;
    int status = 0;
    int target_ep;
    int ep_idx = 0;

    memset(&p_op_l_iov, 0, sizeof(struct iovec));
    memset(&p_op_msg, 0, sizeof(struct fi_msg_rma));
    memset(&p_op_r_iov, 0, sizeof(struct fi_rma_iov));

    if (is_proxy) {
        ep_idx = NVSHMEMT_LIBFABRIC_PROXY_EP_IDX;
    } else {
        ep_idx = NVSHMEMT_LIBFABRIC_HOST_EP_IDX;
    }

    ep = &libfabric_state->eps[ep_idx];
    target_ep = pe * NVSHMEMT_LIBFABRIC_DEFAULT_NUM_EPS + ep_idx;

    remote_handle = &((nvshmemt_libfabric_mem_handle_t *)remote->handle)->hdls[ep_idx];
    local_handle = &((nvshmemt_libfabric_mem_handle_t *)local->handle)->hdls[ep_idx];
    op_size = bytesdesc.elembytes * bytesdesc.nelems;

    if (verb.desc == NVSHMEMI_OP_P) {
        if (libfabric_state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
            nvshmemt_libfabric_gdr_op_ctx_t *p_buf;
            do {
                p_buf = (nvshmemt_libfabric_gdr_op_ctx_t *)nvshmemtLibfabricOpQueue.getNextSend();
                status = p_buf == NULL ? -EAGAIN : 0;
            } while (try_again(tcurr, &status, &num_retries));
            NVSHMEMI_NULL_ERROR_JMP(p_buf, status, NVSHMEMX_ERROR_INTERNAL, out,
                                    "Unable to get buffer for P request.\n");
            num_retries = 0;
            do {
                p_buf->p_op.value = *(uint64_t *)local->ptr;
                status = fi_write(ep->endpoint, &p_buf->p_op.value, op_size, NULL, target_ep,
                                  (uintptr_t)remote->ptr, remote_handle->key, p_buf);
            } while (try_again(tcurr, &status, &num_retries));
        } else {
            p_op_msg.msg_iov = &p_op_l_iov;
            p_op_msg.desc = NULL;  // Local buffer is on the stack
            p_op_msg.iov_count = 1;
            p_op_msg.addr = target_ep;
            p_op_msg.rma_iov = &p_op_r_iov;
            p_op_msg.rma_iov_count = 1;

            p_op_l_iov.iov_base = local->ptr;
            p_op_l_iov.iov_len = op_size;

            if (libfabric_state->prov_info->domain_attr->mr_mode & FI_MR_VIRT_ADDR)
                p_op_r_iov.addr = (uintptr_t)remote->ptr;
            else
                p_op_r_iov.addr = (uintptr_t)remote->offset;
            p_op_r_iov.len = op_size;
            p_op_r_iov.key = remote_handle->key;

            /* The p buffer is on the stack so use
             * FI_INJECT to avoid segfaults during async runs.
             */
            do {
                status = fi_writemsg(ep->endpoint, &p_op_msg, FI_INJECT);
            } while (try_again(tcurr, &status, &num_retries));
        }
    } else if (verb.desc == NVSHMEMI_OP_PUT) {
        uintptr_t remote_addr;
        if (libfabric_state->prov_info->domain_attr->mr_mode & FI_MR_VIRT_ADDR)
            remote_addr = (uintptr_t)remote->ptr;
        else
            remote_addr = (uintptr_t)remote->offset;

        do {
            status = fi_write(ep->endpoint, local->ptr, op_size, local_handle->local_desc,
                              target_ep, remote_addr, remote_handle->key, NULL);
        } while (try_again(tcurr, &status, &num_retries));
    } else if (verb.desc == NVSHMEMI_OP_G || verb.desc == NVSHMEMI_OP_GET) {
        uintptr_t remote_addr;
        if (libfabric_state->prov_info->domain_attr->mr_mode & FI_MR_VIRT_ADDR)
            remote_addr = (uintptr_t)remote->ptr;
        else
            remote_addr = (uintptr_t)remote->offset;

        do {
            status = fi_read(ep->endpoint, local->ptr, op_size, local_handle->local_desc, target_ep,
                             remote_addr, remote_handle->key, NULL);
        } while (try_again(tcurr, &status, &num_retries));
    } else {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "Invalid RMA operation specified.\n");
    }

    if (status) goto out;  // Status set by try_again
    ep->submitted_ops++;

out:
    if (status) {
        NVSHMEMI_ERROR_PRINT("Received an error when trying to post an RMA operation.\n");
    }

    return status;
}

static int nvshmemt_libfabric_gdr_amo(struct nvshmem_transport *transport, int pe, void *curetptr,
                                      amo_verb_t verb, amo_memdesc_t *remote,
                                      amo_bytesdesc_t bytesdesc, int is_proxy) {
    nvshmemt_libfabric_state_t *libfabric_state = (nvshmemt_libfabric_state_t *)transport->state;
    nvshmemt_libfabric_endpoint_t *ep;
    nvshmemt_libfabric_gdr_op_ctx_t *amo;
    uint64_t num_retries = 0;
    int target_ep, ep_idx;
    int status = 0;

    if (is_proxy) {
        ep_idx = NVSHMEMT_LIBFABRIC_PROXY_EP_IDX;
    } else {
        ep_idx = NVSHMEMT_LIBFABRIC_HOST_EP_IDX;
    }

    ep = &libfabric_state->eps[ep_idx];
    target_ep = pe * NVSHMEMT_LIBFABRIC_DEFAULT_NUM_EPS + ep_idx;

    do {
        amo = (nvshmemt_libfabric_gdr_op_ctx_t *)nvshmemtLibfabricOpQueue.getNextSend();
        status = amo == NULL ? -EAGAIN : 0;
    } while (try_again(transport, &status, &num_retries));

    if (status) {
        NVSHMEMI_ERROR_PRINT("Unable to retrieve AMO operation.");
        goto out;
    }

    amo->type = NVSHMEMT_LIBFABRIC_SEND;
    amo->send_amo.op = verb.desc;
    amo->send_amo.target_addr = remote->remote_memdesc.ptr;
    amo->send_amo.ret_addr = remote->retptr;
    amo->send_amo.retflag = remote->retflag;
    amo->send_amo.comp = remote->cmp;
    amo->send_amo.swap_add = remote->val;
    amo->send_amo.size = bytesdesc.elembytes;
    amo->send_amo.ret_ep = transport->my_pe * NVSHMEMT_LIBFABRIC_DEFAULT_NUM_EPS + ep_idx;

    num_retries = 0;
    do {
        status = fi_send(ep->endpoint, (void *)amo, sizeof(nvshmemt_libfabric_gdr_op_ctx_t), NULL,
                         target_ep, (void *)amo);
    } while (try_again(transport, &status, &num_retries));

    if (status) {
        NVSHMEMI_ERROR_PRINT("Received an error when trying to post an AMO operation. %s\n",
                             fi_strerror(status * -1));
        status = NVSHMEMX_ERROR_INTERNAL;
    } else {
        ep->submitted_ops++;
    }

out:
    return status;
}

static int nvshmemt_libfabric_amo(struct nvshmem_transport *transport, int pe, void *curetptr,
                                  amo_verb_t verb, amo_memdesc_t *remote, amo_bytesdesc_t bytesdesc,
                                  int is_proxy) {
    nvshmemt_libfabric_state_t *libfabric_state = (nvshmemt_libfabric_state_t *)transport->state;
    nvshmemt_libfabric_mem_handle_ep_t *remote_handle = NULL, *local_handle = NULL;
    nvshmemt_libfabric_endpoint_t *ep;
    struct fi_msg_atomic amo_msg;
    struct fi_ioc fi_local_iov;
    struct fi_ioc fi_comp_iov;
    struct fi_ioc fi_ret_iov;
    struct fi_rma_ioc fi_remote_iov;
    enum fi_datatype data;
    enum fi_op op;
    uint64_t num_retries = 0;
    int target_ep;
    int status = 0;
    int ep_idx;

    memset(&amo_msg, 0, sizeof(struct fi_msg_atomic));
    memset(&fi_local_iov, 0, sizeof(struct fi_ioc));
    memset(&fi_comp_iov, 0, sizeof(struct fi_ioc));
    memset(&fi_ret_iov, 0, sizeof(struct fi_ioc));
    memset(&fi_remote_iov, 0, sizeof(struct fi_rma_ioc));

    if (is_proxy) {
        ep_idx = NVSHMEMT_LIBFABRIC_PROXY_EP_IDX;
    } else {
        ep_idx = NVSHMEMT_LIBFABRIC_HOST_EP_IDX;
    }

    ep = &libfabric_state->eps[ep_idx];
    target_ep = pe * NVSHMEMT_LIBFABRIC_DEFAULT_NUM_EPS + ep_idx;

    remote_handle =
        &((nvshmemt_libfabric_mem_handle_t *)remote->remote_memdesc.handle)->hdls[ep_idx];
    if (verb.desc > NVSHMEMI_AMO_END_OF_NONFETCH) {
        local_handle = &((nvshmemt_libfabric_mem_handle_t *)remote->ret_handle)->hdls[ep_idx];
    }

    if (bytesdesc.elembytes == 8) {
        data = FI_UINT64;
    } else if (bytesdesc.elembytes == 4) {
        data = FI_UINT32;
    } else {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "Invalid atomic size specified.\n");
    }

    switch (verb.desc) {
        case NVSHMEMI_AMO_SWAP:
        case NVSHMEMI_AMO_SIGNAL:
        case NVSHMEMI_AMO_SIGNAL_SET:
        case NVSHMEMI_AMO_SET: {
            op = FI_ATOMIC_WRITE;
            break;
        }
        case NVSHMEMI_AMO_FETCH_INC:
        case NVSHMEMI_AMO_INC:
        case NVSHMEMI_AMO_FETCH_ADD:
        case NVSHMEMI_AMO_SIGNAL_ADD:
        case NVSHMEMI_AMO_ADD: {
            op = FI_SUM;
            break;
        }
        case NVSHMEMI_AMO_FETCH_AND:
        case NVSHMEMI_AMO_AND: {
            op = FI_BAND;
            break;
        }
        case NVSHMEMI_AMO_FETCH_OR:
        case NVSHMEMI_AMO_OR: {
            op = FI_BOR;
            break;
        }
        case NVSHMEMI_AMO_FETCH_XOR:
        case NVSHMEMI_AMO_XOR: {
            op = FI_BXOR;
            break;
        }
        case NVSHMEMI_AMO_FETCH: {
            op = FI_ATOMIC_READ;
            break;
        }
        case NVSHMEMI_AMO_COMPARE_SWAP: {
            op = FI_CSWAP;
            break;
        }
        default: {
            NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out, "Opcode %d is invalid.\n",
                               verb.desc);
        }
    }

    if (op != FI_ATOMIC_READ) {
        fi_local_iov.addr = &remote->val;
        fi_local_iov.count = 1;
        amo_msg.msg_iov = &fi_local_iov;
        amo_msg.desc = NULL;  // Local operands are on the stack
        amo_msg.iov_count = 1;
    }

    amo_msg.addr = target_ep;

    if (libfabric_state->prov_info->domain_attr->mr_mode & FI_MR_VIRT_ADDR)
        fi_remote_iov.addr = (uintptr_t)remote->remote_memdesc.ptr;
    else
        fi_remote_iov.addr = (uintptr_t)remote->remote_memdesc.offset;

    fi_remote_iov.count = 1;
    fi_remote_iov.key = remote_handle->key;
    amo_msg.rma_iov = &fi_remote_iov;
    amo_msg.rma_iov_count = 1;

    amo_msg.datatype = data;
    amo_msg.op = op;

    amo_msg.context = NULL;
    amo_msg.data = 0;

    if (verb.desc > NVSHMEMI_AMO_END_OF_NONFETCH) {
        fi_ret_iov.addr = remote->retptr;
        fi_ret_iov.count = 1;
        if (verb.desc == NVSHMEMI_AMO_COMPARE_SWAP) {
            fi_comp_iov.addr = &remote->cmp;
            fi_comp_iov.count = 1;
        }
    }

    do {
        if (verb.desc == NVSHMEMI_AMO_COMPARE_SWAP) {
            status = fi_compare_atomicmsg(ep->endpoint, &amo_msg, &fi_comp_iov, NULL, 1,
                                          &fi_ret_iov, &local_handle->local_desc, 1, FI_INJECT);
        } else if (verb.desc < NVSHMEMI_AMO_END_OF_NONFETCH) {
            status = fi_atomicmsg(ep->endpoint, &amo_msg, op == FI_ATOMIC_READ ? 0 : FI_INJECT);
        } else {
            status = fi_fetch_atomicmsg(ep->endpoint, &amo_msg, &fi_ret_iov,
                                        &local_handle->local_desc, 1, FI_INJECT);
        }
    } while (try_again(transport, &status, &num_retries));

    if (status) goto out;  // Status set by try_again

    ep->submitted_ops++;

out:
    if (status) {
        NVSHMEMI_ERROR_PRINT("Received an error when trying to post an AMO operation. %s\n",
                             fi_strerror(status * -1));
        status = NVSHMEMX_ERROR_INTERNAL;
    }
    return status;
}

static int nvshmemt_libfabric_enforce_cst(struct nvshmem_transport *tcurr) {
    nvshmemt_libfabric_state_t *libfabric_state = (nvshmemt_libfabric_state_t *)tcurr->state;
    uint64_t num_retries = 0;
    int status;
    int target_ep;
    int mype = tcurr->my_pe;

#ifdef NVSHMEM_USE_GDRCOPY
    if (use_gdrcopy) {
        if (libfabric_state->provider != NVSHMEMT_LIBFABRIC_PROVIDER_SLINGSHOT) {
            int temp;
            nvshmemt_libfabric_memhandle_info_t *mem_handle_info;

            mem_handle_info =
                (nvshmemt_libfabric_memhandle_info_t *)nvshmemt_mem_handle_cache_get_by_idx(
                    libfabric_state->cache, 0);
            if (!mem_handle_info) {
                goto skip;
            }
            gdrcopy_ftable.copy_from_mapping(mem_handle_info->mh, &temp, mem_handle_info->cpu_ptr,
                                             sizeof(int));
        }
    }

skip:
#endif

    target_ep = mype * NVSHMEMT_LIBFABRIC_DEFAULT_NUM_EPS + NVSHMEMT_LIBFABRIC_PROXY_EP_IDX;
    do {
        struct fi_msg_rma msg;
        struct iovec l_iov;
        struct fi_rma_iov r_iov;
        void *desc = libfabric_state->local_mr_desc[NVSHMEMT_LIBFABRIC_PROXY_EP_IDX];
        uint64_t flags;

        memset(&msg, 0, sizeof(struct fi_msg_rma));
        memset(&l_iov, 0, sizeof(struct iovec));
        memset(&r_iov, 0, sizeof(struct fi_rma_iov));

        l_iov.iov_base = libfabric_state->local_mem_ptr;
        l_iov.iov_len = 8;

        r_iov.addr = 0;  // Zero offset
        r_iov.len = 8;
        r_iov.key = libfabric_state->local_mr_key[NVSHMEMT_LIBFABRIC_PROXY_EP_IDX];

        msg.msg_iov = &l_iov;
        msg.desc = &desc;
        msg.iov_count = 1;
        msg.rma_iov = &r_iov;
        msg.rma_iov_count = 1;
        msg.context = NULL;
        msg.data = 0;

        flags = FI_DELIVERY_COMPLETE;

        if (libfabric_state->prov_info->caps & FI_FENCE) flags |= FI_FENCE;

        status =
            fi_readmsg(libfabric_state->eps[NVSHMEMT_LIBFABRIC_PROXY_EP_IDX].endpoint, &msg, flags);
    } while (try_again(tcurr, &status, &num_retries));

    libfabric_state->eps[target_ep].submitted_ops++;
    return status;
}

static int nvshmemt_libfabric_release_mem_handle(nvshmem_mem_handle_t *mem_handle,
                                                 nvshmem_transport_t t) {
    nvshmemt_libfabric_state_t *libfabric_state = (nvshmemt_libfabric_state_t *)t->state;
    nvshmemt_libfabric_mem_handle_t *fabric_handle;
    int max_reg, status = 0;

    assert(mem_handle != NULL);
    fabric_handle = (nvshmemt_libfabric_mem_handle_t *)mem_handle;

    if (libfabric_state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
        nvshmemt_libfabric_memhandle_info_t *handle_info;
        handle_info = (nvshmemt_libfabric_memhandle_info_t *)nvshmemt_mem_handle_cache_get(
            t, libfabric_state->cache, fabric_handle->buf);
        if (handle_info != NULL) {
#ifdef NVSHMEM_USE_GDRCOPY
            if ((use_gdrcopy == true) && (handle_info->gdr_mapping_size > 0)) {
                status = gdrcopy_ftable.unmap(gdr_desc, handle_info->mh, handle_info->cpu_ptr_base,
                                              handle_info->gdr_mapping_size);
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gdr_unmap failed\n");

                status = gdrcopy_ftable.unpin_buffer(gdr_desc, handle_info->mh);
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gdr_unpin failed\n");
            }
#endif
            if (libfabric_state->cache != NULL)
                nvshmemt_mem_handle_cache_remove(t, libfabric_state->cache, handle_info->ptr);
        }
    }

    if (libfabric_state->prov_info->domain_attr->mr_mode & FI_MR_ENDPOINT)
        max_reg = NVSHMEMT_LIBFABRIC_DEFAULT_NUM_EPS;
    else
        max_reg = 1;

    for (int i = 0; i < max_reg; i++) {
        if (libfabric_state->local_mr[i] == fabric_handle->hdls[i].mr)
            libfabric_state->local_mr[i] = NULL;

        int status = fi_close(&fabric_handle->hdls[i].mr->fid);
        if (status) {
            NVSHMEMI_WARN_PRINT("Error releasing mem handle idx %d (%d): %s\n", i, status,
                                fi_strerror(status * -1));
        }
    }

out:
    return status;
}

static int nvshmemt_libfabric_get_mem_handle(nvshmem_mem_handle_t *mem_handle,
                                             nvshmem_mem_handle_t *mem_handle_in, void *buf,
                                             size_t length, nvshmem_transport_t t,
                                             bool local_only) {
    nvshmemt_libfabric_mem_handle_t *fabric_handle;
    nvshmemt_libfabric_state_t *libfabric_state = (nvshmemt_libfabric_state_t *)t->state;
    cudaPointerAttributes attr = {};
    struct fi_mr_attr mr_attr;
    struct iovec mr_iovec;
    int status;
    bool is_host = true;
    CUdevice gpu_device_id;
    nvshmemt_libfabric_memhandle_info_t *handle_info = NULL;
#ifdef NVSHMEM_USE_GDRCOPY
    gdr_info_t info;
#endif

    status = CUPFN(libfabric_state->table, cuCtxGetDevice(&gpu_device_id));
    if (status != CUDA_SUCCESS) {
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }

    assert(mem_handle != NULL);
    fabric_handle = (nvshmemt_libfabric_mem_handle_t *)mem_handle;
    status = cudaPointerGetAttributes(&attr, buf);
    if (status != cudaSuccess) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                           "Unable to query pointer attributes.\n");
    }

    if (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged) {
        is_host = false;
    }

    memset(&mr_attr, 0, sizeof(struct fi_mr_attr));
    memset(&mr_iovec, 0, sizeof(struct iovec));

    mr_iovec.iov_base = buf;
    mr_iovec.iov_len = length;
    mr_attr.mr_iov = &mr_iovec;
    mr_attr.iov_count = 1;
    mr_attr.access = FI_READ | FI_WRITE;
    if (!local_only) {
        mr_attr.access |= FI_REMOTE_READ | FI_REMOTE_WRITE;
    }
    mr_attr.offset = 0;
    mr_attr.context = NULL;
    if (!is_host) {
        mr_attr.iface = FI_HMEM_CUDA;
        mr_attr.device.cuda = gpu_device_id;
    } else {
        mr_attr.iface = FI_HMEM_SYSTEM;
    }

    fabric_handle->buf = buf;
    if (libfabric_state->prov_info->domain_attr->mr_mode & FI_MR_ENDPOINT) {
        for (int i = 0; i < NVSHMEMT_LIBFABRIC_DEFAULT_NUM_EPS; i++) {
            status =
                fi_mr_regattr(libfabric_state->domain, &mr_attr, 0, &fabric_handle->hdls[i].mr);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "Error registering memory region: %s\n",
                                  fi_strerror(status * -1));

            status =
                fi_mr_bind(fabric_handle->hdls[i].mr, &libfabric_state->eps[i].endpoint->fid, 0);

            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "Error binding MR to EP %d: %s\n", i, fi_strerror(status * -1));

            status = fi_mr_enable(fabric_handle->hdls[i].mr);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "Error enabling MR: %s\n",
                                  fi_strerror(status * -1));

            fabric_handle->hdls[i].key = fi_mr_key(fabric_handle->hdls[i].mr);
            fabric_handle->hdls[i].local_desc = fi_mr_desc(fabric_handle->hdls[i].mr);
        }
    } else {
        struct fid_mr *mr;

        status = fi_mr_regattr(libfabric_state->domain, &mr_attr, 0, &mr);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Error registering memory region: %s\n", fi_strerror(status * -1));

        for (int i = 0; i < NVSHMEMT_LIBFABRIC_DEFAULT_NUM_EPS; i++) {
            fabric_handle->hdls[i].mr = mr;
            fabric_handle->hdls[i].key = fi_mr_key(mr);
            fabric_handle->hdls[i].local_desc = fi_mr_desc(mr);
        }
    }

    if (!local_only && libfabric_state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
        if (libfabric_state->cache == NULL) {
            status = nvshmemt_mem_handle_cache_init(t, &libfabric_state->cache);
            NVSHMEMI_NZ_ERROR_JMP(status, status, out, "mem handle cache initialization failed.");
        }

        handle_info = (nvshmemt_libfabric_memhandle_info_t *)calloc(
            1, sizeof(nvshmemt_libfabric_memhandle_info_t));
        NVSHMEMI_NULL_ERROR_JMP(handle_info, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                "Cannot allocate memory handle info.\n");

        if (!is_host) {
#ifdef NVSHMEM_USE_GDRCOPY
            if (use_gdrcopy) {
                status = gdrcopy_ftable.pin_buffer(gdr_desc, (unsigned long)buf, length, 0, 0,
                                                   &handle_info->mh);
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                      "gdrcopy pin_buffer failed \n");

                status = gdrcopy_ftable.map(gdr_desc, handle_info->mh, &handle_info->cpu_ptr_base,
                                            length);
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                      "gdrcopy map failed \n");

                status = gdrcopy_ftable.get_info(gdr_desc, handle_info->mh, &info);
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                      "gdrcopy get_info failed \n");

                // remember that mappings start on a 64KB boundary, so let's
                // calculate the offset from the head of the mapping to the
                // beginning of the buffer
                handle_info->cpu_ptr =
                    (void *)((char *)handle_info->cpu_ptr_base + ((char *)buf - (char *)info.va));

                handle_info->gdr_mapping_size = length;
                handle_info->ptr = buf;
                status = nvshmemt_mem_handle_cache_add(t, libfabric_state->cache, buf,
                                                       (void *)handle_info);
                NVSHMEMI_NZ_ERROR_JMP(status, status, out,
                                      "Unable to add key to mem handle info cache");
            } else
#endif
            {
                NVSHMEMI_ERROR_PRINT(
                    "GDRCopy support not enabled. Unable to register gpu memory handle info.");
                status = NVSHMEMX_ERROR_INVALID_VALUE;
                goto out;
            }
        } else {
            handle_info->ptr = buf;
            handle_info->cpu_ptr = buf;
            handle_info->gdr_mapping_size = 0;
        }
        status = nvshmemt_mem_handle_cache_add(t, libfabric_state->cache, buf, (void *)handle_info);
        NVSHMEMI_NZ_ERROR_JMP(status, status, out, "Unable to add key to mem handle info cache");
    }

    if (libfabric_state->local_mr[0] == NULL && !local_only) {
        for (int i = 0; i < NVSHMEMT_LIBFABRIC_DEFAULT_NUM_EPS; i++) {
            libfabric_state->local_mr[i] = fabric_handle->hdls[i].mr;
            libfabric_state->local_mr_key[i] = fabric_handle->hdls[i].key;
            libfabric_state->local_mr_desc[i] = fabric_handle->hdls[i].local_desc;
        }
        libfabric_state->local_mem_ptr = buf;
    }

out:
    if (status) {
        if (handle_info) {
            if (libfabric_state->cache) {
                nvshmemt_mem_handle_cache_remove(t, libfabric_state->cache, buf);
            }
            free(handle_info);
        }
    }
    return status;
}

static int nvshmemt_libfabric_can_reach_peer(int *access,
                                             struct nvshmem_transport_pe_info *peer_info,
                                             nvshmem_transport_t t) {
    *access = NVSHMEM_TRANSPORT_CAP_CPU_WRITE | NVSHMEM_TRANSPORT_CAP_CPU_READ |
              NVSHMEM_TRANSPORT_CAP_CPU_ATOMICS;

    return 0;
}

/*
 * TODO: Make the following more general by using fid_nic field in fi_info.
 */
static int ib_iface_get_nic_path(const char *nic_name, const char *nic_class, char **path) {
    int status;

    char device_path[MAXPATHSIZE];
    status = snprintf(device_path, MAXPATHSIZE, "/sys/class/%s/%s/device", nic_class, nic_name);
    if (status < 0 || status >= MAXPATHSIZE) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                           "Unable to fill in device name.\n");
    } else {
        status = NVSHMEMX_SUCCESS;
    }

    *path = realpath(device_path, NULL);
    NVSHMEMI_NULL_ERROR_JMP(*path, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "realpath failed \n");

out:
    return status;
}

static int get_pci_path(int dev, char **pci_path, nvshmem_transport_t t) {
    int status = NVSHMEMX_SUCCESS;
    const char *nic_name, *nic_class;
    nvshmemt_libfabric_state_t *libfabric_state = (nvshmemt_libfabric_state_t *)t->state;

    if ((libfabric_state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_VERBS) ||
        (libfabric_state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA)) {
        nic_class = "infiniband";
    } else {
        nic_class = "cxi";
    }

    nic_name = (const char *)libfabric_state->domain_names[dev].name;

    status = ib_iface_get_nic_path(nic_name, nic_class, pci_path);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "get_pci_path failed \n");

out:
    return status;
}

static int nvshmemt_libfabric_connect_endpoints(nvshmem_transport_t t, int *selected_dev_ids,
                                                int num_selected_devs) {
    nvshmemt_libfabric_state_t *state = (nvshmemt_libfabric_state_t *)t->state;
    nvshmemt_libfabric_ep_name_t *all_ep_names = NULL;
    nvshmemt_libfabric_ep_name_t *local_ep_names = NULL;
    struct fi_info *current_fabric;
    struct fi_av_attr av_attr;
    struct fi_cq_attr cq_attr;
    struct fi_cntr_attr cntr_attr;
    size_t ep_namelen = NVSHMEMT_LIBFABRIC_EP_LEN;
    int status = 0;
    int total_num_eps;
    size_t num_recvs_per_pe = 0;
    int n_pes = t->n_pes;

    if (state->eps) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out_already_connected,
                           "Device already selected. libfabric only supports"
                           " one NIC per PE.\n");
    }

    state->eps = (nvshmemt_libfabric_endpoint_t *)calloc(NVSHMEMT_LIBFABRIC_DEFAULT_NUM_EPS,
                                                         sizeof(nvshmemt_libfabric_endpoint_t));
    NVSHMEMI_NULL_ERROR_JMP(state->eps, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate EPs.");

    current_fabric = state->all_prov_info;
    do {
        if (!strncmp(current_fabric->nic->device_attr->name,
                     state->domain_names[selected_dev_ids[0]].name,
                     NVSHMEMT_LIBFABRIC_DOMAIN_LEN)) {
            break;
        }
        current_fabric = current_fabric->next;
    } while (current_fabric != NULL);
    NVSHMEMI_NULL_ERROR_JMP(current_fabric, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "Unable to find the selected fabric.\n");

    state->prov_info = fi_dupinfo(current_fabric);

    if (state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
        state->num_sends =
            current_fabric->tx_attr->size * NVSHMEMT_LIBFABRIC_DEFAULT_NUM_EPS * t->n_pes;
        state->num_recvs =
            current_fabric->rx_attr->size * NVSHMEMT_LIBFABRIC_DEFAULT_NUM_EPS * t->n_pes;
        size_t elem_size = sizeof(nvshmemt_libfabric_gdr_op_ctx_t);

        num_recvs_per_pe = state->num_recvs / (t->n_pes * NVSHMEMT_LIBFABRIC_DEFAULT_NUM_EPS);

        state->send_buf = calloc(state->num_sends, elem_size);
        NVSHMEMI_NULL_ERROR_JMP(state->send_buf, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                "Unable to allocate EFA send buffers.\n");
        state->recv_buf = calloc(state->num_recvs, elem_size);
        NVSHMEMI_NULL_ERROR_JMP(state->recv_buf, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                "Unable to allocate EFA recv buffers.\n");

        nvshmemtLibfabricOpQueue.putToSendBulk((char *)state->send_buf, elem_size,
                                               state->num_sends);
    }

    status = fi_fabric(state->prov_info->fabric_attr, &state->fabric, NULL);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Failed to allocate fabric: %d: %s\n", status, fi_strerror(status * -1));
    ;

    status = fi_domain(state->fabric, state->prov_info, &state->domain, NULL);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Failed to allocate domain: %d: %s\n", status, fi_strerror(status * -1));

    t->max_op_len = state->prov_info->ep_attr->max_msg_size;

    av_attr.type = FI_AV_TABLE;
    av_attr.rx_ctx_bits = 0;
    av_attr.count = NVSHMEMT_LIBFABRIC_DEFAULT_NUM_EPS * n_pes;
    av_attr.ep_per_node = NVSHMEMT_LIBFABRIC_DEFAULT_NUM_EPS;
    av_attr.name = NULL;
    av_attr.map_addr = NULL;
    av_attr.flags = 0;

    /* Note - This is needed because EFA will only bind AVs to EPs on a 1:1 basis.
     * If EFA ever lifts this requirement, we can reduce the number of AVs required.
     */
    for (int i = 0; i < NVSHMEMT_LIBFABRIC_DEFAULT_NUM_EPS; i++) {
        status = fi_av_open(state->domain, &av_attr, &state->addresses[i], NULL);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Failed to allocate address vector: %d: %s\n", status,
                              fi_strerror(status * -1));
    }

    INFO(state->log_level, "Selected provider %s, fabric %s, nic %s, hmem %s",
         state->prov_info->fabric_attr->prov_name, state->prov_info->fabric_attr->name,
         state->prov_info->nic->device_attr->name, state->prov_info->caps & FI_HMEM ? "yes" : "no");

    assert(state->eps);

    memset(&cq_attr, 0, sizeof(struct fi_cq_attr));
    memset(&cntr_attr, 0, sizeof(struct fi_cntr_attr));

    state->prov_info->ep_attr->tx_ctx_cnt = 0;
    state->prov_info->caps = FI_RMA;
    if ((state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_SLINGSHOT) ||
        (state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_VERBS)) {
        state->prov_info->caps |= FI_ATOMIC;
    } else {
        state->prov_info->caps |= FI_MSG;
    }
    state->prov_info->tx_attr->op_flags = 0;
    state->prov_info->mode = 0;
    state->prov_info->tx_attr->mode = 0;
    state->prov_info->rx_attr->mode = 0;

    if ((state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_SLINGSHOT) ||
        (state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA)) {
        state->prov_info->tx_attr->op_flags = FI_TRANSMIT_COMPLETE;
    }

    cntr_attr.events = FI_CNTR_EVENTS_COMP;
    cntr_attr.wait_obj = FI_WAIT_UNSPEC;
    cntr_attr.wait_set = NULL;
    cntr_attr.flags = 0;

    if (state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_SLINGSHOT) {
        cq_attr.size = 16; /* CQ is only used to capture error events */
        cq_attr.format = FI_CQ_FORMAT_UNSPEC;
        cq_attr.wait_obj = FI_WAIT_NONE;
    }

    if (state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
        cq_attr.format = FI_CQ_FORMAT_MSG;
        cq_attr.wait_obj = FI_WAIT_NONE;
        /* Default size. */
        cq_attr.size = 0;
    }

    local_ep_names = (nvshmemt_libfabric_ep_name_t *)calloc(NVSHMEMT_LIBFABRIC_DEFAULT_NUM_EPS,
                                                            sizeof(nvshmemt_libfabric_ep_name_t));
    NVSHMEMI_NULL_ERROR_JMP(local_ep_names, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate array of endpoint names.");

    total_num_eps = NVSHMEMT_LIBFABRIC_DEFAULT_NUM_EPS * n_pes;
    all_ep_names =
        (nvshmemt_libfabric_ep_name_t *)calloc(total_num_eps, sizeof(nvshmemt_libfabric_ep_name_t));
    NVSHMEMI_NULL_ERROR_JMP(all_ep_names, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate array of endpoint names.");

    for (int i = 0; i < NVSHMEMT_LIBFABRIC_DEFAULT_NUM_EPS; i++) {
        status = fi_endpoint(state->domain, state->prov_info, &state->eps[i].endpoint, NULL);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Unable to allocate endpoint: %d: %s\n", status,
                              fi_strerror(status * -1));

        /* FI_OPT_CUDA_API_PERMITTED was introduced in libfabric 1.18.0 */
        if (state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
            bool prohibit_cuda_api = false;
            status = fi_setopt(&state->eps[i].endpoint->fid, FI_OPT_ENDPOINT,
                               FI_OPT_CUDA_API_PERMITTED, &prohibit_cuda_api, sizeof(bool));
            if (status == -FI_ENOPROTOOPT) {
                NVSHMEMI_WARN_PRINT(
                    "fi_setopt of FI_OPT_CUDA_API_PERMITTED returned as "
                    "not implemented.\n Not setting. This is expected for libfabric "
                    "versions < 1.18.\n");
            } else if (status) {
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                      "Unable to set endpoint CUDA API status: %d: %s\n", status,
                                      fi_strerror(status * -1));
            }
        }

        status = fi_cq_open(state->domain, &cq_attr, &state->eps[i].cq, NULL);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Unable to open completion queue for endpoint: %d: %s\n", status,
                              fi_strerror(status * -1));

        status = fi_cntr_open(state->domain, &cntr_attr, &state->eps[i].counter, NULL);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Unable to open counter for endpoint: %d: %s\n", status,
                              fi_strerror(status * -1));

        status = fi_ep_bind(state->eps[i].endpoint, &state->addresses[i]->fid, 0);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Unable to bind endpoint to address vector: %d: %s\n", status,
                              fi_strerror(status * -1));

        if (state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_VERBS) {
            status = fi_ep_bind(state->eps[i].endpoint, &state->eps[i].cq->fid,
                                FI_SELECTIVE_COMPLETION | FI_TRANSMIT | FI_RECV);
        } else if (state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_SLINGSHOT) {
            status = fi_ep_bind(state->eps[i].endpoint, &state->eps[i].cq->fid,
                                FI_SELECTIVE_COMPLETION | FI_TRANSMIT);
        } else if (state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
            /* EFA is documented as not supporting FI_SELECTIVE_COMPLETION */
            status =
                fi_ep_bind(state->eps[i].endpoint, &state->eps[i].cq->fid, FI_TRANSMIT | FI_RECV);
        } else {
            NVSHMEMI_ERROR_PRINT(
                "Invalid provider identified. This should be impossible. "
                "Possible memory corruption in the state pointer?");
            status = NVSHMEMX_ERROR_INTERNAL;
            goto out;
        }
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Unable to bind endpoint to completion queue: %d: %s\n", status,
                              fi_strerror(status * -1));

#ifdef NVSHMEM_USE_GDRCOPY
        if (use_gdrcopy) {
            status = fi_ep_bind(state->eps[i].endpoint, &state->eps[i].counter->fid,
                                FI_READ | FI_WRITE | FI_SEND);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "Unable to bind endpoint to completion counter: %d: %s\n", status,
                                  fi_strerror(status * -1));
        } else
#endif
        {
            int flags = FI_READ | FI_WRITE;
            if (use_staged_atomics) {
                flags |= FI_SEND;
            }
            status = fi_ep_bind(state->eps[i].endpoint, &state->eps[i].counter->fid, flags);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "Unable to bind endpoint to completion counter: %d: %s\n", status,
                                  fi_strerror(status * -1));
        }

        status = fi_enable(state->eps[i].endpoint);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Unable to enable endpoint: %d: %s\n", status,
                              fi_strerror(status * -1));

        if (state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
            for (size_t j = 0; j < num_recvs_per_pe; j++) {
                nvshmemt_libfabric_gdr_op_ctx_t *op;
                op = (nvshmemt_libfabric_gdr_op_ctx_t *)state->recv_buf;
                op = op + ((num_recvs_per_pe * i) + j);
                assert(op != NULL);
                status = fi_recv(state->eps[i].endpoint, op,
                                 sizeof(nvshmemt_libfabric_gdr_op_ctx_t), NULL, FI_ADDR_UNSPEC, op);
            }
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "Unable to post recv to ep. Error: %d: %s\n", status,
                                  fi_strerror(status * -1));
        }

        status = fi_getname(&state->eps[i].endpoint->fid, local_ep_names[i].name, &ep_namelen);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Unable to get name for endpoint: %d: %s\n", status,
                              fi_strerror(status * -1));
        if (ep_namelen > NVSHMEMT_LIBFABRIC_EP_LEN) {
            NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "Name of EP is too long.");
        }
    }

    status = t->boot_handle->allgather(
        local_ep_names, all_ep_names,
        NVSHMEMT_LIBFABRIC_DEFAULT_NUM_EPS * sizeof(nvshmemt_libfabric_ep_name_t), t->boot_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Failed to gather endpoint names.\n");

    /* We need to insert one at a time since each buffer is larger than the address. */
    for (int j = 0; j < NVSHMEMT_LIBFABRIC_DEFAULT_NUM_EPS; j++) {
        for (int i = 0; i < total_num_eps; i++) {
            status = fi_av_insert(state->addresses[j], &all_ep_names[i], 1, NULL, 0, NULL);
            if (status < 1) {
                NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                   "Unable to insert ep names in address vector: %d: %s\n", status,
                                   fi_strerror(status * -1));
            }

            status = NVSHMEMX_SUCCESS;
        }
    }

out:
    if (status != 0) {
        if (state->eps) {
            for (int i = 0; i < NVSHMEMT_LIBFABRIC_DEFAULT_NUM_EPS; i++) {
                if (state->eps[i].endpoint) {
                    fi_close(&state->eps[i].endpoint->fid);
                    state->eps[i].endpoint = NULL;
                }
                if (state->eps[i].cq) {
                    fi_close(&state->eps[i].cq->fid);
                    state->eps[i].cq = NULL;
                }
                if (state->eps[i].counter) {
                    fi_close(&state->eps[i].counter->fid);
                    state->eps[i].counter = NULL;
                }
            }
            free(state->eps);
            state->eps = NULL;
        }
    }

out_already_connected:
    free(local_ep_names);
    free(all_ep_names);

    return status;
}

static int nvshmemt_libfabric_finalize(nvshmem_transport_t transport) {
    nvshmemt_libfabric_state_t *libfabric_state;
    int status;

    assert(transport);

    libfabric_state = (nvshmemt_libfabric_state_t *)transport->state;

    if (transport->device_pci_paths) {
        for (int i = 0; i < transport->n_devices; i++) {
            free(transport->device_pci_paths[i]);
        }
        free(transport->device_pci_paths);
    }

    size_t mem_handle_cache_size;
    nvshmemt_libfabric_memhandle_info_t *handle_info = NULL;

    if (libfabric_state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
        mem_handle_cache_size = nvshmemt_mem_handle_cache_get_size(libfabric_state->cache);
        for (size_t i = 0; i < mem_handle_cache_size; i++) {
            handle_info =
                (nvshmemt_libfabric_memhandle_info_t *)nvshmemt_mem_handle_cache_get_by_idx(
                    libfabric_state->cache, i);
            if (handle_info) {
                free(handle_info);
            }
        }

        nvshmemt_mem_handle_cache_fini(libfabric_state->cache);
#ifdef NVSHMEM_USE_GDRCOPY
        if (use_gdrcopy) {
            nvshmemt_gdrcopy_ftable_fini(&gdrcopy_ftable, &gdr_desc, &gdrcopy_handle);
        }
#endif
    }

    if (libfabric_state->send_buf) free(libfabric_state->send_buf);
    if (libfabric_state->recv_buf) free(libfabric_state->recv_buf);
    if (libfabric_state->prov_info) {
        fi_freeinfo(libfabric_state->prov_info);
    }

    if (libfabric_state->eps) {
        for (int i = 0; i < NVSHMEMT_LIBFABRIC_DEFAULT_NUM_EPS; i++) {
            if (libfabric_state->eps[i].endpoint) {
                status = fi_close(&libfabric_state->eps[i].endpoint->fid);
                if (status) {
                    NVSHMEMI_WARN_PRINT("Unable to close fabric endpoint.: %d: %s\n", status,
                                        fi_strerror(status * -1));
                }
            }
            if (libfabric_state->eps[i].cq) {
                status = fi_close(&libfabric_state->eps[i].cq->fid);
                if (status) {
                    NVSHMEMI_WARN_PRINT("Unable to close fabric cq: %d: %s\n", status,
                                        fi_strerror(status * -1));
                }
            }
            if (libfabric_state->eps[i].counter) {
                status = fi_close(&libfabric_state->eps[i].counter->fid);
                if (status) {
                    NVSHMEMI_WARN_PRINT("Unable to close fabric counter: %d: %s\n", status,
                                        fi_strerror(status * -1));
                }
            }
        }
        free(libfabric_state->eps);
    }

    for (int i = 0; i < NVSHMEMT_LIBFABRIC_DEFAULT_NUM_EPS; i++) {
        status = fi_close(&libfabric_state->addresses[i]->fid);
        if (status) {
            NVSHMEMI_WARN_PRINT("Unable to close fabric address vector: %d: %s\n", status,
                                fi_strerror(status * -1));
        }
    }

    if (libfabric_state->domain) {
        status = fi_close(&libfabric_state->domain->fid);
        if (status) {
            NVSHMEMI_WARN_PRINT("Unable to close fabric domain: %d: %s\n", status,
                                fi_strerror(status * -1));
        }
    }

    if (libfabric_state->fabric) {
        status = fi_close(&libfabric_state->fabric->fid);
        if (status) {
            NVSHMEMI_WARN_PRINT("Unable to close fabric: %d: %s\n", status,
                                fi_strerror(status * -1));
        }
    }

    free(libfabric_state);

    free(transport);

    return 0;
}

static int nvshmemi_libfabric_init_state(nvshmem_transport_t t, nvshmemt_libfabric_state_t *state) {
    struct fi_info info;
    struct fi_tx_attr tx_attr;
    struct fi_rx_attr rx_attr;
    struct fi_ep_attr ep_attr;
    struct fi_domain_attr domain_attr;
    struct fi_fabric_attr fabric_attr;
    struct fid_nic nic;
    struct fi_av_attr av_attr;
    struct fi_info *returned_fabrics, *current_fabric;
    char *strncpy_output;
    int num_fabrics_returned = 0;

    int status = 0;

    memset(&ep_attr, 0, sizeof(struct fi_ep_attr));
    memset(&av_attr, 0, sizeof(struct fi_av_attr));
    memset(&info, 0, sizeof(struct fi_info));
    memset(&tx_attr, 0, sizeof(struct fi_tx_attr));
    memset(&rx_attr, 0, sizeof(struct fi_rx_attr));
    memset(&domain_attr, 0, sizeof(struct fi_domain_attr));
    memset(&fabric_attr, 0, sizeof(struct fi_fabric_attr));
    memset(&nic, 0, sizeof(struct fid_nic));

    info.tx_attr = &tx_attr;
    info.rx_attr = &rx_attr;
    info.ep_attr = &ep_attr;
    info.domain_attr = &domain_attr;
    info.fabric_attr = &fabric_attr;
    info.nic = &nic;

    info.addr_format = FI_FORMAT_UNSPEC;
    info.caps = FI_RMA | FI_HMEM;

    if (state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_VERBS) {
        info.caps |= FI_ATOMIC;
        domain_attr.mr_mode = FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY;
    } else if (state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_SLINGSHOT) {
        /* TODO: Use FI_FENCE to optimize put_with_signal */
        info.caps |= FI_FENCE | FI_ATOMIC;
        domain_attr.mr_mode = FI_MR_ENDPOINT | FI_MR_ALLOCATED | FI_MR_PROV_KEY;
    } else if (state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
        domain_attr.mr_mode = FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY | FI_MR_HMEM;
        info.caps |= FI_MSG;
    }

    /* Be thread safe at the level of the endpoint completion context. */
    domain_attr.threading = FI_THREAD_COMPLETION;

    ep_attr.type = FI_EP_RDM;  // Reliable datagrams

    status = fi_getinfo(FI_VERSION(NVSHMEMT_LIBFABRIC_MAJ_VER, NVSHMEMT_LIBFABRIC_MIN_VER), NULL,
                        NULL, 0, &info, &returned_fabrics);

    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "No providers matched fi_getinfo query: %d: %s\n", status,
                          fi_strerror(status * -1));
    state->all_prov_info = returned_fabrics;
    for (current_fabric = returned_fabrics; current_fabric != NULL;
         current_fabric = current_fabric->next) {
        num_fabrics_returned++;
    }

    state->domain_names = (nvshmemt_libfabric_domain_name_t *)calloc(
        num_fabrics_returned, sizeof(nvshmemt_libfabric_domain_name_t));
    NVSHMEMI_NULL_ERROR_JMP(state->domain_names, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate domain names.");

    /* Only select unique devices. */
    state->num_domains = 0;
    for (current_fabric = returned_fabrics; current_fabric != NULL;
         current_fabric = current_fabric->next) {
        if (!current_fabric->nic) {
            INFO(state->log_level,
                 "Interface did not return NIC structure to fi_getinfo. Skipping.\n");
            continue;
        }

        if (!current_fabric->tx_attr) {
            INFO(state->log_level,
                 "Interface did not return TX_ATTR structure to fi_getinfo. Skipping.\n");
            continue;
        }

        TRACE(state->log_level, "fi_getinfo returned provider %s, fabric %s, nic %s",
              current_fabric->fabric_attr->prov_name, current_fabric->fabric_attr->name,
              current_fabric->nic->device_attr->name);

        if (state->provider != NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
            if (current_fabric->tx_attr->inject_size < NVSHMEMT_LIBFABRIC_INJECT_BYTES) {
                INFO(state->log_level,
                     "Disabling interface due to insufficient inject data size. reported %lu, "
                     "expected "
                     "%u",
                     current_fabric->tx_attr->inject_size, NVSHMEMT_LIBFABRIC_INJECT_BYTES);
                continue;
            }
        }

        if ((current_fabric->domain_attr->mr_mode & FI_MR_PROV_KEY) == 0) {
            INFO(state->log_level, "Disabling interface due to FI_MR_PROV_KEY support");
            continue;
        }

        for (int i = 0; i <= state->num_domains; i++) {
            if (!strncmp(current_fabric->nic->device_attr->name, state->domain_names[i].name,
                         NVSHMEMT_LIBFABRIC_DOMAIN_LEN)) {
                break;
            } else if (i == state->num_domains) {
                strncpy_output =
                    strncpy(state->domain_names[state->num_domains].name,
                            current_fabric->nic->device_attr->name, NVSHMEMT_LIBFABRIC_DOMAIN_LEN);
                if (strncpy_output == NULL ||
                    (uintptr_t)strncpy_output -
                            (uintptr_t)state->domain_names[state->num_domains].name >=
                        NVSHMEMT_LIBFABRIC_DOMAIN_LEN) {
                    NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                       "Unable to copy domain name for libfabric transport.");
                }
                state->num_domains++;
                break;
            }
        }
    }

    t->n_devices = state->num_domains;
    t->device_pci_paths = (char **)calloc(t->n_devices, sizeof(char *));
    NVSHMEMI_NULL_ERROR_JMP(t->device_pci_paths, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "Unable to allocate paths for IB transport.");
    for (int i = 0; i < t->n_devices; i++) {
        status = get_pci_path(i, &t->device_pci_paths[i], t);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Failed to get paths for PCI devices.");
    }

out:
    if (status) {
        nvshmemt_libfabric_finalize(t);
    }

    return status;
}

int nvshmemt_init(nvshmem_transport_t *t, struct nvshmemi_cuda_fn_table *table, int api_version) {
    nvshmemt_libfabric_state_t *libfabric_state = NULL;
    nvshmem_transport_t transport = NULL;
    struct nvshmemi_options_s options;
    int status = 0;

    if (NVSHMEM_TRANSPORT_MAJOR_VERSION(api_version) != NVSHMEM_TRANSPORT_PLUGIN_MAJOR_VERSION) {
        NVSHMEMI_ERROR_PRINT(
            "NVSHMEM provided an incompatible version of the transport interface. "
            "This transport supports transport API major version %d. Host has %d",
            NVSHMEM_TRANSPORT_PLUGIN_MAJOR_VERSION, NVSHMEM_TRANSPORT_MAJOR_VERSION(api_version));
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    transport = (nvshmem_transport_t)calloc(1, sizeof(*transport));
    NVSHMEMI_NULL_ERROR_JMP(transport, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate memory for libfabric transport.");

    libfabric_state = (nvshmemt_libfabric_state_t *)calloc(1, sizeof(*libfabric_state));
    NVSHMEMI_NULL_ERROR_JMP(libfabric_state, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate memory for libfabric transport state.");
    libfabric_state->table = table;
    transport->state = libfabric_state;

    transport->host_ops.can_reach_peer = nvshmemt_libfabric_can_reach_peer;
    transport->host_ops.connect_endpoints = nvshmemt_libfabric_connect_endpoints;
    transport->host_ops.get_mem_handle = nvshmemt_libfabric_get_mem_handle;
    transport->host_ops.release_mem_handle = nvshmemt_libfabric_release_mem_handle;
    transport->host_ops.rma = nvshmemt_libfabric_rma;
    transport->host_ops.fence = nvshmemt_libfabric_quiet;
    transport->host_ops.quiet = nvshmemt_libfabric_quiet;
    transport->host_ops.finalize = nvshmemt_libfabric_finalize;
    transport->host_ops.show_info = nvshmemt_libfabric_show_info;
    transport->host_ops.progress = nvshmemt_libfabric_progress;
    transport->host_ops.enforce_cst = nvshmemt_libfabric_enforce_cst;

    transport->attr = NVSHMEM_TRANSPORT_ATTR_CONNECTED;
    transport->is_successfully_initialized = true;

    transport->api_version = api_version < NVSHMEM_TRANSPORT_INTERFACE_VERSION
                                 ? api_version
                                 : NVSHMEM_TRANSPORT_INTERFACE_VERSION;

    status = nvshmemi_env_options_init(&options);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Unable to initialize env options.");

    libfabric_state->log_level = nvshmemt_common_get_log_level(&options);

    if (strcmp(options.LIBFABRIC_PROVIDER, "verbs") == 0) {
        libfabric_state->provider = NVSHMEMT_LIBFABRIC_PROVIDER_VERBS;
    } else if (strcmp(options.LIBFABRIC_PROVIDER, "cxi") == 0) {
        libfabric_state->provider = NVSHMEMT_LIBFABRIC_PROVIDER_SLINGSHOT;
    } else if (strcmp(options.LIBFABRIC_PROVIDER, "efa") == 0) {
        libfabric_state->provider = NVSHMEMT_LIBFABRIC_PROVIDER_EFA;
    } else {
        NVSHMEMI_WARN_PRINT("Invalid libfabric transport persona '%s'\n",
                            options.LIBFABRIC_PROVIDER);
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }

    if (libfabric_state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
        transport->atomics_complete_on_quiet = false;
    } else {
        transport->atomics_complete_on_quiet = true;
    }

#ifdef NVSHMEM_USE_GDRCOPY
    if (options.DISABLE_GDRCOPY) {
        use_gdrcopy = false;
    } else if (libfabric_state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
        use_gdrcopy = nvshmemt_gdrcopy_ftable_init(&gdrcopy_ftable, &gdr_desc, &gdrcopy_handle,
                                                   libfabric_state->log_level);
        if (!use_gdrcopy) {
            INFO(libfabric_state->log_level,
                 "GDRCopy Initialization failed."
                 " Device memory will not be supported.\n");
        }
    } else {
        INFO(libfabric_state->log_level,
             "GDRCopy requested, but unused by transport. Disabling.\n");
        use_gdrcopy = false;
    }
#else
    if (libfabric_state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "EFA Provider requires GDRCopy, but it was disabled"
                           " at compile time.\n");
    }
#endif

    if (libfabric_state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
        use_staged_atomics = true;
        transport->host_ops.amo = nvshmemt_libfabric_gdr_amo;
    } else {
        transport->host_ops.amo = nvshmemt_libfabric_amo;
    }

#define NVSHMEMI_SET_ENV_VAR(varname, desired, warning_msg)                              \
    do {                                                                                 \
        const char *env_value = getenv(varname);                                         \
        if (env_value && strcmp(env_value, desired) != 0) {                              \
            NVSHMEMI_WARN_PRINT(warning_msg);                                            \
        } else if (!env_value) {                                                         \
            status = setenv(varname, desired, 1);                                        \
            if (status) {                                                                \
                NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,                 \
                                   "Failed to set environment variable %s.\n", varname); \
            }                                                                            \
        }                                                                                \
    } while (0)

    if (libfabric_state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_VERBS) {
        /* This MLX5 feature is known to cause issues with device memory read and atomic ops. */
        NVSHMEMI_SET_ENV_VAR(
            "MLX5_SCATTER_TO_CQE", "0",
            "MLX5_SCATTER_TO_CQE is set. This may cause issues with device memory read and "
            "atomic ops if the value is not 0.\n");
    }

    if (libfabric_state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
        NVSHMEMI_SET_ENV_VAR(
            "FI_EFA_USE_DEVICE_RDMA", "1",
            "FI_EFA_USE_DEVICE_RDMA is set. This may cause issues with initialization "
            "if the value is not 1.\n");
    }

    if (libfabric_state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_SLINGSHOT) {
        NVSHMEMI_SET_ENV_VAR("FI_CXI_DISABLE_HMEM_DEV_REGISTER", "1",
                             "FI_CXI_DISABLE_HMEM_DEV_REGISTER is set. This may cause issues with "
                             "initialization if the value is not 1.\n");

        NVSHMEMI_SET_ENV_VAR("FI_CXI_OPTIMIZED_MRS", "0",
                             "FI_CXI_OPTIMIZED_MRS is set. This may cause a hang at runtime "
                             "if the value is not 0.\n");
    }

    NVSHMEMI_SET_ENV_VAR("FI_HMEM_CUDA_USE_GDRCOPY", "1",
                         "FI_HMEM_CUDA_USE_GDRCOPY is set. This may cause issues at runtime "
                         "if the value is not 1.\n");

#undef NVSHMEMI_SET_ENV_VAR

    /* Prepare fabric state information. */
    status = nvshmemi_libfabric_init_state(transport, libfabric_state);
    if (status) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out_clean,
                           "Failed to initialize the libfabric state.\n");
    }

    *t = transport;
out:
    if (status) {
        if (transport) {
            nvshmemt_libfabric_finalize(transport);
        }
    }

out_clean:
    return status;
}
