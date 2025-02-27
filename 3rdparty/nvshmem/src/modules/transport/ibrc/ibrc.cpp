/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include <assert.h>
#include <stdint.h>  // IWYU pragma: keep
// IWYU pragma: no_include <bits/stdint-uintn.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
// IWYU pragma: no_include <mm_malloc.h>
#include <string.h>
#include <unistd.h>
#include <deque>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "internal/host_transport/cudawrap.h"
#include "bootstrap_host_transport/env_defs_internal.h"
#ifdef NVSHMEM_USE_GDRCOPY
#include "gdrapi.h"
#endif
#include "infiniband/verbs.h"
#include "non_abi/nvshmem_build_options.h"
#include "device_host_transport/nvshmem_common_transport.h"
#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"
#include "internal/host_transport/nvshmemi_transport_defines.h"
#include "non_abi/nvshmemx_error.h"
#include "internal/host_transport/transport.h"
#include "transport_common.h"
#ifdef NVSHMEM_USE_GDRCOPY
#include "transport_gdr_common.h"
#endif
#include "transport_ib_common.h"

#ifdef NVSHMEM_X86_64
#include <immintrin.h>  // IWYU pragma: keep
#endif
// IWYU pragma: no_include <xmmintrin.h>

#define IBRC_MAX_INLINE_SIZE 128

int ibrc_srq_depth;
#define IBRC_SRQ_MASK (ibrc_srq_depth - 1)

int ibrc_qp_depth;
#define IBRC_REQUEST_QUEUE_MASK (ibrc_qp_depth - 1)
#define IBRC_BUF_SIZE 64

#if defined(NVSHMEM_X86_64)
#define IBRC_CACHELINE 64
#elif defined(NVSHMEM_PPC64LE)
#define IBRC_CACHELINE 128
#elif defined(NVSHMEM_AARCH64)
#define IBRC_CACHELINE 64
#else
#error Unknown cache line size
#endif

#define MAX_NUM_HCAS 16
#define MAX_NUM_PORTS 4
#define MAX_NUM_PES_PER_NODE 32
#ifdef NVSHMEM_USE_GDRCOPY
#define BAR_READ_BUFSIZE (2 * 1024 * 1024)
#else
#define BAR_READ_BUFSIZE (sizeof(uint64_t))
#endif

enum { WAIT_ANY = 0, WAIT_ALL = 1 };

int NVSHMEMT_IBRC_MAX_RD_ATOMIC; /* Maximum number of RDMA Read & Atomic operations that can be
                                  * outstanding per QP
                                  */

struct ibrc_request {
    struct ibv_send_wr sr;
    struct ibv_send_wr *bad_sr;
    struct ibv_sge sge;
};

struct ibrc_atomic_op {
    nvshmemi_amo_t op;
    void *addr;
    void *retaddr;
    uint32_t retrkey;
    uint64_t retflag;
    uint32_t elembytes;
    uint64_t compare;
    uint64_t swap_add;
};

typedef struct ibrc_buf {
    struct ibv_recv_wr rwr;
    struct ibv_recv_wr *bad_rwr;
    struct ibv_sge sge;
    int qp_num;
    char buf[IBRC_BUF_SIZE];
} ibrc_buf_t;
ibrc_buf_t *bpool;
int bpool_size;
static std::vector<void *> bpool_free;
static std::deque<void *> bqueue_toprocess;

struct ibrc_device {
    struct ibv_device *dev;
    struct ibv_context *context;
    struct ibv_pd *pd;
    struct ibv_device_attr device_attr;
    struct ibv_port_attr port_attr[MAX_NUM_PORTS];
    struct nvshmemt_ib_gid_info gid_info[MAX_NUM_PORTS];
    // bpool information
    struct ibv_srq *srq;
    int srq_posted;
    struct ibv_mr *bpool_mr;
    struct ibv_cq *recv_cq;
    struct ibv_cq *send_cq;
};

struct ibrc_ep {
    int devid;
    int portid;
    struct ibv_qp *qp;
    struct ibv_cq *send_cq;
    struct ibv_cq *recv_cq;
    struct ibrc_request *req;
    volatile uint64_t head_op_id;
    volatile uint64_t tail_op_id;
    void *transport;
};

struct ibrc_ep_handle {
    uint32_t qpn;
    uint16_t lid;
    // ROCE
    uint64_t spn;
    uint64_t iid;
};

typedef struct {
    void *devices;
    int *dev_ids;
    int *port_ids;
    int n_dev_ids;
    int proxy_ep_idx;
    int ep_count;
    int selected_dev_id;
    int log_level;
    bool dmabuf_support;
    struct ibrc_ep **ep;
    struct transport_mem_handle_info_cache *cache;
    struct nvshmemi_options_s *options;
    struct nvshmemi_cuda_fn_table *table;
} transport_ibrc_state_t;

typedef struct ibrc_mem_handle_info {
    struct ibv_mr *mr;
    void *ptr;
    size_t size;
#ifdef NVSHMEM_USE_GDRCOPY
    void *cpu_ptr;
    void *cpu_ptr_base;
    gdr_mh_t mh;
#endif
} ibrc_mem_handle_info_t;
ibrc_mem_handle_info_t *dummy_local_mem;
pthread_mutex_t ibrc_mutex_recv_progress;
pthread_mutex_t ibrc_mutex_send_progress;

static std::map<unsigned int, long unsigned int> qp_map;
static uint64_t connected_qp_count;

struct ibrc_ep *ibrc_cst_ep;
static int use_ib_native_atomics = 1;
static bool use_gdrcopy = 0;
#ifdef NVSHMEM_USE_GDRCOPY
static gdr_t gdr_desc;
static struct gdrcopy_function_table gdrcopy_ftable;
static void *gdrcopy_handle = NULL;
static volatile uint64_t atomics_received = 0;
static volatile uint64_t atomics_processed = 0;
static volatile uint64_t atomics_issued = 0;
static volatile uint64_t atomics_completed = 0;
static volatile uint64_t atomics_acked = 0;
#endif

static struct nvshmemt_ibv_function_table ftable;
static void *ibv_handle;

int check_poll_avail(struct ibrc_ep *ep, int wait_predicate);
int progress_send(transport_ibrc_state_t *ibrc_state);
int poll_recv(transport_ibrc_state_t *ibrc_state);

// Allocate memory to be potentially ibv_reg_mr'd. This needs to be
// // allocated on separate pages as those pages will be marked DONTFORK
// // and if they are shared, that could cause a crash in a child process
static int nvshmemi_ib_malloc_debug(void **ptr, size_t size, int log_level, const char *filefunc,
                                    int line) {
    size_t page_size = sysconf(_SC_PAGESIZE);
    void *p;
    int size_aligned = ROUNDUP(size, page_size);
    int ret = posix_memalign(&p, page_size, size_aligned);
    if (ret != 0) return -1;
    memset(p, 0, size);
    *ptr = p;
    INFO(log_level, "%s:%d Ib Alloc Size %ld pointer %p", filefunc, line, size, *ptr);
    return 0;
}
#define nvshmemi_ib_malloc(...) nvshmemi_ib_malloc_debug(__VA_ARGS__, __FILE__, __LINE__)

ibrc_mem_handle_info_t *get_mem_handle_info(nvshmem_transport_t t, void *gpu_ptr) {
    transport_ibrc_state_t *ibrc_state = (transport_ibrc_state_t *)t->state;

    return (ibrc_mem_handle_info_t *)nvshmemt_mem_handle_cache_get(t, ibrc_state->cache, gpu_ptr);
}

inline int refill_srq(struct ibrc_device *device) {
    int status = 0;

    while ((device->srq_posted < ibrc_srq_depth) && !bpool_free.empty()) {
        ibrc_buf_t *buf = (ibrc_buf_t *)bpool_free.back();

        buf->rwr.next = NULL;
        buf->rwr.wr_id = (uint64_t)buf;
        buf->rwr.sg_list = &(buf->sge);
        buf->rwr.num_sge = 1;

        buf->sge.addr = (uint64_t)buf->buf;
        buf->sge.length = IBRC_BUF_SIZE;
        buf->sge.lkey = device->bpool_mr->lkey;

        status = ibv_post_srq_recv(device->srq, &buf->rwr, &buf->bad_rwr);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_post_srq_recv failed \n");

        bpool_free.pop_back();
        device->srq_posted++;
    }

out:
    return status;
}

int nvshmemt_ibrc_show_info(struct nvshmem_transport *transport, int style) {
    NVSHMEMI_ERROR_PRINT("ibrc show info not implemented");
    return 0;
}

static int get_pci_path(int dev, char **pci_path, nvshmem_transport_t t) {
    int status = NVSHMEMX_SUCCESS;

    struct nvshmem_transport *transport = (struct nvshmem_transport *)t;
    transport_ibrc_state_t *ibrc_state = (transport_ibrc_state_t *)transport->state;
    int dev_id = ibrc_state->dev_ids[dev];
    const char *ib_name =
        (const char *)((struct ibrc_device *)ibrc_state->devices)[dev_id].dev->name;

    status = nvshmemt_ib_iface_get_mlx_path(ib_name, pci_path);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "nvshmemt_ib_iface_get_mlx_path failed \n");

out:
    return status;
}

int nvshmemt_ibrc_can_reach_peer(int *access, struct nvshmem_transport_pe_info *peer_info,
                                 nvshmem_transport_t t) {
    int status = 0;

    *access = NVSHMEM_TRANSPORT_CAP_CPU_WRITE | NVSHMEM_TRANSPORT_CAP_CPU_READ |
              NVSHMEM_TRANSPORT_CAP_CPU_ATOMICS;

    return status;
}

static int ep_create(nvshmem_transport_t t, struct ibrc_ep **ep_ptr, int devid,
                     transport_ibrc_state_t *ibrc_state) {
    int status = 0;
    struct ibrc_ep *ep = NULL;
    struct ibv_qp_init_attr init_attr;
    struct ibv_qp_attr attr;
    int flags;
    struct ibrc_device *device =
        ((struct ibrc_device *)ibrc_state->devices + ibrc_state->dev_ids[devid]);
    int portid = ibrc_state->port_ids[devid];
    struct ibv_context *context = device->context;
    struct ibv_pd *pd = device->pd;

    // algining ep structure to prevent split tranactions when accessing head_op_id and
    // tail_op_id which can be used in inter-thread synchronization
    // TODO: use atomic variables instead to rely on language memory model guarantees
    status = posix_memalign((void **)&ep, IBRC_CACHELINE, sizeof(struct ibrc_ep));
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "ep allocation failed \n");
    memset((void *)ep, 0, sizeof(struct ibrc_ep));

    if (!device->send_cq) {
        device->send_cq = ftable.create_cq(context, device->device_attr.max_cqe, NULL, NULL, 0);
        NVSHMEMI_NULL_ERROR_JMP(device->send_cq, status, NVSHMEMX_ERROR_INTERNAL, out,
                                "cq creation failed \n");
    }
    assert(device->send_cq != NULL);
    ep->send_cq = device->send_cq;

    if (!device->srq) {
        struct ibv_srq_init_attr srq_init_attr;
        memset(&srq_init_attr, 0, sizeof(srq_init_attr));

        srq_init_attr.attr.max_wr = ibrc_srq_depth;
        srq_init_attr.attr.max_sge = 1;

        device->srq = ftable.create_srq(pd, &srq_init_attr);
        NVSHMEMI_NULL_ERROR_JMP(device->srq, status, NVSHMEMX_ERROR_INTERNAL, out,
                                "srq creation failed \n");

        device->recv_cq = ftable.create_cq(context, ibrc_srq_depth, NULL, NULL, 0);
        NVSHMEMI_NULL_ERROR_JMP(device->recv_cq, status, NVSHMEMX_ERROR_INTERNAL, out,
                                "cq creation failed \n");
    }
    assert(device->recv_cq != NULL);
    ep->recv_cq = device->recv_cq;

    memset(&init_attr, 0, sizeof(struct ibv_qp_init_attr));
    init_attr.srq = device->srq;
    init_attr.send_cq = ep->send_cq;
    init_attr.recv_cq = ep->recv_cq;
    init_attr.qp_type = IBV_QPT_RC;
    init_attr.cap.max_send_wr = ibrc_qp_depth;
    init_attr.cap.max_recv_wr = 0;
    init_attr.cap.max_send_sge = 1;
    init_attr.cap.max_recv_sge = 0;
    init_attr.cap.max_inline_data = IBRC_MAX_INLINE_SIZE;

    ep->qp = ftable.create_qp(pd, &init_attr);
    NVSHMEMI_NULL_ERROR_JMP(ep->qp, status, NVSHMEMX_ERROR_INTERNAL, out, "qp creation failed \n");

    memset(&attr, 0, sizeof(struct ibv_qp_attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = portid;
    attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_ATOMIC;
    flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;

    status = ftable.modify_qp(ep->qp, &attr, flags);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_modify_qp failed \n");

    ep->req = (struct ibrc_request *)malloc(sizeof(struct ibrc_request) * ibrc_qp_depth);
    NVSHMEMI_NULL_ERROR_JMP(ep->req, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "req allocation failed \n");
    ep->head_op_id = 0;
    ep->tail_op_id = 0;
    ep->transport = (void *)t;
    ep->devid = ibrc_state->dev_ids[devid];
    ep->portid = portid;

    // insert qp into map
    qp_map.insert(std::make_pair((unsigned int)ep->qp->qp_num, (long unsigned int)ep));

    *ep_ptr = ep;

out:
    if (status) {
        if (ep) {
            free(ep);
        }
    }
    return status;
}

static int ep_connect(struct ibrc_ep *ep, struct ibrc_ep_handle *ep_handle) {
    int status = 0;
    struct ibv_qp_attr attr;
    int flags;
    int devid = ep->devid;
    int portid = ep->portid;
    nvshmem_transport_t t = (nvshmem_transport_t)ep->transport;
    transport_ibrc_state_t *ibrc_state = (transport_ibrc_state_t *)t->state;
    struct ibrc_device *device = ((struct ibrc_device *)ibrc_state->devices + devid);
    struct ibv_port_attr *port_attr = device->port_attr + (portid - 1);

    memset(&attr, 0, sizeof(struct ibv_qp_attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = port_attr->active_mtu;
    attr.dest_qp_num = ep_handle->qpn;
    attr.rq_psn = 0;
    if (port_attr->lid == 0) {
        ib_get_gid_index(&ftable, device->context, portid, port_attr->gid_tbl_len,
                         &device->gid_info[portid - 1].local_gid_index, ibrc_state->log_level,
                         ibrc_state->options);
        ftable.query_gid(device->context, portid, device->gid_info[portid - 1].local_gid_index,
                         &device->gid_info[portid - 1].local_gid);
        attr.ah_attr.is_global = 1;
        attr.ah_attr.grh.dgid.global.subnet_prefix = ep_handle->spn;
        attr.ah_attr.grh.dgid.global.interface_id = ep_handle->iid;
        attr.ah_attr.grh.flow_label = 0;
        attr.ah_attr.grh.sgid_index = device->gid_info[portid - 1].local_gid_index;
        attr.ah_attr.grh.hop_limit = 255;
        attr.ah_attr.grh.traffic_class = ibrc_state->options->IB_TRAFFIC_CLASS;
    } else {
        attr.ah_attr.dlid = ep_handle->lid;
        attr.ah_attr.is_global = 0;
    }
    attr.max_dest_rd_atomic = NVSHMEMT_IBRC_MAX_RD_ATOMIC;
    attr.min_rnr_timer = 12;
    attr.ah_attr.sl = ibrc_state->options->IB_SL;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = portid;
    flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
            IBV_QP_MIN_RNR_TIMER | IBV_QP_MAX_DEST_RD_ATOMIC;

    status = ftable.modify_qp(ep->qp, &attr, flags);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_modify_qp failed \n");

    memset(&attr, 0, sizeof(struct ibv_qp_attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = 0;
    attr.timeout = 20;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.max_rd_atomic = NVSHMEMT_IBRC_MAX_RD_ATOMIC;
    flags = IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
            IBV_QP_MAX_QP_RD_ATOMIC;

    status = ftable.modify_qp(ep->qp, &attr, flags);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_modify_qp failed \n");

    // register and post receive buffer pool
    if (!device->bpool_mr) {
        device->bpool_mr = ftable.reg_mr(
            device->pd, bpool, bpool_size * sizeof(ibrc_buf_t),
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        NVSHMEMI_NULL_ERROR_JMP(device->bpool_mr, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                "mem registration failed \n");

        assert(device->srq != NULL);

        status = refill_srq(device);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "refill_srq failed \n");
    }

    connected_qp_count++;
out:
    return status;
}

int ep_get_handle(struct ibrc_ep_handle *ep_handle, struct ibrc_ep *ep) {
    int status = 0;
    nvshmem_transport_t t = (nvshmem_transport_t)ep->transport;
    transport_ibrc_state_t *ibrc_state = (transport_ibrc_state_t *)t->state;
    struct ibrc_device *device = ((struct ibrc_device *)ibrc_state->devices + ep->devid);

    ep_handle->lid = device->port_attr[ep->portid - 1].lid;
    ep_handle->qpn = ep->qp->qp_num;
    if (ep_handle->lid == 0) {
        ep_handle->spn = device->gid_info[ep->portid - 1].local_gid.global.subnet_prefix;
        ep_handle->iid = device->gid_info[ep->portid - 1].local_gid.global.interface_id;
    }

    return status;
}

int setup_cst_loopback(nvshmem_transport_t t, transport_ibrc_state_t *ibrc_state, int dev_id) {
    int status = 0;
    struct ibrc_ep_handle cst_ep_handle;

    status = ep_create(t, &ibrc_cst_ep, dev_id, ibrc_state);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ep_create cst failed \n");

    status = ep_get_handle(&cst_ep_handle, ibrc_cst_ep);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ep_get_handle failed \n");

    status = ep_connect(ibrc_cst_ep, &cst_ep_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ep_connect failed \n");
out:
    return status;
}
int nvshmemt_ibrc_get_mem_handle(nvshmem_mem_handle_t *mem_handle,
                                 nvshmem_mem_handle_t *mem_handle_in, void *buf, size_t length,
                                 nvshmem_transport_t t, bool local_only) {
    int status = 0;
    struct nvshmem_transport *transport = (struct nvshmem_transport *)t;
    transport_ibrc_state_t *ibrc_state = (transport_ibrc_state_t *)transport->state;
    struct ibrc_device *device = ((struct ibrc_device *)ibrc_state->devices +
                                  ibrc_state->dev_ids[ibrc_state->selected_dev_id]);
    struct ibrc_mem_handle_info *handle_info = NULL;
    struct nvshmemt_ib_common_mem_handle *handle;

    status = nvshmemt_ib_common_reg_mem_handle(
        &ftable, device->pd, mem_handle, buf, length, local_only, ibrc_state->dmabuf_support,
        ibrc_state->table, ibrc_state->log_level, ibrc_state->options->IB_ENABLE_RELAXED_ORDERING);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Unable to register memory handle.");

    handle = (struct nvshmemt_ib_common_mem_handle *)mem_handle;

    if (!local_only) {
        handle_info = (struct ibrc_mem_handle_info *)calloc(1, sizeof(struct ibrc_mem_handle_info));
        NVSHMEMI_NULL_ERROR_JMP(handle_info, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                "unable to allocate handle info.\n");

        handle_info->mr = handle->mr;
        handle_info->ptr = buf;
        handle_info->size = length;
    }

#ifdef NVSHMEM_USE_GDRCOPY
    if (use_gdrcopy && !local_only) {
        status =
            gdrcopy_ftable.pin_buffer(gdr_desc, (unsigned long)buf, length, 0, 0, &handle_info->mh);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gdrcopy pin_buffer failed \n");

        status = gdrcopy_ftable.map(gdr_desc, handle_info->mh, &handle_info->cpu_ptr_base, length);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gdrcopy map failed \n");

        gdr_info_t info;
        status = gdrcopy_ftable.get_info(gdr_desc, handle_info->mh, &info);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gdrcopy get_info failed \n");

        // remember that mappings start on a 64KB boundary, so let's
        // calculate the offset from the head of the mapping to the
        // beginning of the buffer
        uintptr_t off;
        off = (uintptr_t)buf - info.va;
        handle_info->cpu_ptr = (void *)((uintptr_t)handle_info->cpu_ptr_base + off);
    }
#endif

    /* The memory handle cache is only used with GDRCopy.
     * Local memory is never used with GDRCopy so it doesn't need
     * to go into the cache.
     * This optimization allows us to greatly simplify the lookup of
     * mem handle info when using the dynamic heap.
     */
    if (!local_only) {
        if (!ibrc_state->cache) {
            status = nvshmemt_mem_handle_cache_init(t, &ibrc_state->cache);
            NVSHMEMI_NZ_ERROR_JMP(status, status, out,
                                  "Unable to initialize mem handle cache in IB transport.");
        }
        status = nvshmemt_mem_handle_cache_add(t, ibrc_state->cache, buf, (void *)handle_info);
        NVSHMEMI_NZ_ERROR_JMP(status, status, out, "Unable to cache mem handle in IB transport.");
    }

    if (!dummy_local_mem) {
        dummy_local_mem = (ibrc_mem_handle_info_t *)malloc(sizeof(ibrc_mem_handle_info_t));
        NVSHMEMI_NULL_ERROR_JMP(dummy_local_mem, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                "dummy_local_mem allocation failed\n");

        nvshmemi_ib_malloc(&dummy_local_mem->ptr, sizeof(uint64_t), ibrc_state->log_level);
        NVSHMEMI_NULL_ERROR_JMP(dummy_local_mem->ptr, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                "dummy_mem allocation failed\n");

        dummy_local_mem->mr = ftable.reg_mr(device->pd, dummy_local_mem->ptr, sizeof(uint64_t),
                                            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                                IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
        NVSHMEMI_NULL_ERROR_JMP(dummy_local_mem->mr, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                "mem registration failed \n");
    }
out:
    if (status) {
        if (!local_only && ibrc_state->cache != NULL) {
            nvshmemt_mem_handle_cache_remove(t, ibrc_state->cache, buf);
            if (handle_info) {
                free(handle_info);
            }
        }
        nvshmemt_ib_common_release_mem_handle(&ftable, mem_handle, ibrc_state->log_level);
    }
    return status;
}

int nvshmemt_ibrc_release_mem_handle(nvshmem_mem_handle_t *mem_handle, nvshmem_transport_t t) {
    struct nvshmemt_ib_common_mem_handle *handle;
    struct ibrc_mem_handle_info *handle_info = NULL;
    transport_ibrc_state_t *state;
    void *addr;
    int status = 0;

    state = (transport_ibrc_state_t *)t->state;
    handle = (struct nvshmemt_ib_common_mem_handle *)mem_handle;
    addr = handle->buf;

    if (!handle->local_only) {
        handle_info =
            (ibrc_mem_handle_info_t *)nvshmemt_mem_handle_cache_get(t, state->cache, addr);
    }

    status = nvshmemt_ib_common_release_mem_handle(&ftable, mem_handle, state->log_level);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "Unable to dereg memory.\n");

    if (handle_info) {
#ifdef NVSHMEM_USE_GDRCOPY
        if (use_gdrcopy) {
            status = gdrcopy_ftable.unmap(gdr_desc, handle_info->mh, handle_info->cpu_ptr_base,
                                          handle_info->size);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gdr_unmap failed\n");

            status = gdrcopy_ftable.unpin_buffer(gdr_desc, handle_info->mh);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gdr_unpin failed\n");
        }
#endif

        if (state->cache != NULL) nvshmemt_mem_handle_cache_remove(t, state->cache, addr);
        free(handle_info);
    }
out:
    return status;
}

int nvshmemt_ibrc_finalize(nvshmem_transport_t transport) {
    int status = 0;
    size_t mem_handle_cache_size;
    transport_ibrc_state_t *state;
    struct ibrc_mem_handle_info *handle_info;

    state = (transport_ibrc_state_t *)transport->state;
    assert(state != NULL);
    mem_handle_cache_size = nvshmemt_mem_handle_cache_get_size(state->cache);

    if (transport->device_pci_paths) {
        for (int i = 0; i < transport->n_devices; i++) {
            free(transport->device_pci_paths[i]);
        }
        free(transport->device_pci_paths);
    }
    if (state->ep) {
        int ep_total_count = state->ep_count * transport->n_pes;
        for (int i = 0; i < ep_total_count; i++) {
            status = ftable.destroy_qp(state->ep[i]->qp);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_destroy_qp failed \n");
        }
        free(state->ep);
    }

    if (ibrc_cst_ep) {
        status = ftable.destroy_qp(ibrc_cst_ep->qp);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_destroy_qp failed \n");
        free(ibrc_cst_ep);
        ibrc_cst_ep = NULL;
    }

    for (size_t i = 0; i < mem_handle_cache_size; i++) {
        handle_info =
            (struct ibrc_mem_handle_info *)nvshmemt_mem_handle_cache_get_by_idx(state->cache, i);
        if (handle_info) {
#ifdef NVSHMEM_USE_GDRCOPY
            if (use_gdrcopy) {
                status = gdrcopy_ftable.unmap(gdr_desc, handle_info->mh, handle_info->cpu_ptr_base,
                                              handle_info->size);
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gdr_unmap failed\n");

                status = gdrcopy_ftable.unpin_buffer(gdr_desc, handle_info->mh);
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gdr_unpin failed\n");
            }
#endif
            free(handle_info);
        }
    }

    nvshmemt_mem_handle_cache_fini(state->cache);

#ifdef NVSHMEM_USE_GDRCOPY
    if (use_gdrcopy) {
        nvshmemt_gdrcopy_ftable_fini(&gdrcopy_ftable, &gdr_desc, &gdrcopy_handle);
    }
#endif

    // clear qp map
    qp_map.clear();

    if (dummy_local_mem) {
        status = ftable.dereg_mr(dummy_local_mem->mr);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_dereg_mr failed \n");
        free(dummy_local_mem);
        dummy_local_mem = NULL;
    }

    if (bpool != NULL) {
        while (!bpool_free.empty()) bpool_free.pop_back();

        free(bpool);
    }
    bqueue_toprocess.clear();

    status = pthread_mutex_destroy(&ibrc_mutex_send_progress);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "pthread_mutex_destroy failed\n");

    status = pthread_mutex_destroy(&ibrc_mutex_recv_progress);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "pthread_mutex_destroy failed\n");

#ifdef NVSHMEM_USE_GDRCOPY
    atomics_received = 0;
    atomics_processed = 0;
    atomics_issued = 0;
    atomics_completed = 0;
    atomics_acked = 0;
#endif
    connected_qp_count = 0;

    if (state->devices) {
        for (int i = 0; i < state->n_dev_ids; i++) {
            int dev_id = state->dev_ids[i];
            if (((struct ibrc_device *)state->devices)[dev_id].bpool_mr) {
                status = ftable.dereg_mr(((struct ibrc_device *)state->devices)[dev_id].bpool_mr);
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                      "ibv_dereg_mr failed \n");
            }
            if (((struct ibrc_device *)state->devices)[dev_id].send_cq) {
                status = ftable.destroy_cq(((struct ibrc_device *)state->devices)[dev_id].send_cq);
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                      "ibv_destroy_cq failed \n");
            }
            if (((struct ibrc_device *)state->devices)[dev_id].recv_cq) {
                status = ftable.destroy_cq(((struct ibrc_device *)state->devices)[dev_id].recv_cq);
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                      "ibv_destroy_cq failed \n");
            }
            if (((struct ibrc_device *)state->devices)[dev_id].srq) {
                status = ftable.destroy_srq(((struct ibrc_device *)state->devices)[dev_id].srq);
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                      "ibv_destroy_srq failed \n");
            }
            if (((struct ibrc_device *)state->devices)[dev_id].pd) {
                status = ftable.dealloc_pd(((struct ibrc_device *)state->devices)[dev_id].pd);
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                      "ibv_dealloc_pd failed \n");
            }
            if (((struct ibrc_device *)state->devices)[dev_id].context) {
                status =
                    ftable.close_device(((struct ibrc_device *)state->devices)[dev_id].context);
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                      "ibv_close_device failed \n");
            }
        }
        free(state->devices);
    }
    if (state->dev_ids) {
        free(state->dev_ids);
    }
    if (state->port_ids) {
        free(state->port_ids);
    }
    if (state->options) {
        free(state->options);
    }
    free(state);

    nvshmemt_ibv_ftable_fini(&ibv_handle);

out:
    return status;
}

#ifdef NVSHMEM_USE_GDRCOPY
template <typename T>
int perform_gdrcopy_amo(struct ibrc_ep *ep, gdr_mh_t mh, struct ibrc_atomic_op *op, void *ptr) {
    int status = 0;

    T old_value, new_value;
    // FIXME: gdrcopy causing duplicate copies for small transfers, using direct LD/ST until this
    // resolved
    // status = gdrcopy_ftable.copy_from_mapping(mh, &old_value, ptr, sizeof(T));
    // NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gdr copy from mapping
    // failed\n");
#if __cplusplus >= 201103L
    // assert size is 64-bit or smaller, issued as single tansaction
    static_assert(sizeof(T) <= 8, "static_assert(sizeof(T) >= 8) failed");
#endif
    old_value = *((volatile T *)ptr);

    switch (op->op) {
        case NVSHMEMI_AMO_SIGNAL:
        case NVSHMEMI_AMO_SIGNAL_SET:
        case NVSHMEMI_AMO_SET:
        case NVSHMEMI_AMO_SWAP: {
            /* The static_cast is used to truncate the uint64_t value of swap_add back to its
             * original length */
            new_value = static_cast<T>(op->swap_add);
            break;
        }
        case NVSHMEMI_AMO_ADD:
        case NVSHMEMI_AMO_SIGNAL_ADD:
        case NVSHMEMI_AMO_FETCH_ADD: {
            new_value = old_value + static_cast<T>(op->swap_add);
            break;
        }
        case NVSHMEMI_AMO_OR:
        case NVSHMEMI_AMO_FETCH_OR: {
            new_value = old_value | static_cast<T>(op->swap_add);
            break;
        }
        case NVSHMEMI_AMO_AND:
        case NVSHMEMI_AMO_FETCH_AND: {
            new_value = old_value & static_cast<T>(op->swap_add);
            break;
        }
        case NVSHMEMI_AMO_XOR:
        case NVSHMEMI_AMO_FETCH_XOR: {
            new_value = old_value ^ static_cast<T>(op->swap_add);
            break;
        }
        case NVSHMEMI_AMO_COMPARE_SWAP: {
            new_value = (old_value == static_cast<T>(op->compare)) ? static_cast<T>(op->swap_add)
                                                                   : old_value;
            break;
        }
        case NVSHMEMI_AMO_FETCH: {
            new_value = old_value;
            break;
        }
        default: {
            NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                               "RMA/AMO verb %d not implemented\n", op->op);
        }
    }

    // FIXME: gdrcopy causing duplicate copies for small transfers, using direct LD/ST until this
    // resolved status = gdrcopy_ftable.copy_to_mapping(mh, ptr, (void *)&new_value, sizeof(T));
    // NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gdr copy to mapping failed\n");
    *((volatile T *)ptr) = new_value;
    STORE_BARRIER();
    {
        nvshmem_transport_t t = (nvshmem_transport_t)ep->transport;
        transport_ibrc_state_t *ibrc_state = (transport_ibrc_state_t *)t->state;
        struct ibv_send_wr *sr, **bad_sr;
        struct ibv_sge *sge;
        int op_id;
        nvshmemi_amo_t ack;
        g_elem_t ret;

        // wait for one send request to become avaialble on the ep
        assert(ibrc_qp_depth >= 1);
        uint32_t outstanding_count = (ibrc_qp_depth - 1);
        while ((ep->head_op_id - ep->tail_op_id) > outstanding_count) {
            status = progress_send(ibrc_state);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "progress_send failed, outstanding_count: %d\n",
                                  outstanding_count);

            // already in processing a recv request
            // only poll recv cq
            status = poll_recv(ibrc_state);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "poll_recv failed, outstanding_count: %d\n", outstanding_count);
        }

        op_id = ep->head_op_id & IBRC_REQUEST_QUEUE_MASK;  // ep->head_op_id % ibrc_qp_depth
        ep->head_op_id++;

        sr = &(ep->req + op_id)->sr;
        bad_sr = &(ep->req + op_id)->bad_sr;
        sge = &(ep->req + op_id)->sge;

        memset(sr, 0, sizeof(ibv_send_wr));
        if (op->op > NVSHMEMI_AMO_END_OF_NONFETCH) {
            ret.data = ret.flag = 0;
            ret.data = old_value;
            ret.flag = op->retflag;

            sr->next = NULL;
            sr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
            sr->send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
            sr->wr_id = NVSHMEMI_AMO_END_OF_NONFETCH;
            sr->num_sge = 1;
            sr->sg_list = sge;

            sr->imm_data = (uint32_t)NVSHMEMI_AMO_ACK;
            sr->wr.rdma.remote_addr = (uint64_t)op->retaddr;
            sr->wr.rdma.rkey = op->retrkey;
            sge->length = sizeof(g_elem_t);
            sge->addr = (uintptr_t)&ret;
            sge->lkey = 0;
        } else {
            ack = NVSHMEMI_AMO_ACK;

            sr->next = NULL;
            sr->opcode = IBV_WR_SEND;
            sr->send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
            sr->wr_id = NVSHMEMI_AMO_ACK;
            sr->num_sge = 1;
            sr->sg_list = sge;

            // dummy send
            sge->length = sizeof(nvshmemi_amo_t);
            sge->addr = (uintptr_t)&ack;
            sge->lkey = 0;
        }

        status = ibv_post_send(ep->qp, sr, bad_sr);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_post_send failed \n");
    }

out:
    return status;
}

int poll_recv(transport_ibrc_state_t *ibrc_state) {
    int status = 0;
    int n_devs = ibrc_state->n_dev_ids;

    // poll all CQs available
    for (int i = 0; i < n_devs; i++) {
        struct ibv_wc wc;
        int devid = ibrc_state->dev_ids[i];
        struct ibrc_device *device = ((struct ibrc_device *)ibrc_state->devices + devid);

        if (!device->recv_cq) continue;

        int ne = ibv_poll_cq(device->recv_cq, 1, &wc);
        if (ne < 0) {
            status = ne;
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_poll_cq failed \n");
        } else if (ne) {
            assert(ne == 1);
            ibrc_buf_t *buf = (ibrc_buf_t *)wc.wr_id;
            if (wc.wc_flags & IBV_WC_WITH_IMM) {
                atomics_acked++;
                TRACE(ibrc_state->log_level, "[%d] atomic acked : %lu \n", getpid(), atomics_acked);
                bpool_free.push_back((void *)buf);
            } else {
                struct ibrc_atomic_op *op = (struct ibrc_atomic_op *)buf->buf;
                if (op->op == NVSHMEMI_AMO_ACK) {
                    atomics_acked++;
                    TRACE(ibrc_state->log_level, "[%d] atomic acked : %lu \n", getpid(),
                          atomics_acked);
                    bpool_free.push_back((void *)buf);
                } else {
                    buf->qp_num = wc.qp_num;
                    atomics_received++;
                    TRACE(ibrc_state->log_level, "[%d] atomic received, enqueued : %lu \n",
                          getpid(), atomics_received);
                    bqueue_toprocess.push_back((void *)buf);
                }
            }
            device->srq_posted--;
        }

        status = refill_srq(device);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "refill_sqr failed \n");
    }

out:
    return status;
}

int process_recv(nvshmem_transport_t t, transport_ibrc_state_t *ibrc_state) {
    int status = 0;

    if (!bqueue_toprocess.empty()) {
        ibrc_buf_t *buf = (ibrc_buf_t *)bqueue_toprocess.front();
        struct ibrc_ep *ep = (struct ibrc_ep *)qp_map.find((unsigned int)buf->qp_num)->second;
        struct ibrc_atomic_op *op = (struct ibrc_atomic_op *)buf->buf;
        ibrc_mem_handle_info_t *mem_handle_info = get_mem_handle_info(t, (void *)op->addr);
        void *ptr = (void *)((uintptr_t)mem_handle_info->cpu_ptr +
                             ((uintptr_t)op->addr - (uintptr_t)mem_handle_info->ptr));

        switch (op->elembytes) {
            case 2:
                perform_gdrcopy_amo<uint16_t>(ep, mem_handle_info->mh, op, ptr);
                break;
            case 4:
                perform_gdrcopy_amo<uint32_t>(ep, mem_handle_info->mh, op, ptr);
                break;
            case 8:
                perform_gdrcopy_amo<uint64_t>(ep, mem_handle_info->mh, op, ptr);
                break;
            default:
                NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                   "invalid element size encountered %u\n", op->elembytes);
        }
        atomics_processed++;
        TRACE(ibrc_state->log_level, "[%d] atomic dequeued and processed : %lu \n", getpid(),
              atomics_processed);

        bqueue_toprocess.pop_front();
        bpool_free.push_back((void *)buf);
    }

out:
    return status;
}

int progress_recv(nvshmem_transport_t t, transport_ibrc_state_t *ibrc_state) {
    int status = 0;

    pthread_mutex_lock(&ibrc_mutex_recv_progress);

    status = poll_recv(ibrc_state);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "poll recv failed \n");

    status = process_recv(t, ibrc_state);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "process recv failed \n");

out:
    pthread_mutex_unlock(&ibrc_mutex_recv_progress);
    return status;
}
#endif

int progress_send(transport_ibrc_state_t *ibrc_state) {
    int status = 0;
    int n_devs = ibrc_state->n_dev_ids;

    pthread_mutex_lock(&ibrc_mutex_send_progress);

    for (int i = 0; i < n_devs; i++) {
        struct ibv_wc wc;
        int devid = ibrc_state->dev_ids[i];
        struct ibrc_device *device = ((struct ibrc_device *)ibrc_state->devices + devid);

        if (!device->send_cq) continue;

        int ne = ibv_poll_cq(device->send_cq, 1, &wc);
        if (ne < 0) {
            status = ne;
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_poll_cq failed \n");
        } else if (ne) {
            if (wc.status) {
                status = wc.status;
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                      "ibv_poll_cq failed, status: %d\n", wc.status);
            }

            assert(ne == 1);
            if (wc.wr_id == NVSHMEMI_OP_AMO) {
#ifdef NVSHMEM_USE_GDRCOPY
                atomics_completed++;
                TRACE(ibrc_state->log_level, "[%d] atomic completed : %lu \n", getpid(),
                      atomics_completed);
#endif
            }

            struct ibrc_ep *ep = (struct ibrc_ep *)qp_map.find((unsigned int)wc.qp_num)->second;
            ep->tail_op_id += ne;
        }
    }

out:
    pthread_mutex_unlock(&ibrc_mutex_send_progress);
    return status;
}

int nvshmemt_ibrc_progress(nvshmem_transport_t t) {
    int status = 0;
    struct nvshmem_transport *transport = (struct nvshmem_transport *)t;
    transport_ibrc_state_t *ibrc_state = (transport_ibrc_state_t *)transport->state;

    status = progress_send(ibrc_state);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "progress_send failed, \n");

#ifdef NVSHMEM_USE_GDRCOPY
    status = progress_recv(t, ibrc_state);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "progress failed \n");
#endif

out:
    return status;
}

int check_poll_avail(struct ibrc_ep *ep, int wait_predicate) {
    int status = 0;
    uint32_t outstanding_count;

    assert(ibrc_qp_depth >= 1);
    outstanding_count = (ibrc_qp_depth - 1);
    if (wait_predicate == WAIT_ALL) outstanding_count = 0;
    nvshmem_transport_t t = (nvshmem_transport_t)ep->transport;
    transport_ibrc_state_t *ibrc_state = (transport_ibrc_state_t *)t->state;

    /* poll until space becomes in local send qp and space in receive qp at target for atomics
     * assuming connected qp cout is symmetric across all processes,
     * connected_qp_count+1 to avoid completely emptying the recv qp at target, leading to perf
     * issues*/
    while (((ep->head_op_id - ep->tail_op_id) > outstanding_count)
#ifdef NVSHMEM_USE_GDRCOPY
           || ((atomics_issued - atomics_acked) > (ibrc_srq_depth / (connected_qp_count + 1)))
#endif
    ) {
        status = progress_send(ibrc_state);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "progress_send failed, outstanding_count: %d\n", outstanding_count);

#ifdef NVSHMEM_USE_GDRCOPY
        status = progress_recv(t, ibrc_state);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "progress_recv failed \n");
#endif
    }

out:
    return status;
}

int nvshmemt_ibrc_rma(struct nvshmem_transport *tcurr, int pe, rma_verb_t verb,
                      rma_memdesc_t *remote, rma_memdesc_t *local, rma_bytesdesc_t bytesdesc,
                      int is_proxy) {
    int status = 0;
    transport_ibrc_state_t *ibrc_state = (transport_ibrc_state_t *)tcurr->state;
    struct ibv_send_wr *sr, **bad_sr;
    struct ibrc_ep *ep;
    struct ibv_sge *sge;
    int op_id;

    if (is_proxy) {
        ep = ibrc_state->ep[(ibrc_state->ep_count * pe + ibrc_state->proxy_ep_idx)];
    } else {
        ep = ibrc_state->ep[(ibrc_state->ep_count * pe)];
    }

    status = check_poll_avail(ep, WAIT_ANY);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "check_poll failed \n");

    op_id = ep->head_op_id & IBRC_REQUEST_QUEUE_MASK;  // ep->head_op_id % ibrc_qp_depth

    sr = &(ep->req + op_id)->sr;
    bad_sr = &(ep->req + op_id)->bad_sr;
    sge = &(ep->req + op_id)->sge;

    memset(sr, 0, sizeof(ibv_send_wr));

    sr->next = NULL;
    sr->send_flags = IBV_SEND_SIGNALED;
    sr->wr_id = NVSHMEMI_OP_PUT;
    sr->num_sge = 1;
    sr->sg_list = sge;

    sr->wr.rdma.remote_addr = (uint64_t)remote->ptr;
    assert(remote->handle);
    sr->wr.rdma.rkey = ((struct nvshmemt_ib_common_mem_handle *)remote->handle)->rkey;
    sge->length = bytesdesc.nelems * bytesdesc.elembytes;
    sge->addr = (uintptr_t)local->ptr;
    /* local->handle is unset for p operations since they are sent by value. */
    if (likely(local->handle != NULL)) {
        sge->lkey = ((struct nvshmemt_ib_common_mem_handle *)local->handle)->lkey;
    }
    if (verb.desc == NVSHMEMI_OP_P) {
        sr->opcode = IBV_WR_RDMA_WRITE;
        sr->send_flags |= IBV_SEND_INLINE;
        TRACE(ibrc_state->log_level, "[PUT] remote_addr %p addr %p rkey %d lkey %d length %x",
              (void *)sr->wr.rdma.remote_addr, (void *)sge->addr, sr->wr.rdma.rkey, sge->lkey,
              sge->length);
    } else if (verb.desc == NVSHMEMI_OP_GET || verb.desc == NVSHMEMI_OP_G) {
        sr->opcode = IBV_WR_RDMA_READ;
        TRACE(ibrc_state->log_level, "[GET] remote_addr %p addr %p rkey %d lkey %d length %x",
              (void *)sr->wr.rdma.remote_addr, (void *)sge->addr, sr->wr.rdma.rkey, sge->lkey,
              sge->length);
    } else if (verb.desc == NVSHMEMI_OP_PUT) {
        sr->opcode = IBV_WR_RDMA_WRITE;
        TRACE(ibrc_state->log_level, "[PUT] remote_addr %p addr %p rkey %d lkey %d length %x",
              (void *)sr->wr.rdma.remote_addr, (void *)sge->addr, sr->wr.rdma.rkey, sge->lkey,
              sge->length);
    } else {
        NVSHMEMI_ERROR_PRINT("RMA/AMO verb not implemented\n");
        exit(-1);
    }

    TRACE(ibrc_state->log_level, "[%d] ibrc post_send dest handle %p rkey %x src handle %p lkey %x",
          getpid(), remote->handle, sr->wr.rdma.rkey, local->handle, sge->lkey);
    status = ibv_post_send(ep->qp, sr, bad_sr);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_post_send failed \n");

    ep->head_op_id++;

    if (unlikely(!verb.is_nbi && verb.desc != NVSHMEMI_OP_P)) {
        check_poll_avail(ep, WAIT_ALL /*1*/);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "check_poll failed \n");
    }
out:
    return status;
}

int nvshmemt_ibrc_amo(struct nvshmem_transport *tcurr, int pe, void *curetptr, amo_verb_t verb,
                      amo_memdesc_t *remote, amo_bytesdesc_t bytesdesc, int is_proxy) {
    int status = 0;
    transport_ibrc_state_t *ibrc_state = (transport_ibrc_state_t *)tcurr->state;
    struct ibrc_ep *ep;
    struct ibv_send_wr *sr, **bad_sr;
    struct ibv_sge *sge;
    int op_id;
    struct ibrc_atomic_op op;

    if (is_proxy) {
        ep = ibrc_state->ep[(ibrc_state->ep_count * pe + ibrc_state->proxy_ep_idx)];
    } else {
        ep = ibrc_state->ep[(ibrc_state->ep_count * pe)];
    }

    status = check_poll_avail(ep, WAIT_ANY);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "check_poll failed \n");

    op_id = ep->head_op_id & IBRC_REQUEST_QUEUE_MASK;  // ep->head_op_id % ibrc_qp_depth
    sr = &(ep->req + op_id)->sr;
    bad_sr = &(ep->req + op_id)->bad_sr;
    sge = &(ep->req + op_id)->sge;

    memset(sr, 0, sizeof(ibv_send_wr));
    memset(sge, 0, sizeof(ibv_sge));

    sr->num_sge = 1;
    sr->sg_list = sge;
    sr->wr_id = NVSHMEMI_OP_AMO;
    sr->next = NULL;

    if (use_ib_native_atomics) {
        if (verb.desc == NVSHMEMI_AMO_SIGNAL_ADD) {
            if (bytesdesc.elembytes == 8) {
                sr->opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
                sr->send_flags = IBV_SEND_SIGNALED;

                sr->wr.atomic.remote_addr = (uint64_t)remote->remote_memdesc.ptr;
                assert(remote->remote_memdesc.handle);
                sr->wr.atomic.rkey =
                    ((struct nvshmemt_ib_common_mem_handle *)remote->remote_memdesc.handle)->rkey;
                sr->wr.atomic.compare_add = remote->val;

                sge->length = bytesdesc.elembytes;
                sge->addr = (uintptr_t)dummy_local_mem->ptr;
                sge->lkey = dummy_local_mem->mr->lkey;
                goto post_op;
            }
        }
    }

#ifdef NVSHMEM_USE_GDRCOPY
    // if gdrcopy is available, use it for all atomics to guarantee
    // atomicity across different ops
    if (use_gdrcopy) {
        ibrc_mem_handle_info_t *mem_handle_info;

        // assuming GDRCopy availability is uniform on all nodes
        op.op = verb.desc;
        op.addr = remote->remote_memdesc.ptr;
        op.retaddr = remote->retptr;
        op.retflag = remote->retflag;
        op.compare = remote->cmp;
        op.swap_add = remote->val;
        op.elembytes = bytesdesc.elembytes;

        // send rkey info
        if (verb.desc > NVSHMEMI_AMO_END_OF_NONFETCH) {
            mem_handle_info = get_mem_handle_info(tcurr, remote->retptr);
            op.retrkey = mem_handle_info->mr->rkey;
        }

        sr->opcode = IBV_WR_SEND;
        sr->send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
        sge->length = sizeof(struct ibrc_atomic_op);
        assert(sge->length <= IBRC_BUF_SIZE);
        sge->addr = (uintptr_t)&op;
        sge->lkey = 0;

        atomics_issued++;
        TRACE(ibrc_state->log_level, "[%d] atomic issued : %lu \n", getpid(), atomics_issued);
        goto post_op;
    }
#endif

    if (use_ib_native_atomics) {
        if (verb.desc == NVSHMEMI_AMO_ADD) {
            if (bytesdesc.elembytes == 8) {
                sr->opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
                sr->send_flags = IBV_SEND_SIGNALED;

                sr->wr.atomic.remote_addr = (uint64_t)remote->remote_memdesc.ptr;
                assert(remote->remote_memdesc.handle);
                sr->wr.atomic.rkey =
                    ((struct nvshmemt_ib_common_mem_handle *)remote->remote_memdesc.handle)->rkey;
                sr->wr.atomic.compare_add = remote->val;

                sge->length = bytesdesc.elembytes;
                sge->addr = (uintptr_t)dummy_local_mem->ptr;
                sge->lkey = dummy_local_mem->mr->lkey;
                goto post_op;
            }
        } else if (verb.desc == NVSHMEMI_AMO_SIGNAL || verb.desc == NVSHMEMI_AMO_SIGNAL_SET) {
            sr->opcode = IBV_WR_RDMA_WRITE;
            sr->send_flags = IBV_SEND_SIGNALED;
            sr->send_flags |= IBV_SEND_INLINE;

            sr->wr.rdma.remote_addr = (uint64_t)remote->remote_memdesc.ptr;
            assert(remote->remote_memdesc.handle);
            sr->wr.rdma.rkey =
                ((struct nvshmemt_ib_common_mem_handle *)remote->remote_memdesc.handle)->rkey;

            sge->length = bytesdesc.elembytes;
            sge->addr = (uintptr_t)&remote->val;
            sge->lkey = 0;
            goto post_op;
        }
    }

    NVSHMEMI_ERROR_EXIT("RMA/AMO verb %d not implemented\n", verb.desc);

post_op:
    status = ibv_post_send(ep->qp, sr, bad_sr);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_post_send failed \n");

    ep->head_op_id++;

out:
    return status;
}

int nvshmemt_ibrc_enforce_cst_at_target(struct nvshmem_transport *tcurr) {
    int status = 0;
    ibrc_mem_handle_info_t *mem_handle_info;
    transport_ibrc_state_t *state;

    state = (transport_ibrc_state_t *)tcurr->state;

    // pick the last region that was inserted
    mem_handle_info =
        (ibrc_mem_handle_info_t *)nvshmemt_mem_handle_cache_get_by_idx(state->cache, 0);
    assert(mem_handle_info != NULL);

#ifdef NVSHMEM_USE_GDRCOPY
    if (use_gdrcopy) {
        int temp;
        gdrcopy_ftable.copy_from_mapping(mem_handle_info->mh, &temp, mem_handle_info->cpu_ptr,
                                         sizeof(int));
        return status;
    }
#endif

    struct ibrc_ep *ep = ibrc_cst_ep;
    struct ibv_send_wr *sr, **bad_sr;
    struct ibv_sge *sge;
    int op_id;

    op_id = ep->head_op_id & IBRC_REQUEST_QUEUE_MASK;  // ep->head_op_id % ibrc_qp_depth
    sr = &(ep->req + op_id)->sr;
    bad_sr = &(ep->req + op_id)->bad_sr;
    sge = &(ep->req + op_id)->sge;

    sr->next = NULL;
    sr->send_flags = IBV_SEND_SIGNALED;
    sr->num_sge = 1;
    sr->sg_list = sge;

    sr->opcode = IBV_WR_RDMA_READ;
    sr->wr.rdma.remote_addr = (uint64_t)mem_handle_info->ptr;
    sr->wr.rdma.rkey = mem_handle_info->mr->rkey;

    sge->length = sizeof(int);
    sge->addr = (uintptr_t)mem_handle_info->ptr;
    sge->lkey = mem_handle_info->mr->lkey;

    status = ibv_post_send(ep->qp, sr, bad_sr);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_post_send failed \n");

    ep->head_op_id++;

    status = check_poll_avail(ep, WAIT_ALL);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "check_poll failed \n");

out:
    return status;
}

int nvshmemt_ibrc_quiet(struct nvshmem_transport *tcurr, int pe, int is_proxy) {
    transport_ibrc_state_t *ibrc_state = (transport_ibrc_state_t *)tcurr->state;
    struct ibrc_ep *ep;
    int status = 0;

    if (is_proxy) {
        ep = ibrc_state->ep[(pe * ibrc_state->ep_count + ibrc_state->proxy_ep_idx)];
    } else {
        ep = ibrc_state->ep[(pe * ibrc_state->ep_count)];
    }

    status = check_poll_avail(ep, WAIT_ALL /*1*/);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "check_poll failed \n");

#ifdef NVSHMEM_USE_GDRCOPY
    while (atomics_acked < atomics_issued) {
        nvshmem_transport_t t = (nvshmem_transport_t)ep->transport;
        status = progress_recv(tcurr, (transport_ibrc_state_t *)t->state);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "progress failed \n");
    }
#endif
out:
    return status;
}

int nvshmemt_ibrc_ep_create(struct ibrc_ep **ep, int devid, nvshmem_transport_t t,
                            transport_ibrc_state_t *ibrc_state) {
    int status = 0;

    status = ep_create(t, ep, devid, ibrc_state);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ep_create failed\n");

    // setup loopback connection on the first device used.
    if (!ibrc_cst_ep) {
        status = setup_cst_loopback(t, ibrc_state, devid);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cst setup failed \n");
    }

out:
    return status;
}

int nvshmemt_ibrc_ep_get_handle(struct ibrc_ep_handle *ep_handle_ptr, struct ibrc_ep *ep) {
    int status = 0;

    status = ep_get_handle(ep_handle_ptr, ep);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ep_get_handle failed \n");

out:
    return status;
}

int nvshmemt_ibrc_ep_destroy(struct ibrc_ep *ep) {
    int status = 0;

    status = check_poll_avail(ep, WAIT_ALL /*1*/);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "check_poll failed \n");

    // TODO: clean up qp, cq, etc.

out:
    return status;
}

int nvshmemt_ibrc_ep_connect(struct ibrc_ep *ep, struct ibrc_ep_handle *ep_handle) {
    int status = 0;

    status = ep_connect(ep, ep_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ep_connect failed \n");

out:
    return status;
}

int nvshmemt_ibrc_connect_endpoints(nvshmem_transport_t t, int *selected_dev_ids,
                                    int num_selected_devs) {
    /* transport side */
    struct ibrc_ep_handle *local_ep_handles = NULL, *ep_handles = NULL;
    transport_ibrc_state_t *ibrc_state = (transport_ibrc_state_t *)t->state;
    int status = 0;
    int n_pes = t->n_pes;
    int ep_count;

    if (ibrc_state->selected_dev_id >= 0) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out_already_connected,
                           "Device already selected. IBRC only supports"
                           " one NIC per PE.\n");
    }

    if (num_selected_devs > 1) {
        INFO(ibrc_state->log_level,
             "IBRC only supports One NIC / PE. All other NICs will be ignored.");
    }

    /* allocate all EPs for transport, plus 1 for the proxy thread. */
    ep_count = ibrc_state->ep_count = MAX_TRANSPORT_EP_COUNT + 1;
    ibrc_state->proxy_ep_idx = MAX_TRANSPORT_EP_COUNT;

    ibrc_state->selected_dev_id = selected_dev_ids[0];

    ibrc_state->ep = (struct ibrc_ep **)calloc(n_pes * ep_count, sizeof(struct ibrc_ep *));
    NVSHMEMI_NULL_ERROR_JMP(ibrc_state->ep, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "failed allocating space for endpoints \n");

    local_ep_handles =
        (struct ibrc_ep_handle *)calloc(n_pes * ep_count, sizeof(struct ibrc_ep_handle));
    NVSHMEMI_NULL_ERROR_JMP(local_ep_handles, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "failed allocating space for local ep handles \n");

    ep_handles = (struct ibrc_ep_handle *)calloc(n_pes * ep_count, sizeof(struct ibrc_ep_handle));
    NVSHMEMI_NULL_ERROR_JMP(ep_handles, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "failed allocating space for ep handles \n");

    for (int j = 0; j < n_pes; j++) {
        for (int k = 0; k < ep_count; k++) {
            nvshmemt_ibrc_ep_create(&ibrc_state->ep[j * ep_count + k], ibrc_state->selected_dev_id,
                                    t, ibrc_state);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "transport create ep failed \n");
            status = nvshmemt_ibrc_ep_get_handle(&local_ep_handles[j * ep_count + k],
                                                 ibrc_state->ep[j * ep_count + k]);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "transport get ep handle failed \n");
        }
    }

    status = t->boot_handle->alltoall((void *)local_ep_handles, (void *)ep_handles,
                                      sizeof(struct ibrc_ep_handle) * ep_count, t->boot_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "allgather of ep handles failed \n");

    for (int j = 0; j < n_pes; j++) {
        for (int k = 0; k < ep_count; k++) {
            status = nvshmemt_ibrc_ep_connect(ibrc_state->ep[j * ep_count + k],
                                              &ep_handles[j * ep_count + k]);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "transport create connect failed \n");
        }
    }
out:
    if (status) {
        ibrc_state->selected_dev_id = -1;
        if (ibrc_state->ep) free(ibrc_state->ep);
    }

out_already_connected:
    if (local_ep_handles) free(local_ep_handles);
    if (ep_handles) free(ep_handles);
    return status;
}

int nvshmemt_init(nvshmem_transport_t *t, struct nvshmemi_cuda_fn_table *table, int api_version) {
    int status = 0;
    struct nvshmem_transport *transport = NULL;
    transport_ibrc_state_t *ibrc_state = NULL;
    struct ibv_device **dev_list = NULL;
    int num_devices;
    struct ibrc_device *device;
    std::vector<std::string> nic_names_n_pes;
    std::vector<std::string> nic_names;
    int exclude_list = 0;
    struct nvshmemt_hca_info hca_list[MAX_NUM_HCAS];
    struct nvshmemt_hca_info pe_hca_mapping[MAX_NUM_PES_PER_NODE];
    int hca_list_count = 0, pe_hca_map_count = 0, user_selection = 0;
    int offset = 0;
    int flag;
    connected_qp_count = 0;
    CUdevice gpu_device_id;

    if (NVSHMEM_TRANSPORT_MAJOR_VERSION(api_version) != NVSHMEM_TRANSPORT_PLUGIN_MAJOR_VERSION) {
        NVSHMEMI_ERROR_PRINT(
            "NVSHMEM provided an incompatible version of the transport interface. "
            "This transport supports transport API major version %d. Host has %d",
            NVSHMEM_TRANSPORT_PLUGIN_MAJOR_VERSION, NVSHMEM_TRANSPORT_MAJOR_VERSION(api_version));
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    transport = (struct nvshmem_transport *)malloc(sizeof(struct nvshmem_transport));
    memset(transport, 0, sizeof(struct nvshmem_transport));
    transport->is_successfully_initialized =
        false; /* set it to true after everything has been successfully initialized */

    ibrc_state = (transport_ibrc_state_t *)calloc(1, sizeof(transport_ibrc_state_t));
    NVSHMEMI_NULL_ERROR_JMP(ibrc_state, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "p2p state allocation failed \n");

    /* set selected device ID to -1 to indicate none is selected. */
    ibrc_state->selected_dev_id = -1;
    transport->state = (void *)ibrc_state;

    ibrc_state->options = (struct nvshmemi_options_s *)calloc(1, sizeof(struct nvshmemi_options_s));
    NVSHMEMI_NULL_ERROR_JMP(ibrc_state->options, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "Unable to allocate options.");

    status = nvshmemi_env_options_init(ibrc_state->options);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Unable to initialize transport options.");

    ibrc_state->log_level = nvshmemt_common_get_log_level(ibrc_state->options);

    if (nvshmemt_ibv_ftable_init(&ibv_handle, &ftable, ibrc_state->log_level)) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                           "Unable to dlopen libibverbs. Skipping devx transport.");
    }

    ftable.fork_init();

    dev_list = ftable.get_device_list(&num_devices);
    NVSHMEMI_NULL_ERROR_JMP(dev_list, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "get_device_list failed \n");

    ibrc_state->devices = calloc(MAX_NUM_HCAS, sizeof(struct ibrc_device));
    NVSHMEMI_NULL_ERROR_JMP(ibrc_state->devices, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "get_device_list failed \n");

    ibrc_state->dev_ids = (int *)malloc(MAX_NUM_PES_PER_NODE * sizeof(int));
    NVSHMEMI_NULL_ERROR_JMP(ibrc_state->dev_ids, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "malloc failed \n");

    ibrc_state->port_ids = (int *)malloc(MAX_NUM_PES_PER_NODE * sizeof(int));
    NVSHMEMI_NULL_ERROR_JMP(ibrc_state->port_ids, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "malloc failed \n");

    status = pthread_mutex_init(&ibrc_mutex_send_progress, NULL);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "pthread_mutex_init failed \n");

    status = pthread_mutex_init(&ibrc_mutex_recv_progress, NULL);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "pthread_mutex_init failed \n");

#ifdef NVSHMEM_USE_GDRCOPY
    if (ibrc_state->options->DISABLE_GDRCOPY) {
        use_gdrcopy = false;
    } else {
        use_gdrcopy = nvshmemt_gdrcopy_ftable_init(&gdrcopy_ftable, &gdr_desc, &gdrcopy_handle,
                                                   ibrc_state->log_level);
    }
#endif

    ibrc_state->table = table;

    if (ibrc_state->options->DISABLE_IB_NATIVE_ATOMICS) {
        use_ib_native_atomics = 0;
    }
    ibrc_srq_depth = ibrc_state->options->SRQ_DEPTH;
    ibrc_qp_depth = ibrc_state->options->QP_DEPTH;

    if (ibrc_state->options->HCA_LIST_provided) {
        user_selection = 1;
        exclude_list = (ibrc_state->options->HCA_LIST[0] == '^');
        hca_list_count = nvshmemt_parse_hca_list(ibrc_state->options->HCA_LIST, hca_list,
                                                 MAX_NUM_HCAS, ibrc_state->log_level);
    }

    if (ibrc_state->options->HCA_PE_MAPPING_provided) {
        if (hca_list_count) {
            NVSHMEMI_WARN_PRINT(
                "Found conflicting parameters NVSHMEM_HCA_LIST and NVSHMEM_HCA_PE_MAPPING, "
                "ignoring "
                "NVSHMEM_HCA_PE_MAPPING \n");
        } else {
            user_selection = 1;
            pe_hca_map_count =
                nvshmemt_parse_hca_list(ibrc_state->options->HCA_PE_MAPPING, pe_hca_mapping,
                                        MAX_NUM_PES_PER_NODE, ibrc_state->log_level);
        }
    }

    INFO(ibrc_state->log_level,
         "Begin - Enumerating IB devices in the system ([<dev_id, device_name, num_ports>]) - ");
    for (int i = 0; i < num_devices; i++) {
        device = (struct ibrc_device *)ibrc_state->devices + i;
        device->dev = dev_list[i];

        device->context = ftable.open_device(device->dev);
        if (!device->context) {
            INFO(ibrc_state->log_level, "open_device failed for IB device at index %d", i);
            continue;
        }

        const char *name = ftable.get_device_name(device->dev);
        NVSHMEMI_NULL_ERROR_JMP(name, status, NVSHMEMX_ERROR_INTERNAL, out,
                                "ibv_get_device_name failed \n");

        status = ftable.query_device(device->context, &device->device_attr);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_query_device failed \n");

        NVSHMEMT_IBRC_MAX_RD_ATOMIC = (device->device_attr).max_qp_rd_atom;
        INFO(ibrc_state->log_level,
             "Enumerated IB devices in the system - device id=%d (of %d), name=%s, num_ports=%d", i,
             num_devices, name, device->device_attr.phys_port_cnt);
        int device_used = 0;
        for (int p = 1; p <= device->device_attr.phys_port_cnt; p++) {
            int allowed_device = 1;
            int replicate_count = 1;
            if (hca_list_count) {
                // filter out based on user hca list
                allowed_device = exclude_list;
                for (int j = 0; j < hca_list_count; j++) {
                    if (!strcmp(hca_list[j].name, name)) {
                        if (hca_list[j].port == -1 || hca_list[j].port == p) {
                            hca_list[j].found = 1;
                            allowed_device = !exclude_list;
                        }
                    }
                }
            } else if (pe_hca_map_count) {
                // filter devices based on user hca-pe mapping
                allowed_device = 0;
                for (int j = 0; j < pe_hca_map_count; j++) {
                    if (!strcmp(pe_hca_mapping[j].name, name)) {
                        if (pe_hca_mapping[j].port == -1 || pe_hca_mapping[j].port == p) {
                            allowed_device = 1;
                            pe_hca_mapping[j].found = 1;
                            replicate_count = pe_hca_mapping[j].count;
                        }
                    }
                }
            }

            if (!allowed_device) {
                continue;
            } else {
                status = ftable.query_port(device->context, p, &device->port_attr[p - 1]);
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                      "ibv_port_query failed \n");

                if ((device->port_attr[p - 1].state != IBV_PORT_ACTIVE) ||
                    (device->port_attr[p - 1].link_layer != IBV_LINK_LAYER_INFINIBAND &&
                     device->port_attr[p - 1].link_layer != IBV_LINK_LAYER_ETHERNET)) {
                    if (user_selection) {
                        NVSHMEMI_WARN_PRINT(
                            "found inactive port or port with non-IB link layer protocol, "
                            "skipping...\n");
                    }
                    continue;
                }

                ib_get_gid_index(&ftable, device->context, p, device->port_attr[p - 1].gid_tbl_len,
                                 &device->gid_info[p - 1].local_gid_index, ibrc_state->log_level,
                                 ibrc_state->options);
                status =
                    ftable.query_gid(device->context, p, device->gid_info[p - 1].local_gid_index,
                                     &device->gid_info[p - 1].local_gid);
                NVSHMEMI_NULL_ERROR_JMP(dev_list, status, NVSHMEMX_ERROR_INTERNAL, out,
                                        "query_gid failed \n");

                if (!device->pd) {
                    device->pd = ftable.alloc_pd(device->context);
                    NVSHMEMI_NULL_ERROR_JMP(device->pd, status, NVSHMEMX_ERROR_INTERNAL, out,
                                            "ibv_alloc_pd failed \n");
                }

                for (int k = 0; k < replicate_count; k++) {
                    ibrc_state->dev_ids[offset] = i;
                    ibrc_state->port_ids[offset] = p;
                    offset++;
                }

                device_used = 1;
            }
        }

        if (!device_used) {
            status = ftable.close_device(device->context);
            if (device->pd) {
                status = ftable.dealloc_pd(device->pd);
            }

            device->context = NULL;
            device->pd = NULL;
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "ibv_close_device or ibv_dealloc_pd failed \n");
        }
    }
    INFO(ibrc_state->log_level, "End - Enumerating IB devices in the system");

    ibrc_state->n_dev_ids = offset;
    INFO(ibrc_state->log_level,
         "Begin - Ordered list of devices for assignment (after processing user provdied env vars "
         "(if any))  - ");
    for (int i = 0; i < ibrc_state->n_dev_ids; i++) {
        INFO(ibrc_state->log_level,
             "Ordered list of devices for assignment - idx=%d (of %d), device id=%d, port_num=%d",
             i, ibrc_state->n_dev_ids, ibrc_state->dev_ids[i], ibrc_state->port_ids[i]);
    }
    INFO(ibrc_state->log_level,
         "End - Ordered list of devices for assignment (after processing user provdied env vars "
         "(if any))");

    if (!ibrc_state->n_dev_ids) {
        INFO(ibrc_state->log_level, "no active IB device found, exiting");
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }

    transport->n_devices = ibrc_state->n_dev_ids;
    transport->device_pci_paths = (char **)calloc(transport->n_devices, sizeof(char *));
    NVSHMEMI_NULL_ERROR_JMP(transport->device_pci_paths, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "Unable to allocate paths for IB transport.");
    for (int i = 0; i < transport->n_devices; i++) {
        status = get_pci_path(i, &transport->device_pci_paths[i], transport);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Failed to get paths for PCI devices.");
    }

    // print devices that were not found
    if (hca_list_count) {
        for (int j = 0; j < hca_list_count; j++) {
            if (hca_list[j].found != 1) {
                NVSHMEMI_WARN_PRINT(
                    "cound not find user specified HCA name: %s port: %d, skipping\n",
                    hca_list[j].name, hca_list[j].port);
            }
        }
    } else if (pe_hca_map_count) {
        // filter devices based on user hca-pe mapping
        for (int j = 0; j < pe_hca_map_count; j++) {
            if (pe_hca_mapping[j].found != 1) {
                NVSHMEMI_WARN_PRINT(
                    "cound not find user specified HCA name: %s port: %d, skipping\n",
                    pe_hca_mapping[j].name, pe_hca_mapping[j].port);
            }
        }
    }

    // allocate buffer pool
    bpool_size = ibrc_srq_depth;
    nvshmemi_ib_malloc((void **)&bpool, bpool_size * sizeof(ibrc_buf_t), ibrc_state->log_level);
    NVSHMEMI_NULL_ERROR_JMP(bpool, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "buf poll allocation failed \n");
    for (int i = 0; i < bpool_size; i++) {
        bpool_free.push_back((void *)(bpool + i));
    }

    transport->host_ops.can_reach_peer = nvshmemt_ibrc_can_reach_peer;
    transport->host_ops.connect_endpoints = nvshmemt_ibrc_connect_endpoints;
    transport->host_ops.get_mem_handle = nvshmemt_ibrc_get_mem_handle;
    transport->host_ops.release_mem_handle = nvshmemt_ibrc_release_mem_handle;
    transport->host_ops.rma = nvshmemt_ibrc_rma;
    transport->host_ops.amo = nvshmemt_ibrc_amo;
    transport->host_ops.fence = NULL;
    transport->host_ops.quiet = nvshmemt_ibrc_quiet;
    transport->host_ops.finalize = nvshmemt_ibrc_finalize;
    transport->host_ops.show_info = nvshmemt_ibrc_show_info;
    transport->host_ops.progress = nvshmemt_ibrc_progress;

    transport->host_ops.enforce_cst = nvshmemt_ibrc_enforce_cst_at_target;
#if !defined(NVSHMEM_PPC64LE) && !defined(NVSHMEM_AARCH64)
    if (!use_gdrcopy)
#endif
        transport->host_ops.enforce_cst_at_target = nvshmemt_ibrc_enforce_cst_at_target;

    transport->attr = NVSHMEM_TRANSPORT_ATTR_CONNECTED;
    transport->is_successfully_initialized = true;
    transport->max_op_len = 1ULL << 30;
    transport->api_version = api_version < NVSHMEM_TRANSPORT_INTERFACE_VERSION
                                 ? api_version
                                 : NVSHMEM_TRANSPORT_INTERFACE_VERSION;

    *t = transport;

    ibrc_state->dmabuf_support = false;

    if (ibrc_state->options->IB_DISABLE_DMABUF) {
        ibrc_state->dmabuf_support = false;
        goto check_nv_peer_mem;
    }

    status = CUPFN(table, cuCtxGetDevice(&gpu_device_id));
    if (status != CUDA_SUCCESS) {
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }
    status = CUPFN(table, cuDeviceGetAttribute(
                              &flag, (CUdevice_attribute)CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED,
                              gpu_device_id));
    if (status != CUDA_SUCCESS) {
        status = 0;
        cudaGetLastError();
    } else if (flag == 1) {
        ibrc_state->dmabuf_support = true;
    }
check_nv_peer_mem:

    if (ibrc_state->dmabuf_support == false) {
        if (nvshmemt_ib_common_nv_peer_mem_available() != NVSHMEMX_SUCCESS) {
            NVSHMEMI_ERROR_PRINT(
                "neither nv_peer_mem, or nvidia_peermem detected. Skipping transport.\n");
            status = NVSHMEMX_ERROR_INTERNAL;
            goto out;
        }
    }

out:

    if (status) {
        if (ibrc_state) {
            if (ibrc_state->devices) {
                free(ibrc_state->devices);
            }
            if (ibrc_state->dev_ids) {
                free(ibrc_state->dev_ids);
            }
            if (ibrc_state->port_ids) {
                free(ibrc_state->port_ids);
            }
            if (ibrc_state->options) {
                free(ibrc_state->options);
            }
            free(ibrc_state);
        }
        if (transport) {
            free(transport);
        }
    }
    return status;
}
