/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include <stdint.h>  // IWYU pragma: keep
#include <stddef.h>
#include <deque>
#include <vector>
#include <mutex>
// IWYU pragma: no_include <bits/stdint-uintn.h>

#include "non_abi/nvshmem_build_options.h"
#include "device_host_transport/nvshmem_common_transport.h"

#ifdef NVSHMEM_USE_GDRCOPY
#include "gdrapi.h"
#endif

#define NVSHMEMT_LIBFABRIC_MAJ_VER 1
#define NVSHMEMT_LIBFABRIC_MIN_VER 5

#define NVSHMEMT_LIBFABRIC_DOMAIN_LEN 32
#define NVSHMEMT_LIBFABRIC_PROVIDER_LEN 32
#define NVSHMEMT_LIBFABRIC_EP_LEN 128

/* one EP for all proxy ops, one for host ops */
#define NVSHMEMT_LIBFABRIC_DEFAULT_NUM_EPS 2
#define NVSHMEMT_LIBFABRIC_PROXY_EP_IDX 1
#define NVSHMEMT_LIBFABRIC_HOST_EP_IDX 0

#define NVSHMEMT_LIBFABRIC_QUIET_TIMEOUT_MS 20

/* Maximum size of inject data. Currently
 * the max size we will use is one element
 * of a given type. Making it 16 bytes in the
 * case of complex number support. */
#ifdef NVSHMEM_COMPLEX_SUPPORT
#define NVSHMEMT_LIBFABRIC_INJECT_BYTES 16
#else
#define NVSHMEMT_LIBFABRIC_INJECT_BYTES 8
#endif

#define NVSHMEMT_LIBFABRIC_MAX_RETRIES (1ULL << 20)

typedef struct {
    char name[NVSHMEMT_LIBFABRIC_DOMAIN_LEN];
} nvshmemt_libfabric_domain_name_t;

typedef struct {
    char name[NVSHMEMT_LIBFABRIC_EP_LEN];
} nvshmemt_libfabric_ep_name_t;

typedef struct {
    struct fid_ep *endpoint;
    struct fid_cq *cq;
    struct fid_cntr *counter;
    uint64_t submitted_ops;
} nvshmemt_libfabric_endpoint_t;

typedef enum {
    NVSHMEMT_LIBFABRIC_PROVIDER_VERBS = 0,
    NVSHMEMT_LIBFABRIC_PROVIDER_SLINGSHOT,
    NVSHMEMT_LIBFABRIC_PROVIDER_EFA
} nvshmemt_libfabric_provider;

typedef enum {
    NVSHMEMT_LIBFABRIC_CONTEXT_P_OP = 0,
    NVSHMEMT_LIBFABRIC_CONTEXT_SEND_AMO,
    NVSHMEMT_LIBFABRIC_CONTEXT_RECV_AMO
} nvshemmt_libfabric_context_t;

class threadSafeOpQueue {
   private:
    std::mutex send_mutex;
    std::mutex recv_mutex;
    std::vector<void *> send;
    std::deque<void *> recv;

   public:
    void *getNextSend() {
        void *elem;
        send_mutex.lock();
        if (send.empty()) {
            send_mutex.unlock();
            return NULL;
        }
        elem = send.back();
        send.pop_back();
        send_mutex.unlock();
        return elem;
    }

    void putToSend(void *elem) {
        send_mutex.lock();
        send.push_back(elem);
        send_mutex.unlock();
        return;
    }

    void putToSendBulk(char *elem, size_t elem_size, size_t num_elems) {
        send_mutex.lock();
        for (size_t i = 0; i < num_elems; i++) {
            send.push_back(elem);
            elem = elem + elem_size;
        }
        send_mutex.unlock();
        return;
    }

    void *getNextRecv() {
        recv_mutex.lock();
        void *elem;
        if (recv.empty()) {
            recv_mutex.unlock();
            return NULL;
        }
        elem = recv.front();
        recv.pop_front();
        recv_mutex.unlock();
        return elem;
    }

    void putToRecv(void *elem) {
        recv_mutex.lock();
        recv.push_back(elem);
        recv_mutex.unlock();
    }

    void putToRecvBulk(char *elem, size_t elem_size, size_t num_elems) {
        recv_mutex.lock();
        for (size_t i = 0; i < num_elems; i++) {
            recv.push_back(elem);
            elem = elem + elem_size;
        }
        recv_mutex.unlock();
        return;
    }
};

typedef struct {
    struct fi_info *prov_info;
    struct fi_info *all_prov_info;
    struct fid_fabric *fabric;
    struct fid_domain *domain;
    struct fid_av *addresses[NVSHMEMT_LIBFABRIC_DEFAULT_NUM_EPS];
    nvshmemt_libfabric_endpoint_t *eps;
    /* local_mr is used only for consistency ops. */
    struct fid_mr *local_mr[2];
    uint64_t local_mr_key[2];
    void *local_mr_desc[2];
    void *local_mem_ptr;
    nvshmemt_libfabric_domain_name_t *domain_names;
    int num_domains;
    nvshmemt_libfabric_provider provider;
    int log_level;
    struct nvshmemi_cuda_fn_table *table;
    size_t num_sends;
    void *send_buf;
    size_t num_recvs;
    void *recv_buf;
    struct transport_mem_handle_info_cache *cache;
} nvshmemt_libfabric_state_t;

typedef enum {
    NVSHMEMT_LIBFABRIC_SEND,
    NVSHMEMT_LIBFABRIC_ACK,
} nvshmemt_libfabric_recv_t;

typedef struct {
    struct fid_mr *mr;
    uint64_t key;
    void *local_desc;
} nvshmemt_libfabric_mem_handle_ep_t;

typedef struct {
    size_t gdr_mapping_size;
    void *ptr;
    void *cpu_ptr;
#ifdef NVSHMEM_USE_GDRCOPY
    gdr_mh_t mh;
    void *cpu_ptr_base;
#endif
} nvshmemt_libfabric_memhandle_info_t;

typedef struct {
    void *buf;
    nvshmemt_libfabric_mem_handle_ep_t hdls[2];
} nvshmemt_libfabric_mem_handle_t;

typedef struct nvshmemt_libfabric_gdr_send_p_op {
    uint64_t value;
} nvshmemt_libfabric_gdr_send_p_op_t;

typedef struct nvshmemt_libfabric_gdr_send_amo_op {
    nvshmemi_amo_t op;
    void *target_addr;
    void *ret_addr;
    uint64_t retflag;
    uint64_t swap_add;
    uint64_t comp;
    uint32_t size;
    int ret_ep;
} nvshmemt_libfabric_gdr_send_amo_op_t;

typedef struct nvshmemt_libfabric_gdr_ret_amo_op {
    void *ret_addr;
    g_elem_t elem;
} nvshmemt_libfabric_gdr_ret_amo_op_t;

typedef struct nvshmemt_libfabric_gdr_op_ctx {
    nvshmemt_libfabric_recv_t type;
    nvshmemt_libfabric_endpoint_t *ep;
    union {
        nvshmemt_libfabric_gdr_send_p_op_t p_op;
        nvshmemt_libfabric_gdr_send_amo_op_t send_amo;
        nvshmemt_libfabric_gdr_ret_amo_op_t ret_amo;
    };
} nvshmemt_libfabric_gdr_op_ctx_t;
