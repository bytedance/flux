/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "ibdevx.h"
#include <assert.h>                                          // for assert
#include <cuda.h>                                            // for CUDA_SUCCESS, CU_DEVICE...
#include <cuda_runtime.h>                                    // for cudaGetLastError
#include <endian.h>                                          // for htobe32, htobe64, htobe16
#include <errno.h>                                           // for ENOMEM
#include <netinet/in.h>                                      // for ntohl
#include <pthread.h>                                         // for pthread_mutex_destroy
#include <stdint.h>                                          // for uint32_t, uint64_t, uin...
#include <stdio.h>                                           // for NULL, printf, size_t
#include <stdlib.h>                                          // for free, calloc, malloc
#include <string.h>                                          // for memset, memcpy, strcmp
#include <unistd.h>                                          // for sysconf, _SC_PAGESIZE
#include <cmath>                                             // for log2
#include <map>                                               // for map, _Rb_tree_iterator
#include <string>                                            // for string
#include <utility>                                           // for pair, make_pair
#include <vector>                                            // for vector
#include "device_host_transport/nvshmem_common_transport.h"  // for NVSHMEMI_OP_P, NVSHMEMI...
#include "internal/host_transport/cudawrap.h"                // for CUPFN, nvshmemi_cuda_fn...
#include "bootstrap_host_transport/env_defs_internal.h"      // for nvshmemi_options_s, nvs...
#include "non_abi/nvshmemx_error.h"                          // for NVSHMEMX_ERROR_INTERNAL
#include "infiniband/mlx5dv.h"                               // for DEVX_SET, mlx5_wqe_ctrl...
#include "infiniband/verbs.h"                                // for ibv_port_attr, ibv_ah_attr
#include "mlx5_ifc.h"                                        // for mlx5_ifc_qpc_bits, mlx5...
#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"  // for bootstrap_handle_t
#include "internal/host_transport/nvshmemi_transport_defines.h"  // for nvshmem_mem_handle_t
#include "internal/host_transport/transport.h"                   // for nvshmem_transport, amo_...
#include "transport_common.h"                                    // for nvshmemt_hca_info, nvsh...
#include "transport_ib_common.h"                                 // for nvshmemt_ib_common_mem_...
#include "transport_mlx5_common.h"                               // for nvshmemt_ib_common_chec...

#define ibdevx_MAX_INLINE_SIZE 128

int ibdevx_srq_depth;
#define ibdevx_SRQ_MASK (ibdevx_srq_depth - 1)

int ibdevx_qp_depth;
#define ibdevx_REQUEST_QUEUE_MASK (ibdevx_qp_depth - 1)
#define ibdevx_BUF_SIZE 64

#if defined(NVSHMEM_X86_64)
#define ibdevx_CACHELINE 64
#elif defined(NVSHMEM_PPC64LE)
#define ibdevx_CACHELINE 128
#elif defined(NVSHMEM_AARCH64)
#define ibdevx_CACHELINE 64
#else
#error Unknown cache line size
#endif

#define MAX_NUM_HCAS 16
#define MAX_NUM_PORTS 4
#define MAX_NUM_PES_PER_NODE 32
#define BAR_READ_BUFSIZE (sizeof(uint64_t))

enum { WAIT_ANY = 0, WAIT_TWO = 1, WAIT_ALL = 2 };

int NVSHMEMT_IBDEVX_MAX_RD_ATOMIC; /* Maximum number of RDMA Read & Atomic operations that can be
                                    * outstanding per QP
                                    */

struct ibdevx_cq {
    void *buf;
    uint32_t num_cqe;
    uint32_t cur_idx;
    uint32_t cqn;
    volatile uint32_t *dbrec;
};

struct ibdevx_device {
    struct ibdevx_cq rcq;
    struct ibdevx_cq scq;
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
    int srqn;
    int pdn;
};

struct __attribute__((__packed__)) ibdevx_rw_inline_data_seg {
    uint32_t byte_count;
    union {
        uint64_t data_64;
        uint32_t data_32;
        uint16_t data_16;
        uint8_t data_8;
    } data;
    uint32_t reserved;
};
#if __cplusplus >= 201103L
static_assert(sizeof(struct ibdevx_rw_inline_data_seg) == 16,
              "static_assert(sizeof(T) == 16) failed");
#endif

struct __attribute__((__packed__, aligned(4))) ibdevx_rw_wqe {
    struct mlx5_wqe_ctrl_seg ctrl;
    struct mlx5_wqe_raddr_seg raddr;
    union {
        struct mlx5_wqe_data_seg data_seg;
        struct ibdevx_rw_inline_data_seg data_inl;
    } data;
};
#if __cplusplus >= 201103L
static_assert(sizeof(struct ibdevx_rw_wqe) == 48, "static_assert(sizeof(T) >= 8) failed");
#endif

struct __attribute__((__packed__)) ibdevx_atomic_32_masked_fetch_add_seg {
    uint32_t add_data;
    uint32_t field_boundary;
    uint64_t reserved;
};
#if __cplusplus >= 201103L
static_assert(sizeof(struct ibdevx_atomic_32_masked_fetch_add_seg) == 16,
              "static_assert(sizeof(T) >= 8) failed");
#endif

struct __attribute__((__packed__)) ibdevx_atomic_64_masked_fetch_add_seg {
    uint64_t add_data;
    uint64_t field_boundary;
};
#if __cplusplus >= 201103L
static_assert(sizeof(struct ibdevx_atomic_64_masked_fetch_add_seg) == 16,
              "static_assert(sizeof(T) >= 8) failed");
#endif

struct __attribute__((__packed__)) ibdevx_atomic_32_masked_compare_swap_seg {
    uint32_t swap_data;
    uint32_t compare_data;
    uint32_t swap_mask;
    uint32_t compare_mask;
};
#if __cplusplus >= 201103L
static_assert(sizeof(struct ibdevx_atomic_32_masked_compare_swap_seg) == 16,
              "static_assert(sizeof(T) >= 8) failed");
#endif

struct __attribute__((__packed__)) ibdevx_atomic_64_masked_compare_swap_seg {
    uint64_t swap;
    uint64_t compare;
};
#if __cplusplus >= 201103L
static_assert(sizeof(struct ibdevx_atomic_64_masked_compare_swap_seg) == 16,
              "static_assert(sizeof(T) >= 8) failed");
#endif

struct __attribute__((__packed__, aligned(4))) ibdevx_atomic_32_wqe {
    struct mlx5_wqe_ctrl_seg ctrl;
    struct mlx5_wqe_raddr_seg raddr;
    union {
        struct ibdevx_atomic_32_masked_fetch_add_seg fa_seg;
        struct ibdevx_atomic_32_masked_compare_swap_seg cs_seg;
        struct mlx5_wqe_atomic_seg no_mask_seg;
    };
    struct mlx5_wqe_data_seg data;
};
#if __cplusplus >= 201103L
static_assert(sizeof(struct ibdevx_atomic_32_wqe) == 64, "static_assert(sizeof(T) >= 8) failed");
#endif

struct __attribute__((__packed__, aligned(4))) ibdevx_atomic_64_amo_wqe {
    struct mlx5_wqe_ctrl_seg ctrl;
    struct mlx5_wqe_raddr_seg raddr;
    union {
        struct ibdevx_atomic_64_masked_fetch_add_seg fa_seg;
        struct mlx5_wqe_atomic_seg no_mask_seg;
    };
    struct mlx5_wqe_data_seg data;
};
#if __cplusplus >= 201103L
static_assert(sizeof(struct ibdevx_atomic_64_amo_wqe) == 64,
              "static_assert(sizeof(T) >= 8) failed");
#endif

struct __attribute__((__packed__)) ibdevx_dbr_buf {
    volatile uint16_t rsvd_1;
    volatile uint16_t rcv_counter;
    volatile uint16_t rsvd_2;
    volatile uint16_t send_counter;
};

struct ibdevx_ep {
    int devid;
    int portid;
    int qpid;
    struct mlx5dv_devx_obj *devx_qp;
    struct mlx5dv_devx_umem *wq_umem;
    struct mlx5dv_devx_umem *db_umem;
    struct mlx5dv_devx_uar *uar;
    void *bf_reg;
    uint32_t bf_reg_size;
    uint32_t bf_reg_offset;
    struct ibv_cq *send_cq;
    struct ibv_cq *recv_cq;
    struct ibv_ah *ah;
    struct ibdevx_dbr_buf *dbr_buf;
    void *wq_buf;
    /* the WQE basic bock index is what gets written to the doorbell record after each op is
     * prepared. */
    uint16_t wqe_bb_idx;
    volatile uint64_t head_op_id;
    volatile uint64_t tail_op_id;
    void *ibdevx_state;
};

struct ibdevx_ep_handle {
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
    struct ibdevx_ep **ep;
    struct nvshmemi_options_s *options;
    struct nvshmemi_cuda_fn_table *table;
} transport_ibdevx_state_t;

struct nvshmemt_ib_common_mem_handle local_dummy_mr;

pthread_mutex_t ibdevx_mutex_send_progress;

static std::map<unsigned int, long unsigned int> qp_map;
static uint64_t connected_qp_count;

struct ibdevx_ep *ibdevx_cst_ep;
static int use_ib_native_atomics = 1;

static struct nvshmemt_ibv_function_table ftable;
static void *ibv_handle;

int check_poll_avail(struct ibdevx_ep *ep, int wait_predicate);
int progress_send(transport_ibdevx_state_t *ibdevx_state);

int nvshmemt_ibdevx_show_info(struct nvshmem_transport *transport, int style) {
    NVSHMEMI_ERROR_PRINT("ibdevx show info not implemented");
    return 0;
}

static int get_pci_path(int dev, char **pci_path, nvshmem_transport_t t) {
    int status = NVSHMEMX_SUCCESS;

    struct nvshmem_transport *transport = (struct nvshmem_transport *)t;
    transport_ibdevx_state_t *ibdevx_state = (transport_ibdevx_state_t *)transport->state;
    int dev_id = ibdevx_state->dev_ids[dev];
    const char *ib_name =
        (const char *)((struct ibdevx_device *)ibdevx_state->devices)[dev_id].dev->name;

    status = nvshmemt_ib_iface_get_mlx_path(ib_name, pci_path);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "nvshmemt_ib_iface_get_mlx_path failed \n");

out:
    return status;
}

int nvshmemt_ibdevx_can_reach_peer(int *access, struct nvshmem_transport_pe_info *peer_info,
                                   nvshmem_transport_t t) {
    int status = 0;

    *access = NVSHMEM_TRANSPORT_CAP_CPU_WRITE | NVSHMEM_TRANSPORT_CAP_CPU_READ |
              NVSHMEM_TRANSPORT_CAP_CPU_ATOMICS;

    return status;
}

static int nvshmemt_ibdevx_mlx5_qp_destroy(struct ibdevx_ep *ep, struct ibdevx_device *device) {
    int status;

    status = mlx5dv_devx_obj_destroy(ep->devx_qp);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "mlx5dv_devx_obj_destroy failed.\n");

    status = mlx5dv_devx_umem_dereg(ep->db_umem);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "mlx5dv_devx_umem_dereg failed.\n");

    status = mlx5dv_devx_umem_dereg(ep->wq_umem);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "mlx5dv_devx_umem_dereg failed.\n");

    mlx5dv_devx_free_uar(ep->uar);

    free(ep->wq_buf);
    free(ep->dbr_buf);

out:
    return status;
}

static int nvshmemt_ibdevx_mlx5_qp_create(struct ibdevx_ep *ep, struct ibdevx_device *device) {
    mlx5dv_obj dv_obj;
    struct mlx5dv_pd dvpd;
    struct mlx5dv_cq dvscq;
    struct mlx5dv_cq dvrcq;
    struct mlx5dv_srq dvsrq;

    struct ibv_pd *pd = device->pd;
    struct ibv_context *context = pd->context;
    struct mlx5dv_devx_uar *uar = NULL;
    struct mlx5_cqe64 *cqe;

    void *qp_context;

    struct mlx5dv_devx_umem *wq_umem = NULL;
    void *wq_buf = NULL;

    struct mlx5dv_devx_umem *db_umem = NULL;
    void *dbr_buf = NULL;

    uint8_t cmd_in[DEVX_ST_SZ_BYTES(create_qp_in)] = {
        0,
    };
    uint8_t cmd_out[DEVX_ST_SZ_BYTES(create_qp_out)] = {
        0,
    };

    uint8_t cmd_cap_in[DEVX_ST_SZ_BYTES(query_hca_cap_in)] = {
        0,
    };
    uint8_t cmd_cap_out[DEVX_ST_SZ_BYTES(query_hca_cap_out)] = {
        0,
    };

    void *cap;
    uint32_t bf_reg_size, log_bf_reg_size;
    int wq_buf_size;
    int status;

    cap = DEVX_ADDR_OF(query_hca_cap_out, cmd_cap_out, capability);
    DEVX_SET(query_hca_cap_in, cmd_cap_in, opcode, MLX5_CMD_OP_QUERY_HCA_CAP);
    DEVX_SET(query_hca_cap_in, cmd_cap_in, op_mod,
             MLX5_SET_HCA_CAP_OP_MOD_GENERAL_DEVICE | HCA_CAP_OPMOD_GET_CUR);

    status = mlx5dv_devx_general_cmd(context, cmd_cap_in, sizeof(cmd_cap_in), cmd_cap_out,
                                     sizeof(cmd_cap_out));
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "mlx5dv_devx_general_cmd failed.\n");

    log_bf_reg_size = DEVX_GET(cmd_hca_cap, cap, log_bf_reg_size);

    // The size of 1st + 2nd half (as when we use alternating DB)
    bf_reg_size = 1LLU << log_bf_reg_size;

    // Allocate UAR. This will be used as a DB/BF register).
    uar = mlx5dv_devx_alloc_uar(context, MLX5DV_UAR_ALLOC_TYPE_BF);
    NVSHMEMI_NULL_ERROR_JMP(uar, status, ENOMEM, out, "cannot allocate mlx5dv_devx_uar\n");

    // Allocate WQ buffer.
    wq_buf_size = ibdevx_qp_depth * MLX5_SEND_WQE_BB;
    status = posix_memalign(&wq_buf, sysconf(_SC_PAGESIZE), wq_buf_size);
    NVSHMEMI_NULL_ERROR_JMP(wq_buf, status, ENOMEM, out, "cannot allocate wq buf for qpair.\n");

    wq_umem = mlx5dv_devx_umem_reg(context, wq_buf, wq_buf_size, 0);
    NVSHMEMI_NULL_ERROR_JMP(wq_umem, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "cannot register wq buf for qpair.\n");

    // Allocate Doorbell Register buffer.
    status = posix_memalign(&dbr_buf, sysconf(_SC_PAGESIZE), NVSHMEMT_IBDEVX_DBSIZE);
    NVSHMEMI_NULL_ERROR_JMP(dbr_buf, status, ENOMEM, out, "cannot allocate dbr buf for qpair.\n");

    db_umem = mlx5dv_devx_umem_reg(context, dbr_buf, NVSHMEMT_IBDEVX_DBSIZE, 0);
    NVSHMEMI_NULL_ERROR_JMP(wq_umem, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "cannot register dbr buf for qpair.\n");

    if (device->pdn == 0) {
        dv_obj.pd.in = pd;
        dv_obj.pd.out = &dvpd;
        status = mlx5dv_init_obj(&dv_obj, MLX5DV_OBJ_PD);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "mlx5dv PD initialization failed.\n");
        device->pdn = dvpd.pdn;
    }

    if (device->scq.cqn == 0) {
        dv_obj.cq.in = ep->send_cq;
        dv_obj.cq.out = &dvscq;
        status = mlx5dv_init_obj(&dv_obj, MLX5DV_OBJ_CQ);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "mlx5dv SCQ initialization failed.\n");
        device->scq.buf = dvscq.buf;
        device->scq.cqn = dvscq.cqn;
        device->scq.cur_idx = 0;
        device->scq.num_cqe = dvscq.cqe_cnt;
        device->scq.dbrec = dvscq.dbrec;

        for (uint32_t i = 0; i < device->scq.num_cqe; ++i) {
            cqe = ((struct mlx5_cqe64 *)device->scq.buf + i);
            cqe->op_own |= MLX5_CQE_OWNER_MASK;
        }
    }
    if (device->srqn == 0) {
        dvsrq.comp_mask = MLX5DV_SRQ_MASK_SRQN;
        dv_obj.srq.in = device->srq;
        dv_obj.srq.out = &dvsrq;
        status = mlx5dv_init_obj(&dv_obj, MLX5DV_OBJ_SRQ);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "mlx5dv SRQ initialization failed.\n");
        device->srqn = dvsrq.srqn;
    }

    if (device->srqn == 0) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                           "Unable to allocate SRQ for your device. "
                           "This may occur if your ofed is older than version 5.0.\n");
    }

    if (device->rcq.cqn == 0) {
        dv_obj.cq.in = ep->recv_cq;
        dv_obj.cq.out = &dvrcq;
        status = mlx5dv_init_obj(&dv_obj, MLX5DV_OBJ_CQ);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "mlx5dv RCQ initialization failed.\n");
        device->rcq.buf = dvrcq.buf;
        device->rcq.cqn = dvrcq.cqn;
        device->rcq.cur_idx = 0;
        device->rcq.num_cqe = dvrcq.cqe_cnt;
        device->rcq.dbrec = dvrcq.dbrec;
    }

    DEVX_SET(create_qp_in, cmd_in, opcode, MLX5_CMD_OP_CREATE_QP);
    DEVX_SET(create_qp_in, cmd_in, wq_umem_id, wq_umem->umem_id);  // WQ buffer

    qp_context = DEVX_ADDR_OF(create_qp_in, cmd_in, qpc);
    DEVX_SET(qpc, qp_context, st, MLX5_QPC_ST_RC);
    DEVX_SET(qpc, qp_context, pm_state, MLX5_QPC_PM_STATE_MIGRATED);
    DEVX_SET(qpc, qp_context, pd, device->pdn);
    DEVX_SET(qpc, qp_context, uar_page, uar->page_id);                   // BF register
    DEVX_SET(qpc, qp_context, rq_type, NVSHMEMT_IBDEVX_SRQ_TYPE_VALUE);  // Shared Receive Queue
    DEVX_SET(qpc, qp_context, srqn_rmpn_xrqn, device->srqn);
    DEVX_SET(qpc, qp_context, cqn_snd, device->scq.cqn);
    DEVX_SET(qpc, qp_context, cqn_rcv, device->rcq.cqn);
    assert(ibdevx_qp_depth != 0);
    DEVX_SET(qpc, qp_context, log_sq_size, (int)(log2(ibdevx_qp_depth)));
    DEVX_SET(qpc, qp_context, log_rq_size, 0);
    DEVX_SET(qpc, qp_context, cs_req, 0);            // Disable CS Request
    DEVX_SET(qpc, qp_context, cs_res, 0);            // Disable CS Response
    DEVX_SET(qpc, qp_context, dbr_umem_valid, 0x1);  // Enable dbr_umem_id
    DEVX_SET64(qpc, qp_context, dbr_addr,
               0);  // Offset 0 of dbr_umem_id (behavior changed because of dbr_umem_valid)
    DEVX_SET(qpc, qp_context, dbr_umem_id, db_umem->umem_id);  // DBR buffer
    DEVX_SET(qpc, qp_context, user_index, 0);
    DEVX_SET(qpc, qp_context, page_offset, 0);

    ep->devx_qp = mlx5dv_devx_obj_create(context, cmd_in, sizeof(cmd_in), cmd_out, sizeof(cmd_out));
    NVSHMEMI_NULL_ERROR_JMP(ep->devx_qp, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "cannot create devx object for qpair.\n");

    ep->qpid = DEVX_GET(create_qp_out, cmd_out, qpn);
    ep->uar = uar;
    ep->bf_reg = ep->uar->reg_addr;
    ep->bf_reg_size = bf_reg_size;
    ep->bf_reg_offset = 0;
    ep->wq_buf = wq_buf;
    ep->wq_umem = wq_umem;
    ep->dbr_buf = (struct ibdevx_dbr_buf *)dbr_buf;
    ep->db_umem = db_umem;

out:
    if (status != 0) {
        if (db_umem) {
            mlx5dv_devx_umem_dereg(db_umem);
        }

        if (dbr_buf) {
            free(dbr_buf);
        }

        if (wq_umem) {
            mlx5dv_devx_umem_dereg(wq_umem);
        }

        if (wq_buf) {
            free(wq_buf);
        }

        if (uar) {
            mlx5dv_devx_free_uar(uar);
        }
    }

    return status;
}

static int device_destroy_shared_ep_resources(struct ibdevx_device *device) {
    int status = 0;

    if (device->recv_cq) {
        status = ftable.destroy_cq(device->recv_cq);
        NVSHMEMI_NZ_ERROR_JMP(status, status, out, "Unable to destroy recv_cq.\n");
    }

    if (device->send_cq) {
        status = ftable.destroy_cq(device->send_cq);
        NVSHMEMI_NZ_ERROR_JMP(status, status, out, "Unable to destroy send_cq.\n");
    }

    if (device->srq) {
        status = ftable.destroy_srq(device->srq);
        NVSHMEMI_NZ_ERROR_JMP(status, status, out, "Unable to destroy srq.\n");
    }

    if (device->pd) {
        status = ftable.dealloc_pd(device->pd);
        NVSHMEMI_NZ_ERROR_JMP(status, status, out, "Unable to deallocate pd.\n");
    }

out:
    return status;
}

static int device_create_shared_ep_resources(struct ibdevx_device *device) {
    int status = 0;

    struct ibv_context *context = device->context;
    struct ibv_pd *pd = device->pd;

    struct ibv_srq_init_attr srq_init_attr;
    memset(&srq_init_attr, 0, sizeof(srq_init_attr));

    srq_init_attr.attr.max_wr = ibdevx_srq_depth;
    srq_init_attr.attr.max_sge = 1;

    device->srq = ftable.create_srq(pd, &srq_init_attr);
    NVSHMEMI_NULL_ERROR_JMP(device->srq, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "srq creation failed \n");

    device->recv_cq = ftable.create_cq(context, ibdevx_srq_depth, NULL, NULL, 0);
    NVSHMEMI_NULL_ERROR_JMP(device->recv_cq, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "cq creation failed \n");

    device->send_cq = ftable.create_cq(context, device->device_attr.max_cqe, NULL, NULL, 0);
    NVSHMEMI_NULL_ERROR_JMP(device->send_cq, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "cq creation failed \n");

out:
    return status;
}

static int ep_destroy(struct ibdevx_ep *ep, int devid, transport_ibdevx_state_t *ibdevx_state) {
    int status = 0;
    struct ibdevx_device *device =
        ((struct ibdevx_device *)ibdevx_state->devices + ibdevx_state->dev_ids[devid]);

    if (ep->devx_qp) {
        status = nvshmemt_ibdevx_mlx5_qp_destroy(ep, device);
        NVSHMEMI_NZ_ERROR_JMP(status, status, out,
                              "Unable to destroy qpair for ep in ibdevx transport.\n");
    }

    if (ep->ah) {
        status = ftable.destroy_ah(ep->ah);
        NVSHMEMI_NZ_ERROR_JMP(status, status, out, "Unable to destroy ah.\n");
    }

    qp_map.erase(ep->qpid);

out:
    free(ep);
    return status;
}

static int ep_create(struct ibdevx_ep **ep_ptr, int devid, transport_ibdevx_state_t *ibdevx_state) {
    int status = 0;
    struct ibdevx_ep *ep = NULL;
    struct ibdevx_device *device =
        ((struct ibdevx_device *)ibdevx_state->devices + ibdevx_state->dev_ids[devid]);
    int portid = ibdevx_state->port_ids[devid];

    // algining ep structure to prevent split tranactions when accessing head_op_id and
    // tail_op_id which can be used in inter-thread synchronization
    // TODO: use atomic variables instead to rely on language memory model guarantees
    status = posix_memalign((void **)&ep, ibdevx_CACHELINE, sizeof(struct ibdevx_ep));
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "ep allocation failed \n");
    memset((void *)ep, 0, sizeof(struct ibdevx_ep));

    if (!device->srq) {
        status = device_create_shared_ep_resources(device);
        NVSHMEMI_NZ_ERROR_JMP(status, status, out,
                              "Unable to create shared qp resources for device.\n");
    }
    ep->send_cq = device->send_cq;
    ep->recv_cq = device->recv_cq;

    status = nvshmemt_ibdevx_mlx5_qp_create(ep, device);
    NVSHMEMI_NZ_ERROR_JMP(status, status, out,
                          "Unable to create qpair for ep in ibdevx transport.\n");

    qp_map.insert(std::make_pair((unsigned int)ep->qpid, (long unsigned int)ep));

    ep->devid = ibdevx_state->dev_ids[devid];
    ep->portid = portid;
    ep->ibdevx_state = ibdevx_state;
    *ep_ptr = ep;

out:
    if (status) {
        if (ep) {
            free(ep);
        }
    }
    return status;
}

static int ep_connect(struct ibdevx_ep *ep, struct ibdevx_ep_handle *ep_handle) {
    int status = 0;
    int devid = ep->devid;
    int portid = ep->portid;
    transport_ibdevx_state_t *ibdevx_state = (transport_ibdevx_state_t *)ep->ibdevx_state;
    struct ibdevx_device *device = ((struct ibdevx_device *)ibdevx_state->devices + devid);
    struct ibv_port_attr *port_attr = device->port_attr + (portid - 1);

    uint8_t cmd_in1[DEVX_ST_SZ_BYTES(rst2init_qp_in)] = {
        0,
    };
    uint8_t cmd_out1[DEVX_ST_SZ_BYTES(rst2init_qp_out)] = {
        0,
    };
    uint8_t cmd_in2[DEVX_ST_SZ_BYTES(init2rtr_qp_in)] = {
        0,
    };
    uint8_t cmd_out2[DEVX_ST_SZ_BYTES(init2rtr_qp_out)] = {
        0,
    };
    uint8_t cmd_in3[DEVX_ST_SZ_BYTES(rtr2rts_qp_in)] = {
        0,
    };
    uint8_t cmd_out3[DEVX_ST_SZ_BYTES(rtr2rts_qp_out)] = {
        0,
    };

    void *qp_context;

    DEVX_SET(rst2init_qp_in, cmd_in1, opcode, MLX5_CMD_OP_RST2INIT_QP);
    DEVX_SET(rst2init_qp_in, cmd_in1, qpn, ep->qpid);

    qp_context = DEVX_ADDR_OF(rst2init_qp_in, cmd_in1, qpc);
    DEVX_SET(qpc, qp_context, pm_state, MLX5_QPC_PM_STATE_MIGRATED);
    DEVX_SET(qpc, qp_context, primary_address_path.vhca_port_num, ep->portid);
    DEVX_SET(qpc, qp_context, primary_address_path.pkey_index, 0);

    DEVX_SET(qpc, qp_context, wq_signature, 0x0);
    DEVX_SET(qpc, qp_context, counter_set_id, 0x0);
    DEVX_SET(qpc, qp_context, lag_tx_port_affinity, 0x0);

    status =
        mlx5dv_devx_obj_modify(ep->devx_qp, cmd_in1, sizeof(cmd_in1), cmd_out1, sizeof(cmd_out1));
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Unable to set qp to state INIT with errno %d.\n", status);

    DEVX_SET(init2rtr_qp_in, cmd_in2, opcode, MLX5_CMD_OP_INIT2RTR_QP);
    DEVX_SET(init2rtr_qp_in, cmd_in2, op_mod, 0x0);
    DEVX_SET(init2rtr_qp_in, cmd_in2, opt_param_mask, NVSHMEMT_IBDEVX_INIT2R2R_PARAM_MASK);
    DEVX_SET(init2rtr_qp_in, cmd_in2, qpn, ep->qpid);

    qp_context = DEVX_ADDR_OF(init2rtr_qp_in, cmd_in2, qpc);
    DEVX_SET(qpc, qp_context, primary_address_path.vhca_port_num, ep->portid);
    DEVX_SET(qpc, qp_context, mtu, port_attr->active_mtu);
    DEVX_SET(qpc, qp_context, log_msg_max, 30);
    DEVX_SET(qpc, qp_context, remote_qpn, ep_handle->qpn);
    DEVX_SET(qpc, qp_context, min_rnr_nak, 12);
    DEVX_SET(qpc, qp_context, rwe, 1); /* remote write access */
    DEVX_SET(qpc, qp_context, rre, 1); /* remote read access */
    DEVX_SET(qpc, qp_context, rae, 1); /* remote atomic access */
    /* Currently, NVSHMEM APIs only support atomics up to 64. This field can be updated to support
     * atomics up to 256 bytes. */
    DEVX_SET(qpc, qp_context, atomic_mode, NVSHMEMT_IBDEVX_MLX5_QPC_ATOMIC_MODE_UP_TO_64B);

    if (port_attr->link_layer == IBV_LINK_LAYER_ETHERNET) {
        ib_get_gid_index(&ftable, device->context, portid, port_attr->gid_tbl_len,
                         &device->gid_info[portid - 1].local_gid_index, ibdevx_state->log_level,
                         ibdevx_state->options);
        /*ftable.query_gid(device->context, portid, device->gid_info[portid - 1].local_gid_index,
                         &device->gid_info[portid - 1].local_gid);*/
        struct ibv_ah_attr ah_attr;
        struct ibv_ah *ah;
        struct mlx5dv_obj dv;
        struct mlx5dv_ah dah;

        ah_attr.is_global = 1;
        ah_attr.port_num = portid;
        ah_attr.grh.dgid.global.subnet_prefix = ep_handle->spn;
        ah_attr.grh.dgid.global.interface_id = ep_handle->iid;
        ah_attr.grh.sgid_index = device->gid_info[portid - 1].local_gid_index;
        ah_attr.grh.traffic_class = ibdevx_state->options->IB_TRAFFIC_CLASS;
        ah_attr.sl = ibdevx_state->options->IB_SL;
        ah_attr.src_path_bits = 0;

        ah = ftable.create_ah(device->pd, &ah_attr);
        NVSHMEMI_NULL_ERROR_JMP(ah, status, NVSHMEMX_ERROR_INTERNAL, out, "Unable to create ah.\n");

        dv.ah.in = ah;
        dv.ah.out = &dah;
        mlx5dv_init_obj(&dv, MLX5DV_OBJ_AH);

        memcpy(DEVX_ADDR_OF(qpc, qp_context, primary_address_path.rmac_47_32), &dah.av->rmac,
               sizeof(dah.av->rmac));
        DEVX_SET(qpc, qp_context, primary_address_path.hop_limit, 255);
        DEVX_SET(qpc, qp_context, primary_address_path.src_addr_index,
                 device->gid_info[portid - 1].local_gid_index);
        DEVX_SET(qpc, qp_context, primary_address_path.eth_prio, ibdevx_state->options->IB_SL);
        DEVX_SET(qpc, qp_context, primary_address_path.udp_sport, ah_attr.dlid);
        DEVX_SET(qpc, qp_context, primary_address_path.dscp,
                 ibdevx_state->options->IB_TRAFFIC_CLASS >> 2);

        memcpy(DEVX_ADDR_OF(qpc, qp_context, primary_address_path.rgid_rip), &dah.av->rgid,
               sizeof(dah.av->rgid));
    } else {
        DEVX_SET(qpc, qp_context, primary_address_path.tclass,
                 ibdevx_state->options->IB_TRAFFIC_CLASS);
        DEVX_SET(qpc, qp_context, primary_address_path.rlid, ep_handle->lid);
        DEVX_SET(qpc, qp_context, primary_address_path.mlid, 0);
        DEVX_SET(qpc, qp_context, primary_address_path.sl, ibdevx_state->options->IB_SL);
        DEVX_SET(qpc, qp_context, primary_address_path.grh, false);
    }

    if (NVSHMEMT_IBDEVX_MAX_RD_ATOMIC == 0) {
        DEVX_SET(qpc, qp_context, log_rra_max, 0);
    } else {
        DEVX_SET(qpc, qp_context, log_rra_max, (int)log2(NVSHMEMT_IBDEVX_MAX_RD_ATOMIC));
    }

    status =
        mlx5dv_devx_obj_modify(ep->devx_qp, cmd_in2, sizeof(cmd_in2), cmd_out2, sizeof(cmd_out2));
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Unable to set qp to state R2R with errno %d.\n", status);

    DEVX_SET(rtr2rts_qp_in, cmd_in3, opcode, MLX5_CMD_OP_RTR2RTS_QP);
    DEVX_SET(rtr2rts_qp_in, cmd_in3, opt_param_mask, 0x0);
    DEVX_SET(rtr2rts_qp_in, cmd_in3, qpn, ep->qpid);

    qp_context = DEVX_ADDR_OF(rtr2rts_qp_in, cmd_in3, qpc);
    if (NVSHMEMT_IBDEVX_MAX_RD_ATOMIC == 0) {
        DEVX_SET(qpc, qp_context, log_sra_max, 0);
    } else {
        DEVX_SET(qpc, qp_context, log_sra_max, (int)log2(NVSHMEMT_IBDEVX_MAX_RD_ATOMIC));
    }
    DEVX_SET(qpc, qp_context, retry_count, 7);
    DEVX_SET(qpc, qp_context, rnr_retry, 7);
    DEVX_SET(qpc, qp_context, next_send_psn, 0);
    DEVX_SET(qpc, qp_context, log_ack_req_freq, 0); /* ack every packet */
    DEVX_SET(qpc, qp_context, primary_address_path.ack_timeout, 20);

    status =
        mlx5dv_devx_obj_modify(ep->devx_qp, cmd_in3, sizeof(cmd_in3), cmd_out3, sizeof(cmd_out3));
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Unable to set qp to state R2S with errno %d.\n", status);

    connected_qp_count++;
out:
    return status;
}

int ep_get_handle(struct ibdevx_ep_handle *ep_handle, struct ibdevx_ep *ep) {
    int status = 0;
    transport_ibdevx_state_t *ibdevx_state = (transport_ibdevx_state_t *)ep->ibdevx_state;
    struct ibdevx_device *device = ((struct ibdevx_device *)ibdevx_state->devices + ep->devid);

    ep_handle->lid = device->port_attr[ep->portid - 1].lid;
    ep_handle->qpn = ep->qpid;
    if (ep_handle->lid == 0) {
        ep_handle->spn = device->gid_info[ep->portid - 1].local_gid.global.subnet_prefix;
        ep_handle->iid = device->gid_info[ep->portid - 1].local_gid.global.interface_id;
    }

    return status;
}

int setup_cst_loopback(transport_ibdevx_state_t *ibdevx_state, int dev_id) {
    int status = 0;
    struct ibdevx_ep_handle cst_ep_handle;

    status = ep_create(&ibdevx_cst_ep, dev_id, ibdevx_state);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ep_create cst failed \n");

    status = ep_get_handle(&cst_ep_handle, ibdevx_cst_ep);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ep_get_handle failed \n");

    status = ep_connect(ibdevx_cst_ep, &cst_ep_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ep_connect failed \n");
out:
    return status;
}

int nvshmemt_ibdevx_get_mem_handle(nvshmem_mem_handle_t *mem_handle,
                                   nvshmem_mem_handle_t *mem_handle_in, void *buf, size_t length,
                                   nvshmem_transport_t t, bool local_only) {
    int status = 0;
    struct nvshmem_transport *transport = (struct nvshmem_transport *)t;
    transport_ibdevx_state_t *ibdevx_state = (transport_ibdevx_state_t *)transport->state;
    struct ibdevx_device *device = ((struct ibdevx_device *)ibdevx_state->devices +
                                    ibdevx_state->dev_ids[ibdevx_state->selected_dev_id]);

    status = nvshmemt_ib_common_reg_mem_handle(&ftable, device->pd, mem_handle, buf, length,
                                               local_only, ibdevx_state->dmabuf_support,
                                               ibdevx_state->table, ibdevx_state->log_level,
                                               ibdevx_state->options->IB_ENABLE_RELAXED_ORDERING);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Unable to register memory handle.");

    if (local_dummy_mr.mr == NULL) {
        uint64_t *local_dummy_mem;

        local_dummy_mem = (uint64_t *)malloc(sizeof(*local_dummy_mem));
        NVSHMEMI_NULL_ERROR_JMP(local_dummy_mem, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                "local dummy mem allocation failed \n");
        local_dummy_mr.mr = ftable.reg_mr(device->pd, local_dummy_mem, sizeof(*local_dummy_mem),
                                          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                              IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
        NVSHMEMI_NULL_ERROR_JMP(local_dummy_mr.mr, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                "mem registration failed \n");
        local_dummy_mr.lkey = local_dummy_mr.mr->lkey;
        local_dummy_mr.rkey = local_dummy_mr.mr->rkey;
    }

out:
    return status;
}

int nvshmemt_ibdevx_release_mem_handle(nvshmem_mem_handle_t *mem_handle, nvshmem_transport_t t) {
    transport_ibdevx_state_t *ibdevx_state = (transport_ibdevx_state_t *)t->state;
    return nvshmemt_ib_common_release_mem_handle(&ftable, mem_handle, ibdevx_state->log_level);
}

int nvshmemt_ibdevx_finalize(nvshmem_transport_t transport) {
    // TODO: Add cleanup of internal state to this function.
    int status = 0;
    struct nvshmem_transport *t = (struct nvshmem_transport *)transport;
    transport_ibdevx_state_t *ibdevx_state = (transport_ibdevx_state_t *)t->state;

    for (int i = 0; i < ibdevx_state->ep_count * t->n_pes; i++) {
        ep_destroy(ibdevx_state->ep[i], ibdevx_state->selected_dev_id, ibdevx_state);
    }

    if (ibdevx_cst_ep) {
        ep_destroy(ibdevx_cst_ep, ibdevx_state->selected_dev_id, ibdevx_state);
    }

    if (local_dummy_mr.mr) {
        void *mem = local_dummy_mr.mr->addr;
        ftable.dereg_mr(local_dummy_mr.mr);
        free(mem);
        memset(&local_dummy_mr, 0, sizeof(local_dummy_mr));
    }

    if (ibdevx_state->devices) {
        for (int i = 0; i < ibdevx_state->n_dev_ids; i++) {
            int dev_id = ibdevx_state->dev_ids[i];
            struct ibdevx_device *device = ((struct ibdevx_device *)ibdevx_state->devices + dev_id);
            if (device->context) {
                status = device_destroy_shared_ep_resources(device);
                NVSHMEMI_NZ_ERROR_JMP(status, status, out,
                                      "Unable to destroy shared qp resources for device.\n");
                status = ftable.close_device(device->context);
                NVSHMEMI_NZ_ERROR_JMP(status, status, out, "Unable to close device.\n");
            }
        }
        free(ibdevx_state->devices);
    }

    // clear qp map
    qp_map.clear();

    if (transport->device_pci_paths) {
        for (int i = 0; i < transport->n_devices; i++) {
            free(transport->device_pci_paths[i]);
        }
        free(transport->device_pci_paths);
    }

    nvshmemt_ibv_ftable_fini(&ibv_handle);

    status = pthread_mutex_destroy(&ibdevx_mutex_send_progress);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "pthread_mutex_destroy failed\n");

out:
    return status;
}

int progress_send(transport_ibdevx_state_t *ibdevx_state) {
    int status = 0;
    int n_devs = ibdevx_state->n_dev_ids;

    pthread_mutex_lock(&ibdevx_mutex_send_progress);

    for (int i = 0; i < n_devs; i++) {
        struct mlx5_cqe64 *cqe;
        volatile struct mlx5_cqe64 *cqe_vol;
        int devid = ibdevx_state->dev_ids[i];
        int ownership;
        struct ibdevx_device *device = ((struct ibdevx_device *)ibdevx_state->devices + devid);
        int comp_code;

        if (!device->scq.cqn) continue;
        cqe = ((struct mlx5_cqe64 *)device->scq.buf + (device->scq.cur_idx % device->scq.num_cqe));
        cqe_vol = cqe;
        ownership = (device->scq.cur_idx / device->scq.num_cqe) % 2;
        if (!((cqe_vol->op_own & 0x00000001) ^ ownership) &&
            mlx5dv_get_cqe_opcode(cqe) != MLX5_CQE_INVALID) {
            /* Anything larger than that opcode indicates an error */
            comp_code = mlx5dv_get_cqe_opcode(cqe);
            if (likely(comp_code <= MLX5_CQE_RESIZE_CQ)) {
                unsigned int qpid =
                    ntohl(cqe_vol->sop_drop_qpn) & NVSHMEMT_IBDEVX_MASK_UPPER_BYTE_32;
                struct ibdevx_ep *ep = (struct ibdevx_ep *)qp_map.find((unsigned int)qpid)->second;
                assert(ep != NULL);
                /* Have to take 64 bit masked compare and swap into consideration. These take up 2
                 * entries in the wq. */
                if (likely((ntohl(cqe_vol->sop_drop_qpn) & NVSHMEMT_IBDEVX_MASK_LOWER_3_BYTES_32) !=
                           MLX5_OPCODE_ATOMIC_MASKED_CS << 24) ||
                    (ntohl(cqe_vol->byte_cnt) < 8)) {
                    ep->tail_op_id++;
                } else {
                    ep->tail_op_id += 2;
                }
                device->scq.cur_idx++;

                STORE_BARRIER();
                /* Update doorbell record */
                *device->scq.dbrec = htobe32((device->scq.cur_idx % device->scq.num_cqe) &
                                             NVSHMEMT_IBDEVX_MASK_UPPER_BYTE_32);
            } else {
                volatile struct mlx5_err_cqe *err = (volatile struct mlx5_err_cqe *)cqe;
                NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                   "Got ibdevx completion with error. hws: %x ves: %x, syn: %x",
                                   err->rsvd1[16], err->vendor_err_synd, err->syndrome);
            }
        }
    }

out:
    pthread_mutex_unlock(&ibdevx_mutex_send_progress);
    return status;
}

int nvshmemt_ibdevx_progress(nvshmem_transport_t t) {
    int status = 0;
    struct nvshmem_transport *transport = (struct nvshmem_transport *)t;
    transport_ibdevx_state_t *ibdevx_state = (transport_ibdevx_state_t *)transport->state;

    status = progress_send(ibdevx_state);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "progress_send failed, \n");

out:
    return status;
}

int check_poll_avail(struct ibdevx_ep *ep, int wait_predicate) {
    int status = 0;
    uint32_t outstanding_count;

    assert(ibdevx_qp_depth > 1);
    if (wait_predicate == WAIT_ANY) {
        outstanding_count = (ibdevx_qp_depth - 1);
    } else if (wait_predicate == WAIT_TWO) {
        outstanding_count = (ibdevx_qp_depth - 2);
    } else {
        outstanding_count = 0;
    }
    transport_ibdevx_state_t *ibdevx_state = (transport_ibdevx_state_t *)ep->ibdevx_state;

    /* poll until space becomes in local send qp */
    while (((ep->head_op_id - ep->tail_op_id) > outstanding_count)) {
        status = progress_send(ibdevx_state);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "progress_send failed, outstanding_count: %d\n", outstanding_count);
    }

out:
    return status;
}

static inline void nvshmemt_ibdevx_post_send(struct ibdevx_ep *ep, void *bb,
                                             uint32_t num_wqe_cons) {
    void *bf_reg;

    STORE_BARRIER();
    ep->dbr_buf->send_counter = htobe16(ep->wqe_bb_idx);
    STORE_BARRIER();
    assert(bb != NULL);
    bf_reg = ((char *)ep->bf_reg + ep->bf_reg_offset);
    memcpy(bf_reg, bb, 8);
    ep->head_op_id += num_wqe_cons;
}

int nvshmemt_ibdevx_rma(struct nvshmem_transport *tcurr, int pe, rma_verb_t verb,
                        rma_memdesc_t *remote, rma_memdesc_t *local, rma_bytesdesc_t bytesdesc,
                        int is_proxy) {
    struct ibdevx_ep *ep;
    struct ibdevx_rw_wqe *wqe;
    transport_ibdevx_state_t *ibdevx_state = (transport_ibdevx_state_t *)tcurr->state;
    int status = 0;

    uintptr_t wqe_bb_idx_64;
    uint32_t wqe_bb_idx_32;
    size_t wqe_size;

    if (is_proxy) {
        ep = ibdevx_state->ep[(ibdevx_state->ep_count * pe + ibdevx_state->proxy_ep_idx)];
    } else {
        ep = ibdevx_state->ep[(ibdevx_state->ep_count * pe)];
    }

    wqe_bb_idx_64 = ep->wqe_bb_idx;
    wqe_bb_idx_32 = ep->wqe_bb_idx;

    status = check_poll_avail(ep, WAIT_ANY);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "check_poll failed \n");

    wqe = (struct ibdevx_rw_wqe *)((char *)ep->wq_buf + ((wqe_bb_idx_64 % ibdevx_qp_depth)
                                                         << NVSHMEMT_IBDEVX_WQE_BB_SHIFT));
    wqe_size = sizeof(struct ibdevx_rw_wqe);
    memset(wqe, 0, sizeof(struct ibdevx_rw_wqe));

    wqe->ctrl.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
    wqe->ctrl.qpn_ds =
        htobe32((uint32_t)(wqe_size / NVSHMEMT_IBDEVX_MLX5_SEND_WQE_DS) | ep->qpid << 8);
    if (verb.desc == NVSHMEMI_OP_GET || verb.desc == NVSHMEMI_OP_G) {
        wqe->ctrl.opmod_idx_opcode = htobe32(MLX5_OPCODE_RDMA_READ | (wqe_bb_idx_32 << 8));
    } else if (verb.desc == NVSHMEMI_OP_PUT || verb.desc == NVSHMEMI_OP_P) {
        wqe->ctrl.opmod_idx_opcode = htobe32(MLX5_OPCODE_RDMA_WRITE | (wqe_bb_idx_32 << 8));
    }

    /* TODO: store the rkeys in BE so we don't have to convert. */
    wqe->raddr.raddr = htobe64((uintptr_t)remote->ptr);
    wqe->raddr.rkey = htobe32(((struct nvshmemt_ib_common_mem_handle *)remote->handle)->rkey);

    if (verb.desc != NVSHMEMI_OP_P) {
        assert(bytesdesc.nelems < (UINT32_MAX / bytesdesc.elembytes));
        wqe->data.data_seg.byte_count = htobe32((uint32_t)(bytesdesc.nelems * bytesdesc.elembytes));
        wqe->data.data_seg.lkey =
            htobe32(((struct nvshmemt_ib_common_mem_handle *)local->handle)->lkey);
        wqe->data.data_seg.addr = htobe64((uintptr_t)local->ptr);
    } else {
        uint32_t bytecount = bytesdesc.nelems * bytesdesc.elembytes;
        /* set inline byte. */
        wqe->data.data_inl.byte_count = htobe32(bytecount | 0x80000000);
        switch (bytecount) {
            case 8: {
                wqe->data.data_inl.data.data_64 = *(uint64_t *)local->ptr;
                break;
            };
            case 4: {
                wqe->data.data_inl.data.data_32 = *(uint32_t *)local->ptr;
                break;
            };
            case 2: {
                wqe->data.data_inl.data.data_16 = *(uint16_t *)local->ptr;
                break;
            };
            case 1: {
                wqe->data.data_inl.data.data_8 = *(uint8_t *)local->ptr;
                break;
            };
            default: {
                NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                                   "Invalid length argument to p: %u\n", bytecount);
            };
        }
    }

    /* A wqe_bb is 64 bytes. Our wqe takes up less than 64 bytes, so increase the count by one. */
    assert(wqe_size <= MLX5_SEND_WQE_BB);
    ep->wqe_bb_idx++;
    nvshmemt_ibdevx_post_send(ep, (void *)wqe, 1);

    if (unlikely(!verb.is_nbi && verb.desc != NVSHMEMI_OP_P)) {
        check_poll_avail(ep, WAIT_ALL /*1*/);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "check_poll failed \n");
    }

out:
    return status;
}

static inline int nvshmemt_ibdevx_amo_32(struct nvshmem_transport *tcurr, int pe, void *curetptr,
                                         amo_verb_t verb, amo_memdesc_t *remote,
                                         amo_bytesdesc_t bytesdesc, int is_proxy) {
    struct ibdevx_ep *ep;
    struct ibdevx_atomic_32_wqe *wqe;
    struct nvshmemt_ib_common_mem_handle *ret_handle;
    transport_ibdevx_state_t *ibdevx_state = (transport_ibdevx_state_t *)tcurr->state;
    int status = 0;

    size_t wqe_size;
    uintptr_t wqe_bb_idx_64;
    uint32_t wqe_bb_idx_32;
    uint32_t swap_add_value = remote->val;
    uint32_t compare = remote->cmp;

    if (is_proxy) {
        ep = ibdevx_state->ep[(ibdevx_state->ep_count * pe + ibdevx_state->proxy_ep_idx)];
    } else {
        ep = ibdevx_state->ep[(ibdevx_state->ep_count * pe)];
    }

    wqe_bb_idx_64 = ep->wqe_bb_idx;
    wqe_bb_idx_32 = ep->wqe_bb_idx;

    wqe = (struct ibdevx_atomic_32_wqe *)((char *)ep->wq_buf + ((wqe_bb_idx_64 % ibdevx_qp_depth)
                                                                << NVSHMEMT_IBDEVX_WQE_BB_SHIFT));

    status = check_poll_avail(ep, WAIT_ANY);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "check_poll failed \n");

    wqe_size = sizeof(struct ibdevx_atomic_32_wqe);

    wqe->ctrl.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
    wqe->ctrl.qpn_ds =
        htobe32((uint32_t)(wqe_size / NVSHMEMT_IBDEVX_MLX5_SEND_WQE_DS) | ep->qpid << 8);
    wqe->raddr.raddr = htobe64((uintptr_t)remote->remote_memdesc.ptr);
    wqe->raddr.rkey =
        htobe32(((struct nvshmemt_ib_common_mem_handle *)remote->remote_memdesc.handle)->rkey);
    wqe->data.byte_count = htobe32((uint32_t)4);

    if (verb.desc < NVSHMEMI_AMO_END_OF_NONFETCH) {
        wqe->data.lkey = htobe32(local_dummy_mr.lkey);
        wqe->data.addr = htobe64((uintptr_t)local_dummy_mr.mr->addr);
    } else {
        ret_handle = (struct nvshmemt_ib_common_mem_handle *)remote->ret_handle;
        assert(ret_handle != NULL);
        wqe->data.lkey = htobe32(ret_handle->lkey);
        wqe->data.addr = htobe64((uintptr_t)remote->retptr);
    }

    switch (verb.desc) {
        case NVSHMEMI_AMO_FETCH_INC:
        case NVSHMEMI_AMO_INC: {
            wqe->ctrl.opmod_idx_opcode =
                htobe32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqe_bb_idx_32 << 8) |
                        NVSHMEMT_IBDEVX_4_BYTE_EXT_AMO_OPMOD);
            wqe->fa_seg.add_data = htobe32(0x00000001);
            wqe->fa_seg.field_boundary = 0;
            break;
        }
        case NVSHMEMI_AMO_SIGNAL:
        case NVSHMEMI_AMO_SIGNAL_SET:
        case NVSHMEMI_AMO_SWAP:
        case NVSHMEMI_AMO_SET: {
            wqe->ctrl.opmod_idx_opcode =
                htobe32(MLX5_OPCODE_ATOMIC_MASKED_CS | (wqe_bb_idx_32 << 8) |
                        NVSHMEMT_IBDEVX_4_BYTE_EXT_AMO_OPMOD);
            wqe->cs_seg.swap_data = htobe32(swap_add_value);
            wqe->cs_seg.compare_data = 0;
            wqe->cs_seg.compare_mask = 0;
            wqe->cs_seg.swap_mask = UINT32_MAX;
            break;
        }
        case NVSHMEMI_AMO_SIGNAL_ADD:
        case NVSHMEMI_AMO_ADD: {
            wqe->ctrl.opmod_idx_opcode =
                htobe32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqe_bb_idx_32 << 8) |
                        NVSHMEMT_IBDEVX_4_BYTE_EXT_AMO_OPMOD);
            wqe->fa_seg.add_data = htobe32(swap_add_value);
            wqe->fa_seg.field_boundary = 0;
            break;
        }
        case NVSHMEMI_AMO_FETCH_AND:
        case NVSHMEMI_AMO_AND: {
            wqe->ctrl.opmod_idx_opcode =
                htobe32(MLX5_OPCODE_ATOMIC_MASKED_CS | (wqe_bb_idx_32 << 8) |
                        NVSHMEMT_IBDEVX_4_BYTE_EXT_AMO_OPMOD);
            wqe->cs_seg.swap_data = htobe32(swap_add_value);
            wqe->cs_seg.compare_data = 0;
            wqe->cs_seg.compare_mask = 0;
            wqe->cs_seg.swap_mask = htobe32(~swap_add_value);
            break;
        }
        case NVSHMEMI_AMO_FETCH_OR:
        case NVSHMEMI_AMO_OR: {
            wqe->ctrl.opmod_idx_opcode =
                htobe32(MLX5_OPCODE_ATOMIC_MASKED_CS | (wqe_bb_idx_32 << 8) |
                        NVSHMEMT_IBDEVX_4_BYTE_EXT_AMO_OPMOD);
            wqe->cs_seg.swap_data = htobe32(swap_add_value);
            wqe->cs_seg.compare_data = 0;
            wqe->cs_seg.compare_mask = 0;
            wqe->cs_seg.swap_mask = htobe32(swap_add_value);
            break;
        }
        case NVSHMEMI_AMO_FETCH_XOR:
        case NVSHMEMI_AMO_XOR: {
            wqe->ctrl.opmod_idx_opcode =
                htobe32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqe_bb_idx_32 << 8) |
                        NVSHMEMT_IBDEVX_4_BYTE_EXT_AMO_OPMOD);
            wqe->fa_seg.add_data = htobe32(swap_add_value);
            wqe->fa_seg.field_boundary = UINT32_MAX;
            break;
        }
        case NVSHMEMI_AMO_FETCH: {
            wqe->ctrl.opmod_idx_opcode =
                htobe32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqe_bb_idx_32 << 8) |
                        NVSHMEMT_IBDEVX_4_BYTE_EXT_AMO_OPMOD);
            wqe->fa_seg.add_data = 0;
            wqe->fa_seg.field_boundary = 0;
            break;
        }
        case NVSHMEMI_AMO_FETCH_ADD: {
            wqe->ctrl.opmod_idx_opcode =
                htobe32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqe_bb_idx_32 << 8) |
                        NVSHMEMT_IBDEVX_4_BYTE_EXT_AMO_OPMOD);
            wqe->fa_seg.add_data = htobe32(swap_add_value);
            wqe->fa_seg.field_boundary = 0;
            break;
        }
        case NVSHMEMI_AMO_COMPARE_SWAP: {
            wqe->ctrl.opmod_idx_opcode =
                htobe32(MLX5_OPCODE_ATOMIC_MASKED_CS | (wqe_bb_idx_32 << 8) |
                        NVSHMEMT_IBDEVX_4_BYTE_EXT_AMO_OPMOD);
            wqe->cs_seg.swap_data = htobe32(swap_add_value);
            wqe->cs_seg.compare_data = htobe32(compare);
            wqe->cs_seg.compare_mask = UINT32_MAX;
            wqe->cs_seg.swap_mask = UINT32_MAX;
            break;
        }
        default: {
            NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out, "Opcode %d is invalid.\n",
                               verb.desc);
        }
    }

    assert(wqe_size <= MLX5_SEND_WQE_BB);
    ep->wqe_bb_idx++;
    nvshmemt_ibdevx_post_send(ep, (void *)wqe, 1);

out:
    return status;
}

static inline int nvshmemt_ibdevx_amo_64(struct nvshmem_transport *tcurr, int pe, void *curetptr,
                                         amo_verb_t verb, amo_memdesc_t *remote,
                                         amo_bytesdesc_t bytesdesc, int is_proxy) {
    struct ibdevx_ep *ep;
    struct mlx5_wqe_ctrl_seg *ctrl;
    struct mlx5_wqe_raddr_seg *raddr;
    struct mlx5_wqe_data_seg *data = NULL, *fa_data = NULL;
    struct ibdevx_atomic_64_amo_wqe *wqe;
    struct ibdevx_atomic_64_masked_compare_swap_seg *cs_data, *cs_mask;
    struct nvshmemt_ib_common_mem_handle *ret_handle;
    transport_ibdevx_state_t *ibdevx_state = (transport_ibdevx_state_t *)tcurr->state;

    void *wqe_1, *wqe_2;

    int status = 0;

    size_t wqe_size;
    uintptr_t wqe_bb_idx_64;
    uint32_t wqe_bb_idx_32;

    if (is_proxy) {
        ep = ibdevx_state->ep[(ibdevx_state->ep_count * pe + ibdevx_state->proxy_ep_idx)];
    } else {
        ep = ibdevx_state->ep[(ibdevx_state->ep_count * pe)];
    }

    wqe_bb_idx_64 = ep->wqe_bb_idx;
    wqe_bb_idx_32 = ep->wqe_bb_idx;

    wqe_1 = (void *)((char *)ep->wq_buf +
                     ((wqe_bb_idx_64 % ibdevx_qp_depth) << NVSHMEMT_IBDEVX_WQE_BB_SHIFT));
    /* only needed if we are doing a compare_swap operation. */
    wqe_2 = (void *)((char *)ep->wq_buf +
                     (((wqe_bb_idx_64 + 1) % ibdevx_qp_depth) << NVSHMEMT_IBDEVX_WQE_BB_SHIFT));

    /* The 64 bit masked compare swap wqe takes up 80 bytes, or two entries in the WQ. */
    status = check_poll_avail(ep, WAIT_TWO);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "check_poll failed \n");
    ctrl = (struct mlx5_wqe_ctrl_seg *)wqe_1;
    raddr = (struct mlx5_wqe_raddr_seg *)((char *)ctrl + sizeof(struct mlx5_wqe_ctrl_seg));
    cs_data =
        (struct ibdevx_atomic_64_masked_compare_swap_seg *)((char *)raddr +
                                                            sizeof(struct mlx5_wqe_raddr_seg));
    cs_mask = (struct ibdevx_atomic_64_masked_compare_swap_seg
                   *)((char *)cs_data + sizeof(struct ibdevx_atomic_64_masked_compare_swap_seg));
    fa_data = (struct mlx5_wqe_data_seg *)((char *)wqe_1 + sizeof(struct mlx5_wqe_ctrl_seg) +
                                           sizeof(struct mlx5_wqe_raddr_seg) +
                                           sizeof(struct ibdevx_atomic_64_masked_fetch_add_seg));
    raddr->raddr = htobe64((uintptr_t)remote->remote_memdesc.ptr);
    raddr->rkey =
        htobe32(((struct nvshmemt_ib_common_mem_handle *)remote->remote_memdesc.handle)->rkey);

    switch (verb.desc) {
        case NVSHMEMI_AMO_FETCH_INC:
        case NVSHMEMI_AMO_INC: {
            wqe = (struct ibdevx_atomic_64_amo_wqe *)ctrl;
            wqe->ctrl.opmod_idx_opcode =
                htobe32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqe_bb_idx_32 << 8) |
                        NVSHMEMT_IBDEVX_8_BYTE_EXT_AMO_OPMOD);
            wqe->fa_seg.add_data = htobe64(0x0000000000000001);
            wqe->fa_seg.field_boundary = 0;
            data = fa_data;
            wqe_size = sizeof(struct ibdevx_atomic_64_amo_wqe);
            break;
        }
        case NVSHMEMI_AMO_SIGNAL:
        case NVSHMEMI_AMO_SIGNAL_SET:
        case NVSHMEMI_AMO_SWAP:
        case NVSHMEMI_AMO_SET: {
            ctrl->opmod_idx_opcode = htobe32(MLX5_OPCODE_ATOMIC_MASKED_CS | (wqe_bb_idx_32 << 8) |
                                             NVSHMEMT_IBDEVX_8_BYTE_EXT_AMO_OPMOD);
            cs_data->swap = htobe64(remote->val);
            cs_data->compare = 0;
            cs_mask->compare = 0;
            cs_mask->swap = UINT64_MAX;
            data = (struct mlx5_wqe_data_seg *)wqe_2;
            wqe_size = 80; /* ctrl_seg + raddr_seg + compare_swap_seg * 2 + data_seg = 80 bytes */
            break;
        }
        case NVSHMEMI_AMO_SIGNAL_ADD:
        case NVSHMEMI_AMO_ADD: {
            wqe = (struct ibdevx_atomic_64_amo_wqe *)ctrl;
            wqe->ctrl.opmod_idx_opcode =
                htobe32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqe_bb_idx_32 << 8) |
                        NVSHMEMT_IBDEVX_8_BYTE_EXT_AMO_OPMOD);
            wqe->fa_seg.add_data = htobe64(remote->val);
            wqe->fa_seg.field_boundary = 0;
            data = fa_data;
            wqe_size = sizeof(struct ibdevx_atomic_64_amo_wqe);
            break;
        }
        case NVSHMEMI_AMO_FETCH_AND:
        case NVSHMEMI_AMO_AND: {
            ctrl->opmod_idx_opcode = htobe32(MLX5_OPCODE_ATOMIC_MASKED_CS | (wqe_bb_idx_32 << 8) |
                                             NVSHMEMT_IBDEVX_8_BYTE_EXT_AMO_OPMOD);
            cs_data->swap = htobe64(remote->val);
            cs_data->compare = 0;
            cs_mask->compare = 0;
            cs_mask->swap = htobe64(~remote->val);
            data = (struct mlx5_wqe_data_seg *)wqe_2;
            wqe_size = 80; /* ctrl_seg + raddr_seg + compare_swap_seg * 2 + data_seg = 80 bytes */
            break;
        }
        case NVSHMEMI_AMO_FETCH_OR:
        case NVSHMEMI_AMO_OR: {
            ctrl->opmod_idx_opcode = htobe32(MLX5_OPCODE_ATOMIC_MASKED_CS | (wqe_bb_idx_32 << 8) |
                                             NVSHMEMT_IBDEVX_8_BYTE_EXT_AMO_OPMOD);
            cs_data->swap = htobe64(remote->val);
            cs_data->compare = 0;
            cs_mask->compare = 0;
            cs_mask->swap = htobe64(remote->val);
            data = (struct mlx5_wqe_data_seg *)wqe_2;
            wqe_size = 80; /* ctrl_seg + raddr_seg + compare_swap_seg * 2 + data_seg = 80 bytes */
            break;
        }
        case NVSHMEMI_AMO_FETCH_XOR:
        case NVSHMEMI_AMO_XOR: {
            wqe = (struct ibdevx_atomic_64_amo_wqe *)ctrl;
            wqe->ctrl.opmod_idx_opcode =
                htobe32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqe_bb_idx_32 << 8) |
                        NVSHMEMT_IBDEVX_8_BYTE_EXT_AMO_OPMOD);
            wqe->fa_seg.add_data = htobe64(remote->val);
            wqe->fa_seg.field_boundary = UINT64_MAX;
            data = fa_data;
            wqe_size = sizeof(struct ibdevx_atomic_64_amo_wqe);
            break;
        }
        case NVSHMEMI_AMO_FETCH: {
            wqe = (struct ibdevx_atomic_64_amo_wqe *)ctrl;
            wqe->ctrl.opmod_idx_opcode =
                htobe32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqe_bb_idx_32 << 8) |
                        NVSHMEMT_IBDEVX_8_BYTE_EXT_AMO_OPMOD);
            wqe->fa_seg.add_data = 0;
            wqe->fa_seg.field_boundary = 0;
            data = fa_data;
            wqe_size = sizeof(struct ibdevx_atomic_64_amo_wqe);
            break;
        }
        /* opmod is 0 */
        case NVSHMEMI_AMO_FETCH_ADD: {
            wqe = (struct ibdevx_atomic_64_amo_wqe *)ctrl;
            wqe->ctrl.opmod_idx_opcode = htobe32(MLX5_OPCODE_ATOMIC_FA | (wqe_bb_idx_32 << 8));
            wqe->no_mask_seg.swap_add = htobe64(remote->val);
            data = fa_data;
            wqe_size = sizeof(struct ibdevx_atomic_64_amo_wqe);
            break;
        }
        /* opmod is 0 */
        case NVSHMEMI_AMO_COMPARE_SWAP: {
            wqe = (struct ibdevx_atomic_64_amo_wqe *)ctrl;
            wqe->ctrl.opmod_idx_opcode = htobe32(MLX5_OPCODE_ATOMIC_CS | (wqe_bb_idx_32 << 8));
            wqe->no_mask_seg.swap_add = htobe64(remote->val);
            wqe->no_mask_seg.compare = htobe64(remote->cmp);
            data = fa_data;
            wqe_size = sizeof(struct ibdevx_atomic_64_amo_wqe);
            break;
        }
        default: {
            NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out, "Opcode %d is invalid.\n",
                               verb.desc);
        }
    }

    data->byte_count = htobe32((uint32_t)8);

    if (verb.desc < NVSHMEMI_AMO_END_OF_NONFETCH) {
        data->lkey = htobe32(local_dummy_mr.lkey);
        data->addr = htobe64((uintptr_t)local_dummy_mr.mr->addr);
    } else {
        ret_handle = (struct nvshmemt_ib_common_mem_handle *)remote->ret_handle;
        assert(ret_handle != NULL);
        data->lkey = htobe32(ret_handle->lkey);
        data->addr = htobe64((uintptr_t)remote->retptr);
    }

    ctrl->fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
    ctrl->qpn_ds = htobe32((uint32_t)(wqe_size / NVSHMEMT_IBDEVX_MLX5_SEND_WQE_DS) | ep->qpid << 8);
    if (wqe_size > MLX5_SEND_WQE_BB) {
        assert(wqe_size <= (MLX5_SEND_WQE_BB * 2));
        ep->wqe_bb_idx += 2;
        nvshmemt_ibdevx_post_send(ep, (void *)wqe_1, 2);
    } else {
        ep->wqe_bb_idx++;
        nvshmemt_ibdevx_post_send(ep, (void *)wqe_1, 1);
    }

out:
    return status;
}

int nvshmemt_ibdevx_amo(struct nvshmem_transport *tcurr, int pe, void *curetptr, amo_verb_t verb,
                        amo_memdesc_t *remote, amo_bytesdesc_t bytesdesc, int is_proxy) {
    int status = 0;

    if (bytesdesc.elembytes == 4) {
        return nvshmemt_ibdevx_amo_32(tcurr, pe, curetptr, verb, remote, bytesdesc, is_proxy);
    } else if (bytesdesc.elembytes == 8) {
        return nvshmemt_ibdevx_amo_64(tcurr, pe, curetptr, verb, remote, bytesdesc, is_proxy);
    } else {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "Invalid atomic length %d specified.\n", bytesdesc.elembytes);
    }

out:
    return status;
}

int nvshmemt_ibdevx_enforce_cst_at_target(struct nvshmem_transport *tcurr) {
    struct ibdevx_ep *ep = ibdevx_cst_ep;
    struct ibdevx_rw_wqe *wqe;

    int status = 0;

    uintptr_t wqe_bb_idx_64 = ep->wqe_bb_idx;
    uint32_t wqe_bb_idx_32 = ep->wqe_bb_idx;
    size_t wqe_size;

    wqe = (struct ibdevx_rw_wqe *)((char *)ep->wq_buf + ((wqe_bb_idx_64 % ibdevx_qp_depth)
                                                         << NVSHMEMT_IBDEVX_WQE_BB_SHIFT));
    wqe_size = sizeof(struct ibdevx_rw_wqe);
    memset(wqe, 0, sizeof(struct ibdevx_rw_wqe));

    wqe->ctrl.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
    wqe->ctrl.qpn_ds =
        htobe32((uint32_t)(wqe_size / NVSHMEMT_IBDEVX_MLX5_SEND_WQE_DS) | ep->qpid << 8);
    wqe->ctrl.opmod_idx_opcode = htobe32(MLX5_OPCODE_RDMA_READ | (wqe_bb_idx_32 << 8));

    wqe->raddr.raddr = htobe64((uintptr_t)local_dummy_mr.mr->addr);
    wqe->raddr.rkey = htobe32(local_dummy_mr.rkey);

    wqe->data.data_seg.byte_count = htobe32((uint32_t)4);
    wqe->data.data_seg.lkey = htobe32(local_dummy_mr.lkey);
    wqe->data.data_seg.addr = htobe64((uintptr_t)local_dummy_mr.mr->addr);

    assert(wqe_size <= MLX5_SEND_WQE_BB);
    ep->wqe_bb_idx++;
    nvshmemt_ibdevx_post_send(ep, (void *)wqe, 1);

    status = check_poll_avail(ep, WAIT_ALL);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "check_poll failed \n");

out:
    return status;
}

int nvshmemt_ibdevx_quiet(struct nvshmem_transport *tcurr, int pe, int is_proxy) {
    transport_ibdevx_state_t *ibdevx_state = (transport_ibdevx_state_t *)tcurr->state;
    struct ibdevx_ep *ep;
    int status = 0;

    if (is_proxy) {
        ep = ibdevx_state->ep[(pe * ibdevx_state->ep_count + ibdevx_state->proxy_ep_idx)];
    } else {
        ep = ibdevx_state->ep[(pe * ibdevx_state->ep_count)];
    }

    status = check_poll_avail(ep, WAIT_ALL /*1*/);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "check_poll failed \n");

out:
    return status;
}

int nvshmemt_ibdevx_ep_create(struct ibdevx_ep **ep, int devid,
                              transport_ibdevx_state_t *ibdevx_state) {
    int status = 0;

    status = ep_create(ep, devid, ibdevx_state);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ep_create failed\n");

    // setup loopback connection on the first device used.
    if (!ibdevx_cst_ep) {
        status = setup_cst_loopback(ibdevx_state, devid);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cst setup failed \n");
    }

out:
    return status;
}

int nvshmemt_ibdevx_ep_get_handle(struct ibdevx_ep_handle *ep_handle_ptr, struct ibdevx_ep *ep) {
    int status = 0;

    status = ep_get_handle(ep_handle_ptr, ep);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ep_get_handle failed \n");

out:
    return status;
}

int nvshmemt_ibdevx_ep_destroy(struct ibdevx_ep *ep) {
    int status = 0;

    status = check_poll_avail(ep, WAIT_ALL /*1*/);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "check_poll failed \n");

    // TODO: clean up qp, cq, etc.

out:
    return status;
}

int nvshmemt_ibdevx_ep_connect(struct ibdevx_ep *ep, struct ibdevx_ep_handle *ep_handle) {
    int status = 0;

    status = ep_connect(ep, ep_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ep_connect failed \n");

out:
    return status;
}

int nvshmemt_ibdevx_connect_endpoints(nvshmem_transport_t t, int *selected_dev_ids,
                                      int num_selected_devs) {
    /* transport side */
    struct ibdevx_ep_handle *local_ep_handles = NULL, *ep_handles = NULL;
    transport_ibdevx_state_t *ibdevx_state = (transport_ibdevx_state_t *)t->state;
    int status = 0;

    int n_pes = t->n_pes;

    int ep_count = ibdevx_state->ep_count = MAX_TRANSPORT_EP_COUNT + 1;

    if (ibdevx_state->selected_dev_id >= 0) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out_already_connected,
                           "Device already selected. IBDEVX only supports"
                           " one NIC per PE.\n");
    }

    if (num_selected_devs > 1) {
        INFO(ibdevx_state->log_level,
             "IBDEVX only supports one NIC / PE. All other NICs will be ignored.");
    }

    /* allocate all EPs for transport, plus 1 for the proxy thread. */
    ibdevx_state->proxy_ep_idx = MAX_TRANSPORT_EP_COUNT;
    ibdevx_state->selected_dev_id = selected_dev_ids[0];

    ibdevx_state->ep = (struct ibdevx_ep **)calloc(n_pes * ep_count, sizeof(struct ibdevx_ep *));
    NVSHMEMI_NULL_ERROR_JMP(ibdevx_state->ep, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "failed allocating space for endpoints \n");

    local_ep_handles =
        (struct ibdevx_ep_handle *)calloc(n_pes * ep_count, sizeof(struct ibdevx_ep_handle));
    NVSHMEMI_NULL_ERROR_JMP(local_ep_handles, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "failed allocating space for local ep handles \n");

    ep_handles =
        (struct ibdevx_ep_handle *)calloc(n_pes * ep_count, sizeof(struct ibdevx_ep_handle));
    NVSHMEMI_NULL_ERROR_JMP(ep_handles, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "failed allocating space for ep handles \n");

    for (int j = 0; j < n_pes; j++) {
        for (int k = 0; k < ep_count; k++) {
            nvshmemt_ibdevx_ep_create(&ibdevx_state->ep[j * ep_count + k],
                                      ibdevx_state->selected_dev_id, ibdevx_state);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "transport create ep failed \n");
            status = nvshmemt_ibdevx_ep_get_handle(&local_ep_handles[j * ep_count + k],
                                                   ibdevx_state->ep[j * ep_count + k]);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "transport get ep handle failed \n");
        }
    }

    status = t->boot_handle->alltoall((void *)local_ep_handles, (void *)ep_handles,
                                      sizeof(struct ibdevx_ep_handle) * ep_count, t->boot_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "allgather of ep handles failed \n");

    for (int j = 0; j < n_pes; j++) {
        for (int k = 0; k < ep_count; k++) {
            status = nvshmemt_ibdevx_ep_connect(ibdevx_state->ep[j * ep_count + k],
                                                &ep_handles[j * ep_count + k]);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "transport create connect failed \n");
        }
    }
out:
    if (status) {
        ibdevx_state->selected_dev_id = -1;
        if (ibdevx_state->ep) free(ibdevx_state->ep);
    }

out_already_connected:
    if (local_ep_handles) free(local_ep_handles);
    if (ep_handles) free(ep_handles);
    return status;
}

int nvshmemt_init(nvshmem_transport_t *t, struct nvshmemi_cuda_fn_table *table, int api_version) {
    int status = 0;
    struct nvshmem_transport *transport = NULL;
    transport_ibdevx_state_t *ibdevx_state = NULL;
    struct ibv_device **dev_list = NULL;
    int num_devices;
    struct ibdevx_device *device;
    std::vector<std::string> nic_names_n_pes;
    std::vector<std::string> nic_names;
    int exclude_list = 0;
    struct nvshmemt_hca_info hca_list[MAX_NUM_HCAS];
    struct nvshmemt_hca_info pe_hca_mapping[MAX_NUM_PES_PER_NODE];
    int hca_list_count = 0, pe_hca_map_count = 0, user_selection = 0;
    int offset = 0;
    int log_qp_depth;
    uint32_t atomic_host_endian_size = 0;
    int flag;
    CUdevice gpu_device_id;

    if (NVSHMEM_TRANSPORT_MAJOR_VERSION(api_version) != NVSHMEM_TRANSPORT_PLUGIN_MAJOR_VERSION) {
        NVSHMEMI_ERROR_PRINT(
            "NVSHMEM provided an incompatible version of the transport interface. "
            "This transport supports transport API major version %d. Host has %d",
            NVSHMEM_TRANSPORT_PLUGIN_MAJOR_VERSION, NVSHMEM_TRANSPORT_MAJOR_VERSION(api_version));
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    memset(&local_dummy_mr, 0, sizeof(local_dummy_mr));

    transport = (struct nvshmem_transport *)malloc(sizeof(struct nvshmem_transport));
    memset(transport, 0, sizeof(struct nvshmem_transport));
    transport->is_successfully_initialized =
        false; /* set it to true after everything has been successfully initialized */

    ibdevx_state = (transport_ibdevx_state_t *)calloc(1, sizeof(transport_ibdevx_state_t));
    NVSHMEMI_NULL_ERROR_JMP(ibdevx_state, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "ibdevx state allocation failed \n");

    /* set selected device ID to -1 to indicate none is selected. */
    ibdevx_state->selected_dev_id = -1;
    transport->state = (void *)ibdevx_state;

    ibdevx_state->options =
        (struct nvshmemi_options_s *)calloc(1, sizeof(struct nvshmemi_options_s));
    NVSHMEMI_NULL_ERROR_JMP(ibdevx_state, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "ibdevx options allocation failed \n");

    status = nvshmemi_env_options_init(ibdevx_state->options);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "Unable to initialize options.\n");

    ibdevx_state->log_level = nvshmemt_common_get_log_level(ibdevx_state->options);

    if (nvshmemt_ibv_ftable_init(&ibv_handle, &ftable, ibdevx_state->log_level)) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                           "Unable to dlopen libibverbs. Skipping devx transport.");
    }

    if (ibdevx_state->options->DISABLE_IB_NATIVE_ATOMICS) {
        use_ib_native_atomics = 0;
    }
    ibdevx_srq_depth = ibdevx_state->options->SRQ_DEPTH;
    ibdevx_qp_depth = ibdevx_state->options->QP_DEPTH;
    log_qp_depth = (int)log2(ibdevx_qp_depth);
    if (log_qp_depth < 1) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "Invalid qp size specified. Please select "
                           "a value greater than or equal to 2.\n");
    }
    if (ibdevx_qp_depth > (1 << log_qp_depth)) {
        if (log_qp_depth >= 30) {
            NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                               "Invalid qp size specified. Please select "
                               "a value less than or equal to 2^30.\n");
        }
        /* DEVX requires a power of 2. So round up to the nearest power here. */
        ibdevx_qp_depth = 1 << (log_qp_depth + 1);
    }

    dev_list = ftable.get_device_list(&num_devices);
    NVSHMEMI_NULL_ERROR_JMP(dev_list, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "get_device_list failed \n");

    ibdevx_state->devices = calloc(MAX_NUM_HCAS, sizeof(struct ibdevx_device));
    NVSHMEMI_NULL_ERROR_JMP(ibdevx_state->devices, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "get_device_list failed \n");

    ibdevx_state->dev_ids = (int *)malloc(MAX_NUM_PES_PER_NODE * sizeof(int));
    NVSHMEMI_NULL_ERROR_JMP(ibdevx_state->dev_ids, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "malloc failed \n");

    ibdevx_state->port_ids = (int *)malloc(MAX_NUM_PES_PER_NODE * sizeof(int));
    NVSHMEMI_NULL_ERROR_JMP(ibdevx_state->port_ids, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "malloc failed \n");

    status = pthread_mutex_init(&ibdevx_mutex_send_progress, NULL);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "pthread_mutex_init failed \n");

    if (ibdevx_state->options->HCA_LIST_provided) {
        user_selection = 1;
        exclude_list = (ibdevx_state->options->HCA_LIST[0] == '^');
        hca_list_count = nvshmemt_parse_hca_list(ibdevx_state->options->HCA_LIST, hca_list,
                                                 MAX_NUM_HCAS, ibdevx_state->log_level);
    }

    if (ibdevx_state->options->HCA_PE_MAPPING_provided) {
        if (hca_list_count) {
            NVSHMEMI_WARN_PRINT(
                "Found conflicting parameters NVSHMEM_HCA_LIST and NVSHMEM_HCA_PE_MAPPING, "
                "ignoring "
                "NVSHMEM_HCA_PE_MAPPING \n");
        } else {
            user_selection = 1;
            pe_hca_map_count =
                nvshmemt_parse_hca_list(ibdevx_state->options->HCA_PE_MAPPING, pe_hca_mapping,
                                        MAX_NUM_PES_PER_NODE, ibdevx_state->log_level);
        }
    }

    INFO(ibdevx_state->log_level,
         "Begin - Enumerating IB devices in the system ([<dev_id, device_name, num_ports>]) - ");
    for (int i = 0; i < num_devices; i++) {
        device = (struct ibdevx_device *)ibdevx_state->devices + i;
        device->dev = dev_list[i];

        device->context = ftable.open_device(device->dev);
        if (!device->context) {
            INFO(ibdevx_state->log_level, "open_device failed for IB device at index %d", i);
            continue;
        }

        const char *name = ftable.get_device_name(device->dev);
        NVSHMEMI_NULL_ERROR_JMP(name, status, NVSHMEMX_ERROR_INTERNAL, out,
                                "ibv_get_device_name failed \n");
        if (!strstr(name, "mlx5")) {
            ftable.close_device(device->context);
            device->context = NULL;
            NVSHMEMI_WARN_PRINT("device %s is not enumerated as an mlx5 device. Skipping...", name);
            continue;
        }

        status = ftable.query_device(device->context, &device->device_attr);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_query_device failed \n");

        if (!nvshmemt_ib_common_query_mlx5_caps(device->context)) {
            ftable.close_device(device->context);
            device->context = NULL;
            NVSHMEMI_WARN_PRINT("device %s is not enumerated as an mlx5 device. Skipping...", name);
            continue;
        }

        /* Report whether we need to do atomic endianness conversions on 8 byte operands. */
        status = nvshmemt_ib_common_query_endianness_conversion_size(&atomic_host_endian_size,
                                                                     device->context);
        if (status != 0) {
            ftable.close_device(device->context);
            device->context = NULL;
        }
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "nvshmemt_ib_common_query_endianness_conversion_size failed.\n");

        status = nvshmemt_ib_common_check_nic_ext_atomic_support(device->context);
        if (status) {
            ftable.close_device(device->context);
            device->context = NULL;
            NVSHMEMI_WARN_PRINT(
                "device %s does not support all necessary atomic operations. You may want to check "
                "the PCI_ATOMIC_MODE value in the NIC firmware. Skipping...\n",
                name);
            continue;
        }

        NVSHMEMT_IBDEVX_MAX_RD_ATOMIC = (device->device_attr).max_qp_rd_atom;
        INFO(ibdevx_state->log_level,
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
                                 &device->gid_info[p - 1].local_gid_index, ibdevx_state->log_level,
                                 ibdevx_state->options);
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
                    ibdevx_state->dev_ids[offset] = i;
                    ibdevx_state->port_ids[offset] = p;
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
    INFO(ibdevx_state->log_level, "End - Enumerating IB devices in the system");

    ibdevx_state->n_dev_ids = offset;
    INFO(ibdevx_state->log_level,
         "Begin - Ordered list of devices for assignment (after processing user provdied env vars "
         "(if any))  - ");
    for (int i = 0; i < ibdevx_state->n_dev_ids; i++) {
        INFO(ibdevx_state->log_level,
             "Ordered list of devices for assignment - idx=%d (of %d), device id=%d, port_num=%d",
             i, ibdevx_state->n_dev_ids, ibdevx_state->dev_ids[i], ibdevx_state->port_ids[i]);
    }
    INFO(ibdevx_state->log_level,
         "End - Ordered list of devices for assignment (after processing user provdied env vars "
         "(if any))");

    if (!ibdevx_state->n_dev_ids) {
        INFO(ibdevx_state->log_level, "no active IB device found, exiting");
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }

    transport->n_devices = ibdevx_state->n_dev_ids;
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

    // TODO: When we introduce a new version of the interface, add logic for handling them.

    transport->host_ops.can_reach_peer = nvshmemt_ibdevx_can_reach_peer;
    transport->host_ops.connect_endpoints = nvshmemt_ibdevx_connect_endpoints;
    transport->host_ops.get_mem_handle = nvshmemt_ibdevx_get_mem_handle;
    transport->host_ops.release_mem_handle = nvshmemt_ibdevx_release_mem_handle;
    transport->host_ops.rma = nvshmemt_ibdevx_rma;
    transport->host_ops.amo = nvshmemt_ibdevx_amo;
    transport->host_ops.fence = NULL;
    transport->host_ops.quiet = nvshmemt_ibdevx_quiet;
    transport->host_ops.finalize = nvshmemt_ibdevx_finalize;
    transport->host_ops.show_info = nvshmemt_ibdevx_show_info;
    transport->host_ops.progress = nvshmemt_ibdevx_progress;
    transport->host_ops.enforce_cst = nvshmemt_ibdevx_enforce_cst_at_target;

    transport->attr = NVSHMEM_TRANSPORT_ATTR_CONNECTED;
    transport->is_successfully_initialized = true;
    transport->atomics_complete_on_quiet = true;
    transport->max_op_len = 1ULL << 30;
    transport->atomic_host_endian_min_size = atomic_host_endian_size;
    transport->api_version = api_version < NVSHMEM_TRANSPORT_INTERFACE_VERSION
                                 ? api_version
                                 : NVSHMEM_TRANSPORT_INTERFACE_VERSION;

    *t = transport;

    ibdevx_state->table = table;
    ibdevx_state->dmabuf_support = false;

    if (ibdevx_state->options->IB_DISABLE_DMABUF) {
        ibdevx_state->dmabuf_support = false;
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
        ibdevx_state->dmabuf_support = true;
    }
check_nv_peer_mem:

    if (ibdevx_state->dmabuf_support == false) {
        if (nvshmemt_ib_common_nv_peer_mem_available() != NVSHMEMX_SUCCESS) {
            NVSHMEMI_ERROR_PRINT(
                "neither nv_peer_mem, or nvidia_peermem detected. Skipping transport.\n");
            status = NVSHMEMX_ERROR_INTERNAL;
            goto out;
        }
    }

out:

    if (status) {
        if (ibdevx_state) {
            if (ibdevx_state->devices) {
                free(ibdevx_state->devices);
            }
            if (ibdevx_state->dev_ids) {
                free(ibdevx_state->dev_ids);
            }
            if (ibdevx_state->port_ids) {
                free(ibdevx_state->port_ids);
            }
            if (ibdevx_state->options) {
                free(ibdevx_state->options);
            }
            free(ibdevx_state);
        }
        if (transport) {
            free(transport);
        }
    }
    return status;
}
