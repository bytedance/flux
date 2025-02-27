/*
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _NVSHMEMI_IBGDA_COMMON_H_
#define _NVSHMEMI_IBGDA_COMMON_H_

#define NVSHMEMI_IBGDA_QP_MANAGEMENT_PADDING 24
#define NVSHMEMI_IBGDA_STATE_PADDING 64

#define NVSHMEMI_IBGDA_SCALAR_INVALID -1
#define NVSHMEMI_IBGDA_USSCALAR_INVALID 0xFFFF
#define NVSHMEMI_IBGDA_USCALAR_INVALID 0xFFFFFFFF
#define NVSHMEMI_IBGDA_ULSCALAR_INVALID 0xFFFFFFFFFFFFFFFF

#include <infiniband/mlx5dv.h>  // for mlx5_wqe_av
#include <linux/types.h>        // for __be32
#if not defined __CUDACC_RTC__
#include <stddef.h>  // for size_t
#include <stdint.h>  // for uint64_t, uint32_t, uint16_t, uint8_t
#include <limits.h>

#define nvshmemi_init_ibgda_device_cq(cq)                            \
    do {                                                             \
        cq.version = (1 << 16) + sizeof(nvshmemi_ibgda_device_cq_t); \
        cq.cqe = NULL;                                               \
        cq.prod_idx = NULL;                                          \
        cq.cons_idx = NULL;                                          \
        cq.resv_head = NULL;                                         \
        cq.ready_head = NULL;                                        \
        cq.cqn = NVSHMEMI_IBGDA_USCALAR_INVALID;                     \
        cq.ncqes = NVSHMEMI_IBGDA_USCALAR_INVALID;                   \
        cq.qpn = NVSHMEMI_IBGDA_USCALAR_INVALID;                     \
        cq.qp_type = NVSHMEMI_IBGDA_DEVICE_QP_TYPE_MAX;              \
        cq.dbrec = NULL;                                             \
    } while (0);
#define nvshmemi_init_ibgda_device_qp_management(qp_man)                            \
    do {                                                                            \
        qp_man.version = (1 << 16) + sizeof(nvshmemi_ibgda_device_qp_management_t); \
        qp_man.post_send_lock = NVSHMEMI_IBGDA_SCALAR_INVALID;                      \
        qp_man.tx_wq.resv_head = NVSHMEMI_IBGDA_ULSCALAR_INVALID;                   \
        qp_man.tx_wq.ready_head = NVSHMEMI_IBGDA_ULSCALAR_INVALID;                  \
        qp_man.tx_wq.prod_idx = NVSHMEMI_IBGDA_ULSCALAR_INVALID;                    \
        qp_man.tx_wq.cons_idx = NVSHMEMI_IBGDA_ULSCALAR_INVALID;                    \
        qp_man.tx_wq.get_head = NVSHMEMI_IBGDA_ULSCALAR_INVALID;                    \
        qp_man.tx_wq.get_tail = NVSHMEMI_IBGDA_ULSCALAR_INVALID;                    \
        qp_man.ibuf.head = NVSHMEMI_IBGDA_ULSCALAR_INVALID;                         \
        qp_man.ibuf.tail = NVSHMEMI_IBGDA_ULSCALAR_INVALID;                         \
    } while (0);
#define nvshmemi_init_ibgda_device_qp(qp)                            \
    do {                                                             \
        qp.version = (1 << 16) + sizeof(nvshmemi_ibgda_device_qp_t); \
        qp.qp_type = NVSHMEMI_IBGDA_DEVICE_QP_TYPE_MAX;              \
        qp.qpn = NVSHMEMI_IBGDA_USCALAR_INVALID;                     \
        qp.dev_idx = NVSHMEMI_IBGDA_USCALAR_INVALID;                 \
        qp.ibuf.nslots = NVSHMEMI_IBGDA_USCALAR_INVALID;             \
        qp.ibuf.buf = NULL;                                          \
        qp.ibuf.lkey = NVSHMEMI_IBGDA_USCALAR_INVALID;               \
        qp.ibuf.rkey = NVSHMEMI_IBGDA_USCALAR_INVALID;               \
        qp.tx_wq.nwqes = NVSHMEMI_IBGDA_USSCALAR_INVALID;            \
        qp.tx_wq.wqe = NULL;                                         \
        qp.tx_wq.dbrec = NULL;                                       \
        qp.tx_wq.bf = NULL;                                          \
        qp.tx_wq.cq = NULL;                                          \
        nvshmemi_init_ibgda_device_qp_management(qp.mvars);          \
    } while (0);
#define nvshmemi_init_ibgda_device_local_only_memhandle(mhandle)                          \
    do {                                                                                  \
        mhandle.version = (1 << 16) + sizeof(nvshmemi_ibgda_device_local_only_mhandle_t); \
        mhandle.is_sysmem_scope = false;                                                  \
        mhandle.start = NVSHMEMI_IBGDA_ULSCALAR_INVALID;                                  \
        mhandle.end = NVSHMEMI_IBGDA_ULSCALAR_INVALID;                                    \
        mhandle.next = NULL;                                                              \
        for (int i = 0; i < NVSHMEMI_IBGDA_MAX_DEVICES_PER_PE; i++) {                     \
            mhandle.lkeys[i] = NVSHMEMI_IBGDA_USCALAR_INVALID;                            \
        }                                                                                 \
    } while (0);
#define nvshmemi_init_ibgda_device_state(state)                            \
    do {                                                                   \
        state.version = (1 << 16) + sizeof(nvshmemi_ibgda_device_state_t); \
        state.num_shared_dcis = NVSHMEMI_IBGDA_USCALAR_INVALID;            \
        state.num_exclusive_dcis = NVSHMEMI_IBGDA_USCALAR_INVALID;         \
        state.dci_map_type = NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_INVALID;    \
        state.ndcts_per_pe = NVSHMEMI_IBGDA_USCALAR_INVALID;               \
        state.num_qp_groups = NVSHMEMI_IBGDA_USCALAR_INVALID;              \
        state.num_dct_groups = NVSHMEMI_IBGDA_USCALAR_INVALID;             \
        state.num_rc_per_pe = NVSHMEMI_IBGDA_USCALAR_INVALID;              \
        state.rc_map_type = NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_INVALID;     \
        state.num_requests_in_batch = NVSHMEMI_IBGDA_USCALAR_INVALID;      \
        state.log2_cumem_granularity = NVSHMEMI_IBGDA_ULSCALAR_INVALID;    \
        state.num_devices_initialized = NVSHMEMI_IBGDA_SCALAR_INVALID;     \
        state.nic_buf_on_gpumem = false;                                   \
        state.support_half_av_seg = false;                                 \
        state.may_skip_cst = false;                                        \
        state.globalmem.qp_group_switches = NULL;                          \
        state.globalmem.cqs = NULL;                                        \
        state.globalmem.dcis = NULL;                                       \
        state.globalmem.rcs = NULL;                                        \
        state.globalmem.local_only_mhandle_head = NULL;                    \
        state.globalmem.dcts = NULL;                                       \
        state.globalmem.lkeys = NULL;                                      \
        state.globalmem.rkeys = NULL;                                      \
        state.extra = NULL;                                                \
    } while (0);

#else
#include <cuda/std/cstddef>
#include "cuda/std/cstdint"
#include <cuda/std/climits>
#endif

#define NVSHMEMI_IBGDA_MIN_QP_DEPTH 128
#define NVSHMEMI_IBGDA_MAX_QP_DEPTH 32768
#define NVSHMEMI_IBGDA_IBUF_SLOT_SIZE 256  // 32 threads * sizeof(uint64_t)

#define NVSHMEMI_IBGDA_MAX_CONST_LKEYS 64
#define NVSHMEMI_IBGDA_MAX_CONST_RKEYS 64
#define NVSHMEMI_IBGDA_MAX_CONST_DCTS 128

/* This is determined by the size of nvshmem_mem_handle_t*/
#define NVSHMEMI_IBGDA_MAX_DEVICES_PER_PE 15

typedef enum {
    NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI = 1,
    NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCT = 2,
    NVSHMEMI_IBGDA_DEVICE_QP_TYPE_RC = 3,
    NVSHMEMI_IBGDA_DEVICE_QP_TYPE_MAX = INT_MAX,
} nvshmemi_ibgda_device_qp_type_t;

typedef enum {
    NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_CTA = 0,
    NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_SM,
    NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_WARP,
    NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_DCT,
    NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_INVALID = INT_MAX
} nvshmemi_ibgda_device_qp_map_type_t;

typedef struct {
    int version;
    nvshmemi_ibgda_device_qp_type_t qp_type;
    __be32 *dbrec;
    void *cqe;
    uint64_t *prod_idx;
    uint64_t *cons_idx;
    uint64_t *resv_head;
    uint64_t *ready_head;
    uint32_t cqn;
    uint32_t ncqes;
    uint32_t qpn;
} nvshmemi_ibgda_device_cq_v1;
static_assert(sizeof(nvshmemi_ibgda_device_cq_v1) == 72, "ibgda_device_cq_v1 must be 72 bytes.");

typedef nvshmemi_ibgda_device_cq_v1 nvshmemi_ibgda_device_cq_t;

// Variables for queue management.
// They are always in global memory.
typedef struct {
    int version;
    int post_send_lock;
    struct {
        // All indexes are in wqebb unit
        uint64_t resv_head;   // last reserved wqe idx + 1
        uint64_t ready_head;  // last ready wqe idx + 1
        uint64_t prod_idx;    // posted wqe idx + 1 (producer index + 1)
        uint64_t cons_idx;    // polled wqe idx + 1 (consumer index + 1)
        uint64_t get_head;    // last wqe idx + 1 with a "fetch" operation (g, get, amo_fetch)
        uint64_t get_tail;    // last wqe idx + 1 polled with cst; get_tail > get_head is possible
    } tx_wq;
    struct {
        uint64_t head;
        uint64_t tail;
    } ibuf;
    char padding[NVSHMEMI_IBGDA_QP_MANAGEMENT_PADDING];
} __attribute__((__aligned__(8))) nvshmemi_ibgda_device_qp_management_v1;
static_assert(sizeof(nvshmemi_ibgda_device_qp_management_v1) == 96,
              "ibgda_device_qp_management_v1 must be 96 bytes.");

typedef nvshmemi_ibgda_device_qp_management_v1 nvshmemi_ibgda_device_qp_management_t;

typedef struct nvshmemi_ibgda_device_qp {
    int version;
    nvshmemi_ibgda_device_qp_type_t qp_type;
    uint32_t qpn;
    uint32_t dev_idx;
    struct {
        uint32_t nslots;  // num slots for fetch; always a power of 2
        void *buf;        // first NVSHMEMI_IBGDA_IBUF_SLOT_SIZE is for non-fetch
        __be32 lkey;
        __be32 rkey;
    } ibuf;  // Internal buffer
    struct {
        uint16_t nwqes;  // num wqes; some wqes may consume n wqebbs
        void *wqe;
        __be32 *dbrec;
        void *bf;
        nvshmemi_ibgda_device_cq_t *cq;
        // May point to mvars.prod_idx or internal prod_idx
        uint64_t *prod_idx;
    } tx_wq;
    nvshmemi_ibgda_device_qp_management_v1 mvars;  // management variables
} nvshmemi_ibgda_device_qp_v1;
static_assert(sizeof(nvshmemi_ibgda_device_qp_v1) == 184, "ibgda_device_qp_v1 must be 184 bytes.");

typedef nvshmemi_ibgda_device_qp_v1 nvshmemi_ibgda_device_qp_t;

typedef struct mlx5_wqe_av nvshmemi_ibgda_device_dct_t;

typedef struct nvshmemi_ibgda_device_local_only_mhandle {
    int version;
    bool is_sysmem_scope;
    uint64_t start;
    uint64_t end;
    struct nvshmemi_ibgda_device_local_only_mhandle *next;
    __be32 lkeys[NVSHMEMI_IBGDA_MAX_DEVICES_PER_PE];
} nvshmemi_ibgda_device_local_only_mhandle_v1;
static_assert(sizeof(nvshmemi_ibgda_device_local_only_mhandle_v1) == 96,
              "ibgda_device_local_only_mhandle_v1 must be 96 bytes.");

typedef nvshmemi_ibgda_device_local_only_mhandle_v1 nvshmemi_ibgda_device_local_only_mhandle_t;

// This is a stable structure.
typedef struct {
    __be32 key;
    uint64_t next_addr;  // end of this address range + 1
} nvshmemi_ibgda_device_key_t;

typedef struct {
    int version;
    uint32_t num_shared_dcis;
    uint32_t num_exclusive_dcis;
    nvshmemi_ibgda_device_qp_map_type_t dci_map_type;
    uint32_t ndcts_per_pe;
    uint32_t num_qp_groups;
    uint32_t num_dct_groups;
    uint32_t num_rc_per_pe;
    nvshmemi_ibgda_device_qp_map_type_t rc_map_type;
    uint32_t num_requests_in_batch; /* always a power of 2 */
    size_t log2_cumem_granularity;
    int num_devices_initialized;
    bool nic_buf_on_gpumem;
    bool support_half_av_seg;
    bool may_skip_cst;
    bool use_async_postsend;

    struct {
        // lkeys[idx] gives the lkey of chunk idx.
        nvshmemi_ibgda_device_key_t lkeys[NVSHMEMI_IBGDA_MAX_CONST_LKEYS];

        // rkeys[idx * npes + pe] gives rkey of chunck idx targeting peer pe.
        nvshmemi_ibgda_device_key_t rkeys[NVSHMEMI_IBGDA_MAX_CONST_RKEYS];

        nvshmemi_ibgda_device_dct_t dcts[NVSHMEMI_IBGDA_MAX_CONST_DCTS];
    } constmem;

    struct {
        uint8_t *qp_group_switches;
        nvshmemi_ibgda_device_cq_t *cqs;  // For both dcis and rcs. CQs for DCIs come first.
        nvshmemi_ibgda_device_qp_t *dcis;
        nvshmemi_ibgda_device_qp_t *rcs;
        nvshmemi_ibgda_device_local_only_mhandle *local_only_mhandle_head;

        // For dcts that cannot be contained in constmem.lkeys.
        // dcts[idx - NVSHMEMI_IBGDA_MAX_CONST_DCTS] gives the dct of idx.
        nvshmemi_ibgda_device_dct_t *dcts;

        // For lkeys that cannot be contained in constmem.lkeys.
        // lkeys[idx - NVSHMEMI_IBGDA_MAX_CONST_LKEYS] gives the lkey of chunk idx.
        nvshmemi_ibgda_device_key_t *lkeys;

        // For rkeys that cannot be contained in constmem.rkeys.
        // rkeys[(idx * npes + pe) - NVSHMEMI_IBGDA_MAX_CONST_RKEYS] gives rkey of chunck idx
        // targeting peer pe.
        nvshmemi_ibgda_device_key_t *rkeys;
    } globalmem;

    void *extra;
    uint8_t reserved[NVSHMEMI_IBGDA_STATE_PADDING];
} nvshmemi_ibgda_device_state_v1;
static_assert(sizeof(nvshmemi_ibgda_device_state_v1) == 8384,
              "ibgda_device_state_v1 must be 8384 bytes.");

typedef nvshmemi_ibgda_device_state_v1 nvshmemi_ibgda_device_state_t;

#if defined(__CUDACC_RDC__)
#define EXTERN_CONSTANT extern __constant__
EXTERN_CONSTANT nvshmemi_ibgda_device_state_t nvshmemi_ibgda_device_state_d;
#undef EXTERN_CONSTANT
#endif

#endif /* _NVSHMEMI_IBGDA_COMMON_H_ */
