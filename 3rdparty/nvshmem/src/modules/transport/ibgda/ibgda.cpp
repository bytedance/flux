/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include <assert.h>                                      // for assert
#include <cuda.h>                                        // for CUDA_SUCCESS, CUdevice, CUd...
#include <cuda_runtime.h>                                // for cudaFree, cudaMalloc, cudaM...
#include <driver_types.h>                                // for cudaSuccess, cudaMemcpyHost...
#include <endian.h>                                      // for htobe32, htobe64
#include <errno.h>                                       // for ENOMEM
#include <linux/types.h>                                 // for __be32
#include <math.h>                                        // for ceil, log2
#include <stddef.h>                                      // for NULL, size_t, offsetof
#include <stdint.h>                                      // for uint8_t, uint64_t, uint32_t
#include <stdio.h>                                       // for fprintf, stderr, printf
#include <stdlib.h>                                      // for free, calloc, malloc, posix...
#include <unistd.h>                                      // for _SC_PAGESIZE
#include <string.h>                                      // for memset, memcpy, strcmp, strstr
#include <sys/types.h>                                   // for off_t
#include <algorithm>                                     // for for_each, remove_if, max
#include <cctype>                                        // for tolower, isspace
#include <string>                                        // for basic_string, string, opera...
#include <vector>                                        // for vector
#include "device_host_transport/nvshmem_common_ibgda.h"  // for nvshmemi_ibgda_device_state_t
#include "internal/host_transport/cudawrap.h"            // for CUPFN, nvshmemi_cuda_fn_table
#include "bootstrap_host_transport/env_defs_internal.h"  // for nvshmemi_options_s, nvshmem...
#include "non_abi/nvshmemx_error.h"                      // for NVSHMEMX_ERROR_INTERNAL
#include "non_abi/nvshmem_build_options.h"               // IWYU pragma: keep
#include "infiniband/mlx5dv.h"                           // for DEVX_SET, DEVX_ST_SZ_BYTES
#include "infiniband/verbs.h"                            // for ibv_ah_attr, ibv_port_attr
#include "mlx5_ifc.h"                                    // for mlx5_ifc_qpc_bits, mlx5_ifc...
#include "mlx5_prm.h"                                    // for mlx5_ifc_cqc_bits, mlx5_ifc...
#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"  // for bootstrap_handle_t
#include "internal/host_transport/nvshmemi_transport_defines.h"  // for nvshmem_mem_handle_t, NVSHM...
#include "internal/host_transport/transport.h"  // for nvshmem_transport, nvshmem_...
#include "transport_common.h"                   // for nvshmemt_ibv_function_table
#include "transport_ib_common.h"                // for nvshmemt_ib_common_mem_handle
#include "transport_mlx5_common.h"              // for nvshmemt_ib_common_check_ni...
#ifdef NVSHMEM_USE_GDRCOPY
#include "transport_gdr_common.h"
#endif

#define CUDA_RUNTIME_ERROR_STRING(result)                                         \
    do {                                                                          \
        if (unlikely(cudaSuccess != result)) {                                    \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
        }                                                                         \
    } while (0)

#define NVSHMEMI_IBGDA_CQE_SIZE 64
#define NVSHMEMI_IBGDA_MAX_INLINE_SIZE (8 * 32)

#define MAX_NUM_HCAS 16
#define MAX_NUM_PORTS 4
#define MAX_NUM_PES_PER_NODE 32

#define IBGDA_DC_ACCESS_KEY 0x5623CEAF

#define IBGDA_MLX5_QPC_ATOMIC_MODE_UP_TO_64BIT 0x3
#define IBGDA_DBRSIZE 8
#define IBGDA_SRQ_TYPE_VALUE 0x1

#define IBGDA_LOG_MAX_MSG_SIZE 30  // 30 is max allowed on IB QPs
#define IBGDA_MIN_RNR_NAK 12

#define IBGDA_GRH_HOP_LIMIT 255

// First slot is reserved for non-fetch operations.
#define IBGDA_IBUF_RESERVED_SLOTS 1

#define IBGDA_GPAGE_BITS 16
#define IBGDA_GPAGE_SIZE (1ULL << IBGDA_GPAGE_BITS)
#define IBGDA_GPAGE_OFF (IBGDA_GPAGE_SIZE - 1)
#define IBGDA_GPAGE_MASK (~(IBGDA_GPAGE_OFF))

#define IBGDA_ACCESS_ONCE(x) (*(volatile typeof(x) *)&(x))
#define IBGDA_READ_ONCE(x) IBGDA_ACCESS_ONCE(x)
#define IBGDA_WRITE_ONCE(x, v) (IBGDA_ACCESS_ONCE(x) = (v))

#define IBGDA_MIN(x, y) ((x) < (y) ? (x) : (y))
#define IBGDA_MAX(x, y) ((x) > (y) ? (x) : (y))

#define IBGDA_ROUND_UP(V, SIZE) (((V) + (SIZE)-1) / (SIZE) * (SIZE))

#define IBGDA_ROUND_UP_POW2(_n)                 \
    ({                                          \
        typeof(_n) pow2 = 0;                    \
        assert((_n) >= 1);                      \
        for (pow2 = 1; pow2 < (_n); pow2 <<= 1) \
            ;                                   \
        pow2;                                   \
    })

#define IBGDA_ROUND_UP_POW2_OR_0(_n) (((_n) == 0) ? 0 : IBGDA_ROUND_UP_POW2(_n))

#define IBGDA_ROUND_DOWN_POW2_OR_0(_n)                  \
    ({                                                  \
        typeof(_n) pow2 = IBGDA_ROUND_UP_POW2_OR_0(_n); \
        (((_n) < pow2) ? pow2 / 2 : pow2);              \
    })

template <typename T>
inline T IBGDA_ILOG2(T _n) {
    return (T)ceil(log2((double)_n));
}

#define IBGDA_ILOG2_OR0(_n) (((_n) == 0) ? 0 : IBGDA_ILOG2(_n))

enum { IBGDA_MLX5_QPC_ST_RC = 0x0, IBGDA_MLX5_QPC_ST_DCI = 0x5 };

enum {
    IBGDA_MLX5_UMEM_VALID_DISABLE = 0x0,
    IBGDA_MLX5_UMEM_VALID_ENABLE = 0x1,
};

enum { IBGDA_MLX5_NC_UAR_SIZE = 8 };

typedef enum {
    IBGDA_NIC_MAPPING_MEMTYPE_REQUEST_AUTO = 0,
    IBGDA_NIC_MAPPING_MEMTYPE_REQUEST_GPUMEM,
    IBGDA_NIC_MAPPING_MEMTYPE_REQUEST_HOSTMEM,
} ibgda_nic_mapping_memtype_reqeust_t;

typedef enum {
    IBGDA_MEM_TYPE_HOST = 0,
    IBGDA_MEM_TYPE_GPU = 1,
    IBGDA_MEM_TYPE_NIC = 2,
} ibgda_mem_type_t;

typedef enum {
    IBGDA_NIC_HANDLER_AUTO = 0,
    IBGDA_NIC_HANDLER_GPU,
    IBGDA_NIC_HANDLER_CPU,
} ibgda_nic_handler_t;

struct ibgda_mem_object {
    ibgda_mem_type_t mem_type;
    struct {
        void *cpu_ptr;
        void *gpu_ptr;
        size_t size;
    } base;
    struct {
        void *cpu_ptr;
        void *gpu_ptr;
        size_t size;
    } aligned;
    union {
        struct mlx5dv_devx_umem *umem;
        struct mlx5dv_devx_uar *uar;
    };
    bool has_cpu_mapping : 1;
    bool has_gpu_mapping : 1;
    bool has_nic_mapping : 1;
#ifdef NVSHMEM_USE_GDRCOPY
    gdr_mh_t mh;
#endif
};

struct ibgda_cq {
    struct mlx5dv_devx_obj *devx_cq;
    uint32_t cqn;
    uint32_t num_cqe;
    struct ibgda_mem_object *cq_mobject;
    struct ibgda_mem_object *dbr_mobject;
    struct mlx5dv_devx_uar *uar;
    off_t cq_offset;
    off_t dbr_offset;
};

struct ibgda_ep {
    nvshmemi_ibgda_device_qp_type_t qp_type;

    union {
        struct mlx5dv_devx_obj *devx_qp;
        struct ibv_qp *ib_qp;
    };
    uint32_t qpn;
    int portid;

    size_t sq_cnt;
    off_t sq_buf_offset;
    size_t rq_cnt;
    off_t rq_buf_offset;

    struct ibgda_mem_object *wq_mobject;
    struct ibgda_mem_object *dbr_mobject;
    struct ibgda_mem_object *uar_mobject;

    off_t wq_offset;
    off_t dbr_offset;

    struct ibgda_cq *send_cq;
    struct ibv_ah *ah;

    uint32_t user_index;
};

struct ibgda_mem_handle {
    struct nvshmemt_ib_common_mem_handle dev_mem_handles[NVSHMEMI_IBGDA_MAX_DEVICES_PER_PE];
    int num_devs;
};

struct ibgda_dct_handle {
    nvshmemi_ibgda_device_dct_t dev_dct;
    bool support_half_av_seg;
};

struct ibgda_rc_handle {
    uint32_t qpn;
    uint16_t lid;
    // RoCE
    uint64_t spn;
    uint64_t iid;
};

struct ibgda_internal_buffer {
    struct ibgda_mem_object *mem_object;
    struct nvshmemt_ib_common_mem_handle *mem_handle;
};

struct ibgda_device {
    struct ibv_device *dev;
    struct ibv_pd *pd; /* protection domain */
    struct ibv_context *context;
    struct ibv_device_attr device_attr;
    struct ibv_port_attr port_attr[MAX_NUM_PORTS];
    struct nvshmemt_ib_gid_info gid_info[MAX_NUM_PORTS];
    struct {
        int num_eps;
        struct ibgda_ep **eps;
        struct ibgda_dct_handle *dct_handles;
        struct ibv_pd *pd; /* parent domain */
        struct ibv_srq *srq;
        struct ibv_cq *send_cq;
        struct ibv_cq *recv_cq;
        struct ibv_ah *ah;
        struct mlx5dv_ah dah;
        struct ibv_ah_attr ah_attr;
    } dct;
    struct {
        struct ibv_srq *srq;
        struct ibv_cq *recv_cq;
        struct ibgda_mem_object *wq_mobject;
        struct ibgda_mem_object *dbr_mobject;
        struct ibgda_internal_buffer internal_buf;
        size_t wq_buf_size_per_qp;
        off_t cur_wq_off;
        off_t cur_dbr_off;
        int pdn;
        int srqn;
        int rcqn;
        struct ibgda_mem_object *prod_idx_mobject;
        uint64_t *prod_idx_cache;
        uint64_t *prod_idx_snapshot;
    } qp_shared_object;  // For DCI and RC
    struct {
        size_t cq_buf_size_per_cq;
        struct ibgda_mem_object *cq_mobject;
        struct ibgda_mem_object *dbr_mobject;
        off_t cur_cq_off;
        off_t cur_dbr_off;
    } cq_shared_object;
    struct {
        struct ibgda_ep **eps;
        int num_eps;
        int num_shared_eps;
        nvshmemi_ibgda_device_qp_map_type_t map_by;
    } dci;
    struct {
        struct ibgda_ep **eps;
        struct ibgda_rc_handle *peer_ep_handles;
        int num_eps_per_pe;
        nvshmemi_ibgda_device_qp_map_type_t map_by;
    } rc;
    bool support_nic_buf_on_gpumem;
    bool support_nic_buf_on_hostmem;
    bool support_half_av_seg;
    bool may_skip_cst;
    ibgda_nic_handler_t nic_handler;
};

typedef struct {
    struct nvshmemi_options_s *options;
    void *devices;
    int *dev_ids;
    int *port_ids;
    int *selected_dev_ids;
    int n_dev_ids;
    int n_devs_selected;
    int log_level;
    bool cuda_support_dmabuf;
    bool dmabuf_support_for_data_buffers;
    bool dmabuf_support_for_control_buffers;
    cudaStream_t my_stream;
} nvshmemt_ibgda_state_t;

struct ibgda_device_local_only_mhandle_cache {
    nvshmemi_ibgda_device_local_only_mhandle_t mhandle;
    void *
        dev_ptr;  // Ptr to GPU buffer that contains a copy of this mhandle. CPU cannot dereference.
};

// CPU cannot dereference next
static std::vector<struct ibgda_device_local_only_mhandle_cache> ibgda_device_local_only_mhandles;

static std::vector<nvshmemi_ibgda_device_key_t> ibgda_device_lkeys;
static std::vector<nvshmemi_ibgda_device_key_t> ibgda_device_rkeys;

// Ptr to GPU buffer. CPU cannot dereference.
static void *ibgda_device_lkeys_d = 0;
static void *ibgda_device_rkeys_d = 0;

/* transport constants */
static int ibgda_qp_depth = 0;
static int ibgda_srq_depth;
static int ibgda_num_requests_in_batch;
static int ibgda_num_fetch_slots_per_dci;
static int ibgda_num_fetch_slots_per_rc;

/* ibv state */
static struct nvshmemt_ibv_function_table ftable;
static void *ibv_handle;

/* CUDA function table */
static struct nvshmemi_cuda_fn_table *ibgda_cuda_syms;

#ifdef NVSHMEM_USE_GDRCOPY
static gdr_t gdr_desc;
static struct gdrcopy_function_table gdrcopy_ftable;
static void *gdrcopy_handle = NULL;
#endif
static bool use_gdrcopy = 0;

static ibgda_mem_type_t ibgda_nic_buf_location;
static ibgda_nic_handler_t ibgda_nic_handler;

static int ibgda_parse_qp_map_by(nvshmemi_ibgda_device_qp_map_type_t *out_map_by, const char *str) {
    int status = 0;
    nvshmemi_ibgda_device_qp_map_type_t map_by;
    std::string req = str;

    // Trim whitespace
    req.erase(std::remove_if(req.begin(), req.end(), ::isspace), req.end());

    // To lower case
    std::for_each(req.begin(), req.end(), [](decltype(*req.begin()) &c) { c = ::tolower(c); });

    if (req == "cta") {
        map_by = NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_CTA;
    } else if (req == "sm") {
        map_by = NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_SM;
    } else if (req == "warp") {
        map_by = NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_WARP;
    } else if (req == "dct") {
        map_by = NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_DCT;
    } else {
        status = NVSHMEMX_ERROR_INVALID_VALUE;
    }

    if (status == 0) {
        *out_map_by = map_by;
    }

    return status;
}

static int ibgda_parse_nic_handler_request(ibgda_nic_handler_t *out_loc, const char *str) {
    int status = 0;
    ibgda_nic_handler_t loc;
    std::string req = str;

    // Trim whitespace
    req.erase(std::remove_if(req.begin(), req.end(), ::isspace), req.end());

    // To lower case
    std::for_each(req.begin(), req.end(), [](decltype(*req.begin()) &c) { c = ::tolower(c); });

    if (req == "auto") {
        loc = IBGDA_NIC_HANDLER_AUTO;
    } else if (req == "gpu") {
        loc = IBGDA_NIC_HANDLER_GPU;
    } else if (req == "cpu") {
        loc = IBGDA_NIC_HANDLER_CPU;
    } else {
        status = NVSHMEMX_ERROR_INVALID_VALUE;
    }

    if (status == 0) {
        *out_loc = loc;
    }

    return status;
}

static size_t ibgda_get_host_page_size() {
    static size_t host_page_size = 0;
    if (!host_page_size) host_page_size = sysconf(_SC_PAGESIZE);
    return host_page_size;
}

int nvshmemt_ibgda_progress(nvshmem_transport_t t) {
    nvshmemt_ibgda_state_t *ibgda_state = (nvshmemt_ibgda_state_t *)t->state;
    int n_devs_selected = ibgda_state->n_devs_selected;
    int n_pes = t->n_pes;
    struct mlx5_wqe_ctrl_seg ctrl_seg;
    for (int j = 0; j < n_devs_selected; j++) {
        struct ibgda_device *device;
        int dev_idx;
        int num_prod_idx_slots;
        uint64_t *prod_idx_cache;
        uint64_t *prod_idx_snapshot;
        uint64_t *prod_idx_array;

        dev_idx = ibgda_state->selected_dev_ids[j];
        device = (struct ibgda_device *)ibgda_state->devices + dev_idx;
        num_prod_idx_slots = device->dci.num_eps + device->rc.num_eps_per_pe * n_pes;
        prod_idx_cache = device->qp_shared_object.prod_idx_cache;
        prod_idx_snapshot = device->qp_shared_object.prod_idx_snapshot;
        prod_idx_array = (uint64_t *)device->qp_shared_object.prod_idx_mobject->aligned.cpu_ptr;

        gdrcopy_ftable.copy_from_mapping(device->qp_shared_object.prod_idx_mobject->mh,
                                         prod_idx_snapshot, prod_idx_array,
                                         sizeof(uint64_t) * num_prod_idx_slots);

        for (int i = 0; i < num_prod_idx_slots; ++i) {
            uint64_t prod_idx = prod_idx_snapshot[i];
            struct ibgda_ep *ep;
            __be32 *dbrec;
            __be64 *bf;
            if (prod_idx_cache[i] < prod_idx) {
                if (i < device->dci.num_eps)
                    ep = device->dci.eps[i];
                else
                    ep = device->rc.eps[i - device->dci.num_eps];

                dbrec = (__be32 *)((uintptr_t)ep->dbr_mobject->aligned.cpu_ptr + ep->dbr_offset +
                                   sizeof(__be32));
                bf = (__be64 *)ep->uar_mobject->aligned.cpu_ptr;

                memset((void *)&ctrl_seg, 0, sizeof(ctrl_seg));
                ctrl_seg.qpn_ds = htobe32(ep->qpn << 8);
                ctrl_seg.opmod_idx_opcode = htobe32(prod_idx << 8);

                IBGDA_WRITE_ONCE(*dbrec, htobe32(prod_idx & 0xffff));
                STORE_BARRIER();
                IBGDA_WRITE_ONCE(*bf, *((__be64 *)&ctrl_seg));

                prod_idx_cache[i] = prod_idx;
            }
        }
    }
    return 0;
}

int nvshmemt_ibgda_show_info(struct nvshmem_transport *transport, int style) {
    NVSHMEMI_ERROR_PRINT("ibgda show info not implemented");
    return 0;
}

static int get_pci_path(int dev, char **pci_path, nvshmem_transport_t t) {
    int status = NVSHMEMX_SUCCESS;

    struct nvshmem_transport *transport = (struct nvshmem_transport *)t;
    nvshmemt_ibgda_state_t *ibgda_state = (nvshmemt_ibgda_state_t *)transport->state;
    int dev_id = ibgda_state->dev_ids[dev];
    const char *ib_name =
        (const char *)((struct ibgda_device *)ibgda_state->devices)[dev_id].dev->name;

    status = nvshmemt_ib_iface_get_mlx_path(ib_name, pci_path);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "nvshmemt_ib_iface_get_mlx_path failed \n");

out:
    return status;
}

int nvshmemt_ibgda_can_reach_peer(int *access, struct nvshmem_transport_pe_info *peer_info,
                                  nvshmem_transport_t t) {
    int status = 0;

    *access = NVSHMEM_TRANSPORT_CAP_GPU_WRITE | NVSHMEM_TRANSPORT_CAP_GPU_READ |
              NVSHMEM_TRANSPORT_CAP_GPU_ATOMICS;

    return status;
}

int nvshmemt_ibgda_get_mem_handle(nvshmem_mem_handle_t *mem_handle,
                                  nvshmem_mem_handle_t *mem_handle_in, void *buf, size_t length,
                                  nvshmem_transport_t t, bool local_only) {
    int status = 0;
    struct nvshmem_transport *transport = (struct nvshmem_transport *)t;
    nvshmemt_ibgda_state_t *ibgda_state = (nvshmemt_ibgda_state_t *)transport->state;

    __be32 device_lkey;
    struct ibgda_mem_handle *handle;

    nvshmemi_ibgda_device_local_only_mhandle_t *device_mhandle_d = NULL;
    bool did_emplace = false;

    nvshmemi_ibgda_device_state_t *ibgda_device_state;
    ibgda_device_state = (nvshmemi_ibgda_device_state_t *)transport->type_specific_shared_state;
    assert(ibgda_device_state != NULL);
    int n_devs_selected = ibgda_state->n_devs_selected;

    memset((void *)mem_handle, 0, sizeof(*mem_handle));
    handle = (struct ibgda_mem_handle *)mem_handle;
    handle->num_devs = n_devs_selected;

    for (int i = 0; i < n_devs_selected; ++i) {
        struct ibgda_device *device =
            ((struct ibgda_device *)ibgda_state->devices + ibgda_state->selected_dev_ids[i]);
        nvshmem_mem_handle_t *dev_handle = (nvshmem_mem_handle_t *)&handle->dev_mem_handles[i];

        status = nvshmemt_ib_common_reg_mem_handle(
            &ftable, device->pd, dev_handle, buf, length, local_only,
            ibgda_state->dmabuf_support_for_data_buffers, ibgda_cuda_syms, ibgda_state->log_level,
            ibgda_state->options->IB_ENABLE_RELAXED_ORDERING);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Unable to register memory handle.\n");
    }

    if (local_only) {
        struct ibgda_device_local_only_mhandle_cache device_mhandle_cache;
        nvshmemi_ibgda_device_local_only_mhandle_t *device_mhandle_h =
            &device_mhandle_cache.mhandle;
        nvshmemi_init_ibgda_device_local_only_memhandle((*device_mhandle_h));

        void *mhandle_gpu_ptr;

        cudaPointerAttributes buf_attributes;

        status = cudaPointerGetAttributes(&buf_attributes, buf);
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                              "cudaPointerGetAttributes failed.\n");

        status = cudaMalloc((void **)&device_mhandle_d, sizeof(*device_mhandle_d));
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                              "cudaMalloc failed.\n");

        device_mhandle_h->start = (uint64_t)buf;
        device_mhandle_h->end = (uint64_t)buf + length - 1;
        device_mhandle_h->is_sysmem_scope = (buf_attributes.type != cudaMemoryTypeDevice);
        device_mhandle_h->next = NULL;
        for (int i = 0; i < n_devs_selected; ++i) {
            device_lkey = htobe32(handle->dev_mem_handles[i].lkey);
            device_mhandle_h->lkeys[i] = device_lkey;
        }

        status = cudaMemcpyAsync((void *)device_mhandle_d, (const void *)device_mhandle_h,
                                 sizeof(*device_mhandle_d), cudaMemcpyHostToDevice,
                                 ibgda_state->my_stream);
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                              "Copying device_mhandle to GPU memory failed.\n");

        device_mhandle_cache.dev_ptr = device_mhandle_d;

        if (ibgda_device_local_only_mhandles.empty()) {
            ibgda_device_state->globalmem.local_only_mhandle_head = device_mhandle_d;
        } else {
            struct ibgda_device_local_only_mhandle_cache *last_mhandle_cache =
                &ibgda_device_local_only_mhandles.back();
            mhandle_gpu_ptr = (void *)((uintptr_t)last_mhandle_cache->dev_ptr +
                                       offsetof(nvshmemi_ibgda_device_local_only_mhandle_t, next));
            last_mhandle_cache->mhandle.next = device_mhandle_d;
            status = cudaMemcpyAsync(mhandle_gpu_ptr, (const void *)&device_mhandle_d,
                                     sizeof(device_mhandle_d), cudaMemcpyHostToDevice,
                                     ibgda_state->my_stream);
            NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                                  "Setting local_only_mhandle in GPU memory failed.\n");
        }

        ibgda_device_local_only_mhandles.emplace_back(device_mhandle_cache);
        did_emplace = true;
    } else {
        size_t num_lkeys;
        size_t num_elements;

        // length must be divisible by cumem_granularity, which is a power of 2.
        assert((length & ((1ULL << transport->log2_cumem_granularity) - 1)) == 0);

        num_elements = length >> transport->log2_cumem_granularity;
        while (num_elements > 0) {
            for (int i = 0; i < n_devs_selected; i++) {
                device_lkey = htobe32(handle->dev_mem_handles[i].lkey);
                nvshmemi_ibgda_device_key_t dev_key;
                dev_key.key = device_lkey;
                dev_key.next_addr = (uint64_t)buf + length;
                ibgda_device_lkeys.emplace_back(dev_key);
            }
            --num_elements;
        }

        did_emplace = true;

        if (ibgda_device_lkeys_d) {
            status = cudaFree(ibgda_device_lkeys_d);
            NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                                  "cudaFree failed.\n");
            ibgda_device_lkeys_d = 0;
        }

        num_lkeys = ibgda_device_lkeys.size();

        // Put lkeys in constant memory first for cache optimization
        memcpy(ibgda_device_state->constmem.lkeys, ibgda_device_lkeys.data(),
               IBGDA_MIN(num_lkeys, NVSHMEMI_IBGDA_MAX_CONST_LKEYS) *
                   sizeof(nvshmemi_ibgda_device_key_t));

        // If we have overflow, put the rest in global memory
        if (num_lkeys > NVSHMEMI_IBGDA_MAX_CONST_LKEYS) {
            size_t lkeys_array_size =
                sizeof(nvshmemi_ibgda_device_key_t) * (num_lkeys - NVSHMEMI_IBGDA_MAX_CONST_LKEYS);

            nvshmemi_ibgda_device_key_t *data_ptr =
                &ibgda_device_lkeys.data()[NVSHMEMI_IBGDA_MAX_CONST_LKEYS];

            status = cudaMalloc(&ibgda_device_lkeys_d, lkeys_array_size);
            NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                  "cudaMalloc failed.\n");

            status = cudaMemcpyAsync(ibgda_device_lkeys_d, (const void *)data_ptr, lkeys_array_size,
                                     cudaMemcpyHostToDevice, ibgda_state->my_stream);
            NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                                  "Copying lkeys to GPU memory failed.\n");
        }
        ibgda_device_state->globalmem.lkeys = (nvshmemi_ibgda_device_key_t *)ibgda_device_lkeys_d;
    }

    status = cudaStreamSynchronize(ibgda_state->my_stream);
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                          "stream synchronize failed.\n");

out:
    if (status) {
        if (device_mhandle_d) cudaFree(device_mhandle_d);
        if (did_emplace) {
            if (local_only) {
                // Recoverable
                ibgda_device_local_only_mhandles.pop_back();
            } else {
                // Unrecoverable
                ibgda_device_lkeys.clear();
            }
        }

        for (int i = 0; i < n_devs_selected; ++i) {
            nvshmemt_ib_common_release_mem_handle(
                &ftable, (nvshmem_mem_handle_t *)&handle->dev_mem_handles[i],
                ibgda_state->log_level);
        }
    }
    return status;
}

static int ibgda_mobject_nic_map(struct ibgda_mem_object *mobject, struct ibv_context *context,
                                 uint32_t access, bool use_dmabuf = false) {
    int status = 0;
    void *addr;
    struct mlx5dv_devx_umem *umem = NULL;

    assert(mobject);
    assert(!mobject->has_nic_mapping);
    assert(context);

    if (mobject->mem_type == IBGDA_MEM_TYPE_GPU) {
        addr = (void *)mobject->aligned.gpu_ptr;
    } else if (mobject->mem_type == IBGDA_MEM_TYPE_HOST) {
        addr = mobject->aligned.cpu_ptr;
    } else {
        status = NVSHMEMX_ERROR_INTERNAL;
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "invalid mem_type specified.\n");
        assert(0);
    }

    if (use_dmabuf && mobject->mem_type == IBGDA_MEM_TYPE_GPU) {
#ifdef HAVE_MLX5DV_UMEM_MASK_DMABUF
        int fd;
        struct mlx5dv_devx_umem_in umem_in = {
            0,
        };
        const size_t host_page_size = ibgda_get_host_page_size();
        size_t dmabuf_size = IBGDA_ROUND_UP(mobject->aligned.size, host_page_size);
        CUCHECKGOTO(ibgda_cuda_syms,
                    cuMemGetHandleForAddressRange(&fd, (CUdeviceptr)addr, dmabuf_size,
                                                  CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0),
                    status, out);
        umem_in.addr = 0;
        umem_in.size = mobject->aligned.size;
        umem_in.access = access;
        umem_in.pgsz_bitmap = UINT64_MAX & ~(host_page_size - 1);
        umem_in.comp_mask = MLX5DV_UMEM_MASK_DMABUF;
        umem_in.dmabuf_fd = fd;
        umem = mlx5dv_devx_umem_reg_ex(context, &umem_in);
        close(fd);
#else
        status = NVSHMEMX_ERROR_NOT_SUPPORTED;
        goto out;
#endif
    } else {
        umem = mlx5dv_devx_umem_reg(context, addr, mobject->aligned.size, access);
    }
    if (!umem) {
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }

    mobject->umem = umem;
    mobject->has_nic_mapping = true;

out:
    return status;
}

static void ibgda_mobject_nic_unmap(struct ibgda_mem_object *mobject) {
    int status = 0;

    assert(mobject);
    assert(mobject->has_nic_mapping);
    assert(mobject->mem_type != IBGDA_MEM_TYPE_NIC);
    assert(mobject->umem);

    status = mlx5dv_devx_umem_dereg(mobject->umem);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "mlx5dv_devx_umem_dereg failed.\n");

    mobject->has_nic_mapping = false;
    mobject->umem = NULL;

out:
    return;
}

static int ibgda_gpu_mem_alloc(struct ibgda_mem_object **pmobject, size_t size, size_t alignment,
                               bool host_mapping) {
    int status = 0;

    int attr_val;

    void *ptr = 0;
    void *aligned_ptr;
    size_t bufsize = size;

    void *cpu_ptr_base = NULL;
    void *cpu_ptr = NULL;

    struct ibgda_mem_object *mobject =
        (struct ibgda_mem_object *)calloc(1, sizeof(struct ibgda_mem_object));
    NVSHMEMI_NULL_ERROR_JMP(mobject, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate a new mobject.\n");

    if (alignment > 0) bufsize = size + alignment - 1;

    status = cudaMalloc(&ptr, bufsize);
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                          "cudaMalloc failed.\n");

    attr_val = 1;
    status =
        CUPFN(ibgda_cuda_syms,
              cuPointerSetAttribute(&attr_val, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, (CUdeviceptr)ptr));
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "cuPointerSetAttribute failed.\n");

    status = cudaMemset(ptr, 0, bufsize);
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                          "cudaMemset failed.\n");

    if (alignment > 0) {
        aligned_ptr = (void *)((size_t)((char *)ptr + alignment - 1) & (~(alignment - 1)));
    } else {
        aligned_ptr = ptr;
    }

    if (host_mapping) {
#ifdef NVSHMEM_USE_GDRCOPY
        if (use_gdrcopy) {
            status = gdrcopy_ftable.pin_buffer(gdr_desc, (unsigned long)aligned_ptr,
                                               IBGDA_ROUND_UP(size, IBGDA_GPAGE_SIZE), 0, 0,
                                               &mobject->mh);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "gdrcopy pin_buffer failed \n");

            status = gdrcopy_ftable.map(gdr_desc, mobject->mh, &cpu_ptr_base, size);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gdrcopy map failed \n");

            gdr_info_t info;
            status = gdrcopy_ftable.get_info(gdr_desc, mobject->mh, &info);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "gdrcopy get_info failed \n");

            // remember that mappings start on a 64KB boundary, so let's
            // calculate the offset from the head of the mapping to the
            // beginning of the buffer
            uintptr_t off;
            off = (uintptr_t)aligned_ptr - info.va;
            cpu_ptr = (void *)((uintptr_t)cpu_ptr_base + off);

            mobject->base.cpu_ptr = cpu_ptr_base;
            mobject->aligned.cpu_ptr = cpu_ptr;
        } else
#endif
        {
            NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_NOT_SUPPORTED, out,
                               "host_mapping is not supported as GDRCopy is disable \n");
        }
    }

    mobject->mem_type = IBGDA_MEM_TYPE_GPU;

    mobject->base.gpu_ptr = ptr;
    mobject->base.size = bufsize;

    mobject->aligned.gpu_ptr = aligned_ptr;
    mobject->aligned.size = size;

    mobject->has_cpu_mapping = host_mapping;
    mobject->has_gpu_mapping = true;
    mobject->has_nic_mapping = false;

    *pmobject = mobject;

out:
    if (status) {
        if (ptr) {
            cudaError_t _status = cudaFree(ptr);
            CUDA_RUNTIME_ERROR_STRING(_status);
        }

        if (mobject) free(mobject);
    }
    return status;
}

static void ibgda_gpu_mem_free(struct ibgda_mem_object *mobject) {
    int status = 0;

    if (!mobject) return;

    assert(mobject->mem_type == IBGDA_MEM_TYPE_GPU);

#ifdef NVSHMEM_USE_GDRCOPY
    if (mobject->has_cpu_mapping) {
        assert(use_gdrcopy);

        status = gdrcopy_ftable.unmap(gdr_desc, mobject->mh, mobject->base.cpu_ptr,
                                      mobject->aligned.size);
        if (status) {
            NVSHMEMI_WARN_PRINT("gdr_unmap failed ... Continue\n");
        }

        status = gdrcopy_ftable.unpin_buffer(gdr_desc, mobject->mh);
        if (status) {
            NVSHMEMI_WARN_PRINT("gdr_unpin failed ... Continue\n");
        }
    }
#endif

    status = cudaFree(mobject->base.gpu_ptr);
    CUDA_RUNTIME_ERROR_STRING((cudaError_t)status);

    free(mobject);
}

static int ibgda_host_mem_alloc(struct ibgda_mem_object **pmobject, size_t size, size_t alignment,
                                bool gpu_mapping) {
    int status;

    void *ptr = NULL;

    bool did_host_reg = false;
    void *gpu_ptr;

    struct ibgda_mem_object *mobject =
        (struct ibgda_mem_object *)calloc(1, sizeof(struct ibgda_mem_object));
    NVSHMEMI_NULL_ERROR_JMP(mobject, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate a new mobject.\n");

    status = posix_memalign(&ptr, alignment, size);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "posix_memalign failed.\n");

    memset(ptr, 0, size);

    if (gpu_mapping) {
        status = cudaHostRegister(ptr, size, cudaHostRegisterPortable | cudaHostRegisterMapped);
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                              "cudaHostRegister failed.\n");
        did_host_reg = true;

        status = cudaHostGetDevicePointer(&gpu_ptr, ptr, 0);
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                              "cudaHostGetDevicePointer failed.\n");

        mobject->base.gpu_ptr = gpu_ptr;
        mobject->aligned.gpu_ptr = gpu_ptr;
        mobject->has_gpu_mapping = true;
    }

    mobject->base.cpu_ptr = ptr;
    mobject->base.size = size;

    mobject->aligned.cpu_ptr = ptr;
    mobject->aligned.size = size;

    mobject->has_cpu_mapping = true;

    *pmobject = mobject;

out:
    if (status) {
        if (did_host_reg) {
            cudaError_t _status = cudaHostUnregister(ptr);
            CUDA_RUNTIME_ERROR_STRING(_status);
        }
        if (ptr) free(ptr);
        if (mobject) free(mobject);
    }
    return status;
}

static void ibgda_host_mem_free(struct ibgda_mem_object *mobject) {
    cudaError_t status;

    if (!mobject) return;

    assert(mobject->mem_type == IBGDA_MEM_TYPE_HOST);

    if (mobject->has_gpu_mapping) {
        status = cudaHostUnregister(mobject->base.cpu_ptr);
        CUDA_RUNTIME_ERROR_STRING(status);
    }

    free(mobject->base.cpu_ptr);

    free(mobject);
}

static int ibgda_nic_mem_gpu_map(struct ibgda_mem_object **pmobject, struct mlx5dv_devx_uar *uar,
                                 size_t size) {
    int status = 0;
    bool did_host_reg = false;

    void *ptr = 0;

    struct ibgda_mem_object *mobject =
        (struct ibgda_mem_object *)calloc(1, sizeof(struct ibgda_mem_object));
    NVSHMEMI_NULL_ERROR_JMP(mobject, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate a new mobject.\n");

    status = cudaHostRegister(
        uar->reg_addr, size,
        cudaHostRegisterPortable | cudaHostRegisterMapped | cudaHostRegisterIoMemory);
    if (status != cudaSuccess) {
        NVSHMEMI_WARN_PRINT(
            "cudaHostRegister with IoMemory failed with error=%d. We may need to use a fallback "
            "path.\n",
            status);
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }
    did_host_reg = true;

    status = cudaHostGetDevicePointer(&ptr, uar->reg_addr, 0);
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                          "cudaHostGetDevicePointer failed.\n");

    mobject->mem_type = IBGDA_MEM_TYPE_NIC;

    mobject->base.cpu_ptr = uar->reg_addr;
    mobject->base.gpu_ptr = ptr;
    mobject->base.size = size;

    mobject->aligned.cpu_ptr = uar->reg_addr;
    mobject->aligned.gpu_ptr = ptr;
    mobject->aligned.size = size;

    mobject->uar = uar;

    mobject->has_cpu_mapping = true;
    mobject->has_gpu_mapping = true;
    mobject->has_nic_mapping = true;

    *pmobject = mobject;

out:
    if (status) {
        if (did_host_reg) {
            cudaError_t _status = cudaHostUnregister(uar->reg_addr);
            CUDA_RUNTIME_ERROR_STRING(_status);
        }
        if (mobject) free(mobject);
    }
    return status;
}

static void ibgda_nic_mem_gpu_unmap(struct ibgda_mem_object *mobject) {
    cudaError_t status;

    if (!mobject) return;

    assert(mobject->mem_type == IBGDA_MEM_TYPE_NIC);

    status = cudaHostUnregister(mobject->uar->reg_addr);
    CUDA_RUNTIME_ERROR_STRING(status);

    free(mobject);
}

static int ibgda_nic_mem_cpu_map(struct ibgda_mem_object **pmobject, struct mlx5dv_devx_uar *uar,
                                 size_t size) {
    int status = 0;

    struct ibgda_mem_object *mobject =
        (struct ibgda_mem_object *)calloc(1, sizeof(struct ibgda_mem_object));
    NVSHMEMI_NULL_ERROR_JMP(mobject, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate a new mobject.\n");

    mobject->mem_type = IBGDA_MEM_TYPE_NIC;

    mobject->base.cpu_ptr = uar->reg_addr;
    mobject->base.size = size;

    mobject->aligned.cpu_ptr = uar->reg_addr;
    mobject->aligned.size = size;

    mobject->uar = uar;

    mobject->has_cpu_mapping = true;
    mobject->has_nic_mapping = true;

    *pmobject = mobject;

out:
    return status;
}

static void ibgda_nic_mem_cpu_unmap(struct ibgda_mem_object *mobject) {
    assert(mobject->mem_type == IBGDA_MEM_TYPE_NIC);

    free(mobject);
}

static inline int ibgda_nic_control_alloc(struct ibgda_mem_object **pmobject, size_t size,
                                          size_t alignment) {
    assert(ibgda_nic_buf_location == IBGDA_MEM_TYPE_GPU ||
           ibgda_nic_buf_location == IBGDA_MEM_TYPE_HOST);
    if (ibgda_nic_buf_location == IBGDA_MEM_TYPE_GPU)
        return ibgda_gpu_mem_alloc(pmobject, size, alignment, false);
    else
        return ibgda_host_mem_alloc(pmobject, size, alignment, true);
}

static inline void ibgda_nic_control_free(struct ibgda_mem_object *mobject) {
    assert(ibgda_nic_buf_location == IBGDA_MEM_TYPE_GPU ||
           ibgda_nic_buf_location == IBGDA_MEM_TYPE_HOST);
    if (ibgda_nic_buf_location == IBGDA_MEM_TYPE_GPU)
        ibgda_gpu_mem_free(mobject);
    else
        ibgda_host_mem_free(mobject);
}

static int ibgda_create_cq(struct ibgda_cq **pgcq, struct ibgda_device *device) {
    int status = 0;

    struct ibgda_cq *gcq = NULL;

    struct ibv_pd *pd = device->pd;
    struct ibv_context *context = pd->context;

    void *cq_context;

    uint8_t cmd_in[DEVX_ST_SZ_BYTES(create_cq_in)] = {
        0,
    };
    uint8_t cmd_out[DEVX_ST_SZ_BYTES(create_cq_out)] = {
        0,
    };

    size_t num_cqe = IBGDA_ROUND_UP_POW2_OR_0(ibgda_qp_depth);

    struct mlx5dv_devx_umem *cq_umem = device->cq_shared_object.cq_mobject->umem;
    off_t cq_offset = device->cq_shared_object.cur_cq_off;

    struct mlx5dv_devx_umem *dbr_umem = device->cq_shared_object.dbr_mobject->umem;
    off_t dbr_offset = device->cq_shared_object.cur_dbr_off;

    struct mlx5dv_devx_uar *uar = NULL;

    uint32_t eqn;

    gcq = (struct ibgda_cq *)calloc(1, sizeof(struct ibgda_cq));
    NVSHMEMI_NULL_ERROR_JMP(gcq, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate mem for cq.\n");

    // Query the first EQ
    status = mlx5dv_devx_query_eqn(context, 0, &eqn);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "mlx5dv_devx_query_eqn failed.\n");

    // CQ needs UAR but IBGDA never uses it.
    // So, we don't map this UAR to GPU space.
    uar = mlx5dv_devx_alloc_uar(context, MLX5DV_UAR_ALLOC_TYPE_NC);
    NVSHMEMI_NULL_ERROR_JMP(uar, status, ENOMEM, out, "cannot allocate mlx5dv_devx_uar\n");

    DEVX_SET(create_cq_in, cmd_in, opcode, MLX5_CMD_OP_CREATE_CQ);
    DEVX_SET(create_cq_in, cmd_in, cq_umem_id, cq_umem->umem_id);  // CQ buffer
    DEVX_SET(create_cq_in, cmd_in, cq_umem_valid,
             IBGDA_MLX5_UMEM_VALID_ENABLE);  // Enable cq_umem_id
    DEVX_SET64(create_cq_in, cmd_in, cq_umem_offset, cq_offset);

    cq_context = DEVX_ADDR_OF(create_cq_in, cmd_in, cq_context);
    DEVX_SET(cqc, cq_context, dbr_umem_valid, IBGDA_MLX5_UMEM_VALID_ENABLE);
    DEVX_SET(cqc, cq_context, cqe_sz, MLX5_CQE_SIZE_64B);
    DEVX_SET(cqc, cq_context, cc, 0x1);  // Use collapsed CQ
    DEVX_SET(cqc, cq_context, oi, 0x1);  // Allow overrun
    DEVX_SET(cqc, cq_context, dbr_umem_id, dbr_umem->umem_id);
    DEVX_SET(cqc, cq_context, log_cq_size, IBGDA_ILOG2_OR0(num_cqe));
    DEVX_SET(cqc, cq_context, uar_page, uar->page_id);
    DEVX_SET(cqc, cq_context, c_eqn, eqn);
    DEVX_SET(cqc, cq_context, log_page_size, IBGDA_GPAGE_BITS - MLX5_ADAPTER_PAGE_SHIFT);
    DEVX_SET64(cqc, cq_context, dbr_addr, dbr_offset);  // DBR offset

    gcq->devx_cq =
        mlx5dv_devx_obj_create(context, cmd_in, sizeof(cmd_in), cmd_out, sizeof(cmd_out));
    NVSHMEMI_NULL_ERROR_JMP(gcq->devx_cq, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "Unable to create CQ.\n");

    gcq->cqn = DEVX_GET(create_cq_out, cmd_out, cqn);
    gcq->num_cqe = num_cqe;
    gcq->cq_mobject = device->cq_shared_object.cq_mobject;
    gcq->cq_offset = cq_offset;
    gcq->dbr_mobject = device->cq_shared_object.dbr_mobject;
    gcq->dbr_offset = dbr_offset;
    gcq->uar = uar;

    device->cq_shared_object.cur_cq_off += device->cq_shared_object.cq_buf_size_per_cq;
    device->cq_shared_object.cur_dbr_off += IBGDA_DBRSIZE;

    *pgcq = gcq;

out:
    if (status) {
        if (uar) mlx5dv_devx_free_uar(uar);
        if (gcq) free(gcq);
    }
    return status;
}

static void ibgda_destroy_cq(struct ibgda_cq *gcq) {
    if (!gcq) return;

    if (gcq->devx_cq) {
        mlx5dv_devx_obj_destroy(gcq->devx_cq);
    }

    if (gcq->uar) {
        mlx5dv_devx_free_uar(gcq->uar);
    }

    free(gcq);
}

static void ibgda_get_device_cq(nvshmemi_ibgda_device_cq_t *dev_cq, const struct ibgda_cq *cq) {
    dev_cq->cqn = cq->cqn;
    dev_cq->ncqes = cq->num_cqe;

    assert(cq->cq_mobject->has_gpu_mapping);
    dev_cq->cqe = (void *)((uintptr_t)cq->cq_mobject->aligned.gpu_ptr + cq->cq_offset);

    assert(cq->dbr_mobject->has_gpu_mapping);
    dev_cq->dbrec = (__be32 *)((uintptr_t)cq->dbr_mobject->aligned.gpu_ptr + cq->dbr_offset);
}

static int ibgda_qp_rst2init(struct ibgda_ep *ep, const struct ibgda_device *device, int portid) {
    int status = 0;

    uint8_t cmd_in[DEVX_ST_SZ_BYTES(rst2init_qp_in)] = {
        0,
    };
    uint8_t cmd_out[DEVX_ST_SZ_BYTES(rst2init_qp_out)] = {
        0,
    };

    void *qpc;

    const struct ibv_port_attr *port_attr = device->port_attr + (portid - 1);

    assert(ep->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI ||
           ep->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_RC);

    DEVX_SET(rst2init_qp_in, cmd_in, opcode, MLX5_CMD_OP_RST2INIT_QP);
    DEVX_SET(rst2init_qp_in, cmd_in, qpn, ep->qpn);

    qpc = DEVX_ADDR_OF(rst2init_qp_in, cmd_in, qpc);
    if (ep->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI) {
        DEVX_SET64(qpc, qpc, dc_access_key, IBGDA_DC_ACCESS_KEY);
    } else if (ep->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_RC) {
        DEVX_SET(qpc, qpc, rwe, 1); /* remote write access */
        DEVX_SET(qpc, qpc, rre, 1); /* remote read access */
        DEVX_SET(qpc, qpc, rae, 1); /* remote atomic access */
        /* Currently, NVSHMEM APIs only support atomics up to 64. This field can be updated to
         * support atomics up to 256 bytes. */
        DEVX_SET(qpc, qpc, atomic_mode, IBGDA_MLX5_QPC_ATOMIC_MODE_UP_TO_64BIT);
    }

    DEVX_SET(qpc, qpc, primary_address_path.vhca_port_num, portid);

    if (port_attr->link_layer == IBV_LINK_LAYER_INFINIBAND)
        DEVX_SET(qpc, qpc, primary_address_path.pkey_index, 0);

    DEVX_SET(qpc, qpc, pm_state, MLX5_QPC_PM_STATE_MIGRATED);
    DEVX_SET(qpc, qpc, counter_set_id, 0x0);  // Not connected to a counter set

    status = mlx5dv_devx_obj_modify(ep->devx_qp, cmd_in, sizeof(cmd_in), cmd_out, sizeof(cmd_out));
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Error in mlx5dv_devx_obj_modify for RST2INIT_QP with syndrome %x\n",
                          DEVX_GET(rst2init_qp_out, cmd_out, syndrome));

    ep->portid = portid;

out:
    return status;
}

static int ibgda_dci_init2rtr(nvshmemt_ibgda_state_t *ibgda_state, struct ibgda_ep *ep,
                              const struct ibgda_device *device, int portid) {
    int status = 0;

    uint8_t cmd_in[DEVX_ST_SZ_BYTES(init2rtr_qp_in)] = {
        0,
    };
    uint8_t cmd_out[DEVX_ST_SZ_BYTES(init2rtr_qp_out)] = {
        0,
    };

    void *qpc;

    const struct ibv_port_attr *port_attr = device->port_attr + (portid - 1);

    assert(ep->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI);

    DEVX_SET(init2rtr_qp_in, cmd_in, opcode, MLX5_CMD_OP_INIT2RTR_QP);
    DEVX_SET(init2rtr_qp_in, cmd_in, qpn, ep->qpn);

    qpc = DEVX_ADDR_OF(init2rtr_qp_in, cmd_in, qpc);
    DEVX_SET(qpc, qpc, mtu, port_attr->active_mtu);
    DEVX_SET(qpc, qpc, log_msg_max, IBGDA_LOG_MAX_MSG_SIZE);

    if (port_attr->link_layer == IBV_LINK_LAYER_INFINIBAND) {
        DEVX_SET(qpc, qpc, primary_address_path.sl, ibgda_state->options->IB_SL);
    } else if (port_attr->link_layer == IBV_LINK_LAYER_ETHERNET) {
        DEVX_SET(qpc, qpc, primary_address_path.tclass, ibgda_state->options->IB_TRAFFIC_CLASS);
        DEVX_SET(qpc, qpc, primary_address_path.eth_prio, ibgda_state->options->IB_SL);
        DEVX_SET(qpc, qpc, primary_address_path.dscp, ibgda_state->options->IB_TRAFFIC_CLASS >> 2);
    }

    status = mlx5dv_devx_obj_modify(ep->devx_qp, cmd_in, sizeof(cmd_in), cmd_out, sizeof(cmd_out));
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Error in mlx5dv_devx_obj_modify for INIT2RTR_QP with syndrome %x\n",
                          DEVX_GET(init2rtr_qp_out, cmd_out, syndrome));

out:
    return status;
}

static int ibgda_rc_init2rtr(nvshmemt_ibgda_state_t *ibgda_state, struct ibgda_ep *ep,
                             const struct ibgda_device *device, int portid,
                             struct ibgda_rc_handle *peer_ep_handle) {
    int status = 0;

    uint8_t cmd_in[DEVX_ST_SZ_BYTES(init2rtr_qp_in)] = {
        0,
    };
    uint8_t cmd_out[DEVX_ST_SZ_BYTES(init2rtr_qp_out)] = {
        0,
    };

    void *qpc;

    const struct ibv_port_attr *port_attr = device->port_attr + (portid - 1);

    assert(ep->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_RC);

    DEVX_SET(init2rtr_qp_in, cmd_in, opcode, MLX5_CMD_OP_INIT2RTR_QP);
    DEVX_SET(init2rtr_qp_in, cmd_in, qpn, ep->qpn);

    qpc = DEVX_ADDR_OF(init2rtr_qp_in, cmd_in, qpc);
    DEVX_SET(qpc, qpc, mtu, port_attr->active_mtu);
    DEVX_SET(qpc, qpc, log_msg_max, IBGDA_LOG_MAX_MSG_SIZE);
    DEVX_SET(qpc, qpc, remote_qpn, peer_ep_handle->qpn);
    DEVX_SET(qpc, qpc, min_rnr_nak, IBGDA_MIN_RNR_NAK);
    DEVX_SET(qpc, qpc, log_rra_max, IBGDA_ILOG2_OR0(device->device_attr.max_qp_rd_atom));

    if (port_attr->link_layer == IBV_LINK_LAYER_INFINIBAND) {
        DEVX_SET(qpc, qpc, primary_address_path.tclass, ibgda_state->options->IB_TRAFFIC_CLASS);
        DEVX_SET(qpc, qpc, primary_address_path.rlid, peer_ep_handle->lid);
        DEVX_SET(qpc, qpc, primary_address_path.mlid, 0);
        DEVX_SET(qpc, qpc, primary_address_path.sl, ibgda_state->options->IB_SL);
        DEVX_SET(qpc, qpc, primary_address_path.grh, false);
    } else if (port_attr->link_layer == IBV_LINK_LAYER_ETHERNET) {
        ib_get_gid_index(&ftable, device->context, portid, port_attr->gid_tbl_len,
                         (int *)&device->gid_info[portid - 1].local_gid_index,
                         ibgda_state->log_level, ibgda_state->options);
        ftable.query_gid(device->context, portid, device->gid_info[portid - 1].local_gid_index,
                         (ibv_gid *)&device->gid_info[portid - 1].local_gid);
        struct ibv_ah_attr ah_attr;
        struct ibv_ah *ah;
        struct mlx5dv_obj dv;
        struct mlx5dv_ah dah;

        ah_attr.is_global = 1;
        ah_attr.port_num = portid;
        ah_attr.grh.dgid.global.subnet_prefix = peer_ep_handle->spn;
        ah_attr.grh.dgid.global.interface_id = peer_ep_handle->iid;
        ah_attr.grh.sgid_index = device->gid_info[portid - 1].local_gid_index;
        ah_attr.grh.traffic_class = ibgda_state->options->IB_TRAFFIC_CLASS;
        ah_attr.sl = ibgda_state->options->IB_SL;
        ah_attr.src_path_bits = 0;

        ah = ftable.create_ah(device->pd, &ah_attr);
        NVSHMEMI_NULL_ERROR_JMP(ah, status, NVSHMEMX_ERROR_INTERNAL, out, "Unable to create ah.\n");

        dv.ah.in = ah;
        dv.ah.out = &dah;
        mlx5dv_init_obj(&dv, MLX5DV_OBJ_AH);

        memcpy(DEVX_ADDR_OF(qpc, qpc, primary_address_path.rmac_47_32), &dah.av->rmac,
               sizeof(dah.av->rmac));
        DEVX_SET(qpc, qpc, primary_address_path.hop_limit, IBGDA_GRH_HOP_LIMIT);
        DEVX_SET(qpc, qpc, primary_address_path.src_addr_index,
                 device->gid_info[portid - 1].local_gid_index);
        DEVX_SET(qpc, qpc, primary_address_path.eth_prio, ibgda_state->options->IB_SL);
        DEVX_SET(qpc, qpc, primary_address_path.udp_sport, ah_attr.dlid);
        DEVX_SET(qpc, qpc, primary_address_path.dscp, ibgda_state->options->IB_TRAFFIC_CLASS >> 2);

        memcpy(DEVX_ADDR_OF(qpc, qpc, primary_address_path.rgid_rip), &dah.av->rgid,
               sizeof(dah.av->rgid));
        ep->ah = ah;
    }

    status = mlx5dv_devx_obj_modify(ep->devx_qp, cmd_in, sizeof(cmd_in), cmd_out, sizeof(cmd_out));
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Error in mlx5dv_devx_obj_modify for INIT2RTR_QP with syndrome %x\n",
                          DEVX_GET(init2rtr_qp_out, cmd_out, syndrome));
out:
    return status;
}

static int ibgda_qp_rtr2rts(struct ibgda_ep *ep, const struct ibgda_device *device, int portid) {
    int status = 0;

    uint8_t cmd_in[DEVX_ST_SZ_BYTES(rtr2rts_qp_in)] = {
        0,
    };
    uint8_t cmd_out[DEVX_ST_SZ_BYTES(rtr2rts_qp_out)] = {
        0,
    };

    void *qpc;

    assert(ep->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI ||
           ep->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_RC);

    DEVX_SET(rtr2rts_qp_in, cmd_in, opcode, MLX5_CMD_OP_RTR2RTS_QP);
    DEVX_SET(rtr2rts_qp_in, cmd_in, qpn, ep->qpn);

    qpc = DEVX_ADDR_OF(rtr2rts_qp_in, cmd_in, qpc);
    DEVX_SET(qpc, qpc, log_ack_req_freq, 0x0);  // Ack every packet
    DEVX_SET(qpc, qpc, log_sra_max, IBGDA_ILOG2_OR0(device->device_attr.max_qp_rd_atom));
    DEVX_SET(qpc, qpc, next_send_psn, 0x0);
    DEVX_SET(qpc, qpc, retry_count, 7);
    DEVX_SET(qpc, qpc, rnr_retry, 7);
    DEVX_SET(qpc, qpc, primary_address_path.ack_timeout, 20);

    status = mlx5dv_devx_obj_modify(ep->devx_qp, cmd_in, sizeof(cmd_in), cmd_out, sizeof(cmd_out));
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Error in mlx5dv_devx_obj_modify for RTR2RTS_QP with syndrome %x\n",
                          DEVX_GET(rtr2rts_qp_out, cmd_out, syndrome));

out:
    return status;
}

static int ibgda_destroy_internal_buffer(nvshmemt_ibgda_state_t *ibgda_state,
                                         struct ibgda_device *device) {
    int status = 0;

    struct ibgda_mem_object *internal_buf_mobject = NULL;
    struct nvshmemt_ib_common_mem_handle *internal_buf_mhandle = NULL;

    internal_buf_mobject = device->qp_shared_object.internal_buf.mem_object;
    internal_buf_mhandle = device->qp_shared_object.internal_buf.mem_handle;

    if (internal_buf_mhandle) {
        nvshmemt_ib_common_release_mem_handle(&ftable, (nvshmem_mem_handle_t *)internal_buf_mhandle,
                                              ibgda_state->log_level);
        free(internal_buf_mhandle);
    }

    if (internal_buf_mobject) {
        ibgda_gpu_mem_free(internal_buf_mobject);
    }

    return status;
}

static int ibgda_create_internal_buffer(struct ibgda_internal_buffer *internal_buf,
                                        nvshmemt_ibgda_state_t *ibgda_state,
                                        struct ibgda_device *device, int n_pes) {
    int status = 0;

    struct ibgda_mem_object *internal_buf_mobject = NULL;
    struct nvshmemt_ib_common_mem_handle *internal_buf_mhandle = NULL;

    size_t size_per_dci =
        NVSHMEMI_IBGDA_IBUF_SLOT_SIZE * (ibgda_num_fetch_slots_per_dci + IBGDA_IBUF_RESERVED_SLOTS);
    size_t size_per_rc =
        NVSHMEMI_IBGDA_IBUF_SLOT_SIZE * (ibgda_num_fetch_slots_per_rc + IBGDA_IBUF_RESERVED_SLOTS);
    size_t buf_size =
        (size_per_dci * device->dci.num_eps) + (size_per_rc * device->rc.num_eps_per_pe * n_pes);

    status = ibgda_gpu_mem_alloc(&internal_buf_mobject, buf_size, IBGDA_GPAGE_SIZE, false);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "cannot allocate internal buffer.\n");

    internal_buf_mhandle =
        (struct nvshmemt_ib_common_mem_handle *)calloc(1, sizeof(*internal_buf_mhandle));
    NVSHMEMI_NULL_ERROR_JMP(internal_buf_mhandle, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate internal_buf_mhandle.\n");

    status = nvshmemt_ib_common_reg_mem_handle(
        &ftable, device->pd, (nvshmem_mem_handle_t *)internal_buf_mhandle,
        (void *)internal_buf_mobject->aligned.gpu_ptr, internal_buf_mobject->aligned.size, false,
        ibgda_state->dmabuf_support_for_data_buffers, ibgda_cuda_syms, ibgda_state->log_level,
        ibgda_state->options->IB_ENABLE_RELAXED_ORDERING);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Unable to register memory for IBGDA transport.\n");

    internal_buf->mem_object = internal_buf_mobject;
    internal_buf->mem_handle = internal_buf_mhandle;

out:
    if (status) {
        if (internal_buf_mhandle) {
            nvshmemt_ib_common_release_mem_handle(
                &ftable, (nvshmem_mem_handle_t *)internal_buf_mhandle, ibgda_state->log_level);
            free(internal_buf_mhandle);
        }
        if (internal_buf_mobject) ibgda_gpu_mem_free(internal_buf_mobject);
    }
    return status;
}

static void ibgda_destroy_cq_shared_objects(nvshmemt_ibgda_state_t *ibgda_state,
                                            struct ibgda_device *device) {
    if (device->cq_shared_object.dbr_mobject) {
        if (device->cq_shared_object.dbr_mobject->has_nic_mapping)
            ibgda_mobject_nic_unmap(device->cq_shared_object.dbr_mobject);
        ibgda_nic_control_free(device->cq_shared_object.dbr_mobject);
    }

    if (device->cq_shared_object.cq_mobject) {
        if (device->cq_shared_object.cq_mobject->has_nic_mapping)
            ibgda_mobject_nic_unmap(device->cq_shared_object.cq_mobject);
        ibgda_nic_control_free(device->cq_shared_object.cq_mobject);
    }
}

static int ibgda_destroy_qp_shared_objects(nvshmemt_ibgda_state_t *ibgda_state,
                                           struct ibgda_device *device) {
    int status = 0;

    status = ibgda_destroy_internal_buffer(ibgda_state, device);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "ibgda_destroy_internal_buffer failed.\n");

    if (device->qp_shared_object.recv_cq) {
        status = ftable.destroy_cq(device->qp_shared_object.recv_cq);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "ibgda_destroy_cq failed for recv_cq.\n");
    }

    if (device->qp_shared_object.srq) {
        status = ftable.destroy_srq(device->qp_shared_object.srq);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "destroy_srq failed.\n");
    }

    if (device->qp_shared_object.prod_idx_mobject)
        ibgda_gpu_mem_free(device->qp_shared_object.prod_idx_mobject);

    if (device->qp_shared_object.prod_idx_cache) free(device->qp_shared_object.prod_idx_cache);

    if (device->qp_shared_object.prod_idx_snapshot)
        free(device->qp_shared_object.prod_idx_snapshot);

    if (device->qp_shared_object.dbr_mobject) {
        if (device->qp_shared_object.dbr_mobject->has_nic_mapping)
            ibgda_mobject_nic_unmap(device->qp_shared_object.dbr_mobject);
        if (ibgda_nic_handler == IBGDA_NIC_HANDLER_GPU)
            ibgda_nic_control_free(device->qp_shared_object.dbr_mobject);
        else
            ibgda_host_mem_free(device->qp_shared_object.dbr_mobject);
    }

    if (device->qp_shared_object.wq_mobject) {
        if (device->qp_shared_object.wq_mobject->has_nic_mapping)
            ibgda_mobject_nic_unmap(device->qp_shared_object.wq_mobject);
        ibgda_nic_control_free(device->qp_shared_object.wq_mobject);
    }

out:
    return status;
}

static int ibgda_create_cq_shared_objects(nvshmemt_ibgda_state_t *ibgda_state,
                                          struct ibgda_device *device, int n_pes) {
    int status = 0;

    struct ibv_context *context = device->context;

    unsigned int num_cqs = device->dci.num_eps + device->rc.num_eps_per_pe * n_pes;

    assert(ibgda_qp_depth > 0);
    size_t num_cqe = IBGDA_ROUND_UP_POW2_OR_0(ibgda_qp_depth);
    size_t cq_buf_size_per_cq = num_cqe * NVSHMEMI_IBGDA_CQE_SIZE;
    size_t cq_buf_size = num_cqs * cq_buf_size_per_cq;

    size_t dbr_buf_size = IBGDA_DBRSIZE * num_cqs;

    struct ibgda_mem_object *cq_mobject = NULL;
    struct ibgda_mem_object *dbr_mobject = NULL;

    // Allocate and map CQ buffer for all CQs.
    status = ibgda_nic_control_alloc(&cq_mobject, cq_buf_size, IBGDA_GPAGE_SIZE);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cannot allocate cq buf.\n");

    status = cudaMemset(cq_mobject->base.gpu_ptr, 0xff, cq_mobject->base.size);
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                          "cudaMemset failed.\n");

    status = ibgda_mobject_nic_map(cq_mobject, context, IBV_ACCESS_LOCAL_WRITE,
                                   ibgda_state->dmabuf_support_for_control_buffers);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cannot register cq buf.\n");

    // Allocate and map Doorbell Record buffer for all CQs.
    status = ibgda_nic_control_alloc(&dbr_mobject, dbr_buf_size, IBGDA_GPAGE_SIZE);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cannot allocate dbr buf.\n");

    status = ibgda_mobject_nic_map(dbr_mobject, context, IBV_ACCESS_LOCAL_WRITE,
                                   ibgda_state->dmabuf_support_for_control_buffers);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cannot register dbr buf.\n");

    // Output
    device->cq_shared_object.cq_buf_size_per_cq = cq_buf_size_per_cq;
    device->cq_shared_object.cq_mobject = cq_mobject;
    device->cq_shared_object.dbr_mobject = dbr_mobject;

out:
    if (status) {
        if (dbr_mobject) {
            if (dbr_mobject->has_nic_mapping) ibgda_mobject_nic_unmap(dbr_mobject);
            ibgda_nic_control_free(dbr_mobject);
        }
        if (cq_mobject) {
            if (cq_mobject->has_nic_mapping) ibgda_mobject_nic_unmap(cq_mobject);
            ibgda_nic_control_free(cq_mobject);
        }
    }
    return status;
}

static int ibgda_create_qp_shared_objects(nvshmemt_ibgda_state_t *ibgda_state,
                                          struct ibgda_device *device, int n_pes) {
    int status = 0;

    struct ibv_context *context = device->context;
    struct ibv_pd *pd = device->pd;

    struct ibv_srq *srq = NULL;
    struct ibv_srq_init_attr srq_init_attr;

    struct ibv_cq *recv_cq = NULL;

    struct ibgda_mem_object *prod_idx_mobject = NULL;
    uint64_t *prod_idx_cache = NULL;
    uint64_t *prod_idx_snapshot = NULL;
    unsigned int num_eps = device->dci.num_eps + device->rc.num_eps_per_pe * n_pes;

    mlx5dv_obj dv_obj;
    struct mlx5dv_pd dvpd;
    struct mlx5dv_cq dvscq;
    struct mlx5dv_cq dvrcq;
    struct mlx5dv_srq dvsrq;

    int pdn = 0;
    int srqn = 0;
    int rcqn = 0;

    assert(ibgda_qp_depth > 0);
    size_t num_wqebb = IBGDA_ROUND_UP_POW2_OR_0(ibgda_qp_depth);

    size_t wq_buf_size_per_qp;
    size_t wq_buf_size;
    struct ibgda_mem_object *wq_mobject = NULL;

    size_t dbr_buf_size;
    struct ibgda_mem_object *dbr_mobject = NULL;

    // Initialization
    memset(&srq_init_attr, 0, sizeof(srq_init_attr));
    memset(&dvpd, 0, sizeof(dvpd));
    memset(&dvscq, 0, sizeof(dvscq));
    memset(&dvrcq, 0, sizeof(dvrcq));
    memset(&dvsrq, 0, sizeof(dvsrq));

    // Query pdn
    memset(&dv_obj, 0, sizeof(dv_obj));
    dv_obj.pd.in = pd;
    dv_obj.pd.out = &dvpd;

    status = mlx5dv_init_obj(&dv_obj, MLX5DV_OBJ_PD);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "mlx5dv PD initialization failed.\n");

    pdn = dvpd.pdn;

    // Create srq on host memory.
    srq_init_attr.attr.max_wr = ibgda_srq_depth;
    srq_init_attr.attr.max_sge = 1;

    srq = ftable.create_srq(pd, &srq_init_attr);
    NVSHMEMI_NULL_ERROR_JMP(srq, status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_create_srq failed.\n");

    memset(&dv_obj, 0, sizeof(dv_obj));
    dvsrq.comp_mask = MLX5DV_SRQ_MASK_SRQN;
    dv_obj.srq.in = srq;
    dv_obj.srq.out = &dvsrq;

    status = mlx5dv_init_obj(&dv_obj, MLX5DV_OBJ_SRQ);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "mlx5dv SRQ initialization failed.\n");

    srqn = dvsrq.srqn;
    NVSHMEMI_EQ_ERROR_JMP(srqn, 0, NVSHMEMX_ERROR_INTERNAL, out,
                          "Unable to allocate SRQ for your device. "
                          "This may occur if your ofed is older than version 5.0.\n");

    // Create recv_cq on host memory.
    recv_cq = ftable.create_cq(context, ibgda_srq_depth, NULL, NULL, 0);
    NVSHMEMI_NULL_ERROR_JMP(recv_cq, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "ibv_create_cq for recv_cq failed.\n");

    memset(&dv_obj, 0, sizeof(dv_obj));
    dv_obj.cq.in = recv_cq;
    dv_obj.cq.out = &dvrcq;

    status = mlx5dv_init_obj(&dv_obj, MLX5DV_OBJ_CQ);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "mlx5dv RCQ initialization failed.\n");

    rcqn = dvrcq.cqn;

    status = ibgda_create_internal_buffer(&device->qp_shared_object.internal_buf, ibgda_state,
                                          device, n_pes);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "ibgda_create_internal_buffer failed.\n");

    if (ibgda_nic_handler == IBGDA_NIC_HANDLER_CPU) {
        status = ibgda_gpu_mem_alloc(&prod_idx_mobject, sizeof(uint64_t) * num_eps,
                                     IBGDA_GPAGE_SIZE, true);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "cannot allocate prod_idx_mobject.\n");

        prod_idx_cache = (uint64_t *)calloc(num_eps, sizeof(uint64_t));
        NVSHMEMI_NULL_ERROR_JMP(prod_idx_cache, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                "Unable to allocate mem for prod_idx_cache.\n");

        prod_idx_snapshot = (uint64_t *)calloc(num_eps, sizeof(uint64_t));
        NVSHMEMI_NULL_ERROR_JMP(prod_idx_snapshot, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                "Unable to allocate mem for prod_idx_snapshot.\n");
    }

    // Allocate and map WQ buffer for all QPs.
    wq_buf_size_per_qp = num_wqebb * MLX5_SEND_WQE_BB;  // num_wqebb is always a power of 2
    wq_buf_size = wq_buf_size_per_qp * num_eps;
    status = ibgda_nic_control_alloc(&wq_mobject, wq_buf_size, IBGDA_GPAGE_SIZE);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cannot allocate wq buf.\n");

    status = ibgda_mobject_nic_map(wq_mobject, context, IBV_ACCESS_LOCAL_WRITE,
                                   ibgda_state->dmabuf_support_for_control_buffers);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cannot register wq buf.\n");

    // Allocate and map Doorbell Record buffer for all QPs.
    dbr_buf_size = IBGDA_DBRSIZE * num_eps;
    if (ibgda_nic_handler == IBGDA_NIC_HANDLER_GPU)
        status = ibgda_nic_control_alloc(&dbr_mobject, dbr_buf_size, IBGDA_GPAGE_SIZE);
    else
        status = ibgda_host_mem_alloc(&dbr_mobject, dbr_buf_size, IBGDA_GPAGE_SIZE, true);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cannot allocate dbr buf.\n");

    status = ibgda_mobject_nic_map(dbr_mobject, context, IBV_ACCESS_LOCAL_WRITE,
                                   ibgda_state->dmabuf_support_for_control_buffers);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cannot register dbr buf.\n");

    // Output
    device->qp_shared_object.srq = srq;
    device->qp_shared_object.recv_cq = recv_cq;
    device->qp_shared_object.pdn = pdn;
    device->qp_shared_object.srqn = srqn;
    device->qp_shared_object.rcqn = rcqn;
    device->qp_shared_object.prod_idx_mobject = prod_idx_mobject;
    device->qp_shared_object.prod_idx_cache = prod_idx_cache;
    device->qp_shared_object.prod_idx_snapshot = prod_idx_snapshot;
    device->qp_shared_object.wq_buf_size_per_qp = wq_buf_size_per_qp;
    device->qp_shared_object.wq_mobject = wq_mobject;
    device->qp_shared_object.dbr_mobject = dbr_mobject;

out:
    if (status) {
        if (dbr_mobject) {
            if (dbr_mobject->has_nic_mapping) ibgda_mobject_nic_unmap(dbr_mobject);
            ibgda_nic_control_free(dbr_mobject);
        }
        if (wq_mobject) {
            if (wq_mobject->has_nic_mapping) ibgda_mobject_nic_unmap(wq_mobject);
            ibgda_nic_control_free(wq_mobject);
        }
        if (recv_cq) ftable.destroy_cq(recv_cq);
        if (srq) ftable.destroy_srq(srq);
        if (prod_idx_mobject) ibgda_gpu_mem_free(prod_idx_mobject);
        if (prod_idx_cache) free(prod_idx_cache);
        if (prod_idx_snapshot) free(prod_idx_snapshot);
    }
    return status;
}

static int ibgda_alloc_and_map_qp_uar(struct ibv_context *context, ibgda_nic_handler_t handler,
                                      struct ibgda_mem_object **out_mobject) {
    int status = 0;

    struct mlx5dv_devx_uar *uar = NULL;
    struct ibgda_mem_object *uar_mobject = NULL;
    size_t uar_reg_size = 0;
    uint8_t log_bf_reg_size = 0;

#ifdef HAVE_MLX5DV_UAR_ALLOC_TYPE_NC_DEDICATED
    uar = mlx5dv_devx_alloc_uar(context, MLX5DV_UAR_ALLOC_TYPE_NC_DEDICATED);
    if (uar)
        uar_reg_size = IBGDA_MLX5_NC_UAR_SIZE;
    else
#endif
    {
        uint8_t cmd_cap_in[DEVX_ST_SZ_BYTES(query_hca_cap_in)] = {
            0,
        };
        uint8_t cmd_cap_out[DEVX_ST_SZ_BYTES(query_hca_cap_out)] = {
            0,
        };
        void *cap;

        DEVX_SET(query_hca_cap_in, cmd_cap_in, opcode, MLX5_CMD_OP_QUERY_HCA_CAP);
        DEVX_SET(query_hca_cap_in, cmd_cap_in, op_mod,
                 MLX5_SET_HCA_CAP_OP_MOD_GENERAL_DEVICE | (MLX5_CAP_GENERAL << 1) |
                     HCA_CAP_OPMOD_GET_CUR);

        status = mlx5dv_devx_general_cmd(context, cmd_cap_in, sizeof(cmd_cap_in), cmd_cap_out,
                                         sizeof(cmd_cap_out));
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "mlx5dv_devx_general_cmd for hca cap failed.\n");

        cap = DEVX_ADDR_OF(query_hca_cap_out, cmd_cap_out, capability.cmd_hca_cap);
        log_bf_reg_size = DEVX_GET(cmd_hca_cap, cap, log_bf_reg_size);

        // The size of 1st + 2nd half (as when we use alternating DB)
        uar_reg_size = 1LLU << log_bf_reg_size;

        // Allocate UAR. This will be used as a DB/BF register.
        uar = mlx5dv_devx_alloc_uar(context, MLX5DV_UAR_ALLOC_TYPE_BF);
        NVSHMEMI_NULL_ERROR_JMP(uar, status, ENOMEM, out, "cannot allocate mlx5dv_devx_uar\n");
    }

    // Map the UAR to GPU
    if (handler == IBGDA_NIC_HANDLER_GPU) {
        status = ibgda_nic_mem_gpu_map(&uar_mobject, uar, uar_reg_size);
        if (status) {
            NVSHMEMI_WARN_PRINT(
                "ibgda_nic_mem_gpu_map failed. We may need to use the CPU fallback path.\n");
            status = NVSHMEMX_ERROR_INTERNAL;
            goto out;
        }
    } else {
        status = ibgda_nic_mem_cpu_map(&uar_mobject, uar, uar_reg_size);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "ibgda_nic_mem_cpu_map failed.\n");
    }

    *out_mobject = uar_mobject;

out:
    if (status) {
        if (uar_mobject) {
            if (handler == IBGDA_NIC_HANDLER_GPU)
                ibgda_nic_mem_gpu_unmap(uar_mobject);
            else
                ibgda_nic_mem_cpu_unmap(uar_mobject);
        }
        if (uar) mlx5dv_devx_free_uar(uar);
    }
    return status;
}

static void ibgda_unmap_and_free_qp_uar(struct ibgda_mem_object *mobject) {
    struct mlx5dv_devx_uar *uar = NULL;

    if (!mobject) return;

    uar = mobject->uar;

    if (mobject->has_gpu_mapping)
        ibgda_nic_mem_gpu_unmap(mobject);
    else
        ibgda_nic_mem_cpu_unmap(mobject);

    if (uar) mlx5dv_devx_free_uar(uar);
}

/**
 * Create a RC or DCI QP.
 * DCT creation is not handled by this function.
 */
static int ibgda_create_qp(struct ibgda_ep **ep_ptr, struct ibgda_device *device, int portid,
                           uint32_t qp_idx, nvshmemi_ibgda_device_qp_type_t qp_type) {
    struct ibv_pd *pd = device->pd;
    struct ibv_context *context = pd->context;
    struct ibgda_ep *ep = NULL;

    void *qp_context;

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

    struct ibgda_mem_object *uar_mobject = NULL;

    struct mlx5dv_devx_umem *wq_umem = NULL;
    off_t wq_offset = 0;

    struct mlx5dv_devx_umem *dbr_umem = NULL;
    off_t dbr_offset = 0;

    int cqe_version = 0;

    struct ibgda_cq *send_cq = NULL;

    size_t num_wqebb = IBGDA_ROUND_UP_POW2_OR_0(ibgda_qp_depth);

    int status = 0;

    assert(qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI ||
           qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_RC);

    DEVX_SET(query_hca_cap_in, cmd_cap_in, opcode, MLX5_CMD_OP_QUERY_HCA_CAP);
    DEVX_SET(
        query_hca_cap_in, cmd_cap_in, op_mod,
        MLX5_SET_HCA_CAP_OP_MOD_GENERAL_DEVICE | (MLX5_CAP_GENERAL << 1) | HCA_CAP_OPMOD_GET_CUR);

    status = mlx5dv_devx_general_cmd(context, cmd_cap_in, sizeof(cmd_cap_in), cmd_cap_out,
                                     sizeof(cmd_cap_out));
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "mlx5dv_devx_general_cmd for hca cap failed.\n");

    cap = DEVX_ADDR_OF(query_hca_cap_out, cmd_cap_out, capability.cmd_hca_cap);
    cqe_version = DEVX_GET(cmd_hca_cap, cap, cqe_version);
    if (cqe_version != 1) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_NOT_SUPPORTED, out,
                           "hca_cap.cqe_version != 1 is not supported.\n");
    }

    // Create send_cq on GPU memory.
    status = ibgda_create_cq(&send_cq, device);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibgda_create_cq failed.\n");

    ep = (struct ibgda_ep *)calloc(1, sizeof(struct ibgda_ep));
    NVSHMEMI_NULL_ERROR_JMP(ep, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate mem for ep.\n");

    // Allocate and map UAR. This will be used as a DB/BF register.
    status = ibgda_alloc_and_map_qp_uar(context, ibgda_nic_handler, &uar_mobject);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "ibgda_alloc_and_map_qp_uar failed\n");

    wq_umem = device->qp_shared_object.wq_mobject->umem;
    wq_offset = device->qp_shared_object.cur_wq_off;

    dbr_umem = device->qp_shared_object.dbr_mobject->umem;
    dbr_offset = device->qp_shared_object.cur_dbr_off;

    DEVX_SET(create_qp_in, cmd_in, opcode, MLX5_CMD_OP_CREATE_QP);
    DEVX_SET(create_qp_in, cmd_in, wq_umem_id, wq_umem->umem_id);  // WQ buffer
    DEVX_SET64(create_qp_in, cmd_in, wq_umem_offset, wq_offset);
    DEVX_SET(create_qp_in, cmd_in, wq_umem_valid,
             IBGDA_MLX5_UMEM_VALID_ENABLE);  // Enable wq_umem_id

    qp_context = DEVX_ADDR_OF(create_qp_in, cmd_in, qpc);
    DEVX_SET(qpc, qp_context, st,
             qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI ? IBGDA_MLX5_QPC_ST_DCI
                                                          : IBGDA_MLX5_QPC_ST_RC);
    DEVX_SET(qpc, qp_context, pm_state, MLX5_QPC_PM_STATE_MIGRATED);
    DEVX_SET(qpc, qp_context, pd, device->qp_shared_object.pdn);
    DEVX_SET(qpc, qp_context, uar_page, uar_mobject->uar->page_id);  // BF register
    DEVX_SET(qpc, qp_context, rq_type, IBGDA_SRQ_TYPE_VALUE);        // Shared Receive Queue
    DEVX_SET(qpc, qp_context, srqn_rmpn_xrqn, device->qp_shared_object.srqn);
    DEVX_SET(qpc, qp_context, cqn_snd, send_cq->cqn);
    DEVX_SET(qpc, qp_context, cqn_rcv, device->qp_shared_object.rcqn);
    DEVX_SET(qpc, qp_context, log_sq_size, IBGDA_ILOG2_OR0(num_wqebb));
    DEVX_SET(qpc, qp_context, log_rq_size, 0);
    DEVX_SET(qpc, qp_context, cs_req, 0);                                     // Disable CS Request
    DEVX_SET(qpc, qp_context, cs_res, 0);                                     // Disable CS Response
    DEVX_SET(qpc, qp_context, dbr_umem_valid, IBGDA_MLX5_UMEM_VALID_ENABLE);  // Enable dbr_umem_id
    DEVX_SET64(qpc, qp_context, dbr_addr,
               dbr_offset);  // Offset of dbr_umem_id (behavior changed because of dbr_umem_valid)
    DEVX_SET(qpc, qp_context, dbr_umem_id, dbr_umem->umem_id);  // DBR buffer
    DEVX_SET(qpc, qp_context, user_index, qp_idx);
    DEVX_SET(qpc, qp_context, page_offset, 0);

    ep->devx_qp = mlx5dv_devx_obj_create(context, cmd_in, sizeof(cmd_in), cmd_out, sizeof(cmd_out));
    NVSHMEMI_NULL_ERROR_JMP(ep->devx_qp, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "Unable to create QP for EP.\n");

    ep->qpn = DEVX_GET(create_qp_out, cmd_out, qpn);
    ep->portid = portid;

    ep->sq_cnt = num_wqebb;
    ep->sq_buf_offset = 0;

    ep->rq_cnt = 0;
    ep->rq_buf_offset = 0;

    ep->wq_mobject = device->qp_shared_object.wq_mobject;
    ep->wq_offset = wq_offset;
    device->qp_shared_object.cur_wq_off += device->qp_shared_object.wq_buf_size_per_qp;

    ep->dbr_mobject = device->qp_shared_object.dbr_mobject;
    ep->dbr_offset = dbr_offset;
    device->qp_shared_object.cur_dbr_off += IBGDA_DBRSIZE;

    ep->uar_mobject = uar_mobject;

    ep->send_cq = send_cq;

    ep->qp_type = qp_type;

    ep->user_index = qp_idx;

    *ep_ptr = ep;

out:
    if (status) {
        if (uar_mobject) ibgda_unmap_and_free_qp_uar(uar_mobject);
        if (send_cq) ibgda_destroy_cq(send_cq);
        if (ep) free(ep);
    }

    return status;
}

static int ibgda_get_rc_handle(struct ibgda_rc_handle *rc_handle, const struct ibgda_ep *ep,
                               const struct ibgda_device *device) {
    const struct ibv_port_attr *port_attr = &device->port_attr[ep->portid - 1];
    const union ibv_gid *gid = &device->gid_info[ep->portid - 1].local_gid;

    assert(ep->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_RC);

    rc_handle->qpn = ep->qpn;
    rc_handle->lid = port_attr->lid;
    if (rc_handle->lid == 0) {
        rc_handle->spn = gid->global.subnet_prefix;
        rc_handle->iid = gid->global.interface_id;
    }

    return 0;
}

static int ibgda_destroy_dct_shared_objects(nvshmemt_ibgda_state_t *ibgda_state,
                                            struct ibgda_device *device) {
    int status = 0;

    if (device->dct.ah) {
        status = ftable.destroy_ah(device->dct.ah);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "destroy_ah failed.\n");
    }
    if (device->dct.send_cq) {
        status = ftable.destroy_cq(device->dct.send_cq);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "destroy_cq failed for send_cq.\n");
    }
    if (device->dct.recv_cq) {
        status = ftable.destroy_cq(device->dct.recv_cq);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "destroy_cq failed for recv_cq.\n");
    }
    if (device->dct.srq) {
        status = ftable.destroy_srq(device->dct.srq);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "destroy_srq failed for dct.srq.\n");
    }
    if (device->dct.pd) {
        status = ftable.dealloc_pd(device->dct.pd);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "ibv_dealloc_pd failed for dct.pd.\n");
    }

out:
    return status;
}

static int ibgda_create_dct_shared_objects(nvshmemt_ibgda_state_t *ibgda_state,
                                           struct ibgda_device *device, int portid) {
    int status = 0;

    const struct ibv_port_attr *port_attr = device->port_attr + (portid - 1);
    struct ibv_context *context = device->context;

    struct ibv_pd *pd = NULL;
    struct ibv_parent_domain_init_attr pd_init_attr;

    struct ibv_srq *srq = NULL;
    struct ibv_srq_init_attr srq_init_attr;

    struct ibv_cq *send_cq = NULL;
    struct ibv_cq *recv_cq = NULL;

    struct ibv_ah *ah = NULL;
    struct mlx5dv_ah dah;
    struct ibv_ah_attr ah_attr;
    struct mlx5dv_obj dv;

    bool support_half_av_seg;
    int hca_support_compact_address_vector;

    uint8_t cmd_cap_in[DEVX_ST_SZ_BYTES(query_hca_cap_in)] = {
        0,
    };
    uint8_t cmd_cap_out[DEVX_ST_SZ_BYTES(query_hca_cap_out)] = {
        0,
    };
    void *cap;

    DEVX_SET(query_hca_cap_in, cmd_cap_in, opcode, MLX5_CMD_OP_QUERY_HCA_CAP);
    DEVX_SET(
        query_hca_cap_in, cmd_cap_in, op_mod,
        MLX5_SET_HCA_CAP_OP_MOD_GENERAL_DEVICE | (MLX5_CAP_GENERAL << 1) | HCA_CAP_OPMOD_GET_CUR);

    status = mlx5dv_devx_general_cmd(context, cmd_cap_in, sizeof(cmd_cap_in), cmd_cap_out,
                                     sizeof(cmd_cap_out));
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "mlx5dv_devx_general_cmd for hca cap failed.\n");

    cap = DEVX_ADDR_OF(query_hca_cap_out, cmd_cap_out, capability.cmd_hca_cap);
    hca_support_compact_address_vector = DEVX_GET(cmd_hca_cap, cap, compact_address_vector);

    memset(&pd_init_attr, 0, sizeof(pd_init_attr));
    memset(&srq_init_attr, 0, sizeof(srq_init_attr));

    pd_init_attr.pd = device->pd;
    pd = ibv_alloc_parent_domain(context, &pd_init_attr);
    NVSHMEMI_NULL_ERROR_JMP(pd, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "ibv_alloc_parent_domain failed.\n");

    srq_init_attr.attr.max_wr = ibgda_srq_depth;
    srq_init_attr.attr.max_sge = 1;

    srq = ftable.create_srq(pd, &srq_init_attr);
    NVSHMEMI_NULL_ERROR_JMP(srq, status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_create_srq failed.\n");

    send_cq = ftable.create_cq(context, ibgda_srq_depth, NULL, NULL, 0);
    NVSHMEMI_NULL_ERROR_JMP(send_cq, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "ibv_create_cq for send_cq failed.\n");

    recv_cq = ftable.create_cq(context, ibgda_srq_depth, NULL, NULL, 0);
    NVSHMEMI_NULL_ERROR_JMP(recv_cq, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "ibv_create_cq for recv_cq failed.\n");

    if (port_attr->lid == 0) {
        ib_get_gid_index(&ftable, device->context, portid, port_attr->gid_tbl_len,
                         (int *)&device->gid_info[portid - 1].local_gid_index,
                         ibgda_state->log_level, ibgda_state->options);
        ftable.query_gid(device->context, portid, device->gid_info[portid - 1].local_gid_index,
                         (ibv_gid *)&device->gid_info[portid - 1].local_gid);
        ah_attr.is_global = 1;
        ah_attr.grh.dgid.global.subnet_prefix =
            device->gid_info[portid - 1].local_gid.global.subnet_prefix;
        ah_attr.grh.dgid.global.interface_id =
            device->gid_info[portid - 1].local_gid.global.interface_id;
        ah_attr.grh.flow_label = 0;
        ah_attr.grh.sgid_index = device->gid_info[portid - 1].local_gid_index;
        ah_attr.grh.traffic_class = ibgda_state->options->IB_TRAFFIC_CLASS;
        ah_attr.grh.hop_limit = IBGDA_GRH_HOP_LIMIT;
        support_half_av_seg = false;
    } else {
        // Only IB supports is_global = 0.
        assert(port_attr->link_layer == IBV_LINK_LAYER_INFINIBAND);
        ah_attr.dlid = port_attr->lid;
        ah_attr.is_global = 0;
        support_half_av_seg = hca_support_compact_address_vector;
    }
    ah_attr.sl = ibgda_state->options->IB_SL;
    ah_attr.src_path_bits = 0;
    ah_attr.port_num = portid;

    ah = ftable.create_ah(device->pd, &ah_attr);
    NVSHMEMI_NULL_ERROR_JMP(ah, status, NVSHMEMX_ERROR_INTERNAL, out, "Unable to create ah.\n");

    dv.ah.in = ah;
    dv.ah.out = &dah;
    mlx5dv_init_obj(&dv, MLX5DV_OBJ_AH);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "mlx5dv AH initialization failed.\n");

    device->dct.pd = pd;
    device->dct.srq = srq;
    device->dct.send_cq = send_cq;
    device->dct.recv_cq = recv_cq;
    device->dct.ah = ah;
    memcpy(&device->dct.dah, &dah, sizeof(dah));
    memcpy(&device->dct.ah_attr, &ah_attr, sizeof(ah_attr));
    device->support_half_av_seg = support_half_av_seg;

out:
    if (status) {
        if (recv_cq) ftable.destroy_cq(recv_cq);
        if (send_cq) ftable.destroy_cq(send_cq);
        if (srq) ftable.destroy_srq(srq);
    }
    return status;
}

static int ibgda_create_dct(nvshmemt_ibgda_state_t *ibgda_state, struct ibgda_ep **ep_ptr,
                            const struct ibgda_device *device, int portid) {
    int status = 0;

    struct ibgda_ep *ep = NULL;
    struct ibv_qp *ib_qp = NULL;

    struct ibv_qp_init_attr_ex ib_qp_attr_ex;
    struct mlx5dv_qp_init_attr dv_init_attr;
    struct ibv_qp_attr ib_qp_attr;

    const struct ibv_port_attr *port_attr = device->port_attr + (portid - 1);

    memset(&ib_qp_attr_ex, 0, sizeof(ib_qp_attr_ex));
    memset(&dv_init_attr, 0, sizeof(dv_init_attr));

    ep = (struct ibgda_ep *)calloc(1, sizeof(struct ibgda_ep));
    NVSHMEMI_NULL_ERROR_JMP(ep, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate mem for ep.\n");

    dv_init_attr.comp_mask = MLX5DV_QP_INIT_ATTR_MASK_DC;
    dv_init_attr.dc_init_attr.dc_type = MLX5DV_DCTYPE_DCT;
    dv_init_attr.dc_init_attr.dct_access_key = IBGDA_DC_ACCESS_KEY;

    ib_qp_attr_ex.pd = device->dct.pd;
    ib_qp_attr_ex.comp_mask = IBV_QP_INIT_ATTR_PD;
    ib_qp_attr_ex.qp_type = IBV_QPT_DRIVER;
    ib_qp_attr_ex.srq = device->dct.srq;
    ib_qp_attr_ex.send_cq = device->dct.send_cq;
    ib_qp_attr_ex.recv_cq = device->dct.recv_cq;

    ib_qp_attr_ex.cap.max_send_wr = ibgda_state->options->QP_DEPTH;
    ib_qp_attr_ex.cap.max_recv_wr = ibgda_state->options->QP_DEPTH;
    ib_qp_attr_ex.cap.max_send_sge = 1;
    ib_qp_attr_ex.cap.max_recv_sge = 1;
    ib_qp_attr_ex.cap.max_inline_data = NVSHMEMI_IBGDA_MAX_INLINE_SIZE;

    ib_qp = mlx5dv_create_qp(device->context, &ib_qp_attr_ex, &dv_init_attr);
    NVSHMEMI_NULL_ERROR_JMP(ib_qp, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "mlx5dv_create_qp failed.\n");

    // RST2INIT
    memset(&ib_qp_attr, 0, sizeof(ib_qp_attr));
    ib_qp_attr.qp_state = IBV_QPS_INIT;
    ib_qp_attr.pkey_index = 0;
    ib_qp_attr.port_num = portid;
    ib_qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                                 IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_ATOMIC;

    status = ftable.modify_qp(ib_qp, &ib_qp_attr,
                              IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "ibv_modify_qp rst2init for dct failed.\n");

    // INIT2RTR
    memset(&ib_qp_attr, 0, sizeof(ib_qp_attr));
    ib_qp_attr.qp_state = IBV_QPS_RTR;
    ib_qp_attr.path_mtu = port_attr->active_mtu;
    ib_qp_attr.min_rnr_timer = 12;
    memcpy(&ib_qp_attr.ah_attr, &device->dct.ah_attr, sizeof(ib_qp_attr.ah_attr));

    status = ftable.modify_qp(ib_qp, &ib_qp_attr,
                              IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_MIN_RNR_TIMER);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "ibv_modify_qp init2rtr for dct failed.\n");

    ep->qp_type = NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCT;

    ep->ib_qp = ib_qp;
    ep->qpn = ib_qp->qp_num;
    ep->portid = portid;

    *ep_ptr = ep;

out:
    if (status) {
        if (ib_qp) {
            int _status = ftable.destroy_qp(ib_qp);
            if (_status) NVSHMEMI_ERROR_PRINT("ibv_destroy_qp for dct failed.\n");
        }
        if (ep) free(ep);
    }
    return status;
}

static int ibgda_get_dct_handle(struct ibgda_dct_handle *dct_handle, const struct ibgda_ep *ep,
                                const struct ibgda_device *device) {
    assert(ep->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCT);

    memcpy(&dct_handle->dev_dct, device->dct.dah.av, sizeof(dct_handle->dev_dct));
    // Don't do htobe32 here as we need to determine whether the ext field should be set or not.
    dct_handle->dev_dct.dqp_dct = ep->qpn;
    dct_handle->dev_dct.key.dc_key = htobe64(IBGDA_DC_ACCESS_KEY);
    dct_handle->support_half_av_seg = device->support_half_av_seg;

    return 0;
}

static int ibgda_destroy_ep(struct ibgda_ep *ep) {
    int status = 0;

    if (!ep) return status;
    if (ep->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCT) {
        if (ep->ib_qp) {
            int status = ftable.destroy_qp(ep->ib_qp);
            if (status) NVSHMEMI_ERROR_PRINT("ibv_destroy_qp failed.\n");
        }
    } else {
        if (ep->devx_qp) {
            mlx5dv_devx_obj_destroy(ep->devx_qp);
        }

        if (ep->uar_mobject) {
            ibgda_unmap_and_free_qp_uar(ep->uar_mobject);
        }
    }

    if (ep->send_cq) {
        ibgda_destroy_cq(ep->send_cq);
    }

    if (ep->ah) {
        ftable.destroy_ah(ep->ah);
    }

    free(ep);

    return status;
}

static void ibgda_get_device_qp_mvars(nvshmemi_ibgda_device_qp_management_t *dev_mvars,
                                      struct ibgda_device *device, const struct ibgda_ep *ep) {
    memset(dev_mvars, 0, sizeof(*dev_mvars));
}

static void ibgda_get_device_qp(nvshmemi_ibgda_device_qp_t *dev_qp, struct ibgda_device *device,
                                const struct ibgda_ep *ep, int selected_dev_idx) {
    uintptr_t ibuf_dci_start;
    uintptr_t ibuf_rc_start;
    void *ibuf_ptr = NULL;

    size_t size_per_dci =
        NVSHMEMI_IBGDA_IBUF_SLOT_SIZE * (ibgda_num_fetch_slots_per_dci + IBGDA_IBUF_RESERVED_SLOTS);
    size_t size_per_rc =
        NVSHMEMI_IBGDA_IBUF_SLOT_SIZE * (ibgda_num_fetch_slots_per_rc + IBGDA_IBUF_RESERVED_SLOTS);

    assert(ep->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI ||
           ep->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_RC);

    dev_qp->qpn = ep->qpn;

    assert(ep->wq_mobject->has_gpu_mapping);
    dev_qp->tx_wq.wqe = (void *)((uintptr_t)ep->wq_mobject->aligned.gpu_ptr + ep->wq_offset);

    if (ibgda_nic_handler == IBGDA_NIC_HANDLER_GPU) {
        assert(ep->dbr_mobject->has_gpu_mapping);
        dev_qp->tx_wq.dbrec = (__be32 *)((uintptr_t)ep->dbr_mobject->aligned.gpu_ptr +
                                         ep->dbr_offset + sizeof(__be32));

        assert(ep->uar_mobject->has_gpu_mapping);
        dev_qp->tx_wq.bf = (void *)ep->uar_mobject->aligned.gpu_ptr;
    }

    dev_qp->tx_wq.nwqes = ep->sq_cnt;

    ibuf_dci_start = (uintptr_t)device->qp_shared_object.internal_buf.mem_object->aligned.gpu_ptr;
    ibuf_rc_start = ibuf_dci_start + (size_per_dci * device->dci.num_eps);

    if (ep->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI) {
        ibuf_ptr = (void *)(ibuf_dci_start + (size_per_dci * ep->user_index));
        dev_qp->ibuf.nslots = ibgda_num_fetch_slots_per_dci;
    } else if (ep->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_RC) {
        ibuf_ptr = (void *)(ibuf_rc_start + (size_per_rc * ep->user_index));
        dev_qp->ibuf.nslots = ibgda_num_fetch_slots_per_rc;
    }

    dev_qp->ibuf.lkey = htobe32(device->qp_shared_object.internal_buf.mem_handle->lkey);
    dev_qp->ibuf.rkey = htobe32(device->qp_shared_object.internal_buf.mem_handle->rkey);
    dev_qp->ibuf.buf = ibuf_ptr;

    dev_qp->qp_type = ep->qp_type;
    dev_qp->dev_idx = selected_dev_idx;

    ibgda_get_device_qp_mvars(&dev_qp->mvars, device, ep);
}

static void ibgda_get_device_dct(nvshmemi_ibgda_device_dct_t *dev_dct,
                                 const struct ibgda_dct_handle *dct_handle,
                                 const struct ibgda_device *device) {
    memcpy(dev_dct, &dct_handle->dev_dct, sizeof(*dev_dct));
    dev_dct->dqp_dct =
        htobe32(((device->support_half_av_seg ? 0ULL : 1ULL) << 31) | dev_dct->dqp_dct);
}

static int ibgda_setup_gpu_state(nvshmem_transport_t t) {
    nvshmemt_ibgda_state_t *ibgda_state;
    ibgda_state = (nvshmemt_ibgda_state_t *)t->state;

    nvshmemi_ibgda_device_state_t *ibgda_device_state_h;
    ibgda_device_state_h = (nvshmemi_ibgda_device_state_t *)t->type_specific_shared_state;

    nvshmemi_ibgda_device_dct_t *dct_d = NULL;
    nvshmemi_ibgda_device_dct_t *dct_h = NULL;

    nvshmemi_ibgda_device_qp_t *dci_d = NULL;
    nvshmemi_ibgda_device_qp_t *dci_h = NULL;

    nvshmemi_ibgda_device_qp_t *rc_d = NULL;
    nvshmemi_ibgda_device_qp_t *rc_h = NULL;

    nvshmemi_ibgda_device_cq_t *cq_d = NULL;
    nvshmemi_ibgda_device_cq_t *cq_h = NULL;

    uint8_t *qp_group_switches_d = NULL;

    const size_t mvars_offset = offsetof(nvshmemi_ibgda_device_qp_t, mvars);
    const size_t prod_idx_offset = offsetof(nvshmemi_ibgda_device_qp_management_t, tx_wq.prod_idx);
    const size_t cons_t_offset = offsetof(nvshmemi_ibgda_device_qp_management_t, tx_wq.cons_idx);
    const size_t wqe_h_offset = offsetof(nvshmemi_ibgda_device_qp_management_t, tx_wq.resv_head);
    const size_t wqe_t_offset = offsetof(nvshmemi_ibgda_device_qp_management_t, tx_wq.ready_head);

    nvshmemi_ibgda_device_qp_map_type_t rc_map_type = NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_INVALID;
    nvshmemi_ibgda_device_qp_map_type_t dc_map_type = NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_INVALID;

    int n_pes = t->n_pes;
    int mype = t->my_pe;
    int n_devs_selected = ibgda_state->n_devs_selected;
    int num_qp_groups = 0;
    int num_dct_handles = 0;
    int num_dct_cache_handles = 0;
    int num_rc_handles = 0;
    int num_cq_handles = 0;
    int num_dci_handles = 0;
    int num_shared_dci_handles = 0;
    int status = 0;
    int cq_idx = 0;
    bool skip_cst = true;
    bool support_half_av_seg = true;

    int num_elements;

    assert(ibgda_device_state_h != 0);
    memset(ibgda_device_state_h, 0, sizeof(*ibgda_device_state_h));

    /* calculate buffer sizes and constants start */
    for (int j = 0; j < n_devs_selected; j++) {
        struct ibgda_device *device;
        int dev_idx;
        dev_idx = ibgda_state->selected_dev_ids[j];
        device = (struct ibgda_device *)ibgda_state->devices + dev_idx;
        dc_map_type = device->dci.map_by;
        rc_map_type = device->rc.map_by;
        skip_cst &= device->may_skip_cst;
        support_half_av_seg &= device->support_half_av_seg;
        num_dct_handles += device->dct.num_eps * n_pes;
        num_dci_handles += device->dci.num_eps;
        num_rc_handles += device->rc.num_eps_per_pe * n_pes;
        num_cq_handles += device->dci.num_eps + (device->rc.num_eps_per_pe * (n_pes - 1));
        num_shared_dci_handles += device->dci.num_shared_eps;
    }
    num_elements = num_dct_handles - NVSHMEMI_IBGDA_MAX_CONST_DCTS;
    assert(num_dci_handles - num_shared_dci_handles >= 0);

    if (num_rc_handles > 0) {
        num_qp_groups = num_rc_handles / n_devs_selected / n_pes;
    } else {
        num_qp_groups = num_dci_handles / n_devs_selected;
    }
    /* calculate buffer sizes and constants end */

    /* allocate host memory for dct, rc, cq, dci start */
    dct_h = (nvshmemi_ibgda_device_dct_t *)calloc(num_dct_handles, sizeof(*dct_h));
    NVSHMEMI_NULL_ERROR_JMP(dct_h, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "dct calloc err.");

    dci_h = (nvshmemi_ibgda_device_qp_t *)calloc(num_dci_handles, sizeof(*dci_h));
    NVSHMEMI_NULL_ERROR_JMP(dci_h, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "dci calloc err.");
    for (int i = 0; i < num_dci_handles; i++) {
        nvshmemi_init_ibgda_device_qp(dci_h[i]);
    }

    if (num_rc_handles > 0) {
        rc_h = (nvshmemi_ibgda_device_qp_t *)calloc(num_rc_handles, sizeof(*rc_h));
        NVSHMEMI_NULL_ERROR_JMP(rc_h, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "rc calloc err.");
        for (int i = 0; i < num_dci_handles; i++) {
            nvshmemi_init_ibgda_device_qp(dci_h[i]);
        }
    }

    cq_h = (nvshmemi_ibgda_device_cq_t *)calloc(num_cq_handles, sizeof(*cq_h));
    NVSHMEMI_NULL_ERROR_JMP(cq_h, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "cq calloc err.");
    for (int i = 0; i < num_cq_handles; i++) {
        nvshmemi_init_ibgda_device_cq(cq_h[i]);
    }
    /* allocate host memory for dct, rc, cq, dci end */

    /* allocate device memory for dct, rc, cq, dci start */
    if (num_dct_handles > NVSHMEMI_IBGDA_MAX_CONST_DCTS) {
        num_dct_cache_handles = num_dct_handles - NVSHMEMI_IBGDA_MAX_CONST_DCTS;
        status = cudaMalloc(&dct_d, num_elements * sizeof(*dct_d));
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                              "dct_d allocation failed.\n");
    }

    status = cudaMalloc(&dci_d, num_dci_handles * sizeof(*dci_d));
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "dci cudaM err.");

    if (num_rc_handles > 0) {
        status = cudaMalloc(&rc_d, num_rc_handles * sizeof(*rc_d));
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                              "rc_d cudaM err.\n");
    }

    status = cudaMalloc(&cq_d, num_cq_handles * sizeof(*cq_d));
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "cq cudaM err.");

    status = cudaMalloc(&qp_group_switches_d, num_qp_groups * sizeof(*qp_group_switches_d));
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                          "qp_group_switches_d cudaM err.");
    /* allocate device memory for dct, rc, cq, dci end */

    /* Get and store information for dct, rc, cq, dci start */
    for (int i = 0; i < num_dct_handles / n_devs_selected; ++i) {
        for (int j = 0; j < n_devs_selected; j++) {
            int arr_idx = i * n_devs_selected + j;
            int dev_idx = ibgda_state->selected_dev_ids[j];
            struct ibgda_device *device = (struct ibgda_device *)ibgda_state->devices + dev_idx;
            ibgda_get_device_dct(&dct_h[arr_idx], &device->dct.dct_handles[i], device);
        }
    }

    for (int i = 0; i < num_dci_handles / n_devs_selected; ++i) {
        for (int j = 0; j < n_devs_selected; j++) {
            int arr_idx = i * n_devs_selected + j;
            int dev_idx = ibgda_state->selected_dev_ids[j];
            struct ibgda_device *device = (struct ibgda_device *)ibgda_state->devices + dev_idx;
            uintptr_t base_mvars_d_addr = (uintptr_t)(&dci_d[arr_idx]) + mvars_offset;

            ibgda_get_device_qp(&dci_h[arr_idx], device, device->dci.eps[i], j);
            dci_h[arr_idx].tx_wq.cq = &cq_d[cq_idx];

            ibgda_get_device_cq(&cq_h[cq_idx], device->dci.eps[i]->send_cq);
            cq_h[cq_idx].cons_idx = (uint64_t *)(base_mvars_d_addr + cons_t_offset);
            cq_h[cq_idx].resv_head = (uint64_t *)(base_mvars_d_addr + wqe_h_offset);
            cq_h[cq_idx].ready_head = (uint64_t *)(base_mvars_d_addr + wqe_t_offset);
            cq_h[cq_idx].qpn = dci_h[arr_idx].qpn;
            cq_h[cq_idx].qp_type = dci_h[arr_idx].qp_type;

            if (ibgda_nic_handler == IBGDA_NIC_HANDLER_GPU) {
                dci_h[arr_idx].tx_wq.prod_idx = (uint64_t *)(base_mvars_d_addr + prod_idx_offset);
                cq_h[cq_idx].prod_idx = (uint64_t *)(base_mvars_d_addr + prod_idx_offset);
            } else {
                dci_h[arr_idx].tx_wq.prod_idx =
                    &((uint64_t *)device->qp_shared_object.prod_idx_mobject->aligned.gpu_ptr)[i];
                cq_h[cq_idx].prod_idx = dci_h[arr_idx].tx_wq.prod_idx;
            }

            ++cq_idx;
        }
    }

    if (num_rc_handles > 0) {
        for (int i = 0; i < num_rc_handles / n_devs_selected; ++i) {
            int arr_offset = i * n_devs_selected;
            /* No RC QP to self */
            if ((i / (num_rc_handles / n_devs_selected / n_pes)) == mype) {
                continue;
            }
            for (int j = 0; j < n_devs_selected; j++) {
                int arr_idx = arr_offset + j;
                int dev_idx = ibgda_state->selected_dev_ids[j];
                struct ibgda_device *device = (struct ibgda_device *)ibgda_state->devices + dev_idx;
                uintptr_t base_mvars_d_addr = (uintptr_t)(&rc_d[arr_idx]) + mvars_offset;

                ibgda_get_device_qp(&rc_h[arr_idx], device, device->rc.eps[i], j);

                rc_h[arr_idx].tx_wq.cq = &cq_d[cq_idx];

                ibgda_get_device_cq(&cq_h[cq_idx], device->rc.eps[i]->send_cq);
                cq_h[cq_idx].cons_idx = (uint64_t *)(base_mvars_d_addr + cons_t_offset);
                cq_h[cq_idx].resv_head = (uint64_t *)(base_mvars_d_addr + wqe_h_offset);
                cq_h[cq_idx].ready_head = (uint64_t *)(base_mvars_d_addr + wqe_t_offset);
                cq_h[cq_idx].qpn = rc_h[arr_idx].qpn;
                cq_h[cq_idx].qp_type = rc_h[arr_idx].qp_type;

                if (ibgda_nic_handler == IBGDA_NIC_HANDLER_GPU) {
                    rc_h[arr_idx].tx_wq.prod_idx =
                        (uint64_t *)(base_mvars_d_addr + prod_idx_offset);
                    cq_h[cq_idx].prod_idx = (uint64_t *)(base_mvars_d_addr + prod_idx_offset);
                } else {
                    rc_h[arr_idx].tx_wq.prod_idx =
                        &((uint64_t *)device->qp_shared_object.prod_idx_mobject->aligned
                              .gpu_ptr)[device->dci.num_eps + i];
                    cq_h[cq_idx].prod_idx = rc_h[arr_idx].tx_wq.prod_idx;
                }

                ++cq_idx;
            }
        }
    }
    cudaMemsetAsync(qp_group_switches_d, 0, num_qp_groups * sizeof(*qp_group_switches_d),
                    ibgda_state->my_stream);
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                          "qp_group_switches_d set err.");
    /* Get and store information for dct, rc, cq, dci end */

    /* Cache DCTs in constant memory start */
    memcpy(ibgda_device_state_h->constmem.dcts, dct_h,
           sizeof(*dct_h) * IBGDA_MIN(num_dct_handles, NVSHMEMI_IBGDA_MAX_CONST_DCTS));
    /* Cache DCTs in constant memory end */

    /* Copy host side structs to device side structs start */
    /* Add the rest of DCTs to global memory */
    if (num_dct_cache_handles > NVSHMEMI_IBGDA_MAX_CONST_DCTS) {
        const void *dct_h_ptr = (const void *)&dct_h[NVSHMEMI_IBGDA_MAX_CONST_DCTS];
        status = cudaMemcpyAsync(dct_d, dct_h_ptr, sizeof(*dct_d) * num_elements,
                                 cudaMemcpyHostToDevice, ibgda_state->my_stream);
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out, "dct copy err.");
    }

    status = cudaMemcpyAsync(dci_d, (const void *)dci_h, sizeof(*dci_h) * num_dci_handles,
                             cudaMemcpyHostToDevice, ibgda_state->my_stream);
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out, "dci copy err.");

    if (num_rc_handles > 0) {
        status = cudaMemcpyAsync(rc_d, (const void *)rc_h, sizeof(*rc_h) * num_rc_handles,
                                 cudaMemcpyHostToDevice, ibgda_state->my_stream);
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out, "rc copy err.");
    }

    status = cudaMemcpyAsync(cq_d, (const void *)cq_h, sizeof(*cq_h) * num_cq_handles,
                             cudaMemcpyHostToDevice, ibgda_state->my_stream);
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out, "cq copy err.");
    /* Copy host side structs to device side structs end */

    /* Post the device state start */
    ibgda_device_state_h->globalmem.qp_group_switches = qp_group_switches_d;
    ibgda_device_state_h->globalmem.dcis = dci_d;
    ibgda_device_state_h->globalmem.rcs = rc_d;
    ibgda_device_state_h->globalmem.dcts = dct_d;
    ibgda_device_state_h->globalmem.cqs = cq_d;

    ibgda_device_state_h->num_qp_groups = num_qp_groups;
    ibgda_device_state_h->log2_cumem_granularity = t->log2_cumem_granularity;
    ibgda_device_state_h->num_shared_dcis = num_shared_dci_handles;
    ibgda_device_state_h->num_exclusive_dcis = num_dci_handles - num_shared_dci_handles;
    ibgda_device_state_h->dci_map_type = dc_map_type;
    ibgda_device_state_h->ndcts_per_pe = num_dct_handles / n_devs_selected / n_pes;
    ibgda_device_state_h->num_dct_groups = IBGDA_MAX(
        ibgda_device_state_h->num_exclusive_dcis / (num_dct_handles / n_devs_selected), 1);
    ibgda_device_state_h->num_rc_per_pe = num_rc_handles / n_devs_selected / n_pes;
    ibgda_device_state_h->rc_map_type = rc_map_type;
    ibgda_device_state_h->num_requests_in_batch = ibgda_num_requests_in_batch;
    ibgda_device_state_h->support_half_av_seg = support_half_av_seg;
    ibgda_device_state_h->may_skip_cst = skip_cst;
    ibgda_device_state_h->use_async_postsend = (ibgda_nic_handler == IBGDA_NIC_HANDLER_CPU);
    ibgda_device_state_h->num_devices_initialized = n_devs_selected;
    assert(ibgda_nic_buf_location == IBGDA_MEM_TYPE_GPU ||
           ibgda_nic_buf_location == IBGDA_MEM_TYPE_HOST);
    ibgda_device_state_h->nic_buf_on_gpumem = (ibgda_nic_buf_location == IBGDA_MEM_TYPE_GPU);
    status = cudaStreamSynchronize(ibgda_state->my_stream);
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out, "stream sync err.");
    /* Post the device state start */

out:
    if (status) {
        if (dci_d) cudaFree(dci_d);
        if (dct_d) cudaFree(dct_d);
        if (cq_d) cudaFree(cq_d);
        if (rc_d) cudaFree(rc_d);
        if (qp_group_switches_d) cudaFree(qp_group_switches_d);
    }
    if (dci_h) free(dci_h);
    if (dct_h) free(dct_h);
    if (cq_h) free(cq_h);
    if (rc_h) free(rc_h);
    return status;
}

static bool ibgda_cst_is_required(struct ibgda_device *device, CUdevice dev_id) {
    bool rval = true;

    int order = 0;
    if (CUPFN(ibgda_cuda_syms,
              cuDeviceGetAttribute(
                  &order, (CUdevice_attribute)CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING,
                  dev_id))) {
        NVSHMEMI_WARN_PRINT("Cannot query dev attr. Assuming no GDR write ordering\n");
    } else {
        // GPU guarantees incoming PCIe write ordering. No need to do CST.
        if (order >= CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER) rval = false;
    }

    return rval;
}

int nvshmemt_ibgda_connect_endpoints(nvshmem_transport_t t, int *selected_dev_ids,
                                     int num_selected_devs) {
    /* global state start */
    nvshmemt_ibgda_state_t *ibgda_state = (nvshmemt_ibgda_state_t *)t->state;
    struct nvshmemi_options_s *options = ibgda_state->options;
    int status = 0;
    /* global state end */

    if (!options->IBGDA_ENABLE_MULTI_PORT && num_selected_devs > 1) {
        INFO(ibgda_state->log_level,
             "Multi-port for IBGDA is disabled by the env. Using 1 device instead "
             "of %d.",
             num_selected_devs);
        num_selected_devs = 1;
    }

    if (num_selected_devs > NVSHMEMI_IBGDA_MAX_DEVICES_PER_PE) {
        NVSHMEMI_WARN_PRINT("IBGDA only supports %d devices, but the lib has requested %d.\n",
                            NVSHMEMI_IBGDA_MAX_DEVICES_PER_PE, num_selected_devs);
        num_selected_devs = NVSHMEMI_IBGDA_MAX_DEVICES_PER_PE;
        NVSHMEMI_WARN_PRINT("Using %d devices.\n", num_selected_devs);
    }
    /* Constants for resource creation start */
    int mype = t->my_pe;
    int n_pes = t->n_pes;
    int num_dct_eps = options->IBGDA_NUM_DCT;
    int num_dci_eps = options->IBGDA_NUM_DCI;
    int num_shared_dci_eps = options->IBGDA_NUM_SHARED_DCI;
    int num_rc_eps_per_pe = options->IBGDA_NUM_RC_PER_PE;
    int num_rc_eps = num_rc_eps_per_pe * n_pes;
    /* constants for resource creation end */

    /* loop variables start */
    struct ibgda_dct_handle *local_dct_handles = NULL;
    struct ibgda_rc_handle *local_rc_handles = NULL;
    struct ibgda_device *device = NULL;
    int curr_dev_id = 0;
    int init_dev_cnt = 0;
    int portid = 0;
    /* loop variables end */

    /* cuda info start */
    CUdevice gpu_device_id;
    int mtpb;
    int mpc;
    int warp_size;
    /* cuda info end */

    /* shared dev info start */
    nvshmemi_ibgda_device_qp_map_type_t rc_map_type = NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_INVALID;
    nvshmemi_ibgda_device_qp_map_type_t dc_map_type = NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_INVALID;
    bool support_half_av_seg = true;
    bool skip_cst = true;
    /* shared dev info end */

    if (ibgda_state->selected_dev_ids) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out_already_connected,
                           "Device already selected. IBGDA only supports"
                           " one initialization per PE.\n");
    }

    /* Get CUDA information start */
    if (CUPFN(ibgda_cuda_syms, cuCtxGetDevice(&gpu_device_id))) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cuCtxGetDevice failed.\n");
    }

    if (cudaDeviceGetAttribute(&warp_size, cudaDevAttrWarpSize, gpu_device_id)) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "querying warp size failed.");
    }

    if (cudaDeviceGetAttribute(&mtpb, cudaDevAttrMaxThreadsPerBlock, gpu_device_id)) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                           "query max threads per block fail.");
    }

    if (cudaDeviceGetAttribute(&mpc, cudaDevAttrMultiProcessorCount, gpu_device_id)) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "query mpc count fail.");
    }
    /* Get CUDA information end */

    /* Get shared dev info start */
    status = ibgda_parse_qp_map_by(&rc_map_type, options->IBGDA_RC_MAP_BY);
    NVSHMEMI_NZ_ERROR_JMP(status, status, out, "IBGDA_RC_MAP_BY is not valid.");
    INFO(ibgda_state->log_level, "IBGDA_RC_MAP_BY is set to %s.", options->IBGDA_RC_MAP_BY);

    status = ibgda_parse_qp_map_by(&dc_map_type, ibgda_state->options->IBGDA_DCI_MAP_BY);
    NVSHMEMI_NZ_ERROR_JMP(status, status, out, "IBGDA_DCI_MAP_BY is not valid.");
    INFO(ibgda_state->log_level, "IBGDA_DCI_MAP_BY is set to %s.", options->IBGDA_DCI_MAP_BY);
    /* Get shared dev info end */

    /* Allocate global structs start */
    ibgda_state->selected_dev_ids = (int *)calloc(num_selected_devs, sizeof(*selected_dev_ids));
    NVSHMEMI_NULL_ERROR_JMP(ibgda_state->selected_dev_ids, status, NVSHMEMX_ERROR_OUT_OF_MEMORY,
                            out, "allocation of selected_device_ids failed.\n");

    local_dct_handles = (struct ibgda_dct_handle *)calloc(num_dct_eps, sizeof(*local_dct_handles));
    NVSHMEMI_NULL_ERROR_JMP(local_dct_handles, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "allocation of local_dct_handles failed.\n");

    local_rc_handles = (struct ibgda_rc_handle *)calloc(num_rc_eps, sizeof(*local_rc_handles));
    NVSHMEMI_NULL_ERROR_JMP(local_rc_handles, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "allocation of local_rc_handles failed.\n");
    /* Allocate global structs end */

    /* recalculate mappings for QP types start */
    if (ibgda_num_fetch_slots_per_dci < warp_size) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "NVSHMEM_IBGDA_NUM_FETCH_SLOTS_PER_DCI must be at least %d.\n",
                           warp_size);
    }

    if (num_dci_eps <= 0) {
        switch (dc_map_type) {
            case NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_CTA:
            case NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_SM:
                num_dci_eps = mpc;
                break;
            case NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_WARP:
                num_dci_eps = mpc * warp_size;
                break;
            case NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_DCT:
                num_dci_eps = num_dct_eps * n_pes;
                break;
            default:
                NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                                   "NVSHMEM_IBGDA_DCI_MAP_BY=%s is not supported.\n",
                                   ibgda_state->options->IBGDA_DCI_MAP_BY);
                break;
        }
        num_dci_eps = num_dci_eps + num_shared_dci_eps;
    }
    assert(num_dci_eps > 0);
    if (num_rc_eps_per_pe < 0) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "NVSHMEM_IBGDA_NUM_RC_PER_PE must be positive or zero.\n");
    } else if (num_rc_eps_per_pe > 0) {
        if (ibgda_num_fetch_slots_per_rc < warp_size) {
            NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                               "NVSHMEM_IBGDA_NUM_FETCH_SLOTS_PER_RC must be at least %d.\n",
                               warp_size);
        }

        switch (rc_map_type) {
            case NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_CTA:
            case NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_SM:
            case NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_WARP:
                break;
            default:
                NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                                   "NVSHMEM_IBGDA_RC_MAP_BY=%s is not supported.\n",
                                   ibgda_state->options->IBGDA_RC_MAP_BY);
                break;
        }
    }
    /* recalculate mappings for QP types end */

    /* Check configured arguments start */
    if ((num_dct_eps * num_selected_devs) < 2) {
        NVSHMEMI_WARN_PRINT("NVSHMEM_IBGDA_NUM_DCT < 2 and may impact performance.");
    }

    if (num_shared_dci_eps > num_dci_eps) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "IBGDA_NUM_SHARED_DCI > IBGDA_NUM_DCI.");
    }
    /* check configured args stop */

    for (int i = 0; i < num_selected_devs; i++) {
        if (selected_dev_ids[i] < 0 || selected_dev_ids[i] >= ibgda_state->n_dev_ids) {
            NVSHMEMI_ERROR_PRINT("Invalid device ID %d.\n", selected_dev_ids[i]);
            if (i > 0) {
                goto out_already_connected;
            } else {
                goto out;
            }
        }
        curr_dev_id = ibgda_state->dev_ids[selected_dev_ids[i]];

        /* set device info start */
        device = ((struct ibgda_device *)ibgda_state->devices + curr_dev_id);
        portid = ibgda_state->port_ids[selected_dev_ids[i]];
        skip_cst &= (!ibgda_cst_is_required(device, gpu_device_id));
        device->dci.map_by = dc_map_type;
        device->rc.map_by = rc_map_type;
        device->dct.num_eps = num_dct_eps;
        device->dci.num_eps = num_dci_eps;
        device->dci.num_shared_eps = num_shared_dci_eps;
        device->rc.num_eps_per_pe = num_rc_eps_per_pe;
        /* set device info end */

        /* allocate device structs start */
        device->dct.dct_handles = (struct ibgda_dct_handle *)calloc(
            device->dct.num_eps * n_pes, sizeof(*device->dct.dct_handles));
        NVSHMEMI_NULL_ERROR_JMP(device->dct.dct_handles, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                "allocation of dct_handles failed.");

        device->dct.eps = (struct ibgda_ep **)calloc(device->dct.num_eps, sizeof(*device->dct.eps));
        NVSHMEMI_NULL_ERROR_JMP(device->dct.eps, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                "allocation of dct.eps failed.");

        device->dci.eps = (struct ibgda_ep **)calloc(device->dci.num_eps, sizeof(*device->dci.eps));
        NVSHMEMI_NULL_ERROR_JMP(device->dci.eps, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                "allocation of dci.eps failed.");

        device->rc.peer_ep_handles =
            (struct ibgda_rc_handle *)calloc(num_rc_eps, sizeof(*device->rc.peer_ep_handles));
        NVSHMEMI_NULL_ERROR_JMP(device->rc.peer_ep_handles, status, NVSHMEMX_ERROR_OUT_OF_MEMORY,
                                out, "allocation of rc.peer_ep_handles failed.");

        device->rc.eps = (struct ibgda_ep **)calloc(num_rc_eps, sizeof(*device->dci.eps));
        NVSHMEMI_NULL_ERROR_JMP(device->rc.eps, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                "allocation of rc.eps failed.");
        /* allocate device structs end */

        /* create shared device objects start */
        status = ibgda_create_cq_shared_objects(ibgda_state, device, n_pes);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "ibgda_create_cq_shared_objects failed.\n");

        status = ibgda_create_qp_shared_objects(ibgda_state, device, n_pes);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "ibgda_create_qp_shared_objects failed.");

        status = ibgda_create_dct_shared_objects(ibgda_state, device, portid);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "create DCT share err.");
        /* create shared device objects end */

        /* create DCTs start */
        for (int i = 0; i < device->dct.num_eps; ++i) {
            status = ibgda_create_dct(ibgda_state, &device->dct.eps[i], device, portid);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "ibgda_create_dct failed on DCT #%d.", i);

            status = ibgda_get_dct_handle(&local_dct_handles[i], device->dct.eps[i], device);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "ibgda_get_dct_handle failed on DCT #%d.", i);
        }
        /* create DCTs end */

        /* Gather DCT handles start */
        status = t->boot_handle->allgather(
            (void *)local_dct_handles, (void *)device->dct.dct_handles,
            sizeof(*local_dct_handles) * device->dct.num_eps, t->boot_handle);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "allgather of dct failed.");
        /* Gather DCT handles end */

        /* create and assign DCIs start */
        INFO(ibgda_state->log_level, "Creating %d DCI QPs (shared: %d, exclusive: %d)",
             device->dci.num_eps, device->dci.num_shared_eps,
             device->dci.num_eps - device->dci.num_shared_eps);

        for (int i = 0; i < device->dci.num_eps; ++i) {
            status = ibgda_create_qp(&device->dci.eps[i], device, portid, i,
                                     NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "ibgda_create_dci failed on DCI #%d.", i);
        }

        // Transition DCI to RTS.
        for (int i = 0; i < device->dci.num_eps; ++i) {
            status = ibgda_qp_rst2init(device->dci.eps[i], device, portid);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "ibgda_qp_rst2init failed on DCI #%d.", i);

            status = ibgda_dci_init2rtr(ibgda_state, device->dci.eps[i], device, portid);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "ibgda_dci_init2rtr failed on DCI #%d.", i);

            status = ibgda_qp_rtr2rts(device->dci.eps[i], device, portid);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "ibgda_qp_rtr2rts failed on DCI #%d.", i);
        }
        /* create and assign DCIs end */

        /* create and assign RCs start */
        INFO(ibgda_state->log_level, "Creating %d RC QPs", device->rc.num_eps_per_pe);
        for (int i = 0; i < num_rc_eps; ++i) {
            // Do not create loopback to self
            if (i / device->rc.num_eps_per_pe == mype) {
                continue;
            }
            status = ibgda_create_qp(&device->rc.eps[i], device, portid, i,
                                     NVSHMEMI_IBGDA_DEVICE_QP_TYPE_RC);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "ibgda_create_dci failed on RC #%d.", i);

            status = ibgda_get_rc_handle(&local_rc_handles[i], device->rc.eps[i], device);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "ibgda_get_rc_handle failed on RC #%d.", i);
        }

        if (num_rc_eps) {
            status = t->boot_handle->alltoall(
                (void *)local_rc_handles, (void *)device->rc.peer_ep_handles,
                sizeof(*local_rc_handles) * device->rc.num_eps_per_pe, t->boot_handle);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "alltoall of rc failed.");
        }

        for (int i = 0; i < num_rc_eps; ++i) {
            // No loopback to self
            if (i / device->rc.num_eps_per_pe == mype) {
                continue;
            }
            status = ibgda_qp_rst2init(device->rc.eps[i], device, portid);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "ibgda_qp_rst2init failed on RC #%d.", i);

            status = ibgda_rc_init2rtr(ibgda_state, device->rc.eps[i], device, portid,
                                       &device->rc.peer_ep_handles[i]);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "ibgda_rc_init2rtr failed on RC #%d.", i);

            status = ibgda_qp_rtr2rts(device->rc.eps[i], device, portid);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "ibgda_qp_rtr2rts failed on RC #%d.", i);
        }
        /* create and assign RCs end */

        /* skip half_av_seg check if any devices, EPs, or QPs don't support it. */
        if (support_half_av_seg) {
            for (int i = 0; i < device->dct.num_eps * n_pes; ++i) {
                support_half_av_seg &= device->dct.dct_handles[i].support_half_av_seg;
            }
        }

        ibgda_state->selected_dev_ids[init_dev_cnt] = curr_dev_id;
        ++init_dev_cnt;
    }

    /* Multiple devices break our CST optimizations. */
    if (init_dev_cnt > 1) {
        skip_cst = false;
    }

    /* set all device support_half_av_seg and need_cst together start */
    for (int i = 0; i < init_dev_cnt; i++) {
        curr_dev_id = ibgda_state->selected_dev_ids[i];
        device = ((struct ibgda_device *)ibgda_state->devices + curr_dev_id);
        device->support_half_av_seg = support_half_av_seg;
        device->may_skip_cst = skip_cst;
    }
    /* set all device support_half_av_seg and need_cst together end */

out:
    if (status) {
        if (ibgda_state->selected_dev_ids && init_dev_cnt == 0) {
            free(ibgda_state->selected_dev_ids);
            ibgda_state->selected_dev_ids = NULL;
        } else {
            status = 0;
        }
    }

    if (init_dev_cnt) {
        ibgda_state->n_devs_selected = init_dev_cnt;
        // Setup QPs / CQs on GPU.
        status = ibgda_setup_gpu_state(t);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "ibgda_setup_gpu_state failed.");
    }

    if (init_dev_cnt < num_selected_devs) {
        NVSHMEMI_WARN_PRINT("Failed to initialize all selected devices. Perf may be limited.");
    }

out_already_connected:
    if (local_rc_handles) {
        free(local_rc_handles);
    }

    if (local_dct_handles) {
        free(local_dct_handles);
    }
    return status;
}

int nvshmemt_ibgda_release_mem_handle(nvshmem_mem_handle_t *mem_handle, nvshmem_transport_t t) {
    int status = 0;
    nvshmemt_ibgda_state_t *ibgda_state = (nvshmemt_ibgda_state_t *)t->state;

    nvshmemi_ibgda_device_state_t *ibgda_device_state =
        (nvshmemi_ibgda_device_state_t *)t->type_specific_shared_state;
    assert(ibgda_device_state != NULL);

    struct ibgda_mem_handle *ibgda_mem_handle = (struct ibgda_mem_handle *)mem_handle;
    struct nvshmemt_ib_common_mem_handle *handle =
        (struct nvshmemt_ib_common_mem_handle *)&ibgda_mem_handle->dev_mem_handles[0];
    if (handle->local_only) {
        uint32_t position = 0;
        struct ibgda_device_local_only_mhandle_cache *prev_mhandle_cache = NULL;
        struct ibgda_device_local_only_mhandle_cache *next_mhandle_cache = NULL;
        struct ibgda_device_local_only_mhandle_cache *curr_mhandle_cache = NULL;
        void *mhandle_gpu_ptr;

        // Find the position in the host-side cache.
        for (auto it = ibgda_device_local_only_mhandles.begin();
             it != ibgda_device_local_only_mhandles.end(); ++it) {
            if (it->mhandle.start == (uint64_t)handle->buf) {
                curr_mhandle_cache = &ibgda_device_local_only_mhandles.data()[position];
                if (position > 0)
                    prev_mhandle_cache = &ibgda_device_local_only_mhandles.data()[position - 1];
                if (position < ibgda_device_local_only_mhandles.size() - 1)
                    next_mhandle_cache = &ibgda_device_local_only_mhandles.data()[position + 1];
                break;
            }
            ++position;
        }
        NVSHMEMI_NULL_ERROR_JMP(curr_mhandle_cache, status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                                "mem_handle is not registered.\n");

        // Remove this element from the linked list on both host and GPU.
        if (prev_mhandle_cache) {
            if (next_mhandle_cache)
                prev_mhandle_cache->mhandle.next =
                    (nvshmemi_ibgda_device_local_only_mhandle_t *)next_mhandle_cache->dev_ptr;
            else
                prev_mhandle_cache->mhandle.next = NULL;
            mhandle_gpu_ptr = (void *)((uintptr_t)prev_mhandle_cache->dev_ptr +
                                       offsetof(nvshmemi_ibgda_device_local_only_mhandle_t, next));
            status =
                cudaMemcpyAsync(mhandle_gpu_ptr, (const void *)&prev_mhandle_cache->mhandle.next,
                                sizeof(prev_mhandle_cache->mhandle.next), cudaMemcpyHostToDevice,
                                ibgda_state->my_stream);
            NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                                  "Setting local_only_mhandle in GPU memory failed.\n");
        } else {
            // The caller will trigger device state update.
            if (next_mhandle_cache)
                ibgda_device_state->globalmem.local_only_mhandle_head =
                    (nvshmemi_ibgda_device_local_only_mhandle_t *)next_mhandle_cache->dev_ptr;
            else
                ibgda_device_state->globalmem.local_only_mhandle_head = NULL;
        }

        // Free the copy of this element on GPU.
        status = cudaFree(curr_mhandle_cache->dev_ptr);
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                              "cudaFree failed.\n");

        ibgda_device_local_only_mhandles.erase(ibgda_device_local_only_mhandles.begin() + position);
    }

    // TODO: Clean up non-local-only mem_handle

    for (int i = 0; i < ibgda_state->n_devs_selected; i++) {
        handle = (struct nvshmemt_ib_common_mem_handle *)&ibgda_mem_handle->dev_mem_handles[i];
        status = nvshmemt_ib_common_release_mem_handle(&ftable, (nvshmem_mem_handle_t *)handle,
                                                       ibgda_state->log_level);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "nvshmemt_ib_common_release_mem_handle failed.\n");
    }

    status = cudaStreamSynchronize(ibgda_state->my_stream);
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                          "stream synchronize failed.\n");

out:
    return status;
}

int nvshmemt_ibgda_finalize(nvshmem_transport_t transport) {
    struct ibgda_device *device = NULL;
    assert(transport != NULL);
    nvshmemt_ibgda_state_t *ibgda_state = (nvshmemt_ibgda_state_t *)transport->state;
    nvshmemi_ibgda_device_state_t *ibgda_device_state_h;

    int status = 0, dev_id;
    int n_pes = transport->n_pes;
    int mype = transport->my_pe;
    int num_rc_eps;

    if (!ibgda_state) {
        goto out;
    }

    ibgda_device_lkeys.clear();
    ibgda_device_rkeys.clear();

    if (ibgda_device_lkeys_d) {
        cudaFree(ibgda_device_lkeys_d);
        ibgda_device_lkeys_d = 0;
    }
    if (ibgda_device_rkeys_d) {
        cudaFree(ibgda_device_rkeys_d);
        ibgda_device_rkeys_d = 0;
    }

    ibgda_device_state_h = (nvshmemi_ibgda_device_state_t *)transport->type_specific_shared_state;
    if (ibgda_device_state_h) {
        if (ibgda_device_state_h->globalmem.dcts) cudaFree(ibgda_device_state_h->globalmem.dcts);
        if (ibgda_device_state_h->globalmem.dcis) cudaFree(ibgda_device_state_h->globalmem.dcis);
        if (ibgda_device_state_h->globalmem.cqs) cudaFree(ibgda_device_state_h->globalmem.cqs);
        if (ibgda_device_state_h->globalmem.rcs) cudaFree(ibgda_device_state_h->globalmem.rcs);
        if (ibgda_device_state_h->globalmem.qp_group_switches)
            cudaFree(ibgda_device_state_h->globalmem.qp_group_switches);
    }

    for (int i = 0; i < ibgda_state->n_devs_selected; i++) {
        dev_id = ibgda_state->selected_dev_ids[i];
        device = ((struct ibgda_device *)ibgda_state->devices + dev_id);

        for (int i = 0; i < device->dci.num_eps; ++i) {
            status = ibgda_destroy_ep(device->dci.eps[i]);
        }

        for (int i = 0; i < device->dct.num_eps; ++i) {
            status = ibgda_destroy_ep(device->dct.eps[i]);
        }

        num_rc_eps = device->rc.num_eps_per_pe * n_pes;
        for (int i = 0; i < num_rc_eps; ++i) {
            // Do not create loopback to self
            if (i / device->rc.num_eps_per_pe == mype) {
                continue;
            }
            status = ibgda_destroy_ep(device->rc.eps[i]);
        }

        status = ibgda_destroy_qp_shared_objects(ibgda_state, device);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "ibgda_destroy_qp_shared_objects failed.\n");

        status = ibgda_destroy_dct_shared_objects(ibgda_state, device);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "ibgda_destroy_dct_shared_objects failed.\n");

        ibgda_destroy_cq_shared_objects(ibgda_state, device);
    }

    /* Free all devices, not just ones we used. */
    for (int i = 0; i < ibgda_state->n_dev_ids; i++) {
        device = (struct ibgda_device *)ibgda_state->devices + ibgda_state->dev_ids[i];
        if (device->pd) {
            status = ftable.dealloc_pd(device->pd);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_dealloc_pd failed \n");
        }

        if (device->context) {
            status = ftable.close_device(device->context);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "ibv_close_device failed \n");
        }
    }

#ifdef NVSHMEM_USE_GDRCOPY
    if (use_gdrcopy) {
        nvshmemt_gdrcopy_ftable_fini(&gdrcopy_ftable, &gdr_desc, &gdrcopy_handle);
    }
#endif

    nvshmemt_ibv_ftable_fini(&ibv_handle);

    if (transport->state) {
        if (ibgda_state->selected_dev_ids) {
            free(ibgda_state->selected_dev_ids);
            ibgda_state->selected_dev_ids = NULL;
        }
        free(transport->state);
    }

    if (transport->device_pci_paths) {
        for (int i = 0; i < transport->n_devices; i++) {
            free(transport->device_pci_paths[i]);
        }
        free(transport->device_pci_paths);
    }

out:
    free(transport);
    return status;
}

int nvshmemt_ibgda_add_device_remote_mem_handles(nvshmem_transport_t t, int transport_stride,
                                                 nvshmem_mem_handle_t *mem_handles,
                                                 uint64_t heap_offset, size_t size) {
    nvshmemt_ibgda_state_t *ibgda_state = (nvshmemt_ibgda_state_t *)t->state;
    int status = 0;
    int n_pes = t->n_pes;

    size_t num_rkeys;

    nvshmemi_ibgda_device_state_t *ibgda_device_state;

    ibgda_device_state = (nvshmemi_ibgda_device_state_t *)t->type_specific_shared_state;
    assert(ibgda_device_state != NULL);

    static_assert(sizeof(struct nvshmemt_ib_common_mem_handle) <= NVSHMEM_MEM_HANDLE_SIZE,
                  "static_assert(sizeof(T) <= NVSHMEM_MEM_HANDLE_SIZE) failed");

    size_t num_elements;
    // size must be divisible by cumem_granularity, which is a power of 2.
    assert((size & ((1ULL << t->log2_cumem_granularity) - 1)) == 0);

    num_elements = size >> t->log2_cumem_granularity;
    while (num_elements > 0) {
        for (int i = 0; i < n_pes; ++i) {
            // sizeof(struct ibgda_mem_handle) <= sizeof(nvshmem_mem_handle_t)
            // So, we calculate the pointer with nvshmem_mem_handle_t and convert to
            // ibgda_mem_handle later.
            struct ibgda_mem_handle *gmhandle =
                (struct ibgda_mem_handle *)&mem_handles[i * transport_stride + t->index];
            for (int j = 0; j < gmhandle->num_devs; j++) {
                struct nvshmemt_ib_common_mem_handle *handle =
                    (struct nvshmemt_ib_common_mem_handle *)&gmhandle->dev_mem_handles[j];
                nvshmemi_ibgda_device_key_t device_key;
                device_key.key = htobe32(handle->rkey);
                device_key.next_addr = heap_offset + size;

                ibgda_device_rkeys.emplace_back(device_key);
            }
        }
        --num_elements;
    }

    if (ibgda_device_rkeys_d) {
        status = cudaFree(ibgda_device_rkeys_d);
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                              "cudaFree failed.\n");
        ibgda_device_rkeys_d = 0;
    }

    num_rkeys = ibgda_device_rkeys.size();

    // For cache optimization, put rkeys in constant memory first.
    memcpy(
        ibgda_device_state->constmem.rkeys, ibgda_device_rkeys.data(),
        IBGDA_MIN(num_rkeys, NVSHMEMI_IBGDA_MAX_CONST_RKEYS) * sizeof(nvshmemi_ibgda_device_key_t));

    // Put the rest that don't fit in constant memory in global memory
    if (num_rkeys > NVSHMEMI_IBGDA_MAX_CONST_RKEYS) {
        size_t rkeys_array_size =
            sizeof(nvshmemi_ibgda_device_key_t) * (num_rkeys - NVSHMEMI_IBGDA_MAX_CONST_RKEYS);

        nvshmemi_ibgda_device_key_t *data_ptr =
            &ibgda_device_rkeys.data()[NVSHMEMI_IBGDA_MAX_CONST_RKEYS];

        status = cudaMalloc(&ibgda_device_rkeys_d, rkeys_array_size);
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                              "cudaMalloc failed.\n");

        status = cudaMemcpyAsync(ibgda_device_rkeys_d, (const void *)data_ptr, rkeys_array_size,
                                 cudaMemcpyHostToDevice, ibgda_state->my_stream);
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                              "Copying rkeys to GPU memory failed.\n");

        status = cudaStreamSynchronize(ibgda_state->my_stream);
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                              "stream synchronize failed.\n");
    }

    ibgda_device_state->globalmem.rkeys = (nvshmemi_ibgda_device_key_t *)ibgda_device_rkeys_d;
out:
    if (status) {
        // Unrecoverable error
        if (ibgda_device_rkeys_d) cudaFree(ibgda_device_rkeys_d);
        ibgda_device_rkeys.clear();
    }
    return status;
}

static ibgda_nic_mapping_memtype_reqeust_t ibgda_parse_nic_mapping_memtype_request(
    const char *str) {
    std::string req = str;

    // Trim whitespace
    req.erase(std::remove_if(req.begin(), req.end(), ::isspace), req.end());

    // To lower case
    std::for_each(req.begin(), req.end(), [](decltype(*req.begin()) &c) { c = ::tolower(c); });

    if (req == "gpumem")
        return IBGDA_NIC_MAPPING_MEMTYPE_REQUEST_GPUMEM;
    else if (req == "hostmem")
        return IBGDA_NIC_MAPPING_MEMTYPE_REQUEST_HOSTMEM;
    else
        return IBGDA_NIC_MAPPING_MEMTYPE_REQUEST_AUTO;
}

static int ibgda_check_nic_mapping_memtypes(nvshmemt_ibgda_state_t *ibgda_state,
                                            struct ibgda_device *device,
                                            ibgda_nic_mapping_memtype_reqeust_t request_memtype) {
    int status = 0;

    bool try_gpumem = ((request_memtype == IBGDA_NIC_MAPPING_MEMTYPE_REQUEST_AUTO) ||
                       (request_memtype == IBGDA_NIC_MAPPING_MEMTYPE_REQUEST_GPUMEM));
    bool try_hostmem = ((request_memtype == IBGDA_NIC_MAPPING_MEMTYPE_REQUEST_AUTO) ||
                        (request_memtype == IBGDA_NIC_MAPPING_MEMTYPE_REQUEST_HOSTMEM));

    bool can_use_gpumem = false;
    bool can_use_hostmem = false;

    struct ibgda_mem_object *mobject = NULL;

    if (try_gpumem) {
        status = ibgda_gpu_mem_alloc(&mobject, IBGDA_DBRSIZE, IBGDA_GPAGE_SIZE, false);
        if (status) goto out_try_gpumem;

        if (!ibgda_state->options->IB_DISABLE_DMABUF && ibgda_state->cuda_support_dmabuf) {
            status = ibgda_mobject_nic_map(mobject, device->context, IBV_ACCESS_LOCAL_WRITE, true);
            ibgda_state->dmabuf_support_for_control_buffers = (status == 0);
        }

        if (!ibgda_state->dmabuf_support_for_control_buffers) {
            status = ibgda_mobject_nic_map(mobject, device->context, IBV_ACCESS_LOCAL_WRITE, false);
            if (status) goto out_try_gpumem;
        }

        can_use_gpumem = true;

    out_try_gpumem:
        if (mobject) {
            if (mobject->has_nic_mapping) ibgda_mobject_nic_unmap(mobject);
            ibgda_gpu_mem_free(mobject);
        }
        mobject = NULL;
        status = 0;
    }

    if (try_hostmem) {
        status = ibgda_host_mem_alloc(&mobject, IBGDA_DBRSIZE, IBGDA_GPAGE_SIZE, true);
        if (status) goto out_try_hostmem;

        status = ibgda_mobject_nic_map(mobject, device->context, IBV_ACCESS_LOCAL_WRITE);
        if (status) goto out_try_hostmem;

        can_use_hostmem = true;

    out_try_hostmem:
        if (mobject) {
            if (mobject->has_nic_mapping) ibgda_mobject_nic_unmap(mobject);
            ibgda_host_mem_free(mobject);
        }
        mobject = NULL;
        status = 0;
    }

    device->support_nic_buf_on_gpumem = can_use_gpumem;
    device->support_nic_buf_on_hostmem = can_use_hostmem;

    if (!can_use_gpumem && !can_use_hostmem) return NVSHMEMX_ERROR_NOT_SUPPORTED;

    return 0;
}

static int ibgda_check_gpu_mapping_nic_uar(struct ibgda_device *device) {
    int status = 0;

    struct ibgda_mem_object *mobject = NULL;

    status = ibgda_alloc_and_map_qp_uar(device->context, IBGDA_NIC_HANDLER_GPU, &mobject);
    if (status) {
        NVSHMEMI_WARN_PRINT(
            "ibgda_alloc_and_map_qp_uar with GPU as handler failed. We may need to enter the CPU "
            "fallback path.\n");
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }

out:
    if (mobject) ibgda_unmap_and_free_qp_uar(mobject);
    return status;
}

int nvshmemt_init(nvshmem_transport_t *t, struct nvshmemi_cuda_fn_table *table, int api_version) {
    struct nvshmemt_hca_info hca_list[MAX_NUM_HCAS];
    struct nvshmemt_hca_info pe_hca_mapping[MAX_NUM_PES_PER_NODE];
    struct nvshmemi_options_s *options = NULL;

    int status = 0;
    int exclude_list = 0;
    int hca_list_count = 0;
    int pe_hca_map_count = 0;
    int user_selection = 0;
    int offset = 0;
    int num_devices = 0;
    int lowest_stream_priority;
    int highest_stream_priority;
    int flag;
    uint32_t atomic_host_endian_size = 0;
    CUdevice gpu_device_id;

    struct nvshmem_transport *transport = NULL;
    nvshmemt_ibgda_state_t *ibgda_state;
    struct ibgda_device *device;
    struct ibv_device **dev_list = NULL;

    bool nic_buf_on_gpumem = true;
    bool nic_buf_on_hostmem = true;

    ibgda_nic_mapping_memtype_reqeust_t nic_mapping_memtype_request;

    ibgda_nic_handler_t nic_handler_request;
    ibgda_nic_handler_t nic_handler = IBGDA_NIC_HANDLER_GPU;

    if (NVSHMEM_TRANSPORT_MAJOR_VERSION(api_version) != NVSHMEM_TRANSPORT_PLUGIN_MAJOR_VERSION) {
        NVSHMEMI_ERROR_PRINT(
            "NVSHMEM provided an incompatible version of the transport interface. "
            "This transport supports transport API major version %d. Host has %d",
            NVSHMEM_TRANSPORT_PLUGIN_MAJOR_VERSION, NVSHMEM_TRANSPORT_MAJOR_VERSION(api_version));
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    ibgda_cuda_syms = table;

    options = (struct nvshmemi_options_s *)calloc(1, sizeof(struct nvshmemi_options_s));
    NVSHMEMI_NULL_ERROR_JMP(options, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate options stuct for ibgda transport.\n");

    status = nvshmemi_env_options_init(options);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Unable to initialize NVSHMEM options.\n");

    transport = (struct nvshmem_transport *)malloc(sizeof(struct nvshmem_transport));
    NVSHMEMI_NULL_ERROR_JMP(transport, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate transport stuct for ibgda transport.\n");
    memset(transport, 0, sizeof(struct nvshmem_transport));

    ibgda_srq_depth = options->SRQ_DEPTH;
    if (ibgda_srq_depth <= 0) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "NVSHMEM_SRQ_DEPTH must be a positive number.\n");
    }

    ibgda_qp_depth = options->QP_DEPTH;
    if (ibgda_qp_depth > 0) {
        ibgda_qp_depth = IBGDA_ROUND_UP_POW2_OR_0(ibgda_qp_depth);
    }
    if (ibgda_qp_depth <= 0) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "NVSHMEM_QP_DEPTH must be a positive number.\n");
    } else if (ibgda_qp_depth < NVSHMEMI_IBGDA_MIN_QP_DEPTH) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "NVSHMEM_QP_DEPTH must be at least %d.\n", NVSHMEMI_IBGDA_MIN_QP_DEPTH);
    } else if (ibgda_qp_depth > NVSHMEMI_IBGDA_MAX_QP_DEPTH) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "NVSHMEM_QP_DEPTH can be at most %d.\n", NVSHMEMI_IBGDA_MAX_QP_DEPTH);
    }

    ibgda_num_requests_in_batch = options->IBGDA_NUM_REQUESTS_IN_BATCH;
    if (ibgda_num_requests_in_batch > 0) {
        ibgda_num_requests_in_batch = IBGDA_ROUND_UP_POW2_OR_0(ibgda_num_requests_in_batch);
    }
    if (ibgda_num_requests_in_batch <= 0) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "NVSHMEM_IBGDA_NUM_REQUESTS_IN_BATCH must be a positive number.\n");
    } else if (ibgda_num_requests_in_batch > ibgda_qp_depth) {
        NVSHMEMI_ERROR_JMP(
            status, NVSHMEMX_ERROR_INVALID_VALUE, out,
            "NVSHMEM_IBGDA_NUM_REQUESTS_IN_BATCH must not be larger than QP depth.\n");
    }

    ibgda_num_fetch_slots_per_dci = options->IBGDA_NUM_FETCH_SLOTS_PER_DCI;
    if (ibgda_num_fetch_slots_per_dci > 0) {
        ibgda_num_fetch_slots_per_dci = IBGDA_ROUND_UP_POW2_OR_0(ibgda_num_fetch_slots_per_dci);
    }
    if (ibgda_num_fetch_slots_per_dci <= 0) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "NVSHMEM_IBGDA_NUM_FETCH_SLOTS_PER_DCI must be a positive number.\n");
    }

    ibgda_num_fetch_slots_per_rc = options->IBGDA_NUM_FETCH_SLOTS_PER_RC;
    if (ibgda_num_fetch_slots_per_rc > 0) {
        ibgda_num_fetch_slots_per_rc = IBGDA_ROUND_UP_POW2_OR_0(ibgda_num_fetch_slots_per_rc);
    }
    if (ibgda_num_fetch_slots_per_rc <= 0) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "NVSHMEM_IBGDA_NUM_FETCH_SLOTS_PER_RC must be a positive number.\n");
    }

    ibgda_state = (nvshmemt_ibgda_state_t *)calloc(1, sizeof(nvshmemt_ibgda_state_t));
    NVSHMEMI_NULL_ERROR_JMP(ibgda_state, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "p2p state allocation failed \n");
    transport->state = (void *)ibgda_state;

    ibgda_state->log_level = nvshmemt_common_get_log_level(options);
    ibgda_state->options = options;

    if (nvshmemt_ibv_ftable_init(&ibv_handle, &ftable, ibgda_state->log_level)) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                           "Unable to dlopen libibverbs. Skipping IBGDA transport.\n");
    }

#ifdef NVSHMEM_USE_GDRCOPY
    if (options->DISABLE_GDRCOPY) {
        use_gdrcopy = false;
    } else {
        use_gdrcopy = nvshmemt_gdrcopy_ftable_init(&gdrcopy_ftable, &gdr_desc, &gdrcopy_handle,
                                                   ibgda_state->log_level);
    }
#endif

    status = CUPFN(ibgda_cuda_syms, cuCtxGetDevice(&gpu_device_id));
    if (status != CUDA_SUCCESS) {
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }
    status =
        CUPFN(ibgda_cuda_syms,
              cuDeviceGetAttribute(&flag, (CUdevice_attribute)CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED,
                                   gpu_device_id));
    if (status != CUDA_SUCCESS) {
        status = 0;
        cudaGetLastError();
        ibgda_state->cuda_support_dmabuf = false;
    } else {
        ibgda_state->cuda_support_dmabuf = (flag == 1);
    }

    ibgda_state->dmabuf_support_for_data_buffers = ibgda_state->cuda_support_dmabuf;
    if (options->IB_DISABLE_DMABUF) {
        ibgda_state->dmabuf_support_for_data_buffers = false;
    }

    if (ibgda_state->dmabuf_support_for_data_buffers == false) {
        if (nvshmemt_ib_common_nv_peer_mem_available() != NVSHMEMX_SUCCESS) {
            NVSHMEMI_ERROR_PRINT(
                "neither nv_peer_mem, or nvidia_peermem detected. Skipping transport.\n");
            status = NVSHMEMX_ERROR_INTERNAL;
            goto out;
        }
    }

    dev_list = ftable.get_device_list(&num_devices);
    NVSHMEMI_NULL_ERROR_JMP(dev_list, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "get_device_list failed \n");

    ibgda_state->devices = calloc(MAX_NUM_HCAS, sizeof(struct ibgda_device));
    NVSHMEMI_NULL_ERROR_JMP(ibgda_state->devices, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "get_device_list failed \n");

    ibgda_state->dev_ids = (int *)malloc(MAX_NUM_PES_PER_NODE * sizeof(int));
    NVSHMEMI_NULL_ERROR_JMP(ibgda_state->dev_ids, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "malloc failed \n");

    ibgda_state->port_ids = (int *)malloc(MAX_NUM_PES_PER_NODE * sizeof(int));
    NVSHMEMI_NULL_ERROR_JMP(ibgda_state->port_ids, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "malloc failed \n");
    if (options->HCA_LIST_provided) {
        user_selection = 1;
        exclude_list = (options->HCA_LIST[0] == '^');
        hca_list_count = nvshmemt_parse_hca_list(options->HCA_LIST, hca_list, MAX_NUM_HCAS,
                                                 ibgda_state->log_level);
    }

    if (options->HCA_PE_MAPPING_provided) {
        if (hca_list_count) {
            NVSHMEMI_WARN_PRINT(
                "Found conflicting parameters NVSHMEM_HCA_LIST and NVSHMEM_HCA_PE_MAPPING, "
                "ignoring "
                "NVSHMEM_HCA_PE_MAPPING \n");
        } else {
            user_selection = 1;
            pe_hca_map_count =
                nvshmemt_parse_hca_list(options->HCA_PE_MAPPING, pe_hca_mapping,
                                        MAX_NUM_PES_PER_NODE, ibgda_state->log_level);
        }
    }

    nic_mapping_memtype_request =
        ibgda_parse_nic_mapping_memtype_request(options->IBGDA_FORCE_NIC_BUF_MEMTYPE);
#ifdef NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY
    if (nic_mapping_memtype_request == IBGDA_NIC_MAPPING_MEMTYPE_REQUEST_AUTO) {
        nic_mapping_memtype_request = IBGDA_NIC_MAPPING_MEMTYPE_REQUEST_GPUMEM;
    }
    if (nic_mapping_memtype_request != IBGDA_NIC_MAPPING_MEMTYPE_REQUEST_GPUMEM) {
        NVSHMEMI_ERROR_JMP(
            status, NVSHMEMX_ERROR_NOT_SUPPORTED, out,
            "GPU-initiated communication is compiled with GPU memory support only.\n");
    }
#endif

    status = ibgda_parse_nic_handler_request(&nic_handler_request, options->IBGDA_NIC_HANDLER);
    NVSHMEMI_NZ_ERROR_JMP(status, status, out, "NVSHMEM_IBGDA_NIC_HANDLER is not valid.");

    if (!use_gdrcopy) {
        if (nic_handler_request == IBGDA_NIC_HANDLER_AUTO) {
            nic_handler_request = IBGDA_NIC_HANDLER_GPU;
        } else if (nic_handler_request == IBGDA_NIC_HANDLER_CPU) {
            NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_NOT_SUPPORTED, out,
                               "NVSHMEM_IBGDA_NIC_HANDLER=cpu requires GDRCopy.\n");
        }
    }

    INFO(ibgda_state->log_level,
         "Begin - Enumerating IB devices in the system ([<dev_id, device_name, num_ports>]) - \n");
    for (int i = 0; i < num_devices; i++) {
        device = (struct ibgda_device *)ibgda_state->devices + i;
        device->dev = dev_list[i];

        device->context = ftable.open_device(device->dev);
        if (!device->context) {
            INFO(ibgda_state->log_level, "open_device failed for IB device at index %d\n", i);
            continue;
        }

        const char *name = ftable.get_device_name(device->dev);
        NVSHMEMI_NULL_ERROR_JMP(name, status, NVSHMEMX_ERROR_INTERNAL, out,
                                "ibv_get_device_name failed \n");
        if (!strstr(name, "mlx5")) {
            ftable.close_device(device->context);
            device->context = NULL;
            NVSHMEMI_WARN_PRINT("device %s is not enumerated as an mlx5 device. Skipping...\n",
                                name);
            continue;
        }

        status = ftable.query_device(device->context, &device->device_attr);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_query_device failed \n");

        if (!nvshmemt_ib_common_query_mlx5_caps(device->context)) {
            ftable.close_device(device->context);
            device->context = NULL;
            NVSHMEMI_WARN_PRINT("device %s is not enumerated as an mlx5 device. Skipping...\n",
                                name);
            continue;
        }

        if (nic_handler_request == IBGDA_NIC_HANDLER_CPU) {
            device->nic_handler = IBGDA_NIC_HANDLER_CPU;
        } else {
            status = ibgda_check_gpu_mapping_nic_uar(device);
            device->nic_handler = status ? IBGDA_NIC_HANDLER_CPU : IBGDA_NIC_HANDLER_GPU;
            if (status && nic_handler_request == IBGDA_NIC_HANDLER_GPU) {
                ftable.close_device(device->context);
                device->context = NULL;
                NVSHMEMI_WARN_PRINT("GPU cannot map UAR of device %s. Skipping...\n", name);
                continue;
            }
        }

        status = ibgda_check_nic_mapping_memtypes(ibgda_state, device, nic_mapping_memtype_request);
        if (status) {
            ftable.close_device(device->context);
            device->context = NULL;
            NVSHMEMI_WARN_PRINT(
                "device %s cannot allocate buffer on the specified memory type. Skipping...\n",
                name);
            continue;
        }

        if (device->support_nic_buf_on_gpumem && !ibgda_state->options->IB_DISABLE_DMABUF &&
            !ibgda_state->dmabuf_support_for_control_buffers) {
            INFO(ibgda_state->log_level,
                 "The system does not support registering the NIC control buffers with DMABUF. "
                 "Fallback to use either nv_peer_mem or nvidia_peermem.\n");
        }

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

        INFO(ibgda_state->log_level,
             "Enumerated IB devices in the system - device id=%d (of %d), name=%s, num_ports=%d\n",
             i, num_devices, name, device->device_attr.phys_port_cnt);
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

                // IBGDA supports IB and RoCE.
                if ((device->port_attr[p - 1].state != IBV_PORT_ACTIVE) ||
                    ((device->port_attr[p - 1].link_layer != IBV_LINK_LAYER_INFINIBAND) &&
                     (device->port_attr[p - 1].link_layer != IBV_LINK_LAYER_ETHERNET))) {
                    if (user_selection) {
                        NVSHMEMI_WARN_PRINT(
                            "found inactive port or port with non IB/RoCE link layer protocol, "
                            "skipping...\n");
                    }
                    continue;
                }

                ib_get_gid_index(&ftable, device->context, p, device->port_attr[p - 1].gid_tbl_len,
                                 &device->gid_info[p - 1].local_gid_index, ibgda_state->log_level,
                                 options);
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
                    ibgda_state->dev_ids[offset] = i;
                    ibgda_state->port_ids[offset] = p;
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
            continue;
        }

        /* Report whether we need to do atomic endianness conversions on 8 byte operands. */
        status = nvshmemt_ib_common_query_endianness_conversion_size(&atomic_host_endian_size,
                                                                     device->context);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "nvshmemt_ib_common_query_endianness_conversion_size failed.\n");
    }
    INFO(ibgda_state->log_level, "End - Enumerating IB devices in the system\n");

    ibgda_state->n_dev_ids = offset;
    INFO(ibgda_state->log_level,
         "Begin - Ordered list of devices for assignment (after processing user provdied env vars "
         "(if any))  - \n");
    for (int i = 0; i < ibgda_state->n_dev_ids; i++) {
        INFO(ibgda_state->log_level,
             "Ordered list of devices for assignment - idx=%d (of %d), device id=%d, port_num=%d\n",
             i, ibgda_state->n_dev_ids, ibgda_state->dev_ids[i], ibgda_state->port_ids[i]);

        device = (struct ibgda_device *)ibgda_state->devices + ibgda_state->dev_ids[i];
        nic_buf_on_gpumem &= device->support_nic_buf_on_gpumem;
        nic_buf_on_hostmem &= device->support_nic_buf_on_hostmem;
        if (device->nic_handler == IBGDA_NIC_HANDLER_CPU) nic_handler = IBGDA_NIC_HANDLER_CPU;
    }
    INFO(ibgda_state->log_level,
         "End - Ordered list of devices for assignment (after processing user provdied env vars "
         "(if any))\n");

    if (!ibgda_state->n_dev_ids) {
        INFO(
            ibgda_state->log_level,
            "no active IB device that supports GPU-initiated communication is found, exiting...\n");
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }

    transport->n_devices = ibgda_state->n_dev_ids;
    transport->device_pci_paths = (char **)calloc(transport->n_devices, sizeof(char *));
    NVSHMEMI_NULL_ERROR_JMP(transport->device_pci_paths, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "Unable to allocate paths for IB transport.\n");
    for (int i = 0; i < transport->n_devices; i++) {
        status = get_pci_path(i, &transport->device_pci_paths[i], transport);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Failed to get paths for PCI devices.\n");
    }

    assert(nic_buf_on_gpumem || nic_buf_on_hostmem);
    if (nic_buf_on_gpumem) {
        ibgda_nic_buf_location = IBGDA_MEM_TYPE_GPU;
        INFO(ibgda_state->log_level, "NIC buffer will be on GPU memory.\n");
    } else {
        ibgda_nic_buf_location = IBGDA_MEM_TYPE_HOST;
        INFO(ibgda_state->log_level, "NIC buffer will be on host memory.\n");
    }

    assert(nic_handler == IBGDA_NIC_HANDLER_GPU || nic_handler == IBGDA_NIC_HANDLER_CPU);
    if (nic_handler == IBGDA_NIC_HANDLER_CPU) {
        assert(use_gdrcopy);
        INFO(ibgda_state->log_level, "NIC handler will be CPU.\n");
    } else {
        INFO(ibgda_state->log_level, "NIC handler will be GPU.\n");
    }
    ibgda_nic_handler = nic_handler;

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

    status = cudaDeviceGetStreamPriorityRange(&lowest_stream_priority, &highest_stream_priority);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "stream priority query failed. \n");
    status = cudaStreamCreateWithPriority(&ibgda_state->my_stream, cudaStreamNonBlocking,
                                          highest_stream_priority);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "internal stream creation failed. \n");

    transport->host_ops.can_reach_peer = nvshmemt_ibgda_can_reach_peer;
    transport->host_ops.connect_endpoints = nvshmemt_ibgda_connect_endpoints;
    transport->host_ops.get_mem_handle = nvshmemt_ibgda_get_mem_handle;
    transport->host_ops.release_mem_handle = nvshmemt_ibgda_release_mem_handle;
    transport->host_ops.show_info = nvshmemt_ibgda_show_info;
    transport->host_ops.progress =
        ((nic_handler == IBGDA_NIC_HANDLER_GPU) ? NULL : nvshmemt_ibgda_progress);
    transport->host_ops.finalize = nvshmemt_ibgda_finalize;
    transport->host_ops.rma = NULL;
    transport->host_ops.amo = NULL;
    transport->host_ops.fence = NULL;
    transport->host_ops.quiet = NULL;
    transport->host_ops.enforce_cst = NULL;
    transport->host_ops.add_device_remote_mem_handles =
        nvshmemt_ibgda_add_device_remote_mem_handles;

    transport->attr = NVSHMEM_TRANSPORT_ATTR_CONNECTED;
    transport->is_successfully_initialized = true;
    transport->max_op_len = 1ULL << 30;
    transport->atomic_host_endian_min_size = atomic_host_endian_size;
    transport->no_proxy = (nic_handler == IBGDA_NIC_HANDLER_GPU);
    transport->type = NVSHMEM_TRANSPORT_LIB_CODE_IBGDA;
    transport->api_version = api_version < NVSHMEM_TRANSPORT_INTERFACE_VERSION
                                 ? api_version
                                 : NVSHMEM_TRANSPORT_INTERFACE_VERSION;

    *t = transport;

out:
    if (status) {
        if (options) {
            free(options);
        }
        if (transport) {
            if (transport->device_pci_paths) {
                free(transport->device_pci_paths);
            }
            free(transport);
        }
    }
    // TODO: Implement cleanup
    return status;
}
