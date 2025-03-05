#ifndef NVSHMEM_TYPES_H
#define NVSHMEM_TYPES_H

#define INIT_ARGS_PADDING 96
#define TEAM_CONFIG_PADDING 56
#define REDUCE_PADDING 32
#define TIMEOUT_PADDING 16
#define COLL_ENV_VARS_PADDING 472
#define COLL_ENV_VARS_V2_PADDING 424

#define RED_REC_INVALID_SCALAR -1
#define TEAM_CONFIG_SCALAR_INVALID -1
#define TEAM_SCALAR_INVALID -1
#define COLL_ENV_PARAMS_SCALAR_INVALID -1
#define COLL_ENV_PARAMS_USCALAR_INVALID 0xFFFFFFFF
#define COLL_ENV_PARAMS_ULSCALAR_INVALID 0xFFFFFFFFFFFFFFFF
#define TIMEOUT_ULSCALAR_INVALID 0
#define TEAM_SCALAR_INVALID -1
#define TEAM_USCALAR_INVALID 0xFFFFFFFF
#define TEAM_ULSCALAR_INVALID 0xFFFFFFFFFFFFFFFF
#define TEAM_ULSCALAR_DEFAULT 0
#define STATE_SCALAR_INVALID -1
#define STATE_USCALAR_INVALID 0xFFFFFFFF
#define STATE_ULSCALAR_INVALID 0xFFFFFFFFFFFFFFFF

#include "bootstrap_device_host/nvshmem_uniqueid.h"

#define NVSHMEM_INIT_ATTR_VER 1

#if not defined __CUDACC_RTC__
#include <stddef.h>
#include <stdint.h>
#include <limits.h>

#define nvshmemx_init_init_attr_ver_only(attr)                                     \
    do {                                                                           \
        attr.version = (1 << 16) + sizeof(nvshmemx_init_attr_t);                   \
        attr.args.version = (1 << 16) + sizeof(nvshmemx_init_args_t);              \
        attr.args.uid_args.version = (1 << 16) + sizeof(nvshmemx_uniqueid_args_t); \
    } while (0);

#define NVSHMEMX_INIT_ARGS_INITIALIZER                          \
    {                                                           \
        (1 << 16) + sizeof(nvshmemx_init_args_t), /* version */ \
            NVSHMEMX_UNIQUEID_ARGS_INITIALIZER, {               \
            0                                                   \
        }                                                       \
    }

#define NVSHMEMX_INIT_ATTR_INITIALIZER                           \
    {                                                            \
        (1 << 16) + sizeof(nvshmemx_init_attr_t), /* version */  \
            NULL,                                 /* mpi_comm */ \
            NVSHMEMX_INIT_ARGS_INITIALIZER                       \
    }

#define NVSHMEMI_RED_REX_INITIALIZER                                        \
    {                                                                       \
        (1 << 16) + sizeof(nvshmemi_reduce_recexch_t), /* version */        \
            RED_REC_INVALID_SCALAR,                    /* step1_sendto */   \
            NULL,                                      /* step1_recvfrom */ \
            NULL,                                      /* step2_nbrs */     \
            RED_REC_INVALID_SCALAR,                    /* step1_nrecvs */   \
            RED_REC_INVALID_SCALAR,                    /* step2_nphases */  \
        {                                                                   \
            0                                                               \
        }                                                                   \
    }

#define NVSHMEMI_TEAM_CONFIG_INITIALIZER                              \
    {                                                                 \
        (1 << 16) + sizeof(nvshmem_team_config_t), /* version */      \
            TEAM_CONFIG_SCALAR_INVALID,            /* num_contexts */ \
        {                                                             \
            0                                                         \
        }                                                             \
    }

#define NVSHMEMI_TEAM_INITIALIZER                                                               \
    {                                                                                           \
        (2 << 16) + sizeof(nvshmemi_team_t),                       /* version */                \
            TEAM_SCALAR_INVALID,                                   /* my_pe */                  \
            TEAM_SCALAR_INVALID,                                   /* start */                  \
            TEAM_SCALAR_INVALID,                                   /* stride */                 \
            TEAM_SCALAR_INVALID,                                   /* size */                   \
            TEAM_SCALAR_INVALID,                                   /* team_idx */               \
            NVSHMEMI_TEAM_CONFIG_INITIALIZER, TEAM_SCALAR_INVALID, /* config_mask */            \
            NULL,                                                  /* nccl_comm */              \
            NVSHMEMI_RED_REX_INITIALIZER, TEAM_ULSCALAR_INVALID,   /* rdxn_count */             \
            TEAM_USCALAR_INVALID,                                  /* ll_flag */                \
            TEAM_ULSCALAR_DEFAULT,                                 /* alltoall_pwrk[0] */       \
            TEAM_ULSCALAR_DEFAULT,                                 /* alltoall_pwrk[1] */       \
            TEAM_ULSCALAR_DEFAULT,                                 /* alltoall_count */         \
            TEAM_ULSCALAR_INVALID,                                 /* bcast_count */            \
            TEAM_ULSCALAR_INVALID,                                 /* bcast_sync_offset */      \
            TEAM_ULSCALAR_INVALID,                                 /* fcollect_count */         \
            TEAM_USCALAR_INVALID,                                  /* fcollect_ll_flag */       \
            false,                                                 /* are_gpus_p2p_connected */ \
            false,                                                 /* is_team_node */           \
            TEAM_SCALAR_INVALID,                                   /* team_node */              \
            false,                                                 /* is_team_same_mype_node */ \
            TEAM_SCALAR_INVALID,                                   /* team_same_mype_node */    \
            NULL,                                                  /* nvls_rsc */               \
            NULL,                                                  /* nvls_rsc_base_ptr */      \
        {                                                                                       \
            TEAM_SCALAR_INVALID                                                                 \
        } /* team_dups */                                                                       \
    }

#define NVSHMEMI_GPU_COLL_PARAMS_INITIALIZER                                          \
    {                                                                                 \
        (2 << 16) + sizeof(gpu_coll_env_params_t), /* version */                      \
            COLL_ENV_PARAMS_SCALAR_INVALID,        /* barrier_dissem_kval */          \
            COLL_ENV_PARAMS_SCALAR_INVALID,        /* barrier_tg_dissem_kval */       \
            COLL_ENV_PARAMS_SCALAR_INVALID,        /* reduce_recexch_kval */          \
            COLL_ENV_PARAMS_SCALAR_INVALID,        /* bcast_tree_kval */              \
            COLL_ENV_PARAMS_SCALAR_INVALID,        /* bcast_algo */                   \
            COLL_ENV_PARAMS_SCALAR_INVALID,        /* reduce_algo */                  \
            COLL_ENV_PARAMS_ULSCALAR_INVALID,      /* fcollect_ll_threshold */        \
            COLL_ENV_PARAMS_ULSCALAR_INVALID,      /* fcollect_nvls_threshold */      \
            COLL_ENV_PARAMS_ULSCALAR_INVALID,      /* reduce_scratch_size */          \
            COLL_ENV_PARAMS_SCALAR_INVALID,        /* fcollect_algo */                \
            COLL_ENV_PARAMS_ULSCALAR_INVALID,      /* reducescatter_nvls_threshold */ \
            COLL_ENV_PARAMS_SCALAR_INVALID,        /* reducescatter_algo */           \
            COLL_ENV_PARAMS_SCALAR_INVALID,        /* reduce_maxloc_algo */           \
            COLL_ENV_PARAMS_ULSCALAR_INVALID,      /* fcollect_ll128_threahold */     \
        {                                                                             \
            0                                                                         \
        }                                                                             \
    }

#define NVSHMEMI_TIMEOUT_INITIALIZER                                      \
    {                                                                     \
        (1 << 16) + sizeof(nvshmemi_timeout_t), /* version */             \
            TIMEOUT_ULSCALAR_INVALID,           /* signal */              \
            TIMEOUT_ULSCALAR_INVALID,           /* caller */              \
            TIMEOUT_ULSCALAR_INVALID,           /* signal_addr */         \
            TIMEOUT_ULSCALAR_INVALID,           /* signal_val_found */    \
            TIMEOUT_ULSCALAR_INVALID,           /* signal_val_expected */ \
        {                                                                 \
            0                                                             \
        }                                                                 \
    }

#define NVSHMEMI_DEVICE_HOST_STATE_INITIALIZER                                                    \
    {                                                                                             \
        (1 << 16) + sizeof(nvshmemi_device_host_state_t), /* version */                           \
            STATE_SCALAR_INVALID,                         /* mype */                              \
            STATE_SCALAR_INVALID,                         /* npes */                              \
            STATE_SCALAR_INVALID,                         /* node_mype */                         \
            STATE_SCALAR_INVALID,                         /* node_npes */                         \
            NVSHMEMI_PE_DIST_MAX,                         /* pe_dist */                           \
            STATE_SCALAR_INVALID,                         /* proxy */                             \
            STATE_SCALAR_INVALID,                         /* atomics_sync */                      \
            STATE_SCALAR_INVALID,                         /* job_connectivity */                  \
            false,                                        /* proxy_ops_are_ordered */             \
            false,                                        /* atomics_complete_on_quiet */         \
            NULL,                                         /* heap_base */                         \
            STATE_ULSCALAR_INVALID,                       /* heap_size */                         \
            NULL,                                         /* peer_heap_base_p2p */                \
            NULL,                                         /* peer_heap_base_remote */             \
            false,                                        /* symmetric_heap_kind */               \
            false,                                        /* enable_rail_opt */                   \
            STATE_USCALAR_INVALID,                        /* atomics_le_min_size */               \
            NULL,                                         /* timeout */                           \
            NULL,                                         /* test_wait_any_start_idx_ptr */       \
            NULL,                                         /* team_pool */                         \
            NULL,                                         /* psync_pool */                        \
            NULL,                                         /* sync_counter */                      \
            NVSHMEMI_GPU_COLL_PARAMS_INITIALIZER,         /* gpu_coll_env_params_var */           \
            NULL,                                         /* proxy_channels_buf */                \
            NULL,                                         /* proxy_channel_g_buf */               \
            NULL,                                         /* proxy_channel_g_coalescing_buf */    \
            NULL,                                         /* proxy_channel_g_buf_head_ptr */      \
            STATE_ULSCALAR_INVALID,                       /* proxy_channel_g_buf_size */          \
            STATE_ULSCALAR_INVALID,                       /* proxy_channel_g_buf_log_size */      \
            NULL,                                         /* proxy_channels_issue */              \
            NULL,                                         /* proxy_channels_complete */           \
            NULL,                                         /* proxy_channels_complete_local_ptr */ \
            NULL,                                         /* proxy_channels_quiet_issue */        \
            NULL,                                         /* proxy_channels_quiet_ack */          \
            NULL,                                         /* proxy_channels_cst_issue */          \
            NULL,                                         /* proxy_channels_cst_ack */            \
            STATE_ULSCALAR_INVALID,                       /* proxy_channel_buf_size */            \
            STATE_USCALAR_INVALID,                        /* proxy_channel_buf_logsize */         \
            NULL,                                         /* global_exit_request_state */         \
            NULL,                                         /* global_exit_code */                  \
            false,                                        /* ibgda_is_initialized */              \
            false,                                        /* nvshmemi_is_nvshmem_initialized */   \
            false                                         /* nvshmemi_is_nvshmem_bootstrapped */  \
    }
#else
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/climits>
#endif

typedef int32_t nvshmem_team_t;
typedef nvshmem_team_t nvshmemx_team_t;

typedef enum {
    NVSHMEMI_PE_DIST_ROUNDROBIN = 0,
    NVSHMEMI_PE_DIST_BLOCK,
    NVSHMEMI_PE_DIST_MISC,
    NVSHMEMI_PE_DIST_MAX = INT_MAX
} nvshmemi_pe_dist_t;

typedef struct {
    int version;
    nvshmemx_uniqueid_args_t uid_args;
    char content[INIT_ARGS_PADDING];
} nvshmemx_init_args_v1;
static_assert(sizeof(nvshmemx_init_args_v1) == 128, "init_args_v1 must be 128 bytes.");

typedef nvshmemx_init_args_v1 nvshmemx_init_args_t;

typedef struct {
    int version;
    void *mpi_comm;
    nvshmemx_init_args_t args;
} nvshmemx_init_attr_v1;
static_assert(sizeof(nvshmemx_init_attr_v1) == 144, "init_attr_v1 must be 144 bytes.");

typedef nvshmemx_init_attr_v1 nvshmemx_init_attr_t;

typedef struct {
    int version;
    int step1_sendto;
    int *step1_recvfrom;
    int **step2_nbrs;
    int step1_nrecvs;
    int step2_nphases;
    char padding[REDUCE_PADDING];
} nvshmemi_reduce_recexch_v1;
static_assert(sizeof(nvshmemi_reduce_recexch_v1) == 64, "reduce_recexch_v1 must be 64 bytes.");

typedef nvshmemi_reduce_recexch_v1 nvshmemi_reduce_recexch_t;

typedef struct {
    int version;
    int num_contexts;
    char padding[TEAM_CONFIG_PADDING];
} nvshmem_team_config_v1;
static_assert(sizeof(nvshmem_team_config_v1) == 64, "team_config_v1 must be 64 bytes.");

typedef nvshmem_team_config_v1 nvshmem_team_config_t;

typedef struct {
    int version;
    int my_pe;
    int start, stride, size;
    int team_idx;
    nvshmem_team_config_t config;
    long config_mask;
    void *nccl_comm; /* To be cast to ncclComm_t whenever used */
    nvshmemi_reduce_recexch_t reduce_recexch;
    size_t rdxn_count;
    uint32_t ll_flag;
    uint64_t alltoall_pwrk[2];
    uint64_t alltoall_count;
    uint64_t bcast_count;
    uint64_t bcast_sync_offset;
    uint64_t fcollect_count;
    uint32_t fcollect_ll_flag;
    bool are_gpus_p2p_connected;
    bool is_team_node; /* If set to true, 'team_node' refers to rsvd NVSHMEMX_TEAM_NODE */
    nvshmem_team_t team_node;
    bool is_team_same_mype_node; /* If set to true, 'team_same_mype_node' refers to rsvd
                                    NVSHMEMX_TEAM_SAME_MYPE_NODE */
    nvshmem_team_t team_same_mype_node;
    void *nvls_rsc;          /* To be cast to nvshmemi_nvls_rsc whenever used */
    void *nvls_rsc_base_ptr; /* Shared b/w GPU threads of this team */
    nvshmem_team_t team_dups[128];
} nvshmemi_team_v2;

typedef struct {
    int version;
    int my_pe;
    int start, stride, size;
    int team_idx;
    nvshmem_team_config_t config;
    long config_mask;
    void *nccl_comm; /* To be cast to ncclComm_t whenever used */
    nvshmemi_reduce_recexch_t reduce_recexch;
    size_t rdxn_count;
    uint32_t ll_flag;
    uint64_t alltoall_pwrk[2];
    uint64_t alltoall_count;
    uint64_t bcast_count;
    uint64_t bcast_sync_offset;
    uint64_t fcollect_count;
    uint32_t fcollect_ll_flag;
    bool are_gpus_p2p_connected;
    bool is_team_node; /* If set to true, 'team_node' refers to rsvd NVSHMEMX_TEAM_NODE */
    nvshmem_team_t team_node;
    bool is_team_same_mype_node; /* If set to true, 'team_same_mype_node' refers to rsvd
                                    NVSHMEMX_TEAM_SAME_MYPE_NODE */
    nvshmem_team_t team_same_mype_node;
} nvshmemi_team_v1;
static_assert(sizeof(nvshmemi_team_v1) == 256, "team_v1 must be 256 bytes.");
static_assert(sizeof(nvshmemi_team_v2) == 784, "team_v2 must be 784 bytes.");

typedef nvshmemi_team_v2 nvshmemi_team_t;

typedef struct {
    int version;
    int barrier_dissem_kval;
    int barrier_tg_dissem_kval;
    int reduce_recexch_kval;
    int bcast_tree_kval;
    int bcast_algo;
    int reduce_algo;
    size_t fcollect_ll_threshold;
    char padding[COLL_ENV_VARS_PADDING];
} gpu_coll_env_params_v1;
static_assert(sizeof(gpu_coll_env_params_v1) == 512, "gpu_coll_env_params_v1 must be 512 bytes.");

typedef struct {
    int version;
    int barrier_dissem_kval;
    int barrier_tg_dissem_kval;
    int reduce_recexch_kval;
    int bcast_tree_kval;
    int bcast_algo;
    int reduce_algo;
    size_t fcollect_ll_threshold;
    size_t fcollect_nvls_threshold;
    size_t reduce_scratch_size;
    int fcollect_algo;
    size_t reducescatter_nvls_threshold;
    int reducescatter_algo;
    int reduce_maxloc_algo;
    size_t fcollect_ll128_threshold;
    char padding[COLL_ENV_VARS_V2_PADDING];
} gpu_coll_env_params_v2;
static_assert(sizeof(gpu_coll_env_params_v2) == 512, "gpu_coll_env_params_v2 must be 512 bytes.");

typedef gpu_coll_env_params_v2 gpu_coll_env_params_t;

typedef struct {
    int version;
    uint64_t signal;
    uint64_t caller;
    uint64_t signal_addr;
    uint64_t signal_val_found;
    uint64_t signal_val_expected;
    char padding[TIMEOUT_PADDING];
} nvshmemi_timeout_v1;
static_assert(sizeof(nvshmemi_timeout_v1) == 64, "timeout_v1 must be 64 bytes.");

typedef nvshmemi_timeout_v1 nvshmemi_timeout_t;

typedef struct {
    int version;
    int mype;
    int npes;
    int node_mype;
    int node_npes;
    nvshmemi_pe_dist_t pe_dist;
    int proxy;
    int atomics_sync;
    int job_connectivity;
    bool proxy_ops_are_ordered;
    bool atomics_complete_on_quiet;
    void *heap_base;
    size_t heap_size;
    void **peer_heap_base_p2p;
    void **peer_heap_base_remote;
    bool symmetric_heap_kind;
    bool enable_rail_opt;
    uint32_t atomics_le_min_size;

    nvshmemi_timeout_t *timeout;
    unsigned long long *test_wait_any_start_idx_ptr;

    nvshmemi_team_t **team_pool;
    long *psync_pool;
    long *sync_counter;
    gpu_coll_env_params_t gpu_coll_env_params_var;

    /* channel */
    void *proxy_channels_buf; /* requests are written in this buffer */
    char *proxy_channel_g_buf;
    char *proxy_channel_g_coalescing_buf;
    uint64_t *proxy_channel_g_buf_head_ptr; /* next location to be assigned to a thread */
    uint64_t proxy_channel_g_buf_size;      /* Total size of g_buf in bytes */
    uint64_t proxy_channel_g_buf_log_size;  /* Total size of g_buf in bytes */
    uint64_t *proxy_channels_issue;         /* last byte of the last request */
    uint64_t *
        proxy_channels_complete; /* shared betwen CPU and GPU threads - only write by CPU thread and
                                      read by GPU threads. This is allocated on the system memory */
    uint64_t *proxy_channels_complete_local_ptr; /* shared only between GPU threads */
    uint64_t *proxy_channels_quiet_issue;
    uint64_t *proxy_channels_quiet_ack;
    uint64_t *proxy_channels_cst_issue;
    uint64_t *proxy_channels_cst_ack;
    uint64_t proxy_channel_buf_size; /* Maximum number of inflight requests in bytes OR
                                                   maximum channel length */
    uint32_t proxy_channel_buf_logsize;
    int *global_exit_request_state;
    int *global_exit_code;

    bool ibgda_is_initialized;
    bool nvshmemi_is_nvshmem_initialized;
    bool nvshmemi_is_nvshmem_bootstrapped;
} nvshmemi_device_host_state_v1;
static_assert(sizeof(nvshmemi_device_host_state_v1) == 776,
              "device_host_state_v1 must be 776 bytes.");

typedef nvshmemi_device_host_state_v1 nvshmemi_device_host_state_t;

#endif /* NVSHMEM_TYPES_H */
