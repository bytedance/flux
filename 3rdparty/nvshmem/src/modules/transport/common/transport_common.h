/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _TRANSPORT_COMMON_H
#define _TRANSPORT_COMMON_H

#define __STDC_FORMAT_MACROS 1

#include <stdint.h>  // IWYU pragma: keep
// IWYU pragma: no_include <bits/stdint-uintn.h>
#include <dlfcn.h>
#include <stdio.h>
#include <strings.h>

#include "internal/host_transport/transport.h"
#include "bootstrap_host_transport/env_defs_internal.h"

#define MAXPATHSIZE 1024
#define MAX_TRANSPORT_EP_COUNT 1

#define likely(x) __builtin_expect((x), 1)
#define unlikely(x) __builtin_expect((x), 0)

#define TRANSPORT_LOG_NONE 0
#define TRANSPORT_LOG_VERSION 1
#define TRANSPORT_LOG_WARN 2
#define TRANSPORT_LOG_INFO 3
#define TRANSPORT_LOG_ABORT 4
#define TRANSPORT_LOG_TRACE 5

#if defined(NVSHMEM_x86_64)
#define MEM_BARRIER() asm volatile("mfence" ::: "memory")
#define STORE_BARRIER() asm volatile("sfence" ::: "memory")
#define LOAD_BARRIER() asm volatile("lfence" ::: "memory")
#elif defined(NVSHMEM_PPC64LE)
#define MEM_BARRIER() asm volatile("sync" ::: "memory")
#define STORE_BARRIER() MEM_BARRIER()
#define LOAD_BARRIER() MEM_BARRIER()
#elif defined(NVSHMEM_AARCH64)
#define MEM_BARRIER() asm volatile("dmb sy" ::: "memory")
#define STORE_BARRIER() asm volatile("dmb st" ::: "memory")
#define LOAD_BARRIER() MEM_BARRIER()
#else
#define MEM_BARRIER() asm volatile("" ::: "memory")
#define STORE_BARRIER() MEM_BARRIER()
#define LOAD_BARRIER() MEM_BARRIER()
#endif

#define INFO(LOG_LEVEL, fmt, ...)                                                  \
    do {                                                                           \
        if (LOG_LEVEL >= TRANSPORT_LOG_INFO) {                                     \
            fprintf(stderr, "%s %d " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
        }                                                                          \
    } while (0)

#define TRACE(LOG_LEVEL, fmt, ...)                                                 \
    do {                                                                           \
        if (LOG_LEVEL >= TRANSPORT_LOG_TRACE) {                                    \
            fprintf(stderr, "%s %d " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
        }                                                                          \
    } while (0)

#define LOAD_SYM(handle, symbol, funcptr)  \
    do {                                   \
        void **cast = (void **)&funcptr;   \
        void *tmp = dlsym(handle, symbol); \
        *cast = tmp;                       \
    } while (0)

static inline int nvshmemt_common_get_log_level(struct nvshmemi_options_s *options) {
    if (!options->DEBUG_provided && !options->DEBUG_SUBSYS_provided) {
        return TRANSPORT_LOG_NONE;
    } else if (strncasecmp(options->DEBUG, "VERSION", 8) == 0) {
        return TRANSPORT_LOG_VERSION;
    } else if (strncasecmp(options->DEBUG, "WARN", 5) == 0) {
        return TRANSPORT_LOG_WARN;
    } else if (strncasecmp(options->DEBUG, "INFO", 5) == 0) {
        return TRANSPORT_LOG_INFO;
    } else if (strncasecmp(options->DEBUG, "ABORT", 6) == 0) {
        return TRANSPORT_LOG_ABORT;
    } else if (strncasecmp(options->DEBUG, "TRACE", 6) == 0) {
        return TRANSPORT_LOG_TRACE;
    }

    return TRANSPORT_LOG_INFO;
}

struct transport_mem_handle_info_cache;  // IWYU pragma: keep

struct nvshmemt_hca_info {
    char name[64];
    int port;
    int count;
    int found;
};

typedef int (*pci_path_cb)(int dev, char **pcipath, struct nvshmem_transport *transport);

int nvshmemt_parse_hca_list(const char *string, struct nvshmemt_hca_info *hca_list, int max_count,
                            int log_level);
int nvshmemt_ib_iface_get_mlx_path(const char *ib_name, char **path);

struct nvshmemt_ibv_function_table {
    int (*fork_init)(void);
    struct ibv_ah *(*create_ah)(struct ibv_pd *pd, struct ibv_ah_attr *ah_attr);
    struct ibv_device **(*get_device_list)(int *num_devices);
    const char *(*get_device_name)(struct ibv_device *device);
    struct ibv_context *(*open_device)(struct ibv_device *device);
    int (*close_device)(struct ibv_context *context);
    int (*query_device)(struct ibv_context *context, struct ibv_device_attr *device_attr);
    int (*query_port)(struct ibv_context *context, uint8_t port_num,
                      struct ibv_port_attr *port_attr);
    struct ibv_pd *(*alloc_pd)(struct ibv_context *context);
    struct ibv_mr *(*reg_mr)(struct ibv_pd *pd, void *addr, size_t length, int access);
    struct ibv_mr *(*reg_dmabuf_mr)(struct ibv_pd *pd, uint64_t offset, size_t length,
                                    uint64_t iova, int fd, int access);
    int (*dereg_mr)(struct ibv_mr *mr);
    struct ibv_cq *(*create_cq)(struct ibv_context *context, int cqe, void *cq_context,
                                struct ibv_comp_channel *channel, int comp_vector);
    struct ibv_qp *(*create_qp)(struct ibv_pd *pd, struct ibv_qp_init_attr *qp_init_attr);
    struct ibv_srq *(*create_srq)(struct ibv_pd *pd, struct ibv_srq_init_attr *srq_init_attr);
    int (*dealloc_pd)(struct ibv_pd *pd);
    int (*modify_qp)(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask);
    int (*query_gid)(struct ibv_context *context, uint8_t port_num, int index, union ibv_gid *gid);
    int (*destroy_qp)(struct ibv_qp *qp);
    int (*destroy_cq)(struct ibv_cq *cq);
    int (*destroy_srq)(struct ibv_srq *srq);
    int (*destroy_ah)(struct ibv_ah *ah);
};

int nvshmemt_ibv_ftable_init(void **ibv_handle, struct nvshmemt_ibv_function_table *ftable,
                             int log_level);
void nvshmemt_ibv_ftable_fini(void **ibv_handle);

int nvshmemt_mem_handle_cache_init(nvshmem_transport_t t,
                                   struct transport_mem_handle_info_cache **cache);
int nvshmemt_mem_handle_cache_add(nvshmem_transport_t t,
                                  struct transport_mem_handle_info_cache *cache, void *addr,
                                  void *mem_handle_info);
void *nvshmemt_mem_handle_cache_get(nvshmem_transport_t t,
                                    struct transport_mem_handle_info_cache *cache, void *addr);
void *nvshmemt_mem_handle_cache_get_by_idx(struct transport_mem_handle_info_cache *cache,
                                           size_t idx);
size_t nvshmemt_mem_handle_cache_get_size(struct transport_mem_handle_info_cache *cache);
int nvshmemt_mem_handle_cache_remove(nvshmem_transport_t t,
                                     struct transport_mem_handle_info_cache *cache, void *addr);
int nvshmemt_mem_handle_cache_fini(struct transport_mem_handle_info_cache *cache);

extern "C" {
int nvshmemt_init(nvshmem_transport_t *transport, struct nvshmemi_cuda_fn_table *table,
                  int api_version);
}

#endif
