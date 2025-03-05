/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "transport_common.h"
#include <dlfcn.h>
#include <stdint.h>
#include <stdlib.h>

#include "non_abi/nvshmemx_error.h"

struct transport_mem_handle_info_cache {
    void **cache;
    uint64_t size;
    uint64_t address_granularity;
    uintptr_t address_mask;
};

int nvshmemt_parse_hca_list(const char *string, struct nvshmemt_hca_info *hca_list, int max_count,
                            int log_level) {
    if (!string) return 0;

    const char *ptr = string;
    // Ignore "^" name, will be detected outside of this function
    if (ptr[0] == '^') ptr++;

    int if_num = 0;
    int if_counter = 0;
    int segment_counter = 0;
    char c;
    do {
        c = *ptr;
        if (c == ':') {
            if (segment_counter == 0) {
                if (if_counter > 0) {
                    hca_list[if_num].name[if_counter] = '\0';
                    hca_list[if_num].port = atoi(ptr + 1);
                    hca_list[if_num].found = 0;
                    if_num++;
                    if_counter = 0;
                    segment_counter++;
                }
            } else {
                hca_list[if_num - 1].count = atoi(ptr + 1);
                segment_counter = 0;
            }
            c = *(ptr + 1);
            while (c != ',' && c != ':' && c != '\0') {
                ptr++;
                c = *(ptr + 1);
            }
        } else if (c == ',' || c == '\0') {
            if (if_counter > 0) {
                hca_list[if_num].name[if_counter] = '\0';
                hca_list[if_num].found = 0;
                if_num++;
                if_counter = 0;
            }
            segment_counter = 0;
        } else {
            if (if_counter == 0) {
                hca_list[if_num].port = -1;
                hca_list[if_num].count = 1;
            }
            hca_list[if_num].name[if_counter] = c;
            if_counter++;
        }
        ptr++;
    } while (if_num < max_count && c);

    INFO(log_level, "Begin - Parsed HCA list provided by user - ");
    for (int i = 0; i < if_num; i++) {
        INFO(log_level,
             "Parsed HCA list provided by user - i=%d (of %d), name=%s, port=%d, count=%d", i,
             if_num, hca_list[i].name, hca_list[i].port, hca_list[i].count);
    }
    INFO(log_level, "End - Parsed HCA list provided by user");

    return if_num;
}

int nvshmemt_ib_iface_get_mlx_path(const char *ib_name, char **path) {
    int status;

    char device_path[MAXPATHSIZE];
    status = snprintf(device_path, MAXPATHSIZE, "/sys/class/infiniband/%s/device", ib_name);
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

int nvshmemt_ibv_ftable_init(void **ibv_handle, struct nvshmemt_ibv_function_table *ftable,
                             int log_level) {
    *ibv_handle = dlopen("libibverbs.so.1", RTLD_LAZY);
    if (*ibv_handle == NULL) {
        INFO(log_level, "libibverbs not found on the system.");
        return -1;
    }

    LOAD_SYM(*ibv_handle, "ibv_fork_init", ftable->fork_init);
    LOAD_SYM(*ibv_handle, "ibv_create_ah", ftable->create_ah);
    LOAD_SYM(*ibv_handle, "ibv_get_device_list", ftable->get_device_list);
    LOAD_SYM(*ibv_handle, "ibv_get_device_name", ftable->get_device_name);
    LOAD_SYM(*ibv_handle, "ibv_open_device", ftable->open_device);
    LOAD_SYM(*ibv_handle, "ibv_close_device", ftable->close_device);
    LOAD_SYM(*ibv_handle, "ibv_query_port", ftable->query_port);
    LOAD_SYM(*ibv_handle, "ibv_query_device", ftable->query_device);
    LOAD_SYM(*ibv_handle, "ibv_alloc_pd", ftable->alloc_pd);
    LOAD_SYM(*ibv_handle, "ibv_reg_mr", ftable->reg_mr);
    LOAD_SYM(*ibv_handle, "ibv_reg_dmabuf_mr", ftable->reg_dmabuf_mr);
    LOAD_SYM(*ibv_handle, "ibv_dereg_mr", ftable->dereg_mr);
    LOAD_SYM(*ibv_handle, "ibv_create_cq", ftable->create_cq);
    LOAD_SYM(*ibv_handle, "ibv_create_qp", ftable->create_qp);
    LOAD_SYM(*ibv_handle, "ibv_create_srq", ftable->create_srq);
    LOAD_SYM(*ibv_handle, "ibv_modify_qp", ftable->modify_qp);
    LOAD_SYM(*ibv_handle, "ibv_query_gid", ftable->query_gid);
    LOAD_SYM(*ibv_handle, "ibv_dealloc_pd", ftable->dealloc_pd);
    LOAD_SYM(*ibv_handle, "ibv_destroy_qp", ftable->destroy_qp);
    LOAD_SYM(*ibv_handle, "ibv_destroy_cq", ftable->destroy_cq);
    LOAD_SYM(*ibv_handle, "ibv_destroy_srq", ftable->destroy_srq);
    LOAD_SYM(*ibv_handle, "ibv_destroy_ah", ftable->destroy_ah);

    return 0;
}

void nvshmemt_ibv_ftable_fini(void **ibv_handle) {
    int status;

    if (ibv_handle) {
        status = dlclose(*ibv_handle);
        if (status) {
            NVSHMEMI_ERROR_PRINT("Unable to close libibverbs handle.");
        }
    }
}

int nvshmemt_mem_handle_cache_init(nvshmem_transport_t t,
                                   struct transport_mem_handle_info_cache **cache) {
    struct transport_mem_handle_info_cache *cache_pointer;

    if (cache == NULL) {
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    *cache = (struct transport_mem_handle_info_cache *)calloc(
        1, sizeof(struct transport_mem_handle_info_cache));
    if (!(*cache)) {
        NVSHMEMI_ERROR_PRINT("Unable to allocate mem handle cache in transport code.");
        return NVSHMEMX_ERROR_OUT_OF_MEMORY;
    }

    cache_pointer = *cache;

    cache_pointer->cache = (void **)calloc(1000, sizeof(void *));
    if (!(cache_pointer->cache)) {
        NVSHMEMI_ERROR_PRINT("Unable to allocate mem handle cache in transport code.");
        return NVSHMEMX_ERROR_OUT_OF_MEMORY;
    }
    cache_pointer->size = 1000;
    cache_pointer->address_granularity = 1ULL << t->log2_cumem_granularity;
    cache_pointer->address_mask = (uintptr_t)(~(cache_pointer->address_granularity - 1));

    return NVSHMEMX_SUCCESS;
}

int nvshmemt_mem_handle_cache_add(nvshmem_transport_t t,
                                  struct transport_mem_handle_info_cache *cache, void *addr,
                                  void *mem_handle_info) {
    uint64_t addr_offset;
    uint64_t arr_idx;

    if (addr < t->heap_base) {
        NVSHMEMI_ERROR_PRINT("Unable to process pointers outside of the heap.");
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    addr_offset = (uint64_t)((char *)addr - (char *)t->heap_base);
    if (addr_offset % cache->address_granularity) {
        NVSHMEMI_ERROR_PRINT("Unable to process unaligned pointers.");
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    arr_idx = addr_offset / cache->address_granularity;

    if (arr_idx >= cache->size) {
        size_t new_cache_size = cache->size * 2 > arr_idx ? cache->size * 2 : arr_idx + 1;
        void *new_cache;
        new_cache = realloc(cache->cache, new_cache_size);
        if (new_cache == NULL) {
            NVSHMEMI_ERROR_PRINT("Unable to reallocate larger heap cache.");
            return NVSHMEMX_ERROR_OUT_OF_MEMORY;
        }

        cache->cache = (void **)new_cache;
        cache->size = new_cache_size;
    }

    cache->cache[arr_idx] = mem_handle_info;
    return NVSHMEMX_SUCCESS;
}

void *nvshmemt_mem_handle_cache_get(nvshmem_transport_t t,
                                    struct transport_mem_handle_info_cache *cache, void *addr) {
    uintptr_t addr_offset;
    uintptr_t aligned_addr;
    uint64_t arr_idx;

    if (addr < t->heap_base) {
        NVSHMEMI_ERROR_PRINT("Unable to process pointers outside of the heap.");
        return NULL;
    }

    addr_offset = (uintptr_t)((char *)addr - (char *)t->heap_base);
    aligned_addr = addr_offset & cache->address_mask;
    arr_idx = (uint64_t)aligned_addr / cache->address_granularity;

    if (arr_idx >= cache->size) {
        NVSHMEMI_ERROR_PRINT("Address not registered. Unable to get handle for it.");
        return NULL;
    }

    return cache->cache[arr_idx];
}

void *nvshmemt_mem_handle_cache_get_by_idx(struct transport_mem_handle_info_cache *cache,
                                           size_t idx) {
    if (idx > cache->size) {
        NVSHMEMI_ERROR_PRINT("Index out of bounds. Unable to get handle for it.");
    }
    return cache->cache[idx];
}
size_t nvshmemt_mem_handle_cache_get_size(struct transport_mem_handle_info_cache *cache) {
    return cache->size;
}

int nvshmemt_mem_handle_cache_remove(nvshmem_transport_t t,
                                     struct transport_mem_handle_info_cache *cache, void *addr) {
    uint64_t addr_offset;
    uint64_t arr_idx;

    if (addr < t->heap_base) {
        NVSHMEMI_ERROR_PRINT("Unable to process pointers outside of the heap.");
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    addr_offset = (uint64_t)((char *)addr - (char *)t->heap_base);
    if (addr_offset % cache->address_granularity) {
        NVSHMEMI_ERROR_PRINT("Unable to process unaligned pointers.");
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    arr_idx = addr_offset / cache->address_granularity;

    if (arr_idx >= cache->size) {
        NVSHMEMI_ERROR_PRINT("Address not registered. Unable to unregister it.");
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    cache->cache[arr_idx] = NULL;
    return NVSHMEMX_SUCCESS;
}

int nvshmemt_mem_handle_cache_fini(struct transport_mem_handle_info_cache *cache) {
    if (cache == NULL) {
        NVSHMEMI_ERROR_PRINT("Mem handle cache not initialized, cannot finalize it.");
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    if (cache->cache) {
        free(cache->cache);
    }

    free(cache);

    return NVSHMEMX_SUCCESS;
}
