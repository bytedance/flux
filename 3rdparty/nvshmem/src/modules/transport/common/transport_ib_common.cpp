/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "transport_ib_common.h"
#include <assert.h>                            // for assert
#include <cuda.h>                              // for CUdeviceptr, CU_MEM_RANGE_HANDLE_TY...
#include <cuda_runtime.h>                      // for cudaGetLastError, cudaPointerGetAtt...
#include <driver_types.h>                      // for cudaPointerAttributes, cudaMemoryTy...
#include <errno.h>                             // for errno
#include <infiniband/verbs.h>                  // for IBV_ACCESS_LOCAL_WRITE, IBV_ACCESS_...
#include <stdint.h>                            // for uintptr_t, uint64_t
#include <string.h>                            // for strerror
#include <unistd.h>                            // for access, close, sysconf, F_OK, _SC_P...
#include "internal/host_transport/cudawrap.h"  // for nvshmemi_cuda_fn_table, CUCHECKGOTO
#include "non_abi/nvshmemx_error.h"            // for NVSHMEMX_ERROR_INTERNAL, NVSHMEMX_S...
#include "transport_common.h"                  // for nvshmemt_ibv_function_table, INFO

int nvshmemt_ib_common_nv_peer_mem_available() {
    if (access("/sys/kernel/mm/memory_peers/nv_mem/version", F_OK) == 0) {
        return NVSHMEMX_SUCCESS;
    }
    if (access("/sys/kernel/mm/memory_peers/nvidia-peermem/version", F_OK) == 0) {
        return NVSHMEMX_SUCCESS;
    }

    return NVSHMEMX_ERROR_INTERNAL;
}

int nvshmemt_ib_common_reg_mem_handle(struct nvshmemt_ibv_function_table *ftable, struct ibv_pd *pd,
                                      nvshmem_mem_handle_t *mem_handle, void *buf, size_t length,
                                      bool local_only, bool dmabuf_support,
                                      struct nvshmemi_cuda_fn_table *table, int log_level,
                                      bool relaxed_ordering) {
    struct nvshmemt_ib_common_mem_handle *handle =
        (struct nvshmemt_ib_common_mem_handle *)mem_handle;
    struct ibv_mr *mr = NULL;
    int status = 0;
    int ro_flag = 0;
    bool host_memory = false;

    assert(sizeof(struct nvshmemt_ib_common_mem_handle) <= NVSHMEM_MEM_HANDLE_SIZE);

    cudaPointerAttributes attr;
    status = cudaPointerGetAttributes(&attr, buf);
    if (status != cudaSuccess) {
        host_memory = true;
        status = 0;
        cudaGetLastError();
    } else if (attr.type != cudaMemoryTypeDevice) {
        host_memory = true;
    }

#if defined(HAVE_IBV_ACCESS_RELAXED_ORDERING)
#if HAVE_IBV_ACCESS_RELAXED_ORDERING == 1
    // IBV_ACCESS_RELAXED_ORDERING has been introduced to rdma-core since v28.0.
    if (relaxed_ordering) {
        ro_flag = IBV_ACCESS_RELAXED_ORDERING;
    }
#endif
#endif

    if (ftable->reg_dmabuf_mr != NULL && !host_memory && dmabuf_support &&
        CUPFN(table, cuMemGetHandleForAddressRange)) {
        size_t page_size = sysconf(_SC_PAGESIZE);
        size_t size_aligned;
        CUdeviceptr p;
        p = (CUdeviceptr)((uintptr_t)buf & ~(page_size - 1));
        size_aligned =
            ((length + (uintptr_t)buf - (uintptr_t)p + page_size - 1) / page_size) * page_size;

        CUCHECKGOTO(table,
                    cuMemGetHandleForAddressRange(&handle->fd, (CUdeviceptr)p, size_aligned,
                                                  CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0),
                    status, out);

        mr = ftable->reg_dmabuf_mr(pd, 0, size_aligned, (uint64_t)p, handle->fd,
                                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                       IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC | ro_flag);
        if (mr == NULL) {
            close(handle->fd);
            goto reg_dmabuf_failure;
        }

        INFO(log_level, "ibv_reg_dmabuf_mr handle %p handle->mr %p", handle, handle->mr);
    } else {
    reg_dmabuf_failure:

        handle->fd = 0;
        mr = ftable->reg_mr(pd, buf, length,
                            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC | ro_flag);
        NVSHMEMI_NULL_ERROR_JMP(mr, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                "mem registration failed. Reason: %s\n", strerror(errno));
        INFO(log_level, "ibv_reg_mr handle %p handle->mr %p", handle, handle->mr);
    }

    handle->buf = buf;
    handle->lkey = mr->lkey;
    handle->rkey = mr->rkey;
    handle->mr = mr;
    handle->local_only = local_only;

out:
    return status;
}

int nvshmemt_ib_common_release_mem_handle(struct nvshmemt_ibv_function_table *ftable,
                                          nvshmem_mem_handle_t *mem_handle, int log_level) {
    int status = 0;
    struct nvshmemt_ib_common_mem_handle *handle =
        (struct nvshmemt_ib_common_mem_handle *)mem_handle;

    INFO(log_level, "ibv_dereg_mr handle %p handle->mr %p", handle, handle->mr);
    if (handle->mr) {
        status = ftable->dereg_mr((struct ibv_mr *)handle->mr);
        if (handle->fd) close(handle->fd);
    }
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_dereg_mr failed \n");

out:
    return status;
}
