/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include <assert.h>                                              // for assert
#include <cuda.h>                                                // for CUDA_SUCCESS
#include <cuda_runtime.h>                                        // for cudaGetDevice
#include <stdint.h>                                              // for uintptr_t
#include <string.h>                                              // for memcpy, NULL
#include <sys/types.h>                                           // for off_t
#include "device_host/nvshmem_types.h"                           // for nvshmemi_team_t
#include "non_abi/nvshmemx_error.h"                              // for NVSHMEMI_NE_ERR...
#include "internal/host/debug.h"                                 // for WARN
#include "internal/host/nvshmem_internal.h"                      // for nvshmemi_cuda_syms
#include "internal/host/nvshmemi_nvls_rsc.hpp"                   // for nvshmemi_nvls_rsc
#include "internal/host/nvshmemi_symmetric_heap.hpp"             // for nvshmemi_symmet...
#include "internal/host/nvshmemi_mem_transport.hpp"              // for nvshmemi_mem...
#include "internal/host/sockets.h"                               // for ipcCl...
#include "internal/host/util.h"                                  // for CUDA_RUNTIME_CHECK
#include "internal/host_transport/cudawrap.h"                    // for CUPFN, nvshmemi...
#include "internal/host_transport/nvshmemi_transport_defines.h"  // for nvshmem_mem_han...

namespace nvls {

void nvshmemi_nvls_rsc::set_group_prop(uint64_t mem_size) {
    /* align sizes to mcGran */
    prop_.size = mem_size;
    prop_.numDevices = n_devices_;
    prop_.handleTypes = alloc_mem_handle_type_;
    prop_.flags = 0;
    return;
}

/**
 * Assumptions
 * This support assumes a single heap in NVSHMEM.
 * For multiple heaps, additional support to qualify the memprop max selection is needed
 */

nvshmemi_nvls_rsc::nvshmemi_nvls_rsc(nvshmemi_team_t *team, nvshmemi_state_t *state) {
    int cuda_dev;
    int status = -1;
    CUDA_RUNTIME_CHECK(cudaGetDevice(&cuda_dev));
    status = CUPFN(nvshmemi_cuda_syms, cuDeviceGet(&current_dev_, cuda_dev));
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "cuDeviceGet failed\n");

    if (team->size < 2) {
        status = NVSHMEMX_ERROR_NOT_SUPPORTED;
        WARN("Unsupported NVLS team size\n");
        goto out;
    }

    if (!state->is_platform_nvls) {
        status = NVSHMEMX_ERROR_NOT_SUPPORTED;
        WARN("Missing NVLS platform support. Skipping NVLS initialization\n");
        goto out;
    }

    state_ = state;
    alloc_granularity_ = state->heap_obj->get_mem_granularity();
    alloc_mem_handle_type_ = state->heap_obj->get_mem_handle_type();
    virt_alloc_size_ = state->heap_obj->get_reserve_size();
    n_devices_ = team->size;

out:
    if (status) throw nvshmemi_nvls_exception("Unable to initialize NVLS resource\n");
}

void nvshmemi_nvls_rsc::invalidate_rsc(void) {
    state_ = nullptr;
    alloc_granularity_ = 0;
    alloc_mem_handle_type_ = CU_MEM_HANDLE_TYPE_NONE;
    prop_ = {};
    virt_alloc_size_ = 0;
    n_devices_ = 0;
    NVSHMEMU_FOR_EACH(i, cumc_handles_.size()) {
        int status = CUPFN(nvshmemi_cuda_syms, cuMemRelease(cumc_handles_[i].first));
        INFO(NVSHMEM_TEAM, "Releasing multicast group handle %lld on GPU device %d (status = %d)\n",
             cumc_handles_[i].first, current_dev_, status);
    }

    current_dev_ = (CUdevice)(-1);
    cumc_handles_.clear();
    mc_base_ptr_ = NULL;
    mc_granularity_ = 0;
    rsc_refcount_ = 0;
    owner_team_ = NVSHMEM_TEAM_INVALID;
}

nvshmemi_nvls_rsc::~nvshmemi_nvls_rsc(void) { invalidate_rsc(); }

int nvshmemi_nvls_rsc::export_group(uint64_t mem_size, char *shareable_handle) {
    int status = -1;
    nvshmemi_state_t *state = get_state();
    assert(shareable_handle != nullptr);
    CUmemGenericAllocationHandle mc_handle;
    set_group_prop(mem_size);
    status = CUPFN(
        nvshmemi_cuda_syms,
        cuMulticastGetGranularity(&mc_granularity_, &prop_, CU_MULTICAST_GRANULARITY_RECOMMENDED));

    INFO(NVSHMEM_TEAM,
         "GPU [%d] mem_granularity: %zu mem_handle_type: %d n_devices: %zu "
         "mc_granularity: %zu virtual_alloc_size: %ld\n",
         current_dev_, alloc_granularity_, alloc_mem_handle_type_, n_devices_, mc_granularity_,
         virt_alloc_size_);

    status = CUPFN(nvshmemi_cuda_syms, cuMulticastCreate(&mc_handle, &prop_));
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "cuMulticastCreate failed \n");

    INFO(NVSHMEM_TEAM, "Creating mcHandle %lld on GPU device %d of size: %zu\n", mc_handle,
         current_dev_, mem_size);
    cumc_handles_.push_back(std::make_pair(mc_handle, mem_size));
    status = CUPFN(nvshmemi_cuda_syms, cuMemExportToShareableHandle(shareable_handle, mc_handle,
                                                                    alloc_mem_handle_type_, 0));

    if (state->heap_obj->is_cuda_mem_handle_type_fabric()) {
        NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                              "cuMemExportToShareableHandle failed for fabric handles\n");
        INFO(NVSHMEM_TEAM, "Exporting mcHandle %lld via FH on GPU device %d\n", mc_handle,
             current_dev_);

    } else {
        NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                              "cuMemExportToShareableHandle failed for ipc handles\n");
        INFO(NVSHMEM_TEAM, "Exporting mcHandle %lld via POSIX FD on GPU device %d\n", mc_handle,
             current_dev_);
        status = 0;
    }

out:
    return (status);
}

int nvshmemi_nvls_rsc::import_group(char *shareable_handle, CUmemGenericAllocationHandle *mc_handle,
                                    uint64_t mem_size) {
    nvshmemi_state_t *state = get_state();
    int status = -1;
    if (state->heap_obj->is_cuda_mem_handle_type_ipc()) {
        int fd = *(int *)shareable_handle;
        status = CUPFN(nvshmemi_cuda_syms,
                       cuMemImportFromShareableHandle(mc_handle, (void *)(uintptr_t)fd,
                                                      alloc_mem_handle_type_));
        NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                              "cuMemImportFromShareableHandle failed for ipc handles\n");
        INFO(NVSHMEM_TEAM, "Importing mcHandle %lld via POSIX FD on GPU device %d\n", *mc_handle,
             current_dev_);
        close(fd);
    } else {
        status = CUPFN(nvshmemi_cuda_syms,
                       cuMemImportFromShareableHandle(mc_handle, (void *)shareable_handle,
                                                      alloc_mem_handle_type_));
        NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                              "cuMemImportFromShareableHandle failed for fabric handles \n");
        INFO(NVSHMEM_TEAM, "Importing mcHandle %lld via FH on GPU device %d\n", *mc_handle,
             current_dev_);
    }

    /* Add peer handle to mc group array */
    cumc_handles_.push_back(std::make_pair(*mc_handle, mem_size));

out:
    return (status);
}

int nvshmemi_nvls_rsc::subscribe_group(CUmemGenericAllocationHandle *mc_handle) {
    int status = -1;
    status = CUPFN(nvshmemi_cuda_syms, cuMulticastAddDevice(*mc_handle, current_dev_));
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "cuMulticastAddDevice failed \n");

    INFO(NVSHMEM_TEAM, "Adding mcHandle %lld to GPU device %d\n", *mc_handle, current_dev_);
out:
    return (status);
}

int nvshmemi_nvls_rsc::reserve_group_mem(void) {
    int status = -1;
    status = CUPFN(nvshmemi_cuda_syms,
                   cuMemAddressReserve((CUdeviceptr *)&mc_base_ptr_, virt_alloc_size_,
                                       alloc_granularity_, (CUdeviceptr)NULL, 0));
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "cuMemAddressReserve failed for mc base ptr \n");
    INFO(NVSHMEM_TEAM, "Reserving mc base ptr %p on GPU device %d\n", mc_base_ptr_, current_dev_);
out:
    return (status);
}

int nvshmemi_nvls_rsc::free_group_mem(void) {
    int status = -1;
    INFO(NVSHMEM_TEAM, "Freeing mc base ptr %p on GPU device %d\n", mc_base_ptr_, current_dev_);
    status =
        CUPFN(nvshmemi_cuda_syms, cuMemAddressFree((CUdeviceptr)mc_base_ptr_, virt_alloc_size_));
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "cuMemAddressFree failed for mc base ptr \n");
out:
    return (status);
}

/**
 * Assumptions:
 * (a) Caller must make sure that nvls rsc is bound to mem_handle on all GPUs in the team, so a
 * collective barrier needs to be called on the team, prior to binding memory. Else, a bind can
 * block if not all devices are added to the mc handle
 */
int nvshmemi_nvls_rsc::bind_group_mem(CUmemGenericAllocationHandle *mc_handle,
                                      nvshmem_mem_handle_t *mem_handle, size_t mem_size,
                                      off_t mem_offset, off_t mc_offset) {
    int status = -1;
    status =
        CUPFN(nvshmemi_cuda_syms,
              cuMulticastBindMem(*mc_handle, mc_offset, *(CUmemGenericAllocationHandle *)mem_handle,
                                 mem_offset, mem_size, 0));
    NVSHMEMI_NE_ERROR_JMP(
        status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
        "cuMulticastBindMem failed for mem_handle %p mc_offset %lx on device %d\n", mem_handle,
        mc_offset, current_dev_);
    INFO(NVSHMEM_TEAM,
         "Bind multicast group handle %lld to mem_handle %lld for mem size %zu, mem offset %zu, "
         "mc offset %lx\n on GPU device %d\n",
         *mc_handle, *(CUmemGenericAllocationHandle *)mem_handle, mem_size, mem_offset, mc_offset,
         current_dev_);
out:
    return (status);
}

int nvshmemi_nvls_rsc::unbind_group_mem(CUmemGenericAllocationHandle *mc_handle, off_t mc_offset,
                                        size_t mem_size) {
    int status = -1;
    status =
        CUPFN(nvshmemi_cuda_syms, cuMulticastUnbind(*mc_handle, current_dev_, mc_offset, mem_size));
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "cuMulticastUnbind failed for mc_offset %lx on device %d\n", mc_offset,
                          current_dev_);
    INFO(NVSHMEM_TEAM,
         "Unbind multicast group handle %lld for mem size %zu "
         "mc offset %lx\n on GPU device %d\n",
         *mc_handle, mem_size, mc_offset, current_dev_);
out:
    return (status);
}

int nvshmemi_nvls_rsc::map_group_mem(CUmemGenericAllocationHandle *mc_handle, size_t mem_size,
                                     off_t mem_offset, off_t mc_offset) {
    int status = -1;
    CUmemAccessDesc access;
    status = CUPFN(nvshmemi_cuda_syms,
                   cuMemMap((CUdeviceptr)NVSHMEMI_SYMMETRIC_HEAP_OFFSET(mc_base_ptr_, mc_offset),
                            mem_size, mem_offset, *mc_handle, 0));
    NVSHMEMI_NE_ERROR_JMP(
        status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
        "cuMemMap failed to map %ld bytes handle at address: %p offset %ld on device %d\n",
        mem_size, NVSHMEMI_SYMMETRIC_HEAP_OFFSET(mc_base_ptr_, mc_offset), mem_offset,
        current_dev_);

    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = current_dev_;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    status =
        CUPFN(nvshmemi_cuda_syms,
              cuMemSetAccess((CUdeviceptr)NVSHMEMI_SYMMETRIC_HEAP_OFFSET(mc_base_ptr_, mc_offset),
                             mem_size, (const CUmemAccessDesc *)&access, 1));
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "cuMemSetAccess failed for address: %p on device %d\n",
                          NVSHMEMI_SYMMETRIC_HEAP_OFFSET(mc_base_ptr_, mc_offset), current_dev_);

    INFO(NVSHMEM_TEAM,
         "Mapping multicast group handle %lld @ mc heap addr %lx for mem size %zu "
         "mc offset %lx mem offset %lx on GPU device %d\n",
         *mc_handle, NVSHMEMI_SYMMETRIC_HEAP_OFFSET(mc_base_ptr_, mc_offset), mem_size, mc_offset,
         mem_offset, current_dev_);
out:
    return (status);
}

int nvshmemi_nvls_rsc::unmap_group_mem(off_t mc_offset, uint64_t mem_size) {
    int status = -1;
    status = CUPFN(
        nvshmemi_cuda_syms,
        cuMemUnmap((CUdeviceptr)NVSHMEMI_SYMMETRIC_HEAP_OFFSET(mc_base_ptr_, mc_offset), mem_size));
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "cuMemUnmap failed to map %ld bytes at address: %p\n", mem_size,
                          NVSHMEMI_SYMMETRIC_HEAP_OFFSET(mc_base_ptr_, mc_offset));

    INFO(NVSHMEM_TEAM,
         "Unmapping mc heap addr %lx for mem size %zu "
         "mc offset %lx\n on GPU device %d\n",
         NVSHMEMI_SYMMETRIC_HEAP_OFFSET(mc_base_ptr_, mc_offset), mem_size, mc_offset,
         current_dev_);
out:
    return (status);
}

}  // namespace nvls
