/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef NVSHMEMI_NVLS_RSC_HPP
#define NVSHMEMI_NVLS_RSC_HPP

#include <stdlib.h>
#include <vector>
#include <array>
#include <iostream>
#include <utility>
#include "internal/host/nvshmem_internal.h"                      // for nvshmemi_state
#include "internal/host_transport/nvshmemi_transport_defines.h"  // for nvshmem_mem_handle_t
#include "internal/host_transport/cudawrap.h"                    // for CUmemGenericAllocationHandle
#include "internal/host/sockets.h"                               // for ipcCl...

namespace nvls {

/**
 * NVLINK SHARP Resource Exception
 * (a) Manages exception handling
 */
class nvshmemi_nvls_exception final : public std::exception {
   public:
    nvshmemi_nvls_exception(const char *message) : msg_(message) {}
    const char *what() const noexcept { return msg_.c_str(); }

   private:
    std::string msg_;
};

/**
 * NVLINK SHARP Resource Manager
 * (a) Manages one or more multicast group per team via mc handles interface
 * (b) Manages Subscription of MC mappings for one or more multicast group
 */
class nvshmemi_nvls_rsc final {
   public:
    /* ctor/dtor */
    explicit nvshmemi_nvls_rsc(nvshmemi_team_t *team, nvshmemi_state_t *state);
    ~nvshmemi_nvls_rsc(void);
    /* Public setter/getter */
    nvshmemi_state_t *get_state(void) const { return state_; }
    size_t get_refcount(void) const { return rsc_refcount_; }
    void add_refcount(void) { rsc_refcount_++; }
    void del_refcount(void) { rsc_refcount_--; }
    CUmemGenericAllocationHandle *get_mc_handle_ptr(int idx) { return &cumc_handles_[idx].first; }
    uint64_t get_mc_handle_ptr_size(int idx) { return cumc_handles_[idx].second; }
    int get_owner(void) const { return owner_team_; }
    size_t get_mc_handle_size(void) const { return cumc_handles_.size(); }
    inline void *get_mc_base() const { return mc_base_ptr_; }
    void assign_owner(nvshmemi_team_t *team) { owner_team_ = team->team_idx; }
    void release_owner(void) { owner_team_ = NVSHMEM_TEAM_INVALID; }
    bool is_owner(nvshmemi_team_t *other) {
        return (owner_team_ != NVSHMEM_TEAM_INVALID && owner_team_ == other->team_idx);
    }

    /* Operate per mc handle */
    int export_group(uint64_t mem_size, char *shareable_handle);
    int import_group(char *shareable_handle, CUmemGenericAllocationHandle *mc_handle,
                     uint64_t mem_size);
    int subscribe_group(CUmemGenericAllocationHandle *mc_handle);
    int reserve_group_mem(void);
    int free_group_mem(void);
    int bind_group_mem(CUmemGenericAllocationHandle *mc_handle, nvshmem_mem_handle_t *mem_handle,
                       size_t mem_size, off_t mem_offset, off_t mc_offset);
    int map_group_mem(CUmemGenericAllocationHandle *mc_handle, size_t mem_size, off_t mem_offset,
                      off_t mc_offset);
    int unbind_group_mem(CUmemGenericAllocationHandle *mc_handle, off_t mc_offset, size_t mem_size);
    int unmap_group_mem(off_t mc_offset, size_t mem_size);

   private:
    void invalidate_rsc(void);
    void set_group_prop(uint64_t mem_size);
    nvshmemi_state_t *state_;
    unsigned int n_devices_ = 0;
    CUmemAllocationHandleType alloc_mem_handle_type_ = CU_MEM_HANDLE_TYPE_NONE;
    uint64_t virt_alloc_size_ = 0;
    size_t alloc_granularity_ = 0;
    void *mc_base_ptr_ = NULL; /* This MC heap supports monotonically increasing UC memory mappings,
                                  that don't shrink over time */
    size_t mc_granularity_ = 0;
    size_t rsc_refcount_ = 0;
    CUmulticastObjectProp prop_ = {};
    CUdevice current_dev_;
    std::vector<std::pair<CUmemGenericAllocationHandle, uint64_t>> cumc_handles_;
    int owner_team_ = NVSHMEM_TEAM_INVALID;
};

}  // namespace nvls

#endif /*! NVSHMEMI_NVLS_RSC_HPP */
