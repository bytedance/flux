/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef NVSHMEMI_MEM_TRANSPORT_HPP
#define NVSHMEMI_MEM_TRANSPORT_HPP

#include <cstdlib>
#include <cstdbool>
#include <climits>
#include <memory>
#include <map>
#include <algorithm>
#include "internal/host/nvshmem_internal.h"
#include "internal/host/util.h"
#include "internal/host/nvshmemi_symmetric_heap.hpp"
#include "internal/host/nvmlwrap.h"

static_assert(sizeof(nvshmem_mem_handle_t) % sizeof(uint64_t) == 0,
              "nvshmem_mem_handle_t size is not a multiple of 8B");

/**
 * This is a singleton class managing memory kind specific business logic for p2p transport
 */
class nvshmemi_mem_p2p_transport final {
   public:
    ~nvshmemi_mem_p2p_transport();
    nvshmemi_mem_p2p_transport(const nvshmemi_mem_p2p_transport &obj) = delete;
    static nvshmemi_mem_p2p_transport *get_instance(int mype, int npes) {
        if (p2p_objref_ == nullptr) {
            p2p_objref_ = new nvshmemi_mem_p2p_transport(mype, npes);
            return p2p_objref_;
        } else {
            return p2p_objref_;
        }
    }

    void print_mem_handle(int pe_id, int transport_idx, nvshmemi_symmetric_heap &obj);

    struct nvml_function_table *get_nvml_ftable(void) {
        return &nvml_ftable_;
    }
    CUmemAllocationHandleType get_mem_handle_type(void) const { return nvshmemi_mem_handle_type_; }
    bool is_mnnvl_fabric(void) const { return nvshmemi_has_mnnvl_fabric_; }
    bool is_initialized(void) const { return !errored_on_initialization_; }
    int create_proc_map(nvshmemi_symmetric_heap &obj);
    std::map<pid_t, int> get_proc_map(void) const { return proc_map_; }
    bool is_nvl_connected_pe(int pe) {
        /* Check if the peer GPU is connected via the MNNVL fabric */
        auto it =
            std::find(nvshmemi_nvl_connected_pes_.begin(), nvshmemi_nvl_connected_pes_.end(), pe);
        if (it != nvshmemi_nvl_connected_pes_.end()) return true;
        return false;
    }

   private:
    explicit nvshmemi_mem_p2p_transport(int mype, int npes);
    static nvshmemi_mem_p2p_transport *p2p_objref_;  // singleton instance
    std::map<pid_t, int> proc_map_;
    void *nvml_handle_ = nullptr;
    struct nvml_function_table nvml_ftable_;
    bool nvshmemi_has_mnnvl_fabric_ = false;
    std::vector<int> nvshmemi_nvl_connected_pes_;
    bool errored_on_initialization_ = true;
    CUmemAllocationHandleType nvshmemi_mem_handle_type_ = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
};

/**
 * This is a singleton class managing memory kind specific business logic for remote transport
 */
class nvshmemi_mem_remote_transport final {
   public:
    ~nvshmemi_mem_remote_transport() {
        if (remote_objref_ != nullptr) remote_objref_ = nullptr;
    }
    nvshmemi_mem_remote_transport(const nvshmemi_mem_remote_transport &obj) = delete;
    nvshmemi_mem_remote_transport(nvshmemi_mem_remote_transport &&obj) = delete;
    static nvshmemi_mem_remote_transport *get_instance(void) noexcept {
        if (remote_objref_ == nullptr) {
            remote_objref_ = new nvshmemi_mem_remote_transport();
            return remote_objref_;
        } else {
            return remote_objref_;
        }
    }

    int gather_mem_handles(nvshmemi_symmetric_heap &obj, uint64_t heap_offset, size_t size);
    /* On-demand registration and release of memory */
    int register_mem_handle(nvshmem_mem_handle_t *local_handles, int transport_idx,
                            nvshmem_mem_handle_t *in, void *buf, size_t size,
                            nvshmem_transport_t current);
    int release_mem_handles(nvshmem_mem_handle_t *handles, nvshmemi_symmetric_heap &obj);

    int is_mem_handle_null(nvshmem_mem_handle_t *handle) {
        NVSHMEMU_FOR_EACH(i, (sizeof(nvshmem_mem_handle_t) / sizeof(uint64_t))) {
            if (*((uint64_t *)handle + i) != (uint64_t)0) return 0;
        }

        return 1;
    }

   private:
    explicit nvshmemi_mem_remote_transport(void) noexcept {};
    static nvshmemi_mem_remote_transport *remote_objref_;  // singleton instance
};

#endif /* MEM_TRANSPORT_HPP */
