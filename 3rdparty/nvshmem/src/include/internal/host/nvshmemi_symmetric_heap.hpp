/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef NVSHMEMI_SYMMETRIC_HEAP_HPP
#define NVSHMEMI_SYMMETRIC_HEAP_HPP

#include <climits>
#include <cstdlib>
#include <cstdbool>
#include <cuda.h>
#include <memory>
#include <tuple>
#include <vector>
#include "internal/host/nvshmem_internal.h"
#include "internal/host/nvshmemi_types.h"
#include "internal/host/util.h"
#include "internal/host_transport/nvshmemi_transport_defines.h"
#include "internal/host_transport/cudawrap.h"

/// Forward declarations for future friends
class nvshmemi_mem_p2p_transport;
class nvshmemi_mem_remote_transport;

enum { NVSHMEMX_MALLOC = 0, NVSHMEMX_CALLOC, NVSHMEMX_ALIGN, NVSHMEMX_ALLOC_MAX };

#define NVSHMEMI_SYMMETRIC_HEAP_OFFSET(base, off) (void *)((uint8_t *)(base) + off)

/**
 * This class manages symmetric heap per kind. Today, we support global heap kind for all teams. To
 * support multiple heap kinds, we must create multiple instances of this class for the desired
 * team. Class hierarchy is defined here to allow for sharing of common code as much as possible and
 * only bifurcating when there is a functional/behavior difference by memory kind.
 *
 *                              nvshmemi_symmetric_heap
 *                    -------------------------------------------------
 *                  |                                                   |
 *    nvshmemi_symmetric_heap_static                     nvshmemi_symmetric_heap_dynamic
 *     |                         |                                      |
 * sysmem_static              vidmem_static                     vidmem_dynamic
 *       |                         |                                    |
 *      SHM                      PINNED                                VMM
 *
 * Supported memory kinds: sysmem (linux shm), vidmem (cudaMalloc), vidmem (cuMemCreate)
 */

class nvshmemi_symmetric_heap {
   public:
    explicit nvshmemi_symmetric_heap(nvshmemi_state_t *state) noexcept : state_(state) {}
    virtual ~nvshmemi_symmetric_heap();

    /** Getters and Setters of protected members */
    size_t get_mem_granularity() const { return mem_granularity_; }
    size_t get_log2_cumem_granularity() const { return log2_mem_granularity_; }
    uint64_t get_reserve_size(void) const { return reserved_heap_size_; }
    size_t get_physical_heap_size(void) const { return physical_heap_size_; }
    CUmemAllocationHandleType get_mem_handle_type(void) { return mem_handle_type_; }
    bool is_cuda_mem_handle_type_ipc(void) const {
        return (mem_handle_type_ == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
    }

    bool is_cuda_mem_handle_type_fabric(void) const {
        return (mem_handle_type_ == CU_MEM_HANDLE_TYPE_FABRIC);
    }

    /** Common to all memory kinds */
    /**
     * This function will statically reserve heap memory based on memory kind in child class
     * For dynamic vidmem types, only virtual memory is reserved and mspace is initialized
     * For static vidmem and sysmem, virtual and physical memory is reserved/pre-allocated and
     * mspace is initialized
     *
     * @param void
     * @return On success, return 0 and on failure return non-zero NVSHMEM internal error code.
     */
    virtual int reserve_heap(void) = 0;

    virtual int setup_symmetric_heap() = 0;
    virtual int cleanup_symmetric_heap() = 0;

    /**
     * Given an address, size, pe index, and transport index, retrieve the corresponding
     * transport handle.
     */
    inline nvshmem_mem_handle *get_transport_mem_handle(void *addr, size_t *len, int pe,
                                                        int transport_idx);

    inline size_t get_mem_handle_addr_offset(void *addr);

    void *get_base() { return heap_base_; }
    void **get_local_pe_base() { return peer_heap_base_p2p_; }
    void *get_global_base() { return global_heap_base_; }
    void **get_remote_pe_base() { return peer_heap_base_remote_; }
    size_t get_size() { return heap_size_; }

    /** Top-level public facing functions */
    virtual void *heap_malloc(size_t size);
    virtual void *heap_calloc(size_t size, size_t count);
    virtual void *heap_align(size_t size, size_t alignment);
    virtual void heap_deallocate(void *ptr);

    /**
     * NVLS specific member functions - only concretized in dynamic heaps
     */
    virtual int nvls_create_heap_memory_by_team(nvshmemi_team_t *team) = 0;
    virtual int nvls_bind_heap_memory_by_team(nvshmemi_team_t *team) = 0;
    virtual int nvls_map_heap_memory_by_team(nvshmemi_team_t *team) = 0;
    virtual void nvls_unmap_heap_memory_by_team(nvshmemi_team_t *team) = 0;
    virtual void nvls_unbind_heap_memory_by_team(nvshmemi_team_t *team) = 0;

   private:
    /**
     * This function will finalize mspace container managing virtual address range
     * defined by device state heap base/size
     *
     * @param void
     * @return On success, return 0 and on failure return non-zero NVSHMEM internal error code.
     */
    friend class nvshmemi_mem_p2p_transport;     // friend class declaration
    friend class nvshmemi_mem_remote_transport;  // friend class declaration

   protected:
    nvshmemi_mem_remote_transport *get_remoteref(void) { return (remote_ref_); }
    nvshmemi_mem_p2p_transport *get_p2pref(void) { return (p2p_ref_); }
    nvshmemi_state_t *get_state() { return state_; }

    void set_p2p_transport(nvshmemi_mem_p2p_transport *obj) { p2p_ref_ = obj; }
    void set_remote_transport(nvshmemi_mem_remote_transport *obj) { remote_ref_ = obj; }
    void set_mem_handle_type(CUmemAllocationHandleType type) { mem_handle_type_ = type; }

    virtual void *allocate_symmetric_memory(size_t size, size_t count, size_t alignment,
                                            int type) = 0;

    void inc_heap_handle_cache(void) { heap_handle_cache_++; }
    bool empty_heap_handle_cache(void) { return heap_handle_cache_ == 0; }

    /**
     * Given a buf, size address range, map the heap into PE address space
     */
    virtual int map_heap_memory(void *buf, size_t size);
    /**
     * Given a buffer, size and input memory handle, register the heap into PE address space
     */
    virtual int register_heap_memory(nvshmem_mem_handle_t *input, void *buffer, size_t size);
    /**
     * Given a buffer, size, index to valid transport and PE#, map the buffer range into target PE
     * address space
     */
    virtual int map_heap_chunk(int pe_id, int transport_idx, char *buf = nullptr,
                               size_t size = 0) = 0;
    /**
     * Given a buffer, size and input memory handle, register the chunk into PE address space
     * across initialized mem transports
     */
    virtual int register_heap_chunk(nvshmem_mem_handle_t *mem_handle, void *buf, size_t size) = 0;
    /**
     * Given an collection of local memory handles across all PEs, establish pairwise memory handles
     * for processes connected over p2p transport
     */
    virtual int exchange_heap_memory_handle(nvshmem_mem_handle_t *local_handles) = 0;

    /**
     * Given a peer mem handle, import the buffer range to target buf object
     */
    virtual int import_memory(nvshmem_mem_handle_t *peer_handle, void **buf, size_t length = 0) = 0;
    /**
     * Given a buf, release and unmap the heap from PE address space
     */
    virtual int release_memory(void *buf, size_t size = 0) = 0;

    /* internal allocation functions */
    virtual void *heap_allocate(size_t size, size_t count, size_t alignment, int type);

    /**
     * Given a value, this API will calculate collectively from all PEs in the team if memory
     * is symmetric with respect to value passed into this API.
     *
     * @param value     templated value to compare for all PEs
     * @return On success, return 0 and on failure return 1
     */
    template <typename T>
    int is_symmetric(T value);

    /**
     * This function will initialize mspace container for managing virtual address range
     */
    virtual int setup_mspace() = 0;

    /**
     * This function will destroy the mspace container initialized in setup_mspace()
     */
    int cleanup_mspace(void);

    /**
     * Given an address and size,
     */
    virtual void update_idx_in_handle(void *addr, size_t size);

    /**
     * Given a mem_granualarity, this API will compute heap size attributes such as heapextra
     * alignbytes and logarithmic2 value of mem_granularity
     */
    void set_heap_size_attr(size_t memgran, size_t *extra, size_t *align, size_t *log_memgran);
    /**
     * This function will allgather the base address for heap_base for p2p/remote transport
     */
    int allgather_peer_base(void);
    /**
     * Allocate virtual memory chunk from the previously init mspace
     */
    void *allocate_virtual_memory_from_mspace(size_t size, size_t count, size_t alignment,
                                              int type);
    nvshmemi_state_t *state_ = nullptr;  // store a reference of device state instance
    CUmemAllocationHandleType mem_handle_type_ = CU_MEM_HANDLE_TYPE_NONE;
    size_t mem_granularity_ = 0;
    size_t log2_mem_granularity_ = 0;
    size_t physical_heap_size_ = 0;
    size_t heap_size_ = 0;
    uint64_t reserved_heap_size_ = 0;
    void *global_heap_base_ = nullptr;
    void *heap_base_ = nullptr;
    void **peer_heap_base_remote_ = nullptr;
    void **peer_heap_base_p2p_ = nullptr;
    int heap_handle_cache_ = 0;
    nvshmemi_mem_remote_transport *remote_ref_ =
        nullptr;                                     // holds an instance of remote abstraction
    nvshmemi_mem_p2p_transport *p2p_ref_ = nullptr;  // holds an instance of memp2p abstraction
    mspace *heap_mspace_ = nullptr;
    std::vector<std::vector<nvshmem_mem_handle>> handles_;
    std::vector<std::tuple<size_t, void *, size_t>> idx_in_handles_;
};

inline nvshmem_mem_handle *nvshmemi_symmetric_heap::get_transport_mem_handle(void *addr,
                                                                             size_t *len, int pe,
                                                                             int transport_idx) {
    size_t addr_idx;
    size_t handle_idx;
    size_t handle_size;
    size_t handle_sub_index;
    size_t offset;

    void *handle_start_addr;

    if (addr < heap_base_ || addr > (char *)heap_base_ + heap_size_) {
        return NULL;
    }

    offset = (char *)addr - (char *)heap_base_;
    addr_idx = offset >> log2_mem_granularity_;

    handle_idx = std::get<0>(idx_in_handles_[addr_idx]);
    handle_sub_index = pe * get_state()->num_initialized_transports + transport_idx;
    handle_start_addr = std::get<1>(idx_in_handles_[addr_idx]);
    handle_size = std::get<2>(idx_in_handles_[addr_idx]);

    if (len) {
        *len = handle_size - ((char *)addr - (char *)handle_start_addr);
    }
    return &handles_[handle_idx][handle_sub_index];
}

inline size_t nvshmemi_symmetric_heap::get_mem_handle_addr_offset(void *addr) {
    size_t addr_idx;
    size_t heap_offset;
    size_t offset;
    void *start_addr;

    heap_offset = (char *)addr - (char *)heap_base_;
    addr_idx = heap_offset >> log2_mem_granularity_;
    start_addr = std::get<1>(idx_in_handles_[addr_idx]);

    offset = (char *)addr - (char *)start_addr;

    return offset;
}

class nvshmemi_symmetric_heap_static : public nvshmemi_symmetric_heap {
   public:
    explicit nvshmemi_symmetric_heap_static(nvshmemi_state_t *state) noexcept;
    virtual ~nvshmemi_symmetric_heap_static() = default;

    virtual int reserve_heap(void);
    virtual int setup_symmetric_heap(void);
    virtual int cleanup_symmetric_heap(void);

   protected:
    virtual int allocate_heap_memory() = 0;
    virtual int free_heap_memory(void *addr) = 0;

    virtual void *allocate_symmetric_memory(size_t size, size_t count, size_t alignment, int type);

    virtual int register_heap_memory_handle(nvshmem_mem_handle_t *local, int transport_idx,
                                            nvshmem_mem_handle_t *in, void *buf, size_t size,
                                            nvshmem_transport_t current) = 0;
    virtual int register_heap_chunk(nvshmem_mem_handle_t *mem_handle, void *buf, size_t size);
    virtual int setup_mspace();

    /**
     * Stubbed functions that are not supported by static heaps
     */
    int nvls_create_heap_memory_by_team(nvshmemi_team_t *team) {
        assert(0);
        return (NVSHMEMX_ERROR_NOT_SUPPORTED);
    }
    int nvls_bind_heap_memory_by_team(nvshmemi_team_t *team) {
        assert(0);
        return (NVSHMEMX_ERROR_NOT_SUPPORTED);
    }

    int nvls_map_heap_memory_by_team(nvshmemi_team_t *team) {
        assert(0);
        return (NVSHMEMX_ERROR_NOT_SUPPORTED);
    }

    void nvls_unmap_heap_memory_by_team(nvshmemi_team_t *team) {
        assert(0);
        return;
    }

    void nvls_unbind_heap_memory_by_team(nvshmemi_team_t *team) {
        assert(0);
        return;
    }

    /**
     * Given a buf, length, export the buffer range to target mem handle
     */
    virtual int export_memory(nvshmem_mem_handle_t *mem_handle, void *buf, size_t length) = 0;

   private:
};

class nvshmemi_symmetric_heap_dynamic : public nvshmemi_symmetric_heap {
   public:
    explicit nvshmemi_symmetric_heap_dynamic(nvshmemi_state_t *state) noexcept;
    virtual ~nvshmemi_symmetric_heap_dynamic() = default;

   protected:
    /* Stubbed implementation, accessible in derived class only */
    virtual int allocate_physical_memory_to_heap(size_t size) {
        return (NVSHMEMX_ERROR_NOT_SUPPORTED);
    }
    virtual int export_memory(nvshmem_mem_handle_t *mem_handle,
                              nvshmem_mem_handle_t *mem_handle_in) = 0;
    virtual int register_heap_chunk(nvshmem_mem_handle_t *mem_handle, void *buf, size_t size);
    virtual int setup_mspace();

   private:
};

class nvshmemi_symmetric_heap_vidmem_static : public nvshmemi_symmetric_heap_static {
   public:
    explicit nvshmemi_symmetric_heap_vidmem_static(nvshmemi_state_t *state) noexcept
        : nvshmemi_symmetric_heap_static(state) {}
    virtual ~nvshmemi_symmetric_heap_vidmem_static() = default;
};

class nvshmemi_symmetric_heap_vidmem_static_pinned final
    : public nvshmemi_symmetric_heap_vidmem_static {
   public:
    explicit nvshmemi_symmetric_heap_vidmem_static_pinned(nvshmemi_state_t *state) noexcept
        : nvshmemi_symmetric_heap_vidmem_static(state) {}
    ~nvshmemi_symmetric_heap_vidmem_static_pinned() = default;

   protected:
    int allocate_heap_memory();
    int free_heap_memory(void *addr);
    int exchange_heap_memory_handle(nvshmem_mem_handle_t *local_handles);
    int register_heap_memory_handle(nvshmem_mem_handle_t *local, int transport_idx,
                                    nvshmem_mem_handle_t *in, void *buf, size_t size,
                                    nvshmem_transport_t current);
    int map_heap_chunk(int pe_id, int transport_idx, char *buf = nullptr, size_t size = 0);

    int export_memory(nvshmem_mem_handle_t *mem_handle, void *buf, size_t length);
    int import_memory(nvshmem_mem_handle_t *mem_handle, void **buf, size_t length = 0);
    int release_memory(void *buf, size_t size = 0);
};

class nvshmemi_symmetric_heap_vidmem_dynamic : public nvshmemi_symmetric_heap_dynamic {
   public:
    explicit nvshmemi_symmetric_heap_vidmem_dynamic(nvshmemi_state_t *state) noexcept
        : nvshmemi_symmetric_heap_dynamic(state) {}
    virtual ~nvshmemi_symmetric_heap_vidmem_dynamic() = default;
};

class nvshmemi_symmetric_heap_vidmem_dynamic_vmm final
    : public nvshmemi_symmetric_heap_vidmem_dynamic {
   public:
    explicit nvshmemi_symmetric_heap_vidmem_dynamic_vmm(nvshmemi_state_t *state) noexcept
        : nvshmemi_symmetric_heap_vidmem_dynamic(state) {}
    ~nvshmemi_symmetric_heap_vidmem_dynamic_vmm() = default;
    int reserve_heap(void);
    int setup_symmetric_heap(void);
    int cleanup_symmetric_heap(void);
    /* Operates on a current state of the entire heap */
    int nvls_create_heap_memory_by_team(nvshmemi_team_t *team);
    int nvls_bind_heap_memory_by_team(nvshmemi_team_t *team);
    int nvls_map_heap_memory_by_team(nvshmemi_team_t *team);
    void nvls_unmap_heap_memory_by_team(nvshmemi_team_t *team);
    void nvls_unbind_heap_memory_by_team(nvshmemi_team_t *team);

   protected:
    CUmemGenericAllocationHandle get_cumem_handle_ptr(int i) {
        return (std::get<0>(cumem_handles_[i]));
    }
    off_t get_cumem_handle_alloc_offset(int i) { return std::get<1>(cumem_handles_[i]); }
    off_t get_cumem_handle_mmap_offset(int i) { return std::get<2>(cumem_handles_[i]); }
    size_t get_cumem_handle_mmap_size(int i) { return std::get<3>(cumem_handles_[i]); }
    size_t get_cumem_handle_size(void) { return cumem_handles_.size(); }
    void print_cumem_handles(void);
    int exchange_heap_memory_handle(nvshmem_mem_handle_t *local_handles);
    int map_heap_chunk(int pe_id, int transport_idx, char *buf, size_t size);
    int import_memory(nvshmem_mem_handle_t *mem_handle, void **buf, size_t length);
    int export_memory(nvshmem_mem_handle_t *mem_handle, nvshmem_mem_handle_t *mem_handle_in);
    int release_memory(void *buf, size_t size);
    void *allocate_symmetric_memory(size_t size, size_t count, size_t alignment, int type);
    int allocate_physical_memory_to_heap(size_t size);
    int nvls_broadcast_heap_handle_fabric(char *shareable_handle, size_t length, int root,
                                          nvshmemi_team_t *team);
    int nvls_broadcast_heap_handle_ipc(char *shareable_handle, int root, nvshmemi_team_t *team);
    int nvls_broadcast_heap_handle_by_team(char *shareable_handle, size_t length,
                                           nvshmemi_team_t *team);
    /* Operates on a given allocation request of size mem_size */
    int nvls_create_heap_memory_by_size(nvshmemi_team_t *team, uint64_t mem_size);
    int nvls_bind_heap_memory_by_size(nvshmemi_team_t *team, nvshmem_mem_handle_t *mem_handle,
                                      off_t mc_offset, off_t mmap_offset, size_t mmap_size);
    int nvls_map_heap_memory_by_size(nvshmemi_team_t *team, uint64_t mem_size, off_t mmap_offset,
                                     off_t mc_offset);
    int nvls_create_heap_memory(uint64_t mem_size);
    int nvls_bind_heap_memory(nvshmem_mem_handle_t *mem_handle, off_t mc_offset, off_t mmap_offset,
                              size_t mmap_size);
    int nvls_map_heap_memory(uint64_t mem_size, off_t mmap_offset, off_t mc_offset);

   private:
    void set_cuda_mem_prop(__attribute__((unused)) void *prop, int mem_handle_type) {
        CUmemAllocationProp *memprop = (CUmemAllocationProp *)(prop);
        (*memprop).type = CU_MEM_ALLOCATION_TYPE_PINNED;
        (*memprop).location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        (*memprop).location.id = static_cast<int>(this->state_->device_id);
        (*memprop).requestedHandleTypes = (CUmemAllocationHandleType)(mem_handle_type);
        (*memprop).allocFlags.gpuDirectRDMACapable = 1;
        return;
    }

    std::vector<std::tuple<CUmemGenericAllocationHandle, off_t, off_t, size_t>> cumem_handles_;
};

class nvshmemi_symmetric_heap_sysmem_static : public nvshmemi_symmetric_heap_static {
   public:
    explicit nvshmemi_symmetric_heap_sysmem_static(nvshmemi_state_t *state) noexcept
        : nvshmemi_symmetric_heap_static(state) {}
    virtual ~nvshmemi_symmetric_heap_sysmem_static() = default;
};

class nvshmemi_symmetric_heap_sysmem_static_shm final
    : public nvshmemi_symmetric_heap_sysmem_static {
   public:
    explicit nvshmemi_symmetric_heap_sysmem_static_shm(nvshmemi_state_t *state) noexcept
        : nvshmemi_symmetric_heap_sysmem_static(state) {}
    ~nvshmemi_symmetric_heap_sysmem_static_shm() = default;
    static void atexit_heap_handler(void) {
        // Iterate over all objects and close any stale fd
        for (auto i = 0U; i < nvshmemi_symmetric_heap_sysmem_static_shm::infos_.size(); i++) {
            INFO(NVSHMEM_MEM, "Closing file descriptor: %d for sym heap\n",
                 nvshmemi_symmetric_heap_sysmem_static_shm::infos_[i].shm_fd);
            close(nvshmemi_symmetric_heap_sysmem_static_shm::infos_[i].shm_fd);
        }
    }
    int register_heap_memory_handle(nvshmem_mem_handle_t *local, int transport_idx,
                                    nvshmem_mem_handle_t *in, void *buf, size_t size,
                                    nvshmem_transport_t current);

   protected:
    int allocate_heap_memory();
    int free_heap_memory(void *addr);

    int exchange_heap_memory_handle(nvshmem_mem_handle_t *local_handles);
    int map_heap_chunk(int pe_id, int transport_idx, char *buf = nullptr, size_t size = 0);

    /** Stubbed out for P2P transport as export is non-action, import is done at allocation time,
     * release is done at cleanup time **/
    int export_memory(nvshmem_mem_handle_t *mem_handle, void *buf, size_t length);
    int import_memory(nvshmem_mem_handle_t *mem_handle, void **buf, size_t length = 0);
    int release_memory(void *buf, size_t size = 0);

   private:
    char heap_name_[NAME_MAX] = {0};
    nvshmemi_shared_memory_info_t heap_info_;
    // shared global storage for all objects
    static std::vector<nvshmemi_shared_memory_info_t> infos_;
};

#endif /* SYMMETRIC_HEAP_HPP */
