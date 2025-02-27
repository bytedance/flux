/*
 * Copyright (c) 2016-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include <assert.h>                                                        // for assert
#include <cuda.h>                                                          // for CUDA_...
#include <cuda_runtime.h>                                                  // for cudaFree
#include <driver_types.h>                                                  // for cudaH...
#include <ext/alloc_traits.h>                                              // for __all...
#include <stdint.h>                                                        // for uintp...
#include <stdio.h>                                                         // for size_t
#include <stdlib.h>                                                        // for calloc
#include <string.h>                                                        // for memset
#include <unistd.h>                                                        // for pid_t
#include <algorithm>                                                       // for max
#include <iosfwd>                                                          // for std
#include <map>                                                             // for map
#include <memory>                                                          // for alloc...
#include <tuple>                                                           // for tuple
#include <typeinfo>                                                        // for type_...
#include <utility>                                                         // for pair
#include <vector>                                                          // for vector
#include "device_host/nvshmem_types.h"                                     // for nvshm...
#include "device_host/nvshmem_common.cuh"                                  // for nvshm...
#include "host/nvshmem_api.h"                                              // for nvshm...
#include "host/nvshmemx_api.h"                                             // for nvshm...
#include "non_abi/nvshmemx_error.h"                                        // for NVSHM...
#include "non_abi/nvshmem_build_options.h"                                 // IWYU pragma: keep
#include "device_host_transport/nvshmem_common_transport.h"                // for g_elem_t
#include "internal/host/debug.h"                                           // for INFO
#include "internal/host/nvshmem_internal.h"                                // for nvshm...
#include "internal/host/error_codes_internal.h"                            // for NVSHM...
#include "internal/host/custom_malloc.h"                                   // for mspace
#include "internal/host/nvshmem_nvtx.hpp"                                  // for nvtx_...
#include "internal/host/nvshmemi_symmetric_heap.hpp"                       // for nvshm...
#include "internal/host/nvshmemi_mem_transport.hpp"                        // for nvshm...
#include "internal/host/nvshmemi_team.h"                                   // for nvshm...
#include "internal/host/nvshmemi_types.h"                                  // for nvshm...
#include "internal/host/shared_memory.h"                                   // for share...
#include "internal/host/sockets.h"                                         // for ipcCl...
#include "internal/host/util.h"                                            // for nvshm...
#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"  // for nvshm...
#include "internal/host_transport/cudawrap.h"                              // for CUPFN
#include "bootstrap_host_transport/env_defs_internal.h"                    // for nvshm...
#include "internal/host_transport/nvshmemi_transport_defines.h"            // for nvshm...
#include "internal/host_transport/transport.h"                             // for nvshm..
#include "internal/host/nvshmemi_nvls_rsc.hpp"

#ifdef NVSHMEM_USE_DLMALLOC
#include "dlmalloc.h"
#endif

using namespace std;

static_assert(sizeof(CUmemGenericAllocationHandle) <= NVSHMEM_MEM_HANDLE_SIZE,
              "sizeof(CUmemGenericAllocationHandle) <= NVSHMEM_MEM_HANDLE_SIZE");

/**
 * By OpenSHMEM spec standard, coll sync are not needed
 * if size == 0 or if ptr is NULL
 */
#define NVSHMEMI_IS_NO_ACTION_BY_SIZE(size) ((size) == 0)
#define NVSHMEMI_IS_NO_ACTION_BY_PTR(ptr) ((ptr) == NULL)

/**
 * Global static variables references, shared by dependent classes
 */
std::vector<nvshmemi_shared_memory_info_t> nvshmemi_symmetric_heap_sysmem_static_shm::infos_;
nvshmemi_mem_remote_transport *nvshmemi_mem_remote_transport::remote_objref_;
nvshmemi_mem_p2p_transport *nvshmemi_mem_p2p_transport::p2p_objref_;

void nvshmemi_init_symmetric_heap(nvshmemi_state_t *state, bool is_vmm, int heap_kind) {
    nvshmemi_symmetric_heap_sysmem_static_shm *nvshmemi_sysmem_shm = nullptr;
    nvshmemi_symmetric_heap_vidmem_dynamic_vmm *nvshmemi_vidmem_vmm = nullptr;
    nvshmemi_symmetric_heap_vidmem_static_pinned *nvshmemi_vidmem_static = nullptr;

    if (state->heap_obj != nullptr) {
        return;
    }

    if (nvshmemi_vidmem_vmm == nullptr && is_vmm) {
        nvshmemi_vidmem_vmm = new nvshmemi_symmetric_heap_vidmem_dynamic_vmm(state);
        state->heap_obj = dynamic_cast<nvshmemi_symmetric_heap *>(nvshmemi_vidmem_vmm);
    } else if (nvshmemi_sysmem_shm == nullptr && heap_kind == NVSHMEMI_HEAP_KIND_SYSMEM) {
        nvshmemi_sysmem_shm = new nvshmemi_symmetric_heap_sysmem_static_shm(state);
        state->heap_obj = dynamic_cast<nvshmemi_symmetric_heap *>(nvshmemi_sysmem_shm);
    } else if (nvshmemi_vidmem_static == nullptr && heap_kind == NVSHMEMI_HEAP_KIND_VIDMEM) {
        nvshmemi_vidmem_static = new nvshmemi_symmetric_heap_vidmem_static_pinned(state);
        state->heap_obj = dynamic_cast<nvshmemi_symmetric_heap *>(nvshmemi_vidmem_static);
    }

    if (state->heap_obj == nullptr) {
        NVSHMEMI_ERROR_EXIT("Requested Heap Kind: %d(0-VIDMEM,1-SYSMEM,>3-INVALID), with VMM: %s\n",
                            heap_kind, (is_vmm ? "Yes" : "No"));
    }
}

void nvshmemi_fini_symmetric_heap(nvshmemi_state_t *state) {
    if (dynamic_cast<nvshmemi_symmetric_heap_vidmem_dynamic_vmm *>(state->heap_obj) != nullptr) {
        auto *vmm_obj = dynamic_cast<nvshmemi_symmetric_heap_vidmem_dynamic_vmm *>(state->heap_obj);
        NVSHMEMU_HOST_PTR_DELETE(vmm_obj);
    } else if (dynamic_cast<nvshmemi_symmetric_heap_sysmem_static_shm *>(state->heap_obj) !=
               nullptr) {
        auto *sysmem_obj =
            dynamic_cast<nvshmemi_symmetric_heap_sysmem_static_shm *>(state->heap_obj);
        NVSHMEMU_HOST_PTR_DELETE(sysmem_obj);
    } else if (dynamic_cast<nvshmemi_symmetric_heap_vidmem_static_pinned *>(state->heap_obj) !=
               nullptr) {
        auto *vidmem_obj =
            dynamic_cast<nvshmemi_symmetric_heap_vidmem_static_pinned *>(state->heap_obj);
        NVSHMEMU_HOST_PTR_DELETE(vidmem_obj);
    }

    state->heap_obj = nullptr;
}

/**
 * nvshmemi_symmetric_heap common functions
 */
template <typename T>
int nvshmemi_symmetric_heap::is_symmetric(T value) {
    int status = 0;
    nvshmemi_state_t *state = get_state();
    T *scratch;
    /* TODO: need to handle multi-threaded scenarios */
    if (!nvshmemi_options.ENABLE_ERROR_CHECKS) return 0;

    scratch = (T *)std::calloc(state->npes, sizeof(T));
    NVSHMEMI_NULL_ERROR_JMP(scratch, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Failed to allocate scratch space for heap symmetry check \n");
    status = nvshmemi_boot_handle.allgather((void *)&value, (void *)scratch, sizeof(T),
                                            &nvshmemi_boot_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "allgather in symmetry check failed \n");

    for (int i = 0; i < state->npes; i++) {
        status = (*((T *)scratch + i) == value) ? 0 : 1;
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_SYMMETRY, out, "symmetry check failed \n");
    }

    NVSHMEMU_HOST_PTR_FREE(scratch);
out:
    return status;
}

void *nvshmemi_symmetric_heap::heap_malloc(size_t size) {
    return heap_allocate(size, 0, 0, NVSHMEMX_MALLOC);
}

void *nvshmemi_symmetric_heap::heap_calloc(size_t size, size_t count) {
    return heap_allocate(size, count, 0, NVSHMEMX_CALLOC);
}

void *nvshmemi_symmetric_heap::heap_align(size_t size, size_t alignment) {
    return heap_allocate(size, 0, alignment, NVSHMEMX_ALIGN);
}

void *nvshmemi_symmetric_heap::heap_allocate(size_t size, size_t count, size_t alignment,
                                             int type) {
    int status = 0;
    void *ptr = NULL;

    assert(get_state() != nullptr);
    status = is_symmetric(size);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                          "symmetry check for size failed\n");

    ptr = allocate_symmetric_memory(size, count, alignment, type);
    /* Don't inspect the ptr as caller will decide if its okay to have it to be NULL or non-NULL */

    INFO(NVSHMEM_MEM, "[%d] type: %s allocated %zu bytes, %zu count, %zu alignment ptr: %p",
         get_state()->mype, typeid(decltype(*this)).name(), size, count, alignment, ptr);

out:
    return ptr;
}

void nvshmemi_symmetric_heap::heap_deallocate(void *ptr) {
    heap_mspace_->deallocate(ptr);
    INFO(NVSHMEM_MEM, "[%d] freeing buf: %p type: %s", get_state()->mype, ptr,
         typeid(decltype(*this)).name());
    nvshmemi_update_device_state();
    return;
}

void nvshmemi_symmetric_heap::update_idx_in_handle(void *addr, size_t size) {
    idx_in_handles_.push_back(std::make_tuple(handles_.size() - 1, (char *)(addr), size));
}

void nvshmemi_symmetric_heap::set_heap_size_attr(size_t mem_granularity, size_t *heapextra,
                                                 size_t *alignbytes, size_t *logmem_granularity) {
    *alignbytes = NVSHMEMI_MALLOC_ALIGNMENT;
    assert((mem_granularity & (mem_granularity - 1)) == 0);
    *logmem_granularity = nvshmemu_compute_log2(mem_granularity);
    *heapextra = NUM_G_BUF_ELEMENTS * sizeof(g_elem_t) + nvshmemi_get_teams_mem_requirement() +
                 G_COALESCING_BUF_SIZE + 4 * (*alignbytes) +
                 20 * (*alignbytes);  // alignbytes, providing capacity for 2 allocations for
                                      // the library and 10 allocations for the user
}

int nvshmemi_symmetric_heap::cleanup_mspace(void) {
    if (heap_mspace_ != nullptr) {
        NVSHMEMU_HOST_PTR_DELETE(heap_mspace_);
    }

    return 0;
}

int nvshmemi_symmetric_heap::allgather_peer_base() {
    int status;
    nvshmemi_state_t *state = get_state();

    // Base virtual address of heap_base for all PEs (needed for REMOTE)
    peer_heap_base_remote_ = (void **)std::calloc(state->npes, sizeof(void *));
    NVSHMEMI_NULL_ERROR_JMP(peer_heap_base_remote_, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "failed allocating space for peer heap base remote\n");

    status = nvshmemi_boot_handle.allgather((void *)&heap_base_, (void *)peer_heap_base_remote_,
                                            sizeof(void *), &nvshmemi_boot_handle);

    // Base virtual address of heap_base for my PE (needed for P2P)
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "allgather of heap base for all PE failed \n");

    peer_heap_base_p2p_ = (void **)std::calloc(state->npes, sizeof(void *));
    NVSHMEMI_NULL_ERROR_JMP(peer_heap_base_p2p_, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "failed allocating space for peer heap base p2p\n");

    peer_heap_base_p2p_[state->mype] = heap_base_;
out:
    if (status) {
        NVSHMEMU_HOST_PTR_FREE(peer_heap_base_p2p_);
        NVSHMEMU_HOST_PTR_FREE(peer_heap_base_remote_);
    }

    return (status);
}

/**
 * This will clear heap state handle map and mspace objects
 */
nvshmemi_symmetric_heap::~nvshmemi_symmetric_heap() {
    nvshmemi_state_t *state = get_state();
    NVSHMEMU_FOR_EACH(i, state->npes) {
        NVSHMEMU_FOR_EACH(j, state->num_initialized_transports) {
            bool is_p2p_transport =
                NVSHMEMU_IS_BIT_SET(state->transport_bitmap, j) &&
                (NVSHMEMI_TRANSPORT_IS_CAP(state->transports[j], i, NVSHMEM_TRANSPORT_CAP_MAP));
            if (!is_p2p_transport) continue;

            NVSHMEMU_FOR_EACH(k, handles_.size()) {
                close(*(int *)&handles_[k][i * state->num_initialized_transports + j]);
            }
        }
    }

    handles_.clear();
    idx_in_handles_.clear();
    NVSHMEMU_HOST_PTR_FREE(peer_heap_base_p2p_);
}

int nvshmemi_symmetric_heap::map_heap_memory(void *buf, size_t size) {
    nvshmemi_state_t *state = get_state();
    nvshmemi_mem_p2p_transport &p2ptran = *(get_p2pref());
    int status = 0;
    int i = (state->mype + 1) % state->npes;
    while (i != state->mype) {
        NVSHMEMU_FOR_EACH_IF(
            j, state->num_initialized_transports,
            (NVSHMEMU_IS_BIT_SET(state->transport_map[state->mype * state->npes + i], j) &&
             NVSHMEMI_TRANSPORT_IS_CAP(state->transports[j], i, NVSHMEM_TRANSPORT_CAP_MAP)),
            {
                INFO(NVSHMEM_MEM, "Mapping Buf: %p Size: %zu PE ID: %d, P2P Transport Idx: %d\n",
                     buf, size, i, j);
                status = map_heap_chunk(i, j, (char *)buf, size);
                if (status) {
                    // map operation failed, remove cap of transport
                    state->transports[j]->cap[i] ^= NVSHMEM_TRANSPORT_CAP_MAP;
                    status = 0;
                    continue;
                }

                p2ptran.print_mem_handle(i, j, *this);
                break;
            })

        i = (i + 1) % state->npes;
    }

    if (nvshmemi_device_state.enable_rail_opt == 1) {
        if (empty_heap_handle_cache()) {
            for (size_t idx = 0; idx < (heap_size_ * state->npes_node) / mem_granularity_; idx++) {
                update_idx_in_handle((char *)global_heap_base_, heap_size_ * state->npes_node);
            }

            inc_heap_handle_cache();
        }

        goto out;
    } else {
        for (size_t idx = 0; idx < (size / mem_granularity_); idx++) {
            update_idx_in_handle((char *)heap_base_ + physical_heap_size_, size);
        }
    }

    physical_heap_size_ += size;

out:
    if (empty_heap_handle_cache()) inc_heap_handle_cache();  // Signal that cache is created
    return (status);
}

/**
 * nvshmemi_symmetric_heap kind allocate/free memory functions
 */
int nvshmemi_symmetric_heap_vidmem_static_pinned::allocate_heap_memory() {
    int status = 0;
    status = cudaMalloc(&heap_base_, heap_size_);
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                          "cuMemAlloc failed \n");
    reserved_heap_size_ = heap_size_;
out:
    return status;
}

int nvshmemi_symmetric_heap_sysmem_static_shm::allocate_heap_memory() {
    int status = 0;
    size_t shm_size = heap_size_ * nvshmemi_boot_handle.npes_node;
    int ret = snprintf(heap_name_, 100, "sysmem_symm_heap");
    if (ret < 0) {
        NVSHMEMI_ERROR_EXIT("snprintf failed\n");
    }

    if (nvshmemi_boot_handle.mype_node == 0) {
        if (shared_memory_create(heap_name_, shm_size, &heap_info_) != 0) {
            NVSHMEMI_ERROR_EXIT("Failed to create shared memory slab\n");
        }
    }

    status = nvshmemi_boot_handle.barrier(&nvshmemi_boot_handle);
    if (nvshmemi_boot_handle.mype_node != 0) {
        if (shared_memory_open(heap_name_, shm_size, &heap_info_) != 0) {
            NVSHMEMI_ERROR_EXIT("Failed to open shared memory slab\n");
        }
    }

    atexit(nvshmemi_symmetric_heap_sysmem_static_shm::
               atexit_heap_handler); /* This forces sysmem shared memory heap to be only one */
    nvshmemi_symmetric_heap_sysmem_static_shm::infos_.push_back(heap_info_);
    /* Do first touch, for NUMA awareness */
    memset((char *)heap_info_.addr + nvshmemi_boot_handle.mype_node * heap_size_, 0, heap_size_);

    status = nvshmemi_boot_handle.barrier(&nvshmemi_boot_handle);
    CUDA_RUNTIME_CHECK(cudaHostRegister(heap_info_.addr, shm_size, cudaHostRegisterDefault));
    CUDA_RUNTIME_CHECK(cudaHostGetDevicePointer(&global_heap_base_, heap_info_.addr, 0));
    heap_base_ = (char *)global_heap_base_ + nvshmemi_boot_handle.mype_node * heap_size_;
    reserved_heap_size_ = shm_size;

    return status;
}

int nvshmemi_symmetric_heap_vidmem_static_pinned::free_heap_memory(void *addr) {
    CUDA_RUNTIME_CHECK(cudaFree(addr));
    return (NVSHMEMI_SUCCESS);
}

int nvshmemi_symmetric_heap_sysmem_static_shm::free_heap_memory(void *unused_addr
                                                                __attribute__((unused))) {
    CUDA_RUNTIME_CHECK(cudaHostUnregister(heap_info_.addr));
    shared_memory_close(heap_name_, &heap_info_);
    return (NVSHMEMI_SUCCESS);
}

int nvshmemi_symmetric_heap_static::setup_mspace() {
    heap_mspace_ = new mspace(heap_base_, heap_size_);
    heap_mspace_->track_large_chunks(1);
    return 0;
}

int nvshmemi_symmetric_heap_dynamic::setup_mspace() {
    heap_mspace_ = new mspace(heap_base_, physical_heap_size_);
    heap_mspace_->track_large_chunks(1);
    return 0;
}

/* Constructor for static/dynamic class */
nvshmemi_symmetric_heap_static::nvshmemi_symmetric_heap_static(nvshmemi_state_t *state) noexcept
    : nvshmemi_symmetric_heap(state) {
    set_p2p_transport(nvshmemi_mem_p2p_transport::get_instance(state->mype, state->npes));
    set_remote_transport(nvshmemi_mem_remote_transport::get_instance());
    state->p2p_transport = get_p2pref();
}

nvshmemi_symmetric_heap_dynamic::nvshmemi_symmetric_heap_dynamic(nvshmemi_state_t *state) noexcept
    : nvshmemi_symmetric_heap(state) {
    set_p2p_transport(nvshmemi_mem_p2p_transport::get_instance(state->mype, state->npes));
    /** Today we discover the p2p transport and use 1 mem_handle_type for all dynamic heap
        In the future, we can return a bitmap of allocation handle type and use that to decide how
       to allocate multiple heaps by type
        */
    set_remote_transport(nvshmemi_mem_remote_transport::get_instance());
    set_mem_handle_type((get_p2pref()->get_mem_handle_type()));
    state->p2p_transport = get_p2pref();
}

int nvshmemi_symmetric_heap_static::reserve_heap(void) {
    int status;
    size_t heapextra = 0, alignbytes = 0;
    mem_granularity_ = NVSHMEMI_MAX_HANDLE_LENGTH;
    set_heap_size_attr(mem_granularity_, &heapextra, &alignbytes, &(log2_mem_granularity_));
    heap_size_ = NVSHMEMU_ROUND_UP(nvshmemi_options.SYMMETRIC_SIZE + heapextra, mem_granularity_);
    physical_heap_size_ = 0;

    bool data =
        true; /*A boolean attribute which when set, ensures that synchronous memory operations
                    initiated on the region of memory that ptr points to will always synchronize.*/

    allocate_heap_memory();

    status = CUPFN(
        nvshmemi_cuda_syms,
        cuPointerSetAttribute(&data, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, (CUdeviceptr)(heap_base_)));
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                          "cuPointerSetAttribute failed \n");

    INFO(NVSHMEM_MEM,
         "[%d] heap type: %s heap base: %p NVSHMEM_SYMMETRIC_SIZE %lu total %lu heapextra %lu",
         state_->mype, typeid(this).name(), heap_base_, nvshmemi_options.SYMMETRIC_SIZE, heap_size_,
         heapextra);

    status = setup_mspace();
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "memory space initialization failed \n");

    INFO(NVSHMEM_MEM, "[%d] heap type: %s cumem_granularity: %zu, log2_mem_granularity: %zu\n",
         state_->mype, typeid(decltype(*this)).name(), mem_granularity_, log2_mem_granularity_);

out:
    if (status) {
        free_heap_memory(heap_base_);
    }

    return (status);
}

int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::reserve_heap() {
    int status;
    size_t alignbytes = 0, heapextra = 0;
    CUmemAllocationProp prop = {};
    set_cuda_mem_prop((void *)&prop, get_mem_handle_type());

    status = CUPFN(nvshmemi_cuda_syms,
                   cuMemGetAllocationGranularity(&mem_granularity_, &prop,
                                                 CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "cuMemGetAllocationGranularity failed \n");
    mem_granularity_ = std::max(nvshmemi_options.CUMEM_GRANULARITY, mem_granularity_);
    mem_granularity_ = mem_granularity_ < NVSHMEMI_MAX_HANDLE_LENGTH ? mem_granularity_
                                                                     : NVSHMEMI_MAX_HANDLE_LENGTH;
    set_heap_size_attr(mem_granularity_, &heapextra, &alignbytes, &log2_mem_granularity_);
    INFO(NVSHMEM_MEM, "[%d] heap type: %s allocate_local_heap, heapextra = %lld", state_->mype,
         typeid(decltype(this)).name(), heapextra);
    heap_size_ = std::max(nvshmemi_options.MAX_MEMORY_PER_GPU, heapextra);
    heap_size_ = NVSHMEMU_ROUND_UP(heap_size_, mem_granularity_);
    physical_heap_size_ = 0;

    status =
        CUPFN(nvshmemi_cuda_syms, cuMemAddressReserve((CUdeviceptr *)&global_heap_base_,
                                                      nvshmemi_options.MAX_P2P_GPUS * heap_size_,
                                                      alignbytes, (CUdeviceptr)NULL, 0));
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "cuMemAddressReserve failed \n");
    heap_base_ = (void *)((uintptr_t)global_heap_base_);
    status = setup_mspace();
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "memory space initialization failed \n");

    INFO(NVSHMEM_MEM,
         "[%d] heap type: %s heap base: %p NVSHMEM_SYMMETRIC_SIZE %lu total %lu heapextra %lu",
         state_->mype, typeid(decltype(this)).name(), heap_base_, nvshmemi_options.SYMMETRIC_SIZE,
         heap_size_, heapextra);
    reserved_heap_size_ = nvshmemi_options.MAX_P2P_GPUS * heap_size_;
out:
    return status;
}

/**
 * nvshmemi_symmetric_heap kind setup/cleanup heap functionality
 */
int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::cleanup_symmetric_heap() {
    nvshmemi_state_t *state = get_state();
    int status = 0;
    nvshmemi_mem_remote_transport &remote_tran = *(get_remoteref());
    INFO(NVSHMEM_MEM, "[%d] Entering %s::cleanup_symmetric_heap\n", state->mype,
         typeid(decltype(this)).name());
    NVSHMEMU_FOR_EACH_IF(
        i, state->npes, (((int)i == state->mype) && (heap_base_ != NULL)),
        {NVSHMEMU_FOR_EACH_IF(
            j, handles_.size(), ((j == 0) || (j > 0 && !nvshmemi_device_state.enable_rail_opt)), {
                status = remote_tran.release_mem_handles(
                    &handles_[j][i * state->num_initialized_transports],
                    *(dynamic_cast<nvshmemi_symmetric_heap *>(this)));
                NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                                      "cleanup local handles failed \n");
                status = CUPFN(nvshmemi_cuda_syms,
                               cuMemUnmap((CUdeviceptr)heap_base_, physical_heap_size_));
                NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                                      "release memory failed for p2p on heap dynamic (my PE)\n");
                NVSHMEMU_FOR_EACH(i, cumem_handles_.size()) {
                    status =
                        CUPFN(nvshmemi_cuda_syms, cuMemRelease(std::get<0>(cumem_handles_[i])));
                    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                                          "cuMemRelease failed \n");
                }

                cumem_handles_.clear();
            })})

    /* Release and Unmap memory for peer PE */
    NVSHMEMU_FOR_EACH_IF(
        i, state->npes, ((int)i != state->mype) && peer_heap_base_p2p_[i] != NULL, {
            INFO(NVSHMEM_MEM, "calling release_memory on buf: %p size: %zu\n",
                 peer_heap_base_p2p_[i], heap_size_);
            status = release_memory(peer_heap_base_p2p_[i], heap_size_);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "release memory failed for p2p on heap dynamic (peer PE)\n");
        })

    status =
        CUPFN(nvshmemi_cuda_syms, cuMemAddressFree((CUdeviceptr)global_heap_base_,
                                                   nvshmemi_options.MAX_P2P_GPUS * heap_size_));
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "cuMemAddressFree failed \n");

    INFO(NVSHMEM_MEM, "[%d] Leaving %s::cleanup_symmetric_heap\n", state->mype,
         typeid(decltype(this)).name());
out:
    return status;
}

int nvshmemi_symmetric_heap_static::cleanup_symmetric_heap() {
    nvshmemi_state_t *state = get_state();
    nvshmemi_mem_remote_transport &remotetran = *(get_remoteref());
    int status = 0;
    INFO(NVSHMEM_MEM, "[%d] Entering %s::cleanup_symmetric_heap\n", state->mype,
         typeid(decltype(this)).name());

    NVSHMEMU_FOR_EACH_IF(i, state->npes, (((int)i == state->mype) && (heap_base_ != NULL)), {
        NVSHMEMU_FOR_EACH(j, handles_.size()) {
            if ((j == 0) || (j > 0 && !nvshmemi_device_state.enable_rail_opt)) {
                status = remotetran.release_mem_handles(
                    &handles_[j][i * state->num_initialized_transports],
                    *(dynamic_cast<nvshmemi_symmetric_heap *>(this)));
                NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                                      "release memory handles failed for remote on heap static\n");
            }
        }

        if (peer_heap_base_p2p_ != nullptr) {
            status = free_heap_memory(peer_heap_base_p2p_[i]);
            NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                                  "free_heap_memory failed \n");
        }
    })

    NVSHMEMU_FOR_EACH_IF(
        i, state->npes, ((int)i != state->mype) && peer_heap_base_p2p_[i] != NULL, {
            INFO(NVSHMEM_MEM, "calling release_memory on buf: %p \n", peer_heap_base_p2p_[i]);
            status = release_memory(peer_heap_base_p2p_[i]);
            NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                                  "release memory failed for p2p on heap static\n");
        })

    INFO(NVSHMEM_MEM, "[%d] Leaving %s::cleanup_symmetric_heap\n", state->mype,
         typeid(decltype(this)).name());

out:
    return (status);
}

int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::setup_symmetric_heap() {
    int status = 0;
    nvshmemi_state_t *state = get_state();
    nvshmem_transport_t *transports = (nvshmem_transport_t *)state->transports;
    int p2p_counter = 1;

    status = allgather_peer_base();
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Failed to allgather PEs peer_base values\n");

    for (int i = ((state->mype + 1) % state->npes); i != state->mype; i = ((i + 1) % state->npes)) {
        NVSHMEMU_FOR_EACH_IF(
            j, state->num_initialized_transports,
            (NVSHMEMU_IS_BIT_SET(state->transport_map[state->mype * state->npes + i], j)), {
                if (NVSHMEMI_TRANSPORT_IS_CAP(transports[j], i, NVSHMEM_TRANSPORT_CAP_MAP)) {
                    peer_heap_base_p2p_[i] =
                        (void *)((uintptr_t)global_heap_base_ + heap_size_ * p2p_counter++);
                    INFO(NVSHMEM_MEM, "[%d] Peer Heap Base [%d]: %p\n", state->mype, i,
                         peer_heap_base_p2p_[i]);
                    break;
                }
            })
    }

    // Retrieve the proc_map and see if its non-zero size
    // If zero size, initialize proc map
    if (get_p2pref()->get_proc_map().size() == 0) {
        get_p2pref()->create_proc_map(*(dynamic_cast<nvshmemi_symmetric_heap *>(this)));
        if (get_p2pref()->get_proc_map().size() == 0) {
            INFO(NVSHMEM_MEM,
                 "Peer PE to PID map (nvshmemi_mem_p2p_transport::proc_map_) is empty as either "
                 "P2P is disabled or P2P initialized failed\n");
        }
    }

out:
    return (status);
}

int nvshmemi_symmetric_heap_static::setup_symmetric_heap(void) {
    int status;

    status = allgather_peer_base();
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Failed to allgather PEs peer_base values\n");

    status = register_heap_memory(NULL, heap_base_, heap_size_);
    NVSHMEMI_NE_ERROR_JMP(status, 0, NVSHMEMX_ERROR_INTERNAL, out,
                          "register heap handle failed \n");
out:
    if (status) {
        // if handles has been allocated, try and cleanup all heap state
        // else cleanup local handles only
        cleanup_symmetric_heap();
        if (heap_size_) free_heap_memory(heap_base_);
    }

    return (status);
}

int nvshmemi_symmetric_heap_vidmem_static_pinned::map_heap_chunk(int pe_id, int transport_idx,
                                                                 char *buf, size_t size) {
    nvshmemi_state_t *state = get_state();
    return ((empty_heap_handle_cache())
                ? import_memory(
                      &handles_.back()[pe_id * state->num_initialized_transports + transport_idx],
                      (peer_heap_base_p2p_ + pe_id))
                : 0);
}

int nvshmemi_symmetric_heap_sysmem_static_shm::map_heap_chunk(int pe_id, int transport_idx,
                                                              char *buf, size_t size) {
    nvshmemi_state_t *state = get_state();
    if (empty_heap_handle_cache()) {
        peer_heap_base_p2p_[state->mype] = heap_base_;
        peer_heap_base_p2p_[pe_id] =
            (char *)global_heap_base_ + (pe_id % state->npes_node) * heap_size_;
    }

    return (0); /* This is a NOOP for sysmem shared memory as it is already mmap during allocation
                   time for all PEs in the node, so no need to import it per buffer range */
}

int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::map_heap_chunk(int pe_id, int transport_idx,
                                                               char *buf, size_t size) {
    void *buf_map = nullptr;
    nvshmemi_state_t *state = get_state();
    INFO(NVSHMEM_MEM,
         "calling import_memory on buf: %p size: %zu heap_base_: %p "
         "peer_heap_base_p2p_[%d]: %p\n",
         buf, size, heap_base_, pe_id, peer_heap_base_p2p_[pe_id]);
    buf_map = (void *)(buf - (char *)heap_base_ + (char *)(peer_heap_base_p2p_[pe_id]));
    return (
        import_memory(&handles_.back()[pe_id * state->num_initialized_transports + transport_idx],
                      &buf_map, size));
}

int nvshmemi_symmetric_heap_sysmem_static_shm::exchange_heap_memory_handle(
    nvshmem_mem_handle_t *local_handles) {
    return 0;
}

int nvshmemi_symmetric_heap_vidmem_static_pinned::exchange_heap_memory_handle(
    nvshmem_mem_handle_t *local_handles) {
    return 0;
}

int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::exchange_heap_memory_handle(
    nvshmem_mem_handle_t *local_handles) {
    int status = 0;
    ipcHandle *myIpcHandle = NULL;
    nvshmemi_state_t *state = get_state();
    map<pid_t, ipcHandle *> recvIpcHandles;
    pid_t pid = getpid();

    // Assuming handles can be used for intra-node GPU comms
    if (!is_cuda_mem_handle_type_ipc()) return 0;

    auto p2p_processes = get_p2pref()->get_proc_map();

    NVSHMEMI_IPC_CHECK(ipcOpenSocket(myIpcHandle, pid, pid));
    /**
     * myIpcHandle PE0: /tmp/socket-100-100
     * myIpcHandle PE1: /tmp/socket-101-101
     *
     * recvIpcHandle PE0: /tmp/socket-101-100
     * recvIpcHandle PE1: /tmp/socket-100-101
     *
     * sendFd PE0: myIpcHandle from 100 to 101
     * sendFd PE1: myIpcHandle from 101 to 100
     */

    /* Open all sockets */
    for (std::map<pid_t, int>::iterator it1 = p2p_processes.begin(); it1 != p2p_processes.end();
         ++it1) {
        pid_t sending_process = it1->first;
        if (pid != sending_process) { /* Don't recv from yourself */
            ipcHandle *recvIpcHandle = NULL;
            NVSHMEMI_IPC_CHECK(ipcOpenSocket(recvIpcHandle, sending_process, pid));
            recvIpcHandles[sending_process] = recvIpcHandle;
        }
    }

    /* Wait for all processes to open their sockets */
    status = nvshmemi_boot_handle.barrier(&nvshmemi_boot_handle);

    /* Send all FDs */
    for (std::map<pid_t, int>::iterator it1 = p2p_processes.begin(); it1 != p2p_processes.end();
         ++it1) {
        pid_t receiving_process = it1->first;
        if (pid != receiving_process) { /* Don't send to yourself */
            NVSHMEMI_IPC_CHECK(
                ipcSendFd(myIpcHandle, *(int *)local_handles, pid, receiving_process));
        }
    }

    /* Recv all FDs */
    for (std::map<pid_t, int>::iterator it1 = p2p_processes.begin(); it1 != p2p_processes.end();
         ++it1) {
        pid_t sending_process = it1->first;
        if (pid != sending_process) { /* Don't recv from  yourself */
            NVSHMEMI_IPC_CHECK(ipcRecvFd(
                recvIpcHandles[sending_process],
                (int *)&handles_.back()[it1->second * state->num_initialized_transports]));
        }
    }

    status = nvshmemi_boot_handle.barrier(&nvshmemi_boot_handle);
    NVSHMEMI_IPC_CHECK(ipcCloseSocket(myIpcHandle));
    for (std::map<pid_t, ipcHandle *>::iterator it = recvIpcHandles.begin();
         it != recvIpcHandles.end(); it++) {
        NVSHMEMI_IPC_CHECK(ipcCloseSocket(it->second));
    }

    return (status);
}

// cached handles
// STATIC sysmem + REMOTE + rail optimizable enabled + iter=1 of reg -> retrieve the cached handle
// P2P + STATIC SYSMEM or STATIC VIDMEM + iter=1 of reg -> retrieve the cached handle

// new handles
// STATIC sysmem/vidmem + P2P + iter=0 of reg -> new handles on entire heap
// DYNAMIC vidmem + P2P -> new handles on buf, size
// DYNAMIC vidmem + REMOTE -> new handles on buf,size
// STATIC vidmem + REMOTE -> new handles on buf,size
// STATIC sysmem + REMOTE + rail optimization disabled -> new handles on buf,size
// STATIC sysmem + REMOTE + rail optimizable enabled + iter=0 of reg -> new hanldes on entire heap

int nvshmemi_symmetric_heap_sysmem_static_shm::register_heap_memory_handle(
    nvshmem_mem_handle_t *local_handles, int transport_idx, nvshmem_mem_handle_t *in, void *buf,
    size_t size, nvshmem_transport_t current) {
    nvshmemi_state_t *state = get_state();
    nvshmemi_mem_remote_transport &remote_tran = *(get_remoteref());
    // register and retrieve local handles, dynamically sized requesting buf, size range if RAIL OPT
    // is disabled else register local handles for entire sysmem heap if CACHE is empty else cached
    // handles for the sysmem heap
    int status = 0;
    if (nvshmemi_device_state.enable_rail_opt == 0) {
        status =
            remote_tran.register_mem_handle(local_handles, transport_idx, in, buf, size, current);
    } else {
        if (empty_heap_handle_cache()) {
            status =
                remote_tran.register_mem_handle(local_handles, transport_idx, in, global_heap_base_,
                                                heap_size_ * state->npes_node, current);
        } else {
            local_handles[transport_idx] = handles_.front().data()[transport_idx];
        }
    }

    return (status);
}

int nvshmemi_symmetric_heap_vidmem_static_pinned::register_heap_memory_handle(
    nvshmem_mem_handle_t *local_handles, int transport_idx, nvshmem_mem_handle_t *in, void *buf,
    size_t size, nvshmem_transport_t current) {
    nvshmemi_mem_remote_transport &remote_tran = *(get_remoteref());
    return remote_tran.register_mem_handle(local_handles, transport_idx, in, buf, size, current);
}

int nvshmemi_symmetric_heap_static::register_heap_chunk(nvshmem_mem_handle_t *mem_handle_in,
                                                        void *buf, size_t size) {
    nvshmemi_state_t *state = get_state();
    nvshmemi_mem_remote_transport &remotetran = *(get_remoteref());
    nvshmem_transport_t *transports = (nvshmem_transport_t *)state->transports;
    nvshmem_mem_handle_t local_handles[state->num_initialized_transports];
    nvshmem_mem_handle_t *map_handles = nullptr;
    nvshmem_transport_t current;
    int status = 0;

    // assuming symmetry of transports across all PEs
    memset(local_handles, 0, sizeof(nvshmem_mem_handle_t) * state->num_initialized_transports);
    assert(buf != nullptr);
    assert(size < NVSHMEMI_DMA_BUF_MAX_LENGTH);

    // Register or retrieve local memory handles for the requested buffer, size range
    NVSHMEMU_FOR_EACH_IF(
        i, state->num_initialized_transports, NVSHMEMU_IS_BIT_SET(state->transport_bitmap, i), {
            current = transports[i];
            // No cached handles, so create one
            if (empty_heap_handle_cache()) {
                if (NVSHMEMI_TRANSPORT_IS_CAP(current, state->mype, NVSHMEM_TRANSPORT_CAP_MAP)) {
                    INFO(NVSHMEM_MEM,
                         "[%d] heap type: %s calling export_memory for buf: %p "
                         "size: %lu",
                         state->mype, typeid(decltype(this)).name(), buf, size);

                    status = export_memory(local_handles + i, heap_base_, heap_size_);
                    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                          "export memory failed for p2p on heap static \n");
                } else {
                    INFO(NVSHMEM_MEM,
                         "[%d] heap type: %s calling get_mem_handle for transport: %d buf: %p "
                         "size: %lu",
                         state->mype, typeid(decltype(this)).name(), i, buf, size);

                    status = register_heap_memory_handle(&local_handles[0], static_cast<int>(i),
                                                         mem_handle_in, buf, size, current);
                    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                          "register_heap_memory_handle failed for remote \n");
                }
            } else {
                // cached handles, if any so reuse
                if (NVSHMEMI_TRANSPORT_IS_CAP(current, state->mype, NVSHMEM_TRANSPORT_CAP_MAP)) {
                    map_handles = handles_.front().data();
                    local_handles[i] = map_handles[i];
                } else {
                    INFO(NVSHMEM_MEM,
                         "[%d] heap type: %s calling get_mem_handle for transport: %d buf: %p "
                         "size: %lu",
                         state->mype, typeid(decltype(this)).name(), i, buf, size);

                    status = register_heap_memory_handle(&local_handles[0], static_cast<int>(i),
                                                         mem_handle_in, buf, size, current);
                    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                          "register_heap_memory_handle failed for remote \n");
                }
            }
        })

    // Allgather memory handle for all PEs in the team
    handles_.push_back(
        vector<nvshmem_mem_handle_t>(state->num_initialized_transports * state->npes));

    status = nvshmemi_boot_handle.allgather(
        (void *)local_handles, (void *)(handles_.back().data()),
        (sizeof(nvshmem_mem_handle_t) * state->num_initialized_transports), &nvshmemi_boot_handle);

    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "allgather of mem handles failed \n");

    if (nvshmemi_device_state.enable_rail_opt == 1) {
        status = remotetran.gather_mem_handles(*(dynamic_cast<nvshmemi_symmetric_heap *>(this)), 0,
                                               heap_size_ * state->npes_node);
        NVSHMEMI_NZ_ERROR_JMP(
            status, NVSHMEMX_ERROR_INTERNAL, out,
            "allgather of mem handles for remotetransport failed (on rail optimized networks) \n");
    } else {
        status = remotetran.gather_mem_handles(*(dynamic_cast<nvshmemi_symmetric_heap *>(this)),
                                               physical_heap_size_, size);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "allgather of mem handles for remotetransport failed \n");
    }

    // Memory map the handles for all capability mapped transports
    // Update nvshmemi_state with the retrieved memory handles
    status = map_heap_memory(buf, size);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "map_heap_memory failed \n");

out:
    return (status);
}

int nvshmemi_symmetric_heap_dynamic::register_heap_chunk(nvshmem_mem_handle_t *mem_handle_in,
                                                         void *buf, size_t size) {
    nvshmemi_state_t *state = get_state();
    nvshmem_transport_t *transports = (nvshmem_transport_t *)state->transports;
    nvshmem_mem_handle_t local_handles[state->num_initialized_transports];
    nvshmemi_mem_remote_transport &remotetran = *(get_remoteref());
    nvshmem_transport_t current;
    int status = 0;

    // assuming symmetry of transports across all PEs
    memset(local_handles, 0, sizeof(nvshmem_mem_handle_t) * state->num_initialized_transports);
    assert(buf != nullptr);
    assert(size < NVSHMEMI_DMA_BUF_MAX_LENGTH);

    // Register or retrieve local memory handles for the requested buffer, size range
    NVSHMEMU_FOR_EACH_IF(
        idx, state->num_initialized_transports, NVSHMEMU_IS_BIT_SET(state->transport_bitmap, idx), {
            current = transports[idx];
            if (NVSHMEMI_TRANSPORT_IS_CAP(current, state->mype, NVSHMEM_TRANSPORT_CAP_MAP)) {
                INFO(NVSHMEM_MEM, "[%d] heap type: %s calling export_memory buf: %p size: %lu",
                     state->mype, typeid(decltype(this)).name(), buf, size);

                status =
                    export_memory((nvshmem_mem_handle_t *)(local_handles + idx), mem_handle_in);
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                      "export_memory failed for p2p on heap dynamic \n");
            } else {
                INFO(
                    NVSHMEM_MEM,
                    "[%d] heap type: %s calling get_mem_handle for transport: %d buf: %p size: %lu",
                    state->mype, typeid(decltype(this)).name(), idx, buf, size);
                status = remotetran.register_mem_handle(&local_handles[0], idx, mem_handle_in, buf,
                                                        size, current);
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                      "register_mem_handle failed for remote \n");
            }
        })

    // Allgather memory handle for remote connected PEs
    handles_.push_back(
        vector<nvshmem_mem_handle_t>(state->num_initialized_transports * state->npes));

    status = nvshmemi_boot_handle.allgather(
        (void *)local_handles, (void *)(handles_.back().data()),
        sizeof(nvshmem_mem_handle_t) * state->num_initialized_transports, &nvshmemi_boot_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "allgather of mem handles failed \n");

    status = remotetran.gather_mem_handles(*(dynamic_cast<nvshmemi_symmetric_heap *>(this)),
                                           physical_heap_size_, size);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "allgather of mem handles failed for remotetransport\n");

    // Exchange send/recv memory handles for p2p connected PEs
    exchange_heap_memory_handle(&local_handles[0]);
    heap_mspace_->add_new_chunk((char *)heap_base_ + physical_heap_size_, size);

    // Memory map the handles for all capability mapped transports
    // Update nvshmemi_state with the retrieved memory handles
    status = map_heap_memory(buf, size);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "map_heap_memory failed \n");
out:
    return (status);
}

/**
 * nvshmemi_symmetric_heap_vidmem kind specific export/import/release operations
 */
int nvshmemi_symmetric_heap_vidmem_static_pinned::export_memory(nvshmem_mem_handle_t *mem_handle,
                                                                void *buf, size_t length) {
    int status = 0;
    cudaIpcMemHandle_t *ipc_handle = (cudaIpcMemHandle_t *)mem_handle;

    assert(sizeof(cudaIpcMemHandle_t) <= NVSHMEM_MEM_HANDLE_SIZE);
    INFO(NVSHMEM_MEM, "calling cuIpcGetMemHandle on buf: %p size: %zu", buf, length);

    status = cudaIpcGetMemHandle(ipc_handle, buf);
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INVALID_VALUE, out,
                          "cudaIpcGetMemHandle failed \n");
out:
    return (status);
}

int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::export_memory(nvshmem_mem_handle_t *mem_handle,
                                                              nvshmem_mem_handle_t *mem_handle_in) {
    int status = 0;
    CUmemGenericAllocationHandle *handle_in =
        reinterpret_cast<CUmemGenericAllocationHandle *>(mem_handle_in);
    INFO(NVSHMEM_MEM, "calling cuMemExportToShareableHandle on handle: %p", handle_in);
    status = CUPFN(nvshmemi_cuda_syms, cuMemExportToShareableHandle((void *)mem_handle, *handle_in,
                                                                    get_mem_handle_type(), 0));
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "cuMemExportToShareableHandle failed \n");
out:
    return (status);
}

int nvshmemi_symmetric_heap_sysmem_static_shm::export_memory(nvshmem_mem_handle_t *mem_handle,
                                                             void *buf, size_t length) {
    return (0); /** This is a NOOP for linux sysmem shared memory as entire memory is mapped to all
                   PEs at allocation time, so there is no step needed to export the memory */
}

int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::import_memory(nvshmem_mem_handle_t *mem_handle,
                                                              void **buf, size_t size) {
    int status = 0;
    CUmemGenericAllocationHandle peer_handle;
    CUmemAccessDesc access;
    CUdevice gpu_device_id;

    status = CUPFN(nvshmemi_cuda_syms, cuCtxGetDevice(&gpu_device_id));
    if (status != CUDA_SUCCESS) {
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }

    if (get_mem_handle_type() == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
        int fd = *(int *)mem_handle;
        status = CUPFN(nvshmemi_cuda_syms,
                       cuMemImportFromShareableHandle(&peer_handle, (void *)(uintptr_t)fd,
                                                      get_mem_handle_type()));
    } else {
        status =
            CUPFN(nvshmemi_cuda_syms, cuMemImportFromShareableHandle(
                                          &peer_handle, (void *)mem_handle, get_mem_handle_type()));
    }
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "cuMemImportFromShareableHandle failed state->device_id : %d \n",
                          gpu_device_id);

    status = CUPFN(nvshmemi_cuda_syms, cuMemMap((CUdeviceptr)*buf, size, 0, peer_handle, 0));
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "cuMemMap failed to map %ld bytes handle at address: %p\n", size, *buf);
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = gpu_device_id;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    status = CUPFN(nvshmemi_cuda_syms, cuMemRelease(peer_handle));
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "cuMemRelease failed \n");
    status = CUPFN(nvshmemi_cuda_syms,
                   cuMemSetAccess((CUdeviceptr)*buf, size, (const CUmemAccessDesc *)&access, 1));
out:
    return (status);
}

int nvshmemi_symmetric_heap_vidmem_static_pinned::import_memory(nvshmem_mem_handle_t *mem_handle,
                                                                void **buf, size_t size) {
    int status = 0;
    cudaIpcMemHandle_t *ipc_handle = (cudaIpcMemHandle_t *)mem_handle;

    status = cudaIpcOpenMemHandle(buf, *ipc_handle, cudaIpcMemLazyEnablePeerAccess);
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INVALID_VALUE, out,
                          "cudaIpcOpenMemHandle failed with error %d \n", status);
out:
    return (status);
}

int nvshmemi_symmetric_heap_sysmem_static_shm::import_memory(nvshmem_mem_handle_t *mem_handle,
                                                             void **buf, size_t size) {
    return (0); /** This is a NOOP for linux sysmem shared memory as entire memory is mapped to all
                   PEs at allocation time, so there is no step needed to export the memory */
}

int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::release_memory(void *buf, size_t size) {
    int status = 0;
    status = CUPFN(nvshmemi_cuda_syms, cuMemUnmap((CUdeviceptr)buf, size));
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INVALID_VALUE, out,
                          "cuMemUnmap failed with error %d \n", status);
out:
    return (status);
}

int nvshmemi_symmetric_heap_vidmem_static_pinned::release_memory(void *buf, size_t size) {
    int status = 0;
    status = cudaIpcCloseMemHandle(buf);
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INVALID_VALUE, out,
                          "cudaIpcCloseMemHandle failed with error %d \n", status);
out:
    return (status);
}

int nvshmemi_symmetric_heap_sysmem_static_shm::release_memory(void *buf, size_t size) {
    return (0); /** This is a NOOP for linux sysmem shared memory as entire memory is munmap to all
                   PEs at cleanup time, so there is no step needed to release buffer range */
}

int nvshmemi_symmetric_heap::register_heap_memory(nvshmem_mem_handle_t *input, void *buffer,
                                                  size_t size) {
    int status = 0;
    size_t remaining_size;
    size_t registration_size;
    char *buf = (char *)buffer;

    if (size == 0) {
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    remaining_size = size;
    do {
        registration_size = NVSHMEMI_MAX_HANDLE_LENGTH >= remaining_size
                                ? remaining_size
                                : NVSHMEMI_MAX_HANDLE_LENGTH;
        status = register_heap_chunk(input, buf, registration_size);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "register heap memory failed \n");

        assert(remaining_size >= registration_size);
        remaining_size -= registration_size;
        buf += registration_size;
    } while (remaining_size);

out:
    return status;
}

int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::nvls_broadcast_heap_handle_ipc(
    char *shareable_handle, int root, nvshmemi_team_t *team) {
    pid_t pid = getpid();
    ipcHandle *myIpcHandle = NULL;
    ipcHandle *recvIpcHandle = NULL;
    auto p2p_processes = get_p2pref()->get_proc_map();
    pid_t root_process;
    int fd = -1;
    /**
     * myIpcHandle PE0: /tmp/socket-100-100
     * recvIpcHandle PE1: /tmp/socket-100-101
     *
     * sendFd PE0: myIpcHandle from 100 to 101
     * recvFd PE1: recvIpcHandle from 100 to 101
     */

    if (team->my_pe == root) {
        /* Open socket to send to all processes */
        NVSHMEMI_IPC_CHECK(ipcOpenSocket(myIpcHandle, pid, pid));
        root_process = pid;
    } else {
        /* Open socket to recv from root process */
        for (auto it = p2p_processes.begin(); it != p2p_processes.end(); ++it) {
            if (it->second == nvshmemi_team_pe(team, root)) {
                root_process = it->first;
                NVSHMEMI_IPC_CHECK(ipcOpenSocket(recvIpcHandle, root_process, pid));
                break;
            }
        }
    }

    /* Wait for all processes in the team to open their sockets */
    nvshmem_barrier(team->team_idx);

    if (root == team->my_pe) {
        /* Send fd from root to all */
        for (std::map<pid_t, int>::iterator it1 = p2p_processes.begin(); it1 != p2p_processes.end();
             ++it1) {
            pid_t receiving_process = it1->first;
            if (pid != receiving_process &&
                nvshmemi_team_translate_pe(nvshmemi_team_pool[NVSHMEM_TEAM_WORLD], it1->second,
                                           team) !=
                    NVSHMEM_TEAM_INVALID) { /* Don't send to yourself or don't send it to a PE not
                                               in the team */
                NVSHMEMI_IPC_CHECK(
                    ipcSendFd(myIpcHandle, *(int *)shareable_handle, pid, receiving_process));
            }
        }

        INFO(NVSHMEM_INIT, "Sending shareable handle from PID %d over IPC Socket Handle %p\n", pid,
             myIpcHandle);
    } else {
        /* Recv fd at all from root */
        NVSHMEMI_IPC_CHECK(ipcRecvFd(recvIpcHandle, &fd));
        INFO(NVSHMEM_INIT,
             "Receiving shareable handle to PID %d over IPC Socket Handle %p => converted fd "
             "%d\n",
             pid, recvIpcHandle, fd);
    }

    /* Wait for all processes to finish send/recv */
    nvshmem_barrier(team->team_idx);
    if (team->my_pe == root) {
        NVSHMEMI_IPC_CHECK(ipcCloseSocket(myIpcHandle));
        fd = *(int *)shareable_handle;
        close(fd);
    } else {
        NVSHMEMI_IPC_CHECK(ipcCloseSocket(recvIpcHandle));
        memcpy(shareable_handle, &fd, sizeof(int));
    }

    return 0;
}

int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::nvls_broadcast_heap_handle_fabric(
    char *buffer, size_t length, int root, nvshmemi_team_t *team) {
    /* This is technical debt where we are using the REDUCE op psync as scratchpad for src/dst of
     * broadcast broadcast's psync is used for LL8 and other algorithms, making it non-trivial to
     * share when issued from the host as a src or dest buffer.
     *
     * When reduce coll supports LL8 algorithm, we need to clean this up as a independent scratch
     * space
     */
    long *pWrk = nvshmemi_team_get_psync(team, REDUCE);
    if (team->my_pe == root) {
        CUDA_RUNTIME_CHECK(cudaMemcpy(pWrk, buffer, length, cudaMemcpyHostToDevice));
        CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
        for (int i = 0; i < team->size; i++) {
            nvshmemx_char_put_nbi_on_stream((char *)pWrk, (const char *)pWrk, length,
                                            team->start + i * team->stride, (cudaStream_t)0);
        }
        nvshmem_barrier(team->team_idx);
    } else {
        nvshmem_barrier(team->team_idx);
        CUDA_RUNTIME_CHECK(cudaMemcpy(buffer, pWrk, length, cudaMemcpyDeviceToHost));
        CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
    }
    return (0);
}

int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::nvls_broadcast_heap_handle_by_team(
    char *buffer, size_t length, nvshmemi_team_t *team) {
    int status = NVSHMEMX_ERROR_INTERNAL;
    int root = 0; /* every team's PE0 */
    if (is_cuda_mem_handle_type_fabric()) {
        status = nvls_broadcast_heap_handle_fabric(buffer, length, root, team);
    } else {
        status = nvls_broadcast_heap_handle_ipc(buffer, root, team);
    }

    return (status);
}

int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::nvls_create_heap_memory_by_size(
    nvshmemi_team_t *team, uint64_t mem_size) {
    int status = -1;
    char shareable_handle[64] = {0};
    CUmemGenericAllocationHandle *my_handle;
    CUmemGenericAllocationHandle peer_handle;
    nvls::nvshmemi_nvls_rsc *nvls_obj = reinterpret_cast<nvls::nvshmemi_nvls_rsc *>(team->nvls_rsc);
    // Prune for duplicate teams that inherit the rsc, but own the resource
    if (!nvls_obj->is_owner(team)) return 0;

    /* team PE0 will export MC group */
    if (team->my_pe == 0) {
        status = nvls_obj->export_group(mem_size, &shareable_handle[0]);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, cleanup,
                              "Exporting multicast group failed for pe %d\n", team->my_pe);

        /* Get the most recently allocated mc_handle */
        my_handle = nvls_obj->get_mc_handle_ptr(nvls_obj->get_mc_handle_size() - 1);
        NVSHMEMI_NULL_ERROR_JMP(my_handle, status, NVSHMEMX_ERROR_INTERNAL, cleanup,
                                "No active multicast group for pe %d\n", team->my_pe);

        status = nvls_broadcast_heap_handle_by_team(&shareable_handle[0], sizeof(shareable_handle),
                                                    team);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, cleanup,
                              "Broadcasting exported multicast group for pe %d failed\n",
                              team->my_pe);

        status = nvls_obj->subscribe_group(my_handle);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, cleanup,
                              "Subscribing multicast group failed for pe %d\n", team->my_pe);

    } else {
        status = nvls_broadcast_heap_handle_by_team(&shareable_handle[0], sizeof(shareable_handle),
                                                    team);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, cleanup,
                              "Broadcasting exported multicast group for pe %d failed\n",
                              team->my_pe);

        status = nvls_obj->import_group(&shareable_handle[0], &peer_handle, mem_size);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, cleanup,
                              "Importing multicast group failed for pe %d\n", team->my_pe);

        status = nvls_obj->subscribe_group(&peer_handle);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, cleanup,
                              "Subscribing multicast group failed for pe %d\n", team->my_pe);
    }

    nvshmem_barrier(team->team_idx);
cleanup:
    return (status);
}

int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::nvls_create_heap_memory(uint64_t mem_size) {
    nvshmemi_team_t *team = NULL;
    int status = 0; /* Passthrough for the case where no teams have NVLS resource */

    if (!get_state()->is_platform_nvls) return (status);

    NVSHMEMU_FOR_EACH_IF(
        i, nvshmemi_max_teams,
        nvshmemi_team_pool != NULL && nvshmemi_team_pool[i] != NULL &&
            nvshmemi_team_support_nvls(nvshmemi_team_pool[i]),
        {
            team = nvshmemi_team_pool[i];
            status = nvls_create_heap_memory_by_size(team, mem_size);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, cleanup,
                                  "Creating mc handle for team ID: %d failed\n", team->team_idx);
            INFO(NVSHMEM_INIT, "Setting up mcHandle for team ID: %d\n", team->team_idx);
        })

cleanup:
    return (status);
}

int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::nvls_bind_heap_memory_by_size(
    nvshmemi_team_t *team, nvshmem_mem_handle_t *mem_handle, off_t mc_offset, off_t mmap_offset,
    size_t mmap_size) {
    int status = -1;
    CUmemGenericAllocationHandle *mc_handle = NULL;
    nvls::nvshmemi_nvls_rsc *nvls_obj = reinterpret_cast<nvls::nvshmemi_nvls_rsc *>(team->nvls_rsc);
    // Prune for duplicate teams that inherit the rsc, but own the resource
    if (!nvls_obj->is_owner(team)) return 0;

    /* Get the most recently allocated mc_handle */
    mc_handle = nvls_obj->get_mc_handle_ptr(nvls_obj->get_mc_handle_size() - 1);
    NVSHMEMI_NULL_ERROR_JMP(mc_handle, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "No active MC group for team idx %d\n", team->team_idx);

    INFO(NVSHMEM_MEM,
         "type: %s binding multicast group %ld to memory handle %p mmap size %zu, mc "
         "offset "
         "%lx mmap offset %lx\n",
         typeid(decltype(this)).name(), *mc_handle, mem_handle, mmap_size, mc_offset, mmap_offset);
    status = nvls_obj->bind_group_mem(mc_handle, mem_handle, mmap_size, mmap_offset, mc_offset);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Binding mem_handle %p to MC group %lld failed \n", mem_handle,
                          *mc_handle);
out:
    if (status) {
        print_cumem_handles();
    }
    return (status);
}

int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::nvls_bind_heap_memory(
    nvshmem_mem_handle_t *mem_handle, off_t mc_offset, off_t mmap_offset, size_t mmap_size) {
    int status = 0; /* Passthrough for the case where no teams have NVLS resource */
    if (!get_state()->is_platform_nvls) return (status);
    NVSHMEMU_FOR_EACH_IF(i, nvshmemi_max_teams,
                         nvshmemi_team_pool != NULL && nvshmemi_team_pool[i] != NULL &&
                             nvshmemi_team_support_nvls(nvshmemi_team_pool[i]),
                         {
                             status =
                                 nvls_bind_heap_memory_by_size(nvshmemi_team_pool[i], mem_handle,
                                                               mc_offset, mmap_offset, mmap_size);
                             NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                                   "Binding MC handle for team ID: %d failed\n",
                                                   nvshmemi_team_pool[i]->team_idx);
                             INFO(NVSHMEM_INIT, "Binding mc handle for team ID: %d\n",
                                  nvshmemi_team_pool[i]->team_idx);
                         })

out:
    return (status);
}

int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::nvls_map_heap_memory_by_size(nvshmemi_team_t *team,
                                                                             uint64_t size,
                                                                             off_t mmap_offset,
                                                                             off_t mc_offset) {
    int status = -1;
    CUmemGenericAllocationHandle *mc_handle = NULL;

    nvls::nvshmemi_nvls_rsc *nvls_obj = reinterpret_cast<nvls::nvshmemi_nvls_rsc *>(team->nvls_rsc);
    // Prune for duplicate teams that inherit the rsc, but own the resource
    if (!nvls_obj->is_owner(team)) return 0;

    /* Get the most recently allocated mc_handle */
    mc_handle = nvls_obj->get_mc_handle_ptr(nvls_obj->get_mc_handle_size() - 1);
    NVSHMEMI_NULL_ERROR_JMP(mc_handle, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "No active MC group for team idx %d\n", team->team_idx);
    INFO(NVSHMEM_MEM,
         "type: %s mapping multicast group %ld of size %zu, mc "
         "offset "
         "%lx mmap offset %lx\n",
         typeid(decltype(this)).name(), *mc_handle, size, mc_offset, mmap_offset);

    status = nvls_obj->map_group_mem(mc_handle, size, mmap_offset, mc_offset);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Mapping mem size %zu to MC group %lld failed \n", size, *mc_handle);
out:
    if (status) {
        print_cumem_handles();
    }
    return (status);
}

int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::nvls_map_heap_memory(uint64_t size,
                                                                     off_t mmap_offset,
                                                                     off_t mc_offset) {
    int status = 0; /* Passthrough for the case where no teams have NVLS resource */
    if (!get_state()->is_platform_nvls) return (status);
    NVSHMEMU_FOR_EACH_IF(i, nvshmemi_max_teams,
                         nvshmemi_team_pool != NULL && nvshmemi_team_pool[i] != NULL &&
                             nvshmemi_team_support_nvls(nvshmemi_team_pool[i]),
                         {
                             status = nvls_map_heap_memory_by_size(nvshmemi_team_pool[i], size,
                                                                   mmap_offset, mc_offset);
                             NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                                   "Mapping MC handle for team ID: %d failed\n",
                                                   nvshmemi_team_pool[i]->team_idx);
                             INFO(NVSHMEM_INIT, "Mapping mc handle for team ID: %d\n",
                                  nvshmemi_team_pool[i]->team_idx);
                         })
out:
    return (status);
}

int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::nvls_create_heap_memory_by_team(
    nvshmemi_team_t *team) {
    return nvls_create_heap_memory_by_size(team, physical_heap_size_);
}

int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::nvls_bind_heap_memory_by_team(
    nvshmemi_team_t *team) {
    int status = -1;
    CUmemGenericAllocationHandle mem_handle;
    off_t mc_offset, mmap_offset;
    size_t mmap_size;
    /* Iterate over heap's list of tuple <mem_handle, mc_offset, mmap_offset, mmap_size> */
    NVSHMEMU_FOR_EACH(i, get_cumem_handle_size()) {
        mem_handle = get_cumem_handle_ptr(i);
        mc_offset = get_cumem_handle_alloc_offset(i);
        mmap_offset = get_cumem_handle_mmap_offset(i);
        mmap_size = get_cumem_handle_mmap_size(i);
        /* Bind UC handles to MC handle at heap_offset */
        status = nvls_bind_heap_memory_by_size(team, (nvshmem_mem_handle_t *)&mem_handle, mc_offset,
                                               mmap_offset, mmap_size);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, cleanup,
                              "Binding multicast groups to UC mem handle %lld, mmap size %zu, mc "
                              "offset %ld, mmap offset %ld failed for pe %d team ID %d\n",
                              mem_handle, mmap_size, mc_offset, mmap_offset, team->my_pe,
                              team->team_idx);
    }

cleanup:
    return (status);
}

void nvshmemi_symmetric_heap_vidmem_dynamic_vmm::print_cumem_handles(void) {
    NVSHMEMU_FOR_EACH(i, get_cumem_handle_size()) {
        INFO(NVSHMEM_MEM,
             "[%d] UC mem_handle: %lld mc_offset: %ld mmap_offset: %ld mmap_size: %zu\n",
             get_state()->mype, get_cumem_handle_ptr(i), get_cumem_handle_alloc_offset(i),
             get_cumem_handle_mmap_offset(i), get_cumem_handle_mmap_size(i));
    }
    return;
}

int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::nvls_map_heap_memory_by_team(
    nvshmemi_team_t *team) {
    /* Map MC handle + mmap_offset = 0 to mc base + mc_offset=0 */
    return nvls_map_heap_memory_by_size(team, physical_heap_size_, 0, 0);
}

void nvshmemi_symmetric_heap_vidmem_dynamic_vmm::nvls_unmap_heap_memory_by_team(
    nvshmemi_team_t *team) {
    nvls::nvshmemi_nvls_rsc *nvls_obj = reinterpret_cast<nvls::nvshmemi_nvls_rsc *>(team->nvls_rsc);
    if (nvls_obj == nullptr || !nvls_obj->is_owner(team)) return;
    nvls_obj->unmap_group_mem(0, physical_heap_size_);
    return;
}

void nvshmemi_symmetric_heap_vidmem_dynamic_vmm::nvls_unbind_heap_memory_by_team(
    nvshmemi_team_t *team) {
    nvls::nvshmemi_nvls_rsc *nvls_obj = reinterpret_cast<nvls::nvshmemi_nvls_rsc *>(team->nvls_rsc);
    if (nvls_obj == nullptr || !nvls_obj->is_owner(team)) return;
    NVSHMEMU_FOR_EACH(i, nvls_obj->get_mc_handle_size()) {
        nvls_obj->unbind_group_mem(nvls_obj->get_mc_handle_ptr(i), 0,
                                   nvls_obj->get_mc_handle_ptr_size(i));
    }

    return;
}

int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::allocate_physical_memory_to_heap(size_t size) {
    size = ((size + mem_granularity_ - 1) / mem_granularity_) * mem_granularity_;
    INFO(NVSHMEM_MEM, "type: %s adding new physical backing of size %zu bytes",
         typeid(decltype(this)).name(), size);

    CUmemGenericAllocationHandle cumem_handle;
    CUmemAllocationProp prop = {};
    CUmemAccessDesc access;
    char *buf_end, *buf_start;
    off_t heap_offset = 0;
    size_t remaining_size;
    size_t register_size;
    size_t adjusted_max_handle_len;
    off_t mmap_offset =
        0; /* CUDA doesn't support non-zero mem_offset of a UC mem handle, so force to 0 */
    int status;
    nvshmemi_state_t *state = get_state();
    set_cuda_mem_prop((void *)&prop, get_mem_handle_type());

    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = state->device_id;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    assert(size % mem_granularity_ == 0);
    assert(mem_granularity_ <= NVSHMEMI_MAX_HANDLE_LENGTH);

    /* Round Down */
    adjusted_max_handle_len = mem_granularity_ * (NVSHMEMI_MAX_HANDLE_LENGTH / mem_granularity_);
    status = nvls_create_heap_memory(size);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "create heap MC memory failed\n");
    remaining_size = size;
    buf_start = (char *)heap_base_ + physical_heap_size_;
    do {
        buf_end = (char *)heap_base_ + physical_heap_size_;
        register_size =
            remaining_size > adjusted_max_handle_len ? adjusted_max_handle_len : remaining_size;

        status = CUPFN(nvshmemi_cuda_syms, cuMemCreate(&cumem_handle, register_size,
                                                       (const CUmemAllocationProp *)&prop, 0));
        NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                              "cuMemCreate failed \n");

        /* Global offset in the heap */
        heap_offset = (off_t)(physical_heap_size_);
        cumem_handles_.push_back(
            std::make_tuple(cumem_handle, heap_offset /*mc_offset*/, mmap_offset, register_size));

        status = CUPFN(nvshmemi_cuda_syms,
                       cuMemMap((CUdeviceptr)buf_end, register_size, mmap_offset, cumem_handle, 0));
        NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                              "cuMemMap failed \n");

        status = CUPFN(nvshmemi_cuda_syms, cuMemSetAccess((CUdeviceptr)buf_end, register_size,
                                                          (const CUmemAccessDesc *)&access, 1));
        NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                              "cuMemSetAccess failed \n");
        status = nvls_bind_heap_memory(
            (nvshmem_mem_handle_t *)&cumem_handle,
            (off_t)(buf_end - buf_start) /*mc_offset or local offset per request */, mmap_offset,
            register_size);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "bind heap MC memory failed\n");
        status =
            register_heap_memory((nvshmem_mem_handle_t *)&cumem_handle, buf_end, register_size);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "register heap UC memory failed \n");
        remaining_size -= register_size;
    } while (remaining_size > 0);

    /* The above loop works iff remaining_size and adjusted_max_handle_len are both multiples of
     * mem_granularity_ */

    /* Request is layout as follows
     * Alloc start -> buf_start
     * ........................
     * Alloc end   -> buf_end + register_size
     */
    heap_offset = (off_t)(buf_start - (char *)heap_base_);
    status = nvls_map_heap_memory(size, 0 /*mem_offset=0*/, heap_offset /*mc_offset*/);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "map heap MC memory failed\n");
    status = nvshmemi_boot_handle.barrier(
        &nvshmemi_boot_handle); /* Wait for all PEs to setup the new memory */
out:
    if (status) {
        print_cumem_handles();
        cleanup_symmetric_heap();
    }

    return status;
}

void *nvshmemi_symmetric_heap::allocate_virtual_memory_from_mspace(size_t size, size_t count,
                                                                   size_t alignment, int type) {
    void *ptr = NULL;
    switch (type) {
        case NVSHMEMX_MALLOC:
            ptr = heap_mspace_->allocate(size);
            break;
        case NVSHMEMX_CALLOC:
            ptr = heap_mspace_->allocate_zeroed(count, size);
            break;
        case NVSHMEMX_ALIGN:
            ptr = heap_mspace_->allocate_aligned(alignment, size);
            break;
        default:
            return (NULL);
    }

    return (ptr);
}

void *nvshmemi_symmetric_heap_vidmem_dynamic_vmm::allocate_symmetric_memory(size_t size,
                                                                            size_t count,
                                                                            size_t alignment,
                                                                            int type) {
    int status = 0;
    void *ptr = NULL;

    ptr = allocate_virtual_memory_from_mspace(size, count, alignment, type);
    if ((size > 0) && (ptr == NULL)) {
        status = allocate_physical_memory_to_heap(size + alignment);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "allocate_physical_memory_to_heap failed\n");
        ptr = allocate_virtual_memory_from_mspace(size, count, alignment, type);
        /* Only update the device state when physical heap is allocated successfully */
        status = nvshmemi_update_device_state();
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "nvshmemi_update_device_state failed\n");
    }

out:
    return (ptr);
}

void *nvshmemi_symmetric_heap_static::allocate_symmetric_memory(size_t size, size_t count,
                                                                size_t alignment, int type) {
    void *ptr = NULL;

    ptr = allocate_virtual_memory_from_mspace(size, count, alignment, type);
    if ((count > 0 && type == NVSHMEMX_CALLOC) && (size > 0) && (ptr == NULL)) {
        NVSHMEMI_ERROR_EXIT(
            "nvshmem malloc failed (hint: check if total allocation has exceeded NVSHMEM "
            "symmetric size = %zu, NVSHMEM symmetric size can be increased using "
            "NVSHMEM_SYMMETRIC_SIZE environment variable) \n",
            nvshmemi_options.SYMMETRIC_SIZE);
    }

    return (ptr);
}

extern "C" {

void nvshmemi_free(void *ptr) {
    if (ptr == NULL) return;

    nvshmemi_state->heap_obj->heap_deallocate(ptr);
}

void *nvshmemi_malloc(size_t size) { return nvshmemi_state->heap_obj->heap_malloc(size); }

}  // extern "C"

void *nvshmem_malloc(size_t size) {
    void *ptr = NULL;

    NVTX_FUNC_RANGE_IN_GROUP(ALLOC);

    NVSHMEMU_THREAD_CS_ENTER();
    nvshmemi_check_state_and_init();

    if (NVSHMEMI_IS_NO_ACTION_BY_SIZE(size)) {
        goto exit_and_return;
    }

    ptr = nvshmemi_state->heap_obj->heap_malloc(size);

    nvshmemi_barrier_all();

exit_and_return:
    NVSHMEMU_THREAD_CS_EXIT();

    return ptr;
}

void *nvshmem_calloc(size_t count, size_t size) {
    void *ptr = NULL;

    NVTX_FUNC_RANGE_IN_GROUP(ALLOC);

    NVSHMEMU_THREAD_CS_ENTER();
    nvshmemi_check_state_and_init();

    if (NVSHMEMI_IS_NO_ACTION_BY_SIZE(size)) {
        goto exit_and_return;
    }

    ptr = nvshmemi_state->heap_obj->heap_calloc(size, count);

    nvshmemi_barrier_all();

exit_and_return:
    NVSHMEMU_THREAD_CS_EXIT();

    return ptr;
}

void *nvshmem_align(size_t alignment, size_t size) {
    void *ptr = NULL;

    NVTX_FUNC_RANGE_IN_GROUP(ALLOC);

    NVSHMEMU_THREAD_CS_ENTER();
    nvshmemi_check_state_and_init();

    if (NVSHMEMI_IS_NO_ACTION_BY_SIZE(size)) {
        goto exit_and_return;
    }

    ptr = nvshmemi_state->heap_obj->heap_align(size, alignment);

    nvshmemi_barrier_all();

exit_and_return:
    NVSHMEMU_THREAD_CS_EXIT();

    return ptr;
}

void nvshmem_free(void *ptr) {
    NVTX_FUNC_RANGE_IN_GROUP(ALLOC);

    NVSHMEMU_THREAD_CS_ENTER();

    NVSHMEMI_CHECK_INIT_STATUS();

    if (NVSHMEMI_IS_NO_ACTION_BY_PTR(ptr)) {
        goto exit_and_return;
    }

    nvshmemi_barrier_all();

    nvshmemi_free(ptr);

exit_and_return:
    NVSHMEMU_THREAD_CS_EXIT();
}

void *nvshmem_ptr(const void *ptr, int pe) {
    if (ptr >= nvshmemi_device_state.heap_base) {
        uintptr_t offset = (char *)ptr - (char *)nvshmemi_device_state.heap_base;

        if (offset < nvshmemi_device_state.heap_size) {
            void *peer_addr = nvshmemi_state->heap_obj->get_local_pe_base()[pe];
            if (peer_addr != NULL) peer_addr = (void *)((char *)peer_addr + offset);
            return peer_addr;
        }
    }

    return NULL;
}

void *nvshmemx_mc_ptr(nvshmem_team_t team, const void *ptr) {
    uintptr_t offset = (char *)ptr - (char *)nvshmemi_device_state.heap_base;
    if (ptr >= nvshmemi_device_state.heap_base && offset < nvshmemi_device_state.heap_size) {
        nvls::nvshmemi_nvls_rsc *nvls =
            reinterpret_cast<nvls::nvshmemi_nvls_rsc *>(nvshmemi_team_pool[team]->nvls_rsc);
        void *mc_addr = nvls->get_mc_base();
        if (mc_addr != NULL) mc_addr = (void *)((char *)mc_addr + offset);
        return mc_addr;
    } else {
        return NULL;
    }
}
