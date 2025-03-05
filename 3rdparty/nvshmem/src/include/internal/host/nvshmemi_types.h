/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _NVSHMEMI_TYPES_H
#define _NVSHMEMI_TYPES_H

#include <atomic>
#include <cuda.h>
#include <cuda_runtime.h>
#include <tuple>
#include <vector>
#include "device_host/nvshmem_types.h"
#include "internal/host_transport/nvshmemi_transport_defines.h"
#include "internal/host_transport/transport.h"

#define MAX_PES_PER_GPU 48
typedef struct nvshmemi_mps_shmdata {
    volatile size_t nprocesses;
    volatile std::atomic<int> barrier;
    volatile std::atomic<bool> sense;
    volatile cudaIpcEventHandle_t event_handle[MAX_PES_PER_GPU];
} nvshmemi_mps_shmdata_t;
typedef struct nvshmemi_shared_memory_info_t {
    void *addr;
    size_t size;
    int shm_fd;
} nvshmemi_shared_memory_info;

class nvshmemi_symmetric_heap;
class nvshmemi_mem_p2p_transport;

typedef struct nvshmemi_state_dec {
    /*PE state*/
    int mype;
    int npes;
    int mype_node;
    int npes_node;
    /*device state*/
    int device_id;
    CUcontext cucontext;
    /*symmetric heap state*/
    nvshmemi_symmetric_heap *heap_obj;
    bool host_memory_registration_supported;
    /*transport info*/
    nvshmemi_mem_p2p_transport *p2p_transport;
    uint32_t atomic_host_endian_min_size;
    int transport_bitmap;
    int *transport_map;
    struct nvshmem_transport_pe_info *pe_info;
    struct nvshmem_transport **transports;
    int num_initialized_transports;
    /*consolidated rma ops*/
    int *selected_transport_for_rma;
    int *selected_transport_for_amo;

    cudaStream_t my_stream;
    // proxy
    void *proxy;
    cudaStream_t *custreams;
    cudaEvent_t *cuevents;
    bool *active_internal_streams;
    bool used_internal_streams;
    /* MPS support */
    cudaEvent_t mps_event;
    cudaEvent_t
        same_gpu_other_pe_mps_events[MAX_PES_PER_GPU - 1]; /* CUDA IPC mapped mps_events from the
                                                              PEs sharing the same GPU */
    nvshmemi_shared_memory_info_t shm_info;
    /* NVLS */
    bool is_platform_nvls;
    bool is_platform_nvl;
    bool are_nics_ll128_compliant;
} nvshmemi_state_t;

extern nvshmemi_state_t *nvshmemi_state;

extern bootstrap_handle_t nvshmemi_boot_handle;

extern nvshmemi_device_host_state_t nvshmemi_device_state;
extern nvshmemi_team_t **nvshmemi_team_pool;

#endif
