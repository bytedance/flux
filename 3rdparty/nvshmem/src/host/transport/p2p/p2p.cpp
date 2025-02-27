/*
 * Copyright (c) 2016-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "p2p.h"
#include <cuda_runtime.h>                             // for cudaD...
#include <driver_types.h>                             // for cudaD...
#include <nvml.h>                                     // for NVML_...
#include <stdio.h>                                    // for snprintf
#include <stdlib.h>                                   // for free
#include <string.h>                                   // for memset
#include "internal/host_transport/cudawrap.h"         // for CUPFN
#include "non_abi/nvshmemx_error.h"                   // for NVSHM...
#include "internal/host/debug.h"                      // for INFO
#include "internal/host/nvshmem_internal.h"           // for nvshm...
#include "internal/host/nvmlwrap.h"                   // for nvml_...
#include "internal/host/nvshmemi_symmetric_heap.hpp"  // for nvshm...
#include "internal/host/nvshmemi_mem_transport.hpp"   // for nvshm...
#include "internal/host/nvshmemi_types.h"             // for nvshm...
#include "internal/host/util.h"                       // for NVSHM...
#include "internal/host_transport/transport.h"        // for nvshm...

int nvshmemt_p2p_init(nvshmem_transport_t *transport);

int nvshmemt_p2p_show_info(struct nvshmem_transport *transport, int style) {
    /*XXX : not implemented*/
    return 0;
}

int nvshmemt_p2p_can_reach_peer(int *access, struct nvshmem_transport_pe_info *peer_info,
                                nvshmem_transport_t transport) {
    int status = 0;
    int found = 0;
    int p2p_connected = 0;
    CUdevice peer_cudev;
    int peer_devid;
    transport_p2p_state_t *p2p_state = (transport_p2p_state_t *)transport->state;
    int atomics_supported = 0;
    char remote_pcie_bus_id[NVSHMEM_PCIE_BDF_BUFFER_LEN];
    bool is_heap_vmm = false;
    struct nvml_function_table *nvml_ftable = nvshmemi_state->p2p_transport->get_nvml_ftable();

    nvmlReturn_t nvml_status;
    nvmlDevice_t local_device;
    nvmlDevice_t remote_device;
    nvmlGpuP2PStatus_t stat;

    INFO(NVSHMEM_TRANSPORT,
         "[%p] ndev %d pcie_devid %x cudevice %x peer host hash %lx p2p host hash %lx", p2p_state,
         p2p_state->ndev, peer_info->pcie_id.dev_id, p2p_state->cudevice, peer_info->hostHash,
         p2p_state->hostHash);

    /* Check if the peer GPU is connected via the MNNVL fabric */
    if (nvshmemi_state->p2p_transport->is_nvl_connected_pe(peer_info->pe)) {
        *access = NVSHMEM_TRANSPORT_CAP_MAP | NVSHMEM_TRANSPORT_CAP_MAP_GPU_ST |
                  NVSHMEM_TRANSPORT_CAP_MAP_GPU_LD | NVSHMEM_TRANSPORT_CAP_MAP_GPU_ATOMICS;
        goto out;
    }

    if (peer_info->hostHash != p2p_state->hostHash) {
        *access = 0;
        goto out;
    }

    /*find device with the given pcie id*/
    for (int j = 0; j < p2p_state->ndev; j++) {
        if ((p2p_state->pcie_ids[j].dev_id == peer_info->pcie_id.dev_id) &&
            (p2p_state->pcie_ids[j].bus_id == peer_info->pcie_id.bus_id) &&
            (p2p_state->pcie_ids[j].domain_id == peer_info->pcie_id.domain_id)) {
            peer_cudev = p2p_state->cudev[j];
            peer_devid = p2p_state->devid[j];
            found = 1;
            break;
        }
    }

    /** Check if heap is VMM type or not */
    if (nullptr !=
        dynamic_cast<nvshmemi_symmetric_heap_vidmem_dynamic_vmm *>(nvshmemi_state->heap_obj)) {
        is_heap_vmm = true;
    }

    /* In the case where we don't have access to the GPU directly,
     * and we aren't using VMM, look using NVML.
     */
    if (!found) {
        if (nvshmemi_cuda_driver_version >= 12000 || !is_heap_vmm) {
            status = snprintf(remote_pcie_bus_id, NVSHMEM_PCIE_BDF_BUFFER_LEN, "%x:%x:%x.0",
                              peer_info->pcie_id.domain_id, peer_info->pcie_id.bus_id,
                              peer_info->pcie_id.dev_id);
            if (status < 0 || status > NVSHMEM_PCIE_BDF_BUFFER_LEN) {
                INFO(NVSHMEM_TRANSPORT, "Unable to prepare buffer for NVML device detection.\n");
                status = 0;
                goto out;
            }

            status = 0;
            nvml_status =
                nvml_ftable->nvmlDeviceGetHandleByPciBusId(remote_pcie_bus_id, &remote_device);
            if (nvml_status != NVML_SUCCESS) {
                INFO(NVSHMEM_TRANSPORT, "Unable to dereference device by UUID using NVML.\n");
                goto out;
            }
            nvml_status =
                nvml_ftable->nvmlDeviceGetHandleByPciBusId(p2p_state->pcie_bdf, &local_device);
            if (nvml_status != NVML_SUCCESS) {
                INFO(NVSHMEM_TRANSPORT, "Unable to dereference device by UUID using NVML.\n");
                goto out;
            }
            nvml_status = nvml_ftable->nvmlDeviceGetP2PStatus(local_device, remote_device,
                                                              NVML_P2P_CAPS_INDEX_READ, &stat);
            if (nvml_status != NVML_SUCCESS) {
                *access = 0;
                INFO(
                    NVSHMEM_TRANSPORT,
                    "Unable to get read status using NVML. Disabling P2P communication for pe %d\n",
                    peer_info->pe);
                goto out;
            } else if (stat == NVML_P2P_STATUS_OK) {
                *access |= NVSHMEM_TRANSPORT_CAP_MAP | NVSHMEM_TRANSPORT_CAP_MAP_GPU_LD;
            }
            nvml_status = nvml_ftable->nvmlDeviceGetP2PStatus(local_device, remote_device,
                                                              NVML_P2P_CAPS_INDEX_WRITE, &stat);
            if (nvml_status != NVML_SUCCESS) {
                *access = 0;
                INFO(NVSHMEM_TRANSPORT,
                     "Unable to get write status using NVML. Disabling P2P communication for pe "
                     "%d\n",
                     peer_info->pe);
                goto out;
            } else if (stat == NVML_P2P_STATUS_OK) {
                *access |= NVSHMEM_TRANSPORT_CAP_MAP | NVSHMEM_TRANSPORT_CAP_MAP_GPU_ST;
            }
            nvml_status = nvml_ftable->nvmlDeviceGetP2PStatus(local_device, remote_device,
                                                              NVML_P2P_CAPS_INDEX_ATOMICS, &stat);
            if (nvml_status != NVML_SUCCESS) {
                INFO(NVSHMEM_TRANSPORT, "Unable to get atomic status using NVML.\n");
            } else if (stat == NVML_P2P_STATUS_OK) {
                *access |= NVSHMEM_TRANSPORT_CAP_MAP_GPU_ATOMICS;
            }
            goto out;
        } else {
            /* In the case of CUDA VMM, we can't export a memory handle so LD/ST is also not
             * available. */
            WARN(
                "Some CUDA devices are not visible,\n"
                "likely hidden by CUDA_VISIBLE_DEVICES. Using a network transport to reach these.\n"
                "Disabling VMM usage (dynamic heap) by setting NVSHMEM_DISABLE_CUDA_VMM=1 could "
                "provide better performance.");
            goto out;
        }
    }

    if (peer_cudev == p2p_state->cudevice) {
        *access = NVSHMEM_TRANSPORT_CAP_MAP | NVSHMEM_TRANSPORT_CAP_MAP_GPU_ST |
                  NVSHMEM_TRANSPORT_CAP_MAP_GPU_LD | NVSHMEM_TRANSPORT_CAP_MAP_GPU_ATOMICS;
        goto out;
    }

    // use CanAccessPeer if device is visible
    status = cudaDeviceCanAccessPeer(&p2p_connected, p2p_state->device_id, peer_devid);
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "cudaDeviceCanAccessPeer failed \n");

    if (p2p_connected) {
        *access = NVSHMEM_TRANSPORT_CAP_MAP | NVSHMEM_TRANSPORT_CAP_MAP_GPU_ST |
                  NVSHMEM_TRANSPORT_CAP_MAP_GPU_LD;
        status = cudaDeviceGetP2PAttribute(&atomics_supported, cudaDevP2PAttrNativeAtomicSupported,
                                           p2p_state->device_id, peer_devid);
        NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                              "cudaDeviceGetP2PAttribute failed \n");
        if (atomics_supported) {
            *access |= NVSHMEM_TRANSPORT_CAP_MAP_GPU_ATOMICS;
        }
    }

    if (nvshmemi_state->is_platform_nvl) {
        status = snprintf(remote_pcie_bus_id, NVSHMEM_PCIE_BDF_BUFFER_LEN, "%x:%x:%x.0",
                          peer_info->pcie_id.domain_id, peer_info->pcie_id.bus_id,
                          peer_info->pcie_id.dev_id);
        if (status < 0 || status > NVSHMEM_PCIE_BDF_BUFFER_LEN) {
            INFO(NVSHMEM_TRANSPORT, "Unable to prepare buffer for NVML device detection.\n");
            nvshmemi_state->is_platform_nvl = false;
            goto out;
        }

        status = 0;

        nvml_status =
            nvml_ftable->nvmlDeviceGetHandleByPciBusId(remote_pcie_bus_id, &remote_device);
        if (nvml_status != NVML_SUCCESS) {
            INFO(NVSHMEM_TRANSPORT,
                 "Unable to dereference device by UUID using NVML for NVL check.\n");
            nvshmemi_state->is_platform_nvl = false;
            goto out;
        }
        nvml_status =
            nvml_ftable->nvmlDeviceGetHandleByPciBusId(p2p_state->pcie_bdf, &local_device);
        if (nvml_status != NVML_SUCCESS) {
            INFO(NVSHMEM_TRANSPORT,
                 "Unable to dereference device by UUID using NVML for NVL check.\n");
            nvshmemi_state->is_platform_nvl = false;
            goto out;
        }

        nvml_status = nvml_ftable->nvmlDeviceGetP2PStatus(local_device, remote_device,
                                                          NVML_P2P_CAPS_INDEX_NVLINK, &stat);
        if (nvml_status != NVML_SUCCESS || stat != NVML_P2P_STATUS_OK) {
            nvshmemi_state->is_platform_nvl = false;
        }
    }

out:
    return status;
}

int nvshmemt_p2p_finalize(nvshmem_transport_t transport) {
    if (!transport) return 0;

    if (transport->state) {
        transport_p2p_state_t *p2p_state = (transport_p2p_state_t *)transport->state;

        free(p2p_state->cudev);

        free(p2p_state->pcie_ids);

        free(p2p_state);
    }

    free(transport);
    return 0;
}

int nvshmemt_p2p_init(nvshmem_transport_t *t) {
    int status = 0;
    int nbytes = 0;
    struct nvshmem_transport *transport = NULL;
    transport_p2p_state_t *p2p_state = NULL;

    transport = (struct nvshmem_transport *)malloc(sizeof(struct nvshmem_transport));
    NVSHMEMI_NULL_ERROR_JMP(transport, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "p2p transport allocation failed \n");
    memset(transport, 0, sizeof(struct nvshmem_transport));
    transport->is_successfully_initialized =
        false; /* set it to true after everything has been successfully initialized */

    p2p_state = (transport_p2p_state_t *)calloc(1, sizeof(transport_p2p_state_t));
    NVSHMEMI_NULL_ERROR_JMP(p2p_state, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "p2p state allocation failed \n");

    status = CUPFN(nvshmemi_cuda_syms, cuCtxGetDevice(&p2p_state->cudevice));
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "cuCtxGetDevice failed \n");

    p2p_state->hostHash = getHostHash();

    status = cudaGetDeviceCount(&p2p_state->ndev);
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "cudaGetDeviceCount failed \n");

    p2p_state->cudev = (CUdevice *)malloc(sizeof(CUdevice) * p2p_state->ndev);
    NVSHMEMI_NULL_ERROR_JMP(p2p_state->cudev, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "p2p dev array allocation failed \n");

    p2p_state->devid = (int *)malloc(sizeof(int) * p2p_state->ndev);
    NVSHMEMI_NULL_ERROR_JMP(p2p_state->devid, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "p2p dev array allocation failed \n");

    p2p_state->pcie_ids = (pcie_id_t *)malloc(sizeof(pcie_id_t) * p2p_state->ndev);
    NVSHMEMI_NULL_ERROR_JMP(p2p_state->pcie_ids, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "p2p pcie_ids array allocation failed \n");

    for (int i = 0; i < p2p_state->ndev; i++) {
        status = CUPFN(nvshmemi_cuda_syms, cuDeviceGet(&p2p_state->cudev[i], i));
        NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                              "cuDeviceGet failed \n");
        p2p_state->devid[i] = i;

        if (p2p_state->cudev[i] == p2p_state->cudevice) {
            p2p_state->device_id = i;
            cudaDeviceProp prop;
            status = cudaGetDeviceProperties(&prop, i);
            NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                                  "cudaGetDeviceProperties failed \n");
            nbytes = snprintf(p2p_state->pcie_bdf, NVSHMEM_PCIE_BDF_BUFFER_LEN, "%x:%x:%x.0",
                              prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
            if (nbytes < 0 || nbytes > NVSHMEM_PCIE_BDF_BUFFER_LEN) {
                status = NVSHMEMX_ERROR_INTERNAL;
                NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                   "Unable to set device pcie bdf for our local device.\n");
            }
        }

        status = nvshmemi_get_pcie_attrs(&p2p_state->pcie_ids[i], p2p_state->cudev[i]);
        NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                              "nvshmemi_get_pcie_attrs failed \n");
    }

    transport->host_ops.can_reach_peer = nvshmemt_p2p_can_reach_peer;
    transport->host_ops.finalize = nvshmemt_p2p_finalize;
    transport->host_ops.show_info = nvshmemt_p2p_show_info;

    transport->attr = NVSHMEM_TRANSPORT_ATTR_NO_ENDPOINTS;
    transport->state = p2p_state;
    transport->is_successfully_initialized = true;
    transport->no_proxy = true;

    *t = transport;
out:
    if (status) {
        NVSHMEMU_HOST_PTR_FREE(transport);
        if (p2p_state) {
            NVSHMEMU_HOST_PTR_FREE(p2p_state->cudev);
            NVSHMEMU_HOST_PTR_FREE(p2p_state->pcie_ids);
            NVSHMEMU_HOST_PTR_FREE(p2p_state);
        }
    }

    /* If p2p mem transport intiailization failed during discovery for platform reasons, mark p2p as
     * not ready for nvshmem initialization in the caller */
    if (nvshmemi_state->p2p_transport != nullptr) {
        if (nvshmemi_state->p2p_transport->is_initialized()) {
            return (status);
        } else {
            return (status | NVSHMEMX_ERROR_INTERNAL);
        }
    } else {
        return (status | NVSHMEMX_ERROR_INTERNAL);
    }
}
