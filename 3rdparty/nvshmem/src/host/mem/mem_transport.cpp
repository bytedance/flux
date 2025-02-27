/*
 * Copyright (c) 2016-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include <cuda.h>                                                          // for CUdevice
#include <cuda_runtime.h>                                                  // for cudaG...
#include <driver_types.h>                                                  // for cudaD...
#include <ext/alloc_traits.h>                                              // for __all...
#include <nvml.h>                                                          // for NVML_...
#include <stdint.h>                                                        // for uint64_t
#include <stdio.h>                                                         // for NULL
#include <stdlib.h>                                                        // for malloc
#include <string.h>                                                        // for memcmp
#include <unistd.h>                                                        // for pid_t
#include <map>                                                             // for map
#include <memory>                                                          // for alloc...
#include <vector>                                                          // for vector
#include "internal/host_transport/cudawrap.h"                              // for CUPFN
#include "non_abi/nvshmemx_error.h"                                        // for NVSHM...
#include "internal/host/debug.h"                                           // for INFO
#include "internal/host/nvshmem_internal.h"                                // for nvshm...
#include "internal/host/nvmlwrap.h"                                        // for nvmlG...
#include "internal/host/nvshmemi_symmetric_heap.hpp"                       // for nvshm...
#include "internal/host/nvshmemi_mem_transport.hpp"                        // for nvshm...
#include "internal/host/nvshmemi_types.h"                                  // for nvshm...
#include "internal/host/util.h"                                            // for NVSHM...
#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"  // for nvshm...
#include "internal/host_transport/nvshmemi_transport_defines.h"            // for nvshm...
#include "internal/host_transport/transport.h"                             // for nvshm...
#include "non_abi/nvshmem_build_options.h"                                 // for NVSHM...

/**
 * nvshmemi_mem_p2p_transport specific functions
 */

void nvshmemi_mem_p2p_transport::print_mem_handle(int pe_id, int transport_idx,
                                                  nvshmemi_symmetric_heap &obj) {
    int i = pe_id;
    int j = transport_idx;
    nvshmemi_state_t *state = obj.get_state();
    char *hex = nvshmemu_hexdump(&obj.handles_.back()[i * state->num_initialized_transports + j],
                                 sizeof(CUipcMemHandle));
    INFO(NVSHMEM_INIT, "[%d] cuIpcOpenMemHandle fromhandle 0x%s", state->mype, hex);
    NVSHMEMU_HOST_PTR_FREE(hex);
    INFO(NVSHMEM_INIT, "[%d] cuIpcOpenMemHandle tobuf %p", state->mype,
         *(obj.peer_heap_base_p2p_ + i));
}

int nvshmemi_mem_p2p_transport::create_proc_map(nvshmemi_symmetric_heap &obj) {
    pid_t pid = 0;
    pid_t *peer_pids = NULL;
    pid = getpid();
    int status = 0;
    if (proc_map_.size() > 0) {
        return 0;
    }
    peer_pids = (pid_t *)std::malloc(sizeof(pid_t) * obj.get_state()->npes);
    status = nvshmemi_boot_handle.allgather((void *)&pid, (void *)peer_pids, sizeof(pid_t),
                                            &nvshmemi_boot_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "allgather of pids failed \n");

    NVSHMEMU_FOR_EACH(pe, obj.get_state()->npes){NVSHMEMU_FOR_EACH_IF(
        j, obj.get_state()->num_initialized_transports,
        (NVSHMEMU_IS_BIT_SET(obj.get_state()->transport_bitmap, j) &&
         (obj.get_state()->transports[j]->cap[pe] & NVSHMEM_TRANSPORT_CAP_MAP)),
        { proc_map_[peer_pids[pe]] = pe; })}

    INFO(NVSHMEM_MEM, "I am connected to %lu p2p processes (including myself)", proc_map_.size());
out:
    NVSHMEMU_HOST_PTR_FREE(peer_pids);
    return (status);
}

nvshmemi_mem_p2p_transport::nvshmemi_mem_p2p_transport(int mype, int npes) {
    int status = 0;
    int nvml_status = 0;
    int device_id = -1;
    int ndev;
    int nbytes = 0;
    CUdevice cudevice;
    CUdevice *cudev = NULL;
    char pcie_bdf[NVSHMEM_PCIE_BDF_BUFFER_LEN] = {0};
    bool *peer_error_status = NULL;

    errored_on_initialization_ =
        true; /* By default, p2p is not initialized, so some features may be disabled */

    cudaDeviceProp prop;
    int flag = false;
    nvmlDevice_t local_device;
    nvmlGpuFabricInfoV_t fabricInfo = {}, fabricInfo1 = {}, fabricInfo2 = {};
    nvmlGpuFabricInfoV_t *pe_fabricInfo = nullptr;
    fabricInfo.version = nvmlGpuFabricInfo_v2;
    fabricInfo1.version = nvmlGpuFabricInfo_v2;
    fabricInfo2.version = nvmlGpuFabricInfo_v2;

    /* start NVML Library */
    nvml_status = nvshmemi_nvml_ftable_init(&nvml_ftable_, &nvml_handle_);
    if (nvml_status != NVML_SUCCESS) {
        status = NVSHMEMX_ERROR_INTERNAL;
        INFO(NVSHMEM_MEM, "Unable to open NVML. Some features will be disabled.");
        goto out;
    }

    nvml_status = nvml_ftable_.nvmlInit();
    if (nvml_status != NVML_SUCCESS) {
        status = NVSHMEMX_ERROR_INTERNAL;
        INFO(NVSHMEM_MEM, "Unable to initialize NVML. Some features will be disabled.");
        goto out;
    }

    /* Discover cudevice instance and device ID */
    status = cudaGetDeviceCount(&ndev);
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "cudaGetDeviceCount failed \n");

    status = CUPFN(nvshmemi_cuda_syms, cuCtxGetDevice(&cudevice));
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "cuCtxGetDevice failed \n");

    cudev = (CUdevice *)std::malloc(sizeof(CUdevice) * ndev);
    NVSHMEMI_NULL_ERROR_JMP(cudev, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "cudev array allocation failed \n");

    NVSHMEMU_FOR_EACH(i, ndev) {
        status = CUPFN(nvshmemi_cuda_syms, cuDeviceGet(&cudev[i], i));
        NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                              "cuDeviceGet failed \n");
        if (cudev[i] == cudevice) {
            device_id = i;
            cudaDeviceProp prop;
            status = cudaGetDeviceProperties(&prop, i);
            NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                                  "cudaGetDeviceProperties failed \n");
            nbytes = snprintf(pcie_bdf, NVSHMEM_PCIE_BDF_BUFFER_LEN, "%x:%x:%x.0", prop.pciDomainID,
                              prop.pciBusID, prop.pciDeviceID);
            if (nbytes < 0 || nbytes > NVSHMEM_PCIE_BDF_BUFFER_LEN) {
                status = NVSHMEMX_ERROR_INTERNAL;
                NVSHMEMI_ERROR_JMP(nbytes, NVSHMEMX_ERROR_INTERNAL, out,
                                   "Unable to set device pcie bdf for our local device.\n");
            }
        }
    }

    /* For the assigned device_id, discover nvmlDevice properties */
    cudaGetDeviceProperties(&prop, device_id);
    if (nvshmemi_cuda_driver_version >= 12040 && prop.major >= 9 &&
        !nvshmemi_options.DISABLE_MNNVL) {
        nvml_status = nvml_ftable_.nvmlDeviceGetHandleByPciBusId(pcie_bdf, &local_device);
        NVSHMEMI_NE_ERROR_JMP(nvml_status, NVML_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                              "nvmlDeviceGetHandleByPciBusId failed \n");

        /* Some platforms with older driver may not support this API, so bypass MNNVL discovery */
        if (nvml_ftable_.nvmlDeviceGetGpuFabricInfoV == NULL) {
            INFO(NVSHMEM_INIT,
                 "nvmlDeviceGetGpuFabricInfoV not found. Detection of MNNVL environment will not "
                 "be attempted\n");
            status |= NVSHMEMX_SUCCESS;
            goto out;
        }

        fabricInfo.clusterUuid[0] = '\0';
        nvml_status = nvml_ftable_.nvmlDeviceGetGpuFabricInfoV(local_device, &fabricInfo);
        NVSHMEMI_NE_ERROR_JMP(nvml_status, NVML_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                              "nvmlDeviceGetGpuFabricInfoV() failed... Detection of MNNVL "
                              "environment will not be attempted");

        pe_fabricInfo = (nvmlGpuFabricInfoV_t *)std::malloc(sizeof(nvmlGpuFabricInfoV_t) * npes);
        NVSHMEMI_NULL_ERROR_JMP(pe_fabricInfo, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                "pe_fabricInfo array allocation failed\n");

        pe_fabricInfo[mype] = fabricInfo;
        status =
            nvshmemi_boot_handle.allgather((void *)&pe_fabricInfo[mype], (void *)pe_fabricInfo,
                                           sizeof(nvmlGpuFabricInfoV_t), &nvshmemi_boot_handle);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "allgather of pe_fabricInfo failed \n");

        nvshmemi_has_mnnvl_fabric_ = 1;
        CUPFN(nvshmemi_cuda_syms,
              cuDeviceGetAttribute(
                  &flag,
                  static_cast<CUdevice_attribute>(CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED),
                  device_id));
        if (!flag) nvshmemi_has_mnnvl_fabric_ = 0;

        fabricInfo1 = pe_fabricInfo[mype];
        if (fabricInfo1.state < NVML_GPU_FABRIC_STATE_COMPLETED ||
            fabricInfo1.clusterUuid[0] == '\0')
            nvshmemi_has_mnnvl_fabric_ = 0;

        nvshmemi_mem_handle_type_ =
            (nvshmemi_has_mnnvl_fabric_ && flag)
                ? static_cast<CUmemAllocationHandleType>(CU_MEM_HANDLE_TYPE_FABRIC)
                : CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

        for (int i = 0; i < npes && nvshmemi_has_mnnvl_fabric_; i++) {
            fabricInfo2 = pe_fabricInfo[i];
            if ((fabricInfo2.state == NVML_GPU_FABRIC_STATE_COMPLETED) &&
                (fabricInfo2.clusterUuid[0] != '\0') &&
                (memcmp(fabricInfo1.clusterUuid, fabricInfo2.clusterUuid,
                        NVML_GPU_FABRIC_UUID_LEN) == 0) &&
                (fabricInfo1.cliqueId == fabricInfo2.cliqueId)) {
                nvshmemi_nvl_connected_pes_.push_back(i);
            }
        }

        if (nvshmemi_has_mnnvl_fabric_) {
            INFO(NVSHMEM_MEM, "Multi-node NVLink is supported and enabled on this platform");
        }
    }

    if (nvshmemi_options.CUMEM_HANDLE_TYPE_provided) {
        if (strcmp_case_insensitive(nvshmemi_options.CUMEM_HANDLE_TYPE, "FABRIC") == 0)
            nvshmemi_mem_handle_type_ = CU_MEM_HANDLE_TYPE_FABRIC;
        else
            nvshmemi_mem_handle_type_ = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    }

    INFO(NVSHMEM_MEM, "Symmetric Memory Heap Handle Type: %s\n",
         nvshmemi_mem_handle_type_ == CU_MEM_HANDLE_TYPE_FABRIC ? "Fabric Handle"
                                                                : "POSIX File Descriptor");
out:
    if (status == 0) errored_on_initialization_ = false;

    NVSHMEMU_HOST_PTR_FREE(cudev);
    if ((status || nvml_status) && nvml_ftable_.nvmlShutdown != NULL) {
        nvml_status = nvml_ftable_.nvmlShutdown();
        if (nvml_status != NVML_SUCCESS) {
            INFO(NVSHMEM_MEM, "Unable to stop NVML library in NVSHMEM.");
        }
        nvshmemi_nvml_ftable_fini(&nvml_ftable_, &nvml_handle_);
        if (status)
            INFO(NVSHMEM_MEM,
                 "Unable to intialize mem p2p transport (likely non-fatal). status = %d\n", status);
    }

    NVSHMEMU_HOST_PTR_FREE(pe_fabricInfo);
    /* Successful initialization on my PE and peer PEs
       This is to avoid a case where some PEs on some node are P2P reachable and some PEs on some
       node are not, causing asymmetry
    */
    peer_error_status = (bool *)std::calloc(sizeof(bool), npes);
    nvshmemi_boot_handle.allgather((void *)(&errored_on_initialization_), peer_error_status,
                                   sizeof(bool), &nvshmemi_boot_handle);
    NVSHMEMU_FOR_EACH(i, npes) {
        if (static_cast<int>(i) != mype && peer_error_status[i] != errored_on_initialization_) {
            errored_on_initialization_ = true;
            break;
        }
    }

    NVSHMEMU_HOST_PTR_FREE(peer_error_status);
}

nvshmemi_mem_p2p_transport::~nvshmemi_mem_p2p_transport() {
    int nvml_status = 0;
    if (nvml_handle_) {
        nvml_status = nvml_ftable_.nvmlShutdown();
        if (nvml_status != NVML_SUCCESS) {
            INFO(NVSHMEM_MEM, "Unable to stop NVML library in NVSHMEM.");
        }
        nvshmemi_nvml_ftable_fini(&nvml_ftable_, &nvml_handle_);
    }
    proc_map_.clear();
    if (p2p_objref_ != nullptr) p2p_objref_ = nullptr;
}

/**
 * nvshmemi_mem_remote_transport specific functions
 */
int nvshmemi_mem_remote_transport::gather_mem_handles(nvshmemi_symmetric_heap &obj,
                                                      uint64_t heap_offset, size_t size) {
    int status = 0;

    NVSHMEMU_FOR_EACH(i, obj.get_state()->num_initialized_transports) {
        nvshmem_transport_t tcurr = obj.get_state()->transports[i];
        if (NVSHMEMU_IS_BIT_SET(obj.get_state()->transport_bitmap, i) &&
            NVSHMEMI_TRANSPORT_OPS_IS_ADD_DEVICE_REMOTE_MEM(tcurr)) {
            status = tcurr->host_ops.add_device_remote_mem_handles(
                tcurr, obj.get_state()->num_initialized_transports, obj.handles_.back().data(),
                heap_offset, size);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "add_device_remote_mem_handles failed \n");

            status = nvshmemi_update_device_state();
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "nvshmemi_update_device_state() failed \n");
        }
    }
out:
    return status;
}

int nvshmemi_mem_remote_transport::register_mem_handle(nvshmem_mem_handle_t *local_handles,
                                                       int transport_idx, nvshmem_mem_handle_t *in,
                                                       void *buf, size_t size,
                                                       nvshmem_transport_t current) {
    if (!NVSHMEMI_TRANSPORT_OPS_IS_GET_MEM(current)) return 0;
    return current->host_ops.get_mem_handle((nvshmem_mem_handle_t *)(local_handles + transport_idx),
                                            in, buf, size, current, false);
}

int nvshmemi_mem_remote_transport::release_mem_handles(nvshmem_mem_handle_t *handles,
                                                       nvshmemi_symmetric_heap &obj) {
    int status = 0;
    NVSHMEMU_FOR_EACH_IF(i, obj.get_state()->num_initialized_transports,
                         NVSHMEMU_IS_BIT_SET(obj.get_state()->transport_bitmap, i) &&
                             NVSHMEMI_TRANSPORT_OPS_IS_RELEASE_MEM(obj.get_state()->transports[i]),
                         {
                             if (!is_mem_handle_null(&handles[i])) {
                                 status =
                                     obj.get_state()->transports[i]->host_ops.release_mem_handle(
                                         &handles[i], obj.get_state()->transports[i]);
                                 NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                                       "transport release memhandle failed \n");
                             }
                         })
out:
    return status;
}
