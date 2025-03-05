/*
 * Copyright (c) 2016-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "topo.h"
#include <ctype.h>                                                         // for tolower
#include <cuda.h>                                                          // for CUDA_SUCCESS
#include <cuda_runtime.h>                                                  // for cudaDevice...
#include <driver_types.h>                                                  // for cudaDevice...
#include <limits.h>                                                        // for PATH_MAX
#include <stdio.h>                                                         // for NULL, fclose
#include <stdlib.h>                                                        // for free, calloc
#include <string.h>                                                        // for strlen
#include <list>                                                            // for _List_iter...
#include "non_abi/nvshmemx_error.h"                                        // for NVSHMEMX_E...
#include "internal/host/debug.h"                                           // for INFO, NVSH...
#include "internal/host/nvshmem_internal.h"                                // for nvshmemi_s...
#include "internal/host/nvshmemi_mem_transport.hpp"                        // for nvshm...
#include "internal/host/nvshmemi_types.h"                                  // for nvshmemi_state
#include "internal/host/util.h"                                            // for getHostHash
#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"  // for bootstrap_...
#include "internal/host_transport/cudawrap.h"                              // for CUPFN, nvs...
#include "bootstrap_host_transport/env_defs_internal.h"                    // for nvshmemi_o...
#include "internal/host_transport/nvshmemi_transport_defines.h"            // for pcie_id_t
#include "internal/host_transport/transport.h"                             // for nvshmem_tr...

#define MAX_BUSID_SIZE 16
#define MAXPATHSIZE 1024

bool nvshmemi_is_mpg_run = 0;

enum pe_device_assignment {
    PE_DEVICE_NOT_ASSIGNED = -1,
    PE_DEVICE_NO_OPTIMAL_ASSIGNMENT = -2,
};

/* Enumeration of possible PCIe paths and sister arrays for perf characteristics and string
 * representations */
enum pci_distance {
    PATH_PIX = 0,
    PATH_PXB = 1,
    PATH_PHB = 2,
    PATH_NODE = 3,
    PATH_SYS = 4,
    PATH_COUNT = 5
};
static const int pci_distance_perf[PATH_COUNT] = {4, 4, 3, 2, 1};
static const char *pci_distance_string[PATH_COUNT] = {"PIX", "PXB", "PHB", "NODE", "SYS"};

static int get_cuda_bus_id(int cuda_dev, char *bus_id) {
    int status = NVSHMEMX_SUCCESS;
    cudaError_t err;

    err = cudaDeviceGetPCIBusId(bus_id, MAX_BUSID_SIZE, cuda_dev);
    if (err != cudaSuccess) {
        NVSHMEMI_ERROR_PRINT("cudaDeviceGetPCIBusId failed with error: %d \n", err);
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }

out:
    return status;
}

static int get_numa_id(char *path) {
    char npath[PATH_MAX];
    snprintf(npath, PATH_MAX, "%s/numa_node", path);
    npath[PATH_MAX - 1] = '\0';

    int numaId = -1;
    FILE *file = fopen(npath, "r");
    if (file == NULL) return -1;
    if (fscanf(file, "%d", &numaId) == EOF) {
        fclose(file);
        return -1;
    }
    fclose(file);

    return numaId;
}

static int get_device_path(char *bus_id, char **path) {
    int status = NVSHMEMX_SUCCESS;
    char pathname[MAXPATHSIZE + 1];
    char *cuda_rpath;
    char bus_path[] = "/sys/class/pci_bus/0000:00/device";

    for (int i = 0; i < 16; i++) bus_id[i] = tolower(bus_id[i]);
    memcpy(bus_path + sizeof("/sys/class/pci_bus/") - 1, bus_id, sizeof("0000:00") - 1);

    cuda_rpath = realpath(bus_path, NULL);
    NVSHMEMI_NULL_ERROR_JMP(cuda_rpath, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "realpath failed \n");

    strncpy(pathname, cuda_rpath, MAXPATHSIZE);
    strncpy(pathname + strlen(pathname), "/", MAXPATHSIZE - strlen(pathname));
    strncpy(pathname + strlen(pathname), bus_id, MAXPATHSIZE - strlen(pathname));
    free(cuda_rpath);

    *path = realpath(pathname, NULL);
    NVSHMEMI_NULL_ERROR_JMP(*path, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "realpath failed \n");

out:
    return status;
}

static enum pci_distance get_pci_distance(char *cuda_path, char *mlx_path) {
    int score = 0;
    int depth = 0;
    int same = 1;
    size_t i;
    for (i = 0; i < strlen(cuda_path); i++) {
        if (cuda_path[i] != mlx_path[i]) same = 0;
        if (cuda_path[i] == '/') {
            depth++;
            if (same == 1) score++;
        }
    }
    if (score <= 3) {
        /* Split the former PATH_SOC distance into PATH_NODE and PATH_SYS based on numaId */
        int numaId1 = get_numa_id(cuda_path);
        int numaId2 = get_numa_id(mlx_path);
        return ((numaId1 == numaId2) ? PATH_NODE : PATH_SYS);
    }
    if (score == 4) return PATH_PHB;
    if (score == depth - 1) return PATH_PIX;
    return PATH_PXB;
}

typedef struct nvshmemi_path_pair_info {
    int pe_idx;
    int dev_idx;
    enum pci_distance pcie_distance;
} nvshmemi_path_pair_info_t;

int nvshmemi_get_devices_by_distance(int *device_arr, int max_dev_per_pe,
                                     struct nvshmem_transport *tcurr) {
    struct dev_info {
        char *dev_path;
        int use_count;
    } *dev_info_all = NULL;

    struct gpu_info {
        char gpu_bus_id[MAX_BUSID_SIZE];
    } gpu_info, *gpu_info_all = NULL;

    std::list<nvshmemi_path_pair_info_t> pe_dev_pairs;
    std::list<nvshmemi_path_pair_info_t>::iterator pairs_iter;

    int ndev = tcurr->n_devices;
    int mype = nvshmemi_state->mype;
    int n_pes = nvshmemi_state->npes;
    int n_pes_node = nvshmemi_state->npes_node;
    CUdevice gpu_device_id;

    char **cuda_device_paths = NULL;
    int *pe_selected_devices = NULL;
    enum pci_distance *pe_device_distance = NULL;
    int *used_devs = NULL;

    int mype_array_index = -1, mydev_index = -1;
    int i, dev_id, pe_id, pe_pair_index;
    int devices_assigned = 0;
    int mype_device_count = 0;
    int status = NVSHMEMX_ERROR_INTERNAL;

    if (ndev <= 0) {
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "transport devices (setup_connections) failed \n");
    }

    status = CUPFN(nvshmemi_cuda_syms, cuCtxGetDevice(&gpu_device_id));
    if (status != CUDA_SUCCESS) {
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }

    /* Allocate data structures start */
    /* Array of dev_info structures of size # of local NICs */
    dev_info_all = (struct dev_info *)calloc(ndev, sizeof(struct dev_info));
    NVSHMEMI_NULL_ERROR_JMP(dev_info_all, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "dev_info_all allocation failed \n");

    /* Array of GPU bus IDs of size n_pes*/
    gpu_info_all = (struct gpu_info *)calloc(n_pes, sizeof(struct gpu_info));
    NVSHMEMI_NULL_ERROR_JMP(gpu_info_all, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "gpu_info_all allocation failed \n");

    /* array linking each GPU on our node with it's pcie path */
    cuda_device_paths = (char **)calloc(n_pes_node, sizeof(char *));
    NVSHMEMI_NULL_ERROR_JMP(cuda_device_paths, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate memory for PE/NIC Mapping.\n");

    /* Array of size n_pes_node * max_dev_per_pe storing the accepted mappings of PE to Dev(s) */
    pe_selected_devices = (int *)calloc(n_pes_node * max_dev_per_pe, sizeof(int));
    NVSHMEMI_NULL_ERROR_JMP(pe_selected_devices, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate memory for PE/NIC Mapping.\n");
    for (pe_id = 0; pe_id < n_pes_node * max_dev_per_pe; pe_id++) {
        pe_selected_devices[pe_id] = -1;
    }

    pe_device_distance =
        (enum pci_distance *)calloc(n_pes_node * max_dev_per_pe, sizeof(enum pci_distance));
    NVSHMEMI_NULL_ERROR_JMP(pe_device_distance, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate memory for PE/NIC Mapping.\n");
    for (pe_id = 0; pe_id < n_pes_node * max_dev_per_pe; pe_id++) {
        pe_device_distance[pe_id] = PATH_COUNT;
    }

    used_devs = (int *)calloc(ndev, sizeof(int));
    NVSHMEMI_NULL_ERROR_JMP(used_devs, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate memory for PE/NIC Mapping.\n");
    /* Allocate data structures end */

    /* Gather GPU and NIC paths start */
    status = get_cuda_bus_id(gpu_device_id, gpu_info.gpu_bus_id);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "get cuda busid failed \n");

    status = nvshmemi_boot_handle.allgather((void *)&gpu_info, (void *)gpu_info_all,
                                            sizeof(struct gpu_info), &nvshmemi_boot_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "allgather of gpu_info failed \n");

    pe_id = 0;
    for (i = 0; i < n_pes; i++) {
        if (nvshmemi_state->pe_info[i].hostHash != nvshmemi_state->pe_info[mype].hostHash) {
            continue;
        }

        status = get_device_path(gpu_info_all[i].gpu_bus_id, &cuda_device_paths[pe_id]);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "get cuda path failed \n");
        /* to get back to our PE after the algorithm finishes. */
        if (i == mype) {
            mype_array_index = pe_id * max_dev_per_pe;
        }

        pe_id++;
        if (pe_id == n_pes_node) {
            break;
        }
    }

    if (pe_id != n_pes_node || mype_array_index == -1) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                           "Number of PEs found doesn't match the PE node count.\n");
    }

    for (i = 0; i < ndev; i++) {
        dev_info_all[i].dev_path = tcurr->device_pci_paths[i];
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "get device path failed \n");
    }
    /* Gather GPU and NIC paths end */

    /* Get path distances start */
    /* construct a n_pes_node * ndev array of distance measurements */
    for (pe_id = 0; pe_id < n_pes_node; pe_id++) {
        for (dev_id = 0; dev_id < ndev; dev_id++) {
            enum pci_distance distance_compare;
            distance_compare =
                get_pci_distance(cuda_device_paths[pe_id], dev_info_all[dev_id].dev_path);
            if (unlikely(pe_dev_pairs.empty())) {
                pe_dev_pairs.push_front({pe_id, dev_id, distance_compare});
            } else {
                for (pairs_iter = pe_dev_pairs.begin(); pairs_iter != pe_dev_pairs.end();
                     pairs_iter++) {
                    if (distance_compare < (*pairs_iter).pcie_distance) {
                        break;
                    }
                }
                INFO(NVSHMEM_TOPO, "PE %d: %s dev %d: %s distance: %d\n", pe_id,
                     cuda_device_paths[pe_id], dev_id, dev_info_all[dev_id].dev_path,
                     distance_compare);
                pe_dev_pairs.insert(pairs_iter, {pe_id, dev_id, distance_compare});
            }
        }
    }
    /* Get path distances end */

    /* loop one, do initial assignments of NIC(s) to each GPU */
    for (pairs_iter = pe_dev_pairs.begin(); pairs_iter != pe_dev_pairs.end(); pairs_iter++) {
        bool need_more_assignments = 0;
        int pe_base_index = (*pairs_iter).pe_idx * max_dev_per_pe;
        /* skip pairs where the GPU already has a partner in the first loop */
        for (pe_pair_index = 0; pe_pair_index < max_dev_per_pe; pe_pair_index++)
            if (pe_selected_devices[pe_base_index + pe_pair_index] == PE_DEVICE_NOT_ASSIGNED) {
                need_more_assignments = 1;
                break;
            }

        if (!need_more_assignments) {
            continue;
        }

        if (pci_distance_perf[(*pairs_iter).pcie_distance] <
            pci_distance_perf[pe_device_distance[pe_base_index]]) {
            /* This NIC and all subsequent ones are less optimal than the already selected NICs
             * They can be safely ignored and we assign -2 to indicate that there are no more
             * optimal NICs for this GPU.
             */
            for (; pe_pair_index < max_dev_per_pe; pe_pair_index++) {
                pe_selected_devices[pe_base_index + pe_pair_index] =
                    PE_DEVICE_NO_OPTIMAL_ASSIGNMENT;
                /* While not technically assigned, we need to account for these NICs to make
                 * forward progress.
                 */
                devices_assigned++;
            }
        } else {
            /* This NIC is optimal for this GPU. */
            INFO(NVSHMEM_TOPO, "Pairing PE %d with device %d at distance %d\n",
                 (*pairs_iter).pe_idx, (*pairs_iter).dev_idx, (*pairs_iter).pcie_distance);
            pe_selected_devices[pe_base_index + pe_pair_index] = (*pairs_iter).dev_idx;
            pe_device_distance[pe_base_index + pe_pair_index] = (*pairs_iter).pcie_distance;
            used_devs[(*pairs_iter).dev_idx]++;
            devices_assigned++;
        }

        if (devices_assigned == n_pes_node * max_dev_per_pe) {
            break;
        }
    }

    /* loop two, load balance the NICs. */
    for (pe_id = 0; pe_id < n_pes_node * max_dev_per_pe; pe_id++) {
        int nic_density;
        if (pe_selected_devices[pe_id] < 0) {
            continue;
        }
        nic_density = used_devs[pe_selected_devices[pe_id]];

        /* Can't find a less populated NIC if ours is only assigned to 1 gpu. */
        if (nic_density < 2) {
            continue;
        }

        for (pairs_iter = pe_dev_pairs.begin(); pairs_iter != pe_dev_pairs.end(); pairs_iter++) {
            /* Never change for a less optimal NIC. */

            if ((*pairs_iter).pe_idx != pe_id) {
                continue;
            }

            if (pci_distance_perf[(*pairs_iter).pcie_distance] <
                pci_distance_perf[pe_device_distance[pe_id]]) {
                break;
            }

            if ((nic_density - used_devs[(*pairs_iter).dev_idx]) >= 2) {
                INFO(NVSHMEM_TOPO, "Re-Pairing PE %d with device %d at distance %d\n",
                     (*pairs_iter).pe_idx, (*pairs_iter).dev_idx, (*pairs_iter).pcie_distance);
                used_devs[pe_selected_devices[pe_id]]--;
                used_devs[(*pairs_iter).dev_idx]++;
                nic_density = used_devs[(*pairs_iter).dev_idx];
                pe_selected_devices[pe_id] = (*pairs_iter).dev_idx;
                pe_device_distance[pe_id] = (*pairs_iter).pcie_distance;
                if (nic_density < 2) {
                    break;
                }
            }
        }
    }

    for (pe_pair_index = 0; pe_pair_index < max_dev_per_pe; pe_pair_index++) {
        if (pe_selected_devices[mype_array_index + pe_pair_index] >= 0) {
            mydev_index = pe_selected_devices[mype_array_index + pe_pair_index];
            device_arr[pe_pair_index] = mydev_index;
            mype_device_count++;
            INFO(NVSHMEM_TOPO, "Our PE is sharing its NIC at index %d with %d other PEs.\n",
                 used_devs[mydev_index], mype_device_count);
        }
    }

    if (mype_device_count == 0) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                           "No NICs were assigned to our PE.\n");
    }

    /* No need to report this in a loop - All Devices will have the same perf characteristics. */
    if (pci_distance_perf[pe_device_distance[mype_array_index]] < pci_distance_perf[PATH_PIX]) {
        nvshmemi_state->are_nics_ll128_compliant = false;
        INFO(NVSHMEM_TOPO,
             "Our PE is connected to a NIC with pci distance %s."
             "this will provide less than optimal performance.\n",
             pci_distance_string[pe_device_distance[mype_array_index]]);
    }

out:
    if (dev_info_all) {
        free(dev_info_all);
    }

    if (gpu_info_all) {
        free(gpu_info_all);
    }

    if (cuda_device_paths) {
        for (i = 0; i < n_pes_node; i++) {
            if (cuda_device_paths[i]) {
                free(cuda_device_paths[i]);
            }
        }
        free(cuda_device_paths);
    }

    pe_dev_pairs.clear();

    if (pe_selected_devices) {
        free(pe_selected_devices);
    }

    if (used_devs) {
        free(used_devs);
    }

    if (pe_device_distance) {
        free(pe_device_distance);
    }

    return status;
}

int nvshmemi_build_transport_map(nvshmemi_state_t *state) {
    int status = 0;
    int *local_map = NULL;

    state->transport_map = (int *)calloc(state->npes * state->npes, sizeof(int));
    NVSHMEMI_NULL_ERROR_JMP(state->transport_map, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "access map allocation failed \n");

    local_map = (int *)calloc(state->npes, sizeof(int));
    NVSHMEMI_NULL_ERROR_JMP(local_map, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "access map allocation failed \n");

    state->transport_bitmap = 0;

    for (int i = 0; i < state->npes; i++) {
        int reach_any = 0;

        for (int j = 0; j < state->num_initialized_transports; j++) {
            int reach = 0;

            if (!state->transports[j]) {
                continue;
            }

            status = state->transports[j]->host_ops.can_reach_peer(&reach, &state->pe_info[i],
                                                                   state->transports[j]);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "can reach peer failed \n");
            INFO(NVSHMEM_TOPO, "[%d] reach %d to peer %d over transport %d", state->mype, reach, i,
                 j);

            state->transports[j]->cap[i] = reach;
            reach_any |= reach;

            if (reach) {
                int m = 1 << j;
                local_map[i] |= m;
                /* Add transport to the bitmap if this is the first PE to use it. */
                if ((state->transport_bitmap & m) == 0) {
                    state->transport_bitmap |= m;
                }
            }
        }

        if ((!reach_any) && (!nvshmemi_options.BYPASS_ACCESSIBILITY_CHECK)) {
            status = NVSHMEMX_ERROR_NOT_SUPPORTED;
            fprintf(stderr, "%s:%d: [GPU %d] Peer GPU %d is not accessible, exiting ... \n",
                    __FILE__, __LINE__, state->mype, i);
            goto out;
        }
    }
    INFO(NVSHMEM_TOPO, "[%d] transport bitmap: %x", state->mype, state->transport_bitmap);

    status = nvshmemi_boot_handle.allgather((void *)local_map, (void *)state->transport_map,
                                            sizeof(int) * state->npes, &nvshmemi_boot_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "allgather of ipc handles failed \n");

out:
    if (local_map) free(local_map);
    if (status) {
        if (state->transport_map) free(state->transport_map);
    }
    return status;
}

int nvshmemi_get_pcie_attrs(pcie_id_t *pcie_id, int devid) {
    int status = 0;
    cudaDeviceProp prop;

    status = cudaGetDeviceProperties(&prop, devid);
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "cudaDeviceGetAttribute failed \n");
    pcie_id->dev_id = prop.pciDeviceID;
    pcie_id->bus_id = prop.pciBusID;
    pcie_id->domain_id = prop.pciDomainID;

out:
    return status;
}

int nvshmemi_detect_same_device(nvshmemi_state_t *state) {
    int status = 0;
    nvshmem_transport_pe_info_t my_info;

    my_info.pe = state->mype;
    status = nvshmemi_get_pcie_attrs(&my_info.pcie_id, state->device_id);
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "getPcieAttrs failed \n");

    my_info.hostHash = getHostHash();
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, state->device_id);
    my_info.gpu_uuid = prop.uuid;

    // TODO: move this to a topo init function as it is reused in other functions in topo that
    // follow
    state->pe_info =
        (nvshmem_transport_pe_info_t *)malloc(sizeof(nvshmem_transport_pe_info_t) * state->npes);
    NVSHMEMI_NULL_ERROR_JMP(state->pe_info, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "topo init info allocation failed \n");
    status =
        nvshmemi_boot_handle.allgather((void *)&my_info, (void *)state->pe_info,
                                       sizeof(nvshmem_transport_pe_info_t), &nvshmemi_boot_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "allgather of ipc handles failed \n");

    for (int i = 0; i < state->npes; i++) {
        (state->pe_info + i)->pe = i;
        if (i == state->mype) continue;

        status = (((state->pe_info + i)->hostHash == my_info.hostHash) &&
                  ((state->pe_info + i)->pcie_id.dev_id == my_info.pcie_id.dev_id) &&
                  ((state->pe_info + i)->pcie_id.bus_id == my_info.pcie_id.bus_id) &&
                  ((state->pe_info + i)->pcie_id.domain_id == my_info.pcie_id.domain_id));
        if (status) {
            INFO(NVSHMEM_INIT, "More than 1 PE per GPU detected. This is an MPG run.\n");
#if defined(NVSHMEM_PPC64LE)
            NVSHMEMI_ERROR_EXIT("MPG support is currently not available on P9 platforms");
#endif
            nvshmemi_is_mpg_run = 1;
            status = 0;
        }
        /*NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_NOT_SUPPORTED, out,
                     "two PEs sharing a GPU is not supported \n");*/
    }

out:
    if (status) {
        state->cucontext = NULL;
        if (!state->pe_info) free(state->pe_info);
    }
    return status;
}
