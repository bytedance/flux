/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef __TOPO_H
#define __TOPO_H
#include "internal/host/nvshmemi_types.h"  // for nvshmemi_state_t

int nvshmemi_get_devices_by_distance(int *device_arr, int max_dev_per_pe,
                                     struct nvshmem_transport *tcurr);
int nvshmemi_detect_same_device(nvshmemi_state_t *state);
int nvshmemi_build_transport_map(nvshmemi_state_t *state);

#endif
