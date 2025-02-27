/*
 * Copyright (c) 2016-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _P2P_H
#define _P2P_H

#include <stdint.h>  // IWYU pragma: keep
// IWYU pragma: no_include <bits/stdint-uintn.h>
#include <cuda.h>
#include "internal/host_transport/nvshmemi_transport_defines.h"

typedef struct {
    int ndev;
    CUdevice *cudev;
    int *devid;
    CUdeviceptr *curetval;
    CUdevice cudevice;
    int device_id;
    uint64_t hostHash;
    pcie_id_t *pcie_ids;
    char pcie_bdf[NVSHMEM_PCIE_BDF_BUFFER_LEN];
} transport_p2p_state_t;

#endif
