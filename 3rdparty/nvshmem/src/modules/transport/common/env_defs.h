/****
 * Copyright (c) 2016-2024, NVIDIA Corporation.  All rights reserved.
 *
 * Copyright 2011 Sandia Corporation. Under the terms of Contract
 * DE-AC04-94AL85000 with Sandia Corporation, the U.S.  Government
 * retains certain rights in this software.
 *
 * Copyright (c) 2017 Intel Corporation. All rights reserved.
 * This software is available to you under the BSD license.
 *
 * Portions of this file are derived from Sandia OpenSHMEM.
 *
 * See COPYRIGHT for license information
 ****/

/* NVSHMEMI_ENV_DEF( name, kind, default, category, short description )
 *
 * Kinds: long, size, bool, string
 * Categories: NVSHMEMI_ENV_CAT_OPENSHMEM, NVSHMEMI_ENV_CAT_OTHER,
 *             NVSHMEMI_ENV_CAT_COLLECTIVES, NVSHMEMI_ENV_CAT_TRANSPORT,
 *             NVSHMEMI_ENV_CAT_HIDDEN
 */

// for NVSHMEM_IBGDA_SUPPORT
#include "non_abi/nvshmem_build_options.h"  // IWYU pragma: keep

#ifndef NVSHMEM_ENV_DEFS_INTERNAL
#include "bootstrap_host_transport/env_defs_internal.h"  // for NVSHMEMI_ENV_DEF
#endif

#ifdef NVSHMEMI_ENV_DEF

NVSHMEMI_ENV_DEF(DEBUG, string, "", NVSHMEMI_ENV_CAT_OPENSHMEM,
                 "Set to enable debugging messages.\n"
                 "Optional values: VERSION, WARN, INFO, ABORT, TRACE")

/** Debugging **/

NVSHMEMI_ENV_DEF(DEBUG_SUBSYS, string, "", NVSHMEMI_ENV_CAT_HIDDEN,
                 "Comma separated list of debugging message sources. Prefix with '^' to exclude.\n"
                 "Values: INIT, COLL, P2P, PROXY, TRANSPORT, MEM, BOOTSTRAP, TOPO, UTIL, ALL")
/** Transport **/

NVSHMEMI_ENV_DEF(DISABLE_IB_NATIVE_ATOMICS, bool, false, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Disable use of InfiniBand native atomics")
NVSHMEMI_ENV_DEF(DISABLE_GDRCOPY, bool, false, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Disable use of GDRCopy in IB RC Transport")
NVSHMEMI_ENV_DEF(IB_DISABLE_DMABUF, bool, false, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Disable use of DMABUF in IBRC/IBDEVX/IBGDA Transports")
NVSHMEMI_ENV_DEF(IB_GID_INDEX, int, -1, NVSHMEMI_ENV_CAT_TRANSPORT, "Source GID Index for ROCE")
NVSHMEMI_ENV_DEF(IB_TRAFFIC_CLASS, int, 0, NVSHMEMI_ENV_CAT_TRANSPORT, "Traffic calss for ROCE")
NVSHMEMI_ENV_DEF(IB_ADDR_FAMILY, string, "AF_INET", NVSHMEMI_ENV_CAT_TRANSPORT,
                 "IP address family associated to IB GID "
                 "dynamically selected by NVSHMEM when NVSHMEM_IB_GID_INDEX is left unset")
NVSHMEMI_ENV_DEF(
    IB_ADDR_RANGE, string, "::/0", NVSHMEMI_ENV_CAT_TRANSPORT,
    "Defines the range of "
    "valid GIDs dynamically selected by NVSHMEM when NVSHMEM_IB_GID_INDEX is left unset")
NVSHMEMI_ENV_DEF(IB_ROCE_VERSION_NUM, int, 2, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "ROCE version associated to IB GID dynamically selected by NVSHMEM "
                 "when NVSHMEM_IB_GID_INDEX is left unset")
NVSHMEMI_ENV_DEF(IB_SL, int, 0, NVSHMEMI_ENV_CAT_TRANSPORT, "Service level to use over IB/ROCE")
NVSHMEMI_ENV_DEF(
    IB_ENABLE_RELAXED_ORDERING, bool, true, NVSHMEMI_ENV_CAT_TRANSPORT,
    "Enable PCIe relaxed ordering on transports over IB/ROCE (e.g., IBRC, IBGDA, IBDEVX)")

NVSHMEMI_ENV_DEF(HCA_LIST, string, "", NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Comma-separated list of HCAs to use in the NVSHMEM application. Entries "
                 "are of the form ``hca_name:port``, e.g. ``mlx5_1:1,mlx5_2:2`` and entries "
                 "prefixed by ^ are excluded. ``NVSHMEM_ENABLE_NIC_PE_MAPPING`` must be set to "
                 "1 for this variable to be effective.")

NVSHMEMI_ENV_DEF(HCA_PE_MAPPING, string, "", NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Specifies mapping of HCAs to PEs as a comma-separated list. Each entry "
                 "in the comma separated list is of the form ``hca_name:port:count``.  For "
                 "example, ``mlx5_0:1:2,mlx5_0:2:2`` indicates that PE0, PE1 are mapped to "
                 "port 1 of mlx5_0, and PE2, PE3 are mapped to port 2 of mlx5_0. "
                 "``NVSHMEM_ENABLE_NIC_PE_MAPPING`` must be set to 1 for this variable to be "
                 "effective.")
NVSHMEMI_ENV_DEF(QP_DEPTH, int, 1024, NVSHMEMI_ENV_CAT_HIDDEN, "Number of WRs in QP")
NVSHMEMI_ENV_DEF(SRQ_DEPTH, int, 16384, NVSHMEMI_ENV_CAT_HIDDEN, "Number of WRs in SRQ")

NVSHMEMI_ENV_DEF(DISABLE_LOCAL_ONLY_PROXY, bool, false, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "When running on an NVLink-only configuaration (No-IB, No-UCX), completely "
                 "disable the proxy thread. This will disable device side global exit and "
                 "device side wait timeout polling (enabled by ``NVSHMEM_TIMEOUT_DEVICE_POLLING`` "
                 "build-time variable) because these are processed by the proxy thread.")

NVSHMEMI_ENV_DEF(LIBFABRIC_PROVIDER, string, "cxi", NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Set the feature set provider for the libfabric transport: cxi, efa, verbs")

#if defined(NVSHMEM_IBGDA_SUPPORT) || defined(NVSHMEM_ENV_ALL)
/** GPU-initiated communication **/
NVSHMEMI_ENV_DEF(IBGDA_ENABLE_MULTI_PORT, bool, false, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Enable multiple NICs per PE if available. Note: Enabling this on "
                 "Hopper+ for latency sensitive applications is discouraged.")
NVSHMEMI_ENV_DEF(IBGDA_NUM_DCT, int, 2, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Number of DCT QPs used in GPU-initiated communication transport.")
NVSHMEMI_ENV_DEF(IBGDA_NUM_DCI, int, 1, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Total number of DCI QPs used in GPU-initiated communication transport. "
                 "Set to 0 or a negative number to use automatic configuration.")
NVSHMEMI_ENV_DEF(IBGDA_NUM_SHARED_DCI, int, 1, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Number of DCI QPs in the shared pool. "
                 "The rest of DCI QPs (NVSHMEM_IBGDA_NUM_DCI - NVSHMEM_IBGDA_NUM_SHARED_DCI) are "
                 "exclusively assigned. "
                 "Valid value: [1, NVSHMEM_IBGDA_NUM_DCI].")
NVSHMEMI_ENV_DEF(IBGDA_DCI_MAP_BY, string, "cta", NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Specifies how exclusive DCI QPs are assigned. "
                 "Choices are: cta, sm, warp, dct.\n\n"
                 "- cta: round-robin by CTA ID (default).\n"
                 "- sm: round-robin by SM ID.\n"
                 "- warp: round-robin by Warp ID.\n"
                 "- dct: round-robin by DCT ID.")
NVSHMEMI_ENV_DEF(IBGDA_NUM_RC_PER_PE, int, 2, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Number of RC QPs per peer PE used in GPU-initiated communication transport. "
                 "Set to 0 to disable RC QPs (default 2). "
                 "If set to a positive number, DCI will be used for enforcing consistency only.")
NVSHMEMI_ENV_DEF(IBGDA_RC_MAP_BY, string, "cta", NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Specifies how RC QPs are assigned. "
                 "Choices are: cta, sm, warp.\n\n"
                 "- cta: round-robin by CTA ID (default).\n"
                 "- sm: round-robin by SM ID.\n"
                 "- warp: round-robin by Warp ID.")
NVSHMEMI_ENV_DEF(IBGDA_FORCE_NIC_BUF_MEMTYPE, string, "gpumem", NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Force NIC buffer memory type. Valid choices are: gpumem (default), hostmem. "
                 "For other values, use auto discovery.")
NVSHMEMI_ENV_DEF(IBGDA_NUM_REQUESTS_IN_BATCH, int, 32, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Number of requests to be batched before submitting to the NIC. "
                 "It will be rounded up to the nearest power of 2. "
                 "Set to 1 for aggressive submission.")
NVSHMEMI_ENV_DEF(IBGDA_NUM_FETCH_SLOTS_PER_DCI, int, 1024, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Number of internal buffer slots for fetch operations for each DCI QP. "
                 "It will be rounded up to the nearest power of 2.")
NVSHMEMI_ENV_DEF(IBGDA_NUM_FETCH_SLOTS_PER_RC, int, 1024, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Number of internal buffer slots for fetch operations for each RC QP. "
                 "It will be rounded up to the nearest power of 2.")
NVSHMEMI_ENV_DEF(IBGDA_NIC_HANDLER, string, "auto", NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Specifies the processor used for ringing NIC's DB. "
                 "Choices are: auto, gpu, cpu.\n\n"
                 "- auto: use GPU SMs and fallback to CPU if it is not supported (default).\n"
                 "- gpu: use GPU SMs.\n"
                 "- cpu: use CPU.")
NVSHMEMI_ENV_DEF(IB_ENABLE_IBGDA, bool, false, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Set to enable GPU-initiated communication transport.")
#endif

#endif
