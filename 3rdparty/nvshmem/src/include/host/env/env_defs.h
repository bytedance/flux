/****
 * Copyright (c) 2016-2023, NVIDIA Corporation.  All rights reserved.
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

#include <stddef.h>  // for size_t
#ifndef NVSHMEM_ENV_DEFS_INTERNAL
#include "bootstrap_host_transport/env_defs_internal.h"  // IWYU pragma: keep
#endif
#include "non_abi/nvshmem_build_options.h"  // for NVSHMEM_IBGDA_SUPPORT

#ifdef NVSHMEMI_ENV_DEF

NVSHMEMI_ENV_DEF(VERSION, bool, false, NVSHMEMI_ENV_CAT_OPENSHMEM,
                 "Print library version at startup")
NVSHMEMI_ENV_DEF(INFO, bool, false, NVSHMEMI_ENV_CAT_OPENSHMEM,
                 "Print environment variable options at startup")
NVSHMEMI_ENV_DEF(INFO_HIDDEN, bool, false, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Print hidden environment variable options at startup")
NVSHMEMI_ENV_DEF(DISABLE_NVLS, bool, false, NVSHMEMI_ENV_CAT_OPENSHMEM,
                 "Disable NVLS SHARP resources for collectives, even if available for platform")

NVSHMEMI_ENV_DEF(DISABLE_NVLS_SHARING, bool, true, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Disable NVLS SHARP resource sharing for user-defined teams")

NVSHMEMI_ENV_DEF(SYMMETRIC_SIZE, size, (size_t)(SYMMETRIC_SIZE_DEFAULT), NVSHMEMI_ENV_CAT_OPENSHMEM,
                 "Specifies the size (in bytes) of the symmetric heap memory per PE. The resulting "
                 "size is implementation-defined and must be at least as large as the integer "
                 "ceiling of the product of the numeric prefix and the scaling factor. The allowed "
                 "character suffixes for the scaling factor are as follows:\n"
                 "\n"
                 "  *  k or K multiplies by 2^10 (kibibytes)\n"
                 "  *  m or M multiplies by 2^20 (mebibytes)\n"
                 "  *  g or G multiplies by 2^30 (gibibytes)\n"
                 "  *  t or T multiplies by 2^40 (tebibytes)\n"
                 "\n"
                 "For example, string '20m' is equivalent to the integer value 20971520, or 20 "
                 "mebibytes. Similarly the string '3.1M' is equivalent to the integer value "
                 "3250586. Only one multiplier is recognized and any characters following the "
                 "multiplier are ignored, so '20kk' will not produce the same result as '20m'. "
                 "Usage of string '.5m' will yield the same result as the string '0.5m'.\n"
                 "An invalid value for ``NVSHMEM_SYMMETRIC_SIZE`` is an error, which the NVSHMEM "
                 "library shall report by either returning a nonzero value from "
                 "``nvshmem_init_thread`` or causing program termination.")
NVSHMEMI_ENV_DEF(HEAP_KIND, string, "DEVICE", NVSHMEMI_ENV_CAT_HIDDEN,
                 "Specify the memory kind used by the NVSHMEM symmetric heap.\n"
                 "Allowed values: VIDMEM, SYSMEM")
NVSHMEMI_ENV_DEF(ENABLE_RAIL_OPT, bool, 0, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Enable Rail Optimization when heap is in SYSMEM")
NVSHMEMI_ENV_DEF(DEBUG, string, "", NVSHMEMI_ENV_CAT_OPENSHMEM,
                 "Set to enable debugging messages.\n"
                 "Optional values: VERSION, WARN, INFO, ABORT, TRACE")

/** Bootstrap **/

NVSHMEMI_ENV_DEF(BOOTSTRAP, string, "PMI", NVSHMEMI_ENV_CAT_BOOTSTRAP,
                 "Name of the default bootstrap that should be used to initialize NVSHMEM.\n"
                 "Allowed values: PMI, MPI, SHMEM, plugin, UID")

#if defined(NVSHMEM_DEFAULT_PMIX)
#define NVSHMEMI_ENV_BOOTSTRAP_PMI_DEFAULT "PMIX"
#elif defined(NVSHMEM_DEFAULT_PMI2)
#define NVSHMEMI_ENV_BOOTSTRAP_PMI_DEFAULT "PMI-2"
#else
#define NVSHMEMI_ENV_BOOTSTRAP_PMI_DEFAULT "PMI"
#endif

NVSHMEMI_ENV_DEF(BOOTSTRAP_PMI, string, NVSHMEMI_ENV_BOOTSTRAP_PMI_DEFAULT,
                 NVSHMEMI_ENV_CAT_BOOTSTRAP,
                 "Name of the PMI bootstrap that should be used to initialize NVSHMEM.\n"
                 "Allowed values: PMI, PMI-2, PMIX")

#undef NVSHMEMI_ENV_BOOTSTRAP_PMI_DEFAULT

NVSHMEMI_ENV_DEF(BOOTSTRAP_PLUGIN, string, "", NVSHMEMI_ENV_CAT_BOOTSTRAP,
                 "Absolute path to or name of the bootstrap plugin file to load "
                 "when NVSHMEM_BOOTSTRAP=plugin is specified")

NVSHMEMI_ENV_DEF(BOOTSTRAP_MPI_PLUGIN, string, "nvshmem_bootstrap_mpi.so",
                 NVSHMEMI_ENV_CAT_BOOTSTRAP,
                 "Absolute path to or name of the MPI bootstrap "
                 "plugin file. \nNVSHMEM will search for the plugin based on linux linker "
                 "priorities. See `man dlopen`")

NVSHMEMI_ENV_DEF(BOOTSTRAP_SHMEM_PLUGIN, string, "nvshmem_bootstrap_shmem.so",
                 NVSHMEMI_ENV_CAT_BOOTSTRAP,
                 "Absolute path to or name of the SHMEM bootstrap "
                 "plugin file. \nNVSHMEM will search for the plugin based on linux linker "
                 "priorities. See `man dlopen`")

NVSHMEMI_ENV_DEF(BOOTSTRAP_PMI_PLUGIN, string, "nvshmem_bootstrap_pmi.so",
                 NVSHMEMI_ENV_CAT_BOOTSTRAP,
                 "Absolute path to or name of the PMI bootstrap "
                 "plugin file. \nNVSHMEM will search for the plugin based on linux linker "
                 "priorities. See `man dlopen`")

NVSHMEMI_ENV_DEF(BOOTSTRAP_PMI2_PLUGIN, string, "nvshmem_bootstrap_pmi2.so",
                 NVSHMEMI_ENV_CAT_BOOTSTRAP,
                 "Absolute path to or name of the PMI-2 bootstrap "
                 "plugin file. \nNVSHMEM will search for the plugin based on linux linker "
                 "priorities. See `man dlopen`")

NVSHMEMI_ENV_DEF(BOOTSTRAP_PMIX_PLUGIN, string, "nvshmem_bootstrap_pmix.so",
                 NVSHMEMI_ENV_CAT_BOOTSTRAP,
                 "Absolute path to or name of the PMIx bootstrap "
                 "plugin file. \nNVSHMEM will search for the plugin based on linux linker "
                 "priorities. See `man dlopen`")

NVSHMEMI_ENV_DEF(BOOTSTRAP_UID_PLUGIN, string, "nvshmem_bootstrap_uid.so",
                 NVSHMEMI_ENV_CAT_BOOTSTRAP,
                 "Absolute path to or name of the UID bootstrap "
                 "plugin file. \nNVSHMEM will search for the plugin based on linux linker "
                 "priorities. See `man dlopen`")

NVSHMEMI_ENV_DEF(BOOTSTRAP_TWO_STAGE, bool, false, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Ignore CUDA device setting during initialization,"
                 "forcing two-stage initialization")

/** Library initialization **/
NVSHMEMI_ENV_DEF(CUDA_PATH, string, "", NVSHMEMI_ENV_CAT_OTHER,
                 "Path to directory containing libcuda.so (for use when not in default location)")

/** Debugging **/

NVSHMEMI_ENV_DEF(DEBUG_ATTACH_DELAY, int, 0, NVSHMEMI_ENV_CAT_OTHER,
                 "Delay (in seconds) during the first call to NVSHMEM_INIT to allow for attaching "
                 "a debuggger (Default 0)")

NVSHMEMI_ENV_DEF(DEBUG_SUBSYS, string, "", NVSHMEMI_ENV_CAT_HIDDEN,
                 "Comma separated list of debugging message sources. Prefix with '^' to exclude.\n"
                 "Values: INIT, COLL, P2P, PROXY, TRANSPORT, MEM, BOOTSTRAP, TOPO, UTIL, ALL")
NVSHMEMI_ENV_DEF(DEBUG_FILE, string, "", NVSHMEMI_ENV_CAT_OTHER,
                 "Debugging output filename, may contain %h for hostname and %p for pid")
NVSHMEMI_ENV_DEF(ENABLE_ERROR_CHECKS, bool, false, NVSHMEMI_ENV_CAT_HIDDEN, "Enable error checks")

NVSHMEMI_ENV_DEF(MAX_TEAMS, long, 32l, NVSHMEMI_ENV_CAT_OTHER,
                 "Maximum number of simultaneous teams allowed")

NVSHMEMI_ENV_DEF(MAX_P2P_GPUS, int, 128, NVSHMEMI_ENV_CAT_OTHER, "Maximum number of P2P GPUs")
NVSHMEMI_ENV_DEF(MAX_MEMORY_PER_GPU, size, (size_t)((size_t)128 * (1 << 30)),
                 NVSHMEMI_ENV_CAT_OTHER, "Maximum memory per GPU")
#if defined(NVSHMEM_PPC64LE)
#define NVSHMEMI_ENV_DISABLE_CUDA_VMM_DEFAULT true
#else
#define NVSHMEMI_ENV_DISABLE_CUDA_VMM_DEFAULT false
#endif

NVSHMEMI_ENV_DEF(DISABLE_CUDA_VMM, bool, NVSHMEMI_ENV_DISABLE_CUDA_VMM_DEFAULT,
                 NVSHMEMI_ENV_CAT_OTHER,
                 "Disable use of CUDA VMM for P2P memory mapping. By default, CUDA VMM is enabled "
                 "on x86 and disabled on P9. CUDA VMM feature in NVSHMEM requires CUDA RT version "
                 "and CUDA Driver version to be greater than or equal to 11.3.")

#undef NVSHMEMI_ENV_DISABLE_CUDA_VMM_DEFAULT

NVSHMEMI_ENV_DEF(DISABLE_P2P, bool, false, NVSHMEMI_ENV_CAT_OTHER,
                 "Disable P2P connectivity of GPUs even when available")

NVSHMEMI_ENV_DEF(DISABLE_MNNVL, bool, false, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Disable MNNVL connectivity for GPUs even when available")

NVSHMEMI_ENV_DEF(IGNORE_CUDA_MPS_ACTIVE_THREAD_PERCENTAGE, bool, false, NVSHMEMI_ENV_CAT_OTHER,
                 "When doing Multi-Process Per GPU (MPG) run, full API support is available "
                 "only if sum of CUDA_MPS_ACTIVE_THREAD_PERCENTAGE of processes running on a "
                 "GPU is <= 100%. Through this variable, user can request NVSHMEM runtime to "
                 "ignore the active thread percentage and allow full MPG support. Users "
                 "enable it at their own risk as NVSHMEM might deadlock.")
NVSHMEMI_ENV_DEF(CUMEM_GRANULARITY, size, (size_t)((size_t)1 << 29), NVSHMEMI_ENV_CAT_OTHER,
                 "Granularity for ``cuMemAlloc``/``cuMemCreate``")

NVSHMEMI_ENV_DEF(CUMEM_HANDLE_TYPE, string, "FILE_DESCRIPTOR", NVSHMEMI_ENV_CAT_HIDDEN,
                 "Handle type for ``cuMemCreate``. Supported are - FABRIC or FILE_DESCRIPTOR")

NVSHMEMI_ENV_DEF(BYPASS_ACCESSIBILITY_CHECK, bool, false, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Bypass peer GPU accessbility checks")

#if defined(NVSHMEM_PPC64LE) || defined(NVSHMEM_ENV_ALL)
NVSHMEMI_ENV_DEF(CUDA_LIMIT_STACK_SIZE, size, (size_t)(0), NVSHMEMI_ENV_CAT_OTHER,
                 "Specify limit on stack size of each GPU thread on P9")
#endif

/** General Collectives **/

NVSHMEMI_ENV_DEF(
    FCOLLECT_NTHREADS, int, 512, NVSHMEMI_ENV_CAT_HIDDEN,
    "Sets number of threads per block for fcollect collective.\n"
    "By default, if no env is set, default value is min(max_occupancy per CTA, msg size per PE).\n"
    "If env is specified, value overrides the default irrespective of max occupancy per CTA\n")

NVSHMEMI_ENV_DEF(
    REDUCESCATTER_NTHREADS, int, 512, NVSHMEMI_ENV_CAT_HIDDEN,
    "Sets number of threads per block for reducescatter collective.\n"
    "By default, if no env is set, default value is min(max_occupancy per CTA, msg size per PE).\n"
    "If env is specified, value overrides the default irrespective of max occupancy per CTA\n")

NVSHMEMI_ENV_DEF(MAX_CTAS, int, 1, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Sets number of blocks per grid for host onstream collective.\n"
                 "By default, if no env is set, default value to 1 CTA\n"
                 "If env is specified, value overrides the default value\n")

NVSHMEMI_ENV_DEF(DISABLE_NCCL, bool, false, NVSHMEMI_ENV_CAT_COLLECTIVES,
                 "Disable use of NCCL for collective operations")
NVSHMEMI_ENV_DEF(BARRIER_DISSEM_KVAL, int, 2, NVSHMEMI_ENV_CAT_COLLECTIVES,
                 "Radix of the dissemination algorithm used for barriers")
NVSHMEMI_ENV_DEF(BARRIER_TG_DISSEM_KVAL, int, 2, NVSHMEMI_ENV_CAT_COLLECTIVES,
                 "Radix of the dissemination algorithm used for thread group barriers")
NVSHMEMI_ENV_DEF(REDUCE_RECEXCH_KVAL, int, 2, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Radix of the recursive exchange reduction algorithm")
NVSHMEMI_ENV_DEF(FCOLLECT_LL_THRESHOLD, size, (size_t)(1 << 11), NVSHMEMI_ENV_CAT_COLLECTIVES,
                 "Message size threshold up to which "
                 "fcollect LL algo will be used\n")
NVSHMEMI_ENV_DEF(FCOLLECT_LL128_THRESHOLD, size, (size_t)(0), NVSHMEMI_ENV_CAT_HIDDEN,
                 "Message size threshold up to which "
                 "the fcollect LL128 algo will be used.\n"
                 "LL128 will be used only when FCOLLECT_LL_THRESHOLD < size")
NVSHMEMI_ENV_DEF(FCOLLECT_NVLS_THRESHOLD, size, (size_t)(1 << 24), NVSHMEMI_ENV_CAT_HIDDEN,
                 "Message size threshold up to which "
                 "fcollect NVLS algo will be used\n")
NVSHMEMI_ENV_DEF(REDUCESCATTER_NVLS_THRESHOLD, size, (size_t)(1 << 24), NVSHMEMI_ENV_CAT_HIDDEN,
                 "Message size threshold up to which "
                 "reducescatter NVLS algo will be used\n")

/* Size = 1 << 19 to manage 2*16B*8GPUs*2M bytes of scratchpad space for reductions */
NVSHMEMI_ENV_DEF(
    REDUCE_SCRATCH_SIZE, size, (size_t)(1 << 19), NVSHMEMI_ENV_CAT_COLLECTIVES,
    "Amount of symmetric heap memory (minimum 16B, multiple of 8B) reserved by runtime "
    "for every team to implement reduce and reducescatter collectives\n")

NVSHMEMI_ENV_DEF(BCAST_TREE_KVAL, int, 2, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Radix of the broadcast tree algorithm")
NVSHMEMI_ENV_DEF(BCAST_ALGO, int, 0, NVSHMEMI_ENV_CAT_COLLECTIVES,
                 "Broadcast algorithm to be used.\n"
                 "  * 0 - use default algorithm selection strategy\n")
NVSHMEMI_ENV_DEF(FCOLLECT_ALGO, int, 0, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Fcollect algorithm to be used. \n"
                 "  * 0 - use default algorithm selection strategy\n")

NVSHMEMI_ENV_DEF(REDUCE_ALGO, int, 0, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Allreduce algorithm to be used. \n"
                 "   * 0 - use default algorithm selection strategy\n")

NVSHMEMI_ENV_DEF(REDMAXLOC_ALGO, int, 1, NVSHMEMI_ENV_CAT_COLLECTIVES,
                 "Reduction algorithm to be used for MAXLOC operation.\n"
                 "  * 1 - default, flag alltoall algorithm\n"
                 "  * 2 - flat reduce + flat bcast\n"
                 "  * 3 - topo-aware two-level reduce + topo-aware bcast\n")
NVSHMEMI_ENV_DEF(REDUCESCATTER_ALGO, int, 0, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Reduce Scatter algorithm to be used. \n"
                 "  * 0 - use default algorithm selection strategy\n")

/** Transport **/

#ifdef NVSHMEM_DEFAULT_UCX
#define NVSHMEMI_ENV_TRANSPORT_DEFAULT "ucx"
#else
#define NVSHMEMI_ENV_TRANSPORT_DEFAULT "ibrc"
#endif

NVSHMEMI_ENV_DEF(ASSERT_ATOMICS_SYNC, bool, false, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Bypass flush on wait_until at target")
NVSHMEMI_ENV_DEF(REMOTE_TRANSPORT, string, NVSHMEMI_ENV_TRANSPORT_DEFAULT,
                 NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Selected transport for remote operations: ibrc, ucx, libfabric, ibdevx, none")
NVSHMEMI_ENV_DEF(BYPASS_FLUSH, bool, false, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Bypass flush in proxy when enforcing consistency")
NVSHMEMI_ENV_DEF(ENABLE_NIC_PE_MAPPING, bool, false, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "When not set or set to 0, a PE is assigned the NIC on the node that is "
                 "closest to it by distance. When set to 1, NVSHMEM either assigns NICs to "
                 "PEs on a round-robin basis or uses ``NVSHMEM_HCA_PE_MAPPING`` or "
                 "``NVSHMEM_HCA_LIST`` when they are specified.")
NVSHMEMI_ENV_DEF(DISABLE_LOCAL_ONLY_PROXY, bool, false, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "When running on an NVLink-only configuaration (No-IB, No-UCX), completely "
                 "disable the proxy thread. This will disable device side global exit and "
                 "device side wait timeout polling (enabled by ``NVSHMEM_TIMEOUT_DEVICE_POLLING`` "
                 "build-time variable) because these are processed by the proxy thread.")

/** Runtime optimimzations **/
NVSHMEMI_ENV_DEF(PROXY_REQUEST_BATCH_MAX, int, 32, NVSHMEMI_ENV_CAT_OTHER,
                 "Maxmum number of requests that the proxy thread processes in a single iteration "
                 "of the progress loop.")

/** NVTX instrumentation **/
NVSHMEMI_ENV_DEF(NVTX, string, "off", NVSHMEMI_ENV_CAT_NVTX,
                 "Set to enable NVTX instrumentation. Accepts a comma separated list of "
                 "instrumentation groups. By default the NVTX instrumentation is disabled.")

#if defined(NVSHMEM_IBGDA_SUPPORT) || defined(NVSHMEM_ENV_ALL)
/** GPU-initiated communication **/
NVSHMEMI_ENV_DEF(IB_ENABLE_IBGDA, bool, false, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Set to enable GPU-initiated communication transport.")
#endif

#endif
