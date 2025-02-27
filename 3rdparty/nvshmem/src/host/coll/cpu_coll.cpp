/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "cpu_coll.h"
#include <assert.h>                                      // for assert
#include <algorithm>                                     // for max
#include <iosfwd>                                        // for std
#include <stdlib.h>                                      // for std
#include "bootstrap_host_transport/env_defs_internal.h"  // for nvshmemi_opt...
#include "device_host/nvshmem_types.h"                   // for gpu_coll_env...
#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"
#include "internal/host/debug.h"             // for WARN
#include "internal/host/nvshmem_internal.h"  // for nvshmemi_is_...
#include "internal/host/util.h"              // for nvshmemi_opt...s
#include "non_abi/nvshmem_build_options.h"   // for NVSHMEM_USE_...
#include "non_abi/nvshmemx_error.h"          // for NVSHMEMI_WAR...

using namespace std;

#ifdef NVSHMEM_USE_NCCL
#include <dlfcn.h>  // for dlsym, dlopen, RTLD...
#include "nccl.h"   // for NCCL_VERSION_CODE
struct nccl_function_table nccl_ftable;
#endif /* NVSHMEM_USE_NCCL */

#define MIN_REDUCE_RECEXCH_VALUE 2

#define LOAD_SYM(handle, symbol, funcptr)  \
    do {                                   \
        void **cast = (void **)&funcptr;   \
        void *tmp = dlsym(handle, symbol); \
        *cast = tmp;                       \
    } while (0)

int nvshmemi_use_nccl = 0;
int nccl_version;

static int nvshmemi_coll_common_cpu_read_env() {
    int status = 0;
    nvshmemi_device_state.gpu_coll_env_params_var.barrier_tg_dissem_kval =
        nvshmemi_options.BARRIER_TG_DISSEM_KVAL;
    nvshmemi_device_state.gpu_coll_env_params_var.fcollect_ll_threshold =
        nvshmemi_options.FCOLLECT_LL_THRESHOLD;
    nvshmemi_device_state.gpu_coll_env_params_var.fcollect_ll128_threshold =
        nvshmemi_options.FCOLLECT_LL128_THRESHOLD;
    nvshmemi_device_state.gpu_coll_env_params_var.fcollect_nvls_threshold =
        nvshmemi_options.FCOLLECT_NVLS_THRESHOLD;
    nvshmemi_device_state.gpu_coll_env_params_var.reducescatter_nvls_threshold =
        nvshmemi_options.REDUCESCATTER_NVLS_THRESHOLD;
    nvshmemi_device_state.gpu_coll_env_params_var.reduce_scratch_size =
        nvshmemi_options.REDUCE_SCRATCH_SIZE;
    nvshmemi_device_state.gpu_coll_env_params_var.reduce_recexch_kval =
        nvshmemi_options.REDUCE_RECEXCH_KVAL;

    if (nvshmemi_device_state.gpu_coll_env_params_var.reduce_recexch_kval > nvshmemi_state->npes)
        nvshmemi_device_state.gpu_coll_env_params_var.reduce_recexch_kval =
            max(MIN_REDUCE_RECEXCH_VALUE, nvshmemi_state->npes);

    nvshmemi_device_state.gpu_coll_env_params_var.bcast_tree_kval =
        nvshmemi_options.BCAST_TREE_KVAL;
    assert(nvshmemi_options.BCAST_TREE_KVAL >= 2);

    nvshmemi_device_state.gpu_coll_env_params_var.fcollect_algo = nvshmemi_options.FCOLLECT_ALGO;
    nvshmemi_device_state.gpu_coll_env_params_var.bcast_algo = nvshmemi_options.BCAST_ALGO;
    nvshmemi_device_state.gpu_coll_env_params_var.reduce_algo = nvshmemi_options.REDUCE_ALGO;
    nvshmemi_device_state.gpu_coll_env_params_var.reduce_maxloc_algo =
        nvshmemi_options.REDMAXLOC_ALGO;
    nvshmemi_device_state.gpu_coll_env_params_var.reducescatter_algo =
        nvshmemi_options.REDUCESCATTER_ALGO;
    return status;
}

void nvshmemi_coll_common_cpu_check_ll128_availability() {
    bool *scratch = (bool *)malloc(sizeof(bool) * nvshmemi_state->npes);

    nvshmemi_boot_handle.allgather(&nvshmemi_state->are_nics_ll128_compliant, scratch, sizeof(bool),
                                   &nvshmemi_boot_handle);
    for (int i = 0; i < nvshmemi_state->npes; i++) {
        nvshmemi_state->are_nics_ll128_compliant &= scratch[i];
    }
    if (!nvshmemi_state->is_platform_nvl || !nvshmemi_state->are_nics_ll128_compliant) {
        INFO(NVSHMEM_INIT, "Disabling LL128 on unsupported platform. NVL OK: %d NIC OK: %d",
             nvshmemi_state->is_platform_nvl, nvshmemi_state->are_nics_ll128_compliant);
        nvshmemi_device_state.gpu_coll_env_params_var.fcollect_ll128_threshold = 0;
    }
    if (nvshmemi_device_state.symmetric_heap_kind == NVSHMEMI_HEAP_KIND_SYSMEM) {
        INFO(NVSHMEM_INIT, "Disabling LL128 due to system memory being used in heap.");
        nvshmemi_device_state.gpu_coll_env_params_var.fcollect_ll128_threshold = 0;
    }

    free(scratch);
}

int nvshmemi_coll_common_cpu_init() {
    int status = 0;
#ifdef NVSHMEM_USE_NCCL
    void *nccl_handle = NULL;
    int nccl_build_version;
    int nccl_major;
    int nccl_build_major;
#endif

    status = nvshmemi_coll_common_cpu_read_env();
    if (status) NVSHMEMI_COLL_CPU_ERR_POP();

#ifdef NVSHMEM_USE_NCCL
    nvshmemi_use_nccl = 1;
    assert(NCCL_VERSION_CODE >= 2000);
    if (nvshmemi_options.DISABLE_NCCL) {
        nvshmemi_use_nccl = 0;
        goto fn_out;
    }

    if (nvshmemi_is_mpg_run) {
        WARN(
            "NVSHMEM has detected multiple PEs per GPU which is not supported "
            "by NCCL and is disabling NCCL accordingly. To silence this warning, "
            "set the NVSHMEM_DISABLE_NCCL=1 variable to explicitly disable NCCL.");
        nvshmemi_use_nccl = 0;
        goto fn_out;
    }

    nccl_handle = dlopen("libnccl.so.2", RTLD_LAZY);
    if (!nccl_handle) {
        NVSHMEMI_WARN_PRINT("NCCL library not found...\n");
        nvshmemi_use_nccl = 0;
        goto fn_out;
    }

    nccl_build_version = NCCL_VERSION_CODE;
    LOAD_SYM(nccl_handle, "ncclGetVersion", nccl_ftable.GetVersion);
    nccl_ftable.GetVersion(&nccl_version);
    if (nccl_version > 10000) {
        nccl_major = nccl_version / 10000;
    } else {
        nccl_major = nccl_version / 1000;
    }
    if (nccl_build_version > 10000) {
        nccl_build_major = nccl_build_version / 10000;
    } else {
        nccl_build_major = nccl_build_version / 1000;
    }
    if (nccl_major != nccl_build_major) {
        NVSHMEMI_WARN_PRINT(
            "NCCL library major version (%d) is different than the"
            " version (%d) with which NVSHMEM was built, skipping use...\n",
            nccl_major, nccl_build_major);
        nvshmemi_use_nccl = 0;
        goto fn_out;
    }
    LOAD_SYM(nccl_handle, "ncclGetUniqueId", nccl_ftable.GetUniqueId);
    LOAD_SYM(nccl_handle, "ncclCommInitRank", nccl_ftable.CommInitRank);
    LOAD_SYM(nccl_handle, "ncclCommDestroy", nccl_ftable.CommDestroy);
    LOAD_SYM(nccl_handle, "ncclAllReduce", nccl_ftable.AllReduce);
    LOAD_SYM(nccl_handle, "ncclReduceScatter", nccl_ftable.ReduceScatter);
    LOAD_SYM(nccl_handle, "ncclBroadcast", nccl_ftable.Broadcast);
    LOAD_SYM(nccl_handle, "ncclAllGather", nccl_ftable.AllGather);
    LOAD_SYM(nccl_handle, "ncclGetErrorString", nccl_ftable.GetErrorString);
    LOAD_SYM(nccl_handle, "ncclGroupStart", nccl_ftable.GroupStart);
    LOAD_SYM(nccl_handle, "ncclGroupEnd", nccl_ftable.GroupEnd);
    if (nccl_version >= 2700) {
        LOAD_SYM(nccl_handle, "ncclSend", nccl_ftable.Send);
        LOAD_SYM(nccl_handle, "ncclRecv", nccl_ftable.Recv);
    }

fn_out:
#endif /* NVSHMEM_USE_NCCL */
    return status;
fn_fail:
    return status;
}

int nvshmemi_coll_common_cpu_finalize() {
    int status = 0;

    return status;
}
