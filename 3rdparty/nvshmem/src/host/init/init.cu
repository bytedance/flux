/*
 * Copyright (c) 2016-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/types.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <set>
#include <list>

#include "host/nvshmemx_api.h"
#include "internal/host/nvshmemi_team.h"
#include "internal/host/nvshmem_internal.h"
#include "internal/host/nvshmem_nvtx.hpp"
#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"
#include "internal/host/nvshmemi_bootstrap_library.h"
#include "non_abi/nvshmem_build_options.h"

#ifdef NVSHMEM_IBGDA_SUPPORT
#include "device_host_transport/nvshmem_common_ibgda.h"
#endif

#include <stdlib.h>
#include <string.h>
#include "topo.h"
#include "internal/host/util.h"
#include "cpu_coll.h"
#include "unistd.h"
#include "internal/host/debug.h"
#include "internal/host/nvshmemi_types.h"
#include "internal/host/shared_memory.h"
#include "internal/host/nvshmemi_symmetric_heap.hpp"

using namespace std;

extern __constant__ nvshmemi_device_host_state_t nvshmemi_device_state_d;
static std::map<void *, int> registered_device_states;
static std::set<nvshmemx_device_lib_init_cb> registered_device_state_cb;

static void nvshmemi_init_debug(void);
static void nvshmemi_init_msg(void);

struct nvshmemi_cuda_fn_table *nvshmemi_cuda_syms;
nvshmemi_state_t *nvshmemi_state;
bootstrap_handle_t nvshmemi_boot_handle;
nvshmemi_session_t *nvshmemi_default_session = nullptr;
nvshmemi_pe_dist_t nvshmemi_pe_dist;
uint64_t *nvshmemi_host_hashes;
int nvshmemi_init_counter = 0;
nvshmem_options_t nvshmem_options;
int nvshmemi_cuda_driver_version;
const char *p_err_str;
int nvshmem_debug_level;
uint64_t nvshmem_debug_mask = NVSHMEM_INIT;  // Default debug sub-system mask is INIT
pthread_mutex_t nvshmem_debug_output_lock;
bool nvshmemi_is_limited_mpg_run = 0;
static uint64_t num_initialized_device_states = 0;
static bool nvshmemi_is_device_state_ready;
int nvshmemi_can_use_cuda_64_bit_stream_memops = false;
int nvshmemi_can_flush_remote_writes = false;
FILE *nvshmem_debug_file = stdout;
static char shm_name[100];
nvshmemi_version_t nvshmemi_host_lib_version = {
    NVSHMEM_INTERLIB_MAJOR_VERSION, NVSHMEM_INTERLIB_MINOR_VERSION, NVSHMEM_INTERLIB_PATCH_VERSION};

#ifdef NVSHMEM_TRACE
std::chrono::high_resolution_clock::time_point nvshmem_epoch;
#endif

void *heap_base_array_dptr = NULL;
void *heap_base_actual_array_dptr = NULL;
int nvshmemi_job_connectivity;

nvshmemi_device_host_state_t nvshmemi_device_state;

void nvshmemi_get_device_state(void **state) { *state = &nvshmemi_device_state; }

#ifdef NVSHMEM_IBGDA_SUPPORT
nvshmemi_ibgda_device_state_t nvshmemi_ibgda_device_state;
static std::map<void *, int> registered_transport_device_states;
void nvshmemi_ibgda_get_device_state(void **state) { *state = &nvshmemi_ibgda_device_state; }
#endif

static inline bool nvshmemi_is_version_compatible(const nvshmemi_version_t version_host,
                                                  const nvshmemi_version_t version_device) {
    if (version_host.major != version_device.major) {
        return 1;
    }
    /* Device is newer, can't interoperate with the older host.
     * For example - a newer device may have added a type or function,
     * when it links to the host, the function will not be found.
     */
    if (version_device.minor > version_host.minor) {
        return 1;
    }

    return 0;
}

static int register_state_ptr(void *common, void *transport) {
    if (registered_device_states.find(common) != registered_device_states.end()) {
        auto it = registered_device_states.find(common);
        it->second++;
    } else {
        registered_device_states.emplace(common, 1);
    }

#ifdef NVSHMEM_IBGDA_SUPPORT
    if (transport != NULL) {
        if (registered_transport_device_states.find(transport) !=
            registered_transport_device_states.end()) {
            auto it = registered_transport_device_states.find(transport);
            it->second++;
        } else {
            registered_transport_device_states.emplace(transport, 1);
        }
#else
    if (transport != NULL) {
        NVSHMEMI_ERROR_PRINT(
            "IBGDA not enabled by host lib, but passed "
            "in by device. Host ignoring IBGDA state.\n");
        return 0;
    }
#endif
#ifdef NVSHMEM_IBGDA_SUPPORT
    }
#endif
    return 0;
}

int nvshmemi_update_device_state() {
    int status = 0;
    int iter;
    nvshmemx_device_lib_init_cb cb;
    void *device_ptr;
    void *transport_device_ptr = NULL;
    while (!registered_device_state_cb.empty()) {
        cb = *(registered_device_state_cb.begin());
        registered_device_state_cb.erase(cb);
        cb(&device_ptr, &transport_device_ptr);

        if (device_ptr == NULL) {
            NVSHMEMI_ERROR_PRINT("Bad device pointer callback registered %p. Skipping\n", cb);
            continue;
        }

        status = register_state_ptr(device_ptr, transport_device_ptr);

        nvshmemi_init_counter++;
        device_ptr = NULL;
        transport_device_ptr = NULL;
    }

    if (!nvshmemi_is_device_state_ready ||
        (num_initialized_device_states < registered_device_states.size())) {
        iter = 0;
        for (auto it = registered_device_states.cbegin(); it != registered_device_states.cend();
             ++it) {
            iter++;
            nvshmemi_device_host_state_t *device_state;
            nvshmemi_get_device_state((void **)&device_state);
            status = cudaMemcpy((it->first), (void *)device_state,
                                sizeof(nvshmemi_device_host_state_t), cudaMemcpyHostToDevice);
            if (status) break;
        }
        num_initialized_device_states = iter;
    }

#ifdef NVSHMEM_IBGDA_SUPPORT
    if (nvshmemi_options.IB_ENABLE_IBGDA) {
        nvshmemi_ibgda_device_state_t *ibgda_device_state;
        nvshmemi_ibgda_get_device_state((void **)&ibgda_device_state);
        for (auto it = registered_transport_device_states.cbegin();
             it != registered_transport_device_states.cend(); ++it) {
            status = cudaMemcpy((it->first), (void *)ibgda_device_state,
                                sizeof(nvshmemi_ibgda_device_state_t), cudaMemcpyHostToDevice);
            if (status) break;
        }
    }
#endif
    return status;
}

static int unregister_state_ptr(void *common, void *transport) {
    nvshmemi_update_device_state();

    bool device_state_found = false;
    for (auto it = registered_device_states.cbegin(); it != registered_device_states.cend();) {
        auto tmp = registered_device_states.find(it->first);
        if (it->first == common) {
            if (tmp->second > 1) {
                tmp->second--;
            } else {
                it = registered_device_states.erase(it);
                num_initialized_device_states--;
            }
            device_state_found = true;
            break;
        } else {
            ++it;
        }
    }

#ifdef NVSHMEM_IBGDA_SUPPORT
    bool transport_state_found = false;
    if (transport != NULL) {
        for (auto it = registered_transport_device_states.cbegin();
             it != registered_transport_device_states.cend();) {
            auto tmp = registered_transport_device_states.find(it->first);
            if (tmp->first == transport) {
                if (tmp->second > 1) {
                    tmp->second--;
                } else {
                    it = registered_transport_device_states.erase(it);
                }
                transport_state_found = true;
                break;
            } else {
                ++it;
            }
        }
        if (!transport_state_found && device_state_found) {
            NVSHMEMI_ERROR_PRINT(
                "Invalid IBGDA handle, but valid device state passed for "
                "removal. This is not a fatal error, but indicates something "
                "unexpected is happening. Standard device state removed.\n");
        }
    }
#endif
    if (device_state_found) {
        return NVSHMEMX_SUCCESS;
    }

    return NVSHMEMX_ERROR_INVALID_VALUE;
}

static int nvshmemi_transport_cap_support_rma(int cap) {
    if (cap & (NVSHMEM_TRANSPORT_CAP_CPU_READ | NVSHMEM_TRANSPORT_CAP_CPU_WRITE |
               NVSHMEM_TRANSPORT_CAP_MAP_GPU_LD | NVSHMEM_TRANSPORT_CAP_MAP_GPU_ST)) {
        return 1;
    }
    return 0;
}

static int nvshmemi_transport_cap_support_amo(int cap) {
    if (cap & (NVSHMEM_TRANSPORT_CAP_CPU_ATOMICS | NVSHMEM_TRANSPORT_CAP_MAP_GPU_ATOMICS)) {
        return 1;
    }
    return 0;
}

int nvshmemx_get_uniqueid(nvshmemx_uniqueid_t *uid) {
    int status = 0;
    nvshmemi_options_init();
    status = bootstrap_preinit(NVSHMEMX_INIT_WITH_UNIQUEID, &nvshmemi_boot_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "bootstrap pre-initialization failed \n");

    if (nvshmemi_boot_handle.pre_init_ops) {
        status = nvshmemi_boot_handle.pre_init_ops->get_unique_id((void *)uid);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "bootstrap get unique ID failed \n");
    } else {
        // Throw an error as network discovery failed or was not performed correctly during preinit
        status = NVSHMEMX_ERROR_INTERNAL;
        NVSHMEMI_NZ_ERROR_JMP(
            status, NVSHMEMI_INVALID_USAGE, out,
            "Bootstrap lacks support for pre init step, so unable to fetch unique ID");
    }

out:
    return (status);
}

// Marshal and unmarshall UID attribute arguments
int nvshmemx_set_attr_uniqueid_args(const int myrank, const int nranks,
                                    const nvshmemx_uniqueid_t *uid,
                                    nvshmemx_init_attr_t *nvshmem_attr) {
    assert(nvshmem_attr != NULL);
    nvshmemx_init_args_t *init_args = (nvshmemx_init_args_t *)(&(nvshmem_attr->args));
    nvshmemx_uniqueid_args_t *uid_args = &((init_args)->uid_args);

    /* Save to uid_args */
    uid_args->id = const_cast<nvshmemx_uniqueid_t *>(uid);  // Don't deepcopy as we are saving a ptr
    uid_args->myrank = myrank;
    uid_args->nranks = nranks;
    return (0);
}

nvshmemx_uniqueid_args_t *nvshmemi_get_attr_uniqueid_args(nvshmemx_init_attr_t *attr) {
    assert(attr != NULL);
    nvshmemx_init_args_t *init_args = (nvshmemx_init_args_t *)(&(attr->args));
    return (&(init_args->uid_args));
}

int nvshmemi_bootstrap_preinit(int flags) {
    int status = 1;
    status = bootstrap_preinit(flags, &nvshmemi_boot_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "bootstrap_preinit failed \n");
out:
    return (status);
}

int nvshmemi_bootstrap(int flags, nvshmemx_init_attr_t *nvshmem_attr) {
    int status = 0;
    uint64_t myHostHash = 0;
    uint64_t *hostHash = NULL;
    int mype, npes;
    int mype_node = 0, npes_node = 0;
    int num_nodes;

    bootstrap_attr_t attr = {};
    status = bootstrap_set_bootattr(flags, nvshmem_attr, (nvshmem_attr) ? &attr : NULL);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "bootstrap_set_bootattr failed \n");
    status = bootstrap_init(flags, &attr, &nvshmemi_boot_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "bootstrap_init failed \n");
    if (nvshmemi_default_session == nullptr) {
        nvshmemi_default_session = (nvshmemi_session_t *)calloc(sizeof(nvshmemi_session_t), 1);
        NVSHMEMI_NULL_ERROR_JMP(nvshmemi_default_session, status, NVSHMEMX_ERROR_INTERNAL, out,
                                "Unable to allocate memory for nvshmem session");
        nvshmemi_default_session->bootstrap = &nvshmemi_boot_handle;
    }

    mype = nvshmemi_boot_handle.pg_rank;
    npes = nvshmemi_boot_handle.pg_size;
    myHostHash = getHostHash();
    hostHash = (uint64_t *)malloc(sizeof(uint64_t) * npes);
    status = nvshmemi_boot_handle.allgather((void *)&myHostHash, (void *)hostHash, sizeof(uint64_t),
                                            &nvshmemi_boot_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "allgather of host hashes failed \n");
    nvshmemi_host_hashes = (uint64_t *)malloc(sizeof(uint64_t) * npes);
    memcpy(nvshmemi_host_hashes, hostHash, sizeof(uint64_t) * npes);
    for (int i = 0; i < npes; i++) {
        if (nvshmemi_host_hashes[i] == myHostHash) {
            if (i == mype) mype_node = npes_node;
            npes_node++;
        }
    }
    nvshmemi_boot_handle.mype_node = mype_node;
    nvshmemi_boot_handle.npes_node = npes_node;

    // Check for same number of PEs on every node
    // Use myHostHash value to indicate a given PE has been counted already
    // This overwrites the hostHash values, but we don't them past this point

    for (int i = 0; i < npes; i++) {
        // If this host's node hasn't been counted yet, count it
        if (hostHash[i] != myHostHash) {
            const uint64_t peer_hash = hostHash[i];
            int npes_peer_node = 0;
            for (int j = i; j < npes; j++) {
                if (peer_hash == hostHash[j]) {
                    npes_peer_node++;
                    hostHash[j] = myHostHash;
                }
            }
            if (npes_peer_node != npes_node) {
                char hostname[1024];
                nvshmemu_gethostname(hostname, 1024);

                NVSHMEMI_ERROR_JMP(
                    status, NVSHMEMX_ERROR_INTERNAL, out,
                    "NVSHMEM requires the same number of PEs on all nodes (%d PEs on %s)\n",
                    npes_node, hostname);
            }
        }
    }

    nvshmem_nvtx_set_thread_name(mype);

    /* Set pe distribution type. First check for round robin distribution. Then check for block
     * distribution. */
    nvshmemi_pe_dist = NVSHMEMI_PE_DIST_MISC;

    if (npes_node != 0) {
        if (npes % npes_node != 0) goto out;
        num_nodes = npes / npes_node;
    } else {
        NVSHMEMI_ERROR_JMP(
            status, NVSHMEMX_ERROR_INTERNAL, out,
            "NVSHMEM hit the error of division by zero: npes_node == 0 what happen!?\n");
    }

    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < npes_node; j++) {
            if (nvshmemi_host_hashes[i * npes_node] != nvshmemi_host_hashes[i * num_nodes + j])
                goto check_roundrobin_dist;
        }
    }
    nvshmemi_pe_dist = NVSHMEMI_PE_DIST_BLOCK;
    INFO(NVSHMEM_INIT, "PE distribution has been identified as NVSHMEMI_PE_DIST_BLOCK");
    goto out;

check_roundrobin_dist:
    for (int i = 0; i < npes_node; i++) {
        for (int j = 0; j < num_nodes; j++) {
            if (nvshmemi_host_hashes[j * npes_node] != nvshmemi_host_hashes[i * num_nodes + j])
                goto out;
        }
    }
    nvshmemi_pe_dist = NVSHMEMI_PE_DIST_ROUNDROBIN;
    INFO(NVSHMEM_INIT, "PE distribution has been identified as NVSHMEMI_PE_DIST_ROUNDROBIN");

out:
    nvshmemi_device_state.pe_dist = nvshmemi_pe_dist;
    if (hostHash) free(hostHash);
    return status;
}

int nvshmemi_init_nvshmemi_state(nvshmemi_state_t *state) {
    int status = 0;
    state->mype = nvshmemi_boot_handle.pg_rank;
    state->npes = nvshmemi_boot_handle.pg_size;
    state->mype_node = nvshmemi_boot_handle.mype_node;
    state->npes_node = nvshmemi_boot_handle.npes_node;
    state->is_platform_nvl = true;
    state->are_nics_ll128_compliant = true;

    return status;
}

static void nvshmemi_detect_nvls_support(nvshmemi_state_t *state) {
    int status = NVSHMEMX_ERROR_INTERNAL;
    int mc_support = 0;
    int cuda_dev;
    state->is_platform_nvls = false; /* By default assume it is not supported */
    CUdevice current_dev;
    CUDA_RUNTIME_CHECK(cudaGetDevice(&cuda_dev));
    status = CUPFN(nvshmemi_cuda_syms, cuDeviceGet(&current_dev, cuda_dev));
    if (status != CUDA_SUCCESS) {
        WARN("cuDeviceGet failed\n");
        return;
    }

    status = CUPFN(
        nvshmemi_cuda_syms,
        cuDeviceGetAttribute(
            &mc_support, static_cast<CUdevice_attribute>(CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED),
            current_dev));
    if (status != CUDA_SUCCESS) {
        WARN("cuDeviceGetAttribute failed\n");
        return;
    }

    if (!mc_support || nvshmemi_options.DISABLE_NVLS) {
        INFO(NVSHMEM_INIT, "cuMulticast is not supported on CUDA or disabled by user\n");
        return;
    }

    if (state->heap_obj != nullptr &&
        dynamic_cast<nvshmemi_symmetric_heap_vidmem_dynamic_vmm *>(state->heap_obj) == nullptr) {
        WARN("Unsupported heap kind for NVLS. Supported are: cuMemCreate\n");
        return;
    }

    /* On newer platform, when using an older CUDA driver at runtime, check if APIs are available */
    if (nvshmemi_cuda_driver_version >= 12010) {
        CUASSERTAPIAVAILABLE(nvshmemi_cuda_syms, cuMulticastCreate);
        CUASSERTAPIAVAILABLE(nvshmemi_cuda_syms, cuMulticastBindMem);
        CUASSERTAPIAVAILABLE(nvshmemi_cuda_syms, cuMulticastUnbind);
        CUASSERTAPIAVAILABLE(nvshmemi_cuda_syms, cuMulticastGetGranularity);
        CUASSERTAPIAVAILABLE(nvshmemi_cuda_syms, cuMulticastAddDevice);
        state->is_platform_nvls = true;
    }

    return;
}

int nvshmemi_get_cucontext(nvshmemi_state_t *state) {
    CUdevice cudevice;
    int leastPriority, greatestPriority;
    int status = NVSHMEMX_SUCCESS;

    CUCHECK(nvshmemi_cuda_syms, cuInit(0));

    status = CUPFN(nvshmemi_cuda_syms, cuCtxGetDevice(&cudevice));
    if (status || nvshmemi_options.BOOTSTRAP_TWO_STAGE) {
        if (nvshmemi_options.BOOTSTRAP_TWO_STAGE) {
            TRACE(NVSHMEM_INIT, "Two-stage initialization requested");
            nvshmemi_options.BOOTSTRAP_TWO_STAGE = false;
        } else
            TRACE(NVSHMEM_INIT, "GPU not selected, cuCtxGetDevice failed, err: %d", status);

        status = NVSHMEMX_ERROR_GPU_NOT_SELECTED;
        goto out;
    } else {
        CUresult cres = CUPFN(nvshmemi_cuda_syms, cuCtxSynchronize());
        if (cres) {
            CUCHECK(nvshmemi_cuda_syms, cuDevicePrimaryCtxRetain(&state->cucontext, cudevice));
            CUCHECK(nvshmemi_cuda_syms, cuCtxSetCurrent(state->cucontext));
            INFO(NVSHMEM_INIT, "retained primary context for device: %d", cudevice);
        } else {
            INFO(NVSHMEM_INIT,
                 "[%d] nvshmemi_get_cucontext->cuCtxSynchronize->CUDA_SUCCESS) my_stream %p",
                 state->mype, state->my_stream);
            CUCHECK(nvshmemi_cuda_syms, cuCtxGetCurrent(&state->cucontext));
            INFO(NVSHMEM_INIT,
                 "in get_cucontext, queried and saved context for device: %d context: %p", cudevice,
                 state->cucontext);
        }

        if (nvshmemi_cuda_driver_version >= 12010) {
            unsigned int flags;
            CUASSERTAPIAVAILABLE(nvshmemi_cuda_syms, cuCtxSetFlags);
            CUCHECK(nvshmemi_cuda_syms, cuCtxGetFlags(&flags));
            CUCHECK(nvshmemi_cuda_syms, cuCtxSetFlags(flags | CU_CTX_SYNC_MEMOPS));
        }
        status = NVSHMEMX_SUCCESS;

        // identify device id
        int count;
        CUdevice curr_device;
        CUDA_RUNTIME_CHECK(cudaGetDeviceCount(&count));

        for (int i = 0; i < count; i++) {
            CUCHECK(nvshmemi_cuda_syms, cuDeviceGet(&curr_device, i));
            if (curr_device == cudevice) {
                state->device_id = i;
                break;
            }
        }
        CUDA_RUNTIME_CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
        CUDA_RUNTIME_CHECK(cudaStreamCreateWithPriority(&state->my_stream, cudaStreamNonBlocking,
                                                        greatestPriority));
        INFO(NVSHMEM_INIT, "[%d] Created stream %p for device %d", state->mype, state->my_stream,
             state->device_id);
    }
out:
    return status;
}

int nvshmemi_setup_stream_priorities(nvshmemi_state_t *state) {
    int status = 0;
    int leastPriority, greatestPriority;

    CUDA_RUNTIME_CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    CUDA_RUNTIME_CHECK(
        cudaStreamCreateWithPriority(&state->my_stream, cudaStreamNonBlocking, greatestPriority));

    return status;
}

int nvshmemi_teardown_handles(nvshmemi_state_t *state) {
    INFO(NVSHMEM_INIT, "In nvshmemi_teardown_handles");
    int status = 0;
    free(state->selected_transport_for_rma);
    free(state->selected_transport_for_amo);
    for (int i = 0; i < MAX_PEER_STREAMS; i++) {
        CUDA_RUNTIME_CHECK_GOTO(cudaStreamDestroy(state->custreams[i]), status, out);
        CUDA_RUNTIME_CHECK_GOTO(cudaEventDestroy(state->cuevents[i]), status, out);
    }
out:
    return status;
}

static int nvshmemi_setup_nvshmem_handles(nvshmemi_state_t *state) {
    int status = 0;
    int dev_attr = 0;
    /* TODO: We should really check all of these allocations. */;
    state->selected_transport_for_rma = (int *)calloc(state->npes, sizeof(int));
    state->selected_transport_for_amo = (int *)calloc(state->npes, sizeof(int));
    CUDA_RUNTIME_CHECK(cudaDeviceGetAttribute(
        &dev_attr, cudaDevAttrCanUseHostPointerForRegisteredMem, state->device_id));
    state->host_memory_registration_supported =
        dev_attr & cudaDevAttrCanUseHostPointerForRegisteredMem;

    for (int pe = 0; pe < state->npes; pe++) {
        state->selected_transport_for_rma[pe] = -1;
        state->selected_transport_for_amo[pe] = -1;
    }
    int tbitmap;
    for (int i = 0; i < state->npes; i++) {
        bool amo_initialized = false, rma_initialized = false;
        tbitmap = state->transport_bitmap;
        for (int j = 0; j < state->num_initialized_transports; j++) {
            if (!(state->transports[j])) {
                tbitmap >>= 1;
                continue;
            }

            if (tbitmap & 1) {
                if (!rma_initialized &&
                    nvshmemi_transport_cap_support_rma(nvshmemi_state->transports[j]->cap[i])) {
                    rma_initialized = true;
                    state->selected_transport_for_rma[i] = j;
                }

                if (!amo_initialized &&
                    nvshmemi_transport_cap_support_amo(nvshmemi_state->transports[j]->cap[i])) {
                    amo_initialized = true;
                    state->selected_transport_for_amo[i] = j;
                }
            }
            tbitmap >>= 1;
        }
    }

    return status;
}

static int nvshmemi_setup_cuda_handles(nvshmemi_state_t *state) {
    int status = 0;
    state->custreams = (cudaStream_t *)malloc(MAX_PEER_STREAMS * sizeof(cudaStream_t));
    state->cuevents = (cudaEvent_t *)malloc(MAX_PEER_STREAMS * sizeof(cudaEvent_t));
    state->active_internal_streams = (bool *)calloc(MAX_PEER_STREAMS, sizeof(bool));
    int leastPriority, greatestPriority;
    CUDA_RUNTIME_CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    for (int i = 0; i < MAX_PEER_STREAMS; i++) {
        CUDA_RUNTIME_CHECK_GOTO(cudaStreamCreateWithPriority(
                                    &state->custreams[i], cudaStreamNonBlocking, greatestPriority),
                                status, out);
        CUDA_RUNTIME_CHECK_GOTO(
            cudaEventCreateWithFlags(&state->cuevents[i], cudaEventDisableTiming), status, out);
    }
out:
    return status;
}

static bool trimNewline(char *str) {
    size_t len = strlen(str);
    if (len > 0 && str[len - 1] == '\n') {
        str[len - 1] = '\0';
    }
    return strlen(str) > 0;
}

static bool mpsServerRunning(int *serverPID) {
    const int lineSize = 256;
    char line[lineSize];
    int ret;
    bool status = false;
    bool serverExist = false;

    FILE *output = popen("echo get_server_list | nvidia-cuda-mps-control 2> /dev/null", "r");
    if (!output) {
        INFO(NVSHMEM_INIT, "popen retuned NULL");
        return false;
    }

    while (fgets(line, lineSize, output) != NULL) {
        serverExist = true;
    }

    ret = pclose(output);
    status = (ret != -1) && WIFEXITED(ret) && (WEXITSTATUS(ret) == 0);
    if (!status) {
        INFO(NVSHMEM_INIT, "pclose retuned error");
        return false;
    }

    if (!serverExist) {
        return false;
    }

    if (!trimNewline(line)) {
        return false;
    }

    if (serverPID) {
        stringstream ss(line);
        int result;
        ss >> result;
        *serverPID = result;
    }

    return true;
}

static bool get_mps_server_active_thread_percentage(float *percentage) {
    FILE *output;
    const int lineSize = 256;
    char line[lineSize];
    int ret;
    char *retstr = NULL;
    bool status = false;
    stringstream cmd;
    int serverPID;
    /* one PE per node queries the control daemon */
    if (nvshmemi_state->mype == nvshmemi_team_node.start) {
        if (!mpsServerRunning(&serverPID)) {
            return false;
        }

        cmd << "echo get_active_thread_percentage " << serverPID << " | nvidia-cuda-mps-control";
        output = popen(cmd.str().c_str(), "r");

        if (!output) {
            return false;
        }

        retstr = fgets(line, lineSize, output);

        ret = pclose(output);
        status = (ret != -1) && WIFEXITED(ret) && (WEXITSTATUS(ret) == 0);
        if (!status || retstr == NULL) {
            return false;
        }

        if (!trimNewline(line)) {
            return false;
        }

        if (percentage) {
            int result;
            stringstream ss(line);
            ss >> result;
            *percentage = result;
        }
    }
    float *scratch = (float *)malloc(sizeof(float) * nvshmemi_state->npes);
    /* for lack of a better available bootstrap collective, using allagther */
    status = nvshmemi_boot_handle.allgather((void *)percentage, (void *)scratch, sizeof(float),
                                            &nvshmemi_boot_handle);
    *percentage = scratch[nvshmemi_team_node.start];
    free(scratch);

    return true;
}

static int nvshmemi_determine_mpg_support_level() {
    int status = 0;
    bool is_mps_server_running = false;
    if (nvshmemi_state->mype == nvshmemi_team_node.start) {
        is_mps_server_running = mpsServerRunning(NULL);
    }
    bool *scratch = (bool *)malloc(sizeof(bool) * nvshmemi_state->npes);
    /* for lack of a better available bootstrap collective, using allagther */
    status = nvshmemi_boot_handle.allgather((void *)&is_mps_server_running, (void *)scratch,
                                            sizeof(bool), &nvshmemi_boot_handle);
    is_mps_server_running = scratch[nvshmemi_team_node.start];
    free(scratch);

    if (!is_mps_server_running) {
        INFO(NVSHMEM_INIT,
             "Multiple PEs per GPU (MPG) detected but MPS is not running. "
             "Hence limited MPG support is available");
        nvshmemi_is_limited_mpg_run = 1;
    } else {
        float active_thread_percentage = 0;
        bool success = get_mps_server_active_thread_percentage(&active_thread_percentage);
        if (!success) {
            INFO(NVSHMEM_INIT, "failed in get_mps_server_active_thread_percentage");
            exit(-1);
        }
        char *env = getenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE");
        if (env) active_thread_percentage = atof(env);

        float *active_percentages = (float *)malloc(sizeof(float) * nvshmemi_state->npes);
        status = nvshmemi_boot_handle.allgather((void *)&active_thread_percentage,
                                                (void *)active_percentages, sizeof(float),
                                                &nvshmemi_boot_handle);
        float total_percentage = 0;
        for (int i = 0; i < nvshmemi_team_same_gpu.size; i += 1) {
            total_percentage += *((float *)active_percentages + nvshmemi_team_same_gpu.start +
                                  i * nvshmemi_team_same_gpu.stride);
        }
        if (total_percentage <= 100.0 ||
            nvshmemi_options.IGNORE_CUDA_MPS_ACTIVE_THREAD_PERCENTAGE) {
            nvshmemi_is_limited_mpg_run = 0;
            INFO(NVSHMEM_INIT,
                 "Multiple PEs per GPU (MPG) detected, MPS is also available, "
                 "and either active thread percentages for PEs on the same GPU add "
                 "up to be <= 100 or user has requested to ignore active thread percentage. "
                 "Hence full MPG support is available. If active thread percentage "
                 "adds to be more than 100, NVSHMEM synchronizing APIs might deadlock.");
        } else {
            nvshmemi_is_limited_mpg_run = 1;
            INFO(NVSHMEM_INIT,
                 "Multiple PEs per PU (MPG) detected, MPS is also available, "
                 "but active thread percentages for PEs on the same GPU add "
                 "up to be greater than 100. Hence limited MPG support is available");
        }
        free(active_percentages);
    }
    return status;
}

static int nvshmemi_setup_limited_mpg_support() {
    int status = 0;
    nvshmemi_mps_shmdata *shm = NULL;
    nvshmemi_shared_memory_info_t *info = &nvshmemi_state->shm_info;
    cudaEvent_t event;
    int counter = 0;

    /* Ensure supported MPS runs */
    /* Do reduction to check to that each GPU has same stride and size for team_same_gpu */
    int ret = snprintf(shm_name, 100, "mps_shm_%d", nvshmemi_team_same_gpu.start);
    if (ret < 0) {
        INFO(NVSHMEM_INIT, "snprintf failed");
        return ret;
    }

    if (nvshmemi_team_same_gpu.start == nvshmemi_state->mype) {
        if (shared_memory_create(shm_name, sizeof(nvshmemi_mps_shmdata), info) != 0) {
            NVSHMEMI_ERROR_EXIT("Failed to create shared memory slab\n");
        }
    }
    status = nvshmemi_boot_handle.barrier(&nvshmemi_boot_handle);
    if (nvshmemi_team_same_gpu.start != nvshmemi_state->mype) {
        if (shared_memory_open(shm_name, sizeof(nvshmemi_mps_shmdata), info) != 0) {
            NVSHMEMI_ERROR_EXIT("Failed to open shared memory slab\n");
        }
    }

    shm = (nvshmemi_mps_shmdata *)info->addr;
    if (nvshmemi_team_same_gpu.start == nvshmemi_state->mype) {
        shm->nprocesses = nvshmemi_team_same_gpu.size;
        shm->barrier = 0;
        shm->sense = 0;
    }
    CUDA_RUNTIME_CHECK(cudaEventCreate(&nvshmemi_state->mps_event,
                                       cudaEventDisableTiming | cudaEventInterprocess));
    CUDA_RUNTIME_CHECK(cudaIpcGetEventHandle(
        (cudaIpcEventHandle_t *)&shm->event_handle[nvshmemi_team_same_gpu.my_pe],
        nvshmemi_state->mps_event));

    std::atomic_thread_fence(std::memory_order_seq_cst);  // flush the data
    status = nvshmemi_boot_handle.barrier(&nvshmemi_boot_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "bootstrap barrier failed \n");

    for (int i = 0; i < nvshmemi_team_same_gpu.size; i++) {
        if (i == nvshmemi_team_same_gpu.my_pe) continue;
        CUDA_RUNTIME_CHECK(
            cudaIpcOpenEventHandle(&event, *(cudaIpcEventHandle_t *)&shm->event_handle[i]));
        nvshmemi_state->same_gpu_other_pe_mps_events[counter++] = event;
    }

out:
    return status;
}

static int nvshmemi_mpg_finalize() {
    shared_memory_close(shm_name, &nvshmemi_state->shm_info);
    CUDA_RUNTIME_CHECK(cudaEventDestroy(nvshmemi_state->mps_event));
    nvshmemi_is_mpg_run = false;
    return 0;
}

static void nvshmemi_query_cuda_attributes() {
    int status = 0;
    CUdevice device;
    status = CUPFN(nvshmemi_cuda_syms, cuCtxGetDevice)(&device);
    if (status != CUDA_SUCCESS) {
        fprintf(stderr, "cuCtxGetDevice failed \n");
        exit(-1);
    }

    status = CUPFN(nvshmemi_cuda_syms, cuDeviceGetAttribute)(
        &nvshmemi_can_use_cuda_64_bit_stream_memops,
        (CUdevice_attribute)CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS, device);
    if (status != CUDA_SUCCESS) {
        nvshmemi_can_use_cuda_64_bit_stream_memops = false;
        CUDA_RUNTIME_CHECK(cudaGetLastError());
    }

    status = CUPFN(nvshmemi_cuda_syms, cuDeviceGetAttribute)(
        &nvshmemi_can_flush_remote_writes,
        (CUdevice_attribute)CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES, device);
    if (status != CUDA_SUCCESS) {
        nvshmemi_can_flush_remote_writes = false;
        CUDA_RUNTIME_CHECK(cudaGetLastError());
    }
}

int nvshmemi_common_init(nvshmemi_state_t *state) {
    int status = 0;
    void *dev_state_ptr = NULL;
    void *transport_dev_state_ptr = NULL;
    bool nvshmemi_use_cuda_vmm = 0;
    cpu_set_t my_set;
    CPU_ZERO(&my_set);

    if (nvshmemi_device_state.nvshmemi_is_nvshmem_initialized) return 0;

    if (!nvshmemi_cuda_syms) {
        nvshmemi_cuda_syms =
            (struct nvshmemi_cuda_fn_table *)calloc(1, sizeof(struct nvshmemi_cuda_fn_table));
        NVSHMEMI_NULL_ERROR_JMP(nvshmemi_cuda_syms, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                "Unable to allocate cuda function table.\n");
    }

    status = nvshmemi_cuda_library_init(nvshmemi_cuda_syms);
    NZ_DEBUG_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem cuda library init failed \n");

    CUDA_RUNTIME_CHECK(cudaDriverGetVersion(&nvshmemi_cuda_driver_version));
    if (strncasecmp(nvshmemi_options.HEAP_KIND, "SYSMEM", 100) == 0) {
        nvshmemi_device_state.symmetric_heap_kind = NVSHMEMI_HEAP_KIND_SYSMEM;
        INFO(NVSHMEM_INIT, "NVSHMEM symmetric heap kind = SYSMEM selected");
    } else {
        nvshmemi_device_state.symmetric_heap_kind = NVSHMEMI_HEAP_KIND_VIDMEM;
        INFO(NVSHMEM_INIT, "NVSHMEM symmetric heap kind = DEVICE selected");
    }

    if (nvshmemi_options.ENABLE_RAIL_OPT == 1) {
        /* Check if npes_node is same on all nodes */
        int *npes_node_all = (int *)malloc(sizeof(int) * nvshmemi_boot_handle.pg_size);
        status = nvshmemi_boot_handle.allgather((void *)&nvshmemi_boot_handle.npes_node,
                                                (void *)npes_node_all, sizeof(int),
                                                &nvshmemi_boot_handle);
        int i;
        for (i = 1; i < nvshmemi_boot_handle.pg_size; i++) {
            if (npes_node_all[i] != npes_node_all[i - 1]) {
                nvshmemi_device_state.enable_rail_opt = 0;
                INFO(NVSHMEM_INIT,
                     "Rail Optimization requested, but npes_node not same for all nodes."
                     "Disabling rail optimization.");
                break;
            }
        }
        free(npes_node_all);

        if (i == nvshmemi_boot_handle.pg_size &&
            nvshmemi_device_state.symmetric_heap_kind == NVSHMEMI_HEAP_KIND_SYSMEM) {
            nvshmemi_device_state.enable_rail_opt = 1;
            INFO(NVSHMEM_INIT, "Enabling Rail Optimization");
        } else {
            INFO(NVSHMEM_INIT,
                 "Rail optimization not supported for symmetric heap in device memory");
        }
    }
    if (nvshmemi_cuda_driver_version >= 11030 && nvshmemi_options.DISABLE_CUDA_VMM == 0 &&
        nvshmemi_device_state.symmetric_heap_kind == NVSHMEMI_HEAP_KIND_VIDMEM) {
        nvshmemi_use_cuda_vmm = 1;
    } else {
        nvshmemi_use_cuda_vmm = 0;
    }

    status = nvshmemi_get_cucontext(state);
    NZ_DEBUG_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem get cucontext failed \n");

    nvshmemi_query_cuda_attributes();

    if (nvshmemi_use_cuda_vmm &&
        nvshmemi_cuda_driver_version < 12050) {  // stream mem ops could not be used with VMM memory
                                                 // until CUDA 12.5 because of a bug in CUDA driver
        nvshmemi_can_use_cuda_64_bit_stream_memops = false;
    }

    if (!nvshmemi_can_use_cuda_64_bit_stream_memops) {
        INFO(NVSHMEM_INIT, "CUDA 64-bit stream memops support is not available");
    }

    /* Set max teams before reserving heap */
    status = nvshmemi_set_max_teams();
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "Requested too many teams.\n");

    /* Context needs to be retrieved and memops flag need to be applied before heap is initialized
     */
    nvshmemi_init_symmetric_heap(state, nvshmemi_use_cuda_vmm,
                                 nvshmemi_device_state.symmetric_heap_kind);

    nvshmemi_detect_nvls_support(state);
    nvshmemi_get_mem_handle(&dev_state_ptr, &transport_dev_state_ptr);
    NVSHMEMI_NULL_ERROR_JMP(dev_state_ptr, status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                            "Unable to query pointer information.\n");
    status = register_state_ptr(dev_state_ptr, transport_dev_state_ptr);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                          "Invalid context pointer passed to nvshmemid_hostlib_init_attr.\n");

    status = nvshmemi_detect_same_device(state);
    NZ_DEBUG_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem detect same device failed \n");

    status = nvshmemi_setup_stream_priorities(state);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "nvshmem setup stream priorities failed \n");

    status = nvshmemi_coll_common_cpu_init();
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cpu collective setup failed \n");

    status = state->heap_obj->reserve_heap();
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "nvshmem reserve static heaps failed \n");

    status = nvshmemi_transport_init(state);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem detect topo failed \n");

    status = nvshmemi_build_transport_map(state);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "building transport map failed \n");

    status = nvshmemi_setup_cuda_handles(state);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cuda handles setup failed \n");

    status = nvshmemi_setup_nvshmem_handles(state);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem handles setup failed \n");

    status = nvshmemi_setup_connections(state);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "nvshmem setup connections failed \n");

    status = state->heap_obj->setup_symmetric_heap();
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "nvshmem register static heaps failed \n");

    nvshmemi_coll_common_cpu_check_ll128_availability();

    status = nvshmemi_init_device_state(state);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "nvshmem device state setup failed \n");

    nvshmemi_update_device_state();
    nvshmemi_device_state.nvshmemi_is_nvshmem_initialized = 1;

    if (sched_getaffinity(0, sizeof(my_set), &my_set) == 0) {
        int core_count = 0;

        for (int i = 0; i < CPU_SETSIZE; i++) {
            if (CPU_ISSET(i, &my_set)) core_count++;
        }

        if (core_count == 1) {
            WARN("Proxy thread shares a core with the main PE, performance may be impacted");
        }
    }

    status = nvshmemi_proxy_init(state, nvshmemi_proxy_level(state));
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "proxy initialization failed \n");

    nvshmemi_boot_handle.barrier(&nvshmemi_boot_handle);
    status = nvshmemi_team_init();
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "team setup failed \n");

    nvshmemi_boot_handle.barrier(&nvshmemi_boot_handle);

    if (nvshmemi_is_mpg_run) {
        status = nvshmemi_determine_mpg_support_level();
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "call to nvshmemi_determine_mpg_support_level failed \n");
    }

    if (nvshmemi_is_limited_mpg_run) {
        status = nvshmemi_setup_limited_mpg_support();
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "mps setup failed \n");
    }
    nvshmemi_boot_handle.barrier(&nvshmemi_boot_handle);

    nvshmemi_update_device_state();
    nvshmemi_is_device_state_ready = 1;

    nvshmemi_barrier_all();
out:
    return status;
}

int nvshmemi_try_common_init(nvshmemi_state_t *state) {
    int status = 0;
    status = nvshmemi_common_init(state);
    if (status) {
        INFO(NVSHMEM_INIT, "nvshmemi_common_init failed, continuing");
        status = 0;
    }

    return status;
}

void nvshmemi_check_state_and_init() {
    if (!nvshmemi_device_state.nvshmemi_is_nvshmem_bootstrapped)
        NVSHMEMI_ERROR_EXIT("nvshmem API called before nvshmem_init \n");
    if (!nvshmemi_device_state.nvshmemi_is_nvshmem_initialized) {
        if (nvshmemi_common_init(nvshmemi_state)) {
            NVSHMEMI_ERROR_EXIT("nvshmem initialization failed, exiting \n");
        }
    }
}

int nvshmemid_init_status() {
    if (!nvshmemi_device_state.nvshmemi_is_nvshmem_bootstrapped)
        return NVSHMEM_STATUS_NOT_INITIALIZED;
    else if (!nvshmemi_device_state.nvshmemi_is_nvshmem_initialized)
        return NVSHMEM_STATUS_IS_BOOTSTRAPPED;
    else if (!nvshmemi_is_mpg_run)
        return NVSHMEM_STATUS_IS_INITIALIZED;
    else if (nvshmemi_is_limited_mpg_run)
        return NVSHMEM_STATUS_LIMITED_MPG;
    else
        return NVSHMEM_STATUS_FULL_MPG;
}

int nvshmemx_init_status() { return nvshmemid_init_status(); }

int nvshmemid_hostlib_init_attr(int requested, int *provided, unsigned int bootstrap_flags,
                                nvshmemx_init_attr_t *attr,
                                nvshmemi_version_t nvshmem_device_lib_version,
                                nvshmemx_device_lib_init_cb cb) {
    int status = 0;

    if (nvshmemi_is_version_compatible(nvshmemi_host_lib_version, nvshmem_device_lib_version) !=
        0) {
        printf("NVSHMEM device library version does not match with NVSHMEM host library version\n");
        return 1;
    }

    if (!nvshmemi_device_state.nvshmemi_is_nvshmem_bootstrapped) {
        nvshmemi_device_state = NVSHMEMI_DEVICE_HOST_STATE_INITIALIZER;
#ifdef NVSHMEM_IBGDA_SUPPORT
        nvshmemi_init_ibgda_device_state(nvshmemi_ibgda_device_state);
#endif
    }

    if (cb) {
        registered_device_state_cb.emplace(cb);
    } else {
        nvshmemi_init_counter++;
    }

    if (!nvshmemi_device_state.nvshmemi_is_nvshmem_bootstrapped) {
        nvshmemi_options_init();
        if (nvshmemi_options.DEBUG_ATTACH_DELAY_provided) {
            printf("application set to sleep for %d seconds after debug init. PID: %d\n",
                   nvshmemi_options.DEBUG_ATTACH_DELAY, getpid());
        }
        nvshmem_nvtx_init();
        status = nvshmemi_bootstrap_preinit(bootstrap_flags);
    }

    NVTX_FUNC_RANGE_IN_GROUP(INIT);

    if (!nvshmemi_device_state.nvshmemi_is_nvshmem_bootstrapped) {
        NVSHMEMU_THREAD_CS_INIT();
        nvshmemi_init_debug();

        if (nvshmemi_options.DEBUG_ATTACH_DELAY) {
            INFO(NVSHMEM_INIT,
                 "sleeping for %d seconds. Now would be a good time to attach a debugger\n",
                 nvshmemi_options.DEBUG_ATTACH_DELAY);
            sleep(nvshmemi_options.DEBUG_ATTACH_DELAY);
        }

        status |= nvshmemi_bootstrap(bootstrap_flags, attr);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem_bootstrap failed \n");

        nvshmemi_init_msg();

        nvshmemi_device_state.nvshmemi_is_nvshmem_bootstrapped = true;
        atexit(bootstrap_finalize);
    }

    if (!nvshmemi_device_state.nvshmemi_is_nvshmem_initialized) {
        if (!nvshmemi_state) {
            nvshmemi_state = (nvshmemi_state_t *)calloc(1, sizeof(nvshmemi_state_t));
            NVSHMEMI_NULL_ERROR_JMP(nvshmemi_state, status, NVSHMEMX_ERROR_INTERNAL, out,
                                    "nvshmemi_init_thread/calloc failed \n");
            nvshmemi_init_nvshmemi_state(nvshmemi_state);
        }

        status = nvshmemi_try_common_init(nvshmemi_state);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "nvshmem common init failed \n");
    } else {
        status = nvshmemi_update_device_state();
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "nvshmem update device state failed \n");
    }

    *provided = NVSHMEM_THREAD_SERIALIZED;

out:
    if (status) NVSHMEMU_THREAD_CS_FINALIZE();

    return (status);
}

int nvshmemx_hostlib_init_attr(unsigned int flags, nvshmemx_init_attr_t *attr) {
    int provided;

    return nvshmemid_hostlib_init_attr(NVSHMEM_THREAD_SERIALIZED, &provided, flags, attr,
                                       nvshmemi_host_lib_version, NULL);
}

void nvshmem_query_thread(int *provided) { *provided = NVSHMEM_THREAD_SERIALIZED; }

#ifndef __CUDA_ARCH__
void nvshmem_global_exit(int status) {
    nvshmemi_device_state.nvshmemi_is_nvshmem_bootstrapped =
        false; /* Set it to 0 so that atexit does not try to finalize_bootstrap */
    /* We can't fix anything if the call to nvshmemi_proxy_finalize fails so don't check the error
     * message. We need to stop the proxy thread before calling global exit to stop a race between
     * the proxy and the atexit bootstrap_finalize function.
     */
    nvshmemi_proxy_finalize(nvshmemi_state);
    /** Bootstraps like UID are agnostic of execution environment, so this API is optional */
    if (nvshmemi_boot_handle.global_exit) nvshmemi_boot_handle.global_exit(status);
}
#endif

void nvshmemid_hostlib_finalize(void *device_ctx, void *transport_device_ctx) {
    NVTX_FUNC_RANGE_IN_GROUP(INIT);

    int status = 0;
    int pid = getpid();
    void *dev_state_ptr;
    void *transport_dev_state_ptr = NULL;

    /* It is invalid to pass a NULL device ctx and a valid transport context */
    if (device_ctx) {
        status = unregister_state_ptr(device_ctx, transport_device_ctx);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                              "Invalid context pointer passed to nvshmemid_hostlib_finalize.\n");
    }

    nvshmemi_init_counter--;
    if (nvshmemi_init_counter != 0) return;

    nvshmemi_get_mem_handle(&dev_state_ptr, &transport_dev_state_ptr);
    NVSHMEMI_NULL_ERROR_JMP(dev_state_ptr, status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                            "Unable to query pointer information.\n");
    status = unregister_state_ptr(dev_state_ptr, transport_dev_state_ptr);
    if (status) {
        INFO(NVSHMEM_INIT,
             "Unable to "
             "unregister internal state. finalized before initialization "
             "complete\n");
        status = 0;
    }

    INFO(NVSHMEM_INIT, "[%d] in nvshmem_finalize:", pid);

    if (nvshmemi_device_state.nvshmemi_is_nvshmem_initialized) {
        nvshmemi_barrier_all();
        nvshmemx_quiet_on_stream(
            nvshmemi_state->my_stream); /* wait for signal ops from barrier to complete */
        status = cudaDeviceSynchronize();
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Teams cleanup device synchronization failed \n");

        /* barrier to ensure all previous ops are complete */
        nvshmemi_boot_handle.barrier(&nvshmemi_boot_handle);

        /* mps finalize */
        if (nvshmemi_is_limited_mpg_run) {
            status = nvshmemi_mpg_finalize();
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "MPS cleanup failed \n");
        }

        /* teams cleanup */
        status = nvshmemi_team_finalize();
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "Teams cleanup failed \n");

        /*cleaning up proxy*/
        if (nvshmemi_proxy_level(nvshmemi_state)) {
            status = nvshmemi_proxy_finalize(nvshmemi_state);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "proxy cleanup failed \n");
        }

        /* collective cleanup */
        status = nvshmemi_coll_common_cpu_finalize();
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "CPU collectives cleanup failed \n");

        status = nvshmemi_teardown_handles(nvshmemi_state);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "handles cleanup failed \n");

        status = nvshmemi_state->heap_obj->cleanup_symmetric_heap();
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "symmetric heap cleanup failed \n");

        nvshmemi_fini_symmetric_heap(nvshmemi_state);

        status = nvshmemi_transport_finalize(nvshmemi_state);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "nvshmem transport finalize failed \n");

        /* Device cleanup */
        if (nvshmemi_device_state.peer_heap_base_p2p)
            CUDA_RUNTIME_CHECK(cudaFree(nvshmemi_device_state.peer_heap_base_p2p));
        if (nvshmemi_device_state.peer_heap_base_remote)
            CUDA_RUNTIME_CHECK(cudaFree(nvshmemi_device_state.peer_heap_base_remote));
        if (nvshmemi_device_state.test_wait_any_start_idx_ptr)
            CUDA_RUNTIME_CHECK(cudaFree(nvshmemi_device_state.test_wait_any_start_idx_ptr));

        /* cleanup state */
        free(nvshmemi_state);

        /* Multi-init Multi-fini*/
        nvshmemi_state = NULL;
        nvshmemi_device_state.nvshmemi_is_nvshmem_initialized = 0;
        nvshmemi_is_device_state_ready = false;
    } else
        nvshmemi_boot_handle.barrier(&nvshmemi_boot_handle);

out:
    if (status) {
        NVSHMEMI_ERROR_PRINT("aborting due to error in nvshmem_finalize \n");
        exit(-1);
    }
}

void nvshmemx_hostlib_finalize() { nvshmemid_hostlib_finalize(NULL, NULL); }

static void nvshmemi_init_debug() {
    const char *nvshmem_debug = nvshmemi_options.DEBUG;
    if (!nvshmemi_options.DEBUG_provided && !nvshmemi_options.DEBUG_SUBSYS_provided) {
        nvshmem_debug_level = NVSHMEM_LOG_NONE;
    } else if (strcmp_case_insensitive(nvshmem_debug, "VERSION") == 0) {
        nvshmem_debug_level = NVSHMEM_LOG_VERSION;
    } else if (strcmp_case_insensitive(nvshmem_debug, "WARN") == 0) {
        nvshmem_debug_level = NVSHMEM_LOG_WARN;
    } else if (strcmp_case_insensitive(nvshmem_debug, "INFO") == 0) {
        nvshmem_debug_level = NVSHMEM_LOG_INFO;
    } else if (strcmp_case_insensitive(nvshmem_debug, "ABORT") == 0) {
        nvshmem_debug_level = NVSHMEM_LOG_ABORT;
    } else if (strcmp_case_insensitive(nvshmem_debug, "TRACE") == 0) {
        nvshmem_debug_level = NVSHMEM_LOG_TRACE;
    } else {
        /* OpenSHMEM spec treats SHMEM_DEBUG as a boolean, enable INFO logging
         * when user-supplied value does match one of the above. */
        nvshmem_debug_level = NVSHMEM_LOG_INFO;
    }

    /* Parse the NVSHMEM_DEBUG_SUBSYS env var
     * This can be a comma separated list such as INIT,COLL
     * or ^INIT,COLL etc
     */
    /* Note: strtok will modify the string, operate on a copy */
    char *nvshmem_debug_subsys = strdup(nvshmemi_options.DEBUG_SUBSYS);
    if (nvshmem_debug_subsys != NULL) {
        char *subsys = strtok(nvshmem_debug_subsys, ",");
        while (subsys != NULL) {
            int invert = 0;
            uint64_t mask = 0;
            if (subsys[0] == '^') {
                invert = 1;
                subsys++;
            }
            if (strcmp_case_insensitive(subsys, "INIT") == 0) {
                mask = NVSHMEM_INIT;
            } else if (strcmp_case_insensitive(subsys, "COLL") == 0) {
                mask = NVSHMEM_COLL;
            } else if (strcmp_case_insensitive(subsys, "P2P") == 0) {
                mask = NVSHMEM_P2P;
            } else if (strcmp_case_insensitive(subsys, "PROXY") == 0) {
                mask = NVSHMEM_PROXY;
            } else if (strcmp_case_insensitive(subsys, "TRANSPORT") == 0) {
                mask = NVSHMEM_TRANSPORT;
            } else if (strcmp_case_insensitive(subsys, "MEM") == 0) {
                mask = NVSHMEM_MEM;
            } else if (strcmp_case_insensitive(subsys, "BOOTSTRAP") == 0) {
                mask = NVSHMEM_BOOTSTRAP;
            } else if (strcmp_case_insensitive(subsys, "TOPO") == 0) {
                mask = NVSHMEM_TOPO;
            } else if (strcmp_case_insensitive(subsys, "UTIL") == 0) {
                mask = NVSHMEM_UTIL;
            } else if (strcmp_case_insensitive(subsys, "ALL") == 0) {
                mask = NVSHMEM_ALL;
            } else {
                mask = 0;
                WARN("Unrecognized value in DEBUG_SUBSYS: %s%s", invert ? "^" : "", subsys);
            }
            if (mask) {
                if (invert)
                    nvshmem_debug_mask &= ~mask;
                else
                    nvshmem_debug_mask |= mask;
            }
            subsys = strtok(NULL, ",");
        }

        free(nvshmem_debug_subsys);
    }

    /* Parse and expand the NVSHMEM_DEBUG_FILE path and
     * then create the debug file. But don't bother unless the
     * NVSHMEM_DEBUG level is > VERSION
     */
    const char *nvshmem_debug_filename = nvshmemi_options.DEBUG_FILE;
    if (nvshmem_debug_level > NVSHMEM_LOG_VERSION && nvshmemi_options.DEBUG_FILE_provided) {
        int c = 0;
        char debugFn[PATH_MAX + 1] = "";
        char *dfn = debugFn;
        while (nvshmem_debug_filename[c] != '\0' && c < PATH_MAX) {
            if (nvshmem_debug_filename[c++] != '%') {
                *dfn++ = nvshmem_debug_filename[c - 1];
                continue;
            }
            switch (nvshmem_debug_filename[c++]) {
                case '%':  // Double %
                    *dfn++ = '%';
                    break;
                case 'h':  // %h = hostname
                    char hostname[1024];
                    nvshmemu_gethostname(hostname, 1024);
                    dfn += snprintf(dfn, PATH_MAX, "%s", hostname);
                    break;
                case 'p':  // %p = pid
                    dfn += snprintf(dfn, PATH_MAX, "%d", getpid());
                    break;
                default:  // Echo everything we don't understand
                    *dfn++ = '%';
                    *dfn++ = nvshmem_debug_filename[c - 1];
                    break;
            }
        }
        *dfn = '\0';
        if (debugFn[0] != '\0') {
            FILE *file = fopen(debugFn, "w");
            if (file != NULL) {
                INFO(NVSHMEM_ALL, "DEBUG file is '%s'", debugFn);
                nvshmem_debug_file = file;
            }
        }
    }
    pthread_mutex_init(&nvshmem_debug_output_lock, NULL);

#ifdef NVSHMEM_TRACE
    nvshmem_epoch = std::chrono::high_resolution_clock::now();
#endif
}

static void nvshmemi_init_msg(void) {
    if (0 == nvshmemi_boot_handle.pg_rank) {
        if (nvshmemi_options.VERSION) printf("%s\n", NVSHMEM_VENDOR_STRING);

        if (nvshmemi_options.DEBUG_provided) {
            int runtimeVersion, driverVersion;
            cudaError_t err;

            printf("NVSHMEM configuration:\n");

            printf("  %-28s %d\n", "CUDA API", CUDA_VERSION);

            err = cudaRuntimeGetVersion(&runtimeVersion);
            if (err != cudaSuccess) runtimeVersion = -1;
            printf("  %-28s %d\n", "CUDA Runtime", runtimeVersion);

            err = cudaDriverGetVersion(&driverVersion);
            if (err != cudaSuccess) driverVersion = -1;
            printf("  %-28s %d\n", "CUDA Driver", driverVersion);

            printf("  %-28s %s %s\n", "Build Timestamp", __DATE__, __TIME__);

            char *build_vars = nvshmemu_wrap(NVSHMEM_BUILD_VARS, NVSHMEMI_WRAPLEN, "\t", 0);
            printf("  %-28s\n\t%s\n", "Build Variables",
                   build_vars ? build_vars : "Error wrapping build vars");
            free(build_vars);

            printf("\n");
        }

        if (nvshmemi_options.INFO) {
            nvshmemi_options_print(NVSHMEMI_OPTIONS_STYLE_INFO);
            nvshmemi_boot_handle.show_info(&nvshmemi_boot_handle, BOOTSTRAP_OPTIONS_STYLE_INFO);
        }
    }

    if (nvshmemi_options.DEBUG_provided || nvshmemi_options.DEBUG_SUBSYS_provided)
        nvshmemu_debug_log_cpuset(NVSHMEM_INIT, "process");
}

int nvshmemi_proxy_level(nvshmemi_state_t *state) {
    for (int i = 0; i < state->num_initialized_transports; i++) {
        if (state->transports[i]->is_successfully_initialized) {
            if (state->transports[i]->no_proxy) {
                continue;
            } else {
                return NVSHMEMI_PROXY_FULL;
            }
        }
    }

    if (nvshmemi_options.DISABLE_LOCAL_ONLY_PROXY) {
        return NVSHMEMI_PROXY_NONE;
    }

    return NVSHMEMI_PROXY_MINIMAL;
}

int set_job_connectivity(nvshmemi_state_t *state) {
    int status;
    int *job_connectivity_all;
    bool proxy_ops_are_ordered = true;
    int gpu_remote_atomics = false;

    // determine job level connectivity among GPUs
    nvshmemi_job_connectivity = NVSHMEMI_JOB_GPU_LDST_ATOMICS;
    for (int i = 0; i < state->npes; i++) {
        int peer_connectivity = NVSHMEMI_JOB_GPU_PROXY;
        void *enforce_cst = NULL;
        // for each PE, pick the best connectivity of any transport
        for (int j = 0; j < state->num_initialized_transports; j++) {
            if (state->transports[j]) {
                if (state->transports[j]->cap[i] & NVSHMEM_TRANSPORT_CAP_MAP_GPU_ATOMICS) {
                    peer_connectivity = (int)NVSHMEMI_JOB_GPU_LDST_ATOMICS;
                } else if (state->transports[j]->cap[i] &
                           (NVSHMEM_TRANSPORT_CAP_MAP_GPU_ST | NVSHMEM_TRANSPORT_CAP_MAP_GPU_LD)) {
                    peer_connectivity = std::min(peer_connectivity, (int)NVSHMEMI_JOB_GPU_LDST);
                }
#ifdef NVSHMEM_IBGDA_SUPPORT
                else if (state->transports[j]->cap[i] &
                         (NVSHMEM_TRANSPORT_CAP_GPU_WRITE | NVSHMEM_TRANSPORT_CAP_GPU_READ |
                          NVSHMEM_TRANSPORT_CAP_GPU_ATOMICS)) {
                    peer_connectivity = std::min(peer_connectivity, (int)NVSHMEMI_JOB_GPU_PROXY);
                    /* Note, these are not mapped atomics. They would be atomics issued from the GPU
                     * over a remote transport (e.g. IBGDA). */
                    if (state->transports[j]->cap[i] & NVSHMEM_TRANSPORT_CAP_GPU_ATOMICS) {
                        gpu_remote_atomics = true;
                    }
                }
#endif
                else {
                    peer_connectivity = std::min(peer_connectivity, (int)NVSHMEMI_JOB_GPU_PROXY);
                    enforce_cst = (void *)state->transports[j]->host_ops.enforce_cst_at_target;
                }
            }
        }

        if ((peer_connectivity == NVSHMEMI_JOB_GPU_PROXY) && (enforce_cst)) {
            peer_connectivity = NVSHMEMI_JOB_GPU_PROXY_CST;
        }

        // for the job, pick the weakest connecitivity to any remote PEs
        nvshmemi_job_connectivity = std::max(nvshmemi_job_connectivity, peer_connectivity);
    }

    /* This case allows us to differentiate between cases where we only support LDST
     * and cases where we have LDST + atomics over a remote transport elsewhere in the code.
     * This catches cases where the remote transport either does, or does not have a proxy.
     */
    gpu_remote_atomics =
        nvshmemi_proxy_level(state) == NVSHMEMI_PROXY_FULL ? true : gpu_remote_atomics;
    if (nvshmemi_job_connectivity == NVSHMEMI_JOB_GPU_LDST && gpu_remote_atomics) {
        nvshmemi_job_connectivity = NVSHMEMI_JOB_GPU_LDST_REMOTE_ATOMICS;
    }

    // agree on maximumg distance for job_connectivity among all PEs
    job_connectivity_all = (int *)malloc(sizeof(int) * state->npes);
    NVSHMEMI_NULL_ERROR_JMP(job_connectivity_all, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "memory allocation for job_connectivity_all failed \n");

    status = nvshmemi_boot_handle.allgather((void *)&nvshmemi_job_connectivity,
                                            (void *)job_connectivity_all, sizeof(int),
                                            &nvshmemi_boot_handle);
    if (status != 0) {
        free(job_connectivity_all);
        NVSHMEMI_ERROR_PRINT("allgather of job_connectivity failed \n");
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }

    for (int i = 0; i < state->npes; i++) {
        nvshmemi_job_connectivity = std::max(nvshmemi_job_connectivity, job_connectivity_all[i]);
    }
    free(job_connectivity_all);
    nvshmemi_device_state.job_connectivity = nvshmemi_job_connectivity;

    // check if all proxy ops are ordered
    for (int i = 0; i < state->num_initialized_transports; i++) {
        if (state->transports[i] && (state->transports[i]->host_ops.fence != NULL))
            proxy_ops_are_ordered = false;
    }
    nvshmemi_device_state.proxy_ops_are_ordered = proxy_ops_are_ordered;

out:
    return status;
}

int nvshmemi_init_device_state(nvshmemi_state_t *state) {
    int status = CUDA_SUCCESS;
    int warp_size = 0;

    CUDA_RUNTIME_CHECK_GOTO(
        cudaDeviceGetAttribute(&warp_size, cudaDevAttrWarpSize, state->device_id), status, out);
    if (NVSHMEMI_WARP_SIZE != warp_size) {
        status = NVSHMEMX_ERROR_INTERNAL;
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "device warp size (%d) does not match assumed warp size (%d)\n",
                              warp_size, NVSHMEMI_WARP_SIZE);
    }

    CUDA_RUNTIME_CHECK_GOTO(cudaMalloc(&heap_base_array_dptr, (state->npes) * sizeof(void *)),
                            status, out);
    CUDA_RUNTIME_CHECK_GOTO(
        cudaMalloc(&heap_base_actual_array_dptr, (state->npes) * sizeof(void *)), status, out);

    status = set_job_connectivity(state);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "set_job_connectivity failed \n");

    CUDA_RUNTIME_CHECK_GOTO(
        cudaMemcpyAsync(heap_base_array_dptr, (const void *)state->heap_obj->get_local_pe_base(),
                        sizeof(void *) * state->npes, cudaMemcpyHostToDevice, state->my_stream),
        status, out);
    CUDA_RUNTIME_CHECK_GOTO(
        cudaMemcpyAsync(heap_base_actual_array_dptr,
                        (const void *)state->heap_obj->get_remote_pe_base(),
                        sizeof(void *) * state->npes, cudaMemcpyHostToDevice, state->my_stream),
        status, out);

    CUDA_RUNTIME_CHECK_GOTO(cudaStreamSynchronize(state->my_stream), status, out);

    nvshmemi_device_state.proxy = nvshmemi_proxy_level(state);

    if (nvshmemi_options.ASSERT_ATOMICS_SYNC)
        nvshmemi_device_state.atomics_sync = 1;
    else
        nvshmemi_device_state.atomics_sync = 0;

    nvshmemi_device_state.atomics_le_min_size = state->atomic_host_endian_min_size;

    for (int i = 0; i < state->npes; i++) {
        int t_idx = state->selected_transport_for_amo[i];
        if (t_idx < 0) {
            continue;
        }
        if (state->transports[t_idx]->atomics_complete_on_quiet) {
            nvshmemi_device_state.atomics_complete_on_quiet = true;
            break;
        }
    }

    nvshmemi_device_state.peer_heap_base_p2p = (void **)heap_base_array_dptr;

    INFO(NVSHMEM_INIT,
         "[%d] status %d cudaErrorInvalidValue %d cudaErrorInvalidSymbol %d "
         "cudaErrorInvalidMemcpyDirection %d cudaErrorNoKernelImageForDevice %d",
         state->mype, status, cudaErrorInvalidValue, cudaErrorInvalidSymbol,
         cudaErrorInvalidMemcpyDirection, cudaErrorNoKernelImageForDevice);

    nvshmemi_device_state.peer_heap_base_remote = (void **)heap_base_actual_array_dptr;
    nvshmemi_device_state.heap_base = state->heap_obj->get_base();
    nvshmemi_device_state.heap_size = state->heap_obj->get_size();
    nvshmemi_device_state.mype = state->mype;
    nvshmemi_device_state.npes = state->npes;
    nvshmemi_device_state.node_mype = state->mype_node;
    nvshmemi_device_state.node_npes = state->npes_node;

    CUDA_RUNTIME_CHECK_GOTO(cudaStreamSynchronize(state->my_stream), status, out);

    unsigned long long *test_wait_any_start_idx_ptr;
    CUDA_RUNTIME_CHECK(
        cudaMalloc((void **)&test_wait_any_start_idx_ptr, sizeof(unsigned long long)));
    CUDA_RUNTIME_CHECK(
        cudaMemset((void *)test_wait_any_start_idx_ptr, 0, sizeof(unsigned long long)));

    nvshmemi_device_state.test_wait_any_start_idx_ptr = test_wait_any_start_idx_ptr;

    nvshmemi_update_device_state();

out:
    if (status) {
        if (heap_base_array_dptr) CUDA_RUNTIME_CHECK(cudaFree(heap_base_array_dptr));
        if (heap_base_actual_array_dptr) CUDA_RUNTIME_CHECK(cudaFree(heap_base_actual_array_dptr));
        if (test_wait_any_start_idx_ptr) CUDA_RUNTIME_CHECK(cudaFree(test_wait_any_start_idx_ptr));
    }
    return status;
}

int nvshmemx_cumodule_init(CUmodule module) {
    int status = 0;
    CUdeviceptr dptr, transport_dptr = 0;
    size_t size;
    nvshmemi_version_t module_nvshmem_version;

    CUCHECKGOTO(nvshmemi_cuda_syms,
                cuModuleGetGlobal(&dptr, &size, module, "nvshmemi_device_lib_version_d"), status,
                out);
    CUDA_RUNTIME_CHECK(cudaMemcpy((void *)&module_nvshmem_version, (const void *)dptr, size,
                                  cudaMemcpyDeviceToHost));
    if (nvshmemi_is_version_compatible(nvshmemi_host_lib_version, module_nvshmem_version) != 0) {
        printf("NVSHMEM version in CUmodule does not match with NVSHMEM host library version\n");
        return 1;
    }

    CUCHECKGOTO(nvshmemi_cuda_syms,
                cuModuleGetGlobal(&dptr, &size, module, "nvshmemi_device_state_d"), status, out);
#ifdef NVSHMEM_IBGDA_SUPPORT
    CUCHECKGOTO(nvshmemi_cuda_syms,
                cuModuleGetGlobal(&transport_dptr, &size, module, "nvshmemi_ibgda_device_state_d"),
                status, out);
#endif
    status = register_state_ptr((void *)dptr, (void *)transport_dptr);
    NVSHMEMI_NE_ERROR_JMP(status, NVSHMEMX_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "Unable to register cumodule state pointer. failed\n");

    status = nvshmemi_update_device_state();
    NVSHMEMI_NE_ERROR_JMP(status, NVSHMEMX_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "Unable to update cumodule state pointer. failed\n");

    status = cudaDeviceSynchronize();
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "cudaMemcpyFromSymbol failed\n");
out:
    return status;
}

int nvshmemx_cumodule_finalize(CUmodule module) {
    int status = 0;
    CUdeviceptr dptr, transport_dptr = 0;
    size_t size;

    CUCHECKGOTO(nvshmemi_cuda_syms,
                cuModuleGetGlobal(&dptr, &size, module, "nvshmemi_device_state_d"), status, out);
#ifdef NVSHMEM_IBGDA_SUPPORT
    CUCHECKGOTO(nvshmemi_cuda_syms,
                cuModuleGetGlobal(&transport_dptr, &size, module, "nvshmemi_ibgda_device_state_d"),
                status, out);
#endif
    status = unregister_state_ptr((void *)dptr, (void *)transport_dptr);
    NVSHMEMI_NE_ERROR_JMP(status, NVSHMEMX_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "Unable to unregister cumodule state pointer. failed\n");

out:
    return status;
}
