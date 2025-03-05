/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include <assert.h>                                                        // for assert
#include <dlfcn.h>                                                         // for dlclose, dlerror
#include <stdint.h>                                                        // for SIZE_MAX
#include <stdio.h>                                                         // for snprintf, NULL
#include <stdlib.h>                                                        // for calloc
#include <strings.h>                                                       // for strncasecmp
#include "device_host/nvshmem_types.h"                                     // for nvshmemi_devi...
#include "device_host/nvshmem_common.cuh"                                  // for nvshmemi_devi...
#include "non_abi/nvshmemx_error.h"                                        // for NVSHMEMI_ERRO...
#include "internal/host/debug.h"                                           // for INFO, NVSHMEM...
#include "internal/host/nvshmem_internal.h"                                // for nvshmemi_loca...
#include "internal/host/error_codes_internal.h"                            // for NVSHMEMI_INTE...
#include "internal/host/nvshmemi_symmetric_heap.hpp"                       // for nvshmemi_symm...
#include "internal/host/nvshmemi_types.h"                                  // for nvshmemi_state_t
#include "internal/host/util.h"                                            // for nvshmemi_options
#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"  // for nvshmemi_boot...
#include "bootstrap_host_transport/env_defs_internal.h"                    // for nvshmemi_opti...
#include "internal/host_transport/transport.h"                             // for nvshmem_trans...
#include "non_abi/nvshmem_build_options.h"                                 // for NVSHMEM_IBGDA...
#include "non_abi/nvshmem_version.h"                                       // for NVSHMEM_TRANS...
#include "topo.h"                                                          // for nvshmemi_get_...

#define TRANSPORT_STRING_MAX_LENGTH 8
#define NVSHMEM_TRANSPORT_COUNT 6
#define IB_TRANSPORT_STRING "ibrc"
#define UCX_TRANSPORT_STRING "ucx"
#define DEVX_TRANSPORT_STRING "ibdevx"
#define LIBFABRIC_TRANSPORT_STRING "libfabric"

static void *transport_lib = NULL;
#ifdef NVSHMEM_IBGDA_SUPPORT
static void *transport_lib_IBGDA = NULL;
#endif

int nvshmemi_transport_show_info(nvshmemi_state_t *state) {
    int status = 0;
    nvshmem_transport_t *transports = (nvshmem_transport_t *)state->transports;
    for (int i = 0; i < state->num_initialized_transports; ++i) {
        transports[i]->host_ops.show_info(transports[i], TRANSPORT_OPTIONS_STYLE_INFO);
    }
    return status;
}

int nvshmemi_transport_init(nvshmemi_state_t *state) {
    int status = 0;
    int index = 0;
#if defined(NVSHMEM_IBRC_SUPPORT) || defined(NVSHMEM_UCX_SUPPORT) || \
    defined(NVSHMEM_LIBFABRIC_SUPPORT) || defined(NVSHMEM_IBDEVX_SUPPORT)
    int transport_skipped;
#endif
    nvshmem_transport_t *transports = NULL;
    nvshmemi_transport_init_fn init_fn;
    const int transport_object_file_len = 100;
    char transport_object_file[transport_object_file_len];
    bool transport_selected = false;
    nvshmem_local_buf_cache_t *tmp_cache_ptr = NULL;

    if (!state->transports)
        state->transports =
            (nvshmem_transport_t *)calloc(NVSHMEM_TRANSPORT_COUNT, sizeof(nvshmem_transport_t));

    transports = (nvshmem_transport_t *)state->transports;

    if (!nvshmemi_options.DISABLE_P2P) {
        status = nvshmemi_local_mem_cache_init(&tmp_cache_ptr);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMI_INTERNAL_ERROR, out,
                              "Unable to allocate transport mem cache.\n");
        status = nvshmemt_p2p_init(&transports[index]);
        if (!status) {
            transports[index]->boot_handle = &nvshmemi_boot_handle;
            transports[index]->heap_base = state->heap_obj->get_base();
            transports[index]->cap = (int *)calloc(state->npes, sizeof(int));
            transports[index]->index = index;
            transports[index]->log2_cumem_granularity =
                nvshmemi_state->heap_obj->get_log2_cumem_granularity();
            transports[index]->cache_handle = tmp_cache_ptr;
            if (transports[index]->max_op_len == 0) transports[index]->max_op_len = SIZE_MAX;
            index++;
        } else {
            nvshmemi_local_mem_cache_fini(tmp_cache_ptr);
            NVSHMEMI_ERROR_PRINT("init failed for transport: P2P");
            status = 0;
        }
    } else {
        WARN("P2P access was disabled in the environment");
    }

#ifdef NVSHMEM_IBRC_SUPPORT
    transport_skipped = strncasecmp(nvshmemi_options.REMOTE_TRANSPORT, IB_TRANSPORT_STRING,
                                    TRANSPORT_STRING_MAX_LENGTH);
    if (transport_skipped) {
        INFO(NVSHMEM_INIT, "IBRC transport skipped in favor of: %s\n",
             nvshmemi_options.REMOTE_TRANSPORT);
    } else {
        status = snprintf(transport_object_file, transport_object_file_len,
                          "nvshmem_transport_ibrc.so.%d", NVSHMEM_TRANSPORT_PLUGIN_MAJOR_VERSION);
        if (status > 0 && status < transport_object_file_len) {
            transport_selected = true;
            goto transport_init;
        } else {
            NVSHMEMI_ERROR_PRINT("snprintf call failed in the transport.\n");
        }
    }
#endif

#ifdef NVSHMEM_UCX_SUPPORT
    transport_skipped = strncasecmp(nvshmemi_options.REMOTE_TRANSPORT, UCX_TRANSPORT_STRING,
                                    TRANSPORT_STRING_MAX_LENGTH);
    if (transport_skipped) {
        INFO(NVSHMEM_INIT, "UCX transport skipped in favor of: %s\n",
             nvshmemi_options.REMOTE_TRANSPORT);
    } else {
        status = snprintf(transport_object_file, transport_object_file_len,
                          "nvshmem_transport_ucx.so.%d", NVSHMEM_TRANSPORT_PLUGIN_MAJOR_VERSION);
        if (status > 0 && status < transport_object_file_len) {
            transport_selected = true;
            goto transport_init;
        } else {
            NVSHMEMI_ERROR_PRINT("snprintf call failed in the transport.\n");
        }
    }
#endif

#ifdef NVSHMEM_IBDEVX_SUPPORT
    transport_skipped = strncasecmp(nvshmemi_options.REMOTE_TRANSPORT, DEVX_TRANSPORT_STRING,
                                    TRANSPORT_STRING_MAX_LENGTH);
    if (transport_skipped) {
        INFO(NVSHMEM_INIT, "IBDEVX transport skipped in favor of: %s\n",
             nvshmemi_options.REMOTE_TRANSPORT);
    } else {
        status = snprintf(transport_object_file, transport_object_file_len,
                          "nvshmem_transport_ibdevx.so.%d", NVSHMEM_TRANSPORT_PLUGIN_MAJOR_VERSION);
        if (status > 0 && status < transport_object_file_len) {
            transport_selected = true;
            goto transport_init;
        } else {
            NVSHMEMI_ERROR_PRINT("snprintf call failed in the transport.\n");
        }
    }
#endif

#ifdef NVSHMEM_LIBFABRIC_SUPPORT
    transport_skipped = strncasecmp(nvshmemi_options.REMOTE_TRANSPORT, LIBFABRIC_TRANSPORT_STRING,
                                    TRANSPORT_STRING_MAX_LENGTH);
    if (transport_skipped) {
        INFO(NVSHMEM_INIT, "Libfabric transport skipped in favor of: %s\n",
             nvshmemi_options.REMOTE_TRANSPORT);
    } else {
        status =
            snprintf(transport_object_file, transport_object_file_len,
                     "nvshmem_transport_libfabric.so.%d", NVSHMEM_TRANSPORT_PLUGIN_MAJOR_VERSION);
        if (status > 0 && status < transport_object_file_len) {
            transport_selected = true;
            goto transport_init;
        } else {
            NVSHMEMI_ERROR_PRINT("snprintf call failed in the transport.\n");
        }
    }
#endif

#if defined(NVSHMEM_IBRC_SUPPORT) || defined(NVSHMEM_UCX_SUPPORT) || \
    defined(NVSHMEM_LIBFABRIC_SUPPORT) || defined(NVSHMEM_IBDEVX_SUPPORT)
transport_init:
#endif

    if (!transport_selected) {
        goto transport_fail;
    }

    transport_lib = dlopen(transport_object_file, RTLD_NOW);
    if (transport_lib == NULL) {
        WARN("Unable to open the %s transport. %s\n", transport_object_file, dlerror());
        goto transport_fail;
    }

    init_fn = (nvshmemi_transport_init_fn)dlsym(transport_lib, "nvshmemt_init");
    if (!init_fn) {
        dlclose(transport_lib);
        transport_lib = NULL;
        WARN("Unable to get info from %s transport.\n", transport_object_file);
        goto transport_fail;
    }

    status = nvshmemi_local_mem_cache_init(&tmp_cache_ptr);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMI_INTERNAL_ERROR, out,
                          "Unable to allocate transport mem cache.\n");

    status = init_fn(&transports[index], nvshmemi_cuda_syms, NVSHMEM_TRANSPORT_INTERFACE_VERSION);
    if (!status) {
        assert(NVSHMEM_TRANSPORT_MAJOR_MINOR_VERSION(transports[index]->api_version) <=
               NVSHMEM_TRANSPORT_MAJOR_MINOR_VERSION(NVSHMEM_TRANSPORT_INTERFACE_VERSION));
        transports[index]->boot_handle = &nvshmemi_boot_handle;
        if (nvshmemi_device_state.enable_rail_opt == 1) {
            transports[index]->heap_base = nvshmemi_state->heap_obj->get_global_base();
        } else {
            transports[index]->heap_base = state->heap_obj->get_base();
        }

        transports[index]->log2_cumem_granularity =
            nvshmemi_state->heap_obj->get_log2_cumem_granularity();
        transports[index]->cap = (int *)calloc(state->npes, sizeof(int));
        transports[index]->index = index;
        transports[index]->my_pe = nvshmemi_state->mype;
        transports[index]->n_pes = nvshmemi_state->npes;
        transports[index]->cache_handle = (void *)tmp_cache_ptr;
        if (transports[index]->max_op_len == 0) transports[index]->max_op_len = SIZE_MAX;
        state->atomic_host_endian_min_size = transports[index]->atomic_host_endian_min_size;
        index++;
    } else {
        nvshmemi_local_mem_cache_fini(tmp_cache_ptr);
        dlclose(transport_lib);
        transport_lib = NULL;
        /* non-fatal error, so changing to a warning */
        NVSHMEMI_WARN_PRINT("init failed for remote transport: %s",
                            nvshmemi_options.REMOTE_TRANSPORT);
        status = 0;
    }
transport_fail:
#ifdef NVSHMEM_IBGDA_SUPPORT
    if (nvshmemi_options.IB_ENABLE_IBGDA) {
        status = snprintf(transport_object_file, transport_object_file_len,
                          "nvshmem_transport_ibgda.so.%d", NVSHMEM_TRANSPORT_PLUGIN_MAJOR_VERSION);
        if (status < 0 || status > transport_object_file_len) {
            WARN("Unable to open the %s transport. %s\n", transport_object_file, dlerror());
            goto out;
        }
        transport_lib_IBGDA = dlopen(transport_object_file, RTLD_NOW);
        if (transport_lib_IBGDA == NULL) {
            WARN("Unable to open the %s transport. %s\n", transport_object_file, dlerror());
            goto out;
        }

        init_fn = (nvshmemi_transport_init_fn)dlsym(transport_lib_IBGDA, "nvshmemt_init");
        if (!init_fn) {
            dlclose(transport_lib_IBGDA);
            transport_lib_IBGDA = NULL;
            WARN("Unable to get info from %s transport.\n", transport_object_file);
            goto out;
        }

        status = nvshmemi_local_mem_cache_init(&tmp_cache_ptr);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMI_INTERNAL_ERROR, out,
                              "Unable to allocate transport mem cache.\n");

        status =
            init_fn(&transports[index], nvshmemi_cuda_syms, NVSHMEM_TRANSPORT_INTERFACE_VERSION);
        if (!status) {
            assert(NVSHMEM_TRANSPORT_MAJOR_MINOR_VERSION(transports[index]->api_version) <=
                   NVSHMEM_TRANSPORT_MAJOR_MINOR_VERSION(NVSHMEM_TRANSPORT_INTERFACE_VERSION));
            transports[index]->boot_handle = &nvshmemi_boot_handle;
            if (nvshmemi_device_state.enable_rail_opt == 1) {
                transports[index]->heap_base = nvshmemi_state->heap_obj->get_global_base();
            } else {
                transports[index]->heap_base = state->heap_obj->get_base();
            }
            transports[index]->log2_cumem_granularity =
                nvshmemi_state->heap_obj->get_log2_cumem_granularity();
            transports[index]->cap = (int *)calloc(state->npes, sizeof(int));
            transports[index]->index = index;
            transports[index]->my_pe = nvshmemi_state->mype;
            transports[index]->n_pes = nvshmemi_state->npes;
            transports[index]->cache_handle = (void *)tmp_cache_ptr;
            nvshmemi_ibgda_get_device_state(&transports[index]->type_specific_shared_state);
            if (transports[index]->max_op_len == 0) transports[index]->max_op_len = SIZE_MAX;
            state->atomic_host_endian_min_size = transports[index]->atomic_host_endian_min_size;
            nvshmemi_device_state.ibgda_is_initialized = true;
            index++;
        } else {
            NVSHMEMI_ERROR_PRINT("init failed for transport: IBGDA");
            nvshmemi_local_mem_cache_fini(tmp_cache_ptr);
            dlclose(transport_lib_IBGDA);
            transport_lib_IBGDA = NULL;
            status = 0;
        }
    } else {
        INFO(NVSHMEM_INIT, "IBGDA Disabled by the environment.");
    }
#endif

    if (index == 0) {
        NVSHMEMI_ERROR_PRINT("Unable to initialize any transports. returning error.");
        status = NVSHMEMX_ERROR_INTERNAL;
    }
out:
    state->num_initialized_transports = index;
    if (status > 0) {
        for (int idx = 0; idx < index; idx++) {
            nvshmemi_local_mem_cache_fini(
                (nvshmem_local_buf_cache_t *)transports[idx]->cache_handle);
        }
    }
    return status;
}

int nvshmemi_transport_finalize(nvshmemi_state_t *state) {
    INFO(NVSHMEM_INIT, "In nvshmemi_transport_finalize");
    int status = 0;
    nvshmem_transport_t *transports = NULL;
    ;

    if (!state->transports) return 0;

    transports = (nvshmem_transport_t *)state->transports;

    for (int i = 0; i < state->num_initialized_transports; i++) {
        if (transports[i]->is_successfully_initialized) {
            if (transports[i]->type == NVSHMEM_TRANSPORT_LIB_CODE_IBGDA) {
                nvshmemi_device_state.ibgda_is_initialized = true;
            }
            if (transports[i]->cache_handle) {
                nvshmemi_transport_buffer_unregister_all(transports[i]);
                nvshmemi_local_mem_cache_fini(
                    (nvshmem_local_buf_cache_t *)transports[i]->cache_handle);
            }

            status = transports[i]->host_ops.finalize(transports[i]);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "transport finalize failed \n");
            status = nvshmemi_update_device_state();
        }
    }
out:
    if (transport_lib) {
        dlclose(transport_lib);
        transport_lib = NULL;
    }

#ifdef NVSHMEM_IBGDA_SUPPORT
    if (transport_lib_IBGDA) {
        dlclose(transport_lib_IBGDA);
        transport_lib_IBGDA = NULL;
    }
#endif
    return status;
}

int nvshmemi_setup_connections(nvshmemi_state_t *state) {
    int status = 0;
    nvshmem_transport_t *transports = (nvshmem_transport_t *)state->transports;
    nvshmem_transport_t tcurr;

    for (int i = 0; i < state->num_initialized_transports; i++) {
        if (!((state->transport_bitmap) & (1 << i))) continue;
        tcurr = transports[i];

        if (!(tcurr->attr & NVSHMEM_TRANSPORT_ATTR_CONNECTED)) {
            continue;
        }

        int devices_temp = tcurr->n_devices / state->npes_node;
        if (devices_temp == 0) devices_temp = 1;
        const int max_devices_per_pe = devices_temp;
        int selected_devices[max_devices_per_pe];
        int found_devices = 0;

        for (int j = 0; j < max_devices_per_pe; j++) {
            selected_devices[j] = -1;
        }

        // assumes symmetry of transport list at all PEs
        if (tcurr->n_devices <= 1) {
            /* return the index of the first available device.
             * -1 if no devices found.
             */
            selected_devices[0] = tcurr->n_devices - 1;
            found_devices++;
        } else if (nvshmemi_options.ENABLE_NIC_PE_MAPPING) {
            selected_devices[0] =
                nvshmemi_state->mype_node % (tcurr->n_devices > 0 ? tcurr->n_devices : 1);
            INFO(NVSHMEM_INIT, "NVSHMEM_ENABLE_NIC_PE_MAPPING = 1, setting dev_id = %d",
                 selected_devices[0]);
            found_devices++;
        } else {
            nvshmemi_get_devices_by_distance(selected_devices, max_devices_per_pe, tcurr);
            for (int i = 0; i < max_devices_per_pe; i++) {
                if (selected_devices[i] == -1) {
                    break;
                }
                found_devices++;
                INFO(NVSHMEM_INIT,
                     "NVSHMEM_ENABLE_NIC_PE_MAPPING = 0, device %d setting dev_id = %d", i,
                     selected_devices[i]);
            }
        }

        /* setting n_devices to 0 is the transports way of
         * letting us know it's managing devices internally.
         */
        if (tcurr->n_devices > 0 && selected_devices[0] == -1) {
            NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "No devices selected.\n");
        }

        status = tcurr->host_ops.connect_endpoints(tcurr, selected_devices, found_devices);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "connect EPS failed \n");
        status = nvshmemi_boot_handle.barrier(&nvshmemi_boot_handle);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "barrier failed \n");

        status = nvshmemi_update_device_state();
    }

out:
    return status;
}
