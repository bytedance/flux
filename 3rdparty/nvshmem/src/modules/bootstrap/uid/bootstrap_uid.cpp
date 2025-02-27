/*
 * Copyright (c) 2016-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */
#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <sys/resource.h>
#include <time.h>
#include <algorithm>
#include <cstring>
#include "bootstrap_device_host/nvshmem_uniqueid.h"
#include "bootstrap_host_transport/env_defs_internal.h"
#include "bootstrap_uid_remap.h"
#include "bootstrap_uid_types.hpp"
#include "bootstrap_util.h"
#include "internal/bootstrap_host/nvshmemi_bootstrap.h"
#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"
#include "ncclSocket/ncclsocket_socket.hpp"

static struct bootstrap_netstate priv_info;
struct nvshmemi_options_s env_attr;
int bootstrap_debug_enable = 0;

#define BOOTSTRAP_IN_PLACE (void*)0x1

static_assert(sizeof(bootstrap_uid_handle) < sizeof(nvshmemx_uniqueid_t),
              "bootstrap_uid_handle implementation is incompatible with nvshmemx_uniqueid_t");

#define AF_NUMERICFORM_DEFAULT 1

#define PARSE_AND_RET_NULL(x) ((x##_provided) ? (const_cast<char*>(x)) : nullptr)

/* Define common types */
#define BOOTPRI_string "\"%s\""

#define BOOTSTRAP_OPTIONS_PRINT_ENV(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC, DESIRED_CAT, STYLE) \
    if (CATEGORY == DESIRED_CAT) {                                                                 \
        switch (STYLE) {                                                                           \
            char* desc_wrapped;                                                                    \
            case BOOTSTRAP_OPTIONS_STYLE_INFO:                                                     \
                desc_wrapped = bootstrap_util_wrap_string(SHORT_DESC, 80, "\t", 1);                \
                printf("  NVSHMEM_%-20s " BOOTPRI_##KIND " (type: %s, default: " BOOTPRI_##KIND    \
                       ")\n\t%s\n",                                                                \
                       #NAME, NVSHFMT_##KIND(env_attr.NAME), #KIND, NVSHFMT_##KIND(DEFAULT),       \
                       desc_wrapped);                                                              \
                free(desc_wrapped);                                                                \
                break;                                                                             \
            case BOOTSTRAP_OPTIONS_STYLE_RST:                                                      \
                desc_wrapped = bootstrap_util_wrap_string(SHORT_DESC, 80, NULL, 0);                \
                printf(".. c:var:: NVSHMEM_%s\n", #NAME);                                          \
                printf("\n");                                                                      \
                printf("| *Type: %s*\n", #KIND);                                                   \
                printf("| *Default: " BOOTPRI_##KIND "*\n", NVSHFMT_##KIND(DEFAULT));              \
                printf("\n");                                                                      \
                printf("%s\n", desc_wrapped);                                                      \
                printf("\n");                                                                      \
                free(desc_wrapped);                                                                \
                break;                                                                             \
            default:                                                                               \
                assert(0);                                                                         \
        }                                                                                          \
    }

static int bootstrap_uid_showinfo(struct bootstrap_handle* handle, int style) {
    bootstrap_util_print_header(style, "Bootstrap Options");
#define NVSHMEMI_ENV_DEF(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC)        \
    BOOTSTRAP_OPTIONS_PRINT_ENV(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC, \
                                NVSHMEMI_ENV_CAT_BOOTSTRAP, style)
#include "env_defs.h"
#undef NVSHMEMI_ENV_DEF
    printf("\n");

    bootstrap_util_print_header(style, "Common Options");
#define NVSHMEMI_ENV_DEF(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC)        \
    BOOTSTRAP_OPTIONS_PRINT_ENV(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC, \
                                NVSHMEMI_ENV_CAT_OPENSHMEM, style)
#include "env_defs.h"
#undef NVSHMEMI_ENV_DEF

    return 0;
}

// Additional sync functions
static bootstrap_result_t bootstrap_net_send(bootstrap_uid_socket_t* sock, void* data, int size) {
    BOOTSTRAP_CHECK(nccl_fn_table(send, sock, &size, sizeof(int)));
    BOOTSTRAP_CHECK(nccl_fn_table(send, sock, data, size));
    return BOOTSTRAP_SUCCESS;
}

static bootstrap_result_t bootstrap_net_recv(bootstrap_uid_socket_t* sock, void* data, int size) {
    int recv_size;
    BOOTSTRAP_CHECK(nccl_fn_table(recv, sock, &recv_size, sizeof(int)));
    if (recv_size > size) {
        BOOTSTRAP_ERROR_PRINT("Message truncated : received %d bytes instead of %d\n", recv_size,
                              size);
        return BOOTSTRAP_INTERNAL_ERROR;
    }

    BOOTSTRAP_CHECK(nccl_fn_table(recv, sock, data, std::min(recv_size, size)));
    return BOOTSTRAP_SUCCESS;
}

static bootstrap_result_t set_files_limit() {
    struct rlimit files_limit;
    BOOTSTRAP_SYSCHECK(getrlimit(RLIMIT_NOFILE, &files_limit), "getrlimit");
    files_limit.rlim_cur = files_limit.rlim_max;
    BOOTSTRAP_SYSCHECK(setrlimit(RLIMIT_NOFILE, &files_limit), "setrlimit");
    return BOOTSTRAP_SUCCESS;
}

static void bootstrap_parse_debug(void) {
    if (!env_attr.DEBUG_provided && !env_attr.DEBUG_SUBSYS_provided) {
        /* This is no-operation */
    } else if (strncasecmp(env_attr.DEBUG, "INFO", 5) == 0 ||
               strncasecmp(env_attr.DEBUG, "WARN", 5) == 0 ||
               strncasecmp(env_attr.DEBUG, "TRACE", 6) == 0 ||
               strncasecmp(env_attr.DEBUG, "ABORT", 6) == 0) {
        bootstrap_debug_enable = 1;
    }
}

/**
 * Bootstrap Network Discovery Functions
 */
static bootstrap_result_t bootstrap_net_init() {
    char line[SOCKET_NAME_MAXLEN + MAX_IF_NAME_SIZE + 2] = {0};
    pthread_mutex_lock(&priv_info.bootstrap_netlock);
    if (!priv_info.bootstrap_netinitdone) {
        if (env_attr.BOOTSTRAP_UID_SESSION_ID_provided) {
            bootstrap_uid_socket_address_t remote_addr;
            if (nccl_fn_table(get_addr_from_string, &remote_addr,
                              env_attr.BOOTSTRAP_UID_SESSION_ID) != BOOTSTRAP_SUCCESS) {
                BOOTSTRAP_ERROR_PRINT(
                    "Invalid NVSHMEM_BOOTSTRAP_UID_SESSION_ID, please use format: <ipv4>:<port> or "
                    "[<ipv6>]:<port> or <hostname>:<port>\n");
                pthread_mutex_unlock(&priv_info.bootstrap_netlock);
                return BOOTSTRAP_INVALID_ARGUMENT;
            }

            if (fn_table.find_interface_match_subnet(priv_info.bootstrap_netifname,
                                                     &priv_info.bootstrap_netifaddr, &remote_addr,
                                                     MAX_IF_NAME_SIZE, 1) <= 0) {
                BOOTSTRAP_ERROR_PRINT("No usable listening interface found\n");
                pthread_mutex_unlock(&priv_info.bootstrap_netlock);
                return BOOTSTRAP_SYSTEM_ERROR;
            }
        } else {
            /* No SESSION ID based interface hinted. Use auto-selection */
            int num_ifs = fn_table.find_interfaces(
                priv_info.bootstrap_netifname, &priv_info.bootstrap_netifaddr, MAX_IF_NAME_SIZE, 1,
                PARSE_AND_RET_NULL(env_attr.BOOTSTRAP_UID_SOCK_IFNAME),
                PARSE_AND_RET_NULL(env_attr.BOOTSTRAP_UID_SOCK_FAMILY),
                PARSE_AND_RET_NULL(env_attr.BOOTSTRAP_UID_SESSION_ID));
            if (num_ifs <= 0) {
                BOOTSTRAP_ERROR_PRINT("No socket interface found\n");
                pthread_mutex_unlock(&priv_info.bootstrap_netlock);
                return BOOTSTRAP_INTERNAL_ERROR;
            }
        }

        sprintf(line, " %s:", priv_info.bootstrap_netifname);
        fn_table.to_string(&priv_info.bootstrap_netifaddr, line + strlen(line),
                           AF_NUMERICFORM_DEFAULT);
        BOOTSTRAP_DEBUG_PRINT(
            "UID bootstrap network %s using: %s\n",
            (!priv_info.bootstrap_netinitdone) ? "is being initialized" : "already initialized",
            line);

        // update the global state
        priv_info.bootstrap_netinitdone = 1;
    }

    sprintf(line, " %s:", priv_info.bootstrap_netifname);
    fn_table.to_string(&priv_info.bootstrap_netifaddr, line + strlen(line), AF_NUMERICFORM_DEFAULT);
    BOOTSTRAP_DEBUG_PRINT("UID bootstrap network using: %s\n", line);

    pthread_mutex_unlock(&priv_info.bootstrap_netlock);
    return BOOTSTRAP_SUCCESS;
}

/**
 * Bootstrap Initialization Exchange Protocol i.e "Phoning Home Protocol"
 * All ranks phone home by sending their netaddr and rank info to root
 * Root rank advertises every rank's ring right peer netaddr info
 */
static void* bootstrap_root(void* rargs) {
    struct bootstrap_root_args* args = (struct bootstrap_root_args*)rargs;
    int peer_uid_version = 0, root_uid_version;
    bootstrap_uid_socket_t* listen_sock = args->listen_sock;
    root_uid_version = args->version;
    uint64_t magic = args->magic;
    bootstrap_result_t res = BOOTSTRAP_SUCCESS;
    int nranks = 0, c = 0;
    struct bootstrap_ext_info info;
    bootstrap_uid_socket_address_t* rank_addresses = NULL;
    bootstrap_uid_socket_address_t* rank_addresses_root =
        NULL;  // for initial rank <-> root information exchange
    bootstrap_uid_socket_address_t* zero = NULL;
    BOOTSTRAP_CHECKGOTO(BOOTSTRAP_CALLOC(&zero, 1), res, out);
    BOOTSTRAP_CHECKGOTO(set_files_limit(), res, out);

    /* Receive addresses from all ranks */
    do {
        bootstrap_uid_socket_t sock;
        BOOTSTRAP_CHECKGOTO(
            nccl_fn_table(init, &sock, nullptr, SOCKET_MAGIC, SOCKET_TYPE_UNKNOWN, nullptr, 0), res,
            out);
        BOOTSTRAP_CHECKGOTO(nccl_fn_table(accept, &sock, listen_sock), res, out);
        // check for wire compatibility for nvshmemx_uniqueid_t
        BOOTSTRAP_CHECKGOTO(
            bootstrap_net_recv(&sock, &(peer_uid_version), sizeof(peer_uid_version)), res, out);
        BOOTSTRAP_CHECKGOTO(
            bootstrap_net_send(&sock, &(root_uid_version), sizeof(root_uid_version)), res, out);
        if (peer_uid_version != root_uid_version) {
            BOOTSTRAP_ERROR_PRINT("UID Bootstrap versions not compatible between PEs\n");
            goto out;
        }

        BOOTSTRAP_CHECKGOTO(bootstrap_net_recv(&sock, &info, sizeof(info)), res, out);
        BOOTSTRAP_CHECKGOTO(nccl_fn_table(close, &sock), res, out);

        if (c == 0) {
            nranks = info.nranks;
            BOOTSTRAP_CHECKGOTO(BOOTSTRAP_CALLOC(&rank_addresses, nranks), res, out);
            BOOTSTRAP_CHECKGOTO(BOOTSTRAP_CALLOC(&rank_addresses_root, nranks), res, out);
        }

        if (nranks != info.nranks) {
            BOOTSTRAP_ERROR_PRINT("mismatch in rank count from procs %d : %d", nranks, info.nranks);
            goto out;
        }

        if (memcmp(zero, &rank_addresses_root[info.rank], sizeof(bootstrap_uid_socket_address_t)) !=
            0) {
            BOOTSTRAP_ERROR_PRINT("rank %d of %d ranks has already checked in", info.rank, nranks);
            goto out;
        }

        // Save the connection handle for that rank
        memcpy(rank_addresses_root + info.rank, &info.ext_address_listen_root,
               sizeof(bootstrap_uid_socket_address_t));
        memcpy(rank_addresses + info.rank, &info.ext_address_listen,
               sizeof(bootstrap_uid_socket_address_t));

        ++c;
        BOOTSTRAP_DEBUG_PRINT("Received connect from rank %d total %d/%d", info.rank, c, nranks);
    } while (c < nranks);
    BOOTSTRAP_DEBUG_PRINT("COLLECTED ALL %d HANDLES", nranks);

    // Send the connect handle for the next rank in the _allgather ring
    for (int r = 0; r < nranks; ++r) {
        int next = (r + 1) % nranks;
        bootstrap_uid_socket_t sock;
        BOOTSTRAP_CHECKGOTO(nccl_fn_table(init, &sock, rank_addresses_root + r, magic,
                                          SOCKET_TYPE_BOOTSTRAP, nullptr, 0),
                            res, out);
        BOOTSTRAP_CHECKGOTO(nccl_fn_table(connect, &sock), res, out);
        BOOTSTRAP_CHECKGOTO(bootstrap_net_send(&sock, rank_addresses + next,
                                               sizeof(bootstrap_uid_socket_address_t)),
                            res, out);
        BOOTSTRAP_CHECKGOTO(nccl_fn_table(close, &sock), res, out);
    }

    BOOTSTRAP_DEBUG_PRINT("SENT OUT ALL %d HANDLES", nranks);

out:
    if (listen_sock != NULL) {
        BOOTSTRAP_INFO(nccl_fn_table(close, listen_sock));
        BOOTSTRAP_PTR_FREE(listen_sock);
    }

    BOOTSTRAP_PTR_FREE(rank_addresses);
    BOOTSTRAP_PTR_FREE(rank_addresses_root);
    BOOTSTRAP_PTR_FREE(zero);
    BOOTSTRAP_PTR_FREE(rargs);
    BOOTSTRAP_DEBUG_PRINT("DONE");
    return NULL;
}

/**
 * Bootstrap Root Thread Creation
 */
static bootstrap_result_t bootstrap_create_root(struct bootstrap_uid_handle* handle) {
    bootstrap_uid_socket_t* listen_sock;
    struct bootstrap_root_args* args;
    pthread_attr_t attr = {};
    bootstrap_result_t res = BOOTSTRAP_SUCCESS;

    BOOTSTRAP_CHECKGOTO(BOOTSTRAP_CALLOC(&listen_sock, 1), res, out);
    BOOTSTRAP_CHECK(nccl_fn_table(init, listen_sock, &handle->addr, handle->magic,
                                  SOCKET_TYPE_BOOTSTRAP, nullptr, 0));
    BOOTSTRAP_CHECK(nccl_fn_table(listen, listen_sock));
    BOOTSTRAP_CHECK(nccl_fn_table(get_addr, listen_sock, &handle->addr));

    BOOTSTRAP_CHECKGOTO(BOOTSTRAP_CALLOC(&args, 1), res, cleanup_old);
    args->listen_sock = listen_sock;
    args->magic = handle->magic;
    args->version = handle->version;

    BOOTSTRAP_NEQCHECK(pthread_attr_init(&attr), 0);
    BOOTSTRAP_NEQCHECK(pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED), 0);
    BOOTSTRAP_NEQCHECK(
        pthread_create(&(priv_info.bootstrap_root), &attr, bootstrap_root, (void*)args), 0);

    return (BOOTSTRAP_SUCCESS);

cleanup_old:
    BOOTSTRAP_PTR_FREE(listen_sock);
out:
    return (res);
}

/**
 * Bootstrap Unique ID Setup
 */
int bootstrap_get_unique_id(void* cookie) {
    struct bootstrap_uid_handle* handle = (struct bootstrap_uid_handle*)(cookie);
    *handle = NVSHMEMX_UNIQUEID_INITIALIZER;
    BOOTSTRAP_NEQCHECK(bootstrap_get_random_data(&handle->magic, sizeof(handle->magic)), 0);
    if (env_attr.BOOTSTRAP_UID_SESSION_ID_provided) {
        if (nccl_fn_table(get_addr_from_string, &handle->addr, env_attr.BOOTSTRAP_UID_SESSION_ID) !=
            BOOTSTRAP_SUCCESS) {
            BOOTSTRAP_ERROR_PRINT(
                "Invalid UID Session ID, please use format: <ipv4>:<port> or "
                "[<ipv6>]:<port> or <hostname>:<port>");
            return BOOTSTRAP_INVALID_ARGUMENT;
        }
    } else {
        memcpy(&(handle->addr), &priv_info.bootstrap_netifaddr,
               sizeof(bootstrap_uid_socket_address_t));
        BOOTSTRAP_CHECK(bootstrap_create_root(handle));
    }

    return BOOTSTRAP_SUCCESS;
}

// Unexpected recv/send incase of tag-matching
static bootstrap_result_t unexpected_enqueue(struct bootstrap_state* state, int peer, int tag,
                                             bootstrap_uid_socket_t* sock) {
    // New unex
    struct unex_conn* unex;
    bootstrap_result_t ret = BOOTSTRAP_SUCCESS;
    struct unex_conn* list = state->unexpected_connections;
    BOOTSTRAP_CHECKGOTO(BOOTSTRAP_CALLOC(&unex, 1), ret, outfp);
    unex->peer = peer;
    unex->tag = tag;
    memcpy(&unex->sock, sock, sizeof(bootstrap_uid_socket_t));

    // Enqueue
    if (list == NULL) {
        state->unexpected_connections = unex;
        return (ret);
    }

    while (list->next) {
        list = list->next;
    }

    list->next = unex;

outfp:
    return (ret);
}

static bootstrap_result_t unexpected_dequeue(struct bootstrap_state* state, int peer, int tag,
                                             bootstrap_uid_socket_t* sock, int* found) {
    struct unex_conn* elem = state->unexpected_connections;
    struct unex_conn* prev = NULL;
    *found = 0;
    while (elem) {
        if (elem->peer == peer && elem->tag == tag) {
            if (prev == NULL) {
                state->unexpected_connections = elem->next;
            } else {
                prev->next = elem->next;
            }

            memcpy(sock, &elem->sock, sizeof(bootstrap_uid_socket_t));
            BOOTSTRAP_PTR_FREE(elem);
            *found = 1;
            return BOOTSTRAP_SUCCESS;
        }

        prev = elem;
        elem = elem->next;
    }

    return BOOTSTRAP_SUCCESS;
}

static void unexpected_free(struct bootstrap_state* state) {
    struct unex_conn* elem = state->unexpected_connections;
    struct unex_conn* prev = NULL;

    while (elem) {
        prev = elem;
        elem = elem->next;
        BOOTSTRAP_PTR_FREE(prev);
    }

    return;
}

/**
 * P2P Communication Ops
 */
static bootstrap_result_t bootstrap_send(void* comm_state, int peer, int tag, void* data,
                                         int size) {
    bootstrap_result_t ret = BOOTSTRAP_SUCCESS;
    struct bootstrap_state* state = (struct bootstrap_state*)comm_state;
    bootstrap_uid_socket_t sock;

    BOOTSTRAP_CHECKGOTO(nccl_fn_table(init, &sock, state->peer_comm_addresses + peer, state->magic,
                                      SOCKET_TYPE_BOOTSTRAP, nullptr, 0),
                        ret, fail);
    BOOTSTRAP_CHECKGOTO(nccl_fn_table(connect, &sock), ret, fail);
    BOOTSTRAP_CHECKGOTO(bootstrap_net_send(&sock, &state->rank, sizeof(int)), ret, fail);
    BOOTSTRAP_CHECKGOTO(bootstrap_net_send(&sock, &tag, sizeof(int)), ret, fail);
    BOOTSTRAP_CHECKGOTO(bootstrap_net_send(&sock, data, size), ret, fail);

exit:
    BOOTSTRAP_CHECK(nccl_fn_table(close, &sock));
    return ret;
fail:
    goto exit;
}

// We can't know who we'll receive from, so we need to receive everything at once
static bootstrap_result_t bootstrap_recv(void* comm_state, int peer, int tag, void* data,
                                         int size) {
    bootstrap_result_t ret = BOOTSTRAP_SUCCESS;
    struct bootstrap_state* state = (struct bootstrap_state*)comm_state;
    bootstrap_uid_socket_t sock;
    int new_peer, new_tag;

    // Search unexpected connections first
    int found;
    BOOTSTRAP_CHECK(unexpected_dequeue(state, peer, tag, &sock, &found));
    if (found) {
        BOOTSTRAP_CHECKGOTO(bootstrap_net_recv(&sock, ((char*)data), size), ret, fail);
        goto exit;
    }

    // Then look for new connections
    while (1) {
        BOOTSTRAP_CHECKGOTO(
            nccl_fn_table(init, &sock, nullptr, SOCKET_MAGIC, SOCKET_TYPE_UNKNOWN, nullptr, 0), ret,
            fail);
        BOOTSTRAP_CHECKGOTO(nccl_fn_table(accept, &sock, &state->listen_sock), ret, fail);
        BOOTSTRAP_CHECKGOTO(bootstrap_net_recv(&sock, &new_peer, sizeof(int)), ret, fail);
        BOOTSTRAP_CHECKGOTO(bootstrap_net_recv(&sock, &new_tag, sizeof(int)), ret, fail);
        if (new_peer == peer && new_tag == tag) {
            BOOTSTRAP_CHECKGOTO(bootstrap_net_recv(&sock, ((char*)data), size), ret, fail);
            goto exit;
        }
        // Unexpected connection. Save for later.
        BOOTSTRAP_CHECKGOTO(unexpected_enqueue(state, new_peer, new_tag, &sock), ret, fail);
    }
exit:
    BOOTSTRAP_CHECK(nccl_fn_table(close, &sock));
    return ret;
fail:
    goto exit;
}

/**
 * Collective Ops
 */
int bootstrap_uid_allgather(const void* send_data, void* recv_data, int size,
                            struct bootstrap_handle* handle) {
    struct bootstrap_state* state = (struct bootstrap_state*)(handle->comm_state);
    int rank = state->rank;
    int nranks = state->nranks;

    BOOTSTRAP_DEBUG_PRINT("rank %d nranks %d size %d", rank, nranks, size);
    char* send_buf = (char*)send_data;
    if (send_data != BOOTSTRAP_IN_PLACE) {
        // As not an inplace operation - copy send_data to recv_data for myrank
        memcpy((char*)recv_data + (rank % nranks) * size, send_buf, size);
    }

    /* Simple ring based _allgather
     * At each step i receive data from (rank-i-1) from left
     * and send previous step's data from (rank-i) to right
     */
    for (int i = 0; i < nranks - 1; i++) {
        size_t rslice = (rank - i - 1 + nranks) % nranks;
        size_t sslice = (rank - i + nranks) % nranks;

        // Send slice to the right
        BOOTSTRAP_CHECK(
            bootstrap_net_send(&state->ring_send_socket, ((char*)recv_data + sslice * size), size));
        // Recv slice from the left
        BOOTSTRAP_CHECK(
            bootstrap_net_recv(&state->ring_recv_socket, ((char*)recv_data + rslice * size), size));
    }

    BOOTSTRAP_DEBUG_PRINT("rank %d nranks %d size %d - DONE", rank, nranks, size);
    return BOOTSTRAP_SUCCESS;
}

int bootstrap_uid_alltoall(const void* send_data, void* recv_data, int size,
                           struct bootstrap_handle* handle) {
    struct bootstrap_state* state = (struct bootstrap_state*)(handle->comm_state);
    char* send_buf = (char*)send_data;
    int rank = state->rank;
    int nranks = state->nranks;
    int tag = 0;
    int chunk_size = size;

    BOOTSTRAP_DEBUG_PRINT("rank %d nranks %d size %d", rank, nranks, size);
    /* TODO: add inplace support for alltoall */
    if (send_data == BOOTSTRAP_IN_PLACE) {
        BOOTSTRAP_ERROR_PRINT("Unsupported inplace operation\n");
        return (BOOTSTRAP_INVALID_USAGE);
    }

    /* Simple ring based _alltoall
     * receive jth data from process i
     * send ith data to process j
     */
    for (int i = 0; i < nranks; i++) {
        size_t left = (rank - i + nranks) % nranks;
        size_t right = (rank + i) % nranks;

        if (right == (size_t)rank && left == (size_t)rank) {
            memcpy(((char*)recv_data + left * chunk_size), (send_buf + right * chunk_size),
                   chunk_size);
            continue;
        }

        tag++;
        // Send slice to the right
        BOOTSTRAP_CHECK(bootstrap_send(handle->comm_state, right, tag,
                                       (send_buf + right * chunk_size), chunk_size));
        // Recv slice from the left
        BOOTSTRAP_CHECK(bootstrap_recv(handle->comm_state, left, tag,
                                       ((char*)recv_data + left * chunk_size), chunk_size));
    }

    BOOTSTRAP_DEBUG_PRINT("rank %d nranks %d size %d - DONE", rank, nranks, size);
    return BOOTSTRAP_SUCCESS;
}

int bootstrap_uid_barrier(struct bootstrap_handle* handle) {
    struct bootstrap_state* state = (struct bootstrap_state*)(handle->comm_state);
    int rank = state->rank;
    int tag = 0;
    int nranks = state->nranks;
    if (nranks == 1) {
        return BOOTSTRAP_SUCCESS;
    }

    BOOTSTRAP_DEBUG_PRINT("rank %d nranks %d tag %x - ENTER", rank, nranks, tag);

    /* Simple intra process barrier
     * Based on the dissemination algorithm by Debra Hensgen, Raphael Finkel, and Udi Manbet,
     * "Two Algorithms for barrier Synchronization," International Journal of Parallel Programming,
     * 17(1):1-17, 1988"
     */
    int data[1];
    for (int mask = 1; mask < nranks; mask <<= 1) {
        int src = (rank - mask + nranks) % nranks;
        int dst = (rank + mask) % nranks;
        tag++;
        BOOTSTRAP_CHECK(bootstrap_send(handle->comm_state, dst, tag, data, sizeof(data)));
        BOOTSTRAP_CHECK(bootstrap_recv(handle->comm_state, src, tag, data, sizeof(data)));
    }

    BOOTSTRAP_DEBUG_PRINT("rank %d nranks %d tag %x - DONE", rank, nranks, tag);
    return BOOTSTRAP_SUCCESS;
}

/**
 * Bootstrap Finalization & Abort operations
 */
int bootstrap_uid_close(struct bootstrap_handle* handle) {
    struct bootstrap_state* state = (struct bootstrap_state*)(handle->comm_state);
    if (state->unexpected_connections != NULL) {
        unexpected_free(state);
        if (state->abort_flag && *(state->abort_flag) == 0) {
            BOOTSTRAP_ERROR_PRINT("Unexpected connections are not empty");
            return BOOTSTRAP_INTERNAL_ERROR;
        }
    }

    BOOTSTRAP_CHECK(nccl_fn_table(close, &state->listen_sock));
    BOOTSTRAP_CHECK(nccl_fn_table(close, &state->ring_send_socket));
    BOOTSTRAP_CHECK(nccl_fn_table(close, &state->ring_recv_socket));
    BOOTSTRAP_PTR_FREE(state->peer_comm_addresses);
    BOOTSTRAP_PTR_FREE(state);
    return BOOTSTRAP_SUCCESS;
}

bootstrap_result_t bootstrap_uid_abort(struct bootstrap_handle* handle) {
    struct bootstrap_state* state = (struct bootstrap_state*)(handle->comm_state);
    if (state == NULL) {
        return BOOTSTRAP_SUCCESS;
    }

    BOOTSTRAP_CHECK(nccl_fn_table(close, &state->listen_sock));
    BOOTSTRAP_CHECK(nccl_fn_table(close, &state->ring_send_socket));
    BOOTSTRAP_CHECK(nccl_fn_table(close, &state->ring_recv_socket));
    BOOTSTRAP_PTR_FREE(state->peer_comm_addresses);
    BOOTSTRAP_PTR_FREE(state);
    return BOOTSTRAP_SUCCESS;
}

/**
 * Bootstrap Pre & Post Initialization
 */
int nvshmemi_bootstrap_plugin_pre_init(bootstrap_handle_t* handle, const int abi_version) {
    int bootstrap_version = NVSHMEMI_BOOTSTRAP_ABI_VERSION;
    if (!nvshmemi_is_bootstrap_compatible(bootstrap_version, abi_version, true)) {
        BOOTSTRAP_ERROR_PRINT(
            "UID bootstrap version (%d) is not compatible with NVSHMEM version (%d)",
            bootstrap_version, abi_version);
        exit(-1);
    }

    // parse all envs and initialize them inside of bootstrap library
    nvshmemi_env_options_init(&env_attr);
    bootstrap_parse_debug();

    // Discover the network for bootstrap, if not done previously.
    // This code needs to be stateful to be able to be called multiple times by the caller
    BOOTSTRAP_CHECK(bootstrap_net_init());
    if (handle->pre_init_ops == nullptr) {
        BOOTSTRAP_CALLOC(&handle->pre_init_ops, 1);
        handle->pre_init_ops->get_unique_id = bootstrap_get_unique_id;
        handle->pre_init_ops->cookie = nullptr;
    }

    return 0;
}

int nvshmemi_bootstrap_plugin_init(void* arg, bootstrap_handle_t* handle, const int abi_version) {
    struct bootstrap_state* state;
    int root_uid_version;
    nvshmemx_uniqueid_args_t* uid_args = nullptr;
    bootstrap_uid_socket_address_t next_addr;
    bootstrap_uid_socket_t sock, listen_sock_root;
    bootstrap_uid_handle* uid_handle = nullptr;
    struct bootstrap_ext_info info = {};

    int bootstrap_version = NVSHMEMI_BOOTSTRAP_ABI_VERSION;
    if (!nvshmemi_is_bootstrap_compatible(bootstrap_version, abi_version, true)) {
        BOOTSTRAP_ERROR_PRINT(
            "UID bootstrap version (%d) is not compatible with NVSHMEM version (%d)",
            bootstrap_version, abi_version);
        exit(-1);
    }

    // interpret arg as nvshmemx_uniqueid_args_t
    uid_args = (nvshmemx_uniqueid_args_t*)(arg);
    handle->pg_rank = (uid_args->myrank);
    handle->pg_size = (uid_args->nranks);
    bootstrap_init_ops_t* ops = handle->pre_init_ops;
    if (ops == nullptr) {
        BOOTSTRAP_ERROR_PRINT("UID Bootstrap Init Failed due to prior pre init failures\n");
        exit(-1);
    }

    // for non-root ranks, inherit the uid_handle from caller
    if (ops->cookie == nullptr) {
        BOOTSTRAP_CALLOC(&uid_handle, 1);
        memcpy(uid_handle, uid_args->id, sizeof(bootstrap_uid_handle));
        ops->cookie = uid_handle;
        // if session ID was set and rank is 0, create a root thread to exchange peer addresses
        // using phoning home protocol
        if (handle->pg_rank == 0 && env_attr.BOOTSTRAP_UID_SESSION_ID_provided) {
            BOOTSTRAP_CHECK(bootstrap_create_root(uid_handle));
        }
    } else {
        uid_handle = (struct bootstrap_uid_handle*)(ops->cookie);
    }

    BOOTSTRAP_CHECK(BOOTSTRAP_CALLOC(&state, 1));
    state->abort_flag =
        nullptr;  // TODO: add support for aborting bootstrap asynchronously in the future
    state->rank = handle->pg_rank;
    state->nranks = handle->pg_size;
    handle->comm_state = (void*)state;  // save the bootstrap state per handle
    state->magic = uid_handle->magic;

    BOOTSTRAP_DEBUG_PRINT("rank %d nranks %d", handle->pg_rank, handle->pg_size);

    info.rank = handle->pg_rank;
    info.nranks = handle->pg_size;
    // Create socket for other ranks to contact me
    BOOTSTRAP_CHECK(nccl_fn_table(init, &state->listen_sock, &priv_info.bootstrap_netifaddr,
                                  state->magic, SOCKET_TYPE_BOOTSTRAP, state->abort_flag, 0));
    BOOTSTRAP_CHECK(nccl_fn_table(listen, &state->listen_sock));
    BOOTSTRAP_CHECK(nccl_fn_table(get_addr, &state->listen_sock, &info.ext_address_listen));

    // Create socket for root to contact me
    BOOTSTRAP_CHECK(nccl_fn_table(init, &listen_sock_root, &priv_info.bootstrap_netifaddr,
                                  state->magic, SOCKET_TYPE_BOOTSTRAP, state->abort_flag, 0));
    BOOTSTRAP_CHECK(nccl_fn_table(listen, &listen_sock_root));
    BOOTSTRAP_CHECK(nccl_fn_table(get_addr, &listen_sock_root, &info.ext_address_listen_root));

    // stagger connection times to avoid an overload of the root
    if (handle->pg_size > 128) {
        long msec = handle->pg_rank;
        struct timespec tv;
        tv.tv_sec = msec / 1000;
        tv.tv_nsec = 1000000 * (msec % 1000);
        BOOTSTRAP_DEBUG_PRINT("rank %d delaying connection to root by %ld msec", handle->pg_rank,
                              msec);
        (void)nanosleep(&tv, NULL);
    }

    // send info on my listening socket to root
    BOOTSTRAP_CHECK(nccl_fn_table(init, &sock, &uid_handle->addr, state->magic,
                                  SOCKET_TYPE_BOOTSTRAP, state->abort_flag, 0));
    BOOTSTRAP_CHECK(nccl_fn_table(connect, &sock));
    // check for wire compatibility for nvshmemx_uniqueid_t
    BOOTSTRAP_CHECK(bootstrap_net_send(&sock, &(uid_handle->version), sizeof(uid_handle->version)));
    BOOTSTRAP_CHECK(bootstrap_net_recv(&sock, &(root_uid_version), sizeof(root_uid_version)));
    if (uid_handle->version != root_uid_version) {
        BOOTSTRAP_ERROR_PRINT("UID Bootstrap versions not compatible between PEs\n");
        exit(-1);
    }

    BOOTSTRAP_CHECK(bootstrap_net_send(&sock, &info, sizeof(info)));
    BOOTSTRAP_CHECK(nccl_fn_table(close, &sock));

    // get info on my "next" rank in the bootstrap ring from root
    BOOTSTRAP_CHECK(
        nccl_fn_table(init, &sock, nullptr, SOCKET_MAGIC, SOCKET_TYPE_UNKNOWN, nullptr, 0));
    BOOTSTRAP_CHECK(nccl_fn_table(accept, &sock, &listen_sock_root));
    BOOTSTRAP_CHECK(bootstrap_net_recv(&sock, &next_addr, sizeof(bootstrap_uid_socket_address_t)));
    BOOTSTRAP_CHECK(nccl_fn_table(close, &sock));
    BOOTSTRAP_CHECK(nccl_fn_table(close, &listen_sock_root));

    BOOTSTRAP_CHECK(nccl_fn_table(init, &state->ring_send_socket, &next_addr, state->magic,
                                  SOCKET_TYPE_BOOTSTRAP, state->abort_flag, 0));
    BOOTSTRAP_CHECK(nccl_fn_table(connect, &state->ring_send_socket));
    // accept the connect request from the previous rank in the _allgather ring
    BOOTSTRAP_CHECK(nccl_fn_table(init, &state->ring_recv_socket, nullptr, SOCKET_MAGIC,
                                  SOCKET_TYPE_UNKNOWN, nullptr, 0));
    BOOTSTRAP_CHECK(nccl_fn_table(accept, &state->ring_recv_socket, &state->listen_sock));

    // _allgather all listen handlers
    BOOTSTRAP_CHECK(BOOTSTRAP_CALLOC(&state->peer_comm_addresses, handle->pg_size));
    BOOTSTRAP_CHECK(
        nccl_fn_table(get_addr, &state->listen_sock, state->peer_comm_addresses + handle->pg_rank));
    BOOTSTRAP_NEQCHECK(bootstrap_uid_allgather(BOOTSTRAP_IN_PLACE, state->peer_comm_addresses,
                                               sizeof(bootstrap_uid_socket_address_t), handle),
                       BOOTSTRAP_SUCCESS);

    handle->version = NVSHMEM_BOOTSTRAP_MAJOR_MINOR_VERSION(abi_version) <
                              NVSHMEM_BOOTSTRAP_MAJOR_MINOR_VERSION(bootstrap_version)
                          ? abi_version
                          : bootstrap_version;
    handle->allgather = bootstrap_uid_allgather;
    handle->alltoall = bootstrap_uid_alltoall;
    handle->barrier = bootstrap_uid_barrier;
    handle->finalize = bootstrap_uid_close;
    handle->global_exit = nullptr;
    handle->show_info = bootstrap_uid_showinfo;
    // TODO: Add future support for how to enable proxy thread for abort & progress tracking
    BOOTSTRAP_DEBUG_PRINT("rank %d nranks %d - DONE", handle->pg_rank, handle->pg_size);
    return BOOTSTRAP_SUCCESS;
}
