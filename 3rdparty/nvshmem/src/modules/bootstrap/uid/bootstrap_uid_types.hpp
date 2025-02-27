/*
 * Copyright (c) 2017-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef BOOTSTRAP_UID_HPP
#define BOOTSTRAP_UID_HPP

#include <pthread.h>                         // for PTHREAD_MUTEX_INITIALIZER
#include <stdint.h>                          // for uint64_t, uint32_t
#include <stdio.h>                           // for fclose, fopen, fread
#include <stdlib.h>                          // for malloc
#include <cstring>                           // for NULL, memset, size_t
#include "bootstrap_uid_remap.h"             // for bootstrap_uid_socket_a...
#include "bootstrap_util.h"                  // for BOOTSTRAP_ERROR_PRINT
#include "ncclSocket/ncclsocket_socket.hpp"  // for MAX_IF_NAME_SIZE

template <typename T>
inline bootstrap_result_t bootstrap_calloc_debug(T** ptr, size_t nelem, const char* filefunc,
                                                 int line) {
    void* p = malloc(nelem * sizeof(T));
    if (p == NULL) {
        BOOTSTRAP_ERROR_PRINT("Unable to malloc %ld bytes", nelem * sizeof(T));
        return BOOTSTRAP_INTERNAL_ERROR;
    }
    // BOOTSTRAP_DEBUG_PRINT("%s:%d malloc Size %ld pointer %p", filefunc, line, nelem*sizeof(T),
    // p);
    memset(p, 0, nelem * sizeof(T));
    *ptr = (T*)p;
    return BOOTSTRAP_SUCCESS;
}

#define BOOTSTRAP_CALLOC(...) bootstrap_calloc_debug(__VA_ARGS__, __FILE__, __LINE__)

/* Socket Root thread fn arguments */
struct bootstrap_root_args {
    bootstrap_uid_socket_t* listen_sock;
    uint64_t magic;
    int version;
};

/* Socket External PEs address connection info */
struct bootstrap_ext_info {
    int rank;
    int nranks;
    bootstrap_uid_socket_address_t ext_address_listen_root;
    bootstrap_uid_socket_address_t ext_address_listen;
};

/* Internal UID implementation */
struct bootstrap_uid_handle {
    int version;
    bootstrap_uid_socket_address_t addr;
    uint64_t magic;
};

/* Socket Net state: name, addr, status, lock */
struct bootstrap_netstate {
    char bootstrap_netifname[MAX_IF_NAME_SIZE + 1];     /* Socket Interface Name */
    bootstrap_uid_socket_address_t bootstrap_netifaddr; /* Socket Interface Address */
    int bootstrap_netinitdone = 0;                      /* Socket Interface Init Status */
    pthread_mutex_t bootstrap_netlock = PTHREAD_MUTEX_INITIALIZER; /* Socket Interface Lock */
    pthread_t bootstrap_root; /* Socket Root Thread for phoning root to non-root peers */
};

/* Socket Interface Selection type */
enum bootstrapInterface_t { findSubnetIf = -1, dontCareIf = -2 };

struct unex_conn {
    int peer;
    int tag;
    bootstrap_uid_socket_t sock;
    struct unex_conn* next;
};

struct bootstrap_state {
    bootstrap_uid_socket_t listen_sock;
    bootstrap_uid_socket_t ring_recv_socket;
    bootstrap_uid_socket_t ring_send_socket;
    bootstrap_uid_socket_address_t* peer_comm_addresses;
    struct unex_conn* unexpected_connections;
    int rank;
    int nranks;
    uint64_t magic;
    volatile uint32_t* abort_flag;
};

/* get any bytes of random data from /dev/urandom, return 0 if it succeeds; else
 * return -1 */
inline int bootstrap_get_random_data(void* buffer, size_t bytes) {
    int ret = 0;
    if (bytes > 0) {
        const size_t one = 1UL;
        FILE* fp = fopen("/dev/urandom", "r");
        if (buffer == NULL || fp == NULL || fread(buffer, bytes, one, fp) != one) {
            ret = -1;
        }

        if (fp) {
            fclose(fp);
        }
    }

    return ret;
}

#endif /* BOOTSTRAP_UID_HPP */
