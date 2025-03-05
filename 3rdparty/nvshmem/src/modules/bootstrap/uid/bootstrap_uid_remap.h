/*
 * Copyright (c) 2017-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef NVSHMEM_BOOTSTRAP_UID_REMAP_H
#define NVSHMEM_BOOTSTRAP_UID_REMAP_H

#include "ncclsocket_socket.hpp"
#include "ncclsocket_nccl.h"
#include "bootstrap_util.h"
#include <stdint.h>

/* Types inherited from ncclSocket.h */
typedef union ncclSocketAddress bootstrap_uid_socket_address_t;
typedef enum ncclSocketState bootstrap_uid_socket_state_t;
typedef enum ncclSocketType bootstrap_uid_socket_type_t;
typedef struct ncclSocket bootstrap_uid_socket_t;

#define SOCKET_TYPE_BOOTSTRAP ncclSocketTypeBootstrap
#define SOCKET_TYPE_UNKNOWN ncclSocketTypeUnknown
#define SOCKET_MAGIC NCCL_SOCKET_MAGIC
#define BOOTSTRAP_UID_SOCKET_SEND NCCL_SOCKET_SEND;
#define BOOTSTRAP_UID_SOCKET_RECV NCCL_SOCKET_RECV;

inline bootstrap_result_t nccl_to_bootstrap_result(ncclResult_t ret) {
    switch (ret) {
        case ncclSuccess:
            return BOOTSTRAP_SUCCESS;
        case ncclUnhandledCudaError:
            return BOOTSTRAP_UNHANDLED_CUDA_ERROR;
        case ncclSystemError:
            return BOOTSTRAP_SYSTEM_ERROR;
        case ncclInternalError:
            return BOOTSTRAP_INTERNAL_ERROR;
        case ncclInvalidArgument:
            return BOOTSTRAP_INVALID_ARGUMENT;
        case ncclInvalidUsage:
            return BOOTSTRAP_INVALID_USAGE;
        case ncclRemoteError:
            return BOOTSTRAP_REMOTE_ERROR;
        case ncclInProgress:
            return BOOTSTRAP_INPROGRESS;
        case ncclNumResults:
            return BOOTSTRAP_NUM_RESULTS;
        default:
            return (BOOTSTRAP_ERROR_MAX);
    }
}

typedef struct bootstrap_uid_socket_fn {
    const char* (*to_string)(bootstrap_uid_socket_address_t* addr, char* buf,
                             const int numericHostForm);
    ncclResult_t (*get_addr_from_string)(bootstrap_uid_socket_address_t* ua,
                                         const char* ip_port_pair);
    int (*find_interface_match_subnet)(char* ifNames, bootstrap_uid_socket_address_t* localAddrs,
                                       bootstrap_uid_socket_address_t* remoteAddr,
                                       int ifNameMaxSize, int maxIfs);
    int (*find_interfaces)(char* ifNames, bootstrap_uid_socket_address_t* ifAddrs,
                           int ifNameMaxSize, int maxIfs, char* envSockIfName,
                           char* envSockIfFamily, char* envUidSessionId);
    ncclResult_t (*init)(bootstrap_uid_socket_t* sock, bootstrap_uid_socket_address_t* addr,
                         uint64_t magic, bootstrap_uid_socket_type_t type,
                         volatile uint32_t* abortFlag, int asyncFlag);
    ncclResult_t (*listen)(bootstrap_uid_socket_t* sock);
    ncclResult_t (*get_addr)(bootstrap_uid_socket_t* sock, bootstrap_uid_socket_address_t* addr);
    ncclResult_t (*connect)(bootstrap_uid_socket_t* sock);
    ncclResult_t (*accept)(bootstrap_uid_socket_t* sock, bootstrap_uid_socket_t* listen_sock);
    ncclResult_t (*get_fd)(bootstrap_uid_socket_t* sock, int* fd);
    ncclResult_t (*set_fd)(int fd, bootstrap_uid_socket_t* sock);
    ncclResult_t (*progress)(int op, bootstrap_uid_socket_t* sock, void* ptr, int size,
                             int* offset);
    ncclResult_t (*wait)(int op, bootstrap_uid_socket_t* sock, void* ptr, int size, int* offset);
    ncclResult_t (*send)(bootstrap_uid_socket_t* sock, void* ptr, int size);
    ncclResult_t (*recv)(bootstrap_uid_socket_t* sock, void* ptr, int size);
    ncclResult_t (*tryrecv)(bootstrap_uid_socket_t* sock, void* ptr, int size, int* closed,
                            bool blocking);
    ncclResult_t (*close)(bootstrap_uid_socket_t* sock);
} bootstrap_uid_socket_fn_t;

const bootstrap_uid_socket_fn_t fn_table = {
    .to_string = ncclSocketToString,
    .get_addr_from_string = ncclSocketGetAddrFromString,
    .find_interface_match_subnet = ncclFindInterfaceMatchSubnet,
    .find_interfaces = ncclFindInterfaces,
    .init = ncclSocketInit,
    .listen = ncclSocketListen,
    .get_addr = ncclSocketGetAddr,
    .connect = ncclSocketConnect,
    .accept = ncclSocketAccept,
    .get_fd = ncclSocketGetFd,
    .set_fd = ncclSocketSetFd,
    .progress = ncclSocketProgress,
    .wait = ncclSocketWait,
    .send = ncclSocketSend,
    .recv = ncclSocketRecv,
    .tryrecv = ncclSocketTryRecv,
    .close = ncclSocketClose
    /* New to nvshmem bootstrap uid */
};

#define nccl_fn_table(func, ...) nccl_to_bootstrap_result(fn_table.func(__VA_ARGS__))

#endif /*! NVSHMEM_BOOTSTRAP_UID_REMAP_H */
