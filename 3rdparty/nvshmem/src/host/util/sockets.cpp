/*
 * Copyright (c) 2016-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "internal/host/sockets.h"
#include <sys/uio.h>  // IWYU pragma: keep
// IWYU pragma: no_include <bits/types/struct_iovec.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#define NVSHMEM_SOCKET_MAX_LEN 52ULL

int ipcOpenSocket(ipcHandle *&handle, pid_t send_process, pid_t recv_process) {
    int sock = 0;
    struct sockaddr_un cliaddr;

    handle = new ipcHandle;

    memset(handle, 0, sizeof(*handle));
    handle->socketName = new char[NVSHMEM_SOCKET_MAX_LEN];

    if ((sock = socket(AF_UNIX, SOCK_DGRAM, 0)) < 0) {
        perror("IPC failure:Socket creation error");
        delete handle->socketName;
        delete handle;
        return -1;
    }

    bzero(&cliaddr, sizeof(cliaddr));
    cliaddr.sun_family = AF_UNIX;
    char temp[NVSHMEM_SOCKET_MAX_LEN] = {0};

    // Create unique name for the socket.
    int name_len = snprintf(temp, NVSHMEM_SOCKET_MAX_LEN, "/tmp/nvshmem-socket-%u-%u", send_process,
                            recv_process);
    if (name_len < 0 || static_cast<unsigned long long>(name_len) >= NVSHMEM_SOCKET_MAX_LEN) {
        printf("Error formatting socket file name\n");
        goto out_error;
    }

    temp[strlen(temp)] = '\0';
    strncpy(cliaddr.sun_path, temp, NVSHMEM_SOCKET_MAX_LEN);
    if (bind(sock, (struct sockaddr *)&cliaddr, sizeof(cliaddr)) < 0) {
        perror(
            "IPC failure: Binding socket failed. If you have any (stale) files"
            "with names like /tmp/nvshmem-socket-<0-9>*, delete or rename them!");
        goto out_error;
    }

    handle->socket = sock;
    strncpy(handle->socketName, temp, NVSHMEM_SOCKET_MAX_LEN);
    handle->socketName[strlen(temp)] = '\0';

    return 0;

out_error:
    delete handle->socketName;
    delete handle;
    close(sock);
    return -1;
}

int ipcCloseSocket(ipcHandle *handle) {
    if (!handle) {
        return -1;
    }

    int rc = 0;
    if (handle->socketName) {
        if (unlink(handle->socketName) < 0) {
            perror("unlink failed for /tmp/nvshmem-socket-<0-9>-* !");
            rc = -1;
        }

        delete[] handle->socketName;
    }
    close(handle->socket);
    delete handle;
    return (rc);
}

int ipcRecvFd(ipcHandle *handle, int *shHandle) {
    struct msghdr msg = {0, 0, 0, 0, 0, 0, 0};
    struct iovec iov[1];

    // Union to guarantee alignment requirements for control array
    union {
        struct cmsghdr cm;
        char control[CMSG_SPACE(sizeof(int))];
    } control_un;

    struct cmsghdr *cmptr;
    int receivedfd;
    char dummy_buffer[1];

    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof(control_un.control);

    iov[0].iov_base = (void *)dummy_buffer;
    iov[0].iov_len = sizeof(dummy_buffer);

    msg.msg_iov = iov;
    msg.msg_iovlen = 1;

    if (recvmsg(handle->socket, &msg, 0) <= 0) {
        perror("IPC failure: Receiving data over socket failed");
        return -1;
    }

    if (((cmptr = CMSG_FIRSTHDR(&msg)) != NULL) && (cmptr->cmsg_len == CMSG_LEN(sizeof(int)))) {
        if ((cmptr->cmsg_level != SOL_SOCKET) || (cmptr->cmsg_type != SCM_RIGHTS)) {
            return -1;
        }

        memmove(&receivedfd, CMSG_DATA(cmptr), sizeof(receivedfd));
        *(int *)shHandle = receivedfd;
    } else {
        return -1;
    }

    return 0;
}

int ipcSendFd(ipcHandle *handle, const int shareableHandle, pid_t send_process,
              pid_t recv_process) {
    struct msghdr msg;
    struct iovec iov[1];

    union {
        struct cmsghdr cm;
        char control[CMSG_SPACE(sizeof(int))];
    } control_un;

    struct cmsghdr *cmptr;
    struct sockaddr_un cliaddr;

    // Construct client address to send this SHareable handle to
    bzero(&cliaddr, sizeof(cliaddr));
    cliaddr.sun_family = AF_UNIX;
    char temp[50];
    int name_len = snprintf(temp, 50, "/tmp/nvshmem-socket-%u-%u", send_process, recv_process);
    if (name_len < 0 || name_len >= 50) {
        printf("Error formatting socket file name\n");
        return -1;
    }
    strncpy(cliaddr.sun_path, temp, 50);

    // Send corresponding shareable handle to the client
    int sendfd = (int)shareableHandle;

    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof(control_un.control);

    cmptr = CMSG_FIRSTHDR(&msg);
    cmptr->cmsg_len = CMSG_LEN(sizeof(int));
    cmptr->cmsg_level = SOL_SOCKET;
    cmptr->cmsg_type = SCM_RIGHTS;

    memmove(CMSG_DATA(cmptr), &sendfd, sizeof(sendfd));

    msg.msg_name = (void *)&cliaddr;
    msg.msg_namelen = sizeof(struct sockaddr_un);

    iov[0].iov_base = (void *)"";
    iov[0].iov_len = 1;
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;
    msg.msg_flags = 0;

    ssize_t sendResult = sendmsg(handle->socket, &msg, 0);
    if (sendResult <= 0) {
        perror("IPC failure: Sending data over socket failed");
        return -1;
    }

    return 0;
}
