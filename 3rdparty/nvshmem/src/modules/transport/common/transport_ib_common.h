/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _TRANSPORT_IB_COMMON_H
#define _TRANSPORT_IB_COMMON_H
#include <arpa/inet.h>                                           // for inet...
#include <errno.h>                                               // for errno
#include <fcntl.h>                                               // for open
#include <limits.h>                                              // for PATH...
#include <netinet/in.h>                                          // for in6_...
#include <stdint.h>                                              // for uint...
#include <stdio.h>                                               // for NULL
#include <stdlib.h>                                              // for getenv
#include <string.h>                                              // for strlen
#include <sys/socket.h>                                          // for AF_INET
#include <sys/un.h>                                              // for sa_f...
#include <unistd.h>                                              // for close
#include "bootstrap_host_transport/env_defs_internal.h"          // for nvsh...
#include "infiniband/verbs.h"                                    // for ibv_gid
#include "internal/host_transport/nvshmemi_transport_defines.h"  // for nvsh...
#include "non_abi/nvshmemx_error.h"                              // for NVSH...
#include "transport_common.h"                                    // for nvsh...
#include "transport_ib_common.h"                                 // lines 26-26

#define DIVUP(x, y) (((x) + (y)-1) / (y))

#define ROUNDUP(x, y) (DIVUP((x), (y)) * (y))

#define NETMASK(bits) (htonl(0xffffffff << (32 - bits)))

struct nvshmemt_ib_common_mem_handle {
    struct ibv_mr *mr;
    void *buf;
    int fd;
    uint32_t lkey;
    uint32_t rkey;
    bool local_only;
};

struct nvshmemt_ib_gid_info {
    uint8_t link_layer;
    union ibv_gid local_gid;
    int32_t local_gid_index;
};

int nvshmemt_ib_common_nv_peer_mem_available();

int nvshmemt_ib_common_reg_mem_handle(struct nvshmemt_ibv_function_table *ftable, struct ibv_pd *pd,
                                      nvshmem_mem_handle_t *mem_handle, void *buf, size_t length,
                                      bool local_only, bool dmabuf_support,
                                      struct nvshmemi_cuda_fn_table *table, int log_level,
                                      bool relaxed_ordering);

int nvshmemt_ib_common_release_mem_handle(struct nvshmemt_ibv_function_table *ftable,
                                          nvshmem_mem_handle_t *mem_handle, int log_level);

/* The following code is for dynamic GID detection for RoCE platforms.
   It has been adapted from NCCL: https://gitlab-master.nvidia.com/nccl/nccl/-/merge_requests/359 */
static sa_family_t env_ib_addr_family(int log_level, nvshmemi_options_s *options) {
    sa_family_t family = AF_INET;
    const char *env = options->IB_ADDR_FAMILY;
    if (env == NULL || strlen(env) == 0) {
        return family;
    }

    INFO(log_level, "NVSHMEM_IB_ADDR_FAMILY set by environment to %s", env);

    if (strcmp(env, "AF_INET") == 0) {
        family = AF_INET;
    } else if (strcmp(env, "AF_INET6") == 0) {
        family = AF_INET6;
    }

    return family;
}

static void *env_ib_addr_range(sa_family_t af, int *prefix_len, int log_level,
                               nvshmemi_options_s *options) {
    *prefix_len = 0;
    static struct in_addr addr;
    static struct in6_addr addr6;
    void *ret = (af == AF_INET) ? (void *)&addr : (void *)&addr6;

    const char *env = options->IB_ADDR_RANGE;
    if (NULL == env || strlen(env) == 0) {
        return NULL;
    }

    INFO(log_level, "NVSHMEM_IB_ADDR_RANGE set by environment to %s", env);

    char addr_string[128] = {0};
    snprintf(addr_string, 128, "%s", env);
    char *addr_str_ptr = addr_string;
    char *mask_str_ptr = strstr(addr_string, "/") + 1;
    if (NULL == mask_str_ptr) {
        return NULL;
    }
    *(mask_str_ptr - 1) = '\0';

    if (inet_pton(af, addr_str_ptr, ret) == 0) {
        INFO(log_level, "NET/IB: Ip address '%s' is invalid for family %s, ignoring address",
             addr_str_ptr, (af == AF_INET) ? "AF_INET" : "AF_INET6");
        return NULL;
    }

    *prefix_len = (int)strtol(mask_str_ptr, NULL, 10);
    if (af == AF_INET && *prefix_len > 32) {
        INFO(log_level, "IB: Ip address mask '%d' is invalid for family %s, ignoring mask",
             *prefix_len, (af == AF_INET) ? "AF_INET" : "AF_INET6");
        *prefix_len = 0;
        ret = NULL;
    } else if (af == AF_INET6 && *prefix_len > 128) {
        INFO(log_level, "IB: Ip address mask '%d' is invalid for family %s, ignoring mask",
             *prefix_len, (af == AF_INET) ? "AF_INET" : "AF_INET6");
        *prefix_len = 0;
        ret = NULL;
    }

    return ret;
}

static sa_family_t get_gid_addr_family(union ibv_gid *gid) {
    const struct in6_addr *a = (struct in6_addr *)gid->raw;
    bool is_ipv4_mapped =
        ((a->s6_addr32[0] | a->s6_addr32[1]) | (a->s6_addr32[2] ^ htonl(0x0000ffff))) == 0UL;
    bool is_ipv4_mapped_multicast =
        (a->s6_addr32[0] == htonl(0xff0e0000) &&
         ((a->s6_addr32[1] | (a->s6_addr32[2] ^ htonl(0x0000ffff))) == 0UL));
    return (is_ipv4_mapped || is_ipv4_mapped_multicast) ? AF_INET : AF_INET6;
}

static bool match_gid_addr_prefix(sa_family_t af, void *prefix, int prefix_len,
                                  union ibv_gid *gid) {
    struct in_addr *base = NULL;
    struct in6_addr *base6 = NULL;
    struct in6_addr *addr6 = NULL;

    if (af == AF_INET) {
        base = (struct in_addr *)prefix;
    } else {
        base6 = (struct in6_addr *)prefix;
    }
    addr6 = (struct in6_addr *)gid->raw;

    int i = 0;
    while (prefix_len > 0 && i < 4) {
        if (af == AF_INET) {
            int mask = NETMASK(prefix_len);
            if ((base->s_addr & mask) ^ (addr6->s6_addr32[3] & mask)) {
                break;
            }
            prefix_len = 0;
            break;
        } else {
            if (prefix_len >= 32) {
                if (base6->s6_addr32[i] ^ addr6->s6_addr32[i]) {
                    break;
                }
                prefix_len -= 32;
                ++i;
            } else {
                int mask = NETMASK(prefix_len);
                if ((base6->s6_addr32[i] & mask) ^ (addr6->s6_addr32[i] & mask)) {
                    break;
                }
                prefix_len = 0;
            }
        }
    }

    return (prefix_len == 0) ? true : false;
}

static bool configured_gid(union ibv_gid *gid) {
    const struct in6_addr *a = (struct in6_addr *)gid->raw;
    int trailer = (a->s6_addr32[1] | a->s6_addr32[2] | a->s6_addr32[3]);
    if (((a->s6_addr32[0] | trailer) == 0UL) ||
        ((a->s6_addr32[0] == htonl(0xfe800000)) && (trailer == 0UL))) {
        return false;
    }
    return true;
}

static bool link_local_gid(union ibv_gid *gid) {
    const struct in6_addr *a = (struct in6_addr *)gid->raw;
    if (a->s6_addr32[0] == htonl(0xfe800000) && a->s6_addr32[1] == 0UL) {
        return true;
    }
    return false;
}

static bool valid_gid(union ibv_gid *gid) { return (configured_gid(gid) && !link_local_gid(gid)); }

static int ib_roce_get_version_num(const char *deviceName, int portNum, int gidIndex,
                                   int *version) {
    char gidRoceVerStr[16] = {0};
    char roceTypePath[PATH_MAX] = {0};
    sprintf(roceTypePath, "/sys/class/infiniband/%s/ports/%d/gid_attrs/types/%d", deviceName,
            portNum, gidIndex);

    int fd = open(roceTypePath, O_RDONLY);
    if (fd == -1) {
        NVSHMEMI_WARN_PRINT("IB: open failed in ib_roce_get_version_num: %s", strerror(errno));
        return NVSHMEMX_ERROR_INTERNAL;
    }
    int ret = read(fd, gidRoceVerStr, 15);
    close(fd);

    if (ret == -1) {
        NVSHMEMI_WARN_PRINT("IB: read failed in ib_roce_get_version_num: %s", strerror(errno));
        return NVSHMEMX_ERROR_INTERNAL;
    }

    if (strlen(gidRoceVerStr)) {
        if (strncmp(gidRoceVerStr, "IB/RoCE v1", strlen("IB/RoCE v1")) == 0 ||
            strncmp(gidRoceVerStr, "RoCE v1", strlen("RoCE v1")) == 0) {
            *version = 1;
        } else if (strncmp(gidRoceVerStr, "RoCE v2", strlen("RoCE v2")) == 0) {
            *version = 2;
        }
    }

    return NVSHMEMX_SUCCESS;
}

static void update_gid_index(struct nvshmemt_ibv_function_table *ftable,
                             struct ibv_context *context, uint8_t portNum, sa_family_t af,
                             void *prefix, int prefixlen, int roceVer, int gidIndexCandidate,
                             int *gidIndex) {
    union ibv_gid gid, gidCandidate;
    ftable->query_gid(context, portNum, *gidIndex, &gid);
    ftable->query_gid(context, portNum, gidIndexCandidate, &gidCandidate);

    sa_family_t usrFam = af;
    sa_family_t gidFam = get_gid_addr_family(&gid);
    sa_family_t gidCandidateFam = get_gid_addr_family(&gidCandidate);
    bool gidCandidateMatchSubnet = match_gid_addr_prefix(usrFam, prefix, prefixlen, &gidCandidate);

    if (gidCandidateFam != gidFam && gidCandidateFam == usrFam && gidCandidateMatchSubnet) {
        *gidIndex = gidIndexCandidate;
    } else {
        if (gidCandidateFam != usrFam || !valid_gid(&gidCandidate) || !gidCandidateMatchSubnet) {
            return;
        }
        int usrRoceVer = roceVer;
        int gidRoceVerNum, gidRoceVerNumCandidate;
        const char *deviceName = ftable->get_device_name(context->device);
        ib_roce_get_version_num(deviceName, portNum, *gidIndex, &gidRoceVerNum);
        ib_roce_get_version_num(deviceName, portNum, gidIndexCandidate, &gidRoceVerNumCandidate);
        if ((gidRoceVerNum != gidRoceVerNumCandidate || !valid_gid(&gid)) &&
            gidRoceVerNumCandidate == usrRoceVer) {
            *gidIndex = gidIndexCandidate;
        }
    }

    return;
}

static void ib_get_gid_index(struct nvshmemt_ibv_function_table *ftable,
                             struct ibv_context *context, uint8_t portNum, int gidTblLen,
                             int *gidIndex, int log_level, nvshmemi_options_s *options) {
    *gidIndex = options->IB_GID_INDEX;
    if (*gidIndex >= 0) {
        return;
    }

    sa_family_t userAddrFamily = env_ib_addr_family(log_level, options);
    int userRoceVersion = options->IB_ROCE_VERSION_NUM;
    int prefixlen;
    void *prefix = env_ib_addr_range(userAddrFamily, &prefixlen, log_level, options);

    *gidIndex = 0;
    for (int gidIndexNext = 1; gidIndexNext < gidTblLen; ++gidIndexNext) {
        update_gid_index(ftable, context, portNum, userAddrFamily, prefix, prefixlen,
                         userRoceVersion, gidIndexNext, gidIndex);
    }

    return;
}
#endif
