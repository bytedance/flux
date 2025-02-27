/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "team_internal.h"
#include "internal/host/nvshmem_internal.h"
#include "internal/host/util.h"
#include "internal/host/nvshmemi_team.h"

/* Team Managment Routines */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef __CUDA_ARCH__
int nvshmem_team_my_pe(nvshmem_team_t team) {
    if (team == NVSHMEM_TEAM_INVALID)
        return -1;
    else if (team == NVSHMEM_TEAM_WORLD)
        return nvshmemi_boot_handle.pg_rank;
    else if (team == NVSHMEMX_TEAM_NODE)
        return nvshmemi_boot_handle.mype_node;
    else
        return nvshmemi_team_pool[team]->my_pe;
}

int nvshmem_team_n_pes(nvshmem_team_t team) {
    if (team == NVSHMEM_TEAM_INVALID)
        return -1;
    else if (team == NVSHMEM_TEAM_WORLD)
        return nvshmemi_boot_handle.pg_size;
    else if (team == NVSHMEMX_TEAM_NODE)
        return nvshmemi_boot_handle.npes_node;
    else
        return nvshmemi_team_pool[team]->size;
}

void nvshmem_team_get_config(nvshmem_team_t team, nvshmem_team_config_t *config) {
    NVSHMEMI_CHECK_INIT_STATUS();
    if (team == NVSHMEM_TEAM_INVALID) return;

    nvshmemi_team_t *myteam = nvshmemi_team_pool[team];
    *config = myteam->config;
    return;
}

int nvshmem_team_translate_pe(nvshmem_team_t src_team, int src_pe, nvshmem_team_t dest_team) {
    if (src_team == NVSHMEM_TEAM_INVALID || dest_team == NVSHMEM_TEAM_INVALID) return -1;
    nvshmemi_team_t *src_teami, *dest_teami;
    NVSHMEMI_CHECK_INIT_STATUS();
    src_teami = nvshmemi_team_pool[src_team];
    dest_teami = nvshmemi_team_pool[dest_team];

    return nvshmemi_team_translate_pe(src_teami, src_pe, dest_teami);
}

int nvshmem_team_split_strided(nvshmem_team_t parent_team, int PE_start, int PE_stride, int PE_size,
                               const nvshmem_team_config_t *config, long config_mask,
                               nvshmem_team_t *new_team) {
    NVSHMEMI_CHECK_INIT_STATUS();
    NVSHMEM_API_NOT_SUPPORTED_WITH_LIMITED_MPG_RUNS();
    if (parent_team == NVSHMEM_TEAM_INVALID) {
        *new_team = NVSHMEM_TEAM_INVALID;
        return 1;
    }
    return nvshmemi_team_split_strided(nvshmemi_team_pool[parent_team], PE_start, PE_stride,
                                       PE_size, config, config_mask, new_team);
}

int nvshmem_team_split_2d(nvshmem_team_t parent_team, int xrange,
                          const nvshmem_team_config_t *xaxis_config, long xaxis_mask,
                          nvshmem_team_t *xaxis_team, const nvshmem_team_config_t *yaxis_config,
                          long yaxis_mask, nvshmem_team_t *yaxis_team) {
    NVSHMEMI_CHECK_INIT_STATUS();
    NVSHMEM_API_NOT_SUPPORTED_WITH_LIMITED_MPG_RUNS();
    if (parent_team == NVSHMEM_TEAM_INVALID) {
        *yaxis_team = NVSHMEM_TEAM_INVALID;
        *xaxis_team = NVSHMEM_TEAM_INVALID;
        return 1;
    }
    return nvshmemi_team_split_2d(nvshmemi_team_pool[parent_team], xrange, xaxis_config, xaxis_mask,
                                  xaxis_team, yaxis_config, yaxis_mask, yaxis_team);
}

void nvshmem_team_destroy(nvshmem_team_t team) {
    NVSHMEMI_CHECK_INIT_STATUS();
    if (team == NVSHMEM_TEAM_WORLD || team == NVSHMEM_TEAM_SHARED || team == NVSHMEMX_TEAM_NODE ||
        team == NVSHMEMX_TEAM_SAME_MYPE_NODE || team == NVSHMEMI_TEAM_SAME_GPU ||
        team == NVSHMEMI_TEAM_GPU_LEADERS)
        NVSHMEMI_ERROR_EXIT("Cannot destroy a pre-defined team");
    if (team == NVSHMEM_TEAM_INVALID) return;

    nvshmemi_team_destroy(nvshmemi_team_pool[team]);
}
#endif

#ifdef __cplusplus
}
#endif
