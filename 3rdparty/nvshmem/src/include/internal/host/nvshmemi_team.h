#ifndef NVSHMEMI_TEAM_H
#define NVSHMEMI_TEAM_H

#define N_PSYNCS_PER_TEAM 1
#include "device_host_transport/nvshmem_constants.h"
#include "device_host/nvshmem_common.cuh"

#define N_PSYNC_BYTES 16

extern nvshmemi_team_t nvshmemi_team_world;
extern nvshmemi_team_t nvshmemi_team_shared;
extern nvshmemi_team_t nvshmemi_team_node;
extern nvshmemi_team_t nvshmemi_team_same_mype_node;
extern nvshmemi_team_t nvshmemi_team_same_gpu;
extern nvshmemi_team_t nvshmemi_team_gpu_leaders;

/* Team Management Routines */

int nvshmemi_set_max_teams(void);

int nvshmemi_team_init(void);

int nvshmemi_team_finalize(void);

int nvshmemi_team_my_pe(nvshmemi_team_t *team);

int nvshmemi_team_n_pes(nvshmemi_team_t *team);

void nvshmemi_team_get_config(nvshmemi_team_t *team, nvshmem_team_config_t *config);

int nvshmemi_team_split_strided(nvshmemi_team_t *parent_team, int PE_start, int PE_stride,
                                int PE_size, const nvshmem_team_config_t *config, long config_mask,
                                nvshmem_team_t *new_team);

int nvshmemi_team_split_2d(nvshmemi_team_t *parent_team, int xrange,
                           const nvshmem_team_config_t *xaxis_config, long xaxis_mask,
                           nvshmem_team_t *xaxis_team, const nvshmem_team_config_t *yaxis_config,
                           long yaxis_mask, nvshmem_team_t *yaxis_team);

void nvshmemi_team_destroy(nvshmemi_team_t *team);

/*int nvshmemi_team_create_ctx(nvshmemi_team_t *team, long options, shmem_ctx_t *ctx);

int nvshmemi_ctx_get_team(shmem_ctx_t ctx, nvshmemi_team_t **team);*/

static inline int nvshmemi_team_pe(nvshmemi_team_t *team, int pe) {
    return team->start + team->stride * pe;
}

size_t nvshmemi_get_teams_mem_requirement();

bool nvshmemi_team_support_nvls(nvshmemi_team_t *team);

long *nvshmemi_team_get_psync(nvshmemi_team_t *team, nvshmemi_team_op_t op);

int nvshmemi_team_translate_pe(nvshmemi_team_t *src_team, int src_pe, nvshmemi_team_t *dest_team);

#endif
