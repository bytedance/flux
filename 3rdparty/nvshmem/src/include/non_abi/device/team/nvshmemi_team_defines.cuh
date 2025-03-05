#ifndef _NVSHMEMI_TEAM_DEFINES_CUH_
#define _NVSHMEMI_TEAM_DEFINES_CUH_

#include "device_host/nvshmem_common.cuh"
#include <cuda_runtime.h>

#ifdef __CUDA_ARCH__

__device__ static inline int nvshmemi_team_my_pe(nvshmem_team_t team) {
    if (team == NVSHMEM_TEAM_INVALID)
        return -1;
    else if (team == NVSHMEM_TEAM_WORLD)
        return nvshmemi_device_state_d.mype;
    else if (team == NVSHMEMX_TEAM_NODE)
        return nvshmemi_device_state_d.node_mype;
    else
        return nvshmemi_device_state_d.team_pool[team]->my_pe;
}

__device__ inline int nvshmemi_team_n_pes(nvshmem_team_t team) {
    if (team == NVSHMEM_TEAM_INVALID)
        return -1;
    else if (team == NVSHMEM_TEAM_WORLD)
        return nvshmemi_device_state_d.npes;
    else if (team == NVSHMEMX_TEAM_NODE)
        return nvshmemi_device_state_d.node_npes;
    else
        return nvshmemi_device_state_d.team_pool[team]->size;
}

static inline __device__ size_t get_fcollect_psync_len_per_team() {
    size_t fcollect_ll_threshold =
        nvshmemi_device_state_d.gpu_coll_env_params_var.fcollect_ll_threshold;
    size_t fcollect_sync_size =
        (2 * 2 * nvshmemi_device_state_d.npes * fcollect_ll_threshold) / sizeof(long);

    return fcollect_sync_size;
}

static inline __device__ size_t get_fcollect_ll128_psync_len_per_team() {
    size_t fcollect_ll128_threshold =
        nvshmemi_device_state_d.gpu_coll_env_params_var.fcollect_ll128_threshold;
    size_t fcollect_ll128_sync_size =
        NVSHMEMI_FCOLLECT_LL128_CALC_PSYNC_SIZE(fcollect_ll128_threshold, char);

    /* scale for npes and two separate psyncs */
    fcollect_ll128_sync_size =
        fcollect_ll128_sync_size * 2 * nvshmemi_device_state_d.npes / sizeof(long);

    return fcollect_ll128_sync_size;
}

static inline __device__ size_t get_psync_len_per_team() {
    size_t fcollect_sync_size = get_fcollect_psync_len_per_team();
    size_t fcollect_ll128_sync_size = get_fcollect_ll128_psync_len_per_team();
    /* sync: Two buffers are used - one for sync/barrier collective ops, the second one during team
       split operation reduce: Two pWrk's are used alternatively across consecutive reduce calls,
       this is to avoid having to put a barrier in between bcast: The buffer is split to do multiple
       consecutive broadcast, when all buffers are used, a barrier is called and then again we begin
       from the start of the buffer fcollect: Two sets of buffer are used to alternate between -
       same way as in reduce. The other fator of 2 is because when using LL double the space is
       needed to fuse flag with data */

    return (2 * NVSHMEMI_SYNC_SIZE +
            nvshmemi_device_state_d.gpu_coll_env_params_var.reduce_scratch_size / sizeof(long) +
            NVSHMEMI_BCAST_SYNC_SIZE + fcollect_sync_size + 2 * NVSHMEMI_ALLTOALL_SYNC_SIZE +
            fcollect_ll128_sync_size);
}

__device__ static inline int nvshmemi_pe_in_active_set(int global_pe, int PE_start, int PE_stride,
                                                       int PE_size) {
    int n = (global_pe - PE_start) / PE_stride;
    if (global_pe < PE_start || (global_pe - PE_start) % PE_stride || n >= PE_size)
        return -1;
    else {
        return n;
    }
}

__device__ inline int nvshmemi_team_translate_pe(nvshmem_team_t src_team, int src_pe,
                                                 nvshmem_team_t dest_team) {
    if (src_team == NVSHMEM_TEAM_INVALID || dest_team == NVSHMEM_TEAM_INVALID) return -1;
    nvshmemi_team_t *src_teami, *dest_teami;

    src_teami = nvshmemi_device_state_d.team_pool[src_team];
    dest_teami = nvshmemi_device_state_d.team_pool[dest_team];
    int src_pe_world, dest_pe = -1;

    if (src_pe > src_teami->size) return -1;

    src_pe_world = src_teami->start + src_pe * src_teami->stride;
    assert(src_pe_world >= src_teami->start && src_pe_world < nvshmemi_device_state_d.npes);

    dest_pe = nvshmemi_pe_in_active_set(src_pe_world, dest_teami->start, dest_teami->stride,
                                        dest_teami->size);

    return dest_pe;
}

__device__ inline long *nvshmemi_team_get_psync(nvshmemi_team_t *team, nvshmemi_team_op_t op) {
    long *team_psync;
    size_t psync_fcollect_len;
    psync_fcollect_len = get_fcollect_psync_len_per_team();
    team_psync = &nvshmemi_device_state_d.psync_pool[team->team_idx * get_psync_len_per_team()];
    switch (op) {
        case SYNC:
            return team_psync;
        case REDUCE:
            return &team_psync
                [2 * NVSHMEMI_SYNC_SIZE +
                 (((nvshmemi_device_state_d.gpu_coll_env_params_var.reduce_scratch_size / 2) /
                   sizeof(long)) *
                  (team->rdxn_count % 2))];
        case BCAST:
            return &team_psync[2 * NVSHMEMI_SYNC_SIZE +
                               nvshmemi_device_state_d.gpu_coll_env_params_var.reduce_scratch_size /
                                   sizeof(long)];
        case FCOLLECT:
            return &team_psync[2 * NVSHMEMI_SYNC_SIZE +
                               nvshmemi_device_state_d.gpu_coll_env_params_var.reduce_scratch_size /
                                   sizeof(long) +
                               NVSHMEMI_BCAST_SYNC_SIZE];
        case ALLTOALL:
            return &team_psync[2 * NVSHMEMI_SYNC_SIZE +
                               nvshmemi_device_state_d.gpu_coll_env_params_var.reduce_scratch_size /
                                   sizeof(long) +
                               NVSHMEMI_BCAST_SYNC_SIZE + psync_fcollect_len +
                               (NVSHMEMI_ALLTOALL_SYNC_SIZE * (team->alltoall_count % 2))];
        case FCOLLECT_128:
            return &team_psync[2 * NVSHMEMI_SYNC_SIZE +
                               nvshmemi_device_state_d.gpu_coll_env_params_var.reduce_scratch_size /
                                   sizeof(long) +
                               NVSHMEMI_BCAST_SYNC_SIZE + psync_fcollect_len +
                               2 * NVSHMEMI_ALLTOALL_SYNC_SIZE];
        default:
            printf("Incorrect argument to nvshmemi_team_get_psync\n");
            return NULL;
    }
}

__device__ inline long *nvshmemi_team_get_sync_counter(nvshmemi_team_t *team) {
    return &nvshmemi_device_state_d.sync_counter[2 * team->team_idx];
}
#endif

#endif
