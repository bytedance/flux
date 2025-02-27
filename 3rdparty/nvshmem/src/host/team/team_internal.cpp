/*
 * Copyright (c) 2016-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "team_internal.h"
#include <assert.h>                                                        // for assert
#include <cuda_runtime.h>                                                  // for cudaMemcpy
#include <driver_types.h>                                                  // for cudaMemcpyHos...
#include <limits.h>                                                        // for CHAR_BIT
#include <stdint.h>                                                        // for uint64_t
#include <stdio.h>                                                         // for snprintf, printf
#include <stdlib.h>                                                        // for free, malloc
#include <string.h>                                                        // for memset, memcmp
#include <cmath>                                                           // for ceil
#include "../coll/rdxn/rdxn.h"                                             // for nvshmemi_call...
#include "device_host/nvshmem_types.h"                                     // for nvshmemi_team_t
#include "cpu_coll.h"                                                      // for nccl_ftable
#include "device_host/nvshmem_common.cuh"                                  // for nvshmemi_pe_i...
#include "bootstrap_host_transport/env_defs_internal.h"                    // for nvshmemi_opti...
#include "host/nvshmem_api.h"                                              // for nvshmem_quiet
#include "host/nvshmem_coll_api.h"                                         // for nvshmem_team_...
#include "host/nvshmemx_api.h"                                             // for nvshmemx_char...
#include "non_abi/nvshmemx_error.h"                                        // for NVSHMEMI_NULL...
#include "internal/host/debug.h"                                           // for INFO, NVSHMEM...
#include "internal/host/nvshmem_internal.h"                                // for nvshmemi_free
#include "internal/host/nvshmemi_coll.h"                                   // for nvshmemi_barrier
#include "internal/host/nvshmemi_symmetric_heap.hpp"                       // for nvshmemi_symm...
#include "internal/host/nvshmemi_team.h"                                   // for N_PSYNC_BYTES
#include "internal/host/nvshmemi_types.h"                                  // for nvshmemi_state
#include "internal/host/util.h"                                            // for CUDA_RUNTIME_...
#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"  // for nvshmemi_boot...
#include "internal/host_transport/transport.h"                             // for nvshmem_trans...
#include "non_abi/nvshmem_build_options.h"                                 // for NVSHMEM_USE_NCCL
#include "internal/host/nvshmemi_nvls_rsc.hpp"                             // for nvshmemi_nvls...
#ifdef NVSHMEM_USE_NCCL
#include "nccl.h"  // for ncclUniqueId
#endif
#include "internal/host_transport/nvshmemi_transport_defines.h"  // for NVSHMEM_MEM_H...

using namespace nvls;

#define NVSHMEMI_DIAG_STRLEN 1024
#define NVSHMEMI_SYNC_VALUE 0

/* 0th entry in team duplicate resources is same team as the encapsulating team. This allows
 * for reuse of the same business logic for nCTA == 1 and nCTA > 1 and minimizes if/else
 */
#define NVSHMEMI_TEAM_DUP_INITIALIZER(teami, team_idx) \
    (teami).team_dups[0] = (team_idx);                 \
    for (int i = 1; i < 128; i++) {                    \
        (teami).team_dups[i] = NVSHMEM_TEAM_INVALID;   \
    }

long nvshmemi_max_teams;

nvshmemi_team_t nvshmemi_team_world;
nvshmemi_team_t nvshmemi_team_shared;
nvshmemi_team_t nvshmemi_team_node;
nvshmemi_team_t nvshmemi_team_same_mype_node;
nvshmemi_team_t nvshmemi_team_same_gpu;
nvshmemi_team_t nvshmemi_team_gpu_leaders;

nvshmemi_team_t *nvshmemi_device_team_world, *nvshmemi_device_team_shared,
    *nvshmemi_device_team_node, *nvshmemi_device_team_same_mype_node,
    *nvshmemi_device_team_same_gpu, *nvshmemi_device_team_gpu_leaders;

nvshmemi_team_t **nvshmemi_team_pool;
long *nvshmemi_psync_pool;
long *nvshmemi_sync_counter;

nvshmemi_team_t **nvshmemi_device_team_pool;

static unsigned char *psync_pool_avail;
static unsigned char *psync_pool_avail_reduced;
static unsigned char *device_psync_pool_avail;
static unsigned char *device_psync_pool_avail_reduced;

static int *team_ret_val;
static int *team_ret_val_reduced;
static int *device_team_ret_val;
static int *device_team_ret_val_reduced;

bool nvshmemi_team_support_nvls(nvshmemi_team_t *team) {
    return ((team->are_gpus_p2p_connected) && (team->nvls_rsc != nullptr));
}

bool nvshmemi_team_is_owner_nvls(nvshmemi_team_t *team) {
    nvshmemi_nvls_rsc *nvls = reinterpret_cast<nvshmemi_nvls_rsc *>(team->nvls_rsc);
    INFO(NVSHMEM_TEAM, "Team ID: %d NVLS Resource Owner ID: %d\n", team->team_idx,
         nvls->get_owner());
    return (nvls->is_owner(team));
}

static bool nvshmemi_team_is_nvls_capable(nvshmemi_team_t *team) {
    return (team->are_gpus_p2p_connected && team->size >= 2 && nvshmemi_state->is_platform_nvls);
}

static bool nvshmemi_team_is_identical(nvshmemi_team_t *t1, nvshmemi_team_t *t2) {
    return (t1->start == t2->start && t1->stride == t2->stride && t1->size == t2->size);
}

static void nvshmemi_team_alloc_device(void) {
    CUDA_RUNTIME_CHECK(cudaMalloc((void **)&nvshmemi_device_team_world, sizeof(nvshmemi_team_t)));
    CUDA_RUNTIME_CHECK(cudaMalloc((void **)&nvshmemi_device_team_shared, sizeof(nvshmemi_team_t)));
    CUDA_RUNTIME_CHECK(cudaMalloc((void **)&nvshmemi_device_team_node, sizeof(nvshmemi_team_t)));
    CUDA_RUNTIME_CHECK(
        cudaMalloc((void **)&nvshmemi_device_team_same_mype_node, sizeof(nvshmemi_team_t)));
    CUDA_RUNTIME_CHECK(
        cudaMalloc((void **)&nvshmemi_device_team_same_gpu, sizeof(nvshmemi_team_t)));
    CUDA_RUNTIME_CHECK(
        cudaMalloc((void **)&nvshmemi_device_team_gpu_leaders, sizeof(nvshmemi_team_t)));
}

static void nvshmemi_team_update_device(void) {
    CUDA_RUNTIME_CHECK(cudaMemcpy(nvshmemi_device_team_world, &nvshmemi_team_world,
                                  sizeof(nvshmemi_team_t), cudaMemcpyHostToDevice));
    CUDA_RUNTIME_CHECK(cudaMemcpy(&nvshmemi_device_team_pool[NVSHMEM_TEAM_WORLD_INDEX],
                                  &nvshmemi_device_team_world, sizeof(nvshmemi_team_t *),
                                  cudaMemcpyHostToDevice));

    CUDA_RUNTIME_CHECK(cudaMemcpy(nvshmemi_device_team_shared, &nvshmemi_team_shared,
                                  sizeof(nvshmemi_team_t), cudaMemcpyHostToDevice));
    CUDA_RUNTIME_CHECK(cudaMemcpy(&nvshmemi_device_team_pool[NVSHMEM_TEAM_SHARED_INDEX],
                                  &nvshmemi_device_team_shared, sizeof(nvshmemi_team_t *),
                                  cudaMemcpyHostToDevice));

    CUDA_RUNTIME_CHECK(cudaMemcpy(nvshmemi_device_team_node, &nvshmemi_team_node,
                                  sizeof(nvshmemi_team_t), cudaMemcpyHostToDevice));
    CUDA_RUNTIME_CHECK(cudaMemcpy(&nvshmemi_device_team_pool[NVSHMEM_TEAM_NODE_INDEX],
                                  &nvshmemi_device_team_node, sizeof(nvshmemi_team_t *),
                                  cudaMemcpyHostToDevice));

    CUDA_RUNTIME_CHECK(cudaMemcpy(nvshmemi_device_team_same_mype_node,
                                  &nvshmemi_team_same_mype_node, sizeof(nvshmemi_team_t),
                                  cudaMemcpyHostToDevice));
    CUDA_RUNTIME_CHECK(cudaMemcpy(&nvshmemi_device_team_pool[NVSHMEM_TEAM_SAME_MYPE_NODE_INDEX],
                                  &nvshmemi_device_team_same_mype_node, sizeof(nvshmemi_team_t *),
                                  cudaMemcpyHostToDevice));

    CUDA_RUNTIME_CHECK(cudaMemcpy(nvshmemi_device_team_same_gpu, &nvshmemi_team_same_gpu,
                                  sizeof(nvshmemi_team_t), cudaMemcpyHostToDevice));
    CUDA_RUNTIME_CHECK(cudaMemcpy(&nvshmemi_device_team_pool[NVSHMEM_TEAM_SAME_GPU_INDEX],
                                  &nvshmemi_device_team_same_gpu, sizeof(nvshmemi_team_t *),
                                  cudaMemcpyHostToDevice));

    CUDA_RUNTIME_CHECK(cudaMemcpy(nvshmemi_device_team_gpu_leaders, &nvshmemi_team_gpu_leaders,
                                  sizeof(nvshmemi_team_t), cudaMemcpyHostToDevice));
    CUDA_RUNTIME_CHECK(cudaMemcpy(&nvshmemi_device_team_pool[NVSHMEM_TEAM_GPU_LEADERS_INDEX],
                                  &nvshmemi_device_team_gpu_leaders, sizeof(nvshmemi_team_t *),
                                  cudaMemcpyHostToDevice));
    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
}

static void nvshmemi_recexchalgo_get_neighbors(nvshmemi_team_t *teami) {
    int i, j, k;
    int p_of_k = 1, log_p_of_k = 0, rem, T, newpe;
    int step1_sendto, step1_nrecvs, step2_nphases;
    int *step1_recvfrom, **step2_nbrs;
    int *step1_recvfrom_device, **step2_nbrs_device;

    int my_pe = teami->my_pe;
    int num_pes = teami->size;
    INFO(NVSHMEM_COLL, "step 1 nbr calculation started, num_pes = %d", num_pes);

    k = nvshmemi_options.REDUCE_RECEXCH_KVAL;
    assert(k > 1);

    if (num_pes < k) /* If size of the active set is less than k, reduce the value of k */
        k = (num_pes > 2) ? num_pes : 2;

    /* Calculate p_of_k, p_of_k is the largest power of k that is less than num_pes */
    while (p_of_k <= num_pes) {
        p_of_k *= k;
        log_p_of_k++;
    }
    p_of_k /= k;
    /* protect against underflow warnings when asserts are disabled. */
    if (log_p_of_k > 0) {
        log_p_of_k--;
    }

    step2_nphases = log_p_of_k;
    step1_recvfrom = (int *)malloc(sizeof(int) * (k - 1));
    assert(step1_recvfrom);
    step2_nbrs = (int **)malloc(sizeof(int *) * step2_nphases);
    assert(step2_nbrs);

    for (int i = 0; i < step2_nphases; i++) {
        step2_nbrs[i] = (int *)malloc(sizeof(int) * (k - 1));
        assert(step2_nbrs[i]);
    }

    rem = num_pes - p_of_k;
    /* rem is the number of PEs that do not particpate in Step 2
     * We need to identify these non-participating PEs. This is done in the following way.
     * The first T PEs are divided into sets of k consecutive PEs each.
     * In each of these sets, the first k-1 PEs are the non-participating
     * PEs while the last PE is the participating PE.
     * The non-participating PEs send their data to the participating PE
     * in their corresponding set.
     */
    T = (rem * k) / (k - 1);

    INFO(NVSHMEM_COLL, "step 1 nbr calculation started. T is %d", T);
    step1_nrecvs = 0;
    step1_sendto = -1;

    /* Step 1 */
    if (my_pe < T) {
        if (my_pe % k != (k - 1)) {                     /* I am a non-participating PE */
            step1_sendto = my_pe + (k - 1 - my_pe % k); /* partipating PE to send the data to */
            /* if the corresponding participating PE is not in T,
             * then send to the Tth PE to preserve non-commutativity */
            if (step1_sendto > T - 1) step1_sendto = T;
            newpe = -1; /* tag this PE as non-participating */
        } else {        /* participating PE */
            for (i = 0; i < k - 1; i++) {
                step1_recvfrom[i] = my_pe - i - 1;
            }
            step1_nrecvs = k - 1;
            newpe = my_pe / k; /* this is the new PE amongst the set of participating PEs */
        }
    } else { /* PE >= T */
        newpe = my_pe - rem;

        if (my_pe == T && (T - 1) % k != k - 1 && T >= 1) {
            int nsenders = (T - 1) % k + 1; /* number of PEs sending their data to me in Step 1 */

            for (j = nsenders - 1; j >= 0; j--) {
                step1_recvfrom[nsenders - 1 - j] = T - nsenders + j;
            }
            step1_nrecvs = nsenders;
        }
    }

    INFO(NVSHMEM_COLL, "step 1 nbr computation completed");

    /* Step 2 */
    if (step1_sendto == -1) { /* calulate step2_nbrs only for participating PEs */
        int *digit = (int *)malloc(sizeof(int) * step2_nphases);
        assert(digit != NULL);
        int temppe = newpe;
        int mask = 0x1;
        int phase = 0, cbit, cnt, nbr, power;

        /* calculate the digits in base k representation of newpe */
        for (i = 0; i < log_p_of_k; i++) digit[i] = 0;

        int remainder, i_digit = 0;
        while (temppe != 0) {
            remainder = temppe % k;
            temppe = temppe / k;
            digit[i_digit] = remainder;
            i_digit++;
        }

        while (mask < p_of_k) {
            cbit =
                digit[phase]; /* phase_th digit changes in this phase, obtain its original value */
            cnt = 0;
            for (i = 0; i < k; i++) { /* there are k-1 neighbors */
                if (i != cbit) {      /* do not generate yourself as your nieighbor */
                    digit[phase] = i; /* this gets us the base k representation of the neighbor */

                    /* calculate the base 10 value of the neighbor PE */
                    nbr = 0;
                    power = 1;
                    for (j = 0; j < log_p_of_k; j++) {
                        nbr += digit[j] * power;
                        power *= k;
                    }

                    /* calculate its real PE and store it */
                    step2_nbrs[phase][cnt] =
                        (nbr < rem / (k - 1)) ? (nbr * k) + (k - 1) : nbr + rem;
                    cnt++;
                }
            }
            INFO(NVSHMEM_COLL, "step 2, phase %d nbr calculation completed", phase);
            digit[phase] = cbit; /* reset the digit to original value */
            phase++;
            mask *= k;
        }
        free(digit);
    }
    // Update with global PE numbers
    if (step1_sendto != -1) step1_sendto = teami->start + step1_sendto * teami->stride;
    for (int i = 0; i < step1_nrecvs; i++)
        step1_recvfrom[i] = teami->start + step1_recvfrom[i] * teami->stride;
    for (int i = 0; i < step2_nphases; i++)
        for (int j = 0; j < k - 1; j++)
            step2_nbrs[i][j] = teami->start + step2_nbrs[i][j] * teami->stride;

    // Copy the data to device memory
    CUDA_RUNTIME_CHECK(cudaMalloc(&step1_recvfrom_device, sizeof(int) * (k - 1)));
    CUDA_RUNTIME_CHECK(cudaMalloc(
        &step2_nbrs_device,
        sizeof(int *) * (step2_nphases + 1))); /* + 1 to make it non-zero otherwise cuMemAlloc
                                                  returns error when step2_nphases is 0 */

    for (int i = 0; i < step2_nphases; i++) {
        void *dev_ptr;
        CUDA_RUNTIME_CHECK(cudaMalloc(&dev_ptr, sizeof(int) * (k - 1)));
        CUDA_RUNTIME_CHECK(cudaMemcpy((int **)step2_nbrs_device + i, &dev_ptr, sizeof(int *),
                                      cudaMemcpyHostToDevice));
    }
    CUDA_RUNTIME_CHECK(cudaMemcpy(step1_recvfrom_device, step1_recvfrom, sizeof(int) * step1_nrecvs,
                                  cudaMemcpyHostToDevice));
    void *dev_ptr, *dev_ptr_2;
    dev_ptr = step2_nbrs_device;
    for (int i = 0; i < step2_nphases; i++) {
        CUDA_RUNTIME_CHECK(
            cudaMemcpy(&dev_ptr_2, (int **)dev_ptr + i, sizeof(int *), cudaMemcpyDeviceToHost));
        CUDA_RUNTIME_CHECK(
            cudaMemcpy(dev_ptr_2, step2_nbrs[i], sizeof(int) * (k - 1), cudaMemcpyHostToDevice));
    }
    teami->reduce_recexch.step1_sendto = step1_sendto;
    teami->reduce_recexch.step1_nrecvs = step1_nrecvs;
    teami->reduce_recexch.step2_nphases = step2_nphases;
    teami->reduce_recexch.step1_recvfrom = step1_recvfrom_device;
    teami->reduce_recexch.step2_nbrs = step2_nbrs_device;

    free(step1_recvfrom);
    for (int i = 0; i < step2_nphases; i++) {
        if (step2_nbrs[i]) {
            free(step2_nbrs[i]);
        }
    }
    free(step2_nbrs);
}

static void nvshmemi_recexchalgo_free_mem(nvshmemi_team_t *teami) {
    CUDA_RUNTIME_CHECK(cudaFree(teami->reduce_recexch.step1_recvfrom));
    for (int i = 0; i < teami->reduce_recexch.step2_nphases; i++) {
        void *dev_ptr;
        CUDA_RUNTIME_CHECK(cudaMemcpy(&dev_ptr, teami->reduce_recexch.step2_nbrs + i, sizeof(int *),
                                      cudaMemcpyDeviceToHost));
        CUDA_RUNTIME_CHECK(cudaFree(dev_ptr));
    }
    CUDA_RUNTIME_CHECK(cudaFree(teami->reduce_recexch.step2_nbrs));
}

static inline void nvshmemi_bit_set(unsigned char *ptr, size_t size, size_t index) {
    assert(size > 0 && (index < size * CHAR_BIT));

    size_t which_byte = index / CHAR_BIT;
    ptr[which_byte] |= (1 << (index % CHAR_BIT));

    return;
}

static inline void nvshmemi_bit_clear(unsigned char *ptr, size_t size, size_t index) {
    assert(size > 0 && (index < size * CHAR_BIT));

    size_t which_byte = index / CHAR_BIT;
    ptr[which_byte] &= ~(1 << (index % CHAR_BIT));

    return;
}

static inline unsigned char nvshmemi_bit_fetch(unsigned char *ptr, size_t index) {
    return (ptr[index / CHAR_BIT] >> (index % CHAR_BIT)) & 1;
}

static inline size_t nvshmemi_bit_1st_nonzero(const unsigned char *ptr, const size_t size) {
    /* The following ignores endianess: */
    for (size_t i = 0; i < size; i++) {
        unsigned char bit_val = ptr[i];
        for (size_t j = 0; bit_val && j < CHAR_BIT; j++) {
            if (bit_val & 1) return i * CHAR_BIT + j;
            bit_val >>= 1;
        }
    }

    return (size_t)-1;
}

/* Create a bit string of the format AAAAAAAA.BBBBBBBB into str for the byte
 * array passed via ptr. */
static inline void nvshmemi_bit_to_string(char *str, size_t str_size, unsigned char *ptr,
                                          size_t ptr_size) {
    size_t off = 0;

    for (size_t i = 0; i < ptr_size; i++) {
        for (size_t j = 0; j < CHAR_BIT; j++) {
            off += snprintf(str + off, str_size - off, "%s",
                            (ptr[i] & (1 << (CHAR_BIT - 1 - j))) ? "1" : "0");
            if (off >= str_size) return;
        }
        if (i < ptr_size - 1) {
            off += snprintf(str + off, str_size - off, ".");
            if (off >= str_size) return;
        }
    }
}

/* Checks whether a PE has a consistent stride given (start, stride, size).
 * This function is useful within a loop across PE IDs, and sets 'start',
 * 'stride' and 'size' accordingly upon exiting the loop. It also assumes
 * 'start' and 'stride' are initialized to a negative number and 'size' to 0.
 * If an inconsistent stride is found, returns -1. */
static inline int check_for_linear_stride(int pe, int *start, int *stride, int *size) {
    if (*start < 0) {
        *start = pe;
        (*size)++;
    } else if (*stride < 0) {
        *stride = pe - *start;
        (*size)++;
    } else if ((pe - *start) % *stride != 0) {
        NVSHMEMI_WARN_PRINT("Detected non-uniform stride inserting PE %d into <%d, %d, %d>\n", pe,
                            *start, *stride, *size);
        return -1;
    } else {
        (*size)++;
    }
    return 0;
}

static inline int nvshmemi_pe_in_active_set(int global_pe, int PE_start, int PE_stride,
                                            int PE_size) {
    int n = (global_pe - PE_start) / PE_stride;
    if (global_pe < PE_start || (global_pe - PE_start) % PE_stride || n >= PE_size)
        return -1;
    else {
        return n;
    }
}

int nvshmemi_team_translate_pe(nvshmemi_team_t *src_team, int src_pe, nvshmemi_team_t *dest_team) {
    int src_pe_world, dest_pe = -1;

    if (src_pe > src_team->size) return -1;

    src_pe_world = src_team->start + src_pe * src_team->stride;
    assert(src_pe_world >= src_team->start && src_pe_world < nvshmemi_state->npes);

    dest_pe = nvshmemi_pe_in_active_set(src_pe_world, dest_team->start, dest_team->stride,
                                        dest_team->size);

    return dest_pe;
}

static inline size_t get_fcollect_psync_len_per_team() {
    size_t fcollect_ll_threshold =
        nvshmemi_device_state.gpu_coll_env_params_var.fcollect_ll_threshold;
    size_t fcollect_sync_size =
        (2 * 2 * nvshmemi_state->npes * fcollect_ll_threshold) / sizeof(long);
    assert(fcollect_ll_threshold % sizeof(long) == 0);

    return fcollect_sync_size;
}

static inline size_t get_fcollect_ll128_psync_len_per_team() {
    size_t fcollect_ll128_threshold =
        nvshmemi_device_state.gpu_coll_env_params_var.fcollect_ll128_threshold;
    size_t fcollect_ll128_sync_size =
        NVSHMEMI_FCOLLECT_LL128_CALC_PSYNC_SIZE(fcollect_ll128_threshold, char);

    /* scale for npes and two separate psyncs */
    fcollect_ll128_sync_size = fcollect_ll128_sync_size * 2 * nvshmemi_state->npes / sizeof(long);

    return fcollect_ll128_sync_size;
}

static inline size_t get_psync_len_per_team() {
    size_t fcollect_sync_size = get_fcollect_psync_len_per_team();
    size_t fcollect_ll128_sync_size = get_fcollect_ll128_psync_len_per_team();
    /* sync: Two buffers are used - one for sync/barrier collective ops, the second one during team
       split operation reduce: Two pWrk's are used alternatively across consecutive reduce calls,
       this is to avoid having to put a barrier in between bcast: The buffer is split to do multiple
       consecutive broadcast, when all buffers are used, a barrier is called and then again we begin
       from the start of the buffer fcollect: Two sets of buffer are used to alternate between -
       same way as in reduce. The other fator of 2 is because when using LL double the space is
       needed to fuse flag with data */

    size_t ans = (2 * NVSHMEMI_SYNC_SIZE +
                  nvshmemi_device_state.gpu_coll_env_params_var.reduce_scratch_size / sizeof(long) +
                  NVSHMEMI_BCAST_SYNC_SIZE + fcollect_sync_size + 2 * NVSHMEMI_ALLTOALL_SYNC_SIZE +
                  fcollect_ll128_sync_size);
    return ans;
}

size_t nvshmemi_get_teams_mem_requirement() {
    size_t psync_size = get_psync_len_per_team();
    size_t teams_mem_req = sizeof(long) * nvshmemi_max_teams * psync_size + /* psync's */
                           2 * N_PSYNC_BYTES +                              /* psync_pool_avail */
                           2 * sizeof(int) +                                /* team_ret_val */
                           2 * sizeof(long) * nvshmemi_max_teams            /* storing counters */
#ifdef NVSHMEM_USE_NCCL
                           + sizeof(ncclUniqueId)
#endif
        ;
    INFO(NVSHMEM_INIT, "team psync mem req %ld bytes, team mem total req %d bytes, max teams %ld\n",
         psync_size, teams_mem_req, nvshmemi_max_teams);
    return teams_mem_req;
}

#ifdef NVSHMEM_USE_NCCL
void nvshmemi_team_init_nccl_comm(nvshmemi_team_t *teami) {
    ncclUniqueId Id;
    int start = teami->start;
    int stride = teami->stride;
    int size = teami->size;
    /* This is technical debt where we are using the REDUCE op psync as scratchpad for src/dst of
     * broadcast broadcast's psync is used for LL8 and other algorithms, making it non-trivial to
     * share when issued from the host as a src or dest buffer.
     *
     * When reduce coll supports LL8 algorithm, we need to clean this up as a independent scratch
     * space
     */
    long *pWrk = nvshmemi_team_get_psync(teami, REDUCE);
    if (teami->my_pe == 0) {
        NCCL_CHECK(nccl_ftable.GetUniqueId(&Id));
        CUDA_RUNTIME_CHECK(cudaMemcpy(pWrk, &Id, sizeof(ncclUniqueId), cudaMemcpyHostToDevice));
        CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
        for (int i = 0; i < size; i++) {
            nvshmemx_char_put_nbi_on_stream((char *)pWrk, (const char *)pWrk, sizeof(ncclUniqueId),
                                            start + i * stride, (cudaStream_t)0);
        }
        nvshmemi_barrier(teami->team_idx);
    } else {
        nvshmemi_barrier(teami->team_idx);
        CUDA_RUNTIME_CHECK(cudaMemcpy(&Id, pWrk, sizeof(ncclUniqueId), cudaMemcpyDeviceToHost));
    }
    INFO(NVSHMEM_TEAM, "Calling ncclCommInitRank, teami->size = %d, teami->my_pe = %d", teami->size,
         teami->my_pe);
    NCCL_CHECK(
        nccl_ftable.CommInitRank((ncclComm_t *)&teami->nccl_comm, teami->size, Id, teami->my_pe));
}
#endif /* NVSHMEM_USE_NCCL */
void nvshmemi_team_set_p2p_connectivity(nvshmemi_team_t *teami) {
    teami->are_gpus_p2p_connected = 1;
    for (int pe = teami->start; pe < teami->start + teami->stride * teami->size;
         pe += teami->stride) {
        if (nvshmemi_state->heap_obj->get_local_pe_base()[pe] == NULL) {
            teami->are_gpus_p2p_connected = 0;
            break;
        }
    }
}

/* NVLS Resource management for teams */
static void nvshmemi_team_destroy_nvls(nvshmemi_team_t *team) {
    if (team->nvls_rsc == nullptr) return; /* NOOP */

    nvshmemi_nvls_rsc *nvls_obj = nullptr;
    nvls_obj = reinterpret_cast<nvshmemi_nvls_rsc *>(team->nvls_rsc);
    if (nvls_obj->get_refcount() == 0) { /* Last reference */
        nvshmemi_state->heap_obj->nvls_unmap_heap_memory_by_team(team);
        nvshmemi_state->heap_obj->nvls_unbind_heap_memory_by_team(team);
        nvls_obj->free_group_mem();
        nvls_obj->release_owner();
        delete nvls_obj;
        cudaFree(team->nvls_rsc_base_ptr);
        team->nvls_rsc = nullptr;
        INFO(NVSHMEM_TEAM, "NVLS Resource Destroyed for Team ID %d\n", team->team_idx);
    } else {
        nvls_obj->del_refcount(); /* Shared nvls resource */
        /* Ownership of NVLS resource is necessary to allow for newly allocated UC memory to be
         * bound and mapped to MC heap */
        if (nvls_obj->is_owner(team)) {
            // Transfer ownership to one of the dup teams
            NVSHMEMU_FOR_EACH_IF(
                i, nvshmemi_max_teams,
                nvshmemi_team_pool[i] != NULL && nvshmemi_team_support_nvls(nvshmemi_team_pool[i]),
                {
                    // Find first duplicate team that shares the nvls rsc and make it the owner
                    if (nvshmemi_team_pool[i]->nvls_rsc == team->nvls_rsc) {
                        nvls_obj->release_owner();
                        nvls_obj->assign_owner(nvshmemi_team_pool[i]);
                        break;
                    }
                })
        }
    }
}

static int nvshmemi_team_create_nvls(nvshmemi_team_t *team) {
    int status = -1;
    uint64_t mc_heap_base;
    nvshmemi_nvls_rsc *nvls_obj = nullptr;

    if (!nvshmemi_team_is_nvls_capable(team)) {
        team->nvls_rsc = nullptr;
        WARN("Skipping NVLINK SHARP resource initialized for team ID: %d\n", team->team_idx);
        return 0;
    }

    try {
        team->nvls_rsc = reinterpret_cast<void *>(new nvshmemi_nvls_rsc(team, nvshmemi_state));
    } catch (nvshmemi_nvls_exception &exp) {
        WARN("NVLINK SHARP resource initialization failed for team ID: %d\n", team->team_idx);
        team->nvls_rsc = nullptr;
        return 0;
    }

    nvls_obj = reinterpret_cast<nvshmemi_nvls_rsc *>(team->nvls_rsc);
    status = nvls_obj->reserve_group_mem();
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, cleanup,
                          "Reserve multicast group mapping failed for pe %d\n", team->my_pe);

    nvls_obj->assign_owner(team);
    CUDA_RUNTIME_CHECK(cudaMalloc((void **)&team->nvls_rsc_base_ptr, sizeof(void *)));
    mc_heap_base = (uint64_t)(nvls_obj->get_mc_base());
    CUDA_RUNTIME_CHECK(
        cudaMemcpy(team->nvls_rsc_base_ptr, &mc_heap_base, sizeof(void *), cudaMemcpyHostToDevice));

    INFO(NVSHMEM_TEAM, "NVLS Resource Created for Team ID %d MC VA Base: %llx\n", team->team_idx,
         mc_heap_base);
    return (status);

cleanup:
    (void)nvls_obj->free_group_mem();
    delete nvls_obj;
    team->nvls_rsc = nullptr;
    return (status);
}

static int nvshmemi_team_bind_nvls(nvshmemi_team_t *team) {
    int status = -1;
    /* Make a MC handle as large as physical heap size */
    nvshmemi_nvls_rsc *nvls_obj = nullptr;
    nvls_obj = reinterpret_cast<nvshmemi_nvls_rsc *>(team->nvls_rsc);
    if (!nvls_obj->is_owner(team)) return 0;

    status = nvshmemi_state->heap_obj->nvls_create_heap_memory_by_team(team);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, cleanup,
                          "Create multicast groups for UC heap failed for pe %d team ID %d\n",
                          team->my_pe, team->team_idx);

    status = nvshmemi_state->heap_obj->nvls_bind_heap_memory_by_team(team);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, cleanup,
                          "Binding multicast groups to UC heap failed for pe %d team ID %d\n",
                          team->my_pe, team->team_idx);
    status = nvshmemi_state->heap_obj->nvls_map_heap_memory_by_team(team);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, cleanup,
                          "Mapping multicast groups for UC heap failed for pe %d team ID %d\n",
                          team->my_pe, team->team_idx);
cleanup:
    return (status);
}

static int nvshmemi_team_nvls_setup(nvshmemi_team_t *team) {
    int status = 0;

    /* Initialize NVLS resources for team supporting P2P connected GPUs */
    status = nvshmemi_team_create_nvls(team);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, cleanup,
                          "NVLS resource initialization failed for team ID: %d\n", team->team_idx);

    /* Any prior UC allocations need to bound to this team's MC groups */
    if (team->nvls_rsc != nullptr) {
        status = nvshmemi_team_bind_nvls(team);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, cleanup,
                              "NVLS resource bind and mapping existing UC mappings to MC heap "
                              "failed for team ID: %d\n",
                              team->team_idx);
    }

cleanup:
    return (status);
}

/* Team Management Routines */
int nvshmemi_set_max_teams(void) {
    nvshmemi_max_teams = nvshmemi_options.MAX_TEAMS;
    if (nvshmemi_max_teams < NVSHMEM_TEAMS_MIN) nvshmemi_max_teams = NVSHMEM_TEAMS_MIN;

    if (nvshmemi_max_teams > N_PSYNC_BYTES * CHAR_BIT) {
        NVSHMEMI_ERROR_EXIT("Requested %ld teams, but only %d are supported\n", nvshmemi_max_teams,
                            N_PSYNC_BYTES * CHAR_BIT);
        return 1;
    }
    return 0;
}

int nvshmemi_team_init(void) {
    long psync_len;
    int start, stride, size;
    int *scratch = NULL;
    int status = 0;
    uint64_t *hostHash = NULL;
    uint64_t myHostHash = 0;
    nvshmem_transport_pe_info_t *pe_info;
    int i;

    nvshmemi_team_world = NVSHMEMI_TEAM_INITIALIZER;
    nvshmemi_team_shared = NVSHMEMI_TEAM_INITIALIZER;
    nvshmemi_team_node = NVSHMEMI_TEAM_INITIALIZER;
    nvshmemi_team_same_mype_node = NVSHMEMI_TEAM_INITIALIZER;
    nvshmemi_team_same_gpu = NVSHMEMI_TEAM_INITIALIZER;
    nvshmemi_team_gpu_leaders = NVSHMEMI_TEAM_INITIALIZER;

    /* Initialize NVSHMEM_TEAM_WORLD */
    nvshmemi_team_world.team_idx = NVSHMEM_TEAM_WORLD_INDEX;
    NVSHMEMI_TEAM_DUP_INITIALIZER(nvshmemi_team_world, NVSHMEM_TEAM_WORLD_INDEX);
    nvshmemi_team_world.start = 0;
    nvshmemi_team_world.stride = 1;
    nvshmemi_team_world.size = nvshmemi_state->npes;
    nvshmemi_team_world.my_pe = nvshmemi_state->mype;
    nvshmemi_team_world.rdxn_count = 0;
    nvshmemi_team_world.config_mask = 0;
    nvshmemi_team_world.ll_flag = 1;
    nvshmemi_team_world.alltoall_count = 0;
    nvshmemi_team_world.bcast_count = 0;
    nvshmemi_team_world.bcast_sync_offset = 0;
    nvshmemi_team_world.fcollect_count = 0;
    nvshmemi_team_set_p2p_connectivity(&nvshmemi_team_world);
    nvshmemi_recexchalgo_get_neighbors(&nvshmemi_team_world);
    nvshmemi_team_world.is_team_node = false;
    nvshmemi_team_world.is_team_same_mype_node = false;

    /* Initialize NVSHMEM_TEAM_SHARED */
    nvshmemi_team_shared.team_idx = NVSHMEM_TEAM_SHARED_INDEX;
    NVSHMEMI_TEAM_DUP_INITIALIZER(nvshmemi_team_shared, NVSHMEM_TEAM_SHARED_INDEX);
    /* Collect list of p2p connected PEs */
    int *p2p_pe_list = (int *)malloc(nvshmemi_team_world.size * sizeof(int));
    int n_p2p_pes = 0;
    int my_idx_in_p2p_list = 0;
    for (int i = 0; i < nvshmemi_team_world.size; i++) {
        if (nvshmemi_state->heap_obj->get_local_pe_base()[i]) {
            if (i == nvshmemi_team_world.my_pe) my_idx_in_p2p_list = n_p2p_pes;
            p2p_pe_list[n_p2p_pes++] = i;
        }
    }

    std::ostringstream ss;
    for (int i = 0; i < n_p2p_pes; i++) {
        ss << p2p_pe_list[i] << " ";
    }
    INFO(NVSHMEM_INIT, "P2P list: %s", ss.str().c_str());

    /* Make sure that n_p2p_pes is same for all PEs to form TEAM_SHARED */
    int *n_p2p_pes_all = (int *)malloc(nvshmemi_team_world.size * sizeof(int));
    int *p2p_pe_list_all = (int *)malloc(sizeof(int) * n_p2p_pes * nvshmemi_team_world.size);

    nvshmemi_boot_handle.allgather((void *)&n_p2p_pes, (void *)n_p2p_pes_all, sizeof(int),
                                   &nvshmemi_boot_handle);

    for (i = 0; i < nvshmemi_team_world.size; i++) {
        if (n_p2p_pes_all[i] != n_p2p_pes) {
            INFO(NVSHMEM_INIT,
                 "n_p2p_pes is not equal across PEs, setting NVSHMEM_TEAM_SHARED to self");
            goto team_shared_single_pe;
        }
    }

    /* Gather p2p lists of all PEs and ensure they are the same */
    nvshmemi_boot_handle.allgather((void *)p2p_pe_list, (void *)p2p_pe_list_all,
                                   sizeof(int) * n_p2p_pes, &nvshmemi_boot_handle);
    for (i = 0; i < n_p2p_pes; i++) {
        if (memcmp((void *)p2p_pe_list, (void *)&p2p_pe_list_all[p2p_pe_list[i] * n_p2p_pes],
                   sizeof(int) * n_p2p_pes) != 0) {
            INFO(NVSHMEM_INIT, "P2P lists are not symmetric, setting NVSHMEM_TEAM_SHARED to self");
            goto team_shared_single_pe;
        }
    }

    for (int i = 2; i < n_p2p_pes; i++) {
        if (p2p_pe_list[i] - p2p_pe_list[i - 1] != p2p_pe_list[i - 1] - p2p_pe_list[i - 2]) {
            INFO(NVSHMEM_INIT,
                 "P2P list is not of the form (start, stride, size). Cannot form "
                 "NVSHMEM_TEAM_SHARED.");
            goto team_shared_single_pe;
        }
    }

    nvshmemi_team_shared.my_pe = my_idx_in_p2p_list;
    nvshmemi_team_shared.start = p2p_pe_list[0];
    nvshmemi_team_shared.stride = n_p2p_pes > 1 ? (p2p_pe_list[1] - p2p_pe_list[0]) : 1;
    nvshmemi_team_shared.size = n_p2p_pes;

    goto team_shared_setup;

team_shared_single_pe:
    nvshmemi_team_shared.my_pe = 0;
    nvshmemi_team_shared.start = nvshmemi_state->mype;
    nvshmemi_team_shared.stride = 1;
    nvshmemi_team_shared.size = 1;
    nvshmemi_team_shared.is_team_node = true;
    nvshmemi_team_shared.is_team_same_mype_node = true;

team_shared_setup:
    free(n_p2p_pes_all);
    free(p2p_pe_list_all);
    free(p2p_pe_list);

    nvshmemi_team_shared.rdxn_count = 0;
    nvshmemi_team_shared.config_mask = 0;

    nvshmemi_team_shared.ll_flag = 1;
    nvshmemi_team_shared.alltoall_count = 0;
    nvshmemi_team_shared.bcast_count = 0;
    nvshmemi_team_shared.bcast_sync_offset = 0;
    nvshmemi_team_shared.fcollect_count = 0;
    nvshmemi_team_shared.are_gpus_p2p_connected = 0;
    nvshmemi_team_set_p2p_connectivity(&nvshmemi_team_shared);
    nvshmemi_recexchalgo_get_neighbors(&nvshmemi_team_shared);
    INFO(NVSHMEM_INIT, "NVSHMEM_TEAM_SHARED: start=%d, stride=%d, size=%d",
         nvshmemi_team_shared.start, nvshmemi_team_shared.stride, nvshmemi_team_shared.size);
    nvshmemi_team_shared.is_team_same_mype_node = false;

    /* Initialize NVSHMEMX_TEAM_NODE */
    nvshmemi_team_node.team_idx = NVSHMEM_TEAM_NODE_INDEX;
    NVSHMEMI_TEAM_DUP_INITIALIZER(nvshmemi_team_node, NVSHMEM_TEAM_NODE_INDEX);
    nvshmemi_team_world.team_node = nvshmemi_team_node.team_idx;
    nvshmemi_team_node.my_pe = nvshmemi_state->mype_node;
    nvshmemi_team_node.rdxn_count = 0;
    nvshmemi_team_node.config_mask = 0;
    nvshmemi_team_node.ll_flag = 1;
    nvshmemi_team_node.alltoall_count = 0;
    nvshmemi_team_node.bcast_count = 0;
    nvshmemi_team_node.bcast_sync_offset = 0;
    nvshmemi_team_node.fcollect_count = 0;
    myHostHash = getHostHash();
    hostHash = (uint64_t *)malloc(sizeof(uint64_t) * nvshmemi_state->npes);
    NVSHMEMI_NULL_ERROR_JMP(hostHash, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, cleanup,
                            "hostHash allocation failed \n");
    status = nvshmemi_boot_handle.allgather((void *)&myHostHash, (void *)hostHash, sizeof(uint64_t),
                                            &nvshmemi_boot_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, cleanup,
                          "allgather of host hashes failed\n");

    /* Search for on-node peer PEs while checking for a consistent stride */
    start = -1;
    stride = -1;
    size = 0;

    for (int pe = 0; pe < nvshmemi_state->npes; pe++) {
        if (hostHash[pe] != myHostHash) continue;

        int ret = check_for_linear_stride(pe, &start, &stride, &size);
        if (ret < 0) {
            start = nvshmemi_state->mype;
            stride = 1;
            size = 1;
            break;
        }
    }
    assert(start >= 0 && size > 0);
    nvshmemi_team_node.start = start;
    nvshmemi_team_node.stride = (stride == -1) ? 1 : stride;
    nvshmemi_team_node.size = size;
    if (nvshmemi_team_is_identical(&nvshmemi_team_world, &nvshmemi_team_node)) {
        nvshmemi_team_world.is_team_node = true;
    }
    if (nvshmemi_team_is_identical(&nvshmemi_team_shared, &nvshmemi_team_node)) {
        nvshmemi_team_shared.is_team_node = true;
    }

    nvshmemi_team_set_p2p_connectivity(&nvshmemi_team_node);
    nvshmemi_recexchalgo_get_neighbors(&nvshmemi_team_node);
    nvshmemi_team_node.is_team_node = true;
    nvshmemi_team_node.is_team_same_mype_node = false;
    INFO(NVSHMEM_INIT, "NVSHMEMX_TEAM_NODE: start=%d, stride=%d, size=%d", nvshmemi_team_node.start,
         nvshmemi_team_node.stride, nvshmemi_team_node.size);

    /* Initialize NVSHMEMX_TEAM_SAME_MYPE_NODE */
    nvshmemi_team_same_mype_node.team_idx = NVSHMEM_TEAM_SAME_MYPE_NODE_INDEX;
    NVSHMEMI_TEAM_DUP_INITIALIZER(nvshmemi_team_same_mype_node, NVSHMEM_TEAM_SAME_MYPE_NODE_INDEX);
    nvshmemi_team_world.team_same_mype_node = nvshmemi_team_same_mype_node.team_idx;
    nvshmemi_team_same_mype_node.my_pe = nvshmemi_state->mype / nvshmemi_state->npes_node;
    nvshmemi_team_same_mype_node.rdxn_count = 0;
    nvshmemi_team_same_mype_node.config_mask = 0;

    nvshmemi_team_same_mype_node.start = nvshmemi_state->mype_node;
    nvshmemi_team_same_mype_node.stride = nvshmemi_state->npes_node;
    nvshmemi_team_same_mype_node.size = nvshmemi_state->npes / nvshmemi_state->npes_node;
    assert(nvshmemi_state->npes % nvshmemi_state->npes_node == 0);
    nvshmemi_team_same_mype_node.ll_flag = 1;
    nvshmemi_team_same_mype_node.alltoall_count = 0;
    nvshmemi_team_same_mype_node.bcast_count = 0;
    nvshmemi_team_same_mype_node.bcast_sync_offset = 0;
    nvshmemi_team_same_mype_node.fcollect_count = 0;
    nvshmemi_team_set_p2p_connectivity(&nvshmemi_team_same_mype_node);
    nvshmemi_recexchalgo_get_neighbors(&nvshmemi_team_same_mype_node);
    nvshmemi_team_same_mype_node.is_team_node = false;
    nvshmemi_team_same_mype_node.is_team_same_mype_node = true;
    INFO(NVSHMEM_INIT, "NVSHMEMX_TEAM_SAME_MYPE_NODE: start=%d, stride=%d, size=%d",
         nvshmemi_team_same_mype_node.start, nvshmemi_team_same_mype_node.stride,
         nvshmemi_team_same_mype_node.size);

    /* Initialize team NVSHMEMI_TEAM_SAME_GPU */
    nvshmemi_team_same_gpu.team_idx = NVSHMEM_TEAM_SAME_GPU_INDEX;
    NVSHMEMI_TEAM_DUP_INITIALIZER(nvshmemi_team_same_gpu, NVSHMEM_TEAM_SAME_GPU_INDEX);
    nvshmemi_team_same_gpu.rdxn_count = 0;
    nvshmemi_team_same_gpu.ll_flag = 1;
    nvshmemi_team_same_gpu.alltoall_count = 0;
    nvshmemi_team_same_gpu.bcast_count = 0;
    nvshmemi_team_same_gpu.bcast_sync_offset = 0;
    nvshmemi_team_same_gpu.fcollect_count = 0;
    nvshmemi_team_same_gpu.config_mask = 0;
    pe_info = nvshmemi_state->pe_info;
    start = -1;
    stride = -1;
    size = 0;
    for (int pe = 0; pe < nvshmemi_state->npes; pe++) {
        if (pe_info[pe].hostHash != pe_info[nvshmemi_state->mype].hostHash ||
            memcmp(&pe_info[pe].gpu_uuid, &pe_info[nvshmemi_state->mype].gpu_uuid,
                   sizeof(cudaUUID_t)) != 0)
            continue;

        int ret = check_for_linear_stride(pe, &start, &stride, &size);
        if (ret < 0) {
            NVSHMEMI_ERROR_EXIT("Could not form NVSHMEMI_TEAM_SAME_GPU\n");
            break;
        }
    }
    assert(start >= 0 && size > 0);
    nvshmemi_team_same_gpu.my_pe = (nvshmemi_state->mype - start) / stride;
    nvshmemi_team_same_gpu.start = start;
    nvshmemi_team_same_gpu.stride = (stride == -1) ? 1 : stride;
    nvshmemi_team_same_gpu.size = size;
    nvshmemi_team_set_p2p_connectivity(&nvshmemi_team_same_gpu);
    nvshmemi_recexchalgo_get_neighbors(&nvshmemi_team_same_gpu);
    nvshmemi_team_same_gpu.is_team_node = true;
    nvshmemi_team_same_gpu.is_team_same_mype_node = false;
    INFO(NVSHMEM_INIT, "NVSHMEMI_TEAM_SAME_GPU: start=%d, stride=%d, size=%d",
         nvshmemi_team_same_gpu.start, nvshmemi_team_same_gpu.stride, nvshmemi_team_same_gpu.size);

    /* All GPUs must have same number of processes (requires for us to form teams) */

    /* Initialize team NVSHMEMI_TEAM_GPU_LEADERS */
    scratch = (int *)malloc(sizeof(int) * nvshmemi_state->npes);
    NVSHMEMI_NULL_ERROR_JMP(scratch, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, cleanup,
                            "Unable to allocate host memory for team creation.\n");
    if (nvshmemi_team_same_gpu.start ==
        nvshmemi_state->mype) { /* Only GPU leaders are part of this team */
        nvshmemi_team_gpu_leaders.team_idx = NVSHMEM_TEAM_GPU_LEADERS_INDEX;
        NVSHMEMI_TEAM_DUP_INITIALIZER(nvshmemi_team_gpu_leaders, NVSHMEM_TEAM_GPU_LEADERS_INDEX);
        nvshmemi_team_gpu_leaders.config_mask = 0;

        nvshmemi_team_gpu_leaders.start = 0;
        nvshmemi_team_gpu_leaders.stride =
            (nvshmemi_team_same_gpu.stride == 1) ? nvshmemi_team_same_gpu.size : 1;
        nvshmemi_team_gpu_leaders.size = nvshmemi_state->npes / nvshmemi_team_same_gpu.size;
        nvshmemi_team_gpu_leaders.my_pe = (nvshmemi_state->mype - nvshmemi_team_gpu_leaders.start) /
                                          nvshmemi_team_gpu_leaders.stride;
        nvshmemi_team_gpu_leaders.rdxn_count = 0;
        nvshmemi_team_gpu_leaders.ll_flag = 1;
        nvshmemi_team_gpu_leaders.alltoall_count = 0;
        nvshmemi_team_gpu_leaders.bcast_count = 0;
        nvshmemi_team_gpu_leaders.bcast_sync_offset = 0;
        nvshmemi_team_gpu_leaders.fcollect_count = 0;
        nvshmemi_team_set_p2p_connectivity(&nvshmemi_team_gpu_leaders);
        nvshmemi_recexchalgo_get_neighbors(&nvshmemi_team_gpu_leaders);
        status =
            nvshmemi_boot_handle.allgather((void *)&nvshmemi_team_gpu_leaders.my_pe,
                                           (void *)scratch, sizeof(int), &nvshmemi_boot_handle);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, cleanup,
                              "allgather of gpu leaders failed\n");
        /* Check whether a valid TEAM_GPU_LEADERS was formed */
        int last_mype = -1;
        for (int i = 0; i < nvshmemi_state->npes; i++) {
            if (scratch[i] != -1) {
                if (scratch[i] != last_mype + 1) {
                    WARN(
                        "NVSHMEMI_TEAM_GPU_LEADERS could not be formed, Limited MPG support will "
                        "not be available\n");
                    break;
                } else {
                    last_mype++;
                }
            }
        }
        /* XXX: Note that we are not setting team_node and team_same_mype_node for
         * nvshmemi_team_gpu_leaders */
        nvshmemi_team_gpu_leaders.is_team_node = false;
        nvshmemi_team_gpu_leaders.is_team_same_mype_node = false;
        INFO(NVSHMEM_INIT, "NVSHMEMI_TEAM_GPU_LEADERS: start=%d, stride=%d, size=%d",
             nvshmemi_team_gpu_leaders.start, nvshmemi_team_gpu_leaders.stride,
             nvshmemi_team_gpu_leaders.size);
    } else {
        int my_pe = -1;
        status = nvshmemi_boot_handle.allgather((void *)&my_pe, (void *)scratch, sizeof(int),
                                                &nvshmemi_boot_handle);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, cleanup,
                              "allgather of gpu leaders failed\n");
    }
    if (nvshmemi_max_teams < NVSHMEM_TEAMS_MIN) nvshmemi_max_teams = NVSHMEM_TEAMS_MIN;

    if (nvshmemi_max_teams > N_PSYNC_BYTES * CHAR_BIT) {
        NVSHMEMI_ERROR_EXIT("Requested %ld teams, but only %d are supported\n", nvshmemi_max_teams,
                            N_PSYNC_BYTES * CHAR_BIT);
        goto cleanup;
    }

    nvshmemi_team_pool = (nvshmemi_team_t **)calloc(nvshmemi_max_teams, sizeof(nvshmemi_team_t *));
    NVSHMEMI_NULL_ERROR_JMP(nvshmemi_team_pool, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, cleanup,
                            "nvshmemi_team_pool allocation failed \n");
    CUDA_RUNTIME_CHECK(cudaMalloc((void **)&nvshmemi_device_team_pool,
                                  nvshmemi_max_teams * sizeof(nvshmemi_team_t *)));
    nvshmemi_device_state.team_pool = nvshmemi_device_team_pool;

    for (long i = 0; i < nvshmemi_max_teams; i++) {
        nvshmemi_team_pool[i] = NULL;
    }

    nvshmemi_call_init_array_kernel<nvshmemi_team_t *>(nvshmemi_device_team_pool,
                                                       nvshmemi_max_teams, NULL);

    nvshmemi_team_pool[NVSHMEM_TEAM_WORLD_INDEX] = &nvshmemi_team_world;
    nvshmemi_team_pool[NVSHMEM_TEAM_SHARED_INDEX] = &nvshmemi_team_shared;
    nvshmemi_team_pool[NVSHMEM_TEAM_NODE_INDEX] = &nvshmemi_team_node;
    nvshmemi_team_pool[NVSHMEM_TEAM_SAME_MYPE_NODE_INDEX] = &nvshmemi_team_same_mype_node;
    nvshmemi_team_pool[NVSHMEM_TEAM_SAME_GPU_INDEX] = &nvshmemi_team_same_gpu;
    if (nvshmemi_team_same_gpu.start == nvshmemi_state->mype)
        nvshmemi_team_pool[NVSHMEM_TEAM_GPU_LEADERS_INDEX] = &nvshmemi_team_gpu_leaders;

    /* Allocate pSync pool, each with the maximum possible size requirement */
    /* Create two pSyncs per team for back-to-back collectives and one for barriers.
     * Array organization:
     *
     * [ (world) (shared) (team 1) (team 2) ...  (world) (shared) (team 1) (team 2) ... ]
     *  <----------- groups 1 & 2-------------->|<------------- group 3 ---------------->
     *  <--- (bcast, collect, reduce, etc.) --->|<------ (barriers and syncs) ---------->
     * */
    psync_len = nvshmemi_max_teams * get_psync_len_per_team();
    nvshmemi_psync_pool = (long *)nvshmemi_malloc(sizeof(long) * psync_len);
    NVSHMEMI_NULL_ERROR_JMP(nvshmemi_psync_pool, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, cleanup,
                            "nvshmemi_psync_pool allocation failed \n");

    nvshmemi_device_state.psync_pool = nvshmemi_psync_pool;

    nvshmemi_call_init_array_kernel<long>(nvshmemi_psync_pool, psync_len, NVSHMEMI_SYNC_VALUE);

    nvshmemi_sync_counter = (long *)nvshmemi_malloc(2 * nvshmemi_max_teams * sizeof(long));
    NVSHMEMI_NULL_ERROR_JMP(nvshmemi_sync_counter, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, cleanup,
                            "nvshmemi_sync_counter allocation failed \n");

    nvshmemi_device_state.sync_counter = nvshmemi_sync_counter;
    nvshmemi_update_device_state();

    nvshmemi_call_init_array_kernel<long>(nvshmemi_sync_counter, 2 * nvshmemi_max_teams, 1);

    /* Convenience pointer to the group-3 pSync array (for barriers and syncs): */
    psync_pool_avail = (unsigned char *)malloc(2 * N_PSYNC_BYTES);
    NVSHMEMI_NULL_ERROR_JMP(psync_pool_avail, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, cleanup,
                            "psync_pool_avail allocation failed \n");
    psync_pool_avail_reduced = &psync_pool_avail[N_PSYNC_BYTES];

    device_psync_pool_avail = (unsigned char *)nvshmemi_malloc(2 * N_PSYNC_BYTES);
    NVSHMEMI_NULL_ERROR_JMP(device_psync_pool_avail, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, cleanup,
                            "device_psync_pool_avail allocation failed \n");
    device_psync_pool_avail_reduced = &device_psync_pool_avail[N_PSYNC_BYTES];
    /* Initialize the psync bits to 1, making all slots available: */
    memset(psync_pool_avail, 0, 2 * N_PSYNC_BYTES);
    for (size_t i = 0; i < (size_t)nvshmemi_max_teams; i++) {
        nvshmemi_bit_set(psync_pool_avail, N_PSYNC_BYTES, i);
    }

    /* Set the bits for NVSHMEM_TEAM_WORLD, NVSHMEM_TEAM_SHARED, NVSHMEMX_TEAM_NODE to 0: */
    nvshmemi_bit_clear(psync_pool_avail, N_PSYNC_BYTES, NVSHMEM_TEAM_WORLD_INDEX);
    nvshmemi_bit_clear(psync_pool_avail, N_PSYNC_BYTES, NVSHMEM_TEAM_SHARED_INDEX);
    nvshmemi_bit_clear(psync_pool_avail, N_PSYNC_BYTES, NVSHMEM_TEAM_NODE_INDEX);
    nvshmemi_bit_clear(psync_pool_avail, N_PSYNC_BYTES, NVSHMEM_TEAM_SAME_MYPE_NODE_INDEX);
    nvshmemi_bit_clear(psync_pool_avail, N_PSYNC_BYTES, NVSHMEM_TEAM_SAME_GPU_INDEX);
    nvshmemi_bit_clear(psync_pool_avail, N_PSYNC_BYTES, NVSHMEM_TEAM_GPU_LEADERS_INDEX);

    /* Initialize an integer used to agree on an equal return value across PEs in team creation: */
    team_ret_val = (int *)malloc(sizeof(int) * 2);
    NVSHMEMI_NULL_ERROR_JMP(team_ret_val, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, cleanup,
                            "team_ret_val allocation failed \n");
    team_ret_val_reduced = &team_ret_val[1];

    device_team_ret_val = (int *)nvshmemi_malloc(sizeof(int) * 2);
    NVSHMEMI_NULL_ERROR_JMP(team_ret_val, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, cleanup,
                            "device_team_ret_val allocation failed \n");
    device_team_ret_val_reduced = &device_team_ret_val[1];

    nvshmemi_boot_handle.barrier(
        &nvshmemi_boot_handle); /* To ensure neccessary setup has been done all PEs */

    nvshmemi_team_alloc_device();
    nvshmemi_team_update_device();
    nvshmemi_boot_handle.barrier(
        &nvshmemi_boot_handle); /* To ensure neccessary setup has been done all PEs */

#ifdef NVSHMEM_USE_NCCL
    if (nvshmemi_use_nccl) {
        /* Setup NCCL usage */
        nvshmemi_team_init_nccl_comm(&nvshmemi_team_world);
        nvshmemi_team_init_nccl_comm(&nvshmemi_team_shared);
        nvshmemi_team_init_nccl_comm(&nvshmemi_team_node);
        nvshmemi_team_init_nccl_comm(&nvshmemi_team_same_mype_node);
        nvshmemi_team_init_nccl_comm(&nvshmemi_team_same_gpu);
        if (nvshmemi_pe_in_active_set(nvshmemi_state->mype, nvshmemi_team_gpu_leaders.start,
                                      nvshmemi_team_gpu_leaders.stride,
                                      nvshmemi_team_gpu_leaders.size) >= 0) {
            nvshmemi_team_init_nccl_comm(&nvshmemi_team_gpu_leaders);
        }
    }
#endif /* NVSHMEM_USE_NCCL */

    /* Setup NVLS resources for all internal p2p connected teams */
    NVSHMEMU_FOR_EACH_IF(
        i, nvshmemi_max_teams,
        nvshmemi_team_pool[i] != NULL && nvshmemi_team_pool[i]->are_gpus_p2p_connected, {
            status = nvshmemi_team_nvls_setup(nvshmemi_team_pool[i]);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, cleanup,
                                  "NVLS resource setup failed for team ID: %d\n",
                                  nvshmemi_team_pool[i]->team_idx);
            if (nvshmemi_team_pool[i]->nvls_rsc) {
                INFO(NVSHMEM_TEAM, "Successful NVLS resource setup for team ID: %d\n",
                     nvshmemi_team_pool[i]->team_idx);
            }
        })

    nvshmemi_boot_handle.barrier(
        &nvshmemi_boot_handle); /* To ensure neccessary setup has been done all PEs */

    nvshmemi_team_update_device();

    nvshmemi_boot_handle.barrier(
        &nvshmemi_boot_handle); /* To ensure neccessary setup has been done all PEs */

#if defined(NVSHMEM_PPC64LE)
    if (nvshmemi_use_nccl) {
        /* Set GPU thread stack size to be max stack size of any kernel invoked by NCCL.
           The value 1256 has been obtained by profiling all NCCL kernels in NCCL 2.8.3-1.
           This value is being set to prevent any memory config during application run
           as that can lead to potential deadlock */
        if (nvshmemi_options.CUDA_LIMIT_STACK_SIZE_provided) {
            CUDA_RUNTIME_CHECK(
                cudaDeviceSetLimit(cudaLimitStackSize, nvshmemi_options.CUDA_LIMIT_STACK_SIZE));
            if (nvshmemi_options.CUDA_LIMIT_STACK_SIZE < 1256)
                NVSHMEMI_WARN_PRINT(
                    "CUDA stack size limit has been set to less than 1256.\n"
                    "This can lead to hangs because a NCCL kernel can need up\n"
                    "to 1256 bytes");
        } else
            CUDA_RUNTIME_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 1256));
    } else if (nvshmemi_options.CUDA_LIMIT_STACK_SIZE_provided) {
        CUDA_RUNTIME_CHECK(
            cudaDeviceSetLimit(cudaLimitStackSize, nvshmemi_options.CUDA_LIMIT_STACK_SIZE));
    }
#endif

cleanup:
    if (scratch) {
        free(scratch);
    }
    if (hostHash) {
        free(hostHash);
    }

    if (status != NVSHMEMX_SUCCESS) {
        if (nvshmemi_team_pool) {
            free(nvshmemi_team_pool);
            nvshmemi_team_pool = NULL;
            cudaFree(nvshmemi_device_team_pool);
            nvshmemi_device_team_pool = NULL;
        }
        if (nvshmemi_psync_pool) {
            nvshmemi_free(nvshmemi_psync_pool);
            nvshmemi_psync_pool = NULL;
        }
        if (psync_pool_avail) {
            free(psync_pool_avail);
            psync_pool_avail = NULL;
        }
        if (device_psync_pool_avail) {
            nvshmemi_free(device_psync_pool_avail);
            device_psync_pool_avail = NULL;
        }
        if (team_ret_val) {
            free(team_ret_val);
            team_ret_val = NULL;
        }
        if (device_team_ret_val) {
            nvshmemi_free(device_team_ret_val);
            device_team_ret_val = NULL;
        }
    }

    return status;
}

int nvshmemi_team_finalize(void) {
    /* Destroy all undestroyed teams */
    for (long i = 0; i < nvshmemi_max_teams; i++) {
        if (nvshmemi_team_pool[i] != NULL) nvshmemi_team_destroy(nvshmemi_team_pool[i]);
    }

    free(nvshmemi_team_pool);
    nvshmemi_team_pool = NULL;
    CUDA_RUNTIME_CHECK(cudaFree(nvshmemi_device_team_pool));

    nvshmemi_free(nvshmemi_psync_pool);
    nvshmemi_free(nvshmemi_sync_counter);

    free(psync_pool_avail);
    nvshmemi_free(device_psync_pool_avail);
    free(team_ret_val);
    nvshmemi_free(device_team_ret_val);
    cudaFree(nvshmemi_device_team_world);
    cudaFree(nvshmemi_device_team_shared);
    cudaFree(nvshmemi_device_team_node);
    cudaFree(nvshmemi_device_team_same_mype_node);
    cudaFree(nvshmemi_device_team_same_gpu);
    cudaFree(nvshmemi_device_team_gpu_leaders);

    return 0;
}

int nvshmemi_team_split_strided(nvshmemi_team_t *parent_team, int PE_start, int PE_stride,
                                int PE_size, const nvshmem_team_config_t *config, long config_mask,
                                nvshmem_team_t *new_team) {
    *new_team = NVSHMEM_TEAM_INVALID;
    nvshmem_barrier(parent_team->team_idx);

    int global_PE_start = nvshmemi_team_pe(parent_team, PE_start);
    int global_PE_stride = parent_team->stride * PE_stride;
    int global_PE_end = global_PE_start + global_PE_stride * (PE_size - 1);

    if (PE_start < 0 || PE_start >= parent_team->size || PE_size <= 0 ||
        PE_size > parent_team->size || PE_stride < 1) {
        NVSHMEMI_WARN_PRINT(
            "Invalid <start, stride, size>: child <%d, %d, %d>, parent <%d, %d, %d>\n", PE_start,
            PE_stride, PE_size, parent_team->start, parent_team->stride, parent_team->size);
        return -1;
    }

    if (global_PE_start >= nvshmemi_state->npes || global_PE_end >= nvshmemi_state->npes) {
        NVSHMEMI_WARN_PRINT("Starting PE (%d) or ending PE (%d) is invalid\n", global_PE_start,
                            global_PE_end);
        return -1;
    }

    /* idx in new team: */
    int my_pe =
        nvshmemi_pe_in_active_set(nvshmemi_state->mype, global_PE_start, global_PE_stride, PE_size);

    long *psync_reduce = nvshmemi_team_get_psync(parent_team, REDUCE);
    long *psync = &nvshmemi_team_get_psync(parent_team, SYNC)[NVSHMEMI_SYNC_SIZE];
    long *sync_counter = &nvshmemi_team_get_sync_counter(parent_team)[1];
    nvshmemi_team_t *myteam = NULL;
    *team_ret_val = 0;
    *team_ret_val_reduced = 0;

    if (my_pe >= 0) {
        char bit_str[NVSHMEMI_DIAG_STRLEN];

        myteam = (nvshmemi_team_t *)calloc(1, sizeof(nvshmemi_team_t));
        (*myteam) = NVSHMEMI_TEAM_INITIALIZER;

        myteam->my_pe = my_pe;
        myteam->start = global_PE_start;
        myteam->stride = global_PE_stride;
        myteam->size = PE_size;
        myteam->rdxn_count = 0;
        myteam->ll_flag = 1;
        myteam->alltoall_count = 0;
        myteam->bcast_count = 0;
        myteam->bcast_sync_offset = 0;
        myteam->fcollect_count = 0;

        if (config) {
            myteam->config = *config;
            myteam->config_mask = config_mask;
        }
        myteam->team_idx = -1;
        nvshmemi_bit_to_string(bit_str, NVSHMEMI_DIAG_STRLEN, psync_pool_avail, N_PSYNC_BYTES);

        CUDA_RUNTIME_CHECK(cudaMemcpy(device_psync_pool_avail, psync_pool_avail, N_PSYNC_BYTES,
                                      cudaMemcpyHostToDevice));
        CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
        nvshmemi_call_reduce_kernel<unsigned char, RDXN_OPS_AND>(
            myteam->start, myteam->stride, myteam->size,
            (unsigned char *)device_psync_pool_avail_reduced,
            (const unsigned char *)device_psync_pool_avail, N_PSYNC_BYTES,
            (unsigned char *)psync_reduce, (long *)(psync), sync_counter);

        CUDA_RUNTIME_CHECK(cudaMemcpy(psync_pool_avail_reduced, device_psync_pool_avail_reduced,
                                      N_PSYNC_BYTES, cudaMemcpyDeviceToHost));

        /* We cannot release the psync here, because this reduction may not
         * have been performed on the entire parent team. */
        nvshmemi_bit_to_string(bit_str, NVSHMEMI_DIAG_STRLEN, psync_pool_avail_reduced,
                               N_PSYNC_BYTES);

        /* Select the least signficant nonzero bit, which corresponds to an available pSync. */
        myteam->team_idx = nvshmemi_bit_1st_nonzero(psync_pool_avail_reduced, N_PSYNC_BYTES);
        NVSHMEMI_TEAM_DUP_INITIALIZER(*myteam, myteam->team_idx);

        nvshmemi_bit_to_string(bit_str, NVSHMEMI_DIAG_STRLEN, psync_pool_avail_reduced,
                               N_PSYNC_BYTES);
        if (myteam->team_idx == -1 || myteam->team_idx >= (int)nvshmemi_max_teams) {
            NVSHMEMI_WARN_PRINT(
                "No more teams available (max = %ld), try setting NVSHMEM_MAX_TEAMS environment "
                "variable\n",
                nvshmemi_max_teams);
            /* No psync was available, but must call barrier across parent team before returning. */
            myteam->team_idx = -1;
            *team_ret_val = 1;
        } else {
            /* Set the selected psync bit to 0, reserving that slot */
            nvshmemi_bit_clear(psync_pool_avail, N_PSYNC_BYTES, myteam->team_idx);

            *new_team = myteam->team_idx;

            nvshmemi_team_pool[myteam->team_idx] = myteam;
            nvshmemi_team_t *device_team_addr;
            CUDA_RUNTIME_CHECK(cudaMalloc((void **)&device_team_addr, sizeof(nvshmemi_team_t)));
            nvshmemi_team_set_p2p_connectivity(myteam);
            nvshmemi_recexchalgo_get_neighbors(myteam);
            CUDA_RUNTIME_CHECK(cudaMemcpy(device_team_addr, myteam, sizeof(nvshmemi_team_t),
                                          cudaMemcpyHostToDevice));
            CUDA_RUNTIME_CHECK(cudaMemcpy(&nvshmemi_device_team_pool[myteam->team_idx],
                                          &device_team_addr, sizeof(nvshmemi_team_t *),
                                          cudaMemcpyHostToDevice));
            CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
#ifdef NVSHMEM_USE_NCCL
            if (nvshmemi_use_nccl) nvshmemi_team_init_nccl_comm(myteam);
#endif
            /*
             * Reuse NVLS resources if teams are identical,
             * else creating a new NVLS resources for p2p connected teams
             */
            if (nvshmemi_team_is_identical(myteam, parent_team) &&
                nvshmemi_team_is_nvls_capable(myteam) && !nvshmemi_options.DISABLE_NVLS_SHARING) {
                myteam->nvls_rsc = parent_team->nvls_rsc; /* Inherit the parent team's nvls_rsc */
                if (myteam->nvls_rsc) {
                    nvshmemi_nvls_rsc *nvls =
                        reinterpret_cast<nvshmemi_nvls_rsc *>(myteam->nvls_rsc);
                    nvls->add_refcount();
                    INFO(NVSHMEM_TEAM,
                         "Successful NVLS resource sharing for new team ID: %d (parent ID: %d)\n",
                         myteam->team_idx, parent_team->team_idx);
                }

            } else if (nvshmemi_team_is_nvls_capable(myteam)) {
                if (nvshmemi_team_nvls_setup(myteam) != 0) {
                    NVSHMEMI_WARN_PRINT("NVLS resource setup failed for team ID: %d\n",
                                        myteam->team_idx);
                    return -1;
                }

                INFO(NVSHMEM_TEAM, "Successful NVLS resource setup for team ID: %d\n",
                     myteam->team_idx);
            } else {
                myteam->nvls_rsc = nullptr; /* NVLS not supported, so no resource created/bound */
            }

            /* Build team_node */
            myteam->is_team_node = false;
            int i;
            for (i = 1; i < myteam->size; i++) {
                if (nvshmemi_host_hashes[myteam->start] !=
                    nvshmemi_host_hashes[myteam->start + i * myteam->stride]) {
                    break;
                }
            }
            if (i == myteam->size) myteam->is_team_node = true;

            myteam->is_team_same_mype_node = true;
            for (int i = 0; i < myteam->size; i++) {
                for (int j = i + 1; j < myteam->size; j++) {
                    if (nvshmemi_host_hashes[myteam->start + i * myteam->stride] ==
                        nvshmemi_host_hashes[myteam->start + j * myteam->stride]) {
                        myteam->is_team_same_mype_node = false;
                    }
                }
            }

            /* count PEs on the same node */
            int team_npes_node = 0;
            for (int i = 0; i < myteam->size; i++) {
                if (nvshmemi_team_translate_pe(myteam, i, &nvshmemi_team_node) != -1) {
                    team_npes_node++;
                }
            }
            if (!myteam->is_team_node && !myteam->is_team_same_mype_node) {
                /* Now I am just going to repurpose device_psync_pool_avail symm memory for the
                   purpose of finding max of team_npes_node */
                assert(sizeof(int) <= N_PSYNC_BYTES);
                CUDA_RUNTIME_CHECK(cudaMemcpy(device_psync_pool_avail, &team_npes_node, sizeof(int),
                                              cudaMemcpyHostToDevice));
                CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
                nvshmemi_call_reduce_kernel<int, RDXN_OPS_MAX>(
                    myteam->start, myteam->stride, myteam->size,
                    (int *)device_psync_pool_avail_reduced, (const int *)device_psync_pool_avail, 1,
                    (int *)psync_reduce, (long *)(psync), sync_counter);

                CUDA_RUNTIME_CHECK(cudaMemcpy(&team_npes_node, device_psync_pool_avail_reduced,
                                              sizeof(int), cudaMemcpyDeviceToHost));
                nvshmemi_team_split_2d(myteam, team_npes_node, NULL, 0, &myteam->team_node, NULL, 0,
                                       &myteam->team_same_mype_node);
            }
            CUDA_RUNTIME_CHECK(cudaMemcpy(device_team_addr, myteam, sizeof(nvshmemi_team_t),
                                          cudaMemcpyHostToDevice));
            CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
        }
        nvshmemi_call_init_array_kernel<long>(sync_counter, 1, 1);
        nvshmemi_call_init_array_kernel<long>(psync, NVSHMEMI_SYNC_SIZE, NVSHMEMI_SYNC_VALUE);
        CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
    }

    /* This barrier on the parent team eliminates problematic race conditions
     * during psync allocation between back-to-back team creations. */
    nvshmem_quiet();
    // nvshmem_barrier(parent_team->start, parent_team->stride, parent_team->size, psync);
    nvshmem_team_sync(parent_team->team_idx);
    /* This OR reduction assures all PEs return the same value.  */
    CUDA_RUNTIME_CHECK(
        cudaMemcpy(device_team_ret_val, team_ret_val, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
    nvshmemi_call_rdxn_on_stream_kernel<int, RDXN_OPS_MAX>(
        parent_team->team_idx, device_team_ret_val_reduced, device_team_ret_val, 1,
        nvshmemi_state->my_stream);
    CUDA_RUNTIME_CHECK(cudaStreamSynchronize(nvshmemi_state->my_stream));
    CUDA_RUNTIME_CHECK(cudaMemcpy(team_ret_val_reduced, device_team_ret_val_reduced, sizeof(int),
                                  cudaMemcpyDeviceToHost));

    /* If no team was available, print some team triplet info and return nonzero. */
    if (myteam != NULL && myteam->team_idx == -1) {
        NVSHMEMI_WARN_PRINT("Team split strided failed: child <%d, %d, %d>, parent <%d, %d, %d>\n",
                            global_PE_start, global_PE_stride, PE_size, parent_team->start,
                            parent_team->stride, parent_team->size);
        /* TODO: In the event one of the PEs fails to create the team, do we need to revert the team
         * on all of the other ones? */
        free(myteam);
    }

    return *team_ret_val_reduced;
}

int nvshmemi_team_split_2d(nvshmemi_team_t *parent_team, int xrange,
                           const nvshmem_team_config_t *xaxis_config, long xaxis_mask,
                           nvshmem_team_t *xaxis_team, const nvshmem_team_config_t *yaxis_config,
                           long yaxis_mask, nvshmem_team_t *yaxis_team) {
    *xaxis_team = NVSHMEM_TEAM_INVALID;
    *yaxis_team = NVSHMEM_TEAM_INVALID;

    if (xrange > parent_team->size) {
        xrange = parent_team->size;
    }

    const int parent_size = parent_team->size;
    const int num_xteams = ceil(parent_size / (float)xrange);
    const int num_yteams = xrange;

    int start = 0;
    int ret = 0;

    for (int i = 0; i < num_xteams; i++) {
        nvshmem_team_t my_xteam;
        int xsize = (i == num_xteams - 1 && parent_size % xrange) ? parent_size % xrange : xrange;
        ret = nvshmemi_team_split_strided(parent_team, start, 1, xsize, xaxis_config, xaxis_mask,
                                          &my_xteam);
        if (ret) {
            NVSHMEMI_ERROR_PRINT("Creation of x-axis team %d of %d failed\n", i + 1, num_xteams);
        }
        start += xrange;

        if (my_xteam != NVSHMEM_TEAM_INVALID) {
            assert(*xaxis_team == NVSHMEM_TEAM_INVALID);
            *xaxis_team = my_xteam;
        }
    }

    start = 0;

    for (int i = 0; i < num_yteams; i++) {
        nvshmem_team_t my_yteam;
        int remainder = parent_size % xrange;
        int yrange = parent_size / xrange;
        int ysize = (remainder && i < remainder) ? yrange + 1 : yrange;

        ret = nvshmemi_team_split_strided(parent_team, start, xrange, ysize, yaxis_config,
                                          yaxis_mask, &my_yteam);
        if (ret) {
            NVSHMEMI_ERROR_PRINT("Creation of y-axis team %d of %d failed\n", i + 1, num_yteams);
        }
        start += 1;

        if (my_yteam != NVSHMEM_TEAM_INVALID) {
            assert(*yaxis_team == NVSHMEM_TEAM_INVALID);
            *yaxis_team = my_yteam;
        }
    }

    nvshmem_quiet();
    nvshmem_team_sync(parent_team->team_idx);

    return 0;
}

static bool inline nvshmemi_is_rsvd_teams(nvshmem_team_t team_idx) {
    /* This team resource shouldn't not be deleted as they are used for collectives APIs during
     * init/finalize */
    return ((team_idx == NVSHMEM_TEAM_INVALID) ||
            (team_idx >= NVSHMEM_TEAM_WORLD_INDEX && team_idx < NVSHMEM_TEAMS_MIN));
}

void nvshmemi_team_destroy(nvshmemi_team_t *team) {
    int idx = team->team_idx;
    if (nvshmemi_bit_fetch(psync_pool_avail, idx)) {
        NVSHMEMI_ERROR_PRINT("Destroying a team without an active pSync\n");
    }
    /* Since it is a collective routine, perform a barrier */
    // nvshmem_barrier(idx);

    if (!team->is_team_node) {
        if (!nvshmemi_is_rsvd_teams(team->team_node) &&
            nvshmemi_team_pool[team->team_node] != NULL) {
            INFO(NVSHMEM_COLL, "Destroy sub-team 1 [%p] at index[%d] for parent-team [%p]",
                 nvshmemi_team_pool[team->team_node], team->team_node, team);
            nvshmemi_team_destroy(nvshmemi_team_pool[team->team_node]);
        }
    }

    if (!team->is_team_same_mype_node) {
        if (!nvshmemi_is_rsvd_teams(team->team_same_mype_node) &&
            nvshmemi_team_pool[team->team_same_mype_node] != NULL) {
            INFO(NVSHMEM_COLL, "Destroy sub-team 2 [%p] at index[%d] for parent-team [%p]",
                 nvshmemi_team_pool[team->team_same_mype_node], team->team_same_mype_node, team);
            nvshmemi_team_destroy(nvshmemi_team_pool[team->team_same_mype_node]);
        }
    }

    nvshmemi_bit_set(psync_pool_avail, N_PSYNC_BYTES, idx);

    nvshmemi_team_pool[idx] = NULL;
    CUDA_RUNTIME_CHECK(cudaMemset(&nvshmemi_device_team_pool[idx], 0, sizeof(nvshmemi_team_t *)));

    nvshmemi_call_init_array_kernel<long>(&nvshmemi_sync_counter[2 * idx], 2, 1);
    nvshmemi_call_init_array_kernel<long>(&nvshmemi_psync_pool[idx * get_psync_len_per_team()],
                                          get_psync_len_per_team(), NVSHMEMI_SYNC_VALUE);
    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

    nvshmemi_team_destroy_nvls(team);
    nvshmemi_recexchalgo_free_mem(team);
#ifdef NVSHMEM_USE_NCCL
    if (nvshmemi_use_nccl) NCCL_CHECK(nccl_ftable.CommDestroy((ncclComm_t)team->nccl_comm));
#endif
    if (team != &nvshmemi_team_world && team != &nvshmemi_team_shared &&
        team != &nvshmemi_team_node && team != &nvshmemi_team_same_mype_node &&
        team != &nvshmemi_team_same_gpu && team != &nvshmemi_team_gpu_leaders) {
        free(team);
    }
    nvshmemi_team_t *device_team_addr;
    CUDA_RUNTIME_CHECK(cudaMemcpy((void **)&device_team_addr, &nvshmemi_device_team_pool[idx],
                                  sizeof(nvshmemi_team_t *), cudaMemcpyDeviceToHost));
    CUDA_RUNTIME_CHECK(cudaFree(device_team_addr));
}

long *nvshmemi_team_get_psync(nvshmemi_team_t *team, nvshmemi_team_op_t op) {
    long *team_psync;
    size_t psync_fcollect_len;
    psync_fcollect_len = get_fcollect_psync_len_per_team();
    team_psync = &nvshmemi_psync_pool[team->team_idx * get_psync_len_per_team()];
    switch (op) {
        case SYNC:
            return team_psync;
        case REDUCE:
            return &team_psync
                [2 * NVSHMEMI_SYNC_SIZE +
                 (((nvshmemi_device_state.gpu_coll_env_params_var.reduce_scratch_size / 2) /
                   sizeof(long)) *
                  (team->rdxn_count % 2))];
        case BCAST:
            return &team_psync[2 * NVSHMEMI_SYNC_SIZE +
                               nvshmemi_device_state.gpu_coll_env_params_var.reduce_scratch_size /
                                   sizeof(long)];
        case FCOLLECT:
            return &team_psync[2 * NVSHMEMI_SYNC_SIZE +
                               nvshmemi_device_state.gpu_coll_env_params_var.reduce_scratch_size /
                                   sizeof(long) +
                               NVSHMEMI_BCAST_SYNC_SIZE];
        case ALLTOALL:
            return &team_psync[2 * NVSHMEMI_SYNC_SIZE +
                               nvshmemi_device_state.gpu_coll_env_params_var.reduce_scratch_size /
                                   sizeof(long) +
                               NVSHMEMI_BCAST_SYNC_SIZE + psync_fcollect_len +
                               (NVSHMEMI_ALLTOALL_SYNC_SIZE * (team->alltoall_count % 2))];
        case FCOLLECT_128:
            return &team_psync[2 * NVSHMEMI_SYNC_SIZE +
                               nvshmemi_device_state.gpu_coll_env_params_var.reduce_scratch_size /
                                   sizeof(long) +
                               NVSHMEMI_BCAST_SYNC_SIZE + psync_fcollect_len +
                               2 * NVSHMEMI_ALLTOALL_SYNC_SIZE];
        default:
            WARN("Incorrect argument to nvshmemi_team_get_psync\n");
            return NULL;
    }
}

long *nvshmemi_team_get_sync_counter(nvshmemi_team_t *team) {
    return &nvshmemi_sync_counter[2 * team->team_idx];
}
