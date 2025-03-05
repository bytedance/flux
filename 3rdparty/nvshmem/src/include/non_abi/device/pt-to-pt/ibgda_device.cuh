/*
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _NVSHMEMI_IBGDA_DEVICE_H_
#define _NVSHMEMI_IBGDA_DEVICE_H_

#include <cuda_runtime.h>
#if not defined __CUDACC_RTC__
#include <limits.h>
#else
#include <cuda/std/climits>
#endif
#include "infiniband/mlx5dv.h"

#include "non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh"
#include "device_host_transport/nvshmem_common_ibgda.h"
#include "non_abi/nvshmem_build_options.h"
#include "utils_device.h"

#include <algorithm>

//#define NVSHMEM_IBGDA_DEBUG
//#define NVSHMEM_TIMEOUT_DEVICE_POLLING

#define NVSHMEMI_MIN(x, y) ((x) < (y) ? (x) : (y))
#define NVSHMEMI_MAX(x, y) ((x) > (y) ? (x) : (y))

#define NVSHMEMI_IBGDA_PTX_OPTIMIZATION_MFENCE

#ifdef NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY
// These PTX optimizations are for GPU memory access only.
// Both data and NIC control objects must be in GPU memory.
#define NVSHMEMI_IBGDA_PTX_OPTIMIZATION_ATOMIC_READ_SET
#define NVSHMEMI_IBGDA_PTX_OPTIMIZATION_STORE_RELEASE
#endif

#define IBGDA_FULL_WARP 0xffffffffU
#define IBGDA_POLL_TIMEOUT 4000000000LLU

/* When we exceed a specific number of threads doing quiet
 * we end up with cache thrashing which causes a significant
 * perf hit. TODO: Tune this number for each supported arch.
 */
#define IBGDA_MAX_THREADS_PER_QUIET 32

// MLX5 accepts up to 2 GiB per command
#define IBGDA_MAX_TRANSFER_SIZE 2147483648LLU

#ifndef likely
#define likely(x) (__builtin_expect(!!(x), 1))
#endif

#ifndef unlikely
#define unlikely(x) (__builtin_expect(!!(x), 0))
#endif

#ifndef ACCESS_ONCE
#define ACCESS_ONCE(x) (*(volatile typeof(x) *)&(x))
#endif

/**
 * DO NOT use BSWAP(READ_ONCE(x)) as it could create a bug.
 * BSWAP is a pre-processor function. It will be unrolled to many READ_ONCE.
 */
#ifndef READ_ONCE
#define READ_ONCE(x) ACCESS_ONCE(x)
#endif

#ifndef WRITE_ONCE
#define WRITE_ONCE(x, v) (ACCESS_ONCE(x) = (v))
#endif

#ifdef NVSHMEM_IBGDA_DEBUG
struct mlx5_err_cqe_ex {
    uint8_t rsvd0[32];
    __be32 srqn;
    uint8_t rsvd1[16];
    uint8_t hw_err_synd;
    uint8_t hw_synd_type;
    uint8_t vendor_err_synd;
    uint8_t syndrome;
    __be32 s_wqe_opcode_qpn;
    __be16 wqe_counter;
    uint8_t signature;
    uint8_t op_own;
};
typedef struct mlx5_err_cqe_ex ibgda_mlx5_err_cqe_t;
#else
typedef struct mlx5_err_cqe ibgda_mlx5_err_cqe_t;
#endif

#define IBGDA_4_BYTE_EXT_AMO_OPMOD 0x08000000
#define IBGDA_8_BYTE_EXT_AMO_OPMOD 0x09000000

typedef enum ibgda_mlx5_fm {
    IBGDA_MLX5_FM_NO_FENCE = 0,
    IBGDA_MLX5_FM_INITIATOR_SMALL_FENCE = 1 << 5,
    IBGDA_MLX5_FM_FENCE = 2 << 5,
    IBGDA_MLX5_FM_STRONG_ORDERING = 3 << 5,
    IBGDA_MLX5_FM_FENCE_AND_INITIATOR_SMALL_FENCE = 4 << 5,
    OBGDA_MLX5_FM_OP_MAX = INT_MAX,
} ibgda_mlx5_fm_t;

enum {
    IBGDA_MLX5_OPCODE_DUMP = 0x23,
    IBGDA_MLX5_OPCODE_SENTINEL = INT_MAX

};

typedef struct mlx5_wqe_ctrl_seg __attribute__((__aligned__(8))) ibgda_ctrl_seg_t;

// The ext flag (in dqp_dct) must be set to disable.
typedef struct {
    __be64 dc_key;
    __be32 dqp_dct;
    uint8_t stat_rate_sl;
    uint8_t fl_mlid;
    __be16 rlid;
} __attribute__((__packed__)) __attribute__((__aligned__(4))) ibgda_half_av_seg_t;
#if __cplusplus >= 201103L
static_assert(sizeof(ibgda_half_av_seg_t) == 16, "sizeof(ibgda_half_av_seg_t) == 16 failed.");
#endif

typedef struct {
    uint32_t add_data;
    uint32_t field_boundary;
    uint64_t reserved;
} __attribute__((__packed__)) ibgda_atomic_32_masked_fa_seg_t;
#if __cplusplus >= 201103L
static_assert(sizeof(ibgda_atomic_32_masked_fa_seg_t) == 16,
              "sizeof(ibgda_atomic_32_masked_fa_seg_t) == 16 failed.");
#endif

typedef struct {
    uint64_t add_data;
    uint64_t field_boundary;
} __attribute__((__packed__)) ibgda_atomic_64_masked_fa_seg_t;
#if __cplusplus >= 201103L
static_assert(sizeof(ibgda_atomic_64_masked_fa_seg_t) == 16,
              "sizeof(ibgda_atomic_64_masked_fa_seg_t) == 16 failed.");
#endif

typedef struct {
    uint32_t swap_data;
    uint32_t compare_data;
    uint32_t swap_mask;
    uint32_t compare_mask;
} __attribute__((__packed__)) ibgda_atomic_32_masked_cs_seg_t;
#if __cplusplus >= 201103L
static_assert(sizeof(ibgda_atomic_32_masked_cs_seg_t) == 16,
              "sizeof(ibgda_atomic_32_masked_cs_seg_t) == 16 failed.");
#endif

typedef struct {
    uint64_t swap;
    uint64_t compare;
} __attribute__((__packed__)) ibgda_atomic_64_masked_cs_seg_t;
#if __cplusplus >= 201103L
static_assert(sizeof(ibgda_atomic_64_masked_cs_seg_t) == 16,
              "sizeof(ibgda_atomic_64_masked_cs_seg_t) == 16 failed.");
#endif

#ifdef __CUDA_ARCH__

#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
__device__ static inline uint64_t ibgda_query_globaltimer() {
    uint64_t ret;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ret)::"memory");
    return ret;
}
#endif /* NVSHMEM_TIMEOUT_DEVICE_POLLING */

__device__ static inline nvshmemi_ibgda_device_state_t *ibgda_get_state() {
    return &nvshmemi_ibgda_device_state_d;
}

__device__ static inline bool ibgda_is_rc_enabled() { return ibgda_get_state()->num_rc_per_pe > 0; }

// Prevent code reordering from both compiler and GPU
__device__ static inline void IBGDA_MFENCE() {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_MFENCE
    asm volatile("fence.acq_rel.cta;" ::: "memory");
#else
    __threadfence_block();
#endif /* NVSHMEMI_IBGDA_PTX_OPTIMIZATION_MFENCE */
}

__device__ static inline void IBGDA_MEMBAR_NO_OPTIMIZATION() {
#ifdef NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY
    __threadfence();
#else
    if (likely(ibgda_get_state()->nic_buf_on_gpumem))
        __threadfence();
    else
        __threadfence_system();
#endif /* NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY */
}

__device__ static inline void IBGDA_MEMBAR() {
// st.release automatically adds membar in SASS.
#ifndef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_STORE_RELEASE

#ifdef NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY
    __threadfence();
#else
    if (likely(ibgda_get_state()->nic_buf_on_gpumem))
        __threadfence();
    else
        __threadfence_system();
#endif /* NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY */

#endif /* NVSHMEMI_IBGDA_PTX_OPTIMIZATION_STORE_RELEASE */
}

__device__ inline int nvshmemi_thread_id_in_warp() {
    int myIdx;
    asm volatile("mov.u32  %0, %laneid;" : "=r"(myIdx));
    return myIdx;
}

__device__ inline int nvshmemi_warp_size() {
    return ((blockDim.x * blockDim.y * blockDim.z) < warpSize)
               ? (blockDim.x * blockDim.y * blockDim.z)
               : warpSize;
}

__device__ inline void nvshmemi_warp_sync() { __syncwarp(); }

__device__ inline int nvshmemi_thread_id_in_block() {
    return (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y);
}

__device__ inline int nvshmemi_block_size() { return (blockDim.x * blockDim.y * blockDim.z); }

__device__ static inline uint32_t ibgda_get_smid() {
    uint32_t smid;
    asm("mov.u32  %0, %%smid;" : "=r"(smid));
    return smid;
}

__device__ static inline uint32_t ibgda_get_ctaid() {
    return (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y);
}

template <typename T>
__device__ static inline void ibgda_store_relaxed(T *ptr, T val) {
    WRITE_ONCE(*ptr, val);
}

template <>
__device__ inline void ibgda_store_relaxed(uint8_t *ptr, uint8_t val) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_STORE_RELEASE
    uint16_t _val = val;
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b8 [%0], %1;" : : "l"(ptr), "h"(_val));
#else
    WRITE_ONCE(*ptr, val);
#endif
}

template <>
__device__ inline void ibgda_store_relaxed(uint16_t *ptr, uint16_t val) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_STORE_RELEASE
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b16 [%0], %1;" : : "l"(ptr), "h"(val));
#else
    WRITE_ONCE(*ptr, val);
#endif
}

template <>
__device__ inline void ibgda_store_relaxed(uint32_t *ptr, uint32_t val) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_STORE_RELEASE
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
#else
    WRITE_ONCE(*ptr, val);
#endif
}

template <>
__device__ inline void ibgda_store_relaxed(uint64_t *ptr, uint64_t val) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_STORE_RELEASE
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b64 [%0], %1;" : : "l"(ptr), "l"(val));
#else
    WRITE_ONCE(*ptr, val);
#endif
}

__device__ static inline void ibgda_store_release(uint32_t *ptr, uint32_t val) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_STORE_RELEASE
    asm volatile("st.release.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
#else
    WRITE_ONCE(*ptr, val);
#endif
}

__device__ static inline void ibgda_store_release(uint64_t *ptr, uint64_t val) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_STORE_RELEASE
    asm volatile("st.release.gpu.global.L1::no_allocate.b64 [%0], %1;" : : "l"(ptr), "l"(val));
#else
    WRITE_ONCE(*ptr, val);
#endif
}

/**
 * DO NOT use BSWAP(ibgda_atomic_read(x)) as it could create a bug.
 * See the comment near READ_ONCE.
 */
__device__ static inline uint8_t ibgda_atomic_read(uint8_t *ptr) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_ATOMIC_READ_SET
    uint16_t ret;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b8 %0, [%1];" : "=h"(ret) : "l"(ptr));
    return (uint8_t)ret;
#else
    return READ_ONCE(*ptr);
#endif
}

__device__ static inline uint16_t ibgda_atomic_read(uint16_t *ptr) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_ATOMIC_READ_SET
    uint16_t ret;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b16 %0, [%1];" : "=h"(ret) : "l"(ptr));
    return ret;
#else
    return READ_ONCE(*ptr);
#endif
}

__device__ static inline uint32_t ibgda_atomic_read(uint32_t *ptr) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_ATOMIC_READ_SET
    uint32_t ret;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
#else
    return READ_ONCE(*ptr);
#endif
}

__device__ static inline uint64_t ibgda_atomic_read(uint64_t *ptr) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_ATOMIC_READ_SET
    uint64_t ret;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
#else
    return READ_ONCE(*ptr);
#endif
}

__device__ static inline void ibgda_atomic_set(int *ptr, int val) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_ATOMIC_READ_SET
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
#else
    WRITE_ONCE(*ptr, val);
#endif
}

__device__ static inline size_t ibgda_cal_transfer_size(size_t req_size, size_t lchunk_size,
                                                        size_t rchunk_size) {
    return NVSHMEMI_MIN(IBGDA_MAX_TRANSFER_SIZE,
                        NVSHMEMI_MIN(req_size, NVSHMEMI_MIN(rchunk_size, lchunk_size)));
}

template <threadgroup_t SCOPE>
__device__ static inline void ibgda_lock_acquire(int *lock) {
    if (nvshmemi_thread_id_in_threadgroup<SCOPE>() == 0)
        while (atomicCAS(lock, 0, 1) == 1)
            ;  // Wait until we get the lock.

    if (SCOPE == NVSHMEMI_THREADGROUP_THREAD)
        IBGDA_MFENCE();  // Prevent reordering before lock is acquired.

    // For other scopes, __syncwarp / __syncthreads guarantee the ordering
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <threadgroup_t SCOPE>
__device__ static inline void ibgda_lock_release(int *lock) {
    // For other scopes, __syncwarp / __syncthreads guarantee the ordering
    nvshmemi_threadgroup_sync<SCOPE>();

    if (SCOPE == NVSHMEMI_THREADGROUP_THREAD)
        IBGDA_MFENCE();  // Prevent reordering before lock is released.

    if (nvshmemi_thread_id_in_threadgroup<SCOPE>() == 0) ibgda_atomic_set(lock, 0);
}

// Multiple threads may update get_head concurrently.
// Only the latest one w.r.t. wqe_idx is important.
__device__ static inline void ibgda_update_get_head(nvshmemi_ibgda_device_qp_t *qp,
                                                    uint64_t new_get_head) {
    nvshmemi_ibgda_device_qp_management_t *mvars = &qp->mvars;
    atomicMax((unsigned long long int *)&mvars->tx_wq.get_head,
              (unsigned long long int)new_get_head);
}

__device__ static inline void ibgda_update_get_tail(nvshmemi_ibgda_device_qp_t *qp,
                                                    uint64_t new_get_tail) {
    nvshmemi_ibgda_device_qp_management_t *mvars = &qp->mvars;
    atomicMax((unsigned long long int *)&mvars->tx_wq.get_tail,
              (unsigned long long int)new_get_tail);
}

__device__ static inline void *ibgda_get_wqe_ptr(nvshmemi_ibgda_device_qp_t *qp, uint16_t wqe_idx) {
    uint16_t cnt = qp->tx_wq.nwqes;
    uint16_t idx = wqe_idx & (cnt - 1);
    return (void *)((uintptr_t)qp->tx_wq.wqe + (idx << MLX5_SEND_WQE_SHIFT));
}

#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
__device__ static inline int ibgda_check_poll_timeout(nvshmemi_ibgda_device_cq_t *cq, uint64_t now,
                                                      uint64_t start, uint64_t idx, int *error) {
    int status = 0;

    struct mlx5_cqe64 *cqe64 = (struct mlx5_cqe64 *)cq->cqe;
    uint8_t opown;
    uint8_t opcode;
    uint16_t wqe_counter;

    if (unlikely(now - start > IBGDA_POLL_TIMEOUT)) {
        *error = -ETIME;

        opown = ibgda_atomic_read(&cqe64->op_own);
        opcode = opown >> 4;

        wqe_counter = ibgda_atomic_read(&cqe64->wqe_counter);
        wqe_counter = BSWAP16(wqe_counter);

        printf(
            "[%d] ibgda_poll_cq timeout:\n"
            "    cons_idx=%#lx, prod_idx=%#lx, cqn=%#x, qpn=%#x, opcode=%#x\n"
            "    wqe_counter=%#x, resv_head=%#lx, ready_head=%#lx\n"
            "    while waiting for idx=%#lx.\n",
            nvshmemi_device_state_d.mype, ibgda_atomic_read(cq->cons_idx),
            ibgda_atomic_read(cq->prod_idx), cq->cqn, cq->qpn, opcode, wqe_counter,
            ibgda_atomic_read(cq->resv_head), ibgda_atomic_read(cq->ready_head), idx);
        status = -1;
    }
    return status;
}
#endif

#if __cplusplus >= 201103L
static_assert(NVSHMEMI_IBGDA_MAX_QP_DEPTH <= 32768,
              "static_assert(NVSHMEMI_IBGDA_MAX_QP_DEPTH <= 32768) failed");
#endif
__device__ static inline int ibgda_poll_cq(nvshmemi_ibgda_device_cq_t *cq, uint64_t idx,
                                           int *error) {
    int status = 0;
    struct mlx5_cqe64 *cqe64 = (struct mlx5_cqe64 *)cq->cqe;

    const uint32_t ncqes = cq->ncqes;

    uint8_t opown;
    uint8_t opcode;
    uint16_t wqe_counter;
    uint16_t new_wqe_counter;

#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
    uint64_t start = ibgda_query_globaltimer();
    uint64_t now;
#endif

    uint64_t cons_idx = ibgda_atomic_read(cq->cons_idx);
    uint64_t new_cons_idx;

    assert(likely(cq->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI ||
                  cq->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_RC));

    if (unlikely(cons_idx >= idx)) goto out;

#ifdef NVSHMEM_IBGDA_DEBUG
    // We can skip opcode == MLX5_CQE_INVALID check because we have already
    // initialized the CQ buffer to 0xff. With the QP depth range we enforce,
    // cons_idx cannot progress unless wqe_counter read from the CQ buffer is
    // a valid value.
    do {
        opown = ibgda_atomic_read(&cqe64->op_own);
        opcode = opown >> 4;

#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
        // TODO: Integrate timeout handler with the core NVSHMEM
        now = ibgda_query_globaltimer();
        status = ibgda_check_poll_timeout(cq, now, start, idx, error);
        if (status != 0) goto check_opcode;
#endif /* NVSHMEM_TIMEOUT_DEVICE_POLLING */
    } while (unlikely(opcode == MLX5_CQE_INVALID));

    // Prevent reordering of the opcode wait above
    IBGDA_MFENCE();
#endif /* NVSHMEM_IBGDA_DEBUG */

#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
    start = ibgda_query_globaltimer();
#endif

    // If idx is a lot greater than cons_idx, we might get incorrect result due
    // to wqe_counter wraparound. We need to check prod_idx to be sure that idx
    // has already been submitted.
    while (unlikely(ibgda_atomic_read(cq->prod_idx) < idx))
        ;
    IBGDA_MFENCE();

    do {
        new_wqe_counter = ibgda_atomic_read(&cqe64->wqe_counter);
        new_wqe_counter = BSWAP16(new_wqe_counter);
#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
        now = ibgda_query_globaltimer();
        status = ibgda_check_poll_timeout(cq, now, start, idx, error);
        if (status != 0) goto check_opcode;

        // Observe progress. Reset the timer.
        if (new_wqe_counter != wqe_counter) start = now;
#endif
        wqe_counter = new_wqe_counter;

        // Another thread may have updated cons_idx.
        cons_idx = ibgda_atomic_read(cq->cons_idx);
        if (likely(cons_idx >= idx)) goto out;
    }
    // NOTE: This while loop is part of do while above.
    // wqe_counter is the HW consumer index. However, we always maintain index
    // + 1 in SW. To be able to compare with idx, we need to use wqe_counter +
    // 1. Because wqe_counter is uint16_t, it may wraparound. Still we know for
    // sure that if idx - wqe_counter - 1 < ncqes, wqe_counter + 1 is less than
    // idx, and thus we need to wait. We don't need to wait when idx ==
    // wqe_counter + 1. That's why we use - (uint16_t)2 here to make this case
    // wraparound.
    while (unlikely(((uint16_t)((uint16_t)idx - wqe_counter - (uint16_t)2) < ncqes)));

    // new_cons_idx is uint64_t but wqe_counter is uint16_t. Thus, we get the
    // MSB from idx. We also need to take care of wraparound.
    ++wqe_counter;
    new_cons_idx =
        (idx & ~(0xffffULL) | wqe_counter) + (((uint16_t)idx > wqe_counter) ? 0x10000ULL : 0x0);
    atomicMax((unsigned long long int *)cq->cons_idx, (unsigned long long int)new_cons_idx);

#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
check_opcode:
#endif

    // NVSHMEM always treats CQE errors as fatal.
    // Even if this error doesn't belong to the CQE in cons_idx,
    // we will just report and terminate the process.
    opown = ibgda_atomic_read(&cqe64->op_own);
    opcode = opown >> 4;

    if (unlikely(opcode == MLX5_CQE_REQ_ERR)) {
        ibgda_mlx5_err_cqe_t *cqe_err = (ibgda_mlx5_err_cqe_t *)cqe64;
        *error = cqe_err->syndrome;
#ifdef NVSHMEM_IBGDA_DEBUG
        __be16 wqe_counter = ibgda_atomic_read(&cqe_err->wqe_counter);
        __be32 s_wqe_opcode_qpn = ibgda_atomic_read(&cqe_err->s_wqe_opcode_qpn);
        printf(
            "[%d] got completion with err:\n"
            "   syndrome=%#x, vendor_err_synd=%#x, hw_err_synd=%#x, hw_synd_type=%#x,\n"
            "   wqe_counter=%#x, s_wqe_opcode_qpn=%#x,\n"
            "   cqn=%#x, cons_idx=%#lx, prod_idx=%#lx, idx=%#lx\n",
            nvshmemi_device_state_d.mype, cqe_err->syndrome, cqe_err->vendor_err_synd,
            cqe_err->hw_err_synd, cqe_err->hw_synd_type, BSWAP16(wqe_counter),
            BSWAP32(s_wqe_opcode_qpn), cq->cqn, cons_idx, ibgda_atomic_read(cq->prod_idx), idx);
#endif /* NVSHMEM_IBGDA_DEBUG */
        status = -1;
    }

out:
    // Prevent reordering of this function and subsequent instructions
    IBGDA_MFENCE();

    return status;
}

__device__ static inline void ibgda_write_nop_wqe(nvshmemi_ibgda_device_qp_t *qp, uint16_t wqe_idx,
                                                  void **out_wqes) {
    ibgda_ctrl_seg_t ctrl_seg;

    ibgda_ctrl_seg_t *ctrl_seg_ptr = (ibgda_ctrl_seg_t *)out_wqes[0];

    ctrl_seg = {
        0,
    };
    ctrl_seg.qpn_ds = HTOBE32((qp->qpn << 8) | 2);
    ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
    ctrl_seg.opmod_idx_opcode = HTOBE32((wqe_idx << 8) | MLX5_OPCODE_NOP);

    // wqe_ptr will not be consumed by GPU.
    // WRITE_ONCE ensures that compiler will not removed this code.
    uint32_t *dst = (uint32_t *)ctrl_seg_ptr;
    uint32_t *src = (uint32_t *)&ctrl_seg;
    for (int i = 0; i < sizeof(*ctrl_seg_ptr) / sizeof(uint32_t); ++i)
        ibgda_store_relaxed(&dst[i], src[i]);
}

__device__ static inline void ibgda_write_dump_wqe(nvshmemi_ibgda_device_qp_t *qp, uint64_t laddr,
                                                   __be32 lkey, uint32_t bytes, uint16_t wqe_idx,
                                                   ibgda_mlx5_fm_t fm, void **out_wqes) {
    ibgda_ctrl_seg_t ctrl_seg;
    struct mlx5_wqe_data_seg data_seg;

    ibgda_ctrl_seg_t *ctrl_seg_ptr = (ibgda_ctrl_seg_t *)out_wqes[0];
    struct mlx5_wqe_data_seg *data_seg_ptr =
        (struct mlx5_wqe_data_seg *)((uintptr_t)out_wqes[0] + sizeof(*ctrl_seg_ptr));

    data_seg.byte_count = HTOBE32(bytes);
    data_seg.lkey = lkey;
    data_seg.addr = HTOBE64(laddr);

    ctrl_seg = {
        0,
    };
    ctrl_seg.qpn_ds = HTOBE32((qp->qpn << 8) | 2);
    ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE | fm;
    ctrl_seg.opmod_idx_opcode = HTOBE32((wqe_idx << 8) | IBGDA_MLX5_OPCODE_DUMP);

    // wqe_ptr will not be consumed by GPU.
    // WRITE_ONCE ensures that compiler will not removed this code.
    uint32_t *dst = (uint32_t *)ctrl_seg_ptr;
    uint32_t *src = (uint32_t *)&ctrl_seg;
    for (int i = 0; i < sizeof(*ctrl_seg_ptr) / sizeof(uint32_t); ++i)
        ibgda_store_relaxed(&dst[i], src[i]);

    dst = (uint32_t *)data_seg_ptr;
    src = (uint32_t *)&data_seg;
    for (int i = 0; i < sizeof(*data_seg_ptr) / sizeof(uint32_t); ++i)
        ibgda_store_relaxed(&dst[i], src[i]);
}

template <bool support_half_av_seg>
__device__ static inline void ibgda_write_rdma_write_wqe(nvshmemi_ibgda_device_qp_t *qp,
                                                         nvshmemi_ibgda_device_dct_t *dct,
                                                         uint64_t laddr, __be32 lkey,
                                                         uint64_t raddr, __be32 rkey,
                                                         uint32_t bytes, uint16_t wqe_idx,
                                                         uint8_t fm_ce_se, void **out_wqes) {
    ibgda_ctrl_seg_t ctrl_seg;
    struct mlx5_wqe_raddr_seg raddr_seg;
    struct mlx5_wqe_data_seg data_seg;

    ibgda_ctrl_seg_t *ctrl_seg_ptr = (ibgda_ctrl_seg_t *)out_wqes[0];
    void *av_seg_ptr = (void *)((uintptr_t)ctrl_seg_ptr + sizeof(*ctrl_seg_ptr));
    struct mlx5_wqe_raddr_seg *raddr_seg_ptr;
    struct mlx5_wqe_data_seg *data_seg_ptr;

    size_t av_seg_size;
    int ds;

    if (qp->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI) {
        if (support_half_av_seg) {
            ds = 4;
            av_seg_size = sizeof(ibgda_half_av_seg_t);
            raddr_seg_ptr = (struct mlx5_wqe_raddr_seg *)((uintptr_t)av_seg_ptr + av_seg_size);
        } else {
            ds = 6;
            av_seg_size = sizeof(struct mlx5_wqe_av);
            raddr_seg_ptr = (struct mlx5_wqe_raddr_seg *)out_wqes[1];
        }
    } else {
        ds = 3;
        av_seg_size = 0;
        raddr_seg_ptr = (struct mlx5_wqe_raddr_seg *)((uintptr_t)av_seg_ptr + av_seg_size);
    }
    data_seg_ptr = (struct mlx5_wqe_data_seg *)((uintptr_t)raddr_seg_ptr + sizeof(*raddr_seg_ptr));

    raddr_seg.raddr = HTOBE64(raddr);
    raddr_seg.rkey = rkey;
    raddr_seg.reserved = 0;

    data_seg.byte_count = HTOBE32(bytes);
    data_seg.lkey = lkey;
    data_seg.addr = HTOBE64(laddr);

    ctrl_seg = {
        0,
    };
    ctrl_seg.qpn_ds = HTOBE32((qp->qpn << 8) | ds);
    ctrl_seg.fm_ce_se = fm_ce_se;
    ctrl_seg.opmod_idx_opcode = HTOBE32((wqe_idx << 8) | MLX5_OPCODE_RDMA_WRITE);

    uint32_t *dst = (uint32_t *)ctrl_seg_ptr;
    uint32_t *src = (uint32_t *)&ctrl_seg;
    for (int i = 0; i < sizeof(*ctrl_seg_ptr) / sizeof(uint32_t); ++i)
        ibgda_store_relaxed(&dst[i], src[i]);

    if (av_seg_size > 0) {
        dst = (uint32_t *)av_seg_ptr;
        src = (uint32_t *)dct;
        for (int i = 0; i < av_seg_size / sizeof(uint32_t); ++i)
            ibgda_store_relaxed(&dst[i], src[i]);
    }

    dst = (uint32_t *)raddr_seg_ptr;
    src = (uint32_t *)&raddr_seg;
    for (int i = 0; i < sizeof(*raddr_seg_ptr) / sizeof(uint32_t); ++i)
        ibgda_store_relaxed(&dst[i], src[i]);

    dst = (uint32_t *)data_seg_ptr;
    src = (uint32_t *)&data_seg;
    for (int i = 0; i < sizeof(*data_seg_ptr) / sizeof(uint32_t); ++i)
        ibgda_store_relaxed(&dst[i], src[i]);
}

template <bool support_half_av_seg>
__device__ static inline void ibgda_write_rdma_write_inl_wqe(nvshmemi_ibgda_device_qp_t *qp,
                                                             nvshmemi_ibgda_device_dct_t *dct,
                                                             const void *val, uint64_t raddr,
                                                             __be32 rkey, uint32_t bytes,
                                                             uint16_t wqe_idx, uint8_t fm_ce_se,
                                                             void **out_wqes) {
    ibgda_ctrl_seg_t ctrl_seg;
    struct mlx5_wqe_raddr_seg raddr_seg;
    struct mlx5_wqe_inl_data_seg inl_seg;

    ibgda_ctrl_seg_t *ctrl_seg_ptr = (ibgda_ctrl_seg_t *)out_wqes[0];
    void *av_seg_ptr = (void *)((uintptr_t)ctrl_seg_ptr + sizeof(*ctrl_seg_ptr));
    struct mlx5_wqe_raddr_seg *raddr_seg_ptr;
    struct mlx5_wqe_inl_data_seg *inl_seg_ptr;
    void *wqe_data_ptr;

    size_t av_seg_size;
    int ds;

    // Allow up to 12 bytes
    assert(likely(bytes <= 12));

    if (qp->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI) {
        if (support_half_av_seg) {
            ds = 4;
            av_seg_size = sizeof(ibgda_half_av_seg_t);
            raddr_seg_ptr = (struct mlx5_wqe_raddr_seg *)((uintptr_t)av_seg_ptr + av_seg_size);
        } else {
            ds = 6;
            av_seg_size = sizeof(struct mlx5_wqe_av);
            raddr_seg_ptr = (struct mlx5_wqe_raddr_seg *)out_wqes[1];
        }
    } else {
        ds = 3;
        av_seg_size = 0;
        raddr_seg_ptr = (struct mlx5_wqe_raddr_seg *)av_seg_ptr;
    }
    inl_seg_ptr =
        (struct mlx5_wqe_inl_data_seg *)((uintptr_t)raddr_seg_ptr + sizeof(*raddr_seg_ptr));
    wqe_data_ptr = (void *)((uintptr_t)inl_seg_ptr + sizeof(*inl_seg_ptr));

    raddr_seg.raddr = HTOBE64(raddr);
    raddr_seg.rkey = rkey;
    raddr_seg.reserved = 0;

    inl_seg.byte_count = HTOBE32(bytes | MLX5_INLINE_SEG);

    ctrl_seg = {
        0,
    };
    ctrl_seg.qpn_ds = HTOBE32((qp->qpn << 8) | ds);
    ctrl_seg.fm_ce_se = fm_ce_se;
    ctrl_seg.opmod_idx_opcode = HTOBE32((wqe_idx << 8) | MLX5_OPCODE_RDMA_WRITE);

    uint32_t *dst = (uint32_t *)ctrl_seg_ptr;
    uint32_t *src = (uint32_t *)&ctrl_seg;
    for (int i = 0; i < sizeof(*ctrl_seg_ptr) / sizeof(uint32_t); ++i)
        ibgda_store_relaxed(&dst[i], src[i]);

    if (av_seg_size > 0) {
        dst = (uint32_t *)av_seg_ptr;
        src = (uint32_t *)dct;
        for (int i = 0; i < av_seg_size / sizeof(uint32_t); ++i)
            ibgda_store_relaxed(&dst[i], src[i]);
    }

    dst = (uint32_t *)raddr_seg_ptr;
    src = (uint32_t *)&raddr_seg;
    for (int i = 0; i < sizeof(*raddr_seg_ptr) / sizeof(uint32_t); ++i)
        ibgda_store_relaxed(&dst[i], src[i]);

    dst = (uint32_t *)inl_seg_ptr;
    src = (uint32_t *)&inl_seg;
    for (int i = 0; i < sizeof(*inl_seg_ptr) / sizeof(uint32_t); ++i)
        ibgda_store_relaxed(&dst[i], src[i]);

    switch (bytes) {
        case 1:
            ibgda_store_relaxed((uint8_t *)wqe_data_ptr, *((uint8_t *)val));
            break;
        case 2:
            ibgda_store_relaxed((uint16_t *)wqe_data_ptr, *((uint16_t *)val));
            break;
        case 4:
            ibgda_store_relaxed((uint32_t *)wqe_data_ptr, *((uint32_t *)val));
            break;
        case 8:
            // wqe_data_ptr is aligned at 4B. We cannot use uint64_t here.
            ibgda_store_relaxed(&(((uint32_t *)wqe_data_ptr)[0]), ((uint32_t *)val)[0]);
            ibgda_store_relaxed(&(((uint32_t *)wqe_data_ptr)[1]), ((uint32_t *)val)[1]);
            break;
        default:
            memcpy(wqe_data_ptr, val, bytes);
    }
}

/**
 * For DC, support only half av seg.
 * The header already consumes 1 wqebb and leaves 12 bytes for inline data.
 * The last wqebb is no-op.
 * One wqebb is 64 bytes.
 * Pre-calculate as it is faster to do lookup.
 * Formula: ceil(((sizeof(T) * 32) - 12) / 64) + 2
 *
 * For RC
 * The header already consumes 1 wqebb and leaves 12 + 16 bytes for inline data.
 * The last wqebb is no-op.
 * One wqebb is 64 bytes.
 * Pre-calculate as it is faster to do lookup.
 * Formula: ceil(((sizeof(T) * 32) - (12 + 16)) / 64) + 2
 */
template <typename T, nvshmemi_ibgda_device_qp_type_t qp_type>
__device__ static inline uint32_t ibgda_get_num_wqes_in_inl_combine_warp() {
    if (qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI) {
        // DC supports up to 16 DS WQE
        switch (sizeof(T)) {
            case 1:
            case 2:
                return 3;
            case 4:
                return 4;
            default:
#ifdef NVSHMEM_IBGDA_DEBUG
                printf("Unsupported type.\n");
#endif
                assert(0);
                return 0;
        }
    } else {
        // RC supports up to 64 DS WQE
        switch (sizeof(T)) {
            case 1:
            case 2:
                return 3;
            case 4:
                return 4;
            case 8:
                return 6;
            default:
#ifdef NVSHMEM_IBGDA_DEBUG
                printf("Unsupported type.\n");
#endif
                assert(0);
                return 0;
        }
    }
}

/**
 * For DC, support only half av seg.
 * The header already consumes 4 ds and leaves 12 bytes for inline data.
 * One ds is 16 bytes.
 * Pre-calculate as it is faster to do lookup.
 * Formula: ceil(((sizeof(T) * 32) - 12) / 16) + 4
 *
 * For RC
 * The header already consumes 3 ds and leaves 12 bytes for inline data.
 * One ds is 16 bytes.
 * Pre-calculate as it is faster to do lookup.
 * Formula: ceil(((sizeof(T) * 32) - 12) / 16) + 3
 */
template <typename T, nvshmemi_ibgda_device_qp_type_t qp_type>
__device__ static inline uint32_t ibgda_get_ds_in_inl_combine_warp() {
    if (qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI) {
        // DC supports up to 16 DS WQE
        switch (sizeof(T)) {
            case 1:
                return 6;
            case 2:
                return 8;
            case 4:
                return 12;
            default:
#ifdef NVSHMEM_IBGDA_DEBUG
                printf("Unsupported type.\n");
#endif
                assert(0);
                return 0;
        }
    } else {
        // DC supports up to 16 DS WQE
        switch (sizeof(T)) {
            case 1:
                return 5;
            case 2:
                return 7;
            case 4:
                return 11;
            case 8:
                return 19;
            default:
#ifdef NVSHMEM_IBGDA_DEBUG
                printf("Unsupported type.\n");
#endif
                assert(0);
                return 0;
        }
    }
}

template <typename T>
__device__ static inline void ibgda_write_rdma_write_inl_wqe_combine_warp(
    nvshmemi_ibgda_device_qp_t *qp, nvshmemi_ibgda_device_dct_t *dct, const T val, uint64_t _raddr,
    __be32 rkey, uint16_t wqe_idx, int my_tid, void **out_wqes) {
    ibgda_ctrl_seg_t ctrl_seg;
    struct mlx5_wqe_raddr_seg raddr_seg;
    struct mlx5_wqe_inl_data_seg inl_seg;

    ibgda_ctrl_seg_t *ctrl_seg_ptr = (ibgda_ctrl_seg_t *)out_wqes[0];
    void *av_seg_ptr = (void *)((uintptr_t)ctrl_seg_ptr + sizeof(*ctrl_seg_ptr));
    struct mlx5_wqe_raddr_seg *raddr_seg_ptr;
    struct mlx5_wqe_inl_data_seg *inl_seg_ptr;

    size_t av_seg_size;
    int ds;

    uint32_t bytes = sizeof(T);
    uint64_t raddr = _raddr - (my_tid * bytes);

    int remaining_size_for_data_in_first_wqebb;
    uint32_t nop_relative_wqe_idx;

    if (qp->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI) {
        ds = ibgda_get_ds_in_inl_combine_warp<T, NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI>();
        av_seg_size = sizeof(ibgda_half_av_seg_t);
        remaining_size_for_data_in_first_wqebb = 12;
        nop_relative_wqe_idx =
            ibgda_get_num_wqes_in_inl_combine_warp<T, NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI>() - 1;
    } else {
        ds = ibgda_get_ds_in_inl_combine_warp<T, NVSHMEMI_IBGDA_DEVICE_QP_TYPE_RC>();
        av_seg_size = 0;
        remaining_size_for_data_in_first_wqebb = 28;
        nop_relative_wqe_idx =
            ibgda_get_num_wqes_in_inl_combine_warp<T, NVSHMEMI_IBGDA_DEVICE_QP_TYPE_RC>() - 1;
    }

    raddr_seg_ptr = (struct mlx5_wqe_raddr_seg *)((uintptr_t)av_seg_ptr + av_seg_size);
    inl_seg_ptr =
        (struct mlx5_wqe_inl_data_seg *)((uintptr_t)raddr_seg_ptr + sizeof(*raddr_seg_ptr));

    raddr_seg.raddr = HTOBE64(raddr);
    raddr_seg.rkey = rkey;
    raddr_seg.reserved = 0;

    inl_seg.byte_count = HTOBE32((bytes * warpSize) | MLX5_INLINE_SEG);

    ctrl_seg = {
        0,
    };
    ctrl_seg.qpn_ds = HTOBE32((qp->qpn << 8) | ds);
    // ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
    // This RDMA WRITE wqe will not get CQ update to avoid dynamic size calculation in poll_cq.
    // Instead, the NO-OP wqe (last one) will get CQ update because it is always 1 WQEBB.
    ctrl_seg.fm_ce_se = 0;
    ctrl_seg.opmod_idx_opcode = HTOBE32((wqe_idx << 8) | MLX5_OPCODE_RDMA_WRITE);

    uint32_t *dst = (uint32_t *)ctrl_seg_ptr;
    uint32_t *src = (uint32_t *)&ctrl_seg;
    for (int i = 0; i < sizeof(*ctrl_seg_ptr) / sizeof(uint32_t); ++i)
        ibgda_store_relaxed(&dst[i], src[i]);

    if (av_seg_size > 0) {
        dst = (uint32_t *)av_seg_ptr;
        src = (uint32_t *)dct;
        for (int i = 0; i < av_seg_size / sizeof(uint32_t); ++i)
            ibgda_store_relaxed(&dst[i], src[i]);
    }

    dst = (uint32_t *)raddr_seg_ptr;
    src = (uint32_t *)&raddr_seg;
    for (int i = 0; i < sizeof(*raddr_seg_ptr) / sizeof(uint32_t); ++i)
        ibgda_store_relaxed(&dst[i], src[i]);

    dst = (uint32_t *)inl_seg_ptr;
    src = (uint32_t *)&inl_seg;
    for (int i = 0; i < sizeof(*inl_seg_ptr) / sizeof(uint32_t); ++i)
        ibgda_store_relaxed(&dst[i], src[i]);

    uint32_t my_base_data_idx = my_tid * bytes;
    if (bytes <= 4) {
        T *wqe_data_ptr;
        if (my_base_data_idx < remaining_size_for_data_in_first_wqebb)
            wqe_data_ptr = (T *)((uintptr_t)inl_seg_ptr + sizeof(*inl_seg_ptr) + my_base_data_idx);
        else {
            uint32_t my_data_idx = my_base_data_idx - remaining_size_for_data_in_first_wqebb;
            int my_data_in_wqe_idx = my_data_idx / 64 + 1;
            my_data_idx &= (64 - 1);  // my_data_idx % 64
            wqe_data_ptr = (T *)((uintptr_t)out_wqes[my_data_in_wqe_idx] + my_data_idx);
        }
        ibgda_store_relaxed(wqe_data_ptr, val);
    } else {
        // wqe_data_ptr is 4-byte aligned but not 8-byte aligned.
        assert(likely(bytes == 8 && qp->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_RC));
        uint32_t *wqe_data_ptr;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            uint32_t my_data_idx = my_base_data_idx + (i * 4);
            if (my_data_idx < remaining_size_for_data_in_first_wqebb)
                wqe_data_ptr =
                    (uint32_t *)((uintptr_t)inl_seg_ptr + sizeof(*inl_seg_ptr) + my_data_idx);
            else {
                uint32_t my_idx = my_data_idx - remaining_size_for_data_in_first_wqebb;
                int my_data_in_wqe_idx = my_idx / 64 + 1;
                my_idx &= (64 - 1);  // my_idx % 64
                wqe_data_ptr = (uint32_t *)((uintptr_t)out_wqes[my_data_in_wqe_idx] + my_idx);
            }
            ibgda_store_relaxed(wqe_data_ptr, *((uint32_t *)&val + i));
        }
    }

    wqe_idx += nop_relative_wqe_idx;
    ctrl_seg = {
        0,
    };
    ctrl_seg.qpn_ds = HTOBE32((qp->qpn << 8) | 1);
    ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
    ctrl_seg.opmod_idx_opcode = HTOBE32((wqe_idx << 8) | MLX5_OPCODE_NOP);

    ctrl_seg_ptr = (ibgda_ctrl_seg_t *)out_wqes[nop_relative_wqe_idx];

    dst = (uint32_t *)ctrl_seg_ptr;
    src = (uint32_t *)&ctrl_seg;
    for (int i = 0; i < sizeof(*ctrl_seg_ptr) / sizeof(uint32_t); ++i)
        ibgda_store_relaxed(&dst[i], src[i]);
}

/**
 * For DCI with sizeof(T) == 8 only.
 * DC supports up to 16 DS WQE.
 * For sizeof(T) == 8, we split to two WQEs of inline size 8 * 16
 */
template <typename T>
__device__ static inline void ibgda_write_rdma_write_inl_wqe_combine_warp_for_dci_8B(
    nvshmemi_ibgda_device_qp_t *dci, nvshmemi_ibgda_device_dct_t *dct, const T val, uint64_t _raddr,
    __be32 rkey, uint16_t _wqe_idx, int my_tid, void **out_wqes) {
    assert(likely(sizeof(T) == 8 && dci->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI));

    // base_tid = my_tid >= 16 ? 16 : 0;
    int base_tid = my_tid & (~0xF);

    // base_wqe_idx = base_tid / 4;
    int base_out_wqe_idx = base_tid >> 2;

    uint16_t wqe_idx = _wqe_idx + base_out_wqe_idx;

    ibgda_ctrl_seg_t ctrl_seg;
    struct mlx5_wqe_raddr_seg raddr_seg;
    struct mlx5_wqe_inl_data_seg inl_seg;

    ibgda_ctrl_seg_t *ctrl_seg_ptr = (ibgda_ctrl_seg_t *)out_wqes[base_out_wqe_idx];
    void *av_seg_ptr = (void *)((uintptr_t)ctrl_seg_ptr + sizeof(*ctrl_seg_ptr));
    struct mlx5_wqe_raddr_seg *raddr_seg_ptr;
    struct mlx5_wqe_inl_data_seg *inl_seg_ptr;
    uint32_t *wqe_data_ptr;

    size_t av_seg_size;
    int ds = ibgda_get_ds_in_inl_combine_warp<uint32_t, NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI>();

    uint64_t raddr = _raddr - ((my_tid - base_tid) * 8);

    av_seg_size = sizeof(ibgda_half_av_seg_t);
    raddr_seg_ptr = (struct mlx5_wqe_raddr_seg *)((uintptr_t)av_seg_ptr + av_seg_size);
    inl_seg_ptr =
        (struct mlx5_wqe_inl_data_seg *)((uintptr_t)raddr_seg_ptr + sizeof(*raddr_seg_ptr));

    raddr_seg.raddr = HTOBE64(raddr);
    raddr_seg.rkey = rkey;
    raddr_seg.reserved = 0;

    inl_seg.byte_count = HTOBE32((8 * warpSize / 2) | MLX5_INLINE_SEG);

    ctrl_seg = {
        0,
    };
    ctrl_seg.qpn_ds = HTOBE32((dci->qpn << 8) | ds);
    // ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
    // This RDMA WRITE wqe will not get CQ update to avoid dynamic size calculation in poll_cq.
    // Instead, the NO-OP wqe (last one) will get CQ update because it is always 1 WQEBB.
    ctrl_seg.fm_ce_se = 0;
    ctrl_seg.opmod_idx_opcode = HTOBE32((wqe_idx << 8) | MLX5_OPCODE_RDMA_WRITE);

    uint32_t *dst = (uint32_t *)ctrl_seg_ptr;
    uint32_t *src = (uint32_t *)&ctrl_seg;
    for (int i = 0; i < sizeof(*ctrl_seg_ptr) / sizeof(uint32_t); ++i)
        ibgda_store_relaxed(&dst[i], src[i]);

    dst = (uint32_t *)av_seg_ptr;
    src = (uint32_t *)dct;
    for (int i = 0; i < av_seg_size / sizeof(uint32_t); ++i) ibgda_store_relaxed(&dst[i], src[i]);

    dst = (uint32_t *)raddr_seg_ptr;
    src = (uint32_t *)&raddr_seg;
    for (int i = 0; i < sizeof(*raddr_seg_ptr) / sizeof(uint32_t); ++i)
        ibgda_store_relaxed(&dst[i], src[i]);

    dst = (uint32_t *)inl_seg_ptr;
    src = (uint32_t *)&inl_seg;
    for (int i = 0; i < sizeof(*inl_seg_ptr) / sizeof(uint32_t); ++i)
        ibgda_store_relaxed(&dst[i], src[i]);

    for (int i = 0; i < 2; ++i) {
        uint32_t my_data_idx = ((my_tid - base_tid) * 2 + i) * 4;
        if (my_data_idx < 12)
            wqe_data_ptr =
                (uint32_t *)((uintptr_t)inl_seg_ptr + sizeof(*inl_seg_ptr) + my_data_idx);
        else {
            my_data_idx -= 12;
            int my_data_in_wqe_idx = my_data_idx / 64 + 1;
            my_data_idx &= (64 - 1);  // my_data_idx % 64
            wqe_data_ptr = (uint32_t *)((uintptr_t)out_wqes[my_data_in_wqe_idx + base_out_wqe_idx] +
                                        my_data_idx);
        }

        ibgda_store_relaxed(wqe_data_ptr, ((uint32_t *)&val)[i]);
    }

    uint32_t nop_relative_wqe_idx =
        ibgda_get_num_wqes_in_inl_combine_warp<uint32_t, NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI>() - 1;

    wqe_idx += nop_relative_wqe_idx;
    ctrl_seg = {
        0,
    };
    ctrl_seg.qpn_ds = HTOBE32((dci->qpn << 8) | 1);
    ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
    ctrl_seg.opmod_idx_opcode = HTOBE32((wqe_idx << 8) | MLX5_OPCODE_NOP);

    ctrl_seg_ptr = (ibgda_ctrl_seg_t *)out_wqes[nop_relative_wqe_idx + base_out_wqe_idx];

    dst = (uint32_t *)ctrl_seg_ptr;
    src = (uint32_t *)&ctrl_seg;
    for (int i = 0; i < sizeof(*ctrl_seg_ptr) / sizeof(uint32_t); ++i)
        ibgda_store_relaxed(&dst[i], src[i]);
}

template <bool support_half_av_seg>
__device__ static inline void ibgda_write_rdma_read_wqe(nvshmemi_ibgda_device_qp_t *qp,
                                                        nvshmemi_ibgda_device_dct_t *dct,
                                                        uint64_t laddr, __be32 lkey, uint64_t raddr,
                                                        __be32 rkey, uint32_t bytes,
                                                        uint16_t wqe_idx, uint8_t fm_ce_se,
                                                        void **out_wqes) {
    ibgda_ctrl_seg_t ctrl_seg;
    struct mlx5_wqe_raddr_seg raddr_seg;
    struct mlx5_wqe_data_seg data_seg;

    ibgda_ctrl_seg_t *ctrl_seg_ptr = (ibgda_ctrl_seg_t *)out_wqes[0];
    void *av_seg_ptr = (void *)((uintptr_t)ctrl_seg_ptr + sizeof(*ctrl_seg_ptr));
    struct mlx5_wqe_raddr_seg *raddr_seg_ptr;
    struct mlx5_wqe_data_seg *data_seg_ptr;

    size_t av_seg_size;
    int ds;

    if (qp->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI) {
        if (support_half_av_seg) {
            ds = 4;
            av_seg_size = sizeof(ibgda_half_av_seg_t);
            raddr_seg_ptr = (struct mlx5_wqe_raddr_seg *)((uintptr_t)av_seg_ptr + av_seg_size);
        } else {
            ds = 6;
            av_seg_size = sizeof(struct mlx5_wqe_av);
            raddr_seg_ptr = (struct mlx5_wqe_raddr_seg *)out_wqes[1];
        }
    } else {
        ds = 3;
        av_seg_size = 0;
        raddr_seg_ptr = (struct mlx5_wqe_raddr_seg *)((uintptr_t)av_seg_ptr + av_seg_size);
    }
    data_seg_ptr = (struct mlx5_wqe_data_seg *)((uintptr_t)raddr_seg_ptr + sizeof(*raddr_seg_ptr));

    raddr_seg.raddr = HTOBE64(raddr);
    raddr_seg.rkey = rkey;
    raddr_seg.reserved = 0;

    data_seg.byte_count = HTOBE32(bytes);
    data_seg.lkey = lkey;
    data_seg.addr = HTOBE64(laddr);

    ctrl_seg = {
        0,
    };
    ctrl_seg.qpn_ds = HTOBE32((qp->qpn << 8) | ds);
    ctrl_seg.fm_ce_se = fm_ce_se;
    ctrl_seg.opmod_idx_opcode = HTOBE32((wqe_idx << 8) | MLX5_OPCODE_RDMA_READ);

    uint32_t *dst = (uint32_t *)ctrl_seg_ptr;
    uint32_t *src = (uint32_t *)&ctrl_seg;
    for (int i = 0; i < sizeof(*ctrl_seg_ptr) / sizeof(uint32_t); ++i)
        ibgda_store_relaxed(&dst[i], src[i]);

    if (av_seg_size > 0) {
        dst = (uint32_t *)av_seg_ptr;
        src = (uint32_t *)dct;
        for (int i = 0; i < av_seg_size / sizeof(uint32_t); ++i)
            ibgda_store_relaxed(&dst[i], src[i]);
    }

    dst = (uint32_t *)raddr_seg_ptr;
    src = (uint32_t *)&raddr_seg;
    for (int i = 0; i < sizeof(*raddr_seg_ptr) / sizeof(uint32_t); ++i)
        ibgda_store_relaxed(&dst[i], src[i]);

    dst = (uint32_t *)data_seg_ptr;
    src = (uint32_t *)&data_seg;
    for (int i = 0; i < sizeof(*data_seg_ptr) / sizeof(uint32_t); ++i)
        ibgda_store_relaxed(&dst[i], src[i]);
}

template <typename T>
__device__ static inline uint32_t ibgda_get_num_wqes_in_atomic(
    nvshmemi_amo_t amo_op, nvshmemi_ibgda_device_qp_type_t qp_type) {
    if (qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI)
        return 2;
    else if (sizeof(T) == 8) {
        // RC
        switch (amo_op) {
            case NVSHMEMI_AMO_SIGNAL:
            case NVSHMEMI_AMO_SIGNAL_SET:
            case NVSHMEMI_AMO_SWAP:
            case NVSHMEMI_AMO_SET:
            case NVSHMEMI_AMO_FETCH_AND:
            case NVSHMEMI_AMO_AND:
            case NVSHMEMI_AMO_FETCH_OR:
            case NVSHMEMI_AMO_OR:
                return 2;
        }
    }
    return 1;
}

template <bool support_half_av_seg>
__device__ static inline void ibgda_write_atomic_wqe(
    nvshmemi_ibgda_device_qp_t *qp, nvshmemi_ibgda_device_dct_t *dct, const void *val_1,
    const void *val_2, uint64_t laddr, __be32 lkey, uint64_t raddr, __be32 rkey, uint32_t bytes,
    uint16_t wqe_idx, nvshmemi_amo_t amo_op, uint8_t fm_ce_se, void **out_wqes) {
    ibgda_ctrl_seg_t ctrl_seg;
    struct mlx5_wqe_raddr_seg raddr_seg;
    struct mlx5_wqe_atomic_seg atomic_seg_1;
    struct mlx5_wqe_atomic_seg atomic_seg_2;
    struct mlx5_wqe_data_seg data_seg;

    ibgda_ctrl_seg_t *ctrl_seg_ptr = (ibgda_ctrl_seg_t *)out_wqes[0];
    void *av_seg_ptr = (void *)((uintptr_t)ctrl_seg_ptr + sizeof(*ctrl_seg_ptr));
    struct mlx5_wqe_raddr_seg *raddr_seg_ptr;
    struct mlx5_wqe_atomic_seg *atomic_seg_1_ptr;
    struct mlx5_wqe_atomic_seg *atomic_seg_2_ptr;
    struct mlx5_wqe_data_seg *data_seg_ptr;

    size_t av_seg_size;
    int ds;

    if (qp->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI) {
        if (support_half_av_seg) {
            ds = 5;
            av_seg_size = sizeof(ibgda_half_av_seg_t);
            raddr_seg_ptr = (struct mlx5_wqe_raddr_seg *)((uintptr_t)av_seg_ptr + av_seg_size);
            atomic_seg_1_ptr =
                (struct mlx5_wqe_atomic_seg *)((uintptr_t)raddr_seg_ptr + sizeof(*raddr_seg_ptr));
            atomic_seg_2_ptr = (struct mlx5_wqe_atomic_seg *)out_wqes[1];
        } else {
            ds = 7;
            av_seg_size = sizeof(struct mlx5_wqe_av);
            raddr_seg_ptr = (struct mlx5_wqe_raddr_seg *)out_wqes[1];
            atomic_seg_1_ptr =
                (struct mlx5_wqe_atomic_seg *)((uintptr_t)raddr_seg_ptr + sizeof(*raddr_seg_ptr));
            atomic_seg_2_ptr = (struct mlx5_wqe_atomic_seg *)((uintptr_t)atomic_seg_1_ptr +
                                                              sizeof(*atomic_seg_1_ptr));
        }
    } else {
        ds = 4;
        av_seg_size = 0;
        raddr_seg_ptr = (struct mlx5_wqe_raddr_seg *)((uintptr_t)av_seg_ptr + av_seg_size);
        atomic_seg_1_ptr =
            (struct mlx5_wqe_atomic_seg *)((uintptr_t)raddr_seg_ptr + sizeof(*raddr_seg_ptr));
        atomic_seg_2_ptr =
            (struct mlx5_wqe_atomic_seg *)((uintptr_t)atomic_seg_1_ptr + sizeof(*atomic_seg_1_ptr));
    }
    data_seg_ptr = (struct mlx5_wqe_data_seg *)atomic_seg_2_ptr;

    raddr_seg.raddr = HTOBE64(raddr);
    raddr_seg.rkey = rkey;
    raddr_seg.reserved = 0;

    ctrl_seg = {
        0,
    };

    assert(likely(bytes == 4 || bytes == 8));
    switch (amo_op) {
        case NVSHMEMI_AMO_FETCH_INC:
        case NVSHMEMI_AMO_INC: {
            if (bytes == 4) {
                ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqe_idx << 8) |
                                                    IBGDA_4_BYTE_EXT_AMO_OPMOD);

                ibgda_atomic_32_masked_fa_seg_t *atomic_32_masked_fa_seg =
                    (ibgda_atomic_32_masked_fa_seg_t *)&atomic_seg_1;
                atomic_32_masked_fa_seg->add_data = HTOBE32((uint32_t)1);
                atomic_32_masked_fa_seg->field_boundary = 0;
            } else {
                ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqe_idx << 8) |
                                                    IBGDA_8_BYTE_EXT_AMO_OPMOD);

                ibgda_atomic_64_masked_fa_seg_t *atomic_64_masked_fa_seg =
                    (ibgda_atomic_64_masked_fa_seg_t *)&atomic_seg_1;
                atomic_64_masked_fa_seg->add_data = HTOBE64((uint64_t)1);
                atomic_64_masked_fa_seg->field_boundary = 0;
            }
            break;
        }
        case NVSHMEMI_AMO_SIGNAL:
        case NVSHMEMI_AMO_SIGNAL_SET:
        case NVSHMEMI_AMO_SWAP:
        case NVSHMEMI_AMO_SET: {
            if (bytes == 4) {
                ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_CS | (wqe_idx << 8) |
                                                    IBGDA_4_BYTE_EXT_AMO_OPMOD);

                ibgda_atomic_32_masked_cs_seg_t *atomic_32_masked_cs_seg =
                    (ibgda_atomic_32_masked_cs_seg_t *)&atomic_seg_1;
                atomic_32_masked_cs_seg->swap_data = HTOBE32(*(uint32_t *)val_1);
                atomic_32_masked_cs_seg->compare_data = 0;
                atomic_32_masked_cs_seg->compare_mask = 0;
                atomic_32_masked_cs_seg->swap_mask = UINT32_MAX;
            } else {
                ++ds;
                ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_CS | (wqe_idx << 8) |
                                                    IBGDA_8_BYTE_EXT_AMO_OPMOD);

                ibgda_atomic_64_masked_cs_seg_t *atomic_64_masked_cs_data_seg =
                    (ibgda_atomic_64_masked_cs_seg_t *)&atomic_seg_1;
                atomic_64_masked_cs_data_seg->swap = HTOBE64(*(uint64_t *)val_1);
                atomic_64_masked_cs_data_seg->compare = 0;

                ibgda_atomic_64_masked_cs_seg_t *atomic_64_masked_cs_mask_seg =
                    (ibgda_atomic_64_masked_cs_seg_t *)&atomic_seg_2;
                atomic_64_masked_cs_mask_seg->swap = UINT64_MAX;
                atomic_64_masked_cs_mask_seg->compare = 0;

                if (qp->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI)
                    data_seg_ptr =
                        (struct mlx5_wqe_data_seg *)((uintptr_t)atomic_seg_2_ptr +
                                                     sizeof(*atomic_64_masked_cs_mask_seg));
                else
                    data_seg_ptr = (struct mlx5_wqe_data_seg *)out_wqes[1];
            }
            break;
        }
        case NVSHMEMI_AMO_SIGNAL_ADD:
        case NVSHMEMI_AMO_ADD: {
            if (bytes == 4) {
                ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqe_idx << 8) |
                                                    IBGDA_4_BYTE_EXT_AMO_OPMOD);

                ibgda_atomic_32_masked_fa_seg_t *atomic_32_masked_fa_seg =
                    (ibgda_atomic_32_masked_fa_seg_t *)&atomic_seg_1;
                atomic_32_masked_fa_seg->add_data = HTOBE32(*(uint32_t *)val_1);
                atomic_32_masked_fa_seg->field_boundary = 0;
            } else {
                ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqe_idx << 8) |
                                                    IBGDA_8_BYTE_EXT_AMO_OPMOD);

                ibgda_atomic_64_masked_fa_seg_t *atomic_64_masked_fa_seg =
                    (ibgda_atomic_64_masked_fa_seg_t *)&atomic_seg_1;
                atomic_64_masked_fa_seg->add_data = HTOBE64(*(uint64_t *)val_1);
                atomic_64_masked_fa_seg->field_boundary = 0;
            }
            break;
        }
        case NVSHMEMI_AMO_FETCH_AND:
        case NVSHMEMI_AMO_AND: {
            if (bytes == 4) {
                ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_CS | (wqe_idx << 8) |
                                                    IBGDA_4_BYTE_EXT_AMO_OPMOD);

                ibgda_atomic_32_masked_cs_seg_t *atomic_32_masked_cs_seg =
                    (ibgda_atomic_32_masked_cs_seg_t *)&atomic_seg_1;
                atomic_32_masked_cs_seg->swap_data = HTOBE32(*(uint32_t *)val_1);
                atomic_32_masked_cs_seg->compare_data = 0;
                atomic_32_masked_cs_seg->compare_mask = 0;
                atomic_32_masked_cs_seg->swap_mask = HTOBE32(~(*(uint32_t *)val_1));
            } else {
                ++ds;
                ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_CS | (wqe_idx << 8) |
                                                    IBGDA_8_BYTE_EXT_AMO_OPMOD);

                ibgda_atomic_64_masked_cs_seg_t *atomic_64_masked_cs_data_seg =
                    (ibgda_atomic_64_masked_cs_seg_t *)&atomic_seg_1;
                atomic_64_masked_cs_data_seg->swap = HTOBE64(*(uint64_t *)val_1);
                atomic_64_masked_cs_data_seg->compare = 0;

                ibgda_atomic_64_masked_cs_seg_t *atomic_64_masked_cs_mask_seg =
                    (ibgda_atomic_64_masked_cs_seg_t *)&atomic_seg_2;
                atomic_64_masked_cs_mask_seg->swap = HTOBE64(~(*(uint64_t *)val_1));
                atomic_64_masked_cs_mask_seg->compare = 0;

                if (qp->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI)
                    data_seg_ptr =
                        (struct mlx5_wqe_data_seg *)((uintptr_t)atomic_seg_2_ptr +
                                                     sizeof(*atomic_64_masked_cs_mask_seg));
                else
                    data_seg_ptr = (struct mlx5_wqe_data_seg *)out_wqes[1];
            }
            break;
        }
        case NVSHMEMI_AMO_FETCH_OR:
        case NVSHMEMI_AMO_OR: {
            if (bytes == 4) {
                ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_CS | (wqe_idx << 8) |
                                                    IBGDA_4_BYTE_EXT_AMO_OPMOD);

                ibgda_atomic_32_masked_cs_seg_t *atomic_32_masked_cs_seg =
                    (ibgda_atomic_32_masked_cs_seg_t *)&atomic_seg_1;
                atomic_32_masked_cs_seg->swap_data = HTOBE32(*(uint32_t *)val_1);
                atomic_32_masked_cs_seg->compare_data = 0;
                atomic_32_masked_cs_seg->compare_mask = 0;
                atomic_32_masked_cs_seg->swap_mask = HTOBE32(*(uint32_t *)val_1);
            } else {
                ++ds;
                ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_CS | (wqe_idx << 8) |
                                                    IBGDA_8_BYTE_EXT_AMO_OPMOD);

                ibgda_atomic_64_masked_cs_seg_t *atomic_64_masked_cs_data_seg =
                    (ibgda_atomic_64_masked_cs_seg_t *)&atomic_seg_1;
                atomic_64_masked_cs_data_seg->swap = HTOBE64(*(uint64_t *)val_1);
                atomic_64_masked_cs_data_seg->compare = 0;

                ibgda_atomic_64_masked_cs_seg_t *atomic_64_masked_cs_mask_seg =
                    (ibgda_atomic_64_masked_cs_seg_t *)&atomic_seg_2;
                atomic_64_masked_cs_mask_seg->swap = HTOBE64(*(uint64_t *)val_1);
                atomic_64_masked_cs_mask_seg->compare = 0;

                if (qp->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI)
                    data_seg_ptr =
                        (struct mlx5_wqe_data_seg *)((uintptr_t)atomic_seg_2_ptr +
                                                     sizeof(*atomic_64_masked_cs_mask_seg));
                else
                    data_seg_ptr = (struct mlx5_wqe_data_seg *)out_wqes[1];
            }
            break;
        }
        case NVSHMEMI_AMO_FETCH_XOR:
        case NVSHMEMI_AMO_XOR: {
            if (bytes == 4) {
                ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqe_idx << 8) |
                                                    IBGDA_4_BYTE_EXT_AMO_OPMOD);

                ibgda_atomic_32_masked_fa_seg_t *atomic_32_masked_fa_seg =
                    (ibgda_atomic_32_masked_fa_seg_t *)&atomic_seg_1;
                atomic_32_masked_fa_seg->add_data = HTOBE32(*(uint32_t *)val_1);
                atomic_32_masked_fa_seg->field_boundary = UINT32_MAX;
            } else {
                ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqe_idx << 8) |
                                                    IBGDA_8_BYTE_EXT_AMO_OPMOD);

                ibgda_atomic_64_masked_fa_seg_t *atomic_64_masked_fa_seg =
                    (ibgda_atomic_64_masked_fa_seg_t *)&atomic_seg_1;
                atomic_64_masked_fa_seg->add_data = HTOBE64(*(uint64_t *)val_1);
                atomic_64_masked_fa_seg->field_boundary = UINT64_MAX;
            }
            break;
        }
        case NVSHMEMI_AMO_FETCH: {
            if (bytes == 4) {
                ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqe_idx << 8) |
                                                    IBGDA_4_BYTE_EXT_AMO_OPMOD);

                ibgda_atomic_32_masked_fa_seg_t *atomic_32_masked_fa_seg =
                    (ibgda_atomic_32_masked_fa_seg_t *)&atomic_seg_1;
                atomic_32_masked_fa_seg->add_data = 0;
                atomic_32_masked_fa_seg->field_boundary = 0;
            } else {
                ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqe_idx << 8) |
                                                    IBGDA_8_BYTE_EXT_AMO_OPMOD);

                ibgda_atomic_64_masked_fa_seg_t *atomic_64_masked_fa_seg =
                    (ibgda_atomic_64_masked_fa_seg_t *)&atomic_seg_1;
                atomic_64_masked_fa_seg->add_data = 0;
                atomic_64_masked_fa_seg->field_boundary = 0;
            }
            break;
        }
        case NVSHMEMI_AMO_FETCH_ADD: {
            if (bytes == 4) {
                ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqe_idx << 8) |
                                                    IBGDA_4_BYTE_EXT_AMO_OPMOD);

                ibgda_atomic_32_masked_fa_seg_t *atomic_32_masked_fa_seg =
                    (ibgda_atomic_32_masked_fa_seg_t *)&atomic_seg_1;
                atomic_32_masked_fa_seg->add_data = HTOBE32(*(uint32_t *)val_1);
                atomic_32_masked_fa_seg->field_boundary = 0;
            } else {
                ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_FA | (wqe_idx << 8));
                atomic_seg_1.swap_add = HTOBE64(*(uint64_t *)val_1);
            }
            break;
        }
        case NVSHMEMI_AMO_COMPARE_SWAP: {
            if (bytes == 4) {
                ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_CS | (wqe_idx << 8) |
                                                    IBGDA_4_BYTE_EXT_AMO_OPMOD);

                ibgda_atomic_32_masked_cs_seg_t *atomic_32_masked_cs_seg =
                    (ibgda_atomic_32_masked_cs_seg_t *)&atomic_seg_1;
                atomic_32_masked_cs_seg->swap_data = HTOBE32(*(uint32_t *)val_1);
                atomic_32_masked_cs_seg->compare_data = HTOBE32(*(uint32_t *)val_2);
                atomic_32_masked_cs_seg->compare_mask = UINT32_MAX;
                atomic_32_masked_cs_seg->swap_mask = UINT32_MAX;
            } else {
                ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_CS | (wqe_idx << 8));
                atomic_seg_1.swap_add = HTOBE64(*(uint64_t *)val_1);
                atomic_seg_1.compare = HTOBE64(*(uint64_t *)val_2);
            }
            break;
        }
        default: { assert(0); }
    }

    ctrl_seg.qpn_ds = HTOBE32((qp->qpn << 8) | ds);

    data_seg.byte_count = HTOBE32(bytes);
    data_seg.lkey = lkey;
    data_seg.addr = HTOBE64(laddr);

    ctrl_seg.fm_ce_se = fm_ce_se;

    uint32_t *dst = (uint32_t *)ctrl_seg_ptr;
    uint32_t *src = (uint32_t *)&ctrl_seg;
    for (int i = 0; i < sizeof(*ctrl_seg_ptr) / sizeof(uint32_t); ++i)
        ibgda_store_relaxed(&dst[i], src[i]);

    if (av_seg_size > 0) {
        dst = (uint32_t *)av_seg_ptr;
        src = (uint32_t *)dct;
        for (int i = 0; i < av_seg_size / sizeof(uint32_t); ++i)
            ibgda_store_relaxed(&dst[i], src[i]);
    }

    dst = (uint32_t *)raddr_seg_ptr;
    src = (uint32_t *)&raddr_seg;
    for (int i = 0; i < sizeof(*raddr_seg_ptr) / sizeof(uint32_t); ++i)
        ibgda_store_relaxed(&dst[i], src[i]);

    dst = (uint32_t *)atomic_seg_1_ptr;
    src = (uint32_t *)&atomic_seg_1;
    for (int i = 0; i < sizeof(*atomic_seg_1_ptr) / sizeof(uint32_t); ++i)
        ibgda_store_relaxed(&dst[i], src[i]);

    dst = (uint32_t *)atomic_seg_2_ptr;
    src = (uint32_t *)&atomic_seg_2;
    for (int i = 0; i < sizeof(*atomic_seg_2_ptr) / sizeof(uint32_t); ++i)
        ibgda_store_relaxed(&dst[i], src[i]);

    dst = (uint32_t *)data_seg_ptr;
    src = (uint32_t *)&data_seg;
    for (int i = 0; i < sizeof(*data_seg_ptr) / sizeof(uint32_t); ++i)
        ibgda_store_relaxed(&dst[i], src[i]);
}

__device__ static inline void ibgda_update_dbr(nvshmemi_ibgda_device_qp_t *qp,
                                               uint32_t dbrec_head) {
    // DBREC contains the index of the next empty WQEBB.
    __be32 dbrec_val;
    __be32 *dbrec_ptr = qp->tx_wq.dbrec;

    // This is equivalent to
    // WRITE_ONCE(dbrec_ptr, HTOBE32(dbrec_head & 0xffff));
    asm volatile(
        "{\n\t"
        ".reg .b32 mask1;\n\t"
        ".reg .b32 dbrec_head_16b;\n\t"
        ".reg .b32 ign;\n\t"
        ".reg .b32 mask2;\n\t"
        "mov.b32 mask1, 0xffff;\n\t"
        "mov.b32 mask2, 0x123;\n\t"
        "and.b32 dbrec_head_16b, %1, mask1;\n\t"
        "prmt.b32 %0, dbrec_head_16b, ign, mask2;\n\t"
        "}"
        : "=r"(dbrec_val)
        : "r"(dbrec_head));
    ibgda_store_release(dbrec_ptr, dbrec_val);
}

__device__ static inline void ibgda_ring_db(nvshmemi_ibgda_device_qp_t *qp, uint16_t prod_idx) {
    uint64_t *bf_ptr = (uint64_t *)qp->tx_wq.bf;
    ibgda_ctrl_seg_t ctrl_seg = {.opmod_idx_opcode = HTOBE32(prod_idx << 8),
                                 .qpn_ds = HTOBE32(qp->qpn << 8)};

    ibgda_store_release(bf_ptr, *((uint64_t *)&ctrl_seg));
}

template <bool need_strong_flush>
__device__ static inline void ibgda_post_send(nvshmemi_ibgda_device_qp_t *qp,
                                              uint64_t new_prod_idx) {
    nvshmemi_ibgda_device_qp_management_t *mvars = &qp->mvars;
    uint64_t old_prod_idx;

    // Update prod_idx before ringing the db so that we know which index is needed in quiet/fence.
    ibgda_lock_acquire<NVSHMEMI_THREADGROUP_THREAD>(&mvars->post_send_lock);

    if (need_strong_flush)
        old_prod_idx = atomicMax((unsigned long long int *)&mvars->tx_wq.prod_idx,
                                 (unsigned long long int)new_prod_idx);
    else
        old_prod_idx = atomicMax_block((unsigned long long int *)&mvars->tx_wq.prod_idx,
                                       (unsigned long long int)new_prod_idx);

    if (likely(new_prod_idx > old_prod_idx)) {
        IBGDA_MEMBAR();
        ibgda_update_dbr(qp, new_prod_idx);
        IBGDA_MEMBAR();
        ibgda_ring_db(qp, new_prod_idx);
    }

    ibgda_lock_release<NVSHMEMI_THREADGROUP_THREAD>(&mvars->post_send_lock);
}

// If `qp` is shared among CTAs, need_strong_flush must be set to true because
// we must push prior writes from this CTA to L2 before coalescing DB.
template <bool need_strong_flush>
__device__ static inline void ibgda_submit_requests(nvshmemi_ibgda_device_qp_t *qp,
                                                    uint64_t base_wqe_idx, uint16_t num_wqes) {
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();
    nvshmemi_ibgda_device_qp_management_t *mvars = &qp->mvars;
    uint64_t mask = ~((uint64_t)(state->num_requests_in_batch - 1));

    uint64_t new_wqe_idx = base_wqe_idx + num_wqes;

    unsigned long long int *ready_idx =
        (unsigned long long int *)(state->use_async_postsend ? qp->tx_wq.prod_idx
                                                             : &mvars->tx_wq.ready_head);

    // WQE writes must be finished first.
    if (need_strong_flush)
        // membar from a different CTA does not push prior writes of this CTA.
        // We must push them out first because a different CTA might post-send for us.
        IBGDA_MEMBAR_NO_OPTIMIZATION();
    else
        // It is ok for those wqes to not be visible to the GPU scope yet.
        // ibgda_post_send will take care of that (if we choose to call it).
        IBGDA_MFENCE();

    // Wait for prior WQE slots to be filled first.
    // They might not be post-sent yet.
    if (need_strong_flush)
        while (atomicCAS(ready_idx, (unsigned long long int)base_wqe_idx,
                         (unsigned long long int)new_wqe_idx) != base_wqe_idx)
            ;  // wait here
    else
        while (atomicCAS_block(ready_idx, (unsigned long long int)base_wqe_idx,
                               (unsigned long long int)new_wqe_idx) != base_wqe_idx)
            ;  // wait here

    IBGDA_MFENCE();

    if (!state->use_async_postsend) {
        bool do_post_send =
            (new_wqe_idx ==
             ibgda_atomic_read(&mvars->tx_wq.resv_head))  // No concurrent submissions
            || ((base_wqe_idx & mask) !=
                (new_wqe_idx & mask))  // Num of not-yet-posted wqes is beyond the threshold.
            || (num_wqes >= state->num_requests_in_batch);  // The number of wqes in this submission
                                                            // reaches the threshold.

        if (do_post_send) ibgda_post_send<need_strong_flush>(qp, new_wqe_idx);
    }
}

__device__ static inline uint64_t ibgda_quiet(nvshmemi_ibgda_device_qp_t *qp) {
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();
    uint64_t prod_idx = state->use_async_postsend ? ibgda_atomic_read(qp->tx_wq.prod_idx)
                                                  : ibgda_atomic_read(&qp->mvars.tx_wq.ready_head);
    nvshmemi_ibgda_device_cq_t cq = *qp->tx_wq.cq;

    int err = 0;
    int status = ibgda_poll_cq(&cq, prod_idx, &err);
    // TODO: Integrate the error handler with the core NVSHMEM
#ifdef NVSHMEM_IBGDA_DEBUG
    if (status) {
        printf("ibgda_poll_cq failed with error=%d.\n", err);
    }
#endif
    assert(likely(status == 0));
    return prod_idx;
}

__device__ static inline void ibgda_wait_for_slot_availability(nvshmemi_ibgda_device_qp_t *qp,
                                                               uint64_t wqe_idx) {
    int status = 0;
    int err = 0;
    uint16_t nwqes = qp->tx_wq.nwqes;

    // We don't want wqe_idx - nwqes to wraparound.
    if (likely(wqe_idx >= nwqes)) {
        nvshmemi_ibgda_device_cq_t cq = *qp->tx_wq.cq;
        status = ibgda_poll_cq(&cq, wqe_idx - nwqes, &err);
        // TODO: Integrate the error handler with the core NVSHMEM
        if (status) {
            printf("ibgda_poll_cq failed with error=%d.\n", err);
        }
        assert(likely(status == 0));
    }
    IBGDA_MFENCE();
}

__device__ static inline int ibgda_get_proxy_pe(int pe) {
    if (nvshmemi_device_state_d.enable_rail_opt == 1)
        return (pe / nvshmemi_device_state_d.node_npes) * nvshmemi_device_state_d.node_npes +
               nvshmemi_device_state_d.node_mype;
    return pe;
}

__device__ static inline uint32_t ibgda_get_dct_id(int pe, int dev_idx) {
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();
    uint32_t id = ibgda_get_ctaid();
    /* There are ndcts_per_pe * state->num_devices_initialized per pe. */
    uint32_t dct_id = (pe * state->ndcts_per_pe * state->num_devices_initialized) +
                      (((id % state->ndcts_per_pe) * state->num_devices_initialized) + dev_idx);
    return dct_id;
}

__device__ static inline nvshmemi_ibgda_device_dct_t *ibgda_get_dct(int pe, int dev_idx) {
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();
    uint32_t dct_idx = ibgda_get_dct_id(pe, dev_idx);

    if (dct_idx < NVSHMEMI_IBGDA_MAX_CONST_DCTS) return &state->constmem.dcts[dct_idx];

    return &state->globalmem.dcts[dct_idx - NVSHMEMI_IBGDA_MAX_CONST_DCTS];
}

__device__ static inline nvshmemi_ibgda_device_qp_t *ibgda_get_dci(int pe,
                                                                   bool *out_shared_among_ctas) {
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();
    uint32_t id;
    uint32_t dev_offset;
    bool shared_among_ctas = false;
    uint32_t warpid = nvshmemi_thread_id_in_block() / nvshmemi_warp_size();

    switch (state->dci_map_type) {
        case NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_CTA:
            id = ibgda_get_ctaid();
            break;
        case NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_SM:
            id = ibgda_get_smid();
            shared_among_ctas = true;
            break;
        case NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_WARP:
            id = ibgda_get_ctaid() * nvshmemi_block_size() / nvshmemi_warp_size() + warpid;
            break;
        case NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_DCT: {
            uint32_t dct_id;
            uint32_t group_id =
                ibgda_get_ctaid() * nvshmemi_block_size() / nvshmemi_warp_size() + warpid;

            dct_id = ibgda_get_dct_id(pe, 0);
            id = (group_id % state->num_dct_groups) * state->ndcts_per_pe *
                     nvshmemi_device_state_d.npes * state->num_devices_initialized +
                 dct_id * state->num_devices_initialized;
            shared_among_ctas = true;
            break;
        }
        default:
            assert(0);
            break;
    }
    dev_offset = ++state->globalmem.qp_group_switches[id % state->num_qp_groups];

    /* round down */
    id = id / state->num_devices_initialized;
    /* add dev index */
    id = (id * state->num_devices_initialized) + (dev_offset % state->num_devices_initialized);

    uint32_t idx;
    if (id < state->num_exclusive_dcis)
        idx = id;
    else {
        idx = state->num_exclusive_dcis + (id % state->num_shared_dcis);
        shared_among_ctas = true;
    }

    *out_shared_among_ctas = shared_among_ctas;
    return &state->globalmem.dcis[idx];
}

__device__ static inline nvshmemi_ibgda_device_qp_t *ibgda_get_rc(int pe,
                                                                  bool *out_shared_among_ctas) {
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();
    uint32_t id;
    uint32_t idx;
    uint32_t dev_offset;
    uint32_t warpid = nvshmemi_thread_id_in_block() / nvshmemi_warp_size();

    assert(pe != nvshmemi_device_state_d.mype);

    switch (state->rc_map_type) {
        case NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_CTA:
            id = ibgda_get_ctaid();
            break;
        case NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_SM:
            id = ibgda_get_smid();
            break;
        case NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_WARP:
            id = ibgda_get_ctaid() * nvshmemi_block_size() / nvshmemi_warp_size() + warpid;
            break;
        default:
            assert(0);
            break;
    }

    dev_offset = ++state->globalmem.qp_group_switches[id % state->num_qp_groups];

    /* round down */
    id = id / state->num_devices_initialized;
    id = (id * state->num_devices_initialized) + (dev_offset % state->num_devices_initialized);

    idx = (pe * state->num_rc_per_pe * state->num_devices_initialized) +
          (id % (state->num_rc_per_pe * state->num_devices_initialized));

    *out_shared_among_ctas = true;
    return &state->globalmem.rcs[idx];
}

__device__ static inline nvshmemi_ibgda_device_qp_t *ibgda_get_qp(int pe,
                                                                  bool *out_shared_among_ctas) {
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();

    if (ibgda_is_rc_enabled() && pe != nvshmemi_device_state_d.mype)
        return ibgda_get_rc(pe, out_shared_among_ctas);
    else
        return ibgda_get_dci(pe, out_shared_among_ctas);
}

__device__ static inline void ibgda_get_lkey(uint64_t addr, __be32 *lkey, size_t *chunk_size,
                                             bool *is_sysmem_scope, uint32_t dev_idx) {
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();
    uint64_t heap_start = (uint64_t)nvshmemi_device_state_d.heap_base;
    uint64_t heap_end = heap_start + nvshmemi_device_state_d.heap_size - 1;
    size_t max_len = 1ULL << 30;
    if (heap_start <= addr && addr <= heap_end) {
        // addr in the symmetric heap
        uint64_t idx = ((addr - heap_start) >> state->log2_cumem_granularity) *
                           state->num_devices_initialized +
                       dev_idx;
        nvshmemi_ibgda_device_key_t device_key;

        if (idx < NVSHMEMI_IBGDA_MAX_CONST_LKEYS)
            device_key = state->constmem.lkeys[idx];
        else
            device_key = state->globalmem.lkeys[idx - NVSHMEMI_IBGDA_MAX_CONST_LKEYS];

        assert(addr < device_key.next_addr);

        *lkey = device_key.key;
        *chunk_size = device_key.next_addr - addr;
        *chunk_size = *chunk_size < max_len ? *chunk_size : max_len;
        *is_sysmem_scope = (nvshmemi_device_state_d.symmetric_heap_kind == 1);
        return;
    } else {
        // local-only addr
        nvshmemi_ibgda_device_local_only_mhandle_t *mhandle =
            state->globalmem.local_only_mhandle_head;

        while (mhandle) {
            if (mhandle->start <= addr && addr <= mhandle->end) {
                *lkey = mhandle->lkeys[dev_idx];
                *chunk_size = mhandle->end - addr + 1;
                *chunk_size = *chunk_size < max_len ? *chunk_size : max_len;
                *is_sysmem_scope = mhandle->is_sysmem_scope;
                return;
            }
            mhandle = mhandle->next;
        }
    }

    // lkey is not found.
    assert(0);
}

__device__ static inline void ibgda_get_raddr_rkey(uint64_t addr, int dst_pe, int proxy_pe,
                                                   uint64_t *out_raddr, __be32 *out_rkey,
                                                   size_t *out_chunk_size, uint32_t dev_idx) {
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();
    uint64_t heap_start = (uint64_t)nvshmemi_device_state_d.heap_base;
    uint64_t roffset = addr - heap_start;
    int npes;
    // nvcc from CUDA12.0 - 12.2 seems to have a bug. It causes
    // nvshmemi_device_state_d.npes to become 0 in this function.
    // WAR: Force reload of nvshmemi_device_state_d.npes. We may reload from L1
    // most of the time, so the performance hit is minimal.
    asm volatile("ld.b32 %0, [%1];" : "=r"(npes) : "l"(&nvshmemi_device_state_d.npes));

    uint64_t idx =
        ((roffset >> state->log2_cumem_granularity) * npes * state->num_devices_initialized) +
        (proxy_pe * state->num_devices_initialized) + dev_idx;
    nvshmemi_ibgda_device_key_t device_key;
    uint64_t raddr;

    if (idx < NVSHMEMI_IBGDA_MAX_CONST_RKEYS)
        device_key = state->constmem.rkeys[idx];
    else
        device_key = state->globalmem.rkeys[idx - NVSHMEMI_IBGDA_MAX_CONST_RKEYS];

    assert(roffset < device_key.next_addr);

    raddr = (uint64_t)nvshmemi_device_state_d.peer_heap_base_remote[proxy_pe] + roffset;
    if (dst_pe != proxy_pe)
        raddr += (dst_pe % nvshmemi_device_state_d.node_npes - nvshmemi_device_state_d.node_mype) *
                 nvshmemi_device_state_d.heap_size;

    *out_raddr = raddr;
    *out_rkey = device_key.key;
    *out_chunk_size = device_key.next_addr - roffset;
}

__device__ static inline uint64_t ibgda_reserve_wqe_slots(nvshmemi_ibgda_device_qp_t *qp,
                                                          unsigned long long int num_wqes,
                                                          bool is_qp_shared_among_ctas) {
    nvshmemi_ibgda_device_qp_management_t *mvars = &qp->mvars;
    uint64_t wqe_idx;

// OK to keep this conditional since we only support one build per major verion.
#if CUDART_VERSION >= 12000
    if (is_qp_shared_among_ctas)
        wqe_idx = atomicAdd((unsigned long long int *)&mvars->tx_wq.resv_head, num_wqes);
    else
        wqe_idx = atomicAdd_block((unsigned long long int *)&mvars->tx_wq.resv_head, num_wqes);
#else
    // WAR NVBUG 3749055. The fix is in nvcc of CUDA 12.0 and later.
    if (is_qp_shared_among_ctas)
        asm volatile("atom.relaxed.gpu.global.add.u64 %0, [%1], %2;"
                     : "=l"(wqe_idx)
                     : "l"(&mvars->tx_wq.resv_head), "l"(num_wqes));
    else
        asm volatile("atom.relaxed.cta.global.add.u64 %0, [%1], %2;"
                     : "=l"(wqe_idx)
                     : "l"(&mvars->tx_wq.resv_head), "l"(num_wqes));
#endif
    // If last slot is available, all prior slots are also available.
    ibgda_wait_for_slot_availability(qp, wqe_idx + num_wqes);
    return wqe_idx;
}

__device__ static inline uint64_t ibgda_reserve_ibuf_slots(nvshmemi_ibgda_device_qp_t *qp,
                                                           unsigned long long int num_slots) {
    nvshmemi_ibgda_device_qp_management_t *mvars = &qp->mvars;
    uint32_t nslots = qp->ibuf.nslots;
    uint64_t base_idx = atomicAdd((unsigned long long int *)&mvars->ibuf.head, num_slots);
    uint64_t idx = base_idx + num_slots;

    // Wait until the slots become available.
    while (idx - ibgda_atomic_read(&mvars->ibuf.tail) > nslots)
        ;

    // Prevent the reordering of the above wait loop.
    IBGDA_MFENCE();

    return base_idx;
}

__device__ static inline void ibgda_release_ibuf(nvshmemi_ibgda_device_qp_t *qp,
                                                 unsigned long long int base_idx,
                                                 unsigned long long int num_slots) {
    nvshmemi_ibgda_device_qp_management_t *mvars = &qp->mvars;
    unsigned long long int new_idx = base_idx + num_slots;
    IBGDA_MFENCE();
    // Wait here.
    while (atomicCAS((unsigned long long int *)&mvars->ibuf.tail, (unsigned long long int)base_idx,
                     new_idx) != base_idx)
        ;
    IBGDA_MFENCE();
}

__device__ static inline uint64_t ibgda_get_ibuf_addr(nvshmemi_ibgda_device_qp_t *qp,
                                                      uint64_t idx) {
    idx = idx & (qp->ibuf.nslots - 1);

    // buf[0] is reserved for non-fetch operations
    return (uint64_t)qp->ibuf.buf + NVSHMEMI_IBGDA_IBUF_SLOT_SIZE * (idx + 1);
}

__device__ static inline bool ibgda_can_coalesce_warp(unsigned int amask,
                                                      nvshmemi_ibgda_device_qp_t *qp) {
    int pred_same_qp;

    if (amask != IBGDA_FULL_WARP) return false;

    __match_all_sync(amask, qp->qpn, &pred_same_qp);
    if (!pred_same_qp) return false;

    return true;
}

__device__ static inline bool ibgda_can_coalesce_warp_pe(unsigned int amask, int pe) {
    int pred_same_pe;

    if (amask != IBGDA_FULL_WARP) return false;

    __match_all_sync(amask, pe, &pred_same_pe);
    if (!pred_same_pe) return false;

    return true;
}

__device__ static inline uint64_t ibgda_cst(nvshmemi_ibgda_device_qp_t *dci,
                                            bool is_dci_shared_among_ctas) {
    assert(likely(dci->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI));

    nvshmemi_ibgda_device_dct_t *dct = ibgda_get_dct(nvshmemi_device_state_d.mype, dci->dev_idx);

    uint64_t laddr = (uint64_t)dci->ibuf.buf;
    __be32 lkey = dci->ibuf.lkey;

    const int num_wqes = 1;

    uint64_t base_wqe_idx = ibgda_reserve_wqe_slots(dci, num_wqes, is_dci_shared_among_ctas);

    void *wqe_ptrs[1];
    wqe_ptrs[0] = ibgda_get_wqe_ptr(dci, base_wqe_idx);

    // DUMP OP causes the NIC to read laddr, which is always on GPU memory.
    // For CST, it is cheaper than RDMA READ.
    ibgda_write_dump_wqe(dci, laddr, lkey, sizeof(char), base_wqe_idx, IBGDA_MLX5_FM_NO_FENCE,
                         wqe_ptrs);

    // Don't update get_head here because this is internal cst
    if (is_dci_shared_among_ctas)
        ibgda_submit_requests<true>(dci, base_wqe_idx, num_wqes);
    else
        ibgda_submit_requests<false>(dci, base_wqe_idx, num_wqes);

    return ibgda_quiet(dci);
}

__device__ static inline uint64_t ibgda_quiet_with_cst(nvshmemi_ibgda_device_qp_t *qp,
                                                       bool is_qp_shared_among_ctas) {
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();
    nvshmemi_ibgda_device_qp_management_t *mvars = &qp->mvars;

    uint64_t get_head;
    uint64_t ticket;
    uint64_t get_tail;

    if (state->may_skip_cst) {
        ticket = ibgda_quiet(qp);
    } else {
        // We want to read get_head before calling ibgda_quiet. Thus, ticket =
        // ibgda_quiet(qp) cannot be combined.
        get_head = ibgda_atomic_read(&mvars->tx_wq.get_head);
        ticket = ibgda_quiet(qp);
        get_tail = ibgda_atomic_read(&mvars->tx_wq.get_tail);

        // TODO: Change to WAIT + DUMP
        // In that case, we don't have to do quiet first
        if (get_tail < get_head) {
            if (qp->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI) {
                ticket = ibgda_cst(qp, is_qp_shared_among_ctas);
                ibgda_update_get_tail(qp, ticket);
            } else {
                // We don't have RC loopback to self.
                // So, we grab a DCI for CST.
                bool is_dci_shared_among_ctas;
                nvshmemi_ibgda_device_qp_t *dci =
                    ibgda_get_dci(nvshmemi_device_state_d.mype, &is_dci_shared_among_ctas);
                uint64_t cst_ticket = ibgda_cst(dci, is_dci_shared_among_ctas);
                ibgda_update_get_tail(dci, cst_ticket);
                ibgda_update_get_tail(qp, ticket);
            }
        }
    }

    return ticket;
}

template <nvshmemi_op_t channel_op, bool nbi, bool support_half_av_seg>
__device__ static inline void ibgda_rma_thread(uint64_t rptr, uint64_t lptr, size_t remaining_size,
                                               int dst_pe, int proxy_pe) {
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();
    unsigned int amask = __activemask();
    bool can_coalesce_warp = ibgda_can_coalesce_warp_pe(amask, proxy_pe);
    int my_tid;
    int tg_size;

    const bool need_cst = (channel_op == NVSHMEMI_OP_GET) && !state->may_skip_cst;
    const bool need_immediate_cst = !nbi && need_cst;

    int is_qp_shared_among_ctas;
    nvshmemi_ibgda_device_qp_t *qp;
    nvshmemi_ibgda_device_dct_t *dct;

    if (can_coalesce_warp) {
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();
        if (my_tid == 0) {
            qp = ibgda_get_qp(proxy_pe, (bool *)&is_qp_shared_among_ctas);
        }
        qp = (nvshmemi_ibgda_device_qp_t *)__shfl_sync(IBGDA_FULL_WARP, (uintptr_t)qp, 0);
        is_qp_shared_among_ctas = __shfl_sync(IBGDA_FULL_WARP, is_qp_shared_among_ctas, 0);
    } else {
        qp = ibgda_get_qp(proxy_pe, (bool *)&is_qp_shared_among_ctas);
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_THREAD>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_THREAD>();
    }
    dct = ibgda_get_dct(proxy_pe, qp->dev_idx);

    const bool need_additional_wqe =
        need_immediate_cst ||
        ((qp->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI) && !support_half_av_seg);
    int num_wqes_per_cmd =
        (qp->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI) ? (support_half_av_seg ? 1 : 2) : 1;

    bool did_quiet = false;

    if (unlikely(remaining_size == 0)) return;

    while (remaining_size > 0) {
        amask = __activemask();

        bool is_data_buf_in_sysmem;

        __be32 lkey;
        size_t lchunk_size;
        ibgda_get_lkey(lptr, &lkey, &lchunk_size, &is_data_buf_in_sysmem, qp->dev_idx);

        __be32 rkey;
        uint64_t raddr;
        size_t rchunk_size;
        ibgda_get_raddr_rkey(rptr, dst_pe, proxy_pe, &raddr, &rkey, &rchunk_size, qp->dev_idx);

        size_t transfer_size = ibgda_cal_transfer_size(remaining_size, lchunk_size, rchunk_size);

        can_coalesce_warp = ibgda_can_coalesce_warp(amask, qp);
        if (can_coalesce_warp) {
            my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
            tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();
        } else {
            my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_THREAD>();
            tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_THREAD>();
        }

        int num_wqes = num_wqes_per_cmd * tg_size + (need_additional_wqe ? 1 : 0);

        uint64_t base_wqe_idx;

        if (my_tid == 0) {
            base_wqe_idx = ibgda_reserve_wqe_slots(qp, num_wqes, is_qp_shared_among_ctas);
        }

        if (can_coalesce_warp) {
            base_wqe_idx = __shfl_sync(amask, base_wqe_idx, 0);
        }

        uint64_t my_wqe_idx = base_wqe_idx + (my_tid * num_wqes_per_cmd);

        void *wqe_ptrs[2];
        wqe_ptrs[0] = ibgda_get_wqe_ptr(qp, my_wqe_idx);
        wqe_ptrs[1] = ibgda_get_wqe_ptr(qp, my_wqe_idx + 1);

        // Generate CQE only if we create the last WQE in the group.
        uint8_t fm_ce_se =
            (!need_additional_wqe && (my_tid == tg_size - 1)) ? MLX5_WQE_CTRL_CQ_UPDATE : 0;

        switch (channel_op) {
            case NVSHMEMI_OP_PUT:
                ibgda_write_rdma_write_wqe<support_half_av_seg>(qp, dct, lptr, lkey, raddr, rkey,
                                                                transfer_size, my_wqe_idx, fm_ce_se,
                                                                wqe_ptrs);
                break;
            case NVSHMEMI_OP_GET:
                ibgda_write_rdma_read_wqe<support_half_av_seg>(qp, dct, lptr, lkey, raddr, rkey,
                                                               transfer_size, my_wqe_idx, fm_ce_se,
                                                               wqe_ptrs);
                break;
            default:
#ifdef NVSHMEM_IBGDA_DEBUG
                printf("Unsupported channel_op.\n");
#endif
                assert(0);
        }

        if (can_coalesce_warp) {
            nvshmemi_warp_sync();
        }

        if (my_tid == tg_size - 1) {
            if (need_immediate_cst) {
                // Enqueue CST op in the QP.  This command has NIC Fence, which
                // waits for all prior READ/ATOMIC to finish before issuing this
                // DUMP.
                my_wqe_idx += num_wqes_per_cmd;
                wqe_ptrs[0] = ibgda_get_wqe_ptr(qp, my_wqe_idx);
                ibgda_write_dump_wqe(qp, (uint64_t)qp->ibuf.buf, qp->ibuf.lkey, sizeof(char),
                                     my_wqe_idx, IBGDA_MLX5_FM_FENCE, wqe_ptrs);
            } else {
                if (need_additional_wqe) {
                    my_wqe_idx += num_wqes_per_cmd;
                    wqe_ptrs[0] = ibgda_get_wqe_ptr(qp, my_wqe_idx);
                    ibgda_write_nop_wqe(qp, my_wqe_idx, wqe_ptrs);
                }

                if (need_cst) {
                    // For nbi, we will do CST in QUIET.
                    // GET index must be visible before the new cons index.
                    ibgda_update_get_head(qp, base_wqe_idx + num_wqes);
                }
            }

            // Require membar.sys to push data buffer to the point of consistency.
            if (channel_op == NVSHMEMI_OP_PUT && is_data_buf_in_sysmem) __threadfence_system();

            if (is_qp_shared_among_ctas)
                ibgda_submit_requests<true>(qp, base_wqe_idx, num_wqes);
            else
                ibgda_submit_requests<false>(qp, base_wqe_idx, num_wqes);
        }

        remaining_size -= transfer_size;

        rptr += transfer_size;
        lptr += transfer_size;

        if (can_coalesce_warp) {
            if (!nbi) {
                bool do_coalesce_quiet = __all_sync(amask, remaining_size == 0);
                if (do_coalesce_quiet && my_tid == tg_size - 1) {
                    // CST, if required, has already been enqueued. We simply need to
                    // do ibgda_quiet here.
                    ibgda_quiet(qp);
                }
                did_quiet |= do_coalesce_quiet;
            }
            nvshmemi_warp_sync();
        }
    }

    if (!nbi && !did_quiet) {
        // CST, if required, has already been enqueued. We simply need to
        // do ibgda_quiet here.
        ibgda_quiet(qp);
    }
}

#if __cplusplus >= 201103L
static_assert(NVSHMEMI_IBGDA_MIN_QP_DEPTH >= 64,
              "static_assert(NVSHMEMI_IBGDA_MIN_QP_DEPTH >= 64) failed");
#endif
template <threadgroup_t SCOPE, nvshmemi_op_t channel_op, bool nbi, bool support_half_av_seg>
__device__ static inline void ibgda_rma(uint64_t req_rptr, uint64_t req_lptr, size_t bytes,
                                        int dst_pe, int proxy_pe) {
    assert(SCOPE == NVSHMEMI_THREADGROUP_WARP || SCOPE == NVSHMEMI_THREADGROUP_BLOCK);

    // Use only warp 0
    int my_tid = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();

    const bool need_cst = (channel_op == NVSHMEMI_OP_GET) && !state->may_skip_cst;
    const bool need_immediate_cst = !nbi && need_cst;
    bool need_additional_wqe;

    int is_qp_shared_among_ctas = 0;
    nvshmemi_ibgda_device_qp_t *qp;
    nvshmemi_ibgda_device_dct_t *dct;

    int num_wqes;
    int num_wqes_per_cmd;

    uint64_t base_wqe_idx;
    uint64_t my_wqe_idx;

    void *wqe_ptrs[2];

    size_t remaining_size = bytes;

    size_t transfer_size;
    size_t my_transfer_size = 0;

    uint64_t rptr = req_rptr;
    uint64_t lptr = req_lptr;

    __be32 lkey;
    __be32 my_lkey = 0;
    uint64_t my_laddr;
    size_t lchunk_size;

    __be32 rkey;
    __be32 my_rkey = 0;
    uint64_t raddr;
    uint64_t my_raddr;
    size_t rchunk_size;

    int chunk_idx = 0;

    bool is_data_buf_in_sysmem;

    uint8_t fm_ce_se;

    if (unlikely(remaining_size == 0)) goto out;

    // Not warp 0, wait at the exit.
    if (my_tid >= tg_size) {
        goto out;
    }
    my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();

    if (my_tid == 0) {
        qp = ibgda_get_qp(proxy_pe, (bool *)&is_qp_shared_among_ctas);
    }
    qp = (nvshmemi_ibgda_device_qp_t *)__shfl_sync(IBGDA_FULL_WARP, (uintptr_t)qp, 0);
    is_qp_shared_among_ctas = __shfl_sync(IBGDA_FULL_WARP, is_qp_shared_among_ctas, 0);
    dct = ibgda_get_dct(proxy_pe, qp->dev_idx);

    need_additional_wqe =
        need_immediate_cst ||
        ((qp->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI) && !support_half_av_seg);

    num_wqes_per_cmd =
        (qp->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI) ? (support_half_av_seg ? 1 : 2) : 1;

    // Calculate how many chunks we need to send.
    while (remaining_size > 0) {
        ibgda_get_lkey(lptr, &lkey, &lchunk_size, &is_data_buf_in_sysmem, qp->dev_idx);
        ibgda_get_raddr_rkey(rptr, dst_pe, proxy_pe, &raddr, &rkey, &rchunk_size, qp->dev_idx);
        transfer_size = ibgda_cal_transfer_size(remaining_size, lchunk_size, rchunk_size);
        if (my_tid == chunk_idx) {
            my_lkey = lkey;
            my_laddr = lptr;
            my_rkey = rkey;
            my_raddr = raddr;
            my_transfer_size = transfer_size;
        }

        remaining_size -= transfer_size;
        rptr += transfer_size;
        lptr += transfer_size;

        ++chunk_idx;
    }

    // Too many chunks. Use ibgda_rma_thread to handle it instead.
    if (unlikely(chunk_idx > tg_size)) {
        if (my_tid == 0) {
            ibgda_rma_thread<channel_op, nbi, support_half_av_seg>(req_rptr, req_lptr, bytes,
                                                                   dst_pe, proxy_pe);
        }
        goto out;
    }

    num_wqes = num_wqes_per_cmd * chunk_idx + (need_additional_wqe ? 1 : 0);

    if (my_tid == 0) {
        base_wqe_idx = ibgda_reserve_wqe_slots(qp, num_wqes, is_qp_shared_among_ctas);
    }

    base_wqe_idx = __shfl_sync(IBGDA_FULL_WARP, base_wqe_idx, 0);
    my_wqe_idx = base_wqe_idx + (my_tid * num_wqes_per_cmd);

    // Generate CQE only if we create the last WQE in the group.
    fm_ce_se = (!need_additional_wqe && (my_tid == chunk_idx - 1)) ? MLX5_WQE_CTRL_CQ_UPDATE : 0;

    if (my_tid < chunk_idx) {
        wqe_ptrs[0] = ibgda_get_wqe_ptr(qp, my_wqe_idx);
        wqe_ptrs[1] = ibgda_get_wqe_ptr(qp, my_wqe_idx + 1);

        switch (channel_op) {
            case NVSHMEMI_OP_PUT:
                ibgda_write_rdma_write_wqe<support_half_av_seg>(qp, dct, my_laddr, my_lkey,
                                                                my_raddr, my_rkey, my_transfer_size,
                                                                my_wqe_idx, fm_ce_se, wqe_ptrs);
                break;
            case NVSHMEMI_OP_GET:
                ibgda_write_rdma_read_wqe<support_half_av_seg>(qp, dct, my_laddr, my_lkey, my_raddr,
                                                               my_rkey, my_transfer_size,
                                                               my_wqe_idx, fm_ce_se, wqe_ptrs);
                break;
            default:
#ifdef NVSHMEM_IBGDA_DEBUG
                printf("Unsupported channel_op.\n");
#endif
                assert(0);
        }
    }

    nvshmemi_warp_sync();

    if (my_tid == chunk_idx - 1) {
        if (need_immediate_cst) {
            my_wqe_idx += num_wqes_per_cmd;
            wqe_ptrs[0] = ibgda_get_wqe_ptr(qp, my_wqe_idx);
            // Enqueue CST op in the QP.  This command has NIC Fence, which
            // waits for all prior READ/ATOMIC to finish before issuing this
            // DUMP.
            ibgda_write_dump_wqe(qp, (uint64_t)qp->ibuf.buf, qp->ibuf.lkey, sizeof(char),
                                 my_wqe_idx, IBGDA_MLX5_FM_FENCE, wqe_ptrs);
        } else {
            if (need_additional_wqe) {
                my_wqe_idx += num_wqes_per_cmd;
                wqe_ptrs[0] = ibgda_get_wqe_ptr(qp, my_wqe_idx);
                ibgda_write_nop_wqe(qp, my_wqe_idx, wqe_ptrs);
            }

            if (need_cst) {
                // For nbi, we will do CST in QUIET.
                // GET index must be visible before the new cons index.
                // ibgda_submit_requests has fence, which guarantees the ordering.
                ibgda_update_get_head(qp, base_wqe_idx + num_wqes);
            }
        }

        // Require membar.sys to push data buffer to the point of consistency.
        if (channel_op == NVSHMEMI_OP_PUT && is_data_buf_in_sysmem) __threadfence_system();

        if (is_qp_shared_among_ctas)
            ibgda_submit_requests<true>(qp, base_wqe_idx, num_wqes);
        else
            ibgda_submit_requests<false>(qp, base_wqe_idx, num_wqes);

        if (!nbi) {
            // CST, if required, has already been enqueued. We simply need to
            // do ibgda_quiet here.
            ibgda_quiet(qp);
        }
    }

out:
    nvshmemi_threadgroup_sync<SCOPE>();
}

/**
 * RMA P base
 */
#if __cplusplus >= 201103L
static_assert(NVSHMEMI_IBGDA_MIN_QP_DEPTH >= 64,
              "static_assert(NVSHMEMI_IBGDA_MIN_QP_DEPTH >= 64) failed");
#endif
template <typename T, bool is_full_warp, bool can_combine_data, bool support_half_av_seg>
__device__ static inline void nvshmemi_ibgda_rma_p_impl(void *rptr, const T value, int dst_pe) {
    static_assert((can_combine_data && is_full_warp) || (!can_combine_data),
                  "can_combine_data check 1 failed.\n");
    static_assert((can_combine_data && support_half_av_seg) || (!can_combine_data),
                  "can_combine_data check 2 failed.\n");

    int my_tid;
    int tg_size;
    int proxy_pe = ibgda_get_proxy_pe(dst_pe);
    int is_qp_shared_among_ctas;
    nvshmemi_ibgda_device_qp_t *qp;
    nvshmemi_ibgda_device_dct_t *dct;

    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();

    if (is_full_warp) {
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();
        if (my_tid == 0) {
            qp = ibgda_get_qp(proxy_pe, (bool *)&is_qp_shared_among_ctas);
        }
        qp = (nvshmemi_ibgda_device_qp_t *)__shfl_sync(IBGDA_FULL_WARP, (uintptr_t)qp, 0);
        is_qp_shared_among_ctas = __shfl_sync(IBGDA_FULL_WARP, is_qp_shared_among_ctas, 0);
    } else {
        qp = ibgda_get_qp(proxy_pe, (bool *)&is_qp_shared_among_ctas);
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_THREAD>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_THREAD>();
    }
    dct = ibgda_get_dct(proxy_pe, qp->dev_idx);

    __be32 rkey;
    uint64_t raddr;
    size_t rchunk_size;
    ibgda_get_raddr_rkey((uint64_t)rptr, dst_pe, proxy_pe, &raddr, &rkey, &rchunk_size,
                         qp->dev_idx);

    // With proper alignment (requirement of NVSHMEM), one element cannot span multiple chunks.
    assert(rchunk_size >= sizeof(T));

    int num_wqes_per_cmd;
    int num_wqes;

    bool need_additional_wqe = false;

    if (can_combine_data) {
        if (qp->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_RC) {
            num_wqes_per_cmd =
                ibgda_get_num_wqes_in_inl_combine_warp<T, NVSHMEMI_IBGDA_DEVICE_QP_TYPE_RC>();
        } else if (sizeof(T) == 8) {
            num_wqes_per_cmd =
                2 * ibgda_get_num_wqes_in_inl_combine_warp<uint32_t,
                                                           NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI>();
        } else {
            num_wqes_per_cmd =
                ibgda_get_num_wqes_in_inl_combine_warp<T, NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI>();
        }
        num_wqes = num_wqes_per_cmd;
    } else {
        num_wqes_per_cmd =
            (qp->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI) ? (support_half_av_seg ? 1 : 2) : 1;
        num_wqes = num_wqes_per_cmd * tg_size;
    }

    if (!can_combine_data && num_wqes_per_cmd > 1) {
        ++num_wqes;
        need_additional_wqe = true;
    }

    uint64_t base_wqe_idx;

    if (my_tid == 0) {
        base_wqe_idx = ibgda_reserve_wqe_slots(qp, num_wqes, is_qp_shared_among_ctas);
    }

    if (is_full_warp) {
        base_wqe_idx = __shfl_sync(IBGDA_FULL_WARP, base_wqe_idx, 0);
    }

    // Generate CQE only if we create the last WQE in the group.
    uint8_t fm_ce_se =
        (!need_additional_wqe && (my_tid == tg_size - 1)) ? MLX5_WQE_CTRL_CQ_UPDATE : 0;

    uint64_t my_wqe_idx =
        can_combine_data ? base_wqe_idx : base_wqe_idx + (my_tid * num_wqes_per_cmd);

    void *wqe_ptrs[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        wqe_ptrs[i] = ibgda_get_wqe_ptr(qp, my_wqe_idx + i);
    }

    if (can_combine_data && sizeof(T) == 8 && qp->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI)
        ibgda_write_rdma_write_inl_wqe_combine_warp_for_dci_8B<T>(qp, dct, value, raddr, rkey,
                                                                  my_wqe_idx, my_tid, wqe_ptrs);
    else if (can_combine_data)
        ibgda_write_rdma_write_inl_wqe_combine_warp<T>(qp, dct, value, raddr, rkey, my_wqe_idx,
                                                       my_tid, wqe_ptrs);
    else
        ibgda_write_rdma_write_inl_wqe<support_half_av_seg>(qp, dct, &value, raddr, rkey, sizeof(T),
                                                            my_wqe_idx, fm_ce_se, wqe_ptrs);

    if (is_full_warp) nvshmemi_warp_sync();

    if (my_tid == tg_size - 1) {
        if (need_additional_wqe) {
            my_wqe_idx += num_wqes_per_cmd;
            wqe_ptrs[0] = ibgda_get_wqe_ptr(qp, my_wqe_idx);
            ibgda_write_nop_wqe(qp, my_wqe_idx, wqe_ptrs);
        }

        if (is_qp_shared_among_ctas)
            ibgda_submit_requests<true>(qp, base_wqe_idx, num_wqes);
        else
            ibgda_submit_requests<false>(qp, base_wqe_idx, num_wqes);
    }

    if (is_full_warp) nvshmemi_warp_sync();
}

template <typename T>
__device__ inline void nvshmemi_ibgda_rma_p(void *rptr, const T value, int dst_pe) {
    unsigned int amask = __activemask();
    bool can_combine_data = false;
    int pred_pe = 0;
    int pred_contiguous = 0;
    int pred_rkey = 0;
    int my_tid;

    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();

    if (amask == IBGDA_FULL_WARP) {
        /* TODO: Adding multi-dev support could have caused a regression with coalescing. */
        nvshmemi_ibgda_device_state_t *state = ibgda_get_state();
        __be32 rkey;
        uint64_t raddr;
        size_t rchunk_size;
        int proxy_pe = ibgda_get_proxy_pe(dst_pe);
        ibgda_get_raddr_rkey((uint64_t)rptr, dst_pe, proxy_pe, &raddr, &rkey, &rchunk_size, 0);
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
        __match_all_sync(IBGDA_FULL_WARP, dst_pe, &pred_pe);
        __match_all_sync(IBGDA_FULL_WARP, (uintptr_t)(rptr) - (my_tid * sizeof(T)),
                         &pred_contiguous);
        __match_all_sync(IBGDA_FULL_WARP, rkey, &pred_rkey);
        can_combine_data = (pred_pe && pred_contiguous && pred_rkey && state->support_half_av_seg);

        if (can_combine_data)
            nvshmemi_ibgda_rma_p_impl<T, true, true, true>(rptr, value, dst_pe);
        else if (state->support_half_av_seg)
            nvshmemi_ibgda_rma_p_impl<T, true, false, true>(rptr, value, dst_pe);
        else
            nvshmemi_ibgda_rma_p_impl<T, true, false, false>(rptr, value, dst_pe);
    } else if (state->support_half_av_seg)
        nvshmemi_ibgda_rma_p_impl<T, false, false, true>(rptr, value, dst_pe);
    else
        nvshmemi_ibgda_rma_p_impl<T, false, false, false>(rptr, value, dst_pe);
}

/**
 * RMA G base
 */
template <typename T, bool support_half_av_seg>
__device__ inline T nvshmemi_ibgda_rma_g_impl(void *rptr, int dst_pe, int proxy_pe) {
    unsigned int amask = __activemask();
    int my_tid;
    int tg_size;

    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();
    const bool need_cst = !state->may_skip_cst;

    uint64_t base_wqe_idx;
    uint64_t base_ibuf_idx;

    T ret;

    int is_qp_shared_among_ctas;
    nvshmemi_ibgda_device_dct_t *dct;
    nvshmemi_ibgda_device_qp_t *qp;

    __be32 rkey;
    uint64_t raddr;
    size_t rchunk_size;

    bool can_coalesce_warp = ibgda_can_coalesce_warp_pe(amask, proxy_pe);
    bool can_combine_data = false;
    int pred_contiguous = 0;
    int pred_rkey = 0;

    if (can_coalesce_warp) {
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();
        if (my_tid == 0) {
            qp = ibgda_get_qp(proxy_pe, (bool *)&is_qp_shared_among_ctas);
        }
        qp = (nvshmemi_ibgda_device_qp_t *)__shfl_sync(IBGDA_FULL_WARP, (uintptr_t)qp, 0);
        is_qp_shared_among_ctas = __shfl_sync(IBGDA_FULL_WARP, is_qp_shared_among_ctas, 0);
        ibgda_get_raddr_rkey((uint64_t)rptr, dst_pe, proxy_pe, &raddr, &rkey, &rchunk_size,
                             qp->dev_idx);

        __match_all_sync(IBGDA_FULL_WARP, (uintptr_t)(rptr) - (my_tid * sizeof(T)),
                         &pred_contiguous);
        __match_all_sync(IBGDA_FULL_WARP, rkey, &pred_rkey);
        can_combine_data = (pred_contiguous && pred_rkey);
    } else {
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_THREAD>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_THREAD>();
        qp = ibgda_get_qp(proxy_pe, (bool *)&is_qp_shared_among_ctas);
        ibgda_get_raddr_rkey((uint64_t)rptr, dst_pe, proxy_pe, &raddr, &rkey, &rchunk_size,
                             qp->dev_idx);
    }
    dct = ibgda_get_dct(proxy_pe, qp->dev_idx);

    const bool need_additional_wqe =
        need_cst || ((qp->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI) && !support_half_av_seg);

    int num_wqes_per_cmd =
        (qp->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI) ? (support_half_av_seg ? 1 : 2) : 1;

    int num_wqes = (can_combine_data ? num_wqes_per_cmd : num_wqes_per_cmd * tg_size) +
                   (need_additional_wqe ? 1 : 0);

    int num_ibuf_slots = can_coalesce_warp ? 1 : tg_size;

    if (my_tid == 0) {
        base_ibuf_idx = ibgda_reserve_ibuf_slots(qp, num_ibuf_slots);
        base_wqe_idx = ibgda_reserve_wqe_slots(qp, num_wqes, is_qp_shared_among_ctas);
    }

    if (can_coalesce_warp) {
        base_wqe_idx = __shfl_sync(amask, base_wqe_idx, 0);
        base_ibuf_idx = __shfl_sync(amask, base_ibuf_idx, 0);
    }

    uint64_t my_wqe_idx =
        can_combine_data ? base_wqe_idx : base_wqe_idx + (my_tid * num_wqes_per_cmd);
    uint64_t my_ibuf_idx = can_coalesce_warp ? base_ibuf_idx : base_ibuf_idx + my_tid;

    void *wqe_ptrs[2];
    wqe_ptrs[0] = ibgda_get_wqe_ptr(qp, my_wqe_idx);
    wqe_ptrs[1] = ibgda_get_wqe_ptr(qp, my_wqe_idx + 1);

    uint64_t laddr =
        ibgda_get_ibuf_addr(qp, my_ibuf_idx) + (can_coalesce_warp ? my_tid * sizeof(T) : 0);
    __be32 lkey = qp->ibuf.lkey;

    // Generate CQE only if we create the last WQE in the group.
    uint8_t fm_ce_se = (!need_additional_wqe && ((can_combine_data && (my_tid == 0)) ||
                                                 (!can_combine_data && (my_tid == tg_size - 1))))
                           ? MLX5_WQE_CTRL_CQ_UPDATE
                           : 0;

    if (!can_combine_data) {
        ibgda_write_rdma_read_wqe<support_half_av_seg>(qp, dct, laddr, lkey, raddr, rkey, sizeof(T),
                                                       my_wqe_idx, fm_ce_se, wqe_ptrs);

    } else if (my_tid == 0) {
        ibgda_write_rdma_read_wqe<support_half_av_seg>(
            qp, dct, laddr, lkey, raddr, rkey, sizeof(T) * tg_size, my_wqe_idx, fm_ce_se, wqe_ptrs);
    }

    if (can_coalesce_warp) nvshmemi_warp_sync();

    if (need_additional_wqe && (my_tid == (tg_size - 1))) {
        my_wqe_idx += num_wqes_per_cmd;
        wqe_ptrs[0] = ibgda_get_wqe_ptr(qp, my_wqe_idx);
        fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;

        if (need_cst)
            // Enqueue CST op in the QP.  This command has NIC Fence, which
            // waits for all prior READ/ATOMIC to finish before issuing this
            // DUMP.
            ibgda_write_dump_wqe(qp, (uint64_t)qp->ibuf.buf, qp->ibuf.lkey, sizeof(char),
                                 my_wqe_idx, IBGDA_MLX5_FM_FENCE, wqe_ptrs);
        else
            ibgda_write_nop_wqe(qp, my_wqe_idx, wqe_ptrs);
    }
    if (fm_ce_se > 0) {
        if (is_qp_shared_among_ctas)
            ibgda_submit_requests<true>(qp, base_wqe_idx, num_wqes);
        else
            ibgda_submit_requests<false>(qp, base_wqe_idx, num_wqes);

        ibgda_quiet(qp);
    }

    if (can_coalesce_warp) nvshmemi_warp_sync();

    ret = READ_ONCE(*(T *)laddr);

    if (can_coalesce_warp) nvshmemi_warp_sync();

    if (my_tid == tg_size - 1) ibgda_release_ibuf(qp, base_ibuf_idx, num_ibuf_slots);

    if (can_coalesce_warp) nvshmemi_warp_sync();

    return ret;
}

template <typename T>
__device__ inline T nvshmemi_ibgda_rma_g(void *rptr, int dst_pe) {
    T ret;
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();

    int proxy_pe = ibgda_get_proxy_pe(dst_pe);

    if (state->support_half_av_seg)
        ret = nvshmemi_ibgda_rma_g_impl<T, true>(rptr, dst_pe, proxy_pe);
    else
        ret = nvshmemi_ibgda_rma_g_impl<T, false>(rptr, dst_pe, proxy_pe);
    return ret;
}

/**
 * RMA NBI base
 */
template <threadgroup_t SCOPE, nvshmemi_op_t channel_op>
__device__ inline void nvshmemi_ibgda_rma_nbi(void *rptr, void *lptr, size_t bytes, int dst_pe) {
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();
    int proxy_pe = ibgda_get_proxy_pe(dst_pe);
    if (SCOPE == NVSHMEMI_THREADGROUP_THREAD) {
        if (state->support_half_av_seg) {
            ibgda_rma_thread<channel_op, true, true>((uint64_t)rptr, (uint64_t)lptr, bytes, dst_pe,
                                                     proxy_pe);
        } else {
            ibgda_rma_thread<channel_op, true, false>((uint64_t)rptr, (uint64_t)lptr, bytes, dst_pe,
                                                      proxy_pe);
        }
    } else {
        if (state->support_half_av_seg) {
            ibgda_rma<SCOPE, channel_op, true, true>((uint64_t)rptr, (uint64_t)lptr, bytes, dst_pe,
                                                     proxy_pe);
        } else {
            ibgda_rma<SCOPE, channel_op, true, false>((uint64_t)rptr, (uint64_t)lptr, bytes, dst_pe,
                                                      proxy_pe);
        }
    }
}

/**
 * RMA (blocking) base
 */
template <threadgroup_t SCOPE, nvshmemi_op_t channel_op>
__device__ inline void nvshmemi_ibgda_rma(void *rptr, void *lptr, size_t bytes, int dst_pe) {
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();
    int proxy_pe = ibgda_get_proxy_pe(dst_pe);
    if (SCOPE == NVSHMEMI_THREADGROUP_THREAD) {
        if (state->support_half_av_seg) {
            ibgda_rma_thread<channel_op, false, true>((uint64_t)rptr, (uint64_t)lptr, bytes, dst_pe,
                                                      proxy_pe);
        } else {
            ibgda_rma_thread<channel_op, false, false>((uint64_t)rptr, (uint64_t)lptr, bytes,
                                                       dst_pe, proxy_pe);
        }
    } else {
        if (state->support_half_av_seg) {
            ibgda_rma<SCOPE, channel_op, false, true>((uint64_t)rptr, (uint64_t)lptr, bytes, dst_pe,
                                                      proxy_pe);
        } else {
            ibgda_rma<SCOPE, channel_op, false, false>((uint64_t)rptr, (uint64_t)lptr, bytes,
                                                       dst_pe, proxy_pe);
        }
    }
}

/**
 * AMO non-fetch base
 */
template <typename T, bool support_half_av_seg>
__device__ inline void nvshmemi_ibgda_amo_nonfetch_impl(void *rptr, const T value, int pe,
                                                        nvshmemi_amo_t op) {
    unsigned int amask = __activemask();
    int my_tid;
    int tg_size;

    int is_qp_shared_among_ctas;
    nvshmemi_ibgda_device_qp_t *qp;
    nvshmemi_ibgda_device_dct_t *dct;

    __be32 rkey;
    uint64_t raddr;
    size_t rchunk_size;

    bool can_coalesce_warp = ibgda_can_coalesce_warp_pe(amask, pe);

    if (can_coalesce_warp) {
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();
        if (my_tid == 0) {
            qp = ibgda_get_qp(pe, (bool *)&is_qp_shared_among_ctas);
        }
        qp = (nvshmemi_ibgda_device_qp_t *)__shfl_sync(IBGDA_FULL_WARP, (uintptr_t)qp, 0);
        is_qp_shared_among_ctas = __shfl_sync(IBGDA_FULL_WARP, is_qp_shared_among_ctas, 0);
    } else {
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_THREAD>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_THREAD>();
        qp = ibgda_get_qp(pe, (bool *)&is_qp_shared_among_ctas);
    }
    dct = ibgda_get_dct(pe, qp->dev_idx);
    ibgda_get_raddr_rkey((uint64_t)rptr, pe, pe, &raddr, &rkey, &rchunk_size, qp->dev_idx);

    int num_wqes_per_cmd = ibgda_get_num_wqes_in_atomic<T>(op, qp->qp_type);

    const bool need_additional_wqe = (num_wqes_per_cmd > 1);

    int num_wqes = num_wqes_per_cmd * tg_size + (need_additional_wqe ? 1 : 0);

    uint64_t base_wqe_idx;

    if (my_tid == 0) base_wqe_idx = ibgda_reserve_wqe_slots(qp, num_wqes, is_qp_shared_among_ctas);

    if (can_coalesce_warp) base_wqe_idx = __shfl_sync(amask, base_wqe_idx, 0);

    uint64_t my_wqe_idx = base_wqe_idx + (my_tid * num_wqes_per_cmd);

    void *wqe_ptrs[2];
    wqe_ptrs[0] = ibgda_get_wqe_ptr(qp, my_wqe_idx);
    wqe_ptrs[1] = ibgda_get_wqe_ptr(qp, my_wqe_idx + 1);

    uint8_t fm_ce_se =
        (!need_additional_wqe && (my_tid == tg_size - 1)) ? MLX5_WQE_CTRL_CQ_UPDATE : 0;

    ibgda_write_atomic_wqe<support_half_av_seg>(qp, dct, &value, NULL, (uint64_t)qp->ibuf.buf,
                                                qp->ibuf.lkey, raddr, rkey, sizeof(T), my_wqe_idx,
                                                op, fm_ce_se, wqe_ptrs);

    if (can_coalesce_warp) nvshmemi_warp_sync();

    if (my_tid == tg_size - 1) {
        if (need_additional_wqe) {
            my_wqe_idx += num_wqes_per_cmd;
            wqe_ptrs[0] = ibgda_get_wqe_ptr(qp, my_wqe_idx);
            ibgda_write_nop_wqe(qp, my_wqe_idx, wqe_ptrs);
        }

        if (is_qp_shared_among_ctas)
            ibgda_submit_requests<true>(qp, base_wqe_idx, num_wqes);
        else
            ibgda_submit_requests<false>(qp, base_wqe_idx, num_wqes);
    }

    if (can_coalesce_warp) nvshmemi_warp_sync();
}

template <typename T>
__device__ inline void nvshmemi_ibgda_amo_nonfetch(void *rptr, const T value, int pe,
                                                   nvshmemi_amo_t op) {
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();

    if (state->support_half_av_seg)
        nvshmemi_ibgda_amo_nonfetch_impl<T, true>(rptr, value, pe, op);
    else
        nvshmemi_ibgda_amo_nonfetch_impl<T, false>(rptr, value, pe, op);
}

/**
 * AMO fetch base
 */
template <typename T, bool support_half_av_seg>
__device__ inline T nvshmemi_ibgda_amo_fetch_impl(void *rptr, const T value, const T compare,
                                                  int pe, nvshmemi_amo_t op) {
    unsigned int amask = __activemask();
    int my_tid;
    int tg_size;

    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();
    const bool need_cst = !state->may_skip_cst;

    T ret;

    int is_qp_shared_among_ctas;
    nvshmemi_ibgda_device_qp_t *qp;
    nvshmemi_ibgda_device_dct_t *dct;

    __be32 rkey;
    uint64_t raddr;
    size_t rchunk_size;

    bool can_coalesce_warp = ibgda_can_coalesce_warp_pe(amask, pe);

    if (can_coalesce_warp) {
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();
        if (my_tid == 0) {
            qp = ibgda_get_qp(pe, (bool *)&is_qp_shared_among_ctas);
        }
        qp = (nvshmemi_ibgda_device_qp_t *)__shfl_sync(IBGDA_FULL_WARP, (uintptr_t)qp, 0);
        is_qp_shared_among_ctas = __shfl_sync(IBGDA_FULL_WARP, is_qp_shared_among_ctas, 0);
    } else {
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_THREAD>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_THREAD>();
        qp = ibgda_get_qp(pe, (bool *)&is_qp_shared_among_ctas);
    }
    dct = ibgda_get_dct(pe, qp->dev_idx);
    ibgda_get_raddr_rkey((uint64_t)rptr, pe, pe, &raddr, &rkey, &rchunk_size, qp->dev_idx);

    int num_wqes_per_cmd = ibgda_get_num_wqes_in_atomic<T>(op, qp->qp_type);

    const bool need_additional_wqe = (num_wqes_per_cmd > 1) || need_cst;

    int num_wqes = num_wqes_per_cmd * tg_size + (need_additional_wqe ? 1 : 0);

    uint64_t base_wqe_idx;
    uint64_t base_ibuf_idx;

    if (my_tid == 0) {
        base_ibuf_idx = ibgda_reserve_ibuf_slots(qp, tg_size);
        base_wqe_idx = ibgda_reserve_wqe_slots(qp, num_wqes, is_qp_shared_among_ctas);
    }

    if (can_coalesce_warp) {
        base_wqe_idx = __shfl_sync(amask, base_wqe_idx, 0);
        base_ibuf_idx = __shfl_sync(amask, base_ibuf_idx, 0);
    }

    uint64_t my_wqe_idx = base_wqe_idx + (my_tid * num_wqes_per_cmd);
    uint64_t my_ibuf_idx = base_ibuf_idx + my_tid;

    uint64_t laddr = ibgda_get_ibuf_addr(qp, my_ibuf_idx);
    __be32 lkey = qp->ibuf.lkey;

    void *wqe_ptrs[2];
    wqe_ptrs[0] = ibgda_get_wqe_ptr(qp, my_wqe_idx);
    wqe_ptrs[1] = ibgda_get_wqe_ptr(qp, my_wqe_idx + 1);

    uint8_t fm_ce_se =
        (!need_additional_wqe && (my_tid == tg_size - 1)) ? MLX5_WQE_CTRL_CQ_UPDATE : 0;

    ibgda_write_atomic_wqe<support_half_av_seg>(qp, dct, &value, &compare, laddr, lkey, raddr, rkey,
                                                sizeof(T), my_wqe_idx, op, fm_ce_se, wqe_ptrs);

    if (can_coalesce_warp) nvshmemi_warp_sync();

    if (my_tid == tg_size - 1) {
        if (need_additional_wqe) {
            my_wqe_idx += num_wqes_per_cmd;
            wqe_ptrs[0] = ibgda_get_wqe_ptr(qp, my_wqe_idx);

            if (need_cst)
                // Enqueue CST op in the QP.  This command has NIC Fence, which
                // waits for all prior READ/ATOMIC to finish before issuing this
                // DUMP.
                ibgda_write_dump_wqe(qp, (uint64_t)qp->ibuf.buf, qp->ibuf.lkey, sizeof(char),
                                     my_wqe_idx, IBGDA_MLX5_FM_FENCE, wqe_ptrs);
            else
                ibgda_write_nop_wqe(qp, my_wqe_idx, wqe_ptrs);
        }

        if (is_qp_shared_among_ctas)
            ibgda_submit_requests<true>(qp, base_wqe_idx, num_wqes);
        else
            ibgda_submit_requests<false>(qp, base_wqe_idx, num_wqes);

        ibgda_quiet(qp);
    }

    if (can_coalesce_warp) nvshmemi_warp_sync();

    ret = READ_ONCE(*(T *)laddr);
    if (sizeof(T) == 4) ret = BSWAP32((uint32_t)ret);

    if (can_coalesce_warp) nvshmemi_warp_sync();

    if (my_tid == tg_size - 1) ibgda_release_ibuf(qp, base_ibuf_idx, tg_size);

    if (can_coalesce_warp) nvshmemi_warp_sync();

    return ret;
}

template <typename T>
__device__ inline T nvshmemi_ibgda_amo_fetch(void *rptr, const T value, const T compare, int pe,
                                             nvshmemi_amo_t op) {
    T ret;
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();

    if (state->support_half_av_seg)
        ret = nvshmemi_ibgda_amo_fetch_impl<T, true>(rptr, value, compare, pe, op);
    else
        ret = nvshmemi_ibgda_amo_fetch_impl<T, false>(rptr, value, compare, pe, op);
    return ret;
}

#if __cplusplus >= 201103L
static_assert(NVSHMEMI_IBGDA_MIN_QP_DEPTH >= 128,
              "static_assert(NVSHMEMI_IBGDA_MIN_QP_DEPTH >= 128) failed");
#endif
template <bool is_nbi, bool support_half_av_seg>
__device__ static inline void nvshmemi_ibgda_put_signal_thread_impl(void *rptr, void *lptr,
                                                                    size_t bytes, void *sig_rptr,
                                                                    uint64_t signal,
                                                                    nvshmemi_amo_t sig_op, int pe) {
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();
    nvshmemi_ibgda_device_qp_t *qp;
    nvshmemi_ibgda_device_dct_t *dct;
    size_t lchunk_size;
    size_t rchunk_size;
    size_t sig_rchunk_size;
    uint64_t sig_raddr;
    uint64_t raddr;

    unsigned int amask = __activemask();
    int my_tid;
    int tg_size;
    __be32 lkey;
    __be32 rkey;
    __be32 sig_rkey;

    bool can_coalesce_warp = ibgda_can_coalesce_warp_pe(amask, pe);
    int is_qp_shared_among_ctas;
    bool is_data_buf_in_sysmem;

    if (can_coalesce_warp) {
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();
        if (my_tid == 0) {
            qp = ibgda_get_qp(pe, (bool *)&is_qp_shared_among_ctas);
        }
        qp = (nvshmemi_ibgda_device_qp_t *)__shfl_sync(IBGDA_FULL_WARP, (uintptr_t)qp, 0);
        is_qp_shared_among_ctas = __shfl_sync(IBGDA_FULL_WARP, is_qp_shared_among_ctas, 0);
    } else {
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_THREAD>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_THREAD>();
        qp = ibgda_get_qp(pe, (bool *)&is_qp_shared_among_ctas);
    }
    dct = ibgda_get_dct(pe, qp->dev_idx);
    ibgda_get_lkey((uint64_t)lptr, &lkey, &lchunk_size, &is_data_buf_in_sysmem, qp->dev_idx);
    ibgda_get_raddr_rkey((uint64_t)rptr, pe, pe, &raddr, &rkey, &rchunk_size, qp->dev_idx);
    ibgda_get_raddr_rkey((uint64_t)sig_rptr, pe, pe, &sig_raddr, &sig_rkey, &sig_rchunk_size,
                         qp->dev_idx);

    const int num_atomic_wqes_per_cmd = ibgda_get_num_wqes_in_atomic<uint64_t>(sig_op, qp->qp_type);
    const bool need_additional_wqe = (num_atomic_wqes_per_cmd > 1);
    int num_wqes;
    uint8_t fm_ce_se;

    size_t transfer_size = ibgda_cal_transfer_size(bytes, lchunk_size, rchunk_size);
    uint64_t base_wqe_idx;
    uint64_t my_wqe_idx;

    if (transfer_size == bytes) {
        amask = __activemask();
        can_coalesce_warp = ibgda_can_coalesce_warp(amask, qp);
        if (can_coalesce_warp) {
            my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
            tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();
        } else {
            my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_THREAD>();
            tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_THREAD>();
        }

        int num_rdma_write_wqes_per_cmd =
            (qp->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI) ? (support_half_av_seg ? 1 : 2) : 1;

        int num_wqes_per_cmd = num_rdma_write_wqes_per_cmd + num_atomic_wqes_per_cmd;
        num_wqes = num_wqes_per_cmd * tg_size + (need_additional_wqe ? 1 : 0);

        if (my_tid == 0) {
            base_wqe_idx = ibgda_reserve_wqe_slots(qp, num_wqes, is_qp_shared_among_ctas);
        }

        if (can_coalesce_warp) {
            base_wqe_idx = __shfl_sync(amask, base_wqe_idx, 0);
        }

        my_wqe_idx = base_wqe_idx + (my_tid * num_wqes_per_cmd);

        void *wqe_ptrs[4];
        wqe_ptrs[0] = ibgda_get_wqe_ptr(qp, my_wqe_idx);
        wqe_ptrs[1] = ibgda_get_wqe_ptr(qp, my_wqe_idx + 1);
        wqe_ptrs[2] = ibgda_get_wqe_ptr(qp, my_wqe_idx + 2);
        wqe_ptrs[3] = ibgda_get_wqe_ptr(qp, my_wqe_idx + 3);

        ibgda_write_rdma_write_wqe<support_half_av_seg>(qp, dct, (uint64_t)lptr, lkey, raddr, rkey,
                                                        bytes, my_wqe_idx, 0, wqe_ptrs);

        fm_ce_se = (!need_additional_wqe && (my_tid == tg_size - 1)) ? MLX5_WQE_CTRL_CQ_UPDATE : 0;

        ibgda_write_atomic_wqe<support_half_av_seg>(
            qp, dct, &signal, NULL, (uint64_t)qp->ibuf.buf, qp->ibuf.lkey, sig_raddr, sig_rkey,
            sizeof(signal), my_wqe_idx + num_rdma_write_wqes_per_cmd, sig_op, fm_ce_se,
            &wqe_ptrs[num_rdma_write_wqes_per_cmd]);

        if (can_coalesce_warp) {
            nvshmemi_warp_sync();
        }

        if (my_tid == tg_size - 1) {
            if (need_additional_wqe) {
                my_wqe_idx += num_wqes_per_cmd;
                wqe_ptrs[0] = ibgda_get_wqe_ptr(qp, my_wqe_idx);
                ibgda_write_nop_wqe(qp, my_wqe_idx, wqe_ptrs);
            }

            // Require membar.sys to push data buffer to the point of consistency.
            if (is_data_buf_in_sysmem) __threadfence_system();
            if (is_qp_shared_among_ctas)
                ibgda_submit_requests<true>(qp, base_wqe_idx, num_wqes);
            else
                ibgda_submit_requests<false>(qp, base_wqe_idx, num_wqes);

            if (!is_nbi) {
                ibgda_quiet(qp);
            }
        }

        if (can_coalesce_warp) {
            nvshmemi_warp_sync();
        }
    } else {
        ibgda_rma_thread<NVSHMEMI_OP_PUT, true, support_half_av_seg>(
            (uintptr_t)rptr, (uintptr_t)lptr, bytes, pe, pe);

        num_wqes = num_atomic_wqes_per_cmd + (need_additional_wqe ? 1 : 0);

        base_wqe_idx = ibgda_reserve_wqe_slots(qp, num_wqes, is_qp_shared_among_ctas);
        my_wqe_idx = base_wqe_idx;

        void *wqe_ptrs[2];
        wqe_ptrs[0] = ibgda_get_wqe_ptr(qp, my_wqe_idx);
        wqe_ptrs[1] = ibgda_get_wqe_ptr(qp, my_wqe_idx + 1);

        fm_ce_se = (!need_additional_wqe) ? MLX5_WQE_CTRL_CQ_UPDATE : 0;

        ibgda_write_atomic_wqe<support_half_av_seg>(
            qp, dct, &signal, NULL, (uint64_t)qp->ibuf.buf, qp->ibuf.lkey, sig_raddr, sig_rkey,
            sizeof(signal), my_wqe_idx, sig_op, fm_ce_se, wqe_ptrs);

        if (need_additional_wqe) {
            my_wqe_idx += num_atomic_wqes_per_cmd;
            wqe_ptrs[0] = ibgda_get_wqe_ptr(qp, my_wqe_idx);
            ibgda_write_nop_wqe(qp, my_wqe_idx, wqe_ptrs);
        }

        if (is_qp_shared_among_ctas)
            ibgda_submit_requests<true>(qp, base_wqe_idx, num_wqes);
        else
            ibgda_submit_requests<false>(qp, base_wqe_idx, num_wqes);

        if (!is_nbi) {
            ibgda_quiet(qp);
        }
    }
}

/**
 * PUT SIGNAL base
 */
#if __cplusplus >= 201103L
static_assert(NVSHMEMI_IBGDA_MIN_QP_DEPTH >= 64,
              "static_assert(NVSHMEMI_IBGDA_MIN_QP_DEPTH >= 64) failed");
#endif
template <threadgroup_t SCOPE, bool is_nbi, bool support_half_av_seg>
__device__ static inline void nvshmemi_ibgda_put_signal_impl(void *req_rptr, void *req_lptr,
                                                             size_t bytes, void *sig_rptr,
                                                             uint64_t signal, nvshmemi_amo_t sig_op,
                                                             int pe) {
    assert(SCOPE == NVSHMEMI_THREADGROUP_WARP || SCOPE == NVSHMEMI_THREADGROUP_BLOCK);

    // Use only wrap 0
    int my_tid = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();

    int is_qp_shared_among_ctas;
    nvshmemi_ibgda_device_qp_t *qp;
    nvshmemi_ibgda_device_dct_t *dct;

    int num_rdma_write_wqes_per_cmd;
    int num_atomic_wqes_per_cmd;
    bool need_additional_wqe;

    int num_wqes;

    uint64_t base_wqe_idx;
    uint64_t my_wqe_idx;

    void *wqe_ptrs[2];

    size_t remaining_size = bytes;

    size_t transfer_size;
    size_t my_transfer_size = 0;

    uint64_t rptr = (uint64_t)req_rptr;
    uint64_t lptr = (uint64_t)req_lptr;

    __be32 lkey;
    __be32 my_lkey = 0;
    uint64_t my_laddr;
    size_t lchunk_size;

    __be32 rkey;
    __be32 my_rkey = 0;
    uint64_t raddr;
    uint64_t my_raddr;
    size_t rchunk_size;

    int chunk_idx = 0;

    bool is_data_buf_in_sysmem;

    // Not warp 0, wait at the exit.
    if (my_tid >= tg_size) {
        goto out;
    }

    my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();

    if (my_tid == 0) {
        qp = ibgda_get_qp(pe, (bool *)&is_qp_shared_among_ctas);
    }
    qp = (nvshmemi_ibgda_device_qp_t *)__shfl_sync(IBGDA_FULL_WARP, (uintptr_t)qp, 0);
    is_qp_shared_among_ctas = __shfl_sync(IBGDA_FULL_WARP, is_qp_shared_among_ctas, 0);
    dct = ibgda_get_dct(pe, qp->dev_idx);

    num_rdma_write_wqes_per_cmd =
        (qp->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI) ? (support_half_av_seg ? 1 : 2) : 1;

    num_atomic_wqes_per_cmd = ibgda_get_num_wqes_in_atomic<uint64_t>(sig_op, qp->qp_type);

    need_additional_wqe = (num_atomic_wqes_per_cmd > 1);

    // Calculate how many chunks we need to send.
    while (remaining_size > 0) {
        ibgda_get_lkey(lptr, &lkey, &lchunk_size, &is_data_buf_in_sysmem, qp->dev_idx);
        ibgda_get_raddr_rkey(rptr, pe, pe, &raddr, &rkey, &rchunk_size, qp->dev_idx);
        transfer_size = ibgda_cal_transfer_size(remaining_size, lchunk_size, rchunk_size);
        if (my_tid == chunk_idx) {
            my_lkey = lkey;
            my_laddr = lptr;
            my_rkey = rkey;
            my_raddr = raddr;
            my_transfer_size = transfer_size;
        }

        remaining_size -= transfer_size;
        rptr += transfer_size;
        lptr += transfer_size;

        ++chunk_idx;
    }

    // Too many chunks. Use nvshmemi_ibgda_put_signal_thread_impl to handle it instead.
    // Note that we need one thread to handle amo.
    if (unlikely(chunk_idx > tg_size - 1)) {
        if (my_tid == 0) {
            nvshmemi_ibgda_put_signal_thread_impl<is_nbi, support_half_av_seg>(
                req_rptr, req_lptr, bytes, sig_rptr, signal, sig_op, pe);
        }
        goto out;
    }

    num_wqes = num_rdma_write_wqes_per_cmd * chunk_idx + num_atomic_wqes_per_cmd +
               (need_additional_wqe ? 1 : 0);

    if (my_tid == 0) {
        base_wqe_idx = ibgda_reserve_wqe_slots(qp, num_wqes, is_qp_shared_among_ctas);
    }

    base_wqe_idx = __shfl_sync(IBGDA_FULL_WARP, base_wqe_idx, 0);
    my_wqe_idx = base_wqe_idx + (my_tid * num_rdma_write_wqes_per_cmd);

    wqe_ptrs[0] = ibgda_get_wqe_ptr(qp, my_wqe_idx);
    wqe_ptrs[1] = ibgda_get_wqe_ptr(qp, my_wqe_idx + 1);

    if (my_tid < chunk_idx) {
        ibgda_write_rdma_write_wqe<support_half_av_seg>(qp, dct, my_laddr, my_lkey, my_raddr,
                                                        my_rkey, my_transfer_size, my_wqe_idx, 0,
                                                        wqe_ptrs);
    } else if (my_tid == chunk_idx) {
        __be32 sig_rkey;
        uint64_t sig_raddr;
        size_t sig_rchunk_size;
        ibgda_get_raddr_rkey((uint64_t)sig_rptr, pe, pe, &sig_raddr, &sig_rkey, &sig_rchunk_size,
                             qp->dev_idx);

        uint8_t fm_ce_se = (!need_additional_wqe) ? MLX5_WQE_CTRL_CQ_UPDATE : 0;

        ibgda_write_atomic_wqe<support_half_av_seg>(
            qp, dct, &signal, NULL, (uint64_t)qp->ibuf.buf, qp->ibuf.lkey, sig_raddr, sig_rkey,
            sizeof(signal), my_wqe_idx, sig_op, fm_ce_se, wqe_ptrs);

        if (need_additional_wqe) {
            my_wqe_idx += num_atomic_wqes_per_cmd;
            wqe_ptrs[0] = ibgda_get_wqe_ptr(qp, my_wqe_idx);
            ibgda_write_nop_wqe(qp, my_wqe_idx, wqe_ptrs);
        }
    }

    nvshmemi_warp_sync();

    if (my_tid == chunk_idx) {
        // Require membar.sys to push data buffer to the point of consistency.
        if (is_data_buf_in_sysmem) __threadfence_system();

        if (is_qp_shared_among_ctas)
            ibgda_submit_requests<true>(qp, base_wqe_idx, num_wqes);
        else
            ibgda_submit_requests<false>(qp, base_wqe_idx, num_wqes);

        if (!is_nbi) {
            ibgda_quiet(qp);
        }
    }

out:
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <threadgroup_t SCOPE>
__device__ inline void nvshmemi_ibgda_put_signal(void *rptr, void *lptr, size_t bytes,
                                                 void *sig_rptr, uint64_t signal,
                                                 nvshmemi_amo_t sig_op, int pe, bool is_nbi) {
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();
    if (SCOPE == NVSHMEMI_THREADGROUP_THREAD) {
        if (is_nbi && state->support_half_av_seg)
            nvshmemi_ibgda_put_signal_thread_impl<true, true>(rptr, lptr, bytes, sig_rptr, signal,
                                                              sig_op, pe);
        else if (is_nbi && !state->support_half_av_seg)
            nvshmemi_ibgda_put_signal_thread_impl<true, false>(rptr, lptr, bytes, sig_rptr, signal,
                                                               sig_op, pe);
        else if (!is_nbi && state->support_half_av_seg)
            nvshmemi_ibgda_put_signal_thread_impl<false, true>(rptr, lptr, bytes, sig_rptr, signal,
                                                               sig_op, pe);
        else
            nvshmemi_ibgda_put_signal_thread_impl<false, false>(rptr, lptr, bytes, sig_rptr, signal,
                                                                sig_op, pe);
    } else {
        if (is_nbi && state->support_half_av_seg)
            nvshmemi_ibgda_put_signal_impl<SCOPE, true, true>(rptr, lptr, bytes, sig_rptr, signal,
                                                              sig_op, pe);
        else if (is_nbi && !state->support_half_av_seg)
            nvshmemi_ibgda_put_signal_impl<SCOPE, true, false>(rptr, lptr, bytes, sig_rptr, signal,
                                                               sig_op, pe);
        else if (!is_nbi && state->support_half_av_seg)
            nvshmemi_ibgda_put_signal_impl<SCOPE, false, true>(rptr, lptr, bytes, sig_rptr, signal,
                                                               sig_op, pe);
        else
            nvshmemi_ibgda_put_signal_impl<SCOPE, false, false>(rptr, lptr, bytes, sig_rptr, signal,
                                                                sig_op, pe);
    }
}

template <threadgroup_t SCOPE>
__device__ inline void nvshmemi_ibgda_quiet() {
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();
    nvshmemi_ibgda_device_qp_t *qp;
    uint32_t ndcis = state->num_shared_dcis + state->num_exclusive_dcis;
    uint32_t nrcs =
        state->num_rc_per_pe * nvshmemi_device_state_d.npes * state->num_devices_initialized;
    uint32_t index_in_scope = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    uint32_t scope_size = nvshmemi_threadgroup_size<SCOPE>();

    scope_size =
        scope_size > IBGDA_MAX_THREADS_PER_QUIET ? IBGDA_MAX_THREADS_PER_QUIET : scope_size;

    if (index_in_scope < scope_size) {
        for (uint32_t i = index_in_scope; i < ndcis; i += scope_size) {
            qp = &state->globalmem.dcis[i];
            ibgda_quiet_with_cst(qp, true);
        }

        for (uint32_t i = index_in_scope; i < nrcs; i += scope_size) {
            if (i / (state->num_rc_per_pe * state->num_devices_initialized) ==
                nvshmemi_device_state_d.mype) {
                continue;
            }
            qp = &state->globalmem.rcs[i];
            ibgda_quiet_with_cst(qp, true);
        }
    }
}

template <threadgroup_t SCOPE>
__device__ inline void nvshmemi_ibgda_fence() {
    // Multiple QPs may target the same PE before fence.
    // We need to quiet those QPs.
    // TODO: Make it more efficient.
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();
    uint32_t ndcis = state->num_shared_dcis + state->num_exclusive_dcis;
    uint32_t index_in_scope = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    uint32_t scope_size = nvshmemi_threadgroup_size<SCOPE>();
    uint32_t nrcs = state->num_rc_per_pe * nvshmemi_device_state_d.npes;
    nvshmemi_ibgda_device_qp_t *qp;

    // As all WQEs always go to the same QP, FENCE is naturally guaranteed.
    if (unlikely(ndcis + nrcs <= 1)) return;

    scope_size =
        scope_size > IBGDA_MAX_THREADS_PER_QUIET ? IBGDA_MAX_THREADS_PER_QUIET : scope_size;

    // Fence does not guarantee the completion of prior operations.
    // It is ok for GET to finish without data arrival.
    // Use ibgda_quiet here instead of ibgda_quiet_with_cst since it is cheaper.
    if (index_in_scope < scope_size) {
        for (uint32_t i = index_in_scope; i < ndcis; i += scope_size) {
            qp = &state->globalmem.dcis[i];
            ibgda_quiet(qp);
        }

        for (uint32_t i = index_in_scope; i < nrcs; i += scope_size) {
            if (i / state->num_rc_per_pe == nvshmemi_device_state_d.mype) continue;
            qp = &state->globalmem.rcs[i];
            ibgda_quiet(qp);
        }
    }

    nvshmemi_threadgroup_sync<SCOPE>();
}

__device__ inline void nvshmemi_ibgda_enforce_consistency_at_target(bool use_membar) {
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();

    if (!state->may_skip_cst) {
        bool is_dci_shared_among_ctas;
        // We don't have RC loopback to self.
        // So, DCI is always used here.
        nvshmemi_ibgda_device_qp_t *dci;

        /* We must run the cst op on all devices */
        for (int i = 0; i < state->num_devices_initialized; i++) {
            dci = ibgda_get_dci(nvshmemi_device_state_d.mype, &is_dci_shared_among_ctas);
            ibgda_cst(dci, is_dci_shared_among_ctas);
        }
    }

    // TODO: This fence is from the design of Proxy.
    // Review if we still need it when we fully move to IBGDA -- especially for on-stream API.
    if (use_membar) {
        __threadfence_system();  // XXX: prevents store to issue_d reordered to before load from
                                 // cst_ack_d (breaks cst -> rma)
    }
}

#endif /* __CUDA_ARCH__ */

#endif /* _NVSHMEMI_IBGDA_DEVICE_H_ */
