/****
 * Copyright (c) 2014, NVIDIA Corporation.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright notice,
 *      this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *    * Neither the name of the NVIDIA Corporation nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * The U.S. Department of Energy funded the development of this software
 * under subcontract 7078610 with Lawrence Berkeley National Laboratory.
 *
 ****/

#ifndef PROXY_DEVICE_CUH
#define PROXY_DEVICE_CUH

#include <cuda_runtime.h>
#include "utils_device.h"
#include "non_abi/device/wait/nvshmemi_wait_until_apis.cuh"
/* this file does not directly use the definitions from device_host/nvshmem_proxy_channel.h */
/* But the way the requests are filled in directly represents those structures. */
#include "device_host/nvshmem_proxy_channel.h"  // IWYU pragma: keep

#ifdef __CUDA_ARCH__
static __forceinline__ __device__ void check_channel_availability(uint64_t tail_idx) {
    uint64_t complete;
    complete = *((volatile uint64_t *)nvshmemi_device_state_d.proxy_channels_complete_local_ptr);
    if ((complete + nvshmemi_device_state_d.proxy_channel_buf_size - 1) < tail_idx) {
        nvshmemi_wait_until_greater_than_equals_add<uint64_t>(
            nvshmemi_device_state_d.proxy_channels_complete,
            nvshmemi_device_state_d.proxy_channel_buf_size - 1, tail_idx,
            NVSHMEMI_CALL_SITE_PROXY_CHECK_CHANNEL_AVAILABILITY);
        atomicMax(
            (unsigned long long int *)nvshmemi_device_state_d.proxy_channels_complete_local_ptr,
            *nvshmemi_device_state_d.proxy_channels_complete);
        __threadfence_system();  // XXX: prevents store to buf_d reordered to before load from
                                 // complete_d (breaks rma)
    }
}

static __device__ inline void nvshmemi_proxy_quiet(bool use_membar) {
    uint64_t quiet_issue;
    quiet_issue = (*(volatile uint64_t *)nvshmemi_device_state_d.proxy_channels_issue);
    atomicMax((unsigned long long int *)nvshmemi_device_state_d.proxy_channels_quiet_issue,
              quiet_issue);

    nvshmemi_wait_until_greater_than_equals<uint64_t>(
        nvshmemi_device_state_d.proxy_channels_quiet_ack, quiet_issue,
        NVSHMEMI_CALL_SITE_PROXY_QUIET);
    if (use_membar) {
        __threadfence_system();  // XXX: prevents store to issue_d reordered to before load from
                                 // quiet_ack_d (breaks quiet -> rma)
    }
}

static __device__ inline void nvshmemi_proxy_global_exit(int status) {
    int rc;

    rc = atomicCAS(nvshmemi_device_state_d.global_exit_request_state,
                   PROXY_GLOBAL_EXIT_NOT_REQUESTED, PROXY_GLOBAL_EXIT_INIT);
    if (rc == PROXY_GLOBAL_EXIT_NOT_REQUESTED) {
        *nvshmemi_device_state_d.global_exit_code = status;
        __threadfence_system();
        rc = atomicCAS(nvshmemi_device_state_d.global_exit_request_state, PROXY_GLOBAL_EXIT_INIT,
                       PROXY_GLOBAL_EXIT_REQUESTED);
        assert(rc == PROXY_GLOBAL_EXIT_INIT);
    }
    /* Note, this will block indefinitely, but that is fine as nvshmemi_global_exit should never
     * return. */
    long long int now, later;
    asm volatile("mov.u64  %0, %globaltimer;" : "=l"(now));
    do {
        asm volatile("mov.u64  %0, %globaltimer;" : "=l"(later));
    } while (later >= now);
}

static __device__ inline void nvshmemi_proxy_enforce_consistency_at_target(bool use_membar) {
    uint64_t cst_issue;
    cst_issue = (*(volatile uint64_t *)nvshmemi_device_state_d.proxy_channels_issue);
    atomicMax((unsigned long long int *)nvshmemi_device_state_d.proxy_channels_cst_issue,
              cst_issue);

    nvshmemi_wait_until_greater_than_equals<uint64_t>(
        nvshmemi_device_state_d.proxy_channels_cst_ack, cst_issue,
        NVSHMEMI_CALL_SITE_PROXY_ENFORCE_CONSISTENCY_AT_TARGET);
    if (use_membar) {
        __threadfence_system();  // XXX: prevents store to issue_d reordered to before load from
                                 // cst_ack_d (breaks cst -> rma)
    }
}

static __device__ inline void copy_to_channel(void *ptr, uint32_t size) {
    channel_bounce_buffer_t bounce;
    uint64_t idx, tail_idx;
    volatile uint64_t *channel_ptr;
    char *src_ptr = (char *)ptr;
    uint32_t size_with_flags;
    uint32_t size_remaining;

    /* idx is an every increasing counter. Since it is 64 bit integer, practically
    it will not overflow */
    size_with_flags = (size * 8) / 7;
    if (size_with_flags % 8) {
        size_with_flags += (8 - (size_with_flags % 8));
    }
    idx = atomicAdd((unsigned long long int *)nvshmemi_device_state_d.proxy_channels_issue,
                    size_with_flags);
    tail_idx = idx + (size_with_flags - 1);

    // flow-control
    check_channel_availability(tail_idx);

    for (size_remaining = size; size_remaining > 7; idx += sizeof(uint64_t), size_remaining -= 7) {
        channel_ptr = (volatile uint64_t *)((uint64_t)nvshmemi_device_state_d.proxy_channels_buf +
                                            (idx & (CHANNEL_BUF_SIZE - 1)));
        memcpy(&bounce.bytes[1], src_ptr, 7);
        /* Note, the second to last bit being set denotes a continuation of the same request to the
         * other side. */
        bounce.bytes[0] =
            (char)(!((idx >> nvshmemi_device_state_d.proxy_channel_buf_logsize) & 1) | 0x10);
        *channel_ptr = bounce.whole_buffer;
        src_ptr += 7;
    }

    channel_ptr = (volatile uint64_t *)((uint64_t)nvshmemi_device_state_d.proxy_channels_buf +
                                        (idx & (CHANNEL_BUF_SIZE - 1)));
    memcpy(&bounce.bytes[1], src_ptr, size_remaining);
    bounce.bytes[0] = (char)!((idx >> nvshmemi_device_state_d.proxy_channel_buf_logsize) & 1);
    *channel_ptr = bounce.whole_buffer;
}

static __forceinline__ __device__ void transfer_dma(void *rptr, void *lptr, size_t bytes, int pe,
                                                    int channel_op) {
    uint64_t idx, tail_idx, *req;
    int size = PROXY_DMA_REQ_BYTES;
    int group_size = 1;
    void *buf_ptr = nvshmemi_device_state_d.proxy_channels_buf;
    void *base_ptr = nvshmemi_device_state_d.heap_base;
    const uint64_t mask_lowest_byte = 0xFFFFFFFFFFFFFF00u;
    const uint64_t mask_upper_7_bytes = 0x00000000000000FFu;

    __threadfence();

    /* idx is an every increasing counter. Since it is 64 bit integer, practically
    it will not overflow */
    idx = atomicAdd((unsigned long long int *)nvshmemi_device_state_d.proxy_channels_issue, size);
    tail_idx = idx + (size - 1);

    // flow-control
    check_channel_availability(tail_idx);

    req = (uint64_t *)((uint8_t *)buf_ptr + (idx & (CHANNEL_BUF_SIZE - 1)));
    uint64_t curr_flag = !((idx >> nvshmemi_device_state_d.proxy_channel_buf_logsize) & 1);
    /* curr_flag is either 0 or 1. Starting at idx = 0 to idx =
     * nvshmemi_device_state_d.proxy_channel_buf_size - 1, it will be 1, then for next
     * nvshmemi_device_state_d.proxy_channel_buf_size idx values it will be 0, and so
     * on.
     */
    uint64_t roffset = (uint64_t)((char *)rptr - (char *)base_ptr);
    uint64_t laddr = (uint64_t)lptr;
    uint64_t op = channel_op;
    uint16_t pe_u16 = pe;
    uint64_t size_u64 = bytes;

    /* base_request_t
     * 32 | 8 | 8 | 8 | 8
     * roffset_high | roffset_low | op | group_size | flag */
    *((volatile uint64_t *)req) =
        (uint64_t)((roffset << 24) | (op << 16) | (group_size << 8) | curr_flag);

    /* put_dma_request_0
     * 56 | 8
     * laddr_high | flag */
    idx += CHANNEL_ENTRY_BYTES;
    uint64_t laddr_high = laddr & mask_lowest_byte;
    req = (uint64_t *)((uint8_t *)buf_ptr + (idx & (CHANNEL_BUF_SIZE - 1)));
    curr_flag = !((idx >> nvshmemi_device_state_d.proxy_channel_buf_logsize) & 1);
    *((volatile uint64_t *)req) = laddr_high | curr_flag;

    /* put_dma_request_1
     * 32 | 16 | 8 | 8
     * size_high | size_low | laddr_low | flag */
    idx += CHANNEL_ENTRY_BYTES;
    uint64_t laddr_low = laddr & mask_upper_7_bytes;
    req = (uint64_t *)((uint8_t *)buf_ptr +
                       (idx & (nvshmemi_device_state_d.proxy_channel_buf_size - 1)));
    curr_flag = !((idx >> nvshmemi_device_state_d.proxy_channel_buf_logsize) & 1);
    *((volatile uint64_t *)req) = (uint64_t)(size_u64 << 16 | laddr_low << 8 | curr_flag);

    /* put_dma_request_2
     * 32 | 16 | 8 | 8
     * resv2 | pe | resv1 | flag */
    idx += CHANNEL_ENTRY_BYTES;
    req = (uint64_t *)((uint8_t *)buf_ptr + (idx & (CHANNEL_BUF_SIZE - 1)));
    curr_flag = !((idx >> nvshmemi_device_state_d.proxy_channel_buf_logsize) & 1);
    *((volatile uint64_t *)req) = (uint64_t)((pe_u16 << 16) | curr_flag);
}

/*XXX : Only no const version is used*/
template <nvshmemi_op_t channel_op>
static __device__ inline void nvshmemi_proxy_rma(void *rptr, void *lptr, size_t bytes, int pe) {
    assert(0);
    /*XXX:to be used for 1) inline 2) DMA with ack 3) DMA by staging in another buffer*/
}

static __device__ __forceinline__ void nvshmemi_proxy_rma_nbi(void *rptr, void *lptr, size_t bytes,
                                                              int pe, nvshmemi_op_t op) {
    if (!bytes) return;
    transfer_dma(rptr, lptr, bytes, pe, op);
}

template <typename T>
static __device__ inline T nvshmemi_proxy_rma_g(void *source, int pe) {
// check for CC >= 7.0 because the code depends on __match_all_sync
#if __CUDA_ARCH__ >= 700
    constexpr unsigned int full_warp = 0xffffffffu;
    const unsigned int amask = __activemask();
    unsigned int laneId;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneId));
    int pred_pe = 0;
    int pred_contigous = 0;

    // check if warp is coalesced, if all lanes put to the same PE and if buffer is contigous
    if (full_warp == amask && full_warp == __match_all_sync(full_warp, pe, &pred_pe) &&
        full_warp == __match_all_sync(full_warp,
                                      reinterpret_cast<uintptr_t>(reinterpret_cast<char *>(source) -
                                                                  (laneId * sizeof(T))),
                                      &pred_contigous)) {
        assert(pred_pe && pred_contigous);

        g_elem_t *elem = nullptr;
        int coalescing_buf_byte_offset = -1;
        if (0 == laneId) {
            uint64_t counter = atomicAdd(
                (unsigned long long int *)nvshmemi_device_state_d.proxy_channel_g_buf_head_ptr, 1);
            uint64_t idx = counter * sizeof(g_elem_t);
            uint64_t idx_in_buf = idx & (nvshmemi_device_state_d.proxy_channel_g_buf_size - 1);
            elem = (g_elem_t *)(nvshmemi_device_state_d.proxy_channel_g_buf + idx_in_buf);
            coalescing_buf_byte_offset =
                (idx_in_buf / sizeof(g_elem_t)) * NVSHMEMI_WARP_SIZE * sizeof(uint64_t);
            // Using a g_elem_t as to for a coalescing buffer at the same index
            uint64_t flag = (idx >> nvshmemi_device_state_d.proxy_channel_g_buf_log_size) * 2;

            /* wait until element can be used */
            nvshmemi_wait_until_greater_than_equals<uint64_t>((volatile uint64_t *)&(elem->flag),
                                                              flag, NVSHMEMI_CALL_SITE_G_WAIT_FLAG);
            static_assert(sizeof(T) <= sizeof(uint64_t), "sizeof(T) exceeds sizeof(uint64_t)");
            nvshmemi_proxy_rma_nbi(source,
                                   (void *)(nvshmemi_device_state_d.proxy_channel_g_coalescing_buf +
                                            coalescing_buf_byte_offset),
                                   NVSHMEMI_WARP_SIZE * sizeof(T), pe, NVSHMEMI_OP_G);
            nvshmemi_proxy_quiet(false);
        }
        coalescing_buf_byte_offset = __shfl_sync(amask, coalescing_buf_byte_offset, 0);
        T *__restrict__ coalescing_buf = reinterpret_cast<T *>(
            nvshmemi_device_state_d.proxy_channel_g_coalescing_buf + coalescing_buf_byte_offset);
        T return_val = coalescing_buf[laneId];
        __syncwarp(amask);
        if (0 == laneId) {
            __threadfence();
            /* release the element for the next thread */
            elem->flag += 2;
        }

        return return_val;
    } else
#endif  //__CUDA_ARCH__ >= 700
    {
        uint64_t counter = atomicAdd(
            (unsigned long long int *)nvshmemi_device_state_d.proxy_channel_g_buf_head_ptr, 1);
        uint64_t idx = counter * sizeof(g_elem_t);
        uint64_t idx_in_buf = idx & (nvshmemi_device_state_d.proxy_channel_g_buf_size - 1);
        g_elem_t *elem = (g_elem_t *)(nvshmemi_device_state_d.proxy_channel_g_buf + idx_in_buf);
        uint64_t flag = (idx >> nvshmemi_device_state_d.proxy_channel_g_buf_log_size) * 2;

        /* wait until element can be used */
        nvshmemi_wait_until_greater_than_equals<uint64_t>((volatile uint64_t *)&(elem->flag), flag,
                                                          NVSHMEMI_CALL_SITE_G_WAIT_FLAG);

        nvshmemi_proxy_rma_nbi(source, (void *)elem, sizeof(T), pe, NVSHMEMI_OP_G);
        nvshmemi_proxy_quiet(false);

        __threadfence();
        T return_val = *(T *)(&(elem->data));
        __threadfence();
        /* release the element for the next thread */
        elem->flag += 2;

        return return_val;
    }
}

static __forceinline__ __device__ void convert_val_to_uint64(uint64_t *dest, void *value,
                                                             size_t size) {
    uint8_t byte_buffer_1;
    uint16_t byte_buffer_2;
    uint32_t byte_buffer_4;
    uint64_t byte_buffer_8;

    switch (size) {
        case 1:
            memcpy(&byte_buffer_1, value, size);
            *dest = byte_buffer_1;
            break;
        case 2:
            memcpy(&byte_buffer_2, value, size);
            *dest = byte_buffer_2;
            break;
        case 4:
            memcpy(&byte_buffer_4, value, size);
            *dest = byte_buffer_4;
            break;
        case 8:
            memcpy(&byte_buffer_8, value, size);
            *dest = byte_buffer_8;
            break;
        default:
            printf("Invalid size value provided to convert_val_to_uint64 in proxy_device.cu.\n");
    }
}

template <typename T>
static __forceinline__ __device__ void transfer_inline(void *rptr, T value, int pe,
                                                       nvshmemi_op_t optype) {
    uint64_t idx, tail_idx, *req;
    int size = PROXY_INLINE_REQ_BYTES;
    int group_size = 1;
    void *buf_ptr = nvshmemi_device_state_d.proxy_channels_buf;
    void *base_ptr = nvshmemi_device_state_d.heap_base;

    idx = atomicAdd((unsigned long long int *)nvshmemi_device_state_d.proxy_channels_issue, size);
    tail_idx = idx + (size - 1);

    // flow-control
    check_channel_availability(tail_idx);

    req = (uint64_t *)((uint8_t *)buf_ptr + (idx & (CHANNEL_BUF_SIZE - 1)));
    uint64_t curr_flag = !((idx >> nvshmemi_device_state_d.proxy_channel_buf_logsize) & 1);
    uint64_t roffset = (uint64_t)((char *)rptr - (char *)base_ptr);
    uint64_t op = optype;
    uint16_t pe_u16 = pe;
    uint64_t size_u64 = sizeof(T);

    uint64_t lvalue_buffer = 0;
    const uint64_t mask_4_bytes = 0x00000000ffffffff;

    convert_val_to_uint64(&lvalue_buffer, &value, sizeof(T));

    /* base_request_t
     * 32 | 8 | 8 | 8 | 8
     * roffset_high | roffset_low | op | group_size | flag */
    *((volatile uint64_t *)req) =
        (uint64_t)((roffset << 24) | (op << 16) | (group_size << 8) | curr_flag);

    /* put_inline_request_0
     * 32 | 16 | 8 | 8
     * lvalue (low) | pe | resv | flag */
    idx += CHANNEL_ENTRY_BYTES;
    req = (uint64_t *)((uint8_t *)buf_ptr + (idx & (CHANNEL_BUF_SIZE - 1)));
    curr_flag = !((idx >> nvshmemi_device_state_d.proxy_channel_buf_logsize) & 1);
    /* mask lower four bytes. */
    const uint64_t lvalue_low = lvalue_buffer & mask_4_bytes;
    *((volatile uint64_t *)req) = (lvalue_low << 32 | ((uint64_t)pe_u16 << 16) | curr_flag);

    /* put_inline_request_1
     * 32 | 16 | 8 | 8
     * lvalue(high) | size | resv | flag */
    idx += CHANNEL_ENTRY_BYTES;
    req = (uint64_t *)((uint8_t *)buf_ptr + (idx & (CHANNEL_BUF_SIZE - 1)));
    curr_flag = !((idx >> nvshmemi_device_state_d.proxy_channel_buf_logsize) & 1);
    const uint64_t lvalue_high = lvalue_buffer & ~mask_4_bytes;
    *((volatile uint64_t *)req) = (lvalue_high | size_u64 << 16 | curr_flag);
}

template <typename T>
static __device__ inline void nvshmemi_proxy_rma_p(void *rptr, const T value, int pe) {
    transfer_inline<T>(rptr, value, pe, NVSHMEMI_OP_P);
}

template <typename T>
static __device__ inline void amo(void *rptr,
                                  uint64_t g_buf_counter /* used only for fetch atomics */,
                                  T swap_add, T compare, int pe, nvshmemi_amo_t amo_op) {
    uint64_t idx, tail_idx, *req;
    int size = PROXY_AMO_REQ_BYTES;
    int group_size = 1;
    void *buf_ptr = nvshmemi_device_state_d.proxy_channels_buf;
    void *base_ptr = nvshmemi_device_state_d.heap_base;

    idx = atomicAdd((unsigned long long int *)nvshmemi_device_state_d.proxy_channels_issue, size);
    tail_idx = idx + (size - 1);

    // flow-control
    check_channel_availability(tail_idx);

    req = (uint64_t *)((uint8_t *)buf_ptr + (idx & (CHANNEL_BUF_SIZE - 1)));
    uint64_t curr_flag = !((idx >> nvshmemi_device_state_d.proxy_channel_buf_logsize) & 1);
    uint64_t roffset = (uint64_t)((char *)rptr - (char *)base_ptr);
    uint64_t op = NVSHMEMI_OP_AMO;
    uint64_t amo = amo_op;
    uint16_t pe_u16 = pe;
    uint64_t size_u64 = sizeof(T);
    uint64_t swap_add_buffer;
    uint64_t compare_buffer;
    const uint64_t mask_1_byte = 0x00FFFFFFFFFFFFFF;
    const uint64_t mask_4_bytes = 0x00000000FFFFFFFF;
    const uint64_t mask_7_bytes = 0x00000000000000FF;

    convert_val_to_uint64(&swap_add_buffer, &swap_add, sizeof(T));
    convert_val_to_uint64(&compare_buffer, &compare, sizeof(T));

    /* base_request_t
     * 32 | 8 | 8 | 8 | 8
     * roffset_high | roffset_low | op | group_size | flag */
    *((volatile uint64_t *)req) =
        (uint64_t)((roffset << 24) | (op << 16) | (group_size << 8) | curr_flag);

    /* amo_request_0
     * 32 | 16 | 8 | 8
     * swap_add_low | pe | amo | flag */
    idx += CHANNEL_ENTRY_BYTES;
    req = (uint64_t *)((uint8_t *)buf_ptr + (idx & (CHANNEL_BUF_SIZE - 1)));
    curr_flag = !((idx >> nvshmemi_device_state_d.proxy_channel_buf_logsize) & 1);
    const uint64_t swap_add_low = swap_add_buffer & mask_4_bytes;
    *((volatile uint64_t *)req) =
        (swap_add_low << 32 | ((uint64_t)pe_u16 << 16) | (amo << 8) | curr_flag);

    /* amo_request_1
     * 32 | 16 | 8 | 8
     * swap_add_high | size | compare_low | flag */
    idx += CHANNEL_ENTRY_BYTES;
    req = (uint64_t *)((uint8_t *)buf_ptr + (idx & (CHANNEL_BUF_SIZE - 1)));
    curr_flag = !((idx >> nvshmemi_device_state_d.proxy_channel_buf_logsize) & 1);
    const uint64_t swap_add_high = swap_add_buffer & ~mask_4_bytes;
    const uint64_t compare_low = compare_buffer & mask_7_bytes;
    *((volatile uint64_t *)req) = (swap_add_high | size_u64 << 16 | compare_low << 8 | curr_flag);

    /* amo_request_2
     * 32 | 16 | 8 | 8
     * comapare_high | flag */
    idx += CHANNEL_ENTRY_BYTES;
    req = (uint64_t *)((uint8_t *)buf_ptr + (idx & (CHANNEL_BUF_SIZE - 1)));
    curr_flag = !((idx >> nvshmemi_device_state_d.proxy_channel_buf_logsize) & 1);
    /* only mask the low byte for */
    const uint64_t compare_high = compare_buffer & ~mask_7_bytes;
    *((volatile uint64_t *)req) = (compare_high | curr_flag);

    /* amo_request_3
     * 32 | 16 | 8 | 8
     * g_buf_counter_low | flag */
    idx += CHANNEL_ENTRY_BYTES;
    req = (uint64_t *)((uint8_t *)buf_ptr + (idx & (CHANNEL_BUF_SIZE - 1)));
    curr_flag = !((idx >> nvshmemi_device_state_d.proxy_channel_buf_logsize) & 1);
    /* assumes g_buf_counter <= (1 << 56) */
    const uint64_t g_buf_counter_low = g_buf_counter & mask_1_byte;
    *((volatile uint64_t *)req) = (g_buf_counter_low << 8 | curr_flag);
}

template <typename T>
static __device__ inline void nvshmemi_proxy_amo_nonfetch(void *rptr, T swap_add, int pe,
                                                          nvshmemi_amo_t op) {
    amo<T>(rptr, 0 /* dummy value */, swap_add, 0, pe, op);
}

template <typename T>
static __device__ inline void nvshmemi_proxy_amo_fetch(void *rptr, void *lptr, T swap_add,
                                                       T compare, int pe, nvshmemi_amo_t op) {
    uint64_t counter = atomicAdd(
        (unsigned long long int *)nvshmemi_device_state_d.proxy_channel_g_buf_head_ptr, 1);
    uint64_t idx = counter * sizeof(g_elem_t);
    uint64_t idx_in_buf = idx & (nvshmemi_device_state_d.proxy_channel_g_buf_size - 1);
    g_elem_t *elem = (g_elem_t *)(nvshmemi_device_state_d.proxy_channel_g_buf + idx_in_buf);
    uint64_t flag = (idx >> nvshmemi_device_state_d.proxy_channel_g_buf_log_size) * 2;
    size_t atomic_size = sizeof(T);

    /* wait until element can be used */
    nvshmemi_wait_until_greater_than_equals<uint64_t>((volatile uint64_t *)&(elem->flag), flag,
                                                      NVSHMEMI_CALL_SITE_AMO_FETCH_WAIT_FLAG);
    __threadfence();

    amo<T>(rptr, counter, swap_add, compare, pe, op);

    /* The IBDEVX transport doesn't rely on an active message from the receiver for atomics. */
    if (nvshmemi_device_state_d.atomics_complete_on_quiet) {
        nvshmemi_proxy_quiet(false);
        elem->flag += 1;
        /* MLNX NICs will typically return 8 byte atomics in host-endian and 4 byte atomics in
         * big-enian. This behavior is controlled by flags read from the NIC at runtime.
         */
        if (atomic_size < nvshmemi_device_state_d.atomics_le_min_size) {
            if (atomic_size == 8) {
                NTOH64(&elem->data);
            } else if (atomic_size == 4) {
                NTOH32(&elem->data);
            } else {
                /* Our APIs currently only support 4 and 8-byte atomics. Any other size is a fatal
                 * error. */
                assert(false);
            }
        }
    } else {
        nvshmemi_wait_until_greater_than_equals<uint64_t>(
            (volatile uint64_t *)&(elem->flag), flag + 1, NVSHMEMI_CALL_SITE_AMO_FETCH_WAIT_DATA);
    }
    __threadfence();
    T return_val = *(T *)(&(elem->data));
    __threadfence();

    /* release the element for the next thread */
    elem->flag += 1;

    *((T *)lptr) = return_val;
}

static __device__ inline void nvshmemi_proxy_fence() {
    // making it a no-op as it is a no-op for IB RC, the only transport
    uint64_t idx, tail_idx, *req;
    int size = sizeof(uint64_t);

    idx = atomicAdd((unsigned long long int *)nvshmemi_device_state_d.proxy_channels_issue, size);
    tail_idx = idx + (size - 1);

    // flow-control
    check_channel_availability(tail_idx);

    req = (uint64_t *)((uint64_t)nvshmemi_device_state_d.proxy_channels_buf +
                       (idx & (nvshmemi_device_state_d.proxy_channel_buf_size - 1)));
    uint64_t curr_flag = !((idx >> nvshmemi_device_state_d.proxy_channel_buf_logsize) & 1);
    uint64_t op = NVSHMEMI_OP_FENCE;

    /* base_request_t
     * 32 | 8 | 8 | 8 | 8
     * resv | resv | op | resv | flag */
    *((volatile uint64_t *)req) = (uint64_t)((op << 16) | curr_flag);

    return;
}

#endif /* __CUDA_ARCH__ */
#endif /* PROXY_DEVICE_CUH */
