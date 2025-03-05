/*
 * Copyright (c) 2016-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _NVSHMEM_PROXY_CHANNEL_H_
#define _NVSHMEM_PROXY_CHANNEL_H_

#if not defined __CUDACC_RTC__
#include <stdint.h>
#else
#include <cuda/std/cstdint>
#endif

/* Note: this is only safe because we are using this across a single system
 * that shares the GPUs endianness. This struct is not actually portable.
 */
typedef union channel_bounce_buffer {
    char bytes[8];
    uint64_t whole_buffer;
} channel_bounce_buffer_t;

/* base_request_t
 * 32 | 8 | 8 | 8 | 8
 * roffset_high | roffset_low | op | group_size | flag */
typedef struct __attribute__((packed)) base_request {
    volatile uint8_t flag;
    uint8_t groupsize;
    uint8_t op;
    uint8_t roffset_low;   // target is remote
    uint32_t roffset_high; /*used as pe for base-only requests*/
} base_request_t;
static_assert(sizeof(base_request_t) == 8, "request_size must be 8 bytes.");

/* put_dma_request_0
 * 32 | 16 | 8 | 8
 * laddr_high | laddr_3| laddr_2 | flag */
typedef struct __attribute__((packed)) put_dma_request_0 {
    volatile uint8_t flag;
    uint8_t laddr_2;
    uint16_t laddr_3;  // source is local
    uint32_t laddr_high;
} put_dma_request_0_t;
static_assert(sizeof(put_dma_request_0) == 8, "request_size must be 8 bytes.");

/* put_dma_request_1
 * 32 | 16 | 8 | 8
 * size_high | size_low | laddr_low | flag */
typedef struct __attribute__((packed)) put_dma_request_1 {
    volatile uint8_t flag;
    uint8_t laddr_low;
    uint16_t size_low;
    uint32_t size_high;
} put_dma_request_1_t;
static_assert(sizeof(put_dma_request_1) == 8, "request_size must be 8 bytes.");

/* put_dma_request_2
 * 32 | 16 | 8 | 8
 * resv2 | pe | resv | flag */
typedef struct __attribute__((packed)) put_dma_request_2 {
    volatile uint8_t flag;
    uint8_t resv;
    uint16_t pe;
    uint32_t resv1;
} put_dma_request_2_t;
static_assert(sizeof(put_dma_request_2) == 8, "request_size must be 8 bytes.");

/* put_inline_request_0
 * 32 | 16 | 8 | 8
 * loffset_high | loffset_low | pe | flag */
typedef struct __attribute__((packed)) put_inline_request_0 {
    volatile uint8_t flag;
    uint8_t resv;
    uint16_t pe;
    uint32_t lvalue_low;
} put_inline_request_0_t;
static_assert(sizeof(put_inline_request_0) == 8, "request_size must be 8 bytes.");

/* put_inline_request_1
 * 32 | 16 | 8 | 8
 * size_high | size_low | resv | flag */
typedef struct __attribute__((packed)) put_inline_request_1 {
    volatile uint8_t flag;
    uint8_t resv;
    uint16_t size;
    uint32_t lvalue_high;
} put_inline_request_1_t;
static_assert(sizeof(put_inline_request_1) == 8, "request_size must be 8 bytes.");

/* amo_request_0
 * 32 | 16 | 8 | 8
 * lvalue_low | pe | amo | flag */
typedef struct __attribute__((packed)) amo_request_0 {
    volatile uint8_t flag;
    uint8_t amo;
    uint16_t pe;
    uint32_t swap_add_low;
} amo_request_0_t;
static_assert(sizeof(amo_request_0) == 8, "request_size must be 8 bytes.");

/* amo_request_1
 * 32 | 16 | 8 | 8
 * lvalue_high | resv | size | flag */
typedef struct __attribute__((packed)) amo_request_1 {
    volatile uint8_t flag;
    uint8_t compare_low;
    uint16_t size;
    uint32_t swap_add_high;
} amo_request_1_t;
static_assert(sizeof(amo_request_1) == 8, "request_size must be 8 bytes.");

/* amo_request_2
 * 56 | 8
 * compare_high | flag */
typedef struct __attribute__((packed)) amo_request_2 {
    volatile uint8_t flag;
    uint8_t compare_high[7];
} amo_request_2_t;
static_assert(sizeof(amo_request_2) == 8, "request_size must be 8 bytes.");

typedef struct __attribute__((packed)) amo_request_3 {
    volatile uint8_t flag;
    uint8_t g_buf_counter[7];
} amo_request_3_t;
static_assert(sizeof(amo_request_3) == 8, "request_size must be 8 bytes.");

#endif
