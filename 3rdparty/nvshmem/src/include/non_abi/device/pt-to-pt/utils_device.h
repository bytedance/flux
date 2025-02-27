/*
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _COMM_DEVICE_UTILS_H
#define _COMM_DEVICE_UTILS_H

#if not defined __CUDACC_RTC__
#include <stdint.h>
#else
#include "cuda/std/cstdint"
#endif

#include <cuda_runtime.h>

#define NVSHMEMI_COMM_DEVICE_UTILS_USE_PTX

#define NTOH64(x)                                                             \
    *x = ((*(x)&0xFF00000000000000) >> 56 | (*(x)&0x00FF000000000000) >> 40 | \
          (*(x)&0x0000FF0000000000) >> 24 | (*(x)&0x000000FF00000000) >> 8 |  \
          (*(x)&0x00000000FF000000) << 8 | (*(x)&0x0000000000FF0000) << 24 |  \
          (*(x)&0x000000000000FF00) << 40 | (*(x)&0x00000000000000FF) << 56)

#define NTOH32(x)                                                                     \
    *x = ((*(x)&0xFF000000) >> 24 | (*(x)&0x00FF0000) >> 8 | (*(x)&0x0000FF00) << 8 | \
          (*(x)&0x000000FF) << 24)

#ifdef NVSHMEMI_COMM_DEVICE_UTILS_USE_PTX

__device__ static inline uint64_t BSWAP64(uint64_t x) {
    uint64_t ret;
    asm volatile(
        "{\n\t"
        ".reg .b32 mask;\n\t"
        ".reg .b32 ign;\n\t"
        ".reg .b32 lo;\n\t"
        ".reg .b32 hi;\n\t"
        ".reg .b32 new_lo;\n\t"
        ".reg .b32 new_hi;\n\t"
        "mov.b32 mask, 0x0123;\n\t"
        "mov.b64 {lo,hi}, %1;\n\t"
        "prmt.b32 new_hi, lo, ign, mask;\n\t"
        "prmt.b32 new_lo, hi, ign, mask;\n\t"
        "mov.b64 %0, {new_lo,new_hi};\n\t"
        "}"
        : "=l"(ret)
        : "l"(x));
    return ret;
}

__device__ static inline uint32_t BSWAP32(uint32_t x) {
    uint32_t ret;
    asm volatile(
        "{\n\t"
        ".reg .b32 mask;\n\t"
        ".reg .b32 ign;\n\t"
        "mov.b32 mask, 0x0123;\n\t"
        "prmt.b32 %0, %1, ign, mask;\n\t"
        "}"
        : "=r"(ret)
        : "r"(x));
    return ret;
}

__device__ static inline uint16_t BSWAP16(uint16_t x) {
    uint16_t ret;

    uint32_t a = (uint32_t)x;
    uint32_t d;
    asm volatile(
        "{\n\t"
        ".reg .b32 mask;\n\t"
        ".reg .b32 ign;\n\t"
        "mov.b32 mask, 0x4401;\n\t"
        "mov.b32 ign, 0x0;\n\t"
        "prmt.b32 %0, %1, ign, mask;\n\t"
        "}"
        : "=r"(d)
        : "r"(a));
    ret = (uint16_t)d;
    return ret;
}

#else /* NVSHMEMI_COMM_DEVICE_UTILS_USE_PTX */

#define BSWAP64(x)                                                               \
    ((((x)&0xff00000000000000ull) >> 56) | (((x)&0x00ff000000000000ull) >> 40) | \
     (((x)&0x0000ff0000000000ull) >> 24) | (((x)&0x000000ff00000000ull) >> 8) |  \
     (((x)&0x00000000ff000000ull) << 8) | (((x)&0x0000000000ff0000ull) << 24) |  \
     (((x)&0x000000000000ff00ull) << 40) | (((x)&0x00000000000000ffull) << 56))

#define BSWAP32(x)                                                                  \
    ((((x)&0xff000000) >> 24) | (((x)&0x00ff0000) >> 8) | (((x)&0x0000ff00) << 8) | \
     (((x)&0x000000ff) << 24))

#define BSWAP16(x) ((((x)&0xff00) >> 8) | (((x)&0x00ff) << 8))

#endif /* NVSHMEMI_COMM_DEVICE_UTILS_USE_PTX */

#define HTOBE64(x) BSWAP64(x)
#define HTOBE32(x) BSWAP32(x)
#define HTOBE16(x) BSWAP16(x)

#endif /* _COMM_DEVICE_UTILS_H */
