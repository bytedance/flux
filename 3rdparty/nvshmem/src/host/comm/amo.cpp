/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#define NVSHMEMI_HOST_ONLY
#include <stddef.h>                        // for ptrdiff_t, size_t
#include <stdint.h>                        // for int32_t, int64_t, uint32_t
#include "host/nvshmem_api.h"              // for nvshmem_double_atomic_fetch
#include "non_abi/nvshmemx_error.h"        // for NVSHMEMI_ERROR_PRINT
#include "internal/host/nvshmemi_types.h"  // for nvshmemi_state, nvshmemi_s...

#define NVSHMEM_TYPE_INC_NOT_IMPLEMENTED(Name, TYPE)                                \
    void nvshmem_##Name##_atomic_inc(TYPE *target, int pe) {                        \
        NVSHMEMI_ERROR_PRINT("[%d] nvshmem_" #Name "_atomic_inc() not implemented", \
                             nvshmemi_state->mype);                                 \
    }

NVSHMEM_TYPE_INC_NOT_IMPLEMENTED(uint, unsigned int)
NVSHMEM_TYPE_INC_NOT_IMPLEMENTED(ulong, unsigned long)
NVSHMEM_TYPE_INC_NOT_IMPLEMENTED(ulonglong, unsigned long long)
NVSHMEM_TYPE_INC_NOT_IMPLEMENTED(int32, int32_t)
NVSHMEM_TYPE_INC_NOT_IMPLEMENTED(uint32, uint32_t)
NVSHMEM_TYPE_INC_NOT_IMPLEMENTED(int64, int64_t) /*XXX:not implemented*/
NVSHMEM_TYPE_INC_NOT_IMPLEMENTED(uint64, uint64_t)
NVSHMEM_TYPE_INC_NOT_IMPLEMENTED(int, int)
NVSHMEM_TYPE_INC_NOT_IMPLEMENTED(long, long)
NVSHMEM_TYPE_INC_NOT_IMPLEMENTED(longlong, long long) /*XXX:not implemented*/
NVSHMEM_TYPE_INC_NOT_IMPLEMENTED(size, size_t)
NVSHMEM_TYPE_INC_NOT_IMPLEMENTED(ptrdiff, ptrdiff_t) /*XXX:not implemented*/

#define NVSHMEM_TYPE_ADD_NOT_IMPLEMENTED(Name, TYPE)                                \
    void nvshmem_##Name##_atomic_add(TYPE *target, TYPE value, int pe) {            \
        NVSHMEMI_ERROR_PRINT("[%d] nvshmem_" #Name "_atomic_add() not implemented", \
                             nvshmemi_state->mype);                                 \
    }

NVSHMEM_TYPE_ADD_NOT_IMPLEMENTED(uint, unsigned int)
NVSHMEM_TYPE_ADD_NOT_IMPLEMENTED(ulong, unsigned long)
NVSHMEM_TYPE_ADD_NOT_IMPLEMENTED(ulonglong, unsigned long long)
NVSHMEM_TYPE_ADD_NOT_IMPLEMENTED(int32, int32_t)
NVSHMEM_TYPE_ADD_NOT_IMPLEMENTED(uint32, uint32_t)
NVSHMEM_TYPE_ADD_NOT_IMPLEMENTED(int64, int64_t) /*XXX:not implemented*/
NVSHMEM_TYPE_ADD_NOT_IMPLEMENTED(uint64, uint64_t)
NVSHMEM_TYPE_ADD_NOT_IMPLEMENTED(int, int)
NVSHMEM_TYPE_ADD_NOT_IMPLEMENTED(long, long)
NVSHMEM_TYPE_ADD_NOT_IMPLEMENTED(longlong, long long) /*XXX:not implemented*/
NVSHMEM_TYPE_ADD_NOT_IMPLEMENTED(size, size_t)
NVSHMEM_TYPE_ADD_NOT_IMPLEMENTED(ptrdiff, ptrdiff_t) /*XXX:not implemented*/

#define NVSHMEM_TYPE_SET_NOT_IMPLEMENTED(Name, TYPE)                                \
    void nvshmem_##Name##_atomic_set(TYPE *target, TYPE value, int pe) {            \
        NVSHMEMI_ERROR_PRINT("[%d] nvshmem_" #Name "_atomic_set() not implemented", \
                             nvshmemi_state->mype);                                 \
    }
NVSHMEM_TYPE_SET_NOT_IMPLEMENTED(uint, unsigned int)
NVSHMEM_TYPE_SET_NOT_IMPLEMENTED(ulong, unsigned long)
NVSHMEM_TYPE_SET_NOT_IMPLEMENTED(ulonglong, unsigned long long)
NVSHMEM_TYPE_SET_NOT_IMPLEMENTED(int32, int32_t)
NVSHMEM_TYPE_SET_NOT_IMPLEMENTED(uint32, uint32_t)
NVSHMEM_TYPE_SET_NOT_IMPLEMENTED(int64, int64_t)
NVSHMEM_TYPE_SET_NOT_IMPLEMENTED(uint64, uint64_t)
NVSHMEM_TYPE_SET_NOT_IMPLEMENTED(int, int)
NVSHMEM_TYPE_SET_NOT_IMPLEMENTED(long, long)
NVSHMEM_TYPE_SET_NOT_IMPLEMENTED(longlong, long long)
NVSHMEM_TYPE_SET_NOT_IMPLEMENTED(size, size_t)
NVSHMEM_TYPE_SET_NOT_IMPLEMENTED(ptrdiff, ptrdiff_t)
NVSHMEM_TYPE_SET_NOT_IMPLEMENTED(float, float)
NVSHMEM_TYPE_SET_NOT_IMPLEMENTED(double, double)

#define NVSHMEM_TYPE_AND_NOT_IMPLEMENTED(Name, TYPE)                                \
    void nvshmem_##Name##_atomic_and(TYPE *target, TYPE value, int pe) {            \
        NVSHMEMI_ERROR_PRINT("[%d] nvshmem_" #Name "_atomic_and() not implemented", \
                             nvshmemi_state->mype);                                 \
    }
NVSHMEM_TYPE_AND_NOT_IMPLEMENTED(uint, unsigned int)
NVSHMEM_TYPE_AND_NOT_IMPLEMENTED(ulong, unsigned long)
NVSHMEM_TYPE_AND_NOT_IMPLEMENTED(ulonglong, unsigned long long)
NVSHMEM_TYPE_AND_NOT_IMPLEMENTED(int32, int32_t)
NVSHMEM_TYPE_AND_NOT_IMPLEMENTED(uint32, uint32_t)
NVSHMEM_TYPE_AND_NOT_IMPLEMENTED(int64, int64_t)
NVSHMEM_TYPE_AND_NOT_IMPLEMENTED(uint64, uint64_t)

#define NVSHMEM_TYPE_OR_NOT_IMPLEMENTED(Name, TYPE)                                \
    void nvshmem_##Name##_atomic_or(TYPE *target, TYPE value, int pe) {            \
        NVSHMEMI_ERROR_PRINT("[%d] nvshmem_" #Name "_atomic_or() not implemented", \
                             nvshmemi_state->mype);                                \
    }
NVSHMEM_TYPE_OR_NOT_IMPLEMENTED(uint, unsigned int)
NVSHMEM_TYPE_OR_NOT_IMPLEMENTED(ulong, unsigned long)
NVSHMEM_TYPE_OR_NOT_IMPLEMENTED(ulonglong, unsigned long long)
NVSHMEM_TYPE_OR_NOT_IMPLEMENTED(int32, int32_t)
NVSHMEM_TYPE_OR_NOT_IMPLEMENTED(uint32, uint32_t)
NVSHMEM_TYPE_OR_NOT_IMPLEMENTED(int64, int64_t)
NVSHMEM_TYPE_OR_NOT_IMPLEMENTED(uint64, uint64_t)

#define NVSHMEM_TYPE_XOR_NOT_IMPLEMENTED(Name, TYPE)                                \
    void nvshmem_##Name##_atomic_xor(TYPE *target, TYPE value, int pe) {            \
        NVSHMEMI_ERROR_PRINT("[%d] nvshmem_" #Name "_atomic_xor() not implemented", \
                             nvshmemi_state->mype);                                 \
    }
NVSHMEM_TYPE_XOR_NOT_IMPLEMENTED(uint, unsigned int)
NVSHMEM_TYPE_XOR_NOT_IMPLEMENTED(ulong, unsigned long)
NVSHMEM_TYPE_XOR_NOT_IMPLEMENTED(ulonglong, unsigned long long)
NVSHMEM_TYPE_XOR_NOT_IMPLEMENTED(int32, int32_t)
NVSHMEM_TYPE_XOR_NOT_IMPLEMENTED(uint32, uint32_t)
NVSHMEM_TYPE_XOR_NOT_IMPLEMENTED(int64, int64_t)
NVSHMEM_TYPE_XOR_NOT_IMPLEMENTED(uint64, uint64_t)

#define NVSHMEM_TYPE_FETCH_NOT_IMPLEMENTED(Name, TYPE)                                \
    TYPE nvshmem_##Name##_atomic_fetch(const TYPE *target, int pe) {                  \
        NVSHMEMI_ERROR_PRINT("[%d] nvshmem_" #Name "_atomic_fetch() not implemented", \
                             nvshmemi_state->mype);                                   \
        return 0;                                                                     \
    }
NVSHMEM_TYPE_FETCH_NOT_IMPLEMENTED(uint, unsigned int)
NVSHMEM_TYPE_FETCH_NOT_IMPLEMENTED(ulong, unsigned long)
NVSHMEM_TYPE_FETCH_NOT_IMPLEMENTED(ulonglong, unsigned long long)
NVSHMEM_TYPE_FETCH_NOT_IMPLEMENTED(int32, int32_t)
NVSHMEM_TYPE_FETCH_NOT_IMPLEMENTED(uint32, uint32_t)
NVSHMEM_TYPE_FETCH_NOT_IMPLEMENTED(int64, int64_t)
NVSHMEM_TYPE_FETCH_NOT_IMPLEMENTED(uint64, uint64_t)
NVSHMEM_TYPE_FETCH_NOT_IMPLEMENTED(int, int)
NVSHMEM_TYPE_FETCH_NOT_IMPLEMENTED(long, long)
NVSHMEM_TYPE_FETCH_NOT_IMPLEMENTED(longlong, long long)
NVSHMEM_TYPE_FETCH_NOT_IMPLEMENTED(size, size_t)
NVSHMEM_TYPE_FETCH_NOT_IMPLEMENTED(ptrdiff, ptrdiff_t)
NVSHMEM_TYPE_FETCH_NOT_IMPLEMENTED(float, float)
NVSHMEM_TYPE_FETCH_NOT_IMPLEMENTED(double, double)

#define NVSHMEM_TYPE_FETCH_INC_NOT_IMPLEMENTED(Name, TYPE)                                \
    TYPE nvshmem_##Name##_atomic_fetch_inc(TYPE *target, int pe) {                        \
        NVSHMEMI_ERROR_PRINT("[%d] nvshmem_" #Name "_atomic_fetch_inc() not implemented", \
                             nvshmemi_state->mype);                                       \
        return 0;                                                                         \
    }

NVSHMEM_TYPE_FETCH_INC_NOT_IMPLEMENTED(uint, unsigned int)
NVSHMEM_TYPE_FETCH_INC_NOT_IMPLEMENTED(ulong, unsigned long)
NVSHMEM_TYPE_FETCH_INC_NOT_IMPLEMENTED(ulonglong, unsigned long long)
NVSHMEM_TYPE_FETCH_INC_NOT_IMPLEMENTED(int32, int32_t)
NVSHMEM_TYPE_FETCH_INC_NOT_IMPLEMENTED(uint32, uint32_t)
NVSHMEM_TYPE_FETCH_INC_NOT_IMPLEMENTED(int64, int64_t) /*XXX:not implemented*/
NVSHMEM_TYPE_FETCH_INC_NOT_IMPLEMENTED(uint64, uint64_t)
NVSHMEM_TYPE_FETCH_INC_NOT_IMPLEMENTED(int, int)
NVSHMEM_TYPE_FETCH_INC_NOT_IMPLEMENTED(long, long)
NVSHMEM_TYPE_FETCH_INC_NOT_IMPLEMENTED(longlong, long long) /*XXX:not implemented*/
NVSHMEM_TYPE_FETCH_INC_NOT_IMPLEMENTED(size, size_t)
NVSHMEM_TYPE_FETCH_INC_NOT_IMPLEMENTED(ptrdiff, ptrdiff_t) /*XXX:not implemented*/

#define NVSHMEM_TYPE_FETCH_ADD_NOT_IMPLEMENTED(Name, TYPE)                           \
    TYPE nvshmem_##Name##_atomic_fetch_add(TYPE *target, TYPE value, int pe) {       \
        NVSHMEMI_ERROR_PRINT("[%d] nvshmem_" #Name "_atomic_fadd() not implemented", \
                             nvshmemi_state->mype);                                  \
        return 0;                                                                    \
    }

NVSHMEM_TYPE_FETCH_ADD_NOT_IMPLEMENTED(uint, unsigned int)
NVSHMEM_TYPE_FETCH_ADD_NOT_IMPLEMENTED(ulong, unsigned long)
NVSHMEM_TYPE_FETCH_ADD_NOT_IMPLEMENTED(ulonglong, unsigned long long)
NVSHMEM_TYPE_FETCH_ADD_NOT_IMPLEMENTED(int32, int32_t)
NVSHMEM_TYPE_FETCH_ADD_NOT_IMPLEMENTED(uint32, uint32_t)
NVSHMEM_TYPE_FETCH_ADD_NOT_IMPLEMENTED(int64, int64_t) /*XXX:not implemented*/
NVSHMEM_TYPE_FETCH_ADD_NOT_IMPLEMENTED(uint64, uint64_t)
NVSHMEM_TYPE_FETCH_ADD_NOT_IMPLEMENTED(int, int)
NVSHMEM_TYPE_FETCH_ADD_NOT_IMPLEMENTED(long, long)
NVSHMEM_TYPE_FETCH_ADD_NOT_IMPLEMENTED(longlong, long long) /*XXX:not implemented*/
NVSHMEM_TYPE_FETCH_ADD_NOT_IMPLEMENTED(size, size_t)
NVSHMEM_TYPE_FETCH_ADD_NOT_IMPLEMENTED(ptrdiff, ptrdiff_t) /*XXX:not implemented*/

#define NVSHMEM_TYPE_SWAP_NOT_IMPLEMENTED(Name, TYPE)                                \
    TYPE nvshmem_##Name##_atomic_swap(TYPE *target, TYPE value, int pe) {            \
        NVSHMEMI_ERROR_PRINT("[%d] nvshmem_" #Name "_atomic_swap() not implemented", \
                             nvshmemi_state->mype);                                  \
        return 0;                                                                    \
    }
NVSHMEM_TYPE_SWAP_NOT_IMPLEMENTED(uint, unsigned int)
NVSHMEM_TYPE_SWAP_NOT_IMPLEMENTED(ulong, unsigned long)
NVSHMEM_TYPE_SWAP_NOT_IMPLEMENTED(ulonglong, unsigned long long)
NVSHMEM_TYPE_SWAP_NOT_IMPLEMENTED(int32, int32_t)
NVSHMEM_TYPE_SWAP_NOT_IMPLEMENTED(uint32, uint32_t)
NVSHMEM_TYPE_SWAP_NOT_IMPLEMENTED(int64, int64_t)
NVSHMEM_TYPE_SWAP_NOT_IMPLEMENTED(uint64, uint64_t)
NVSHMEM_TYPE_SWAP_NOT_IMPLEMENTED(int, int)
NVSHMEM_TYPE_SWAP_NOT_IMPLEMENTED(long, long)
NVSHMEM_TYPE_SWAP_NOT_IMPLEMENTED(longlong, long long)
NVSHMEM_TYPE_SWAP_NOT_IMPLEMENTED(size, size_t)
NVSHMEM_TYPE_SWAP_NOT_IMPLEMENTED(ptrdiff, ptrdiff_t)
NVSHMEM_TYPE_SWAP_NOT_IMPLEMENTED(float, float)
NVSHMEM_TYPE_SWAP_NOT_IMPLEMENTED(double, double)

#define NVSHMEM_TYPE_COMPARE_SWAP_NOT_IMPLEMENTED(Name, TYPE)                                \
    TYPE nvshmem_##Name##_atomic_compare_swap(TYPE *target, TYPE cond, TYPE value, int pe) { \
        NVSHMEMI_ERROR_PRINT("[%d] nvshmem_" #Name "_atomic_compare_swap() not implemented", \
                             nvshmemi_state->mype);                                          \
        return value;                                                                        \
    }
NVSHMEM_TYPE_COMPARE_SWAP_NOT_IMPLEMENTED(uint, unsigned int)
NVSHMEM_TYPE_COMPARE_SWAP_NOT_IMPLEMENTED(ulong, unsigned long)
NVSHMEM_TYPE_COMPARE_SWAP_NOT_IMPLEMENTED(ulonglong, unsigned long long)
NVSHMEM_TYPE_COMPARE_SWAP_NOT_IMPLEMENTED(int32, int32_t)
NVSHMEM_TYPE_COMPARE_SWAP_NOT_IMPLEMENTED(uint32, uint32_t)
NVSHMEM_TYPE_COMPARE_SWAP_NOT_IMPLEMENTED(int64, int64_t)
NVSHMEM_TYPE_COMPARE_SWAP_NOT_IMPLEMENTED(uint64, uint64_t)
NVSHMEM_TYPE_COMPARE_SWAP_NOT_IMPLEMENTED(int, int)
NVSHMEM_TYPE_COMPARE_SWAP_NOT_IMPLEMENTED(long, long)
NVSHMEM_TYPE_COMPARE_SWAP_NOT_IMPLEMENTED(longlong, long long)
NVSHMEM_TYPE_COMPARE_SWAP_NOT_IMPLEMENTED(size, size_t)
NVSHMEM_TYPE_COMPARE_SWAP_NOT_IMPLEMENTED(ptrdiff, ptrdiff_t)

#define NVSHMEM_TYPE_FETCH_AND_NOT_IMPLEMENTED(Name, TYPE)                                \
    TYPE nvshmem_##Name##_atomic_fetch_and(TYPE *target, TYPE value, int pe) {            \
        NVSHMEMI_ERROR_PRINT("[%d] nvshmem_" #Name "_atomic_fetch_and() not implemented", \
                             nvshmemi_state->mype);                                       \
        return value;                                                                     \
    }
NVSHMEM_TYPE_FETCH_AND_NOT_IMPLEMENTED(uint, unsigned int)
NVSHMEM_TYPE_FETCH_AND_NOT_IMPLEMENTED(ulong, unsigned long)
NVSHMEM_TYPE_FETCH_AND_NOT_IMPLEMENTED(ulonglong, unsigned long long)
NVSHMEM_TYPE_FETCH_AND_NOT_IMPLEMENTED(int32, int32_t)
NVSHMEM_TYPE_FETCH_AND_NOT_IMPLEMENTED(uint32, uint32_t)
NVSHMEM_TYPE_FETCH_AND_NOT_IMPLEMENTED(int64, int64_t)
NVSHMEM_TYPE_FETCH_AND_NOT_IMPLEMENTED(uint64, uint64_t)

#define NVSHMEM_TYPE_FETCH_OR_NOT_IMPLEMENTED(Name, TYPE)                                \
    TYPE nvshmem_##Name##_atomic_fetch_or(TYPE *target, TYPE value, int pe) {            \
        NVSHMEMI_ERROR_PRINT("[%d] nvshmem_" #Name "_atomic_fetch_or() not implemented", \
                             nvshmemi_state->mype);                                      \
        return value;                                                                    \
    }
NVSHMEM_TYPE_FETCH_OR_NOT_IMPLEMENTED(uint, unsigned int)
NVSHMEM_TYPE_FETCH_OR_NOT_IMPLEMENTED(ulong, unsigned long)
NVSHMEM_TYPE_FETCH_OR_NOT_IMPLEMENTED(ulonglong, unsigned long long)
NVSHMEM_TYPE_FETCH_OR_NOT_IMPLEMENTED(int32, int32_t)
NVSHMEM_TYPE_FETCH_OR_NOT_IMPLEMENTED(uint32, uint32_t)
NVSHMEM_TYPE_FETCH_OR_NOT_IMPLEMENTED(int64, int64_t)
NVSHMEM_TYPE_FETCH_OR_NOT_IMPLEMENTED(uint64, uint64_t)

#define NVSHMEM_TYPE_FETCH_XOR_NOT_IMPLEMENTED(Name, TYPE)                                \
    TYPE nvshmem_##Name##_atomic_fetch_xor(TYPE *target, TYPE value, int pe) {            \
        NVSHMEMI_ERROR_PRINT("[%d] nvshmem_" #Name "_atomic_fetch_xor() not implemented", \
                             nvshmemi_state->mype);                                       \
        return value;                                                                     \
    }
NVSHMEM_TYPE_FETCH_XOR_NOT_IMPLEMENTED(uint, unsigned int)
NVSHMEM_TYPE_FETCH_XOR_NOT_IMPLEMENTED(ulong, unsigned long)
NVSHMEM_TYPE_FETCH_XOR_NOT_IMPLEMENTED(ulonglong, unsigned long long)
NVSHMEM_TYPE_FETCH_XOR_NOT_IMPLEMENTED(int32, int32_t)
NVSHMEM_TYPE_FETCH_XOR_NOT_IMPLEMENTED(uint32, uint32_t)
NVSHMEM_TYPE_FETCH_XOR_NOT_IMPLEMENTED(int64, int64_t)
NVSHMEM_TYPE_FETCH_XOR_NOT_IMPLEMENTED(uint64, uint64_t)
