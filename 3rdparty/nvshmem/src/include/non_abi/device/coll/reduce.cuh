/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef REDUCE_DEVICE_CUH
#define REDUCE_DEVICE_CUH

#include <cuda_runtime.h>
#include "non_abi/device/pt-to-pt/proxy_device.cuh"
#include "non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh"
#include "non_abi/device/common/nvshmemi_common_device.cuh"
#include "non_abi/device/team/nvshmemi_team_defines.cuh"
#include "non_abi/nvshmem_build_options.h"
#ifdef NVSHMEM_ENABLE_ALL_DEVICE_INLINING
#include "non_abi/device/pt-to-pt/transfer_device.cuh"
#else
#include "non_abi/device/pt-to-pt/nvshmemi_transfer_api.cuh"
#endif
#include "non_abi/device/wait/nvshmemi_wait_until_apis.cuh"
#include "utils.cuh"
#include "fcollect.cuh"
#include "broadcast.cuh"

#ifdef __CUDACC__
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;
#endif

#ifdef __CUDA_ARCH__

/* This is used to create custom floating type traits such fp16, fp8, etc
 * For now, using this to create compile-time checks for float/double
 */
#if not defined __CUDACC_RTC__
template <typename T>
struct is_float : std::false_type {};
template <>
struct is_float<float> : std::true_type {};

template <typename T>
struct is_double : std::false_type {};
template <>
struct is_double<double> : std::true_type {};

#else

template <typename T>
struct is_float : cuda::std::false_type {};
template <>
struct is_float<float> : cuda::std::true_type {};

template <typename T>
struct is_double : cuda::std::false_type {};
template <>
struct is_double<double> : cuda::std::true_type {};

#endif

#define GPU_BITS_COPY_THREADGROUP_DIRECT(TYPENAME, TYPE, dest, src, nelems, myIdx, groupSize) \
    do {                                                                                      \
        int i;                                                                                \
        for (i = myIdx; i < nelems; i += groupSize) {                                         \
            *((TYPE *)dest + i) = *((TYPE *)src + i);                                         \
        }                                                                                     \
    } while (0)

template <typename T, rdxn_ops_t op>
#if not defined __CUDACC_RTC__
__device__ inline typename std::enable_if<std::is_integral<T>::value, T>::type perform_gpu_rdxn(
    T op1, T op2) {
#else
__device__ inline typename cuda::std::enable_if<cuda::std::is_integral<T>::value, T>::type
perform_gpu_rdxn(T op1, T op2) {
#endif
    switch (op) {
        case RDXN_OPS_SUM:
            return op1 + op2;
        case RDXN_OPS_PROD:
            return op1 * op2;
        case RDXN_OPS_AND:
            return op1 & op2;
        case RDXN_OPS_OR:
            return op1 | op2;
        case RDXN_OPS_XOR:
            return op1 ^ op2;
        case RDXN_OPS_MIN:
            return (op1 < op2) ? op1 : op2;
        case RDXN_OPS_MAX:
            return (op1 > op2) ? op1 : op2;
        default:
            printf("Unsupported rdxn op\n");
            assert(0);
            return T();
    }
}

template <typename T, rdxn_ops_t op>
#if not defined __CUDACC_RTC__
__device__ inline typename std::enable_if<!std::is_integral<T>::value, T>::type perform_gpu_rdxn(
    T op1, T op2) {
#else
__device__ inline typename cuda::std::enable_if<!cuda::std::is_integral<T>::value, T>::type
perform_gpu_rdxn(T op1, T op2) {
#endif
    switch (op) {
        case RDXN_OPS_SUM:
            return op1 + op2;
        case RDXN_OPS_PROD:
            return op1 * op2;
        case RDXN_OPS_MIN:
            return (op1 < op2) ? op1 : op2;
        case RDXN_OPS_MAX:
            return (op1 > op2) ? op1 : op2;
        default:
            printf("Unsupported rdxn op\n");
            assert(0);
            return T();
    }
}

template <>
__device__ inline double2 perform_gpu_rdxn<double2, RDXN_OPS_MAXLOC>(double2 op1, double2 op2) {
    return (op1.x > op2.x) ? op1 : op2;
}

template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
static inline __device__ void gpu_linear_reduce_threadgroup(TYPE *x, TYPE *y, TYPE *z,
                                                            size_t nelems) {
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    int i;
    for (i = myIdx; i < nelems; i += groupSize) {
        z[i] = perform_gpu_rdxn<TYPE, OP>(x[i], y[i]);
    }
}

#define float_ZERO 0.0f
#define NVSHMEMI_MCAST_PTX_REG_TYPE_u32 "r"
#define NVSHMEMI_MCAST_PTX_REG_TYPE_b32 "r"
#define NVSHMEMI_MCAST_PTX_REG_TYPE_s32 "r"
#define NVSHMEMI_MCAST_PTX_REG_TYPE_f32 "f"
#define NVSHMEMI_MCAST_PTX_REG_TYPE_u64 "l"
#define NVSHMEMI_MCAST_PTX_REG_TYPE_b64 "l"
#define NVSHMEMI_MCAST_PTX_REG_TYPE_s64 "l"
#define NVSHMEMI_MCAST_PTX_REG_TYPE_f64 "d"
#define NVSHMEMI_MCAST_PTX_REG_TYPE_f16x2 "f"

// mcast ldreduce+multimem.st of 16B
// The requirement to use these primitives is that nelems % UNROLL == 0
template <typename TYPE, threadgroup_t SCOPE, int UNROLL>
__device__ inline void nvshmemi_f32_add_reduce_mcast16_v4_threadgroup(int4 *dest,
                                                                      const int4 *source,
                                                                      size_t nelems) {
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    for (size_t j = myIdx * UNROLL; j < nelems; j += groupSize * UNROLL) {
        uint32_t u4[4 * UNROLL];
#pragma unroll UNROLL
        for (int u = 0; u < UNROLL; u++) {
            asm("multimem.ld_reduce.global.add.v4.f32 {%0, %1, %2, %3}, [%4];"
                : "=r"(u4[4 * u]), "=r"(u4[4 * u + 1]), "=r"(u4[4 * u + 2]), "=r"(u4[4 * u + 3])
                : "l"(source + j + u));
        }
#pragma unroll UNROLL
        for (int u = 0; u < UNROLL; u++) {
            asm("multimem.st.global.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(dest + j + u),
                "r"(u4[4 * u]), "r"(u4[4 * u + 1]), "r"(u4[4 * u + 2]), "r"(u4[4 * u + 3]));
        }
    }
}

// mcast ldreduce+multimem.st of 8B
#define NVSHMEMI_MCAST8_ALLREDUCE_THREADGROUP_SUM_V2(CXX_TYPE, PTX_TYPE)           \
    template <typename TYPE, threadgroup_t SCOPE>                                  \
    __device__ inline void nvshmemi_##PTX_TYPE##_add_reduce_mcast8_v2_threadgroup( \
        uint64_t *dest, const uint64_t *source, size_t nelems) {                   \
        int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();                    \
        int groupSize = nvshmemi_threadgroup_size<SCOPE>();                        \
        for (size_t j = myIdx; j < nelems; j += groupSize) {                       \
            CXX_TYPE val1[2];                                                      \
            asm("multimem.ld_reduce.global.add.v2." #PTX_TYPE " {%0, %1}, [%2];"   \
                : "=" NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1[0]),             \
                  "=" NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1[1])              \
                : "l"(source + j));                                                \
            asm("multimem.st.global.v2.f32 [%0], {%1, %2};" ::"l"(dest + j),       \
                NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1[0]),                   \
                NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1[1]));                  \
        }                                                                          \
    }

// mcast ldreduce+multimem.st of 4B
#define NVSHMEMI_MCAST4_ALLREDUCE_THREADGROUP(OP, CXX_TYPE, PTX_TYPE)              \
    template <typename TYPE, threadgroup_t SCOPE>                                  \
    __device__ inline void nvshmemi_##PTX_TYPE##_##OP##_reduce_mcast4_threadgroup( \
        uint32_t *dest, const uint32_t *source, size_t nelems) {                   \
        int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();                    \
        int groupSize = nvshmemi_threadgroup_size<SCOPE>();                        \
        for (size_t j = myIdx; j < nelems; j += groupSize) {                       \
            CXX_TYPE val1;                                                         \
            asm("multimem.ld_reduce.global." #OP "." #PTX_TYPE " %0, [%1];"        \
                : "=" NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1)                 \
                : "l"(source + j));                                                \
            asm("multimem.st.global.f32 [%0], %1;" ::"l"(dest + j),                \
                NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1));                     \
        }                                                                          \
    }

// mcast ldreduce+st of 16B
// The requirement to use these primitives is that nelems % UNROLL == 0
template <typename TYPE, threadgroup_t SCOPE, int UNROLL>
__device__ inline void nvshmemi_f32_add_local_reduce_mcast16_v4_threadgroup(int4 *dest,
                                                                            const int4 *source,
                                                                            size_t nelems) {
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    for (size_t j = myIdx * UNROLL; j < nelems; j += groupSize * UNROLL) {
        union {
            uint32_t u4[4 * UNROLL];
            uint64_t u8[2 * UNROLL];
        };
#pragma unroll UNROLL
        for (int u = 0; u < UNROLL; u++) {
            asm("multimem.ld_reduce.global.add.v4.f32 {%0, %1, %2, %3}, [%4];"
                : "=r"(u4[4 * u]), "=r"(u4[4 * u + 1]), "=r"(u4[4 * u + 2]), "=r"(u4[4 * u + 3])
                : "l"(source + j + u));
        }
#pragma unroll UNROLL
        for (int u = 0; u < UNROLL; u++) {
            asm("st.global.v2.b64 [%0], {%1, %2};" ::"l"(dest + j + u), "l"(u8[2 * u]),
                "l"(u8[2 * u + 1]));
        }
    }
}

// The requirement to use these primitives is that nelems % UNROLL == 0
#define NVSHMEMI_MCAST16_REDUCEOP_THREADGROUP_MINMAX_V4(OP, CXX_TYPE, PTX_TYPE)                 \
    template <typename TYPE, threadgroup_t SCOPE, int UNROLL>                                   \
    __device__ inline void nvshmemi_##PTX_TYPE##_##OP##_local_reduce_mcast16_v4_threadgroup(    \
        int4 *dest, const int4 *source, size_t nelems) {                                        \
        int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();                                 \
        int groupSize = nvshmemi_threadgroup_size<SCOPE>();                                     \
        for (size_t j = myIdx * UNROLL; j < nelems; j += groupSize * UNROLL) {                  \
            union {                                                                             \
                uint32_t u4[4 * UNROLL];                                                        \
                uint64_t u8[2];                                                                 \
            };                                                                                  \
            _Pragma("unroll UNROLL") for (int u = 0; u < UNROLL; u++) {                         \
                asm("multimem.ld_reduce.global." #OP ".v4." #PTX_TYPE                           \
                    ""                                                                          \
                    " {%0, %1, %2, %3}, [%4];"                                                  \
                    : "=r"(u4[4 * u + 0]), "=r"(u4[4 * u + 1]), "=r"(u4[4 * u + 2]),            \
                      "=r"(u4[4 * u + 3])                                                       \
                    : "l"(source + j + u));                                                     \
            }                                                                                   \
            _Pragma("unroll UNROLL") for (int u = 0; u < UNROLL; u++) {                         \
                asm("st.global.v2.b64 [%0], {%1, %2};" ::"l"(dest + j + u), "l"(u8[2 * u + 0]), \
                    "l"(u8[2 * u + 1]));                                                        \
            }                                                                                   \
        }                                                                                       \
    }

// mcast ldreduce+st of 8B
#define NVSHMEMI_MCAST8_REDUCEOP_THREADGROUP_MINMAX(OP, CXX_TYPE, PTX_TYPE)              \
    template <typename TYPE, threadgroup_t SCOPE>                                        \
    __device__ inline void nvshmemi_##PTX_TYPE##_##OP##_local_reduce_mcast8_threadgroup( \
        uint64_t *dest, const uint64_t *source, size_t nelems) {                         \
        int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();                          \
        int groupSize = nvshmemi_threadgroup_size<SCOPE>();                              \
        for (size_t j = myIdx; j < nelems; j += groupSize) {                             \
            CXX_TYPE val1;                                                               \
            asm("multimem.ld_reduce.global." #OP "." #PTX_TYPE " %0, [%1];"              \
                : "=" NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1)                       \
                : "l"(source + j));                                                      \
            asm("st.global.b64 [%0], %1;" ::"l"(dest + j),                               \
                NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1));                           \
        }                                                                                \
    }

#define NVSHMEMI_MCAST8_REDUCEOP_THREADGROUP_SUM(CXX_TYPE, PTX_TYPE)                  \
    template <typename TYPE, threadgroup_t SCOPE>                                     \
    __device__ inline void nvshmemi_##PTX_TYPE##_add_local_reduce_mcast8_threadgroup( \
        uint64_t *dest, const uint64_t *source, size_t nelems) {                      \
        int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();                       \
        int groupSize = nvshmemi_threadgroup_size<SCOPE>();                           \
        for (size_t j = myIdx; j < nelems; j += groupSize) {                          \
            CXX_TYPE val1;                                                            \
            asm("multimem.ld_reduce.global.add." #PTX_TYPE " %0, [%1];"               \
                : "=" NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1)                    \
                : "l"(source + j));                                                   \
            asm("st.global.b64 [%0], %1;" ::"l"(dest + j),                            \
                NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1));                        \
        }                                                                             \
    }

#define NVSHMEMI_MCAST8_REDUCEOP_THREADGROUP(OP, CXX_TYPE, PTX_TYPE)                     \
    template <typename TYPE, threadgroup_t SCOPE>                                        \
    __device__ inline void nvshmemi_##PTX_TYPE##_##OP##_local_reduce_mcast8_threadgroup( \
        uint64_t *dest, const uint64_t *source, size_t nelems) {                         \
        int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();                          \
        int groupSize = nvshmemi_threadgroup_size<SCOPE>();                              \
        for (size_t j = myIdx; j < nelems; j += groupSize) {                             \
            CXX_TYPE val1;                                                               \
            asm("multimem.ld_reduce.global." #OP "." #PTX_TYPE " %0, [%1];"              \
                : "=" NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1)                       \
                : "l"(source + j));                                                      \
            asm("st.global.b64 [%0], %1;" ::"l"(dest + j),                               \
                NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1));                           \
        }                                                                                \
    }

#define NVSHMEMI_MCAST8_REDUCEOP_THREADGROUP_SUM_V2(CXX_TYPE, PTX_TYPE)                  \
    template <typename TYPE, threadgroup_t SCOPE>                                        \
    __device__ inline void nvshmemi_##PTX_TYPE##_add_local_reduce_mcast8_v2_threadgroup( \
        uint64_t *dest, const uint64_t *source, size_t nelems) {                         \
        int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();                          \
        int groupSize = nvshmemi_threadgroup_size<SCOPE>();                              \
        for (size_t j = myIdx; j < nelems; j += groupSize) {                             \
            CXX_TYPE val1[2];                                                            \
            asm("multimem.ld_reduce.global.add.v2." #PTX_TYPE " {%0, %1}, [%2];"         \
                : "=" NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1[0]),                   \
                  "=" NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1[1])                    \
                : "l"(source + j));                                                      \
            asm("st.global.v2.b32 [%0], {%1, %2};" ::"l"(dest + j),                      \
                NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1[0]),                         \
                NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1[1]));                        \
        }                                                                                \
    }

// mcast ldreduce+st of 4B
#define NVSHMEMI_MCAST4_REDUCEOP_THREADGROUP(OP, CXX_TYPE, PTX_TYPE)                     \
    template <typename TYPE, threadgroup_t SCOPE>                                        \
    __device__ inline void nvshmemi_##PTX_TYPE##_##OP##_local_reduce_mcast4_threadgroup( \
        uint32_t *dest, const uint32_t *source, size_t nelems) {                         \
        int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();                          \
        int groupSize = nvshmemi_threadgroup_size<SCOPE>();                              \
        for (size_t j = myIdx; j < nelems; j += groupSize) {                             \
            CXX_TYPE val1;                                                               \
            asm("multimem.ld_reduce.global." #OP "." #PTX_TYPE " %0, [%1];"              \
                : "=" NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1)                       \
                : "l"(source + j));                                                      \
            asm("st.global.b32 [%0], %1;" ::"l"(dest + j),                               \
                NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1));                           \
        }                                                                                \
    }

#define NVSHMEMI_MCAST_RDXN_OP_IS_CAP(OP) \
    (OP != RDXN_OPS_PROD && OP != RDXN_OPS_MAXLOC && OP != RDXN_OPS_sentinel)
#define NVSHMEMI_MCAST_RDXN_OP_IS_CAP_8B(OP) (NVSHMEMI_MCAST_RDXN_OP_IS_CAP(OP))
#define NVSHMEMI_MCAST_RDXN_OP_IS_CAP_16B(OP) \
    (OP == RDXN_OPS_SUM || OP == RDXN_OPS_MIN || OP == RDXN_OPS_MAX)
#define NVSHMEMI_MCAST_RDXN_OP_IS_CAP_UNTYPED(OP) \
    (OP == RDXN_OPS_SUM || OP == RDXN_OPS_AND || OP == RDXN_OPS_OR || OP == RDXN_OPS_XOR)

/* Supported matrix here:
 * https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-multimem-ld-reduce-multimem-st-multimem-red
 */

/* 4B reduce + copy primitives */
NVSHMEMI_MCAST4_REDUCEOP_THREADGROUP(add, uint32_t, u32)
NVSHMEMI_MCAST4_REDUCEOP_THREADGROUP(add, int32_t, s32)
NVSHMEMI_MCAST4_REDUCEOP_THREADGROUP(add, float, f32)
NVSHMEMI_MCAST4_REDUCEOP_THREADGROUP(min, uint32_t, u32)
NVSHMEMI_MCAST4_REDUCEOP_THREADGROUP(min, int32_t, s32)
NVSHMEMI_MCAST4_REDUCEOP_THREADGROUP(max, uint32_t, u32)
NVSHMEMI_MCAST4_REDUCEOP_THREADGROUP(max, int32_t, s32)
NVSHMEMI_MCAST4_REDUCEOP_THREADGROUP(and, uint32_t, b32)
NVSHMEMI_MCAST4_REDUCEOP_THREADGROUP(xor, uint32_t, b32)
NVSHMEMI_MCAST4_REDUCEOP_THREADGROUP(or, uint32_t, b32)

/* 8B reduce + copy primitives */
NVSHMEMI_MCAST8_REDUCEOP_THREADGROUP_SUM(uint64_t, u64)
NVSHMEMI_MCAST8_REDUCEOP_THREADGROUP_SUM(double, f64)
NVSHMEMI_MCAST8_REDUCEOP_THREADGROUP_SUM_V2(float, f32)
NVSHMEMI_MCAST8_REDUCEOP_THREADGROUP_MINMAX(min, uint64_t, u64)
NVSHMEMI_MCAST8_REDUCEOP_THREADGROUP_MINMAX(min, int64_t, s64)
NVSHMEMI_MCAST8_REDUCEOP_THREADGROUP_MINMAX(max, uint64_t, u64)
NVSHMEMI_MCAST8_REDUCEOP_THREADGROUP_MINMAX(max, int64_t, s64)
NVSHMEMI_MCAST8_REDUCEOP_THREADGROUP(and, uint64_t, b64)
NVSHMEMI_MCAST8_REDUCEOP_THREADGROUP(xor, uint64_t, b64)
NVSHMEMI_MCAST8_REDUCEOP_THREADGROUP(or, uint64_t, b64)

/* 16B reduce + copy primitives */
NVSHMEMI_MCAST16_REDUCEOP_THREADGROUP_MINMAX_V4(min, uint32_t, f16x2)
NVSHMEMI_MCAST16_REDUCEOP_THREADGROUP_MINMAX_V4(max, uint32_t, f16x2)

/* 16B allreduce(SUM) primitives */
NVSHMEMI_MCAST8_ALLREDUCE_THREADGROUP_SUM_V2(float, f32)
NVSHMEMI_MCAST4_ALLREDUCE_THREADGROUP(add, float, f32)

template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
__device__ inline int nvshmemi_local_reduce_mcast_threadgroup(TYPE *__restrict__ dest,
                                                              const TYPE *__restrict__ src,
                                                              size_t nreduce) {
    constexpr bool is_unsigned = std::is_integral<TYPE>::value && std::is_unsigned<TYPE>::value;
    constexpr bool is_signed = std::is_integral<TYPE>::value && std::is_signed<TYPE>::value;
    constexpr bool is_float_v = is_float<TYPE>::value;
    constexpr bool is_double_v = is_double<TYPE>::value;
    size_t len = nreduce * sizeof(TYPE);

    if ((uintptr_t)dest % sizeof(int4) == 0 && (uintptr_t)src % sizeof(int4) == 0 &&
        len >= sizeof(int4) && NVSHMEMI_MCAST_RDXN_OP_IS_CAP_16B(OP)) {
        const size_t nelems = len / sizeof(int4);
        int4 *__restrict__ dst_p = (int4 *)dest;
        const int4 *__restrict__ src_p = (const int4 *)src;

        switch (OP) {
            case RDXN_OPS_SUM:
                if (is_unsigned || is_signed || is_float_v) {
                    if (len >= 64 && len % 64 == 0)
                        nvshmemi_f32_add_local_reduce_mcast16_v4_threadgroup<TYPE, SCOPE, 4>(
                            dst_p, src_p, nelems);
                    else
                        nvshmemi_f32_add_local_reduce_mcast16_v4_threadgroup<TYPE, SCOPE, 1>(
                            dst_p, src_p, nelems);
                } else if (is_double_v)
                    goto use_8B_aligned;  // double doesn't support vec for multimem, so fallback to
                                          // non-vec double multimem
                break;
            case RDXN_OPS_MIN:
                if (len >= 64 && len % 64 == 0)
                    nvshmemi_f16x2_min_local_reduce_mcast16_v4_threadgroup<TYPE, SCOPE, 4>(
                        dst_p, src_p, nelems);
                else
                    nvshmemi_f16x2_min_local_reduce_mcast16_v4_threadgroup<TYPE, SCOPE, 1>(
                        dst_p, src_p, nelems);
                break;
            case RDXN_OPS_MAX:
                if (len >= 64 && len % 64 == 0)
                    nvshmemi_f16x2_max_local_reduce_mcast16_v4_threadgroup<TYPE, SCOPE, 4>(
                        dst_p, src_p, nelems);
                else
                    nvshmemi_f16x2_max_local_reduce_mcast16_v4_threadgroup<TYPE, SCOPE, 1>(
                        dst_p, src_p, nelems);
                break;
            default:
                break;
        }

        len -= nelems * sizeof(int4);
        if (0 == len) return 0;
        dest = (TYPE *)(dst_p + nelems);
        src = (TYPE *)(src_p + nelems);
    }

use_8B_aligned:
    if ((uintptr_t)dest % sizeof(uint64_t) == 0 && (uintptr_t)src % sizeof(uint64_t) == 0 &&
        len >= sizeof(uint64_t) && NVSHMEMI_MCAST_RDXN_OP_IS_CAP_8B(OP)) {
        const size_t nelems = len / sizeof(uint64_t);
        uint64_t *__restrict__ dst_p = (uint64_t *)dest;
        const uint64_t *__restrict__ src_p = (const uint64_t *)src;
        switch (OP) {
            case RDXN_OPS_SUM:
                if (is_unsigned || is_signed)
                    nvshmemi_u64_add_local_reduce_mcast8_threadgroup<TYPE, SCOPE>(dst_p, src_p,
                                                                                  nelems);
                else if (is_float_v)
                    nvshmemi_f32_add_local_reduce_mcast8_v2_threadgroup<TYPE, SCOPE>(dst_p, src_p,
                                                                                     nelems);
                else if (is_double_v)
                    nvshmemi_f64_add_local_reduce_mcast8_threadgroup<TYPE, SCOPE>(dst_p, src_p,
                                                                                  nelems);
                break;
            case RDXN_OPS_MIN:
                if (is_unsigned)
                    nvshmemi_u64_min_local_reduce_mcast8_threadgroup<TYPE, SCOPE>(dst_p, src_p,
                                                                                  nelems);
                else if (is_signed)
                    nvshmemi_s64_min_local_reduce_mcast8_threadgroup<TYPE, SCOPE>(dst_p, src_p,
                                                                                  nelems);
                break;
            case RDXN_OPS_MAX:
                if (is_unsigned)
                    nvshmemi_u64_max_local_reduce_mcast8_threadgroup<TYPE, SCOPE>(dst_p, src_p,
                                                                                  nelems);
                else if (is_signed)
                    nvshmemi_s64_max_local_reduce_mcast8_threadgroup<TYPE, SCOPE>(dst_p, src_p,
                                                                                  nelems);
                break;
            case RDXN_OPS_AND:
                nvshmemi_b64_and_local_reduce_mcast8_threadgroup<TYPE, SCOPE>(dst_p, src_p, nelems);
                break;
            case RDXN_OPS_XOR:
                nvshmemi_b64_xor_local_reduce_mcast8_threadgroup<TYPE, SCOPE>(dst_p, src_p, nelems);
                break;
            case RDXN_OPS_OR:
                nvshmemi_b64_or_local_reduce_mcast8_threadgroup<TYPE, SCOPE>(dst_p, src_p, nelems);
                break;
            default:
                break;
        }

        len -= nelems * sizeof(uint64_t);
        if (0 == len) return 0;
        dest = (TYPE *)(dst_p + nelems);
        src = (TYPE *)(src_p + nelems);
    }

    if ((uintptr_t)dest % sizeof(uint32_t) == 0 && (uintptr_t)src % sizeof(uint32_t) == 0 &&
        len >= sizeof(uint32_t)) {
        const size_t nelems = len / sizeof(uint32_t);
        uint32_t *__restrict__ dst_p = (uint32_t *)dest;
        const uint32_t *__restrict__ src_p = (const uint32_t *)src;
        switch (OP) {
            case RDXN_OPS_SUM:
                if (is_unsigned)
                    nvshmemi_u32_add_local_reduce_mcast4_threadgroup<TYPE, SCOPE>(dst_p, src_p,
                                                                                  nelems);
                else if (is_signed)
                    nvshmemi_s32_add_local_reduce_mcast4_threadgroup<TYPE, SCOPE>(dst_p, src_p,
                                                                                  nelems);
                else if (is_float_v)
                    nvshmemi_f32_add_local_reduce_mcast4_threadgroup<TYPE, SCOPE>(dst_p, src_p,
                                                                                  nelems);
                break;
            case RDXN_OPS_MIN:
                if (is_unsigned)
                    nvshmemi_u32_min_local_reduce_mcast4_threadgroup<TYPE, SCOPE>(dst_p, src_p,
                                                                                  nelems);
                else if (is_signed)
                    nvshmemi_s32_min_local_reduce_mcast4_threadgroup<TYPE, SCOPE>(dst_p, src_p,
                                                                                  nelems);
                break;
            case RDXN_OPS_MAX:
                if (is_unsigned)
                    nvshmemi_u32_max_local_reduce_mcast4_threadgroup<TYPE, SCOPE>(dst_p, src_p,
                                                                                  nelems);
                else if (is_signed)
                    nvshmemi_s32_max_local_reduce_mcast4_threadgroup<TYPE, SCOPE>(dst_p, src_p,
                                                                                  nelems);
                break;
            case RDXN_OPS_XOR:
                nvshmemi_b32_xor_local_reduce_mcast4_threadgroup<TYPE, SCOPE>(dst_p, src_p, nelems);
                break;
            case RDXN_OPS_AND:
                nvshmemi_b32_and_local_reduce_mcast4_threadgroup<TYPE, SCOPE>(dst_p, src_p, nelems);
                break;
            case RDXN_OPS_OR:
                nvshmemi_b32_or_local_reduce_mcast4_threadgroup<TYPE, SCOPE>(dst_p, src_p, nelems);
                break;
            default:
                break;
        }

        len -= nelems * sizeof(uint32_t);
        if (0 == len) return 0;
    }

    /* Return the remainder length, incase the caller wants to retry with unicast */
    return (len);
}

template <typename TYPE, rdxn_ops_t OP>
__device__ static inline void gpu_rdxn_on_demand_2(int start, int stride, int size, TYPE *dest,
                                                   const TYPE *source, size_t nelems, TYPE *pWrk,
                                                   volatile long *pSync,
                                                   volatile long *sync_counter) {
    int next_rank = -1;
    TYPE *op1 = NULL, *op2 = NULL;
    size_t i;
    volatile TYPE *tmp_operand;
    int my_active_set_pe = ((nvshmemi_device_state_d.mype - start) / stride);
    tmp_operand = (TYPE *)pWrk;
    nvshmemi_put_threadgroup<TYPE, NVSHMEMI_THREADGROUP_THREAD>(dest, source, nelems,
                                                                nvshmemi_device_state_d.mype);
    for (i = 1; i < size; i++) {
        next_rank = start + ((my_active_set_pe + i) % size) * stride;
        nvshmemi_put_nbi_threadgroup<TYPE, NVSHMEMI_THREADGROUP_THREAD>((TYPE *)tmp_operand, source,
                                                                        nelems, next_rank);
        nvshmemi_quiet<NVSHMEMI_THREADGROUP_THREAD>();
        sync_dissem_threadgroup_2<NVSHMEMI_THREADGROUP_THREAD>(start, stride, size, pSync,
                                                               sync_counter);
        op1 = (TYPE *)dest;
        op2 = (TYPE *)tmp_operand;
        gpu_linear_reduce_threadgroup<TYPE, OP, NVSHMEMI_THREADGROUP_THREAD>(op1, op2, op1, nelems);
        sync_dissem_threadgroup_2<NVSHMEMI_THREADGROUP_THREAD>(start, stride, size, pSync,
                                                               sync_counter);
    }
}

template <typename TYPE, rdxn_ops_t OP>
__device__ static inline void gpu_rdxn_on_demand(nvshmem_team_t team, TYPE *dest,
                                                 const TYPE *source, size_t nelems) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int start = teami->start;
    int stride = teami->stride;
    int size = teami->size;
    TYPE *pWrk = (TYPE *)nvshmemi_team_get_psync(teami, REDUCE);
    volatile long *pSync = nvshmemi_team_get_psync(teami, SYNC);
    volatile long *sync_counter = nvshmemi_team_get_sync_counter(teami);

    gpu_rdxn_on_demand_2<TYPE, OP>(start, stride, size, dest, source, nelems, pWrk, pSync,
                                   sync_counter);
}

/* pWrk usage - (k - 1) * nreduce for step 1
              - k * step2_nphases * nreduce for receiving step 2 data
              - step2_nphases * nreduce for sending data of each phase */
template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
__device__ static inline void gpu_rdxn_recexch_threadgroup(nvshmem_team_t team, TYPE *dst,
                                                           const TYPE *source, size_t nreduce) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    TYPE *pWrk = (TYPE *)nvshmemi_team_get_psync(teami, REDUCE);
    volatile long *pSync = (volatile long *)nvshmemi_team_get_psync(teami, SYNC);
    volatile long *sync_counter = (volatile long *)nvshmemi_team_get_sync_counter(teami);
    const int step1_sendto = teami->reduce_recexch.step1_sendto;
    const int step1_nrecvs = teami->reduce_recexch.step1_nrecvs;
    const int *step1_recvfrom = teami->reduce_recexch.step1_recvfrom;
    const int step2_nphases = teami->reduce_recexch.step2_nphases;
    int **step2_nbrs = teami->reduce_recexch.step2_nbrs;
    const int rank = nvshmemi_device_state_d.mype;
    const int k = nvshmemi_device_state_d.gpu_coll_env_params_var.reduce_recexch_kval;

    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    int in_step2 = (step1_sendto == -1); /* whether this rank participates in Step 2 */

    if (in_step2 == 1) {
        for (size_t i = myIdx; i < nreduce; i += groupSize) {
            dst[i] = source[i];
        }
        nvshmemi_threadgroup_sync<SCOPE>();
    }

    if (in_step2 == 0) {
        size_t offset = (step1_sendto - rank - 1) * nreduce;
        nvshmemi_put_nbi_threadgroup<TYPE, SCOPE>(pWrk + offset, source, nreduce, step1_sendto);
        if (!myIdx) {
            nvshmemi_fence();
            nvshmemi_signal_for_barrier<long>((long *)(pSync + rank), sync_counter[0],
                                              step1_sendto);
        }
    } else if (step1_nrecvs != 0) {
        for (int i = 0; i < step1_nrecvs; i += 1) {
            nvshmemi_wait_until<long>((long *)pSync + step1_recvfrom[i], NVSHMEM_CMP_GE,
                                      sync_counter[0]);
            size_t offset = (rank - step1_recvfrom[i] - 1) * nreduce;
            gpu_linear_reduce_threadgroup<TYPE, OP, SCOPE>(dst, (pWrk + offset), dst, nreduce);
        }
        nvshmemi_threadgroup_sync<SCOPE>();
    }

    /* Step 2 */
    if (in_step2) {
        size_t send_offset = (k - 1) * nreduce + k * step2_nphases * nreduce;
        size_t recv_offset = (k - 1) * nreduce;
        for (int phase = 0; phase < step2_nphases; phase++) {
            int num_small = k - 1;
            for (int i = 0; i < k - 1; i++) {
                if (step2_nbrs[phase][i] > rank) {
                    num_small = i;
                    break;
                }
            }
            /* copy the data to end of pWrk that can be used as source for puts
                while we use dst for reduction */
            for (size_t i = myIdx; i < nreduce; i += groupSize) {
                pWrk[send_offset + phase * nreduce + i] = dst[i];
            }
            nvshmemi_threadgroup_sync<SCOPE>();
            for (int i = 0; i < k - 1; i++) {
                size_t offset = recv_offset + k * phase * nreduce + num_small * nreduce;
                nvshmemi_put_nbi_threadgroup<TYPE, SCOPE>(pWrk + offset,
                                                          pWrk + send_offset + phase * nreduce,
                                                          nreduce, step2_nbrs[phase][i]);
            }
            if (!myIdx) nvshmemi_fence();
            nvshmemi_threadgroup_sync<SCOPE>();
            for (int i = myIdx; i < k - 1; i += groupSize) {
                nvshmemi_signal_for_barrier<long>((long *)(pSync + rank), sync_counter[0],
                                                  step2_nbrs[phase][i]);
            }

            for (int i = 0; i < k - 1; i += 1) {
                nvshmemi_wait_until<uint64_t>((uint64_t *)(pSync + step2_nbrs[phase][i]),
                                              NVSHMEM_CMP_GE, sync_counter[0]);
                int offset = recv_offset + k * phase * nreduce;
                if (step2_nbrs[phase][i] < rank)
                    offset += i * nreduce;
                else
                    offset += (i + 1) * nreduce;
                gpu_linear_reduce_threadgroup<TYPE, OP, SCOPE>(dst, (pWrk + offset), dst, nreduce);
            }
            /*nvshmem_quiet(); */ /*wait for my puts to complete */
        }
    }

    /* Step 3 */
    if (step1_nrecvs > 0) {
        for (int i = 0; i < step1_nrecvs; i++) {
            nvshmemi_put_nbi_threadgroup<TYPE, SCOPE>(dst, dst, nreduce, step1_recvfrom[i]);
        }
        if (!myIdx) nvshmemi_fence();
        nvshmemi_threadgroup_sync<SCOPE>();
        for (int i = myIdx; i < step1_nrecvs; i += groupSize) {
            nvshmemi_signal_for_barrier<long>((long *)(pSync + rank), sync_counter[0],
                                              step1_recvfrom[i]);
        }
    } else if (step1_sendto != -1) {
        if (!myIdx)
            nvshmemi_wait_until<uint64_t>((uint64_t *)(pSync + step1_sendto), NVSHMEM_CMP_GE,
                                          sync_counter[0]);
    }
    nvshmemi_threadgroup_sync<SCOPE>();
    if (!myIdx) sync_counter[0] += 1;
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <typename T, rdxn_ops_t OP, threadgroup_t SCOPE>
static inline __device__ void gpu_head_check_op_threadgroup(T *dest, T *src, T *actual_src,
                                                            size_t nelems) {
    int i, k;
    int subelems = sizeof(TYPE) / sizeof(uint32_t);
    volatile uint32_t *header = NULL;
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>;
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    TYPE tmp;
    uint32_t *tmp_ptr = (uint32_t *)&tmp;
    uint32_t *payload = NULL;
    for (i = myIdx; i < nelems; i += groupSize) {
        for (k = 0; k < subelems; k++) {
            payload = (uint32_t *)((uint64_t *)src + (i * subelems) + k);
            header = (uint32_t *)payload + 1;
            while (1 != *header)
                ;
            *header = 0;
            *(tmp_ptr + k) = *payload;
        }
        dest[i] = perform_gpu_rdxn<T, OP>(actual_src[i], tmp);
    }
}

template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
static inline __device__ void gpu_head_checkall_op_threadgroup(nvshmem_team_t team, TYPE *dest,
                                                               TYPE *src, TYPE *actual_src,
                                                               size_t nelems) {
    int i, j, k;
    int subelems = sizeof(TYPE) / sizeof(uint32_t);
    volatile uint32_t *header = NULL;
    TYPE tmp;
    uint32_t *tmp_ptr = (uint32_t *)&tmp;
    uint32_t *payload = NULL;
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int start = teami->start;
    int stride = teami->stride;
    int size = teami->size;
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    int my_active_set_pe = ((nvshmemi_device_state_d.mype - start) / stride);

    for (j = (my_active_set_pe - 1); j >= 0; j--) {
        for (i = myIdx; i < nelems; i += groupSize) {
            for (k = 0; k < subelems; k++) {
                payload =
                    (uint32_t *)((uint64_t *)src + (i * subelems) + k + (nelems * subelems * j));
                header = (uint32_t *)payload + 1;
                while (1 != *header)
                    ;
                *header = 0;
                *(tmp_ptr + k) = *payload;
            }
            dest[i] = perform_gpu_rdxn<TYPE, OP>(src_ptr[i], tmp);
        }
        src_ptr = dest;
    }
    for (j = size - 1; j > my_active_set_pe; j--) {
        for (i = myIdx; i < nelems; i += groupSize) {
            for (k = 0; k < subelems; k++) {
                payload =
                    (uint32_t *)((uint64_t *)src + (i * subelems) + k + (nelems * subelems * j));
                header = (uint32_t *)payload + 1;
                while (1 != *header)
                    ;
                *header = 0;
                *(tmp_ptr + k) = *payload;
            }
            dest[i] = perform_gpu_rdxn<TYPE, OP>(src_ptr[i], tmp);
        }
        src_ptr = dest;
    }
}

template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
static inline __device__ void gpu_linear_reduce_threadgroup_p2p_get(nvshmem_team_t team, TYPE *x,
                                                                    TYPE *y, int next_rank, TYPE *z,
                                                                    size_t nelems) {
    int i;
    int group_nelems = ((nelems / groupSize) * groupSize);
    int excess = nelems - group_nelems;
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int start = teami->start;
    int stride = teami->stride;
    int size = teami->size;
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    TYPE *pWrk = (TYPE *)nvshmemi_team_get_psync(teami, REDUCE);
    for (i = myIdx; i < group_nelems; i += groupSize) {
        nvshmemi_get_threadgroup<TYPE, NVSHMEMI_THREADGROUP_THREAD>(
            (TYPE *)((TYPE *)pWrk + myIdx), (TYPE *)((TYPE *)y + i), 1, next_rank);
        nvshmemi_barrier_threadgroup<SCOPE>(team);
        z[i] = perform_gpu_rdxn<TYPE, OP>(x[i], pWrk[myId]);
    }
    if (excess) {
        if (i < nelems) {
            nvshmemi_get_threadgroup<TYPE, NVSHMEMI_THREADGROUP_THREAD>(
                (TYPE *)((TYPE *)pWrk + myIdx), (TYPE *)((TYPE *)y + i), 1, next_rank);
        }
        nvshmemi_barrier_threadgroup<SCOPE>(team);
        if (i < nelems) {
            z[i] = perform_gpu_rdxn<TYPE, OP>(x[i], pWrk[myIdx]);
        }
    }
}

template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
static inline __device__ void gpu_linear_reduce_threadgroup_p2p_put(TYPE *x, TYPE *y, int next_rank,
                                                                    TYPE *z, size_t offset,
                                                                    size_t nelems) {
    int i;
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int start = teami->start;
    int stride = teami->stride;
    int size = teami->size;
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    TYPE *pWrk = (TYPE *)nvshmemi_team_get_psync(teami, REDUCE);
    int group_nelems = ((nelems / groupSize) * groupSize);
    int excess = nelems - group_nelems;
    for (i = myIdx; i < group_nelems; i += groupSize) {
        nvshmemi_put_nbi_threadgroup<TYPE, NVSHMEMI_THREADGROUP_THREAD>(
            (TYPE *)((TYPE *)pWrk + (myIdx + ((offset & 1) * groupSize))),
            (const TYPE *)((TYPE *)y + i), 1, next_rank);
        nvshmemi_barrier_threadgroup<SCOPE>(team);
        z[i] = perform_gpu_rdxn<TYPE, OP>(x[i], *(pWrk + (myIdx + ((offset & 1) * groupSize))));
    }
    offset++;
    if (excess) {
        if (i < nelems) {
            nvshmemi_put_nbi_threadgroup<TYPE, NVSHMEMI_THREADGROUP_THREAD>(
                (TYPE *)((TYPE *)pWrk + (myIdx + ((offset & 1) * groupSize))),
                (const TYPE *)((TYPE *)y + i), 1, next_rank);
        }
        nvshmemi_barrier_threadgroup<SCOPE>(team);
        if (i < nelems) {
            z[i] = perform_gpu_rdxn<TYPE, OP>(x[i], *(pWrk + (myIdx + ((offset & 1) * groupSize))));
        }
    }
    offset++;
}

template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
static inline __device__ void gpu_linear_reduce_threadgroup_p2p_put_direct(TYPE *x, TYPE *y,
                                                                           int next_rank, TYPE *z,
                                                                           size_t offset,
                                                                           size_t nelems) {
    int i;
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int start = teami->start;
    int stride = teami->stride;
    int size = teami->size;
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    TYPE *pWrk = (TYPE *)nvshmemi_team_get_psync(teami, REDUCE);
    int group_nelems = ((nelems / groupSize) * groupSize);
    int excess = nelems - group_nelems;
    for (i = myIdx; i < group_nelems; i += groupSize) {
        *((TYPE *)((TYPE *)pWrk + (myIdx + ((offset & 1) * groupSize)))) =
            *((TYPE *)((TYPE *)y + i));
        nvshmemi_barrier_threadgroup<SCOPE>(team);
        z[i] = perform_gpu_rdxn<TYPE, OP>(x[i], *(pWrk + (myIdx + ((offset & 1) * groupSize))));
    }
    offset++;
    if (excess) {
        if (i < nelems) {
            *((TYPE *)((TYPE *)pWrk + (myIdx + ((offset & 1) * groupSize)))) =
                *((TYPE *)((TYPE *)y + i));
        }
        nvshmemi_barrier_threadgroup<SCOPE>(team);
        if (i < nelems) {
            z[i] = perform_gpu_rdxn<TYPE, OP>(x[i], *(pWrk + (myIdx + ((offset & 1) * groupSize))));
        }
    }
    offset++;
}

template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
static inline __device__ void nvshmemi_gpu_rdxn_threadgroup_zcopy_get_bar_direct(
    nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nreduce) {
    int next_rank = -1;
    int src_offset = -1;
    int next_offset = -1;
    char *base = NULL;
    char *peer_base = NULL;
    char *peer_source = NULL;
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int start = teami->start;
    int stride = teami->stride;
    int size = teami->size;
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    TYPE *pWrk = (TYPE *)nvshmemi_team_get_psync(teami, REDUCE);
    int i;
    int my_active_set_pe = ((nvshmemi_device_state_d.mype - start) / stride);

    base = (char *)((void *)__ldg(
        (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p +
        nvshmemi_device_state_d.mype));
    src_offset = ((char *)source - base);

    next_rank = start + ((my_active_set_pe + 1) % size) * stride;
    next_offset = src_offset;
    peer_base = (char *)((void *)__ldg(
        (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + next_rank));
    peer_source = peer_base + next_offset;
    gpu_linear_reduce_threadgroup<TYPE, OP, SCOPE>((void *)source, peer_source, dest, nreduce);

    for (i = 2; i < size; i++) {
        next_rank = start + ((my_active_set_pe + i) % size) * stride;
        next_offset = src_offset;
        peer_base = (char *)((void *)__ldg(
            (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + next_rank));
        peer_source = peer_base + next_offset;
        gpu_linear_reduce_threadgroup<TYPE, OP, SCOPE>(dest, peer_source, dest, nreduce);
    }
    nvshmemi_barrier_threadgroup<SCOPE>(team);
}

// barrier limited
template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
static inline __device__ void nvshmemi_gpu_rdxn_threadgroup_get_bar(nvshmem_team_t team, TYPE *dest,
                                                                    const TYPE *source,
                                                                    size_t nreduce) {
    int next_rank = -1;
    size_t src_offset;
    size_t next_offset;
    char *base = NULL;
    char *peer_base = NULL;
    char *peer_source = NULL;
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int start = teami->start;
    int stride = teami->stride;
    int size = teami->size;
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    TYPE *pWrk = (TYPE *)nvshmemi_team_get_psync(teami, REDUCE);
    int i;
    int my_active_set_pe = ((nvshmemi_device_state_d.mype - start) / stride);

    next_rank = start + ((my_active_set_pe + 1) % size) * stride;
    gpu_linear_reduce_threadgroup_p2p_get<TYPE, OP, SCOPE>(team, source, source, next_rank, dest,
                                                           nreduce);
    for (i = 2; i < size; i++) {
        next_rank = start + ((my_active_set_pe + i) % size) * stride;
        gpu_linear_reduce_threadgroup_p2p_get<TYPE, OP, SCOPE>(team, dest, source, next_rank, dest,
                                                               nreduce);
    }
    nvshmemi_barrier_threadgroup<SCOPE>(team);
}

// barrier limited
template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
static inline void nvshmemi_gpu_rdxn_threadgroup_put_bar(nvshmem_team_t team, TYPE *dest,
                                                         const TYPE *source, size_t nreduce) {
    int next_rank = -1;
    int counter = 0;
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int start = teami->start;
    int stride = teami->stride;
    int size = teami->size;
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    TYPE *pWrk = (TYPE *)nvshmemi_team_get_psync(teami, REDUCE);
    int i;
    int my_active_set_pe = ((nvshmemi_device_state_d.mype - start) / stride);

    next_rank = start + ((my_active_set_pe + 1) % size) * stride;
    gpu_linear_reduce_threadgroup_p2p_put<TYPE, OP, SCOPE>(source, source, next_rank, dest, counter,
                                                           nreduce);

    for (i = 2; i < size; i++) {
        next_rank = start + ((my_active_set_pe + i) % size) * stride;
        gpu_linear_reduce_threadgroup_p2p_put<TYPE, OP, SCOPE>(dest, source, next_rank, dest,
                                                               counter, nreduce);
    }
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
static inline __device__ void gpu_rdxn_segment_threadgroup(nvshmem_team_t team, TYPE *dest,
                                                           const TYPE *source, size_t nelems) {
    int type_size = sizeof(TYPE);
    size_t msg_len = nelems * type_size;
    int next_rank = -1;
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int start = teami->start;
    int stride = teami->stride;
    int size = teami->size;
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    TYPE *pWrk = (TYPE *)nvshmemi_team_get_psync(teami, REDUCE);
    TYPE *op1 = NULL, *op2 = NULL;
    int i;
    size_t j;
    volatile TYPE *tmp_operand;
    size_t remainder = 0;
    size_t rnds_floor = 0;
    size_t offset = 0;
    int pe_offset = 0;
    int pes_per_round = 0;
    int round = 0;
    size_t exchange_size = 0;
    int my_active_set_pe = ((nvshmemi_device_state_d.mype - start) / stride);
    size_t nvshm_gpu_rdxn_seg_size =
        (nvshmemi_device_state_d.gpu_coll_env_params_var.reduce_scratch_size / 2) / sizeof(long);

    tmp_operand = (TYPE *)pWrk;
    nvshmemi_put_nbi_threadgroup<TYPE, SCOPE>((TYPE *)dest, (const TYPE *)source, nelems,
                                              nvshmemi_device_state_d.mype);

    rnds_floor = msg_len / nvshm_gpu_rdxn_seg_size;
    remainder = msg_len % nvshm_gpu_rdxn_seg_size;

    for (j = 0; j < rnds_floor; j++) {
        exchange_size = nvshm_gpu_rdxn_seg_size;
        for (i = 1; i < size; i++) {
            next_rank = start + ((my_active_set_pe + i) % size) * stride;
            nvshmemi_put_nbi_threadgroup<TYPE, SCOPE>((TYPE *)tmp_operand,
                                                      (const TYPE *)source + offset,
                                                      (exchange_size / sizeof(TYPE)), next_rank);
            nvshmemi_barrier_threadgroup<SCOPE>(team);
            op1 = (TYPE *)dest + offset;
            op2 = (TYPE *)tmp_operand;
            gpu_linear_reduce_threadgroup<TYPE, OP, SCOPE>(op1, op2, op1,
                                                           (exchange_size / sizeof(TYPE)));
            nvshmemi_sync_threadgroup<SCOPE>(team);
        }
        offset += (exchange_size / sizeof(TYPE));
    }
    if (remainder != 0) {
        exchange_size = remainder;
        pes_per_round = nvshm_gpu_rdxn_seg_size / remainder;
        pe_offset = 1;
        do {
            round = 0;
            for (i = pe_offset; ((round < pes_per_round) && (i < size)); i++) {
                next_rank = start + ((my_active_set_pe + i) % size) * stride;
                nvshmemi_put_nbi_threadgroup<TYPE, SCOPE>(
                    (TYPE *)((TYPE *)tmp_operand + (round * (exchange_size / sizeof(TYPE)))),
                    (TYPE *)source + offset, (exchange_size / sizeof(TYPE)), next_rank);
                round++;
                pe_offset++;
            }
            nvshmemi_barrier_threadgroup<SCOPE>(team);
            for (i = 0; i < round; i++) {
                op1 = (TYPE *)dest + offset;
                op2 = (TYPE *)((TYPE *)tmp_operand + (i * (exchange_size / sizeof(TYPE))));
                gpu_linear_reduce_threadgroup<TYPE, OP, SCOPE>(op1, op2, op1,
                                                               (exchange_size / sizeof(TYPE)));
            }
            nvshmemi_sync_threadgroup<SCOPE>(team);
        } while (pe_offset < size);
    }
}

template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
static inline __device__ void nvshmemi_gpu_rdxn_threadgroup_put_bar_direct(nvshmem_team_t team,
                                                                           TYPE *dest,
                                                                           const TYPE *source,
                                                                           size_t nreduce) {
    int next_rank = -1;
    size_t src_offset = -1;
    size_t next_offset = -1;
    char *base = NULL;
    char *peer_base = NULL;
    char *peer_source = NULL;
    int counter = 0;
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int start = teami->start;
    int stride = teami->stride;
    int size = teami->size;
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    TYPE *pWrk = (TYPE *)nvshmemi_team_get_psync(teami, REDUCE);
    int my_active_set_pe = ((nvshmemi_device_state_d.mype - start) / stride);
    int i;

    base = (char *)((void *)__ldg(
        (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p +
        nvshmemi_device_state_d.mype));
    src_offset = ((char *)source - base);

    next_rank = start + ((my_active_set_pe + 1) % size) * stride;
    next_offset = src_offset;
    peer_base = (char *)((void *)__ldg(
        (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + next_rank));
    peer_source = peer_base + next_offset;
    gpu_linear_reduce_threadgroup_p2p_put_direct<TYPE, OP, SCOPE>(source, peer_source, next_rank,
                                                                  dest, counter, nreduce);

    for (i = 2; i < size; i++) {
        next_rank = start + ((my_active_set_pe + i) % size) * stride;
        next_offset = src_offset;
        peer_base = (char *)((void *)__ldg(
            (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + next_rank));
        peer_source = peer_base + next_offset;
        gpu_linear_reduce_threadgroup_p2p_put_direct<TYPE, OP, SCOPE>(dest, peer_source, next_rank,
                                                                      dest, counter, nreduce);
    }
    nvshmemi_threadgroup_sync();
}

template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
static inline __device__ void nvshmemi_gpu_rdxn_threadgroup_putring(nvshmem_team_t team, TYPE *dest,
                                                                    const TYPE *source,
                                                                    size_t nelems) {
    int next_rank = -1;
    int prev_rank = -1;
    int i;
    size_t j;
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int start = teami->start;
    int stride = teami->stride;
    int size = teami->size;
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    TYPE *pWrk = (TYPE *)nvshmemi_team_get_psync(teami, REDUCE);
    int PE_end = start + (stride * size);
    uint32_t tmp[2];
    uint64_t *tmp_rdxn = pWrk;
    uint64_t *tmp_rdxn_2 = &tmp_rdxn[1];
    uint32_t payld;
    volatile uint32_t *my_notif_ptr = NULL;
    size_t subelems = sizeof(TYPE) / sizeof(uint32_t);
    tmp[1] = 1;
    next_rank =
        (nvshmemi_mype_d != (PE_end - stride)) ? (nvshmemi_device_state_d.mype + stride) : start;
    prev_rank =
        (nvshmemi_mype_d != start) ? (nvshmemi_device_state_d.mype - stride) : (PE_end - stride);
    my_notif_ptr = (uint32_t *)((uint32_t *)((uint64_t *)pWrk + (nelems * subelems)) + myIdx);

    for (j = myIdx; j < nelems * subelems; j += groupSize) {
        payld = *((uint32_t *)source + j);
        tmp[0] = payld;
        *tmp_rdxn = *((uint64_t *)tmp);
        nvshmemi_signal_for_barrier<long>((long *)(pWrk + j), *(long *)tmp_rdxn, next_rank);
    }
    gpu_head_check_op_threadgroup<TYPE, OP, SCOPE>(dest, pWrk, source, nelems);
    /* sync needed on volta (intermittent hangs seen otherwise) */
    nvshmem_threadgroup_sync<SCOPE>();

    for (i = 1; i < (size - 1); i++) {
        __threadfence_system();
        /* Don't want notification to overtake data transfer */
        *tmp_rdxn_2 = 1;
        nvshmemi_signal_for_barrier<long>((long *)(pWrk + (nelems * subelems) + myIdx),
                                          *(long *)tmp_rdxn_2, prev_rank);
        while (1 != *my_notif_ptr)
            ;
        *my_notif_ptr = 0;
        for (j = myIdx; j < nelems * subelems; j += groupSize) {
            payld = *((uint32_t *)dest + j);
            tmp[0] = payld;
            *tmp_rdxn = *((uint64_t *)tmp);
            nvshmemi_signal_for_barrier<long>((long *)(pWrk + j), *(long *)tmp_rdxn, next_rank);
        }
        gpu_head_check_op_threadgroup<TYPE, OP, SCOPE>(dest, pWrk, source, nelems);
        nvshmemi_threadgroup_sync<SCOPE>();
    }
}

template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
static inline __device__ void nvshmemi_gpu_rdxn_threadgroup_putring_direct(nvshmem_team_t team,
                                                                           TYPE *dest,
                                                                           const TYPE *source,
                                                                           size_t nelems) {
    size_t offset;
    char *notif_pwrk_dest;
    char *round_pwrk_dest;
    int next_rank = -1;
    int prev_rank = -1;
    int i;
    size_t j;
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int start = teami->start;
    int stride = teami->stride;
    int size = teami->size;
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    TYPE *pWrk = (TYPE *)nvshmemi_team_get_psync(teami, REDUCE);
    int PE_end = start + (stride * size);
    uint32_t tmp[2];
    uint32_t payld;
    uint32_t *notif_ptr = NULL;
    volatile uint32_t *my_notif_ptr = NULL;
    size_t subelems = sizeof(TYPE) / sizeof(uint32_t);
    tmp[1] = 1;
    offset = (char *)pWrk -
             (char *)(__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p +
                            nvshmemi_mype_d));
    my_notif_ptr = (uint32_t *)((uint32_t *)((uint64_t *)pWrk + (nelems * subelems)) + myIdx);
    next_rank = (nvshmemi_device_state_d.mype != (PE_end - stride))
                    ? (nvshmemi_device_state_d.mype + stride)
                    : start;
    round_pwrk_dest =
        (char *)(__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p +
                       next_rank)) +
        offset;
    prev_rank = (nvshmemi_device_state_d.mype != start) ? (nvshmemi_device_state_d.mype - stride)
                                                        : (PE_end - stride);
    notif_pwrk_dest =
        (char *)(__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p +
                       prev_rank)) +
        offset;
    notif_ptr =
        (uint32_t *)((uint32_t *)((uint64_t *)notif_pwrk_dest + (nelems * subelems)) + myIdx);

    for (j = myIdx; j < nelems * subelems; j += groupSize) {
        payld = *((uint32_t *)source + j);
        tmp[0] = payld;
        *((uint64_t *)round_pwrk_dest + j) = *((uint64_t *)tmp);
    }
    gpu_head_check_op_threadgroup<TYPE, OP, SCOPE>(dest, pWrk, source, nelems);
    /* sync needed on volta (intermittent hangs seen otherwise) */
    nvshmemi_threadgroup_sync<SCOPE>();

    for (i = 1; i < (size - 1); i++) {
        __threadfence_system();
        /* Don't want notification to overtake data transfer */
        *notif_ptr = 1;
        while (1 != *my_notif_ptr)
            ;
        *my_notif_ptr = 0;
        for (j = myIdx; j < nelems * subelems; j += groupSize) {
            payld = *((uint32_t *)dest + j);
            tmp[0] = payld;
            *((uint64_t *)round_pwrk_dest + j) = *((uint64_t *)tmp);
        }
        gpu_head_check_op_threadgroup<TYPE, OP, SCOPE>(dest, pWrk, source, nelems);
        nvshmemi_threadgroup_sync<SCOPE>();
    }
}

template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
static inline __device__ void nvshmemi_gpu_rdxn_threadgroup_putall(nvshmem_team_t team, TYPE *dest,
                                                                   const TYPE *source,
                                                                   size_t nelems) {
    int i;
    size_t j;
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int start = teami->start;
    int stride = teami->stride;
    int size = teami->size;
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    TYPE *pWrk = (TYPE *)nvshmemi_team_get_psync(teami, REDUCE);
    int PE_end = start + (stride * size);
    uint64_t *tmp_rdxn = pWrk;
    uint32_t tmp[2];
    uint32_t payld;
    size_t subelems = sizeof(TYPE) / sizeof(uint32_t);
    int my_active_set_pe = ((nvshmemi_device_state_d.mype - start) / stride);
    tmp[1] = 1;

    for (j = myIdx; j < nelems * subelems; j += groupSize) {
        payld = *((uint32_t *)source + j);
        tmp[0] = payld;
        *tmp_rdxn = *((uint64_t *)tmp);
        for (i = start; i < nvshmemi_device_state_d.mype; i += stride) {
            nvshmemi_signal_for_barrier<long>(
                (long *)(pWrk + j + (nelems * subelems * my_active_set_pe)), *(long *)tmp_rdxn, i);
        }
        for (i = nvshmemi_device_state_d.mype + stride; i < PE_end; i += stride) {
            nvshmemi_signal_for_barrier<long>(
                (long *)(pWrk + j + (nelems * subelems * my_active_set_pe)), *(long *)tmp_rdxn, i);
        }
    }
    gpu_head_checkall_op_threadgroup<TYPE, OP, SCOPE>(dest, pWrk, source, nelems);
    __threadfence();
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
static inline __device__ void nvshmemi_gpu_rdxn_threadgroup_putall_direct(nvshmem_team_t team,
                                                                          TYPE *dest, TYPE *source,
                                                                          size_t nelems) {
    size_t offset;
    char *round_pwrk_dest;
    int i, j;
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int start = teami->start;
    int stride = teami->stride;
    int size = teami->size;
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    TYPE *pWrk = (TYPE *)nvshmemi_team_get_psync(teami, REDUCE);
    int PE_end = start + (stride * size);
    uint32_t tmp[2];
    uint32_t payld;
    size_t subelems = sizeof(TYPE) / sizeof(uint32_t);
    int my_active_set_pe = ((nvshmemi_device_state_d.mype - start) / stride);
    tmp[1] = 1;
    offset = (char *)pWrk -
             (char *)(__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p +
                            nvshmemi_device_state_d.mype));

    for (j = myIdx; j < nelems * subelems; j += groupSize) {
        payld = *((uint32_t *)source + j);
        tmp[0] = payld;
        for (i = nvshmemi_device_state_d.mype + stride; i < PE_end; i += stride) {
            round_pwrk_dest =
                (char *)(__ldg(
                    (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + i)) +
                offset;
            *((uint64_t *)round_pwrk_dest + j + (nelems * subelems * my_active_set_pe)) =
                *((uint64_t *)tmp);
        }
        for (i = start; i < nvshmemi_device_state_d.mype; i += stride) {
            round_pwrk_dest =
                (char *)(__ldg(
                    (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + i)) +
                offset;
            *((uint64_t *)round_pwrk_dest + j + (nelems * subelems * my_active_set_pe)) =
                *((uint64_t *)tmp);
        }
    }
    gpu_head_checkall_op_threadgroup<TYPE, OP, SCOPE>(dest, pWrk, source, nelems);
    __threadfence();
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
static __forceinline__ __device__ void nvshmemi_gpu_rdxn_hierarchical_fcollect_threadgroup(
    nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nreduce) {
    nvshmemi_team_t *teami_node = nvshmemi_device_state_d.team_pool[NVSHMEM_TEAM_NODE_INDEX];
    nvshmemi_team_t *teami_same_mype_node =
        nvshmemi_device_state_d.team_pool[NVSHMEMX_TEAM_SAME_MYPE_NODE];
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();

    if (!myIdx) { /* Only one thread should increment rdxn_count */
        teami_node->rdxn_count++;
        teami_same_mype_node->rdxn_count++;
    }
    nvshmemi_threadgroup_sync<SCOPE>();

    TYPE *pWrk = (TYPE *)nvshmemi_team_get_psync(teami_node, REDUCE);
    if (teami_node->size >= 2)
        nvshmemi_fcollect_threadgroup<TYPE, SCOPE>(
            NVSHMEMX_TEAM_NODE, pWrk, source, nvshmemi_team_my_pe(NVSHMEMX_TEAM_NODE) * nreduce,
            nreduce);
    else
        nvshmemi_memcpy_threadgroup<SCOPE>(dest, source, nreduce * sizeof(TYPE));

    if (teami_node->size >= 2) {
        for (int j = myIdx; j < nreduce; j += groupSize) {
            gpu_linear_reduce_threadgroup<TYPE, OP, NVSHMEMI_THREADGROUP_THREAD>(
                pWrk + j, pWrk + nreduce + j, dest + j, 1);
            for (int i = 2; i < teami_node->size; i++) {
                gpu_linear_reduce_threadgroup<TYPE, OP, NVSHMEMI_THREADGROUP_THREAD>(
                    pWrk + i * nreduce + j, dest + j, dest + j, 1);
            }
        }
    }
    nvshmemi_threadgroup_sync<SCOPE>();

    if (teami_same_mype_node->size >= 2) {
        pWrk = (TYPE *)nvshmemi_team_get_psync(teami_same_mype_node, REDUCE);
        nvshmemi_fcollect_threadgroup<TYPE, SCOPE>(
            NVSHMEMX_TEAM_SAME_MYPE_NODE, pWrk, dest,
            nvshmemi_team_my_pe(NVSHMEMX_TEAM_SAME_MYPE_NODE) * nreduce, nreduce);
#if CUDART_VERSION >= 12000 && defined(__cplusplus) && __cplusplus >= 201703L
        if constexpr (SCOPE == NVSHMEMI_THREADGROUP_BLOCK && OP == RDXN_OPS_SUM &&
                      sizeof(TYPE) >= 4 && sizeof(TYPE) <= 8) {
            for (int i = myIdx; i < nreduce; i += groupSize) *(dest + i) = 0;
            nvshmemi_threadgroup_sync<SCOPE>();
            auto block = cg::this_thread_block();
            auto tile = cg::tiled_partition<32>(block);
            for (int j = 0; j < nreduce; j++) {
                cg::reduce_update_async(
                    tile, cuda::atomic_ref<TYPE, cuda::thread_scope_block>(dest[j]),
                    (myIdx < teami_same_mype_node->size) ? *((TYPE *)pWrk + myIdx * nreduce + j)
                                                         : (TYPE)0,
                    cg::plus<TYPE>());
            }
        } else
#endif
        {
            for (int j = myIdx; j < nreduce; j += groupSize) {
                gpu_linear_reduce_threadgroup<TYPE, OP, NVSHMEMI_THREADGROUP_THREAD>(
                    (TYPE *)pWrk + j, (TYPE *)pWrk + nreduce + j, dest + j, 1);
                for (int i = 2; i < teami_same_mype_node->size; i++) {
                    gpu_linear_reduce_threadgroup<TYPE, OP, NVSHMEMI_THREADGROUP_THREAD>(
                        (TYPE *)pWrk + i * nreduce + j, dest + j, dest + j, 1);
                }
            }
        }
        nvshmemi_threadgroup_sync<SCOPE>();
    }
}

template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
static inline __device__ void nvshmemi_gpu_rdxn_threadgroup(nvshmem_team_t team, TYPE *dest,
                                                            const TYPE *source, size_t nreduce) {
#ifdef NVSHMEM_GPU_COLL_USE_LDST
#ifdef NVSHMEM_DISABLE_COLL_POLL
    nvshmemi_gpu_rdxn_threadgroup_zcopy_get_bar_direct<TYPE, OP, SCOPE>(team, dest, source,
                                                                        nreduce);
#else
    size_t subelems = sizeof(TYPE) / sizeof(uint32_t);
    size_t pwrk_req_sz_allgather = ((subelems * nreduce) * sizeof(uint64_t)) * size;
    size_t wrk_size =
        ((nvshmemi_device_state_d.gpu_coll_env_params_var.reduce_scratch_size / 2) / sizeof(long)) *
        sizeof(TYPE);
    if (subelems && (pwrk_req_sz_allgather <= wrk_size)) {
        nvshmemi_gpu_rdxn_threadgroup_putall_direct<TYPE, OP, SCOPE>(team, dest, source, nreduce);
    } else {
        nvshmemi_gpu_rdxn_threadgroup_zcopy_get_bar_direct<TYPE, OP, SCOPE>(team, dest, source,
                                                                            nreduce);
    }
#endif
#else
#ifdef NVSHMEM_DISABLE_COLL_POLL
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int start = teami->start;
    int stride = teami->stride;
    int size = teami->size;
    int k = nvshmemi_device_state_d.gpu_coll_env_params_var.reduce_recexch_kval;
    if (start == 0 && stride == 1 && size == nvshmemi_device_state_d.npes && sizeof(TYPE) >= 4 &&
        nreduce % 2 == 0 &&
        nvshmemi_device_state_d.gpu_coll_env_params_var.reduce_scratch_size / 2 >=
            size * nreduce * sizeof(TYPE) &&
        nvshmemi_device_state_d.gpu_coll_env_params_var.fcollect_ll_threshold >=
            nreduce * sizeof(TYPE) &&
        SCOPE == NVSHMEMI_THREADGROUP_BLOCK)
        nvshmemi_gpu_rdxn_hierarchical_fcollect_threadgroup<TYPE, OP, SCOPE>(team, dest, source,
                                                                             nreduce);
    else if (start == 0 && stride == 1 && size == nvshmemi_device_state_d.npes &&
             ((nvshmemi_device_state_d.gpu_coll_env_params_var.reduce_scratch_size / 2) /
              sizeof(long)) >=
                 ((k - 1) * nreduce + k * teami->reduce_recexch.step2_nphases * nreduce +
                  teami->reduce_recexch.step2_nphases * nreduce)) {
        gpu_rdxn_recexch_threadgroup<TYPE, OP, SCOPE>(team, dest, source, nreduce);
    } else {
        gpu_rdxn_segment_threadgroup<TYPE, OP, SCOPE>(team, dest, source, nreduce);
    }
#else
    size_t subelems = sizeof(TYPE) / sizeof(uint32_t);
    size_t pwrk_req_sz_allgather = ((subelems * nreduce) * sizeof(uint64_t)) * size;
    int groupSize = nvshmemi_threadgroup_size<SCCOPE>();
    size_t pwrk_req_sz_ring =
        ((subelems * nreduce) * sizeof(uint64_t)) + (groupSize * sizeof(uint32_t));
    size_t wrk_size =
        ((nvshmemi_device_state_d.gpu_coll_env_params_var.reduce_scratch_size / 2) / sizeof(long)) *
        sizeof(TYPE);
    if (subelems && pwrk_req_sz_allgather <= wrk_size) {
        nvshmemi_gpu_rdxn_threadgroup_putall<TYPE, OP, SCOPE>(team, dest, source, nreduce);
    } else if (subelems && pwrk_req_sz_ring <= wrk_size) {
        nvshmemi_gpu_rdxn_threadgroup_putring<TYPE, OP, SCOPE>(team, dest, source, nreduce);
    } else {
        nvshmemi_gpu_rdxn_threadgroup_put_bar<TYPE, OP, SCOPE>(team, dest, source, nreduce);
    }
#endif
#endif
}

/* Works for inplace and out-of-place reduction */
template <typename TYPE, threadgroup_t SCOPE>
__device__ inline void nvshmemi_add_reduce_mcast_threadroup(nvshmemi_team_t *teami,
                                                            TYPE *__restrict__ dst_ptr,
                                                            const TYPE *__restrict__ src_ptr,
                                                            int nreduce) {
    TYPE *src = (TYPE *)nvshmemi_mc_ptr(teami, src_ptr);
    TYPE *dest = (TYPE *)nvshmemi_mc_ptr(teami, dst_ptr);
    nvshmemi_threadgroup_sync<SCOPE>();
    size_t len = nreduce * sizeof(TYPE);

    if ((uintptr_t)dest % sizeof(int4) == 0 && (uintptr_t)src % sizeof(int4) == 0 &&
        len >= sizeof(int4)) {
        const size_t nelems = len / sizeof(int4);
        int4 *__restrict__ dst_p = (int4 *)dest;
        const int4 *__restrict__ src_p = (const int4 *)src;
        if (len >= 64 && len % 64 == 0)
            nvshmemi_f32_add_reduce_mcast16_v4_threadgroup<TYPE, SCOPE, 4>(dst_p, src_p, nelems);
        else
            nvshmemi_f32_add_reduce_mcast16_v4_threadgroup<TYPE, SCOPE, 1>(dst_p, src_p, nelems);
        len -= nelems * sizeof(int4);
        if (0 == len) return;
        dest = (TYPE *)(dst_p + nelems);
        src = (TYPE *)(src_p + nelems);
    }

    if ((uintptr_t)dest % sizeof(uint64_t) == 0 && (uintptr_t)src % sizeof(uint64_t) == 0 &&
        len >= sizeof(uint64_t)) {
        const size_t nelems = len / sizeof(uint64_t);
        uint64_t *__restrict__ dst_p = (uint64_t *)dest;
        const uint64_t *__restrict__ src_p = (const uint64_t *)src;
        nvshmemi_f32_add_reduce_mcast8_v2_threadgroup<TYPE, SCOPE>(dst_p, src_p, nelems);
        len -= nelems * sizeof(uint64_t);
        if (0 == len) return;
        dest = (TYPE *)(dst_p + nelems);
        src = (TYPE *)(src_p + nelems);
    }

    if ((uintptr_t)dest % sizeof(uint32_t) == 0 && (uintptr_t)src % sizeof(uint32_t) == 0 &&
        len >= sizeof(uint32_t)) {
        const size_t nelems = len / sizeof(uint32_t);
        uint32_t *__restrict__ dst_p = (uint32_t *)dest;
        const uint32_t *__restrict__ src_p = (const uint32_t *)src;
        nvshmemi_f32_add_reduce_mcast4_threadgroup<TYPE, SCOPE>(dst_p, src_p, nelems);
        len -= nelems * sizeof(uint32_t);
        if (0 == len) return;
    }

    return;
}

template <typename TYPE, threadgroup_t SCOPE>
__device__ inline void nvshmemi_add_reduce_nvls_twoshot_threadgroup(nvshmem_team_t team, TYPE *dest,
                                                                    const TYPE *source,
                                                                    size_t nreduce) {
#if __CUDA_ARCH__ >= 900 && CUDART_VERSION >= 12010
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int my_idx_in_active_set = (nvshmemi_device_state_d.mype - teami->start) / (teami->stride);
    /* Divide nreduce by team size and handle for the 3 cases */
    int elems_per_pe = nreduce / teami->size;
    int elems_remain = nreduce % teami->size;
    // Case 1: elems_per_pe == 0 => GPU [size-1] does the work on nreduce
    // Case 2: elems_per_pe != 0 and elems_remain != 0 => GPU [0-size-2] does elems_per_pe,
    // GPU[size-1] does elems_per_pe + elems_remain Case 3: elems_per_pe != 0 and elems_remain == 0
    // => all GPUs do work for elems_per_pe
    int my_nelems = elems_per_pe;
    if (my_idx_in_active_set == (teami->size - 1)) {
        my_nelems = elems_per_pe + elems_remain;
    }

    if (my_nelems > 0) {
        nvshmemi_add_reduce_mcast_threadroup<TYPE, SCOPE>(
            teami, dest + elems_per_pe * my_idx_in_active_set,
            source + elems_per_pe * my_idx_in_active_set, my_nelems);
    }

    nvshmemi_barrier_threadgroup<SCOPE>(team);
#else
    assert(0 && "Unsupported NVLS on this platform\n");
#endif
}

/* This is the entry function for any rdxn collective op - host, on-stream, device
   There is only one exception - nvshmemi_reduce_kernel that is directly calling
   a specific reduction algorithm. That is a special need for team creation */
template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
__device__ inline void nvshmemi_reduce_threadgroup(nvshmem_team_t team, TYPE *dest,
                                                   const TYPE *source, size_t nreduce) {
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    if (!myIdx) /* Only one thread should increment rdxn_count */
        nvshmemi_device_state_d.team_pool[team]->rdxn_count += 1;
    nvshmemi_threadgroup_sync<SCOPE>();

    constexpr bool is_rdxn_sum = (OP == RDXN_OPS_SUM);
    constexpr bool is_float_v = is_float<TYPE>::value;
    int reduce_algo = nvshmemi_device_state_d.gpu_coll_env_params_var.reduce_algo;
    switch (reduce_algo) {
        case 0:
        case 3: /* NVLS Two Shot Allreduce for REDUCE_SUM and FLOAT dtype */
            if (is_rdxn_sum && is_float_v &&
                nvshmemi_device_state_d.team_pool[team]->nvls_rsc_base_ptr != NULL &&
                (nreduce * sizeof(TYPE)) % 4 == 0)
                reduce_algo = 3;
            else {
                reduce_algo = 0;
            }
            break;
        default:
            break;
    }

    switch (reduce_algo) {
        case 3: /* NVLS Two Shot Allreduce (RS + AG) */
            nvshmemi_add_reduce_nvls_twoshot_threadgroup<TYPE, SCOPE>(team, dest, source, nreduce);
            break;
        default:
            nvshmemi_gpu_rdxn_threadgroup<TYPE, OP, SCOPE>(team, dest, source, nreduce);
            break;
    }
}

static __device__ inline void nvshmemi_double2_maxloc_reduce_alltoall_block(nvshmem_team_t team,
                                                                            double2 *dest,
                                                                            const double2 *source) {
#define SCOPE NVSHMEMI_THREADGROUP_BLOCK
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    const int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    const int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    if (!myIdx) { /* Only one thread should increment */
        teami->ll_flag++;
        teami->rdxn_count += 1;
    }
    nvshmemi_threadgroup_sync<SCOPE>();
    const uint32_t ll_flag = teami->ll_flag;
    char *pWrk = (char *)nvshmemi_team_get_psync(teami, REDUCE);

    nvshmemi_packLL_naive<double2, SCOPE>((uint64_t *)pWrk, source, 1, ll_flag);
    nvshmemi_threadgroup_sync<SCOPE>();
    const int my_pe = nvshmemi_team_my_pe(team);
    const int n_pes = nvshmemi_team_n_pes(team);
    for (int i = myIdx + 1; i < n_pes; i += groupSize) {
        int peer = (my_pe + i) % n_pes;
        size_t offset = 2 * sizeof(double2) * (my_pe + 2);
        nvshmemi_put_nbi_threadgroup<uint64_t, NVSHMEMI_THREADGROUP_THREAD>(
            (uint64_t *)(pWrk + offset), (uint64_t *)(pWrk), sizeof(double2) / sizeof(uint32_t),
            nvshmemi_team_translate_pe(team, peer, NVSHMEM_TEAM_WORLD));
    }

    if (!myIdx) {
        dest[0] = source[0];
        for (int i = 1; i < n_pes; i++) {
            int peer = (my_pe + i) % n_pes;
            size_t offset = 2 * sizeof(double2) * (peer + 2);
            nvshmemi_recvLL<double2, NVSHMEMI_THREADGROUP_THREAD>(
                (double2 *)(pWrk + 2 * sizeof(double2)), (uint64_t *)(pWrk + offset), 1, ll_flag);
            dest[0] = perform_gpu_rdxn<double2, RDXN_OPS_MAXLOC>(
                dest[0], *((double2 *)(pWrk + 2 * sizeof(double2))));
        }
    }
    nvshmemi_threadgroup_sync<SCOPE>();
#undef SCOPE
}

static __device__ inline void nvshmemi_double2_maxloc_rooted_reduce_flat_block(
    nvshmem_team_t team, double2 *dest, const double2 *source) {
#define SCOPE NVSHMEMI_THREADGROUP_BLOCK
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    const int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    const int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    if (!myIdx) { /* Only one thread should increment */
        teami->ll_flag++;
        teami->rdxn_count += 1;
    }
    nvshmemi_threadgroup_sync<SCOPE>();
    const uint32_t ll_flag = teami->ll_flag;
    char *pWrk = (char *)nvshmemi_team_get_psync(teami, REDUCE);

    if (nvshmemi_team_my_pe(team) != 0) {
        nvshmemi_packLL_naive<double2, SCOPE>((uint64_t *)pWrk, source, 1, ll_flag);
        size_t offset = 2 * sizeof(double2) * nvshmemi_team_my_pe(team);
        nvshmemi_put_nbi_threadgroup<uint64_t, SCOPE>(
            (uint64_t *)(pWrk + offset), (uint64_t *)(pWrk), sizeof(double2) / sizeof(uint32_t),
            nvshmemi_team_translate_pe(team, 0, NVSHMEM_TEAM_WORLD));
    } else {
        dest[0] = source[0];
        if (!myIdx) {
            for (int i = 1; i < teami->size; i += 1) {
                size_t offset = 2 * sizeof(double2) * i;
                nvshmemi_recvLL<double2, NVSHMEMI_THREADGROUP_THREAD>(
                    (double2 *)pWrk, (uint64_t *)(pWrk + offset), 1, ll_flag);
                dest[0] = perform_gpu_rdxn<double2, RDXN_OPS_MAXLOC>(dest[0], *(double2 *)pWrk);
            }
        }
        nvshmemi_threadgroup_sync<SCOPE>();
    }
#undef SCOPE
}

/* double2 MAXLOC implementation */
static __device__ inline int nvshmemx_double2_maxloc_reduce_block(nvshmem_team_t team,
                                                                  double2 *dest,
                                                                  const double2 *source,
                                                                  size_t nreduce) {
#ifdef NVSHMEM_DEBUG
    assert(nreduce == 1);
#endif
#define SCOPE NVSHMEMI_THREADGROUP_BLOCK
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    switch (nvshmemi_device_state_d.gpu_coll_env_params_var.reduce_maxloc_algo) {
        case 1: /*  Alltoall algorithm */
            nvshmemi_double2_maxloc_reduce_alltoall_block(team, dest, source);
            break;
        case 2: /* Topo-unaware: Flat reduce + Flat bcast */
            nvshmemi_double2_maxloc_rooted_reduce_flat_block(team, dest, source);
            nvshmemi_bcast_internode_tree_threadgroup<double2, SCOPE>(
                team, dest, dest, 1, 0, nvshmemi_team_n_pes(team) - 1);
            break;
        case 3: /* Topo aware two-level flat reduce + Topo aware two-level tree bcast */
            if (teami->is_team_node || teami->is_team_same_mype_node) {
                nvshmemi_double2_maxloc_rooted_reduce_flat_block(team, dest, source);
            } else {
                nvshmemi_double2_maxloc_rooted_reduce_flat_block(teami->team_node, dest, source);
                if (nvshmemi_team_my_pe(teami->team_node) == 0) {
                    nvshmemi_double2_maxloc_rooted_reduce_flat_block(teami->team_same_mype_node,
                                                                     dest, dest);
                }
            }
            nvshmemi_bcast_hierarchical_threadgroup<double2, SCOPE>(team, dest, dest, 1, 0);
            break;
        case 4: /* Topo aware two-level flat reduce + Topo aware two-level tree bcast */
            if (teami->is_team_node || teami->is_team_same_mype_node) {
                nvshmemi_double2_maxloc_reduce_alltoall_block(team, dest, source);
            } else {
                nvshmemi_double2_maxloc_reduce_alltoall_block(teami->team_node, dest, source);
                nvshmemi_double2_maxloc_reduce_alltoall_block(teami->team_same_mype_node, dest,
                                                              dest);
            }
            break;
        default:
            assert(0);
            break;
    }
#undef SCOPE
    return 0;
}

#endif /* __CUDA_ARCH__ */
#endif /* REDUCE_DEVICE_CUH */
