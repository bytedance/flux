/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See COPYRIGHT.txt for license information
 */

#ifndef _NVSHMEMX_API_H_
#define _NVSHMEMX_API_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>
#include "device_host_transport/nvshmem_constants.h"
#include "device_host/nvshmem_common.cuh"
#include "non_abi/nvshmem_version.h"
#include "host/nvshmemx_coll_api.h"
#include "host/nvshmem_macros.h"
#include "non_abi/nvshmemx_error.h"
#include "host/nvshmem_api.h"

int nvshmemi_collective_launch(const void *func, dim3 gridDims, dim3 blockDims, void **args,
                               size_t sharedMem, cudaStream_t stream);

int nvshmemi_collective_launch_query_gridsize(const void *func, dim3 blockDims, void **args,
                                              size_t sharedMem, int *gridsize);

#ifdef __cplusplus
extern "C" {
#endif

enum flags {
    NVSHMEMX_INIT_THREAD_PES = 1,
    NVSHMEMX_INIT_WITH_MPI_COMM = 1 << 1,
    NVSHMEMX_INIT_WITH_SHMEM = 1 << 2,
    NVSHMEMX_INIT_WITH_UNIQUEID = 1 << 3,
    NVSHMEMX_INIT_MAX = 1 << 31
};

// Heap management extensions
int nvshmemx_buffer_register(void *addr, size_t length);
int nvshmemx_buffer_unregister(void *addr);
void nvshmemx_buffer_unregister_all();

// Initialization & Finalization extensions
int nvshmemx_hostlib_init_attr(unsigned int flags, nvshmemx_init_attr_t *attr);
void nvshmemx_hostlib_finalize();

static inline int nvshmemx_init_attr(unsigned int flags, nvshmemx_init_attr_t *attributes) {
    int status = 0, requested = NVSHMEM_THREAD_SERIALIZED, provided;
    nvshmemi_version_t app_nvshmem_version = {NVSHMEM_INTERLIB_MAJOR_VERSION,
                                              NVSHMEM_INTERLIB_MINOR_VERSION,
                                              NVSHMEM_INTERLIB_PATCH_VERSION};
    if (attributes != NULL) {
        nvshmemx_init_init_attr_ver_only((*attributes));
    }
    status = nvshmemi_init_thread(requested, &provided, flags, attributes, app_nvshmem_version);
    NONZERO_EXIT(status, "aborting due to error in nvshmemi_init_thread \n");
    return status;
}

int nvshmemx_set_attr_uniqueid_args(const int myrank, const int nranks,
                                    const nvshmemx_uniqueid_t *uniqueid,
                                    nvshmemx_init_attr_t *attr);
int nvshmemx_get_uniqueid(nvshmemx_uniqueid_t *uniqueid);

int nvshmemx_cumodule_init(CUmodule module);
int nvshmemx_cumodule_finalize(CUmodule module);

//////////////////// Put On Stream ////////////////////

#define NVSHMEMX_DECL_TYPE_P_ON_STREAM(NAME, TYPE) \
    void nvshmemx_##NAME##_p_on_stream(TYPE *dest, const TYPE value, int pe, cudaStream_t cstrm);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_DECL_TYPE_P_ON_STREAM)
#undef NVSHMEMX_DECL_TYPE_P_ON_STREAM

#define NVSHMEMX_DECL_TYPE_PUT_ON_STREAM(NAME, TYPE)                                            \
    void nvshmemx_##NAME##_put_on_stream(TYPE *dest, const TYPE *source, size_t nelems, int pe, \
                                         cudaStream_t cstrm);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_DECL_TYPE_PUT_ON_STREAM)
#undef NVSHMEMX_DECL_TYPE_PUT_ON_STREAM

#define NVSHMEMX_DECL_SIZE_PUT_ON_STREAM(NAME)                                                 \
    void nvshmemx_put##NAME##_on_stream(void *dest, const void *source, size_t nelems, int pe, \
                                        cudaStream_t cstrm);

NVSHMEMI_REPT_FOR_SIZES(NVSHMEMX_DECL_SIZE_PUT_ON_STREAM)
#undef NVSHMEMX_DECL_SIZE_PUT_ON_STREAM

void nvshmemx_putmem_on_stream(void *dest, const void *source, size_t bytes, int pe,
                               cudaStream_t cstrm);

#define NVSHMEMX_DECL_TYPE_PUT_SIGNAL_ON_STREAM(NAME, TYPE)                                      \
    void nvshmemx_##NAME##_put_signal_on_stream(TYPE *dest, const TYPE *source, size_t nelems,   \
                                                uint64_t *sig_addr, uint64_t signal, int sig_op, \
                                                int pe, cudaStream_t cstrm);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_DECL_TYPE_PUT_SIGNAL_ON_STREAM)
#undef NVSHMEMX_DECL_TYPE_PUT_SIGNAL_ON_STREAM

#define NVSHMEMX_DECL_SIZE_PUT_SIGNAL_ON_STREAM(NAME)                                           \
    void nvshmemx_put##NAME##_signal_on_stream(void *dest, const void *source, size_t nelems,   \
                                               uint64_t *sig_addr, uint64_t signal, int sig_op, \
                                               int pe, cudaStream_t cstrm);

NVSHMEMI_REPT_FOR_SIZES(NVSHMEMX_DECL_SIZE_PUT_SIGNAL_ON_STREAM)
#undef NVSHMEMX_DECL_SIZE_PUT_SIGNAL_ON_STREAM

void nvshmemx_putmem_signal_on_stream(void *dest, const void *source, size_t bytes,
                                      uint64_t *sig_addr, uint64_t signal, int sig_op, int pe,
                                      cudaStream_t cstrm);

#define NVSHMEMX_DECL_TYPE_PUT_SIGNAL_NBI_ON_STREAM(NAME, TYPE)                                    \
    void nvshmemx_##NAME##_put_signal_nbi_on_stream(TYPE *dest, const TYPE *source, size_t nelems, \
                                                    uint64_t *sig_addr, uint64_t signal,           \
                                                    int sig_op, int pe, cudaStream_t cstrm);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_DECL_TYPE_PUT_SIGNAL_NBI_ON_STREAM)
#undef NVSHMEMX_DECL_TYPE_PUT_SIGNAL_NBI_ON_STREAM

#define NVSHMEMX_DECL_SIZE_PUT_SIGNAL_NBI_ON_STREAM(NAME)                                         \
    void nvshmemx_put##NAME##_signal_nbi_on_stream(void *dest, const void *source, size_t nelems, \
                                                   uint64_t *sig_addr, uint64_t signal,           \
                                                   int sig_op, int pe, cudaStream_t cstrm);

NVSHMEMI_REPT_FOR_SIZES(NVSHMEMX_DECL_SIZE_PUT_SIGNAL_NBI_ON_STREAM)
#undef NVSHMEMX_DECL_SIZE_PUT_SIGNAL_NBI_ON_STREAM

void nvshmemx_putmem_signal_nbi_on_stream(void *dest, const void *source, size_t bytes,
                                          uint64_t *sig_addr, uint64_t signal, int sig_op, int pe,
                                          cudaStream_t cstrm);

#define NVSHMEMX_DECL_TYPE_IPUT_ON_STREAM(NAME, TYPE)                                    \
    void nvshmemx_##NAME##_iput_on_stream(TYPE *dest, const TYPE *source, ptrdiff_t dst, \
                                          ptrdiff_t sst, size_t nelems, int pe,          \
                                          cudaStream_t cstrm);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_DECL_TYPE_IPUT_ON_STREAM)
#undef NVSHMEMX_DECL_TYPE_IPUT_ON_STREAM

#define NVSHMEMX_DECL_SIZE_IPUT_ON_STREAM(NAME)                                         \
    void nvshmemx_iput##NAME##_on_stream(void *dest, const void *source, ptrdiff_t dst, \
                                         ptrdiff_t sst, size_t nelems, int pe,          \
                                         cudaStream_t cstrm);

NVSHMEMI_REPT_FOR_SIZES(NVSHMEMX_DECL_SIZE_IPUT_ON_STREAM)
#undef NVSHMEMX_DECL_SIZE_IPUT_ON_STREAM

#define NVSHMEMX_DECL_TYPE_PUT_NBI_ON_STREAM(NAME, TYPE)                                    \
    void nvshmemx_##NAME##_put_nbi_on_stream(TYPE *dest, const TYPE *source, size_t nelems, \
                                             int pe, cudaStream_t cstrm);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_DECL_TYPE_PUT_NBI_ON_STREAM)
#undef NVSHMEMX_DECL_TYPE_PUT_NBI_ON_STREAM

#define NVSHMEMX_DECL_SIZE_PUT_NBI_ON_STREAM(NAME)                                                 \
    void nvshmemx_put##NAME##_nbi_on_stream(void *dest, const void *source, size_t nelems, int pe, \
                                            cudaStream_t cstrm);

NVSHMEMI_REPT_FOR_SIZES(NVSHMEMX_DECL_SIZE_PUT_NBI_ON_STREAM)
#undef NVSHMEMX_DECL_SIZE_PUT_NBI_ON_STREAM

void nvshmemx_putmem_nbi_on_stream(void *dest, const void *source, size_t bytes, int pe,
                                   cudaStream_t cstrm);

//////////////////// Get On Stream ////////////////////

#define NVSHMEMX_DECL_TYPE_G_ON_STREAM(NAME, TYPE) \
    TYPE nvshmemx_##NAME##_g_on_stream(const TYPE *src, int pe, cudaStream_t cstrm);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_DECL_TYPE_G_ON_STREAM)
#undef NVSHMEMX_DECL_TYPE_G_ON_STREAM

#define NVSHMEMX_DECL_TYPE_GET_ON_STREAM(NAME, TYPE)                                            \
    void nvshmemx_##NAME##_get_on_stream(TYPE *dest, const TYPE *source, size_t nelems, int pe, \
                                         cudaStream_t cstrm);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_DECL_TYPE_GET_ON_STREAM)
#undef NVSHMEMX_DECL_TYPE_GET_ON_STREAM

#define NVSHMEMX_DECL_TYPE_IGET_ON_STREAM(NAME, TYPE)                                    \
    void nvshmemx_##NAME##_iget_on_stream(TYPE *dest, const TYPE *source, ptrdiff_t dst, \
                                          ptrdiff_t sst, size_t nelems, int pe,          \
                                          cudaStream_t cstrm);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_DECL_TYPE_IGET_ON_STREAM)
#undef NVSHMEMX_DECL_TYPE_IGET_ON_STREAM

#define NVSHMEMX_DECL_SIZE_GET_ON_STREAM(NAME)                                                 \
    void nvshmemx_get##NAME##_on_stream(void *dest, const void *source, size_t nelems, int pe, \
                                        cudaStream_t cstrm);

NVSHMEMI_REPT_FOR_SIZES(NVSHMEMX_DECL_SIZE_GET_ON_STREAM)
#undef NVSHMEMX_DECL_SIZE_GET_ON_STREAM

void nvshmemx_getmem_on_stream(void *dest, const void *source, size_t bytes, int pe,
                               cudaStream_t cstrm);

#define NVSHMEMX_DECL_SIZE_IGET_ON_STREAM(NAME)                                         \
    void nvshmemx_iget##NAME##_on_stream(void *dest, const void *source, ptrdiff_t dst, \
                                         ptrdiff_t sst, size_t nelems, int pe,          \
                                         cudaStream_t cstrm);

NVSHMEMI_REPT_FOR_SIZES(NVSHMEMX_DECL_SIZE_IGET_ON_STREAM)
#undef NVSHMEMX_DECL_SIZE_IGET_ON_STREAM

#define NVSHMEMX_DECL_TYPE_GET_NBI_ON_STREAM(NAME, TYPE)                                    \
    void nvshmemx_##NAME##_get_nbi_on_stream(TYPE *dest, const TYPE *source, size_t nelems, \
                                             int pe, cudaStream_t cstrm);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_DECL_TYPE_GET_NBI_ON_STREAM)
#undef NVSHMEMX_DECL_TYPE_GET_NBI_ON_STREAM

#define NVSHMEMX_DECL_SIZE_GET_NBI_ON_STREAM(NAME)                                                 \
    void nvshmemx_get##NAME##_nbi_on_stream(void *dest, const void *source, size_t nelems, int pe, \
                                            cudaStream_t cstrm);

NVSHMEMI_REPT_FOR_SIZES(NVSHMEMX_DECL_SIZE_GET_NBI_ON_STREAM)
#undef NVSHMEMX_DECL_SIZE_GET_NBI_ON_STREAM

void nvshmemx_getmem_nbi_on_stream(void *dest, const void *source, size_t bytes, int pe,
                                   cudaStream_t cstrm);

//////////////////// Synchronization On Stream ////////////////////

void nvshmemx_quiet_on_stream(cudaStream_t cstrm);

void nvshmemx_signal_op_on_stream(uint64_t *sig_addr, uint64_t signal, int sig_op, int pe,
                                  cudaStream_t cstrm);

#define NVSHMEMX_DECL_WAIT_UNTIL_ON_STREAM(NAME, Type)                               \
    void nvshmemx_##NAME##_wait_until_on_stream(Type *ivar, int cmp, Type cmp_value, \
                                                cudaStream_t cstream);
NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMX_DECL_WAIT_UNTIL_ON_STREAM)
#undef NVSHMEMX_DECL_WAIT_UNTIL_ON_STREAM

#define NVSHMEMX_DECL_WAIT_UNTIL_ALL_ON_STREAM(NAME, Type)                                         \
    void nvshmemx_##NAME##_wait_until_all_on_stream(Type *ivars, size_t nelems, const int *status, \
                                                    int cmp, Type cmp_value,                       \
                                                    cudaStream_t cstream);
NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMX_DECL_WAIT_UNTIL_ALL_ON_STREAM)
#undef NVSHMEMX_DECL_WAIT_UNTIL_ALL_ON_STREAM

#define NVSHMEMX_DECL_WAIT_UNTIL_ALL_VECTOR_ON_STREAM(NAME, Type)                      \
    void nvshmemx_##NAME##_wait_until_all_vector_on_stream(Type *ivars, size_t nelems, \
                                                           const int *status, int cmp, \
                                                           Type *cmp_value, cudaStream_t cstream);
NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMX_DECL_WAIT_UNTIL_ALL_VECTOR_ON_STREAM)
#undef NVSHMEMX_DECL_WAIT_UNTIL_ALL_VECTOR_ON_STREAM

void nvshmemx_signal_wait_until_on_stream(uint64_t *sig_addr, int cmp, uint64_t cmp_value,
                                          cudaStream_t cstream);
//////////////////// Put on Thread Group ////////////////////

#define NVSHMEMX_DECL_TYPE_PUT_THREADGROUP(NAME, TYPE)                                         \
    __device__ void nvshmemx_##NAME##_put_warp(TYPE *dest, const TYPE *source, size_t nelems,  \
                                               int pe);                                        \
    __device__ void nvshmemx_##NAME##_put_block(TYPE *dest, const TYPE *source, size_t nelems, \
                                                int pe);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_DECL_TYPE_PUT_THREADGROUP)
#undef NVSHMEMX_DECL_TYPE_PUT_THREADGROUP

#define NVSHMEMX_DECL_SIZE_PUT_THREADGROUP(NAME)                                              \
    __device__ void nvshmemx_put##NAME##_warp(void *dest, const void *source, size_t nelems,  \
                                              int pe);                                        \
    __device__ void nvshmemx_put##NAME##_block(void *dest, const void *source, size_t nelems, \
                                               int pe);

NVSHMEMI_REPT_FOR_SIZES(NVSHMEMX_DECL_SIZE_PUT_THREADGROUP)
#undef NVSHMEMX_DECL_SIZE_PUT_THREADGROUP

__device__ void nvshmemx_putmem_warp(void *dest, const void *source, size_t bytes, int pe);
__device__ void nvshmemx_putmem_block(void *dest, const void *source, size_t bytes, int pe);

/* __device__ nvshmem_<typename>_put_signal_scope */
#define NVSHMEMI_TYPENAME_PUT_SIGNAL_SCOPE_DECL(SCOPE, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE) \
    __device__ void nvshmemx_##TYPENAME##_put_signal##SC_SUFFIX(                             \
        TYPE *dest, const TYPE *source, size_t nelems, uint64_t *sig_addr, uint64_t signal,  \
        int sig_op, int pe);
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(NVSHMEMI_TYPENAME_PUT_SIGNAL_SCOPE_DECL, warp,
                                                 _warp, x)
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(NVSHMEMI_TYPENAME_PUT_SIGNAL_SCOPE_DECL, block,
                                                 _block, x)
#undef NVSHMEMI_TYPENAME_PUT_SIGNAL_SCOPE_DECL

/* __device__ nvshmem_putmem_signal_scope */
#define NVSHMEMI_PUTMEM_SIGNAL_SCOPE_DECL(SCOPE, SC_SUFFIX, SC_PREFIX)                   \
    __device__ void nvshmemx_putmem_signal##SC_SUFFIX(void *dest, const void *source,    \
                                                      size_t nelems, uint64_t *sig_addr, \
                                                      uint64_t signal, int sig_op, int pe);

NVSHMEMI_PUTMEM_SIGNAL_SCOPE_DECL(warp, _warp, x)
NVSHMEMI_PUTMEM_SIGNAL_SCOPE_DECL(block, _block, x)
#undef NVSHMEMI_PUTMEM_SIGNAL_SCOPE_DECL

/* __device__ nvshmem_putsize_signal_scope */
#define NVSHMEMI_PUTSIZE_SIGNAL_SCOPE_DECL(SCOPE, SC_SUFFIX, SC_PREFIX, BITS)                 \
    __device__ void nvshmemx_put##BITS##_signal##SC_SUFFIX(void *dest, const void *source,    \
                                                           size_t nelems, uint64_t *sig_addr, \
                                                           uint64_t signal, int sig_op, int pe);

NVSHMEMI_REPT_FOR_SIZES_WITH_SCOPE2(NVSHMEMI_PUTSIZE_SIGNAL_SCOPE_DECL, warp, _warp, x)
NVSHMEMI_REPT_FOR_SIZES_WITH_SCOPE2(NVSHMEMI_PUTSIZE_SIGNAL_SCOPE_DECL, block, _block, x)
#undef NVSHMEMI_REPT_PUTSIZE_SIGNAL_FOR_SCOPE

/* __device__ nvshmem_<typename>_put_signal_nbi_scope */
#define NVSHMEMI_TYPENAME_PUT_SIGNAL_NBI_SCOPE_DECL(SCOPE, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE) \
    __device__ void nvshmemx_##TYPENAME##_put_signal_nbi##SC_SUFFIX(                             \
        TYPE *dest, const TYPE *source, size_t nelems, uint64_t *sig_addr, uint64_t signal,      \
        int sig_op, int pe);
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(NVSHMEMI_TYPENAME_PUT_SIGNAL_NBI_SCOPE_DECL, warp,
                                                 _warp, x)
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(NVSHMEMI_TYPENAME_PUT_SIGNAL_NBI_SCOPE_DECL, block,
                                                 _block, x)
#undef NVSHMEMI_TYPENAME_PUT_SIGNAL_NBI_SCOPE_DECL

/* __device__ nvshmem_putmem_signal_nbi_scope */
#define NVSHMEMI_PUTMEM_SIGNAL_NBI_SCOPE_DECL(SCOPE, SC_SUFFIX, SC_PREFIX)                   \
    __device__ void nvshmemx_putmem_signal_nbi##SC_SUFFIX(void *dest, const void *source,    \
                                                          size_t nelems, uint64_t *sig_addr, \
                                                          uint64_t signal, int sig_op, int pe);

NVSHMEMI_PUTMEM_SIGNAL_NBI_SCOPE_DECL(warp, _warp, x)
NVSHMEMI_PUTMEM_SIGNAL_NBI_SCOPE_DECL(block, _block, x)
#undef NVSHMEMI_PUTMEM_SIGNAL_NBI_SCOPE_DECL

/* __device__ nvshmem_putsize_signal_nbi_scope */
#define NVSHMEMI_PUTSIZE_SIGNAL_NBI_SCOPE_DECL(SCOPE, SC_SUFFIX, SC_PREFIX, BITS)           \
    __device__ void nvshmemx_put##BITS##_signal_nbi##SC_SUFFIX(                             \
        void *dest, const void *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, \
        int sig_op, int pe);

NVSHMEMI_REPT_FOR_SIZES_WITH_SCOPE2(NVSHMEMI_PUTSIZE_SIGNAL_NBI_SCOPE_DECL, warp, _warp, x)
NVSHMEMI_REPT_FOR_SIZES_WITH_SCOPE2(NVSHMEMI_PUTSIZE_SIGNAL_NBI_SCOPE_DECL, block, _block, x)
#undef NVSHMEMI_REPT_PUTSIZE_SIGNAL_NBI_FOR_SCOPE

#define NVSHMEMX_DECL_TYPE_IPUT_THREADGROUP(NAME, TYPE)                                         \
    __device__ void nvshmemx_##NAME##_iput_warp(TYPE *dest, const TYPE *source, ptrdiff_t dst,  \
                                                ptrdiff_t sst, size_t nelems, int pe);          \
    __device__ void nvshmemx_##NAME##_iput_block(TYPE *dest, const TYPE *source, ptrdiff_t dst, \
                                                 ptrdiff_t sst, size_t nelems, int pe);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_DECL_TYPE_IPUT_THREADGROUP)
#undef NVSHMEMX_DECL_TYPE_IPUT_THREADGROUP

#define NVSHMEMX_DECL_SIZE_IPUT_THREADGROUP(NAME)                                              \
    __device__ void nvshmemx_iput##NAME##_warp(void *dest, const void *source, ptrdiff_t dst,  \
                                               ptrdiff_t sst, size_t nelems, int pe);          \
    __device__ void nvshmemx_iput##NAME##_block(void *dest, const void *source, ptrdiff_t dst, \
                                                ptrdiff_t sst, size_t nelems, int pe);

NVSHMEMI_REPT_FOR_SIZES(NVSHMEMX_DECL_SIZE_IPUT_THREADGROUP)
#undef NVSHMEMX_DECL_SIZE_IPUT_THREADGROUP

#define NVSHMEMX_DECL_TYPE_PUT_NBI_THREADGROUP(NAME, TYPE)                                         \
    __device__ void nvshmemx_##NAME##_put_nbi_warp(TYPE *dest, const TYPE *source, size_t nelems,  \
                                                   int pe);                                        \
    __device__ void nvshmemx_##NAME##_put_nbi_block(TYPE *dest, const TYPE *source, size_t nelems, \
                                                    int pe);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_DECL_TYPE_PUT_NBI_THREADGROUP)
#undef NVSHMEMX_DECL_TYPE_PUT_NBI_THREADGROUP

#define NVSHMEMX_DECL_SIZE_PUT_NBI_THREADGROUP(NAME)                                              \
    __device__ void nvshmemx_put##NAME##_nbi_warp(void *dest, const void *source, size_t nelems,  \
                                                  int pe);                                        \
    __device__ void nvshmemx_put##NAME##_nbi_block(void *dest, const void *source, size_t nelems, \
                                                   int pe);

NVSHMEMI_REPT_FOR_SIZES(NVSHMEMX_DECL_SIZE_PUT_NBI_THREADGROUP)
#undef NVSHMEMX_DECL_SIZE_PUT_NBI_THREADGROUP

__device__ void nvshmemx_putmem_nbi_warp(void *dest, const void *source, size_t bytes, int pe);
__device__ void nvshmemx_putmem_nbi_block(void *dest, const void *source, size_t bytes, int pe);

//////////////////// Get on Thread Group ////////////////////

#define NVSHMEMX_DECL_TYPE_GET_THREADGROUP(NAME, TYPE)                                         \
    __device__ void nvshmemx_##NAME##_get_warp(TYPE *dest, const TYPE *source, size_t nelems,  \
                                               int pe);                                        \
    __device__ void nvshmemx_##NAME##_get_block(TYPE *dest, const TYPE *source, size_t nelems, \
                                                int pe);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_DECL_TYPE_GET_THREADGROUP)
#undef NVSHMEMX_DECL_TYPE_GET_THREADGROUP

#define NVSHMEMX_DECL_TYPE_IGET_THREADGROUP(NAME, TYPE)                                         \
    __device__ void nvshmemx_##NAME##_iget_warp(TYPE *dest, const TYPE *source, ptrdiff_t dst,  \
                                                ptrdiff_t sst, size_t nelems, int pe);          \
    __device__ void nvshmemx_##NAME##_iget_block(TYPE *dest, const TYPE *source, ptrdiff_t dst, \
                                                 ptrdiff_t sst, size_t nelems, int pe);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_DECL_TYPE_IGET_THREADGROUP)
#undef NVSHMEMX_DECL_TYPE_IGET_THREADGROUP

#define NVSHMEMX_DECL_SIZE_GET_THREADGROUP(NAME)                                              \
    __device__ void nvshmemx_get##NAME##_warp(void *dest, const void *source, size_t nelems,  \
                                              int pe);                                        \
    __device__ void nvshmemx_get##NAME##_block(void *dest, const void *source, size_t nelems, \
                                               int pe);
NVSHMEMI_REPT_FOR_SIZES(NVSHMEMX_DECL_SIZE_GET_THREADGROUP)
#undef NVSHMEMX_DECL_SIZE_GET_THREADGROUP

__device__ void nvshmemx_getmem_warp(void *dest, const void *source, size_t bytes, int pe);
__device__ void nvshmemx_getmem_block(void *dest, const void *source, size_t bytes, int pe);

#define NVSHMEMX_DECL_SIZE_IGET_THREADGROUP(NAME)                                              \
    __device__ void nvshmemx_iget##NAME##_warp(void *dest, const void *source, ptrdiff_t dst,  \
                                               ptrdiff_t sst, size_t nelems, int pe);          \
    __device__ void nvshmemx_iget##NAME##_block(void *dest, const void *source, ptrdiff_t dst, \
                                                ptrdiff_t sst, size_t nelems, int pe);

NVSHMEMI_REPT_FOR_SIZES(NVSHMEMX_DECL_SIZE_IGET_THREADGROUP)
#undef NVSHMEMX_DECL_SIZE_IGET_THREADGROUP

#define NVSHMEMX_DECL_TYPE_GET_NBI_THREADGROUP(NAME, TYPE)                                         \
    __device__ void nvshmemx_##NAME##_get_nbi_warp(TYPE *dest, const TYPE *source, size_t nelems,  \
                                                   int pe);                                        \
    __device__ void nvshmemx_##NAME##_get_nbi_block(TYPE *dest, const TYPE *source, size_t nelems, \
                                                    int pe);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_DECL_TYPE_GET_NBI_THREADGROUP)
#undef NVSHMEMX_DECL_TYPE_GET_NBI_THREADGROUP

#define NVSHMEMX_DECL_SIZE_GET_NBI_THREADGROUP(NAME)                                              \
    __device__ void nvshmemx_get##NAME##_nbi_warp(void *dest, const void *source, size_t nelems,  \
                                                  int pe);                                        \
    __device__ void nvshmemx_get##NAME##_nbi_block(void *dest, const void *source, size_t nelems, \
                                                   int pe);

NVSHMEMI_REPT_FOR_SIZES(NVSHMEMX_DECL_SIZE_GET_NBI_THREADGROUP)
#undef NVSHMEMX_DECL_SIZE_GET_NBI_THREADGROUP

__device__ void nvshmemx_getmem_nbi_warp(void *dest, const void *source, size_t bytes, int pe);
__device__ void nvshmemx_getmem_nbi_block(void *dest, const void *source, size_t bytes, int pe);

//////////////////// Signal ////////////////////

NVSHMEMI_HOSTDEVICE_PREFIX void nvshmemx_signal_op(uint64_t *sig_addr, uint64_t signal, int sig_op,
                                                   int pe);

#ifdef __cplusplus
}
#endif

#endif /* _NVSHMEMX_API_H_ */
