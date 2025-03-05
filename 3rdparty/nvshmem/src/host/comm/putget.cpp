/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include <cuda_runtime.h>                                    // for cudaMemcpyAsync
#include <driver_types.h>                                    // for cudaStream_t
#include <stddef.h>                                          // for NULL, size_t
#include <stdint.h>                                          // for uint64_t
#include <stdlib.h>                                          // for exit
#include "device_host/nvshmem_types.h"                       // for nvshmemi_device...
#include "device_host/nvshmem_common.cuh"                    // for NVSHMEMI_REPT_F...
#include "host/nvshmem_api.h"                                // for nvshmem_char_put
#include "host/nvshmemx_api.h"                               // for nvshmemx_char_g...
#include "non_abi/nvshmemx_error.h"                          // for NVSHMEMI_ERROR_...
#include "device_host_transport/nvshmem_common_transport.h"  // for NVSHMEMI_OP_PUT
#include "internal/host/debug.h"                             // for INFO, NVSHMEM_P2P
#include "internal/host/nvshmem_internal.h"                  // for NVSHMEMI_CHECK_...
#include "internal/host/nvshmem_nvtx.hpp"                    // for nvtx_cond_range
#include "internal/host/nvshmemi_symmetric_heap.hpp"         // for nvshmemi_symmet...
#include "internal/host/nvshmemi_types.h"                    // for nvshmemi_state
#include "internal/host/util.h"                              // for CUDA_RUNTIME_CH...
#include "internal/host_transport/transport.h"               // for rma_bytesdesc_t

#define NOT_A_CUDA_STREAM ((cudaStream_t)0)

enum {
    NO_NBI = 0,
    NBI,
};

enum {
    NO_ASYNC = 0,
    ASYNC,
};

enum {
    SRC_STRIDE_CONTIG = 1,
};

enum {
    DEST_STRIDE_CONTIG = 1,
};

int nvshmemi_proxy_rma_launcher(void *args[], cudaStream_t cstrm, bool is_nbi, bool is_signal);

static int nvshmemi_p2p_rma_optimized(cudaStream_t custrm /* internal stream */, cudaEvent_t cuev,
                                      rma_verb_t verb, rma_memdesc_t *dest, rma_memdesc_t *src,
                                      rma_bytesdesc_t bytesdesc, uint64_t *sig_addr,
                                      uint64_t signal, int sig_op, int pe) {
    int status = 0;
    bool is_contig = ((bytesdesc.srcstride == 1) && (bytesdesc.deststride == 1)) ? true : false;
    bool is_single_word =
        ((verb.desc == NVSHMEMI_OP_P) || (verb.desc == NVSHMEMI_OP_G)) ? true : false;
    if (verb.is_stream) {
        if (verb.is_nbi) {
            CUDA_RUNTIME_CHECK_GOTO(cudaEventRecord(cuev, verb.cstrm), status, out);
            CUDA_RUNTIME_CHECK_GOTO(cudaStreamWaitEvent(custrm, cuev, 0), status, out);
            if (is_contig) { /*can include iput,iget in future*/
                CUDA_RUNTIME_CHECK_GOTO(
                    cudaMemcpyAsync(dest->ptr, src->ptr, bytesdesc.nelems * bytesdesc.elembytes,
                                    cudaMemcpyDeviceToDevice, custrm),
                    status, out);
                if (verb.desc == NVSHMEMI_OP_PUT_SIGNAL)
                    nvshmemi_signal_op_on_stream(sig_addr, signal, sig_op, pe, custrm);
            }
        } else { /*!is_nbi*/
            if (is_contig) {
                if (is_single_word) {
                    if (verb.desc == NVSHMEMI_OP_P) {
                        CUDA_RUNTIME_CHECK_GOTO(
                            cudaMemcpyAsync(dest->ptr, src->ptr,
                                            bytesdesc.nelems * bytesdesc.elembytes,
                                            cudaMemcpyHostToDevice, verb.cstrm),
                            status, out);
                    } else { /*!is P*/
                        CUDA_RUNTIME_CHECK_GOTO(
                            cudaMemcpyAsync(dest->ptr, src->ptr,
                                            bytesdesc.nelems * bytesdesc.elembytes,
                                            cudaMemcpyDeviceToHost, verb.cstrm),
                            status, out);
                    }
                } else { /*!is_single_word*/
                    CUDA_RUNTIME_CHECK_GOTO(
                        cudaMemcpyAsync(dest->ptr, src->ptr, bytesdesc.nelems * bytesdesc.elembytes,
                                        cudaMemcpyDeviceToDevice, verb.cstrm),
                        status, out);
                    if (verb.desc == NVSHMEMI_OP_PUT_SIGNAL)
                        nvshmemi_signal_op_on_stream(sig_addr, signal, sig_op, pe, verb.cstrm);
                }
            } else { /*!is_contig*/
                CUDA_RUNTIME_CHECK_GOTO(
                    cudaMemcpy2DAsync(dest->ptr, bytesdesc.deststride * bytesdesc.elembytes,
                                      src->ptr, bytesdesc.srcstride * bytesdesc.elembytes,
                                      bytesdesc.elembytes, bytesdesc.nelems,
                                      cudaMemcpyDeviceToDevice, verb.cstrm),
                    status, out);
            } /*is_contig*/
        }     /*is_nbi*/
    } else {  /*!is_stream*/
        if (verb.is_nbi) {
            if (is_contig) { /*can include iput,iget in future*/
                CUDA_RUNTIME_CHECK_GOTO(
                    cudaMemcpyAsync(dest->ptr, src->ptr, bytesdesc.nelems * bytesdesc.elembytes,
                                    cudaMemcpyDeviceToDevice, custrm),
                    status, out);
            }
        } else { /*!is_nbi*/
            if (is_contig) {
                if (is_single_word) {
                    if (verb.desc == NVSHMEMI_OP_P) {
                        CUDA_RUNTIME_CHECK_GOTO(
                            cudaMemcpyAsync(dest->ptr, src->ptr,
                                            bytesdesc.nelems * bytesdesc.elembytes,
                                            cudaMemcpyHostToDevice, custrm),
                            status, out);
                    } else { /*!is P*/
                        CUDA_RUNTIME_CHECK_GOTO(
                            cudaMemcpyAsync(dest->ptr, src->ptr,
                                            bytesdesc.nelems * bytesdesc.elembytes,
                                            cudaMemcpyDeviceToHost, custrm),
                            status, out);
                    }
                } else { /*!is_single_word*/
                    CUDA_RUNTIME_CHECK_GOTO(
                        cudaMemcpyAsync(dest->ptr, src->ptr, bytesdesc.nelems * bytesdesc.elembytes,
                                        cudaMemcpyDeviceToDevice, custrm),
                        status, out);
                }
            } else { /*!is_contig*/
                CUDA_RUNTIME_CHECK_GOTO(
                    cudaMemcpy2DAsync(dest->ptr, bytesdesc.deststride * bytesdesc.elembytes,
                                      src->ptr, bytesdesc.srcstride * bytesdesc.elembytes,
                                      bytesdesc.elembytes, bytesdesc.nelems,
                                      cudaMemcpyDeviceToDevice, custrm),
                    status, out);
            } /*is_contig*/
            CUDA_RUNTIME_CHECK_GOTO(cudaStreamSynchronize(custrm), status, out);
        } /*is_nbi*/
    }     /*is_stream*/

out:
    return status;
}

static int nvshmemi_p2p_rma_registered(cudaStream_t custrm /* internal stream */, cudaEvent_t cuev,
                                       rma_verb_t verb, rma_memdesc_t *dest, rma_memdesc_t *src,
                                       rma_bytesdesc_t bytesdesc, uint64_t *sig_addr,
                                       uint64_t signal, int sig_op, int pe) {
    cudaStream_t stream_for_op;
    int status = 0;

    if ((!verb.is_stream) || verb.is_nbi) {
        stream_for_op = (cudaStream_t)custrm;
    } else {
        stream_for_op = verb.cstrm;
    }

    if (verb.is_nbi && verb.is_stream) {
        CUDA_RUNTIME_CHECK_GOTO(cudaEventRecord(cuev, verb.cstrm), status, out);
        CUDA_RUNTIME_CHECK_GOTO(cudaStreamWaitEvent(custrm, cuev, 0), status, out);
    }

    if ((bytesdesc.srcstride == 1) && (bytesdesc.deststride == 1)) {
        status = cudaMemcpyAsync(dest->ptr, src->ptr, bytesdesc.nelems * bytesdesc.elembytes,
                                 cudaMemcpyDefault, stream_for_op);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cuMemcpyAsync() failed\n");
        if (verb.desc == NVSHMEMI_OP_PUT_SIGNAL) {
            nvshmemi_signal_op_on_stream(sig_addr, signal, sig_op, pe, stream_for_op);
        }
    } else {
        status = cudaMemcpy2DAsync(dest->ptr, bytesdesc.deststride * bytesdesc.elembytes, src->ptr,
                                   bytesdesc.srcstride * bytesdesc.elembytes, bytesdesc.elembytes,
                                   bytesdesc.nelems, cudaMemcpyDefault, stream_for_op);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cudaMemcpy2DAsync() failed\n");
    }

    if (!verb.is_stream && !verb.is_nbi) {
        CUDA_RUNTIME_CHECK_GOTO(cudaStreamSynchronize(custrm), status, out);
    }

out:
    return status;
}

static inline int nvshmemi_prepare_and_post_mapped_rma(rma_verb_t verb, size_t nelems,
                                                       size_t elembytes, uint64_t *sig_addr,
                                                       uint64_t signal, void *local, void *remote,
                                                       ptrdiff_t lstride, ptrdiff_t rstride,
                                                       int sig_op, int pe) {
    cudaStream_t custrm = nvshmemi_state->custreams[pe % MAX_PEER_STREAMS];
    cudaEvent_t cuev = nvshmemi_state->cuevents[pe % MAX_PEER_STREAMS];
    void *destptr_actual, *srcptr_actual;
    rma_memdesc_t dest, src;
    rma_bytesdesc_t bytesdesc = {(size_t)nelems, (int)elembytes, lstride, rstride};

    if (verb.is_nbi) {
        nvshmemi_state->active_internal_streams[pe % MAX_PEER_STREAMS] = 1;
        nvshmemi_state->used_internal_streams = 1;
    }

    if ((verb.desc == NVSHMEMI_OP_P) || (verb.desc == NVSHMEMI_OP_PUT) ||
        (verb.desc == NVSHMEMI_OP_PUT_SIGNAL)) {
        NVSHMEMU_MAPPED_PTR_TRANSLATE(destptr_actual, remote, pe)
        dest.ptr = (void *)destptr_actual;
        dest.offset = (char *)remote - (char *)(nvshmemi_device_state.heap_base);
        src.ptr = local;
    } else {
        NVSHMEMU_MAPPED_PTR_TRANSLATE(srcptr_actual, remote, pe)
        src.ptr = (void *)srcptr_actual;
        src.offset = (char *)remote - (char *)(nvshmemi_device_state.heap_base);
        dest.ptr = local;
        bytesdesc.srcstride = rstride;
        bytesdesc.deststride = lstride;
    }

    /* when memory is in the heap, we can make some assumptions about memory locations that reduce
     * op latency by ~100ns. */
    if (local >= nvshmemi_device_state.heap_base &&
        (local <
         (void *)((char *)nvshmemi_device_state.heap_base + nvshmemi_device_state.heap_size))) {
        return nvshmemi_p2p_rma_optimized(custrm, cuev, verb, &dest, &src, bytesdesc, sig_addr,
                                          signal, sig_op, pe);
    }
    return nvshmemi_p2p_rma_registered(custrm, cuev, verb, &dest, &src, bytesdesc, sig_addr, signal,
                                       sig_op, pe);
}

static void nvshmemi_prepare_and_post_rma(const char *apiname, nvshmemi_op_t desc, int is_nbi,
                                          int is_stream, void *lptr, void *rptr, ptrdiff_t lstride,
                                          ptrdiff_t rstride, size_t nelems, size_t elembytes,
                                          uint64_t *sig_addr, uint64_t signal, int sig_op, int pe,
                                          cudaStream_t cstrm) {
    rma_verb_t verb = {desc, is_nbi, is_stream, cstrm};
    int t = nvshmemi_state->selected_transport_for_rma[pe];
    rma_bytesdesc_t bytesdesc = {(size_t)nelems, (int)elembytes, 1, 1};
    struct nvshmem_transport *tcurr = nvshmemi_state->transports[t];
    int status = 0;

    /* Mapper Peer */
    if (nvshmemi_state->heap_obj->get_local_pe_base()[pe]) {
        status = nvshmemi_prepare_and_post_mapped_rma(verb, nelems, elembytes, sig_addr, signal,
                                                      lptr, rptr, lstride, rstride, sig_op, pe);
        goto out;
    }

    /* Remote Peer strided ops, G operations, and put_signal are not supported for remote
     * transports. */
    if ((lstride > 1) || (rstride > 1) || desc == NVSHMEMI_OP_G ||
        (verb.desc == NVSHMEMI_OP_PUT_SIGNAL && !is_stream)) {
        status = NVSHMEMX_ERROR_INTERNAL;
        NVSHMEMI_ERROR_PRINT("NOT IMPLEMENTED %s \n", apiname);
        goto out;
    }

    /* IBGDA will not set the RMA transport because it doesn't work on host APIs.
     * On stream will work though.
     */
    if (t < 0 && !verb.is_stream) {
        NVSHMEMI_ERROR_EXIT("[%d] rma not supported on transport to pe: %d \n",
                            nvshmemi_state->mype, pe);
    }

    /* off stream */
    if (!verb.is_stream) {
        if (verb.desc == NVSHMEMI_OP_P) {
            rma_memdesc_t localdesc, remotedesc;
            localdesc.ptr = lptr;
            localdesc.handle = NULL;
            NVSHMEMU_UNMAPPED_PTR_PE_TRANSLATE(remotedesc.ptr, rptr, pe);
            nvshmemi_get_remote_mem_handle(&remotedesc, NULL, rptr, pe, t);
            status = tcurr->host_ops.rma(tcurr, pe, verb, &remotedesc, &localdesc, bytesdesc, 0);
            if (unlikely(status)) {
                NVSHMEMI_ERROR_PRINT("aborting due to error in process_channel_dma\n");
                exit(-1);
            }
        } else {
            nvshmemi_process_multisend_rma(tcurr, t, pe, verb, rptr, lptr, nelems * elembytes, 0);
        }
        goto out;
    }

    /* on stream */
    if (verb.desc == NVSHMEMI_OP_PUT_SIGNAL) {
        verb.desc = NVSHMEMI_OP_PUT;
        void *args[] = {&rptr, &lptr, &bytesdesc, &sig_addr, &signal, &sig_op, &pe, &verb.desc};
        status = nvshmemi_proxy_rma_launcher(args, cstrm, is_nbi, true);
    } else {
        void *args[] = {&rptr, &lptr, &bytesdesc, &pe, &verb.desc};
        status = nvshmemi_proxy_rma_launcher(args, cstrm, is_nbi, false);
    }

out:
    if (status) {
        NVSHMEMI_ERROR_PRINT("aborting due to error in %s \n", apiname);
        exit(-1);
    }
}

/***** Put APIs ******/

#define NVSHMEM_TYPE_PUT(Name, TYPE)                                                              \
    void nvshmem_##Name##_put(TYPE *dest, const TYPE *source, size_t nelems, int pe) {            \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                                   \
        NVSHMEMI_CHECK_INIT_STATUS();                                                             \
        /*INFO(NVSHMEM_P2P, "[%d] bulk put : (remote)dest %p, (local)source %p, %d elements,      \
         * remote PE %d", nvshmemi_state->mype, dest, source, nelems, pe);*/                      \
        nvshmemi_prepare_and_post_rma("nvshmem_" #Name "_put", NVSHMEMI_OP_PUT, NO_NBI, NO_ASYNC, \
                                      (void *)source, (void *)dest, SRC_STRIDE_CONTIG,            \
                                      DEST_STRIDE_CONTIG, nelems, sizeof(TYPE), NULL, 0, -1, pe,  \
                                      NOT_A_CUDA_STREAM);                                         \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEM_TYPE_PUT)
#undef NVSHMEM_TYPE_PUT

#define NVSHMEMX_TYPE_PUT_ON_STREAM(Name, TYPE)                                                   \
    void nvshmemx_##Name##_put_on_stream(TYPE *dest, const TYPE *source, size_t nelems, int pe,   \
                                         cudaStream_t cstrm) {                                    \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                                   \
        NVSHMEMI_CHECK_INIT_STATUS();                                                             \
        nvshmemi_prepare_and_post_rma("nvshmem_" #Name "_put_on_stream", NVSHMEMI_OP_PUT, NO_NBI, \
                                      ASYNC, (void *)source, (void *)dest, SRC_STRIDE_CONTIG,     \
                                      DEST_STRIDE_CONTIG, nelems, sizeof(TYPE), NULL, 0, -1, pe,  \
                                      cstrm);                                                     \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_TYPE_PUT_ON_STREAM)
#undef NVSHMEMX_TYPE_PUT_ON_STREAM

#define NVSHMEM_PUTSIZE(Name, Type)                                                              \
    void nvshmem_put##Name(void *dest, const void *source, size_t nelems, int pe) {              \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                                  \
        NVSHMEMI_CHECK_INIT_STATUS();                                                            \
        nvshmemi_prepare_and_post_rma("nvshmem_put" #Name "", NVSHMEMI_OP_PUT, NO_NBI, NO_ASYNC, \
                                      (void *)source, (void *)dest, SRC_STRIDE_CONTIG,           \
                                      DEST_STRIDE_CONTIG, nelems, sizeof(Type), NULL, 0, -1, pe, \
                                      NOT_A_CUDA_STREAM);                                        \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEM_PUTSIZE)
#undef NVSHMEM_PUTSIZE

#define NVSHMEMX_PUTSIZE_ON_STREAM(Name, Type)                                                    \
    void nvshmemx_put##Name##_on_stream(void *dest, const void *source, size_t nelems, int pe,    \
                                        cudaStream_t cstrm) {                                     \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                                   \
        NVSHMEMI_CHECK_INIT_STATUS();                                                             \
        nvshmemi_prepare_and_post_rma("nvshmemx_put" #Name "_on_stream", NVSHMEMI_OP_PUT, NO_NBI, \
                                      ASYNC, (void *)source, (void *)dest, SRC_STRIDE_CONTIG,     \
                                      DEST_STRIDE_CONTIG, nelems, sizeof(Type), NULL, 0, -1, pe,  \
                                      cstrm);                                                     \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEMX_PUTSIZE_ON_STREAM)
#undef NVSHMEMX_PUTSIZE_ON_STREAM

/*XXX:Should other comm/put APIs call into nvshmem_putmem (suggested by SP)*/
void nvshmem_putmem(void *dest, const void *source, size_t bytes, int pe) {
    NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);
    NVSHMEMI_CHECK_INIT_STATUS();
    INFO(NVSHMEM_P2P,
         "[%d] untyped put : (remote)dest %p, (local)source %p, %zu bytes, remote PE %d",
         nvshmemi_state->mype, dest, source, bytes, pe);
    nvshmemi_prepare_and_post_rma("nvshmem_putmem", NVSHMEMI_OP_PUT, NO_NBI, NO_ASYNC,
                                  (void *)source, (void *)dest, SRC_STRIDE_CONTIG,
                                  DEST_STRIDE_CONTIG, bytes, 1, NULL, 0, -1, pe, NOT_A_CUDA_STREAM);
}

void nvshmemx_putmem_on_stream(void *dest, const void *source, size_t bytes, int pe,
                               cudaStream_t cstrm) {
    NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);
    NVSHMEMI_CHECK_INIT_STATUS();
    INFO(NVSHMEM_P2P,
         "[%d] untyped put : (remote)dest %p, (local)source %p, %zu bytes, remote PE %d",
         nvshmemi_state->mype, dest, source, bytes, pe);
    nvshmemi_prepare_and_post_rma("nvshmemx_putmem_on_stream", NVSHMEMI_OP_PUT, NO_NBI, ASYNC,
                                  (void *)source, (void *)dest, SRC_STRIDE_CONTIG,
                                  DEST_STRIDE_CONTIG, bytes, 1, NULL, 0, -1, pe, cstrm);
}

#define NVSHMEM_TYPE_P(Name, TYPE)                                                            \
    void nvshmem_##Name##_p(TYPE *dest, const TYPE value, int pe) {                           \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                               \
        NVSHMEMI_CHECK_INIT_STATUS();                                                         \
        nvshmemi_prepare_and_post_rma("nvshmem_" #Name "_p", NVSHMEMI_OP_P, NO_NBI, NO_ASYNC, \
                                      (void *)&value, (void *)dest, SRC_STRIDE_CONTIG,        \
                                      DEST_STRIDE_CONTIG, 1, sizeof(TYPE), NULL, 0, -1, pe,   \
                                      NOT_A_CUDA_STREAM);                                     \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEM_TYPE_P)
#undef NVSHMEM_TYPE_P

#define NVSHMEMX_TYPE_P_ON_STREAM(Name, TYPE)                                                      \
    void nvshmemx_##Name##_p_on_stream(TYPE *dest, const TYPE value, int pe, cudaStream_t cstrm) { \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                                    \
        NVSHMEMI_CHECK_INIT_STATUS();                                                              \
        nvshmemi_prepare_and_post_rma("nvshmem_" #Name "_p_on_stream", NVSHMEMI_OP_P, NO_NBI,      \
                                      ASYNC, (void *)&value, (void *)dest, SRC_STRIDE_CONTIG,      \
                                      DEST_STRIDE_CONTIG, 1, sizeof(TYPE), NULL, 0, -1, pe,        \
                                      cstrm);                                                      \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_TYPE_P_ON_STREAM)
#undef NVSHMEMX_TYPE_P_ON_STREAM

#define NVSHMEM_TYPE_IPUT(Name, TYPE)                                                              \
    void nvshmem_##Name##_iput(TYPE *dest, const TYPE *source, ptrdiff_t dst, ptrdiff_t sst,       \
                               size_t nelems, int pe) {                                            \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                                    \
        NVSHMEMI_CHECK_INIT_STATUS();                                                              \
        nvshmemi_prepare_and_post_rma("nvshmem_" #Name "_iput", NVSHMEMI_OP_PUT, NO_NBI, NO_ASYNC, \
                                      (void *)source, (void *)dest, sst, dst, nelems,              \
                                      sizeof(TYPE), NULL, 0, -1, pe, NOT_A_CUDA_STREAM);           \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEM_TYPE_IPUT)
#undef NVSHMEM_TYPE_IPUT

#define NVSHMEMX_TYPE_IPUT_ON_STREAM(Name, TYPE)                                                   \
    void nvshmemx_##Name##_iput_on_stream(TYPE *dest, const TYPE *source, ptrdiff_t dst,           \
                                          ptrdiff_t sst, size_t nelems, int pe,                    \
                                          cudaStream_t cstrm) {                                    \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                                    \
        NVSHMEMI_CHECK_INIT_STATUS();                                                              \
        nvshmemi_prepare_and_post_rma("nvshmem_" #Name "_iput_on_stream", NVSHMEMI_OP_PUT, NO_NBI, \
                                      ASYNC, (void *)source, (void *)dest, sst, dst, nelems,       \
                                      sizeof(TYPE), NULL, 0, -1, pe, cstrm);                       \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_TYPE_IPUT_ON_STREAM)
#undef NVSHMEMX_TYPE_IPUT_ON_STREAM

#define NVSHMEM_IPUTSIZE(Name, Type)                                                              \
    void nvshmem_iput##Name(void *dest, const void *source, ptrdiff_t dst, ptrdiff_t sst,         \
                            size_t nelems, int pe) {                                              \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                                   \
        NVSHMEMI_CHECK_INIT_STATUS();                                                             \
        nvshmemi_prepare_and_post_rma("nvshmem_iput" #Name "", NVSHMEMI_OP_PUT, NO_NBI, NO_ASYNC, \
                                      (void *)source, (void *)dest, sst, dst, nelems,             \
                                      sizeof(Type), NULL, 0, -1, pe, NOT_A_CUDA_STREAM);          \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEM_IPUTSIZE)
#undef NVSHMEM_IPUTSIZE

#define NVSHMEMX_IPUTSIZE_ON_STREAM(Name, Type)                                                   \
    void nvshmemx_iput##Name##_on_stream(void *dest, const void *source, ptrdiff_t dst,           \
                                         ptrdiff_t sst, size_t nelems, int pe,                    \
                                         cudaStream_t cstrm) {                                    \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                                   \
        NVSHMEMI_CHECK_INIT_STATUS();                                                             \
        nvshmemi_prepare_and_post_rma("nvshmem_iput" #Name "_on_stream", NVSHMEMI_OP_PUT, NO_NBI, \
                                      ASYNC, (void *)source, (void *)dest, sst, dst, nelems,      \
                                      sizeof(Type), NULL, 0, -1, pe, cstrm);                      \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEMX_IPUTSIZE_ON_STREAM)
#undef NVSHMEMX_IPUTSIZE_ON_STREAM

#define NVSHMEM_TYPE_PUT_NBI(type, TYPE)                                                           \
    void nvshmem_##type##_put_nbi(TYPE *dest, const TYPE *source, size_t nelems, int pe) {         \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_NONBLOCKING);                                                 \
        NVSHMEMI_CHECK_INIT_STATUS();                                                              \
        nvshmemi_prepare_and_post_rma("nvshmem_" #type "_put_nbi", NVSHMEMI_OP_PUT, NBI, NO_ASYNC, \
                                      (void *)source, (void *)dest, SRC_STRIDE_CONTIG,             \
                                      DEST_STRIDE_CONTIG, nelems, sizeof(TYPE), NULL, 0, -1, pe,   \
                                      NOT_A_CUDA_STREAM);                                          \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEM_TYPE_PUT_NBI)
#undef NVSHMEM_TYPE_PUT_NBI

/* PUT_SIGNAL API */
#define NVSHMEMX_TYPE_PUT_SIGNAL_ON_STREAM(Name, TYPE)                                             \
    void nvshmemx_##Name##_put_signal_on_stream(TYPE *dest, const TYPE *source, size_t nelems,     \
                                                uint64_t *sig_addr, uint64_t signal, int sig_op,   \
                                                int pe, cudaStream_t cstrm) {                      \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                                    \
        NVSHMEMI_CHECK_INIT_STATUS();                                                              \
        nvshmemi_prepare_and_post_rma("nvshmem_" #Name "_put_signal_on_stream",                    \
                                      NVSHMEMI_OP_PUT_SIGNAL, NO_NBI, ASYNC, (void *)source,       \
                                      (void *)dest, SRC_STRIDE_CONTIG, DEST_STRIDE_CONTIG, nelems, \
                                      sizeof(TYPE), sig_addr, signal, sig_op, pe, cstrm);          \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_TYPE_PUT_SIGNAL_ON_STREAM)
#undef NVSHMEMX_TYPE_PUT_SIGNAL_ON_STREAM

#define NVSHMEMX_PUTSIZE_SIGNAL_ON_STREAM(Name, Type)                                              \
    void nvshmemx_put##Name##_signal_on_stream(void *dest, const void *source, size_t nelems,      \
                                               uint64_t *sig_addr, uint64_t signal, int sig_op,    \
                                               int pe, cudaStream_t cstrm) {                       \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                                    \
        NVSHMEMI_CHECK_INIT_STATUS();                                                              \
        nvshmemi_prepare_and_post_rma("nvshmemx_put" #Name "_signal_on_stream",                    \
                                      NVSHMEMI_OP_PUT_SIGNAL, NO_NBI, ASYNC, (void *)source,       \
                                      (void *)dest, SRC_STRIDE_CONTIG, DEST_STRIDE_CONTIG, nelems, \
                                      sizeof(Type), sig_addr, signal, sig_op, pe, cstrm);          \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEMX_PUTSIZE_SIGNAL_ON_STREAM)
#undef NVSHMEMX_PUTSIZE_SIGNAL_ON_STREAM

void nvshmemx_putmem_signal_on_stream(void *dest, const void *source, size_t bytes,
                                      uint64_t *sig_addr, uint64_t signal, int sig_op, int pe,
                                      cudaStream_t cstrm) {
    NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);
    NVSHMEMI_CHECK_INIT_STATUS();
    TRACE(NVSHMEM_P2P,
          "[%d] untyped put : (remote)dest %p, (local)source %p, %zu bytes, remote PE %d",
          nvshmemi_state->mype, dest, source, bytes, pe);
    nvshmemi_prepare_and_post_rma("nvshmemx_putmem_signal_on_stream", NVSHMEMI_OP_PUT_SIGNAL,
                                  NO_NBI, ASYNC, (void *)source, (void *)dest, SRC_STRIDE_CONTIG,
                                  DEST_STRIDE_CONTIG, bytes, 1, sig_addr, signal, sig_op, pe,
                                  cstrm);
}

#define NVSHMEMX_TYPE_PUT_NBI_ON_STREAM(type, TYPE)                                                \
    void nvshmemx_##type##_put_nbi_on_stream(TYPE *dest, const TYPE *source, size_t nelems,        \
                                             int pe, cudaStream_t cstrm) {                         \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_NONBLOCKING);                                                 \
        NVSHMEMI_CHECK_INIT_STATUS();                                                              \
        nvshmemi_prepare_and_post_rma("nvshmem_" #type "_put_nbi_on_stream", NVSHMEMI_OP_PUT, NBI, \
                                      ASYNC, (void *)source, (void *)dest, SRC_STRIDE_CONTIG,      \
                                      DEST_STRIDE_CONTIG, nelems, sizeof(TYPE), NULL, 0, -1, pe,   \
                                      cstrm);                                                      \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_TYPE_PUT_NBI_ON_STREAM)
#undef NVSHMEMX_TYPE_PUT_NBI_ON_STREAM

#define NVSHMEM_PUTSIZE_NBI(Name, Type)                                                           \
    void nvshmem_put##Name##_nbi(void *dest, const void *source, size_t nelems, int pe) {         \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_NONBLOCKING);                                                \
        NVSHMEMI_CHECK_INIT_STATUS();                                                             \
        nvshmemi_prepare_and_post_rma("nvshmem_put" #Name "_nbi", NVSHMEMI_OP_PUT, NBI, NO_ASYNC, \
                                      (void *)source, (void *)dest, SRC_STRIDE_CONTIG,            \
                                      DEST_STRIDE_CONTIG, nelems, sizeof(Type), NULL, 0, -1, pe,  \
                                      NOT_A_CUDA_STREAM);                                         \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEM_PUTSIZE_NBI)
#undef NVSHMEM_PUTSIZE_NBI

#define NVSHMEMX_PUTSIZE_NBI_ON_STREAM(Name, Type)                                                 \
    void nvshmemx_put##Name##_nbi_on_stream(void *dest, const void *source, size_t nelems, int pe, \
                                            cudaStream_t cstrm) {                                  \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_NONBLOCKING);                                                 \
        NVSHMEMI_CHECK_INIT_STATUS();                                                              \
        nvshmemi_prepare_and_post_rma("nvshmem_put" #Name "_nbi_on_stream", NVSHMEMI_OP_PUT, NBI,  \
                                      ASYNC, (void *)source, (void *)dest, SRC_STRIDE_CONTIG,      \
                                      DEST_STRIDE_CONTIG, nelems, sizeof(Type), NULL, 0, -1, pe,   \
                                      cstrm);                                                      \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEMX_PUTSIZE_NBI_ON_STREAM)
#undef NVSHMEMX_PUTSIZE_NBI_ON_STREAM

void nvshmem_putmem_nbi(void *dest, const void *source, size_t bytes, int pe) {
    NVTX_FUNC_RANGE_IN_GROUP(RMA_NONBLOCKING);
    NVSHMEMI_CHECK_INIT_STATUS();
    INFO(NVSHMEM_P2P,
         "[%d] untyped put : (remote)dest %p, (local)source %p, %zu bytes, remote PE %d",
         nvshmemi_state->mype, dest, source, bytes, pe);
    nvshmemi_prepare_and_post_rma("nvshmem_putmem_nbi", NVSHMEMI_OP_PUT, NBI, NO_ASYNC,
                                  (void *)source, (void *)dest, SRC_STRIDE_CONTIG,
                                  DEST_STRIDE_CONTIG, bytes, 1, NULL, 0, -1, pe, NOT_A_CUDA_STREAM);
}

void nvshmemx_putmem_nbi_on_stream(void *dest, const void *source, size_t bytes, int pe,
                                   cudaStream_t cstrm) {
    NVTX_FUNC_RANGE_IN_GROUP(RMA_NONBLOCKING);
    NVSHMEMI_CHECK_INIT_STATUS();
    INFO(NVSHMEM_P2P,
         "[%d] untyped put : (remote)dest %p, (local)source %p, %zu bytes, remote PE %d",
         nvshmemi_state->mype, dest, source, bytes, pe);
    nvshmemi_prepare_and_post_rma("nvshmem_putmem_nbi_on_stream", NVSHMEMI_OP_PUT, NBI, ASYNC,
                                  (void *)source, (void *)dest, SRC_STRIDE_CONTIG,
                                  DEST_STRIDE_CONTIG, bytes, 1, NULL, 0, -1, pe, cstrm);
}

/* PUT_SIGNAL_NBI */
#define NVSHMEMX_TYPE_PUT_SIGNAL_NBI_ON_STREAM(type, TYPE)                                         \
    void nvshmemx_##type##_put_signal_nbi_on_stream(TYPE *dest, const TYPE *source, size_t nelems, \
                                                    uint64_t *sig_addr, uint64_t signal,           \
                                                    int sig_op, int pe, cudaStream_t cstrm) {      \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_NONBLOCKING);                                                 \
        NVSHMEMI_CHECK_INIT_STATUS();                                                              \
        nvshmemi_prepare_and_post_rma("nvshmem_" #type "_put_signal_nbi_on_stream",                \
                                      NVSHMEMI_OP_PUT_SIGNAL, NBI, ASYNC, (void *)source,          \
                                      (void *)dest, SRC_STRIDE_CONTIG, DEST_STRIDE_CONTIG, nelems, \
                                      sizeof(TYPE), sig_addr, signal, sig_op, pe, cstrm);          \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_TYPE_PUT_SIGNAL_NBI_ON_STREAM)
#undef NVSHMEMX_TYPE_PUT_SIGNAL_NBI_ON_STREAM

#define NVSHMEMX_PUTSIZE_SIGNAL_NBI_ON_STREAM(Name, Type)                                          \
    void nvshmemx_put##Name##_signal_nbi_on_stream(void *dest, const void *source, size_t nelems,  \
                                                   uint64_t *sig_addr, uint64_t signal,            \
                                                   int sig_op, int pe, cudaStream_t cstrm) {       \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_NONBLOCKING);                                                 \
        NVSHMEMI_CHECK_INIT_STATUS();                                                              \
        nvshmemi_prepare_and_post_rma("nvshmem_put" #Name "_signal_nbi_on_stream",                 \
                                      NVSHMEMI_OP_PUT_SIGNAL, NBI, ASYNC, (void *)source,          \
                                      (void *)dest, SRC_STRIDE_CONTIG, DEST_STRIDE_CONTIG, nelems, \
                                      sizeof(Type), sig_addr, signal, sig_op, pe, cstrm);          \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEMX_PUTSIZE_SIGNAL_NBI_ON_STREAM)
#undef NVSHMEMX_PUTSIZE_SIGNAL_NBI_ON_STREAM

void nvshmemx_putmem_signal_nbi_on_stream(void *dest, const void *source, size_t bytes,
                                          uint64_t *sig_addr, uint64_t signal, int sig_op, int pe,
                                          cudaStream_t cstrm) {
    NVTX_FUNC_RANGE_IN_GROUP(RMA_NONBLOCKING);
    NVSHMEMI_CHECK_INIT_STATUS();
    INFO(NVSHMEM_P2P,
         "[%d] untyped put : (remote)dest %p, (local)source %p, %zu bytes, remote PE %d",
         nvshmemi_state->mype, dest, source, bytes, pe);
    nvshmemi_prepare_and_post_rma("nvshmem_putmem_signal_nbi_on_stream", NVSHMEMI_OP_PUT_SIGNAL,
                                  NBI, ASYNC, (void *)source, (void *)dest, SRC_STRIDE_CONTIG,
                                  DEST_STRIDE_CONTIG, bytes, 1, sig_addr, signal, sig_op, pe,
                                  cstrm);
}

/***** Get APIs ******/

#define NVSHMEM_TYPE_GET(Name, TYPE)                                                              \
    void nvshmem_##Name##_get(TYPE *dest, const TYPE *source, size_t nelems, int pe) {            \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                                   \
        NVSHMEMI_CHECK_INIT_STATUS();                                                             \
        nvshmemi_prepare_and_post_rma("nvshmem_" #Name "_get", NVSHMEMI_OP_GET, NO_NBI, NO_ASYNC, \
                                      (void *)dest, (void *)source, DEST_STRIDE_CONTIG,           \
                                      SRC_STRIDE_CONTIG, nelems, sizeof(TYPE), NULL, 0, -1, pe,   \
                                      NOT_A_CUDA_STREAM);                                         \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEM_TYPE_GET)
#undef NVSHMEM_TYPE_GET

#define NVSHMEMX_TYPE_GET_ON_STREAM(Name, TYPE)                                                   \
    void nvshmemx_##Name##_get_on_stream(TYPE *dest, const TYPE *source, size_t nelems, int pe,   \
                                         cudaStream_t cstrm) {                                    \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                                   \
        NVSHMEMI_CHECK_INIT_STATUS();                                                             \
        nvshmemi_prepare_and_post_rma("nvshmem_" #Name "_get_on_stream", NVSHMEMI_OP_GET, NO_NBI, \
                                      ASYNC, (void *)dest, (void *)source, DEST_STRIDE_CONTIG,    \
                                      SRC_STRIDE_CONTIG, nelems, sizeof(TYPE), NULL, 0, -1, pe,   \
                                      cstrm);                                                     \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_TYPE_GET_ON_STREAM)
#undef NVSHMEMX_TYPE_GET_ON_STREAM

#define NVSHMEM_GETSIZE(Name, Type)                                                              \
    void nvshmem_get##Name(void *dest, const void *source, size_t nelems, int pe) {              \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                                  \
        NVSHMEMI_CHECK_INIT_STATUS();                                                            \
        nvshmemi_prepare_and_post_rma("nvshmem_get" #Name "", NVSHMEMI_OP_GET, NO_NBI, NO_ASYNC, \
                                      (void *)dest, (void *)source, DEST_STRIDE_CONTIG,          \
                                      SRC_STRIDE_CONTIG, nelems, sizeof(Type), NULL, 0, -1, pe,  \
                                      NOT_A_CUDA_STREAM);                                        \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEM_GETSIZE)
#undef NVSHMEM_GETSIZE

#define NVSHMEMX_GETSIZE_ON_STREAM(Name, Type)                                                    \
    void nvshmemx_get##Name##_on_stream(void *dest, const void *source, size_t nelems, int pe,    \
                                        cudaStream_t cstrm) {                                     \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                                   \
        NVSHMEMI_CHECK_INIT_STATUS();                                                             \
        nvshmemi_prepare_and_post_rma("nvshmemx_get" #Name "_on_stream", NVSHMEMI_OP_GET, NO_NBI, \
                                      ASYNC, (void *)dest, (void *)source, DEST_STRIDE_CONTIG,    \
                                      SRC_STRIDE_CONTIG, nelems, sizeof(Type), NULL, 0, -1, pe,   \
                                      cstrm);                                                     \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEMX_GETSIZE_ON_STREAM)
#undef NVSHMEMX_GETSIZE_ON_STREAM

void nvshmem_getmem(void *dest, const void *source, size_t bytes, int pe) {
    NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);
    NVSHMEMI_CHECK_INIT_STATUS();
    INFO(NVSHMEM_P2P,
         "[%d] untyped get : (local)dest %p, (remote)source %p, %zu bytes, remote PE %d",
         nvshmemi_state->mype, dest, source, bytes, pe);
    nvshmemi_prepare_and_post_rma("nvshmem_getmem", NVSHMEMI_OP_GET, NO_NBI, NO_ASYNC, (void *)dest,
                                  (void *)source, DEST_STRIDE_CONTIG, SRC_STRIDE_CONTIG, bytes, 1,
                                  NULL, 0, -1, pe, NOT_A_CUDA_STREAM);
}

void nvshmemx_getmem_on_stream(void *dest, const void *source, size_t bytes, int pe,
                               cudaStream_t cstrm) {
    NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);
    NVSHMEMI_CHECK_INIT_STATUS();
    INFO(NVSHMEM_P2P,
         "[%d] untyped get : (local)dest %p, (remote)source %p, %zu bytes, remote PE %d",
         nvshmemi_state->mype, dest, source, bytes, pe);
    nvshmemi_prepare_and_post_rma("nvshmemx_getmem_on_stream", NVSHMEMI_OP_GET, NO_NBI, ASYNC,
                                  (void *)dest, (void *)source, DEST_STRIDE_CONTIG,
                                  SRC_STRIDE_CONTIG, bytes, 1, NULL, 0, -1, pe, cstrm);
}

#define NVSHMEM_TYPE_G(Name, TYPE)                                                            \
    TYPE nvshmem_##Name##_g(const TYPE *source, int pe) {                                     \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                               \
        NVSHMEMI_CHECK_INIT_STATUS();                                                         \
        TYPE value;                                                                           \
        INFO(NVSHMEM_P2P, "[%d] single get : (remote)source %p, remote PE %d",                \
             nvshmemi_state->mype, source, pe);                                               \
        nvshmemi_prepare_and_post_rma("nvshmem_" #Name "_g", NVSHMEMI_OP_G, NO_NBI, NO_ASYNC, \
                                      (void *)&value, (void *)source, DEST_STRIDE_CONTIG,     \
                                      SRC_STRIDE_CONTIG, 1, sizeof(TYPE), NULL, 0, -1, pe,    \
                                      NOT_A_CUDA_STREAM);                                     \
        return value;                                                                         \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEM_TYPE_G)
#undef NVSHMEM_TYPE_G

#define NVSHMEMX_TYPE_G_ON_STREAM(Name, TYPE)                                                      \
    TYPE nvshmemx_##Name##_g_on_stream(const TYPE *source, int pe, cudaStream_t cstrm) {           \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                                    \
        NVSHMEMI_CHECK_INIT_STATUS();                                                              \
        TYPE value;                                                                                \
        INFO(NVSHMEM_P2P, "[%d] single get : (remote)source %p, remote PE %d",                     \
             nvshmemi_state->mype, source, pe);                                                    \
        nvshmemi_prepare_and_post_rma("nvshmem_" #Name "_g_on_stream", NVSHMEMI_OP_G, NO_NBI,      \
                                      ASYNC, (void *)&value, (void *)source, DEST_STRIDE_CONTIG,   \
                                      SRC_STRIDE_CONTIG, 1, sizeof(TYPE), NULL, 0, -1, pe, cstrm); \
        return value;                                                                              \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_TYPE_G_ON_STREAM)
#undef NVSHMEMX_TYPE_G_ON_STREAM

#define NVSHMEM_TYPE_IGET(Name, TYPE)                                                              \
    void nvshmem_##Name##_iget(TYPE *dest, const TYPE *source, ptrdiff_t dst, ptrdiff_t sst,       \
                               size_t nelems, int pe) {                                            \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                                    \
        NVSHMEMI_CHECK_INIT_STATUS();                                                              \
        nvshmemi_prepare_and_post_rma("nvshmem_" #Name "_iget", NVSHMEMI_OP_GET, NO_NBI, NO_ASYNC, \
                                      (void *)dest, (void *)source, dst, sst, nelems,              \
                                      sizeof(TYPE), NULL, 0, -1, pe, NOT_A_CUDA_STREAM);           \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEM_TYPE_IGET)
#undef NVSHMEM_TYPE_IGET

#define NVSHMEMX_TYPE_IGET_ON_STREAM(Name, TYPE)                                                   \
    void nvshmemx_##Name##_iget_on_stream(TYPE *dest, const TYPE *source, ptrdiff_t dst,           \
                                          ptrdiff_t sst, size_t nelems, int pe,                    \
                                          cudaStream_t cstrm) {                                    \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                                    \
        NVSHMEMI_CHECK_INIT_STATUS();                                                              \
        nvshmemi_prepare_and_post_rma("nvshmem_" #Name "_iget_on_stream", NVSHMEMI_OP_GET, NO_NBI, \
                                      ASYNC, (void *)dest, (void *)source, dst, sst, nelems,       \
                                      sizeof(TYPE), NULL, 0, -1, pe, cstrm);                       \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_TYPE_IGET_ON_STREAM)
#undef NVSHMEMX_TYPE_IGET_ON_STREAM

#define NVSHMEM_IGETSIZE(Name, Type)                                                              \
    void nvshmem_iget##Name(void *dest, const void *source, ptrdiff_t dst, ptrdiff_t sst,         \
                            size_t nelems, int pe) {                                              \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                                   \
        NVSHMEMI_CHECK_INIT_STATUS();                                                             \
        nvshmemi_prepare_and_post_rma("nvshmem_iget" #Name "", NVSHMEMI_OP_GET, NO_NBI, NO_ASYNC, \
                                      (void *)dest, (void *)source, dst, sst, nelems,             \
                                      sizeof(Type), NULL, 0, -1, pe, NOT_A_CUDA_STREAM);          \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEM_IGETSIZE)
#undef NVSHMEM_IGETSIZE

#define NVSHMEMX_IGETSIZE_ON_STREAM(Name, Type)                                                   \
    void nvshmemx_iget##Name##_on_stream(void *dest, const void *source, ptrdiff_t dst,           \
                                         ptrdiff_t sst, size_t nelems, int pe,                    \
                                         cudaStream_t cstrm) {                                    \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                                   \
        NVSHMEMI_CHECK_INIT_STATUS();                                                             \
        nvshmemi_prepare_and_post_rma("nvshmem_iget" #Name "_on_stream", NVSHMEMI_OP_GET, NO_NBI, \
                                      ASYNC, (void *)dest, (void *)source, dst, sst, nelems,      \
                                      sizeof(Type), NULL, 0, -1, pe, cstrm);                      \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEMX_IGETSIZE_ON_STREAM)
#undef NVSHMEMX_IGETSIZE_ON_STREAM

#define NVSHMEM_TYPE_GET_NBI(type, TYPE)                                                           \
    void nvshmem_##type##_get_nbi(TYPE *dest, const TYPE *source, size_t nelems, int pe) {         \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_NONBLOCKING);                                                 \
        NVSHMEMI_CHECK_INIT_STATUS();                                                              \
        nvshmemi_prepare_and_post_rma("nvshmem_" #type "_get_nbi", NVSHMEMI_OP_GET, NBI, NO_ASYNC, \
                                      (void *)dest, (void *)source, DEST_STRIDE_CONTIG,            \
                                      SRC_STRIDE_CONTIG, nelems, sizeof(TYPE), NULL, 0, -1, pe,    \
                                      NOT_A_CUDA_STREAM);                                          \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEM_TYPE_GET_NBI)
#undef NVSHMEM_TYPE_GET_NBI

#define NVSHMEMX_TYPE_GET_NBI_ON_STREAM(type, TYPE)                                                \
    void nvshmemx_##type##_get_nbi_on_stream(TYPE *dest, const TYPE *source, size_t nelems,        \
                                             int pe, cudaStream_t cstrm) {                         \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_NONBLOCKING);                                                 \
        NVSHMEMI_CHECK_INIT_STATUS();                                                              \
        nvshmemi_prepare_and_post_rma("nvshmem_" #type "_get_nbi_on_stream", NVSHMEMI_OP_GET, NBI, \
                                      ASYNC, (void *)dest, (void *)source, DEST_STRIDE_CONTIG,     \
                                      SRC_STRIDE_CONTIG, nelems, sizeof(TYPE), NULL, 0, -1, pe,    \
                                      cstrm);                                                      \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_TYPE_GET_NBI_ON_STREAM)
#undef NVSHMEMX_TYPE_GET_NBI_ON_STREAM

#define NVSHMEM_GETSIZE_NBI(Name, Type)                                                           \
    void nvshmem_get##Name##_nbi(void *dest, const void *source, size_t nelems, int pe) {         \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_NONBLOCKING);                                                \
        NVSHMEMI_CHECK_INIT_STATUS();                                                             \
        nvshmemi_prepare_and_post_rma("nvshmem_get" #Name "_nbi", NVSHMEMI_OP_GET, NBI, NO_ASYNC, \
                                      (void *)dest, (void *)source, DEST_STRIDE_CONTIG,           \
                                      SRC_STRIDE_CONTIG, nelems, sizeof(Type), NULL, 0, -1, pe,   \
                                      NOT_A_CUDA_STREAM);                                         \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEM_GETSIZE_NBI)
#undef NVSHMEM_GETSIZE_NBI

#define NVSHMEMX_GETSIZE_NBI_ON_STREAM(Name, Type)                                                 \
    void nvshmemx_get##Name##_nbi_on_stream(void *dest, const void *source, size_t nelems, int pe, \
                                            cudaStream_t cstrm) {                                  \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_NONBLOCKING);                                                 \
        NVSHMEMI_CHECK_INIT_STATUS();                                                              \
        nvshmemi_prepare_and_post_rma("nvshmem_get" #Name "_nbi_on_stream", NVSHMEMI_OP_GET, NBI,  \
                                      ASYNC, (void *)dest, (void *)source, DEST_STRIDE_CONTIG,     \
                                      SRC_STRIDE_CONTIG, nelems, sizeof(Type), NULL, 0, -1, pe,    \
                                      cstrm);                                                      \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEMX_GETSIZE_NBI_ON_STREAM)
#undef NVSHMEMX_GETSIZE_NBI_ON_STREAM

void nvshmem_getmem_nbi(void *dest, const void *source, size_t bytes, int pe) {
    NVTX_FUNC_RANGE_IN_GROUP(RMA_NONBLOCKING);
    NVSHMEMI_CHECK_INIT_STATUS();
    INFO(NVSHMEM_P2P,
         "[%d] untyped get : (local)dest %p, (remote)source %p, %zu bytes, remote PE %d",
         nvshmemi_state->mype, dest, source, bytes, pe);
    nvshmemi_prepare_and_post_rma("nvshmem_getmem_nbi", NVSHMEMI_OP_GET, NBI, NO_ASYNC,
                                  (void *)dest, (void *)source, DEST_STRIDE_CONTIG,
                                  SRC_STRIDE_CONTIG, bytes, 1, NULL, 0, -1, pe, NOT_A_CUDA_STREAM);
}

void nvshmemx_getmem_nbi_on_stream(void *dest, const void *source, size_t bytes, int pe,
                                   cudaStream_t cstrm) {
    NVTX_FUNC_RANGE_IN_GROUP(RMA_NONBLOCKING);
    NVSHMEMI_CHECK_INIT_STATUS();
    INFO(NVSHMEM_P2P,
         "[%d] untyped get : (local)dest %p, (remote)source %p, %zu bytes, remote PE %d",
         nvshmemi_state->mype, dest, source, bytes, pe);
    nvshmemi_prepare_and_post_rma("nvshmem_getmem_nbi_on_stream", NVSHMEMI_OP_GET, NBI, ASYNC,
                                  (void *)dest, (void *)source, DEST_STRIDE_CONTIG,
                                  SRC_STRIDE_CONTIG, bytes, 1, NULL, 0, -1, pe, cstrm);
}
