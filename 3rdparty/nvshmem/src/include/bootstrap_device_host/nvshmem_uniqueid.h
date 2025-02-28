
#ifndef _NVSHMEM_UNIQUEID_H_
#define _NVSHMEM_UNIQUEID_H_

#define UNIQUEID_PADDING 124
#define UNIQUEID_ARGS_INVALID -1
#if not defined __CUDACC_RTC__
#include <stddef.h>  // for NULL
#define NVSHMEMX_UNIQUEID_INITIALIZER                          \
    {                                                          \
        (1 << 16) + sizeof(nvshmemx_uniqueid_t), /* version */ \
        {                                                      \
            0                                                  \
        }                                                      \
    }

#define NVSHMEMX_UNIQUEID_ARGS_INITIALIZER                          \
    {                                                               \
        (1 << 16) + sizeof(nvshmemx_uniqueid_args_t), /* version */ \
            NULL,                                     /* id */      \
            UNIQUEID_ARGS_INVALID,                    /* myrank */  \
            UNIQUEID_ARGS_INVALID                     /* nranks */  \
    }
#endif
typedef struct {
    int version;
    char internal[UNIQUEID_PADDING];
} nvshmemx_uniqueid_v1;
static_assert(sizeof(nvshmemx_uniqueid_v1) == 128, "uniqueid_v1 must be 128 bytes.");

typedef nvshmemx_uniqueid_v1 nvshmemx_uniqueid_t;

typedef struct {
    int version;
    nvshmemx_uniqueid_v1 *id;
    int myrank;
    int nranks;
} nvshmemx_uniqueid_args_v1;
static_assert(sizeof(nvshmemx_uniqueid_args_v1) == 24, "uniqueid_args_v1 must be 24 bytes.");

typedef nvshmemx_uniqueid_args_v1 nvshmemx_uniqueid_args_t;

#endif
