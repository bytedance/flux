#ifndef _NVSHMEM_DEVICE_MACROS_H_
#define _NVSHMEM_DEVICE_MACROS_H_

#include "non_abi/nvshmem_build_options.h"  // IWYU pragma: keep

#ifdef NVSHMEM_ENABLE_ALL_DEVICE_INLINING
#define NVSHMEMI_DEVICE_INLINE inline
#else
#define NVSHMEMI_DEVICE_INLINE __noinline__
#endif

#endif