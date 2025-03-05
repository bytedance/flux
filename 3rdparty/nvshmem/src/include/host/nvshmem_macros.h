
#ifndef _NVSHMEM_MACROS_H_
#define _NVSHMEM_MACROS_H_

#include <cuda_runtime.h>

#ifdef __CUDA_ARCH__
#ifdef NVSHMEMI_HOST_ONLY
#define NVSHMEMI_HOSTDEVICE_PREFIX __host__
#else
#ifdef NVSHMEMI_DEVICE_ONLY
#define NVSHMEMI_HOSTDEVICE_PREFIX __device__
#else
#define NVSHMEMI_HOSTDEVICE_PREFIX __host__ __device__
#endif
#endif
#else
#define NVSHMEMI_HOSTDEVICE_PREFIX
#endif
#endif