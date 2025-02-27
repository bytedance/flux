/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See COPYRIGHT.txt for license information
 */

#ifndef _NVSHMEM_H_
#define _NVSHMEM_H_

#include "non_abi/nvshmem_build_options.h"
/* NVRTC only compiles device code. Leave out host headers */
#if not defined __CUDACC_RTC__
#include "host/nvshmem_api.h"
#include "host/nvshmemx_api.h"
#endif
#include "device/nvshmem_defines.h"
#include "device/nvshmem_coll_defines.cuh"
#include "device/nvshmemx_defines.h"

#endif
