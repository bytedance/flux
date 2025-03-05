/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_SOCKET_DEBUG_H_
#define NCCL_SOCKET_DEBUG_H_

#include "bootstrap_util.h"  // for BOOTSTRAP_DEBUG_PRINT, BOOTSTRAP_ERROR_P...

extern thread_local int ncclDebugNoWarn;
typedef enum {NCCL_LOG_NONE=0, NCCL_LOG_VERSION=1, NCCL_LOG_WARN=2, NCCL_LOG_INFO=3, NCCL_LOG_ABORT=4, NCCL_LOG_TRACE=5} ncclDebugLogLevel;
typedef enum {NCCL_INIT=1, NCCL_COLL=2, NCCL_P2P=4, NCCL_SHM=8, NCCL_NET=16, NCCL_GRAPH=32, NCCL_TUNING=64, NCCL_ENV=128, NCCL_ALLOC=256, NCCL_CALL=512, NCCL_PROXY=1024, NCCL_NVLS=2048, NCCL_ALL=~0} ncclDebugLogSubSys;

#define WARN(...) BOOTSTRAP_ERROR_PRINT(__VA_ARGS__) 
#define INFO(FLAGS, ...) BOOTSTRAP_DEBUG_PRINT(__VA_ARGS__)
#define TRACE_CALL(...) BOOTSTRAP_DEBUG_PRINT(__VA_ARGS__)
#define TRACE(...) //nop

#endif