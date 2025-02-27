/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _IBRC_H
#define _IBRC_H

#define NVSHMEMT_IBDEVX_DBSIZE 8
/* 64 bytes per WQE BB shift = log2(64) for easy multiplication. */
#define NVSHMEMT_IBDEVX_WQE_BB_SHIFT 6

/* Atomic mode for our transport */
#define NVSHMEMT_IBDEVX_MLX5_QPC_ATOMIC_MODE_UP_TO_64B 0x3

#define NVSHMEMT_IBDEVX_MLX5_SEND_WQE_DS 0x10

/* Indicates to DEVX that we should be using an SRQ. */
#define NVSHMEMT_IBDEVX_SRQ_TYPE_VALUE 0x1

/* Enables remote read/write/atomic access for a QP */
#define NVSHMEMT_IBDEVX_INIT2R2R_PARAM_MASK 0xE

/* Important byte masks. */
#define NVSHMEMT_IBDEVX_MASK_UPPER_BYTE_32 0x00FFFFFF
#define NVSHMEMT_IBDEVX_MASK_LOWER_3_BYTES_32 0xFF000000

/* OPMOD Constants for AMOs. */
#define NVSHMEMT_IBDEVX_4_BYTE_EXT_AMO_OPMOD 0x08000000
#define NVSHMEMT_IBDEVX_8_BYTE_EXT_AMO_OPMOD 0x09000000

#endif
