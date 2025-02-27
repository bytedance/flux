/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _TRANSPORT_MLX5_COMMON_H
#define _TRANSPORT_MLX5_COMMON_H

#include <stdint.h>  // IWYU pragma: keep
// IWYU pragma: no_include <bits/stdint-uintn.h>

bool nvshmemt_ib_common_query_mlx5_caps(struct ibv_context *context);
int nvshmemt_ib_common_query_endianness_conversion_size(uint32_t *endianness_mode,
                                                        struct ibv_context *context);
int nvshmemt_ib_common_check_nic_ext_atomic_support(struct ibv_context *context);

/* These values are not defined on all systems.
 * However, they can be traced back to a kernel enum with
 * these values.
 */
#ifndef MLX5DV_UAR_ALLOC_TYPE_BF
#define MLX5DV_UAR_ALLOC_TYPE_BF 0x0
#endif

#ifndef MLX5DV_UAR_ALLOC_TYPE_NC
#define MLX5DV_UAR_ALLOC_TYPE_NC 0x1
#endif

enum {
    MLX5_ATOMIC_CAP_OP_SUPPORT_CAS = 0x1,
    MLX5_ATOMIC_CAP_OP_SUPPORT_FA = 0x2,
    MLX5_ATOMIC_CAP_OP_SUPPORT_MASKED_CAS = 0x4,
    MLX5_ATOMIC_CAP_OP_SUPPORT_MASKED_FA = 0x8,
};

enum {
    MLX5_ATOMIC_CAP_SIZE_SUPPORT_4B = 0x4,
    MLX5_ATOMIC_CAP_SIZE_SUPPORT_8B = 0x8,
};

#endif
