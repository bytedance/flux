/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "transport_mlx5_common.h"
#include <infiniband/mlx5dv.h>
#include <stdint.h>

#include "mlx5_ifc.h"
#include "non_abi/nvshmemx_error.h"

bool nvshmemt_ib_common_query_mlx5_caps(struct ibv_context *context) {
    int status;
    uint8_t cmd_cap_in[DEVX_ST_SZ_BYTES(query_hca_cap_in)] = {
        0,
    };
    uint8_t cmd_cap_out[DEVX_ST_SZ_BYTES(query_hca_cap_out)] = {
        0,
    };

    DEVX_SET(query_hca_cap_in, cmd_cap_in, opcode, MLX5_CMD_OP_QUERY_HCA_CAP);
    DEVX_SET(
        query_hca_cap_in, cmd_cap_in, op_mod,
        MLX5_SET_HCA_CAP_OP_MOD_GENERAL_DEVICE | (MLX5_CAP_GENERAL << 1) | HCA_CAP_OPMOD_GET_CUR);

    status = mlx5dv_devx_general_cmd(context, cmd_cap_in, sizeof(cmd_cap_in), cmd_cap_out,
                                     sizeof(cmd_cap_out));

    if (status == 0) {
        return true;
    }
    return false;
}

int nvshmemt_ib_common_query_endianness_conversion_size(uint32_t *endianness_mode,
                                                        struct ibv_context *context) {
    void *cap;
    uint32_t amo_endianness_mode;
    uint32_t amo_endianness_mode_support;
    int status;
    uint8_t cmd_cap_in[DEVX_ST_SZ_BYTES(query_hca_cap_in)] = {
        0,
    };
    uint8_t cmd_cap_out[DEVX_ST_SZ_BYTES(query_hca_cap_out)] = {
        0,
    };

    DEVX_SET(query_hca_cap_in, cmd_cap_in, opcode, MLX5_CMD_OP_QUERY_HCA_CAP);
    DEVX_SET(
        query_hca_cap_in, cmd_cap_in, op_mod,
        MLX5_SET_HCA_CAP_OP_MOD_GENERAL_DEVICE | (MLX5_CAP_ATOMIC << 1) | HCA_CAP_OPMOD_GET_CUR);
    status = mlx5dv_devx_general_cmd(context, cmd_cap_in, sizeof(cmd_cap_in), cmd_cap_out,
                                     sizeof(cmd_cap_out));
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "mlx5dv_devx_general_cmd for atomic caps failed.\n");

    cap = DEVX_ADDR_OF(query_hca_cap_out, cmd_cap_out, capability);
    amo_endianness_mode_support =
        DEVX_GET(atomic_caps, cap, supported_atomic_req_8B_endianness_mode_1);
    amo_endianness_mode = DEVX_GET(atomic_caps, cap, atomic_req_8B_endianness_mode);
    if (amo_endianness_mode_support && amo_endianness_mode) {
        *endianness_mode = 8;
    } else {
        *endianness_mode = UINT32_MAX;
    }

out:
    return status;
}

int nvshmemt_ib_common_check_nic_ext_atomic_support(struct ibv_context *context) {
    int status = 0;

    uint8_t cmd_cap_in[DEVX_ST_SZ_BYTES(query_hca_cap_in)] = {
        0,
    };
    uint8_t cmd_cap_out[DEVX_ST_SZ_BYTES(query_hca_cap_out)] = {
        0,
    };
    void *cap;
    uint16_t atomic_operations;
    uint16_t atomic_size_qp;
    uint16_t atomic_size_dc;

    DEVX_SET(query_hca_cap_in, cmd_cap_in, opcode, MLX5_CMD_OP_QUERY_HCA_CAP);
    DEVX_SET(
        query_hca_cap_in, cmd_cap_in, op_mod,
        MLX5_SET_HCA_CAP_OP_MOD_GENERAL_DEVICE | (MLX5_CAP_ATOMIC << 1) | HCA_CAP_OPMOD_GET_CUR);

    status = mlx5dv_devx_general_cmd(context, cmd_cap_in, sizeof(cmd_cap_in), cmd_cap_out,
                                     sizeof(cmd_cap_out));
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "mlx5dv_devx_general_cmd for hca cap failed.\n");

    cap = DEVX_ADDR_OF(query_hca_cap_out, cmd_cap_out, capability.atomic_caps);
    atomic_operations = DEVX_GET(atomic_caps, cap, atomic_operations);
    atomic_size_qp = DEVX_GET(atomic_caps, cap, atomic_size_qp);
    atomic_size_dc = DEVX_GET(atomic_caps, cap, atomic_size_dc);

    if (!(atomic_operations & MLX5_ATOMIC_CAP_OP_SUPPORT_CAS)) {
        NVSHMEMI_WARN_PRINT("device does not support atomic compared and swap\n");
        status = NVSHMEMX_ERROR_NOT_SUPPORTED;
        goto out;
    }

    if (!(atomic_operations & MLX5_ATOMIC_CAP_OP_SUPPORT_FA)) {
        NVSHMEMI_WARN_PRINT("device does not support atomic fetch and add\n");
        status = NVSHMEMX_ERROR_NOT_SUPPORTED;
        goto out;
    }

    if (!(atomic_operations & MLX5_ATOMIC_CAP_OP_SUPPORT_MASKED_CAS)) {
        NVSHMEMI_WARN_PRINT("device does not support atomic masked compared and swap\n");
        status = NVSHMEMX_ERROR_NOT_SUPPORTED;
        goto out;
    }

    if (!(atomic_operations & MLX5_ATOMIC_CAP_OP_SUPPORT_MASKED_FA)) {
        NVSHMEMI_WARN_PRINT("device does not support atomic masked fetch and add\n");
        status = NVSHMEMX_ERROR_NOT_SUPPORTED;
        goto out;
    }

    if ((!(atomic_size_qp & MLX5_ATOMIC_CAP_SIZE_SUPPORT_4B)) ||
        (!(atomic_size_dc & MLX5_ATOMIC_CAP_SIZE_SUPPORT_4B))) {
        NVSHMEMI_WARN_PRINT("device does not support 4B atomics\n");
        status = NVSHMEMX_ERROR_NOT_SUPPORTED;
        goto out;
    }

    if ((!(atomic_size_qp & MLX5_ATOMIC_CAP_SIZE_SUPPORT_8B)) ||
        (!(atomic_size_dc & MLX5_ATOMIC_CAP_SIZE_SUPPORT_8B))) {
        NVSHMEMI_WARN_PRINT("device does not support 4B atomics\n");
        status = NVSHMEMX_ERROR_NOT_SUPPORTED;
        goto out;
    }

out:
    return status;
}
