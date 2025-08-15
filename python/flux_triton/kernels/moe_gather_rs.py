################################################################################
#
# Copyright 2025 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################
from typing import Dict, List

import torch
import triton
import triton.language as tl
from flux_triton.extra import __syncthreads, atomic_add, tid
from flux_triton.tools.compile_aot import aot_compile_spaces


def get_tune_config(
    M: int,  # TODO(houqi.1993) small M and large M should use different M
    N: int,
    K: int,
    world_size: int,
    trans_a: bool,
    trans_b: bool,
    dtype: torch.dtype,
):
    if dtype in [torch.float16, torch.bfloat16]:
        return {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "num_warps": 4,
            "num_stages": 4,
        }
    if dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        assert not trans_a and trans_b
        return {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "num_warps": 4,
            "num_stages": 4,
        }
    if dtype in [torch.int8]:
        assert not trans_a and trans_b
        return {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "num_warps": 4,
            "num_stages": 4,
        }
    assert False, f"not supported dtypes: {dtype}"


signature = (
    "*{input_dtype}:16, *{input_dtype}:16, *{output_dtype}:16, "  # A/B/C
    "*fp32:16, *fp32:16, *fp32:16, "  # bias/input_scale/weight_scale/output_scale
    "*i32:16, *i32:16, "  # gather_a_ptr/expert_idx
    "i32, i32:16, i32:16, "  # M/N/K
    "i32, "  # E,  # not used
    "i32, "  # M_valid
    "i32:16, i32:1, i32:16, i32:1, i32:16, i32:16, i32:1, "  # A/B/C strides. with transpose_weight=True
    "*i32:16, "  # flags,
    "%N_SPLIT, "
    "%BLOCK_SIZE_M, "
    "%BLOCK_SIZE_N, "
    "%BLOCK_SIZE_K"
)
_grid = [
    "((M + %BLOCK_SIZE_M - 1) / %BLOCK_SIZE_M) * ((N + %BLOCK_SIZE_N - 1) / %BLOCK_SIZE_N)",
    "1",
    "1",
]
N_SPLITS = [1, 2, 4, 5, 10]


def _with_N_splits(configs, N_SPLITS):
    config_new = []
    for config in configs:
        for N_SPLIT in N_SPLITS:
            c = config.copy()
            c.update({"N_SPLIT": N_SPLIT})
            config_new.append(c)
    return config_new


@aot_compile_spaces(
    {
        "moe_gather_rs_grouped_gemm_fp16": {
            "signature": signature.format(input_dtype="fp16", output_dtype="fp16"),
            "grid": _grid,
            "triton_algo_infos": _with_N_splits(
                [
                    {
                        "BLOCK_SIZE_M": 128,
                        "BLOCK_SIZE_N": 128,
                        "BLOCK_SIZE_K": 64,
                        "num_warps": 4,
                        "num_stages": 4,
                    },
                ],
                N_SPLITS,
            ),
        },
        "moe_gather_rs_grouped_gemm_bf16": {
            "signature": signature.format(input_dtype="bf16", output_dtype="bf16"),
            "grid": _grid,
            "triton_algo_infos": _with_N_splits(
                [
                    {
                        "BLOCK_SIZE_M": 128,
                        "BLOCK_SIZE_N": 128,
                        "BLOCK_SIZE_K": 64,
                        "num_warps": 4,
                        "num_stages": 4,
                    },
                ],
                N_SPLITS,
            ),
        },
        "moe_gather_rs_grouped_gemm_s8": {
            "signature": signature.format(input_dtype="i8", output_dtype="bf16"),
            "grid": _grid,
            "triton_algo_infos": _with_N_splits(
                [
                    {
                        "BLOCK_SIZE_M": 64,
                        "BLOCK_SIZE_N": 128,
                        "BLOCK_SIZE_K": 64,
                        "num_warps": 4,
                        "num_stages": 4,
                    },
                ],
                N_SPLITS,
            ),
        },
    }
)
@triton.jit
def moe_gather_rs_grouped_gemm_kernel(
    A,
    B,
    C,
    input_scale,
    weight_scale,
    output_vec_scale,
    gather_index,
    expert_index,
    M,
    N,
    K,
    E,  # not used
    M_valid,
    A_stride_m,
    A_stride_k,
    B_stride_e,
    B_stride_k,
    B_stride_n,
    C_stride_m,
    C_stride_n,
    flags,
    N_SPLIT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_block_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_block_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_block_n_per_split = num_block_n // N_SPLIT
    num_blocks_per_split = num_block_n_per_split * num_block_m
    split_id = pid // num_blocks_per_split
    pid_rem = pid - split_id * num_blocks_per_split
    pid_m = pid_rem // num_block_n_per_split
    pid_n = pid_rem % num_block_n_per_split + split_id * num_block_n_per_split

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_gather_a = tl.load(gather_index + offs_token_id)
    token_mask = offs_gather_a < M_valid

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + offs_gather_a[:, None] * A_stride_m + offs_k[None, :] * A_stride_k

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_be = tl.load(expert_index + pid_m)
    b_ptrs = B + offs_be * B_stride_e + offs_k[:, None] * B_stride_k + offs_bn[None, :] * B_stride_n

    if A.dtype.element_ty == tl.int8:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K))
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K))

        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * A_stride_k
        b_ptrs += BLOCK_SIZE_K * B_stride_k

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + offs_gather_a[:, None] * C_stride_m + offs_cn[None, :] * C_stride_n
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)

    if A.dtype.element_ty == tl.int8:
        accumulator = (
            accumulator
            * tl.load(output_vec_scale + offs_gather_a, mask=token_mask)[:, None]
            * tl.load(input_scale + offs_gather_a, mask=token_mask)[:, None]  # [1] input_scale
            * tl.load(weight_scale + offs_be * N + offs_cn, mask=offs_cn < N)[
                None, :
            ]  # per-N weight
        ).to(tl.bfloat16)
    else:  # TODO(houqi.1993) not support FP8 now
        accumulator = (
            accumulator
            * tl.load(output_vec_scale + offs_gather_a, mask=token_mask)[:, None]
            * tl.load(input_scale)
            * tl.load(weight_scale + offs_be)
        )
        accumulator = accumulator.to(A.dtype.element_ty)

    tl.store(c_ptrs, accumulator, mask=c_mask)

    thread_idx = tid(axis=0)
    __syncthreads()
    if thread_idx == 0:
        count = atomic_add(flags + split_id + N_SPLIT, 1, semantic="release", scope="gpu")
        if count == num_block_m * num_block_n // N_SPLIT - 1:
            atomic_add(flags + split_id, 1, semantic="release", scope="sys")


@triton.jit
def moe_gather_rs_grouped_gemm_with_groups_kernel(
    A,  # for passing A.dtype
    As,  # List[torch.Tensor] with shape (M_valid, K)
    Bs,  # List[torch.Tensor] with shape (K, N)
    Cs,  # List[torch.Tensor] with shape (M_valid, N)
    input_scales,  # List[torch.Tensor] with shape (1,) of torch.float32
    weight_scales,  # List[torch.Tensor] with shape (E,) of torch.float32
    output_vec_scales,  # List[torch.Tensor] with shape (M_valid // num_groups,) of torch.float32
    gather_index,  # List[torch.Tensor] with shape (M_valid // num_groups) of torch.int32
    expert_index,
    M,  # pad(M_valid)
    N,
    K,
    E,  # not used
    M_valid,  # before paded.
    A_stride_m,
    A_stride_k,
    B_stride_e,
    B_stride_k,
    B_stride_n,
    C_stride_m,
    C_stride_n,
    flags,
    N_SPLIT: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_block_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_block_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_block_n_per_split = num_block_n // N_SPLIT
    num_blocks_per_split = num_block_n_per_split * num_block_m * NUM_GROUPS
    split_id = pid // num_blocks_per_split
    pid_rem = pid - split_id * num_blocks_per_split
    pid_m_g = pid_rem // num_block_n_per_split
    pid_n = pid_rem % num_block_n_per_split + split_id * num_block_n_per_split
    pid_m = pid_m_g % num_block_m
    gid = pid_m_g // num_block_m

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_gather_a = tl.load(gather_index + offs_token_id)
    row_mask = offs_gather_a < M_valid

    A = tl.multiple_of(tl.load(As + gid).to(tl.pointer_type(A.dtype.element_ty)), 16)
    B = tl.multiple_of(tl.load(Bs + gid).to(tl.pointer_type(A.dtype.element_ty)), 16)
    if A.dtype.element_ty == tl.int8:
        C = tl.multiple_of(tl.load(Cs + gid).to(tl.pointer_type(tl.bfloat16)), 16)
    else:
        C = tl.multiple_of(tl.load(Cs + gid).to(tl.pointer_type(A.dtype.element_ty)), 16)
    input_scale = tl.load(input_scales + gid).to(tl.pointer_type(tl.float32))
    weight_scale = tl.load(weight_scales + gid).to(tl.pointer_type(tl.float32))
    output_vec_scale = tl.load(output_vec_scales + gid).to(tl.pointer_type(tl.float32))

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + offs_gather_a[:, None] * A_stride_m + offs_k[None, :] * A_stride_k

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_be = tl.load(expert_index + pid_m)
    b_ptrs = B + offs_be * B_stride_e + offs_k[:, None] * B_stride_k + offs_bn[None, :] * B_stride_n

    if A.dtype.element_ty == tl.int8:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=row_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K))
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K))

        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * A_stride_k
        b_ptrs += BLOCK_SIZE_K * B_stride_k

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + offs_gather_a[:, None] * C_stride_m + offs_cn[None, :] * C_stride_n
    c_mask = row_mask[:, None] & (offs_cn[None, :] < N)
    if A.dtype.element_ty == tl.int8:
        accumulator = (
            accumulator
            * tl.load(output_vec_scale + offs_gather_a, mask=row_mask)[:, None]
            * tl.load(input_scale + offs_gather_a, mask=row_mask)[:, None]
            * tl.load(weight_scale + offs_be * N + offs_cn, mask=offs_cn < N)[None, :]
        ).to(tl.bfloat16)
    else:  # TODO(houqi.1993) not support FP8 now
        accumulator = (
            accumulator
            * tl.load(output_vec_scale + offs_gather_a, mask=row_mask)[:, None]
            * tl.load(input_scale)
            * tl.load(weight_scale + offs_be)
        ).to(A.dtype.element_ty)

    tl.store(c_ptrs, accumulator, mask=c_mask)

    thread_idx = tid(axis=0)
    __syncthreads()
    if thread_idx == 0:
        count = atomic_add(flags + split_id + N_SPLIT, 1, semantic="release", scope="gpu")
        if count == num_blocks_per_split - 1:
            atomic_add(flags + split_id, 1, semantic="release", scope="sys")


def run_moe_gather_rs_grouped_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    output_vec_scale: torch.Tensor,
    gather_A_index: torch.Tensor,
    expert_idx: torch.Tensor,
    M: int,
    N: int,
    K: int,
    E: int,
    num_valid_tokens: int,
    barrier: torch.Tensor,
    N_SPLIT: int,
    config: Dict[str, int],
):
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    moe_gather_rs_grouped_gemm_kernel[grid](
        A,
        B,
        C,
        input_scale,
        weight_scale,
        output_vec_scale,
        gather_A_index,
        expert_idx,
        M,
        N,
        K,
        E,
        num_valid_tokens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        C.stride(0),
        C.stride(1),
        barrier,
        N_SPLIT,
        **config,
    )
    return C


def run_moe_gather_rs_grouped_gemm_with_groups(
    A_list: List[torch.Tensor],
    B_list: List[torch.Tensor],
    C_list: List[torch.Tensor],
    input_scale_list: List[torch.Tensor],
    weight_scale_list: List[torch.Tensor],
    output_vec_scale_list: List[torch.Tensor],
    gather_A_index: torch.Tensor,
    expert_idx: torch.Tensor,
    M: int,
    N: int,
    K: int,
    E: int,
    num_valid_tokens: int,
    barrier: torch.Tensor,
    N_SPLIT: int,
    config: Dict[str, int],
):
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"])
        * META["NUM_GROUPS"]
        * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    L = len(A_list)
    A, B, C = A_list[0], B_list[0], C_list[0]
    import numpy as np

    device_ptrs = torch.from_numpy(
        np.array(
            [A.data_ptr() for A in A_list]
            + [B.data_ptr() for B in B_list]
            + [C.data_ptr() for C in C_list]
            + [input_scale.data_ptr() for input_scale in input_scale_list]
            + [weight_scale.data_ptr() for weight_scale in weight_scale_list]
            + [output_vec_scale.data_ptr() for output_vec_scale in output_vec_scale_list],
            dtype=np.int64,
        )
    ).cuda()
    A_ptrs = device_ptrs[:L]
    B_ptrs = device_ptrs[L : L + L]
    C_ptrs = device_ptrs[L * 2 : L * 2 + L]
    input_scale_ptrs = device_ptrs[L * 3 : L * 3 + L]
    weight_scale_ptrs = device_ptrs[L * 4 : L * 4 + L]
    output_vec_scale_ptrs = device_ptrs[L * 5 : L * 5 + L]
    moe_gather_rs_grouped_gemm_with_groups_kernel[grid](
        A,
        A_ptrs,  # A ptrs
        B_ptrs,  # B ptrs
        C_ptrs,
        input_scale_ptrs,  # input_scale ptrs
        weight_scale_ptrs,  # weight_scale ptrs
        output_vec_scale_ptrs,  # output_vec_scale ptrs
        gather_A_index,
        expert_idx,
        M,
        N,
        K,
        E,
        num_valid_tokens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        C.stride(0),
        C.stride(1),
        barrier,
        N_SPLIT=N_SPLIT,
        NUM_GROUPS=L,
        **config,
    )
    return C_list
