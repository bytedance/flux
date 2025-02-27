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
from typing import Dict, Optional

import torch
import triton
import triton.language as tl
from flux_triton.extra import __syncthreads, ld_acquire, tid
from flux_triton.tools.compile_aot import aot_compile_spaces

signature = (
    "*{input_dtype}:16, *{input_dtype}:16, *{output_dtype}:16, "  # A/B/C
    "*fp32:16, *fp32:16, *fp32:16, *fp32:16, "  # bias/input_scale/weight_scale/output_scale
    "*i32:16, *i32:16, *i32:16, *i32:16, *i32:16, "  # gather_a_ptr/scatter_d_ptr/expert_idx/rank_start_idx/rank_end_idx
    "*i32:16, "  # M_pad_ptr
    "i32:16, i32:16, i32, i32, "  # N/K/E/M
    "i32:16, i32:1, i32:16, i32:1, i32:16, i32:16, i32:1, "  # A/B/C strides
    "*i32:16, "  # barrier_ptr
    "%WITH_BIAS, "
    "%BLOCK_SIZE_M, "
    "%BLOCK_SIZE_N, "
    "%BLOCK_SIZE_K, "
    "%GROUP_SIZE_M"
)
_grid = [
    "((M + %BLOCK_SIZE_M - 1) / %BLOCK_SIZE_M + E) * ((N + %BLOCK_SIZE_N - 1) / %BLOCK_SIZE_N)",
    "1",
    "1",
]


def get_triton_algo_info(dtype: torch.dtype, with_bias: bool):
    if dtype == torch.int8:
        return {
            "WITH_BIAS": with_bias,
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 4,
            "num_warps": 4,
            "num_stages": 4,
        }
    else:
        return {
            "WITH_BIAS": with_bias,
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "num_warps": 4,
            "num_stages": 4,
        }


@aot_compile_spaces(
    {
        "moe_ag_scatter_grouped_gemm_fp16": {
            "signature": signature.format(input_dtype="fp16", output_dtype="fp16"),
            "grid": _grid,
            "triton_algo_infos": [
                get_triton_algo_info(torch.float16, with_bias=False),
            ],
        },
        "moe_ag_scatter_grouped_gemm_bf16": {
            "signature": signature.format(input_dtype="bf16", output_dtype="bf16"),
            "grid": _grid,
            "triton_algo_infos": [
                get_triton_algo_info(torch.bfloat16, with_bias=False),
            ],
        },
        "moe_ag_scatter_grouped_gemm_s8": {
            "signature": signature.format(input_dtype="i8", output_dtype="bf16"),
            "grid": _grid,
            "triton_algo_infos": [
                get_triton_algo_info(torch.int8, with_bias=False),
                get_triton_algo_info(torch.int8, with_bias=True),
            ],
        },
    }
)
@triton.jit
def moe_ag_scatter_grouped_gemm_kernel(
    A,
    B,
    C,
    bias,
    input_scale,
    weight_scale,
    output_scale,
    gather_a_index,
    scatter_d_index,
    expert_idx,
    rank_start_idx,
    rank_end_idx,
    M_pad_ptr,
    N,
    K,
    E,  # used to calculate the number of blocks
    M,
    A_stride_m,
    A_stride_k,
    B_stride_e,
    B_stride_k,
    B_stride_n,
    C_stride_m,
    C_stride_n,
    barrier,
    WITH_BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    M_pad = tl.load(M_pad_ptr)
    num_block_m = tl.cdiv(M_pad, BLOCK_SIZE_M)
    num_block_n = tl.cdiv(N, BLOCK_SIZE_N)
    if pid >= num_block_m * num_block_n:
        return

    num_blocks_per_group = GROUP_SIZE_M * num_block_n
    group_id = pid // num_blocks_per_group
    group_size = min(num_block_m - group_id * GROUP_SIZE_M, GROUP_SIZE_M)
    pid_m = group_id * GROUP_SIZE_M + pid % group_size
    pid_n = pid % num_blocks_per_group // group_size

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_gather_a = tl.load(gather_a_index + offs_token_id)
    offs_scatter_d = tl.load(scatter_d_index + offs_token_id)
    token_mask = offs_scatter_d < M

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + offs_gather_a[:, None] * A_stride_m + offs_k[None, :] * A_stride_k

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_be = tl.load(expert_idx + pid_m)
    b_ptrs = B + offs_be * B_stride_e + offs_k[:, None] * B_stride_k + offs_bn[None, :] * B_stride_n

    segment_start = tl.load(rank_start_idx + pid_m)
    segment_end = tl.load(rank_end_idx + pid_m)
    thread_idx = tid(axis=0)
    if thread_idx >= segment_start and thread_idx <= segment_end:
        while ld_acquire(barrier + thread_idx, "sys") != 1:
            pass
    __syncthreads()

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
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    if A.dtype.element_ty == tl.int8:
        input_scale = tl.load(input_scale + offs_gather_a, mask=token_mask)[:, None]
        weight_scale = tl.load(weight_scale + offs_be * N + offs_cn, mask=offs_cn < N)[None, :]
        accumulator = accumulator.to(tl.float32)
        accumulator *= input_scale * weight_scale
        accumulator = accumulator.to(tl.bfloat16)
    else:
        accumulator = tl.load(output_scale + offs_be) * accumulator
        if tl.constexpr(A.dtype.element_ty == tl.float8e4nv) or tl.constexpr(
            A.dtype.element_ty == tl.float8e5
        ):
            accumulator = accumulator.to(tl.bfloat16)
        else:
            accumulator = accumulator.to(A.dtype.element_ty)  # support BF16 now

    # NOTE bias is supposed to be continous
    if WITH_BIAS:
        accumulator += tl.load(bias + offs_be * N + offs_cn, mask=offs_cn < N)[None, :]

    c_ptrs = C + offs_scatter_d[:, None] * C_stride_m + offs_cn[None, :] * C_stride_n
    tl.store(c_ptrs, accumulator, mask=c_mask)


def run_moe_ag_scatter_grouped_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    bias: Optional[torch.Tensor],
    input_scale: Optional[torch.Tensor],
    weight_scale: Optional[torch.Tensor],
    output_scale: torch.Tensor,
    gather_A_index: torch.Tensor,
    scatter_D_index: torch.Tensor,
    expert_idx: torch.Tensor,
    rank_start_idx: torch.Tensor,
    rank_end_idx: torch.Tensor,
    M_this_ep_pad: torch.Tensor,
    N: int,
    K: int,
    E: int,
    M_this_ep: int,
    barrier: torch.Tensor,
    **triton_algo_info: Dict[str, int],
):
    grid = lambda META: (
        (triton.cdiv(M_this_ep, META["BLOCK_SIZE_M"]) + E) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    moe_ag_scatter_grouped_gemm_kernel[grid](
        A,
        B,
        C,
        bias,
        input_scale,
        weight_scale,
        output_scale,
        gather_A_index,
        scatter_D_index,
        expert_idx,
        rank_start_idx,
        rank_end_idx,
        M_this_ep_pad,
        N,
        K,
        E,
        M_this_ep,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        C.stride(0),
        C.stride(1),
        barrier,
        **triton_algo_info,
    )
    return C
