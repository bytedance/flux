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

import argparse
import dataclasses
import os
import time
import numpy as np
from typing import Optional
from functools import partial

import torch
import triton
import triton.language as tl

import flux
import flux.testing
from flux.testing import DTYPE_MAP, init_seed
from flux.util import is_fp8_dtype


@dataclasses.dataclass
class BlockWiseScalingConfig:
    BLOCK_M: int
    BLOCK_N: int
    BLOCK_K: int
    ScaleMsPerTile: int
    ScaleNsPerTile: int


class PerfResult:
    def __init__(self, name: str, output: torch.Tensor, gemm_time_ms: float) -> None:
        self.name = name
        self.output = output
        self.gemm_time_ms = gemm_time_ms

    def __repr__(self) -> str:
        return f"{self.name}: gemm {self.gemm_time_ms:.3f} ms"


def perf_gemm(iters: int, name: str, fn: callable):
    warmup_iters = 5
    for i in range(warmup_iters):
        output = fn()
    torch.cuda.synchronize()
    total_time = 0
    start = time.time()
    for i in range(iters):
        output = fn()
    torch.cuda.synchronize()
    end = time.time()
    total_time = end - start
    return PerfResult(name=name, output=output, gemm_time_ms=total_time / iters * 1000)


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,  #
    blockscale_a_ptr,
    blockscale_b_ptr,  #
    c_ptr,  #
    M,
    N,
    K,  #
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,  #
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    ScaleMsPerTile: tl.constexpr,
    MMA_PROMOTION_INTERVAL: tl.constexpr,  #
):

    num_block_k = tl.cdiv(K, BLOCK_SIZE_K)

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_am = tl.where(offs_am < M, offs_am, 0)
    offs_bn = tl.where(offs_bn < N, offs_bn, 0)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_scalem = tl.arange(0, ScaleMsPerTile)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    blockscale_a_ptrs = blockscale_a_ptr + pid_m * num_block_k * ScaleMsPerTile + offs_scalem
    blockscale_b_ptrs = blockscale_b_ptr + pid_n * num_block_k

    accumulator_tmp = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        accumulator_cur = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        blockscale_a = tl.load(blockscale_a_ptrs)
        blockscale_b = tl.load(blockscale_b_ptrs)
        accumulator_cur = tl.dot(a, b, accumulator_cur)
        accumulator_cur = blockscale_a[:, None] * blockscale_b[None, None] * accumulator_cur
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        blockscale_a_ptrs += ScaleMsPerTile
        blockscale_b_ptrs += 1
        accumulator_tmp = accumulator_tmp + accumulator_cur
        if (k + 1) % MMA_PROMOTION_INTERVAL == 0:
            accumulator = accumulator + accumulator_tmp
            accumulator_tmp = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    accumulator = accumulator + accumulator_tmp

    if c_ptr.dtype.element_ty == tl.float8e4nv:
        c = accumulator.to(tl.float8e4nv)
    else:
        c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_dense_matmul(
    a, b, bias, blockscale_a, blockscale_b, output_dtype, BlockWiseScalingConfig
):
    # Check constraints.
    b = b.permute(1, 0).contiguous()
    assert bias == None
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    assert ScaleMsPerTile == BlockWiseScalingConfig.BLOCK_M
    M, K = a.shape
    K, N = b.shape
    BLOCK_M, BLOCK_N, BLOCK_K == BlockWiseScalingConfig.BLOCK_M, BlockWiseScalingConfig.BLOCK_N, BlockWiseScalingConfig.BLOCK_K
    assert N % BLOCK_N == 0 and K % BLOCK_K == 0 and M % BLOCK_M == 0

    c = torch.empty((M, N), device=a.device, dtype=output_dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    matmul_kernel[grid](
        a,
        b,
        blockscale_a,
        blockscale_b,
        c,  #
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        BLOCK_SIZE_M=BlockWiseScalingConfig.BLOCK_M,
        BLOCK_SIZE_N=BlockWiseScalingConfig.BLOCK_N,
        BLOCK_SIZE_K=BlockWiseScalingConfig.BLOCK_K,
        ScaleMsPerTile=BlockWiseScalingConfig.ScaleMsPerTile,
        GROUP_SIZE_M=8,
        MMA_PROMOTION_INTERVAL=4,
    )
    return c


def torch_dense_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    input_scale: Optional[torch.Tensor],  # [blockscale_m, blockscale_k, ScaleMsPerTile]
    weight_scale: Optional[torch.Tensor],  # [blockscale_n, blockscale_k]
    output_dtype: torch.dtype,
    config: BlockWiseScalingConfig,
):
    is_groupwise_b = config.ScaleNsPerTile == config.BLOCK_N
    M, K = input.shape
    N, K = weight.shape
    BLOCK_M, BLOCK_N, BLOCK_K = config.BLOCK_M, config.BLOCK_N, config.BLOCK_K
    ScaleMsPerTile = config.ScaleMsPerTile

    assert ScaleMsPerTile == BLOCK_M
    assert (
        N % BLOCK_N == 0 and K % BLOCK_K == 0 and M % BLOCK_M == 0
    ), f"{N} {K} {M} : {BLOCK_N} {BLOCK_K} {BLOCK_M}"
    assert bias == None

    num_block_m = M // BLOCK_M
    num_block_n = N // BLOCK_N
    num_block_k = K // BLOCK_K

    # [num_block_m, num_block_k, BLOCK_M, BLOCK_K]
    input = (
        input.reshape(num_block_m, BLOCK_M, num_block_k, BLOCK_K).permute(0, 2, 1, 3).contiguous()
    )
    # [num_block_n, num_block_k, BLOCK_K, BLOCK_N]
    weight = (
        weight.reshape(num_block_n, BLOCK_N, num_block_k, BLOCK_K).permute(0, 2, 3, 1).contiguous()
    )

    input = input.reshape(num_block_m, 1, num_block_k, BLOCK_M, BLOCK_K)
    weight = weight.reshape(1, num_block_n, num_block_k, BLOCK_K, BLOCK_N)
    input_scale = input_scale.reshape(num_block_m, 1, num_block_k, BLOCK_M, 1)
    if not is_groupwise_b:
        weight_scale = weight_scale.reshape(1, num_block_n, num_block_k, 1, 1)
    else:
        weight_scale = weight_scale.reshape(1, num_block_n, num_block_k, 1, BLOCK_N)
    input = input.to(torch.bfloat16)
    weight = weight.to(torch.bfloat16)

    def fn():
        acc = torch.zeros((num_block_m, num_block_n, BLOCK_M, BLOCK_N), dtype=torch.float32).cuda()

        acc_per_tile = torch.matmul(input, weight).to(
            torch.float32
        )  # [num_block_m, num_block_n, num_block_k, BLOCK_M, BLOCK_N]
        acc_per_tile = input_scale * weight_scale * acc_per_tile

        acc_per_tile = torch.sum(
            acc_per_tile, dim=2
        )  # [num_block_m, num_block_n, BLOCK_M, BLOCK_N]

        output = acc_per_tile.to(output_dtype)
        output = output.permute(0, 2, 1, 3).contiguous().reshape(M, N)

        return output

    return fn()


def _get_inputs_per_split(
    input: torch.Tensor,  # [M, K]
    input_splits_cpu: torch.Tensor,  # [num_experts]
    weight: torch.Tensor,  # [num_experts, N, K]
    bias: Optional[torch.Tensor],
    input_scale: Optional[
        torch.Tensor
    ],  # [blockscale_m, blockscale_k, ScaleMsPerTile] OR [groupscale_m, blockscale_k]
    weight_scale: Optional[
        torch.Tensor
    ],  # [num_experts, blockscale_n, blockscale_k, ScaleNsPerTile]
    split_idx: int,
):
    M, K = input.shape
    num_experts, N, K = weight.shape
    BLOCK_M, BLOCK_N, BLOCK_K = config.BLOCK_M, config.BLOCK_N, config.BLOCK_K
    ScaleMsPerTile = config.ScaleMsPerTile

    assert ScaleMsPerTile == BLOCK_M
    pre_sum = torch.cumsum(input_splits_cpu, dim=0)
    ed = pre_sum[split_idx].item()
    st = ed - input_splits_cpu[split_idx].item()

    scale_st_padded = 0
    scale_st = 0
    for i in range(split_idx):
        scale_st_padded += int(np.ceil(input_splits_cpu[i] / BLOCK_M))
        scale_st += input_splits_cpu[i]
    scale_ed_padded = scale_st_padded + int(np.ceil(input_splits_cpu[split_idx] / BLOCK_M))
    scale_ed = scale_st + input_splits_cpu[split_idx]

    cur_input = input[st:ed]
    cur_weight = weight[split_idx]
    cur_bias = bias[st:ed] if bias != None else None

    if input_scale != None:
        cur_input_scale = (
            input_scale[scale_st_padded:scale_ed_padded]
            if len(input_scale.size()) == 3
            else input_scale[scale_st:scale_ed]
            .view(-1, input_scale.shape[-1])
            .contiguous()
            .transpose(0, 1)
            .contiguous()
            .transpose(0, 1)
        )
    else:
        cur_input_scale = None
    cur_weight_scale = weight_scale[split_idx] if weight_scale != None else None
    return (cur_input, cur_weight, cur_bias, cur_input_scale, cur_weight_scale)


def _get_inputs_per_split_wgrad(
    input: torch.Tensor,  # [M, K]
    input_splits_cpu: torch.Tensor,  # [num_experts]
    weight: torch.Tensor,  # [N, K]
    bias: Optional[torch.Tensor],  # [E, M, N]
    input_scale: Optional[
        torch.Tensor
    ],  # [blockscale_m, blockscale_k, ScaleMsPerTile] OR [groupscale_m, blockscale_k]
    weight_scale: Optional[
        torch.Tensor
    ],  # [blockscale_n, blockscale_k, ScaleNsPerTile] OR [groupscale_n, blockscale_k]
    split_idx: int,
):

    BLOCK_M, BLOCK_N, BLOCK_K = config.BLOCK_M, config.BLOCK_N, config.BLOCK_K
    ScaleMsPerTile = config.ScaleMsPerTile

    assert ScaleMsPerTile == BLOCK_M
    assert ScaleNsPerTile == BLOCK_N
    pre_sum = torch.cumsum(input_splits_cpu, dim=0)
    ed = pre_sum[split_idx].item()
    st = ed - input_splits_cpu[split_idx].item()

    scale_st = 0
    for i in range(split_idx):
        scale_st += int(np.ceil(input_splits_cpu[i] / BLOCK_M))
    scale_ed = scale_st + int(np.ceil(input_splits_cpu[split_idx] / BLOCK_M))

    cur_input = input[:, st:ed]
    cur_weight = weight[:, st:ed]
    cur_bias = bias[split_idx] if bias != None else None
    if input_scale != None:
        cur_input_scale = (
            input_scale[:, scale_st:scale_ed]
            if len(input_scale.size()) == 3
            else input_scale[:, scale_st:scale_ed]
        )
    else:
        cur_input_scale = None
    if weight_scale != None:
        cur_weight_scale = (
            weight_scale[:, scale_st:scale_ed]
            if len(weight_scale.size()) == 3
            else weight_scale[:, scale_st:scale_ed]
        )
    else:
        cur_weight_scale = None
    return (cur_input, cur_weight, cur_bias, cur_input_scale, cur_weight_scale)


def limited_rand(elements, shape):
    total_elems = torch.prod(torch.tensor(shape)).item()
    indices = torch.randint(0, len(elements), (total_elems,), device=elements.device)
    return elements[indices].view(shape)


def group_gemm_wrapper(
    input: torch.Tensor,  # [M, K]
    input_splits_cpu: torch.Tensor,  # [num_experts]
    weight: torch.Tensor,  # [num_experts, N, K]
    bias: Optional[torch.Tensor],
    input_scale: Optional[torch.Tensor],  # [blockscale_m, blockscale_k, ScaleMsPerTile]
    weight_scale: Optional[torch.Tensor],  # [num_experts, blockscale_n, blockscale_k]
    output_dtype: torch.dtype,
    config: BlockWiseScalingConfig,
    dense_gemm_fn: callable,
    need_padding: bool,
):
    M, N = input.shape[0], weight.shape[1]
    K = input.shape[1]
    num_experts = input_splits_cpu.shape[0]
    BLOCK_M = config.BLOCK_M

    all_outs = []
    for i in range(num_experts):
        cur_input, cur_weight, cur_bias, cur_input_scale, cur_weight_scale = _get_inputs_per_split(
            input, input_splits_cpu, weight, bias, input_scale, weight_scale, i
        )
        if need_padding:
            cur_M = cur_input.shape[0]
            cur_padded_M = int(np.ceil(cur_M / BLOCK_M)) * BLOCK_M
            cur_padded_input = torch.zeros(
                (cur_padded_M, K), dtype=input.dtype, device=input.device
            )
            cur_padded_input[:cur_M].copy_(cur_input)
            cur_padded_out = dense_gemm_fn(
                cur_padded_input.contiguous(),
                cur_weight.contiguous(),
                cur_bias,
                cur_input_scale.contiguous(),
                cur_weight_scale.contiguous(),
                output_dtype,
                config,
            )
            cur_out = cur_padded_out[:cur_M]
        else:
            cur_out = dense_gemm_fn(
                cur_input,
                cur_weight,
                cur_bias,
                cur_input_scale,
                cur_weight_scale,
                output_dtype,
                config,
            )
        all_outs.append(cur_out)
    if len(all_outs) == 0:
        return all_outs[0]

    out = torch.empty((M, N), dtype=output_dtype, device="cuda")
    pre = 0
    for seg in all_outs:
        nxt = pre + seg.shape[0]
        out[pre:nxt] = seg
        pre = nxt
    return out


def group_gemm_wrapper_cutlass(
    input: torch.Tensor,  # [M, K]
    input_splits_cpu: torch.Tensor,  # [num_experts]
    weight: torch.Tensor,  # [num_experts, N, K]
    bias: Optional[torch.Tensor],
    input_scale: Optional[torch.Tensor],  # [groupscale_m, blockscale_k]
    weight_scale: Optional[torch.Tensor],  # [num_experts, blockscale_n, blockscale_k]
    output_dtype: torch.dtype,
    config: BlockWiseScalingConfig,
    dense_gemm_fn: callable,
    need_padding: bool,
):
    M, N = input.shape[0], weight.shape[1]
    K = input.shape[1]
    num_experts = input_splits_cpu.shape[0]
    BLOCK_M = config.BLOCK_M

    all_outs = []
    for i in range(num_experts):
        cur_input, cur_weight, cur_bias, cur_input_scale, cur_weight_scale = _get_inputs_per_split(
            input, input_splits_cpu, weight, bias, input_scale, weight_scale, i
        )
        if need_padding:
            cur_M = cur_input.shape[0]
            cur_padded_M = int(np.ceil(cur_M / BLOCK_M)) * BLOCK_M
            cur_padded_input = torch.zeros(
                (cur_padded_M, K), dtype=input.dtype, device=input.device
            )
            cur_padded_input[:cur_M].copy_(cur_input)
            cur_padded_out = dense_gemm_fn(
                cur_padded_input.contiguous(),
                cur_weight.contiguous(),
                cur_bias,
                cur_input_scale,
                cur_weight_scale,
                output_dtype,
                config,
            )
            cur_out = cur_padded_out[:cur_M]
        else:
            cur_out = dense_gemm_fn(
                cur_input,
                cur_weight,
                cur_bias,
                cur_input_scale,
                cur_weight_scale,
                output_dtype,
                config,
            )
        all_outs.append(cur_out)
    if len(all_outs) == 0:
        return all_outs[0]

    out = torch.empty((M, N), dtype=output_dtype, device="cuda")
    pre = 0
    for seg in all_outs:
        nxt = pre + seg.shape[0]
        out[pre:nxt] = seg
        pre = nxt
    return out


def group_Wgradwrapper(
    input: torch.Tensor,  # [M, K]
    input_splits_cpu: torch.Tensor,  # [num_experts]
    weight: torch.Tensor,  # [N, K]
    bias: Optional[torch.Tensor],
    input_scale: Optional[torch.Tensor],  # [blockscale_m, blockscale_k, ScaleMsPerTile]
    weight_scale: Optional[torch.Tensor],  # [blockscale_n, blockscale_k, ScaleNsPerTile]]
    output_dtype: torch.dtype,
    config: BlockWiseScalingConfig,
    dense_gemm_fn: callable,
    need_padding: bool,
):
    M, N = input.shape[0], weight.shape[0]
    K = input.shape[1]
    num_experts = input_splits_cpu.shape[0]
    BLOCK_K = config.BLOCK_K

    E = input_splits_cpu.shape[0]
    all_outs = []

    for i in range(num_experts):
        cur_input, cur_weight, cur_bias, cur_input_scale, cur_weight_scale = (
            _get_inputs_per_split_wgrad(
                input, input_splits_cpu, weight, bias, input_scale, weight_scale, i
            )
        )
        if need_padding:
            cur_K = cur_input.shape[1]
            cur_padded_K = int(np.ceil(cur_K / BLOCK_K)) * BLOCK_K
            cur_padded_input = torch.zeros(
                (M, cur_padded_K), dtype=input.dtype, device=input.device
            )
            cur_padded_weight = torch.zeros(
                (N, cur_padded_K), dtype=weight.dtype, device=weight.device
            )

            cur_padded_input[:, :cur_K].copy_(cur_input)
            cur_padded_weight[:, :cur_K].copy_(cur_weight)

            cur_out = dense_gemm_fn(
                cur_padded_input,
                cur_padded_weight,
                cur_bias,
                cur_input_scale,
                cur_weight_scale,
                output_dtype,
                config,
            )
        else:
            raise NotImplementedError("does not support non-padding")

        all_outs.append(cur_out)
    if len(all_outs) == 0:
        return all_outs[0]

    out = torch.empty((E, M, N), dtype=output_dtype, device="cuda")
    for i in range(E):
        out[i] = all_outs[i]
    return out


def group_Wgradwrapper_cutlass(
    input: torch.Tensor,  # [M, K]
    input_splits_cpu: torch.Tensor,  # [num_experts]
    weight: torch.Tensor,  # [N, K]
    bias: Optional[torch.Tensor],
    input_scale: Optional[torch.Tensor],  # [groupscale_m, blockscale_k]
    weight_scale: Optional[torch.Tensor],  # [groupscale_n, blockscale_k]
    output_dtype: torch.dtype,
    config: BlockWiseScalingConfig,
    dense_gemm_fn: callable,
    need_padding: bool,
):
    M, N = input.shape[0], weight.shape[0]
    K = input.shape[1]
    num_experts = input_splits_cpu.shape[0]
    BLOCK_K = config.BLOCK_K

    E = input_splits_cpu.shape[0]
    all_outs = []

    for i in range(num_experts):
        cur_input, cur_weight, cur_bias, cur_input_scale, cur_weight_scale = (
            _get_inputs_per_split_wgrad(
                input, input_splits_cpu, weight, bias, input_scale, weight_scale, i
            )
        )
        if need_padding:
            cur_K = cur_input.shape[1]
            cur_padded_K = int(np.ceil(cur_K / BLOCK_K)) * BLOCK_K
            cur_padded_input = torch.zeros(
                (M, cur_padded_K), dtype=input.dtype, device=input.device
            )
            cur_padded_weight = torch.zeros(
                (N, cur_padded_K), dtype=weight.dtype, device=weight.device
            )

            cur_padded_input[:, :cur_K].copy_(cur_input)
            cur_padded_weight[:, :cur_K].copy_(cur_weight)

            cur_out = dense_gemm_fn(
                cur_padded_input,
                cur_padded_weight,
                cur_bias,
                cur_input_scale,
                cur_weight_scale,
                output_dtype,
                config,
            )
        else:
            raise NotImplementedError("does not support non-padding")

        all_outs.append(cur_out)
    if len(all_outs) == 0:
        return all_outs[0]

    out = torch.empty((E, M, N), dtype=output_dtype, device="cuda")
    for i in range(E):
        out[i] = all_outs[i]
    return out


def flux_dense_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    input_scale: Optional[torch.Tensor],  # [blockscale_m, blockscale_k, ScaleMsPerTile]
    weight_scale: Optional[torch.Tensor],  # [blockscale_n, blockscale_k]
    output_dtype: torch.dtype,
    config: BlockWiseScalingConfig,
):
    return flux_op.forward(
        input, weight, bias=bias, input_scale=input_scale, weight_scale=weight_scale
    )


def reference_dense_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    input_scale: Optional[torch.Tensor],  # [blockscale_m, blockscale_k, ScaleMsPerTile]
    weight_scale: Optional[torch.Tensor],  # [blockscale_n, blockscale_k]
    output_dtype: torch.dtype,
    config: BlockWiseScalingConfig,
):
    is_groupwise_b = config.ScaleNsPerTile == config.BLOCK_N
    return flux_op.reference(
        input,
        weight,
        bias=bias,
        input_scale=input_scale,
        weight_scale=weight_scale,
        is_groupwise_b=is_groupwise_b,
    )


def flux_group_gemm_multistream(
    input: torch.Tensor,  # [M, K]
    input_splits_cpu: torch.Tensor,  # [num_experts]
    weight: torch.Tensor,  # [num_experts, N, K]
    bias: Optional[torch.Tensor],
    input_scale: Optional[torch.Tensor],  # [blockscale_m, blockscale_k, ScaleMsPerTile]
    weight_scale: Optional[torch.Tensor],  # [num_experts, blockscale_n, blockscale_k]
    output_dtype: torch.dtype,
    config: BlockWiseScalingConfig,
):
    is_groupwise_b = config.ScaleNsPerTile == config.BLOCK_N
    if is_groupwise_b:
        return flux_op.wgrad_multistream(
            input,
            input_splits_cpu,
            weight,
            bias=bias,
            input_scale=input_scale,
            weight_scale=weight_scale,
        )
    else:
        return flux_op.forward_multistream(
            input,
            input_splits_cpu,
            weight,
            bias=bias,
            input_scale=input_scale,
            weight_scale=weight_scale,
        )


def flux_group_gemm(
    input: torch.Tensor,  # [M, K]
    input_splits_cpu: torch.Tensor,  # [num_experts]
    weight: torch.Tensor,  # [num_experts, N, K]
    bias: Optional[torch.Tensor],
    input_scale: Optional[torch.Tensor],  # [blockscale_m, blockscale_k]
    weight_scale: Optional[torch.Tensor],  # [num_experts, blockscale_n, blockscale_k]
    output_dtype: torch.dtype,
    config: BlockWiseScalingConfig,
):
    is_groupwise_b = config.ScaleNsPerTile == config.BLOCK_N

    if is_groupwise_b:
        M = input.shape[0]
        N = weight.shape[0]
        K = input.shape[1]

        # input [M, K] --> [M, K1] [M, K2], ...
        # weight [N, K] --> [N, K1] [N, K2], ...
        input_new = torch.zeros(input.numel(), dtype=input.dtype, device=input.device).contiguous()
        weight_new = torch.zeros(
            weight.numel(), dtype=weight.dtype, device=weight.device
        ).contiguous()
        for i in range(input_splits_cpu.shape[0]):
            st = int(input_splits_cpu[:i].sum().item())
            ed = st + input_splits_cpu[i]
            input_new[st * M : ed * M].copy_(input[:, st:ed].contiguous().view(-1))
            weight_new[st * N : ed * N].copy_(weight[:, st:ed].contiguous().view(-1))

        # input_scale [M, blockscale_k] for [M, K] [M, K1] [M, K2]
        # weight_scale [N, blockscale_k] for [N, K]
        input_scale_new = torch.zeros(
            input_scale.numel(), dtype=input_scale.dtype, device=input.device
        ).contiguous()
        weight_scale_new = torch.zeros(
            weight_scale.numel(), dtype=weight_scale.dtype, device=weight.device
        ).contiguous()
        for i in range(input_splits_cpu.shape[0]):
            st = int(input_splits_cpu[:i].sum().item()) // BLOCK_K
            ed = st + input_splits_cpu[i] // BLOCK_K
            input_scale_new[st * M : ed * M].copy_(
                input_scale[:, st:ed].transpose(0, 1).contiguous().view(-1)
            )
            weight_scale_new[st * N : ed * N].copy_(
                weight_scale[:, st:ed].transpose(0, 1).contiguous().view(-1)
            )

        return flux_op.wgrad_grouped(
            input_new.view(M, K),
            input_splits_cpu,
            weight_new.view(N, K),
            bias=bias,
            input_scale=input_scale_new,
            weight_scale=weight_scale_new,
        )
    else:
        # input_scale [M, blockscale_k]
        # weight_scale [num_experts, blockscale_n, blockscale_k]
        input_scale_new = torch.zeros(
            input_scale.numel(), dtype=input_scale.dtype, device=input.device
        ).contiguous()
        weight_scale_new = torch.zeros(
            weight_scale.numel(), dtype=weight_scale.dtype, device=weight.device
        ).contiguous()

        for i in range(input_splits_cpu.shape[0]):
            st = int(input_splits_cpu[:i].sum().item())
            ed = st + input_splits_cpu[i]
            input_scale_new[st * input_scale.shape[1] : ed * input_scale.shape[1]].copy_(
                input_scale[st:ed].transpose(0, 1).contiguous().view(-1)
            )
            weight_scale_new[
                i
                * weight_scale.shape[1]
                * weight_scale.shape[2] : (i + 1)
                * weight_scale.shape[1]
                * weight_scale.shape[2]
            ].copy_(weight_scale[i].transpose(0, 1).contiguous().view(-1))

        return flux_op.forward_grouped(
            input,
            input_splits_cpu,
            weight,
            bias=bias,
            input_scale=input_scale_new,
            weight_scale=weight_scale_new,
        )


def rand_tensor(shape: list[int], dtype: torch.dtype):
    if dtype in [torch.int32, torch.int8]:
        return torch.randint(-127, 128, shape, dtype=dtype).cuda()
    elif is_fp8_dtype(dtype):
        data = torch.rand(shape, dtype=torch.bfloat16) * 4 - 2  # [-2, 2)
        return data.cuda().to(dtype)
    else:
        return torch.rand(shape, dtype=dtype).cuda() * 2 - 1  # [-1, 1)


def rand_splits(num_experts, M, BLOCK_M):
    num_block = M
    probs = torch.exp(torch.randn(num_experts))
    probs = probs / probs.sum()
    counts = torch.multinomial(probs, num_samples=num_block, replacement=True)
    tensor = torch.bincount(counts, minlength=num_experts)
    return tensor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("--num_experts", type=int, default=1)
    parser.add_argument("--num_streams", type=int, default=1)
    parser.add_argument("--iters", default=1, type=int, help="perf iterations")
    parser.add_argument(
        "--input_dtype",
        default="float8_e4m3fn",
        type=str,
        choices=["float8_e4m3fn", "float8_e5m2"],
    )
    parser.add_argument(
        "--weight_dtype",
        default="float8_e4m3fn",
        type=str,
        choices=["float8_e4m3fn", "float8_e5m2"],
    )
    parser.add_argument(
        "--output_dtype",
        default="bfloat16",
        type=str,
        choices=["float8_e4m3fn", "float8_e5m2", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--scale_type_a",
        default="row",
        type=str,
        choices=["row"],
        help="a always use groupwise scaling(1x128)",
    )
    parser.add_argument(
        "--scale_type_b",
        default="block",
        type=str,
        choices=["block", "col"],
    )
    ## TODO: add --output_transpose for selecting RCC layout
    parser.add_argument("--triton", default=False, action="store_true", help="use triton")
    parser.add_argument(
        "--limited_rand",
        default=False,
        action="store_true",
        help="use limited_rand to gen input/weight",
    )
    parser.add_argument(
        "--profile",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="dump torch.profiler.profile",
    )
    parser.add_argument(
        "--has_bias", default=False, action="store_true", help="whether to add bias"
    )

    return parser.parse_args()


THRESHOLD_MAP = {
    torch.float16: 1e-2,
    torch.bfloat16: 2e-2,
    torch.float32: 1e-2,
    torch.float8_e4m3fn: 2e-1,
    torch.float8_e5m2: 3e-1,
    torch.int8: 0,
    torch.int32: 0,
}

if __name__ == "__main__":
    init_seed()

    args = parse_args()
    input_dtype = DTYPE_MAP[args.input_dtype]
    weight_dtype = DTYPE_MAP[args.weight_dtype]
    is_fp8 = is_fp8_dtype(input_dtype) and is_fp8_dtype(weight_dtype)
    if args.output_dtype == "":
        output_dtype = input_dtype
    else:
        output_dtype = DTYPE_MAP[args.output_dtype]

    is_groupwise_b = args.scale_type_b == "col"

    if not is_fp8:
        raise ValueError("group/block-wise scaling gemm only support fp8")

    if is_groupwise_b and args.triton:
        raise ValueError("triton does not support matrix b with per-col scale")

    mixed_inputs = input_dtype != weight_dtype
    if mixed_inputs and output_dtype != torch.bfloat16:
        args.triton = False

    flux_op = flux.BlockScaleGemm(
        input_dtype=input_dtype,
        weight_dtype=weight_dtype,
        output_dtype=output_dtype,
        num_streams=args.num_streams,
    )

    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 128
    ScaleMsPerTile = BLOCK_M
    ScaleNsPerTile = BLOCK_N if is_groupwise_b else 1
    config = BlockWiseScalingConfig(
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        ScaleMsPerTile=ScaleMsPerTile,
        ScaleNsPerTile=ScaleNsPerTile,
    )

    input = None
    weight = None
    if is_fp8:
        torch.use_deterministic_algorithms(False, warn_only=True)

    input = rand_tensor((args.M, args.K), dtype=input_dtype)
    if is_groupwise_b:
        # [M, K] [N, K] --> [E, M, N]
        weight = rand_tensor((args.N, args.K), dtype=weight_dtype)
        input_splits_cpu = rand_splits(args.num_experts, args.K, BLOCK_K)
        # TODO (hanshi.s) fix Padding
        input_splits_cpu = rand_splits(args.num_experts, args.K // 128, BLOCK_K)
        input_splits_cpu *= 128
        assert input_splits_cpu.sum() == args.K

        print(f"input_splits_cpu = {input_splits_cpu}")
        if args.limited_rand:
            elements = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32).cuda()
            input = limited_rand(elements, (args.M, args.K)).to(input_dtype)
            weight = limited_rand(elements, (args.N, args.K)).to(weight_dtype)

        blockscale_k = 0
        for i in range(args.num_experts):
            blockscale_k += int(np.ceil(input_splits_cpu[i].item() / BLOCK_K))
        blockscale_n = int(np.ceil(args.N / BLOCK_N))
        blockscale_m = int(np.ceil(args.M / BLOCK_M))

        print(blockscale_m, blockscale_n, blockscale_k)

        input_scale = rand_tensor(
            (blockscale_m, blockscale_k, ScaleMsPerTile), dtype=torch.float32
        ).cuda()
        weight_scale = rand_tensor(
            (blockscale_n, blockscale_k, ScaleNsPerTile), dtype=torch.float32
        ).cuda()

    else:
        weight = rand_tensor((args.num_experts, args.N, args.K), dtype=weight_dtype)

        input_splits_cpu = rand_splits(args.num_experts, args.M, BLOCK_M)
        # TODO (hanshi.s) fix Padding
        input_splits_cpu = rand_splits(args.num_experts, args.M // 4, BLOCK_M)
        input_splits_cpu *= 4

        assert input_splits_cpu.sum() == args.M
        print(f"input_splits_cpu = {input_splits_cpu}")
        if args.limited_rand:
            elements = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32).cuda()
            input = limited_rand(elements, (args.M, args.K)).to(input_dtype)
            weight = limited_rand(elements, (args.num_experts, args.N, args.K)).to(weight_dtype)

        blockscale_m = 0
        for i in range(args.num_experts):
            blockscale_m += int(np.ceil(input_splits_cpu[i].item() / BLOCK_M))
        blockscale_n = int(np.ceil(args.N / BLOCK_N))
        blockscale_k = int(np.ceil(args.K / BLOCK_K))

        print(blockscale_m, blockscale_n, blockscale_k)

        input_scale = None
        weight_scale = None
        input_scale = rand_tensor(
            (blockscale_m, blockscale_k, ScaleMsPerTile), dtype=torch.float32
        ).cuda()
        weight_scale = rand_tensor(
            (args.num_experts, blockscale_n, blockscale_k, ScaleNsPerTile), dtype=torch.float32
        ).cuda()

    bias = None
    if args.has_bias:
        raise NotImplementedError("does not support bias")

    ctx = flux.get_torch_prof_ctx(args.profile)

    if is_groupwise_b:
        groupscale_m = blockscale_m * ScaleMsPerTile
        # [blockscale_m, ScaleMsPerTile, blockscale_k] --> [groupscale_m, blockscale_k]
        input_scale_cutlass = (
            input_scale.transpose(1, 2)
            .contiguous()
            .reshape(groupscale_m, blockscale_k)
            .transpose(0, 1)
            .contiguous()
        )
    else:
        assert ScaleMsPerTile == BLOCK_K
        groupscale_m = sum(input_splits_cpu)
        input_scale_cutlass = torch.zeros(
            (blockscale_k, groupscale_m), dtype=input_scale.dtype, device=input_scale.device
        )

        st = 0
        st_padded = 0
        for i in range(args.num_experts):
            ed = st + input_splits_cpu[i]
            ed_padded = st_padded + int(np.ceil(input_splits_cpu[i] / BLOCK_M))
            input_scale_cutlass[:, st:ed].copy_(
                input_scale[st_padded:ed_padded]
                .transpose(0, 1)
                .reshape(blockscale_k, -1)[:, : input_splits_cpu[i]]
            )
            st = ed
            st_padded = ed_padded

    input_scale_cutlass = input_scale_cutlass.contiguous().transpose(0, 1)

    if is_groupwise_b:
        # [blockscale_n, blockscale_k, ScaleNsPerTile]
        groupscale_n = blockscale_n * ScaleNsPerTile
        weight_scale_cutlass = (
            weight_scale.transpose(1, 2)
            .contiguous()
            .reshape(groupscale_n, blockscale_k)
            .transpose(0, 1)
            .contiguous()
            .transpose(0, 1)
        )
    else:
        # weight_scale shape: [num_experts, blockscale_n, blockscale_k]
        assert ScaleNsPerTile == 1
        weight_scale_cutlass = weight_scale.transpose(1, 2).contiguous().transpose(1, 2).squeeze(-1)

    with ctx:

        Wrapper = group_Wgradwrapper if is_groupwise_b else group_gemm_wrapper
        cutlassWrapper = (
            group_Wgradwrapper_cutlass if is_groupwise_b else group_gemm_wrapper_cutlass
        )

        group_gemm_with_padding_dispatcher = partial(
            Wrapper,
            input=input,
            input_splits_cpu=input_splits_cpu,
            weight=weight,
            bias=bias,
            input_scale=input_scale,
            weight_scale=weight_scale,
            output_dtype=output_dtype,
            config=config,
            need_padding=True,
        )

        group_gemm_dispatcher_cutlass = partial(
            cutlassWrapper,
            input=input,
            input_splits_cpu=input_splits_cpu,
            weight=weight,
            bias=bias,
            input_scale=input_scale_cutlass,
            weight_scale=weight_scale_cutlass,
            output_dtype=output_dtype,
            config=config,
            need_padding=False if not is_groupwise_b else True,
        )

        flux_multi_stream_fn = partial(
            flux_group_gemm_multistream,
            input=input,
            input_splits_cpu=input_splits_cpu,
            weight=weight,
            bias=bias,
            input_scale=input_scale_cutlass,
            weight_scale=weight_scale_cutlass,
            output_dtype=output_dtype,
            config=config,
        )

        flux_group_gemm_fn = partial(
            flux_group_gemm,
            input=input,
            input_splits_cpu=input_splits_cpu,
            weight=weight,
            bias=bias,
            input_scale=input_scale_cutlass,
            weight_scale=weight_scale_cutlass,
            output_dtype=output_dtype,
            config=config,
        )

        if not is_groupwise_b:
            if args.triton:
                perf_result_triton = perf_gemm(
                    args.iters,
                    "triton",
                    partial(group_gemm_with_padding_dispatcher, dense_gemm_fn=triton_dense_matmul),
                )

            perf_result_flux = perf_gemm(
                args.iters,
                "flux",
                partial(group_gemm_dispatcher_cutlass, dense_gemm_fn=flux_dense_gemm),
            )

        perf_result_torch = perf_gemm(
            args.iters,
            "torch",
            partial(group_gemm_with_padding_dispatcher, dense_gemm_fn=torch_dense_gemm),
        )

        perf_result_reference = perf_gemm(
            args.iters,
            "reference",
            partial(group_gemm_dispatcher_cutlass, dense_gemm_fn=reference_dense_gemm),
        )

        perf_result_flux_multi_stream = perf_gemm(
            args.iters, "flux_multi_stream", flux_multi_stream_fn
        )

        perf_result_flux_group = perf_gemm(args.iters, "flux_group", flux_group_gemm_fn)

    if args.profile:
        prof_dir = f"prof/blockscale_gemm"
        os.makedirs(prof_dir, exist_ok=True)
        ctx.export_chrome_trace(f"{prof_dir}/trace.json.gz")

    flux.testing.print_gemm_sol_time(args.M, args.N, args.K, input_dtype)

    if is_groupwise_b:
        perf_results = [perf_result_reference]
    else:
        perf_results = [perf_result_reference, perf_result_flux]
        if args.triton:
            perf_results.append(perf_result_triton)

    if args.limited_rand:
        # torch use bf16 to impl fp8 gemm, only using limted_rand can pass the test.
        perf_results.append(perf_result_torch)

    for perf in perf_results:
        print(perf)

    atol = THRESHOLD_MAP[output_dtype]
    rtol = THRESHOLD_MAP[output_dtype]

    for perf_result in perf_results:
        compare_out = perf_result.output
        flux_out = perf_result_flux_multi_stream.output
        if is_fp8_dtype(output_dtype) or output_dtype == torch.float32:
            compare_out = compare_out.to(torch.bfloat16)
            flux_out = perf_result_flux_multi_stream.output.to(torch.bfloat16)
        is_bitwise_match = flux.bitwise_check(compare_out, flux_out)
        print(
            f"{perf_result_flux_multi_stream.name} vs {perf_result.name} is bitwise match: {is_bitwise_match}"
        )
        flux.torch_allclose(compare_out, flux_out, atol=atol, rtol=rtol)

        flux_out = perf_result_flux_group.output
        if is_fp8_dtype(output_dtype) or output_dtype == torch.float32:
            compare_out = compare_out.to(torch.bfloat16)
            flux_out = flux_out.to(torch.bfloat16)
        is_bitwise_match = flux.bitwise_check(compare_out, flux_out)
        print(
            f"{perf_result_flux_group.name} vs {perf_result.name} is bitwise match: {is_bitwise_match}"
        )
        flux.torch_allclose(compare_out, flux_out, atol=atol, rtol=rtol)
