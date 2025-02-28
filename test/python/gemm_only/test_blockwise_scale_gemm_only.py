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
    M, K = input.shape
    N, K = weight.shape
    BLOCK_M, BLOCK_N, BLOCK_K = config.BLOCK_M, config.BLOCK_N, config.BLOCK_K
    ScaleMsPerTile = config.ScaleMsPerTile

    assert ScaleMsPerTile == BLOCK_M

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
    weight_scale = weight_scale.reshape(1, num_block_n, num_block_k, 1, 1)

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
    input_scale: Optional[torch.Tensor],  # [blockscale_m, blockscale_k, ScaleMsPerTile]
    weight_scale: Optional[torch.Tensor],  # [num_experts, blockscale_n, blockscale_k]
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
    assert ed % BLOCK_M == 0 and st % BLOCK_M == 0

    cur_input = input[st:ed]
    cur_weight = weight[split_idx]
    cur_bias = bias[st:ed] if bias != None else None
    cur_input_scale = input_scale[st // BLOCK_M : ed // BLOCK_M] if input_scale != None else None
    cur_weight_scale = weight_scale[split_idx] if weight_scale != None else None
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
):
    M, N = input.shape[0], weight.shape[1]
    num_experts = input_splits_cpu.shape[0]

    all_outs = []
    for i in range(num_experts):
        cur_input, cur_weight, cur_bias, cur_input_scale, cur_weight_scale = _get_inputs_per_split(
            input, input_splits_cpu, weight, bias, input_scale, weight_scale, i
        )
        cur_out = dense_gemm_fn(
            cur_input, cur_weight, cur_bias, cur_input_scale, cur_weight_scale, output_dtype, config
        )
        all_outs.append(cur_out)
    if len(all_outs) == 0:
        return all_outs[0]

    # torch.cat(all_outs, dim = 0)
    out = torch.empty((M, N), dtype=output_dtype, device="cuda")
    pre = 0
    for seg in all_outs:
        nxt = pre + seg.shape[0]
        out[pre:nxt] = seg
        pre = nxt
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
    return flux_op.reference(
        input, weight, bias=bias, input_scale=input_scale, weight_scale=weight_scale
    )


def flux_group_gemm(
    input: torch.Tensor,  # [M, K]
    input_splits_cpu: torch.Tensor,  # [num_experts]
    weight: torch.Tensor,  # [num_experts, N, K]
    bias: Optional[torch.Tensor],
    input_scale: Optional[torch.Tensor],  # [blockscale_m, blockscale_k, ScaleMsPerTile]
    weight_scale: Optional[torch.Tensor],  # [num_experts, blockscale_n, blockscale_k]
    output_dtype: torch.dtype,
    config: BlockWiseScalingConfig,
):
    return flux_op.forward_multistream(
        input,
        input_splits_cpu,
        weight,
        bias=bias,
        input_scale=input_scale,
        weight_scale=weight_scale,
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
    num_block = M // BLOCK_M
    probs = torch.exp(torch.randn(num_experts))
    probs = probs / probs.sum()
    counts = torch.multinomial(probs, num_samples=num_block, replacement=True)
    tensor = torch.bincount(counts, minlength=num_experts) * BLOCK_M
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
        "--dtype",
        default="float8_e4m3fn",
        type=str,
        choices=list(DTYPE_MAP.keys()),
    )
    ## TODO: add --input_dtype, weight_dtype
    parser.add_argument(
        "--input_dtype",
        default="float8_e4m3fn",
        type=str,
        choices=list(DTYPE_MAP.keys()),
    )
    parser.add_argument(
        "--weight_dtype",
        default="float8_e4m3fn",
        type=str,
        choices=list(DTYPE_MAP.keys()),
    )
    ## TODO: add --output_transpose for selecting RCC layout
    parser.add_argument("--triton", default=False, action="store_true", help="use triton")
    parser.add_argument(
        "--profile",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="dump torch.profiler.profile",
    )
    parser.add_argument(
        "--output_dtype",
        default="",
        type=str,
        help="allowed data type:: bfloat16,float16.",
    )
    parser.add_argument(
        "--has_bias", default=False, action="store_true", help="whether to add bias"
    )

    return parser.parse_args()


THRESHOLD_MAP = {
    torch.float16: 1e-2,
    torch.bfloat16: 2e-2,
    torch.float8_e4m3fn: 2e-1,
    torch.float8_e5m2: 2e-2,
    torch.int8: 0,
    torch.int32: 0,
}

if __name__ == "__main__":
    init_seed()

    args = parse_args()
    dtype = DTYPE_MAP[args.dtype]
    is_fp8 = is_fp8_dtype(dtype)
    if args.output_dtype == "":
        output_dtype = dtype
    else:
        output_dtype = DTYPE_MAP[args.output_dtype]

    if not is_fp8:
        raise ValueError("group/block-wise scaling gemm only support fp8")

    flux_op = flux.BlockScaleGemm(
        input_dtype=dtype,
        output_dtype=output_dtype,
        num_streams=args.num_streams,
    )

    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 128
    ScaleMsPerTile = BLOCK_M
    config = BlockWiseScalingConfig(
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, ScaleMsPerTile=ScaleMsPerTile
    )

    ## TODO: relax problem size check
    # assert args.M % BLOCK_M == 0 and args.N % BLOCK_N == 0 and args.K % BLOCK_K == 0
    blockscale_m = int(np.ceil(args.M / BLOCK_M))
    blockscale_n = int(np.ceil(args.N / BLOCK_N))
    blockscale_k = int(np.ceil(args.K / BLOCK_K))

    print(blockscale_m, blockscale_n, blockscale_k)

    input = None
    weight = None
    if is_fp8:
        torch.use_deterministic_algorithms(False, warn_only=True)

    input = rand_tensor((args.M, args.K), dtype=dtype)
    weight = rand_tensor((args.num_experts, args.N, args.K), dtype=dtype)
    input_splits_cpu = rand_splits(args.num_experts, args.M, BLOCK_M)
    assert input_splits_cpu.sum() == args.M
    print(f"input_splits_cpu = {input_splits_cpu}")
    # elements = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32).cuda()
    # input = limited_rand(elements, (args.M, args.K)).to(dtype)
    # weight = limited_rand(elements, (args.num_experts, args.N, args.K)).to(dtype)

    input_scale = None
    weight_scale = None
    input_scale = rand_tensor(
        (blockscale_m, blockscale_k, ScaleMsPerTile), dtype=torch.float32
    ).cuda()
    weight_scale = rand_tensor(
        (args.num_experts, blockscale_n, blockscale_k), dtype=torch.float32
    ).cuda()

    bias = None
    if args.has_bias:
        raise NotImplementedError("does not support bias")

    ctx = flux.get_torch_prof_ctx(args.profile)

    with ctx:
        group_gemm_dispatcher = partial(
            group_gemm_wrapper,
            input=input,
            input_splits_cpu=input_splits_cpu,
            weight=weight,
            bias=bias,
            input_scale=input_scale,
            weight_scale=weight_scale,
            output_dtype=output_dtype,
            config=config,
        )

        flux_multi_stream_fn = partial(
            flux_group_gemm,
            input=input,
            input_splits_cpu=input_splits_cpu,
            weight=weight,
            bias=bias,
            input_scale=input_scale,
            weight_scale=weight_scale,
            output_dtype=output_dtype,
            config=config,
        )

        if args.triton:
            perf_result_triton = perf_gemm(
                args.iters,
                "triton",
                partial(group_gemm_dispatcher, dense_gemm_fn=triton_dense_matmul),
            )
        perf_result_torch = perf_gemm(
            args.iters, "torch", partial(group_gemm_dispatcher, dense_gemm_fn=torch_dense_gemm)
        )
        perf_result_reference = perf_gemm(
            args.iters,
            "reference",
            partial(group_gemm_dispatcher, dense_gemm_fn=reference_dense_gemm),
        )
        perf_result_flux = perf_gemm(
            args.iters, "flux", partial(group_gemm_dispatcher, dense_gemm_fn=flux_dense_gemm)
        )
        perf_result_flux_multi_stream = perf_gemm(
            args.iters, "flux_multi_stream", flux_multi_stream_fn
        )

    if args.profile:
        prof_dir = f"prof/blockscale_gemm"
        os.makedirs(prof_dir, exist_ok=True)
        ctx.export_chrome_trace(f"{prof_dir}/trace.json.gz")

    flux.testing.print_gemm_sol_time(args.M, args.N, args.K, dtype)
    print(perf_result_torch)
    print(perf_result_flux)
    print(perf_result_reference)
    if args.triton:
        print(perf_result_triton)
    print(perf_result_flux_multi_stream)

    if args.triton:
        perf_results = [perf_result_triton, perf_result_reference, perf_result_flux]
    else:
        perf_results = [perf_result_reference, perf_result_flux]

    atol = THRESHOLD_MAP[output_dtype]
    rtol = THRESHOLD_MAP[output_dtype]

    for perf_result in perf_results:
        compare_out = perf_result.output.to(torch.bfloat16)
        flux_out = perf_result_flux_multi_stream.output.to(torch.bfloat16)
        is_bitwise_match = flux.bitwise_check(compare_out, flux_out)
        print(
            f"{perf_result_flux_multi_stream.name} vs {perf_result.name} is bitwise match: {is_bitwise_match}"
        )
        flux.torch_allclose(compare_out, flux_out, atol=atol, rtol=rtol)
