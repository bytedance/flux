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
import time
from random import randint
from typing import List, Optional, Tuple

import torch

import flux
import flux.testing
from flux.testing import DTYPE_MAP, init_seed
from flux.util import is_fp8_dtype


class PerfResult:
    def __init__(self, name: str, output: torch.Tensor, gemm_time_ms: float) -> None:
        self.name = name
        self.output = output
        self.gemm_time_ms = gemm_time_ms

    def __repr__(self) -> str:
        return f"{self.name}: gemm {self.gemm_time_ms:.3f} ms"


def perf_gemm(iters: int, name: str, fn: callable, warmup_iters: int = 5):
    for _ in range(warmup_iters):
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


def perf_torch(
    input: torch.Tensor,
    weight: torch.Tensor,
    split_cpu: torch.Tensor,
    is_fp8: bool,
    iters: int,
    warmup_iters: int = 5,
    input_scale: Optional[Tuple[List[torch.Tensor], torch.Tensor]] = None,
    weight_scale: Optional[Tuple[List[torch.Tensor], torch.Tensor]] = None,
):
    alpha_scale = [1.0 for _ in range(weight.size(0))]
    if is_fp8:
        assert input_scale is not None
        assert weight_scale is not None
        weight_scale = (
            weight_scale
            if isinstance(weight_scale, list)
            else [weight_scale for _ in range(weight.size(0))]
        )
        alpha_scale = [input_scale * w for w in weight_scale]
        input = input.to(torch.bfloat16)
        weight = weight.to(torch.bfloat16)

    def fn():
        acc = 0
        output_list = []
        for exp_id in range(weight.size(0)):
            exp_w = weight[exp_id]
            exp_input = input[acc : acc + split_cpu[exp_id]]
            acc += split_cpu[exp_id]
            output_list.append(torch.matmul(exp_input, exp_w.t()) * alpha_scale[exp_id])
        return torch.concat(output_list).to(input.dtype)

    return perf_gemm(iters, "torch", fn, warmup_iters=warmup_iters)


def randomList(m, n):
    # Create an array of size m where
    # every element is initialized to 0
    arr = [0] * m
    # To make the sum of the final list as n
    for _ in range(n):
        # Increment any random element
        # from the array by 1
        arr[randint(0, n) % m] += 1
    return arr


def perf_flux(
    input: torch.Tensor,
    weight: torch.Tensor,
    split_cpu: torch.Tensor,
    is_fp8: bool,
    iters: int,
    warmup_iters: int = 5,
    input_scale: Optional[Tuple[List[torch.Tensor], torch.Tensor]] = None,
    weight_scale: Optional[Tuple[List[torch.Tensor], torch.Tensor]] = None,
):
    def _to_scales(input_scale):
        if input_scale is None or isinstance(input_scale, list):
            return input_scale
        return [input_scale]

    output_dtype = torch.bfloat16 if is_fp8 else input.dtype
    op = flux.GemmGroupedV2(weight, weight.size(0), input.dtype, output_dtype)
    input_scale = _to_scales(input_scale)
    weight_scale = _to_scales(weight_scale)

    def fn():
        return op.forward(
            input,
            split_cpu,
            input_scale=input_scale,
            weight_scale=weight_scale,
            fast_accum=False,
            sm_margin=args.sm_margin,
        )

    return perf_gemm(iters, "flux", fn, warmup_iters=warmup_iters)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=10240)
    parser.add_argument("-N", type=int, default=4096)
    parser.add_argument("-K", type=int, default=4096)
    parser.add_argument("-G", type=int, default=32)
    parser.add_argument(
        "--fast_accum",
        default=False,
        action="store_true",
        help="use fast accum. please use only with FP8 types",
    )
    parser.add_argument("--iters", default=50, type=int, help="perf iterations")
    parser.add_argument("--warmup_iters", default=5, type=int, help="warmup iterations")
    parser.add_argument("--sm_margin", default=0, type=int, help="sm margin")
    parser.add_argument(
        "--dtype", default="bfloat16", choices=list(DTYPE_MAP.keys()), type=str, help="data type"
    )

    return parser.parse_args()


if __name__ == "__main__":
    init_seed()
    args = parse_args()
    dtype = DTYPE_MAP[args.dtype]
    is_fp8 = is_fp8_dtype(dtype)

    input = None
    weight = None
    if is_fp8:
        torch.use_deterministic_algorithms(False, warn_only=True)
        input = torch.rand((args.M, args.K), dtype=torch.bfloat16).cuda() / 10
        weight = torch.rand((args.G, args.N, args.K), dtype=torch.bfloat16).cuda() / 10
        input = input.to(dtype)
        weight = weight.to(dtype)
    else:
        input = torch.rand((args.M, args.K), dtype=dtype).cuda() / 10
        weight = torch.rand((args.G, args.N, args.K), dtype=dtype).cuda() / 10

    split_cpu = torch.tensor(randomList(args.G, args.M)).cpu().to(torch.int32)
    print(split_cpu)

    input_scale = None
    weight_scale = None

    if is_fp8:
        input_scale = torch.rand(1, dtype=torch.float32).cuda()
        weight_scale = [torch.rand(1, dtype=torch.float32).cuda() for n in range(args.G)]

    perf_result_flux = perf_flux(
        input,
        weight,
        split_cpu,
        is_fp8,
        args.iters,
        args.warmup_iters,
        input_scale,
        weight_scale,
    )
    perf_result_torch = perf_torch(
        input,
        weight,
        split_cpu,
        is_fp8,
        args.iters,
        args.warmup_iters,
        input_scale,
        weight_scale,
    )
    flux.testing.print_grouped_gemm_sol_time_ms(args.M, args.N, args.K, args.G, dtype)
    print(perf_result_torch)
    print(perf_result_flux)

    flux_output = perf_result_flux.output
    torch_output = perf_result_torch.output
    atol = 1e-2 if dtype == torch.float16 else 2e-2
    rtol = 1e-2 if dtype == torch.float16 else 2e-2
    flux.torch_allclose(flux_output, torch_output, atol=atol, rtol=rtol)
