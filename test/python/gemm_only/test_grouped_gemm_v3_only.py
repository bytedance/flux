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
import os
import time
from random import randint

import torch

import flux
import flux.testing
from flux.testing import DTYPE_MAP, init_seed


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


def perf_torch(input: torch.Tensor, weight: torch.Tensor, split_cpu: torch.Tensor, iters: int):
    def fn():
        acc = 0
        output_list = []
        for exp_id in range(weight.size(0)):
            exp_w = weight[exp_id]
            exp_input = input[acc : acc + split_cpu[exp_id]]
            acc += split_cpu[exp_id]
            output_list.append(torch.matmul(exp_input, exp_w.t()))
        return torch.concat(output_list)

    return perf_gemm(iters, "torch", fn)


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
    iters: int,
):
    op = flux.GemmGroupedV3(weight, weight.size(0))

    def fn():
        return op.forward(input, split_cpu)

    return perf_gemm(iters, "flux", fn)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=10240)
    parser.add_argument("-N", type=int, default=4096)
    parser.add_argument("-K", type=int, default=4096)
    parser.add_argument("-G", type=int, default=32)
    parser.add_argument("--iters", default=50, type=int, help="perf iterations")
    parser.add_argument("--dtype", default="bfloat16", type=str, help="data type")

    return parser.parse_args()


if __name__ == "__main__":
    init_seed()
    args = parse_args()
    dtype = DTYPE_MAP[args.dtype]
    input = torch.rand((args.M, args.K), dtype=dtype).cuda() / 10
    weight = torch.rand((args.G, args.N, args.K), dtype=dtype).cuda() / 10
    split_cpu = torch.tensor(randomList(args.G, args.M)).cpu()
    print(split_cpu)
    perf_result_flux = perf_flux(input, weight, split_cpu, args.iters)
    perf_result_torch = perf_torch(input, weight, split_cpu, args.iters)
    flux.testing.print_grouped_gemm_sol_time_ms(args.M, args.N, args.K, args.G, dtype)
    print(perf_result_torch)
    print(perf_result_flux)

    flux_output = perf_result_flux.output
    torch_output = perf_result_torch.output
    atol = 1e-2 if dtype == torch.float16 else 2e-2
    rtol = 1e-2 if dtype == torch.float16 else 2e-2
    flux.torch_allclose(flux_output, torch_output, atol=atol, rtol=rtol)
