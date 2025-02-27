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
import random
import time
from random import randint
from typing import List

import torch

import flux
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


def perf_scatter(
    inputs: List[torch.Tensor],
    scatter_idx: torch.Tensor,
    topk: int,
    iters: int,
):
    def fn():
        return flux.topk_scatter_reduce(inputs, scatter_idx, topk)

    return perf_gemm(iters, "flux_scatter", fn)


def torch_scatter(
    inputs: List[torch.Tensor],
    scatter_idx: torch.Tensor,
    topk: int,
    iters: int,
):

    n_dim = inputs[0].size(1)
    input_groups = len(inputs)
    new_m = inputs[0].size(0) // topk
    out = torch.zeros(new_m, n_dim, dtype=inputs[0].dtype, device=inputs[0].device)
    tmp = torch.zeros(inputs[0].size(0), n_dim, dtype=inputs[0].dtype, device=inputs[0].device)
    for group_id in range(input_groups):
        tmp += inputs[group_id]
    for i in range(inputs[0].size(0)):
        pos = scatter_idx[i].item()
        out[i // topk] += tmp[pos]
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=40960)
    parser.add_argument("-N", type=int, default=8192)
    parser.add_argument("-K", type=int, default=640)
    parser.add_argument("-G", type=int, default=32)
    parser.add_argument("--iters", default=100, type=int, help="perf iterations")
    parser.add_argument("--dtype", default="bfloat16", type=str, choices=list(DTYPE_MAP.keys()))
    parser.add_argument(
        "--transpose_weight", default=False, action="store_true", help="whether to transpose weight"
    )
    parser.add_argument(
        "--scatter", default=False, action="store_true", help="whether to scatter input"
    )
    parser.add_argument("--topk", default=5, type=int, help="topk")
    parser.add_argument("--input_groups", default=1, type=int, help="input_groups")
    parser.add_argument("--profile", default=False, action="store_true", help="whether to profile")
    return parser.parse_args()


if __name__ == "__main__":
    init_seed()
    args = parse_args()
    dtype = DTYPE_MAP[args.dtype]
    inputs = []
    for i in range(args.input_groups):
        inputs.append(torch.rand((args.M, args.N), dtype=dtype).cuda() - 0.5)
    scatter_idx = list(range(0, args.M))
    random.shuffle(scatter_idx)
    scatter_idx = torch.tensor(scatter_idx, dtype=torch.int32).cuda()
    ctx = flux.util.get_torch_prof_ctx(args.profile)
    perf_result_scatter = perf_scatter(inputs, scatter_idx, args.topk, args.iters)
    torch_scatter_out = torch_scatter(inputs, scatter_idx, args.topk, args.iters)
    flux_scatter_out = perf_result_scatter.output
    print(perf_result_scatter)
    print(perf_result_scatter.output)
    atol = 1e-2 if dtype == torch.float16 else 2e-2
    rtol = 1e-2 if dtype == torch.float16 else 2e-2
    flux.torch_allclose(perf_result_scatter.output, torch_scatter_out, atol=atol, rtol=rtol)
    # import ipdb

    # ipdb.set_trace()
