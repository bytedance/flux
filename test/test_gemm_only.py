################################################################################
#
# Copyright 2023 ByteDance Ltd. and/or its affiliates. All rights reserved.
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
import datetime
import torch
import numpy as np
import flux

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(True, warn_only=True)
torch.set_printoptions(precision=8)
torch.manual_seed(3)
torch.cuda.manual_seed_all(3)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
np.random.seed(3)


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


def perf_torch(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, iters: int):
    def fn():
        return (
            torch.matmul(input, weight.t())
            if bias is None
            else (torch.matmul(input, weight.t()) + bias)
        )

    return perf_gemm(iters, "torch", fn)


def perf_flux(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    transpose_weight: bool,
    iters: int,
):
    m = input.size(0)
    if transpose_weight:
        weight = weight.t().contiguous()
        n = weight.size(1)
    else:
        n = weight.size(0)
    output = torch.empty([m, n], dtype=input.dtype, device=input.device, requires_grad=False)
    op = flux.GemmOnly(weight.dtype, transpose_weight=transpose_weight)

    def fn():
        return op.forward(input, weight, bias=bias, output_buf=output)

    return perf_gemm(iters, "flux", fn)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("--iters", default=50, type=int, help="perf iterations")
    parser.add_argument("--dtype", default="bfloat16", type=str, help="data type")
    parser.add_argument(
        "--has_bias", default=False, action="store_true", help="whether to add bias"
    )
    parser.add_argument(
        "--transpose_weight", default=False, action="store_true", help="whether to transpose weight"
    )

    return parser.parse_args()


DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16}

if __name__ == "__main__":
    args = parse_args()
    dtype = DTYPE_MAP[args.dtype]
    input = torch.rand((args.M, args.K), dtype=dtype).cuda() / 10
    weight = torch.rand((args.N, args.K), dtype=dtype).cuda() / 10
    bias = torch.rand((args.M, args.N), dtype=dtype).cuda() if args.has_bias else None
    perf_result_flux = perf_flux(input, weight, bias, args.transpose_weight, args.iters)
    perf_result_torch = perf_torch(input, weight, bias, args.iters)
    print(f"SOL gemm {flux.estimate_gemm_sol_time_ms(args.M, args.N, args.K):.3f}ms")
    print(perf_result_torch)
    print(perf_result_flux)

    flux_output = perf_result_flux.output
    torch_output = perf_result_torch.output
    atol = 1e-2 if dtype == torch.float16 else 2e-2
    rtol = 1e-2 if dtype == torch.float16 else 2e-2
    flux.torch_allclose(flux_output, torch_output, atol=atol, rtol=rtol)
