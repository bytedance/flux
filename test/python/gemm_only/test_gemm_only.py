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
from typing import Optional

import torch

import flux
import flux.testing
from flux.testing import DTYPE_MAP, init_seed, matmul_int8
from flux.util import is_fp8_dtype


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


def perf_torch(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    input_scale: Optional[torch.Tensor],
    weight_scale: Optional[torch.Tensor],
    is_fp8: bool,
    is_s8_dequant: bool,
    iters: int,
    output_dtype: torch.dtype,
):
    alpha_scale = 1.0
    if is_fp8:
        alpha_scale = input_scale * weight_scale
        input = input.to(torch.bfloat16)
        weight = weight.to(torch.bfloat16)

    def fn():
        if is_s8_dequant:
            accum = matmul_int8(input, weight.t()).to(torch.float32)
            output = input_scale * weight_scale * accum
        elif input.dtype == torch.int8:
            output = matmul_int8(input, weight.t())
        else:
            output = alpha_scale * torch.matmul(input, weight.t())
        if is_fp8 or is_s8_dequant:
            output = output.to(torch.bfloat16)
        else:
            output = output.to(output_dtype)
        if bias is not None:
            output = output + bias
        return output

    return perf_gemm(iters, "torch", fn)


def perf_flux(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    input_scale: Optional[torch.Tensor],
    weight_scale: Optional[torch.Tensor],
    transpose_weight: bool,
    is_fp8: bool,
    is_s8_dequant: bool,
    iters: int,
    output_dtype: torch.dtype,
):
    m = input.size(0)
    if transpose_weight:
        assert (
            is_fp8 == False and is_s8_dequant == False
        ), "FP8/S8 GEMM does not support transpose weight (RRR layout)"
        weight = weight.t().contiguous()
        n = weight.size(1)
    else:
        n = weight.size(0)

    def _check_tensor_shape(tensor, shape):
        if not isinstance(tensor, torch.Tensor):
            return False
        if len(tensor.size()) != len(shape):
            return False
        for x, y in zip(list(tensor.size()), shape):
            if x != y:
                return False
        return True

    if is_s8_dequant:
        if not _check_tensor_shape(input_scale, (m, 1)):
            raise ValueError("input_scale's shape should be (m, 1) for S8 GEMM")
        if not _check_tensor_shape(weight_scale, (1, n)):
            raise ValueError("weight_scale's shape should be (1, n) for S8 GEMM")

    output = torch.empty([m, n], dtype=output_dtype, device=input.device, requires_grad=False)
    ## TODO: remove below once moe fp8 gemm invoke get fixed
    use_fp8_gemm = True if is_fp8 else False
    op = flux.GemmOnly(
        input_dtype=input.dtype,
        output_dtype=output_dtype,
        transpose_weight=transpose_weight,
        use_fp8_gemm=use_fp8_gemm,
    )

    def fn():
        return op.forward(
            input,
            weight,
            bias=bias,
            output_buf=output,
            input_scale=input_scale,
            weight_scale=weight_scale,
            output_scale=None,
            fast_accum=False,
        )

    return perf_gemm(iters, "flux", fn)


def rand_tensor(shape: list[int], dtype: torch.dtype):
    if dtype in [torch.int32, torch.int8]:
        return torch.randint(-127, 128, shape, dtype=dtype).cuda()
    elif is_fp8_dtype(dtype):
        data = torch.rand(shape, dtype=torch.bfloat16).cuda() / 10
        return data.to(dtype)
    else:
        return torch.rand(shape, dtype=dtype).cuda() / 10


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("--iters", default=50, type=int, help="perf iterations")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        type=str,
        choices=list(DTYPE_MAP.keys()),
    )
    parser.add_argument(
        "--output_dtype",
        default="",
        type=str,
        help="allowed data type:: bfloat16,float16,s32.",
    )
    parser.add_argument(
        "--has_bias", default=False, action="store_true", help="whether to add bias"
    )
    parser.add_argument(
        "--transpose_weight", default=False, action="store_true", help="whether to transpose weight"
    )

    return parser.parse_args()


THRESHOLD_MAP = {
    torch.float16: 1e-2,
    torch.bfloat16: 2e-2,
    torch.float8_e4m3fn: 2e-2,
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
        output_dtype = torch.bfloat16 if is_fp8 or dtype == torch.int8 else dtype
    else:
        output_dtype = DTYPE_MAP[args.output_dtype]
    is_s8_dequant = dtype == torch.int8 and output_dtype == torch.bfloat16

    if is_s8_dequant:
        if args.transpose_weight:
            raise ValueError("s8 gemm with dequant must in RCR layout")
    input = None
    weight = None
    if is_fp8:
        torch.use_deterministic_algorithms(False, warn_only=True)

    input = rand_tensor((args.M, args.K), dtype=dtype)
    weight = rand_tensor((args.N, args.K), dtype=dtype)

    input_scale = None
    weight_scale = None

    if is_fp8:
        input_scale = rand_tensor(1, dtype=torch.float32).cuda()
        weight_scale = rand_tensor(1, dtype=torch.float32).cuda()
    elif is_s8_dequant:
        input_scale = rand_tensor((args.M, 1), dtype=torch.float32)
        weight_scale = rand_tensor((1, args.N), dtype=torch.float32)

    bias = None
    if args.has_bias:
        bias_dtype = output_dtype
        bias_shape = (1, args.N) if is_fp8 or is_s8_dequant else (args.M, args.N)
        bias = rand_tensor(bias_shape, bias_dtype)

    perf_result_flux = perf_flux(
        input,
        weight,
        bias,
        input_scale,
        weight_scale,
        args.transpose_weight,
        is_fp8,
        is_s8_dequant,
        args.iters,
        output_dtype,
    )
    perf_result_torch = perf_torch(
        input,
        weight,
        bias,
        input_scale,
        weight_scale,
        is_fp8,
        is_s8_dequant,
        args.iters,
        output_dtype,
    )

    flux.testing.print_gemm_sol_time(args.M, args.N, args.K, dtype)
    print(perf_result_torch)
    print(perf_result_flux)

    flux_output = perf_result_flux.output
    torch_output = perf_result_torch.output

    is_bitwise_match = flux.bitwise_check(flux_output, torch_output)
    print("is bitwise match: ", is_bitwise_match)
    atol = THRESHOLD_MAP[flux_output.dtype]
    rtol = THRESHOLD_MAP[flux_output.dtype]
    flux.torch_allclose(flux_output, torch_output, atol=atol, rtol=rtol)
