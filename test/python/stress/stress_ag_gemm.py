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
from typing import Optional

import torch
import torch.distributed

import flux
from flux.testing import (
    DTYPE_MAP,
    RING_MODE_MAP,
    all_gather_into_tensor_with_fp8,
    generate_data,
    initialize_distributed,
    zeros_with_fp8,
    matmul_int8,
)
from flux.util import is_fp8_dtype


@torch.no_grad()
def run_torch(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    input_scale: Optional[torch.Tensor],
    weight_scale: Optional[torch.Tensor],
):
    local_M, K = input.shape
    M = local_M * TP_GROUP.size()
    full_input = torch.zeros(
        (M, K),
        dtype=input.dtype,
        device=torch.cuda.current_device(),
        requires_grad=False,
    )
    all_gather_into_tensor_with_fp8(full_input, input, group=TP_GROUP)
    full_input_scale = (
        torch.zeros((M, 1), dtype=input_scale.dtype, device=torch.cuda.current_device())
        if is_s8_dequant
        else None
    )

    alpha_scale = 1.0
    if is_fp8:
        alpha_scale = input_scale * weight_scale
        input = input.to(torch.bfloat16)
        weight = weight.to(torch.bfloat16)
        full_input = full_input.to(torch.bfloat16)

    if is_s8_dequant:
        assert input_scale is not None
        torch.distributed.all_gather_into_tensor(full_input_scale, input_scale, group=TP_GROUP)

    if is_s8_dequant:
        accum = matmul_int8(full_input, weight.t()).to(torch.float32)
        output = full_input_scale * weight_scale * accum
        output = output.to(torch.bfloat16)
    else:
        output = alpha_scale * torch.matmul(full_input, weight.t())
    if is_fp8:
        output = output.to(torch.bfloat16)
    if bias is not None:
        output += bias

    return output, full_input


def parse_args():
    parser = argparse.ArgumentParser()
    # common stress arguments
    parser.add_argument("--iters", default=100, type=int, help="perf iterations")
    parser.add_argument("--verify-iters", default=10, type=int)
    parser.add_argument("--no-verify-iters", default=40, type=int)
    parser.add_argument("--seed", type=int, default=42)
    # AG related arguments
    parser.add_argument("-M", type=int, default=10240)
    parser.add_argument("-N", type=int, default=16128)
    parser.add_argument("-K", type=int, default=4608)
    parser.add_argument(
        "--dtype", default="bfloat16", type=str, choices=list(DTYPE_MAP.keys()), help="data type"
    )
    parser.add_argument(
        "--gather_input", default=False, action="store_true", help="output gather results"
    )
    parser.add_argument(
        "--transpose_weight",
        action="store_true",
        default=False,
    )
    parser.add_argument("--has_bias", default=False, action="store_true", help="whether have bias")
    parser.add_argument(
        "--fastacc",
        default=False,
        action="store_true",
        help="whether to use fast accumulation (FP8 Gemm only)",
    )
    parser.add_argument(
        "--ring_mode",
        default="auto",
        choices=["auto", "all2all", "ring1d", "ring2d"],
        help="ring mode. auto for auto detect",
    )
    parser.add_argument(
        "--use_cuda_core_local",
        action=argparse.BooleanOptionalAction,
        help="use cuda core to impl local copy, auto select if not specified",
    )
    parser.add_argument(
        "--use_cuda_core_ag",
        action=argparse.BooleanOptionalAction,
        help="use cuda core to impl all gather, auto select if not specified",
    )
    return parser.parse_args()


THRESHOLD_MAP = {
    torch.float16: 1e-2,
    torch.bfloat16: 1e-3,
    torch.float8_e4m3fn: 1e-2,
    torch.float8_e5m2: 1e-2,
    torch.int8: 0,
}

if __name__ == "__main__":
    TP_GROUP = initialize_distributed()
    RANK, WORLD_SIZE, NNODES = TP_GROUP.rank(), TP_GROUP.size(), flux.testing.NNODES()

    args = parse_args()
    assert args.M % TP_GROUP.size() == 0
    assert args.N % TP_GROUP.size() == 0
    assert args.K % TP_GROUP.size() == 0
    local_N = args.N // TP_GROUP.size()
    input_dtype = DTYPE_MAP[args.dtype]
    is_fp8 = is_fp8_dtype(input_dtype)
    is_s8_dequant = input_dtype == torch.int8
    output_dtype = torch.bfloat16 if is_fp8 or is_s8_dequant else input_dtype
    assert not (
        args.transpose_weight and (is_fp8 or is_s8_dequant)
    ), "FP8 GEMM does not support RRR layout"

    MAX_M = args.M
    flux_op = flux.AGKernel(
        TP_GROUP,
        NNODES,
        MAX_M,
        local_N,
        args.K,
        input_dtype,
        output_dtype,
    )

    random.seed(args.seed)  # always use the same random seed to ensure M the same

    ag_option = flux.AllGatherOption()
    ag_option.mode = RING_MODE_MAP[args.ring_mode]
    ag_option.use_cuda_core_local = args.use_cuda_core_local
    ag_option.use_cuda_core_ag = args.use_cuda_core_ag

    def _make_data(local_M):
        #### data config start ####
        scale = TP_GROUP.rank() + 1
        if is_s8_dequant:
            data_config = [
                ((local_M, args.K), input_dtype, (127, 0)),  # A
                ((local_N, args.K), input_dtype, (127, 0)),  # B
                None if not args.has_bias else ((1, local_N), torch.bfloat16, (scale, 0)),  # bias
                ((local_M, 1), torch.float32, (1, 0)),  # input_scale
                ((1, local_N), torch.float32, (1, 0)),  # weight_scale
            ]
        elif is_fp8:
            data_config = [
                ((local_M, args.K), input_dtype, (0.01 * scale, 0)),  # A
                ((local_N, args.K), input_dtype, (0.01 * scale, 0)),  # B
                (
                    None if not args.has_bias else ((1, local_N), torch.bfloat16, (0.1 * scale, 0))
                ),  # bias
                ((1, 1), torch.float32, (1, 0)),  # input_scale
                ((1, 1), torch.float32, (1, 0)),  # weight_scale
            ]
        else:
            data_config = [
                ((local_M, args.K), input_dtype, (0.01 * scale, 0)),  # A
                ((local_N, args.K), input_dtype, (0.01 * scale, 0)),  # B
                (  # bias
                    None
                    if not args.has_bias
                    else ((local_M * WORLD_SIZE, local_N), input_dtype, (0.1 * scale, 0))
                ),
                None,  # input_scale
                None,
            ]  # weight_scale
        #### data config done ####

        generator = generate_data(data_config)
        return list(next(generator))

    is_bitwise = True
    atol = THRESHOLD_MAP[input_dtype]
    rtol = THRESHOLD_MAP[input_dtype]
    for n in range(args.iters):
        # generate data for verify
        input_list = [
            _make_data(random.randint(1, MAX_M // WORLD_SIZE)) for _ in range(args.verify_iters)
        ]
        flux_out_list, torch_out_list = [], []
        # torch runs
        for input, weight, bias, input_scale, weight_scale in input_list:
            torch_out, torch_gathered_input = run_torch(
                input, weight, bias, input_scale, weight_scale
            )
            torch_out_list.append((torch_out, torch_gathered_input))
        # flux runs
        for input, weight, bias, input_scale, weight_scale in input_list:
            M, _ = input.shape
            flux_gathered_input = (
                zeros_with_fp8(
                    (M * TP_GROUP.size(), args.K),
                    dtype=input_dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=False,
                )
                if args.gather_input
                else None
            )
            w = weight if not args.transpose_weight else weight.t()
            flux_out = flux_op.forward(
                input,
                w,
                bias=bias,
                input_scale=input_scale,
                weight_scale=weight_scale,
                output_scale=None,
                fast_accum=args.fastacc,
                transpose_weight=args.transpose_weight,
                all_gather_option=ag_option,
                gathered_input=flux_gathered_input,
            )
            flux_out_list.append((flux_out, flux_gathered_input))
        # verify
        for idx, ((torch_out, torch_gathered_input), (flux_out, flux_gathered_input)) in enumerate(
            zip(torch_out_list, flux_out_list)
        ):
            try:
                flux.torch_allclose(flux_out, torch_out, atol=atol, rtol=rtol, verbose=False)
                if args.gather_input:
                    flux.torch_allclose(
                        flux_gathered_input,
                        torch_gathered_input,
                        atol=1e-10,
                        rtol=1e-10,
                        verbose=False,
                    )
            except Exception as e:
                print(f"iter {n}/{idx} verify failed, {e}")
                torch.save(torch_out, f"torch_out_{RANK}_{n}_{idx}.pt")
                torch.save(flux_out, f"flux_out_{RANK}_{n}_{idx}.pt")
                torch.save(
                    input_list[idx],
                    f"ag_gemm_input_{RANK}_{n}_{idx}.pt",
                )
                raise e
            if not flux.bitwise_check(torch_out, flux_out):
                is_bitwise = False

        # just runs, check if hangs
        for _ in range(args.no_verify_iters):
            local_M = random.randint(1, MAX_M // WORLD_SIZE)
            input, weight, bias, input_scale, weight_scale = _make_data(local_M)
            M, _ = input.shape
            flux_gathered_input = (
                zeros_with_fp8(
                    (M * TP_GROUP.size(), args.K),
                    dtype=input_dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=False,
                )
                if args.gather_input
                else None
            )
            w = weight if not args.transpose_weight else weight.t()
            flux_out = flux_op.forward(
                input,
                w,
                bias=bias,
                input_scale=input_scale,
                weight_scale=weight_scale,
                output_scale=None,
                fast_accum=args.fastacc,
                transpose_weight=args.transpose_weight,
                all_gather_option=ag_option,
                gathered_input=flux_gathered_input,
            )

        if (n + 1) % 10 == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            print(f"runs {n} iterations done")
    if is_bitwise:
        print(f"rank[{TP_GROUP.rank()}]: ✅  torch vs flux bitwise match")
    else:
        print(f"rank[{TP_GROUP.rank()}]: ❌  torch vs flux not bitwise match")
