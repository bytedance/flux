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

# usage: torchrun --node_rank=0 --nproc_per_node=8 --nnodes=1 --rdzv_id=none --master_addr=127.0.0.1 --master_port=23456 test/python/stress/stress_gemm_rs.py 2048 10240 40960
import argparse
import random
from typing import Optional

import torch
import torch.distributed

import flux
from flux.testing import DTYPE_MAP, initialize_distributed, generate_data, matmul_int8

THRESHOLD_MAP = {
    torch.float16: 1e-2,
    torch.bfloat16: 2e-2,
    torch.float8_e4m3fn: 3e-2,
    torch.float8_e5m2: 3e-2,
    torch.int8: 2e-1,
}


@torch.no_grad()
def forward_torch(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    input_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
    transpose_weight: bool = False,
):
    input_dtype = input.dtype
    is_fp8 = flux.util.is_fp8_dtype(input_dtype)
    is_s8 = torch.int8 == input_dtype
    output_dtype = torch.bfloat16 if is_fp8 or is_s8 else input_dtype
    m = input.size(0)
    with flux.util.with_torch_deterministic(False):
        if transpose_weight:
            w = weight.t().contiguous()
            n = w.size(1)
        else:
            n = weight.size(0)
            w = weight

        full_output = torch.zeros(
            [m, n],
            dtype=output_dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )
        output = torch.zeros(
            [m // WORLD_SIZE, n],
            dtype=output_dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )

    op = (
        flux.GemmOnly(
            input_dtype=input.dtype,
            output_dtype=output_dtype,
            transpose_weight=transpose_weight,
            use_fp8_gemm=is_fp8,
        )
        if is_fp8
        else None
    )

    if is_fp8:
        full_output = op.forward(
            input,
            w,
            bias=bias,
            output_buf=full_output,
            input_scale=input_scale,
            weight_scale=weight_scale,
            output_scale=None,
            fast_accum=False,
        )
    elif is_s8:
        accum = matmul_int8(input, weight.t()).to(torch.float32)
        full_output = input_scale * weight_scale * accum
        full_output = full_output.to(output_dtype)
    else:
        full_output = torch.matmul(input, weight.t())
    if bias is not None and not is_fp8:
        # only apply bias on rank 0 for s8 gemm
        if not is_s8 or (is_s8 and TP_GROUP.rank() == 0):
            full_output += bias
    torch.distributed.reduce_scatter_tensor(output, full_output, group=TP_GROUP)
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    # common stress arguments
    parser.add_argument("--iters", default=100, type=int, help="perf iterations")
    parser.add_argument("--verify-iters", default=10, type=int)
    parser.add_argument("--no-verify-iters", default=40, type=int)
    parser.add_argument("--seed", type=int, default=42)
    # RS related arguments
    parser.add_argument("-M", type=int, default=10240)
    parser.add_argument("-N", type=int, default=4608)
    parser.add_argument("-K", type=int, default=16128)
    parser.add_argument(
        "--dtype", default="bfloat16", type=str, choices=list(DTYPE_MAP.keys()), help="data type"
    )
    parser.add_argument(
        "--transpose_weight", default=False, action="store_true", help="whether to transpose weight"
    )
    parser.add_argument(
        "--fuse_reduction", default=False, action="store_true", help="fuse reduction to gemm"
    )
    parser.add_argument(
        "--ring_reduction",
        default=False,
        action="store_true",
        help="reduce paritial output with ring order",
    )
    parser.add_argument("--has_bias", default=False, action="store_true", help="whether have bias")
    parser.add_argument(
        "--verify-per-n-iters",
        type=int,
        default=1,
        help="check allclose per N iterations. 0 for no check result",
    )
    return parser.parse_args()


if __name__ == "__main__":
    TP_GROUP = initialize_distributed()
    RANK, WORLD_SIZE, NNODES = TP_GROUP.rank(), TP_GROUP.size(), flux.testing.NNODES()

    args = parse_args()

    input_dtype = DTYPE_MAP[args.dtype]
    is_fp8 = flux.util.is_fp8_dtype(input_dtype)
    is_s8 = torch.int8 == input_dtype
    output_dtype = torch.bfloat16 if is_fp8 or is_s8 else input_dtype
    if args.transpose_weight and (is_fp8 or is_s8):
        raise ValueError("FP8 GEMM does not support RRR layout")

    assert args.M % TP_GROUP.size() == 0
    assert args.K % TP_GROUP.size() == 0
    local_K = args.K // TP_GROUP.size()

    gemm_rs_op = flux.GemmRS(
        TP_GROUP,
        NNODES,
        args.M,
        args.N,
        input_dtype,
        output_dtype,
        transpose_weight=args.transpose_weight,
        fuse_reduction=args.fuse_reduction,
        ring_reduction=args.ring_reduction,
    )

    atol, rtol = THRESHOLD_MAP[input_dtype], THRESHOLD_MAP[input_dtype]
    random.seed(args.seed)

    def _make_data(M):
        scale = TP_GROUP.rank() + 1
        if is_s8:
            data_config = [
                ((M, local_K), input_dtype, (127, 0)),  # A
                ((args.N, local_K), input_dtype, (127, 0)),  # B
                None if not args.has_bias else ((1, args.N), torch.bfloat16, (24, -12)),  # bias
                ((M, 1), torch.float32, (1 / 1024.0, 0)),  # input_scale
                ((1, args.N), torch.float32, (1 / (args.K * 4 / 1024.0), 0)),  # weight_scale
            ]
        elif is_fp8:
            data_config = [
                ((M, local_K), input_dtype, (0.01 * scale, 0)),  # A
                ((args.N, local_K), input_dtype, (0.01 * scale, 0)),  # B
                None,  # bias. not supported now. ((1, args.N), torch.bfloat16, (0.1 * scale, 0))
                ((1), torch.float32, (1, 0)),  # input_scale
                ((1), torch.float32, (1, 0)),  # weight_scale
            ]
        else:
            data_config = [
                ((M, local_K), input_dtype, (0.01 * scale, 0)),  # A
                ((args.N, local_K), input_dtype, (0.01 * scale, 0)),  # B
                (  # bias
                    None if not args.has_bias else ((M, args.N), input_dtype, (0.1 * scale, 0))
                ),
                None,  # input_scale
                None,  # weight_scale
            ]

        assert not (args.has_bias and is_fp8), "FP8 does not support bias"
        generator = generate_data(data_config)
        input, weight, bias, input_scale, weight_scale = next(generator)
        return input, weight, bias, input_scale, weight_scale

    is_bitwise = True
    MAX_M = args.M
    for n in range(args.iters):
        torch.cuda.empty_cache()
        # generate data for verify
        input_list = [
            _make_data(random.randint(1, MAX_M // WORLD_SIZE) * WORLD_SIZE)
            for _ in range(args.verify_iters)
        ]
        flux_out_list, torch_out_list = [], []
        # torch runs
        for input, weight, bias, input_scale, weight_scale in input_list:
            torch_out = forward_torch(
                input,
                weight,
                bias,
                input_scale,
                weight_scale,
                args.transpose_weight,
            )
            torch_out_list.append(torch_out)
        # flux runs
        for input, weight, bias, input_scale, weight_scale in input_list:
            with flux.util.with_torch_deterministic(False):
                w = weight if not args.transpose_weight else weight.t().contiguous()

            flux_out = gemm_rs_op.forward(
                input,
                w,
                bias=bias,
                input_scale=input_scale,
                weight_scale=weight_scale,
                output_scale=None,
                fast_accum=False,
            )
            flux_out_list.append(flux_out)
        # verify
        for idx, (torch_out, flux_out) in enumerate(zip(torch_out_list, flux_out_list)):
            try:
                flux.torch_allclose(flux_out, torch_out, atol=atol, rtol=rtol, verbose=False)
            except Exception as e:
                print(f"iter {n}/{idx} verify failed, {e}")
                torch.save(torch_out, f"torch_out_{RANK}_{n}_{idx}.pt")
                torch.save(flux_out, f"flux_out_{RANK}_{n}_{idx}.pt")
                torch.save(
                    input_list[idx],
                    f"gemm_rs_input_{RANK}_{n}_{idx}.pt",
                )
                raise e
            if not flux.bitwise_check(torch_out, flux_out):
                is_bitwise = False

        # just runs, check if hangs
        for _ in range(args.no_verify_iters):
            M = random.randint(1, MAX_M // WORLD_SIZE) * WORLD_SIZE
            input, weight, bias, input_scale, weight_scale = _make_data(M)
            with flux.util.with_torch_deterministic(False):
                w = weight if not args.transpose_weight else weight.t().contiguous()

            flux_out = gemm_rs_op.forward(
                input,
                w,
                bias=bias,
                input_scale=input_scale,
                weight_scale=weight_scale,
                output_scale=None,
                fast_accum=False,
            )

        if (n + 1) % 10 == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            print(f"runs {n} iterations done")

    if is_bitwise:
        print(f"rank[{TP_GROUP.rank()}]: ✅  torch vs flux bitwise match")
    else:
        print(f"rank[{TP_GROUP.rank()}]: ❌  torch vs flux not bitwise match")
