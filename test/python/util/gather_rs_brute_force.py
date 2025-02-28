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

import torch
import torch.distributed

import flux
import flux.testing
from flux.testing import DTYPE_MAP, initialize_distributed


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
    split_cpu: torch.Tensor,
    iters: int,
    token_index: torch.Tensor,
    topk_index: torch.Tensor,
    topk: int,
):
    def fn():
        output_list = []
        for exp_id in range(weight.size(0)):
            exp_w = weight[exp_id]
            exp_input = input[split_cpu[exp_id] : split_cpu[exp_id + 1]]
            output_list.append(torch.matmul(exp_input, exp_w.t()))
        # M N
        full_output = torch.concat(output_list)
        new_index = topk * token_index + topk_index

        output1 = torch.empty_like(full_output)
        output1[new_index] = full_output
        topk_reduce = output1.view((full_output.size(0) // topk, topk, full_output.size(1))).sum(1)
        output2 = torch.zeros(
            (full_output.size(0) // TP_GROUP.size() // topk, full_output.size(1)),
            dtype=topk_reduce.dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )

        torch.distributed.reduce_scatter_tensor(output2, topk_reduce, group=TP_GROUP)
        return output2

    return perf_gemm(iters, "torch", fn)


def perf_flux(
    input: torch.Tensor,
    weight: torch.Tensor,
    split_cpu: torch.Tensor,
    transpose_weight: bool,
    iters: int,
    max_m: int,
    token_index: torch.Tensor,
    topk_index: torch.Tensor,
    topk,
    routing_idx,
):
    m = input.size(0)
    n_dim = weight.size(1)
    print(weight.size())
    op = flux.GemmGroupedV3GatherRS(weight, weight.size(0), max_m, n_dim, RANK, WORLD_SIZE)

    def fn():
        return op.forward_gather_rs(input, split_cpu, topk, routing_idx)

    return perf_gemm(iters, "flux", fn)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=40960)
    parser.add_argument("-N", type=int, default=8192)
    parser.add_argument("-K", type=int, default=8192)
    parser.add_argument("-G", type=int, default=32)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--iters", default=100, type=int, help="perf iterations")
    parser.add_argument("--dtype", default="bfloat16", type=str, help="data type")
    parser.add_argument(
        "--transpose_weight", default=False, action="store_true", help="whether to transpose weight"
    )
    parser.add_argument("--profile", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    TP_GROUP = initialize_distributed()
    RANK, WORLD_SIZE, NNODES = TP_GROUP.rank(), TP_GROUP.size(), flux.testing.NNODES()

    args = parse_args()

    for M in range(40, 40960, 40):
        print("M = ", M)
        args.M = M
        assert args.M % TP_GROUP.size() == 0
        assert args.K % TP_GROUP.size() == 0
        assert args.M % args.topk == 0
        local_K = args.K // TP_GROUP.size()

        dtype = DTYPE_MAP[args.dtype]
        input = torch.rand((args.M, local_K), dtype=dtype).cuda() - 0.5

        weight = torch.rand((args.G, args.N, local_K), dtype=dtype).cuda() - 0.5
        ori_token_num = args.M // args.topk
        random_seq_len, random_gate, token_index, topk_index, routing_idx = flux.testing.gate_func(
            ori_token_num, args.G, args.topk, dist="random"
        )
        if RANK == 0:
            print(random_seq_len)
            print("token_index", token_index)
            print("topk_index", topk_index)
        torch.distributed.barrier()

        split_cpu = torch.zeros(args.G + 1, dtype=torch.int32)
        split_cpu[1:] = torch.cumsum(random_seq_len, 0)
        print(split_cpu.size())
        print(split_cpu)
        ctx = flux.get_torch_prof_ctx(args.profile)
        with ctx:
            perf_result_flux = perf_flux(
                input,
                weight,
                split_cpu,
                args.transpose_weight,
                args.iters,
                args.M,
                token_index,
                topk_index,
                args.topk,
                routing_idx,
            )
            perf_result_torch = perf_torch(
                input, weight, split_cpu, args.iters, token_index, topk_index, args.topk
            )
        if args.profile:
            ctx.export_chrome_trace(f"trace_rank{TP_GROUP.rank()}.json")
        flux.testing.print_grouped_gemm_sol_time_ms(
            args.M, args.N, local_K, args.G, input_dtype=dtype
        )
        print(perf_result_torch)
        print(perf_result_flux)

        flux_output = perf_result_flux.output
        torch_output = perf_result_torch.output
        atol = 1e-2 if dtype == torch.float16 else 2e-2
        rtol = 1e-2 if dtype == torch.float16 else 2e-2
        print(flux_output)
        print(torch_output)
        torch.distributed.barrier()
        flux.torch_allclose(flux_output, torch_output, atol=atol, rtol=rtol)
