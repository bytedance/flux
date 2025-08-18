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
from typing import List, Tuple, Union

import torch
import torch.distributed

import flux
import flux.testing
from flux.testing import (
    DTYPE_MAP,
    initialize_distributed,
    gate_func,
    moe_gather_rs_forward_torch,
    generate_data,
)
from flux.testing.perf_db_helper import log_perf, set_global_args, should_log_to_rds
from flux.util import get_arch


class PerfResult:
    def __init__(self, name: str, output: torch.Tensor, gemm_time_ms: float) -> None:
        self.name = name
        self.output = output
        self.gemm_time_ms = gemm_time_ms
        self.total_ms = self.gemm_time_ms

    def __repr__(self) -> str:
        return f"{self.name}: gemm {self.gemm_time_ms:.3f} ms"


def perf_gemm(iters: int, warmup_iters: int, name: str, fn: callable):
    for _ in range(warmup_iters):
        output = fn()
    torch.cuda.synchronize()
    total_time = 0
    start = time.time()
    for _ in range(iters):
        output = fn()
    torch.cuda.synchronize()
    end = time.time()
    total_time = end - start
    return PerfResult(name=name, output=output, gemm_time_ms=total_time / iters * 1000)


def perf_torch(
    inputs: Union[torch.Tensor, List[torch.Tensor]],
    weights: Union[torch.Tensor, List[torch.Tensor]],
    split_cpu: torch.Tensor,
    iters: int,
    warmup_iters: int,
    token_index: torch.Tensor,
    topk_index: torch.Tensor,
    topk: int,
    input_scales: Union[torch.Tensor, List[torch.Tensor], None],
    weight_scales: Union[torch.Tensor, List[torch.Tensor], None],
    output_vec_scales: Union[torch.Tensor, List[torch.Tensor], None],
    do_all_reduce: bool = False,
):

    return perf_gemm(
        iters,
        warmup_iters,
        f"torch #{TP_GROUP.rank()}",
        lambda: moe_gather_rs_forward_torch(
            TP_GROUP,
            args.M,
            eid_start,
            ep_rank_m_start,
            ep_rank_m_end,
            inputs,
            weights,
            split_cpu,
            token_index,
            topk_index,
            topk,
            input_scales,
            weight_scales,
            output_vec_scales,
            do_all_reduce,
            fast_acc=args.fastacc,
        ),
    )


def perf_flux(
    input: Union[torch.Tensor, List[torch.Tensor]],
    weight: Union[torch.Tensor, List[torch.Tensor]],
    split_cpu: torch.Tensor,
    iters: int,
    warmup_iters: int,
    max_m: int,
    topk: int,
    routing_idx: torch.Tensor,
    input_scale: Union[torch.Tensor, List[torch.Tensor], None],
    weight_scale: Union[torch.Tensor, List[torch.Tensor], None],
    output_vec_scale: Union[torch.Tensor, List[torch.Tensor], None],
    do_all_reduce: bool = False,
    use_read_mode: bool = False,
):
    n_dim = args.N
    if isinstance(weight, torch.Tensor):
        assert weight.size(1) == n_dim
    elif isinstance(weight, list):
        assert weight[0].size(1) == n_dim

    input_dtype = input.dtype if isinstance(input, torch.Tensor) else input[0].dtype
    is_fp8 = flux.is_fp8_dtype(input_dtype)
    is_s8 = input_dtype == torch.int8
    output_dtype = torch.bfloat16 if is_fp8 or is_s8 else input_dtype
    if flux.util.get_arch() >= 90:
        op = flux.GemmGroupedV3GatherRS(
            args.G, max_m, n_dim, topk, RANK, WORLD_SIZE, args.T, args.E, args.input_groups
        )
    else:
        op = flux.GemmGroupedV2GatherRSOp(
            TP_GROUP,
            args.G,
            max_m,
            n_dim,
            topk,
            output_dtype,
            args.T,
            args.E,
            args.input_groups,
            do_all_reduce=do_all_reduce,
            use_read_mode=use_read_mode,
        )

    def fn():
        is_v2 = get_arch() < 90
        extra_args = {} if not is_v2 else {"bias": None}
        if isinstance(input, torch.Tensor):
            return op.forward_gather_rs(
                input,
                weight,
                split_cpu,
                routing_idx,
                input_scale=input_scale,
                weight_scale=weight_scale,
                output_vec_scale=output_vec_scale,
                fast_accum=args.fastacc,
                sm_margin=args.sm_margin,
                **extra_args,
            )
        else:
            return op.forward_gather_rs_multiple(
                input,
                weight,
                split_cpu,
                routing_idx,
                input_scale=input_scale,
                weight_scale=weight_scale,
                output_vec_scale=output_vec_scale,
                fast_accum=args.fastacc,
                sm_margin=args.sm_margin,
                **extra_args,
            )

    return perf_gemm(iters, warmup_iters, f"flux #{TP_GROUP.rank()}", fn)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=32768)
    parser.add_argument("-N", type=int, default=8192)
    parser.add_argument("-K", type=int, default=8192)
    parser.add_argument("-G", type=int, default=32)
    parser.add_argument("-E", type=int, default=1, help="Expert parallel world size")
    parser.add_argument("-T", type=int, default=8, help="Tensor parallel world size")
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--iters", default=10, type=int, help="perf iterations")
    parser.add_argument("--warmup_iters", default=5, type=int, help="warmup iterations")
    parser.add_argument("--sm_margin", default=0, type=int, help="sm margin")
    parser.add_argument(
        "--dtype", default="bfloat16", help="data type", choices=list(DTYPE_MAP.keys())
    )
    parser.add_argument(
        "--fastacc", default=False, action="store_true", help="whether to enbale fast accumulate"
    )
    parser.add_argument(
        "--dist",
        type=str,
        default="random",
        choices=["uniform", "random", "random_with_first_k_experts"],
    )
    parser.add_argument("--profile", action="store_true", default=False)
    parser.add_argument("--input_groups", type=int, default=1, help="The number of input groups")
    parser.add_argument(
        "--all_reduce", default=False, action="store_true", help="whether to use all_reduce"
    )
    parser.add_argument("--use_read_mode", default=False, action="store_true")
    parser.add_argument(
        "--debug", default=False, action="store_true", help="whether to use debug mode"
    )
    return parser.parse_args()


ABSOLUTE_THRESHOLD_MAP = {
    torch.float16: 1e-2,
    torch.bfloat16: 2e-2,
    torch.float8_e4m3fn: 3e-2,
    torch.float8_e5m2: 3e-2,
    torch.int8: 2e-1,
}

RELATIVE_THRESHOLD_MAP = {
    torch.float16: 1e-2,
    torch.bfloat16: 2e-2,
    torch.float8_e4m3fn: 3e-2,
    torch.float8_e5m2: 3e-2,
    torch.int8: 3e-2,
}


def _print_tensor_desc(name, tensor: Tuple[torch.Tensor, List[torch.Tensor]]):
    if isinstance(tensor, torch.Tensor):
        print(f"{name}: {tensor.shape} of {tensor.dtype} at {tensor.device}")
    elif isinstance(tensor, List):
        print(
            f"{name}: {tensor[0].shape} of {tensor[0].dtype} at {tensor[0].device} of group {(len(tensor))}"
        )


if __name__ == "__main__":
    TP_GROUP = initialize_distributed()
    RANK, WORLD_SIZE, NNODES = TP_GROUP.rank(), TP_GROUP.size(), flux.testing.NNODES()

    args = parse_args()

    assert args.M % TP_GROUP.size() == 0
    assert args.K % TP_GROUP.size() == 0
    assert args.M % args.topk == 0
    assert TP_GROUP.size() == args.T * args.E
    local_K = args.K // args.T

    ori_token_num = args.M // args.topk
    generator = torch.Generator("cuda")
    generator.manual_seed(12345)
    random_seq_len, random_gate, token_index, topk_index, routing_idx = gate_func(
        ori_token_num, args.G, args.topk, args.dist, generator
    )

    split_cpu = random_seq_len
    n_experts_per_rank = args.G // args.E
    ep_rank = TP_GROUP.rank() // args.T
    tp_rank = TP_GROUP.rank() % args.T
    eid_start = ep_rank * n_experts_per_rank
    eid_end = eid_start + n_experts_per_rank
    partial_split_cpu = split_cpu[eid_start:eid_end]
    ep_rank_m_start = 0
    for i in range(eid_start):
        ep_rank_m_start += split_cpu[i]
    M_cur_ep_rank = torch.sum(random_seq_len[eid_start:eid_end]).item()
    ep_rank_m_end = ep_rank_m_start + M_cur_ep_rank

    input_dtype = DTYPE_MAP[args.dtype]
    is_fp8 = flux.util.is_fp8_dtype(input_dtype)
    is_s8 = torch.int8 == input_dtype
    output_dtype = torch.bfloat16 if is_fp8 or is_s8 else input_dtype

    weight_scale = 0.5 / 255.0 if is_s8 else 1
    if is_s8:
        data_config = [
            ((M_cur_ep_rank, local_K), input_dtype, (127, 0)),  # input
            ((n_experts_per_rank, args.N, local_K), input_dtype, (127, 0)),  # weight
            ((n_experts_per_rank, args.N), torch.float32, (weight_scale, 0)),  # weight_scale
        ]
    else:
        data_config = [
            ((M_cur_ep_rank, local_K), input_dtype, (0.1, 0)),  # input
            ((n_experts_per_rank, args.N, local_K), input_dtype, (0.1, 0)),  # weight
            ((n_experts_per_rank,), torch.float32, (weight_scale, 0)),  # weight_scale
        ]
    input_scale = 1 / 255.0 if is_s8 else 1
    input_scale_shape = M_cur_ep_rank if is_s8 else 1
    data_config += [
        ((input_scale_shape,), torch.float32, (input_scale, 0)),  # input_scale
        ((M_cur_ep_rank,), torch.float32, (1, 0)),  # output_scale
    ]

    generator = generate_data(data_config)
    inputs, weights, weight_scales, input_scales, output_vec_scales = [
        *zip(*[list(next(generator)) for _ in range(args.input_groups)])
    ]
    if args.E == 1 and flux.util.get_arch() >= 90:
        [x.fill_(1) for x in output_vec_scales]

    if args.debug:
        for _input, _weight in zip(inputs, weights):
            _input.fill_(0)
            _weight.fill_(1)
            _input[:, 0] = torch.arange(1, M_cur_ep_rank + 1, dtype=torch.float32) % (
                M_cur_ep_rank // TP_GROUP.size()
            )

        for input_scale in input_scales:
            if input_scale is not None:
                input_scale.fill_(1)
        for weight_scale in weight_scales:
            if weight_scale is not None:
                weight_scale.fill_(1)
        for output_vec_scale in output_vec_scales:
            if output_vec_scale is not None:
                output_vec_scale.fill_(2)

    if args.input_groups == 1:
        inputs = inputs[0]
        weights = weights[0]
        input_scales = input_scales[0]
        weight_scales = weight_scales[0]
        output_vec_scales = output_vec_scales[0]
        _print_tensor_desc("input", inputs)
        _print_tensor_desc("weight", weights)
    print("split_cpu:", random_seq_len)
    _print_tensor_desc("token_index", token_index)
    _print_tensor_desc("topk_index", topk_index)
    _print_tensor_desc("weight_scale", weight_scales)
    _print_tensor_desc("input_scale", input_scales)
    _print_tensor_desc("routing_idx: ", routing_idx)
    torch.distributed.barrier()

    with flux.group_profile(
        name="moe_gather_rs_" + os.environ["TORCHELASTIC_RUN_ID"],
        do_prof=args.profile,
        group=TP_GROUP,
    ):
        perf_result_flux = perf_flux(
            inputs,
            weights,
            split_cpu,
            args.iters,
            args.warmup_iters,
            args.M,
            args.topk,
            routing_idx,
            input_scales,
            weight_scales,
            output_vec_scales,
            args.all_reduce,
            args.use_read_mode,
        )

        perf_result_torch = perf_torch(
            inputs,
            weights,
            split_cpu,
            args.iters,
            args.warmup_iters,
            token_index,
            topk_index,
            args.topk,
            input_scales,
            weight_scales,
            output_vec_scales,
            args.all_reduce,
        )

    if TP_GROUP.rank() == 0:
        flux.testing.print_grouped_gemm_sol_time_ms(
            args.M,
            args.N,
            local_K,
            n_experts_per_rank,
            input_dtype,
            num_groups=args.input_groups,
        )
    TP_GROUP.barrier()
    if should_log_to_rds():
        set_global_args("moe_gather_rs", args)
    flux.exec_in_rank_order(TP_GROUP, lambda: log_perf(perf_result_torch))
    flux.exec_in_rank_order(TP_GROUP, lambda: log_perf(perf_result_flux))
    atol, rtol = ABSOLUTE_THRESHOLD_MAP[input_dtype], RELATIVE_THRESHOLD_MAP[input_dtype]
    TP_GROUP.barrier()

    def check_result():
        print(f"#{TP_GROUP.rank()} Threshold = Atol:{atol}  Rtol:{rtol}")
        print(f"flux  output shape: {flux_output.size()}")
        print(f"torch output shape: {torch_output.size()}")

        try:
            flux.torch_allclose(flux_output, torch_output, atol=atol, rtol=rtol)
        except Exception as e:
            torch.save(flux_output, f"flux_output_{TP_GROUP.rank()}.pt")
            torch.save(torch_output, f"torch_output_{TP_GROUP.rank()}.pt")
            print("❌ flux and torch not matches")
            raise e
        else:
            print("✅ flux and torch matches")

    torch_output = perf_result_torch.output
    flux_output = perf_result_flux.output
    flux.exec_in_rank_order(TP_GROUP, check_result)
