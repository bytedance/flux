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
import logging
import torch
import torch.distributed

import flux
from flux.testing import (
    DTYPE_MAP,
    initialize_distributed,
    gate_func,
    moe_gather_rs_forward_torch,
    generate_data,
)


def parse_args():
    parser = argparse.ArgumentParser()
    # common stress arguments
    parser.add_argument("--iters", default=100, type=int, help="perf iterations")
    parser.add_argument("--verify-iters", default=10, type=int)
    parser.add_argument("--no-verify-iters", default=40, type=int)
    parser.add_argument("--seed", type=int, default=42)
    # Gather+RS related arguments
    parser.add_argument("-M", type=int, default=163840)
    parser.add_argument("-N", type=int, default=6144)
    parser.add_argument("-K", type=int, default=5120)
    parser.add_argument("-G", type=int, default=32)
    parser.add_argument("-E", type=int, default=8, help="Expert parallel world size")
    parser.add_argument("-T", type=int, default=1, help="Tensor parallel world size")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--sm_margin", default=0, type=int, help="sm margin")
    parser.add_argument(
        "--dtype", default="bfloat16", help="data type", choices=list(DTYPE_MAP.keys())
    )
    parser.add_argument(
        "--fastacc", default=False, action="store_true", help="whether to enbale fast accumulate"
    )
    parser.add_argument(
        "--ep_in_dp", default=False, action="store_true", help="whether to enbale ep in dp"
    )
    parser.add_argument(
        "--dist",
        type=str,
        default="random",
        choices=["uniform", "random", "random_with_first_k_experts"],
    )
    parser.add_argument("--triton", default=False, action="store_true", help="use triton")
    parser.add_argument("--input_groups", type=int, default=1, help="The number of input groups")
    parser.add_argument(
        "--all_reduce", default=False, action="store_true", help="whether to use all_reduce"
    )
    parser.add_argument("--use_read_mode", default=False, action="store_true")
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

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
    )
    TP_GROUP = initialize_distributed()
    RANK, WORLD_SIZE, NNODES = TP_GROUP.rank(), TP_GROUP.size(), flux.testing.NNODES()

    args = parse_args()
    random.seed(args.seed)  # force use the same seed

    assert args.M % TP_GROUP.size() == 0
    assert args.K % TP_GROUP.size() == 0
    assert args.M % args.topk == 0
    assert TP_GROUP.size() == args.T * args.E
    local_K = args.K // args.T
    MAX_M = args.M
    MAX_TOTAL_TOKEN = MAX_M // args.topk
    assert MAX_TOTAL_TOKEN % TP_GROUP.size() == 0
    MAX_TOKEN_PER_RANK = MAX_TOTAL_TOKEN // TP_GROUP.size()
    input_dtype = DTYPE_MAP[args.dtype]
    is_fp8 = flux.util.is_fp8_dtype(input_dtype)
    is_s8 = torch.int8 == input_dtype
    output_dtype = torch.bfloat16 if is_s8 or is_fp8 else input_dtype
    atol, rtol = ABSOLUTE_THRESHOLD_MAP[input_dtype], RELATIVE_THRESHOLD_MAP[input_dtype]

    if flux.util.get_arch() >= 90:
        op = flux.GemmGroupedV3GatherRS(
            args.G,
            MAX_M,
            args.N,
            args.topk,
            RANK,
            WORLD_SIZE,
            args.T,
            args.E,
            args.input_groups,
            args.ep_in_dp,
        )
    else:
        op = flux.GemmGroupedV2GatherRSOp(
            TP_GROUP,
            args.G,
            MAX_M,
            args.N,
            args.topk,
            output_dtype,
            args.T,
            args.E,
            args.input_groups,
            do_all_reduce=args.all_reduce,
            use_read_mode=args.use_read_mode,
        )

    def _make_data(ntokens):
        tmp_generator = torch.Generator("cuda")
        tmp_generator.manual_seed(12345)
        random_seq_len, _, token_index, topk_index, routing_idx = gate_func(
            ntokens, args.G, args.topk, args.dist, tmp_generator
        )
        splits_cpu = random_seq_len
        n_experts_per_rank = args.G // args.E
        ep_rank = TP_GROUP.rank() // args.T
        tp_rank = TP_GROUP.rank() % args.T
        eid_start = ep_rank * n_experts_per_rank
        eid_end = eid_start + n_experts_per_rank
        ep_rank_m_start = int(torch.sum(splits_cpu[:eid_start]))
        M_cur_ep_rank = int(torch.sum(splits_cpu[eid_start:eid_end]))
        ep_rank_m_end = ep_rank_m_start + M_cur_ep_rank

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
        return (
            inputs,
            weights,
            weight_scales,
            input_scales,
            output_vec_scales,
            splits_cpu,
            token_index,
            topk_index,
            routing_idx,
            ntokens,
            eid_start,
            ep_rank_m_start,
            ep_rank_m_end,
        )

    M_step = TP_GROUP.size() * args.topk
    assert args.M % M_step == 0
    ag_input_len_cpu = None
    for iter in range(args.iters):
        print(f"Iter :", iter)
        total_token_cur_run = random.randint(M_step, args.M // M_step) * M_step // args.topk
        if args.ep_in_dp:
            token_random_start = max(1, MAX_TOKEN_PER_RANK - 128 * RANK)
            token_num_cur_rank = random.randint(token_random_start, MAX_TOKEN_PER_RANK)
            ag_input_len = torch.zeros(TP_GROUP.size(), dtype=torch.int32).cuda()
            input_len = torch.tensor([token_num_cur_rank], dtype=torch.int32).cuda()
            torch.distributed.all_gather_into_tensor(ag_input_len, input_len, group=TP_GROUP)
            ag_input_len_cpu = ag_input_len.cpu()
            total_token_cur_run = torch.sum(ag_input_len_cpu).item()
        torch.cuda.empty_cache()
        print("total_toten", total_token_cur_run)
        input_list = [_make_data(total_token_cur_run) for _ in range(args.input_groups)]
        flux_out_list, torch_out_list = [], []
        # torch runs
        for (
            inputs,
            weights,
            weight_scales,
            input_scales,
            output_vec_scales,
            splits_cpu,
            token_index,
            topk_index,
            _,
            ntokens,
            eid_start,
            ep_rank_m_start,
            ep_rank_m_end,
        ) in input_list:
            if args.ep_in_dp:
                args.all_reduce = True
            torch_out = moe_gather_rs_forward_torch(
                TP_GROUP,
                ntokens * args.topk,
                eid_start,
                ep_rank_m_start,
                ep_rank_m_end,
                inputs,
                weights,
                splits_cpu,
                token_index,
                topk_index,
                args.topk,
                input_scales,
                weight_scales,
                output_vec_scales,
                args.all_reduce,
                args.fastacc,
            )
            if args.ep_in_dp:
                input_len_cum_sum = torch.cumsum(ag_input_len_cpu, dim=0)
                start = 0 if RANK == 0 else input_len_cum_sum[RANK - 1]
                end = input_len_cum_sum[RANK]
                torch_out = torch_out[start:end]
            torch_out_list.append(torch_out)
        is_v2 = flux.util.get_arch() < 90
        extra_args = (
            {"n_tokens_per_rank": ag_input_len_cpu if args.ep_in_dp else None}
            if not is_v2
            else {"bias": None}
        )
        # flux runs
        for (
            inputs,
            weights,
            weight_scales,
            input_scales,
            output_vec_scales,
            splits_cpu,
            _,
            _,
            routing_idx,
            _,
            _,
            _,
            _,
        ) in input_list:
            if args.input_groups == 1:
                flux_out = op.forward_gather_rs(
                    inputs[0],
                    weights[0],
                    splits_cpu,
                    routing_idx,
                    input_scale=input_scales[0],
                    weight_scale=weight_scales[0],
                    output_vec_scale=output_vec_scales[0],
                    fast_accum=args.fastacc,
                    sm_margin=args.sm_margin,
                    **extra_args,
                )
            else:
                flux_out = op.forward_gather_rs_multiple(
                    inputs,
                    weights,
                    splits_cpu,
                    routing_idx,
                    input_scale=input_scales,
                    weight_scale=weight_scales,
                    output_vec_scale=output_vec_scales,
                    fast_accum=args.fastacc,
                    sm_margin=args.sm_margin,
                    **extra_args,
                )
            flux_out_list.append(flux_out)

        for idx, (torch_out, flux_out) in enumerate(zip(torch_out_list, flux_out_list)):
            try:
                flux.torch_allclose(flux_out, torch_out, atol=atol, rtol=rtol, verbose=False)
            except Exception as e:
                torch.save(flux_out, f"flux_out_{TP_GROUP.rank()}_{iter}_{idx}.pt")
                torch.save(torch_out, f"torch_out_{TP_GROUP.rank()}_{iter}_{idx}.pt")
                torch.save(
                    input_list[idx],
                    f"moe_gather_rs_input_{RANK}_{iter}_{idx}.pt",
                )
                logging.error("❌ flux and torch not matches")
                raise e

        # flux runs without verify
        for _ in range(args.no_verify_iters):
            (
                inputs,
                weights,
                weight_scales,
                input_scales,
                output_vec_scales,
                splits_cpu,
                _,
                _,
                routing_idx,
                _,
                _,
                _,
                _,
            ) = random.choice(input_list)
            if args.input_groups == 1:
                flux_out = op.forward_gather_rs(
                    inputs[0],
                    weights[0],
                    splits_cpu,
                    routing_idx,
                    input_scale=input_scales[0],
                    weight_scale=weight_scales[0],
                    output_vec_scale=output_vec_scales[0],
                    fast_accum=args.fastacc,
                    sm_margin=args.sm_margin,
                    **extra_args,
                )
            else:
                flux_out = op.forward_gather_rs_multiple(
                    inputs,
                    weights,
                    splits_cpu,
                    routing_idx,
                    input_scale=input_scales,
                    weight_scale=weight_scales,
                    output_vec_scale=output_vec_scales,
                    fast_accum=args.fastacc,
                    sm_margin=args.sm_margin,
                    **extra_args,
                )
        if (iter + 1) % 10 == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            print(torch_out_list)
            print(flux_out_list)
            print(f"runs {iter + 1} iterations done")
