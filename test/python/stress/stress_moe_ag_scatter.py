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
import logging
import random
from functools import partial
from typing import Any, List, Optional

import torch
import torch.distributed

import flux
from flux.testing import DTYPE_MAP, MoeAgScatterWithTorch, MoeMlp1Ctx
from flux.testing.utils import RING_MODE_MAP

torch.manual_seed(3)
torch.cuda.manual_seed_all(3)
DIST_ENV = flux.get_dist_env()
TP_GROUP = DIST_ENV.get_world()
EP_GROUP = None
torch.cuda.set_device(DIST_ENV.LOCAL_RANK)
print = partial(print, flush=True)


def take_first_or_none(x: Optional[List[Any]]):
    return x[0] if x is not None else None


def init_ep_group(ep_size: int):
    assert DIST_ENV.WORLD_SIZE % ep_size == 0, f"{DIST_ENV.WORLD_SIZE} % {ep_size} != 0"
    global EP_GROUP
    assert EP_GROUP is None, "EP_GROUP already initialized"

    assert TP_GROUP.size() % ep_size == 0, f"{TP_GROUP.size()} % {ep_size} != 0"
    ffn_tp_size = TP_GROUP.size() // ep_size

    temp_groups = []
    for i in range(ffn_tp_size):
        ranks = list(range(i, DIST_ENV.WORLD_SIZE, ffn_tp_size))
        temp_groups.append(ranks)

    ep_groups = []
    for group in temp_groups:
        for i in range(0, len(group), ep_size):
            ep_groups.append(group[i : i + ep_size])

    for ranks in ep_groups:
        group = DIST_ENV.new_group(ranks)
        if DIST_ENV.RANK in ranks:
            EP_GROUP = group


@torch.no_grad()
def forward_torch(ctx: MoeMlp1Ctx, gemm_only_op):
    ctx.clear_outputs()
    MoeAgScatterWithTorch.comm_impl(ctx, TP_GROUP)
    MoeAgScatterWithTorch.scatter_impl(ctx)
    MoeAgScatterWithTorch.gemm_impl(ctx, gemm_only_op)


def parse_args():
    parser = argparse.ArgumentParser()
    # common stress arguments
    parser.add_argument("--iters", default=100, type=int, help="perf iterations")
    parser.add_argument("--verify-iters", default=10, type=int)
    parser.add_argument("--no-verify-iters", default=40, type=int)
    parser.add_argument("--seed", type=int, default=42)
    # AG+Scatter related arguments
    parser.add_argument("--B", type=int, default=4)
    parser.add_argument("--S", type=int, default=4096)
    parser.add_argument("--H", type=int, default=6144)
    parser.add_argument("--ffn_hidden_size", type=int, default=1280)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("-G", type=int, default=128)
    parser.add_argument(
        "--dtype", default="float8_e4m3fn", help="data type", choices=list(DTYPE_MAP.keys())
    )
    parser.add_argument("--ep", default=8, type=int, help="ep size")
    parser.add_argument("--weight_groups", default=2, type=int, help="num of weight groups")
    parser.add_argument(
        "--fast_accum", default=False, action="store_true", help="fp8 use fast accum"
    )
    parser.add_argument(
        "--profile", default=False, action="store_true", help="dump torch.profiler.profile"
    )
    parser.add_argument(
        "--drop_token",
        default=True,
        action="store_true",
        help="if True, splits will have an additional item, tokens in whitch are droped",
    )
    parser.add_argument("--sm_margin", default=0, type=int, help="sm margin")
    parser.add_argument(
        "--dist",
        type=str,
        default="random",
        choices=["uniform", "random", "random_with_first_k_experts"],
    )
    parser.add_argument("--fill_nan", default=False, action="store_true", help="fill nan")
    parser.add_argument(
        "--stable_index",
        default=False,  # use harder cases.
        action="store_true",
        help="use sorted gather_index for each expert",
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


def _make_extra_args(ctx, ag_option):
    extra_args = {
        "ag_option": ag_option,
    }
    if ctx.weight_groups == 1:
        extra_args.update(
            {
                "bias": take_first_or_none(ctx.bias),
                "input_scale": take_first_or_none(ctx.input_scale),
                "weight_scale": take_first_or_none(ctx.weight_scale),
            }
        )
    else:
        extra_args.update(
            {
                "bias": ctx.bias,
                "input_scale": ctx.input_scale,
                "weight_scale": ctx.weight_scale,
            }
        )
    return extra_args


def _run_flux(ctx, gathered_input, ag_option):
    extra_args = _make_extra_args(ctx, ag_option)
    if ctx.weight_groups == 1:
        op.forward(
            inputs_shard=ctx.inputs_shard,
            weights=ctx.weights[0],
            splits_gpu=ctx.splits_gpu,
            scatter_index=ctx.scatter_index,
            output_scale=take_first_or_none(ctx.output_scale),
            outputs_buf=take_first_or_none(ctx.outputs),
            fast_accum=ctx.fast_accum,
            sm_margin=args.sm_margin,
            allgather_output=gathered_input,
            **extra_args,
        )
    else:
        op.forward_multiple_weights(
            inputs_shard=ctx.inputs_shard,
            weights=ctx.weights,
            splits_gpu=ctx.splits_gpu,
            scatter_index=ctx.scatter_index,
            output_scale=ctx.output_scale,
            outputs_buf=ctx.outputs,
            fast_accum=ctx.fast_accum,
            sm_margin=args.sm_margin,
            allgather_output=gathered_input,
            **extra_args,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
    )
    args = parse_args()
    init_ep_group(args.ep)
    print("before flux_shm initialization")
    flux.init_flux_shm(TP_GROUP)
    torch.cuda.synchronize()
    print("after flux_shm initialization")

    input_dtype = DTYPE_MAP[args.dtype]
    is_fp8 = flux.util.is_fp8_dtype(input_dtype)
    is_s8 = torch.int8 == input_dtype
    output_dtype = torch.bfloat16 if is_fp8 or is_s8 else input_dtype
    ag_option = flux.AllGatherOption()
    ag_option.use_cuda_core_local = args.use_cuda_core_local
    ag_option.use_cuda_core_ag = args.use_cuda_core_ag
    ag_option.mode = RING_MODE_MAP[args.ring_mode]

    random.seed(args.seed)  # set all ranks to the same seed

    gemm_only_op = (
        flux.GemmOnly(
            input_dtype,
            input_dtype,
            output_dtype,
            use_fp8_gemm=flux.is_fp8_dtype(input_dtype),
        )
        if is_fp8
        else None
    )

    atol = 1e-2 if output_dtype == torch.float16 else 2e-2
    rtol = 1e-2 if output_dtype == torch.float16 else 2e-2

    tp_env = flux.DistEnvTPWithEP(tp_group=TP_GROUP, nnodes=DIST_ENV.NNODES, ep_group=EP_GROUP)
    moe_args = flux.MoeArguments(
        max_ntokens=args.B * args.S,
        hidden=args.H,
        ffn_hidden=args.ffn_hidden_size,
        nexperts=args.num_experts,
        topk=args.topk,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
    )

    if flux.util.get_arch() >= 90:
        op = flux.GemmGroupedV3AGScatter(tp_env=tp_env, moe_args=moe_args)
    else:
        op = flux.GemmGroupedV2AGScatterOp(tp_env=tp_env, moe_args=moe_args)

    generator = torch.Generator(device="cuda")
    generator.manual_seed(12345)  # use the same random generator to generate the same distribution

    def _make_data(bs):
        return MoeMlp1Ctx(
            DIST_ENV,
            TP_GROUP,
            EP_GROUP,
            b=1,
            s=bs,
            h=args.H,
            ffn_size=args.ffn_hidden_size,
            nexperts=args.num_experts,
            topk=args.topk,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            dist=args.dist,
            fast_accum=args.fast_accum,
            weight_groups=args.weight_groups,
            drop_token=args.drop_token,
            generator=generator,
            stable=args.stable_index,
        )

    for n in range(args.iters):
        # generate data for verify
        moe_ctxes = [
            _make_data(
                random.randint(1, args.B * args.S // DIST_ENV.WORLD_SIZE) * DIST_ENV.WORLD_SIZE
            )
            for _ in range(args.verify_iters)
        ]

        flux_out_list, torch_out_list = [], []
        if args.fill_nan and moe_ctxes:
            moe_ctxes[0].inputs_shard[torch.rand_like(moe_ctxes[0].inputs_shard) < 0.1].fill_(
                torch.nan
            )
        # flux runs
        for idx, moe_ctx in enumerate(moe_ctxes):
            op.clear_buffers()
            _run_flux(moe_ctx, None, ag_option)
            flux_out_list.append(moe_ctx.get_outputs_clone())
        # torch runs
        for moe_ctx in moe_ctxes:
            moe_ctx.clear_outputs()
            forward_torch(moe_ctx, gemm_only_op)
            torch_out_list.append(moe_ctx.get_outputs_clone())

        # verify
        for idx, (torch_outs, flux_outs) in enumerate(zip(torch_out_list, flux_out_list)):
            if args.fill_nan and idx == 0:
                continue
            for flux_out, torch_out in zip(flux_outs, torch_outs):
                try:
                    flux.torch_allclose(flux_out, torch_out, atol=atol, rtol=rtol, verbose=False)
                except Exception as e:
                    torch.save(flux_outs, f"flux_{TP_GROUP.rank()}_{n}_{idx}.pt")
                    torch.save(torch_outs, f"torch_{TP_GROUP.rank()}_{n}_{idx}.pt")
                    logging.error("❌ check failed")
                    torch.save(
                        moe_ctxes[idx],
                        f"moe_ag_scatter_input_{TP_GROUP.rank()}_{n}_{idx}.pt",
                    )
                    raise e

        # just runs, check if hangs
        for j in range(args.no_verify_iters):
            op.clear_buffers()
            _run_flux(moe_ctx, None, ag_option)
        if (n + 1) % 10 == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            logging.info(f"runs {n + 1} iterations done")
